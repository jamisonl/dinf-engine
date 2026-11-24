//! DFlash 2 block drafter for Qwen 3.8 targets.
//!
//! The draft owns its five small transformer blocks and a rolling injected
//! KV cache. It shares the target token embedding and output head, as required
//! by the published GGUF format.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::qmatmul::QMatMul;
// Selector tables only gather embedding rows; they never dispatch a projection.
use anyhow::{Context, Result};
use candle_core::quantized::QMatMul as CandleEmbedding;
use candle_core::quantized::{GgmlDType, QTensor, gguf_file};
use candle_core::{D, DType, Device, IndexOp, Module, Tensor};

use super::SpeculativeSharedWeights;
use crate::gguf_limits::{validate_gguf_header_budget, validate_layer_count};

#[path = "dflash2_cpu.rs"]
mod cpu;

const MAX_DFLASH_HIDDEN_SIZE: usize = 16_384;
const MAX_DFLASH_HEAD_DIM: usize = 1_024;
const MAX_DFLASH_CONTEXT_LENGTH: usize = 1_048_576;

#[derive(Clone)]
struct DraftKvCache {
    k: Tensor,
    v: Tensor,
}

/// Per-request DFlash state. The target and draft caches advance together.
#[derive(Clone)]
pub(crate) struct DFlash2Cache {
    layers: Vec<Option<DraftKvCache>>,
    next_position: usize,
}

impl DFlash2Cache {
    pub(crate) fn fork(&self) -> Result<Self> {
        Ok(Self {
            layers: self
                .layers
                .iter()
                .map(|layer| {
                    layer
                        .as_ref()
                        .map(|kv| {
                            Ok(DraftKvCache {
                                k: super::copy_persistent_tensor(&kv.k)?,
                                v: super::copy_persistent_tensor(&kv.v)?,
                            })
                        })
                        .transpose()
                })
                .collect::<Result<_>>()?,
            next_position: self.next_position,
        })
    }

    pub(crate) fn live_bytes(&self) -> usize {
        self.layers.iter().flatten().fold(0usize, |bytes, kv| {
            bytes
                .saturating_add(
                    kv.k.elem_count()
                        .saturating_mul(kv.k.dtype().size_in_bytes()),
                )
                .saturating_add(
                    kv.v.elem_count()
                        .saturating_mul(kv.v.dtype().size_in_bytes()),
                )
        })
    }

    pub(crate) fn clear(&mut self) {
        self.layers.iter_mut().for_each(|cache| *cache = None);
        self.next_position = 0;
    }

    pub(crate) fn next_position(&self) -> usize {
        self.next_position
    }
}

pub(super) enum DraftQMatMul {
    Candle(QMatMul),
    #[cfg(target_os = "macos")]
    MetalTensor(super::tensor_qmatmul_metal::MetalTensorQMatMul),
}

impl DraftQMatMul {
    pub(super) fn from_qtensor(qt: Arc<QTensor>, _device: &Device) -> Result<Self> {
        #[cfg(target_os = "macos")]
        if let Some(matmul) = super::tensor_qmatmul_metal::MetalTensorQMatMul::try_from_qtensor(
            Arc::clone(&qt),
            _device,
        )? {
            return Ok(Self::MetalTensor(matmul));
        }
        Ok(Self::Candle(QMatMul::from_arc(qt)?))
    }

    /// [`Self::forward`] for `[rows, in]` inputs of a few rows, read once in
    /// decode's per-row reduction on Metal; other inputs as `forward`.
    pub(super) fn forward_rows(&self, input: &Tensor) -> Result<Tensor> {
        #[cfg(target_os = "macos")]
        if let Self::MetalTensor(matmul) = self
            && input.rank() == 2
            && (2..=8).contains(&input.dim(0)?)
        {
            return matmul.forward_decode_rows(input).map_err(Into::into);
        }
        self.forward(input)
    }

    pub(super) fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Candle(matmul) => {
                if let Some(output) = super::qwen3moe::dflash_cpu_projection(matmul, input)? {
                    return Ok(output);
                }
                matmul.forward(input).map_err(Into::into)
            }
            #[cfg(target_os = "macos")]
            Self::MetalTensor(matmul) => {
                let tokens = input.dim(D::Minus2)?;
                if input.rank() == 2 && tokens >= 3 {
                    matmul
                        .forward(&input.unsqueeze(0)?)?
                        .squeeze(0)
                        .map_err(Into::into)
                } else {
                    matmul.forward(input).map_err(Into::into)
                }
            }
        }
    }
}

pub(super) struct RmsNorm {
    pub(super) weight: Tensor,
    pub(super) eps: f64,
}

impl RmsNorm {
    pub(super) fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let dtype = input.dtype();
        let input = input.to_dtype(DType::F32)?;
        if dtype == DType::F32
            && let Some(output) =
                super::qwen3moe::dflash_cpu_rms_norm(&input, &self.weight, self.eps)?
        {
            return output.to_dtype(dtype).map_err(Into::into);
        }
        #[cfg(target_os = "macos")]
        if let Some(output) =
            super::fused_norm_metal::try_fused_rms_norm(&input, &self.weight, self.eps as f32)?
        {
            return output.to_dtype(dtype).map_err(Into::into);
        }
        let variance = (&input * &input)?.mean_keepdim(D::Minus1)?;
        input
            .broadcast_div(&(variance + self.eps)?.sqrt()?)?
            .to_dtype(dtype)?
            .broadcast_mul(&self.weight)
            .map_err(Into::into)
    }
}

pub(super) fn take_qtensor(
    tensors: &mut HashMap<String, Arc<QTensor>>,
    name: &str,
) -> Result<Arc<QTensor>> {
    tensors
        .remove(name)
        .with_context(|| format!("missing DFlash tensor {name}"))
}

pub(super) fn take_norm(
    tensors: &mut HashMap<String, Arc<QTensor>>,
    name: &str,
    device: &Device,
    eps: f64,
) -> Result<RmsNorm> {
    let weight = take_qtensor(tensors, name)?.dequantize(device)?;
    Ok(RmsNorm { weight, eps })
}

pub(super) fn take_linear(
    tensors: &mut HashMap<String, Arc<QTensor>>,
    name: &str,
    device: &Device,
) -> Result<DraftQMatMul> {
    DraftQMatMul::from_qtensor(take_qtensor(tensors, name)?, device)
}

struct DraftMlp {
    gate: DraftQMatMul,
    up: DraftQMatMul,
    down: DraftQMatMul,
}

impl DraftMlp {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(input)?;
        let up = self.up.forward(input)?;
        let activated = match super::qwen3moe::dflash_cpu_swiglu(&gate, &up)? {
            Some(activated) => activated,
            None => (&candle_nn::Activation::Silu.forward(&gate)? * &up)?,
        };
        self.down.forward(&activated)
    }
}

struct DynamicConv {
    base: Tensor,
    projection: DraftQMatMul,
    kernel_size: usize,
    group_size: usize,
    hidden_size: usize,
}

impl DynamicConv {
    fn coefficients(&self, input: &Tensor) -> Result<Tensor> {
        self.projection.forward(input)
    }

    fn apply(&self, input: &Tensor, coefficients: &Tensor, side: usize) -> Result<Tensor> {
        let (batch, seq_len, hidden_size) = input.dims3()?;
        anyhow::ensure!(
            hidden_size == self.hidden_size,
            "DFlash conv hidden size changed"
        );
        anyhow::ensure!(side < 2, "DFlash conv side must be zero or one");
        let groups = hidden_size / self.group_size;
        #[cfg(target_os = "macos")]
        if let Some(output) = super::dflash2_conv_metal::try_dynamic_conv(
            input,
            coefficients,
            &self.base,
            side,
            self.kernel_size,
            self.group_size,
        )? {
            return Ok(output);
        }
        let coefficients = coefficients.reshape((batch, seq_len, 2, self.kernel_size, groups))?;

        let mut output: Option<Tensor> = None;
        for tap in 0..self.kernel_size {
            let values = if tap == 0 {
                input.clone()
            } else if tap >= seq_len {
                Tensor::zeros(input.shape(), input.dtype(), input.device())?
            } else {
                let zeros =
                    Tensor::zeros((batch, tap, hidden_size), input.dtype(), input.device())?;
                Tensor::cat(&[&zeros, &input.narrow(1, 0, seq_len - tap)?], 1)?
            };
            let dynamic = coefficients
                .i((.., .., side, tap, ..))?
                .unsqueeze(3)?
                .broadcast_as((batch, seq_len, groups, self.group_size))?
                .reshape((batch, seq_len, hidden_size))?;
            let base = self.base.i((side, tap, ..))?;
            let weights = dynamic.broadcast_add(&base)?;
            let term = (&weights * &values)?;
            output = Some(match output {
                Some(current) => (&current + &term)?,
                None => term,
            });
        }
        output.context("DFlash dynamic convolution has no taps")
    }
}

fn take_dynamic_conv(
    tensors: &mut HashMap<String, Arc<QTensor>>,
    prefix: &str,
    kind: &str,
    device: &Device,
    kernel_size: usize,
    group_size: usize,
    hidden_size: usize,
) -> Result<DynamicConv> {
    let base = take_qtensor(tensors, &format!("{prefix}.{kind}_conv_base"))?.dequantize(device)?;
    let projection = take_linear(
        tensors,
        &format!("{prefix}.{kind}_conv_proj.weight"),
        device,
    )?;
    Ok(DynamicConv {
        base,
        projection,
        kernel_size,
        group_size,
        hidden_size,
    })
}

struct DraftBlock {
    attn_norm: RmsNorm,
    q: DraftQMatMul,
    k: DraftQMatMul,
    v: DraftQMatMul,
    output: DraftQMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    attn_conv: DynamicConv,
    ffn_norm: RmsNorm,
    mlp: DraftMlp,
    ffn_conv: DynamicConv,
}

impl DraftBlock {
    fn injected_kv(
        &self,
        fused_target: &Tensor,
        start_pos: usize,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, seq_len, _) = fused_target.dims3()?;
        let k = self
            .k
            .forward(fused_target)?
            .reshape((batch, seq_len, kv_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v
            .forward(fused_target)?
            .reshape((batch, seq_len, kv_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self.k_norm.forward(&k)?;
        let k = apply_rope(&k, start_pos, rope_cos, rope_sin)?;
        Ok((k.to_dtype(DType::F16)?, v.to_dtype(DType::F16)?))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_noise(
        &self,
        input: &Tensor,
        cache: Option<&DraftKvCache>,
        start_pos: usize,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = input.dims3()?;
        let noise_norm = self.attn_norm.forward(input)?;
        let attn_coefficients = self.attn_conv.coefficients(&noise_norm)?;
        let attention_input = self.attn_conv.apply(&noise_norm, &attn_coefficients, 0)?;

        let q = self
            .q
            .forward(&attention_input)?
            .reshape((batch, seq_len, heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k
            .forward(&attention_input)?
            .reshape((batch, seq_len, kv_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v
            .forward(&attention_input)?
            .reshape((batch, seq_len, kv_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let q = apply_rope(&self.q_norm.forward(&q)?, start_pos, rope_cos, rope_sin)?;
        let k = apply_rope(&self.k_norm.forward(&k)?, start_pos, rope_cos, rope_sin)?;
        let k = k.to_dtype(DType::F16)?;
        let v = v.to_dtype(DType::F16)?;
        let (k, v) = match cache {
            Some(cache) => (
                Tensor::cat(&[&cache.k, &k], 2)?,
                Tensor::cat(&[&cache.v, &v], 2)?,
            ),
            None => (k, v),
        };

        let scale = 1.0 / (head_dim as f32).sqrt();
        let attention = if let Some(output) = cpu::attention(&q, &k, &v, scale)? {
            output
        } else if let Some(output) = super::try_fused_f16_sdpa_native(&q, &k, &v, false, scale)? {
            output
        } else {
            let repeats = heads / kv_heads;
            let k = repeat_kv(k.to_dtype(DType::F32)?, repeats)?;
            let v = repeat_kv(v.to_dtype(DType::F32)?, repeats)?;
            candle_nn::ops::softmax_last_dim(&(q.matmul(&k.transpose(2, 3)?)? * scale as f64)?)?
                .matmul(&v)?
        };
        let attention = attention
            .to_dtype(DType::F32)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, heads * head_dim))?;
        let attention = self.output.forward(&attention)?;
        let attention = self.attn_conv.apply(&attention, &attn_coefficients, 1)?;
        let residual = (input + &attention)?;

        let ffn_norm = self.ffn_norm.forward(&residual)?;
        let ffn_coefficients = self.ffn_conv.coefficients(&ffn_norm)?;
        let ffn_input = self.ffn_conv.apply(&ffn_norm, &ffn_coefficients, 0)?;
        let ffn = self.mlp.forward(&ffn_input)?;
        let ffn = self.ffn_conv.apply(&ffn, &ffn_coefficients, 1)?;
        (&residual + &ffn).map_err(Into::into)
    }
}

/// Loaded DFlash 2 draft model.
pub(crate) struct DFlash2Draft {
    target_layers: Vec<usize>,
    block_size: usize,
    mask_token_id: u32,
    hidden_size: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    sliding_window: usize,
    fc: DraftQMatMul,
    encoder_norm: RmsNorm,
    blocks: Vec<DraftBlock>,
    output_norm: RmsNorm,
    selector_hidden: DraftQMatMul,
    selector_predecessor: CandleEmbedding,
    selector_successor: CandleEmbedding,
    selector_top_k: usize,
    token_embedding: Arc<QTensor>,
    target_output: super::qwen35::Qwen35QMatMul,
    rope_cos: Tensor,
    rope_sin: Tensor,
    device: Device,
}

impl DFlash2Draft {
    pub(crate) fn from_gguf(
        path: &Path,
        device: &Device,
        shared: SpeculativeSharedWeights,
    ) -> Result<Self> {
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("failed to open DFlash GGUF: {}", path.display()))?;
        validate_gguf_header_budget(&mut file)
            .with_context(|| format!("failed to validate DFlash GGUF: {}", path.display()))?;
        let content = gguf_file::Content::read(&mut file)
            .with_context(|| format!("failed to parse DFlash GGUF: {}", path.display()))?;
        anyhow::ensure!(
            metadata_string(&content.metadata, "general.architecture")? == "dflash",
            "speculative sidecar is not a DFlash GGUF"
        );

        let block_count = metadata_usize(&content.metadata, "dflash.block_count")?;
        let block_size = metadata_usize(&content.metadata, "dflash.block_size")?;
        let hidden_size = metadata_usize(&content.metadata, "dflash.embedding_length")?;
        let heads = metadata_usize(&content.metadata, "dflash.attention.head_count")?;
        let kv_heads = metadata_usize(&content.metadata, "dflash.attention.head_count_kv")?;
        let head_dim = metadata_usize(&content.metadata, "dflash.attention.key_length")?;
        let context_length = metadata_usize(&content.metadata, "dflash.context_length")?;
        let sliding_window = metadata_usize(&content.metadata, "dflash.attention.sliding_window")?;
        let conv_kernel = metadata_usize(&content.metadata, "dflash.conv_kernel_size")?;
        let conv_group = metadata_usize(&content.metadata, "dflash.conv_group_size")?;
        let selector_top_k = metadata_usize(&content.metadata, "dflash.selector_top_k")?;
        let selector_rank = metadata_usize(&content.metadata, "dflash.selector_rank")?;
        let target_layers = metadata_usize_array(&content.metadata, "dflash.target_layers")?;
        let mask_token_id = u32::try_from(metadata_usize(
            &content.metadata,
            "tokenizer.ggml.mask_token_id",
        )?)
        .context("DFlash mask token exceeds u32")?;
        let eps = metadata_f64(&content.metadata, "dflash.attention.layer_norm_rms_epsilon")?;
        let rope_theta = metadata_f64(&content.metadata, "dflash.rope.freq_base")?;

        validate_layer_count(block_count, "DFlash block count")?;
        anyhow::ensure!(
            (1..=MAX_DFLASH_HIDDEN_SIZE).contains(&hidden_size),
            "DFlash hidden size must be in 1..={MAX_DFLASH_HIDDEN_SIZE}"
        );
        anyhow::ensure!(
            (1..=MAX_DFLASH_CONTEXT_LENGTH).contains(&context_length),
            "DFlash context length must be in 1..={MAX_DFLASH_CONTEXT_LENGTH}"
        );
        anyhow::ensure!(
            block_size >= 2 && block_size <= context_length,
            "DFlash block size must be in 2..={context_length}"
        );
        anyhow::ensure!(
            (2..=MAX_DFLASH_HEAD_DIM).contains(&head_dim) && head_dim.is_multiple_of(2),
            "invalid DFlash RoPE width"
        );
        anyhow::ensure!(
            heads > 0 && kv_heads > 0 && heads % kv_heads == 0,
            "invalid DFlash GQA geometry"
        );
        let attention_width = heads
            .checked_mul(head_dim)
            .context("DFlash attention width overflow")?;
        anyhow::ensure!(
            attention_width <= hidden_size,
            "invalid DFlash attention width"
        );
        anyhow::ensure!(
            conv_group > 0 && hidden_size % conv_group == 0,
            "invalid DFlash conv group size"
        );
        anyhow::ensure!(
            conv_kernel > 0 && conv_kernel <= context_length,
            "DFlash conv kernel must be in 1..={context_length}"
        );
        anyhow::ensure!(selector_top_k > 0, "DFlash selector has no candidates");
        anyhow::ensure!(selector_rank > 0, "DFlash selector rank is zero");
        anyhow::ensure!(
            sliding_window >= block_size && sliding_window <= context_length,
            "DFlash sliding window must be in {block_size}..={context_length}"
        );
        anyhow::ensure!(
            eps.is_finite() && eps > 0.0,
            "DFlash norm epsilon must be finite and positive"
        );
        anyhow::ensure!(
            rope_theta.is_finite() && rope_theta > 0.0,
            "DFlash RoPE frequency base must be finite and positive"
        );
        let fused_width = target_layers
            .len()
            .checked_mul(hidden_size)
            .context("DFlash target width overflow")?;
        anyhow::ensure!(
            fused_width > hidden_size,
            "DFlash target layer list is incomplete"
        );
        hidden_size
            .checked_mul(conv_kernel)
            .and_then(|n| n.checked_mul(2))
            .context("DFlash conv base size overflow")?;
        (hidden_size / conv_group)
            .checked_mul(conv_kernel)
            .and_then(|n| n.checked_mul(2))
            .context("DFlash conv coefficient width overflow")?;
        block_size
            .checked_mul(selector_top_k)
            .context("DFlash candidate count overflow")?;
        (block_size - 1)
            .checked_mul(selector_top_k)
            .and_then(|n| n.checked_mul(selector_top_k))
            .context("DFlash selector score count overflow")?;
        selector_rank
            .checked_mul(selector_top_k)
            .context("DFlash selector feature count overflow")?;
        context_length
            .checked_mul(head_dim / 2)
            .context("DFlash RoPE table size overflow")?;
        let (vocab_size, target_hidden_size) = shared.token_embedding.shape().dims2()?;
        anyhow::ensure!(
            target_hidden_size == hidden_size,
            "DFlash and target hidden sizes differ"
        );
        anyhow::ensure!(
            selector_top_k <= vocab_size,
            "DFlash selector candidate count exceeds target vocabulary"
        );
        anyhow::ensure!(
            (mask_token_id as usize) < vocab_size,
            "DFlash mask token exceeds target vocabulary"
        );

        let token_embedding =
            super::qwen35::prepare_packed_embedding(shared.token_embedding, device)?;
        let names = content.tensor_infos.keys().cloned().collect::<Vec<_>>();
        let mut tensors = HashMap::with_capacity(names.len());
        for name in names {
            // These tables only gather rows. Keep Q4K packed on CPU to avoid
            // Metal's F16 scale division, and upload only the gathered rows.
            let load_device = if device.is_metal()
                && matches!(
                    name.as_str(),
                    "selector_predecessor.weight" | "selector_successor.weight"
                )
                && content.tensor_infos[&name].ggml_dtype == GgmlDType::Q4K
            {
                &Device::Cpu
            } else {
                device
            };
            let tensor = content
                .tensor(&mut file, &name, load_device)
                .with_context(|| format!("failed to load DFlash tensor {name}"))?;
            tensors.insert(name, Arc::new(tensor));
        }

        // Consume source entries as each tensor is lowered. The Metal path owns an uploaded
        // packed buffer, so retaining the source Arc here would keep a second copy alive.
        let mut blocks = Vec::with_capacity(block_count);
        for layer in 0..block_count {
            let prefix = format!("blk.{layer}");
            blocks.push(DraftBlock {
                attn_norm: take_norm(
                    &mut tensors,
                    &format!("{prefix}.attn_norm.weight"),
                    device,
                    eps,
                )?,
                q: take_linear(&mut tensors, &format!("{prefix}.attn_q.weight"), device)?,
                k: take_linear(&mut tensors, &format!("{prefix}.attn_k.weight"), device)?,
                v: take_linear(&mut tensors, &format!("{prefix}.attn_v.weight"), device)?,
                output: take_linear(
                    &mut tensors,
                    &format!("{prefix}.attn_output.weight"),
                    device,
                )?,
                q_norm: take_norm(
                    &mut tensors,
                    &format!("{prefix}.attn_q_norm.weight"),
                    device,
                    eps,
                )?,
                k_norm: take_norm(
                    &mut tensors,
                    &format!("{prefix}.attn_k_norm.weight"),
                    device,
                    eps,
                )?,
                attn_conv: take_dynamic_conv(
                    &mut tensors,
                    &prefix,
                    "attn",
                    device,
                    conv_kernel,
                    conv_group,
                    hidden_size,
                )?,
                ffn_norm: take_norm(
                    &mut tensors,
                    &format!("{prefix}.ffn_norm.weight"),
                    device,
                    eps,
                )?,
                mlp: DraftMlp {
                    gate: take_linear(&mut tensors, &format!("{prefix}.ffn_gate.weight"), device)?,
                    up: take_linear(&mut tensors, &format!("{prefix}.ffn_up.weight"), device)?,
                    down: take_linear(&mut tensors, &format!("{prefix}.ffn_down.weight"), device)?,
                },
                ffn_conv: take_dynamic_conv(
                    &mut tensors,
                    &prefix,
                    "ffn",
                    device,
                    conv_kernel,
                    conv_group,
                    hidden_size,
                )?,
            });
        }

        let (rope_cos, rope_sin) = precompute_rope(head_dim, rope_theta, context_length, device)?;
        tracing::info!(
            path = %path.display(),
            block_count,
            block_size,
            selector_top_k,
            ?target_layers,
            "DFlash 2 draft loaded"
        );
        Ok(Self {
            target_layers,
            block_size,
            mask_token_id,
            hidden_size,
            heads,
            kv_heads,
            head_dim,
            sliding_window,
            fc: take_linear(&mut tensors, "fc.weight", device)?,
            encoder_norm: take_norm(&mut tensors, "enc.output_norm.weight", device, eps)?,
            blocks,
            output_norm: take_norm(&mut tensors, "output_norm.weight", device, eps)?,
            selector_hidden: take_linear(&mut tensors, "selector_hidden.weight", device)?,
            selector_predecessor: CandleEmbedding::from_arc(take_qtensor(
                &mut tensors,
                "selector_predecessor.weight",
            )?)?,
            selector_successor: CandleEmbedding::from_arc(take_qtensor(
                &mut tensors,
                "selector_successor.weight",
            )?)?,
            selector_top_k,
            token_embedding,
            target_output: shared.output,
            rope_cos,
            rope_sin,
            device: device.clone(),
        })
    }

    pub(crate) fn cache_budget_bytes(&self) -> Result<usize> {
        let vector = super::fixed_state_tensor_bytes(
            &[self.kv_heads, self.sliding_window, self.head_dim],
            &self.device,
        )?;
        vector
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_mul(self.blocks.len()))
            .context("draft request cache size overflow")
    }

    pub(crate) fn new_cache(&self) -> DFlash2Cache {
        DFlash2Cache {
            layers: (0..self.blocks.len()).map(|_| None).collect(),
            next_position: 0,
        }
    }

    pub(crate) fn target_layers(&self) -> &[usize] {
        &self.target_layers
    }

    pub(crate) fn block_size(&self) -> usize {
        self.block_size
    }

    pub(crate) fn context_length(&self) -> usize {
        self.rope_cos.dims()[0]
    }

    /// A complete CPU drafter with zero weights for session boundary tests.
    #[cfg(test)]
    pub(crate) fn test_zero_draft(block_size: usize, context_length: usize) -> Result<Self> {
        use candle_core::quantized::GgmlDType;

        let device = Device::Cpu;
        let matrix = |rows, cols| -> Result<QMatMul> {
            let tensor = Tensor::zeros((rows, cols), DType::F32, &device)?;
            Ok(QMatMul::from_arc(Arc::new(QTensor::quantize(
                &tensor,
                GgmlDType::F32,
            )?))?)
        };
        let selector = |rows, cols| -> Result<CandleEmbedding> {
            Ok(CandleEmbedding::from_arc(Arc::new(QTensor::quantize(
                &Tensor::zeros((rows, cols), DType::F32, &device)?,
                GgmlDType::F32,
            )?))?)
        };
        let linear = |rows, cols| matrix(rows, cols).map(DraftQMatMul::Candle);
        let norm = || -> Result<RmsNorm> {
            Ok(RmsNorm {
                weight: Tensor::ones(64, DType::F32, &device)?,
                eps: 1e-6,
            })
        };
        let conv = || -> Result<DynamicConv> {
            Ok(DynamicConv {
                base: Tensor::zeros((2, 1, 64), DType::F32, &device)?,
                projection: linear(2, 64)?,
                kernel_size: 1,
                group_size: 64,
                hidden_size: 64,
            })
        };
        let embedding = Arc::new(QTensor::quantize(
            &Tensor::zeros((3, 64), DType::F32, &device)?,
            GgmlDType::F32,
        )?);
        let (rope_cos, rope_sin) = precompute_rope(64, 10_000.0, context_length, &device)?;
        Ok(Self {
            target_layers: vec![0, 1],
            block_size,
            mask_token_id: 0,
            hidden_size: 64,
            heads: 1,
            kv_heads: 1,
            head_dim: 64,
            sliding_window: context_length,
            fc: linear(64, 128)?,
            encoder_norm: norm()?,
            blocks: vec![DraftBlock {
                attn_norm: norm()?,
                q: linear(64, 64)?,
                k: linear(64, 64)?,
                v: linear(64, 64)?,
                output: linear(64, 64)?,
                q_norm: norm()?,
                k_norm: norm()?,
                attn_conv: conv()?,
                ffn_norm: norm()?,
                mlp: DraftMlp {
                    gate: linear(64, 64)?,
                    up: linear(64, 64)?,
                    down: linear(64, 64)?,
                },
                ffn_conv: conv()?,
            }],
            output_norm: norm()?,
            selector_hidden: linear(1, 64)?,
            selector_predecessor: selector(3, 1)?,
            selector_successor: selector(3, 1)?,
            selector_top_k: 1,
            token_embedding: Arc::clone(&embedding),
            target_output: super::qwen35::Qwen35QMatMul::Candle(QMatMul::from_arc(embedding)?),
            rope_cos,
            rope_sin,
            device,
        })
    }

    /// Fuse committed target rows and append their projected K/V to the draft.
    pub(crate) fn inject(
        &self,
        cache: &mut DFlash2Cache,
        layer_inputs: &[Tensor],
        start_pos: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            start_pos == cache.next_position,
            "non-contiguous DFlash injection: expected {}, got {start_pos}",
            cache.next_position
        );
        anyhow::ensure!(
            layer_inputs.len() == self.target_layers.len(),
            "DFlash received {} target layers, expected {}",
            layer_inputs.len(),
            self.target_layers.len()
        );
        let seq_len = layer_inputs
            .first()
            .context("DFlash target feature list is empty")?
            .dim(0)?;
        anyhow::ensure!(seq_len > 0, "DFlash cannot inject an empty sequence");
        for input in layer_inputs {
            anyhow::ensure!(
                input.dims2()? == (seq_len, self.hidden_size),
                "DFlash target feature shape changed"
            );
        }
        let refs = layer_inputs.iter().collect::<Vec<_>>();
        let fused = Tensor::cat(&refs, 1)?.unsqueeze(0)?;
        let fused = self.encoder_norm.forward(&self.fc.forward(&fused)?)?;

        let mut next = cache.clone();
        let retained_past = self.sliding_window.saturating_sub(self.block_size);
        for (layer, block) in self.blocks.iter().enumerate() {
            let (k, v) = block.injected_kv(
                &fused,
                start_pos,
                &self.rope_cos,
                &self.rope_sin,
                self.kv_heads,
                self.head_dim,
            )?;
            append_rolling_kv(&mut next.layers[layer], &k, &v, retained_past)?;
        }
        next.next_position = start_pos
            .checked_add(seq_len)
            .context("DFlash position overflow")?;
        *cache = next;
        Ok(())
    }

    /// Draft `block_size - 1` tokens in one non-causal pass.
    pub(crate) fn draft(
        &self,
        cache: &DFlash2Cache,
        anchor: u32,
        start_pos: usize,
        block_size: usize,
    ) -> Result<Vec<u32>> {
        anyhow::ensure!(
            start_pos == cache.next_position,
            "DFlash target and draft caches are out of step"
        );
        anyhow::ensure!(
            (2..=self.block_size).contains(&block_size),
            "DFlash block size {block_size} is outside 2..={}",
            self.block_size
        );
        let mut tokens = vec![self.mask_token_id; block_size];
        tokens[0] = anchor;
        let mut hidden = super::qwen35::packed_embedding_rows(
            &self.token_embedding,
            tokens.as_slice(),
            &self.device,
        )?
        .reshape((1, block_size, self.hidden_size))?;
        for (layer, block) in self.blocks.iter().enumerate() {
            hidden = block.forward_noise(
                &hidden,
                cache.layers[layer].as_ref(),
                start_pos,
                &self.rope_cos,
                &self.rope_sin,
                self.heads,
                self.kv_heads,
                self.head_dim,
            )?;
        }
        let hidden = self.output_norm.forward(&hidden)?;
        let logits = self.output_logits(&hidden.squeeze(0)?)?;
        self.select_path(anchor, &hidden, &logits, block_size)
    }

    fn output_logits(&self, input: &Tensor) -> Result<Tensor> {
        Ok(match &self.target_output {
            super::qwen35::Qwen35QMatMul::Candle(matmul) => {
                match super::qwen3moe::dflash_cpu_projection(matmul, input)? {
                    Some(output) => output,
                    None => self.target_output.forward(input)?,
                }
            }
            #[cfg(target_os = "macos")]
            super::qwen35::Qwen35QMatMul::MetalTensor(_) => self.target_output.forward(input)?,
        })
    }

    fn select_path(
        &self,
        anchor: u32,
        hidden: &Tensor,
        logits: &Tensor,
        block_size: usize,
    ) -> Result<Vec<u32>> {
        let logits = logits.contiguous()?;
        anyhow::ensure!(
            logits.dims2()?.0 == block_size,
            "DFlash output row count changed"
        );
        #[cfg(target_os = "macos")]
        let metal_top_k = if self.device.is_metal() {
            // The validity check is deferred to the candidate read below,
            // which the CPU selector tables need anyway: one GPU round trip.
            super::dflash2_topk_metal::try_top_k_deferred(
                &logits.to_dtype(DType::F32)?,
                self.selector_top_k,
            )?
        } else {
            None
        };
        #[cfg(not(target_os = "macos"))]
        let metal_top_k: Option<(Tensor, Tensor)> = None;

        let (candidates, unary) = if let Some(top_k) = metal_top_k {
            top_k
        } else {
            let rows = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
            let mut candidate_ids = Vec::with_capacity(block_size * self.selector_top_k);
            let mut candidate_logits = Vec::with_capacity(candidate_ids.capacity());
            for (row_index, row) in rows.into_iter().enumerate() {
                anyhow::ensure!(
                    row.iter().all(|logit| logit.is_finite()),
                    "DFlash vocabulary logits must be finite (row {row_index})"
                );
                let indexed = select_top_k(row, self.selector_top_k)?;
                candidate_ids.extend(indexed.iter().map(|(id, _)| *id as u32));
                candidate_logits.extend(indexed.into_iter().map(|(_, logit)| logit));
            }
            (
                Tensor::from_vec(
                    candidate_ids,
                    (block_size, self.selector_top_k),
                    &self.device,
                )?,
                Tensor::from_vec(
                    candidate_logits,
                    (block_size, self.selector_top_k),
                    &self.device,
                )?,
            )
        };
        let selector_hidden = self.selector_hidden.forward(hidden)?.squeeze(0)?;
        // One host read of the candidates serves the CPU table gathers below
        // and the deferred non-finite check from the Metal top-k.
        let candidate_rows = candidates.to_vec2::<u32>()?;
        check_candidate_ids(&candidate_rows, logits.dims2()?.1)?;
        #[cfg(target_os = "macos")]
        if self.device.is_metal() && self.selector_top_k <= 16 {
            let host_ids = |rows: std::ops::Range<usize>| {
                candidate_rows[rows]
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<u32>>()
            };
            let successor_ids = Tensor::new(host_ids(1..block_size), &Device::Cpu)?;
            let successors = self
                .selector_successor
                .embedding(&successor_ids)?
                .to_device(&self.device)?;
            let mut predecessor_values = vec![anchor];
            if block_size > 2 {
                predecessor_values.extend(host_ids(1..block_size - 1));
            }
            let predecessor_ids = Tensor::new(predecessor_values, &Device::Cpu)?;
            let predecessors = self
                .selector_predecessor
                .embedding(&predecessor_ids)?
                .to_device(&self.device)?;
            if let Some(selected) = super::dflash2_selector_metal::try_select_path(
                &selector_hidden,
                &predecessors,
                &successors,
                &unary,
                &candidates,
                block_size,
                self.selector_top_k,
            )? {
                return selected.to_vec1::<u32>().map_err(Into::into);
            }
        }
        let mut score_matrices = Vec::with_capacity(block_size - 1);
        for position in 1..block_size {
            let candidate_ids = candidates.i(position)?.contiguous()?;
            let successors = self
                .selector_successor
                .embedding(&candidate_ids)?
                .to_device(&self.device)?;
            let predecessor_ids = if position == 1 {
                Tensor::new(&[anchor], &self.device)?
            } else {
                candidates.i(position - 1)?.contiguous()?
            };
            let predecessor = self
                .selector_predecessor
                .embedding(&predecessor_ids)?
                .to_device(&self.device)?;
            let conditioned = predecessor.broadcast_mul(&selector_hidden.i(position)?)?;
            let mut scores = conditioned
                .matmul(&successors.transpose(0, 1)?)?
                .broadcast_add(&unary.i(position)?)?;
            if position == 1 {
                scores = scores.broadcast_as((self.selector_top_k, self.selector_top_k))?;
            }
            score_matrices.push(scores.unsqueeze(0)?);
        }

        let candidates = candidates.narrow(0, 1, block_size - 1)?.to_vec2::<u32>()?;
        let scores = Tensor::cat(&score_matrices, 0)?.to_vec3::<f32>()?;
        let mut predecessor = 0usize;
        let mut result = Vec::with_capacity(block_size - 1);
        for position in 0..block_size - 1 {
            let row = &scores[position][predecessor];
            let selected =
                row.iter().enumerate().fold(
                    0usize,
                    |best, (index, score)| {
                        if score > &row[best] { index } else { best }
                    },
                );
            result.push(candidates[position][selected]);
            predecessor = selected;
        }
        Ok(result)
    }
}

/// The deferred Metal top-k marks non-finite logits with IDs equal to the
/// vocabulary size; fail closed on any out-of-range candidate.
fn check_candidate_ids(ids: &[Vec<u32>], vocab: usize) -> Result<()> {
    anyhow::ensure!(
        ids.iter().flatten().all(|&id| (id as usize) < vocab),
        "DFlash vocabulary logits must be finite"
    );
    Ok(())
}

fn select_top_k(row: Vec<f32>, top_k: usize) -> Result<Vec<(usize, f32)>> {
    anyhow::ensure!(
        top_k > 0 && top_k <= row.len(),
        "DFlash selector candidate count exceeds vocabulary"
    );
    let mut indexed = row.into_iter().enumerate().collect::<Vec<_>>();
    let compare = |left: &(usize, f32), right: &(usize, f32)| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    };
    if top_k < indexed.len() {
        indexed.select_nth_unstable_by(top_k, compare);
        indexed.truncate(top_k);
    }
    indexed.sort_unstable_by(compare);
    Ok(indexed)
}

fn append_rolling_kv(
    cache: &mut Option<DraftKvCache>,
    k: &Tensor,
    v: &Tensor,
    keep: usize,
) -> Result<()> {
    if keep == 0 {
        *cache = None;
        return Ok(());
    }
    let (mut k, mut v) = match cache.as_ref() {
        Some(cache) => (
            Tensor::cat(&[&cache.k, k], 2)?,
            Tensor::cat(&[&cache.v, v], 2)?,
        ),
        None => (k.clone(), v.clone()),
    };
    let len = k.dim(2)?;
    if len > keep {
        k = k.narrow(2, len - keep, keep)?;
        v = v.narrow(2, len - keep, keep)?;
    }
    // Row views can retain a whole prompt, and pooled projections can reuse
    // larger buffers even before the window fills. Store only live rows.
    k = super::copy_persistent_tensor(&k)?;
    v = super::copy_persistent_tensor(&v)?;
    *cache = Some(DraftKvCache { k, v });
    Ok(())
}

pub(super) fn repeat_kv(input: Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(input);
    }
    let (batch, heads, seq_len, head_dim) = input.dims4()?;
    input
        .unsqueeze(2)?
        .expand((batch, heads, repeats, seq_len, head_dim))?
        .reshape((batch, heads * repeats, seq_len, head_dim))?
        .contiguous()
        .map_err(Into::into)
}

fn precompute_rope(
    head_dim: usize,
    theta: f64,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    anyhow::ensure!(
        (2..=MAX_DFLASH_HEAD_DIM).contains(&head_dim) && head_dim.is_multiple_of(2),
        "invalid DFlash RoPE width"
    );
    anyhow::ensure!(
        (1..=MAX_DFLASH_CONTEXT_LENGTH).contains(&context_length),
        "invalid DFlash RoPE context length"
    );
    anyhow::ensure!(
        theta.is_finite() && theta > 0.0,
        "invalid DFlash RoPE frequency base"
    );
    context_length
        .checked_mul(head_dim / 2)
        .context("DFlash RoPE table size overflow")?;
    let context_end = u32::try_from(context_length).context("DFlash context length exceeds u32")?;
    if let Some(tables) = cpu::rope_tables(head_dim, theta as f32, context_length, device)? {
        return Ok(tables);
    }
    let half = head_dim / 2;
    let inv_freq = (0..half)
        .map(|index| 1.0 / (theta as f32).powf(2.0 * index as f32 / head_dim as f32))
        .collect::<Vec<_>>();
    let inv_freq = Tensor::new(inv_freq.as_slice(), device)?.reshape((1, half))?;
    let positions = Tensor::arange(0u32, context_end, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?;
    let angles = positions.matmul(&inv_freq)?;
    Ok((angles.cos()?, angles.sin()?))
}

fn apply_rope(input: &Tensor, start_pos: usize, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let seq_len = input.dim(2)?;
    let cos = cos.narrow(0, start_pos, seq_len)?;
    let sin = sin.narrow(0, start_pos, seq_len)?;
    if let Some(output) = cpu::rope(input, &cos, &sin)? {
        return Ok(output);
    }
    candle_nn::rotary_emb::rope(&input.contiguous()?, &cos, &sin).map_err(Into::into)
}

pub(super) fn metadata_string(md: &HashMap<String, gguf_file::Value>, key: &str) -> Result<String> {
    match md.get(key) {
        Some(gguf_file::Value::String(value)) => Ok(value.clone()),
        _ => anyhow::bail!("missing or invalid DFlash metadata key {key}"),
    }
}

pub(super) fn metadata_usize(md: &HashMap<String, gguf_file::Value>, key: &str) -> Result<usize> {
    match md.get(key) {
        Some(gguf_file::Value::U32(value)) => Ok(*value as usize),
        Some(gguf_file::Value::U64(value)) => usize::try_from(*value).context(key.to_string()),
        Some(gguf_file::Value::I32(value)) => usize::try_from(*value).context(key.to_string()),
        _ => anyhow::bail!("missing or invalid DFlash metadata key {key}"),
    }
}

pub(super) fn metadata_usize_array(
    md: &HashMap<String, gguf_file::Value>,
    key: &str,
) -> Result<Vec<usize>> {
    let Some(gguf_file::Value::Array(values)) = md.get(key) else {
        anyhow::bail!("missing or invalid DFlash metadata key {key}");
    };
    values
        .iter()
        .map(|value| match value {
            gguf_file::Value::U32(value) => Ok(*value as usize),
            gguf_file::Value::U64(value) => usize::try_from(*value).context(key.to_string()),
            gguf_file::Value::I32(value) => usize::try_from(*value).context(key.to_string()),
            _ => anyhow::bail!("invalid integer in DFlash metadata key {key}"),
        })
        .collect()
}

pub(super) fn metadata_f64(md: &HashMap<String, gguf_file::Value>, key: &str) -> Result<f64> {
    match md.get(key) {
        Some(gguf_file::Value::F32(value)) => Ok(*value as f64),
        Some(gguf_file::Value::F64(value)) => Ok(*value),
        _ => anyhow::bail!("missing or invalid DFlash metadata key {key}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;

    fn shared_weights() -> SpeculativeSharedWeights {
        let tensor = Tensor::zeros((4, 64), DType::F32, &Device::Cpu).unwrap();
        let embedding = Arc::new(QTensor::quantize(&tensor, GgmlDType::F32).unwrap());
        SpeculativeSharedWeights {
            token_embedding: Arc::clone(&embedding),
            output: super::super::qwen35::Qwen35QMatMul::Candle(
                QMatMul::from_arc(embedding).unwrap(),
            ),
        }
    }

    fn metadata_fixture(key: &str, value: gguf_file::Value) -> tempfile::NamedTempFile {
        let mut metadata = vec![
            (
                "general.architecture",
                gguf_file::Value::String("dflash".into()),
            ),
            ("dflash.block_count", gguf_file::Value::U64(1)),
            ("dflash.block_size", gguf_file::Value::U64(2)),
            ("dflash.embedding_length", gguf_file::Value::U64(64)),
            ("dflash.attention.head_count", gguf_file::Value::U64(1)),
            ("dflash.attention.head_count_kv", gguf_file::Value::U64(1)),
            ("dflash.attention.key_length", gguf_file::Value::U64(64)),
            ("dflash.context_length", gguf_file::Value::U64(32)),
            ("dflash.attention.sliding_window", gguf_file::Value::U64(32)),
            ("dflash.conv_kernel_size", gguf_file::Value::U64(4)),
            ("dflash.conv_group_size", gguf_file::Value::U64(16)),
            ("dflash.selector_top_k", gguf_file::Value::U64(1)),
            ("dflash.selector_rank", gguf_file::Value::U64(1)),
            (
                "dflash.target_layers",
                gguf_file::Value::Array(vec![gguf_file::Value::U32(0), gguf_file::Value::U32(1)]),
            ),
            ("tokenizer.ggml.mask_token_id", gguf_file::Value::U64(0)),
            (
                "dflash.attention.layer_norm_rms_epsilon",
                gguf_file::Value::F64(1e-6),
            ),
            ("dflash.rope.freq_base", gguf_file::Value::F64(10_000.0)),
        ];
        metadata
            .iter_mut()
            .find(|(name, _)| *name == key)
            .unwrap()
            .1 = value;
        let metadata = metadata
            .iter()
            .map(|(name, value)| (*name, value))
            .collect::<Vec<_>>();
        let mut file = tempfile::NamedTempFile::new().unwrap();
        gguf_file::write(file.as_file_mut(), &metadata, &[]).unwrap();
        file
    }

    #[test]
    fn loader_rejects_invalid_geometry_before_loading_weights() {
        for (key, value, message) in [
            (
                "dflash.conv_group_size",
                gguf_file::Value::U64(0),
                "conv group size",
            ),
            (
                "dflash.conv_group_size",
                gguf_file::Value::U64(3),
                "conv group size",
            ),
            (
                "dflash.attention.head_count",
                gguf_file::Value::U64(usize::MAX as u64),
                "attention width overflow",
            ),
            (
                "dflash.attention.key_length",
                gguf_file::Value::U64(0),
                "RoPE width",
            ),
            (
                "dflash.embedding_length",
                gguf_file::Value::U64(0),
                "hidden size",
            ),
            (
                "dflash.block_count",
                gguf_file::Value::U64(usize::MAX as u64),
                "block count",
            ),
            (
                "dflash.context_length",
                gguf_file::Value::U64(0),
                "context length",
            ),
            (
                "dflash.context_length",
                gguf_file::Value::U64(usize::MAX as u64),
                "context length",
            ),
            (
                "dflash.selector_top_k",
                gguf_file::Value::U64(5),
                "exceeds target vocabulary",
            ),
            (
                "dflash.selector_rank",
                gguf_file::Value::U64(0),
                "selector rank",
            ),
            (
                "dflash.conv_kernel_size",
                gguf_file::Value::U64(usize::MAX as u64),
                "conv kernel",
            ),
            (
                "dflash.attention.layer_norm_rms_epsilon",
                gguf_file::Value::F64(f64::NAN),
                "norm epsilon",
            ),
            (
                "dflash.rope.freq_base",
                gguf_file::Value::F64(f64::INFINITY),
                "frequency base",
            ),
        ] {
            let fixture = metadata_fixture(key, value);
            let error =
                match DFlash2Draft::from_gguf(fixture.path(), &Device::Cpu, shared_weights()) {
                    Ok(_) => panic!("accepted invalid {key}"),
                    Err(error) => error.to_string(),
                };
            assert!(error.contains(message), "{key}: {error}");
        }
    }

    #[test]
    fn selector_accepts_the_whole_vocabulary_and_keeps_tie_order() {
        assert_eq!(
            select_top_k(vec![0.0, 1.0, 1.0, -1.0], 4).unwrap(),
            vec![(1, 1.0), (2, 1.0), (0, 0.0), (3, -1.0)]
        );
        assert_eq!(
            select_top_k(vec![0.0, 1.0, 1.0, -1.0], 2).unwrap(),
            vec![(1, 1.0), (2, 1.0)]
        );
        assert!(select_top_k(vec![0.0], 2).is_err());
        assert!(select_top_k(vec![0.0], 0).is_err());
    }

    fn rolling_cache_storage_is_bounded(device: &Device) -> Result<()> {
        for heads in [1, 2] {
            for (source_len, keep) in [(64, 0), (64, 4), (3, 4)] {
                let source = Tensor::ones((1, heads, 64, 8), DType::F16, device)?
                    .narrow(2, 0, source_len)?;
                let mut cache = None;
                append_rolling_kv(&mut cache, &source, &source, keep)?;
                if keep == 0 {
                    assert!(cache.is_none());
                    continue;
                }
                let cache = cache.as_ref().unwrap();
                for tensor in [&cache.k, &cache.v] {
                    assert_eq!(tensor.dims(), &[1, heads, keep.min(source_len), 8]);
                    let expected = tensor.elem_count() * DType::F16.size_in_bytes();
                    let (storage, layout) = tensor.storage_and_layout();
                    assert_eq!(layout.start_offset(), 0);
                    match &*storage {
                        candle_core::Storage::Cpu(candle_core::CpuStorage::F16(values)) => {
                            assert_eq!(values.len() * 2, expected);
                        }
                        #[cfg(target_os = "macos")]
                        candle_core::Storage::Metal(storage) => {
                            assert_eq!(storage.buffer().length(), expected);
                        }
                        _ => anyhow::bail!("unexpected rolling-cache storage"),
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn rolling_cache_drops_large_cpu_backings() -> Result<()> {
        rolling_cache_storage_is_bounded(&Device::Cpu)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rolling_cache_drops_large_metal_backings() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        rolling_cache_storage_is_bounded(&device)
    }

    #[test]
    fn rope_rejects_invalid_geometry_before_allocation() {
        assert!(precompute_rope(0, 10_000.0, 32, &Device::Cpu).is_err());
        assert!(precompute_rope(64, 10_000.0, usize::MAX, &Device::Cpu).is_err());
        assert!(precompute_rope(64, f64::NAN, 32, &Device::Cpu).is_err());
    }
}

#[cfg(test)]
#[path = "dflash2_reference_tests.rs"]
mod reference_tests;
