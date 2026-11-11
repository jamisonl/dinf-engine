//! Text-only Qwen3-VL conditioning encoder.
//!
//! Qwen3-VL uses a Qwen3 decoder stack for its language side.  This module
//! deliberately implements only that stack: image/video tokens and the
//! vision tower are owned by the future multimodal pipeline.  The output is
//! the hidden state immediately after the final decoder layer, before the
//! model's final RMSNorm.  Qwen Image consumes that pre-norm conditioning
//! state.

use std::sync::Arc;

use anyhow::ensure;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{
    Activation, Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_no_bias, rms_norm,
};
use serde::Deserialize;

use super::attention::attention;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "text_linear_mps.rs"]
mod text_linear_mps;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "text_norm_metal.rs"]
mod text_norm_metal;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "text_silu_metal.rs"]
mod text_silu_metal;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "text_softmax_metal.rs"]
mod text_softmax_metal;

// These limits keep an untrusted JSON config from turning load into an
// unbounded allocation while still allowing small deterministic fixtures and
// future Qwen3-VL variants.  The released 8B text config is 4096/12288,
// 36 layers, 32/8 heads, and head_dim=128.
const MAX_HIDDEN_SIZE: usize = 16_384;
const MAX_INTERMEDIATE_SIZE: usize = 65_536;
const MAX_LAYERS: usize = 128;
const MAX_HEADS: usize = 128;
const MAX_KV_HEADS: usize = 128;
const MAX_HEAD_DIM: usize = 1024;
const MAX_CONTEXT: usize = 1_048_576;
const MAX_VOCAB_SIZE: usize = 2_000_000;
const MAX_KV_REPEAT: usize = 16;
const EXPECTED_ROPE_THETA: f64 = 5_000_000.0;

/// The outer Qwen3-VL configuration wrapper.
///
/// The vision fields and top-level model metadata are intentionally ignored
/// by serde here.  The text encoder needs only `text_config`; the pipeline
/// owns parsing and validating the vision side when it is added.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct TextEncoderConfig {
    pub text_config: TextConfig,
}

/// The Qwen3-VL-8B language-model configuration.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct TextConfig {
    pub attention_bias: bool,
    pub head_dim: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    #[serde(default, alias = "rope_parameters")]
    pub rope_scaling: Option<RopeScalingConfig>,
    pub rope_theta: f64,
    pub vocab_size: usize,
}

/// The only rope scaling metadata currently compatible with the text-only
/// path.  Qwen3-VL's released config describes default, interleaved MRoPE;
/// text positions use the same value on each axis, so this reduces to the
/// standard one-dimensional half rotation below.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RopeScalingConfig {
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub mrope_interleaved: Option<bool>,
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
}

impl TextConfig {
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            !self.attention_bias,
            "Qwen3-VL text attention must use bias-free projections"
        );
        ensure!(
            self.hidden_size > 0 && self.hidden_size <= MAX_HIDDEN_SIZE,
            "Qwen3-VL text hidden size must be in 1..={MAX_HIDDEN_SIZE}, got {}",
            self.hidden_size
        );
        ensure!(
            self.intermediate_size > 0 && self.intermediate_size <= MAX_INTERMEDIATE_SIZE,
            "Qwen3-VL text intermediate size must be in 1..={MAX_INTERMEDIATE_SIZE}, got {}",
            self.intermediate_size
        );
        ensure!(
            self.num_hidden_layers > 0 && self.num_hidden_layers <= MAX_LAYERS,
            "Qwen3-VL text layer count must be in 1..={MAX_LAYERS}, got {}",
            self.num_hidden_layers
        );
        ensure!(
            self.num_attention_heads > 0 && self.num_attention_heads <= MAX_HEADS,
            "Qwen3-VL text head count must be in 1..={MAX_HEADS}, got {}",
            self.num_attention_heads
        );
        ensure!(
            self.num_key_value_heads > 0 && self.num_key_value_heads <= MAX_KV_HEADS,
            "Qwen3-VL text KV head count must be in 1..={MAX_KV_HEADS}, got {}",
            self.num_key_value_heads
        );
        ensure!(
            self.head_dim > 0 && self.head_dim <= MAX_HEAD_DIM && self.head_dim.is_multiple_of(2),
            "Qwen3-VL text head dimension must be a positive even value <= {MAX_HEAD_DIM}, got {}",
            self.head_dim
        );
        ensure!(
            self.num_attention_heads
                .is_multiple_of(self.num_key_value_heads),
            "Qwen3-VL attention heads must be divisible by KV heads"
        );
        ensure!(
            self.num_attention_heads.checked_mul(self.head_dim) == Some(self.hidden_size),
            "Qwen3-VL hidden size must equal heads * head_dim"
        );
        ensure!(
            self.num_attention_heads / self.num_key_value_heads <= MAX_KV_REPEAT,
            "Qwen3-VL text KV repeat factor must be <= {MAX_KV_REPEAT}"
        );
        ensure!(
            self.hidden_act == Activation::Silu,
            "Qwen3-VL text MLP must use SiLU activation"
        );
        ensure!(
            self.vocab_size > 0 && self.vocab_size <= MAX_VOCAB_SIZE,
            "Qwen3-VL text vocabulary must be in 1..={MAX_VOCAB_SIZE}, got {}",
            self.vocab_size
        );
        ensure!(
            self.max_position_embeddings > 0 && self.max_position_embeddings <= MAX_CONTEXT,
            "Qwen3-VL text max_position_embeddings must be in 1..={MAX_CONTEXT}, got {}",
            self.max_position_embeddings
        );
        ensure!(
            self.rms_norm_eps.is_finite()
                && (self.rms_norm_eps as f32).is_finite()
                && (self.rms_norm_eps as f32) > 0.0,
            "Qwen3-VL text rms_norm_eps must be positive and finite in F32"
        );
        ensure!(
            self.rope_theta.is_finite() && self.rope_theta == EXPECTED_ROPE_THETA,
            "Qwen3-VL text rope_theta must be {EXPECTED_ROPE_THETA}, got {}",
            self.rope_theta
        );
        if let Some(scaling) = &self.rope_scaling {
            ensure!(
                scaling.rope_type.as_deref().unwrap_or("default") == "default",
                "Qwen3-VL text only supports default rope scaling"
            );
            if let Some(interleaved) = scaling.mrope_interleaved {
                ensure!(
                    interleaved,
                    "Qwen3-VL text requires interleaved default MRoPE metadata"
                );
            }
            if let Some(section) = &scaling.mrope_section {
                ensure!(
                    !section.is_empty()
                        && section
                            .iter()
                            .all(|value| *value > 0 && value.is_multiple_of(2)),
                    "Qwen3-VL text MRoPE sections must be positive even dimensions"
                );
            }
        }
        Ok(())
    }
}

/// Standard one-dimensional half-rotation RoPE for text positions.
///
/// Qwen3-VL's multimodal path uses three-axis MRoPE, but text-only
/// conditioning uses the ordinary Qwen3 sequence axis.  The table is built
/// in F32 and converted to the model dtype, matching Candle's Qwen3 path.
#[derive(Debug, Clone)]
struct RotaryEmbedding {
    theta: f64,
    head_dim: usize,
    max_position_embeddings: usize,
    device: Device,
    dtype: DType,
}

impl RotaryEmbedding {
    fn new(config: &TextConfig, device: &Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            theta: config.rope_theta,
            head_dim: config.head_dim,
            max_position_embeddings: config.max_position_embeddings,
            device: device.clone(),
            dtype,
        })
    }

    fn table(&self, seq_len: usize) -> Result<(Tensor, Tensor)> {
        if seq_len > self.max_position_embeddings {
            candle_core::bail!(
                "Qwen3-VL text sequence length {seq_len} exceeds rotary context {}",
                self.max_position_embeddings
            );
        }
        let half_dim = self.head_dim / 2;
        // Candle's standard `rope` is the half-rotation (NeoX) layout.  The
        // exponent uses the even coordinate `2*i`, as in the reference
        // Qwen3 implementation, rather than the compressed index `i`.
        // The BF16 reference forms these frequencies in F32 before applying
        // F32 trig and casting the tables. Preserve the existing F32 path.
        let inv_freq = (0..half_dim)
            .map(|index| {
                if self.dtype == DType::BF16 {
                    (self.theta as f32)
                        .powf(2.0 * index as f32 / self.head_dim as f32)
                        .recip()
                } else {
                    (1.0 / self.theta.powf(2.0 * index as f64 / self.head_dim as f64)) as f32
                }
            })
            .collect::<Vec<_>>();
        let inv_freq =
            Tensor::from_vec(inv_freq, (1, half_dim), &self.device)?.to_dtype(DType::F32)?;
        let positions = Tensor::arange(0u32, seq_len as u32, &self.device)?
            .to_dtype(DType::F32)?
            .reshape((seq_len, 1))?;
        let angles = positions.matmul(&inv_freq)?;
        Ok((
            angles.cos()?.to_dtype(self.dtype)?,
            angles.sin()?.to_dtype(self.dtype)?,
        ))
    }
}

fn text_activation(x: &Tensor, activation: Activation) -> Result<Tensor> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if activation == Activation::Silu
        && let Some(y) = text_silu_metal::try_silu(x)?
    {
        return Ok(y);
    }
    x.apply(&activation)
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    activation: Activation,
}

impl Mlp {
    fn load(config: &TextConfig, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(
                config.hidden_size,
                config.intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up_proj: linear_no_bias(
                config.hidden_size,
                config.intermediate_size,
                vb.pp("up_proj"),
            )?,
            down_proj: linear_no_bias(
                config.intermediate_size,
                config.hidden_size,
                vb.pp("down_proj"),
            )?,
            activation: config.hidden_act,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let gate = text_activation(&text_linear(&self.gate_proj, hidden)?, self.activation)?;
        let up = text_linear(&self.up_proj, hidden)?;
        text_linear(&self.down_proj, &(gate * up)?)
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
}

impl Attention {
    fn load(config: &TextConfig, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            q_proj: linear_no_bias(
                config.hidden_size,
                config.num_attention_heads * config.head_dim,
                vb.pp("q_proj"),
            )?,
            k_proj: linear_no_bias(
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
                vb.pp("k_proj"),
            )?,
            v_proj: linear_no_bias(
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
                vb.pp("v_proj"),
            )?,
            o_proj: linear_no_bias(
                config.num_attention_heads * config.head_dim,
                config.hidden_size,
                vb.pp("o_proj"),
            )?,
            q_norm: rms_norm(config.head_dim, config.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms_norm(config.head_dim, config.rms_norm_eps, vb.pp("k_norm"))?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        config: &TextConfig,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = hidden.dims3()?;
        let q = text_linear(&self.q_proj, hidden)?
            .reshape((batch, seq_len, config.num_attention_heads, config.head_dim))?
            .transpose(1, 2)?;
        let k = text_linear(&self.k_proj, hidden)?
            .reshape((batch, seq_len, config.num_key_value_heads, config.head_dim))?
            .transpose(1, 2)?;
        let v = text_linear(&self.v_proj, hidden)?
            .reshape((batch, seq_len, config.num_key_value_heads, config.head_dim))?
            .transpose(1, 2)?;

        let q = normalize_heads(
            &self.q_norm,
            &q,
            batch,
            config.num_attention_heads,
            seq_len,
            config.head_dim,
        )?;
        let k = normalize_heads(
            &self.k_norm,
            &k,
            batch,
            config.num_key_value_heads,
            seq_len,
            config.head_dim,
        )?;
        let q = text_rope(&q.contiguous()?, cos, sin)?;
        let k = text_rope(&k.contiguous()?, cos, sin)?;
        let repeats = config.num_attention_heads / config.num_key_value_heads;
        let k = repeat_kv(&k, repeats)?;
        let v = repeat_kv(&v, repeats)?;
        let attended = text_attention(&q, &k, &v)?;
        text_linear(
            &self.o_proj,
            &attended
                .transpose(1, 2)?
                .reshape((batch, seq_len, config.hidden_size))?,
        )
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn load(config: &TextConfig, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::load(config, vb.pp("self_attn"))?,
            mlp: Mlp::load(config, vb.pp("mlp"))?,
            input_layernorm: rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        config: &TextConfig,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let attention_input = text_rms_norm(&self.input_layernorm, hidden)?;
        let attention_output = self.self_attn.forward(&attention_input, config, cos, sin)?;
        let hidden = (hidden + attention_output)?;
        let mlp_input = text_rms_norm(&self.post_attention_layernorm, &hidden)?;
        let mlp_output = self.mlp.forward(&mlp_input)?;
        hidden + mlp_output
    }
}

/// Native text-only Qwen3-VL conditioning encoder.
pub(crate) struct TextEncoder {
    config: TextConfig,
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    rotary: Arc<RotaryEmbedding>,
}

impl TextEncoder {
    /// Load Qwen3-VL text weights from a checkpoint-root `VarBuilder`.
    ///
    /// The expected names are under `model.language_model`, matching the
    /// released Qwen3-VL safetensors layout.  The final `norm` is deliberately
    /// not loaded because callers need the final layer's pre-RMSNorm state.
    pub(crate) fn load(vb: VarBuilder<'_>, config: TextEncoderConfig) -> anyhow::Result<Self> {
        config.text_config.validate()?;
        ensure!(
            matches!(vb.dtype(), DType::BF16 | DType::F32),
            "Qwen3-VL text encoder requires BF16 or F32 weights, got {:?}",
            vb.dtype()
        );

        let text_config = config.text_config;
        let vb = vb.pp("model").pp("language_model");
        let embed_tokens = embedding(
            text_config.vocab_size,
            text_config.hidden_size,
            vb.pp("embed_tokens"),
        )?;
        let rotary = Arc::new(RotaryEmbedding::new(&text_config, vb.device(), vb.dtype())?);
        let mut layers = Vec::with_capacity(text_config.num_hidden_layers);
        for index in 0..text_config.num_hidden_layers {
            layers.push(DecoderLayer::load(&text_config, vb.pp("layers").pp(index))?);
        }
        Ok(Self {
            config: text_config,
            embed_tokens,
            layers,
            rotary,
        })
    }

    /// Encode `[batch, sequence]` token IDs into `[batch, sequence, 4096]`
    /// pre-final-norm hidden states.
    pub(crate) fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (batch, seq_len) = input_ids.dims2()?;
        if batch == 0 || seq_len == 0 {
            candle_core::bail!("Qwen3-VL text encoder requires non-empty batch and sequence")
        }
        if !matches!(input_ids.dtype(), DType::U8 | DType::U32 | DType::I64) {
            candle_core::bail!(
                "Qwen3-VL text token IDs must use u8, u32, or i64, got {:?}",
                input_ids.dtype()
            )
        }
        if seq_len > self.config.max_position_embeddings {
            candle_core::bail!(
                "Qwen3-VL text sequence length {seq_len} exceeds max_position_embeddings {}",
                self.config.max_position_embeddings
            )
        }

        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let (cos, sin) = self.rotary.table(seq_len)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &self.config, &cos, &sin)?;
        }
        // Intentionally return the pre-final-RMSNorm state.
        Ok(hidden)
    }
}

/// Match the BF16 products and add in Qwen3-VL's eager rotary expression.
/// Each tensor operation stores a BF16 result; the F32 path stays unchanged.
fn text_rope(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    if input.dtype() != DType::BF16 {
        return candle_nn::rotary_emb::rope(input, cos, sin);
    }
    let (_, _, seq, width) = input.dims4()?;
    if width == 0
        || !width.is_multiple_of(2)
        || cos.dims() != [seq, width / 2]
        || sin.dims() != cos.dims()
        || cos.dtype() != DType::BF16
        || sin.dtype() != DType::BF16
        || !input.device().same_device(cos.device())
        || !input.device().same_device(sin.device())
    {
        candle_core::bail!("BF16 text rotary requires matching sequence/half-width tables")
    }
    let half = width / 2;
    let cos = Tensor::cat(&[cos, cos], 1)?.reshape((1, 1, seq, width))?;
    let sin = Tensor::cat(&[sin, sin], 1)?.reshape((1, 1, seq, width))?;
    let rotated = Tensor::cat(
        &[
            &input.narrow(3, half, half)?.neg()?,
            &input.narrow(3, 0, half)?,
        ],
        3,
    )?;
    input
        .broadcast_mul(&cos)?
        .add(&rotated.broadcast_mul(&sin)?)
}

fn text_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    if q.dtype() != DType::BF16 || !q.device().is_metal() {
        return attention(q, k, v, true, 0);
    }
    bf16_text_attention(q, k, v, 32)
}

/// The reference uses eager BF16 score/scale/mask operations, F32 softmax,
/// then BF16 probabilities and the value product. Keep each softmax row whole.
/// Keep full query rows when one score plane fits. Tile heads to bound scores
/// to four million values; larger queries retain the bounded query fallback.
fn bf16_text_attention(q: &Tensor, k: &Tensor, v: &Tensor, head_limit: usize) -> Result<Tensor> {
    const MAX_SCORE_VALUES: usize = 4 * 1024 * 1024;
    let (batch, heads, seq, width) = q.dims4()?;
    if batch == 0
        || heads == 0
        || seq == 0
        || width == 0
        || seq > MAX_CONTEXT
        || head_limit == 0
        || head_limit > 32
        || k.dims() != q.dims()
        || v.dims() != q.dims()
        || q.dtype() != DType::BF16
        || k.dtype() != DType::BF16
        || v.dtype() != DType::BF16
        || !q.device().is_metal()
        || !q.device().same_device(k.device())
        || !q.device().same_device(v.device())
    {
        candle_core::bail!("BF16 text attention requires matching nonempty B/H/S/D tensors")
    }
    let full_plane = seq.checked_mul(seq);
    let (head_tile, query_tile) = match full_plane {
        Some(plane) if plane <= MAX_SCORE_VALUES => {
            (heads.min(head_limit).min(MAX_SCORE_VALUES / plane), seq)
        }
        _ => {
            let heads = heads.min(head_limit).min(MAX_SCORE_VALUES / seq);
            (heads, 32.min(MAX_SCORE_VALUES / (heads * seq)))
        }
    };
    let scale = 1.0 / (width as f64).sqrt();
    let mut batches = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut head_outputs = Vec::new();
        for h in (0..heads).step_by(head_tile) {
            let nh = head_tile.min(heads - h);
            let q = q.narrow(0, b, 1)?.narrow(1, h, nh)?;
            let k = k.narrow(0, b, 1)?.narrow(1, h, nh)?.contiguous()?;
            let v = v.narrow(0, b, 1)?.narrow(1, h, nh)?.contiguous()?;
            let mut outputs = Vec::new();
            for start in (0..seq).step_by(query_tile) {
                let count = query_tile.min(seq - start);
                let query = q.narrow(2, start, count)?.contiguous()?;
                let scores = (bf16_matmul(&query, &k, true)? * scale)?;
                let mask = (start..start + count)
                    .flat_map(|row| {
                        (0..seq).map(move |col| {
                            if col <= row {
                                0.0_f32
                            } else {
                                f32::from_bits(0xff7f_0000)
                            }
                        })
                    })
                    .collect::<Vec<_>>();
                let mask = Tensor::from_vec(mask, (1, 1, count, seq), q.device())?
                    .to_dtype(DType::BF16)?;
                let scores = scores.broadcast_add(&mask)?;
                let probs = bf16_softmax(&scores)?;
                outputs.push(bf16_matmul(&probs, &v, false)?);
            }
            head_outputs.push(Tensor::cat(&outputs, 2)?);
        }
        batches.push(Tensor::cat(&head_outputs, 1)?);
    }
    Tensor::cat(&batches, 0)
}

fn bf16_matmul(a: &Tensor, b: &Tensor, transpose_rhs: bool) -> Result<Tensor> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let Some(out) = text_linear_mps::try_matmul(a, b, transpose_rhs)? {
        return Ok(out);
    }
    a.matmul(&if transpose_rhs {
        b.transpose(b.rank() - 2, b.rank() - 1)?
    } else {
        b.clone()
    })
}
fn bf16_softmax(scores: &Tensor) -> Result<Tensor> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let Some(out) = text_softmax_metal::try_softmax(scores)? {
        return Ok(out);
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let Some(out) = text_linear_mps::try_softmax(scores)? {
        return Ok(out);
    }
    candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(DType::BF16)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) fn try_bf16_linear(input: &Tensor, weight: &Tensor) -> Result<Option<Tensor>> {
    text_linear_mps::try_linear(input, weight)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) fn try_bf16_gate_silu(input: &Tensor) -> Result<Option<Tensor>> {
    text_silu_metal::try_silu_chunked(input)
}

fn text_linear(linear: &Linear, input: &Tensor) -> Result<Tensor> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if linear.bias().is_none()
        && let Some(output) = text_linear_mps::try_linear(input, linear.weight())?
    {
        return Ok(output);
    }
    linear.forward(input)
}

pub(super) fn text_rms_norm(norm: &RmsNorm, input: &Tensor) -> Result<Tensor> {
    if input.dtype() == DType::BF16 {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if let Some(output) = text_norm_metal::try_norm(input, norm.weight(), norm.eps())? {
            return Ok(output);
        }
        // Qwen3-VL normalizes in F32, then casts before multiplying the BF16
        // weight. The fused Metal kernel instead rounds the inverse scale.
        norm.forward_diff(input)
    } else {
        norm.forward(input)
    }
}

fn normalize_heads(
    norm: &RmsNorm,
    input: &Tensor,
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<Tensor> {
    text_rms_norm(norm, &input.flatten(0, 2)?)?.reshape((batch, heads, seq_len, head_dim))
}

fn repeat_kv(input: &Tensor, repeats: usize) -> Result<Tensor> {
    let (batch, kv_heads, seq_len, head_dim) = input.dims4()?;
    if repeats == 0 {
        candle_core::bail!("Qwen3-VL text KV repeat factor must be non-zero")
    }
    input
        .unsqueeze(2)?
        .expand((batch, kv_heads, repeats, seq_len, head_dim))?
        .reshape((batch, kv_heads * repeats, seq_len, head_dim))?
        .contiguous()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use serde_json::json;

    #[cfg(target_os = "macos")]
    #[test]
    fn bf16_frequency_rounding_matches_f32_contract() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let rotary = RotaryEmbedding {
            head_dim: 128,
            theta: 5_000_000.0,
            max_position_embeddings: 4096,
            device,
            dtype: DType::BF16,
        };
        let (cos, sin) = rotary.table(34)?;
        let cv = cos.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let sv = sin.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        // Native F32 frequency construction crosses these two BF16 bins;
        // constructing the frequency in F64 does not.
        assert_eq!(cv[10][1].to_bits(), (-0.004_333_496_f32).to_bits());
        assert_eq!(sv[20][1].to_bits(), (-0.008_666_992_f32).to_bits());
        for i in 0..64 {
            let actual = 5_000_000_f32.powf(2.0 * i as f32 / 128.0).recip();
            let exact = 5_000_000_f64.powf(2.0 * i as f64 / 128.0).recip();
            assert!((f64::from(actual) - exact).abs() <= 2.0 * f64::from(f32::EPSILON) * exact);
        }
        assert!(rotary.table(4097).is_err());
        Ok(())
    }

    #[test]
    fn bf16_text_norm_keeps_inverse_scale_in_f32() -> Result<()> {
        let devices = std::iter::once(Device::Cpu);
        #[cfg(target_os = "macos")]
        let devices = devices.chain(crate::test_metal_device());
        for device in devices {
            // Cover the actual head and decoder widths with an independent
            // scalar reference. Every input and weight is exact in BF16.
            for width in [128, 4096] {
                let values = [0.125_f32, 0.375, 0.625, 0.875].repeat(width / 4);
                let weights = [0.75_f32, 1.0, 1.25, 1.5].repeat(width / 4);
                let eps = 1e-6_f32;
                let mean_square = values.iter().map(|x| x * x).sum::<f32>() / width as f32;
                let inverse = (mean_square + eps).sqrt().recip();
                let round = |x| half::bf16::from_f32(x).to_f32();
                let expected = values
                    .iter()
                    .zip(&weights)
                    .map(|(&x, &weight)| round(round(x * inverse) * weight))
                    .collect::<Vec<_>>();
                let early_rounded = values
                    .iter()
                    .zip(&weights)
                    .map(|(&x, &weight)| round(round(x * round(inverse)) * weight))
                    .collect::<Vec<_>>();
                assert_ne!(expected, early_rounded);
                let input = Tensor::from_slice(&values, (1, 1, 1, width), &device)?
                    .to_dtype(DType::BF16)?;
                let weight = Tensor::from_slice(&weights, width, &device)?.to_dtype(DType::BF16)?;
                let norm = RmsNorm::new(weight, f64::from(eps));
                for output in [
                    text_rms_norm(&norm, &input)?,
                    normalize_heads(&norm, &input, 1, 1, 1, width)?,
                ] {
                    let actual = output
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    assert_eq!(
                        actual, expected,
                        "BF16 text RMSNorm width={width} {device:?}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn bf16_text_rotary_rounds_products_before_add() -> Result<()> {
        let device = Device::Cpu;
        let input = Tensor::from_vec(
            vec![0.51171875_f32, -1.234375, 1.9921875, 0.2734375],
            (1, 1, 1, 4),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let cos = Tensor::from_vec(vec![0.6796875_f32, 0.921875], (1, 2), &device)?
            .to_dtype(DType::BF16)?;
        let sin = Tensor::from_vec(vec![0.73828125_f32, -0.3828125], (1, 2), &device)?
            .to_dtype(DType::BF16)?;
        let x = input
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let c = cos.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let si = sin.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let round = |v: f32| half::bf16::from_f32(v).to_f32();
        let rotated = [-x[2], -x[3], x[0], x[1]];
        let expected = (0..4)
            .map(|i| round(round(x[i] * c[i % 2]) + round(rotated[i] * si[i % 2])))
            .collect::<Vec<_>>();
        assert_eq!(
            text_rope(&input, &cos, &sin)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            expected
        );
        assert!(text_rope(&input, &cos.narrow(1, 0, 1)?, &sin).is_err());
        Ok(())
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn bf16_text_attention_matches_f64_causal_store_boundaries() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let n = 5;
        let width = 4;
        let values = |m: usize| {
            (0..n * width)
                .map(|i| ((i * m % 17) as f32 - 8.0) / 16.0)
                .collect::<Vec<_>>()
        };
        let q = values(3);
        let k = values(7);
        let v = values(11);
        let round = |x: f64| f64::from(half::bf16::from_f32(x as f32).to_f32());
        let mut expected = Vec::new();
        for i in 0..n {
            let scores = (0..=i)
                .map(|j| {
                    round(
                        round(
                            (0..width)
                                .map(|z| f64::from(q[i * width + z]) * f64::from(k[j * width + z]))
                                .sum(),
                        ) / 2.0,
                    )
                })
                .collect::<Vec<_>>();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exp = scores.iter().map(|x| (x - max).exp()).collect::<Vec<_>>();
            let sum = exp.iter().sum::<f64>();
            let probs = exp.iter().map(|x| round(x / sum)).collect::<Vec<_>>();
            for z in 0..width {
                expected.push(round(
                    (0..=i)
                        .map(|j| probs[j] * f64::from(v[j * width + z]))
                        .sum(),
                ) as f32);
            }
        }
        let make =
            |x: &[f32]| Tensor::from_slice(x, (1, 1, n, width), &device)?.to_dtype(DType::BF16);
        let q = make(&q)?;
        let k = make(&k)?;
        let v = make(&v)?;
        for tile in [1, 3, 32] {
            assert_eq!(
                bf16_text_attention(&q, &k, &v, tile)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                expected
            );
        }
        for tile in [0, 33] {
            assert!(bf16_text_attention(&q, &k, &v, tile).is_err());
        }
        assert!(text_attention(&q, &k.narrow(2, 0, 4)?, &v).is_err());
        let f = q.to_dtype(DType::F32)?;
        assert_eq!(
            text_attention(&f, &f, &f)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            attention(&f, &f, &f, true, 0)?
                .flatten_all()?
                .to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn bf16_text_metal_matches_eager_fixture_across_query_tiles_and_views() -> Result<()> {
        use sha2::{Digest, Sha256};
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let seq = 34;
        let values = |n: usize, m: usize, divisor: f32| {
            (0..n)
                .map(|i| ((i * m % 257) as f32 - 128.0) / divisor)
                .collect::<Vec<_>>()
        };
        let make = |m, divisor| {
            Tensor::from_vec(values(2 * seq * 128, m, divisor), (1, 2, seq, 128), &device)?
                .to_dtype(DType::BF16)
        };
        let q = make(13, 64.0)?;
        let k = make(31, 64.0)?;
        let v = make(7, 128.0)?;
        let cos = Tensor::from_vec(values(seq * 64, 11, 128.0), (seq, 64), &device)?
            .to_dtype(DType::BF16)?;
        let sin = Tensor::from_vec(values(seq * 64, 23, 128.0), (seq, 64), &device)?
            .to_dtype(DType::BF16)?;
        let bits = |t: &Tensor| -> Result<Vec<u8>> {
            Ok(t.to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        };
        // Synthetic PyTorch MPS BF16 eager outputs, Transformers 5.22.1.
        assert_eq!(
            format!("{:x}", Sha256::digest(bits(&text_rope(&q, &cos, &sin)?)?)),
            "5240783126826592f90a548a2db476e34256db68c6cc4f39ed8dbd064bdcd296"
        );
        let expected = bits(&text_attention(&q, &k, &v)?)?;
        assert_eq!(
            format!("{:x}", Sha256::digest(&expected)),
            "c961c6d1e8e5d52a065e41d9e4d2080d7cb89be2e71f3d88de425d6b31f1b30c"
        );
        for tile in [1, 5, 7, 32] {
            assert_eq!(bits(&bf16_text_attention(&q, &k, &v, tile)?)?, expected);
        }
        let two_with_offset = |t: &Tensor| Tensor::cat(&[t, t, t], 0)?.narrow(0, 1, 2);
        let output = text_attention(
            &two_with_offset(&q)?,
            &two_with_offset(&k)?,
            &two_with_offset(&v)?,
        )?;
        assert_eq!(bits(&output)?, [expected.clone(), expected].concat());
        Ok(())
    }

    fn config() -> TextConfig {
        serde_json::from_value(json!({
            "attention_bias": false,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "max_position_embeddings": 262144,
            "num_attention_heads": 32,
            "num_hidden_layers": 36,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-6,
            "rope_theta": 5000000,
            "vocab_size": 151936,
            "rope_scaling": {
                "mrope_interleaved": true,
                "mrope_section": [24, 20, 20],
                "rope_type": "default"
            },
            "use_cache": true
        }))
        .expect("Qwen3-VL text config")
    }

    #[test]
    fn released_text_architecture_validates_and_outer_wrapper_deserializes() {
        let config: TextEncoderConfig = serde_json::from_value(json!({
            "text_config": {
                "attention_bias": false,
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "max_position_embeddings": 262144,
                "num_attention_heads": 32,
                "num_hidden_layers": 36,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-6,
                "rope_theta": 5000000,
                "vocab_size": 151936
            },
            "vision_config": {"depth": 27},
            "model_type": "qwen3_vl"
        }))
        .expect("outer Qwen3-VL config");
        config.text_config.validate().unwrap();
    }

    #[test]
    fn architecture_mismatch_is_rejected() {
        let mut config = config();
        config.num_hidden_layers = MAX_LAYERS + 1;
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("layer count"));
    }

    #[test]
    fn normalization_epsilon_must_survive_f32_conversion() {
        for eps in [1e-100, 1e300] {
            let mut config = config();
            config.rms_norm_eps = eps;
            assert!(
                config.validate().is_err(),
                "accepted invalid F32 epsilon {eps}"
            );
        }
        config().validate().unwrap();
    }

    #[test]
    fn unsupported_rope_scaling_is_rejected_instead_of_ignored() {
        let mut config = config();
        config.rope_scaling = Some(RopeScalingConfig {
            rope_type: Some("yarn".into()),
            mrope_interleaved: None,
            mrope_section: None,
        });
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("default rope scaling"));
    }

    #[test]
    fn text_rope_is_standard_half_rotation() -> Result<()> {
        let config = config();
        let rope = RotaryEmbedding::new(&config, &Device::Cpu, DType::F32)?;
        let (cos, sin) = rope.table(2)?;
        let cos = cos.to_vec2::<f32>()?;
        let sin = sin.to_vec2::<f32>()?;
        assert!((cos[0][0] - 1.0).abs() < 1e-6);
        assert!((sin[1][0] - 1.0_f32.sin()).abs() < 1e-6);
        let expected_phase = 1.0 / config.rope_theta.powf(2.0 / config.head_dim as f64);
        assert!((cos[1][1] - expected_phase.cos() as f32).abs() < 1e-6);
        assert!((sin[1][1] - expected_phase.sin() as f32).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn repeat_kv_has_expected_gqa_shape() -> Result<()> {
        let input = Tensor::zeros((2, 8, 3, 128), DType::F32, &Device::Cpu)?;
        let repeated = repeat_kv(&input, 4)?;
        assert_eq!(repeated.dims(), &[2, 32, 3, 128]);
        Ok(())
    }

    #[test]
    fn tiny_zero_weight_encoder_has_pre_norm_shape() -> anyhow::Result<()> {
        let tiny = TextConfig {
            attention_bias: false,
            head_dim: 4,
            hidden_act: Activation::Silu,
            hidden_size: 8,
            intermediate_size: 16,
            max_position_embeddings: 8,
            num_attention_heads: 2,
            num_hidden_layers: 1,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_scaling: None,
            rope_theta: EXPECTED_ROPE_THETA,
            vocab_size: 8,
        };
        let encoder = TextEncoder::load(
            VarBuilder::zeros(DType::F32, &Device::Cpu),
            TextEncoderConfig { text_config: tiny },
        )?;
        let input_ids = Tensor::from_vec(vec![0_u32, 1, 2], (1, 3), &Device::Cpu)?;
        let output = encoder.forward(&input_ids)?;
        assert_eq!(output.dims(), &[1, 3, 8]);
        Ok(())
    }
}
