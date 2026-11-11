//! Muse Spark Open text decoder for the canonical muse-spark GGUF layout.
//!
//! The converter folds centered norm offsets, QK and router scales, and the
//! learned residual gates into the GGUF tensors. This stage consumes those
//! folded values once. It does not treat the latent expert block as a dense MoE.
//!
//! Math reference: Hugging Face Transformers' modular_muse_spark.py.
//! GGUF tensor contract: ggml-org/llama.cpp PR 30616, revision
//! 335916ad3f0cab876a7a3cf16faaaaea28bf9a96.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::quantized::QTensor;
use candle_core::{D, DType, Device, IndexOp, Module, Tensor};

#[cfg(any(target_os = "macos", feature = "cuda"))]
use super::indexed_moe::{IndexedMoeMatMul, indexed_moe_available, silu_mul, weighted_route_sum};
use super::{
    DecodeSafeQMatMul, KvCache, LayerCache, ModelArchitecture, StageInput, StageOutput,
    with_kv_cache,
};
use crate::gguf_loader::{LayerRange, StageTensorLoader, StageTensors};
use crate::turboquant::TurboQuantConfig;

type Weights = HashMap<String, Arc<QTensor>>;

struct Norm {
    weight: Tensor,
    eps: f64,
}

impl Norm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        rms(input, self.eps)?
            .broadcast_mul(&self.weight)
            .map_err(Into::into)
    }

    fn input_scaled(&self, input: &Tensor) -> Result<Tensor> {
        rms(&input.broadcast_mul(&self.weight)?, self.eps)
    }
}

fn rms(input: &Tensor, eps: f64) -> Result<Tensor> {
    let input = input.to_dtype(DType::F32)?;
    let variance = input.sqr()?.mean_keepdim(D::Minus1)?;
    input
        .broadcast_div(&(variance + eps)?.sqrt()?)
        .map_err(Into::into)
}

fn load_vector(weights: &Weights, name: &str, size: usize, device: &Device) -> Result<Tensor> {
    let weight = weights
        .get(name)
        .with_context(|| format!("missing {name}"))?;
    anyhow::ensure!(
        weight.shape().dims() == [size],
        "invalid {name} shape {:?}; expected [{size}]",
        weight.shape()
    );
    let tensor = weight.dequantize(device)?.to_dtype(DType::F32)?;
    anyhow::ensure!(
        tensor
            .to_vec1::<f32>()?
            .iter()
            .all(|value| value.is_finite()),
        "non-finite {name}"
    );
    Ok(tensor)
}

fn load_norm(
    weights: &Weights,
    name: &str,
    size: usize,
    eps: f64,
    device: &Device,
) -> Result<Norm> {
    Ok(Norm {
        weight: load_vector(weights, name, size, device)?,
        eps,
    })
}

fn load_projection(
    weights: &Weights,
    name: &str,
    output: usize,
    input: usize,
    device: &Device,
) -> Result<DecodeSafeQMatMul> {
    let weight = weights
        .get(name)
        .with_context(|| format!("missing {name}"))?;
    anyhow::ensure!(
        weight.shape().dims() == [output, input],
        "invalid {name} shape {:?}; expected [{output}, {input}]",
        weight.shape()
    );
    Ok(DecodeSafeQMatMul::from_qtensor(Arc::clone(weight), device)?.with_matrix_decode())
}

struct Residual {
    alpha: Tensor,
    beta: Tensor,
}

impl Residual {
    fn load(weights: &Weights, prefix: &str, hidden: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            alpha: load_vector(weights, &format!("{prefix}_alpha.weight"), hidden, device)?,
            beta: load_vector(weights, &format!("{prefix}_beta.weight"), hidden, device)?,
        })
    }

    fn forward(&self, residual: &Tensor, branch: &Tensor) -> Result<Tensor> {
        (residual.broadcast_mul(&self.alpha)? + branch.broadcast_mul(&self.beta)?)
            .map_err(Into::into)
    }
}

struct Attention {
    q: DecodeSafeQMatMul,
    k: DecodeSafeQMatMul,
    v: DecodeSafeQMatMul,
    out: DecodeSafeQMatMul,
    gate: DecodeSafeQMatMul,
    q_norm: Norm,
    k_norm: Norm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    rope_theta: f64,
    window: Option<usize>,
    eps: f64,
    quant: Option<Arc<TurboQuantConfig>>,
}

impl Attention {
    fn forward(&self, input: &Tensor, cache: &mut Option<KvCache>, start: usize) -> Result<Tensor> {
        let (batch, seq, _) = input.dims3()?;
        let project = |op: &DecodeSafeQMatMul, heads: usize| -> Result<Tensor> {
            Ok(op
                .forward(input)?
                .reshape((batch, seq, heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let mut q = self.q_norm.forward(&project(&self.q, self.heads)?)?;
        let mut k = self.k_norm.forward(&project(&self.k, self.kv_heads)?)?;
        let v = project(&self.v, self.kv_heads)?;
        if self.window.is_some() {
            // GGUF Q/K are already permuted into interleaved rotary order.
            // Build only this pass's positions, not a million-token table.
            let (cos, sin) =
                rope_range(self.rope_dim, self.rope_theta, start, seq, input.device())?;
            q = rotary(&q, self.rope_dim, &cos, &sin)?;
            k = rotary(&k, self.rope_dim, &cos, &sin)?;
        }
        let (q, mut k, mut v, rotation) =
            super::prepare_quantized_attention(cache, q, k, v, self.quant.as_deref())?;
        if seq == 1
            && let Some(window) = self.window
        {
            let len = k.dim(2)?;
            if len > window {
                k = k.narrow(2, len - window, window)?;
                v = v.narrow(2, len - window, window)?;
            }
        }
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let fused = match self.window {
            Some(window) if seq > 1 => {
                super::try_fused_f16_sdpa_windowed(&q, &k, &v, window, scale)?
            }
            _ => super::try_fused_f16_sdpa(&q, &k, &v, seq > 1, scale)?,
        };
        let output = if let Some(output) = fused {
            output
        } else {
            let repeat = self.heads / self.kv_heads;
            let k = repeat_kv(&k.to_dtype(DType::F32)?, repeat)?;
            let v = repeat_kv(&v.to_dtype(DType::F32)?, repeat)?;
            let scores = (q.to_dtype(DType::F32)?.matmul(&k.transpose(2, 3)?)? * scale as f64)?;
            let kv_len = k.dim(2)?;
            let scores = if seq > 1 {
                let mask = attention_mask(seq, kv_len, self.window, input.device())?;
                scores.broadcast_add(&mask)?
            } else {
                scores
            };
            candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?
        };
        let output = super::restore_quantized_attention_output(output, rotation)?;
        // Normalize each value head before flattening and applying its gate.
        let output = rms(&output, self.eps)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq, self.heads * self.head_dim))?;
        let gate = candle_nn::Activation::Sigmoid.forward(&self.gate.forward(input)?)?;
        self.out.forward(&(output * gate)?)
    }
}

fn rope_range(
    dim: usize,
    theta: f64,
    start: usize,
    seq: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let end = start
        .checked_add(seq)
        .context("Muse Spark rotary position overflow")?;
    anyhow::ensure!(
        end <= u32::MAX as usize,
        "Muse Spark rotary position is too large"
    );
    let inv = (0..dim / 2)
        .map(|i| theta.powf(-2.0 * i as f64 / dim as f64) as f32)
        .collect::<Vec<_>>();
    let inv = Tensor::new(inv.as_slice(), device)?.reshape((1, dim / 2))?;
    let pos = Tensor::arange(start as u32, end as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((seq, 1))?;
    let angles = pos.matmul(&inv)?;
    Ok((angles.cos()?, angles.sin()?))
}

fn rotary(input: &Tensor, dim: usize, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let width = input.dim(D::Minus1)?;
    let rotated =
        candle_nn::rotary_emb::rope_i(&input.narrow(D::Minus1, 0, dim)?.contiguous()?, cos, sin)?;
    if dim == width {
        return Ok(rotated);
    }
    Tensor::cat(
        &[&rotated, &input.narrow(D::Minus1, dim, width - dim)?],
        D::Minus1,
    )
    .map_err(Into::into)
}

fn repeat_kv(input: &Tensor, repeat: usize) -> Result<Tensor> {
    if repeat == 1 {
        return Ok(input.clone());
    }
    let (batch, heads, seq, dim) = input.dims4()?;
    input
        .unsqueeze(2)?
        .broadcast_as((batch, heads, repeat, seq, dim))?
        .contiguous()?
        .reshape((batch, heads * repeat, seq, dim))
        .map_err(Into::into)
}

fn attention_mask(
    seq: usize,
    kv_len: usize,
    window: Option<usize>,
    device: &Device,
) -> Result<Tensor> {
    let past = kv_len
        .checked_sub(seq)
        .context("Muse Spark KV cache is shorter than this pass")?;
    let mut mask = Vec::with_capacity(
        seq.checked_mul(kv_len)
            .context("Muse Spark attention mask size overflow")?,
    );
    for query in 0..seq {
        let position = past + query;
        for key in 0..kv_len {
            mask.push(
                if key > position
                    || window.is_some_and(|window| position - key.min(position) >= window)
                {
                    f32::NEG_INFINITY
                } else {
                    0.0
                },
            );
        }
    }
    Tensor::from_vec(mask, (1, 1, seq, kv_len), device).map_err(Into::into)
}

struct Expert {
    gate: DecodeSafeQMatMul,
    up: DecodeSafeQMatMul,
    down: DecodeSafeQMatMul,
}

#[cfg(any(target_os = "macos", feature = "cuda"))]
struct IndexedExperts {
    gate: IndexedMoeMatMul,
    up: IndexedMoeMatMul,
    down: IndexedMoeMatMul,
}

enum Experts {
    Split(Vec<Expert>),
    #[cfg(any(target_os = "macos", feature = "cuda"))]
    Indexed(Box<IndexedExperts>),
}

struct Moe {
    router: Tensor,
    correction: Tensor,
    down: DecodeSafeQMatMul,
    up: DecodeSafeQMatMul,
    pre_norm: Norm,
    post_norm: Norm,
    experts: Experts,
    latent: usize,
    active: usize,
    scale: f64,
}

impl Moe {
    fn forward(&self, input: &Tensor, prefill: bool) -> Result<Tensor> {
        let (batch, seq, hidden) = input.dims3()?;
        let rows = batch
            .checked_mul(seq)
            .context("Muse Spark token count overflow")?;
        let flat = input.reshape((rows, hidden))?.to_dtype(DType::F32)?;
        // GGUF folds output_multiplier into this fp32 router weight.
        let logits = flat.matmul(&self.router.t()?)?;
        let probabilities = candle_nn::Activation::Sigmoid.forward(&logits)?;
        let scores = probabilities.broadcast_add(&self.correction)?;
        let latent = self.pre_norm.forward(&self.down.forward(&flat)?)?;
        let output = match &self.experts {
            #[cfg(any(target_os = "macos", feature = "cuda"))]
            Experts::Indexed(experts) => {
                let ids = scores
                    .contiguous()?
                    .arg_sort_last_dim(false)?
                    .narrow(D::Minus1, 0, self.active)?
                    .contiguous()?;
                let weights = probabilities.gather(&ids, D::Minus1)?;
                let weights = weights.broadcast_div(&(weights.sum_keepdim(D::Minus1)? + 1e-15)?)?;
                let weights = (weights * self.scale)?;
                let chunk = (4096 / self.active).max(1);
                let mut outputs = Vec::new();
                for start in (0..rows).step_by(chunk) {
                    let len = chunk.min(rows - start);
                    let ids = ids.narrow(0, start, len)?;
                    let project = |op: &IndexedMoeMatMul, value: &Tensor| {
                        if prefill {
                            op.forward_prefill(value, &ids)
                        } else {
                            op.forward(value, &ids)
                        }
                    };
                    let x = latent
                        .narrow(0, start, len)?
                        .reshape((len, 1, self.latent))?;
                    let activated =
                        silu_mul(&project(&experts.gate, &x)?, &project(&experts.up, &x)?)?;
                    let output = self
                        .post_norm
                        .input_scaled(&project(&experts.down, &activated)?)?;
                    outputs.push(weighted_route_sum(
                        &output,
                        &weights.narrow(0, start, len)?,
                    )?);
                }
                Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 0)?
            }
            Experts::Split(experts) => {
                let _ = prefill;
                let scores = scores.to_vec2::<f32>()?;
                let probabilities = probabilities.to_vec2::<f32>()?;
                let mut assignments = vec![Vec::<(u32, f32)>::new(); experts.len()];
                for (row, scores) in scores.iter().enumerate() {
                    anyhow::ensure!(
                        scores.iter().all(|value| value.is_finite()),
                        "non-finite Muse Spark expert scores"
                    );
                    let mut selected = scores.iter().copied().enumerate().collect::<Vec<_>>();
                    selected.sort_by(|a, b| b.1.total_cmp(&a.1));
                    selected.truncate(self.active);
                    let sum = selected
                        .iter()
                        .map(|(id, _)| probabilities[row][*id])
                        .sum::<f32>()
                        + 1e-15;
                    for (id, _) in selected {
                        assignments[id]
                            .push((row as u32, probabilities[row][id] / sum * self.scale as f32));
                    }
                }
                let mut routed = Vec::new();
                let mut row_ids = Vec::new();
                for (expert, assignments) in experts.iter().zip(assignments) {
                    if assignments.is_empty() {
                        continue;
                    }
                    let ids = assignments.iter().map(|(row, _)| *row).collect::<Vec<_>>();
                    let ids_tensor = Tensor::new(ids.as_slice(), input.device())?;
                    let x = latent.index_select(&ids_tensor, 0)?;
                    let activated = (candle_nn::Activation::Silu
                        .forward(&expert.gate.forward(&x)?)?
                        * expert.up.forward(&x)?)?;
                    let output = self
                        .post_norm
                        .input_scaled(&expert.down.forward(&activated)?)?;
                    let weights = assignments
                        .iter()
                        .map(|(_, weight)| *weight)
                        .collect::<Vec<_>>();
                    let weights = Tensor::new(weights.as_slice(), input.device())?
                        .reshape((weights.len(), 1))?;
                    routed.push(output.broadcast_mul(&weights)?);
                    row_ids.extend(ids);
                }
                let output = Tensor::cat(&routed.iter().collect::<Vec<_>>(), 0)?;
                let ids = Tensor::new(row_ids.as_slice(), input.device())?;
                Tensor::zeros((rows, self.latent), DType::F32, input.device())?
                    .index_add(&ids, &output, 0)?
            }
        };
        self.up
            .forward(&output)?
            .reshape((batch, seq, hidden))
            .map_err(Into::into)
    }
}

fn split_projection(
    packed: &QTensor,
    count: usize,
    output: usize,
    input: usize,
    device: &Device,
) -> Result<Vec<DecodeSafeQMatMul>> {
    anyhow::ensure!(
        packed.shape().dims() == [count, output, input],
        "Muse Spark packed expert shape mismatch"
    );
    let elements = output
        .checked_mul(input)
        .context("Muse Spark expert size overflow")?;
    let dtype = packed.dtype();
    anyhow::ensure!(
        elements.is_multiple_of(dtype.block_size()),
        "Muse Spark expert row is not quantization aligned"
    );
    let bytes = (elements / dtype.block_size())
        .checked_mul(dtype.type_size())
        .context("Muse Spark expert byte size overflow")?;
    let raw = packed.data()?;
    anyhow::ensure!(
        raw.len()
            == count
                .checked_mul(bytes)
                .context("Muse Spark expert storage overflow")?,
        "Muse Spark expert storage length mismatch"
    );
    let mut result = Vec::with_capacity(count);
    for index in 0..count {
        let tensor = candle_core::quantized::ggml_file::qtensor_from_ggml(
            dtype,
            &raw[index * bytes..(index + 1) * bytes],
            vec![output, input],
            device,
        )?;
        result
            .push(DecodeSafeQMatMul::from_qtensor(Arc::new(tensor), device)?.with_matrix_decode());
    }
    Ok(result)
}

fn load_experts(
    weights: &mut Weights,
    prefix: &str,
    count: usize,
    intermediate: usize,
    latent: usize,
    device: &Device,
) -> Result<Experts> {
    let take = |weights: &mut Weights, suffix: &str| {
        weights
            .remove(&format!("{prefix}.{suffix}.weight"))
            .with_context(|| format!("missing {prefix}.{suffix}.weight"))
    };
    let gate = take(weights, "ffn_gate_exps")?;
    let up = take(weights, "ffn_up_exps")?;
    let down = take(weights, "ffn_down_exps")?;
    #[cfg(any(target_os = "macos", feature = "cuda"))]
    // Metal's indexed matrix kernel reads full 32-wide inner tiles. Floating
    // tensors permit smaller widths, so keep those on the exact split path.
    if (!device.is_metal() || (latent.is_multiple_of(32) && intermediate.is_multiple_of(32)))
        && [&gate, &up, &down]
            .iter()
            .all(|weight| indexed_moe_available(device, weight.dtype()))
    {
        return Ok(Experts::Indexed(Box::new(IndexedExperts {
            gate: IndexedMoeMatMul::from_qtensor(&gate, count, intermediate, latent, device)?,
            up: IndexedMoeMatMul::from_qtensor(&up, count, intermediate, latent, device)?,
            down: IndexedMoeMatMul::from_qtensor(&down, count, latent, intermediate, device)?,
        })));
    }
    let gates = split_projection(&gate, count, intermediate, latent, device)?;
    let ups = split_projection(&up, count, intermediate, latent, device)?;
    let downs = split_projection(&down, count, latent, intermediate, device)?;
    Ok(Experts::Split(
        gates
            .into_iter()
            .zip(ups)
            .zip(downs)
            .map(|((gate, up), down)| Expert { gate, up, down })
            .collect(),
    ))
}

struct Block {
    attn_norm: Norm,
    attention: Attention,
    attn_residual: Residual,
    ffn_norm: Norm,
    moe: Moe,
    ffn_post_norm: Norm,
    ffn_residual: Residual,
    post_eps: f64,
}

/// A head, middle, tail, or solo Muse Spark Open text stage.
pub struct MuseSparkStage {
    embedding: Option<Arc<QTensor>>,
    blocks: Vec<Block>,
    output_norm: Option<Norm>,
    output: Option<DecodeSafeQMatMul>,
    head: bool,
    tail: bool,
    hidden: usize,
    vocab: usize,
    eps: f64,
    logit_scale: f64,
    softcap: f64,
    device: Device,
}

impl MuseSparkStage {
    /// Load one layer's source tensors at a time and release them after upload.
    pub fn from_streaming_gguf(
        path: &Path,
        range: LayerRange,
        device: &Device,
        quant: Option<Arc<TurboQuantConfig>>,
    ) -> Result<Self> {
        range.validate()?;
        let mut loader = StageTensorLoader::open(path)?;
        let md = loader.metadata().clone();
        anyhow::ensure!(
            md.architecture == "muse-spark" && range.total_layers == md.num_layers,
            "not a matching Muse Spark stage"
        );
        let mut result = Self {
            embedding: None,
            blocks: Vec::with_capacity(range.layer_count()),
            output_norm: None,
            output: None,
            head: range.is_head(),
            tail: range.is_tail(),
            hidden: md.hidden_dim,
            vocab: md.vocab_size,
            eps: md.rms_norm_eps,
            logit_scale: md.logit_scale,
            softcap: md.final_logit_softcapping,
            device: device.clone(),
        };
        for layer in range.start..range.end {
            let tensors = loader
                .load_range(
                    LayerRange::try_new(layer, layer + 1, range.total_layers)?,
                    device,
                )
                .with_context(|| format!("failed to stream Muse Spark layer {layer}"))?;
            let mut stage = Self::from_stage_tensors(tensors, quant.clone())?;
            if stage.embedding.is_some() {
                result.embedding = stage.embedding.take();
            }
            result.blocks.append(&mut stage.blocks);
            if stage.output_norm.is_some() {
                result.output_norm = stage.output_norm.take();
            }
            if stage.output.is_some() {
                result.output = stage.output.take();
            }
            drop(stage);
            device
                .synchronize()
                .with_context(|| format!("failed to finish streaming Muse Spark layer {layer}"))?;
        }
        Ok(result)
    }

    /// Load canonical, converter-folded GGUF tensors for this layer range.
    pub fn from_stage_tensors(
        stage: StageTensors,
        quant: Option<Arc<TurboQuantConfig>>,
    ) -> Result<Self> {
        stage.range.validate()?;
        let StageTensors {
            range,
            metadata: md,
            mut tensors,
            device,
            ..
        } = stage;
        anyhow::ensure!(
            md.architecture == "muse-spark",
            "Muse Spark needs muse-spark GGUF tensors"
        );
        anyhow::ensure!(
            md.hidden_dim > 0 && md.moe_latent_size > 0 && md.expert_ffn_dim > 0,
            "invalid Muse Spark hidden dimensions"
        );
        anyhow::ensure!(
            md.num_attention_heads > 0
                && md.num_kv_heads > 0
                && md.num_attention_heads.is_multiple_of(md.num_kv_heads),
            "invalid Muse Spark head counts"
        );
        anyhow::ensure!(
            md.head_dim > 0 && md.sliding_window > 0,
            "invalid Muse Spark attention dimensions"
        );
        anyhow::ensure!(
            md.expert_count > 0
                && md.expert_used_count > 0
                && md.expert_used_count <= md.expert_count,
            "invalid Muse Spark expert counts"
        );
        anyhow::ensure!(
            md.expert_gating_func == 2 && md.expert_weights_norm,
            "Muse Spark requires normalized sigmoid expert routing"
        );
        anyhow::ensure!(
            md.sliding_window_pattern.len() == md.num_layers && range.total_layers == md.num_layers,
            "Muse Spark sliding layer pattern length mismatch"
        );
        anyhow::ensure!(
            md.rms_norm_eps.is_finite()
                && md.rms_norm_eps > 0.0
                && md.post_norm_eps.is_finite()
                && md.post_norm_eps > 0.0,
            "invalid Muse Spark normalization epsilon"
        );
        anyhow::ensure!(
            md.expert_weights_scale.is_finite()
                && md.expert_weights_scale > 0.0
                && md.logit_scale.is_finite()
                && md.logit_scale > 0.0
                && md.final_logit_softcapping.is_finite()
                && md.final_logit_softcapping > 0.0,
            "invalid Muse Spark output scaling"
        );
        let rope_dim = if md.rope_dimension_count == 0 {
            md.head_dim
        } else {
            md.rope_dimension_count
        };
        anyhow::ensure!(
            rope_dim > 0
                && rope_dim <= md.head_dim
                && rope_dim.is_multiple_of(2)
                && md.rope_theta_swa.is_finite()
                && md.rope_theta_swa > 0.0,
            "invalid Muse Spark rotary dimensions"
        );
        anyhow::ensure!(
            md.rope_scaling_factor == 1.0,
            "Muse Spark scaled RoPE is not supported"
        );
        let hidden = md.hidden_dim;
        let latent = md.moe_latent_size;
        let embedding = if range.is_head() {
            let tensor = tensors
                .get("token_embd.weight")
                .context("missing token_embd.weight")?;
            anyhow::ensure!(
                tensor.shape().dims() == [md.vocab_size, hidden],
                "Muse Spark embedding shape mismatch"
            );
            Some(Arc::clone(tensor))
        } else {
            None
        };
        let mut blocks = Vec::with_capacity(range.layer_count());
        for layer in range.start..range.end {
            let prefix = format!("blk.{layer}");
            let norm = |suffix: &str, size, eps| {
                load_norm(
                    &tensors,
                    &format!("{prefix}.{suffix}.weight"),
                    size,
                    eps,
                    &device,
                )
            };
            let projection = |suffix: &str, output, input| {
                load_projection(
                    &tensors,
                    &format!("{prefix}.{suffix}.weight"),
                    output,
                    input,
                    &device,
                )
            };
            let q_dim = md
                .num_attention_heads
                .checked_mul(md.head_dim)
                .context("Muse Spark query width overflow")?;
            let kv_dim = md
                .num_kv_heads
                .checked_mul(md.head_dim)
                .context("Muse Spark key/value width overflow")?;
            let attention = Attention {
                q: projection("attn_q", q_dim, hidden)?,
                k: projection("attn_k", kv_dim, hidden)?,
                v: projection("attn_v", kv_dim, hidden)?,
                out: projection("attn_output", hidden, q_dim)?,
                gate: projection("attn_gate", q_dim, hidden)?,
                q_norm: norm("attn_q_norm", md.head_dim, md.rms_norm_eps)?,
                k_norm: norm("attn_k_norm", md.head_dim, md.rms_norm_eps)?,
                heads: md.num_attention_heads,
                kv_heads: md.num_kv_heads,
                head_dim: md.head_dim,
                rope_dim,
                rope_theta: md.rope_theta_swa,
                window: md.sliding_window_pattern[layer].then_some(md.sliding_window),
                eps: md.rms_norm_eps,
                quant: quant.clone(),
            };
            let router_name = format!("{prefix}.ffn_gate_inp.weight");
            let router = tensors
                .get(&router_name)
                .with_context(|| format!("missing {router_name}"))?;
            anyhow::ensure!(
                router.shape().dims() == [md.expert_count, hidden],
                "Muse Spark router shape mismatch"
            );
            let router = router.dequantize(&device)?.to_dtype(DType::F32)?;
            let moe = Moe {
                router,
                correction: load_vector(
                    &tensors,
                    &format!("{prefix}.ffn_exp_probs_b.bias"),
                    md.expert_count,
                    &device,
                )?,
                down: projection("ffn_latent_down", latent, hidden)?,
                up: projection("ffn_latent_up", hidden, latent)?,
                pre_norm: norm("ffn_pre_exp_norm", latent, md.rms_norm_eps)?,
                post_norm: norm("ffn_post_exp_norm", latent, md.post_norm_eps)?,
                experts: load_experts(
                    &mut tensors,
                    &prefix,
                    md.expert_count,
                    md.expert_ffn_dim,
                    latent,
                    &device,
                )?,
                latent,
                active: md.expert_used_count,
                scale: md.expert_weights_scale,
            };
            blocks.push(Block {
                attn_norm: load_norm(
                    &tensors,
                    &format!("{prefix}.attn_norm.weight"),
                    hidden,
                    md.rms_norm_eps,
                    &device,
                )?,
                attention,
                attn_residual: Residual::load(
                    &tensors,
                    &format!("{prefix}.attn_resid"),
                    hidden,
                    &device,
                )?,
                ffn_norm: load_norm(
                    &tensors,
                    &format!("{prefix}.ffn_norm.weight"),
                    hidden,
                    md.rms_norm_eps,
                    &device,
                )?,
                moe,
                ffn_post_norm: load_norm(
                    &tensors,
                    &format!("{prefix}.post_ffw_norm.weight"),
                    hidden,
                    md.post_norm_eps,
                    &device,
                )?,
                ffn_residual: Residual::load(
                    &tensors,
                    &format!("{prefix}.ffn_resid"),
                    hidden,
                    &device,
                )?,
                post_eps: md.post_norm_eps,
            });
        }
        let output_norm = range
            .is_tail()
            .then(|| {
                load_norm(
                    &tensors,
                    "output_norm.weight",
                    hidden,
                    md.rms_norm_eps,
                    &device,
                )
            })
            .transpose()?;
        let output = range
            .is_tail()
            .then(|| load_projection(&tensors, "output.weight", md.vocab_size, hidden, &device))
            .transpose()?;
        Ok(Self {
            embedding,
            blocks,
            output_norm,
            output,
            head: range.is_head(),
            tail: range.is_tail(),
            hidden,
            vocab: md.vocab_size,
            eps: md.rms_norm_eps,
            logit_scale: md.logit_scale,
            softcap: md.final_logit_softcapping,
            device,
        })
    }

    fn forward_impl(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
        prefill: bool,
        cache_only: bool,
    ) -> Result<StageOutput> {
        anyhow::ensure!(
            seq_len > 0 && cache.len() == self.blocks.len(),
            "Muse Spark sequence or layer cache length mismatch"
        );
        let mut hidden = match input {
            StageInput::Tokens(tokens) => {
                anyhow::ensure!(tokens.len() == seq_len, "Muse Spark token count mismatch");
                self.embed_tokens(&tokens)?
                    .reshape((1, seq_len, self.hidden))?
            }
            StageInput::Activations {
                tensor,
                bidirectional_spans,
                positions,
            } => {
                anyhow::ensure!(
                    bidirectional_spans.is_empty() && positions.is_none(),
                    "Muse Spark text stage does not support media positions or spans"
                );
                let tensor = if tensor.rank() == 2 {
                    tensor.unsqueeze(0)?
                } else {
                    tensor
                };
                anyhow::ensure!(
                    tensor.dims3()? == (1, seq_len, self.hidden),
                    "Muse Spark activation shape mismatch"
                );
                tensor.to_dtype(DType::F32)?
            }
        };
        for (block, cache) in self.blocks.iter().zip(cache) {
            let branch = block.attn_norm.forward(&hidden)?;
            let branch = with_kv_cache(cache, |cache| {
                block.attention.forward(&branch, cache, start_pos)
            })?;
            hidden = block
                .attn_residual
                .forward(&hidden, &rms(&branch, block.post_eps)?)?;
            let branch = block
                .moe
                .forward(&block.ffn_norm.forward(&hidden)?, prefill)?;
            hidden = block
                .ffn_residual
                .forward(&hidden, &block.ffn_post_norm.input_scaled(&branch)?)?;
        }
        if self.tail && !cache_only {
            let last = hidden.i((.., seq_len - 1, ..))?;
            let last = self
                .output_norm
                .as_ref()
                .context("Muse Spark tail has no output norm")?
                .forward(&last)?;
            let logits = self
                .output
                .as_ref()
                .context("Muse Spark tail has no output projection")?
                .forward_decode(&last)?;
            let logits = ((logits * (self.logit_scale / self.softcap))?.tanh()? * self.softcap)?;
            Ok(StageOutput::Logits(logits.squeeze(0)?))
        } else {
            Ok(StageOutput::Activations(hidden.squeeze(0)?))
        }
    }
}

impl ModelArchitecture for MuseSparkStage {
    fn forward(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_impl(input, cache, seq_len, start_pos, false, false)
    }
    fn forward_prefill(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_impl(input, cache, seq_len, start_pos, true, false)
    }
    fn forward_prefill_cache_only(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_impl(input, cache, seq_len, start_pos, true, true)
    }
    fn embed_tokens(&self, tokens: &[u32]) -> Result<Tensor> {
        anyhow::ensure!(
            self.head
                && !tokens.is_empty()
                && tokens.iter().all(|token| (*token as usize) < self.vocab),
            "invalid Muse Spark token embedding request"
        );
        let ids = Tensor::new(tokens, &self.device)?;
        rms(
            &self
                .embedding
                .as_ref()
                .context("Muse Spark head has no embedding")?
                .embedding(&ids)?,
            self.eps,
        )
    }
    fn prefers_f32_decode_activations(&self) -> bool {
        true
    }
    fn prefers_f32_prefill_activations(&self) -> bool {
        true
    }
    fn family(&self) -> &str {
        "muse-spark"
    }
    fn hidden_dim(&self) -> usize {
        self.hidden
    }
    fn layer_count(&self) -> usize {
        self.blocks.len()
    }
    fn vocab_size(&self) -> usize {
        self.vocab
    }
}
