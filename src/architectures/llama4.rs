//! Llama 4 architecture — GQA attention + MoE with shared experts + iRoPE.
//!
//! Key differences from standard Llama:
//! - Alternating dense and MoE FFN layers (controlled by moe_layer_step)
//! - iRoPE: some layers skip position encoding (NoPE layers)
//! - Top-1 routing with shared expert
//! - Same Q/K/V/O attention projections as Llama

use std::collections::HashMap;
use std::sync::Arc;

use crate::qmatmul::QMatMul;
use anyhow::{Context, Result};
#[cfg(any(target_os = "macos", feature = "cuda"))]
use candle_core::D;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Tensor};

#[cfg(target_os = "macos")]
use super::fused_norm_metal::try_fused_add_rms_norm;
#[cfg(target_os = "macos")]
use super::fused_route_metal::fused_llama4_topk_route;
#[cfg(any(target_os = "macos", feature = "cuda"))]
use super::indexed_moe::{IndexedMoeMatMul, indexed_moe_available, silu_mul as indexed_silu_mul};
use super::{
    DecodeSafeQMatMul, KvCache, LayerCache, ModelArchitecture, StageInput, StageOutput,
    with_kv_cache,
};
use crate::gguf_loader::StageTensors;
use crate::turboquant::TurboQuantConfig;

// ===== Helpers =====

const DEFAULT_ROPE_CACHE_TOKENS: usize = 8192;
const NOPE_TEMPERATURE_FLOOR: usize = 8192;

fn uses_rope(layer_idx: usize, chunk_size: usize) -> bool {
    chunk_size == 0 || !(layer_idx + 1).is_multiple_of(4)
}

fn apply_nope_temperature(q: &Tensor, start_pos: usize) -> Result<Tensor> {
    let seq_len = q.dim(2)?;
    let end_pos = start_pos
        .checked_add(seq_len)
        .context("Llama 4 attention position overflow")?;
    if end_pos < NOPE_TEMPERATURE_FLOOR {
        return Ok(q.clone());
    }
    // Match llama.cpp's input scale: position arithmetic is f32, followed by
    // the f64 log/multiply/add and a final f32 store. The first scaled query
    // is position 8191, one token before chunked attention starts a new block.
    let scales = (start_pos..end_pos)
        .map(|pos| {
            let bucket = ((pos as f32 + 1.0) / NOPE_TEMPERATURE_FLOOR as f32).floor();
            ((f64::from(bucket) + 1.0).ln() * f64::from(0.1f32) + 1.0) as f32
        })
        .collect::<Vec<_>>();
    let scales = Tensor::from_vec(scales, (1, 1, seq_len, 1), q.device())?.to_dtype(q.dtype())?;
    Ok(q.broadcast_mul(&scales)?)
}

fn is_moe_layer(
    layer_idx: usize,
    first_dense_layer_count: usize,
    moe_step: usize,
    has_moe: bool,
) -> bool {
    has_moe
        && layer_idx >= first_dense_layer_count
        && (layer_idx + 1 - first_dense_layer_count).is_multiple_of(moe_step)
}

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input = if x.is_contiguous() {
            x.clone()
        } else {
            x.contiguous()?
        };
        candle_nn::ops::rms_norm(&input, &self.weight, self.eps as f32).map_err(Into::into)
    }

    fn forward_residual(&self, lhs: &Tensor, rhs: &Tensor) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        if let Some(result) =
            super::residual_norm_cuda::forward(lhs, rhs, &self.weight, self.eps as f32)?
        {
            return Ok(result);
        }
        #[cfg(target_os = "macos")]
        if let Some(result) = try_fused_add_rms_norm(lhs, rhs, &self.weight, self.eps as f32)? {
            return Ok(result);
        }
        let residual = (lhs + rhs)?;
        let normalized = self.forward(&residual)?;
        Ok((residual, normalized))
    }
}

fn silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    #[cfg(any(target_os = "macos", feature = "cuda"))]
    return indexed_silu_mul(gate, up);
    #[cfg(not(any(target_os = "macos", feature = "cuda")))]
    Ok((candle_nn::Activation::Silu.forward(gate)? * up)?)
}

fn qtensor_to_qmatmul(qt: Arc<QTensor>) -> Result<QMatMul> {
    QMatMul::from_arc(qt).map_err(Into::into)
}

fn split_expert_tensor(
    packed: &QTensor,
    num_experts: usize,
    out_dim: usize,
    in_dim: usize,
    device: &Device,
) -> Result<Vec<QMatMul>> {
    let ggml_dtype: GgmlDType = packed.dtype();
    let elements_per_expert = out_dim * in_dim;
    let block_size = ggml_dtype.block_size();
    anyhow::ensure!(
        elements_per_expert.is_multiple_of(block_size),
        "Expert dimensions ({out_dim}x{in_dim}={elements_per_expert}) not aligned to block size ({block_size})"
    );

    let bytes_per_expert = (elements_per_expert / block_size) * ggml_dtype.type_size();
    let raw_data = packed.data()?;
    anyhow::ensure!(
        raw_data.len() >= num_experts * bytes_per_expert,
        "Packed expert tensor too small: {} bytes, need {} ({num_experts} experts x {bytes_per_expert})",
        raw_data.len(),
        num_experts * bytes_per_expert,
    );

    let mut experts = Vec::with_capacity(num_experts);
    for expert_idx in 0..num_experts {
        let start = expert_idx * bytes_per_expert;
        let end = start + bytes_per_expert;
        let expert_qt = candle_core::quantized::ggml_file::qtensor_from_ggml(
            ggml_dtype,
            &raw_data[start..end],
            vec![out_dim, in_dim],
            device,
        )?;
        experts.push(QMatMul::from_arc(Arc::new(expert_qt))?);
    }

    Ok(experts)
}

fn load_rope_freqs(
    tensors: &HashMap<String, Arc<QTensor>>,
    head_dim: usize,
    device: &Device,
) -> Result<Option<Tensor>> {
    let Some(factors) = tensors.get("rope_freqs.weight") else {
        return Ok(None);
    };
    anyhow::ensure!(
        head_dim > 0 && head_dim.is_multiple_of(2) && factors.shape().dims() == [head_dim / 2],
        "Llama 4 rope_freqs.weight must have shape [{}] for head width {head_dim}",
        head_dim / 2
    );
    let factors = factors.dequantize(device)?.to_dtype(DType::F32)?;
    anyhow::ensure!(
        factors
            .to_vec1::<f32>()?
            .iter()
            .all(|factor| factor.is_finite() && *factor > 0.0),
        "Llama 4 rope_freqs.weight factors must be finite and positive"
    );
    Ok(Some(factors))
}

fn precompute_rope_cos_sin(
    head_dim: usize,
    theta: f64,
    context_length: usize,
    device: &Device,
    freq_factors: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    precompute_rope_cos_sin_range(head_dim, theta, 0, context_length, device, freq_factors)
}

fn precompute_rope_cos_sin_range(
    head_dim: usize,
    theta: f64,
    start_pos: usize,
    seq_len: usize,
    device: &Device,
    freq_factors: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
    let end_pos = start_pos
        .checked_add(seq_len)
        .context("Llama 4 RoPE position overflow")?;
    let start_pos = u32::try_from(start_pos).context("Llama 4 RoPE position exceeds u32")?;
    let end_pos = u32::try_from(end_pos).context("Llama 4 RoPE position exceeds u32")?;
    let positions = Tensor::arange(start_pos, end_pos, device)?
        .to_dtype(DType::F32)?
        .reshape((seq_len, 1))?;
    let mut angles = positions.matmul(&inv_freq.reshape((1, half_dim))?)?;
    if let Some(factors) = freq_factors {
        // GGUF stores divisors, not inverse frequencies. Apply them once to
        // the angles, as ggml_rope_ext does for Scout's 1x/16x factors.
        angles = angles.broadcast_div(&factors.reshape((1, half_dim))?)?;
    }
    Ok((angles.cos()?, angles.sin()?))
}

fn apply_rotary_emb(
    x: &Tensor,
    start_pos: usize,
    cos: &Tensor,
    sin: &Tensor,
    theta: f64,
    freq_factors: Option<&Tensor>,
) -> Result<Tensor> {
    let (_, _, seq_len, head_dim) = x.dims4()?;
    let end_pos = start_pos
        .checked_add(seq_len)
        .context("Llama 4 RoPE position overflow")?;
    let (cos, sin) = if end_pos <= cos.dim(0)? {
        (
            cos.narrow(0, start_pos, seq_len)?,
            sin.narrow(0, start_pos, seq_len)?,
        )
    } else {
        precompute_rope_cos_sin_range(
            head_dim,
            theta,
            start_pos,
            seq_len,
            x.device(),
            freq_factors,
        )?
    };
    Ok(candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)?)
}

fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::softmax_last_dim(x).map_err(Into::into)
}

fn create_causal_mask(seq_len: usize, total_seq_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..total_seq_len).map(move |j| {
                let query_pos = total_seq_len - seq_len + i;
                if j <= query_pos {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Ok(Tensor::from_vec(
        mask,
        (1, 1, seq_len, total_seq_len),
        device,
    )?)
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, num_kv_heads, seq_len, head_dim) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, num_kv_heads, n_rep, seq_len, head_dim))?
        .reshape((b, num_kv_heads * n_rep, seq_len, head_dim))?
        .contiguous()
        .map_err(Into::into)
}

fn causal_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let seq_len = q.dim(2)?;
    let scale = 1.0 / (q.dim(3)? as f64).sqrt();
    if let Some(output) = super::try_fused_f16_sdpa_native(q, k, v, seq_len > 1, scale as f32)? {
        return Ok(output);
    }
    let n_rep = q.dim(1)? / k.dim(1)?;
    let k = repeat_kv(k.to_dtype(q.dtype())?, n_rep)?;
    let v = repeat_kv(v.to_dtype(q.dtype())?, n_rep)?;
    let att = (q.contiguous()?.matmul(&k.t()?.contiguous()?)? * scale)?;
    let att = if seq_len > 1 {
        let mask = create_causal_mask(seq_len, att.dim(3)?, q.device())?;
        att.broadcast_add(&mask)?
    } else {
        att
    };
    Ok(softmax_last_dim(&att)?.matmul(&v.contiguous()?)?)
}

fn cached_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    start_pos: usize,
    chunk_size: usize,
) -> Result<Tensor> {
    let seq_len = q.dim(2)?;
    let kv_len = k.dim(2)?;
    anyhow::ensure!(
        seq_len > 0 && kv_len >= seq_len,
        "invalid Llama 4 KV length"
    );
    anyhow::ensure!(v.dim(2)? == kv_len, "Llama 4 K/V lengths differ");
    let end_pos = start_pos
        .checked_add(seq_len)
        .context("Llama 4 attention position overflow")?;
    let cache_start = end_pos
        .checked_sub(kv_len)
        .context("Llama 4 KV cache precedes position zero")?;
    if chunk_size == 0 {
        return causal_attention(q, k, v);
    }

    // Chunked attention admits only keys in the query's fixed position block;
    // it is not a rolling left window. Split a call that crosses a block edge
    // so each native SDPA call can use its ordinary causal mask. Only narrow
    // read views: full-attention layers and snapshots keep their complete KV.
    let mut outputs = Vec::new();
    let mut offset = 0;
    while offset < seq_len {
        let position = start_pos + offset;
        let block_start = position - position % chunk_size;
        let len = (chunk_size - position % chunk_size).min(seq_len - offset);
        let visible_start = block_start.max(cache_start);
        let visible_len = position + len - visible_start;
        let key_offset = visible_start - cache_start;
        outputs.push(causal_attention(
            &q.narrow(2, offset, len)?,
            &k.narrow(2, key_offset, visible_len)?,
            &v.narrow(2, key_offset, visible_len)?,
        )?);
        offset += len;
    }
    if outputs.len() == 1 {
        Ok(outputs.pop().unwrap())
    } else {
        Ok(Tensor::cat(&outputs, 2)?)
    }
}

// ===== Attention (standard GQA, optionally skip RoPE for NoPE layers) =====

struct Attention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: DecodeSafeQMatMul,
    o_proj: QMatMul,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_cos: Tensor,
    rope_sin: Tensor,
    rope_theta: f64,
    rope_freqs: Option<Tensor>,
    use_rope: bool, // false for NoPE layers
    chunk_size: usize,
    qk_norm: Option<RmsNorm>,
    quant_config: Option<Arc<TurboQuantConfig>>,
}

impl Attention {
    fn forward(
        &self,
        x: &Tensor,
        kv_cache: &mut Option<KvCache>,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;

        #[cfg(feature = "cuda")]
        let grouped_qkv = super::dense_qmatmul_cuda::forward(
            x,
            &[&self.q_proj, &self.k_proj, self.v_proj.quantized()],
        )?;
        #[cfg(not(feature = "cuda"))]
        let grouped_qkv: Option<Vec<Tensor>> = None;
        let (q, k, v) = if let Some(mut outputs) = grouped_qkv {
            anyhow::ensure!(outputs.len() == 3, "grouped Llama 4 QKV count mismatch");
            let v = outputs.pop().unwrap();
            let k = outputs.pop().unwrap();
            let q = outputs.pop().unwrap();
            (q, k, v)
        } else {
            (
                self.q_proj.forward(x)?,
                self.k_proj.forward(x)?,
                self.v_proj.forward(x)?,
            )
        };
        let (q, k, v) = if seq_len == 1 {
            (
                q.reshape((b, self.num_heads, 1, self.head_dim))?,
                k.reshape((b, self.num_kv_heads, 1, self.head_dim))?,
                v.reshape((b, self.num_kv_heads, 1, self.head_dim))?,
            )
        } else {
            #[cfg(target_os = "macos")]
            if let Some(layout) = super::qkv_layout_metal::try_fused_separate_qkv_layout(
                &q,
                &k,
                &v,
                super::qkv_layout_metal::QkvLayoutDimensions {
                    batch: b,
                    seq_len: s,
                    num_heads: self.num_heads,
                    num_kv_heads: self.num_kv_heads,
                    head_dim: self.head_dim,
                },
            )? {
                layout
            } else {
                self.portable_qkv_layout(q, k, v, b, s)?
            }
            #[cfg(not(target_os = "macos"))]
            self.portable_qkv_layout(q, k, v, b, s)?
        };

        // Apply RoPE only if this layer uses it (iRoPE: some layers are NoPE)
        let (q, k) = if self.use_rope {
            let q = apply_rotary_emb(
                &q,
                start_pos,
                &self.rope_cos,
                &self.rope_sin,
                self.rope_theta,
                self.rope_freqs.as_ref(),
            )?;
            let k = apply_rotary_emb(
                &k,
                start_pos,
                &self.rope_cos,
                &self.rope_sin,
                self.rope_theta,
                self.rope_freqs.as_ref(),
            )?;
            if let Some(norm) = &self.qk_norm {
                (norm.forward(&q)?, norm.forward(&k)?)
            } else {
                (q, k)
            }
        } else {
            (apply_nope_temperature(&q, start_pos)?, k)
        };

        // Cache compact KV heads. Fused Metal attention consumes GQA directly;
        // the portable path expands heads only after the cache update.
        let (q, k, v, rotation) =
            super::prepare_quantized_attention(kv_cache, q, k, v, self.quant_config.as_deref())?;
        let out = cached_attention(&q, &k, &v, start_pos, self.chunk_size)?;
        let out = super::restore_quantized_attention_output(out, rotation)?;

        #[cfg(target_os = "macos")]
        let out = if let Some(output) =
            super::qkv_layout_metal::try_fused_attention_output_layout(&out)?
        {
            output
        } else {
            out.to_dtype(x.dtype())?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b, s, self.num_heads * self.head_dim))?
        };
        #[cfg(not(target_os = "macos"))]
        let out =
            out.transpose(1, 2)?
                .contiguous()?
                .reshape((b, s, self.num_heads * self.head_dim))?;
        #[cfg(all(feature = "cuda", not(target_os = "macos")))]
        if let Some(output) = super::dense_qmatmul_cuda::forward_one_to_f32(&out, &self.o_proj)? {
            return Ok(output);
        }
        #[cfg(not(target_os = "macos"))]
        let out = out.to_dtype(x.dtype())?;
        self.o_proj.forward(&out).map_err(Into::into)
    }

    fn portable_qkv_layout(
        &self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        batch: usize,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        Ok((q, k, v))
    }
}

// ===== MLP (SwiGLU) =====

struct Mlp {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(activated) =
            super::dense_qmatmul_cuda::silu_gate_up(x, &self.gate_proj, &self.up_proj)?
        {
            if let Some(output) =
                super::dense_qmatmul_cuda::forward_one_to_f32(&activated, &self.down_proj)?
            {
                return Ok(output);
            }
            return self.down_proj.forward(&activated).map_err(Into::into);
        }
        #[cfg(feature = "cuda")]
        let grouped = super::dense_qmatmul_cuda::forward(x, &[&self.gate_proj, &self.up_proj])?;
        #[cfg(not(feature = "cuda"))]
        let grouped: Option<Vec<Tensor>> = None;
        let (gate, up) = if let Some(mut outputs) = grouped {
            anyhow::ensure!(outputs.len() == 2, "grouped Llama 4 MLP count mismatch");
            let up = outputs.pop().unwrap();
            let gate = outputs.pop().unwrap();
            (gate, up)
        } else {
            (self.gate_proj.forward(x)?, self.up_proj.forward(x)?)
        };
        #[cfg(feature = "cuda")]
        if let Some(output) = super::dense_qmatmul_cuda::silu_mul_down(&gate, &up, &self.down_proj)?
        {
            return Ok(output);
        }
        let activated = silu_mul(&gate, &up)?;
        self.down_proj.forward(&activated).map_err(Into::into)
    }
}

// ===== MoE =====

struct MoeExpert {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

impl MoeExpert {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = silu_mul(&gate, &up)?;
        self.down_proj.forward(&activated).map_err(Into::into)
    }
}

#[cfg(any(target_os = "macos", feature = "cuda"))]
struct IndexedMoeExperts {
    gate_proj: IndexedMoeMatMul,
    up_proj: IndexedMoeMatMul,
    down_proj: IndexedMoeMatMul,
}

#[cfg(any(target_os = "macos", feature = "cuda"))]
impl IndexedMoeExperts {
    const MAX_TOKENS_PER_DISPATCH: usize = 4096;

    fn forward(&self, flat_x: &Tensor, expert_ids: &Tensor, weights: &Tensor) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = flat_x.dims2()?;
        anyhow::ensure!(
            expert_ids.dims2()? == (num_tokens, 1) && weights.dims2()? == (num_tokens, 1),
            "indexed Llama 4 routing requires exactly one expert per token"
        );

        let mut outputs = Vec::with_capacity(num_tokens.div_ceil(Self::MAX_TOKENS_PER_DISPATCH));
        for start in (0..num_tokens).step_by(Self::MAX_TOKENS_PER_DISPATCH) {
            let len = (num_tokens - start).min(Self::MAX_TOKENS_PER_DISPATCH);
            let input = flat_x.narrow(0, start, len)?.to_dtype(DType::F32)?;
            let ids = expert_ids.narrow(0, start, len)?;
            let route_weights = weights.narrow(0, start, len)?;
            let weighted_input = input
                .broadcast_mul(&route_weights)?
                .reshape((len, 1, hidden_dim))?;
            let activated = if let Some(activated) =
                self.gate_proj
                    .silu_gate_up(&self.up_proj, &weighted_input, &ids)?
            {
                activated
            } else {
                let gate = self.gate_proj.forward(&weighted_input, &ids)?;
                let up = self.up_proj.forward(&weighted_input, &ids)?;
                indexed_silu_mul(&gate, &up)?
            };
            outputs.push(self.down_proj.forward(&activated, &ids)?.squeeze(1)?);
        }

        let output_refs = outputs.iter().collect::<Vec<_>>();
        Tensor::cat(&output_refs, 0).map_err(Into::into)
    }
}

enum MoeExperts {
    Split(Vec<MoeExpert>),
    #[cfg(any(target_os = "macos", feature = "cuda"))]
    Indexed(Box<IndexedMoeExperts>),
}

impl MoeExperts {
    fn len(&self) -> usize {
        match self {
            Self::Split(experts) => experts.len(),
            #[cfg(any(target_os = "macos", feature = "cuda"))]
            Self::Indexed(experts) => experts.gate_proj.num_experts(),
        }
    }
}

struct MoeLayer {
    gate: Tensor,
    experts: MoeExperts,
    shared_expert: Option<Mlp>,
    num_active: usize, // top-k (1 for Llama 4)
    layer_idx: usize,
}

impl MoeLayer {
    fn selected_experts(&self, logits: &[f32]) -> Result<Vec<(usize, f32)>> {
        anyhow::ensure!(
            logits.len() == self.experts.len(),
            "Llama 4 router returned {} logits for {} experts",
            logits.len(),
            self.experts.len()
        );
        anyhow::ensure!(
            logits.iter().all(|value| value.is_finite()),
            "non-finite Llama 4 router logits at layer {}",
            self.layer_idx,
        );
        anyhow::ensure!(
            self.num_active <= logits.len(),
            "Llama 4 requests {} active experts from a pool of {}",
            self.num_active,
            logits.len()
        );
        let mut indexed = logits.iter().copied().enumerate().collect::<Vec<_>>();
        indexed.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        indexed.truncate(self.num_active);
        for (_, weight) in &mut indexed {
            *weight = 1.0 / (1.0 + (-*weight).exp());
        }
        Ok(indexed)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_dim) = x.dims3()?;
        let num_tokens = b_sz * seq_len;
        let flat_x = x.reshape((num_tokens, hidden_dim))?;

        let router_logits = flat_x.matmul(&self.gate.t()?)?;

        #[cfg(target_os = "macos")]
        let fused_indexed_routing = match &self.experts {
            MoeExperts::Indexed(_) => {
                Some(fused_llama4_topk_route(&router_logits.contiguous()?, 1)?)
            }
            _ => None,
        };
        #[cfg(all(feature = "cuda", not(target_os = "macos")))]
        let fused_indexed_routing: Option<(Tensor, Tensor)> = None;

        #[cfg(any(target_os = "macos", feature = "cuda"))]
        let indexed_output = if let MoeExperts::Indexed(experts) = &self.experts {
            anyhow::ensure!(
                self.num_active == 1,
                "indexed Llama 4 requires top-1 routing"
            );
            let (expert_ids, weights) = if let Some(routing) = fused_indexed_routing {
                routing
            } else {
                let expert_ids = router_logits
                    .contiguous()?
                    .arg_sort_last_dim(false)?
                    .narrow(D::Minus1, 0, 1)?
                    .contiguous()?;
                let selected_logits = router_logits.gather(&expert_ids, D::Minus1)?;
                let weights = candle_nn::Activation::Sigmoid.forward(&selected_logits)?;
                (expert_ids, weights)
            };
            Some(experts.forward(&flat_x, &expert_ids, &weights)?)
        } else {
            None
        };
        #[cfg(not(any(target_os = "macos", feature = "cuda")))]
        let indexed_output: Option<Tensor> = None;

        let output = if let Some(output) = indexed_output {
            output
        } else {
            #[cfg(any(target_os = "macos", feature = "cuda"))]
            let MoeExperts::Split(experts) = &self.experts else {
                anyhow::bail!("indexed Llama 4 experts are unavailable on this platform")
            };
            #[cfg(not(any(target_os = "macos", feature = "cuda")))]
            let MoeExperts::Split(experts) = &self.experts;
            let rows = router_logits.to_vec2::<f32>()?;
            let mut token_outputs = Vec::with_capacity(num_tokens);
            for (token_idx, token_logits) in rows.iter().enumerate() {
                let selected = self.selected_experts(token_logits)?;
                let token_input = flat_x.i(token_idx)?.unsqueeze(0)?;
                let mut token_output = Tensor::zeros((1, hidden_dim), DType::F32, x.device())?;
                for &(expert_idx, weight) in &selected {
                    let weighted_input = (&token_input * weight as f64)?;
                    let expert_out = experts[expert_idx].forward(&weighted_input)?;
                    token_output = (token_output + expert_out.to_dtype(DType::F32)?)?;
                }
                token_outputs.push(token_output);
            }
            let output_refs = token_outputs.iter().collect::<Vec<_>>();
            Tensor::cat(&output_refs, 0)?
        };

        let mut result = output
            .to_dtype(x.dtype())?
            .reshape((b_sz, seq_len, hidden_dim))?;

        if let Some(ref shared) = self.shared_expert {
            result = (result + shared.forward(x)?)?;
        }

        Ok(result)
    }
}

// ===== Block types =====

enum Llama4Block {
    Dense(DenseBlock),
    Moe(MoeBlock),
}

impl Llama4Block {
    fn attn_norm(&self) -> &RmsNorm {
        match self {
            Self::Dense(block) => &block.attn_norm,
            Self::Moe(block) => &block.attn_norm,
        }
    }

    fn ffn_norm(&self) -> &RmsNorm {
        match self {
            Self::Dense(block) => &block.ffn_norm,
            Self::Moe(block) => &block.ffn_norm,
        }
    }
}

struct DenseBlock {
    attn_norm: RmsNorm,
    attn: Attention,
    ffn_norm: RmsNorm,
    ffn: Mlp,
}

struct MoeBlock {
    attn_norm: RmsNorm,
    attn: Attention,
    ffn_norm: RmsNorm,
    moe: MoeLayer,
}

// ===== Llama 4 Stage =====

/// A loaded Llama 4 stage for pipeline execution.
pub struct Llama4Stage {
    embedding: Option<Tensor>,
    blocks: Vec<Llama4Block>,
    output_norm: Option<RmsNorm>,
    lm_head: Option<DecodeSafeQMatMul>,
    is_head: bool,
    is_tail: bool,
    hidden_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    device: Device,
}

impl Llama4Stage {
    /// Construct a Llama 4 stage from its assigned GGUF tensors.
    ///
    /// Returns an error when the alternating attention/MoE schedule or any
    /// required tensor is missing or incompatible.
    pub fn from_stage_tensors(
        stage: StageTensors,
        quant_config: Option<Arc<TurboQuantConfig>>,
    ) -> Result<Self> {
        let md = &stage.metadata;
        let device = &stage.device;
        let range = &stage.range;
        let tensors = &stage.tensors;
        anyhow::ensure!(
            md.sliding_window == 0 || md.sliding_window_pattern.len() == md.num_layers,
            "Llama 4 chunked attention requires a mask pattern for every layer"
        );
        // Validate the small shared tensor before allocating embeddings or
        // attention tables. Every pipeline stage loads this GGUF tensor.
        let rope_freqs = load_rope_freqs(tensors, md.head_dim, device)?;
        #[cfg(feature = "cuda")]
        {
            super::dense_qmatmul_cuda::precompile(device)?;
            super::residual_norm_cuda::precompile(device)?;
        }

        let embedding = if range.is_head() {
            let embd = tensors
                .get("token_embd.weight")
                .context("missing token_embd.weight")?;
            Some(embd.as_ref().dequantize(device)?)
        } else {
            None
        };

        let rope_cache_tokens = md.context_length.min(DEFAULT_ROPE_CACHE_TOKENS);
        let (rope_cos, rope_sin) = precompute_rope_cos_sin(
            md.head_dim,
            md.rope_theta,
            rope_cache_tokens,
            device,
            rope_freqs.as_ref(),
        )?;
        let (chunk_rope_cos, chunk_rope_sin) = if md.sliding_window > 0
            && md.rope_theta_swa != 0.0
            && md.rope_theta_swa != md.rope_theta
        {
            precompute_rope_cos_sin(
                md.head_dim,
                md.rope_theta_swa,
                rope_cache_tokens,
                device,
                rope_freqs.as_ref(),
            )?
        } else {
            (rope_cos.clone(), rope_sin.clone())
        };

        // Determine layer types:
        // - moe_layer_freq > 0: every Nth layer is MoE (starting from first_k_dense_replace)
        // - iRoPE: Llama 4 uses RoPE on 3 layers, then a NoPE layer.
        let has_moe = md.expert_count > 0;
        let moe_step = if md.moe_layer_freq > 0 {
            md.moe_layer_freq
        } else {
            1
        };

        let mut blocks = Vec::with_capacity(range.layer_count());
        for layer_idx in range.start..range.end {
            let prefix = format!("blk.{layer_idx}");
            let attn_norm = load_rms_norm(
                tensors,
                &format!("{prefix}.attn_norm"),
                md.rms_norm_eps,
                device,
            )?;
            let ffn_norm = load_rms_norm(
                tensors,
                &format!("{prefix}.ffn_norm"),
                md.rms_norm_eps,
                device,
            )?;

            let chunked = md.sliding_window > 0 && md.sliding_window_pattern[layer_idx];
            let (layer_cos, layer_sin) = if chunked {
                (&chunk_rope_cos, &chunk_rope_sin)
            } else {
                (&rope_cos, &rope_sin)
            };
            let attn = build_attention(
                tensors,
                &prefix,
                md,
                (layer_cos, layer_sin, rope_freqs.as_ref()),
                layer_idx,
                quant_config.clone(),
            )?;

            // Determine if this layer is MoE or dense
            let is_moe_layer = is_moe_layer(layer_idx, md.first_k_dense_replace, moe_step, has_moe);

            if is_moe_layer {
                let moe = build_moe_layer(tensors, &prefix, layer_idx, md, device)?;
                blocks.push(Llama4Block::Moe(MoeBlock {
                    attn_norm,
                    attn,
                    ffn_norm,
                    moe,
                }));
            } else {
                let ffn = build_mlp(tensors, &prefix)?;
                blocks.push(Llama4Block::Dense(DenseBlock {
                    attn_norm,
                    attn,
                    ffn_norm,
                    ffn,
                }));
            }
        }

        let output_norm = if range.is_tail() {
            Some(load_rms_norm(
                tensors,
                "output_norm",
                md.rms_norm_eps,
                device,
            )?)
        } else {
            None
        };

        let lm_head = if range.is_tail() {
            let head = tensors
                .get("output.weight")
                .or_else(|| tensors.get("token_embd.weight"))
                .context("missing output.weight")?;
            Some(DecodeSafeQMatMul::from_qtensor(Arc::clone(head), device)?)
        } else {
            None
        };

        tracing::info!(
            "Llama4 stage: layers [{}, {}), {} blocks ({} dense + {} MoE), {} experts (top-{})",
            range.start,
            range.end,
            blocks.len(),
            blocks
                .iter()
                .filter(|b| matches!(b, Llama4Block::Dense(_)))
                .count(),
            blocks
                .iter()
                .filter(|b| matches!(b, Llama4Block::Moe(_)))
                .count(),
            md.expert_count,
            md.expert_used_count,
        );
        tracing::debug!(
            rope_cache_tokens,
            context_length = md.context_length,
            "Llama 4 RoPE cache initialized"
        );

        Ok(Self {
            embedding,
            blocks,
            output_norm,
            lm_head,
            is_head: range.is_head(),
            is_tail: range.is_tail(),
            hidden_dim: md.hidden_dim,
            num_layers: md.num_layers,
            vocab_size: md.vocab_size,
            device: device.clone(),
        })
    }
}

impl ModelArchitecture for Llama4Stage {
    fn forward(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        let mut hidden_states = match input {
            StageInput::Tokens(tokens) => {
                anyhow::ensure!(self.is_head, "only head stage can receive tokens");
                let embedding = self.embedding.as_ref().unwrap();
                let token_tensor = Tensor::new(tokens.as_slice(), &self.device)?;
                embedding
                    .index_select(&token_tensor, 0)?
                    .reshape((1, seq_len, self.hidden_dim))?
            }
            StageInput::Activations { tensor, .. } => {
                if tensor.dims().len() == 2 {
                    tensor.unsqueeze(0)?
                } else {
                    tensor
                }
            }
        };

        let mut next_attn_normed = None;
        let mut tail_normed = None;
        for (i, block) in self.blocks.iter().enumerate() {
            let normed = if let Some(normed) = next_attn_normed.take() {
                normed
            } else {
                block.attn_norm().forward(&hidden_states)?
            };
            let attn_out = match block {
                Llama4Block::Dense(block) => with_kv_cache(&mut cache[i], |kv| {
                    block.attn.forward(&normed, kv, seq_len, start_pos)
                })?,
                Llama4Block::Moe(block) => with_kv_cache(&mut cache[i], |kv| {
                    block.attn.forward(&normed, kv, seq_len, start_pos)
                })?,
            };
            let (next_hidden_states, normed) = block
                .ffn_norm()
                .forward_residual(&hidden_states, &attn_out)?;
            hidden_states = next_hidden_states;
            let ffn_out = match block {
                Llama4Block::Dense(block) => block.ffn.forward(&normed)?,
                Llama4Block::Moe(block) => block.moe.forward(&normed)?,
            };

            if let Some(next_block) = self.blocks.get(i + 1) {
                let (next_hidden_states, normed) = next_block
                    .attn_norm()
                    .forward_residual(&hidden_states, &ffn_out)?;
                hidden_states = next_hidden_states;
                next_attn_normed = Some(normed);
            } else if self.is_tail {
                let (next_hidden_states, normed) = self
                    .output_norm
                    .as_ref()
                    .unwrap()
                    .forward_residual(&hidden_states, &ffn_out)?;
                hidden_states = next_hidden_states;
                tail_normed = Some(normed);
            } else {
                hidden_states = (hidden_states + ffn_out)?;
            }
        }

        if self.is_tail {
            let normed = if let Some(normed) = tail_normed {
                normed
            } else {
                self.output_norm.as_ref().unwrap().forward(&hidden_states)?
            };
            let last_hidden = if seq_len > 1 {
                normed.i((.., seq_len - 1, ..))?
            } else {
                normed.squeeze(1)?
            };
            #[cfg(feature = "cuda")]
            let logits = match super::dense_qmatmul_cuda::forward_one_to_f32(
                &last_hidden,
                self.lm_head.as_ref().unwrap().quantized(),
            )? {
                Some(logits) => logits,
                None => self.lm_head.as_ref().unwrap().forward(&last_hidden)?,
            };
            #[cfg(not(feature = "cuda"))]
            let logits = self.lm_head.as_ref().unwrap().forward(&last_hidden)?;
            Ok(StageOutput::Logits(logits.squeeze(0)?))
        } else {
            Ok(StageOutput::Activations(hidden_states.squeeze(0)?))
        }
    }

    fn embed_tokens(&self, tokens: &[u32]) -> Result<Tensor> {
        anyhow::ensure!(self.is_head, "only a Llama 4 head stage can embed tokens");
        anyhow::ensure!(!tokens.is_empty(), "cannot embed an empty token list");
        let token_tensor = Tensor::new(tokens, &self.device)?;
        self.embedding
            .as_ref()
            .context("Llama 4 head stage has no token embedding")?
            .index_select(&token_tensor, 0)
            .map_err(Into::into)
    }

    fn family(&self) -> &str {
        "llama4"
    }
    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }
    fn layer_count(&self) -> usize {
        self.num_layers
    }
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

// ===== Builder Functions =====

fn build_attention(
    tensors: &HashMap<String, Arc<QTensor>>,
    prefix: &str,
    md: &crate::gguf_loader::GgufMetadata,
    rope: (&Tensor, &Tensor, Option<&Tensor>),
    layer_idx: usize,
    quant_config: Option<Arc<TurboQuantConfig>>,
) -> Result<Attention> {
    let (rope_cos, rope_sin, rope_freqs) = rope;
    let get = |name: &str| -> Result<QMatMul> {
        let t = tensors
            .get(name)
            .with_context(|| format!("missing tensor: {name}"))?;
        qtensor_to_qmatmul(Arc::clone(t))
    };
    let qk_norm = if md.expert_count != 128 {
        Some(RmsNorm {
            weight: Tensor::ones(md.head_dim, DType::F32, rope_cos.device())?,
            eps: md.rms_norm_eps,
        })
    } else {
        None
    };

    let chunk_size = if md.sliding_window > 0 && md.sliding_window_pattern[layer_idx] {
        md.sliding_window
    } else {
        0
    };
    Ok(Attention {
        q_proj: get(&format!("{prefix}.attn_q.weight"))?,
        k_proj: get(&format!("{prefix}.attn_k.weight"))?,
        v_proj: DecodeSafeQMatMul::from_qtensor(
            Arc::clone(
                tensors
                    .get(&format!("{prefix}.attn_v.weight"))
                    .with_context(|| format!("missing tensor: {prefix}.attn_v.weight"))?,
            ),
            rope_cos.device(),
        )?,
        o_proj: get(&format!("{prefix}.attn_output.weight"))?,
        num_heads: md.num_attention_heads,
        num_kv_heads: md.num_kv_heads,
        head_dim: md.head_dim,
        rope_cos: rope_cos.clone(),
        rope_sin: rope_sin.clone(),
        rope_theta: if chunk_size > 0 && md.rope_theta_swa != 0.0 {
            md.rope_theta_swa
        } else {
            md.rope_theta
        },
        rope_freqs: rope_freqs.cloned(),
        use_rope: uses_rope(layer_idx, md.sliding_window),
        chunk_size,
        qk_norm,
        quant_config,
    })
}

fn build_mlp(tensors: &HashMap<String, Arc<QTensor>>, prefix: &str) -> Result<Mlp> {
    let get = |name: &str| -> Result<QMatMul> {
        let t = tensors
            .get(name)
            .with_context(|| format!("missing tensor: {name}"))?;
        qtensor_to_qmatmul(Arc::clone(t))
    };
    Ok(Mlp {
        gate_proj: get(&format!("{prefix}.ffn_gate.weight"))?,
        up_proj: get(&format!("{prefix}.ffn_up.weight"))?,
        down_proj: get(&format!("{prefix}.ffn_down.weight"))?,
    })
}

fn build_moe_layer(
    tensors: &HashMap<String, Arc<QTensor>>,
    prefix: &str,
    layer_idx: usize,
    md: &crate::gguf_loader::GgufMetadata,
    device: &Device,
) -> Result<MoeLayer> {
    let gate_tensor = tensors
        .get(&format!("{prefix}.ffn_gate_inp.weight"))
        .context("missing ffn_gate_inp.weight (MoE router)")?;
    let gate = gate_tensor.as_ref().dequantize(device)?;

    let packed_experts = if let (Some(gate_exps), Some(up_exps), Some(down_exps)) = (
        tensors.get(&format!("{prefix}.ffn_gate_exps.weight")),
        tensors.get(&format!("{prefix}.ffn_up_exps.weight")),
        tensors.get(&format!("{prefix}.ffn_down_exps.weight")),
    ) {
        Some((gate_exps, up_exps, down_exps))
    } else {
        None
    };

    let split_experts = || -> Result<Vec<MoeExpert>> {
        if let Some((gate_exps, up_exps, down_exps)) = packed_experts {
            let gate_mats = split_expert_tensor(
                gate_exps.as_ref(),
                md.expert_count,
                md.expert_ffn_dim,
                md.hidden_dim,
                device,
            )
            .with_context(|| format!("{prefix}.ffn_gate_exps split"))?;
            let up_mats = split_expert_tensor(
                up_exps.as_ref(),
                md.expert_count,
                md.expert_ffn_dim,
                md.hidden_dim,
                device,
            )
            .with_context(|| format!("{prefix}.ffn_up_exps split"))?;
            let down_mats = split_expert_tensor(
                down_exps.as_ref(),
                md.expert_count,
                md.hidden_dim,
                md.expert_ffn_dim,
                device,
            )
            .with_context(|| format!("{prefix}.ffn_down_exps split"))?;
            return Ok(gate_mats
                .into_iter()
                .zip(up_mats)
                .zip(down_mats)
                .map(|((gate_proj, up_proj), down_proj)| MoeExpert {
                    gate_proj,
                    up_proj,
                    down_proj,
                })
                .collect());
        }

        let mut split = Vec::with_capacity(md.expert_count);
        for expert in 0..md.expert_count {
            let get = |name: &str| -> Result<QMatMul> {
                let tensor = tensors
                    .get(name)
                    .with_context(|| format!("missing tensor: {name}"))?;
                qtensor_to_qmatmul(Arc::clone(tensor))
            };
            split.push(MoeExpert {
                gate_proj: get(&format!("{prefix}.ffn_gate.{expert}.weight"))?,
                up_proj: get(&format!("{prefix}.ffn_up.{expert}.weight"))?,
                down_proj: get(&format!("{prefix}.ffn_down.{expert}.weight"))?,
            });
        }
        Ok(split)
    };

    #[cfg(any(target_os = "macos", feature = "cuda"))]
    let experts = if let Some((gate_exps, up_exps, down_exps)) = packed_experts {
        let use_indexed = md.expert_used_count.max(1) == 1
            && indexed_moe_available(device, gate_exps.dtype())
            && indexed_moe_available(device, up_exps.dtype())
            && indexed_moe_available(device, down_exps.dtype());
        if use_indexed {
            MoeExperts::Indexed(Box::new(IndexedMoeExperts {
                gate_proj: IndexedMoeMatMul::from_qtensor(
                    gate_exps.as_ref(),
                    md.expert_count,
                    md.expert_ffn_dim,
                    md.hidden_dim,
                    device,
                )?,
                up_proj: IndexedMoeMatMul::from_qtensor(
                    up_exps.as_ref(),
                    md.expert_count,
                    md.expert_ffn_dim,
                    md.hidden_dim,
                    device,
                )?,
                down_proj: IndexedMoeMatMul::from_qtensor(
                    down_exps.as_ref(),
                    md.expert_count,
                    md.hidden_dim,
                    md.expert_ffn_dim,
                    device,
                )?,
            }))
        } else {
            MoeExperts::Split(split_experts()?)
        }
    } else {
        MoeExperts::Split(split_experts()?)
    };
    #[cfg(not(any(target_os = "macos", feature = "cuda")))]
    let experts = MoeExperts::Split(split_experts()?);

    // Shared expert
    let shared_expert = if tensors.contains_key(&format!("{prefix}.ffn_gate_shexp.weight")) {
        let get = |name: &str| -> Result<QMatMul> {
            let t = tensors
                .get(name)
                .with_context(|| format!("missing shared expert tensor: {name}"))?;
            qtensor_to_qmatmul(Arc::clone(t))
        };
        Some(Mlp {
            gate_proj: get(&format!("{prefix}.ffn_gate_shexp.weight"))?,
            up_proj: get(&format!("{prefix}.ffn_up_shexp.weight"))?,
            down_proj: get(&format!("{prefix}.ffn_down_shexp.weight"))?,
        })
    } else {
        None
    };

    Ok(MoeLayer {
        gate,
        experts,
        shared_expert,
        num_active: md.expert_used_count.max(1), // at least top-1
        layer_idx,
    })
}

fn load_rms_norm(
    tensors: &HashMap<String, Arc<QTensor>>,
    prefix: &str,
    eps: f64,
    device: &Device,
) -> Result<RmsNorm> {
    let weight = tensors
        .get(&format!("{prefix}.weight"))
        .with_context(|| format!("missing {prefix}.weight for RmsNorm"))?
        .as_ref()
        .dequantize(device)?;
    Ok(RmsNorm { weight, eps })
}

#[cfg(test)]
#[path = "llama4_attention_tests.rs"]
mod attention_tests;

#[cfg(test)]
mod tests {
    use crate::architectures::needs_metal_single_token_fallback;
    use candle_core::quantized::GgmlDType;
    use candle_core::{Device, Tensor};

    use super::{apply_rotary_emb, is_moe_layer, precompute_rope_cos_sin};

    #[test]
    fn metal_single_token_fallback_is_limited_to_affected_k_quants() {
        assert!(needs_metal_single_token_fallback(GgmlDType::Q4K));
        assert!(needs_metal_single_token_fallback(GgmlDType::Q6K));
        assert!(!needs_metal_single_token_fallback(GgmlDType::Q2K));
        assert!(!needs_metal_single_token_fallback(GgmlDType::Q8_0));
    }

    #[test]
    fn test_llama4_all_moe_layers() {
        // An all-MoE configuration: 48 layers, first_k_dense=0, 16 experts.
        let first_k_dense = 0;
        let moe_step = 1;
        let expert_count = 16;

        for layer in 0..48 {
            let is_moe = is_moe_layer(layer, first_k_dense, moe_step, expert_count > 0);
            assert!(is_moe, "all layers should be MoE when first_k_dense=0");
        }
    }

    #[test]
    fn test_llama4_alternating_layers() {
        // Hypothetical: first_k_dense=0, moe_step=2 → alternating dense/MoE
        let first_k_dense = 0;
        let moe_step = 2;

        let is_moe = |layer| is_moe_layer(layer, first_k_dense, moe_step, true);

        assert!(!is_moe(0));
        assert!(is_moe(1));
        assert!(!is_moe(2));
        assert!(is_moe(3));
    }

    #[test]
    fn dynamic_rope_matches_full_cache() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let head_dim = 8;
        let theta = 500_000.0;
        let values = (0..48)
            .map(|index| index as f32 / 17.0 - 1.0)
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(values, (1, 2, 3, head_dim), &device)?;
        let (small_cos, small_sin) = precompute_rope_cos_sin(head_dim, theta, 2, &device, None)?;
        let (full_cos, full_sin) = precompute_rope_cos_sin(head_dim, theta, 8, &device, None)?;

        let actual = apply_rotary_emb(&input, 3, &small_cos, &small_sin, theta, None)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = apply_rotary_emb(&input, 3, &full_cos, &full_sin, theta, None)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((*actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        Ok(())
    }
}
