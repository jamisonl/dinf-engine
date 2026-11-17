//! MiniMax M2.5 architecture — GQA with Q/K norms + MoE with packed experts.
//!
//! All layers are MoE (no dense layers). Uses packed expert tensors (ffn_gate_exps,
//! ffn_down_exps, ffn_up_exps) which need to be split per-expert. Expert probability
//! bias (ffn_exp_probs_b) affects routing selection.
//!
//! Q/K normalization applied after projection, before RoPE.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::qmatmul::QMatMul;
use anyhow::{Context, Result};
use candle_core::quantized::QTensor;
use candle_core::{D, DType, Device, IndexOp, Module, Tensor};

#[cfg(target_os = "macos")]
use super::fused_norm_metal::try_fused_add_rms_norm;
#[cfg(target_os = "macos")]
use super::fused_route_metal::fused_sigmoid_topk_route;
#[cfg(any(target_os = "macos", feature = "cuda"))]
use super::indexed_moe::{
    IndexedMoeMatMul, indexed_moe_available, normalized_route_weights, silu_mul, weighted_route_sum,
};
use super::{
    DecodeSafeQMatMul, KvCache, LayerCache, ModelArchitecture, StageInput, StageOutput,
    with_kv_cache,
};
use crate::gguf_loader::{GgufMetadata, LayerRange, StageTensorLoader, StageTensors};
use crate::turboquant::TurboQuantConfig;

// ===== Helpers =====

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(x, &self.weight, self.eps as f32).map_err(Into::into)
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

fn qtensor_to_qmatmul(qt: Arc<QTensor>) -> Result<QMatMul> {
    QMatMul::from_arc(qt).map_err(Into::into)
}

fn precompute_rope_cos_sin(
    rope_dim: usize,
    theta: f64,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half_dim = rope_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / rope_dim as f32))
        .collect();
    let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
    let positions = Tensor::arange(0u32, context_length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?;
    let angles = positions.matmul(&inv_freq.reshape((1, half_dim))?)?;
    Ok((angles.cos()?, angles.sin()?))
}

fn apply_partial_rotary_emb(
    x: &Tensor,
    start_pos: usize,
    cos: &Tensor,
    sin: &Tensor,
    rope_dim: usize,
) -> Result<Tensor> {
    let (_, _, seq_len, head_dim) = x.dims4()?;
    anyhow::ensure!(
        rope_dim > 0 && rope_dim <= head_dim && rope_dim.is_multiple_of(2),
        "invalid MiniMax RoPE dimension {rope_dim} for head dimension {head_dim}"
    );
    let cos = cos.narrow(0, start_pos, seq_len)?;
    let sin = sin.narrow(0, start_pos, seq_len)?;
    if rope_dim == head_dim {
        return Ok(candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)?);
    }

    let rotated =
        candle_nn::rotary_emb::rope(&x.narrow(D::Minus1, 0, rope_dim)?.contiguous()?, &cos, &sin)?;
    let passthrough = x
        .narrow(D::Minus1, rope_dim, head_dim - rope_dim)?
        .contiguous()?;
    Tensor::cat(&[&rotated, &passthrough], D::Minus1).map_err(Into::into)
}

fn apply_partial_rotary_emb_ragged(
    q: &Tensor,
    k: &Tensor,
    positions: &[usize],
    cos: &Tensor,
    sin: &Tensor,
    rope_dim: usize,
) -> Result<(Tensor, Tensor)> {
    let (batch, _, q_seq_len, q_head_dim) = q.dims4()?;
    let (k_batch, _, k_seq_len, k_head_dim) = k.dims4()?;
    anyhow::ensure!(
        batch == k_batch && q_seq_len == 1 && k_seq_len == 1,
        "ragged MiniMax RoPE is decode-only"
    );
    anyhow::ensure!(
        positions.len() == batch,
        "ragged MiniMax RoPE position mismatch"
    );
    anyhow::ensure!(
        q_head_dim == k_head_dim && rope_dim <= q_head_dim,
        "ragged MiniMax RoPE dimension mismatch"
    );

    let (rotated_q, rotated_k) =
        if let Some(rotated) = super::try_metal_ragged_rope(q, k, cos, sin, positions, true)? {
            rotated
        } else {
            let mut q_lanes = Vec::with_capacity(batch);
            let mut k_lanes = Vec::with_capacity(batch);
            for (lane, position) in positions.iter().copied().enumerate() {
                q_lanes.push(apply_partial_rotary_emb(
                    &q.narrow(0, lane, 1)?,
                    position,
                    cos,
                    sin,
                    rope_dim,
                )?);
                k_lanes.push(apply_partial_rotary_emb(
                    &k.narrow(0, lane, 1)?,
                    position,
                    cos,
                    sin,
                    rope_dim,
                )?);
            }
            let q_refs = q_lanes.iter().collect::<Vec<_>>();
            let k_refs = k_lanes.iter().collect::<Vec<_>>();
            return Ok((Tensor::cat(&q_refs, 0)?, Tensor::cat(&k_refs, 0)?));
        };

    Ok((rotated_q, rotated_k))
}

fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::softmax_last_dim(x).map_err(Into::into)
}

fn cpu_causal_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let query_len = q.dim(2)?;
    let key_len = k.dim(2)?;
    anyhow::ensure!(
        query_len > 0 && key_len >= query_len,
        "invalid causal attention lengths"
    );
    // Masked future keys still change the CPU softmax's SIMD reduction width.
    // Use each query's visible prefix so full, chunked, and scalar prefill use
    // the same arithmetic. Small differences otherwise alter later MoE routes.
    let mut rows = Vec::with_capacity(query_len);
    for row in 0..query_len {
        let visible = key_len - query_len + row + 1;
        let query = q.narrow(2, row, 1)?.contiguous()?;
        let keys = k.narrow(2, 0, visible)?.t()?.contiguous()?;
        let values = v.narrow(2, 0, visible)?.contiguous()?;
        let scores = (query.matmul(&keys)? * scale)?;
        rows.push(softmax_last_dim(&scores)?.matmul(&values)?);
    }
    Tensor::cat(&rows, 2).map_err(Into::into)
}

fn sigmoid_router_probs(logits: &Tensor) -> Result<Tensor> {
    candle_nn::Activation::Sigmoid
        .forward(logits)
        .map_err(Into::into)
}

fn project_router(input: &Tensor, gate: &Tensor, decode: bool) -> Result<Tensor> {
    let (rows, _) = input.dims2()?;
    let gate = gate.t()?;
    if rows > 1 && (input.device().is_cpu() || (decode && input.device().is_metal())) {
        // Keep each request on the scalar router's F32 reduction path. A
        // flattened batch selects GEMM instead of GEMV; its low-bit changes
        // can alter expert selection across MiniMax's 62 layers. CPU prompt
        // chunks also need this rule, including a one-token final chunk.
        let lanes = (0..rows)
            .map(|lane| input.narrow(0, lane, 1)?.matmul(&gate))
            .collect::<candle_core::Result<Vec<_>>>()?;
        return Tensor::cat(&lanes, 0).map_err(Into::into);
    }
    if !decode && rows == 1 && input.device().is_metal() {
        // Keep a one-token prompt tail on the same matrix reduction as the
        // rest of the prompt. This duplicate never enters the KV cache.
        return Tensor::cat(&[input, input], 0)?
            .matmul(&gate)?
            .narrow(0, 0, 1)?
            .contiguous()
            .map_err(Into::into);
    }
    input.matmul(&gate).map_err(Into::into)
}

fn prompt_tail_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Option<Tensor>> {
    if k.dim(2)? == 1 {
        // With one key, softmax is exactly one for every query head.
        return Ok(Some(repeat_kv(v.clone(), q.dim(1)? / k.dim(1)?)?));
    }
    // Two queries select prompt attention math. Keep the last query, whose
    // bottom-right causal row sees every real key. Never duplicate cached K/V.
    let queries = Tensor::cat(&[q, q], 2)?;
    super::try_metal_f16_sdpa_native(&queries, k, v, true, scale)?
        .map(|output| output.narrow(2, 1, 1)?.contiguous())
        .transpose()
        .map_err(Into::into)
}

static ATTN_NORM_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_RESIDUAL_NS: AtomicU64 = AtomicU64::new(0);
static FFN_NORM_NS: AtomicU64 = AtomicU64::new(0);
static ROUTER_NS: AtomicU64 = AtomicU64::new(0);
static EXPERT_NS: AtomicU64 = AtomicU64::new(0);
static FFN_RESIDUAL_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_QKV_PROJ_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_QK_NORM_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_ROPE_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_KV_UPDATE_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_SDPA_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_O_PROJ_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_CALLS: AtomicU64 = AtomicU64::new(0);

fn layer_profile_enabled() -> bool {
    super::profiling::layer_profiling_enabled()
}

fn finish_profile_region(device: &Device, started: Instant, counter: &AtomicU64) -> Result<()> {
    device.synchronize()?;
    counter.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(())
}

/// Reset the process-local MiniMax layer profiling counters.
pub fn reset_layer_profile() {
    for counter in [
        &ATTN_NORM_NS,
        &ATTN_NS,
        &ATTN_RESIDUAL_NS,
        &FFN_NORM_NS,
        &ROUTER_NS,
        &EXPERT_NS,
        &FFN_RESIDUAL_NS,
        &ATTN_QKV_PROJ_NS,
        &ATTN_QK_NORM_NS,
        &ATTN_ROPE_NS,
        &ATTN_KV_UPDATE_NS,
        &ATTN_SDPA_NS,
        &ATTN_O_PROJ_NS,
        &PROFILE_CALLS,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

/// Print the process-local MiniMax layer profiling counters to stderr.
pub fn report_layer_profile() {
    let calls = PROFILE_CALLS.load(Ordering::Relaxed).max(1) as f64;
    let ms_per_call =
        |counter: &AtomicU64| counter.load(Ordering::Relaxed) as f64 / calls / 1_000_000.0;
    eprintln!(
        "DINF_MINIMAX_PROFILE calls={} per_layer_ms={{attn_norm:{:.4},attention:{:.4},attn_residual:{:.4},ffn_norm:{:.4},router:{:.4},experts:{:.4},ffn_residual:{:.4}}}",
        PROFILE_CALLS.load(Ordering::Relaxed),
        ms_per_call(&ATTN_NORM_NS),
        ms_per_call(&ATTN_NS),
        ms_per_call(&ATTN_RESIDUAL_NS),
        ms_per_call(&FFN_NORM_NS),
        ms_per_call(&ROUTER_NS),
        ms_per_call(&EXPERT_NS),
        ms_per_call(&FFN_RESIDUAL_NS),
    );
    eprintln!(
        "DINF_MINIMAX_ATTN_PROFILE per_layer_ms={{qkv_proj:{:.4},qk_norm:{:.4},rope:{:.4},kv_update:{:.4},sdpa:{:.4},o_proj:{:.4}}}",
        ms_per_call(&ATTN_QKV_PROJ_NS),
        ms_per_call(&ATTN_QK_NORM_NS),
        ms_per_call(&ATTN_ROPE_NS),
        ms_per_call(&ATTN_KV_UPDATE_NS),
        ms_per_call(&ATTN_SDPA_NS),
        ms_per_call(&ATTN_O_PROJ_NS),
    );
}

fn expert_input_with_m1_padding(x: &Tensor, pad_single_token: bool) -> Result<(Tensor, bool)> {
    if pad_single_token && x.dim(D::Minus2)? == 1 {
        // Candle routes m=1 to Metal's quantized GEMV, which is unstable for
        // MiniMax's Q2_K/Q3_K expert shapes. Pad once and carry both rows
        // through all three projections instead of padding every matmul.
        return Ok((Tensor::cat(&[x, x], D::Minus2)?, true));
    }
    Ok((x.clone(), false))
}

fn create_causal_mask(seq_len: usize, total_seq_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..total_seq_len).map(move |j| {
                if j <= total_seq_len - seq_len + i {
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
    let (b, h, s, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, h, n_rep, s, d))?
        .reshape((b, h * n_rep, s, d))?
        .contiguous()
        .map_err(Into::into)
}

// ===== Attention with Q/K norms =====

struct Attention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: DecodeSafeQMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    rope_cos: Tensor,
    rope_sin: Tensor,
    quant_config: Option<Arc<TurboQuantConfig>>,
}

impl Attention {
    fn forward(
        &self,
        x: &Tensor,
        kv_cache: &mut Option<KvCache>,
        seq_len: usize,
        start_pos: usize,
        prefill: bool,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let matrix_tail = prefill && seq_len == 1 && x.device().is_metal();
        let n_rep = self.num_heads / self.num_kv_heads;
        let profile = seq_len == 1 && layer_profile_enabled();

        #[cfg(feature = "cuda")]
        let grouped_qkv = if seq_len == 1 && !profile {
            super::dense_qmatmul_cuda::forward(
                x,
                &[&self.q_proj, &self.k_proj, self.v_proj.quantized()],
            )?
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let grouped_qkv: Option<Vec<Tensor>> = None;

        // MiniMax normalizes the full Q/K projection before splitting heads.
        let (q_raw, k_raw, v) = if matrix_tail {
            let rows = Tensor::cat(&[x, x], 1)?;
            (
                self.q_proj.forward(&rows)?.narrow(1, 0, 1)?.contiguous()?,
                self.k_proj.forward(&rows)?.narrow(1, 0, 1)?.contiguous()?,
                self.v_proj.forward(&rows)?.narrow(1, 0, 1)?.contiguous()?,
            )
        } else if let Some(mut outputs) = grouped_qkv {
            anyhow::ensure!(outputs.len() == 3, "grouped MiniMax QKV count mismatch");
            let v = outputs.pop().unwrap();
            let k = outputs.pop().unwrap();
            let q = outputs.pop().unwrap();
            (q, k, v)
        } else if profile {
            let started = Instant::now();
            let q = self.q_proj.forward(x)?;
            let k = self.k_proj.forward(x)?;
            let v = self.v_proj.forward(x)?;
            finish_profile_region(x.device(), started, &ATTN_QKV_PROJ_NS)?;
            (q, k, v)
        } else {
            (
                self.q_proj.forward(x)?,
                self.k_proj.forward(x)?,
                self.v_proj.forward(x)?,
            )
        };
        let started = profile.then(Instant::now);
        let q = self.q_norm.forward(&q_raw)?;
        let k = self.k_norm.forward(&k_raw)?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &ATTN_QK_NORM_NS)?;
        }
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

        let started = profile.then(Instant::now);
        let (q, k) = if seq_len == 1 && !matrix_tail {
            let positions = vec![start_pos; b];
            if let Some(rotated) = super::try_metal_ragged_rope(
                &q,
                &k,
                &self.rope_cos,
                &self.rope_sin,
                &positions,
                true,
            )? {
                rotated
            } else {
                (
                    apply_partial_rotary_emb(
                        &q,
                        start_pos,
                        &self.rope_cos,
                        &self.rope_sin,
                        self.rope_dim,
                    )?,
                    apply_partial_rotary_emb(
                        &k,
                        start_pos,
                        &self.rope_cos,
                        &self.rope_sin,
                        self.rope_dim,
                    )?,
                )
            }
        } else {
            (
                apply_partial_rotary_emb(
                    &q,
                    start_pos,
                    &self.rope_cos,
                    &self.rope_sin,
                    self.rope_dim,
                )?,
                apply_partial_rotary_emb(
                    &k,
                    start_pos,
                    &self.rope_cos,
                    &self.rope_sin,
                    self.rope_dim,
                )?,
            )
        };
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &ATTN_ROPE_NS)?;
        }

        let started = profile.then(Instant::now);
        let (q, k, v, rotation) =
            super::prepare_quantized_attention(kv_cache, q, k, v, self.quant_config.as_deref())?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &ATTN_KV_UPDATE_NS)?;
        }
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let started = profile.then(Instant::now);
        let tail_attention = if matrix_tail {
            prompt_tail_attention(&q, &k, &v, scale as f32)?
        } else {
            None
        };
        let out = if let Some(output) = tail_attention {
            output
        } else if let Some(output) =
            super::try_fused_f16_sdpa_native(&q, &k, &v, seq_len > 1, scale as f32)?
        {
            output
        } else {
            let k = repeat_kv(k.to_dtype(q.dtype())?, n_rep)?;
            let v = repeat_kv(v.to_dtype(q.dtype())?, n_rep)?;
            if q.device().is_cpu() {
                cpu_causal_attention(&q, &k, &v, scale)?
            } else {
                let att = (q.contiguous()?.matmul(&k.t()?.contiguous()?)? * scale)?;
                let att = if seq_len > 1 {
                    att.broadcast_add(&create_causal_mask(seq_len, att.dim(3)?, x.device())?)?
                } else {
                    att
                };
                softmax_last_dim(&att)?.matmul(&v.contiguous()?)?
            }
        };
        let out = super::restore_quantized_attention_output(out, rotation)?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &ATTN_SDPA_NS)?;
        }
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
        if !profile
            && let Some(output) = super::dense_qmatmul_cuda::forward_one_to_f32(&out, &self.o_proj)?
        {
            return Ok(output);
        }
        #[cfg(not(target_os = "macos"))]
        let out = out.to_dtype(x.dtype())?;
        let started = profile.then(Instant::now);
        let output = if matrix_tail {
            self.o_proj
                .forward(&Tensor::cat(&[&out, &out], 1)?)?
                .narrow(1, 0, 1)?
                .contiguous()?
        } else {
            self.o_proj.forward(&out)?
        };
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &ATTN_O_PROJ_NS)?;
        }
        Ok(output)
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

    fn forward_packed_ragged_decode(
        &self,
        x: &Tensor,
        cache: &mut Option<KvCache>,
        positions: &[usize],
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        anyhow::ensure!(seq_len == 1, "ragged attention is decode-only");
        anyhow::ensure!(
            batch == positions.len(),
            "ragged attention position mismatch"
        );

        let q = self.q_norm.forward(&self.q_proj.forward(x)?)?;
        let k = self.k_norm.forward(&self.k_proj.forward(x)?)?;
        let q = q.reshape((batch, self.num_heads, 1, self.head_dim))?;
        let k = k.reshape((batch, self.num_kv_heads, 1, self.head_dim))?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((batch, self.num_kv_heads, 1, self.head_dim))?;
        let (q, k) = apply_partial_rotary_emb_ragged(
            &q,
            &k,
            positions,
            &self.rope_cos,
            &self.rope_sin,
            self.rope_dim,
        )?;

        let Some(KvCache::PreallocatedF16(dense)) = cache.as_mut() else {
            anyhow::bail!("packed ragged decode requires preallocated f16 KV storage")
        };
        dense.append_ragged(&k, &v, positions)?;
        let valid_lengths = positions
            .iter()
            .map(|position| position.saturating_add(1))
            .collect::<Vec<_>>();
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let out = if let Some(output) =
            super::try_fused_f16_ragged_sdpa(&q, &dense.k, &dense.v, &valid_lengths, scale as f32)?
        {
            output
        } else {
            let n_rep = self.num_heads / self.num_kv_heads;
            let mut outputs = Vec::with_capacity(batch);
            for (lane, valid_len) in valid_lengths.iter().copied().enumerate() {
                let lane_q = q.narrow(0, lane, 1)?;
                let lane_k = dense.k.narrow(0, lane, 1)?.narrow(2, 0, valid_len)?;
                let lane_v = dense.v.narrow(0, lane, 1)?.narrow(2, 0, valid_len)?;
                if let Some(output) =
                    super::try_fused_f16_sdpa(&lane_q, &lane_k, &lane_v, false, scale as f32)?
                {
                    outputs.push(output);
                } else {
                    let lane_k = repeat_kv(lane_k.to_dtype(lane_q.dtype())?, n_rep)?;
                    let lane_v = repeat_kv(lane_v.to_dtype(lane_q.dtype())?, n_rep)?;
                    let weights = (lane_q.matmul(&lane_k.transpose(2, 3)?)? * scale)?;
                    outputs.push(softmax_last_dim(&weights)?.matmul(&lane_v)?);
                }
            }
            let output_refs = outputs.iter().collect::<Vec<_>>();
            Tensor::cat(&output_refs, 0)?
        };
        let out = out.transpose(1, 2)?.contiguous()?.reshape((
            batch,
            1,
            self.num_heads * self.head_dim,
        ))?;
        let output = self.o_proj.forward(&out)?;
        Ok(output)
    }
}

// ===== MoE with packed experts + bias =====

struct MoeExpert {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

impl MoeExpert {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (x, padded) = expert_input_with_m1_padding(x, x.device().is_metal())?;
        let gate = self.gate_proj.forward(&x)?;
        let up = self.up_proj.forward(&x)?;
        let activated = (candle_nn::Activation::Silu.forward(&gate)? * up)?;
        let output = self.down_proj.forward(&activated)?;
        if padded {
            output.narrow(D::Minus2, 0, 1).map_err(Into::into)
        } else {
            Ok(output)
        }
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
    const MAX_ROUTE_ROWS_PER_DISPATCH: usize = 4096;

    fn forward(
        &self,
        flat_x: &Tensor,
        expert_ids: &Tensor,
        weights: &Tensor,
        prefill: bool,
    ) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = flat_x.dims2()?;
        let (id_tokens, top_k) = expert_ids.dims2()?;
        anyhow::ensure!(
            num_tokens == id_tokens && weights.dims2()? == (num_tokens, top_k),
            "indexed MiniMax routing tensor geometry differs"
        );
        let chunk_tokens = (Self::MAX_ROUTE_ROWS_PER_DISPATCH / top_k).max(1);
        let mut outputs = Vec::with_capacity(num_tokens.div_ceil(chunk_tokens));
        for start in (0..num_tokens).step_by(chunk_tokens) {
            let len = (num_tokens - start).min(chunk_tokens);
            let input = flat_x
                .narrow(0, start, len)?
                .to_dtype(DType::F32)?
                .reshape((len, 1, hidden_dim))?;
            let ids = expert_ids.narrow(0, start, len)?;
            let route_weights = weights.narrow(0, start, len)?;
            let project = |op: &IndexedMoeMatMul, input: &Tensor| {
                if prefill {
                    op.forward_prefill(input, &ids)
                } else {
                    op.forward(input, &ids)
                }
            };
            let gate = project(&self.gate_proj, &input)?;
            let up = project(&self.up_proj, &input)?;
            let activated = silu_mul(&gate, &up)?;
            let output = project(&self.down_proj, &activated)?;
            outputs.push(weighted_route_sum(&output, &route_weights)?);
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
    expert_bias: Option<Tensor>,
    num_active: usize,
    layer_idx: usize,
}

impl MoeLayer {
    fn selected_experts(&self, routing: &[f32]) -> Result<Vec<(usize, f32)>> {
        let expert_count = self.experts.len();
        anyhow::ensure!(
            routing.len() == expert_count * 2,
            "MiniMax routing row has {} values, expected {}",
            routing.len(),
            expert_count * 2,
        );
        anyhow::ensure!(
            self.num_active <= expert_count,
            "MiniMax requests {} active experts from a pool of {}",
            self.num_active,
            expert_count,
        );
        let (scores, probs) = routing.split_at(expert_count);
        anyhow::ensure!(
            scores.iter().all(|value| value.is_finite())
                && probs.iter().all(|value| value.is_finite()),
            "non-finite MiniMax router values at layer {}",
            self.layer_idx,
        );
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut selected = indexed[..self.num_active]
            .iter()
            .map(|(expert_idx, _)| (*expert_idx, probs[*expert_idx]))
            .collect::<Vec<_>>();
        let weight_sum = selected.iter().map(|(_, weight)| weight).sum::<f32>();
        for (_, weight) in &mut selected {
            *weight /= weight_sum.max(6.103_515_6e-5);
        }
        Ok(selected)
    }

    fn forward(&self, x: &Tensor, prefill: bool) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_dim) = x.dims3()?;
        let num_tokens = b_sz * seq_len;
        let flat_x = x.reshape((num_tokens, hidden_dim))?;

        let profile = num_tokens == 1 && layer_profile_enabled();
        let router_started = Instant::now();
        let router_logits = project_router(&flat_x, &self.gate, seq_len == 1 && !prefill)?;

        #[cfg(target_os = "macos")]
        let fused_indexed_routing = match (&self.experts, &self.expert_bias) {
            (MoeExperts::Indexed(_), Some(bias)) => Some(fused_sigmoid_topk_route(
                &router_logits,
                bias,
                self.num_active,
            )?),
            _ => None,
        };
        #[cfg(not(target_os = "macos"))]
        let fused_indexed_routing: Option<(Tensor, Tensor)> = None;

        let router_probs = if fused_indexed_routing.is_some() {
            None
        } else {
            Some(sigmoid_router_probs(&router_logits)?)
        };

        // Apply expert bias to selection (bias affects which experts are chosen, not weights)
        let selection_scores = if let Some(router_probs) = router_probs.as_ref() {
            Some(if let Some(ref bias) = self.expert_bias {
                (router_probs.clone() + bias.broadcast_left(b_sz * seq_len)?)?
            } else {
                router_probs.clone()
            })
        } else {
            None
        };

        #[cfg(any(target_os = "macos", feature = "cuda"))]
        let indexed_routing = if let Some(routing) = fused_indexed_routing {
            Some(routing)
        } else {
            match &self.experts {
                MoeExperts::Indexed(_) => {
                    let selection_scores = selection_scores
                        .as_ref()
                        .context("missing MiniMax selection scores")?;
                    let router_probs = router_probs
                        .as_ref()
                        .context("missing MiniMax router probabilities")?;
                    let expert_ids = selection_scores
                        .contiguous()?
                        .arg_sort_last_dim(false)?
                        .narrow(D::Minus1, 0, self.num_active)?
                        .contiguous()?;
                    let expert_weights = normalized_route_weights(router_probs, &expert_ids)?;
                    Some((expert_ids, expert_weights))
                }
                MoeExperts::Split(_) => None,
            }
        };
        #[cfg(not(any(target_os = "macos", feature = "cuda")))]
        let indexed_routing: Option<(Tensor, Tensor)> = None;

        let routing = if indexed_routing.is_none() {
            // The split fallback executes assignments on the host. Read scores
            // and probabilities together so it still incurs only one sync.
            let selection_scores = selection_scores
                .as_ref()
                .context("missing MiniMax selection scores")?;
            let router_probs = router_probs
                .as_ref()
                .context("missing MiniMax router probabilities")?;
            Some(
                Tensor::cat(&[selection_scores, router_probs], D::Minus1)?
                    .to_dtype(DType::F32)?
                    .to_vec2::<f32>()?,
            )
        } else {
            None
        };
        if profile {
            finish_profile_region(x.device(), router_started, &ROUTER_NS)?;
        }

        let expert_started = Instant::now();
        let output = match &self.experts {
            #[cfg(any(target_os = "macos", feature = "cuda"))]
            MoeExperts::Indexed(experts) => {
                let cpu_routing;
                let (expert_ids, expert_weights) = if let Some(indexed) = indexed_routing.as_ref() {
                    (&indexed.0, &indexed.1)
                } else {
                    let routing = routing
                        .as_ref()
                        .context("missing host MiniMax routing values")?;
                    let mut ids = Vec::with_capacity(num_tokens * self.num_active);
                    let mut weights = Vec::with_capacity(num_tokens * self.num_active);
                    for row in routing {
                        for (expert_id, weight) in self.selected_experts(row)? {
                            ids.push(expert_id as u32);
                            weights.push(weight);
                        }
                    }
                    cpu_routing = (
                        Tensor::from_vec(ids, (num_tokens, self.num_active), flat_x.device())?,
                        Tensor::from_vec(weights, (num_tokens, self.num_active), flat_x.device())?,
                    );
                    (&cpu_routing.0, &cpu_routing.1)
                };
                experts.forward(&flat_x, expert_ids, expert_weights, prefill || seq_len > 1)?
            }
            MoeExperts::Split(experts) if num_tokens == 1 => {
                let routing = routing
                    .as_ref()
                    .context("missing split MiniMax routing values")?;
                let selected = self.selected_experts(&routing[0])?;
                let mut outputs = Vec::with_capacity(selected.len());
                for (expert_idx, weight) in selected {
                    let expert_out = experts[expert_idx].forward(&flat_x)?;
                    outputs.push((expert_out.to_dtype(DType::F32)? * weight as f64)?);
                }
                let output_refs = outputs.iter().collect::<Vec<_>>();
                Tensor::cat(&output_refs, 0)?.sum(0)?.unsqueeze(0)?
            }
            MoeExperts::Split(experts) => {
                let routing = routing
                    .as_ref()
                    .context("missing split MiniMax routing values")?;
                // Group assignments by expert. Prefill and continuous batches
                // run each selected expert once for all of its tokens.
                let mut assignments = vec![Vec::<(usize, usize, f32)>::new(); experts.len()];
                for (token_idx, row) in routing.iter().enumerate() {
                    for (route, (expert_idx, weight)) in
                        self.selected_experts(row)?.into_iter().enumerate()
                    {
                        assignments[expert_idx].push((token_idx, route, weight));
                    }
                }

                let mut routed_outputs = Vec::new();
                let mut routed_token_indices = Vec::new();
                for (expert_idx, expert_assignments) in assignments.iter().enumerate() {
                    if expert_assignments.is_empty() {
                        continue;
                    }
                    let token_indices = expert_assignments
                        .iter()
                        .map(|(token_idx, _, _)| *token_idx as u32)
                        .collect::<Vec<_>>();
                    let indices = Tensor::new(token_indices.as_slice(), x.device())?;
                    let expert_input = flat_x.index_select(&indices, 0)?;
                    let expert_out = experts[expert_idx].forward(&expert_input)?;
                    let weights = expert_assignments
                        .iter()
                        .map(|(_, _, weight)| *weight)
                        .collect::<Vec<_>>();
                    let weights =
                        Tensor::new(weights.as_slice(), x.device())?.reshape((weights.len(), 1))?;
                    routed_outputs.push(expert_out.to_dtype(DType::F32)?.broadcast_mul(&weights)?);
                    if x.device().is_cpu() {
                        // Restore score rank before summing, as in the scalar
                        // path. Expert-ID order changes F32 cancellation when
                        // a prompt chunk contains only one token.
                        routed_token_indices.extend(
                            expert_assignments
                                .iter()
                                .map(|(token, route, _)| (token * self.num_active + route) as u32),
                        );
                    } else {
                        routed_token_indices.extend(token_indices);
                    }
                }
                let output_refs = routed_outputs.iter().collect::<Vec<_>>();
                let routed_outputs = Tensor::cat(&output_refs, 0)?;
                let routed_token_indices =
                    Tensor::new(routed_token_indices.as_slice(), x.device())?;
                let output_rows = if x.device().is_cpu() {
                    num_tokens * self.num_active
                } else {
                    num_tokens
                };
                let output = Tensor::zeros((output_rows, hidden_dim), DType::F32, x.device())?
                    .index_add(&routed_token_indices, &routed_outputs, 0)?;
                if x.device().is_cpu() {
                    output
                        .reshape((num_tokens, self.num_active, hidden_dim))?
                        .sum(1)?
                } else {
                    output
                }
            }
        };
        if profile {
            finish_profile_region(x.device(), expert_started, &EXPERT_NS)?;
        }

        Ok(output
            .to_dtype(x.dtype())?
            .reshape((b_sz, seq_len, hidden_dim))?)
    }
}

// ===== MoE Block (all layers are MoE) =====

struct MoeBlock {
    attn_norm: RmsNorm,
    attn: Attention,
    ffn_norm: RmsNorm,
    moe: MoeLayer,
}

// ===== MiniMax M2 Stage =====

/// A loaded MiniMax M2/M2.5 stage for pipeline execution.
pub struct MiniMaxM2Stage {
    embedding: Option<Tensor>,
    blocks: Vec<MoeBlock>,
    output_norm: Option<RmsNorm>,
    lm_head: Option<DecodeSafeQMatMul>,
    is_head: bool,
    is_tail: bool,
    hidden_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    device: Device,
}

impl MiniMaxM2Stage {
    /// Load one layer's source tensors at a time, releasing them after upload.
    /// Share a single RoPE table across all layers, as in the whole-stage path.
    pub fn from_streaming_gguf(
        gguf_path: &Path,
        range: LayerRange,
        device: &Device,
        quant_config: Option<Arc<TurboQuantConfig>>,
    ) -> Result<Self> {
        range.validate()?;
        let mut loader = StageTensorLoader::open(gguf_path)?;
        let metadata = loader.metadata().clone();
        anyhow::ensure!(metadata.architecture == "minimax-m2", "not a MiniMax stage");
        anyhow::ensure!(
            range.total_layers == metadata.num_layers,
            "MiniMax layer count mismatch"
        );
        let (rope_cos, rope_sin) = Self::rope_tables(&metadata, device)?;
        let mut embedding = None;
        let mut blocks = Vec::with_capacity(range.layer_count());
        let mut output_norm = None;
        let mut lm_head = None;
        for layer in range.start..range.end {
            let tensors = loader
                .load_range(
                    LayerRange::try_new(layer, layer + 1, range.total_layers)?,
                    device,
                )
                .with_context(|| format!("failed to stream MiniMax layer {layer}"))?;
            let mut stage = Self::from_stage_tensors_with_rope(
                tensors,
                quant_config.clone(),
                &rope_cos,
                &rope_sin,
            )?;
            if stage.embedding.is_some() {
                embedding = stage.embedding.take();
            }
            blocks.append(&mut stage.blocks);
            if stage.output_norm.is_some() {
                output_norm = stage.output_norm.take();
            }
            if stage.lm_head.is_some() {
                lm_head = stage.lm_head.take();
            }
            drop(stage);
            device
                .synchronize()
                .with_context(|| format!("failed to finish streaming MiniMax layer {layer}"))?;
        }
        Ok(Self {
            embedding,
            blocks,
            output_norm,
            lm_head,
            is_head: range.is_head(),
            is_tail: range.is_tail(),
            hidden_dim: metadata.hidden_dim,
            num_layers: metadata.num_layers,
            vocab_size: metadata.vocab_size,
            device: device.clone(),
        })
    }

    fn rope_tables(md: &GgufMetadata, device: &Device) -> Result<(Tensor, Tensor)> {
        let rope_dim = if md.rope_dimension_count > 0 {
            md.rope_dimension_count
        } else {
            md.head_dim
        };
        precompute_rope_cos_sin(rope_dim, md.rope_theta, md.context_length, device)
    }

    /// Construct a MiniMax stage from its assigned GGUF tensors.
    ///
    /// Returns an error when required metadata, packed expert tensors, or
    /// attention projections are missing or have incompatible dimensions.
    pub fn from_stage_tensors(
        stage: StageTensors,
        quant_config: Option<Arc<TurboQuantConfig>>,
    ) -> Result<Self> {
        stage.range.validate()?;
        let (rope_cos, rope_sin) = Self::rope_tables(&stage.metadata, &stage.device)?;
        Self::from_stage_tensors_with_rope(stage, quant_config, &rope_cos, &rope_sin)
    }

    fn from_stage_tensors_with_rope(
        stage: StageTensors,
        quant_config: Option<Arc<TurboQuantConfig>>,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> Result<Self> {
        let StageTensors {
            range,
            metadata,
            mut tensors,
            device,
            ..
        } = stage;
        let md = &metadata;
        #[cfg(feature = "cuda")]
        {
            super::dense_qmatmul_cuda::precompile(&device)?;
            super::residual_norm_cuda::precompile(&device)?;
        }
        anyhow::ensure!(
            md.expert_gating_func == 2,
            "unsupported MiniMax expert gating function {}; expected sigmoid (2)",
            md.expert_gating_func,
        );

        let embedding = if range.is_head() {
            let embd = tensors
                .get("token_embd.weight")
                .context("missing token_embd.weight")?;
            Some(embd.as_ref().dequantize(&device)?)
        } else {
            None
        };

        let mut blocks = Vec::with_capacity(range.layer_count());
        for layer_idx in range.start..range.end {
            let prefix = format!("blk.{layer_idx}");

            let attn = build_attention(
                &tensors,
                &prefix,
                md,
                &device,
                rope_cos,
                rope_sin,
                quant_config.clone(),
            )?;
            let attn_norm = load_rms_norm(
                &tensors,
                &format!("{prefix}.attn_norm"),
                md.rms_norm_eps,
                &device,
            )?;
            let ffn_norm = load_rms_norm(
                &tensors,
                &format!("{prefix}.ffn_norm"),
                md.rms_norm_eps,
                &device,
            )?;
            let moe = build_moe_layer(&mut tensors, &prefix, layer_idx, md, &device)?;

            blocks.push(MoeBlock {
                attn_norm,
                attn,
                ffn_norm,
                moe,
            });
        }

        let output_norm = if range.is_tail() {
            Some(load_rms_norm(
                &tensors,
                "output_norm",
                md.rms_norm_eps,
                &device,
            )?)
        } else {
            None
        };

        let lm_head = if range.is_tail() {
            let head = tensors
                .get("output.weight")
                .or_else(|| tensors.get("token_embd.weight"))
                .context("missing output.weight")?;
            Some(DecodeSafeQMatMul::from_qtensor(Arc::clone(head), &device)?.with_matrix_decode())
        } else {
            None
        };

        tracing::info!(
            "MiniMax-M2 stage: layers [{}, {}), {} MoE blocks, {} experts (top-{})",
            range.start,
            range.end,
            blocks.len(),
            md.expert_count,
            md.expert_used_count,
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
            device,
        })
    }

    fn forward_packed_ragged_decode_batch(
        &self,
        tokens: &[u32],
        caches: &mut [&mut [Option<LayerCache>]],
        batch_cache: &mut [Option<LayerCache>],
        positions: &[usize],
    ) -> Result<Vec<Tensor>> {
        let batch = tokens.len();
        let token_tensor = Tensor::new(tokens, &self.device)?;
        let mut hidden_states = self
            .embedding
            .as_ref()
            .unwrap()
            .index_select(&token_tensor, 0)?
            .reshape((batch, 1, self.hidden_dim))?;

        let mut next_batch_cache = Vec::with_capacity(self.blocks.len());
        let mut next_attn_normed = None;
        let mut tail_normed = None;
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let mut packed = match &batch_cache[layer_idx] {
                Some(cache) => super::clone_decode_layer_cache(cache)?,
                None => super::pack_ragged_decode_layer_cache(caches, layer_idx, positions)?,
            };
            let normed = if let Some(normed) = next_attn_normed.take() {
                normed
            } else {
                block.attn_norm.forward(&hidden_states)?
            };
            let attn_out =
                block
                    .attn
                    .forward_packed_ragged_decode(&normed, &mut packed, positions)?;
            let (next_hidden_states, normed) =
                block.ffn_norm.forward_residual(&hidden_states, &attn_out)?;
            hidden_states = next_hidden_states;
            let moe_out = block.moe.forward(&normed, false)?;
            if let Some(next_block) = self.blocks.get(layer_idx + 1) {
                let (next_hidden_states, normed) = next_block
                    .attn_norm
                    .forward_residual(&hidden_states, &moe_out)?;
                hidden_states = next_hidden_states;
                next_attn_normed = Some(normed);
            } else {
                let output_norm = self.output_norm.as_ref().unwrap();
                let (next_hidden_states, normed) =
                    output_norm.forward_residual(&hidden_states, &moe_out)?;
                hidden_states = next_hidden_states;
                tail_normed = Some(normed);
            }
            next_batch_cache.push(packed);
        }

        let normed = if let Some(normed) = tail_normed {
            normed
        } else {
            self.output_norm.as_ref().unwrap().forward(&hidden_states)?
        };
        let logits = self
            .lm_head
            .as_ref()
            .unwrap()
            .forward_decode(&normed.squeeze(1)?)?;
        let outputs = (0..batch)
            .map(|lane| logits.narrow(0, lane, 1)?.squeeze(0).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;
        let valid_lengths = positions
            .iter()
            .map(|position| position.saturating_add(1))
            .collect::<Vec<_>>();

        for (layer_idx, cache) in next_batch_cache.iter().enumerate() {
            super::publish_ragged_decode_layer_cache_views(
                caches,
                layer_idx,
                cache,
                &valid_lengths,
            )?;
        }
        for (slot, cache) in batch_cache.iter_mut().zip(next_batch_cache) {
            *slot = cache.map(LayerCache::Attention);
        }
        Ok(outputs)
    }
}

impl MiniMaxM2Stage {
    fn forward_impl(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
        prefill: bool,
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
            let profile = seq_len == 1 && layer_profile_enabled();
            if profile {
                PROFILE_CALLS.fetch_add(1, Ordering::Relaxed);
            }
            let started = Instant::now();
            let normed = if let Some(normed) = next_attn_normed.take() {
                normed
            } else {
                block.attn_norm.forward(&hidden_states)?
            };
            if profile {
                finish_profile_region(normed.device(), started, &ATTN_NORM_NS)?;
            }
            let started = Instant::now();
            let attn_out = with_kv_cache(&mut cache[i], |kv| {
                block.attn.forward(&normed, kv, seq_len, start_pos, prefill)
            })
            .with_context(|| {
                format!(
                    "MiniMax local layer {i} attention, {seq_len} tokens at position {start_pos}"
                )
            })?;
            if profile {
                finish_profile_region(attn_out.device(), started, &ATTN_NS)?;
            }
            let normed = if profile {
                let started = Instant::now();
                hidden_states = (hidden_states + attn_out)?;
                finish_profile_region(hidden_states.device(), started, &ATTN_RESIDUAL_NS)?;
                let started = Instant::now();
                let normed = block.ffn_norm.forward(&hidden_states)?;
                finish_profile_region(normed.device(), started, &FFN_NORM_NS)?;
                normed
            } else {
                let (next_hidden_states, normed) =
                    block.ffn_norm.forward_residual(&hidden_states, &attn_out)?;
                hidden_states = next_hidden_states;
                normed
            };
            let moe_out = block.moe.forward(&normed, prefill)?;
            let started = Instant::now();
            if profile {
                hidden_states = (hidden_states + moe_out)?;
                finish_profile_region(hidden_states.device(), started, &FFN_RESIDUAL_NS)?;
            } else if let Some(next_block) = self.blocks.get(i + 1) {
                let (next_hidden_states, normed) = next_block
                    .attn_norm
                    .forward_residual(&hidden_states, &moe_out)?;
                hidden_states = next_hidden_states;
                next_attn_normed = Some(normed);
            } else if self.is_tail {
                let output_norm = self.output_norm.as_ref().unwrap();
                let (next_hidden_states, normed) =
                    output_norm.forward_residual(&hidden_states, &moe_out)?;
                hidden_states = next_hidden_states;
                tail_normed = Some(normed);
            } else {
                hidden_states = (hidden_states + moe_out)?;
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
            let logits = self
                .lm_head
                .as_ref()
                .unwrap()
                .forward_decode(&last_hidden)?;
            Ok(StageOutput::Logits(logits.squeeze(0)?))
        } else {
            Ok(StageOutput::Activations(hidden_states.squeeze(0)?))
        }
    }
}

impl ModelArchitecture for MiniMaxM2Stage {
    fn forward(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_impl(input, cache, seq_len, start_pos, false)
    }

    fn forward_prefill(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_impl(input, cache, seq_len, start_pos, self.device.is_metal())
    }

    fn prefers_f32_decode_activations(&self) -> bool {
        // F16 stage boundaries changed greedy output in the three-stage
        // REAP-50 Q2_K test; F32 activations matched the solo run. Preserve
        // the residual values used by later attention and expert routing.
        true
    }

    fn prefers_f32_prefill_activations(&self) -> bool {
        // Prompt activations also build the downstream KV state. Keeping
        // decode F32 cannot recover precision lost when that state was built.
        true
    }

    fn supports_decode_batch(&self) -> bool {
        self.is_head
            && self.is_tail
            && self
                .blocks
                .iter()
                .all(|block| block.attn.quant_config.is_none())
    }

    fn decode_batch_requires_preallocated_attention(&self) -> bool {
        true
    }

    fn forward_decode_batch(
        &self,
        tokens: &[u32],
        caches: &mut [&mut [Option<LayerCache>]],
        batch_cache: &mut [Option<LayerCache>],
        positions: &[usize],
    ) -> Result<Vec<Tensor>> {
        anyhow::ensure!(
            self.supports_decode_batch(),
            "MiniMax decode batching unavailable"
        );
        anyhow::ensure!(!tokens.is_empty(), "decode batch cannot be empty");
        anyhow::ensure!(tokens.len() == caches.len(), "decode batch lane mismatch");
        anyhow::ensure!(
            tokens.len() == positions.len(),
            "decode batch position mismatch"
        );
        anyhow::ensure!(
            batch_cache.len() == self.blocks.len(),
            "persistent decode cache layer mismatch"
        );
        if !positions.iter().all(|position| *position == positions[0]) {
            return self.forward_packed_ragged_decode_batch(tokens, caches, batch_cache, positions);
        }

        let batch = tokens.len();
        let start_pos = positions[0];
        let token_tensor = Tensor::new(tokens, &self.device)?;
        let mut hidden_states = self
            .embedding
            .as_ref()
            .unwrap()
            .index_select(&token_tensor, 0)?
            .reshape((batch, 1, self.hidden_dim))?;

        let mut next_batch_cache = Vec::with_capacity(self.blocks.len());
        let mut next_attn_normed = None;
        let mut tail_normed = None;
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let mut batched_cache = match &batch_cache[layer_idx] {
                Some(cache) => super::clone_decode_layer_cache(cache)?,
                None => super::pack_decode_layer_cache(caches, layer_idx)?,
            };
            let normed = if let Some(normed) = next_attn_normed.take() {
                normed
            } else {
                block.attn_norm.forward(&hidden_states)?
            };
            let attn_out = block
                .attn
                .forward(&normed, &mut batched_cache, 1, start_pos, false)?;
            let (next_hidden_states, normed) =
                block.ffn_norm.forward_residual(&hidden_states, &attn_out)?;
            hidden_states = next_hidden_states;
            let moe_out = block.moe.forward(&normed, false)?;
            if let Some(next_block) = self.blocks.get(layer_idx + 1) {
                let (next_hidden_states, normed) = next_block
                    .attn_norm
                    .forward_residual(&hidden_states, &moe_out)?;
                hidden_states = next_hidden_states;
                next_attn_normed = Some(normed);
            } else {
                let output_norm = self.output_norm.as_ref().unwrap();
                let (next_hidden_states, normed) =
                    output_norm.forward_residual(&hidden_states, &moe_out)?;
                hidden_states = next_hidden_states;
                tail_normed = Some(normed);
            }
            next_batch_cache.push(batched_cache);
        }

        let normed = if let Some(normed) = tail_normed {
            normed
        } else {
            self.output_norm.as_ref().unwrap().forward(&hidden_states)?
        };
        let logits = self
            .lm_head
            .as_ref()
            .unwrap()
            .forward_decode(&normed.squeeze(1)?)?;
        let outputs = (0..batch)
            .map(|lane| logits.narrow(0, lane, 1)?.squeeze(0).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;

        for (layer_idx, cache) in next_batch_cache.iter().enumerate() {
            super::publish_decode_layer_cache_views(caches, layer_idx, cache)?;
        }
        for (slot, cache) in batch_cache.iter_mut().zip(next_batch_cache) {
            *slot = cache.map(LayerCache::Attention);
        }
        Ok(outputs)
    }

    fn family(&self) -> &str {
        "minimax-m2"
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
    device: &Device,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
    quant_config: Option<Arc<TurboQuantConfig>>,
) -> Result<Attention> {
    let get = |name: &str| -> Result<QMatMul> {
        let t = tensors
            .get(name)
            .with_context(|| format!("missing tensor: {name}"))?;
        qtensor_to_qmatmul(Arc::clone(t))
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
            device,
        )?
        .with_matrix_decode(),
        o_proj: get(&format!("{prefix}.attn_output.weight"))?,
        q_norm: load_rms_norm(
            tensors,
            &format!("{prefix}.attn_q_norm"),
            md.rms_norm_eps,
            device,
        )?,
        k_norm: load_rms_norm(
            tensors,
            &format!("{prefix}.attn_k_norm"),
            md.rms_norm_eps,
            device,
        )?,
        num_heads: md.num_attention_heads,
        num_kv_heads: md.num_kv_heads,
        head_dim: md.head_dim,
        rope_dim: if md.rope_dimension_count > 0 {
            md.rope_dimension_count
        } else {
            md.head_dim
        },
        rope_cos: rope_cos.clone(),
        rope_sin: rope_sin.clone(),
        quant_config,
    })
}

fn build_moe_layer(
    tensors: &mut HashMap<String, Arc<QTensor>>,
    prefix: &str,
    layer_idx: usize,
    md: &crate::gguf_loader::GgufMetadata,
    device: &Device,
) -> Result<MoeLayer> {
    let gate_tensor = tensors
        .get(&format!("{prefix}.ffn_gate_inp.weight"))
        .context("missing ffn_gate_inp.weight")?;
    let gate = gate_tensor.as_ref().dequantize(device)?;

    // Expert bias (optional)
    let expert_bias = tensors
        .get(&format!("{prefix}.exp_probs_b.bias"))
        .or_else(|| tensors.get(&format!("{prefix}.exp_probs_b.weight")))
        .map(|t| t.as_ref().dequantize(device))
        .transpose()?;

    // Packed expert tensors — need splitting
    let gate_exps_name = format!("{prefix}.ffn_gate_exps.weight");
    let experts = if tensors.contains_key(&gate_exps_name) {
        // Consume packed CPU tensors as each layer is expanded so a solo load
        // never retains both complete packed and per-expert copies.
        let gate_exps = tensors
            .remove(&gate_exps_name)
            .context("missing ffn_gate_exps")?;
        let up_exps = tensors
            .remove(&format!("{prefix}.ffn_up_exps.weight"))
            .context("missing ffn_up_exps")?;
        let down_exps = tensors
            .remove(&format!("{prefix}.ffn_down_exps.weight"))
            .context("missing ffn_down_exps")?;

        if layer_idx == 0 {
            tracing::info!(
                layer_idx,
                gate = ?gate_exps.dtype(),
                up = ?up_exps.dtype(),
                down = ?down_exps.dtype(),
                "MiniMax packed expert dtypes"
            );
        }

        #[cfg(any(target_os = "macos", feature = "cuda"))]
        {
            let use_indexed = indexed_moe_available(device, gate_exps.dtype())
                && indexed_moe_available(device, up_exps.dtype())
                && indexed_moe_available(device, down_exps.dtype());
            if use_indexed {
                let experts = MoeExperts::Indexed(Box::new(IndexedMoeExperts {
                    gate_proj: IndexedMoeMatMul::from_qtensor(
                        &gate_exps,
                        md.expert_count,
                        md.expert_ffn_dim,
                        md.hidden_dim,
                        device,
                    )?,
                    up_proj: IndexedMoeMatMul::from_qtensor(
                        &up_exps,
                        md.expert_count,
                        md.expert_ffn_dim,
                        md.hidden_dim,
                        device,
                    )?,
                    down_proj: IndexedMoeMatMul::from_qtensor(
                        &down_exps,
                        md.expert_count,
                        md.hidden_dim,
                        md.expert_ffn_dim,
                        device,
                    )?,
                }));
                // Uploads are asynchronous; retain the packed CPU backing until
                // the three complete expert tensors have finished copying.
                device.synchronize()?;
                experts
            } else {
                let experts = MoeExperts::Split(split_packed_experts(
                    &gate_exps,
                    &up_exps,
                    &down_exps,
                    md.expert_count,
                    md.expert_ffn_dim,
                    md.hidden_dim,
                    device,
                )?);
                device.synchronize()?;
                experts
            }
        }
        #[cfg(not(any(target_os = "macos", feature = "cuda")))]
        {
            MoeExperts::Split(split_packed_experts(
                &gate_exps,
                &up_exps,
                &down_exps,
                md.expert_count,
                md.expert_ffn_dim,
                md.hidden_dim,
                device,
            )?)
        }
    } else {
        // Per-expert format fallback
        let mut exps = Vec::with_capacity(md.expert_count);
        for e in 0..md.expert_count {
            let get = |name: &str| -> Result<QMatMul> {
                let t = tensors
                    .get(name)
                    .with_context(|| format!("missing tensor: {name}"))?;
                qtensor_to_qmatmul(Arc::clone(t))
            };
            exps.push(MoeExpert {
                gate_proj: get(&format!("{prefix}.ffn_gate.{e}.weight"))?,
                up_proj: get(&format!("{prefix}.ffn_up.{e}.weight"))?,
                down_proj: get(&format!("{prefix}.ffn_down.{e}.weight"))?,
            });
        }
        MoeExperts::Split(exps)
    };

    Ok(MoeLayer {
        gate,
        experts,
        expert_bias,
        num_active: md.expert_used_count.max(1),
        layer_idx,
    })
}

/// Split packed expert tensors [n_experts, out_dim, in_dim] into per-expert QMatMul.
fn split_packed_experts(
    gate_packed: &Arc<QTensor>,
    up_packed: &Arc<QTensor>,
    down_packed: &Arc<QTensor>,
    num_experts: usize,
    ffn_dim: usize,
    hidden_dim: usize,
    device: &Device,
) -> Result<Vec<MoeExpert>> {
    let split_one = |packed: &QTensor, out_dim: usize, in_dim: usize| -> Result<Vec<QMatMul>> {
        let dtype = packed.dtype();
        let block_size = dtype.block_size();
        let type_size = dtype.type_size();
        let elements_per_expert = out_dim * in_dim;
        anyhow::ensure!(
            elements_per_expert.is_multiple_of(block_size),
            "Expert dims ({out_dim}x{in_dim}={elements_per_expert}) not block-aligned for {dtype:?} (block_size={block_size})"
        );
        let bytes_per_expert = (elements_per_expert / block_size) * type_size;
        let raw = packed.data()?;
        let mut results = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            let start = i * bytes_per_expert;
            let end = start + bytes_per_expert;
            let qt = candle_core::quantized::ggml_file::qtensor_from_ggml(
                dtype,
                &raw[start..end],
                vec![out_dim, in_dim],
                device,
            )?;
            results.push(QMatMul::from_qtensor(qt)?);
        }
        Ok(results)
    };

    let gates = split_one(gate_packed, ffn_dim, hidden_dim)?;
    let ups = split_one(up_packed, ffn_dim, hidden_dim)?;
    let downs = split_one(down_packed, hidden_dim, ffn_dim)?;

    let mut experts = Vec::with_capacity(num_experts);
    for i in 0..num_experts {
        experts.push(MoeExpert {
            gate_proj: gates[i].clone(),
            up_proj: ups[i].clone(),
            down_proj: downs[i].clone(),
        });
    }
    Ok(experts)
}

fn load_rms_norm(
    tensors: &HashMap<String, Arc<QTensor>>,
    prefix: &str,
    eps: f64,
    device: &Device,
) -> Result<RmsNorm> {
    let weight = tensors
        .get(&format!("{prefix}.weight"))
        .with_context(|| format!("missing {prefix}.weight"))?
        .as_ref()
        .dequantize(device)?;
    Ok(RmsNorm { weight, eps })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_router_is_independent_of_prefill_chunk_size() -> Result<()> {
        let device = &Device::Cpu;
        let (rows, width, experts) = (51, 3072, 256);
        let values = |len: usize, salt: usize| {
            (0..len)
                .map(|i| (((i * salt + 19) % 1021) as f32 - 510.0) / 513.7)
                .collect::<Vec<_>>()
        };
        let input = Tensor::from_vec(values(rows * width, 17), (rows, width), device)?;
        let gate = Tensor::from_vec(values(experts * width, 13), (experts, width), device)?;
        let full = project_router(&input, &gate, false)?.to_vec2::<f32>()?;
        for chunk_size in [1, 3, 8, 32] {
            for start in (0..rows).step_by(chunk_size) {
                let count = chunk_size.min(rows - start);
                let chunk = project_router(&input.narrow(0, start, count)?, &gate, false)?
                    .to_vec2::<f32>()?;
                assert_eq!(chunk, full[start..start + count], "router at {start}");
            }
        }
        Ok(())
    }

    #[test]
    fn cpu_causal_attention_matches_reference_and_chunked_prefixes() -> Result<()> {
        let device = &Device::Cpu;
        let (batch, heads, rows, dim, prefix) = (2, 3, 51, 16, 13);
        let keys = rows + prefix;
        let values = |len: usize, salt: usize| {
            (0..len)
                .map(|i| (((i * salt + 19) % 1021) as f32 - 510.0) / 513.7)
                .collect::<Vec<_>>()
        };
        let q_values = values(batch * heads * rows * dim, 23);
        let k_values = values(batch * heads * keys * dim, 31);
        let v_values = values(batch * heads * keys * dim, 47);
        let q = Tensor::from_vec(q_values.clone(), (batch, heads, rows, dim), device)?;
        let k = Tensor::from_vec(k_values.clone(), (batch, heads, keys, dim), device)?;
        let v = Tensor::from_vec(v_values.clone(), (batch, heads, keys, dim), device)?;
        let scale = 1.0 / (dim as f64).sqrt();
        let full = cpu_causal_attention(&q, &k, &v, scale)?;
        let actual = full.flatten_all()?.to_vec1::<f32>()?;
        // Independent scalar F64 softmax and weighted sum, including a cached
        // prefix. This checks the bottom-right causal offset and both heads
        // and batches, rather than comparing two calls to the same formula.
        for lane in 0..batch * heads {
            for row in 0..rows {
                let visible = prefix + row + 1;
                let scores = (0..visible)
                    .map(|key| {
                        (0..dim)
                            .map(|col| {
                                f64::from(q_values[(lane * rows + row) * dim + col])
                                    * f64::from(k_values[(lane * keys + key) * dim + col])
                            })
                            .sum::<f64>()
                            * scale
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights = scores
                    .iter()
                    .map(|score| (score - max).exp())
                    .collect::<Vec<_>>();
                let sum = weights.iter().sum::<f64>();
                for col in 0..dim {
                    let expected = weights
                        .iter()
                        .enumerate()
                        .map(|(key, weight)| {
                            weight / sum * f64::from(v_values[(lane * keys + key) * dim + col])
                        })
                        .sum::<f64>();
                    let actual = f64::from(actual[(lane * rows + row) * dim + col]);
                    assert!(
                        (actual - expected).abs() < 1e-6,
                        "lane={lane} row={row} col={col}: {actual} vs {expected}"
                    );
                }
            }
        }
        for chunk_size in [1, 3, 8, 32] {
            for start in (0..rows).step_by(chunk_size) {
                let count = chunk_size.min(rows - start);
                let chunk = cpu_causal_attention(
                    &q.narrow(2, start, count)?,
                    &k.narrow(2, 0, prefix + start + count)?,
                    &v.narrow(2, 0, prefix + start + count)?,
                    scale,
                )?
                .flatten_all()?
                .to_vec1::<f32>()?;
                let expected = full
                    .narrow(2, start, count)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert_eq!(chunk, expected, "chunk_size={chunk_size} start={start}");
            }
        }
        Ok(())
    }

    #[test]
    fn minimax_activations_stay_f32_across_pipeline_stages() {
        for (is_head, is_tail) in [(true, false), (false, false), (false, true)] {
            let stage = MiniMaxM2Stage {
                embedding: None,
                blocks: Vec::new(),
                output_norm: None,
                lm_head: None,
                is_head,
                is_tail,
                hidden_dim: 0,
                num_layers: 0,
                vocab_size: 0,
                device: Device::Cpu,
            };
            assert!(stage.prefers_f32_prefill_activations());
            assert!(stage.prefers_f32_decode_activations());
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_decode_batch_supports_solo_but_rejects_split_stages() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        // No weights are needed to check the device/capability gate.
        let mut stage = MiniMaxM2Stage {
            embedding: None,
            blocks: Vec::new(),
            output_norm: None,
            lm_head: None,
            is_head: true,
            is_tail: true,
            hidden_dim: 0,
            num_layers: 0,
            vocab_size: 0,
            device: Device::Cpu,
        };
        assert!(stage.supports_decode_batch());
        stage.device = device;
        assert!(stage.supports_decode_batch());
        let error = stage
            .forward_decode_batch(&[], &mut [], &mut [], &[])
            .unwrap_err();
        assert!(error.to_string().contains("decode batch cannot be empty"));
        stage.is_tail = false;
        assert!(!stage.supports_decode_batch());
        let error = stage
            .forward_decode_batch(&[], &mut [], &mut [], &[])
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("MiniMax decode batching unavailable")
        );
        stage.is_tail = true;
        stage.is_head = false;
        assert!(!stage.supports_decode_batch());
        Ok(())
    }

    #[test]
    fn minimax_router_uses_sigmoid_probabilities() -> Result<()> {
        let logits = Tensor::new(&[-1.0f32, 0.0, 1.0], &Device::Cpu)?;
        let probabilities = sigmoid_router_probs(&logits)?.to_vec1::<f32>()?;
        assert!((probabilities[0] - 0.268_941_43).abs() < 1e-6);
        assert!((probabilities[1] - 0.5).abs() < 1e-6);
        assert!((probabilities[2] - 0.731_058_6).abs() < 1e-6);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_decode_router_matches_each_scalar_lane_exactly() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (experts, width) = (128, 3072);
        let gate = Tensor::from_vec(
            (0..experts * width)
                .map(|i| ((i * 13 % 257) as f32 - 128.) / 1024.)
                .collect::<Vec<_>>(),
            (experts, width),
            &device,
        )?;
        for lanes in [2, 4] {
            let backing = Tensor::from_vec(
                (0..(lanes + 1) * width)
                    .map(|i| ((i * 11 % 127) as f32 - 63.) / 128.)
                    .collect::<Vec<_>>(),
                (lanes + 1, width),
                &device,
            )?;
            let input = backing.narrow(0, 1, lanes)?;
            let batch = project_router(&input, &gate, true)?.to_vec2::<f32>()?;
            for (lane, actual) in batch.iter().enumerate() {
                let expected =
                    project_router(&input.narrow(0, lane, 1)?, &gate, true)?.to_vec2::<f32>()?;
                assert_eq!(*actual, expected[0]);
            }
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_one_token_prompt_router_matches_matrix_rows() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (experts, width) = (128, 3072);
        let gate = Tensor::from_vec(
            (0..experts * width)
                .map(|i| ((i * 13 % 257) as f32 - 128.) / 1024.)
                .collect::<Vec<_>>(),
            (experts, width),
            &device,
        )?;
        let input = Tensor::from_vec(
            (0..128 * width)
                .map(|i| ((i * 11 % 127) as f32 - 63.) / 128.)
                .collect::<Vec<_>>(),
            (128, width),
            &device,
        )?;
        let expected = project_router(&input, &gate, false)?.to_vec2::<f32>()?;
        for (row, expected) in expected.iter().enumerate() {
            let actual =
                project_router(&input.narrow(0, row, 1)?, &gate, false)?.to_vec2::<f32>()?;
            assert_eq!(actual[0], *expected, "prompt row {row}");
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_prompt_tail_attention_matches_full_prompt_last_row() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (heads, kv_heads, dim) = (8, 2, 128);
        let scale = (1.0 / (dim as f64).sqrt()) as f32;
        for len in [1, 2, 7, 63, 721] {
            let q = Tensor::from_vec(
                (0..heads * len * dim)
                    .map(|i| ((i * 13 % 257) as f32 - 128.) / 64.)
                    .collect::<Vec<_>>(),
                (1, heads, len, dim),
                &device,
            )?;
            let k = Tensor::from_vec(
                (0..kv_heads * len * dim)
                    .map(|i| ((i * 11 % 127) as f32 - 63.) / 128.)
                    .collect::<Vec<_>>(),
                (1, kv_heads, len, dim),
                &device,
            )?
            .to_dtype(DType::F16)?;
            let v = (&k * 0.7)?;
            let expected = super::super::try_metal_f16_sdpa_native(&q, &k, &v, len > 1, scale)?
                .context("test attention unavailable")?
                .narrow(2, len - 1, 1)?
                .flatten_all()?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            let actual =
                prompt_tail_attention(&q.narrow(2, len - 1, 1)?.contiguous()?, &k, &v, scale)?
                    .context("test prompt attention unavailable")?
                    .flatten_all()?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
            for (i, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(actual, expected, "attention length {len}, column {i}");
            }
            assert_eq!(k.dim(2)?, len);
            assert_eq!(v.dim(2)?, len);
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn prompt_tail_attention_preserves_quantized_cache_fallback() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let q = Tensor::zeros((1, 2, 1, 128), DType::F32, &device)?;
        let kv = Tensor::zeros((1, 1, 2, 128), DType::F32, &device)?;
        assert!(prompt_tail_attention(&q, &kv, &kv, 0.1)?.is_none());
        Ok(())
    }

    #[test]
    fn partial_rope_preserves_non_rotary_dimensions() -> Result<()> {
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 1, 4), &Device::Cpu)?;
        let cos = Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?;
        let sin = Tensor::ones((1, 1), DType::F32, &Device::Cpu)?;
        let output = apply_partial_rotary_emb(&x, 0, &cos, &sin, 2)?;
        assert_eq!(
            output
                .narrow(D::Minus1, 2, 2)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            vec![3.0, 4.0]
        );
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_partial_rope_matches_lane_local_reference() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (batch, q_heads, k_heads, head_dim, rope_dim) = (2, 2, 1, 8, 4);
        let positions = [1usize, 3];
        let q = Tensor::from_vec(
            (0..batch * q_heads * head_dim)
                .map(|index| (index as f32 * 0.13).sin())
                .collect::<Vec<_>>(),
            (batch, q_heads, 1, head_dim),
            &device,
        )?;
        let k = Tensor::from_vec(
            (0..batch * k_heads * head_dim)
                .map(|index| (index as f32 * 0.17).cos())
                .collect::<Vec<_>>(),
            (batch, k_heads, 1, head_dim),
            &device,
        )?;
        let (cos, sin) = precompute_rope_cos_sin(rope_dim, 10_000.0, 8, &device)?;
        let (metal_q, metal_k) =
            super::super::try_metal_ragged_rope(&q, &k, &cos, &sin, &positions, true)?
                .context("partial Metal RoPE unavailable")?;

        let mut expected_q = Vec::with_capacity(batch);
        let mut expected_k = Vec::with_capacity(batch);
        for (lane, position) in positions.iter().copied().enumerate() {
            expected_q.push(apply_partial_rotary_emb(
                &q.narrow(0, lane, 1)?,
                position,
                &cos,
                &sin,
                rope_dim,
            )?);
            expected_k.push(apply_partial_rotary_emb(
                &k.narrow(0, lane, 1)?,
                position,
                &cos,
                &sin,
                rope_dim,
            )?);
        }
        let expected_q = Tensor::cat(&expected_q.iter().collect::<Vec<_>>(), 0)?;
        let expected_k = Tensor::cat(&expected_k.iter().collect::<Vec<_>>(), 0)?;
        for (actual, expected) in [(metal_q, expected_q), (metal_k, expected_k)] {
            let actual = actual.flatten_all()?.to_vec1::<f32>()?;
            let expected = expected.flatten_all()?.to_vec1::<f32>()?;
            let max_abs_diff = actual
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(max_abs_diff < 1e-6, "max_abs_diff={max_abs_diff}");
        }
        Ok(())
    }

    #[test]
    fn padded_single_token_expert_input_duplicates_row_once() -> Result<()> {
        let x = Tensor::new(&[[2.0f32, 3.0]], &Device::Cpu)?;
        let (padded, was_padded) = expert_input_with_m1_padding(&x, true)?;
        assert!(was_padded);
        assert_eq!(padded.dims2()?, (2, 2));
        assert_eq!(
            padded.to_vec2::<f32>()?,
            vec![vec![2.0, 3.0], vec![2.0, 3.0]]
        );
        Ok(())
    }

    #[test]
    fn grouped_moe_preserves_weighted_expert_output() -> Result<()> {
        let identity = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &Device::Cpu)?;
        let expert = || MoeExpert {
            gate_proj: QMatMul::Tensor(identity.clone()),
            up_proj: QMatMul::Tensor(identity.clone()),
            down_proj: QMatMul::Tensor(identity.clone()),
        };
        let layer = MoeLayer {
            gate: Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?,
            experts: MoeExperts::Split(vec![expert(), expert()]),
            expert_bias: None,
            num_active: 2,
            layer_idx: 0,
        };
        let input = Tensor::new(&[[[1.0f32, 2.0], [3.0, 4.0]]], &Device::Cpu)?;
        let output = layer.forward(&input, false)?;
        let expected = (candle_nn::Activation::Silu.forward(&input)? * &input)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5);
        }

        let input = Tensor::new(&[[[2.0f32, 3.0]]], &Device::Cpu)?;
        let output = layer.forward(&input, false)?;
        let expected = (candle_nn::Activation::Silu.forward(&input)? * &input)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5);
        }
        Ok(())
    }

    #[test]
    fn cpu_grouped_moe_keeps_scalar_route_sum_order() -> Result<()> {
        let device = &Device::Cpu;
        let identity = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], device)?;
        let experts = [-1e8, 1e8, 1.0]
            .into_iter()
            .map(|scale| -> Result<_> {
                Ok(MoeExpert {
                    gate_proj: QMatMul::Tensor(identity.clone()),
                    up_proj: QMatMul::Tensor(identity.clone()),
                    down_proj: QMatMul::Tensor((&identity * scale)?),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let layer = MoeLayer {
            gate: Tensor::zeros((3, 2), DType::F32, device)?,
            experts: MoeExperts::Split(experts),
            expert_bias: Some(Tensor::new(&[1.0f32, 3.0, 2.0], device)?),
            num_active: 3,
            layer_idx: 0,
        };
        let input = Tensor::new(&[[[1.0f32, 2.0], [3.0, 4.0], [2.0, 1.0]]], device)?;
        let full = layer.forward(&input, false)?.to_vec3::<f32>()?;
        for (row, expected) in full[0].iter().enumerate() {
            let scalar = layer
                .forward(&input.narrow(1, row, 1)?, false)?
                .to_vec3::<f32>()?;
            assert_eq!(&scalar[0][0], expected, "route sum at row {row}");
        }
        Ok(())
    }
}
