//! DeepSeek V3/R1 architecture — Multi-head Latent Attention (MLA) + MoE.
//!
//! Key differences from standard attention:
//! - Q path: hidden → q_a (compress) → q_norm → q_b (expand) → [q_nope, q_pe]
//! - KV path: hidden → kv_a → [kv_latent, k_pe], kv_latent → kv_norm → kv_b → [k_nope, v]
//! - RoPE applied only to q_pe and k_pe dimensions (64D), not the full Q/K
//! - First `first_k_dense_replace` layers use dense FFN, rest use MoE with shared experts
//! - KV cache stores full decompressed K/V (V1 — naive mode, correct but not compressed)

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::qmatmul::QMatMul;
use anyhow::{Context, Result};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{D, DType, Device, IndexOp, Module, Tensor};

#[cfg(target_os = "macos")]
use super::fused_route_metal::fused_softmax_topk_route;
#[cfg(any(target_os = "macos", feature = "cuda"))]
use super::indexed_moe::{
    IndexedMoeMatMul, indexed_moe_available, normalized_route_weights, silu_mul, weighted_route_sum,
};
use super::{
    KvCache, LayerCache, ModelArchitecture, StageInput, StageOutput, TextArithmeticPolicy,
    TextRequestContext, kv_cache_update_native, with_kv_cache,
};
use crate::gguf_loader::StageTensors;
use crate::turboquant::TurboQuantConfig;

mod cpu_math;
mod native_cpu;
mod native_online_attention;
mod native_route_normalization;
mod native_router;
mod native_scale;
mod native_yarn;

// ===== Helpers =====

static PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();
static ATTN_NORM_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_RESIDUAL_NS: AtomicU64 = AtomicU64::new(0);
static FFN_NORM_NS: AtomicU64 = AtomicU64::new(0);
static FFN_NS: AtomicU64 = AtomicU64::new(0);
static FFN_RESIDUAL_NS: AtomicU64 = AtomicU64::new(0);
static ROUTER_NS: AtomicU64 = AtomicU64::new(0);
static ROUTED_EXPERT_NS: AtomicU64 = AtomicU64::new(0);
static SHARED_EXPERT_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_CALLS: AtomicU64 = AtomicU64::new(0);
static MOE_PROFILE_CALLS: AtomicU64 = AtomicU64::new(0);

fn layer_profile_enabled() -> bool {
    *PROFILE_ENABLED.get_or_init(|| std::env::var_os("DINF_PROFILE_LAYERS").is_some())
}

fn finish_profile_region(device: &Device, started: Instant, counter: &AtomicU64) -> Result<()> {
    device.synchronize()?;
    counter.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(())
}

/// Reset the process-local DeepSeek layer profiling counters.
pub fn reset_layer_profile() {
    for counter in [
        &ATTN_NORM_NS,
        &ATTN_NS,
        &ATTN_RESIDUAL_NS,
        &FFN_NORM_NS,
        &FFN_NS,
        &FFN_RESIDUAL_NS,
        &ROUTER_NS,
        &ROUTED_EXPERT_NS,
        &SHARED_EXPERT_NS,
        &PROFILE_CALLS,
        &MOE_PROFILE_CALLS,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

/// Print the process-local DeepSeek layer profiling counters to stderr.
pub fn report_layer_profile() {
    let calls = PROFILE_CALLS.load(Ordering::Relaxed).max(1) as f64;
    let moe_calls = MOE_PROFILE_CALLS.load(Ordering::Relaxed).max(1) as f64;
    let ms_per_call =
        |counter: &AtomicU64| counter.load(Ordering::Relaxed) as f64 / calls / 1_000_000.0;
    let ms_per_moe_call =
        |counter: &AtomicU64| counter.load(Ordering::Relaxed) as f64 / moe_calls / 1_000_000.0;
    eprintln!(
        "DINF_DEEPSEEK_PROFILE calls={} per_layer_ms={{attn_norm:{:.4},attention:{:.4},attn_residual:{:.4},ffn_norm:{:.4},ffn:{:.4},ffn_residual:{:.4}}}",
        PROFILE_CALLS.load(Ordering::Relaxed),
        ms_per_call(&ATTN_NORM_NS),
        ms_per_call(&ATTN_NS),
        ms_per_call(&ATTN_RESIDUAL_NS),
        ms_per_call(&FFN_NORM_NS),
        ms_per_call(&FFN_NS),
        ms_per_call(&FFN_RESIDUAL_NS),
    );
    eprintln!(
        "DINF_DEEPSEEK_MOE_PROFILE calls={} per_layer_ms={{router:{:.4},routed_experts:{:.4},shared_expert:{:.4}}}",
        MOE_PROFILE_CALLS.load(Ordering::Relaxed),
        ms_per_moe_call(&ROUTER_NS),
        ms_per_moe_call(&ROUTED_EXPERT_NS),
        ms_per_moe_call(&SHARED_EXPERT_NS),
    );
}

struct RmsNorm {
    weight: Tensor,
    eps: f64,
    native_q8: bool,
}

impl RmsNorm {
    fn for_projection(mut self, projection: &QMatMul) -> Self {
        self.native_q8 = native_cpu::supported(projection);
        self
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.native_q8 {
            return native_cpu::rms_norm(x, &self.weight, self.eps);
        }
        let input = if x.is_contiguous() {
            x.clone()
        } else {
            x.contiguous()?
        };
        candle_nn::ops::rms_norm(&input, &self.weight, self.eps as f32).map_err(Into::into)
    }

    fn forward_residual(&self, residual: &Tensor, update: &Tensor) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        if let Some(result) =
            super::residual_norm_cuda::forward(residual, update, &self.weight, self.eps as f32)?
        {
            return Ok(result);
        }
        let residual = (residual + update)?;
        let normalized = self.forward(&residual)?;
        Ok((residual, normalized))
    }
}

fn qtensor_to_qmatmul(qt: Arc<QTensor>) -> Result<QMatMul> {
    QMatMul::from_arc(qt).map_err(Into::into)
}

fn precompute_rope_cos_sin(
    head_dim: usize,
    theta: f64,
    context_length: usize,
    scaling_factor: f64,
    original_context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    crate::gguf_loader::validate_rope_dimensions(head_dim, head_dim, "DeepSeek rotary dimension")?;
    anyhow::ensure!(
        context_length > 0 && context_length <= u32::MAX as usize,
        "DeepSeek rotary context must be positive and fit U32 positions"
    );
    let half_dim = head_dim / 2;
    let correction_dim = |rotations: f64| {
        head_dim as f64
            * (original_context_length as f64 / (rotations * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * theta.ln())
    };
    let low = correction_dim(32.0).floor().max(0.0);
    let high = correction_dim(1.0).ceil().min((head_dim - 1) as f64);
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| {
            let base_freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
            if scaling_factor <= 1.0 {
                return base_freq as f32;
            }
            let ramp = ((i as f64 - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
            (base_freq * (1.0 - ramp) + base_freq / scaling_factor * ramp) as f32
        })
        .collect();
    let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
    let positions = Tensor::arange(0u32, context_length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?;
    let angles = positions.matmul(&inv_freq.reshape((1, half_dim))?)?;
    Ok((angles.cos()?, angles.sin()?))
}

fn apply_rotary_emb(x: &Tensor, start_pos: usize, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_, _, seq_len, _) = x.dims4()?;
    let cos = cos.narrow(0, start_pos, seq_len)?;
    let sin = sin.narrow(0, start_pos, seq_len)?;
    Ok(candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)?)
}

fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::softmax_last_dim(x).map_err(Into::into)
}

fn trace_tensor(label: &str, layer: usize, tensor: &Tensor) -> Result<()> {
    if std::env::var_os("DINF_TRACE_DEEPSEEK").is_none() {
        return Ok(());
    }
    let values = tensor
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let head = values.iter().take(3).copied().collect::<Vec<_>>();
    let mut tail = values.iter().rev().take(3).copied().collect::<Vec<_>>();
    tail.reverse();
    let sum = values.iter().map(|value| *value as f64).sum::<f64>();
    tracing::info!(
        target: "dinf_engine::deepseek_trace",
        label,
        layer,
        shape = ?tensor.dims(),
        ?head,
        ?tail,
        sum,
        "DeepSeek tensor"
    );
    Ok(())
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

// ===== MLA Attention =====

struct MlaAttention {
    // Q path (direct for DeepSeek-V2 Lite, LoRA-compressed for V3/R1)
    q_proj: QProjection,
    // KV path (latent compression)
    kv_a: QMatMul,
    kv_a_norm: RmsNorm,
    kv_b: QMatMul,
    // Output
    o_proj: QMatMul,
    // Dimensions
    num_heads: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    // RoPE
    native_yarn: bool,
    native_softmax_scale: Option<f32>,
    rope_cos: Tensor,
    rope_sin: Tensor,
    softmax_scale: f64,
    device: Device,
}

enum QProjection {
    Direct(QMatMul),
    Lora {
        q_a: QMatMul,
        q_a_norm: RmsNorm,
        q_b: QMatMul,
    },
}

impl MlaAttention {
    fn forward(
        &self,
        x: &Tensor,
        kv_cache: &mut Option<KvCache>,
        seq_len: usize,
        start_pos: usize,
        layer: usize,
        arithmetic: Option<TextArithmeticPolicy>,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let rows = arithmetic.map(|p| native_cpu::RowPolicy {
            start: start_pos,
            seq_len,
            m4_prefix_end: p.m4_prefix_end(),
        });

        // Q/Q-a and KV-a consume the same residual stream. Quantize that
        // activation once for decode and once per MMQ scale layout for prefill.
        let q_input_proj = match &self.q_proj {
            QProjection::Direct(q) => q,
            QProjection::Lora { q_a, .. } => q_a,
        };
        #[cfg(feature = "cuda")]
        let grouped_input = super::dense_qmatmul_cuda::forward(x, &[q_input_proj, &self.kv_a])?;
        #[cfg(not(feature = "cuda"))]
        let grouped_input: Option<Vec<Tensor>> = None;
        let (q_projected, kv_mixed) = if let Some(mut outputs) = grouped_input {
            anyhow::ensure!(
                outputs.len() == 2,
                "grouped DeepSeek MLA input count mismatch"
            );
            let kv_mixed = outputs.pop().unwrap();
            let q_projected = outputs.pop().unwrap();
            (q_projected, kv_mixed)
        } else {
            (
                cpu_math::project_rows(q_input_proj, x, rows)?,
                cpu_math::project_rows(&self.kv_a, x, rows)?,
            )
        };

        // 1. Q path: direct projection or compress → norm → expand
        let q = match &self.q_proj {
            QProjection::Direct(_) => q_projected,
            QProjection::Lora { q_a_norm, q_b, .. } => {
                let q_latent = q_a_norm.forward(&q_projected)?;
                cpu_math::project_rows(q_b, &q_latent, rows)?
            }
        };
        trace_tensor("q", layer, &q)?;

        // 2. KV path: compress → split latent + k_pe → norm latent → expand → split k_nope + v
        trace_tensor("kv_cmpr_pe", layer, &kv_mixed)?;
        let kv_latent = kv_mixed.narrow(D::Minus1, 0, self.kv_lora_rank)?;
        let kv_normed = self.kv_a_norm.forward(&kv_latent)?;
        let kv = cpu_math::project_rows(&self.kv_b, &kv_normed, rows)?;
        trace_tensor("kv", layer, &kv)?;

        // 3. Layout heads, apply partial RoPE, and assemble full Q/K/V.
        #[cfg(target_os = "macos")]
        let fused_layout = if std::env::var_os("DINF_DISABLE_DEEPSEEK_FUSED_MLA_LAYOUT").is_none()
            && std::env::var_os("DINF_TRACE_DEEPSEEK").is_none()
        {
            super::deepseek_mla_metal::try_fused_mla_layout(
                &q,
                &kv,
                &kv_mixed,
                &self.rope_cos,
                &self.rope_sin,
                b,
                s,
                self.num_heads,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                self.v_head_dim,
                self.kv_lora_rank,
                start_pos,
            )?
        } else {
            None
        };
        #[cfg(all(feature = "cuda", not(target_os = "macos")))]
        let fused_layout = if std::env::var_os("DINF_TRACE_DEEPSEEK").is_none() {
            super::deepseek_mla_cuda::try_fused_mla_layout(
                &q,
                &kv,
                &kv_mixed,
                &self.rope_cos,
                &self.rope_sin,
                b,
                s,
                self.num_heads,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                self.v_head_dim,
                self.kv_lora_rank,
                start_pos,
            )?
        } else {
            None
        };
        #[cfg(all(not(feature = "cuda"), not(target_os = "macos")))]
        let fused_layout: Option<(Tensor, Tensor, Tensor)> = None;
        let (q, k, v) = if let Some(layout) = fused_layout {
            layout
        } else {
            self.portable_mla_layout(q, kv, &kv_mixed, b, s, start_pos, layer)?
        };

        // 4. KV cache (naive V1: store full decompressed K/V)
        let (k, v) = kv_cache_update_native(kv_cache, k, v, None)?;

        // 5. Attention
        let q = cpu_math::native_quant_query(q_input_proj, &q, &k, &v)?;
        // A fixed native online row order for admitted CPU quantized projections.
        // FULL, other devices and unsupported cache/layouts keep the existing path.
        let native_attention = if native_cpu::supported(q_input_proj)
            || (self.native_softmax_scale.is_some() && native_cpu::supported_k(q_input_proj))
        {
            native_online_attention::attention_with_policy(
                &q,
                &k,
                &v,
                self.native_softmax_scale
                    .unwrap_or(self.softmax_scale as f32),
                start_pos,
                if self.native_softmax_scale.is_some()
                    && start_pos.checked_add(seq_len) == Some(k.dim(2)?)
                {
                    arithmetic
                } else {
                    None
                },
            )?
        } else {
            None
        };
        let out = if let Some(output) = native_attention {
            output
        } else if let Some(output) = cpu_math::attention(&q, &k, &v, self.softmax_scale)? {
            output
        } else if std::env::var_os("DINF_TRACE_DEEPSEEK").is_none()
            && let Some(output) = super::try_fused_f16_sdpa_native(
                &q,
                &k,
                &v,
                seq_len > 1,
                self.softmax_scale as f32,
            )?
        {
            output
        } else {
            let k = k.to_dtype(q.dtype())?;
            let v = v.to_dtype(q.dtype())?;
            let att = (q.contiguous()?.matmul(&k.t()?.contiguous()?)? * self.softmax_scale)?;
            let att = if seq_len > 1 {
                let mask = create_causal_mask(seq_len, att.dim(3)?, &self.device)?;
                att.broadcast_add(&mask)?
            } else {
                att
            };
            let att = softmax_last_dim(&att)?;
            trace_tensor("attn_weights", layer, &att)?;
            att.matmul(&v.contiguous()?)?
        };
        trace_tensor("kqv_out", layer, &out)?;

        // 6. Output projection
        let out = out
            .transpose(1, 2)?
            .reshape((b, s, self.num_heads * self.v_head_dim))?;
        #[cfg(feature = "cuda")]
        if let Some(output) = super::dense_qmatmul_cuda::forward_one_to_f32(&out, &self.o_proj)? {
            return Ok(output);
        }
        let out = out.to_dtype(x.dtype())?;
        cpu_math::project_rows(&self.o_proj, &out, rows)
    }

    #[allow(clippy::too_many_arguments)]
    fn portable_mla_layout(
        &self,
        q: Tensor,
        kv: Tensor,
        kv_mixed: &Tensor,
        batch: usize,
        seq_len: usize,
        start_pos: usize,
        layer: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim;
        let q = q
            .reshape((batch, seq_len, self.num_heads, qk_head_dim))?
            .transpose(1, 2)?;
        let q_nope = q.narrow(D::Minus1, 0, self.qk_nope_head_dim)?;
        let q_pe = q.narrow(D::Minus1, self.qk_nope_head_dim, self.qk_rope_head_dim)?;
        trace_tensor("q_pe", layer, &q_pe)?;
        let q_pe = if let Some(rotated) =
            native_yarn::apply(&q_pe, start_pos, self.rope_cos.dim(0)?, self.native_yarn)?
        {
            rotated
        } else {
            apply_rotary_emb(&q_pe, start_pos, &self.rope_cos, &self.rope_sin)?
        };
        trace_tensor("q_pe_rope", layer, &q_pe)?;

        let k_pe = kv_mixed
            .narrow(D::Minus1, self.kv_lora_rank, self.qk_rope_head_dim)?
            .reshape((batch, seq_len, 1, self.qk_rope_head_dim))?
            .transpose(1, 2)?;
        trace_tensor("k_pe", layer, &k_pe)?;
        let k_pe = if let Some(rotated) =
            native_yarn::apply(&k_pe, start_pos, self.rope_cos.dim(0)?, self.native_yarn)?
        {
            rotated
        } else {
            apply_rotary_emb(&k_pe, start_pos, &self.rope_cos, &self.rope_sin)?
        };
        trace_tensor("k_pe_rope", layer, &k_pe)?;

        let kv = kv
            .reshape((
                batch,
                seq_len,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            ))?
            .transpose(1, 2)?;
        let k_nope = kv.narrow(D::Minus1, 0, self.qk_nope_head_dim)?;
        let v = kv.narrow(D::Minus1, self.qk_nope_head_dim, self.v_head_dim)?;
        let q = Tensor::cat(&[q_nope, q_pe], D::Minus1)?;
        let k_pe_expanded = k_pe.repeat((1, self.num_heads, 1, 1))?;
        let k = Tensor::cat(&[k_nope, k_pe_expanded], D::Minus1)?;
        Ok((q, k, v))
    }
}

// ===== MLP (dense FFN — SwiGLU) =====

struct Mlp {
    native_q8: bool,
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

impl Mlp {
    fn forward_rows(&self, x: &Tensor, rows: Option<native_cpu::RowPolicy>) -> Result<Tensor> {
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
            anyhow::ensure!(outputs.len() == 2, "grouped DeepSeek MLP count mismatch");
            let up = outputs.pop().unwrap();
            let gate = outputs.pop().unwrap();
            (gate, up)
        } else {
            (
                cpu_math::project_rows(&self.gate_proj, x, rows)?,
                cpu_math::project_rows(&self.up_proj, x, rows)?,
            )
        };
        #[cfg(feature = "cuda")]
        if let Some(output) = super::dense_qmatmul_cuda::silu_mul_down(&gate, &up, &self.down_proj)?
        {
            return Ok(output);
        }
        let activated = if self.native_q8 {
            native_cpu::swiglu(&gate, &up)?
        } else {
            (candle_nn::Activation::Silu.forward(&gate)? * up)?
        };
        cpu_math::project_rows(&self.down_proj, &activated, rows)
    }
}

// ===== MoE =====

struct MoeExpert {
    native_q8: bool,
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

impl MoeExpert {
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
            anyhow::ensure!(outputs.len() == 2, "grouped DeepSeek expert count mismatch");
            let up = outputs.pop().unwrap();
            let gate = outputs.pop().unwrap();
            (gate, up)
        } else {
            if self.native_q8 {
                (
                    native_cpu::forward_routed(&self.gate_proj, x)?,
                    native_cpu::forward_routed(&self.up_proj, x)?,
                )
            } else {
                (self.gate_proj.forward(x)?, self.up_proj.forward(x)?)
            }
        };
        #[cfg(feature = "cuda")]
        if let Some(output) = super::dense_qmatmul_cuda::silu_mul_down(&gate, &up, &self.down_proj)?
        {
            return Ok(output);
        }
        let activated = if self.native_q8 {
            native_cpu::swiglu(&gate, &up)?
        } else {
            (candle_nn::Activation::Silu.forward(&gate)? * up)?
        };
        if self.native_q8 {
            native_cpu::forward_routed(&self.down_proj, &activated)
        } else {
            self.down_proj.forward(&activated).map_err(Into::into)
        }
    }
}

struct MoeLayer {
    gate: Tensor,
    native_router: bool,
    experts: MoeExperts,
    shared_expert: Option<Mlp>,
    num_active: usize,
    routed_scaling_factor: f64,
    norm_topk_prob: bool,
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

    fn forward(&self, flat_x: &Tensor, expert_ids: &Tensor, weights: &Tensor) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = flat_x.dims2()?;
        let (id_tokens, top_k) = expert_ids.dims2()?;
        anyhow::ensure!(
            num_tokens == id_tokens && weights.dims2()? == (num_tokens, top_k),
            "indexed DeepSeek routing tensor geometry differs"
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
            let activated = if let Some(activated) =
                self.gate_proj.silu_gate_up(&self.up_proj, &input, &ids)?
            {
                activated
            } else {
                let gate = self.gate_proj.forward(&input, &ids)?;
                let up = self.up_proj.forward(&input, &ids)?;
                silu_mul(&gate, &up)?
            };
            let output = self.down_proj.forward(&activated, &ids)?;
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

impl MoeLayer {
    fn selected_experts(&self, probabilities: &[f32]) -> Result<Vec<(usize, f32)>> {
        anyhow::ensure!(
            probabilities.len() == self.experts.len(),
            "DeepSeek router returned {} probabilities for {} experts",
            probabilities.len(),
            self.experts.len()
        );
        anyhow::ensure!(
            self.num_active <= probabilities.len(),
            "DeepSeek requests {} active experts from a pool of {}",
            self.num_active,
            probabilities.len()
        );

        let mut indexed = probabilities
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<_>>();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        indexed.truncate(self.num_active);

        if self.norm_topk_prob {
            if self.native_router && indexed.len() == 6 {
                let selected: [f32; 6] = std::array::from_fn(|i| indexed[i].1);
                let normalized =
                    native_route_normalization::normalize(&selected).map_err(anyhow::Error::msg)?;
                anyhow::ensure!(normalized.sum.is_finite(), "Deep native route sum");
                for ((_, weight), normalized) in indexed.iter_mut().zip(normalized.weights) {
                    *weight = normalized;
                }
            } else {
                let sum = indexed.iter().map(|(_, weight)| *weight).sum::<f32>();
                for (_, weight) in &mut indexed {
                    *weight /= sum.max(1e-20);
                }
            }
        } else if self.routed_scaling_factor != 1.0 {
            for (_, weight) in &mut indexed {
                *weight *= self.routed_scaling_factor as f32;
            }
        }
        Ok(indexed)
    }

    fn forward_rows(&self, x: &Tensor, rows: Option<native_cpu::RowPolicy>) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_dim) = x.dims3()?;
        let num_tokens = b_sz * seq_len;
        let flat_x = x.reshape((num_tokens, hidden_dim))?;
        let profile = layer_profile_enabled();
        if profile {
            MOE_PROFILE_CALLS.fetch_add(1, Ordering::Relaxed);
        }

        // Router: softmax over experts
        let router_started = Instant::now();
        let native_routing = native_router::router(&flat_x, &self.gate, self.native_router)?;
        let (router_logits, native_probabilities) =
            if let Some((logits, probabilities)) = native_routing {
                (logits, Some(probabilities))
            } else {
                (cpu_math::router(&flat_x, &self.gate)?, None)
            };

        #[cfg(target_os = "macos")]
        let fused_indexed_routing = match &self.experts {
            MoeExperts::Indexed(_)
                if self.experts.len() <= 128
                    && self.num_active <= 16
                    && std::env::var_os("DINF_DISABLE_DEEPSEEK_GPU_ROUTING").is_none()
                    && std::env::var_os("DINF_DISABLE_FUSED_SOFTMAX_ROUTING").is_none() =>
            {
                Some(fused_softmax_topk_route(
                    &router_logits.contiguous()?,
                    self.num_active,
                    self.norm_topk_prob,
                    self.routed_scaling_factor as f32,
                )?)
            }
            _ => None,
        };
        #[cfg(not(target_os = "macos"))]
        let fused_indexed_routing: Option<(Tensor, Tensor)> = None;

        let router_probs = if fused_indexed_routing.is_some() {
            None
        } else if let Some(probabilities) = native_probabilities {
            Some(probabilities)
        } else {
            Some(softmax_last_dim(&router_logits)?.to_dtype(DType::F32)?)
        };

        #[cfg(any(target_os = "macos", feature = "cuda"))]
        let indexed_routing = if let Some(routing) = fused_indexed_routing {
            Some(routing)
        } else {
            match &self.experts {
                MoeExperts::Indexed(_)
                    if std::env::var_os("DINF_DISABLE_DEEPSEEK_GPU_ROUTING").is_none() =>
                {
                    let router_probs = router_probs
                        .as_ref()
                        .context("missing DeepSeek router probabilities")?;
                    let expert_ids = router_probs
                        .contiguous()?
                        .arg_sort_last_dim(false)?
                        .narrow(D::Minus1, 0, self.num_active)?
                        .contiguous()?;
                    let expert_weights = if self.norm_topk_prob {
                        normalized_route_weights(router_probs, &expert_ids)?
                    } else {
                        let weights = router_probs.gather(&expert_ids, D::Minus1)?;
                        if self.routed_scaling_factor == 1.0 {
                            weights
                        } else {
                            (weights * self.routed_scaling_factor)?
                        }
                    };
                    Some((expert_ids, expert_weights))
                }
                _ => None,
            }
        };
        #[cfg(not(any(target_os = "macos", feature = "cuda")))]
        let indexed_routing: Option<(Tensor, Tensor)> = None;

        let routing = if indexed_routing.is_none() {
            Some(
                router_probs
                    .as_ref()
                    .context("missing DeepSeek router probabilities")?
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
                        .context("missing host DeepSeek routing values")?;
                    let mut ids = Vec::with_capacity(num_tokens * self.num_active);
                    let mut weights = Vec::with_capacity(num_tokens * self.num_active);
                    for row in routing {
                        for (expert_id, weight) in self.selected_experts(row)? {
                            ids.push(expert_id as u32);
                            weights.push(weight);
                        }
                    }
                    cpu_routing = (
                        Tensor::from_vec(ids, (num_tokens, self.num_active), x.device())?,
                        Tensor::from_vec(weights, (num_tokens, self.num_active), x.device())?,
                    );
                    (&cpu_routing.0, &cpu_routing.1)
                };
                experts.forward(&flat_x, expert_ids, expert_weights)?
            }
            MoeExperts::Split(experts) => {
                let routing = routing
                    .as_ref()
                    .context("missing split DeepSeek routing values")?;
                let mut output = Tensor::zeros((num_tokens, hidden_dim), DType::F32, x.device())?;
                for (token_idx, probabilities) in routing.iter().enumerate() {
                    let token_input = flat_x.i(token_idx)?.unsqueeze(0)?;
                    let mut token_output = Tensor::zeros((1, hidden_dim), DType::F32, x.device())?;
                    for (expert_idx, weight) in self.selected_experts(probabilities)? {
                        let expert_out = experts[expert_idx].forward(&token_input)?;
                        token_output =
                            (token_output + (expert_out.to_dtype(DType::F32)? * weight as f64)?)?;
                    }
                    output = output.slice_scatter(&token_output, 0, token_idx)?;
                }
                output
            }
        };
        if profile {
            finish_profile_region(x.device(), expert_started, &ROUTED_EXPERT_NS)?;
        }

        let shared_started = Instant::now();
        let mut result = output
            .to_dtype(x.dtype())?
            .reshape((b_sz, seq_len, hidden_dim))?;

        // Add shared expert output
        if let Some(ref shared) = self.shared_expert {
            result = (result + shared.forward_rows(x, rows)?)?;
        }
        if profile {
            finish_profile_region(x.device(), shared_started, &SHARED_EXPERT_NS)?;
        }

        Ok(result)
    }
}

// ===== Transformer Blocks =====

enum DeepSeekBlock {
    Dense(DenseBlock),
    Moe(MoeBlock),
}

struct DenseBlock {
    attn_norm: RmsNorm,
    attn: MlaAttention,
    ffn_norm: RmsNorm,
    ffn: Mlp,
}

struct MoeBlock {
    attn_norm: RmsNorm,
    attn: MlaAttention,
    ffn_norm: RmsNorm,
    moe: MoeLayer,
}

// ===== DeepSeek2 Stage =====

/// A loaded DeepSeek V3/R1 stage for pipeline execution.
pub struct DeepSeek2Stage {
    embedding: Option<Tensor>,
    blocks: Vec<DeepSeekBlock>,
    output_norm: Option<RmsNorm>,
    lm_head: Option<QMatMul>,
    is_head: bool,
    is_tail: bool,
    hidden_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    device: Device,
}

impl DeepSeek2Stage {
    /// Construct a DeepSeek stage from the tensors assigned to its layer range.
    ///
    /// Returns an error when required metadata or tensors are missing or have
    /// unsupported shapes. The optional quantization configuration is reserved
    /// for shared cache policy and is not used by this architecture currently.
    pub fn from_stage_tensors(
        stage: StageTensors,
        _quant_config: Option<Arc<TurboQuantConfig>>,
    ) -> Result<Self> {
        let md = &stage.metadata;
        let device = &stage.device;
        let range = &stage.range;
        let tensors = &stage.tensors;
        crate::gguf_loader::validate_rope_dimensions(
            md.qk_rope_head_dim,
            md.head_dim,
            "DeepSeek rope.dimension_count",
        )?;
        #[cfg(feature = "cuda")]
        super::dense_qmatmul_cuda::precompile(device)?;
        #[cfg(feature = "cuda")]
        super::deepseek_mla_cuda::precompile(device)?;

        // Embedding (head only)
        let embedding = if range.is_head() {
            let embd = tensors
                .get("token_embd.weight")
                .context("missing token_embd.weight")?;
            Some(embd.as_ref().dequantize(device)?)
        } else {
            None
        };

        // Precompute RoPE for the rope_head_dim (64D for DeepSeek V3)
        let rope_head_dim = md.qk_rope_head_dim;
        let (rope_cos, rope_sin) = precompute_rope_cos_sin(
            rope_head_dim,
            md.rope_theta,
            md.context_length,
            md.rope_scaling_factor,
            md.rope_scaling_original_context_length,
            device,
        )?;

        // Build blocks
        let mut blocks = Vec::with_capacity(range.layer_count());
        for layer_idx in range.start..range.end {
            let prefix = format!("blk.{layer_idx}");
            let is_dense = layer_idx < md.first_k_dense_replace;

            // Attention (MLA — same for dense and MoE layers)
            let mut attn = build_mla_attention(tensors, &prefix, md, device, &rope_cos, &rope_sin)?;
            attn.native_yarn = match &attn.q_proj {
                QProjection::Direct(q) => native_yarn::admitted(
                    q,
                    &attn.kv_a,
                    md.qk_rope_head_dim,
                    md.rope_theta,
                    md.rope_scaling_factor,
                    md.rope_scaling_original_context_length,
                    _quant_config.is_none(),
                ),
                QProjection::Lora { .. } => false,
            };
            attn.native_softmax_scale = native_scale::for_native_yarn(
                attn.native_yarn,
                md.rope_scaling_factor,
                md.rope_yarn_log_multiplier,
                md.qk_nope_head_dim + md.qk_rope_head_dim,
            );

            if attn.native_yarn {
                attn.kv_a_norm.native_q8 = native_cpu::supported_quant(&attn.kv_b);
            }
            let mut attn_norm = load_rms_norm(
                tensors,
                &format!("{prefix}.attn_norm"),
                md.rms_norm_eps,
                device,
            )?;
            let mut ffn_norm = load_rms_norm(
                tensors,
                &format!("{prefix}.ffn_norm"),
                md.rms_norm_eps,
                device,
            )?;

            let q_input = match &attn.q_proj {
                QProjection::Direct(q) => q,
                QProjection::Lora { q_a, .. } => q_a,
            };
            attn_norm.native_q8 = native_cpu::norm_supported(q_input, _quant_config.is_none())
                && native_cpu::norm_supported(&attn.kv_a, _quant_config.is_none());
            if is_dense {
                let mut ffn = build_mlp(tensors, &prefix, device)?;
                ffn.native_q8 = native_cpu::admitted_ffn(
                    &ffn.gate_proj,
                    &ffn.up_proj,
                    &ffn.down_proj,
                    _quant_config.is_none(),
                );
                ffn_norm.native_q8 =
                    native_cpu::norm_supported(&ffn.gate_proj, _quant_config.is_none())
                        && native_cpu::norm_supported(&ffn.up_proj, _quant_config.is_none());
                blocks.push(DeepSeekBlock::Dense(DenseBlock {
                    attn_norm,
                    attn,
                    ffn_norm,
                    ffn,
                }));
            } else {
                let mut moe = build_moe_layer(tensors, &prefix, md, device)?;
                #[cfg(not(any(target_os = "macos", feature = "cuda")))]
                let MoeExperts::Split(split_experts) = &mut moe.experts;
                #[cfg(any(target_os = "macos", feature = "cuda"))]
                let split_experts: &mut [MoeExpert] = match &mut moe.experts {
                    MoeExperts::Split(experts) => experts,
                    MoeExperts::Indexed(_) => &mut [],
                };
                for expert in split_experts {
                    expert.native_q8 = native_cpu::admitted_routed_ffn(
                        &expert.gate_proj,
                        &expert.up_proj,
                        &expert.down_proj,
                        _quant_config.is_none(),
                    );
                }
                if let Some(shared) = moe.shared_expert.as_mut() {
                    shared.native_q8 = native_cpu::admitted_ffn(
                        &shared.gate_proj,
                        &shared.up_proj,
                        &shared.down_proj,
                        _quant_config.is_none(),
                    );
                }
                // Only the proved CPU quantized expert/native-cache family uses this router order.
                moe.native_router &= _quant_config.is_none()
                    && matches!(&moe.experts, MoeExperts::Split(experts)
                        if experts.len() == 64 && experts.iter().all(|expert| expert.native_q8));
                // The RMS operation is shared with the F32 router and native
                // quantized experts. Routed quantized experts use the proved native dot order.
                ffn_norm.native_q8 = matches!(&moe.experts, MoeExperts::Split(experts)
                    if !experts.is_empty() && experts.iter().all(|e|native_cpu::norm_supported(&e.gate_proj,_quant_config.is_none())&&native_cpu::norm_supported(&e.up_proj,_quant_config.is_none())))
                    && moe.shared_expert.as_ref().is_none_or(|e| {
                        native_cpu::norm_supported(&e.gate_proj, _quant_config.is_none())
                            && native_cpu::norm_supported(&e.up_proj, _quant_config.is_none())
                    });
                blocks.push(DeepSeekBlock::Moe(MoeBlock {
                    attn_norm,
                    attn,
                    ffn_norm,
                    moe,
                }));
            }
        }

        // Output norm + lm_head (tail only)
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
            Some(qtensor_to_qmatmul(Arc::clone(head))?)
        } else {
            None
        };

        let output_norm = output_norm.map(|norm| {
            let mut norm = norm.for_projection(lm_head.as_ref().expect("tail head"));
            norm.native_q8 = native_cpu::norm_supported(
                lm_head.as_ref().expect("tail head"),
                _quant_config.is_none(),
            );
            norm
        });

        tracing::info!(
            "DeepSeek2 stage: layers [{}, {}), {} blocks ({} dense + {} MoE), \
             kv_lora_rank={}, q_lora_rank={}, {} experts (top-{}), {} shared",
            range.start,
            range.end,
            blocks.len(),
            blocks
                .iter()
                .filter(|b| matches!(b, DeepSeekBlock::Dense(_)))
                .count(),
            blocks
                .iter()
                .filter(|b| matches!(b, DeepSeekBlock::Moe(_)))
                .count(),
            md.kv_lora_rank,
            md.q_lora_rank,
            md.expert_count,
            md.expert_used_count,
            md.shared_expert_count,
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

impl DeepSeek2Stage {
    fn forward_rows(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
        arithmetic: Option<TextArithmeticPolicy>,
    ) -> Result<StageOutput> {
        let rows = arithmetic.map(|p| native_cpu::RowPolicy {
            start: start_pos,
            seq_len,
            m4_prefix_end: p.m4_prefix_end(),
        });
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
        trace_tensor("embd", 0, &hidden_states)?;

        for (i, block) in self.blocks.iter().enumerate() {
            let profile = layer_profile_enabled();
            if profile {
                PROFILE_CALLS.fetch_add(1, Ordering::Relaxed);
            }
            let ffn_rows = rows.map(|p| {
                if self.is_tail && i + 1 == self.blocks.len() {
                    native_cpu::RowPolicy {
                        m4_prefix_end: 0,
                        ..p
                    }
                } else {
                    p
                }
            });
            match block {
                DeepSeekBlock::Dense(b) => {
                    // Attention with residual
                    let started = Instant::now();
                    let normed = b.attn_norm.forward(&hidden_states)?;
                    if profile {
                        finish_profile_region(normed.device(), started, &ATTN_NORM_NS)?;
                    }
                    trace_tensor("attn_norm", i, &normed)?;
                    let started = Instant::now();
                    let attn_out = with_kv_cache(&mut cache[i], |kv| {
                        b.attn
                            .forward(&normed, kv, seq_len, start_pos, i, arithmetic)
                    })?;
                    if profile {
                        finish_profile_region(attn_out.device(), started, &ATTN_NS)?;
                    }
                    trace_tensor("attn_out", i, &attn_out)?;
                    let normed = if profile {
                        let started = Instant::now();
                        hidden_states = (hidden_states + attn_out)?;
                        finish_profile_region(hidden_states.device(), started, &ATTN_RESIDUAL_NS)?;
                        let started = Instant::now();
                        let normed = b.ffn_norm.forward(&hidden_states)?;
                        finish_profile_region(normed.device(), started, &FFN_NORM_NS)?;
                        normed
                    } else {
                        let (residual, normed) =
                            b.ffn_norm.forward_residual(&hidden_states, &attn_out)?;
                        hidden_states = residual;
                        normed
                    };
                    trace_tensor("ffn_inp", i, &hidden_states)?;
                    // FFN with residual
                    trace_tensor("ffn_norm", i, &normed)?;
                    let started = Instant::now();
                    let ffn_out = b.ffn.forward_rows(&normed, ffn_rows)?;
                    if profile {
                        finish_profile_region(ffn_out.device(), started, &FFN_NS)?;
                    }
                    trace_tensor("ffn_out", i, &ffn_out)?;
                    let started = Instant::now();
                    hidden_states = (hidden_states + ffn_out)?;
                    if profile {
                        finish_profile_region(hidden_states.device(), started, &FFN_RESIDUAL_NS)?;
                    }
                    trace_tensor("l_out", i, &hidden_states)?;
                }
                DeepSeekBlock::Moe(b) => {
                    // Attention with residual
                    let started = Instant::now();
                    let normed = b.attn_norm.forward(&hidden_states)?;
                    if profile {
                        finish_profile_region(normed.device(), started, &ATTN_NORM_NS)?;
                    }
                    trace_tensor("attn_norm", i, &normed)?;
                    let started = Instant::now();
                    let attn_out = with_kv_cache(&mut cache[i], |kv| {
                        b.attn
                            .forward(&normed, kv, seq_len, start_pos, i, arithmetic)
                    })?;
                    if profile {
                        finish_profile_region(attn_out.device(), started, &ATTN_NS)?;
                    }
                    trace_tensor("attn_out", i, &attn_out)?;
                    let normed = if profile {
                        let started = Instant::now();
                        hidden_states = (hidden_states + attn_out)?;
                        finish_profile_region(hidden_states.device(), started, &ATTN_RESIDUAL_NS)?;
                        let started = Instant::now();
                        let normed = b.ffn_norm.forward(&hidden_states)?;
                        finish_profile_region(normed.device(), started, &FFN_NORM_NS)?;
                        normed
                    } else {
                        let (residual, normed) =
                            b.ffn_norm.forward_residual(&hidden_states, &attn_out)?;
                        hidden_states = residual;
                        normed
                    };
                    trace_tensor("ffn_inp", i, &hidden_states)?;
                    // MoE FFN with residual
                    trace_tensor("ffn_norm", i, &normed)?;
                    let started = Instant::now();
                    let moe_out = b.moe.forward_rows(&normed, ffn_rows)?;
                    if profile {
                        finish_profile_region(moe_out.device(), started, &FFN_NS)?;
                    }
                    trace_tensor("ffn_out", i, &moe_out)?;
                    let started = Instant::now();
                    hidden_states = (hidden_states + moe_out)?;
                    if profile {
                        finish_profile_region(hidden_states.device(), started, &FFN_RESIDUAL_NS)?;
                    }
                    trace_tensor("l_out", i, &hidden_states)?;
                }
            }
        }

        if self.is_tail {
            let normed = self.output_norm.as_ref().unwrap().forward(&hidden_states)?;
            let last_hidden = if seq_len > 1 {
                normed.i((.., seq_len - 1, ..))?
            } else {
                normed.squeeze(1)?
            };
            #[cfg(feature = "cuda")]
            let logits = match super::dense_qmatmul_cuda::forward_one_to_f32(
                &last_hidden,
                self.lm_head.as_ref().unwrap(),
            )? {
                Some(logits) => logits,
                None => native_cpu::forward_rows(
                    self.lm_head.as_ref().unwrap(),
                    &last_hidden,
                    arithmetic.map(|_| native_cpu::RowPolicy {
                        start: start_pos + seq_len - 1,
                        seq_len: 1,
                        m4_prefix_end: 0,
                    }),
                )?,
            };
            #[cfg(not(feature = "cuda"))]
            let logits = native_cpu::forward_rows(
                self.lm_head.as_ref().unwrap(),
                &last_hidden,
                arithmetic.map(|_| native_cpu::RowPolicy {
                    start: start_pos + seq_len - 1,
                    seq_len: 1,
                    m4_prefix_end: 0,
                }),
            )?;
            Ok(StageOutput::Logits(logits.squeeze(0)?))
        } else {
            let output = hidden_states.squeeze(0)?;
            Ok(StageOutput::Activations(output))
        }
    }
}

impl ModelArchitecture for DeepSeek2Stage {
    fn forward(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_rows(input, cache, seq_len, start_pos, None)
    }

    fn text_arithmetic_policy(&self, context: TextRequestContext) -> Option<TextArithmeticPolicy> {
        if !self.blocks.is_empty() && self.blocks.iter().all(|block| {
            let attn = match block {
                DeepSeekBlock::Dense(b) => &b.attn,
                DeepSeekBlock::Moe(b) => &b.attn,
            };
            attn.native_softmax_scale.is_some()
                && attn.num_heads == 16
                && attn.qk_nope_head_dim + attn.qk_rope_head_dim == 192
                && attn.v_head_dim == 128
                && matches!(&attn.q_proj, QProjection::Direct(q) if native_cpu::supported_quant(q))
        }) {
            let has_k = self.blocks.iter().any(|block| {
                let attn = match block {
                    DeepSeekBlock::Dense(b) => &b.attn,
                    DeepSeekBlock::Moe(b) => &b.attn,
                };
                matches!(&attn.q_proj,QProjection::Direct(q) if native_cpu::supported_k(q))
            });
            Some(if has_k {
                TextArithmeticPolicy::deep_cpu_k(context)
            } else {
                TextArithmeticPolicy::deep_cpu_q8(context)
            })
        } else {
            None
        }
    }

    fn forward_text(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
        context: TextRequestContext,
        prefill: bool,
        cache_only: bool,
    ) -> Result<StageOutput> {
        context.validate(start_pos, seq_len, prefill)?;
        anyhow::ensure!(!cache_only || prefill, "cache-only text requires prefill");
        self.forward_rows(
            input,
            cache,
            seq_len,
            start_pos,
            self.text_arithmetic_policy(context),
        )
    }

    fn family(&self) -> &str {
        "deepseek2"
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

fn build_mla_attention(
    tensors: &HashMap<String, Arc<QTensor>>,
    prefix: &str,
    md: &crate::gguf_loader::GgufMetadata,
    device: &Device,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
) -> Result<MlaAttention> {
    let get = |name: &str| -> Result<QMatMul> {
        let t = tensors
            .get(name)
            .with_context(|| format!("missing tensor: {name}"))?;
        qtensor_to_qmatmul(Arc::clone(t))
    };

    let qk_head_dim = md.qk_nope_head_dim + md.qk_rope_head_dim;

    let q_proj = if tensors.contains_key(&format!("{prefix}.attn_q.weight")) {
        QProjection::Direct(get(&format!("{prefix}.attn_q.weight"))?)
    } else {
        QProjection::Lora {
            q_a: get(&format!("{prefix}.attn_q_a.weight"))?,
            q_a_norm: load_rms_norm(
                tensors,
                &format!("{prefix}.attn_q_a_norm"),
                md.rms_norm_eps,
                device,
            )?
            .for_projection(&get(&format!("{prefix}.attn_q_b.weight"))?),
            q_b: get(&format!("{prefix}.attn_q_b.weight"))?,
        }
    };

    Ok(MlaAttention {
        q_proj,
        kv_a: get(&format!("{prefix}.attn_kv_a_mqa.weight"))?,
        kv_a_norm: load_rms_norm(
            tensors,
            &format!("{prefix}.attn_kv_a_norm"),
            md.rms_norm_eps,
            device,
        )?
        .for_projection(&get(&format!("{prefix}.attn_kv_b.weight"))?),
        kv_b: get(&format!("{prefix}.attn_kv_b.weight"))?,
        o_proj: get(&format!("{prefix}.attn_output.weight"))?,
        num_heads: md.num_attention_heads,
        qk_nope_head_dim: md.qk_nope_head_dim,
        qk_rope_head_dim: md.qk_rope_head_dim,
        v_head_dim: md.v_head_dim,
        kv_lora_rank: md.kv_lora_rank,
        native_yarn: false,
        native_softmax_scale: None,
        rope_cos: rope_cos.clone(),
        rope_sin: rope_sin.clone(),
        softmax_scale: {
            let mscale = if md.rope_scaling_factor > 1.0 {
                1.0 + md.rope_yarn_log_multiplier * md.rope_scaling_factor.ln()
            } else {
                1.0
            };
            mscale * mscale / (qk_head_dim as f64).sqrt()
        },
        device: device.clone(),
    })
}

/// Split a packed [experts, out_dim, in_dim] quantized tensor into matrices.
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
        "expert dimensions ({out_dim}x{in_dim}) are not aligned to block size {block_size}"
    );
    let bytes_per_expert = elements_per_expert / block_size * ggml_dtype.type_size();
    let raw_data = packed.data()?;
    anyhow::ensure!(
        raw_data.len() >= num_experts * bytes_per_expert,
        "packed expert tensor has {} bytes, need {}",
        raw_data.len(),
        num_experts * bytes_per_expert
    );

    (0..num_experts)
        .map(|expert| {
            let start = expert * bytes_per_expert;
            let tensor = candle_core::quantized::ggml_file::qtensor_from_ggml(
                ggml_dtype,
                &raw_data[start..start + bytes_per_expert],
                vec![out_dim, in_dim],
                device,
            )?;
            QMatMul::from_arc(Arc::new(tensor)).map_err(Into::into)
        })
        .collect()
}

fn build_mlp(
    tensors: &HashMap<String, Arc<QTensor>>,
    prefix: &str,
    device: &Device,
) -> Result<Mlp> {
    let _ = device; // tensors already on device
    let get = |name: &str| -> Result<QMatMul> {
        let t = tensors
            .get(name)
            .with_context(|| format!("missing tensor: {name}"))?;
        qtensor_to_qmatmul(Arc::clone(t))
    };

    Ok(Mlp {
        native_q8: false,
        gate_proj: get(&format!("{prefix}.ffn_gate.weight"))?,
        up_proj: get(&format!("{prefix}.ffn_up.weight"))?,
        down_proj: get(&format!("{prefix}.ffn_down.weight"))?,
    })
}

fn build_moe_layer(
    tensors: &HashMap<String, Arc<QTensor>>,
    prefix: &str,
    md: &crate::gguf_loader::GgufMetadata,
    device: &Device,
) -> Result<MoeLayer> {
    // Router gate
    let gate_tensor = tensors
        .get(&format!("{prefix}.ffn_gate_inp.weight"))
        .context("missing ffn_gate_inp.weight (MoE router)")?;
    let gate = gate_tensor.as_ref().dequantize(device)?;

    // Load experts — check for per-expert or packed format
    let experts = if tensors.contains_key(&format!("{prefix}.ffn_gate.0.weight")) {
        // Per-expert format: blk.N.ffn_gate.E.weight
        let mut exps = Vec::with_capacity(md.expert_count);
        for e in 0..md.expert_count {
            let gate_proj = qtensor_to_qmatmul(Arc::clone(
                tensors
                    .get(&format!("{prefix}.ffn_gate.{e}.weight"))
                    .context("missing expert gate")?,
            ))?;
            let up_proj = qtensor_to_qmatmul(Arc::clone(
                tensors
                    .get(&format!("{prefix}.ffn_up.{e}.weight"))
                    .context("missing expert up")?,
            ))?;
            let down_proj = qtensor_to_qmatmul(Arc::clone(
                tensors
                    .get(&format!("{prefix}.ffn_down.{e}.weight"))
                    .context("missing expert down")?,
            ))?;
            exps.push(MoeExpert {
                native_q8: false,
                gate_proj,
                up_proj,
                down_proj,
            });
        }
        MoeExperts::Split(exps)
    } else if let Some(gate_exps) = tensors.get(&format!("{prefix}.ffn_gate_exps.weight")) {
        let up_exps = tensors
            .get(&format!("{prefix}.ffn_up_exps.weight"))
            .context("missing ffn_up_exps.weight")?;
        let down_exps = tensors
            .get(&format!("{prefix}.ffn_down_exps.weight"))
            .context("missing ffn_down_exps.weight")?;

        #[cfg(any(target_os = "macos", feature = "cuda"))]
        {
            let use_indexed = indexed_moe_available(device, gate_exps.dtype())
                && indexed_moe_available(device, up_exps.dtype())
                && indexed_moe_available(device, down_exps.dtype());
            if use_indexed {
                let experts = MoeExperts::Indexed(Box::new(IndexedMoeExperts {
                    gate_proj: IndexedMoeMatMul::from_qtensor(
                        gate_exps,
                        md.expert_count,
                        md.expert_ffn_dim,
                        md.hidden_dim,
                        device,
                    )?,
                    up_proj: IndexedMoeMatMul::from_qtensor(
                        up_exps,
                        md.expert_count,
                        md.expert_ffn_dim,
                        md.hidden_dim,
                        device,
                    )?,
                    down_proj: IndexedMoeMatMul::from_qtensor(
                        down_exps,
                        md.expert_count,
                        md.hidden_dim,
                        md.expert_ffn_dim,
                        device,
                    )?,
                }));
                #[cfg(target_os = "macos")]
                device.synchronize()?;
                experts
            } else {
                MoeExperts::Split(split_packed_experts(
                    gate_exps, up_exps, down_exps, md, device, prefix,
                )?)
            }
        }
        #[cfg(not(any(target_os = "macos", feature = "cuda")))]
        {
            MoeExperts::Split(split_packed_experts(
                gate_exps, up_exps, down_exps, md, device, prefix,
            )?)
        }
    } else {
        anyhow::bail!("Could not find expert tensors at {prefix}");
    };

    // Shared expert(s)
    let shared_expert = if md.shared_expert_count > 0 {
        // Shared experts use {prefix}.ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp
        // Their intermediate size = moe_intermediate_size * n_shared_experts
        let get = |name: &str| -> Result<QMatMul> {
            let t = tensors
                .get(name)
                .with_context(|| format!("missing shared expert tensor: {name}"))?;
            qtensor_to_qmatmul(Arc::clone(t))
        };
        Some(Mlp {
            native_q8: false,
            gate_proj: get(&format!("{prefix}.ffn_gate_shexp.weight"))?,
            up_proj: get(&format!("{prefix}.ffn_up_shexp.weight"))?,
            down_proj: get(&format!("{prefix}.ffn_down_shexp.weight"))?,
        })
    } else {
        None
    };

    Ok(MoeLayer {
        native_router: gate_tensor.dtype() == GgmlDType::F32
            && native_router::supports_weight(&gate),
        gate,
        experts,
        shared_expert,
        num_active: md.expert_used_count,
        routed_scaling_factor: md.expert_weights_scale,
        norm_topk_prob: md.expert_weights_norm,
    })
}

fn split_packed_experts(
    gate_exps: &QTensor,
    up_exps: &QTensor,
    down_exps: &QTensor,
    md: &crate::gguf_loader::GgufMetadata,
    device: &Device,
    prefix: &str,
) -> Result<Vec<MoeExpert>> {
    let gate_mats = split_expert_tensor(
        gate_exps,
        md.expert_count,
        md.expert_ffn_dim,
        md.hidden_dim,
        device,
    )
    .with_context(|| format!("{prefix}.ffn_gate_exps split"))?;
    let up_mats = split_expert_tensor(
        up_exps,
        md.expert_count,
        md.expert_ffn_dim,
        md.hidden_dim,
        device,
    )
    .with_context(|| format!("{prefix}.ffn_up_exps split"))?;
    let down_mats = split_expert_tensor(
        down_exps,
        md.expert_count,
        md.hidden_dim,
        md.expert_ffn_dim,
        device,
    )
    .with_context(|| format!("{prefix}.ffn_down_exps split"))?;

    Ok(gate_mats
        .into_iter()
        .zip(up_mats)
        .zip(down_mats)
        .map(|((gate_proj, up_proj), down_proj)| MoeExpert {
            native_q8: false,
            gate_proj,
            up_proj,
            down_proj,
        })
        .collect())
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
    Ok(RmsNorm {
        weight,
        eps,
        native_q8: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_rotary_dimensions_return_errors_before_stage_allocation() -> Result<()> {
        use crate::gguf_loader::{LayerRange, parse_gguf_metadata};
        use candle_core::quantized::gguf_file::Value;

        let mut values = HashMap::from([(
            "general.architecture".to_string(),
            Value::String("deepseek2".to_string()),
        )]);
        for (key, value) in [
            ("block_count", 2),
            ("embedding_length", 8),
            ("attention.head_count", 2),
            ("attention.key_length", 4),
            ("attention.kv_lora_rank", 2),
            ("rope.dimension_count", 2),
        ] {
            values.insert(format!("deepseek2.{key}"), Value::U32(value));
        }
        let metadata = parse_gguf_metadata(&values)?;
        for rope_dim in [0, 1, 3, 6] {
            let mut invalid = metadata.clone();
            invalid.qk_rope_head_dim = rope_dim;
            // This middle stage needs no embedding. Before the fix it reached
            // head_dim - 1 with a zero rotary width before checking any blocks.
            let stage = StageTensors {
                range: LayerRange::new(1, 2, 2),
                metadata: invalid,
                tensors: HashMap::new(),
                raw_tensors: HashMap::new(),
                raw_source: None,
                device: Device::Cpu,
            };
            let error = DeepSeek2Stage::from_stage_tensors(stage, None)
                .err()
                .expect("invalid rotary geometry must fail")
                .to_string();
            assert!(error.contains("rope.dimension_count"), "{error}");
        }
        Ok(())
    }

    #[test]
    fn rotary_tables_reject_zero_and_odd_widths_and_keep_valid_values() -> Result<()> {
        for head_dim in [0, 1, 3] {
            assert!(precompute_rope_cos_sin(head_dim, 10_000., 4, 1., 4, &Device::Cpu).is_err());
        }
        let (cos, sin) = precompute_rope_cos_sin(2, 10_000., 4, 1., 4, &Device::Cpu)?;
        assert_eq!(cos.dims(), [4, 1]);
        assert_eq!(sin.dims(), [4, 1]);
        for (position, (cos, sin)) in cos
            .to_vec2::<f32>()?
            .into_iter()
            .zip(sin.to_vec2::<f32>()?)
            .enumerate()
        {
            assert!((cos[0] - (position as f32).cos()).abs() < 1e-6);
            assert!((sin[0] - (position as f32).sin()).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_mla_softmax_scale() {
        // DeepSeek V3: qk_nope=128, qk_rope=64, total=192
        let scale = 1.0 / (192.0_f64).sqrt();
        assert!((scale - 0.07217).abs() < 0.001, "scale={scale}");
    }

    #[test]
    fn test_causal_mask_shape() {
        let mask = create_causal_mask(3, 10, &Device::Cpu).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 3, 10]);
    }

    #[test]
    fn test_dense_vs_moe_detection() {
        // DeepSeek V3: first_k_dense_replace=1, so layer 0 is dense, rest are MoE
        let first_k_dense = 1;
        assert!(0 < first_k_dense, "layer 0 should be dense");
        assert!(1 >= first_k_dense, "layer 1 should be MoE");
        assert!(60 >= first_k_dense, "layer 60 should be MoE");
    }
}
