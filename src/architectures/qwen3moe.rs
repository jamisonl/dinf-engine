use std::sync::Arc;

use crate::qmatmul::QMatMul;
use anyhow::{Context, Result};
use candle_core::D;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Tensor};

#[cfg(target_os = "macos")]
#[cfg(any(target_os = "macos", feature = "cuda"))]
use super::indexed_moe::{
    IndexedMoeMatMul, indexed_moe_available, normalized_route_weights, silu_mul, weighted_route_sum,
};
use super::{
    KvCache, LayerCache, ModelArchitecture, StageInput, StageOutput, TextArithmeticPolicy,
    TextRequestContext, with_kv_cache,
};
use crate::gguf_loader::StageTensors;
use crate::turboquant::TurboQuantConfig;

mod attention;
mod bf16;
mod bf16_cpu_math;
mod cpu_math;
mod native_argsort;
mod native_cpu;
mod native_cpu_norm;

// DFlash submits complete feature/noise planes, so its native CPU row policy
// follows each call rather than a text request's prefill boundary.
pub(super) fn dflash_cpu_projection(matmul: &QMatMul, input: &Tensor) -> Result<Option<Tensor>> {
    if !native_cpu::supported_k(matmul)
        || !input.device().is_cpu()
        || input.dtype() != DType::F32
        || !matches!(input.rank(), 2 | 3)
    {
        return Ok(None);
    }
    let rows = input.dim(D::Minus2)?;
    if rows == 0 {
        return Ok(None);
    }
    native_cpu::project_rows(
        matmul,
        &input.contiguous()?,
        Some(native_cpu::RowPolicy {
            start: 0,
            seq_len: rows,
            m4_prefix_end: rows / 4 * 4,
        }),
    )
}

pub(super) fn dflash_cpu_swiglu(gate: &Tensor, up: &Tensor) -> Result<Option<Tensor>> {
    if cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && gate.device().is_cpu()
        && up.device().is_cpu()
        && gate.dtype() == DType::F32
        && up.dtype() == DType::F32
        && gate.dims() == up.dims()
        && gate
            .dims()
            .last()
            .is_some_and(|d| *d > 0 && d.is_multiple_of(4))
    {
        return native_cpu::swiglu(gate, up).map(Some);
    }
    Ok(None)
}

pub(super) fn dflash_cpu_rms_norm(
    input: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Result<Option<Tensor>> {
    if cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && input.device().is_cpu()
        && weight.device().is_cpu()
        && input.dtype() == DType::F32
        && weight.dtype() == DType::F32
    {
        return native_cpu_norm::rms_norm(input, weight, eps).map(Some);
    }
    Ok(None)
}

#[cfg(target_os = "macos")]
mod native_head_norm_metal;
#[cfg(target_os = "macos")]
mod native_o_metal;
#[cfg(target_os = "macos")]
mod native_qkv_metal;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) mod native_rope_compat;
#[cfg(target_os = "macos")]
mod native_rotary_metal;
mod native_route_normalization;
mod native_router;
#[cfg(target_os = "macos")]
mod native_router_metal;
use bf16::Bf16Experts;

/// Extra temporary storage allowed for grouped BF16 Metal prefill.
pub(crate) const BF16_GROUP_WORKSPACE_BYTES: usize = 256 * 1024 * 1024;

/// Maximum temporary storage for the native-cache F32 attention path.
pub(crate) const F32_ATTENTION_WORKSPACE_BYTES: usize = 512 * 1024 * 1024;

/// A pipeline stage running the Qwen3-MoE architecture.
///
/// Features: GQA with QK normalization, Mixture-of-Experts FFN with top-k routing.
pub struct Qwen3MoeStage {
    embedding: Option<Tensor>,
    blocks: Vec<Qwen3MoeBlock>,
    output_norm: Option<RmsNorm>,
    lm_head: Option<QMatMul>,
    head_native_quant: bool,
    is_head: bool,
    is_tail: bool,
    hidden_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    device: Device,
}

struct Qwen3MoeBlock {
    attn_norm: RmsNorm,
    attn: Attention,
    ffn_norm: RmsNorm,
    moe: MoeLayer,
}

struct Attention {
    #[cfg(target_os = "macos")]
    native_qkv: Option<native_qkv_metal::NativeQkv>,
    /// Metal 4 tensor-core Q, K, V and output projections for prompt chunks.
    #[cfg(target_os = "macos")]
    prompt_projections: Option<Box<[super::tensor_qmatmul_metal::MetalTensorQMatMul; 4]>>,
    #[cfg(target_os = "macos")]
    native_o: Option<native_o_metal::NativeO>,
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
    #[cfg(target_os = "macos")]
    native_bf16_rotary: bool,
    rope_cos: Tensor,
    rope_sin: Tensor,
    quant_config: Option<Arc<TurboQuantConfig>>,
}

struct MoeLayer {
    #[cfg(target_os = "macos")]
    native_metal_router: Option<native_router_metal::NativeRouter>,
    /// Router weights: [num_experts, hidden_dim]
    gate: Tensor,
    experts: MoeExperts,
    num_active: usize,
    native_router: bool,
    // Keep native tie order separate from FULL BF16 router arithmetic.
    native_k_argsort: bool,
}

struct MoeExpert {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
    native_quant: bool,
}

struct RmsNorm {
    weight: Tensor,
    eps: f64,
    native_quant: bool,
    native_bf16: bool,
}

impl RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.native_quant {
            return native_cpu::rms_norm(x, &self.weight, self.eps);
        }
        if self.native_bf16
            && x.device().is_cpu()
            && x.dtype() == DType::F32
            && self.weight.device().is_cpu()
            && self.weight.dtype() == DType::F32
        {
            return native_cpu_norm::rms_norm(x, &self.weight, self.eps);
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
        // One Metal dispatch for the add and the norm.
        #[cfg(target_os = "macos")]
        if !self.native_quant
            && let Some(result) = super::fused_norm_metal::try_fused_add_rms_norm(
                residual,
                update,
                &self.weight,
                self.eps as f32,
            )?
        {
            return Ok(result);
        }
        let residual = (residual + update)?;
        let normalized = self.forward(&residual)?;
        Ok((residual, normalized))
    }
}

fn project_rows(
    weights: &QMatMul,
    input: &Tensor,
    native_cache: bool,
    row_policy: Option<native_cpu::RowPolicy>,
) -> Result<Tensor> {
    if let Some(output) = bf16::native_projection(weights, input, native_cache)? {
        return Ok(output);
    }
    cpu_math::project_quant_rows(weights, input, native_cache, row_policy)
}

impl Attention {
    fn forward(
        &self,
        x: &Tensor,
        kv_cache: &mut Option<KvCache>,
        seq_len: usize,
        start_pos: usize,
        arithmetic: Option<TextArithmeticPolicy>,
    ) -> Result<Tensor> {
        self.forward_rows(x, kv_cache, seq_len, start_pos, None, arithmetic)
    }

    fn forward_rows(
        &self,
        x: &Tensor,
        kv_cache: &mut Option<KvCache>,
        seq_len: usize,
        start_pos: usize,
        row_policy: Option<native_cpu::RowPolicy>,
        arithmetic: Option<TextArithmeticPolicy>,
    ) -> Result<Tensor> {
        let (b_sz, _, _) = x.dims3()?;

        #[cfg(feature = "cuda")]
        let grouped_qkv = if seq_len == 1 {
            super::dense_qmatmul_cuda::forward(x, &[&self.q_proj, &self.k_proj, &self.v_proj])?
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let grouped_qkv: Option<Vec<Tensor>> = None;
        #[cfg(target_os = "macos")]
        let native_qkv = if self.quant_config.is_none()
            && native_qkv_metal::supports(&self.q_proj, &self.k_proj, &self.v_proj)
            && let (Some(native), Some(policy)) = (&self.native_qkv, arithmetic)
        {
            native.forward(x, start_pos, policy)?
        } else {
            None
        };
        #[cfg(not(target_os = "macos"))]
        let native_qkv: Option<(Tensor, Tensor, Tensor)> = {
            let _ = arithmetic;
            None
        };
        #[cfg(target_os = "macos")]
        let native_qkv_used = native_qkv.is_some();
        let (q, k, v) = if let Some(outputs) = native_qkv {
            outputs
        } else if let Some(mut outputs) = grouped_qkv {
            anyhow::ensure!(outputs.len() == 3, "grouped Qwen3-MoE QKV count mismatch");
            let v = outputs.pop().unwrap();
            let k = outputs.pop().unwrap();
            let q = outputs.pop().unwrap();
            (q, k, v)
        } else {
            #[cfg(target_os = "macos")]
            if seq_len > 1
                && x.dtype() == DType::F32
                && let Some(proj) = self.prompt_projections.as_deref()
            {
                (
                    proj[0].forward(x)?,
                    proj[1].forward(x)?,
                    proj[2].forward(x)?,
                )
            } else {
                (
                    project_rows(&self.q_proj, x, self.quant_config.is_none(), row_policy)?,
                    project_rows(&self.k_proj, x, self.quant_config.is_none(), row_policy)?,
                    project_rows(&self.v_proj, x, self.quant_config.is_none(), row_policy)?,
                )
            }
            #[cfg(not(target_os = "macos"))]
            (
                project_rows(&self.q_proj, x, self.quant_config.is_none(), row_policy)?,
                project_rows(&self.k_proj, x, self.quant_config.is_none(), row_policy)?,
                project_rows(&self.v_proj, x, self.quant_config.is_none(), row_policy)?,
            )
        };

        // The proved D128 norm policy belongs only to the admitted Q4 QKV path.
        // Other weights, epsilon values and devices retain the original norm path.
        #[cfg(target_os = "macos")]
        let native_qk_norm = native_qkv_used
            && self.q_norm.eps as f32 == 1e-6f32
            && self.k_norm.eps as f32 == 1e-6f32
            && self.q_norm.weight.dims() == [128]
            && self.k_norm.weight.dims() == [128]
            && self.q_norm.weight.dtype() == DType::F32
            && self.k_norm.weight.dtype() == DType::F32
            && self.q_norm.weight.device().same_device(q.device())
            && self.k_norm.weight.device().same_device(k.device());
        #[cfg(target_os = "macos")]
        let fused_qk_layout_norm = if !native_qk_norm
            && seq_len > 1
            && std::env::var_os("DINF_DISABLE_QWEN3_FUSED_QK_LAYOUT_NORM").is_none()
        {
            // Fold the NeoX rope into the per-head norm when the tables cover
            // this chunk; otherwise normalize here and rotate below.
            let norm_rope =
                |input: &Tensor, norm: &RmsNorm, heads: usize| -> Result<Option<(Tensor, bool)>> {
                    if let Some(output) = super::fused_norm_metal::try_fused_head_rms_norm_rope(
                        input,
                        &norm.weight,
                        &self.rope_cos,
                        &self.rope_sin,
                        start_pos,
                        b_sz,
                        seq_len,
                        heads,
                        self.head_dim,
                        norm.eps as f32,
                    )? {
                        return Ok(Some((output, true)));
                    }
                    Ok(super::fused_norm_metal::try_fused_head_rms_norm(
                        input,
                        &norm.weight,
                        b_sz,
                        seq_len,
                        heads,
                        self.head_dim,
                        norm.eps as f32,
                    )?
                    .map(|output| (output, false)))
                };
            let q = norm_rope(&q, &self.q_norm, self.num_heads)?;
            let k = norm_rope(&k, &self.k_norm, self.num_kv_heads)?;
            match (q, k) {
                (Some(q), Some(k)) => Some((q, k)),
                _ => None,
            }
        } else {
            None
        };
        #[cfg(not(target_os = "macos"))]
        let fused_qk_layout_norm: Option<((Tensor, bool), (Tensor, bool))> = None;

        // Reshape to [batch, heads, seq_len, head_dim]
        // Rope is already applied when both fused outputs carry it.
        let mut roped = false;
        let (q, k, v, norms_fused) =
            if let Some(((q, q_roped), (k, k_roped))) = fused_qk_layout_norm {
                if q_roped != k_roped {
                    anyhow::bail!("fused Qwen3 Q/K rope availability differs");
                }
                roped = q_roped;
                let v = v
                    .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?;
                (q, k, v, true)
            } else if seq_len == 1 {
                let q = q.reshape((b_sz, self.num_heads, 1, self.head_dim))?;
                let k = k.reshape((b_sz, self.num_kv_heads, 1, self.head_dim))?;
                let v = v.reshape((b_sz, self.num_kv_heads, 1, self.head_dim))?;
                (q, k, v, false)
            } else {
                let q = q
                    .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?;
                let k = k
                    .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?;
                let v = v
                    .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?;
                (q, k, v, false)
            };

        // QK normalization: apply per-head-dim RMS norm to Q and K
        let normalize = |value: Tensor, norm: &RmsNorm| -> Result<Tensor> {
            if norms_fused {
                return Ok(value);
            }
            #[cfg(target_os = "macos")]
            if native_qk_norm {
                return native_head_norm_metal::forward(&value, &norm.weight, norm.eps as f32)?
                    .context("admitted Qwen Q4 native head norm declined");
            }
            norm.forward(&value)
        };
        let q = normalize(q, &self.q_norm)?;
        let k = normalize(k, &self.k_norm)?;

        // Apply RoPE
        let rotary = |projection: &QMatMul, input: &Tensor| -> Result<Tensor> {
            if let Some(output) = cpu_math::quantized_rotary(
                projection,
                input,
                start_pos,
                self.rope_theta,
                self.rope_cos.dim(0)?,
                self.quant_config.is_none(),
            )? {
                return Ok(output);
            }
            if let Some(output) = bf16_cpu_math::bf16_rotary(
                projection,
                input,
                start_pos,
                self.rope_theta,
                self.rope_cos.dim(0)?,
                self.quant_config.is_none(),
            )? {
                return Ok(output);
            }
            #[cfg(target_os = "macos")]
            if (self.native_bf16_rotary || native_qkv_used)
                && let Some(output) = native_rotary_metal::forward(
                    input,
                    start_pos,
                    self.rope_theta,
                    self.rope_cos.dim(0)?,
                    self.quant_config.is_none(),
                )?
            {
                return Ok(output);
            }
            apply_rotary_emb(input, start_pos, &self.rope_cos, &self.rope_sin)
        };
        let (q, k) = if roped {
            (q, k)
        } else {
            (rotary(&self.q_proj, &q)?, rotary(&self.k_proj, &k)?)
        };

        let (q, k, v, rotation) =
            super::prepare_quantized_attention(kv_cache, q, k, v, self.quant_config.as_deref())?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_output = if let Some(output) = cpu_math::quantized_attention(
            &self.q_proj,
            &q,
            &k,
            &v,
            self.quant_config.is_none(),
            start_pos,
            arithmetic,
        )? {
            output
        } else if let Some(output) = bf16_cpu_math::bf16_attention_rows(
            &self.q_proj,
            &q,
            &k,
            &v,
            self.quant_config.is_none(),
            start_pos,
            arithmetic,
        )? {
            output
        } else if let Some(output) = cpu_math::attention(&q, &k, &v, scale)? {
            output
        } else if let Some(output) =
            attention::try_metal_f32_sdpa_from_native_f16(&q, &k, &v, seq_len > 1, scale as f32)?
        {
            output
        } else if let Some(output) =
            super::try_fused_f16_sdpa_native(&q, &k, &v, seq_len > 1, scale as f32)?
        {
            output
        } else {
            let n_rep = self.num_heads / self.num_kv_heads;
            let k = repeat_kv(k.to_dtype(q.dtype())?, n_rep)?;
            let v = repeat_kv(v.to_dtype(q.dtype())?, n_rep)?;
            let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
            let attn_weights = if seq_len > 1 {
                let mask = create_causal_mask(seq_len, attn_weights.dim(3)?, x.device())?;
                attn_weights.broadcast_add(&mask)?
            } else {
                attn_weights
            };
            softmax_last_dim(&attn_weights)?.matmul(&v)?
        };
        let attn_output = super::restore_quantized_attention_output(attn_output, rotation)?;

        #[cfg(target_os = "macos")]
        let attn_output = if std::env::var_os("DINF_DISABLE_FUSED_ATTENTION_OUTPUT_LAYOUT")
            .is_none()
            && let Some(output) =
                super::qkv_layout_metal::try_fused_attention_output_layout(&attn_output)?
        {
            output
        } else {
            attn_output
                .to_dtype(x.dtype())?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b_sz, seq_len, ()))?
        };
        #[cfg(not(target_os = "macos"))]
        let attn_output =
            attn_output
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b_sz, seq_len, ()))?;
        #[cfg(all(feature = "cuda", not(target_os = "macos")))]
        if let Some(output) =
            super::dense_qmatmul_cuda::forward_one_to_f32(&attn_output, &self.o_proj)?
        {
            return Ok(output);
        }
        #[cfg(not(target_os = "macos"))]
        let attn_output = attn_output.to_dtype(x.dtype())?;
        #[cfg(target_os = "macos")]
        if native_qkv_used
            && self.quant_config.is_none()
            && native_o_metal::supports(&self.o_proj)
            && let (Some(native), Some(policy)) = (&self.native_o, arithmetic)
            && let Some(output) = native.forward(&attn_output, start_pos, policy)?
        {
            return Ok(output);
        }
        #[cfg(target_os = "macos")]
        if seq_len > 1
            && let Some(proj) = self.prompt_projections.as_deref()
        {
            return Ok(proj[3].forward(&attn_output.to_dtype(DType::F32)?)?);
        }
        let output = project_rows(
            &self.o_proj,
            &attn_output,
            self.quant_config.is_none(),
            row_policy,
        )?;
        Ok(output)
    }
}

impl MoeExpert {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.native_quant {
            let gate = native_cpu::forward_routed(&self.gate_proj, x)?;
            let up = native_cpu::forward_routed(&self.up_proj, x)?;
            let activated = native_cpu::swiglu(&gate, &up)?;
            return native_cpu::forward_routed(&self.down_proj, &activated);
        }
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = (candle_nn::Activation::Silu.forward(&gate)? * up)?;
        self.down_proj.forward(&activated).map_err(Into::into)
    }
}

#[cfg(target_os = "macos")]
fn native_metal_expert_types(
    gate: &QTensor,
    up: &QTensor,
    down: &QTensor,
    device: &Device,
) -> bool {
    gate.dtype() == GgmlDType::Q4K
        && up.dtype() == GgmlDType::Q4K
        && matches!(down.dtype(), GgmlDType::Q4K | GgmlDType::Q6K)
        && gate.shape().dims() == [128, 768, 2048]
        && up.shape().dims() == [128, 768, 2048]
        && down.shape().dims() == [128, 2048, 768]
        // Packed expert banks load on CPU; from_qtensor uploads them to this device.
        && device.is_metal()
        && [gate, up, down]
            .iter()
            .all(|tensor| tensor.device().is_cpu() || tensor.device().same_device(device))
}

#[cfg(any(target_os = "macos", feature = "cuda"))]
struct IndexedMoeExperts {
    #[cfg(target_os = "macos")]
    native_vector: bool,
    gate_proj: IndexedMoeMatMul,
    up_proj: IndexedMoeMatMul,
    down_proj: IndexedMoeMatMul,
}

#[cfg(any(target_os = "macos", feature = "cuda"))]
impl IndexedMoeExperts {
    const MAX_ROUTE_ROWS_PER_DISPATCH: usize = 4096;
    #[cfg(target_os = "macos")]
    const MATRIX_MIN_TOKENS: usize = 32;
    /// Below this the projections may use the vector kernel, which takes IDs.
    #[cfg(target_os = "macos")]
    const ROUTE_PLAN_MIN_TOKENS: usize = 64;

    fn forward(
        &self,
        flat_x: &Tensor,
        expert_ids: &Tensor,
        weights: &Tensor,
        _native_request: bool,
    ) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = flat_x.dims2()?;
        let (id_tokens, top_k) = expert_ids.dims2()?;
        anyhow::ensure!(
            num_tokens == id_tokens && weights.dims2()? == (num_tokens, top_k),
            "indexed Qwen3-MoE routing tensor geometry differs"
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
            // On Metal, sanitize the route IDs once for gate, up and down
            // (this also converts the fused route's f32 IDs to u32).
            #[cfg(target_os = "macos")]
            let sanitized = self.gate_proj.metal_sanitize_ids(&ids)?;
            #[cfg(not(target_os = "macos"))]
            let sanitized: Option<Tensor> = None;
            let ids = sanitized.clone().unwrap_or(ids);
            let route_weights = weights.narrow(0, start, len)?;
            let project = |projection: &IndexedMoeMatMul, x: &Tensor| -> Result<Tensor> {
                // Like llama.cpp (ggml_metal_op_mul_mat_id_use_mm), batches of
                // 32 or more tokens use the matrix kernel; only short batches
                // keep the per-token vector kernel.
                #[cfg(target_os = "macos")]
                {
                    let native =
                        self.native_vector && _native_request && len < Self::MATRIX_MIN_TOKENS;
                    if sanitized.is_some() {
                        return projection.forward_metal_sanitized(x, &ids, native);
                    }
                    if native {
                        return projection.forward_native_metal_vector(x, &ids);
                    }
                }
                projection.forward(x, &ids)
            };
            // Batches on the matrix kernel share one set of route tables
            // across gate, up and down instead of rebuilding them per call.
            #[cfg(target_os = "macos")]
            let plan = if len >= Self::ROUTE_PLAN_MIN_TOKENS
                && !(self.native_vector && _native_request && len < Self::MATRIX_MIN_TOKENS)
            {
                self.gate_proj.metal_route_plan(&ids)?
            } else {
                None
            };
            #[cfg(target_os = "macos")]
            if let Some(plan) = plan.as_ref() {
                let activated = match self.gate_proj.forward_gate_up_silu_with_metal_route_plan(
                    &self.up_proj,
                    &input,
                    plan,
                )? {
                    Some(activated) => activated,
                    None => {
                        let gate = self.gate_proj.forward_with_metal_route_plan(&input, plan)?;
                        let up = self.up_proj.forward_with_metal_route_plan(&input, plan)?;
                        silu_mul(&gate, &up)?
                    }
                };
                let output = self
                    .down_proj
                    .forward_with_metal_route_plan(&activated, plan)?;
                outputs.push(weighted_route_sum(&output, &route_weights)?);
                continue;
            }
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
    Bf16(Box<Bf16Experts>),
    #[cfg(any(target_os = "macos", feature = "cuda"))]
    Indexed(Box<IndexedMoeExperts>),
}

impl MoeExperts {
    fn len(&self) -> usize {
        match self {
            Self::Split(experts) => experts.len(),
            Self::Bf16(experts) => experts.len(),
            #[cfg(any(target_os = "macos", feature = "cuda"))]
            Self::Indexed(experts) => experts.gate_proj.num_experts(),
        }
    }
}

impl MoeLayer {
    fn selected_experts(&self, probabilities: &[f32]) -> Result<Vec<(usize, f32)>> {
        anyhow::ensure!(
            probabilities.len() == self.experts.len(),
            "Qwen3-MoE router returned {} probabilities for {} experts",
            probabilities.len(),
            self.experts.len()
        );
        anyhow::ensure!(
            self.num_active <= probabilities.len(),
            "Qwen3-MoE requests {} active experts from a pool of {}",
            self.num_active,
            probabilities.len()
        );

        let mut indexed = if self.native_k_argsort {
            native_argsort::descending_indices(probabilities)
                .map_err(anyhow::Error::msg)?
                .into_iter()
                .map(|index| (index, probabilities[index]))
                .collect::<Vec<_>>()
        } else {
            let mut indexed = probabilities
                .iter()
                .copied()
                .enumerate()
                .collect::<Vec<_>>();
            indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
            indexed
        };
        indexed.truncate(self.num_active);
        if self.native_router {
            let values: Vec<_> = indexed.iter().map(|(_, weight)| *weight).collect();
            let normalized =
                native_route_normalization::normalize(&values).map_err(anyhow::Error::msg)?;
            anyhow::ensure!(normalized.sum.is_finite(), "Qwen native route sum");
            for ((_, weight), normalized) in indexed.iter_mut().zip(normalized.weights) {
                *weight = normalized;
            }
            return Ok(indexed);
        }
        let weight_sum = indexed.iter().map(|(_, weight)| *weight).sum::<f32>();
        for (_, weight) in &mut indexed {
            *weight /= weight_sum.max(1e-20);
        }
        Ok(indexed)
    }

    fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        arithmetic: Option<TextArithmeticPolicy>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_dim) = x.dims3()?;
        let num_tokens = b_sz * seq_len;
        let flat_x = x.reshape((num_tokens, hidden_dim))?;

        #[cfg(target_os = "macos")]
        let native_logits =
            if let (Some(router), Some(policy)) = (&self.native_metal_router, arithmetic) {
                router.forward(x, start_pos, policy)?
            } else {
                None
            };
        #[cfg(not(target_os = "macos"))]
        let native_logits: Option<Tensor> = {
            let _ = (start_pos, arithmetic);
            None
        };
        let (router_logits, native_probabilities) = if let Some(logits) = native_logits {
            (logits.reshape((num_tokens, self.experts.len()))?, None)
        } else {
            match native_router::router(&flat_x, &self.gate, self.native_router)? {
                Some((logits, probabilities)) => (logits, Some(probabilities)),
                None => (cpu_math::router(&flat_x, &self.gate)?, None),
            }
        };

        #[cfg(target_os = "macos")]
        let fused_indexed_routing = match &self.experts {
            MoeExperts::Indexed(_)
                if self.experts.len() <= 128
                    && self.num_active <= 16
                    && std::env::var_os("DINF_DISABLE_QWEN3_GPU_ROUTING").is_none()
                    && std::env::var_os("DINF_DISABLE_FUSED_SOFTMAX_ROUTING").is_none() =>
            {
                // f32 IDs: the experts' single sanitize pass converts them.
                Some(super::fused_route_metal::fused_softmax_topk_route_f32_ids(
                    &router_logits.contiguous()?,
                    self.num_active,
                    true,
                    1.0,
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
                    if std::env::var_os("DINF_DISABLE_QWEN3_GPU_ROUTING").is_none() =>
                {
                    let router_probs = router_probs
                        .as_ref()
                        .context("missing Qwen3-MoE router probabilities")?;
                    let expert_ids = router_probs
                        .contiguous()?
                        .arg_sort_last_dim(false)?
                        .narrow(D::Minus1, 0, self.num_active)?
                        .contiguous()?;
                    let expert_weights = normalized_route_weights(router_probs, &expert_ids)?;
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
                    .context("missing Qwen3-MoE router probabilities")?
                    .to_vec2::<f32>()?,
            )
        } else {
            None
        };

        let output = match &self.experts {
            MoeExperts::Bf16(experts) => {
                let routing = routing
                    .as_ref()
                    .context("missing BF16 Qwen3-MoE routing values")?;
                let routes = routing
                    .iter()
                    .map(|row| self.selected_experts(row))
                    .collect::<Result<Vec<_>>>()?;
                experts.forward(&flat_x, &routes)?
            }
            #[cfg(any(target_os = "macos", feature = "cuda"))]
            MoeExperts::Indexed(experts) => {
                let cpu_routing;
                let (expert_ids, expert_weights) = if let Some(indexed) = indexed_routing.as_ref() {
                    (&indexed.0, &indexed.1)
                } else {
                    let routing = routing
                        .as_ref()
                        .context("missing host Qwen3-MoE routing values")?;
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
                experts.forward(
                    &flat_x,
                    expert_ids,
                    expert_weights,
                    arithmetic.is_some_and(|policy| policy.qwen_metal_rows_at(start_pos).is_some()),
                )?
            }
            MoeExperts::Split(experts) => {
                let routing = routing
                    .as_ref()
                    .context("missing split Qwen3-MoE routing values")?;
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

        output
            .to_dtype(x.dtype())?
            .reshape((b_sz, seq_len, hidden_dim))
            .map_err(Into::into)
    }
}

impl Qwen3MoeBlock {
    fn forward_rows(
        &self,
        x: &Tensor,
        kv_cache: &mut Option<KvCache>,
        seq_len: usize,
        start_pos: usize,
        row_policy: Option<native_cpu::RowPolicy>,
        arithmetic: Option<TextArithmeticPolicy>,
    ) -> Result<Tensor> {
        // Pre-norm attention
        let residual = x;
        let x = self.attn_norm.forward(x)?;
        let x = self
            .attn
            .forward_rows(&x, kv_cache, seq_len, start_pos, row_policy, arithmetic)?;
        // Residual add and FFN pre-norm together.
        let (x, ffn_out) = self.ffn_norm.forward_residual(residual, &x)?;
        let residual = &x;
        let ffn_out = self.moe.forward(&ffn_out, start_pos, arithmetic)?;
        let x = (residual + ffn_out)?;

        Ok(x)
    }
}

fn qtensor_to_qmatmul(qt: Arc<QTensor>) -> Result<QMatMul> {
    // Preserve the stored BF16 CPU projection contract. The packed kernel
    // rounds activations to BF16 before its dot product, as native CPU does.
    // from_arc expands BF16 to F32 and would silently remove that rounding.
    // Metal keeps its existing F32-input projection path.
    if qt.device().is_cpu() && qt.dtype() == GgmlDType::BF16 {
        return Ok(QMatMul::QTensor(qt));
    }
    QMatMul::from_arc(qt).map_err(Into::into)
}

/// Split a packed expert QTensor into per-expert QMatMul instances.
///
/// The packed tensor has shape [num_experts, out_dim, in_dim] in quantized format.
/// Extracts raw bytes via QTensor::data() and creates per-expert QTensors on the target device.
fn split_expert_tensor(
    packed: &QTensor,
    num_experts: usize,
    out_dim: usize,
    in_dim: usize,
    device: &Device,
) -> Result<Vec<QMatMul>> {
    let dtype = packed.dtype();
    let ggml_dtype: GgmlDType = dtype;

    // Calculate bytes per expert from quantization block math
    let elements_per_expert = out_dim * in_dim;
    let block_size = ggml_dtype.block_size();
    anyhow::ensure!(
        elements_per_expert.is_multiple_of(block_size),
        "Expert dimensions ({out_dim}×{in_dim}={elements_per_expert}) not aligned to block size ({block_size})"
    );
    let blocks_per_expert = elements_per_expert / block_size;
    let bytes_per_block = ggml_dtype.type_size();
    let bytes_per_expert = blocks_per_expert * bytes_per_block;

    // packed is on CPU (loaded there by the GGUF loader), so data() works.
    let raw_data = packed.data()?;
    anyhow::ensure!(
        raw_data.len() >= num_experts * bytes_per_expert,
        "Packed expert tensor too small: {} bytes, need {} ({num_experts} experts × {bytes_per_expert})",
        raw_data.len(),
        num_experts * bytes_per_expert,
    );

    // Create per-expert QTensors on the target device (Metal/Cuda/Cpu).
    let mut experts = Vec::with_capacity(num_experts);
    for i in 0..num_experts {
        let start = i * bytes_per_expert;
        let end = start + bytes_per_expert;
        let expert_bytes = &raw_data[start..end];

        let expert_qt = candle_core::quantized::ggml_file::qtensor_from_ggml(
            ggml_dtype,
            expert_bytes,
            vec![out_dim, in_dim],
            device,
        )?;

        experts.push(QMatMul::from_arc(Arc::new(expert_qt))?);
    }

    Ok(experts)
}

impl Qwen3MoeStage {
    /// Construct a Qwen 3 MoE stage from its assigned GGUF tensors.
    ///
    /// Returns an error when required attention, router, or packed expert
    /// tensors are missing or incompatible with the model metadata.
    pub fn from_stage_tensors(
        stage: StageTensors,
        quant_config: Option<Arc<TurboQuantConfig>>,
    ) -> Result<Self> {
        let md = &stage.metadata;
        let device = &stage.device;
        let range = &stage.range;
        let tensors = &stage.tensors;
        #[cfg(feature = "cuda")]
        {
            super::dense_qmatmul_cuda::precompile(device)?;
            super::residual_norm_cuda::precompile(device)?;
        }

        let num_experts = md.expert_count;
        let num_active = md.expert_used_count;
        let expert_ffn_dim = md.expert_ffn_dim;

        anyhow::ensure!(num_experts > 0, "qwen3moe requires expert_count > 0");
        anyhow::ensure!(num_active > 0, "qwen3moe requires expert_used_count > 0");

        // Embedding (head only)
        let embedding = if range.is_head() {
            let embd = tensors
                .get("token_embd.weight")
                .context("missing token_embd.weight for head stage")?;
            Some(embd.as_ref().dequantize(device)?)
        } else {
            None
        };

        // Precompute RoPE cos/sin
        let (rope_cos, rope_sin) =
            precompute_rope_cos_sin(md.head_dim, md.rope_theta, md.context_length, device)?;

        // Transformer blocks
        let mut blocks = Vec::with_capacity(range.layer_count());
        for layer_idx in range.start..range.end {
            let prefix = format!("blk.{layer_idx}");
            let block = build_moe_block(
                tensors,
                &prefix,
                md,
                device,
                &rope_cos,
                &rope_sin,
                num_experts,
                num_active,
                expert_ffn_dim,
                quant_config.clone(),
            )?;
            blocks.push(block);
        }

        // Output norm + lm_head (tail only)
        let mut output_norm = if range.is_tail() {
            let norm_w = tensors
                .get("output_norm.weight")
                .context("missing output_norm.weight")?;
            Some(RmsNorm {
                weight: norm_w.as_ref().dequantize(device)?,
                eps: md.rms_norm_eps,
                native_quant: false,
                native_bf16: false,
            })
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

        let head_native_quant =
            quant_config.is_none() && lm_head.as_ref().is_some_and(native_cpu::supported);
        if let Some(norm) = &mut output_norm {
            norm.native_quant = head_native_quant;
            if let Some(head) = &lm_head {
                norm.native_bf16 =
                    bf16_cpu_math::bf16_norm_projection(head, quant_config.is_none());
            }
        }

        // Infer vocab_size from output weights if not in metadata
        let vocab_size = if md.vocab_size > 0 {
            md.vocab_size
        } else if let Some(emb) = &embedding {
            emb.dim(0)?
        } else {
            151936 // Qwen3 default
        };

        Ok(Self {
            embedding,
            blocks,
            output_norm,
            lm_head,
            head_native_quant,
            is_head: range.is_head(),
            is_tail: range.is_tail(),
            hidden_dim: md.hidden_dim,
            num_layers: range.layer_count(),
            vocab_size,
            device: device.clone(),
        })
    }
}

impl Qwen3MoeStage {
    fn forward_rows(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
        arithmetic: Option<TextArithmeticPolicy>,
    ) -> Result<StageOutput> {
        let row_policy = arithmetic.map(|p| native_cpu::RowPolicy {
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

        #[cfg(feature = "cuda")]
        let fuse_decode_norms = seq_len == 1 && self.device.is_cuda();
        #[cfg(not(feature = "cuda"))]
        let fuse_decode_norms = false;

        let mut tail_normed = None;
        if fuse_decode_norms {
            let mut next_attn_normed = None;
            for (i, block) in self.blocks.iter().enumerate() {
                let normed = if let Some(normed) = next_attn_normed.take() {
                    normed
                } else {
                    block.attn_norm.forward(&hidden_states)?
                };
                let attn_out = with_kv_cache(&mut cache[i], |kv| {
                    block
                        .attn
                        .forward(&normed, kv, seq_len, start_pos, arithmetic)
                })?;
                let (next_hidden_states, normed) =
                    block.ffn_norm.forward_residual(&hidden_states, &attn_out)?;
                hidden_states = next_hidden_states;
                let moe_out = block.moe.forward(&normed, start_pos, arithmetic)?;
                if let Some(next_block) = self.blocks.get(i + 1) {
                    let (next_hidden_states, normed) = next_block
                        .attn_norm
                        .forward_residual(&hidden_states, &moe_out)?;
                    hidden_states = next_hidden_states;
                    next_attn_normed = Some(normed);
                } else if self.is_tail {
                    let (next_hidden_states, normed) = self
                        .output_norm
                        .as_ref()
                        .unwrap()
                        .forward_residual(&hidden_states, &moe_out)?;
                    hidden_states = next_hidden_states;
                    tail_normed = Some(normed);
                } else {
                    hidden_states = (hidden_states + moe_out)?;
                }
            }
        } else {
            for (i, block) in self.blocks.iter().enumerate() {
                hidden_states = with_kv_cache(&mut cache[i], |kv| {
                    block.forward_rows(
                        &hidden_states,
                        kv,
                        seq_len,
                        start_pos,
                        row_policy,
                        arithmetic,
                    )
                })?;
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
            let head = self.lm_head.as_ref().unwrap();
            let logits = if self.head_native_quant {
                native_cpu::forward_rows(
                    head,
                    &last_hidden,
                    arithmetic.map(|_| native_cpu::RowPolicy {
                        start: 0,
                        seq_len: 1,
                        m4_prefix_end: 0,
                    }),
                )?
            } else if self.output_norm.as_ref().unwrap().native_bf16 {
                bf16_cpu_math::project_bf16(head, &last_hidden, true)?
            } else {
                head.forward(&last_hidden)?
            };
            let logits = logits.squeeze(0)?;
            Ok(StageOutput::Logits(logits))
        } else {
            let output = hidden_states.squeeze(0)?;
            Ok(StageOutput::Activations(output))
        }
    }
}

impl ModelArchitecture for Qwen3MoeStage {
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
        #[cfg(target_os = "macos")]
        if std::env::var_os("DINF_DISABLE_INDEXED_MOE_MV").is_none()
            && !self.blocks.is_empty()
            && self.blocks.iter().all(|b| {
                b.attn.quant_config.is_none()
                    && b.attn.native_qkv.is_some()
                    && native_qkv_metal::supports(&b.attn.q_proj, &b.attn.k_proj, &b.attn.v_proj)
            })
        {
            return Some(TextArithmeticPolicy::qwen_metal_qkv(context));
        }
        if context.prompt_end() <= 4096
            && self.device.is_cpu()
            && !self.blocks.is_empty()
            && self.blocks.iter().all(|b| {
                b.attn.quant_config.is_none()
                    && b.attn.num_heads == 32
                    && b.attn.num_kv_heads == 4
                    && b.attn.head_dim == 128
                    && [
                        &b.attn.q_proj,
                        &b.attn.k_proj,
                        &b.attn.v_proj,
                        &b.attn.o_proj,
                    ]
                    .iter()
                    .all(|p| bf16_cpu_math::bf16_norm_projection(p, true))
            })
            && self
                .lm_head
                .as_ref()
                .is_none_or(|h| bf16_cpu_math::bf16_norm_projection(h, true))
        {
            return Some(TextArithmeticPolicy::qwen_cpu_bf16(context));
        }
        let supported = self.device.is_cpu()
            && !self.blocks.is_empty()
            && self.blocks.iter().all(|b| {
                b.attn.quant_config.is_none()
                    && native_cpu::supported_k(&b.attn.q_proj)
                    && native_cpu::supported(&b.attn.k_proj)
                    && native_cpu::supported_k(&b.attn.v_proj)
                    && native_cpu::supported_k(&b.attn.o_proj)
            })
            && self
                .lm_head
                .as_ref()
                .is_none_or(|h| self.head_native_quant && native_cpu::supported_k(h));
        supported.then(|| TextArithmeticPolicy::qwen_cpu_k_native_attention(context))
    }

    #[allow(clippy::too_many_arguments)]
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
        "qwen3moe"
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

fn native_k_router_admitted(
    native_cache_quant: bool,
    stored_router: GgmlDType,
    expert_types: [GgmlDType; 3],
    gate: &Tensor,
    active: usize,
) -> bool {
    native_cache_quant
        && stored_router == GgmlDType::F32
        && active == 8
        && native_router::supports_weight(gate)
        && expert_types
            .into_iter()
            .all(|dtype| matches!(dtype, GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K))
}

#[allow(clippy::too_many_arguments)]
fn build_moe_block(
    tensors: &std::collections::HashMap<String, Arc<QTensor>>,
    prefix: &str,
    md: &crate::gguf_loader::GgufMetadata,
    device: &Device,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
    num_experts: usize,
    num_active: usize,
    expert_ffn_dim: usize,
    quant_config: Option<Arc<TurboQuantConfig>>,
) -> Result<Qwen3MoeBlock> {
    let get = |name: &str| -> Result<Arc<QTensor>> {
        tensors
            .get(&format!("{prefix}.{name}"))
            .cloned()
            .with_context(|| format!("missing tensor: {prefix}.{name}"))
    };

    let mut attn_norm = RmsNorm {
        weight: get("attn_norm.weight")?.as_ref().dequantize(device)?,
        eps: md.rms_norm_eps,
        native_quant: false,
        native_bf16: false,
    };

    #[cfg(target_os = "macos")]
    let native_qkv = native_qkv_metal::NativeQkv::new(
        get("attn_q.weight")?.as_ref(),
        get("attn_k.weight")?.as_ref(),
        get("attn_v.weight")?.as_ref(),
        quant_config.is_none(),
        md.num_attention_heads,
        md.num_kv_heads,
        md.head_dim,
    )?;
    #[cfg(target_os = "macos")]
    let native_o = native_o_metal::NativeO::new(
        get("attn_output.weight")?.as_ref(),
        native_qkv.is_some(),
        quant_config.is_none(),
    )?;
    // Prompt chunks use the tensor-core kernels. Share the native decode
    // kernels' packed weights when present; otherwise copy from the GGUF.
    #[cfg(target_os = "macos")]
    let prompt_projections = if device.is_metal()
        && std::env::var_os("DINF_DISABLE_QWEN3_TENSOR_PROMPT_PROJECTIONS").is_none()
    {
        use super::tensor_qmatmul_metal::MetalTensorQMatMul;
        let names = [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
        ];
        let mut built = Vec::with_capacity(4);
        for (index, name) in names.iter().enumerate() {
            let qt = get(name)?;
            let (out_dim, in_dim) = qt.shape().dims2()?;
            let shared = match (index, &native_qkv, &native_o) {
                (0..=2, Some(native), _) => Some(native.packed()[index]),
                (3, _, Some(native)) => Some(native.packed()),
                _ => None,
            };
            let projection = match shared {
                Some(raw) => {
                    MetalTensorQMatMul::try_from_raw(raw, qt.dtype(), out_dim, in_dim, device)?
                }
                None => MetalTensorQMatMul::try_from_qtensor(Arc::clone(&qt), device)?,
            };
            match projection {
                Some(projection) => built.push(projection),
                None => break,
            }
        }
        built
            .try_into()
            .ok()
            .map(|array: [MetalTensorQMatMul; 4]| Box::new(array))
    } else {
        None
    };
    let mut attn = Attention {
        #[cfg(target_os = "macos")]
        prompt_projections,
        #[cfg(target_os = "macos")]
        native_qkv,
        #[cfg(target_os = "macos")]
        native_o,
        q_proj: qtensor_to_qmatmul(get("attn_q.weight")?)?,
        k_proj: qtensor_to_qmatmul(get("attn_k.weight")?)?,
        v_proj: qtensor_to_qmatmul(get("attn_v.weight")?)?,
        o_proj: qtensor_to_qmatmul(get("attn_output.weight")?)?,
        q_norm: RmsNorm {
            weight: get("attn_q_norm.weight")?.as_ref().dequantize(device)?,
            eps: md.rms_norm_eps,
            native_quant: false,
            native_bf16: false,
        },
        k_norm: RmsNorm {
            weight: get("attn_k_norm.weight")?.as_ref().dequantize(device)?,
            eps: md.rms_norm_eps,
            native_quant: false,
            native_bf16: false,
        },
        num_heads: md.num_attention_heads,
        num_kv_heads: md.num_kv_heads,
        head_dim: md.head_dim,
        rope_theta: md.rope_theta,
        #[cfg(target_os = "macos")]
        native_bf16_rotary: native_rotary_metal::supports_weights(
            get("attn_q.weight")?.as_ref(),
            get("attn_k.weight")?.as_ref(),
        ),
        rope_cos: rope_cos.clone(),
        rope_sin: rope_sin.clone(),
        quant_config,
    };

    let native_cache = attn.quant_config.is_none();
    attn_norm.native_bf16 = [&attn.q_proj, &attn.k_proj, &attn.v_proj]
        .iter()
        .all(|p| bf16_cpu_math::bf16_norm_projection(p, native_cache));
    attn.q_norm.native_bf16 = bf16_cpu_math::bf16_norm_projection(&attn.q_proj, native_cache);
    attn.k_norm.native_bf16 = bf16_cpu_math::bf16_norm_projection(&attn.k_proj, native_cache);
    let native_quant = attn.quant_config.is_none()
        && native_cpu::supported(&attn.q_proj)
        && native_cpu::supported(&attn.k_proj)
        && native_cpu::supported(&attn.v_proj);
    attn_norm.native_quant = native_quant;
    attn.q_norm.native_quant = native_quant;
    attn.k_norm.native_quant = native_quant;

    let mut ffn_norm = RmsNorm {
        weight: get("ffn_norm.weight")?.as_ref().dequantize(device)?,
        eps: md.rms_norm_eps,
        native_quant: false,
        native_bf16: false,
    };

    // Router weights (F32 tensor, dequantize for matmul)
    let stored_gate = get("ffn_gate_inp.weight")?;
    #[cfg(target_os = "macos")]
    let native_metal_router = native_router_metal::NativeRouter::new(
        stored_gate.as_ref(),
        attn.native_qkv.is_some(),
        attn.quant_config.is_none(),
    )?;
    let gate = stored_gate.as_ref().dequantize(device)?;

    let gate_exps = get("ffn_gate_exps.weight")?;
    let up_exps = get("ffn_up_exps.weight")?;
    let down_exps = get("ffn_down_exps.weight")?;
    ffn_norm.native_bf16 = [&gate_exps, &up_exps, &down_exps]
        .iter()
        .all(|w| bf16_cpu_math::bf16_norm_weight(w, native_cache));
    ffn_norm.native_quant = native_quant
        && [&gate_exps, &up_exps, &down_exps].iter().all(|w| {
            matches!(
                w.dtype(),
                GgmlDType::Q8_0 | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K
            )
        });

    let split_experts = || -> Result<MoeExperts> {
        let gate_mats = split_expert_tensor(
            gate_exps.as_ref(),
            num_experts,
            expert_ffn_dim,
            md.hidden_dim,
            device,
        )
        .with_context(|| format!("{prefix}.ffn_gate_exps split"))?;
        let up_mats = split_expert_tensor(
            up_exps.as_ref(),
            num_experts,
            expert_ffn_dim,
            md.hidden_dim,
            device,
        )
        .with_context(|| format!("{prefix}.ffn_up_exps split"))?;
        let down_mats = split_expert_tensor(
            down_exps.as_ref(),
            num_experts,
            md.hidden_dim,
            expert_ffn_dim,
            device,
        )
        .with_context(|| format!("{prefix}.ffn_down_exps split"))?;

        Ok(MoeExperts::Split(
            gate_mats
                .into_iter()
                .zip(up_mats)
                .zip(down_mats)
                .map(|((gate_proj, up_proj), down_proj)| MoeExpert {
                    native_quant: native_quant
                        && native_cpu::supported(&gate_proj)
                        && native_cpu::supported(&up_proj)
                        && native_cpu::supported(&down_proj),
                    gate_proj,
                    up_proj,
                    down_proj,
                })
                .collect(),
        ))
    };

    let bounded_bf16 = (device.is_cpu() || device.is_metal())
        && [&gate_exps, &up_exps, &down_exps]
            .iter()
            .any(|tensor| tensor.dtype() == GgmlDType::BF16);
    let experts = if bounded_bf16 {
        MoeExperts::Bf16(Box::new(
            Bf16Experts::new(
                Arc::clone(&gate_exps),
                Arc::clone(&up_exps),
                Arc::clone(&down_exps),
                num_experts,
                md.hidden_dim,
                expert_ffn_dim,
                device,
            )?
            .with_native_cache(native_cache),
        ))
    } else {
        #[cfg(any(target_os = "macos", feature = "cuda"))]
        let experts = {
            let use_indexed = indexed_moe_available(device, gate_exps.dtype())
                && indexed_moe_available(device, up_exps.dtype())
                && indexed_moe_available(device, down_exps.dtype());
            if use_indexed {
                let indexed = IndexedMoeExperts {
                    #[cfg(target_os = "macos")]
                    native_vector: std::env::var_os("DINF_DISABLE_INDEXED_MOE_MV").is_none()
                        && attn.native_qkv.is_some()
                        && attn.quant_config.is_none()
                        && native_metal_expert_types(
                            gate_exps.as_ref(),
                            up_exps.as_ref(),
                            down_exps.as_ref(),
                            device,
                        ),
                    gate_proj: IndexedMoeMatMul::from_qtensor(
                        gate_exps.as_ref(),
                        num_experts,
                        expert_ffn_dim,
                        md.hidden_dim,
                        device,
                    )?,
                    up_proj: IndexedMoeMatMul::from_qtensor(
                        up_exps.as_ref(),
                        num_experts,
                        expert_ffn_dim,
                        md.hidden_dim,
                        device,
                    )?,
                    down_proj: IndexedMoeMatMul::from_qtensor(
                        down_exps.as_ref(),
                        num_experts,
                        md.hidden_dim,
                        expert_ffn_dim,
                        device,
                    )?,
                };
                #[cfg(target_os = "macos")]
                device.synchronize()?;
                MoeExperts::Indexed(Box::new(indexed))
            } else {
                split_experts()?
            }
        };
        #[cfg(not(any(target_os = "macos", feature = "cuda")))]
        let experts = split_experts()?;
        experts
    };

    let native_k_argsort = native_k_router_admitted(
        native_quant,
        stored_gate.dtype(),
        [gate_exps.dtype(), up_exps.dtype(), down_exps.dtype()],
        &gate,
        num_active,
    );
    let native_router = native_k_argsort
        || (ffn_norm.native_bf16
            && stored_gate.dtype() == GgmlDType::F32
            && num_experts == 128
            && num_active == 8
            && native_router::supports_weight(&gate));
    let moe = MoeLayer {
        #[cfg(target_os = "macos")]
        native_metal_router,
        gate,
        experts,
        num_active,
        native_router,
        native_k_argsort,
    };

    Ok(Qwen3MoeBlock {
        attn_norm,
        attn,
        ffn_norm,
        moe,
    })
}

fn precompute_rope_cos_sin(
    head_dim: usize,
    theta: f64,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32))
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
    candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin).map_err(Into::into)
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
    let mask = Tensor::from_vec(mask, (1, 1, seq_len, total_seq_len), device)?;
    Ok(mask)
}

#[cfg(test)]
mod native_router_admission_tests {
    use super::*;
    #[test]
    fn native_router_admission_keeps_other_precision_and_cache_paths() -> Result<()> {
        let gate = Tensor::zeros((128, 2048), DType::F32, &Device::Cpu)?;
        let k = [GgmlDType::Q4K, GgmlDType::Q4K, GgmlDType::Q6K];
        assert_eq!(
            native_k_router_admitted(true, GgmlDType::F32, k, &gate, 8),
            cfg!(all(target_os = "macos", target_arch = "aarch64"))
        );
        assert!(!native_k_router_admitted(
            false,
            GgmlDType::F32,
            k,
            &gate,
            8
        ));
        assert!(!native_k_router_admitted(true, GgmlDType::F16, k, &gate, 8));
        assert!(!native_k_router_admitted(true, GgmlDType::F32, k, &gate, 6));
        for dtype in [GgmlDType::Q8_0, GgmlDType::BF16, GgmlDType::F32] {
            assert!(!native_k_router_admitted(
                true,
                GgmlDType::F32,
                [dtype; 3],
                &gate,
                8
            ));
        }
        assert!(!native_k_router_admitted(
            true,
            GgmlDType::F32,
            k,
            &gate.narrow(0, 0, 64)?,
            8
        ));
        Ok(())
    }
}

#[cfg(test)]
mod native_argsort_scope_tests {
    use super::*;

    // Deterministic synthetic native graph fixture: cutoff_spread_128.
    // Seven high scores at multiples of17, a middle tie group at i%3==1,
    // and a lower group. Constants are the three measured native softmax
    // results for synthetic logits5,0,-5; no model data enters this test.
    fn probabilities() -> Vec<f32> {
        (0..128)
            .map(|i| {
                f32::from_bits(if i % 17 == 0 && i / 17 < 7 {
                    0x3e0c_a958
                } else if i % 3 == 1 {
                    0x3a72_a108
                } else {
                    0x36d1_41d7
                })
            })
            .collect()
    }

    fn layer(native_router: bool, native_k_argsort: bool) -> Result<MoeLayer> {
        let tensor = Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?;
        let experts = (0..128)
            .map(|_| MoeExpert {
                gate_proj: QMatMul::Tensor(tensor.clone()),
                up_proj: QMatMul::Tensor(tensor.clone()),
                down_proj: QMatMul::Tensor(tensor.clone()),
                native_quant: false,
            })
            .collect();
        Ok(MoeLayer {
            #[cfg(target_os = "macos")]
            native_metal_router: None,
            gate: tensor,
            experts: MoeExperts::Split(experts),
            num_active: 8,
            native_router,
            native_k_argsort,
        })
    }

    #[test]
    fn ordinary_sort_keeps_the_existing_stable_order() -> Result<()> {
        let scores = probabilities();
        use sha2::Digest;
        let bytes: Vec<_> = scores
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert_eq!(
            format!("{:x}", sha2::Sha256::digest(&bytes)),
            "1a999af938e8a3d026064af90fb14d0b83e5f3c0286df369c7e3bdb85e1ec0e9"
        );
        let selected = layer(false, false)?.selected_experts(&scores)?;
        assert_eq!(
            selected.iter().map(|x| x.0).collect::<Vec<_>>(),
            [0, 17, 34, 51, 68, 85, 102, 1]
        );
        let mut original: Vec<_> = scores.into_iter().enumerate().collect();
        original.sort_by(|a, b| b.1.total_cmp(&a.1));
        original.truncate(8);
        let sum = original.iter().map(|x| x.1).sum::<f32>();
        for x in &mut original {
            x.1 /= sum.max(1e-20);
        }
        assert_eq!(
            selected
                .iter()
                .map(|x| (x.0, x.1.to_bits()))
                .collect::<Vec<_>>(),
            original
                .iter()
                .map(|x| (x.0, x.1.to_bits()))
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn k_sort_matches_native_tie_without_changing_full_router_sort() -> Result<()> {
        let scores = probabilities();
        let full = layer(true, false)?.selected_experts(&scores)?;
        assert_eq!(
            full.iter().map(|x| x.0).collect::<Vec<_>>(),
            [0, 17, 34, 51, 68, 85, 102, 1]
        );
        let k = layer(true, true)?.selected_experts(&scores)?;
        assert_eq!(
            k.iter().map(|x| x.0).collect::<Vec<_>>(),
            [17, 102, 85, 68, 0, 51, 34, 64]
        );
        assert_eq!(
            k.iter().map(|x| x.1.to_bits()).collect::<Vec<_>>(),
            [
                0x3e122522, 0x3e122522, 0x3e122522, 0x3e122522, 0x3e122522, 0x3e122522, 0x3e122522,
                0x3a7c1672
            ]
        );
        Ok(())
    }
}
