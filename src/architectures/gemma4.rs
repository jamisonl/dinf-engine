use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use crate::qmatmul::QMatMul;
use anyhow::{Context, Result};
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, IndexOp, Module, Tensor};

use super::{
    AttentionSpan, KvCache, LayerCache, ModelArchitecture, StageInput, StageOutput, with_kv_cache,
};
use crate::gguf_loader::StageTensors;
use crate::turboquant::TurboQuantConfig;

static DECODE_BATCH_LOGGED: OnceLock<()> = OnceLock::new();
#[cfg(target_os = "macos")]
static METAL4_QMM_LOGGED: OnceLock<()> = OnceLock::new();
static ATTN_NORM_NS: AtomicU64 = AtomicU64::new(0);
static Q_PROJ_NS: AtomicU64 = AtomicU64::new(0);
static K_PROJ_NS: AtomicU64 = AtomicU64::new(0);
static V_PROJ_NS: AtomicU64 = AtomicU64::new(0);
static QKV_LAYOUT_NS: AtomicU64 = AtomicU64::new(0);
static Q_NORM_NS: AtomicU64 = AtomicU64::new(0);
static K_NORM_NS: AtomicU64 = AtomicU64::new(0);
static V_NORM_NS: AtomicU64 = AtomicU64::new(0);
static ROPE_NS: AtomicU64 = AtomicU64::new(0);
static KV_UPDATE_NS: AtomicU64 = AtomicU64::new(0);
static FULL_SDPA_NS: AtomicU64 = AtomicU64::new(0);
static SLIDING_SDPA_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_OUTPUT_LAYOUT_NS: AtomicU64 = AtomicU64::new(0);
static O_PROJ_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_POST_NORM_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_RESIDUAL_NS: AtomicU64 = AtomicU64::new(0);
static FFN_NORM_NS: AtomicU64 = AtomicU64::new(0);
static FFN_UP_NS: AtomicU64 = AtomicU64::new(0);
static FFN_GATE_NS: AtomicU64 = AtomicU64::new(0);
static FFN_ACTIVATION_NS: AtomicU64 = AtomicU64::new(0);
static FFN_DOWN_NS: AtomicU64 = AtomicU64::new(0);
static FFN_POST_NORM_NS: AtomicU64 = AtomicU64::new(0);
static FFN_RESIDUAL_NS: AtomicU64 = AtomicU64::new(0);
static OUTPUT_SCALE_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_CALLS: AtomicU64 = AtomicU64::new(0);
static PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();

fn layer_profile_enabled() -> bool {
    *PROFILE_ENABLED.get_or_init(|| std::env::var_os("DINF_PROFILE_LAYERS").is_some())
}

fn finish_profile_region(device: &Device, started: Instant, counter: &AtomicU64) -> Result<()> {
    device.synchronize()?;
    counter.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(())
}

/// Reset the process-local Gemma 4 layer profiling counters.
pub fn reset_layer_profile() {
    for counter in [
        &ATTN_NORM_NS,
        &Q_PROJ_NS,
        &K_PROJ_NS,
        &V_PROJ_NS,
        &QKV_LAYOUT_NS,
        &Q_NORM_NS,
        &K_NORM_NS,
        &V_NORM_NS,
        &ROPE_NS,
        &KV_UPDATE_NS,
        &FULL_SDPA_NS,
        &SLIDING_SDPA_NS,
        &ATTN_OUTPUT_LAYOUT_NS,
        &O_PROJ_NS,
        &ATTN_POST_NORM_NS,
        &ATTN_RESIDUAL_NS,
        &FFN_NORM_NS,
        &FFN_UP_NS,
        &FFN_GATE_NS,
        &FFN_ACTIVATION_NS,
        &FFN_DOWN_NS,
        &FFN_POST_NORM_NS,
        &FFN_RESIDUAL_NS,
        &OUTPUT_SCALE_NS,
        &PROFILE_CALLS,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

/// Print the process-local Gemma 4 layer profiling counters to stderr.
pub fn report_layer_profile() {
    let calls = PROFILE_CALLS.load(Ordering::Relaxed).max(1) as f64;
    let ms_per_call =
        |counter: &AtomicU64| counter.load(Ordering::Relaxed) as f64 / calls / 1_000_000.0;
    eprintln!(
        "DINF_GEMMA4_ATTN_PROFILE calls={} per_layer_ms={{attn_norm:{:.4},q_proj:{:.4},k_proj:{:.4},v_proj:{:.4},qkv_layout:{:.4},q_norm:{:.4},k_norm:{:.4},v_norm:{:.4},rope:{:.4},kv_update:{:.4},full_sdpa:{:.4},sliding_sdpa:{:.4},output_layout:{:.4},o_proj:{:.4},post_norm:{:.4},residual:{:.4}}}",
        PROFILE_CALLS.load(Ordering::Relaxed),
        ms_per_call(&ATTN_NORM_NS),
        ms_per_call(&Q_PROJ_NS),
        ms_per_call(&K_PROJ_NS),
        ms_per_call(&V_PROJ_NS),
        ms_per_call(&QKV_LAYOUT_NS),
        ms_per_call(&Q_NORM_NS),
        ms_per_call(&K_NORM_NS),
        ms_per_call(&V_NORM_NS),
        ms_per_call(&ROPE_NS),
        ms_per_call(&KV_UPDATE_NS),
        ms_per_call(&FULL_SDPA_NS),
        ms_per_call(&SLIDING_SDPA_NS),
        ms_per_call(&ATTN_OUTPUT_LAYOUT_NS),
        ms_per_call(&O_PROJ_NS),
        ms_per_call(&ATTN_POST_NORM_NS),
        ms_per_call(&ATTN_RESIDUAL_NS),
    );
    eprintln!(
        "DINF_GEMMA4_FFN_PROFILE calls={} per_layer_ms={{ffn_norm:{:.4},up:{:.4},gate:{:.4},activation:{:.4},down:{:.4},post_norm:{:.4},residual:{:.4},output_scale:{:.4}}}",
        PROFILE_CALLS.load(Ordering::Relaxed),
        ms_per_call(&FFN_NORM_NS),
        ms_per_call(&FFN_UP_NS),
        ms_per_call(&FFN_GATE_NS),
        ms_per_call(&FFN_ACTIVATION_NS),
        ms_per_call(&FFN_DOWN_NS),
        ms_per_call(&FFN_POST_NORM_NS),
        ms_per_call(&FFN_RESIDUAL_NS),
        ms_per_call(&OUTPUT_SCALE_NS),
    );
}

/// Gemma 4 text architecture.
///
/// Covers the local text-only Gemma 4 GGUF layout:
/// - interleaved sliding-window and full attention layers
/// - per-head Q/K RMSNorm
/// - unweighted V RMSNorm
/// - post-attention and post-FFN RMSNorms
/// - GELU-gated FFN
/// - per-layer output scale after the full block residual
pub struct Gemma4Stage {
    embedding: Option<Tensor>,
    blocks: Vec<TransformerBlock>,
    output_norm: Option<RmsNorm>,
    lm_head: Option<QMatMul>,
    is_head: bool,
    is_tail: bool,
    hidden_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    final_logit_softcapping: f64,
    device: Device,
}

struct TransformerBlock {
    attn_norm: RmsNorm,
    attn: Attention,
    attn_post_norm: RmsNorm,
    ffn_norm: RmsNorm,
    ffn: Mlp,
    /// Routed experts beside the shared `ffn` (Gemma 4 26B-A4B).
    moe: Option<Box<GemmaMoe>>,
    ffn_post_norm: RmsNorm,
    layer_output_scale: Option<Tensor>,
}

/// Gemma 4 routed experts, following llama.cpp's `gemma4.cpp`: the router
/// reads the unweighted RMSNorm of the attention output scaled by
/// `gate_inp.scale / sqrt(hidden)`, softmax top-k weights are renormalized,
/// each expert computes `down(gelu(gate) * up) * down_exps.scale[expert]`,
/// and the weighted sum passes through `post_ffw_norm_2`.
struct GemmaMoe {
    /// `[experts, hidden]` router projection with `gate_inp.scale /
    /// sqrt(hidden)` folded into its columns; it reads the unweighted RMSNorm.
    router: Tensor,
    pre_norm: RmsNorm,
    shared_post_norm: RmsNorm,
    experts_post_norm: RmsNorm,
    experts: GemmaExperts,
    /// Per-expert output scale, `[experts]`.
    down_scale: Tensor,
    num_active: usize,
    rms_norm_eps: f64,
}

#[cfg(any(target_os = "macos", feature = "cuda"))]
struct GemmaExperts {
    /// Packed `[experts, 2 * expert_ffn, hidden]`, gate rows first.
    gate_up: super::indexed_moe::IndexedMoeMatMul,
    down: super::indexed_moe::IndexedMoeMatMul,
}

#[cfg(not(any(target_os = "macos", feature = "cuda")))]
struct GemmaExperts;

impl GemmaMoe {
    /// Token count from which the router matmul uses a contiguous transpose.
    const ROUTER_TRANSPOSE_MIN_TOKENS: usize = 32;
    /// Route rows per expert dispatch, as in Qwen3-MoE.
    #[cfg(any(target_os = "macos", feature = "cuda"))]
    const MAX_ROUTE_ROWS_PER_DISPATCH: usize = 4096;
    /// Token batches at or above this share Metal route tables across the
    /// gate/up and down projections.
    #[cfg(target_os = "macos")]
    const ROUTE_PLAN_MIN_TOKENS: usize = 64;

    /// Routed experts before `post_ffw_norm_2`. `expert_input` is the
    /// `pre_ffw_norm_2` output and `router_input` the unweighted RMSNorm of
    /// the attention residual.
    #[cfg(any(target_os = "macos", feature = "cuda"))]
    fn forward(&self, expert_input: &Tensor, router_input: &Tensor) -> Result<Tensor> {
        use super::indexed_moe::{normalized_route_weights, weighted_route_sum};

        let (b_sz, seq_len, hidden) = expert_input.dims3()?;
        let tokens = b_sz * seq_len;
        let logits = router_input
            .reshape((tokens, hidden))?
            .matmul(&if tokens >= Self::ROUTER_TRANSPOSE_MIN_TOKENS {
                // A contiguous transpose selects a faster GEMM for prompts;
                // single tokens keep the transposed view's GEMV.
                self.router.t()?.contiguous()?
            } else {
                self.router.t()?
            })?
            .contiguous()?;
        let expert_input = expert_input.reshape((tokens, hidden))?;

        // Metal: one fused softmax/top-k route that emits validated u32 IDs
        // (no sanitize pass) shared by both projections, and expert scales
        // applied in the route sum. Its weights are f32 bits in a u32 tensor.
        #[cfg(target_os = "macos")]
        let fused = if logits.device().is_metal()
            && std::env::var_os("DINF_DISABLE_FUSED_SOFTMAX_ROUTING").is_none()
        {
            Some(super::fused_route_metal::fused_softmax_topk_route_u32(
                &logits,
                self.num_active,
                true,
                1.0,
            )?)
        } else {
            None
        };
        #[cfg(not(target_os = "macos"))]
        let fused: Option<(Tensor, Tensor)> = None;
        let sanitized = fused.is_some();
        let (ids, weights) = match fused {
            Some(routing) => routing,
            None => {
                let probabilities = candle_nn::ops::softmax_last_dim(&logits)?;
                let ids = probabilities
                    .arg_sort_last_dim(false)?
                    .narrow(candle_core::D::Minus1, 0, self.num_active)?
                    .contiguous()?;
                let weights = normalized_route_weights(&probabilities, &ids)?;
                // Fold each selected expert's output scale into its weight.
                let expert_scales = self
                    .down_scale
                    .index_select(&ids.flatten_all()?, 0)?
                    .reshape((tokens, self.num_active))?;
                (ids, (weights * expert_scales)?)
            }
        };

        let chunk = (Self::MAX_ROUTE_ROWS_PER_DISPATCH / self.num_active).max(1);
        let mut outputs = Vec::with_capacity(tokens.div_ceil(chunk));
        for start in (0..tokens).step_by(chunk) {
            let len = (tokens - start).min(chunk);
            let input = expert_input
                .narrow(0, start, len)?
                .reshape((len, 1, hidden))?;
            let chunk_ids = ids.narrow(0, start, len)?.contiguous()?;
            #[cfg(target_os = "macos")]
            let plan = if len >= Self::ROUTE_PLAN_MIN_TOKENS {
                self.experts.gate_up.metal_route_plan(&chunk_ids)?
            } else {
                None
            };
            #[cfg(target_os = "macos")]
            let routed = if let Some(plan) = plan.as_ref() {
                let activated = match self
                    .experts
                    .gate_up
                    .forward_geglu_with_metal_route_plan(&input, plan)?
                {
                    Some(activated) => activated,
                    None => geglu_packed(
                        &self
                            .experts
                            .gate_up
                            .forward_with_metal_route_plan(&input, plan)?,
                    )?,
                };
                Some(
                    self.experts
                        .down
                        .forward_with_metal_route_plan(&activated, plan)?,
                )
            } else {
                None
            };
            #[cfg(not(target_os = "macos"))]
            let routed: Option<Tensor> = None;
            let routed = match routed {
                Some(routed) => routed,
                None => {
                    let project = |projection: &super::indexed_moe::IndexedMoeMatMul,
                                   x: &Tensor|
                     -> Result<Tensor> {
                        #[cfg(target_os = "macos")]
                        if sanitized {
                            return projection.forward_metal_sanitized(x, &chunk_ids, false);
                        }
                        projection.forward(x, &chunk_ids)
                    };
                    let gate_up = project(&self.experts.gate_up, &input)?;
                    project(&self.experts.down, &geglu_packed(&gate_up)?)?
                }
            };
            let chunk_weights = weights.narrow(0, start, len)?.contiguous()?;
            #[cfg(target_os = "macos")]
            if sanitized {
                outputs.push(super::indexed_moe_metal::fused_expert_scaled_route_sum(
                    &routed,
                    &chunk_weights,
                    &chunk_ids,
                    &self.down_scale,
                )?);
                continue;
            }
            outputs.push(weighted_route_sum(&routed, &chunk_weights)?);
        }
        let refs = outputs.iter().collect::<Vec<_>>();
        Tensor::cat(&refs, 0)?
            .reshape((b_sz, seq_len, hidden))
            .map_err(Into::into)
    }

    #[cfg(not(any(target_os = "macos", feature = "cuda")))]
    fn forward(&self, _expert_input: &Tensor, _router_input: &Tensor) -> Result<Tensor> {
        anyhow::bail!("Gemma 4 mixture-of-experts layers require Metal or CUDA")
    }
}

/// `gelu(gate) * up` over packed `[..., gate | up]` rows.
fn geglu_packed(gate_up: &Tensor) -> Result<Tensor> {
    #[cfg(target_os = "macos")]
    if gate_up.device().is_metal()
        && gate_up.dtype() == DType::F32
        && gate_up.is_contiguous()
        && std::env::var_os("DINF_DISABLE_FUSED_GELU_MUL").is_none()
    {
        return super::indexed_moe_metal::fused_geglu(gate_up).map_err(Into::into);
    }
    let half = gate_up.dim(candle_core::D::Minus1)? / 2;
    let gate = gate_up
        .narrow(candle_core::D::Minus1, 0, half)?
        .contiguous()?;
    let up = gate_up
        .narrow(candle_core::D::Minus1, half, half)?
        .contiguous()?;
    gelu_mul(&gate, &up)
}

struct Attention {
    q_proj: GemmaQMatMul,
    k_proj: GemmaQMatMul,
    v_proj: Option<GemmaQMatMul>,
    o_proj: GemmaQMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    rope_theta: f64,
    rope_freqs: Option<Tensor>,
    rope_tables: Arc<RopeTables>,
    sliding_window: Option<usize>,
    scale: f64,
    rms_norm_eps: f64,
    quant_config: Option<Arc<TurboQuantConfig>>,
}

/// Rotary cos/sin rows for the positions of the current step. Every sliding
/// layer shares one rotary configuration and every global layer the other, so
/// the first layer of each kind builds the rows and the rest reuse them.
#[derive(Default)]
struct RopeTables {
    entries: Mutex<[Option<RopeTableEntry>; 2]>,
}

struct RopeTableEntry {
    start_pos: usize,
    seq_len: usize,
    cos: Tensor,
    sin: Tensor,
}

impl RopeTables {
    fn get(
        &self,
        sliding: bool,
        start_pos: usize,
        seq_len: usize,
        device: &Device,
        build: impl FnOnce() -> Result<(Tensor, Tensor)>,
    ) -> Result<(Tensor, Tensor)> {
        let mut entries = self.entries.lock().unwrap();
        let slot = &mut entries[usize::from(sliding)];
        if let Some(entry) = slot.as_ref()
            && entry.start_pos == start_pos
            && entry.seq_len == seq_len
            && entry.cos.device().same_device(device)
        {
            return Ok((entry.cos.clone(), entry.sin.clone()));
        }
        let (cos, sin) = build()?;
        *slot = Some(RopeTableEntry {
            start_pos,
            seq_len,
            cos: cos.clone(),
            sin: sin.clone(),
        });
        Ok((cos, sin))
    }
}

struct Mlp {
    gate_proj: GemmaQMatMul,
    up_proj: GemmaQMatMul,
    down_proj: GemmaQMatMul,
}

enum GemmaQMatMul {
    Candle(QMatMul),
    #[cfg(target_os = "macos")]
    MetalTensor(super::tensor_qmatmul_metal::MetalTensorQMatMul),
}

impl GemmaQMatMul {
    fn from_qtensor(qt: Arc<QTensor>, _device: &Device) -> Result<Self> {
        #[cfg(target_os = "macos")]
        if std::env::var_os("DINF_DISABLE_METAL4_QMM").is_none()
            && let Some(matmul) = super::tensor_qmatmul_metal::MetalTensorQMatMul::try_from_qtensor(
                Arc::clone(&qt),
                _device,
            )?
        {
            METAL4_QMM_LOGGED.get_or_init(|| {
                tracing::info!("Metal 4 tensor quantized matmul enabled for Gemma 4 prefill");
            });
            return Ok(Self::MetalTensor(matmul));
        }

        Ok(Self::Candle(qtensor_to_qmatmul(qt)?))
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Candle(matmul) => matmul.forward(x).map_err(Into::into),
            #[cfg(target_os = "macos")]
            Self::MetalTensor(matmul) => matmul.forward(x).map_err(Into::into),
        }
    }

    #[cfg(feature = "cuda")]
    fn candle(&self) -> Option<&QMatMul> {
        match self {
            Self::Candle(matmul) => Some(matmul),
            #[cfg(target_os = "macos")]
            Self::MetalTensor(_) => None,
        }
    }
}

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        #[cfg(feature = "cuda")]
        if x.device().is_cuda() {
            return candle_nn::ops::rms_norm(&x_f32, &self.weight, self.eps as f32)?
                .to_dtype(dtype)
                .map_err(Into::into);
        }
        #[cfg(target_os = "macos")]
        if let Some(result) =
            super::fused_norm_metal::try_fused_rms_norm(&x_f32, &self.weight, self.eps as f32)?
        {
            return result.to_dtype(dtype).map_err(Into::into);
        }
        let variance = (&x_f32 * &x_f32)?.mean_keepdim(candle_core::D::Minus1)?;
        #[cfg(target_os = "macos")]
        if let Some(result) = super::fused_norm_metal::try_fused_rms_norm_epilogue(
            &x_f32,
            &variance,
            &self.weight,
            self.eps as f32,
        )? {
            return result.to_dtype(dtype).map_err(Into::into);
        }
        let x_normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        Ok(x_normed.to_dtype(dtype)?.broadcast_mul(&self.weight)?)
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

fn rms_norm_no_weight(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    #[cfg(target_os = "macos")]
    if let Some(result) = super::fused_norm_metal::try_fused_rms_norm_no_weight(&x_f32, eps as f32)?
    {
        return result.to_dtype(dtype).map_err(Into::into);
    }
    let variance = (&x_f32 * &x_f32)?.mean_keepdim(candle_core::D::Minus1)?;
    let x_normed = x_f32.broadcast_div(&(variance + eps)?.sqrt()?)?;
    Ok(x_normed.to_dtype(dtype)?)
}

impl Attention {
    fn forward(
        &self,
        x: &Tensor,
        kv_cache: &mut Option<KvCache>,
        seq_len: usize,
        start_pos: usize,
        bidirectional_spans: &[AttentionSpan],
    ) -> Result<Tensor> {
        let (b_sz, _, _) = x.dims3()?;
        let profile = layer_profile_enabled();

        #[cfg(feature = "cuda")]
        let grouped_qkv = if !profile {
            match (
                self.q_proj.candle(),
                self.k_proj.candle(),
                self.v_proj.as_ref().and_then(GemmaQMatMul::candle),
            ) {
                (Some(q), Some(k), Some(v)) => super::dense_qmatmul_cuda::forward(x, &[q, k, v])?,
                _ => None,
            }
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let grouped_qkv: Option<Vec<Tensor>> = None;
        let (q, k, v) = if let Some(mut outputs) = grouped_qkv {
            anyhow::ensure!(outputs.len() == 3, "grouped Gemma 4 QKV count mismatch");
            let v = outputs.pop().unwrap();
            let k = outputs.pop().unwrap();
            let q = outputs.pop().unwrap();
            (q, k, v)
        } else {
            let started = profile.then(Instant::now);
            let q = self.q_proj.forward(x)?;
            if let Some(started) = started {
                finish_profile_region(x.device(), started, &Q_PROJ_NS)?;
            }
            let started = profile.then(Instant::now);
            let k = self.k_proj.forward(x)?;
            if let Some(started) = started {
                finish_profile_region(x.device(), started, &K_PROJ_NS)?;
            }
            let started = profile.then(Instant::now);
            let v = match &self.v_proj {
                Some(v_proj) => v_proj.forward(x)?,
                None => k.clone(),
            };
            if let Some(started) = started {
                finish_profile_region(x.device(), started, &V_PROJ_NS)?;
            }
            (q, k, v)
        };

        let started = profile.then(Instant::now);
        let (cos, sin) = self.rope_tables.get(
            self.sliding_window.is_some(),
            start_pos,
            seq_len,
            q.device(),
            || {
                rope_cos_sin(
                    self.rope_dim,
                    self.rope_theta,
                    start_pos,
                    seq_len,
                    q.device(),
                    self.rope_freqs.as_ref(),
                )
            },
        )?;
        // Per-head RMSNorm, NeoX rotation and the [batch, heads, seq, dim]
        // layout in one dispatch each for Q and K. The tables hold exactly
        // this step's rows, so they start at row zero.
        let fused_norm_rope: Option<(Tensor, Tensor, Tensor)> = {
            #[cfg(target_os = "macos")]
            {
                if self.rope_dim == self.head_dim && !profile {
                    let q = super::fused_norm_metal::try_fused_head_rms_norm_rope(
                        &q,
                        &self.q_norm.weight,
                        &cos,
                        &sin,
                        0,
                        b_sz,
                        seq_len,
                        self.num_heads,
                        self.head_dim,
                        self.q_norm.eps as f32,
                    )?;
                    let k = super::fused_norm_metal::try_fused_head_rms_norm_rope(
                        &k,
                        &self.k_norm.weight,
                        &cos,
                        &sin,
                        0,
                        b_sz,
                        seq_len,
                        self.num_kv_heads,
                        self.head_dim,
                        self.k_norm.eps as f32,
                    )?;
                    let v = super::fused_norm_metal::try_fused_head_rms_norm_no_weight(
                        &v,
                        b_sz,
                        seq_len,
                        self.num_kv_heads,
                        self.head_dim,
                        self.rms_norm_eps as f32,
                    )?;
                    match (q, k, v) {
                        (Some(q), Some(k), Some(v)) => Some((q, k, v)),
                        _ => None,
                    }
                } else {
                    None
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                None
            }
        };
        if let Some(started) = started.filter(|_| fused_norm_rope.is_some()) {
            finish_profile_region(x.device(), started, &ROPE_NS)?;
        }
        let (q, k, v) = if let Some(fused) = fused_norm_rope {
            fused
        } else {
            let (q, k, v) = self.norm_and_rope(q, k, v, b_sz, seq_len, &cos, &sin)?;
            (q, k, v)
        };

        let started = profile.then(Instant::now);
        let (q, mut k, mut v, rotation) =
            super::prepare_quantized_attention(kv_cache, q, k, v, self.quant_config.as_deref())?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &KV_UPDATE_NS)?;
        }

        if seq_len == 1
            && let Some(window) = self.sliding_window
        {
            let kv_seq = k.dim(2)?;
            if kv_seq > window {
                let offset = kv_seq - window;
                k = k.narrow(2, offset, window)?;
                v = v.narrow(2, offset, window)?;
            }
        }

        let started = profile.then(Instant::now);
        let fused = match self.sliding_window {
            Some(window) if seq_len > 1 && bidirectional_spans.is_empty() => {
                super::try_fused_f16_sdpa_windowed_native(&q, &k, &v, window, self.scale as f32)?
            }
            Some(_) if seq_len > 1 => None,
            _ => super::try_fused_f16_sdpa_native(&q, &k, &v, seq_len > 1, self.scale as f32)?,
        };
        let attn_output = if let Some(output) = fused {
            output
        } else {
            let n_rep = self.num_heads / self.num_kv_heads;
            let k = repeat_kv(k.to_dtype(q.dtype())?, n_rep)?;
            let v = repeat_kv(v.to_dtype(q.dtype())?, n_rep)?;
            let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
            let kv_seq = attn_weights.dim(3)?;
            let attn_weights = if seq_len > 1 {
                let mask = create_attention_mask(
                    seq_len,
                    kv_seq,
                    start_pos,
                    self.sliding_window,
                    bidirectional_spans,
                    x.device(),
                )?;
                attn_weights.broadcast_add(&mask)?
            } else {
                attn_weights
            };
            softmax_last_dim(&attn_weights)?.matmul(&v)?
        };
        let attn_output = super::restore_quantized_attention_output(attn_output, rotation)?;
        if let Some(started) = started {
            let counter = if self.sliding_window.is_some() {
                &SLIDING_SDPA_NS
            } else {
                &FULL_SDPA_NS
            };
            finish_profile_region(x.device(), started, counter)?;
        }
        let started = profile.then(Instant::now);
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
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &ATTN_OUTPUT_LAYOUT_NS)?;
        }
        let started = profile.then(Instant::now);
        #[cfg(all(feature = "cuda", not(target_os = "macos")))]
        if !profile
            && let Some(o_proj) = self.o_proj.candle()
            && let Some(output) =
                super::dense_qmatmul_cuda::forward_one_to_f32(&attn_output, o_proj)?
        {
            return Ok(output);
        }
        #[cfg(not(target_os = "macos"))]
        let attn_output = attn_output.to_dtype(x.dtype())?;
        let output = self.o_proj.forward(&attn_output)?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &O_PROJ_NS)?;
        }
        Ok(output)
    }

    /// Unfused per-head norms and rotation, `[batch, seq, heads * dim]` in,
    /// `[batch, heads, seq, dim]` out.
    #[allow(clippy::too_many_arguments)]
    fn norm_and_rope(
        &self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        b_sz: usize,
        seq_len: usize,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let profile = layer_profile_enabled();
        let device = q.device().clone();
        let started = profile.then(Instant::now);
        let fused_layout_norm: Option<(Tensor, Tensor, Tensor)> = {
            #[cfg(target_os = "macos")]
            {
                if seq_len > 1 {
                    let q = super::fused_norm_metal::try_fused_head_rms_norm(
                        &q,
                        &self.q_norm.weight,
                        b_sz,
                        seq_len,
                        self.num_heads,
                        self.head_dim,
                        self.q_norm.eps as f32,
                    )?;
                    let k = super::fused_norm_metal::try_fused_head_rms_norm(
                        &k,
                        &self.k_norm.weight,
                        b_sz,
                        seq_len,
                        self.num_kv_heads,
                        self.head_dim,
                        self.k_norm.eps as f32,
                    )?;
                    let v = super::fused_norm_metal::try_fused_head_rms_norm_no_weight(
                        &v,
                        b_sz,
                        seq_len,
                        self.num_kv_heads,
                        self.head_dim,
                        self.rms_norm_eps as f32,
                    )?;
                    match (q, k, v) {
                        (Some(q), Some(k), Some(v)) => Some((q, k, v)),
                        _ => None,
                    }
                } else {
                    None
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                None
            }
        };
        let (q, k, v, norms_fused) = if let Some((q, k, v)) = fused_layout_norm {
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
        if let Some(started) = started {
            finish_profile_region(&device, started, &QKV_LAYOUT_NS)?;
        }

        let started = profile.then(Instant::now);
        let q = if norms_fused {
            q
        } else {
            self.q_norm.forward(&q)?
        };
        if let Some(started) = started.filter(|_| !norms_fused) {
            finish_profile_region(&device, started, &Q_NORM_NS)?;
        }
        let started = profile.then(Instant::now);
        let k = if norms_fused {
            k
        } else {
            self.k_norm.forward(&k)?
        };
        if let Some(started) = started.filter(|_| !norms_fused) {
            finish_profile_region(&device, started, &K_NORM_NS)?;
        }
        let started = profile.then(Instant::now);
        let v = if norms_fused {
            v
        } else {
            rms_norm_no_weight(&v, self.rms_norm_eps)?
        };
        if let Some(started) = started.filter(|_| !norms_fused) {
            finish_profile_region(&device, started, &V_NORM_NS)?;
        }

        let started = profile.then(Instant::now);
        let q = apply_rotary_emb(&q, self.rope_dim, cos, sin)?;
        let k = apply_rotary_emb(&k, self.rope_dim, cos, sin)?;
        if let Some(started) = started {
            finish_profile_region(&device, started, &ROPE_NS)?;
        }
        Ok((q, k, v))
    }

    /// Decode lanes with different absolute positions while keeping the
    /// projection and output kernels batched. Cache append and attention stay
    /// lane-local so sliding-window bounds and valid KV lengths remain exact.
    fn forward_ragged_decode(
        &self,
        x: &Tensor,
        caches: &mut [&mut Option<KvCache>],
        positions: &[usize],
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        anyhow::ensure!(seq_len == 1, "ragged attention is decode-only");
        anyhow::ensure!(batch == caches.len(), "ragged attention cache mismatch");
        anyhow::ensure!(
            batch == positions.len(),
            "ragged attention position mismatch"
        );

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = match &self.v_proj {
            Some(v_proj) => v_proj.forward(x)?,
            None => k.clone(),
        };
        let q = q.reshape((batch, self.num_heads, 1, self.head_dim))?;
        let k = k.reshape((batch, self.num_kv_heads, 1, self.head_dim))?;
        let v = v.reshape((batch, self.num_kv_heads, 1, self.head_dim))?;
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let v = rms_norm_no_weight(&v, self.rms_norm_eps)?;
        let n_rep = self.num_heads / self.num_kv_heads;
        let mut outputs = Vec::with_capacity(batch);

        for lane in 0..batch {
            let (cos, sin) = rope_cos_sin(
                self.rope_dim,
                self.rope_theta,
                positions[lane],
                1,
                q.device(),
                self.rope_freqs.as_ref(),
            )?;
            let q = apply_rotary_emb(&q.narrow(0, lane, 1)?, self.rope_dim, &cos, &sin)?;
            let k = apply_rotary_emb(&k.narrow(0, lane, 1)?, self.rope_dim, &cos, &sin)?;
            let v = v.narrow(0, lane, 1)?;
            let (mut k, mut v) = super::kv_cache_update_native(
                &mut *caches[lane],
                k,
                v,
                self.quant_config.as_deref(),
            )?;
            if let Some(window) = self.sliding_window {
                let kv_seq = k.dim(2)?;
                if kv_seq > window {
                    let offset = kv_seq - window;
                    k = k.narrow(2, offset, window)?;
                    v = v.narrow(2, offset, window)?;
                }
            }
            let output = if let Some(output) =
                super::try_fused_f16_sdpa(&q, &k, &v, false, self.scale as f32)?
            {
                output
            } else {
                let k = repeat_kv(k.to_dtype(q.dtype())?, n_rep)?;
                let v = repeat_kv(v.to_dtype(q.dtype())?, n_rep)?;
                let weights = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
                softmax_last_dim(&weights)?.matmul(&v)?
            };
            outputs.push(output);
        }

        let output_refs = outputs.iter().collect::<Vec<_>>();
        let output = Tensor::cat(&output_refs, 0)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, 1, ()))?;
        self.o_proj.forward(&output)
    }
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let profile = layer_profile_enabled();
        #[cfg(feature = "cuda")]
        if !profile
            && let (Some(gate), Some(up), Some(down)) = (
                self.gate_proj.candle(),
                self.up_proj.candle(),
                self.down_proj.candle(),
            )
            && let Some(activated) = super::dense_qmatmul_cuda::gelu_gate_up(x, gate, up)?
        {
            if let Some(output) = super::dense_qmatmul_cuda::forward_one_to_f32(&activated, down)? {
                return Ok(output);
            }
            return self.down_proj.forward(&activated);
        }
        #[cfg(feature = "cuda")]
        let grouped = if !profile {
            match (self.gate_proj.candle(), self.up_proj.candle()) {
                (Some(gate), Some(up)) => super::dense_qmatmul_cuda::forward(x, &[gate, up])?,
                _ => None,
            }
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let grouped: Option<Vec<Tensor>> = None;
        let (gate, up) = if let Some(mut outputs) = grouped {
            anyhow::ensure!(outputs.len() == 2, "grouped Gemma 4 MLP count mismatch");
            let up = outputs.pop().unwrap();
            let gate = outputs.pop().unwrap();
            (gate, up)
        } else {
            let started = profile.then(Instant::now);
            let up = self.up_proj.forward(x)?;
            if let Some(started) = started {
                finish_profile_region(x.device(), started, &FFN_UP_NS)?;
            }
            let started = profile.then(Instant::now);
            let gate = self.gate_proj.forward(x)?;
            if let Some(started) = started {
                finish_profile_region(x.device(), started, &FFN_GATE_NS)?;
            }
            (gate, up)
        };
        let started = profile.then(Instant::now);
        let activated = gelu_mul(&gate, &up)?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &FFN_ACTIVATION_NS)?;
        }
        let started = profile.then(Instant::now);
        let output = self.down_proj.forward(&activated)?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &FFN_DOWN_NS)?;
        }
        Ok(output)
    }
}

impl TransformerBlock {
    /// One layer. `normed` is `attn_norm(x)` when the previous layer already
    /// produced it; with `next_norm`, the result also carries `next_norm` of
    /// the output when a fused kernel can write it, for the next layer.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        normed: Option<Tensor>,
        next_norm: Option<&RmsNorm>,
        kv_cache: &mut Option<KvCache>,
        seq_len: usize,
        start_pos: usize,
        bidirectional_spans: &[AttentionSpan],
    ) -> Result<(Tensor, Option<Tensor>)> {
        let profile = layer_profile_enabled();
        if profile {
            PROFILE_CALLS.fetch_add(1, Ordering::Relaxed);
        }
        let residual = x;
        let started = profile.then(Instant::now);
        let x_attn = match normed {
            Some(normed) => normed,
            None => self.attn_norm.forward(x)?,
        };
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &ATTN_NORM_NS)?;
        }
        let x_attn =
            self.attn
                .forward(&x_attn, kv_cache, seq_len, start_pos, bidirectional_spans)?;
        if !profile {
            if let Some(moe) = &self.moe {
                return self.moe_tail(moe, residual, &x_attn, next_norm);
            }
            let (attn_out, ffn_in) = self.post_attention(residual, &x_attn)?;
            let ffn_out = self.feed_forward(&attn_out, &ffn_in)?;
            return self.post_ffn(&attn_out, &ffn_out, next_norm);
        }
        let started = profile.then(Instant::now);
        let x_attn = self.attn_post_norm.forward(&x_attn)?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &ATTN_POST_NORM_NS)?;
        }
        let (attn_out, ffn_out) = if profile {
            let started = Instant::now();
            let attn_out = (residual + x_attn)?;
            finish_profile_region(x.device(), started, &ATTN_RESIDUAL_NS)?;
            let started = Instant::now();
            let ffn_out = self.ffn_norm.forward(&attn_out)?;
            finish_profile_region(x.device(), started, &FFN_NORM_NS)?;
            (attn_out, ffn_out)
        } else {
            self.ffn_norm.forward_residual(residual, &x_attn)?
        };
        let ffn_out = self.feed_forward(&attn_out, &ffn_out)?;
        let started = profile.then(Instant::now);
        let ffn_out = self.ffn_post_norm.forward(&ffn_out)?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &FFN_POST_NORM_NS)?;
        }
        let started = profile.then(Instant::now);
        let mut block_out = (&attn_out + ffn_out)?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &FFN_RESIDUAL_NS)?;
        }
        if let Some(scale) = &self.layer_output_scale {
            let started = profile.then(Instant::now);
            block_out = block_out.broadcast_mul(scale)?;
            if let Some(started) = started {
                finish_profile_region(x.device(), started, &OUTPUT_SCALE_NS)?;
            }
        }

        Ok((block_out, None))
    }

    fn forward_ragged_decode(
        &self,
        x: &Tensor,
        normed: Option<Tensor>,
        next_norm: Option<&RmsNorm>,
        caches: &mut [&mut Option<KvCache>],
        positions: &[usize],
    ) -> Result<(Tensor, Option<Tensor>)> {
        let residual = x;
        let x_attn = match normed {
            Some(normed) => normed,
            None => self.attn_norm.forward(x)?,
        };
        let x_attn = self
            .attn
            .forward_ragged_decode(&x_attn, caches, positions)?;
        if let Some(moe) = &self.moe {
            return self.moe_tail(moe, residual, &x_attn, next_norm);
        }
        let (attn_out, ffn_in) = self.post_attention(residual, &x_attn)?;
        let ffn_out = self.feed_forward(&attn_out, &ffn_in)?;
        self.post_ffn(&attn_out, &ffn_out, next_norm)
    }

    /// The FFN output before `post_ffw_norm`: the dense MLP, or for MoE
    /// layers the normalized shared MLP plus the normalized routed experts.
    fn feed_forward(&self, attn_out: &Tensor, ffn_in: &Tensor) -> Result<Tensor> {
        let shared = self.ffn.forward(ffn_in)?;
        let Some(moe) = &self.moe else {
            return Ok(shared);
        };
        let shared = moe.shared_post_norm.forward(&shared)?;
        let routed = moe.forward(
            &moe.pre_norm.forward(attn_out)?,
            &rms_norm_no_weight(attn_out, moe.rms_norm_eps)?,
        )?;
        let routed = moe.experts_post_norm.forward(&routed)?;
        (shared + routed).map_err(Into::into)
    }

    /// Everything after attention in a MoE layer: on Metal one kernel writes
    /// the residual plus the shared, expert and router inputs, and one kernel
    /// applies both branch norms, the post-FFN norm, the residual and scale.
    fn moe_tail(
        &self,
        moe: &GemmaMoe,
        residual: &Tensor,
        x_attn: &Tensor,
        next_norm: Option<&RmsNorm>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        #[cfg(target_os = "macos")]
        if let Some(planes) = super::fused_norm_metal::try_fused_moe_attention_planes(
            x_attn,
            residual,
            &self.attn_post_norm.weight,
            &self.ffn_norm.weight,
            &moe.pre_norm.weight,
            moe.rms_norm_eps as f32,
        )? {
            let shared = self.ffn.forward(&planes.shared)?;
            let routed = moe.forward(&planes.experts, &planes.router)?;
            // One shared eps: carry the next norm only when it matches.
            let next_weight = next_norm
                .filter(|norm| norm.eps == moe.rms_norm_eps)
                .map(|norm| &norm.weight);
            if let Some(out) = super::fused_norm_metal::try_fused_dual_post_norm_residual(
                &shared,
                &routed,
                &planes.sum,
                &moe.shared_post_norm.weight,
                &moe.experts_post_norm.weight,
                &self.ffn_post_norm.weight,
                moe.rms_norm_eps as f32,
                self.layer_output_scale.as_ref(),
                next_weight,
            )? {
                return Ok(out);
            }
            let combined = (moe.shared_post_norm.forward(&shared)?
                + moe.experts_post_norm.forward(&routed)?)?;
            return self.post_ffn(&planes.sum, &combined, next_norm);
        }
        let (attn_out, ffn_in) = self.post_attention(residual, x_attn)?;
        let ffn_out = self.feed_forward(&attn_out, &ffn_in)?;
        self.post_ffn(&attn_out, &ffn_out, next_norm)
    }

    /// Attention post-norm and residual, then the FFN input norm. Returns the
    /// new residual stream and the normalized FFN input.
    fn post_attention(&self, residual: &Tensor, x_attn: &Tensor) -> Result<(Tensor, Tensor)> {
        #[cfg(target_os = "macos")]
        if let Some((attn_out, Some(ffn_in))) =
            super::fused_norm_metal::try_fused_post_norm_residual(
                x_attn,
                residual,
                &self.attn_post_norm.weight,
                self.attn_post_norm.eps as f32,
                Some((&self.ffn_norm.weight, self.ffn_norm.eps as f32)),
                None,
            )?
        {
            return Ok((attn_out, ffn_in));
        }
        let x_attn = self.attn_post_norm.forward(x_attn)?;
        self.ffn_norm.forward_residual(residual, &x_attn)
    }

    /// FFN post-norm, residual and the optional per-layer output scale, plus
    /// `next_norm` of the result when the fused kernel writes it.
    fn post_ffn(
        &self,
        attn_out: &Tensor,
        ffn_out: &Tensor,
        next_norm: Option<&RmsNorm>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        #[cfg(target_os = "macos")]
        if let Some(out) = super::fused_norm_metal::try_fused_post_norm_residual(
            ffn_out,
            attn_out,
            &self.ffn_post_norm.weight,
            self.ffn_post_norm.eps as f32,
            next_norm.map(|norm| (&norm.weight, norm.eps as f32)),
            self.layer_output_scale.as_ref(),
        )? {
            return Ok(out);
        }
        let ffn_out = self.ffn_post_norm.forward(ffn_out)?;
        let mut block_out = (attn_out + ffn_out)?;
        if let Some(scale) = &self.layer_output_scale {
            block_out = block_out.broadcast_mul(scale)?;
        }
        Ok((block_out, None))
    }
}

fn qtensor_to_qmatmul(qt: Arc<QTensor>) -> Result<QMatMul> {
    QMatMul::from_arc(qt).map_err(Into::into)
}

fn lm_head_forward(lm_head: &QMatMul, hidden: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Some(logits) = super::dense_qmatmul_cuda::forward_one_to_f32(hidden, lm_head)? {
        return Ok(logits);
    }
    lm_head.forward(hidden).map_err(Into::into)
}

impl Gemma4Stage {
    /// Construct a Gemma 4 stage from the tensors assigned to its layer range.
    ///
    /// Returns an error when required metadata or tensors are missing or have
    /// incompatible dimensions. The optional cache quantization policy is
    /// passed through to the stage's cache implementation.
    pub fn from_stage_tensors(
        stage: StageTensors,
        quant_config: Option<Arc<TurboQuantConfig>>,
    ) -> Result<Self> {
        let StageTensors {
            range,
            metadata: md,
            mut tensors,
            device,
            ..
        } = stage;
        let is_head = range.is_head();
        let is_tail = range.is_tail();
        #[cfg(feature = "cuda")]
        super::dense_qmatmul_cuda::precompile(&device)?;

        let embedding = if is_head {
            let embd = tensors
                .get("token_embd.weight")
                .context("missing token_embd.weight for head stage")?;
            Some(embd.as_ref().dequantize(&device)?)
        } else {
            None
        };

        let mut blocks = Vec::with_capacity(range.layer_count());
        let rope_tables = Arc::new(RopeTables::default());
        for layer_idx in range.start..range.end {
            let prefix = format!("blk.{layer_idx}");
            blocks.push(build_block(
                &mut tensors,
                &prefix,
                layer_idx,
                &md,
                &device,
                quant_config.clone(),
                &rope_tables,
            )?);
        }

        let output_norm = if is_tail {
            let norm_w = tensors
                .remove("output_norm.weight")
                .context("missing output_norm.weight for tail stage")?;
            Some(RmsNorm {
                weight: norm_w.as_ref().dequantize(&device)?,
                eps: md.rms_norm_eps,
            })
        } else {
            None
        };

        let lm_head = if is_tail {
            let head = tensors
                .remove("output.weight")
                .or_else(|| tensors.remove("token_embd.weight"))
                .context("missing output.weight or token_embd.weight for tail stage")?;
            Some(qtensor_to_qmatmul(head)?)
        } else {
            None
        };

        Ok(Self {
            embedding,
            blocks,
            output_norm,
            lm_head,
            is_head,
            is_tail,
            hidden_dim: md.hidden_dim,
            num_layers: range.layer_count(),
            vocab_size: md.vocab_size,
            final_logit_softcapping: md.final_logit_softcapping,
            device,
        })
    }
}

fn clone_decode_session_cache(cache: &[Option<LayerCache>]) -> Result<Vec<Option<LayerCache>>> {
    cache
        .iter()
        .map(|slot| match slot {
            None => Ok(None),
            Some(layer) => {
                super::clone_decode_layer_cache(layer).map(|cache| cache.map(LayerCache::Attention))
            }
        })
        .collect()
}

impl Gemma4Stage {
    fn forward_ragged_decode_batch(
        &self,
        tokens: &[u32],
        caches: &mut [&mut [Option<LayerCache>]],
        positions: &[usize],
    ) -> Result<Vec<Tensor>> {
        let batch = tokens.len();
        let embedding = self.embedding.as_ref().unwrap();
        let token_tensor = Tensor::new(tokens, &self.device)?;
        let mut hidden_states =
            embedding
                .index_select(&token_tensor, 0)?
                .reshape((batch, 1, self.hidden_dim))?;
        hidden_states = (hidden_states * (self.hidden_dim as f64).sqrt())?;

        // Cache metadata is cloned transactionally. Appends can fill unused
        // preallocated slots, but logical lengths are published only after the
        // entire model forward succeeds.
        let mut next_caches = caches
            .iter()
            .map(|cache| clone_decode_session_cache(cache))
            .collect::<Result<Vec<_>>>()?;

        let mut normed = None;
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let next_norm = self.blocks.get(layer_idx + 1).map(|next| &next.attn_norm);
            let mut layer_caches = Vec::with_capacity(batch);
            for cache in &mut next_caches {
                let kv = match cache[layer_idx].take() {
                    Some(LayerCache::Attention(kv)) => Some(kv),
                    None => None,
                    Some(_) => anyhow::bail!("ragged decode encountered non-attention state"),
                };
                layer_caches.push(kv);
            }
            let mut layer_refs = layer_caches.iter_mut().collect::<Vec<_>>();
            (hidden_states, normed) = block.forward_ragged_decode(
                &hidden_states,
                normed,
                next_norm,
                &mut layer_refs,
                positions,
            )?;
            for (cache, kv) in next_caches.iter_mut().zip(layer_caches) {
                cache[layer_idx] = kv.map(LayerCache::Attention);
            }
        }

        let normed = self.output_norm.as_ref().unwrap().forward(&hidden_states)?;
        let mut logits = lm_head_forward(self.lm_head.as_ref().unwrap(), &normed.squeeze(1)?)?;
        if self.final_logit_softcapping > 0.0 {
            logits = (logits * (1.0 / self.final_logit_softcapping))?.tanh()?;
            logits = (logits * self.final_logit_softcapping)?;
        }
        let outputs = (0..batch)
            .map(|lane| logits.narrow(0, lane, 1)?.squeeze(0).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;

        for (target, source) in caches.iter_mut().zip(next_caches) {
            anyhow::ensure!(target.len() == source.len(), "ragged cache layer mismatch");
            for (slot, next) in target.iter_mut().zip(source) {
                *slot = next;
            }
        }
        Ok(outputs)
    }
}

impl Gemma4Stage {
    fn forward_inner(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
        output_logits: bool,
    ) -> Result<StageOutput> {
        let (mut hidden_states, bidirectional_spans) = match input {
            StageInput::Tokens(tokens) => {
                anyhow::ensure!(self.is_head, "only head stage can receive tokens");
                let embedding = self.embedding.as_ref().unwrap();
                let token_tensor = Tensor::new(tokens.as_slice(), &self.device)?;
                let hidden = embedding.index_select(&token_tensor, 0)?.reshape((
                    1,
                    seq_len,
                    self.hidden_dim,
                ))?;
                ((hidden * (self.hidden_dim as f64).sqrt())?, Vec::new())
            }
            StageInput::Activations {
                tensor,
                bidirectional_spans,
                ..
            } => {
                let tensor = if tensor.dims().len() == 2 {
                    tensor.unsqueeze(0)?
                } else {
                    tensor
                };
                (tensor, bidirectional_spans)
            }
        };
        validate_attention_spans(&bidirectional_spans, start_pos, seq_len)?;

        let mut normed = None;
        for (i, block) in self.blocks.iter().enumerate() {
            let next_norm = self.blocks.get(i + 1).map(|next| &next.attn_norm);
            let carried = normed.take();
            (hidden_states, normed) = with_kv_cache(&mut cache[i], |kv| {
                block.forward(
                    &hidden_states,
                    carried,
                    next_norm,
                    kv,
                    seq_len,
                    start_pos,
                    &bidirectional_spans,
                )
            })?;
        }

        if self.is_tail && output_logits {
            let normed = self.output_norm.as_ref().unwrap().forward(&hidden_states)?;
            let last_hidden = if seq_len > 1 {
                normed.i((.., seq_len - 1, ..))?
            } else {
                normed.squeeze(1)?
            };
            let logits = lm_head_forward(self.lm_head.as_ref().unwrap(), &last_hidden)?;
            let mut logits = logits.squeeze(0)?;
            if self.final_logit_softcapping > 0.0 {
                logits = (logits * (1.0 / self.final_logit_softcapping))?.tanh()?;
                logits = (logits * self.final_logit_softcapping)?;
            }
            Ok(StageOutput::Logits(logits))
        } else {
            Ok(StageOutput::Activations(hidden_states.squeeze(0)?))
        }
    }
}

impl ModelArchitecture for Gemma4Stage {
    fn forward(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_inner(input, cache, seq_len, start_pos, true)
    }

    fn forward_prefill_cache_only(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_inner(input, cache, seq_len, start_pos, false)
    }

    fn embed_tokens(&self, tokens: &[u32]) -> Result<Tensor> {
        anyhow::ensure!(self.is_head, "only a Gemma 4 head stage can embed tokens");
        anyhow::ensure!(!tokens.is_empty(), "cannot embed an empty token list");
        let token_tensor = Tensor::new(tokens, &self.device)?;
        let embeddings = self
            .embedding
            .as_ref()
            .context("Gemma 4 head stage has no token embedding")?
            .index_select(&token_tensor, 0)?;
        (embeddings * (self.hidden_dim as f64).sqrt()).map_err(Into::into)
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
            "Gemma 4 decode batching unavailable"
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
        let ragged = !positions.iter().all(|position| *position == positions[0]);
        DECODE_BATCH_LOGGED.get_or_init(|| {
            tracing::info!(
                batch_size = tokens.len(),
                ragged,
                "Gemma 4 tensor decode batching active"
            );
        });
        if ragged {
            let outputs = self.forward_ragged_decode_batch(tokens, caches, positions)?;
            for slot in batch_cache {
                *slot = None;
            }
            return Ok(outputs);
        }

        let start_pos = positions[0];
        let batch = tokens.len();
        let embedding = self.embedding.as_ref().unwrap();
        let token_tensor = Tensor::new(tokens, &self.device)?;
        let mut hidden_states =
            embedding
                .index_select(&token_tensor, 0)?
                .reshape((batch, 1, self.hidden_dim))?;
        hidden_states = (hidden_states * (self.hidden_dim as f64).sqrt())?;

        let mut next_batch_cache = Vec::with_capacity(self.blocks.len());
        let mut normed = None;
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let next_norm = self.blocks.get(layer_idx + 1).map(|next| &next.attn_norm);
            let mut batched_cache = match &batch_cache[layer_idx] {
                Some(cache) => super::clone_decode_layer_cache(cache)?,
                None => super::pack_decode_layer_cache(caches, layer_idx)?,
            };
            (hidden_states, normed) = block.forward(
                &hidden_states,
                normed,
                next_norm,
                &mut batched_cache,
                1,
                start_pos,
                &[],
            )?;
            next_batch_cache.push(batched_cache);
        }

        let normed = self.output_norm.as_ref().unwrap().forward(&hidden_states)?;
        let mut logits = lm_head_forward(self.lm_head.as_ref().unwrap(), &normed.squeeze(1)?)?;
        if self.final_logit_softcapping > 0.0 {
            logits = (logits * (1.0 / self.final_logit_softcapping))?.tanh()?;
            logits = (logits * self.final_logit_softcapping)?;
        }
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
        "gemma4"
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

fn build_block(
    tensors: &mut HashMap<String, Arc<QTensor>>,
    prefix: &str,
    layer_idx: usize,
    md: &crate::gguf_loader::GgufMetadata,
    device: &Device,
    quant_config: Option<Arc<TurboQuantConfig>>,
    rope_tables: &Arc<RopeTables>,
) -> Result<TransformerBlock> {
    let is_swa = md
        .sliding_window_pattern
        .get(layer_idx)
        .copied()
        .unwrap_or(false);
    let head_dim = if is_swa && md.head_dim_swa > 0 {
        md.head_dim_swa
    } else {
        md.head_dim
    };
    let rope_dim = if is_swa {
        md.rope_dimension_count_swa.max(head_dim)
    } else {
        md.rope_dimension_count.max(head_dim)
    };
    let rope_theta = if is_swa && md.rope_theta_swa > 0.0 {
        md.rope_theta_swa
    } else {
        md.rope_theta
    };
    let num_kv_heads = md
        .head_count_kv_arr
        .get(layer_idx)
        .copied()
        .filter(|&n| n > 0)
        .unwrap_or(md.num_kv_heads);
    let rope_freqs = if is_swa {
        None
    } else {
        tensors
            .get("rope_freqs.weight")
            .map(|qt| qt.as_ref().dequantize(device))
            .transpose()?
    };
    let attn = Attention {
        q_proj: GemmaQMatMul::from_qtensor(
            take_block_tensor(tensors, prefix, "attn_q.weight")?,
            device,
        )?,
        k_proj: GemmaQMatMul::from_qtensor(
            take_block_tensor(tensors, prefix, "attn_k.weight")?,
            device,
        )?,
        v_proj: take_optional_block_tensor(tensors, prefix, "attn_v.weight")
            .map(|qt| GemmaQMatMul::from_qtensor(qt, device))
            .transpose()?,
        o_proj: GemmaQMatMul::from_qtensor(
            take_block_tensor(tensors, prefix, "attn_output.weight")?,
            device,
        )?,
        q_norm: RmsNorm {
            weight: take_block_tensor(tensors, prefix, "attn_q_norm.weight")?
                .as_ref()
                .dequantize(device)?,
            eps: md.rms_norm_eps,
        },
        k_norm: RmsNorm {
            weight: take_block_tensor(tensors, prefix, "attn_k_norm.weight")?
                .as_ref()
                .dequantize(device)?,
            eps: md.rms_norm_eps,
        },
        num_heads: md.num_attention_heads,
        num_kv_heads,
        head_dim,
        rope_dim,
        rope_theta,
        rope_freqs,
        rope_tables: Arc::clone(rope_tables),
        sliding_window: if is_swa && md.sliding_window > 0 {
            Some(md.sliding_window)
        } else {
            None
        },
        // Gemma 4 uses self.scaling = 1.0 rather than the usual 1/sqrt(head_dim).
        scale: if md.attention_scale > 0.0 {
            md.attention_scale
        } else {
            1.0
        },
        rms_norm_eps: md.rms_norm_eps,
        quant_config,
    };

    let layer_output_scale =
        take_optional_block_tensor(tensors, prefix, "layer_output_scale.weight")
            .map(|qt| qt.as_ref().dequantize(device))
            .transpose()?;
    let moe = build_moe(tensors, prefix, md, device)?.map(Box::new);

    Ok(TransformerBlock {
        attn_norm: RmsNorm {
            weight: take_block_tensor(tensors, prefix, "attn_norm.weight")?
                .as_ref()
                .dequantize(device)?,
            eps: md.rms_norm_eps,
        },
        attn,
        attn_post_norm: RmsNorm {
            weight: take_block_tensor(tensors, prefix, "post_attention_norm.weight")?
                .as_ref()
                .dequantize(device)?,
            eps: md.rms_norm_eps,
        },
        ffn_norm: RmsNorm {
            weight: take_block_tensor(tensors, prefix, "ffn_norm.weight")?
                .as_ref()
                .dequantize(device)?,
            eps: md.rms_norm_eps,
        },
        ffn: Mlp {
            gate_proj: GemmaQMatMul::from_qtensor(
                take_block_tensor(tensors, prefix, "ffn_gate.weight")?,
                device,
            )?,
            up_proj: GemmaQMatMul::from_qtensor(
                take_block_tensor(tensors, prefix, "ffn_up.weight")?,
                device,
            )?,
            down_proj: GemmaQMatMul::from_qtensor(
                take_block_tensor(tensors, prefix, "ffn_down.weight")?,
                device,
            )?,
        },
        moe,
        ffn_post_norm: RmsNorm {
            weight: take_block_tensor(tensors, prefix, "post_ffw_norm.weight")?
                .as_ref()
                .dequantize(device)?,
            eps: md.rms_norm_eps,
        },
        layer_output_scale,
    })
}

/// Routed experts for layers that carry a router (`ffn_gate_inp`).
fn build_moe(
    tensors: &mut HashMap<String, Arc<QTensor>>,
    prefix: &str,
    md: &crate::gguf_loader::GgufMetadata,
    device: &Device,
) -> Result<Option<GemmaMoe>> {
    let Some(router) = take_optional_block_tensor(tensors, prefix, "ffn_gate_inp.weight") else {
        return Ok(None);
    };
    let (experts, active, expert_ffn) = (md.expert_count, md.expert_used_count, md.expert_ffn_dim);
    anyhow::ensure!(
        experts > 0 && active > 0 && active <= experts && expert_ffn > 0,
        "Gemma 4 {prefix} has a router but invalid expert metadata"
    );
    let hidden = md.hidden_dim;
    let mut dequantize = |name: &str| -> Result<Tensor> {
        take_block_tensor(tensors, prefix, name)?
            .as_ref()
            .dequantize(device)?
            .to_dtype(DType::F32)
            .map_err(Into::into)
    };
    let norm = |weight: Tensor| RmsNorm {
        weight,
        eps: md.rms_norm_eps,
    };
    let router = router.as_ref().dequantize(device)?.to_dtype(DType::F32)?;
    anyhow::ensure!(
        router.dims() == [experts, hidden],
        "Gemma 4 {prefix} router shape {:?} differs from [{experts}, {hidden}]",
        router.dims()
    );
    let router_scale = (dequantize("ffn_gate_inp.scale")? / (hidden as f64).sqrt())?;
    let down_scale = dequantize("ffn_down_exps.scale")?;
    anyhow::ensure!(
        router_scale.dims() == [hidden] && down_scale.dims() == [experts],
        "Gemma 4 {prefix} expert scale shapes differ"
    );
    let router = router
        .broadcast_mul(&router_scale.unsqueeze(0)?)?
        .contiguous()?;
    let pre_norm = norm(dequantize("pre_ffw_norm_2.weight")?);
    let shared_post_norm = norm(dequantize("post_ffw_norm_1.weight")?);
    let experts_post_norm = norm(dequantize("post_ffw_norm_2.weight")?);
    let gate_up = take_block_tensor(tensors, prefix, "ffn_gate_up_exps.weight")
        .with_context(|| format!("Gemma 4 {prefix} needs packed ffn_gate_up_exps experts"))?;
    let down = take_block_tensor(tensors, prefix, "ffn_down_exps.weight")?;

    #[cfg(any(target_os = "macos", feature = "cuda"))]
    let experts_impl = {
        use super::indexed_moe::{IndexedMoeMatMul, indexed_moe_available};
        anyhow::ensure!(
            indexed_moe_available(device, gate_up.dtype())
                && indexed_moe_available(device, down.dtype()),
            "Gemma 4 mixture-of-experts layers need indexed expert kernels for {:?}/{:?} on this device",
            gate_up.dtype(),
            down.dtype()
        );
        let gate_up =
            IndexedMoeMatMul::from_qtensor(&gate_up, experts, 2 * expert_ffn, hidden, device)?;
        #[cfg(target_os = "macos")]
        gate_up.prepare_geglu()?;
        GemmaExperts {
            gate_up,
            down: IndexedMoeMatMul::from_qtensor(&down, experts, hidden, expert_ffn, device)?,
        }
    };
    #[cfg(not(any(target_os = "macos", feature = "cuda")))]
    let experts_impl = {
        let _ = (gate_up, down);
        anyhow::bail!("Gemma 4 mixture-of-experts models require Metal or CUDA")
    };

    Ok(Some(GemmaMoe {
        router,
        pre_norm,
        shared_post_norm,
        experts_post_norm,
        experts: experts_impl,
        down_scale,
        num_active: active,
        rms_norm_eps: md.rms_norm_eps,
    }))
}

fn take_block_tensor(
    tensors: &mut HashMap<String, Arc<QTensor>>,
    prefix: &str,
    name: &str,
) -> Result<Arc<QTensor>> {
    tensors
        .remove(&format!("{prefix}.{name}"))
        .with_context(|| format!("missing tensor: {prefix}.{name}"))
}

fn take_optional_block_tensor(
    tensors: &mut HashMap<String, Arc<QTensor>>,
    prefix: &str,
    name: &str,
) -> Option<Arc<QTensor>> {
    tensors.remove(&format!("{prefix}.{name}"))
}

fn apply_rotary_emb(x: &Tensor, rope_dim: usize, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_, _, _, head_dim) = x.dims4()?;
    if rope_dim >= head_dim {
        return candle_nn::rotary_emb::rope(&x.contiguous()?, cos, sin).map_err(Into::into);
    }

    let x_rope = x.narrow(3, 0, rope_dim)?;
    let x_pass = x.narrow(3, rope_dim, head_dim - rope_dim)?;
    let x_rope = candle_nn::rotary_emb::rope(&x_rope.contiguous()?, cos, sin)?;
    Tensor::cat(&[&x_rope, &x_pass.contiguous()?], 3).map_err(Into::into)
}

fn rope_cos_sin(
    rope_dim: usize,
    theta: f64,
    start_pos: usize,
    seq_len: usize,
    device: &Device,
    freq_factors: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let half_dim = rope_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / rope_dim as f32))
        .collect();
    let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
    let positions = Tensor::arange(start_pos as u32, (start_pos + seq_len) as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((seq_len, 1))?;
    let mut angles = positions.matmul(&inv_freq.reshape((1, half_dim))?)?;
    if let Some(freq_factors) = freq_factors {
        let freq_factors = freq_factors
            .narrow(0, 0, half_dim)?
            .reshape((1, half_dim))?;
        angles = angles.broadcast_div(&freq_factors)?;
    }
    Ok((angles.cos()?, angles.sin()?))
}

/// Candle's `NewGelu` of `gate` times `up`, fused on Metal.
fn gelu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    #[cfg(target_os = "macos")]
    if gate.device().is_metal()
        && gate.dtype() == DType::F32
        && up.dtype() == DType::F32
        && gate.is_contiguous()
        && up.is_contiguous()
        && gate.shape() == up.shape()
        && std::env::var_os("DINF_DISABLE_FUSED_GELU_MUL").is_none()
    {
        return super::indexed_moe_metal::fused_gelu_mul(gate, up).map_err(Into::into);
    }
    (candle_nn::Activation::NewGelu.forward(gate)? * up).map_err(Into::into)
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

fn create_attention_mask(
    seq_len: usize,
    kv_seq: usize,
    start_pos: usize,
    sliding_window: Option<usize>,
    bidirectional_spans: &[AttentionSpan],
    device: &Device,
) -> Result<Tensor> {
    let kv_start = start_pos + seq_len - kv_seq;
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            let query_pos = start_pos + i;
            (0..kv_seq).map(move |j| {
                let key_pos = kv_start + j;
                let causal = key_pos <= query_pos;
                let same_bidirectional_span = sliding_window.is_some()
                    && bidirectional_spans.iter().any(|span| {
                        let end = span.start + span.len;
                        query_pos >= span.start
                            && query_pos < end
                            && key_pos >= span.start
                            && key_pos < end
                    });
                let in_window = sliding_window
                    .map(|window| key_pos + window > query_pos)
                    .unwrap_or(true);
                if (causal || same_bidirectional_span) && in_window {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Tensor::from_vec(mask, (1, 1, seq_len, kv_seq), device).map_err(Into::into)
}

fn validate_attention_spans(
    spans: &[AttentionSpan],
    start_pos: usize,
    seq_len: usize,
) -> Result<()> {
    let end_pos = start_pos
        .checked_add(seq_len)
        .context("Gemma 4 prompt position overflow")?;
    let mut prior_end = start_pos;
    for span in spans {
        let span_end = span
            .start
            .checked_add(span.len)
            .context("Gemma 4 image span overflow")?;
        anyhow::ensure!(span.len > 0, "Gemma 4 image span cannot be empty");
        anyhow::ensure!(
            span.start >= start_pos && span_end <= end_pos,
            "Gemma 4 image span lies outside this prefill"
        );
        anyhow::ensure!(
            span.start >= prior_end,
            "Gemma 4 image spans must be sorted and disjoint"
        );
        prior_end = span_end;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architectures::PreallocatedKvCache;

    fn qmatrix(rows: usize, cols: usize) -> QMatMul {
        QMatMul::Tensor(Tensor::zeros((rows, cols), DType::F32, &Device::Cpu).unwrap())
    }

    fn matrix(rows: usize, cols: usize) -> GemmaQMatMul {
        GemmaQMatMul::Candle(qmatrix(rows, cols))
    }

    fn norm(size: usize) -> RmsNorm {
        RmsNorm {
            weight: Tensor::ones(size, DType::F32, &Device::Cpu).unwrap(),
            eps: 1e-6,
        }
    }

    fn cache(batch: usize, len: usize) -> KvCache {
        let shape = (batch, 1, len, 4);
        let k = Tensor::zeros(shape, DType::F16, &Device::Cpu).unwrap();
        let v = Tensor::zeros(shape, DType::F16, &Device::Cpu).unwrap();
        KvCache::PreallocatedF16(PreallocatedKvCache::from_f16(k, v).unwrap())
    }

    fn tiny_stage() -> Gemma4Stage {
        let attention = Attention {
            q_proj: matrix(4, 4),
            k_proj: matrix(4, 4),
            v_proj: Some(matrix(4, 4)),
            o_proj: matrix(4, 4),
            q_norm: norm(4),
            k_norm: norm(4),
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 4,
            rope_dim: 4,
            rope_theta: 10_000.0,
            rope_freqs: None,
            rope_tables: Arc::default(),
            sliding_window: None,
            scale: 1.0,
            rms_norm_eps: 1e-6,
            quant_config: None,
        };
        Gemma4Stage {
            embedding: Some(Tensor::zeros((8, 4), DType::F32, &Device::Cpu).unwrap()),
            blocks: vec![TransformerBlock {
                attn_norm: norm(4),
                attn: attention,
                attn_post_norm: norm(4),
                ffn_norm: norm(4),
                ffn: Mlp {
                    gate_proj: matrix(8, 4),
                    up_proj: matrix(8, 4),
                    down_proj: matrix(4, 8),
                },
                moe: None,
                ffn_post_norm: norm(4),
                layer_output_scale: None,
            }],
            output_norm: Some(norm(4)),
            lm_head: Some(qmatrix(8, 4)),
            is_head: true,
            is_tail: true,
            hidden_dim: 4,
            num_layers: 1,
            vocab_size: 8,
            device: Device::Cpu,
            final_logit_softcapping: 30.0,
        }
    }

    #[test]
    fn cache_only_prefill_preserves_gemma_cache_and_logits() -> Result<()> {
        let mut stage = tiny_stage();
        let weights = |rows, cols| -> Result<Tensor> {
            Ok(Tensor::from_vec(
                (0..rows * cols).map(|i| (i as f32 + 1.) / 100.).collect(),
                (rows, cols),
                &Device::Cpu,
            )?)
        };
        stage.embedding = Some(weights(8, 4)?);
        stage.lm_head = Some(QMatMul::Tensor(weights(8, 4)?));
        stage.blocks[0].attn.k_proj = GemmaQMatMul::Candle(QMatMul::Tensor(weights(4, 4)?));
        stage.blocks[0].attn.v_proj = Some(GemmaQMatMul::Candle(QMatMul::Tensor(weights(4, 4)?)));
        let mut expected = vec![None];
        let mut actual = vec![None];
        for (start, tokens) in [(0, vec![1, 2, 3]), (3, vec![4])] {
            stage.forward_prefill(
                StageInput::Tokens(tokens.clone()),
                &mut expected,
                tokens.len(),
                start,
            )?;
            let output = stage.forward_prefill_cache_only(
                StageInput::Tokens(tokens.clone()),
                &mut actual,
                tokens.len(),
                start,
            )?;
            assert!(matches!(output, StageOutput::Activations(_)));
            let dense = |cache: &[Option<LayerCache>]| -> Result<(Tensor, Tensor)> {
                match &cache[0] {
                    Some(LayerCache::Attention(KvCache::F16 { k, v })) => {
                        Ok((k.clone(), v.clone()))
                    }
                    Some(LayerCache::Attention(KvCache::PreallocatedF16(kv))) => kv.active_f16(),
                    _ => anyhow::bail!("wrong cache"),
                }
            };
            let (ak, av) = dense(&actual)?;
            let (ek, ev) = dense(&expected)?;
            for (a, e) in [(ak, ek), (av, ev)] {
                assert_eq!(
                    a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                    e.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?
                );
            }
        }
        let StageOutput::Logits(expected) =
            stage.forward_prefill(StageInput::Tokens(vec![5]), &mut expected, 1, 4)?
        else {
            panic!("no logits")
        };
        let StageOutput::Logits(actual) =
            stage.forward_prefill(StageInput::Tokens(vec![5]), &mut actual, 1, 4)?
        else {
            panic!("no logits")
        };
        assert_eq!(actual.to_vec1::<f32>()?, expected.to_vec1::<f32>()?);
        Ok(())
    }

    #[test]
    fn local_attention_is_bidirectional_only_inside_image_span() {
        let spans = [AttentionSpan { start: 1, len: 3 }];
        let mask = create_attention_mask(6, 6, 0, Some(4), &spans, &Device::Cpu)
            .unwrap()
            .i((0, 0))
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(mask[1][3], 0.0);
        assert_eq!(mask[3][1], 0.0);
        assert!(mask[0][1].is_infinite() && mask[0][1].is_sign_negative());
        assert!(mask[4][0].is_infinite() && mask[4][0].is_sign_negative());

        let full = create_attention_mask(6, 6, 0, None, &spans, &Device::Cpu)
            .unwrap()
            .i((0, 0))
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert!(full[1][3].is_infinite() && full[1][3].is_sign_negative());
    }

    #[test]
    fn image_attention_spans_must_be_sorted_and_inside_prefill() {
        assert!(validate_attention_spans(&[AttentionSpan { start: 2, len: 3 }], 0, 8).is_ok());
        assert!(
            validate_attention_spans(
                &[
                    AttentionSpan { start: 4, len: 2 },
                    AttentionSpan { start: 3, len: 1 },
                ],
                0,
                8,
            )
            .is_err()
        );
        assert!(validate_attention_spans(&[AttentionSpan { start: 7, len: 2 }], 0, 8).is_err());
    }

    #[test]
    fn ragged_decode_discards_stale_uniform_batch_cache() {
        let stage = tiny_stage();
        let mut first = vec![None];
        let mut second = vec![Some(LayerCache::Attention(cache(1, 1)))];
        let mut caches = vec![first.as_mut_slice(), second.as_mut_slice()];
        let mut batch_cache = vec![Some(LayerCache::Attention(cache(2, 1)))];

        let outputs = stage
            .forward_decode_batch(&[1, 2], &mut caches, &mut batch_cache, &[0, 1])
            .unwrap();

        assert_eq!(outputs.len(), 2);
        assert!(batch_cache[0].is_none());
        assert!(first[0].is_some());
        assert!(second[0].is_some());
    }
}
