use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use crate::qmatmul::QMatMul;
use anyhow::{Context, Result};
use candle_core::quantized::{GgmlDType, QStorage, QTensor};
use candle_core::{D, DType, Device, IndexOp, Module, Tensor};

#[cfg(target_os = "macos")]
use super::DeltaNetReplay;
use super::activation::softplus;
use super::{
    DeltaNetCache, DeltaNetSnapshots, DeltaNetSpeculativeState, KvCache, LayerCache,
    ModelArchitecture, MropePositionIds, SpeculativeSharedWeights, StageInput, StageOutput,
};
use crate::gguf_loader::StageTensors;
use crate::turboquant::TurboQuantConfig;

static DN_NORM_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(target_os = "macos")]
static METAL4_QMM_LOGGED: OnceLock<()> = OnceLock::new();
#[cfg(target_os = "macos")]
static FUSED_Q6K_SWIGLU_LOGGED: OnceLock<()> = OnceLock::new();
#[cfg(target_os = "macos")]
static FUSED_Q6K_FFN_RESIDUAL_LOGGED: OnceLock<()> = OnceLock::new();
#[cfg(target_os = "macos")]
static FUSED_Q8_SSM_RESIDUAL_LOGGED: OnceLock<()> = OnceLock::new();
#[cfg(target_os = "macos")]
static FUSED_SSM_GATE_LOGGED: OnceLock<()> = OnceLock::new();
#[cfg(target_os = "macos")]
static FUSED_DECODE_QKV_LOGGED: OnceLock<()> = OnceLock::new();
#[cfg(target_os = "macos")]
static GROUPED_DELTANET_LOGGED: OnceLock<()> = OnceLock::new();
#[cfg(target_os = "macos")]
static GROUPED_ATTENTION_LOGGED: OnceLock<()> = OnceLock::new();
static DN_PROJECTIONS_NS: AtomicU64 = AtomicU64::new(0);
static DN_SCAN_NS: AtomicU64 = AtomicU64::new(0);
static DN_OUTPUT_NS: AtomicU64 = AtomicU64::new(0);
static DN_FFN_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_NORM_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_PROJECTIONS_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_LAYOUT_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_SDPA_OUTPUT_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_FFN_NS: AtomicU64 = AtomicU64::new(0);
static DN_PROFILE_CALLS: AtomicU64 = AtomicU64::new(0);
static ATTN_PROFILE_CALLS: AtomicU64 = AtomicU64::new(0);
static PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();
static DECODE_BATCH_LOGGED: OnceLock<()> = OnceLock::new();

fn layer_profile_enabled() -> bool {
    *PROFILE_ENABLED.get_or_init(|| std::env::var_os("DINF_PROFILE_LAYERS").is_some())
}

fn finish_profile_region(device: &Device, started: Instant, counter: &AtomicU64) -> Result<()> {
    device.synchronize()?;
    counter.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(())
}

/// Reset the process-local Qwen 3.5 layer profiling counters.
pub fn reset_layer_profile() {
    for counter in [
        &DN_NORM_NS,
        &DN_PROJECTIONS_NS,
        &DN_SCAN_NS,
        &DN_OUTPUT_NS,
        &DN_FFN_NS,
        &ATTN_NORM_NS,
        &ATTN_PROJECTIONS_NS,
        &ATTN_LAYOUT_NS,
        &ATTN_SDPA_OUTPUT_NS,
        &ATTN_FFN_NS,
        &DN_PROFILE_CALLS,
        &ATTN_PROFILE_CALLS,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

/// Print the process-local Qwen 3.5 layer profiling counters to stderr.
pub fn report_layer_profile() {
    let report = |name: &str, calls: &AtomicU64, counters: &[(&str, &AtomicU64)]| {
        let calls_value = calls.load(Ordering::Relaxed);
        let divisor = calls_value.max(1) as f64 * 1_000_000.0;
        let regions = counters
            .iter()
            .map(|(label, counter)| {
                format!(
                    "{label}:{:.4}",
                    counter.load(Ordering::Relaxed) as f64 / divisor
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        eprintln!("DINF_QWEN35_{name}_PROFILE calls={calls_value} per_layer_ms={{{regions}}}");
    };
    report(
        "DELTANET",
        &DN_PROFILE_CALLS,
        &[
            ("norm", &DN_NORM_NS),
            ("projections", &DN_PROJECTIONS_NS),
            ("scan", &DN_SCAN_NS),
            ("output", &DN_OUTPUT_NS),
            ("ffn", &DN_FFN_NS),
        ],
    );
    report(
        "ATTENTION",
        &ATTN_PROFILE_CALLS,
        &[
            ("norm", &ATTN_NORM_NS),
            ("projections", &ATTN_PROJECTIONS_NS),
            ("layout", &ATTN_LAYOUT_NS),
            ("sdpa_output", &ATTN_SDPA_OUTPUT_NS),
            ("ffn", &ATTN_FFN_NS),
        ],
    );
}

// ── Qwen 3.5 architecture ───────────────────────────────────────────────
//
// Hybrid architecture: 75% DeltaNet layers (linear recurrent attention) +
// 25% Gated Attention layers (softmax attention with output gate).
//
// DeltaNet layers use a fixed-size recurrent state that does NOT grow with
// sequence length. Attention layers use standard KV cache.

/// Top-level stage holding all blocks.
pub struct Qwen35Stage {
    embedding: Option<Arc<QTensor>>,
    blocks: Vec<Qwen35Block>,
    output_norm: Option<RmsNorm>,
    lm_head: Option<Qwen35QMatMul>,
    is_head: bool,
    is_tail: bool,
    hidden_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    /// Global layer indices for this stage's blocks (for cache type dispatch).
    #[allow(dead_code)]
    layer_indices: Vec<usize>,
    /// Interval between full attention layers (e.g. 4).
    #[allow(dead_code)]
    full_attention_interval: usize,
    device: Device,
}

enum Qwen35Block {
    DeltaNet(DeltaNetBlock),
    GatedAttention(GatedAttentionBlock),
}

// ── DeltaNet block ───────────────────────────────────────────────────────

struct DeltaNetBlock {
    attn_norm: RmsNorm,
    projections: DeltaNetProjections,
    conv1d_weight: Tensor, // [kernel_size, conv_dim] (stored transposed from GGUF [conv_dim, kernel_size] → we store as-is)
    a_log: Tensor,         // [num_v_heads] log decay rate
    dt_bias: Tensor,       // [num_v_heads] discretization bias
    ssm_norm: RmsNorm,     // per-head RMSNorm, weight=[value_head_dim]
    out_proj: Qwen35QMatMul, // [value_dim, hidden]
    ffn_norm: RmsNorm,
    ffn: Mlp,
    // Cached dimensions
    num_k_heads: usize,    // 16 (group_count)
    num_v_heads: usize,    // 32 (time_step_rank)
    key_head_dim: usize,   // 128 (state_size)
    value_head_dim: usize, // 128
    key_dim: usize,        // num_k_heads * key_head_dim = 2048
    value_dim: usize,      // num_v_heads * value_head_dim = 4096
    conv_dim: usize,       // key_dim*2 + value_dim = 8192
    conv_kernel: usize,    // 4
}

// ── Gated Attention block ────────────────────────────────────────────────

struct GatedAttentionBlock {
    attn_norm: RmsNorm,
    input_projections: AttentionInputProjections,
    o_proj: Qwen35QMatMul, // [hidden, hidden]
    q_norm: RmsNorm,       // per-head, weight=[head_dim]
    k_norm: RmsNorm,       // per-head, weight=[head_dim]
    ffn_norm: RmsNorm,
    ffn: Mlp,
    rope_cos: Tensor, // [context_length, rope_dims/2]
    rope_sin: Tensor,
    mrope_sections: Option<[usize; 3]>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dims: usize, // partial RoPE dimension count (e.g. 64 of 256)
    quant_config: Option<Arc<TurboQuantConfig>>,
}

// ── Shared components ────────────────────────────────────────────────────

struct Mlp {
    input_projections: MlpInputProjections,
    down_proj: Qwen35QMatMul,
}

enum MlpInputProjections {
    Separate {
        gate: Qwen35QMatMul,
        up: Qwen35QMatMul,
    },
    /// Gate and up rows are packed into one quantized matrix. One Metal
    /// dispatch produces the native `[gate, up]` layout consumed directly by
    /// the fused SwiGLU epilogue.
    Grouped {
        projection: Qwen35QMatMul,
        intermediate_dim: usize,
    },
}

struct DeltaNetProjections {
    qkv_gate: HomogeneousProjectionPair,
    alpha_beta: HomogeneousProjectionPair,
}

enum HomogeneousProjectionPair {
    Separate {
        first: Qwen35QMatMul,
        second: Qwen35QMatMul,
    },
    Grouped(Qwen35QMatMul),
}

enum AttentionInputProjections {
    Separate {
        q: Qwen35QMatMul,
        k: Qwen35QMatMul,
        v: Qwen35QMatMul,
    },
    Grouped(Qwen35QMatMul),
}

enum DeltaNetRollback {
    Snapshots(Tensor),
    #[cfg(target_os = "macos")]
    Replay(super::qwen35_deltanet_metal::DeltaNetReplayTerms),
}

#[derive(Clone)]
pub(crate) enum Qwen35QMatMul {
    Candle(QMatMul),
    #[cfg(target_os = "macos")]
    MetalTensor(super::tensor_qmatmul_metal::MetalTensorQMatMul),
}

// Candle's Metal Q4K row gather divides the scale by 16 in F16 before
// widening to F32. Small scales lose bits. Keep this lookup table packed on
// CPU and transfer only the selected F32 rows; never expand the whole table.
// The GGUF loader places it on CPU directly. This copy covers custom stage
// tensors; it briefly holds the caller's Metal buffer and a packed CPU copy.
pub(super) fn prepare_packed_embedding(
    embedding: Arc<QTensor>,
    device: &Device,
) -> Result<Arc<QTensor>> {
    if device.is_metal() && embedding.device().is_metal() && embedding.dtype() == GgmlDType::Q4K {
        return copy_packed_tensor(&embedding, &Device::Cpu);
    }
    Ok(embedding)
}

fn copy_packed_tensor(tensor: &QTensor, device: &Device) -> Result<Arc<QTensor>> {
    let storage = QStorage::from_data(tensor.data()?, device, tensor.dtype())?;
    Ok(Arc::new(QTensor::new(storage, tensor.shape().clone())?))
}

pub(super) fn packed_embedding_rows(
    embedding: &QTensor,
    tokens: &[u32],
    device: &Device,
) -> Result<Tensor> {
    let ids = Tensor::new(tokens, &embedding.device())?;
    Ok(embedding.embedding(&ids)?.to_device(device)?)
}

// A tied output head still runs on Metal. Its matrix path may upload the CPU
// packed table directly; Candle's fallback needs a device QTensor instead.
fn qmatmul_source_on_device(tensor: Arc<QTensor>, device: &Device) -> Result<Arc<QTensor>> {
    if device.is_metal() && tensor.device().is_cpu() && tensor.dtype() == GgmlDType::Q4K {
        return copy_packed_tensor(&tensor, device);
    }
    Ok(tensor)
}

impl Qwen35QMatMul {
    fn from_qtensor(qt: Arc<QTensor>, _device: &Device) -> Result<Self> {
        #[cfg(target_os = "macos")]
        if std::env::var_os("DINF_DISABLE_METAL4_QMM").is_none()
            && let Some(matmul) = super::tensor_qmatmul_metal::MetalTensorQMatMul::try_from_qtensor(
                Arc::clone(&qt),
                _device,
            )?
        {
            let profile = matmul.profile_label();
            METAL4_QMM_LOGGED.get_or_init(|| {
                tracing::info!(
                    profile,
                    "Metal 4 tensor quantized matmul enabled for Qwen3.5"
                );
            });
            return Ok(Self::MetalTensor(matmul));
        }

        Ok(Self::Candle(qtensor_to_qmatmul(qmatmul_source_on_device(
            qt, _device,
        )?)?))
    }

    #[cfg(target_os = "macos")]
    fn try_grouped(qts: &[Arc<QTensor>], device: &Device) -> Result<Option<Self>> {
        if std::env::var_os("DINF_DISABLE_METAL4_QMM").is_some() {
            return Ok(None);
        }
        Ok(
            super::tensor_qmatmul_metal::MetalTensorQMatMul::try_from_qtensors(qts, device)?
                .map(Self::MetalTensor),
        )
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Candle(matmul) => matmul.forward(x).map_err(Into::into),
            #[cfg(target_os = "macos")]
            Self::MetalTensor(matmul) => {
                let tokens = x.dim(D::Minus2)?;
                let preferred_tile = if speculative_batch::active() {
                    Some(8)
                } else {
                    (65..=192).contains(&tokens).then_some(64)
                };
                if x.rank() == 2 && tokens >= 3 {
                    matmul
                        .forward_with_tile(&x.unsqueeze(0)?, preferred_tile)?
                        .squeeze(0)
                        .map_err(Into::into)
                } else {
                    matmul
                        .forward_with_tile(x, preferred_tile)
                        .map_err(Into::into)
                }
            }
        }
    }

    fn forward_decode_rows(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            #[cfg(target_os = "macos")]
            Self::MetalTensor(matmul) => matmul.forward_decode_rows(x).map_err(Into::into),
            Self::Candle(matmul) => matmul.forward(x).map_err(Into::into),
        }
    }

    fn has_metal_decode_rows(&self) -> bool {
        #[cfg(target_os = "macos")]
        if matches!(self, Self::MetalTensor(_)) {
            return true;
        }
        false
    }

    fn forward_single_token_groups(&self, _x: &Tensor) -> Result<Option<Vec<Tensor>>> {
        match self {
            #[cfg(target_os = "macos")]
            Self::MetalTensor(matmul) => matmul.forward_single_token_groups(_x).map_err(Into::into),
            Self::Candle(_) => Ok(None),
        }
    }

    fn forward_swiglu(&self, _x: &Tensor) -> Result<Option<Tensor>> {
        match self {
            #[cfg(target_os = "macos")]
            Self::MetalTensor(matmul) => matmul.forward_swiglu(_x).map_err(Into::into),
            Self::Candle(_) => Ok(None),
        }
    }

    fn forward_residual(&self, _x: &Tensor, _residual: &Tensor) -> Result<Option<Tensor>> {
        match self {
            #[cfg(target_os = "macos")]
            Self::MetalTensor(matmul) => matmul.forward_residual(_x, _residual).map_err(Into::into),
            Self::Candle(_) => Ok(None),
        }
    }

    fn forward_qwen35_ssm_residual(
        &self,
        _x: &Tensor,
        _residual: &Tensor,
    ) -> Result<Option<Tensor>> {
        match self {
            #[cfg(target_os = "macos")]
            Self::MetalTensor(matmul) => matmul
                .forward_qwen35_ssm_residual(_x, _residual)
                .map_err(Into::into),
            Self::Candle(_) => Ok(None),
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

// ── Primitive functions ──────────────────────────────────────────────────

/// L2-normalize along the last dimension.
fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let sq_sum = (x * x)?.sum_keepdim(D::Minus1)?;
    let norm = (sq_sum + 1e-12)?.sqrt()?;
    Ok(x.broadcast_div(&norm)?)
}

fn silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    #[cfg(target_os = "macos")]
    if std::env::var_os("DINF_DISABLE_FUSED_SILU_MUL").is_none()
        && gate.device().is_metal()
        && gate.dtype() == DType::F32
        && up.dtype() == DType::F32
        && gate.is_contiguous()
        && up.is_contiguous()
        && gate.shape() == up.shape()
    {
        return super::indexed_moe_metal::fused_silu_mul(gate, up).map_err(Into::into);
    }
    let activated = candle_nn::Activation::Silu.forward(gate)?;
    (&activated * up).map_err(Into::into)
}

/// Causal depthwise conv1d step: update sliding window state and produce output.
///
/// state: [conv_dim, kernel_size], input: [conv_dim]
/// weight: [kernel_size, conv_dim] from GGUF
///
/// Returns (output [conv_dim], updated state).
fn conv1d_step(state: &Tensor, input: &Tensor, weight: &Tensor) -> Result<(Tensor, Tensor)> {
    let (_conv_dim, kernel_size) = state.dims2()?;

    // Shift state left and insert new input on the right
    let shifted = if kernel_size > 1 {
        let old = state.narrow(1, 1, kernel_size - 1)?;
        Tensor::cat(&[&old, &input.unsqueeze(1)?], 1)?
    } else {
        input.unsqueeze(1)?
    };

    // Depthwise convolution: element-wise multiply and sum across kernel dim
    // weight: [conv_dim, kernel_size] — already matches state layout
    let output = (&shifted * weight)?.sum(1)?; // [conv_dim]

    Ok((output, shifted))
}

fn deltanet_conv_snapshots(
    initial_state: &Tensor,
    inputs: &Tensor,
    kernel_size: usize,
) -> Result<Tensor> {
    let (seq_len, _) = inputs.dims2()?;
    let history = Tensor::cat(&[initial_state, &inputs.transpose(0, 1)?], 1)?;
    let snapshots = (0..seq_len)
        .map(|row| history.narrow(1, row + 1, kernel_size)?.unsqueeze(0))
        .collect::<candle_core::Result<Vec<_>>>()?;
    Tensor::cat(&snapshots, 0).map_err(Into::into)
}

/// Single-head, single-token Gated DeltaNet recurrence step.
///
/// Uses the delta rule with error correction (ICLR 2025):
///   1. Decay: S_dec = exp(g) * S
///   2. Retrieve: retrieved = S_dec^T @ k
///   3. Delta: delta = beta * (v - retrieved)
///   4. Update: S' = S_dec + k ⊗ delta
///   5. Output: o = S'^T @ q
///
/// S: [key_head_dim, value_head_dim] — recurrent state
/// q, k: [key_head_dim], v: [value_head_dim]
/// beta: scalar (update gate), g: scalar (decay exponent, negative)
///
/// Returns (output [value_head_dim], new_S).
fn deltanet_step(
    s: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: f32,
    g: f32,
) -> Result<(Tensor, Tensor)> {
    let decay = g.exp();

    // Step 1: Decay state
    let decayed_s = (s * decay as f64)?;

    // Step 2: Retrieve predicted value from decayed state
    let k_col = k.unsqueeze(1)?; // [key_head_dim, 1]
    let retrieved = decayed_s.t()?.matmul(&k_col)?.squeeze(1)?; // [value_head_dim]

    // Step 3: Delta rule — error correction
    let delta = ((v - &retrieved)? * beta as f64)?; // [value_head_dim]

    // Step 4: Update state with outer product of k and delta
    let delta_row = delta.unsqueeze(0)?; // [1, value_head_dim]
    let update = k_col.matmul(&delta_row)?; // [key_head_dim, value_head_dim]
    let new_s = decayed_s.add(&update)?;

    // Step 5: Output from updated state
    let q_col = q.unsqueeze(1)?; // [key_head_dim, 1]
    let output = new_s.t()?.matmul(&q_col)?.squeeze(1)?; // [value_head_dim]

    Ok((output, new_s))
}

/// DeltaNet forward for all heads, single token.
///
/// q: [num_v_heads, key_head_dim] (after GQA expansion)
/// k: [num_v_heads, key_head_dim] (after GQA expansion)
/// v: [num_v_heads, value_head_dim]
/// beta: [num_v_heads] (update gates)
/// g: [num_v_heads] (decay exponents)
/// state: [num_v_heads, key_head_dim, value_head_dim]
///
/// Returns (output [num_v_heads, value_head_dim], new_state).
fn deltanet_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &[f32],
    g: &[f32],
    state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let num_heads = q.dim(0)?;

    let mut outputs = Vec::with_capacity(num_heads);
    let mut new_states = Vec::with_capacity(num_heads);

    for h in 0..num_heads {
        let q_h = q.get(h)?;
        let k_h = k.get(h)?;
        let v_h = v.get(h)?;
        let s_h = state.get(h)?;

        let (out_h, new_s_h) = deltanet_step(&s_h, &q_h, &k_h, &v_h, beta[h], g[h])?;
        outputs.push(out_h.unsqueeze(0)?);
        new_states.push(new_s_h.unsqueeze(0)?);
    }

    let output = Tensor::cat(&outputs, 0)?; // [num_v_heads, value_head_dim]
    let new_state = Tensor::cat(&new_states, 0)?; // [num_v_heads, key_head_dim, value_head_dim]

    Ok((output, new_state))
}

/// Apply partial rotary position embedding.
///
/// x: [batch, heads, seq_len, head_dim]
/// Only the first `rope_dims` dimensions get RoPE; the rest pass through unchanged.
#[cfg(test)]
fn apply_partial_rotary_emb(
    x: &Tensor,
    start_pos: usize,
    cos: &Tensor,
    sin: &Tensor,
    rope_dims: usize,
) -> Result<Tensor> {
    apply_partial_rotary_emb_with_positions(x, start_pos, cos, sin, rope_dims, None, None)
}

fn apply_partial_rotary_emb_with_positions(
    x: &Tensor,
    start_pos: usize,
    cos: &Tensor,
    sin: &Tensor,
    rope_dims: usize,
    positions: Option<&MropePositionIds>,
    mrope_sections: Option<[usize; 3]>,
) -> Result<Tensor> {
    let (_, _, seq_len, head_dim) = x.dims4()?;
    anyhow::ensure!(
        rope_dims > 0 && rope_dims <= head_dim && rope_dims.is_multiple_of(2),
        "invalid Qwen3.5 rotary dimension {rope_dims} for head dimension {head_dim}"
    );

    let (cos, sin) = if let Some(positions) = positions {
        anyhow::ensure!(
            positions.len() == seq_len,
            "M-RoPE schedule has {} positions for a sequence of {seq_len} tokens",
            positions.len()
        );
        let sections = mrope_sections
            .context("Qwen multimodal input requires GGUF rope.dimension_sections metadata")?;
        (
            select_mrope_table(cos, positions, sections)?,
            select_mrope_table(sin, positions, sections)?,
        )
    } else {
        (
            cos.narrow(0, start_pos, seq_len)?,
            sin.narrow(0, start_pos, seq_len)?,
        )
    };

    if rope_dims >= head_dim {
        // Full RoPE (interleaved for Qwen 3.5)
        return candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin).map_err(Into::into);
    }

    // Split into RoPE dims and passthrough dims
    let x_rope = x.narrow(3, 0, rope_dims)?;
    let x_pass = x.narrow(3, rope_dims, head_dim - rope_dims)?;

    let x_rope = candle_nn::rotary_emb::rope(&x_rope.contiguous()?, &cos, &sin)?;

    Tensor::cat(&[&x_rope, &x_pass.contiguous()?], 3).map_err(Into::into)
}

/// Return the scalar text position supported by the decode-preparation
/// fusion. Any explicit M-RoPE schedule stays on the axis-aware path, even
/// when it happens to contain a single token.
#[cfg(any(target_os = "macos", test))]
fn fused_decode_text_position(
    seq_len: usize,
    start_pos: usize,
    positions: Option<&MropePositionIds>,
) -> Option<usize> {
    (seq_len == 1 && positions.is_none()).then_some(start_pos)
}

fn parse_mrope_sections(raw: &[usize], rope_dims: usize) -> Result<Option<[usize; 3]>> {
    if raw.is_empty() {
        return Ok(None);
    }
    anyhow::ensure!(
        raw.len() >= 3,
        "Qwen M-RoPE metadata must contain temporal, height, and width sections"
    );
    anyhow::ensure!(
        raw.iter().skip(3).all(|section| *section == 0),
        "unsupported non-zero fourth Qwen M-RoPE section"
    );
    anyhow::ensure!(
        rope_dims.is_multiple_of(2),
        "Qwen M-RoPE dimension must be even"
    );
    let sections = [raw[0], raw[1], raw[2]];
    anyhow::ensure!(
        sections.iter().sum::<usize>() == rope_dims / 2,
        "Qwen M-RoPE sections {:?} do not cover {} rotary frequencies",
        sections,
        rope_dims / 2
    );
    Ok(Some(sections))
}

/// Select the temporal/height/width row for every interleaved rotary frequency.
/// This matches Transformers' `apply_interleaved_mrope`: temporal is the base,
/// then indices 1,4,7,... come from height and 2,5,8,... from width.
fn select_mrope_table(
    table: &Tensor,
    positions: &MropePositionIds,
    sections: [usize; 3],
) -> Result<Tensor> {
    let (table_len, half_dim) = table.dims2()?;
    anyhow::ensure!(
        sections.iter().sum::<usize>() == half_dim,
        "M-RoPE sections {:?} do not match table width {half_dim}",
        sections
    );

    let mut frequency_axes = vec![0usize; half_dim];
    for index in (1..sections[1].saturating_mul(3)).step_by(3) {
        anyhow::ensure!(
            index < half_dim,
            "height M-RoPE section exceeds table width"
        );
        frequency_axes[index] = 1;
    }
    for index in (2..sections[2].saturating_mul(3)).step_by(3) {
        anyhow::ensure!(index < half_dim, "width M-RoPE section exceeds table width");
        frequency_axes[index] = 2;
    }
    let counts = [0usize, 1, 2].map(|axis| {
        frequency_axes
            .iter()
            .filter(|selected| **selected == axis)
            .count()
    });
    anyhow::ensure!(
        counts == sections,
        "M-RoPE sections {:?} are incompatible with interleaved table width {half_dim}",
        sections
    );

    let axes = positions.axes();
    let mut selected: Option<Tensor> = None;
    for (axis, axis_positions) in axes.iter().enumerate() {
        if let Some(max_position) = axis_positions.iter().copied().max() {
            anyhow::ensure!(
                (max_position as usize) < table_len,
                "M-RoPE position {max_position} exceeds table length {table_len}"
            );
        }
        let indices = Tensor::new(*axis_positions, table.device())?;
        let values = table.index_select(&indices, 0)?;
        let mask = frequency_axes
            .iter()
            .map(|selected_axis| u8::from(*selected_axis == axis) as f32)
            .collect::<Vec<_>>();
        let mask =
            Tensor::from_vec(mask, (1, half_dim), table.device())?.to_dtype(table.dtype())?;
        let values = values.broadcast_mul(&mask)?;
        selected = Some(match selected {
            Some(current) => current.broadcast_add(&values)?,
            None => values,
        });
    }
    selected.context("M-RoPE table selection produced no axes")
}

// ── Component implementations ────────────────────────────────────────────

impl RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        #[cfg(target_os = "macos")]
        if let Some(result) =
            super::fused_norm_metal::try_fused_rms_norm(&x_f32, &self.weight, self.eps as f32)?
        {
            return result.to_dtype(dtype).map_err(Into::into);
        }
        let variance = (&x_f32 * &x_f32)?.mean_keepdim(D::Minus1)?;
        let x_normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let result = x_normed.to_dtype(dtype)?.broadcast_mul(&self.weight)?;
        Ok(result)
    }

    fn forward_residual(&self, residual: &Tensor, update: &Tensor) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        if let Some(result) =
            super::residual_norm_cuda::forward(residual, update, &self.weight, self.eps as f32)?
        {
            return Ok(result);
        }
        #[cfg(target_os = "macos")]
        if let Some(result) = super::fused_norm_metal::try_fused_add_rms_norm(
            residual,
            update,
            &self.weight,
            self.eps as f32,
        )? {
            return Ok(result);
        }
        let residual = (residual + update)?;
        let normalized = self.forward(&residual)?;
        Ok((residual, normalized))
    }
}

impl Mlp {
    fn has_metal_decode_rows(&self) -> bool {
        self.down_proj.has_metal_decode_rows()
            && match &self.input_projections {
                MlpInputProjections::Separate { gate, up } => {
                    gate.has_metal_decode_rows() && up.has_metal_decode_rows()
                }
                MlpInputProjections::Grouped { projection, .. } => {
                    projection.has_metal_decode_rows()
                }
            }
    }

    fn forward_decode_rows(&self, x: &Tensor, residual: &Tensor) -> Result<Tensor> {
        let activated = match &self.input_projections {
            MlpInputProjections::Separate { gate, up } => {
                silu_mul(&gate.forward_decode_rows(x)?, &up.forward_decode_rows(x)?)?
            }
            MlpInputProjections::Grouped {
                projection,
                intermediate_dim,
            } => {
                let all = projection.forward_decode_rows(x)?;
                #[cfg(target_os = "macos")]
                if x.device().is_metal()
                    && std::env::var_os("DINF_DISABLE_FUSED_SILU_MUL").is_none()
                {
                    super::indexed_moe_metal::fused_swiglu(&all)?
                } else {
                    silu_mul(
                        &all.narrow(2, 0, *intermediate_dim)?,
                        &all.narrow(2, *intermediate_dim, *intermediate_dim)?,
                    )?
                }
                #[cfg(not(target_os = "macos"))]
                silu_mul(
                    &all.narrow(2, 0, *intermediate_dim)?,
                    &all.narrow(2, *intermediate_dim, *intermediate_dim)?,
                )?
            }
        };
        (residual + self.down_proj.forward_decode_rows(&activated)?).map_err(Into::into)
    }

    fn grouped_activation(&self, x: &Tensor) -> Result<Option<Tensor>> {
        if let MlpInputProjections::Grouped {
            projection,
            intermediate_dim,
        } = &self.input_projections
        {
            if let Some(activated) = projection.forward_swiglu(x)? {
                #[cfg(target_os = "macos")]
                FUSED_Q6K_SWIGLU_LOGGED.get_or_init(|| {
                    tracing::info!("fused pre-M5 Q6_K SwiGLU enabled for Qwen3.5");
                });
                return Ok(Some(activated));
            }
            let gate_up = projection.forward(x)?;
            anyhow::ensure!(
                gate_up.dim(D::Minus1)? == intermediate_dim * 2,
                "grouped Qwen3.5 MLP output width changed"
            );
            #[cfg(target_os = "macos")]
            let activated = if std::env::var_os("DINF_DISABLE_FUSED_SILU_MUL").is_none() {
                super::indexed_moe_metal::fused_swiglu(&gate_up)?
            } else {
                let gate = gate_up.narrow(D::Minus1, 0, *intermediate_dim)?;
                let up = gate_up.narrow(D::Minus1, *intermediate_dim, *intermediate_dim)?;
                silu_mul(&gate, &up)?
            };
            #[cfg(not(target_os = "macos"))]
            let activated = {
                let gate = gate_up.narrow(D::Minus1, 0, *intermediate_dim)?;
                let up = gate_up.narrow(D::Minus1, *intermediate_dim, *intermediate_dim)?;
                silu_mul(&gate, &up)?
            };
            return Ok(Some(activated));
        }
        Ok(None)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(activated) = self.grouped_activation(x)? {
            return self.down_proj.forward(&activated);
        }
        let MlpInputProjections::Separate {
            gate: gate_proj,
            up: up_proj,
        } = &self.input_projections
        else {
            unreachable!()
        };
        #[cfg(feature = "cuda")]
        let grouped = match (gate_proj.candle(), up_proj.candle()) {
            (Some(gate), Some(up)) => super::dense_qmatmul_cuda::forward(x, &[gate, up])?,
            _ => None,
        };
        #[cfg(not(feature = "cuda"))]
        let grouped: Option<Vec<Tensor>> = None;
        let (gate, up) = if let Some(mut outputs) = grouped {
            anyhow::ensure!(outputs.len() == 2, "grouped Qwen3.5 MLP count mismatch");
            let up = outputs.pop().unwrap();
            let gate = outputs.pop().unwrap();
            (gate, up)
        } else {
            (gate_proj.forward(x)?, up_proj.forward(x)?)
        };
        #[cfg(feature = "cuda")]
        if let Some(down) = self.down_proj.candle()
            && let Some(output) = super::dense_qmatmul_cuda::silu_mul_down(&gate, &up, down)?
        {
            return Ok(output);
        }
        let activated = silu_mul(&gate, &up)?;
        self.down_proj.forward(&activated)
    }

    fn forward_residual(&self, x: &Tensor, residual: &Tensor) -> Result<Tensor> {
        if let Some(activated) = self.grouped_activation(x)? {
            if let Some(output) = self.down_proj.forward_residual(&activated, residual)? {
                #[cfg(target_os = "macos")]
                FUSED_Q6K_FFN_RESIDUAL_LOGGED.get_or_init(|| {
                    tracing::info!(
                        "fused pre-M5 Q6_K FFN down projection/residual enabled for Qwen3.5"
                    );
                });
                return Ok(output);
            }
            let projected = self.down_proj.forward(&activated)?;
            return (residual + projected).map_err(Into::into);
        }
        let projected = self.forward(x)?;
        (residual + projected).map_err(Into::into)
    }
}

impl HomogeneousProjectionPair {
    fn has_metal_decode_rows(&self) -> bool {
        match self {
            Self::Separate { first, second } => {
                first.has_metal_decode_rows() && second.has_metal_decode_rows()
            }
            Self::Grouped(projection) => projection.has_metal_decode_rows(),
        }
    }

    fn forward_decode_rows(&self, x: &Tensor, first_width: usize) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Separate { first, second } => Ok((
                first.forward_decode_rows(x)?,
                second.forward_decode_rows(x)?,
            )),
            Self::Grouped(projection) => {
                let all = projection.forward_decode_rows(x)?;
                Ok((
                    all.narrow(2, 0, first_width)?.contiguous()?,
                    all.narrow(2, first_width, all.dim(2)? - first_width)?
                        .contiguous()?,
                ))
            }
        }
    }

    fn forward(
        &self,
        x: &Tensor,
        seq_len: usize,
        first_width: usize,
        second_width: usize,
        label: &str,
    ) -> Result<(Tensor, Tensor)> {
        if let Self::Grouped(projection) = self {
            if seq_len == 1 {
                if let Some(mut outputs) = projection.forward_single_token_groups(x)? {
                    anyhow::ensure!(
                        outputs.len() == 2,
                        "grouped Qwen {label} output count changed"
                    );
                    let second = outputs.pop().unwrap();
                    let first = outputs.pop().unwrap();
                    return Ok((first, second));
                }
                tracing::debug!(
                    label,
                    "zero-copy grouped Qwen decode split unavailable; using exact copy fallback"
                );
            }

            let all = projection.forward(x)?;
            anyhow::ensure!(
                all.dim(D::Minus1)? == first_width + second_width,
                "grouped Qwen {label} projection width changed"
            );
            let first = all.narrow(2, 0, first_width)?.contiguous()?;
            let second = all.narrow(2, first_width, second_width)?.contiguous()?;
            return Ok((first, second));
        }

        let Self::Separate { first, second } = self else {
            unreachable!()
        };
        Ok((first.forward(x)?, second.forward(x)?))
    }
}

impl DeltaNetProjections {
    fn forward(
        &self,
        x: &Tensor,
        seq_len: usize,
        _profile: bool,
        conv_dim: usize,
        value_dim: usize,
        gate_dim: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        if matches!(self.qkv_gate, HomogeneousProjectionPair::Grouped(_))
            || matches!(self.alpha_beta, HomogeneousProjectionPair::Grouped(_))
        {
            #[cfg(target_os = "macos")]
            GROUPED_DELTANET_LOGGED.get_or_init(|| {
                tracing::info!("homogeneous grouped DeltaNet projections enabled for Qwen3.5");
            });
            let (qkv, gate) =
                self.qkv_gate
                    .forward(x, seq_len, conv_dim, value_dim, "DeltaNet QKV/gate")?;
            let (alpha, beta) =
                self.alpha_beta
                    .forward(x, seq_len, gate_dim, gate_dim, "DeltaNet alpha/beta")?;
            return Ok((
                qkv.squeeze(0)?.to_dtype(DType::F32)?,
                gate.to_dtype(DType::F32)?,
                alpha.squeeze(0)?.to_dtype(DType::F32)?,
                beta.squeeze(0)?.to_dtype(DType::F32)?,
            ));
        }
        let (
            HomogeneousProjectionPair::Separate {
                first: qkv,
                second: gate,
            },
            HomogeneousProjectionPair::Separate {
                first: alpha,
                second: beta,
            },
        ) = (&self.qkv_gate, &self.alpha_beta)
        else {
            unreachable!()
        };
        #[cfg(feature = "cuda")]
        let grouped = if seq_len == 1 && !_profile {
            match (qkv.candle(), gate.candle(), alpha.candle(), beta.candle()) {
                (Some(qkv), Some(gate), Some(alpha), Some(beta)) => {
                    super::dense_qmatmul_cuda::forward(x, &[qkv, gate, alpha, beta])?
                }
                _ => None,
            }
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let grouped: Option<Vec<Tensor>> = None;
        if let Some(mut outputs) = grouped {
            anyhow::ensure!(
                outputs.len() == 4,
                "grouped Qwen3.5 DeltaNet projection count mismatch"
            );
            let beta = outputs.pop().unwrap();
            let alpha = outputs.pop().unwrap();
            let gate = outputs.pop().unwrap();
            let qkv = outputs.pop().unwrap();
            Ok((qkv.squeeze(0)?, gate, alpha.squeeze(0)?, beta.squeeze(0)?))
        } else {
            Ok((
                qkv.forward(x)?.squeeze(0)?.to_dtype(DType::F32)?,
                gate.forward(x)?.to_dtype(DType::F32)?,
                alpha.forward(x)?.squeeze(0)?.to_dtype(DType::F32)?,
                beta.forward(x)?.squeeze(0)?.to_dtype(DType::F32)?,
            ))
        }
    }
}

impl AttentionInputProjections {
    fn has_metal_decode_rows(&self) -> bool {
        match self {
            Self::Separate { q, k, v } => {
                q.has_metal_decode_rows() && k.has_metal_decode_rows() && v.has_metal_decode_rows()
            }
            Self::Grouped(projection) => projection.has_metal_decode_rows(),
        }
    }

    fn forward(
        &self,
        x: &Tensor,
        seq_len: usize,
        _profile: bool,
        q_width: usize,
        kv_width: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        if let Self::Grouped(projection) = self {
            #[cfg(target_os = "macos")]
            GROUPED_ATTENTION_LOGGED.get_or_init(|| {
                tracing::info!("grouped Q/K/V projections enabled for Qwen3.5 attention");
            });
            if seq_len == 1 {
                if let Some(mut outputs) = projection.forward_single_token_groups(x)? {
                    anyhow::ensure!(
                        outputs.len() == 3,
                        "grouped Qwen attention output count changed"
                    );
                    let v = outputs.pop().unwrap();
                    let k = outputs.pop().unwrap();
                    let q = outputs.pop().unwrap();
                    return Ok((q, k, v));
                }
                tracing::debug!(
                    "zero-copy grouped Qwen attention decode split unavailable; using exact copy fallback"
                );
            }

            // Multi-token grouped output is token-major. Materialize the
            // three projection planes to preserve the existing contiguous
            // prefill layout consumed by the attention preparation path.
            let all = projection.forward(x)?;
            anyhow::ensure!(
                all.dim(D::Minus1)? == q_width + kv_width * 2,
                "grouped Qwen attention projection width changed"
            );
            let q = all.narrow(2, 0, q_width)?.contiguous()?;
            let k = all.narrow(2, q_width, kv_width)?.contiguous()?;
            let v = all.narrow(2, q_width + kv_width, kv_width)?.contiguous()?;
            return Ok((q, k, v));
        }

        let Self::Separate { q, k, v } = self else {
            unreachable!()
        };
        #[cfg(feature = "cuda")]
        let grouped = if seq_len == 1 && !_profile {
            match (q.candle(), k.candle(), v.candle()) {
                (Some(q), Some(k), Some(v)) => super::dense_qmatmul_cuda::forward(x, &[q, k, v])?,
                _ => None,
            }
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let grouped: Option<Vec<Tensor>> = None;
        if let Some(mut outputs) = grouped {
            anyhow::ensure!(
                outputs.len() == 3,
                "grouped Qwen3.5 attention count mismatch"
            );
            let v = outputs.pop().unwrap();
            let k = outputs.pop().unwrap();
            let q = outputs.pop().unwrap();
            Ok((q, k, v))
        } else {
            Ok((q.forward(x)?, k.forward(x)?, v.forward(x)?))
        }
    }
}

// ── DeltaNet block forward ───────────────────────────────────────────────

impl DeltaNetBlock {
    fn forward(
        &self,
        x: &Tensor,
        cache: &mut Option<DeltaNetCache>,
        _seq_len: usize,
        _start_pos: usize,
        keep_speculative_snapshots: bool,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _hidden_dim) = x.dims3()?;
        anyhow::ensure!(
            b_sz == 1,
            "Qwen3.5 DeltaNet currently requires batch size 1"
        );
        let profile = layer_profile_enabled();
        if profile {
            DN_PROFILE_CALLS.fetch_add(1, Ordering::Relaxed);
        }

        // These projections depend only on this layer's input, so run them as
        // prompt-sized GEMMs before the recurrent scan rather than as GEMVs.
        let started = profile.then(Instant::now);
        let normed = self.attn_norm.forward(x)?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &DN_NORM_NS)?;
        }
        let started = profile.then(Instant::now);
        let (qkv_all, gate_all, alpha_all, beta_all) = self.projections.forward(
            &normed,
            seq_len,
            profile,
            self.conv_dim,
            self.value_dim,
            self.num_v_heads,
        )?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &DN_PROJECTIONS_NS)?;
        }
        let gated = self.scan_and_gate(
            (qkv_all, gate_all, alpha_all, beta_all),
            cache,
            keep_speculative_snapshots,
        )?;
        self.output_and_ffn(x, &gated, false)
    }

    /// Only the scan reads or advances a request's recurrent state. Decode
    /// batches share projections, but call this once per independent request.
    fn scan_and_gate(
        &self,
        projections: (Tensor, Tensor, Tensor, Tensor),
        cache: &mut Option<DeltaNetCache>,
        keep_speculative_snapshots: bool,
    ) -> Result<Tensor> {
        let (qkv_all, gate_all, alpha_all, beta_all) = projections;
        let seq_len = qkv_all.dim(0)?;
        let device = qkv_all.device();
        let profile = layer_profile_enabled();
        #[cfg(target_os = "macos")]
        let metal_scan_enabled = if seq_len == 1 {
            std::env::var_os("DINF_DISABLE_QWEN35_FUSED_DECODE_METAL").is_none()
        } else {
            std::env::var_os("DINF_DISABLE_QWEN35_PREFILL_METAL").is_none()
        };
        let started = profile.then(Instant::now);
        let (conv_state, rec_state) = match cache.take() {
            Some(c) => (c.conv_state, c.recurrent_state),
            None => (
                Tensor::zeros((self.conv_dim, self.conv_kernel), DType::F32, device)?,
                Tensor::zeros(
                    (self.num_v_heads, self.key_head_dim, self.value_head_dim),
                    DType::F32,
                    device,
                )?,
            ),
        };
        let initial_conv_state = conv_state.clone();
        #[cfg(target_os = "macos")]
        let initial_rec_state = rec_state.clone();
        #[cfg(feature = "cuda")]
        let cuda_prefill = if !keep_speculative_snapshots
            && device.is_cuda()
            && std::env::var_os("DINF_DISABLE_QWEN35_DELTANET_CUDA").is_none()
            && (seq_len == 1 || std::env::var_os("DINF_DISABLE_QWEN35_PREFILL_CUDA").is_none())
            && super::qwen35_deltanet_cuda::supports(
                seq_len,
                self.conv_dim,
                self.conv_kernel,
                self.num_k_heads,
                self.num_v_heads,
                self.key_head_dim,
                self.value_head_dim,
            ) {
            let (output, conv_state, recurrent_state) = super::qwen35_deltanet_cuda::prefill(
                &qkv_all.contiguous()?,
                &conv_state,
                &rec_state,
                &self.conv1d_weight,
                &alpha_all.contiguous()?,
                &beta_all.contiguous()?,
                &self.a_log,
                &self.dt_bias,
                self.num_k_heads,
                self.num_v_heads,
                self.key_head_dim,
                self.value_head_dim,
                self.conv_kernel,
            )?;
            Some((output, conv_state, recurrent_state, None, false))
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let cuda_prefill: Option<(Tensor, Tensor, Tensor, Option<DeltaNetRollback>, bool)> = None;

        #[cfg(target_os = "macos")]
        let metal_prefill = if cuda_prefill.is_none()
            && device.is_metal()
            && self.conv_kernel <= 8
            && self.key_head_dim <= 256
            && std::env::var_os("DINF_DISABLE_QWEN35_DELTANET_METAL").is_none()
            && metal_scan_enabled
            && (!keep_speculative_snapshots || seq_len > 1)
        {
            let fuse_scan_gate = seq_len == 1
                && !keep_speculative_snapshots
                && self.key_head_dim == 128
                && self.value_head_dim == 128
                && gate_all.dtype() == DType::F32
                && gate_all.is_contiguous()
                && gate_all.dims() == [1, 1, self.value_dim]
                && self.ssm_norm.weight.dtype() == DType::F32
                && self.ssm_norm.weight.is_contiguous()
                && self.ssm_norm.weight.dims() == [self.value_head_dim]
                && (self.ssm_norm.eps as f32).is_finite()
                && self.ssm_norm.eps >= 0.0
                && std::env::var_os("DINF_DISABLE_QWEN35_FUSED_DELTANET_EPILOGUE").is_none()
                && std::env::var_os("DINF_DISABLE_QWEN35_FUSED_SSM_GATE").is_none()
                && std::env::var_os("DINF_DISABLE_FUSED_RMSNORM").is_none()
                && std::env::var_os("DINF_DISABLE_FUSED_HEAD_RMSNORM").is_none();
            let scan = super::qwen35_deltanet_metal::prefill(
                &qkv_all.contiguous()?,
                &conv_state,
                &rec_state,
                &self.conv1d_weight,
                &alpha_all.contiguous()?,
                &beta_all.contiguous()?,
                &self.a_log,
                &self.dt_bias,
                self.num_k_heads,
                self.num_v_heads,
                self.key_head_dim,
                self.value_head_dim,
                self.conv_kernel,
                keep_speculative_snapshots,
                &gate_all,
                &self.ssm_norm.weight,
                self.ssm_norm.eps as f32,
                fuse_scan_gate,
            )?;
            let (output, conv_state, recurrent_state, replay, scan_output_gated) = scan;
            Some((
                output,
                conv_state,
                recurrent_state,
                replay.map(DeltaNetRollback::Replay),
                scan_output_gated,
            ))
        } else {
            None
        };
        #[cfg(not(target_os = "macos"))]
        let metal_prefill: Option<(
            Tensor,
            Tensor,
            Tensor,
            Option<DeltaNetRollback>,
            bool,
        )> = None;

        let accelerated_prefill = cuda_prefill.or(metal_prefill);
        let (recurrent_output, conv_state, rec_state, rollback_state, scan_output_gated) =
            if let Some(result) = accelerated_prefill {
                result
            } else {
                let mut conv_state = conv_state;
                let mut rec_state = rec_state;
                let mut recurrent_outputs = Vec::with_capacity(seq_len);
                let mut recurrent_snapshots =
                    keep_speculative_snapshots.then(|| Vec::with_capacity(seq_len));
                for t in 0..seq_len {
                    let qkv = qkv_all.get(t)?.contiguous()?;
                    let alpha_in = alpha_all.get(t)?.contiguous()?;
                    let beta_in = beta_all.get(t)?.contiguous()?;

                    let (conv_out, new_conv_state) =
                        conv1d_step(&conv_state, &qkv, &self.conv1d_weight)?;
                    conv_state = new_conv_state;
                    let conv_out = candle_nn::Activation::Silu.forward(&conv_out)?;
                    let q_raw = conv_out.narrow(0, 0, self.key_dim)?;
                    let k_raw = conv_out.narrow(0, self.key_dim, self.key_dim)?;
                    let v_raw = conv_out.narrow(0, self.key_dim * 2, self.value_dim)?;
                    let q = q_raw.reshape((self.num_k_heads, self.key_head_dim))?;
                    let k = k_raw.reshape((self.num_k_heads, self.key_head_dim))?;
                    let v = v_raw.reshape((self.num_v_heads, self.value_head_dim))?;
                    let gqa_ratio = self.num_v_heads / self.num_k_heads;
                    let q = tile_heads(&q, gqa_ratio)?;
                    let k = tile_heads(&k, gqa_ratio)?;
                    let q = l2_normalize(&q)?;
                    let k = l2_normalize(&k)?;
                    let scale = 1.0 / (self.key_head_dim as f64).sqrt();
                    let q = (q * scale)?;

                    #[cfg(target_os = "macos")]
                    let metal_result = if device.is_metal()
                        && std::env::var_os("DINF_DISABLE_QWEN35_DELTANET_METAL").is_none()
                    {
                        Some(super::qwen35_deltanet_metal::step(
                            &rec_state,
                            &q.contiguous()?,
                            &k.contiguous()?,
                            &v.contiguous()?,
                            &beta_in,
                            &alpha_in,
                            &self.a_log,
                            &self.dt_bias,
                        )?)
                    } else {
                        None
                    };
                    #[cfg(not(target_os = "macos"))]
                    let metal_result: Option<(Tensor, Tensor)> = None;

                    let (output, new_rec_state) = if let Some(result) = metal_result {
                        result
                    } else {
                        let beta_vals: Vec<f32> = candle_nn::Activation::Sigmoid
                            .forward(&beta_in)?
                            .to_vec1()?;
                        let alpha_biased = (&alpha_in + &self.dt_bias)?;
                        let g_tensor = (&self.a_log * &softplus(&alpha_biased)?)?;
                        let g_vals: Vec<f32> = g_tensor.to_vec1()?;
                        deltanet_forward(&q, &k, &v, &beta_vals, &g_vals, &rec_state)?
                    };
                    rec_state = new_rec_state;
                    recurrent_outputs.push(output.unsqueeze(0)?);
                    if let Some(snapshots) = recurrent_snapshots.as_mut() {
                        snapshots.push(rec_state.unsqueeze(0)?);
                    }
                }
                let rollback_state = recurrent_snapshots
                    .map(|snapshots| Tensor::cat(&snapshots, 0))
                    .map(|snapshots| snapshots.map(DeltaNetRollback::Snapshots))
                    .transpose()?;
                (
                    Tensor::cat(&recurrent_outputs, 0)?,
                    conv_state,
                    rec_state,
                    rollback_state,
                    false,
                )
            };

        let speculative_state = if keep_speculative_snapshots {
            Some(
                match rollback_state.context("Qwen DeltaNet rollback data is missing")? {
                    DeltaNetRollback::Snapshots(recurrent_state) => {
                        DeltaNetSpeculativeState::Snapshots(DeltaNetSnapshots {
                            conv_state: super::copy_persistent_tensor(&deltanet_conv_snapshots(
                                &initial_conv_state,
                                &qkv_all,
                                self.conv_kernel,
                            )?)?,
                            recurrent_state: super::copy_persistent_tensor(&recurrent_state)?,
                        })
                    }
                    #[cfg(target_os = "macos")]
                    DeltaNetRollback::Replay(replay) => {
                        DeltaNetSpeculativeState::Replay(DeltaNetReplay {
                            initial_conv_state,
                            // Shared verification slices are contiguous but
                            // retain the projections for every other lane.
                            conv_inputs: super::copy_persistent_tensor(&qkv_all)?,
                            initial_recurrent_state: initial_rec_state,
                            keys: replay.keys,
                            deltas: replay.deltas,
                            decays: replay.decays,
                        })
                    }
                },
            )
        } else {
            None
        };
        *cache = Some(DeltaNetCache {
            conv_state,
            recurrent_state: rec_state,
            speculative_state,
        });
        if let Some(started) = started {
            finish_profile_region(device, started, &DN_SCAN_NS)?;
        }

        // Normalize, gate, and project the complete scan output as one GEMM.
        let started = profile.then(Instant::now);
        #[cfg(target_os = "macos")]
        let fused_gated = if scan_output_gated {
            None
        } else {
            super::fused_norm_metal::try_fused_head_rms_norm_silu_gate(
                &recurrent_output,
                &gate_all,
                &self.ssm_norm.weight,
                seq_len,
                self.num_v_heads,
                self.value_head_dim,
                self.ssm_norm.eps as f32,
            )?
        };
        #[cfg(not(target_os = "macos"))]
        let fused_gated: Option<Tensor> = None;
        let gated = if scan_output_gated {
            #[cfg(target_os = "macos")]
            FUSED_SSM_GATE_LOGGED.get_or_init(|| {
                tracing::info!("fused DeltaNet scan/RMSNorm/SiLU gate enabled for Qwen3.5");
            });
            recurrent_output.reshape((1, seq_len, self.value_dim))?
        } else if let Some(gated) = fused_gated {
            #[cfg(target_os = "macos")]
            FUSED_SSM_GATE_LOGGED.get_or_init(|| {
                tracing::info!("fused SSM RMSNorm/SiLU gate enabled for Qwen3.5");
            });
            gated
        } else {
            let normed_output = self
                .ssm_norm
                .forward(&recurrent_output)?
                .reshape((1, seq_len, self.value_dim))?
                .to_dtype(DType::F32)?;
            silu_mul(&gate_all, &normed_output)?
        };
        if let Some(started) = started {
            finish_profile_region(device, started, &DN_OUTPUT_NS)?;
        }
        Ok(gated)
    }

    fn output_and_ffn(&self, x: &Tensor, gated: &Tensor, decode_rows: bool) -> Result<Tensor> {
        let input_dtype = x.dtype();
        let residual = x.to_dtype(DType::F32)?;
        let profile = layer_profile_enabled();
        let started = profile.then(Instant::now);
        let fused_residual = if decode_rows {
            None
        } else {
            self.out_proj
                .forward_qwen35_ssm_residual(gated, &residual)?
        };
        let (residual_accum, ffn_in) = if let Some(residual_accum) = fused_residual {
            #[cfg(target_os = "macos")]
            FUSED_Q8_SSM_RESIDUAL_LOGGED.get_or_init(|| {
                tracing::info!(
                    "fused pre-M5 Q8_0 SSM output projection/residual enabled for Qwen3.5"
                );
            });
            let ffn_in = self.ffn_norm.forward(&residual_accum)?;
            (residual_accum, ffn_in)
        } else {
            let projected = if decode_rows {
                self.out_proj.forward_decode_rows(gated)?
            } else {
                self.out_proj.forward(gated)?
            }
            .to_dtype(DType::F32)?;
            self.ffn_norm.forward_residual(&residual, &projected)?
        };
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &DN_OUTPUT_NS)?;
        }

        // FFN
        let started = profile.then(Instant::now);
        let residual = &residual_accum;
        let result = if decode_rows {
            self.ffn.forward_decode_rows(&ffn_in, residual)?
        } else {
            self.ffn.forward_residual(&ffn_in, residual)?
        }
        .to_dtype(input_dtype)?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &DN_FFN_NS)?;
        }

        Ok(result)
    }

    fn forward_decode_batch(
        &self,
        x: &Tensor,
        caches: &mut [Option<DeltaNetCache>],
    ) -> Result<Tensor> {
        // Rows are requests, not successive tokens of one recurrent scan.
        let lanes = caches.len();
        let profile = layer_profile_enabled();
        if profile {
            DN_PROFILE_CALLS.fetch_add(1, Ordering::Relaxed);
        }
        let started = profile.then(Instant::now);
        let normed = self.attn_norm.forward(x)?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &DN_NORM_NS)?;
        }
        let started = profile.then(Instant::now);
        let (qkv, gate) = self
            .projections
            .qkv_gate
            .forward_decode_rows(&normed, self.conv_dim)?;
        let (alpha, beta) = self
            .projections
            .alpha_beta
            .forward_decode_rows(&normed, self.num_v_heads)?;
        let (qkv, alpha, beta) = (qkv.squeeze(0)?, alpha.squeeze(0)?, beta.squeeze(0)?);
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &DN_PROJECTIONS_NS)?;
        }
        let mut gated = Vec::with_capacity(lanes);
        for (lane, cache) in caches.iter_mut().enumerate() {
            gated.push(self.scan_and_gate(
                (
                    qkv.narrow(0, lane, 1)?.contiguous()?,
                    gate.narrow(1, lane, 1)?.contiguous()?,
                    alpha.narrow(0, lane, 1)?.contiguous()?,
                    beta.narrow(0, lane, 1)?.contiguous()?,
                ),
                cache,
                false,
            )?);
        }
        self.output_and_ffn(x, &Tensor::cat(&gated, 1)?, true)
    }
}

// ── Gated Attention block forward ────────────────────────────────────────

impl GatedAttentionBlock {
    fn forward(
        &self,
        x: &Tensor,
        kv_cache: &mut Option<KvCache>,
        seq_len: usize,
        start_pos: usize,
        positions: Option<&MropePositionIds>,
    ) -> Result<Tensor> {
        let profile = layer_profile_enabled();
        if profile {
            ATTN_PROFILE_CALLS.fetch_add(1, Ordering::Relaxed);
        }

        // Pre-norm
        let started = profile.then(Instant::now);
        let normed = self.attn_norm.forward(x)?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &ATTN_NORM_NS)?;
        }

        // Q projection (doubled: per-head interleaved [query, gate])
        // Reference: reshape to [b, seq, num_heads, head_dim*2], then split last dim
        let started = profile.then(Instant::now);
        let (q_full, k, v) = self.input_projections.forward(
            &normed,
            seq_len,
            profile,
            self.num_heads * self.head_dim * 2,
            self.num_kv_heads * self.head_dim,
        )?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &ATTN_PROJECTIONS_NS)?;
        }
        let gated_output =
            self.attention_and_gate((q_full, k, v), kv_cache, start_pos, positions)?;
        self.output_and_ffn(x, &gated_output, false)
    }

    fn attention_and_gate(
        &self,
        projections: (Tensor, Tensor, Tensor),
        kv_cache: &mut Option<KvCache>,
        start_pos: usize,
        positions: Option<&MropePositionIds>,
    ) -> Result<Tensor> {
        let (q_full, k, v) = projections;
        let (b_sz, seq_len, _) = q_full.dims3()?;
        let device = q_full.device().clone();
        let profile = layer_profile_enabled();
        // Reshape to [batch, heads, seq_len, head_dim]. For ordinary
        // single-token text decode, one Metal dispatch also performs Q/K
        // normalization and partial RoPE. Positioned M-RoPE activations retain
        // the exact axis-selection path below.
        let started = profile.then(Instant::now);
        #[cfg(target_os = "macos")]
        let fused_decode_layout =
            if let Some(position) = fused_decode_text_position(seq_len, start_pos, positions) {
                super::qkv_layout_metal::try_fused_qwen35_decode_qkv_norm_rope_layout(
                    &q_full,
                    &k,
                    &v,
                    &self.q_norm.weight,
                    &self.k_norm.weight,
                    &self.rope_cos,
                    &self.rope_sin,
                    self.q_norm.eps as f32,
                    self.k_norm.eps as f32,
                    super::qkv_layout_metal::Qwen35DecodeQkvDimensions {
                        batch: b_sz,
                        num_heads: self.num_heads,
                        num_kv_heads: self.num_kv_heads,
                        head_dim: self.head_dim,
                        rope_dims: self.rope_dims,
                        position,
                    },
                )?
            } else {
                None
            };
        #[cfg(not(target_os = "macos"))]
        let fused_decode_layout: Option<(Tensor, Tensor, Tensor, Tensor)> = None;
        let qk_prepared = fused_decode_layout.is_some();
        let (query, k, v, gate) = if let Some(layout) = fused_decode_layout {
            #[cfg(target_os = "macos")]
            FUSED_DECODE_QKV_LOGGED.get_or_init(|| {
                tracing::info!("fused QKV layout/RMSNorm/partial-RoPE decode enabled for Qwen3.5");
            });
            layout
        } else if seq_len == 1 {
            let q_reshaped = q_full.reshape((b_sz, 1, self.num_heads, self.head_dim * 2))?;
            let query = q_reshaped.narrow(3, 0, self.head_dim)?.contiguous()?;
            let gate = q_reshaped
                .narrow(3, self.head_dim, self.head_dim)?
                .contiguous()?;
            // [b, 1, heads, dim] → [b, heads, 1, dim] (same data layout for seq=1)
            let q = query.reshape((b_sz, self.num_heads, 1, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, 1, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, 1, self.head_dim))?;
            let g = gate.reshape((b_sz, self.num_heads, 1, self.head_dim))?;
            (q, k, v, g)
        } else {
            #[cfg(target_os = "macos")]
            if std::env::var_os("DINF_DISABLE_QWEN35_FUSED_GATED_QKV_LAYOUT").is_none()
                && let Some(layout) = super::qkv_layout_metal::try_fused_gated_qkv_layout(
                    &q_full,
                    &k,
                    &v,
                    super::qkv_layout_metal::QkvLayoutDimensions {
                        batch: b_sz,
                        seq_len,
                        num_heads: self.num_heads,
                        num_kv_heads: self.num_kv_heads,
                        head_dim: self.head_dim,
                    },
                )?
            {
                layout
            } else {
                self.portable_gated_qkv_layout(q_full, k, v, b_sz, seq_len)?
            }
            #[cfg(not(target_os = "macos"))]
            self.portable_gated_qkv_layout(q_full, k, v, b_sz, seq_len)?
        };

        let (query, k) = if qk_prepared {
            (query, k)
        } else {
            // QK normalization (per-head RMSNorm applied on last dim = head_dim)
            let query = self.q_norm.forward(&query)?;
            let k = self.k_norm.forward(&k)?;

            // Partial RoPE (only first rope_dims of head_dim)
            let query = apply_partial_rotary_emb_with_positions(
                &query,
                start_pos,
                &self.rope_cos,
                &self.rope_sin,
                self.rope_dims,
                positions,
                self.mrope_sections,
            )?;
            let k = apply_partial_rotary_emb_with_positions(
                &k,
                start_pos,
                &self.rope_cos,
                &self.rope_sin,
                self.rope_dims,
                positions,
                self.mrope_sections,
            )?;
            (query, k)
        };

        let (query, k, v, rotation) = super::prepare_quantized_attention(
            kv_cache,
            query,
            k,
            v,
            self.quant_config.as_deref(),
        )?;
        if let Some(started) = started {
            finish_profile_region(&device, started, &ATTN_LAYOUT_NS)?;
        }
        let started = profile.then(Instant::now);
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_output = if let Some(output) =
            super::try_fused_f16_sdpa_native(&query, &k, &v, seq_len > 1, scale as f32)?
        {
            output
        } else {
            let n_rep = self.num_heads / self.num_kv_heads;
            let k = repeat_kv(k.to_dtype(query.dtype())?, n_rep)?;
            let v = repeat_kv(v.to_dtype(query.dtype())?, n_rep)?;
            let attn_weights = (query.matmul(&k.transpose(2, 3)?)? * scale)?;
            let attn_weights = if seq_len > 1 {
                let mask = create_causal_mask(seq_len, attn_weights.dim(3)?, &device)?;
                attn_weights.broadcast_add(&mask)?
            } else {
                attn_weights
            };
            softmax_last_dim(&attn_weights)?.matmul(&v)?
        };
        let attn_output = super::restore_quantized_attention_output(attn_output, rotation)?;

        // Output gating: attn_out * sigmoid(gate)
        #[cfg(target_os = "macos")]
        let gated_output = if std::env::var_os(
            "DINF_DISABLE_QWEN35_FUSED_GATED_ATTENTION_OUTPUT_LAYOUT",
        )
        .is_none()
            && let Some(output) =
                super::qkv_layout_metal::try_fused_sigmoid_gated_attention_output_layout(
                    &attn_output,
                    &gate,
                )? {
            output
        } else {
            let gate_sigmoid = candle_nn::Activation::Sigmoid.forward(&gate)?;
            (attn_output.to_dtype(gate_sigmoid.dtype())? * &gate_sigmoid)?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b_sz, seq_len, ()))?
        };
        #[cfg(not(target_os = "macos"))]
        let gated_output = {
            let gate_sigmoid = candle_nn::Activation::Sigmoid.forward(&gate)?;
            (attn_output.to_dtype(gate_sigmoid.dtype())? * &gate_sigmoid)?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b_sz, seq_len, ()))?
        };

        if let Some(started) = started {
            finish_profile_region(&device, started, &ATTN_SDPA_OUTPUT_NS)?;
        }
        Ok(gated_output)
    }

    fn output_and_ffn(
        &self,
        x: &Tensor,
        gated_output: &Tensor,
        decode_rows: bool,
    ) -> Result<Tensor> {
        let profile = layer_profile_enabled();
        let started = profile.then(Instant::now);
        let projected = if decode_rows {
            self.o_proj.forward_decode_rows(gated_output)?
        } else {
            self.o_proj.forward(gated_output)?
        };
        let (x, ffn_in) = self.ffn_norm.forward_residual(x, &projected)?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &ATTN_SDPA_OUTPUT_NS)?;
        }

        // FFN
        let started = profile.then(Instant::now);
        let residual = &x;
        let result = if decode_rows {
            self.ffn.forward_decode_rows(&ffn_in, residual)?
        } else {
            self.ffn.forward_residual(&ffn_in, residual)?
        };
        if let Some(started) = started {
            finish_profile_region(result.device(), started, &ATTN_FFN_NS)?;
        }

        Ok(result)
    }

    fn forward_decode_batch(
        &self,
        x: &Tensor,
        caches: &mut [Option<KvCache>],
        positions: &[usize],
    ) -> Result<Tensor> {
        let profile = layer_profile_enabled();
        if profile {
            ATTN_PROFILE_CALLS.fetch_add(1, Ordering::Relaxed);
        }
        let started = profile.then(Instant::now);
        let normed = self.attn_norm.forward(x)?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &ATTN_NORM_NS)?;
        }
        let started = profile.then(Instant::now);
        let (q, k, v) = match &self.input_projections {
            AttentionInputProjections::Separate { q, k, v } => (
                q.forward_decode_rows(&normed)?,
                k.forward_decode_rows(&normed)?,
                v.forward_decode_rows(&normed)?,
            ),
            AttentionInputProjections::Grouped(projection) => {
                let all = projection.forward_decode_rows(&normed)?;
                let q_width = self.num_heads * self.head_dim * 2;
                let kv_width = self.num_kv_heads * self.head_dim;
                (
                    all.narrow(2, 0, q_width)?.contiguous()?,
                    all.narrow(2, q_width, kv_width)?.contiguous()?,
                    all.narrow(2, q_width + kv_width, kv_width)?.contiguous()?,
                )
            }
        };
        let mut gated = Vec::with_capacity(caches.len());
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &ATTN_PROJECTIONS_NS)?;
        }
        for (lane, (cache, position)) in caches.iter_mut().zip(positions).enumerate() {
            gated.push(self.attention_and_gate(
                (
                    q.narrow(1, lane, 1)?.contiguous()?,
                    k.narrow(1, lane, 1)?.contiguous()?,
                    v.narrow(1, lane, 1)?.contiguous()?,
                ),
                cache,
                *position,
                None,
            )?);
        }
        self.output_and_ffn(x, &Tensor::cat(&gated, 1)?, true)
    }

    fn portable_gated_qkv_layout(
        &self,
        q_full: Tensor,
        k: Tensor,
        v: Tensor,
        batch: usize,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let q_reshaped = q_full.reshape((batch, seq_len, self.num_heads, self.head_dim * 2))?;
        let query = q_reshaped.narrow(3, 0, self.head_dim)?.contiguous()?;
        let gate = q_reshaped
            .narrow(3, self.head_dim, self.head_dim)?
            .contiguous()?;
        let query = query.transpose(1, 2)?.contiguous()?;
        let k = k
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let gate = gate.transpose(1, 2)?.contiguous()?;
        Ok((query, k, v, gate))
    }
}

// ── Stage construction ───────────────────────────────────────────────────

fn qtensor_to_qmatmul(qt: Arc<QTensor>) -> Result<QMatMul> {
    QMatMul::from_arc(qt).map_err(Into::into)
}

fn take_qtensor(
    tensors: &mut std::collections::HashMap<String, Arc<QTensor>>,
    prefix: &str,
    name: &str,
) -> Result<Arc<QTensor>> {
    tensors
        .remove(&format!("{prefix}.{name}"))
        .with_context(|| format!("missing tensor: {prefix}.{name}"))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum QwenOutput {
    Last,
    All,
    Hidden,
}

impl Qwen35Stage {
    /// Construct a Qwen 3.5 stage from its assigned GGUF tensors.
    ///
    /// Returns an error when the hybrid attention schedule, recurrent-state
    /// dimensions, or required tensors are missing or inconsistent.
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

        #[cfg(target_os = "macos")]
        super::qwen35_deltanet_metal::precompile(&device)?;
        #[cfg(feature = "cuda")]
        {
            super::qwen35_deltanet_cuda::precompile(&device)?;
            super::dense_qmatmul_cuda::precompile(&device)?;
        }

        let full_attention_interval = if md.full_attention_interval > 0 {
            md.full_attention_interval
        } else {
            4 // default for Qwen 3.5
        };

        // SSM dimensions (from metadata or defaults for 9B)
        let ssm_state_size = if md.ssm_state_size > 0 {
            md.ssm_state_size
        } else {
            128
        };
        let ssm_conv_kernel = if md.ssm_conv_kernel > 0 {
            md.ssm_conv_kernel
        } else {
            4
        };
        let ssm_group_count = if md.ssm_group_count > 0 {
            md.ssm_group_count
        } else {
            16
        };
        let ssm_time_step_rank = if md.ssm_time_step_rank > 0 {
            md.ssm_time_step_rank
        } else {
            32
        };

        // Derived DeltaNet dimensions. Qwen3.5 decouples hidden size from SSM
        // inner/value size (for example 27B has hidden=5120, inner=6144).
        let key_head_dim = ssm_state_size; // 128
        let num_k_heads = ssm_group_count; // 16
        let num_v_heads = ssm_time_step_rank; // 48 for Qwen3.6-27B
        let value_dim = if md.ssm_inner_size > 0 {
            md.ssm_inner_size
        } else {
            md.hidden_dim
        };
        anyhow::ensure!(
            value_dim % num_v_heads == 0,
            "qwen35 SSM inner_size/value_dim ({value_dim}) must be divisible by time_step_rank ({num_v_heads})"
        );
        let value_head_dim = value_dim / num_v_heads;
        let key_dim = num_k_heads * key_head_dim;
        let conv_dim = key_dim * 2 + value_dim;

        // Attention dimensions
        let attn_num_heads = md.num_attention_heads; // 16
        let attn_num_kv_heads = md.num_kv_heads; // 4
        let attn_head_dim = md.head_dim; // 256
        let rope_dims = if md.rope_dimension_count > 0 {
            md.rope_dimension_count
        } else {
            64
        };
        let mrope_sections = parse_mrope_sections(&md.rope_dimension_sections, rope_dims)?;

        // Embedding (head only)
        let embedding = if range.is_head() {
            let embd = tensors
                .remove("token_embd.weight")
                .context("missing token_embd.weight for head stage")?;
            Some(prepare_packed_embedding(embd, &device)?)
        } else {
            None
        };

        // The generic stage filter includes the token embedding at a tail in
        // case the output is tied. An untied tail-only stage never uses it, so
        // do not carry that source buffer through construction.
        if !range.is_head() && range.is_tail() && tensors.contains_key("output.weight") {
            tensors.remove("token_embd.weight");
        }

        // Precompute partial RoPE tables (only rope_dims dimensions)
        let (rope_cos, rope_sin) =
            precompute_rope_cos_sin(rope_dims, md.rope_theta, md.context_length, &device)?;

        // Build blocks
        let mut blocks = Vec::with_capacity(range.layer_count());
        let mut layer_indices = Vec::with_capacity(range.layer_count());

        for layer_idx in range.start..range.end {
            let prefix = format!("blk.{layer_idx}");
            let is_attention =
                (layer_idx % full_attention_interval) == (full_attention_interval - 1);

            let block = if is_attention {
                Qwen35Block::GatedAttention(build_gated_attention_block(
                    &mut tensors,
                    &prefix,
                    &md,
                    &device,
                    &rope_cos,
                    &rope_sin,
                    mrope_sections,
                    attn_num_heads,
                    attn_num_kv_heads,
                    attn_head_dim,
                    rope_dims,
                    quant_config.clone(),
                )?)
            } else {
                Qwen35Block::DeltaNet(build_deltanet_block(
                    &mut tensors,
                    &prefix,
                    &md,
                    &device,
                    num_k_heads,
                    num_v_heads,
                    key_head_dim,
                    value_head_dim,
                    key_dim,
                    value_dim,
                    conv_dim,
                    ssm_conv_kernel,
                )?)
            };

            blocks.push(block);
            layer_indices.push(layer_idx);

            // Metal QMM construction copies packed bytes into the custom
            // tensor backend. Drop every source tensor for this layer now so
            // the stage map does not retain a second copy of the full model
            // until construction finishes. QMatMul fallback paths retain
            // their own Arc when they still need the quantized source.
            let layer_prefix = format!("{prefix}.");
            let keys_to_remove: Vec<String> = tensors
                .keys()
                .filter(|key| key.starts_with(&layer_prefix))
                .cloned()
                .collect();
            for key in keys_to_remove {
                tensors.remove(&key);
            }
        }

        // Output norm + lm_head (tail only)
        let output_norm = if range.is_tail() {
            let norm_w = tensors
                .remove("output_norm.weight")
                .context("missing output_norm.weight")?;
            Some(RmsNorm {
                weight: norm_w.as_ref().dequantize(&device)?,
                eps: md.rms_norm_eps,
            })
        } else {
            None
        };

        let lm_head = if range.is_tail() {
            let head = match tensors.remove("output.weight") {
                Some(head) => head,
                None => embedding
                    .as_ref()
                    .cloned()
                    .or_else(|| tensors.remove("token_embd.weight"))
                    .context("missing output.weight or token_embd.weight")?,
            };
            Some(Qwen35QMatMul::from_qtensor(head, &device)?)
        } else {
            None
        };

        tracing::info!(
            "Qwen35Stage: {} blocks ({} DeltaNet + {} Attention), layers [{}, {})",
            blocks.len(),
            blocks
                .iter()
                .filter(|b| matches!(b, Qwen35Block::DeltaNet(_)))
                .count(),
            blocks
                .iter()
                .filter(|b| matches!(b, Qwen35Block::GatedAttention(_)))
                .count(),
            range.start,
            range.end,
        );

        Ok(Self {
            embedding,
            blocks,
            output_norm,
            lm_head,
            is_head: range.is_head(),
            is_tail: range.is_tail(),
            hidden_dim: md.hidden_dim,
            num_layers: range.layer_count(),
            vocab_size: md.vocab_size,
            layer_indices,
            full_attention_interval,
            device,
        })
    }
}

// ── ModelArchitecture implementation ─────────────────────────────────────

impl ModelArchitecture for Qwen35Stage {
    fn fixed_cache_bytes(&self, verification_rows: usize) -> Result<usize> {
        self.blocks.iter().try_fold(0usize, |total, block| {
            let Qwen35Block::DeltaNet(block) = block else {
                return Ok(total);
            };
            let recurrent = super::fixed_state_tensor_bytes(
                &[block.num_v_heads, block.key_head_dim, block.value_head_dim],
                &self.device,
            )?;
            let convolution = super::fixed_state_tensor_bytes(
                &[block.conv_dim, block.conv_kernel],
                &self.device,
            )?;
            // Verification keeps the prior checkpoint and the current state.
            // Portable scans also retain every row for prefix rollback.
            let copies = if verification_rows == 0 {
                1
            } else if self.device.is_metal()
                && block.conv_kernel <= 8
                && block.key_head_dim <= 256
                && std::env::var_os("DINF_DISABLE_QWEN35_DELTANET_METAL").is_none()
                && std::env::var_os("DINF_DISABLE_QWEN35_PREFILL_METAL").is_none()
            {
                2
            } else {
                verification_rows
                    .checked_add(2)
                    .context("verification state count overflow")?
            };
            let replay_bytes = if verification_rows > 0 && copies == 2 {
                [
                    vec![verification_rows, block.conv_dim],
                    vec![verification_rows, block.num_v_heads, block.key_head_dim],
                    vec![verification_rows, block.num_v_heads, block.value_head_dim],
                    vec![verification_rows, block.num_v_heads],
                ]
                .iter()
                .try_fold(0usize, |total, dims| {
                    super::fixed_state_tensor_bytes(dims, &self.device)?
                        .checked_add(total)
                        .context("Qwen replay state size overflow")
                })?
            } else {
                0
            };
            recurrent
                .checked_add(convolution)
                .and_then(|bytes| bytes.checked_mul(copies))
                .and_then(|bytes| bytes.checked_add(replay_bytes))
                .and_then(|bytes| total.checked_add(bytes))
                .context("Qwen fixed request state size overflow")
        })
    }

    fn supports_speculative_batch(&self) -> bool {
        #[cfg(target_os = "macos")]
        if self.device.is_metal() && !super::tensor_qmatmul_metal::supports_shared_verification() {
            return false;
        }
        self.supports_decode_batch()
    }

    fn forward_speculative_batch(
        &self,
        inputs: &[Vec<u32>],
        caches: &mut [&mut [Option<LayerCache>]],
        positions: &[usize],
        layer_ids: &[usize],
    ) -> Result<Vec<super::SpeculativeBatchOutput>> {
        speculative_batch::with_batch(|| self.verify_batch(inputs, caches, positions, layer_ids))
    }

    fn forward(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_with_captures(input, cache, seq_len, start_pos, &[], QwenOutput::Last)
            .map(|(output, _)| output)
    }

    fn forward_prefill_cache_only(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_with_captures(input, cache, seq_len, start_pos, &[], QwenOutput::Hidden)
            .map(|(output, _)| output)
    }

    fn forward_with_layer_inputs(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
        layer_ids: &[usize],
        all_logits: bool,
    ) -> Result<(StageOutput, Vec<Tensor>)> {
        self.forward_with_captures(
            input,
            cache,
            seq_len,
            start_pos,
            layer_ids,
            if all_logits {
                QwenOutput::All
            } else {
                QwenOutput::Last
            },
        )
    }

    fn speculative_shared_weights(&self) -> Option<SpeculativeSharedWeights> {
        Some(SpeculativeSharedWeights {
            token_embedding: self.embedding.as_ref()?.clone(),
            output: self.lm_head.as_ref()?.clone(),
        })
    }

    fn embed_tokens(&self, tokens: &[u32]) -> Result<Tensor> {
        anyhow::ensure!(self.is_head, "only a Qwen 3.5 head stage can embed tokens");
        anyhow::ensure!(!tokens.is_empty(), "cannot embed an empty token list");
        let embedding = self
            .embedding
            .as_ref()
            .context("Qwen 3.5 head stage has no token embedding")?;
        packed_embedding_rows(embedding, tokens, &self.device)
    }

    fn family(&self) -> &str {
        "qwen35"
    }

    fn supports_decode_batch(&self) -> bool {
        self.is_head
            && self.is_tail
            && std::env::var_os("DINF_DISABLE_QWEN35_DECODE_BATCH").is_none()
            && (self.device.is_cpu() || self.has_metal_decode_rows())
            && self.blocks.iter().all(|block| match block {
                Qwen35Block::GatedAttention(block) => block.quant_config.is_none(),
                Qwen35Block::DeltaNet(_) => true,
            })
    }

    fn decode_batch_uses_packed_cache(&self) -> bool {
        false
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
            "Qwen decode batching unavailable"
        );
        anyhow::ensure!(!tokens.is_empty(), "decode batch cannot be empty");
        anyhow::ensure!(tokens.len() == caches.len(), "decode batch lane mismatch");
        anyhow::ensure!(
            tokens.len() == positions.len(),
            "decode batch position mismatch"
        );
        anyhow::ensure!(
            batch_cache.len() == self.blocks.len()
                && caches.iter().all(|cache| cache.len() == self.blocks.len()),
            "decode batch cache layer mismatch"
        );

        // Validate and clone metadata before any model work. Kernels create
        // new recurrent tensors; KV appends only fill unused storage. Publish
        // lengths and state together after every layer and the output succeed.
        let mut next_caches = caches
            .iter()
            .map(|cache| {
                cache
                    .iter()
                    .zip(&self.blocks)
                    .map(|(slot, block)| {
                        anyhow::ensure!(
                            matches!(
                                (slot, block),
                                (None, _)
                                    | (Some(LayerCache::DeltaNet(_)), Qwen35Block::DeltaNet(_))
                                    | (
                                        Some(LayerCache::Attention(_)),
                                        Qwen35Block::GatedAttention(_)
                                    )
                            ),
                            "Qwen decode batch cache type mismatch"
                        );
                        slot.as_ref().map(super::checkpoint_layer_cache).transpose()
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let mut hidden = self
            .embed_tokens(tokens)?
            .reshape((1, tokens.len(), self.hidden_dim))?;
        for (layer, block) in self.blocks.iter().enumerate() {
            hidden = match block {
                Qwen35Block::DeltaNet(block) => {
                    let mut states = next_caches
                        .iter_mut()
                        .map(|cache| match cache[layer].take() {
                            Some(LayerCache::DeltaNet(state)) => Some(state),
                            None => None,
                            _ => unreachable!("cache types checked above"),
                        })
                        .collect::<Vec<_>>();
                    let output = block.forward_decode_batch(&hidden, &mut states)?;
                    for (cache, state) in next_caches.iter_mut().zip(states) {
                        cache[layer] = state.map(LayerCache::DeltaNet);
                    }
                    output
                }
                Qwen35Block::GatedAttention(block) => {
                    let mut states = next_caches
                        .iter_mut()
                        .map(|cache| match cache[layer].take() {
                            Some(LayerCache::Attention(state)) => Some(state),
                            None => None,
                            _ => unreachable!("cache types checked above"),
                        })
                        .collect::<Vec<_>>();
                    let output = block.forward_decode_batch(&hidden, &mut states, positions)?;
                    for (cache, state) in next_caches.iter_mut().zip(states) {
                        cache[layer] = state.map(LayerCache::Attention);
                    }
                    output
                }
            };
        }
        let normed = self
            .output_norm
            .as_ref()
            .context("missing Qwen output norm")?
            .forward(&hidden)?;
        let logits = self
            .lm_head
            .as_ref()
            .context("missing Qwen output head")?
            .forward_decode_rows(&normed.squeeze(0)?)?;
        let outputs = (0..tokens.len())
            .map(|lane| logits.get(lane).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;
        for (cache, next) in caches.iter_mut().zip(next_caches) {
            for (slot, state) in cache.iter_mut().zip(next) {
                *slot = state;
            }
        }
        // Do not pack recurrent state or retain another request's KV storage.
        for slot in batch_cache {
            *slot = None;
        }
        DECODE_BATCH_LOGGED.get_or_init(|| {
            tracing::info!(
                batch_size = tokens.len(),
                "Qwen tensor decode batching active"
            );
        });
        Ok(outputs)
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

impl Qwen35Stage {
    fn has_metal_decode_rows(&self) -> bool {
        self.lm_head
            .as_ref()
            .is_some_and(Qwen35QMatMul::has_metal_decode_rows)
            && self.blocks.iter().all(|block| match block {
                Qwen35Block::DeltaNet(block) => {
                    block.projections.qkv_gate.has_metal_decode_rows()
                        && block.projections.alpha_beta.has_metal_decode_rows()
                        && block.out_proj.has_metal_decode_rows()
                        && block.ffn.has_metal_decode_rows()
                }
                Qwen35Block::GatedAttention(block) => {
                    block.input_projections.has_metal_decode_rows()
                        && block.o_proj.has_metal_decode_rows()
                        && block.ffn.has_metal_decode_rows()
                }
            })
    }

    fn forward_with_captures(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
        layer_ids: &[usize],
        output: QwenOutput,
    ) -> Result<(StageOutput, Vec<Tensor>)> {
        let all_logits = output == QwenOutput::All;
        anyhow::ensure!(
            layer_ids.windows(2).all(|pair| pair[0] < pair[1]),
            "speculative target layer IDs must be sorted and unique"
        );
        anyhow::ensure!(
            !all_logits || self.is_tail,
            "all-logit target passes require a tail stage"
        );
        let (mut hidden_states, positions) = match input {
            StageInput::Tokens(tokens) => {
                anyhow::ensure!(self.is_head, "only head stage can receive tokens");
                anyhow::ensure!(
                    tokens.len() == seq_len,
                    "received {} tokens for sequence length {seq_len}",
                    tokens.len()
                );
                let embedding = self.embedding.as_ref().unwrap();
                (
                    packed_embedding_rows(embedding, tokens.as_slice(), &self.device)?
                        .reshape((1, seq_len, self.hidden_dim))?,
                    None,
                )
            }
            StageInput::Activations {
                tensor,
                bidirectional_spans,
                positions,
            } => {
                anyhow::ensure!(
                    bidirectional_spans.is_empty(),
                    "Qwen multimodal attention must remain causal"
                );
                if let Some(positions) = positions.as_ref() {
                    anyhow::ensure!(
                        positions.len() == seq_len,
                        "M-RoPE schedule has {} entries for sequence length {seq_len}",
                        positions.len()
                    );
                }
                let tensor = if tensor.dims().len() == 2 {
                    tensor.unsqueeze(0)?
                } else {
                    tensor
                };
                (tensor, positions)
            }
        };

        let mut captures = Vec::with_capacity(layer_ids.len());
        let mut next_capture = 0usize;
        for (i, block) in self.blocks.iter().enumerate() {
            let layer_id = self.layer_indices[i];
            // GGUF stores each trained output layer as `layer_id + 1`, so the
            // requested feature is the input entering this stored layer.
            if layer_ids.get(next_capture).copied() == Some(layer_id) {
                captures.push(hidden_states.squeeze(0)?.to_dtype(DType::F32)?);
                next_capture += 1;
            }
            hidden_states = match block {
                Qwen35Block::DeltaNet(dn_block) => {
                    // Extract or create DeltaNet cache
                    let mut dn_cache = match cache[i].take() {
                        Some(LayerCache::DeltaNet(c)) => Some(c),
                        Some(_) => panic!("expected DeltaNet cache"),
                        None => None,
                    };
                    let result = dn_block.forward(
                        &hidden_states,
                        &mut dn_cache,
                        seq_len,
                        start_pos,
                        all_logits,
                    )?;
                    cache[i] = dn_cache.map(LayerCache::DeltaNet);
                    result
                }
                Qwen35Block::GatedAttention(attn_block) => {
                    // Extract or create KV cache
                    let mut kv_cache = match cache[i].take() {
                        Some(LayerCache::Attention(kv)) => Some(kv),
                        Some(_) => panic!("expected Attention cache"),
                        None => None,
                    };
                    let result = attn_block.forward(
                        &hidden_states,
                        &mut kv_cache,
                        seq_len,
                        start_pos,
                        positions.as_ref(),
                    )?;
                    cache[i] = kv_cache.map(LayerCache::Attention);
                    result
                }
            };
        }

        anyhow::ensure!(
            next_capture == layer_ids.len(),
            "Qwen target stage did not contain every requested DFlash layer input"
        );

        if self.is_tail && output != QwenOutput::Hidden {
            let normed = self.output_norm.as_ref().unwrap().forward(&hidden_states)?;
            let head_input = if all_logits {
                normed.squeeze(0)?
            } else if seq_len > 1 {
                normed.i((.., seq_len - 1, ..))?
            } else {
                normed.squeeze(1)?
            };

            let logits = self.lm_head.as_ref().unwrap().forward(&head_input)?;
            let logits = if all_logits {
                logits
            } else {
                logits.squeeze(0)?
            };

            Ok((StageOutput::Logits(logits), captures))
        } else {
            let output = hidden_states.squeeze(0)?;
            Ok((StageOutput::Activations(output), captures))
        }
    }
}

// ── Block builders ───────────────────────────────────────────────────────

fn build_mlp(
    tensors: &mut std::collections::HashMap<String, Arc<QTensor>>,
    prefix: &str,
    device: &Device,
) -> Result<Mlp> {
    let gate = take_qtensor(tensors, prefix, "ffn_gate.weight")?;
    let up = take_qtensor(tensors, prefix, "ffn_up.weight")?;
    let (gate_dim, gate_input_dim) = gate.shape().dims2()?;
    let (up_dim, up_input_dim) = up.shape().dims2()?;
    anyhow::ensure!(
        gate_dim == up_dim && gate_input_dim == up_input_dim,
        "Qwen3.5 MLP gate/up projection geometry differs"
    );

    #[cfg(target_os = "macos")]
    let grouped = if std::env::var_os("DINF_DISABLE_QWEN35_FUSED_MLP").is_none() {
        Qwen35QMatMul::try_grouped(&[Arc::clone(&gate), Arc::clone(&up)], device)?
    } else {
        None
    };
    #[cfg(not(target_os = "macos"))]
    let grouped: Option<Qwen35QMatMul> = None;

    let input_projections = match grouped {
        Some(projection) => MlpInputProjections::Grouped {
            projection,
            intermediate_dim: gate_dim,
        },
        None => MlpInputProjections::Separate {
            gate: Qwen35QMatMul::from_qtensor(gate, device)?,
            up: Qwen35QMatMul::from_qtensor(up, device)?,
        },
    };
    let down_proj =
        Qwen35QMatMul::from_qtensor(take_qtensor(tensors, prefix, "ffn_down.weight")?, device)?;
    Ok(Mlp {
        input_projections,
        down_proj,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_deltanet_block(
    tensors: &mut std::collections::HashMap<String, Arc<QTensor>>,
    prefix: &str,
    md: &crate::gguf_loader::GgufMetadata,
    device: &Device,
    num_k_heads: usize,
    num_v_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    conv_kernel: usize,
) -> Result<DeltaNetBlock> {
    let attn_norm = RmsNorm {
        weight: take_qtensor(tensors, prefix, "attn_norm.weight")?
            .as_ref()
            .dequantize(device)?,
        eps: md.rms_norm_eps,
    };

    #[cfg(target_os = "macos")]
    let grouped_projections = std::env::var_os("DINF_DISABLE_QWEN35_GROUPED_PROJECTIONS").is_none()
        && std::env::var_os("DINF_DISABLE_QWEN35_GROUPED_DELTANET").is_none();
    let qkv_gate = {
        let qkv = take_qtensor(tensors, prefix, "attn_qkv.weight")?;
        let gate = take_qtensor(tensors, prefix, "attn_gate.weight")?;
        #[cfg(target_os = "macos")]
        let grouped = if grouped_projections {
            Qwen35QMatMul::try_grouped(&[Arc::clone(&qkv), Arc::clone(&gate)], device)?
        } else {
            None
        };
        #[cfg(not(target_os = "macos"))]
        let grouped: Option<Qwen35QMatMul> = None;
        match grouped {
            Some(grouped) => HomogeneousProjectionPair::Grouped(grouped),
            None => HomogeneousProjectionPair::Separate {
                first: Qwen35QMatMul::from_qtensor(qkv, device)?,
                second: Qwen35QMatMul::from_qtensor(gate, device)?,
            },
        }
    };
    // Build each homogeneous group in its own scope. This drops the first
    // pair's source tensors before materializing the second packed matrix,
    // bounding construction overlap to the larger group instead of their sum.
    let alpha_beta = {
        let alpha = take_qtensor(tensors, prefix, "ssm_alpha.weight")?;
        let beta = take_qtensor(tensors, prefix, "ssm_beta.weight")?;
        #[cfg(target_os = "macos")]
        let grouped = if grouped_projections {
            Qwen35QMatMul::try_grouped(&[Arc::clone(&alpha), Arc::clone(&beta)], device)?
        } else {
            None
        };
        #[cfg(not(target_os = "macos"))]
        let grouped: Option<Qwen35QMatMul> = None;
        match grouped {
            Some(grouped) => HomogeneousProjectionPair::Grouped(grouped),
            None => HomogeneousProjectionPair::Separate {
                first: Qwen35QMatMul::from_qtensor(alpha, device)?,
                second: Qwen35QMatMul::from_qtensor(beta, device)?,
            },
        }
    };
    let projections = DeltaNetProjections {
        qkv_gate,
        alpha_beta,
    };

    // Conv1d weight: depthwise conv [conv_dim, kernel_size] after GGUF load
    let conv1d_weight = take_qtensor(tensors, prefix, "ssm_conv1d.weight")?
        .as_ref()
        .dequantize(device)?;

    // ssm_a: NO .weight suffix!
    let a_log = take_qtensor(tensors, prefix, "ssm_a")?
        .as_ref()
        .dequantize(device)?;

    // dt bias
    let dt_bias = take_qtensor(tensors, prefix, "ssm_dt.bias")?
        .as_ref()
        .dequantize(device)?;

    // Per-head RMSNorm for SSM output
    // NOTE: ssm_norm uses Qwen3_5RMSNormGated which does plain `weight * normalized`
    // (initialized to ones, no +1). The GGUF converter does NOT add +1 to this norm.
    let ssm_norm = RmsNorm {
        weight: take_qtensor(tensors, prefix, "ssm_norm.weight")?
            .as_ref()
            .dequantize(device)?,
        eps: md.rms_norm_eps,
    };

    let out_proj =
        Qwen35QMatMul::from_qtensor(take_qtensor(tensors, prefix, "ssm_out.weight")?, device)?;

    let ffn_norm_weight = tensors
        .remove(&format!("{prefix}.ffn_norm.weight"))
        .or_else(|| tensors.remove(&format!("{prefix}.post_attention_norm.weight")))
        .with_context(|| format!("missing tensor: {prefix}.post_attention_norm.weight"))?;
    let ffn_norm = RmsNorm {
        weight: ffn_norm_weight.as_ref().dequantize(device)?,
        eps: md.rms_norm_eps,
    };

    let ffn = build_mlp(tensors, prefix, device)?;

    Ok(DeltaNetBlock {
        attn_norm,
        projections,
        conv1d_weight,
        a_log,
        dt_bias,
        ssm_norm,
        out_proj,
        ffn_norm,
        ffn,
        num_k_heads,
        num_v_heads,
        key_head_dim,
        value_head_dim,
        key_dim,
        value_dim,
        conv_dim,
        conv_kernel,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_gated_attention_block(
    tensors: &mut std::collections::HashMap<String, Arc<QTensor>>,
    prefix: &str,
    md: &crate::gguf_loader::GgufMetadata,
    device: &Device,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
    mrope_sections: Option<[usize; 3]>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dims: usize,
    quant_config: Option<Arc<TurboQuantConfig>>,
) -> Result<GatedAttentionBlock> {
    let attn_norm = RmsNorm {
        weight: take_qtensor(tensors, prefix, "attn_norm.weight")?
            .as_ref()
            .dequantize(device)?,
        eps: md.rms_norm_eps,
    };

    let input_projections = {
        let q = take_qtensor(tensors, prefix, "attn_q.weight")?;
        let k = take_qtensor(tensors, prefix, "attn_k.weight")?;
        let v = take_qtensor(tensors, prefix, "attn_v.weight")?;
        #[cfg(target_os = "macos")]
        let grouped = if std::env::var_os("DINF_DISABLE_QWEN35_GROUPED_PROJECTIONS").is_none()
            && std::env::var_os("DINF_DISABLE_QWEN35_GROUPED_ATTENTION_PROJECTIONS").is_none()
        {
            Qwen35QMatMul::try_grouped(&[Arc::clone(&q), Arc::clone(&k), Arc::clone(&v)], device)?
        } else {
            None
        };
        #[cfg(not(target_os = "macos"))]
        let grouped: Option<Qwen35QMatMul> = None;
        match grouped {
            Some(grouped) => AttentionInputProjections::Grouped(grouped),
            None => AttentionInputProjections::Separate {
                q: Qwen35QMatMul::from_qtensor(q, device)?,
                k: Qwen35QMatMul::from_qtensor(k, device)?,
                v: Qwen35QMatMul::from_qtensor(v, device)?,
            },
        }
    };
    let o_proj =
        Qwen35QMatMul::from_qtensor(take_qtensor(tensors, prefix, "attn_output.weight")?, device)?;

    let q_norm = RmsNorm {
        weight: take_qtensor(tensors, prefix, "attn_q_norm.weight")?
            .as_ref()
            .dequantize(device)?,
        eps: md.rms_norm_eps,
    };
    let k_norm = RmsNorm {
        weight: take_qtensor(tensors, prefix, "attn_k_norm.weight")?
            .as_ref()
            .dequantize(device)?,
        eps: md.rms_norm_eps,
    };

    let ffn_norm_weight = tensors
        .remove(&format!("{prefix}.ffn_norm.weight"))
        .or_else(|| tensors.remove(&format!("{prefix}.post_attention_norm.weight")))
        .with_context(|| format!("missing tensor: {prefix}.post_attention_norm.weight"))?;
    let ffn_norm = RmsNorm {
        weight: ffn_norm_weight.as_ref().dequantize(device)?,
        eps: md.rms_norm_eps,
    };

    let ffn = build_mlp(tensors, prefix, device)?;

    Ok(GatedAttentionBlock {
        attn_norm,
        input_projections,
        o_proj,
        q_norm,
        k_norm,
        ffn_norm,
        ffn,
        rope_cos: rope_cos.clone(),
        rope_sin: rope_sin.clone(),
        mrope_sections,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_dims,
        quant_config,
    })
}

// ── Shared utility functions ─────────────────────────────────────────────

fn precompute_rope_cos_sin(
    rope_dims: usize,
    theta: f64,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    // Standard RoPE: cos/sin of shape [ctx, half_dim]
    // For text-only Qwen 3.5 inference, M-RoPE interleaving is a no-op
    // (all 3 position sections get the same position IDs).
    let half_dim = rope_dims / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / rope_dims as f32))
        .collect();
    let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
    let positions = Tensor::arange(0u32, context_length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?;
    let angles = positions.matmul(&inv_freq.reshape((1, half_dim))?)?; // [ctx, half_dim]
    Ok((angles.cos()?, angles.sin()?))
}

/// Repeat heads for GQA (interleave): [num_heads, dim] → [num_heads * n_rep, dim]
/// Tile heads for GQA: [num_heads, dim] → [num_heads * n_rep, dim]
/// Result order: [h0, h1, ..., h15, h0, h1, ..., h15] — matches GGUF tiled V ordering.
/// The GGUF converter reorders V-head-related tensors from HF grouped to ggml tiled order.
fn tile_heads(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let reps: Vec<&Tensor> = (0..n_rep).map(|_| x).collect();
    Tensor::cat(&reps, 0)?.contiguous().map_err(Into::into)
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
    Tensor::from_vec(mask, (1, 1, seq_len, total_seq_len), device).map_err(Into::into)
}

#[path = "qwen35_speculative_batch.rs"]
mod speculative_batch;

// ── Unit tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "qwen35_batch_tests.rs"]
mod batch_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn tiny_scale_q4k_embedding() -> Result<Arc<QTensor>> {
        let mut block = vec![0u8; 144];
        block[..2].copy_from_slice(&1u16.to_le_bytes()); // smallest positive F16
        block[4..8].fill(1);
        block[12..16].fill(1);
        block[16..].fill(0xff);
        let storage = QStorage::from_data(block.into(), &Device::Cpu, GgmlDType::Q4K)?;
        Ok(Arc::new(QTensor::new(storage, (1, 256))?))
    }

    #[test]
    fn packed_q4k_rows_keep_subnormal_scale_bits() -> Result<()> {
        let embedding = tiny_scale_q4k_embedding()?;
        let prepared = prepare_packed_embedding(Arc::clone(&embedding), &Device::Cpu)?;
        assert!(Arc::ptr_eq(&prepared, &embedding));
        assert_eq!(prepared.data()?.len(), 144);
        let rows = packed_embedding_rows(&prepared, &[0, 0], &Device::Cpu)?;
        assert_eq!(rows.dims(), &[2, 256]);
        let expected = 15.0f32 * 2.0f32.powi(-24);
        assert!(
            rows.flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|v| v.to_bits() == expected.to_bits())
        );
        assert!(packed_embedding_rows(&prepared, &[1], &Device::Cpu).is_err());
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn packed_q4k_metal_lookup_and_tied_head_keep_cpu_bits() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let cpu = tiny_scale_q4k_embedding()?;
        let metal = copy_packed_tensor(&cpu, &device)?;
        let prepared = prepare_packed_embedding(metal, &device)?;
        assert!(prepared.device().is_cpu());
        assert_eq!(prepared.data()?.as_ref(), cpu.data()?.as_ref());
        let expected = packed_embedding_rows(&cpu, &[0, 0], &Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let actual = packed_embedding_rows(&prepared, &[0, 0], &device)?;
        assert!(actual.device().is_metal());
        assert_eq!(
            actual
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        // Exercise the Candle output-head fallback without a process-wide env override.
        let tied = qmatmul_source_on_device(Arc::clone(&prepared), &device)?;
        assert!(tied.device().is_metal());
        assert_eq!(tied.data()?.as_ref(), prepared.data()?.as_ref());
        let output =
            qtensor_to_qmatmul(tied)?.forward(&Tensor::ones((1, 256), DType::F32, &device)?)?;
        assert_eq!(output.dims(), &[1, 1]);
        assert!(output.to_vec2::<f32>()?[0][0].is_finite());
        let regular = Qwen35QMatMul::from_qtensor(prepared, &device)?;
        let output = regular.forward(&Tensor::ones((1, 256), DType::F32, &device)?)?;
        assert_eq!(output.dims(), &[1, 1]);
        assert!(output.to_vec2::<f32>()?[0][0].is_finite());
        Ok(())
    }

    fn embedding_source(rows: usize, hidden: usize) -> Result<Tensor> {
        let values = (0..rows * hidden)
            .map(|index| ((index * 17 % 113) as f32 - 56.0) / 31.0)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (rows, hidden), &Device::Cpu).map_err(Into::into)
    }

    #[test]
    fn packed_token_embedding_matches_full_dequantized_lookup() -> Result<()> {
        let source = embedding_source(4, 256)?;
        let packed = QTensor::quantize(&source, candle_core::quantized::GgmlDType::Q4K)?;
        let ids = Tensor::new(&[3u32, 0, 3], &Device::Cpu)?;

        let actual = packed.embedding(&ids)?;
        let expected = packed.dequantize(&Device::Cpu)?.index_select(&ids, 0)?;

        assert_eq!(actual.dims(), &[3, 256]);
        assert_eq!(actual.dtype(), DType::F32);
        assert_eq!(actual.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn packed_token_embedding_metal_matches_cpu() -> Result<()> {
        if candle_metal_kernels::metal::Device::all().is_empty() {
            return Ok(());
        }
        let source = embedding_source(4, 256)?;
        let cpu_packed = QTensor::quantize(&source, candle_core::quantized::GgmlDType::Q4K)?;
        let cpu_ids = Tensor::new(&[3u32, 0, 3], &Device::Cpu)?;
        let expected = cpu_packed.embedding(&cpu_ids)?.to_vec2::<f32>()?;

        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let metal_packed =
            QTensor::quantize_onto(&source, candle_core::quantized::GgmlDType::Q4K, &device)?;
        let ids = Tensor::new(&[3u32, 0, 3], &device)?;
        let actual = metal_packed.embedding(&ids)?;
        device.synchronize()?;
        let actual = actual.to_device(&Device::Cpu)?.to_vec2::<f32>()?;

        let max_diff = actual
            .iter()
            .flatten()
            .zip(expected.iter().flatten())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        anyhow::ensure!(max_diff <= 5e-3, "Metal embedding drifted by {max_diff}");
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn grouped_q6k_mlp_matches_separate_gate_up() -> Result<()> {
        if candle_metal_kernels::metal::Device::all().is_empty() {
            return Ok(());
        }
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let in_dim = 512;
        let intermediate_dim = 385;
        let make_projection = |phase: f32| -> Result<Arc<QTensor>> {
            let values = (0..intermediate_dim * in_dim)
                .map(|index| {
                    let x = index as f32;
                    (x * 0.0031 + phase).sin() * 0.2 + (x * 0.0073 - phase).cos() * 0.05
                })
                .collect::<Vec<_>>();
            let source = Tensor::from_vec(values, (intermediate_dim, in_dim), &Device::Cpu)?
                .to_device(&device)?;
            Ok(Arc::new(QTensor::quantize(
                &source,
                candle_core::quantized::GgmlDType::Q6K,
            )?))
        };
        let gate = make_projection(0.17)?;
        let up = make_projection(-0.29)?;
        let Some(grouped) =
            Qwen35QMatMul::try_grouped(&[Arc::clone(&gate), Arc::clone(&up)], &device)?
        else {
            return Ok(());
        };
        let gate = Qwen35QMatMul::from_qtensor(gate, &device)?;
        let up = Qwen35QMatMul::from_qtensor(up, &device)?;

        for (tokens, rank2) in [
            (1usize, true),
            (1, false),
            (3, true),
            (17, true),
            (17, false),
        ] {
            let values = (0..tokens * in_dim)
                .map(|index| {
                    let x = index as f32;
                    (x * 0.013).sin() * 0.4 - (x * 0.005).cos() * 0.1
                })
                .collect::<Vec<_>>();
            let input = Tensor::from_vec(values, (1, tokens, in_dim), &device)?;
            let input = if rank2 { input.squeeze(0)? } else { input };
            let gate_output = gate.forward(&input)?;
            let up_output = up.forward(&input)?;
            // Use the unfused model math, not silu_mul's separate fast-math
            // kernel. That kernel can reassociate the final multiply, while
            // grouped SwiGLU preserves Candle's rounded SiLU intermediate.
            let expected = (gate_output.silu()? * &up_output)?;
            let grouped_output = grouped.forward(&input)?;
            let separate_output = Tensor::cat(&[&gate_output, &up_output], D::Minus1)?;
            assert_eq!(
                grouped_output.flatten_all()?.to_vec1::<f32>()?,
                separate_output.flatten_all()?.to_vec1::<f32>()?,
                "grouped Q6_K projections tokens={tokens} rank2={rank2}"
            );
            let actual = if tokens == 1 {
                match grouped.forward_swiglu(&input)? {
                    Some(actual) => actual,
                    None => {
                        super::super::indexed_moe_metal::fused_swiglu(&grouped.forward(&input)?)?
                    }
                }
            } else {
                anyhow::ensure!(
                    grouped.forward_swiglu(&input)?.is_none(),
                    "fused Q6_K SwiGLU accepted multi-token input"
                );
                super::super::indexed_moe_metal::fused_swiglu(&grouped.forward(&input)?)?
            };
            let expected = expected.flatten_all()?.to_vec1::<f32>()?;
            let actual = actual.flatten_all()?.to_vec1::<f32>()?;
            let mut squared_error = 0f64;
            let mut squared_reference = 0f64;
            let mut max_abs = 0f32;
            for (&expected, &actual) in expected.iter().zip(&actual) {
                let error = (expected - actual).abs();
                max_abs = max_abs.max(error);
                squared_error += f64::from(error).powi(2);
                squared_reference += f64::from(expected).powi(2);
            }
            let relative_rmse = (squared_error / squared_reference.max(f64::EPSILON)).sqrt();
            anyhow::ensure!(actual.iter().all(|value| value.is_finite()));
            anyhow::ensure!(
                max_abs <= 2e-5 && relative_rmse <= 1e-6,
                "grouped Q6_K MLP tokens={tokens} rank2={rank2} drifted: max_abs={max_abs}, relative_rmse={relative_rmse}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_l2_normalize() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[3.0f32, 4.0], &device)?;
        let norm = l2_normalize(&x)?;
        let vals: Vec<f32> = norm.to_vec1()?;
        assert!((vals[0] - 0.6).abs() < 1e-5);
        assert!((vals[1] - 0.8).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_l2_normalize_2d() -> Result<()> {
        let device = Device::Cpu;
        // 2 heads, dim=2 each
        let x = Tensor::new(&[[3.0f32, 4.0], [0.0, 5.0]], &device)?;
        let norm = l2_normalize(&x)?;
        let vals: Vec<Vec<f32>> = norm.to_vec2()?;
        assert!((vals[0][0] - 0.6).abs() < 1e-5);
        assert!((vals[0][1] - 0.8).abs() < 1e-5);
        assert!((vals[1][0] - 0.0).abs() < 1e-3);
        assert!((vals[1][1] - 1.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_softplus() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[0.0f32, 1.0, -1.0, 25.0], &device)?;
        let result = softplus(&x)?;
        let vals: Vec<f32> = result.to_vec1()?;
        let expected = [std::f32::consts::LN_2, 1.3133, 0.3133, 25.0];
        for (got, exp) in vals.iter().zip(expected.iter()) {
            assert!(
                (got - exp).abs() < 0.01,
                "softplus: got {got}, expected {exp}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_conv1d_step() -> Result<()> {
        let device = Device::Cpu;
        let conv_dim = 4;
        let kernel_size = 3;

        // State: zeros initially
        let state = Tensor::zeros((conv_dim, kernel_size), DType::F32, &device)?;
        // Weight: [conv_dim, kernel_size] — all 1s for simplicity
        let weight = Tensor::ones((conv_dim, kernel_size), DType::F32, &device)?;
        // Input: [1, 2, 3, 4]
        let input = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?;

        let (output, new_state) = conv1d_step(&state, &input, &weight)?;

        // After first step: state has [0, 0, input] in each channel
        // output = sum across kernel dim: 0 + 0 + input[c] = input[c]
        let out_vals: Vec<f32> = output.to_vec1()?;
        assert_eq!(out_vals.len(), conv_dim);
        assert!((out_vals[0] - 1.0).abs() < 1e-5);
        assert!((out_vals[3] - 4.0).abs() < 1e-5);

        // Second step with same input
        let input2 = Tensor::new(&[5.0f32, 6.0, 7.0, 8.0], &device)?;
        let (output2, _) = conv1d_step(&new_state, &input2, &weight)?;
        let out_vals2: Vec<f32> = output2.to_vec1()?;
        // state was [0, 1, 5] for channel 0, so output = 0*1 + 1*1 + 5*1 = 6
        assert!((out_vals2[0] - 6.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn speculative_conv_snapshots_keep_each_row() -> Result<()> {
        let device = Device::Cpu;
        let initial = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device)?;
        let inputs = Tensor::new(&[[7.0f32, 8.0], [9.0, 10.0], [11.0, 12.0]], &device)?;
        let snapshots = deltanet_conv_snapshots(&initial, &inputs, 3)?;
        assert_eq!(
            snapshots.to_vec3::<f32>()?,
            vec![
                vec![vec![2.0, 3.0, 7.0], vec![5.0, 6.0, 8.0]],
                vec![vec![3.0, 7.0, 9.0], vec![6.0, 8.0, 10.0]],
                vec![vec![7.0, 9.0, 11.0], vec![8.0, 10.0, 12.0]],
            ]
        );
        Ok(())
    }

    #[test]
    fn test_deltanet_step() -> Result<()> {
        let device = Device::Cpu;
        let key_dim = 4;
        let value_dim = 4;

        // Start with empty state
        let s = Tensor::zeros((key_dim, value_dim), DType::F32, &device)?;
        let q = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0], &device)?;
        let k = Tensor::new(&[0.0f32, 1.0, 0.0, 0.0], &device)?;
        let v = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?;
        let beta = 1.0f32;
        let g = 0.0f32; // decay = exp(0) = 1.0 (no decay)

        let (output, new_s) = deltanet_step(&s, &q, &k, &v, beta, g)?;

        // Delta rule with S=0:
        //   decayed_S = 0, retrieved = 0
        //   delta = beta * (v - 0) = v
        //   S' = 0 + k outer v = k^T @ v (same as naive for S=0)
        //   output = S'^T @ q → q selects row 0, which is zeros
        let out_vals: Vec<f32> = output.to_vec1()?;
        assert!((out_vals[0]).abs() < 1e-5);

        // Query along k direction to retrieve stored value
        let q2 = Tensor::new(&[0.0f32, 1.0, 0.0, 0.0], &device)?;
        // beta=0 means no update (delta=0), just retrieve from state
        let (output2, _) = deltanet_step(&new_s, &q2, &k, &v, 0.0, 0.0)?;
        let out_vals2: Vec<f32> = output2.to_vec1()?;
        // S' unchanged (beta=0), q2 selects row 1 = [1,2,3,4]
        assert!((out_vals2[0] - 1.0).abs() < 1e-5);
        assert!((out_vals2[3] - 4.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_tile_heads() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device)?; // [2, 2]
        let tiled = tile_heads(&x, 3)?; // [6, 2]
        assert_eq!(tiled.dims(), &[6, 2]);
        let vals: Vec<Vec<f32>> = tiled.to_vec2()?;
        // Tiled: [h0, h1, h0, h1, h0, h1]
        assert_eq!(vals[0], vec![1.0, 2.0]); // h0
        assert_eq!(vals[1], vec![3.0, 4.0]); // h1
        assert_eq!(vals[2], vec![1.0, 2.0]); // h0
        assert_eq!(vals[3], vec![3.0, 4.0]); // h1
        assert_eq!(vals[4], vec![1.0, 2.0]); // h0
        assert_eq!(vals[5], vec![3.0, 4.0]); // h1
        Ok(())
    }

    #[test]
    fn test_partial_rotary_emb() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 8;
        let rope_dims = 4;

        // Create dummy input: [1, 1, 1, head_dim]
        let x = Tensor::ones((1, 1, 1, head_dim), DType::F32, &device)?;

        // Precompute for rope_dims
        let half = rope_dims / 2;
        let cos = Tensor::ones((4, half), DType::F32, &device)?;
        let sin = Tensor::zeros((4, half), DType::F32, &device)?;

        let result = apply_partial_rotary_emb(&x, 0, &cos, &sin, rope_dims)?;
        assert_eq!(result.dims(), &[1, 1, 1, head_dim]);

        // With cos=1, sin=0, RoPE should be identity → output = input
        let vals: Vec<f32> = result.flatten_all()?.to_vec1()?;
        for v in &vals {
            assert!((v - 1.0).abs() < 1e-5, "expected ~1.0, got {v}");
        }

        Ok(())
    }

    #[test]
    fn mrope_table_interleaves_temporal_height_and_width() -> Result<()> {
        let device = Device::Cpu;
        let table = Tensor::from_vec(
            (0..20).map(|value| value as f32).collect::<Vec<_>>(),
            (5, 4),
            &device,
        )?;
        let positions = MropePositionIds::new(vec![0, 1], vec![2, 3], vec![4, 0])?;

        let selected = select_mrope_table(&table, &positions, [2, 1, 1])?;
        assert_eq!(
            selected.to_vec2::<f32>()?,
            vec![vec![0.0, 9.0, 18.0, 3.0], vec![4.0, 13.0, 2.0, 7.0]]
        );
        Ok(())
    }

    #[test]
    fn fused_decode_position_never_collapses_mrope_axes() -> Result<()> {
        assert_eq!(fused_decode_text_position(1, 37, None), Some(37));
        assert_eq!(fused_decode_text_position(2, 37, None), None);

        let equal_axes = MropePositionIds::text(19, 1)?;
        assert_eq!(fused_decode_text_position(1, 37, Some(&equal_axes)), None);
        let image_axes = MropePositionIds::new(vec![19], vec![7], vec![11])?;
        assert_eq!(fused_decode_text_position(1, 37, Some(&image_axes)), None);
        Ok(())
    }

    #[test]
    fn qwen_mrope_sections_accept_gguf_trailing_zero() -> Result<()> {
        assert_eq!(
            parse_mrope_sections(&[11, 11, 10, 0], 64)?,
            Some([11, 11, 10])
        );
        assert!(parse_mrope_sections(&[11, 11, 10, 1], 64).is_err());
        assert!(parse_mrope_sections(&[11, 11, 9], 64).is_err());
        Ok(())
    }

    #[test]
    fn equal_axis_mrope_matches_text_rotary() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 12;
        let rope_dims = 8;
        let seq_len = 3;
        let start_pos = 2;
        let x = Tensor::from_vec(
            (0..head_dim * seq_len)
                .map(|value| value as f32 / 10.0)
                .collect::<Vec<_>>(),
            (1, 1, seq_len, head_dim),
            &device,
        )?;
        let (cos, sin) = precompute_rope_cos_sin(rope_dims, 10_000.0, 8, &device)?;

        let text = apply_partial_rotary_emb(&x, start_pos, &cos, &sin, rope_dims)?;
        let positions = MropePositionIds::text(start_pos, seq_len)?;
        let multimodal = apply_partial_rotary_emb_with_positions(
            &x,
            start_pos,
            &cos,
            &sin,
            rope_dims,
            Some(&positions),
            Some([2, 1, 1]),
        )?;

        let text = text.flatten_all()?.to_vec1::<f32>()?;
        let multimodal = multimodal.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(text.len(), multimodal.len());
        for (expected, actual) in text.iter().zip(&multimodal) {
            assert!((expected - actual).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn mrope_decode_delta_continues_after_compressed_image_grid() -> Result<()> {
        let prefill = MropePositionIds::new(
            vec![0, 1, 1, 1, 2, 2, 2, 3],
            vec![0, 1, 1, 2, 1, 2, 2, 3],
            vec![0, 1, 2, 1, 2, 1, 2, 3],
        )?;
        let delta = prefill.decode_delta(8)?;
        assert_eq!(delta, -4);

        let decode = MropePositionIds::text_with_delta(8, delta, 2)?;
        assert_eq!(decode.temporal(), &[4, 5]);
        assert_eq!(decode.height(), &[4, 5]);
        assert_eq!(decode.width(), &[4, 5]);
        Ok(())
    }
}

#[cfg(test)]
#[path = "qwen35_softplus_tests.rs"]
mod softplus_tests;
