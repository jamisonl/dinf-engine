use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use crate::qmatmul::QMatMul;
use anyhow::{Context, Result};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{D, DType, Device, IndexOp, Module, Tensor};

use super::activation::softplus;
#[cfg(target_os = "macos")]
use super::fused_route_metal::fused_sigmoid_topk_route;
#[cfg(any(target_os = "macos", feature = "cuda"))]
use super::indexed_moe::normalized_route_weights;
#[cfg(all(feature = "cuda", not(target_os = "macos")))]
use super::indexed_moe::{IndexedMoeMatMul, indexed_moe_available, weighted_route_sum};
#[cfg(target_os = "macos")]
use super::indexed_moe_metal::{MetalIndexedMoeMatMul, fused_weighted_route_sum};
use super::{KvCache, LayerCache, Mamba2Cache, ModelArchitecture, StageInput, StageOutput};
use crate::gguf_loader::StageTensors;
use crate::turboquant::TurboQuantConfig;

static MAMBA_PROFILE_NS: AtomicU64 = AtomicU64::new(0);
static MAMBA_NORM_PROFILE_NS: AtomicU64 = AtomicU64::new(0);
static MAMBA_INPUT_PROJ_PROFILE_NS: AtomicU64 = AtomicU64::new(0);
static MAMBA_INPUT_LAYOUT_PROFILE_NS: AtomicU64 = AtomicU64::new(0);
static MAMBA_SCAN_PROFILE_NS: AtomicU64 = AtomicU64::new(0);
static MAMBA_OUTPUT_PROFILE_NS: AtomicU64 = AtomicU64::new(0);
static MAMBA_PREFILL_PROFILE_CALLS: AtomicU64 = AtomicU64::new(0);
static DENSE_PROFILE_NS: AtomicU64 = AtomicU64::new(0);
static MOE_PROFILE_NS: AtomicU64 = AtomicU64::new(0);
static MOE_ROUTE_PROFILE_NS: AtomicU64 = AtomicU64::new(0);
static MOE_ROUTE_PLAN_PROFILE_NS: AtomicU64 = AtomicU64::new(0);
static MOE_INDEXED_UP_PROFILE_NS: AtomicU64 = AtomicU64::new(0);
static MOE_INDEXED_ACT_PROFILE_NS: AtomicU64 = AtomicU64::new(0);
static MOE_INDEXED_DOWN_PROFILE_NS: AtomicU64 = AtomicU64::new(0);
static MOE_INDEXED_SUM_PROFILE_NS: AtomicU64 = AtomicU64::new(0);
static MOE_SHARED_PROFILE_NS: AtomicU64 = AtomicU64::new(0);
static MOE_PREFILL_PROFILE_CALLS: AtomicU64 = AtomicU64::new(0);
static ATTENTION_PROFILE_NS: AtomicU64 = AtomicU64::new(0);
static MAMBA_PROFILE_CALLS: AtomicU64 = AtomicU64::new(0);
static DENSE_PROFILE_CALLS: AtomicU64 = AtomicU64::new(0);
static MOE_PROFILE_CALLS: AtomicU64 = AtomicU64::new(0);
static ATTENTION_PROFILE_CALLS: AtomicU64 = AtomicU64::new(0);
static MOE_ROUTE_STATS_PLANS: AtomicU64 = AtomicU64::new(0);
static MOE_ROUTE_STATS_ROUTES: AtomicU64 = AtomicU64::new(0);
static MOE_ROUTE_STATS_OCCUPIED_EXPERTS: AtomicU64 = AtomicU64::new(0);
static MOE_ROUTE_STATS_TILES: AtomicU64 = AtomicU64::new(0);
static MOE_ROUTE_STATS_TILES_64: AtomicU64 = AtomicU64::new(0);
static MOE_ROUTE_STATS_MULTI_TILE_EXPERTS: AtomicU64 = AtomicU64::new(0);
static MOE_ROUTE_STATS_MAX_ROUTES: AtomicU64 = AtomicU64::new(0);
static PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();
#[cfg(target_os = "macos")]
static ROUTE_STATS_ENABLED: OnceLock<bool> = OnceLock::new();
#[cfg(target_os = "macos")]
static NEMOTRON_METAL4_QMM_LOGGED: OnceLock<()> = OnceLock::new();

fn layer_profile_enabled() -> bool {
    *PROFILE_ENABLED.get_or_init(|| std::env::var_os("DINF_PROFILE_LAYERS").is_some())
}

#[cfg(target_os = "macos")]
fn route_stats_enabled() -> bool {
    *ROUTE_STATS_ENABLED.get_or_init(|| std::env::var_os("DINF_PROFILE_MOE_ROUTES").is_some())
}

/// Reset the process-local Nemotron-H layer profiling counters.
pub fn reset_layer_profile() {
    for counter in [
        &MAMBA_PROFILE_NS,
        &MAMBA_NORM_PROFILE_NS,
        &MAMBA_INPUT_PROJ_PROFILE_NS,
        &MAMBA_INPUT_LAYOUT_PROFILE_NS,
        &MAMBA_SCAN_PROFILE_NS,
        &MAMBA_OUTPUT_PROFILE_NS,
        &MAMBA_PREFILL_PROFILE_CALLS,
        &DENSE_PROFILE_NS,
        &MOE_PROFILE_NS,
        &MOE_ROUTE_PROFILE_NS,
        &MOE_ROUTE_PLAN_PROFILE_NS,
        &MOE_INDEXED_UP_PROFILE_NS,
        &MOE_INDEXED_ACT_PROFILE_NS,
        &MOE_INDEXED_DOWN_PROFILE_NS,
        &MOE_INDEXED_SUM_PROFILE_NS,
        &MOE_SHARED_PROFILE_NS,
        &MOE_PREFILL_PROFILE_CALLS,
        &ATTENTION_PROFILE_NS,
        &MAMBA_PROFILE_CALLS,
        &DENSE_PROFILE_CALLS,
        &MOE_PROFILE_CALLS,
        &ATTENTION_PROFILE_CALLS,
        &MOE_ROUTE_STATS_PLANS,
        &MOE_ROUTE_STATS_ROUTES,
        &MOE_ROUTE_STATS_OCCUPIED_EXPERTS,
        &MOE_ROUTE_STATS_TILES,
        &MOE_ROUTE_STATS_TILES_64,
        &MOE_ROUTE_STATS_MULTI_TILE_EXPERTS,
        &MOE_ROUTE_STATS_MAX_ROUTES,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

/// Print the process-local Nemotron-H layer profiling counters to stderr.
pub fn report_layer_profile() {
    let per_call_ms = |time: &AtomicU64, calls: &AtomicU64| {
        time.load(Ordering::Relaxed) as f64
            / calls.load(Ordering::Relaxed).max(1) as f64
            / 1_000_000.0
    };
    let mamba_input_proj = per_call_ms(&MAMBA_INPUT_PROJ_PROFILE_NS, &MAMBA_PREFILL_PROFILE_CALLS);
    let mamba_input_layout =
        per_call_ms(&MAMBA_INPUT_LAYOUT_PROFILE_NS, &MAMBA_PREFILL_PROFILE_CALLS);
    eprintln!(
        "DINF_NEMOTRON_PROFILE calls={{mamba:{},dense:{},moe:{},attention:{}}} per_block_ms={{mamba:{:.3},dense:{:.3},moe:{:.3},attention:{:.3}}}",
        MAMBA_PROFILE_CALLS.load(Ordering::Relaxed),
        DENSE_PROFILE_CALLS.load(Ordering::Relaxed),
        MOE_PROFILE_CALLS.load(Ordering::Relaxed),
        ATTENTION_PROFILE_CALLS.load(Ordering::Relaxed),
        per_call_ms(&MAMBA_PROFILE_NS, &MAMBA_PROFILE_CALLS),
        per_call_ms(&DENSE_PROFILE_NS, &DENSE_PROFILE_CALLS),
        per_call_ms(&MOE_PROFILE_NS, &MOE_PROFILE_CALLS),
        per_call_ms(&ATTENTION_PROFILE_NS, &ATTENTION_PROFILE_CALLS),
    );
    eprintln!(
        "DINF_NEMOTRON_MAMBA_PREFILL_PROFILE calls={} per_block_ms={{norm:{:.3},input:{:.3},input_proj:{:.3},input_layout:{:.3},scan:{:.3},output:{:.3}}}",
        MAMBA_PREFILL_PROFILE_CALLS.load(Ordering::Relaxed),
        per_call_ms(&MAMBA_NORM_PROFILE_NS, &MAMBA_PREFILL_PROFILE_CALLS),
        mamba_input_proj + mamba_input_layout,
        mamba_input_proj,
        mamba_input_layout,
        per_call_ms(&MAMBA_SCAN_PROFILE_NS, &MAMBA_PREFILL_PROFILE_CALLS),
        per_call_ms(&MAMBA_OUTPUT_PROFILE_NS, &MAMBA_PREFILL_PROFILE_CALLS),
    );
    eprintln!(
        "DINF_NEMOTRON_MOE_PREFILL_PROFILE calls={} per_block_ms={{route:{:.3},route_plan:{:.3},indexed_up:{:.3},indexed_act:{:.3},indexed_down:{:.3},indexed_sum:{:.3},shared:{:.3}}}",
        MOE_PREFILL_PROFILE_CALLS.load(Ordering::Relaxed),
        per_call_ms(&MOE_ROUTE_PROFILE_NS, &MOE_PREFILL_PROFILE_CALLS),
        per_call_ms(&MOE_ROUTE_PLAN_PROFILE_NS, &MOE_PREFILL_PROFILE_CALLS),
        per_call_ms(&MOE_INDEXED_UP_PROFILE_NS, &MOE_PREFILL_PROFILE_CALLS),
        per_call_ms(&MOE_INDEXED_ACT_PROFILE_NS, &MOE_PREFILL_PROFILE_CALLS),
        per_call_ms(&MOE_INDEXED_DOWN_PROFILE_NS, &MOE_PREFILL_PROFILE_CALLS),
        per_call_ms(&MOE_INDEXED_SUM_PROFILE_NS, &MOE_PREFILL_PROFILE_CALLS),
        per_call_ms(&MOE_SHARED_PROFILE_NS, &MOE_PREFILL_PROFILE_CALLS),
    );
    let route_plans = MOE_ROUTE_STATS_PLANS.load(Ordering::Relaxed);
    if route_plans != 0 {
        let routes = MOE_ROUTE_STATS_ROUTES.load(Ordering::Relaxed);
        let tiles = MOE_ROUTE_STATS_TILES.load(Ordering::Relaxed);
        eprintln!(
            "DINF_NEMOTRON_MOE_ROUTE_PROFILE plans={} per_plan={{routes:{:.1},occupied_experts:{:.1},tiles32:{:.1},tiles64:{:.1},multi_tile_experts:{:.1}}} tile_utilization={:.3} max_routes_per_expert={}",
            route_plans,
            routes as f64 / route_plans as f64,
            MOE_ROUTE_STATS_OCCUPIED_EXPERTS.load(Ordering::Relaxed) as f64 / route_plans as f64,
            tiles as f64 / route_plans as f64,
            MOE_ROUTE_STATS_TILES_64.load(Ordering::Relaxed) as f64 / route_plans as f64,
            MOE_ROUTE_STATS_MULTI_TILE_EXPERTS.load(Ordering::Relaxed) as f64 / route_plans as f64,
            routes as f64 / (tiles.max(1) * 32) as f64,
            MOE_ROUTE_STATS_MAX_ROUTES.load(Ordering::Relaxed),
        );
    }
}

fn record_profile_region(
    device: &Device,
    started: Instant,
    elapsed: &AtomicU64,
    calls: &AtomicU64,
) -> Result<()> {
    device.synchronize()?;
    elapsed.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    calls.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

fn record_profile_time(device: &Device, started: Instant, elapsed: &AtomicU64) -> Result<()> {
    device.synchronize()?;
    elapsed.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(())
}

fn trace_layer_tensor(label: &str, layer: isize, tensor: &Tensor, seq_len: usize) -> Result<()> {
    if std::env::var_os("DINF_TRACE_NEMOTRON_LAYERS").is_none() {
        return Ok(());
    }
    if let Some(requested) = std::env::var("DINF_TRACE_NEMOTRON_SEQ_LEN")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        && requested != seq_len
    {
        return Ok(());
    }

    let values = tensor
        .i((0, seq_len - 1, ..))?
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let head = values.iter().take(6).copied().collect::<Vec<_>>();
    let sum = values.iter().map(|value| *value as f64).sum::<f64>();
    let sum_sq = values
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum::<f64>();
    eprintln!(
        "DINF_NEMOTRON_LAYER label={label} layer={layer} seq_len={seq_len} head={head:?} sum={sum:.9} sum_sq={sum_sq:.9}"
    );
    Ok(())
}

// ── Nemotron-H architecture ─────────────────────────────────────────────
//
// Hybrid architecture with three block types:
//   - Mamba-2 SSM (23 layers): fixed-size recurrent state
//   - MoE FFN (23 layers): sigmoid gating, top-k routing, ReLU² activation
//   - GQA Attention (6 layers): no RoPE, standard softmax attention
//
// Each layer has exactly ONE computation type (no attention+FFN combos).

/// A loaded Nemotron-H hybrid stage for pipeline execution.
pub struct NemotronStage {
    embedding: Option<Tensor>,
    blocks: Vec<NemotronBlock>,
    output_norm: Option<RmsNorm>,
    lm_head: Option<QMatMul>,
    is_head: bool,
    is_tail: bool,
    hidden_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    device: Device,
}

enum NemotronBlock {
    Mamba2(Mamba2Block),
    Dense(DenseBlock),
    MoE(MoeBlock),
    Attention(AttentionBlock),
}

// ── Mamba-2 block ────────────────────────────────────────────────────────

struct Mamba2Block {
    norm: RmsNorm,
    ssm_in: NemotronQMatMul, // [hidden, 2*d_inner + 2*n_group*d_state + n_heads]
    conv1d_weight: Tensor,   // [d_conv, kernel_size] (depthwise, squeezed)
    conv1d_bias: Tensor,     // [d_conv], zero-filled when absent
    dt_bias: Tensor,         // [n_heads, 1]
    a: Tensor,               // [n_heads, 1] (already -exp(A_log))
    d: Tensor,               // [n_heads, 1] (feedthrough/skip)
    ssm_norm: Tensor,        // [n_group, group_size] (group norm weight, NOT RmsNorm)
    ssm_out: NemotronQMatMul, // [d_inner, hidden]
    d_inner: usize,          // 4096
    d_state: usize,          // 128
    n_heads: usize,          // 64
    n_group: usize,          // 8
    head_dim: usize,         // 64 (d_inner / n_heads)
    d_conv: usize,           // d_inner + 2*n_group*d_state
    kernel_size: usize,      // 4
}

enum NemotronQMatMul {
    Candle(QMatMul),
    #[cfg(target_os = "macos")]
    MetalTensor(super::tensor_qmatmul_metal::MetalTensorQMatMul),
}

impl NemotronQMatMul {
    fn from_qtensor(qt: Arc<QTensor>, _device: &Device) -> Result<Self> {
        #[cfg(target_os = "macos")]
        if std::env::var_os("DINF_DISABLE_METAL4_QMM").is_none()
            && let Some(matmul) = super::tensor_qmatmul_metal::MetalTensorQMatMul::try_from_qtensor(
                Arc::clone(&qt),
                _device,
            )?
        {
            NEMOTRON_METAL4_QMM_LOGGED.get_or_init(|| {
                tracing::info!(
                    "Metal 4 tensor quantized matmul enabled for Nemotron prompt projections"
                );
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

    fn forward_relu_squared(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Candle(matmul) => relu_squared(&matmul.forward(x)?),
            #[cfg(target_os = "macos")]
            Self::MetalTensor(matmul)
                if std::env::var_os("DINF_DISABLE_NEMOTRON_SHARED_RELU2_FUSION").is_none() =>
            {
                matmul.forward_relu_squared(x).map_err(Into::into)
            }
            #[cfg(target_os = "macos")]
            Self::MetalTensor(matmul) => relu_squared(&matmul.forward(x)?),
        }
    }

    fn forward_relu_squared_down(&self, x: &Tensor, down: &Self) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let (Some(up), Some(down)) = (self.candle(), down.candle()) {
            let projected = match super::dense_qmatmul_cuda::forward_one_to_f32(x, up)? {
                Some(projected) => projected,
                None => up.forward(x)?,
            };
            if let Some(output) = super::dense_qmatmul_cuda::relu_squared_down(&projected, down)? {
                return Ok(output);
            }
            return down.forward(&relu_squared(&projected)?).map_err(Into::into);
        }
        let activated = self.forward_relu_squared(x)?;
        down.forward(&activated)
    }
}

// ── MoE block ────────────────────────────────────────────────────────────

struct MoeBlock {
    norm: RmsNorm,
    router: Tensor,              // [n_experts, hidden] (dequantized gate weights)
    expert_bias: Option<Tensor>, // [n_experts] (e_score_correction)
    experts: NemotronExperts,
    shared_expert: Option<SharedExpert>,
    num_active: usize,  // top-6
    weights_scale: f64, // 2.5
    weights_norm: bool, // true
}

struct DenseBlock {
    norm: RmsNorm,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

struct NemotronExpert {
    up_proj: QMatMul,   // [hidden, expert_dim]
    down_proj: QMatMul, // [expert_dim, hidden]
}

struct SharedExpert {
    up_proj: NemotronQMatMul,
    down_proj: NemotronQMatMul,
}

#[cfg(target_os = "macos")]
struct MetalIndexedNemotronExperts {
    up_proj: MetalIndexedMoeMatMul,
    down_proj: MetalIndexedMoeMatMul,
}

#[cfg(all(feature = "cuda", not(target_os = "macos")))]
struct CudaIndexedNemotronExperts {
    up_proj: IndexedMoeMatMul,
    down_proj: IndexedMoeMatMul,
}

enum NemotronExperts {
    Split(Vec<NemotronExpert>),
    #[cfg(target_os = "macos")]
    MetalIndexed(MetalIndexedNemotronExperts),
    #[cfg(all(feature = "cuda", not(target_os = "macos")))]
    CudaIndexed(CudaIndexedNemotronExperts),
}

impl NemotronExperts {
    fn len(&self) -> usize {
        match self {
            Self::Split(experts) => experts.len(),
            #[cfg(target_os = "macos")]
            Self::MetalIndexed(experts) => experts.up_proj.num_experts(),
            #[cfg(all(feature = "cuda", not(target_os = "macos")))]
            Self::CudaIndexed(experts) => experts.up_proj.num_experts(),
        }
    }

    #[cfg(any(target_os = "macos", feature = "cuda"))]
    fn is_indexed(&self) -> bool {
        match self {
            Self::Split(_) => false,
            #[cfg(target_os = "macos")]
            Self::MetalIndexed(_) => true,
            #[cfg(all(feature = "cuda", not(target_os = "macos")))]
            Self::CudaIndexed(_) => true,
        }
    }
}

// ── Attention block ──────────────────────────────────────────────────────

struct AttentionBlock {
    norm: RmsNorm,
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    num_heads: usize,    // 32
    num_kv_heads: usize, // 2
    head_dim: usize,     // 128
    attn_scale: f64,
    quant_config: Option<Arc<TurboQuantConfig>>,
}

// ── Shared components ────────────────────────────────────────────────────

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
}

// ── Primitive functions ──────────────────────────────────────────────────

/// ReLU²(x) = max(0, x)²
fn relu_squared(x: &Tensor) -> Result<Tensor> {
    let relu = x.relu()?;
    Ok((&relu * &relu)?)
}

/// Causal depthwise conv1d step: update sliding window state and produce output.
///
/// state: [d_conv, kernel_size], input: [d_conv]
/// weight: [d_conv, kernel_size]
fn conv1d_step(state: &Tensor, input: &Tensor, weight: &Tensor) -> Result<(Tensor, Tensor)> {
    let (_d_conv, kernel_size) = state.dims2()?;

    let shifted = if kernel_size > 1 {
        let old = state.narrow(1, 1, kernel_size - 1)?;
        Tensor::cat(&[&old, &input.unsqueeze(1)?], 1)?
    } else {
        input.unsqueeze(1)?
    };

    let output = (&shifted * weight)?.sum(1)?; // [d_conv]
    Ok((output, shifted))
}

/// Batched Mamba-2 SSM step for every head.
///
/// state: [n_heads, d_state, head_dim]
/// x: [n_heads, head_dim]
/// B, C: [n_group, d_state]
/// a, dt, d: [n_heads]
///
/// Returns (output [n_heads, head_dim], new_state).
struct Mamba2SsmStep<'a> {
    state: &'a Tensor,
    x: &'a Tensor,
    b: &'a Tensor,
    c: &'a Tensor,
    a: &'a Tensor,
    dt_raw: &'a Tensor,
    dt_bias: &'a Tensor,
    d: &'a Tensor,
    n_group: usize,
}

fn mamba2_ssm_step_batched(step: Mamba2SsmStep<'_>) -> Result<(Tensor, Tensor)> {
    let Mamba2SsmStep {
        state,
        x,
        b,
        c,
        a,
        dt_raw,
        dt_bias,
        d,
        n_group,
    } = step;
    #[cfg(target_os = "macos")]
    if state.device().is_metal() {
        return super::nemotron_mamba::mamba2_ssm_step_metal(
            state, x, dt_raw, a, b, c, d, dt_bias, n_group,
        );
    }

    let (n_heads, d_state, head_dim) = state.dims3()?;
    let heads_per_group = n_heads / n_group;
    let dt = softplus(&(dt_raw + dt_bias)?)?;

    // Repeat each B/C group for its contiguous run of heads. Materializing this
    // small view is much cheaper than synchronizing 128 individual head slices.
    let b = b
        .unsqueeze(1)?
        .expand((n_group, heads_per_group, d_state))?
        .contiguous()?
        .reshape((n_heads, d_state))?;
    let c = c
        .unsqueeze(1)?
        .expand((n_group, heads_per_group, d_state))?
        .contiguous()?
        .reshape((n_heads, d_state))?;

    // state' = state * exp(A * dt) + B * (x * dt)
    let decay = (a * &dt)?.exp()?.reshape((n_heads, 1, 1))?;
    let x_dt = x.broadcast_mul(&dt.reshape((n_heads, 1))?)?;
    let update = b
        .reshape((n_heads, d_state, 1))?
        .broadcast_mul(&x_dt.reshape((n_heads, 1, head_dim))?)?;
    let new_state = (state.broadcast_mul(&decay)? + update)?;

    // y = C @ state' + D * x
    let y = new_state
        .broadcast_mul(&c.reshape((n_heads, d_state, 1))?)?
        .sum(1)?;
    let output = (y + x.broadcast_mul(&d.reshape((n_heads, 1))?)?)?;

    Ok((output, new_state))
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

// ── Mamba-2 block forward ────────────────────────────────────────────────

impl Mamba2Block {
    #[cfg(target_os = "macos")]
    fn forward_metal_prefill(&self, x: &Tensor, cache: &mut Option<Mamba2Cache>) -> Result<Tensor> {
        let input_dtype = x.dtype();
        let (batch_size, seq_len, _hidden_dim) = x.dims3()?;
        anyhow::ensure!(
            batch_size == 1,
            "Nemotron Mamba2 prefill requires batch size 1"
        );

        let profile = layer_profile_enabled();
        if profile {
            MAMBA_PREFILL_PROFILE_CALLS.fetch_add(1, Ordering::Relaxed);
            x.device().synchronize()?;
        }
        let started = Instant::now();
        let residual = x.to_dtype(DType::F32)?;
        let normed = self.norm.forward(x)?;
        if profile {
            record_profile_time(x.device(), started, &MAMBA_NORM_PROFILE_NS)?;
        }
        let started = Instant::now();
        let combined = self.ssm_in.forward(&normed)?;
        if profile {
            record_profile_time(x.device(), started, &MAMBA_INPUT_PROJ_PROFILE_NS)?;
        }
        let started = Instant::now();
        let combined = combined.squeeze(0)?.to_dtype(DType::F32)?;
        let strided_inputs = std::env::var_os("DINF_DISABLE_NEMOTRON_STRIDED_INPUTS").is_none();
        let z = combined.narrow(1, 0, self.d_inner)?;
        let conv_input = combined.narrow(1, self.d_inner, self.d_conv)?;
        let dt_raw = combined.narrow(1, self.d_inner + self.d_conv, self.n_heads)?;
        let (z, conv_input, dt_raw) = if strided_inputs {
            (z, conv_input, dt_raw)
        } else {
            (
                z.contiguous()?,
                conv_input.contiguous()?,
                dt_raw.contiguous()?,
            )
        };
        if profile {
            record_profile_time(x.device(), started, &MAMBA_INPUT_LAYOUT_PROFILE_NS)?;
        }

        let started = Instant::now();
        let (conv_state, ssm_state) = match cache.take() {
            Some(cache) => (cache.conv_state, cache.ssm_state),
            None => (
                Tensor::zeros((self.d_conv, self.kernel_size), DType::F32, x.device())?,
                Tensor::zeros(
                    (self.n_heads, self.d_state, self.head_dim),
                    DType::F32,
                    x.device(),
                )?,
            ),
        };
        let (y, conv_state, ssm_state) = super::nemotron_mamba::prefill(
            &conv_input,
            &conv_state,
            &ssm_state,
            &self.conv1d_weight,
            &self.conv1d_bias,
            &dt_raw,
            &self.a,
            &self.d,
            &self.dt_bias,
            self.n_heads,
            self.d_state,
            self.head_dim,
            self.n_group,
            self.kernel_size,
        )?;
        if profile {
            record_profile_time(x.device(), started, &MAMBA_SCAN_PROFILE_NS)?;
        }
        *cache = Some(Mamba2Cache {
            conv_state,
            ssm_state,
        });

        let started = Instant::now();
        let y = y.reshape((seq_len, self.d_inner))?;
        let normalized = if std::env::var_os("DINF_DISABLE_NEMOTRON_FUSED_GATE_NORM").is_none() {
            super::nemotron_mamba::fused_gate_group_norm(&z, &y, &self.ssm_norm)?.reshape((
                1,
                seq_len,
                self.d_inner,
            ))?
        } else {
            let gate = candle_nn::Activation::Silu.forward(&z)?;
            let gated = (&gate * &y)?;
            let group_size = self.d_inner / self.n_group;
            let grouped = gated.reshape((seq_len, self.n_group, group_size))?;
            let variance = (&grouped * &grouped)?.mean_keepdim(D::Minus1)?;
            let normalized = grouped.broadcast_div(&(variance + 1e-5)?.sqrt()?)?;
            normalized
                .broadcast_mul(&self.ssm_norm)?
                .reshape((1, seq_len, self.d_inner))?
        };
        let projected = self.ssm_out.forward(&normalized)?.to_dtype(DType::F32)?;
        let result = (&residual + &projected)?
            .to_dtype(input_dtype)
            .map_err(anyhow::Error::from)?;
        if profile {
            record_profile_time(x.device(), started, &MAMBA_OUTPUT_PROFILE_NS)?;
        }
        Ok(result)
    }

    #[cfg(feature = "cuda")]
    fn forward_cuda(&self, x: &Tensor, cache: &mut Option<Mamba2Cache>) -> Result<Tensor> {
        let input_dtype = x.dtype();
        let (batch_size, seq_len, _hidden_dim) = x.dims3()?;
        anyhow::ensure!(
            batch_size == 1,
            "Nemotron CUDA Mamba2 requires batch size 1"
        );

        // Keep the existing prompt profile meaningful. Decode uses this same
        // path, but is deliberately excluded from the prefill-only counters.
        let profile = seq_len > 1 && layer_profile_enabled();
        if profile {
            MAMBA_PREFILL_PROFILE_CALLS.fetch_add(1, Ordering::Relaxed);
            x.device().synchronize()?;
        }
        let started = Instant::now();
        let residual = x.to_dtype(DType::F32)?;
        let normed = self.norm.forward(x)?;
        if profile {
            record_profile_time(x.device(), started, &MAMBA_NORM_PROFILE_NS)?;
        }

        let started = Instant::now();
        let combined = self.ssm_in.forward(&normed)?;
        if profile {
            record_profile_time(x.device(), started, &MAMBA_INPUT_PROJ_PROFILE_NS)?;
        }
        let started = Instant::now();
        let combined = combined.squeeze(0)?.to_dtype(DType::F32)?;
        let z = combined.narrow(1, 0, self.d_inner)?;
        let conv_input = combined.narrow(1, self.d_inner, self.d_conv)?;
        let dt_raw = combined.narrow(1, self.d_inner + self.d_conv, self.n_heads)?;
        let (z, conv_input, dt_raw) =
            if std::env::var_os("DINF_DISABLE_NEMOTRON_STRIDED_INPUTS").is_none() {
                (z, conv_input, dt_raw)
            } else {
                (
                    z.contiguous()?,
                    conv_input.contiguous()?,
                    dt_raw.contiguous()?,
                )
            };
        if profile {
            record_profile_time(x.device(), started, &MAMBA_INPUT_LAYOUT_PROFILE_NS)?;
        }

        let started = Instant::now();
        let (conv_state, ssm_state) = match cache.take() {
            Some(cache) => (cache.conv_state, cache.ssm_state),
            None => (
                Tensor::zeros((self.d_conv, self.kernel_size), DType::F32, x.device())?,
                Tensor::zeros(
                    (self.n_heads, self.d_state, self.head_dim),
                    DType::F32,
                    x.device(),
                )?,
            ),
        };
        let (normalized, conv_state, ssm_state) = super::nemotron_mamba_cuda::forward(
            &z,
            &conv_input,
            &dt_raw,
            &conv_state,
            &ssm_state,
            &self.conv1d_weight,
            &self.conv1d_bias,
            &self.a,
            &self.d,
            &self.dt_bias,
            &self.ssm_norm,
            self.n_heads,
            self.d_state,
            self.head_dim,
            self.n_group,
            self.kernel_size,
        )?;
        if profile {
            record_profile_time(x.device(), started, &MAMBA_SCAN_PROFILE_NS)?;
        }
        *cache = Some(Mamba2Cache {
            conv_state,
            ssm_state,
        });

        let started = Instant::now();
        let normalized = normalized.reshape((1, seq_len, self.d_inner))?;
        let projected = self.ssm_out.forward(&normalized)?.to_dtype(DType::F32)?;
        let result = (&residual + &projected)?
            .to_dtype(input_dtype)
            .map_err(anyhow::Error::from)?;
        if profile {
            record_profile_time(x.device(), started, &MAMBA_OUTPUT_PROFILE_NS)?;
        }
        Ok(result)
    }

    fn forward(
        &self,
        x: &Tensor,
        cache: &mut Option<Mamba2Cache>,
        _seq_len: usize,
        _start_pos: usize,
    ) -> Result<Tensor> {
        let input_dtype = x.dtype();
        let (_b_sz, seq_len, _hidden_dim) = x.dims3()?;

        #[cfg(feature = "cuda")]
        if x.device().is_cuda()
            && std::env::var_os("DINF_DISABLE_NEMOTRON_MAMBA_CUDA").is_none()
            && (seq_len == 1 || std::env::var_os("DINF_DISABLE_NEMOTRON_PREFILL_CUDA").is_none())
            && super::nemotron_mamba_cuda::supports(
                seq_len,
                self.d_conv,
                self.kernel_size,
                self.n_heads,
                self.d_state,
                self.head_dim,
                self.n_group,
            )
        {
            return self.forward_cuda(x, cache);
        }

        #[cfg(target_os = "macos")]
        if seq_len > 1
            && x.device().is_metal()
            && self.kernel_size <= 8
            && self.d_state <= 256
            && std::env::var_os("DINF_DISABLE_NEMOTRON_PREFILL_METAL").is_none()
        {
            return self.forward_metal_prefill(x, cache);
        }

        let x = x.to_dtype(DType::F32)?;
        let (mut conv_state, mut ssm_state) = match cache.take() {
            Some(c) => (c.conv_state, c.ssm_state),
            None => {
                let conv_state =
                    Tensor::zeros((self.d_conv, self.kernel_size), DType::F32, x.device())?;
                let ssm_state = Tensor::zeros(
                    (self.n_heads, self.d_state, self.head_dim),
                    DType::F32,
                    x.device(),
                )?;
                (conv_state, ssm_state)
            }
        };
        let dt_bias = self.dt_bias.flatten_all()?;
        let a = self.a.flatten_all()?;
        let d = self.d.flatten_all()?;
        let mut token_outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t = x.i((.., t, ..))?.contiguous()?; // [1, hidden]

            // Pre-norm
            let normed = self.norm.forward(&x_t.unsqueeze(1)?)?.squeeze(1)?;

            // 1. Project to combined ssm_in
            let combined = self.ssm_in.forward(&normed)?; // [1, 2*d_inner + 2*n_group*d_state + n_heads]
            let combined = combined.squeeze(0)?.to_dtype(DType::F32)?;

            // 2. Split: z[d_inner], xBC[d_conv], dt[n_heads]
            let z = combined.narrow(0, 0, self.d_inner)?;
            let xbc = combined.narrow(0, self.d_inner, self.d_conv)?;
            let dt_raw = combined.narrow(0, self.d_inner + self.d_conv, self.n_heads)?;

            // 3. Conv1d step
            let (conv_out, new_conv_state) = conv1d_step(&conv_state, &xbc, &self.conv1d_weight)?;
            conv_state = new_conv_state;

            let conv_out = (conv_out + &self.conv1d_bias)?;

            // SiLU activation on conv output
            let conv_out = candle_nn::Activation::Silu.forward(&conv_out)?;

            // 4. Split conv output: x_ssm[d_inner], B[n_group*d_state], C[n_group*d_state]
            let x_ssm = conv_out.narrow(0, 0, self.d_inner)?;
            let b_raw = conv_out.narrow(0, self.d_inner, self.n_group * self.d_state)?;
            let c_raw = conv_out.narrow(
                0,
                self.d_inner + self.n_group * self.d_state,
                self.n_group * self.d_state,
            )?;

            // 5. SSM step for all heads in one vectorized graph. The Metal path
            // fuses dt softplus, state update, state reduction, and D skip.
            let x_heads = x_ssm.reshape((self.n_heads, self.head_dim))?;
            let b_groups = b_raw.reshape((self.n_group, self.d_state))?;
            let c_groups = c_raw.reshape((self.n_group, self.d_state))?;
            let (y, new_ssm_state) = mamba2_ssm_step_batched(Mamba2SsmStep {
                state: &ssm_state,
                x: &x_heads,
                b: &b_groups,
                c: &c_groups,
                a: &a,
                dt_raw: &dt_raw,
                dt_bias: &dt_bias,
                d: &d,
                n_group: self.n_group,
            })?;
            ssm_state = new_ssm_state;

            // 7. Output gating: SiLU(z) * y (gate FIRST, then norm)
            let gate = candle_nn::Activation::Silu.forward(&z)?;
            let y_flat = y.reshape(self.d_inner)?;
            let gated = (&gate * &y_flat)?;

            // 8. Group RMS norm on gated output
            // ssm_norm weight is [n_group, group_size] where group_size = d_inner / n_group
            let group_size = self.d_inner / self.n_group;
            let y_grouped = gated.reshape((self.n_group, group_size))?;
            let variance = (&y_grouped * &y_grouped)?.mean_keepdim(D::Minus1)?;
            let y_normed = y_grouped.broadcast_div(&(variance + 1e-5)?.sqrt()?)?;
            let gated = (y_normed * &self.ssm_norm)?.reshape(self.d_inner)?;

            // 9. Output projection + residual
            let projected = self
                .ssm_out
                .forward(&gated.unsqueeze(0)?)?
                .squeeze(0)?
                .to_dtype(DType::F32)?;
            let new_hidden = (&x_t + &projected.unsqueeze(0)?)?;
            token_outputs.push(new_hidden.unsqueeze(1)?);
        }

        *cache = Some(Mamba2Cache {
            conv_state,
            ssm_state,
        });

        Tensor::cat(&token_outputs, 1)?
            .to_dtype(input_dtype)
            .map_err(Into::into)
    }
}

// ── MoE block forward ────────────────────────────────────────────────────

impl NemotronExpert {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // ReLU²(up(x)) → down
        let up = self.up_proj.forward(x)?;
        let activated = relu_squared(&up)?;
        self.down_proj.forward(&activated).map_err(Into::into)
    }
}

#[cfg(target_os = "macos")]
impl MetalIndexedNemotronExperts {
    const MAX_ROUTE_ROWS_PER_DISPATCH: usize = 4096;

    fn forward(&self, flat_x: &Tensor, expert_ids: &Tensor, weights: &Tensor) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = flat_x.dims2()?;
        let (id_tokens, top_k) = expert_ids.dims2()?;
        anyhow::ensure!(
            num_tokens == id_tokens && weights.dims2()? == (num_tokens, top_k),
            "indexed Nemotron routing tensor geometry differs"
        );

        let chunk_tokens = (Self::MAX_ROUTE_ROWS_PER_DISPATCH / top_k).max(1);
        let profile = layer_profile_enabled() && num_tokens > 1;
        if profile {
            MOE_PREFILL_PROFILE_CALLS.fetch_add(1, Ordering::Relaxed);
        }
        let mut outputs = Vec::with_capacity(num_tokens.div_ceil(chunk_tokens));
        for start in (0..num_tokens).step_by(chunk_tokens) {
            let len = (num_tokens - start).min(chunk_tokens);
            let input = flat_x
                .narrow(0, start, len)?
                .to_dtype(DType::F32)?
                .reshape((len, 1, hidden_dim))?;
            let ids = expert_ids.narrow(0, start, len)?;
            let route_weights = weights.narrow(0, start, len)?;
            let started = Instant::now();
            let route_plan = if len >= 64
                && std::env::var_os("DINF_DISABLE_COMPACT_MOE_METAL").is_none()
                && std::env::var_os("DINF_DISABLE_INDEXED_MOE_ROUTE_PLAN_REUSE").is_none()
            {
                Some(self.up_proj.prepare_route_plan(&ids)?)
            } else {
                None
            };
            if profile {
                record_profile_time(flat_x.device(), started, &MOE_ROUTE_PLAN_PROFILE_NS)?;
            }
            if route_stats_enabled()
                && let Some(plan) = route_plan.as_ref()
            {
                let stats = plan.stats()?;
                MOE_ROUTE_STATS_PLANS.fetch_add(1, Ordering::Relaxed);
                MOE_ROUTE_STATS_ROUTES.fetch_add(stats.routes as u64, Ordering::Relaxed);
                MOE_ROUTE_STATS_OCCUPIED_EXPERTS
                    .fetch_add(stats.occupied_experts as u64, Ordering::Relaxed);
                MOE_ROUTE_STATS_TILES.fetch_add(stats.route_tiles as u64, Ordering::Relaxed);
                MOE_ROUTE_STATS_TILES_64.fetch_add(stats.route_tiles_64 as u64, Ordering::Relaxed);
                MOE_ROUTE_STATS_MULTI_TILE_EXPERTS
                    .fetch_add(stats.multi_tile_experts as u64, Ordering::Relaxed);
                MOE_ROUTE_STATS_MAX_ROUTES
                    .fetch_max(stats.max_routes_per_expert as u64, Ordering::Relaxed);
            }
            let started = Instant::now();
            let up = if let Some(plan) = route_plan.as_ref() {
                self.up_proj.forward_with_route_plan(&input, plan)?
            } else {
                self.up_proj.forward(&input, &ids)?
            };
            if profile {
                record_profile_time(flat_x.device(), started, &MOE_INDEXED_UP_PROFILE_NS)?;
            }
            let started = Instant::now();
            let output = if let Some(plan) = route_plan.as_ref() {
                self.down_proj
                    .forward_relu_squared_with_route_plan(&up, plan)?
            } else {
                self.down_proj.forward_relu_squared(&up, &ids)?
            };
            if profile {
                record_profile_time(flat_x.device(), started, &MOE_INDEXED_DOWN_PROFILE_NS)?;
            }
            let started = Instant::now();
            outputs.push(fused_weighted_route_sum(&output, &route_weights)?);
            if profile {
                record_profile_time(flat_x.device(), started, &MOE_INDEXED_SUM_PROFILE_NS)?;
            }
        }

        let started = Instant::now();
        let output_refs = outputs.iter().collect::<Vec<_>>();
        let output = Tensor::cat(&output_refs, 0)?;
        if profile {
            record_profile_time(flat_x.device(), started, &MOE_INDEXED_SUM_PROFILE_NS)?;
        }
        Ok(output)
    }
}

#[cfg(all(feature = "cuda", not(target_os = "macos")))]
impl CudaIndexedNemotronExperts {
    const MAX_ROUTE_ROWS_PER_DISPATCH: usize = 4096;

    fn forward(&self, flat_x: &Tensor, expert_ids: &Tensor, weights: &Tensor) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = flat_x.dims2()?;
        let (id_tokens, top_k) = expert_ids.dims2()?;
        anyhow::ensure!(
            num_tokens == id_tokens && weights.dims2()? == (num_tokens, top_k),
            "indexed Nemotron routing tensor geometry differs"
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
            let up = self.up_proj.forward(&input, &ids)?;
            let output = self.down_proj.forward_relu_squared(&up, &ids)?;
            outputs.push(weighted_route_sum(&output, &route_weights)?);
        }

        let output_refs = outputs.iter().collect::<Vec<_>>();
        Tensor::cat(&output_refs, 0).map_err(Into::into)
    }
}

impl DenseBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normed = self.norm.forward(x)?;
        #[cfg(feature = "cuda")]
        let output = match super::dense_qmatmul_cuda::forward_one_to_f32(&normed, &self.up_proj)? {
            Some(up) => match super::dense_qmatmul_cuda::relu_squared_down(&up, &self.down_proj)? {
                Some(output) => output,
                None => self.down_proj.forward(&relu_squared(&up)?)?,
            },
            None => {
                let up = self.up_proj.forward(&normed)?;
                self.down_proj.forward(&relu_squared(&up)?)?
            }
        };
        #[cfg(not(feature = "cuda"))]
        let output = {
            let up = self.up_proj.forward(&normed)?;
            self.down_proj.forward(&relu_squared(&up)?)?
        };
        (x + output).map_err(Into::into)
    }
}

impl MoeBlock {
    fn selected_experts(
        &self,
        selection_scores: &[f32],
        probabilities: &[f32],
    ) -> Result<Vec<(usize, f32)>> {
        let expert_count = self.experts.len();
        anyhow::ensure!(
            selection_scores.len() == expert_count && probabilities.len() == expert_count,
            "Nemotron router geometry differs from its expert count"
        );
        anyhow::ensure!(
            self.num_active <= expert_count,
            "Nemotron requests {} active experts from a pool of {expert_count}",
            self.num_active
        );

        let mut indexed = selection_scores
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<_>>();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        indexed.truncate(self.num_active);
        let mut selected = indexed
            .into_iter()
            .map(|(expert, _)| (expert, probabilities[expert]))
            .collect::<Vec<_>>();
        if self.weights_norm {
            let weight_sum = selected
                .iter()
                .map(|(_, weight)| *weight)
                .sum::<f32>()
                .max(6.103_515_6e-5);
            for (_, weight) in &mut selected {
                *weight /= weight_sum;
            }
        }
        Ok(selected)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_dim) = x.dims3()?;
        let input_dtype = x.dtype();
        let profile = layer_profile_enabled() && seq_len > 1;
        if profile {
            x.device().synchronize()?;
        }
        let route_started = Instant::now();
        let normed = self.norm.forward(x)?;
        let num_tokens = b_sz * seq_len;
        let flat_x = normed.reshape((num_tokens, hidden_dim))?;

        let router_logits = flat_x.matmul(&self.router.t()?)?;

        #[cfg(target_os = "macos")]
        let fused_indexed_routing = match (&self.experts, &self.expert_bias) {
            (NemotronExperts::MetalIndexed(_), Some(bias))
                if self.weights_norm
                    && self.experts.len() <= 128
                    && self.num_active <= 16
                    && std::env::var_os("DINF_DISABLE_NEMOTRON_GPU_ROUTING").is_none()
                    && std::env::var_os("DINF_DISABLE_FUSED_TOPK_ROUTING").is_none() =>
            {
                Some(fused_sigmoid_topk_route(
                    &router_logits.contiguous()?,
                    bias,
                    self.num_active,
                )?)
            }
            _ => None,
        };
        #[cfg(not(target_os = "macos"))]
        let fused_indexed_routing: Option<(Tensor, Tensor)> = None;

        let router_probs = if fused_indexed_routing.is_some() {
            None
        } else {
            Some(
                candle_nn::Activation::Sigmoid
                    .forward(&router_logits)?
                    .to_dtype(DType::F32)?,
            )
        };

        let selection_scores = if let Some(router_probs) = router_probs.as_ref() {
            Some(if let Some(bias) = &self.expert_bias {
                router_probs.broadcast_add(bias)?
            } else {
                router_probs.clone()
            })
        } else {
            None
        };

        #[cfg(any(target_os = "macos", feature = "cuda"))]
        let indexed_routing = if let Some(routing) = fused_indexed_routing {
            Some(routing)
        } else if self.experts.is_indexed()
            && std::env::var_os("DINF_DISABLE_NEMOTRON_GPU_ROUTING").is_none()
        {
            let selection_scores = selection_scores
                .as_ref()
                .context("missing Nemotron selection scores")?;
            let router_probs = router_probs
                .as_ref()
                .context("missing Nemotron router probabilities")?;
            let expert_ids = selection_scores
                .contiguous()?
                .arg_sort_last_dim(false)?
                .narrow(D::Minus1, 0, self.num_active)?
                .contiguous()?;
            let expert_weights = if self.weights_norm {
                normalized_route_weights(router_probs, &expert_ids)?
            } else {
                router_probs.gather(&expert_ids, D::Minus1)?
            };
            Some((expert_ids, expert_weights))
        } else {
            None
        };
        #[cfg(not(any(target_os = "macos", feature = "cuda")))]
        let indexed_routing: Option<(Tensor, Tensor)> = None;

        let host_routing = if indexed_routing.is_none() {
            Some((
                selection_scores
                    .as_ref()
                    .context("missing Nemotron selection scores")?
                    .to_vec2::<f32>()?,
                router_probs
                    .as_ref()
                    .context("missing Nemotron router probabilities")?
                    .to_vec2::<f32>()?,
            ))
        } else {
            None
        };

        let output = match &self.experts {
            #[cfg(target_os = "macos")]
            NemotronExperts::MetalIndexed(experts) => {
                let cpu_routing;
                let (expert_ids, expert_weights) = if let Some(indexed) = indexed_routing.as_ref() {
                    (&indexed.0, &indexed.1)
                } else {
                    let (selection_rows, probability_rows) = host_routing
                        .as_ref()
                        .context("missing host Nemotron routing values")?;
                    let mut ids = Vec::with_capacity(num_tokens * self.num_active);
                    let mut weights = Vec::with_capacity(num_tokens * self.num_active);
                    for (selection, probabilities) in selection_rows.iter().zip(probability_rows) {
                        for (expert, weight) in self.selected_experts(selection, probabilities)? {
                            ids.push(expert as u32);
                            weights.push(weight);
                        }
                    }
                    cpu_routing = (
                        Tensor::from_vec(ids, (num_tokens, self.num_active), x.device())?,
                        Tensor::from_vec(weights, (num_tokens, self.num_active), x.device())?,
                    );
                    (&cpu_routing.0, &cpu_routing.1)
                };
                let scaled_weights = if self.weights_scale == 1.0 {
                    expert_weights.clone()
                } else {
                    (expert_weights * self.weights_scale)?
                };
                if profile {
                    record_profile_time(x.device(), route_started, &MOE_ROUTE_PROFILE_NS)?;
                }
                experts.forward(&flat_x, expert_ids, &scaled_weights)?
            }
            #[cfg(all(feature = "cuda", not(target_os = "macos")))]
            NemotronExperts::CudaIndexed(experts) => {
                let indexed = indexed_routing
                    .as_ref()
                    .context("missing CUDA Nemotron routing values")?;
                let scaled_weights = if self.weights_scale == 1.0 {
                    indexed.1.clone()
                } else {
                    (&indexed.1 * self.weights_scale)?
                };
                if profile {
                    record_profile_time(x.device(), route_started, &MOE_ROUTE_PROFILE_NS)?;
                }
                experts.forward(&flat_x, &indexed.0, &scaled_weights)?
            }
            NemotronExperts::Split(experts) => {
                let (selection_rows, probability_rows) = host_routing
                    .as_ref()
                    .context("missing split Nemotron routing values")?;
                if profile {
                    record_profile_time(x.device(), route_started, &MOE_ROUTE_PROFILE_NS)?;
                    MOE_PREFILL_PROFILE_CALLS.fetch_add(1, Ordering::Relaxed);
                }
                let mut output = Tensor::zeros((num_tokens, hidden_dim), DType::F32, x.device())?;
                for (token_idx, (selection, probabilities)) in
                    selection_rows.iter().zip(probability_rows).enumerate()
                {
                    let token_input = flat_x.i(token_idx)?.unsqueeze(0)?;
                    let mut token_output = Tensor::zeros((1, hidden_dim), DType::F32, x.device())?;
                    for (expert, weight) in self.selected_experts(selection, probabilities)? {
                        let expert_out = experts[expert]
                            .forward(&token_input)?
                            .to_dtype(DType::F32)?;
                        token_output =
                            (token_output + (expert_out * (weight as f64 * self.weights_scale))?)?;
                    }
                    output = output.slice_scatter(&token_output, 0, token_idx)?;
                }
                output
            }
        };

        // Add shared expert output
        let shared_started = Instant::now();
        let output = if let Some(shared) = &self.shared_expert {
            let shared_input = flat_x
                .to_dtype(DType::F32)?
                .reshape((1, num_tokens, hidden_dim))?;
            let shared_out = shared
                .up_proj
                .forward_relu_squared_down(&shared_input, &shared.down_proj)?
                .to_dtype(DType::F32)?
                .reshape((num_tokens, hidden_dim))?;
            (output + shared_out)?
        } else {
            output
        };
        if profile {
            record_profile_time(x.device(), shared_started, &MOE_SHARED_PROFILE_NS)?;
        }

        // Reshape back and add residual
        let output = output
            .to_dtype(input_dtype)?
            .reshape((b_sz, seq_len, hidden_dim))?;
        (x + output).map_err(Into::into)
    }
}

// ── Attention block forward ──────────────────────────────────────────────

impl AttentionBlock {
    fn forward(
        &self,
        x: &Tensor,
        kv_cache: &mut Option<KvCache>,
        seq_len: usize,
        _start_pos: usize,
    ) -> Result<Tensor> {
        let (b_sz, _, _) = x.dims3()?;

        // Pre-norm
        let normed = self.norm.forward(x)?;

        #[cfg(feature = "cuda")]
        let grouped_qkv = super::dense_qmatmul_cuda::forward(
            &normed,
            &[&self.q_proj, &self.k_proj, &self.v_proj],
        )?;
        #[cfg(not(feature = "cuda"))]
        let grouped_qkv: Option<Vec<Tensor>> = None;
        let (q, k, v) = if let Some(mut outputs) = grouped_qkv {
            anyhow::ensure!(
                outputs.len() == 3,
                "grouped Nemotron attention count mismatch"
            );
            let v = outputs.pop().unwrap();
            let k = outputs.pop().unwrap();
            let q = outputs.pop().unwrap();
            (q, k, v)
        } else {
            (
                self.q_proj.forward(&normed)?,
                self.k_proj.forward(&normed)?,
                self.v_proj.forward(&normed)?,
            )
        };

        // Reshape to [batch, heads, seq_len, head_dim]
        let (q, k, v) = if seq_len == 1 {
            let q = q.reshape((b_sz, self.num_heads, 1, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, 1, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, 1, self.head_dim))?;
            (q, k, v)
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
            (q, k, v)
        };

        let (q, k, v, rotation) =
            super::prepare_quantized_attention(kv_cache, q, k, v, self.quant_config.as_deref())?;
        let attn_output = if let Some(output) =
            super::try_fused_f16_sdpa_native(&q, &k, &v, seq_len > 1, self.attn_scale as f32)?
        {
            output
        } else {
            let n_rep = self.num_heads / self.num_kv_heads;
            let k = repeat_kv(k.to_dtype(q.dtype())?, n_rep)?;
            let v = repeat_kv(v.to_dtype(q.dtype())?, n_rep)?;
            let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.attn_scale)?;
            let attn_weights = if seq_len > 1 {
                let mask = create_causal_mask(seq_len, attn_weights.dim(3)?, x.device())?;
                attn_weights.broadcast_add(&mask)?
            } else {
                attn_weights
            };
            softmax_last_dim(&attn_weights)?.matmul(&v)?
        };
        let attn_output = super::restore_quantized_attention_output(attn_output, rotation)?;

        let attn_output =
            attn_output
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b_sz, seq_len, ()))?;
        #[cfg(feature = "cuda")]
        let projected =
            match super::dense_qmatmul_cuda::forward_one_to_f32(&attn_output, &self.o_proj)? {
                Some(output) => output,
                None => self.o_proj.forward(&attn_output.to_dtype(x.dtype())?)?,
            };
        #[cfg(not(feature = "cuda"))]
        let projected = self.o_proj.forward(&attn_output.to_dtype(x.dtype())?)?;

        // Residual
        (x + projected).map_err(Into::into)
    }
}

// ── Stage construction ───────────────────────────────────────────────────

fn qtensor_to_qmatmul(qt: Arc<QTensor>) -> Result<QMatMul> {
    QMatMul::from_arc(qt).map_err(Into::into)
}

/// Split a packed expert QTensor into per-expert QMatMul instances.
fn split_expert_tensor(
    packed: &QTensor,
    num_experts: usize,
    out_dim: usize,
    in_dim: usize,
    device: &Device,
) -> Result<Vec<QMatMul>> {
    let dtype = packed.dtype();
    let ggml_dtype: GgmlDType = dtype;

    let elements_per_expert = out_dim * in_dim;
    let block_size = ggml_dtype.block_size();
    anyhow::ensure!(
        elements_per_expert.is_multiple_of(block_size),
        "Expert dimensions ({out_dim}×{in_dim}={elements_per_expert}) not aligned to block size ({block_size})"
    );
    let blocks_per_expert = elements_per_expert / block_size;
    let bytes_per_block = ggml_dtype.type_size();
    let bytes_per_expert = blocks_per_expert * bytes_per_block;

    let raw_data = packed.data()?;
    anyhow::ensure!(
        raw_data.len() >= num_experts * bytes_per_expert,
        "Packed expert tensor too small: {} bytes, need {}",
        raw_data.len(),
        num_experts * bytes_per_expert,
    );

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

impl NemotronStage {
    /// Construct a Nemotron-H stage from its assigned GGUF tensors.
    ///
    /// Returns an error when the per-layer Mamba/attention/MoE schedule or
    /// required tensors are missing or inconsistent.
    pub fn from_stage_tensors(
        stage: StageTensors,
        quant_config: Option<Arc<TurboQuantConfig>>,
    ) -> Result<Self> {
        let md = &stage.metadata;
        let device = &stage.device;
        let range = &stage.range;
        range.validate()?;
        anyhow::ensure!(
            range.total_layers == md.num_layers,
            "Nemotron layer range total differs from metadata"
        );
        // Take ownership so we can drop packed expert tensors after splitting
        let mut tensors = stage.tensors;

        // Per-layer arrays determine block types
        let kv_arr = &md.head_count_kv_arr;
        let ff_arr = &md.feed_forward_length_arr;

        anyhow::ensure!(
            !kv_arr.is_empty() && !ff_arr.is_empty(),
            "nemotron_h requires per-layer head_count_kv and feed_forward_length arrays"
        );
        anyhow::ensure!(
            kv_arr.len() == md.num_layers && ff_arr.len() == md.num_layers,
            "per-layer array length mismatch: kv={}, ff={}, layers={}",
            kv_arr.len(),
            ff_arr.len(),
            md.num_layers,
        );

        // SSM dimensions
        let d_state = if md.ssm_state_size > 0 {
            md.ssm_state_size
        } else {
            128
        };
        let kernel_size = if md.ssm_conv_kernel > 0 {
            md.ssm_conv_kernel
        } else {
            4
        };
        let n_group = if md.ssm_group_count > 0 {
            md.ssm_group_count
        } else {
            8
        };
        let d_inner = if md.ssm_inner_size > 0 {
            md.ssm_inner_size
        } else {
            md.hidden_dim
        };
        let n_heads = if md.ssm_time_step_rank > 0 {
            md.ssm_time_step_rank
        } else {
            64
        };
        let has_mamba =
            (range.start..range.end).any(|index| kv_arr[index] == 0 && ff_arr[index] == 0);
        let (head_dim, d_conv) = if has_mamba {
            anyhow::ensure!(
                d_inner > 0 && d_inner.is_multiple_of(n_heads),
                "Nemotron SSM inner width must be a positive multiple of its head count"
            );
            anyhow::ensure!(
                n_heads.is_multiple_of(n_group),
                "Nemotron SSM head count must be divisible by its group count"
            );
            let state_width = n_group
                .checked_mul(d_state)
                .context("Nemotron SSM group width overflow")?;
            let d_conv = state_width
                .checked_mul(2)
                .and_then(|width| d_inner.checked_add(width))
                .context("Nemotron SSM convolution width overflow")?;
            let input_width = d_inner
                .checked_add(d_conv)
                .and_then(|width| width.checked_add(n_heads))
                .context("Nemotron SSM input width overflow")?;
            d_conv
                .checked_mul(kernel_size)
                .context("Nemotron SSM convolution state overflow")?;
            n_heads
                .checked_mul(d_state)
                .and_then(|width| width.checked_mul(d_inner / n_heads))
                .context("Nemotron SSM recurrent state overflow")?;
            for layer in range.start..range.end {
                if kv_arr[layer] != 0 || ff_arr[layer] != 0 {
                    continue;
                }
                for (suffix, expected) in [
                    ("ssm_in.weight", [input_width, md.hidden_dim]),
                    ("ssm_out.weight", [md.hidden_dim, d_inner]),
                    ("ssm_conv1d.weight", [d_conv, kernel_size]),
                ] {
                    let name = format!("blk.{layer}.{suffix}");
                    let tensor = tensors
                        .get(&name)
                        .with_context(|| format!("missing tensor: {name}"))?;
                    anyhow::ensure!(
                        tensor.shape().dims() == expected,
                        "{name} shape {:?} does not match Nemotron SSM metadata {expected:?}",
                        tensor.shape().dims()
                    );
                }
            }
            (d_inner / n_heads, d_conv)
        } else {
            // Unused SSM metadata must not affect attention/FFN-only ranges.
            (0, 0)
        };

        #[cfg(feature = "cuda")]
        super::dense_qmatmul_cuda::precompile(device)?;
        #[cfg(target_os = "macos")]
        super::nemotron_mamba::precompile(device)?;
        #[cfg(feature = "cuda")]
        super::nemotron_mamba_cuda::precompile(device)?;

        // MoE parameters
        let num_experts = md.expert_count;
        let num_active = md.expert_used_count;
        let expert_ffn_dim = md.expert_ffn_dim;

        // Attention parameters
        let attn_num_heads = md.num_attention_heads;
        let attn_head_dim = md.head_dim;
        let attn_scale = if md.attention_scale > 0.0 {
            md.attention_scale
        } else {
            1.0 / (attn_head_dim as f64).sqrt()
        };
        // Embedding (head only)
        let embedding = if range.is_head() {
            let embd = tensors
                .get("token_embd.weight")
                .context("missing token_embd.weight for head stage")?;
            Some(embd.as_ref().dequantize(device)?)
        } else {
            None
        };

        // Build blocks
        let mut blocks = Vec::with_capacity(range.layer_count());
        let mut n_mamba2 = 0usize;
        let mut n_dense = 0usize;
        let mut n_moe = 0usize;
        let mut n_attn = 0usize;

        for layer_idx in range.start..range.end {
            let prefix = format!("blk.{layer_idx}");
            tracing::debug!("Building layer {layer_idx}...");

            let layer_kv = kv_arr[layer_idx];
            let layer_ff = ff_arr[layer_idx];

            let get = |tensors: &std::collections::HashMap<String, Arc<QTensor>>,
                       name: &str|
             -> Result<Arc<QTensor>> {
                tensors
                    .get(&format!("{prefix}.{name}"))
                    .cloned()
                    .with_context(|| format!("missing tensor: {prefix}.{name}"))
            };

            let block = if layer_kv == 0 && layer_ff == 0 {
                // Mamba-2 layer
                n_mamba2 += 1;
                let norm = RmsNorm {
                    weight: get(&tensors, "attn_norm.weight")?
                        .as_ref()
                        .dequantize(device)?,
                    eps: md.rms_norm_eps,
                };
                let ssm_in =
                    NemotronQMatMul::from_qtensor(get(&tensors, "ssm_in.weight")?, device)?;
                let conv1d_weight = get(&tensors, "ssm_conv1d.weight")?
                    .as_ref()
                    .dequantize(device)?;
                let conv1d_bias = tensors
                    .get(&format!("{prefix}.ssm_conv1d.bias"))
                    .map(|t| t.as_ref().dequantize(device))
                    .transpose()?;
                let conv1d_bias = match conv1d_bias {
                    Some(bias) => bias,
                    None => Tensor::zeros(d_conv, DType::F32, device)?,
                };
                let dt_bias = get(&tensors, "ssm_dt.bias")?.as_ref().dequantize(device)?;
                let a = get(&tensors, "ssm_a")?.as_ref().dequantize(device)?;
                let d_tensor = get(&tensors, "ssm_d")?.as_ref().dequantize(device)?;
                let ssm_norm_w = get(&tensors, "ssm_norm.weight")?
                    .as_ref()
                    .dequantize(device)?;
                let ssm_out =
                    NemotronQMatMul::from_qtensor(get(&tensors, "ssm_out.weight")?, device)?;

                NemotronBlock::Mamba2(Mamba2Block {
                    norm,
                    ssm_in,
                    conv1d_weight,
                    conv1d_bias,
                    dt_bias,
                    a,
                    d: d_tensor,
                    ssm_norm: ssm_norm_w,
                    ssm_out,
                    d_inner,
                    d_state,
                    n_heads,
                    n_group,
                    head_dim,
                    d_conv,
                    kernel_size,
                })
            } else if layer_ff > 0 && tensors.contains_key(&format!("{prefix}.ffn_gate_inp.weight"))
            {
                // MoE FFN layer
                n_moe += 1;
                let norm = RmsNorm {
                    weight: get(&tensors, "attn_norm.weight")?
                        .as_ref()
                        .dequantize(device)?,
                    eps: md.rms_norm_eps,
                };

                let router = get(&tensors, "ffn_gate_inp.weight")?
                    .as_ref()
                    .dequantize(device)?;

                let expert_bias = tensors
                    .get(&format!("{prefix}.exp_probs_b.bias"))
                    .map(|t| t.as_ref().dequantize(device))
                    .transpose()?;

                let up_key = format!("{prefix}.ffn_up_exps.weight");
                let down_key = format!("{prefix}.ffn_down_exps.weight");
                let up_exps = tensors
                    .remove(&up_key)
                    .with_context(|| format!("missing tensor: {up_key}"))?;
                let down_exps = tensors
                    .remove(&down_key)
                    .with_context(|| format!("missing tensor: {down_key}"))?;

                let split_experts = || -> Result<NemotronExperts> {
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
                    Ok(NemotronExperts::Split(
                        up_mats
                            .into_iter()
                            .zip(down_mats)
                            .map(|(up_proj, down_proj)| NemotronExpert { up_proj, down_proj })
                            .collect(),
                    ))
                };

                #[cfg(target_os = "macos")]
                let experts = {
                    let use_indexed_metal = device.is_metal()
                        && std::env::var_os("DINF_DISABLE_INDEXED_MOE_METAL").is_none()
                        && MetalIndexedMoeMatMul::supports_dtype(up_exps.dtype())
                        && MetalIndexedMoeMatMul::supports_dtype(down_exps.dtype());
                    if use_indexed_metal {
                        let experts = NemotronExperts::MetalIndexed(MetalIndexedNemotronExperts {
                            up_proj: MetalIndexedMoeMatMul::from_qtensor(
                                up_exps.as_ref(),
                                num_experts,
                                expert_ffn_dim,
                                md.hidden_dim,
                                device,
                            )?,
                            down_proj: MetalIndexedMoeMatMul::from_qtensor(
                                down_exps.as_ref(),
                                num_experts,
                                md.hidden_dim,
                                expert_ffn_dim,
                                device,
                            )?,
                        });
                        device.synchronize()?;
                        experts
                    } else {
                        split_experts()?
                    }
                };
                #[cfg(all(feature = "cuda", not(target_os = "macos")))]
                let experts = if indexed_moe_available(device, up_exps.dtype())
                    && indexed_moe_available(device, down_exps.dtype())
                {
                    NemotronExperts::CudaIndexed(CudaIndexedNemotronExperts {
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
                    })
                } else {
                    split_experts()?
                };
                #[cfg(not(any(target_os = "macos", feature = "cuda")))]
                let experts = split_experts()?;
                drop(up_exps);
                drop(down_exps);

                // Shared expert (optional)
                let shared_expert = if md.shared_expert_count > 0 {
                    let sh_up = tensors.get(&format!("{prefix}.ffn_up_shexp.weight"));
                    let sh_down = tensors.get(&format!("{prefix}.ffn_down_shexp.weight"));
                    match (sh_up, sh_down) {
                        (Some(up), Some(down)) => Some(SharedExpert {
                            up_proj: NemotronQMatMul::from_qtensor(Arc::clone(up), device)?,
                            down_proj: NemotronQMatMul::from_qtensor(Arc::clone(down), device)?,
                        }),
                        _ => None,
                    }
                } else {
                    None
                };

                NemotronBlock::MoE(MoeBlock {
                    norm,
                    router,
                    expert_bias,
                    experts,
                    shared_expert,
                    num_active,
                    weights_scale: md.expert_weights_scale,
                    weights_norm: md.expert_weights_norm,
                })
            } else if layer_ff > 0 {
                // Dense ReLU² FFN layer used by Nemotron Nano variants.
                n_dense += 1;
                NemotronBlock::Dense(DenseBlock {
                    norm: RmsNorm {
                        weight: get(&tensors, "attn_norm.weight")?
                            .as_ref()
                            .dequantize(device)?,
                        eps: md.rms_norm_eps,
                    },
                    up_proj: qtensor_to_qmatmul(get(&tensors, "ffn_up.weight")?)?,
                    down_proj: qtensor_to_qmatmul(get(&tensors, "ffn_down.weight")?)?,
                })
            } else {
                // Attention layer (layer_kv > 0)
                // Nemotron-H uses raw Q/K. Its generic RoPE metadata is not
                // part of the architecture's attention computation.
                n_attn += 1;
                let layer_kv_heads = layer_kv;
                let norm = RmsNorm {
                    weight: get(&tensors, "attn_norm.weight")?
                        .as_ref()
                        .dequantize(device)?,
                    eps: md.rms_norm_eps,
                };

                NemotronBlock::Attention(AttentionBlock {
                    norm,
                    q_proj: qtensor_to_qmatmul(get(&tensors, "attn_q.weight")?)?,
                    k_proj: qtensor_to_qmatmul(get(&tensors, "attn_k.weight")?)?,
                    v_proj: qtensor_to_qmatmul(get(&tensors, "attn_v.weight")?)?,
                    o_proj: qtensor_to_qmatmul(get(&tensors, "attn_output.weight")?)?,
                    num_heads: attn_num_heads,
                    num_kv_heads: layer_kv_heads,
                    head_dim: attn_head_dim,
                    attn_scale,
                    quant_config: quant_config.clone(),
                })
            };

            blocks.push(block);

            // Drop all remaining tensors for this layer to free GPU memory
            let layer_prefix = format!("blk.{layer_idx}.");
            let keys_to_remove: Vec<String> = tensors
                .keys()
                .filter(|k| k.starts_with(&layer_prefix))
                .cloned()
                .collect();
            for key in keys_to_remove {
                tensors.remove(&key);
            }
        }

        // Output norm + lm_head (tail only)
        let output_norm = if range.is_tail() {
            let norm_w = tensors
                .get("output_norm.weight")
                .context("missing output_norm.weight")?;
            Some(RmsNorm {
                weight: norm_w.as_ref().dequantize(device)?,
                eps: md.rms_norm_eps,
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

        tracing::info!(
            "NemotronStage: {} blocks ({} Mamba-2 + {} dense + {} MoE + {} Attention), layers [{}, {})",
            blocks.len(),
            n_mamba2,
            n_dense,
            n_moe,
            n_attn,
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
            device: device.clone(),
        })
    }
}

// ── ModelArchitecture implementation ─────────────────────────────────────

impl ModelArchitecture for NemotronStage {
    fn fixed_cache_bytes(&self, _verification_rows: usize) -> Result<usize> {
        self.blocks.iter().try_fold(0usize, |total, block| {
            let NemotronBlock::Mamba2(block) = block else {
                return Ok(total);
            };
            let recurrent = super::fixed_state_tensor_bytes(
                &[block.n_heads, block.d_state, block.head_dim],
                &self.device,
            )?;
            let convolution =
                super::fixed_state_tensor_bytes(&[block.d_conv, block.kernel_size], &self.device)?;
            recurrent
                .checked_add(convolution)
                .and_then(|bytes| total.checked_add(bytes))
                .context("Mamba fixed request state size overflow")
        })
    }

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

        let profile = layer_profile_enabled();
        if profile {
            self.device.synchronize()?;
        }
        trace_layer_tensor("embedding", -1, &hidden_states, seq_len)?;

        for (i, block) in self.blocks.iter().enumerate() {
            let started = Instant::now();
            hidden_states = match block {
                NemotronBlock::Mamba2(mamba_block) => {
                    let mut m2_cache = match cache[i].take() {
                        Some(LayerCache::Mamba2(c)) => Some(c),
                        Some(_) => panic!("expected Mamba2 cache"),
                        None => None,
                    };
                    let result =
                        mamba_block.forward(&hidden_states, &mut m2_cache, seq_len, start_pos)?;
                    cache[i] = m2_cache.map(LayerCache::Mamba2);
                    result
                }
                NemotronBlock::Dense(dense_block) => dense_block.forward(&hidden_states)?,
                NemotronBlock::MoE(moe_block) => {
                    // MoE layers are stateless — no cache needed
                    moe_block.forward(&hidden_states)?
                }
                NemotronBlock::Attention(attn_block) => {
                    let mut kv_cache = match cache[i].take() {
                        Some(LayerCache::Attention(kv)) => Some(kv),
                        Some(_) => panic!("expected Attention cache"),
                        None => None,
                    };
                    let result =
                        attn_block.forward(&hidden_states, &mut kv_cache, seq_len, start_pos)?;
                    cache[i] = kv_cache.map(LayerCache::Attention);
                    result
                }
            };
            trace_layer_tensor("block_out", i as isize, &hidden_states, seq_len)?;
            if profile {
                let (elapsed, calls) = match block {
                    NemotronBlock::Mamba2(_) => (&MAMBA_PROFILE_NS, &MAMBA_PROFILE_CALLS),
                    NemotronBlock::Dense(_) => (&DENSE_PROFILE_NS, &DENSE_PROFILE_CALLS),
                    NemotronBlock::MoE(_) => (&MOE_PROFILE_NS, &MOE_PROFILE_CALLS),
                    NemotronBlock::Attention(_) => {
                        (&ATTENTION_PROFILE_NS, &ATTENTION_PROFILE_CALLS)
                    }
                };
                record_profile_region(&self.device, started, elapsed, calls)?;
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
                None => self.lm_head.as_ref().unwrap().forward(&last_hidden)?,
            };
            #[cfg(not(feature = "cuda"))]
            let logits = self.lm_head.as_ref().unwrap().forward(&last_hidden)?;
            let logits = logits.squeeze(0)?;
            Ok(StageOutput::Logits(logits))
        } else {
            let output = hidden_states.squeeze(0)?;
            Ok(StageOutput::Activations(output))
        }
    }

    fn family(&self) -> &str {
        "nemotron_h"
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

// ── Unit tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_ssm_stage() -> Result<StageTensors> {
        use crate::gguf_loader::{LayerRange, parse_gguf_metadata};
        use candle_core::quantized::gguf_file::Value;
        let mut values = std::collections::HashMap::from([(
            "general.architecture".to_string(),
            Value::String("nemotron_h".into()),
        )]);
        for (key, value) in [
            ("block_count", 2),
            ("embedding_length", 8),
            ("attention.head_count", 1),
            ("attention.key_length", 8),
            ("context_length", 8),
            ("vocab_size", 4),
            ("ssm.inner_size", 8),
            ("ssm.time_step_rank", 2),
            ("ssm.group_count", 1),
            ("ssm.state_size", 2),
            ("ssm.conv_kernel", 3),
        ] {
            values.insert(format!("nemotron_h.{key}"), Value::U32(value));
        }
        for key in ["attention.head_count_kv", "feed_forward_length"] {
            values.insert(
                format!("nemotron_h.{key}"),
                Value::Array(vec![Value::U32(0); 2]),
            );
        }
        Ok(StageTensors {
            range: LayerRange::new(1, 2, 2),
            metadata: parse_gguf_metadata(&values)?,
            tensors: std::collections::HashMap::new(),
            raw_tensors: std::collections::HashMap::new(),
            raw_source: None,
            device: Device::Cpu,
        })
    }

    fn add_tiny_ssm_weights(stage: &mut StageTensors) -> Result<()> {
        for (name, shape) in [
            ("blk.1.attn_norm.weight", vec![8]),
            ("blk.1.ssm_in.weight", vec![22, 8]),
            ("blk.1.ssm_out.weight", vec![8, 8]),
            ("blk.1.ssm_conv1d.weight", vec![12, 3]),
            ("blk.1.ssm_dt.bias", vec![2]),
            ("blk.1.ssm_a", vec![2]),
            ("blk.1.ssm_d", vec![2]),
            ("blk.1.ssm_norm.weight", vec![1, 8]),
            ("output_norm.weight", vec![8]),
            ("output.weight", vec![4, 8]),
        ] {
            let tensor = Tensor::ones(shape, DType::F32, &Device::Cpu)?;
            stage.tensors.insert(
                name.to_string(),
                Arc::new(QTensor::quantize(&tensor, GgmlDType::F32)?),
            );
        }
        Ok(())
    }

    #[test]
    fn ssm_stage_rejects_layer_ranges_outside_metadata() -> Result<()> {
        let mut stage = tiny_ssm_stage()?;
        stage.range = crate::gguf_loader::LayerRange::new(1, 3, 3);
        let error = NemotronStage::from_stage_tensors(stage, None)
            .err()
            .expect("mismatched layer range must fail")
            .to_string();
        assert!(error.contains("layer range total"), "{error}");
        Ok(())
    }

    #[test]
    fn ssm_geometry_rejects_overflow_and_bad_groups_before_weights() -> Result<()> {
        for (groups, heads, inner, state) in [
            (1usize << (usize::BITS - 1), 2, 8, 2),
            (1, 3, 8, 2),
            (3, 2, 8, 2),
            (1, 2, 8, usize::MAX),
            (1, 2, usize::MAX - 1, 2),
        ] {
            let mut stage = tiny_ssm_stage()?;
            stage.metadata.ssm_group_count = groups;
            stage.metadata.ssm_time_step_rank = heads;
            stage.metadata.ssm_inner_size = inner;
            stage.metadata.ssm_state_size = state;
            let error = NemotronStage::from_stage_tensors(stage, None)
                .err()
                .expect("bad SSM geometry must fail")
                .to_string();
            assert!(error.contains("Nemotron SSM"), "{error}");
        }
        Ok(())
    }

    #[test]
    fn ssm_projection_shapes_reject_large_metadata_before_bias_allocation() -> Result<()> {
        let mut stage = tiny_ssm_stage()?;
        add_tiny_ssm_weights(&mut stage)?;
        stage.metadata.ssm_state_size = 1 << 36;
        let error = NemotronStage::from_stage_tensors(stage, None)
            .err()
            .expect("mismatched SSM projection must fail")
            .to_string();
        assert!(error.contains("ssm_in.weight shape"), "{error}");
        let mut stage = tiny_ssm_stage()?;
        stage.metadata.head_count_kv_arr[1] = 1;
        stage.metadata.ssm_group_count = usize::MAX;
        stage.metadata.ssm_inner_size = usize::MAX;
        let error = NemotronStage::from_stage_tensors(stage, None)
            .err()
            .expect("missing attention weights must fail")
            .to_string();
        assert!(
            error.contains("missing"),
            "unused SSM metadata changed an attention stage: {error}"
        );
        Ok(())
    }

    #[test]
    fn valid_small_ssm_stage_runs_without_optional_convolution_bias() -> Result<()> {
        let mut stage = tiny_ssm_stage()?;
        add_tiny_ssm_weights(&mut stage)?;
        let model = NemotronStage::from_stage_tensors(stage, None)?;
        let mut cache = [None];
        for position in 0..2 {
            let output = model.forward(
                StageInput::activations(Tensor::ones((1, 8), DType::F32, &Device::Cpu)?),
                &mut cache,
                1,
                position,
            )?;
            let StageOutput::Logits(logits) = output else {
                anyhow::bail!("tail must return logits");
            };
            assert!(
                logits
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|value| value.is_finite())
            );
            assert!(matches!(cache[0], Some(LayerCache::Mamba2(_))));
        }
        Ok(())
    }

    #[test]
    fn test_relu_squared() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0], &device)?;
        let result = relu_squared(&x)?;
        let vals: Vec<f32> = result.to_vec1()?;
        let expected = [0.0, 0.0, 0.0, 1.0, 4.0, 9.0];
        for (got, exp) in vals.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-5, "relu²: got {got}, expected {exp}");
        }
        Ok(())
    }

    #[test]
    fn test_mamba2_ssm_step() -> Result<()> {
        let device = Device::Cpu;
        let d_state = 4;
        let head_dim = 4;

        let state = Tensor::zeros((1, d_state, head_dim), DType::F32, &device)?;
        let x = Tensor::new(&[[1.0f32, 0.0, 0.0, 0.0]], &device)?;
        let b = Tensor::new(&[[1.0f32, 0.0, 0.0, 0.0]], &device)?;
        let c = Tensor::new(&[[1.0f32, 0.0, 0.0, 0.0]], &device)?;
        let a = Tensor::new(&[-0.1f32], &device)?;
        // inverse_softplus(1) so the recurrence sees an effective dt of 1.
        let dt_raw = Tensor::new(&[0.541_324_85f32], &device)?;
        let dt_bias = Tensor::zeros(1, DType::F32, &device)?;
        let d = Tensor::new(&[0.5f32], &device)?;

        let (output, new_state) = mamba2_ssm_step_batched(Mamba2SsmStep {
            state: &state,
            x: &x,
            b: &b,
            c: &c,
            a: &a,
            dt_raw: &dt_raw,
            dt_bias: &dt_bias,
            d: &d,
            n_group: 1,
        })?;

        // decay = exp(-0.1*1) ≈ 0.905
        // B_bar = B * 1.0 = [1,0,0,0]
        // state_new = 0 + outer([1,0,0,0], [1,0,0,0]) = [[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        // y = C @ state_new = [1,0,0,0] @ [[1,0,0,0],...] = [1,0,0,0]
        // output = y + D*x = [1,0,0,0] + 0.5*[1,0,0,0] = [1.5,0,0,0]
        let out_vals: Vec<f32> = output.flatten_all()?.to_vec1()?;
        assert!((out_vals[0] - 1.5).abs() < 1e-4, "got {}", out_vals[0]);
        assert!((out_vals[1] - 0.0).abs() < 1e-4);

        // Second step with same input — should show decay effect
        let (output2, _) = mamba2_ssm_step_batched(Mamba2SsmStep {
            state: &new_state,
            x: &x,
            b: &b,
            c: &c,
            a: &a,
            dt_raw: &dt_raw,
            dt_bias: &dt_bias,
            d: &d,
            n_group: 1,
        })?;
        let out_vals2: Vec<f32> = output2.flatten_all()?.to_vec1()?;
        // decay ≈ 0.905, state[0,0] = 0.905*1 + 1 = 1.905
        // y = C @ state = [1.905, 0, 0, 0]
        // output = 1.905 + 0.5 = 2.405
        assert!((out_vals2[0] - 2.405).abs() < 0.01, "got {}", out_vals2[0]);

        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_mamba2_metal_matches_portable_path() -> Result<()> {
        let cpu = Device::Cpu;
        let Some(metal) = crate::test_metal_device() else {
            return Ok(());
        };
        let n_heads = 4;
        let d_state = 8;
        let head_dim = 4;
        let n_group = 2;

        let state_values: Vec<f32> = (0..n_heads * d_state * head_dim)
            .map(|i| (i as f32 * 0.17).sin() * 0.2)
            .collect();
        let x_values: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| (i as f32 * 0.11).cos() * 0.3)
            .collect();
        let b_values: Vec<f32> = (0..n_group * d_state)
            .map(|i| (i as f32 * 0.07).sin() * 0.4)
            .collect();
        let c_values: Vec<f32> = (0..n_group * d_state)
            .map(|i| (i as f32 * 0.13).cos() * 0.4)
            .collect();
        let a_values = vec![-0.1f32, -0.2, -0.3, -0.4];
        let dt_values = vec![-0.5f32, 0.0, 0.5, 1.0];
        let dt_bias_values = vec![0.1f32, -0.2, 0.3, -0.4];
        let d_values = vec![0.5f32, 0.75, 1.0, 1.25];

        let run = |device: &Device| -> Result<(Vec<f32>, Vec<f32>)> {
            let state =
                Tensor::from_vec(state_values.clone(), (n_heads, d_state, head_dim), device)?;
            let x = Tensor::from_vec(x_values.clone(), (n_heads, head_dim), device)?;
            let b = Tensor::from_vec(b_values.clone(), (n_group, d_state), device)?;
            let c = Tensor::from_vec(c_values.clone(), (n_group, d_state), device)?;
            let a = Tensor::from_vec(a_values.clone(), n_heads, device)?;
            let dt_raw = Tensor::from_vec(dt_values.clone(), n_heads, device)?;
            let dt_bias = Tensor::from_vec(dt_bias_values.clone(), n_heads, device)?;
            let d = Tensor::from_vec(d_values.clone(), n_heads, device)?;
            let (output, next_state) = mamba2_ssm_step_batched(Mamba2SsmStep {
                state: &state,
                x: &x,
                b: &b,
                c: &c,
                a: &a,
                dt_raw: &dt_raw,
                dt_bias: &dt_bias,
                d: &d,
                n_group,
            })?;
            Ok((
                output.flatten_all()?.to_vec1()?,
                next_state.flatten_all()?.to_vec1()?,
            ))
        };

        let (cpu_output, cpu_state) = run(&cpu)?;
        let (metal_output, metal_state) = run(&metal)?;
        for (expected, actual) in cpu_output.iter().zip(&metal_output) {
            assert!((expected - actual).abs() < 1e-5);
        }
        for (expected, actual) in cpu_state.iter().zip(&metal_state) {
            assert!((expected - actual).abs() < 1e-5);
        }
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
        let d_conv = 4;
        let kernel_size = 3;

        let state = Tensor::zeros((d_conv, kernel_size), DType::F32, &device)?;
        let weight = Tensor::ones((d_conv, kernel_size), DType::F32, &device)?;
        let input = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?;

        let (output, new_state) = conv1d_step(&state, &input, &weight)?;

        let out_vals: Vec<f32> = output.to_vec1()?;
        assert!((out_vals[0] - 1.0).abs() < 1e-5);
        assert!((out_vals[3] - 4.0).abs() < 1e-5);

        let input2 = Tensor::new(&[5.0f32, 6.0, 7.0, 8.0], &device)?;
        let (output2, _) = conv1d_step(&new_state, &input2, &weight)?;
        let out_vals2: Vec<f32> = output2.to_vec1()?;
        assert!((out_vals2[0] - 6.0).abs() < 1e-5);

        Ok(())
    }
}

#[cfg(test)]
mod softplus_reference_tests {
    use super::*;

    #[test]
    fn softplus_keeps_negative_gates() -> Result<()> {
        let values = [-30.0f32, -20., -15., -8.];
        let actual = softplus(&Tensor::new(&values, &Device::Cpu)?)?.to_vec1::<f32>()?;
        for (&x, actual) in values.iter().zip(actual) {
            let expected = f64::from(x).exp().ln_1p();
            let relative = (f64::from(actual) - expected).abs() / expected;
            assert!(
                relative < 2e-6,
                "softplus({x})={actual}, reference={expected}"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "nemotron_softplus_tests.rs"]
mod softplus_decay_tests;
