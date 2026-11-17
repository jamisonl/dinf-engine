use std::sync::Arc;
#[cfg(target_os = "macos")]
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::qmatmul::QMatMul;
use anyhow::{Context, Result};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Tensor};

use super::{
    KvCache, LayerCache, ModelArchitecture, StageInput, StageOutput, TextArithmeticPolicy,
    TextRequestContext, with_kv_cache,
};
use crate::gguf_loader::StageTensors;
use crate::turboquant::TurboQuantConfig;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod native_k_dot;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod native_k_dot_m1;
mod native_online_attention;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod native_q8_dot;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod native_q8k;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod native_rope_compat;
mod native_split_attention;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod native_swiglu_compat;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod prepared_k_dot;

static ATTN_NORM_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_RESIDUAL_NS: AtomicU64 = AtomicU64::new(0);
static FFN_NORM_NS: AtomicU64 = AtomicU64::new(0);
static FFN_NS: AtomicU64 = AtomicU64::new(0);
static FFN_RESIDUAL_NS: AtomicU64 = AtomicU64::new(0);
static QKV_PROJ_NS: AtomicU64 = AtomicU64::new(0);
static QKV_LAYOUT_NS: AtomicU64 = AtomicU64::new(0);
static ROPE_NS: AtomicU64 = AtomicU64::new(0);
static KV_UPDATE_NS: AtomicU64 = AtomicU64::new(0);
static SDPA_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_OUTPUT_LAYOUT_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_OUTPUT_PROJ_NS: AtomicU64 = AtomicU64::new(0);
static GATE_UP_PROJ_NS: AtomicU64 = AtomicU64::new(0);
static FFN_ACTIVATION_NS: AtomicU64 = AtomicU64::new(0);
static DOWN_PROJ_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_CALLS: AtomicU64 = AtomicU64::new(0);
#[cfg(target_os = "macos")]
static METAL4_QMM_LOGGED: OnceLock<()> = OnceLock::new();

fn layer_profile_enabled() -> bool {
    super::profiling::layer_profiling_enabled()
}

fn finish_profile_region(device: &Device, started: Instant, counter: &AtomicU64) -> Result<()> {
    device.synchronize()?;
    counter.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(())
}

/// Reset the process-local Phi-3 layer profiling counters.
pub fn reset_layer_profile() {
    for counter in [
        &ATTN_NORM_NS,
        &ATTN_NS,
        &ATTN_RESIDUAL_NS,
        &FFN_NORM_NS,
        &FFN_NS,
        &FFN_RESIDUAL_NS,
        &QKV_PROJ_NS,
        &QKV_LAYOUT_NS,
        &ROPE_NS,
        &KV_UPDATE_NS,
        &SDPA_NS,
        &ATTN_OUTPUT_LAYOUT_NS,
        &ATTN_OUTPUT_PROJ_NS,
        &GATE_UP_PROJ_NS,
        &FFN_ACTIVATION_NS,
        &DOWN_PROJ_NS,
        &PROFILE_CALLS,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

/// Print the process-local Phi-3 layer profiling counters to stderr.
pub fn report_layer_profile() {
    let calls = PROFILE_CALLS.load(Ordering::Relaxed).max(1) as f64;
    let ms_per_call =
        |counter: &AtomicU64| counter.load(Ordering::Relaxed) as f64 / calls / 1_000_000.0;
    eprintln!(
        "DINF_PHI3_PROFILE calls={} per_layer_ms={{attn_norm:{:.4},attention:{:.4},attn_residual:{:.4},ffn_norm:{:.4},ffn:{:.4},ffn_residual:{:.4}}}",
        PROFILE_CALLS.load(Ordering::Relaxed),
        ms_per_call(&ATTN_NORM_NS),
        ms_per_call(&ATTN_NS),
        ms_per_call(&ATTN_RESIDUAL_NS),
        ms_per_call(&FFN_NORM_NS),
        ms_per_call(&FFN_NS),
        ms_per_call(&FFN_RESIDUAL_NS),
    );
    eprintln!(
        "DINF_PHI3_ATTN_PROFILE per_layer_ms={{qkv_proj:{:.4},qkv_layout:{:.4},rope:{:.4},kv_update:{:.4},sdpa:{:.4},output_layout:{:.4},output_proj:{:.4}}}",
        ms_per_call(&QKV_PROJ_NS),
        ms_per_call(&QKV_LAYOUT_NS),
        ms_per_call(&ROPE_NS),
        ms_per_call(&KV_UPDATE_NS),
        ms_per_call(&SDPA_NS),
        ms_per_call(&ATTN_OUTPUT_LAYOUT_NS),
        ms_per_call(&ATTN_OUTPUT_PROJ_NS),
    );
    eprintln!(
        "DINF_PHI3_FFN_PROFILE per_layer_ms={{gate_up:{:.4},activation:{:.4},down:{:.4}}}",
        ms_per_call(&GATE_UP_PROJ_NS),
        ms_per_call(&FFN_ACTIVATION_NS),
        ms_per_call(&DOWN_PROJ_NS),
    );
}

/// A pipeline stage running the Phi-3 architecture.
///
/// Key differences from Llama:
/// - Fused QKV projection (single `attn_qkv.weight`)
/// - Fused gate+up FFN (`ffn_up.weight` is [hidden, 2*intermediate])
/// - Partial RoPE (rope_dimension_count may be < head_dim)
pub struct Phi3Stage {
    embedding: Option<Tensor>,
    blocks: Vec<TransformerBlock>,
    output_norm: Option<RmsNorm>,
    lm_head: Option<QMatMul>,
    head_native_cpu: bool,
    is_head: bool,
    is_tail: bool,
    hidden_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    device: Device,
}

struct TransformerBlock {
    attn_norm: RmsNorm,
    attn: Attention,
    ffn_norm: RmsNorm,
    ffn: Mlp,
}

struct Attention {
    qkv_proj: Phi3QMatMul,
    o_proj: Phi3QMatMul,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// RoPE dimension count (may be < head_dim for partial RoPE).
    rope_dim: usize,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    rope_theta: f32,
    rope_cos: Tensor,
    rope_sin: Tensor,
    quant_config: Option<Arc<TurboQuantConfig>>,
}

struct Mlp {
    /// Fused gate + up projection: output is [batch, seq, 2 * intermediate_dim]
    gate_up_proj: Phi3QMatMul,
    down_proj: Phi3QMatMul,
    intermediate_dim: usize,
}

enum Phi3QMatMul {
    Candle(QMatMul),
    NativeCpu(QMatMul),
    #[cfg(target_os = "macos")]
    MetalTensor(super::tensor_qmatmul_metal::MetalTensorQMatMul),
}

impl Phi3QMatMul {
    /// Native-compatible query precision for CPU quantized Phi attention.
    /// Declining this path leaves the caller's original query and scale intact.
    fn cpu_quant_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        head_dim: usize,
        native_cache: bool,
        arithmetic: Option<(usize, TextArithmeticPolicy)>,
    ) -> Result<Option<Tensor>> {
        let Self::NativeCpu(QMatMul::QTensor(weight)) = self else {
            return Ok(None);
        };
        if !self.native_cpu_policy()
            || !native_cache
            || !weight.device().is_cpu()
            || weight.dtype().block_size() == 1
            || !q.device().is_cpu()
            || !q.device().same_device(k.device())
            || !q.device().same_device(v.device())
            || q.dtype() != DType::F32
            || k.dtype() != DType::F16
            || v.dtype() != DType::F16
            || [q, k, v].iter().any(|x| x.rank() != 4)
            || head_dim == 0
            || q.dim(3)? != head_dim
        {
            return Ok(None);
        }
        let query = cpu_quant_query(q, head_dim)?;
        // Explicit K requests keep the native attention phase by absolute row.
        // Q8 and legacy calls retain their proved complete-prefix order.
        native_online_attention::attention(&query, k, v, arithmetic)
    }

    fn native_cpu_policy(&self) -> bool {
        matches!(self, Self::NativeCpu(matmul) if native_cpu_quant_weight(matmul))
    }

    fn from_qtensor(qt: Arc<QTensor>, _device: &Device, native_contract: bool) -> Result<Self> {
        #[cfg(target_os = "macos")]
        if let Some(matmul) = super::tensor_qmatmul_metal::MetalTensorQMatMul::try_from_qtensor(
            Arc::clone(&qt),
            _device,
        )? {
            METAL4_QMM_LOGGED.get_or_init(|| {
                tracing::info!("Metal 4 tensor quantized matmul enabled for Phi-3 prefill");
            });
            return Ok(Self::MetalTensor(matmul));
        }

        let matmul = qtensor_to_qmatmul(qt)?;
        if native_contract && native_cpu_quant_weight(&matmul) {
            Ok(Self::NativeCpu(matmul))
        } else {
            Ok(Self::Candle(matmul))
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Candle(matmul) => super::cpu_math::project(matmul, x),
            Self::NativeCpu(matmul) => {
                if let Some(output) = cpu_native_k_projection(matmul, x)? {
                    return Ok(output);
                }
                let adjusted = if native_cpu_quant_weight(matmul) {
                    cpu_quant_input(matmul, x)?
                } else {
                    None
                };
                let input = adjusted.as_ref().unwrap_or(x);
                if let Some(output) = cpu_native_q8_projection(matmul, input)? {
                    return Ok(output);
                }
                super::cpu_math::project(matmul, input)
            }
            #[cfg(target_os = "macos")]
            Self::MetalTensor(matmul) => matmul.forward(x).map_err(Into::into),
        }
    }

    fn request_native_k(&self) -> bool {
        matches!(self, Self::NativeCpu(mm) if native_cpu_k_weight(mm))
    }

    fn forward_rows(&self, x: &Tensor, policy: Option<KRowPolicy>) -> Result<Tensor> {
        if let (Some(policy), Self::NativeCpu(mm)) = (policy, self)
            && let Some(output) = cpu_native_k_projection_rows(mm, x, policy)?
        {
            return Ok(output);
        }
        self.forward(x)
    }

    #[cfg(feature = "cuda")]
    fn candle(&self) -> Option<&QMatMul> {
        match self {
            Self::Candle(matmul) | Self::NativeCpu(matmul) => Some(matmul),
            #[cfg(target_os = "macos")]
            Self::MetalTensor(_) => None,
        }
    }
}

#[derive(Clone, Copy)]
enum KRowPolicy {
    Request { start: usize, cutoff: usize },
    SelectedM1,
}

fn native_cpu_k_weight(matmul: &QMatMul) -> bool {
    matches!(matmul, QMatMul::QTensor(weight)
        if weight.device().is_cpu() && matches!(weight.dtype(), GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K))
}

fn cpu_native_k_projection_rows(
    matmul: &QMatMul,
    x: &Tensor,
    policy: KRowPolicy,
) -> Result<Option<Tensor>> {
    if !native_cpu_k_weight(matmul) {
        return Ok(None);
    }
    match policy {
        KRowPolicy::SelectedM1 => cpu_native_k_projection_order(matmul, x, true),
        KRowPolicy::Request { start, cutoff } => {
            let (batch, rows, _) = x.dims3()?;
            anyhow::ensure!(
                batch == 1 && rows > 0,
                "Phi request arithmetic requires one nonempty text lane"
            );
            let bulk = cutoff.saturating_sub(start).min(rows);
            if bulk == rows {
                return cpu_native_k_projection(matmul, x);
            }
            if bulk == 0 {
                return cpu_native_k_projection_order(matmul, x, true);
            }
            let left = cpu_native_k_projection(matmul, &x.narrow(1, 0, bulk)?)?
                .context("Phi M4 text prefix admission")?;
            let right =
                cpu_native_k_projection_order(matmul, &x.narrow(1, bulk, rows - bulk)?, true)?
                    .context("Phi M1 text suffix admission")?;
            Ok(Some(Tensor::cat(&[&left, &right], 1)?))
        }
    }
}

// Admit only the proved unscaled NeoX geometry. This is a construction guard
// for every part of the compatibility contract, including norms and the head.
fn native_cpu_contract(
    md: &crate::gguf_loader::GgufMetadata,
    rope_dim: usize,
    native_cache: bool,
    has_rope_factors: bool,
) -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && native_cache
        && md.hidden_dim == 3072
        && md.intermediate_dim == 8192
        && md.num_attention_heads == 32
        && md.num_kv_heads == 32
        && md.head_dim == 96
        && rope_dim == 96
        && md.rope_theta == 10000.0
        && md.rope_scaling_factor == 1.0
        && md.rope_scaling_original_context_length == md.context_length
        && md.context_length == 4096
        && md.rope_dimension_sections.is_empty()
        && md.rope_yarn_log_multiplier == 0.0
        && md.rope_yarn_beta_fast == 32.0
        && md.rope_yarn_beta_slow == 1.0
        && md.rope_dimension_count_swa == 0
        && md.head_dim_swa == 0
        && md.rope_theta_swa == 0.0
        && md.sliding_window == 0
        && !has_rope_factors
}

// This explicit compatibility policy follows the measured Apple ARM CPU path.
// Dense weights, other devices/hosts and custom KV configurations keep their
// original operations; forward checks still inspect the actual weight.
fn native_cpu_quant_weight(matmul: &QMatMul) -> bool {
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return false;
    }
    matches!(matmul, QMatMul::QTensor(w) if w.device().is_cpu() && matches!(w.dtype(),
        GgmlDType::Q8_0 | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K))
}

/// Use the proved whole-block integer sum and FMA order for admitted CPU Q8.
/// The caller owns the whole Phi/native-cache contract; this checks the actual
/// projection and input again. Other weight formats keep their existing dot.
fn cpu_native_q8_projection(matmul: &QMatMul, x: &Tensor) -> Result<Option<Tensor>> {
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (matmul, x);
        Ok(None)
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let QMatMul::QTensor(weight) = matmul else {
            return Ok(None);
        };
        if !weight.device().is_cpu()
            || weight.dtype() != GgmlDType::Q8_0
            || !x.device().is_cpu()
            || x.dtype() != DType::F32
            || !x.is_contiguous()
        {
            return Ok(None);
        }
        anyhow::ensure!(
            x.rank() >= 2 && weight.rank() == 2,
            "Phi Q8 projection rank"
        );
        let width = *x.dims().last().context("Phi Q8 input width")?;
        let dims = weight.shape().dims();
        let columns = dims[0];
        anyhow::ensure!(
            width > 0 && width.is_multiple_of(32) && dims[1] == width && columns > 0,
            "Phi Q8 projection shape"
        );
        let rows = x.elem_count() / width;
        anyhow::ensure!(rows > 0, "Phi Q8 input is empty");
        // QTensor::quantize consumes storage, not a tensor's nonzero offset.
        // Copy only the admitted logical span into a fresh zero-offset tensor.
        let (storage, layout) = x.storage_and_layout();
        let candle_core::Storage::Cpu(storage) = &*storage else {
            anyhow::bail!("Phi Q8 projection requires CPU storage");
        };
        let (start, end) = layout.contiguous_offsets().context("Phi Q8 input span")?;
        let values = storage
            .as_slice::<f32>()?
            .get(start..end)
            .context("Phi Q8 storage span")?;
        let packed_input = Tensor::from_vec(values.to_vec(), (rows, width), x.device())?;
        let activations = QTensor::quantize(&packed_input, GgmlDType::Q8_0)?;
        let quantized = activations.data()?;
        let weights = weight.data()?;
        let row_bytes = (width / 32)
            .checked_mul(34)
            .context("Phi Q8 byte overflow")?;
        anyhow::ensure!(
            quantized.len()
                == rows
                    .checked_mul(row_bytes)
                    .context("Phi Q8 activation overflow")?
                && weights.len()
                    == columns
                        .checked_mul(row_bytes)
                        .context("Phi Q8 weight overflow")?,
            "Phi Q8 encoded extent"
        );
        let total = rows
            .checked_mul(columns)
            .context("Phi Q8 output overflow")?;
        let workers = candle_core::utils::get_num_threads().max(1).min(total);
        let chunk = total.div_ceil(workers);
        let slots: Vec<_> = (0..workers).map(|_| std::sync::Mutex::new(None)).collect();
        // Each worker owns complete output elements. No reduction crosses a
        // worker boundary, so changing worker count preserves the dot order.
        candle_core::cpu::kernels::par_for_each(workers, |worker| {
            let begin = worker * chunk;
            let end = (begin + chunk).min(total);
            let result = (|| -> Result<Vec<f32>> {
                if begin >= end {
                    return Ok(Vec::new());
                }
                let mut output = vec![0f32; end - begin];
                let mut at = begin;
                while at < end {
                    let row = at / columns;
                    let column = at % columns;
                    let count = (columns - column).min(end - at);
                    native_q8_dot::project_q8_0(
                        &weights[column * row_bytes..(column + count) * row_bytes],
                        &quantized[row * row_bytes..(row + 1) * row_bytes],
                        width,
                        &mut output[at - begin..at - begin + count],
                    )
                    .map_err(anyhow::Error::msg)?;
                    at += count;
                }
                Ok(output)
            })();
            *slots[worker].lock().expect("new Phi Q8 worker lock") = Some(result);
        });
        let mut output = Vec::with_capacity(total);
        for slot in slots {
            let result = slot
                .into_inner()
                .map_err(|_| anyhow::anyhow!("Phi Q8 worker lock poisoned"))?
                .context("Phi Q8 worker did not finish")?;
            output.extend(result?);
        }
        anyhow::ensure!(output.len() == total, "Phi Q8 output extent");
        let mut shape = x.dims().to_vec();
        *shape.last_mut().context("Phi Q8 output rank")? = columns;
        Ok(Some(Tensor::from_vec(
            output,
            shape.as_slice(),
            x.device(),
        )?))
    }
}

/// Native M4 K-quant order for every row, including decode, to keep chunk
/// invariance. Native Q8K encoding runs directly on the original F32 values;
/// it does not apply the prior Candle ties-away input adjustment.
fn cpu_native_k_projection(matmul: &QMatMul, x: &Tensor) -> Result<Option<Tensor>> {
    cpu_native_k_projection_order(matmul, x, false)
}

fn cpu_native_k_projection_order(matmul: &QMatMul, x: &Tensor, m1: bool) -> Result<Option<Tensor>> {
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (matmul, x, m1);
        Ok(None)
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let QMatMul::QTensor(weight) = matmul else {
            return Ok(None);
        };
        if !weight.device().is_cpu()
            || !matches!(
                weight.dtype(),
                GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K
            )
            || !x.device().is_cpu()
            || x.dtype() != DType::F32
            || !x.is_contiguous()
        {
            return Ok(None);
        }
        anyhow::ensure!(
            x.rank() >= 2 && weight.rank() == 2,
            "Phi K-quant projection rank"
        );
        let width = *x.dims().last().context("Phi K-quant input width")?;
        let dims = weight.shape().dims();
        let columns = dims[0];
        anyhow::ensure!(
            width > 0 && width.is_multiple_of(256) && dims[1] == width && columns > 0,
            "Phi K-quant projection shape"
        );
        let rows = x.elem_count() / width;
        anyhow::ensure!(rows > 0, "Phi K-quant input is empty");
        // Encode only the admitted logical F32 span, including a nonzero
        // tensor offset, with the native Q8_K row encoder.
        let (storage, layout) = x.storage_and_layout();
        let candle_core::Storage::Cpu(storage) = &*storage else {
            anyhow::bail!("Phi K-quant projection requires CPU storage");
        };
        let (start, end) = layout
            .contiguous_offsets()
            .context("Phi K-quant input span")?;
        let values = storage
            .as_slice::<f32>()?
            .get(start..end)
            .context("Phi K-quant storage span")?;
        let quantized = native_q8k::quantize(values).map_err(anyhow::Error::msg)?;
        let kind = match weight.dtype() {
            GgmlDType::Q4K => native_k_dot::Kind::Q4,
            GgmlDType::Q5K => native_k_dot::Kind::Q5,
            GgmlDType::Q6K => native_k_dot::Kind::Q6,
            _ => unreachable!("checked K-quant dtype"),
        };
        let weights = weight.data()?;
        let row_bytes = (width / 256)
            .checked_mul(kind.block_bytes())
            .context("Phi K-quant byte overflow")?;
        let activation_bytes = (width / 256)
            .checked_mul(292)
            .context("Phi Q8K byte overflow")?;
        anyhow::ensure!(
            quantized.len()
                == rows
                    .checked_mul(activation_bytes)
                    .context("Phi K-quant activation overflow")?
                && weights.len()
                    == columns
                        .checked_mul(row_bytes)
                        .context("Phi K-quant weight overflow")?,
            "Phi K-quant encoded extent"
        );
        // Preparation changes integer layout only. Small calls use the same
        // canonical M4 order directly; large calls decode each weight once.
        let prepared = if rows >= 8 && !m1 {
            let weights =
                prepared_k_dot::Prepared::new(kind, &weights, width).map_err(anyhow::Error::msg)?;
            anyhow::ensure!(
                weights.columns() == columns && weights.bytes() <= 256 * 1024 * 1024,
                "Phi prepared K-quant extent"
            );
            let inputs = quantized
                .chunks_exact(activation_bytes)
                .map(|row| prepared_k_dot::Input::new(row, width).map_err(anyhow::Error::msg))
                .collect::<Result<Vec<_>>>()?;
            Some((weights, inputs))
        } else {
            None
        };
        let total = rows
            .checked_mul(columns)
            .context("Phi K-quant output overflow")?;
        let workers = candle_core::utils::get_num_threads().max(1).min(total);
        let chunk = total.div_ceil(workers);
        let slots: Vec<_> = (0..workers).map(|_| std::sync::Mutex::new(None)).collect();
        // Each worker owns complete output elements. No reduction crosses a
        // worker boundary, so changing worker count preserves the dot order.
        candle_core::cpu::kernels::par_for_each(workers, |worker| {
            let begin = worker * chunk;
            let end = (begin + chunk).min(total);
            let result = (|| -> Result<Vec<f32>> {
                if begin >= end {
                    return Ok(Vec::new());
                }
                let mut output = vec![0f32; end - begin];
                let mut at = begin;
                while at < end {
                    let row = at / columns;
                    let column = at % columns;
                    let count = (columns - column).min(end - at);
                    let dst = &mut output[at - begin..at - begin + count];
                    if let Some((weights, inputs)) = &prepared {
                        weights
                            .project(&inputs[row], column, dst)
                            .map_err(anyhow::Error::msg)?;
                    } else {
                        let weights = &weights[column * row_bytes..(column + count) * row_bytes];
                        let input =
                            &quantized[row * activation_bytes..(row + 1) * activation_bytes];
                        if m1 {
                            let kind = match kind {
                                native_k_dot::Kind::Q4 => native_k_dot_m1::Kind::Q4,
                                native_k_dot::Kind::Q5 => native_k_dot_m1::Kind::Q5,
                                native_k_dot::Kind::Q6 => native_k_dot_m1::Kind::Q6,
                            };
                            native_k_dot_m1::project(kind, weights, input, width, dst)
                                .map_err(anyhow::Error::msg)?;
                        } else {
                            native_k_dot::project(kind, weights, input, width, dst)
                                .map_err(anyhow::Error::msg)?;
                        }
                    }
                    at += count;
                }
                Ok(output)
            })();
            *slots[worker].lock().expect("new Phi K-quant worker lock") = Some(result);
        });
        let mut output = Vec::with_capacity(total);
        for slot in slots {
            let result = slot
                .into_inner()
                .map_err(|_| anyhow::anyhow!("Phi K-quant worker lock poisoned"))?
                .context("Phi K-quant worker did not finish")?;
            output.extend(result?);
        }
        anyhow::ensure!(output.len() == total, "Phi K-quant output extent");
        let mut shape = x.dims().to_vec();
        *shape.last_mut().context("Phi K-quant output rank")? = columns;
        Ok(Some(Tensor::from_vec(
            output,
            shape.as_slice(),
            x.device(),
        )?))
    }
}

/// Change only Phi CPU activation-quantizer half ties to nearest-even.
/// Candle still chooses the scales and runs its unchanged weight kernels.
/// Norms, residuals and attention keep the original input tensor.
/// This experiment follows the measured ARM CPU quantizer tie policy.
fn cpu_quant_input(matmul: &QMatMul, x: &Tensor) -> Result<Option<Tensor>> {
    if !cfg!(target_arch = "aarch64") {
        return Ok(None);
    }
    let QMatMul::QTensor(weight) = matmul else {
        return Ok(None);
    };
    if !x.device().is_cpu() || !weight.device().is_cpu() || x.dtype() != DType::F32 {
        return Ok(None);
    }
    let block = match weight.dtype() {
        GgmlDType::Q8_0 => 32,
        GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K => 256,
        _ => return Ok(None),
    };
    // Leave unsupported layouts to Candle's existing contiguous-input guard.
    if !x.is_contiguous() {
        return Ok(None);
    }
    let width = *x.dims().last().context("Phi quantized input needs rows")?;
    anyhow::ensure!(
        x.rank() >= 2
            && width > 0
            && width.is_multiple_of(block)
            && weight.rank() == 2
            && weight.shape().dims()[1] == width,
        "Phi quantized input shape does not match the weight"
    );
    let (storage, layout) = x.storage_and_layout();
    let candle_core::Storage::Cpu(storage) = &*storage else {
        anyhow::bail!("Phi quantized input must use CPU storage");
    };
    let (start, end) = layout
        .contiguous_offsets()
        .context("Phi quantized input span")?;
    let values = storage
        .as_slice::<f32>()?
        .get(start..end)
        .context("Phi quantized input storage span")?;
    even_quant_input_values(values, block)?
        .map(|values| Tensor::from_vec(values, x.shape(), x.device()).map_err(Into::into))
        .transpose()
}

fn even_quant_input_values(values: &[f32], block: usize) -> Result<Option<Vec<f32>>> {
    anyhow::ensure!(
        matches!(block, 32 | 256) && values.len().is_multiple_of(block),
        "Phi quantized input has an incomplete block"
    );
    let mut adjusted: Option<Vec<f32>> = None;
    for (block_index, row) in values.chunks_exact(block).enumerate() {
        // Keep the existing quantizer's behavior on nonfinite or degenerate data.
        if row.iter().any(|x| !x.is_finite()) {
            continue;
        }
        let amax = row.iter().fold(0.0_f32, |m, x| m.max(x.abs()));
        let multiplier = if block == 32 {
            let scale = amax / 127.0_f32;
            if scale == 0.0 { 0.0 } else { 1.0_f32 / scale }
        } else {
            // Q8_K uses -127 / signed_max. Its magnitude has the same tie
            // positions for either sign. Keep every maximum unchanged, so
            // Candle retains its signed-max choice and scale bits.
            127.0_f32 / amax
        };
        if multiplier == 0.0 || !multiplier.is_finite() {
            continue;
        }
        for (lane, &value) in row.iter().enumerate() {
            let product = value * multiplier;
            let target = product.round_ties_even();
            if product.round() == target {
                continue;
            }
            anyhow::ensure!(
                product.is_finite() && value.abs() < amax,
                "Phi quantizer tie must not change a block maximum"
            );
            // Only odd ties-away codes need a change, always toward zero.
            // One F32 step may still multiply to the same half tie. Take the
            // first step that gives the desired code, with a checked bound.
            let mut changed = value;
            for _ in 0..4 {
                changed = f32::from_bits(changed.to_bits() - 1);
                if (changed * multiplier).round() == target {
                    break;
                }
            }
            anyhow::ensure!(
                (changed * multiplier).round() == target,
                "Phi quantizer tie adjustment exceeded four F32 steps"
            );
            adjusted.get_or_insert_with(|| values.to_vec())[block_index * block + lane] = changed;
        }
    }
    Ok(adjusted)
}

fn cpu_quant_query(q: &Tensor, head_dim: usize) -> Result<Tensor> {
    anyhow::ensure!(head_dim > 0, "query head width must be positive");
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    // A binary multiply preserves signed zero; scalar affine adds +0.
    let scaled = q.broadcast_mul(&Tensor::new(scale, q.device())?)?;
    Ok(scaled.to_dtype(DType::F16)?.to_dtype(DType::F32)?)
}

struct RmsNorm {
    weight: Tensor,
    eps: f64,
    native_cpu: bool,
}

impl RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.native_cpu && x.device().is_cpu() && x.dtype() == DType::F32 {
            return cpu_rms_norm(x, &self.weight, self.eps);
        }
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = (&x * &x)?.mean_keepdim(candle_core::D::Minus1)?;
        #[cfg(target_os = "macos")]
        if let Some(result) = super::fused_norm_metal::try_fused_rms_norm_epilogue(
            &x,
            &variance,
            &self.weight,
            self.eps as f32,
        )? {
            return result.to_dtype(dtype).map_err(Into::into);
        }
        let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let result = x_normed.to_dtype(dtype)?.broadcast_mul(&self.weight)?;
        Ok(result)
    }
}

/// Keep Phi CPU normalization stable before activation quantization.
/// Accumulate F32 squared terms in F64, then use the CPU reference's F32
/// mean, reciprocal and two multiplies. This improves the sum's precision;
/// it does not evaluate the whole normalization in F64.
fn cpu_rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    anyhow::ensure!(
        x.device().is_cpu()
            && weight.device().is_cpu()
            && x.dtype() == DType::F32
            && weight.dtype() == DType::F32,
        "Phi CPU RMSNorm requires F32 CPU input and weight"
    );
    let width = *x
        .dims()
        .last()
        .context("Phi CPU RMSNorm requires a row dimension")?;
    anyhow::ensure!(
        width > 0 && weight.dims() == [width],
        "Phi CPU RMSNorm weight must match the nonempty last input dimension"
    );
    anyhow::ensure!(
        eps.is_finite() && eps >= 0.0 && (eps as f32).is_finite(),
        "Phi CPU RMSNorm epsilon must be finite and nonnegative in F32"
    );
    if x.elem_count() == 0 {
        return Tensor::zeros(x.shape(), DType::F32, x.device()).map_err(Into::into);
    }
    // Contiguous tensors share storage; only strided views need a copy.
    let x = x.contiguous()?;
    let weight = weight.contiguous()?;
    let (xs, xl) = x.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let (candle_core::Storage::Cpu(xs), candle_core::Storage::Cpu(ws)) = (&*xs, &*ws) else {
        anyhow::bail!("Phi CPU RMSNorm storage must be on CPU");
    };
    let (x_start, x_end) = xl
        .contiguous_offsets()
        .context("Phi CPU RMSNorm input layout")?;
    let (w_start, w_end) = wl
        .contiguous_offsets()
        .context("Phi CPU RMSNorm weight layout")?;
    let values = xs
        .as_slice::<f32>()?
        .get(x_start..x_end)
        .context("Phi CPU RMSNorm input span")?;
    let weights = ws
        .as_slice::<f32>()?
        .get(w_start..w_end)
        .context("Phi CPU RMSNorm weight span")?;
    let mut output = Vec::with_capacity(values.len());
    for row in values.chunks_exact(width) {
        let sum = row
            .iter()
            .map(|&value| f64::from(value * value))
            .sum::<f64>();
        let mean = (sum / width as f64) as f32;
        let scale = 1.0_f32 / (mean + eps as f32).sqrt();
        output.extend(
            row.iter()
                .zip(weights)
                .map(|(&value, &weight)| (value * scale) * weight),
        );
    }
    Tensor::from_vec(output, x.shape(), x.device()).map_err(Into::into)
}

impl Attention {
    fn rotary(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if self.qkv_proj.native_cpu_policy()
            && self.quant_config.is_none()
            && x.device().is_cpu()
            && x.dtype() == DType::F32
        {
            return native_cpu_rotary(x, start_pos, self.rope_theta, self.rope_cos.dim(0)?);
        }
        apply_rotary_emb(x, start_pos, &self.rope_cos, &self.rope_sin)
    }

    fn forward(
        &self,
        x: &Tensor,
        kv_cache: &mut Option<KvCache>,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<Tensor> {
        self.forward_rows(x, kv_cache, seq_len, start_pos, None, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_rows(
        &self,
        x: &Tensor,
        kv_cache: &mut Option<KvCache>,
        seq_len: usize,
        start_pos: usize,
        row_policy: Option<KRowPolicy>,
        arithmetic: Option<TextArithmeticPolicy>,
    ) -> Result<Tensor> {
        let (b_sz, _, _) = x.dims3()?;
        let profile = layer_profile_enabled();

        // Fused QKV projection
        let started = profile.then(Instant::now);
        let qkv = self.qkv_proj.forward_rows(x, row_policy)?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &QKV_PROJ_NS)?;
        }

        // Split: Q = [num_heads * head_dim], K = [num_kv_heads * head_dim], V = [num_kv_heads * head_dim]
        let started = profile.then(Instant::now);
        let q_size = self.num_heads * self.head_dim;
        let kv_size = self.num_kv_heads * self.head_dim;
        let q = qkv.narrow(candle_core::D::Minus1, 0, q_size)?;
        let k = qkv.narrow(candle_core::D::Minus1, q_size, kv_size)?;
        let v = qkv.narrow(candle_core::D::Minus1, q_size + kv_size, kv_size)?;

        // Reshape to [batch, heads, seq_len, head_dim]
        let (q, k, v) = if seq_len == 1 {
            let q = q.reshape((b_sz, self.num_heads, 1, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, 1, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, 1, self.head_dim))?;
            (q, k, v)
        } else {
            #[cfg(target_os = "macos")]
            if let Some(layout) = super::qkv_layout_metal::try_fused_qkv_layout(
                &qkv,
                b_sz,
                seq_len,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )? {
                layout
            } else {
                self.portable_qkv_layout(q, k, v, b_sz, seq_len)?
            }
            #[cfg(not(target_os = "macos"))]
            self.portable_qkv_layout(q, k, v, b_sz, seq_len)?
        };
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &QKV_LAYOUT_NS)?;
        }

        // Apply RoPE (partial if rope_dim < head_dim)
        let started = profile.then(Instant::now);
        let (q, k) = if self.rope_dim < self.head_dim {
            // Partial RoPE: rotate first rope_dim dims, pass-through the rest
            let q_rot = q.narrow(candle_core::D::Minus1, 0, self.rope_dim)?;
            let q_pass = q.narrow(
                candle_core::D::Minus1,
                self.rope_dim,
                self.head_dim - self.rope_dim,
            )?;
            let k_rot = k.narrow(candle_core::D::Minus1, 0, self.rope_dim)?;
            let k_pass = k.narrow(
                candle_core::D::Minus1,
                self.rope_dim,
                self.head_dim - self.rope_dim,
            )?;

            let q_rot = self.rotary(&q_rot, start_pos)?;
            let k_rot = self.rotary(&k_rot, start_pos)?;

            let q = Tensor::cat(&[&q_rot, &q_pass], candle_core::D::Minus1)?.contiguous()?;
            let k = Tensor::cat(&[&k_rot, &k_pass], candle_core::D::Minus1)?.contiguous()?;
            (q, k)
        } else {
            let q = self.rotary(&q, start_pos)?;
            let k = self.rotary(&k, start_pos)?;
            (q, k)
        };
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &ROPE_NS)?;
        }

        let started = profile.then(Instant::now);
        let (q, k, v, rotation) =
            super::prepare_quantized_attention(kv_cache, q, k, v, self.quant_config.as_deref())?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &KV_UPDATE_NS)?;
        }
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let started = profile.then(Instant::now);
        let attn_output = if let Some(output) =
            super::try_fused_f16_sdpa_native(&q, &k, &v, seq_len > 1, scale as f32)?
        {
            output
        } else if let Some(output) = self.qkv_proj.cpu_quant_attention(
            &q,
            &k,
            &v,
            self.head_dim,
            self.quant_config.is_none(),
            arithmetic.map(|policy| (start_pos, policy)),
        )? {
            output
        } else if let Some(output) = super::cpu_math::attention(&q, &k, &v, scale)? {
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
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &SDPA_NS)?;
        }

        let started = profile.then(Instant::now);
        #[cfg(target_os = "macos")]
        let attn_output = if let Some(output) =
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
        if let Some(o_proj) = self.o_proj.candle()
            && let Some(output) =
                super::dense_qmatmul_cuda::forward_one_to_f32(&attn_output, o_proj)?
        {
            if let Some(started) = started {
                finish_profile_region(x.device(), started, &ATTN_OUTPUT_PROJ_NS)?;
            }
            return Ok(output);
        }
        #[cfg(not(target_os = "macos"))]
        let attn_output = attn_output.to_dtype(x.dtype())?;
        let output = self.o_proj.forward_rows(&attn_output, row_policy)?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &ATTN_OUTPUT_PROJ_NS)?;
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
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_rows(x, None)
    }

    fn forward_rows(&self, x: &Tensor, row_policy: Option<KRowPolicy>) -> Result<Tensor> {
        let profile = layer_profile_enabled();

        // Fused gate+up projection: [batch, seq, 2 * intermediate_dim]
        let started = profile.then(Instant::now);
        let gate_up = self.gate_up_proj.forward_rows(x, row_policy)?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &GATE_UP_PROJ_NS)?;
        }

        #[cfg(feature = "cuda")]
        if !profile && let Some(down_proj) = self.down_proj.candle() {
            let gate = gate_up.narrow(candle_core::D::Minus1, 0, self.intermediate_dim)?;
            let up = gate_up.narrow(
                candle_core::D::Minus1,
                self.intermediate_dim,
                self.intermediate_dim,
            )?;
            if let Some(output) = super::dense_qmatmul_cuda::silu_mul_down(&gate, &up, down_proj)? {
                return Ok(output);
            }
        }

        // Split into gate (first half) and up (second half)
        let started = profile.then(Instant::now);
        #[cfg(target_os = "macos")]
        let activated = if gate_up.device().is_metal() {
            super::indexed_moe_metal::fused_swiglu(&gate_up)?
        } else {
            self.portable_swiglu(&gate_up)?
        };
        #[cfg(not(target_os = "macos"))]
        let activated = self.portable_swiglu(&gate_up)?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &FFN_ACTIVATION_NS)?;
        }
        let started = profile.then(Instant::now);
        let down = self.down_proj.forward_rows(&activated, row_policy)?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &DOWN_PROJ_NS)?;
        }
        Ok(down)
    }

    fn portable_swiglu(&self, gate_up: &Tensor) -> Result<Tensor> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if self.gate_up_proj.native_cpu_policy()
            && self.down_proj.native_cpu_policy()
            && gate_up.device().is_cpu()
            && gate_up.dtype() == DType::F32
            && self.intermediate_dim == 8192
        {
            return native_cpu_swiglu(gate_up);
        }
        let gate = gate_up.narrow(candle_core::D::Minus1, 0, self.intermediate_dim)?;
        let up = gate_up.narrow(
            candle_core::D::Minus1,
            self.intermediate_dim,
            self.intermediate_dim,
        )?;
        (candle_nn::Activation::Silu.forward(&gate)? * up).map_err(Into::into)
    }
}

impl TransformerBlock {
    fn forward(
        &self,
        x: &Tensor,
        kv_cache: &mut Option<KvCache>,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<Tensor> {
        self.forward_rows(x, kv_cache, seq_len, start_pos, None, false, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_rows(
        &self,
        x: &Tensor,
        kv_cache: &mut Option<KvCache>,
        seq_len: usize,
        start_pos: usize,
        row_policy: Option<KRowPolicy>,
        selected_output: bool,
        arithmetic: Option<TextArithmeticPolicy>,
    ) -> Result<Tensor> {
        let profile = layer_profile_enabled();
        if profile {
            PROFILE_CALLS.fetch_add(1, Ordering::Relaxed);
        }

        let residual = x;
        let started = profile.then(Instant::now);
        let x = self.attn_norm.forward(x)?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &ATTN_NORM_NS)?;
        }
        let started = profile.then(Instant::now);
        let x = if row_policy.is_some() {
            self.attn
                .forward_rows(&x, kv_cache, seq_len, start_pos, row_policy, arithmetic)?
        } else {
            self.attn.forward(&x, kv_cache, seq_len, start_pos)?
        };
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &ATTN_NS)?;
        }
        let started = profile.then(Instant::now);
        let x = (residual + x)?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &ATTN_RESIDUAL_NS)?;
        }

        // The final stage returns only this row. Attention has already stored
        // every K/V row, so this does not shorten or alter the causal cache.
        let x = if selected_output {
            x.narrow(1, seq_len - 1, 1)?
        } else {
            x
        };
        let ffn_policy = if selected_output {
            Some(KRowPolicy::SelectedM1)
        } else {
            row_policy
        };
        let residual = &x;
        let started = profile.then(Instant::now);
        let ffn_out = self.ffn_norm.forward(&x)?;
        if let Some(started) = started {
            finish_profile_region(ffn_out.device(), started, &FFN_NORM_NS)?;
        }
        let started = profile.then(Instant::now);
        let ffn_out = if ffn_policy.is_some() {
            self.ffn.forward_rows(&ffn_out, ffn_policy)?
        } else {
            self.ffn.forward(&ffn_out)?
        };
        if let Some(started) = started {
            finish_profile_region(ffn_out.device(), started, &FFN_NS)?;
        }
        let started = profile.then(Instant::now);
        let x = (residual + ffn_out)?;
        if let Some(started) = started {
            finish_profile_region(x.device(), started, &FFN_RESIDUAL_NS)?;
        }

        Ok(x)
    }
}

fn qtensor_to_qmatmul(qt: Arc<QTensor>) -> Result<QMatMul> {
    QMatMul::from_arc(qt).map_err(Into::into)
}

impl Phi3Stage {
    /// Construct a Phi-3 stage from its assigned GGUF tensors.
    ///
    /// Returns an error when required fused projections, metadata, or partial
    /// RoPE dimensions are missing or inconsistent.
    pub fn from_stage_tensors(
        stage: StageTensors,
        quant_config: Option<Arc<TurboQuantConfig>>,
    ) -> Result<Self> {
        let md = &stage.metadata;
        let device = &stage.device;
        let range = &stage.range;
        let tensors = &stage.tensors;
        // Check partial rotary width before dequantizing the embedding or
        // building tables from metadata. Zero selects the full head width.
        let rope_dim = if md.rope_dimension_count > 0 {
            md.rope_dimension_count
        } else {
            md.head_dim
        };
        crate::gguf_loader::validate_rope_dimensions(
            rope_dim,
            md.head_dim,
            "phi3.rope.dimension_count",
        )?;
        validate_attention_geometry(&stage)?;
        #[cfg(feature = "cuda")]
        super::dense_qmatmul_cuda::precompile(device)?;

        // Embedding (head only)
        let embedding = if range.is_head() {
            let embd = tensors
                .get("token_embd.weight")
                .context("missing token_embd.weight for head stage")?;
            Some(embd.as_ref().dequantize(device)?)
        } else {
            None
        };

        let (rope_cos, rope_sin) =
            precompute_rope_cos_sin(rope_dim, md.rope_theta, md.context_length, device)?;

        // Transformer blocks
        let mut blocks = Vec::with_capacity(range.layer_count());
        let native_contract = native_cpu_contract(
            md,
            rope_dim,
            quant_config.is_none(),
            tensors
                .keys()
                .any(|name| name.starts_with("rope_freqs") || name.starts_with("rope_factors")),
        );
        let build_context = BlockBuildContext {
            tensors,
            metadata: md,
            device,
            rope_cos: &rope_cos,
            rope_sin: &rope_sin,
            rope_dim,
            native_contract,
            quant_config,
        };
        for layer_idx in range.start..range.end {
            let prefix = format!("blk.{layer_idx}");
            let block = build_block(&build_context, &prefix)?;
            blocks.push(block);
        }

        // Output norm + lm_head (tail only)
        let lm_head = if range.is_tail() {
            let head = tensors
                .get("output.weight")
                .or_else(|| tensors.get("token_embd.weight"))
                .context("missing output.weight for tail stage")?;
            Some(qtensor_to_qmatmul(Arc::clone(head))?)
        } else {
            None
        };

        let head_native_cpu =
            native_contract && lm_head.as_ref().is_some_and(native_cpu_quant_weight);

        let output_norm = if range.is_tail() {
            let norm_w = tensors
                .get("output_norm.weight")
                .context("missing output_norm.weight for tail stage")?;
            Some(RmsNorm {
                weight: norm_w.as_ref().dequantize(device)?,
                eps: md.rms_norm_eps,
                native_cpu: head_native_cpu,
            })
        } else {
            None
        };

        Ok(Self {
            embedding,
            blocks,
            output_norm,
            lm_head,
            head_native_cpu,
            is_head: range.is_head(),
            is_tail: range.is_tail(),
            hidden_dim: md.hidden_dim,
            num_layers: range.layer_count(),
            vocab_size: md.vocab_size,
            device: device.clone(),
        })
    }
}

impl Phi3Stage {
    fn forward_rows(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
        arithmetic: Option<TextArithmeticPolicy>,
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

        for (i, block) in self.blocks.iter().enumerate() {
            hidden_states = with_kv_cache(&mut cache[i], |kv| {
                if let Some(policy) = arithmetic {
                    block.forward_rows(
                        &hidden_states,
                        kv,
                        seq_len,
                        start_pos,
                        Some(KRowPolicy::Request {
                            start: start_pos,
                            cutoff: policy.m4_prefix_end(),
                        }),
                        self.is_tail && i + 1 == self.blocks.len(),
                        Some(policy),
                    )
                } else {
                    block.forward(&hidden_states, kv, seq_len, start_pos)
                }
            })?;
        }

        if self.is_tail {
            let normed = self.output_norm.as_ref().unwrap().forward(&hidden_states)?;
            let output_rows = normed.dim(1)?;
            let last_hidden = if output_rows > 1 {
                normed.i((.., output_rows - 1, ..))?
            } else {
                normed.squeeze(1)?
            };
            let head = self.lm_head.as_ref().unwrap();
            let k_logits = if self.head_native_cpu {
                cpu_native_k_projection_order(head, &last_hidden, arithmetic.is_some())?
            } else {
                None
            };
            let logits = if let Some(logits) = k_logits {
                logits
            } else {
                let adjusted = if self.head_native_cpu && native_cpu_quant_weight(head) {
                    cpu_quant_input(head, &last_hidden)?
                } else {
                    None
                };
                let head_input = adjusted.as_ref().unwrap_or(&last_hidden);
                if self.head_native_cpu {
                    match cpu_native_q8_projection(head, head_input)? {
                        Some(logits) => logits,
                        None => head.forward(head_input)?,
                    }
                } else {
                    head.forward(head_input)?
                }
            };
            let logits = logits.squeeze(0)?;
            Ok(StageOutput::Logits(logits))
        } else {
            let output = hidden_states.squeeze(0)?;
            Ok(StageOutput::Activations(output))
        }
    }
}

impl ModelArchitecture for Phi3Stage {
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
        let supported = !self.blocks.is_empty()
            && self.blocks.iter().all(|block| {
                block.attn.qkv_proj.request_native_k()
                    && block.attn.o_proj.request_native_k()
                    && block.ffn.gate_up_proj.request_native_k()
                    && block.ffn.down_proj.request_native_k()
            })
            && self
                .lm_head
                .as_ref()
                .is_none_or(|head| self.head_native_cpu && native_cpu_k_weight(head));
        supported.then(|| TextArithmeticPolicy::phi_k_native_attention(context))
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
        _cache_only: bool,
    ) -> Result<StageOutput> {
        context.validate(start_pos, seq_len, prefill)?;
        self.forward_rows(
            input,
            cache,
            seq_len,
            start_pos,
            self.text_arithmetic_policy(context),
        )
    }

    fn family(&self) -> &str {
        "phi"
    }
    fn prefers_f32_decode_activations(&self) -> bool {
        true
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

struct BlockBuildContext<'a> {
    tensors: &'a std::collections::HashMap<String, Arc<QTensor>>,
    metadata: &'a crate::gguf_loader::GgufMetadata,
    device: &'a Device,
    rope_cos: &'a Tensor,
    rope_sin: &'a Tensor,
    rope_dim: usize,
    native_contract: bool,
    quant_config: Option<Arc<TurboQuantConfig>>,
}

fn build_block(context: &BlockBuildContext<'_>, prefix: &str) -> Result<TransformerBlock> {
    let BlockBuildContext {
        tensors,
        metadata: md,
        device,
        rope_cos,
        rope_sin,
        rope_dim,
        native_contract,
        quant_config,
    } = context;
    let get = |name: &str| -> Result<Arc<QTensor>> {
        tensors
            .get(&format!("{prefix}.{name}"))
            .cloned()
            .with_context(|| format!("missing tensor: {prefix}.{name}"))
    };

    let qkv_proj = Phi3QMatMul::from_qtensor(get("attn_qkv.weight")?, device, *native_contract)?;
    let attn_norm = RmsNorm {
        weight: get("attn_norm.weight")?.as_ref().dequantize(device)?,
        eps: md.rms_norm_eps,
        native_cpu: qkv_proj.native_cpu_policy(),
    };

    let attn = Attention {
        qkv_proj,
        o_proj: Phi3QMatMul::from_qtensor(get("attn_output.weight")?, device, *native_contract)?,
        num_heads: md.num_attention_heads,
        num_kv_heads: md.num_kv_heads,
        head_dim: md.head_dim,
        rope_dim: *rope_dim,
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        rope_theta: md.rope_theta as f32,
        rope_cos: (*rope_cos).clone(),
        rope_sin: (*rope_sin).clone(),
        quant_config: quant_config.clone(),
    };

    let gate_up_proj = Phi3QMatMul::from_qtensor(get("ffn_up.weight")?, device, *native_contract)?;
    let ffn_norm = RmsNorm {
        weight: get("ffn_norm.weight")?.as_ref().dequantize(device)?,
        eps: md.rms_norm_eps,
        native_cpu: gate_up_proj.native_cpu_policy(),
    };

    let ffn = Mlp {
        gate_up_proj,
        down_proj: Phi3QMatMul::from_qtensor(get("ffn_down.weight")?, device, *native_contract)?,
        intermediate_dim: md.intermediate_dim,
    };

    Ok(TransformerBlock {
        attn_norm,
        attn,
        ffn_norm,
        ffn,
    })
}

fn validate_attention_geometry(stage: &StageTensors) -> Result<()> {
    stage.range.validate()?;
    let md = &stage.metadata;
    anyhow::ensure!(
        md.num_attention_heads > 0
            && md.num_kv_heads > 0
            && md.num_attention_heads.is_multiple_of(md.num_kv_heads),
        "Phi-3 attention heads must be a positive multiple of KV heads"
    );
    let query_width = md
        .num_attention_heads
        .checked_mul(md.head_dim)
        .context("Phi-3 query width overflow")?;
    let kv_width = md
        .num_kv_heads
        .checked_mul(md.head_dim)
        .context("Phi-3 KV width overflow")?;
    let qkv_width = kv_width
        .checked_mul(2)
        .and_then(|width| query_width.checked_add(width))
        .context("Phi-3 fused QKV width overflow")?;
    for layer in stage.range.start..stage.range.end {
        for (suffix, expected) in [
            ("attn_qkv.weight", [qkv_width, md.hidden_dim]),
            ("attn_output.weight", [md.hidden_dim, query_width]),
        ] {
            let name = format!("blk.{layer}.{suffix}");
            let tensor = stage
                .tensors
                .get(&name)
                .with_context(|| format!("missing tensor: {name}"))?;
            anyhow::ensure!(
                tensor.shape().dims() == expected,
                "{name} shape {:?} does not match attention metadata {expected:?}",
                tensor.shape().dims()
            );
        }
    }
    Ok(())
}

fn precompute_rope_cos_sin(
    dim: usize,
    theta: f64,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    crate::gguf_loader::validate_rope_dimensions(dim, dim, "Phi-3 rotary dimension")?;
    anyhow::ensure!(
        context_length > 0 && context_length <= u32::MAX as usize,
        "Phi-3 rotary context must be positive and fit U32 positions"
    );
    let half_dim = dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / dim as f32))
        .collect();
    let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;

    let positions = Tensor::arange(0u32, context_length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?;

    let angles = positions.matmul(&inv_freq.reshape((1, half_dim))?)?;
    let cos = angles.cos()?;
    let sin = angles.sin()?;
    Ok((cos, sin))
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn native_cpu_swiglu(gate_up: &Tensor) -> Result<Tensor> {
    let (batch, queries, width) = gate_up.dims3()?;
    anyhow::ensure!(
        gate_up.device().is_cpu() && gate_up.dtype() == DType::F32 && width == 16384,
        "native CPU SwiGLU requires F32 CPU and the proved fused width"
    );
    // Every row has 8192 gate and 8192 up values, so the native four-lane
    // grouping has no scalar tail and never crosses a token boundary.
    let rows = gate_up.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let mut output = Vec::with_capacity(rows.len() / 2);
    for row in rows.chunks_exact(width) {
        output.extend(native_swiglu_compat::native_swiglu(
            &row[..8192],
            &row[8192..],
        ));
    }
    Tensor::from_vec(output, (batch, queries, 8192), gate_up.device()).map_err(Into::into)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn native_cpu_rotary(x: &Tensor, start: usize, base: f32, context: usize) -> Result<Tensor> {
    let (batch, heads, queries, dim) = x.dims4()?;
    anyhow::ensure!(
        x.device().is_cpu()
            && x.dtype() == DType::F32
            && dim > 0
            && dim.is_multiple_of(2)
            && base.is_finite()
            && base > 0.0,
        "native CPU rotary requires F32 CPU and a positive even width/base"
    );
    let end = start
        .checked_add(queries)
        .context("native CPU rotary position overflow")?;
    anyhow::ensure!(
        end <= context,
        "native CPU rotary exceeds the original context"
    );
    let rows = x
        .transpose(1, 2)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let width = heads
        .checked_mul(dim)
        .context("native CPU rotary head extent overflow")?;
    if width == 0 || queries == 0 || batch == 0 {
        return Tensor::zeros(x.shape(), x.dtype(), x.device()).map_err(Into::into);
    }
    let mut output = Vec::with_capacity(rows.len());
    for (row, input) in rows.chunks_exact(width).enumerate() {
        let position = start + row % queries;
        let (_, cos, sin, _) = native_rope_compat::native_rope_table(position, dim, base);
        output.extend(native_rope_compat::native_rope_rotate(
            input, dim, &cos, &sin,
        ));
    }
    Tensor::from_vec(output, (batch, queries, heads, dim), x.device())?
        .transpose(1, 2)?
        .contiguous()
        .map_err(Into::into)
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
mod tests {
    use super::*;

    #[test]
    fn native_contract_requires_proved_geometry_and_actual_quant_weights() -> Result<()> {
        let mut md = metadata()?;
        md.hidden_dim = 3072;
        md.intermediate_dim = 8192;
        md.num_attention_heads = 32;
        md.num_kv_heads = 32;
        md.head_dim = 96;
        md.rope_theta = 10000.0;
        md.context_length = 4096;
        md.rope_scaling_original_context_length = 4096;
        let host = cfg!(all(target_os = "macos", target_arch = "aarch64"));
        assert_eq!(native_cpu_contract(&md, 96, true, false), host);
        assert!(!native_cpu_contract(&md, 96, false, false));
        assert!(!native_cpu_contract(&md, 96, true, true));
        assert!(!native_cpu_contract(&md, 48, true, false));
        md.intermediate_dim = 8191;
        assert!(!native_cpu_contract(&md, 96, true, false));
        md.intermediate_dim = 8192;
        md.rope_scaling_factor = 2.0;
        assert!(!native_cpu_contract(&md, 96, true, false));
        md.rope_scaling_factor = 1.0;
        md.num_kv_heads = 8;
        assert!(!native_cpu_contract(&md, 96, true, false));
        md.num_kv_heads = 32;
        md.context_length = 8192;
        assert!(!native_cpu_contract(&md, 96, true, false));
        let w = Tensor::ones((32, 32), DType::F32, &Device::Cpu)?;
        let quant = Arc::new(QTensor::quantize(&w, GgmlDType::Q8_0)?);
        let enabled = Phi3QMatMul::from_qtensor(Arc::clone(&quant), &Device::Cpu, true)?;
        let disabled = Phi3QMatMul::from_qtensor(quant, &Device::Cpu, false)?;
        assert_eq!(enabled.native_cpu_policy(), host);
        assert!(!disabled.native_cpu_policy());
        let dense = Phi3QMatMul::from_qtensor(
            Arc::new(QTensor::quantize(&w, GgmlDType::F32)?),
            &Device::Cpu,
            true,
        )?;
        assert!(!dense.native_cpu_policy());
        Ok(())
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn native_q8_projection_preserves_offset_and_row_chunking() -> Result<()> {
        let (rows, width, outputs) = (7, 64, 13);
        let weights = Tensor::from_vec(
            (0..width * outputs)
                .map(|i| ((i * 17 % 257) as f32 - 128.0) / 512.0)
                .collect::<Vec<_>>(),
            (outputs, width),
            &Device::Cpu,
        )?;
        let matmul = QMatMul::QTensor(Arc::new(QTensor::quantize(&weights, GgmlDType::Q8_0)?));
        let mut values = vec![12345.0f32; width];
        values.extend((0..rows * width).map(|i| ((i * 31 % 251) as f32 - 125.0) / 256.0));
        let storage = Tensor::from_vec(values, (rows + 1, width), &Device::Cpu)?;
        let view = storage.narrow(0, 1, rows)?;
        assert_eq!(view.layout().start_offset(), width);
        let expected = cpu_native_q8_projection(&matmul, &view)?.context("Q8 admitted")?;
        let direct = Tensor::from_vec(
            view.flatten_all()?.to_vec1::<f32>()?,
            (rows, width),
            &Device::Cpu,
        )?;
        let direct = cpu_native_q8_projection(&matmul, &direct)?.context("Q8 admitted")?;
        let bits = |x: &Tensor| -> Result<Vec<u32>> {
            Ok(x.flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(f32::to_bits)
                .collect())
        };
        assert_eq!(bits(&expected)?, bits(&direct)?);
        for chunk in [1, 3, 5] {
            let mut parts = Vec::new();
            for start in (0..rows).step_by(chunk) {
                parts.push(
                    cpu_native_q8_projection(
                        &matmul,
                        &view.narrow(0, start, chunk.min(rows - start))?,
                    )?
                    .context("Q8 admitted")?,
                );
            }
            assert_eq!(bits(&expected)?, bits(&Tensor::cat(&parts, 0)?)?);
        }
        Ok(())
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn native_q8_projection_declines_other_formats_and_noncontiguous_inputs() -> Result<()> {
        let weights = Tensor::ones((4, 256), DType::F32, &Device::Cpu)?;
        let input = Tensor::ones((2, 256), DType::F32, &Device::Cpu)?;
        for dtype in [
            GgmlDType::F32,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
        ] {
            let matmul = QMatMul::QTensor(Arc::new(QTensor::quantize(&weights, dtype)?));
            assert!(cpu_native_q8_projection(&matmul, &input)?.is_none());
        }
        let dense = QMatMul::Tensor(weights.clone());
        assert!(cpu_native_q8_projection(&dense, &input)?.is_none());
        let quant = QMatMul::QTensor(Arc::new(QTensor::quantize(&weights, GgmlDType::Q8_0)?));
        assert!(cpu_native_q8_projection(&quant, &input.to_dtype(DType::F16)?)?.is_none());
        let transposed = Tensor::ones((256, 2), DType::F32, &Device::Cpu)?.t()?;
        assert!(!transposed.is_contiguous());
        assert!(cpu_native_q8_projection(&quant, &transposed)?.is_none());
        assert!(
            cpu_native_q8_projection(&quant, &Tensor::ones((1, 32), DType::F32, &Device::Cpu)?)
                .is_err()
        );
        Ok(())
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn native_q8_route_preserves_original_fallback_and_proved_byte_helper() -> Result<()> {
        let width = 64;
        let weights = Tensor::from_vec(
            (0..width * 4)
                .map(|i| ((i * 7 % 63) as f32 - 31.0) / 32.0)
                .collect::<Vec<_>>(),
            (4, width),
            &Device::Cpu,
        )?;
        let weight = Arc::new(QTensor::quantize(&weights, GgmlDType::Q8_0)?);
        let input = Tensor::from_vec(
            (0..3 * width)
                .map(|i| ((i * 13 % 127) as f32 - 63.0) / 64.0)
                .collect::<Vec<_>>(),
            (1, 3, width),
            &Device::Cpu,
        )?;
        let original = QMatMul::QTensor(Arc::clone(&weight));
        let fallback = Phi3QMatMul::Candle(QMatMul::QTensor(Arc::clone(&weight)));
        assert_eq!(
            fallback.forward(&input)?.flatten_all()?.to_vec1::<f32>()?,
            original.forward(&input)?.flatten_all()?.to_vec1::<f32>()?
        );
        let adjusted = cpu_quant_input(&original, &input)?;
        let input = adjusted.as_ref().unwrap_or(&input);
        let encoded = QTensor::quantize(&input.contiguous()?, GgmlDType::Q8_0)?;
        let bytes = encoded.data()?;
        let weight_bytes = weight.data()?;
        let mut expected = [0f32; 12];
        for row in 0..3 {
            native_q8_dot::project_q8_0(
                &weight_bytes,
                &bytes[row * 68..(row + 1) * 68],
                width,
                &mut expected[row * 4..(row + 1) * 4],
            )
            .map_err(anyhow::Error::msg)?;
        }
        let actual = cpu_native_q8_projection(&original, input)?
            .context("Q8 admitted")?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn native_rotary_adapter_keeps_row_positions_and_chunk_views() -> Result<()> {
        let (batch, heads, queries, dim, start) = (2, 3, 7, 96, 3260);
        let input = Tensor::from_vec(
            (0..batch * heads * queries * dim)
                .map(|i| ((i * 17 % 257) as f32 - 128.0) / 128.0)
                .collect::<Vec<_>>(),
            (batch, heads, queries, dim),
            &Device::Cpu,
        )?;
        let baseline = native_cpu_rotary(&input, start, 10000.0, 4096)?;
        let values = baseline.flatten_all()?.to_vec1::<f32>()?;
        let original = input.flatten_all()?.to_vec1::<f32>()?;
        for b in 0..batch {
            for h in 0..heads {
                for row in 0..queries {
                    let offset = ((b * heads + h) * queries + row) * dim;
                    let (_, cos, sin, factor) =
                        native_rope_compat::native_rope_table(start + row, dim, 10000.0);
                    assert_eq!(factor.to_bits(), 0x3f534db0);
                    let expected = native_rope_compat::native_rope_rotate(
                        &original[offset..offset + dim],
                        dim,
                        &cos,
                        &sin,
                    );
                    assert!(
                        values[offset..offset + dim]
                            .iter()
                            .zip(expected)
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    );
                }
            }
        }
        for chunk in [1, 3, 5] {
            let mut parts = Vec::new();
            for row in (0..queries).step_by(chunk) {
                parts.push(native_cpu_rotary(
                    &input.narrow(2, row, chunk.min(queries - row))?,
                    start + row,
                    10000.0,
                    4096,
                )?);
            }
            let actual = Tensor::cat(&parts, 2)?.flatten_all()?.to_vec1::<f32>()?;
            assert!(
                actual
                    .iter()
                    .zip(&values)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
        assert!(native_cpu_rotary(&input, 4095, 10000.0, 4096).is_err());
        assert!(native_cpu_rotary(&input, 0, f32::NAN, 4096).is_err());
        Ok(())
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn native_swiglu_adapter_keeps_fused_rows_and_strides() -> Result<()> {
        let width = 16384;
        let raw: Vec<f32> = (0..2 * 3 * width)
            .map(|i| ((i * 17 % 257) as f32 - 128.0) / 8.0)
            .collect();
        let base = Tensor::from_vec(raw.clone(), (2, 3, width), &Device::Cpu)?;
        let view = base.narrow(1, 1, 2)?;
        let actual = native_cpu_swiglu(&view)?.flatten_all()?.to_vec1::<f32>()?;
        let mut expected = Vec::new();
        for batch in 0..2 {
            for row in 1..3 {
                let start = (batch * 3 + row) * width;
                expected.extend(native_swiglu_compat::native_swiglu(
                    &raw[start..start + 8192],
                    &raw[start + 8192..start + width],
                ));
            }
        }
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert!(
            base.flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .zip(raw)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        let empty = Tensor::zeros((1, 0, width), DType::F32, &Device::Cpu)?;
        assert_eq!(native_cpu_swiglu(&empty)?.dims(), &[1, 0, 8192]);
        assert!(native_cpu_swiglu(&Tensor::zeros((1, 1, 12), DType::F32, &Device::Cpu)?).is_err());
        Ok(())
    }

    #[test]
    fn cpu_quant_query_matches_fixed_half_rounding_bits() -> Result<()> {
        // Fixed IEEE results include both half-tie parities, subnormals and
        // signed zero. The native F32 scale differs from a cast F64 scale.
        assert_eq!((1.0_f32 / 96.0_f32.sqrt()).to_bits(), 0x3dd105eb);
        let input_bits = [
            0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x349cc471, 0xb49cc471, 0x356b26aa,
            0xb56b26aa, 0x411cd80a, 0xc11cd80a, 0x411cff3b, 0xc11cff3b, 0x3a1cb0d8, 0xba1cb0d8,
        ];
        let expected = [
            0x00000000, 0x80000000, 0x3dd10000, 0xbdd10000, 0x00000000, 0x80000000, 0x34000000,
            0xb4000000, 0x3f800000, 0xbf800000, 0x3f804000, 0xbf804000, 0x387fc000, 0xb87fc000,
        ];
        let input = Tensor::from_vec(
            input_bits
                .iter()
                .copied()
                .map(f32::from_bits)
                .collect::<Vec<_>>(),
            (1, 1, 1, input_bits.len()),
            &Device::Cpu,
        )?;
        let actual = cpu_quant_query(&input, 96)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected
        );
        let unchanged = input.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(
            unchanged.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            input_bits
        );
        assert!(cpu_quant_query(&input, 0).is_err());
        Ok(())
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn query_test_projection() -> Result<Phi3QMatMul> {
        let weights = Tensor::ones((32, 32), DType::F32, &Device::Cpu)?;
        Ok(Phi3QMatMul::NativeCpu(QMatMul::QTensor(Arc::new(
            QTensor::quantize(&weights, GgmlDType::Q8_0)?,
        ))))
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cpu_quant_attention_declines_other_paths_without_changing_inputs() -> Result<()> {
        let projection = query_test_projection()?;
        let q = Tensor::ones((1, 1, 1, 96), DType::F32, &Device::Cpu)?;
        let kv = q.to_dtype(DType::F16)?;
        let call = |p: &Phi3QMatMul, q: &Tensor, k: &Tensor, v: &Tensor, d, native| {
            p.cpu_quant_attention(q, k, v, d, native, None)
        };
        assert!(call(&projection, &q, &kv, &kv, 96, true)?.is_some());
        assert!(call(&projection, &q, &kv, &kv, 96, false)?.is_none());
        assert!(call(&projection, &kv, &kv, &kv, 96, true)?.is_none());
        assert!(call(&projection, &q, &q, &kv, 96, true)?.is_none());
        assert!(call(&projection, &q, &kv, &q, 96, true)?.is_none());
        for width in [0, 95, 128] {
            assert!(call(&projection, &q, &kv, &kv, width, true)?.is_none());
        }
        let dense = Tensor::ones((32, 32), DType::F32, &Device::Cpu)?;
        for weight in [
            QMatMul::Tensor(dense.clone()),
            QMatMul::TensorF16(dense.to_dtype(DType::F16)?),
            QMatMul::QTensor(Arc::new(QTensor::quantize(&dense, GgmlDType::F32)?)),
        ] {
            assert!(call(&Phi3QMatMul::Candle(weight), &q, &kv, &kv, 96, true)?.is_none());
        }
        let unsupported = Tensor::ones((1, 1, 1, 32), DType::F32, &Device::Cpu)?;
        let half = unsupported.to_dtype(DType::F16)?;
        assert!(call(&projection, &unsupported, &half, &half, 32, true)?.is_none());
        let flat = q.flatten_all()?;
        assert!(call(&projection, &flat, &kv, &kv, 96, true)?.is_none());
        assert!(
            q.flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|x| x.to_bits() == 1_f32.to_bits())
        );
        Ok(())
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cpu_quant_attention_keeps_chunks_and_matches_own_effective_query_f64() -> Result<()> {
        let (heads, total, dim) = (2, 137, 96);
        let data = |salt: usize| {
            (0..heads * total * dim)
                .map(|i| (((i * 73 + salt) % 509) as f32 - 254.0) / 256.0)
                .collect::<Vec<_>>()
        };
        let q = Tensor::from_vec(data(17), (1, heads, total, dim), &Device::Cpu)?;
        let k = Tensor::from_vec(data(71), (1, heads, total, dim), &Device::Cpu)?
            .to_dtype(DType::F16)?;
        let v = Tensor::from_vec(data(93), (1, heads, total, dim), &Device::Cpu)?
            .to_dtype(DType::F16)?;
        let projection = query_test_projection()?;
        let run = |chunk: usize| -> Result<Tensor> {
            let mut parts = Vec::new();
            for start in (0..total).step_by(chunk) {
                let take = chunk.min(total - start);
                parts.push(
                    projection
                        .cpu_quant_attention(
                            &q.narrow(2, start, take)?,
                            &k.narrow(2, 0, start + take)?,
                            &v.narrow(2, 0, start + take)?,
                            dim,
                            true,
                            None,
                        )?
                        .expect("supported query/cache layout"),
                );
            }
            Ok(Tensor::cat(&parts, 2)?)
        };
        let baseline = run(total)?.flatten_all()?.to_vec1::<f32>()?;
        for chunk in [1, 5, 7, 32, 128] {
            let actual = run(chunk)?.flatten_all()?.to_vec1::<f32>()?;
            assert!(
                actual
                    .iter()
                    .zip(&baseline)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
        let effective = cpu_quant_query(&q, dim)?.flatten_all()?.to_vec1::<f32>()?;
        let keys = k.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let values = v.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        for head in 0..heads {
            for pos in [0, 31, 136] {
                let qi = (head * total + pos) * dim;
                let scores = (0..=pos)
                    .map(|key| {
                        let ki = (head * total + key) * dim;
                        (0..dim)
                            .map(|j| f64::from(effective[qi + j]) * f64::from(keys[ki + j]))
                            .sum::<f64>()
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let probabilities = scores.iter().map(|x| (x - max).exp()).collect::<Vec<_>>();
                let denominator = probabilities.iter().sum::<f64>();
                for d in 0..dim {
                    let expected = probabilities
                        .iter()
                        .enumerate()
                        .map(|(key, p)| {
                            p / denominator * f64::from(values[(head * total + key) * dim + d])
                        })
                        .sum::<f64>();
                    assert!((f64::from(baseline[qi + d]) - expected).abs() <= 2e-5);
                }
            }
        }
        Ok(())
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cpu_native_k_projection_offsets_chunks_and_f64() -> Result<()> {
        let (rows, width, columns) = (9, 512, 8);
        let weights = Tensor::from_vec(
            (0..columns * width)
                .map(|i| ((i * 29 % 509) as f32 - 254.0) / 1024.0)
                .collect::<Vec<_>>(),
            (columns, width),
            &Device::Cpu,
        )?;
        let base = Tensor::from_vec(
            (0..(rows + 2) * width)
                .map(|i| ((i * 73 % 503) as f32 - 251.0) / 512.0)
                .collect::<Vec<_>>(),
            (rows + 2, width),
            &Device::Cpu,
        )?;
        let view = base.narrow(0, 1, rows)?;
        assert!(view.layout().start_offset() > 0);
        let values = view.flatten_all()?.to_vec1::<f32>()?;
        for kind in [GgmlDType::Q4K, GgmlDType::Q5K, GgmlDType::Q6K] {
            let weight = Arc::new(QTensor::quantize(&weights, kind)?);
            let decoded = weight
                .dequantize(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let weight_bytes = weight.data()?.to_vec();
            let raw_kind = match kind {
                GgmlDType::Q4K => native_k_dot::Kind::Q4,
                GgmlDType::Q5K => native_k_dot::Kind::Q5,
                GgmlDType::Q6K => native_k_dot::Kind::Q6,
                _ => unreachable!(),
            };
            let mm = QMatMul::QTensor(weight);
            let full = cpu_native_k_projection(&mm, &view)?.context("K-quant admitted")?;
            let actual = full.flatten_all()?.to_vec1::<f32>()?;
            let encoded = native_q8k::quantize(&values).map_err(anyhow::Error::msg)?;
            let mut raw_expected = vec![0f32; rows * columns];
            for (row, input) in encoded.chunks_exact((width / 256) * 292).enumerate() {
                native_k_dot::project(
                    raw_kind,
                    &weight_bytes,
                    input,
                    width,
                    &mut raw_expected[row * columns..(row + 1) * columns],
                )
                .map_err(anyhow::Error::msg)?;
            }
            assert!(
                actual
                    .iter()
                    .zip(&raw_expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            let copied = Tensor::from_vec(values.clone(), (rows, width), &Device::Cpu)?;
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                cpu_native_k_projection(&mm, &copied)?
                    .unwrap()
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>()
            );
            for chunk in [1, 2, 4, 8] {
                let parts = (0..rows)
                    .step_by(chunk)
                    .map(|start| {
                        cpu_native_k_projection(
                            &mm,
                            &view.narrow(0, start, chunk.min(rows - start))?,
                        )?
                        .context("K chunk")
                    })
                    .collect::<Result<Vec<_>>>()?;
                let split = Tensor::cat(&parts, 0)?.flatten_all()?.to_vec1::<f32>()?;
                assert!(
                    actual
                        .iter()
                        .zip(split)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
            let packed = native_q8k::quantize(&values).map_err(anyhow::Error::msg)?;
            for row in 0..rows {
                for col in 0..columns {
                    let mut expected = 0f64;
                    let mut magnitude = 0f64;
                    for d in 0..width {
                        let block = (row * (width / 256) + d / 256) * 292;
                        let scale =
                            f32::from_le_bytes(packed[block..block + 4].try_into().unwrap());
                        let input = f64::from(scale) * f64::from(packed[block + 4 + d % 256] as i8);
                        let term = input * f64::from(decoded[col * width + d]);
                        expected += term;
                        magnitude += term.abs();
                    }
                    assert!(
                        (f64::from(actual[row * columns + col]) - expected).abs()
                            <= 2e-6 + 2e-6 * magnitude
                    );
                }
            }
        }
        Ok(())
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn request_k_rows_match_both_native_orders_at_arbitrary_chunks() -> Result<()> {
        let (rows, width, columns) = (137, 256, 8);
        let weights = Tensor::from_vec(
            (0..columns * width)
                .map(|i| ((i * 29 % 509) as f32 - 254.0) / 1024.0)
                .collect::<Vec<_>>(),
            (columns, width),
            &Device::Cpu,
        )?;
        let base = Tensor::from_vec(
            (0..(rows + 2) * width)
                .map(|i| ((i * 73 % 503) as f32 - 251.0) / 512.0)
                .collect::<Vec<_>>(),
            (1, rows + 2, width),
            &Device::Cpu,
        )?;
        let view = base.narrow(1, 1, rows)?;
        assert!(view.layout().start_offset() > 0);
        let bits = |x: Tensor| -> Result<Vec<u32>> {
            Ok(x.flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .map(|v| v.to_bits())
                .collect())
        };
        for kind in [GgmlDType::Q4K, GgmlDType::Q5K, GgmlDType::Q6K] {
            let mm = QMatMul::QTensor(Arc::new(QTensor::quantize(&weights, kind)?));
            for cutoff in [0, 4, 32, 128, 136, 140] {
                let whole = bits(
                    cpu_native_k_projection_rows(
                        &mm,
                        &view,
                        KRowPolicy::Request { start: 0, cutoff },
                    )?
                    .context("whole request")?,
                )?;
                let mut expected = Vec::new();
                for row in 0..rows {
                    expected.extend(bits(
                        cpu_native_k_projection_order(
                            &mm,
                            &view.narrow(1, row, 1)?,
                            row >= cutoff,
                        )?
                        .context("single row order")?,
                    )?);
                }
                assert_eq!(whole, expected);
                for size in [1, 5, 7, 32, 128] {
                    let mut split = Vec::new();
                    for start in (0..rows).step_by(size) {
                        let x = view.narrow(1, start, size.min(rows - start))?;
                        split.extend(bits(
                            cpu_native_k_projection_rows(
                                &mm,
                                &x,
                                KRowPolicy::Request { start, cutoff },
                            )?
                            .context("chunk")?,
                        )?);
                    }
                    assert_eq!(whole, split, "dtype={kind:?} cutoff={cutoff} chunk={size}");
                }
                let tail = view.narrow(1, rows - 1, 1)?;
                assert_eq!(
                    bits(
                        cpu_native_k_projection_rows(&mm, &tail, KRowPolicy::SelectedM1)?.unwrap()
                    )?,
                    bits(cpu_native_k_projection_order(&mm, &tail, true)?.unwrap())?
                );
            }
        }
        Ok(())
    }

    #[test]
    fn cpu_native_k_projection_declines_q8_dense_and_bad_layouts() -> Result<()> {
        let weights = Tensor::ones((8, 256), DType::F32, &Device::Cpu)?;
        let input = Tensor::ones((2, 256), DType::F32, &Device::Cpu)?;
        let dense = QMatMul::Tensor(weights.clone());
        let q8 = QMatMul::QTensor(Arc::new(QTensor::quantize(&weights, GgmlDType::Q8_0)?));
        assert!(cpu_native_k_projection(&dense, &input)?.is_none());
        assert!(cpu_native_k_projection(&q8, &input)?.is_none());
        let k = QMatMul::QTensor(Arc::new(QTensor::quantize(&weights, GgmlDType::Q4K)?));
        assert!(cpu_native_k_projection(&k, &input.to_dtype(DType::F16)?)?.is_none());
        assert!(cpu_native_k_projection(&k, &input.t()?)?.is_none());
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            assert!(
                cpu_native_k_projection(&k, &Tensor::ones((1, 512), DType::F32, &Device::Cpu)?)
                    .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn cpu_dense_projection_wrapper_keeps_fixed_rows_across_chunks() -> Result<()> {
        let weights = Tensor::from_vec(
            (0..96 * 128)
                .map(|i| (i % 101) as f32 / 101.0 - 0.5)
                .collect::<Vec<_>>(),
            (96, 128),
            &Device::Cpu,
        )?;
        let projection = Phi3QMatMul::Candle(QMatMul::Tensor(weights));
        let input = Tensor::from_vec(
            (0..37 * 128)
                .map(|i| (i % 37) as f32 / 37.0)
                .collect::<Vec<_>>(),
            (1, 37, 128),
            &Device::Cpu,
        )?;
        let baseline = projection
            .forward(&input)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for chunk in [1, 5, 7, 32, 37] {
            let parts = (0..37)
                .step_by(chunk)
                .map(|start| projection.forward(&input.narrow(1, start, chunk.min(37 - start))?))
                .collect::<Result<Vec<_>>>()?;
            let actual = Tensor::cat(&parts, 1)?.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(actual.len(), baseline.len());
            assert!(
                actual
                    .iter()
                    .zip(&baseline)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
        Ok(())
    }
    use crate::gguf_loader::{GgufMetadata, LayerRange, parse_gguf_metadata};
    use candle_core::quantized::{GgmlDType, gguf_file::Value};
    use std::collections::HashMap;

    fn metadata() -> Result<GgufMetadata> {
        let mut values = HashMap::from([(
            "general.architecture".to_string(),
            Value::String("phi3".to_string()),
        )]);
        for (key, value) in [
            ("block_count", 2),
            ("embedding_length", 8),
            ("attention.head_count", 2),
            ("attention.key_length", 4),
            ("context_length", 4),
            ("feed_forward_length", 8),
            ("vocab_size", 4),
        ] {
            values.insert(format!("phi3.{key}"), Value::U32(value));
        }
        parse_gguf_metadata(&values)
    }

    fn middle_stage(metadata: GgufMetadata) -> StageTensors {
        StageTensors {
            range: LayerRange::new(1, 2, 2),
            metadata,
            tensors: HashMap::new(),
            raw_tensors: HashMap::new(),
            raw_source: None,
            device: Device::Cpu,
        }
    }

    fn add_small_weights(stage: &mut StageTensors) -> Result<()> {
        for (name, shape) in [
            ("blk.1.attn_norm.weight", vec![8]),
            ("blk.1.attn_qkv.weight", vec![24, 8]),
            ("blk.1.attn_output.weight", vec![8, 8]),
            ("blk.1.ffn_norm.weight", vec![8]),
            ("blk.1.ffn_up.weight", vec![16, 8]),
            ("blk.1.ffn_down.weight", vec![8, 8]),
            ("output_norm.weight", vec![8]),
            ("output.weight", vec![4, 8]),
        ] {
            let tensor = Tensor::ones(shape.as_slice(), DType::F32, &Device::Cpu)?;
            stage.tensors.insert(
                name.to_string(),
                Arc::new(QTensor::quantize(&tensor, GgmlDType::F32)?),
            );
        }
        Ok(())
    }

    #[test]
    fn malformed_partial_rope_rejects_before_missing_block_or_table_allocation() -> Result<()> {
        for rope_dim in [1, 3, 6] {
            let mut md = metadata()?;
            md.rope_dimension_count = rope_dim;
            let error = Phi3Stage::from_stage_tensors(middle_stage(md), None)
                .err()
                .expect("invalid partial rotary geometry must fail")
                .to_string();
            assert!(error.contains("phi3.rope.dimension_count"), "{error}");
        }
        let mut stage = middle_stage(metadata()?);
        add_small_weights(&mut stage)?;
        stage.metadata.head_dim = 1 << 36;
        let error = Phi3Stage::from_stage_tensors(stage, None)
            .err()
            .expect("projection mismatch must fail before large rotary allocation")
            .to_string();
        assert!(error.contains("attn_qkv.weight shape"), "{error}");
        Ok(())
    }

    #[test]
    fn full_default_and_valid_partial_rope_stages_run_on_cpu() -> Result<()> {
        for rope_dim in [0, 2, 4] {
            let mut md = metadata()?;
            md.rope_dimension_count = rope_dim;
            let mut stage = middle_stage(md);
            add_small_weights(&mut stage)?;
            let model = Phi3Stage::from_stage_tensors(stage, None)?;
            let input = StageInput::activations(Tensor::ones((1, 8), DType::F32, &Device::Cpu)?);
            let output = model.forward(input, &mut [None], 1, 0)?;
            let StageOutput::Logits(logits) = output else {
                anyhow::bail!("Phi-3 tail did not produce logits");
            };
            assert_eq!(logits.dims(), [4]);
            assert!(
                logits
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|value| value.is_finite())
            );
        }
        Ok(())
    }

    #[test]
    fn rotary_table_helper_rejects_empty_odd_or_truncated_geometry() {
        for dim in [0, 1, 3] {
            assert!(precompute_rope_cos_sin(dim, 10_000., 4, &Device::Cpu).is_err());
        }
        assert!(precompute_rope_cos_sin(2, 10_000., 0, &Device::Cpu).is_err());
        if let Ok(context) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert!(precompute_rope_cos_sin(2, 10_000., context, &Device::Cpu).is_err());
        }
    }
}
