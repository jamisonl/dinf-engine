//! Fused CUDA norm, residual and scale steps of Gemma 4.
//!
//! Between attention and the feed-forward products a layer applies the
//! post-attention norm, the residual, and one to three norms of the result;
//! after them it applies up to three post norms, the residual, the layer
//! scale and the next layer's input norm. The portable path launches each
//! separately. Here each group is one kernel, one block per token.
//!
//! Gemma 4's logits move measurably when these values change in the last
//! bit, so both kernels perform the portable path's operations in its order
//! and produce its bits: Candle's RMSNorm sum of squares for rows of 1024
//! values or more (a fused multiply-add chain per thread of a 1024-thread
//! block, an xor-shuffle sum per warp, then one across warps),
//! `(scale * x) * weight`, and every addition and product rounded on its
//! own. Narrower rows keep the portable path.

use std::sync::OnceLock;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use candle_core::cuda_backend::cudarc::driver::{CudaView, LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use candle_core::{
    CpuStorage, CudaDevice, CudaStorage, DType, Device, Layout, Shape, Storage, Tensor,
};

const BLOCK_THREADS: usize = 1024;
const MIN_HIDDEN: usize = 1024;
const MAX_HIDDEN: usize = 8192;
/// Most routes per token the post-FFN kernel sums itself.
const MAX_ROUTES: usize = 64;
const CUDA_MODULE_NAME: &str = "dinf_gemma4_norm_fusion_v7";
const PLANES_KERNEL_NAME: &str = "dinf_gemma4_attention_planes_f32";
const TAIL_KERNEL_NAME: &str = "dinf_gemma4_post_ffn_f32";

const CUDA_SOURCE: &str = r#"
static __device__ __forceinline__ float dinf_warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, mask);
    }
    return value;
}

// Sum of `value` over a 1024-thread block in Candle's order, returned to
// every thread. `partials` holds one slot per warp and must not be reused
// by a later sum, because no barrier follows the read.
static __device__ __forceinline__ float dinf_block_sum(float value, float *partials) {
    const int lane = threadIdx.x & 31;
    value = dinf_warp_sum(value);
    if (lane == 0) {
        partials[threadIdx.x >> 5] = value;
    }
    __syncthreads();
    return dinf_warp_sum(partials[lane]);
}

// Two block sums at one barrier, each as `dinf_block_sum` would compute it.
static __device__ __forceinline__ void dinf_block_sum_pair(
    float &first,
    float &second,
    float *first_partials,
    float *second_partials) {
    const int lane = threadIdx.x & 31;
    first = dinf_warp_sum(first);
    second = dinf_warp_sum(second);
    if (lane == 0) {
        first_partials[threadIdx.x >> 5] = first;
        second_partials[threadIdx.x >> 5] = second;
    }
    __syncthreads();
    first = dinf_warp_sum(first_partials[lane]);
    second = dinf_warp_sum(second_partials[lane]);
}

// Candle's RMSNorm scale from a row's sum of squares.
static __device__ __forceinline__ float dinf_rms_scale(float sum, int hidden, float eps) {
    const float mean = sum / hidden;
    return rsqrtf(mean + eps);
}

// `(scale * x) * weight`, each product rounded.
static __device__ __forceinline__ float dinf_normed(float scale, float x, float weight) {
    return __fmul_rn(__fmul_rn(scale, x), weight);
}

// Columns one thread of a 1024-thread block owns in the widest row. A
// thread keeps the values it carries between steps in registers, so nothing
// it wrote is read back. Weights are loaded in the step that uses them:
// holding every input at once would exceed the registers a 1024-thread
// block may use.
#define DINF_SLOTS 8

// One block of 1024 threads per token. The output holds planes of
// `rows * hidden` values. Plane 0: `residual + rms_norm(x_attn) *
// post_weight`. Plane 1: its RMSNorm with `ffn_weight`. With `has_moe`,
// plane 2 is its RMSNorm with `pre_weight` and plane 3 its unweighted
// RMSNorm, both with `moe_eps`. With `has_quant` (one row only) the Q8_1
// forms of plane 1 and, with `has_moe`, plane 2 follow the planes.
extern "C" __global__ void dinf_gemma4_attention_planes_f32(
    const float *x_attn,
    const float *residual,
    const float *post_weight,
    const float *ffn_weight,
    const float *pre_weight,
    float *output,
    int rows,
    int hidden,
    int has_moe,
    int has_quant,
    float post_eps,
    float ffn_eps,
    float moe_eps) {
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    const long long row = (long long) blockIdx.x * hidden;
    const long long plane = (long long) rows * hidden;
    __shared__ float partials[2][32];
    x_attn += row;
    residual += row;
    float *attn_out = output + row;

    float values[DINF_SLOTS];
    float sum = 0.0f;
#pragma unroll
    for (int slot = 0; slot < DINF_SLOTS; ++slot) {
        const int column = tid + slot * threads;
        if (column < hidden) {
            values[slot] = x_attn[column];
            sum = __fmaf_rn(values[slot], values[slot], sum);
        }
    }
    const float post_scale =
        dinf_rms_scale(dinf_block_sum(sum, partials[0]), hidden, post_eps);

    sum = 0.0f;
#pragma unroll
    for (int slot = 0; slot < DINF_SLOTS; ++slot) {
        const int column = tid + slot * threads;
        if (column < hidden) {
            const float value = __fadd_rn(
                residual[column],
                dinf_normed(post_scale, values[slot], post_weight[column]));
            values[slot] = value;
            attn_out[column] = value;
            sum = __fmaf_rn(value, value, sum);
        }
    }
    sum = dinf_block_sum(sum, partials[1]);
    const float ffn_scale = dinf_rms_scale(sum, hidden, ffn_eps);
    const float moe_scale = dinf_rms_scale(sum, hidden, moe_eps);
    // With `has_quant` the loop runs over the quantizer's padded row, whose
    // length is a multiple of 32, so a warp's lanes stay together.
    const int end = has_quant ? dinf_q8_1_padded(hidden) : hidden;
    const int planes = has_moe ? 4 : 2;
    float *quantized = output + planes * plane;
    const int quantized_row = end / 32 * 9;
#pragma unroll
    for (int slot = 0; slot < DINF_SLOTS; ++slot) {
        const int column = tid + slot * threads;
        if (column < end) {
            float ffn_in = 0.0f;
            float expert_in = 0.0f;
            if (column < hidden) {
                ffn_in = dinf_normed(ffn_scale, values[slot], ffn_weight[column]);
                attn_out[plane + column] = ffn_in;
                if (has_moe) {
                    // The unweighted norm multiplies by a unit weight, which
                    // is exact.
                    const float normalized = __fmul_rn(moe_scale, values[slot]);
                    expert_in = __fmul_rn(normalized, pre_weight[column]);
                    attn_out[2 * plane + column] = expert_in;
                    attn_out[3 * plane + column] = normalized;
                }
            }
            if (has_quant) {
                dinf_q8_1_store(quantized, column, ffn_in);
                if (has_moe) {
                    dinf_q8_1_store(quantized + quantized_row, column, expert_in);
                }
            }
        }
    }
}

// One block of 1024 threads per token. `y` is `shared`, or with `has_moe`
// the sum of `shared` and `routed` each RMS-normalized with its own weight.
// With `top_k`, `routed` holds `top_k` rows per token and the routed value
// is their sum weighted by `route_weights`, accumulated in route order with
// one fused multiply-add per route. Plane 0 is
// `(attn_out + rms_norm(y) * post_weight) * scale`; with `has_next`, plane 1
// is its RMSNorm with `next_weight`, followed with `has_quant` (one row
// only) by plane 1's Q8_1 form.
extern "C" __global__ void dinf_gemma4_post_ffn_f32(
    const float *shared,
    const float *routed,
    const float *route_weights,
    const float *attn_out,
    const float *shared_weight,
    const float *routed_weight,
    const float *post_weight,
    const float *scale,
    const float *next_weight,
    float *output,
    int rows,
    int hidden,
    int has_moe,
    int top_k,
    int scale_len,
    int has_next,
    int has_quant,
    float moe_eps,
    float post_eps,
    float next_eps) {
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    const long long row = (long long) blockIdx.x * hidden;
    const long long plane = (long long) rows * hidden;
    __shared__ float partials[4][32];
    shared += row;
    routed += top_k > 0 ? row * top_k : row;
    route_weights += (long long) blockIdx.x * top_k;
    attn_out += row;
    float *block_out = output + row;

    // `values` is `shared`, then `y`, then the block output.
    float values[DINF_SLOTS];
    float routed_values[DINF_SLOTS];
    float shared_sum = 0.0f;
    float routed_sum = 0.0f;
#pragma unroll
    for (int slot = 0; slot < DINF_SLOTS; ++slot) {
        const int column = tid + slot * threads;
        if (column < hidden) {
            values[slot] = shared[column];
            if (has_moe) {
                float value;
                if (top_k > 0) {
                    value = 0.0f;
                    for (int route = 0; route < top_k; ++route) {
                        value = __fmaf_rn(
                            routed[(long long) route * hidden + column],
                            route_weights[route],
                            value);
                    }
                } else {
                    value = routed[column];
                }
                routed_values[slot] = value;
                shared_sum = __fmaf_rn(values[slot], values[slot], shared_sum);
                routed_sum = __fmaf_rn(value, value, routed_sum);
            }
        }
    }
    if (has_moe) {
        dinf_block_sum_pair(shared_sum, routed_sum, partials[0], partials[1]);
        const float shared_scale = dinf_rms_scale(shared_sum, hidden, moe_eps);
        const float routed_scale = dinf_rms_scale(routed_sum, hidden, moe_eps);
#pragma unroll
        for (int slot = 0; slot < DINF_SLOTS; ++slot) {
            const int column = tid + slot * threads;
            if (column < hidden) {
                values[slot] = __fadd_rn(
                    dinf_normed(shared_scale, values[slot], shared_weight[column]),
                    dinf_normed(routed_scale, routed_values[slot], routed_weight[column]));
            }
        }
    }

    float sum = 0.0f;
#pragma unroll
    for (int slot = 0; slot < DINF_SLOTS; ++slot) {
        if (tid + slot * threads < hidden) {
            sum = __fmaf_rn(values[slot], values[slot], sum);
        }
    }
    const float post_scale =
        dinf_rms_scale(dinf_block_sum(sum, partials[2]), hidden, post_eps);

    sum = 0.0f;
#pragma unroll
    for (int slot = 0; slot < DINF_SLOTS; ++slot) {
        const int column = tid + slot * threads;
        if (column < hidden) {
            float value = __fadd_rn(
                attn_out[column],
                dinf_normed(post_scale, values[slot], post_weight[column]));
            if (scale_len == 1) {
                value = __fmul_rn(value, scale[0]);
            } else if (scale_len > 1) {
                value = __fmul_rn(value, scale[column]);
            }
            values[slot] = value;
            block_out[column] = value;
            sum = __fmaf_rn(value, value, sum);
        }
    }
    if (has_next) {
        const float next_scale =
            dinf_rms_scale(dinf_block_sum(sum, partials[3]), hidden, next_eps);
        // As in the planes kernel, `has_quant` extends the loop to the
        // quantizer's padded row.
        const int end = has_quant ? dinf_q8_1_padded(hidden) : hidden;
#pragma unroll
        for (int slot = 0; slot < DINF_SLOTS; ++slot) {
            const int column = tid + slot * threads;
            if (column < end) {
                float value = 0.0f;
                if (column < hidden) {
                    value = dinf_normed(next_scale, values[slot], next_weight[column]);
                    block_out[plane + column] = value;
                }
                if (has_quant) {
                    dinf_q8_1_store(output + 2 * plane, column, value);
                }
            }
        }
    }
}
"#;

static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
static FAST_PATH_LOGGED: OnceLock<()> = OnceLock::new();

fn ptx() -> candle_core::Result<&'static str> {
    match PTX.get_or_init(|| {
        compile_ptx_with_opts(
            format!("{}{}", super::q8_1_epilogue_cuda::SOURCE, CUDA_SOURCE),
            CompileOptions {
                name: Some("gemma4_norm_fusion.cu".to_string()),
                use_fast_math: Some(false),
                ..Default::default()
            },
        )
        .map(|ptx| ptx.to_src())
        .map_err(|error| error.to_string())
    }) {
        Ok(ptx) => Ok(ptx.as_str()),
        Err(error) => candle_core::bail!("failed to compile CUDA Gemma 4 norm fusion: {error}"),
    }
}

pub(super) fn precompile(device: &Device) -> anyhow::Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    let device = device.as_cuda_device()?;
    device.get_or_load_custom_func(PLANES_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
    device.get_or_load_custom_func(TAIL_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
    Ok(())
}

/// A contiguous F32 view of exactly `elements` values on `device`.
fn storage_view<'a>(
    name: &str,
    storage: &'a CudaStorage,
    layout: &Layout,
    elements: usize,
    device: &CudaDevice,
) -> candle_core::Result<CudaView<'a, f32>> {
    if storage.dtype() != DType::F32
        || storage.device().id() != device.id()
        || !layout.is_contiguous()
        || layout.shape().elem_count() != elements
    {
        candle_core::bail!("CUDA Gemma 4 norm fusion {name} must be {elements} contiguous f32")
    }
    let slice = storage.as_cuda_slice::<f32>()?;
    let start = layout.start_offset();
    if start + elements > slice.len() {
        candle_core::bail!("CUDA Gemma 4 norm fusion {name} view exceeds its storage")
    }
    Ok(slice.slice(start..start + elements))
}

fn captured_view<'a>(
    name: &str,
    storage: &'a Storage,
    layout: &Layout,
    elements: usize,
    device: &CudaDevice,
) -> candle_core::Result<CudaView<'a, f32>> {
    let Storage::Cuda(storage) = storage else {
        candle_core::bail!("CUDA Gemma 4 norm fusion {name} is not on CUDA")
    };
    storage_view(name, storage, layout, elements, device)
}

type StorageGuard<'a> = (candle_core::StorageRef<'a>, &'a Layout);

fn optional_guard(tensor: &Option<Tensor>) -> Option<StorageGuard<'_>> {
    tensor.as_ref().map(Tensor::storage_and_layout)
}

fn optional_view<'a>(
    name: &str,
    guard: &'a Option<StorageGuard<'_>>,
    elements: usize,
    device: &CudaDevice,
) -> candle_core::Result<Option<CudaView<'a, f32>>> {
    match guard {
        Some((storage, layout)) => captured_view(name, storage, layout, elements, device).map(Some),
        None => Ok(None),
    }
}

/// F32 elements after the planes that hold the Q8_1 forms of `forms` planes
/// of one token. Longer inputs carry none.
fn quantized_tail(rows: usize, hidden: usize, forms: usize) -> usize {
    if rows == 1 {
        forms * super::q8_1_epilogue_cuda::tail_elements(hidden)
    } else {
        0
    }
}

/// Token rows and row width of `[.., hidden]` rows this module fuses.
fn row_geometry(rows: &Tensor) -> Option<(usize, usize)> {
    let hidden = *rows.dims().last()?;
    if !(MIN_HIDDEN..=MAX_HIDDEN).contains(&hidden) {
        return None;
    }
    let count = rows.elem_count() / hidden;
    (count > 0 && count <= i32::MAX as usize / hidden).then_some((count, hidden))
}

/// True when every tensor is contiguous F32 on `device` with `elements`
/// values.
fn rows_supported(device: &Device, elements: usize, tensors: &[&Tensor]) -> bool {
    tensors.iter().all(|tensor| {
        tensor.dtype() == DType::F32
            && tensor.is_contiguous()
            && tensor.elem_count() == elements
            && tensor.device().same_device(device)
    })
}

/// Routed-expert inputs for [`try_attention_planes`].
pub(super) struct MoePlanes<'a> {
    pub(super) pre_weight: &'a Tensor,
    pub(super) eps: f32,
}

struct AttentionPlanes {
    post_weight: Tensor,
    ffn_weight: Tensor,
    pre_weight: Option<Tensor>,
    rows: usize,
    hidden: usize,
    eps: [f32; 3],
}

impl candle_core::CustomOp2 for AttentionPlanes {
    fn name(&self) -> &'static str {
        "cuda-gemma4-attention-planes"
    }

    fn cpu_fwd(
        &self,
        _x: &CpuStorage,
        _x_layout: &Layout,
        _residual: &CpuStorage,
        _residual_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda-gemma4-attention-planes is only dispatched on CUDA")
    }

    fn cuda_fwd(
        &self,
        x_storage: &CudaStorage,
        x_layout: &Layout,
        residual_storage: &CudaStorage,
        residual_layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let (rows, hidden) = (self.rows, self.hidden);
        let device = x_storage.device();
        let x = storage_view(
            "attention output",
            x_storage,
            x_layout,
            rows * hidden,
            device,
        )?;
        let residual = storage_view(
            "residual",
            residual_storage,
            residual_layout,
            rows * hidden,
            device,
        )?;
        let (post_guard, post_layout) = self.post_weight.storage_and_layout();
        let post = captured_view(
            "post-attention weight",
            &post_guard,
            post_layout,
            hidden,
            device,
        )?;
        let (ffn_guard, ffn_layout) = self.ffn_weight.storage_and_layout();
        let ffn = captured_view("FFN norm weight", &ffn_guard, ffn_layout, hidden, device)?;
        let pre_guard = self
            .pre_weight
            .as_ref()
            .map(|weight| weight.storage_and_layout());
        let pre = match &pre_guard {
            Some((storage, layout)) => Some(captured_view(
                "expert norm weight",
                storage,
                layout,
                hidden,
                device,
            )?),
            None => None,
        };
        let has_moe = i32::from(pre.is_some());
        let planes = if pre.is_some() { 4 } else { 2 };
        // Without experts the kernel never reads this pointer.
        let pre = pre.unwrap_or_else(|| ffn.slice(..));
        let has_quant = i32::from(rows == 1);
        let tail = quantized_tail(rows, hidden, if pre_guard.is_some() { 2 } else { 1 });
        // SAFETY: Each token's block writes its row of every plane, and one
        // token's block writes the whole Q8_1 tail.
        let mut output = unsafe { device.alloc::<f32>(planes * rows * hidden + tail)? };
        let function =
            device.get_or_load_custom_func(PLANES_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
        let (rows_arg, hidden_arg) = (rows as i32, hidden as i32);
        let mut builder = function.builder();
        builder
            .arg(&x)
            .arg(&residual)
            .arg(&post)
            .arg(&ffn)
            .arg(&pre)
            .arg(&mut output)
            .arg(&rows_arg)
            .arg(&hidden_arg)
            .arg(&has_moe)
            .arg(&has_quant)
            .arg(&self.eps[0])
            .arg(&self.eps[1])
            .arg(&self.eps[2]);
        // SAFETY: Row views hold `rows * hidden` and weight views `hidden`
        // checked f32 values.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (BLOCK_THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from((planes, rows * hidden)),
        ))
    }
}

/// Post-attention planes of each token: `attn_out = residual +
/// post_norm(x_attn)` and `ffn_norm(attn_out)`, plus with `moe` the expert
/// input `pre_norm(attn_out)` and the unweighted router input. Each result
/// has `residual`'s shape, whose last dimension is the row width. Returns
/// `None` outside the kernel's geometry.
#[allow(clippy::type_complexity)]
pub(super) fn try_attention_planes(
    x_attn: &Tensor,
    residual: &Tensor,
    post: (&Tensor, f32),
    ffn: (&Tensor, f32),
    moe: Option<MoePlanes<'_>>,
) -> anyhow::Result<Option<(Tensor, Tensor, Option<(Tensor, Tensor)>)>> {
    let Some((rows, hidden)) = row_geometry(residual) else {
        return Ok(None);
    };
    let mut weights = vec![post.0, ffn.0];
    if let Some(moe) = &moe {
        weights.push(moe.pre_weight);
    }
    if !residual.device().is_cuda()
        || !rows_supported(residual.device(), rows * hidden, &[x_attn, residual])
        || !rows_supported(residual.device(), hidden, &weights)
    {
        return Ok(None);
    }
    let shape = residual.shape().clone();
    let packed = x_attn.apply_op2_no_bwd(
        residual,
        &AttentionPlanes {
            post_weight: post.0.clone(),
            ffn_weight: ffn.0.clone(),
            pre_weight: moe.as_ref().map(|moe| moe.pre_weight.clone()),
            rows,
            hidden,
            eps: [post.1, ffn.1, moe.as_ref().map_or(0.0, |moe| moe.eps)],
        },
    )?;
    if rows == 1 {
        // The kernel wrote the Q8_1 forms of the FFN and expert inputs
        // after the planes.
        let planes = if moe.is_some() { 4 } else { 2 };
        let form = super::q8_1_epilogue_cuda::tail_elements(hidden);
        super::q8_1_epilogue_cuda::publish(&packed, hidden, hidden, planes * hidden)?;
        if moe.is_some() {
            super::q8_1_epilogue_cuda::publish(
                &packed,
                2 * hidden,
                hidden,
                planes * hidden + form,
            )?;
        }
    }
    let plane = |index: usize| packed.narrow(0, index, 1)?.reshape(shape.clone());
    let experts = if moe.is_some() {
        Some((plane(2)?, plane(3)?))
    } else {
        None
    };
    FAST_PATH_LOGGED.get_or_init(|| {
        tracing::info!("CUDA fused Gemma 4 norm planes enabled");
    });
    Ok(Some((plane(0)?, plane(1)?, experts)))
}

/// Routed-expert inputs for [`try_post_ffn`].
pub(super) struct MoeTail<'a> {
    /// The routed branch, `[.., hidden]` like the other rows; or with
    /// `route_weights`, each token's unsummed `[top_k, hidden]` routes.
    pub(super) routed: &'a Tensor,
    /// F32 `[tokens, top_k]` weights of unsummed routes.
    pub(super) route_weights: Option<&'a Tensor>,
    pub(super) shared_weight: &'a Tensor,
    pub(super) routed_weight: &'a Tensor,
    pub(super) eps: f32,
}

struct PostFfn {
    attn_out: Tensor,
    routed: Option<Tensor>,
    route_weights: Option<Tensor>,
    /// Routes per token when `routed` is unsummed, otherwise zero.
    top_k: usize,
    shared_weight: Option<Tensor>,
    routed_weight: Option<Tensor>,
    post_weight: Tensor,
    scale: Option<Tensor>,
    next_weight: Option<Tensor>,
    rows: usize,
    hidden: usize,
    eps: [f32; 3],
}

impl candle_core::CustomOp1 for PostFfn {
    fn name(&self) -> &'static str {
        "cuda-gemma4-post-ffn"
    }

    fn cpu_fwd(
        &self,
        _shared: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda-gemma4-post-ffn is only dispatched on CUDA")
    }

    fn cuda_fwd(
        &self,
        shared_storage: &CudaStorage,
        shared_layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let (rows, hidden) = (self.rows, self.hidden);
        let device = shared_storage.device();
        let shared = storage_view(
            "FFN output",
            shared_storage,
            shared_layout,
            rows * hidden,
            device,
        )?;
        let (attn_guard, attn_layout) = self.attn_out.storage_and_layout();
        let attn_out = captured_view(
            "attention residual",
            &attn_guard,
            attn_layout,
            rows * hidden,
            device,
        )?;
        let (post_guard, post_layout) = self.post_weight.storage_and_layout();
        let post = captured_view("post-FFN weight", &post_guard, post_layout, hidden, device)?;

        let routed_guard = optional_guard(&self.routed);
        let shared_weight_guard = optional_guard(&self.shared_weight);
        let routed_weight_guard = optional_guard(&self.routed_weight);
        let scale_guard = optional_guard(&self.scale);
        let next_guard = optional_guard(&self.next_weight);
        let view =
            |name: &str, guard, elements: usize| optional_view(name, guard, elements, device);
        let route_weights_guard = optional_guard(&self.route_weights);
        let routed = view(
            "routed output",
            &routed_guard,
            rows * self.top_k.max(1) * hidden,
        )?;
        let route_weights = view("route weights", &route_weights_guard, rows * self.top_k)?;
        let shared_weight = view("shared post weight", &shared_weight_guard, hidden)?;
        let routed_weight = view("routed post weight", &routed_weight_guard, hidden)?;
        let scale_len = self.scale.as_ref().map_or(0, Tensor::elem_count);
        let scale = view("layer scale", &scale_guard, scale_len)?;
        let next = view("next norm weight", &next_guard, hidden)?;
        let has_moe = i32::from(routed.is_some());
        let has_next = i32::from(next.is_some());
        let planes = if next.is_some() { 2 } else { 1 };
        // Absent optional inputs are never read; pass a valid pointer.
        let placeholder = || post.slice(..);
        let routed = routed.unwrap_or_else(|| shared.slice(..));
        let route_weights = route_weights.unwrap_or_else(placeholder);
        let shared_weight = shared_weight.unwrap_or_else(placeholder);
        let routed_weight = routed_weight.unwrap_or_else(placeholder);
        let scale = scale.unwrap_or_else(placeholder);
        let next = next.unwrap_or_else(placeholder);

        let has_quant = i32::from(rows == 1 && has_next == 1);
        let tail = quantized_tail(rows, hidden, has_next as usize);
        // SAFETY: Each token's block writes its row of every plane, and one
        // token's block writes the whole Q8_1 tail.
        let mut output = unsafe { device.alloc::<f32>(planes * rows * hidden + tail)? };
        let function =
            device.get_or_load_custom_func(TAIL_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
        let (rows_arg, hidden_arg, scale_len_arg) = (rows as i32, hidden as i32, scale_len as i32);
        let top_k_arg = self.top_k as i32;
        let mut builder = function.builder();
        builder
            .arg(&shared)
            .arg(&routed)
            .arg(&route_weights)
            .arg(&attn_out)
            .arg(&shared_weight)
            .arg(&routed_weight)
            .arg(&post)
            .arg(&scale)
            .arg(&next)
            .arg(&mut output)
            .arg(&rows_arg)
            .arg(&hidden_arg)
            .arg(&has_moe)
            .arg(&top_k_arg)
            .arg(&scale_len_arg)
            .arg(&has_next)
            .arg(&has_quant)
            .arg(&self.eps[0])
            .arg(&self.eps[1])
            .arg(&self.eps[2]);
        // SAFETY: Every view was checked to hold the elements its flag
        // lets the kernel read; `hidden` fits the kernel's shared row.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (BLOCK_THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from((planes, rows * hidden)),
        ))
    }
}

/// The post-FFN step of each token: `(attn_out + post_norm(y)) * scale`,
/// where `y` is `ffn_out` or, with `moe`, the sum of the shared and routed
/// branches after their own norms. With `next`, also returns that norm of
/// the result for the next layer. Returns `None` outside the kernel's
/// geometry.
pub(super) fn try_post_ffn(
    ffn_out: &Tensor,
    attn_out: &Tensor,
    post: (&Tensor, f32),
    moe: Option<MoeTail<'_>>,
    scale: Option<&Tensor>,
    next: Option<(&Tensor, f32)>,
) -> anyhow::Result<Option<(Tensor, Option<Tensor>)>> {
    let Some((rows, hidden)) = row_geometry(attn_out) else {
        return Ok(None);
    };
    let mut row_tensors = vec![ffn_out, attn_out];
    let mut weights = vec![post.0];
    let device = attn_out.device();
    let mut top_k = 0;
    if let Some(moe) = &moe {
        match moe.route_weights {
            // Unsummed routes: `top_k` rows and weights per token.
            Some(route_weights) => {
                top_k = route_weights.elem_count() / rows;
                if top_k == 0
                    || top_k > MAX_ROUTES
                    || !rows_supported(device, rows * top_k, &[route_weights])
                    || !rows_supported(device, rows * top_k * hidden, &[moe.routed])
                {
                    return Ok(None);
                }
            }
            None => row_tensors.push(moe.routed),
        }
        weights.extend([moe.shared_weight, moe.routed_weight]);
    }
    if let Some((weight, _)) = next {
        weights.push(weight);
    }
    let scale_supported = scale.is_none_or(|scale| {
        scale.dtype() == DType::F32
            && scale.is_contiguous()
            && scale.device().same_device(device)
            && (scale.elem_count() == 1 || scale.elem_count() == hidden)
    });
    if !device.is_cuda()
        || !scale_supported
        || !rows_supported(device, rows * hidden, &row_tensors)
        || !rows_supported(device, hidden, &weights)
    {
        return Ok(None);
    }
    let shape = attn_out.shape().clone();
    let packed = ffn_out.apply_op1_no_bwd(&PostFfn {
        attn_out: attn_out.clone(),
        routed: moe.as_ref().map(|moe| moe.routed.clone()),
        route_weights: moe.as_ref().and_then(|moe| moe.route_weights.cloned()),
        top_k,
        shared_weight: moe.as_ref().map(|moe| moe.shared_weight.clone()),
        routed_weight: moe.as_ref().map(|moe| moe.routed_weight.clone()),
        post_weight: post.0.clone(),
        scale: scale.cloned(),
        next_weight: next.map(|(weight, _)| weight.clone()),
        rows,
        hidden,
        eps: [
            moe.as_ref().map_or(0.0, |moe| moe.eps),
            post.1,
            next.map_or(0.0, |(_, eps)| eps),
        ],
    })?;
    if rows == 1 && next.is_some() {
        // The kernel wrote the next layer's input in Q8_1 form after the
        // planes.
        super::q8_1_epilogue_cuda::publish(&packed, hidden, hidden, 2 * hidden)?;
    }
    let plane = |index: usize| packed.narrow(0, index, 1)?.reshape(shape.clone());
    let next_normed = if next.is_some() {
        Some(plane(1)?)
    } else {
        None
    };
    Ok(Some((plane(0)?, next_normed)))
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::{MoePlanes, MoeTail, try_attention_planes, try_post_ffn};

    fn values(elements: usize, multiplier: usize) -> Vec<f32> {
        (0..elements)
            .map(|index| ((index * multiplier % 1009) as f32 - 504.0) / 211.0)
            .collect()
    }

    fn rms_norm(x: &[f32], weight: Option<&[f32]>, eps: f32) -> Vec<f32> {
        let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let scale = 1.0 / (mean + eps).sqrt();
        x.iter()
            .enumerate()
            .map(|(index, value)| value * scale * weight.map_or(1.0, |weight| weight[index]))
            .collect()
    }

    fn close(actual: &Tensor, expected: &[f32]) -> anyhow::Result<()> {
        let actual = actual
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 2e-5 + expected.abs() * 2e-5,
                "mismatch at {index}: {actual} != {expected}"
            );
        }
        Ok(())
    }

    fn bits(tensor: &Tensor) -> anyhow::Result<Vec<u32>> {
        Ok(tensor
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }

    /// Both kernels produce the bits of the portable CUDA path, which is
    /// Candle's RMSNorm kernel and separate elementwise launches, for one
    /// token and for a prompt, with and without routed experts.
    #[test]
    fn fused_norm_steps_match_the_portable_path_bit_for_bit() -> anyhow::Result<()> {
        use candle_nn::ops::rms_norm;

        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        for (rows, hidden) in [(1usize, 2816usize), (5, 2816), (3, 1024), (2, 5376)] {
            let rows_tensor = |multiplier: usize| {
                Tensor::from_vec(
                    values(rows * hidden, multiplier),
                    (1, rows, hidden),
                    &device,
                )
            };
            let weight =
                |multiplier: usize| Tensor::from_vec(values(hidden, multiplier), hidden, &device);
            let unit = Tensor::ones(hidden, candle_core::DType::F32, &device)?;
            let (x_attn, residual) = (rows_tensor(17)?, rows_tensor(23)?);
            let (post_w, ffn_w, pre_w) = (weight(29)?, weight(31)?, weight(37)?);
            let (post_eps, ffn_eps, moe_eps) = (1e-6f32, 2e-6, 3e-6);
            let label = format!("rows {rows} hidden {hidden}");

            let attn_expected = (&residual + rms_norm(&x_attn, &post_w, post_eps)?)?;
            for with_moe in [false, true] {
                let (attn_out, ffn_in, experts) = try_attention_planes(
                    &x_attn,
                    &residual,
                    (&post_w, post_eps),
                    (&ffn_w, ffn_eps),
                    with_moe.then_some(MoePlanes {
                        pre_weight: &pre_w,
                        eps: moe_eps,
                    }),
                )?
                .expect("supported geometry");
                assert_eq!(attn_out.dims(), [1, rows, hidden]);
                assert_eq!(bits(&attn_out)?, bits(&attn_expected)?, "{label} residual");
                assert_eq!(
                    bits(&ffn_in)?,
                    bits(&rms_norm(&attn_expected, &ffn_w, ffn_eps)?)?,
                    "{label} FFN input"
                );
                if let Some((expert_in, router_in)) = experts {
                    assert_eq!(
                        bits(&expert_in)?,
                        bits(&rms_norm(&attn_expected, &pre_w, moe_eps)?)?,
                        "{label} expert input"
                    );
                    assert_eq!(
                        bits(&router_in)?,
                        bits(&rms_norm(&attn_expected, &unit, moe_eps)?)?,
                        "{label} router input"
                    );
                }
            }

            let (shared, routed, attn_out) = (rows_tensor(41)?, rows_tensor(43)?, rows_tensor(47)?);
            let (shared_w, routed_w, tail_w, next_w) =
                (weight(53)?, weight(59)?, weight(61)?, weight(67)?);
            let (tail_eps, next_eps) = (4e-6f32, 5e-6f32);
            let scales = [
                None,
                Some(Tensor::from_vec(vec![0.75f32], 1, &device)?),
                Some(weight(71)?),
            ];
            for with_moe in [false, true] {
                for scale in &scales {
                    let combined = if with_moe {
                        (rms_norm(&shared, &shared_w, moe_eps)?
                            + rms_norm(&routed, &routed_w, moe_eps)?)?
                    } else {
                        shared.clone()
                    };
                    let mut expected = (&attn_out + rms_norm(&combined, &tail_w, tail_eps)?)?;
                    if let Some(scale) = scale {
                        expected = expected.broadcast_mul(scale)?;
                    }
                    let (block_out, next) = try_post_ffn(
                        &shared,
                        &attn_out,
                        (&tail_w, tail_eps),
                        with_moe.then_some(MoeTail {
                            routed: &routed,
                            route_weights: None,
                            shared_weight: &shared_w,
                            routed_weight: &routed_w,
                            eps: moe_eps,
                        }),
                        scale.as_ref(),
                        Some((&next_w, next_eps)),
                    )?
                    .expect("supported geometry");
                    assert_eq!(bits(&block_out)?, bits(&expected)?, "{label} block output");
                    assert_eq!(
                        bits(&next.expect("next norm requested"))?,
                        bits(&rms_norm(&expected, &next_w, next_eps)?)?,
                        "{label} next norm"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn narrow_rows_keep_the_portable_path() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        let hidden = 512usize;
        let row = Tensor::from_vec(values(hidden, 17), (1, 1, hidden), &device)?;
        let weight = Tensor::from_vec(values(hidden, 29), hidden, &device)?;
        assert!(
            try_attention_planes(&row, &row, (&weight, 1e-6), (&weight, 1e-6), None)?.is_none()
        );
        assert!(try_post_ffn(&row, &row, (&weight, 1e-6), None, None, None)?.is_none());
        Ok(())
    }

    #[test]
    fn fused_norm_steps_match_reference() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        // Wider than one pass of the 1024-thread block.
        let hidden = 2816usize;
        let tensor =
            |data: &[f32], shape: &[usize]| Tensor::from_vec(data.to_vec(), shape, &device);
        let row = |multiplier| values(hidden, multiplier);
        let (x_attn, residual) = (row(17), row(23));
        let (post_w, ffn_w, pre_w) = (row(29), row(31), row(37));
        let (post_eps, ffn_eps, moe_eps) = (1e-6f32, 2e-6, 3e-6);

        let attn_expected = residual
            .iter()
            .zip(rms_norm(&x_attn, Some(&post_w), post_eps))
            .map(|(residual, normed)| residual + normed)
            .collect::<Vec<_>>();
        for with_moe in [false, true] {
            let pre = tensor(&pre_w, &[hidden])?;
            let (attn_out, ffn_in, experts) = try_attention_planes(
                &tensor(&x_attn, &[1, 1, hidden])?,
                &tensor(&residual, &[1, 1, hidden])?,
                (&tensor(&post_w, &[hidden])?, post_eps),
                (&tensor(&ffn_w, &[hidden])?, ffn_eps),
                with_moe.then_some(MoePlanes {
                    pre_weight: &pre,
                    eps: moe_eps,
                }),
            )?
            .expect("supported geometry");
            assert_eq!(attn_out.dims(), [1, 1, hidden]);
            close(&attn_out, &attn_expected)?;
            close(&ffn_in, &rms_norm(&attn_expected, Some(&ffn_w), ffn_eps))?;
            assert_eq!(experts.is_some(), with_moe);
            if let Some((expert_in, router_in)) = experts {
                close(&expert_in, &rms_norm(&attn_expected, Some(&pre_w), moe_eps))?;
                close(&router_in, &rms_norm(&attn_expected, None, moe_eps))?;
            }
        }

        let (shared, routed, attn_out) = (row(41), row(43), row(47));
        let (shared_w, routed_w, tail_w, next_w) = (row(53), row(59), row(61), row(67));
        let (tail_eps, next_eps, scale) = (4e-6f32, 5e-6f32, 0.75f32);
        for with_moe in [false, true] {
            let combined = if with_moe {
                rms_norm(&shared, Some(&shared_w), moe_eps)
                    .iter()
                    .zip(rms_norm(&routed, Some(&routed_w), moe_eps))
                    .map(|(shared, routed)| shared + routed)
                    .collect::<Vec<_>>()
            } else {
                shared.clone()
            };
            let expected = attn_out
                .iter()
                .zip(rms_norm(&combined, Some(&tail_w), tail_eps))
                .map(|(attn, normed)| (attn + normed) * scale)
                .collect::<Vec<_>>();
            let routed_tensor = tensor(&routed, &[1, 1, hidden])?;
            let (shared_weight, routed_weight) =
                (tensor(&shared_w, &[hidden])?, tensor(&routed_w, &[hidden])?);
            let (block_out, next) = try_post_ffn(
                &tensor(&shared, &[1, 1, hidden])?,
                &tensor(&attn_out, &[1, 1, hidden])?,
                (&tensor(&tail_w, &[hidden])?, tail_eps),
                with_moe.then_some(MoeTail {
                    routed: &routed_tensor,
                    route_weights: None,
                    shared_weight: &shared_weight,
                    routed_weight: &routed_weight,
                    eps: moe_eps,
                }),
                Some(&tensor(&[scale], &[1])?),
                Some((&tensor(&next_w, &[hidden])?, next_eps)),
            )?
            .expect("supported geometry");
            close(&block_out, &expected)?;
            close(
                &next.expect("next norm requested"),
                &rms_norm(&expected, Some(&next_w), next_eps),
            )?;
        }
        Ok(())
    }

    /// With unsummed routes the post-FFN kernel produces the bits it does
    /// from the route-sum kernel's output, for one token and for a prompt.
    #[test]
    fn post_ffn_sums_routes_like_the_route_sum_kernel() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        // The widest row uses every register slot of the kernel.
        for (hidden, top_k, tokens) in [(2816usize, 8usize, 1usize), (2816, 8, 3), (8192, 5, 1)] {
            let rows = |multiplier: usize| {
                Tensor::from_vec(
                    values(tokens * hidden, multiplier),
                    (1, tokens, hidden),
                    &device,
                )
            };
            let weight =
                |multiplier: usize| Tensor::from_vec(values(hidden, multiplier), hidden, &device);
            let (shared, attn_out) = (rows(3)?, rows(5)?);
            let (shared_w, routed_w, tail_w, next_w) =
                (weight(7)?, weight(11)?, weight(13)?, weight(17)?);
            let routes = Tensor::from_vec(
                values(tokens * top_k * hidden, 19),
                (tokens, top_k, hidden),
                &device,
            )?;
            let route_weights =
                Tensor::from_vec(values(tokens * top_k, 23), (tokens, top_k), &device)?;
            let summed =
                crate::architectures::indexed_moe::weighted_route_sum(&routes, &route_weights)?
                    .reshape((1, tokens, hidden))?;
            let run = |routed: &Tensor, route_weights: Option<&Tensor>| {
                try_post_ffn(
                    &shared,
                    &attn_out,
                    (&tail_w, 1e-6),
                    Some(MoeTail {
                        routed,
                        route_weights,
                        shared_weight: &shared_w,
                        routed_weight: &routed_w,
                        eps: 1e-6,
                    }),
                    None,
                    Some((&next_w, 1e-6)),
                )
            };
            let (expected_out, expected_next) = run(&summed, None)?.expect("supported geometry");
            let (actual_out, actual_next) =
                run(&routes, Some(&route_weights))?.expect("supported geometry");
            assert_eq!(bits(&actual_out)?, bits(&expected_out)?, "{tokens} tokens");
            assert_eq!(
                bits(&actual_next.expect("requested"))?,
                bits(&expected_next.expect("requested"))?,
                "{tokens} tokens"
            );
        }
        Ok(())
    }
}
