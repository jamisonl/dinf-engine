//! Fused CUDA glue for one Gemma 4 decode token's routed experts.
//!
//! The expert products themselves use the existing indexed kernels. Around
//! them the portable path spends about fifteen launches on routing (softmax,
//! sort, gather, normalize, scale) and on the GELU gate (two slices, GELU,
//! multiply). Each becomes one launch here. The router projection of one
//! token is one launch as well, in place of a BLAS call that brings its own
//! workspace allocation and copy.

use std::sync::OnceLock;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use candle_core::{CpuStorage, CudaStorage, DType, Device, Layout, Shape, Storage, Tensor};

const CUDA_MODULE_NAME: &str = "dinf_gemma4_moe_decode_v7";
const ROUTE_KERNEL_NAME: &str = "dinf_gemma4_moe_route";
const GEGLU_KERNEL_NAME: &str = "dinf_gemma4_packed_geglu_f32";
const GELU_MUL_KERNEL_NAME: &str = "dinf_gemma4_gelu_mul_f32";
const ROUTER_KERNEL_NAME: &str = "dinf_gemma4_router_logits_f32";
const TOP_K_KERNEL_NAME: &str = "dinf_gemma4_top_k_f32";
const ROUTER_THREADS: usize = 128;
const MAX_EXPERTS: usize = 1024;
const MAX_TOP_K: usize = 32;
/// Most threads that share one expert's ranking in the route kernel.
const ROUTE_RANK_CHUNKS: usize = 8;
const GEGLU_THREADS: usize = 256;

const CUDA_SOURCE: &str = r#"
#define DINF_ROUTE_DENOMINATOR_FLOOR 6.103515625e-5f

// Block-wide reductions: warps reduce with shuffles and meet at one barrier,
// where a block-wide tree would need one barrier per level. Each returns the
// result to every thread. `partials` holds one slot per warp and must not be
// reused by a later reduction, because no barrier follows the read. Blocks
// must be whole warps.
static __device__ __forceinline__ float dinf_warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, mask);
    }
    return value;
}

static __device__ __forceinline__ float dinf_warp_max(float value) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask));
    }
    return value;
}

static __device__ __forceinline__ float dinf_block_sum(float value, float *partials) {
    const int lane = threadIdx.x & 31;
    value = dinf_warp_sum(value);
    if (lane == 0) {
        partials[threadIdx.x >> 5] = value;
    }
    __syncthreads();
    return dinf_warp_sum(lane < (int) (blockDim.x >> 5) ? partials[lane] : 0.0f);
}

static __device__ __forceinline__ float dinf_block_max(float value, float *partials) {
    const int lane = threadIdx.x & 31;
    value = dinf_warp_max(value);
    if (lane == 0) {
        partials[threadIdx.x >> 5] = value;
    }
    __syncthreads();
    return dinf_warp_max(
        lane < (int) (blockDim.x >> 5) ? partials[lane] : -3.402823466e+38f);
}

// One block. Writes `top_k` expert IDs and `top_k` weights: selected softmax
// probabilities divided by their left-to-right sum, times each expert's
// output scale. Experts rank by logit, ties to the lower index, which is the
// probability order.
//
// The first `owners` threads each own one expert and carry the softmax, in
// the positions a block of `owners` threads would give them, so its sums
// keep their order. The block is `owners * chunks` threads: thread
// `chunk * owners + expert` counts the experts of its chunk that rank ahead
// of `expert`, which splits each owner's count over `chunks` threads.
extern "C" __global__ void dinf_gemma4_moe_route(
    const float *logits,
    const float *expert_scale,
    unsigned int *route_ids,
    float *route_weights,
    int experts,
    int top_k,
    int owners) {
    const int tid = threadIdx.x;
    const int chunks = blockDim.x / owners;
    const int expert = tid % owners;
    const int chunk = tid / owners;
    const bool owner = chunk == 0 && expert < experts;
    __shared__ int ahead[1024];
    __shared__ float partials[2][32];
    __shared__ float selected[32];
    __shared__ unsigned int selected_ids[32];

    float logit = -3.402823466e+38f;
    int rank = 0;
    if (expert < experts) {
        logit = logits[expert];
        const int span = (experts + chunks - 1) / chunks;
        const int last = min(experts, (chunk + 1) * span);
        for (int other = chunk * span; other < last; ++other) {
            const float other_logit = logits[other];
            rank += (other_logit > logit) || (other_logit == logit && other < expert);
        }
    }
    ahead[tid] = rank;
    const float max_logit =
        dinf_block_max(owner ? logit : -3.402823466e+38f, partials[0]);

    const float value = owner ? expf(logit - max_logit) : 0.0f;
    const float total = dinf_block_sum(value, partials[1]);

    if (owner) {
        for (int other_chunk = 1; other_chunk < chunks; ++other_chunk) {
            rank += ahead[other_chunk * owners + expert];
        }
        if (rank < top_k) {
            selected[rank] = value / total;
            selected_ids[rank] = (unsigned int) expert;
        }
    }
    __syncthreads();
    if (tid == 0) {
        float weight_sum = 0.0f;
        for (int slot = 0; slot < top_k; ++slot) {
            weight_sum += selected[slot];
        }
        const float denominator = fminf(
            fmaxf(weight_sum, DINF_ROUTE_DENOMINATOR_FLOOR), 3.402823466e+38f);
        for (int slot = 0; slot < top_k; ++slot) {
            const unsigned int expert = selected_ids[slot];
            route_ids[slot] = expert;
            route_weights[slot] = (selected[slot] / denominator) * expert_scale[expert];
        }
    }
}

// One block per token: the indices of its `top_k` largest values, largest
// first, ties to the lower index. It replaces a full sort of each row when
// only the leading entries are kept; the values themselves are untouched,
// so everything computed from the selection keeps its bits.
extern "C" __global__ void dinf_gemma4_top_k_f32(
    const float *values,
    unsigned int *ids,
    int columns,
    int top_k) {
    values += (long long) blockIdx.x * columns;
    ids += (long long) blockIdx.x * top_k;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = values[column];
        int rank = 0;
        for (int other = 0; other < columns; ++other) {
            const float other_value = values[other];
            rank += (other_value > value) || (other_value == value && other < column);
        }
        if (rank < top_k) {
            ids[rank] = (unsigned int) column;
        }
    }
}

// One block per expert: the dot product of that expert's router row with the
// token's router input, summed per thread, then per warp, then across warps.
extern "C" __global__ void dinf_gemma4_router_logits_f32(
    const float *router,
    const float *input,
    float *logits,
    int hidden) {
    __shared__ float partials[32];
    const int tid = threadIdx.x;
    const float *row = router + (long long) blockIdx.x * hidden;
    float sum = 0.0f;
    for (int column = tid; column < hidden; column += blockDim.x) {
        sum += row[column] * input[column];
    }
    sum = dinf_block_sum(sum, partials);
    if (tid == 0) {
        logits[blockIdx.x] = sum;
    }
}

// Candle's `gelu` (tanh form) of each row's first half times its second half.
extern "C" __global__ void dinf_gemma4_packed_geglu_f32(
    const float *gate_up,
    float *output,
    long long rows,
    long long half) {
    const long long index = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * half) {
        return;
    }
    const long long row = index / half;
    const long long column = index - row * half;
    const float gate = gate_up[row * 2 * half + column];
    const float up = gate_up[row * 2 * half + half + column];
    const float alpha = gate + 0.044715f * (gate * gate * gate);
    const float gelu = 0.5f * gate * (1.0f + tanhf(0.7978845608028654f * alpha));
    output[index] = gelu * up;
}

// Candle's `gelu` (tanh form) of `gate` times `up`, elementwise.
extern "C" __global__ void dinf_gemma4_gelu_mul_f32(
    const float *gate,
    const float *up,
    float *output,
    long long elements) {
    const long long index = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const float value = gate[index];
    const float alpha = value + 0.044715f * (value * value * value);
    output[index] = 0.5f * value * (1.0f + tanhf(0.7978845608028654f * alpha)) * up[index];
}
"#;

static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
static FAST_PATH_LOGGED: OnceLock<()> = OnceLock::new();

fn ptx() -> candle_core::Result<&'static str> {
    match PTX.get_or_init(|| {
        compile_ptx_with_opts(
            CUDA_SOURCE,
            CompileOptions {
                name: Some("gemma4_moe_decode.cu".to_string()),
                use_fast_math: Some(false),
                ..Default::default()
            },
        )
        .map(|ptx| ptx.to_src())
        .map_err(|error| error.to_string())
    }) {
        Ok(ptx) => Ok(ptx.as_str()),
        Err(error) => candle_core::bail!("failed to compile CUDA Gemma 4 MoE decode: {error}"),
    }
}

pub(super) fn precompile(device: &Device) -> anyhow::Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    let device = device.as_cuda_device()?;
    device.get_or_load_custom_func(ROUTE_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
    device.get_or_load_custom_func(GEGLU_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
    device.get_or_load_custom_func(GELU_MUL_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
    device.get_or_load_custom_func(ROUTER_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
    device.get_or_load_custom_func(TOP_K_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
    Ok(())
}

struct TopK {
    rows: usize,
    columns: usize,
    top_k: usize,
}

impl candle_core::CustomOp1 for TopK {
    fn name(&self) -> &'static str {
        "cuda-gemma4-top-k"
    }

    fn cpu_fwd(
        &self,
        _storage: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda-gemma4-top-k is only dispatched on CUDA")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let elements = self.rows * self.columns;
        let slice = storage.as_cuda_slice::<f32>()?;
        let start = layout.start_offset();
        if !layout.is_contiguous()
            || layout.shape().elem_count() != elements
            || start + elements > slice.len()
        {
            candle_core::bail!("CUDA top-k input geometry differs")
        }
        let values = slice.slice(start..start + elements);
        let device = storage.device();
        // SAFETY: Ranks are a permutation of each row's columns, so every
        // row's `top_k` slots are written.
        let mut ids = unsafe { device.alloc::<u32>(self.rows * self.top_k)? };
        let function =
            device.get_or_load_custom_func(TOP_K_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
        let (columns, top_k) = (self.columns as i32, self.top_k as i32);
        let mut builder = function.builder();
        builder.arg(&values).arg(&mut ids).arg(&columns).arg(&top_k);
        // SAFETY: The view was bounds-checked above and the grid has one
        // block per row.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (self.rows as u32, 1, 1),
                block_dim: (
                    self.columns.next_power_of_two().clamp(32, MAX_EXPERTS) as u32,
                    1,
                    1,
                ),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(ids, device.clone()),
            Shape::from((self.rows, self.top_k)),
        ))
    }
}

/// Indices of each row's `top_k` largest values, largest first and ties to
/// the lower index, as U32 `[rows, top_k]`. `values` is F32 `[rows, columns]`
/// without NaNs. Returns `None` off CUDA or outside the kernel's geometry.
pub(super) fn try_top_k(values: &Tensor, top_k: usize) -> anyhow::Result<Option<Tensor>> {
    let Ok((rows, columns)) = values.dims2() else {
        return Ok(None);
    };
    if !values.device().is_cuda()
        || values.dtype() != DType::F32
        || !values.is_contiguous()
        || rows == 0
        || rows > u32::MAX as usize
        || columns == 0
        || columns > MAX_EXPERTS
        || top_k == 0
        || top_k > columns
    {
        return Ok(None);
    }
    Ok(Some(values.apply_op1_no_bwd(&TopK {
        rows,
        columns,
        top_k,
    })?))
}

struct RouterLogits {
    experts: usize,
    hidden: usize,
}

impl candle_core::CustomOp2 for RouterLogits {
    fn name(&self) -> &'static str {
        "cuda-gemma4-router-logits"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _router: &CpuStorage,
        _router_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda-gemma4-router-logits is only dispatched on CUDA")
    }

    fn cuda_fwd(
        &self,
        input_storage: &CudaStorage,
        input_layout: &Layout,
        router_storage: &CudaStorage,
        router_layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let device = input_storage.device();
        let weights = self.experts * self.hidden;
        let input_slice = input_storage.as_cuda_slice::<f32>()?;
        let router_slice = router_storage.as_cuda_slice::<f32>()?;
        let (input_start, router_start) =
            (input_layout.start_offset(), router_layout.start_offset());
        if router_storage.device().id() != device.id()
            || !input_layout.is_contiguous()
            || !router_layout.is_contiguous()
            || input_layout.shape().elem_count() != self.hidden
            || router_layout.shape().elem_count() != weights
            || input_start + self.hidden > input_slice.len()
            || router_start + weights > router_slice.len()
        {
            candle_core::bail!("CUDA router logits input geometry differs")
        }
        let input = input_slice.slice(input_start..input_start + self.hidden);
        let router = router_slice.slice(router_start..router_start + weights);
        // SAFETY: Each block writes its expert's logit.
        let mut output = unsafe { device.alloc::<f32>(self.experts)? };
        let function =
            device.get_or_load_custom_func(ROUTER_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
        let hidden = self.hidden as i32;
        let mut builder = function.builder();
        builder
            .arg(&router)
            .arg(&input)
            .arg(&mut output)
            .arg(&hidden);
        // SAFETY: Both views were bounds-checked above; the grid has one
        // block per expert row.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (self.experts as u32, 1, 1),
                block_dim: (ROUTER_THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from((1, self.experts)),
        ))
    }
}

/// Router logits of one token: `router` (`[experts, hidden]`) times `input`
/// (`hidden` elements), as F32 `[1, experts]`. Returns `None` off CUDA or for
/// other layouts.
pub(super) fn try_router_logits(input: &Tensor, router: &Tensor) -> anyhow::Result<Option<Tensor>> {
    let Ok((experts, hidden)) = router.dims2() else {
        return Ok(None);
    };
    if !input.device().is_cuda()
        || !input.device().same_device(router.device())
        || input.dtype() != DType::F32
        || router.dtype() != DType::F32
        || !input.is_contiguous()
        || !router.is_contiguous()
        || input.elem_count() != hidden
        || experts == 0
        || hidden == 0
        || experts > MAX_EXPERTS
        || hidden > i32::MAX as usize
    {
        return Ok(None);
    }
    Ok(Some(input.apply_op2_no_bwd(
        router,
        &RouterLogits { experts, hidden },
    )?))
}

/// Route one token. `logits` is F32 `[1, experts]` and `expert_scale` F32
/// `[experts]`. Returns U32 expert IDs and F32 weights, both `[1, top_k]`,
/// or `None` outside this kernel's geometry.
pub(super) fn try_route(
    logits: &Tensor,
    expert_scale: &Tensor,
    top_k: usize,
) -> anyhow::Result<Option<(Tensor, Tensor)>> {
    let device = logits.device();
    let Ok((1, experts)) = logits.dims2() else {
        return Ok(None);
    };
    if !device.is_cuda()
        || [logits, expert_scale].iter().any(|tensor| {
            tensor.dtype() != DType::F32
                || !tensor.is_contiguous()
                || !tensor.device().same_device(device)
        })
        || expert_scale.dims() != [experts]
        || experts == 0
        || experts > MAX_EXPERTS
        || top_k == 0
        || top_k > MAX_TOP_K
        || top_k > experts
    {
        return Ok(None);
    }
    let cuda = device.as_cuda_device()?;
    // SAFETY: The kernel below writes all `top_k` IDs and weights before
    // either tensor is returned, on the device's single stream.
    let ids = unsafe { Tensor::empty((1, top_k), DType::U32, device)? };
    let weights = unsafe { Tensor::empty((1, top_k), DType::F32, device)? };
    {
        let (logits_guard, logits_layout) = logits.storage_and_layout();
        let (scale_guard, scale_layout) = expert_scale.storage_and_layout();
        let (ids_guard, ids_layout) = ids.storage_and_layout();
        let (weights_guard, weights_layout) = weights.storage_and_layout();
        let (
            Storage::Cuda(logits_storage),
            Storage::Cuda(scale_storage),
            Storage::Cuda(ids_storage),
            Storage::Cuda(weights_storage),
        ) = (&*logits_guard, &*scale_guard, &*ids_guard, &*weights_guard)
        else {
            return Ok(None);
        };
        let logits_slice = logits_storage.as_cuda_slice::<f32>()?;
        let scale_slice = scale_storage.as_cuda_slice::<f32>()?;
        let ids_slice = ids_storage.as_cuda_slice::<u32>()?;
        let weights_slice = weights_storage.as_cuda_slice::<f32>()?;
        let (logits_start, scale_start) =
            (logits_layout.start_offset(), scale_layout.start_offset());
        let (ids_start, weights_start) = (ids_layout.start_offset(), weights_layout.start_offset());
        if logits_start + experts > logits_slice.len()
            || scale_start + experts > scale_slice.len()
            || ids_start + top_k > ids_slice.len()
            || weights_start + top_k > weights_slice.len()
        {
            return Ok(None);
        }
        let logits_view = logits_slice.slice(logits_start..logits_start + experts);
        let scale_view = scale_slice.slice(scale_start..scale_start + experts);
        let ids_view = ids_slice.slice(ids_start..ids_start + top_k);
        let weights_view = weights_slice.slice(weights_start..weights_start + top_k);
        let function = cuda.get_or_load_custom_func(ROUTE_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
        let (experts_arg, top_k_arg) = (experts as i32, top_k as i32);
        // One owner thread per expert, in whole warps, and as many threads
        // per owner as fit a block to share its ranking.
        let owners = experts.next_power_of_two().clamp(32, MAX_EXPERTS);
        let chunks = (MAX_EXPERTS / owners).min(ROUTE_RANK_CHUNKS);
        let owners_arg = owners as i32;
        let mut builder = function.builder();
        builder
            .arg(&logits_view)
            .arg(&scale_view)
            .arg(&ids_view)
            .arg(&weights_view)
            .arg(&experts_arg)
            .arg(&top_k_arg)
            .arg(&owners_arg);
        // SAFETY: All four views were bounds-checked; `experts` fits the
        // kernel's shared arrays and `top_k` its 32 slots. The outputs are
        // freshly allocated and not yet shared.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: ((owners * chunks) as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
    }
    FAST_PATH_LOGGED.get_or_init(|| {
        tracing::info!("CUDA fused Gemma 4 expert routing and GELU gate enabled");
    });
    Ok(Some((ids, weights)))
}

struct PackedGeglu {
    rows: usize,
    half: usize,
}

impl candle_core::CustomOp1 for PackedGeglu {
    fn name(&self) -> &'static str {
        "cuda-gemma4-packed-geglu"
    }

    fn cpu_fwd(
        &self,
        _storage: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda-gemma4-packed-geglu is only dispatched on CUDA")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let elements = self.rows * self.half * 2;
        let slice = storage.as_cuda_slice::<f32>()?;
        let start = layout.start_offset();
        if !layout.is_contiguous()
            || layout.shape().elem_count() != elements
            || start + elements > slice.len()
        {
            candle_core::bail!("CUDA packed GEGLU input geometry differs")
        }
        let input = slice.slice(start..start + elements);
        let device = storage.device();
        let output_elements = self.rows * self.half;
        // SAFETY: The kernel writes every output element.
        let mut output = unsafe { device.alloc::<f32>(output_elements)? };
        let function =
            device.get_or_load_custom_func(GEGLU_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
        let (rows, half) = (self.rows as i64, self.half as i64);
        let mut builder = function.builder();
        builder.arg(&input).arg(&mut output).arg(&rows).arg(&half);
        // SAFETY: The input view was bounds-checked above.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (output_elements.div_ceil(GEGLU_THREADS) as u32, 1, 1),
                block_dim: (GEGLU_THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        let mut dims = layout.dims().to_vec();
        if let Some(last) = dims.last_mut() {
            *last = self.half;
        }
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from(dims),
        ))
    }
}

/// `gelu(gate) * up` over contiguous F32 rows packed as `[..., gate | up]`.
/// Returns `None` off CUDA or for other layouts.
pub(super) fn try_packed_geglu(gate_up: &Tensor) -> anyhow::Result<Option<Tensor>> {
    let width = gate_up.dim(candle_core::D::Minus1)?;
    let elements = gate_up.elem_count();
    if !gate_up.device().is_cuda()
        || gate_up.dtype() != DType::F32
        || !gate_up.is_contiguous()
        || width == 0
        || !width.is_multiple_of(2)
        || elements / 2 > (i32::MAX as usize) * GEGLU_THREADS / 2
    {
        return Ok(None);
    }
    Ok(Some(gate_up.apply_op1_no_bwd(&PackedGeglu {
        rows: elements / width,
        half: width / 2,
    })?))
}

struct GeluMul;

impl candle_core::CustomOp2 for GeluMul {
    fn name(&self) -> &'static str {
        "cuda-gemma4-gelu-mul"
    }

    fn cpu_fwd(
        &self,
        _gate: &CpuStorage,
        _gate_layout: &Layout,
        _up: &CpuStorage,
        _up_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda-gemma4-gelu-mul is only dispatched on CUDA")
    }

    fn cuda_fwd(
        &self,
        gate_storage: &CudaStorage,
        gate_layout: &Layout,
        up_storage: &CudaStorage,
        up_layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let elements = gate_layout.shape().elem_count();
        let device = gate_storage.device();
        for (storage, layout) in [(gate_storage, gate_layout), (up_storage, up_layout)] {
            if storage.dtype() != DType::F32
                || storage.device().id() != device.id()
                || !layout.is_contiguous()
                || layout.shape().elem_count() != elements
            {
                candle_core::bail!("CUDA GELU multiply input geometry differs")
            }
        }
        let (gate_start, up_start) = (gate_layout.start_offset(), up_layout.start_offset());
        let gate_slice = gate_storage.as_cuda_slice::<f32>()?;
        let up_slice = up_storage.as_cuda_slice::<f32>()?;
        if gate_start + elements > gate_slice.len() || up_start + elements > up_slice.len() {
            candle_core::bail!("CUDA GELU multiply view exceeds its storage")
        }
        let gate = gate_slice.slice(gate_start..gate_start + elements);
        let up = up_slice.slice(up_start..up_start + elements);
        // SAFETY: The kernel writes every output element.
        let mut output = unsafe { device.alloc::<f32>(elements)? };
        let function =
            device.get_or_load_custom_func(GELU_MUL_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
        let elements_arg = elements as i64;
        let mut builder = function.builder();
        builder
            .arg(&gate)
            .arg(&up)
            .arg(&mut output)
            .arg(&elements_arg);
        // SAFETY: Both input views were bounds-checked above.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (elements.div_ceil(GEGLU_THREADS) as u32, 1, 1),
                block_dim: (GEGLU_THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            gate_layout.shape().clone(),
        ))
    }
}

/// `gelu(gate) * up` for contiguous F32 tensors of one shape on CUDA.
/// Returns `None` for other inputs.
pub(super) fn try_gelu_mul(gate: &Tensor, up: &Tensor) -> anyhow::Result<Option<Tensor>> {
    let elements = gate.elem_count();
    if !gate.device().is_cuda()
        || !gate.device().same_device(up.device())
        || gate.dtype() != DType::F32
        || up.dtype() != DType::F32
        || !gate.is_contiguous()
        || !up.is_contiguous()
        || gate.shape() != up.shape()
        || elements == 0
        || elements.div_ceil(GEGLU_THREADS) > i32::MAX as usize
    {
        return Ok(None);
    }
    Ok(Some(gate.apply_op2_no_bwd(up, &GeluMul)?))
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::{try_packed_geglu, try_route, try_router_logits};

    #[test]
    fn router_logits_match_the_host_products() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        for (experts, hidden) in [(1, 1), (5, 127), (128, 2816), (33, 300)] {
            let router = values(experts * hidden, 17, 373.0);
            let input = values(hidden, 29, 211.0);
            let logits = try_router_logits(
                &Tensor::from_vec(input.clone(), (1, hidden), &device)?,
                &Tensor::from_vec(router.clone(), (experts, hidden), &device)?,
            )?
            .expect("CUDA router logits should be supported");
            assert_eq!(logits.dims(), [1, experts]);
            let logits = logits.flatten_all()?.to_vec1::<f32>()?;
            for (expert, logit) in logits.iter().enumerate() {
                let row = &router[expert * hidden..(expert + 1) * hidden];
                let expected = row
                    .iter()
                    .zip(&input)
                    .map(|(weight, value)| *weight as f64 * *value as f64)
                    .sum::<f64>();
                let scale = row
                    .iter()
                    .zip(&input)
                    .map(|(weight, value)| (*weight as f64 * *value as f64).abs())
                    .sum::<f64>();
                assert!(
                    (*logit as f64 - expected).abs() <= 1e-5 * scale.max(1.0),
                    "expert {expert}: {logit} against {expected}"
                );
            }
        }
        let cpu = Tensor::zeros((1, 4), candle_core::DType::F32, &Device::Cpu)?;
        let router = Tensor::zeros((2, 4), candle_core::DType::F32, &Device::Cpu)?;
        assert!(try_router_logits(&cpu, &router)?.is_none());
        Ok(())
    }

    fn values(elements: usize, multiplier: usize, scale: f32) -> Vec<f32> {
        (0..elements)
            .map(|index| ((index * multiplier % 1009) as f32 - 504.0) / scale)
            .collect()
    }

    #[test]
    fn top_k_matches_a_full_sort() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        let (rows, columns, top_k) = (5usize, 128usize, 8usize);
        let mut data = values(rows * columns, 131, 97.0);
        // Ties, including at the cut.
        data[3] = data[70];
        data[columns + 9] = data[columns + 10];
        let ids = super::try_top_k(
            &Tensor::from_vec(data.clone(), (rows, columns), &device)?,
            top_k,
        )?
        .expect("supported geometry")
        .to_device(&Device::Cpu)?
        .to_vec2::<u32>()?;
        for (row, ids) in ids.iter().enumerate() {
            let values = &data[row * columns..(row + 1) * columns];
            let mut order = (0..columns as u32).collect::<Vec<_>>();
            order.sort_by(|a, b| {
                values[*b as usize]
                    .total_cmp(&values[*a as usize])
                    .then(a.cmp(b))
            });
            assert_eq!(ids, &order[..top_k], "row {row}");
        }
        Ok(())
    }

    #[test]
    fn fused_route_matches_reference() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        let (experts, top_k) = (128usize, 8usize);
        let logits = values(experts, 131, 97.0);
        let scale = (0..experts)
            .map(|i| 0.5 + i as f32 * 0.125)
            .collect::<Vec<_>>();
        let (ids, weights) = try_route(
            &Tensor::from_vec(logits.clone(), (1, experts), &device)?,
            &Tensor::from_vec(scale.clone(), experts, &device)?,
            top_k,
        )?
        .expect("supported geometry");
        let ids = ids
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let weights = weights
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps = logits.iter().map(|l| (l - max).exp()).collect::<Vec<_>>();
        let total = exps.iter().sum::<f32>();
        let mut order = (0..experts).collect::<Vec<_>>();
        order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]).then(a.cmp(b)));
        let chosen = &order[..top_k];
        let selected = chosen.iter().map(|e| exps[*e] / total).collect::<Vec<_>>();
        let denominator = selected.iter().sum::<f32>().max(6.103_515_625e-5);
        assert_eq!(ids, chosen.iter().map(|e| *e as u32).collect::<Vec<_>>());
        for (slot, expert) in chosen.iter().enumerate() {
            let expected = selected[slot] / denominator * scale[*expert];
            assert!(
                (weights[slot] - expected).abs() <= expected.abs() * 1e-5,
                "slot {slot}: {} != {expected}",
                weights[slot]
            );
        }
        Ok(())
    }

    #[test]
    fn packed_geglu_matches_separate_cuda_ops() -> anyhow::Result<()> {
        use candle_nn::Module;

        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        let (rows, half) = (8usize, 704usize);
        let packed = Tensor::from_vec(
            values(rows * half * 2, 17, 97.0),
            (1, rows, half * 2),
            &device,
        )?;
        let fused = try_packed_geglu(&packed)?.expect("supported layout");
        assert_eq!(fused.dims(), [1, rows, half]);
        let gate = packed.narrow(2, 0, half)?.contiguous()?;
        let up = packed.narrow(2, half, half)?.contiguous()?;
        let reference = (candle_nn::Activation::NewGelu.forward(&gate)? * &up)?;
        let separate = super::try_gelu_mul(&gate, &up)?.expect("supported layout");
        assert_eq!(separate.dims(), reference.dims());
        let flat = |tensor: &Tensor| -> anyhow::Result<Vec<f32>> {
            Ok(tensor
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?)
        };
        for (index, (actual, expected)) in flat(&fused)?.iter().zip(&flat(&reference)?).enumerate()
        {
            assert!(
                (actual - expected).abs() <= 1e-5 + expected.abs() * 1e-5,
                "element {index}: {actual} != {expected}"
            );
        }
        Ok(())
    }
}
