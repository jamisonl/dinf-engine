//! Fused CUDA preparation of Qwen3.5 text tokens for attention.
//!
//! The portable path splits the packed query/gate projection, applies Q and K
//! RMSNorm and partial split-half RoPE, and lays the results out as heads in
//! about twenty launches. This kernel does all of it in one, mirroring the
//! Metal kernel in `qkv_layout_metal.rs`: the packed output holds Q, K, then
//! the raw gate, each as `[batch, heads, tokens, head_dim]`. For one token V
//! already has that layout and stays a reshape; for a prompt it is one
//! transposing copy.

use std::sync::OnceLock;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use candle_core::cuda_backend::cudarc::driver::{CudaView, LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use candle_core::{
    CpuStorage, CudaDevice, CudaStorage, DType, Device, Layout, Shape, Storage, Tensor,
};

/// Threads per head row: one per value, in whole warps. From 1024 values on
/// this is also Candle's `rmsnorm` block size, whose two-level sum the
/// kernels then run directly.
fn block_threads(head_dim: usize) -> u32 {
    (head_dim.div_ceil(32) * 32).min(1024) as u32
}
const MAX_HEAD_DIM: usize = 1024;
const CUDA_MODULE_NAME: &str = "dinf_qwen35_decode_layout_v7";
const CUDA_KERNEL_NAME: &str = "dinf_qwen35_decode_qkv_norm_rope_f32";
const HEAD_NORM_KERNEL_NAME: &str = "dinf_qkv_head_norm_rope_f32";

const CUDA_SOURCE: &str = r#"
// The portable path normalizes with Candle's `rmsnorm` kernel and rotates
// with its `rope` kernel. Some models' logits move measurably when a
// prompt's keys and values change in the last bit, so these kernels perform
// the same operations in the same order and produce the same bits.
//
// Candle sums a row's squares in a 32-thread block below 1024 values and a
// 1024-thread block from there on: each thread adds its strided values in
// order, the warp adds its lanes with xor shuffles, and the large block
// adds its warps the same way. Below 1024 values this kernel runs one
// thread per value so the loads are parallel, and its first warp then
// forms the 32 strided sums in Candle's order from shared memory. The norm
// is `scale * x * weight` and the rotation is written with the fused
// multiplies Candle's build produces.
static __device__ __forceinline__ float dinf_warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

static __device__ __forceinline__ unsigned short dinf_to_half(float value) {
    unsigned short result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(value));
    return result;
}

// RMSNorm of `source[0..head_dim]`, weighted unless `weight` is null, with
// split-half RoPE over the first `rotated` values, written to `destination`
// and, unless it is null, rounded to half precision to `half_destination`.
// `values` holds `head_dim` floats and `sums` 33. The block is whole warps
// with at least one thread per value up to 1024 values.
static __device__ __forceinline__ void dinf_head_norm_rope(
    const float *source,
    const float *weight,
    float eps,
    const float *cos_row,
    const float *sin_row,
    float *destination,
    unsigned short *half_destination,
    int head_dim,
    int rotated,
    float *values,
    float *sums) {
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    const int lane_id = tid % 32;

    for (int col = tid; col < head_dim; col += threads) {
        values[col] = source[col];
    }
    __syncthreads();

    // Candle's block size for this row width.
    const int block_size = head_dim < 1024 ? 32 : 1024;
    if (tid < block_size) {
        float tmp = 0.0f;
        for (int col = tid; col < head_dim; col += block_size) {
            const float xi = values[col];
            tmp += xi * xi;
        }
        tmp = dinf_warp_reduce_sum(tmp);
        if (block_size > 32) {
            if (lane_id == 0) {
                sums[tid / 32] = tmp;
            }
            __syncthreads();
            tmp = sums[lane_id];
            tmp = dinf_warp_reduce_sum(tmp);
        }
        if (tid == 0) {
            const float mean = tmp / head_dim;
            sums[32] = rsqrtf(mean + eps);
        }
    }
    __syncthreads();
    const float scale = sums[32];

    // Each thread owns its columns of `values`.
    for (int col = tid; col < head_dim; col += threads) {
        const float value = weight == nullptr
            ? scale * values[col]
            : scale * values[col] * weight[col];
        if (col < rotated) {
            values[col] = value;
        } else {
            destination[col] = value;
            if (half_destination != nullptr) {
                half_destination[col] = dinf_to_half(value);
            }
        }
    }
    __syncthreads();

    const int half = rotated >> 1;
    for (int col = tid; col < half; col += threads) {
        const float first = values[col];
        const float second = values[col + half];
        const float c = cos_row[col];
        const float s = sin_row[col];
        const float low = fmaf(first, c, -__fmul_rn(second, s));
        const float high = fmaf(first, s, __fmul_rn(second, c));
        destination[col] = low;
        destination[col + half] = high;
        if (half_destination != nullptr) {
            half_destination[col] = dinf_to_half(low);
            half_destination[col + half] = dinf_to_half(high);
        }
    }
}

// One block per head row of one token: query rows first, then key rows.
// Input rows are ordered [batch, token, head] and output rows
// [batch, head, token]; token `t` uses row `t` of the RoPE tables passed in.
// The packed output holds Q, K, then each query row's raw gate.
extern "C" __global__ void dinf_qwen35_decode_qkv_norm_rope_f32(
    const float *q_input,
    const float *k_input,
    const float *q_weight,
    const float *k_weight,
    const float *cos_row,
    const float *sin_row,
    float *output,
    int q_rows,
    int kv_rows,
    int q_heads,
    int kv_heads,
    int seq_len,
    int head_dim,
    int rope_dims,
    float q_eps,
    float k_eps) {
    const int row = blockIdx.x;
    __shared__ float values[1024];
    __shared__ float sums[33];

    const int q_elements = q_rows * head_dim;
    const int kv_elements = kv_rows * head_dim;
    const bool is_query = row < q_rows;
    const int local_row = is_query ? row : row - q_rows;
    const int heads = is_query ? q_heads : kv_heads;
    const int head = local_row % heads;
    const int token = (local_row / heads) % seq_len;
    const int batch = local_row / (heads * seq_len);
    const int output_row = (batch * heads + head) * seq_len + token;
    const float *source = is_query
        ? q_input + local_row * head_dim * 2
        : k_input + local_row * head_dim;
    float *destination = output + (is_query ? 0 : q_elements) + output_row * head_dim;

    if (is_query) {
        float *gate = output + q_elements + kv_elements + output_row * head_dim;
        for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
            gate[col] = source[head_dim + col];
        }
    }
    dinf_head_norm_rope(
        source,
        is_query ? q_weight : k_weight,
        is_query ? q_eps : k_eps,
        cos_row + token * (rope_dims >> 1),
        sin_row + token * (rope_dims >> 1),
        destination,
        nullptr,
        head_dim,
        rope_dims,
        values,
        sums);
}

// Per-head RMSNorm of separate Q, K and V projections: Q and K are weighted
// and receive split-half RoPE over their first `rope_dims` values, V is
// unweighted and not rotated. One block per head row of one token. Input
// rows are ordered [batch, token, head]; the packed output holds Q, K, then
// V, each ordered [batch, head, token]. Token `t` uses row `t` of the RoPE
// tables passed in. With `has_cache`, K and V are also rounded to half
// precision into `[batch, head, capacity]` caches, token `t` at sequence
// position `cache_start + t`.
extern "C" __global__ void dinf_qkv_head_norm_rope_f32(
    const float *q_input,
    const float *k_input,
    const float *v_input,
    const float *q_weight,
    const float *k_weight,
    const float *cos_row,
    const float *sin_row,
    float *output,
    int q_rows,
    int kv_rows,
    int q_heads,
    int kv_heads,
    int seq_len,
    int head_dim,
    int rope_dims,
    float q_eps,
    float k_eps,
    float v_eps,
    unsigned short *k_cache,
    unsigned short *v_cache,
    long long cache_capacity,
    long long cache_start,
    int has_cache) {
    const int row = blockIdx.x;
    __shared__ float values[1024];
    __shared__ float sums[33];

    const int group = row < q_rows ? 0 : (row < q_rows + kv_rows ? 1 : 2);
    const int group_start = group == 0 ? 0 : (group == 1 ? q_rows : q_rows + kv_rows);
    const int local_row = row - group_start;
    const int heads = group == 0 ? q_heads : kv_heads;
    const int head = local_row % heads;
    const int token = (local_row / heads) % seq_len;
    const int batch = local_row / (heads * seq_len);
    const int output_row = group_start + (batch * heads + head) * seq_len + token;
    const float *source = (group == 0 ? q_input : (group == 1 ? k_input : v_input))
        + local_row * head_dim;
    const float *weight = group == 0 ? q_weight : (group == 1 ? k_weight : nullptr);
    unsigned short *cache_row = nullptr;
    if (has_cache && group != 0) {
        const long long position =
            ((long long) batch * heads + head) * cache_capacity + cache_start + token;
        cache_row = (group == 1 ? k_cache : v_cache) + position * head_dim;
    }
    dinf_head_norm_rope(
        source,
        weight,
        group == 0 ? q_eps : (group == 1 ? k_eps : v_eps),
        cos_row + token * (rope_dims >> 1),
        sin_row + token * (rope_dims >> 1),
        output + output_row * head_dim,
        cache_row,
        head_dim,
        group == 2 ? 0 : rope_dims,
        values,
        sums);
}
"#;

static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
static FAST_PATH_LOGGED: OnceLock<()> = OnceLock::new();

fn ptx() -> candle_core::Result<&'static str> {
    match PTX.get_or_init(|| {
        compile_ptx_with_opts(
            CUDA_SOURCE,
            CompileOptions {
                name: Some("qwen35_decode_layout.cu".to_string()),
                use_fast_math: Some(false),
                ..Default::default()
            },
        )
        .map(|ptx| ptx.to_src())
        .map_err(|error| error.to_string())
    }) {
        Ok(ptx) => Ok(ptx.as_str()),
        Err(error) => candle_core::bail!("failed to compile CUDA Qwen3.5 decode layout: {error}"),
    }
}

pub(super) fn precompile(device: &Device) -> anyhow::Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    let device = device.as_cuda_device()?;
    device.get_or_load_custom_func(CUDA_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
    device.get_or_load_custom_func(HEAD_NORM_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
    Ok(())
}

/// Shape of the attention inputs. `position` is the first token's.
#[derive(Clone, Copy)]
pub(super) struct DecodeQkvDimensions {
    pub(super) batch: usize,
    pub(super) num_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) rope_dims: usize,
    pub(super) position: usize,
}

struct DecodeQkvNormRope {
    q_weight: Tensor,
    k_weight: Tensor,
    cos: Tensor,
    sin: Tensor,
    dimensions: DecodeQkvDimensions,
    seq_len: usize,
    q_eps: f32,
    k_eps: f32,
}

impl candle_core::CustomOp2 for DecodeQkvNormRope {
    fn name(&self) -> &'static str {
        "cuda-qwen35-decode-qkv-norm-rope"
    }

    fn cpu_fwd(
        &self,
        _q: &CpuStorage,
        _q_layout: &Layout,
        _k: &CpuStorage,
        _k_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda-qwen35-decode-qkv-norm-rope is only dispatched on CUDA")
    }

    fn cuda_fwd(
        &self,
        q_storage: &CudaStorage,
        q_layout: &Layout,
        k_storage: &CudaStorage,
        k_layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let DecodeQkvDimensions {
            batch,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dims,
            position,
        } = self.dimensions;
        let seq_len = self.seq_len;
        let q_rows = batch * seq_len * num_heads;
        let kv_rows = batch * seq_len * num_kv_heads;
        let q_elements = q_rows * head_dim;
        let kv_elements = kv_rows * head_dim;
        let half_rope = rope_dims / 2;
        let device = q_storage.device();
        ensure_same_device("K projection", device, k_storage)?;

        let q_input = storage_view("Q projection", q_storage, q_layout, 0, q_elements * 2)?;
        let k_input = storage_view("K projection", k_storage, k_layout, 0, kv_elements)?;
        let (q_weight_guard, q_weight_layout) = self.q_weight.storage_and_layout();
        let q_weight = captured_view(
            "Q norm weight",
            &q_weight_guard,
            q_weight_layout,
            0,
            head_dim,
            device,
        )?;
        let (k_weight_guard, k_weight_layout) = self.k_weight.storage_and_layout();
        let k_weight = captured_view(
            "K norm weight",
            &k_weight_guard,
            k_weight_layout,
            0,
            head_dim,
            device,
        )?;
        let table_offset = position * half_rope;
        let (cos_guard, cos_layout) = self.cos.storage_and_layout();
        let cos = captured_view(
            "RoPE cos",
            &cos_guard,
            cos_layout,
            table_offset,
            seq_len * half_rope,
            device,
        )?;
        let (sin_guard, sin_layout) = self.sin.storage_and_layout();
        let sin = captured_view(
            "RoPE sin",
            &sin_guard,
            sin_layout,
            table_offset,
            seq_len * half_rope,
            device,
        )?;

        let output_elements = q_elements * 2 + kv_elements;
        // SAFETY: Query rows write their Q and gate rows and key rows write
        // their K rows, so every output element is written.
        let mut output = unsafe { device.alloc::<f32>(output_elements)? };
        let function =
            device.get_or_load_custom_func(CUDA_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
        let config = LaunchConfig {
            grid_dim: ((q_rows + kv_rows) as u32, 1, 1),
            block_dim: (block_threads(self.dimensions.head_dim), 1, 1),
            shared_mem_bytes: 0,
        };
        let (q_rows, kv_rows) = (q_rows as i32, kv_rows as i32);
        let (q_heads, kv_heads) = (num_heads as i32, num_kv_heads as i32);
        let seq_len = seq_len as i32;
        let (head_dim, rope_dims) = (head_dim as i32, rope_dims as i32);
        let mut builder = function.builder();
        builder
            .arg(&q_input)
            .arg(&k_input)
            .arg(&q_weight)
            .arg(&k_weight)
            .arg(&cos)
            .arg(&sin)
            .arg(&mut output)
            .arg(&q_rows)
            .arg(&kv_rows)
            .arg(&q_heads)
            .arg(&kv_heads)
            .arg(&seq_len)
            .arg(&head_dim)
            .arg(&rope_dims)
            .arg(&self.q_eps)
            .arg(&self.k_eps);
        // SAFETY: Shapes, dtypes, view bounds, devices and index limits were
        // checked by `try_forward` and the view helpers above.
        unsafe { builder.launch(config) }.w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from(output_elements),
        ))
    }
}

fn ensure_same_device(
    name: &str,
    expected: &CudaDevice,
    actual: &CudaStorage,
) -> candle_core::Result<()> {
    if expected.id() != actual.device().id() {
        candle_core::bail!("CUDA Qwen3.5 decode layout {name} is on a different device")
    }
    Ok(())
}

/// `elements` f32 values starting `offset` elements into a contiguous layout.
fn storage_view<'a>(
    name: &str,
    storage: &'a CudaStorage,
    layout: &Layout,
    offset: usize,
    elements: usize,
) -> candle_core::Result<CudaView<'a, f32>> {
    if storage.dtype() != DType::F32 || !layout.is_contiguous() {
        candle_core::bail!("CUDA Qwen3.5 decode layout {name} must be contiguous f32")
    }
    let slice = storage.as_cuda_slice::<f32>()?;
    let end = offset
        .checked_add(elements)
        .filter(|end| *end <= layout.shape().elem_count());
    let start = layout.start_offset().checked_add(offset);
    match (start, end) {
        (Some(start), Some(_)) if start + elements <= slice.len() => {
            Ok(slice.slice(start..start + elements))
        }
        _ => candle_core::bail!("CUDA Qwen3.5 decode layout {name} view exceeds its storage"),
    }
}

fn captured_view<'a>(
    name: &str,
    storage: &'a Storage,
    layout: &Layout,
    offset: usize,
    elements: usize,
    expected_device: &CudaDevice,
) -> candle_core::Result<CudaView<'a, f32>> {
    let storage = match storage {
        Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("CUDA Qwen3.5 decode layout {name} is not on CUDA"),
    };
    ensure_same_device(name, expected_device, storage)?;
    storage_view(name, storage, layout, offset, elements)
}

/// Returns `(query, key, value, gate)` as `[batch, heads, tokens, head_dim]`
/// for `[batch, tokens, width]` projections of consecutive text positions,
/// or `None` when the inputs are not that CUDA f32 layout.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_forward(
    q_full: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    q_eps: f32,
    k_eps: f32,
    dimensions: DecodeQkvDimensions,
    seq_len: usize,
) -> anyhow::Result<Option<(Tensor, Tensor, Tensor, Tensor)>> {
    let DecodeQkvDimensions {
        batch,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_dims,
        position,
    } = dimensions;
    let Some(q_dim) = num_heads
        .checked_mul(head_dim)
        .and_then(|elements| elements.checked_mul(2))
    else {
        return Ok(None);
    };
    let Some(kv_dim) = num_kv_heads.checked_mul(head_dim) else {
        return Ok(None);
    };
    let table_supported = match (cos.dims2(), sin.dims2()) {
        (Ok((cos_positions, cos_half)), Ok((sin_positions, sin_half))) => {
            cos_positions == sin_positions
                && cos_half == sin_half
                && cos_half.checked_mul(2) == Some(rope_dims)
                && position
                    .checked_add(seq_len)
                    .is_some_and(|end| end <= cos_positions)
        }
        _ => false,
    };
    let tensors = [q_full, k, v, q_weight, k_weight, cos, sin];
    let token_rows = batch.checked_mul(seq_len).filter(|rows| *rows > 0);
    let total_rows = token_rows
        .and_then(|rows| rows.checked_mul(num_heads + num_kv_heads))
        .filter(|rows| *rows <= i32::MAX as usize);
    let total_elements = token_rows
        .and_then(|rows| rows.checked_mul(q_dim + kv_dim))
        .filter(|elements| *elements <= i32::MAX as usize);
    if !q_full.device().is_cuda()
        || tensors.iter().any(|tensor| {
            tensor.dtype() != DType::F32
                || !tensor.is_contiguous()
                || !tensor.device().same_device(q_full.device())
        })
        || head_dim == 0
        || head_dim > MAX_HEAD_DIM
        || rope_dims == 0
        || !rope_dims.is_multiple_of(2)
        || rope_dims > head_dim
        || !table_supported
        || total_rows.is_none()
        || total_elements.is_none()
        || num_heads == 0
        || num_kv_heads == 0
        || q_full.dims() != [batch, seq_len, q_dim]
        || k.dims() != [batch, seq_len, kv_dim]
        || v.dims() != [batch, seq_len, kv_dim]
        || q_weight.dims() != [head_dim]
        || k_weight.dims() != [head_dim]
    {
        return Ok(None);
    }
    let packed = q_full.apply_op2_no_bwd(
        k,
        &DecodeQkvNormRope {
            q_weight: q_weight.clone(),
            k_weight: k_weight.clone(),
            cos: cos.clone(),
            sin: sin.clone(),
            dimensions,
            seq_len,
            q_eps,
            k_eps,
        },
    )?;
    let q_elements = batch * seq_len * num_heads * head_dim;
    let kv_elements = batch * seq_len * num_kv_heads * head_dim;
    let query = packed
        .narrow(0, 0, q_elements)?
        .reshape((batch, num_heads, seq_len, head_dim))?;
    let key = packed.narrow(0, q_elements, kv_elements)?.reshape((
        batch,
        num_kv_heads,
        seq_len,
        head_dim,
    ))?;
    let gate = packed
        .narrow(0, q_elements + kv_elements, q_elements)?
        .reshape((batch, num_heads, seq_len, head_dim))?;
    let value = if seq_len == 1 {
        v.reshape((batch, num_kv_heads, 1, head_dim))?
    } else {
        v.reshape((batch, seq_len, num_kv_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?
    };
    FAST_PATH_LOGGED.get_or_init(|| {
        tracing::info!("CUDA fused QKV layout/RMSNorm/partial-RoPE enabled for Qwen3.5");
    });
    Ok(Some((query, key, value, gate)))
}

/// Per-head norms for separate Q, K and V projections.
/// F16 key and value caches, `[batch, kv_heads, capacity, head_dim]`, and
/// the sequence position at which the kernel writes the new rows.
pub(super) struct KvAppendSlot {
    pub(super) k: Tensor,
    pub(super) v: Tensor,
    pub(super) start: usize,
}

/// A contiguous F16 cache view holding `elements` values on `device`.
fn cache_view<'a>(
    name: &str,
    storage: &'a Storage,
    layout: &Layout,
    elements: usize,
    device: &CudaDevice,
) -> candle_core::Result<CudaView<'a, half::f16>> {
    let Storage::Cuda(storage) = storage else {
        candle_core::bail!("{name} is not on CUDA")
    };
    ensure_same_device(name, device, storage)?;
    if storage.dtype() != DType::F16
        || !layout.is_contiguous()
        || layout.shape().elem_count() != elements
    {
        candle_core::bail!("{name} must be {elements} contiguous f16 values")
    }
    let slice = storage.as_cuda_slice::<half::f16>()?;
    let start = layout.start_offset();
    if start + elements > slice.len() {
        candle_core::bail!("{name} view exceeds its storage")
    }
    Ok(slice.slice(start..start + elements))
}

struct QkvHeadNormRope {
    v: Tensor,
    append: Option<KvAppendSlot>,
    q_weight: Tensor,
    k_weight: Tensor,
    cos: Tensor,
    sin: Tensor,
    dimensions: DecodeQkvDimensions,
    seq_len: usize,
    eps: [f32; 3],
}

impl candle_core::CustomOp2 for QkvHeadNormRope {
    fn name(&self) -> &'static str {
        "cuda-qkv-head-norm-rope"
    }

    fn cpu_fwd(
        &self,
        _q: &CpuStorage,
        _q_layout: &Layout,
        _k: &CpuStorage,
        _k_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda-qkv-head-norm-rope is only dispatched on CUDA")
    }

    fn cuda_fwd(
        &self,
        q_storage: &CudaStorage,
        q_layout: &Layout,
        k_storage: &CudaStorage,
        k_layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let DecodeQkvDimensions {
            batch,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dims,
            position,
        } = self.dimensions;
        let seq_len = self.seq_len;
        let q_rows = batch * seq_len * num_heads;
        let kv_rows = batch * seq_len * num_kv_heads;
        let half_rope = rope_dims / 2;
        let device = q_storage.device();
        ensure_same_device("K projection", device, k_storage)?;

        let q_input = storage_view("Q projection", q_storage, q_layout, 0, q_rows * head_dim)?;
        let k_input = storage_view("K projection", k_storage, k_layout, 0, kv_rows * head_dim)?;
        let (v_guard, v_layout) = self.v.storage_and_layout();
        let v_input = captured_view(
            "V projection",
            &v_guard,
            v_layout,
            0,
            kv_rows * head_dim,
            device,
        )?;
        let (q_weight_guard, q_weight_layout) = self.q_weight.storage_and_layout();
        let q_weight = captured_view(
            "Q norm weight",
            &q_weight_guard,
            q_weight_layout,
            0,
            head_dim,
            device,
        )?;
        let (k_weight_guard, k_weight_layout) = self.k_weight.storage_and_layout();
        let k_weight = captured_view(
            "K norm weight",
            &k_weight_guard,
            k_weight_layout,
            0,
            head_dim,
            device,
        )?;
        let table_offset = position * half_rope;
        let (cos_guard, cos_layout) = self.cos.storage_and_layout();
        let cos = captured_view(
            "RoPE cos",
            &cos_guard,
            cos_layout,
            table_offset,
            seq_len * half_rope,
            device,
        )?;
        let (sin_guard, sin_layout) = self.sin.storage_and_layout();
        let sin = captured_view(
            "RoPE sin",
            &sin_guard,
            sin_layout,
            table_offset,
            seq_len * half_rope,
            device,
        )?;

        // The caches, when the kernel also appends K and V to them.
        let cache_guards = self
            .append
            .as_ref()
            .map(|slot| (slot.k.storage_and_layout(), slot.v.storage_and_layout()));
        let caches = match (&self.append, &cache_guards) {
            (Some(slot), Some(((k_guard, k_cache_layout), (v_guard, v_cache_layout)))) => {
                let capacity = slot.k.dim(2)?;
                if slot.start + seq_len > capacity {
                    candle_core::bail!("KV append exceeds the preallocated capacity")
                }
                let elements = batch * num_kv_heads * capacity * head_dim;
                Some((
                    cache_view("key cache", k_guard, k_cache_layout, elements, device)?,
                    cache_view("value cache", v_guard, v_cache_layout, elements, device)?,
                    capacity as i64,
                    slot.start as i64,
                ))
            }
            _ => None,
        };

        let output_elements = (q_rows + kv_rows * 2) * head_dim;
        // SAFETY: Each head row's block writes all of its `head_dim` outputs.
        let mut output = unsafe { device.alloc::<f32>(output_elements)? };
        let function =
            device.get_or_load_custom_func(HEAD_NORM_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
        let config = LaunchConfig {
            grid_dim: ((q_rows + kv_rows * 2) as u32, 1, 1),
            block_dim: (block_threads(self.dimensions.head_dim), 1, 1),
            shared_mem_bytes: 0,
        };
        let (q_rows, kv_rows) = (q_rows as i32, kv_rows as i32);
        let (q_heads, kv_heads) = (num_heads as i32, num_kv_heads as i32);
        let seq_len = seq_len as i32;
        let (head_dim, rope_dims) = (head_dim as i32, rope_dims as i32);
        let mut builder = function.builder();
        builder
            .arg(&q_input)
            .arg(&k_input)
            .arg(&v_input)
            .arg(&q_weight)
            .arg(&k_weight)
            .arg(&cos)
            .arg(&sin)
            .arg(&mut output)
            .arg(&q_rows)
            .arg(&kv_rows)
            .arg(&q_heads)
            .arg(&kv_heads)
            .arg(&seq_len)
            .arg(&head_dim)
            .arg(&rope_dims)
            .arg(&self.eps[0])
            .arg(&self.eps[1])
            .arg(&self.eps[2]);
        let has_cache = i32::from(caches.is_some());
        // Without caches the kernel never uses these pointers.
        let (placeholder, no_position) = (0u64, 0i64);
        match &caches {
            Some((k_cache, v_cache, capacity, start)) => {
                builder.arg(k_cache).arg(v_cache).arg(capacity).arg(start);
            }
            None => {
                builder
                    .arg(&placeholder)
                    .arg(&placeholder)
                    .arg(&no_position)
                    .arg(&no_position);
            }
        }
        builder.arg(&has_cache);
        // SAFETY: Shapes, dtypes, view bounds, devices and index limits were
        // checked by `try_forward_qkv_head_norm_rope` and the view helpers;
        // the cache views hold every row the kernel writes.
        unsafe { builder.launch(config) }.w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from(output_elements),
        ))
    }
}

/// Q, K and V head norms and Q/K split-half RoPE of `seq_len` tokens in one
/// launch. `q`, `k` and `v` hold `heads * head_dim` values per token, ordered
/// [batch, token, head], in any shape; `cos` and `sin` are
/// `[positions, rope_dims / 2]` and `dimensions.position` selects the first
/// token's row. V is normalized without a weight. Returns
/// `(query, key, value)` as `[batch, heads, seq_len, head_dim]`, or `None`
/// outside the kernel's geometry.
///
/// With `append`, the same launch also rounds the key and value rows to
/// F16 and writes them to the caches at `append.start`, as the separate
/// cast-and-append launch would. The caller then counts the rows as
/// appended. Caches that do not match the geometry give `None`.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_forward_qkv_head_norm_rope(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    eps: [f32; 3],
    dimensions: DecodeQkvDimensions,
    seq_len: usize,
    append: Option<KvAppendSlot>,
) -> anyhow::Result<Option<(Tensor, Tensor, Tensor)>> {
    let DecodeQkvDimensions {
        batch,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_dims,
        position,
    } = dimensions;
    let table_supported = match (cos.dims2(), sin.dims2()) {
        (Ok((cos_positions, cos_half)), Ok((sin_positions, sin_half))) => {
            cos_positions == sin_positions
                && cos_half == sin_half
                && cos_half.checked_mul(2) == Some(rope_dims)
                && position
                    .checked_add(seq_len)
                    .is_some_and(|end| end <= cos_positions)
        }
        _ => false,
    };
    let tensors = [q, k, v, q_weight, k_weight, cos, sin];
    let token_rows = batch.checked_mul(seq_len).filter(|rows| *rows > 0);
    let q_elements = token_rows
        .and_then(|rows| rows.checked_mul(num_heads))
        .and_then(|rows| rows.checked_mul(head_dim));
    let kv_elements = token_rows
        .and_then(|rows| rows.checked_mul(num_kv_heads))
        .and_then(|rows| rows.checked_mul(head_dim));
    let (Some(q_elements), Some(kv_elements)) = (q_elements, kv_elements) else {
        return Ok(None);
    };
    if !q.device().is_cuda()
        || tensors.iter().any(|tensor| {
            tensor.dtype() != DType::F32
                || !tensor.is_contiguous()
                || !tensor.device().same_device(q.device())
        })
        || head_dim == 0
        || head_dim > MAX_HEAD_DIM
        || rope_dims == 0
        || !rope_dims.is_multiple_of(2)
        || rope_dims > head_dim
        || !table_supported
        || num_heads == 0
        || num_kv_heads == 0
        || q_elements + kv_elements * 2 > i32::MAX as usize
        || q.elem_count() != q_elements
        || k.elem_count() != kv_elements
        || v.elem_count() != kv_elements
        || q_weight.dims() != [head_dim]
        || k_weight.dims() != [head_dim]
    {
        return Ok(None);
    }
    if let Some(slot) = &append {
        let cache_supported = |cache: &Tensor| {
            cache.dtype() == DType::F16
                && cache.is_contiguous()
                && cache.device().same_device(q.device())
                && cache.dims4().is_ok_and(|(lanes, heads, capacity, dim)| {
                    (lanes, heads, dim) == (batch, num_kv_heads, head_dim)
                        && slot
                            .start
                            .checked_add(seq_len)
                            .is_some_and(|end| end <= capacity)
                })
        };
        if !cache_supported(&slot.k) || !cache_supported(&slot.v) || slot.k.dims() != slot.v.dims()
        {
            return Ok(None);
        }
    }
    let packed = q.apply_op2_no_bwd(
        k,
        &QkvHeadNormRope {
            v: v.clone(),
            append,
            q_weight: q_weight.clone(),
            k_weight: k_weight.clone(),
            cos: cos.clone(),
            sin: sin.clone(),
            dimensions,
            seq_len,
            eps,
        },
    )?;
    let query = packed
        .narrow(0, 0, q_elements)?
        .reshape((batch, num_heads, seq_len, head_dim))?;
    let key = packed.narrow(0, q_elements, kv_elements)?.reshape((
        batch,
        num_kv_heads,
        seq_len,
        head_dim,
    ))?;
    let value = packed
        .narrow(0, q_elements + kv_elements, kv_elements)?
        .reshape((batch, num_kv_heads, seq_len, head_dim))?;
    Ok(Some((query, key, value)))
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::{DecodeQkvDimensions, try_forward};

    fn values(elements: usize, multiplier: usize) -> Vec<f32> {
        (0..elements)
            .map(|index| ((index * multiplier % 101) as f32 - 50.0) / 37.0)
            .collect()
    }

    /// Reference RMSNorm and partial split-half RoPE for one head row.
    fn reference_row(
        row: &[f32],
        weight: &[f32],
        cos: &[f32],
        sin: &[f32],
        rope_dims: usize,
        eps: f32,
    ) -> Vec<f32> {
        let mean = row.iter().map(|value| value * value).sum::<f32>() / row.len() as f32;
        let scale = 1.0 / (mean + eps).sqrt();
        let normalized = row
            .iter()
            .zip(weight)
            .map(|(value, weight)| value * scale * weight)
            .collect::<Vec<_>>();
        let half = rope_dims / 2;
        let mut output = normalized.clone();
        for frequency in 0..half {
            let (first, second) = (normalized[frequency], normalized[frequency + half]);
            output[frequency] = first * cos[frequency] - second * sin[frequency];
            output[frequency + half] = first * sin[frequency] + second * cos[frequency];
        }
        output
    }

    /// The portable CUDA path for `[batch, tokens, heads * head_dim]` rows:
    /// head layout, Candle's RMSNorm, then Candle's RoPE over the first
    /// `rope_dims` values.
    #[allow(clippy::too_many_arguments)]
    fn portable(
        rows: &Tensor,
        heads: usize,
        head_dim: usize,
        weight: &Tensor,
        eps: f32,
        rope_dims: usize,
        cos: &Tensor,
        sin: &Tensor,
    ) -> anyhow::Result<Tensor> {
        let (batch, seq_len, _) = rows.dims3()?;
        let rows = rows
            .reshape((batch, seq_len, heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let normalized = candle_nn::ops::rms_norm(&rows, weight, eps)?;
        if rope_dims == 0 {
            return Ok(normalized);
        }
        if rope_dims == head_dim {
            return Ok(candle_nn::rotary_emb::rope(&normalized, cos, sin)?);
        }
        let rotated = candle_nn::rotary_emb::rope(
            &normalized.narrow(3, 0, rope_dims)?.contiguous()?,
            cos,
            sin,
        )?;
        let rest = normalized
            .narrow(3, rope_dims, head_dim - rope_dims)?
            .contiguous()?;
        Ok(Tensor::cat(&[&rotated, &rest], 3)?)
    }

    /// Assert equal bits, reporting where the rows of `head_dim` values first
    /// differ and how many values do.
    fn assert_same_bits(
        actual: &Tensor,
        expected: &Tensor,
        head_dim: usize,
        label: &str,
    ) -> anyhow::Result<()> {
        let (actual, expected) = (bits(actual)?, bits(expected)?);
        assert_eq!(actual.len(), expected.len(), "{label} length");
        let different = actual
            .iter()
            .zip(&expected)
            .enumerate()
            .filter(|(_, (actual, expected))| actual != expected)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        assert!(
            different.is_empty(),
            "{label}: {} of {} values differ, first at row {} column {}",
            different.len(),
            actual.len(),
            different[0] / head_dim,
            different[0] % head_dim
        );
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

    /// Both kernels reproduce the portable path's bits, for one token and for
    /// prompts, with partial and full rotation, at both of Candle's RMSNorm
    /// block sizes.
    #[test]
    fn fused_kernels_match_the_portable_path_bit_for_bit() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        for (head_dim, rope_dims, batch, seq_len) in [
            (256usize, 64usize, 1usize, 1usize),
            (256, 256, 1, 7),
            (64, 32, 2, 5),
            (1024, 512, 1, 3),
        ] {
            let (num_heads, num_kv_heads) = (6usize, 2usize);
            let (positions, position) = (11usize, 3usize);
            let half = rope_dims / 2;
            let rows = batch * seq_len;
            let tensor = |elements: usize, multiplier: usize, shape: &[usize]| {
                Tensor::from_vec(values(elements, multiplier), shape, &device)
            };
            let q_weight = tensor(head_dim, 31, &[head_dim])?;
            let k_weight = tensor(head_dim, 37, &[head_dim])?;
            let unit = Tensor::ones(head_dim, candle_core::DType::F32, &device)?;
            let table = |f: fn(f32) -> f32| {
                Tensor::from_vec(
                    (0..positions * half)
                        .map(|index| f(index as f32 * 0.37))
                        .collect::<Vec<_>>(),
                    (positions, half),
                    &device,
                )
            };
            let (cos, sin) = (table(f32::cos)?, table(f32::sin)?);
            let (step_cos, step_sin) = (
                cos.narrow(0, position, seq_len)?.contiguous()?,
                sin.narrow(0, position, seq_len)?.contiguous()?,
            );
            let (q_eps, k_eps, v_eps) = (1e-6f32, 2e-6f32, 3e-6f32);
            let dimensions = DecodeQkvDimensions {
                batch,
                num_heads,
                num_kv_heads,
                head_dim,
                rope_dims,
                position,
            };
            let label = format!("head_dim {head_dim} rope {rope_dims} tokens {seq_len}");

            // Separate Q, K and V projections.
            let q = tensor(
                rows * num_heads * head_dim,
                17,
                &[batch, seq_len, num_heads * head_dim],
            )?;
            let k = tensor(
                rows * num_kv_heads * head_dim,
                23,
                &[batch, seq_len, num_kv_heads * head_dim],
            )?;
            let v = tensor(
                rows * num_kv_heads * head_dim,
                29,
                &[batch, seq_len, num_kv_heads * head_dim],
            )?;
            let (query, key, value) = super::try_forward_qkv_head_norm_rope(
                &q,
                &k,
                &v,
                &q_weight,
                &k_weight,
                &cos,
                &sin,
                [q_eps, k_eps, v_eps],
                dimensions,
                seq_len,
                None,
            )?
            .expect("supported geometry");

            // With caches, the same launch writes the key and value rows
            // rounded to F16 at the given position and nothing else, and
            // its F32 outputs keep their bits.
            let (capacity, start) = (seq_len + 9, 5);
            let cache = || {
                Tensor::zeros(
                    (batch, num_kv_heads, capacity, head_dim),
                    candle_core::DType::F16,
                    &device,
                )
            };
            let (k_cache, v_cache) = (cache()?, cache()?);
            let (cached_query, cached_key, cached_value) = super::try_forward_qkv_head_norm_rope(
                &q,
                &k,
                &v,
                &q_weight,
                &k_weight,
                &cos,
                &sin,
                [q_eps, k_eps, v_eps],
                dimensions,
                seq_len,
                Some(super::KvAppendSlot {
                    k: k_cache.clone(),
                    v: v_cache.clone(),
                    start,
                }),
            )?
            .expect("supported geometry");
            assert_same_bits(
                &cached_query,
                &query,
                head_dim,
                &format!("{label} cached query"),
            )?;
            assert_same_bits(&cached_key, &key, head_dim, &format!("{label} cached key"))?;
            assert_same_bits(
                &cached_value,
                &value,
                head_dim,
                &format!("{label} cached value"),
            )?;
            for (cache, rows, name) in [(&k_cache, &key, "key"), (&v_cache, &value, "value")] {
                let half_bits = |tensor: &Tensor| -> anyhow::Result<Vec<u16>> {
                    Ok(tensor
                        .to_device(&Device::Cpu)?
                        .flatten_all()?
                        .to_vec1::<half::f16>()?
                        .into_iter()
                        .map(half::f16::to_bits)
                        .collect())
                };
                assert_eq!(
                    half_bits(&cache.narrow(2, start, seq_len)?.contiguous()?)?,
                    half_bits(&rows.to_dtype(candle_core::DType::F16)?)?,
                    "{label} {name} cache rows"
                );
                let untouched = Tensor::cat(
                    &[
                        cache.narrow(2, 0, start)?,
                        cache.narrow(2, start + seq_len, capacity - start - seq_len)?,
                    ],
                    2,
                )?;
                assert!(
                    half_bits(&untouched)?.iter().all(|bits| *bits == 0),
                    "{label} {name} cache outside the appended rows"
                );
            }
            let expected_key = portable(
                &k,
                num_kv_heads,
                head_dim,
                &k_weight,
                k_eps,
                rope_dims,
                &step_cos,
                &step_sin,
            )?;
            assert_same_bits(
                &query,
                &portable(
                    &q, num_heads, head_dim, &q_weight, q_eps, rope_dims, &step_cos, &step_sin,
                )?,
                head_dim,
                &format!("{label} query"),
            )?;
            assert_same_bits(&key, &expected_key, head_dim, &format!("{label} key"))?;
            assert_same_bits(
                &value,
                &portable(
                    &v,
                    num_kv_heads,
                    head_dim,
                    &unit,
                    v_eps,
                    0,
                    &step_cos,
                    &step_sin,
                )?,
                head_dim,
                &format!("{label} value"),
            )?;

            // Packed query/gate projection.
            let q_full = tensor(
                rows * num_heads * head_dim * 2,
                41,
                &[batch, seq_len, num_heads * head_dim * 2],
            )?;
            let (query, key, _, _) = try_forward(
                &q_full, &k, &v, &q_weight, &k_weight, &cos, &sin, q_eps, k_eps, dimensions,
                seq_len,
            )?
            .expect("supported geometry");
            let q_only = q_full
                .reshape((batch, seq_len, num_heads, head_dim * 2))?
                .narrow(3, 0, head_dim)?
                .contiguous()?
                .reshape((batch, seq_len, num_heads * head_dim))?;
            assert_same_bits(
                &query,
                &portable(
                    &q_only, num_heads, head_dim, &q_weight, q_eps, rope_dims, &step_cos, &step_sin,
                )?,
                head_dim,
                &format!("{label} packed query"),
            )?;
            assert_same_bits(
                &key,
                &expected_key,
                head_dim,
                &format!("{label} packed key"),
            )?;
        }
        Ok(())
    }

    #[test]
    fn fused_qkv_head_norm_rope_matches_reference() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        // Full-width RoPE (64 of 64) and partial (32 of 64), for one token,
        // a prompt, and a batch of prompts.
        for (rope_dims, batch, seq_len) in [
            (64usize, 1usize, 1usize),
            (32, 1, 1),
            (64, 1, 5),
            (32, 2, 3),
        ] {
            let (num_heads, num_kv_heads, head_dim) = (4, 2, 64);
            let (positions, position) = (9, 2);
            let half = rope_dims / 2;
            let rows = batch * seq_len;
            let q_values = values(rows * num_heads * head_dim, 17);
            let k_values = values(rows * num_kv_heads * head_dim, 23);
            let v_values = values(rows * num_kv_heads * head_dim, 29);
            let q_weight = values(head_dim, 31);
            let k_weight = values(head_dim, 37);
            let cos = (0..positions * half)
                .map(|i| (i as f32 * 0.37).cos())
                .collect::<Vec<_>>();
            let sin = (0..positions * half)
                .map(|i| (i as f32 * 0.37).sin())
                .collect::<Vec<_>>();
            let eps = [1e-6f32, 2e-6, 3e-6];
            let tensor =
                |data: &[f32], shape: &[usize]| Tensor::from_vec(data.to_vec(), shape, &device);
            let (query, key, value) = super::try_forward_qkv_head_norm_rope(
                &tensor(&q_values, &[batch, seq_len, num_heads * head_dim])?,
                &tensor(&k_values, &[batch, seq_len, num_kv_heads * head_dim])?,
                &tensor(&v_values, &[batch, seq_len, num_kv_heads * head_dim])?,
                &tensor(&q_weight, &[head_dim])?,
                &tensor(&k_weight, &[head_dim])?,
                &tensor(&cos, &[positions, half])?,
                &tensor(&sin, &[positions, half])?,
                eps,
                DecodeQkvDimensions {
                    batch,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    rope_dims,
                    position,
                },
                seq_len,
                None,
            )?
            .expect("supported geometry");
            assert_eq!(query.dims(), [batch, num_heads, seq_len, head_dim]);
            assert_eq!(key.dims(), [batch, num_kv_heads, seq_len, head_dim]);
            assert_eq!(value.dims(), [batch, num_kv_heads, seq_len, head_dim]);
            let flat = |tensor: &Tensor| -> anyhow::Result<Vec<f32>> {
                Ok(tensor
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?)
            };
            let unit = vec![1.0f32; head_dim];
            // Outputs are ordered [batch, head, token]; inputs
            // [batch, token, head].
            let expect = |data: &[f32], heads: usize, weight: &[f32], rope: usize, eps: f32| {
                let mut expected = Vec::new();
                for item in 0..batch {
                    for head in 0..heads {
                        for token in 0..seq_len {
                            let table = (position + token) * half;
                            let start = ((item * seq_len + token) * heads + head) * head_dim;
                            expected.extend(reference_row(
                                &data[start..start + head_dim],
                                weight,
                                &cos[table..table + half],
                                &sin[table..table + half],
                                rope,
                                eps,
                            ));
                        }
                    }
                }
                expected
            };
            let close = |actual: &[f32], expected: &[f32]| {
                assert_eq!(actual.len(), expected.len());
                for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    assert!(
                        (actual - expected).abs() <= 1e-5 + expected.abs() * 1e-5,
                        "batch {batch} tokens {seq_len} mismatch at {index}: {actual} != {expected}"
                    );
                }
            };
            close(
                &flat(&query)?,
                &expect(&q_values, num_heads, &q_weight, rope_dims, eps[0]),
            );
            close(
                &flat(&key)?,
                &expect(&k_values, num_kv_heads, &k_weight, rope_dims, eps[1]),
            );
            close(
                &flat(&value)?,
                &expect(&v_values, num_kv_heads, &unit, 0, eps[2]),
            );
        }
        Ok(())
    }

    #[test]
    fn fused_layout_matches_reference_for_one_token_and_a_prompt() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        for (batch, seq_len) in [(1usize, 1usize), (1, 5), (2, 3)] {
            let (num_heads, num_kv_heads, head_dim, rope_dims) = (6, 2, 256, 64);
            let (positions, position) = (9, 3);
            let half = rope_dims / 2;
            let rows = batch * seq_len;
            let q_values = values(rows * num_heads * head_dim * 2, 17);
            let k_values = values(rows * num_kv_heads * head_dim, 23);
            let v_values = values(rows * num_kv_heads * head_dim, 29);
            let q_weight = values(head_dim, 31);
            let k_weight = values(head_dim, 37);
            let cos = (0..positions * half)
                .map(|index| (index as f32 * 0.37).cos())
                .collect::<Vec<_>>();
            let sin = (0..positions * half)
                .map(|index| (index as f32 * 0.37).sin())
                .collect::<Vec<_>>();
            let (q_eps, k_eps) = (1e-6f32, 2e-6f32);

            let tensor =
                |data: &[f32], shape: &[usize]| Tensor::from_vec(data.to_vec(), shape, &device);
            let (query, key, value, gate) = try_forward(
                &tensor(&q_values, &[batch, seq_len, num_heads * head_dim * 2])?,
                &tensor(&k_values, &[batch, seq_len, num_kv_heads * head_dim])?,
                &tensor(&v_values, &[batch, seq_len, num_kv_heads * head_dim])?,
                &tensor(&q_weight, &[head_dim])?,
                &tensor(&k_weight, &[head_dim])?,
                &tensor(&cos, &[positions, half])?,
                &tensor(&sin, &[positions, half])?,
                q_eps,
                k_eps,
                DecodeQkvDimensions {
                    batch,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    rope_dims,
                    position,
                },
                seq_len,
            )?
            .expect("supported geometry");
            assert_eq!(query.dims(), [batch, num_heads, seq_len, head_dim]);
            assert_eq!(key.dims(), [batch, num_kv_heads, seq_len, head_dim]);
            assert_eq!(value.dims(), [batch, num_kv_heads, seq_len, head_dim]);
            assert_eq!(gate.dims(), [batch, num_heads, seq_len, head_dim]);

            let flat = |tensor: &Tensor| -> anyhow::Result<Vec<f32>> {
                Ok(tensor
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?)
            };
            // Outputs are ordered [batch, head, token]; inputs
            // [batch, token, head].
            let mut expected_query = Vec::new();
            let mut expected_gate = Vec::new();
            let mut expected_key = Vec::new();
            let mut expected_value = Vec::new();
            for item in 0..batch {
                for head in 0..num_heads {
                    for token in 0..seq_len {
                        let table = (position + token) * half;
                        let start = ((item * seq_len + token) * num_heads + head) * head_dim * 2;
                        expected_query.extend(reference_row(
                            &q_values[start..start + head_dim],
                            &q_weight,
                            &cos[table..table + half],
                            &sin[table..table + half],
                            rope_dims,
                            q_eps,
                        ));
                        expected_gate
                            .extend_from_slice(&q_values[start + head_dim..start + head_dim * 2]);
                    }
                }
                for head in 0..num_kv_heads {
                    for token in 0..seq_len {
                        let table = (position + token) * half;
                        let start = ((item * seq_len + token) * num_kv_heads + head) * head_dim;
                        expected_key.extend(reference_row(
                            &k_values[start..start + head_dim],
                            &k_weight,
                            &cos[table..table + half],
                            &sin[table..table + half],
                            rope_dims,
                            k_eps,
                        ));
                        expected_value.extend_from_slice(&v_values[start..start + head_dim]);
                    }
                }
            }
            let close = |actual: &[f32], expected: &[f32]| {
                assert_eq!(actual.len(), expected.len());
                for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    assert!(
                        (actual - expected).abs() <= 1e-5 + expected.abs() * 1e-5,
                        "batch {batch} tokens {seq_len} mismatch at {index}: {actual} != {expected}"
                    );
                }
            };
            close(&flat(&query)?, &expected_query);
            close(&flat(&key)?, &expected_key);
            assert_eq!(flat(&gate)?, expected_gate);
            assert_eq!(flat(&value)?, expected_value);
        }
        Ok(())
    }
}
