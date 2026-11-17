//! Single-token CUDA attention over a native F16 key/value cache.
//!
//! The general fused path hands decode steps to FlashAttention, which first
//! re-lays-out the whole cache as `[batch, sequence, heads, dim]` and casts
//! the query to F16 on every token and layer. One query token needs none of
//! that: this kernel reads the cache through its existing strides, keeps the
//! query and all sums in F32, and writes the head outputs in one launch.

use std::sync::OnceLock;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use candle_core::{CpuStorage, CudaStorage, DType, Device, Layout, Shape, Tensor};

/// Threads per segment block. The combine kernel uses a quarter as many.
const BLOCK_THREADS: usize = 512;
const MAX_HEAD_DIM: usize = 1024;
/// Longer caches keep the FlashAttention path, which tiles the sequence.
const MAX_KV_LEN: usize = 16_384;
/// Cache positions per block. Eight threads share each position's dot
/// product, so a 512-thread block covers 64.
const SEGMENT_LEN: usize = 64;
const CUDA_MODULE_NAME: &str = "dinf_decode_sdpa_v6";
const SEGMENT_KERNEL_NAME: &str = "dinf_decode_sdpa_segment_f16kv";
const COMBINE_KERNEL_NAME: &str = "dinf_decode_sdpa_combine_f32";

// One head's cache is split into segments so the work spreads across the
// device instead of queueing on one block. Each segment block computes a
// softmax relative to its own maximum score and writes, per head and segment,
// `[v_dim unnormalized outputs, local maximum, local weight sum]`. A second
// small kernel rescales the segments to the head's global maximum and
// normalizes.
const CUDA_SOURCE: &str = r#"
#include <cuda_fp16.h>

#define DINF_LOWEST -3.402823466e+38f

extern "C" __global__ void dinf_decode_sdpa_segment_f16kv(
    const float *query,
    const __half *keys,
    const __half *values,
    float *partials,
    int kv_len,
    int segment_len,
    int qk_dim,
    int v_dim,
    int q_heads,
    int kv_heads,
    long long k_batch_stride,
    long long k_head_stride,
    long long k_seq_stride,
    long long v_batch_stride,
    long long v_head_stride,
    long long v_seq_stride,
    float scale) {
    const int row = blockIdx.x;
    const int segment = blockIdx.y;
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    const int batch = row / q_heads;
    const int head = row % q_heads;
    const int kv_head = head / (q_heads / kv_heads);
    const int first = segment * segment_len;
    const int count = max(0, min(segment_len, kv_len - first));
    const __half *key_rows = keys + batch * k_batch_stride + kv_head * k_head_stride
        + first * k_seq_stride;
    const __half *value_rows = values + batch * v_batch_stride + kv_head * v_head_stride
        + first * v_seq_stride;
    float *partial = partials
        + ((long long) row * gridDim.y + segment) * (long long) (v_dim + 2);

    __shared__ float query_row[1024];
    __shared__ float weights[512];
    __shared__ float weight_slots[512];
    __shared__ float reduce[32];
    // Two interleaved position partitions of each output column.
    __shared__ float partial_sum[2][1024];
    for (int column = tid; column < qk_dim; column += threads) {
        query_row[column] = query[(long long) row * qk_dim + column];
    }
    __syncthreads();

    // Eight threads split each position's dot product over interleaved
    // columns; `count` is at most an eighth of the block. The eight are
    // neighbouring lanes of one warp, which add their parts pairwise:
    // ((0+1)+(2+3))+((4+5)+(6+7)).
    const int position = tid >> 3;
    const int part = tid & 7;
    float score = 0.0f;
    if (position < count) {
        const __half *key = key_rows + position * k_seq_stride;
        for (int column = part; column < qk_dim; column += 8) {
            score = fmaf(query_row[column], __half2float(key[column]), score);
        }
    }
    score += __shfl_xor_sync(0xffffffffu, score, 1);
    score += __shfl_xor_sync(0xffffffffu, score, 2);
    score += __shfl_xor_sync(0xffffffffu, score, 4);
    score = part == 0 && position < count ? score * scale : DINF_LOWEST;

    // The largest score: per warp by shuffles, then across the warps.
    const int lane = tid & 31;
    const int warps = threads >> 5;
    float local_max = score;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, mask));
    }
    if (lane == 0) {
        reduce[tid >> 5] = local_max;
    }
    __syncthreads();
    local_max = lane < warps ? reduce[lane] : DINF_LOWEST;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, mask));
    }

    const float weight = part == 0 && position < count ? expf(score - local_max) : 0.0f;
    if (part == 0) {
        weights[position] = weight;
    }
    // The weight sum pairs thread `t` with `t + threads / 2`, then halves
    // the distance down to one: the first warp adds the sixteen warps' lanes
    // in that order, then its own lanes by shuffles, which continue it.
    weight_slots[tid] = weight;
    __syncthreads();
    if (tid < 32) {
        float level[16];
#pragma unroll
        for (int warp = 0; warp < 16; ++warp) {
            level[warp] = warp < warps ? weight_slots[warp * 32 + tid] : 0.0f;
        }
#pragma unroll
        for (int width = 8; width > 0; width >>= 1) {
#pragma unroll
            for (int warp = 0; warp < width; ++warp) {
                level[warp] += level[warp + width];
            }
        }
        float total = level[0];
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            total += __shfl_xor_sync(0xffffffffu, total, mask);
        }
        if (tid == 0) {
            partial[v_dim] = local_max;
            partial[v_dim + 1] = total;
        }
    }

    // Half the block takes each interleaved position partition; within it
    // threads take adjacent value columns, so cache rows are read coalesced.
    const int value_part = tid >> 8;
    for (int column = tid & 255; column < v_dim; column += 256) {
        float sum = 0.0f;
        for (int index = value_part; index < count; index += 2) {
            sum = fmaf(
                weights[index],
                __half2float(value_rows[index * v_seq_stride + column]),
                sum);
        }
        partial_sum[value_part][column] = sum;
    }
    __syncthreads();
    if (value_part == 0) {
        for (int column = tid; column < v_dim; column += 256) {
            partial[column] = partial_sum[0][column] + partial_sum[1][column];
        }
    }
}

// With `has_gate`, each output is multiplied by the sigmoid of its gate.
// With `has_quant` the Q8_1 form of all rows as one activation follows the
// output; `v_dim` and the block size are then multiples of 32.
extern "C" __global__ void dinf_decode_sdpa_combine_f32(
    const float *partials,
    const float *gate,
    float *output,
    int segments,
    int v_dim,
    int has_gate,
    int has_quant,
    int rows) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const long long stride = v_dim + 2;
    const float *row_partials = partials + (long long) row * segments * stride;

    float global_max = DINF_LOWEST;
    for (int segment = 0; segment < segments; ++segment) {
        global_max = fmaxf(global_max, row_partials[segment * stride + v_dim]);
    }
    float total = 0.0f;
    for (int segment = 0; segment < segments; ++segment) {
        const float *partial = row_partials + segment * stride;
        total = fmaf(expf(partial[v_dim] - global_max), partial[v_dim + 1], total);
    }
    const float inverse_total = 1.0f / total;
    for (int column = tid; column < v_dim; column += blockDim.x) {
        float sum = 0.0f;
        for (int segment = 0; segment < segments; ++segment) {
            const float *partial = row_partials + segment * stride;
            sum = fmaf(expf(partial[v_dim] - global_max), partial[column], sum);
        }
        float value = sum * inverse_total;
        if (has_gate) {
            value *= 1.0f / (1.0f + expf(-gate[(long long) row * v_dim + column]));
        }
        output[(long long) row * v_dim + column] = value;
        if (has_quant) {
            dinf_q8_1_store(output + rows * v_dim, row * v_dim + column, value);
        }
    }
    if (has_quant && row == rows - 1) {
        // The quantizer's row is padded with zero blocks.
        const int values = rows * v_dim;
        dinf_q8_1_block *blocks = (dinf_q8_1_block *) (output + values);
        for (int index = values + tid; index < dinf_q8_1_padded(values); index += blockDim.x) {
            blocks[index / 32].qs[index % 32] = 0;
            if (index % 32 == 0) {
                blocks[index / 32].ds[0] = 0;
                blocks[index / 32].ds[1] = 0;
            }
        }
    }
}
"#;

static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
static FAST_PATH_LOGGED: OnceLock<()> = OnceLock::new();

fn include_paths() -> Vec<String> {
    let mut paths = Vec::new();
    let mut push = |include: std::path::PathBuf| {
        if include.join("cuda_fp16.h").is_file() {
            let include = include.to_string_lossy().into_owned();
            if !paths.contains(&include) {
                paths.push(include);
            }
        }
    };
    for variable in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
        if let Some(root) = std::env::var_os(variable) {
            let root = std::path::PathBuf::from(root);
            push(root.clone());
            push(root.join("include"));
        }
    }
    for candidate in ["/usr/local/cuda/include", "/opt/cuda/include"] {
        push(std::path::PathBuf::from(candidate));
    }
    paths
}

fn ptx() -> candle_core::Result<&'static str> {
    match PTX.get_or_init(|| {
        let include_paths = include_paths();
        if include_paths.is_empty() {
            return Err(
                "CUDA toolkit include directory containing cuda_fp16.h was not found".into(),
            );
        }
        compile_ptx_with_opts(
            format!("{}{}", super::q8_1_epilogue_cuda::SOURCE, CUDA_SOURCE),
            CompileOptions {
                include_paths,
                name: Some("decode_sdpa.cu".to_string()),
                use_fast_math: Some(false),
                ..Default::default()
            },
        )
        .map(|ptx| ptx.to_src())
        .map_err(|error| error.to_string())
    }) {
        Ok(ptx) => Ok(ptx.as_str()),
        Err(error) => candle_core::bail!("failed to compile CUDA decode attention: {error}"),
    }
}

/// Whether the kernel compiled for this toolkit.
fn available() -> bool {
    ptx().is_ok()
}

/// Whether the combine kernel also writes the Q8_1 form of its output: one
/// token, with every warp of a row holding one 32-value block.
fn quantized_output(batch: usize, rows: usize, v_dim: usize) -> bool {
    batch == 1 && v_dim.is_multiple_of(32) && rows * v_dim <= i32::MAX as usize - 512
}

#[derive(Clone, Copy)]
struct Geometry {
    batch: usize,
    q_heads: usize,
    kv_heads: usize,
    kv_len: usize,
    qk_dim: usize,
    v_dim: usize,
}

struct DecodeSdpa {
    geometry: Geometry,
    scale: f32,
    /// F32 `[batch, heads, 1, v_dim]` gate whose sigmoid scales the output.
    gate: Option<Tensor>,
}

/// Strides (in elements) and the checked extent of a `[batch, heads,
/// sequence, dim]` cache view whose last dimension is dense.
fn cache_strides(layout: &Layout, storage_len: usize) -> candle_core::Result<[i64; 3]> {
    let dims = layout.dims();
    let stride = layout.stride();
    if dims.len() != 4 || stride.len() != 4 || stride[3] != 1 {
        candle_core::bail!("CUDA decode attention cache view must have a dense last dimension")
    }
    let last = (0..3).try_fold(layout.start_offset(), |extent, axis| {
        dims[axis]
            .checked_sub(1)
            .and_then(|steps| steps.checked_mul(stride[axis]))
            .and_then(|offset| extent.checked_add(offset))
    });
    match last.and_then(|last| last.checked_add(dims[3])) {
        Some(end) if end <= storage_len && stride.iter().all(|s| *s <= i64::MAX as usize) => {
            Ok([stride[0] as i64, stride[1] as i64, stride[2] as i64])
        }
        _ => candle_core::bail!("CUDA decode attention cache view exceeds its storage"),
    }
}

impl candle_core::CustomOp3 for DecodeSdpa {
    fn name(&self) -> &'static str {
        "cuda-decode-sdpa-f16kv"
    }

    fn cpu_fwd(
        &self,
        _q: &CpuStorage,
        _q_layout: &Layout,
        _k: &CpuStorage,
        _k_layout: &Layout,
        _v: &CpuStorage,
        _v_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda-decode-sdpa-f16kv is only dispatched on CUDA")
    }

    fn cuda_fwd(
        &self,
        q_storage: &CudaStorage,
        q_layout: &Layout,
        k_storage: &CudaStorage,
        k_layout: &Layout,
        v_storage: &CudaStorage,
        v_layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let Geometry {
            batch,
            q_heads,
            kv_heads,
            kv_len,
            qk_dim,
            v_dim,
        } = self.geometry;
        let rows = batch * q_heads;
        let device = q_storage.device();
        if device.id() != k_storage.device().id() || device.id() != v_storage.device().id() {
            candle_core::bail!("CUDA decode attention inputs are on different devices")
        }
        if q_storage.dtype() != DType::F32
            || k_storage.dtype() != DType::F16
            || v_storage.dtype() != DType::F16
            || !q_layout.is_contiguous()
            || q_layout.shape().elem_count() != rows * qk_dim
            || k_layout.dims() != [batch, kv_heads, kv_len, qk_dim]
            || v_layout.dims() != [batch, kv_heads, kv_len, v_dim]
        {
            candle_core::bail!("CUDA decode attention input geometry differs")
        }
        let q_slice = q_storage.as_cuda_slice::<f32>()?;
        let q_start = q_layout.start_offset();
        if q_start + rows * qk_dim > q_slice.len() {
            candle_core::bail!("CUDA decode attention query view exceeds its storage")
        }
        let query = q_slice.slice(q_start..q_start + rows * qk_dim);
        let k_slice = k_storage.as_cuda_slice::<half::f16>()?;
        let v_slice = v_storage.as_cuda_slice::<half::f16>()?;
        let k_strides = cache_strides(k_layout, k_slice.len())?;
        let v_strides = cache_strides(v_layout, v_slice.len())?;
        let keys = k_slice.slice(k_layout.start_offset()..);
        let values = v_slice.slice(v_layout.start_offset()..);

        let segment_len = SEGMENT_LEN;
        let segments = kv_len.div_ceil(segment_len);
        let partial_elements = rows * segments * (v_dim + 2);
        // SAFETY: Every segment block writes its whole partial row, and the
        // combine blocks write every output column.
        let mut partials = unsafe { device.alloc::<f32>(partial_elements)? };
        // One token's output also carries its Q8_1 form for the output
        // projection, when warps line up with the quantizer's blocks.
        let has_quant = i32::from(quantized_output(batch, rows, v_dim));
        let tail = if has_quant == 1 {
            super::q8_1_epilogue_cuda::tail_elements(rows * v_dim)
        } else {
            0
        };
        let mut output = unsafe { device.alloc::<f32>(rows * v_dim + tail)? };
        let segment_function =
            device.get_or_load_custom_func(SEGMENT_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
        let combine_function =
            device.get_or_load_custom_func(COMBINE_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
        let (kv_len_arg, segment_len_arg) = (kv_len as i32, segment_len as i32);
        let (qk_dim_arg, v_dim_arg) = (qk_dim as i32, v_dim as i32);
        let (q_heads_arg, kv_heads_arg) = (q_heads as i32, kv_heads as i32);
        let segments_arg = segments as i32;
        let rows_arg = rows as i32;
        let mut builder = segment_function.builder();
        builder
            .arg(&query)
            .arg(&keys)
            .arg(&values)
            .arg(&mut partials)
            .arg(&kv_len_arg)
            .arg(&segment_len_arg)
            .arg(&qk_dim_arg)
            .arg(&v_dim_arg)
            .arg(&q_heads_arg)
            .arg(&kv_heads_arg)
            .arg(&k_strides[0])
            .arg(&k_strides[1])
            .arg(&k_strides[2])
            .arg(&v_strides[0])
            .arg(&v_strides[1])
            .arg(&v_strides[2])
            .arg(&self.scale);
        // SAFETY: Dtypes, shapes, strided cache extents, devices and index
        // limits were checked above and in `try_forward`.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (rows as u32, segments as u32, 1),
                block_dim: (BLOCK_THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        let gate_guard = self.gate.as_ref().map(|gate| gate.storage_and_layout());
        let gate = match &gate_guard {
            Some((storage, layout)) => {
                let candle_core::Storage::Cuda(storage) = &**storage else {
                    candle_core::bail!("CUDA decode attention gate is not on CUDA")
                };
                let slice = storage.as_cuda_slice::<f32>()?;
                let start = layout.start_offset();
                if storage.device().id() != device.id()
                    || !layout.is_contiguous()
                    || layout.shape().elem_count() != rows * v_dim
                    || start + rows * v_dim > slice.len()
                {
                    candle_core::bail!("CUDA decode attention gate geometry differs")
                }
                Some(slice.slice(start..start + rows * v_dim))
            }
            None => None,
        };
        let has_gate = i32::from(gate.is_some());
        // Without a gate the kernel never reads this pointer.
        let gate_view = gate.unwrap_or_else(|| query.slice(..));
        let mut builder = combine_function.builder();
        builder
            .arg(&partials)
            .arg(&gate_view)
            .arg(&mut output)
            .arg(&segments_arg)
            .arg(&v_dim_arg)
            .arg(&has_gate)
            .arg(&has_quant)
            .arg(&rows_arg);
        // SAFETY: `partials` holds `rows * segments` rows of `v_dim + 2`
        // values written by the launch above on the same stream.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: ((BLOCK_THREADS / 4) as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from((batch, q_heads, 1, v_dim)),
        ))
    }
}

pub(super) fn precompile(device: &Device) -> anyhow::Result<()> {
    if !device.is_cuda() || !available() {
        return Ok(());
    }
    let device = device.as_cuda_device()?;
    device.get_or_load_custom_func(SEGMENT_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
    device.get_or_load_custom_func(COMBINE_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
    Ok(())
}

/// Attention for one query token: `q` is F32 `[batch, heads, 1, dim]`, `k` and
/// `v` are F16 cache views `[batch, kv_heads, len, dim]` (strided views are
/// fine). Returns F32 `[batch, heads, 1, v_dim]`, or `None` when the inputs
/// are outside this kernel's geometry.
pub(super) fn try_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
) -> anyhow::Result<Option<Tensor>> {
    try_forward_gated(q, k, v, None, scale)
}

/// [`try_forward`] with the output multiplied by `sigmoid(gate)`, where
/// `gate` is F32 `[batch, heads, 1, v_dim]`.
pub(super) fn try_forward_gated(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    gate: Option<&Tensor>,
    scale: f32,
) -> anyhow::Result<Option<Tensor>> {
    if !q.device().is_cuda()
        || !q.device().same_device(k.device())
        || !q.device().same_device(v.device())
        || q.dtype() != DType::F32
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
        || !q.is_contiguous()
        || q.rank() != 4
        || k.rank() != 4
        || v.rank() != 4
    {
        return Ok(None);
    }
    let (batch, q_heads, q_seq, qk_dim) = q.dims4()?;
    let (k_batch, kv_heads, kv_len, k_dim) = k.dims4()?;
    let (v_batch, v_heads, v_len, v_dim) = v.dims4()?;
    if q_seq != 1
        || (k_batch, v_batch) != (batch, batch)
        || v_heads != kv_heads
        || v_len != kv_len
        || k_dim != qk_dim
        || kv_heads == 0
        || !q_heads.is_multiple_of(kv_heads)
        || kv_len == 0
        || kv_len > MAX_KV_LEN
        || qk_dim == 0
        || qk_dim > MAX_HEAD_DIM
        || v_dim == 0
        || v_dim > MAX_HEAD_DIM
        || batch.checked_mul(q_heads).is_none_or(|rows| rows > 65_535)
        || gate.is_some_and(|gate| {
            gate.dtype() != DType::F32
                || !gate.is_contiguous()
                || gate.dims() != [batch, q_heads, 1, v_dim]
                || !gate.device().same_device(q.device())
        })
        || !available()
    {
        return Ok(None);
    }
    let output = q.apply_op3_no_bwd(
        k,
        v,
        &DecodeSdpa {
            geometry: Geometry {
                batch,
                q_heads,
                kv_heads,
                kv_len,
                qk_dim,
                v_dim,
            },
            scale,
            gate: gate.cloned(),
        },
    )?;
    if quantized_output(batch, batch * q_heads, v_dim) {
        // The combine kernel wrote the Q8_1 form of the whole output, which
        // the output projection reads as one activation.
        let values = q_heads * v_dim;
        super::q8_1_epilogue_cuda::publish(&output, 0, values, values)?;
    }
    FAST_PATH_LOGGED.get_or_init(|| {
        tracing::info!("CUDA single-token F16-cache attention enabled");
    });
    Ok(Some(output))
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    use super::try_forward;

    fn values(elements: usize, multiplier: usize) -> Vec<f32> {
        (0..elements)
            .map(|index| ((index * multiplier % 101) as f32 - 50.0) / 37.0)
            .collect()
    }

    #[test]
    fn decode_sdpa_matches_reference_on_strided_cache_views() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        if let Err(error) = super::ptx() {
            panic!("decode attention kernel did not compile: {error}");
        }
        // 300 positions span several segments, the last one partial.
        let (batch, q_heads, kv_heads, capacity, start, kv_len) = (1, 4, 2, 320, 7, 300);
        let (qk_dim, scale) = (64usize, 0.125f32);
        let q_values = values(batch * q_heads * qk_dim, 17);
        let k_values = values(batch * kv_heads * capacity * qk_dim, 23);
        let v_values = values(batch * kv_heads * capacity * qk_dim, 29);
        let q = Tensor::from_vec(q_values.clone(), (batch, q_heads, 1, qk_dim), &device)?;
        let cache = |data: &[f32]| -> candle_core::Result<Tensor> {
            Tensor::from_vec(data.to_vec(), (batch, kv_heads, capacity, qk_dim), &device)?
                .to_dtype(DType::F16)?
                .narrow(2, start, kv_len)
        };
        let (k, v) = (cache(&k_values)?, cache(&v_values)?);
        assert!(!k.is_contiguous());
        let output = try_forward(&q, &k, &v, scale)?
            .expect("supported geometry")
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let gate_values = values(batch * q_heads * qk_dim, 31);
        let gate = Tensor::from_vec(gate_values.clone(), (batch, q_heads, 1, qk_dim), &device)?;
        let gated = super::try_forward_gated(&q, &k, &v, Some(&gate), scale)?
            .expect("supported geometry")
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (index, (gated, plain)) in gated.iter().zip(&output).enumerate() {
            let expected = plain / (1.0 + (-gate_values[index]).exp());
            assert!(
                (gated - expected).abs() <= 1e-5 + expected.abs() * 1e-5,
                "gated output {index}: {gated} != {expected}"
            );
        }

        let round = |value: f32| half::f16::from_f32(value).to_f32();
        let n_rep = q_heads / kv_heads;
        for head in 0..q_heads {
            let kv_head = head / n_rep;
            let query = &q_values[head * qk_dim..(head + 1) * qk_dim];
            let row = |data: &[f32], position: usize| -> Vec<f32> {
                let offset = (kv_head * capacity + start + position) * qk_dim;
                data[offset..offset + qk_dim]
                    .iter()
                    .map(|v| round(*v))
                    .collect()
            };
            let scores = (0..kv_len)
                .map(|position| {
                    row(&k_values, position)
                        .iter()
                        .zip(query)
                        .map(|(key, query)| key * query)
                        .sum::<f32>()
                        * scale
                })
                .collect::<Vec<_>>();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
            let total = weights.iter().sum::<f32>();
            for column in 0..qk_dim {
                let expected = (0..kv_len)
                    .map(|position| weights[position] * row(&v_values, position)[column])
                    .sum::<f32>()
                    / total;
                let actual = output[head * qk_dim + column];
                assert!(
                    (actual - expected).abs() <= 2e-4 + expected.abs() * 2e-4,
                    "head {head} column {column}: {actual} != {expected}"
                );
            }
        }
        Ok(())
    }
}
