//! Fused CUDA cast-and-append of one key/value pair into a preallocated F16
//! cache.
//!
//! The portable path casts K and V to F16 and copies each into its cache
//! allocation: about six launches per layer and token. This kernel rounds
//! and writes both in one.

use std::sync::OnceLock;

use candle_core::cuda_backend::WrapErr;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use candle_core::{DType, Storage, Tensor};

const BLOCK_THREADS: usize = 256;
const CUDA_MODULE_NAME: &str = "dinf_kv_cast_append_v1";
const CUDA_KERNEL_NAME: &str = "dinf_kv_cast_append_f32_f16";

// Sources are `[rows, incoming, dim]`; caches are `[rows, capacity, dim]`.
// One index space covers the key elements, then the value elements.
const CUDA_SOURCE: &str = r#"
#include <cuda_fp16.h>

extern "C" __global__ void dinf_kv_cast_append_f32_f16(
    const float *k_source,
    const float *v_source,
    __half *k_cache,
    __half *v_cache,
    long long rows,
    long long incoming,
    long long k_dim,
    long long v_dim,
    long long capacity,
    long long start) {
    const long long index = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    const long long k_row = incoming * k_dim;
    const long long v_row = incoming * v_dim;
    const long long k_total = rows * k_row;
    if (index < k_total) {
        const long long row = index / k_row;
        k_cache[(row * capacity + start) * k_dim + index % k_row] =
            __float2half(k_source[index]);
        return;
    }
    const long long v_index = index - k_total;
    if (v_index < rows * v_row) {
        const long long row = v_index / v_row;
        v_cache[(row * capacity + start) * v_dim + v_index % v_row] =
            __float2half(v_source[v_index]);
    }
}
"#;

static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
static FAST_PATH_LOGGED: OnceLock<()> = OnceLock::new();

fn ptx() -> Option<&'static str> {
    PTX.get_or_init(|| {
        let mut include_paths = Vec::new();
        let mut push = |include: std::path::PathBuf| {
            if include.join("cuda_fp16.h").is_file() {
                let include = include.to_string_lossy().into_owned();
                if !include_paths.contains(&include) {
                    include_paths.push(include);
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
        if include_paths.is_empty() {
            return Err(
                "CUDA toolkit include directory containing cuda_fp16.h was not found".into(),
            );
        }
        compile_ptx_with_opts(
            CUDA_SOURCE,
            CompileOptions {
                include_paths,
                name: Some("kv_cast_append.cu".to_string()),
                use_fast_math: Some(false),
                ..Default::default()
            },
        )
        .map(|ptx| ptx.to_src())
        .map_err(|error| error.to_string())
    })
    .as_ref()
    .ok()
    .map(String::as_str)
}

/// Round F32 `k` and `v` (`[batch, heads, incoming, dim]`) to F16 and write
/// them at sequence offset `start` of the contiguous F16 caches
/// (`[batch, heads, capacity, dim]`). Returns `false` without writing when
/// the tensors are outside this kernel's geometry; the caller must have
/// checked that `start + incoming` fits the capacity.
pub(super) fn try_cast_append_pair(
    k_cache: &Tensor,
    v_cache: &Tensor,
    k: &Tensor,
    v: &Tensor,
    start: usize,
) -> anyhow::Result<bool> {
    let device = k.device();
    let tensors = [k_cache, v_cache, k, v];
    if !device.is_cuda()
        || tensors
            .iter()
            .any(|tensor| !tensor.device().same_device(device) || !tensor.is_contiguous())
        || k.dtype() != DType::F32
        || v.dtype() != DType::F32
        || k_cache.dtype() != DType::F16
        || v_cache.dtype() != DType::F16
    {
        return Ok(false);
    }
    let (batch, heads, incoming, k_dim) = k.dims4()?;
    let (v_batch, v_heads, v_incoming, v_dim) = v.dims4()?;
    let (ck_batch, ck_heads, capacity, ck_dim) = k_cache.dims4()?;
    let (cv_batch, cv_heads, v_capacity, cv_dim) = v_cache.dims4()?;
    let rows = batch * heads;
    let total = rows
        .checked_mul(incoming)
        .and_then(|elements| elements.checked_mul(k_dim + v_dim));
    let Some(total) = total.filter(|total| *total > 0) else {
        return Ok(false);
    };
    let blocks = total.div_ceil(BLOCK_THREADS);
    if (v_batch, v_heads, v_incoming) != (batch, heads, incoming)
        || (ck_batch, ck_heads, ck_dim) != (batch, heads, k_dim)
        || (cv_batch, cv_heads, cv_dim) != (batch, heads, v_dim)
        || v_capacity != capacity
        || start.checked_add(incoming).is_none_or(|end| end > capacity)
        || blocks > i32::MAX as usize
    {
        return Ok(false);
    }
    let Some(ptx) = ptx() else {
        return Ok(false);
    };

    let cuda = device.as_cuda_device()?;
    let (k_guard, k_layout) = k.storage_and_layout();
    let (v_guard, v_layout) = v.storage_and_layout();
    let (ck_guard, ck_layout) = k_cache.storage_and_layout();
    let (cv_guard, cv_layout) = v_cache.storage_and_layout();
    let (Storage::Cuda(k_storage), Storage::Cuda(v_storage)) = (&*k_guard, &*v_guard) else {
        return Ok(false);
    };
    let (Storage::Cuda(ck_storage), Storage::Cuda(cv_storage)) = (&*ck_guard, &*cv_guard) else {
        return Ok(false);
    };
    let view = |len: usize, start: usize, elements: usize| {
        start.checked_add(elements).filter(|end| *end <= len)
    };
    let k_slice = k_storage.as_cuda_slice::<f32>()?;
    let v_slice = v_storage.as_cuda_slice::<f32>()?;
    let ck_slice = ck_storage.as_cuda_slice::<half::f16>()?;
    let cv_slice = cv_storage.as_cuda_slice::<half::f16>()?;
    let (k_start, v_start) = (k_layout.start_offset(), v_layout.start_offset());
    let (ck_start, cv_start) = (ck_layout.start_offset(), cv_layout.start_offset());
    let (Some(k_end), Some(v_end), Some(ck_end), Some(cv_end)) = (
        view(k_slice.len(), k_start, rows * incoming * k_dim),
        view(v_slice.len(), v_start, rows * incoming * v_dim),
        view(ck_slice.len(), ck_start, rows * capacity * k_dim),
        view(cv_slice.len(), cv_start, rows * capacity * v_dim),
    ) else {
        return Ok(false);
    };
    let k_source = k_slice.slice(k_start..k_end);
    let v_source = v_slice.slice(v_start..v_end);
    let k_destination = ck_slice.slice(ck_start..ck_end);
    let v_destination = cv_slice.slice(cv_start..cv_end);

    let function = cuda.get_or_load_custom_func(CUDA_KERNEL_NAME, CUDA_MODULE_NAME, ptx)?;
    let (rows, incoming) = (rows as i64, incoming as i64);
    let (k_dim, v_dim) = (k_dim as i64, v_dim as i64);
    let (capacity, start) = (capacity as i64, start as i64);
    let mut builder = function.builder();
    builder
        .arg(&k_source)
        .arg(&v_source)
        .arg(&k_destination)
        .arg(&v_destination)
        .arg(&rows)
        .arg(&incoming)
        .arg(&k_dim)
        .arg(&v_dim)
        .arg(&capacity)
        .arg(&start);
    // SAFETY: Dtypes, shapes, devices and all four view extents were checked
    // above. The kernel writes only `[start, start + incoming)` of each cache
    // row; the cache owner calls this before exposing those positions, on the
    // device's single stream, so no reader observes a partial write.
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (BLOCK_THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .w()?;
    FAST_PATH_LOGGED.get_or_init(|| {
        tracing::info!("CUDA fused F16 KV cast and append enabled");
    });
    Ok(true)
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    use super::try_cast_append_pair;

    #[test]
    fn cast_append_matches_cast_and_slice_set() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        let (batch, heads, capacity, incoming, start) = (2, 3, 9, 2, 4);
        let (k_dim, v_dim) = (40, 24);
        let values = |elements: usize, multiplier: usize| -> Vec<f32> {
            (0..elements)
                .map(|index| ((index * multiplier % 1009) as f32 - 500.0) / 373.0)
                .collect()
        };
        let cache = |dim: usize, multiplier| -> candle_core::Result<Tensor> {
            Tensor::from_vec(
                values(batch * heads * capacity * dim, multiplier),
                (batch, heads, capacity, dim),
                &device,
            )?
            .to_dtype(DType::F16)
        };
        let source = |dim: usize, multiplier| {
            Tensor::from_vec(
                values(batch * heads * incoming * dim, multiplier),
                (batch, heads, incoming, dim),
                &device,
            )
        };
        let (k, v) = (source(k_dim, 17)?, source(v_dim, 23)?);
        let (k_cache, v_cache) = (cache(k_dim, 29)?, cache(v_dim, 31)?);
        let (k_expected, v_expected) = (cache(k_dim, 29)?, cache(v_dim, 31)?);
        k_expected.slice_set(&k.to_dtype(DType::F16)?, 2, start)?;
        v_expected.slice_set(&v.to_dtype(DType::F16)?, 2, start)?;

        assert!(try_cast_append_pair(&k_cache, &v_cache, &k, &v, start)?);
        let bits = |tensor: &Tensor| -> anyhow::Result<Vec<u16>> {
            Ok(tensor
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<half::f16>()?
                .into_iter()
                .map(half::f16::to_bits)
                .collect())
        };
        assert_eq!(bits(&k_cache)?, bits(&k_expected)?);
        assert_eq!(bits(&v_cache)?, bits(&v_expected)?);
        // Out-of-range appends are declined, not written.
        assert!(!try_cast_append_pair(
            &k_cache,
            &v_cache,
            &k,
            &v,
            capacity - 1
        )?);
        Ok(())
    }
}
