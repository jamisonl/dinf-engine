//! Fused single-token CUDA residual addition and RMSNorm.
//!
//! Pre-norm transformer blocks repeatedly add a residual and immediately
//! normalize the result. Returning the rounded residual and its normalized
//! view from one kernel removes one launch and one allocation at each norm
//! boundary while preserving Candle's f32 reduction order. The kernel also
//! writes the normalized row's Q8_1 form, which the quantized products of
//! that row then use instead of launching their own quantization (see
//! `q8_1_epilogue_cuda`).

use std::sync::OnceLock;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use candle_core::cuda_backend::cudarc::driver::{CudaView, LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use candle_core::{
    CpuStorage, CudaDevice, CudaStorage, DType, Device, Layout, Shape, Storage, Tensor,
};

const WARP_SIZE: usize = 32;
const CUDA_MODULE_NAME: &str = "dinf_residual_rms_norm_v2";
const CUDA_KERNEL_NAME: &str = "dinf_residual_rms_norm_f32";

const CUDA_SOURCE: &str = r#"
extern "C" __global__ void dinf_residual_rms_norm_f32(
    const float *residual,
    const float *update,
    const float *weight,
    float *output,
    int hidden_dim,
    int block_size,
    float eps) {
    const int tid = threadIdx.x;
    float sum = 0.0f;

    for (int column = tid; column < hidden_dim; column += block_size) {
        // The portable path materializes this addition as an f32 tensor before
        // RMSNorm. Prevent contraction across that rounding boundary.
        const float value = __fadd_rn(residual[column], update[column]);
        output[column] = value;
        sum = __fmaf_rn(value, value, sum);
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, mask, 32));
    }

    if (block_size > 32) {
        __shared__ float warp_sums[32];
        const int warp = tid / 32;
        const int lane = tid % 32;
        if (lane == 0) {
            warp_sums[warp] = sum;
        }
        __syncthreads();
        sum = warp_sums[lane];
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, mask, 32));
        }
    }

    const float mean = sum / hidden_dim;
    const float scale = rsqrtf(mean + eps);
    // The loop runs over the quantizer's padded row, whose length is a
    // multiple of 32, so a warp's lanes stay together: the normalized row's
    // Q8_1 form follows the two planes.
    const int end = dinf_q8_1_padded(hidden_dim);
    for (int column = tid; column < end; column += block_size) {
        float value = 0.0f;
        if (column < hidden_dim) {
            const float scaled = __fmul_rn(scale, output[column]);
            value = __fmul_rn(scaled, weight[column]);
            output[hidden_dim + column] = value;
        }
        dinf_q8_1_store(output + 2 * hidden_dim, column, value);
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
                name: Some("residual_rms_norm.cu".to_string()),
                use_fast_math: Some(false),
                ..Default::default()
            },
        )
        .map(|ptx| ptx.to_src())
        .map_err(|error| error.to_string())
    }) {
        Ok(ptx) => Ok(ptx.as_str()),
        Err(error) => candle_core::bail!("failed to compile CUDA residual RMSNorm: {error}"),
    }
}

pub(super) fn precompile(device: &Device) -> anyhow::Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    let device = device.as_cuda_device()?;
    device.get_or_load_custom_func(CUDA_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
    Ok(())
}

struct ResidualRmsNorm {
    weight: Tensor,
    hidden_dim: usize,
    block_size: usize,
    eps: f32,
}

impl candle_core::CustomOp2 for ResidualRmsNorm {
    fn name(&self) -> &'static str {
        "cuda-residual-rms-norm"
    }

    fn cpu_fwd(
        &self,
        _residual: &CpuStorage,
        _residual_layout: &Layout,
        _update: &CpuStorage,
        _update_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda-residual-rms-norm is only dispatched on CUDA")
    }

    fn cuda_fwd(
        &self,
        residual_storage: &CudaStorage,
        residual_layout: &Layout,
        update_storage: &CudaStorage,
        update_layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        validate_input(
            "residual",
            residual_storage,
            residual_layout,
            self.hidden_dim,
        )?;
        validate_input("update", update_storage, update_layout, self.hidden_dim)?;
        let device = residual_storage.device();
        ensure_same_device("update", device, update_storage)?;

        let (weight_guard, weight_layout) = self.weight.storage_and_layout();
        let weight = captured_view(
            "RMSNorm weight",
            &weight_guard,
            weight_layout,
            self.hidden_dim,
            device,
        )?;
        let residual = checked_view(
            "residual",
            residual_storage.as_cuda_slice::<f32>()?,
            residual_layout,
            self.hidden_dim,
        )?;
        let update = checked_view(
            "update",
            update_storage.as_cuda_slice::<f32>()?,
            update_layout,
            self.hidden_dim,
        )?;

        let output_elements = self
            .hidden_dim
            .checked_mul(2)
            .and_then(|planes| {
                planes.checked_add(super::q8_1_epilogue_cuda::tail_elements(self.hidden_dim))
            })
            .ok_or_else(|| candle_core::Error::Msg("residual RMSNorm output overflow".into()))?;
        // SAFETY: The kernel writes both complete output planes and the
        // normalized plane's Q8_1 form after them.
        let mut output = unsafe { device.alloc::<f32>(output_elements)? };
        let function =
            device.get_or_load_custom_func(CUDA_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (self.block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let hidden_dim = self.hidden_dim as i32;
        let block_size = self.block_size as i32;
        let mut builder = function.builder();
        builder
            .arg(&residual)
            .arg(&update)
            .arg(&weight)
            .arg(&mut output)
            .arg(&hidden_dim)
            .arg(&block_size)
            .arg(&self.eps);
        // SAFETY: Shapes, dtypes, bounds, launch dimensions, and devices were checked above.
        unsafe { builder.launch(config) }.w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from((2, self.hidden_dim)),
        ))
    }
}

fn validate_input(
    name: &str,
    storage: &CudaStorage,
    layout: &Layout,
    elements: usize,
) -> candle_core::Result<()> {
    if storage.dtype() != DType::F32
        || !layout.is_contiguous()
        || layout.shape().elem_count() != elements
    {
        candle_core::bail!(
            "CUDA residual RMSNorm {name} must be contiguous f32 with {elements} elements"
        )
    }
    let _ = checked_view(name, storage.as_cuda_slice::<f32>()?, layout, elements)?;
    Ok(())
}

fn ensure_same_device(
    name: &str,
    expected: &CudaDevice,
    actual: &CudaStorage,
) -> candle_core::Result<()> {
    if expected.id() != actual.device().id() {
        candle_core::bail!("CUDA residual RMSNorm {name} is on a different device")
    }
    Ok(())
}

fn captured_view<'a>(
    name: &str,
    storage: &'a Storage,
    layout: &Layout,
    elements: usize,
    expected_device: &CudaDevice,
) -> candle_core::Result<CudaView<'a, f32>> {
    let storage = match storage {
        Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("{name} is not on CUDA"),
    };
    validate_input(name, storage, layout, elements)?;
    ensure_same_device(name, expected_device, storage)?;
    checked_view(name, storage.as_cuda_slice::<f32>()?, layout, elements)
}

fn checked_view<'a>(
    name: &str,
    slice: &'a candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    layout: &Layout,
    elements: usize,
) -> candle_core::Result<CudaView<'a, f32>> {
    let start = layout.start_offset();
    let end = start
        .checked_add(elements)
        .ok_or_else(|| candle_core::Error::Msg(format!("{name} CUDA view overflow")))?;
    if end > slice.len() {
        candle_core::bail!("{name} CUDA view exceeds its storage")
    }
    Ok(slice.slice(start..end))
}

pub(super) fn forward(
    residual: &Tensor,
    update: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> anyhow::Result<Option<(Tensor, Tensor)>> {
    let hidden_dim = residual.dim(candle_core::D::Minus1)?;
    if hidden_dim == 0
        || !residual.device().is_cuda()
        || residual.dtype() != DType::F32
        || update.dtype() != DType::F32
        || weight.dtype() != DType::F32
        || !residual.is_contiguous()
        || !update.is_contiguous()
        || !weight.is_contiguous()
        || residual.shape() != update.shape()
        || residual.elem_count() != hidden_dim
        || weight.elem_count() != hidden_dim
        || hidden_dim > i32::MAX as usize - 512
        || !residual.device().same_device(update.device())
        || !residual.device().same_device(weight.device())
    {
        return Ok(None);
    }

    let block_size = if hidden_dim < 1024 { WARP_SIZE } else { 1024 };
    let shape = residual.shape().clone();
    let packed = residual.apply_op2_no_bwd(
        update,
        &ResidualRmsNorm {
            weight: weight.clone(),
            hidden_dim,
            block_size,
            eps,
        },
    )?;
    super::q8_1_epilogue_cuda::publish(&packed, hidden_dim, hidden_dim, 2 * hidden_dim)?;
    let residual = packed.narrow(0, 0, 1)?.reshape(shape.clone())?;
    let normalized = packed.narrow(0, 1, 1)?.reshape(shape)?;
    FAST_PATH_LOGGED.get_or_init(|| {
        tracing::info!("CUDA fused residual + RMSNorm decode path enabled");
    });
    Ok(Some((residual, normalized)))
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::forward;

    fn values(elements: usize, multiplier: usize) -> Vec<f32> {
        (0..elements)
            .map(|index| ((index * multiplier % 101) as f32 - 50.0) / 37.0)
            .collect()
    }

    fn offset_tensor(
        device: &Device,
        elements: usize,
        multiplier: usize,
    ) -> candle_core::Result<Tensor> {
        let mut padded = vec![0.0f32; elements * 2];
        padded[elements..].copy_from_slice(&values(elements, multiplier));
        Tensor::from_vec(padded, (2, 1, elements), device)?.narrow(0, 1, 1)
    }

    #[test]
    fn fused_residual_rms_norm_matches_separate_cuda_ops() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        let eps = 1e-6f32;
        for hidden_dim in [256, 1536, 3584, 4096] {
            let residual = offset_tensor(&device, hidden_dim, 17)?;
            let update = offset_tensor(&device, hidden_dim, 23)?;
            let weight = offset_tensor(&device, hidden_dim, 31)?
                .squeeze(0)?
                .squeeze(0)?;
            let expected_residual = (&residual + &update)?;
            let expected_normalized = candle_nn::ops::rms_norm(&expected_residual, &weight, eps)?;
            let (actual_residual, actual_normalized) =
                forward(&residual, &update, &weight, eps)?.expect("eligible fused residual norm");

            for (name, actual, expected) in [
                ("residual", actual_residual, expected_residual),
                ("normalized", actual_normalized, expected_normalized),
            ] {
                let actual = actual
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let expected = expected
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let max_abs_diff = actual
                    .iter()
                    .zip(expected.iter())
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0f32, f32::max);
                assert_eq!(
                    max_abs_diff, 0.0,
                    "{name} hidden_dim={hidden_dim} drifted by {max_abs_diff}",
                );
            }
        }
        Ok(())
    }
}
