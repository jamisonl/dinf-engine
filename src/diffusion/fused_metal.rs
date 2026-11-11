//! Native F32/BF16 diffusion kernels. CPU/CUDA retain the reference graph.
//! Reduction rounding can differ from separate kernels; no dtype is reduced.
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{
    CpuStorage, CustomOp2, CustomOp3, DType, Layout, MetalStorage, Result, Shape, Tensor,
};
use candle_metal_kernels::metal::{ComputePipeline, Library};
#[cfg(target_arch = "aarch64")]
use objc2_foundation::{NSOperatingSystemVersion, NSProcessInfo};
use objc2_metal::{MTLCompileOptions, MTLSize};
#[cfg(target_arch = "aarch64")]
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLGPUFamily};

pub(super) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("DINF_DIFFUSION_FUSED").as_deref() != Ok("0"))
}

struct Programs {
    library: Library,
    pipelines: HashMap<String, ComputePipeline>,
}

fn pipeline(
    device: &candle_core::MetalDevice,
    operation: &str,
    dtype: DType,
) -> Result<ComputePipeline> {
    static CACHE: OnceLock<Mutex<HashMap<DeviceId, Programs>>> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| candle_core::Error::Msg("diffusion kernel cache poisoned".into()))?;
    if let std::collections::hash_map::Entry::Vacant(entry) = cache.entry(device.id()) {
        let options = MTLCompileOptions::new();
        #[allow(deprecated)]
        options.setFastMathEnabled(false);
        let library = device
            .metal_device()
            .new_library_with_source(include_str!("fused.metal"), Some(&options))
            .map_err(candle_core::Error::wrap)?;
        entry.insert(Programs {
            library,
            pipelines: Default::default(),
        });
    }
    let name = format!(
        "{operation}_{}",
        if dtype == DType::BF16 { "bf16" } else { "f32" }
    );
    let programs = cache
        .get_mut(&device.id())
        .expect("inserted kernel library");
    if let Some(pipeline) = programs.pipelines.get(&name) {
        return Ok(pipeline.clone());
    }
    let function = programs
        .library
        .get_function(&name, None)
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    programs.pipelines.insert(name, pipeline.clone());
    Ok(pipeline)
}

fn offset(
    storage: &MetalStorage,
    layout: &Layout,
    device: &candle_core::MetalDevice,
    dtype: DType,
) -> Result<usize> {
    if storage.device().id() != device.id() || storage.dtype() != dtype || !layout.is_contiguous() {
        candle_core::bail!(
            "diffusion kernel inputs must be contiguous, matching precision, on one Metal device"
        );
    }
    let count = layout.shape().elem_count();
    u32::try_from(count).map_err(candle_core::Error::wrap)?;
    let end = layout
        .start_offset()
        .checked_add(count)
        .and_then(|v| v.checked_mul(dtype.size_in_bytes()))
        .ok_or_else(|| candle_core::Error::Msg("diffusion kernel span overflow".into()))?;
    if count == 0 || end > storage.buffer().length() {
        candle_core::bail!("diffusion kernel input exceeds buffer or is empty");
    }
    Ok(layout.start_offset() * dtype.size_in_bytes())
}

fn validate(inputs: &[&Tensor]) -> Result<()> {
    let first = inputs[0];
    if !matches!(first.dtype(), DType::F32 | DType::BF16)
        || !first.device().is_metal()
        || inputs
            .iter()
            .any(|t| !first.device().same_device(t.device()))
    {
        candle_core::bail!("diffusion kernels require F32/BF16 inputs on one Metal device");
    }
    Ok(())
}

struct NormScale {
    eps: f32,
    #[cfg(target_arch = "aarch64")]
    native_bf16: bool,
}

impl CustomOp2 for NormScale {
    fn name(&self) -> &'static str {
        "diffusion-norm-scale"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("diffusion fused normalization requires Metal")
    }
    fn metal_fwd(
        &self,
        x: &MetalStorage,
        xl: &Layout,
        scale: &MetalStorage,
        sl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let device = x.device();
        let xo = offset(x, xl, device, x.dtype())?;
        let so = offset(scale, sl, device, x.dtype())?;
        if xl.shape() != sl.shape() {
            candle_core::bail!("diffusion normalization shapes differ");
        }
        let width = *xl
            .dims()
            .last()
            .ok_or_else(|| candle_core::Error::Msg("missing normalization width".into()))?;
        if width == 0 || width > 32768 {
            candle_core::bail!("unsupported diffusion normalization width");
        }
        let count = xl.shape().elem_count();
        let operation = "norm_scale";
        #[cfg(target_arch = "aarch64")]
        let operation = if self.native_bf16 {
            "norm_scale_native"
        } else {
            operation
        };
        let pipeline = pipeline(device, operation, x.dtype())?;
        // Preserve the legacy reduction unless the native BF16 path is selected.
        let threads = (width / 2)
            .next_power_of_two()
            .clamp(32, 1024)
            .min(pipeline.max_total_threads_per_threadgroup());
        #[cfg(target_arch = "aarch64")]
        let threads = if self.native_bf16 {
            let threads = native_norm_threads(xl.dims(), x.dtype(), self.eps).ok_or_else(|| {
                candle_core::Error::Msg("native norm shape exceeds bounds".into())
            })?;
            if !native_norm_pipeline_admits(device, &pipeline, threads) {
                candle_core::bail!("native norm pipeline exceeds device limits");
            }
            threads
        } else {
            threads
        };
        let output = device
            .new_buffer_builder()
            .with_size_for(count, x.dtype())
            .with_label(self.name())
            .build()?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder.set_input_buffer(0, Some(x.buffer()), xo);
        encoder.set_input_buffer(1, Some(scale.buffer()), so);
        encoder.set_output_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, &(width as u32));
        encoder.set_bytes(4, &self.eps);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: count / width,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), count, x.dtype()),
            xl.shape().clone(),
        ))
    }
}

#[derive(Clone, Copy)]
enum Ternary {
    Residual,
    Rope,
    #[cfg(target_arch = "aarch64")]
    RopeNativeBf16,
}

impl CustomOp3 for Ternary {
    fn name(&self) -> &'static str {
        match self {
            Self::Residual => "gated_residual",
            Self::Rope => "rope",
            #[cfg(target_arch = "aarch64")]
            Self::RopeNativeBf16 => "rope_native",
        }
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("diffusion fused epilogues require Metal")
    }
    fn metal_fwd(
        &self,
        x: &MetalStorage,
        xl: &Layout,
        a: &MetalStorage,
        al: &Layout,
        b: &MetalStorage,
        bl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let device = x.device();
        let extra_dtype = match self {
            Self::Residual => x.dtype(),
            Self::Rope => DType::F32,
            #[cfg(target_arch = "aarch64")]
            Self::RopeNativeBf16 => DType::F32,
        };
        let offsets = [
            offset(x, xl, device, x.dtype())?,
            offset(a, al, device, extra_dtype)?,
            offset(b, bl, device, extra_dtype)?,
        ];
        let count = xl.shape().elem_count();
        let (work, sequence, half_dim) = match self {
            Self::Residual => {
                if xl.shape() != al.shape() || xl.shape() != bl.shape() {
                    candle_core::bail!("diffusion residual shapes differ");
                }
                (count, 0, 0)
            }
            Self::Rope => {
                let (_, _, sequence, dim) = xl.shape().dims4()?;
                if dim % 2 != 0 || al.dims() != [sequence, dim / 2] || al.shape() != bl.shape() {
                    candle_core::bail!("diffusion RoPE shapes differ");
                }
                (count / 2, sequence, dim / 2)
            }
            #[cfg(target_arch = "aarch64")]
            Self::RopeNativeBf16 => {
                let (_, _, sequence, dim) = xl.shape().dims4()?;
                if dim % 2 != 0 || al.dims() != [sequence, dim / 2] || al.shape() != bl.shape() {
                    candle_core::bail!("diffusion RoPE shapes differ");
                }
                (count / 2, sequence, dim / 2)
            }
        };
        let output = device
            .new_buffer_builder()
            .with_size_for(count, x.dtype())
            .with_label(self.name())
            .build()?;
        let pipeline = pipeline(device, self.name(), x.dtype())?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        for (i, storage) in [x, a, b].into_iter().enumerate() {
            encoder.set_input_buffer(i, Some(storage.buffer()), offsets[i]);
        }
        encoder.set_output_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &(work as u32));
        let is_rope = matches!(self, Self::Rope);
        #[cfg(target_arch = "aarch64")]
        let is_rope = is_rope || matches!(self, Self::RopeNativeBf16);
        if is_rope {
            encoder.set_bytes(5, &(sequence as u32));
            encoder.set_bytes(6, &(half_dim as u32));
        }
        let threads = 256.min(pipeline.max_total_threads_per_threadgroup());
        encoder.dispatch_thread_groups(
            MTLSize {
                width: work.div_ceil(threads),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), count, x.dtype()),
            xl.shape().clone(),
        ))
    }
}

pub(super) fn norm_scale(input: &Tensor, scale: &Tensor, eps: f32) -> Result<Tensor> {
    validate(&[input, scale])?;
    if input.dims() != scale.dims()
        || input.dtype() != scale.dtype()
        || !eps.is_finite()
        || eps <= 0.0
    {
        candle_core::bail!("invalid diffusion normalization inputs");
    }
    input.contiguous()?.apply_op2_no_bwd(
        &scale.contiguous()?,
        &NormScale {
            eps,
            #[cfg(target_arch = "aarch64")]
            native_bf16: false,
        },
    )
}

pub(super) fn gated_residual(hidden: &Tensor, gate: &Tensor, update: &Tensor) -> Result<Tensor> {
    validate(&[hidden, gate, update])?;
    if [gate, update]
        .iter()
        .any(|t| t.dims() != hidden.dims() || t.dtype() != hidden.dtype())
    {
        candle_core::bail!("invalid diffusion residual inputs");
    }
    hidden.contiguous()?.apply_op3_no_bwd(
        &gate.contiguous()?,
        &update.contiguous()?,
        &Ternary::Residual,
    )
}

pub(super) fn rope(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    validate(&[input, cos, sin])?;
    input
        .contiguous()?
        .apply_op3_no_bwd(&cos.contiguous()?, &sin.contiguous()?, &Ternary::Rope)
}

// The native BF16 model multiplies F32 complex pairs before BF16 rounding.
// Reuse the same layout, storage-span and output checks as the legacy kernel.
#[cfg(target_arch = "aarch64")]
pub(super) fn rope_native_bf16(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    validate(&[input, cos, sin])?;
    let (batch, heads, sequence, dim) = input.dims4()?;
    if input.dtype() != DType::BF16
        || cos.dtype() != DType::F32
        || sin.dtype() != DType::F32
        || batch == 0
        || heads == 0
        || sequence == 0
        || dim == 0
        || dim % 2 != 0
        || cos.dims() != [sequence, dim / 2]
        || cos.dims() != sin.dims()
        || cos.elem_count() > 8 * 1024 * 1024
    {
        candle_core::bail!("invalid native BF16 diffusion RoPE input or table");
    }
    u32::try_from(input.elem_count()).map_err(candle_core::Error::wrap)?;
    input.contiguous()?.apply_op3_no_bwd(
        &cos.contiguous()?,
        &sin.contiguous()?,
        &Ternary::RopeNativeBf16,
    )
}

#[cfg(target_arch = "aarch64")]
fn native_norm_threads(dims: &[usize], dtype: DType, eps: f32) -> Option<usize> {
    if dtype != DType::BF16
        || !(2..=4).contains(&dims.len())
        || !eps.is_finite()
        || eps <= 0.0
        || dims.contains(&0)
    {
        return None;
    }
    let width = *dims.last()?;
    let count = dims.iter().try_fold(1_usize, |n, d| n.checked_mul(*d))?;
    if width > 4096 || count > 16 * 1024 * 1024 {
        return None;
    }
    Some(width.div_ceil(4))
}

#[cfg(target_arch = "aarch64")]
fn native_norm_pipeline_admits(
    device: &candle_core::MetalDevice,
    p: &ComputePipeline,
    threads: usize,
) -> bool {
    p.max_total_threads_per_threadgroup() >= threads
        && p.as_ref().threadExecutionWidth() == 32
        && p.as_ref().staticThreadgroupMemoryLength()
            <= device.metal_device().as_ref().maxThreadgroupMemoryLength()
}

// Keep the native BF16 arithmetic independent of the optional fusion speed flag.
// Every decline happens before contiguous copies or output allocation.
#[cfg(target_arch = "aarch64")]
pub(super) fn try_native_norm_scale(
    input: &Tensor,
    scale: &Tensor,
    eps: f32,
) -> Result<Option<Tensor>> {
    let Some(threads) = native_norm_threads(input.dims(), input.dtype(), eps) else {
        return Ok(None);
    };
    if scale.dtype() != DType::BF16
        || scale.dims() != input.dims()
        || !input.device().is_metal()
        || !input.device().same_device(scale.device())
    {
        return Ok(None);
    }
    let device = input.device().as_metal_device()?;
    let os =
        NSProcessInfo::processInfo().isOperatingSystemAtLeastVersion(NSOperatingSystemVersion {
            majorVersion: 15,
            minorVersion: 0,
            patchVersion: 0,
        });
    if !os
        || !device
            .metal_device()
            .as_ref()
            .supportsFamily(MTLGPUFamily::Apple7)
    {
        return Ok(None);
    }
    let p = pipeline(device, "norm_scale_native", DType::BF16)?;
    if !native_norm_pipeline_admits(device, &p, threads) {
        return Ok(None);
    }
    Ok(Some(input.contiguous()?.apply_op2_no_bwd(
        &scale.contiguous()?,
        &NormScale {
            eps,
            native_bf16: true,
        },
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::D;

    fn values(t: &Tensor) -> Result<Vec<f32>> {
        t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
    }
    fn data(count: usize, device: &candle_core::Device, dtype: DType) -> Result<Tensor> {
        Tensor::from_vec(
            (0..count)
                .map(|i| (i as f32 * 0.013).sin() * 3.1)
                .collect::<Vec<_>>(),
            count,
            device,
        )?
        .to_dtype(dtype)
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn native_bf16_norm_checks_bounds_before_copies() -> Result<()> {
        assert_eq!(
            native_norm_threads(&[4096, 4096], DType::BF16, 1e-6),
            Some(1024)
        );
        assert_eq!(
            native_norm_threads(&[1, 4096, 1, 4096], DType::BF16, 1e-6),
            Some(1024)
        );
        assert_eq!(native_norm_threads(&[1, 4097], DType::BF16, 1e-6), None);
        assert_eq!(native_norm_threads(&[4097, 4096], DType::BF16, 1e-6), None);
        assert_eq!(
            native_norm_threads(&[usize::MAX, 2], DType::BF16, 1e-6),
            None
        );
        for dims in [vec![], vec![7], vec![0, 7], vec![1, 1, 1, 1, 7]] {
            assert_eq!(native_norm_threads(&dims, DType::BF16, 1e-6), None);
        }
        for dtype in [DType::F32, DType::F16] {
            assert_eq!(native_norm_threads(&[2, 7], dtype, 1e-6), None);
        }
        for eps in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(native_norm_threads(&[2, 7], DType::BF16, eps), None);
        }
        let cpu = Tensor::zeros((2, 7), DType::BF16, &candle_core::Device::Cpu)?;
        assert!(try_native_norm_scale(&cpu, &cpu, 1e-6)?.is_none());
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let x = Tensor::zeros((2, 7), DType::BF16, &device)?;
        assert!(try_native_norm_scale(&x.to_dtype(DType::F32)?, &x, 1e-6)?.is_none());
        assert!(try_native_norm_scale(&x, &x.to_dtype(DType::F32)?, 1e-6)?.is_none());
        assert!(try_native_norm_scale(&x, &x.narrow(0, 0, 1)?, 1e-6)?.is_none());
        assert!(try_native_norm_scale(&x, &cpu, 1e-6)?.is_none());
        // This huge view uses one stored value. A decline must precede its copy.
        let large = Tensor::zeros((1, 1), DType::BF16, &device)?.broadcast_as((4097, 4096))?;
        assert!(try_native_norm_scale(&large, &large, 1e-6)?.is_none());
        Ok(())
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn native_bf16_norm_handles_views_and_preserves_rounding() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for width in [1, 7, 31, 37, 128, 129, 257, 4096] {
            let x = data(8 * width, &device, DType::BF16)?
                .narrow(0, width, 6 * width)?
                .reshape((2, 3, width))?
                .transpose(0, 1)?;
            let scale = ((&x.contiguous()? * 0.125)? + 1.0)?;
            let scale = scale.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
            let Some(actual) = try_native_norm_scale(&x, &scale, 1e-6)? else {
                continue;
            };
            let contiguous = try_native_norm_scale(&x.contiguous()?, &scale.contiguous()?, 1e-6)?
                .expect("same admitted shape and device");
            assert_eq!(values(&actual)?, values(&contiguous)?);
            let input = values(&x.contiguous()?)?;
            let scales = values(&scale.contiguous()?)?;
            let actual = values(&actual)?;
            for (row, (scale, actual)) in input
                .chunks_exact(width)
                .zip(scales.chunks_exact(width).zip(actual.chunks_exact(width)))
            {
                let mean = row.iter().map(|v| f64::from(*v)).sum::<f64>() / width as f64;
                let var = row
                    .iter()
                    .map(|v| (f64::from(*v) - mean).powi(2))
                    .sum::<f64>()
                    / width as f64;
                for ((input, scale), actual) in row.iter().zip(scale).zip(actual) {
                    let normalized = half::bf16::from_f64(
                        (f64::from(*input) - mean) / (var + f64::from(1e-6_f32)).sqrt(),
                    );
                    let expected = half::bf16::from_f32(normalized.to_f32() * scale).to_f32();
                    assert!(actual.is_finite());
                    // Keep the pre-existing BF16 norm tolerance against independent F64 math.
                    assert!(
                        (actual - expected).abs() <= 0.016,
                        "width={width}: {actual} != {expected}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn native_bf16_rope_matches_complex_fma_with_strides_and_offsets() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for strided in [false, true] {
            let x = data(2 * 3 * 7 * 128 + 128, &device, DType::BF16)?.narrow(
                0,
                128,
                2 * 3 * 7 * 128,
            )?;
            let x = if strided {
                x.reshape((2, 7, 3, 128))?.transpose(1, 2)?
            } else {
                x.reshape((2, 3, 7, 128))?
            };
            let angles = data(9 * 64, &device, DType::F32)?.reshape((9, 64))?;
            let cos = angles.cos()?.narrow(0, 2, 7)?;
            let sin = angles.sin()?.narrow(0, 2, 7)?;
            let cos = if strided {
                cos.transpose(0, 1)?.contiguous()?.transpose(0, 1)?
            } else {
                cos
            };
            let sin = if strided {
                sin.transpose(0, 1)?.contiguous()?.transpose(0, 1)?
            } else {
                sin
            };
            let input = values(&x.contiguous()?)?;
            let c = values(&cos.contiguous()?)?;
            let s = values(&sin.contiguous()?)?;
            let actual = values(&rope_native_bf16(&x, &cos, &sin)?)?;
            for (i, pair) in input.chunks_exact(2).enumerate() {
                let position = i % (7 * 64);
                let (real, imag) = (pair[0], pair[1]);
                let expected = [
                    real.mul_add(c[position], -(imag * s[position])),
                    real.mul_add(s[position], imag * c[position]),
                ];
                for side in 0..2 {
                    let expected = half::bf16::from_f32(expected[side]).to_f32();
                    assert_eq!(actual[2 * i + side].to_bits(), expected.to_bits());
                }
            }
            assert!(rope_native_bf16(&x.to_dtype(DType::F32)?, &cos, &sin).is_err());
            assert!(rope_native_bf16(&x, &cos.narrow(0, 0, 6)?, &sin).is_err());
            assert!(rope_native_bf16(&x, &cos.to_dtype(DType::BF16)?, &sin).is_err());
        }
        Ok(())
    }

    #[test]
    fn fused_norm_matches_two_pass_reference_including_bf16_rounding() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for dtype in [DType::F32, DType::BF16] {
            for width in [1, 31, 64, 128, 1536, 4096] {
                // Use nonzero storage offsets as well as odd row lengths.
                let x = data(8 * width, &device, dtype)?
                    .narrow(0, width, 6 * width)?
                    .reshape((2, 3, width))?;
                let scale = ((&x * 0.125)? + 1.0)?;
                let f = x.to_dtype(DType::F32)?;
                let mean = (f.sum_keepdim(D::Minus1)? / width as f64)?;
                let centered = f.broadcast_sub(&mean)?;
                let var = (centered.sqr()?.sum_keepdim(D::Minus1)? / width as f64)?;
                let expected = centered
                    .broadcast_div(&(&var + 1e-6)?.sqrt()?)?
                    .to_dtype(dtype)?
                    .mul(&scale)?;
                let got = norm_scale(&x, &scale, 1e-6)?;
                let mut max = 0_f32;
                for (a, b) in values(&got)?.into_iter().zip(values(&expected)?) {
                    assert!(a.is_finite());
                    max = max.max((a - b).abs());
                    assert!(
                        (a - b).abs() <= if dtype == DType::BF16 { 0.016 } else { 2e-5 },
                        "{dtype:?} width={width}: {a} != {b}"
                    );
                }
                eprintln!("norm {dtype:?} width={width} max_error={max}");
            }
        }
        Ok(())
    }

    #[test]
    fn fused_residual_and_rope_preserve_reference_values_and_offsets() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for dtype in [DType::F32, DType::BF16] {
            let x = data(2 * 3 * 7 * 128 + 128, &device, dtype)?
                .narrow(0, 128, 2 * 3 * 7 * 128)?
                .reshape((2, 3, 7, 128))?;
            let gate = (&x * 0.1)?.tanh()?;
            let update = (&x * 0.7)?;
            assert_eq!(
                values(&gated_residual(&x, &gate, &update)?)?,
                values(&(&x + (&gate * &update)?)?)?
            );
            let angles = data(9 * 64, &device, DType::F32)?.reshape((9, 64))?;
            let cos = angles.cos()?.narrow(0, 2, 7)?;
            let sin = angles.sin()?.narrow(0, 2, 7)?;
            let expected = super::super::transformer::apply_rope_reference(&x, &cos, &sin)?;
            assert_eq!(values(&rope(&x, &cos, &sin)?)?, values(&expected)?);
            assert!(rope(&x, &cos.narrow(0, 0, 6)?, &sin).is_err());
            assert!(norm_scale(&x, &gate, f32::NAN).is_err());
        }
        Ok(())
    }
}
