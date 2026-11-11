//! Metal 4 BF16 matrix products with F32 accumulation and BF16 output.
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{CpuStorage, CustomOp2, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::{MTLCompileOptions, MTLDevice, MTLGPUFamily, MTLLanguageVersion, MTLSize};

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("DINF_DIFFUSION_DENSE").as_deref() != Ok("0"))
}

fn pipeline(device: &candle_core::MetalDevice) -> Result<Option<ComputePipeline>> {
    static CACHE: OnceLock<Mutex<HashMap<DeviceId, Option<ComputePipeline>>>> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| candle_core::Error::Msg("diffusion dense kernel cache poisoned".into()))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }
    // This path is measured on Apple10 (M5). Older GPUs keep Candle's
    // existing kernels even if they expose Metal 4 tensor operations.
    if !supports_device(device) {
        cache.insert(device.id(), None);
        return Ok(None);
    }
    let compiled = (|| -> Result<ComputePipeline> {
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version4_0);
        #[allow(deprecated)]
        options.setFastMathEnabled(false);
        let library = device
            .metal_device()
            .new_library_with_source(include_str!("dense.metal"), Some(&options))
            .map_err(candle_core::Error::wrap)?;
        let function = library
            .get_function("diffusion_dense_bf16", None)
            .map_err(candle_core::Error::wrap)?;
        device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(candle_core::Error::wrap)
    })();
    let pipeline = match compiled {
        Ok(pipeline) if pipeline.max_total_threads_per_threadgroup() >= 128 => Some(pipeline),
        Ok(_) => None,
        Err(error) => {
            tracing::warn!(%error, "BF16 Metal 4 diffusion matmul unavailable; using Candle");
            None
        }
    };
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

fn supports_device(device: &candle_core::MetalDevice) -> bool {
    let metal = device.metal_device().as_ref();
    metal.supportsFamily(MTLGPUFamily::Metal4) && metal.supportsFamily(MTLGPUFamily::Apple10)
}

pub(super) fn try_linear(input: &Tensor, weight: &Tensor) -> Result<Option<Tensor>> {
    if !enabled()
        || !input.device().is_metal()
        || input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || !input.device().same_device(weight.device())
        || input.rank() < 2
        || weight.rank() != 2
    {
        return Ok(None);
    }
    let (columns, inner) = weight.dims2()?;
    if inner == 0
        || columns == 0
        || !inner.is_multiple_of(32)
        || input.dim(candle_core::D::Minus1)? != inner
        || input.elem_count() / inner < 16
    {
        return Ok(None);
    }
    let Some(pipeline) = pipeline(input.device().as_metal_device()?)? else {
        return Ok(None);
    };
    Ok(Some(input.contiguous()?.apply_op2_no_bwd(
        &weight.contiguous()?,
        &Dense { pipeline },
    )?))
}

struct Dense {
    pipeline: ComputePipeline,
}

fn checked_offset(storage: &MetalStorage, layout: &Layout) -> Result<usize> {
    let count = layout.shape().elem_count();
    let end = layout
        .start_offset()
        .checked_add(count)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| candle_core::Error::Msg("diffusion dense buffer span overflow".into()))?;
    if storage.dtype() != DType::BF16
        || !layout.is_contiguous()
        || count == 0
        || count > i32::MAX as usize
        || end > storage.buffer().length()
    {
        candle_core::bail!("invalid BF16 diffusion dense buffer");
    }
    Ok(layout.start_offset() * 2)
}

impl CustomOp2 for Dense {
    fn name(&self) -> &'static str {
        "diffusion-dense-bf16"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("diffusion dense kernel requires Metal")
    }
    fn metal_fwd(
        &self,
        input: &MetalStorage,
        il: &Layout,
        weight: &MetalStorage,
        wl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let io = checked_offset(input, il)?;
        let wo = checked_offset(weight, wl)?;
        let (columns, inner) = wl.shape().dims2()?;
        if input.device().id() != weight.device().id()
            || inner == 0
            || !inner.is_multiple_of(32)
            || il.dims().last() != Some(&inner)
        {
            candle_core::bail!("invalid diffusion dense shapes or devices");
        }
        let rows = il.shape().elem_count() / inner;
        let count = rows
            .checked_mul(columns)
            .filter(|n| *n <= i32::MAX as usize)
            .ok_or_else(|| candle_core::Error::Msg("diffusion dense output too large".into()))?;
        let device = input.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(count, DType::BF16)
            .with_label(self.name())
            .build()?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.pipeline);
        let encoder: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder.set_input_buffer(0, Some(input.buffer()), io);
        encoder.set_input_buffer(1, Some(weight.buffer()), wo);
        encoder.set_output_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, &(rows as u32));
        encoder.set_bytes(4, &(columns as u32));
        encoder.set_bytes(5, &(inner as u32));
        encoder.dispatch_thread_groups(
            MTLSize {
                width: rows.div_ceil(128),
                height: columns.div_ceil(64),
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        let mut shape = il.dims().to_vec();
        *shape.last_mut().unwrap() = columns;
        Ok((
            MetalStorage::new(output, device.clone(), count, DType::BF16),
            Shape::from(shape),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{D, Device};
    fn values(t: &Tensor) -> Result<Vec<f32>> {
        t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
    }
    fn data(rows: usize, cols: usize, phase: f32, device: &Device) -> Result<Tensor> {
        Tensor::from_vec(
            (0..rows * cols)
                .map(|i| (i as f32 * 0.137 + phase).sin() * 0.2)
                .collect::<Vec<_>>(),
            (rows, cols),
            device,
        )?
        .to_dtype(DType::BF16)
    }
    #[test]
    fn dense_bf16_matches_f32_reference_with_tails_and_offsets() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        if !supports_device(device.as_metal_device()?) {
            return Ok(());
        }
        let pipeline =
            pipeline(device.as_metal_device()?)?.expect("Metal 4 BF16 kernel must compile");
        for (m, n, k) in [
            (16, 31, 32),
            (65, 67, 96),
            (128, 144, 256),
            (33, 512, 4096),
            (41, 511, 12288),
        ] {
            let x = data(m + 1, k, 0.1, &device)?.narrow(0, 1, m)?;
            let w = data(n + 1, k, 0.9, &device)?.narrow(0, 1, n)?;
            let expected = x
                .to_dtype(DType::F32)?
                .matmul(&w.to_dtype(DType::F32)?.t()?)?;
            let out = x.apply_op2_no_bwd(
                &w,
                &Dense {
                    pipeline: pipeline.clone(),
                },
            )?;
            assert_eq!(out.dtype(), DType::BF16);
            for (a, b) in values(&out)?.into_iter().zip(values(&expected)?) {
                assert!(
                    a.is_finite() && (a - b).abs() <= 0.004 * b.abs() + 0.001,
                    "{m}x{n}x{k}: {a} != {b}"
                );
            }
        }
        // Values outside FP16's range must remain finite, including batched input.
        let x = Tensor::full(100000_f32, (2, 16, 32), &device)?.to_dtype(DType::BF16)?;
        let w = Tensor::full(0.001_f32, (17, 32), &device)?.to_dtype(DType::BF16)?;
        let out = x.apply_op2_no_bwd(&w, &Dense { pipeline })?;
        assert_eq!(out.dims(), [2, 16, 17]);
        assert!(values(&out)?.iter().all(|x| x.is_finite() && *x > 3000.));
        assert!(try_linear(&x.narrow(D::Minus1, 0, 31)?, &w)?.is_none());
        Ok(())
    }

    #[test]
    fn dense_dispatch_preserves_fallbacks_and_strided_inputs() -> Result<()> {
        let cpu = Device::Cpu;
        let cpu_x = data(16, 64, 0.1, &cpu)?;
        let cpu_w = data(32, 64, 0.9, &cpu)?;
        assert!(try_linear(&cpu_x, &cpu_w)?.is_none());
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let x = cpu_x.to_device(&device)?;
        let w = cpu_w.to_device(&device)?;
        assert!(try_linear(&x.to_dtype(DType::F32)?, &w.to_dtype(DType::F32)?)?.is_none());
        assert!(try_linear(&x.narrow(0, 0, 1)?, &w)?.is_none());
        assert!(try_linear(&x, &cpu_w)?.is_none());
        if !supports_device(device.as_metal_device()?) || !enabled() {
            return Ok(());
        }
        // Transposed storage requires a copy; it must not be treated as packed.
        let x = data(64, 35, 0.1, &device)?.t()?;
        let w = data(64, 71, 0.9, &device)?.t()?;
        let out = try_linear(&x, &w)?.expect("eligible BF16 projection");
        let expected = x
            .to_dtype(DType::F32)?
            .matmul(&w.to_dtype(DType::F32)?.t()?)?;
        for (a, b) in values(&out)?.into_iter().zip(values(&expected)?) {
            assert!(a.is_finite() && (a - b).abs() <= 0.004 * b.abs() + 0.001);
        }
        Ok(())
    }
    #[test]
    #[ignore = "isolated GPU performance probe"]
    fn dense_bf16_speed() -> Result<()> {
        let device = Device::new_metal(0)?;
        let pipeline = pipeline(device.as_metal_device()?)?.expect("Metal 4 BF16 kernel");
        for (m, n, k) in [
            (1024, 4096, 4096),
            (4096, 4096, 4096),
            (4096, 12288, 4096),
            (4096, 4096, 12288),
        ] {
            let x = data(m, k, 0.1, &device)?;
            let w = data(n, k, 0.9, &device)?;
            for fused in [false, true, false, true] {
                device.synchronize()?;
                let start = std::time::Instant::now();
                for _ in 0..5 {
                    let _out = if fused {
                        x.apply_op2_no_bwd(
                            &w,
                            &Dense {
                                pipeline: pipeline.clone(),
                            },
                        )?
                    } else {
                        x.matmul(&w.t()?)?
                    };
                    device.synchronize()?;
                }
                eprintln!(
                    "dense {m}x{n}x{k} fused={fused} ms={:.3}",
                    start.elapsed().as_secs_f64() * 200.
                );
            }
        }
        Ok(())
    }
}
