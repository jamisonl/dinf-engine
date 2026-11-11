//! Native top-six output combine. The shared add and BF16 boundary stay outside.
//!
//! For routes 0..6: weighted = F32(fma(down, weight, +0)), then
//! sum = F32(sum + weighted), starting at +0. Never a weighted FMA or tree.
//! Weights are already F32; the old f32 -> f64 affine argument -> f32 roundtrip
//! was exact. Each volatile intermediate preserves the old dispatch's F32 store.
//! Six retained rows total 122,880 bytes; no packed weights or extra fences.
use candle_core::{CpuStorage, CustomOp1, DType, Layout, Result, Shape, Tensor};

const WIDTH: usize = 5120;
const ROUTES: usize = 6;

/// Only contiguous [1, 5120] F32 outputs on the same Metal device are accepted.
pub(super) fn ordered(downs: &[Tensor; ROUTES], weights: &[f32; ROUTES]) -> Result<Tensor> {
    for down in downs {
        if !down.device().is_metal()
            || !down.device().same_device(downs[0].device())
            || down.dtype() != DType::F32
            || down.dims() != [1, WIDTH]
            || !down.is_contiguous()
        {
            candle_core::bail!(
                "Maya route combine requires six contiguous [1,5120] F32 tensors on one Metal device"
            );
        }
    }
    downs[0].apply_op1_no_bwd(&Ordered {
        tail: std::array::from_fn(|i| &downs[i + 1]),
        weights: *weights,
    })
}

struct Ordered<'a> {
    #[cfg_attr(
        not(target_os = "macos"),
        expect(
            dead_code,
            reason = "The CPU entry rejects this operation; Metal reads the retained route tensors."
        )
    )]
    tail: [&'a Tensor; ROUTES - 1],
    #[cfg_attr(
        not(target_os = "macos"),
        expect(
            dead_code,
            reason = "The CPU entry rejects this operation; Metal reads these route weights."
        )
    )]
    weights: [f32; ROUTES],
}

impl CustomOp1 for Ordered<'_> {
    fn name(&self) -> &'static str {
        "dinf-maya-ordered-route-combine"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("Maya route combine requires Metal");
    }

    #[cfg(target_os = "macos")]
    fn metal_fwd(
        &self,
        first: &candle_core::MetalStorage,
        layout: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        metal::forward(self, first, layout)
    }
}

#[cfg(target_os = "macos")]
mod metal {
    use super::{Ordered, ROUTES, WIDTH};
    use candle_core::backend::BackendStorage;
    use candle_core::{DType, Layout, MetalStorage, Result, Shape, Storage};
    use candle_metal_kernels::metal::ComputePipeline;
    use objc2_metal::{MTLCompileOptions, MTLMathFloatingPointFunctions, MTLMathMode, MTLSize};
    use std::collections::HashMap;
    use std::sync::{LazyLock, Mutex};

    // Match Candle affine/binary compile policy, including its exact env parser.
    // Volatile F32 intermediates and clang controls below restore the store
    // boundaries that existed between those separately compiled dispatches.
    fn pipeline(device: &candle_core::MetalDevice) -> Result<ComputePipeline> {
        static PIPELINES: LazyLock<
            Mutex<HashMap<candle_core::metal_backend::DeviceId, ComputePipeline>>,
        > = LazyLock::new(|| Mutex::new(HashMap::new()));
        let mut cache = PIPELINES.lock().map_err(|_| {
            candle_core::Error::Msg("Maya route combine pipeline lock poisoned".into())
        })?;
        if let Some(p) = cache.get(&device.id()) {
            return Ok(p.clone());
        }
        let options = MTLCompileOptions::new();
        let fast = std::env::var("CANDLE_METAL_ENABLE_FAST_MATH")
            .map(|v| matches!(v.as_str(), "true" | "t" | "yes" | "y" | "1"))
            .unwrap_or(true);
        options.setMathMode(if fast {
            MTLMathMode::Fast
        } else {
            MTLMathMode::Relaxed
        });
        options.setMathFloatingPointFunctions(if fast {
            MTLMathFloatingPointFunctions::Fast
        } else {
            MTLMathFloatingPointFunctions::Precise
        });
        let library = device
            .metal_device()
            .new_library_with_source(SOURCE, Some(&options))
            .map_err(candle_core::Error::wrap)?;
        let function = library
            .get_function("maya_ordered_route_combine", None)
            .map_err(candle_core::Error::wrap)?;
        let p = device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(candle_core::Error::wrap)?;
        cache.insert(device.id(), p.clone());
        Ok(p)
    }

    fn span(layout: &Layout, buffer_bytes: usize) -> Result<usize> {
        if !layout.is_contiguous() || layout.dims() != [1, WIDTH] {
            candle_core::bail!("invalid Maya route combine layout");
        }
        let offset = layout
            .start_offset()
            .checked_mul(4)
            .ok_or_else(|| candle_core::Error::Msg("Maya route combine offset overflow".into()))?;
        let end = offset
            .checked_add(WIDTH * 4)
            .ok_or_else(|| candle_core::Error::Msg("Maya route combine span overflow".into()))?;
        if end > buffer_bytes {
            candle_core::bail!("Maya route combine span outside buffer");
        }
        Ok(offset)
    }

    pub(super) fn forward(
        op: &Ordered<'_>,
        first: &MetalStorage,
        layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let guards: [_; ROUTES - 1] = std::array::from_fn(|i| op.tail[i].storage_and_layout());
        let mut inputs = [(first, layout); ROUTES];
        for (i, (storage, layout)) in guards.iter().enumerate() {
            let Storage::Metal(storage) = &**storage else {
                candle_core::bail!("Maya route combine requires Metal storage");
            };
            inputs[i + 1] = (storage, layout);
        }
        let device = first.device();
        let mut offsets = [0; ROUTES];
        // Validate every operand and span before allocating or obtaining an encoder.
        for (i, (storage, layout)) in inputs.iter().enumerate() {
            if storage.dtype() != DType::F32 || storage.device().id() != device.id() {
                candle_core::bail!("Maya route combine requires F32 on one Metal device");
            }
            offsets[i] = span(layout, storage.buffer().length())?;
        }
        let p = pipeline(device)?;
        let threads = p.max_total_threads_per_threadgroup().min(256);
        if threads == 0 {
            candle_core::bail!("Maya route combine has no Metal threads");
        }
        let output = device
            .new_buffer_builder()
            .with_size_for(WIDTH, DType::F32)
            .with_label("maya-ordered-route-combine")
            .build()?;
        let encoder = device.command_encoder()?;
        let encoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&p);
        for (i, (storage, _)) in inputs.iter().enumerate() {
            encoder.set_input_buffer(i, Some(storage.buffer()), offsets[i]);
        }
        encoder.set_bytes(6, &op.weights);
        encoder.set_output_buffer(7, Some(&output), 0);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: WIDTH.div_ceil(threads),
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
            MetalStorage::new(output, device.clone(), WIDTH, DType::F32),
            Shape::from((1, WIDTH)),
        ))
    }

    const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
kernel void maya_ordered_route_combine(
    device const float *d0 [[buffer(0)]],
    device const float *d1 [[buffer(1)]],
    device const float *d2 [[buffer(2)]],
    device const float *d3 [[buffer(3)]],
    device const float *d4 [[buffer(4)]],
    device const float *d5 [[buffer(5)]],
    constant float *weights [[buffer(6)]],
    device float *out [[buffer(7)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= 5120) return;
    volatile float sum = 0.0f;
    volatile float weighted = fma(d0[i], weights[0], 0.0f);
    sum = sum + weighted;
    weighted = fma(d1[i], weights[1], 0.0f);
    sum = sum + weighted;
    weighted = fma(d2[i], weights[2], 0.0f);
    sum = sum + weighted;
    weighted = fma(d3[i], weights[3], 0.0f);
    sum = sum + weighted;
    weighted = fma(d4[i], weights[4], 0.0f);
    sum = sum + weighted;
    weighted = fma(d5[i], weights[5], 0.0f);
    sum = sum + weighted;
    out[i] = sum;
}
"#;

    #[cfg(test)]
    #[test]
    fn route_combine_invalid_offsets_and_spans() {
        let layout = |offset| Layout::new(Shape::from((1, WIDTH)), vec![WIDTH, 1], offset);
        assert_eq!(span(&layout(3), (WIDTH + 3) * 4).unwrap(), 12);
        assert!(span(&layout(1), WIDTH * 4).is_err());
        assert!(span(&layout(usize::MAX), usize::MAX).is_err());
        assert!(span(&layout(usize::MAX / 4), usize::MAX).is_err());
        assert!(
            span(
                &Layout::new(Shape::from((1, WIDTH)), vec![WIDTH * 2, 2], 0),
                WIDTH * 8
            )
            .is_err()
        );
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use candle_core::Device;

    fn tensors(
        data: [Vec<f32>; ROUTES],
        device: &Device,
        offset: bool,
    ) -> Result<[Tensor; ROUTES]> {
        let rows = data
            .into_iter()
            .enumerate()
            .map(|(i, mut row)| {
                if offset {
                    let prefix = i + 1;
                    let mut backing = vec![f32::NAN; prefix];
                    backing.append(&mut row);
                    backing.push(f32::INFINITY);
                    Tensor::from_vec(backing, WIDTH + prefix + 1, device)?
                        .narrow(0, prefix, WIDTH)?
                        .reshape((1, WIDTH))
                } else {
                    Tensor::from_vec(row, (1, WIDTH), device)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        rows.try_into()
            .map_err(|_| candle_core::Error::Msg("route fixture count differs".into()))
    }

    // The unchanged production graph is the only arithmetic oracle.
    fn reference(downs: &[Tensor; ROUTES], weights: &[f32; ROUTES]) -> Result<Tensor> {
        let mut out = Tensor::zeros((1, WIDTH), DType::F32, downs[0].device())?;
        for (down, &weight) in downs.iter().zip(weights) {
            out = out.add(&down.affine(f64::from(weight), 0.0)?)?;
        }
        Ok(out)
    }

    fn bits(t: &Tensor) -> Result<Vec<u32>> {
        Ok(t.flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }

    fn exact(downs: &[Tensor; ROUTES], weights: &[f32; ROUTES]) -> Result<()> {
        let expected = bits(&reference(downs, weights)?)?;
        let actual = bits(&ordered(downs, weights)?)?;
        for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                a, b,
                "route combine bits differ at column {i}: actual={a:08x}, reference={b:08x}"
            );
        }
        Ok(())
    }

    #[test]
    fn route_combine_real_metal_exact_finite_and_offset_views() -> Result<()> {
        let device = Device::new_metal(0)?;
        let weights = [0.03125, 0.15625, 0.21875, 0.28125, 0.34375, 0.46875];
        let boundary = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            7.0,
            -7.0,
            20.0,
            -20.0,
            448.0,
            -448.0,
            70_000.0,
            -70_000.0,
            1.0039063,
            -1.0039063,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::from_bits(0x007fffff),
            -f32::from_bits(0x007fffff),
            f32::MAX,
            -f32::MAX,
            1.0e-30,
            -1.0e-30,
        ];
        for offset in [false, true] {
            let data = std::array::from_fn(|route| {
                (0..WIDTH)
                    .map(|i| {
                        if i < WIDTH / 2 {
                            boundary[(i + route * 7) % boundary.len()]
                        } else {
                            f32::from_bits(
                                0x3f000000 + ((i * 7919 + route * 104729) % 0x1000000) as u32,
                            ) * if (i + route) % 2 == 0 { 1.0 } else { -1.0 }
                        }
                    })
                    .collect()
            });
            exact(&tensors(data, &device, offset)?, &weights)?;
            let zeros = std::array::from_fn(|r| {
                (0..WIDTH)
                    .map(|i| if (i + r) % 2 == 0 { -0.0 } else { 0.0 })
                    .collect()
            });
            let zeros = tensors(zeros, &device, offset)?;
            exact(&zeros, &weights)?;
            exact(&zeros, &[0.0, -0.0, 0.125, 0.25, 0.5, 1.0])?;
        }
        eprintln!(
            "MAYA_ROUTE_COMBINE_METAL_EXACT: 5120-wide finite, subnormal, signed-zero, offset fixtures"
        );
        device.synchronize()
    }

    #[test]
    fn route_combine_real_metal_exact_order_and_cancellation() -> Result<()> {
        let device = Device::new_metal(0)?;
        let weights = [0.125, 0.25, 0.125, 0.25, 0.5, 0.25];
        let values = [134217728.0, 4.0, -134217728.0, 12.0, -2.0, 2.0];
        let downs = tensors(
            std::array::from_fn(|r| vec![values[r]; WIDTH]),
            &device,
            true,
        )?;
        exact(&downs, &weights)?;
        let mut reordered = downs.clone();
        reordered.swap(1, 2);
        let mut reordered_weights = weights;
        reordered_weights.swap(1, 2);
        let original = bits(&reference(&downs, &weights)?)?;
        let wrong_order = bits(&reference(&reordered, &reordered_weights)?)?;
        assert!(
            original.iter().zip(&wrong_order).all(|(a, b)| a != b),
            "fixture must distinguish route order at every column"
        );
        exact(&reordered, &reordered_weights)?;
        device.synchronize()
    }

    #[test]
    fn route_combine_real_metal_exact_nonfinite() -> Result<()> {
        let device = Device::new_metal(0)?;
        let cases = [
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(0xffc01234),
            0.0,
            -0.0,
            1.0,
            -1.0,
        ];
        let downs = tensors(
            std::array::from_fn(|r| {
                (0..WIDTH)
                    .map(|i| cases[(i / 8usize.pow(r as u32) + r) % cases.len()])
                    .collect()
            }),
            &device,
            true,
        )?;
        exact(&downs, &[0.125, 0.25, 0.125, 0.25, 0.5, 0.25])?;
        exact(&downs, &[0.0, -0.0, 0.125, 0.25, 0.5, 1.0])?;
        device.synchronize()
    }

    #[test]
    fn route_combine_rejects_invalid_shape_dtype_device_and_stride() -> Result<()> {
        let device = Device::new_metal(0)?;
        let valid = Tensor::zeros((1, WIDTH), DType::F32, &device)?;
        let mut downs = std::array::from_fn(|_| valid.clone());
        for invalid in [
            Tensor::zeros(WIDTH, DType::F32, &device)?,
            Tensor::zeros((1, WIDTH - 1), DType::F32, &device)?,
            Tensor::zeros((1, WIDTH), DType::F16, &device)?,
            Tensor::zeros((1, WIDTH), DType::F32, &Device::Cpu)?,
            Tensor::zeros((WIDTH, 2), DType::F32, &device)?
                .narrow(1, 0, 1)?
                .transpose(0, 1)?,
            Tensor::zeros((1, WIDTH), DType::F32, &Device::new_metal(0)?)?,
        ] {
            downs[4] = invalid;
            assert!(ordered(&downs, &[0.25; ROUTES]).is_err());
        }
        device.synchronize()
    }
}
