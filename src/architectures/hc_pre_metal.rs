//! Exact input gates and four-stream reduction in one Metal dispatch.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{
    CpuStorage, CustomOp3, DType, Layout, MetalStorage, Result, Shape, Storage, Tensor,
};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::{MTLCompileOptions, MTLSize};

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void dinf_hc_pre4(
    device const float *input [[buffer(0)]],
    device const float *mixes [[buffer(1)]],
    device const float *base [[buffer(2)]],
    device const float *scale [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant uint &hidden [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float weights[4];
  const uint row = group.y;
  if (tid < 4) {
    // Match the separate multiply, add, sigmoid, then affine kernels.
    volatile float product = mixes[row * 24 + tid] * scale[0];
    const float value = product + base[tid];
    const float gate = fast::divide(1.0f, 1.0f + fast::exp(-value));
    weights[tid] = fma(gate, 1.0f, eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint col = group.x * 256 + tid;
  if (col >= hidden) return;
  const uint start = row * 4 * hidden + col;
  // Round each F32 product before adding; do not contract to FMA.
  volatile float a = weights[0] * input[start];
  volatile float b = weights[1] * input[start + hidden];
  volatile float c = weights[2] * input[start + 2 * hidden];
  volatile float d = weights[3] * input[start + 3 * hidden];
  // Candle starts two lanes at zero and sums streams 0+2 and 1+3.
  output[row * hidden + col] = ((0.0f + a) + c) + ((0.0f + b) + d);
}
"#;

fn fast_math(setting: Option<&str>) -> bool {
    setting.is_none_or(|v| matches!(v, "true" | "t" | "yes" | "y" | "1"))
}

fn pipeline(device: &candle_core::MetalDevice) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<DeviceId, ComputePipeline>>> = OnceLock::new();
    let mut cache = PIPELINES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| candle_core::Error::Msg("HC pre pipeline cache poisoned".into()))?;
    if let Some(pipeline) = cache.get(&device.id()) {
        return Ok(pipeline.clone());
    }
    let options = MTLCompileOptions::new();
    #[allow(deprecated)]
    options.setFastMathEnabled(false);
    let library = device
        .metal_device()
        .new_library_with_source(SOURCE, Some(&options))
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function("dinf_hc_pre4", None)
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

struct Pre<'a> {
    scale: &'a Tensor,
    eps: f32,
}

impl CustomOp3 for Pre<'_> {
    fn name(&self) -> &'static str {
        "dinf-hc-pre4"
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
        candle_core::bail!("HC fused pre mix requires Metal")
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        il: &Layout,
        mixes: &MetalStorage,
        ml: &Layout,
        base: &MetalStorage,
        bl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (guard, sl) = self.scale.storage_and_layout();
        let Storage::Metal(scale) = &*guard else {
            candle_core::bail!("HC scale requires Metal");
        };
        let device = input.device();
        let (batch, rows, _, hidden) = il.shape().dims4()?;
        let mut offsets = [0; 4];
        for (i, (storage, layout)) in [(input, il), (mixes, ml), (base, bl), (scale, sl)]
            .into_iter()
            .enumerate()
        {
            let _ = u32::try_from(layout.shape().elem_count()).map_err(candle_core::Error::wrap)?;
            if storage.dtype() != DType::F32
                || !layout.is_contiguous()
                || storage.device().id() != device.id()
            {
                candle_core::bail!("HC pre inputs must be contiguous f32 on one Metal device");
            }
            offsets[i] = layout
                .start_offset()
                .checked_mul(4)
                .ok_or_else(|| candle_core::Error::Msg("HC pre offset overflow".into()))?;
            let end = layout
                .start_offset()
                .checked_add(layout.shape().elem_count())
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| candle_core::Error::Msg("HC pre span overflow".into()))?;
            if end > storage.buffer().length() {
                candle_core::bail!("HC pre input exceeds buffer");
            }
        }
        let count = il.shape().elem_count() / 4;
        let output = device
            .new_buffer_builder()
            .with_size_for(count, DType::F32)
            .with_label("dinf-hc-pre4")
            .build()?;
        let pipeline = pipeline(device)?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        for (i, storage) in [input, mixes, base, scale].into_iter().enumerate() {
            encoder.set_input_buffer(i, Some(storage.buffer()), offsets[i]);
        }
        encoder.set_output_buffer(4, Some(&output), 0);
        encoder.set_bytes(5, &(hidden as u32));
        encoder.set_bytes(6, &self.eps);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: hidden.div_ceil(256),
                height: batch * rows,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), count, DType::F32),
            Shape::from((batch, rows, hidden)),
        ))
    }
}

pub(super) fn forward(
    input: &Tensor,
    mixes: &Tensor,
    scale: &Tensor,
    base: &Tensor,
    eps: f64,
) -> anyhow::Result<Tensor> {
    let (batch, rows, streams, hidden) = input.dims4()?;
    anyhow::ensure!(
        batch > 0
            && rows > 0
            && streams == 4
            && hidden > 0
            && mixes.dims() == [batch, rows, 24]
            && scale.dims() == [3]
            && base.dims() == [24],
        "invalid HC pre input shapes"
    );
    anyhow::ensure!(
        eps.is_finite() && eps >= 0. && (eps as f32).is_finite(),
        "invalid HC pre epsilon"
    );
    anyhow::ensure!(
        [input, mixes, scale, base]
            .into_iter()
            .all(|t| t.dtype() == DType::F32
                && t.device().is_metal()
                && t.device().same_device(input.device())),
        "HC pre inputs must be f32 on one Metal device"
    );
    static FAST_MATH: OnceLock<bool> = OnceLock::new();
    if !*FAST_MATH.get_or_init(|| {
        fast_math(
            std::env::var("CANDLE_METAL_ENABLE_FAST_MATH")
                .ok()
                .as_deref(),
        )
    }) {
        // The precise-math sigmoid rounds differently from fast::exp/divide.
        return super::hc_pre::reference(input, mixes, scale, base, eps);
    }
    let scale = scale.contiguous()?;
    Ok(input.contiguous()?.apply_op3_no_bwd(
        &mixes.contiguous()?,
        &base.contiguous()?,
        &Pre {
            scale: &scale,
            eps: eps as f32,
        },
    )?)
}

#[cfg(test)]
mod tests {
    use super::super::hc_pre::reference;
    use super::*;

    fn bits(t: &Tensor) -> anyhow::Result<Vec<u32>> {
        Ok(t.flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }

    #[test]
    fn pre_mix_matches_candle_bits() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for hidden in [1, 2, 7, 17, 64, 257, 2048, 4096] {
            for rows in [1, 3, 32, 129] {
                let make = |n: usize, salt: usize| -> Result<Tensor> {
                    Tensor::from_vec(
                        (0..n)
                            .map(|i| {
                                (((i * 37 + salt * 19) % 101) as f32 - 50.) * 1.013
                                    / (1 + i % 7) as f32
                            })
                            .collect::<Vec<_>>(),
                        n,
                        &device,
                    )
                };
                let x = make((rows + 2) * 4 * hidden, 5)?
                    .reshape((1, rows + 2, 4, hidden))?
                    .narrow(1, 1, rows)?;
                let m = make((rows + 2) * 24, 7)?
                    .reshape((1, rows + 2, 24))?
                    .narrow(1, 1, rows)?;
                let s = make(5, 11)?.narrow(0, 1, 3)?;
                let b = make(26, 13)?.narrow(0, 1, 24)?;
                for eps in [0., 1e-6, 0.001] {
                    assert_eq!(
                        bits(&forward(&x, &m, &s, &b, eps)?)?,
                        bits(&reference(&x, &m, &s, &b, eps)?)?,
                        "hidden={hidden} rows={rows} eps={eps}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn pre_mix_strides_batches_and_rounding() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for mode in 0..5 {
            let make = |n: usize, salt: u32| -> Result<Tensor> {
                let mut state = salt;
                Tensor::from_vec(
                    (0..n)
                        .map(|i| {
                            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                            match mode {
                                0 => {
                                    f32::from_bits((state & 0x807fffff) | ((87 + state % 81) << 23))
                                }
                                1 => {
                                    if i % 2 == 0 {
                                        0.
                                    } else {
                                        -0.
                                    }
                                }
                                2 => f32::from_bits(state & 0x807fffff),
                                3 => {
                                    if salt == 5 {
                                        -0.
                                    } else {
                                        1.
                                    }
                                }
                                _ => ((state >> 8) as f32 / 16777216. - 0.5) * 2.,
                            }
                        })
                        .collect::<Vec<_>>(),
                    n,
                    &device,
                )
            };
            let x = make(2 * 3 * 4 * 17, 5)?
                .reshape((2, 3, 17, 4))?
                .transpose(2, 3)?;
            let m = make(2 * 3 * 24, 7)?.reshape((3, 2, 24))?.transpose(0, 1)?;
            let s = make(3, 11)?;
            let b = make(24, 13)?;
            let before = [&x, &m, &s, &b]
                .into_iter()
                .map(bits)
                .collect::<anyhow::Result<Vec<_>>>()?;
            assert_eq!(
                bits(&forward(&x, &m, &s, &b, 1e-6)?)?,
                bits(&reference(&x, &m, &s, &b, 1e-6)?)?,
                "mode={mode}"
            );
            let after = [&x, &m, &s, &b]
                .into_iter()
                .map(bits)
                .collect::<anyhow::Result<Vec<_>>>()?;
            assert_eq!(before, after);
        }
        Ok(())
    }

    #[test]
    fn pre_gates_match_over_sigmoid_range() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let rows = 4097;
        let x = Tensor::from_vec(
            (0..rows * 16)
                .map(|i| if (i / 4) % 4 == i % 4 { 1f32 } else { 0. })
                .collect::<Vec<_>>(),
            (1, rows, 4, 4),
            &device,
        )?;
        let m = Tensor::from_vec(
            (0..rows * 24)
                .map(|i| (i / 24) as f32 * 0.0625 - 128. + (i % 24) as f32 * 0.0123)
                .collect::<Vec<_>>(),
            (1, rows, 24),
            &device,
        )?;
        let b = Tensor::from_vec(
            (0..24).map(|i| i as f32 * 0.13 - 1.7).collect::<Vec<_>>(),
            24,
            &device,
        )?;
        for scale in [-0.73f32, 0., 1., 0.37] {
            let s = Tensor::new(&[scale, 0., 0.], &device)?;
            assert_eq!(
                bits(&forward(&x, &m, &s, &b, 1e-6)?)?,
                bits(&reference(&x, &m, &s, &b, 1e-6)?)?,
                "scale={scale}"
            );
        }
        Ok(())
    }

    #[test]
    fn pre_mix_rejects_bad_inputs_and_keeps_fallback() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let x = Tensor::ones((1, 3, 4, 8), DType::F32, &device)?;
        let m = Tensor::ones((1, 3, 24), DType::F32, &device)?;
        let s = Tensor::ones(3, DType::F32, &device)?;
        let b = Tensor::ones(24, DType::F32, &device)?;
        assert!(forward(&x.squeeze(0)?, &m, &s, &b, 1e-6).is_err());
        assert!(forward(&x.narrow(1, 0, 0)?, &m.narrow(1, 0, 0)?, &s, &b, 1e-6).is_err());
        assert!(forward(&x, &m.narrow(2, 0, 23)?, &s, &b, 1e-6).is_err());
        assert!(forward(&x, &m, &s.to_device(&candle_core::Device::Cpu)?, &b, 1e-6).is_err());
        assert!(forward(&x, &m, &s.to_dtype(DType::F16)?, &b, 1e-6).is_err());
        for eps in [f64::NAN, f64::INFINITY, f64::MAX, -1.] {
            assert!(forward(&x, &m, &s, &b, eps).is_err());
        }
        let (xh, mh, sh, bh) = (
            x.to_dtype(DType::F16)?,
            m.to_dtype(DType::F16)?,
            s.to_dtype(DType::F16)?,
            b.to_dtype(DType::F16)?,
        );
        assert_eq!(
            bits(
                &super::super::hc_pre::forward(&xh, &mh, &sh, &bh, 1e-6, true)?
                    .to_dtype(DType::F32)?
            )?,
            bits(&reference(&xh, &mh, &sh, &bh, 1e-6)?.to_dtype(DType::F32)?)?
        );
        let cpu = candle_core::Device::Cpu;
        let (x, m, s, b) = (
            x.to_device(&cpu)?,
            m.to_device(&cpu)?,
            s.to_device(&cpu)?,
            b.to_device(&cpu)?,
        );
        assert_eq!(
            bits(&super::super::hc_pre::forward(&x, &m, &s, &b, 1e-6, true)?)?,
            bits(&reference(&x, &m, &s, &b, 1e-6)?)?
        );
        Ok(())
    }

    #[test]
    fn candle_math_setting() {
        assert!(fast_math(None));
        for value in ["true", "t", "yes", "y", "1"] {
            assert!(fast_math(Some(value)));
        }
        for value in ["false", "0", "", "TRUE", "invalid"] {
            assert!(!fast_math(Some(value)));
        }
    }
}
