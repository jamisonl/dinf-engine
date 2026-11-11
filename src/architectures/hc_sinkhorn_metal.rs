//! Shared post-softmax 4x4 normalization. One thread owns one whole matrix.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{CpuStorage, CustomOp1, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::{MTLCompileOptions, MTLSize};

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void dinf_hc_sinkhorn4(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant uint &iterations [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  float m[16];
  for (uint i = 0; i < 16; ++i) m[i] = input[16 * gid + i] + eps;
  for (uint step = 0; step < iterations; ++step) {
    if (step > 0) {
      for (uint r = 0; r < 4; ++r) {
        // Candle's four-value sum uses two lanes: 0+2, 1+3, then the sum.
        const float sum = (m[4*r] + m[4*r+2]) + (m[4*r+1] + m[4*r+3]);
        const float denominator = sum + eps;
        for (uint c = 0; c < 4; ++c) m[4*r+c] = fast::divide(m[4*r+c], denominator);
      }
    }
    for (uint c = 0; c < 4; ++c) {
      const float sum = (m[c] + m[8+c]) + (m[4+c] + m[12+c]);
      const float denominator = sum + eps;
      for (uint r = 0; r < 4; ++r) m[4*r+c] = fast::divide(m[4*r+c], denominator);
    }
  }
  for (uint i = 0; i < 16; ++i) output[16 * gid + i] = m[i];
}
"#;

// Match Candle's setting, which also takes effect when its shaders compile.
fn fast_math(setting: Option<&str>) -> bool {
    setting.is_none_or(|v| matches!(v, "true" | "t" | "yes" | "y" | "1"))
}

fn pipeline(device: &candle_core::MetalDevice) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<DeviceId, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| candle_core::Error::Msg("HC Sinkhorn pipeline cache poisoned".into()))?;
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
        .get_function("dinf_hc_sinkhorn4", None)
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

struct Sinkhorn {
    iterations: u32,
    eps: f32,
}

impl CustomOp1 for Sinkhorn {
    fn name(&self) -> &'static str {
        "dinf-hc-sinkhorn4"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("HC fused Sinkhorn requires Metal")
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        if storage.dtype() != DType::F32
            || !layout.is_contiguous()
            || !layout.dims().ends_with(&[4, 4])
        {
            candle_core::bail!("HC fused Sinkhorn requires contiguous f32 4x4 matrices")
        }
        let elements = layout.shape().elem_count();
        let elements_u32 = u32::try_from(elements).map_err(candle_core::Error::wrap)?;
        if elements == 0 {
            candle_core::bail!("HC fused Sinkhorn matrices are empty");
        }
        let offset = layout
            .start_offset()
            .checked_mul(4)
            .ok_or_else(|| candle_core::Error::Msg("HC Sinkhorn offset overflow".into()))?;
        let end = layout
            .start_offset()
            .checked_add(elements)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| candle_core::Error::Msg("HC Sinkhorn span overflow".into()))?;
        if end > storage.buffer().length() {
            candle_core::bail!("HC Sinkhorn input exceeds buffer");
        }
        let device = storage.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-hc-sinkhorn4")
            .build()?;
        let pipeline = pipeline(device)?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder.set_input_buffer(0, Some(storage.buffer()), offset);
        encoder.set_output_buffer(1, Some(&output), 0);
        encoder.set_bytes(2, &(elements_u32 / 16));
        encoder.set_bytes(3, &self.iterations);
        encoder.set_bytes(4, &self.eps);
        encoder.dispatch_threads(
            MTLSize {
                width: elements / 16,
                height: 1,
                depth: 1,
            },
            candle_metal_kernels::utils::get_block_dims((elements / 16).min(128), 1, 1),
        );
        Ok((
            MetalStorage::new(output, device.clone(), elements, DType::F32),
            layout.shape().clone(),
        ))
    }
}

pub(super) fn forward(matrix: &Tensor, iterations: usize, eps: f64) -> anyhow::Result<Tensor> {
    anyhow::ensure!(
        (1..=128).contains(&iterations)
            && eps.is_finite()
            && (eps as f32) > 0.
            && (eps as f32).is_finite(),
        "invalid HC Sinkhorn parameters"
    );
    anyhow::ensure!(
        matrix.device().is_metal()
            && matrix.dtype() == DType::F32
            && matrix.rank() == 4
            && matrix.dims().ends_with(&[4, 4])
            && matrix.elem_count() > 0,
        "HC fused Sinkhorn requires nonempty Metal f32 [batch, rows, 4, 4] matrices"
    );
    static FAST_MATH: OnceLock<bool> = OnceLock::new();
    if !*FAST_MATH.get_or_init(|| {
        fast_math(
            std::env::var("CANDLE_METAL_ENABLE_FAST_MATH")
                .ok()
                .as_deref(),
        )
    }) {
        // Candle's precise-math division does not round like fast::divide.
        // Keep the original graph for that diagnostic setting.
        return super::hc_sinkhorn::finish(matrix.clone(), iterations, eps);
    }
    Ok(matrix.contiguous()?.apply_op1_no_bwd(&Sinkhorn {
        iterations: iterations as u32,
        eps: eps as f32,
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(probabilities: &Tensor, iterations: usize, eps: f64) -> anyhow::Result<Tensor> {
        let mut expected = (probabilities + eps)?;
        for i in 0..iterations {
            if i > 0 {
                let sum = (expected
                    .sum(candle_core::D::Minus1)?
                    .unsqueeze(candle_core::D::Minus1)?
                    + eps)?;
                expected = expected.broadcast_div(&sum)?;
            }
            let sum = (expected.sum(2)?.unsqueeze(2)? + eps)?;
            expected = expected.broadcast_div(&sum)?;
        }
        Ok(expected)
    }

    fn bits(tensor: &Tensor) -> anyhow::Result<Vec<u32>> {
        Ok(tensor
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
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

    #[test]
    fn fused_sinkhorn_matches_candle_bits() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for rows in [1, 2, 3, 8, 32, 129] {
            let logits = Tensor::from_vec(
                (0..(rows + 2) * 16)
                    .map(|i| ((i * 17 % 101) as f32 - 50.) / 7.3)
                    .collect::<Vec<_>>(),
                (1, rows + 2, 4, 4),
                &device,
            )?;
            let probabilities = candle_nn::ops::softmax_last_dim(&logits)?.narrow(1, 1, rows)?;
            for iterations in [1, 2, 8, 20, 32, 128] {
                for eps in [1e-6, 0.001] {
                    let expected = reference(&probabilities, iterations, eps)?;
                    let actual = forward(&probabilities, iterations, eps)?;
                    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
                    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "rows={rows} iterations={iterations} eps={eps}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn strided_extreme_inputs_are_exact_and_unchanged() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for scale in [0., 1., 1000., 1e20] {
            let logits = Tensor::from_vec(
                (0..48)
                    .map(|i| ((i * 7 % 13) as f32 - 6.) * scale)
                    .collect::<Vec<_>>(),
                (1, 3, 4, 4),
                &device,
            )?;
            let probabilities = candle_nn::ops::softmax_last_dim(&logits)?;
            let before = bits(&probabilities)?;
            for input in [probabilities.clone(), probabilities.transpose(2, 3)?] {
                for eps in [1e-6, 0.1, 1.] {
                    assert_eq!(
                        bits(&forward(&input, 20, eps)?)?,
                        bits(&reference(&input, 20, eps)?)?
                    );
                }
            }
            assert_eq!(bits(&probabilities)?, before);
        }
        Ok(())
    }

    #[test]
    fn rejects_invalid_inputs() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let valid = Tensor::ones((1, 2, 4, 4), DType::F32, &device)?;
        for iterations in [0, 129, usize::MAX] {
            assert!(forward(&valid, iterations, 1e-6).is_err());
        }
        for eps in [
            -1.,
            0.,
            f64::NAN,
            f64::INFINITY,
            f64::MIN_POSITIVE,
            f64::MAX,
        ] {
            assert!(forward(&valid, 20, eps).is_err());
        }
        for invalid in [
            valid.to_dtype(DType::F16)?,
            valid.reshape((1, 2, 2, 8))?,
            Tensor::zeros((1, 0, 4, 4), DType::F32, &device)?,
            valid.to_device(&candle_core::Device::Cpu)?,
        ] {
            assert!(forward(&invalid, 20, 1e-6).is_err());
        }
        Ok(())
    }
}
