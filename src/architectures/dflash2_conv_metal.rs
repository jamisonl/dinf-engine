use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{CpuStorage, CustomOp3, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;

const DFLASH_CONV_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void dinf_dflash_dynamic_conv_f32(
    device const float *input [[buffer(0)]],
    device const float *coefficients [[buffer(1)]],
    device const float *base [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &elements [[buffer(4)]],
    constant uint &seq_len [[buffer(5)]],
    constant uint &hidden_size [[buffer(6)]],
    constant uint &groups [[buffer(7)]],
    constant uint &kernel_size [[buffer(8)]],
    constant uint &group_size [[buffer(9)]],
    constant uint &side [[buffer(10)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= elements) {
    return;
  }

  const uint hidden = index % hidden_size;
  const uint token_row = index / hidden_size;
  const uint position = token_row % seq_len;
  const uint batch = token_row / seq_len;
  const uint group = hidden / group_size;
  float value = 0.0f;
  for (uint tap = 0; tap < kernel_size && tap <= position; ++tap) {
    const uint coefficient_index =
        ((((batch * seq_len + position) * 2 + side) * kernel_size + tap) *
            groups) + group;
    const uint base_index = (side * kernel_size + tap) * hidden_size + hidden;
    const uint input_index =
        ((batch * seq_len + position - tap) * hidden_size) + hidden;
    value += (coefficients[coefficient_index] + base[base_index]) * input[input_index];
  }
  output[index] = value;
}
"#;

static PIPELINES: OnceLock<Mutex<HashMap<DeviceId, ComputePipeline>>> = OnceLock::new();

fn pipeline(device: &candle_core::MetalDevice) -> Result<ComputePipeline> {
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(pipeline) = pipelines.lock().unwrap().get(&device.id()) {
        return Ok(pipeline.clone());
    }
    let library = device
        .metal_device()
        .new_library_with_source(DFLASH_CONV_MSL, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function("dinf_dflash_dynamic_conv_f32", None)
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    pipelines
        .lock()
        .unwrap()
        .insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

struct DFlashDynamicConv {
    side: usize,
    kernel_size: usize,
    group_size: usize,
}

impl CustomOp3 for DFlashDynamicConv {
    fn name(&self) -> &'static str {
        "dinf-metal-dflash-dynamic-conv"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _coefficients: &CpuStorage,
        _coefficients_layout: &Layout,
        _base: &CpuStorage,
        _base_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("DFlash dynamic convolution is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        input_layout: &Layout,
        coefficients: &MetalStorage,
        coefficients_layout: &Layout,
        base: &MetalStorage,
        base_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if input.dtype() != DType::F32
            || coefficients.dtype() != DType::F32
            || base.dtype() != DType::F32
            || !input_layout.is_contiguous()
            || !coefficients_layout.is_contiguous()
            || !base_layout.is_contiguous()
        {
            candle_core::bail!("DFlash dynamic convolution requires contiguous f32 inputs")
        }
        if input.device().id() != coefficients.device().id()
            || input.device().id() != base.device().id()
        {
            candle_core::bail!("DFlash dynamic convolution inputs are on different devices")
        }
        let (batch, seq_len, hidden_size) = input_layout.shape().dims3()?;
        if self.side >= 2
            || self.kernel_size == 0
            || self.group_size == 0
            || !hidden_size.is_multiple_of(self.group_size)
        {
            candle_core::bail!("DFlash dynamic convolution geometry differs")
        }
        let groups = hidden_size / self.group_size;
        let coefficient_elements = batch * seq_len * 2 * self.kernel_size * groups;
        if coefficients_layout.shape().elem_count() != coefficient_elements
            || base_layout.shape().elem_count() != 2 * self.kernel_size * hidden_size
        {
            candle_core::bail!(
                "DFlash dynamic convolution coefficient shape differs: coefficients {:?} ({} elements, expected {coefficient_elements}), base {:?} ({} elements, expected {})",
                coefficients_layout.shape(),
                coefficients_layout.shape().elem_count(),
                base_layout.shape(),
                base_layout.shape().elem_count(),
                2 * self.kernel_size * hidden_size,
            )
        }
        let elements = input_layout.shape().elem_count();
        if [
            elements,
            seq_len,
            hidden_size,
            groups,
            self.kernel_size,
            self.group_size,
            self.side,
        ]
        .into_iter()
        .any(|value| value > u32::MAX as usize)
        {
            candle_core::bail!("DFlash dynamic convolution geometry exceeds u32")
        }

        let device = input.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-dflash-dynamic-conv")
            .build()?;
        let pipeline = pipeline(device)?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-dflash-dynamic-conv");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(input.buffer()),
            input_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            1,
            Some(coefficients.buffer()),
            coefficients_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            2,
            Some(base.buffer()),
            base_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(3, Some(&output), 0);
        encoder_ref.set_bytes(4, &(elements as u32));
        encoder_ref.set_bytes(5, &(seq_len as u32));
        encoder_ref.set_bytes(6, &(hidden_size as u32));
        encoder_ref.set_bytes(7, &(groups as u32));
        encoder_ref.set_bytes(8, &(self.kernel_size as u32));
        encoder_ref.set_bytes(9, &(self.group_size as u32));
        encoder_ref.set_bytes(10, &(self.side as u32));
        encoder_ref.dispatch_threads(
            objc2_metal::MTLSize {
                width: elements,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: pipeline.max_total_threads_per_threadgroup().min(256),
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), elements, DType::F32),
            input_layout.shape().clone(),
        ))
    }
}

pub(super) fn try_dynamic_conv(
    input: &Tensor,
    coefficients: &Tensor,
    base: &Tensor,
    side: usize,
    kernel_size: usize,
    group_size: usize,
) -> Result<Option<Tensor>> {
    if !input.device().is_metal()
        || input.dtype() != DType::F32
        || coefficients.dtype() != DType::F32
        || base.dtype() != DType::F32
    {
        return Ok(None);
    }
    input
        .apply_op3_no_bwd(
            coefficients,
            base,
            &DFlashDynamicConv {
                side,
                kernel_size,
                group_size,
            },
        )
        .map(Some)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::try_dynamic_conv;

    #[test]
    fn dynamic_conv_matches_cpu_reference() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (batch, seq_len, hidden_size, group_size, kernel_size) = (1, 5, 48, 16, 2);
        let groups = hidden_size / group_size;
        let input_values = (0..batch * seq_len * hidden_size)
            .map(|index| (index as f32 - 97.0) / 53.0)
            .collect::<Vec<_>>();
        let coefficient_values = (0..batch * seq_len * 2 * kernel_size * groups)
            .map(|index| (index as f32 - 31.0) / 101.0)
            .collect::<Vec<_>>();
        let base_values = (0..2 * kernel_size * hidden_size)
            .map(|index| (index as f32 - 7.0) / 37.0)
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(input_values.clone(), (batch, seq_len, hidden_size), &device)?;
        let coefficients = Tensor::from_vec(
            coefficient_values.clone(),
            (batch, seq_len, 2 * kernel_size * groups),
            &device,
        )?;
        let base = Tensor::from_vec(base_values.clone(), (2, kernel_size, hidden_size), &device)?;

        for side in 0..2 {
            let output =
                try_dynamic_conv(&input, &coefficients, &base, side, kernel_size, group_size)?
                    .unwrap()
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
            let mut expected = vec![0.0f32; input_values.len()];
            for batch_index in 0..batch {
                for position in 0..seq_len {
                    for hidden in 0..hidden_size {
                        let group = hidden / group_size;
                        for tap in 0..kernel_size.min(position + 1) {
                            let coefficient_index =
                                ((((batch_index * seq_len + position) * 2 + side) * kernel_size
                                    + tap)
                                    * groups)
                                    + group;
                            let base_index = (side * kernel_size + tap) * hidden_size + hidden;
                            let input_index =
                                ((batch_index * seq_len + position - tap) * hidden_size) + hidden;
                            let output_index =
                                ((batch_index * seq_len + position) * hidden_size) + hidden;
                            expected[output_index] += (coefficient_values[coefficient_index]
                                + base_values[base_index])
                                * input_values[input_index];
                        }
                    }
                }
            }
            for (actual, expected) in output.iter().zip(&expected) {
                assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
            }
        }
        Ok(())
    }
}
