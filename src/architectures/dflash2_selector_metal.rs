use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{CpuStorage, CustomOp2, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;

const DFLASH_SELECTOR_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void dinf_dflash_select_path_f32(
    device const float *packed [[buffer(0)]],
    device const uint *candidates [[buffer(1)]],
    device uint *output [[buffer(2)]],
    constant uint &block_size [[buffer(3)]],
    constant uint &top_k [[buffer(4)]],
    constant uint &rank [[buffer(5)]],
    constant uint &predecessor_rows [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float scores[16];
  threadgroup uint selected;
  if (tid == 0) {
    selected = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint hidden_offset = 0;
  const uint predecessor_offset = block_size * rank;
  const uint successor_offset = predecessor_offset + predecessor_rows * rank;
  const uint successor_rows = (block_size - 1) * top_k;
  const uint unary_offset = successor_offset + successor_rows * rank;

  for (uint position = 0; position + 1 < block_size; ++position) {
    if (tid < top_k) {
      const uint predecessor_row = position == 0
          ? 0
          : 1 + (position - 1) * top_k + selected;
      const uint successor_row = position * top_k + tid;
      const uint hidden_row = position + 1;
      float score = packed[unary_offset + hidden_row * top_k + tid];
      for (uint column = 0; column < rank; ++column) {
        score += packed[predecessor_offset + predecessor_row * rank + column]
            * packed[hidden_offset + hidden_row * rank + column]
            * packed[successor_offset + successor_row * rank + column];
      }
      scores[tid] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      uint best = 0;
      for (uint candidate = 1; candidate < top_k; ++candidate) {
        if (scores[candidate] > scores[best]) {
          best = candidate;
        }
      }
      selected = best;
      output[position] = candidates[(position + 1) * top_k + best];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
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
        .new_library_with_source(DFLASH_SELECTOR_MSL, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function("dinf_dflash_select_path_f32", None)
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

struct DFlashSelectPath {
    block_size: usize,
    top_k: usize,
    rank: usize,
    predecessor_rows: usize,
}

impl CustomOp2 for DFlashSelectPath {
    fn name(&self) -> &'static str {
        "dinf-metal-dflash-select-path"
    }

    fn cpu_fwd(
        &self,
        _packed: &CpuStorage,
        _packed_layout: &Layout,
        _candidates: &CpuStorage,
        _candidates_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("DFlash path selection is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        packed: &MetalStorage,
        packed_layout: &Layout,
        candidates: &MetalStorage,
        candidates_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if packed.dtype() != DType::F32
            || candidates.dtype() != DType::U32
            || !packed_layout.is_contiguous()
            || !candidates_layout.is_contiguous()
        {
            candle_core::bail!("DFlash path selection requires contiguous f32/u32 inputs")
        }
        if packed.device().id() != candidates.device().id() {
            candle_core::bail!("DFlash path selection inputs are on different devices")
        }
        if self.block_size < 2 || self.top_k == 0 || self.top_k > 16 || self.rank == 0 {
            candle_core::bail!("DFlash path selection geometry differs")
        }
        let successor_rows = (self.block_size - 1) * self.top_k;
        let expected_packed = self.block_size * self.rank
            + self.predecessor_rows * self.rank
            + successor_rows * self.rank
            + self.block_size * self.top_k;
        if packed_layout.shape().elem_count() != expected_packed
            || candidates_layout.shape().elem_count() != self.block_size * self.top_k
            || self.predecessor_rows != 1 + self.block_size.saturating_sub(2) * self.top_k
        {
            candle_core::bail!("DFlash path selection input shape differs")
        }
        if [
            self.block_size,
            self.top_k,
            self.rank,
            self.predecessor_rows,
        ]
        .into_iter()
        .any(|value| value > u32::MAX as usize)
        {
            candle_core::bail!("DFlash path selection geometry exceeds u32")
        }

        let output_elements = self.block_size - 1;
        let device = packed.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, DType::U32)
            .with_label("dinf-dflash-select-path")
            .build()?;
        let pipeline = pipeline(device)?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-dflash-select-path");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(packed.buffer()),
            packed_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            1,
            Some(candidates.buffer()),
            candidates_layout.start_offset() * DType::U32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(2, Some(&output), 0);
        encoder_ref.set_bytes(3, &(self.block_size as u32));
        encoder_ref.set_bytes(4, &(self.top_k as u32));
        encoder_ref.set_bytes(5, &(self.rank as u32));
        encoder_ref.set_bytes(6, &(self.predecessor_rows as u32));
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), output_elements, DType::U32),
            Shape::from(output_elements),
        ))
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_select_path(
    hidden: &Tensor,
    predecessors: &Tensor,
    successors: &Tensor,
    unary: &Tensor,
    candidates: &Tensor,
    block_size: usize,
    top_k: usize,
) -> Result<Option<Tensor>> {
    if !hidden.device().is_metal()
        || hidden.dtype() != DType::F32
        || predecessors.dtype() != DType::F32
        || successors.dtype() != DType::F32
        || unary.dtype() != DType::F32
        || candidates.dtype() != DType::U32
    {
        return Ok(None);
    }
    let rank = hidden.dim(candle_core::D::Minus1)?;
    let predecessor_rows = predecessors.dim(0)?;
    let hidden = hidden.flatten_all()?;
    let predecessors = predecessors.flatten_all()?;
    let successors = successors.flatten_all()?;
    let unary = unary.flatten_all()?;
    let packed = Tensor::cat(&[&hidden, &predecessors, &successors, &unary], 0)?;
    packed
        .apply_op2_no_bwd(
            &candidates.contiguous()?,
            &DFlashSelectPath {
                block_size,
                top_k,
                rank,
                predecessor_rows,
            },
        )
        .map(Some)
}

#[cfg(test)]
mod tests {
    use candle_core::Tensor;

    use super::try_select_path;

    #[test]
    fn fused_path_matches_cpu_reference() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (block_size, top_k, rank) = (8, 4, 13);
        let predecessor_rows = 1 + (block_size - 2) * top_k;
        let successor_rows = (block_size - 1) * top_k;
        let hidden_values = (0..block_size * rank)
            .map(|index| ((index * 29 % 101) as f32 - 50.0) / 31.0)
            .collect::<Vec<_>>();
        let predecessor_values = (0..predecessor_rows * rank)
            .map(|index| ((index * 43 % 131) as f32 - 65.0) / 37.0)
            .collect::<Vec<_>>();
        let successor_values = (0..successor_rows * rank)
            .map(|index| ((index * 47 % 137) as f32 - 68.0) / 41.0)
            .collect::<Vec<_>>();
        let unary_values = (0..block_size * top_k)
            .map(|index| ((index * 17 % 61) as f32 - 30.0) / 19.0)
            .collect::<Vec<_>>();
        let candidate_values = (0..block_size * top_k)
            .map(|index| (1000 + index) as u32)
            .collect::<Vec<_>>();
        let hidden = Tensor::from_vec(hidden_values.clone(), (block_size, rank), &device)?;
        let predecessors = Tensor::from_vec(
            predecessor_values.clone(),
            (predecessor_rows, rank),
            &device,
        )?;
        let successors =
            Tensor::from_vec(successor_values.clone(), (successor_rows, rank), &device)?;
        let unary = Tensor::from_vec(unary_values.clone(), (block_size, top_k), &device)?;
        let candidates = Tensor::from_vec(candidate_values.clone(), (block_size, top_k), &device)?;
        let actual = try_select_path(
            &hidden,
            &predecessors,
            &successors,
            &unary,
            &candidates,
            block_size,
            top_k,
        )?
        .unwrap()
        .to_vec1::<u32>()?;

        let mut selected = 0usize;
        let mut expected = Vec::with_capacity(block_size - 1);
        for position in 0..block_size - 1 {
            let predecessor_row = if position == 0 {
                0
            } else {
                1 + (position - 1) * top_k + selected
            };
            let mut scores = Vec::with_capacity(top_k);
            for candidate in 0..top_k {
                let successor_row = position * top_k + candidate;
                let hidden_row = position + 1;
                let mut score = unary_values[hidden_row * top_k + candidate];
                for column in 0..rank {
                    score += predecessor_values[predecessor_row * rank + column]
                        * hidden_values[hidden_row * rank + column]
                        * successor_values[successor_row * rank + column];
                }
                scores.push(score);
            }
            selected = (1..top_k).fold(0, |best, candidate| {
                if scores[candidate] > scores[best] {
                    candidate
                } else {
                    best
                }
            });
            expected.push(candidate_values[(position + 1) * top_k + selected]);
        }
        assert_eq!(actual, expected);
        Ok(())
    }
}
