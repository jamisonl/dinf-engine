//! F32 absorbed-MLA value reduction with the same sum order at every prompt size.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{CpuStorage, CustomOp2, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLGPUFamily};

pub(super) fn supported(device: &candle_core::Device) -> Result<bool> {
    Ok(device.is_metal()
        && device
            .as_metal_device()?
            .metal_device()
            .as_ref()
            .supportsFamily(MTLGPUFamily::Apple7))
}

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void dinf_glm5_latent_context_f32(
    device const float *probabilities [[buffer(0)]],
    device const float *latent [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &total [[buffer(3)]],
    constant uint &rank [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float p_tile[16 * 32];
  threadgroup float v_tile[32 * 32];
  threadgroup float result[16 * 32];
  const uint first_row = group.y * 16;
  const uint first_column = group.x * 32;
  const uint simd_row = uint(simd_group) / 2 * 8;
  const uint simd_column = uint(simd_group) % 2 * 16;
  simdgroup_float8x8 accumulator0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
  simdgroup_float8x8 accumulator1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
  // Fixed 32-key tiles, reduced in eight-key steps, use F32 operands and the
  // same sum order for all prompt sizes. No split-K or F16 operand conversion.
  for (uint key = 0; key < total; key += 32) {
    for (uint i = tid; i < 16 * 32; i += 128) {
      const uint row = first_row + i / 32;
      const uint k = key + i % 32;
      p_tile[i] = row < rows && k < total ? probabilities[row * total + k] : 0.0f;
    }
    for (uint i = tid; i < 32 * 32; i += 128) {
      const uint k = key + i / 32;
      const uint column = first_column + i % 32;
      v_tile[i] = k < total && column < rank ? latent[k * rank + column] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint inner = 0; inner < 32; inner += 8) {
      simdgroup_float8x8 p, v0, v1;
      simdgroup_load(p, p_tile + simd_row * 32 + inner, 32);
      simdgroup_load(v0, v_tile + inner * 32 + simd_column, 32);
      simdgroup_load(v1, v_tile + inner * 32 + simd_column + 8, 32);
      simdgroup_multiply_accumulate(accumulator0, p, v0, accumulator0);
      simdgroup_multiply_accumulate(accumulator1, p, v1, accumulator1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  simdgroup_store(accumulator0, result + simd_row * 32 + simd_column, 32);
  simdgroup_store(accumulator1, result + simd_row * 32 + simd_column + 8, 32);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 16 * 32; i += 128) {
    const uint row = first_row + i / 32;
    const uint column = first_column + i % 32;
    if (row < rows && column < rank) output[row * rank + column] = result[i];
  }
}
"#;

static PIPELINES: OnceLock<Mutex<HashMap<DeviceId, ComputePipeline>>> = OnceLock::new();

fn pipeline(device: &candle_core::MetalDevice) -> Result<ComputePipeline> {
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(pipeline) = cache.lock().unwrap().get(&device.id()) {
        return Ok(pipeline.clone());
    }
    let library = device
        .metal_device()
        .new_library_with_source(SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function("dinf_glm5_latent_context_f32", None)
        .map_err(candle_core::Error::wrap)?;
    let compiled = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    Ok(cache
        .lock()
        .unwrap()
        .entry(device.id())
        .or_insert(compiled)
        .clone())
}

struct LatentContext;

impl CustomOp2 for LatentContext {
    fn name(&self) -> &'static str {
        "dinf-glm5-latent-context"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("GLM5 latent context kernel requires Metal")
    }

    fn metal_fwd(
        &self,
        probabilities: &MetalStorage,
        probability_layout: &Layout,
        latent: &MetalStorage,
        latent_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (batch, heads, rows, total) = probability_layout.shape().dims4()?;
        let (latent_batch, latent_heads, latent_rows, rank) = latent_layout.shape().dims4()?;
        if batch != 1
            || latent_batch != 1
            || latent_heads != 1
            || latent_rows != total
            || heads == 0
            || rows == 0
            || total == 0
            || total > u32::MAX as usize - 31
            || rank == 0
            || probabilities.dtype() != DType::F32
            || latent.dtype() != DType::F32
            || !probability_layout.is_contiguous()
            || !latent_layout.is_contiguous()
            || probabilities.device().id() != latent.device().id()
        {
            candle_core::bail!(
                "GLM5 latent context requires contiguous F32 probabilities and a shared latent cache"
            )
        }
        let shape = Shape::from((batch, heads, rows, rank));
        let elements = shape.elem_count();
        if [
            probability_layout.shape().elem_count(),
            latent_layout.shape().elem_count(),
            elements,
        ]
        .iter()
        .any(|&count| count > u32::MAX as usize)
        {
            candle_core::bail!("GLM5 latent context geometry exceeds u32")
        }
        let device = probabilities.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-glm5-latent-context")
            .build()?;
        let pipeline = pipeline(device)?;
        if pipeline.as_ref().threadExecutionWidth() != 32
            || pipeline.max_total_threads_per_threadgroup() < 128
        {
            candle_core::bail!("GLM5 latent context requires four 32-thread SIMD groups")
        }
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-glm5-latent-context");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder.set_input_buffer(
            0,
            Some(probabilities.buffer()),
            probability_layout.start_offset() * 4,
        );
        encoder.set_input_buffer(1, Some(latent.buffer()), latent_layout.start_offset() * 4);
        encoder.set_output_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, &(total as u32));
        encoder.set_bytes(4, &(rank as u32));
        encoder.set_bytes(5, &((heads * rows) as u32));
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: rank.div_ceil(32),
                height: (heads * rows).div_ceil(16),
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), elements, DType::F32),
            shape,
        ))
    }
}

pub(super) fn weighted_context(probabilities: &Tensor, latent: &Tensor) -> Result<Tensor> {
    probabilities
        .contiguous()?
        .apply_op2_no_bwd(&latent.contiguous()?, &LatentContext)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{D, Device};

    fn test_device() -> anyhow::Result<Option<Device>> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(None);
        };
        if !supported(&device)? {
            eprintln!("GLM context kernel needs an Apple7-family or newer GPU");
            return Ok(None);
        }
        Ok(Some(device))
    }

    #[test]
    fn context_matches_cpu_and_is_exact_across_causal_partitions() -> anyhow::Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        for rank in [7, 128, 512] {
            let (heads, rows, total) = (3, 152, 152);
            let latent = Tensor::from_vec(
                (0..(total + 1) * rank)
                    .map(|i| (i as f32 * 0.019).sin())
                    .collect::<Vec<_>>(),
                (1, 1, total + 1, rank),
                &Device::Cpu,
            )?
            .to_dtype(DType::F16)?
            .to_dtype(DType::F32)?;
            // Nonzero storage offsets in both inputs, as cache views have.
            let latent = latent.narrow(2, 1, total)?;
            let mut data = vec![0f32; heads * rows * total];
            for head in 0..heads {
                for row in 0..rows {
                    for key in 0..=row {
                        data[(head * rows + row) * total + key] =
                            ((head * 31 + row * 17 + key) as f32 * 0.031).cos().exp()
                                / (row + 1) as f32;
                    }
                }
            }
            let probabilities = Tensor::from_vec(data, (1, heads, rows, total), &Device::Cpu)?;
            let reference = probabilities.broadcast_matmul(&latent)?;
            let probabilities = Tensor::cat(
                &[
                    &Tensor::zeros((1, heads, 1, total), DType::F32, &device)?,
                    &probabilities.to_device(&device)?,
                ],
                2,
            )?
            .narrow(2, 1, rows)?;
            let latent = Tensor::cat(
                &[
                    &Tensor::zeros((1, 1, 1, rank), DType::F32, &device)?,
                    &latent.to_device(&device)?,
                ],
                2,
            )?
            .narrow(2, 1, total)?;
            let whole = weighted_context(&probabilities, &latent)?;
            let error = (&whole.to_device(&Device::Cpu)? - &reference)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            anyhow::ensure!(error < 2e-6, "rank {rank} CPU error {error}");
            for chunk in [1, 7, 16, 32, 128] {
                for start in (0..rows).step_by(chunk) {
                    let len = chunk.min(rows - start);
                    let end = start + len;
                    let actual = weighted_context(
                        &probabilities
                            .narrow(2, start, len)?
                            .narrow(D::Minus1, 0, end)?,
                        &latent.narrow(2, 0, end)?,
                    )?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                    let expected = whole
                        .narrow(2, start, len)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    anyhow::ensure!(
                        actual == expected,
                        "rank {rank}: {start}+{len} changed context"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn long_sparse_context_preserves_cached_prefix_and_masked_values() -> anyhow::Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        let (heads, rows, total, rank) = (2, 17, 4097, 512);
        let prefix = total - rows;
        let latent = Tensor::from_vec(
            (0..total * rank)
                .map(|i| (i as f32 * 0.019).sin())
                .collect::<Vec<_>>(),
            (1, 1, total, rank),
            &Device::Cpu,
        )?;
        let mut data = vec![0f32; heads * rows * total];
        for head in 0..heads {
            for row in 0..rows {
                for key in (0..=prefix + row).step_by(3) {
                    data[(head * rows + row) * total + key] =
                        ((head * 31 + row * 17 + key) as f32 * 0.031).cos().exp()
                            / (prefix + row + 1) as f32;
                }
            }
        }
        let probabilities = Tensor::from_vec(data, (1, heads, rows, total), &Device::Cpu)?;
        let reference = probabilities.broadcast_matmul(&latent)?;
        let probabilities = probabilities.to_device(&device)?;
        let latent = latent.to_device(&device)?;
        let whole = weighted_context(&probabilities, &latent)?;
        let error = (&whole.to_device(&Device::Cpu)? - reference)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        anyhow::ensure!(error < 2e-6, "long context CPU error {error}");
        for start in 0..rows {
            let end = prefix + start + 1;
            let actual = weighted_context(
                &probabilities.narrow(2, start, 1)?.narrow(3, 0, end)?,
                &latent.narrow(2, 0, end)?,
            )?
            .flatten_all()?
            .to_vec1::<f32>()?;
            let expected = whole.narrow(2, start, 1)?.flatten_all()?.to_vec1::<f32>()?;
            anyhow::ensure!(actual == expected, "cached row {start} changed context");
        }
        let masked = Tensor::new(&[[[[1f32, 0.]]]], &device)?;
        let values = Tensor::new(&[[[[2f32, -3.], [1e30, -1e30]]]], &device)?;
        assert_eq!(
            weighted_context(&masked, &values)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            [2., -3.]
        );
        Ok(())
    }

    #[test]
    fn context_rejects_invalid_geometry_and_precision() -> anyhow::Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        let probabilities = Tensor::ones((1, 2, 3, 5), DType::F32, &device)?;
        let latent = Tensor::ones((1, 1, 5, 7), DType::F32, &device)?;
        assert!(weighted_context(&probabilities.to_dtype(DType::F16)?, &latent).is_err());
        assert!(weighted_context(&probabilities, &latent.to_dtype(DType::F16)?).is_err());
        for shape in [(1, 2, 5, 7), (1, 1, 4, 7), (1, 1, 5, 0), (2, 1, 5, 7)] {
            assert!(
                weighted_context(&probabilities, &Tensor::zeros(shape, DType::F32, &device)?)
                    .is_err()
            );
        }
        assert!(weighted_context(&probabilities.repeat((2, 1, 1, 1))?, &latent).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "manual Metal timing; run without a resident model"]
    fn benchmark_latent_context() -> anyhow::Result<()> {
        let Some(device) = test_device()? else {
            return Ok(());
        };
        for (rows, total) in [(1, 152), (128, 152), (1, 4096), (128, 4096)] {
            let (heads, rank) = (64, 512);
            let probabilities = Tensor::ones((1, heads, rows, total), DType::F32, &device)?
                .affine(1.0 / total as f64, 0.)?;
            let latent = Tensor::ones((1, 1, total, rank), DType::F32, &device)?;
            let flat_probabilities = probabilities.reshape((heads * rows, total))?;
            let flat_latent = latent.reshape((total, rank))?;
            for native in [false, true] {
                let run = || -> Result<Tensor> {
                    if native {
                        weighted_context(&probabilities, &latent)
                    } else {
                        flat_probabilities.matmul(&flat_latent)
                    }
                };
                drop(run()?);
                device.synchronize()?;
                let started = std::time::Instant::now();
                for _ in 0..10 {
                    drop(run()?);
                    device.synchronize()?;
                }
                eprintln!(
                    "context rows={rows} total={total} native={native} ms={:.3}",
                    started.elapsed().as_secs_f64() * 100.
                );
            }
        }
        Ok(())
    }
}
