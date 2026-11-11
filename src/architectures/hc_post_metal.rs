//! Exact four-stream post mix for Apple SIMD-group matrix hardware.

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
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void dinf_hc_post4(
    device const float *update [[buffer(0)]],
    device const float *residual [[buffer(1)]],
    device const float *post [[buffer(2)]],
    device const float *comb [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant uint &hidden [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float a[64], b[64], c[64];
  const uint row = group.y;
  const uint start = group.x * 8;
  for (uint i = tid; i < 64; i += 32) {
    const uint r = i / 8, col = i % 8;
    // Stored combination axes are source, destination; transpose on load.
    a[i] = r < 4 && col < 4 ? comb[row * 16 + col * 4 + r] : 0.0f;
    b[i] = r < 4 && start + col < hidden
      ? residual[(row * 4 + r) * hidden + start + col] : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Match Candle's F32 SIMD-group GEMM, including its zero-padded K=4 tile.
  // A scalar sum or a lower-precision matrix changes the rounding.
  simdgroup_float8x8 am, bm, cm(0.0f);
  simdgroup_load(am, a, 8);
  simdgroup_load(bm, b, 8);
  simdgroup_multiply_accumulate(cm, am, bm, cm);
  simdgroup_store(cm, c, 8);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint dest = tid / 8, col = start + tid % 8;
  if (col < hidden) {
    // The old graph rounds the multiplication before adding the matrix result.
    volatile float weighted = update[row * hidden + col] * post[row * 4 + dest];
    output[(row * 4 + dest) * hidden + col] = weighted + c[tid];
  }
}
"#;

fn pipeline(device: &candle_core::MetalDevice) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<DeviceId, ComputePipeline>>> = OnceLock::new();
    let mut cache = PIPELINES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| candle_core::Error::Msg("HC post pipeline cache poisoned".into()))?;
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
        .get_function("dinf_hc_post4", None)
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    cache.insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

struct Post<'a> {
    combination: &'a Tensor,
}

impl CustomOp3 for Post<'_> {
    fn name(&self) -> &'static str {
        "dinf-hc-post4"
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
        candle_core::bail!("HC fused post mix requires Metal")
    }

    fn metal_fwd(
        &self,
        update: &MetalStorage,
        ul: &Layout,
        residual: &MetalStorage,
        rl: &Layout,
        post: &MetalStorage,
        pl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (guard, cl) = self.combination.storage_and_layout();
        let Storage::Metal(comb) = &*guard else {
            candle_core::bail!("HC combination must use Metal");
        };
        let device = update.device();
        let mut offsets = [0; 4];
        for (i, (storage, layout)) in [(update, ul), (residual, rl), (post, pl), (comb, cl)]
            .into_iter()
            .enumerate()
        {
            let _ = u32::try_from(layout.shape().elem_count()).map_err(candle_core::Error::wrap)?;
            if storage.dtype() != DType::F32
                || !layout.is_contiguous()
                || storage.device().id() != device.id()
            {
                candle_core::bail!("HC post inputs must be contiguous f32 on one Metal device");
            }
            offsets[i] = layout
                .start_offset()
                .checked_mul(4)
                .ok_or_else(|| candle_core::Error::Msg("HC post offset overflow".into()))?;
            let end = layout
                .start_offset()
                .checked_add(layout.shape().elem_count())
                .and_then(|v| v.checked_mul(4))
                .ok_or_else(|| candle_core::Error::Msg("HC post span overflow".into()))?;
            if end > storage.buffer().length() {
                candle_core::bail!("HC post input exceeds buffer");
            }
        }
        let (_, _, _, hidden) = rl.shape().dims4()?;
        let rows = rl.shape().elem_count() / (4 * hidden);
        let elements = rl.shape().elem_count();
        let hidden_u32 = u32::try_from(hidden).map_err(candle_core::Error::wrap)?;
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-hc-post4")
            .build()?;
        let pipeline = pipeline(device)?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        for (i, storage) in [update, residual, post, comb].into_iter().enumerate() {
            encoder.set_input_buffer(i, Some(storage.buffer()), offsets[i]);
        }
        encoder.set_output_buffer(4, Some(&output), 0);
        encoder.set_bytes(5, &hidden_u32);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: hidden.div_ceil(8),
                height: rows,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), elements, DType::F32),
            rl.shape().clone(),
        ))
    }
}

pub(super) fn forward(
    update: &Tensor,
    residual: &Tensor,
    post: &Tensor,
    combination: &Tensor,
) -> anyhow::Result<Tensor> {
    let (batch, rows, streams, hidden) = residual.dims4()?;
    anyhow::ensure!(
        batch > 0 && rows > 0 && streams == 4 && hidden > 1,
        "HC fused post mix requires nonempty four-stream matrices with width > 1"
    );
    anyhow::ensure!(
        update.dims() == [batch, rows, hidden]
            && post.dims() == [batch, rows, 4]
            && combination.dims() == [batch, rows, 4, 4],
        "HC post input shapes differ"
    );
    let combination = combination.contiguous()?;
    Ok(update.contiguous()?.apply_op3_no_bwd(
        &residual.contiguous()?,
        &post.contiguous()?,
        &Post {
            combination: &combination,
        },
    )?)
}

#[cfg(test)]
mod tests {
    use super::super::hc_post::reference;
    use super::*;

    fn bits(t: &Tensor) -> anyhow::Result<Vec<u32>> {
        Ok(t.flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }

    #[test]
    fn post_mix_matches_candle_bits() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for hidden in [2, 7, 8, 17, 64, 257, 2048, 4096] {
            for rows in [1, 3, 32] {
                let make = |n: usize, salt: usize| -> Result<Tensor> {
                    Tensor::from_vec(
                        (0..n)
                            .map(|i| {
                                let v = ((i * 37 + salt * 19) % 101) as f32 - 50.;
                                v * 1.013 / (1 + i % 7) as f32
                            })
                            .collect::<Vec<_>>(),
                        n,
                        &device,
                    )
                };
                let u = make((rows + 2) * hidden, 3)?
                    .reshape((1, rows + 2, hidden))?
                    .narrow(1, 1, rows)?;
                let r = make((rows + 2) * 4 * hidden, 5)?
                    .reshape((1, rows + 2, 4, hidden))?
                    .narrow(1, 1, rows)?;
                let p = make((rows + 2) * 4, 7)?
                    .reshape((1, rows + 2, 4))?
                    .narrow(1, 1, rows)?;
                let c = make((rows + 2) * 16, 11)?
                    .reshape((1, rows + 2, 4, 4))?
                    .narrow(1, 1, rows)?;
                assert_eq!(
                    bits(&forward(&u, &r, &p, &c)?)?,
                    bits(&reference(&u, &r, &p, &c)?)?,
                    "hidden={hidden} rows={rows}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn post_mix_strides_batches_and_rounding() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for mode in 0..4 {
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
                                2 => f32::from_bits(state & 0x807fffff), // Subnormal inputs.
                                _ => ((state >> 8) as f32 / 16777216. - 0.5) * 2.,
                            }
                        })
                        .collect::<Vec<_>>(),
                    n,
                    &device,
                )
            };
            let u = make(2 * 3 * 17, 3)?.reshape((3, 2, 17))?.transpose(0, 1)?;
            let r = make(2 * 3 * 4 * 17, 5)?
                .reshape((2, 3, 17, 4))?
                .transpose(2, 3)?;
            let p = make(2 * 3 * 4, 7)?.reshape((3, 2, 4))?.transpose(0, 1)?;
            let c = make(2 * 3 * 16, 11)?
                .reshape((2, 3, 4, 4))?
                .transpose(2, 3)?;
            let before = [&u, &r, &p, &c]
                .into_iter()
                .map(bits)
                .collect::<anyhow::Result<Vec<_>>>()?;
            let expected = bits(&reference(&u, &r.contiguous()?, &p, &c)?)?;
            assert_eq!(bits(&forward(&u, &r, &p, &c)?)?, expected, "mode={mode}");
            let after = [&u, &r, &p, &c]
                .into_iter()
                .map(bits)
                .collect::<anyhow::Result<Vec<_>>>()?;
            assert_eq!(before, after, "must not modify inputs");
        }
        Ok(())
    }

    #[test]
    fn post_mix_rejects_bad_inputs_and_keeps_fallback() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let u = Tensor::ones((1, 1, 8), DType::F32, &device)?;
        let r = Tensor::ones((1, 1, 4, 8), DType::F32, &device)?;
        let p = Tensor::ones((1, 1, 4), DType::F32, &device)?;
        let c = Tensor::ones((1, 1, 4, 4), DType::F32, &device)?;
        assert!(forward(&u, &r, &p.narrow(2, 0, 3)?, &c).is_err());
        assert!(forward(&u, &r, &p, &c.to_device(&candle_core::Device::Cpu)?).is_err());
        assert!(forward(&u, &r.to_dtype(DType::F16)?, &p, &c).is_err());
        let u = u.narrow(2, 0, 1)?.contiguous()?;
        let r = r.narrow(3, 0, 1)?.contiguous()?;
        assert!(forward(&u, &r, &p, &c).is_err());
        assert_eq!(
            bits(&super::super::hc_post::forward(&u, &r, &p, &c, true)?)?,
            bits(&reference(&u, &r, &p, &c)?)?
        );
        let cpu = candle_core::Device::Cpu;
        let (u, r, p, c) = (
            u.to_device(&cpu)?,
            r.to_device(&cpu)?,
            p.to_device(&cpu)?,
            c.to_device(&cpu)?,
        );
        assert_eq!(
            bits(&super::super::hc_post::forward(&u, &r, &p, &c, true)?)?,
            bits(&reference(&u, &r, &p, &c)?)?
        );
        Ok(())
    }
}
