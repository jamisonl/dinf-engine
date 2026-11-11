//! Bounded BF16 image attention matching the pinned native Apple10 MPP path.
use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{CpuStorage, CustomOp3, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline};
use objc2_foundation::{NSOperatingSystemVersion, NSProcessInfo};
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineState, MTLDevice, MTLGPUFamily, MTLLanguageVersion,
    MTLSize,
};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};
const MAX_INPUT_BYTES: usize = 64 * 1024 * 1024;
const MAX_OUTPUT_BYTES: usize = 32 * 1024 * 1024;
const MAX_MASK_BYTES: usize = 16 * 1024 * 1024;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Plan {
    batch: usize,
    heads: usize,
    queries: usize,
    keys: usize,
    mask_values: usize,
}
fn plan(q: &[usize], k: &[usize], v: &[usize], causal: bool, query_start: usize) -> Option<Plan> {
    let (&[b, h, qn, d], &[kb, kh, kn, kd], &[vb, vh, vn, vd]) = (q, k, v) else {
        return None;
    };
    if b == 0
        || h == 0
        || qn < 16
        || kn == 0
        || d != 128
        || (kb, kh, kd) != (b, h, d)
        || (vb, vh, vn, vd) != (b, h, kn, d)
        || [b, h, qn, kn].into_iter().any(|n| n > i32::MAX as usize)
        || (causal && (query_start != 0 || qn > kn))
    {
        return None;
    }
    let elements = |s: &[usize]| s.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
    let bytes = [q, k, v]
        .into_iter()
        .try_fold(0usize, |n, s| n.checked_add(elements(s)?.checked_mul(2)?))?;
    if bytes > MAX_INPUT_BYTES || elements(q)?.checked_mul(2)? > MAX_OUTPUT_BYTES {
        return None;
    }
    let mask_values = if causal { qn.checked_mul(kn)? } else { 1 };
    if mask_values.checked_mul(2)? > MAX_MASK_BYTES {
        return None;
    }
    Some(Plan {
        batch: b,
        heads: h,
        queries: qn,
        keys: kn,
        mask_values,
    })
}
fn supported_device(device: &candle_core::MetalDevice) -> bool {
    let os =
        NSProcessInfo::processInfo().isOperatingSystemAtLeastVersion(NSOperatingSystemVersion {
            majorVersion: 26,
            minorVersion: 2,
            patchVersion: 0,
        });
    let metal = device.metal_device().as_ref();
    os && metal.supportsFamily(MTLGPUFamily::Metal4) && metal.supportsFamily(MTLGPUFamily::Apple10)
}
type Pipes = [ComputePipeline; 2];
static PIPES: OnceLock<Mutex<HashMap<DeviceId, Option<Pipes>>>> = OnceLock::new();
fn pipelines(device: &candle_core::MetalDevice) -> Result<Option<Pipes>> {
    let mut cache = PIPES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| candle_core::Error::Msg("image MPP pipeline lock".into()))?;
    if let Some(p) = cache.get(&device.id()) {
        return Ok(p.clone());
    }
    let pipes = if supported_device(device) {
        let compile = || -> Result<Pipes> {
            let options = MTLCompileOptions::new();
            options.setLanguageVersion(MTLLanguageVersion::Version4_0);
            #[allow(deprecated)]
            options.setFastMathEnabled(false);
            let lib = device
                .metal_device()
                .new_library_with_source(include_str!("image_mpp_attention.metal"), Some(&options))
                .map_err(candle_core::Error::wrap)?;
            let make = |name| -> Result<ComputePipeline> {
                let f = lib
                    .get_function(name, None)
                    .map_err(candle_core::Error::wrap)?;
                let p = device
                    .metal_device()
                    .new_compute_pipeline_state_with_function(&f)
                    .map_err(candle_core::Error::wrap)?;
                if p.max_total_threads_per_threadgroup() < 128
                    || p.as_ref().threadExecutionWidth() != 32
                    || p.as_ref().staticThreadgroupMemoryLength()
                        > device.metal_device().as_ref().maxThreadgroupMemoryLength()
                {
                    candle_core::bail!("image MPP thread limit");
                }
                Ok(p)
            };
            Ok([make("image_mpp_bf16_mask")?, make("image_mpp_bf16")?])
        };
        match compile() {
            Ok(p) => Some(p),
            Err(error) => {
                tracing::warn!(%error,"image MPP attention unavailable; using prior path");
                None
            }
        }
    } else {
        None
    };
    if cache.len() >= 16 {
        cache.clear();
    }
    cache.insert(device.id(), pipes.clone());
    Ok(pipes)
}
pub(super) fn try_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    query_start: usize,
) -> Result<Option<Tensor>> {
    if !q.device().is_metal()
        || q.dtype() != DType::BF16
        || k.dtype() != DType::BF16
        || v.dtype() != DType::BF16
        || !q.device().same_device(k.device())
        || !q.device().same_device(v.device())
    {
        return Ok(None);
    }
    let Some(p) = plan(q.dims(), k.dims(), v.dims(), causal, query_start) else {
        return Ok(None);
    };
    let Some(pipes) = pipelines(q.device().as_metal_device()?)? else {
        return Ok(None);
    };
    // All copy, output and mask limits are checked before materializing views.
    Ok(Some(q.contiguous()?.apply_op3_no_bwd(
        &k.contiguous()?,
        &v.contiguous()?,
        &Attention {
            plan: p,
            causal,
            pipe: pipes[usize::from(!causal)].clone(),
        },
    )?))
}
#[repr(C)]
struct Params {
    batch: i32,
    heads: i32,
    dim: i32,
    queries: i32,
    keys: i32,
    gqa_factor: i32,
    scale: f32,
    softcapping: f32,
    q_strides: [i64; 3],
    k_strides: [i64; 3],
    v_strides: [i64; 3],
    o_strides: [i64; 3],
}
fn strides(heads: usize, rows: usize) -> [i64; 3] {
    [(heads * rows * 128) as i64, (rows * 128) as i64, 128]
}
fn span(storage: &MetalStorage, layout: &Layout) -> Result<usize> {
    let bytes = layout
        .shape()
        .elem_count()
        .checked_mul(2)
        .ok_or_else(|| candle_core::Error::Msg("image MPP span size".into()))?;
    let offset = layout
        .start_offset()
        .checked_mul(2)
        .ok_or_else(|| candle_core::Error::Msg("image MPP span offset".into()))?;
    if storage.dtype() != DType::BF16
        || !layout.is_contiguous()
        || bytes > MAX_INPUT_BYTES
        || offset
            .checked_add(bytes)
            .is_none_or(|n| n > storage.buffer().length())
    {
        candle_core::bail!("image MPP buffer span");
    }
    Ok(offset)
}
struct Attention {
    plan: Plan,
    causal: bool,
    pipe: ComputePipeline,
}
impl CustomOp3 for Attention {
    fn name(&self) -> &'static str {
        "bf16-image-mpp-attention"
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
        candle_core::bail!("image MPP attention requires Metal")
    }
    fn metal_fwd(
        &self,
        q: &MetalStorage,
        ql: &Layout,
        k: &MetalStorage,
        kl: &Layout,
        v: &MetalStorage,
        vl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (qo, ko, vo) = (span(q, ql)?, span(k, kl)?, span(v, vl)?);
        if q.device().id() != k.device().id()
            || q.device().id() != v.device().id()
            || plan(ql.dims(), kl.dims(), vl.dims(), self.causal, 0) != Some(self.plan)
        {
            candle_core::bail!("image MPP geometry changed");
        }
        let p = self.plan;
        let dev = q.device();
        let n = ql.shape().elem_count();
        let output = dev
            .new_buffer_builder()
            .with_size_for(n, DType::BF16)
            .with_label("image-mpp-attention-output")
            .build()?;
        let mask = if self.causal {
            (0..p.mask_values)
                .map(|i| {
                    half::bf16::from_f32(if i % p.keys <= i / p.keys {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    })
                })
                .collect::<Vec<_>>()
        } else {
            vec![half::bf16::ZERO]
        };
        let mask = dev
            .new_buffer_builder()
            .with_data(&mask)
            .with_label("image-mpp-causal-mask")
            .build()?;
        let params = Params {
            batch: p.batch as i32,
            heads: p.heads as i32,
            dim: 128,
            queries: p.queries as i32,
            keys: p.keys as i32,
            gqa_factor: 1,
            scale: (1.0_f64 / 128.0_f64.sqrt()) as f32,
            softcapping: 0.0,
            q_strides: strides(p.heads, p.queries),
            k_strides: strides(p.heads, p.keys),
            v_strides: strides(p.heads, p.keys),
            o_strides: strides(p.heads, p.queries),
        };
        let mask_strides = [0_i64, 0, p.keys as i64, 1];
        let encoder = dev.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.pipe);
        let e: &ComputeCommandEncoder = encoder.as_ref();
        e.set_input_buffer(0, Some(q.buffer()), qo);
        e.set_input_buffer(1, Some(k.buffer()), ko);
        e.set_input_buffer(2, Some(v.buffer()), vo);
        e.set_output_buffer(3, Some(&output), 0);
        e.set_bytes(4, &params);
        e.set_bytes(5, &mask_strides);
        e.set_input_buffer(6, Some(&mask), 0);
        e.dispatch_thread_groups(
            MTLSize {
                width: p.queries.div_ceil(64),
                height: p.heads,
                depth: p.batch,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, dev.clone(), n, DType::BF16),
            ql.shape().clone(),
        ))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admission_limits_and_prefix_start() {
        assert!(
            plan(
                &[1, 32, 32, 128],
                &[1, 32, 32, 128],
                &[1, 32, 32, 128],
                true,
                0
            )
            .is_some()
        );
        assert!(
            plan(
                &[1, 32, 1024, 128],
                &[1, 32, 1056, 128],
                &[1, 32, 1056, 128],
                false,
                32
            )
            .is_some()
        );
        for (q, k, v, c, s) in [
            ([1, 1, 32, 128], [1, 1, 32, 128], [1, 1, 32, 128], true, 1),
            ([1, 1, 33, 128], [1, 1, 32, 128], [1, 1, 32, 128], true, 0),
            ([0, 1, 32, 128], [0, 1, 32, 128], [0, 1, 32, 128], false, 0),
            ([1, 1, 15, 128], [1, 1, 32, 128], [1, 1, 32, 128], false, 0),
            ([1, 1, 32, 64], [1, 1, 32, 64], [1, 1, 32, 64], false, 0),
            ([1, 1, 32, 128], [1, 2, 32, 128], [1, 2, 32, 128], false, 0),
            ([1, 1, 32, 128], [1, 1, 32, 128], [1, 1, 33, 128], false, 0),
            (
                [1, usize::MAX, 32, 128],
                [1, usize::MAX, 32, 128],
                [1, usize::MAX, 32, 128],
                false,
                0,
            ),
            (
                [1, 1, 4096, 128],
                [1, 1, 4096, 128],
                [1, 1, 4096, 128],
                true,
                0,
            ),
            (
                [1, 32, 16384, 128],
                [1, 32, 16384, 128],
                [1, 32, 16384, 128],
                false,
                0,
            ),
        ] {
            assert!(plan(&q, &k, &v, c, s).is_none());
        }
        assert!(plan(&[1, 32, 32], &[1, 32, 32], &[1, 32, 32], false, 0).is_none());
        assert_eq!(std::mem::size_of::<Params>(), 128);
    }

    #[test]
    fn native_bf16_partial_tiles_views_and_f64_bounds() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        if !supported_device(device.as_metal_device()?) {
            return Ok(());
        }
        let cpu = Tensor::zeros((1, 1, 16, 128), DType::BF16, &candle_core::Device::Cpu)?;
        assert!(try_attention(&cpu, &cpu, &cpu, false, 0)?.is_none());
        let f32 = cpu.to_device(&device)?.to_dtype(DType::F32)?;
        assert!(try_attention(&f32, &f32, &f32, false, 0)?.is_none());
        for (index, (b, h, qn, kn, causal)) in [
            (1, 1, 32, 32, true),
            (1, 2, 17, 31, true),
            (2, 3, 31, 33, false),
            (1, 2, 65, 63, false),
        ]
        .into_iter()
        .enumerate()
        {
            let make = |rows: usize, j: usize| -> Result<(Tensor, Vec<f32>)> {
                let data = (0..b * h * rows * 128)
                    .map(|i| (((i * (17 + j * 8) + 31 * j + index * 13) % 197) as f32 - 98.) / 64.)
                    .collect::<Vec<_>>();
                let t = Tensor::from_vec(data.clone(), (b, h, rows, 128), &device)?
                    .to_dtype(DType::BF16)?;
                let pad = Tensor::zeros((b, h, 1, 128), DType::BF16, &device)?;
                let t = Tensor::cat(&[&pad, &t], 2)?.narrow(2, 1, rows)?;
                let t = if index == 0 {
                    t
                } else {
                    t.transpose(2, 3)?.contiguous()?.transpose(2, 3)?
                };
                Ok((t, data))
            };
            let (q, a) = make(qn, 0)?;
            let (k, b_values) = make(kn, 1)?;
            let (v, c) = make(kn, 2)?;
            if index == 0 {
                assert!(q.is_contiguous());
                assert!(q.layout().start_offset() > 0);
            } else {
                assert!(!q.is_contiguous());
            }
            let output = try_attention(&q, &k, &v, causal, if causal { 0 } else { 11 })?
                .expect("qualified MPP kernel");
            let values = output
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert!(values.iter().all(|x| x.is_finite()));
            let mut err = 0.;
            let mut norm = 0.;
            let mut actual_norm = 0.;
            let mut dot = 0.;
            for bh in 0..b * h {
                for row in 0..qn {
                    let visible = if causal { row + 1 } else { kn };
                    let mut scores = Vec::with_capacity(visible);
                    for key in 0..visible {
                        let score = (0..128)
                            .map(|d| {
                                f64::from(a[(bh * qn + row) * 128 + d])
                                    * f64::from(b_values[(bh * kn + key) * 128 + d])
                            })
                            .sum::<f64>()
                            / 128.0_f64.sqrt();
                        scores.push(score);
                    }
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    for s in &mut scores {
                        *s = (*s - max).exp();
                    }
                    let total = scores.iter().sum::<f64>();
                    for d in 0..128 {
                        let expected = scores
                            .iter()
                            .enumerate()
                            .map(|(key, p)| p * f64::from(c[(bh * kn + key) * 128 + d]))
                            .sum::<f64>()
                            / total;
                        let actual = f64::from(values[(bh * qn + row) * 128 + d]);
                        err += (actual - expected).powi(2);
                        norm += expected * expected;
                        actual_norm += actual * actual;
                        dot += actual * expected;
                    }
                }
            }
            assert!(
                (err / norm).sqrt() <= 0.003,
                "BF16 F64 error for case {index}"
            );
            assert!(
                dot / (norm * actual_norm).sqrt() >= 0.99999,
                "BF16 F64 cosine for case {index}"
            );
            assert!(try_attention(&q, &k, &v, true, 1)?.is_none());
        }
        Ok(())
    }
}
