//! BF16 text RMS normalization with the pinned native F32 reduction order.
use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{CpuStorage, CustomOp2, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputeCommandEncoder, ComputePipeline};
use objc2_metal::{MTLCompileOptions, MTLMathFloatingPointFunctions, MTLMathMode, MTLSize};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};
const SOURCE: &str = include_str!("text_norm_metal.metal");
const MAX_VALUES: usize = 16 * 1024 * 1024;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Plan {
    rows: u32,
    width: u32,
    lanes: u32,
    segments: u32,
    chunk: bool,
}
fn plan(rows: usize, width: usize) -> Option<Plan> {
    if rows < 2
        || !matches!(width, 128 | 4096)
        || rows.checked_mul(width).is_none_or(|n| n > MAX_VALUES)
    {
        return None;
    }
    let mut p = Plan {
        rows: rows as u32,
        width: width as u32,
        lanes: 32,
        segments: 1,
        chunk: false,
    };
    if width <= 256 && rows * width >= 65536 {
        p.chunk = true;
        p.lanes = width.div_ceil(16).next_power_of_two().min(32) as u32;
    } else if rows.div_ceil(8) < 64 && width >= 2048 {
        p.chunk = true;
        let max_segments = width.div_ceil(64).min(2048);
        let target = 8192usize.div_ceil(rows).clamp(2, max_segments.max(2));
        p.segments = width.div_ceil(width.div_ceil(target)) as u32;
    }
    Some(p)
}
type Pipelines = [ComputePipeline; 4];
static PIPELINES: OnceLock<Mutex<HashMap<DeviceId, Pipelines>>> = OnceLock::new();
fn pipelines(device: &candle_core::MetalDevice) -> Result<Pipelines> {
    let mut cache = PIPELINES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| candle_core::Error::Msg("text norm pipeline lock".into()))?;
    if let Some(p) = cache.get(&device.id()) {
        return Ok(p.clone());
    }
    let options = MTLCompileOptions::new();
    options.setMathMode(MTLMathMode::Safe);
    options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
    let lib = device
        .metal_device()
        .new_library_with_source(SOURCE, Some(&options))
        .map_err(candle_core::Error::wrap)?;
    let make = |name| -> Result<ComputePipeline> {
        let f = lib
            .get_function(name, None)
            .map_err(candle_core::Error::wrap)?;
        device
            .metal_device()
            .new_compute_pipeline_state_with_function(&f)
            .map_err(candle_core::Error::wrap)
    };
    let p = [
        make("square_input")?,
        make("mean_row")?,
        make("mean_chunk")?,
        make("finish_norm")?,
    ];
    if cache.len() >= 16 {
        cache.clear();
    }
    cache.insert(device.id(), p.clone());
    Ok(p)
}
pub(super) fn try_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Option<Tensor>> {
    if x.dtype() != DType::BF16
        || w.dtype() != DType::BF16
        || !(2..=4).contains(&x.rank())
        || !x.device().same_device(w.device())
        || !super::text_linear_mps::supported_device(x.device())
        || !eps.is_finite()
        || eps <= 0.0
        || !(eps as f32).is_finite()
    {
        return Ok(None);
    }
    let width = *x.dims().last().unwrap();
    if width == 0 || w.dims() != [width] || plan(x.elem_count() / width, width).is_none() {
        return Ok(None);
    }
    Ok(Some(x.contiguous()?.apply_op2_no_bwd(
        &w.contiguous()?,
        &Norm { eps: eps as f32 },
    )?))
}
fn span(x: &MetalStorage, l: &Layout) -> Result<usize> {
    let offset = l
        .start_offset()
        .checked_mul(2)
        .ok_or_else(|| candle_core::Error::Msg("text norm offset".into()))?;
    let count = l.shape().elem_count();
    if x.dtype() != DType::BF16
        || !l.is_contiguous()
        || count > MAX_VALUES
        || offset
            .checked_add(count * 2)
            .is_none_or(|end| end > x.buffer().length())
    {
        candle_core::bail!("text norm span");
    }
    Ok(offset)
}
fn grid(n: usize) -> MTLSize {
    MTLSize {
        width: n,
        height: 1,
        depth: 1,
    }
}
fn reduce(
    device: &candle_core::MetalDevice,
    pipeline: &ComputePipeline,
    input: &Buffer,
    output: &Buffer,
    p: Plan,
    divisor: f32,
) -> Result<()> {
    let encoder = device.command_encoder()?;
    encoder.set_compute_pipeline_state(pipeline);
    let enc: &ComputeCommandEncoder = encoder.as_ref();
    enc.set_input_buffer(0, Some(input), 0);
    enc.set_output_buffer(1, Some(output), 0);
    enc.set_bytes(2, &[p.rows, p.width, p.lanes, p.segments]);
    enc.set_bytes(3, &divisor);
    let groups = if p.chunk {
        (p.rows as usize * p.segments as usize).div_ceil(32 / p.lanes as usize)
    } else {
        p.rows as usize
    };
    enc.dispatch_thread_groups(grid(groups.div_ceil(8)), grid(256));
    Ok(())
}
struct Norm {
    eps: f32,
}
impl CustomOp2 for Norm {
    fn name(&self) -> &'static str {
        "bf16-text-native-rms"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("text native norm requires Metal")
    }
    fn metal_fwd(
        &self,
        x: &MetalStorage,
        xl: &Layout,
        w: &MetalStorage,
        wl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let xo = span(x, xl)?;
        let wo = span(w, wl)?;
        let width = *xl
            .dims()
            .last()
            .ok_or_else(|| candle_core::Error::Msg("text norm rank".into()))?;
        if width == 0 || w.device().id() != x.device().id() || wl.dims() != [width] {
            candle_core::bail!("text norm geometry");
        }
        let n = xl.shape().elem_count();
        let p = plan(n / width, width)
            .ok_or_else(|| candle_core::Error::Msg("text norm dimensions".into()))?;
        let dev = x.device();
        let pipes = pipelines(dev)?;
        let square = dev
            .new_buffer_builder()
            .with_size_for(n, DType::F32)
            .with_label("text-norm-square")
            .build()?;
        let mean = dev
            .new_buffer_builder()
            .with_size_for(p.rows as usize, DType::F32)
            .with_label("text-norm-mean")
            .build()?;
        let out = dev
            .new_buffer_builder()
            .with_size_for(n, DType::BF16)
            .with_label("text-norm-output")
            .build()?;
        {
            let encoder = dev.command_encoder()?;
            encoder.set_compute_pipeline_state(&pipes[0]);
            let e: &ComputeCommandEncoder = encoder.as_ref();
            e.set_input_buffer(0, Some(x.buffer()), xo);
            e.set_output_buffer(1, Some(&square), 0);
            e.set_bytes(2, &(n as u32));
            e.dispatch_threads(grid(n), grid(256));
        }
        if p.segments > 1 {
            let partial = dev
                .new_buffer_builder()
                .with_size_for(p.rows as usize * p.segments as usize, DType::F32)
                .with_label("text-norm-partials")
                .build()?;
            reduce(dev, &pipes[2], &square, &partial, p, 0.0)?;
            reduce(
                dev,
                &pipes[1],
                &partial,
                &mean,
                Plan {
                    width: p.segments,
                    lanes: 32,
                    segments: 1,
                    chunk: false,
                    ..p
                },
                width as f32,
            )?;
        } else {
            reduce(
                dev,
                &pipes[if p.chunk { 2 } else { 1 }],
                &square,
                &mean,
                p,
                width as f32,
            )?;
        }
        {
            let encoder = dev.command_encoder()?;
            encoder.set_compute_pipeline_state(&pipes[3]);
            let e: &ComputeCommandEncoder = encoder.as_ref();
            e.set_input_buffer(0, Some(x.buffer()), xo);
            e.set_input_buffer(1, Some(&mean), 0);
            e.set_input_buffer(2, Some(w.buffer()), wo);
            e.set_output_buffer(3, Some(&out), 0);
            e.set_bytes(4, &[p.rows, p.width]);
            e.set_bytes(5, &self.eps);
            e.dispatch_threads(grid(n), grid(256));
        }
        Ok((
            MetalStorage::new(out, dev.clone(), n, DType::BF16),
            xl.shape().clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use half::bf16;
    #[test]
    fn row_dispatch_and_limits() {
        assert!(!plan(511, 128).unwrap().chunk);
        assert_eq!(plan(512, 128).unwrap().lanes, 8);
        assert_eq!(plan(34, 4096).unwrap().segments, 64);
        assert_eq!(plan(511, 4096).unwrap().segments, 1);
        assert_eq!(plan(504, 4096).unwrap().segments, 17);
        assert!(!plan(512, 4096).unwrap().chunk);
        for (rows, width) in [(0, 128), (1, 128), (2, 127), (usize::MAX, 4096)] {
            assert!(plan(rows, width).is_none());
        }
    }
    #[test]
    fn views_f64_and_declines() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        if !super::super::text_linear_mps::supported_device(&device) {
            return Ok(());
        }
        for width in [128, 4096] {
            let values = [0.125f32, 0.375, 0.625, 0.875].repeat(width / 4);
            let weights = [0.75f32, 1.0, 1.25, 1.5].repeat(width / 4);
            let mean = values
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                / width as f64;
            let inv = (mean + f64::from(1e-6f32)).sqrt().recip();
            let expected = values
                .iter()
                .zip(&weights)
                .map(|(&x, &w)| {
                    bf16::from_f32(bf16::from_f64(f64::from(x) * inv).to_f32() * w).to_bits()
                })
                .collect::<Vec<_>>()
                .repeat(2);
            let x =
                Tensor::from_vec(values.repeat(2), (2, width), &device)?.to_dtype(DType::BF16)?;
            let w = Tensor::from_vec(weights.clone(), width, &device)?.to_dtype(DType::BF16)?;
            let interleaved = values
                .repeat(2)
                .iter()
                .flat_map(|&x| [x, 99.0])
                .collect::<Vec<_>>();
            let view = Tensor::from_vec(interleaved, (2, width, 2), &device)?
                .to_dtype(DType::BF16)?
                .narrow(2, 0, 1)?
                .squeeze(2)?;
            for input in [&x, &view] {
                let y = try_norm(input, &w, 1e-6)?
                    .unwrap()
                    .flatten_all()?
                    .to_vec1::<bf16>()?;
                assert_eq!(y.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), expected);
            }
            assert!(
                try_norm(
                    &x.to_device(&Device::Cpu)?,
                    &w.to_device(&Device::Cpu)?,
                    1e-6
                )?
                .is_none()
            );
            assert!(try_norm(&x.to_dtype(DType::F32)?, &w.to_dtype(DType::F32)?, 1e-6)?.is_none());
            assert!(try_norm(&x, &w, f64::NAN)?.is_none());
            assert!(try_norm(&x, &w, 0.0)?.is_none());
            assert!(try_norm(&x, &w.narrow(0, 0, width - 1)?, 1e-6)?.is_none());
        }
        Ok(())
    }
}
