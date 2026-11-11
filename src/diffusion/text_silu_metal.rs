//! BF16 text SiLU with precise F32 exponential and one final BF16 rounding.
use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{CpuStorage, CustomOp1, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline};
use objc2_metal::{MTLCompileOptions, MTLMathFloatingPointFunctions, MTLMathMode, MTLSize};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};
const SOURCE: &str = include_str!("text_silu_metal.metal");
const MAX_VALUES: usize = 4 * 1024 * 1024;
const MAX_CHUNKED_VALUES: usize = 16 * 1024 * 1024;
fn admitted(n: usize) -> bool {
    n > 0 && n <= MAX_VALUES
}

fn chunked_admitted(n: usize) -> bool {
    n > 0 && n <= MAX_CHUNKED_VALUES
}

// Image gates can exceed one text-sized dispatch. Bound the whole BF16 input
// and output to 32 MiB each before copying; keep every F32 dispatch at the
// unchanged four-million-value limit. The scalar formula is unchanged.
pub(super) fn try_silu_chunked(x: &Tensor) -> Result<Option<Tensor>> {
    if x.dtype() != DType::BF16
        || !(1..=4).contains(&x.rank())
        || !chunked_admitted(x.elem_count())
        || !super::text_linear_mps::supported_device(x.device())
    {
        return Ok(None);
    }
    if admitted(x.elem_count()) {
        return try_silu(x);
    }
    let flat = x.contiguous()?.flatten_all()?;
    let mut chunks = Vec::with_capacity(x.elem_count().div_ceil(MAX_VALUES));
    for start in (0..x.elem_count()).step_by(MAX_VALUES) {
        let chunk = flat.narrow(0, start, MAX_VALUES.min(x.elem_count() - start))?;
        let output = try_silu(&chunk)?
            .ok_or_else(|| candle_core::Error::Msg("admitted BF16 SiLU chunk declined".into()))?;
        chunks.push(output);
    }
    Ok(Some(Tensor::cat(&chunks, 0)?.reshape(x.dims())?))
}
static PIPELINES: OnceLock<Mutex<HashMap<DeviceId, ComputePipeline>>> = OnceLock::new();
fn pipeline(device: &candle_core::MetalDevice) -> Result<ComputePipeline> {
    let mut cache = PIPELINES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| candle_core::Error::Msg("text silu pipeline lock".into()))?;
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
    let f = lib
        .get_function("silu_precise", None)
        .map_err(candle_core::Error::wrap)?;
    let p = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&f)
        .map_err(candle_core::Error::wrap)?;
    if cache.len() >= 16 {
        cache.clear();
    }
    cache.insert(device.id(), p.clone());
    Ok(p)
}
pub(super) fn try_silu(x: &Tensor) -> Result<Option<Tensor>> {
    if x.dtype() != DType::BF16
        || !(1..=4).contains(&x.rank())
        || !super::text_linear_mps::supported_device(x.device())
    {
        return Ok(None);
    }
    if !admitted(x.elem_count()) {
        return Ok(None);
    }
    // The input stays BF16; evaluate the scalar expression in F32 and round once.
    let y = x
        .contiguous()?
        .to_dtype(DType::F32)?
        .apply_op1_no_bwd(&Silu)?;
    Ok(Some(y.to_dtype(DType::BF16)?))
}
struct Silu;
fn grid(n: usize) -> MTLSize {
    MTLSize {
        width: n,
        height: 1,
        depth: 1,
    }
}
impl CustomOp1 for Silu {
    fn name(&self) -> &'static str {
        "bf16-text-native-silu"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("text native silu requires Metal")
    }
    fn metal_fwd(&self, x: &MetalStorage, l: &Layout) -> Result<(MetalStorage, Shape)> {
        let n = l.shape().elem_count();
        let offset = l
            .start_offset()
            .checked_mul(4)
            .ok_or_else(|| candle_core::Error::Msg("text silu offset".into()))?;
        if x.dtype() != DType::F32
            || !l.is_contiguous()
            || !admitted(n)
            || offset
                .checked_add(n * 4)
                .is_none_or(|end| end > x.buffer().length())
        {
            candle_core::bail!("text silu span");
        }
        let dev = x.device();
        let p = pipeline(dev)?;
        let out = dev
            .new_buffer_builder()
            .with_size_for(n, DType::F32)
            .with_label("text-silu-f32")
            .build()?;
        {
            let encoder = dev.command_encoder()?;
            encoder.set_compute_pipeline_state(&p);
            let e: &ComputeCommandEncoder = encoder.as_ref();
            e.set_input_buffer(0, Some(x.buffer()), offset);
            e.set_output_buffer(1, Some(&out), 0);
            e.set_bytes(2, &[n as u32]);
            e.dispatch_thread_groups(grid(n.div_ceil(256)), grid(256));
        }
        Ok((
            MetalStorage::new(out, dev.clone(), n, DType::F32),
            l.shape().clone(),
        ))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    #[test]
    fn bounds() {
        assert!(!admitted(0));
        assert!(admitted(1));
        assert!(admitted(MAX_VALUES));
        assert!(!admitted(MAX_VALUES + 1));
        assert!(!admitted(usize::MAX));
    }
    #[test]
    fn views_f64_and_declines() -> Result<()> {
        assert!(try_silu(&Tensor::zeros((2, 3), DType::BF16, &Device::Cpu)?)?.is_none());
        let Ok(dev) = Device::new_metal(0) else {
            assert!(std::env::var_os("DINF_REQUIRE_METAL_TESTS").is_none());
            return Ok(());
        };
        if !super::super::text_linear_mps::supported_device(&dev) {
            return Ok(());
        }
        assert!(try_silu(&Tensor::zeros((2, 3), DType::F32, &dev)?)?.is_none());
        let vals = (0..257)
            .map(|i| (i as f32 - 128.0) / 16.0)
            .collect::<Vec<_>>();
        let packed = vals.iter().flat_map(|v| [*v, 0.0]).collect::<Vec<_>>();
        let x = Tensor::from_vec(packed, (257, 2), &dev)?
            .to_dtype(DType::BF16)?
            .narrow(1, 0, 1)?
            .squeeze(1)?;
        let y = try_silu(&x)?
            .unwrap()
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        for (x, y) in vals.iter().zip(y) {
            let x = *x as f64;
            let want = x / (1.0 + (-x).exp());
            assert!((y as f64 - want).abs() <= want.abs() / 256.0 + 2f64.powi(-133));
        }
        Ok(())
    }
}

#[cfg(test)]
mod chunked_tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn whole_bounds_and_declines() -> Result<()> {
        assert!(!chunked_admitted(0));
        assert!(chunked_admitted(MAX_VALUES));
        assert!(chunked_admitted(MAX_CHUNKED_VALUES));
        assert!(!chunked_admitted(MAX_CHUNKED_VALUES + 1));
        assert!(!chunked_admitted(usize::MAX));
        assert!(try_silu_chunked(&Tensor::zeros((2, 3), DType::BF16, &Device::Cpu)?)?.is_none());
        let Some(dev) = crate::test_metal_device() else {
            return Ok(());
        };
        assert!(try_silu_chunked(&Tensor::zeros((2, 3), DType::F32, &dev)?)?.is_none());
        assert!(try_silu_chunked(&Tensor::zeros((), DType::BF16, &dev)?)?.is_none());
        assert!(
            try_silu_chunked(&Tensor::zeros(MAX_CHUNKED_VALUES + 1, DType::BF16, &dev)?)?.is_none()
        );
        Ok(())
    }

    #[test]
    fn strided_views_match_native_rounding() -> Result<()> {
        let Some(dev) = crate::test_metal_device() else {
            return Ok(());
        };
        if !super::super::text_linear_mps::supported_device(&dev) {
            return Ok(());
        }
        let values = [5.9375, -87.5, -88.0, -88.5, 0.0, 1.0];
        let input = Tensor::from_vec(
            values.iter().flat_map(|x| [123.0_f32, *x]).collect(),
            (values.len(), 2),
            &dev,
        )?
        .to_dtype(DType::BF16)?
        .narrow(1, 1, 1)?
        .t()?;
        let output = try_silu_chunked(&input)?.unwrap();
        assert_eq!(output.dims(), input.dims());
        assert_eq!(output.dtype(), DType::BF16);
        let expected = try_silu(&input)?
            .unwrap()
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let actual = output
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual, expected);
        assert_eq!(actual[0], 5.90625);
        Ok(())
    }

    #[test]
    fn four_million_boundary_keeps_every_value() -> Result<()> {
        let Some(dev) = crate::test_metal_device() else {
            return Ok(());
        };
        if !super::super::text_linear_mps::supported_device(&dev) {
            return Ok(());
        }
        for n in [MAX_VALUES - 1, MAX_VALUES, MAX_VALUES + 1] {
            let mut values = vec![0.0_f32; n];
            values[0] = 5.9375;
            values[n - 1] = 5.9375;
            if n > MAX_VALUES {
                values[MAX_VALUES - 1] = 5.9375;
            }
            let input = Tensor::from_vec(values.clone(), n, &dev)?.to_dtype(DType::BF16)?;
            let output = try_silu_chunked(&input)?
                .unwrap()
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            for (x, y) in values.iter().zip(output) {
                assert_eq!(y, if *x == 0.0 { 0.0 } else { 5.90625 });
            }
        }
        Ok(())
    }
}
