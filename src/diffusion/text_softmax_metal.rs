//! BF16 text softmax with the pinned native F32 row reduction.
use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{CpuStorage, CustomOp1, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline};
use objc2_metal::{MTLCompileOptions, MTLMathFloatingPointFunctions, MTLMathMode, MTLSize};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};
const SOURCE: &str = include_str!("text_softmax_metal.metal");
const MAX_VALUES: usize = 4 * 1024 * 1024;
fn admitted(rows: usize, cols: usize) -> bool {
    rows > 0
        && cols > 0
        && cols <= 2048
        && (cols <= 1024 || rows >= 128)
        && rows.checked_mul(cols).is_some_and(|n| n <= MAX_VALUES)
}
static PIPELINES: OnceLock<Mutex<HashMap<DeviceId, ComputePipeline>>> = OnceLock::new();
fn pipeline(device: &candle_core::MetalDevice) -> Result<ComputePipeline> {
    let mut cache = PIPELINES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| candle_core::Error::Msg("text softmax pipeline lock".into()))?;
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
        .get_function("softmax_row_f32", None)
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
pub(super) fn try_softmax(x: &Tensor) -> Result<Option<Tensor>> {
    if x.dtype() != DType::BF16
        || !(2..=4).contains(&x.rank())
        || !super::text_linear_mps::supported_device(x.device())
    {
        return Ok(None);
    }
    let cols = *x.dims().last().unwrap();
    if cols == 0 || !admitted(x.elem_count() / cols, cols) {
        return Ok(None);
    }
    // Keep the native storage boundaries: BF16 scores -> F32 softmax -> BF16 probabilities.
    let y = x
        .contiguous()?
        .to_dtype(DType::F32)?
        .apply_op1_no_bwd(&Softmax)?;
    Ok(Some(y.to_dtype(DType::BF16)?))
}
struct Softmax;
fn grid(n: usize) -> MTLSize {
    MTLSize {
        width: n,
        height: 1,
        depth: 1,
    }
}
impl CustomOp1 for Softmax {
    fn name(&self) -> &'static str {
        "bf16-text-native-softmax"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("text native softmax requires Metal")
    }
    fn metal_fwd(&self, x: &MetalStorage, l: &Layout) -> Result<(MetalStorage, Shape)> {
        let n = l.shape().elem_count();
        let cols = *l
            .dims()
            .last()
            .ok_or_else(|| candle_core::Error::Msg("text softmax rank".into()))?;
        let offset = l
            .start_offset()
            .checked_mul(4)
            .ok_or_else(|| candle_core::Error::Msg("text softmax offset".into()))?;
        if x.dtype() != DType::F32
            || !l.is_contiguous()
            || cols == 0
            || !admitted(n / cols, cols)
            || offset
                .checked_add(n * 4)
                .is_none_or(|end| end > x.buffer().length())
        {
            candle_core::bail!("text softmax span");
        }
        let dev = x.device();
        let p = pipeline(dev)?;
        let out = dev
            .new_buffer_builder()
            .with_size_for(n, DType::F32)
            .with_label("text-softmax-f32")
            .build()?;
        {
            let encoder = dev.command_encoder()?;
            encoder.set_compute_pipeline_state(&p);
            let e: &ComputeCommandEncoder = encoder.as_ref();
            e.set_input_buffer(0, Some(x.buffer()), offset);
            e.set_output_buffer(1, Some(&out), 0);
            e.set_bytes(2, &[(n / cols) as u32, cols as u32]);
            e.dispatch_thread_groups(grid((n / cols).div_ceil(8)), grid(256));
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
    fn row_scope_and_limits() {
        assert!(admitted(1, 1));
        assert!(admitted(1, 1024));
        assert!(!admitted(127, 1025));
        assert!(admitted(128, 2048));
        assert!(!admitted(128, 2049));
        assert!(!admitted(0, 1));
        assert!(!admitted(1, 0));
        assert!(!admitted(usize::MAX, 2));
        assert!(!admitted(MAX_VALUES, 2));
    }
    #[test]
    fn views_f64_and_declines() -> Result<()> {
        let cpu = Tensor::zeros((2, 34), DType::BF16, &Device::Cpu)?;
        assert!(try_softmax(&cpu)?.is_none());
        let Ok(dev) = Device::new_metal(0) else {
            assert!(std::env::var_os("DINF_REQUIRE_METAL_TESTS").is_none());
            return Ok(());
        };
        let values = (0..68).map(|i| (i as f32 - 33.0) / 8.0).collect::<Vec<_>>();
        let packed = values.iter().flat_map(|v| [*v, 0.0]).collect::<Vec<_>>();
        let x = Tensor::from_vec(packed, (2, 34, 2), &dev)?
            .to_dtype(DType::BF16)?
            .narrow(2, 0, 1)?
            .squeeze(2)?;
        let y = try_softmax(&x)?
            .unwrap()
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        for (row, actual) in y.iter().enumerate() {
            let input = &values[row * 34..(row + 1) * 34];
            let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let exp = input
                .iter()
                .map(|&v| (v as f64 - max).exp())
                .collect::<Vec<_>>();
            let sum: f64 = exp.iter().sum();
            for (&v, e) in actual.iter().zip(exp) {
                let expected = e / sum;
                assert!((v as f64 - expected).abs() <= expected * 0.004 + 1e-7);
            }
        }
        assert!(try_softmax(&x.to_dtype(DType::F32)?)?.is_none());
        assert!(try_softmax(&Tensor::zeros((2, 1025), DType::BF16, &dev)?)?.is_none());
        Ok(())
    }
}
