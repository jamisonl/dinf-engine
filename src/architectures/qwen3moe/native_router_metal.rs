//! Per-row form of the pinned native Metal F32 router arithmetic.
//! `native_rows` selects the request-wide native microbatch policy, not the
//! size of this tensor call. This component does not choose that policy.
use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{CpuStorage, CustomOp2, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
const SOURCE: &str = include_str!("native_router_metal.metal");
static PIPELINES: OnceLock<Mutex<HashMap<(DeviceId, &'static str), ComputePipeline>>> =
    OnceLock::new();
fn pipeline(device: &candle_core::MetalDevice, name: &'static str) -> Result<ComputePipeline> {
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| candle_core::Error::Msg("F32 router pipeline lock poisoned".into()))?;
    if let Some(p) = cache.get(&(device.id(), name)) {
        return Ok(p.clone());
    }
    let library = device
        .metal_device()
        .new_library_with_source(SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function(name, None)
        .map_err(candle_core::Error::wrap)?;
    let p = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    cache.insert((device.id(), name), p.clone());
    Ok(p)
}
#[derive(Clone, Copy)]
enum Kind {
    F32,
}
impl Kind {
    fn block(self) -> (usize, usize) {
        match self {
            Self::F32 => (1, 4),
        }
    }
    fn dispatch(self, rows: usize) -> (&'static str, usize, usize) {
        match (self, rows) {
            (Self::F32, 1) => ("router_m1", 2, 128),
            (Self::F32, 2) => ("router_ext16", 4, 64),
            (Self::F32, _) => ("router_ext8", 8, 64),
        }
    }
}
struct Projection {
    kind: Kind,
    k: usize,
    n: usize,
    native_rows: usize,
}
impl CustomOp2 for Projection {
    fn name(&self) -> &'static str {
        "qwen-native-metal-router"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("Metal F32 router requires Metal")
    }
    fn metal_fwd(
        &self,
        weights: &MetalStorage,
        wl: &Layout,
        input: &MetalStorage,
        il: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (block, bytes) = self.kind.block();
        let expected = self
            .k
            .checked_div(block)
            .and_then(|v| v.checked_mul(bytes))
            .and_then(|v| v.checked_mul(self.n))
            .ok_or_else(|| candle_core::Error::Msg("F32 router weight overflow".into()))?;
        let rows = il.shape().elem_count() / self.k;
        let (name, outputs_per_group, threads) = self.kind.dispatch(self.native_rows);
        if weights.dtype() != DType::U8
            || input.dtype() != DType::F32
            || weights.device().id() != input.device().id()
            || !wl.is_contiguous()
            || !il.is_contiguous()
            || wl.shape().elem_count() != expected
            || rows == 0
            || !self.n.is_multiple_of(outputs_per_group)
        {
            candle_core::bail!("F32 router storage/layout mismatch")
        }
        let we = wl
            .start_offset()
            .checked_add(expected)
            .ok_or_else(|| candle_core::Error::Msg("F32 router weight extent".into()))?;
        let xe = il
            .start_offset()
            .checked_add(il.shape().elem_count())
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| candle_core::Error::Msg("F32 router input extent".into()))?;
        if we > weights.buffer().length() || xe > input.buffer().length() {
            candle_core::bail!("F32 router storage span")
        }
        let count = rows
            .checked_mul(self.n)
            .ok_or_else(|| candle_core::Error::Msg("F32 router output overflow".into()))?;
        if count > i32::MAX as usize || il.shape().elem_count() > i32::MAX as usize {
            candle_core::bail!("F32 router shader range")
        }
        let device = input.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(count, DType::F32)
            .with_label("qwen-native-router")
            .build()?;
        let p = pipeline(device, name)?;
        let encoder = device.command_encoder()?;
        encoder.set_label("qwen-native-router");
        encoder.set_compute_pipeline_state(&p);
        let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        enc.set_input_buffer(0, Some(weights.buffer()), wl.start_offset());
        enc.set_input_buffer(1, Some(input.buffer()), il.start_offset() * 4);
        enc.set_output_buffer(2, Some(&output), 0);
        enc.set_bytes(3, &(self.k as u32));
        enc.set_bytes(4, &(self.n as u32));
        enc.set_bytes(5, &(rows as u32));
        enc.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: self.n / outputs_per_group,
                height: rows,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        let mut shape = il.shape().dims().to_vec();
        *shape.last_mut().unwrap() = self.n;
        Ok((
            MetalStorage::new(output, device.clone(), count, DType::F32),
            Shape::from(shape),
        ))
    }
}
fn project(
    weights: &Tensor,
    input: &Tensor,
    kind: Kind,
    k: usize,
    n: usize,
    native_rows: usize,
) -> Result<Tensor> {
    if !(1..=8).contains(&native_rows)
        || k != 2048
        || n != 128
        || input.rank() < 2
        || input.dims().last() != Some(&k)
        || !weights.device().is_metal()
        || !input.device().is_metal()
    {
        candle_core::bail!("F32 router component admission")
    }
    let weights = weights.contiguous()?;
    let input = input.contiguous()?;
    weights.apply_op2_no_bwd(
        &input,
        &Projection {
            kind,
            k,
            n,
            native_rows,
        },
    )
}

// Keep the stored F32 router bytes once for the native Metal row policy.
pub(super) struct NativeRouter {
    weight: Tensor,
}
fn shape_type(weight: &candle_core::quantized::QTensor) -> bool {
    weight.dtype() == candle_core::quantized::GgmlDType::F32 && weight.shape().dims() == [128, 2048]
}
impl NativeRouter {
    pub(super) fn new(
        weight: &candle_core::quantized::QTensor,
        qkv_admitted: bool,
        native_cache: bool,
    ) -> Result<Option<Self>> {
        if !qkv_admitted || !native_cache || !shape_type(weight) || !weight.device().is_metal() {
            return Ok(None);
        }
        let bytes = weight.data()?;
        Ok(Some(Self {
            weight: Tensor::from_slice(bytes.as_ref(), bytes.len(), &weight.device())?,
        }))
    }
    pub(super) fn forward(
        &self,
        input: &Tensor,
        start: usize,
        policy: super::super::TextArithmeticPolicy,
    ) -> Result<Option<Tensor>> {
        if input.dtype() != DType::F32
            || !input.device().is_metal()
            || !input.device().same_device(self.weight.device())
            || input.rank() != 3
            || input.dim(2)? != 2048
            || policy.qwen_metal_rows_at(start).is_none()
        {
            return Ok(None);
        }
        let (batch, rows, _) = input.dims3()?;
        if batch == 0 || rows == 0 {
            return Ok(None);
        }
        start
            .checked_add(rows)
            .ok_or_else(|| candle_core::Error::Msg("Metal router position overflow".into()))?;
        let mut output = Vec::new();
        let mut offset = 0;
        while offset < rows {
            let family = policy.qwen_metal_rows_at(start + offset).unwrap();
            let mut end = offset + 1;
            while end < rows && policy.qwen_metal_rows_at(start + end) == Some(family) {
                end += 1;
            }
            let x = input.narrow(1, offset, end - offset)?;
            output.push(project(&self.weight, &x, Kind::F32, 2048, 128, family)?);
            offset = end;
        }
        Ok(Some(Tensor::cat(&output, 1)?))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_core::quantized::{GgmlDType, ggml_file};
    #[test]
    fn native_metal_router_checks_stored_type_device_and_admission() -> Result<()> {
        let weight = |dtype: GgmlDType, n: usize| {
            let bytes = vec![0; n * 2048 / dtype.block_size() * dtype.type_size()];
            ggml_file::qtensor_from_ggml(dtype, &bytes, vec![n, 2048], &Device::Cpu)
        };
        let w = weight(GgmlDType::F32, 128)?;
        assert!(shape_type(&w));
        assert!(NativeRouter::new(&w, true, true)?.is_none());
        assert!(NativeRouter::new(&w, false, true)?.is_none());
        assert!(NativeRouter::new(&w, true, false)?.is_none());
        assert!(!shape_type(&weight(GgmlDType::BF16, 128)?));
        assert!(!shape_type(&weight(GgmlDType::F32, 64)?));
        Ok(())
    }
}
