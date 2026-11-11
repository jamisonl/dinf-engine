//! Per-row form of the pinned native Metal QKV arithmetic.
//! `native_rows` selects the request-wide native microbatch policy, not the
//! size of this tensor call. This component does not choose that policy.
use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{CpuStorage, CustomOp2, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
const SOURCE: &str = include_str!("native_qkv.metal");
static PIPELINES: OnceLock<Mutex<HashMap<(DeviceId, &'static str), ComputePipeline>>> =
    OnceLock::new();
fn pipeline(device: &candle_core::MetalDevice, name: &'static str) -> Result<ComputePipeline> {
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| candle_core::Error::Msg("QKV pipeline lock poisoned".into()))?;
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
pub enum Kind {
    Q4,
    Q8,
    Q6,
}
impl Kind {
    fn block(self) -> (usize, usize) {
        match self {
            Self::Q4 => (256, 144),
            Self::Q8 => (32, 34),
            Self::Q6 => (256, 210),
        }
    }
    fn dispatch(self, rows: usize) -> (&'static str, usize, usize) {
        match (self, rows) {
            (Self::Q4, 1..=3) => ("q4_m1", 4, 64),
            (Self::Q4, _) => ("q4_ext8", 8, 64),
            (Self::Q6, 1..=3) => ("q6_m1", 4, 64),
            (Self::Q6, _) => ("q6_ext8", 8, 64),
            (Self::Q8, 1) => ("q8_m1", 2, 128),
            (Self::Q8, 2) => ("q8_ext16", 4, 64),
            (Self::Q8, _) => ("q8_ext8", 8, 64),
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
        "qwen-native-metal-qkv"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("Metal QKV requires Metal")
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
            .ok_or_else(|| candle_core::Error::Msg("QKV weight overflow".into()))?;
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
            candle_core::bail!("QKV storage/layout mismatch")
        }
        let we = wl
            .start_offset()
            .checked_add(expected)
            .ok_or_else(|| candle_core::Error::Msg("QKV weight extent".into()))?;
        let xe = il
            .start_offset()
            .checked_add(il.shape().elem_count())
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| candle_core::Error::Msg("QKV input extent".into()))?;
        if we > weights.buffer().length() || xe > input.buffer().length() {
            candle_core::bail!("QKV storage span")
        }
        let count = rows
            .checked_mul(self.n)
            .ok_or_else(|| candle_core::Error::Msg("QKV output overflow".into()))?;
        if count > i32::MAX as usize || il.shape().elem_count() > i32::MAX as usize {
            candle_core::bail!("QKV shader range")
        }
        let device = input.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(count, DType::F32)
            .with_label("qwen-native-qkv")
            .build()?;
        let p = pipeline(device, name)?;
        let encoder = device.command_encoder()?;
        encoder.set_label("qwen-native-qkv");
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
        || !matches!(n, 512 | 4096)
        || input.rank() < 2
        || input.dims().last() != Some(&k)
        || !weights.device().is_metal()
        || !input.device().is_metal()
    {
        candle_core::bail!("QKV component admission")
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

// Keep packed bytes once at construction. The public QTensor API copies the
// packed Metal storage; this avoids a model-wide F32 expansion or private access.
pub(super) struct NativeQkv {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    v_kind: Kind,
}

impl NativeQkv {
    /// Packed Q, K and V weights (resident GGML rows), for sharing with the
    /// batched prompt kernels.
    pub(super) fn packed(&self) -> [&Tensor; 3] {
        [&self.q, &self.k, &self.v]
    }
}
fn shape_types(
    q: &candle_core::quantized::QTensor,
    k: &candle_core::quantized::QTensor,
    v: &candle_core::quantized::QTensor,
) -> bool {
    use candle_core::quantized::GgmlDType;
    q.dtype() == GgmlDType::Q4K
        && k.dtype() == GgmlDType::Q8_0
        && matches!(v.dtype(), GgmlDType::Q4K | GgmlDType::Q6K)
        && q.shape().dims() == [4096, 2048]
        && k.shape().dims() == [512, 2048]
        && v.shape().dims() == [512, 2048]
}
pub(super) fn supports(
    q: &crate::qmatmul::QMatMul,
    k: &crate::qmatmul::QMatMul,
    v: &crate::qmatmul::QMatMul,
) -> bool {
    use crate::qmatmul::QMatMul;
    match (q, k, v) {
        (QMatMul::QTensor(q), QMatMul::QTensor(k), QMatMul::QTensor(v)) => {
            shape_types(q, k, v)
                && q.device().is_metal()
                && q.device().same_device(&k.device())
                && q.device().same_device(&v.device())
        }
        _ => false,
    }
}
impl NativeQkv {
    pub(super) fn new(
        q: &candle_core::quantized::QTensor,
        k: &candle_core::quantized::QTensor,
        v: &candle_core::quantized::QTensor,
        native_cache: bool,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Option<Self>> {
        if !native_cache
            || heads != 32
            || kv_heads != 4
            || head_dim != 128
            || !shape_types(q, k, v)
            || !q.device().is_metal()
            || !q.device().same_device(&k.device())
            || !q.device().same_device(&v.device())
        {
            return Ok(None);
        }
        let packed = |w: &candle_core::quantized::QTensor| -> Result<Tensor> {
            let bytes = w.data()?;
            Tensor::from_slice(bytes.as_ref(), bytes.len(), &w.device())
        };
        Ok(Some(Self {
            q: packed(q)?,
            k: packed(k)?,
            v: packed(v)?,
            v_kind: if v.dtype() == candle_core::quantized::GgmlDType::Q4K {
                Kind::Q4
            } else {
                Kind::Q6
            },
        }))
    }
    pub(super) fn forward(
        &self,
        input: &Tensor,
        start: usize,
        policy: super::super::TextArithmeticPolicy,
    ) -> Result<Option<(Tensor, Tensor, Tensor)>> {
        if input.dtype() != DType::F32
            || !input.device().is_metal()
            || !input.device().same_device(self.q.device())
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
            .ok_or_else(|| candle_core::Error::Msg("QKV position overflow".into()))?;
        let mut output = [Vec::new(), Vec::new(), Vec::new()];
        let mut offset = 0;
        while offset < rows {
            let family = policy.qwen_metal_rows_at(start + offset).unwrap();
            let mut end = offset + 1;
            while end < rows && policy.qwen_metal_rows_at(start + end) == Some(family) {
                end += 1;
            }
            let x = input.narrow(1, offset, end - offset)?;
            output[0].push(project(&self.q, &x, Kind::Q4, 2048, 4096, family)?);
            output[1].push(project(&self.k, &x, Kind::Q8, 2048, 512, family)?);
            output[2].push(project(&self.v, &x, self.v_kind, 2048, 512, family)?);
            offset = end;
        }
        Ok(Some((
            Tensor::cat(&output[0], 1)?,
            Tensor::cat(&output[1], 1)?,
            Tensor::cat(&output[2], 1)?,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qmatmul::QMatMul;
    use candle_core::Device;
    use candle_core::quantized::{GgmlDType, ggml_file};
    use std::sync::Arc;
    fn weight(dtype: GgmlDType, n: usize) -> Result<candle_core::quantized::QTensor> {
        let bytes = vec![0u8; n * 2048 / dtype.block_size() * dtype.type_size()];
        ggml_file::qtensor_from_ggml(dtype, &bytes, vec![n, 2048], &Device::Cpu)
    }
    #[test]
    fn native_qkv_checks_stored_types_shapes_and_device() -> Result<()> {
        let q = weight(GgmlDType::Q4K, 4096)?;
        let k = weight(GgmlDType::Q8_0, 512)?;
        let v = weight(GgmlDType::Q6K, 512)?;
        assert!(shape_types(&q, &k, &v));
        assert!(NativeQkv::new(&q, &k, &v, true, 32, 4, 128)?.is_none());
        assert!(!shape_types(&weight(GgmlDType::Q8_0, 4096)?, &k, &v));
        assert!(!shape_types(&q, &weight(GgmlDType::Q4K, 512)?, &v));
        assert!(!shape_types(&q, &k, &weight(GgmlDType::Q5K, 512)?));
        assert!(!shape_types(&weight(GgmlDType::Q4K, 512)?, &k, &v));
        assert!(!supports(
            &QMatMul::QTensor(Arc::new(q)),
            &QMatMul::QTensor(Arc::new(k)),
            &QMatMul::QTensor(Arc::new(v))
        ));
        Ok(())
    }
}
