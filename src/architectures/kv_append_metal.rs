use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{CpuStorage, DType, InplaceOp2, Layout, MetalStorage, Result, Storage, Tensor};
use candle_metal_kernels::metal::ComputePipeline;

const KV_APPEND_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void dinf_kv_cast_append_f32_f16(
    device const float *source [[buffer(0)]],
    device half *destination [[buffer(1)]],
    constant uint &elements [[buffer(2)]],
    constant uint &source_row_width [[buffer(3)]],
    constant uint &destination_row_width [[buffer(4)]],
    constant uint &destination_column [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= elements) {
    return;
  }
  const uint row = gid / source_row_width;
  const uint column = gid - row * source_row_width;
  destination[row * destination_row_width + destination_column + column] = half(source[gid]);
}

kernel void dinf_kv_ragged_cast_append_f32_f16(
    device const float *source [[buffer(0)]],
    device half *destination [[buffer(1)]],
    constant uint *positions [[buffer(2)]],
    constant uint &elements [[buffer(3)]],
    constant uint &heads [[buffer(4)]],
    constant uint &capacity [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= elements) {
    return;
  }
  const uint lane_width = heads * head_dim;
  const uint lane = gid / lane_width;
  const uint lane_column = gid - lane * lane_width;
  const uint head = lane_column / head_dim;
  const uint column = lane_column - head * head_dim;
  const uint destination_index =
      ((lane * heads + head) * capacity + positions[lane]) * head_dim + column;
  destination[destination_index] = half(source[gid]);
}

"#;

#[derive(Clone)]
struct KvAppendPipelines {
    contiguous: ComputePipeline,
    ragged: ComputePipeline,
}

static PIPELINES: OnceLock<Mutex<HashMap<DeviceId, KvAppendPipelines>>> = OnceLock::new();

fn pipelines(device: &candle_core::MetalDevice) -> Result<KvAppendPipelines> {
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = device.id();
    if let Some(pipeline) = pipelines.lock().unwrap().get(&key) {
        return Ok(pipeline.clone());
    }
    let library = device
        .metal_device()
        .new_library_with_source(KV_APPEND_MSL, None)
        .map_err(candle_core::Error::wrap)?;
    let contiguous_function = library
        .get_function("dinf_kv_cast_append_f32_f16", None)
        .map_err(candle_core::Error::wrap)?;
    let ragged_function = library
        .get_function("dinf_kv_ragged_cast_append_f32_f16", None)
        .map_err(candle_core::Error::wrap)?;
    let compiled = KvAppendPipelines {
        contiguous: device
            .metal_device()
            .new_compute_pipeline_state_with_function(&contiguous_function)
            .map_err(candle_core::Error::wrap)?,
        ragged: device
            .metal_device()
            .new_compute_pipeline_state_with_function(&ragged_function)
            .map_err(candle_core::Error::wrap)?,
    };
    let mut cache = pipelines.lock().unwrap();
    Ok(cache.entry(key).or_insert_with(|| compiled.clone()).clone())
}

const MAX_INLINE_POSITION_BYTES: usize = 4 * 1024;

#[derive(Clone)]
struct RaggedCastAppend {
    elements: u32,
    heads: u32,
    capacity: u32,
    head_dim: u32,
    positions: Vec<u32>,
    source_shape: [usize; 4],
    destination_shape: [usize; 4],
}

impl RaggedCastAppend {
    fn plan(destination: &Tensor, source: &Tensor, positions: &[usize]) -> Option<Self> {
        if destination.dtype() != DType::F16
            || source.dtype() != DType::F32
            || !destination.is_contiguous()
            || !source.is_contiguous()
        {
            return None;
        }
        let (batch, heads, capacity, head_dim) = destination.dims4().ok()?;
        let (source_batch, source_heads, incoming, source_head_dim) = source.dims4().ok()?;
        Self::plan_geometry(
            [batch, heads, capacity, head_dim],
            [source_batch, source_heads, incoming, source_head_dim],
            positions,
        )
    }

    fn plan_geometry(
        destination_shape: [usize; 4],
        source_shape: [usize; 4],
        positions: &[usize],
    ) -> Option<Self> {
        let [batch, heads, capacity, head_dim] = destination_shape;
        let [source_batch, source_heads, incoming, source_head_dim] = source_shape;
        if batch == 0
            || heads == 0
            || capacity == 0
            || head_dim == 0
            || incoming != 1
            || (batch, heads, head_dim) != (source_batch, source_heads, source_head_dim)
            || positions.len() != batch
            || positions.len().checked_mul(std::mem::size_of::<u32>())? > MAX_INLINE_POSITION_BYTES
            || positions.iter().any(|position| *position >= capacity)
        {
            return None;
        }
        // Positions are lane-ordered. Repeated values are valid because each
        // lane occupies a disjoint batch slice of the packed destination.
        let lane_width = heads.checked_mul(head_dim)?;
        let elements = batch.checked_mul(lane_width)?;
        let destination_elements = batch
            .checked_mul(heads)?
            .checked_mul(capacity)?
            .checked_mul(head_dim)?;
        let _: u32 = lane_width.try_into().ok()?;
        let _: u32 = destination_elements.try_into().ok()?;
        Some(Self {
            elements: elements.try_into().ok()?,
            heads: heads.try_into().ok()?,
            capacity: capacity.try_into().ok()?,
            head_dim: head_dim.try_into().ok()?,
            positions: positions
                .iter()
                .map(|position| (*position).try_into().ok())
                .collect::<Option<Vec<_>>>()?,
            source_shape,
            destination_shape,
        })
    }

    fn validate_storage(
        &self,
        destination: &MetalStorage,
        destination_layout: &Layout,
        source: &MetalStorage,
        source_layout: &Layout,
    ) -> Result<()> {
        if destination.dtype() != DType::F16
            || source.dtype() != DType::F32
            || !destination_layout.is_contiguous()
            || !source_layout.is_contiguous()
            || destination_layout.shape().dims() != self.destination_shape
            || source_layout.shape().dims() != self.source_shape
        {
            candle_core::bail!("ragged Metal KV append tensor metadata changed after validation")
        }
        validate_buffer_span(destination, destination_layout)?;
        validate_buffer_span(source, source_layout)
    }
}

impl InplaceOp2 for RaggedCastAppend {
    fn name(&self) -> &'static str {
        "dinf-metal-ragged-kv-cast-append"
    }

    fn cpu_fwd(
        &self,
        _destination: &mut CpuStorage,
        _destination_layout: &Layout,
        _source: &CpuStorage,
        _source_layout: &Layout,
    ) -> Result<()> {
        candle_core::bail!("ragged Metal KV append cannot run on CPU storage")
    }

    fn metal_fwd(
        &self,
        destination: &mut MetalStorage,
        destination_layout: &Layout,
        source: &MetalStorage,
        source_layout: &Layout,
    ) -> Result<()> {
        self.validate_storage(destination, destination_layout, source, source_layout)?;
        let device = destination.device();
        if source.device().id() != device.id() {
            candle_core::bail!("ragged Metal KV append tensors are on different devices")
        }

        let pipeline = pipelines(device)?.ragged;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-ragged-kv-cast-append");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(source.buffer()),
            source_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(
            1,
            Some(destination.buffer()),
            destination_layout.start_offset() * DType::F16.size_in_bytes(),
        );
        encoder_ref.set_bytes_directly(
            2,
            std::mem::size_of_val(self.positions.as_slice()),
            self.positions.as_ptr().cast(),
        );
        encoder_ref.set_bytes(3, &self.elements);
        encoder_ref.set_bytes(4, &self.heads);
        encoder_ref.set_bytes(5, &self.capacity);
        encoder_ref.set_bytes(6, &self.head_dim);
        encoder_ref.dispatch_threads(
            objc2_metal::MTLSize {
                width: self.elements as usize,
                height: 1,
                depth: 1,
            },
            candle_metal_kernels::utils::get_block_dims((self.elements as usize).min(256), 1, 1),
        );
        Ok(())
    }
}

fn contiguous_pipeline(device: &candle_core::MetalDevice) -> Result<ComputePipeline> {
    Ok(pipelines(device)?.contiguous)
}

#[derive(Clone, Copy)]
struct CastAppend {
    elements: u32,
    source_row_width: u32,
    destination_row_width: u32,
    destination_column: u32,
    source_shape: [usize; 4],
    destination_shape: [usize; 4],
}

impl CastAppend {
    fn plan(destination: &Tensor, source: &Tensor, start: usize) -> Option<Self> {
        if destination.dtype() != DType::F16
            || source.dtype() != DType::F32
            || !destination.is_contiguous()
            || !source.is_contiguous()
        {
            return None;
        }
        let (batch, heads, capacity, head_dim) = destination.dims4().ok()?;
        let (source_batch, source_heads, incoming, source_head_dim) = source.dims4().ok()?;
        Self::plan_geometry(
            [batch, heads, capacity, head_dim],
            [source_batch, source_heads, incoming, source_head_dim],
            start,
        )
    }

    fn plan_geometry(
        destination_shape: [usize; 4],
        source_shape: [usize; 4],
        start: usize,
    ) -> Option<Self> {
        let [batch, heads, capacity, head_dim] = destination_shape;
        let [source_batch, source_heads, incoming, source_head_dim] = source_shape;
        if batch == 0
            || heads == 0
            || head_dim == 0
            || incoming == 0
            || (batch, heads, head_dim) != (source_batch, source_heads, source_head_dim)
            || start.checked_add(incoming)? > capacity
        {
            return None;
        }
        let rows = batch.checked_mul(heads)?;
        let source_row_width = incoming.checked_mul(head_dim)?;
        let destination_row_width = capacity.checked_mul(head_dim)?;
        let destination_column = start.checked_mul(head_dim)?;
        let elements = rows.checked_mul(source_row_width)?;
        let destination_elements = rows.checked_mul(destination_row_width)?;
        let _: u32 = destination_elements.try_into().ok()?;
        Some(Self {
            elements: elements.try_into().ok()?,
            source_row_width: source_row_width.try_into().ok()?,
            destination_row_width: destination_row_width.try_into().ok()?,
            destination_column: destination_column.try_into().ok()?,
            source_shape,
            destination_shape,
        })
    }

    fn validate_storage(
        &self,
        destination: &MetalStorage,
        destination_layout: &Layout,
        source: &MetalStorage,
        source_layout: &Layout,
    ) -> Result<()> {
        if destination.dtype() != DType::F16
            || source.dtype() != DType::F32
            || !destination_layout.is_contiguous()
            || !source_layout.is_contiguous()
            || destination_layout.shape().dims() != self.destination_shape
            || source_layout.shape().dims() != self.source_shape
        {
            candle_core::bail!("Metal KV append tensor metadata changed after validation")
        }
        validate_buffer_span(destination, destination_layout)?;
        validate_buffer_span(source, source_layout)
    }
}

impl InplaceOp2 for CastAppend {
    fn name(&self) -> &'static str {
        "dinf-metal-kv-cast-append"
    }

    fn cpu_fwd(
        &self,
        _destination: &mut CpuStorage,
        _destination_layout: &Layout,
        _source: &CpuStorage,
        _source_layout: &Layout,
    ) -> Result<()> {
        candle_core::bail!("Metal KV append cannot run on CPU storage")
    }

    fn metal_fwd(
        &self,
        destination: &mut MetalStorage,
        destination_layout: &Layout,
        source: &MetalStorage,
        source_layout: &Layout,
    ) -> Result<()> {
        self.validate_storage(destination, destination_layout, source, source_layout)?;
        let device = destination.device();
        if source.device().id() != device.id() {
            candle_core::bail!("Metal KV append tensors are on different devices")
        }

        let pipeline = contiguous_pipeline(device)?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-kv-cast-append");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(source.buffer()),
            source_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(
            1,
            Some(destination.buffer()),
            destination_layout.start_offset() * DType::F16.size_in_bytes(),
        );
        encoder_ref.set_bytes(2, &self.elements);
        encoder_ref.set_bytes(3, &self.source_row_width);
        encoder_ref.set_bytes(4, &self.destination_row_width);
        encoder_ref.set_bytes(5, &self.destination_column);
        encoder_ref.dispatch_threads(
            objc2_metal::MTLSize {
                width: self.elements as usize,
                height: 1,
                depth: 1,
            },
            candle_metal_kernels::utils::get_block_dims((self.elements as usize).min(256), 1, 1),
        );
        Ok(())
    }
}

fn validate_buffer_span(storage: &MetalStorage, layout: &Layout) -> Result<()> {
    let Some((_, stop)) = layout.contiguous_offsets() else {
        candle_core::bail!("Metal KV append requires contiguous storage")
    };
    let byte_end = stop
        .checked_mul(storage.dtype().size_in_bytes())
        .ok_or_else(|| candle_core::Error::Msg("Metal KV append byte span overflow".into()))?;
    if byte_end > storage.buffer().length() {
        candle_core::bail!(
            "Metal KV append tensor span {byte_end} exceeds buffer length {}",
            storage.buffer().length()
        )
    }
    Ok(())
}

fn metal_buffer(tensor: &Tensor) -> Option<candle_metal_kernels::metal::Buffer> {
    let (storage, _) = tensor.storage_and_layout();
    match &*storage {
        Storage::Metal(storage) => Some(storage.buffer().clone()),
        _ => None,
    }
}

pub(super) fn precompile(device: &candle_core::Device) -> anyhow::Result<()> {
    if std::env::var_os("DINF_DISABLE_FUSED_METAL_KV_APPEND").is_some() || !device.is_metal() {
        return Ok(());
    }
    let _ = pipelines(device.as_metal_device()?)?;
    Ok(())
}

pub(super) fn try_cast_append_pair(
    destination_k: &Tensor,
    destination_v: &Tensor,
    source_k: &Tensor,
    source_v: &Tensor,
    start: usize,
) -> anyhow::Result<bool> {
    if std::env::var_os("DINF_DISABLE_FUSED_METAL_KV_APPEND").is_some()
        || !destination_k.device().is_metal()
        || source_k.dtype() != DType::F32
        || source_v.dtype() != DType::F32
        || !destination_k.device().same_device(destination_v.device())
        || !destination_k.device().same_device(source_k.device())
        || !destination_k.device().same_device(source_v.device())
    {
        return Ok(false);
    }
    let Some(k_op) = CastAppend::plan(destination_k, source_k, start) else {
        return Ok(false);
    };
    let Some(v_op) = CastAppend::plan(destination_v, source_v, start) else {
        return Ok(false);
    };

    // Candle's public in-place API grants one exclusive destination lock per
    // operation. Keep K and V as two dispatches rather than writing the second
    // destination through a shared read guard. This preserves Tensor's mutation
    // contract while still eliminating both cast temporaries and copy passes.
    let Some(destination_k_buffer) = metal_buffer(destination_k) else {
        return Ok(false);
    };
    let Some(destination_v_buffer) = metal_buffer(destination_v) else {
        return Ok(false);
    };
    let Some(source_k_buffer) = metal_buffer(source_k) else {
        return Ok(false);
    };
    let Some(source_v_buffer) = metal_buffer(source_v) else {
        return Ok(false);
    };
    if destination_k_buffer == destination_v_buffer
        || destination_k_buffer == source_k_buffer
        || destination_k_buffer == source_v_buffer
        || destination_v_buffer == source_k_buffer
        || destination_v_buffer == source_v_buffer
    {
        return Ok(false);
    }

    destination_k.inplace_op2(source_k, &k_op)?;
    destination_v.inplace_op2(source_v, &v_op)?;
    Ok(true)
}

pub(super) fn try_cast_append_ragged_pair(
    destination_k: &Tensor,
    destination_v: &Tensor,
    source_k: &Tensor,
    source_v: &Tensor,
    positions: &[usize],
) -> anyhow::Result<bool> {
    if std::env::var_os("DINF_DISABLE_FUSED_METAL_KV_APPEND").is_some()
        || !destination_k.device().is_metal()
        || source_k.dtype() != DType::F32
        || source_v.dtype() != DType::F32
        || !destination_k.device().same_device(destination_v.device())
        || !destination_k.device().same_device(source_k.device())
        || !destination_k.device().same_device(source_v.device())
    {
        return Ok(false);
    }
    let Some(k_op) = RaggedCastAppend::plan(destination_k, source_k, positions) else {
        return Ok(false);
    };
    let Some(v_op) = RaggedCastAppend::plan(destination_v, source_v, positions) else {
        return Ok(false);
    };

    // As in the contiguous append path, use one in-place operation for each
    // destination so Candle holds an exclusive write guard for every buffer.
    let Some(destination_k_buffer) = metal_buffer(destination_k) else {
        return Ok(false);
    };
    let Some(destination_v_buffer) = metal_buffer(destination_v) else {
        return Ok(false);
    };
    let Some(source_k_buffer) = metal_buffer(source_k) else {
        return Ok(false);
    };
    let Some(source_v_buffer) = metal_buffer(source_v) else {
        return Ok(false);
    };
    if destination_k_buffer == destination_v_buffer
        || destination_k_buffer == source_k_buffer
        || destination_k_buffer == source_v_buffer
        || destination_v_buffer == source_k_buffer
        || destination_v_buffer == source_v_buffer
    {
        return Ok(false);
    }

    destination_k.inplace_op2(source_k, &k_op)?;
    destination_v.inplace_op2(source_v, &v_op)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architectures::{PreallocatedKvCache, try_metal_f16_sdpa_native};
    use candle_core::IndexOp;
    use half::f16;

    fn rounded_f16(value: f32) -> f32 {
        f16::from_f32(value).to_f32()
    }

    fn patterned_values(elements: usize, multiplier: usize, scale: f32) -> Vec<f32> {
        (0..elements)
            .map(|index| (((index * multiplier) % 257) as f32 - 128.0) * scale)
            .collect()
    }

    fn assert_f16_values(actual: &Tensor, expected: &[f32], context: &str) -> anyhow::Result<()> {
        let actual = actual
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual.len(), expected.len(), "{context} length");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual,
                &rounded_f16(*expected),
                "{context} differs at element {index}"
            );
        }
        Ok(())
    }

    #[test]
    fn planning_rejects_destination_index_space_larger_than_metal_uint() {
        let rows = 1usize << 16;
        let capacity = 1usize << 16;
        assert!(
            CastAppend::plan_geometry([rows, 1, capacity, 1], [rows, 1, 1, 1], capacity - 1,)
                .is_none()
        );
        assert!(CastAppend::plan_geometry([0, 1, 64, 1], [0, 1, 1, 1], 0).is_none());
        assert!(CastAppend::plan_geometry([1, 0, 64, 1], [1, 0, 1, 1], 0).is_none());
        assert!(CastAppend::plan_geometry([1, 1, 64, 0], [1, 1, 1, 0], 0).is_none());
    }

    #[test]
    fn ragged_planning_preserves_lane_positions_and_rejects_unsafe_geometry() {
        let positions = [7usize, 2, 7, 4];
        let plan = RaggedCastAppend::plan_geometry([4, 8, 64, 128], [4, 8, 1, 128], &positions)
            .expect("valid ragged append geometry");
        assert_eq!(plan.positions, vec![7, 2, 7, 4]);
        assert_eq!(plan.elements, 4 * 8 * 128);

        assert!(
            RaggedCastAppend::plan_geometry([4, 8, 64, 128], [4, 8, 1, 128], &[0, 1, 2]).is_none()
        );
        assert!(
            RaggedCastAppend::plan_geometry([4, 8, 64, 128], [4, 8, 2, 128], &positions).is_none()
        );
        assert!(
            RaggedCastAppend::plan_geometry([4, 8, 64, 128], [4, 8, 1, 128], &[0, 1, 2, 64])
                .is_none()
        );
        assert!(
            RaggedCastAppend::plan_geometry(
                [1, 1usize << 16, 1usize << 16, 1],
                [1, 1usize << 16, 1, 1],
                &[0],
            )
            .is_none()
        );
        assert!(
            RaggedCastAppend::plan_geometry(
                [MAX_INLINE_POSITION_BYTES / 4 + 1, 1, 1, 1],
                [MAX_INLINE_POSITION_BYTES / 4 + 1, 1, 1, 1],
                &vec![0; MAX_INLINE_POSITION_BYTES / 4 + 1],
            )
            .is_none()
        );
    }

    #[test]
    fn cast_append_honors_offsets_unequal_shapes_sentinels_and_source_lifetime()
    -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };

        let (batch, heads, capacity, incoming) = (1usize, 4usize, 64usize, 2usize);
        let (k_dim, v_dim, start) = (256usize, 128usize, 5usize);
        let (prefix, suffix) = (13usize, 17usize);
        let sentinel = -11.25f32;
        let k_elements = batch * heads * capacity * k_dim;
        let v_elements = batch * heads * capacity * v_dim;
        let k_source_elements = batch * heads * incoming * k_dim;
        let v_source_elements = batch * heads * incoming * v_dim;

        let k_backing = Tensor::full(sentinel, (prefix + k_elements + suffix,), &device)?
            .to_dtype(DType::F16)?;
        let v_backing = Tensor::full(sentinel, (prefix + v_elements + suffix,), &device)?
            .to_dtype(DType::F16)?;
        let destination_k = k_backing
            .narrow(0, prefix, k_elements)?
            .reshape((batch, heads, capacity, k_dim))?;
        let destination_v = v_backing
            .narrow(0, prefix, v_elements)?
            .reshape((batch, heads, capacity, v_dim))?;
        assert!(destination_k.layout().start_offset() > 0);
        assert!(destination_v.layout().start_offset() > 0);

        let source_prefix = 7usize;
        let k_values = patterned_values(k_source_elements, 17, 1.0 / 61.0);
        let v_values = patterned_values(v_source_elements, 29, 1.0 / 43.0);
        {
            let mut k_backing_values = vec![91.0; source_prefix];
            k_backing_values.extend_from_slice(&k_values);
            k_backing_values.extend([92.0; 9]);
            let mut v_backing_values = vec![93.0; source_prefix];
            v_backing_values.extend_from_slice(&v_values);
            v_backing_values.extend([94.0; 9]);
            let source_k = Tensor::from_vec(
                k_backing_values,
                (source_prefix + k_source_elements + 9,),
                &device,
            )?
            .narrow(0, source_prefix, k_source_elements)?
            .reshape((batch, heads, incoming, k_dim))?;
            let source_v = Tensor::from_vec(
                v_backing_values,
                (source_prefix + v_source_elements + 9,),
                &device,
            )?
            .narrow(0, source_prefix, v_source_elements)?
            .reshape((batch, heads, incoming, v_dim))?;
            assert!(source_k.layout().start_offset() > 0);
            assert!(source_v.layout().start_offset() > 0);
            assert!(!try_cast_append_pair(
                &destination_k,
                &destination_v,
                &source_k,
                &source_v.to_dtype(DType::F16)?,
                start,
            )?);
            assert!(try_cast_append_pair(
                &destination_k,
                &destination_v,
                &source_k,
                &source_v,
                start,
            )?);
        }

        // The command encoder and Metal resource tracking retain sources until
        // completion even though every Tensor owning them has now been dropped.
        device.synchronize()?;

        let expected_storage = |source: &[f32], head_dim: usize, elements: usize| {
            let mut expected = vec![sentinel; elements];
            let source_row_width = incoming * head_dim;
            let destination_row_width = capacity * head_dim;
            for row in 0..batch * heads {
                let source_start = row * source_row_width;
                let destination_start = row * destination_row_width + start * head_dim;
                expected[destination_start..destination_start + source_row_width]
                    .copy_from_slice(&source[source_start..source_start + source_row_width]);
            }
            expected
        };
        assert_f16_values(
            &destination_k,
            &expected_storage(&k_values, k_dim, k_elements),
            "key cache",
        )?;
        assert_f16_values(
            &destination_v,
            &expected_storage(&v_values, v_dim, v_elements),
            "value cache",
        )?;

        let k_backing_values = k_backing.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let v_backing_values = v_backing.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        assert!(
            k_backing_values[..prefix]
                .iter()
                .chain(&k_backing_values[prefix + k_elements..])
                .all(|value| *value == rounded_f16(sentinel))
        );
        assert!(
            v_backing_values[..prefix]
                .iter()
                .chain(&v_backing_values[prefix + v_elements..])
                .all(|value| *value == rounded_f16(sentinel))
        );
        Ok(())
    }

    #[test]
    fn ragged_cast_append_honors_lane_positions_offsets_unequal_shapes_and_lifetime()
    -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };

        let (batch, capacity) = (4usize, 16usize);
        let (k_heads, k_dim) = (4usize, 128usize);
        let (v_heads, v_dim) = (2usize, 64usize);
        let positions = [7usize, 2, 7, 4];
        let (prefix, suffix) = (11usize, 13usize);
        let sentinel = -9.5f32;
        let k_elements = batch * k_heads * capacity * k_dim;
        let v_elements = batch * v_heads * capacity * v_dim;
        let k_source_elements = batch * k_heads * k_dim;
        let v_source_elements = batch * v_heads * v_dim;

        let k_backing = Tensor::full(sentinel, (prefix + k_elements + suffix,), &device)?
            .to_dtype(DType::F16)?;
        let v_backing = Tensor::full(sentinel, (prefix + v_elements + suffix,), &device)?
            .to_dtype(DType::F16)?;
        let destination_k = k_backing
            .narrow(0, prefix, k_elements)?
            .reshape((batch, k_heads, capacity, k_dim))?;
        let destination_v = v_backing
            .narrow(0, prefix, v_elements)?
            .reshape((batch, v_heads, capacity, v_dim))?;
        assert!(destination_k.layout().start_offset() > 0);
        assert!(destination_v.layout().start_offset() > 0);

        let source_prefix = 5usize;
        let k_values = patterned_values(k_source_elements, 19, 1.0 / 59.0);
        let v_values = patterned_values(v_source_elements, 31, 1.0 / 47.0);
        {
            let mut k_backing_values = vec![81.0; source_prefix];
            k_backing_values.extend_from_slice(&k_values);
            k_backing_values.extend([82.0; 7]);
            let mut v_backing_values = vec![83.0; source_prefix];
            v_backing_values.extend_from_slice(&v_values);
            v_backing_values.extend([84.0; 7]);
            let source_k = Tensor::from_vec(
                k_backing_values,
                (source_prefix + k_source_elements + 7,),
                &device,
            )?
            .narrow(0, source_prefix, k_source_elements)?
            .reshape((batch, k_heads, 1, k_dim))?;
            let source_v = Tensor::from_vec(
                v_backing_values,
                (source_prefix + v_source_elements + 7,),
                &device,
            )?
            .narrow(0, source_prefix, v_source_elements)?
            .reshape((batch, v_heads, 1, v_dim))?;
            assert!(source_k.layout().start_offset() > 0);
            assert!(source_v.layout().start_offset() > 0);
            assert!(try_cast_append_ragged_pair(
                &destination_k,
                &destination_v,
                &source_k,
                &source_v,
                &positions,
            )?);
        }

        // Both source tensors and the position vectors owned by the two ops
        // have been dropped; command-buffer resource tracking must retain all
        // device inputs until the asynchronous writes complete.
        device.synchronize()?;

        let expected_storage = |source: &[f32], heads: usize, head_dim: usize, elements: usize| {
            let mut expected = vec![sentinel; elements];
            for (lane, &position) in positions.iter().enumerate().take(batch) {
                for head in 0..heads {
                    let source_start = (lane * heads + head) * head_dim;
                    let destination_start =
                        ((lane * heads + head) * capacity + position) * head_dim;
                    expected[destination_start..destination_start + head_dim]
                        .copy_from_slice(&source[source_start..source_start + head_dim]);
                }
            }
            expected
        };
        assert_f16_values(
            &destination_k,
            &expected_storage(&k_values, k_heads, k_dim, k_elements),
            "ragged key cache",
        )?;
        assert_f16_values(
            &destination_v,
            &expected_storage(&v_values, v_heads, v_dim, v_elements),
            "ragged value cache",
        )?;

        for backing in [&k_backing, &v_backing] {
            let values = backing.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let active_elements = if backing.id() == k_backing.id() {
                k_elements
            } else {
                v_elements
            };
            assert!(
                values[..prefix]
                    .iter()
                    .chain(&values[prefix + active_elements..])
                    .all(|value| *value == rounded_f16(sentinel))
            );
        }
        Ok(())
    }

    #[test]
    fn sequential_cast_appends_match_reference_cache_and_sdpa() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };

        let (batch, q_heads, kv_heads, capacity, qk_dim, v_dim) =
            (1usize, 8usize, 4usize, 64usize, 256usize, 128usize);
        let initial_tokens = 12usize;
        let mut fused = PreallocatedKvCache {
            k: Tensor::zeros((batch, kv_heads, capacity, qk_dim), DType::F16, &device)?,
            v: Tensor::zeros((batch, kv_heads, capacity, v_dim), DType::F16, &device)?,
            start: 0,
            len: 0,
            capacity,
            allocation_lanes: batch,
        };
        let reference_k = Tensor::zeros((batch, kv_heads, capacity, qk_dim), DType::F16, &device)?;
        let reference_v = Tensor::zeros((batch, kv_heads, capacity, v_dim), DType::F16, &device)?;

        for token in 0..initial_tokens {
            let k_values = patterned_values(batch * kv_heads * qk_dim, 13 + token, 1.0 / 97.0);
            let v_values = patterned_values(batch * kv_heads * v_dim, 31 + token, 1.0 / 83.0);
            let source_k = Tensor::from_vec(k_values, (batch, kv_heads, 1, qk_dim), &device)?;
            let source_v = Tensor::from_vec(v_values, (batch, kv_heads, 1, v_dim), &device)?;

            fused.append(&source_k, &source_v)?;
            reference_k.slice_set(&source_k.to_dtype(DType::F16)?.contiguous()?, 2, token)?;
            reference_v.slice_set(&source_v.to_dtype(DType::F16)?.contiguous()?, 2, token)?;
        }
        assert_eq!(fused.len, initial_tokens);

        fused.evict_oldest(5)?;
        let source_k = Tensor::from_vec(
            patterned_values(batch * kv_heads * qk_dim, 47, 1.0 / 73.0),
            (batch, kv_heads, 1, qk_dim),
            &device,
        )?;
        let source_v = Tensor::from_vec(
            patterned_values(batch * kv_heads * v_dim, 53, 1.0 / 79.0),
            (batch, kv_heads, 1, v_dim),
            &device,
        )?;
        fused.append(&source_k, &source_v)?;
        reference_k.slice_set(
            &source_k.to_dtype(DType::F16)?.contiguous()?,
            2,
            initial_tokens,
        )?;
        reference_v.slice_set(
            &source_v.to_dtype(DType::F16)?.contiguous()?,
            2,
            initial_tokens,
        )?;
        assert_eq!(fused.start, 5);
        assert_eq!(fused.len, initial_tokens - 4);

        let (fused_k, fused_v) = fused.active_f16()?;
        let reference_k = reference_k.narrow(2, fused.start, fused.len)?;
        let reference_v = reference_v.narrow(2, fused.start, fused.len)?;
        let query = Tensor::from_vec(
            patterned_values(batch * q_heads * qk_dim, 37, 1.0 / 109.0),
            (batch, q_heads, 1, qk_dim),
            &device,
        )?;
        let scale = 1.0 / (qk_dim as f32).sqrt();
        let fused_attention = try_metal_f16_sdpa_native(&query, &fused_k, &fused_v, false, scale)?
            .expect("real Qwen attention geometry should use Metal SDPA");
        let reference_attention =
            try_metal_f16_sdpa_native(&query, &reference_k, &reference_v, false, scale)?
                .expect("reference Qwen attention geometry should use Metal SDPA");

        assert_f16_values(
            &fused_k,
            &reference_k
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            "sequential key cache",
        )?;
        assert_f16_values(
            &fused_v,
            &reference_v
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            "sequential value cache",
        )?;
        let fused_attention = fused_attention
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let reference_attention = reference_attention
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(fused_attention, reference_attention);
        Ok(())
    }

    #[test]
    fn sequential_ragged_cast_appends_match_scatter_reference_and_sdpa() -> anyhow::Result<()> {
        use crate::architectures::try_fused_f16_ragged_sdpa;

        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (batch, q_heads, kv_heads, capacity, head_dim) =
            (2usize, 8usize, 4usize, 64usize, 128usize);
        let initial_positions = [2usize, 4usize];
        let cache_elements = batch * kv_heads * capacity * head_dim;
        let mut initial_k = vec![-3.5f32; cache_elements];
        let mut initial_v = vec![4.5f32; cache_elements];
        for (lane, &initial_position) in initial_positions.iter().enumerate().take(batch) {
            for head in 0..kv_heads {
                for position in 0..initial_position {
                    for column in 0..head_dim {
                        let index =
                            ((lane * kv_heads + head) * capacity + position) * head_dim + column;
                        initial_k[index] = ((index * 13 % 251) as f32 - 125.0) * (1.0 / 97.0);
                        initial_v[index] = ((index * 29 % 257) as f32 - 128.0) * (1.0 / 89.0);
                    }
                }
            }
        }
        let make_cache = || -> anyhow::Result<PreallocatedKvCache> {
            Ok(PreallocatedKvCache {
                k: Tensor::from_vec(
                    initial_k.clone(),
                    (batch, kv_heads, capacity, head_dim),
                    &device,
                )?
                .to_dtype(DType::F16)?,
                v: Tensor::from_vec(
                    initial_v.clone(),
                    (batch, kv_heads, capacity, head_dim),
                    &device,
                )?
                .to_dtype(DType::F16)?,
                start: 0,
                len: *initial_positions.iter().max().unwrap(),
                capacity,
                allocation_lanes: batch,
            })
        };
        let mut fused = make_cache()?;
        let mut reference = make_cache()?;
        let mut positions = initial_positions;
        for round in 0usize..5 {
            let k_values = patterned_values(batch * kv_heads * head_dim, 17 + round, 1.0 / 73.0);
            let v_values = patterned_values(batch * kv_heads * head_dim, 37 + round, 1.0 / 83.0);
            let source_k = Tensor::from_vec(k_values, (batch, kv_heads, 1, head_dim), &device)?;
            let source_v = Tensor::from_vec(v_values, (batch, kv_heads, 1, head_dim), &device)?;
            fused.append_ragged(&source_k, &source_v, &positions)?;
            reference.append_ragged(
                &source_k.to_dtype(DType::F16)?,
                &source_v.to_dtype(DType::F16)?,
                &positions,
            )?;
            positions.iter_mut().for_each(|position| *position += 1);
        }
        device.synchronize()?;
        assert_eq!((fused.start, fused.len, fused.capacity), (0, 9, capacity));
        assert_f16_values(
            &fused.k,
            &reference
                .k
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            "sequential ragged key cache",
        )?;
        assert_f16_values(
            &fused.v,
            &reference
                .v
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            "sequential ragged value cache",
        )?;

        let valid_lengths = positions.to_vec();
        let query = Tensor::from_vec(
            patterned_values(batch * q_heads * head_dim, 43, 1.0 / 101.0),
            (batch, q_heads, 1, head_dim),
            &device,
        )?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let fused_attention =
            try_fused_f16_ragged_sdpa(&query, &fused.k, &fused.v, &valid_lengths, scale)?
                .expect("real ragged attention geometry should use Metal SDPA");
        let reference_attention =
            try_fused_f16_ragged_sdpa(&query, &reference.k, &reference.v, &valid_lengths, scale)?
                .expect("reference ragged attention geometry should use Metal SDPA");
        assert_eq!(
            fused_attention
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            reference_attention
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn ragged_growth_and_nonzero_start_keep_the_existing_fallback_contract() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (batch, heads, capacity, head_dim) = (2usize, 2usize, 4usize, 32usize);
        let mut cache = PreallocatedKvCache {
            k: Tensor::full(1.0f32, (batch, heads, capacity, head_dim), &device)?
                .to_dtype(DType::F16)?,
            v: Tensor::full(2.0f32, (batch, heads, capacity, head_dim), &device)?
                .to_dtype(DType::F16)?,
            start: 0,
            len: capacity,
            capacity,
            allocation_lanes: batch,
        };
        let old_k_id = cache.k.id();
        let source_k = Tensor::full(3.0f32, (batch, heads, 1, head_dim), &device)?;
        let source_v = Tensor::full(4.0f32, (batch, heads, 1, head_dim), &device)?;
        cache.append_ragged(&source_k, &source_v, &[2, 4])?;
        device.synchronize()?;
        assert_ne!(cache.k.id(), old_k_id);
        assert_eq!((cache.start, cache.len, cache.capacity), (0, 5, 64));
        assert_eq!(
            cache
                .k
                .i((0, 0, 2, 0))?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()?,
            3.0
        );
        assert_eq!(
            cache
                .v
                .i((1, 0, 4, 0))?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()?,
            4.0
        );
        assert_eq!(
            cache
                .k
                .i((1, 0, 0, 0))?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()?,
            1.0
        );

        cache.evict_oldest(1)?;
        let state = (
            cache.k.id(),
            cache.v.id(),
            cache.start,
            cache.len,
            cache.capacity,
        );
        assert!(cache.append_ragged(&source_k, &source_v, &[3, 4]).is_err());
        assert_eq!(
            (
                cache.k.id(),
                cache.v.id(),
                cache.start,
                cache.len,
                cache.capacity
            ),
            state
        );
        Ok(())
    }

    #[test]
    fn cache_growth_keeps_the_existing_fallback_and_prefix() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (batch, heads, capacity, head_dim) = (1usize, 4usize, 64usize, 256usize);
        let mut cache = PreallocatedKvCache {
            k: Tensor::full(1.0f32, (batch, heads, capacity, head_dim), &device)?
                .to_dtype(DType::F16)?,
            v: Tensor::full(2.0f32, (batch, heads, capacity, head_dim), &device)?
                .to_dtype(DType::F16)?,
            start: 0,
            len: capacity,
            capacity,
            allocation_lanes: batch,
        };
        let old_k_id = cache.k.id();
        let source_k = Tensor::full(3.0f32, (batch, heads, 1, head_dim), &device)?;
        let source_v = Tensor::full(4.0f32, (batch, heads, 1, head_dim), &device)?;
        cache.append(&source_k, &source_v)?;
        assert_ne!(cache.k.id(), old_k_id);
        assert_eq!((cache.start, cache.len, cache.capacity), (0, 65, 128));
        device.synchronize()?;
        assert_eq!(
            cache
                .k
                .i((0, 0, 0, 0))?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()?,
            1.0
        );
        assert_eq!(
            cache
                .v
                .i((0, 0, 0, 0))?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()?,
            2.0
        );
        assert_eq!(
            cache
                .k
                .i((0, 0, 64, 0))?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()?,
            3.0
        );
        assert_eq!(
            cache
                .v
                .i((0, 0, 64, 0))?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()?,
            4.0
        );
        Ok(())
    }

    #[test]
    #[ignore = "manual Metal ABBA microbenchmark"]
    fn benchmark_cast_append_against_cast_then_slice_set() -> anyhow::Result<()> {
        use std::time::{Duration, Instant};

        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (batch, heads, capacity, head_dim) = (1usize, 4usize, 512usize, 256usize);
        let source_k = Tensor::from_vec(
            patterned_values(batch * heads * head_dim, 17, 1.0 / 67.0),
            (batch, heads, 1, head_dim),
            &device,
        )?;
        let source_v = Tensor::from_vec(
            patterned_values(batch * heads * head_dim, 29, 1.0 / 71.0),
            (batch, heads, 1, head_dim),
            &device,
        )?;
        let destination_k = Tensor::zeros((batch, heads, capacity, head_dim), DType::F16, &device)?;
        let destination_v = Tensor::zeros((batch, heads, capacity, head_dim), DType::F16, &device)?;
        let iterations = 2_000usize;
        let append_start = 127usize;
        device.synchronize()?;
        let cold_started = Instant::now();
        assert!(try_cast_append_pair(
            &destination_k,
            &destination_v,
            &source_k,
            &source_v,
            append_start,
        )?);
        device.synchronize()?;
        let cold = cold_started.elapsed();
        let warm_started = Instant::now();
        assert!(try_cast_append_pair(
            &destination_k,
            &destination_v,
            &source_k,
            &source_v,
            append_start,
        )?);
        device.synchronize()?;
        let warm = warm_started.elapsed();
        eprintln!(
            "KV_CAST_APPEND_FIRST_USE cold_ms={:.3} warm_ms={:.3}",
            cold.as_secs_f64() * 1_000.0,
            warm.as_secs_f64() * 1_000.0,
        );
        let measure = |fused: bool| -> anyhow::Result<Duration> {
            device.synchronize()?;
            let started = Instant::now();
            for _ in 0..iterations {
                if fused {
                    assert!(try_cast_append_pair(
                        &destination_k,
                        &destination_v,
                        &source_k,
                        &source_v,
                        append_start,
                    )?);
                } else {
                    let k = source_k.to_dtype(DType::F16)?.contiguous()?;
                    let v = source_v.to_dtype(DType::F16)?.contiguous()?;
                    destination_k.slice_set(&k, 2, append_start)?;
                    destination_v.slice_set(&v, 2, append_start)?;
                }
            }
            device.synchronize()?;
            Ok(started.elapsed())
        };

        measure(false)?;
        measure(true)?;
        let mut reference = Vec::with_capacity(7);
        let mut fused = Vec::with_capacity(7);
        for round in 0usize..7 {
            if round.is_multiple_of(2) {
                fused.push(measure(true)?);
                reference.push(measure(false)?);
            } else {
                reference.push(measure(false)?);
                fused.push(measure(true)?);
            }
        }
        reference.sort_unstable();
        fused.sort_unstable();
        let reference = reference[reference.len() / 2];
        let fused = fused[fused.len() / 2];
        eprintln!(
            "KV_CAST_APPEND_BENCH iterations={iterations} reference_ms={:.3} fused_ms={:.3} speedup={:.3}x",
            reference.as_secs_f64() * 1_000.0,
            fused.as_secs_f64() * 1_000.0,
            reference.as_secs_f64() / fused.as_secs_f64(),
        );
        Ok(())
    }

    #[test]
    #[ignore = "manual Metal ABBA microbenchmark"]
    fn benchmark_ragged_cast_append_batch2_and_batch4() -> anyhow::Result<()> {
        use std::time::{Duration, Instant};

        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for batch in [2usize, 4] {
            let (heads, capacity, head_dim) = (4usize, 512usize, 128usize);
            let positions = (0..batch).map(|lane| 127 + lane * 29).collect::<Vec<_>>();
            let source_k = Tensor::from_vec(
                patterned_values(batch * heads * head_dim, 17, 1.0 / 67.0),
                (batch, heads, 1, head_dim),
                &device,
            )?;
            let source_v = Tensor::from_vec(
                patterned_values(batch * heads * head_dim, 29, 1.0 / 71.0),
                (batch, heads, 1, head_dim),
                &device,
            )?;
            let destination_k =
                Tensor::zeros((batch, heads, capacity, head_dim), DType::F16, &device)?;
            let destination_v =
                Tensor::zeros((batch, heads, capacity, head_dim), DType::F16, &device)?;
            let iterations = 2_000usize;
            let measure = |fused: bool| -> anyhow::Result<Duration> {
                device.synchronize()?;
                let started = Instant::now();
                for _ in 0..iterations {
                    if fused {
                        assert!(try_cast_append_ragged_pair(
                            &destination_k,
                            &destination_v,
                            &source_k,
                            &source_v,
                            &positions,
                        )?);
                    } else {
                        let k = source_k.to_dtype(DType::F16)?.contiguous()?;
                        let v = source_v.to_dtype(DType::F16)?.contiguous()?;
                        let (k_indices, v_indices) =
                            PreallocatedKvCache::ragged_indices(&positions, &k, &v)?;
                        destination_k.scatter_set(&k_indices, &k, 2)?;
                        destination_v.scatter_set(&v_indices, &v, 2)?;
                    }
                }
                device.synchronize()?;
                Ok(started.elapsed())
            };

            measure(false)?;
            measure(true)?;
            let mut reference = Vec::with_capacity(7);
            let mut fused = Vec::with_capacity(7);
            for round in 0usize..7 {
                if round.is_multiple_of(2) {
                    fused.push(measure(true)?);
                    reference.push(measure(false)?);
                } else {
                    reference.push(measure(false)?);
                    fused.push(measure(true)?);
                }
            }
            reference.sort_unstable();
            fused.sort_unstable();
            let reference = reference[reference.len() / 2];
            let fused = fused[fused.len() / 2];
            eprintln!(
                "RAGGED_KV_CAST_APPEND_BENCH batch={batch} iterations={iterations} reference_ms={:.3} fused_ms={:.3} speedup={:.3}x",
                reference.as_secs_f64() * 1_000.0,
                fused.as_secs_f64() * 1_000.0,
                reference.as_secs_f64() / fused.as_secs_f64(),
            );
        }
        Ok(())
    }
}
