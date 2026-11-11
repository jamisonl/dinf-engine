use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{
    CpuStorage, CustomOp1, CustomOp3, DType, Layout, MetalStorage, Result, Shape, Storage, Tensor,
};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::MTLComputePipelineState;

const RAGGED_SDPA_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

#define HEAD_DIM __HEAD_DIM__

kernel void dinf_ragged_sdpa_f16(
    device const half *queries [[buffer(0)]],
    device const half *keys [[buffer(1)]],
    device const half *values [[buffer(2)]],
    device half *out [[buffer(3)]],
    constant uint &gqa_factor [[buffer(4)]],
    constant uint &q_heads [[buffer(5)]],
    constant size_t &k_stride [[buffer(6)]],
    constant size_t &v_stride [[buffer(7)]],
    constant float &scale [[buffer(8)]],
    constant uint *valid_lengths [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr uint BN = 32;
  constexpr uint BD = 32;
  constexpr uint EPT = (HEAD_DIM + BD - 1) / BD;
  constexpr uint STRIDE = BN * HEAD_DIM;

  thread float q[EPT];
  thread float key[EPT];
  thread float output[EPT];
  threadgroup float outputs[BN * BD];
  threadgroup float max_scores[BN];
  threadgroup float sum_exp_scores[BN];

  const uint head_idx = tid.y;
  const uint batch_idx = head_idx / q_heads;
  const uint q_head_idx = head_idx - batch_idx * q_heads;
  const uint kv_heads = q_heads / gqa_factor;
  const uint kv_head_idx = batch_idx * kv_heads + q_head_idx / gqa_factor;
  const uint n = valid_lengths[batch_idx];

  queries += head_idx * HEAD_DIM;
  keys += kv_head_idx * k_stride + simd_gid * HEAD_DIM;
  values += kv_head_idx * v_stride + simd_gid * HEAD_DIM;
  out += head_idx * HEAD_DIM;

  for (uint j = 0; j < EPT; ++j) {
    const uint dim = simd_lid * EPT + j;
    q[j] = dim < HEAD_DIM ? scale * float(queries[dim]) : 0.0f;
    output[j] = 0.0f;
  }

  float max_score = -INFINITY;
  float sum_exp_score = 0.0f;
  for (uint position = simd_gid; position < n; position += BN) {
    for (uint j = 0; j < EPT; ++j) {
      const uint dim = simd_lid * EPT + j;
      key[j] = dim < HEAD_DIM ? float(keys[dim]) : 0.0f;
    }

    float score = 0.0f;
    for (uint j = 0; j < EPT; ++j) {
      score += q[j] * key[j];
    }
    score = simd_sum(score);

    const float next_max = max(max_score, score);
    const float old_factor = fast::exp(max_score - next_max);
    const float exp_score = fast::exp(score - next_max);
    max_score = next_max;
    sum_exp_score = sum_exp_score * old_factor + exp_score;
    for (uint j = 0; j < EPT; ++j) {
      const uint dim = simd_lid * EPT + j;
      const float value = dim < HEAD_DIM ? float(values[dim]) : 0.0f;
      output[j] = output[j] * old_factor + exp_score * value;
    }

    keys += STRIDE;
    values += STRIDE;
  }

  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  const float next_max = simd_max(max_score);
  const float factor = fast::exp(max_score - next_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  for (uint j = 0; j < EPT; ++j) {
    outputs[simd_lid * BD + simd_gid] = output[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    output[j] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) / sum_exp_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (uint j = 0; j < EPT; ++j) {
      const uint dim = simd_gid * EPT + j;
      if (dim < HEAD_DIM) {
        out[dim] = half(output[j]);
      }
    }
  }
}
"#;

const RAGGED_ROPE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

template <typename T>
METAL_FUNC void dinf_ragged_rope_body(
    device const T *src,
    device const float *cos,
    device const float *sin,
    device T *dst,
    constant uint &batch,
    constant uint &heads,
    constant uint &head_dim,
    constant uint &rotary_dim,
    constant bool &neox,
    constant uint *positions,
    uint gid) {
  const uint elements = batch * heads * head_dim;
  if (gid >= elements) {
    return;
  }

  const uint elements_per_lane = heads * head_dim;
  const uint lane = gid / elements_per_lane;
  const uint within_lane = gid - lane * elements_per_lane;
  const uint head = within_lane / head_dim;
  const uint dim = within_lane - head * head_dim;
  const uint head_base = (lane * heads + head) * head_dim;
  if (dim >= rotary_dim) {
    dst[head_base + dim] = src[head_base + dim];
    return;
  }

  const uint half_dim = rotary_dim / 2;
  const uint rotary_idx = neox ? dim % half_dim : dim / 2;
  const uint first = neox ? head_base + rotary_idx : head_base + 2 * rotary_idx;
  const uint second = neox ? first + half_dim : first + 1;
  const uint cs_idx = positions[lane] * half_dim + rotary_idx;
  const float c = cos[cs_idx];
  const float s = sin[cs_idx];

  const float x0 = float(src[first]);
  const float x1 = float(src[second]);
  const bool first_component = neox ? dim < half_dim : dim % 2 == 0;
  dst[head_base + dim] = T(first_component ? x0 * c - x1 * s : x0 * s + x1 * c);
}

kernel void dinf_ragged_rope_f32(
    device const float *src [[buffer(0)]],
    device const float *cos [[buffer(1)]],
    device const float *sin [[buffer(2)]],
    device float *dst [[buffer(3)]],
    constant uint &batch [[buffer(4)]],
    constant uint &heads [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant uint &rotary_dim [[buffer(7)]],
    constant bool &neox [[buffer(8)]],
    constant uint *positions [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
  dinf_ragged_rope_body<float>(
      src, cos, sin, dst, batch, heads, head_dim, rotary_dim, neox, positions, gid);
}

kernel void dinf_ragged_rope_f16(
    device const half *src [[buffer(0)]],
    device const float *cos [[buffer(1)]],
    device const float *sin [[buffer(2)]],
    device half *dst [[buffer(3)]],
    constant uint &batch [[buffer(4)]],
    constant uint &heads [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant uint &rotary_dim [[buffer(7)]],
    constant bool &neox [[buffer(8)]],
    constant uint *positions [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
  dinf_ragged_rope_body<half>(
      src, cos, sin, dst, batch, heads, head_dim, rotary_dim, neox, positions, gid);
}
"#;

static PIPELINES: OnceLock<Mutex<HashMap<(DeviceId, usize), ComputePipeline>>> = OnceLock::new();
static ROPE_PIPELINES: OnceLock<Mutex<HashMap<(DeviceId, DType), ComputePipeline>>> =
    OnceLock::new();

fn pipeline(device: &candle_core::MetalDevice, head_dim: usize) -> Result<ComputePipeline> {
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (device.id(), head_dim);
    if let Some(pipeline) = pipelines.lock().unwrap().get(&key) {
        return Ok(pipeline.clone());
    }

    let source = RAGGED_SDPA_MSL.replace("__HEAD_DIM__", &head_dim.to_string());
    let library = device
        .metal_device()
        .new_library_with_source(&source, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function("dinf_ragged_sdpa_f16", None)
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    pipelines.lock().unwrap().insert(key, pipeline.clone());
    Ok(pipeline)
}

pub(super) fn sdpa_supported(device: &candle_core::MetalDevice, head_dim: usize) -> Result<bool> {
    let pipeline = pipeline(device, head_dim)?;
    Ok(super::metal_ragged_sdpa_launch_supported(
        pipeline.as_ref().threadExecutionWidth(),
        pipeline.max_total_threads_per_threadgroup(),
    ))
}

fn rope_pipeline(device: &candle_core::MetalDevice, dtype: DType) -> Result<ComputePipeline> {
    let pipelines = ROPE_PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (device.id(), dtype);
    if let Some(pipeline) = pipelines.lock().unwrap().get(&key) {
        return Ok(pipeline.clone());
    }
    let name = match dtype {
        DType::F32 => "dinf_ragged_rope_f32",
        DType::F16 => "dinf_ragged_rope_f16",
        _ => candle_core::bail!("ragged Metal RoPE does not support {dtype:?}"),
    };
    let library = device
        .metal_device()
        .new_library_with_source(RAGGED_ROPE_MSL, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function(name, None)
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    pipelines.lock().unwrap().insert(key, pipeline.clone());
    Ok(pipeline)
}

struct RaggedRope<'a> {
    cos: &'a Tensor,
    sin: &'a Tensor,
    positions: Vec<u32>,
    neox: bool,
}

impl CustomOp1 for RaggedRope<'_> {
    fn name(&self) -> &'static str {
        "dinf-ragged-rope"
    }

    fn cpu_fwd(&self, _src: &CpuStorage, _src_layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("dinf ragged RoPE is only dispatched on Metal")
    }

    fn metal_fwd(&self, src: &MetalStorage, src_layout: &Layout) -> Result<(MetalStorage, Shape)> {
        if !matches!(src.dtype(), DType::F16 | DType::F32) || !src_layout.is_contiguous() {
            candle_core::bail!("ragged Metal RoPE requires contiguous f16/f32 input")
        }
        let (batch, heads, seq_len, head_dim) = src_layout.shape().dims4()?;
        if seq_len != 1 || head_dim % 2 != 0 {
            candle_core::bail!("ragged Metal RoPE tensor geometry differs")
        }
        if self.positions.len() != batch {
            candle_core::bail!("ragged Metal RoPE position count differs")
        }

        let (cos_guard, cos_layout) = self.cos.storage_and_layout();
        let (sin_guard, sin_layout) = self.sin.storage_and_layout();
        let cos = match &*cos_guard {
            Storage::Metal(storage) => storage,
            _ => candle_core::bail!("ragged RoPE cos table is not on Metal"),
        };
        let sin = match &*sin_guard {
            Storage::Metal(storage) => storage,
            _ => candle_core::bail!("ragged RoPE sin table is not on Metal"),
        };
        if cos.dtype() != DType::F32
            || sin.dtype() != DType::F32
            || !cos_layout.is_contiguous()
            || !sin_layout.is_contiguous()
        {
            candle_core::bail!("ragged RoPE tables must be contiguous f32")
        }
        let (cos_positions, cos_half_dim) = cos_layout.shape().dims2()?;
        let (sin_positions, sin_half_dim) = sin_layout.shape().dims2()?;
        let rotary_dim = cos_half_dim * 2;
        if cos_positions != sin_positions
            || cos_half_dim != sin_half_dim
            || rotary_dim == 0
            || rotary_dim > head_dim
        {
            candle_core::bail!("ragged Metal RoPE table geometry differs")
        }

        let elements = src_layout.shape().elem_count();
        let device = src.device();
        if device.id() != cos.device().id() || device.id() != sin.device().id() {
            candle_core::bail!("ragged Metal RoPE tensors are on different devices")
        }
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, src.dtype())
            .with_label("dinf-ragged-rope")
            .build()?;
        let pipeline = rope_pipeline(device, src.dtype())?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-ragged-rope");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(src.buffer()),
            src_layout.start_offset() * src.dtype().size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            1,
            Some(cos.buffer()),
            cos_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            2,
            Some(sin.buffer()),
            sin_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(3, Some(&output), 0);
        encoder_ref.set_bytes(4, &(batch as u32));
        encoder_ref.set_bytes(5, &(heads as u32));
        encoder_ref.set_bytes(6, &(head_dim as u32));
        encoder_ref.set_bytes(7, &(rotary_dim as u32));
        encoder_ref.set_bytes(8, &self.neox);
        encoder_ref.set_bytes_directly(
            9,
            std::mem::size_of_val(self.positions.as_slice()),
            self.positions.as_ptr().cast(),
        );
        encoder_ref.dispatch_threads(
            objc2_metal::MTLSize {
                width: elements,
                height: 1,
                depth: 1,
            },
            candle_metal_kernels::utils::get_block_dims(elements.min(256), 1, 1),
        );

        Ok((
            MetalStorage::new(output, device.clone(), elements, src.dtype()),
            src_layout.shape().clone(),
        ))
    }
}

struct RaggedSdpa<'a> {
    valid_lengths: &'a [usize],
    scale: f32,
}

impl CustomOp3 for RaggedSdpa<'_> {
    fn name(&self) -> &'static str {
        "dinf-ragged-sdpa-f16"
    }

    fn cpu_fwd(
        &self,
        _q: &CpuStorage,
        _q_layout: &Layout,
        _k: &CpuStorage,
        _k_layout: &Layout,
        _v: &CpuStorage,
        _v_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("dinf ragged SDPA is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        q: &MetalStorage,
        q_layout: &Layout,
        k: &MetalStorage,
        k_layout: &Layout,
        v: &MetalStorage,
        v_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        for (name, storage, layout) in [
            ("query", q, q_layout),
            ("key", k, k_layout),
            ("value", v, v_layout),
        ] {
            if storage.dtype() != DType::F16 || !layout.is_contiguous() {
                candle_core::bail!("{name} must be contiguous f16 for ragged Metal SDPA")
            }
        }

        let (batch, q_heads, q_seq, head_dim) = q_layout.shape().dims4()?;
        let (k_batch, k_heads, _, k_dim) = k_layout.shape().dims4()?;
        let (v_batch, v_heads, _, v_dim) = v_layout.shape().dims4()?;
        if q_seq != 1 {
            candle_core::bail!("ragged Metal SDPA is decode-only")
        }
        if (batch, head_dim) != (k_batch, k_dim)
            || (batch, head_dim) != (v_batch, v_dim)
            || k_heads != v_heads
            || q_heads % k_heads != 0
        {
            candle_core::bail!("ragged Metal SDPA tensor geometry differs")
        }

        let device = q.device();
        if device.id() != k.device().id() || device.id() != v.device().id() {
            candle_core::bail!("ragged Metal SDPA tensors are on different devices")
        }
        let pipeline = pipeline(device, head_dim)?;
        if !super::metal_ragged_sdpa_launch_supported(
            pipeline.as_ref().threadExecutionWidth(),
            pipeline.max_total_threads_per_threadgroup(),
        ) {
            candle_core::bail!("ragged Metal SDPA requires thirty-two 32-thread SIMD groups")
        }
        let output_elements = q_layout.shape().elem_count();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, DType::F16)
            .with_label("dinf-ragged-sdpa-f16")
            .build()?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-ragged-sdpa-f16");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        let offset = |layout: &Layout| layout.start_offset() * DType::F16.size_in_bytes();
        encoder_ref.set_input_buffer(0, Some(q.buffer()), offset(q_layout));
        encoder_ref.set_input_buffer(1, Some(k.buffer()), offset(k_layout));
        encoder_ref.set_input_buffer(2, Some(v.buffer()), offset(v_layout));
        encoder_ref.set_output_buffer(3, Some(&output), 0);
        encoder_ref.set_bytes(4, &((q_heads / k_heads) as u32));
        encoder_ref.set_bytes(5, &(q_heads as u32));
        encoder_ref.set_bytes(6, &k_layout.stride()[1]);
        encoder_ref.set_bytes(7, &v_layout.stride()[1]);
        encoder_ref.set_bytes(8, &self.scale);
        let lengths = self
            .valid_lengths
            .iter()
            .map(|length| *length as u32)
            .collect::<Vec<_>>();
        encoder_ref.set_bytes_directly(
            9,
            std::mem::size_of_val(lengths.as_slice()),
            lengths.as_ptr().cast(),
        );
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: 1,
                height: batch * q_heads,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        );

        Ok((
            MetalStorage::new(output, device.clone(), output_elements, DType::F16),
            q_layout.shape().clone(),
        ))
    }
}

pub(super) fn f16_ragged_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    valid_lengths: &[usize],
    scale: f32,
) -> Result<Tensor> {
    q.apply_op3_no_bwd(
        k,
        v,
        &RaggedSdpa {
            valid_lengths,
            scale,
        },
    )
}

pub(super) fn ragged_rope(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &[usize],
    neox: bool,
) -> Result<(Tensor, Tensor)> {
    let op = RaggedRope {
        cos,
        sin,
        positions: positions.iter().map(|position| *position as u32).collect(),
        neox,
    };
    let q = q.apply_op1_no_bwd(&op)?;
    let k = k.apply_op1_no_bwd(&op)?;
    Ok((q, k))
}
