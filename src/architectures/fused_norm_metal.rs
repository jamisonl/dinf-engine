use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, CustomOp3, DType, Layout, MetalStorage, Result, Shape, Tensor,
};
use candle_metal_kernels::metal::ComputePipeline;

const FUSED_NORM_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void dinf_add_rmsnorm_f32(
    device const float *lhs [[buffer(0)]],
    device const float *rhs [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &elements [[buffer(4)]],
    constant uint &hidden_dim [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  threadgroup float sums[1024];
  const uint row_offset = row * hidden_dim;

  float sum_sq = 0.0f;
  for (uint column = tid; column < hidden_dim; column += threads) {
    const uint offset = row_offset + column;
    const float value = lhs[offset] + rhs[offset];
    out[elements + offset] = value;
    sum_sq += value * value;
  }
  sums[tid] = sum_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sums[tid] += sums[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float inv_rms = rsqrt(sums[0] / float(hidden_dim) + eps);
  for (uint column = tid; column < hidden_dim; column += threads) {
    const uint offset = row_offset + column;
    out[offset] = out[elements + offset] * inv_rms * weight[column];
  }
}

// dinf_rmsnorm_f32(update, post_weight) added to `residual`, times scale[0]
// when bit 0 of `flags` is set. With bit 1 set, the sum goes to the second
// plane and dinf_rmsnorm_f32(sum, next_weight) to the first, as
// dinf_add_rmsnorm_f32 packs them. Bits 2 and 3 append planes normalized by
// the same RMS with `second_weight` and with no weight, in that order. Thread
// width and reduction order match the separate kernels.
kernel void dinf_post_norm_residual_f32(
    device const float *update [[buffer(0)]],
    device const float *residual [[buffer(1)]],
    device const float *post_weight [[buffer(2)]],
    device const float *next_weight [[buffer(3)]],
    device const float *scale [[buffer(4)]],
    device float *out [[buffer(5)]],
    constant uint &elements [[buffer(6)]],
    constant uint &hidden_dim [[buffer(7)]],
    constant float &post_eps [[buffer(8)]],
    constant float &next_eps [[buffer(9)]],
    constant uint &flags [[buffer(10)]],
    device const float *second_weight [[buffer(11)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  threadgroup float sums[1024];
  const uint row_offset = row * hidden_dim;
  const bool has_scale = (flags & 1u) != 0;
  const bool has_next = (flags & 2u) != 0;
  const bool has_second = (flags & 4u) != 0;
  const bool has_unweighted = (flags & 8u) != 0;
  const uint unweighted_plane = (has_second ? 3u : 2u) * elements;

  float sum_sq = 0.0f;
  for (uint column = tid; column < hidden_dim; column += threads) {
    const float value = update[row_offset + column];
    sum_sq += value * value;
  }
  sums[tid] = sum_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sums[tid] += sums[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float inv_post = rsqrt(sums[0] / float(hidden_dim) + post_eps);
  const float row_scale = has_scale ? scale[0] : 1.0f;
  const uint sum_plane = has_next ? elements : 0;

  float next_sum_sq = 0.0f;
  for (uint column = tid; column < hidden_dim; column += threads) {
    const uint offset = row_offset + column;
    const float normalized = update[offset] * inv_post * post_weight[column];
    float value = residual[offset] + normalized;
    if (has_scale) {
      value = value * row_scale;
    }
    out[sum_plane + offset] = value;
    next_sum_sq += value * value;
  }
  if (!has_next) {
    return;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  sums[tid] = next_sum_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sums[tid] += sums[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float inv_next = rsqrt(sums[0] / float(hidden_dim) + next_eps);
  for (uint column = tid; column < hidden_dim; column += threads) {
    const uint offset = row_offset + column;
    const float normalized = out[elements + offset] * inv_next;
    out[offset] = normalized * next_weight[column];
    if (has_second) {
      out[2 * elements + offset] = normalized * second_weight[column];
    }
    if (has_unweighted) {
      out[unweighted_plane + offset] = normalized;
    }
  }
}

// Gemma 4 MoE FFN tail: rmsnorm(shared) * shared_weight plus
// rmsnorm(routed) * routed_weight, then rmsnorm(that) * post_weight added to
// `residual`, times scale[0] when bit 0 of `flags` is set. With bit 1 set, the
// result moves to the second plane and the first holds rmsnorm(result) *
// next_weight, the next layer's input (projections read it at offset zero).
kernel void dinf_dual_post_norm_residual_f32(
    device const float *shared_in [[buffer(0)]],
    device const float *routed_in [[buffer(1)]],
    device const float *residual [[buffer(2)]],
    device const float *shared_weight [[buffer(3)]],
    device const float *routed_weight [[buffer(4)]],
    device const float *post_weight [[buffer(5)]],
    device const float *scale [[buffer(6)]],
    device float *out [[buffer(7)]],
    constant uint &hidden_dim [[buffer(8)]],
    constant float &eps [[buffer(9)]],
    constant uint &flags [[buffer(10)]],
    device const float *next_weight [[buffer(11)]],
    constant uint &elements [[buffer(12)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  threadgroup float shared_sums[1024];
  threadgroup float routed_sums[1024];
  const uint row_offset = row * hidden_dim;

  float shared_sq = 0.0f;
  float routed_sq = 0.0f;
  for (uint column = tid; column < hidden_dim; column += threads) {
    const float shared = shared_in[row_offset + column];
    const float routed = routed_in[row_offset + column];
    shared_sq += shared * shared;
    routed_sq += routed * routed;
  }
  shared_sums[tid] = shared_sq;
  routed_sums[tid] = routed_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_sums[tid] += shared_sums[tid + stride];
      routed_sums[tid] += routed_sums[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float inv_shared = rsqrt(shared_sums[0] / float(hidden_dim) + eps);
  const float inv_routed = rsqrt(routed_sums[0] / float(hidden_dim) + eps);

  const uint sum_plane = (flags & 2u) != 0 ? elements : 0;
  float combined_sq = 0.0f;
  for (uint column = tid; column < hidden_dim; column += threads) {
    const uint offset = row_offset + column;
    const float combined = shared_in[offset] * inv_shared * shared_weight[column]
        + routed_in[offset] * inv_routed * routed_weight[column];
    out[sum_plane + offset] = combined;
    combined_sq += combined * combined;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  shared_sums[tid] = combined_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_sums[tid] += shared_sums[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float inv_post = rsqrt(shared_sums[0] / float(hidden_dim) + eps);
  const float row_scale = (flags & 1u) != 0 ? scale[0] : 1.0f;
  float next_sq = 0.0f;
  for (uint column = tid; column < hidden_dim; column += threads) {
    const uint offset = row_offset + column;
    float value = residual[offset] + out[sum_plane + offset] * inv_post * post_weight[column];
    if ((flags & 1u) != 0) {
      value = value * row_scale;
    }
    out[sum_plane + offset] = value;
    next_sq += value * value;
  }
  if ((flags & 2u) == 0) {
    return;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  shared_sums[tid] = next_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_sums[tid] += shared_sums[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float inv_next = rsqrt(shared_sums[0] / float(hidden_dim) + eps);
  for (uint column = tid; column < hidden_dim; column += threads) {
    const uint offset = row_offset + column;
    out[offset] = out[elements + offset] * inv_next * next_weight[column];
  }
}

kernel void dinf_rmsnorm_f32(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &hidden_dim [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  threadgroup float sums[1024];
  const uint row_offset = row * hidden_dim;

  float sum_sq = 0.0f;
  for (uint column = tid; column < hidden_dim; column += threads) {
    const float value = input[row_offset + column];
    sum_sq += value * value;
  }
  sums[tid] = sum_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sums[tid] += sums[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float inv_rms = rsqrt(sums[0] / float(hidden_dim) + eps);
  for (uint column = tid; column < hidden_dim; column += threads) {
    const uint offset = row_offset + column;
    out[offset] = input[offset] * inv_rms * weight[column];
  }
}

kernel void dinf_rmsnorm_no_weight_f32(
    device const float *input [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &hidden_dim [[buffer(2)]],
    constant float &eps [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  threadgroup float sums[1024];
  const uint row_offset = row * hidden_dim;

  float sum_sq = 0.0f;
  for (uint column = tid; column < hidden_dim; column += threads) {
    const float value = input[row_offset + column];
    sum_sq += value * value;
  }
  sums[tid] = sum_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sums[tid] += sums[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float inv_rms = rsqrt(sums[0] / float(hidden_dim) + eps);
  for (uint column = tid; column < hidden_dim; column += threads) {
    const uint offset = row_offset + column;
    out[offset] = input[offset] * inv_rms;
  }
}

kernel void dinf_head_rmsnorm_f32(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &seq_len [[buffer(3)]],
    constant uint &heads [[buffer(4)]],
    constant uint &head_dim [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  threadgroup float sums[1024];
  const uint token = row % seq_len;
  const uint lane_head = row / seq_len;
  const uint head = lane_head % heads;
  const uint lane = lane_head / heads;
  const uint source_offset = ((lane * seq_len + token) * heads + head) * head_dim;
  const uint output_offset = row * head_dim;

  float sum_sq = 0.0f;
  for (uint column = tid; column < head_dim; column += threads) {
    const float value = input[source_offset + column];
    sum_sq += value * value;
  }
  sums[tid] = sum_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sums[tid] += sums[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float inv_rms = rsqrt(sums[0] / float(head_dim) + eps);
  for (uint column = tid; column < head_dim; column += threads) {
    out[output_offset + column] =
        input[source_offset + column] * inv_rms * weight[column];
  }
}

kernel void dinf_head_rmsnorm_rope_f32(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &seq_len [[buffer(3)]],
    constant uint &heads [[buffer(4)]],
    constant uint &head_dim [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    device const float *cos_table [[buffer(7)]],
    device const float *sin_table [[buffer(8)]],
    constant uint &start_pos [[buffer(9)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  // dinf_head_rmsnorm_f32 followed by NeoX rotary embedding (Candle's
  // rope kernel arithmetic), without storing the normalized intermediate.
  threadgroup float sums[1024];
  const uint token = row % seq_len;
  const uint lane_head = row / seq_len;
  const uint head = lane_head % heads;
  const uint lane = lane_head / heads;
  const uint source_offset = ((lane * seq_len + token) * heads + head) * head_dim;
  const uint output_offset = row * head_dim;

  float sum_sq = 0.0f;
  for (uint column = tid; column < head_dim; column += threads) {
    const float value = input[source_offset + column];
    sum_sq += value * value;
  }
  sums[tid] = sum_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sums[tid] += sums[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float inv_rms = rsqrt(sums[0] / float(head_dim) + eps);
  const uint half_dim = head_dim / 2;
  const uint table = (start_pos + token) * half_dim;
  for (uint column = tid; column < half_dim; column += threads) {
    const float x1 = input[source_offset + column] * inv_rms * weight[column];
    const float x2 = input[source_offset + column + half_dim] * inv_rms
        * weight[column + half_dim];
    const float c = cos_table[table + column];
    const float s = sin_table[table + column];
    out[output_offset + column] = x1 * c - x2 * s;
    out[output_offset + column + half_dim] = x1 * s + x2 * c;
  }
}

kernel void dinf_head_rmsnorm_no_weight_f32(
    device const float *input [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &seq_len [[buffer(2)]],
    constant uint &heads [[buffer(3)]],
    constant uint &head_dim [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  threadgroup float sums[1024];
  const uint token = row % seq_len;
  const uint lane_head = row / seq_len;
  const uint head = lane_head % heads;
  const uint lane = lane_head / heads;
  const uint source_offset = ((lane * seq_len + token) * heads + head) * head_dim;
  const uint output_offset = row * head_dim;

  float sum_sq = 0.0f;
  for (uint column = tid; column < head_dim; column += threads) {
    const float value = input[source_offset + column];
    sum_sq += value * value;
  }
  sums[tid] = sum_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sums[tid] += sums[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float inv_rms = rsqrt(sums[0] / float(head_dim) + eps);
  for (uint column = tid; column < head_dim; column += threads) {
    out[output_offset + column] = input[source_offset + column] * inv_rms;
  }
}

kernel void dinf_head_rmsnorm_silu_gate_f32(
    device const float *input [[buffer(0)]],
    device const float *gate [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &head_dim [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  threadgroup float sums[1024];
  const uint row_offset = row * head_dim;

  float sum_sq = 0.0f;
  for (uint column = tid; column < head_dim; column += threads) {
    const float value = input[row_offset + column];
    sum_sq += value * value;
  }
  sums[tid] = sum_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sums[tid] += sums[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float inv_rms = rsqrt(sums[0] / float(head_dim) + eps);
  for (uint column = tid; column < head_dim; column += threads) {
    const uint offset = row_offset + column;
    const float gate_value = gate[offset];
    out[offset] = (gate_value / (1.0f + exp(-gate_value)))
        * input[offset] * inv_rms * weight[column];
  }
}

kernel void dinf_rmsnorm_epilogue_f32(
    device const float *input [[buffer(0)]],
    device const float *variance [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &elements [[buffer(4)]],
    constant uint &hidden_dim [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < elements) {
    const uint row = gid / hidden_dim;
    const uint column = gid - row * hidden_dim;
    volatile float shifted = variance[row] + eps;
    volatile float denominator = sqrt(shifted);
    volatile float normalized = input[gid] / denominator;
    out[gid] = normalized * weight[column];
  }
}
"#;

static PIPELINES: OnceLock<Mutex<HashMap<(DeviceId, &'static str), ComputePipeline>>> =
    OnceLock::new();

fn pipeline(device: &candle_core::MetalDevice, name: &'static str) -> Result<ComputePipeline> {
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (device.id(), name);
    if let Some(pipeline) = pipelines.lock().unwrap().get(&key) {
        return Ok(pipeline.clone());
    }
    let library = device
        .metal_device()
        .new_library_with_source(FUSED_NORM_MSL, None)
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

struct AddRmsNorm {
    eps: f32,
}

impl CustomOp3 for AddRmsNorm {
    fn name(&self) -> &'static str {
        "dinf-metal-add-rmsnorm"
    }

    fn cpu_fwd(
        &self,
        _lhs: &CpuStorage,
        _lhs_layout: &Layout,
        _rhs: &CpuStorage,
        _rhs_layout: &Layout,
        _weight: &CpuStorage,
        _weight_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused add RMSNorm is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        lhs: &MetalStorage,
        lhs_layout: &Layout,
        rhs: &MetalStorage,
        rhs_layout: &Layout,
        weight: &MetalStorage,
        weight_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if lhs.dtype() != DType::F32
            || rhs.dtype() != DType::F32
            || weight.dtype() != DType::F32
            || !lhs_layout.is_contiguous()
            || !rhs_layout.is_contiguous()
            || !weight_layout.is_contiguous()
            || lhs_layout.shape() != rhs_layout.shape()
        {
            candle_core::bail!("fused add RMSNorm requires matching contiguous f32 tensors")
        }
        let hidden_dim = lhs_layout.shape().dims().last().copied().unwrap_or(0);
        let elements = lhs_layout.shape().elem_count();
        if hidden_dim == 0
            || elements == 0
            || weight_layout.shape().dims1()? != hidden_dim
            || !elements.is_multiple_of(hidden_dim)
        {
            candle_core::bail!("fused add RMSNorm geometry differs")
        }
        if lhs.device().id() != rhs.device().id() || lhs.device().id() != weight.device().id() {
            candle_core::bail!("fused add RMSNorm tensors are on different Metal devices")
        }

        let device = lhs.device();
        let output_elements = elements * 2;
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, DType::F32)
            .with_label("dinf-add-rmsnorm")
            .build()?;
        let pipeline = pipeline(device, "dinf_add_rmsnorm_f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-add-rmsnorm");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(lhs.buffer()),
            lhs_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            1,
            Some(rhs.buffer()),
            rhs_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            2,
            Some(weight.buffer()),
            weight_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(3, Some(&output), 0);
        encoder_ref.set_bytes(4, &(elements as u32));
        encoder_ref.set_bytes(5, &(hidden_dim as u32));
        encoder_ref.set_bytes(6, &self.eps);

        let thread_width = norm_thread_width(hidden_dim, &pipeline);
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: elements / hidden_dim,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: thread_width,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), output_elements, DType::F32),
            Shape::from(output_elements),
        ))
    }
}

struct RmsNorm {
    eps: f32,
}

impl CustomOp2 for RmsNorm {
    fn name(&self) -> &'static str {
        "dinf-metal-rmsnorm"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _weight: &CpuStorage,
        _weight_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused RMSNorm is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        input_layout: &Layout,
        weight: &MetalStorage,
        weight_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if input.dtype() != DType::F32
            || weight.dtype() != DType::F32
            || !input_layout.is_contiguous()
            || !weight_layout.is_contiguous()
        {
            candle_core::bail!("fused RMSNorm requires contiguous f32 tensors")
        }
        let hidden_dim = input_layout.shape().dims().last().copied().unwrap_or(0);
        let elements = input_layout.shape().elem_count();
        if hidden_dim == 0
            || elements == 0
            || !elements.is_multiple_of(hidden_dim)
            || weight_layout.shape().dims1()? != hidden_dim
            || hidden_dim > u32::MAX as usize
            || elements > u32::MAX as usize
        {
            candle_core::bail!("fused RMSNorm geometry is unsupported")
        }
        if input.device().id() != weight.device().id() {
            candle_core::bail!("fused RMSNorm tensors are on different Metal devices")
        }

        let device = input.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-rmsnorm")
            .build()?;
        let pipeline = pipeline(device, "dinf_rmsnorm_f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-rmsnorm");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(input.buffer()),
            input_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            1,
            Some(weight.buffer()),
            weight_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(2, Some(&output), 0);
        encoder_ref.set_bytes(3, &(hidden_dim as u32));
        encoder_ref.set_bytes(4, &self.eps);

        let thread_width = norm_thread_width(hidden_dim, &pipeline);
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: elements / hidden_dim,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: thread_width,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), elements, DType::F32),
            input_layout.shape().clone(),
        ))
    }
}

struct RmsNormNoWeight {
    eps: f32,
}

impl CustomOp1 for RmsNormNoWeight {
    fn name(&self) -> &'static str {
        "dinf-metal-rmsnorm-no-weight"
    }

    fn cpu_fwd(&self, _input: &CpuStorage, _input_layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused unweighted RMSNorm is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        input_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if input.dtype() != DType::F32 || !input_layout.is_contiguous() {
            candle_core::bail!("fused unweighted RMSNorm requires a contiguous f32 tensor")
        }
        let hidden_dim = input_layout.shape().dims().last().copied().unwrap_or(0);
        let elements = input_layout.shape().elem_count();
        if hidden_dim == 0
            || elements == 0
            || !elements.is_multiple_of(hidden_dim)
            || hidden_dim > u32::MAX as usize
            || elements > u32::MAX as usize
        {
            candle_core::bail!("fused unweighted RMSNorm geometry is unsupported")
        }

        let device = input.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-rmsnorm-no-weight")
            .build()?;
        let pipeline = pipeline(device, "dinf_rmsnorm_no_weight_f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-rmsnorm-no-weight");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(input.buffer()),
            input_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(1, Some(&output), 0);
        encoder_ref.set_bytes(2, &(hidden_dim as u32));
        encoder_ref.set_bytes(3, &self.eps);

        let thread_width = norm_thread_width(hidden_dim, &pipeline);
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: elements / hidden_dim,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: thread_width,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), elements, DType::F32),
            input_layout.shape().clone(),
        ))
    }
}

struct HeadRmsNorm {
    batch: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
}

impl CustomOp2 for HeadRmsNorm {
    fn name(&self) -> &'static str {
        "dinf-metal-head-rmsnorm"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _weight: &CpuStorage,
        _weight_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused head RMSNorm is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        input_layout: &Layout,
        weight: &MetalStorage,
        weight_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let fused_dim = self
            .heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| candle_core::Error::Msg("fused head RMSNorm overflow".into()))?;
        if input.dtype() != DType::F32
            || weight.dtype() != DType::F32
            || !input_layout.is_contiguous()
            || !weight_layout.is_contiguous()
            || input_layout.shape().dims() != [self.batch, self.seq_len, fused_dim]
            || weight_layout.shape().dims1()? != self.head_dim
        {
            candle_core::bail!("fused head RMSNorm requires matching contiguous f32 tensors")
        }
        let elements = input_layout.shape().elem_count();
        if self.batch == 0
            || self.seq_len == 0
            || self.heads == 0
            || self.head_dim == 0
            || self.seq_len > u32::MAX as usize
            || self.heads > u32::MAX as usize
            || self.head_dim > u32::MAX as usize
            || elements > u32::MAX as usize
        {
            candle_core::bail!("fused head RMSNorm geometry is unsupported")
        }
        if input.device().id() != weight.device().id() {
            candle_core::bail!("fused head RMSNorm tensors are on different Metal devices")
        }

        let device = input.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-head-rmsnorm")
            .build()?;
        let pipeline = pipeline(device, "dinf_head_rmsnorm_f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-head-rmsnorm");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(input.buffer()),
            input_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            1,
            Some(weight.buffer()),
            weight_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(2, Some(&output), 0);
        encoder_ref.set_bytes(3, &(self.seq_len as u32));
        encoder_ref.set_bytes(4, &(self.heads as u32));
        encoder_ref.set_bytes(5, &(self.head_dim as u32));
        encoder_ref.set_bytes(6, &self.eps);
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: elements / self.head_dim,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: norm_thread_width(self.head_dim, &pipeline),
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), elements, DType::F32),
            Shape::from_dims(&[self.batch, self.heads, self.seq_len, self.head_dim]),
        ))
    }
}

struct HeadRmsNormNoWeight {
    batch: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
}

impl CustomOp1 for HeadRmsNormNoWeight {
    fn name(&self) -> &'static str {
        "dinf-metal-head-rmsnorm-no-weight"
    }

    fn cpu_fwd(&self, _input: &CpuStorage, _input_layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused unweighted head RMSNorm is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        input_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let fused_dim = self.heads.checked_mul(self.head_dim).ok_or_else(|| {
            candle_core::Error::Msg("fused unweighted head RMSNorm overflow".into())
        })?;
        if input.dtype() != DType::F32
            || !input_layout.is_contiguous()
            || input_layout.shape().dims() != [self.batch, self.seq_len, fused_dim]
        {
            candle_core::bail!("fused unweighted head RMSNorm requires a contiguous f32 tensor")
        }
        let elements = input_layout.shape().elem_count();
        if self.batch == 0
            || self.seq_len == 0
            || self.heads == 0
            || self.head_dim == 0
            || self.seq_len > u32::MAX as usize
            || self.heads > u32::MAX as usize
            || self.head_dim > u32::MAX as usize
            || elements > u32::MAX as usize
        {
            candle_core::bail!("fused unweighted head RMSNorm geometry is unsupported")
        }

        let device = input.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-head-rmsnorm-no-weight")
            .build()?;
        let pipeline = pipeline(device, "dinf_head_rmsnorm_no_weight_f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-head-rmsnorm-no-weight");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(input.buffer()),
            input_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(1, Some(&output), 0);
        encoder_ref.set_bytes(2, &(self.seq_len as u32));
        encoder_ref.set_bytes(3, &(self.heads as u32));
        encoder_ref.set_bytes(4, &(self.head_dim as u32));
        encoder_ref.set_bytes(5, &self.eps);
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: elements / self.head_dim,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: norm_thread_width(self.head_dim, &pipeline),
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), elements, DType::F32),
            Shape::from_dims(&[self.batch, self.heads, self.seq_len, self.head_dim]),
        ))
    }
}

struct HeadRmsNormSiluGate {
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
}

impl CustomOp3 for HeadRmsNormSiluGate {
    fn name(&self) -> &'static str {
        "dinf-metal-head-rmsnorm-silu-gate"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _gate: &CpuStorage,
        _gate_layout: &Layout,
        _weight: &CpuStorage,
        _weight_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused head RMSNorm SiLU gate is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        input_layout: &Layout,
        gate: &MetalStorage,
        gate_layout: &Layout,
        weight: &MetalStorage,
        weight_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let fused_dim = self.heads.checked_mul(self.head_dim).ok_or_else(|| {
            candle_core::Error::Msg("fused head RMSNorm SiLU gate overflow".into())
        })?;
        if input.dtype() != DType::F32
            || gate.dtype() != DType::F32
            || weight.dtype() != DType::F32
            || !input_layout.is_contiguous()
            || !gate_layout.is_contiguous()
            || !weight_layout.is_contiguous()
            || input_layout.shape().dims() != [self.seq_len, self.heads, self.head_dim]
            || gate_layout.shape().dims() != [1, self.seq_len, fused_dim]
            || weight_layout.shape().dims1()? != self.head_dim
        {
            candle_core::bail!("fused head RMSNorm SiLU gate geometry differs")
        }
        let elements = input_layout.shape().elem_count();
        if self.seq_len == 0
            || self.heads == 0
            || self.head_dim == 0
            || elements != gate_layout.shape().elem_count()
            || self.head_dim > u32::MAX as usize
        {
            candle_core::bail!("fused head RMSNorm SiLU gate geometry is unsupported")
        }
        if input.device().id() != gate.device().id() || input.device().id() != weight.device().id()
        {
            candle_core::bail!("fused head RMSNorm SiLU gate tensors use different devices")
        }

        let device = input.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-head-rmsnorm-silu-gate")
            .build()?;
        let pipeline = pipeline(device, "dinf_head_rmsnorm_silu_gate_f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-head-rmsnorm-silu-gate");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(input.buffer()),
            input_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            1,
            Some(gate.buffer()),
            gate_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            2,
            Some(weight.buffer()),
            weight_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(3, Some(&output), 0);
        encoder_ref.set_bytes(4, &(self.head_dim as u32));
        encoder_ref.set_bytes(5, &self.eps);
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: self.seq_len * self.heads,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: norm_thread_width(self.head_dim, &pipeline),
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), elements, DType::F32),
            gate_layout.shape().clone(),
        ))
    }
}

fn norm_thread_width(hidden_dim: usize, pipeline: &ComputePipeline) -> usize {
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(1);
    let max_power_of_two = 1usize << max_threads.ilog2();
    hidden_dim
        .div_ceil(2)
        .next_power_of_two()
        .min(max_power_of_two)
        .max(1)
}

struct RmsNormEpilogue {
    eps: f32,
}

impl CustomOp3 for RmsNormEpilogue {
    fn name(&self) -> &'static str {
        "dinf-metal-rmsnorm-epilogue"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _variance: &CpuStorage,
        _variance_layout: &Layout,
        _weight: &CpuStorage,
        _weight_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused RMSNorm epilogue is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        input_layout: &Layout,
        variance: &MetalStorage,
        variance_layout: &Layout,
        weight: &MetalStorage,
        weight_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if input.dtype() != DType::F32
            || variance.dtype() != DType::F32
            || weight.dtype() != DType::F32
            || !input_layout.is_contiguous()
            || !variance_layout.is_contiguous()
            || !weight_layout.is_contiguous()
        {
            candle_core::bail!("fused RMSNorm epilogue requires contiguous f32 tensors")
        }
        let hidden_dim = input_layout.shape().dims().last().copied().unwrap_or(0);
        let elements = input_layout.shape().elem_count();
        let rows = variance_layout.shape().elem_count();
        if hidden_dim == 0
            || elements == 0
            || !elements.is_multiple_of(hidden_dim)
            || rows != elements / hidden_dim
            || variance_layout.shape().dims().last().copied() != Some(1)
            || weight_layout.shape().dims1()? != hidden_dim
        {
            candle_core::bail!("fused RMSNorm epilogue geometry differs")
        }
        if input.device().id() != variance.device().id()
            || input.device().id() != weight.device().id()
        {
            candle_core::bail!("fused RMSNorm epilogue tensors are on different Metal devices")
        }

        let device = input.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-rmsnorm-epilogue")
            .build()?;
        let pipeline = pipeline(device, "dinf_rmsnorm_epilogue_f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-rmsnorm-epilogue");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(input.buffer()),
            input_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            1,
            Some(variance.buffer()),
            variance_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            2,
            Some(weight.buffer()),
            weight_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(3, Some(&output), 0);
        encoder_ref.set_bytes(4, &(elements as u32));
        encoder_ref.set_bytes(5, &(hidden_dim as u32));
        encoder_ref.set_bytes(6, &self.eps);
        encoder_ref.dispatch_threads(
            objc2_metal::MTLSize {
                width: elements,
                height: 1,
                depth: 1,
            },
            candle_metal_kernels::utils::get_block_dims(elements.min(256), 1, 1),
        );
        Ok((
            MetalStorage::new(output, device.clone(), elements, DType::F32),
            input_layout.shape().clone(),
        ))
    }
}

pub(super) fn try_fused_rms_norm_epilogue(
    input: &Tensor,
    variance: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Option<Tensor>> {
    if !input.device().is_metal()
        || input.dtype() != DType::F32
        || variance.dtype() != DType::F32
        || weight.dtype() != DType::F32
        || !input.is_contiguous()
        || !variance.is_contiguous()
        || !weight.is_contiguous()
    {
        return Ok(None);
    }
    Ok(Some(input.apply_op3_no_bwd(
        variance,
        weight,
        &RmsNormEpilogue { eps },
    )?))
}

pub(super) fn try_fused_rms_norm(
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Option<Tensor>> {
    if std::env::var_os("DINF_DISABLE_FUSED_RMSNORM").is_some()
        || !input.device().is_metal()
        || input.dtype() != DType::F32
        || weight.dtype() != DType::F32
        || !input.is_contiguous()
        || !weight.is_contiguous()
        || input.dim(candle_core::D::Minus1)? != weight.dims1()?
    {
        return Ok(None);
    }
    Ok(Some(input.apply_op2_no_bwd(weight, &RmsNorm { eps })?))
}

pub(super) fn try_fused_rms_norm_no_weight(input: &Tensor, eps: f32) -> Result<Option<Tensor>> {
    if std::env::var_os("DINF_DISABLE_FUSED_RMSNORM").is_some()
        || !input.device().is_metal()
        || input.dtype() != DType::F32
        || !input.is_contiguous()
    {
        return Ok(None);
    }
    Ok(Some(input.apply_op1_no_bwd(&RmsNormNoWeight { eps })?))
}

pub(super) fn try_fused_head_rms_norm(
    input: &Tensor,
    weight: &Tensor,
    batch: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<Option<Tensor>> {
    let Some(fused_dim) = heads.checked_mul(head_dim) else {
        return Ok(None);
    };
    if std::env::var_os("DINF_DISABLE_FUSED_RMSNORM").is_some()
        || std::env::var_os("DINF_DISABLE_FUSED_HEAD_RMSNORM").is_some()
        || !input.device().is_metal()
        || input.dtype() != DType::F32
        || weight.dtype() != DType::F32
        || !input.is_contiguous()
        || !weight.is_contiguous()
        || input.dims() != [batch, seq_len, fused_dim]
        || weight.dims() != [head_dim]
    {
        return Ok(None);
    }
    Ok(Some(input.apply_op2_no_bwd(
        weight,
        &HeadRmsNorm {
            batch,
            seq_len,
            heads,
            head_dim,
            eps,
        },
    )?))
}

/// Per-head RMSNorm plus NeoX rotary embedding, `[batch, seq, heads * dim]`
/// to `[batch, heads, seq, dim]`, in one dispatch. `cos`/`sin` are
/// `[positions, dim / 2]` tables; rows start at `start_pos`.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_fused_head_rms_norm_rope(
    input: &Tensor,
    weight: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    start_pos: usize,
    batch: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<Option<Tensor>> {
    let Some(fused_dim) = heads.checked_mul(head_dim) else {
        return Ok(None);
    };
    if std::env::var_os("DINF_DISABLE_FUSED_RMSNORM").is_some()
        || std::env::var_os("DINF_DISABLE_FUSED_HEAD_RMSNORM_ROPE").is_some()
        || !input.device().is_metal()
        || [input, weight, cos, sin]
            .iter()
            .any(|t| t.dtype() != DType::F32 || !t.is_contiguous())
        || !input.device().same_device(cos.device())
        || !input.device().same_device(sin.device())
        || input.dims() != [batch, seq_len, fused_dim]
        || weight.dims() != [head_dim]
        || !head_dim.is_multiple_of(2)
        || cos.dims() != sin.dims()
        || cos.dims().len() != 2
        || cos.dim(1)? != head_dim / 2
        || start_pos
            .checked_add(seq_len)
            .is_none_or(|end| end > cos.dim(0).unwrap_or(0))
        || start_pos > u32::MAX as usize
    {
        return Ok(None);
    }
    Ok(Some(input.apply_op3_no_bwd(
        weight,
        cos,
        &HeadRmsNormRope {
            batch,
            seq_len,
            heads,
            head_dim,
            eps,
            start_pos,
            sin: sin.clone(),
        },
    )?))
}

struct HeadRmsNormRope {
    batch: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
    start_pos: usize,
    sin: Tensor,
}

impl candle_core::CustomOp3 for HeadRmsNormRope {
    fn name(&self) -> &'static str {
        "dinf-metal-head-rmsnorm-rope"
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
        candle_core::bail!("fused head RMSNorm + rope is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        input_layout: &Layout,
        weight: &MetalStorage,
        weight_layout: &Layout,
        cos: &MetalStorage,
        cos_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let fused_dim = self.heads * self.head_dim;
        let elements = input_layout.shape().elem_count();
        if input_layout.shape().dims() != [self.batch, self.seq_len, fused_dim]
            || !input_layout.is_contiguous()
            || !weight_layout.is_contiguous()
            || !cos_layout.is_contiguous()
            || elements > u32::MAX as usize
            || cos.dtype() != DType::F32
            || weight.dtype() != DType::F32
            || input.dtype() != DType::F32
        {
            candle_core::bail!("fused head RMSNorm + rope tensor metadata changed")
        }
        let (sin_storage, sin_layout) = self.sin.storage_and_layout();
        let candle_core::Storage::Metal(sin) = &*sin_storage else {
            candle_core::bail!("rope sine table must be on Metal")
        };
        if !sin_layout.is_contiguous() || sin_layout.shape() != cos_layout.shape() {
            candle_core::bail!("rope sine table layout differs")
        }
        let device = input.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-head-rmsnorm-rope")
            .build()?;
        let pipeline = pipeline(device, "dinf_head_rmsnorm_rope_f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-head-rmsnorm-rope");
        encoder.set_compute_pipeline_state(&pipeline);
        let e: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        let f32 = DType::F32.size_in_bytes();
        e.set_input_buffer(0, Some(input.buffer()), input_layout.start_offset() * f32);
        e.set_input_buffer(1, Some(weight.buffer()), weight_layout.start_offset() * f32);
        e.set_output_buffer(2, Some(&output), 0);
        e.set_bytes(3, &(self.seq_len as u32));
        e.set_bytes(4, &(self.heads as u32));
        e.set_bytes(5, &(self.head_dim as u32));
        e.set_bytes(6, &self.eps);
        e.set_input_buffer(7, Some(cos.buffer()), cos_layout.start_offset() * f32);
        e.set_input_buffer(8, Some(sin.buffer()), sin_layout.start_offset() * f32);
        e.set_bytes(9, &(self.start_pos as u32));
        e.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: elements / self.head_dim,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: norm_thread_width(self.head_dim, &pipeline),
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), elements, DType::F32),
            Shape::from_dims(&[self.batch, self.heads, self.seq_len, self.head_dim]),
        ))
    }
}

pub(super) fn try_fused_head_rms_norm_no_weight(
    input: &Tensor,
    batch: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<Option<Tensor>> {
    let Some(fused_dim) = heads.checked_mul(head_dim) else {
        return Ok(None);
    };
    if std::env::var_os("DINF_DISABLE_FUSED_RMSNORM").is_some()
        || std::env::var_os("DINF_DISABLE_FUSED_HEAD_RMSNORM").is_some()
        || !input.device().is_metal()
        || input.dtype() != DType::F32
        || !input.is_contiguous()
        || input.dims() != [batch, seq_len, fused_dim]
    {
        return Ok(None);
    }
    Ok(Some(input.apply_op1_no_bwd(&HeadRmsNormNoWeight {
        batch,
        seq_len,
        heads,
        head_dim,
        eps,
    })?))
}

pub(super) fn try_fused_head_rms_norm_silu_gate(
    input: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<Option<Tensor>> {
    let Some(fused_dim) = heads.checked_mul(head_dim) else {
        return Ok(None);
    };
    if std::env::var_os("DINF_DISABLE_QWEN35_FUSED_SSM_GATE").is_some()
        || std::env::var_os("DINF_DISABLE_FUSED_RMSNORM").is_some()
        || std::env::var_os("DINF_DISABLE_FUSED_HEAD_RMSNORM").is_some()
        || !input.device().is_metal()
        || input.dtype() != DType::F32
        || gate.dtype() != DType::F32
        || weight.dtype() != DType::F32
        || !input.is_contiguous()
        || !gate.is_contiguous()
        || !weight.is_contiguous()
        || input.dims() != [seq_len, heads, head_dim]
        || gate.dims() != [1, seq_len, fused_dim]
        || weight.dims() != [head_dim]
    {
        return Ok(None);
    }
    Ok(Some(input.apply_op3_no_bwd(
        gate,
        weight,
        &HeadRmsNormSiluGate {
            seq_len,
            heads,
            head_dim,
            eps,
        },
    )?))
}

pub(super) fn try_fused_add_rms_norm(
    lhs: &Tensor,
    rhs: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Option<(Tensor, Tensor)>> {
    if std::env::var_os("DINF_DISABLE_FUSED_ADD_RMSNORM").is_some()
        || !lhs.device().is_metal()
        || lhs.dtype() != DType::F32
        || rhs.dtype() != DType::F32
        || weight.dtype() != DType::F32
        || !lhs.is_contiguous()
        || !rhs.is_contiguous()
        || !weight.is_contiguous()
        || lhs.shape() != rhs.shape()
        || lhs.dim(candle_core::D::Minus1)? != weight.dims1()?
    {
        return Ok(None);
    }

    let shape = lhs.shape().clone();
    let elements = shape.elem_count();
    let packed = lhs.apply_op3_no_bwd(rhs, weight, &AddRmsNorm { eps })?;
    // Quantized Metal matmul paths do not all honor a non-zero input offset.
    // Keep normalized activations, which feed projections, in the first plane.
    let normalized = packed.narrow(0, 0, elements)?.reshape(shape.clone())?;
    let residual = packed.narrow(0, elements, elements)?.reshape(shape)?;
    Ok(Some((residual, normalized)))
}

struct PostNormResidual {
    post_eps: f32,
    post_weight: Tensor,
    next: Option<(Tensor, f32)>,
    scale: Option<Tensor>,
    /// Extra plane normalized with the next RMS and this weight.
    second: Option<Tensor>,
    /// Extra plane normalized with the next RMS and no weight.
    unweighted: bool,
}

impl CustomOp2 for PostNormResidual {
    fn name(&self) -> &'static str {
        "dinf-metal-post-norm-residual"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused post-norm residual is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        update: &MetalStorage,
        update_layout: &Layout,
        residual: &MetalStorage,
        residual_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let hidden_dim = update_layout.shape().dims().last().copied().unwrap_or(0);
        let elements = update_layout.shape().elem_count();
        if update.dtype() != DType::F32
            || residual.dtype() != DType::F32
            || !update_layout.is_contiguous()
            || !residual_layout.is_contiguous()
            || update_layout.shape() != residual_layout.shape()
            || hidden_dim == 0
            || elements == 0
            || !elements.is_multiple_of(hidden_dim)
            || elements
                .checked_mul(2)
                .is_none_or(|n| n > u32::MAX as usize)
            || update.device().id() != residual.device().id()
        {
            candle_core::bail!("fused post-norm residual tensor metadata changed")
        }
        let device = update.device();
        let f32 = DType::F32.size_in_bytes();
        let (post_storage, post_layout) = self.post_weight.storage_and_layout();
        let candle_core::Storage::Metal(post) = &*post_storage else {
            candle_core::bail!("post-norm weight must be on Metal")
        };
        let next_guard = self
            .next
            .as_ref()
            .map(|(weight, _)| weight.storage_and_layout());
        let scale_guard = self.scale.as_ref().map(Tensor::storage_and_layout);
        let metal = |guard: &Option<(
            std::sync::RwLockReadGuard<'_, candle_core::Storage>,
            &Layout,
        )>|
         -> Result<Option<(MetalStorage, usize)>> {
            match guard {
                None => Ok(None),
                Some((storage, layout)) => match &**storage {
                    candle_core::Storage::Metal(storage) if layout.is_contiguous() => {
                        Ok(Some((storage.clone(), layout.start_offset() * f32)))
                    }
                    _ => candle_core::bail!(
                        "fused post-norm residual operands must be contiguous Metal tensors"
                    ),
                },
            }
        };
        let second_guard = self.second.as_ref().map(Tensor::storage_and_layout);
        let next = metal(&next_guard)?;
        let scale = metal(&scale_guard)?;
        let second = metal(&second_guard)?;
        if (second.is_some() || self.unweighted) && next.is_none() {
            candle_core::bail!("extra post-norm planes require the next RMSNorm")
        }
        if !post_layout.is_contiguous() || post_layout.shape().elem_count() != hidden_dim {
            candle_core::bail!("fused post-norm residual weight shape differs")
        }

        let planes = match &next {
            None => 1,
            Some(_) => 2 + usize::from(second.is_some()) + usize::from(self.unweighted),
        };
        let output_elements = elements * planes;
        if output_elements > u32::MAX as usize {
            candle_core::bail!("fused post-norm residual output is too large")
        }
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, DType::F32)
            .with_label("dinf-post-norm-residual")
            .build()?;
        let pipeline = pipeline(device, "dinf_post_norm_residual_f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-post-norm-residual");
        encoder.set_compute_pipeline_state(&pipeline);
        let e: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        e.set_input_buffer(0, Some(update.buffer()), update_layout.start_offset() * f32);
        e.set_input_buffer(
            1,
            Some(residual.buffer()),
            residual_layout.start_offset() * f32,
        );
        let post_offset = post_layout.start_offset() * f32;
        e.set_input_buffer(2, Some(post.buffer()), post_offset);
        // Unused bindings alias the post-norm weight; the kernel never reads them.
        match &next {
            Some((storage, offset)) => e.set_input_buffer(3, Some(storage.buffer()), *offset),
            None => e.set_input_buffer(3, Some(post.buffer()), post_offset),
        }
        match &scale {
            Some((storage, offset)) => e.set_input_buffer(4, Some(storage.buffer()), *offset),
            None => e.set_input_buffer(4, Some(post.buffer()), post_offset),
        }
        e.set_output_buffer(5, Some(&output), 0);
        e.set_bytes(6, &(elements as u32));
        e.set_bytes(7, &(hidden_dim as u32));
        e.set_bytes(8, &self.post_eps);
        e.set_bytes(9, &self.next.as_ref().map_or(0.0f32, |(_, eps)| *eps));
        let flags = u32::from(scale.is_some())
            | (u32::from(next.is_some()) << 1)
            | (u32::from(second.is_some()) << 2)
            | (u32::from(self.unweighted) << 3);
        e.set_bytes(10, &flags);
        match &second {
            Some((storage, offset)) => e.set_input_buffer(11, Some(storage.buffer()), *offset),
            None => e.set_input_buffer(11, Some(post.buffer()), post_offset),
        }
        e.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: elements / hidden_dim,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: norm_thread_width(hidden_dim, &pipeline),
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), output_elements, DType::F32),
            Shape::from(output_elements),
        ))
    }
}

/// `scale * (residual + rmsnorm(update) * post_weight)` in one dispatch,
/// plus `rmsnorm(sum) * next_weight` when `next` is given. Returns the sum and
/// the optional normalized sum, or `None` when the fused path does not apply.
pub(super) fn try_fused_post_norm_residual(
    update: &Tensor,
    residual: &Tensor,
    post_weight: &Tensor,
    post_eps: f32,
    next: Option<(&Tensor, f32)>,
    scale: Option<&Tensor>,
) -> Result<Option<(Tensor, Option<Tensor>)>> {
    let hidden_dim = update.dim(candle_core::D::Minus1)?;
    let f32_on_device = |tensor: &Tensor| {
        tensor.dtype() == DType::F32
            && tensor.is_contiguous()
            && tensor.device().same_device(update.device())
    };
    if std::env::var_os("DINF_DISABLE_FUSED_POST_NORM_RESIDUAL").is_some()
        || !update.device().is_metal()
        || !f32_on_device(update)
        || !f32_on_device(residual)
        || !f32_on_device(post_weight)
        || update.shape() != residual.shape()
        || post_weight.dims() != [hidden_dim]
        || next.is_some_and(|(weight, _)| !f32_on_device(weight) || weight.dims() != [hidden_dim])
        || scale.is_some_and(|scale| !f32_on_device(scale) || scale.elem_count() != 1)
    {
        return Ok(None);
    }
    let shape = update.shape().clone();
    let elements = shape.elem_count();
    let packed = update.apply_op2_no_bwd(
        residual,
        &PostNormResidual {
            post_eps,
            post_weight: post_weight.clone(),
            next: next.map(|(weight, eps)| (weight.clone(), eps)),
            scale: scale.cloned(),
            second: None,
            unweighted: false,
        },
    )?;
    if next.is_none() {
        return Ok(Some((packed.reshape(shape)?, None)));
    }
    // Normalized activations feed projections: keep them at offset zero.
    let normalized = packed.narrow(0, 0, elements)?.reshape(shape.clone())?;
    let sum = packed.narrow(0, elements, elements)?.reshape(shape)?;
    Ok(Some((sum, Some(normalized))))
}

/// Normalized planes of the attention residual for a Gemma 4 MoE layer.
pub(super) struct MoeAttentionPlanes {
    /// `residual + rmsnorm(update) * post_weight`.
    pub(super) sum: Tensor,
    /// `rmsnorm(sum) * shared_weight`, the shared MLP input.
    pub(super) shared: Tensor,
    /// `rmsnorm(sum) * expert_weight`, the routed experts' input.
    pub(super) experts: Tensor,
    /// `rmsnorm(sum)` without a weight, the router input.
    pub(super) router: Tensor,
}

/// [`try_fused_post_norm_residual`] with the next RMSNorm written three ways
/// for a Gemma 4 MoE layer. All three planes share the sum's RMS and `eps`.
pub(super) fn try_fused_moe_attention_planes(
    update: &Tensor,
    residual: &Tensor,
    post_weight: &Tensor,
    shared_weight: &Tensor,
    expert_weight: &Tensor,
    eps: f32,
) -> Result<Option<MoeAttentionPlanes>> {
    let hidden_dim = update.dim(candle_core::D::Minus1)?;
    let ok = |tensor: &Tensor| {
        tensor.dtype() == DType::F32
            && tensor.is_contiguous()
            && tensor.device().same_device(update.device())
    };
    if std::env::var_os("DINF_DISABLE_FUSED_POST_NORM_RESIDUAL").is_some()
        || !update.device().is_metal()
        || ![update, residual, post_weight, shared_weight, expert_weight]
            .iter()
            .all(|tensor| ok(tensor))
        || update.shape() != residual.shape()
        || [post_weight, shared_weight, expert_weight]
            .iter()
            .any(|weight| weight.dims() != [hidden_dim])
    {
        return Ok(None);
    }
    let shape = update.shape().clone();
    let elements = shape.elem_count();
    let packed = update.apply_op2_no_bwd(
        residual,
        &PostNormResidual {
            post_eps: eps,
            post_weight: post_weight.clone(),
            next: Some((shared_weight.clone(), eps)),
            scale: None,
            second: Some(expert_weight.clone()),
            unweighted: true,
        },
    )?;
    let plane = |index: usize| -> Result<Tensor> {
        packed
            .narrow(0, index * elements, elements)?
            .reshape(shape.clone())
    };
    Ok(Some(MoeAttentionPlanes {
        shared: plane(0)?,
        sum: plane(1)?,
        experts: plane(2)?,
        router: plane(3)?,
    }))
}

struct DualPostNormResidual {
    eps: f32,
    shared_weight: Tensor,
    routed_weight: Tensor,
    post_weight: Tensor,
    residual: Tensor,
    scale: Option<Tensor>,
    /// Next layer's input-norm weight for a second, normalized plane.
    next_weight: Option<Tensor>,
}

impl CustomOp2 for DualPostNormResidual {
    fn name(&self) -> &'static str {
        "dinf-metal-dual-post-norm-residual"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused dual post-norm residual is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        shared: &MetalStorage,
        shared_layout: &Layout,
        routed: &MetalStorage,
        routed_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let hidden_dim = shared_layout.shape().dims().last().copied().unwrap_or(0);
        let elements = shared_layout.shape().elem_count();
        if shared.dtype() != DType::F32
            || routed.dtype() != DType::F32
            || !shared_layout.is_contiguous()
            || !routed_layout.is_contiguous()
            || shared_layout.shape() != routed_layout.shape()
            || hidden_dim == 0
            || !elements.is_multiple_of(hidden_dim)
            || elements > u32::MAX as usize
        {
            candle_core::bail!("fused dual post-norm residual tensor metadata changed")
        }
        let f32 = DType::F32.size_in_bytes();
        let next_weight = self.next_weight.as_ref().unwrap_or(&self.post_weight);
        let guards = [
            &self.residual,
            &self.shared_weight,
            &self.routed_weight,
            &self.post_weight,
            next_weight,
        ]
        .map(Tensor::storage_and_layout);
        let mut operands = Vec::with_capacity(5);
        for (storage, layout) in &guards {
            let candle_core::Storage::Metal(storage) = &**storage else {
                candle_core::bail!("fused dual post-norm residual operands must be on Metal")
            };
            if !layout.is_contiguous() {
                candle_core::bail!("fused dual post-norm residual operands must be contiguous")
            }
            operands.push((storage.clone(), layout.start_offset() * f32));
        }
        if guards[0].1.shape() != shared_layout.shape() {
            candle_core::bail!("fused dual post-norm residual shape differs")
        }
        let scale_guard = self.scale.as_ref().map(Tensor::storage_and_layout);
        let scale = match &scale_guard {
            None => None,
            Some((storage, layout)) => match &**storage {
                candle_core::Storage::Metal(storage) => {
                    Some((storage.clone(), layout.start_offset() * f32))
                }
                _ => candle_core::bail!("layer output scale must be on Metal"),
            },
        };
        let device = shared.device();
        let planes = 1 + usize::from(self.next_weight.is_some());
        let output_elements = elements * planes;
        if output_elements > u32::MAX as usize {
            candle_core::bail!("fused dual post-norm residual output is too large")
        }
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, DType::F32)
            .with_label("dinf-dual-post-norm-residual")
            .build()?;
        let pipeline = pipeline(device, "dinf_dual_post_norm_residual_f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-dual-post-norm-residual");
        encoder.set_compute_pipeline_state(&pipeline);
        let e: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        e.set_input_buffer(0, Some(shared.buffer()), shared_layout.start_offset() * f32);
        e.set_input_buffer(1, Some(routed.buffer()), routed_layout.start_offset() * f32);
        for (index, (storage, offset)) in operands.iter().take(4).enumerate() {
            e.set_input_buffer(2 + index, Some(storage.buffer()), *offset);
        }
        // Never read without bit 1 of `flags` (it aliases post_weight then).
        e.set_input_buffer(11, Some(operands[4].0.buffer()), operands[4].1);
        e.set_bytes(12, &(elements as u32));
        match &scale {
            Some((storage, offset)) => e.set_input_buffer(6, Some(storage.buffer()), *offset),
            // Never read without bit 0 of `flags`.
            None => e.set_input_buffer(6, Some(operands[3].0.buffer()), operands[3].1),
        }
        e.set_output_buffer(7, Some(&output), 0);
        e.set_bytes(8, &(hidden_dim as u32));
        e.set_bytes(9, &self.eps);
        e.set_bytes(
            10,
            &(u32::from(scale.is_some()) | (u32::from(self.next_weight.is_some()) << 1)),
        );
        e.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: elements / hidden_dim,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: norm_thread_width(hidden_dim, &pipeline),
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), output_elements, DType::F32),
            Shape::from(output_elements),
        ))
    }
}

/// Gemma 4 MoE FFN tail in one dispatch:
/// `scale * (residual + rmsnorm(rmsnorm(shared) * shared_weight
///   + rmsnorm(routed) * routed_weight) * post_weight)`, plus
/// `rmsnorm(that) * next_weight` when given (the next layer's input).
#[allow(clippy::too_many_arguments)]
pub(super) fn try_fused_dual_post_norm_residual(
    shared: &Tensor,
    routed: &Tensor,
    residual: &Tensor,
    shared_weight: &Tensor,
    routed_weight: &Tensor,
    post_weight: &Tensor,
    eps: f32,
    scale: Option<&Tensor>,
    next_weight: Option<&Tensor>,
) -> Result<Option<(Tensor, Option<Tensor>)>> {
    let hidden_dim = shared.dim(candle_core::D::Minus1)?;
    let ok = |tensor: &Tensor| {
        tensor.dtype() == DType::F32
            && tensor.is_contiguous()
            && tensor.device().same_device(shared.device())
    };
    if std::env::var_os("DINF_DISABLE_FUSED_POST_NORM_RESIDUAL").is_some()
        || !shared.device().is_metal()
        || ![
            shared,
            routed,
            residual,
            shared_weight,
            routed_weight,
            post_weight,
        ]
        .iter()
        .all(|tensor| ok(tensor))
        || shared.shape() != routed.shape()
        || shared.shape() != residual.shape()
        || [shared_weight, routed_weight, post_weight]
            .iter()
            .any(|weight| weight.dims() != [hidden_dim])
        || scale.is_some_and(|scale| !ok(scale) || scale.elem_count() != 1)
        || next_weight.is_some_and(|weight| !ok(weight) || weight.dims() != [hidden_dim])
    {
        return Ok(None);
    }
    let shape = shared.shape().clone();
    let elements = shape.elem_count();
    let packed = shared.apply_op2_no_bwd(
        routed,
        &DualPostNormResidual {
            eps,
            shared_weight: shared_weight.clone(),
            routed_weight: routed_weight.clone(),
            post_weight: post_weight.clone(),
            residual: residual.clone(),
            scale: scale.cloned(),
            next_weight: next_weight.cloned(),
        },
    )?;
    // Normalized activations feed projections: keep them at offset zero.
    Ok(Some(match next_weight {
        Some(_) => (
            packed
                .narrow(0, elements, elements)?
                .reshape(shape.clone())?,
            Some(packed.narrow(0, 0, elements)?.reshape(shape)?),
        ),
        None => (packed.reshape(shape)?, None),
    }))
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::{
        try_fused_add_rms_norm, try_fused_head_rms_norm, try_fused_head_rms_norm_no_weight,
        try_fused_head_rms_norm_rope, try_fused_head_rms_norm_silu_gate,
        try_fused_post_norm_residual, try_fused_rms_norm, try_fused_rms_norm_epilogue,
        try_fused_rms_norm_no_weight,
    };

    fn max_abs_diff(actual: &Tensor, expected: &Tensor) -> candle_core::Result<f32> {
        let actual = actual
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        let expected = expected
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        Ok(actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max))
    }

    fn metal_device() -> Option<Device> {
        crate::test_metal_device()
    }

    #[test]
    fn fused_rms_norm_matches_candle_reference() -> candle_core::Result<()> {
        let Some(device) = metal_device() else {
            return Ok(());
        };
        for (rows, hidden_dim) in [(21, 256), (3, 5376)] {
            let input = Tensor::from_vec(
                (0..rows * hidden_dim)
                    .map(|index| ((index * 17 % 251) as f32 - 125.0) / 29.0)
                    .collect::<Vec<_>>(),
                (rows, hidden_dim),
                &device,
            )?;
            let weight = Tensor::from_vec(
                (0..hidden_dim)
                    .map(|index| 0.5 + (index % 37) as f32 / 41.0)
                    .collect::<Vec<_>>(),
                hidden_dim,
                &device,
            )?;
            let expected = candle_nn::ops::rms_norm(&input, &weight, 1e-5)?;
            let actual = try_fused_rms_norm(&input, &weight, 1e-5)?.expect("Metal fused path");
            let max_abs_diff = max_abs_diff(&actual, &expected)?;
            assert!(
                max_abs_diff < 1e-5,
                "RMSNorm differs for hidden_dim={hidden_dim}: max_abs_diff={max_abs_diff}"
            );
        }
        Ok(())
    }

    #[test]
    fn fused_unweighted_rms_norm_matches_candle_reference() -> candle_core::Result<()> {
        let Some(device) = metal_device() else {
            return Ok(());
        };
        let (rows, hidden_dim) = (21, 256);
        let input = Tensor::from_vec(
            (0..rows * hidden_dim)
                .map(|index| ((index * 19 % 241) as f32 - 120.0) / 31.0)
                .collect::<Vec<_>>(),
            (rows, hidden_dim),
            &device,
        )?;
        let variance = (&input * &input)?.mean_keepdim(candle_core::D::Minus1)?;
        let expected = input.broadcast_div(&(&variance + 1e-5f64)?.sqrt()?)?;
        let actual = try_fused_rms_norm_no_weight(&input, 1e-5)?.expect("Metal fused path");
        let max_abs_diff = max_abs_diff(&actual, &expected)?;
        assert!(
            max_abs_diff < 1e-5,
            "unweighted RMSNorm differs: max_abs_diff={max_abs_diff}"
        );
        Ok(())
    }

    #[test]
    fn fused_head_layout_norms_match_candle_reference() -> candle_core::Result<()> {
        let Some(device) = metal_device() else {
            return Ok(());
        };
        let (batch, seq_len, heads, head_dim) = (2, 3, 4, 8);
        let input = Tensor::from_vec(
            (0..batch * seq_len * heads * head_dim)
                .map(|index| ((index * 23 % 257) as f32 - 128.0) / 37.0)
                .collect::<Vec<_>>(),
            (batch, seq_len, heads * head_dim),
            &device,
        )?;
        let weight = Tensor::from_vec(
            (0..head_dim)
                .map(|index| 0.75 + index as f32 * 0.05)
                .collect::<Vec<_>>(),
            head_dim,
            &device,
        )?;
        let laid_out = input
            .reshape((batch, seq_len, heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let expected = candle_nn::ops::rms_norm(&laid_out, &weight, 1e-5)?;
        let actual =
            try_fused_head_rms_norm(&input, &weight, batch, seq_len, heads, head_dim, 1e-5)?
                .expect("Metal fused path");
        let weighted_max_abs_diff = max_abs_diff(&actual, &expected)?;
        assert!(
            weighted_max_abs_diff < 1e-5,
            "head RMSNorm differs: max_abs_diff={weighted_max_abs_diff}"
        );

        let variance = (&laid_out * &laid_out)?.mean_keepdim(candle_core::D::Minus1)?;
        let expected = laid_out.broadcast_div(&(&variance + 1e-5f64)?.sqrt()?)?;
        let actual =
            try_fused_head_rms_norm_no_weight(&input, batch, seq_len, heads, head_dim, 1e-5)?
                .expect("Metal fused path");
        let max_abs_diff = max_abs_diff(&actual, &expected)?;
        assert!(
            max_abs_diff < 1e-5,
            "unweighted head RMSNorm differs: max_abs_diff={max_abs_diff}"
        );
        Ok(())
    }

    #[test]
    fn fused_head_rms_norm_rope_matches_norm_then_candle_rope() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (batch, seq_len, heads, head_dim, start) = (2, 37, 4, 128, 11);
        let input = Tensor::randn(0f32, 1.0, (batch, seq_len, heads * head_dim), &device)?;
        let weight = Tensor::randn(1f32, 0.1, head_dim, &device)?;
        let angles = Tensor::randn(0f32, 3.0, (64, head_dim / 2), &device)?;
        let (cos, sin) = (angles.cos()?, angles.sin()?);
        let actual = try_fused_head_rms_norm_rope(
            &input, &weight, &cos, &sin, start, batch, seq_len, heads, head_dim, 1e-6,
        )?
        .expect("supported geometry");
        let normed =
            try_fused_head_rms_norm(&input, &weight, batch, seq_len, heads, head_dim, 1e-6)?
                .expect("supported geometry");
        let expected = candle_nn::rotary_emb::rope(
            &normed,
            &cos.narrow(0, start, seq_len)?,
            &sin.narrow(0, start, seq_len)?,
        )?;
        let diff = (actual - expected)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(diff <= 1e-5, "max abs {diff}");
        Ok(())
    }

    #[test]
    fn fused_head_rms_norm_silu_gate_matches_candle_reference() -> candle_core::Result<()> {
        if candle_metal_kernels::metal::Device::all().is_empty() {
            return Ok(());
        }
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        // Exact Qwen3.5 DeltaNet head geometry.
        let (heads, head_dim) = (32, 128);
        let weight = Tensor::from_vec(
            (0..head_dim)
                .map(|index| 0.75 + index as f32 / 257.0)
                .collect::<Vec<_>>(),
            head_dim,
            &device,
        )?;
        for seq_len in [1, 26] {
            let elements = seq_len * heads * head_dim;
            let input = Tensor::from_vec(
                (0..elements)
                    .map(|index| ((index * 17 % 251) as f32 - 125.0) / 37.0)
                    .collect::<Vec<_>>(),
                (seq_len, heads, head_dim),
                &device,
            )?;
            let gate = Tensor::from_vec(
                (0..elements)
                    .map(|index| ((index * 29 % 241) as f32 - 120.0) / 43.0)
                    .collect::<Vec<_>>(),
                (1, seq_len, heads * head_dim),
                &device,
            )?;
            let normed = candle_nn::ops::rms_norm(&input, &weight, 1e-5)?.reshape((
                1,
                seq_len,
                heads * head_dim,
            ))?;
            let expected = (gate.silu()? * normed)?;
            let actual = try_fused_head_rms_norm_silu_gate(
                &input, &gate, &weight, seq_len, heads, head_dim, 1e-5,
            )?
            .expect("Metal fused path");
            let max_abs_diff = max_abs_diff(&actual, &expected)?;
            assert!(
                max_abs_diff < 2e-5,
                "head RMSNorm SiLU gate differs for seq_len={seq_len}: max_abs_diff={max_abs_diff}"
            );
        }
        Ok(())
    }

    #[test]
    fn fused_rms_norm_epilogue_matches_candle_elementwise() -> candle_core::Result<()> {
        let Some(device) = metal_device() else {
            return Ok(());
        };
        let hidden_dim = 512;
        let weight = Tensor::from_vec(
            (0..hidden_dim)
                .map(|index| 0.5 + (index % 37) as f32 / 41.0)
                .collect::<Vec<_>>(),
            hidden_dim,
            &device,
        )?;
        for tokens in [1, 6, 256] {
            let input = Tensor::from_vec(
                (0..tokens * hidden_dim)
                    .map(|index| ((index * 17 % 251) as f32 - 125.0) / 29.0)
                    .collect::<Vec<_>>(),
                (1, tokens, hidden_dim),
                &device,
            )?;
            let variance = (&input * &input)?.mean_keepdim(candle_core::D::Minus1)?;
            let expected = input
                .broadcast_div(&(&variance + 1e-5f64)?.sqrt()?)?
                .broadcast_mul(&weight)?;
            let actual = try_fused_rms_norm_epilogue(&input, &variance, &weight, 1e-5)?
                .expect("Metal fused path");
            let actual = actual.flatten_all()?.to_vec1::<f32>()?;
            let expected = expected.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(
                actual, expected,
                "RMSNorm epilogue differs for {tokens} tokens"
            );
        }
        Ok(())
    }

    #[test]
    fn fused_add_rms_norm_matches_candle_reference() -> candle_core::Result<()> {
        let Some(device) = metal_device() else {
            return Ok(());
        };
        let lhs = Tensor::from_vec(
            (0..48)
                .map(|index| ((index * 13 % 29) as f32 - 14.0) / 7.0)
                .collect::<Vec<_>>(),
            (2, 3, 8),
            &device,
        )?;
        let rhs = Tensor::from_vec(
            (0..48)
                .map(|index| ((index * 7 % 31) as f32 - 15.0) / 11.0)
                .collect::<Vec<_>>(),
            (2, 3, 8),
            &device,
        )?;
        let weight = Tensor::from_vec(
            (0..8)
                .map(|index| 0.75 + index as f32 * 0.05)
                .collect::<Vec<_>>(),
            8,
            &device,
        )?;
        let expected_residual = (&lhs + &rhs)?;
        let expected_normalized = candle_nn::ops::rms_norm(&expected_residual, &weight, 1e-5)?;
        let (actual_residual, actual_normalized) =
            try_fused_add_rms_norm(&lhs, &rhs, &weight, 1e-5)?.expect("Metal fused path");

        let expected_residual = expected_residual
            .to_device(&Device::Cpu)?
            .to_vec3::<f32>()?;
        let actual_residual = actual_residual.to_device(&Device::Cpu)?.to_vec3::<f32>()?;
        assert_eq!(actual_residual, expected_residual);

        let max_abs_diff = max_abs_diff(&actual_normalized, &expected_normalized)?;
        assert!(max_abs_diff < 1e-5, "max_abs_diff={max_abs_diff}");
        Ok(())
    }
    #[test]
    fn fused_post_norm_residual_matches_separate_kernels() -> candle_core::Result<()> {
        let Some(device) = metal_device() else {
            return Ok(());
        };
        // Gemma 4 31B hidden width, two rows, so the norm uses a full
        // 1024-thread reduction and several columns per thread.
        let hidden = 5376;
        let rows = 2;
        let tensor = |seed: usize, divisor: f32| {
            Tensor::from_vec(
                (0..rows * hidden)
                    .map(|index| ((index * seed % 97) as f32 - 48.0) / divisor)
                    .collect::<Vec<_>>(),
                (rows, 1, hidden),
                &device,
            )
        };
        let update = tensor(13, 9.0)?;
        let residual = tensor(7, 5.0)?;
        let weight = |offset: f32| {
            Tensor::from_vec(
                (0..hidden)
                    .map(|index| offset + (index % 17) as f32 * 0.03)
                    .collect::<Vec<_>>(),
                hidden,
                &device,
            )
        };
        let post_weight = weight(0.6)?;
        let next_weight = weight(0.9)?;
        let scale = Tensor::new(&[0.375f32], &device)?;
        let separate_post = try_fused_rms_norm(&update, &post_weight, 1e-6)?.expect("Metal path");

        let (sum, normalized) = try_fused_post_norm_residual(
            &update,
            &residual,
            &post_weight,
            1e-6,
            Some((&next_weight, 1e-5)),
            None,
        )?
        .expect("Metal fused path");
        let (expected_sum, expected_normalized) =
            try_fused_add_rms_norm(&residual, &separate_post, &next_weight, 1e-5)?
                .expect("Metal path");
        assert!(max_abs_diff(&sum, &expected_sum)? < 1e-6);
        assert!(max_abs_diff(&normalized.expect("next norm"), &expected_normalized)? < 1e-6);

        let (scaled, none) = try_fused_post_norm_residual(
            &update,
            &residual,
            &post_weight,
            1e-6,
            None,
            Some(&scale),
        )?
        .expect("Metal fused path");
        assert!(none.is_none());
        let expected_scaled = (&residual + &separate_post)?.broadcast_mul(&scale)?;
        assert!(max_abs_diff(&scaled, &expected_scaled)? < 1e-6);
        assert_eq!(scaled.dims(), update.dims());
        Ok(())
    }
}
