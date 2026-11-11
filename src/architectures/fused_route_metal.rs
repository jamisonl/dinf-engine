use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, DType, Layout, MetalStorage, Result, Shape, Tensor,
};
use candle_metal_kernels::metal::ComputePipeline;

const FUSED_ROUTE_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Keep hostile/non-finite logits finite before the top-k comparison.  In
// particular, a NaN makes every comparison false and the old reduction could
// publish 0xffffffff as an expert ID.  Clamping also keeps the exp/subtract
// operations finite for tensors supplied by an untrusted caller.
float dinf_sanitize_route_logit(float value) {
  if (isnan(value)) {
    return -1.0e20f;
  }
  if (value == INFINITY) {
    return 1.0e20f;
  }
  if (value == -INFINITY) {
    return -1.0e20f;
  }
  return clamp(value, -1.0e20f, 1.0e20f);
}

kernel void dinf_sigmoid_topk_route_f32(
    device const float *logits [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &expert_count [[buffer(3)]],
    constant uint &top_k [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    uint token [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (token >= tokens) {
    return;
  }

  // GLM-5.3 has 288 experts: nine candidates per SIMD lane.
  float local_scores[9];
  float local_probabilities[9];
  uint local_ids[9];
  for (uint slot = 0; slot < 9; ++slot) {
    const uint expert = lane + slot * 32;
    local_ids[slot] = expert;
    if (expert < expert_count) {
      const float logit = dinf_sanitize_route_logit(
          logits[token * expert_count + expert]);
      const float probability =
          1.0f / (1.0f + exp(-logit));
      const float route_bias =
          dinf_sanitize_route_logit(bias[expert]);
      local_probabilities[slot] = probability;
      local_scores[slot] = probability + route_bias;
    } else {
      local_probabilities[slot] = 0.0f;
      local_scores[slot] = -INFINITY;
    }
  }

  threadgroup uint selected_ids[16];
  threadgroup float selected_probabilities[16];
  for (uint route = 0; route < top_k; ++route) {
    float local_best_score = -INFINITY;
    uint local_best_id = expert_count;
    for (uint slot = 0; slot < 9; ++slot) {
      const float score = local_scores[slot];
      const uint expert = local_ids[slot];
      if (expert < expert_count &&
          (score > local_best_score ||
           (score == local_best_score && expert < local_best_id))) {
        local_best_score = score;
        local_best_id = expert;
      }
    }

    const float selected_score = simd_max(local_best_score);
    const uint candidate_id =
        local_best_score == selected_score ? local_best_id : expert_count;
    const uint selected_id = simd_min(candidate_id);
    float selected_probability = 0.0f;
    for (uint slot = 0; slot < 9; ++slot) {
      if (local_ids[slot] == selected_id) {
        selected_probability = local_probabilities[slot];
        local_scores[slot] = -INFINITY;
      }
    }
    selected_probability = simd_sum(selected_probability);
    if (lane == 0) {
      selected_ids[route] = selected_id;
      selected_probabilities[route] = selected_probability;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) {
    float weight_sum = 0.0f;
    for (uint route = 0; route < top_k; ++route) {
      weight_sum += selected_probabilities[route];
    }
    const float denominator = max(weight_sum, 6.103515625e-5f);
    const uint route_offset = token * top_k;
    const uint plane_elements = tokens * top_k;
    for (uint route = 0; route < top_k; ++route) {
      out[route_offset + route] = float(selected_ids[route]);
      out[plane_elements + route_offset + route] =
          selected_probabilities[route] / denominator;
    }
  }
}

// Output plane 0 holds expert IDs and plane 1 route weights, both as 32-bit
// words: with `u32_ids` unset the IDs are f32 values, otherwise u32 IDs that
// are unique and in range (as dinf_sanitize_moe_ids would make them), so the
// indexed expert kernels can take them without a sanitize pass. Weights are
// always f32 bits.
kernel void dinf_softmax_topk_route_f32(
    device const float *logits [[buffer(0)]],
    device uint *out [[buffer(1)]],
    constant uint &expert_count [[buffer(2)]],
    constant uint &top_k [[buffer(3)]],
    constant uint &tokens [[buffer(4)]],
    constant uint &normalize_topk [[buffer(5)]],
    constant float &scaling_factor [[buffer(6)]],
    constant uint &u32_ids [[buffer(7)]],
    uint token [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (token >= tokens) {
    return;
  }

  float local_logits[4];
  float local_probabilities[4];
  uint local_ids[4];
  float local_max = -INFINITY;
  for (uint slot = 0; slot < 4; ++slot) {
    const uint expert = lane + slot * 32;
    local_ids[slot] = expert;
    const float value = expert < expert_count
        ? dinf_sanitize_route_logit(logits[token * expert_count + expert])
        : -INFINITY;
    local_logits[slot] = value;
    local_max = max(local_max, value);
  }

  const float row_max = simd_max(local_max);
  float local_sum = 0.0f;
  for (uint slot = 0; slot < 4; ++slot) {
    const float probability =
        local_ids[slot] < expert_count ? exp(local_logits[slot] - row_max) : 0.0f;
    local_probabilities[slot] = probability;
    local_sum += probability;
  }
  const float row_sum = max(simd_sum(local_sum), 1.0e-20f);

  threadgroup uint selected_ids[16];
  threadgroup float selected_probabilities[16];
  for (uint route = 0; route < top_k; ++route) {
    float local_best_score = -INFINITY;
    uint local_best_id = expert_count;
    for (uint slot = 0; slot < 4; ++slot) {
      const float score = local_logits[slot];
      const uint expert = local_ids[slot];
      if (expert < expert_count &&
          (score > local_best_score ||
           (score == local_best_score && expert < local_best_id))) {
        local_best_score = score;
        local_best_id = expert;
      }
    }

    const float selected_score = simd_max(local_best_score);
    const uint candidate_id =
        local_best_score == selected_score ? local_best_id : expert_count;
    const uint selected_id = simd_min(candidate_id);
    float selected_probability = 0.0f;
    for (uint slot = 0; slot < 4; ++slot) {
      if (local_ids[slot] == selected_id) {
        selected_probability = local_probabilities[slot] / row_sum;
        local_logits[slot] = -INFINITY;
      }
    }
    selected_probability = simd_sum(selected_probability);
    if (lane == 0) {
      selected_ids[route] = selected_id;
      selected_probabilities[route] = selected_probability;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) {
    float selected_sum = 0.0f;
    if (normalize_topk != 0) {
      for (uint route = 0; route < top_k; ++route) {
        selected_sum += selected_probabilities[route];
      }
      selected_sum = max(selected_sum, 1.0e-20f);
    }
    const uint route_offset = token * top_k;
    const uint plane_elements = tokens * top_k;
    for (uint route = 0; route < top_k; ++route) {
      uint id = selected_ids[route];
      float weight = normalize_topk != 0
          ? selected_probabilities[route] / selected_sum
          : selected_probabilities[route] * scaling_factor;
      if (u32_ids != 0 && id >= expert_count) {
        // No finite logit was left: take the lowest unused expert with zero
        // weight, as the ID sanitizer would.
        for (uint replacement = 0; replacement < expert_count; ++replacement) {
          bool used = false;
          for (uint prior = 0; prior < top_k; ++prior) {
            used = used || selected_ids[prior] == replacement;
          }
          if (!used) {
            id = replacement;
            break;
          }
        }
        selected_ids[route] = id;
        weight = 0.0f;
      }
      out[route_offset + route] = u32_ids != 0 ? id : as_type<uint>(float(id));
      out[plane_elements + route_offset + route] = as_type<uint>(weight);
    }
  }
}

kernel void dinf_llama4_topk_route_f32(
    device const float *logits [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &expert_count [[buffer(2)]],
    constant uint &top_k [[buffer(3)]],
    constant uint &tokens [[buffer(4)]],
    uint token [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (token >= tokens) {
    return;
  }

  float local_logits[4];
  uint local_ids[4];
  for (uint slot = 0; slot < 4; ++slot) {
    const uint expert = lane + slot * 32;
    local_ids[slot] = expert;
    local_logits[slot] = expert < expert_count
        ? dinf_sanitize_route_logit(logits[token * expert_count + expert])
        : -INFINITY;
  }

  threadgroup uint selected_ids[16];
  threadgroup float selected_weights[16];
  for (uint route = 0; route < top_k; ++route) {
    float local_best_score = -INFINITY;
    uint local_best_id = expert_count;
    for (uint slot = 0; slot < 4; ++slot) {
      const float score = local_logits[slot];
      const uint expert = local_ids[slot];
      if (expert < expert_count &&
          (score > local_best_score ||
           (score == local_best_score && expert < local_best_id))) {
        local_best_score = score;
        local_best_id = expert;
      }
    }

    const float selected_score = simd_max(local_best_score);
    const uint candidate_id =
        local_best_score == selected_score ? local_best_id : 0xffffffffu;
    const uint selected_id = simd_min(candidate_id);
    float selected_weight = 0.0f;
    for (uint slot = 0; slot < 4; ++slot) {
      if (local_ids[slot] == selected_id) {
        selected_weight = 1.0f / (1.0f + exp(-local_logits[slot]));
        local_logits[slot] = -INFINITY;
      }
    }
    selected_weight = simd_sum(selected_weight);
    if (lane == 0) {
      selected_ids[route] = selected_id;
      selected_weights[route] = selected_weight;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) {
    const uint route_offset = token * top_k;
    const uint plane_elements = tokens * top_k;
    for (uint route = 0; route < top_k; ++route) {
      out[route_offset + route] = float(selected_ids[route]);
      out[plane_elements + route_offset + route] = selected_weights[route];
    }
  }
}
"#;

static PIPELINES: OnceLock<Mutex<HashMap<(DeviceId, &'static str), ComputePipeline>>> =
    OnceLock::new();

fn checked_route_geometry(
    tokens: usize,
    top_k: usize,
    expert_count: usize,
    operation: &str,
) -> Result<(usize, usize)> {
    let u32_max = u32::MAX as usize;
    if tokens == 0 || top_k == 0 || expert_count == 0 {
        candle_core::bail!("{operation} requires non-empty route geometry")
    }
    if tokens > u32_max || top_k > u32_max || expert_count > u32_max {
        candle_core::bail!("{operation} dimensions exceed Metal u32 limits")
    }
    let route_elements = tokens.checked_mul(top_k).ok_or_else(|| {
        candle_core::Error::Msg(format!("{operation} route count overflows usize"))
    })?;
    let output_elements = route_elements.checked_mul(2).ok_or_else(|| {
        candle_core::Error::Msg(format!("{operation} output size overflows usize"))
    })?;
    route_elements
        .checked_mul(DType::F32.size_in_bytes())
        .ok_or_else(|| {
            candle_core::Error::Msg(format!("{operation} route buffer size overflows usize"))
        })?;
    output_elements
        .checked_mul(DType::F32.size_in_bytes())
        .ok_or_else(|| {
            candle_core::Error::Msg(format!("{operation} output buffer size overflows usize"))
        })?;
    if route_elements > u32_max || output_elements > u32_max {
        candle_core::bail!("{operation} buffer geometry exceeds Metal u32 limits")
    }
    Ok((route_elements, output_elements))
}

fn checked_buffer_offset(layout: &Layout, element_size: usize, operation: &str) -> Result<usize> {
    layout
        .start_offset()
        .checked_mul(element_size)
        .ok_or_else(|| {
            candle_core::Error::Msg(format!("{operation} buffer offset overflows usize"))
        })
}

fn pipeline(device: &candle_core::MetalDevice, name: &'static str) -> Result<ComputePipeline> {
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (device.id(), name);
    if let Some(pipeline) = pipelines.lock().unwrap().get(&key) {
        return Ok(pipeline.clone());
    }
    let library = device
        .metal_device()
        .new_library_with_source(FUSED_ROUTE_MSL, None)
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

struct SigmoidTopKRoute {
    top_k: usize,
}

impl CustomOp2 for SigmoidTopKRoute {
    fn name(&self) -> &'static str {
        "dinf-metal-simd-sigmoid-topk-route"
    }

    fn cpu_fwd(
        &self,
        _logits: &CpuStorage,
        _logits_layout: &Layout,
        _bias: &CpuStorage,
        _bias_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("SIMD top-k routing is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        logits: &MetalStorage,
        logits_layout: &Layout,
        bias: &MetalStorage,
        bias_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if logits.dtype() != DType::F32
            || bias.dtype() != DType::F32
            || !logits_layout.is_contiguous()
            || !bias_layout.is_contiguous()
        {
            candle_core::bail!("SIMD top-k routing requires contiguous f32 logits and bias")
        }
        let (tokens, expert_count) = logits_layout.shape().dims2()?;
        if tokens == 0
            || expert_count == 0
            || expert_count > 288
            || bias_layout.shape().dims1()? != expert_count
            || self.top_k == 0
            || self.top_k > expert_count
            || self.top_k > 16
        {
            candle_core::bail!("SIMD top-k routing geometry differs")
        }
        if logits.device().id() != bias.device().id() {
            candle_core::bail!("SIMD top-k routing tensors are on different Metal devices")
        }

        let (_, output_elements) =
            checked_route_geometry(tokens, self.top_k, expert_count, "SIMD sigmoid top-k")?;
        let logits_offset = checked_buffer_offset(
            logits_layout,
            DType::F32.size_in_bytes(),
            "SIMD sigmoid logits",
        )?;
        let bias_offset =
            checked_buffer_offset(bias_layout, DType::F32.size_in_bytes(), "SIMD sigmoid bias")?;
        let device = logits.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, DType::F32)
            .with_label("dinf-simd-sigmoid-topk-route")
            .build()?;
        let pipeline = pipeline(device, "dinf_sigmoid_topk_route_f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-simd-sigmoid-topk-route");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(0, Some(logits.buffer()), logits_offset);
        encoder_ref.set_input_buffer(1, Some(bias.buffer()), bias_offset);
        encoder_ref.set_output_buffer(2, Some(&output), 0);
        encoder_ref.set_bytes(3, &(expert_count as u32));
        encoder_ref.set_bytes(4, &(self.top_k as u32));
        encoder_ref.set_bytes(5, &(tokens as u32));
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: tokens,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), output_elements, DType::F32),
            Shape::from((2, tokens, self.top_k)),
        ))
    }
}

struct SoftmaxTopKRoute {
    top_k: usize,
    normalize_topk: bool,
    scaling_factor: f32,
    /// Emit validated u32 IDs and f32 weight bits in a u32 tensor.
    u32_ids: bool,
}

struct Llama4TopKRoute {
    top_k: usize,
}

impl CustomOp1 for Llama4TopKRoute {
    fn name(&self) -> &'static str {
        "dinf-metal-simd-llama4-topk-route"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("SIMD Llama 4 top-k routing is only dispatched on Metal")
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        if storage.dtype() != DType::F32 || !layout.is_contiguous() {
            candle_core::bail!("SIMD Llama 4 top-k routing requires contiguous f32 logits")
        }
        let (tokens, expert_count) = layout.shape().dims2()?;
        if tokens == 0
            || expert_count == 0
            || expert_count > 128
            || self.top_k == 0
            || self.top_k > expert_count
            || self.top_k > 16
        {
            candle_core::bail!("SIMD Llama 4 top-k routing geometry differs")
        }

        let (_, output_elements) =
            checked_route_geometry(tokens, self.top_k, expert_count, "SIMD Llama 4 top-k")?;
        let logits_offset =
            checked_buffer_offset(layout, DType::F32.size_in_bytes(), "SIMD Llama 4 logits")?;
        let device = storage.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, DType::F32)
            .with_label("dinf-simd-llama4-topk-route")
            .build()?;
        let pipeline = pipeline(device, "dinf_llama4_topk_route_f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-simd-llama4-topk-route");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(0, Some(storage.buffer()), logits_offset);
        encoder_ref.set_output_buffer(1, Some(&output), 0);
        encoder_ref.set_bytes(2, &(expert_count as u32));
        encoder_ref.set_bytes(3, &(self.top_k as u32));
        encoder_ref.set_bytes(4, &(tokens as u32));
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: tokens,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), output_elements, DType::F32),
            Shape::from((2, tokens, self.top_k)),
        ))
    }
}

impl CustomOp1 for SoftmaxTopKRoute {
    fn name(&self) -> &'static str {
        "dinf-metal-simd-softmax-topk-route"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("SIMD softmax top-k routing is only dispatched on Metal")
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        if storage.dtype() != DType::F32 || !layout.is_contiguous() {
            candle_core::bail!("SIMD softmax top-k routing requires contiguous f32 logits")
        }
        let (tokens, expert_count) = layout.shape().dims2()?;
        if tokens == 0
            || expert_count == 0
            || expert_count > 128
            || self.top_k == 0
            || self.top_k > expert_count
            || self.top_k > 16
            || !self.scaling_factor.is_finite()
        {
            candle_core::bail!("SIMD softmax top-k routing geometry differs")
        }

        let (_, output_elements) =
            checked_route_geometry(tokens, self.top_k, expert_count, "SIMD softmax top-k")?;
        let logits_offset =
            checked_buffer_offset(layout, DType::F32.size_in_bytes(), "SIMD softmax logits")?;
        let device = storage.device();
        let output_dtype = if self.u32_ids { DType::U32 } else { DType::F32 };
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, output_dtype)
            .with_label("dinf-simd-softmax-topk-route")
            .build()?;
        let pipeline = pipeline(device, "dinf_softmax_topk_route_f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-simd-softmax-topk-route");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(0, Some(storage.buffer()), logits_offset);
        encoder_ref.set_output_buffer(1, Some(&output), 0);
        encoder_ref.set_bytes(2, &(expert_count as u32));
        encoder_ref.set_bytes(3, &(self.top_k as u32));
        encoder_ref.set_bytes(4, &(tokens as u32));
        encoder_ref.set_bytes(5, &u32::from(self.normalize_topk));
        encoder_ref.set_bytes(6, &self.scaling_factor);
        encoder_ref.set_bytes(7, &u32::from(self.u32_ids));
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: tokens,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), output_elements, output_dtype),
            Shape::from((2, tokens, self.top_k)),
        ))
    }
}

/// [`fused_softmax_topk_route`] returning unique, in-range u32 expert IDs
/// that need no sanitize pass, and the route weights as f32 bits in a u32
/// tensor (read by `fused_expert_scaled_route_sum`).
pub(super) fn fused_softmax_topk_route_u32(
    logits: &Tensor,
    top_k: usize,
    normalize_topk: bool,
    scaling_factor: f32,
) -> Result<(Tensor, Tensor)> {
    let packed = logits.apply_op1_no_bwd(&SoftmaxTopKRoute {
        top_k,
        normalize_topk,
        scaling_factor,
        u32_ids: true,
    })?;
    let expert_ids = packed.narrow(0, 0, 1)?.squeeze(0)?;
    let weight_bits = packed.narrow(0, 1, 1)?.squeeze(0)?;
    Ok((expert_ids, weight_bits))
}

pub(super) fn fused_sigmoid_topk_route(
    logits: &Tensor,
    bias: &Tensor,
    top_k: usize,
) -> Result<(Tensor, Tensor)> {
    let packed = logits.apply_op2_no_bwd(bias, &SigmoidTopKRoute { top_k })?;
    let expert_ids = packed.narrow(0, 0, 1)?.squeeze(0)?.to_dtype(DType::U32)?;
    let weights = packed.narrow(0, 1, 1)?.squeeze(0)?;
    Ok((expert_ids, weights))
}

pub(super) fn fused_softmax_topk_route(
    logits: &Tensor,
    top_k: usize,
    normalize_topk: bool,
    scaling_factor: f32,
) -> Result<(Tensor, Tensor)> {
    let packed = logits.apply_op1_no_bwd(&SoftmaxTopKRoute {
        top_k,
        normalize_topk,
        scaling_factor,
        u32_ids: false,
    })?;
    let expert_ids = packed.narrow(0, 0, 1)?.squeeze(0)?.to_dtype(DType::U32)?;
    let weights = packed.narrow(0, 1, 1)?.squeeze(0)?;
    Ok((expert_ids, weights))
}

/// [`fused_softmax_topk_route`] returning the packed f32 expert IDs without
/// the u32 cast; pass them to the indexed MoE `sanitize_ids`, which converts.
pub(super) fn fused_softmax_topk_route_f32_ids(
    logits: &Tensor,
    top_k: usize,
    normalize_topk: bool,
    scaling_factor: f32,
) -> Result<(Tensor, Tensor)> {
    let packed = logits.apply_op1_no_bwd(&SoftmaxTopKRoute {
        top_k,
        normalize_topk,
        scaling_factor,
        u32_ids: false,
    })?;
    let expert_ids = packed.narrow(0, 0, 1)?.squeeze(0)?;
    let weights = packed.narrow(0, 1, 1)?.squeeze(0)?;
    Ok((expert_ids, weights))
}

pub(super) fn fused_llama4_topk_route(logits: &Tensor, top_k: usize) -> Result<(Tensor, Tensor)> {
    let packed = logits.apply_op1_no_bwd(&Llama4TopKRoute { top_k })?;
    let expert_ids = packed.narrow(0, 0, 1)?.squeeze(0)?.to_dtype(DType::U32)?;
    let weights = packed.narrow(0, 1, 1)?.squeeze(0)?;
    Ok((expert_ids, weights))
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::{fused_llama4_topk_route, fused_sigmoid_topk_route, fused_softmax_topk_route};

    #[test]
    fn simd_sigmoid_topk_matches_scalar_reference() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let tokens = 2;
        let experts = 288;
        let top_k = 8;
        let logits = (0..tokens * experts)
            .map(|index| ((index * 17 % 251) as f32 - 125.0) / 29.0)
            .collect::<Vec<_>>();
        let bias = (0..experts)
            .map(|index| ((index * 11 % 37) as f32 - 18.0) / 100.0)
            .collect::<Vec<_>>();
        let logits_tensor = Tensor::from_vec(logits.clone(), (tokens, experts), &device)?;
        let bias_tensor = Tensor::new(bias.as_slice(), &device)?;
        let (ids, weights) = fused_sigmoid_topk_route(&logits_tensor, &bias_tensor, top_k)?;
        let ids = ids.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
        let weights = weights.to_device(&Device::Cpu)?.to_vec2::<f32>()?;

        for token in 0..tokens {
            let probabilities = (0..experts)
                .map(|expert| {
                    let value = logits[token * experts + expert];
                    1.0 / (1.0 + (-value).exp())
                })
                .collect::<Vec<f32>>();
            let mut selected = probabilities
                .iter()
                .enumerate()
                .map(|(expert, probability)| (expert, probability + bias[expert]))
                .collect::<Vec<_>>();
            selected.sort_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            let expected_ids = selected[..top_k]
                .iter()
                .map(|(expert, _)| *expert as u32)
                .collect::<Vec<_>>();
            assert_eq!(ids[token], expected_ids);

            let mut expected_weights = expected_ids
                .iter()
                .map(|expert| probabilities[*expert as usize])
                .collect::<Vec<_>>();
            let denominator = expected_weights
                .iter()
                .copied()
                .sum::<f32>()
                .max(6.103_515_6e-5);
            for weight in &mut expected_weights {
                *weight /= denominator;
            }
            for (actual, expected) in weights[token].iter().zip(expected_weights) {
                assert!((*actual - expected).abs() < 1e-6, "{actual} != {expected}");
            }
        }
        Ok(())
    }

    #[test]
    fn simd_softmax_topk_matches_scalar_reference() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let tokens = 2;
        let experts = 64;
        let top_k = 6;
        let scaling_factor = 1.25;
        let logits = (0..tokens * experts)
            .map(|index| ((index * 17 % 251) as f32 - 125.0) / 29.0)
            .collect::<Vec<_>>();
        let logits_tensor = Tensor::from_vec(logits.clone(), (tokens, experts), &device)?;
        let (ids, weights) =
            fused_softmax_topk_route(&logits_tensor, top_k, false, scaling_factor)?;
        let ids = ids.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
        let weights = weights.to_device(&Device::Cpu)?.to_vec2::<f32>()?;

        for token in 0..tokens {
            let row = &logits[token * experts..(token + 1) * experts];
            let row_max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator = row
                .iter()
                .map(|value| (*value - row_max).exp())
                .sum::<f32>();
            let probabilities = row
                .iter()
                .map(|value| (*value - row_max).exp() / denominator)
                .collect::<Vec<_>>();
            let mut selected = row.iter().copied().enumerate().collect::<Vec<_>>();
            selected.sort_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            let expected_ids = selected[..top_k]
                .iter()
                .map(|(expert, _)| *expert as u32)
                .collect::<Vec<_>>();
            assert_eq!(ids[token], expected_ids);
            for (actual, expert) in weights[token].iter().zip(expected_ids) {
                let expected = probabilities[expert as usize] * scaling_factor;
                assert!((*actual - expected).abs() < 1e-6, "{actual} != {expected}");
            }
        }
        Ok(())
    }

    #[test]
    fn simd_llama4_topk_matches_raw_logit_selection_and_sigmoid_weights() -> candle_core::Result<()>
    {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let tokens = 3;
        let experts = 16;
        let top_k = 3;
        let logits = (0..tokens * experts)
            .map(|index| ((index * 19 % 71) as f32 - 35.0) / 11.0)
            .collect::<Vec<_>>();
        let logits_tensor = Tensor::from_vec(logits.clone(), (tokens, experts), &device)?;
        let (ids, weights) = fused_llama4_topk_route(&logits_tensor, top_k)?;
        let ids = ids.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
        let weights = weights.to_device(&Device::Cpu)?.to_vec2::<f32>()?;

        for token in 0..tokens {
            let row = &logits[token * experts..(token + 1) * experts];
            let mut selected = row.iter().copied().enumerate().collect::<Vec<_>>();
            selected.sort_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            let expected_ids = selected[..top_k]
                .iter()
                .map(|(expert, _)| *expert as u32)
                .collect::<Vec<_>>();
            assert_eq!(ids[token], expected_ids);
            for (actual, expert) in weights[token].iter().zip(expected_ids) {
                let logit = row[expert as usize];
                let expected = 1.0 / (1.0 + (-logit).exp());
                assert!((*actual - expected).abs() < 1e-6, "{actual} != {expected}");
            }
        }
        Ok(())
    }

    #[test]
    fn simd_topk_nonfinite_rows_have_safe_deterministic_routes() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let tokens = 2;
        let experts = 16;
        let top_k = 4;
        let expected_ids = (0..top_k as u32).collect::<Vec<_>>();
        let nonfinite_rows = vec![f32::NAN; tokens * experts];
        let negative_infinity_rows = vec![f32::NEG_INFINITY; tokens * experts];

        for logits in [nonfinite_rows.clone(), negative_infinity_rows.clone()] {
            let logits_tensor = Tensor::from_vec(logits, (tokens, experts), &device)?;
            let bias = Tensor::zeros(experts, candle_core::DType::F32, &device)?;
            let (ids, weights) = fused_sigmoid_topk_route(&logits_tensor, &bias, top_k)?;
            let ids = ids.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
            let weights = weights.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
            for (row_ids, row_weights) in ids.iter().zip(weights) {
                assert_eq!(row_ids, &expected_ids);
                assert!(row_ids.iter().all(|&id| id < experts as u32));
                assert!(row_weights.iter().all(|weight| weight.is_finite()));
                assert!(row_weights.iter().all(|&weight| weight == 0.0));
            }
        }

        for logits in [nonfinite_rows.clone(), negative_infinity_rows.clone()] {
            let logits_tensor = Tensor::from_vec(logits, (tokens, experts), &device)?;
            let (ids, weights) = fused_softmax_topk_route(&logits_tensor, top_k, false, 1.25)?;
            let ids = ids.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
            let weights = weights.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
            let expected_weight = 1.25 / experts as f32;
            for (row_ids, row_weights) in ids.iter().zip(weights) {
                assert_eq!(row_ids, &expected_ids);
                assert!(row_ids.iter().all(|&id| id < experts as u32));
                for weight in row_weights {
                    assert!(weight.is_finite());
                    assert!((weight - expected_weight).abs() < 1e-6);
                }
            }
        }

        for logits in [nonfinite_rows, negative_infinity_rows] {
            let logits_tensor = Tensor::from_vec(logits, (tokens, experts), &device)?;
            let (ids, weights) = fused_llama4_topk_route(&logits_tensor, top_k)?;
            let ids = ids.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
            let weights = weights.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
            for (row_ids, row_weights) in ids.iter().zip(weights) {
                assert_eq!(row_ids, &expected_ids);
                assert!(row_ids.iter().all(|&id| id < experts as u32));
                assert!(row_weights.iter().all(|weight| weight.is_finite()));
                assert!(row_weights.iter().all(|&weight| weight == 0.0));
            }
        }
        Ok(())
    }
}
