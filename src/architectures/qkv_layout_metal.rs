use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, CustomOp3, DType, Layout, MetalStorage, Result, Shape,
    Storage, Tensor,
};
use candle_metal_kernels::metal::ComputePipeline;

const QKV_LAYOUT_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void dinf_qkv_head_layout_f32(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &batch [[buffer(2)]],
    constant uint &seq_len [[buffer(3)]],
    constant uint &num_heads [[buffer(4)]],
    constant uint &num_kv_heads [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant uint &elements [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= elements) {
    return;
  }

  const uint q_size = num_heads * head_dim;
  const uint kv_size = num_kv_heads * head_dim;
  const uint fused_dim = q_size + kv_size * 2;
  const uint q_elements = batch * num_heads * seq_len * head_dim;
  const uint kv_elements = batch * num_kv_heads * seq_len * head_dim;

  uint local = gid;
  uint heads = num_heads;
  uint source_base = 0;
  if (gid >= q_elements) {
    local -= q_elements;
    heads = num_kv_heads;
    source_base = q_size;
    if (local >= kv_elements) {
      local -= kv_elements;
      source_base += kv_size;
    }
  }

  const uint column = local % head_dim;
  uint index = local / head_dim;
  const uint token = index % seq_len;
  index /= seq_len;
  const uint head = index % heads;
  const uint lane = index / heads;
  const uint source = (lane * seq_len + token) * fused_dim
      + source_base + head * head_dim + column;
  output[gid] = input[source];
}

kernel void dinf_gated_qkv_head_layout_f32(
    device const float *q_input [[buffer(0)]],
    device const float *k_input [[buffer(1)]],
    device const float *v_input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &batch [[buffer(4)]],
    constant uint &seq_len [[buffer(5)]],
    constant uint &num_heads [[buffer(6)]],
    constant uint &num_kv_heads [[buffer(7)]],
    constant uint &head_dim [[buffer(8)]],
    constant uint &elements [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= elements) {
    return;
  }

  const uint q_elements = batch * num_heads * seq_len * head_dim;
  const uint kv_elements = batch * num_kv_heads * seq_len * head_dim;
  uint local = gid;
  uint heads = num_heads;
  uint gate_offset = 0;
  device const float *source_buffer = q_input;

  if (gid >= q_elements) {
    local -= q_elements;
    heads = num_kv_heads;
    source_buffer = k_input;
    if (local >= kv_elements) {
      local -= kv_elements;
      source_buffer = v_input;
      if (local >= kv_elements) {
        local -= kv_elements;
        heads = num_heads;
        source_buffer = q_input;
        gate_offset = head_dim;
      }
    }
  }

  const uint column = local % head_dim;
  uint index = local / head_dim;
  const uint token = index % seq_len;
  index /= seq_len;
  const uint head = index % heads;
  const uint lane = index / heads;
  const uint source_head_dim = source_buffer == q_input ? head_dim * 2 : head_dim;
  const uint source = ((lane * seq_len + token) * heads + head) * source_head_dim
      + gate_offset + column;
  output[gid] = source_buffer[source];
}

// Qwen3.5 decode-only attention preparation. One threadgroup owns one head
// row. Q/K rows normalize over the full head dimension and apply partial
// split-half RoPE; gate rows are copied raw for the output-gating epilogue.
kernel void dinf_gated_qkv_norm_rope_decode_f32(
    device const float *q_input [[buffer(0)]],
    device const float *k_input [[buffer(1)]],
    device const float *q_weight [[buffer(2)]],
    device const float *k_weight [[buffer(3)]],
    device const float *cos [[buffer(4)]],
    device const float *sin [[buffer(5)]],
    device float *output [[buffer(6)]],
    constant uint &batch [[buffer(7)]],
    constant uint &num_heads [[buffer(8)]],
    constant uint &num_kv_heads [[buffer(9)]],
    constant uint &head_dim [[buffer(10)]],
    constant uint &rope_dims [[buffer(11)]],
    constant uint &position [[buffer(12)]],
    constant float &q_eps [[buffer(13)]],
    constant float &k_eps [[buffer(14)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  threadgroup float scratch[1024];
  const uint q_rows = batch * num_heads;
  const uint kv_rows = batch * num_kv_heads;
  const uint q_elements = q_rows * head_dim;
  const uint kv_elements = kv_rows * head_dim;

  // Packed output order is Q, K, then the raw gate. V is already in the
  // desired decode layout and remains a zero-copy reshape on the host. Each
  // query group copies its gate while Q is read, so exact 4B geometry needs
  // only 16 + 4 threadgroups rather than a second 16-group gate pass.
  const bool is_query = row < q_rows;
  const uint local_row = is_query ? row : row - q_rows;
  const uint source_offset = is_query
      ? local_row * head_dim * 2
      : local_row * head_dim;
  const uint output_offset = is_query
      ? local_row * head_dim
      : q_elements + local_row * head_dim;
  device const float *source = is_query ? q_input : k_input;
  device const float *weight = is_query ? q_weight : k_weight;

  float sum_sq = 0.0f;
  for (uint column = tid; column < head_dim; column += threads) {
    const float value = source[source_offset + column];
    sum_sq += value * value;
    if (is_query) {
      const uint gate_offset = q_elements + kv_elements + local_row * head_dim;
      output[gate_offset + column] = q_input[source_offset + head_dim + column];
    }
  }
  scratch[tid] = sum_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float eps = is_query ? q_eps : k_eps;
  const float inv_rms = rsqrt(scratch[0] / float(head_dim) + eps);
  // Every SIMD group must capture the reduced value before scratch is reused
  // for normalized head data.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint column = tid; column < head_dim; column += threads) {
    const float normalized = source[source_offset + column] * inv_rms * weight[column];
    if (column < rope_dims) {
      scratch[column] = normalized;
    } else {
      output[output_offset + column] = normalized;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint half_rope = rope_dims >> 1;
  const uint table_offset = position * half_rope;
  for (uint column = tid; column < rope_dims; column += threads) {
    const bool upper = column >= half_rope;
    const uint frequency = upper ? column - half_rope : column;
    const float first = scratch[frequency];
    const float second = scratch[frequency + half_rope];
    const float c = cos[table_offset + frequency];
    const float s = sin[table_offset + frequency];
    const float value = upper ? first * s + second * c : first * c - second * s;
    output[output_offset + column] = value;
  }
}

kernel void dinf_separate_qkv_head_layout_f32(
    device const float *q_input [[buffer(0)]],
    device const float *k_input [[buffer(1)]],
    device const float *v_input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &batch [[buffer(4)]],
    constant uint &seq_len [[buffer(5)]],
    constant uint &num_heads [[buffer(6)]],
    constant uint &num_kv_heads [[buffer(7)]],
    constant uint &head_dim [[buffer(8)]],
    constant uint &elements [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= elements) {
    return;
  }

  const uint q_elements = batch * num_heads * seq_len * head_dim;
  const uint kv_elements = batch * num_kv_heads * seq_len * head_dim;
  uint local = gid;
  uint heads = num_heads;
  device const float *source_buffer = q_input;
  if (gid >= q_elements) {
    local -= q_elements;
    heads = num_kv_heads;
    source_buffer = k_input;
    if (local >= kv_elements) {
      local -= kv_elements;
      source_buffer = v_input;
    }
  }

  const uint column = local % head_dim;
  uint index = local / head_dim;
  const uint token = index % seq_len;
  index /= seq_len;
  const uint head = index % heads;
  const uint lane = index / heads;
  const uint source = ((lane * seq_len + token) * heads + head) * head_dim + column;
  output[gid] = source_buffer[source];
}

kernel void dinf_attention_output_layout_f16_f32(
    device const half *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &batch [[buffer(2)]],
    constant uint &heads [[buffer(3)]],
    constant uint &seq_len [[buffer(4)]],
    constant uint &head_dim [[buffer(5)]],
    constant uint &elements [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= elements) {
    return;
  }
  const uint column = gid % head_dim;
  uint index = gid / head_dim;
  const uint head = index % heads;
  index /= heads;
  const uint token = index % seq_len;
  const uint lane = index / seq_len;
  const uint source = ((lane * heads + head) * seq_len + token) * head_dim + column;
  output[gid] = float(input[source]);
}

kernel void dinf_sigmoid_gated_attention_output_layout_f16_f32(
    device const half *input [[buffer(0)]],
    device const float *gate [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &batch [[buffer(3)]],
    constant uint &heads [[buffer(4)]],
    constant uint &seq_len [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant uint &elements [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= elements) {
    return;
  }
  const uint column = gid % head_dim;
  uint index = gid / head_dim;
  const uint head = index % heads;
  index /= heads;
  const uint token = index % seq_len;
  const uint lane = index / seq_len;
  const uint source = ((lane * heads + head) * seq_len + token) * head_dim + column;
  const float gate_value = gate[source];
  output[gid] = float(input[source])
      * (1.0f / (1.0f + exp(-gate_value)));
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
        .new_library_with_source(QKV_LAYOUT_MSL, None)
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

fn norm_thread_width(hidden_dim: usize, pipeline: &ComputePipeline) -> usize {
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(1);
    let max_power_of_two = 1usize << max_threads.ilog2();
    hidden_dim
        .div_ceil(2)
        .next_power_of_two()
        .min(max_power_of_two)
        .max(1)
}

struct QkvHeadLayout {
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

struct GatedQkvHeadLayout {
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

struct GatedQkvNormRopeDecode<'a> {
    q_weight: &'a Tensor,
    k_weight: &'a Tensor,
    cos: &'a Tensor,
    sin: &'a Tensor,
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dims: usize,
    position: usize,
    q_eps: f32,
    k_eps: f32,
}

impl CustomOp2 for GatedQkvNormRopeDecode<'_> {
    fn name(&self) -> &'static str {
        "dinf-metal-gated-qkv-norm-rope-decode"
    }

    fn cpu_fwd(
        &self,
        _q_storage: &CpuStorage,
        _q_layout: &Layout,
        _k_storage: &CpuStorage,
        _k_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused gated QKV decode preparation is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        q_storage: &MetalStorage,
        q_layout: &Layout,
        k_storage: &MetalStorage,
        k_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let Some(q_dim) = self
            .num_heads
            .checked_mul(self.head_dim)
            .and_then(|elements| elements.checked_mul(2))
        else {
            candle_core::bail!("fused gated QKV decode preparation geometry overflow")
        };
        let Some(kv_dim) = self.num_kv_heads.checked_mul(self.head_dim) else {
            candle_core::bail!("fused gated QKV decode preparation geometry overflow")
        };
        if q_storage.dtype() != DType::F32
            || k_storage.dtype() != DType::F32
            || !q_layout.is_contiguous()
            || !k_layout.is_contiguous()
            || q_layout.shape().dims() != [self.batch, 1, q_dim]
            || k_layout.shape().dims() != [self.batch, 1, kv_dim]
        {
            candle_core::bail!(
                "fused gated QKV decode preparation requires matching contiguous f32 tensors"
            )
        }
        if q_storage.device().id() != k_storage.device().id() {
            candle_core::bail!(
                "fused gated QKV decode preparation tensors are on different devices"
            )
        }

        let (q_weight_guard, q_weight_layout) = self.q_weight.storage_and_layout();
        let (k_weight_guard, k_weight_layout) = self.k_weight.storage_and_layout();
        let (cos_guard, cos_layout) = self.cos.storage_and_layout();
        let (sin_guard, sin_layout) = self.sin.storage_and_layout();
        let q_weight = match &*q_weight_guard {
            Storage::Metal(storage) => storage,
            _ => candle_core::bail!("fused QKV query norm weight is not on Metal"),
        };
        let k_weight = match &*k_weight_guard {
            Storage::Metal(storage) => storage,
            _ => candle_core::bail!("fused QKV key norm weight is not on Metal"),
        };
        let cos = match &*cos_guard {
            Storage::Metal(storage) => storage,
            _ => candle_core::bail!("fused QKV cosine table is not on Metal"),
        };
        let sin = match &*sin_guard {
            Storage::Metal(storage) => storage,
            _ => candle_core::bail!("fused QKV sine table is not on Metal"),
        };
        if q_weight.dtype() != DType::F32
            || k_weight.dtype() != DType::F32
            || cos.dtype() != DType::F32
            || sin.dtype() != DType::F32
            || !q_weight_layout.is_contiguous()
            || !k_weight_layout.is_contiguous()
            || !cos_layout.is_contiguous()
            || !sin_layout.is_contiguous()
            || q_weight_layout.shape().dims() != [self.head_dim]
            || k_weight_layout.shape().dims() != [self.head_dim]
        {
            candle_core::bail!("fused QKV norm/RoPE metadata is not contiguous f32")
        }
        let (cos_positions, cos_half_dim) = cos_layout.shape().dims2()?;
        let (sin_positions, sin_half_dim) = sin_layout.shape().dims2()?;
        if cos_positions != sin_positions
            || cos_half_dim != sin_half_dim
            || self.rope_dims == 0
            || !self.rope_dims.is_multiple_of(2)
            || self.rope_dims > self.head_dim
            || cos_half_dim * 2 != self.rope_dims
            || self.position >= cos_positions
            || self.head_dim > 1024
        {
            candle_core::bail!("fused QKV norm/RoPE geometry is unsupported")
        }
        let Some(table_end) = self
            .position
            .checked_mul(cos_half_dim)
            .and_then(|offset| offset.checked_add(cos_half_dim))
        else {
            candle_core::bail!("fused QKV RoPE table offset overflow")
        };
        if table_end > u32::MAX as usize {
            candle_core::bail!("fused QKV RoPE table exceeds 32-bit Metal indexing")
        }
        let device = q_storage.device();
        for extra in [q_weight, k_weight, cos, sin] {
            if device.id() != extra.device().id() {
                candle_core::bail!(
                    "fused gated QKV decode preparation tensors are on different devices"
                )
            }
        }

        let Some(q_rows) = self.batch.checked_mul(self.num_heads) else {
            candle_core::bail!("fused gated QKV decode preparation geometry overflow")
        };
        let Some(kv_rows) = self.batch.checked_mul(self.num_kv_heads) else {
            candle_core::bail!("fused gated QKV decode preparation geometry overflow")
        };
        let Some(q_elements) = q_rows.checked_mul(self.head_dim) else {
            candle_core::bail!("fused gated QKV decode preparation geometry overflow")
        };
        let Some(kv_elements) = kv_rows.checked_mul(self.head_dim) else {
            candle_core::bail!("fused gated QKV decode preparation geometry overflow")
        };
        let Some(elements) = q_elements
            .checked_mul(2)
            .and_then(|elements| elements.checked_add(kv_elements))
        else {
            candle_core::bail!("fused gated QKV decode preparation geometry overflow")
        };
        let Some(rows) = q_rows.checked_add(kv_rows) else {
            candle_core::bail!("fused gated QKV decode preparation geometry overflow")
        };
        if elements == 0
            || self.batch > u32::MAX as usize
            || self.num_heads > u32::MAX as usize
            || self.num_kv_heads > u32::MAX as usize
            || self.head_dim > u32::MAX as usize
            || self.rope_dims > u32::MAX as usize
            || self.position > u32::MAX as usize
            || elements > u32::MAX as usize
            || rows > u32::MAX as usize
        {
            candle_core::bail!("fused gated QKV decode preparation geometry is unsupported")
        }

        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-gated-qkv-norm-rope-decode")
            .build()?;
        let pipeline = pipeline(device, "dinf_gated_qkv_norm_rope_decode_f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-gated-qkv-norm-rope-decode");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(q_storage.buffer()),
            q_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            1,
            Some(k_storage.buffer()),
            k_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            2,
            Some(q_weight.buffer()),
            q_weight_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            3,
            Some(k_weight.buffer()),
            k_weight_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            4,
            Some(cos.buffer()),
            cos_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            5,
            Some(sin.buffer()),
            sin_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(6, Some(&output), 0);
        encoder_ref.set_bytes(7, &(self.batch as u32));
        encoder_ref.set_bytes(8, &(self.num_heads as u32));
        encoder_ref.set_bytes(9, &(self.num_kv_heads as u32));
        encoder_ref.set_bytes(10, &(self.head_dim as u32));
        encoder_ref.set_bytes(11, &(self.rope_dims as u32));
        encoder_ref.set_bytes(12, &(self.position as u32));
        encoder_ref.set_bytes(13, &self.q_eps);
        encoder_ref.set_bytes(14, &self.k_eps);
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: rows,
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
            Shape::from(elements),
        ))
    }
}

impl CustomOp3 for GatedQkvHeadLayout {
    fn name(&self) -> &'static str {
        "dinf-metal-gated-qkv-head-layout"
    }

    fn cpu_fwd(
        &self,
        _q_storage: &CpuStorage,
        _q_layout: &Layout,
        _k_storage: &CpuStorage,
        _k_layout: &Layout,
        _v_storage: &CpuStorage,
        _v_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused gated QKV layout is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        q_storage: &MetalStorage,
        q_layout: &Layout,
        k_storage: &MetalStorage,
        k_layout: &Layout,
        v_storage: &MetalStorage,
        v_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let q_dim = self.num_heads * self.head_dim * 2;
        let kv_dim = self.num_kv_heads * self.head_dim;
        if q_storage.dtype() != DType::F32
            || k_storage.dtype() != DType::F32
            || v_storage.dtype() != DType::F32
            || !q_layout.is_contiguous()
            || !k_layout.is_contiguous()
            || !v_layout.is_contiguous()
            || q_layout.shape().dims() != [self.batch, self.seq_len, q_dim]
            || k_layout.shape().dims() != [self.batch, self.seq_len, kv_dim]
            || v_layout.shape().dims() != [self.batch, self.seq_len, kv_dim]
        {
            candle_core::bail!("fused gated QKV layout requires matching contiguous f32 tensors")
        }
        if q_storage.device().id() != k_storage.device().id()
            || q_storage.device().id() != v_storage.device().id()
        {
            candle_core::bail!("fused gated QKV layout tensors are on different Metal devices")
        }

        let q_elements = self.batch * self.num_heads * self.seq_len * self.head_dim;
        let kv_elements = self.batch * self.num_kv_heads * self.seq_len * self.head_dim;
        let elements = q_elements * 2 + kv_elements * 2;
        if elements == 0
            || self.batch > u32::MAX as usize
            || self.seq_len > u32::MAX as usize
            || self.num_heads > u32::MAX as usize
            || self.num_kv_heads > u32::MAX as usize
            || self.head_dim > u32::MAX as usize
            || elements > u32::MAX as usize
        {
            candle_core::bail!("fused gated QKV layout geometry is unsupported")
        }

        let device = q_storage.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-gated-qkv-head-layout")
            .build()?;
        let pipeline = pipeline(device, "dinf_gated_qkv_head_layout_f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-gated-qkv-head-layout");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(q_storage.buffer()),
            q_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            1,
            Some(k_storage.buffer()),
            k_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            2,
            Some(v_storage.buffer()),
            v_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(3, Some(&output), 0);
        encoder_ref.set_bytes(4, &(self.batch as u32));
        encoder_ref.set_bytes(5, &(self.seq_len as u32));
        encoder_ref.set_bytes(6, &(self.num_heads as u32));
        encoder_ref.set_bytes(7, &(self.num_kv_heads as u32));
        encoder_ref.set_bytes(8, &(self.head_dim as u32));
        encoder_ref.set_bytes(9, &(elements as u32));
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
            Shape::from(elements),
        ))
    }
}

struct SeparateQkvHeadLayout {
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl CustomOp3 for SeparateQkvHeadLayout {
    fn name(&self) -> &'static str {
        "dinf-metal-separate-qkv-head-layout"
    }

    fn cpu_fwd(
        &self,
        _q_storage: &CpuStorage,
        _q_layout: &Layout,
        _k_storage: &CpuStorage,
        _k_layout: &Layout,
        _v_storage: &CpuStorage,
        _v_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused separate QKV layout is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        q_storage: &MetalStorage,
        q_layout: &Layout,
        k_storage: &MetalStorage,
        k_layout: &Layout,
        v_storage: &MetalStorage,
        v_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        if q_storage.dtype() != DType::F32
            || k_storage.dtype() != DType::F32
            || v_storage.dtype() != DType::F32
            || !q_layout.is_contiguous()
            || !k_layout.is_contiguous()
            || !v_layout.is_contiguous()
            || q_layout.shape().dims() != [self.batch, self.seq_len, q_dim]
            || k_layout.shape().dims() != [self.batch, self.seq_len, kv_dim]
            || v_layout.shape().dims() != [self.batch, self.seq_len, kv_dim]
        {
            candle_core::bail!("fused separate QKV layout requires contiguous f32 tensors")
        }
        if q_storage.device().id() != k_storage.device().id()
            || q_storage.device().id() != v_storage.device().id()
        {
            candle_core::bail!("fused separate QKV layout tensors are on different Metal devices")
        }

        let q_elements = self.batch * self.num_heads * self.seq_len * self.head_dim;
        let kv_elements = self.batch * self.num_kv_heads * self.seq_len * self.head_dim;
        let elements = q_elements + kv_elements * 2;
        if elements == 0
            || self.batch > u32::MAX as usize
            || self.seq_len > u32::MAX as usize
            || self.num_heads > u32::MAX as usize
            || self.num_kv_heads > u32::MAX as usize
            || self.head_dim > u32::MAX as usize
            || elements > u32::MAX as usize
        {
            candle_core::bail!("fused separate QKV layout geometry is unsupported")
        }

        let device = q_storage.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-separate-qkv-head-layout")
            .build()?;
        let pipeline = pipeline(device, "dinf_separate_qkv_head_layout_f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-separate-qkv-head-layout");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(q_storage.buffer()),
            q_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            1,
            Some(k_storage.buffer()),
            k_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            2,
            Some(v_storage.buffer()),
            v_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(3, Some(&output), 0);
        encoder_ref.set_bytes(4, &(self.batch as u32));
        encoder_ref.set_bytes(5, &(self.seq_len as u32));
        encoder_ref.set_bytes(6, &(self.num_heads as u32));
        encoder_ref.set_bytes(7, &(self.num_kv_heads as u32));
        encoder_ref.set_bytes(8, &(self.head_dim as u32));
        encoder_ref.set_bytes(9, &(elements as u32));
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
            Shape::from(elements),
        ))
    }
}

impl CustomOp1 for QkvHeadLayout {
    fn name(&self) -> &'static str {
        "dinf-metal-qkv-head-layout"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused QKV layout is only dispatched on Metal")
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let fused_dim = (self.num_heads + self.num_kv_heads * 2) * self.head_dim;
        if storage.dtype() != DType::F32
            || !layout.is_contiguous()
            || layout.shape().dims() != [self.batch, self.seq_len, fused_dim]
        {
            candle_core::bail!("fused QKV layout requires a matching contiguous f32 tensor")
        }
        let elements = layout.shape().elem_count();
        if elements == 0
            || self.batch > u32::MAX as usize
            || self.seq_len > u32::MAX as usize
            || self.num_heads > u32::MAX as usize
            || self.num_kv_heads > u32::MAX as usize
            || self.head_dim > u32::MAX as usize
            || elements > u32::MAX as usize
        {
            candle_core::bail!("fused QKV layout geometry is unsupported")
        }

        let device = storage.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-qkv-head-layout")
            .build()?;
        let pipeline = pipeline(device, "dinf_qkv_head_layout_f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-qkv-head-layout");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(storage.buffer()),
            layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(1, Some(&output), 0);
        encoder_ref.set_bytes(2, &(self.batch as u32));
        encoder_ref.set_bytes(3, &(self.seq_len as u32));
        encoder_ref.set_bytes(4, &(self.num_heads as u32));
        encoder_ref.set_bytes(5, &(self.num_kv_heads as u32));
        encoder_ref.set_bytes(6, &(self.head_dim as u32));
        encoder_ref.set_bytes(7, &(elements as u32));
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
            Shape::from(elements),
        ))
    }
}

struct AttentionOutputLayout;

struct SigmoidGatedAttentionOutputLayout;

impl CustomOp2 for SigmoidGatedAttentionOutputLayout {
    fn name(&self) -> &'static str {
        "dinf-metal-sigmoid-gated-attention-output-layout"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _gate: &CpuStorage,
        _gate_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused gated attention output layout is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        input_layout: &Layout,
        gate: &MetalStorage,
        gate_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if input.dtype() != DType::F16
            || gate.dtype() != DType::F32
            || !input_layout.is_contiguous()
            || !gate_layout.is_contiguous()
            || input_layout.shape() != gate_layout.shape()
        {
            candle_core::bail!(
                "fused sigmoid-gated attention output layout requires matching contiguous f16/f32 tensors"
            )
        }
        if input.device().id() != gate.device().id() {
            candle_core::bail!(
                "fused sigmoid-gated attention output tensors are on different devices"
            )
        }
        let (batch, heads, seq_len, head_dim) = input_layout.shape().dims4()?;
        let elements = input_layout.shape().elem_count();
        if elements == 0
            || batch > u32::MAX as usize
            || heads > u32::MAX as usize
            || seq_len > u32::MAX as usize
            || head_dim > u32::MAX as usize
            || elements > u32::MAX as usize
        {
            candle_core::bail!("fused sigmoid-gated attention output geometry is unsupported")
        }

        let device = input.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-sigmoid-gated-attention-output-layout")
            .build()?;
        let pipeline = pipeline(device, "dinf_sigmoid_gated_attention_output_layout_f16_f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-sigmoid-gated-attention-output-layout");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(input.buffer()),
            input_layout.start_offset() * DType::F16.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            1,
            Some(gate.buffer()),
            gate_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(2, Some(&output), 0);
        encoder_ref.set_bytes(3, &(batch as u32));
        encoder_ref.set_bytes(4, &(heads as u32));
        encoder_ref.set_bytes(5, &(seq_len as u32));
        encoder_ref.set_bytes(6, &(head_dim as u32));
        encoder_ref.set_bytes(7, &(elements as u32));
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
            Shape::from((batch, seq_len, heads * head_dim)),
        ))
    }
}

impl CustomOp1 for AttentionOutputLayout {
    fn name(&self) -> &'static str {
        "dinf-metal-attention-output-layout"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused attention output layout is only dispatched on Metal")
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        if storage.dtype() != DType::F16 || !layout.is_contiguous() {
            candle_core::bail!("fused attention output layout requires a contiguous f16 tensor")
        }
        let (batch, heads, seq_len, head_dim) = layout.shape().dims4()?;
        let elements = layout.shape().elem_count();
        if elements == 0
            || batch > u32::MAX as usize
            || heads > u32::MAX as usize
            || seq_len > u32::MAX as usize
            || head_dim > u32::MAX as usize
            || elements > u32::MAX as usize
        {
            candle_core::bail!("fused attention output layout geometry is unsupported")
        }

        let device = storage.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-attention-output-layout")
            .build()?;
        let pipeline = pipeline(device, "dinf_attention_output_layout_f16_f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-attention-output-layout");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(storage.buffer()),
            layout.start_offset() * DType::F16.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(1, Some(&output), 0);
        encoder_ref.set_bytes(2, &(batch as u32));
        encoder_ref.set_bytes(3, &(heads as u32));
        encoder_ref.set_bytes(4, &(seq_len as u32));
        encoder_ref.set_bytes(5, &(head_dim as u32));
        encoder_ref.set_bytes(6, &(elements as u32));
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
            Shape::from((batch, seq_len, heads * head_dim)),
        ))
    }
}

pub(super) fn try_fused_attention_output_layout(input: &Tensor) -> Result<Option<Tensor>> {
    if !input.device().is_metal() || input.dtype() != DType::F16 || !input.is_contiguous() {
        return Ok(None);
    }
    Ok(Some(input.apply_op1_no_bwd(&AttentionOutputLayout)?))
}

/// Apply the Qwen3.5 attention gate and restore `[batch, seq, heads * dim]`
/// layout in one Metal dispatch.
///
/// `gate` is the raw projection. The kernel evaluates its sigmoid in the same
/// pass as the F16-to-F32 conversion and layout copy, avoiding a standalone
/// elementwise sigmoid buffer and dispatch.
pub(super) fn try_fused_sigmoid_gated_attention_output_layout(
    input: &Tensor,
    gate: &Tensor,
) -> Result<Option<Tensor>> {
    if !input.device().is_metal()
        || input.dtype() != DType::F16
        || gate.dtype() != DType::F32
        || !input.is_contiguous()
        || !gate.is_contiguous()
    {
        return Ok(None);
    }
    Ok(Some(input.apply_op2_no_bwd(
        gate,
        &SigmoidGatedAttentionOutputLayout,
    )?))
}

pub(super) fn try_fused_qkv_layout(
    qkv: &Tensor,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    if !qkv.device().is_metal() || qkv.dtype() != DType::F32 || !qkv.is_contiguous() {
        return Ok(None);
    }
    let packed = qkv.apply_op1_no_bwd(&QkvHeadLayout {
        batch,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
    })?;
    let q_elements = batch * num_heads * seq_len * head_dim;
    let kv_elements = batch * num_kv_heads * seq_len * head_dim;
    let q = packed
        .narrow(0, 0, q_elements)?
        .reshape((batch, num_heads, seq_len, head_dim))?;
    let k = packed.narrow(0, q_elements, kv_elements)?.reshape((
        batch,
        num_kv_heads,
        seq_len,
        head_dim,
    ))?;
    let v = packed
        .narrow(0, q_elements + kv_elements, kv_elements)?
        .reshape((batch, num_kv_heads, seq_len, head_dim))?;
    Ok(Some((q, k, v)))
}

#[derive(Clone, Copy)]
pub(super) struct QkvLayoutDimensions {
    pub(super) batch: usize,
    pub(super) seq_len: usize,
    pub(super) num_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
}

#[derive(Clone, Copy)]
pub(super) struct Qwen35DecodeQkvDimensions {
    pub(super) batch: usize,
    pub(super) num_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) rope_dims: usize,
    pub(super) position: usize,
}

/// Prepare one Qwen3.5 decode token for attention in one Metal dispatch.
///
/// The packed projection is split into query and raw gate rows, Q/K are
/// normalized and receive partial split-half RoPE, and all three outputs are
/// laid out as heads. V already has the correct one-token memory order and is
/// returned as a zero-copy reshape.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_fused_qwen35_decode_qkv_norm_rope_layout(
    q_full: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    q_eps: f32,
    k_eps: f32,
    dimensions: Qwen35DecodeQkvDimensions,
) -> Result<Option<(Tensor, Tensor, Tensor, Tensor)>> {
    let Qwen35DecodeQkvDimensions {
        batch,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_dims,
        position,
    } = dimensions;
    let Some(q_dim) = num_heads
        .checked_mul(head_dim)
        .and_then(|elements| elements.checked_mul(2))
    else {
        return Ok(None);
    };
    let Some(kv_dim) = num_kv_heads.checked_mul(head_dim) else {
        return Ok(None);
    };
    let table_geometry = cos.dims2().ok().zip(sin.dims2().ok());
    let table_supported = table_geometry.is_some_and(
        |((cos_positions, cos_half_dim), (sin_positions, sin_half_dim))| {
            cos_positions == sin_positions
                && cos_half_dim == sin_half_dim
                && cos_half_dim.checked_mul(2) == Some(rope_dims)
                && position < cos_positions
                && position
                    .checked_mul(cos_half_dim)
                    .and_then(|offset| offset.checked_add(cos_half_dim))
                    .is_some_and(|end| end <= u32::MAX as usize)
        },
    );
    if std::env::var_os("DINF_DISABLE_QWEN35_FUSED_DECODE_QKV_NORM_ROPE").is_some()
        || std::env::var_os("DINF_DISABLE_QWEN35_FUSED_GATED_QKV_LAYOUT").is_some()
        || std::env::var_os("DINF_DISABLE_FUSED_RMSNORM").is_some()
        || std::env::var_os("DINF_DISABLE_FUSED_HEAD_RMSNORM").is_some()
        || !q_full.device().is_metal()
        || q_full.dtype() != DType::F32
        || k.dtype() != DType::F32
        || v.dtype() != DType::F32
        || q_weight.dtype() != DType::F32
        || k_weight.dtype() != DType::F32
        || cos.dtype() != DType::F32
        || sin.dtype() != DType::F32
        || !q_full.is_contiguous()
        || !k.is_contiguous()
        || !v.is_contiguous()
        || !q_weight.is_contiguous()
        || !k_weight.is_contiguous()
        || !cos.is_contiguous()
        || !sin.is_contiguous()
        || head_dim > 1024
        || rope_dims == 0
        || !rope_dims.is_multiple_of(2)
        || rope_dims > head_dim
        || !table_supported
        || q_full.dims() != [batch, 1, q_dim]
        || k.dims() != [batch, 1, kv_dim]
        || v.dims() != [batch, 1, kv_dim]
        || q_weight.dims() != [head_dim]
        || k_weight.dims() != [head_dim]
    {
        return Ok(None);
    }
    let packed = q_full.apply_op2_no_bwd(
        k,
        &GatedQkvNormRopeDecode {
            q_weight,
            k_weight,
            cos,
            sin,
            batch,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dims,
            position,
            q_eps,
            k_eps,
        },
    )?;
    let q_elements = batch * num_heads * head_dim;
    let kv_elements = batch * num_kv_heads * head_dim;
    let query = packed
        .narrow(0, 0, q_elements)?
        .reshape((batch, num_heads, 1, head_dim))?;
    let key =
        packed
            .narrow(0, q_elements, kv_elements)?
            .reshape((batch, num_kv_heads, 1, head_dim))?;
    let gate = packed
        .narrow(0, q_elements + kv_elements, q_elements)?
        .reshape((batch, num_heads, 1, head_dim))?;
    let value = v.reshape((batch, num_kv_heads, 1, head_dim))?;
    Ok(Some((query, key, value, gate)))
}

pub(super) fn try_fused_gated_qkv_layout(
    q_full: &Tensor,
    k: &Tensor,
    v: &Tensor,
    dimensions: QkvLayoutDimensions,
) -> Result<Option<(Tensor, Tensor, Tensor, Tensor)>> {
    let QkvLayoutDimensions {
        batch,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
    } = dimensions;
    if !q_full.device().is_metal()
        || q_full.dtype() != DType::F32
        || k.dtype() != DType::F32
        || v.dtype() != DType::F32
        || !q_full.is_contiguous()
        || !k.is_contiguous()
        || !v.is_contiguous()
    {
        return Ok(None);
    }
    let packed = q_full.apply_op3_no_bwd(
        k,
        v,
        &GatedQkvHeadLayout {
            batch,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
        },
    )?;
    let q_elements = batch * num_heads * seq_len * head_dim;
    let kv_elements = batch * num_kv_heads * seq_len * head_dim;
    let q = packed
        .narrow(0, 0, q_elements)?
        .reshape((batch, num_heads, seq_len, head_dim))?;
    let k = packed.narrow(0, q_elements, kv_elements)?.reshape((
        batch,
        num_kv_heads,
        seq_len,
        head_dim,
    ))?;
    let v = packed
        .narrow(0, q_elements + kv_elements, kv_elements)?
        .reshape((batch, num_kv_heads, seq_len, head_dim))?;
    let gate = packed
        .narrow(0, q_elements + kv_elements * 2, q_elements)?
        .reshape((batch, num_heads, seq_len, head_dim))?;
    Ok(Some((q, k, v, gate)))
}

pub(super) fn try_fused_separate_qkv_layout(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    dimensions: QkvLayoutDimensions,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    let QkvLayoutDimensions {
        batch,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
    } = dimensions;
    if !q.device().is_metal()
        || q.dtype() != DType::F32
        || k.dtype() != DType::F32
        || v.dtype() != DType::F32
        || !q.is_contiguous()
        || !k.is_contiguous()
        || !v.is_contiguous()
    {
        return Ok(None);
    }
    let packed = q.apply_op3_no_bwd(
        k,
        v,
        &SeparateQkvHeadLayout {
            batch,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
        },
    )?;
    let q_elements = batch * num_heads * seq_len * head_dim;
    let kv_elements = batch * num_kv_heads * seq_len * head_dim;
    let q = packed
        .narrow(0, 0, q_elements)?
        .reshape((batch, num_heads, seq_len, head_dim))?;
    let k = packed.narrow(0, q_elements, kv_elements)?.reshape((
        batch,
        num_kv_heads,
        seq_len,
        head_dim,
    ))?;
    let v = packed
        .narrow(0, q_elements + kv_elements, kv_elements)?
        .reshape((batch, num_kv_heads, seq_len, head_dim))?;
    Ok(Some((q, k, v)))
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Module, Tensor};

    use super::{
        QkvLayoutDimensions, Qwen35DecodeQkvDimensions, try_fused_attention_output_layout,
        try_fused_gated_qkv_layout, try_fused_qkv_layout,
        try_fused_qwen35_decode_qkv_norm_rope_layout, try_fused_separate_qkv_layout,
        try_fused_sigmoid_gated_attention_output_layout,
    };

    #[allow(clippy::too_many_arguments)]
    fn current_qwen35_decode_prepare(
        q_full: &Tensor,
        k: &Tensor,
        v: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        q_eps: f32,
        k_eps: f32,
        dimensions: Qwen35DecodeQkvDimensions,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor, Tensor)> {
        let Qwen35DecodeQkvDimensions {
            batch,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dims,
            position,
        } = dimensions;
        let q_reshaped = q_full.reshape((batch, 1, num_heads, head_dim * 2))?;
        let query = q_reshaped
            .narrow(3, 0, head_dim)?
            .contiguous()?
            .reshape((batch, num_heads, 1, head_dim))?;
        let gate = q_reshaped
            .narrow(3, head_dim, head_dim)?
            .contiguous()?
            .reshape((batch, num_heads, 1, head_dim))?;
        let key = k.reshape((batch, num_kv_heads, 1, head_dim))?;
        let value = v.reshape((batch, num_kv_heads, 1, head_dim))?;
        let query = super::super::fused_norm_metal::try_fused_rms_norm(&query, q_weight, q_eps)?
            .expect("current Metal Q norm path");
        let key = super::super::fused_norm_metal::try_fused_rms_norm(&key, k_weight, k_eps)?
            .expect("current Metal K norm path");
        let cos = cos.narrow(0, position, 1)?;
        let sin = sin.narrow(0, position, 1)?;
        let apply_rope = |x: &Tensor| -> candle_core::Result<Tensor> {
            let rotary =
                candle_nn::rotary_emb::rope(&x.narrow(3, 0, rope_dims)?.contiguous()?, &cos, &sin)?;
            Tensor::cat(
                &[
                    &rotary,
                    &x.narrow(3, rope_dims, head_dim - rope_dims)?.contiguous()?,
                ],
                3,
            )
        };
        Ok((apply_rope(&query)?, apply_rope(&key)?, value, gate))
    }

    fn assert_tensor_close(
        actual: &Tensor,
        expected: &Tensor,
        tolerance: f32,
        label: &str,
    ) -> candle_core::Result<()> {
        assert_eq!(actual.dims(), expected.dims(), "{label} shape");
        let actual = actual.flatten_all()?.to_vec1::<f32>()?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        let max_abs_diff = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            actual.iter().all(|value| value.is_finite()),
            "{label} produced non-finite output"
        );
        assert!(
            max_abs_diff <= tolerance,
            "{label} drifted: max_abs_diff={max_abs_diff} tolerance={tolerance}"
        );
        Ok(())
    }

    #[test]
    fn fused_qwen35_decode_prepare_matches_current_metal_path() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for dimensions in [
            Qwen35DecodeQkvDimensions {
                batch: 2,
                num_heads: 16,
                num_kv_heads: 4,
                head_dim: 256,
                rope_dims: 64,
                position: 17,
            },
            Qwen35DecodeQkvDimensions {
                batch: 2,
                num_heads: 4,
                num_kv_heads: 2,
                head_dim: 32,
                rope_dims: 32,
                position: 5,
            },
        ] {
            let Qwen35DecodeQkvDimensions {
                batch,
                num_heads,
                num_kv_heads,
                head_dim,
                rope_dims,
                position: _,
            } = dimensions;
            let q_stride = num_heads * head_dim * 2;
            let kv_stride = num_kv_heads * head_dim;
            let q_elements = (batch + 1) * q_stride;
            let kv_elements = (batch + 1) * kv_stride;
            let q_values = (0..q_elements)
                .map(|index| {
                    if index % 521 == 0 {
                        if index.is_multiple_of(2) {
                            512.0
                        } else {
                            -512.0
                        }
                    } else {
                        ((index * 37 % 4093) as f32 - 2046.0) / 128.0
                    }
                })
                .collect::<Vec<_>>();
            let k_values = (0..kv_elements)
                .map(|index| ((index * 53 % 2053) as f32 - 1026.0) / 91.0)
                .collect::<Vec<_>>();
            let v_values = (0..kv_elements)
                .map(|index| ((index * 29 % 997) as f32 - 498.0) / 37.0)
                .collect::<Vec<_>>();
            let q_full = Tensor::from_vec(q_values, (batch + 1, 1, q_stride), &device)?
                .narrow(0, 1, batch)?;
            let k = Tensor::from_vec(k_values, (batch + 1, 1, kv_stride), &device)?
                .narrow(0, 1, batch)?;
            let v = Tensor::from_vec(v_values, (batch + 1, 1, kv_stride), &device)?
                .narrow(0, 1, batch)?;

            // Narrowed views exercise non-zero Metal buffer offsets for all
            // auxiliary tensors bound by the custom op.
            let q_weight = Tensor::from_vec(
                (0..head_dim + 4)
                    .map(|index| ((index * 7 % 41) as f32 - 20.0) / 17.0)
                    .collect::<Vec<_>>(),
                head_dim + 4,
                &device,
            )?
            .narrow(0, 2, head_dim)?;
            let k_weight = Tensor::from_vec(
                (0..head_dim + 6)
                    .map(|index| ((index * 11 % 43) as f32 - 21.0) / 19.0)
                    .collect::<Vec<_>>(),
                head_dim + 6,
                &device,
            )?
            .narrow(0, 3, head_dim)?;
            let table_positions = 32;
            let half_rope = rope_dims / 2;
            let angles = (0..(table_positions + 2) * half_rope)
                .map(|index| index as f32 * 0.013 - 0.7)
                .collect::<Vec<_>>();
            let cos = Tensor::from_vec(
                angles.iter().map(|value| value.cos()).collect::<Vec<_>>(),
                (table_positions + 2, half_rope),
                &device,
            )?
            .narrow(0, 1, table_positions)?;
            let sin = Tensor::from_vec(
                angles.iter().map(|value| value.sin()).collect::<Vec<_>>(),
                (table_positions + 2, half_rope),
                &device,
            )?
            .narrow(0, 1, table_positions)?;
            let q_eps = 1e-6;
            let k_eps = 3e-5;
            let expected = current_qwen35_decode_prepare(
                &q_full, &k, &v, &q_weight, &k_weight, &cos, &sin, q_eps, k_eps, dimensions,
            )?;
            let actual = try_fused_qwen35_decode_qkv_norm_rope_layout(
                &q_full, &k, &v, &q_weight, &k_weight, &cos, &sin, q_eps, k_eps, dimensions,
            )?
            .expect("fused Qwen3.5 decode path");
            assert_tensor_close(&actual.0, &expected.0, 3e-5, "query")?;
            assert_tensor_close(&actual.1, &expected.1, 3e-5, "key")?;
            assert_tensor_close(&actual.2, &expected.2, 0.0, "value")?;
            assert_tensor_close(&actual.3, &expected.3, 0.0, "raw gate")?;
        }
        Ok(())
    }

    #[test]
    fn fused_attention_output_layout_matches_cast_and_transpose() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for (batch, heads, seq_len, head_dim) in [(1, 4, 1, 8), (2, 3, 7, 16)] {
            let values = (0..batch * heads * seq_len * head_dim)
                .map(|index| index as f32 * 0.125 - 13.0)
                .collect::<Vec<_>>();
            let input = Tensor::from_vec(values, (batch, heads, seq_len, head_dim), &device)?
                .to_dtype(DType::F16)?;
            let expected = input
                .to_dtype(DType::F32)?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((batch, seq_len, heads * head_dim))?;
            let actual = try_fused_attention_output_layout(&input)?.expect("Metal fused path");
            assert_eq!(
                actual.flatten_all()?.to_vec1::<f32>()?,
                expected.flatten_all()?.to_vec1::<f32>()?
            );
        }
        Ok(())
    }

    #[test]
    fn fused_gated_attention_output_layout_matches_candle_ops() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for (batch, heads, seq_len, head_dim) in [
            (1, 4, 1, 8),
            (2, 3, 7, 16),
            (1, 16, 1, 256),
            (1, 16, 26, 256),
        ] {
            let elements = batch * heads * seq_len * head_dim;
            let input = Tensor::arange(-128f32, elements as f32 - 128.0, &device)?
                .affine(0.03125, 0.0)?
                .reshape((batch, heads, seq_len, head_dim))?
                .to_dtype(DType::F16)?;
            let gate = Tensor::from_vec(
                (0..elements)
                    .map(|index| ((index * 17 % 257) as f32 - 128.0) / 16.0)
                    .collect::<Vec<_>>(),
                (batch, heads, seq_len, head_dim),
                &device,
            )?;
            let expected = (input.to_dtype(DType::F32)?
                * candle_nn::Activation::Sigmoid.forward(&gate)?)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, heads * head_dim))?;
            let actual = try_fused_sigmoid_gated_attention_output_layout(&input, &gate)?
                .expect("Metal fused path");
            let actual = actual.flatten_all()?.to_vec1::<f32>()?;
            let expected = expected.flatten_all()?.to_vec1::<f32>()?;
            let max_abs_diff = actual
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs_diff <= 2e-6,
                "fused sigmoid gate drifted: max_abs_diff={max_abs_diff}"
            );
        }
        Ok(())
    }

    #[test]
    fn fused_qkv_layout_matches_candle_transposes() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let batch = 2;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let fused_dim = (num_heads + num_kv_heads * 2) * head_dim;
        for seq_len in [2, 7, 256] {
            let values = (0..batch * seq_len * fused_dim)
                .map(|index| index as f32 * 0.25 - 17.0)
                .collect::<Vec<_>>();
            let qkv = Tensor::from_vec(values, (batch, seq_len, fused_dim), &device)?;
            let q_size = num_heads * head_dim;
            let kv_size = num_kv_heads * head_dim;
            let expected_q = qkv
                .narrow(candle_core::D::Minus1, 0, q_size)?
                .reshape((batch, seq_len, num_heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let expected_k = qkv
                .narrow(candle_core::D::Minus1, q_size, kv_size)?
                .reshape((batch, seq_len, num_kv_heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let expected_v = qkv
                .narrow(candle_core::D::Minus1, q_size + kv_size, kv_size)?
                .reshape((batch, seq_len, num_kv_heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let (actual_q, actual_k, actual_v) =
                try_fused_qkv_layout(&qkv, batch, seq_len, num_heads, num_kv_heads, head_dim)?
                    .expect("Metal fused path");
            for (actual, expected) in [
                (actual_q, expected_q),
                (actual_k, expected_k),
                (actual_v, expected_v),
            ] {
                assert_eq!(
                    actual.flatten_all()?.to_vec1::<f32>()?,
                    expected.flatten_all()?.to_vec1::<f32>()?
                );
            }
        }
        Ok(())
    }

    #[test]
    fn fused_gated_qkv_layout_matches_candle_transposes() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let batch = 2;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        for seq_len in [2, 7, 256] {
            let q_values = (0..batch * seq_len * num_heads * head_dim * 2)
                .map(|index| index as f32 * 0.25 - 17.0)
                .collect::<Vec<_>>();
            let kv_values = (0..batch * seq_len * num_kv_heads * head_dim)
                .map(|index| index as f32 * 0.125 - 9.0)
                .collect::<Vec<_>>();
            let q_full = Tensor::from_vec(
                q_values,
                (batch, seq_len, num_heads * head_dim * 2),
                &device,
            )?;
            let k = Tensor::from_vec(
                kv_values.clone(),
                (batch, seq_len, num_kv_heads * head_dim),
                &device,
            )?;
            let v = Tensor::from_vec(
                kv_values.into_iter().rev().collect::<Vec<_>>(),
                (batch, seq_len, num_kv_heads * head_dim),
                &device,
            )?;
            let q_reshaped = q_full.reshape((batch, seq_len, num_heads, head_dim * 2))?;
            let expected_q = q_reshaped
                .narrow(3, 0, head_dim)?
                .contiguous()?
                .transpose(1, 2)?
                .contiguous()?;
            let expected_gate = q_reshaped
                .narrow(3, head_dim, head_dim)?
                .contiguous()?
                .transpose(1, 2)?
                .contiguous()?;
            let expected_k = k
                .reshape((batch, seq_len, num_kv_heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let expected_v = v
                .reshape((batch, seq_len, num_kv_heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let actual = try_fused_gated_qkv_layout(
                &q_full,
                &k,
                &v,
                QkvLayoutDimensions {
                    batch,
                    seq_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                },
            )?
            .expect("Metal fused path");
            for (actual, expected) in [
                (actual.0, expected_q),
                (actual.1, expected_k),
                (actual.2, expected_v),
                (actual.3, expected_gate),
            ] {
                assert_eq!(
                    actual.flatten_all()?.to_vec1::<f32>()?,
                    expected.flatten_all()?.to_vec1::<f32>()?
                );
            }
        }
        Ok(())
    }

    #[test]
    fn fused_separate_qkv_layout_matches_candle_transposes() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let batch = 2;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        for seq_len in [2, 7, 256] {
            let q = Tensor::arange(
                0f32,
                (batch * seq_len * num_heads * head_dim) as f32,
                &device,
            )?
            .reshape((batch, seq_len, num_heads * head_dim))?;
            let k = Tensor::arange(
                -17f32,
                (batch * seq_len * num_kv_heads * head_dim) as f32 - 17.0,
                &device,
            )?
            .reshape((batch, seq_len, num_kv_heads * head_dim))?;
            let v = (&k * -0.25)?;
            let expected_q = q
                .reshape((batch, seq_len, num_heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let expected_k = k
                .reshape((batch, seq_len, num_kv_heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let expected_v = v
                .reshape((batch, seq_len, num_kv_heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let actual = try_fused_separate_qkv_layout(
                &q,
                &k,
                &v,
                QkvLayoutDimensions {
                    batch,
                    seq_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                },
            )?
            .expect("Metal fused path");
            for (actual, expected) in [
                (actual.0, expected_q),
                (actual.1, expected_k),
                (actual.2, expected_v),
            ] {
                assert_eq!(
                    actual.flatten_all()?.to_vec1::<f32>()?,
                    expected.flatten_all()?.to_vec1::<f32>()?
                );
            }
        }
        Ok(())
    }
}
