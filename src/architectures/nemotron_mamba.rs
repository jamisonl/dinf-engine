use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{
    CpuStorage, CustomOp2, CustomOp3, DType, Device, Layout, MetalStorage, Result, Shape, Storage,
    Tensor,
};
use candle_metal_kernels::metal::ComputePipeline;

const MAMBA2_SCAN_MSL: &str = concat!(
    include_str!("activation.metal"),
    r#"
#include <metal_stdlib>
using namespace metal;

kernel void nemotron_mamba2_scan_f32(
    device const float *state   [[buffer(0)]],
    device const float *x       [[buffer(1)]],
    device const float *dt_raw  [[buffer(2)]],
    device const float *a       [[buffer(3)]],
    device const float *b       [[buffer(4)]],
    device const float *c       [[buffer(5)]],
    device const float *d       [[buffer(6)]],
    device const float *dt_bias [[buffer(7)]],
    device float *output        [[buffer(8)]],
    constant uint &n_heads      [[buffer(9)]],
    constant uint &d_state      [[buffer(10)]],
    constant uint &head_dim     [[buffer(11)]],
    constant uint &n_group      [[buffer(12)]],
    uint gid [[thread_position_in_grid]]) {
    const uint output_elements = n_heads * head_dim;
    if (gid >= output_elements) {
        return;
    }

    const uint head = gid / head_dim;
    const uint dim = gid - head * head_dim;
    const uint heads_per_group = n_heads / n_group;
    const uint group = head / heads_per_group;
    const uint state_elements = n_heads * d_state * head_dim;

    const float dt_input = dt_raw[head] + dt_bias[head];
    const float dt = dinf_softplus(dt_input);
    const float decay = exp(dt * a[head]);
    const float x_value = x[gid];
    const float x_dt = x_value * dt;

    float sum = 0.0f;
    for (uint s = 0; s < d_state; ++s) {
        const uint state_index = (head * d_state + s) * head_dim + dim;
        const uint group_index = group * d_state + s;
        const float next = state[state_index] * decay + b[group_index] * x_dt;
        output[state_index] = next;
        sum += next * c[group_index];
    }
    output[state_elements + gid] = sum + d[head] * x_value;
}

kernel void nemotron_conv_scan_silu_f32(
    device const float *input  [[buffer(0)]],
    device const float *state  [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device const float *bias   [[buffer(3)]],
    device float *output       [[buffer(4)]],
    constant uint &seq_len     [[buffer(5)]],
    constant uint &conv_dim    [[buffer(6)]],
    constant uint &kernel_size [[buffer(7)]],
    constant uint &has_bias    [[buffer(8)]],
    constant uint &input_stride [[buffer(9)]],
    uint channel [[thread_position_in_grid]]) {
    if (channel >= conv_dim) {
        return;
    }

    float history[8];
    for (uint k = 0; k < kernel_size; ++k) {
        history[k] = state[channel * kernel_size + k];
    }

    for (uint token = 0; token < seq_len; ++token) {
        for (uint k = 1; k < kernel_size; ++k) {
            history[k - 1] = history[k];
        }
        history[kernel_size - 1] = input[token * input_stride + channel];

        float value = has_bias != 0 ? bias[channel] : 0.0f;
        for (uint k = 0; k < kernel_size; ++k) {
            value += history[k] * weight[channel * kernel_size + k];
        }
        output[token * conv_dim + channel] = value / (1.0f + exp(-value));
    }

    const uint state_offset = seq_len * conv_dim + channel * kernel_size;
    for (uint k = 0; k < kernel_size; ++k) {
        output[state_offset + k] = history[k];
    }
}

kernel void nemotron_mamba2_prefill_f32(
    device const float *state   [[buffer(0)]],
    device const float *x       [[buffer(1)]],
    device const float *dt_raw  [[buffer(2)]],
    device const float *b       [[buffer(3)]],
    device const float *c       [[buffer(4)]],
    device const float *a       [[buffer(5)]],
    device const float *d       [[buffer(6)]],
    device const float *dt_bias [[buffer(7)]],
    device float *output        [[buffer(8)]],
    constant uint &seq_len      [[buffer(9)]],
    constant uint &n_heads      [[buffer(10)]],
    constant uint &d_state      [[buffer(11)]],
    constant uint &head_dim     [[buffer(12)]],
    constant uint &n_group      [[buffer(13)]],
    constant uint &dt_stride    [[buffer(14)]],
    constant uint &conv_stride  [[buffer(15)]],
    uint gid [[thread_position_in_grid]]) {
    const uint y_elements = n_heads * head_dim;
    if (gid >= y_elements) {
        return;
    }

    const uint head = gid / head_dim;
    const uint dim = gid - head * head_dim;
    const uint heads_per_group = n_heads / n_group;
    const uint group = head / heads_per_group;
    float history[256];
    for (uint s = 0; s < d_state; ++s) {
        history[s] = state[(head * d_state + s) * head_dim + dim];
    }

    for (uint token = 0; token < seq_len; ++token) {
        const float dt_input = dt_raw[token * dt_stride + head] + dt_bias[head];
        const float dt = dinf_softplus(dt_input);
        const float decay = exp(dt * a[head]);
        const float x_value = x[token * conv_stride + gid];
        const float x_dt = x_value * dt;
        const uint group_offset = token * conv_stride + group * d_state;

        float sum = 0.0f;
        for (uint s = 0; s < d_state; ++s) {
            history[s] = history[s] * decay + b[group_offset + s] * x_dt;
            sum += history[s] * c[group_offset + s];
        }
        output[token * y_elements + gid] = sum + d[head] * x_value;
    }

    const uint state_offset = seq_len * y_elements;
    for (uint s = 0; s < d_state; ++s) {
        output[state_offset + (head * d_state + s) * head_dim + dim] = history[s];
    }
}

kernel void nemotron_mamba2_prefill_gates_f32(
    device const float *dt_raw  [[buffer(0)]],
    device const float *a       [[buffer(1)]],
    device const float *dt_bias [[buffer(2)]],
    device float2 *gates        [[buffer(3)]],
    constant uint &elements     [[buffer(4)]],
    constant uint &n_heads      [[buffer(5)]],
    constant uint &dt_stride    [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= elements) {
        return;
    }
    const uint head = gid % n_heads;
    const uint token = gid / n_heads;
    const float dt_input = dt_raw[token * dt_stride + head] + dt_bias[head];
    const float dt = dinf_softplus(dt_input);
    gates[gid] = float2(dt, exp(dt * a[head]));
}

kernel void nemotron_mamba2_prefill_parallel_f32(
    device const float *state   [[buffer(0)]],
    device const float *x       [[buffer(1)]],
    device const float *dt_raw  [[buffer(2)]],
    device const float *b       [[buffer(3)]],
    device const float *c       [[buffer(4)]],
    device const float *a       [[buffer(5)]],
    device const float *d       [[buffer(6)]],
    device const float *dt_bias [[buffer(7)]],
    device float *output        [[buffer(8)]],
    constant uint &seq_len      [[buffer(9)]],
    constant uint &n_heads      [[buffer(10)]],
    constant uint &d_state      [[buffer(11)]],
    constant uint &head_dim     [[buffer(12)]],
    constant uint &n_group      [[buffer(13)]],
    device const float2 *gates  [[buffer(14)]],
    constant uint &use_gates    [[buffer(15)]],
    constant uint &dt_stride    [[buffer(16)]],
    constant uint &conv_stride  [[buffer(17)]],
    uint3 group_pos [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint gid = group_pos.x;
    const uint y_elements = n_heads * head_dim;
    if (gid >= y_elements) {
        return;
    }

    const uint head = gid / head_dim;
    const uint dim = gid - head * head_dim;
    const uint heads_per_group = n_heads / n_group;
    const uint group = head / heads_per_group;
    const uint state_count = lane < d_state ? 1 + (d_state - 1 - lane) / 32 : 0;
    float history[8];
    for (uint slot = 0; slot < state_count; ++slot) {
        const uint state_slot = lane + slot * 32;
        const uint state_index = (head * d_state + state_slot) * head_dim + dim;
        history[slot] = state[state_index];
    }

    for (uint token = 0; token < seq_len; ++token) {
        float decay = 0.0f;
        float x_value = 0.0f;
        float x_dt = 0.0f;
        if (lane == 0) {
            const uint gate_index = token * n_heads + head;
            float dt;
            if (use_gates != 0) {
                const float2 gate = gates[gate_index];
                dt = gate.x;
                decay = gate.y;
            } else {
                const float dt_input = dt_raw[token * dt_stride + head] + dt_bias[head];
                dt = dinf_softplus(dt_input);
                decay = exp(dt * a[head]);
            }
            x_value = x[token * conv_stride + gid];
            x_dt = x_value * dt;
        }
        decay = simd_broadcast_first(decay);
        x_value = simd_broadcast_first(x_value);
        x_dt = simd_broadcast_first(x_dt);

        float weighted = 0.0f;
        const uint group_offset = token * conv_stride + group * d_state;
        for (uint slot = 0; slot < state_count; ++slot) {
            const uint state_slot = lane + slot * 32;
            const uint group_index = group_offset + state_slot;
            history[slot] = history[slot] * decay + b[group_index] * x_dt;
            weighted += history[slot] * c[group_index];
        }
        const float sum = simd_sum(weighted);
        if (lane == 0) {
            output[token * y_elements + gid] = sum + d[head] * x_value;
        }
    }

    for (uint slot = 0; slot < state_count; ++slot) {
        const uint state_slot = lane + slot * 32;
        const uint state_index = (head * d_state + state_slot) * head_dim + dim;
        const uint state_offset = seq_len * y_elements;
        output[state_offset + state_index] = history[slot];
    }
}

kernel void nemotron_mamba2_prefill_parallel4_f32(
    device const float *state   [[buffer(0)]],
    device const float *x       [[buffer(1)]],
    device const float *dt_raw  [[buffer(2)]],
    device const float *b       [[buffer(3)]],
    device const float *c       [[buffer(4)]],
    device const float *a       [[buffer(5)]],
    device const float *d       [[buffer(6)]],
    device const float *dt_bias [[buffer(7)]],
    device float *output        [[buffer(8)]],
    constant uint &seq_len      [[buffer(9)]],
    constant uint &n_heads      [[buffer(10)]],
    constant uint &d_state      [[buffer(11)]],
    constant uint &head_dim     [[buffer(12)]],
    constant uint &n_group      [[buffer(13)]],
    device const float2 *gates  [[buffer(14)]],
    constant uint &use_gates    [[buffer(15)]],
    constant uint &dt_stride    [[buffer(16)]],
    constant uint &conv_stride  [[buffer(17)]],
    uint3 group_pos [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint dim_groups = head_dim / 4;
    const uint head = group_pos.x / dim_groups;
    if (head >= n_heads) {
        return;
    }
    const uint dim = (group_pos.x - head * dim_groups) * 4;
    const uint vector_offset = head * head_dim + dim;
    const uint y_elements = n_heads * head_dim;
    const uint heads_per_group = n_heads / n_group;
    const uint group = head / heads_per_group;
    const uint state_count = lane < d_state ? 1 + (d_state - 1 - lane) / 32 : 0;
    float4 history[8];
    for (uint slot = 0; slot < state_count; ++slot) {
        const uint state_slot = lane + slot * 32;
        const uint state_index = (head * d_state + state_slot) * head_dim + dim;
        history[slot] = *((device const float4 *)(state + state_index));
    }

    for (uint token = 0; token < seq_len; ++token) {
        float dt = 0.0f;
        float decay = 0.0f;
        float4 x_value = float4(0.0f);
        if (lane == 0) {
            const uint gate_index = token * n_heads + head;
            if (use_gates != 0) {
                const float2 gate = gates[gate_index];
                dt = gate.x;
                decay = gate.y;
            } else {
                const float dt_input = dt_raw[token * dt_stride + head] + dt_bias[head];
                dt = dinf_softplus(dt_input);
                decay = exp(dt * a[head]);
            }
            x_value = *((device const float4 *)(
                x + token * conv_stride + vector_offset));
        }
        dt = simd_broadcast_first(dt);
        decay = simd_broadcast_first(decay);
        x_value.x = simd_broadcast_first(x_value.x);
        x_value.y = simd_broadcast_first(x_value.y);
        x_value.z = simd_broadcast_first(x_value.z);
        x_value.w = simd_broadcast_first(x_value.w);
        const float4 x_dt = x_value * dt;

        float4 weighted = float4(0.0f);
        const uint group_offset = token * conv_stride + group * d_state;
        for (uint slot = 0; slot < state_count; ++slot) {
            const uint state_slot = lane + slot * 32;
            const uint group_index = group_offset + state_slot;
            history[slot] = history[slot] * decay + b[group_index] * x_dt;
            weighted += history[slot] * c[group_index];
        }
        const float4 sum = float4(
            simd_sum(weighted.x),
            simd_sum(weighted.y),
            simd_sum(weighted.z),
            simd_sum(weighted.w));
        if (lane == 0) {
            *((device float4 *)(output + token * y_elements + vector_offset)) =
                sum + d[head] * x_value;
        }
    }

    const uint state_offset = seq_len * y_elements;
    for (uint slot = 0; slot < state_count; ++slot) {
        const uint state_slot = lane + slot * 32;
        const uint state_index = (head * d_state + state_slot) * head_dim + dim;
        *((device float4 *)(output + state_offset + state_index)) = history[slot];
    }
}

kernel void nemotron_mamba2_gate_group_norm_f32(
    device const float *z      [[buffer(0)]],
    device const float *y      [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device float *output       [[buffer(3)]],
    constant uint &seq_len     [[buffer(4)]],
    constant uint &n_group     [[buffer(5)]],
    constant uint &group_size  [[buffer(6)]],
    constant float &epsilon    [[buffer(7)]],
    constant uint &z_stride    [[buffer(8)]],
    threadgroup float *shared  [[threadgroup(0)]],
    uint3 group_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]) {
    const uint token = group_pos.x;
    const uint group = group_pos.y;
    const uint threads = threads_per_group.x * threads_per_group.y * threads_per_group.z;
    if (token >= seq_len || group >= n_group) {
        return;
    }

    const uint base = (token * n_group + group) * group_size;
    const uint z_base = token * z_stride + group * group_size;
    threadgroup float *gated = shared;
    threadgroup float *partial_sums = shared + group_size;
    float local_sum = 0.0f;
    for (uint index = tid; index < group_size; index += threads) {
        const uint offset = base + index;
        const float z_value = z[z_base + index];
        const float value = z_value / (1.0f + exp(-z_value)) * y[offset];
        gated[index] = value;
        local_sum += value * value;
    }

    const float simd_sum_value = simd_sum(local_sum);
    if (lane == 0) {
        partial_sums[simdgroup] = simd_sum_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint index = 0; index < (threads + 31) / 32; ++index) {
            sum += partial_sums[index];
        }
        partial_sums[0] = rsqrt(sum / float(group_size) + epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = partial_sums[0];
    const uint weight_base = group * group_size;
    for (uint index = tid; index < group_size; index += threads) {
        output[base + index] = gated[index] * scale * weight[weight_base + index];
    }
}
"#,
);

type PipelineCache = OnceLock<Mutex<HashMap<DeviceId, ComputePipeline>>>;

static MAMBA2_SCAN_PIPELINE: PipelineCache = OnceLock::new();
static CONV_SCAN_PIPELINE: PipelineCache = OnceLock::new();
static MAMBA2_PREFILL_GATES_PIPELINE: PipelineCache = OnceLock::new();
static MAMBA2_PARALLEL_PREFILL_PIPELINE: PipelineCache = OnceLock::new();
static MAMBA2_GROUPED_PREFILL_PIPELINE: PipelineCache = OnceLock::new();
static MAMBA2_GATE_GROUP_NORM_PIPELINE: PipelineCache = OnceLock::new();

fn mamba2_scan_pipeline(device: &candle_core::MetalDevice) -> Result<ComputePipeline> {
    named_pipeline(device, "nemotron_mamba2_scan_f32", &MAMBA2_SCAN_PIPELINE)
}

fn named_pipeline(
    device: &candle_core::MetalDevice,
    function_name: &str,
    slot: &PipelineCache,
) -> Result<ComputePipeline> {
    let pipelines = slot.get_or_init(|| Mutex::new(HashMap::new()));
    let device_id = device.id();
    if let Some(pipeline) = pipelines.lock().unwrap().get(&device_id).cloned() {
        return Ok(pipeline);
    }
    let library = device
        .metal_device()
        .new_library_with_source(MAMBA2_SCAN_MSL, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function(function_name, None)
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    let mut pipelines = pipelines.lock().unwrap();
    Ok(pipelines.entry(device_id).or_insert(pipeline).clone())
}

pub(super) fn precompile(device: &Device) -> anyhow::Result<()> {
    if device.is_metal() {
        let device = device.as_metal_device()?;
        mamba2_scan_pipeline(device)?;
        named_pipeline(device, "nemotron_conv_scan_silu_f32", &CONV_SCAN_PIPELINE)?;
        named_pipeline(
            device,
            "nemotron_mamba2_prefill_gates_f32",
            &MAMBA2_PREFILL_GATES_PIPELINE,
        )?;
        named_pipeline(
            device,
            "nemotron_mamba2_prefill_parallel_f32",
            &MAMBA2_PARALLEL_PREFILL_PIPELINE,
        )?;
        named_pipeline(
            device,
            "nemotron_mamba2_prefill_parallel4_f32",
            &MAMBA2_GROUPED_PREFILL_PIPELINE,
        )?;
        named_pipeline(
            device,
            "nemotron_mamba2_gate_group_norm_f32",
            &MAMBA2_GATE_GROUP_NORM_PIPELINE,
        )?;
    }
    Ok(())
}

struct GateGroupNorm;

impl CustomOp3 for GateGroupNorm {
    fn name(&self) -> &'static str {
        "nemotron-mamba2-gate-group-norm"
    }

    fn cpu_fwd(
        &self,
        _z: &CpuStorage,
        _z_layout: &Layout,
        _y: &CpuStorage,
        _y_layout: &Layout,
        _weight: &CpuStorage,
        _weight_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("Nemotron Mamba2 fused gate norm is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        z: &MetalStorage,
        z_layout: &Layout,
        y: &MetalStorage,
        y_layout: &Layout,
        weight: &MetalStorage,
        weight_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if z.dtype() != DType::F32
            || y.dtype() != DType::F32
            || weight.dtype() != DType::F32
            || !y_layout.is_contiguous()
            || !weight_layout.is_contiguous()
        {
            candle_core::bail!("Nemotron Mamba2 fused gate norm requires row-contiguous f32 inputs")
        }
        let (seq_len, inner_dim) = z_layout.shape().dims2()?;
        if z_layout.stride()[1] != 1 || y_layout.shape().dims2()? != (seq_len, inner_dim) {
            candle_core::bail!("Nemotron Mamba2 fused gate inputs differ")
        }
        let (n_group, group_size) = weight_layout.shape().dims2()?;
        if n_group == 0 || group_size == 0 || n_group * group_size != inner_dim {
            candle_core::bail!("Nemotron Mamba2 fused gate norm geometry differs")
        }
        if z.device().id() != y.device().id() || z.device().id() != weight.device().id() {
            candle_core::bail!("Nemotron Mamba2 fused gate norm tensors are on different devices")
        }

        let elements = seq_len * inner_dim;
        let device = z.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("nemotron-mamba2-gate-group-norm")
            .build()?;
        let pipeline = named_pipeline(
            device,
            "nemotron_mamba2_gate_group_norm_f32",
            &MAMBA2_GATE_GROUP_NORM_PIPELINE,
        )?;
        let threads = group_size.min(256).next_multiple_of(32);
        let encoder = device.command_encoder()?;
        encoder.set_label("nemotron-mamba2-gate-group-norm");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        let offset = |layout: &Layout| layout.start_offset() * DType::F32.size_in_bytes();
        encoder_ref.set_input_buffer(0, Some(z.buffer()), offset(z_layout));
        encoder_ref.set_input_buffer(1, Some(y.buffer()), offset(y_layout));
        encoder_ref.set_input_buffer(2, Some(weight.buffer()), offset(weight_layout));
        encoder_ref.set_output_buffer(3, Some(&output), 0);
        encoder_ref.set_bytes(4, &(seq_len as u32));
        encoder_ref.set_bytes(5, &(n_group as u32));
        encoder_ref.set_bytes(6, &(group_size as u32));
        encoder_ref.set_bytes(7, &1e-5_f32);
        encoder_ref.set_bytes(8, &(z_layout.stride()[0] as u32));
        encoder_ref.set_threadgroup_memory_length(
            0,
            (group_size + threads.div_ceil(32)) * std::mem::size_of::<f32>(),
        );
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: seq_len,
                height: n_group,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );

        Ok((
            MetalStorage::new(output, device.clone(), elements, DType::F32),
            z_layout.shape().clone(),
        ))
    }
}

pub(super) fn fused_gate_group_norm(z: &Tensor, y: &Tensor, weight: &Tensor) -> Result<Tensor> {
    z.apply_op3_no_bwd(y, weight, &GateGroupNorm)
}

fn metal_storage<'a>(
    name: &str,
    storage: &'a Storage,
    layout: &Layout,
    expected: usize,
) -> Result<&'a MetalStorage> {
    let storage = match storage {
        Storage::Metal(storage) => storage,
        _ => candle_core::bail!("{name} is not on Metal"),
    };
    if storage.dtype() != DType::F32
        || !layout.is_contiguous()
        || layout.shape().elem_count() != expected
    {
        candle_core::bail!(
            "{name} must be contiguous f32 with {expected} elements for Metal Mamba2 prefill"
        )
    }
    Ok(storage)
}

struct ConvScan<'a> {
    weight: &'a Tensor,
    bias: &'a Tensor,
    seq_len: usize,
    conv_dim: usize,
    kernel_size: usize,
}

impl CustomOp2 for ConvScan<'_> {
    fn name(&self) -> &'static str {
        "nemotron-conv-scan"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _state: &CpuStorage,
        _state_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("nemotron-conv-scan is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        input_layout: &Layout,
        state: &MetalStorage,
        state_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let input_elements = self.seq_len * self.conv_dim;
        let state_elements = self.conv_dim * self.kernel_size;
        if input.dtype() != DType::F32
            || input_layout.shape().dims2()? != (self.seq_len, self.conv_dim)
            || input_layout.stride()[1] != 1
        {
            candle_core::bail!("invalid Metal Mamba2 convolution input")
        }
        if state.dtype() != DType::F32
            || !state_layout.is_contiguous()
            || state_layout.shape().elem_count() != state_elements
        {
            candle_core::bail!("invalid Metal Mamba2 convolution state")
        }

        let (weight_guard, weight_layout) = self.weight.storage_and_layout();
        let weight = metal_storage("weight", &weight_guard, weight_layout, state_elements)?;
        let (bias_guard, bias_layout) = self.bias.storage_and_layout();
        let bias = metal_storage("bias", &bias_guard, bias_layout, self.conv_dim)?;

        let output_elements = input_elements + state_elements;
        let device = input.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, DType::F32)
            .with_label("nemotron-conv-scan")
            .build()?;
        let pipeline = named_pipeline(device, "nemotron_conv_scan_silu_f32", &CONV_SCAN_PIPELINE)?;
        let encoder = device.command_encoder()?;
        encoder.set_label("nemotron-conv-scan");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        let offset = |layout: &Layout| layout.start_offset() * DType::F32.size_in_bytes();
        encoder_ref.set_input_buffer(0, Some(input.buffer()), offset(input_layout));
        encoder_ref.set_input_buffer(1, Some(state.buffer()), offset(state_layout));
        encoder_ref.set_input_buffer(2, Some(weight.buffer()), offset(weight_layout));
        encoder_ref.set_input_buffer(3, Some(bias.buffer()), offset(bias_layout));
        encoder_ref.set_output_buffer(4, Some(&output), 0);
        encoder_ref.set_bytes(5, &(self.seq_len as u32));
        encoder_ref.set_bytes(6, &(self.conv_dim as u32));
        encoder_ref.set_bytes(7, &(self.kernel_size as u32));
        encoder_ref.set_bytes(8, &1_u32);
        encoder_ref.set_bytes(9, &(input_layout.stride()[0] as u32));
        encoder_ref.dispatch_threads(
            objc2_metal::MTLSize {
                width: self.conv_dim,
                height: 1,
                depth: 1,
            },
            candle_metal_kernels::utils::get_block_dims(self.conv_dim.min(256), 1, 1),
        );

        Ok((
            MetalStorage::new(output, device.clone(), output_elements, DType::F32),
            Shape::from(output_elements),
        ))
    }
}

struct Mamba2Prefill<'a> {
    a: &'a Tensor,
    d: &'a Tensor,
    dt_bias: &'a Tensor,
    seq_len: usize,
    n_heads: usize,
    d_state: usize,
    head_dim: usize,
    n_group: usize,
}

impl CustomOp3 for Mamba2Prefill<'_> {
    fn name(&self) -> &'static str {
        "nemotron-mamba2-prefill"
    }

    fn cpu_fwd(
        &self,
        _state: &CpuStorage,
        _state_layout: &Layout,
        _x: &CpuStorage,
        _x_layout: &Layout,
        _dt_raw: &CpuStorage,
        _dt_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("nemotron-mamba2-prefill is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        state: &MetalStorage,
        state_layout: &Layout,
        x: &MetalStorage,
        x_layout: &Layout,
        dt_raw: &MetalStorage,
        dt_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let state_elements = self.n_heads * self.d_state * self.head_dim;
        let token_elements = self.n_heads * self.head_dim;
        let y_elements = self.seq_len * token_elements;
        let group_state_dim = self.n_group * self.d_state;
        let conv_dim = token_elements + 2 * group_state_dim;
        if state.dtype() != DType::F32
            || !state_layout.is_contiguous()
            || state_layout.shape().elem_count() != state_elements
        {
            candle_core::bail!("invalid Metal Mamba2 recurrent state")
        }
        if x.dtype() != DType::F32
            || !x_layout.is_contiguous()
            || x_layout.shape().dims2()? != (self.seq_len, conv_dim)
        {
            candle_core::bail!("invalid Metal Mamba2 recurrent input")
        }
        if dt_raw.dtype() != DType::F32
            || dt_layout.shape().dims2()? != (self.seq_len, self.n_heads)
            || dt_layout.stride()[1] != 1
        {
            candle_core::bail!("invalid Metal Mamba2 time-step input")
        }

        let (a_guard, a_layout) = self.a.storage_and_layout();
        let a = metal_storage("a", &a_guard, a_layout, self.n_heads)?;
        let (d_guard, d_layout) = self.d.storage_and_layout();
        let d = metal_storage("d", &d_guard, d_layout, self.n_heads)?;
        let (dt_bias_guard, dt_bias_layout) = self.dt_bias.storage_and_layout();
        let dt_bias = metal_storage("dt_bias", &dt_bias_guard, dt_bias_layout, self.n_heads)?;

        let output_elements = y_elements + state_elements;
        let device = state.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, DType::F32)
            .with_label("nemotron-mamba2-prefill")
            .build()?;
        let grouped_scan = self.head_dim >= 4 && self.head_dim.is_multiple_of(4);
        let pipeline = if grouped_scan {
            named_pipeline(
                device,
                "nemotron_mamba2_prefill_parallel4_f32",
                &MAMBA2_GROUPED_PREFILL_PIPELINE,
            )?
        } else {
            named_pipeline(
                device,
                "nemotron_mamba2_prefill_parallel_f32",
                &MAMBA2_PARALLEL_PREFILL_PIPELINE,
            )?
        };
        let encoder = device.command_encoder()?;
        encoder.set_label("nemotron-mamba2-prefill");
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        let offset = |layout: &Layout| layout.start_offset() * DType::F32.size_in_bytes();
        let gate_elements = self.seq_len * self.n_heads;
        let gates = device
            .new_buffer_builder()
            .with_size_for(gate_elements * 2, DType::F32)
            .with_label("nemotron-mamba2-prefill-gates")
            .build()?;
        let gate_pipeline = named_pipeline(
            device,
            "nemotron_mamba2_prefill_gates_f32",
            &MAMBA2_PREFILL_GATES_PIPELINE,
        )?;
        encoder_ref.set_compute_pipeline_state(&gate_pipeline);
        encoder_ref.set_input_buffer(0, Some(dt_raw.buffer()), offset(dt_layout));
        encoder_ref.set_input_buffer(1, Some(a.buffer()), offset(a_layout));
        encoder_ref.set_input_buffer(2, Some(dt_bias.buffer()), offset(dt_bias_layout));
        encoder_ref.set_output_buffer(3, Some(&gates), 0);
        encoder_ref.set_bytes(4, &(gate_elements as u32));
        encoder_ref.set_bytes(5, &(self.n_heads as u32));
        encoder_ref.set_bytes(6, &(dt_layout.stride()[0] as u32));
        encoder_ref.dispatch_threads(
            objc2_metal::MTLSize {
                width: gate_elements,
                height: 1,
                depth: 1,
            },
            candle_metal_kernels::utils::get_block_dims(gate_elements.min(256), 1, 1),
        );

        encoder_ref.set_compute_pipeline_state(&pipeline);
        encoder_ref.set_input_buffer(0, Some(state.buffer()), offset(state_layout));
        let conv_offset = offset(x_layout);
        encoder_ref.set_input_buffer(1, Some(x.buffer()), conv_offset);
        encoder_ref.set_input_buffer(2, Some(dt_raw.buffer()), offset(dt_layout));
        encoder_ref.set_input_buffer(
            3,
            Some(x.buffer()),
            conv_offset + token_elements * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            4,
            Some(x.buffer()),
            conv_offset + (token_elements + group_state_dim) * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(5, Some(a.buffer()), offset(a_layout));
        encoder_ref.set_input_buffer(6, Some(d.buffer()), offset(d_layout));
        encoder_ref.set_input_buffer(7, Some(dt_bias.buffer()), offset(dt_bias_layout));
        encoder_ref.set_output_buffer(8, Some(&output), 0);
        encoder_ref.set_bytes(9, &(self.seq_len as u32));
        encoder_ref.set_bytes(10, &(self.n_heads as u32));
        encoder_ref.set_bytes(11, &(self.d_state as u32));
        encoder_ref.set_bytes(12, &(self.head_dim as u32));
        encoder_ref.set_bytes(13, &(self.n_group as u32));
        encoder_ref.set_input_buffer(14, Some(&gates), 0);
        // The scan reads the precomputed gates written above.
        encoder_ref.set_bytes(15, &1u32);
        encoder_ref.set_bytes(16, &(dt_layout.stride()[0] as u32));
        encoder_ref.set_bytes(17, &(x_layout.stride()[0] as u32));
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: if grouped_scan {
                    self.n_heads * (self.head_dim / 4)
                } else {
                    token_elements
                },
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
            Shape::from(output_elements),
        ))
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prefill(
    conv_input: &Tensor,
    conv_state: &Tensor,
    ssm_state: &Tensor,
    conv_weight: &Tensor,
    conv_bias: &Tensor,
    dt_raw: &Tensor,
    a: &Tensor,
    d: &Tensor,
    dt_bias: &Tensor,
    n_heads: usize,
    d_state: usize,
    head_dim: usize,
    n_group: usize,
    kernel_size: usize,
) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
    let (seq_len, conv_dim) = conv_input.dims2()?;
    let inner_dim = n_heads * head_dim;
    let group_state_dim = n_group * d_state;
    anyhow::ensure!(
        conv_dim == inner_dim + 2 * group_state_dim,
        "invalid Nemotron Mamba2 convolution width"
    );
    anyhow::ensure!(
        kernel_size <= 8,
        "Nemotron convolution kernel exceeds Metal limit"
    );
    anyhow::ensure!(d_state <= 256, "Nemotron Mamba2 state exceeds Metal limit");
    anyhow::ensure!(
        n_group > 0 && n_heads.is_multiple_of(n_group),
        "Nemotron Mamba2 heads must divide evenly into groups"
    );

    let conv_packed = conv_input.apply_op2_no_bwd(
        conv_state,
        &ConvScan {
            weight: conv_weight,
            bias: conv_bias,
            seq_len,
            conv_dim,
            kernel_size,
        },
    )?;
    let conv_elements = seq_len * conv_dim;
    let conv = conv_packed
        .narrow(0, 0, conv_elements)?
        .reshape((seq_len, conv_dim))?;
    let next_conv_state = conv_packed
        .narrow(0, conv_elements, conv_dim * kernel_size)?
        .reshape((conv_dim, kernel_size))?;

    let recurrent_packed = ssm_state.apply_op3_no_bwd(
        &conv,
        dt_raw,
        &Mamba2Prefill {
            a,
            d,
            dt_bias,
            seq_len,
            n_heads,
            d_state,
            head_dim,
            n_group,
        },
    )?;
    let y_elements = seq_len * inner_dim;
    let y = recurrent_packed
        .narrow(0, 0, y_elements)?
        .reshape((seq_len, n_heads, head_dim))?;
    let state_elements = n_heads * d_state * head_dim;
    let next_ssm_state = recurrent_packed
        .narrow(0, y_elements, state_elements)?
        .reshape((n_heads, d_state, head_dim))?;
    Ok((y, next_conv_state, next_ssm_state))
}

struct Mamba2Scan<'a> {
    a: &'a Tensor,
    b: &'a Tensor,
    c: &'a Tensor,
    d: &'a Tensor,
    dt_bias: &'a Tensor,
    n_heads: usize,
    d_state: usize,
    head_dim: usize,
    n_group: usize,
}

impl CustomOp3 for Mamba2Scan<'_> {
    fn name(&self) -> &'static str {
        "nemotron-mamba2-scan"
    }

    fn cpu_fwd(
        &self,
        _state: &CpuStorage,
        _state_layout: &Layout,
        _x: &CpuStorage,
        _x_layout: &Layout,
        _dt_raw: &CpuStorage,
        _dt_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("nemotron-mamba2-scan is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        state: &MetalStorage,
        state_layout: &Layout,
        x: &MetalStorage,
        x_layout: &Layout,
        dt_raw: &MetalStorage,
        dt_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        for (name, storage, layout) in [
            ("state", state, state_layout),
            ("x", x, x_layout),
            ("dt_raw", dt_raw, dt_layout),
        ] {
            if storage.dtype() != DType::F32 || !layout.is_contiguous() {
                candle_core::bail!("{name} must be contiguous f32 for Metal Mamba2 scan")
            }
        }

        let (a_guard, a_layout) = self.a.storage_and_layout();
        let (b_guard, b_layout) = self.b.storage_and_layout();
        let (c_guard, c_layout) = self.c.storage_and_layout();
        let (d_guard, d_layout) = self.d.storage_and_layout();
        let (dt_bias_guard, dt_bias_layout) = self.dt_bias.storage_and_layout();
        let a = match &*a_guard {
            Storage::Metal(storage) => storage,
            _ => candle_core::bail!("Mamba2 A tensor is not on Metal"),
        };
        let b = match &*b_guard {
            Storage::Metal(storage) => storage,
            _ => candle_core::bail!("Mamba2 B tensor is not on Metal"),
        };
        let c = match &*c_guard {
            Storage::Metal(storage) => storage,
            _ => candle_core::bail!("Mamba2 C tensor is not on Metal"),
        };
        let d = match &*d_guard {
            Storage::Metal(storage) => storage,
            _ => candle_core::bail!("Mamba2 D tensor is not on Metal"),
        };
        let dt_bias = match &*dt_bias_guard {
            Storage::Metal(storage) => storage,
            _ => candle_core::bail!("Mamba2 dt bias tensor is not on Metal"),
        };
        for (name, storage, layout) in [
            ("a", a, a_layout),
            ("b", b, b_layout),
            ("c", c, c_layout),
            ("d", d, d_layout),
            ("dt_bias", dt_bias, dt_bias_layout),
        ] {
            if storage.dtype() != DType::F32 || !layout.is_contiguous() {
                candle_core::bail!("{name} must be contiguous f32 for Metal Mamba2 scan")
            }
        }

        let state_elements = self.n_heads * self.d_state * self.head_dim;
        let y_elements = self.n_heads * self.head_dim;
        let output_elements = state_elements + y_elements;
        let device = state.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, DType::F32)
            .with_label("nemotron-mamba2-scan")
            .build()?;
        let pipeline = mamba2_scan_pipeline(device)?;
        let encoder = device.command_encoder()?;
        encoder.set_label("nemotron-mamba2-scan");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        let offset = |layout: &Layout| layout.start_offset() * DType::F32.size_in_bytes();
        encoder_ref.set_input_buffer(0, Some(state.buffer()), offset(state_layout));
        encoder_ref.set_input_buffer(1, Some(x.buffer()), offset(x_layout));
        encoder_ref.set_input_buffer(2, Some(dt_raw.buffer()), offset(dt_layout));
        encoder_ref.set_input_buffer(3, Some(a.buffer()), offset(a_layout));
        encoder_ref.set_input_buffer(4, Some(b.buffer()), offset(b_layout));
        encoder_ref.set_input_buffer(5, Some(c.buffer()), offset(c_layout));
        encoder_ref.set_input_buffer(6, Some(d.buffer()), offset(d_layout));
        encoder_ref.set_input_buffer(7, Some(dt_bias.buffer()), offset(dt_bias_layout));
        encoder_ref.set_output_buffer(8, Some(&output), 0);
        encoder_ref.set_bytes(9, &(self.n_heads as u32));
        encoder_ref.set_bytes(10, &(self.d_state as u32));
        encoder_ref.set_bytes(11, &(self.head_dim as u32));
        encoder_ref.set_bytes(12, &(self.n_group as u32));
        let grid = objc2_metal::MTLSize {
            width: y_elements,
            height: 1,
            depth: 1,
        };
        let group = candle_metal_kernels::utils::get_block_dims(y_elements.min(256), 1, 1);
        encoder_ref.dispatch_threads(grid, group);

        Ok((
            MetalStorage::new(output, device.clone(), output_elements, DType::F32),
            Shape::from(output_elements),
        ))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn mamba2_ssm_step_metal(
    state: &Tensor,
    x: &Tensor,
    dt_raw: &Tensor,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    d: &Tensor,
    dt_bias: &Tensor,
    n_group: usize,
) -> anyhow::Result<(Tensor, Tensor)> {
    let (n_heads, d_state, head_dim) = state.dims3()?;
    let op = Mamba2Scan {
        a,
        b,
        c,
        d,
        dt_bias,
        n_heads,
        d_state,
        head_dim,
        n_group,
    };
    let packed = state.apply_op3_no_bwd(x, dt_raw, &op)?;
    let state_elements = n_heads * d_state * head_dim;
    let y_elements = n_heads * head_dim;
    let next_state = packed
        .narrow(0, 0, state_elements)?
        .reshape((n_heads, d_state, head_dim))?;
    let y = packed
        .narrow(0, state_elements, y_elements)?
        .reshape((n_heads, head_dim))?;
    Ok((y, next_state))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mamba_pipeline_cache_is_scoped_to_device_id() -> anyhow::Result<()> {
        let Some(first) = crate::test_metal_device() else {
            return Ok(());
        };
        let second = Device::new_metal(0)?;
        let first_id = first.as_metal_device()?.id();
        let second_id = second.as_metal_device()?.id();
        assert_ne!(first_id, second_id);

        let cache = OnceLock::new();
        named_pipeline(first.as_metal_device()?, "nemotron_mamba2_scan_f32", &cache)?;
        named_pipeline(
            second.as_metal_device()?,
            "nemotron_mamba2_scan_f32",
            &cache,
        )?;
        named_pipeline(first.as_metal_device()?, "nemotron_mamba2_scan_f32", &cache)?;

        let pipelines = cache.get().expect("cache initialized").lock().unwrap();
        assert_eq!(pipelines.len(), 2);
        assert!(pipelines.contains_key(&first_id));
        assert!(pipelines.contains_key(&second_id));
        Ok(())
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{label}[{index}]: got {actual}, expected {expected}"
            );
        }
    }

    fn strided_rows(
        values: &[f32],
        rows: usize,
        columns: usize,
        device: &Device,
    ) -> anyhow::Result<Tensor> {
        let row_stride = columns + 3;
        let mut padded = vec![0.0; rows * row_stride];
        for row in 0..rows {
            padded[row * row_stride + 1..row * row_stride + 1 + columns]
                .copy_from_slice(&values[row * columns..(row + 1) * columns]);
        }
        Ok(Tensor::from_vec(padded, (rows, row_stride), device)?.narrow(1, 1, columns)?)
    }

    #[test]
    fn fused_gate_group_norm_matches_scalar_reference() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (seq_len, n_group, group_size) = (3, 2, 4);
        let inner_dim = n_group * group_size;
        let z_values = (0..seq_len * inner_dim)
            .map(|index| ((index % 11) as f32 - 5.0) * 0.17)
            .collect::<Vec<_>>();
        let y_values = (0..seq_len * inner_dim)
            .map(|index| ((index % 7) as f32 - 3.0) * 0.13)
            .collect::<Vec<_>>();
        let weight_values = (0..inner_dim)
            .map(|index| 0.8 + index as f32 * 0.025)
            .collect::<Vec<_>>();
        let z = strided_rows(&z_values, seq_len, inner_dim, &device)?;
        let y = Tensor::from_vec(y_values.clone(), (seq_len, inner_dim), &device)?;
        let weight = Tensor::from_vec(weight_values.clone(), (n_group, group_size), &device)?;

        let actual = fused_gate_group_norm(&z, &y, &weight)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut expected = vec![0.0; seq_len * inner_dim];
        for token in 0..seq_len {
            for group in 0..n_group {
                let base = (token * n_group + group) * group_size;
                let gated = (0..group_size)
                    .map(|index| {
                        let offset = base + index;
                        z_values[offset] / (1.0 + (-z_values[offset]).exp()) * y_values[offset]
                    })
                    .collect::<Vec<_>>();
                let variance =
                    gated.iter().map(|value| value * value).sum::<f32>() / group_size as f32;
                let scale = (variance + 1e-5).sqrt().recip();
                for index in 0..group_size {
                    expected[base + index] =
                        gated[index] * scale * weight_values[group * group_size + index];
                }
            }
        }
        assert_close(&actual, &expected, 1e-5, "fused gate group norm");
        Ok(())
    }

    fn check_metal_prefill_matches_scalar_reference(
        head_dim: usize,
        strided: bool,
    ) -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (seq_len, n_heads, d_state, n_group, kernel_size) = (3, 2, 3, 1, 3);
        let inner_dim = n_heads * head_dim;
        let group_state_dim = n_group * d_state;
        let conv_dim = inner_dim + 2 * group_state_dim;

        let conv_input_values = (0..seq_len * conv_dim)
            .map(|index| ((index % 13) as f32 - 6.0) * 0.04)
            .collect::<Vec<_>>();
        let conv_state_values = (0..conv_dim * kernel_size)
            .map(|index| ((index % 7) as f32 - 3.0) * 0.02)
            .collect::<Vec<_>>();
        let conv_weight_values = (0..conv_dim * kernel_size)
            .map(|index| match index % kernel_size {
                0 => 0.2,
                1 => -0.15,
                _ => 0.3,
            })
            .collect::<Vec<_>>();
        let conv_bias_values = (0..conv_dim)
            .map(|index| (index as f32 - 4.0) * 0.01)
            .collect::<Vec<_>>();
        let ssm_state_values = (0..n_heads * d_state * head_dim)
            .map(|index| ((index % 9) as f32 - 4.0) * 0.015)
            .collect::<Vec<_>>();
        let dt_values: Vec<f32> = vec![-0.2, 0.1, 0.3, -0.1, 0.2, 0.4];
        let a_values: Vec<f32> = vec![-0.7, -1.1];
        let d_values: Vec<f32> = vec![0.4, -0.2];
        let dt_bias_values: Vec<f32> = vec![0.05, -0.08];

        let conv_input = if strided {
            strided_rows(&conv_input_values, seq_len, conv_dim, &device)?
        } else {
            Tensor::from_vec(conv_input_values.clone(), (seq_len, conv_dim), &device)?
        };
        let conv_state =
            Tensor::from_vec(conv_state_values.clone(), (conv_dim, kernel_size), &device)?;
        let ssm_state = Tensor::from_vec(
            ssm_state_values.clone(),
            (n_heads, d_state, head_dim),
            &device,
        )?;
        let conv_weight =
            Tensor::from_vec(conv_weight_values.clone(), (conv_dim, kernel_size), &device)?;
        let conv_bias = Tensor::from_vec(conv_bias_values.clone(), conv_dim, &device)?;
        let dt_raw = if strided {
            strided_rows(&dt_values, seq_len, n_heads, &device)?
        } else {
            Tensor::from_vec(dt_values.clone(), (seq_len, n_heads), &device)?
        };
        let a = Tensor::from_vec(a_values.clone(), n_heads, &device)?;
        let d = Tensor::from_vec(d_values.clone(), n_heads, &device)?;
        let dt_bias = Tensor::from_vec(dt_bias_values.clone(), n_heads, &device)?;

        let (actual_y, actual_conv_state, actual_ssm_state) = prefill(
            &conv_input,
            &conv_state,
            &ssm_state,
            &conv_weight,
            &conv_bias,
            &dt_raw,
            &a,
            &d,
            &dt_bias,
            n_heads,
            d_state,
            head_dim,
            n_group,
            kernel_size,
        )?;

        let mut histories = conv_state_values
            .chunks_exact(kernel_size)
            .map(<[f32]>::to_vec)
            .collect::<Vec<_>>();
        let mut conv_values = vec![0.0f32; seq_len * conv_dim];
        for token in 0..seq_len {
            for channel in 0..conv_dim {
                histories[channel].rotate_left(1);
                histories[channel][kernel_size - 1] = conv_input_values[token * conv_dim + channel];
                let value = conv_bias_values[channel]
                    + histories[channel]
                        .iter()
                        .zip(
                            &conv_weight_values[channel * kernel_size..(channel + 1) * kernel_size],
                        )
                        .map(|(value, weight)| value * weight)
                        .sum::<f32>();
                conv_values[token * conv_dim + channel] = value / (1.0 + (-value).exp());
            }
        }

        let mut expected_state = ssm_state_values;
        let mut expected_y = vec![0.0f32; seq_len * inner_dim];
        let heads_per_group = n_heads / n_group;
        for token in 0..seq_len {
            for head in 0..n_heads {
                let group = head / heads_per_group;
                let dt_input = dt_values[token * n_heads + head] + dt_bias_values[head];
                let dt = if dt_input <= 20.0 {
                    dt_input.exp().ln_1p()
                } else {
                    dt_input
                };
                let decay = (dt * a_values[head]).exp();
                for dim in 0..head_dim {
                    let y_index = token * inner_dim + head * head_dim + dim;
                    let x = conv_values[token * conv_dim + head * head_dim + dim];
                    let mut sum = 0.0f32;
                    for state_index in 0..d_state {
                        let index = (head * d_state + state_index) * head_dim + dim;
                        let group_index =
                            token * conv_dim + inner_dim + group * d_state + state_index;
                        expected_state[index] =
                            expected_state[index] * decay + conv_values[group_index] * x * dt;
                        let c_index = group_index + group_state_dim;
                        sum += expected_state[index] * conv_values[c_index];
                    }
                    expected_y[y_index] = sum + d_values[head] * x;
                }
            }
        }
        let expected_conv_state = histories.into_iter().flatten().collect::<Vec<_>>();

        assert_close(
            &actual_y.flatten_all()?.to_vec1::<f32>()?,
            &expected_y,
            3e-5,
            "output",
        );
        assert_close(
            &actual_conv_state.flatten_all()?.to_vec1::<f32>()?,
            &expected_conv_state,
            2e-6,
            "conv_state",
        );
        assert_close(
            &actual_ssm_state.flatten_all()?.to_vec1::<f32>()?,
            &expected_state,
            3e-5,
            "ssm_state",
        );
        Ok(())
    }

    #[test]
    fn metal_prefill_matches_scalar_reference() -> anyhow::Result<()> {
        check_metal_prefill_matches_scalar_reference(2, false)
    }

    #[test]
    fn grouped_metal_prefill_matches_scalar_reference() -> anyhow::Result<()> {
        check_metal_prefill_matches_scalar_reference(4, false)
    }

    #[test]
    fn strided_metal_prefill_matches_scalar_reference() -> anyhow::Result<()> {
        check_metal_prefill_matches_scalar_reference(4, true)
    }
}
