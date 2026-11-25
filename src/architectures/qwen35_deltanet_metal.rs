use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{
    CpuStorage, CustomOp2, CustomOp3, DType, Device, Layout, MetalStorage, Result, Shape, Storage,
    Tensor,
};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::{MTLComputePipelineState, MTLDevice};

const DELTANET_MSL: &str = concat!(
    include_str!("activation.metal"),
    r#"
#include <metal_stdlib>
using namespace metal;



kernel void qwen35_deltanet_step_f32(
    device const float *state     [[buffer(0)]],
    device const float *q         [[buffer(1)]],
    device const float *k         [[buffer(2)]],
    device const float *v         [[buffer(3)]],
    device const float *beta_raw  [[buffer(4)]],
    device const float *alpha_raw [[buffer(5)]],
    device const float *a         [[buffer(6)]],
    device const float *dt_bias   [[buffer(7)]],
    device float *output          [[buffer(8)]],
    constant uint &n_heads        [[buffer(9)]],
    constant uint &key_dim        [[buffer(10)]],
    constant uint &value_dim      [[buffer(11)]],
    uint gid [[thread_position_in_grid]]) {
    const uint y_elements = n_heads * value_dim;
    if (gid >= y_elements) {
        return;
    }

    const uint head = gid / value_dim;
    const uint value = gid - head * value_dim;
    const uint qk_offset = head * key_dim;
    const uint state_offset = head * key_dim * value_dim + value;

    const float beta = 1.0f / (1.0f + exp(-beta_raw[head]));
    const float dt_input = alpha_raw[head] + dt_bias[head];
    const float dt = dinf_softplus(dt_input);
    const float decay = exp(a[head] * dt);

    float retrieved = 0.0f;
    for (uint d = 0; d < key_dim; ++d) {
        const uint state_index = state_offset + d * value_dim;
        retrieved += decay * state[state_index] * k[qk_offset + d];
    }

    const float delta = beta * (v[gid] - retrieved);
    float y = 0.0f;
    for (uint d = 0; d < key_dim; ++d) {
        const uint state_index = state_offset + d * value_dim;
        const float next =
            decay * state[state_index] + k[qk_offset + d] * delta;
        output[state_index] = next;
        y += next * q[qk_offset + d];
    }
    output[n_heads * key_dim * value_dim + gid] = y;
}

kernel void qwen35_conv_scan_f32(
    device const float *input       [[buffer(0)]],
    device const float *state       [[buffer(1)]],
    device const float *weight      [[buffer(2)]],
    device float *output            [[buffer(3)]],
    constant uint &seq_len          [[buffer(4)]],
    constant uint &conv_dim         [[buffer(5)]],
    constant uint &kernel_size      [[buffer(6)]],
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
        history[kernel_size - 1] = input[token * conv_dim + channel];

        float value = 0.0f;
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

kernel void qwen35_deltanet_decode_fused_f32(
    device const float *state       [[buffer(0)]],
    device const float *qkv         [[buffer(1)]],
    device const float *conv_state  [[buffer(2)]],
    device const float *conv_weight [[buffer(3)]],
    device const float *alpha_raw  [[buffer(4)]],
    device const float *beta_raw   [[buffer(5)]],
    device const float *a           [[buffer(6)]],
    device const float *dt_bias     [[buffer(7)]],
    device float *output            [[buffer(8)]],
    constant uint &n_k_heads        [[buffer(9)]],
    constant uint &n_v_heads        [[buffer(10)]],
    constant uint &key_dim          [[buffer(11)]],
    constant uint &value_dim        [[buffer(12)]],
    constant uint &kernel_size      [[buffer(13)]],
    device const float *gate        [[buffer(14)]],
    device const float *norm_weight [[buffer(15)]],
    constant float &norm_eps        [[buffer(16)]],
    constant uint &fuse_gate        [[buffer(17)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup float q_values[256];
    threadgroup float k_values[256];
    threadgroup float q_sums[256];
    threadgroup float k_sums[256];
    threadgroup float y_values[256];
    threadgroup float head_terms[4];

    if (head >= n_v_heads) {
        return;
    }

    const uint total_key_dim = n_k_heads * key_dim;
    const uint total_value_dim = n_v_heads * value_dim;
    const uint conv_dim = total_key_dim * 2 + total_value_dim;
    const uint qk_head = head % n_k_heads;
    const uint state_elements = n_v_heads * key_dim * value_dim;
    const uint y_elements = total_value_dim;
    const uint conv_output_offset = state_elements + y_elements;

    float q_sum = 0.0f;
    float k_sum = 0.0f;
    if (tid < key_dim) {
        const uint q_channel = qk_head * key_dim + tid;
        const uint k_channel = total_key_dim + q_channel;
        float q_history[8];
        float k_history[8];
        for (uint index = 0; index < kernel_size; ++index) {
            q_history[index] = conv_state[q_channel * kernel_size + index];
            k_history[index] = conv_state[k_channel * kernel_size + index];
        }
        for (uint index = 1; index < kernel_size; ++index) {
            q_history[index - 1] = q_history[index];
            k_history[index - 1] = k_history[index];
        }
        q_history[kernel_size - 1] = qkv[q_channel];
        k_history[kernel_size - 1] = qkv[k_channel];

        float q_value = 0.0f;
        float k_value = 0.0f;
        for (uint index = 0; index < kernel_size; ++index) {
            q_value += q_history[index]
                * conv_weight[q_channel * kernel_size + index];
            k_value += k_history[index]
                * conv_weight[k_channel * kernel_size + index];
        }
        q_value = q_value / (1.0f + exp(-q_value));
        k_value = k_value / (1.0f + exp(-k_value));
        q_values[tid] = q_value;
        k_values[tid] = k_value;
        q_sum = q_value * q_value;
        k_sum = k_value * k_value;

        // The tiled GQA order maps the first n_k_heads value heads one-to-one
        // with key heads. Those groups own the shared Q/K convolution state.
        if (head < n_k_heads) {
            for (uint index = 0; index < kernel_size; ++index) {
                output[conv_output_offset + q_channel * kernel_size + index] =
                    q_history[index];
                output[conv_output_offset + k_channel * kernel_size + index] =
                    k_history[index];
            }
        }
    }

    float v_value = 0.0f;
    if (tid < value_dim) {
        const uint v_channel = 2 * total_key_dim + head * value_dim + tid;
        float v_history[8];
        for (uint index = 0; index < kernel_size; ++index) {
            v_history[index] = conv_state[v_channel * kernel_size + index];
        }
        for (uint index = 1; index < kernel_size; ++index) {
            v_history[index - 1] = v_history[index];
        }
        v_history[kernel_size - 1] = qkv[v_channel];
        for (uint index = 0; index < kernel_size; ++index) {
            v_value += v_history[index]
                * conv_weight[v_channel * kernel_size + index];
            output[conv_output_offset + v_channel * kernel_size + index] =
                v_history[index];
        }
        v_value = v_value / (1.0f + exp(-v_value));
    }

    q_sums[tid] = q_sum;
    k_sums[tid] = k_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            q_sums[tid] += q_sums[tid + stride];
            k_sums[tid] += k_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        head_terms[0] = rsqrt(q_sums[0] + 1.0e-12f) * rsqrt(float(key_dim));
        head_terms[1] = rsqrt(k_sums[0] + 1.0e-12f);
        head_terms[2] = 1.0f / (1.0f + exp(-beta_raw[head]));
        const float dt_input = alpha_raw[head] + dt_bias[head];
        const float dt = dinf_softplus(dt_input);
        head_terms[3] = exp(a[head] * dt);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < key_dim) {
        q_values[tid] *= head_terms[0];
        k_values[tid] *= head_terms[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < value_dim) {
        const uint state_offset = head * key_dim * value_dim + tid;

        float retrieved = 0.0f;
        for (uint column = 0; column < key_dim; ++column) {
            const uint state_index = state_offset + column * value_dim;
            retrieved += head_terms[3] * state[state_index] * k_values[column];
        }

        const float delta = head_terms[2] * (v_value - retrieved);
        float y = 0.0f;
        for (uint column = 0; column < key_dim; ++column) {
            const uint state_index = state_offset + column * value_dim;
            const float next = head_terms[3] * state[state_index]
                + k_values[column] * delta;
            output[state_index] = next;
            y += next * q_values[column];
        }
        y_values[tid] = y;
    }

    if (fuse_gate == 0) {
        if (tid < value_dim) {
            output[state_elements + head * value_dim + tid] = y_values[tid];
        }
        return;
    }

    // The fused epilogue is enabled only for value_dim=128 with 128 threads.
    // Match dinf_head_rmsnorm_silu_gate_f32's 64-lane reduction order exactly.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64) {
        float sum_sq = 0.0f;
        for (uint column = tid; column < 128; column += 64) {
            const float value = y_values[column];
            sum_sq += value * value;
        }
        q_sums[tid] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (tid < stride) {
            q_sums[tid] += q_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < value_dim) {
        const uint offset = head * value_dim + tid;
        const float gate_value = gate[offset];
        const float inv_rms = rsqrt(q_sums[0] / 128.0f + norm_eps);
        output[state_elements + offset] =
            (gate_value / (1.0f + exp(-gate_value)))
            * y_values[tid] * inv_rms * norm_weight[tid];
    }
}

// Specialization of qwen35_deltanet_decode_fused_f32 for 128x128 heads.
// SPLIT thread slices share each value column: slice j owns key rows
// [j * 128 / SPLIT, (j + 1) * 128 / SPLIT), keeps them in registers, and the
// slices combine their partial dot products through threadgroup memory. The
// generic kernel runs one thread per column over all 128 rows twice, which
// leaves the GPU latency bound with only one threadgroup per head.
constant constexpr uint DELTANET_DECODE_SPLIT = 4;

kernel void qwen35_deltanet_decode_fused_128_f32(
    device const float *state       [[buffer(0)]],
    device const float *qkv         [[buffer(1)]],
    device const float *conv_state  [[buffer(2)]],
    device const float *conv_weight [[buffer(3)]],
    device const float *alpha_raw  [[buffer(4)]],
    device const float *beta_raw   [[buffer(5)]],
    device const float *a           [[buffer(6)]],
    device const float *dt_bias     [[buffer(7)]],
    device float *output            [[buffer(8)]],
    constant uint &n_k_heads        [[buffer(9)]],
    constant uint &n_v_heads        [[buffer(10)]],
    constant uint &key_dim_arg      [[buffer(11)]],
    constant uint &value_dim_arg    [[buffer(12)]],
    constant uint &kernel_size      [[buffer(13)]],
    device const float *gate        [[buffer(14)]],
    device const float *norm_weight [[buffer(15)]],
    constant float &norm_eps        [[buffer(16)]],
    constant uint &fuse_gate        [[buffer(17)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128;
    constexpr uint SPLIT = DELTANET_DECODE_SPLIT;
    constexpr uint CHUNK = D / SPLIT;
    threadgroup float q_values[D];
    threadgroup float k_values[D];
    threadgroup float v_values[D];
    threadgroup float partial[SPLIT * D];
    threadgroup float y_values[D];
    threadgroup float norm_sums[8];
    threadgroup float head_terms[4];

    if (head >= n_v_heads) {
        return;
    }

    const uint total_key_dim = n_k_heads * D;
    const uint total_value_dim = n_v_heads * D;
    const uint qk_head = head % n_k_heads;
    const uint state_elements = n_v_heads * D * D;
    const uint conv_output_offset = state_elements + total_value_dim;

    if (tid < D) {
        const uint q_channel = qk_head * D + tid;
        const uint k_channel = total_key_dim + q_channel;
        const uint v_channel = 2 * total_key_dim + head * D + tid;
        float q_history[8];
        float k_history[8];
        float v_history[8];
        for (uint index = 0; index < kernel_size; ++index) {
            q_history[index] = conv_state[q_channel * kernel_size + index];
            k_history[index] = conv_state[k_channel * kernel_size + index];
            v_history[index] = conv_state[v_channel * kernel_size + index];
        }
        for (uint index = 1; index < kernel_size; ++index) {
            q_history[index - 1] = q_history[index];
            k_history[index - 1] = k_history[index];
            v_history[index - 1] = v_history[index];
        }
        q_history[kernel_size - 1] = qkv[q_channel];
        k_history[kernel_size - 1] = qkv[k_channel];
        v_history[kernel_size - 1] = qkv[v_channel];

        float q_value = 0.0f;
        float k_value = 0.0f;
        float v_value = 0.0f;
        for (uint index = 0; index < kernel_size; ++index) {
            q_value += q_history[index] * conv_weight[q_channel * kernel_size + index];
            k_value += k_history[index] * conv_weight[k_channel * kernel_size + index];
            v_value += v_history[index] * conv_weight[v_channel * kernel_size + index];
            output[conv_output_offset + v_channel * kernel_size + index] = v_history[index];
        }
        q_value = q_value / (1.0f + exp(-q_value));
        k_value = k_value / (1.0f + exp(-k_value));
        q_values[tid] = q_value;
        k_values[tid] = k_value;
        v_values[tid] = v_value / (1.0f + exp(-v_value));

        // The tiled GQA order maps the first n_k_heads value heads one-to-one
        // with key heads. Those groups own the shared Q/K convolution state.
        if (head < n_k_heads) {
            for (uint index = 0; index < kernel_size; ++index) {
                output[conv_output_offset + q_channel * kernel_size + index] = q_history[index];
                output[conv_output_offset + k_channel * kernel_size + index] = k_history[index];
            }
        }
        const float q_sum = simd_sum(q_value * q_value);
        const float k_sum = simd_sum(k_value * k_value);
        if (simd_lane == 0) {
            norm_sums[simd_id] = q_sum;
            norm_sums[4 + simd_id] = k_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        const float q_sum = (norm_sums[0] + norm_sums[1]) + (norm_sums[2] + norm_sums[3]);
        const float k_sum = (norm_sums[4] + norm_sums[5]) + (norm_sums[6] + norm_sums[7]);
        head_terms[0] = rsqrt(q_sum + 1.0e-12f) * rsqrt(float(D));
        head_terms[1] = rsqrt(k_sum + 1.0e-12f);
        head_terms[2] = 1.0f / (1.0f + exp(-beta_raw[head]));
        const float dt_input = alpha_raw[head] + dt_bias[head];
        const float dt = dinf_softplus(dt_input);
        head_terms[3] = exp(a[head] * dt);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint column = tid % D;
    const uint slice = tid / D;
    const uint row0 = slice * CHUNK;
    const uint base = head * D * D + row0 * D + column;
    const float decay = head_terms[3];
    const float k_scale = head_terms[1];
    const float q_scale = head_terms[0];

    float rows[CHUNK];
    float retrieved_part = 0.0f;
    for (uint row = 0; row < CHUNK; ++row) {
        rows[row] = state[base + row * D];
        retrieved_part += decay * rows[row] * (k_values[row0 + row] * k_scale);
    }
    partial[slice * D + column] = retrieved_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float retrieved = 0.0f;
    for (uint part = 0; part < SPLIT; ++part) {
        retrieved += partial[part * D + column];
    }
    const float delta = head_terms[2] * (v_values[column] - retrieved);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float y_part = 0.0f;
    for (uint row = 0; row < CHUNK; ++row) {
        const float next = decay * rows[row] + (k_values[row0 + row] * k_scale) * delta;
        output[base + row * D] = next;
        y_part += next * (q_values[row0 + row] * q_scale);
    }
    partial[slice * D + column] = y_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < D) {
        float y = 0.0f;
        for (uint part = 0; part < SPLIT; ++part) {
            y += partial[part * D + tid];
        }
        y_values[tid] = y;
    }

    if (fuse_gate == 0) {
        if (tid < D) {
            output[state_elements + head * D + tid] = y_values[tid];
        }
        return;
    }

    // Match dinf_head_rmsnorm_silu_gate_f32's 64-lane reduction order exactly.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64) {
        float sum_sq = 0.0f;
        for (uint index = tid; index < 128; index += 64) {
            const float value = y_values[index];
            sum_sq += value * value;
        }
        partial[tid] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < D) {
        const uint offset = head * D + tid;
        const float gate_value = gate[offset];
        const float inv_rms = rsqrt(partial[0] / 128.0f + norm_eps);
        output[state_elements + offset] =
            (gate_value / (1.0f + exp(-gate_value)))
            * y_values[tid] * inv_rms * norm_weight[tid];
    }
}


kernel void qwen35_deltanet_prefill_f32(
    device const float *state       [[buffer(0)]],
    device const float *conv        [[buffer(1)]],
    device const float *alpha_raw   [[buffer(2)]],
    device const float *beta_raw    [[buffer(3)]],
    device const float *a           [[buffer(4)]],
    device const float *dt_bias     [[buffer(5)]],
    device float *output            [[buffer(6)]],
    constant uint &seq_len          [[buffer(7)]],
    constant uint &n_k_heads        [[buffer(8)]],
    constant uint &n_v_heads        [[buffer(9)]],
    constant uint &key_dim          [[buffer(10)]],
    constant uint &value_dim        [[buffer(11)]],
    constant uint &record_replay    [[buffer(12)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup float q_values[256];
    threadgroup float k_values[256];
    threadgroup float q_sums[256];
    threadgroup float k_sums[256];

    if (head >= n_v_heads) {
        return;
    }

    const uint total_key_dim = n_k_heads * key_dim;
    const uint total_value_dim = n_v_heads * value_dim;
    const uint conv_dim = total_key_dim * 2 + total_value_dim;
    const uint qk_head = head % n_k_heads;
    const uint state_output_offset = seq_len * total_value_dim;
    const uint state_elements = n_v_heads * key_dim * value_dim;
    const uint key_output_offset = state_output_offset + state_elements;
    const uint delta_output_offset = key_output_offset
        + seq_len * n_v_heads * key_dim;
    const uint decay_output_offset = delta_output_offset
        + seq_len * n_v_heads * value_dim;

    for (uint token = 0; token < seq_len; ++token) {
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (uint column = tid; column < key_dim; column += threads) {
            const uint q_offset = token * conv_dim + qk_head * key_dim + column;
            const float q = conv[q_offset];
            const float k = conv[q_offset + total_key_dim];
            q_values[column] = q;
            k_values[column] = k;
            q_sum += q * q;
            k_sum += k * k;
        }
        q_sums[tid] = q_sum;
        k_sums[tid] = k_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                q_sums[tid] += q_sums[tid + stride];
                k_sums[tid] += k_sums[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float q_scale = rsqrt(q_sums[0] + 1.0e-12f) * rsqrt(float(key_dim));
        const float k_scale = rsqrt(k_sums[0] + 1.0e-12f);
        const uint gate_offset = token * n_v_heads + head;
        const float beta = 1.0f / (1.0f + exp(-beta_raw[gate_offset]));
        const float dt_input = alpha_raw[gate_offset] + dt_bias[head];
        const float dt = dinf_softplus(dt_input);
        const float decay = exp(a[head] * dt);

        if (record_replay != 0) {
            for (uint column = tid; column < key_dim; column += threads) {
                output[key_output_offset
                    + (token * n_v_heads + head) * key_dim + column] =
                    k_values[column] * k_scale;
            }
            if (tid == 0) {
                output[decay_output_offset + token * n_v_heads + head] = decay;
            }
        }

        for (uint value = tid; value < value_dim; value += threads) {
            const uint head_state_offset = head * key_dim * value_dim + value;
            float retrieved = 0.0f;
            for (uint column = 0; column < key_dim; ++column) {
                const uint state_index = head_state_offset + column * value_dim;
                const float previous = token == 0
                    ? state[state_index]
                    : output[state_output_offset + state_index];
                retrieved += decay * previous * (k_values[column] * k_scale);
            }

            const uint value_offset = token * conv_dim + 2 * total_key_dim
                + head * value_dim + value;
            const float delta = beta * (conv[value_offset] - retrieved);
            float y = 0.0f;
            for (uint column = 0; column < key_dim; ++column) {
                const uint state_index = head_state_offset + column * value_dim;
                const float previous = token == 0
                    ? state[state_index]
                    : output[state_output_offset + state_index];
                const float k = k_values[column] * k_scale;
                const float next = decay * previous + k * delta;
                output[state_output_offset + state_index] = next;
                y += next * (q_values[column] * q_scale);
            }
            output[token * total_value_dim + head * value_dim + value] = y;
            if (record_replay != 0) {
                output[delta_output_offset
                    + (token * n_v_heads + head) * value_dim + value] = delta;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}


kernel void qwen35_deltanet_replay_f32(
    device const float *state  [[buffer(0)]],
    device const float *keys   [[buffer(1)]],
    device const float *deltas [[buffer(2)]],
    device const float *decays [[buffer(3)]],
    device float *output       [[buffer(4)]],
    constant uint &rows        [[buffer(5)]],
    constant uint &n_heads     [[buffer(6)]],
    constant uint &key_dim     [[buffer(7)]],
    constant uint &value_dim   [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    const uint state_elements = n_heads * key_dim * value_dim;
    if (gid >= state_elements) {
        return;
    }
    const uint head_stride = key_dim * value_dim;
    const uint head = gid / head_stride;
    const uint head_index = gid - head * head_stride;
    const uint key = head_index / value_dim;
    const uint value = head_index - key * value_dim;
    float next = state[gid];
    for (uint row = 0; row < rows; ++row) {
        next = decays[row * n_heads + head] * next
            + keys[(row * n_heads + head) * key_dim + key]
                * deltas[(row * n_heads + head) * value_dim + value];
    }
    output[gid] = next;
}
"#,
);

const ONCHIP_PREFILL_MSL: &str = concat!(
    include_str!("activation.metal"),
    r#"
#include <metal_stdlib>
using namespace metal;



// Model-shape specialization for Qwen3.5's 128-dimensional key/value heads,
// following llama.cpp's kernel_gated_delta_net: one SIMD group owns one value
// row of the recurrent state and keeps its 128 key entries in registers
// (four per lane). Each token needs only SIMD reductions, no threadgroup
// barriers. Unlike llama.cpp, the q/k L2 normalization and the beta/decay
// gates are fused here; every SIMD group recomputes them for its token.
kernel void qwen35_deltanet_prefill_onchip_128_f32(
    device const float *state       [[buffer(0)]],
    device const float *conv        [[buffer(1)]],
    device const float *alpha_raw   [[buffer(2)]],
    device const float *beta_raw    [[buffer(3)]],
    device const float *a           [[buffer(4)]],
    device const float *dt_bias     [[buffer(5)]],
    device float *output            [[buffer(6)]],
    constant uint &seq_len          [[buffer(7)]],
    constant uint &n_k_heads        [[buffer(8)]],
    constant uint &n_v_heads        [[buffer(9)]],
    constant uint &key_dim          [[buffer(10)]],
    constant uint &value_dim        [[buffer(11)]],
    constant uint &record_replay    [[buffer(12)]],
    uint group [[threadgroup_position_in_grid]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint Threads = 128;
    constexpr uint RowsPerGroup = Threads / 32;
    constexpr uint GroupsPerHead = 128 / RowsPerGroup;

    if (threads != Threads || key_dim != 128 || value_dim != 128) {
        return;
    }

    const uint head = group / GroupsPerHead;
    const uint value = (group % GroupsPerHead) * RowsPerGroup + simdgroup;
    if (head >= n_v_heads) {
        return;
    }

    const uint total_key_dim = n_k_heads * key_dim;
    const uint total_value_dim = n_v_heads * value_dim;
    const uint conv_dim = total_key_dim * 2 + total_value_dim;
    const uint state_elements = n_v_heads * key_dim * value_dim;
    const uint state_output_offset = seq_len * total_value_dim;
    const uint key_output_offset = state_output_offset + state_elements;
    const uint delta_output_offset = key_output_offset
        + seq_len * n_v_heads * key_dim;
    const uint decay_output_offset = delta_output_offset
        + seq_len * n_v_heads * value_dim;
    const uint qk_head = head % n_k_heads;
    const uint key0 = lane * 4;

    // The state layout is [key][value]; this lane owns keys key0..key0+3 of
    // one value column.
    const uint state_head_offset = head * key_dim * value_dim;
    float s[4];
    for (uint j = 0; j < 4; ++j) {
        s[j] = state[state_head_offset + (key0 + j) * value_dim + value];
    }

    for (uint token = 0; token < seq_len; ++token) {
        const uint q_offset = token * conv_dim + qk_head * key_dim + key0;
        const float4 q = float4(*(device const packed_float4 *)(conv + q_offset));
        const float4 k = float4(
            *(device const packed_float4 *)(conv + q_offset + total_key_dim));
        const float q_sum = simd_sum(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        const float k_sum = simd_sum(k.x * k.x + k.y * k.y + k.z * k.z + k.w * k.w);
        const float q_scale = rsqrt(q_sum + 1.0e-12f) * rsqrt(float(key_dim));
        const float k_scale = rsqrt(k_sum + 1.0e-12f);
        const uint gate_offset = token * n_v_heads + head;
        const float beta = 1.0f / (1.0f + exp(-beta_raw[gate_offset]));
        const float dt = dinf_softplus(alpha_raw[gate_offset] + dt_bias[head]);
        const float decay = exp(a[head] * dt);
        const float kk[4] = {k.x * k_scale, k.y * k_scale, k.z * k_scale, k.w * k_scale};
        const float qq[4] = {q.x * q_scale, q.y * q_scale, q.z * q_scale, q.w * q_scale};

        if (record_replay != 0 && value == 0) {
            for (uint j = 0; j < 4; ++j) {
                output[key_output_offset + gate_offset * key_dim + key0 + j] = kk[j];
            }
            if (lane == 0) {
                output[decay_output_offset + gate_offset] = decay;
            }
        }

        float retrieved = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            retrieved += decay * s[j] * kk[j];
        }
        retrieved = simd_sum(retrieved);
        const float v = conv[token * conv_dim + 2 * total_key_dim
            + head * value_dim + value];
        const float delta = beta * (v - retrieved);
        float y = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            s[j] = decay * s[j] + kk[j] * delta;
            y += s[j] * qq[j];
        }
        y = simd_sum(y);
        if (lane == 0) {
            output[token * total_value_dim + head * value_dim + value] = y;
            if (record_replay != 0) {
                output[delta_output_offset + gate_offset * value_dim + value] = delta;
            }
        }
    }

    for (uint j = 0; j < 4; ++j) {
        output[state_output_offset + state_head_offset
            + (key0 + j) * value_dim + value] = s[j];
    }
}

// The kernel above, split in two so the sequential token loop carries only
// the state update. The prep kernel computes every token's normalized q and
// k and its beta and decay in parallel, with the same expressions in the same
// order (the scan's outputs are bit-identical). The scan then loads each
// token's inputs one token ahead.
//
// prep layout: [token][q | k] (2 * n_k_heads * 128 per token), then beta
// [token][v head], then decay [token][v head].
kernel void qwen35_deltanet_prefill_prep_128_f32(
    device const float *conv        [[buffer(0)]],
    device const float *alpha_raw   [[buffer(1)]],
    device const float *beta_raw    [[buffer(2)]],
    device const float *a           [[buffer(3)]],
    device const float *dt_bias     [[buffer(4)]],
    device float *prep              [[buffer(5)]],
    constant uint &seq_len          [[buffer(6)]],
    constant uint &n_k_heads        [[buffer(7)]],
    constant uint &n_v_heads        [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint key_dim = 128;
    const uint token = group / n_k_heads;
    const uint k_head = group % n_k_heads;
    if (token >= seq_len) {
        return;
    }
    const uint total_key_dim = n_k_heads * key_dim;
    const uint conv_dim = total_key_dim * 2 + n_v_heads * 128;
    const uint key0 = lane * 4;

    const uint q_offset = token * conv_dim + k_head * key_dim + key0;
    const float4 q = float4(*(device const packed_float4 *)(conv + q_offset));
    const float4 k = float4(
        *(device const packed_float4 *)(conv + q_offset + total_key_dim));
    const float q_sum = simd_sum(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    const float k_sum = simd_sum(k.x * k.x + k.y * k.y + k.z * k.z + k.w * k.w);
    const float q_scale = rsqrt(q_sum + 1.0e-12f) * rsqrt(float(key_dim));
    const float k_scale = rsqrt(k_sum + 1.0e-12f);
    const uint out = token * 2 * total_key_dim + k_head * key_dim + key0;
    *(device packed_float4 *)(prep + out) =
        packed_float4(q.x * q_scale, q.y * q_scale, q.z * q_scale, q.w * q_scale);
    *(device packed_float4 *)(prep + out + total_key_dim) =
        packed_float4(k.x * k_scale, k.y * k_scale, k.z * k_scale, k.w * k_scale);

    // gates of the value heads that read this key head (head % n_k_heads)
    const uint heads_per_key = n_v_heads / n_k_heads;
    if (lane < heads_per_key) {
        const uint head = k_head + lane * n_k_heads;
        const uint gate_offset = token * n_v_heads + head;
        const uint gates = seq_len * 2 * total_key_dim;
        prep[gates + gate_offset] = 1.0f / (1.0f + exp(-beta_raw[gate_offset]));
        const float dt = dinf_softplus(alpha_raw[gate_offset] + dt_bias[head]);
        prep[gates + seq_len * n_v_heads + gate_offset] = exp(a[head] * dt);
    }
}

kernel void qwen35_deltanet_prefill_onchip_prepped_128_f32(
    device const float *state       [[buffer(0)]],
    device const float *conv        [[buffer(1)]],
    device const float *prep        [[buffer(2)]],
    device float *output            [[buffer(3)]],
    constant uint &seq_len          [[buffer(4)]],
    constant uint &n_k_heads        [[buffer(5)]],
    constant uint &n_v_heads        [[buffer(6)]],
    constant uint &record_replay    [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint Threads = 128;
    constexpr uint RowsPerGroup = Threads / 32;
    constexpr uint GroupsPerHead = 128 / RowsPerGroup;
    constexpr uint key_dim = 128;
    constexpr uint value_dim = 128;

    if (threads != Threads) {
        return;
    }

    const uint head = group / GroupsPerHead;
    const uint value = (group % GroupsPerHead) * RowsPerGroup + simdgroup;
    if (head >= n_v_heads) {
        return;
    }

    const uint total_key_dim = n_k_heads * key_dim;
    const uint total_value_dim = n_v_heads * value_dim;
    const uint conv_dim = total_key_dim * 2 + total_value_dim;
    const uint state_elements = n_v_heads * key_dim * value_dim;
    const uint state_output_offset = seq_len * total_value_dim;
    const uint key_output_offset = state_output_offset + state_elements;
    const uint delta_output_offset = key_output_offset
        + seq_len * n_v_heads * key_dim;
    const uint decay_output_offset = delta_output_offset
        + seq_len * n_v_heads * value_dim;
    const uint qk_head = head % n_k_heads;
    const uint key0 = lane * 4;
    const uint gates = seq_len * 2 * total_key_dim;

    const uint state_head_offset = head * key_dim * value_dim;
    float s[4];
    for (uint j = 0; j < 4; ++j) {
        s[j] = state[state_head_offset + (key0 + j) * value_dim + value];
    }

    float4 qn, kn;
    float v, beta, decay;
    auto load = [&](uint token) {
        const uint qk = token * 2 * total_key_dim + qk_head * key_dim + key0;
        qn = float4(*(device const packed_float4 *)(prep + qk));
        kn = float4(*(device const packed_float4 *)(prep + qk + total_key_dim));
        v = conv[token * conv_dim + 2 * total_key_dim + head * value_dim + value];
        beta = prep[gates + token * n_v_heads + head];
        decay = prep[gates + seq_len * n_v_heads + token * n_v_heads + head];
    };
    load(0);

    for (uint token = 0; token < seq_len; ++token) {
        const float kk[4] = {kn.x, kn.y, kn.z, kn.w};
        const float qq[4] = {qn.x, qn.y, qn.z, qn.w};
        const float tv = v;
        const float tbeta = beta;
        const float tdecay = decay;
        if (token + 1 < seq_len) {
            load(token + 1);
        }
        const uint gate_offset = token * n_v_heads + head;

        if (record_replay != 0 && value == 0) {
            for (uint j = 0; j < 4; ++j) {
                output[key_output_offset + gate_offset * key_dim + key0 + j] = kk[j];
            }
            if (lane == 0) {
                output[decay_output_offset + gate_offset] = tdecay;
            }
        }

        float retrieved = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            retrieved += tdecay * s[j] * kk[j];
        }
        retrieved = simd_sum(retrieved);
        const float delta = tbeta * (tv - retrieved);
        float y = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            s[j] = tdecay * s[j] + kk[j] * delta;
            y += s[j] * qq[j];
        }
        y = simd_sum(y);
        if (lane == 0) {
            output[token * total_value_dim + head * value_dim + value] = y;
            if (record_replay != 0) {
                output[delta_output_offset + gate_offset * value_dim + value] = delta;
            }
        }
    }

    for (uint j = 0; j < 4; ++j) {
        output[state_output_offset + state_head_offset
            + (key0 + j) * value_dim + value] = s[j];
    }
}
"#,
);

static DELTANET_PREFILL_PREP_PIPELINE: PipelineCache = OnceLock::new();
static DELTANET_PREPPED_SCAN_PIPELINE: PipelineCache = OnceLock::new();

/// The prep and scan pipelines of the split on-chip prefill, when available.
/// `DINF_DISABLE_QWEN35_PREFILL_PREP=1` keeps the single kernel.
fn prepped_prefill_pipelines(
    device: &candle_core::MetalDevice,
) -> Option<(ComputePipeline, ComputePipeline)> {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    if *DISABLED.get_or_init(|| {
        std::env::var_os("DINF_DISABLE_QWEN35_PREFILL_PREP").is_some_and(|v| v == "1")
    }) {
        return None;
    }
    let prep = named_pipeline_from_source(
        device,
        "qwen35_deltanet_prefill_prep_128_f32",
        &DELTANET_PREFILL_PREP_PIPELINE,
        ONCHIP_PREFILL_MSL,
    )
    .ok()?;
    let scan = named_pipeline_from_source(
        device,
        "qwen35_deltanet_prefill_onchip_prepped_128_f32",
        &DELTANET_PREPPED_SCAN_PIPELINE,
        ONCHIP_PREFILL_MSL,
    )
    .ok()?;
    (scan.max_total_threads_per_threadgroup() >= 128).then_some((prep, scan))
}

type PipelineCache = OnceLock<Mutex<HashMap<DeviceId, ComputePipeline>>>;

static DELTANET_PIPELINE: PipelineCache = OnceLock::new();
static FUSED_DECODE_PIPELINE: PipelineCache = OnceLock::new();
static FUSED_DECODE_128_PIPELINE: PipelineCache = OnceLock::new();
static CONV_SCAN_PIPELINE: PipelineCache = OnceLock::new();
static DELTANET_PREFILL_PIPELINE: PipelineCache = OnceLock::new();
static DELTANET_ONCHIP_PREFILL_PIPELINE: PipelineCache = OnceLock::new();
static DELTANET_ONCHIP_PREFILL_FALLBACK: OnceLock<Mutex<HashSet<DeviceId>>> = OnceLock::new();
static DELTANET_REPLAY_PIPELINE: PipelineCache = OnceLock::new();

fn pipeline(device: &candle_core::MetalDevice) -> Result<ComputePipeline> {
    named_pipeline_from_source(
        device,
        "qwen35_deltanet_step_f32",
        &DELTANET_PIPELINE,
        DELTANET_MSL,
    )
}

fn named_pipeline(
    device: &candle_core::MetalDevice,
    name: &str,
    slot: &PipelineCache,
) -> Result<ComputePipeline> {
    named_pipeline_from_source(device, name, slot, DELTANET_MSL)
}

fn named_pipeline_from_source(
    device: &candle_core::MetalDevice,
    name: &str,
    slot: &PipelineCache,
    source: &str,
) -> Result<ComputePipeline> {
    let pipelines = slot.get_or_init(|| Mutex::new(HashMap::new()));
    let device_id = device.id();
    if let Some(pipeline) = pipelines.lock().unwrap().get(&device_id).cloned() {
        return Ok(pipeline);
    }

    let library = device
        .metal_device()
        .new_library_with_source(source, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function(name, None)
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    let mut pipelines = pipelines.lock().unwrap();
    Ok(pipelines.entry(device_id).or_insert(pipeline).clone())
}

fn onchip_prefill_pipeline(device: &candle_core::MetalDevice) -> Result<Option<ComputePipeline>> {
    let fallback_devices =
        DELTANET_ONCHIP_PREFILL_FALLBACK.get_or_init(|| Mutex::new(HashSet::new()));
    onchip_prefill_pipeline_with_cache(device, &DELTANET_ONCHIP_PREFILL_PIPELINE, fallback_devices)
}

fn onchip_prefill_pipeline_with_cache(
    device: &candle_core::MetalDevice,
    pipeline_slot: &PipelineCache,
    fallback_devices: &Mutex<HashSet<DeviceId>>,
) -> Result<Option<ComputePipeline>> {
    let device_id = device.id();
    if fallback_devices.lock().unwrap().contains(&device_id) {
        return Ok(None);
    }
    match named_pipeline_from_source(
        device,
        "qwen35_deltanet_prefill_onchip_128_f32",
        pipeline_slot,
        ONCHIP_PREFILL_MSL,
    ) {
        Ok(pipeline)
            if pipeline.max_total_threads_per_threadgroup() >= 128
                && pipeline.as_ref().staticThreadgroupMemoryLength()
                    <= device.metal_device().as_ref().maxThreadgroupMemoryLength() =>
        {
            Ok(Some(pipeline))
        }
        Ok(pipeline) => {
            fallback_devices.lock().unwrap().insert(device_id);
            tracing::warn!(
                max_threads = pipeline.max_total_threads_per_threadgroup(),
                scratch_bytes = pipeline.as_ref().staticThreadgroupMemoryLength(),
                max_scratch_bytes = device.metal_device().as_ref().maxThreadgroupMemoryLength(),
                "Qwen3.5 on-chip DeltaNet prefill disabled: Metal threadgroup limits are too small"
            );
            Ok(None)
        }
        Err(error) => {
            fallback_devices.lock().unwrap().insert(device_id);
            tracing::warn!(
                error = %error,
                "Qwen3.5 on-chip DeltaNet prefill unavailable; using generic Metal scan"
            );
            Ok(None)
        }
    }
}

pub(super) fn precompile(device: &Device) -> anyhow::Result<()> {
    if device.is_metal() {
        let device = device.as_metal_device()?;
        pipeline(device)?;
        named_pipeline(
            device,
            "qwen35_deltanet_decode_fused_f32",
            &FUSED_DECODE_PIPELINE,
        )?;
        named_pipeline(
            device,
            "qwen35_deltanet_decode_fused_128_f32",
            &FUSED_DECODE_128_PIPELINE,
        )?;
        named_pipeline(device, "qwen35_conv_scan_f32", &CONV_SCAN_PIPELINE)?;
        named_pipeline(
            device,
            "qwen35_deltanet_prefill_f32",
            &DELTANET_PREFILL_PIPELINE,
        )?;
        // The model-shape specialization is best effort.  Its compile failure
        // must never prevent the generic Metal scan from being used.
        let _ = onchip_prefill_pipeline(device)?;
        named_pipeline(
            device,
            "qwen35_deltanet_replay_f32",
            &DELTANET_REPLAY_PIPELINE,
        )?;
    }
    Ok(())
}

struct ConvScan<'a> {
    weight: &'a Tensor,
    seq_len: usize,
    conv_dim: usize,
    kernel_size: usize,
}

impl CustomOp2 for ConvScan<'_> {
    fn name(&self) -> &'static str {
        "qwen35-conv-scan"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _state: &CpuStorage,
        _state_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("qwen35-conv-scan is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        input_layout: &Layout,
        state: &MetalStorage,
        state_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        for (name, storage, layout, expected) in [
            ("input", input, input_layout, self.seq_len * self.conv_dim),
            (
                "state",
                state,
                state_layout,
                self.conv_dim * self.kernel_size,
            ),
        ] {
            if storage.dtype() != DType::F32
                || !layout.is_contiguous()
                || layout.shape().elem_count() != expected
            {
                candle_core::bail!(
                    "{name} must be contiguous f32 with {expected} elements for Qwen3.5 conv scan"
                )
            }
        }
        let (weight_guard, weight_layout) = self.weight.storage_and_layout();
        let weight = metal_storage("weight", &weight_guard, weight_layout)?;
        if weight_layout.shape().elem_count() != self.conv_dim * self.kernel_size {
            candle_core::bail!("invalid Qwen3.5 conv weight shape")
        }

        let output_elements = self.seq_len * self.conv_dim + self.conv_dim * self.kernel_size;
        let device = input.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, DType::F32)
            .with_label("qwen35-conv-scan")
            .build()?;
        let pipeline = named_pipeline(device, "qwen35_conv_scan_f32", &CONV_SCAN_PIPELINE)?;
        let encoder = device.command_encoder()?;
        encoder.set_label("qwen35-conv-scan");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        let offset = |layout: &Layout| layout.start_offset() * DType::F32.size_in_bytes();
        encoder_ref.set_input_buffer(0, Some(input.buffer()), offset(input_layout));
        encoder_ref.set_input_buffer(1, Some(state.buffer()), offset(state_layout));
        encoder_ref.set_input_buffer(2, Some(weight.buffer()), offset(weight_layout));
        encoder_ref.set_output_buffer(3, Some(&output), 0);
        encoder_ref.set_bytes(4, &(self.seq_len as u32));
        encoder_ref.set_bytes(5, &(self.conv_dim as u32));
        encoder_ref.set_bytes(6, &(self.kernel_size as u32));
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

struct DeltaNetPrefill<'a> {
    beta_raw: &'a Tensor,
    a: &'a Tensor,
    dt_bias: &'a Tensor,
    seq_len: usize,
    n_k_heads: usize,
    n_v_heads: usize,
    key_dim: usize,
    value_dim: usize,
    record_replay: bool,
    onchip_state: bool,
}

impl CustomOp3 for DeltaNetPrefill<'_> {
    fn name(&self) -> &'static str {
        "qwen35-deltanet-prefill"
    }

    fn cpu_fwd(
        &self,
        _state: &CpuStorage,
        _state_layout: &Layout,
        _conv: &CpuStorage,
        _conv_layout: &Layout,
        _alpha_raw: &CpuStorage,
        _alpha_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("qwen35-deltanet-prefill is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        state: &MetalStorage,
        state_layout: &Layout,
        conv: &MetalStorage,
        conv_layout: &Layout,
        alpha_raw: &MetalStorage,
        alpha_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let total_key_dim = self.n_k_heads * self.key_dim;
        let total_value_dim = self.n_v_heads * self.value_dim;
        let conv_dim = total_key_dim * 2 + total_value_dim;
        let state_elements = self.n_v_heads * self.key_dim * self.value_dim;
        for (name, storage, layout, expected) in [
            ("state", state, state_layout, state_elements),
            ("conv", conv, conv_layout, self.seq_len * conv_dim),
            (
                "alpha_raw",
                alpha_raw,
                alpha_layout,
                self.seq_len * self.n_v_heads,
            ),
        ] {
            if storage.dtype() != DType::F32
                || !layout.is_contiguous()
                || layout.shape().elem_count() != expected
            {
                candle_core::bail!(
                    "{name} must be contiguous f32 with {expected} elements for Qwen3.5 prefill"
                )
            }
        }

        let (beta_guard, beta_layout) = self.beta_raw.storage_and_layout();
        let (a_guard, a_layout) = self.a.storage_and_layout();
        let (dt_guard, dt_layout) = self.dt_bias.storage_and_layout();
        let beta = metal_storage("beta_raw", &beta_guard, beta_layout)?;
        let a = metal_storage("a", &a_guard, a_layout)?;
        let dt = metal_storage("dt_bias", &dt_guard, dt_layout)?;
        if beta_layout.shape().elem_count() != self.seq_len * self.n_v_heads
            || a_layout.shape().elem_count() != self.n_v_heads
            || dt_layout.shape().elem_count() != self.n_v_heads
        {
            candle_core::bail!("invalid Qwen3.5 prefill gate shape")
        }

        let y_elements = self.seq_len * total_value_dim;
        let replay_elements = if self.record_replay {
            self.seq_len * self.n_v_heads * (self.key_dim + self.value_dim + 1)
        } else {
            0
        };
        let output_elements = y_elements + state_elements + replay_elements;
        let device = state.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, DType::F32)
            .with_label("qwen35-deltanet-prefill")
            .build()?;
        let requested_onchip = self.onchip_state && self.key_dim == 128 && self.value_dim == 128;
        let (pipeline, onchip) = if requested_onchip {
            if let Some(pipeline) = onchip_prefill_pipeline(device)? {
                (pipeline, true)
            } else {
                (
                    named_pipeline(
                        device,
                        "qwen35_deltanet_prefill_f32",
                        &DELTANET_PREFILL_PIPELINE,
                    )?,
                    false,
                )
            }
        } else {
            (
                named_pipeline(
                    device,
                    "qwen35_deltanet_prefill_f32",
                    &DELTANET_PREFILL_PIPELINE,
                )?,
                false,
            )
        };
        // Split on-chip path: gates and normalized q/k first, then the scan.
        let prep_offset = |layout: &Layout| layout.start_offset() * DType::F32.size_in_bytes();
        let heads_per_key = self.n_v_heads / self.n_k_heads.max(1);
        if onchip
            && self.n_k_heads > 0
            && self.n_v_heads.is_multiple_of(self.n_k_heads)
            && heads_per_key <= 32
            && let Some((prep_pipeline, scan_pipeline)) = prepped_prefill_pipelines(device)
        {
            let prep_elements = self.seq_len * (2 * total_key_dim + 2 * self.n_v_heads);
            let prep = device
                .new_buffer_builder()
                .with_size_for(prep_elements, DType::F32)
                .with_label("qwen35-deltanet-prefill-prep")
                .build()?;
            let encoder = device.command_encoder()?;
            encoder.set_label("qwen35-deltanet-prefill");
            let e: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
            e.set_compute_pipeline_state(&prep_pipeline);
            e.set_input_buffer(0, Some(conv.buffer()), prep_offset(conv_layout));
            e.set_input_buffer(1, Some(alpha_raw.buffer()), prep_offset(alpha_layout));
            e.set_input_buffer(2, Some(beta.buffer()), prep_offset(beta_layout));
            e.set_input_buffer(3, Some(a.buffer()), prep_offset(a_layout));
            e.set_input_buffer(4, Some(dt.buffer()), prep_offset(dt_layout));
            e.set_output_buffer(5, Some(&prep), 0);
            e.set_bytes(6, &(self.seq_len as u32));
            e.set_bytes(7, &(self.n_k_heads as u32));
            e.set_bytes(8, &(self.n_v_heads as u32));
            e.dispatch_thread_groups(
                objc2_metal::MTLSize {
                    width: self.seq_len * self.n_k_heads,
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
            e.set_compute_pipeline_state(&scan_pipeline);
            e.set_input_buffer(0, Some(state.buffer()), prep_offset(state_layout));
            e.set_input_buffer(1, Some(conv.buffer()), prep_offset(conv_layout));
            e.set_input_buffer(2, Some(&prep), 0);
            e.set_output_buffer(3, Some(&output), 0);
            e.set_bytes(4, &(self.seq_len as u32));
            e.set_bytes(5, &(self.n_k_heads as u32));
            e.set_bytes(6, &(self.n_v_heads as u32));
            e.set_bytes(7, &(self.record_replay as u32));
            e.dispatch_thread_groups(
                objc2_metal::MTLSize {
                    width: self.n_v_heads * 32,
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
            return Ok((
                MetalStorage::new(output, device.clone(), output_elements, DType::F32),
                Shape::from(output_elements),
            ));
        }
        let encoder = device.command_encoder()?;
        encoder.set_label("qwen35-deltanet-prefill");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        let offset = |layout: &Layout| layout.start_offset() * DType::F32.size_in_bytes();
        encoder_ref.set_input_buffer(0, Some(state.buffer()), offset(state_layout));
        encoder_ref.set_input_buffer(1, Some(conv.buffer()), offset(conv_layout));
        encoder_ref.set_input_buffer(2, Some(alpha_raw.buffer()), offset(alpha_layout));
        encoder_ref.set_input_buffer(3, Some(beta.buffer()), offset(beta_layout));
        encoder_ref.set_input_buffer(4, Some(a.buffer()), offset(a_layout));
        encoder_ref.set_input_buffer(5, Some(dt.buffer()), offset(dt_layout));
        encoder_ref.set_output_buffer(6, Some(&output), 0);
        encoder_ref.set_bytes(7, &(self.seq_len as u32));
        encoder_ref.set_bytes(8, &(self.n_k_heads as u32));
        encoder_ref.set_bytes(9, &(self.n_v_heads as u32));
        encoder_ref.set_bytes(10, &(self.key_dim as u32));
        encoder_ref.set_bytes(11, &(self.value_dim as u32));
        encoder_ref.set_bytes(12, &(self.record_replay as u32));
        let max_threads = pipeline.max_total_threads_per_threadgroup().max(1);
        let max_power_of_two = 1usize << max_threads.ilog2();
        let threads = self
            .key_dim
            .max(self.value_dim)
            .next_power_of_two()
            .min(256)
            .min(max_power_of_two);
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: if onchip {
                    self.n_v_heads * 32
                } else {
                    self.n_v_heads
                },
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: threads,
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

struct FusedDecode<'a> {
    conv_weight: &'a Tensor,
    alpha_raw: &'a Tensor,
    beta_raw: &'a Tensor,
    a: &'a Tensor,
    dt_bias: &'a Tensor,
    gate: &'a Tensor,
    norm_weight: &'a Tensor,
    norm_eps: f32,
    fuse_gate: bool,
    n_k_heads: usize,
    n_v_heads: usize,
    key_dim: usize,
    value_dim: usize,
    kernel_size: usize,
    /// Use `qwen35_deltanet_decode_fused_128_f32`; requires 128x128 heads.
    specialized: bool,
}

impl CustomOp3 for FusedDecode<'_> {
    fn name(&self) -> &'static str {
        "qwen35-deltanet-fused-decode"
    }

    fn cpu_fwd(
        &self,
        _state: &CpuStorage,
        _state_layout: &Layout,
        _qkv: &CpuStorage,
        _qkv_layout: &Layout,
        _conv_state: &CpuStorage,
        _conv_state_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("qwen35-deltanet-fused-decode is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        state: &MetalStorage,
        state_layout: &Layout,
        qkv: &MetalStorage,
        qkv_layout: &Layout,
        conv_state: &MetalStorage,
        conv_state_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let total_key_dim = self.n_k_heads * self.key_dim;
        let total_value_dim = self.n_v_heads * self.value_dim;
        let conv_dim = total_key_dim * 2 + total_value_dim;
        let state_elements = self.n_v_heads * self.key_dim * self.value_dim;
        for (name, storage, layout, expected) in [
            ("state", state, state_layout, state_elements),
            ("qkv", qkv, qkv_layout, conv_dim),
            (
                "conv_state",
                conv_state,
                conv_state_layout,
                conv_dim * self.kernel_size,
            ),
        ] {
            if storage.dtype() != DType::F32
                || !layout.is_contiguous()
                || layout.shape().elem_count() != expected
            {
                candle_core::bail!(
                    "{name} must be contiguous f32 with {expected} elements for Qwen3.5 fused decode"
                )
            }
            if storage.device().id() != state.device().id() {
                candle_core::bail!("Qwen3.5 fused decode tensors are on different devices")
            }
        }

        let (weight_guard, weight_layout) = self.conv_weight.storage_and_layout();
        let (alpha_guard, alpha_layout) = self.alpha_raw.storage_and_layout();
        let (beta_guard, beta_layout) = self.beta_raw.storage_and_layout();
        let (a_guard, a_layout) = self.a.storage_and_layout();
        let (dt_guard, dt_layout) = self.dt_bias.storage_and_layout();
        let weight = metal_storage("conv_weight", &weight_guard, weight_layout)?;
        let alpha = metal_storage("alpha_raw", &alpha_guard, alpha_layout)?;
        let beta = metal_storage("beta_raw", &beta_guard, beta_layout)?;
        let a = metal_storage("a", &a_guard, a_layout)?;
        let dt = metal_storage("dt_bias", &dt_guard, dt_layout)?;
        for storage in [weight, alpha, beta, a, dt] {
            if storage.device().id() != state.device().id() {
                candle_core::bail!("Qwen3.5 fused decode tensors are on different devices")
            }
        }
        if weight_layout.shape().elem_count() != conv_dim * self.kernel_size
            || alpha_layout.shape().elem_count() != self.n_v_heads
            || beta_layout.shape().elem_count() != self.n_v_heads
            || a_layout.shape().elem_count() != self.n_v_heads
            || dt_layout.shape().elem_count() != self.n_v_heads
        {
            candle_core::bail!("invalid Qwen3.5 fused decode tensor shape")
        }
        if self.fuse_gate
            && (self.key_dim != 128
                || self.value_dim != 128
                || !self.norm_eps.is_finite()
                || self.norm_eps < 0.0)
        {
            candle_core::bail!("unsupported Qwen3.5 fused decode gate geometry")
        }

        let y_elements = total_value_dim;
        let conv_state_elements = conv_dim * self.kernel_size;
        let output_elements = state_elements + y_elements + conv_state_elements;
        let device = state.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, DType::F32)
            .with_label("qwen35-deltanet-fused-decode")
            .build()?;
        if self.specialized && (self.key_dim != 128 || self.value_dim != 128) {
            candle_core::bail!("Qwen3.5 specialized fused decode requires 128x128 heads")
        }
        let pipeline = match self.specialized {
            true => named_pipeline(
                device,
                "qwen35_deltanet_decode_fused_128_f32",
                &FUSED_DECODE_128_PIPELINE,
            )?,
            false => named_pipeline(
                device,
                "qwen35_deltanet_decode_fused_f32",
                &FUSED_DECODE_PIPELINE,
            )?,
        };
        let encoder = device.command_encoder()?;
        encoder.set_label("qwen35-deltanet-fused-decode");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        let offset = |layout: &Layout| layout.start_offset() * DType::F32.size_in_bytes();
        encoder_ref.set_input_buffer(0, Some(state.buffer()), offset(state_layout));
        encoder_ref.set_input_buffer(1, Some(qkv.buffer()), offset(qkv_layout));
        encoder_ref.set_input_buffer(2, Some(conv_state.buffer()), offset(conv_state_layout));
        encoder_ref.set_input_buffer(3, Some(weight.buffer()), offset(weight_layout));
        encoder_ref.set_input_buffer(4, Some(alpha.buffer()), offset(alpha_layout));
        encoder_ref.set_input_buffer(5, Some(beta.buffer()), offset(beta_layout));
        encoder_ref.set_input_buffer(6, Some(a.buffer()), offset(a_layout));
        encoder_ref.set_input_buffer(7, Some(dt.buffer()), offset(dt_layout));
        encoder_ref.set_output_buffer(8, Some(&output), 0);
        encoder_ref.set_bytes(9, &(self.n_k_heads as u32));
        encoder_ref.set_bytes(10, &(self.n_v_heads as u32));
        encoder_ref.set_bytes(11, &(self.key_dim as u32));
        encoder_ref.set_bytes(12, &(self.value_dim as u32));
        encoder_ref.set_bytes(13, &(self.kernel_size as u32));
        if self.fuse_gate {
            let (gate_guard, gate_layout) = self.gate.storage_and_layout();
            let (norm_guard, norm_layout) = self.norm_weight.storage_and_layout();
            let gate = metal_storage("gate", &gate_guard, gate_layout)?;
            let norm = metal_storage("norm_weight", &norm_guard, norm_layout)?;
            if gate.device().id() != state.device().id()
                || norm.device().id() != state.device().id()
            {
                candle_core::bail!("Qwen3.5 fused decode tensors are on different devices")
            }
            if gate_layout.shape().elem_count() != total_value_dim
                || norm_layout.shape().elem_count() != self.value_dim
            {
                candle_core::bail!("invalid Qwen3.5 fused decode gate shape")
            }
            encoder_ref.set_input_buffer(14, Some(gate.buffer()), offset(gate_layout));
            encoder_ref.set_input_buffer(15, Some(norm.buffer()), offset(norm_layout));
        } else {
            // The kernel never reads these bindings when the epilogue is off.
            encoder_ref.set_input_buffer(14, Some(state.buffer()), offset(state_layout));
            encoder_ref.set_input_buffer(15, Some(state.buffer()), offset(state_layout));
        }
        encoder_ref.set_bytes(16, &self.norm_eps);
        encoder_ref.set_bytes(17, &(self.fuse_gate as u32));
        let max_threads = pipeline.max_total_threads_per_threadgroup().max(1);
        let max_power_of_two = 1usize << max_threads.ilog2();
        let threads = if self.specialized {
            let threads = FUSED_DECODE_128_THREADS;
            if threads > max_threads {
                candle_core::bail!(
                    "Qwen3.5 fused decode needs {threads} Metal threads, pipeline allows {max_threads}"
                )
            }
            threads
        } else {
            self.key_dim
                .max(self.value_dim)
                .next_power_of_two()
                .min(256)
                .min(max_power_of_two)
        };
        if self.fuse_gate && !self.specialized && threads != 128 {
            candle_core::bail!("Qwen3.5 fused decode gate requires exactly 128 Metal threads")
        }
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: self.n_v_heads,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: threads,
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

/// Threads per head in `qwen35_deltanet_decode_fused_128_f32`: 128 value
/// columns times `DELTANET_DECODE_SPLIT` key-row slices.
const FUSED_DECODE_128_THREADS: usize = 512;

/// Whether decode can use the 128x128 specialization.
fn fused_decode_128_supported(key_dim: usize, value_dim: usize) -> bool {
    key_dim == 128 && value_dim == 128
}

#[allow(clippy::too_many_arguments)]
fn fused_decode(
    qkv: &Tensor,
    conv_state: &Tensor,
    recurrent_state: &Tensor,
    conv_weight: &Tensor,
    alpha_raw: &Tensor,
    beta_raw: &Tensor,
    a: &Tensor,
    dt_bias: &Tensor,
    gate: &Tensor,
    norm_weight: &Tensor,
    norm_eps: f32,
    fuse_gate: bool,
    n_k_heads: usize,
    n_v_heads: usize,
    key_dim: usize,
    value_dim: usize,
    kernel_size: usize,
) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
    let total_key_dim = n_k_heads * key_dim;
    let total_value_dim = n_v_heads * value_dim;
    let conv_dim = total_key_dim * 2 + total_value_dim;
    let packed = recurrent_state.apply_op3_no_bwd(
        qkv,
        conv_state,
        &FusedDecode {
            conv_weight,
            alpha_raw,
            beta_raw,
            a,
            dt_bias,
            gate,
            norm_weight,
            norm_eps,
            fuse_gate,
            n_k_heads,
            n_v_heads,
            key_dim,
            value_dim,
            kernel_size,
            specialized: fused_decode_128_supported(key_dim, value_dim),
        },
    )?;
    let state_elements = n_v_heads * key_dim * value_dim;
    let recurrent_state = packed
        .narrow(0, 0, state_elements)?
        .reshape((n_v_heads, key_dim, value_dim))?;
    let recurrent_output = packed
        .narrow(0, state_elements, total_value_dim)?
        .reshape((1, n_v_heads, value_dim))?;
    let conv_state = packed
        .narrow(0, state_elements + total_value_dim, conv_dim * kernel_size)?
        .reshape((conv_dim, kernel_size))?;
    Ok((recurrent_output, conv_state, recurrent_state))
}

#[allow(clippy::too_many_arguments)]
pub(super) struct DeltaNetReplayTerms {
    pub(super) keys: Tensor,
    pub(super) deltas: Tensor,
    pub(super) decays: Tensor,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prefill(
    qkv: &Tensor,
    conv_state: &Tensor,
    recurrent_state: &Tensor,
    conv_weight: &Tensor,
    alpha_raw: &Tensor,
    beta_raw: &Tensor,
    a: &Tensor,
    dt_bias: &Tensor,
    n_k_heads: usize,
    n_v_heads: usize,
    key_dim: usize,
    value_dim: usize,
    kernel_size: usize,
    record_replay: bool,
    gate: &Tensor,
    norm_weight: &Tensor,
    norm_eps: f32,
    fuse_gate: bool,
) -> anyhow::Result<(Tensor, Tensor, Tensor, Option<DeltaNetReplayTerms>, bool)> {
    let (seq_len, conv_dim) = qkv.dims2()?;
    anyhow::ensure!(seq_len > 0, "Qwen3.5 Metal scan requires a token");
    anyhow::ensure!(
        (1..=8).contains(&kernel_size),
        "Qwen3.5 conv kernel exceeds Metal limit"
    );
    anyhow::ensure!(n_k_heads > 0, "Qwen3.5 Metal scan requires key heads");
    anyhow::ensure!(
        n_v_heads > 0 && n_v_heads.is_multiple_of(n_k_heads),
        "Qwen3.5 Metal scan requires tiled value heads"
    );
    anyhow::ensure!(
        (1..=256).contains(&key_dim),
        "Qwen3.5 key head exceeds Metal limit"
    );
    anyhow::ensure!(
        (1..=256).contains(&value_dim),
        "Qwen3.5 value head exceeds Metal limit"
    );
    if seq_len == 1 {
        anyhow::ensure!(
            !record_replay,
            "Qwen3.5 replay recording needs a multi-token block"
        );
        let fuse_gate = fuse_gate && key_dim == 128 && value_dim == 128;
        let (output, conv_state, recurrent_state) = fused_decode(
            qkv,
            conv_state,
            recurrent_state,
            conv_weight,
            alpha_raw,
            beta_raw,
            a,
            dt_bias,
            gate,
            norm_weight,
            norm_eps,
            fuse_gate,
            n_k_heads,
            n_v_heads,
            key_dim,
            value_dim,
            kernel_size,
        )?;
        return Ok((output, conv_state, recurrent_state, None, fuse_gate));
    }
    let conv_packed = qkv.apply_op2_no_bwd(
        conv_state,
        &ConvScan {
            weight: conv_weight,
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

    let total_value_dim = n_v_heads * value_dim;
    let recurrent_packed = recurrent_state.apply_op3_no_bwd(
        &conv,
        alpha_raw,
        &DeltaNetPrefill {
            beta_raw,
            a,
            dt_bias,
            seq_len,
            n_k_heads,
            n_v_heads,
            key_dim,
            value_dim,
            record_replay,
            // Short scans do not amortize the on-chip tile's setup cost.
            onchip_state: seq_len >= 7,
        },
    )?;
    let recurrent_output_elements = seq_len * total_value_dim;
    let recurrent_output = recurrent_packed
        .narrow(0, 0, recurrent_output_elements)?
        .reshape((seq_len, n_v_heads, value_dim))?;
    let recurrent_state_elements = n_v_heads * key_dim * value_dim;
    let next_recurrent_state = recurrent_packed
        .narrow(0, recurrent_output_elements, recurrent_state_elements)?
        .reshape((n_v_heads, key_dim, value_dim))?;
    let replay = if record_replay {
        let key_elements = seq_len * n_v_heads * key_dim;
        let delta_elements = seq_len * n_v_heads * value_dim;
        let decay_elements = seq_len * n_v_heads;
        let keys = super::copy_persistent_tensor(
            &recurrent_packed
                .narrow(
                    0,
                    recurrent_output_elements + recurrent_state_elements,
                    key_elements,
                )?
                .reshape((seq_len, n_v_heads, key_dim))?,
        )?;
        let deltas = super::copy_persistent_tensor(
            &recurrent_packed
                .narrow(
                    0,
                    recurrent_output_elements + recurrent_state_elements + key_elements,
                    delta_elements,
                )?
                .reshape((seq_len, n_v_heads, value_dim))?,
        )?;
        let decays = super::copy_persistent_tensor(
            &recurrent_packed
                .narrow(
                    0,
                    recurrent_output_elements
                        + recurrent_state_elements
                        + key_elements
                        + delta_elements,
                    decay_elements,
                )?
                .reshape((seq_len, n_v_heads))?,
        )?;
        // Keep only replay terms. A narrow view would retain the scan output
        // and its full recurrent state after the current state is compacted.
        Some(DeltaNetReplayTerms {
            keys,
            deltas,
            decays,
        })
    } else {
        None
    };
    Ok((
        recurrent_output,
        next_conv_state,
        next_recurrent_state,
        replay,
        false,
    ))
}

struct DeltaNetReplay<'a> {
    decays: &'a Tensor,
    rows: usize,
    tape_rows: usize,
    n_heads: usize,
    key_dim: usize,
    value_dim: usize,
}

impl CustomOp3 for DeltaNetReplay<'_> {
    fn name(&self) -> &'static str {
        "qwen35-deltanet-replay"
    }

    fn cpu_fwd(
        &self,
        _state: &CpuStorage,
        _state_layout: &Layout,
        _keys: &CpuStorage,
        _keys_layout: &Layout,
        _deltas: &CpuStorage,
        _deltas_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("qwen35-deltanet-replay is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        state: &MetalStorage,
        state_layout: &Layout,
        keys: &MetalStorage,
        keys_layout: &Layout,
        deltas: &MetalStorage,
        deltas_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let state_elements = self.n_heads * self.key_dim * self.value_dim;
        for (name, storage, layout, expected) in [
            ("state", state, state_layout, state_elements),
            (
                "keys",
                keys,
                keys_layout,
                self.tape_rows * self.n_heads * self.key_dim,
            ),
            (
                "deltas",
                deltas,
                deltas_layout,
                self.tape_rows * self.n_heads * self.value_dim,
            ),
        ] {
            if storage.dtype() != DType::F32
                || !layout.is_contiguous()
                || layout.shape().elem_count() != expected
            {
                candle_core::bail!(
                    "{name} must be contiguous f32 with {expected} elements for Qwen3.5 replay"
                )
            }
            if storage.device().id() != state.device().id() {
                candle_core::bail!("Qwen3.5 replay tensors are on different devices")
            }
        }
        let (decays_guard, decays_layout) = self.decays.storage_and_layout();
        let decays = metal_storage("decays", &decays_guard, decays_layout)?;
        if decays.dtype() != DType::F32
            || !decays_layout.is_contiguous()
            || decays_layout.shape().elem_count() != self.tape_rows * self.n_heads
            || decays.device().id() != state.device().id()
        {
            candle_core::bail!("invalid Qwen3.5 replay decay tensor")
        }

        let device = state.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(state_elements, DType::F32)
            .with_label("qwen35-deltanet-replay")
            .build()?;
        let pipeline = named_pipeline(
            device,
            "qwen35_deltanet_replay_f32",
            &DELTANET_REPLAY_PIPELINE,
        )?;
        let encoder = device.command_encoder()?;
        encoder.set_label("qwen35-deltanet-replay");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        let offset = |layout: &Layout| layout.start_offset() * DType::F32.size_in_bytes();
        encoder_ref.set_input_buffer(0, Some(state.buffer()), offset(state_layout));
        encoder_ref.set_input_buffer(1, Some(keys.buffer()), offset(keys_layout));
        encoder_ref.set_input_buffer(2, Some(deltas.buffer()), offset(deltas_layout));
        encoder_ref.set_input_buffer(3, Some(decays.buffer()), offset(decays_layout));
        encoder_ref.set_output_buffer(4, Some(&output), 0);
        encoder_ref.set_bytes(5, &(self.rows as u32));
        encoder_ref.set_bytes(6, &(self.n_heads as u32));
        encoder_ref.set_bytes(7, &(self.key_dim as u32));
        encoder_ref.set_bytes(8, &(self.value_dim as u32));
        encoder_ref.dispatch_threads(
            objc2_metal::MTLSize {
                width: state_elements,
                height: 1,
                depth: 1,
            },
            candle_metal_kernels::utils::get_block_dims(state_elements.min(256), 1, 1),
        );
        Ok((
            MetalStorage::new(output, device.clone(), state_elements, DType::F32),
            Shape::from((self.n_heads, self.key_dim, self.value_dim)),
        ))
    }
}

pub(super) fn replay(
    state: &Tensor,
    keys: &Tensor,
    deltas: &Tensor,
    decays: &Tensor,
    rows: usize,
) -> anyhow::Result<Tensor> {
    let (tape_rows, n_heads, key_dim) = keys.dims3()?;
    let (delta_rows, delta_heads, value_dim) = deltas.dims3()?;
    anyhow::ensure!(
        state.dims3()? == (n_heads, key_dim, value_dim)
            && decays.dims2()? == (tape_rows, n_heads)
            && (delta_rows, delta_heads) == (tape_rows, n_heads),
        "Qwen3.5 replay tensor shapes differ"
    );
    anyhow::ensure!(
        (1..=tape_rows).contains(&rows),
        "Qwen3.5 replay row count is out of range"
    );
    state
        .apply_op3_no_bwd(
            keys,
            deltas,
            &DeltaNetReplay {
                decays,
                rows,
                tape_rows,
                n_heads,
                key_dim,
                value_dim,
            },
        )
        .map_err(Into::into)
}

struct DeltaNetStep<'a> {
    v: &'a Tensor,
    beta_raw: &'a Tensor,
    alpha_raw: &'a Tensor,
    a: &'a Tensor,
    dt_bias: &'a Tensor,
    n_heads: usize,
    key_dim: usize,
    value_dim: usize,
}

impl CustomOp3 for DeltaNetStep<'_> {
    fn name(&self) -> &'static str {
        "qwen35-deltanet-step"
    }

    fn cpu_fwd(
        &self,
        _state: &CpuStorage,
        _state_layout: &Layout,
        _q: &CpuStorage,
        _q_layout: &Layout,
        _k: &CpuStorage,
        _k_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("qwen35-deltanet-step is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        state: &MetalStorage,
        state_layout: &Layout,
        q: &MetalStorage,
        q_layout: &Layout,
        k: &MetalStorage,
        k_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        for (name, storage, layout) in [
            ("state", state, state_layout),
            ("q", q, q_layout),
            ("k", k, k_layout),
        ] {
            if storage.dtype() != DType::F32 || !layout.is_contiguous() {
                candle_core::bail!("{name} must be contiguous f32 for Qwen3.5 DeltaNet")
            }
        }

        let (v_guard, v_layout) = self.v.storage_and_layout();
        let (beta_guard, beta_layout) = self.beta_raw.storage_and_layout();
        let (alpha_guard, alpha_layout) = self.alpha_raw.storage_and_layout();
        let (a_guard, a_layout) = self.a.storage_and_layout();
        let (dt_guard, dt_layout) = self.dt_bias.storage_and_layout();
        let v = metal_storage("v", &v_guard, v_layout)?;
        let beta = metal_storage("beta_raw", &beta_guard, beta_layout)?;
        let alpha = metal_storage("alpha_raw", &alpha_guard, alpha_layout)?;
        let a = metal_storage("a", &a_guard, a_layout)?;
        let dt = metal_storage("dt_bias", &dt_guard, dt_layout)?;

        let state_elements = self.n_heads * self.key_dim * self.value_dim;
        let y_elements = self.n_heads * self.value_dim;
        let output_elements = state_elements + y_elements;
        let device = state.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, DType::F32)
            .with_label("qwen35-deltanet-step")
            .build()?;
        let pipeline = pipeline(device)?;
        let encoder = device.command_encoder()?;
        encoder.set_label("qwen35-deltanet-step");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        let offset = |layout: &Layout| layout.start_offset() * DType::F32.size_in_bytes();
        encoder_ref.set_input_buffer(0, Some(state.buffer()), offset(state_layout));
        encoder_ref.set_input_buffer(1, Some(q.buffer()), offset(q_layout));
        encoder_ref.set_input_buffer(2, Some(k.buffer()), offset(k_layout));
        encoder_ref.set_input_buffer(3, Some(v.buffer()), offset(v_layout));
        encoder_ref.set_input_buffer(4, Some(beta.buffer()), offset(beta_layout));
        encoder_ref.set_input_buffer(5, Some(alpha.buffer()), offset(alpha_layout));
        encoder_ref.set_input_buffer(6, Some(a.buffer()), offset(a_layout));
        encoder_ref.set_input_buffer(7, Some(dt.buffer()), offset(dt_layout));
        encoder_ref.set_output_buffer(8, Some(&output), 0);
        encoder_ref.set_bytes(9, &(self.n_heads as u32));
        encoder_ref.set_bytes(10, &(self.key_dim as u32));
        encoder_ref.set_bytes(11, &(self.value_dim as u32));
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

fn metal_storage<'a>(
    name: &str,
    storage: &'a Storage,
    layout: &Layout,
) -> Result<&'a MetalStorage> {
    let storage = match storage {
        Storage::Metal(storage) => storage,
        _ => candle_core::bail!("{name} is not on Metal"),
    };
    if storage.dtype() != DType::F32 || !layout.is_contiguous() {
        candle_core::bail!("{name} must be contiguous f32 for Qwen3.5 DeltaNet")
    }
    Ok(storage)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn step(
    state: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta_raw: &Tensor,
    alpha_raw: &Tensor,
    a: &Tensor,
    dt_bias: &Tensor,
) -> anyhow::Result<(Tensor, Tensor)> {
    let (n_heads, key_dim, value_dim) = state.dims3()?;
    let op = DeltaNetStep {
        v,
        beta_raw,
        alpha_raw,
        a,
        dt_bias,
        n_heads,
        key_dim,
        value_dim,
    };
    let packed = state.apply_op3_no_bwd(q, k, &op)?;
    let state_elements = n_heads * key_dim * value_dim;
    let y_elements = n_heads * value_dim;
    let next_state = packed
        .narrow(0, 0, state_elements)?
        .reshape((n_heads, key_dim, value_dim))?;
    let y = packed
        .narrow(0, state_elements, y_elements)?
        .reshape((n_heads, value_dim))?;
    Ok((y, next_state))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_pipeline_cache_is_scoped_to_device_id() -> anyhow::Result<()> {
        let Some(first) = crate::test_metal_device() else {
            return Ok(());
        };
        let second = Device::new_metal(0)?;
        let first_id = first.as_metal_device()?.id();
        let second_id = second.as_metal_device()?.id();
        assert_ne!(first_id, second_id);

        let cache = OnceLock::new();
        named_pipeline_from_source(
            first.as_metal_device()?,
            "qwen35_deltanet_step_f32",
            &cache,
            DELTANET_MSL,
        )?;
        named_pipeline_from_source(
            second.as_metal_device()?,
            "qwen35_deltanet_step_f32",
            &cache,
            DELTANET_MSL,
        )?;
        named_pipeline_from_source(
            first.as_metal_device()?,
            "qwen35_deltanet_step_f32",
            &cache,
            DELTANET_MSL,
        )?;

        let pipelines = cache.get().expect("cache initialized").lock().unwrap();
        assert_eq!(pipelines.len(), 2);
        assert!(pipelines.contains_key(&first_id));
        assert!(pipelines.contains_key(&second_id));
        drop(pipelines);

        let fallback = Mutex::new(HashSet::from([first_id]));
        let fallback_cache = OnceLock::new();
        assert!(
            onchip_prefill_pipeline_with_cache(
                first.as_metal_device()?,
                &fallback_cache,
                &fallback,
            )?
            .is_none()
        );
        assert!(!fallback.lock().unwrap().contains(&second_id));

        // A fallback caused by the first device must not suppress compilation
        // on the second device. If this device cannot compile the specialized
        // kernel, the helper records only that device's own failure.
        let second_result = onchip_prefill_pipeline_with_cache(
            second.as_metal_device()?,
            &fallback_cache,
            &fallback,
        )?;
        if second_result.is_some() {
            assert!(!fallback.lock().unwrap().contains(&second_id));
        } else {
            assert!(fallback.lock().unwrap().contains(&second_id));
        }
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

    #[test]
    fn metal_step_matches_scalar_reference() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (n_heads, key_dim, value_dim) = (2, 4, 3);
        let state_values: Vec<f32> = (0..n_heads * key_dim * value_dim)
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect();
        let q_values: Vec<f32> = vec![0.2, -0.1, 0.3, 0.4, -0.2, 0.5, 0.1, -0.3];
        let k_values: Vec<f32> = vec![0.4, 0.2, -0.3, 0.1, 0.3, -0.4, 0.2, 0.5];
        let v_values: Vec<f32> = vec![0.7, -0.2, 0.4, -0.1, 0.8, 0.3];
        let beta_values: Vec<f32> = vec![0.2, -0.3];
        let alpha_values: Vec<f32> = vec![0.1, 0.4];
        let a_values: Vec<f32> = vec![-0.7, -1.2];
        let dt_values: Vec<f32> = vec![0.05, -0.1];

        let state = Tensor::from_vec(state_values.clone(), (n_heads, key_dim, value_dim), &device)?;
        let q = Tensor::from_vec(q_values.clone(), (n_heads, key_dim), &device)?;
        let k = Tensor::from_vec(k_values.clone(), (n_heads, key_dim), &device)?;
        let v = Tensor::from_vec(v_values.clone(), (n_heads, value_dim), &device)?;
        let beta = Tensor::from_vec(beta_values.clone(), n_heads, &device)?;
        let alpha = Tensor::from_vec(alpha_values.clone(), n_heads, &device)?;
        let a = Tensor::from_vec(a_values.clone(), n_heads, &device)?;
        let dt = Tensor::from_vec(dt_values.clone(), n_heads, &device)?;

        let (actual_y, actual_state) = step(&state, &q, &k, &v, &beta, &alpha, &a, &dt)?;
        let actual_y = actual_y.flatten_all()?.to_vec1::<f32>()?;
        let actual_state = actual_state.flatten_all()?.to_vec1::<f32>()?;

        let mut expected_y = vec![0.0f32; n_heads * value_dim];
        let mut expected_state = vec![0.0f32; state_values.len()];
        for head in 0..n_heads {
            let beta = 1.0 / (1.0 + (-beta_values[head]).exp());
            let dt_input = alpha_values[head] + dt_values[head];
            let softplus = if dt_input <= 20.0 {
                dt_input.exp().ln_1p()
            } else {
                dt_input
            };
            let decay = (a_values[head] * softplus).exp();
            for value in 0..value_dim {
                let mut retrieved = 0.0;
                for d in 0..key_dim {
                    let state_index = (head * key_dim + d) * value_dim + value;
                    let k_value = k_values[head * key_dim + d];
                    retrieved += decay * state_values[state_index] * k_value;
                }
                let y_index = head * value_dim + value;
                let delta = beta * (v_values[y_index] - retrieved);
                for d in 0..key_dim {
                    let state_index = (head * key_dim + d) * value_dim + value;
                    expected_state[state_index] =
                        decay * state_values[state_index] + k_values[head * key_dim + d] * delta;
                    expected_y[y_index] +=
                        expected_state[state_index] * q_values[head * key_dim + d];
                }
            }
        }

        for (index, (actual, expected)) in actual_y.iter().zip(&expected_y).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-4,
                "output {index}: got {actual}, expected {expected}"
            );
        }
        for (index, (actual, expected)) in actual_state.iter().zip(&expected_state).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-4,
                "state {index}: got {actual}, expected {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn metal_prefill_matches_scalar_reference() -> anyhow::Result<()> {
        assert_metal_scan_matches_scalar_reference(3)
    }

    #[test]
    fn metal_fused_decode_matches_scalar_reference() -> anyhow::Result<()> {
        assert_metal_scan_matches_scalar_reference(1)
    }

    #[test]
    fn metal_fused_decode_gate_matches_existing_epilogue_at_model_geometry() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (n_k_heads, n_v_heads, key_dim, value_dim, kernel_size) =
            (16usize, 32usize, 128usize, 128usize, 4usize);
        let total_key_dim = n_k_heads * key_dim;
        let total_value_dim = n_v_heads * value_dim;
        let conv_dim = total_key_dim * 2 + total_value_dim;
        let values = |len: usize, modulus: usize, scale: f32| {
            (0..len)
                .map(|index| ((index % modulus) as f32 - (modulus / 2) as f32) * scale)
                .collect::<Vec<_>>()
        };

        let qkv = Tensor::from_vec(values(conv_dim, 31, 0.015), (1, conv_dim), &device)?;
        let conv_state = Tensor::from_vec(
            values(conv_dim * kernel_size, 17, 0.004),
            (conv_dim, kernel_size),
            &device,
        )?;
        let recurrent_state = Tensor::from_vec(
            values(n_v_heads * key_dim * value_dim, 29, 0.0005),
            (n_v_heads, key_dim, value_dim),
            &device,
        )?;
        let conv_weight = Tensor::from_vec(
            values(conv_dim * kernel_size, 11, 0.025),
            (conv_dim, kernel_size),
            &device,
        )?;
        let adversarial = [-30.0f32, -20.0, -1.0, 0.0, 1.0, 20.0, 30.0, 80.0];
        let alpha = Tensor::from_vec(
            (0..n_v_heads)
                .map(|head| adversarial[head % adversarial.len()])
                .collect::<Vec<_>>(),
            (1, n_v_heads),
            &device,
        )?;
        let beta = Tensor::from_vec(
            (0..n_v_heads)
                .map(|head| adversarial[(head + 3) % adversarial.len()])
                .collect::<Vec<_>>(),
            (1, n_v_heads),
            &device,
        )?;
        let a = Tensor::from_vec(
            (0..n_v_heads)
                .map(|head| -0.15 - (head % 7) as f32 * 0.08)
                .collect::<Vec<_>>(),
            n_v_heads,
            &device,
        )?;
        let dt_bias = Tensor::from_vec(values(n_v_heads, 9, 0.02), n_v_heads, &device)?;
        let gate_values = (0..total_value_dim)
            .map(|index| {
                if index.is_multiple_of(97) {
                    adversarial[(index / 97) % adversarial.len()]
                } else {
                    ((index % 37) as f32 - 18.0) * 0.08
                }
            })
            .collect::<Vec<_>>();
        let norm_values = (0..value_dim)
            .map(|column| 0.7 + (column % 13) as f32 * 0.025)
            .collect::<Vec<_>>();
        let mut padded_gate = vec![123.0f32; 5];
        padded_gate.extend_from_slice(&gate_values);
        let gate = Tensor::from_vec(padded_gate, total_value_dim + 5, &device)?
            .narrow(0, 5, total_value_dim)?
            .reshape((1, 1, total_value_dim))?;
        let mut padded_norm = vec![-123.0f32; 3];
        padded_norm.extend_from_slice(&norm_values);
        let norm_weight =
            Tensor::from_vec(padded_norm, value_dim + 3, &device)?.narrow(0, 3, value_dim)?;
        let eps = 1e-6f32;

        let (raw, raw_conv, raw_state, replay, raw_is_gated) = prefill(
            &qkv,
            &conv_state,
            &recurrent_state,
            &conv_weight,
            &alpha,
            &beta,
            &a,
            &dt_bias,
            n_k_heads,
            n_v_heads,
            key_dim,
            value_dim,
            kernel_size,
            false,
            &gate,
            &norm_weight,
            eps,
            false,
        )?;
        assert!(replay.is_none());
        assert!(!raw_is_gated);
        let expected = crate::architectures::fused_norm_metal::try_fused_head_rms_norm_silu_gate(
            &raw,
            &gate,
            &norm_weight,
            1,
            n_v_heads,
            value_dim,
            eps,
        )?
        .expect("model geometry must use the existing fused gate reference");

        // Keep only the outputs alive before synchronization. This exercises
        // the command buffer's ownership of every newly fused input buffer.
        let (actual, actual_conv, actual_state) = {
            let mut padded_gate = vec![321.0f32; 7];
            padded_gate.extend_from_slice(&gate_values);
            let actual_gate = Tensor::from_vec(padded_gate, total_value_dim + 7, &device)?
                .narrow(0, 7, total_value_dim)?
                .reshape((1, 1, total_value_dim))?;
            let mut padded_norm = vec![-321.0f32; 2];
            padded_norm.extend_from_slice(&norm_values);
            let actual_norm =
                Tensor::from_vec(padded_norm, value_dim + 2, &device)?.narrow(0, 2, value_dim)?;
            let (output, conv, state, replay, output_is_gated) = prefill(
                &qkv,
                &conv_state,
                &recurrent_state,
                &conv_weight,
                &alpha,
                &beta,
                &a,
                &dt_bias,
                n_k_heads,
                n_v_heads,
                key_dim,
                value_dim,
                kernel_size,
                false,
                &actual_gate,
                &actual_norm,
                eps,
                true,
            )?;
            assert!(replay.is_none());
            assert!(output_is_gated);
            (output, conv, state)
        };
        device.synchronize()?;

        assert_close(
            &actual.flatten_all()?.to_vec1::<f32>()?,
            &expected.flatten_all()?.to_vec1::<f32>()?,
            2e-6,
            "gated_output",
        );
        assert_close(
            &actual_conv.flatten_all()?.to_vec1::<f32>()?,
            &raw_conv.flatten_all()?.to_vec1::<f32>()?,
            0.0,
            "conv_state",
        );
        assert_close(
            &actual_state.flatten_all()?.to_vec1::<f32>()?,
            &raw_state.flatten_all()?.to_vec1::<f32>()?,
            0.0,
            "recurrent_state",
        );

        Ok(())
    }

    #[test]
    fn specialized_fused_decode_matches_generic_kernel_at_model_geometry() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (n_k_heads, n_v_heads, key_dim, value_dim, kernel_size) =
            (16usize, 48usize, 128usize, 128usize, 4usize);
        let total_key_dim = n_k_heads * key_dim;
        let total_value_dim = n_v_heads * value_dim;
        let conv_dim = total_key_dim * 2 + total_value_dim;
        let values = |len: usize, modulus: usize, scale: f32| {
            (0..len)
                .map(|index| ((index % modulus) as f32 - (modulus / 2) as f32) * scale)
                .collect::<Vec<_>>()
        };
        let qkv = Tensor::from_vec(values(conv_dim, 31, 0.015), conv_dim, &device)?;
        let conv_state = Tensor::from_vec(
            values(conv_dim * kernel_size, 17, 0.004),
            (conv_dim, kernel_size),
            &device,
        )?;
        let recurrent_state = Tensor::from_vec(
            values(n_v_heads * key_dim * value_dim, 29, 0.0005),
            (n_v_heads, key_dim, value_dim),
            &device,
        )?;
        let conv_weight = Tensor::from_vec(
            values(conv_dim * kernel_size, 11, 0.025),
            (conv_dim, kernel_size),
            &device,
        )?;
        let adversarial = [-30.0f32, -20.0, -1.0, 0.0, 1.0, 20.0, 30.0, 80.0];
        let per_head = |shift: usize| {
            (0..n_v_heads)
                .map(|head| adversarial[(head + shift) % adversarial.len()])
                .collect::<Vec<_>>()
        };
        let alpha = Tensor::from_vec(per_head(0), n_v_heads, &device)?;
        let beta = Tensor::from_vec(per_head(3), n_v_heads, &device)?;
        let a = Tensor::from_vec(
            (0..n_v_heads)
                .map(|head| -0.15 - (head % 7) as f32 * 0.08)
                .collect::<Vec<_>>(),
            n_v_heads,
            &device,
        )?;
        let dt_bias = Tensor::from_vec(values(n_v_heads, 9, 0.02), n_v_heads, &device)?;
        let gate = Tensor::from_vec(values(total_value_dim, 37, 0.08), total_value_dim, &device)?;
        let norm_weight = Tensor::from_vec(
            (0..value_dim)
                .map(|column| 0.7 + (column % 13) as f32 * 0.025)
                .collect::<Vec<_>>(),
            value_dim,
            &device,
        )?;

        for fuse_gate in [false, true] {
            let run = |specialized: bool| -> anyhow::Result<Vec<f32>> {
                Ok(recurrent_state
                    .apply_op3_no_bwd(
                        &qkv,
                        &conv_state,
                        &FusedDecode {
                            conv_weight: &conv_weight,
                            alpha_raw: &alpha,
                            beta_raw: &beta,
                            a: &a,
                            dt_bias: &dt_bias,
                            gate: &gate,
                            norm_weight: &norm_weight,
                            norm_eps: 1e-6,
                            fuse_gate,
                            n_k_heads,
                            n_v_heads,
                            key_dim,
                            value_dim,
                            kernel_size,
                            specialized,
                        },
                    )?
                    .to_vec1::<f32>()?)
            };
            let expected = run(false)?;
            let actual = run(true)?;
            let state_elements = n_v_heads * key_dim * value_dim;
            let (expected_state, expected_rest) = expected.split_at(state_elements);
            let (actual_state, actual_rest) = actual.split_at(state_elements);
            let (expected_y, expected_conv) = expected_rest.split_at(total_value_dim);
            let (actual_y, actual_conv) = actual_rest.split_at(total_value_dim);
            // Only the summation order of the two key-row dot products differs.
            assert_close(actual_state, expected_state, 1e-6, "recurrent_state");
            assert_close(actual_y, expected_y, 1e-5, "output");
            assert_close(actual_conv, expected_conv, 0.0, "conv_state");
        }
        Ok(())
    }

    fn assert_metal_scan_matches_scalar_reference(seq_len: usize) -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (n_k_heads, n_v_heads, key_dim, value_dim, kernel_size) = (2, 4, 4, 3, 3);
        let total_key_dim = n_k_heads * key_dim;
        let total_value_dim = n_v_heads * value_dim;
        let conv_dim = total_key_dim * 2 + total_value_dim;

        let qkv_values = (0..seq_len * conv_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.03)
            .collect::<Vec<_>>();
        let conv_state_values = (0..conv_dim * kernel_size)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
            .collect::<Vec<_>>();
        let conv_weight_values = (0..conv_dim * kernel_size)
            .map(|i| match i % kernel_size {
                0 => 0.2,
                1 => -0.1,
                _ => 0.3,
            })
            .collect::<Vec<_>>();
        let recurrent_state_values = (0..n_v_heads * key_dim * value_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.005)
            .collect::<Vec<_>>();
        let alpha_values = if seq_len == 1 {
            vec![-30.0, -20.0, 20.0, 30.0]
        } else {
            (0..seq_len * n_v_heads)
                .map(|i| ((i % 5) as f32 - 2.0) * 0.1)
                .collect::<Vec<_>>()
        };
        let beta_values = if seq_len == 1 {
            vec![-30.0, -1.0, 1.0, 30.0]
        } else {
            (0..seq_len * n_v_heads)
                .map(|i| ((i % 7) as f32 - 3.0) * 0.08)
                .collect::<Vec<_>>()
        };
        let gate_values = (0..seq_len * total_value_dim)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.04)
            .collect::<Vec<_>>();
        let norm_values = (0..value_dim)
            .map(|i| 0.8 + i as f32 * 0.03)
            .collect::<Vec<_>>();
        let a_values = vec![-0.7, -0.9, -1.1, -1.3];
        let dt_values = vec![0.05, -0.1, 0.15, 0.0];

        let qkv = Tensor::from_vec(qkv_values.clone(), (seq_len, conv_dim), &device)?;
        let conv_state =
            Tensor::from_vec(conv_state_values.clone(), (conv_dim, kernel_size), &device)?;
        let recurrent_state = Tensor::from_vec(
            recurrent_state_values.clone(),
            (n_v_heads, key_dim, value_dim),
            &device,
        )?;
        let conv_weight =
            Tensor::from_vec(conv_weight_values.clone(), (conv_dim, kernel_size), &device)?;
        let alpha = Tensor::from_vec(alpha_values.clone(), (seq_len, n_v_heads), &device)?;
        let beta = Tensor::from_vec(beta_values.clone(), (seq_len, n_v_heads), &device)?;
        let gate = Tensor::from_vec(gate_values, (1, seq_len, total_value_dim), &device)?;
        let norm_weight = Tensor::from_vec(norm_values, value_dim, &device)?;
        let a = Tensor::from_vec(a_values.clone(), n_v_heads, &device)?;
        let dt = Tensor::from_vec(dt_values.clone(), n_v_heads, &device)?;

        let record_replay = seq_len > 1;
        let (
            actual_output,
            actual_conv_state,
            actual_recurrent_state,
            replay_terms,
            output_is_gated,
        ) = prefill(
            &qkv,
            &conv_state,
            &recurrent_state,
            &conv_weight,
            &alpha,
            &beta,
            &a,
            &dt,
            n_k_heads,
            n_v_heads,
            key_dim,
            value_dim,
            kernel_size,
            record_replay,
            &gate,
            &norm_weight,
            1e-6,
            false,
        )?;
        assert!(!output_is_gated);
        if let Some(replay_terms) = replay_terms.as_ref() {
            assert_eq!(replay_terms.keys.dims(), &[seq_len, n_v_heads, key_dim]);
            assert_eq!(replay_terms.deltas.dims(), &[seq_len, n_v_heads, value_dim]);
            assert_eq!(replay_terms.decays.dims(), &[seq_len, n_v_heads]);
            for tensor in [
                &replay_terms.keys,
                &replay_terms.deltas,
                &replay_terms.decays,
            ] {
                let (storage, layout) = tensor.storage_and_layout();
                let Storage::Metal(storage) = &*storage else {
                    panic!("replay terms must use Metal");
                };
                assert_eq!(layout.start_offset(), 0);
                assert_eq!(
                    storage.buffer().length(),
                    tensor.elem_count() * DType::F32.size_in_bytes()
                );
            }
        } else {
            assert!(!record_replay);
        }

        let mut histories = conv_state_values
            .chunks_exact(kernel_size)
            .map(<[f32]>::to_vec)
            .collect::<Vec<_>>();
        let mut conv_values = vec![0.0f32; seq_len * conv_dim];
        for token in 0..seq_len {
            for channel in 0..conv_dim {
                histories[channel].rotate_left(1);
                histories[channel][kernel_size - 1] = qkv_values[token * conv_dim + channel];
                let value = histories[channel]
                    .iter()
                    .zip(&conv_weight_values[channel * kernel_size..(channel + 1) * kernel_size])
                    .map(|(value, weight)| value * weight)
                    .sum::<f32>();
                conv_values[token * conv_dim + channel] = value / (1.0 + (-value).exp());
            }
        }

        let mut expected_state = recurrent_state_values;
        let mut expected_output = vec![0.0f32; seq_len * total_value_dim];
        let mut expected_snapshots = Vec::with_capacity(seq_len * expected_state.len());
        for token in 0..seq_len {
            for head in 0..n_v_heads {
                let qk_head = head % n_k_heads;
                let q = (0..key_dim)
                    .map(|column| conv_values[token * conv_dim + qk_head * key_dim + column])
                    .collect::<Vec<_>>();
                let k = (0..key_dim)
                    .map(|column| {
                        conv_values[token * conv_dim + total_key_dim + qk_head * key_dim + column]
                    })
                    .collect::<Vec<_>>();
                let q_scale = 1.0
                    / ((q.iter().map(|value| value * value).sum::<f32>() + 1e-12).sqrt()
                        * (key_dim as f32).sqrt());
                let k_scale =
                    1.0 / (k.iter().map(|value| value * value).sum::<f32>() + 1e-12).sqrt();
                let gate_index = token * n_v_heads + head;
                let beta = 1.0 / (1.0 + (-beta_values[gate_index]).exp());
                let dt_input = alpha_values[gate_index] + dt_values[head];
                let softplus = if dt_input <= 20.0 {
                    dt_input.exp().ln_1p()
                } else {
                    dt_input
                };
                let decay = (a_values[head] * softplus).exp();

                for value in 0..value_dim {
                    let mut retrieved = 0.0f32;
                    for (column, k_value) in k.iter().enumerate() {
                        let state_index = (head * key_dim + column) * value_dim + value;
                        retrieved += decay * expected_state[state_index] * k_value * k_scale;
                    }
                    let v = conv_values
                        [token * conv_dim + 2 * total_key_dim + head * value_dim + value];
                    let delta = beta * (v - retrieved);
                    let mut y = 0.0f32;
                    for (column, (k_value, q_value)) in k.iter().zip(&q).enumerate() {
                        let state_index = (head * key_dim + column) * value_dim + value;
                        let next = decay * expected_state[state_index] + k_value * k_scale * delta;
                        expected_state[state_index] = next;
                        y += next * q_value * q_scale;
                    }
                    expected_output[token * total_value_dim + head * value_dim + value] = y;
                }
            }
            expected_snapshots.extend_from_slice(&expected_state);
        }
        let expected_conv_state = histories.into_iter().flatten().collect::<Vec<_>>();

        assert_close(
            &actual_output.flatten_all()?.to_vec1::<f32>()?,
            &expected_output,
            2e-5,
            "output",
        );
        assert_close(
            &actual_conv_state.flatten_all()?.to_vec1::<f32>()?,
            &expected_conv_state,
            2e-6,
            "conv_state",
        );
        assert_close(
            &actual_recurrent_state.flatten_all()?.to_vec1::<f32>()?,
            &expected_state,
            2e-5,
            "recurrent_state",
        );
        if let Some(replay_terms) = replay_terms {
            for rows in 1..=seq_len {
                let replayed = replay(
                    &recurrent_state,
                    &replay_terms.keys,
                    &replay_terms.deltas,
                    &replay_terms.decays,
                    rows,
                )?;
                let start = (rows - 1) * expected_state.len();
                assert_close(
                    &replayed.flatten_all()?.to_vec1::<f32>()?,
                    &expected_snapshots[start..start + expected_state.len()],
                    2e-5,
                    "recurrent_replay",
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod onchip_tests;
