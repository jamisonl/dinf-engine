#include <metal_stdlib>
using namespace metal;

// Match Candle's contiguous F32 reduction tree: strided per-thread sums,
// shared-memory halves down to 64 entries, then a 32-lane SIMD sum.
inline float row_sum(float value, threadgroup float *shared, uint tid, uint threads) {
    shared[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride >= 64; stride /= 2) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 32) {
        if (threads >= 64) value = shared[tid] + shared[tid + 32];
        value = simd_sum(value);
        if (tid == 0) shared[0] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return shared[0];
}

template<typename T>
kernel void norm_scale(
    device const T *input [[buffer(0)]],
    device const T *scale [[buffer(1)]],
    device T *output [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup float shared[1024];
    const uint start = row * width;
    float sum = 0.0f;
    for (uint col = tid; col < width; col += threads) sum += float(input[start + col]);
    const float mean = row_sum(sum, shared, tid, threads) * (1.0f / float(width));
    // All threads must have read the shared result before reusing its memory.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = 0.0f;
    for (uint col = tid; col < width; col += threads) {
        const float centered = float(input[start + col]) - mean;
        volatile float square = centered * centered;
        sum += square;
    }
    volatile float variance = row_sum(sum, shared, tid, threads) * (1.0f / float(width));
    const float denom = fast::sqrt(variance + eps);
    for (uint col = tid; col < width; col += threads) {
        // Preserve the original BF16 boundary before the modulation multiply.
        const T normalized = T(fast::divide(float(input[start + col]) - mean, denom));
        output[start + col] = T(normalized * scale[start + col]);
    }
}

template<typename T>
kernel void gated_residual(
    device const T *hidden [[buffer(0)]],
    device const T *gate [[buffer(1)]],
    device const T *update [[buffer(2)]],
    device T *output [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    // Two rounding points, as in the unfused multiply then add; never an FMA.
    volatile T product = T(gate[i] * update[i]);
    output[i] = T(hidden[i] + product);
}

template<typename T>
kernel void rope(
    device const T *input [[buffer(0)]],
    device const float *cosine [[buffer(1)]],
    device const float *sine [[buffer(2)]],
    device T *output [[buffer(3)]],
    constant uint &pairs [[buffer(4)]],
    constant uint &sequence [[buffer(5)]],
    constant uint &half_dim [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= pairs) return;
    const uint pos = i % (sequence * half_dim);
    const float real = float(input[2 * i]), imag = float(input[2 * i + 1]);
    const float c = cosine[pos], s = sine[pos];
    volatile float rc = real * c, is = imag * s, rs = real * s, ic = imag * c;
    output[2 * i] = T(rc - is);
    output[2 * i + 1] = T(rs + ic);
}

// The native BF16 path casts to F32 complex, multiplies, then rounds to BF16.
// Its observed contraction order rounds the second product before each FMA.
// Explicit fma keeps that order with the library's fast-math setting disabled.
template<typename T>
kernel void rope_native(
    device const T *input [[buffer(0)]],
    device const float *cosine [[buffer(1)]],
    device const float *sine [[buffer(2)]],
    device T *output [[buffer(3)]],
    constant uint &pairs [[buffer(4)]],
    constant uint &sequence [[buffer(5)]],
    constant uint &half_dim [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= pairs) return;
    const uint pos = i % (sequence * half_dim);
    const float real = float(input[2 * i]), imag = float(input[2 * i + 1]);
    const float c = cosine[pos], s = sine[pos];
    volatile float is = imag * s, ic = imag * c;
    output[2 * i] = T(fma(real, c, -is));
    output[2 * i + 1] = T(fma(real, s, ic));
}


// Adapted from PyTorch d0f2af8b66b20da4baa50644c0cd3f887a2eae2e
// LayerNorm.metal and c10/metal/reduction_utils.h. Copyright PyTorch contributors.
// Host dispatch: Copyright (c) 2022 Apple Inc.
// BSD-3-Clause; see THIRD_PARTY_LICENSES/PYTORCH-BSD-3-CLAUSE.txt
// and THIRD_PARTY_LICENSES/PYTORCH-image-norm-SOURCE.json.
inline float native_norm_sum(threadgroup float* data, float value, uint tid, uint threads) {
    const float sum = simd_sum(value);
    if (tid % 32 == 0) data[tid / 32] = sum;
    if (threads > 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < (threads + 31) / 32) {
            const float total = simd_sum(data[tid]);
            if (tid == 0) data[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return data[0];
}

template<typename T>
kernel void norm_scale_native(
    device const T* input [[buffer(0)]], device const T* scale [[buffer(1)]],
    device T* output [[buffer(2)]], constant uint& width [[buffer(3)]], constant float& eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    const uint start = row * width, base = tid * 4;
    const int count = min(4, max(0, int(width) - int(base)));
    float values[4];
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        values[i] = i < count ? float(input[start + base + i]) : 0.0f;
        sum += values[i];
    }
    threadgroup float mean_shared[32];
    threadgroup float variance_shared[32];
    const float mean = native_norm_sum(mean_shared, sum, tid, threads) / float(width);
    sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        const float centered = i < count ? values[i] - mean : 0.0f;
        sum = fma(centered, centered, sum);
    }
    const float variance = native_norm_sum(variance_shared, sum, tid, threads) / float(width);
    const float inverse = precise::rsqrt(variance + eps);
    for (uint i = 0; i < 4; i++) {
        const uint column = base + i;
        if (column < width) {
            // Native LayerNorm rounds to BF16 before the separate modulation.
            const T normalized = T((values[i] - mean) * inverse);
            output[start + column] = T(normalized * scale[start + column]);
        }
    }
}

#define instantiate(name, func, type) template [[host_name(name)]] [[kernel]] decltype(func<type>) func<type>;
instantiate("norm_scale_f32", norm_scale, float)
instantiate("gated_residual_f32", gated_residual, float)
instantiate("rope_f32", rope, float)
#if defined(__HAVE_BFLOAT__)
instantiate("norm_scale_bf16", norm_scale, bfloat)
instantiate("norm_scale_native_bf16", norm_scale_native, bfloat)
instantiate("gated_residual_bf16", gated_residual, bfloat)
instantiate("rope_bf16", rope, bfloat)
instantiate("rope_native_bf16", rope_native, bfloat)
#endif
