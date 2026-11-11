// Adapted from llama.cpp 7aca0d2ea8d92b56911233b822db01f9fbb1ae50:
// https://github.com/ggml-org/llama.cpp/blob/7aca0d2ea8d92b56911233b822db01f9fbb1ae50/ggml/src/ggml-metal/kernels/mul_mv_mma.metal
// kernel_mul_mv_mma_gen, lane mapping and mul_mv_mma_store (NT4/RT2/NSG4).
// The Q8 decoder below is dedicated F32, NOT upstream's half scratch decoder.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

inline ushort dinf_q8_mma_fm(ushort lane) {
    return ((lane / 4) & 4) + ((lane / 2) % 4);
}
inline ushort dinf_q8_mma_fn(ushort lane) {
    return ((lane / 4) & 2) * 2 + (lane % 2) * 2;
}

// ci identifies 16 consecutive weights, including a possible final K=32 tail.
// Finite half scale * signed int8 is exactly representable in F32: no half
// product, half matrix, half scratch or full-weight decoded carrier exists.
inline void dinf_q8_mma_decode_f32(device const uchar * row, uint ci,
                                    uint k_dim, thread float4x4 & w) {
    if (16u * ci >= k_dim) {
        w = float4x4(0.0f);
        return;
    }
    device const uchar * block = row + ulong(ci / 2u) * 34u;
    const ushort bits = ushort(block[0]) | (ushort(block[1]) << 8);
    const float scale = float(as_type<half>(bits));
    device const char * q = reinterpret_cast<device const char *>(block + 2)
        + (ci % 2u) * 16u;
    for (ushort c = 0; c < 4; ++c)
        for (ushort r = 0; r < 4; ++r)
            w[c][r] = scale * float(q[4 * c + r]);
}

kernel void dinf_dsv41_q8_f32_mma_nt4_rt2_nsg4(
    device const uchar * weights [[buffer(0)]],
    device const float * input [[buffer(1)]],
    device float * output [[buffer(2)]],
    constant uint & k_dim [[buffer(3)]],
    constant uint & m_dim [[buffer(4)]],
    constant uint & tokens [[buffer(5)]],
    constant uint & row_bytes [[buffer(6)]],
    uint3 tile [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    // Exactly 4 * 2 * 4 * 64 * sizeof(float) = 8192 transient TG bytes.
    threadgroup float reduction[4 * 2 * 4 * 64];
    const ushort fm = dinf_q8_mma_fm(lane), fn = dinf_q8_mma_fn(lane);
    const uint m0 = tile.x * 32u, t0 = tile.y * 16u;
    simdgroup_float8x8 acc[2][4];
    for (ushort rt = 0; rt < 2; ++rt)
        for (ushort t = 0; t < 4; ++t)
            acc[rt][t] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (uint g = sg; g < (k_dim + 63u) / 64u; g += 4u) {
        simdgroup_float8x8 b[2][8];
        for (ushort rt = 0; rt < 2; ++rt) {
            const uint token0 = t0 + 8u * rt + fn;
            const uint k0 = 64u * g + 16u * (fm / 2u) + 8u * (fm % 2u);
            for (ushort s = 0; s < 8; ++s) {
                // Scalar loads deliberately support any F32-aligned input view;
                // reinterpreting as float4 would require 16-byte alignment.
                b[rt][s].thread_elements()[0] = token0 < tokens && k0 + s < k_dim
                    ? input[token0 * k_dim + k0 + s] : 0.0f;
                b[rt][s].thread_elements()[1] = token0 + 1u < tokens && k0 + s < k_dim
                    ? input[(token0 + 1u) * k_dim + k0 + s] : 0.0f;
            }
        }
        const uint ci = 4u * g + fn / 2u;
        for (ushort t = 0; t < 4; ++t) {
            const uint row = m0 + 8u * t + fm;
            float4x4 w;
            if (row < m_dim)
                dinf_q8_mma_decode_f32(weights + ulong(row) * row_bytes, ci, k_dim, w);
            else
                w = float4x4(0.0f);
            for (ushort s = 0; s < 8; ++s) {
                simdgroup_float8x8 a;
                a.thread_elements()[0] = w[s / 4][s % 4];
                a.thread_elements()[1] = w[s / 4 + 2][s % 4];
                for (ushort rt = 0; rt < 2; ++rt)
                    simdgroup_multiply_accumulate(acc[rt][t], a, b[rt][s], acc[rt][t]);
            }
        }
    }
    for (ushort rt = 0; rt < 2; ++rt)
        for (ushort t = 0; t < 4; ++t) {
            const uint i = ((sg * 2u + rt) * 4u + t) * 64u + 2u * lane;
            reduction[i] = acc[rt][t].thread_elements()[0];
            reduction[i + 1u] = acc[rt][t].thread_elements()[1];
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sums[4];
    bool bad = false;
    for (ushort j = 0; j < 4; ++j) {
        const uint i = sg * 32u + lane + 128u * j;
        float sum = 0.0f;
        for (ushort s = 0; s < 4; ++s) sum += reduction[s * 512u + i];
        sums[j] = sum;
        const ushort rt = i / 256u, t = (i / 64u) % 4u;
        const ushort l = (i % 64u) / 2u, e = i % 2u;
        const uint row = m0 + 8u * t + dinf_q8_mma_fm(l);
        const uint token = t0 + 8u * rt + dinf_q8_mma_fn(l) + e;
        bad |= row < m_dim && token < tokens
            && (as_type<uint>(sum) & 0x7f800000u) == 0x7f800000u;
    }
    // Read every partial before reusing the same 8KiB allocation for flags.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool group_bad = simd_any(bad);
    if (lane == 0) reduction[sg] = group_bad ? 1.0f : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool tile_bad = reduction[0] + reduction[1] + reduction[2] + reduction[3] != 0.0f;
    if (tile_bad) {
        // Convergent before-store redo: original NSG4 sum8(q*x), then F32
        // scale, then SIMD and cross-SIMD reductions. No host scan/fence.
        for (uint j = 0; j < 512u; ++j) {
            const uint row = m0 + j % 32u, token = t0 + j / 32u;
            float sum = 0.0f;
            if (row < m_dim && token < tokens) {
                for (uint ib = uint(sg) * 8u + lane / 4u; ib < k_dim / 32u; ib += 32u) {
                    device const uchar * block = weights + ulong(row) * row_bytes + ib * 34u;
                    const ushort bits = ushort(block[0]) | (ushort(block[1]) << 8);
                    const float scale = float(as_type<half>(bits));
                    device const char * q = reinterpret_cast<device const char *>(block + 2) + (lane % 4u) * 8u;
                    float sumq = 0.0f;
                    for (ushort v = 0; v < 8; ++v)
                        sumq += float(q[v]) * input[token * k_dim + ib * 32u + (lane % 4u) * 8u + v];
                    sum += sumq * scale;
                }
            }
            sum = simd_sum(sum);
            // All readers of the flags complete before their storage is reused.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) reduction[sg] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float total = simd_sum(lane < 4 ? reduction[lane] : 0.0f);
            if (sg == 0 && lane == 0 && row < m_dim && token < tokens)
                output[token * m_dim + row] = total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (ushort j = 0; j < 4; ++j) {
            const uint i = sg * 32u + lane + 128u * j;
            const ushort rt = i / 256u, t = (i / 64u) % 4u;
            const ushort l = (i % 64u) / 2u, e = i % 2u;
            const uint row = m0 + 8u * t + dinf_q8_mma_fm(l);
            const uint token = t0 + 8u * rt + dinf_q8_mma_fn(l) + e;
            if (row < m_dim && token < tokens) output[token * m_dim + row] = sums[j];
        }
    }
}
