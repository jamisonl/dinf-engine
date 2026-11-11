// Indexed/scattered wrapper around the qualified IQ1 reuse arithmetic.
// Each y group is ONE weight matrix, never an expert triplet.
struct dinf_packet_rows { uint source[4]; uint destination[4]; };
#define DINF_PACKET_WEIGHT(I) device const uchar *w##I [[buffer(4 + I)]]
template<uint N>
kernel void dinf_iq1_packet(
        device const float *input [[buffer(0)]],
        device float *output [[buffer(1)]],
        constant dinf_packet_rows *packets [[buffer(2)]],
        constant uint3 &geometry [[buffer(3)]],
        DINF_PACKET_WEIGHT(0), DINF_PACKET_WEIGHT(1), DINF_PACKET_WEIGHT(2),
        DINF_PACKET_WEIGHT(3), DINF_PACKET_WEIGHT(4), DINF_PACKET_WEIGHT(5),
        DINF_PACKET_WEIGHT(6), DINF_PACKET_WEIGHT(7), DINF_PACKET_WEIGHT(8),
        DINF_PACKET_WEIGHT(9), DINF_PACKET_WEIGHT(10), DINF_PACKET_WEIGHT(11),
        uint2 group [[threadgroup_position_in_grid]],
        ushort simd_group [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    device const uchar *weights = w0;
    switch (group.y) {
        case 1: weights = w1; break; case 2: weights = w2; break;
        case 3: weights = w3; break; case 4: weights = w4; break;
        case 5: weights = w5; break; case 6: weights = w6; break;
        case 7: weights = w7; break; case 8: weights = w8; break;
        case 9: weights = w9; break; case 10: weights = w10; break;
        case 11: weights = w11; break;
    }
    const uint k_dim = geometry.x, out_dim = geometry.y, weight_row_bytes = geometry.z;
    const uint row = group.x * 4u + uint(simd_group);
    if (row >= out_dim) return;
    const dinf_packet_rows indices = packets[group.y];
    device const float *input0 = input + size_t(indices.source[0]) * k_dim;
    device const float *input1 = input + size_t(indices.source[1]) * k_dim;
    device const float *input2 = input + size_t(indices.source[2]) * k_dim;
    device const float *input3 = input + size_t(indices.source[3]) * k_dim;
    device const block_iq1_s *weight_row =
        (device const block_iq1_s *)(weights + size_t(row) * weight_row_bytes);
    const uint blocks = k_dim / (16u * 16u);
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    for (uint chunk = uint(lane); chunk < blocks * 16u; chunk += 32u) {
        const uint block_index = chunk / 16u;
        const short il = short(chunk % 16u);
        dsv4_decode_scratch<block_iq1_s>::type values;
        dequantize_iq1_s(weight_row + block_index, il, values);
        const uint base = block_index * 16u * 16u + uint(il) * 16u;
        for (uint i = 0; i < 16; ++i) {
            const float weight = float(values[i / 4][i % 4]);
            sum0 += weight * input0[base + i];
            if (N >= 2) sum1 += weight * input1[base + i];
            if (N >= 3) sum2 += weight * input2[base + i];
            if (N >= 4) sum3 += weight * input3[base + i];
        }
    }
    sum0 = simd_sum(sum0);
    if (N >= 2) sum1 = simd_sum(sum1);
    if (N >= 3) sum2 = simd_sum(sum2);
    if (N >= 4) sum3 = simd_sum(sum3);
    if (lane == 0) {
        output[size_t(indices.destination[0]) * out_dim + row] = sum0;
        if (N >= 2) output[size_t(indices.destination[1]) * out_dim + row] = sum1;
        if (N >= 3) output[size_t(indices.destination[2]) * out_dim + row] = sum2;
        if (N >= 4) output[size_t(indices.destination[3]) * out_dim + row] = sum3;
    }
}
#undef DINF_PACKET_WEIGHT
typedef decltype(dinf_iq1_packet<1>) dinf_iq1_packet_fn;
template [[host_name("dinf_iq1_packet_n1")]] kernel dinf_iq1_packet_fn dinf_iq1_packet<1>;
template [[host_name("dinf_iq1_packet_n2")]] kernel dinf_iq1_packet_fn dinf_iq1_packet<2>;
template [[host_name("dinf_iq1_packet_n3")]] kernel dinf_iq1_packet_fn dinf_iq1_packet<3>;
template [[host_name("dinf_iq1_packet_n4")]] kernel dinf_iq1_packet_fn dinf_iq1_packet<4>;

// Adapted from ggml-org/llama.cpp ggml/src/ggml-metal/kernels/mul_mv_mma.metal,
// kernel_mul_mv_mma_gen + lane mapping/store, NT=4 RT=1 NSG=4.
// Revision: 7aca0d2ea8d92b56911233b822db01f9fbb1ae50.
// https://github.com/ggml-org/llama.cpp/blob/7aca0d2ea8d92b56911233b822db01f9fbb1ae50/ggml/src/ggml-metal/kernels/mul_mv_mma.metal
// MIT License, Copyright (c) 2023-2026 The ggml authors.
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
//
// Indexed adaptation: 32 output rows x 8 padded input columns. All dequantized
// weights, B fragments and accumulators are F32; never a half intermediate.
// Four SIMDgroups stripe 64-wide K chunks. Exactly 4096 dynamic TG bytes hold
// 4 * 4 * 64 F32 partials, then four flags after all partial reads finish.
inline uint dinf_packet_fm(uint l) { return ((l / 4u) & 4u) + ((l / 2u) % 4u); }
inline uint dinf_packet_fn(uint l) { return ((l / 4u) & 2u) * 2u + (l % 2u) * 2u; }
#define DINF_PACKET_WEIGHT(I) device const uchar *w##I [[buffer(4 + I)]]
template<uint N>
kernel void dinf_iq1_packet_mma(
        device const float *input [[buffer(0)]],
        device float *output [[buffer(1)]],
        constant dinf_packet_rows *packets [[buffer(2)]],
        constant uint3 &geometry [[buffer(3)]],
        DINF_PACKET_WEIGHT(0), DINF_PACKET_WEIGHT(1), DINF_PACKET_WEIGHT(2),
        DINF_PACKET_WEIGHT(3), DINF_PACKET_WEIGHT(4), DINF_PACKET_WEIGHT(5),
        DINF_PACKET_WEIGHT(6), DINF_PACKET_WEIGHT(7), DINF_PACKET_WEIGHT(8),
        DINF_PACKET_WEIGHT(9), DINF_PACKET_WEIGHT(10), DINF_PACKET_WEIGHT(11),
        threadgroup float *red [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort simd_group [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    device const uchar *weights = w0;
    switch (group.y) {
        case 1: weights = w1; break; case 2: weights = w2; break;
        case 3: weights = w3; break; case 4: weights = w4; break;
        case 5: weights = w5; break; case 6: weights = w6; break;
        case 7: weights = w7; break; case 8: weights = w8; break;
        case 9: weights = w9; break; case 10: weights = w10; break;
        case 11: weights = w11; break;
    }
    const uint k_dim = geometry.x, out_dim = geometry.y, row_bytes = geometry.z;
    const uint fm = dinf_packet_fm(lane), fn = dinf_packet_fn(lane);
    const uint first_row = group.x * 32u;
    const dinf_packet_rows indices = packets[group.y];
    device const block_iq1_s *x[4];
    for (uint t = 0; t < 4; ++t) {
        const uint row = min(first_row + 8u * t + fm, out_dim - 1u);
        x[t] = (device const block_iq1_s *)(weights + size_t(row) * row_bytes);
    }
    simdgroup_float8x8 mc[4];
    for (uint t = 0; t < 4; ++t) mc[t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint g = simd_group; g < k_dim / 64u; g += 4u) {
        simdgroup_float8x8 mb[8];
        const uint base = 64u * g + 16u * (fm / 2u) + 8u * (fm % 2u);
        // Scalar loads also support non-vector-aligned F32 input offsets.
        for (uint s = 0; s < 8; ++s) {
            mb[s].thread_elements()[0] = input[size_t(indices.source[min(fn, N - 1u)]) * k_dim + base + s];
            mb[s].thread_elements()[1] = input[size_t(indices.source[min(fn + 1u, N - 1u)]) * k_dim + base + s];
        }
        const uint ci = 4u * g + fn / 2u;
        for (uint t = 0; t < 4; ++t) {
            float4x4 w;
            dequantize_iq1_s(x[t] + ci / 16u, short(ci % 16u), w);
            for (uint s = 0; s < 8; ++s) {
                simdgroup_float8x8 ma;
                ma.thread_elements()[0] = w[s / 4u][s % 4u];
                ma.thread_elements()[1] = w[s / 4u + 2u][s % 4u];
                simdgroup_multiply_accumulate(mc[t], ma, mb[s], mc[t]);
            }
        }
    }
    for (uint t = 0; t < 4; ++t) {
        red[(uint(simd_group) * 4u + t) * 64u + 2u * lane] = mc[t].thread_elements()[0];
        red[(uint(simd_group) * 4u + t) * 64u + 2u * lane + 1u] = mc[t].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sums[2];
    bool bad = false;
    const uint tid = uint(simd_group) * 32u + lane;
    for (uint e = 0; e < 2; ++e) {
        const uint idx = tid + 128u * e;
        float sum = 0.0f;
        for (uint sg = 0; sg < 4; ++sg) sum += red[sg * 256u + idx];
        sums[e] = sum;
        const uint l = (idx % 64u) / 2u;
        const uint row = first_row + 8u * (idx / 64u) + dinf_packet_fm(l);
        const uint col = dinf_packet_fn(l) + idx % 2u;
        // Candidate compilation explicitly uses Safe math (FastMath=false).
        bad |= row < out_dim && col < N && (as_type<uint>(sum) & 0x7f800000u) == 0x7f800000u;
    }
    // No thread overwrites reduction data until every thread has read it.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool sg_bad = simd_any(bad);
    if (lane == 0) red[simd_group] = sg_bad ? 1.0f : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool tile_bad = red[0] != 0.0f || red[1] != 0.0f || red[2] != 0.0f || red[3] != 0.0f;
    if (tile_bad) {
        // Uniform tile decision, before ANY canonical store. Recompute with the
        // legacy lane/chunk order, including finite scalar cancellation cases.
        for (uint r = simd_group; r < 32u; r += 4u) {
            const uint row = first_row + r;
            if (row >= out_dim) continue; // SIMDgroup-uniform, no further TG barriers.
            device const block_iq1_s *wr = (device const block_iq1_s *)(weights + size_t(row) * row_bytes);
            float scalar[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint chunk = lane; chunk < k_dim / 16u; chunk += 32u) {
                float4x4 w;
                dequantize_iq1_s(wr + chunk / 16u, short(chunk % 16u), w);
                for (uint i = 0; i < 16; ++i) {
                    const float weight = w[i / 4u][i % 4u];
                    for (uint c = 0; c < N; ++c)
                        scalar[c] += weight * input[size_t(indices.source[c]) * k_dim + chunk * 16u + i];
                }
            }
            for (uint c = 0; c < N; ++c) {
                const float sum = simd_sum(scalar[c]);
                if (lane == 0) output[size_t(indices.destination[c]) * out_dim + row] = sum;
            }
        }
    } else {
        for (uint e = 0; e < 2; ++e) {
            const uint idx = tid + 128u * e, l = (idx % 64u) / 2u;
            const uint row = first_row + 8u * (idx / 64u) + dinf_packet_fm(l);
            const uint col = dinf_packet_fn(l) + idx % 2u;
            if (row < out_dim && col < N)
                output[size_t(indices.destination[col]) * out_dim + row] = sums[e];
        }
    }
}
#undef DINF_PACKET_WEIGHT
typedef decltype(dinf_iq1_packet_mma<2>) dinf_iq1_packet_mma_fn;
template [[host_name("dinf_iq1_packet_mma_n2")]] kernel dinf_iq1_packet_mma_fn dinf_iq1_packet_mma<2>;
template [[host_name("dinf_iq1_packet_mma_n3")]] kernel dinf_iq1_packet_mma_fn dinf_iq1_packet_mma<3>;
template [[host_name("dinf_iq1_packet_mma_n4")]] kernel dinf_iq1_packet_mma_fn dinf_iq1_packet_mma<4>;
