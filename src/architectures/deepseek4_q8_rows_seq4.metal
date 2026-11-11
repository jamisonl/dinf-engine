// Exact-F32 attention-prefill Q8 rows_seq4 port, audreyt/ds4
// 0dff39f91721bcf3472813c6d2bebf9353df2a1a, metal/dense.metal:505-587.
// Attribution and MIT license: deepseek41/NOTICE and LICENSE-MIT.
// <=8 sequential tokens share hot weight cache lines, not weight registers.
kernel void dinf_dsv41_q8_f32_rows_seq4(
        device const uchar *weights [[buffer(0)]],
        device const float *input [[buffer(1)]],
        device float *output [[buffer(2)]],
        constant uint &k_dim [[buffer(3)]],
        constant uint &out_dim [[buffer(4)]],
        constant uint &total_rows [[buffer(5)]],
        constant uint &weight_row_bytes [[buffer(6)]],
        uint group [[threadgroup_position_in_grid]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[4][32];
    const uint groups_per_batch = (out_dim + 3u) / 4u;
    const uint row0 = (group % groups_per_batch) * 4u;
    const uint token0 = (group / groups_per_batch) * 8u;
    const uint ix = lane / 4u, il = lane % 4u;
    // Uniform bounds keep every barrier unconditional, including token tails.
    for (uint token = token0; token < min(token0 + 8u, total_rows); ++token) {
        float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint ib = uint(sg) * 8u + ix; ib < k_dim / 32u; ib += 4u * 8u) {
            float yl[8];
            for (ushort i = 0; i < 8; ++i)
                yl[i] = input[token * k_dim + ib * 32u + il * 8u + i];
            for (ushort row = 0; row < 4; ++row) {
                if (row0 + row < out_dim) {
                    device const block_q8_0 *blocks =
                        (device const block_q8_0 *)(weights + size_t(row0 + row) * weight_row_bytes);
                    device const int8_t *qs = blocks[ib].qs + il * 8u;
                    float sumq = 0.0f;
                    for (ushort i = 0; i < 8; ++i) sumq += float(qs[i]) * yl[i];
                    sum[row] += sumq * float(blocks[ib].d);
                }
            }
        }
        for (ushort row = 0; row < 4; ++row) {
            if (sg == 0) partial[row][lane] = 0.0f;
            sum[row] = simd_sum(sum[row]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort row = 0; row < 4; ++row)
            if (lane == 0) partial[row][sg] = sum[row];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort row = 0; row < 4; ++row) {
            const float total = simd_sum(partial[row][lane]);
            if (row0 + row < out_dim && lane == 0 && sg == 0)
                output[token * out_dim + row0 + row] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
