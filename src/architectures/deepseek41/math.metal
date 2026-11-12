#include <metal_stdlib>
using namespace metal;

// Adapted from audreyt/ds4 0dff39f91721bcf3472813c6d2bebf9353df2a1a:
// metal/dsv41.metal:9-20,115-156,265-295 (BF16, RoPE, QDQ, Engram),
// metal/dsv4_kv.metal:58-106 (nearest-even E4M3FN/E2M1), and
// metal/dsv4_hc.metal:141-202,707-750 (Sinkhorn/expansion).
// Norm: metal/norm.metal:47-81, ds4_metal.m:5643-5649.
// Stable HC gates: metal/dsv4_hc.metal:100-104,513,530,1404,1407;
// DS4_METAL_HC_STABLE defaults ON in ds4_metal.m:6927,6954.
// See NOTICE and LICENSE-MIT. Out-of-place F32 carriers; no half conversions.
struct Args {
    uint width, rows, mode, position, inverse, round_all;
    float eps;
    float frequencies[32];
    // RoPE only: row r uses position + (r / rows_per_position) * position_step.
    uint rows_per_position, position_step;
};

static inline float bf16(float x) {
    uint bits = as_type<uint>(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}

static inline float pow2ceil(float x) {
    const uint bits = as_type<uint>(x);
    return as_type<float>((bits & 0x7f800000u) +
                         ((bits & 0x7fffffu) ? 0x800000u : 0u));
}

static inline float e4m3_value(uint code) {
    const uint exponent = code >> 3u, mantissa = code & 7u;
    return exponent == 0u ? float(mantissa) * 0.001953125f
        : (1.0f + float(mantissa) * 0.125f) * as_type<float>((exponent + 120u) << 23u);
}

static inline float e4m3(float x) {
    const float ax = min(abs(x), 448.0f);
    uint lo = 0u, hi = 126u;
    while (lo < hi) {
        const uint mid = (lo + hi + 1u) >> 1u;
        if (e4m3_value(mid) <= ax) lo = mid;
        else hi = mid - 1u;
    }
    if (lo < 126u) {
        const float a = ax - e4m3_value(lo);
        const float b = abs(ax - e4m3_value(lo + 1u));
        if (b < a || (a == b && (lo & 1u))) ++lo;
    }
    return e4m3_value(lo);
}

constant float e2_values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
constant float e2_mid[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
static inline float e2m1(float x) {
    uint code = 0u;
    for (uint i = 0u; i < 7u; ++i)
        if (x > e2_mid[i] || (x == e2_mid[i] && (i & 1u))) code = i + 1u;
    return e2_values[code];
}

kernel void v41_bf16(
        device const float *a [[buffer(0)]],
        device const float *b [[buffer(1)]],
        device const float *c [[buffer(2)]],
        device float *y [[buffer(3)]],
        constant Args &args [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < args.width * args.rows) y[gid] = bf16(a[gid]);
}

kernel void v41_rms(
        device const float *a [[buffer(0)]],
        device const float *weight [[buffer(1)]],
        device const float *c [[buffer(2)]],
        device float *y [[buffer(3)]],
        constant Args &args [[buffer(4)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint threads [[threads_per_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[32];
    if (sg == 0) partial[lane] = 0.0f;
    const ulong offset = ulong(row) * args.width;
    float sum = 0.0f;
    for (uint i = tid; i < args.width / 4u; i += threads) {
        const ulong j = offset + 4u * i;
        // Scalar loads also support contiguous views not aligned to float4.
        const float4 x = float4(a[j], a[j + 1u], a[j + 2u], a[j + 3u]);
        sum += dot(x, x);
    }
    if (tid < args.width % 4u) {
        const float x = a[offset + args.width / 4u * 4u + tid];
        sum += x * x;
    }
    sum = simd_sum(sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = simd_sum(partial[lane]);
    const float scale = 1.0f / sqrt(sum / float(args.width) + args.eps);
    for (uint i = tid; i < args.width; i += threads) {
        const float v = a[offset + i] * scale;
        y[offset + i] = args.mode ? bf16(v * weight[i]) : v;
    }
}

kernel void v41_quantize(
        device const float *a [[buffer(0)]],
        device const float *b [[buffer(1)]],
        device const float *c [[buffer(2)]],
        device float *y [[buffer(3)]],
        constant Args &args [[buffer(4)]],
        uint group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint blocks = (args.width + block - 1u) / block;
    const uint row = group / blocks;
    const uint col = (group % blocks) * block + lane;
    const bool valid = lane < block && col < args.width;
    const ulong i = ulong(row) * args.width + col;
    const float value = valid ? bf16(a[i]) : 0.0f;
    const float amax = simd_max(abs(value));
    float scale;
    if (args.mode == 1u) scale = pow2ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
    else if (args.mode == 2u) scale = pow2ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
    else scale = e4m3(max(amax, 0.01171875f) / 6.0f);
    const float q = args.mode == 1u ? e4m3(abs(value) / scale) : e2m1(abs(value) / scale);
    if (valid) y[i] = bf16(copysign(q, value) * scale);
}

kernel void v41_rope(
        device const float *a [[buffer(0)]],
        device const float *b [[buffer(1)]],
        device const float *c [[buffer(2)]],
        device float *y [[buffer(3)]],
        constant Args &args [[buffer(4)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const ulong offset = ulong(row) * args.width;
    for (uint i = lane; i < args.width - 64u; i += 32u)
        y[offset + i] = args.round_all ? bf16(a[offset + i]) : a[offset + i];
    const uint position = args.position +
        (args.rows_per_position ? row / args.rows_per_position : 0u) * args.position_step;
    const float theta = float(position) * args.frequencies[lane];
    const float co = precise::cos(theta);
    const float si = args.inverse ? -precise::sin(theta) : precise::sin(theta);
    const ulong i = offset + args.width - 64u + 2u * lane;
    const float re = args.round_all ? bf16(a[i]) : a[i];
    const float im = args.round_all ? bf16(a[i + 1u]) : a[i + 1u];
    y[i] = bf16(fma(re, co, -(im * si)));
    y[i + 1u] = bf16(fma(im, co, re * si));
}

kernel void v41_hc_split(
        device const float *mix [[buffer(0)]],
        device const float *scale [[buffer(1)]],
        device const float *base [[buffer(2)]],
        device float *y [[buffer(3)]],
        constant Args &args [[buffer(4)]],
        uint group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (lane != 0u) return;
    // One row per threadgroup; a single-row call keeps the original offsets.
    mix += 24u * group;
    y += 24u * group;
    constexpr float eps = 1.0e-6f;
    for (uint i = 0; i < 4; ++i) {
        const float pre_z = fma(mix[i], scale[0], base[i]);
        const float post_z = fma(mix[4 + i], scale[1], base[4 + i]);
        // Keep the source tanh identity with ordered F32 arithmetic. Under
        // this library's existing safe math mode, plain tanh(50) returns NaN
        // on Apple GPUs; precise::tanh avoids that without changing Engram.
        y[i] = 0.5f * precise::tanh(0.5f * pre_z) + 0.5f + eps;
        y[4 + i] = 1.0f + precise::tanh(0.5f * post_z);
    }
    float comb[16];
    for (uint row = 0; row < 4; ++row) {
        const uint offset = 4 * row;
        for (uint col = 0; col < 4; ++col)
            comb[offset + col] = fma(mix[8 + offset + col], scale[2], base[8 + offset + col]);
        const float peak = max(max(comb[offset], comb[offset + 1]), max(comb[offset + 2], comb[offset + 3]));
        for (uint col = 0; col < 4; ++col) comb[offset + col] = exp(comb[offset + col] - peak);
        const float inv = 1.0f / (comb[offset] + comb[offset + 1] + comb[offset + 2] + comb[offset + 3]);
        for (uint col = 0; col < 4; ++col) comb[offset + col] = comb[offset + col] * inv + eps;
    }
    for (uint iteration = 0; iteration < 20; ++iteration) {
        if (iteration != 0u) {
            for (uint row = 0; row < 4; ++row) {
                const uint o = 4 * row;
                const float inv = 1.0f / (comb[o] + comb[o + 1] + comb[o + 2] + comb[o + 3] + eps);
                for (uint col = 0; col < 4; ++col) comb[o + col] *= inv;
            }
        }
        for (uint col = 0; col < 4; ++col) {
            const float inv = 1.0f / (comb[col] + comb[4 + col] + comb[8 + col] + comb[12 + col] + eps);
            for (uint row = 0; row < 4; ++row) comb[4 * row + col] *= inv;
        }
    }
    for (uint i = 0; i < 16; ++i) y[8 + i] = comb[i];
}

kernel void v41_collapse(
        device const float *residual [[buffer(0)]],
        device const float *pre [[buffer(1)]],
        device const float *c [[buffer(2)]],
        device float *y [[buffer(3)]],
        constant Args &args [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    // Rows of [4,width] residuals; mode is the pre stride (0 means 4).
    if (gid >= args.width * args.rows) return;
    const uint row = gid / args.width, i = gid % args.width;
    const uint stride = args.mode ? args.mode : 4u;
    device const float *r = residual + ulong(row) * 4u * args.width;
    float sum = 0.0f;
    for (uint stream = 0; stream < 4; ++stream)
        sum = fma(r[stream * args.width + i], pre[row * stride + stream], sum);
    y[gid] = bf16(sum);
}

kernel void v41_expand(
        device const float *block [[buffer(0)]],
        device const float *residual [[buffer(1)]],
        device const float *split [[buffer(2)]],
        device float *y [[buffer(3)]],
        constant Args &args [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.width * args.rows) return;
    const uint row = gid / args.width, i = gid % args.width;
    const ulong r = ulong(row) * 4u * args.width;
    device const float *sp = split + 24u * row;
    for (uint dest = 0; dest < 4; ++dest) {
        float sum = block[ulong(row) * args.width + i] * sp[4 + dest];
        // Split's contiguous rows are source streams (GGML ne0=destination).
        for (uint src = 0; src < 4; ++src)
            sum = fma(sp[8 + 4 * src + dest], residual[r + src * args.width + i], sum);
        y[r + dest * args.width + i] = bf16(sum);
    }
}

// HC mix for rows of normalized [4*width] residuals: y[row, o] = a[row] . fc[o].
kernel void v41_hc_mix(
        device const float *a [[buffer(0)]],
        device const float *fc [[buffer(1)]],
        device const float *c [[buffer(2)]],
        device float *y [[buffer(3)]],
        constant Args &args [[buffer(4)]],
        uint row [[threadgroup_position_in_grid]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const ulong offset = ulong(row) * args.width;
    for (uint o = sg; o < 24u; o += 8u) {
        float sum = 0.0f;
        for (uint i = lane; i < args.width; i += 32u)
            sum = fma(a[offset + i], fc[ulong(o) * args.width + i], sum);
        sum = simd_sum(sum);
        if (lane == 0) y[24u * row + o] = sum;
    }
}

kernel void v41_engram(
        device const float *residual [[buffer(0)]],
        device const float *kv [[buffer(1)]],
        device const float *weight [[buffer(2)]],
        device float *y [[buffer(3)]],
        constant Args &args [[buffer(4)]],
        uint group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    // Groups are (row, stream) for rows of [4,width] residuals and [5,width] kv.
    const uint row = group / 4u, stream = group % 4u;
    const ulong offset = ulong(group) * args.width;
    kv += ulong(row) * 5u * args.width;
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (uint i = lane; i < args.width; i += 32u) {
        const float h = residual[offset + i], k = bf16(kv[stream * args.width + i]);
        h2 += h * h;
        k2 += k * k;
        dot += (h * weight[stream * args.width + i]) * k;
    }
    h2 = simd_sum(h2);
    k2 = simd_sum(k2);
    dot = simd_sum(dot) * rsqrt(h2 / float(args.width) + args.eps) *
        rsqrt(k2 / float(args.width) + args.eps) * rsqrt(float(args.width));
    const float gate = 1.0f / (1.0f + exp(-copysign(sqrt(max(abs(dot), 1.0e-6f)), dot)));
    for (uint i = lane; i < args.width; i += 32u)
        y[offset + i] = bf16(fma(gate, bf16(kv[4u * args.width + i]), residual[offset + i]));
}
