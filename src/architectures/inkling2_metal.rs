//! Fused Metal kernels for Inkling 2.
//!
//! Each kernel replaces a run of small tensor ops in `inkling2_loop.rs`,
//! whose plain versions stay the CPU path and the reference for the tests
//! there. All tensors are F32 unless noted.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, CustomOp3, DType, Layout, MetalStorage, Result, Shape, Tensor,
};
use candle_metal_kernels::metal::{Buffer, ComputePipeline};

const INKLING_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Finite stand-in for -inf: the library compiles with fast math.
constant float NEG_LARGE = -1.0e30f;

// Rows come in groups of `groups` per token (row r is group r % groups of
// token r / groups). `x` holds `x_groups` rows per token: one row shared by
// the whole group, or one per group member. Flag 2 = the output is
// group-major [groups][tokens]; otherwise rows are token-major.
struct RowParams { uint rows; uint hidden; float eps; uint groups; uint x_groups; uint flags; };

inline uint x_row(constant RowParams &p, uint row) {
  return p.x_groups == p.groups ? row : (row / p.groups) * p.x_groups;
}

// Position of token-major row `row` in a group-major layout when `major`.
inline uint major_row(constant RowParams &p, uint row, bool major) {
  return major ? (row % p.groups) * (p.rows / p.groups) + row / p.groups : row;
}

// out[r] = [rms(x[r]) | rms(m[r])], one threadgroup per row.
kernel void inkling_rms_pair(
    device const float *x [[buffer(0)]],
    device const float *m [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant RowParams &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[2][32];
  device const float *xr = x + x_row(p, row) * p.hidden;
  device const float *mr = m + row * p.hidden;
  device float *o = out + major_row(p, row, (p.flags & 2u) != 0) * 2 * p.hidden;
  float sx = 0.0f;
  float sm = 0.0f;
  if ((p.hidden & 3u) == 0u) {
    const uint n4 = p.hidden >> 2;
    device const float4 *x4 = (device const float4 *)xr;
    device const float4 *m4 = (device const float4 *)mr;
    device float4 *o4 = (device float4 *)o;
    for (uint i = tid; i < n4; i += threads) {
      const float4 xv = x4[i];
      const float4 mv = m4[i];
      sx += dot(xv, xv);
      sm += dot(mv, mv);
    }
    sx = simd_sum(sx);
    sm = simd_sum(sm);
    if (lane == 0) { partial[0][sg] = sx; partial[1][sg] = sm; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tx = 0.0f;
    float tm = 0.0f;
    for (uint g = 0; g < (threads + 31) / 32; ++g) { tx += partial[0][g]; tm += partial[1][g]; }
    const float ix = rsqrt(tx / float(p.hidden) + p.eps);
    const float im = rsqrt(tm / float(p.hidden) + p.eps);
    for (uint i = tid; i < n4; i += threads) {
      o4[i] = x4[i] * ix;
      o4[n4 + i] = m4[i] * im;
    }
    return;
  }
  for (uint i = tid; i < p.hidden; i += threads) {
    sx += xr[i] * xr[i];
    sm += mr[i] * mr[i];
  }
  sx = simd_sum(sx);
  sm = simd_sum(sm);
  if (lane == 0) { partial[0][sg] = sx; partial[1][sg] = sm; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tx = 0.0f;
  float tm = 0.0f;
  for (uint g = 0; g < (threads + 31) / 32; ++g) { tx += partial[0][g]; tm += partial[1][g]; }
  const float ix = rsqrt(tx / float(p.hidden) + p.eps);
  const float im = rsqrt(tm / float(p.hidden) + p.eps);
  for (uint i = tid; i < p.hidden; i += threads) {
    o[i] = xr[i] * ix;
    o[p.hidden + i] = mr[i] * im;
  }
}

// Half-precision output version of inkling_rms_pair for prompt loop gates.
kernel void inkling_rms_pair_half(
    device const float *x [[buffer(0)]],
    device const float *m [[buffer(1)]],
    device half *out [[buffer(2)]],
    constant RowParams &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[2][32];
  device const float *xr = x + x_row(p, row) * p.hidden;
  device const float *mr = m + row * p.hidden;
  float sx = 0.0f;
  float sm = 0.0f;
  for (uint i = tid; i < p.hidden; i += threads) {
    const float xv = xr[i];
    const float mv = mr[i];
    sx += xv * xv;
    sm += mv * mv;
  }
  sx = simd_sum(sx);
  sm = simd_sum(sm);
  if (lane == 0) { partial[0][sg] = sx; partial[1][sg] = sm; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tx = 0.0f;
  float tm = 0.0f;
  for (uint g = 0; g < (threads + 31) / 32; ++g) { tx += partial[0][g]; tm += partial[1][g]; }
  const float ix = rsqrt(tx / float(p.hidden) + p.eps);
  const float im = rsqrt(tm / float(p.hidden) + p.eps);
  device half *o = out + major_row(p, row, (p.flags & 2u) != 0) * 2 * p.hidden;
  for (uint i = tid; i < p.hidden; i += threads) {
    o[i] = half(xr[i] * ix);
    o[p.hidden + i] = half(mr[i] * im);
  }
}

struct GateTransposeParams { uint tokens; uint heads; uint dim; uint gate_stride; };

// out[t, h * dim + d] = attended[h, t, d] * sigmoid(gate[t, h * dim + d]),
// where gate rows are `gate_stride` apart.
template <typename OutT>
kernel void inkling_gate_transpose_t(
    device const float *attended [[buffer(0)]],
    device const float *gate [[buffer(1)]],
    device OutT *out [[buffer(2)]],
    constant GateTransposeParams &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint width = p.heads * p.dim;
  if (gid >= p.tokens * width) { return; }
  const uint t = gid / width;
  const uint j = gid - t * width;
  const uint h = j / p.dim;
  const uint d = j - h * p.dim;
  const float value = attended[(h * p.tokens + t) * p.dim + d];
  out[gid] = OutT(value / (1.0f + exp(-gate[t * p.gate_stride + j])));
}

template [[host_name("inkling_gate_transpose")]]
kernel decltype(inkling_gate_transpose_t<float>) inkling_gate_transpose_t<float>;
template [[host_name("inkling_gate_transpose_f16")]]
kernel decltype(inkling_gate_transpose_t<half>) inkling_gate_transpose_t<half>;

struct SituParams { uint rows; uint ffn; float beta; float linear_beta; };

// out = beta tanh(g / beta) sigmoid(g) * linear_beta tanh(u / linear_beta),
// where each gate_up row is [g | u].
kernel void inkling_situ(
    device const float *gate_up [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant SituParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.rows * p.ffn) { return; }
  const uint row = gid / p.ffn;
  const uint i = gid - row * p.ffn;
  const float g = gate_up[row * 2 * p.ffn + i];
  const float u = gate_up[row * 2 * p.ffn + p.ffn + i];
  // Clamped: fast-math tanh and exp overflow far out, where both saturate.
  const float gate = p.beta * tanh(clamp(g / p.beta, -15.0f, 15.0f)) / (1.0f + exp(clamp(-g, -80.0f, 80.0f)));
  out[gid] = gate * p.linear_beta * tanh(clamp(u / p.linear_beta, -15.0f, 15.0f));
}

struct AttnParams {
  uint tokens; uint heads; uint kv_heads; uint keys; uint head_dim;
  uint extent; uint window; uint finalize; uint kv_head_stride;
  int first_query; int first_key; uint splits; uint chunk; uint has_gate;
};

// Pass 1: one threadgroup of eight simdgroups per (split, head, token) over
// keys [split * chunk, (split + 1) * chunk). Queries are already scaled;
// each bias row holds the bias by distance (extent + 1 columns), then the
// sink logit.
// Keys at distance d use bias column min(d, extent); keys after the query
// or, with a window, `window` or more back are skipped. Writes the split's
// running max, sum and unnormalized output: [max, sum, acc (head_dim)]; or,
// with `finalize` (one split), the output itself, as the combine pass would.
template <uint D, uint SG = 8>
kernel void inkling_attention_partial(
    device const float *queries [[buffer(0)]],
    device const half *k [[buffer(1)]],
    device const half *v [[buffer(2)]],
    device const float *bias [[buffer(3)]],
    device const float *gate [[buffer(4)]],
    device float *partial [[buffer(5)]],
    constant AttnParams &p [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint PER_LANE = D / 32;
  threadgroup float tg_max[SG];
  threadgroup float tg_sum[SG];
  threadgroup float tg_acc[SG][D];
  const uint split = group.x;
  const uint head = group.y;
  const uint token = group.z;
  const uint kv_head = head / (p.heads / p.kv_heads);
  device const float *row = queries + (token * p.heads + head) * D;
  device const float *table = bias + (token * p.heads + head) * (p.extent + 2);
  device const half *kh = k + kv_head * p.kv_head_stride;
  device const half *vh = v + kv_head * p.kv_head_stride;
  float q[PER_LANE];
  float acc[PER_LANE];
  if (D == 128) {
    const float4 q4 = *(device const float4 *)(row + lane * 4);
    q[0] = q4.x; q[1] = q4.y; q[2] = q4.z; q[3] = q4.w;
    acc[0] = 0.0f; acc[1] = 0.0f; acc[2] = 0.0f; acc[3] = 0.0f;
  } else {
    for (uint i = 0; i < PER_LANE; ++i) { q[i] = row[lane + 32 * i]; acc[i] = 0.0f; }
  }
  float running_max = NEG_LARGE;
  float running_sum = 0.0f;
  const int query = p.first_query + int(token);
  const uint begin = split * p.chunk;
  const uint end = min(p.keys, begin + p.chunk);
  if (D == 128) {
    const float4 q4 = float4(q[0], q[1], q[2], q[3]);
    float4 a4 = float4(0.0f);
    for (uint s = begin + sg; s < end; s += SG) {
      const int distance = query - (p.first_key + int(s));
      if (distance < 0 || (p.window > 0 && distance >= int(p.window))) { continue; }
      const float4 k4 = float4(*(device const half4 *)(kh + s * 128 + lane * 4));
      const float dot_val = simd_sum(dot(q4, k4));
      const float bval = distance < int(p.extent) ? table[uint(distance)] : 0.0f;
      const float score = dot_val + bval;
      const float next_max = max(running_max, score);
      const float correction = exp(running_max - next_max);
      const float weight = exp(score - next_max);
      running_sum = running_sum * correction + weight;
      const float4 v4 = float4(*(device const half4 *)(vh + s * 128 + lane * 4));
      a4 = a4 * correction + weight * v4;
      running_max = next_max;
    }
    if (lane == 0) { tg_max[sg] = running_max; tg_sum[sg] = running_sum; }
    tg_acc[sg][lane * 4 + 0] = a4.x;
    tg_acc[sg][lane * 4 + 1] = a4.y;
    tg_acc[sg][lane * 4 + 2] = a4.z;
    tg_acc[sg][lane * 4 + 3] = a4.w;
  } else {
    for (uint s = begin + sg; s < end; s += SG) {
      const int distance = query - (p.first_key + int(s));
      if (distance < 0 || (p.window > 0 && distance >= int(p.window))) { continue; }
      float dot = 0.0f;
      for (uint i = 0; i < PER_LANE; ++i) { dot += q[i] * float(kh[s * D + lane + 32 * i]); }
      dot = simd_sum(dot);
      const float bval = distance < int(p.extent) ? table[uint(distance)] : 0.0f;
      const float score = dot + bval;
      const float next_max = max(running_max, score);
      const float correction = exp(running_max - next_max);
      const float weight = exp(score - next_max);
      running_sum = running_sum * correction + weight;
      for (uint i = 0; i < PER_LANE; ++i) {
        acc[i] = acc[i] * correction + weight * float(vh[s * D + lane + 32 * i]);
      }
      running_max = next_max;
    }
    if (lane == 0) { tg_max[sg] = running_max; tg_sum[sg] = running_sum; }
    for (uint i = 0; i < PER_LANE; ++i) { tg_acc[sg][lane + 32 * i] = acc[i]; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float split_max = NEG_LARGE;
    for (uint g = 0; g < SG; ++g) { split_max = max(split_max, tg_max[g]); }
    float scale[SG];
    float split_sum = 0.0f;
    for (uint g = 0; g < SG; ++g) {
      scale[g] = exp(tg_max[g] - split_max);
      split_sum += tg_sum[g] * scale[g];
    }
    if (p.finalize != 0) {
      const float sink = table[p.extent + 1];
      const float total_max = max(sink, split_max);
      const float kept = exp(split_max - total_max);
      const float total = exp(sink - total_max) + split_sum * kept;
      const uint base = (token * p.heads + head) * D;
      for (uint i = 0; i < PER_LANE; ++i) {
        float value = 0.0f;
        for (uint g = 0; g < SG; ++g) { value += tg_acc[g][lane + 32 * i] * scale[g]; }
        value = value * kept / total;
        if (p.has_gate != 0) { value /= 1.0f + exp(-gate[base + lane + 32 * i]); }
        partial[base + lane + 32 * i] = value;
      }
      return;
    }
    device float *o = partial + ((token * p.heads + head) * p.splits + split) * (D + 2);
    if (lane == 0) { o[0] = split_max; o[1] = split_sum; }
    for (uint i = 0; i < PER_LANE; ++i) {
      float value = 0.0f;
      for (uint g = 0; g < SG; ++g) { value += tg_acc[g][lane + 32 * i] * scale[g]; }
      o[2 + lane + 32 * i] = value;
    }
  }
}

template [[host_name("inkling_attention_partial_64")]] kernel decltype(inkling_attention_partial<64, 8>) inkling_attention_partial<64, 8>;
template [[host_name("inkling_attention_partial_128")]] kernel decltype(inkling_attention_partial<128, 8>) inkling_attention_partial<128, 8>;
template [[host_name("inkling_attention_partial_256")]] kernel decltype(inkling_attention_partial<256, 8>) inkling_attention_partial<256, 8>;

// GQA=3 + multi-token shared-KV attention partial pass for head_dim=128:
// one threadgroup streams each K/V row once across all 3 query heads of a
// KV head and up to T_TILE tokens (3x..9x fewer KV cache reads).
template <uint T_TILE, uint SG>
kernel void inkling_attention_partial_gqa3_128(
    device const float *queries [[buffer(0)]],
    device const half *k [[buffer(1)]],
    device const half *v [[buffer(2)]],
    device const float *bias [[buffer(3)]],
    device const float *gate [[buffer(4)]],
    device float *partial [[buffer(5)]],
    constant AttnParams &p [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint D = 128;
  constexpr uint PER_LANE = 4;
  constexpr uint GQA = 3;
  constexpr uint Q = GQA * T_TILE;
  threadgroup float tg_max[SG][Q];
  threadgroup float tg_sum[SG][Q];
  threadgroup float tg_acc[SG][Q][D];

  const uint split = group.x;
  const uint kv_head = group.y;
  const uint t0 = group.z * T_TILE;
  const uint h0 = kv_head * GQA;
  device const half *kh = k + kv_head * p.kv_head_stride;
  device const half *vh = v + kv_head * p.kv_head_stride;

  float4 q[Q];
  float4 acc[Q];
  float rmax[Q];
  float rsum[Q];
  #pragma unroll
  for (uint ti = 0; ti < T_TILE; ++ti) {
    const uint tok = t0 + ti;
    const bool valid = tok < p.tokens;
    #pragma unroll
    for (uint g = 0; g < GQA; ++g) {
      const uint qi = ti * GQA + g;
      rmax[qi] = NEG_LARGE;
      rsum[qi] = 0.0f;
      device const float *row = queries + (tok * p.heads + (h0 + g)) * D;
      q[qi] = valid ? *(device const float4 *)(row + lane * 4) : float4(0.0f);
      acc[qi] = float4(0.0f);
    }
  }

  const uint begin = split * p.chunk;
  const uint end = min(p.keys, begin + p.chunk);
  const int max_query = p.first_query + int(min(t0 + T_TILE, p.tokens)) - 1;
  const int min_query = p.first_query + int(t0);
  for (uint s = begin + sg; s < end; s += SG) {
    const int key_pos = p.first_key + int(s);
    if (max_query - key_pos < 0) { continue; }
    if (p.window > 0 && (min_query - key_pos) >= int(p.window)) { continue; }
    const float4 kv = float4(*(device const half4 *)(kh + s * D + lane * 4));
    const float4 vv = float4(*(device const half4 *)(vh + s * D + lane * 4));
    #pragma unroll
    for (uint ti = 0; ti < T_TILE; ++ti) {
      const uint tok = t0 + ti;
      if (tok >= p.tokens) { continue; }
      const int distance = (p.first_query + int(tok)) - key_pos;
      if (distance < 0 || (p.window > 0 && distance >= int(p.window))) { continue; }
      const bool in_extent = distance < int(p.extent);
      #pragma unroll
      for (uint g = 0; g < GQA; ++g) {
        const uint qi = ti * GQA + g;
        const float dot_val = simd_sum(dot(q[qi], kv));
        const float bval = in_extent ? bias[(tok * p.heads + (h0 + g)) * (p.extent + 2) + uint(distance)] : 0.0f;
        const float score = dot_val + bval;
        const float next_max = max(rmax[qi], score);
        const float corr = exp(rmax[qi] - next_max);
        const float w = exp(score - next_max);
        rsum[qi] = rsum[qi] * corr + w;
        acc[qi] = acc[qi] * corr + w * vv;
        rmax[qi] = next_max;
      }
    }
  }

  #pragma unroll
  for (uint qi = 0; qi < Q; ++qi) {
    if (lane == 0) {
      tg_max[sg][qi] = rmax[qi];
      tg_sum[sg][qi] = rsum[qi];
    }
    tg_acc[sg][qi][lane * 4 + 0] = acc[qi].x;
    tg_acc[sg][qi][lane * 4 + 1] = acc[qi].y;
    tg_acc[sg][qi][lane * 4 + 2] = acc[qi].z;
    tg_acc[sg][qi][lane * 4 + 3] = acc[qi].w;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Each simdgroup finalizes one or more of the Q queries in parallel.
  for (uint qi = sg; qi < Q; qi += SG) {
    const uint ti = qi / GQA;
    const uint g = qi % GQA;
    const uint tok = t0 + ti;
    if (tok >= p.tokens) { continue; }
    const uint head = h0 + g;
    float split_max = NEG_LARGE;
    #pragma unroll
    for (uint s_g = 0; s_g < SG; ++s_g) { split_max = max(split_max, tg_max[s_g][qi]); }
    float sc[SG];
    float split_sum = 0.0f;
    #pragma unroll
    for (uint s_g = 0; s_g < SG; ++s_g) {
      sc[s_g] = exp(tg_max[s_g][qi] - split_max);
      split_sum += tg_sum[s_g][qi] * sc[s_g];
    }
    if (p.finalize != 0) {
      const float sink = bias[(tok * p.heads + head) * (p.extent + 2) + p.extent + 1];
      const float total_max = max(sink, split_max);
      const float kept = exp(split_max - total_max);
      const float total = exp(sink - total_max) + split_sum * kept;
      const uint base = (tok * p.heads + head) * D;
      #pragma unroll
      for (uint i = 0; i < PER_LANE; ++i) {
        float value = 0.0f;
        #pragma unroll
        for (uint s_g = 0; s_g < SG; ++s_g) { value += tg_acc[s_g][qi][lane + 32 * i] * sc[s_g]; }
        value = value * kept / total;
        if (p.has_gate != 0) { value /= 1.0f + exp(-gate[base + lane + 32 * i]); }
        partial[base + lane + 32 * i] = value;
      }
    } else {
      device float *o = partial + ((tok * p.heads + head) * p.splits + split) * (D + 2);
      if (lane == 0) { o[0] = split_max; o[1] = split_sum; }
      #pragma unroll
      for (uint i = 0; i < PER_LANE; ++i) {
        float value = 0.0f;
        #pragma unroll
        for (uint s_g = 0; s_g < SG; ++s_g) { value += tg_acc[s_g][qi][lane + 32 * i] * sc[s_g]; }
        o[2 + lane + 32 * i] = value;
      }
    }
  }
}

typedef decltype(inkling_attention_partial_gqa3_128<1, 8>) inkling_attn_gqa3_fn;
template [[host_name("inkling_attention_partial_gqa3_128_t1")]] kernel inkling_attn_gqa3_fn inkling_attention_partial_gqa3_128<1, 8>;
template [[host_name("inkling_attention_partial_gqa3_128_t2")]] kernel inkling_attn_gqa3_fn inkling_attention_partial_gqa3_128<2, 4>;
template [[host_name("inkling_attention_partial_gqa3_128_t3")]] kernel inkling_attn_gqa3_fn inkling_attention_partial_gqa3_128<3, 4>;

// Pass 2: one simdgroup per (head, token) merges the splits and the sink,
// a zero-value logit that only adds to the softmax denominator. With
// `has_gate`, the output is also multiplied by sigmoid(gate logits).
template <uint D>
kernel void inkling_attention_combine(
    device const float *bias [[buffer(0)]],
    device const float *partial [[buffer(1)]],
    device const float *gate [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant AttnParams &p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint PER_LANE = D / 32;
  const uint head = group.x;
  const uint token = group.y;
  const uint stride = D + 2;
  const float sink = bias[(token * p.heads + head) * (p.extent + 2) + p.extent + 1];
  device const float *parts = partial + (token * p.heads + head) * p.splits * stride;
  float total_max = sink;
  for (uint s = 0; s < p.splits; ++s) { total_max = max(total_max, parts[s * stride]); }
  float total = exp(sink - total_max);
  float acc[PER_LANE];
  for (uint i = 0; i < PER_LANE; ++i) { acc[i] = 0.0f; }
  for (uint s = 0; s < p.splits; ++s) {
    const float scale = exp(parts[s * stride] - total_max);
    total += parts[s * stride + 1] * scale;
    for (uint i = 0; i < PER_LANE; ++i) { acc[i] += parts[s * stride + 2 + lane + 32 * i] * scale; }
  }
  const uint base = (token * p.heads + head) * D;
  for (uint i = 0; i < PER_LANE; ++i) {
    float value = acc[i] / total;
    if (p.has_gate != 0) { value /= 1.0f + exp(-gate[base + lane + 32 * i]); }
    out[base + lane + 32 * i] = value;
  }
}

template [[host_name("inkling_attention_combine_64")]] kernel decltype(inkling_attention_combine<64>) inkling_attention_combine<64>;
template [[host_name("inkling_attention_combine_128")]] kernel decltype(inkling_attention_combine<128>) inkling_attention_combine<128>;
template [[host_name("inkling_attention_combine_256")]] kernel decltype(inkling_attention_combine<256>) inkling_attention_combine<256>;

// ---- Single-pass kernels: one dispatch for a run of per-visit steps ----

struct MixOneParams { uint tokens; uint blocks; uint hidden; uint queries; float eps; float scale; float norm_eps; uint flags; };

struct MixSources { device const float *row[8]; };

// Threadgroup scratch for mix_token<S, Q>.
template <uint S, uint Q>
struct MixScratch {
  float sums[32][S * (Q + 1)];
  float weights[Q][S];
  float norm_sums[32][Q];
  float norm_inv[Q];
};

// Attention over depth for one token, by one threadgroup: sources are the
// rows src.row[0..S) (the blocks, then the current state), queries [Q][hidden].
// Compile-time counts keep the per-thread sums in registers. Flags: 1 =
// RMS-normalize each output row, times norm row q (2: one shared norm row);
// 4 = group-major output. Each thread reads source elements i = tid,
// tid + threads, ..., so a current row this thread just wrote needs no
// barrier.
template <uint S, uint Q>
inline void mix_token(thread const MixSources &src, device const float *queries,
                      device const float *norm, device float *out,
                      constant MixOneParams &p, uint token, uint tid, uint threads,
                      uint lane, uint sg, threadgroup MixScratch<S, Q> &scratch) {
  constexpr uint SLOTS = S * (Q + 1);
  const uint groups = (threads + 31) / 32;
  // Slot s: source s's sum of squares; slot (q + 1) * S + s: its dot
  // product with query q.
  float acc[SLOTS];
  #pragma unroll
  for (uint slot = 0; slot < SLOTS; ++slot) { acc[slot] = 0.0f; }
  const bool vec4 = (p.hidden & 3u) == 0u;
  if (vec4) {
    const uint n4 = p.hidden >> 2;
    for (uint i = tid; i < n4; i += threads) {
      float4 qv[Q];
      #pragma unroll
      for (uint q = 0; q < Q; ++q) { qv[q] = ((device const float4 *)(queries + q * p.hidden))[i]; }
      #pragma unroll
      for (uint s = 0; s < S; ++s) {
        const float4 v = ((device const float4 *)src.row[s])[i];
        acc[s] += dot(v, v);
        #pragma unroll
        for (uint q = 0; q < Q; ++q) { acc[(q + 1) * S + s] += dot(v, qv[q]); }
      }
    }
  } else {
    for (uint i = tid; i < p.hidden; i += threads) {
      float qv[Q];
      for (uint q = 0; q < Q; ++q) { qv[q] = queries[q * p.hidden + i]; }
      for (uint s = 0; s < S; ++s) {
        const float v = src.row[s][i];
        acc[s] += v * v;
        for (uint q = 0; q < Q; ++q) { acc[(q + 1) * S + s] += v * qv[q]; }
      }
    }
  }
  for (uint slot = 0; slot < SLOTS; ++slot) {
    const float total = simd_sum(acc[slot]);
    if (lane == 0) { scratch.sums[sg][slot] = total; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < Q) {
    const uint q = tid;
    float inv[S];
    float score[S];
    float best = NEG_LARGE;
    for (uint s = 0; s < S; ++s) {
      float sumsq = 0.0f;
      float dot = 0.0f;
      for (uint g = 0; g < groups; ++g) {
        sumsq += scratch.sums[g][s];
        dot += scratch.sums[g][(q + 1) * S + s];
      }
      inv[s] = rsqrt(sumsq / float(p.hidden) + p.eps);
      score[s] = dot * inv[s] * p.scale;
      best = max(best, score[s]);
    }
    float total = 0.0f;
    for (uint s = 0; s < S; ++s) {
      score[s] = exp(score[s] - best);
      total += score[s];
    }
    for (uint s = 0; s < S; ++s) { scratch.weights[q][s] = score[s] / total * inv[s]; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool normed = (p.flags & 1u) != 0;
  const bool major = (p.flags & 4u) != 0;
  if (vec4 && normed && Q <= 3 && p.hidden <= threads * 12) {
    const uint n4 = p.hidden >> 2;
    float4 vals[Q][3];
    float squares[Q];
    #pragma unroll
    for (uint q = 0; q < Q; ++q) { squares[q] = 0.0f; }
    uint k = 0;
    for (uint i = tid; i < n4; i += threads, ++k) {
      float4 values[S];
      #pragma unroll
      for (uint s = 0; s < S; ++s) { values[s] = ((device const float4 *)src.row[s])[i]; }
      #pragma unroll
      for (uint q = 0; q < Q; ++q) {
        float4 v = float4(0.0f);
        #pragma unroll
        for (uint s = 0; s < S; ++s) { v += scratch.weights[q][s] * values[s]; }
        vals[q][k] = v;
        squares[q] += dot(v, v);
      }
    }
    #pragma unroll
    for (uint q = 0; q < Q; ++q) {
      const float total = simd_sum(squares[q]);
      if (lane == 0) { scratch.norm_sums[sg][q] = total; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < Q) {
      float total = 0.0f;
      for (uint g = 0; g < groups; ++g) { total += scratch.norm_sums[g][tid]; }
      scratch.norm_inv[tid] = rsqrt(total / float(p.hidden) + p.norm_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    k = 0;
    for (uint i = tid; i < n4; i += threads, ++k) {
      #pragma unroll
      for (uint q = 0; q < Q; ++q) {
        const uint out_row = major ? q * p.tokens + token : token * Q + q;
        const uint norm_row = (p.flags & 2u) != 0 ? 0 : q;
        const float4 nw = ((device const float4 *)(norm + norm_row * p.hidden))[i];
        ((device float4 *)(out + out_row * p.hidden))[i] = vals[q][k] * scratch.norm_inv[q] * nw;
      }
    }
    return;
  }
  if (vec4 && !normed) {
    const uint n4 = p.hidden >> 2;
    for (uint i = tid; i < n4; i += threads) {
      float4 values[S];
      #pragma unroll
      for (uint s = 0; s < S; ++s) { values[s] = ((device const float4 *)src.row[s])[i]; }
      #pragma unroll
      for (uint q = 0; q < Q; ++q) {
        float4 v = float4(0.0f);
        #pragma unroll
        for (uint s = 0; s < S; ++s) { v += scratch.weights[q][s] * values[s]; }
        const uint out_row = major ? q * p.tokens + token : token * Q + q;
        ((device float4 *)(out + out_row * p.hidden))[i] = v;
      }
    }
    return;
  }
  if (normed && Q <= 3 && p.hidden <= threads * 12) {
    float vals[Q][12];
    float squares[Q];
    for (uint q = 0; q < Q; ++q) { squares[q] = 0.0f; }
    uint k = 0;
    for (uint i = tid; i < p.hidden; i += threads, ++k) {
      float values[S];
      for (uint s = 0; s < S; ++s) { values[s] = src.row[s][i]; }
      for (uint q = 0; q < Q; ++q) {
        float v = 0.0f;
        for (uint s = 0; s < S; ++s) { v += scratch.weights[q][s] * values[s]; }
        vals[q][k] = v;
        squares[q] += v * v;
      }
    }
    for (uint q = 0; q < Q; ++q) {
      const float total = simd_sum(squares[q]);
      if (lane == 0) { scratch.norm_sums[sg][q] = total; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < Q) {
      float total = 0.0f;
      for (uint g = 0; g < groups; ++g) { total += scratch.norm_sums[g][tid]; }
      scratch.norm_inv[tid] = rsqrt(total / float(p.hidden) + p.norm_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    k = 0;
    for (uint i = tid; i < p.hidden; i += threads, ++k) {
      for (uint q = 0; q < Q; ++q) {
        const uint out_row = major ? q * p.tokens + token : token * Q + q;
        const uint norm_row = (p.flags & 2u) != 0 ? 0 : q;
        out[out_row * p.hidden + i] = vals[q][k] * scratch.norm_inv[q] * norm[norm_row * p.hidden + i];
      }
    }
    return;
  }
  float squares[Q];
  for (uint q = 0; q < Q; ++q) { squares[q] = 0.0f; }
  for (uint i = tid; i < p.hidden; i += threads) {
    float values[S];
    for (uint s = 0; s < S; ++s) { values[s] = src.row[s][i]; }
    for (uint q = 0; q < Q; ++q) {
      float v = 0.0f;
      for (uint s = 0; s < S; ++s) { v += scratch.weights[q][s] * values[s]; }
      const uint out_row = major ? q * p.tokens + token : token * Q + q;
      out[out_row * p.hidden + i] = v;
      squares[q] += v * v;
    }
  }
  if (!normed) { return; }
  for (uint q = 0; q < Q; ++q) {
    const float total = simd_sum(squares[q]);
    if (lane == 0) { scratch.norm_sums[sg][q] = total; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < Q) {
    float total = 0.0f;
    for (uint g = 0; g < groups; ++g) { total += scratch.norm_sums[g][tid]; }
    scratch.norm_inv[tid] = rsqrt(total / float(p.hidden) + p.norm_eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Each thread rescales only the elements it wrote above.
  for (uint i = tid; i < p.hidden; i += threads) {
    for (uint q = 0; q < Q; ++q) {
      const uint out_row = major ? q * p.tokens + token : token * Q + q;
      const uint norm_row = (p.flags & 2u) != 0 ? 0 : q;
      out[out_row * p.hidden + i] *= scratch.norm_inv[q] * norm[norm_row * p.hidden + i];
    }
  }
}

// The depth mix, one threadgroup per token; source s is the token's row of
// buffer s (the blocks, then the current state; later buffers unused).
template <uint S, uint Q>
kernel void inkling_depth_mix_one(
    device const float *b0 [[buffer(0)]],
    device const float *b1 [[buffer(1)]],
    device const float *b2 [[buffer(2)]],
    device const float *b3 [[buffer(3)]],
    device const float *b4 [[buffer(4)]],
    device const float *b5 [[buffer(5)]],
    device const float *b6 [[buffer(6)]],
    device const float *b7 [[buffer(7)]],
    device const float *queries [[buffer(8)]],
    device const float *norm [[buffer(9)]],
    device float *out [[buffer(10)]],
    constant MixOneParams &p [[buffer(11)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup MixScratch<S, Q> scratch;
  const uint row = token * p.hidden;
  const MixSources src = {{b0 + row, b1 + row, b2 + row, b3 + row,
                           b4 + row, b5 + row, b6 + row, b7 + row}};
  mix_token<S, Q>(src, queries, norm, out, p, token, tid, threads, lane, sg, scratch);
}

template [[host_name("inkling_depth_mix_one_2_1")]] kernel decltype(inkling_depth_mix_one<2, 1>) inkling_depth_mix_one<2, 1>;
template [[host_name("inkling_depth_mix_one_2_2")]] kernel decltype(inkling_depth_mix_one<2, 2>) inkling_depth_mix_one<2, 2>;
template [[host_name("inkling_depth_mix_one_2_3")]] kernel decltype(inkling_depth_mix_one<2, 3>) inkling_depth_mix_one<2, 3>;
template [[host_name("inkling_depth_mix_one_2_4")]] kernel decltype(inkling_depth_mix_one<2, 4>) inkling_depth_mix_one<2, 4>;
template [[host_name("inkling_depth_mix_one_3_1")]] kernel decltype(inkling_depth_mix_one<3, 1>) inkling_depth_mix_one<3, 1>;
template [[host_name("inkling_depth_mix_one_3_2")]] kernel decltype(inkling_depth_mix_one<3, 2>) inkling_depth_mix_one<3, 2>;
template [[host_name("inkling_depth_mix_one_3_3")]] kernel decltype(inkling_depth_mix_one<3, 3>) inkling_depth_mix_one<3, 3>;
template [[host_name("inkling_depth_mix_one_3_4")]] kernel decltype(inkling_depth_mix_one<3, 4>) inkling_depth_mix_one<3, 4>;
template [[host_name("inkling_depth_mix_one_4_1")]] kernel decltype(inkling_depth_mix_one<4, 1>) inkling_depth_mix_one<4, 1>;
template [[host_name("inkling_depth_mix_one_4_2")]] kernel decltype(inkling_depth_mix_one<4, 2>) inkling_depth_mix_one<4, 2>;
template [[host_name("inkling_depth_mix_one_4_3")]] kernel decltype(inkling_depth_mix_one<4, 3>) inkling_depth_mix_one<4, 3>;
template [[host_name("inkling_depth_mix_one_4_4")]] kernel decltype(inkling_depth_mix_one<4, 4>) inkling_depth_mix_one<4, 4>;
template [[host_name("inkling_depth_mix_one_5_1")]] kernel decltype(inkling_depth_mix_one<5, 1>) inkling_depth_mix_one<5, 1>;
template [[host_name("inkling_depth_mix_one_5_2")]] kernel decltype(inkling_depth_mix_one<5, 2>) inkling_depth_mix_one<5, 2>;
template [[host_name("inkling_depth_mix_one_5_3")]] kernel decltype(inkling_depth_mix_one<5, 3>) inkling_depth_mix_one<5, 3>;
template [[host_name("inkling_depth_mix_one_5_4")]] kernel decltype(inkling_depth_mix_one<5, 4>) inkling_depth_mix_one<5, 4>;
template [[host_name("inkling_depth_mix_one_6_1")]] kernel decltype(inkling_depth_mix_one<6, 1>) inkling_depth_mix_one<6, 1>;
template [[host_name("inkling_depth_mix_one_6_2")]] kernel decltype(inkling_depth_mix_one<6, 2>) inkling_depth_mix_one<6, 2>;
template [[host_name("inkling_depth_mix_one_6_3")]] kernel decltype(inkling_depth_mix_one<6, 3>) inkling_depth_mix_one<6, 3>;
template [[host_name("inkling_depth_mix_one_6_4")]] kernel decltype(inkling_depth_mix_one<6, 4>) inkling_depth_mix_one<6, 4>;
template [[host_name("inkling_depth_mix_one_7_1")]] kernel decltype(inkling_depth_mix_one<7, 1>) inkling_depth_mix_one<7, 1>;
template [[host_name("inkling_depth_mix_one_7_2")]] kernel decltype(inkling_depth_mix_one<7, 2>) inkling_depth_mix_one<7, 2>;
template [[host_name("inkling_depth_mix_one_7_3")]] kernel decltype(inkling_depth_mix_one<7, 3>) inkling_depth_mix_one<7, 3>;
template [[host_name("inkling_depth_mix_one_7_4")]] kernel decltype(inkling_depth_mix_one<7, 4>) inkling_depth_mix_one<7, 4>;
template [[host_name("inkling_depth_mix_one_8_1")]] kernel decltype(inkling_depth_mix_one<8, 1>) inkling_depth_mix_one<8, 1>;
template [[host_name("inkling_depth_mix_one_8_2")]] kernel decltype(inkling_depth_mix_one<8, 2>) inkling_depth_mix_one<8, 2>;
template [[host_name("inkling_depth_mix_one_8_3")]] kernel decltype(inkling_depth_mix_one<8, 3>) inkling_depth_mix_one<8, 3>;
template [[host_name("inkling_depth_mix_one_8_4")]] kernel decltype(inkling_depth_mix_one<8, 4>) inkling_depth_mix_one<8, 4>;

struct KvFinishParams {
  uint tokens; uint heads; uint dim; uint width;
  uint k_stride; uint v_stride; uint k_taps_stride; uint v_taps_stride;
  float scale; float clip; float eps; uint flags;
  uint capacity; uint row;
};

inline float kv_history(device const float *past, device const float *x, uint stride,
                        uint history, uint channels, uint row, uint c, bool has_past) {
  if (row < history) { return has_past ? past[row * channels + c] : 0.0f; }
  return x[(row - history) * stride + c];
}

// The K and V short convolutions with their head norms, one threadgroup of
// `dim` threads per (head, token, K or V). K is L2-clipped per head, V
// RMS-normalized per head; both are written head-major, to `out`
// [K|V][heads][tokens][dim] or, with flag 4, as f16 into the caches
// [heads][capacity][dim] from `row` on. Threadgroups with token < width - 1
// also write that row of the next conv state ([K|V][width - 1][channels]).
// Flags: 1 = per-token dynamic taps, 2 = a past state (else zeros).
kernel void inkling_kv_finish(
    device const float *k_in [[buffer(0)]],
    device const float *v_in [[buffer(1)]],
    device const float *k_past [[buffer(2)]],
    device const float *v_past [[buffer(3)]],
    device const float *k_weight [[buffer(4)]],
    device const float *v_weight [[buffer(5)]],
    device const float *k_taps [[buffer(6)]],
    device const float *v_taps [[buffer(7)]],
    device float *out [[buffer(8)]],
    device float *state [[buffer(9)]],
    device half *k_cache [[buffer(10)]],
    device half *v_cache [[buffer(11)]],
    constant KvFinishParams &p [[buffer(12)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 thread_index [[thread_position_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[32];
  const uint tid = thread_index.x;
  const uint threads = group_size.x;
  const uint head = group.x;
  const uint t = group.y;
  const uint which = group.z;
  const bool is_v = which == 1;
  device const float *x = is_v ? v_in : k_in;
  device const float *past = is_v ? v_past : k_past;
  device const float *weight = is_v ? v_weight : k_weight;
  device const float *taps = is_v ? v_taps : k_taps;
  const uint stride = is_v ? p.v_stride : p.k_stride;
  const uint taps_stride = is_v ? p.v_taps_stride : p.k_taps_stride;
  const uint history = p.width - 1;
  const uint channels = p.heads * p.dim;
  const bool has_past = (p.flags & 2u) != 0;
  const uint d = tid;
  const uint c = head * p.dim + d;
  if (t < history && d < p.dim) {
    state[(which * history + t) * channels + c] =
        kv_history(past, x, stride, history, channels, p.tokens + t, c, has_past);
  }
  if (t >= p.tokens) { return; }
  float acc = 0.0f;
  if (d < p.dim) {
    acc = kv_history(past, x, stride, history, channels, t + history, c, has_past);
    for (uint j = 0; j < p.width; ++j) {
      acc += weight[c * p.width + j] * kv_history(past, x, stride, history, channels, t + j, c, has_past);
    }
    if ((p.flags & 1u) != 0) {
      device const float *row = taps + t * taps_stride + head * p.width;
      for (uint lag = 0; lag < p.width; ++lag) {
        acc += row[lag] * kv_history(past, x, stride, history, channels, t + history - lag, c, has_past);
      }
    }
    acc *= p.scale;
  }
  const float sumsq = simd_sum(acc * acc);
  if (lane == 0) { partial[sg] = sumsq; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint g = 0; g < (threads + 31) / 32; ++g) { total += partial[g]; }
  const float factor = is_v ? rsqrt(total / float(p.dim) + p.eps)
                            : min(p.clip / (sqrt(total) + 1.0e-6f), 1.0f);
  if (d >= p.dim) { return; }
  if ((p.flags & 4u) != 0) {
    device half *cache = is_v ? v_cache : k_cache;
    cache[(head * p.capacity + p.row + t) * p.dim + d] = half(acc * factor);
  } else {
    out[((which * p.heads + head) * p.tokens + t) * p.dim + d] = acc * factor;
  }
}

struct QueryPrepParams {
  uint tokens; uint heads; uint dim; uint rel; uint extent; uint x_stride;
  uint q_offset; uint rel_offset; uint sink_offset;
  float q_clip; float rel_clip; float q_post; uint flags; uint first_position;
};

// From the stacked query-side projection rows: the scaled, clipped queries
// [tokens][heads][dim] and, per (token, head), the attention bias by distance
// (extent + 1 columns, from the clipped relative coefficients times `basis`
// [rel][extent + 1]) followed by the sink logit. One threadgroup of
// QUERY_PREP_TOKENS simdgroups per (head, block of QUERY_PREP_TOKENS tokens,
// slice of bias columns): simdgroup j handles token j of the block, and each
// thread keeps its basis column in registers across the block's tokens.
// Slice 0 also writes the queries and sinks. R is the relative width rounded
// up (16 or 32). Flag 1 = times token_scale[token] (the log scale). Token t
// has at most `first_position + t` keys before it, so no key lies further
// back and the bias columns past that distance are left unwritten.
constant constexpr uint QUERY_PREP_TOKENS = 8;

template <uint R, typename OutT>
kernel void inkling_query_prep(
    device const float *x [[buffer(0)]],
    device const float *q_scale [[buffer(1)]],
    device const float *rel_scale [[buffer(2)]],
    device const float *token_scale [[buffer(3)]],
    device const float *basis [[buffer(4)]],
    device OutT *q_out [[buffer(5)]],
    device OutT *bias [[buffer(6)]],
    constant QueryPrepParams &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 thread_index [[thread_position_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float rel[QUERY_PREP_TOKENS][R];
  const uint tid = thread_index.x;
  const uint threads = group_size.x;
  const uint head = group.x;
  const uint first = group.y * QUERY_PREP_TOKENS;
  const uint slice = group.z;
  const uint token = first + sg;
  const bool active = sg < QUERY_PREP_TOKENS && token < p.tokens;
  device const float *row = x + token * p.x_stride;
  const float ts = active && (p.flags & 1u) != 0 ? token_scale[token] : 1.0f;
  if (sg < QUERY_PREP_TOKENS) {
    float r = 0.0f;
    if (active && lane < p.rel) {
      r = row[p.rel_offset + head * p.rel + lane] * rel_scale[head * p.rel + lane];
    }
    const float norm = sqrt(simd_sum(r * r));
    if (lane < R) { rel[sg][lane] = r * min(p.rel_clip / (norm + 1.0e-6f), 1.0f) * ts; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint e = slice * threads + tid;
  if (e <= p.extent) {
    float column[R];
    for (uint r = 0; r < R; ++r) { column[r] = r < p.rel ? basis[r * (p.extent + 1) + e] : 0.0f; }
    for (uint j = 0; j < QUERY_PREP_TOKENS; ++j) {
      if (first + j >= p.tokens) { break; }
      if (e > p.first_position + first + j) { continue; }
      float a = 0.0f;
      for (uint r = 0; r < R; ++r) { a += rel[j][r] * column[r]; }
      bias[((first + j) * p.heads + head) * (p.extent + 2) + e] = OutT(a);
    }
  }
  if (slice != 0 || !active) { return; }
  device OutT *b = bias + (token * p.heads + head) * (p.extent + 2);
  if (lane == 0) { b[p.extent + 1] = OutT(row[p.sink_offset + head]); }
  float sumsq = 0.0f;
  for (uint d = lane; d < p.dim; d += 32) {
    const float v = row[p.q_offset + head * p.dim + d] * q_scale[head * p.dim + d];
    sumsq += v * v;
  }
  const float factor = min(p.q_clip / (sqrt(simd_sum(sumsq)) + 1.0e-6f), 1.0f) * ts * p.q_post;
  for (uint d = lane; d < p.dim; d += 32) {
    const float v = row[p.q_offset + head * p.dim + d] * q_scale[head * p.dim + d];
    q_out[(token * p.heads + head) * p.dim + d] = OutT(v * factor);
  }
}

template [[host_name("inkling_query_prep_16")]] kernel decltype(inkling_query_prep<16, float>) inkling_query_prep<16, float>;
template [[host_name("inkling_query_prep_32")]] kernel decltype(inkling_query_prep<32, float>) inkling_query_prep<32, float>;
template [[host_name("inkling_query_prep_16_f16")]] kernel decltype(inkling_query_prep<16, half>) inkling_query_prep<16, half>;
template [[host_name("inkling_query_prep_32_f16")]] kernel decltype(inkling_query_prep<32, half>) inkling_query_prep<32, half>;

struct SoftcapParams { uint count; float cap; };

// Single-pass out[i] = cap * tanh(clamp(x[i] / cap, -15, 15)) for logits.
kernel void inkling_softcap(
    device const float *x [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant SoftcapParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) { return; }
  out[gid] = p.cap * tanh(clamp(x[gid] / p.cap, -15.0f, 15.0f));
}

struct PostConvParams { uint tokens; uint channels; uint width; uint x_stride; uint routes; float scale; uint flags; };

// Input row r of the post convolution: x's row, or with flag 1 the
// route-weighted sum of routed rows [tokens][routes][channels].
inline float post_input(device const float *x, device const float *route_weights,
                        constant PostConvParams &p, uint r, uint c) {
  if ((p.flags & 1u) == 0) { return x[r * p.x_stride + c]; }
  float sum = 0.0f;
  for (uint k = 0; k < p.routes; ++k) {
    sum += route_weights[r * p.routes + k] * x[(r * p.routes + k) * p.channels + c];
  }
  return sum;
}

inline float post_history(device const float *x, device const float *route_weights,
                          device const float *past, constant PostConvParams &p, uint row, uint c) {
  const uint history = p.width - 1;
  if (row < history) { return (p.flags & 4u) != 0 ? past[row * p.channels + c] : 0.0f; }
  return post_input(x, route_weights, p, row - history, c);
}

// A branch's residual short convolution with static taps, plus the residual
// stream (flag 2). Writes out [tokens][channels] and the next state
// [width - 1][channels]. Flag 4 = a past state (else zeros). Flag 8 = also
// write the un-convolved current input row to raw_out [tokens][channels]
// (used by speculative rollback so routed inputs need not be summed twice).
kernel void inkling_post_conv(
    device const float *x [[buffer(0)]],
    device const float *route_weights [[buffer(1)]],
    device const float *past [[buffer(2)]],
    device const float *weight [[buffer(3)]],
    device const float *residual [[buffer(4)]],
    device float *out [[buffer(5)]],
    device float *state [[buffer(6)]],
    device float *raw_out [[buffer(7)]],
    constant PostConvParams &p [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
  const uint history = p.width - 1;
  if (gid >= (p.tokens + history) * p.channels) { return; }
  const uint t = gid / p.channels;
  const uint c = gid - t * p.channels;
  if (t >= p.tokens) {
    const uint r = t - p.tokens;
    state[r * p.channels + c] = post_history(x, route_weights, past, p, p.tokens + r, c);
    return;
  }
  const float current = post_history(x, route_weights, past, p, t + history, c);
  if ((p.flags & 8u) != 0) {
    raw_out[t * p.channels + c] = current;
  }
  float acc = current;
  for (uint j = 0; j < p.width; ++j) {
    const float h = (j == history) ? current : post_history(x, route_weights, past, p, t + j, c);
    acc += weight[c * p.width + j] * h;
  }
  acc *= p.scale;
  if ((p.flags & 2u) != 0) { acc += residual[t * p.channels + c]; }
  out[t * p.channels + c] = acc;
}

struct ConvRollbackParams {
  uint kept;
  uint history;
  uint ch0; uint st0; uint has_p0;
  uint ch1; uint st1; uint has_p1;
  uint ch2; uint st2; uint has_p2;
  uint ch3; uint st3; uint has_p3;
};

// Rebuild all four convolution states of one visit from (past, input[0..kept))
// in a single launch. Slot s owns group.y == s; thread gid writes [r, c] of
// out_s [history][channels].
kernel void inkling_visit_conv_rollback(
    device const float *p0 [[buffer(0)]],
    device const float *i0 [[buffer(1)]],
    device const float *p1 [[buffer(2)]],
    device const float *i1 [[buffer(3)]],
    device const float *p2 [[buffer(4)]],
    device const float *i2 [[buffer(5)]],
    device const float *p3 [[buffer(6)]],
    device const float *i3 [[buffer(7)]],
    device float *o0 [[buffer(8)]],
    device float *o1 [[buffer(9)]],
    device float *o2 [[buffer(10)]],
    device float *o3 [[buffer(11)]],
    constant ConvRollbackParams &p [[buffer(12)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint slot = gid.y;
  device const float *past = slot == 0 ? p0 : (slot == 1 ? p1 : (slot == 2 ? p2 : p3));
  device const float *inp  = slot == 0 ? i0 : (slot == 1 ? i1 : (slot == 2 ? i2 : i3));
  device float *out        = slot == 0 ? o0 : (slot == 1 ? o1 : (slot == 2 ? o2 : o3));
  const uint ch     = slot == 0 ? p.ch0 : (slot == 1 ? p.ch1 : (slot == 2 ? p.ch2 : p.ch3));
  const uint stride = slot == 0 ? p.st0 : (slot == 1 ? p.st1 : (slot == 2 ? p.st2 : p.st3));
  const uint has_p  = slot == 0 ? p.has_p0 : (slot == 1 ? p.has_p1 : (slot == 2 ? p.has_p2 : p.has_p3));
  if (ch == 0 || gid.x >= p.history * ch) { return; }
  const uint r = gid.x / ch;
  const uint c = gid.x - r * ch;
  const uint src_row = p.kept + r;
  out[r * ch + c] = src_row < p.history
      ? (has_p != 0 ? past[src_row * ch + c] : 0.0f)
      : inp[(src_row - p.history) * stride + c];
}

struct GateNormParams { uint rows; uint hidden; float eps; uint groups; uint x_groups; uint flags; uint weight_rows; };

// out = rms((1 - tanh(f)) * x + tanh(g) * m) * weight[(r % groups) % weight_rows]
// for token-major row r (token r / groups), one threadgroup per row. fg is
// group-major [groups][tokens] when flag 1 is set; out is group-major when
// flag 2 is set; x has x_groups rows per token.
template <typename FgT>
kernel void inkling_gate_combine_norm_t(
    device const FgT *fg [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const float *m [[buffer(2)]],
    device const float *weight [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant GateNormParams &p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[32];
  const uint tokens = p.rows / p.groups;
  const uint group = row % p.groups;
  const uint token = row / p.groups;
  const uint fg_row = (p.flags & 1u) != 0 ? group * tokens + token : row;
  const uint out_row = (p.flags & 2u) != 0 ? group * tokens + token : row;
  const uint x_row = p.x_groups == p.groups ? row : token * p.x_groups;
  device const FgT *f = fg + fg_row * 2 * p.hidden;
  device const float *xr = x + x_row * p.hidden;
  device const float *mr = m + row * p.hidden;
  device float *o = out + out_row * p.hidden;
  float sumsq = 0.0f;
  if ((p.hidden & 3u) == 0u && p.hidden <= threads * 24u) {
    const uint n4 = p.hidden >> 2;
    device const vec<FgT, 4> *f4 = (device const vec<FgT, 4> *)f;
    device const float4 *x4 = (device const float4 *)xr;
    device const float4 *m4 = (device const float4 *)mr;
    device float4 *o4 = (device float4 *)o;
    float4 vals[6];
    uint k = 0;
    for (uint i = tid; i < n4; i += threads, ++k) {
      const float4 fv = float4(f4[i]);
      const float4 gv = float4(f4[n4 + i]);
      const float4 value = (1.0f - tanh(fv)) * x4[i] + tanh(gv) * m4[i];
      vals[k] = value;
      sumsq += dot(value, value);
    }
    sumsq = simd_sum(sumsq);
    if (lane == 0) { partial[sg] = sumsq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint g = 0; g < (threads + 31) / 32; ++g) { total += partial[g]; }
    const float inv = rsqrt(total / float(p.hidden) + p.eps);
    device const float4 *w4 = (device const float4 *)(weight + (group % p.weight_rows) * p.hidden);
    k = 0;
    for (uint i = tid; i < n4; i += threads, ++k) {
      o4[i] = vals[k] * inv * w4[i];
    }
    return;
  }
  if (p.hidden <= threads * 16) {
    float vals[16];
    uint k = 0;
    for (uint i = tid; i < p.hidden; i += threads, ++k) {
      const float value = (1.0f - tanh(float(f[i]))) * xr[i] + tanh(float(f[p.hidden + i])) * mr[i];
      vals[k] = value;
      sumsq += value * value;
    }
    sumsq = simd_sum(sumsq);
    if (lane == 0) { partial[sg] = sumsq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint g = 0; g < (threads + 31) / 32; ++g) { total += partial[g]; }
    const float inv = rsqrt(total / float(p.hidden) + p.eps);
    device const float *w = weight + (group % p.weight_rows) * p.hidden;
    k = 0;
    for (uint i = tid; i < p.hidden; i += threads, ++k) {
      o[i] = vals[k] * inv * w[i];
    }
    return;
  }
  for (uint i = tid; i < p.hidden; i += threads) {
    const float value = (1.0f - tanh(float(f[i]))) * xr[i] + tanh(float(f[p.hidden + i])) * mr[i];
    o[i] = value;
    sumsq += value * value;
  }
  sumsq = simd_sum(sumsq);
  if (lane == 0) { partial[sg] = sumsq; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint g = 0; g < (threads + 31) / 32; ++g) { total += partial[g]; }
  const float inv = rsqrt(total / float(p.hidden) + p.eps);
  device const float *w = weight + (group % p.weight_rows) * p.hidden;
  // Each thread rescales only the elements it wrote above.
  for (uint i = tid; i < p.hidden; i += threads) { o[i] *= inv * w[i]; }
}

template [[host_name("inkling_gate_combine_norm")]]
kernel decltype(inkling_gate_combine_norm_t<float>) inkling_gate_combine_norm_t<float>;
template [[host_name("inkling_gate_combine_norm_half")]]
kernel decltype(inkling_gate_combine_norm_t<half>) inkling_gate_combine_norm_t<half>;

struct RouteParams { uint tokens; uint experts; uint top_k; float scale; };

// Sigmoid router: experts are chosen by sigmoid(logit) + bias (lowest index
// on ties), weighted by their unbiased sigmoids normalized over the top k,
// times `scale`. One simdgroup per token; writes unique u32 IDs and F32
// weights [tokens][top_k]. At most 256 experts and 16 routes.
kernel void inkling_sigmoid_route(
    device const float *logits [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    device uint *ids [[buffer(2)]],
    device float *weights [[buffer(3)]],
    constant RouteParams &p [[buffer(4)]],
    uint token [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint SLOTS = 8;
  float scores[SLOTS];
  float probabilities[SLOTS];
  for (uint slot = 0; slot < SLOTS; ++slot) {
    const uint expert = lane + slot * 32;
    if (expert < p.experts) {
      const float probability = 1.0f / (1.0f + exp(-clamp(logits[token * p.experts + expert], -1.0e20f, 1.0e20f)));
      probabilities[slot] = probability;
      scores[slot] = probability + clamp(bias[expert], -1.0e20f, 1.0e20f);
    } else {
      probabilities[slot] = 0.0f;
      scores[slot] = NEG_LARGE;
    }
  }
  uint chosen[16];
  float chosen_probability[16];
  float total = 0.0f;
  for (uint route = 0; route < p.top_k; ++route) {
    float best = NEG_LARGE;
    uint best_id = 0xffffffffu;
    for (uint slot = 0; slot < SLOTS; ++slot) {
      const uint expert = lane + slot * 32;
      if (expert < p.experts && (scores[slot] > best || (scores[slot] == best && expert < best_id))) {
        best = scores[slot];
        best_id = expert;
      }
    }
    const float winner = simd_max(best);
    const uint id = simd_min(best == winner ? best_id : 0xffffffffu);
    float probability = 0.0f;
    for (uint slot = 0; slot < SLOTS; ++slot) {
      if (lane + slot * 32 == id) {
        probability = probabilities[slot];
        scores[slot] = 2.0f * NEG_LARGE;
      }
    }
    probability = simd_sum(probability);
    chosen[route] = id;
    chosen_probability[route] = probability;
    total += probability;
  }
  if (lane == 0) {
    const float factor = p.scale / max(total, 6.103515625e-5f);
    for (uint route = 0; route < p.top_k; ++route) {
      ids[token * p.top_k + route] = chosen[route];
      weights[token * p.top_k + route] = chosen_probability[route] * factor;
    }
  }
}


struct Q8RowParams { uint rows; uint hidden; };

struct Q8Embed1Params { uint rows; uint hidden; uint token_id; };

// Single-token Q8_0 embedding lookup without a host-to-device token-ID buffer
// allocation (token_id is passed via set_bytes). Produces bitwise-identical
// F32 values to `inkling_q8_rows` / CPU Q8_0 dequantization.
kernel void inkling_q8_embed_1(
    device const uchar *table [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant Q8Embed1Params &p [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  const uint id = min(p.token_id, p.rows - 1u);
  device const uchar *row = table + ulong(id) * (p.hidden / 32u) * 34u;
  for (uint i = tid; i < p.hidden; i += threads) {
    device const uchar *block = row + (i >> 5) * 34u;
    out[i] = float(*(device const half *)block) * float(((device const char *)block)[2u + (i & 31u)]);
  }
}

// out[t] = row ids[t] of a Q8_0 table (34-byte blocks: an f16 scale, then
// 32 int8 values), widened in f32 so each value is exact. One threadgroup
// per token.
kernel void inkling_q8_rows(
    device const uchar *table [[buffer(0)]],
    device const uint *ids [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant Q8RowParams &p [[buffer(3)]],
    uint token [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  const uint id = min(ids[token], p.rows - 1);
  device const uchar *row = table + ulong(id) * (p.hidden / 32) * 34;
  for (uint i = lane; i < p.hidden; i += threads) {
    device const uchar *block = row + (i / 32) * 34;
    const float scale = float(*(device const half *)block);
    out[ulong(token) * p.hidden + i] = scale * float(((device const char *)block)[2 + i % 32]);
  }
}

struct Bf16MvParams { uint in_dim; uint out_dim; float softcap; };

// y[t] = W x[t] for BF16 W [out, in] and F32 x [T, in], reading each weight
// once across all T tokens with exact F32 widening and decode's reduction
// order. Optional softcap (> 0) applies softcap * tanh(y / softcap) in place.
// Each of the NSG simdgroups makes R output rows.
template <uint T, uint R, uint NSG>
kernel void inkling_bf16_matvec_rows(
    device const bfloat *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant Bf16MvParams &p [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort simd [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  const uint first = (group.x * NSG + simd) * R;
  float sums[R][T];
  #pragma unroll
  for (uint r = 0; r < R; ++r) {
    #pragma unroll
    for (uint t = 0; t < T; ++t) {
      sums[r][t] = 0.0f;
    }
  }
  if ((p.in_dim & 255u) == 0u) {
    for (uint i = lane * 8; i < p.in_dim; i += 256) {
      float4 xv0[T], xv1[T];
      #pragma unroll
      for (uint t = 0; t < T; ++t) {
        device const float *xt = x + ulong(t) * p.in_dim + i;
        xv0[t] = *(device const float4 *)xt;
        xv1[t] = *(device const float4 *)(xt + 4);
      }
      #pragma unroll
      for (uint r = 0; r < R; ++r) {
        const uint o = first + r;
        if (o < p.out_dim) {
          device const bfloat *wo = w + ulong(o) * p.in_dim + i;
          const float4 wv0 = float4(*(device const bfloat4 *)wo);
          const float4 wv1 = float4(*(device const bfloat4 *)(wo + 4));
          #pragma unroll
          for (uint t = 0; t < T; ++t) {
            sums[r][t] += dot(wv0, xv0[t]) + dot(wv1, xv1[t]);
          }
        }
      }
    }
  } else {
    for (uint i = lane * 4; i < p.in_dim; i += 128) {
      float4 xv[T];
      #pragma unroll
      for (uint t = 0; t < T; ++t) {
        xv[t] = *(device const float4 *)(x + ulong(t) * p.in_dim + i);
      }
      #pragma unroll
      for (uint r = 0; r < R; ++r) {
        const uint o = first + r;
        if (o < p.out_dim) {
          const float4 wv = float4(*(device const bfloat4 *)(w + ulong(o) * p.in_dim + i));
          #pragma unroll
          for (uint t = 0; t < T; ++t) {
            sums[r][t] += dot(wv, xv[t]);
          }
        }
      }
    }
  }
  #pragma unroll
  for (uint r = 0; r < R; ++r) {
    const uint o = first + r;
    #pragma unroll
    for (uint t = 0; t < T; ++t) {
      const float total = simd_sum(sums[r][t]);
      if (lane == 0 && o < p.out_dim) {
        const float out_val = p.softcap > 0.0f
            ? (p.softcap * tanh(clamp(total / p.softcap, -15.0f, 15.0f)))
            : total;
        y[ulong(t) * p.out_dim + o] = out_val;
      }
    }
  }
}

struct Bf16MvManyParams {
  uint in_dim;
  uint out0; uint out1; uint out2;
  uint tiles0; uint tiles01;
};

// Up to three single-token BF16 projections of the same input width in one
// dispatch, eliminating per-projection launch barriers for Q/K/V.
kernel void inkling_bf16_matvec_many(
    device const bfloat *w0 [[buffer(0)]],
    device const float *x0 [[buffer(1)]],
    device const bfloat *w1 [[buffer(2)]],
    device const float *x1 [[buffer(3)]],
    device const bfloat *w2 [[buffer(4)]],
    device const float *x2 [[buffer(5)]],
    device float *y0 [[buffer(6)]],
    device float *y1 [[buffer(7)]],
    device float *y2 [[buffer(8)]],
    constant Bf16MvManyParams &p [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort simd [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  constexpr uint R = 4;
  constexpr uint NSG = 2;
  device const bfloat *w;
  device const float *x;
  device float *y;
  uint out_dim;
  uint tile;
  if (group.x < p.tiles0) {
    w = w0; x = x0; y = y0; out_dim = p.out0; tile = group.x;
  } else if (group.x < p.tiles01) {
    w = w1; x = x1; y = y1; out_dim = p.out1; tile = group.x - p.tiles0;
  } else {
    w = w2; x = x2; y = y2; out_dim = p.out2; tile = group.x - p.tiles01;
  }
  const uint first = (tile * NSG + simd) * R;
  float sums[R] = {};
  if ((p.in_dim & 255u) == 0u) {
    for (uint i = lane * 8; i < p.in_dim; i += 256) {
      const float4 xv0 = *(device const float4 *)(x + i);
      const float4 xv1 = *(device const float4 *)(x + i + 4);
      #pragma unroll
      for (uint r = 0; r < R; ++r) {
        const uint o = first + r;
        if (o < out_dim) {
          device const bfloat *wo = w + ulong(o) * p.in_dim + i;
          sums[r] += dot(float4(*(device const bfloat4 *)wo), xv0)
                   + dot(float4(*(device const bfloat4 *)(wo + 4)), xv1);
        }
      }
    }
  } else {
    for (uint i = lane * 4; i < p.in_dim; i += 128) {
      const float4 xv = *(device const float4 *)(x + i);
      #pragma unroll
      for (uint r = 0; r < R; ++r) {
        const uint o = first + r;
        if (o < out_dim) {
          sums[r] += dot(float4(*(device const bfloat4 *)(w + ulong(o) * p.in_dim + i)), xv);
        }
      }
    }
  }
  #pragma unroll
  for (uint r = 0; r < R; ++r) {
    const uint o = first + r;
    const float total = simd_sum(sums[r]);
    if (lane == 0 && o < out_dim) {
      y[o] = total;
    }
  }
}

inline float q4_0_half_dot(device const uchar *blk, float sumy, thread const float *yl, uint il) {
  const float d = float(*(device const half *)blk);
  const ushort4 qs = *(device const packed_ushort4 *)(blk + 2 + il);
  float2 acc = float2(0.0f);
  acc.x += yl[0] * float(qs.x & 0x000Fu) + yl[1] * float(qs.x & 0x0F00u)
         + yl[2] * float(qs.y & 0x000Fu) + yl[3] * float(qs.y & 0x0F00u)
         + yl[4] * float(qs.z & 0x000Fu) + yl[5] * float(qs.z & 0x0F00u)
         + yl[6] * float(qs.w & 0x000Fu) + yl[7] * float(qs.w & 0x0F00u);
  acc.y += yl[8]  * float(qs.x & 0x00F0u) + yl[9]  * float(qs.x & 0xF000u)
         + yl[10] * float(qs.y & 0x00F0u) + yl[11] * float(qs.y & 0xF000u)
         + yl[12] * float(qs.z & 0x00F0u) + yl[13] * float(qs.z & 0xF000u)
         + yl[14] * float(qs.w & 0x00F0u) + yl[15] * float(qs.w & 0xF000u);
  return d * (sumy * -8.0f + acc.x + acc.y);
}

inline void q4_0_run_half_sg(
    device const uchar *w,
    device const float *x,
    device float *y,
    uint in_dim,
    uint out_dim,
    uint first_row,
    ushort sub) {
  const uint nb = in_dim >> 5;
  const uint ix = sub >> 1;
  const uint il = (sub & 1u) << 3;
  float s0 = 0.0f;
  float s1 = 0.0f;
  device const uchar *r0 = first_row + 0u < out_dim ? w + ulong(first_row + 0u) * ulong(nb) * 18ul : w;
  device const uchar *r1 = first_row + 1u < out_dim ? w + ulong(first_row + 1u) * ulong(nb) * 18ul : w;
  for (uint ib = ix; ib < nb; ib += 8u) {
    device const float4 *xb = (device const float4 *)(x + (ib << 5) + il);
    const float4 a0 = xb[0];
    const float4 a1 = xb[1];
    const float4 b0 = xb[4];
    const float4 b1 = xb[5];
    const float sumy = ((a0.x + a0.y) + (a0.z + a0.w)) + ((a1.x + a1.y) + (a1.z + a1.w))
                     + ((b0.x + b0.y) + (b0.z + b0.w)) + ((b1.x + b1.y) + (b1.z + b1.w));
    const float yl[16] = {
      a0.x, a0.y * (1.0f / 256.0f), a0.z, a0.w * (1.0f / 256.0f),
      a1.x, a1.y * (1.0f / 256.0f), a1.z, a1.w * (1.0f / 256.0f),
      b0.x * (1.0f / 16.0f), b0.y * (1.0f / 4096.0f), b0.z * (1.0f / 16.0f), b0.w * (1.0f / 4096.0f),
      b1.x * (1.0f / 16.0f), b1.y * (1.0f / 4096.0f), b1.z * (1.0f / 16.0f), b1.w * (1.0f / 4096.0f),
    };
    const ulong boff = ulong(ib) * 18ul;
    if (first_row + 0u < out_dim) { s0 += q4_0_half_dot(r0 + boff, sumy, yl, il); }
    if (first_row + 1u < out_dim) { s1 += q4_0_half_dot(r1 + boff, sumy, yl, il); }
  }
  s0 += simd_shuffle_xor(s0, 8);
  s0 += simd_shuffle_xor(s0, 4);
  s0 += simd_shuffle_xor(s0, 2);
  s0 += simd_shuffle_xor(s0, 1);
  s1 += simd_shuffle_xor(s1, 8);
  s1 += simd_shuffle_xor(s1, 4);
  s1 += simd_shuffle_xor(s1, 2);
  s1 += simd_shuffle_xor(s1, 1);
  if (sub == 0) {
    if (first_row + 0u < out_dim) { y[first_row + 0u] = s0; }
    if (first_row + 1u < out_dim) { y[first_row + 1u] = s1; }
  }
}

// Single-token Q4_0 matvec: 16 lanes per 2 rows (4 rows/simdgroup, 2 simdgroups/TG
// = 8 rows/TG) so in_dim=2304 (72 blocks) and 3072 (96 blocks) run 9/12 exact
// iterations with zero warp tail and packed_ushort4 + float4 loads.
kernel void inkling_q4_0_matvec_1(
    device const uchar *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant Bf16MvParams &p [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort simd [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  const ushort sub = lane & 15u;
  const uint half_sg = lane >> 4;
  const uint first_row = (group.x * 2u + simd) * 4u + half_sg * 2u;
  q4_0_run_half_sg(w, x, y, p.in_dim, p.out_dim, first_row, sub);
}

// Up to three single-token Q4_0 projections of the same input width in one
// dispatch (8 rows/TG, 16 lanes per 2 rows).
kernel void inkling_q4_0_matvec_many(
    device const uchar *w0 [[buffer(0)]],
    device const float *x0 [[buffer(1)]],
    device const uchar *w1 [[buffer(2)]],
    device const float *x1 [[buffer(3)]],
    device const uchar *w2 [[buffer(4)]],
    device const float *x2 [[buffer(5)]],
    device float *y0 [[buffer(6)]],
    device float *y1 [[buffer(7)]],
    device float *y2 [[buffer(8)]],
    constant Bf16MvManyParams &p [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort simd [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  device const uchar *w;
  device const float *x;
  device float *y;
  uint out_dim;
  uint tile;
  if (group.x < p.tiles0) {
    w = w0; x = x0; y = y0; out_dim = p.out0; tile = group.x;
  } else if (group.x < p.tiles01) {
    w = w1; x = x1; y = y1; out_dim = p.out1; tile = group.x - p.tiles0;
  } else {
    w = w2; x = x2; y = y2; out_dim = p.out2; tile = group.x - p.tiles01;
  }
  const ushort sub = lane & 15u;
  const uint half_sg = lane >> 4;
  const uint first_row = (tile * 2u + simd) * 4u + half_sg * 2u;
  q4_0_run_half_sg(w, x, y, p.in_dim, out_dim, first_row, sub);
}

struct MvPostConv1Params {
  uint in_dim; uint out_dim; uint width; uint has_past; uint has_residual; float conv_scale;
};

inline void apply_post_conv_1(
    uint c,
    float current,
    device const float *past,
    device const float *conv_weight,
    device const float *residual,
    device float *out,
    device float *state,
    constant MvPostConv1Params &p) {
  const uint history = p.width - 1u;
  for (uint r = 0; r < history; ++r) {
    const float sv = (r + 1u < history)
        ? (p.has_past != 0 ? past[(r + 1u) * p.out_dim + c] : 0.0f)
        : current;
    state[r * p.out_dim + c] = sv;
  }
  float acc = current;
  for (uint j = 0; j < history; ++j) {
    const float h = p.has_past != 0 ? past[j * p.out_dim + c] : 0.0f;
    acc += conv_weight[c * p.width + j] * h;
  }
  acc += conv_weight[c * p.width + history] * current;
  acc *= p.conv_scale;
  if (p.has_residual != 0) { acc += residual[c]; }
  out[c] = acc;
}

// Single-token fused Q4_0 out_proj + static post_conv + optional residual.
kernel void inkling_q4_0_matvec_post_conv_1(
    device const uchar *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const float *past [[buffer(2)]],
    device const float *conv_weight [[buffer(3)]],
    device const float *residual [[buffer(4)]],
    device float *out [[buffer(5)]],
    device float *state [[buffer(6)]],
    constant MvPostConv1Params &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort simd [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  const ushort sub = lane & 15u;
  const uint half_sg = lane >> 4;
  const uint first_row = (group.x * 2u + simd) * 4u + half_sg * 2u;
  const uint nb = p.in_dim >> 5;
  const uint ix = sub >> 1;
  const uint il = (sub & 1u) << 3;
  float s0 = 0.0f;
  float s1 = 0.0f;
  device const uchar *r0 = first_row + 0u < p.out_dim ? w + ulong(first_row + 0u) * ulong(nb) * 18ul : w;
  device const uchar *r1 = first_row + 1u < p.out_dim ? w + ulong(first_row + 1u) * ulong(nb) * 18ul : w;
  for (uint ib = ix; ib < nb; ib += 8u) {
    device const float4 *xb = (device const float4 *)(x + (ib << 5) + il);
    const float4 a0 = xb[0];
    const float4 a1 = xb[1];
    const float4 b0 = xb[4];
    const float4 b1 = xb[5];
    const float sumy = ((a0.x + a0.y) + (a0.z + a0.w)) + ((a1.x + a1.y) + (a1.z + a1.w))
                     + ((b0.x + b0.y) + (b0.z + b0.w)) + ((b1.x + b1.y) + (b1.z + b1.w));
    const float yl[16] = {
      a0.x, a0.y * (1.0f / 256.0f), a0.z, a0.w * (1.0f / 256.0f),
      a1.x, a1.y * (1.0f / 256.0f), a1.z, a1.w * (1.0f / 256.0f),
      b0.x * (1.0f / 16.0f), b0.y * (1.0f / 4096.0f), b0.z * (1.0f / 16.0f), b0.w * (1.0f / 4096.0f),
      b1.x * (1.0f / 16.0f), b1.y * (1.0f / 4096.0f), b1.z * (1.0f / 16.0f), b1.w * (1.0f / 4096.0f),
    };
    const ulong boff = ulong(ib) * 18ul;
    if (first_row + 0u < p.out_dim) { s0 += q4_0_half_dot(r0 + boff, sumy, yl, il); }
    if (first_row + 1u < p.out_dim) { s1 += q4_0_half_dot(r1 + boff, sumy, yl, il); }
  }
  s0 += simd_shuffle_xor(s0, 8);
  s0 += simd_shuffle_xor(s0, 4);
  s0 += simd_shuffle_xor(s0, 2);
  s0 += simd_shuffle_xor(s0, 1);
  s1 += simd_shuffle_xor(s1, 8);
  s1 += simd_shuffle_xor(s1, 4);
  s1 += simd_shuffle_xor(s1, 2);
  s1 += simd_shuffle_xor(s1, 1);
  if (sub == 0) {
    if (first_row + 0u < p.out_dim) { apply_post_conv_1(first_row + 0u, s0, past, conv_weight, residual, out, state, p); }
    if (first_row + 1u < p.out_dim) { apply_post_conv_1(first_row + 1u, s1, past, conv_weight, residual, out, state, p); }
  }
}

// Single-token fused BF16 out_proj + static post_conv + optional residual.
kernel void inkling_bf16_matvec_post_conv_1(
    device const bfloat *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const float *past [[buffer(2)]],
    device const float *conv_weight [[buffer(3)]],
    device const float *residual [[buffer(4)]],
    device float *out [[buffer(5)]],
    device float *state [[buffer(6)]],
    constant MvPostConv1Params &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort simd [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  constexpr uint R = 4;
  constexpr uint NSG = 2;
  const uint first = (group.x * NSG + simd) * R;
  float sums[R] = {};
  if ((p.in_dim & 255u) == 0u) {
    for (uint i = lane * 8u; i < p.in_dim; i += 256u) {
      const float4 xv0 = *(device const float4 *)(x + i);
      const float4 xv1 = *(device const float4 *)(x + i + 4);
      #pragma unroll
      for (uint r = 0; r < R; ++r) {
        const uint o = first + r;
        if (o < p.out_dim) {
          device const bfloat *wo = w + ulong(o) * p.in_dim + i;
          sums[r] += dot(float4(*(device const bfloat4 *)wo), xv0)
                   + dot(float4(*(device const bfloat4 *)(wo + 4)), xv1);
        }
      }
    }
  } else {
    for (uint i = lane * 4u; i < p.in_dim; i += 128u) {
      const float4 xv = *(device const float4 *)(x + i);
      #pragma unroll
      for (uint r = 0; r < R; ++r) {
        const uint o = first + r;
        if (o < p.out_dim) {
          sums[r] += dot(float4(*(device const bfloat4 *)(w + ulong(o) * p.in_dim + i)), xv);
        }
      }
    }
  }
  #pragma unroll
  for (uint r = 0; r < R; ++r) {
    const uint o = first + r;
    const float total = simd_sum(sums[r]);
    if (lane == 0 && o < p.out_dim) {
      apply_post_conv_1(o, total, past, conv_weight, residual, out, state, p);
    }
  }
}

// Single-token 4-row-amortized Q8_0 matvec with optional fused softcap:
// 4 lanes per 32-weight block (8 blocks/iter = 9 exact iters on in_dim=2304),
// 4 rows/simdgroup sharing x loads, 2 simdgroups/TG = 8 rows/TG.
kernel void inkling_q8_0_matvec_1(
    device const uchar *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant Bf16MvParams &p [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort simd [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  constexpr uint R = 4;
  const uint first = (group.x * 2u + simd) * R;
  const uint nb = p.in_dim >> 5;
  const uint ib0 = lane >> 2;
  const uint sub = lane & 3u;
  device const uchar *ptrs[R];
  #pragma unroll
  for (uint r = 0; r < R; ++r) {
    ptrs[r] = first + r < p.out_dim ? w + ulong(first + r) * ulong(nb) * 34ul : w;
  }
  float sums[R] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint ib = ib0; ib < nb; ib += 8u) {
    device const float4 *x4 = (device const float4 *)(x + (ib << 5) + (sub << 3));
    const float4 xv0 = x4[0];
    const float4 xv1 = x4[1];
    const ulong boff = ulong(ib) * 34ul;
    #pragma unroll
    for (uint r = 0; r < R; ++r) {
      if (first + r < p.out_dim) {
        device const uchar *blk = ptrs[r] + boff;
        const float d = float(*(device const half *)blk);
        device const packed_char4 *qs = (device const packed_char4 *)(blk + 2 + (sub << 3));
        sums[r] += d * (dot(float4(char4(qs[0])), xv0) + dot(float4(char4(qs[1])), xv1));
      }
    }
  }
  #pragma unroll
  for (uint r = 0; r < R; ++r) {
    const float total = simd_sum(sums[r]);
    if (lane == 0 && first + r < p.out_dim) {
      y[first + r] = p.softcap > 0.0f
          ? (p.softcap * tanh(clamp(total / p.softcap, -15.0f, 15.0f)))
          : total;
    }
  }
}

typedef decltype(inkling_bf16_matvec_rows<1, 4, 2>) inkling_bf16_mv_t;
template [[host_name("inkling_bf16_matvec_1")]] kernel inkling_bf16_mv_t inkling_bf16_matvec_rows<1, 4, 2>;
template [[host_name("inkling_bf16_matvec_2")]] kernel inkling_bf16_mv_t inkling_bf16_matvec_rows<2, 4, 2>;
template [[host_name("inkling_bf16_matvec_3")]] kernel inkling_bf16_mv_t inkling_bf16_matvec_rows<3, 4, 2>;
template [[host_name("inkling_bf16_matvec_4")]] kernel inkling_bf16_mv_t inkling_bf16_matvec_rows<4, 4, 2>;
template [[host_name("inkling_bf16_matvec_5")]] kernel inkling_bf16_mv_t inkling_bf16_matvec_rows<5, 2, 4>;
template [[host_name("inkling_bf16_matvec_6")]] kernel inkling_bf16_mv_t inkling_bf16_matvec_rows<6, 2, 4>;
template [[host_name("inkling_bf16_matvec_7")]] kernel inkling_bf16_mv_t inkling_bf16_matvec_rows<7, 2, 4>;
template [[host_name("inkling_bf16_matvec_8")]] kernel inkling_bf16_mv_t inkling_bf16_matvec_rows<8, 2, 4>;

struct F32MvParams { uint batch; uint in_dim; uint out_dim; };

// Batched y[b, t, o] = sum_k x[b, t, k] * W[b, o, k] for F32/F16 W [B, out, in]
// and x [B, T, in] (in % 4 == 0). One simdgroup per (o, b) reads W once for
// all T rows, keeping 64..192 threadgroups active on skinny router/gate banks.
template <uint T, typename WT>
kernel void inkling_f32_matvec_t_rows(
    device const WT *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant F32MvParams &p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
  const uint o = group.x;
  const uint b = group.y;
  if (o >= p.out_dim || b >= p.batch) { return; }
  device const WT *wb = w + (ulong(b) * p.out_dim + o) * p.in_dim;
  device const float *xb = x + ulong(b) * T * p.in_dim;
  float sums[T];
  #pragma unroll
  for (uint t = 0; t < T; ++t) { sums[t] = 0.0f; }
  for (uint i = lane * 4; i < p.in_dim; i += 128) {
    const float4 wv = float4(*(device const vec<WT, 4> *)(wb + i));
    #pragma unroll
    for (uint t = 0; t < T; ++t) {
      const float4 xv = *(device const float4 *)(xb + ulong(t) * p.in_dim + i);
      sums[t] += dot(wv, xv);
    }
  }
  device float *yb = y + ulong(b) * T * p.out_dim;
  #pragma unroll
  for (uint t = 0; t < T; ++t) {
    const float total = simd_sum(sums[t]);
    if (lane == 0) {
      yb[ulong(t) * p.out_dim + o] = total;
    }
  }
}

// Single-token split-K version of inkling_f32_matvec_t_rows: NSG=6 simdgroups
// (192 threads, stride 768) cooperate on each (o, b) output row so in_dim=2304
// (3 exact iters) and 4608 (6 exact iters) have zero warp tail.
template <typename WT>
kernel void inkling_f32_matvec_t_1_splitk(
    device const WT *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant F32MvParams &p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  constexpr uint NSG = 6;
  threadgroup float partial[NSG];
  const uint o = group.x;
  const uint b = group.y;
  if (o >= p.out_dim || b >= p.batch) { return; }
  device const WT *wb = w + (ulong(b) * p.out_dim + o) * p.in_dim;
  device const float *xb = x + ulong(b) * p.in_dim;
  float sum = 0.0f;
  const uint tid = sg * 32u + lane;
  for (uint i = tid * 4u; i < p.in_dim; i += NSG * 128u) {
    const float4 wv = float4(*(device const vec<WT, 4> *)(wb + i));
    const float4 xv = *(device const float4 *)(xb + i);
    sum += dot(wv, xv);
  }
  sum = simd_sum(sum);
  if (lane == 0) { partial[sg] = sum; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float v = lane < NSG ? partial[lane] : 0.0f;
    v = simd_sum(v);
    if (lane == 0) {
      y[ulong(b) * p.out_dim + o] = v;
    }
  }
}

template [[host_name("inkling_f32_mv_t_1")]]
kernel decltype(inkling_f32_matvec_t_1_splitk<float>) inkling_f32_matvec_t_1_splitk<float>;
template [[host_name("inkling_f16_mv_t_1")]]
kernel decltype(inkling_f32_matvec_t_1_splitk<half>) inkling_f32_matvec_t_1_splitk<half>;

typedef decltype(inkling_f32_matvec_t_rows<2, float>) inkling_f32_mv_t_fn;
template [[host_name("inkling_f32_mv_t_2")]] kernel inkling_f32_mv_t_fn inkling_f32_matvec_t_rows<2, float>;
template [[host_name("inkling_f32_mv_t_3")]] kernel inkling_f32_mv_t_fn inkling_f32_matvec_t_rows<3, float>;
template [[host_name("inkling_f32_mv_t_4")]] kernel inkling_f32_mv_t_fn inkling_f32_matvec_t_rows<4, float>;
template [[host_name("inkling_f32_mv_t_5")]] kernel inkling_f32_mv_t_fn inkling_f32_matvec_t_rows<5, float>;
template [[host_name("inkling_f32_mv_t_6")]] kernel inkling_f32_mv_t_fn inkling_f32_matvec_t_rows<6, float>;
template [[host_name("inkling_f32_mv_t_7")]] kernel inkling_f32_mv_t_fn inkling_f32_matvec_t_rows<7, float>;
template [[host_name("inkling_f32_mv_t_8")]] kernel inkling_f32_mv_t_fn inkling_f32_matvec_t_rows<8, float>;

typedef decltype(inkling_f32_matvec_t_rows<2, half>) inkling_f16_mv_t_fn;
template [[host_name("inkling_f16_mv_t_2")]] kernel inkling_f16_mv_t_fn inkling_f32_matvec_t_rows<2, half>;
template [[host_name("inkling_f16_mv_t_3")]] kernel inkling_f16_mv_t_fn inkling_f32_matvec_t_rows<3, half>;
template [[host_name("inkling_f16_mv_t_4")]] kernel inkling_f16_mv_t_fn inkling_f32_matvec_t_rows<4, half>;
template [[host_name("inkling_f16_mv_t_5")]] kernel inkling_f16_mv_t_fn inkling_f32_matvec_t_rows<5, half>;
template [[host_name("inkling_f16_mv_t_6")]] kernel inkling_f16_mv_t_fn inkling_f32_matvec_t_rows<6, half>;
template [[host_name("inkling_f16_mv_t_7")]] kernel inkling_f16_mv_t_fn inkling_f32_matvec_t_rows<7, half>;
template [[host_name("inkling_f16_mv_t_8")]] kernel inkling_f16_mv_t_fn inkling_f32_matvec_t_rows<8, half>;

// Batched y[b, t, o] = sum_k x[b, t, k] * W[b, k, o] for F32/F16 W [B, K, out]
// and x [B, T, K] with K <= 64 (loop-gate forget_gain). 64 threads per group
// stage x in threadgroup memory and stream W coalesced once across all T rows.
template <uint T, typename WT>
kernel void inkling_f32_matvec_nt_rows(
    device const WT *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant F32MvParams &p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup float sx[T][64];
  const uint b = group.y;
  const uint o = group.x * 64 + tid;
  device const float *xb = x + ulong(b) * T * p.in_dim;
  if (tid < p.in_dim) {
    #pragma unroll
    for (uint t = 0; t < T; ++t) {
      sx[t][tid] = xb[ulong(t) * p.in_dim + tid];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (o >= p.out_dim) { return; }
  device const WT *wb = w + ulong(b) * p.in_dim * p.out_dim + o;
  float sums[T];
  #pragma unroll
  for (uint t = 0; t < T; ++t) { sums[t] = 0.0f; }
  for (uint k = 0; k < p.in_dim; ++k) {
    const float wv = float(wb[ulong(k) * p.out_dim]);
    #pragma unroll
    for (uint t = 0; t < T; ++t) {
      sums[t] += sx[t][k] * wv;
    }
  }
  device float *yb = y + ulong(b) * T * p.out_dim;
  #pragma unroll
  for (uint t = 0; t < T; ++t) {
    yb[ulong(t) * p.out_dim + o] = sums[t];
  }
}

// Single-token split-K float4 version of inkling_f32_matvec_nt_rows: 4
// simdgroups (128 threads) cooperate on 128 output columns (float4 per lane),
// splitting K across simdgroups for 4x parallelism and 128-bit weight loads.
template <typename WT>
kernel void inkling_f32_matvec_nt_1_splitk(
    device const WT *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant F32MvParams &p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  constexpr uint NSG = 4;
  threadgroup float sx[64];
  threadgroup float4 partial[NSG][32];
  const uint b = group.y;
  device const float *xb = x + ulong(b) * p.in_dim;
  if (tid < p.in_dim) {
    sx[tid] = xb[tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint o4 = (group.x * 32u + lane) * 4u;
  float4 sum = float4(0.0f);
  if (o4 < p.out_dim) {
    device const WT *wb = w + ulong(b) * p.in_dim * p.out_dim + o4;
    for (uint k = sg; k < p.in_dim; k += NSG) {
      const float4 wv = float4(*(device const vec<WT, 4> *)(wb + ulong(k) * p.out_dim));
      sum += sx[k] * wv;
    }
  }
  partial[sg][lane] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0 && o4 < p.out_dim) {
    const float4 total = (partial[0][lane] + partial[1][lane]) + (partial[2][lane] + partial[3][lane]);
    *(device float4 *)(y + ulong(b) * p.out_dim + o4) = total;
  }
}

template [[host_name("inkling_f32_mv_nt_1_v4")]]
kernel decltype(inkling_f32_matvec_nt_1_splitk<float>) inkling_f32_matvec_nt_1_splitk<float>;
template [[host_name("inkling_f16_mv_nt_1_v4")]]
kernel decltype(inkling_f32_matvec_nt_1_splitk<half>) inkling_f32_matvec_nt_1_splitk<half>;

typedef decltype(inkling_f32_matvec_nt_rows<1, float>) inkling_f32_mv_nt_fn;
template [[host_name("inkling_f32_mv_nt_1")]] kernel inkling_f32_mv_nt_fn inkling_f32_matvec_nt_rows<1, float>;
template [[host_name("inkling_f32_mv_nt_2")]] kernel inkling_f32_mv_nt_fn inkling_f32_matvec_nt_rows<2, float>;
template [[host_name("inkling_f32_mv_nt_3")]] kernel inkling_f32_mv_nt_fn inkling_f32_matvec_nt_rows<3, float>;
template [[host_name("inkling_f32_mv_nt_4")]] kernel inkling_f32_mv_nt_fn inkling_f32_matvec_nt_rows<4, float>;
template [[host_name("inkling_f32_mv_nt_5")]] kernel inkling_f32_mv_nt_fn inkling_f32_matvec_nt_rows<5, float>;
template [[host_name("inkling_f32_mv_nt_6")]] kernel inkling_f32_mv_nt_fn inkling_f32_matvec_nt_rows<6, float>;
template [[host_name("inkling_f32_mv_nt_7")]] kernel inkling_f32_mv_nt_fn inkling_f32_matvec_nt_rows<7, float>;
template [[host_name("inkling_f32_mv_nt_8")]] kernel inkling_f32_mv_nt_fn inkling_f32_matvec_nt_rows<8, float>;

typedef decltype(inkling_f32_matvec_nt_rows<1, half>) inkling_f16_mv_nt_fn;
template [[host_name("inkling_f16_mv_nt_1")]] kernel inkling_f16_mv_nt_fn inkling_f32_matvec_nt_rows<1, half>;
template [[host_name("inkling_f16_mv_nt_2")]] kernel inkling_f16_mv_nt_fn inkling_f32_matvec_nt_rows<2, half>;
template [[host_name("inkling_f16_mv_nt_3")]] kernel inkling_f16_mv_nt_fn inkling_f32_matvec_nt_rows<3, half>;
template [[host_name("inkling_f16_mv_nt_4")]] kernel inkling_f16_mv_nt_fn inkling_f32_matvec_nt_rows<4, half>;
template [[host_name("inkling_f16_mv_nt_5")]] kernel inkling_f16_mv_nt_fn inkling_f32_matvec_nt_rows<5, half>;
template [[host_name("inkling_f16_mv_nt_6")]] kernel inkling_f16_mv_nt_fn inkling_f32_matvec_nt_rows<6, half>;
template [[host_name("inkling_f16_mv_nt_7")]] kernel inkling_f16_mv_nt_fn inkling_f32_matvec_nt_rows<7, half>;
template [[host_name("inkling_f16_mv_nt_8")]] kernel inkling_f16_mv_nt_fn inkling_f32_matvec_nt_rows<8, half>;

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
        .new_library_with_source(INKLING_MSL, None)
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

fn f32_input<'a>(
    storage: &'a MetalStorage,
    layout: &Layout,
    what: &str,
) -> Result<(&'a Buffer, usize)> {
    if storage.dtype() != DType::F32 || !layout.is_contiguous() {
        candle_core::bail!("Inkling 2 {what} must be contiguous f32")
    }
    Ok((storage.buffer(), layout.start_offset() * 4))
}

fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| candle_core::Error::Msg(format!("Inkling 2 {what} too large")))
}

/// Encode one kernel over `inputs` into a new F32 output of `elements`.
#[allow(clippy::too_many_arguments)]
fn launch<P>(
    device: &candle_core::MetalDevice,
    name: &'static str,
    inputs: &[(&Buffer, usize)],
    elements: usize,
    params: &P,
    groups: (usize, usize, usize),
    threads: usize,
    shape: Shape,
) -> Result<(MetalStorage, Shape)> {
    let output = alloc_scratch_buffer(device, elements.max(1) * DType::F32.size_in_bytes())?;
    let pipeline = pipeline(device, name)?;
    let threads = if threads == 0 {
        pipeline.max_total_threads_per_threadgroup().clamp(32, 1024) / 32 * 32
    } else {
        threads
    };
    let encoder = device.command_encoder()?;
    encoder.set_label(name);
    encoder.set_compute_pipeline_state(&pipeline);
    let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
    for (index, (buffer, offset)) in inputs.iter().enumerate() {
        encoder_ref.set_input_buffer(index, Some(buffer), *offset);
    }
    encoder_ref.set_output_buffer(inputs.len(), Some(&output), 0);
    encoder_ref.set_bytes(inputs.len() + 1, params);
    encoder_ref.dispatch_thread_groups(
        objc2_metal::MTLSize {
            width: groups.0,
            height: groups.1,
            depth: groups.2,
        },
        objc2_metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    drop(encoder);
    Ok((
        MetalStorage::new(output, device.clone(), elements, DType::F32),
        shape,
    ))
}

/// The Metal buffer and byte offset of a contiguous F32 tensor.
fn tensor_buffer(tensor: &Tensor, what: &str) -> Result<(Buffer, usize)> {
    if tensor.dtype() != DType::F32 || !tensor.is_contiguous() {
        candle_core::bail!("Inkling 2 {what} must be contiguous f32")
    }
    let (storage, layout) = tensor.storage_and_layout();
    let candle_core::Storage::Metal(storage) = &*storage else {
        candle_core::bail!("Inkling 2 {what} must be on Metal")
    };
    Ok((storage.buffer().clone(), layout.start_offset() * 4))
}

fn metal_only<T>() -> Result<T> {
    candle_core::bail!("Inkling 2 fused kernels run on Metal only")
}

/// The buffer, byte offset and row stride (elements) of a 2-D F32 tensor
/// whose rows are contiguous but may be a slice of wider rows.
fn f32_rows<'a>(
    storage: &'a MetalStorage,
    layout: &Layout,
    what: &str,
) -> Result<(&'a Buffer, usize, usize)> {
    let stride = layout.stride();
    let (rows, width) = layout.shape().dims2()?;
    if storage.dtype() != DType::F32 || stride[1] != 1 || (rows > 1 && stride[0] < width) {
        candle_core::bail!("Inkling 2 {what} must be f32 with contiguous rows")
    }
    Ok((
        storage.buffer(),
        layout.start_offset() * 4,
        if rows > 1 { stride[0] } else { width },
    ))
}

/// Most block states and queries one depth-mix call supports (eight
/// sources including the current row).
pub(super) const DEPTH_MIX_MAX_BLOCKS: usize = 7;
pub(super) const DEPTH_MIX_MAX_QUERIES: usize = 4;

#[repr(C)]
struct RowParams {
    rows: u32,
    hidden: u32,
    eps: f32,
    groups: u32,
    x_groups: u32,
    flags: u32,
}

/// `RowParams` flag: the output is group-major.
const OUTPUT_GROUP_MAJOR: u32 = 2;

/// Rows per token and shared `x` rows per token; see `RowParams` in the MSL.
#[derive(Clone, Copy)]
struct Groups {
    groups: usize,
    x_groups: usize,
}

impl Groups {
    const NONE: Self = Self {
        groups: 1,
        x_groups: 1,
    };

    /// The `x` row count matching `rows` grouped rows.
    fn x_rows(self, rows: usize) -> usize {
        rows / self.groups * self.x_groups
    }
}

struct RmsPair {
    eps: f32,
    groups: Groups,
    flags: u32,
}

impl CustomOp2 for RmsPair {
    fn name(&self) -> &'static str {
        "inkling2-rms-pair"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        metal_only()
    }
    fn metal_fwd(
        &self,
        x: &MetalStorage,
        xl: &Layout,
        m: &MetalStorage,
        ml: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let hidden = *ml.shape().dims().last().unwrap_or(&0);
        let rows = ml.shape().elem_count() / hidden.max(1);
        if hidden == 0
            || !rows.is_multiple_of(self.groups.groups)
            || xl.shape().elem_count() != self.groups.x_rows(rows) * hidden
        {
            candle_core::bail!("Inkling 2 gate inputs differ in shape")
        }
        let params = RowParams {
            rows: to_u32(rows, "gate rows")?,
            hidden: to_u32(hidden, "gate width")?,
            eps: self.eps,
            groups: to_u32(self.groups.groups, "gate groups")?,
            x_groups: to_u32(self.groups.x_groups, "gate groups")?,
            flags: self.flags,
        };
        launch(
            x.device(),
            "inkling_rms_pair",
            &[
                f32_input(x, xl, "gate input")?,
                f32_input(m, ml, "gate message")?,
            ],
            rows * 2 * hidden,
            &params,
            (rows, 1, 1),
            256,
            Shape::from((rows, 2 * hidden)),
        )
    }
}

/// `[rms(x) | rms(message)]` per row.
pub(super) fn rms_pair(x: &Tensor, message: &Tensor, eps: f32) -> Result<Tensor> {
    x.apply_op2_no_bwd(
        message,
        &RmsPair {
            eps,
            groups: Groups::NONE,
            flags: 0,
        },
    )
}

/// `[rms(x) | rms(message)]` per row, written in F16 for prompt loop gates.
pub(super) fn rms_pair_half(x: &Tensor, message: &Tensor, eps: f32) -> Result<Tensor> {
    rms_pair_half_grouped(x, message, eps, Groups::NONE, 0)
}

fn rms_pair_half_grouped(
    x: &Tensor,
    message: &Tensor,
    eps: f32,
    groups: Groups,
    flags: u32,
) -> Result<Tensor> {
    let device = metal_device(x)?;
    let hidden = *message.dims().last().unwrap_or(&0);
    let rows = message.elem_count() / hidden.max(1);
    if hidden == 0
        || !rows.is_multiple_of(groups.groups)
        || x.elem_count() != groups.x_rows(rows) * hidden
    {
        candle_core::bail!("Inkling 2 gate inputs differ in shape")
    }
    let kept = (held(x)?, held(message)?);
    let xb = tensor_buffer(&kept.0, "gate input")?;
    let mb = tensor_buffer(&kept.1, "gate message")?;
    let params = RowParams {
        rows: to_u32(rows, "gate rows")?,
        hidden: to_u32(hidden, "gate width")?,
        eps,
        groups: to_u32(groups.groups, "gate groups")?,
        x_groups: to_u32(groups.x_groups, "gate groups")?,
        flags,
    };
    let out = alloc_scratch_buffer(&device, rows * 2 * hidden * DType::F16.size_in_bytes())?;
    encode(
        &device,
        "inkling_rms_pair_half",
        &[(&xb.0, xb.1), (&mb.0, mb.1)],
        &[(&out, 0)],
        &params,
        (rows, 1, 1),
        256,
    )?;
    Ok(wrap_output(&device, out, DType::F16, (rows, 2 * hidden)))
}

/// The Q, K and V loop gates for `x` `[tokens, 1 or 3, hidden]` (one input
/// shared by Q, K and V, or one each) and `messages` `[tokens, 3, hidden]`,
/// with `down` and `forget_gain` `[3, rank, 2 * hidden]`, each gated row then
/// RMS-normalized and times `norm` row `g` (`[3, hidden]`, or `[1, hidden]`
/// for all). `eps` serves both norms. Returns group-major `[3, tokens, hidden]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn message_gates(
    x: &Tensor,
    messages: &Tensor,
    down: &Tensor,
    forget_gain: &Tensor,
    down_f16: Option<&Tensor>,
    forget_gain_f16: Option<&Tensor>,
    eps: f32,
    norm: &Tensor,
) -> Result<Tensor> {
    let (tokens, x_groups, hidden) = x.dims3()?;
    let groups = Groups {
        groups: 3,
        x_groups,
    };
    if tokens > 8
        && let (Some(down_h), Some(fg_h)) = (down_f16, forget_gain_f16)
    {
        let joined = rms_pair_half_grouped(x, messages, eps, groups, OUTPUT_GROUP_MAJOR)?
            .reshape((3, tokens, 2 * hidden))?;
        if let Some(reduced) =
            super::inkling2_tensor_metal::try_f16_rows_times_transposed(&joined, down_h)?
            && let Some(projected) =
                super::inkling2_tensor_metal::try_f16_rows_times_nt(&reduced, fg_h)?
        {
            return gate_combine_norm(&projected, x, messages, norm, 3, eps, true, true);
        }
    }
    // Group-major throughout, so each gate's rows are one batch.
    let joined = x
        .apply_op2_no_bwd(
            messages,
            &RmsPair {
                eps,
                groups,
                flags: OUTPUT_GROUP_MAJOR,
            },
        )?
        .reshape((3, tokens, 2 * hidden))?;
    let use_f16_mv = tokens == 1 && hidden >= 256;
    let down_w = if use_f16_mv {
        down_f16.unwrap_or(down)
    } else {
        down
    };
    let fg_w = if use_f16_mv {
        forget_gain_f16.unwrap_or(forget_gain)
    } else {
        forget_gain
    };
    let reduced = if (1..=8).contains(&tokens)
        && let Some(reduced) = try_f32_matvec_t_rows(&joined, down_w)?
    {
        reduced
    } else if tokens > 8
        && let Some(reduced) =
            super::inkling2_tensor_metal::try_rows_times_transposed(&joined, down)?
    {
        reduced
    } else {
        joined.matmul(&down.transpose(1, 2)?)?
    };
    let projected = if (1..=8).contains(&tokens)
        && let Some(projected) = try_f32_matvec_nt_rows(&reduced, fg_w)?
    {
        projected
    } else if tokens > 8
        && let Some(projected) =
            super::inkling2_tensor_metal::try_rows_times_nt(&reduced, forget_gain)?
    {
        projected
    } else {
        reduced.matmul(forget_gain)?
    };
    gate_combine_norm(&projected, x, messages, norm, 3, eps, true, true)
}

#[repr(C)]
struct GateTransposeParams {
    tokens: u32,
    heads: u32,
    dim: u32,
    gate_stride: u32,
}

struct GateTranspose;

impl CustomOp2 for GateTranspose {
    fn name(&self) -> &'static str {
        "inkling2-gate-transpose"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        metal_only()
    }
    fn metal_fwd(
        &self,
        a: &MetalStorage,
        al: &Layout,
        g: &MetalStorage,
        gl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (heads, tokens, dim) = al.shape().dims3()?;
        let (gate_buffer, gate_offset, gate_stride) = f32_rows(g, gl, "attention gate")?;
        if gl.shape().dims() != [tokens, heads * dim] {
            candle_core::bail!("Inkling 2 gate transpose geometry differs")
        }
        let params = GateTransposeParams {
            tokens: to_u32(tokens, "gate tokens")?,
            heads: to_u32(heads, "gate heads")?,
            dim: to_u32(dim, "gate width")?,
            gate_stride: to_u32(gate_stride, "gate stride")?,
        };
        let elements = tokens * heads * dim;
        launch(
            a.device(),
            "inkling_gate_transpose",
            &[
                f32_input(a, al, "attention output")?,
                (gate_buffer, gate_offset),
            ],
            elements,
            &params,
            (elements.div_ceil(256), 1, 1),
            256,
            Shape::from((tokens, heads * dim)),
        )
    }
}

/// `[heads, tokens, dim]` attention output to `[tokens, heads * dim]`,
/// multiplied by the sigmoid of `gate` (`[tokens, heads * dim]` logits,
/// whose rows may be slices of wider rows).
pub(super) fn gate_transpose(attended: &Tensor, gate: &Tensor) -> Result<Tensor> {
    attended.apply_op2_no_bwd(gate, &GateTranspose)
}

/// [`gate_transpose`] writing F16 directly for BF16 output projections.
pub(super) fn gate_transpose_f16(attended: &Tensor, gate: &Tensor) -> Result<Tensor> {
    let device = metal_device(attended)?;
    let (heads, tokens, dim) = attended.dims3()?;
    if gate.dims() != [tokens, heads * dim] {
        candle_core::bail!("Inkling 2 gate transpose geometry differs")
    }
    let kept = held(attended)?;
    let att = tensor_buffer(&kept, "attention output")?;
    let gate_view = row_view(gate, "attention gate")?;
    let params = GateTransposeParams {
        tokens: to_u32(tokens, "gate tokens")?,
        heads: to_u32(heads, "gate heads")?,
        dim: to_u32(dim, "gate width")?,
        gate_stride: to_u32(gate_view.2, "gate stride")?,
    };
    let elements = tokens * heads * dim;
    let out = alloc_scratch_buffer(&device, elements * DType::F16.size_in_bytes())?;
    encode(
        &device,
        "inkling_gate_transpose_f16",
        &[(&att.0, att.1), (&gate_view.0, gate_view.1)],
        &[(&out, 0)],
        &params,
        (elements.div_ceil(256), 1, 1),
        256,
    )?;
    Ok(wrap_output(&device, out, DType::F16, (tokens, heads * dim)))
}

#[repr(C)]
struct SituParams {
    rows: u32,
    ffn: u32,
    beta: f32,
    linear_beta: f32,
}

struct Situ {
    beta: f32,
    linear_beta: f32,
}

impl CustomOp1 for Situ {
    fn name(&self) -> &'static str {
        "inkling2-situ"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        metal_only()
    }
    fn metal_fwd(&self, gate_up: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let dims = layout.shape().dims();
        let width = *dims.last().unwrap_or(&0);
        if width == 0 || !width.is_multiple_of(2) {
            candle_core::bail!("Inkling 2 SiTU input must have an even last dimension")
        }
        let ffn = width / 2;
        let rows = layout.shape().elem_count() / width;
        let params = SituParams {
            rows: to_u32(rows, "SiTU rows")?,
            ffn: to_u32(ffn, "SiTU width")?,
            beta: self.beta,
            linear_beta: self.linear_beta,
        };
        let mut shape = dims.to_vec();
        *shape.last_mut().unwrap() = ffn;
        let elements = rows * ffn;
        launch(
            gate_up.device(),
            "inkling_situ",
            &[f32_input(gate_up, layout, "SiTU input")?],
            elements,
            &params,
            (elements.div_ceil(256), 1, 1),
            256,
            Shape::from(shape),
        )
    }
}

/// SiTU over packed `[..., gate | up]` rows.
pub(super) fn situ(gate_up: &Tensor, beta: f32, linear_beta: f32) -> Result<Tensor> {
    gate_up.apply_op1_no_bwd(&Situ { beta, linear_beta })
}

#[repr(C)]
struct AttnParams {
    tokens: u32,
    heads: u32,
    kv_heads: u32,
    keys: u32,
    head_dim: u32,
    extent: u32,
    window: u32,
    finalize: u32,
    kv_head_stride: u32,
    first_query: i32,
    first_key: i32,
    splits: u32,
    chunk: u32,
    has_gate: u32,
}

/// Keys per attention split; more splits keep the GPU busy for one query.
const KEYS_PER_SPLIT: usize = 64;
/// Most splits per query and head.
const MAX_SPLITS: usize = 1024;
/// Threadgroups the first pass aims for.
const TARGET_THREADGROUPS: usize = 4096;

struct Attention {
    /// `[tokens, heads, extent + 2]` F32: the bias by distance, then the sink
    /// logit.
    bias: Tensor,
    head_dim: usize,
    extent: usize,
    window: usize,
    first_query: usize,
    first_key: usize,
    /// `[tokens, heads * head_dim]` F32 gate logits applied to the output.
    gate: Option<Tensor>,
}

/// Largest query count the fused attention handles. Every query of a chunk
/// reads the same keys, which stay in the GPU caches.
pub(super) const FUSED_ATTENTION_MAX_TOKENS: usize = 2048;

impl CustomOp3 for Attention {
    fn name(&self) -> &'static str {
        "inkling2-attention"
    }
    #[allow(clippy::too_many_arguments)]
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        metal_only()
    }
    #[allow(clippy::too_many_arguments)]
    fn metal_fwd(
        &self,
        q: &MetalStorage,
        ql: &Layout,
        k: &MetalStorage,
        kl: &Layout,
        v: &MetalStorage,
        vl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (tokens, heads, q_dim) = ql.shape().dims3()?;
        let (batch, kv_heads, keys, head_dim) = kl.shape().dims4()?;
        let kv_strides = kl.stride();
        if batch != 1
            || vl.shape() != kl.shape()
            || vl.stride() != kv_strides
            || vl.start_offset() != kl.start_offset()
            || k.dtype() != DType::F16
            || v.dtype() != DType::F16
            || head_dim != self.head_dim
            || !matches!(head_dim, 64 | 128 | 256)
            || kv_heads == 0
            || heads % kv_heads != 0
            || q_dim != head_dim
            || self.bias.dims() != [tokens, heads, self.extent + 2]
            || kv_strides[3] != 1
            || kv_strides[2] != head_dim
            || keys == 0
        {
            candle_core::bail!("Inkling 2 fused attention geometry differs")
        }
        // For 1 token or short/sliding caches (<= 2048 keys), the higher-occupancy
        // per-head kernel wins; for multi-token verification on longer full-attention
        // contexts, sharing KV reads across the 3 GQA heads and 2-3 tokens cuts DRAM
        // reads by 6x-9x.
        let use_gqa3 = head_dim == 128 && heads == kv_heads * 3 && tokens >= 2 && keys > 2048;
        let (partial_name, groups_y, groups_z, partial_threads) = if use_gqa3 {
            if tokens == 1 {
                ("inkling_attention_partial_gqa3_128_t1", kv_heads, 1, 256)
            } else if tokens == 3 || tokens == 6 {
                (
                    "inkling_attention_partial_gqa3_128_t3",
                    kv_heads,
                    tokens.div_ceil(3),
                    128,
                )
            } else {
                (
                    "inkling_attention_partial_gqa3_128_t2",
                    kv_heads,
                    tokens.div_ceil(2),
                    128,
                )
            }
        } else {
            let name = match head_dim {
                64 => "inkling_attention_partial_64",
                128 => "inkling_attention_partial_128",
                _ => "inkling_attention_partial_256",
            };
            (name, heads, tokens, 256)
        };
        let combine_name = match head_dim {
            64 => "inkling_attention_combine_64",
            128 => "inkling_attention_combine_128",
            _ => "inkling_attention_combine_256",
        };
        // Enough splits to fill the GPU for a few queries, few for long
        // chunks. A single split also finishes the output, with no merge.
        let wanted = (TARGET_THREADGROUPS / (groups_y * groups_z)).max(1);
        let splits = keys
            .div_ceil(KEYS_PER_SPLIT)
            .min(wanted)
            .clamp(1, MAX_SPLITS);
        let chunk = keys.div_ceil(splits);
        let splits = keys.div_ceil(chunk);
        let params = AttnParams {
            tokens: to_u32(tokens, "attention tokens")?,
            heads: to_u32(heads, "attention heads")?,
            kv_heads: to_u32(kv_heads, "attention kv heads")?,
            keys: to_u32(keys, "attention keys")?,
            head_dim: to_u32(head_dim, "attention head width")?,
            extent: to_u32(self.extent, "attention extent")?,
            window: to_u32(self.window, "attention window")?,
            finalize: u32::from(splits == 1),
            kv_head_stride: to_u32(kv_strides[1], "attention kv stride")?,
            first_query: i32::try_from(self.first_query)
                .map_err(|_| candle_core::Error::Msg("Inkling 2 position too large".into()))?,
            first_key: i32::try_from(self.first_key)
                .map_err(|_| candle_core::Error::Msg("Inkling 2 position too large".into()))?,
            splits: to_u32(splits, "attention splits")?,
            chunk: to_u32(chunk, "attention split width")?,
            has_gate: u32::from(self.gate.is_some()),
        };
        let device = q.device();
        let bias = tensor_buffer(&self.bias, "attention bias")?;
        let gate = match &self.gate {
            Some(gate) => {
                if gate.dims() != [tokens, heads * head_dim] {
                    candle_core::bail!("Inkling 2 attention gate geometry differs")
                }
                Some(tensor_buffer(gate, "attention gate")?)
            }
            None => None,
        };
        let gate_input = match &gate {
            Some((buffer, offset)) => (buffer, *offset),
            None => (&bias.0, bias.1),
        };
        let inputs = [
            f32_input(q, ql, "attention queries")?,
            (k.buffer(), kl.start_offset() * 2),
            (v.buffer(), vl.start_offset() * 2),
            (&bias.0, bias.1),
            gate_input,
        ];
        if splits == 1 {
            return launch(
                device,
                partial_name,
                &inputs,
                tokens * heads * head_dim,
                &params,
                (1, groups_y, groups_z),
                partial_threads,
                Shape::from((tokens, heads, head_dim)),
            );
        }
        let (partial, _) = launch(
            device,
            partial_name,
            &inputs,
            tokens * heads * splits * (head_dim + 2),
            &params,
            (splits, groups_y, groups_z),
            partial_threads,
            Shape::from((tokens, heads, splits, head_dim + 2)),
        )?;
        launch(
            device,
            combine_name,
            &[(&bias.0, bias.1), (partial.buffer(), 0), gate_input],
            tokens * heads * head_dim,
            &params,
            (heads, tokens, 1),
            32,
            Shape::from((tokens, heads, head_dim)),
        )
    }
}

/// Attention for a few queries straight from the f16 cache.
///
/// `q` is `[tokens, heads, head_dim]`, already scaled; `bias` is
/// `[tokens, heads, extent + 2]`: the bias for each distance (column `extent`
/// is every distance past it), then the sink logit. `k`, `v` are the active
/// `[1, kv_heads, keys, head_dim]` f16 cache views, whose first key sits at
/// absolute position `first_key`. With `gate` (`[tokens, heads * head_dim]`
/// logits), the output is multiplied by its sigmoid. Returns
/// `[tokens, heads, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn attention(
    q: &Tensor,
    bias: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    extent: usize,
    window: Option<usize>,
    first_query: usize,
    first_key: usize,
    gate: Option<&Tensor>,
) -> Result<Tensor> {
    q.apply_op3_no_bwd(
        k,
        v,
        &Attention {
            bias: bias.contiguous()?,
            head_dim,
            extent,
            window: window.unwrap_or(0),
            first_query,
            first_key,
            gate: gate.map(Tensor::contiguous).transpose()?,
        },
    )
}

// ---- Single-pass launches. These bypass candle's custom ops, which take at
// most three inputs and return one output.

/// `tensor` as contiguous rows. Keep the result alive until the kernel is
/// encoded: a dropped copy's buffer can come back from candle's pool as the
/// same kernel's output.
fn held(tensor: &Tensor) -> Result<Tensor> {
    tensor.contiguous()
}

/// The Metal buffer and byte offset of a contiguous f16 tensor.
fn f16_buffer(tensor: &Tensor, what: &str) -> Result<(Buffer, usize)> {
    if tensor.dtype() != DType::F16 || !tensor.is_contiguous() {
        candle_core::bail!("Inkling 2 {what} must be contiguous f16")
    }
    let (storage, layout) = tensor.storage_and_layout();
    let candle_core::Storage::Metal(storage) = &*storage else {
        candle_core::bail!("Inkling 2 {what} must be on Metal")
    };
    Ok((storage.buffer().clone(), layout.start_offset() * 2))
}

fn metal_device(tensor: &Tensor) -> Result<candle_core::MetalDevice> {
    match tensor.device() {
        candle_core::Device::Metal(device) => Ok(device.clone()),
        _ => metal_only(),
    }
}

/// The buffer, byte offset and row stride of a 2-D F32 tensor whose rows are
/// contiguous but may be a slice of wider rows.
fn row_view(tensor: &Tensor, what: &str) -> Result<(Buffer, usize, usize)> {
    let (storage, layout) = tensor.storage_and_layout();
    let candle_core::Storage::Metal(storage) = &*storage else {
        candle_core::bail!("Inkling 2 {what} must be on Metal")
    };
    let (buffer, offset, stride) = f32_rows(storage, layout, what)?;
    Ok((buffer.clone(), offset, stride))
}

const SCRATCH_POOL_MAX_BYTES: usize = 8 * 1024 * 1024;
const SCRATCH_POOL_SMALL_BUCKET_BYTES: usize = 64 * 1024;
const SCRATCH_POOL_LARGE_BUCKET_BYTES: usize = 2 * 1024 * 1024;
const SCRATCH_POOL_SMALL_MAX_PER_BUCKET: usize = 512;
const SCRATCH_POOL_DEFAULT_MAX_PER_BUCKET: usize = 256;
const SCRATCH_POOL_LARGE_MAX_PER_BUCKET: usize = 16;

fn scratch_pool_bucket_cap(bucket: usize) -> usize {
    if bucket <= SCRATCH_POOL_SMALL_BUCKET_BYTES {
        SCRATCH_POOL_SMALL_MAX_PER_BUCKET
    } else if bucket > SCRATCH_POOL_LARGE_BUCKET_BYTES {
        SCRATCH_POOL_LARGE_MAX_PER_BUCKET
    } else {
        SCRATCH_POOL_DEFAULT_MAX_PER_BUCKET
    }
}

/// Persistent Metal scratch buffer pool that survives `wait_until_completed()`
/// (unlike Candle's `BufferMap`, which purges all cached buffers on every sync).
/// Allocates directly on `MTLDevice` (`StorageModeShared | HazardTrackingModeUntracked`)
/// so `Arc::strong_count == 1` accurately reflects pool-only ownership. Safe under
/// concurrent Metal execution because `candle-metal-kernels` tracks RAW/WAW/WAR
/// within encoders and inserts fence waits at encoder boundaries.
pub(crate) fn alloc_scratch_buffer(
    device: &candle_core::MetalDevice,
    bytes: usize,
) -> Result<std::sync::Arc<Buffer>> {
    use objc2_metal::MTLResourceOptions as Options;
    let bucket = bytes.max(1).next_power_of_two();
    if bucket <= SCRATCH_POOL_MAX_BYTES {
        static POOL: OnceLock<Mutex<HashMap<(DeviceId, usize), Vec<std::sync::Arc<Buffer>>>>> =
            OnceLock::new();
        let pool = POOL.get_or_init(|| Mutex::new(HashMap::new()));
        let key = (device.id(), bucket);
        {
            let map = pool.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(list) = map.get(&key) {
                if let Some(buf) = list.iter().find(|b| std::sync::Arc::strong_count(b) == 1) {
                    return Ok(buf.clone());
                }
            }
        }
        let options =
            Options(Options::StorageModeShared.0 | Options::HazardTrackingModeUntracked.0);
        let buf = std::sync::Arc::new(
            device
                .device()
                .new_buffer(bucket, options)
                .map_err(candle_core::Error::wrap)?,
        );
        let mut map = pool.lock().unwrap_or_else(|e| e.into_inner());
        let list = map.entry(key).or_default();
        if list.len() < scratch_pool_bucket_cap(bucket) {
            list.push(buf.clone());
        }
        return Ok(buf);
    }
    device.new_buffer_builder().with_size(bytes.max(1)).build()
}

fn new_output(
    device: &candle_core::MetalDevice,
    elements: usize,
    _label: &str,
) -> Result<std::sync::Arc<Buffer>> {
    alloc_scratch_buffer(device, elements.max(1) * DType::F32.size_in_bytes())
}

fn wrap_output(
    device: &candle_core::MetalDevice,
    buffer: std::sync::Arc<Buffer>,
    dtype: DType,
    shape: impl Into<Shape>,
) -> Tensor {
    let shape = shape.into();
    let elements = shape.elem_count();
    Tensor::from((
        candle_core::Storage::Metal(MetalStorage::new(buffer, device.clone(), elements, dtype)),
        shape,
    ))
}

/// Encode `name` with `inputs`, then `outputs` (buffer, byte offset), then
/// `params` as buffers in that order. `threads` 0 means the pipeline's largest threadgroup.
fn encode<P>(
    device: &candle_core::MetalDevice,
    name: &'static str,
    inputs: &[(&Buffer, usize)],
    outputs: &[(&Buffer, usize)],
    params: &P,
    groups: (usize, usize, usize),
    threads: usize,
) -> Result<()> {
    let pipeline = pipeline(device, name)?;
    let threads = if threads == 0 {
        pipeline.max_total_threads_per_threadgroup().clamp(32, 1024) / 32 * 32
    } else {
        threads
    };
    let encoder = device.command_encoder()?;
    encoder.set_label(name);
    encoder.set_compute_pipeline_state(&pipeline);
    let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
    for (index, (buffer, offset)) in inputs.iter().enumerate() {
        encoder_ref.set_input_buffer(index, Some(buffer), *offset);
    }
    for (index, (buffer, offset)) in outputs.iter().enumerate() {
        encoder_ref.set_output_buffer(inputs.len() + index, Some(buffer), *offset);
    }
    encoder_ref.set_bytes(inputs.len() + outputs.len(), params);
    encoder_ref.dispatch_thread_groups(
        objc2_metal::MTLSize {
            width: groups.0,
            height: groups.1,
            depth: groups.2,
        },
        objc2_metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Threads per depth-mix threadgroup: decode measured faster with 512 than
/// with 256 or 1,024 (fewer partial sums to merge, enough loads in flight).
const DEPTH_MIX_THREADS: usize = 512;

/// `inkling_depth_mix_one<sources, queries>` by `[blocks - 1][queries - 1]`.
const DEPTH_MIX_ONE_KERNELS: [[&str; 4]; 7] = [
    [
        "inkling_depth_mix_one_2_1",
        "inkling_depth_mix_one_2_2",
        "inkling_depth_mix_one_2_3",
        "inkling_depth_mix_one_2_4",
    ],
    [
        "inkling_depth_mix_one_3_1",
        "inkling_depth_mix_one_3_2",
        "inkling_depth_mix_one_3_3",
        "inkling_depth_mix_one_3_4",
    ],
    [
        "inkling_depth_mix_one_4_1",
        "inkling_depth_mix_one_4_2",
        "inkling_depth_mix_one_4_3",
        "inkling_depth_mix_one_4_4",
    ],
    [
        "inkling_depth_mix_one_5_1",
        "inkling_depth_mix_one_5_2",
        "inkling_depth_mix_one_5_3",
        "inkling_depth_mix_one_5_4",
    ],
    [
        "inkling_depth_mix_one_6_1",
        "inkling_depth_mix_one_6_2",
        "inkling_depth_mix_one_6_3",
        "inkling_depth_mix_one_6_4",
    ],
    [
        "inkling_depth_mix_one_7_1",
        "inkling_depth_mix_one_7_2",
        "inkling_depth_mix_one_7_3",
        "inkling_depth_mix_one_7_4",
    ],
    [
        "inkling_depth_mix_one_8_1",
        "inkling_depth_mix_one_8_2",
        "inkling_depth_mix_one_8_3",
        "inkling_depth_mix_one_8_4",
    ],
];

#[repr(C)]
struct MixOneParams {
    tokens: u32,
    blocks: u32,
    hidden: u32,
    queries: u32,
    eps: f32,
    scale: f32,
    norm_eps: f32,
    flags: u32,
}

/// [`depth_mix`] in one dispatch, reading each block in place: `blocks` are
/// `[tokens, hidden]` each, `current` `[tokens, hidden]`, `queries`
/// `[queries, hidden]`. With `norm` (`weight` `[queries or 1, hidden]`, eps),
/// each output row is RMS-normalized and weighted. Returns `[tokens, queries,
/// hidden]`, or `[queries, tokens, hidden]` when `group_major`.
pub(super) fn depth_mix_one(
    blocks: &[Tensor],
    current: &Tensor,
    queries: &Tensor,
    eps: f32,
    norm: Option<(&Tensor, f32)>,
    group_major: bool,
) -> Result<Tensor> {
    let device = metal_device(current)?;
    let (tokens, hidden) = current.dims2()?;
    let (query_count, query_hidden) = queries.dims2()?;
    if blocks.is_empty()
        || blocks.len() > DEPTH_MIX_MAX_BLOCKS
        || query_count == 0
        || query_count > DEPTH_MIX_MAX_QUERIES
        || query_hidden != hidden
        || blocks.iter().any(|block| block.dims() != [tokens, hidden])
    {
        candle_core::bail!("Inkling 2 depth-mix geometry differs")
    }
    let kept = [current, queries]
        .into_iter()
        .chain(blocks)
        .chain(norm.map(|(weight, _)| weight))
        .map(held)
        .collect::<Result<Vec<_>>>()?;
    let current = tensor_buffer(&kept[0], "mix current")?;
    let queries = tensor_buffer(&kept[1], "mix queries")?;
    let mut sources = kept[2..2 + blocks.len()]
        .iter()
        .map(|block| tensor_buffer(block, "mix block"))
        .collect::<Result<Vec<_>>>()?;
    sources.resize(8, current.clone());
    let (norm_buffer, norm_eps, flags) = match norm {
        Some((weight, norm_eps)) => {
            let rows = weight.dim(0)?;
            if weight.dims2()?.1 != hidden || (rows != 1 && rows != query_count) {
                candle_core::bail!("Inkling 2 depth-mix norm geometry differs")
            }
            let shared = u32::from(rows == 1 && query_count > 1) << 1;
            let weight = kept.last().expect("held norm weight");
            (tensor_buffer(weight, "mix norm")?, norm_eps, 1 | shared)
        }
        None => (current.clone(), 0.0, 0),
    };
    let flags = flags | (u32::from(group_major) << 2);
    let params = MixOneParams {
        tokens: to_u32(tokens, "mix tokens")?,
        blocks: to_u32(blocks.len(), "mix blocks")?,
        hidden: to_u32(hidden, "mix width")?,
        queries: to_u32(query_count, "mix queries")?,
        eps,
        scale: 1.0 / (hidden as f32).sqrt(),
        norm_eps,
        flags,
    };
    let elements = tokens * query_count * hidden;
    let name = DEPTH_MIX_ONE_KERNELS[blocks.len() - 1][query_count - 1];
    let output = new_output(&device, elements, name)?;
    let mut inputs = sources
        .iter()
        .map(|(buffer, offset)| (buffer, *offset))
        .collect::<Vec<_>>();
    inputs.push((&queries.0, queries.1));
    inputs.push((&norm_buffer.0, norm_buffer.1));
    encode(
        &device,
        name,
        &inputs,
        &[(&output, 0)],
        &params,
        (tokens, 1, 1),
        DEPTH_MIX_THREADS,
    )?;
    let shape = if group_major {
        Shape::from((query_count, tokens, hidden))
    } else {
        Shape::from((tokens, query_count, hidden))
    };
    Ok(wrap_output(&device, output, DType::F32, shape))
}

#[repr(C)]
struct KvFinishParams {
    tokens: u32,
    heads: u32,
    dim: u32,
    width: u32,
    k_stride: u32,
    v_stride: u32,
    k_taps_stride: u32,
    v_taps_stride: u32,
    scale: f32,
    clip: f32,
    eps: f32,
    flags: u32,
    capacity: u32,
    row: u32,
}

/// Where [`kv_finish`] writes the keys and values.
pub(super) enum KvOutput<'a> {
    /// New F32 tensors.
    Fresh,
    /// In place, as f16, into cache storage `[1, heads, capacity, dim]` from
    /// `row` on.
    Cache {
        k: &'a Tensor,
        v: &'a Tensor,
        row: usize,
    },
}

/// Keys and values ready for the cache, and the next conv states.
pub(super) struct KvFinished {
    /// `[1, heads, tokens, dim]`, L2-clipped per head (fresh output only).
    pub(super) k: Option<Tensor>,
    /// `[1, heads, tokens, dim]`, RMS-normalized per head (fresh output only).
    pub(super) v: Option<Tensor>,
    /// `[width - 1, heads * dim]` each.
    pub(super) k_state: Tensor,
    pub(super) v_state: Tensor,
}

/// The K and V short convolutions (`k_lin`, `v_lin` `[tokens, heads * dim]`
/// rows, after `past` `[width - 1, heads * dim]` states or zeros), with
/// static `weight`s `[heads * dim, width]` and optional per-token `taps`
/// `[tokens, heads * width]`, then K clipped to an L2 norm of `clip` and V
/// RMS-normalized per head, in one dispatch, written to `output`.
#[allow(clippy::too_many_arguments)]
pub(super) fn kv_finish(
    k_lin: &Tensor,
    v_lin: &Tensor,
    past: Option<(&Tensor, &Tensor)>,
    k_weight: &Tensor,
    v_weight: &Tensor,
    taps: Option<(&Tensor, &Tensor)>,
    heads: usize,
    clip: f32,
    eps: f32,
    output: KvOutput<'_>,
) -> Result<KvFinished> {
    let device = metal_device(k_lin)?;
    let (tokens, channels) = k_lin.dims2()?;
    let (weight_channels, width) = k_weight.dims2()?;
    let dim = channels / heads.max(1);
    if heads == 0
        || heads * dim != channels
        || dim > 1024
        || width < 2
        || v_lin.dims() != [tokens, channels]
        || weight_channels != channels
        || v_weight.dims() != [channels, width]
        || past.is_some_and(|(k, v)| {
            k.dims() != [width - 1, channels] || v.dims() != [width - 1, channels]
        })
        || taps.is_some_and(|(k, v)| {
            k.dims() != [tokens, heads * width] || v.dims() != [tokens, heads * width]
        })
    {
        candle_core::bail!("Inkling 2 key/value finish geometry differs")
    }
    let k_in = row_view(k_lin, "key input")?;
    let v_in = row_view(v_lin, "value input")?;
    let past_held = past
        .map(|(k, v)| Ok::<_, candle_core::Error>((held(k)?, held(v)?)))
        .transpose()?;
    let past = past_held
        .as_ref()
        .map(|(k, v)| {
            Ok::<_, candle_core::Error>((
                tensor_buffer(k, "key state")?,
                tensor_buffer(v, "value state")?,
            ))
        })
        .transpose()?;
    let weights_held = (held(k_weight)?, held(v_weight)?);
    let k_weight = tensor_buffer(&weights_held.0, "key conv weight")?;
    let v_weight = tensor_buffer(&weights_held.1, "value conv weight")?;
    let taps = taps
        .map(|(k, v)| {
            Ok::<_, candle_core::Error>((row_view(k, "key taps")?, row_view(v, "value taps")?))
        })
        .transpose()?;
    let terms = width + if taps.is_some() { width } else { 0 };
    let params = KvFinishParams {
        tokens: to_u32(tokens, "key/value tokens")?,
        heads: to_u32(heads, "key/value heads")?,
        dim: to_u32(dim, "key/value head width")?,
        width: to_u32(width, "conv width")?,
        k_stride: to_u32(k_in.2, "key stride")?,
        v_stride: to_u32(v_in.2, "value stride")?,
        k_taps_stride: taps
            .as_ref()
            .map_or(Ok(0), |t| to_u32(t.0.2, "tap stride"))?,
        v_taps_stride: taps
            .as_ref()
            .map_or(Ok(0), |t| to_u32(t.1.2, "tap stride"))?,
        scale: 1.0 / ((terms + 1) as f32).sqrt(),
        clip,
        eps,
        flags: u32::from(taps.is_some()) | (u32::from(past.is_some()) << 1),
        capacity: 0,
        row: 0,
    };
    let mut params = params;
    let caches = match output {
        KvOutput::Fresh => None,
        KvOutput::Cache { k, v, row } => {
            let (batch, cache_heads, capacity, cache_dim) = k.dims4()?;
            if batch != 1
                || (cache_heads, cache_dim) != (heads, dim)
                || v.dims() != k.dims()
                || k.dtype() != DType::F16
                || v.dtype() != DType::F16
                || !k.is_contiguous()
                || !v.is_contiguous()
                || row + tokens > capacity
            {
                candle_core::bail!("Inkling 2 key/value cache geometry differs")
            }
            params.flags |= 4;
            params.capacity = to_u32(capacity, "cache capacity")?;
            params.row = to_u32(row, "cache row")?;
            Some((f16_buffer(k, "key cache")?, f16_buffer(v, "value cache")?))
        }
    };
    let history = width - 1;
    let main_elements = if caches.is_some() {
        1
    } else {
        2 * tokens * channels
    };
    let main = new_output(&device, main_elements, "inkling_kv_finish")?;
    let state = new_output(&device, 2 * history * channels, "inkling_kv_state")?;
    let (k_past, v_past) = match &past {
        Some((k, v)) => ((&k.0, k.1), (&v.0, v.1)),
        None => ((&k_in.0, k_in.1), (&v_in.0, v_in.1)),
    };
    let (k_taps, v_taps) = match &taps {
        Some((k, v)) => ((&k.0, k.1), (&v.0, v.1)),
        None => ((&k_in.0, k_in.1), (&v_in.0, v_in.1)),
    };
    encode(
        &device,
        "inkling_kv_finish",
        &[
            (&k_in.0, k_in.1),
            (&v_in.0, v_in.1),
            k_past,
            v_past,
            (&k_weight.0, k_weight.1),
            (&v_weight.0, v_weight.1),
            k_taps,
            v_taps,
        ],
        &[
            (&main, 0),
            (&state, 0),
            caches.as_ref().map_or((&main, 0), |c| (&c.0.0, c.0.1)),
            caches.as_ref().map_or((&main, 0), |c| (&c.1.0, c.1.1)),
        ],
        &params,
        (heads, tokens.max(history), 2),
        dim.div_ceil(32) * 32,
    )?;
    let state = wrap_output(&device, state, DType::F32, (2, history, channels));
    let (k, v) = match caches {
        Some(_) => (None, None),
        None => {
            let main = wrap_output(&device, main, DType::F32, (2, 1, heads, tokens, dim));
            (Some(main.get(0)?), Some(main.get(1)?))
        }
    };
    Ok(KvFinished {
        k,
        v,
        k_state: state.get(0)?,
        v_state: state.get(1)?,
    })
}

/// Tokens per query-preparation threadgroup (`QUERY_PREP_TOKENS` in MSL).
const QUERY_PREP_TOKENS: usize = 8;

#[repr(C)]
struct QueryPrepParams {
    tokens: u32,
    heads: u32,
    dim: u32,
    rel: u32,
    extent: u32,
    x_stride: u32,
    q_offset: u32,
    rel_offset: u32,
    sink_offset: u32,
    q_clip: f32,
    rel_clip: f32,
    q_post: f32,
    flags: u32,
    first_position: u32,
}

/// Column offsets of the queries, relative coefficients and sink logits in
/// the stacked query-side projection's rows.
pub(super) struct QuerySide {
    pub(super) q: usize,
    pub(super) rel: usize,
    pub(super) sink: usize,
}

/// From the stacked query-side projection `x` `[tokens, width]`: the queries
/// times `q_scale` `[heads, dim]`, clipped to an L2 norm of `q_clip`, times
/// `token_scale` (if any) and `q_post` (`[tokens, heads, dim]`); and the
/// attention bias `[tokens, heads, extent + 2]`, the relative coefficients
/// times `rel_scale` `[heads, rel]`, clipped to `rel_clip` and times
/// `token_scale`, multiplied into `basis` `[rel, extent + 1]`, then the sink.
#[allow(clippy::too_many_arguments, dead_code)]
pub(super) fn query_prep(
    x: &Tensor,
    columns: &QuerySide,
    heads: usize,
    dim: usize,
    q_scale: &Tensor,
    rel_scale: &Tensor,
    token_scale: Option<&Tensor>,
    basis: &Tensor,
    q_clip: f32,
    rel_clip: f32,
    q_post: f32,
) -> Result<(Tensor, Tensor)> {
    query_prep_from(
        x,
        columns,
        heads,
        dim,
        q_scale,
        rel_scale,
        token_scale,
        basis,
        q_clip,
        rel_clip,
        q_post,
        usize::MAX,
    )
}

/// [`query_prep`] for a chunk whose first token has at most `first_position`
/// keys before it: token `t`'s bias row is written up to distance
/// `first_position + t` only, the furthest back a key of its can lie. The
/// columns past it hold whatever the buffer held; a short prompt needs a
/// fraction of the table, which is most of this kernel's output.
#[allow(clippy::too_many_arguments)]
pub(super) fn query_prep_from(
    x: &Tensor,
    columns: &QuerySide,
    heads: usize,
    dim: usize,
    q_scale: &Tensor,
    rel_scale: &Tensor,
    token_scale: Option<&Tensor>,
    basis: &Tensor,
    q_clip: f32,
    rel_clip: f32,
    q_post: f32,
    first_position: usize,
) -> Result<(Tensor, Tensor)> {
    query_prep_from_inner(
        x,
        columns,
        heads,
        dim,
        q_scale,
        rel_scale,
        token_scale,
        basis,
        q_clip,
        rel_clip,
        q_post,
        first_position,
        false,
    )
}

/// [`query_prep_from`] writing both queries and attention bias in F16.
#[allow(clippy::too_many_arguments)]
pub(super) fn query_prep_from_f16(
    x: &Tensor,
    columns: &QuerySide,
    heads: usize,
    dim: usize,
    q_scale: &Tensor,
    rel_scale: &Tensor,
    token_scale: Option<&Tensor>,
    basis: &Tensor,
    q_clip: f32,
    rel_clip: f32,
    q_post: f32,
    first_position: usize,
) -> Result<(Tensor, Tensor)> {
    query_prep_from_inner(
        x,
        columns,
        heads,
        dim,
        q_scale,
        rel_scale,
        token_scale,
        basis,
        q_clip,
        rel_clip,
        q_post,
        first_position,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn query_prep_from_inner(
    x: &Tensor,
    columns: &QuerySide,
    heads: usize,
    dim: usize,
    q_scale: &Tensor,
    rel_scale: &Tensor,
    token_scale: Option<&Tensor>,
    basis: &Tensor,
    q_clip: f32,
    rel_clip: f32,
    q_post: f32,
    first_position: usize,
    output_f16: bool,
) -> Result<(Tensor, Tensor)> {
    let device = metal_device(x)?;
    let (tokens, width) = x.dims2()?;
    let (rel, columns_plus) = basis.dims2()?;
    let extent = columns_plus.saturating_sub(1);
    if columns_plus == 0
        || rel == 0
        || rel > 32
        || dim > 1024
        || q_scale.dims() != [heads, dim]
        || rel_scale.dims() != [heads, rel]
        || token_scale.is_some_and(|t| t.elem_count() != tokens)
        || columns.q + heads * dim > width
        || columns.rel + heads * rel > width
        || columns.sink + heads > width
    {
        candle_core::bail!("Inkling 2 query preparation geometry differs")
    }
    let rows = row_view(x, "query-side rows")?;
    let kept = (held(q_scale)?, held(rel_scale)?, held(basis)?);
    let token_held = token_scale.map(held).transpose()?;
    let q_scale = tensor_buffer(&kept.0, "query scale")?;
    let rel_scale = tensor_buffer(&kept.1, "relative scale")?;
    let basis = tensor_buffer(&kept.2, "relative basis")?;
    let token_scale = token_held
        .as_ref()
        .map(|t| tensor_buffer(t, "token scale"))
        .transpose()?;
    let params = QueryPrepParams {
        tokens: to_u32(tokens, "query tokens")?,
        heads: to_u32(heads, "query heads")?,
        dim: to_u32(dim, "query head width")?,
        rel: to_u32(rel, "relative width")?,
        extent: to_u32(extent, "relative extent")?,
        x_stride: to_u32(rows.2, "query-side stride")?,
        q_offset: to_u32(columns.q, "query column")?,
        rel_offset: to_u32(columns.rel, "relative column")?,
        sink_offset: to_u32(columns.sink, "sink column")?,
        q_clip,
        rel_clip,
        q_post,
        flags: u32::from(token_scale.is_some()),
        first_position: u32::try_from(first_position.min(extent)).unwrap_or(u32::MAX),
    };
    // One simdgroup per token of a block; see the kernel.
    let threads = 32 * QUERY_PREP_TOKENS;
    // The last token reaches the furthest column.
    let columns_written = extent.min(first_position.saturating_add(tokens - 1)) + 1;
    let name = match (rel <= 16, output_f16) {
        (true, false) => "inkling_query_prep_16",
        (false, false) => "inkling_query_prep_32",
        (true, true) => "inkling_query_prep_16_f16",
        (false, true) => "inkling_query_prep_32_f16",
    };
    let out_dtype = if output_f16 { DType::F16 } else { DType::F32 };
    let q_out = alloc_scratch_buffer(&device, tokens * heads * dim * out_dtype.size_in_bytes())?;
    let bias = alloc_scratch_buffer(
        &device,
        tokens * heads * (extent + 2) * out_dtype.size_in_bytes(),
    )?;
    let token_scale = token_scale
        .as_ref()
        .map_or((&rows.0, rows.1), |t| (&t.0, t.1));
    encode(
        &device,
        name,
        &[
            (&rows.0, rows.1),
            (&q_scale.0, q_scale.1),
            (&rel_scale.0, rel_scale.1),
            token_scale,
            (&basis.0, basis.1),
        ],
        &[(&q_out, 0), (&bias, 0)],
        &params,
        (
            heads,
            tokens.div_ceil(QUERY_PREP_TOKENS),
            columns_written.div_ceil(threads),
        ),
        threads,
    )?;
    Ok((
        wrap_output(&device, q_out, out_dtype, (tokens, heads, dim)),
        wrap_output(&device, bias, out_dtype, (tokens, heads, extent + 2)),
    ))
}

#[repr(C)]
struct SoftcapParams {
    count: u32,
    cap: f32,
}

/// Single-pass `cap * tanh(clamp(x / cap, -15, 15))` on an F32 tensor.
/// Accepts a 1-row narrowed slice in place (no copy) when `x` is 2-D `[1, N]`.
pub(super) fn softcap_f32(x: &Tensor, cap: f32) -> Result<Tensor> {
    let device = metal_device(x)?;
    let (xh_opt, xb, xoff) = if x.rank() == 2 && x.dim(0)? == 1 {
        let (buf, off, _) = row_view(x, "softcap input")?;
        (None, buf, off)
    } else {
        let xh = held(x)?;
        let (buf, off) = tensor_buffer(&xh, "softcap input")?;
        (Some(xh), buf, off)
    };
    let _keep = xh_opt;
    let count = x.elem_count();
    let params = SoftcapParams {
        count: to_u32(count, "softcap count")?,
        cap,
    };
    let out = new_output(&device, count, "inkling_softcap")?;
    encode(
        &device,
        "inkling_softcap",
        &[(&xb, xoff)],
        &[(&out, 0)],
        &params,
        (count.div_ceil(256), 1, 1),
        256,
    )?;
    Ok(wrap_output(&device, out, DType::F32, x.shape().clone()))
}

#[repr(C)]
struct PostConvParams {
    tokens: u32,
    channels: u32,
    width: u32,
    x_stride: u32,
    routes: u32,
    scale: f32,
    flags: u32,
}

/// Input of [`post_conv`]: a branch output, or routed expert rows to sum.
pub(super) enum PostInput<'a> {
    /// `[tokens, channels]` rows.
    Rows(&'a Tensor),
    /// `routed` `[tokens, routes, channels]` summed with `weights`
    /// `[tokens, routes]`.
    Routes {
        routed: &'a Tensor,
        weights: &'a Tensor,
    },
}

/// A branch's residual short convolution with static `weight`
/// `[channels, width]` after `past` (`[width - 1, channels]`, or zeros),
/// plus `residual` `[tokens, channels]` if given. Returns the output
/// `[tokens, channels]` and the next state `[width - 1, channels]`.
pub(super) fn post_conv(
    input: PostInput<'_>,
    past: Option<&Tensor>,
    weight: &Tensor,
    residual: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let (out, state, _) = post_conv_inner(input, past, weight, residual, false)?;
    Ok((out, state))
}

/// [`post_conv`], also returning the un-convolved `[tokens, channels]` input
/// rows (for speculative rollback when `input` is `PostInput::Routes`).
pub(super) fn post_conv_with_raw(
    input: PostInput<'_>,
    past: Option<&Tensor>,
    weight: &Tensor,
    residual: Option<&Tensor>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (out, state, raw) = post_conv_inner(input, past, weight, residual, true)?;
    Ok((out, state, raw.expect("raw post-conv rows")))
}

fn post_conv_inner(
    input: PostInput<'_>,
    past: Option<&Tensor>,
    weight: &Tensor,
    residual: Option<&Tensor>,
    emit_raw: bool,
) -> Result<(Tensor, Tensor, Option<Tensor>)> {
    let (channels, width) = weight.dims2()?;
    let mut kept = Vec::new();
    let (device, tokens, x, x_stride, routes, route_weights) = match input {
        PostInput::Rows(x) => {
            let (tokens, x_channels) = x.dims2()?;
            if x_channels != channels {
                candle_core::bail!("Inkling 2 post-conv geometry differs")
            }
            let rows = row_view(x, "post-conv input")?;
            (metal_device(x)?, tokens, (rows.0, rows.1), rows.2, 0, None)
        }
        PostInput::Routes { routed, weights } => {
            let (tokens, routes, routed_channels) = routed.dims3()?;
            if routed_channels != channels || weights.dims() != [tokens, routes] {
                candle_core::bail!("Inkling 2 routed post-conv geometry differs")
            }
            kept.push(held(routed)?);
            kept.push(held(weights)?);
            (
                metal_device(routed)?,
                tokens,
                tensor_buffer(&kept[0], "routed rows")?,
                channels,
                routes,
                Some(tensor_buffer(&kept[1], "route weights")?),
            )
        }
    };
    if width < 2
        || past.is_some_and(|p| p.dims() != [width - 1, channels])
        || residual.is_some_and(|r| r.dims() != [tokens, channels])
    {
        candle_core::bail!("Inkling 2 post-conv geometry differs")
    }
    let past_held = past.map(held).transpose()?;
    let weight_held = held(weight)?;
    let residual_held = residual.map(held).transpose()?;
    let past = past_held
        .as_ref()
        .map(|p| tensor_buffer(p, "post-conv state"))
        .transpose()?;
    let weight = tensor_buffer(&weight_held, "post-conv weight")?;
    let residual = residual_held
        .as_ref()
        .map(|r| tensor_buffer(r, "residual"))
        .transpose()?;
    let params = PostConvParams {
        tokens: to_u32(tokens, "post-conv tokens")?,
        channels: to_u32(channels, "post-conv channels")?,
        width: to_u32(width, "post-conv width")?,
        x_stride: to_u32(x_stride, "post-conv stride")?,
        routes: to_u32(routes, "routes")?,
        scale: 1.0 / ((width + 1) as f32).sqrt(),
        flags: u32::from(route_weights.is_some())
            | (u32::from(residual.is_some()) << 1)
            | (u32::from(past.is_some()) << 2)
            | (u32::from(emit_raw) << 3),
    };
    let history = width - 1;
    let out = new_output(&device, tokens * channels, "inkling_post_conv")?;
    let state = new_output(&device, history * channels, "inkling_post_state")?;
    let raw_out = if emit_raw {
        Some(new_output(&device, tokens * channels, "inkling_post_raw")?)
    } else {
        None
    };
    let dummy = (&x.0, x.1);
    let elements = (tokens + history) * channels;
    encode(
        &device,
        "inkling_post_conv",
        &[
            (&x.0, x.1),
            route_weights.as_ref().map_or(dummy, |w| (&w.0, w.1)),
            past.as_ref().map_or(dummy, |p| (&p.0, p.1)),
            (&weight.0, weight.1),
            residual.as_ref().map_or(dummy, |r| (&r.0, r.1)),
        ],
        &[
            (&out, 0),
            (&state, 0),
            raw_out.as_ref().map_or((&state, 0), |raw| (raw, 0)),
        ],
        &params,
        (elements.div_ceil(256), 1, 1),
        256,
    )?;
    let raw = raw_out.map(|buf| wrap_output(&device, buf, DType::F32, (tokens, channels)));
    Ok((
        wrap_output(&device, out, DType::F32, (tokens, channels)),
        wrap_output(&device, state, DType::F32, (history, channels)),
        raw,
    ))
}

#[repr(C)]
struct ConvRollbackParams {
    kept: u32,
    history: u32,
    ch0: u32,
    st0: u32,
    has_p0: u32,
    ch1: u32,
    st1: u32,
    has_p1: u32,
    ch2: u32,
    st2: u32,
    has_p2: u32,
    ch3: u32,
    st3: u32,
    has_p3: u32,
}

/// Rebuild all present convolution states of one visit after speculative
/// rollback, keeping the first `kept` input rows, in a single GPU launch.
pub(super) fn visit_conv_rollback(
    slots: &[Option<(Option<Tensor>, Tensor, usize)>; 4],
    kept: usize,
) -> Result<[Option<Tensor>; 4]> {
    let mut device = None;
    let mut history = None;
    let mut max_channels = 0usize;
    let mut past_held: [Option<Tensor>; 4] = [None, None, None, None];
    let mut input_views: [Option<(Buffer, usize, usize, usize)>; 4] = [None, None, None, None];
    for (idx, slot) in slots.iter().enumerate() {
        let Some((past, input, width)) = slot else {
            continue;
        };
        if input.dtype() != DType::F32 || !input.device().is_metal() {
            candle_core::bail!("Inkling 2 conv rollback requires Metal F32 tensors")
        }
        let h = width.saturating_sub(1);
        if let Some(prev_h) = history {
            if prev_h != h {
                candle_core::bail!("Inkling 2 conv rollback widths differ across slots")
            }
        } else {
            history = Some(h);
        }
        if device.is_none() {
            device = Some(metal_device(input)?);
        }
        let (rows, channels) = input.dims2()?;
        if kept > rows || channels == 0 {
            candle_core::bail!("Inkling 2 conv rollback kept rows out of range")
        }
        let (ibuf, ioff, istride) = row_view(input, "conv rollback input")?;
        if let Some(p) = past {
            past_held[idx] = Some(held(p)?);
        }
        max_channels = max_channels.max(channels);
        input_views[idx] = Some((ibuf, ioff, istride, channels));
    }
    let Some(device) = device else {
        return Ok([None, None, None, None]);
    };
    let history = history.unwrap_or(0);
    if history == 0 {
        return Ok([None, None, None, None]);
    }
    let fallback_in = input_views
        .iter()
        .flatten()
        .next()
        .map(|(b, o, _, _)| (b.clone(), *o))
        .unwrap();
    let mut past_bufs: [(Buffer, usize, u32); 4] =
        std::array::from_fn(|_| (fallback_in.0.clone(), fallback_in.1, 0u32));
    for idx in 0..4 {
        if let Some(p) = &past_held[idx] {
            let (b, o) = tensor_buffer(p, "conv rollback past")?;
            past_bufs[idx] = (b, o, 1);
        }
    }
    let mut out_bufs: [Option<std::sync::Arc<Buffer>>; 4] = [None, None, None, None];
    let mut ch = [0u32; 4];
    let mut st = [0u32; 4];
    for idx in 0..4 {
        if let Some((_, _, stride, channels)) = &input_views[idx] {
            ch[idx] = to_u32(*channels, "conv rollback channels")?;
            st[idx] = to_u32(*stride, "conv rollback stride")?;
            out_bufs[idx] = Some(new_output(
                &device,
                history * *channels,
                "inkling_conv_rollback",
            )?);
        }
    }
    let fallback_out = out_bufs.iter().flatten().next().unwrap().clone();
    let params = ConvRollbackParams {
        kept: to_u32(kept, "conv rollback kept")?,
        history: to_u32(history, "conv rollback history")?,
        ch0: ch[0],
        st0: st[0],
        has_p0: past_bufs[0].2,
        ch1: ch[1],
        st1: st[1],
        has_p1: past_bufs[1].2,
        ch2: ch[2],
        st2: st[2],
        has_p2: past_bufs[2].2,
        ch3: ch[3],
        st3: st[3],
        has_p3: past_bufs[3].2,
    };
    let in_ref = |idx: usize| -> (&Buffer, usize) {
        match &input_views[idx] {
            Some((b, o, _, _)) => (b, *o),
            None => (&fallback_in.0, fallback_in.1),
        }
    };
    let out_ref = |idx: usize| -> (&Buffer, usize) {
        match &out_bufs[idx] {
            Some(b) => (b, 0),
            None => (&fallback_out, 0),
        }
    };
    let elements = history * max_channels;
    encode(
        &device,
        "inkling_visit_conv_rollback",
        &[
            (&past_bufs[0].0, past_bufs[0].1),
            in_ref(0),
            (&past_bufs[1].0, past_bufs[1].1),
            in_ref(1),
            (&past_bufs[2].0, past_bufs[2].1),
            in_ref(2),
            (&past_bufs[3].0, past_bufs[3].1),
            in_ref(3),
        ],
        &[out_ref(0), out_ref(1), out_ref(2), out_ref(3)],
        &params,
        (elements.div_ceil(256), 4, 1),
        256,
    )?;
    Ok(std::array::from_fn(|idx| {
        out_bufs[idx]
            .take()
            .map(|buf| wrap_output(&device, buf, DType::F32, (history, ch[idx] as usize)))
    }))
}

#[repr(C)]
struct GateNormParams {
    rows: u32,
    hidden: u32,
    eps: f32,
    groups: u32,
    x_groups: u32,
    flags: u32,
    weight_rows: u32,
}

/// [`gate_combine`] followed by an RMS norm with `weight` `[rows, hidden]`
/// (row `group % rows`), in one dispatch. `forget_gain` is `[groups, tokens,
/// 2 * hidden]` group-major when `forget_gain_group_major`, else token-major
/// `[tokens * groups, 2 * hidden]`; `x` is `[tokens, x_groups, hidden]`
/// (one row per token or per group) and `message` `[tokens, groups,
/// hidden]`. Returns `[groups, tokens, hidden]` when `group_major_out`, else
/// `[tokens, groups, hidden]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn gate_combine_norm(
    forget_gain: &Tensor,
    x: &Tensor,
    message: &Tensor,
    weight: &Tensor,
    groups: usize,
    eps: f32,
    forget_gain_group_major: bool,
    group_major_out: bool,
) -> Result<Tensor> {
    let device = metal_device(x)?;
    let (tokens, x_groups, hidden) = x.dims3()?;
    let (weight_rows, weight_hidden) = weight.dims2()?;
    if groups == 0
        || (x_groups != 1 && x_groups != groups)
        || message.dims() != [tokens, groups, hidden]
        || forget_gain.elem_count() != tokens * groups * 2 * hidden
        || weight_hidden != hidden
        || weight_rows == 0
    {
        candle_core::bail!("Inkling 2 gate-norm geometry differs")
    }
    let fg_is_f16 = forget_gain.dtype() == DType::F16;
    let kept = [forget_gain, x, message, weight]
        .into_iter()
        .map(held)
        .collect::<Result<Vec<_>>>()?;
    let fg = if fg_is_f16 {
        f16_buffer(&kept[0], "gate projections")?
    } else {
        tensor_buffer(&kept[0], "gate projections")?
    };
    let xb = tensor_buffer(&kept[1], "gate input")?;
    let mb = tensor_buffer(&kept[2], "gate message")?;
    let wb = tensor_buffer(&kept[3], "gate norm weight")?;
    let rows = tokens * groups;
    let params = GateNormParams {
        rows: to_u32(rows, "gate rows")?,
        hidden: to_u32(hidden, "gate width")?,
        eps,
        groups: to_u32(groups, "gate groups")?,
        x_groups: to_u32(x_groups, "gate inputs")?,
        flags: u32::from(forget_gain_group_major) | (u32::from(group_major_out) << 1),
        weight_rows: to_u32(weight_rows, "gate norm rows")?,
    };
    let name = if fg_is_f16 {
        "inkling_gate_combine_norm_half"
    } else {
        "inkling_gate_combine_norm"
    };
    let out = new_output(&device, rows * hidden, name)?;
    encode(
        &device,
        name,
        &[(&fg.0, fg.1), (&xb.0, xb.1), (&mb.0, mb.1), (&wb.0, wb.1)],
        &[(&out, 0)],
        &params,
        (rows, 1, 1),
        0,
    )?;
    let shape = if group_major_out {
        Shape::from((groups, tokens, hidden))
    } else {
        Shape::from((tokens, groups, hidden))
    };
    Ok(wrap_output(&device, out, DType::F32, shape))
}

#[repr(C)]
struct RouteParams {
    tokens: u32,
    experts: u32,
    top_k: u32,
    scale: f32,
}

/// Sigmoid routing: experts chosen by `sigmoid(logits) + bias`, weighted by
/// their sigmoids normalized over the `top_k` and times `scale`. Returns
/// unique in-range U32 IDs and F32 weights, both `[tokens, top_k]`.
pub(super) fn sigmoid_route(
    logits: &Tensor,
    bias: &Tensor,
    top_k: usize,
    scale: f32,
) -> Result<(Tensor, Tensor)> {
    let device = metal_device(logits)?;
    let (tokens, experts) = logits.dims2()?;
    if experts == 0 || experts > 256 || top_k == 0 || top_k > 16 || top_k > experts {
        candle_core::bail!("Inkling 2 routing geometry differs")
    }
    if bias.dims() != [experts] {
        candle_core::bail!("Inkling 2 router bias geometry differs")
    }
    let kept = (held(logits)?, held(bias)?);
    let logits = tensor_buffer(&kept.0, "router logits")?;
    let bias = tensor_buffer(&kept.1, "router bias")?;
    let params = RouteParams {
        tokens: to_u32(tokens, "route tokens")?,
        experts: to_u32(experts, "experts")?,
        top_k: to_u32(top_k, "routes")?,
        scale,
    };
    let ids = new_output(&device, tokens * top_k, "inkling_route_ids")?;
    let weights = new_output(&device, tokens * top_k, "inkling_route_weights")?;
    encode(
        &device,
        "inkling_sigmoid_route",
        &[(&logits.0, logits.1), (&bias.0, bias.1)],
        &[(&ids, 0), (&weights, 0)],
        &params,
        (tokens, 1, 1),
        32,
    )?;
    Ok((
        wrap_output(&device, ids, DType::U32, (tokens, top_k)),
        wrap_output(&device, weights, DType::F32, (tokens, top_k)),
    ))
}

#[repr(C)]
struct Q8RowParams {
    rows: u32,
    hidden: u32,
}

/// Bytes in one Q8_0 block: an f16 scale and 32 int8 values.
const Q8_0_BLOCK_BYTES: usize = 34;

/// Rows `ids` of a `[rows, hidden]` Q8_0 table held as raw bytes on Metal,
/// as F32 `[ids, hidden]`. Each value is exact; Candle's Metal row gather
/// multiplies in half precision.
pub(super) fn q8_rows(table: &Tensor, rows: usize, hidden: usize, ids: &Tensor) -> Result<Tensor> {
    let device = metal_device(table)?;
    let tokens = ids.dim(0)?;
    if rows == 0
        || !hidden.is_multiple_of(32)
        || table.dtype() != DType::U8
        || !table.is_contiguous()
        || table.elem_count() != rows * hidden / 32 * Q8_0_BLOCK_BYTES
        || ids.dtype() != DType::U32
        || ids.rank() != 1
    {
        candle_core::bail!("Inkling 2 Q8_0 table geometry differs")
    }
    let ids = held(ids)?;
    let (table_storage, table_layout) = table.storage_and_layout();
    let candle_core::Storage::Metal(table_storage) = &*table_storage else {
        return metal_only();
    };
    let (ids_storage, ids_layout) = ids.storage_and_layout();
    let candle_core::Storage::Metal(ids_storage) = &*ids_storage else {
        return metal_only();
    };
    let params = Q8RowParams {
        rows: to_u32(rows, "table rows")?,
        hidden: to_u32(hidden, "hidden")?,
    };
    let out = new_output(&device, tokens * hidden, "inkling_q8_rows")?;
    if tokens > 0 {
        encode(
            &device,
            "inkling_q8_rows",
            &[
                (table_storage.buffer(), table_layout.start_offset()),
                (ids_storage.buffer(), ids_layout.start_offset() * 4),
            ],
            &[(&out, 0)],
            &params,
            (tokens, 1, 1),
            256,
        )?;
    }
    Ok(wrap_output(&device, out, DType::F32, (tokens, hidden)))
}

/// Rows up to which [`bf16_matvec`] serves a BF16 projection.
pub(super) const BF16_MATVEC_MAX_ROWS: usize = 8;

const BF16_MV_KERNELS: [(&str, usize); 8] = [
    ("inkling_bf16_matvec_1", 64),
    ("inkling_bf16_matvec_2", 64),
    ("inkling_bf16_matvec_3", 64),
    ("inkling_bf16_matvec_4", 64),
    ("inkling_bf16_matvec_5", 128),
    ("inkling_bf16_matvec_6", 128),
    ("inkling_bf16_matvec_7", 128),
    ("inkling_bf16_matvec_8", 128),
];

#[repr(C)]
struct Bf16MvParams {
    in_dim: u32,
    out_dim: u32,
    softcap: f32,
}

/// `x` `[rows, in]` F32 through BF16 `weight` `[out, in]`, as F32
/// `[rows, out]`, reading each weight once across `rows <= 8`, widened
/// exactly and summed in f32.
pub(super) fn bf16_matvec(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    bf16_matvec_with_softcap(x, weight, None, 0.0)
}

/// [`bf16_matvec`] over the first `out_limit` rows (if given) with optional
/// fused logit `softcap` (`> 0`).
pub(super) fn bf16_matvec_with_softcap(
    x: &Tensor,
    weight: &Tensor,
    out_limit: Option<usize>,
    softcap: f32,
) -> Result<Tensor> {
    let device = metal_device(weight)?;
    let (rows, width) = x.dims2()?;
    let (weight_out, in_dim) = weight.dims2()?;
    let out = out_limit.unwrap_or(weight_out);
    if width != in_dim
        || out > weight_out
        || !in_dim.is_multiple_of(4)
        || rows > BF16_MATVEC_MAX_ROWS
        || weight.dtype() != DType::BF16
        || !weight.is_contiguous()
    {
        candle_core::bail!("Inkling 2 BF16 projection geometry differs")
    }
    let x = held(x)?;
    let (x_buffer, x_offset) = tensor_buffer(&x, "BF16 projection input")?;
    let (w_storage, w_layout) = weight.storage_and_layout();
    let candle_core::Storage::Metal(w_storage) = &*w_storage else {
        return metal_only();
    };
    if !x_offset.is_multiple_of(16) || !(w_layout.start_offset() * 2).is_multiple_of(8) {
        candle_core::bail!("Inkling 2 BF16 projection operands are misaligned")
    }
    let params = Bf16MvParams {
        in_dim: to_u32(in_dim, "BF16 projection width")?,
        out_dim: to_u32(out, "BF16 projection rows")?,
        softcap,
    };
    let y = new_output(&device, rows * out, "inkling_bf16_matvec")?;
    if rows > 0 {
        let (name, threads) = BF16_MV_KERNELS[rows - 1];
        encode(
            &device,
            name,
            &[
                (w_storage.buffer(), w_layout.start_offset() * 2),
                (&x_buffer, x_offset),
            ],
            &[(&y, 0)],
            &params,
            (out.div_ceil(8), 1, 1),
            threads,
        )?;
    }
    Ok(wrap_output(&device, y, DType::F32, (rows, out)))
}

#[repr(C)]
struct Bf16MvManyParams {
    in_dim: u32,
    out0: u32,
    out1: u32,
    out2: u32,
    tiles0: u32,
    tiles01: u32,
}

/// Up to three single-token BF16 projections `(x, weight)` of the same input
/// width in one Metal launch.
pub(super) fn bf16_matvec_many(pairs: &[(&Tensor, &Tensor)]) -> Result<Vec<Tensor>> {
    if !(2..=3).contains(&pairs.len()) {
        candle_core::bail!("Inkling 2 bf16_matvec_many expects 2 or 3 projections")
    }
    let device = metal_device(pairs[0].1)?;
    let (_, in_dim) = pairs[0].1.dims2()?;
    if !in_dim.is_multiple_of(4) {
        return pairs.iter().map(|(x, w)| bf16_matvec(x, w)).collect();
    }
    let mut xs = Vec::with_capacity(pairs.len());
    let mut x_bufs = Vec::with_capacity(pairs.len());
    let mut w_bufs = Vec::with_capacity(pairs.len());
    let mut outs = Vec::with_capacity(pairs.len());
    let mut y_bufs = Vec::with_capacity(pairs.len());
    for (x, weight) in pairs {
        let (rows, width) = x.dims2()?;
        let (out, w_in) = weight.dims2()?;
        if rows != 1
            || width != in_dim
            || w_in != in_dim
            || weight.dtype() != DType::BF16
            || !weight.is_contiguous()
        {
            return pairs.iter().map(|(x, w)| bf16_matvec(x, w)).collect();
        }
        let xh = held(x)?;
        let (xb, xoff) = tensor_buffer(&xh, "BF16 projection input")?;
        let (w_storage, w_layout) = weight.storage_and_layout();
        let candle_core::Storage::Metal(w_storage) = &*w_storage else {
            return metal_only();
        };
        let woff = w_layout.start_offset() * 2;
        if !xoff.is_multiple_of(16) || !woff.is_multiple_of(8) {
            return pairs.iter().map(|(x, w)| bf16_matvec(x, w)).collect();
        }
        xs.push(xh);
        x_bufs.push((xb, xoff));
        w_bufs.push((w_storage.buffer().clone(), woff));
        outs.push(out);
        y_bufs.push(new_output(&device, out, "inkling_bf16_matvec_many")?);
    }
    let _keep_xs = xs;
    let tiles0 = outs[0].div_ceil(8);
    let tiles1 = outs[1].div_ceil(8);
    let tiles2 = outs.get(2).map_or(0, |o| o.div_ceil(8));
    let total_tiles = tiles0 + tiles1 + tiles2;
    let params = Bf16MvManyParams {
        in_dim: to_u32(in_dim, "BF16 projection width")?,
        out0: to_u32(outs[0], "BF16 projection rows")?,
        out1: to_u32(outs[1], "BF16 projection rows")?,
        out2: to_u32(*outs.get(2).unwrap_or(&0), "BF16 projection rows")?,
        tiles0: to_u32(tiles0, "BF16 projection tiles")?,
        tiles01: to_u32(tiles0 + tiles1, "BF16 projection tiles")?,
    };
    let (w2_b, w2_o) = w_bufs.get(2).unwrap_or(&w_bufs[1]);
    let (x2_b, x2_o) = x_bufs.get(2).unwrap_or(&x_bufs[1]);
    let y2_b = y_bufs.get(2).unwrap_or(&y_bufs[1]);
    encode(
        &device,
        "inkling_bf16_matvec_many",
        &[
            (&w_bufs[0].0, w_bufs[0].1),
            (&x_bufs[0].0, x_bufs[0].1),
            (&w_bufs[1].0, w_bufs[1].1),
            (&x_bufs[1].0, x_bufs[1].1),
            (w2_b, *w2_o),
            (x2_b, *x2_o),
        ],
        &[(&y_bufs[0], 0), (&y_bufs[1], 0), (y2_b, 0)],
        &params,
        (total_tiles, 1, 1),
        64,
    )?;
    Ok(y_bufs
        .into_iter()
        .zip(outs)
        .map(|(y, out)| wrap_output(&device, y, DType::F32, (1, out)))
        .collect())
}

/// Single-token Q4_0 projection `[1, in_dim] @ [out_dim, in_dim]^T` using
/// 8 lanes/row (zero warp tail on 2304/3072).
pub(super) fn q4_0_matvec_1(
    x: &Tensor,
    weights_u8: &Tensor,
    out_dim: usize,
    in_dim: usize,
) -> Result<Tensor> {
    let device = metal_device(weights_u8)?;
    let (rows, width) = x.dims2()?;
    if rows != 1 || width != in_dim || !in_dim.is_multiple_of(32) {
        candle_core::bail!("Inkling 2 Q4_0 matvec_1 geometry differs")
    }
    let xh = held(x)?;
    let (x_buf, x_off) = tensor_buffer(&xh, "Q4_0 matvec input")?;
    let (w_storage, w_layout) = weights_u8.storage_and_layout();
    let candle_core::Storage::Metal(w_storage) = &*w_storage else {
        return metal_only();
    };
    let params = Bf16MvParams {
        in_dim: to_u32(in_dim, "Q4_0 width")?,
        out_dim: to_u32(out_dim, "Q4_0 rows")?,
        softcap: 0.0,
    };
    let y = new_output(&device, out_dim, "inkling_q4_0_matvec_1")?;
    encode(
        &device,
        "inkling_q4_0_matvec_1",
        &[
            (w_storage.buffer(), w_layout.start_offset()),
            (&x_buf, x_off),
        ],
        &[(&y, 0)],
        &params,
        (out_dim.div_ceil(8), 1, 1),
        64,
    )?;
    Ok(wrap_output(&device, y, DType::F32, (1, out_dim)))
}

#[repr(C)]
struct Q8Embed1Params {
    rows: u32,
    hidden: u32,
    token_id: u32,
}

/// Single-token Q8_0 embedding lookup passing `token_id` via `set_bytes`
/// without allocating an ID buffer (bitwise identical to `q8_rows`).
pub(super) fn q8_embed_1(
    table: &Tensor,
    rows: usize,
    hidden: usize,
    token_id: u32,
) -> Result<Tensor> {
    let device = metal_device(table)?;
    let (t_storage, t_layout) = table.storage_and_layout();
    let candle_core::Storage::Metal(t_storage) = &*t_storage else {
        return metal_only();
    };
    let params = Q8Embed1Params {
        rows: to_u32(rows, "embedding rows")?,
        hidden: to_u32(hidden, "embedding width")?,
        token_id,
    };
    let out = new_output(&device, hidden, "inkling_q8_embed_1")?;
    encode(
        &device,
        "inkling_q8_embed_1",
        &[(t_storage.buffer(), t_layout.start_offset())],
        &[(&out, 0)],
        &params,
        (1, 1, 1),
        256,
    )?;
    Ok(wrap_output(&device, out, DType::F32, (1, hidden)))
}

#[repr(C)]
struct MvPostConv1Params {
    in_dim: u32,
    out_dim: u32,
    width: u32,
    has_past: u32,
    has_residual: u32,
    conv_scale: f32,
}

/// Single-token fused Q4_0 projection + static `post_conv` + optional `residual`.
#[allow(clippy::too_many_arguments)]
pub(super) fn q4_0_matvec_post_conv_1(
    x: &Tensor,
    weights_u8: &Tensor,
    out_dim: usize,
    in_dim: usize,
    past: Option<&Tensor>,
    conv_weight: &Tensor,
    residual: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let device = metal_device(weights_u8)?;
    let (_, conv_width) = conv_weight.dims2()?;
    let history = conv_width - 1;
    let xh = held(x)?;
    let (x_buf, x_off) = tensor_buffer(&xh, "Q4_0 matvec input")?;
    let (w_storage, w_layout) = weights_u8.storage_and_layout();
    let candle_core::Storage::Metal(w_storage) = &*w_storage else {
        return metal_only();
    };
    let past_h = past.map(held).transpose()?;
    let past_b = past_h
        .as_ref()
        .map(|p| tensor_buffer(p, "post_conv past"))
        .transpose()?;
    let cw_h = held(conv_weight)?;
    let (cw_buf, cw_off) = tensor_buffer(&cw_h, "post_conv weight")?;
    let res_h = residual.map(held).transpose()?;
    let res_b = res_h
        .as_ref()
        .map(|r| tensor_buffer(r, "post_conv residual"))
        .transpose()?;
    let params = MvPostConv1Params {
        in_dim: to_u32(in_dim, "Q4_0 width")?,
        out_dim: to_u32(out_dim, "Q4_0 rows")?,
        width: to_u32(conv_width, "conv width")?,
        has_past: u32::from(past_b.is_some()),
        has_residual: u32::from(res_b.is_some()),
        conv_scale: 1.0 / ((conv_width + 1) as f32).sqrt(),
    };
    let out = new_output(&device, out_dim, "inkling_q4_0_mv_pc_out")?;
    let state = new_output(&device, history * out_dim, "inkling_q4_0_mv_pc_state")?;
    let past_in = past_b.as_ref().map_or((&x_buf, x_off), |b| (&b.0, b.1));
    let res_in = res_b.as_ref().map_or((&x_buf, x_off), |b| (&b.0, b.1));
    encode(
        &device,
        "inkling_q4_0_matvec_post_conv_1",
        &[
            (w_storage.buffer(), w_layout.start_offset()),
            (&x_buf, x_off),
            past_in,
            (&cw_buf, cw_off),
            res_in,
        ],
        &[(&out, 0), (&state, 0)],
        &params,
        (out_dim.div_ceil(8), 1, 1),
        64,
    )?;
    Ok((
        wrap_output(&device, out, DType::F32, (1, out_dim)),
        wrap_output(&device, state, DType::F32, (history, out_dim)),
    ))
}

/// Single-token fused BF16 projection + static `post_conv` + optional `residual`.
#[allow(clippy::too_many_arguments)]
pub(super) fn bf16_matvec_post_conv_1(
    x: &Tensor,
    weight: &Tensor,
    past: Option<&Tensor>,
    conv_weight: &Tensor,
    residual: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let device = metal_device(weight)?;
    let (out_dim, in_dim) = weight.dims2()?;
    let (_, conv_width) = conv_weight.dims2()?;
    let history = conv_width - 1;
    let xh = held(x)?;
    let (x_buf, x_off) = tensor_buffer(&xh, "BF16 matvec input")?;
    let (w_storage, w_layout) = weight.storage_and_layout();
    let candle_core::Storage::Metal(w_storage) = &*w_storage else {
        return metal_only();
    };
    let past_h = past.map(held).transpose()?;
    let past_b = past_h
        .as_ref()
        .map(|p| tensor_buffer(p, "post_conv past"))
        .transpose()?;
    let cw_h = held(conv_weight)?;
    let (cw_buf, cw_off) = tensor_buffer(&cw_h, "post_conv weight")?;
    let res_h = residual.map(held).transpose()?;
    let res_b = res_h
        .as_ref()
        .map(|r| tensor_buffer(r, "post_conv residual"))
        .transpose()?;
    let params = MvPostConv1Params {
        in_dim: to_u32(in_dim, "BF16 width")?,
        out_dim: to_u32(out_dim, "BF16 rows")?,
        width: to_u32(conv_width, "conv width")?,
        has_past: u32::from(past_b.is_some()),
        has_residual: u32::from(res_b.is_some()),
        conv_scale: 1.0 / ((conv_width + 1) as f32).sqrt(),
    };
    let out = new_output(&device, out_dim, "inkling_bf16_mv_pc_out")?;
    let state = new_output(&device, history * out_dim, "inkling_bf16_mv_pc_state")?;
    let past_in = past_b.as_ref().map_or((&x_buf, x_off), |b| (&b.0, b.1));
    let res_in = res_b.as_ref().map_or((&x_buf, x_off), |b| (&b.0, b.1));
    encode(
        &device,
        "inkling_bf16_matvec_post_conv_1",
        &[
            (w_storage.buffer(), w_layout.start_offset() * 2),
            (&x_buf, x_off),
            past_in,
            (&cw_buf, cw_off),
            res_in,
        ],
        &[(&out, 0), (&state, 0)],
        &params,
        (out_dim.div_ceil(8), 1, 1),
        64,
    )?;
    Ok((
        wrap_output(&device, out, DType::F32, (1, out_dim)),
        wrap_output(&device, state, DType::F32, (history, out_dim)),
    ))
}

/// Single-token 4-row-amortized Q8_0 projection with optional fused `softcap`.
pub(super) fn q8_0_matvec_with_softcap(
    x: &Tensor,
    weights_u8: &Tensor,
    out_dim: usize,
    in_dim: usize,
    out_limit: Option<usize>,
    softcap: f32,
) -> Result<Tensor> {
    let device = metal_device(weights_u8)?;
    let (rows, width) = x.dims2()?;
    let out = out_limit.unwrap_or(out_dim);
    if rows != 1 || width != in_dim || out > out_dim || !in_dim.is_multiple_of(32) {
        candle_core::bail!("Inkling 2 Q8_0 matvec_1 geometry differs")
    }
    let xh = held(x)?;
    let (x_buf, x_off) = tensor_buffer(&xh, "Q8_0 matvec input")?;
    let (w_storage, w_layout) = weights_u8.storage_and_layout();
    let candle_core::Storage::Metal(w_storage) = &*w_storage else {
        return metal_only();
    };
    let params = Bf16MvParams {
        in_dim: to_u32(in_dim, "Q8_0 width")?,
        out_dim: to_u32(out, "Q8_0 rows")?,
        softcap,
    };
    let y = new_output(&device, out, "inkling_q8_0_matvec_1")?;
    encode(
        &device,
        "inkling_q8_0_matvec_1",
        &[
            (w_storage.buffer(), w_layout.start_offset()),
            (&x_buf, x_off),
        ],
        &[(&y, 0)],
        &params,
        (out.div_ceil(8), 1, 1),
        64,
    )?;
    Ok(wrap_output(&device, y, DType::F32, (1, out)))
}

/// Up to three single-token Q4_0 projections `(x, weights_u8, out_dim)` of the
/// same `in_dim` in one Metal launch.
pub(super) fn q4_0_matvec_many(
    pairs: &[(&Tensor, &Tensor, usize)],
    in_dim: usize,
) -> Result<Vec<Tensor>> {
    if !(2..=3).contains(&pairs.len()) || !in_dim.is_multiple_of(32) {
        candle_core::bail!("Inkling 2 q4_0_matvec_many geometry differs")
    }
    let device = metal_device(pairs[0].1)?;
    let mut xs = Vec::with_capacity(pairs.len());
    let mut x_bufs = Vec::with_capacity(pairs.len());
    let mut w_bufs = Vec::with_capacity(pairs.len());
    let mut outs = Vec::with_capacity(pairs.len());
    let mut y_bufs = Vec::with_capacity(pairs.len());
    for &(x, weights_u8, out) in pairs {
        let xh = held(x)?;
        let (xb, xoff) = tensor_buffer(&xh, "Q4_0 matvec input")?;
        let (w_storage, w_layout) = weights_u8.storage_and_layout();
        let candle_core::Storage::Metal(w_storage) = &*w_storage else {
            return metal_only();
        };
        xs.push(xh);
        x_bufs.push((xb, xoff));
        w_bufs.push((w_storage.buffer().clone(), w_layout.start_offset()));
        outs.push(out);
        y_bufs.push(new_output(&device, out, "inkling_q4_0_matvec_many")?);
    }
    let _keep_xs = xs;
    let tiles0 = outs[0].div_ceil(8);
    let tiles1 = outs[1].div_ceil(8);
    let tiles2 = outs.get(2).map_or(0, |o| o.div_ceil(8));
    let total_tiles = tiles0 + tiles1 + tiles2;
    let params = Bf16MvManyParams {
        in_dim: to_u32(in_dim, "Q4_0 width")?,
        out0: to_u32(outs[0], "Q4_0 rows")?,
        out1: to_u32(outs[1], "Q4_0 rows")?,
        out2: to_u32(*outs.get(2).unwrap_or(&0), "Q4_0 rows")?,
        tiles0: to_u32(tiles0, "Q4_0 tiles")?,
        tiles01: to_u32(tiles0 + tiles1, "Q4_0 tiles")?,
    };
    let (w2_b, w2_o) = w_bufs.get(2).unwrap_or(&w_bufs[1]);
    let (x2_b, x2_o) = x_bufs.get(2).unwrap_or(&x_bufs[1]);
    let y2_b = y_bufs.get(2).unwrap_or(&y_bufs[1]);
    encode(
        &device,
        "inkling_q4_0_matvec_many",
        &[
            (&w_bufs[0].0, w_bufs[0].1),
            (&x_bufs[0].0, x_bufs[0].1),
            (&w_bufs[1].0, w_bufs[1].1),
            (&x_bufs[1].0, x_bufs[1].1),
            (w2_b, *w2_o),
            (x2_b, *x2_o),
        ],
        &[(&y_bufs[0], 0), (&y_bufs[1], 0), (y2_b, 0)],
        &params,
        (total_tiles, 1, 1),
        64,
    )?;
    Ok(y_bufs
        .into_iter()
        .zip(outs)
        .map(|(y, out)| wrap_output(&device, y, DType::F32, (1, out)))
        .collect())
}

#[repr(C)]
struct F32MvParams {
    batch: u32,
    in_dim: u32,
    out_dim: u32,
}

const F32_MV_T_KERNELS: [&str; 8] = [
    "inkling_f32_mv_t_1",
    "inkling_f32_mv_t_2",
    "inkling_f32_mv_t_3",
    "inkling_f32_mv_t_4",
    "inkling_f32_mv_t_5",
    "inkling_f32_mv_t_6",
    "inkling_f32_mv_t_7",
    "inkling_f32_mv_t_8",
];

const F32_MV_NT_KERNELS: [&str; 8] = [
    "inkling_f32_mv_nt_1",
    "inkling_f32_mv_nt_2",
    "inkling_f32_mv_nt_3",
    "inkling_f32_mv_nt_4",
    "inkling_f32_mv_nt_5",
    "inkling_f32_mv_nt_6",
    "inkling_f32_mv_nt_7",
    "inkling_f32_mv_nt_8",
];

const F16_MV_T_KERNELS: [&str; 8] = [
    "inkling_f16_mv_t_1",
    "inkling_f16_mv_t_2",
    "inkling_f16_mv_t_3",
    "inkling_f16_mv_t_4",
    "inkling_f16_mv_t_5",
    "inkling_f16_mv_t_6",
    "inkling_f16_mv_t_7",
    "inkling_f16_mv_t_8",
];

const F16_MV_NT_KERNELS: [&str; 8] = [
    "inkling_f16_mv_nt_1",
    "inkling_f16_mv_nt_2",
    "inkling_f16_mv_nt_3",
    "inkling_f16_mv_nt_4",
    "inkling_f16_mv_nt_5",
    "inkling_f16_mv_nt_6",
    "inkling_f16_mv_nt_7",
    "inkling_f16_mv_nt_8",
];

/// Batched `[B, T, in] @ [B, out, in]^T` (or 2-D `[T, in] @ [out, in]^T`)
/// reading F32 or F16 `weight` once across `1 <= T <= 8` rows. Returns `None` if
/// shape or alignment falls outside the fast path.
pub(super) fn try_f32_matvec_t_rows(x: &Tensor, weight: &Tensor) -> Result<Option<Tensor>> {
    let w_f16 = weight.dtype() == DType::F16;
    if !x.device().is_metal() || x.dtype() != DType::F32 || (!w_f16 && weight.dtype() != DType::F32)
    {
        return Ok(None);
    }
    let (batch, tokens, in_dim, out_dim, rank2) = match (x.dims(), weight.dims()) {
        ([t, k], [o, wk]) if k == wk => (1, *t, *k, *o, true),
        ([b, t, k], [wb, o, wk]) if b == wb && k == wk => (*b, *t, *k, *o, false),
        _ => return Ok(None),
    };
    if !(1..=8).contains(&tokens) || in_dim == 0 || out_dim == 0 || !in_dim.is_multiple_of(4) {
        return Ok(None);
    }
    let x = held(x)?;
    let weight = held(weight)?;
    let (x_buf, x_off) = tensor_buffer(&x, "F32 mv_t input")?;
    let (w_buf, w_off) = if w_f16 {
        f16_buffer(&weight, "F16 mv_t weight")?
    } else {
        tensor_buffer(&weight, "F32 mv_t weight")?
    };
    let w_align = if w_f16 { 8 } else { 16 };
    if !x_off.is_multiple_of(16) || !w_off.is_multiple_of(w_align) {
        return Ok(None);
    }
    let device = metal_device(&x)?;
    let params = F32MvParams {
        batch: to_u32(batch, "F32 mv_t batch")?,
        in_dim: to_u32(in_dim, "F32 mv_t in_dim")?,
        out_dim: to_u32(out_dim, "F32 mv_t out_dim")?,
    };
    let kernel = if w_f16 {
        F16_MV_T_KERNELS[tokens - 1]
    } else {
        F32_MV_T_KERNELS[tokens - 1]
    };
    let threads = if tokens == 1 { 192 } else { 32 };
    let y = new_output(&device, batch * tokens * out_dim, kernel)?;
    encode(
        &device,
        kernel,
        &[(&w_buf, w_off), (&x_buf, x_off)],
        &[(&y, 0)],
        &params,
        (out_dim, batch, 1),
        threads,
    )?;
    let shape = if rank2 {
        Shape::from((tokens, out_dim))
    } else {
        Shape::from((batch, tokens, out_dim))
    };
    Ok(Some(wrap_output(&device, y, DType::F32, shape)))
}

/// Batched `[B, T, K] @ [B, K, out]` (or 2-D `[T, K] @ [K, out]`) reading F32
/// or F16 `weight` coalesced once across `1 <= T <= 8` rows for `K <= 64`.
pub(super) fn try_f32_matvec_nt_rows(x: &Tensor, weight: &Tensor) -> Result<Option<Tensor>> {
    let w_f16 = weight.dtype() == DType::F16;
    if !x.device().is_metal() || x.dtype() != DType::F32 || (!w_f16 && weight.dtype() != DType::F32)
    {
        return Ok(None);
    }
    let (batch, tokens, k_dim, out_dim, rank2) = match (x.dims(), weight.dims()) {
        ([t, k], [wk, o]) if k == wk => (1, *t, *k, *o, true),
        ([b, t, k], [wb, wk, o]) if b == wb && k == wk => (*b, *t, *k, *o, false),
        _ => return Ok(None),
    };
    if !(1..=8).contains(&tokens) || k_dim == 0 || k_dim > 64 || out_dim == 0 {
        return Ok(None);
    }
    let x = held(x)?;
    let weight = held(weight)?;
    let (x_buf, x_off) = tensor_buffer(&x, "F32 mv_nt input")?;
    let (w_buf, w_off) = if w_f16 {
        f16_buffer(&weight, "F16 mv_nt weight")?
    } else {
        tensor_buffer(&weight, "F32 mv_nt weight")?
    };
    let device = metal_device(&x)?;
    let params = F32MvParams {
        batch: to_u32(batch, "F32 mv_nt batch")?,
        in_dim: to_u32(k_dim, "F32 mv_nt k_dim")?,
        out_dim: to_u32(out_dim, "F32 mv_nt out_dim")?,
    };
    let w_align = if w_f16 { 8 } else { 16 };
    let use_v4 = tokens == 1 && out_dim.is_multiple_of(4) && w_off.is_multiple_of(w_align);
    let kernel = if use_v4 {
        if w_f16 {
            "inkling_f16_mv_nt_1_v4"
        } else {
            "inkling_f32_mv_nt_1_v4"
        }
    } else if w_f16 {
        F16_MV_NT_KERNELS[tokens - 1]
    } else {
        F32_MV_NT_KERNELS[tokens - 1]
    };
    let (groups_x, threads) = if use_v4 {
        (out_dim.div_ceil(128), 128)
    } else {
        (out_dim.div_ceil(64), 64)
    };
    let y = new_output(&device, batch * tokens * out_dim, kernel)?;
    encode(
        &device,
        kernel,
        &[(&w_buf, w_off), (&x_buf, x_off)],
        &[(&y, 0)],
        &params,
        (groups_x, batch, 1),
        threads,
    )?;
    let shape = if rank2 {
        Shape::from((tokens, out_dim))
    } else {
        Shape::from((batch, tokens, out_dim))
    };
    Ok(Some(wrap_output(&device, y, DType::F32, shape)))
}
