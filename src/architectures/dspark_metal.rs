//! Fused Metal kernels for the DSpark speculative drafter.
//!
//! Replaces the per-row Q8_0 projections, CPU RoPE uploads, multi-dispatch
//! feature mixes, and 21-dispatch-per-step Markov head loop with single-read
//! multi-row Q8_0/Q4_0 projections, fused QKV prep, and a coalesced fused
//! Markov score+argmax pass.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use candle_core::metal_backend::DeviceId;
use candle_core::{DType, MetalStorage, Result, Shape, Storage, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputePipeline};

const DSPARK_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant float NEG_LARGE = -1.0e30f;

// ---- Multi-row Q8_0 matvec (34-byte blocks: half d, char qs[32]) ----

struct Q8MvParams {
  uint in_dim;
  uint out_dim;
  uint flags; // bit 0 = add residual
};

// Reads each Q8_0 weight block once across T rows (1 <= T <= 8).
// Each of the NSG simdgroups makes R output rows (or 1 gate/up SwiGLU pair with R=2).
template <uint T, uint R, uint NSG, bool SWIGLU>
kernel void dspark_q8_0_mv_rows(
    device const uchar *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const float *res [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant Q8MvParams &p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort simd [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  const uint nb = p.in_dim / 32;
  const uint row_bytes = nb * 34;
  const uint half_rows = p.out_dim / 2;
  uint rows[R];
  if (SWIGLU) {
    const uint r = group.x * NSG + simd;
    rows[0] = r < half_rows ? r : p.out_dim;
    rows[1] = r < half_rows ? half_rows + r : p.out_dim;
  } else {
    const uint first = (group.x * NSG + simd) * R;
    #pragma unroll
    for (uint r = 0; r < R; ++r) {
      rows[r] = first + r;
    }
  }
  const uint ix = lane / 4; // 0..7
  const uint il = lane % 4; // 0..3 (8 weights per lane)
  float sums[R][T];
  #pragma unroll
  for (uint r = 0; r < R; ++r) {
    #pragma unroll
    for (uint t = 0; t < T; ++t) {
      sums[r][t] = 0.0f;
    }
  }
  for (uint ib = ix; ib < nb; ib += 8) {
    const uint x_col = ib * 32 + il * 8;
    #pragma unroll
    for (uint r = 0; r < R; ++r) {
      const uint o = rows[r];
      if (o < p.out_dim) {
        device const uchar *blk = w + ulong(o) * row_bytes + ulong(ib) * 34;
        const float d = float(*(device const half *)blk);
        device const packed_char4 *qs = (device const packed_char4 *)(blk + 2 + il * 8);
        const float4 w0 = float4(qs[0]);
        const float4 w1 = float4(qs[1]);
        #pragma unroll
        for (uint t = 0; t < T; ++t) {
          device const float4 *x4 = (device const float4 *)(x + ulong(t) * p.in_dim + x_col);
          sums[r][t] += (dot(w0, x4[0]) + dot(w1, x4[1])) * d;
        }
      }
    }
  }
  if (SWIGLU) {
    #pragma unroll
    for (uint t = 0; t < T; ++t) {
      const float gv = simd_sum(sums[0][t]);
      const float uv = simd_sum(sums[1][t]);
      if (lane == 0 && rows[0] < half_rows) {
        const float s = gv / (1.0f + exp(-clamp(gv, -80.0f, 80.0f)));
        y[ulong(t) * half_rows + rows[0]] = s * uv;
      }
    }
    return;
  }
  const bool add_res = (p.flags & 1u) != 0;
  #pragma unroll
  for (uint r = 0; r < R; ++r) {
    const uint o = rows[r];
    #pragma unroll
    for (uint t = 0; t < T; ++t) {
      const float total = simd_sum(sums[r][t]);
      if (lane == 0 && o < p.out_dim) {
        const ulong idx = ulong(t) * p.out_dim + o;
        y[idx] = add_res ? (total + res[idx]) : total;
      }
    }
  }
}

typedef decltype(dspark_q8_0_mv_rows<1, 2, 2, false>) dspark_q8_mv_fn;
template [[host_name("dspark_q8_mv_1")]] kernel dspark_q8_mv_fn dspark_q8_0_mv_rows<1, 2, 2, false>;
template [[host_name("dspark_q8_mv_2")]] kernel dspark_q8_mv_fn dspark_q8_0_mv_rows<2, 2, 2, false>;
template [[host_name("dspark_q8_mv_3")]] kernel dspark_q8_mv_fn dspark_q8_0_mv_rows<3, 2, 2, false>;
template [[host_name("dspark_q8_mv_4")]] kernel dspark_q8_mv_fn dspark_q8_0_mv_rows<4, 2, 2, false>;
template [[host_name("dspark_q8_mv_5")]] kernel dspark_q8_mv_fn dspark_q8_0_mv_rows<5, 1, 4, false>;
template [[host_name("dspark_q8_mv_6")]] kernel dspark_q8_mv_fn dspark_q8_0_mv_rows<6, 1, 4, false>;
template [[host_name("dspark_q8_mv_7")]] kernel dspark_q8_mv_fn dspark_q8_0_mv_rows<7, 1, 4, false>;
template [[host_name("dspark_q8_mv_8")]] kernel dspark_q8_mv_fn dspark_q8_0_mv_rows<8, 1, 4, false>;

template [[host_name("dspark_q8_swiglu_1")]] kernel dspark_q8_mv_fn dspark_q8_0_mv_rows<1, 2, 2, true>;
template [[host_name("dspark_q8_swiglu_2")]] kernel dspark_q8_mv_fn dspark_q8_0_mv_rows<2, 2, 2, true>;
template [[host_name("dspark_q8_swiglu_3")]] kernel dspark_q8_mv_fn dspark_q8_0_mv_rows<3, 2, 2, true>;
template [[host_name("dspark_q8_swiglu_4")]] kernel dspark_q8_mv_fn dspark_q8_0_mv_rows<4, 2, 2, true>;
template [[host_name("dspark_q8_swiglu_5")]] kernel dspark_q8_mv_fn dspark_q8_0_mv_rows<5, 2, 4, true>;
template [[host_name("dspark_q8_swiglu_6")]] kernel dspark_q8_mv_fn dspark_q8_0_mv_rows<6, 2, 4, true>;
template [[host_name("dspark_q8_swiglu_7")]] kernel dspark_q8_mv_fn dspark_q8_0_mv_rows<7, 2, 4, true>;
template [[host_name("dspark_q8_swiglu_8")]] kernel dspark_q8_mv_fn dspark_q8_0_mv_rows<8, 2, 4, true>;

// ---- Draft-only Q4_0 quantizer (from BF16 [out, in]) and multi-row matvec ----
// Standard GGML Q4_0 block: 2-byte half scale d, then 16 bytes where byte j
// holds low nibble = weight j (0..15) and high nibble = weight j + 16 (16..31),
// stored unsigned +8 (signed value = nibble - 8).

kernel void dspark_quantize_bf16_to_q4_0(
    device const ushort *src [[buffer(0)]],
    device uchar *dst [[buffer(1)]],
    constant uint &blocks [[buffer(2)]],
    uint block [[thread_position_in_grid]]) {
  if (block >= blocks) { return; }
  device const ushort *in = src + ulong(block) * 32;
  float vals[32];
  float max_abs = 0.0f;
  float signed_max = 0.0f;
  for (uint i = 0; i < 32; ++i) {
    const float v = as_type<float>(uint(in[i]) << 16);
    vals[i] = v;
    const float a = fabs(v);
    if (a > max_abs) {
      max_abs = a;
      signed_max = v;
    }
  }
  const float d = signed_max / -8.0f;
  const float id = d != 0.0f ? 1.0f / d : 0.0f;
  device uchar *out = dst + ulong(block) * 18;
  *(device half *)out = half(d);
  for (uint j = 0; j < 16; ++j) {
    const int q0 = clamp(int(vals[j] * id + 8.5f), 0, 15);
    const int q1 = clamp(int(vals[j + 16] * id + 8.5f), 0, 15);
    out[2 + j] = uchar(q0 | (q1 << 4));
  }
}

kernel void dspark_quantize_bf16_to_q8_0(
    device const ushort *src [[buffer(0)]],
    device uchar *dst [[buffer(1)]],
    constant uint &blocks [[buffer(2)]],
    uint block [[thread_position_in_grid]]) {
  if (block >= blocks) { return; }
  device const ushort *in = src + ulong(block) * 32;
  float vals[32];
  float max_abs = 0.0f;
  for (uint i = 0; i < 32; ++i) {
    const float v = as_type<float>(uint(in[i]) << 16);
    vals[i] = v;
    max_abs = max(max_abs, fabs(v));
  }
  const float d = max_abs / 127.0f;
  const float id = d != 0.0f ? 1.0f / d : 0.0f;
  device uchar *out = dst + ulong(block) * 34;
  *(device half *)out = half(d);
  device char *qs = (device char *)(out + 2);
  for (uint i = 0; i < 32; ++i) {
    qs[i] = char(clamp(int(rint(vals[i] * id)), -127, 127));
  }
}

kernel void dspark_quantize_f16_to_q8_0(
    device const half *src [[buffer(0)]],
    device uchar *dst [[buffer(1)]],
    constant uint &blocks [[buffer(2)]],
    uint block [[thread_position_in_grid]]) {
  if (block >= blocks) { return; }
  device const half *in = src + ulong(block) * 32;
  float vals[32];
  float max_abs = 0.0f;
  for (uint i = 0; i < 32; ++i) {
    const float v = float(in[i]);
    vals[i] = v;
    max_abs = max(max_abs, fabs(v));
  }
  const float d = max_abs / 127.0f;
  const float id = d != 0.0f ? 1.0f / d : 0.0f;
  device uchar *out = dst + ulong(block) * 34;
  *(device half *)out = half(d);
  device char *qs = (device char *)(out + 2);
  for (uint i = 0; i < 32; ++i) {
    qs[i] = char(clamp(int(rint(vals[i] * id)), -127, 127));
  }
}

kernel void dspark_quantize_f16_to_q4_0(
    device const half *src [[buffer(0)]],
    device uchar *dst [[buffer(1)]],
    constant uint &blocks [[buffer(2)]],
    uint block [[thread_position_in_grid]]) {
  if (block >= blocks) { return; }
  device const half *in = src + ulong(block) * 32;
  float vals[32];
  float max_abs = 0.0f;
  float signed_max = 0.0f;
  for (uint i = 0; i < 32; ++i) {
    const float v = float(in[i]);
    vals[i] = v;
    const float a = fabs(v);
    if (a > max_abs) {
      max_abs = a;
      signed_max = v;
    }
  }
  const float d = signed_max / -8.0f;
  const float id = d != 0.0f ? 1.0f / d : 0.0f;
  device uchar *out = dst + ulong(block) * 18;
  *(device half *)out = half(d);
  for (uint j = 0; j < 16; ++j) {
    const int q0 = clamp(int(vals[j] * id + 8.5f), 0, 15);
    const int q1 = clamp(int(vals[j + 16] * id + 8.5f), 0, 15);
    out[2 + j] = uchar(q0 | (q1 << 4));
  }
}

template <uint T, uint R, uint NSG>
kernel void dspark_q4_0_mv_rows(
    device const uchar *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant Q8MvParams &p [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort simd [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  const uint nb = p.in_dim / 32;
  const uint row_bytes = nb * 18;
  const uint first = (group.x * NSG + simd) * R;
  const uint ix = lane / 2; // 0..15 (16 blocks per step)
  const uint il = lane % 2; // 0..1 (8 bytes = 8 lo + 8 hi weights per lane)
  float sums[R][T];
  #pragma unroll
  for (uint r = 0; r < R; ++r) {
    #pragma unroll
    for (uint t = 0; t < T; ++t) {
      sums[r][t] = 0.0f;
    }
  }
  for (uint ib = ix; ib < nb; ib += 16) {
    const uint x_lo = ib * 32 + il * 8;
    const uint x_hi = x_lo + 16;
    #pragma unroll
    for (uint r = 0; r < R; ++r) {
      const uint o = first + r;
      if (o < p.out_dim) {
        device const uchar *blk = w + ulong(o) * row_bytes + ulong(ib) * 18;
        const float d = float(*(device const half *)blk);
        device const ushort *q16 = (device const ushort *)(blk + 2 + il * 8);
        const uint p0 = q16[0];
        const uint p1 = q16[1];
        const uint p2 = q16[2];
        const uint p3 = q16[3];
        const float4 lo0 = float4(int(p0 & 15) - 8, int((p0 >> 8) & 15) - 8,
                                  int(p1 & 15) - 8, int((p1 >> 8) & 15) - 8);
        const float4 lo1 = float4(int(p2 & 15) - 8, int((p2 >> 8) & 15) - 8,
                                  int(p3 & 15) - 8, int((p3 >> 8) & 15) - 8);
        const float4 hi0 = float4(int((p0 >> 4) & 15) - 8, int(p0 >> 12) - 8,
                                  int((p1 >> 4) & 15) - 8, int(p1 >> 12) - 8);
        const float4 hi1 = float4(int((p2 >> 4) & 15) - 8, int(p2 >> 12) - 8,
                                  int((p3 >> 4) & 15) - 8, int(p3 >> 12) - 8);
        #pragma unroll
        for (uint t = 0; t < T; ++t) {
          device const float4 *xl = (device const float4 *)(x + ulong(t) * p.in_dim + x_lo);
          device const float4 *xh = (device const float4 *)(x + ulong(t) * p.in_dim + x_hi);
          sums[r][t] += (dot(lo0, xl[0]) + dot(lo1, xl[1]) + dot(hi0, xh[0]) + dot(hi1, xh[1])) * d;
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
        y[ulong(t) * p.out_dim + o] = total;
      }
    }
  }
}

typedef decltype(dspark_q4_0_mv_rows<1, 4, 2>) dspark_q4_mv_fn;
template [[host_name("dspark_q4_mv_1")]] kernel dspark_q4_mv_fn dspark_q4_0_mv_rows<1, 4, 2>;
template [[host_name("dspark_q4_mv_2")]] kernel dspark_q4_mv_fn dspark_q4_0_mv_rows<2, 4, 2>;
template [[host_name("dspark_q4_mv_3")]] kernel dspark_q4_mv_fn dspark_q4_0_mv_rows<3, 4, 2>;
template [[host_name("dspark_q4_mv_4")]] kernel dspark_q4_mv_fn dspark_q4_0_mv_rows<4, 4, 2>;
template [[host_name("dspark_q4_mv_5")]] kernel dspark_q4_mv_fn dspark_q4_0_mv_rows<5, 2, 4>;
template [[host_name("dspark_q4_mv_6")]] kernel dspark_q4_mv_fn dspark_q4_0_mv_rows<6, 2, 4>;
template [[host_name("dspark_q4_mv_7")]] kernel dspark_q4_mv_fn dspark_q4_0_mv_rows<7, 2, 4>;
template [[host_name("dspark_q4_mv_8")]] kernel dspark_q4_mv_fn dspark_q4_0_mv_rows<8, 2, 4>;

// ---- Fused feature mix + RMSNorm ----

struct MixNormParams {
  uint rows;
  uint hidden;
  uint features;
  float eps;
  float w0; float w1; float w2; float w3;
};

kernel void dspark_mix_norm(
    device const float *f0 [[buffer(0)]],
    device const float *f1 [[buffer(1)]],
    device const float *f2 [[buffer(2)]],
    device const float *f3 [[buffer(3)]],
    device const float *norm [[buffer(4)]],
    device float *out [[buffer(5)]],
    constant MixNormParams &p [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[32];
  const ulong base = ulong(row) * p.hidden;
  float sumsq = 0.0f;
  for (uint i = tid; i < p.hidden; i += threads) {
    float v = p.w0 * f0[base + i] + p.w1 * f1[base + i];
    if (p.features > 2) { v += p.w2 * f2[base + i]; }
    if (p.features > 3) { v += p.w3 * f3[base + i]; }
    out[base + i] = v;
    sumsq += v * v;
  }
  sumsq = simd_sum(sumsq);
  if (lane == 0) { partial[sg] = sumsq; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint g = 0; g < (threads + 31) / 32; ++g) { total += partial[g]; }
  const float inv = rsqrt(total / float(p.hidden) + p.eps);
  for (uint i = tid; i < p.hidden; i += threads) {
    out[base + i] *= inv * norm[i];
  }
}

// ---- Fused QKV head-norm + NeoX RoPE + F16 layout ----
// For draft: qkv is [T, (heads + 2*kv_heads)*dim]; writes
//   q_out [1, heads*T, 1, dim] F16 (folded head*T + t),
//   k_out [1, kv_heads, T, dim] F16,
//   v_out [1, kv_heads, T, dim] F16.
// For inject: heads == 0, qkv is [T, 2*kv_heads*dim]; writes k_out and v_out.

struct QkvPrepParams {
  uint rows;
  uint heads;
  uint kv_heads;
  uint head_dim;
  uint start_pos;
  float eps;
  float log_theta; // ln(rope_theta)
};

kernel void dspark_qkv_prep(
    device const float *qkv [[buffer(0)]],
    device const float *q_norm [[buffer(1)]],
    device const float *k_norm [[buffer(2)]],
    device half *q_out [[buffer(3)]],
    device half *k_out [[buffer(4)]],
    device half *v_out [[buffer(5)]],
    constant QkvPrepParams &p [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[32];
  threadgroup float buf[256];
  const uint slot = group.x;
  const uint t = group.y;
  const uint total_heads = p.heads + 2 * p.kv_heads;
  if (slot >= total_heads || t >= p.rows) { return; }
  const uint stride = total_heads * p.head_dim;
  const uint d = tid;
  device const float *src = qkv + ulong(t) * stride + ulong(slot) * p.head_dim;
  if (slot >= p.heads + p.kv_heads) {
    // V: just transpose + cast to F16.
    const uint vh = slot - p.heads - p.kv_heads;
    if (d < p.head_dim) {
      v_out[(ulong(vh) * p.rows + t) * p.head_dim + d] = half(src[d]);
    }
    return;
  }
  const bool is_q = slot < p.heads;
  device const float *nw = is_q ? q_norm : k_norm;
  const float x = d < p.head_dim ? src[d] : 0.0f;
  const float s = simd_sum(x * x);
  if (lane == 0) { partial[sg] = s; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  const uint groups = (p.head_dim + 31) / 32;
  for (uint g = 0; g < groups; ++g) { total += partial[g]; }
  const float inv = rsqrt(total / float(p.head_dim) + p.eps);
  if (d < p.head_dim) {
    buf[d] = x * inv * nw[d];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint half_dim = p.head_dim / 2;
  if (d < half_dim) {
    const float inv_freq = exp(-p.log_theta * (float(2 * d) / float(p.head_dim)));
    const float angle = float(p.start_pos + t) * inv_freq;
    const float c = cos(angle);
    const float sn = sin(angle);
    const float v0 = buf[d];
    const float v1 = buf[d + half_dim];
    const half r0 = half(v0 * c - v1 * sn);
    const half r1 = half(v0 * sn + v1 * c);
    if (is_q) {
      const ulong dst_base = (ulong(slot) * p.rows + t) * p.head_dim;
      q_out[dst_base + d] = r0;
      q_out[dst_base + d + half_dim] = r1;
    } else {
      const uint kh = slot - p.heads;
      const ulong dst_base = (ulong(kh) * p.rows + t) * p.head_dim;
      k_out[dst_base + d] = r0;
      k_out[dst_base + d + half_dim] = r1;
    }
  }
}

// Unfold [1, heads*T, 1, dim] F16 attention output to [T, heads*dim] F32.
struct UnfoldParams { uint rows; uint heads; uint dim; };

kernel void dspark_attn_unfold(
    device const half *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant UnfoldParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  const uint width = p.heads * p.dim;
  if (gid >= p.rows * width) { return; }
  const uint t = gid / width;
  const uint rem = gid - t * width;
  const uint h = rem / p.dim;
  const uint d = rem - h * p.dim;
  dst[gid] = float(src[(ulong(h) * p.rows + t) * p.dim + d]);
}

// ---- Tiled GQA-shared non-causal attention for DSpark (head_dim == 128) ----
// Each threadgroup (8 simdgroups = 256 threads) stages 32 F16 K/V rows in
// threadgroup memory and evaluates 32 queries (4 per simdgroup) sharing one
// KV head, cutting KV DRAM reads by up to 32x and writing [rows, heads*128]
// F32 directly without a separate unfold pass.

struct DSparkAttendParams {
  uint rows;
  uint heads;
  uint kv_heads;
  uint gqa;
  uint q_per_kv;
  uint keys;
  uint kv_head_stride;
  uint splits;
  uint chunk;
  uint finalize;
  float scale;
};

kernel void dspark_attend_partial_128(
    device const half *q [[buffer(0)]],
    device const half *k [[buffer(1)]],
    device const half *v [[buffer(2)]],
    device float *out_or_partial_o [[buffer(3)]],
    device float2 *partial_ml [[buffer(4)]],
    constant DSparkAttendParams &p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half s_k[32][128];
  threadgroup half s_v[32][128];
  const uint split = group.x;
  const uint kh = group.y;
  const uint q_tile = group.z;
  const uint q_base = q_tile * 32 + sg * 4;

  float4 qv[4];
  float4 acc[4];
  float m[4];
  float l[4];
  #pragma unroll
  for (uint qi = 0; qi < 4; ++qi) {
    const uint q_idx = q_base + qi;
    if (q_idx < p.q_per_kv) {
      device const half4 *qp =
          (device const half4 *)(q + (ulong(kh) * p.q_per_kv + q_idx) * 128 + lane * 4);
      qv[qi] = float4(*qp) * p.scale;
    } else {
      qv[qi] = float4(0.0f);
    }
    acc[qi] = float4(0.0f);
    m[qi] = NEG_LARGE;
    l[qi] = 0.0f;
  }

  const uint begin = split * p.chunk;
  const uint end = min(p.keys, begin + p.chunk);
  device const half *k_base = k + ulong(kh) * p.kv_head_stride;
  device const half *v_base = v + ulong(kh) * p.kv_head_stride;

  for (uint tile_start = begin; tile_start < end; tile_start += 32) {
    const uint tile_len = min(32u, end - tile_start);
    #pragma unroll
    for (uint sr = sg; sr < 32; sr += 8) {
      if (sr < tile_len) {
        const ulong kv_off = ulong(tile_start + sr) * 128 + lane * 4;
        *(threadgroup half4 *)(&s_k[sr][lane * 4]) = *(device const half4 *)(k_base + kv_off);
        *(threadgroup half4 *)(&s_v[sr][lane * 4]) = *(device const half4 *)(v_base + kv_off);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 0; s < tile_len; ++s) {
      const float4 k4 = float4(*(threadgroup const half4 *)(&s_k[s][lane * 4]));
      const float4 v4 = float4(*(threadgroup const half4 *)(&s_v[s][lane * 4]));
      #pragma unroll
      for (uint qi = 0; qi < 4; ++qi) {
        const float score = simd_sum(dot(qv[qi], k4));
        const float next_m = max(m[qi], score);
        const float corr = exp(m[qi] - next_m);
        const float w = exp(score - next_m);
        l[qi] = l[qi] * corr + w;
        acc[qi] = acc[qi] * corr + w * v4;
        m[qi] = next_m;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (p.finalize != 0) {
    #pragma unroll
    for (uint qi = 0; qi < 4; ++qi) {
      const uint q_idx = q_base + qi;
      if (q_idx < p.q_per_kv) {
        const uint local_h = q_idx / p.rows;
        const uint row = q_idx - local_h * p.rows;
        const uint head = kh * p.gqa + local_h;
        const float inv_l = l[qi] > 0.0f ? (1.0f / l[qi]) : 0.0f;
        *(device float4 *)(out_or_partial_o + (ulong(row) * p.heads + head) * 128 + lane * 4) =
            acc[qi] * inv_l;
      }
    }
    return;
  }

  #pragma unroll
  for (uint qi = 0; qi < 4; ++qi) {
    const uint q_idx = q_base + qi;
    if (q_idx < p.q_per_kv) {
      const ulong q_global = ulong(kh) * p.q_per_kv + q_idx;
      const ulong slot = q_global * p.splits + split;
      *(device float4 *)(out_or_partial_o + slot * 128 + lane * 4) = acc[qi];
      if (lane == 0) {
        partial_ml[slot] = float2(m[qi], l[qi]);
      }
    }
  }
}

kernel void dspark_attend_combine_128(
    device const float *partial_o [[buffer(0)]],
    device const float2 *partial_ml [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant DSparkAttendParams &p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint total_q = p.kv_heads * p.q_per_kv;
  const uint q_global = group * 4 + sg;
  if (q_global >= total_q) { return; }
  const uint kh = q_global / p.q_per_kv;
  const uint q_idx = q_global - kh * p.q_per_kv;
  const uint local_h = q_idx / p.rows;
  const uint row = q_idx - local_h * p.rows;
  const uint head = kh * p.gqa + local_h;
  const ulong base_slot = ulong(q_global) * p.splits;

  float m_max = NEG_LARGE;
  for (uint s = 0; s < p.splits; ++s) {
    const float2 ml = partial_ml[base_slot + s];
    if (ml.y > 0.0f) {
      m_max = max(m_max, ml.x);
    }
  }
  float total_l = 0.0f;
  float4 acc = float4(0.0f);
  for (uint s = 0; s < p.splits; ++s) {
    const float2 ml = partial_ml[base_slot + s];
    if (ml.y > 0.0f) {
      const float corr = exp(ml.x - m_max);
      total_l += ml.y * corr;
      const float4 po = *(device const float4 *)(partial_o + (base_slot + s) * 128 + lane * 4);
      acc += po * corr;
    }
  }
  const float inv_l = total_l > 0.0f ? (1.0f / total_l) : 0.0f;
  *(device float4 *)(out + (ulong(row) * p.heads + head) * 128 + lane * 4) = acc * inv_l;
}

// ---- Fused Markov RNN step (rank <= 256, 1 threadgroup of 256 threads) ----

struct MarkovRnnParams { uint rank; uint vocab; uint step; uint steps; uint anchor; };

kernel void dspark_markov_rnn_step(
    device const uint *prev_id [[buffer(0)]],
    device const ushort *markov_w1 [[buffer(1)]], // [vocab, rank] BF16
    device const float *joint_rec_t [[buffer(2)]], // [3*rank, 2*rank] F32
    device const float *context [[buffer(3)]], // [steps, 3*rank] F32
    device float *state [[buffer(4)]], // [rank] F32 in/out
    device float *latent [[buffer(5)]], // [rank] F32 out
    device uint *early_ctrl [[buffer(6)]], // [3] U32: stopped, invalid, best_k
    device float *early_state [[buffer(7)]], // [4] F32: survival, expected, best_rate, _
    constant MarkovRnnParams &p [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (p.step > 0 && early_ctrl[0] != 0u) { return; }
  threadgroup float s_in[512];
  const uint rank = p.rank;
  const uint prev = min(p.step == 0 ? p.anchor : prev_id[0], p.vocab - 1);
  if (p.step == 0 && tid == 0) {
    early_ctrl[0] = 0u;
    early_ctrl[1] = 0u;
    early_ctrl[2] = p.steps;
    early_state[0] = 1.0f;
    early_state[1] = 1.0f;
    early_state[2] = 0.0f;
  }
  if (tid < rank) {
    s_in[tid] = p.step == 0 ? 0.0f : state[tid];
    const ushort b = markov_w1[ulong(prev) * rank + tid];
    s_in[rank + tid] = as_type<float>(uint(b) << 16);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid >= rank) { return; }
  const uint width = 2 * rank;
  device const float *ctx = context + ulong(p.step) * 3 * rank;
  device const float *wg = joint_rec_t + ulong(tid) * width;
  device const float *wc = joint_rec_t + ulong(rank + tid) * width;
  device const float *wl = joint_rec_t + ulong(2 * rank + tid) * width;
  float g_raw = ctx[tid];
  float c_raw = ctx[rank + tid];
  float l_raw = ctx[2 * rank + tid];
  for (uint k = 0; k < width; ++k) {
    const float v = s_in[k];
    g_raw += v * wg[k];
    c_raw += v * wc[k];
    l_raw += v * wl[k];
  }
  const float gate = 1.0f / (1.0f + exp(-clamp(g_raw, -80.0f, 80.0f)));
  const float cand = tanh(clamp(c_raw, -15.0f, 15.0f));
  state[tid] = cand + gate * (s_in[tid] - cand);
  latent[tid] = tanh(clamp(l_raw, -15.0f, 15.0f));
}

struct MarkovArgmaxParams {
  uint vocab;
  uint rank;
  uint chunks;
  uint step;
  uint steps;
  float softcap;
  float cost_base;
  float cost_slope;
};

// ---- Coalesced fused Markov score + argmax + online softmax prob (rank == 256, Q4_0 fold_t) ----
// fold_q4 is [vocab, 8 * 18] bytes (28.8 MB for 200k vocab, half of Q8_0).
// One simdgroup computes 1 vocab row: 8 blocks x 4 lanes/block = 32 lanes (4 lo + 4 hi weights/lane).
kernel void dspark_markov_argmax_partial_q4(
    device const uchar *fold_q4 [[buffer(0)]],
    device const float *latent [[buffer(1)]],
    device const float *base_step [[buffer(2)]],
    device const float *fold_bias [[buffer(3)]],
    device float *partial_scores [[buffer(4)]],
    device uint *partial_ids [[buffer(5)]],
    device float *partial_sums [[buffer(6)]],
    device atomic_uint *early_ctrl [[buffer(7)]],
    constant MarkovArgmaxParams &p [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (p.step > 0 && atomic_load_explicit(early_ctrl, memory_order_relaxed) != 0u) { return; }
  threadgroup float s_lat[256];
  threadgroup float sg_scores[8];
  threadgroup uint sg_ids[8];
  threadgroup float sg_sums[8];
  if (tid < 256) {
    s_lat[tid] = latent[tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint chunk_size = (p.vocab + p.chunks - 1) / p.chunks;
  const uint begin = group * chunk_size;
  const uint end = min(p.vocab, begin + chunk_size);
  float raw_best = -INFINITY;
  float best_score = -INFINITY;
  uint best_id = 0xffffffffu;
  float sum_exp = 0.0f;

  const uint ib = lane / 4; // 0..7
  const uint il = lane % 4; // 0..3 (4 bytes = 4 lo + 4 hi weights)
  const uint l_lo_off = ib * 32 + il * 4;
  const float4 l_lo = *(threadgroup const float4 *)(s_lat + l_lo_off);
  const float4 l_hi = *(threadgroup const float4 *)(s_lat + l_lo_off + 16);
  for (uint v = begin + sg; v < end; v += 8) {
    device const uchar *blk = fold_q4 + ulong(v) * (8 * 18) + ib * 18;
    const float d = float(*(device const half *)blk);
    device const ushort *q16 = (device const ushort *)(blk + 2 + il * 4);
    const uint p0 = q16[0];
    const uint p1 = q16[1];
    const float4 lo = float4(int(p0 & 15) - 8, int((p0 >> 8) & 15) - 8,
                             int(p1 & 15) - 8, int((p1 >> 8) & 15) - 8);
    const float4 hi = float4(int((p0 >> 4) & 15) - 8, int(p0 >> 12) - 8,
                             int((p1 >> 4) & 15) - 8, int(p1 >> 12) - 8);
    const float dot_val = (dot(lo, l_lo) + dot(hi, l_hi)) * d;
    const float bias = simd_sum(dot_val);
    if (lane == 0) {
      const float raw = base_step[v] + bias + fold_bias[v];
      if (!isfinite(raw)) {
        atomic_fetch_or_explicit(early_ctrl + 1, 1u, memory_order_relaxed);
      } else if (raw > raw_best - 14.0f) {
        const float s = p.softcap > 0.0f
            ? (p.softcap * tanh(clamp(raw / p.softcap, -15.0f, 15.0f)))
            : raw;
        if (s > best_score) {
          sum_exp = sum_exp * exp(best_score - s) + 1.0f;
          best_score = s;
          raw_best = raw;
          best_id = v;
        } else {
          if (s == best_score && v < best_id) {
            best_id = v;
          }
          sum_exp += exp(s - best_score);
        }
      }
    }
  }
  if (lane == 0) {
    sg_scores[sg] = best_score;
    sg_ids[sg] = best_id;
    sg_sums[sg] = sum_exp;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float sc = lane < 8 ? sg_scores[lane] : -INFINITY;
    const uint id = lane < 8 ? sg_ids[lane] : 0xffffffffu;
    const float sm = lane < 8 ? sg_sums[lane] : 0.0f;
    const float win_sc = simd_max(sc);
    const uint win_id = simd_min(sc == win_sc ? id : 0xffffffffu);
    const float win_sum = simd_sum(lane < 8 && isfinite(sc) ? sm * exp(sc - win_sc) : 0.0f);
    if (lane == 0) {
      partial_scores[group] = win_sc;
      partial_ids[group] = win_id;
      partial_sums[group] = win_sum;
    }
  }
}

// ---- Coalesced fused Markov score + argmax + online softmax prob (rank == 256, Q8_0 fold_t) ----
// fold_q8 is [vocab, 8 * 34] bytes (54.4 MB for 200k vocab, half of F16).
// One simdgroup computes 1 vocab row: 8 blocks x 4 lanes/block = 32 lanes.
kernel void dspark_markov_argmax_partial_q8(
    device const uchar *fold_q8 [[buffer(0)]],
    device const float *latent [[buffer(1)]],
    device const float *base_step [[buffer(2)]],
    device const float *fold_bias [[buffer(3)]],
    device float *partial_scores [[buffer(4)]],
    device uint *partial_ids [[buffer(5)]],
    device float *partial_sums [[buffer(6)]],
    device atomic_uint *early_ctrl [[buffer(7)]],
    constant MarkovArgmaxParams &p [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (p.step > 0 && atomic_load_explicit(early_ctrl, memory_order_relaxed) != 0u) { return; }
  threadgroup float s_lat[256];
  threadgroup float sg_scores[8];
  threadgroup uint sg_ids[8];
  threadgroup float sg_sums[8];
  if (tid < 256) {
    s_lat[tid] = latent[tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint chunk_size = (p.vocab + p.chunks - 1) / p.chunks;
  const uint begin = group * chunk_size;
  const uint end = min(p.vocab, begin + chunk_size);
  float raw_best = -INFINITY;
  float best_score = -INFINITY;
  uint best_id = 0xffffffffu;
  float sum_exp = 0.0f;

  const uint ib = lane / 4; // 0..7
  const uint il = lane % 4; // 0..3
  const uint l_off = ib * 32 + il * 8;
  const float4 l0 = *(threadgroup const float4 *)(s_lat + l_off);
  const float4 l1 = *(threadgroup const float4 *)(s_lat + l_off + 4);
  for (uint v = begin + sg; v < end; v += 8) {
    device const uchar *blk = fold_q8 + ulong(v) * (8 * 34) + ib * 34;
    const float d = float(*(device const half *)blk);
    device const packed_char4 *qs = (device const packed_char4 *)(blk + 2 + il * 8);
    const float dot_val = (dot(float4(qs[0]), l0) + dot(float4(qs[1]), l1)) * d;
    const float bias = simd_sum(dot_val);
    if (lane == 0) {
      const float raw = base_step[v] + bias + fold_bias[v];
      if (!isfinite(raw)) {
        atomic_fetch_or_explicit(early_ctrl + 1, 1u, memory_order_relaxed);
      } else if (raw > raw_best - 14.0f) {
        const float s = p.softcap > 0.0f
            ? (p.softcap * tanh(clamp(raw / p.softcap, -15.0f, 15.0f)))
            : raw;
        if (s > best_score) {
          sum_exp = sum_exp * exp(best_score - s) + 1.0f;
          best_score = s;
          raw_best = raw;
          best_id = v;
        } else {
          if (s == best_score && v < best_id) {
            best_id = v;
          }
          sum_exp += exp(s - best_score);
        }
      }
    }
  }
  if (lane == 0) {
    sg_scores[sg] = best_score;
    sg_ids[sg] = best_id;
    sg_sums[sg] = sum_exp;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float sc = lane < 8 ? sg_scores[lane] : -INFINITY;
    const uint id = lane < 8 ? sg_ids[lane] : 0xffffffffu;
    const float sm = lane < 8 ? sg_sums[lane] : 0.0f;
    const float win_sc = simd_max(sc);
    const uint win_id = simd_min(sc == win_sc ? id : 0xffffffffu);
    const float win_sum = simd_sum(lane < 8 && isfinite(sc) ? sm * exp(sc - win_sc) : 0.0f);
    if (lane == 0) {
      partial_scores[group] = win_sc;
      partial_ids[group] = win_id;
      partial_sums[group] = win_sum;
    }
  }
}

// ---- Coalesced fused Markov score + argmax (F16 fold_t) ----
// fold_t is [vocab, rank] F16; each simdgroup computes 1 vocab row at a time
// (32 lanes x 8 halfs = 256 elements in one coalesced 512-byte read).

kernel void dspark_markov_argmax_partial(
    device const half *fold_t [[buffer(0)]],
    device const float *latent [[buffer(1)]],
    device const float *base_step [[buffer(2)]],
    device const float *fold_bias [[buffer(3)]],
    device float *partial_scores [[buffer(4)]],
    device uint *partial_ids [[buffer(5)]],
    device float *partial_sums [[buffer(6)]],
    device atomic_uint *early_ctrl [[buffer(7)]],
    constant MarkovArgmaxParams &p [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (p.step > 0 && atomic_load_explicit(early_ctrl, memory_order_relaxed) != 0u) { return; }
  threadgroup float s_lat[256];
  threadgroup float sg_scores[8];
  threadgroup uint sg_ids[8];
  threadgroup float sg_sums[8];
  if (tid < p.rank) {
    // Match the F16 cast of latent before fold matmul.
    s_lat[tid] = float(half(latent[tid]));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint chunk_size = (p.vocab + p.chunks - 1) / p.chunks;
  const uint begin = group * chunk_size;
  const uint end = min(p.vocab, begin + chunk_size);
  float best_score = -INFINITY;
  uint best_id = 0xffffffffu;
  float sum_exp = 0.0f;

  if (p.rank == 256) {
    const float4 l0 = *(threadgroup const float4 *)(s_lat + lane * 8);
    const float4 l1 = *(threadgroup const float4 *)(s_lat + lane * 8 + 4);
    for (uint v = begin + sg; v < end; v += 8) {
      device const half4 *f4 = (device const half4 *)(fold_t + ulong(v) * 256 + lane * 8);
      const float dot_val = dot(float4(f4[0]), l0) + dot(float4(f4[1]), l1);
      const float bias = simd_sum(dot_val);
      if (lane == 0) {
        const float raw = base_step[v] + bias + fold_bias[v];
        if (!isfinite(raw)) {
          atomic_fetch_or_explicit(early_ctrl + 1, 1u, memory_order_relaxed);
        } else {
          const float s = p.softcap > 0.0f
              ? (p.softcap * tanh(clamp(raw / p.softcap, -15.0f, 15.0f)))
              : raw;
          if (s > best_score) {
            sum_exp = sum_exp * exp(best_score - s) + 1.0f;
            best_score = s;
            best_id = v;
          } else {
            if (s == best_score && v < best_id) {
              best_id = v;
            }
            sum_exp += exp(s - best_score);
          }
        }
      }
    }
  } else {
    for (uint v = begin + sg; v < end; v += 8) {
      float acc = 0.0f;
      for (uint k = lane; k < p.rank; k += 32) {
        acc += float(fold_t[ulong(v) * p.rank + k]) * s_lat[k];
      }
      const float bias = simd_sum(acc);
      if (lane == 0) {
        const float raw = base_step[v] + bias + fold_bias[v];
        if (!isfinite(raw)) {
          atomic_fetch_or_explicit(early_ctrl + 1, 1u, memory_order_relaxed);
        } else {
          const float s = p.softcap > 0.0f
              ? (p.softcap * tanh(clamp(raw / p.softcap, -15.0f, 15.0f)))
              : raw;
          if (s > best_score) {
            sum_exp = sum_exp * exp(best_score - s) + 1.0f;
            best_score = s;
            best_id = v;
          } else {
            if (s == best_score && v < best_id) {
              best_id = v;
            }
            sum_exp += exp(s - best_score);
          }
        }
      }
    }
  }
  if (lane == 0) {
    sg_scores[sg] = best_score;
    sg_ids[sg] = best_id;
    sg_sums[sg] = sum_exp;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float sc = lane < 8 ? sg_scores[lane] : -INFINITY;
    const uint id = lane < 8 ? sg_ids[lane] : 0xffffffffu;
    const float sm = lane < 8 ? sg_sums[lane] : 0.0f;
    const float win_sc = simd_max(sc);
    const uint win_id = simd_min(sc == win_sc ? id : 0xffffffffu);
    const float win_sum = simd_sum(lane < 8 && isfinite(sc) ? sm * exp(sc - win_sc) : 0.0f);
    if (lane == 0) {
      partial_scores[group] = win_sc;
      partial_ids[group] = win_id;
      partial_sums[group] = win_sum;
    }
  }
}

kernel void dspark_markov_argmax_merge(
    device const float *partial_scores [[buffer(0)]],
    device const uint *partial_ids [[buffer(1)]],
    device const float *partial_sums [[buffer(2)]],
    device uint *early_ctrl [[buffer(3)]],
    device float *early_state [[buffer(4)]],
    device uint *prev_out [[buffer(5)]],
    device uint *out_buf [[buffer(6)]], // [steps] ids, [1] valid_len, [steps] prob bits
    constant MarkovArgmaxParams &p [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (p.step > 0 && early_ctrl[0] != 0u) { return; }
  threadgroup float sg_scores[8];
  threadgroup uint sg_ids[8];
  threadgroup float sg_sums[8];
  float best_score = -INFINITY;
  uint best_id = 0xffffffffu;
  float sum_exp = 0.0f;
  for (uint i = tid; i < p.chunks; i += 256) {
    const float sc = partial_scores[i];
    const uint id = partial_ids[i];
    const float sm = partial_sums[i];
    if (!isfinite(sc)) { continue; }
    if (sc > best_score) {
      sum_exp = sum_exp * exp(best_score - sc) + sm;
      best_score = sc;
      best_id = id;
    } else {
      if (sc == best_score && id < best_id) {
        best_id = id;
      }
      sum_exp += sm * exp(sc - best_score);
    }
  }
  const float s_max = simd_max(best_score);
  const uint s_id = simd_min(best_score == s_max ? best_id : 0xffffffffu);
  const float s_sum = simd_sum(isfinite(best_score) ? sum_exp * exp(best_score - s_max) : 0.0f);
  if (lane == 0) {
    sg_scores[sg] = s_max;
    sg_ids[sg] = s_id;
    sg_sums[sg] = s_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float sc = lane < 8 ? sg_scores[lane] : -INFINITY;
    const uint id = lane < 8 ? sg_ids[lane] : 0xffffffffu;
    const float sm = lane < 8 ? sg_sums[lane] : 0.0f;
    const float win_sc = simd_max(sc);
    const uint win_id = simd_min(sc == win_sc ? id : 0xffffffffu);
    const float win_sum = simd_sum(lane < 8 && isfinite(sc) ? sm * exp(sc - win_sc) : 0.0f);
    if (lane == 0) {
      const uint chosen = (early_ctrl[1] != 0 || win_id >= p.vocab) ? p.vocab : win_id;
      const float prob = win_sum > 0.0f ? (1.0f / win_sum) : 0.0f;
      prev_out[0] = chosen;
      out_buf[p.step] = chosen;
      out_buf[p.steps + 1 + p.step] = as_type<uint>(prob);
      if (chosen >= p.vocab) {
        out_buf[p.steps] = p.step + 1;
        early_ctrl[0] = 1u;
      } else if (p.cost_slope > 0.0f) {
        const float surv = early_state[0] * clamp(prob, 0.0f, 1.0f);
        const float exp_tok = early_state[1] + surv;
        const float rate = exp_tok / (p.cost_base + p.cost_slope * float(p.step + 1));
        float best_rate = early_state[2];
        uint best_k = p.step == 0 ? 1u : early_ctrl[2];
        if (rate > best_rate) {
          best_rate = rate;
          best_k = p.step + 1;
        }
        early_state[0] = surv;
        early_state[1] = exp_tok;
        early_state[2] = best_rate;
        early_ctrl[2] = best_k;
        out_buf[p.steps] = best_k;
        if (surv <= best_rate * p.cost_slope) {
          early_ctrl[0] = 1u;
        }
      } else {
        out_buf[p.steps] = p.step + 1;
      }
    }
  }
}

// ---- Sampled Markov head for exact Leviathan speculative rejection sampling ----
// Each simdgroup tracks top-2 raw scores with zero tanh/exp inside the vocab
// loop; each chunk emits its top-4 softcapped candidates, and merge builds the
// exact normalized top-32 (temperature, top_k, top_p) draft distribution q_s,
// samples d_s ~ q_s, feeds d_s into the next RNN step, and outputs q_s.

constant constexpr uint DSPARK_SAMPLE_CHUNK_K = 4;
constant constexpr uint DSPARK_SAMPLE_TOP_K = 32;
constant constexpr uint DSPARK_SAMPLE_STEP_STRIDE = 2 + 2 * DSPARK_SAMPLE_TOP_K; // 66 u32s

struct MarkovSampleParams {
  uint vocab;
  uint rank;
  uint chunks;
  uint step;
  uint steps;
  uint top_k;
  float softcap;
  float temperature;
  float top_p;
  float step_u;
  float cost_base;
  float cost_slope;
};

inline float dspark_softcap(float raw, float cap) {
  return cap > 0.0f ? (cap * tanh(clamp(raw / cap, -15.0f, 15.0f))) : raw;
}

inline void dspark_emit_chunk_top4(
    float r0, uint i0, float r1, uint i1,
    float softcap,
    device float *partial_scores,
    device uint *partial_ids,
    uint group, uint sg, uint lane,
    threadgroup float *sg_sc,
    threadgroup uint *sg_id) {
  if (lane == 0) {
    sg_sc[sg * 2] = isfinite(r0) ? dspark_softcap(r0, softcap) : -INFINITY;
    sg_id[sg * 2] = isfinite(r0) ? i0 : 0xffffffffu;
    sg_sc[sg * 2 + 1] = isfinite(r1) ? dspark_softcap(r1, softcap) : -INFINITY;
    sg_id[sg * 2 + 1] = isfinite(r1) ? i1 : 0xffffffffu;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float my_sc = lane < 16 ? sg_sc[lane] : -INFINITY;
    uint my_id = lane < 16 ? sg_id[lane] : 0xffffffffu;
    #pragma unroll
    for (uint k = 0; k < DSPARK_SAMPLE_CHUNK_K; ++k) {
      const float w_sc = simd_max(my_sc);
      const uint w_id = simd_min(my_sc == w_sc ? my_id : 0xffffffffu);
      if (lane == 0) {
        partial_scores[group * DSPARK_SAMPLE_CHUNK_K + k] = w_sc;
        partial_ids[group * DSPARK_SAMPLE_CHUNK_K + k] = w_id;
      }
      if (my_id == w_id) {
        my_sc = -INFINITY;
        my_id = 0xffffffffu;
      }
    }
  }
}

kernel void dspark_markov_sample_partial_q4(
    device const uchar *fold_q4 [[buffer(0)]],
    device const float *latent [[buffer(1)]],
    device const float *base_step [[buffer(2)]],
    device const float *fold_bias [[buffer(3)]],
    device float *partial_scores [[buffer(4)]],
    device uint *partial_ids [[buffer(5)]],
    device atomic_uint *early_ctrl [[buffer(6)]],
    constant MarkovSampleParams &p [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (p.step > 0 && atomic_load_explicit(early_ctrl, memory_order_relaxed) != 0u) { return; }
  threadgroup float s_lat[256];
  threadgroup float sg_sc[16];
  threadgroup uint sg_id[16];
  if (tid < 256) { s_lat[tid] = latent[tid]; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint chunk_size = (p.vocab + p.chunks - 1) / p.chunks;
  const uint begin = group * chunk_size;
  const uint end = min(p.vocab, begin + chunk_size);
  float r0 = -INFINITY, r1 = -INFINITY;
  uint i0 = 0xffffffffu, i1 = 0xffffffffu;

  const uint ib = lane / 4;
  const uint il = lane % 4;
  const uint l_lo_off = ib * 32 + il * 4;
  const float4 l_lo = *(threadgroup const float4 *)(s_lat + l_lo_off);
  const float4 l_hi = *(threadgroup const float4 *)(s_lat + l_lo_off + 16);
  for (uint v = begin + sg; v < end; v += 8) {
    device const uchar *blk = fold_q4 + ulong(v) * (8 * 18) + ib * 18;
    const float d = float(*(device const half *)blk);
    device const ushort *q16 = (device const ushort *)(blk + 2 + il * 4);
    const uint p0 = q16[0];
    const uint p1 = q16[1];
    const float4 lo = float4(int(p0 & 15) - 8, int((p0 >> 8) & 15) - 8,
                             int(p1 & 15) - 8, int((p1 >> 8) & 15) - 8);
    const float4 hi = float4(int((p0 >> 4) & 15) - 8, int(p0 >> 12) - 8,
                             int((p1 >> 4) & 15) - 8, int(p1 >> 12) - 8);
    const float bias = simd_sum((dot(lo, l_lo) + dot(hi, l_hi)) * d);
    if (lane == 0) {
      const float raw = base_step[v] + bias + fold_bias[v];
      if (!isfinite(raw)) {
        atomic_fetch_or_explicit(early_ctrl + 1, 1u, memory_order_relaxed);
      } else if (raw > r0) {
        r1 = r0; i1 = i0;
        r0 = raw; i0 = v;
      } else if (raw > r1) {
        r1 = raw; i1 = v;
      }
    }
  }
  dspark_emit_chunk_top4(r0, i0, r1, i1, p.softcap, partial_scores, partial_ids,
                         group, sg, lane, sg_sc, sg_id);
}

kernel void dspark_markov_sample_partial_q8(
    device const uchar *fold_q8 [[buffer(0)]],
    device const float *latent [[buffer(1)]],
    device const float *base_step [[buffer(2)]],
    device const float *fold_bias [[buffer(3)]],
    device float *partial_scores [[buffer(4)]],
    device uint *partial_ids [[buffer(5)]],
    device atomic_uint *early_ctrl [[buffer(6)]],
    constant MarkovSampleParams &p [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (p.step > 0 && atomic_load_explicit(early_ctrl, memory_order_relaxed) != 0u) { return; }
  threadgroup float s_lat[256];
  threadgroup float sg_sc[16];
  threadgroup uint sg_id[16];
  if (tid < 256) { s_lat[tid] = latent[tid]; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint chunk_size = (p.vocab + p.chunks - 1) / p.chunks;
  const uint begin = group * chunk_size;
  const uint end = min(p.vocab, begin + chunk_size);
  float r0 = -INFINITY, r1 = -INFINITY;
  uint i0 = 0xffffffffu, i1 = 0xffffffffu;

  const uint ib = lane / 4;
  const uint il = lane % 4;
  const uint l_off = ib * 32 + il * 8;
  const float4 l0 = *(threadgroup const float4 *)(s_lat + l_off);
  const float4 l1 = *(threadgroup const float4 *)(s_lat + l_off + 4);
  for (uint v = begin + sg; v < end; v += 8) {
    device const uchar *blk = fold_q8 + ulong(v) * (8 * 34) + ib * 34;
    const float d = float(*(device const half *)blk);
    device const packed_char4 *qs = (device const packed_char4 *)(blk + 2 + il * 8);
    const float bias = simd_sum((dot(float4(qs[0]), l0) + dot(float4(qs[1]), l1)) * d);
    if (lane == 0) {
      const float raw = base_step[v] + bias + fold_bias[v];
      if (!isfinite(raw)) {
        atomic_fetch_or_explicit(early_ctrl + 1, 1u, memory_order_relaxed);
      } else if (raw > r0) {
        r1 = r0; i1 = i0;
        r0 = raw; i0 = v;
      } else if (raw > r1) {
        r1 = raw; i1 = v;
      }
    }
  }
  dspark_emit_chunk_top4(r0, i0, r1, i1, p.softcap, partial_scores, partial_ids,
                         group, sg, lane, sg_sc, sg_id);
}

kernel void dspark_markov_sample_partial(
    device const half *fold_t [[buffer(0)]],
    device const float *latent [[buffer(1)]],
    device const float *base_step [[buffer(2)]],
    device const float *fold_bias [[buffer(3)]],
    device float *partial_scores [[buffer(4)]],
    device uint *partial_ids [[buffer(5)]],
    device atomic_uint *early_ctrl [[buffer(6)]],
    constant MarkovSampleParams &p [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (p.step > 0 && atomic_load_explicit(early_ctrl, memory_order_relaxed) != 0u) { return; }
  threadgroup float s_lat[256];
  threadgroup float sg_sc[16];
  threadgroup uint sg_id[16];
  if (tid < p.rank) { s_lat[tid] = float(half(latent[tid])); }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint chunk_size = (p.vocab + p.chunks - 1) / p.chunks;
  const uint begin = group * chunk_size;
  const uint end = min(p.vocab, begin + chunk_size);
  float r0 = -INFINITY, r1 = -INFINITY;
  uint i0 = 0xffffffffu, i1 = 0xffffffffu;

  if (p.rank == 256) {
    const float4 l0 = *(threadgroup const float4 *)(s_lat + lane * 8);
    const float4 l1 = *(threadgroup const float4 *)(s_lat + lane * 8 + 4);
    for (uint v = begin + sg; v < end; v += 8) {
      device const half4 *f4 = (device const half4 *)(fold_t + ulong(v) * 256 + lane * 8);
      const float bias = simd_sum(dot(float4(f4[0]), l0) + dot(float4(f4[1]), l1));
      if (lane == 0) {
        const float raw = base_step[v] + bias + fold_bias[v];
        if (!isfinite(raw)) {
          atomic_fetch_or_explicit(early_ctrl + 1, 1u, memory_order_relaxed);
        } else if (raw > r0) {
          r1 = r0; i1 = i0;
          r0 = raw; i0 = v;
        } else if (raw > r1) {
          r1 = raw; i1 = v;
        }
      }
    }
  } else {
    for (uint v = begin + sg; v < end; v += 8) {
      float acc = 0.0f;
      for (uint k = lane; k < p.rank; k += 32) {
        acc += float(fold_t[ulong(v) * p.rank + k]) * s_lat[k];
      }
      const float bias = simd_sum(acc);
      if (lane == 0) {
        const float raw = base_step[v] + bias + fold_bias[v];
        if (!isfinite(raw)) {
          atomic_fetch_or_explicit(early_ctrl + 1, 1u, memory_order_relaxed);
        } else if (raw > r0) {
          r1 = r0; i1 = i0;
          r0 = raw; i0 = v;
        } else if (raw > r1) {
          r1 = raw; i1 = v;
        }
      }
    }
  }
  dspark_emit_chunk_top4(r0, i0, r1, i1, p.softcap, partial_scores, partial_ids,
                         group, sg, lane, sg_sc, sg_id);
}

kernel void dspark_markov_sample_merge(
    device const float *partial_scores [[buffer(0)]],
    device const uint *partial_ids [[buffer(1)]],
    device uint *early_ctrl [[buffer(2)]],
    device float *early_state [[buffer(3)]],
    device uint *prev_out [[buffer(4)]],
    device uint *out_buf [[buffer(5)]], // [1] valid_len, then steps * 66 u32s
    constant MarkovSampleParams &p [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (p.step > 0 && early_ctrl[0] != 0u) { return; }
  threadgroup float tg_sc[2048];
  threadgroup uint tg_id[2048];
  threadgroup float sg_sc[8];
  threadgroup uint sg_id[8];
  threadgroup float top_sc[DSPARK_SAMPLE_TOP_K];
  threadgroup uint top_id[DSPARK_SAMPLE_TOP_K];
  threadgroup uint cur_win_id;
  threadgroup float cur_win_sc;

  const uint total = min(p.chunks * DSPARK_SAMPLE_CHUNK_K, 2048u);
  for (uint i = tid; i < total; i += 256) {
    tg_sc[i] = partial_scores[i];
    tg_id[i] = partial_ids[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint found = 0;
  for (uint k = 0; k < DSPARK_SAMPLE_TOP_K; ++k) {
    float best_sc = -INFINITY;
    uint best_id = 0xffffffffu;
    for (uint i = tid; i < total; i += 256) {
      const float sc = tg_sc[i];
      const uint id = tg_id[i];
      if (sc > best_sc || (sc == best_sc && id < best_id)) {
        best_sc = sc;
        best_id = id;
      }
    }
    const float s_max = simd_max(best_sc);
    const uint s_id = simd_min(best_sc == s_max ? best_id : 0xffffffffu);
    if (lane == 0) {
      sg_sc[sg] = s_max;
      sg_id[sg] = s_id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      const float sc = lane < 8 ? sg_sc[lane] : -INFINITY;
      const uint id = lane < 8 ? sg_id[lane] : 0xffffffffu;
      const float w_sc = simd_max(sc);
      const uint w_id = simd_min(sc == w_sc ? id : 0xffffffffu);
      if (lane == 0) {
        cur_win_sc = w_sc;
        cur_win_id = w_id;
        top_sc[k] = w_sc;
        top_id[k] = w_id;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint w_id = cur_win_id;
    if (!isfinite(cur_win_sc) || w_id >= p.vocab) {
      break;
    }
    found = k + 1;
    for (uint i = tid; i < total; i += 256) {
      if (tg_id[i] == w_id) {
        tg_sc[i] = -INFINITY;
        tg_id[i] = 0xffffffffu;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0) {
    const uint step_off = 1 + p.step * DSPARK_SAMPLE_STEP_STRIDE;
    if (early_ctrl[1] != 0 || found == 0) {
      prev_out[0] = p.vocab;
      out_buf[step_off] = p.vocab;
      out_buf[step_off + 1] = 0;
      out_buf[0] = p.step + 1;
      early_ctrl[0] = 1u;
      return;
    }
    uint n = found;
    if (p.top_k > 0 && p.top_k < n) { n = p.top_k; }
    const float inv_t = 1.0f / max(p.temperature, 1.0e-6f);
    const float m = top_sc[0];
    float probs[DSPARK_SAMPLE_TOP_K];
    float sum_w = 0.0f;
    for (uint j = 0; j < n; ++j) {
      const float w = exp((top_sc[j] - m) * inv_t);
      probs[j] = w;
      sum_w += w;
    }
    uint kept = n;
    if (p.top_p < 1.0f && sum_w > 0.0f) {
      float cum = 0.0f;
      for (uint j = 0; j < n; ++j) {
        cum += probs[j] / sum_w;
        if (cum >= p.top_p) {
          kept = j + 1;
          break;
        }
      }
    }
    float kept_sum = 0.0f;
    for (uint j = 0; j < kept; ++j) { kept_sum += probs[j]; }
    const float inv_kept = kept_sum > 0.0f ? (1.0f / kept_sum) : 0.0f;
    for (uint j = 0; j < kept; ++j) { probs[j] *= inv_kept; }

    const float u = clamp(p.step_u, 0.0f, 0.99999994f);
    uint chosen = top_id[kept - 1];
    float cum = 0.0f;
    for (uint j = 0; j < kept; ++j) {
      cum += probs[j];
      if (u < cum) {
        chosen = top_id[j];
        break;
      }
    }
    prev_out[0] = chosen;
    out_buf[step_off] = chosen;
    out_buf[step_off + 1] = kept;
    for (uint j = 0; j < kept; ++j) {
      out_buf[step_off + 2 + j * 2] = top_id[j];
      out_buf[step_off + 2 + j * 2 + 1] = as_type<uint>(probs[j]);
    }
    if (p.cost_slope > 0.0f) {
      // At T < 1, probs[0] is already temperature-sharpened and tracks
      // rejection-sampling acceptance closely; blend with the chosen token's
      // probability so low-probability draws shorten the remaining block.
      float chosen_p = probs[0];
      for (uint j = 0; j < kept; ++j) {
        if (top_id[j] == chosen) {
          chosen_p = probs[j];
          break;
        }
      }
      const float p_est = clamp(0.75f * probs[0] + 0.25f * chosen_p, 0.0f, 0.96f);
      const float surv = early_state[0] * p_est;
      const float exp_tok = early_state[1] + surv;
      const float rate = exp_tok / (p.cost_base + p.cost_slope * float(p.step + 1));
      float best_rate = early_state[2];
      uint best_k = p.step == 0 ? 1u : early_ctrl[2];
      if (rate > best_rate) {
        best_rate = rate;
        best_k = p.step + 1;
      }
      early_state[0] = surv;
      early_state[1] = exp_tok;
      early_state[2] = best_rate;
      early_ctrl[2] = best_k;
      out_buf[0] = best_k;
      if (surv <= best_rate * p.cost_slope) {
        early_ctrl[0] = 1u;
      }
    } else {
      out_buf[0] = p.step + 1;
    }
  }
}
"#;

static LIBRARIES: OnceLock<Mutex<HashMap<DeviceId, candle_metal_kernels::metal::Library>>> =
    OnceLock::new();
static PIPELINES: OnceLock<Mutex<HashMap<(DeviceId, &'static str), ComputePipeline>>> =
    OnceLock::new();

fn library(device: &candle_core::MetalDevice) -> Result<candle_metal_kernels::metal::Library> {
    let libraries = LIBRARIES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(lib) = libraries.lock().unwrap().get(&device.id()) {
        return Ok(lib.clone());
    }
    let lib = device
        .metal_device()
        .new_library_with_source(DSPARK_MSL, None)
        .map_err(candle_core::Error::wrap)?;
    libraries.lock().unwrap().insert(device.id(), lib.clone());
    Ok(lib)
}

fn pipeline(device: &candle_core::MetalDevice, name: &'static str) -> Result<ComputePipeline> {
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (device.id(), name);
    if let Some(p) = pipelines.lock().unwrap().get(&key) {
        return Ok(p.clone());
    }
    let function = library(device)?
        .get_function(name, None)
        .map_err(candle_core::Error::wrap)?;
    let p = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    pipelines.lock().unwrap().insert(key, p.clone());
    Ok(p)
}

/// Precompile the DSpark Metal library and hot pipelines at load time so the
/// first draft request incurs no shader compilation stall.
pub(super) fn precompile(device: &candle_core::MetalDevice) -> Result<()> {
    let _ = library(device)?;
    for name in Q8_MV_KERNELS
        .iter()
        .chain(Q8_SWIGLU_KERNELS.iter())
        .chain(Q4_MV_KERNELS.iter())
    {
        pipeline(device, name)?;
    }
    for name in [
        "dspark_mix_norm",
        "dspark_qkv_prep",
        "dspark_attn_unfold",
        "dspark_attend_partial_128",
        "dspark_attend_combine_128",
        "dspark_markov_rnn_step",
        "dspark_markov_argmax_partial",
        "dspark_markov_argmax_partial_q8",
        "dspark_markov_argmax_partial_q4",
        "dspark_markov_argmax_merge",
        "dspark_markov_sample_partial",
        "dspark_markov_sample_partial_q8",
        "dspark_markov_sample_partial_q4",
        "dspark_markov_sample_merge",
    ] {
        pipeline(device, name)?;
    }
    Ok(())
}

fn metal_view(tensor: &Tensor, dtype: DType, what: &str) -> Result<(Buffer, usize)> {
    if tensor.dtype() != dtype || !tensor.is_contiguous() {
        candle_core::bail!("DSpark Metal {what} must be contiguous {dtype:?}")
    }
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Metal(storage) = &*storage else {
        candle_core::bail!("DSpark Metal {what} must be on Metal")
    };
    Ok((
        storage.buffer().clone(),
        layout.start_offset() * dtype.size_in_bytes(),
    ))
}

fn metal_device(tensor: &Tensor) -> Result<candle_core::MetalDevice> {
    match tensor.device() {
        candle_core::Device::Metal(device) => Ok(device.clone()),
        _ => candle_core::bail!("DSpark Metal kernel requires Metal device"),
    }
}

fn wrap(
    device: &candle_core::MetalDevice,
    buffer: Arc<Buffer>,
    dtype: DType,
    shape: impl Into<Shape>,
) -> Tensor {
    let shape = shape.into();
    let elements = shape.elem_count();
    Tensor::from((
        Storage::Metal(MetalStorage::new(buffer, device.clone(), elements, dtype)),
        shape,
    ))
}

fn new_buf(
    device: &candle_core::MetalDevice,
    elements: usize,
    dtype: DType,
    label: &str,
) -> Result<Arc<Buffer>> {
    device
        .new_buffer_builder()
        .with_size_for(elements.max(1), dtype)
        .with_label(label)
        .build()
}

fn to_u32(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| candle_core::Error::Msg(format!("DSpark {what} too large")))
}

const Q8_MV_KERNELS: [&str; 8] = [
    "dspark_q8_mv_1",
    "dspark_q8_mv_2",
    "dspark_q8_mv_3",
    "dspark_q8_mv_4",
    "dspark_q8_mv_5",
    "dspark_q8_mv_6",
    "dspark_q8_mv_7",
    "dspark_q8_mv_8",
];

const Q8_SWIGLU_KERNELS: [&str; 8] = [
    "dspark_q8_swiglu_1",
    "dspark_q8_swiglu_2",
    "dspark_q8_swiglu_3",
    "dspark_q8_swiglu_4",
    "dspark_q8_swiglu_5",
    "dspark_q8_swiglu_6",
    "dspark_q8_swiglu_7",
    "dspark_q8_swiglu_8",
];

const Q4_MV_KERNELS: [&str; 8] = [
    "dspark_q4_mv_1",
    "dspark_q4_mv_2",
    "dspark_q4_mv_3",
    "dspark_q4_mv_4",
    "dspark_q4_mv_5",
    "dspark_q4_mv_6",
    "dspark_q4_mv_7",
    "dspark_q4_mv_8",
];

#[repr(C)]
struct Q8MvParams {
    in_dim: u32,
    out_dim: u32,
    flags: u32,
}

/// Multi-row Q8_0 matvec reading each weight block once across `1 <= rows <= 8`.
pub(super) fn q8_0_matvec_rows(
    weights_u8: &Tensor,
    out_dim: usize,
    in_dim: usize,
    x: &Tensor,
    residual: Option<&Tensor>,
    swiglu: bool,
) -> Result<Tensor> {
    let device = metal_device(weights_u8)?;
    let (rows, width) = x.dims2()?;
    if width != in_dim
        || !in_dim.is_multiple_of(32)
        || !(1..=8).contains(&rows)
        || (swiglu && (!out_dim.is_multiple_of(2) || residual.is_some()))
    {
        candle_core::bail!("DSpark Q8_0 multi-row matvec geometry mismatch")
    }
    let x = x.contiguous()?;
    let res_held = residual.map(Tensor::contiguous).transpose()?;
    let (w_buf, w_off) = metal_view(weights_u8, DType::U8, "Q8_0 weights")?;
    let (x_buf, x_off) = metal_view(&x, DType::F32, "Q8_0 input")?;
    let res_view = res_held
        .as_ref()
        .map(|r| metal_view(r, DType::F32, "Q8_0 residual"))
        .transpose()?;
    let out_cols = if swiglu { out_dim / 2 } else { out_dim };
    let y = new_buf(&device, rows * out_cols, DType::F32, "dspark_q8_mv")?;
    let params = Q8MvParams {
        in_dim: to_u32(in_dim, "in_dim")?,
        out_dim: to_u32(out_dim, "out_dim")?,
        flags: u32::from(res_view.is_some()),
    };
    let name = if swiglu {
        Q8_SWIGLU_KERNELS[rows - 1]
    } else {
        Q8_MV_KERNELS[rows - 1]
    };
    let nsg = if rows <= 4 { 2 } else { 4 };
    let tiles = if swiglu {
        out_cols.div_ceil(nsg)
    } else {
        out_dim.div_ceil(4)
    };
    let encoder = device.command_encoder()?;
    encoder.set_label(name);
    let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
    enc.set_compute_pipeline_state(&pipeline(&device, name)?);
    enc.set_input_buffer(0, Some(&w_buf), w_off);
    enc.set_input_buffer(1, Some(&x_buf), x_off);
    match &res_view {
        Some((rb, ro)) => enc.set_input_buffer(2, Some(rb), *ro),
        None => enc.set_input_buffer(2, Some(&x_buf), x_off),
    }
    enc.set_output_buffer(3, Some(&y), 0);
    enc.set_bytes(4, &params);
    enc.dispatch_thread_groups(
        objc2_metal::MTLSize {
            width: tiles,
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: nsg * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(wrap(&device, y, DType::F32, (rows, out_cols)))
}

/// Quantize a BF16 `[rows, cols]` weight matrix on Metal to packed Q4_0 bytes.
pub(super) fn quantize_bf16_to_q4_0(weight: &Tensor) -> Result<Tensor> {
    let device = metal_device(weight)?;
    let (rows, cols) = weight.dims2()?;
    if !cols.is_multiple_of(32) || weight.dtype() != DType::BF16 || !weight.is_contiguous() {
        candle_core::bail!("DSpark Q4_0 quantizer expects contiguous BF16 [rows, mult_of_32]")
    }
    let blocks = rows * (cols / 32);
    let (src_buf, src_off) = metal_view(weight, DType::BF16, "BF16 head")?;
    let out = new_buf(&device, blocks * 18, DType::U8, "dspark_head_q4_0")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("dspark_quantize_bf16_to_q4_0");
    let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
    enc.set_compute_pipeline_state(&pipeline(&device, "dspark_quantize_bf16_to_q4_0")?);
    enc.set_input_buffer(0, Some(&src_buf), src_off);
    enc.set_output_buffer(1, Some(&out), 0);
    enc.set_bytes(2, &to_u32(blocks, "Q4_0 blocks")?);
    enc.dispatch_threads(
        objc2_metal::MTLSize {
            width: blocks,
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(wrap(&device, out, DType::U8, blocks * 18))
}

/// Quantize a BF16 `[rows, cols]` weight matrix on Metal to packed Q8_0 bytes.
pub(super) fn quantize_bf16_to_q8_0(weight: &Tensor) -> Result<Tensor> {
    let device = metal_device(weight)?;
    let (rows, cols) = weight.dims2()?;
    if !cols.is_multiple_of(32) || weight.dtype() != DType::BF16 || !weight.is_contiguous() {
        candle_core::bail!("DSpark Q8_0 quantizer expects contiguous BF16 [rows, mult_of_32]")
    }
    let blocks = rows * (cols / 32);
    let (src_buf, src_off) = metal_view(weight, DType::BF16, "BF16 head")?;
    let out = new_buf(&device, blocks * 34, DType::U8, "dspark_head_q8_0")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("dspark_quantize_bf16_to_q8_0");
    let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
    enc.set_compute_pipeline_state(&pipeline(&device, "dspark_quantize_bf16_to_q8_0")?);
    enc.set_input_buffer(0, Some(&src_buf), src_off);
    enc.set_output_buffer(1, Some(&out), 0);
    enc.set_bytes(2, &to_u32(blocks, "Q8_0 blocks")?);
    enc.dispatch_threads(
        objc2_metal::MTLSize {
            width: blocks,
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(wrap(&device, out, DType::U8, blocks * 34))
}

/// Quantize an F16 `[rows, cols]` matrix on Metal to packed Q8_0 bytes.
pub(super) fn quantize_f16_to_q8_0(weight: &Tensor) -> Result<Tensor> {
    let device = metal_device(weight)?;
    let (rows, cols) = weight.dims2()?;
    if !cols.is_multiple_of(32) || weight.dtype() != DType::F16 || !weight.is_contiguous() {
        candle_core::bail!("DSpark F16->Q8_0 quantizer expects contiguous F16 [rows, mult_of_32]")
    }
    let blocks = rows * (cols / 32);
    let (src_buf, src_off) = metal_view(weight, DType::F16, "F16 fold_t")?;
    let out = new_buf(&device, blocks * 34, DType::U8, "dspark_fold_q8_0")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("dspark_quantize_f16_to_q8_0");
    let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
    enc.set_compute_pipeline_state(&pipeline(&device, "dspark_quantize_f16_to_q8_0")?);
    enc.set_input_buffer(0, Some(&src_buf), src_off);
    enc.set_output_buffer(1, Some(&out), 0);
    enc.set_bytes(2, &to_u32(blocks, "Q8_0 blocks")?);
    enc.dispatch_threads(
        objc2_metal::MTLSize {
            width: blocks,
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(wrap(&device, out, DType::U8, blocks * 34))
}

/// Quantize an F16 `[rows, cols]` matrix on Metal to packed Q4_0 bytes.
pub(super) fn quantize_f16_to_q4_0(weight: &Tensor) -> Result<Tensor> {
    let device = metal_device(weight)?;
    let (rows, cols) = weight.dims2()?;
    if !cols.is_multiple_of(32) || weight.dtype() != DType::F16 || !weight.is_contiguous() {
        candle_core::bail!("DSpark F16->Q4_0 quantizer expects contiguous F16 [rows, mult_of_32]")
    }
    let blocks = rows * (cols / 32);
    let (src_buf, src_off) = metal_view(weight, DType::F16, "F16 fold_t")?;
    let out = new_buf(&device, blocks * 18, DType::U8, "dspark_fold_q4_0")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("dspark_quantize_f16_to_q4_0");
    let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
    enc.set_compute_pipeline_state(&pipeline(&device, "dspark_quantize_f16_to_q4_0")?);
    enc.set_input_buffer(0, Some(&src_buf), src_off);
    enc.set_output_buffer(1, Some(&out), 0);
    enc.set_bytes(2, &to_u32(blocks, "Q4_0 blocks")?);
    enc.dispatch_threads(
        objc2_metal::MTLSize {
            width: blocks,
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(wrap(&device, out, DType::U8, blocks * 18))
}

/// Multi-row Q4_0 matvec reading each weight block once across `1 <= rows <= 8`.
pub(super) fn q4_0_matvec_rows(
    weights_u8: &Tensor,
    out_dim: usize,
    in_dim: usize,
    x: &Tensor,
) -> Result<Tensor> {
    let device = metal_device(weights_u8)?;
    let (rows, width) = x.dims2()?;
    if width != in_dim || !in_dim.is_multiple_of(32) || !(1..=8).contains(&rows) {
        candle_core::bail!("DSpark Q4_0 multi-row matvec geometry mismatch")
    }
    let x = x.contiguous()?;
    let (w_buf, w_off) = metal_view(weights_u8, DType::U8, "Q4_0 weights")?;
    let (x_buf, x_off) = metal_view(&x, DType::F32, "Q4_0 input")?;
    let y = new_buf(&device, rows * out_dim, DType::F32, "dspark_q4_mv")?;
    let params = Q8MvParams {
        in_dim: to_u32(in_dim, "in_dim")?,
        out_dim: to_u32(out_dim, "out_dim")?,
        flags: 0,
    };
    let name = Q4_MV_KERNELS[rows - 1];
    let threads = if rows <= 4 { 64 } else { 128 };
    let encoder = device.command_encoder()?;
    encoder.set_label(name);
    let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
    enc.set_compute_pipeline_state(&pipeline(&device, name)?);
    enc.set_input_buffer(0, Some(&w_buf), w_off);
    enc.set_input_buffer(1, Some(&x_buf), x_off);
    enc.set_output_buffer(2, Some(&y), 0);
    enc.set_bytes(3, &params);
    enc.dispatch_thread_groups(
        objc2_metal::MTLSize {
            width: out_dim.div_ceil(8),
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(wrap(&device, y, DType::F32, (rows, out_dim)))
}

#[repr(C)]
struct MixNormParams {
    rows: u32,
    hidden: u32,
    features: u32,
    eps: f32,
    w0: f32,
    w1: f32,
    w2: f32,
    w3: f32,
}

/// Fused weighted sum of 2..=4 target feature tensors + RMSNorm.
pub(super) fn mix_norm(
    features: &[Tensor],
    weights: &[f32],
    norm: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    if !(2..=4).contains(&features.len()) || weights.len() != features.len() {
        candle_core::bail!("DSpark mix_norm supports 2..=4 features")
    }
    let device = metal_device(&features[0])?;
    let (rows, hidden) = features[0].dims2()?;
    let held: Vec<Tensor> = features
        .iter()
        .map(|f| f.to_dtype(DType::F32)?.contiguous())
        .collect::<Result<_>>()?;
    let norm_held = norm.to_dtype(DType::F32)?.contiguous()?;
    let views: Vec<(Buffer, usize)> = held
        .iter()
        .map(|f| metal_view(f, DType::F32, "mix feature"))
        .collect::<Result<_>>()?;
    let (n_buf, n_off) = metal_view(&norm_held, DType::F32, "mix norm")?;
    let out = new_buf(&device, rows * hidden, DType::F32, "dspark_mix_norm")?;
    let params = MixNormParams {
        rows: to_u32(rows, "mix rows")?,
        hidden: to_u32(hidden, "mix hidden")?,
        features: features.len() as u32,
        eps,
        w0: weights[0],
        w1: weights[1],
        w2: weights.get(2).copied().unwrap_or(0.0),
        w3: weights.get(3).copied().unwrap_or(0.0),
    };
    let f2 = views.get(2).unwrap_or(&views[0]);
    let f3 = views.get(3).unwrap_or(&views[0]);
    let encoder = device.command_encoder()?;
    encoder.set_label("dspark_mix_norm");
    let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
    enc.set_compute_pipeline_state(&pipeline(&device, "dspark_mix_norm")?);
    enc.set_input_buffer(0, Some(&views[0].0), views[0].1);
    enc.set_input_buffer(1, Some(&views[1].0), views[1].1);
    enc.set_input_buffer(2, Some(&f2.0), f2.1);
    enc.set_input_buffer(3, Some(&f3.0), f3.1);
    enc.set_input_buffer(4, Some(&n_buf), n_off);
    enc.set_output_buffer(5, Some(&out), 0);
    enc.set_bytes(6, &params);
    enc.dispatch_thread_groups(
        objc2_metal::MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(wrap(&device, out, DType::F32, (rows, hidden)))
}

#[repr(C)]
struct QkvPrepParams {
    rows: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    eps: f32,
    log_theta: f32,
}

/// Fused Q/K head RMSNorm + NeoX RoPE + F16 transpose from stacked `[rows, (heads + 2*kv_heads)*dim]`.
/// When `heads > 0`, returns `(Some(q_folded [1, heads*rows, 1, dim] F16), k [1, kv_heads, rows, dim] F16, v [1, kv_heads, rows, dim] F16)`.
/// When `heads == 0`, returns `(None, k, v)`.
#[allow(clippy::too_many_arguments)]
pub(super) fn qkv_prep(
    qkv: &Tensor,
    q_norm: Option<&Tensor>,
    k_norm: &Tensor,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    start_pos: usize,
    eps: f32,
    rope_theta: f64,
) -> Result<(Option<Tensor>, Tensor, Tensor)> {
    let device = metal_device(qkv)?;
    let (rows, width) = qkv.dims2()?;
    let total_heads = heads + 2 * kv_heads;
    if width != total_heads * head_dim || head_dim > 256 || !head_dim.is_multiple_of(2) {
        candle_core::bail!("DSpark qkv_prep geometry mismatch")
    }
    let qkv = qkv.contiguous()?;
    let kn_held = k_norm.to_dtype(DType::F32)?.contiguous()?;
    let qn_held = q_norm
        .map(|n| n.to_dtype(DType::F32)?.contiguous())
        .transpose()?;
    let (qkv_buf, qkv_off) = metal_view(&qkv, DType::F32, "qkv")?;
    let (kn_buf, kn_off) = metal_view(&kn_held, DType::F32, "k_norm")?;
    let qn_view = qn_held
        .as_ref()
        .map(|n| metal_view(n, DType::F32, "q_norm"))
        .transpose()?;
    let q_out = if heads > 0 {
        Some(new_buf(
            &device,
            heads * rows * head_dim,
            DType::F16,
            "dspark_q_folded",
        )?)
    } else {
        None
    };
    let k_out = new_buf(&device, kv_heads * rows * head_dim, DType::F16, "dspark_k")?;
    let v_out = new_buf(&device, kv_heads * rows * head_dim, DType::F16, "dspark_v")?;
    let params = QkvPrepParams {
        rows: to_u32(rows, "qkv rows")?,
        heads: to_u32(heads, "heads")?,
        kv_heads: to_u32(kv_heads, "kv_heads")?,
        head_dim: to_u32(head_dim, "head_dim")?,
        start_pos: to_u32(start_pos, "start_pos")?,
        eps,
        log_theta: (rope_theta as f32).ln(),
    };
    let encoder = device.command_encoder()?;
    encoder.set_label("dspark_qkv_prep");
    let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
    enc.set_compute_pipeline_state(&pipeline(&device, "dspark_qkv_prep")?);
    enc.set_input_buffer(0, Some(&qkv_buf), qkv_off);
    match &qn_view {
        Some((b, o)) => enc.set_input_buffer(1, Some(b), *o),
        None => enc.set_input_buffer(1, Some(&kn_buf), kn_off),
    }
    enc.set_input_buffer(2, Some(&kn_buf), kn_off);
    enc.set_output_buffer(3, Some(q_out.as_ref().unwrap_or(&k_out)), 0);
    enc.set_output_buffer(4, Some(&k_out), 0);
    enc.set_output_buffer(5, Some(&v_out), 0);
    enc.set_bytes(6, &params);
    let threads = head_dim.div_ceil(32) * 32;
    enc.dispatch_thread_groups(
        objc2_metal::MTLSize {
            width: total_heads,
            height: rows,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok((
        q_out.map(|b| wrap(&device, b, DType::F16, (1, heads * rows, 1, head_dim))),
        wrap(&device, k_out, DType::F16, (1, kv_heads, rows, head_dim)),
        wrap(&device, v_out, DType::F16, (1, kv_heads, rows, head_dim)),
    ))
}

#[repr(C)]
struct UnfoldParams {
    rows: u32,
    heads: u32,
    dim: u32,
}

/// Unfold `[1, heads * rows, 1, dim]` F16 attention output into `[rows, heads * dim]` F32.
pub(super) fn attn_unfold(src: &Tensor, rows: usize, heads: usize, dim: usize) -> Result<Tensor> {
    let device = metal_device(src)?;
    let src = src.contiguous()?;
    let (s_buf, s_off) = metal_view(&src, DType::F16, "attn_unfold src")?;
    let elements = rows * heads * dim;
    let dst = new_buf(&device, elements, DType::F32, "dspark_attn_unfold")?;
    let params = UnfoldParams {
        rows: to_u32(rows, "rows")?,
        heads: to_u32(heads, "heads")?,
        dim: to_u32(dim, "dim")?,
    };
    let encoder = device.command_encoder()?;
    encoder.set_label("dspark_attn_unfold");
    let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
    enc.set_compute_pipeline_state(&pipeline(&device, "dspark_attn_unfold")?);
    enc.set_input_buffer(0, Some(&s_buf), s_off);
    enc.set_output_buffer(1, Some(&dst), 0);
    enc.set_bytes(2, &params);
    enc.dispatch_threads(
        objc2_metal::MTLSize {
            width: elements,
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(wrap(&device, dst, DType::F32, (rows, heads * dim)))
}

#[repr(C)]
struct DSparkAttendParams {
    rows: u32,
    heads: u32,
    kv_heads: u32,
    gqa: u32,
    q_per_kv: u32,
    keys: u32,
    kv_head_stride: u32,
    splits: u32,
    chunk: u32,
    finalize: u32,
    scale: f32,
}

/// Tiled GQA-shared non-causal attention for DSpark (`head_dim == 128`).
///
/// `q_folded` is `[1, heads * rows, 1, 128]` F16 (contiguous); `k` and `v` are
/// `[1, kv_heads, keys, 128]` F16 (may be narrowed views into preallocated KV
/// storage). Returns `[rows, heads * 128]` F32 directly.
pub(super) fn attend_128(
    q_folded: &Tensor,
    k: &Tensor,
    v: &Tensor,
    rows: usize,
    heads: usize,
    kv_heads: usize,
    scale: f32,
) -> Result<Tensor> {
    let device = metal_device(q_folded)?;
    if kv_heads == 0 || !heads.is_multiple_of(kv_heads) || rows == 0 {
        candle_core::bail!("DSpark attend_128 invalid head/row geometry")
    }
    let q_folded = q_folded.contiguous()?;
    let (qb, qh, qs, qd) = q_folded.dims4()?;
    let (kb, kvh, keys, kd) = k.dims4()?;
    let (vb, vvh, vkeys, vd) = v.dims4()?;
    if qb != 1
        || qh != heads * rows
        || qs != 1
        || qd != 128
        || kb != 1
        || vb != 1
        || kvh != kv_heads
        || vvh != kv_heads
        || keys != vkeys
        || kd != 128
        || vd != 128
        || keys == 0
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
    {
        candle_core::bail!("DSpark attend_128 tensor shape/dtype mismatch")
    }
    let (q_buf, q_off) = metal_view(&q_folded, DType::F16, "attend_128 q")?;
    let (k_buf, k_off, k_head_stride) = {
        let (storage, layout) = k.storage_and_layout();
        let Storage::Metal(storage) = &*storage else {
            candle_core::bail!("DSpark attend_128 k must be on Metal")
        };
        let s = layout.stride();
        if s[3] != 1 || s[2] != 128 {
            candle_core::bail!("DSpark attend_128 k inner strides must be contiguous")
        }
        (storage.buffer().clone(), layout.start_offset() * 2, s[1])
    };
    let (v_buf, v_off, v_head_stride) = {
        let (storage, layout) = v.storage_and_layout();
        let Storage::Metal(storage) = &*storage else {
            candle_core::bail!("DSpark attend_128 v must be on Metal")
        };
        let s = layout.stride();
        if s[3] != 1 || s[2] != 128 {
            candle_core::bail!("DSpark attend_128 v inner strides must be contiguous")
        }
        (storage.buffer().clone(), layout.start_offset() * 2, s[1])
    };
    if k_head_stride != v_head_stride {
        candle_core::bail!("DSpark attend_128 k/v head strides differ")
    }
    let gqa = heads / kv_heads;
    let q_per_kv = gqa * rows;
    let q_tiles = q_per_kv.div_ceil(32);
    let wanted = (192 / (kv_heads * q_tiles)).max(1);
    let splits = keys.div_ceil(64).min(wanted).clamp(1, 64);
    let chunk = keys.div_ceil(splits);
    let splits = keys.div_ceil(chunk);
    let params = DSparkAttendParams {
        rows: to_u32(rows, "rows")?,
        heads: to_u32(heads, "heads")?,
        kv_heads: to_u32(kv_heads, "kv_heads")?,
        gqa: to_u32(gqa, "gqa")?,
        q_per_kv: to_u32(q_per_kv, "q_per_kv")?,
        keys: to_u32(keys, "keys")?,
        kv_head_stride: to_u32(k_head_stride, "kv_head_stride")?,
        splits: to_u32(splits, "splits")?,
        chunk: to_u32(chunk, "chunk")?,
        finalize: u32::from(splits == 1),
        scale,
    };
    let out_buf = new_buf(&device, rows * heads * 128, DType::F32, "dspark_attend_out")?;
    let p_part = pipeline(&device, "dspark_attend_partial_128")?;
    if splits == 1 {
        let encoder = device.command_encoder()?;
        encoder.set_label("dspark_attend_partial_128");
        let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        enc.set_compute_pipeline_state(&p_part);
        enc.set_input_buffer(0, Some(&q_buf), q_off);
        enc.set_input_buffer(1, Some(&k_buf), k_off);
        enc.set_input_buffer(2, Some(&v_buf), v_off);
        enc.set_output_buffer(3, Some(&out_buf), 0);
        enc.set_output_buffer(4, Some(&out_buf), 0);
        enc.set_bytes(5, &params);
        enc.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: 1,
                height: kv_heads,
                depth: q_tiles,
            },
            objc2_metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        return Ok(wrap(&device, out_buf, DType::F32, (rows, heads * 128)));
    }
    let total_q = kv_heads * q_per_kv;
    let partial_o = new_buf(
        &device,
        total_q * splits * 128,
        DType::F32,
        "dspark_attend_po",
    )?;
    let partial_ml = new_buf(
        &device,
        total_q * splits * 2,
        DType::F32,
        "dspark_attend_pml",
    )?;
    let p_comb = pipeline(&device, "dspark_attend_combine_128")?;
    {
        let encoder = device.command_encoder()?;
        encoder.set_label("dspark_attend_128");
        let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        enc.set_compute_pipeline_state(&p_part);
        enc.set_input_buffer(0, Some(&q_buf), q_off);
        enc.set_input_buffer(1, Some(&k_buf), k_off);
        enc.set_input_buffer(2, Some(&v_buf), v_off);
        enc.set_output_buffer(3, Some(&partial_o), 0);
        enc.set_output_buffer(4, Some(&partial_ml), 0);
        enc.set_bytes(5, &params);
        enc.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: splits,
                height: kv_heads,
                depth: q_tiles,
            },
            objc2_metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        enc.set_compute_pipeline_state(&p_comb);
        enc.set_input_buffer(0, Some(&partial_o), 0);
        enc.set_input_buffer(1, Some(&partial_ml), 0);
        enc.set_output_buffer(2, Some(&out_buf), 0);
        enc.set_bytes(3, &params);
        enc.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: total_q.div_ceil(4),
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }
    Ok(wrap(&device, out_buf, DType::F32, (rows, heads * 128)))
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum FoldFormat {
    F16,
    Q8_0,
    Q4_0,
}

#[repr(C)]
struct MarkovRnnParams {
    rank: u32,
    vocab: u32,
    step: u32,
    steps: u32,
    anchor: u32,
}

#[repr(C)]
struct MarkovArgmaxParams {
    vocab: u32,
    rank: u32,
    chunks: u32,
    step: u32,
    steps: u32,
    softcap: f32,
    cost_base: f32,
    cost_slope: f32,
}

/// Run the `steps`-step Markov head loop on Metal with optional GPU-side exact
/// mathematical early stopping (`cost_slope > 0`), returning `(drafted_ids, drafted_probs)`
/// truncated to the optimal prefix length in a single host readback.
#[allow(clippy::too_many_arguments)]
pub(super) fn markov_decode_steps(
    anchor: u32,
    steps: usize,
    rank: usize,
    vocab: usize,
    softcap: f32,
    cost_base: f32,
    cost_slope: f32,
    markov_w1: &Tensor,   // [vocab_rows, rank] BF16
    joint_rec_t: &Tensor, // [3*rank, 2*rank] F32
    context: &Tensor,     // [steps, 3*rank] F32
    base: &Tensor,        // [steps, vocab] F32
    fold_t: &Tensor,      // [vocab, rank] F16, Q8_0 U8, or Q4_0 U8
    fold_format: FoldFormat,
    fold_bias: &Tensor, // [1, vocab] F32
) -> Result<(Vec<u32>, Vec<f32>)> {
    if rank > 256 || steps == 0 {
        candle_core::bail!("DSpark Metal Markov head expects 0 < steps and rank <= 256")
    }
    if fold_format != FoldFormat::F16 && rank != 256 {
        candle_core::bail!("DSpark quantized fold_t requires rank 256")
    }
    let device = metal_device(fold_t)?;
    let markov_w1 = markov_w1.contiguous()?;
    let joint_rec_t = joint_rec_t.contiguous()?;
    let context = context.contiguous()?;
    let base = base.contiguous()?;
    let fold_t = fold_t.contiguous()?;
    let fold_bias = fold_bias.contiguous()?;
    let (w1_buf, w1_off) = metal_view(&markov_w1, DType::BF16, "markov_w1")?;
    let (jr_buf, jr_off) = metal_view(&joint_rec_t, DType::F32, "joint_rec_t")?;
    let (ctx_buf, ctx_off) = metal_view(&context, DType::F32, "markov context")?;
    let (base_buf, base_off) = metal_view(&base, DType::F32, "markov base")?;
    let (ft_buf, ft_off) = metal_view(
        &fold_t,
        if fold_format == FoldFormat::F16 {
            DType::F16
        } else {
            DType::U8
        },
        "fold_t",
    )?;
    let (fb_buf, fb_off) = metal_view(&fold_bias, DType::F32, "fold_bias")?;

    let state_buf = new_buf(&device, rank, DType::F32, "dspark_state")?;
    let prev_buf = new_buf(&device, 1, DType::U32, "dspark_prev")?;
    let early_ctrl = new_buf(&device, 3, DType::U32, "dspark_early_ctrl")?;
    let early_state = new_buf(&device, 4, DType::F32, "dspark_early_state")?;
    let latent = new_buf(&device, rank, DType::F32, "dspark_latent")?;
    let out_buf = new_buf(&device, 2 * steps + 1, DType::U32, "dspark_out")?;
    let chunks: usize = 512.min(vocab.div_ceil(8)).max(1);
    let partial_scores = new_buf(&device, chunks, DType::F32, "dspark_p_scores")?;
    let partial_ids = new_buf(&device, chunks, DType::U32, "dspark_p_ids")?;
    let partial_sums = new_buf(&device, chunks, DType::F32, "dspark_p_sums")?;

    let p_rnn = pipeline(&device, "dspark_markov_rnn_step")?;
    let p_part = pipeline(
        &device,
        match fold_format {
            FoldFormat::Q4_0 => "dspark_markov_argmax_partial_q4",
            FoldFormat::Q8_0 => "dspark_markov_argmax_partial_q8",
            FoldFormat::F16 => "dspark_markov_argmax_partial",
        },
    )?;
    let p_merge = pipeline(&device, "dspark_markov_argmax_merge")?;

    {
        let encoder = device.command_encoder()?;
        encoder.set_label("dspark_markov_loop");
        let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        for step in 0..steps {
            let rnn_params = MarkovRnnParams {
                rank: to_u32(rank, "rank")?,
                vocab: to_u32(vocab, "vocab")?,
                step: to_u32(step, "step")?,
                steps: to_u32(steps, "steps")?,
                anchor,
            };
            enc.set_compute_pipeline_state(&p_rnn);
            enc.set_input_buffer(0, Some(&prev_buf), 0);
            enc.set_input_buffer(1, Some(&w1_buf), w1_off);
            enc.set_input_buffer(2, Some(&jr_buf), jr_off);
            enc.set_input_buffer(3, Some(&ctx_buf), ctx_off);
            enc.set_output_buffer(4, Some(&state_buf), 0);
            enc.set_output_buffer(5, Some(&latent), 0);
            enc.set_output_buffer(6, Some(&early_ctrl), 0);
            enc.set_output_buffer(7, Some(&early_state), 0);
            enc.set_bytes(8, &rnn_params);
            enc.dispatch_thread_groups(
                objc2_metal::MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );

            let am_params = MarkovArgmaxParams {
                vocab: to_u32(vocab, "vocab")?,
                rank: to_u32(rank, "rank")?,
                chunks: to_u32(chunks, "chunks")?,
                step: to_u32(step, "step")?,
                steps: to_u32(steps, "steps")?,
                softcap,
                cost_base,
                cost_slope,
            };
            let base_step_off = base_off + step * vocab * DType::F32.size_in_bytes();
            enc.set_compute_pipeline_state(&p_part);
            enc.set_input_buffer(0, Some(&ft_buf), ft_off);
            enc.set_input_buffer(1, Some(&latent), 0);
            enc.set_input_buffer(2, Some(&base_buf), base_step_off);
            enc.set_input_buffer(3, Some(&fb_buf), fb_off);
            enc.set_output_buffer(4, Some(&partial_scores), 0);
            enc.set_output_buffer(5, Some(&partial_ids), 0);
            enc.set_output_buffer(6, Some(&partial_sums), 0);
            enc.set_output_buffer(7, Some(&early_ctrl), 0);
            enc.set_bytes(8, &am_params);
            enc.dispatch_thread_groups(
                objc2_metal::MTLSize {
                    width: chunks,
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );

            enc.set_compute_pipeline_state(&p_merge);
            enc.set_input_buffer(0, Some(&partial_scores), 0);
            enc.set_input_buffer(1, Some(&partial_ids), 0);
            enc.set_input_buffer(2, Some(&partial_sums), 0);
            enc.set_output_buffer(3, Some(&early_ctrl), 0);
            enc.set_output_buffer(4, Some(&early_state), 0);
            enc.set_output_buffer(5, Some(&prev_buf), 0);
            enc.set_output_buffer(6, Some(&out_buf), 0);
            enc.set_bytes(7, &am_params);
            enc.dispatch_thread_groups(
                objc2_metal::MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }
    }
    let raw = wrap(&device, out_buf, DType::U32, 2 * steps + 1).to_vec1::<u32>()?;
    let valid_len = (raw[steps] as usize).clamp(1, steps);
    let ids = raw[..valid_len].to_vec();
    let probs = raw[steps + 1..steps + 1 + valid_len]
        .iter()
        .map(|&bits| f32::from_bits(bits))
        .collect();
    Ok((ids, probs))
}

#[repr(C)]
struct MarkovSampleParams {
    vocab: u32,
    rank: u32,
    chunks: u32,
    step: u32,
    steps: u32,
    top_k: u32,
    softcap: f32,
    temperature: f32,
    top_p: f32,
    step_u: f32,
    cost_base: f32,
    cost_slope: f32,
}

const SAMPLE_CHUNK_K: usize = 4;
const SAMPLE_TOP_K: usize = 32;
const SAMPLE_STEP_STRIDE: usize = 2 + 2 * SAMPLE_TOP_K;

/// Run the sampled `steps`-step Markov head on Metal for exact Leviathan
/// speculative rejection sampling, returning `(drafted_ids, draft_distributions)`
/// in a single host readback.
#[allow(clippy::too_many_arguments)]
pub(super) fn markov_sample_steps(
    anchor: u32,
    steps: usize,
    rank: usize,
    vocab: usize,
    softcap: f32,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    step_u: &[f32; 8],
    cost_base: f32,
    cost_slope: f32,
    markov_w1: &Tensor,
    joint_rec_t: &Tensor,
    context: &Tensor,
    base: &Tensor,
    fold_t: &Tensor,
    fold_format: FoldFormat,
    fold_bias: &Tensor,
) -> Result<(Vec<u32>, Vec<Vec<(u32, f32)>>)> {
    if rank > 256 || steps == 0 || steps > 8 {
        candle_core::bail!("DSpark Metal sampled Markov head expects 1..=8 steps and rank <= 256")
    }
    if fold_format != FoldFormat::F16 && rank != 256 {
        candle_core::bail!("DSpark quantized fold_t requires rank 256")
    }
    let device = metal_device(fold_t)?;
    let markov_w1 = markov_w1.contiguous()?;
    let joint_rec_t = joint_rec_t.contiguous()?;
    let context = context.contiguous()?;
    let base = base.contiguous()?;
    let fold_t = fold_t.contiguous()?;
    let fold_bias = fold_bias.contiguous()?;
    let (w1_buf, w1_off) = metal_view(&markov_w1, DType::BF16, "markov_w1")?;
    let (jr_buf, jr_off) = metal_view(&joint_rec_t, DType::F32, "joint_rec_t")?;
    let (ctx_buf, ctx_off) = metal_view(&context, DType::F32, "markov context")?;
    let (base_buf, base_off) = metal_view(&base, DType::F32, "markov base")?;
    let (ft_buf, ft_off) = metal_view(
        &fold_t,
        if fold_format == FoldFormat::F16 {
            DType::F16
        } else {
            DType::U8
        },
        "fold_t",
    )?;
    let (fb_buf, fb_off) = metal_view(&fold_bias, DType::F32, "fold_bias")?;

    let state_buf = new_buf(&device, rank, DType::F32, "dspark_state")?;
    let prev_buf = new_buf(&device, 1, DType::U32, "dspark_prev")?;
    let early_ctrl = new_buf(&device, 3, DType::U32, "dspark_early_ctrl")?;
    let early_state = new_buf(&device, 4, DType::F32, "dspark_early_state")?;
    let latent = new_buf(&device, rank, DType::F32, "dspark_latent")?;
    let out_elems = 1 + steps * SAMPLE_STEP_STRIDE;
    let out_buf = new_buf(&device, out_elems, DType::U32, "dspark_sample_out")?;
    let chunks: usize = 512.min(vocab.div_ceil(8)).max(1);
    let partial_scores = new_buf(
        &device,
        chunks * SAMPLE_CHUNK_K,
        DType::F32,
        "dspark_sp_scores",
    )?;
    let partial_ids = new_buf(
        &device,
        chunks * SAMPLE_CHUNK_K,
        DType::U32,
        "dspark_sp_ids",
    )?;

    let p_rnn = pipeline(&device, "dspark_markov_rnn_step")?;
    let p_part = pipeline(
        &device,
        match fold_format {
            FoldFormat::Q4_0 => "dspark_markov_sample_partial_q4",
            FoldFormat::Q8_0 => "dspark_markov_sample_partial_q8",
            FoldFormat::F16 => "dspark_markov_sample_partial",
        },
    )?;
    let p_merge = pipeline(&device, "dspark_markov_sample_merge")?;

    {
        let encoder = device.command_encoder()?;
        encoder.set_label("dspark_markov_sample_loop");
        let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        for step in 0..steps {
            let rnn_params = MarkovRnnParams {
                rank: to_u32(rank, "rank")?,
                vocab: to_u32(vocab, "vocab")?,
                step: to_u32(step, "step")?,
                steps: to_u32(steps, "steps")?,
                anchor,
            };
            enc.set_compute_pipeline_state(&p_rnn);
            enc.set_input_buffer(0, Some(&prev_buf), 0);
            enc.set_input_buffer(1, Some(&w1_buf), w1_off);
            enc.set_input_buffer(2, Some(&jr_buf), jr_off);
            enc.set_input_buffer(3, Some(&ctx_buf), ctx_off);
            enc.set_output_buffer(4, Some(&state_buf), 0);
            enc.set_output_buffer(5, Some(&latent), 0);
            enc.set_output_buffer(6, Some(&early_ctrl), 0);
            enc.set_output_buffer(7, Some(&early_state), 0);
            enc.set_bytes(8, &rnn_params);
            enc.dispatch_thread_groups(
                objc2_metal::MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );

            let sp_params = MarkovSampleParams {
                vocab: to_u32(vocab, "vocab")?,
                rank: to_u32(rank, "rank")?,
                chunks: to_u32(chunks, "chunks")?,
                step: to_u32(step, "step")?,
                steps: to_u32(steps, "steps")?,
                top_k: to_u32(top_k.min(SAMPLE_TOP_K), "top_k")?,
                softcap,
                temperature,
                top_p,
                step_u: step_u[step],
                cost_base,
                cost_slope,
            };
            let base_step_off = base_off + step * vocab * DType::F32.size_in_bytes();
            enc.set_compute_pipeline_state(&p_part);
            enc.set_input_buffer(0, Some(&ft_buf), ft_off);
            enc.set_input_buffer(1, Some(&latent), 0);
            enc.set_input_buffer(2, Some(&base_buf), base_step_off);
            enc.set_input_buffer(3, Some(&fb_buf), fb_off);
            enc.set_output_buffer(4, Some(&partial_scores), 0);
            enc.set_output_buffer(5, Some(&partial_ids), 0);
            enc.set_output_buffer(6, Some(&early_ctrl), 0);
            enc.set_bytes(7, &sp_params);
            enc.dispatch_thread_groups(
                objc2_metal::MTLSize {
                    width: chunks,
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );

            enc.set_compute_pipeline_state(&p_merge);
            enc.set_input_buffer(0, Some(&partial_scores), 0);
            enc.set_input_buffer(1, Some(&partial_ids), 0);
            enc.set_output_buffer(2, Some(&early_ctrl), 0);
            enc.set_output_buffer(3, Some(&early_state), 0);
            enc.set_output_buffer(4, Some(&prev_buf), 0);
            enc.set_output_buffer(5, Some(&out_buf), 0);
            enc.set_bytes(6, &sp_params);
            enc.dispatch_thread_groups(
                objc2_metal::MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }
    }
    let raw = wrap(&device, out_buf, DType::U32, out_elems).to_vec1::<u32>()?;
    let valid_len = (raw[0] as usize).clamp(1, steps);
    let mut ids = Vec::with_capacity(valid_len);
    let mut dists = Vec::with_capacity(valid_len);
    for step in 0..valid_len {
        let base = 1 + step * SAMPLE_STEP_STRIDE;
        let chosen = raw[base];
        let kept = (raw[base + 1] as usize).min(SAMPLE_TOP_K);
        let mut dist = Vec::with_capacity(kept);
        for j in 0..kept {
            let id = raw[base + 2 + j * 2];
            let prob = f32::from_bits(raw[base + 2 + j * 2 + 1]);
            dist.push((id, prob));
        }
        ids.push(chosen);
        dists.push(dist);
    }
    Ok((ids, dists))
}
