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
  float sx = 0.0f;
  float sm = 0.0f;
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
  device float *o = out + major_row(p, row, (p.flags & 2u) != 0) * 2 * p.hidden;
  for (uint i = tid; i < p.hidden; i += threads) {
    o[i] = xr[i] * ix;
    o[p.hidden + i] = mr[i] * im;
  }
}

struct GateTransposeParams { uint tokens; uint heads; uint dim; uint gate_stride; };

// out[t, h * dim + d] = attended[h, t, d] * sigmoid(gate[t, h * dim + d]),
// where gate rows are `gate_stride` apart.
kernel void inkling_gate_transpose(
    device const float *attended [[buffer(0)]],
    device const float *gate [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant GateTransposeParams &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint width = p.heads * p.dim;
  if (gid >= p.tokens * width) { return; }
  const uint t = gid / width;
  const uint j = gid - t * width;
  const uint h = j / p.dim;
  const uint d = j - h * p.dim;
  const float value = attended[(h * p.tokens + t) * p.dim + d];
  out[gid] = value / (1.0f + exp(-gate[t * p.gate_stride + j]));
}

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
  const float gate = p.beta * tanh(g / p.beta) / (1.0f + exp(-g));
  out[gid] = gate * p.linear_beta * tanh(u / p.linear_beta);
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
template <uint D>
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
  constexpr uint SG = 8;
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
  for (uint i = 0; i < PER_LANE; ++i) { q[i] = row[lane + 32 * i]; acc[i] = 0.0f; }
  float running_max = NEG_LARGE;
  float running_sum = 0.0f;
  const int query = p.first_query + int(token);
  const uint begin = split * p.chunk;
  const uint end = min(p.keys, begin + p.chunk);
  for (uint s = begin + sg; s < end; s += SG) {
    const int distance = query - (p.first_key + int(s));
    if (distance < 0 || (p.window > 0 && distance >= int(p.window))) { continue; }
    float dot = 0.0f;
    for (uint i = 0; i < PER_LANE; ++i) { dot += q[i] * float(kh[s * D + lane + 32 * i]); }
    dot = simd_sum(dot);
    const float score = dot + table[min(uint(distance), p.extent)];
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

template [[host_name("inkling_attention_partial_64")]] kernel decltype(inkling_attention_partial<64>) inkling_attention_partial<64>;
template [[host_name("inkling_attention_partial_128")]] kernel decltype(inkling_attention_partial<128>) inkling_attention_partial<128>;
template [[host_name("inkling_attention_partial_256")]] kernel decltype(inkling_attention_partial<256>) inkling_attention_partial<256>;

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
  for (uint slot = 0; slot < SLOTS; ++slot) { acc[slot] = 0.0f; }
  for (uint i = tid; i < p.hidden; i += threads) {
    float qv[Q];
    for (uint q = 0; q < Q; ++q) { qv[q] = queries[q * p.hidden + i]; }
    for (uint s = 0; s < S; ++s) {
      const float v = src.row[s][i];
      acc[s] += v * v;
      for (uint q = 0; q < Q; ++q) { acc[(q + 1) * S + s] += v * qv[q]; }
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
  float q_clip; float rel_clip; float q_post; uint flags;
};

// From the stacked query-side projection rows: the scaled, clipped queries
// [tokens][heads][dim] and, per (token, head), the attention bias by distance
// (extent + 1 columns, from the clipped relative coefficients times `basis`
// [rel][extent + 1]) followed by the sink logit. One threadgroup of
// QUERY_PREP_TOKENS simdgroups per (head, block of QUERY_PREP_TOKENS tokens,
// slice of bias columns): simdgroup j handles token j of the block, and each
// thread keeps its basis column in registers across the block's tokens.
// Slice 0 also writes the queries and sinks. R is the relative width rounded
// up (16 or 32). Flag 1 = times token_scale[token] (the log scale).
constant constexpr uint QUERY_PREP_TOKENS = 8;

template <uint R>
kernel void inkling_query_prep(
    device const float *x [[buffer(0)]],
    device const float *q_scale [[buffer(1)]],
    device const float *rel_scale [[buffer(2)]],
    device const float *token_scale [[buffer(3)]],
    device const float *basis [[buffer(4)]],
    device float *q_out [[buffer(5)]],
    device float *bias [[buffer(6)]],
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
      float a = 0.0f;
      for (uint r = 0; r < R; ++r) { a += rel[j][r] * column[r]; }
      bias[((first + j) * p.heads + head) * (p.extent + 2) + e] = a;
    }
  }
  if (slice != 0 || !active) { return; }
  device float *b = bias + (token * p.heads + head) * (p.extent + 2);
  if (lane == 0) { b[p.extent + 1] = row[p.sink_offset + head]; }
  float sumsq = 0.0f;
  for (uint d = lane; d < p.dim; d += 32) {
    const float v = row[p.q_offset + head * p.dim + d] * q_scale[head * p.dim + d];
    sumsq += v * v;
  }
  const float factor = min(p.q_clip / (sqrt(simd_sum(sumsq)) + 1.0e-6f), 1.0f) * ts * p.q_post;
  for (uint d = lane; d < p.dim; d += 32) {
    const float v = row[p.q_offset + head * p.dim + d] * q_scale[head * p.dim + d];
    q_out[(token * p.heads + head) * p.dim + d] = v * factor;
  }
}

template [[host_name("inkling_query_prep_16")]] kernel decltype(inkling_query_prep<16>) inkling_query_prep<16>;
template [[host_name("inkling_query_prep_32")]] kernel decltype(inkling_query_prep<32>) inkling_query_prep<32>;

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
// [width - 1][channels]. Flag 4 = a past state (else zeros).
kernel void inkling_post_conv(
    device const float *x [[buffer(0)]],
    device const float *route_weights [[buffer(1)]],
    device const float *past [[buffer(2)]],
    device const float *weight [[buffer(3)]],
    device const float *residual [[buffer(4)]],
    device float *out [[buffer(5)]],
    device float *state [[buffer(6)]],
    constant PostConvParams &p [[buffer(7)]],
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
  float acc = post_history(x, route_weights, past, p, t + history, c);
  for (uint j = 0; j < p.width; ++j) {
    acc += weight[c * p.width + j] * post_history(x, route_weights, past, p, t + j, c);
  }
  acc *= p.scale;
  if ((p.flags & 2u) != 0) { acc += residual[t * p.channels + c]; }
  out[t * p.channels + c] = acc;
}

struct GateNormParams { uint rows; uint hidden; float eps; uint groups; uint x_groups; uint flags; uint weight_rows; };

// out = rms((1 - tanh(f)) * x + tanh(g) * m) * weight[(r % groups) % weight_rows]
// for token-major row r (token r / groups), one threadgroup per row. fg is
// group-major [groups][tokens] when flag 1 is set; out is group-major when
// flag 2 is set; x has x_groups rows per token.
kernel void inkling_gate_combine_norm(
    device const float *fg [[buffer(0)]],
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
  device const float *f = fg + fg_row * 2 * p.hidden;
  device const float *xr = x + x_row * p.hidden;
  device const float *mr = m + row * p.hidden;
  device float *o = out + out_row * p.hidden;
  float sumsq = 0.0f;
  for (uint i = tid; i < p.hidden; i += threads) {
    const float value = (1.0f - tanh(f[i])) * xr[i] + tanh(f[p.hidden + i]) * mr[i];
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
    let output = device
        .new_buffer_builder()
        .with_size_for(elements.max(1), DType::F32)
        .with_label(name)
        .build()?;
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

/// The Q, K and V loop gates for `x` `[tokens, 1 or 3, hidden]` (one input
/// shared by Q, K and V, or one each) and `messages` `[tokens, 3, hidden]`,
/// with `down` and `forget_gain` `[3, rank, 2 * hidden]`, each gated row then
/// RMS-normalized and times `norm` row `g` (`[3, hidden]`, or `[1, hidden]`
/// for all). `eps` serves both norms. Returns group-major `[3, tokens, hidden]`.
pub(super) fn message_gates(
    x: &Tensor,
    messages: &Tensor,
    down: &Tensor,
    forget_gain: &Tensor,
    eps: f32,
    norm: &Tensor,
) -> Result<Tensor> {
    let (tokens, x_groups, hidden) = x.dims3()?;
    let groups = Groups {
        groups: 3,
        x_groups,
    };
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
    let projected = joined.matmul(&down.transpose(1, 2)?)?.matmul(forget_gain)?;
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
pub(super) const FUSED_ATTENTION_MAX_TOKENS: usize = 512;

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
        // Enough splits to fill the GPU for a few queries, few for long
        // chunks. A single split also finishes the output, with no merge.
        let wanted = (TARGET_THREADGROUPS / (tokens * heads)).max(1);
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
        let (partial_name, combine_name) = match head_dim {
            64 => (
                "inkling_attention_partial_64",
                "inkling_attention_combine_64",
            ),
            128 => (
                "inkling_attention_partial_128",
                "inkling_attention_combine_128",
            ),
            _ => (
                "inkling_attention_partial_256",
                "inkling_attention_combine_256",
            ),
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
                (1, heads, tokens),
                256,
                Shape::from((tokens, heads, head_dim)),
            );
        }
        let (partial, _) = launch(
            device,
            partial_name,
            &inputs,
            tokens * heads * splits * (head_dim + 2),
            &params,
            (splits, heads, tokens),
            256,
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

fn new_output(
    device: &candle_core::MetalDevice,
    elements: usize,
    label: &str,
) -> Result<std::sync::Arc<Buffer>> {
    device
        .new_buffer_builder()
        .with_size_for(elements.max(1), DType::F32)
        .with_label(label)
        .build()
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
#[allow(clippy::too_many_arguments)]
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
    };
    // One simdgroup per token of a block; see the kernel.
    let threads = 32 * QUERY_PREP_TOKENS;
    let name = if rel <= 16 {
        "inkling_query_prep_16"
    } else {
        "inkling_query_prep_32"
    };
    let q_out = new_output(&device, tokens * heads * dim, name)?;
    let bias = new_output(&device, tokens * heads * (extent + 2), "inkling_query_bias")?;
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
            (extent + 1).div_ceil(threads),
        ),
        threads,
    )?;
    Ok((
        wrap_output(&device, q_out, DType::F32, (tokens, heads, dim)),
        wrap_output(&device, bias, DType::F32, (tokens, heads, extent + 2)),
    ))
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
            | (u32::from(past.is_some()) << 2),
    };
    let history = width - 1;
    let out = new_output(&device, tokens * channels, "inkling_post_conv")?;
    let state = new_output(&device, history * channels, "inkling_post_state")?;
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
        &[(&out, 0), (&state, 0)],
        &params,
        (elements.div_ceil(256), 1, 1),
        256,
    )?;
    Ok((
        wrap_output(&device, out, DType::F32, (tokens, channels)),
        wrap_output(&device, state, DType::F32, (history, channels)),
    ))
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
    let kept = [forget_gain, x, message, weight]
        .into_iter()
        .map(held)
        .collect::<Result<Vec<_>>>()?;
    let fg = tensor_buffer(&kept[0], "gate projections")?;
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
    let out = new_output(&device, rows * hidden, "inkling_gate_combine_norm")?;
    encode(
        &device,
        "inkling_gate_combine_norm",
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
