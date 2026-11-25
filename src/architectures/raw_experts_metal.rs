//! Metal kernels for raw expert banks (NVFP4, MXFP8 and IQ4_NL, laid out as
//! in `raw_experts`).
//!
//! A few tokens run a vector kernel that widens each weight to F32 exactly
//! and applies the per-expert output scale. Prompts reuse the compact
//! indexed matmul (`indexed_moe_metal`) with these block types; it stages
//! weights in half precision, as it does for the GGML types, and the scale
//! is applied afterwards.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use candle_core::metal_backend::DeviceId;
use candle_core::{DType, MetalStorage, Result, Shape, Storage, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputePipeline, Library};

use super::raw_experts::RawFormat;

const RAW_EXPERTS_MSL: &str = r#"
// NVFP4: four UE4M3 scales, then 32 bytes; byte j of 16-weight group g holds
// weight j (low nibble) and j + 8 (high nibble) of that group.
typedef struct { uchar scales[4]; uchar qs[32]; } block_nvfp4_raw;
// MXFP8 in memory: the UE8M0 scale, three padding bytes, 32 E4M3 weights.
typedef struct { uchar scale; uchar pad[3]; uchar qs[32]; } block_mxfp8_raw;

// E2M1 values doubled, as ggml's kvalues_fp4; raw_ue4m3 halves the scale.
constant float raw_fp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
constant float raw_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                1, 13, 25, 38, 53, 69, 89, 113};

// 2^e for -126 <= e <= 127, exactly.
inline float raw_pow2(int e) { return as_type<float>(uint(e + 127) << 23); }

// ggml's UE4M3 scale, halved; 0 and 0x7f read as zero, the sign is ignored.
inline float raw_ue4m3(uchar x) {
  if (x == 0 || x == 0x7f) {
    return 0.0f;
  }
  const int e = (x >> 3) & 15;
  const float m = float(x & 7);
  return e == 0 ? m * raw_pow2(-10) : (8.0f + m) * raw_pow2(e - 11);
}

inline float raw_e4m3(uchar x) {
  const int e = (x >> 3) & 15;
  const float m = float(x & 7);
  const float v = e == 0 ? m * raw_pow2(-9) : (8.0f + m) * raw_pow2(e - 10);
  return (x & 0x80) ? -v : v;
}

// UE8M0: 2^(x - 127); 2^-127 reads as zero.
inline float raw_e8m0(uchar x) { return as_type<float>(uint(x) << 23); }

// The compact matmul's loaders: 16 weights of sub-block il.
template <typename type4x4>
void dequantize_nvfp4_raw(device const block_nvfp4_raw *xb, short il, thread type4x4 &reg) {
  const float s = raw_ue4m3(xb->scales[il]);
  device const uchar *q = xb->qs + 8 * il;
  for (int i = 0; i < 8; ++i) {
    reg[i / 4][i % 4] = raw_fp4[q[i] & 15] * s;
    reg[(i + 8) / 4][(i + 8) % 4] = raw_fp4[q[i] >> 4] * s;
  }
}

template <typename type4x4>
void dequantize_mxfp8_raw(device const block_mxfp8_raw *xb, short il, thread type4x4 &reg) {
  const float s = raw_e8m0(xb->scale);
  for (int i = 0; i < 16; ++i) {
    reg[i / 4][i % 4] = raw_e4m3(xb->qs[16 * il + i]) * s;
  }
}

template [[host_name("dinf_raw_mm_id_nvfp4")]]
kernel dinf_compact_mm_id_t
    dinf_kernel_mul_mm_id_compact<block_nvfp4_raw, 4, dequantize_nvfp4_raw>;
template [[host_name("dinf_raw_mm_id_mxfp8")]]
kernel dinf_compact_mm_id_t
    dinf_kernel_mul_mm_id_compact<block_mxfp8_raw, 2, dequantize_mxfp8_raw>;
template [[host_name("dinf_raw_mm_id_iq4_nl")]]
kernel dinf_compact_mm_id_t
    dinf_kernel_mul_mm_id_compact<block_iq4_nl, 2, dequantize_iq4_nl>;

// The vector kernel's formats: `dot` is the product of 16 weights of group
// g with x as four float4s. E2M1 and E4M3 decode without a table: their
// magnitude bits placed in a half's exponent and mantissa give the value
// times 2^-14 (E2M1) or 2^-8 (E4M3), the same decode as MLX's. IQ4_NL reads
// a 256-entry table (two values per byte) from threadgroup memory; a
// constant-memory table serializes when lanes read different entries.
inline half raw_fp4_half(uint n) {
  return as_type<half>(ushort(((n & 8u) << 12) | ((n & 7u) << 9)));
}

// Low and high nibbles of the four bytes of w, in byte order, times 2^-14.
inline half4 raw_fp4_lo_h(uint w) {
  const ushort4 b = ushort4(as_type<uchar4>(w));
  return as_type<half4>(((b & ushort4(8)) << 12) | ((b & ushort4(7)) << 9));
}
inline half4 raw_fp4_hi_h(uint w) {
  const ushort4 b = ushort4(as_type<uchar4>(w)) >> 4;
  return as_type<half4>(((b & ushort4(8)) << 12) | ((b & ushort4(7)) << 9));
}
inline float4 raw_fp4_lo(uint w) { return float4(raw_fp4_lo_h(w)); }
inline float4 raw_fp4_hi(uint w) { return float4(raw_fp4_hi_h(w)); }

inline half raw_e4m3_half(uint b) {
  return as_type<half>(ushort(((b & 0x80) << 8) | ((b & 0x7f) << 7)));
}
// The four E4M3 bytes of w, in byte order, times 2^-8.
inline half4 raw_e4m3_4_h(uint w) {
  const ushort4 b = ushort4(as_type<uchar4>(w));
  return as_type<half4>(((b & ushort4(0x80)) << 8) | ((b & ushort4(0x7f)) << 7));
}
inline float4 raw_e4m3_4(uint w) { return float4(raw_e4m3_4_h(w)); }

inline float raw_sum4(float4 v) { return (v.x + v.y) + (v.z + v.w); }
inline float raw_sum4_h(half4 v) { return float((v.x + v.y) + (v.z + v.w)); }

struct raw_nvfp4 {
  static constexpr constant uint BLOCK = 64;
  static constexpr constant uint BYTES = 36;
  static constexpr constant bool TABLE = false;
  static float2 entry(uint) { return float2(0.0f); }
  // Byte j of the group holds weights j (low nibble) and j + 8 (high).
  static void load(device const uchar *block, uint g, thread float4 *w,
                   thread float &gs, threadgroup const float2 *) {
    const uint2 q = *(device const packed_uint2 *)(block + 4 + 8 * g);
    const uint w0 = q.x;
    const uint w1 = q.y;
    w[0] = raw_fp4_lo(w0);
    w[1] = raw_fp4_lo(w1);
    w[2] = raw_fp4_hi(w0);
    w[3] = raw_fp4_hi(w1);
    // raw_ue4m3 is the halved scale; 2^15 undoes it and the decode's 2^-14.
    const uchar sc = block[g];
    const half sh = (sc == 0x7fu) ? 0.0h : as_type<half>(ushort((sc & 0x7fu) << 7)) * half(128.0h);
    gs = float(sh) * 32768.0f;
  }
  static float dot(device const uchar *block, uint g, thread const float4 *x,
                   threadgroup const float2 *lut) {
    float4 w[4];
    float gs;
    load(block, g, w, gs, lut);
    const float4 acc = (x[0] * w[0] + x[1] * w[1]) + (x[2] * w[2] + x[3] * w[3]);
    return raw_sum4(acc) * gs;
  }
};

struct raw_mxfp8 {
  static constexpr constant uint BLOCK = 32;
  static constexpr constant uint BYTES = 36;
  static constexpr constant bool TABLE = false;
  static float2 entry(uint) { return float2(0.0f); }
  static void load(device const uchar *block, uint g, thread float4 *w,
                   thread float &gs, threadgroup const float2 *) {
    const uint4 q = *(device const packed_uint4 *)(block + 4 + 16 * g);
    w[0] = raw_e4m3_4(q.x);
    w[1] = raw_e4m3_4(q.y);
    w[2] = raw_e4m3_4(q.z);
    w[3] = raw_e4m3_4(q.w);
    gs = raw_e8m0(block[0]) * 256.0f;
  }
  static float dot(device const uchar *block, uint g, thread const float4 *x,
                   threadgroup const float2 *lut) {
    float4 w[4];
    float gs;
    load(block, g, w, gs, lut);
    const float4 acc = (x[0] * w[0] + x[1] * w[1]) + (x[2] * w[2] + x[3] * w[3]);
    return raw_sum4(acc) * gs;
  }
};

struct raw_iq4_nl {
  static constexpr constant uint BLOCK = 32;
  static constexpr constant uint BYTES = 18;
  static constexpr constant bool TABLE = true;
  static float2 entry(uint i) { return float2(raw_iq4nl[i & 15], raw_iq4nl[i >> 4]); }
  // Group 0 is the low nibbles of the 16 bytes, group 1 the high ones.
  static void load(device const uchar *block, uint g, thread float4 *w,
                   thread float &gs, threadgroup const float2 *lut) {
    device const ushort *q = (device const ushort *)(block + 2);
    for (uint h = 0; h < 4; ++h) {
      const uint p0 = q[2 * h];
      const uint p1 = q[2 * h + 1];
      const float2 a = lut[p0 & 255], b = lut[p0 >> 8], c = lut[p1 & 255], d = lut[p1 >> 8];
      w[h] = g == 0 ? float4(a.x, b.x, c.x, d.x) : float4(a.y, b.y, c.y, d.y);
    }
    gs = float(*(device const half *)block);
  }
  static float dot(device const uchar *block, uint g, thread const float4 *x,
                   threadgroup const float2 *lut) {
    float4 w[4];
    float gs;
    load(block, g, w, gs, lut);
    float4 acc = 0.0f;
    for (uint h = 0; h < 4; ++h) {
      acc += x[h] * w[h];
    }
    return raw_sum4(acc) * gs;
  }
};

struct RawMvParams {
  uint in_dim; uint out_dim; uint top_k; uint input_rows; uint experts; uint has_scales;
  ulong row_bytes; ulong expert_bytes; float beta; float linear_beta;
};

constant constexpr uint RAW_MV_ROWS = 2;
constant constexpr uint RAW_MV_SIMDGROUPS = 2;
constant constexpr uint RAW_MV1_SITU_SG = 4;
constant constexpr uint RAW_MV1_DOWN_SG = 2;
constant constexpr uint GROUPED_MV_SIMDGROUPS = 4;

// out[route][row] = scale[e] * W_e[row] . x for e = ids[route], with lanes
// spread over the row's 16-weight groups.
// - SITU: 16 lanes per row (lanes 0..15 = gate row r, lanes 16..31 = up row
//   half + r), so in_dim=2304 (144 groups) runs 9 exact iterations with zero
//   warp tail and half the register pressure.
// - !SITU: 8 lanes per row (4 rows per simdgroup), so in_dim=1152 (72 groups)
//   runs 9 exact iterations with zero warp tail and 4x L1 broadcast on x.
template <typename F, bool SITU>
kernel void dinf_raw_mv(
    device const uchar *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device const uint *ids [[buffer(2)]],
    device const float *scales [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant RawMvParams &p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  constexpr uint NSG = SITU ? RAW_MV1_SITU_SG : RAW_MV1_DOWN_SG;
  threadgroup float2 lut[F::TABLE ? 256 : 1];
  if (F::TABLE) {
    for (uint i = tid; i < 256; i += NSG * 32) {
      lut[i] = F::entry(i);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  constexpr uint GROUPS = F::BLOCK / 16;
  const uint route = group.y;
  const uint token = route / p.top_k;
  const uint expert = min(ids[route], p.experts - 1);
  device const float *x = input + ulong(p.input_rows == 1 ? token : route) * p.in_dim;
  device const uchar *bank = weights + ulong(expert) * p.expert_bytes;
  const uint groups = p.in_dim / 16;
  const float scale = p.has_scales != 0 ? scales[expert] : 1.0f;

  if (SITU) {
    const uint half_rows = p.out_dim / 2;
    const uint r = group.x * RAW_MV1_SITU_SG + simd;
    const uint sub = lane & 15u;
    const uint row = r < half_rows ? (lane < 16u ? r : half_rows + r) : p.out_dim;
    float sum = 0.0f;
    if (row < p.out_dim) {
      device const uchar *row_ptr = bank + ulong(row) * p.row_bytes;
      for (uint gi = sub; gi < groups; gi += 16u) {
        device const float4 *x4 = (device const float4 *)(x + 16u * gi);
        const float4 xg[4] = {x4[0], x4[1], x4[2], x4[3]};
        const uint b = gi / GROUPS;
        const uint g = gi % GROUPS;
        sum += F::dot(row_ptr + ulong(b) * F::BYTES, g, xg, lut);
      }
    }
    sum += simd_shuffle_xor(sum, 8);
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    const float uv_raw = simd_shuffle_xor(sum, 16);
    if (lane == 0 && r < half_rows) {
      const float gv = sum * scale;
      const float uv = uv_raw * scale;
      // Clamped: fast-math tanh and exp overflow far out, where both saturate.
      const float gate = p.beta * tanh(clamp(gv / p.beta, -15.0f, 15.0f)) / (1.0f + exp(clamp(-gv, -80.0f, 80.0f)));
      out[ulong(route) * half_rows + r] =
          gate * p.linear_beta * tanh(clamp(uv / p.linear_beta, -15.0f, 15.0f));
    }
    return;
  }

  const uint sub = lane & 7u;
  const uint row_in_sg = lane >> 3;
  const uint row = (group.x * RAW_MV1_DOWN_SG + simd) * 4u + row_in_sg;
  float sum = 0.0f;
  if (row < p.out_dim) {
    device const uchar *row_ptr = bank + ulong(row) * p.row_bytes;
    for (uint gi = sub; gi < groups; gi += 8u) {
      device const float4 *x4 = (device const float4 *)(x + 16u * gi);
      const float4 xg[4] = {x4[0], x4[1], x4[2], x4[3]};
      const uint b = gi / GROUPS;
      const uint g = gi % GROUPS;
      sum += F::dot(row_ptr + ulong(b) * F::BYTES, g, xg, lut);
    }
  }
  sum += simd_shuffle_xor(sum, 4);
  sum += simd_shuffle_xor(sum, 2);
  sum += simd_shuffle_xor(sum, 1);
  if (sub == 0 && row < p.out_dim) {
    out[ulong(route) * p.out_dim + row] = sum * scale;
  }
}

typedef decltype(dinf_raw_mv<raw_nvfp4, false>) dinf_raw_mv_t;
template [[host_name("dinf_raw_mv_nvfp4")]] kernel dinf_raw_mv_t dinf_raw_mv<raw_nvfp4, false>;
template [[host_name("dinf_raw_mv_mxfp8")]] kernel dinf_raw_mv_t dinf_raw_mv<raw_mxfp8, false>;
template [[host_name("dinf_raw_mv_iq4_nl")]] kernel dinf_raw_mv_t dinf_raw_mv<raw_iq4_nl, false>;
template [[host_name("dinf_raw_mv_situ_nvfp4")]] kernel dinf_raw_mv_t dinf_raw_mv<raw_nvfp4, true>;
template [[host_name("dinf_raw_mv_situ_mxfp8")]] kernel dinf_raw_mv_t dinf_raw_mv<raw_mxfp8, true>;
template [[host_name("dinf_raw_mv_situ_iq4_nl")]] kernel dinf_raw_mv_t dinf_raw_mv<raw_iq4_nl, true>;

struct RawMvDownPostConvParams {
  uint in_dim; uint out_dim; uint top_k; uint experts; uint has_scales;
  uint width; uint has_past; float conv_scale;
  ulong row_bytes; ulong expert_bytes;
};

// Single-token fused MoE down + weighted route sum + short post_conv + residual:
// 8 simdgroups (256 threads) per 4 output channels, simdgroup k handles route k
// (8 lanes/channel, zero warp tail on in_dim=1152), reducing across routes in
// threadgroup memory and applying the static post_conv + residual in one launch.
template <typename F>
kernel void dinf_raw_mv_down_post_conv(
    device const uchar *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device const uint *ids [[buffer(2)]],
    device const float *scales [[buffer(3)]],
    device const float *route_weights [[buffer(4)]],
    device const float *past [[buffer(5)]],
    device const float *conv_weight [[buffer(6)]],
    device const float *residual [[buffer(7)]],
    device float *out [[buffer(8)]],
    device float *state [[buffer(9)]],
    constant RawMvDownPostConvParams &p [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float2 lut[F::TABLE ? 256 : 1];
  threadgroup float tg_sum[8][4];
  if (F::TABLE) {
    for (uint i = tid; i < 256; i += 256) {
      lut[i] = F::entry(i);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  constexpr uint GROUPS = F::BLOCK / 16;
  const uint sub = lane & 7u;
  const uint row_in_sg = lane >> 3;
  const uint row = group * 4u + row_in_sg;
  float sum = 0.0f;
  if (simd < p.top_k && row < p.out_dim) {
    const uint expert = min(ids[simd], p.experts - 1u);
    device const float *x = input + ulong(simd) * p.in_dim;
    device const uchar *row_ptr = weights + ulong(expert) * p.expert_bytes + ulong(row) * p.row_bytes;
    const uint groups = p.in_dim / 16u;
    for (uint gi = sub; gi < groups; gi += 8u) {
      device const float4 *x4 = (device const float4 *)(x + 16u * gi);
      const float4 xg[4] = {x4[0], x4[1], x4[2], x4[3]};
      const uint b = gi / GROUPS;
      const uint g = gi % GROUPS;
      sum += F::dot(row_ptr + ulong(b) * F::BYTES, g, xg, lut);
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (sub == 0) {
      const float es = p.has_scales != 0 ? scales[expert] : 1.0f;
      tg_sum[simd][row_in_sg] = sum * es * route_weights[simd];
    }
  } else if (sub == 0) {
    tg_sum[simd][row_in_sg] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0 && lane < 4u) {
    const uint c = group * 4u + lane;
    if (c < p.out_dim) {
      float current = 0.0f;
      for (uint k = 0; k < p.top_k; ++k) {
        current += tg_sum[k][lane];
      }
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
      out[c] = acc * p.conv_scale + residual[c];
    }
  }
}

typedef decltype(dinf_raw_mv_down_post_conv<raw_nvfp4>) dinf_raw_mv_down_pc_t;
template [[host_name("dinf_raw_mv_down_pc_nvfp4")]] kernel dinf_raw_mv_down_pc_t dinf_raw_mv_down_post_conv<raw_nvfp4>;
template [[host_name("dinf_raw_mv_down_pc_mxfp8")]] kernel dinf_raw_mv_down_pc_t dinf_raw_mv_down_post_conv<raw_mxfp8>;
template [[host_name("dinf_raw_mv_down_pc_iq4_nl")]] kernel dinf_raw_mv_down_pc_t dinf_raw_mv_down_post_conv<raw_iq4_nl>;

constant constexpr uint RAW_GROUP_MAX_ROUTES = 8;

// Group up to 128 routes by expert ID into active slots of <= 8 routes each,
// so few-token verification passes read each active expert's weights once.
kernel void dinf_raw_group_routes(
    device const uint *ids [[buffer(0)]],
    device uint *active_count [[buffer(1)]],
    device uint *slot_experts [[buffer(2)]],
    device uint *slot_counts [[buffer(3)]],
    device uint *slot_routes [[buffer(4)]],
    constant uint4 &shape [[buffer(5)]], // (routes, experts, tiles0, tiles1)
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  threadgroup uint s_ids[128];
  threadgroup atomic_uint counter;
  const uint routes = shape.x;
  const uint experts = shape.y;
  if (tid == 0) {
    atomic_store_explicit(&counter, 0u, memory_order_relaxed);
  }
  for (uint r = tid; r < routes; r += threads) {
    s_ids[r] = min(ids[r], experts - 1);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < experts; e += threads) {
    uint chunk[RAW_GROUP_MAX_ROUTES];
    uint count = 0;
    for (uint r = 0; r < routes; ++r) {
      if (s_ids[r] == e) {
        chunk[count++] = r;
        if (count == RAW_GROUP_MAX_ROUTES) {
          const uint slot = atomic_fetch_add_explicit(&counter, 1u, memory_order_relaxed);
          slot_experts[slot] = e;
          slot_counts[slot] = count;
          for (uint j = 0; j < count; ++j) {
            slot_routes[slot * RAW_GROUP_MAX_ROUTES + j] = chunk[j];
          }
          count = 0;
        }
      }
    }
    if (count > 0) {
      const uint slot = atomic_fetch_add_explicit(&counter, 1u, memory_order_relaxed);
      slot_experts[slot] = e;
      slot_counts[slot] = count;
      for (uint j = 0; j < count; ++j) {
        slot_routes[slot * RAW_GROUP_MAX_ROUTES + j] = chunk[j];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    const uint n_active = atomic_load_explicit(&counter, memory_order_relaxed);
    active_count[0] = n_active;
    const uint h = max(n_active, 1u);
    active_count[1] = shape.z;
    active_count[2] = h;
    active_count[3] = 1u;
    active_count[4] = shape.w;
    active_count[5] = h;
    active_count[6] = 1u;
  }
}

// Compile-time route count K keeps accumulators and pointers strictly in
// registers (avoiding AGX device-stack spills from dynamic array indexing).
template <typename F, bool SITU, uint K>
inline void raw_mv_grouped_run(
    device const uchar *bank,
    device const float *input,
    device const uint *slot_routes,
    uint slot,
    uint n,
    const thread uint *rows,
    float scale,
    device float *out,
    constant RawMvParams &p,
    ushort lane,
    threadgroup const float2 *lut) {
  constexpr uint GROUPS = F::BLOCK / 16;
  const uint half_rows = p.out_dim / 2;
  uint r_ids[K];
  uint x_rows[K];
  #pragma unroll
  for (uint k = 0; k < K; ++k) {
    const uint route = slot_routes[slot * RAW_GROUP_MAX_ROUTES + (k < n ? k : 0)];
    r_ids[k] = route;
    x_rows[k] = p.input_rows == 1 ? (route / p.top_k) : route;
  }
  float sums[RAW_MV_ROWS][K];
  #pragma unroll
  for (uint r = 0; r < RAW_MV_ROWS; ++r) {
    #pragma unroll
    for (uint k = 0; k < K; ++k) {
      sums[r][k] = 0.0f;
    }
  }
  const uint groups = p.in_dim / 16;
  if (K <= 2) {
    for (uint gi = lane; gi < groups; gi += 32) {
      const uint b = gi / GROUPS;
      const uint g = gi % GROUPS;
      float4 xg[K][4];
      #pragma unroll
      for (uint k = 0; k < K; ++k) {
        device const float4 *x4 =
            (device const float4 *)(input + ulong(x_rows[k]) * p.in_dim + 16 * gi);
        xg[k][0] = x4[0];
        xg[k][1] = x4[1];
        xg[k][2] = x4[2];
        xg[k][3] = x4[3];
      }
      #pragma unroll
      for (uint r = 0; r < RAW_MV_ROWS; ++r) {
        const uint row = rows[r];
        if (row < p.out_dim) {
          float4 w[4];
          float gs;
          F::load(bank + ulong(row) * p.row_bytes + ulong(b) * F::BYTES, g, w, gs, lut);
          #pragma unroll
          for (uint k = 0; k < K; ++k) {
            const float4 acc =
                (xg[k][0] * w[0] + xg[k][1] * w[1]) + (xg[k][2] * w[2] + xg[k][3] * w[3]);
            sums[r][k] += raw_sum4(acc) * gs;
          }
        }
      }
    }
  } else {
    for (uint gi = lane; gi < groups; gi += 32) {
      const uint b = gi / GROUPS;
      const uint g = gi % GROUPS;
      #pragma unroll
      for (uint r = 0; r < RAW_MV_ROWS; ++r) {
        const uint row = rows[r];
        if (row < p.out_dim) {
          float4 w[4];
          float gs;
          F::load(bank + ulong(row) * p.row_bytes + ulong(b) * F::BYTES, g, w, gs, lut);
          #pragma unroll
          for (uint k = 0; k < K; ++k) {
            if (k < n) {
              device const float4 *x4 =
                  (device const float4 *)(input + ulong(x_rows[k]) * p.in_dim + 16 * gi);
              const float4 acc = (x4[0] * w[0] + x4[1] * w[1]) + (x4[2] * w[2] + x4[3] * w[3]);
              sums[r][k] += raw_sum4(acc) * gs;
            }
          }
        }
      }
    }
  }
  if (SITU) {
    #pragma unroll
    for (uint k = 0; k < K; ++k) {
      if (k < n) {
        const float gv = simd_sum(sums[0][k]) * scale;
        const float uv = simd_sum(sums[1][k]) * scale;
        if (lane == 0 && rows[0] < p.out_dim) {
          const float gate = p.beta * tanh(clamp(gv / p.beta, -15.0f, 15.0f)) / (1.0f + exp(clamp(-gv, -80.0f, 80.0f)));
          out[ulong(r_ids[k]) * half_rows + rows[0]] =
              gate * p.linear_beta * tanh(clamp(uv / p.linear_beta, -15.0f, 15.0f));
        }
      }
    }
    return;
  }
  #pragma unroll
  for (uint r = 0; r < RAW_MV_ROWS; ++r) {
    const uint row = rows[r];
    #pragma unroll
    for (uint k = 0; k < K; ++k) {
      if (k < n) {
        const float total = simd_sum(sums[r][k]);
        if (lane == 0 && row < p.out_dim) {
          out[ulong(r_ids[k]) * p.out_dim + row] = total * scale;
        }
      }
    }
  }
}

// Grouped vector kernel: each active slot owns one expert and up to 8 routes
// sharing it, dequantizing each 16-weight group once across all routes in the
// slot with the exact arithmetic of dinf_raw_mv.
template <typename F, bool SITU>
kernel void dinf_raw_mv_grouped(
    device const uchar *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device const uint *active_count [[buffer(2)]],
    device const uint *slot_experts [[buffer(3)]],
    device const uint *slot_counts [[buffer(4)]],
    device const uint *slot_routes [[buffer(5)]],
    device const float *scales [[buffer(6)]],
    device float *out [[buffer(7)]],
    constant RawMvParams &p [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  const uint slot = group.y;
  if (slot >= active_count[0]) {
    return;
  }
  threadgroup float2 lut[F::TABLE ? 256 : 1];
  if (F::TABLE) {
    for (uint i = tid; i < 256; i += GROUPED_MV_SIMDGROUPS * 32) {
      lut[i] = F::entry(i);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint expert = slot_experts[slot];
  const uint n = slot_counts[slot];
  static_assert(!SITU || RAW_MV_ROWS == 2, "SiTU pairs one gate and one up row");
  const uint half_rows = p.out_dim / 2;
  uint rows[RAW_MV_ROWS];
  if (SITU) {
    const uint r = group.x * GROUPED_MV_SIMDGROUPS + simd;
    rows[0] = r < half_rows ? r : p.out_dim;
    rows[1] = r < half_rows ? half_rows + r : p.out_dim;
  } else {
    for (uint r = 0; r < RAW_MV_ROWS; ++r) {
      rows[r] = (group.x * GROUPED_MV_SIMDGROUPS + simd) * RAW_MV_ROWS + r;
    }
  }
  device const uchar *bank = weights + ulong(expert) * p.expert_bytes;
  const float scale = p.has_scales != 0 ? scales[expert] : 1.0f;
  switch (n) {
    case 1: raw_mv_grouped_run<F, SITU, 1>(bank, input, slot_routes, slot, 1, rows, scale, out, p, lane, lut); break;
    case 2: raw_mv_grouped_run<F, SITU, 2>(bank, input, slot_routes, slot, 2, rows, scale, out, p, lane, lut); break;
    case 3: raw_mv_grouped_run<F, SITU, 3>(bank, input, slot_routes, slot, 3, rows, scale, out, p, lane, lut); break;
    case 4: raw_mv_grouped_run<F, SITU, 4>(bank, input, slot_routes, slot, 4, rows, scale, out, p, lane, lut); break;
    case 5: raw_mv_grouped_run<F, SITU, 5>(bank, input, slot_routes, slot, 5, rows, scale, out, p, lane, lut); break;
    case 6: raw_mv_grouped_run<F, SITU, 6>(bank, input, slot_routes, slot, 6, rows, scale, out, p, lane, lut); break;
    case 7: raw_mv_grouped_run<F, SITU, 7>(bank, input, slot_routes, slot, 7, rows, scale, out, p, lane, lut); break;
    default: raw_mv_grouped_run<F, SITU, 8>(bank, input, slot_routes, slot, n, rows, scale, out, p, lane, lut); break;
  }
}

typedef decltype(dinf_raw_mv_grouped<raw_nvfp4, false>) dinf_raw_mv_grouped_t;
template [[host_name("dinf_raw_mv_grouped_nvfp4")]] kernel dinf_raw_mv_grouped_t dinf_raw_mv_grouped<raw_nvfp4, false>;
template [[host_name("dinf_raw_mv_grouped_mxfp8")]] kernel dinf_raw_mv_grouped_t dinf_raw_mv_grouped<raw_mxfp8, false>;
template [[host_name("dinf_raw_mv_grouped_iq4_nl")]] kernel dinf_raw_mv_grouped_t dinf_raw_mv_grouped<raw_iq4_nl, false>;
template [[host_name("dinf_raw_mv_grouped_situ_nvfp4")]] kernel dinf_raw_mv_grouped_t dinf_raw_mv_grouped<raw_nvfp4, true>;
template [[host_name("dinf_raw_mv_grouped_situ_mxfp8")]] kernel dinf_raw_mv_grouped_t dinf_raw_mv_grouped<raw_mxfp8, true>;
template [[host_name("dinf_raw_mv_grouped_situ_iq4_nl")]] kernel dinf_raw_mv_grouped_t dinf_raw_mv_grouped<raw_iq4_nl, true>;

// Prompt experts with a few routes each (raw_prompt_moe_metal.rs): the
// grouped vector arithmetic over one queued tile of sorted routes. A tile is
// an expert and `valid` consecutive sorted rows from `base`. The `[gate; up]`
// kernel (SITU) reads each route's token row of `input` through
// `sorted_routes`; the other reads the route's own sorted row. Both write
// sorted rows of `out`.
struct SortedTile { uint expert; uint base; uint valid; uint pad; };
struct SortedTinyParams {
  uint in_dim; uint out_dim; uint top_k; uint has_scales;
  ulong row_bytes; ulong expert_bytes; float beta; float linear_beta;
};

// Up to this many routes keep their activation groups in registers while a
// group's rows are multiplied; more would spill.
constant constexpr uint SORTED_TINY_REGISTER_ROUTES = 4;

template <typename F, bool SITU, uint K, typename OutT>
inline void sorted_tiny_run(
    device const uchar *bank,
    device const float *input,
    device const uint *sorted_routes,
    uint base,
    uint n,
    const thread uint *rows,
    float scale,
    device OutT *out,
    constant SortedTinyParams &p,
    ushort lane,
    threadgroup const float2 *lut) {
  constexpr uint GROUPS = F::BLOCK / 16;
  const uint half_rows = p.out_dim / 2;
  uint x_rows[K];
  #pragma unroll
  for (uint k = 0; k < K; ++k) {
    const uint row = base + (k < n ? k : 0);
    x_rows[k] = SITU ? sorted_routes[row] / p.top_k : row;
  }
  float sums[RAW_MV_ROWS][K];
  #pragma unroll
  for (uint r = 0; r < RAW_MV_ROWS; ++r) {
    #pragma unroll
    for (uint k = 0; k < K; ++k) {
      sums[r][k] = 0.0f;
    }
  }
  const uint groups = p.in_dim / 16;
  if (K <= SORTED_TINY_REGISTER_ROUTES) {
    for (uint gi = lane; gi < groups; gi += 32) {
      const uint b = gi / GROUPS;
      const uint g = gi % GROUPS;
      float4 xg[K][4];
      #pragma unroll
      for (uint k = 0; k < K; ++k) {
        device const float4 *x4 =
            (device const float4 *)(input + ulong(x_rows[k]) * p.in_dim + 16 * gi);
        xg[k][0] = x4[0];
        xg[k][1] = x4[1];
        xg[k][2] = x4[2];
        xg[k][3] = x4[3];
      }
      #pragma unroll
      for (uint r = 0; r < RAW_MV_ROWS; ++r) {
        const uint row = rows[r];
        if (row < p.out_dim) {
          float4 w[4];
          float gs;
          F::load(bank + ulong(row) * p.row_bytes + ulong(b) * F::BYTES, g, w, gs, lut);
          #pragma unroll
          for (uint k = 0; k < K; ++k) {
            const float4 acc =
                (xg[k][0] * w[0] + xg[k][1] * w[1]) + (xg[k][2] * w[2] + xg[k][3] * w[3]);
            sums[r][k] += raw_sum4(acc) * gs;
          }
        }
      }
    }
  } else {
    for (uint gi = lane; gi < groups; gi += 32) {
      const uint b = gi / GROUPS;
      const uint g = gi % GROUPS;
      #pragma unroll
      for (uint r = 0; r < RAW_MV_ROWS; ++r) {
        const uint row = rows[r];
        if (row < p.out_dim) {
          float4 w[4];
          float gs;
          F::load(bank + ulong(row) * p.row_bytes + ulong(b) * F::BYTES, g, w, gs, lut);
          #pragma unroll
          for (uint k = 0; k < K; ++k) {
            if (k < n) {
              device const float4 *x4 =
                  (device const float4 *)(input + ulong(x_rows[k]) * p.in_dim + 16 * gi);
              const float4 acc = (x4[0] * w[0] + x4[1] * w[1]) + (x4[2] * w[2] + x4[3] * w[3]);
              sums[r][k] += raw_sum4(acc) * gs;
            }
          }
        }
      }
    }
  }
  if (SITU) {
    #pragma unroll
    for (uint k = 0; k < K; ++k) {
      if (k < n) {
        const float gv = simd_sum(sums[0][k]) * scale;
        const float uv = simd_sum(sums[1][k]) * scale;
        if (lane == 0 && rows[0] < p.out_dim) {
          const float gate = p.beta * tanh(clamp(gv / p.beta, -15.0f, 15.0f)) / (1.0f + exp(clamp(-gv, -80.0f, 80.0f)));
          out[ulong(base + k) * half_rows + rows[0]] =
              OutT(gate * p.linear_beta * tanh(clamp(uv / p.linear_beta, -15.0f, 15.0f)));
        }
      }
    }
    return;
  }
  #pragma unroll
  for (uint r = 0; r < RAW_MV_ROWS; ++r) {
    const uint row = rows[r];
    #pragma unroll
    for (uint k = 0; k < K; ++k) {
      if (k < n) {
        const float total = simd_sum(sums[r][k]);
        if (lane == 0 && row < p.out_dim) {
          out[ulong(base + k) * p.out_dim + row] = OutT(total * scale);
        }
      }
    }
  }
}

// Threadgroups are (tiles, row tiles); `tile_counts[4]` is the queue length.
template <typename F, bool SITU, typename OutT = conditional_t<SITU, float, bfloat>>
kernel void dinf_sorted_tiny(
    device const uchar *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device const uint *sorted_routes [[buffer(2)]],
    device const uint *tile_counts [[buffer(3)]],
    device const SortedTile *tiles [[buffer(4)]],
    device const float *scales [[buffer(5)]],
    device OutT *out [[buffer(6)]],
    constant SortedTinyParams &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  if (group.x >= tile_counts[4]) {
    return;
  }
  threadgroup float2 lut[F::TABLE ? 256 : 1];
  if (F::TABLE) {
    for (uint i = tid; i < 256; i += RAW_MV_SIMDGROUPS * 32) {
      lut[i] = F::entry(i);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const SortedTile tile = tiles[group.x];
  static_assert(!SITU || RAW_MV_ROWS == 2, "SiTU pairs one gate and one up row");
  const uint half_rows = p.out_dim / 2;
  uint rows[RAW_MV_ROWS];
  if (SITU) {
    const uint r = group.y * RAW_MV_SIMDGROUPS + simd;
    rows[0] = r < half_rows ? r : p.out_dim;
    rows[1] = r < half_rows ? half_rows + r : p.out_dim;
  } else {
    for (uint r = 0; r < RAW_MV_ROWS; ++r) {
      rows[r] = (group.y * RAW_MV_SIMDGROUPS + simd) * RAW_MV_ROWS + r;
    }
  }
  device const uchar *bank = weights + ulong(tile.expert) * p.expert_bytes;
  const float scale = p.has_scales != 0 ? scales[tile.expert] : 1.0f;
  const uint n = tile.valid;
  switch (n) {
    case 1: sorted_tiny_run<F, SITU, 1>(bank, input, sorted_routes, tile.base, 1, rows, scale, out, p, lane, lut); break;
    case 2: sorted_tiny_run<F, SITU, 2>(bank, input, sorted_routes, tile.base, 2, rows, scale, out, p, lane, lut); break;
    case 3: sorted_tiny_run<F, SITU, 3>(bank, input, sorted_routes, tile.base, 3, rows, scale, out, p, lane, lut); break;
    case 4: sorted_tiny_run<F, SITU, 4>(bank, input, sorted_routes, tile.base, 4, rows, scale, out, p, lane, lut); break;
    default: sorted_tiny_run<F, SITU, 8>(bank, input, sorted_routes, tile.base, n, rows, scale, out, p, lane, lut); break;
  }
}

typedef decltype(dinf_sorted_tiny<raw_nvfp4, false>) dinf_sorted_tiny_down_t;
typedef decltype(dinf_sorted_tiny<raw_nvfp4, true>) dinf_sorted_tiny_situ_t;
template [[host_name("dinf_sorted_tiny_nvfp4")]] kernel dinf_sorted_tiny_down_t dinf_sorted_tiny<raw_nvfp4, false>;
template [[host_name("dinf_sorted_tiny_mxfp8")]] kernel dinf_sorted_tiny_down_t dinf_sorted_tiny<raw_mxfp8, false>;
template [[host_name("dinf_sorted_tiny_iq4_nl")]] kernel dinf_sorted_tiny_down_t dinf_sorted_tiny<raw_iq4_nl, false>;
template [[host_name("dinf_sorted_tiny_situ_nvfp4")]] kernel dinf_sorted_tiny_situ_t dinf_sorted_tiny<raw_nvfp4, true>;
template [[host_name("dinf_sorted_tiny_situ_mxfp8")]] kernel dinf_sorted_tiny_situ_t dinf_sorted_tiny<raw_mxfp8, true>;
template [[host_name("dinf_sorted_tiny_situ_iq4_nl")]] kernel dinf_sorted_tiny_situ_t dinf_sorted_tiny<raw_iq4_nl, true>;

// MXFP8 blocks from the GGUF's E4M3 values and UE8M0 scales, counting
// NaN codes into `bad`.
kernel void dinf_raw_mxfp8_blocks(
    device const uchar *values [[buffer(0)]],
    device const uchar *scales [[buffer(1)]],
    device uchar *blocks [[buffer(2)]],
    device atomic_uint *bad [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint block [[thread_position_in_grid]]) {
  if (block >= count) {
    return;
  }
  const uchar scale = scales[block];
  device uchar *out = blocks + ulong(block) * 36;
  out[0] = scale;
  out[1] = 0;
  out[2] = 0;
  out[3] = 0;
  bool nan = scale == 0xff;
  device const uchar *v = values + ulong(block) * 32;
  for (int i = 0; i < 32; ++i) {
    const uchar b = v[i];
    nan = nan || (b & 0x7f) == 0x7f;
    out[4 + i] = b;
  }
  if (nan) {
    atomic_fetch_add_explicit(bad, 1u, memory_order_relaxed);
  }
}

// F16 values of BF16 ones, rounded to nearest as Candle's casts round.
kernel void dinf_raw_bf16_to_f16(
    device const ushort *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) {
    output[index] = half(as_type<float>(uint(input[index]) << 16));
  }
}

// out[route][...] *= scales[ids[route]]: the matmul path's expert scales.
kernel void dinf_raw_scale_routes(
    device float *out [[buffer(0)]],
    device const uint *ids [[buffer(1)]],
    device const float *scales [[buffer(2)]],
    constant uint2 &shape [[buffer(3)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x < shape.x) {
    out[ulong(position.y) * shape.x + position.x] *= scales[min(ids[position.y], shape.y - 1)];
  }
}
"#;

/// Routes below this many tokens run the vector kernel (1..=8, covering
/// single-token decode and up to 8-row speculative verification).
const VECTOR_MAX_TOKENS: usize = 9;
/// Max routes supported by the grouped expert-deduplication vector kernel.
const GROUPED_MAX_ROUTES: usize = 128;
/// Routes packed into one active-expert slot (matches `RAW_GROUP_MAX_ROUTES`).
const GROUPED_SLOT_ROUTES: usize = 8;

type LibraryKey = DeviceId;
static LIBRARIES: OnceLock<Mutex<HashMap<LibraryKey, Library>>> = OnceLock::new();
static PIPELINES: OnceLock<Mutex<HashMap<(DeviceId, &'static str), ComputePipeline>>> =
    OnceLock::new();

fn library(device: &candle_core::MetalDevice) -> Result<Library> {
    let libraries = LIBRARIES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(library) = libraries.lock().unwrap().get(&device.id()) {
        return Ok(library.clone());
    }
    let source = format!(
        "{}\n{}\n{}",
        candle_metal_kernels::source::QUANTIZED,
        super::indexed_moe_metal::COMPACT_INDEXED_MOE_MSL,
        RAW_EXPERTS_MSL
    );
    let library = device
        .metal_device()
        .new_library_with_source(&source, None)
        .map_err(candle_core::Error::wrap)?;
    libraries
        .lock()
        .unwrap()
        .insert(device.id(), library.clone());
    Ok(library)
}

pub(super) fn pipeline(
    device: &candle_core::MetalDevice,
    name: &'static str,
) -> Result<ComputePipeline> {
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (device.id(), name);
    if let Some(pipeline) = pipelines.lock().unwrap().get(&key) {
        return Ok(pipeline.clone());
    }
    let function = library(device)?
        .get_function(name, None)
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    pipelines.lock().unwrap().insert(key, pipeline.clone());
    Ok(pipeline)
}

/// Metal 4 expert kernels: the half-input tensor-op kernel the GGML types use
/// (`METAL4_INDEXED_Q4_MSL`), built for each raw format, with or without
/// per-expert output scales.
type Metal4Key = (DeviceId, RawFormat, bool);
static METAL4_PIPELINES: OnceLock<Mutex<HashMap<Metal4Key, ComputePipeline>>> = OnceLock::new();

/// Weight rows per Metal 4 tile, as the GGML half-input kernels use.
const METAL4_TILE_ROWS: usize = 128;
/// Routes per Metal 4 tile.
const METAL4_ROUTE_TILE: usize = 32;

/// The half-input Metal 4 expert kernel's source for `format`, with its
/// block type and dequantizer.
pub(super) fn metal4_source(format: RawFormat, scales: bool) -> String {
    let define = match format {
        RawFormat::Nvfp4 => "DINF_NVFP4",
        RawFormat::Mxfp8 => "DINF_MXFP8",
        RawFormat::Iq4Nl => "DINF_IQ4_NL",
    };
    format!(
        "#define DINF_TILE_ROWS_128 1\n#define DINF_PRECOMPUTE_ROUTE_OFFSETS 1\n\
         #define DINF_HALF_INPUT 1\n#define {define} 1\n{}{}",
        if scales {
            "#define DINF_EXPERT_SCALES 1\n"
        } else {
            ""
        },
        super::indexed_moe_metal::METAL4_INDEXED_Q4_MSL
    )
}

fn metal4_pipeline(
    device: &candle_core::MetalDevice,
    format: RawFormat,
    scales: bool,
) -> Result<ComputePipeline> {
    use objc2_metal::{MTLCompileOptions, MTLLanguageVersion};
    let pipelines = METAL4_PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (device.id(), format, scales);
    if let Some(pipeline) = pipelines.lock().unwrap().get(&key) {
        return Ok(pipeline.clone());
    }
    let name = match format {
        RawFormat::Nvfp4 => "dinf_metal4_indexed_nvfp4",
        RawFormat::Mxfp8 => "dinf_metal4_indexed_mxfp8",
        RawFormat::Iq4Nl => "dinf_metal4_indexed_iq4_nl",
    };
    let source = metal4_source(format, scales);
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version4_0);
    let library = device
        .metal_device()
        .new_library_with_source(&source, Some(&options))
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

pub(super) fn uses_metal4(device: &candle_core::MetalDevice) -> bool {
    std::env::var_os("DINF_RAW_MOE_COMPACT").is_none()
        && super::indexed_moe_metal::supports_metal4_indexed_moe(device)
}

/// Kernel names and block geometry, `(weights, bytes)` per block.
fn kernels(format: RawFormat) -> (&'static str, &'static str, usize, usize) {
    match format {
        RawFormat::Nvfp4 => ("dinf_raw_mv_nvfp4", "dinf_raw_mm_id_nvfp4", 64, 36),
        RawFormat::Mxfp8 => ("dinf_raw_mv_mxfp8", "dinf_raw_mm_id_mxfp8", 32, 36),
        RawFormat::Iq4Nl => ("dinf_raw_mv_iq4_nl", "dinf_raw_mm_id_iq4_nl", 32, 18),
    }
}

fn situ_kernel(format: RawFormat) -> &'static str {
    match format {
        RawFormat::Nvfp4 => "dinf_raw_mv_situ_nvfp4",
        RawFormat::Mxfp8 => "dinf_raw_mv_situ_mxfp8",
        RawFormat::Iq4Nl => "dinf_raw_mv_situ_iq4_nl",
    }
}

fn grouped_kernel(format: RawFormat, situ: bool) -> &'static str {
    match (format, situ) {
        (RawFormat::Nvfp4, false) => "dinf_raw_mv_grouped_nvfp4",
        (RawFormat::Mxfp8, false) => "dinf_raw_mv_grouped_mxfp8",
        (RawFormat::Iq4Nl, false) => "dinf_raw_mv_grouped_iq4_nl",
        (RawFormat::Nvfp4, true) => "dinf_raw_mv_grouped_situ_nvfp4",
        (RawFormat::Mxfp8, true) => "dinf_raw_mv_grouped_situ_mxfp8",
        (RawFormat::Iq4Nl, true) => "dinf_raw_mv_grouped_situ_iq4_nl",
    }
}

/// The prompt kernel for a queued tile of a few sorted routes of one expert.
pub(super) fn sorted_tiny_kernel(format: RawFormat, situ: bool) -> &'static str {
    match (format, situ) {
        (RawFormat::Nvfp4, false) => "dinf_sorted_tiny_nvfp4",
        (RawFormat::Mxfp8, false) => "dinf_sorted_tiny_mxfp8",
        (RawFormat::Iq4Nl, false) => "dinf_sorted_tiny_iq4_nl",
        (RawFormat::Nvfp4, true) => "dinf_sorted_tiny_situ_nvfp4",
        (RawFormat::Mxfp8, true) => "dinf_sorted_tiny_situ_mxfp8",
        (RawFormat::Iq4Nl, true) => "dinf_sorted_tiny_situ_iq4_nl",
    }
}

/// Whether [`forward_situ`] serves `tokens` tokens.
pub(crate) fn situ_serves(tokens: usize) -> bool {
    tokens < VECTOR_MAX_TOKENS
}

/// Compile the kernels a bank of `format`, with or without per-expert
/// `scales`, uses on `device`.
pub(crate) fn precompile(
    format: RawFormat,
    scales: bool,
    device: &candle_core::MetalDevice,
) -> Result<()> {
    let (vector, matrix, _, _) = kernels(format);
    pipeline(device, vector)?;
    pipeline(device, matrix)?;
    pipeline(device, situ_kernel(format))?;
    pipeline(device, "dinf_raw_group_routes")?;
    pipeline(device, grouped_kernel(format, false))?;
    pipeline(device, grouped_kernel(format, true))?;
    pipeline(device, "dinf_raw_scale_routes")?;
    pipeline(device, "dinf_raw_mxfp8_blocks")?;
    if uses_metal4(device) {
        metal4_pipeline(device, format, scales)?;
        super::raw_prompt_moe_metal::precompile(format, scales, device)?;
        pipeline(device, sorted_tiny_kernel(format, false))?;
        pipeline(device, sorted_tiny_kernel(format, true))?;
    }
    Ok(())
}

#[repr(C)]
struct RawMvParams {
    in_dim: u32,
    out_dim: u32,
    top_k: u32,
    input_rows: u32,
    experts: u32,
    has_scales: u32,
    row_bytes: u64,
    expert_bytes: u64,
    beta: f32,
    linear_beta: f32,
}

pub(super) fn metal_view(tensor: &Tensor, dtype: DType, what: &str) -> Result<(Buffer, usize)> {
    if tensor.dtype() != dtype || !tensor.is_contiguous() {
        candle_core::bail!("raw expert {what} must be contiguous {dtype:?}")
    }
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Metal(storage) = &*storage else {
        candle_core::bail!("raw expert {what} must be on Metal")
    };
    Ok((
        storage.buffer().clone(),
        layout.start_offset() * dtype.size_in_bytes(),
    ))
}

/// An exact-size private buffer for a resident bank. Candle's pool rounds
/// sizes up to a power of two, which would cost a quarter or more of each
/// MXFP8 bank. It is not pooled, so it is freed with its last tensor.
pub(super) fn exact_buffer(device: &candle_core::MetalDevice, bytes: usize) -> Result<Arc<Buffer>> {
    use objc2_metal::MTLResourceOptions as Options;
    let options = Options(Options::StorageModePrivate.0 | Options::HazardTrackingModeUntracked.0);
    let buffer = device
        .device()
        .new_buffer(bytes.max(1), options)
        .map_err(candle_core::Error::wrap)?;
    Ok(Arc::new(buffer))
}

pub(super) fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| candle_core::Error::Msg(format!("raw expert {what} too large")))
}

/// Grouped route plan shared between a visit's `gate_up` and `down` raw-expert
/// dispatches, packed into one small buffer:
/// `[active_count (1) | slot_experts (routes) | slot_counts (routes) | slot_routes (routes * 8)]`.
pub(crate) struct RawRoutePlan {
    buf: Arc<Buffer>,
    routes: usize,
    tiles0: usize,
    tiles1: usize,
}

impl RawRoutePlan {
    const HEADER_U32S: usize = 7;
    fn active_count_offset(&self) -> usize {
        0
    }
    fn indirect_offset(&self, tiles: usize) -> Option<usize> {
        if tiles == self.tiles0 {
            Some(4)
        } else if tiles == self.tiles1 {
            Some(16)
        } else {
            None
        }
    }
    fn slot_experts_offset(&self) -> usize {
        Self::HEADER_U32S * 4
    }
    fn slot_counts_offset(&self) -> usize {
        (Self::HEADER_U32S + self.routes) * 4
    }
    fn slot_routes_offset(&self) -> usize {
        (Self::HEADER_U32S + self.routes * 2) * 4
    }
}

/// Group `ids` `[tokens, top_k]` U32 by expert ID on Metal when the grouped
/// vector path applies (`2 <= tokens < VECTOR_MAX_TOKENS`, `routes <= 128`).
pub(crate) fn group_routes(
    ids: &Tensor,
    experts: usize,
    tiles0: usize,
    tiles1: usize,
) -> Result<Option<RawRoutePlan>> {
    let candle_core::Device::Metal(device) = ids.device() else {
        return Ok(None);
    };
    let (tokens, top_k) = ids.dims2()?;
    let routes = tokens * top_k;
    if !(2..VECTOR_MAX_TOKENS).contains(&tokens)
        || routes == 0
        || routes > GROUPED_MAX_ROUTES
        || experts == 0
    {
        return Ok(None);
    }
    let ids = ids.contiguous()?;
    let (ids_buffer, ids_offset) = metal_view(&ids, DType::U32, "ids")?;
    let total_u32s = RawRoutePlan::HEADER_U32S + routes * (2 + GROUPED_SLOT_ROUTES);
    let buf = device
        .new_buffer_builder()
        .with_size_for(total_u32s, DType::U32)
        .with_label("dinf-raw-route-plan")
        .build()?;
    let plan = RawRoutePlan {
        buf,
        routes,
        tiles0,
        tiles1,
    };
    let encoder = device.command_encoder()?;
    encoder.set_label("dinf-raw-group-routes");
    let encoder: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline(device, "dinf_raw_group_routes")?);
    encoder.set_input_buffer(0, Some(&ids_buffer), ids_offset);
    encoder.set_output_buffer(1, Some(&plan.buf), plan.active_count_offset());
    encoder.set_output_buffer(2, Some(&plan.buf), plan.slot_experts_offset());
    encoder.set_output_buffer(3, Some(&plan.buf), plan.slot_counts_offset());
    encoder.set_output_buffer(4, Some(&plan.buf), plan.slot_routes_offset());
    let group_shape = [
        to_u32(routes, "routes")?,
        to_u32(experts, "experts")?,
        to_u32(tiles0, "tiles")?,
        to_u32(tiles1, "tiles")?,
    ];
    encoder.set_bytes(5, &group_shape);
    encoder.dispatch_thread_groups(
        objc2_metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(Some(plan))
}

/// `input` `[tokens, 1 | top_k, in_dim]` F32 through the experts `ids`
/// `[tokens, top_k]` U32 of a `[experts, out_dim, in_dim]` bank of `format`
/// held as `bytes` on Metal, giving `[tokens, top_k, out_dim]` F32.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward(
    format: RawFormat,
    bytes: &Tensor,
    scales: Option<&Tensor>,
    input: &Tensor,
    ids: &Tensor,
    experts: usize,
    out_dim: usize,
    in_dim: usize,
) -> Result<Tensor> {
    forward_with_plan(
        format, bytes, scales, input, ids, None, experts, out_dim, in_dim,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_with_plan(
    format: RawFormat,
    bytes: &Tensor,
    scales: Option<&Tensor>,
    input: &Tensor,
    ids: &Tensor,
    plan: Option<&RawRoutePlan>,
    experts: usize,
    out_dim: usize,
    in_dim: usize,
) -> Result<Tensor> {
    run(
        format,
        bytes,
        scales,
        input,
        ids,
        plan,
        [experts, out_dim, in_dim],
        None,
    )
}

/// [`forward`] of a `[gate; up]` bank for up to eight tokens, with SiTU
/// (`beta`, `linear_beta`) applied in the same kernel: `[tokens, top_k,
/// out_dim / 2]`.
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn forward_situ(
    format: RawFormat,
    bytes: &Tensor,
    scales: Option<&Tensor>,
    input: &Tensor,
    ids: &Tensor,
    [experts, out_dim, in_dim]: [usize; 3],
    beta: f32,
    linear_beta: f32,
) -> Result<Tensor> {
    forward_situ_with_plan(
        format,
        bytes,
        scales,
        input,
        ids,
        None,
        [experts, out_dim, in_dim],
        beta,
        linear_beta,
    )
}

#[repr(C)]
struct RawMvDownPostConvParams {
    in_dim: u32,
    out_dim: u32,
    top_k: u32,
    experts: u32,
    has_scales: u32,
    width: u32,
    has_past: u32,
    conv_scale: f32,
    row_bytes: u64,
    expert_bytes: u64,
}

fn down_pc_kernel(format: RawFormat) -> &'static str {
    match format {
        RawFormat::Nvfp4 => "dinf_raw_mv_down_pc_nvfp4",
        RawFormat::Mxfp8 => "dinf_raw_mv_down_pc_mxfp8",
        RawFormat::Iq4Nl => "dinf_raw_mv_down_pc_iq4_nl",
    }
}

/// Single-token fused `down` projection + route-weighted sum + static `post_conv`
/// + `residual`, returning `(out [1, out_dim], state [width - 1, out_dim])`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_down_post_conv_1(
    format: RawFormat,
    bytes: &Tensor,
    scales: Option<&Tensor>,
    input: &Tensor,
    ids: &Tensor,
    route_weights: &Tensor,
    past: Option<&Tensor>,
    conv_weight: &Tensor,
    residual: &Tensor,
    [experts, out_dim, in_dim]: [usize; 3],
) -> Result<(Tensor, Tensor)> {
    let candle_core::Device::Metal(device) = bytes.device() else {
        candle_core::bail!("raw expert bank is not on Metal")
    };
    let (_, _, block, block_bytes) = kernels(format);
    let (tokens, top_k, width) = input.dims3()?;
    let (conv_channels, conv_width) = conv_weight.dims2()?;
    if tokens != 1
        || top_k == 0
        || top_k > 8
        || width != in_dim
        || conv_channels != out_dim
        || conv_width < 2
        || !in_dim.is_multiple_of(block.max(32))
    {
        candle_core::bail!("raw expert fused down+post_conv geometry differs")
    }
    let row_bytes = in_dim / block * block_bytes;
    let expert_bytes = out_dim * row_bytes;
    let input = input.contiguous()?;
    let ids = ids.contiguous()?;
    let route_weights = route_weights.contiguous()?;
    let past_c = past.map(Tensor::contiguous).transpose()?;
    let conv_weight = conv_weight.contiguous()?;
    let residual = residual.contiguous()?;
    let (w_buf, w_off) = metal_view(bytes, DType::U8, "weights")?;
    let (in_buf, in_off) = metal_view(&input, DType::F32, "input")?;
    let (ids_buf, ids_off) = metal_view(&ids, DType::U32, "ids")?;
    let scales_v = scales
        .map(|s| metal_view(s, DType::F32, "scales"))
        .transpose()?;
    let (rw_buf, rw_off) = metal_view(&route_weights, DType::F32, "route_weights")?;
    let past_v = past_c
        .as_ref()
        .map(|p| metal_view(p, DType::F32, "past"))
        .transpose()?;
    let (cw_buf, cw_off) = metal_view(&conv_weight, DType::F32, "conv_weight")?;
    let (res_buf, res_off) = metal_view(&residual, DType::F32, "residual")?;

    let history = conv_width - 1;
    let out_buf = super::inkling2_metal::alloc_scratch_buffer(
        device,
        out_dim.max(1) * DType::F32.size_in_bytes(),
    )?;
    let state_buf = super::inkling2_metal::alloc_scratch_buffer(
        device,
        (history * out_dim).max(1) * DType::F32.size_in_bytes(),
    )?;
    let params = RawMvDownPostConvParams {
        in_dim: to_u32(in_dim, "input width")?,
        out_dim: to_u32(out_dim, "output width")?,
        top_k: to_u32(top_k, "top_k")?,
        experts: to_u32(experts, "experts")?,
        has_scales: u32::from(scales_v.is_some()),
        width: to_u32(conv_width, "conv width")?,
        has_past: u32::from(past_v.is_some()),
        conv_scale: 1.0 / ((conv_width + 1) as f32).sqrt(),
        row_bytes: row_bytes as u64,
        expert_bytes: expert_bytes as u64,
    };
    let encoder = device.command_encoder()?;
    encoder.set_label("dinf-raw-down-post-conv");
    let encoder: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline(device, down_pc_kernel(format))?);
    encoder.set_input_buffer(0, Some(&w_buf), w_off);
    encoder.set_input_buffer(1, Some(&in_buf), in_off);
    encoder.set_input_buffer(2, Some(&ids_buf), ids_off);
    match &scales_v {
        Some((b, o)) => encoder.set_input_buffer(3, Some(b), *o),
        None => encoder.set_input_buffer(3, Some(&in_buf), in_off),
    }
    encoder.set_input_buffer(4, Some(&rw_buf), rw_off);
    match &past_v {
        Some((b, o)) => encoder.set_input_buffer(5, Some(b), *o),
        None => encoder.set_input_buffer(5, Some(&in_buf), in_off),
    }
    encoder.set_input_buffer(6, Some(&cw_buf), cw_off);
    encoder.set_input_buffer(7, Some(&res_buf), res_off);
    encoder.set_output_buffer(8, Some(&out_buf), 0);
    encoder.set_output_buffer(9, Some(&state_buf), 0);
    encoder.set_bytes(10, &params);
    encoder.dispatch_thread_groups(
        objc2_metal::MTLSize {
            width: out_dim.div_ceil(4),
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    let out_t = Tensor::from((
        Storage::Metal(MetalStorage::new(
            out_buf,
            device.clone(),
            out_dim,
            DType::F32,
        )),
        Shape::from((1, out_dim)),
    ));
    let state_t = Tensor::from((
        Storage::Metal(MetalStorage::new(
            state_buf,
            device.clone(),
            history * out_dim,
            DType::F32,
        )),
        Shape::from((history, out_dim)),
    ));
    Ok((out_t, state_t))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_situ_with_plan(
    format: RawFormat,
    bytes: &Tensor,
    scales: Option<&Tensor>,
    input: &Tensor,
    ids: &Tensor,
    plan: Option<&RawRoutePlan>,
    [experts, out_dim, in_dim]: [usize; 3],
    beta: f32,
    linear_beta: f32,
) -> Result<Tensor> {
    if !situ_serves(input.dim(0)?) || !out_dim.is_multiple_of(2) {
        candle_core::bail!("raw expert SiTU runs on up to eight tokens of an even bank")
    }
    run(
        format,
        bytes,
        scales,
        input,
        ids,
        plan,
        [experts, out_dim, in_dim],
        Some((beta, linear_beta)),
    )
}

#[allow(clippy::too_many_arguments)]
fn run(
    format: RawFormat,
    bytes: &Tensor,
    scales: Option<&Tensor>,
    input: &Tensor,
    ids: &Tensor,
    plan: Option<&RawRoutePlan>,
    [experts, out_dim, in_dim]: [usize; 3],
    situ: Option<(f32, f32)>,
) -> Result<Tensor> {
    let candle_core::Device::Metal(device) = bytes.device() else {
        candle_core::bail!("raw expert bank is not on Metal")
    };
    let (vector, matrix, block, block_bytes) = kernels(format);
    let (tokens, input_rows, width) = input.dims3()?;
    let (id_tokens, top_k) = ids.dims2()?;
    if width != in_dim
        || id_tokens != tokens
        || !(input_rows == 1 || input_rows == top_k)
        || !in_dim.is_multiple_of(block.max(32))
        || experts == 0
        || top_k == 0
    {
        candle_core::bail!(
            "raw expert input {:?} does not fit ids {:?} and [{experts}, {out_dim}, {in_dim}]",
            input.dims(),
            ids.dims()
        )
    }
    let row_bytes = in_dim / block * block_bytes;
    let expert_bytes = out_dim * row_bytes;
    if bytes.elem_count() != experts * expert_bytes {
        candle_core::bail!("raw expert bank holds {} bytes", bytes.elem_count())
    }
    let input = input.contiguous()?;
    let ids = ids.contiguous()?;
    let (weights, weights_offset) = metal_view(bytes, DType::U8, "weights")?;
    let (input_buffer, input_offset) = metal_view(&input, DType::F32, "input")?;
    let (ids_buffer, ids_offset) = metal_view(&ids, DType::U32, "ids")?;
    let scales = scales
        .map(|scales| metal_view(scales, DType::F32, "scales"))
        .transpose()?;
    let routes = tokens * top_k;
    let width = if situ.is_some() { out_dim / 2 } else { out_dim };
    let elements = routes * width;
    let output = super::inkling2_metal::alloc_scratch_buffer(
        device,
        elements.max(1) * DType::F32.size_in_bytes(),
    )?;
    let wrap = |output: Arc<Buffer>| {
        let shape = Shape::from((tokens, top_k, width));
        Tensor::from((
            Storage::Metal(MetalStorage::new(
                output,
                device.clone(),
                elements,
                DType::F32,
            )),
            shape,
        ))
    };
    if routes == 0 {
        return Ok(wrap(output));
    }
    let encoder = device.command_encoder()?;
    encoder.set_label("dinf-raw-experts");
    let encoder: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();

    if tokens < VECTOR_MAX_TOKENS {
        let params = RawMvParams {
            in_dim: to_u32(in_dim, "input width")?,
            out_dim: to_u32(out_dim, "output width")?,
            top_k: to_u32(top_k, "routes")?,
            input_rows: to_u32(input_rows, "input rows")?,
            experts: to_u32(experts, "experts")?,
            has_scales: u32::from(scales.is_some()),
            row_bytes: row_bytes as u64,
            expert_bytes: expert_bytes as u64,
            beta: situ.map_or(1.0, |(beta, _)| beta),
            linear_beta: situ.map_or(1.0, |(_, linear_beta)| linear_beta),
        };
        // SITU: 4 simdgroups (128 threads), 1 gate+up pair per simdgroup (4 pairs/TG).
        // !SITU: 2 simdgroups (64 threads), 4 rows per simdgroup (8 rows/TG).
        let (tiles, mv1_threads) = match situ {
            Some(_) => ((out_dim / 2).div_ceil(4), 128),
            None => (out_dim.div_ceil(8), 64),
        };
        if tokens >= 2 && routes <= GROUPED_MAX_ROUTES {
            // Four simdgroups of two rows (or one gate/up pair) per threadgroup.
            let tiles = match situ {
                Some(_) => (out_dim / 2).div_ceil(4),
                None => out_dim.div_ceil(8),
            };
            let owned_plan;
            let active_plan = match plan {
                Some(p) if p.routes == routes => p,
                _ => {
                    let total_u32s = RawRoutePlan::HEADER_U32S + routes * (2 + GROUPED_SLOT_ROUTES);
                    let buf = device
                        .new_buffer_builder()
                        .with_size_for(total_u32s, DType::U32)
                        .with_label("dinf-raw-route-plan")
                        .build()?;
                    owned_plan = RawRoutePlan {
                        buf,
                        routes,
                        tiles0: tiles,
                        tiles1: tiles,
                    };
                    encoder.set_compute_pipeline_state(&pipeline(device, "dinf_raw_group_routes")?);
                    encoder.set_input_buffer(0, Some(&ids_buffer), ids_offset);
                    encoder.set_output_buffer(
                        1,
                        Some(&owned_plan.buf),
                        owned_plan.active_count_offset(),
                    );
                    encoder.set_output_buffer(
                        2,
                        Some(&owned_plan.buf),
                        owned_plan.slot_experts_offset(),
                    );
                    encoder.set_output_buffer(
                        3,
                        Some(&owned_plan.buf),
                        owned_plan.slot_counts_offset(),
                    );
                    encoder.set_output_buffer(
                        4,
                        Some(&owned_plan.buf),
                        owned_plan.slot_routes_offset(),
                    );
                    let group_shape = [
                        to_u32(routes, "routes")?,
                        to_u32(experts, "experts")?,
                        to_u32(tiles, "tiles")?,
                        to_u32(tiles, "tiles")?,
                    ];
                    encoder.set_bytes(5, &group_shape);
                    encoder.dispatch_thread_groups(
                        objc2_metal::MTLSize {
                            width: 1,
                            height: 1,
                            depth: 1,
                        },
                        objc2_metal::MTLSize {
                            width: 128,
                            height: 1,
                            depth: 1,
                        },
                    );
                    &owned_plan
                }
            };
            encoder.set_compute_pipeline_state(&pipeline(
                device,
                grouped_kernel(format, situ.is_some()),
            )?);
            encoder.set_input_buffer(0, Some(&weights), weights_offset);
            encoder.set_input_buffer(1, Some(&input_buffer), input_offset);
            encoder.set_input_buffer(2, Some(&active_plan.buf), active_plan.active_count_offset());
            encoder.set_input_buffer(3, Some(&active_plan.buf), active_plan.slot_experts_offset());
            encoder.set_input_buffer(4, Some(&active_plan.buf), active_plan.slot_counts_offset());
            encoder.set_input_buffer(5, Some(&active_plan.buf), active_plan.slot_routes_offset());
            match &scales {
                Some((buffer, offset)) => encoder.set_input_buffer(6, Some(buffer), *offset),
                None => encoder.set_input_buffer(6, Some(&input_buffer), input_offset),
            }
            encoder.set_output_buffer(7, Some(&output), 0);
            encoder.set_bytes(8, &params);
            let tg = objc2_metal::MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            };
            if let Some(indirect_offset) = active_plan.indirect_offset(tiles) {
                encoder.dispatch_thread_groups_indirect(&active_plan.buf, indirect_offset, tg);
            } else {
                encoder.dispatch_thread_groups(
                    objc2_metal::MTLSize {
                        width: tiles,
                        height: routes,
                        depth: 1,
                    },
                    tg,
                );
            }
            return Ok(wrap(output));
        }
        let name = if situ.is_some() {
            situ_kernel(format)
        } else {
            vector
        };
        encoder.set_compute_pipeline_state(&pipeline(device, name)?);
        encoder.set_input_buffer(0, Some(&weights), weights_offset);
        encoder.set_input_buffer(1, Some(&input_buffer), input_offset);
        encoder.set_input_buffer(2, Some(&ids_buffer), ids_offset);
        match &scales {
            Some((buffer, offset)) => encoder.set_input_buffer(3, Some(buffer), *offset),
            // Unread without scales; any bound buffer will do.
            None => encoder.set_input_buffer(3, Some(&input_buffer), input_offset),
        }
        encoder.set_output_buffer(4, Some(&output), 0);
        encoder.set_bytes(5, &params);
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: tiles,
                height: routes,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: mv1_threads,
                height: 1,
                depth: 1,
            },
        );
        return Ok(wrap(output));
    }

    if situ.is_some() {
        candle_core::bail!("raw expert SiTU runs only in the vector kernel")
    }
    // Compact route map: each expert's routes, then one matmul tile set per
    // expert over them.
    let counts = device
        .new_buffer_builder()
        .with_size_for(experts, DType::U32)
        .with_label("dinf-raw-experts-counts")
        .build()?;
    let route_map = device
        .new_buffer_builder()
        .with_size_for(experts * tokens, DType::U32)
        .with_label("dinf-raw-experts-route-map")
        .build()?;
    encoder.set_input_buffer(0, Some(&ids_buffer), ids_offset);
    encoder.set_output_buffer(1, Some(&counts), 0);
    encoder.set_output_buffer(2, Some(&route_map), 0);
    encoder.set_bytes(3, &to_u32(experts, "experts")?);
    encoder.set_bytes(4, &to_u32(tokens, "tokens")?);
    encoder.set_bytes(5, &to_u32(top_k, "routes")?);
    encoder.set_bytes(6, &to_u32(top_k, "route stride")?);
    super::indexed_moe_metal::dispatch_compact_routes(device, encoder, experts, tokens, top_k)?;

    if uses_metal4(device) {
        // Tiles of up to 32 routes of one expert, queued by the route kernel.
        let tile_capacity = experts + routes.div_ceil(METAL4_ROUTE_TILE);
        let tile_count = device
            .new_buffer_builder()
            .with_size_for(1, DType::U32)
            .with_label("dinf-raw-experts-tile-count")
            .build()?;
        let tile_map = device
            .new_buffer_builder()
            .with_size_for(tile_capacity * 2, DType::U32)
            .with_label("dinf-raw-experts-tile-map")
            .build()?;
        encoder.set_compute_pipeline_state(&super::indexed_moe_metal::compact_moe_pipeline(
            device,
            "dinf_build_moe_route_tiles",
        )?);
        encoder.set_input_buffer(0, Some(&counts), 0);
        encoder.set_output_buffer(1, Some(&tile_count), 0);
        encoder.set_output_buffer(2, Some(&tile_map), 0);
        encoder.set_bytes(3, &to_u32(experts, "experts")?);
        encoder.set_bytes(4, &to_u32(METAL4_ROUTE_TILE, "route tile")?);
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: experts.clamp(1, 256).next_multiple_of(32),
                height: 1,
                depth: 1,
            },
        );
        encoder.set_compute_pipeline_state(&metal4_pipeline(device, format, scales.is_some())?);
        encoder.set_input_buffer(0, Some(&weights), weights_offset);
        encoder.set_input_buffer(1, Some(&input_buffer), input_offset);
        encoder.set_output_buffer(2, Some(&output), 0);
        encoder.set_input_buffer(3, Some(&counts), 0);
        encoder.set_input_buffer(4, Some(&route_map), 0);
        encoder.set_bytes(5, &to_u32(in_dim, "input width")?);
        encoder.set_bytes(6, &to_u32(out_dim, "output width")?);
        encoder.set_bytes(7, &to_u32(tokens, "tokens")?);
        encoder.set_bytes(8, &to_u32(top_k, "routes")?);
        encoder.set_bytes(9, &to_u32(input_rows, "input rows")?);
        encoder.set_bytes(10, &to_u32(row_bytes, "row bytes")?);
        encoder.set_bytes(11, &to_u32(expert_bytes, "expert bytes")?);
        encoder.set_bytes(12, &0u32);
        encoder.set_input_buffer(13, Some(&tile_count), 0);
        encoder.set_input_buffer(14, Some(&tile_map), 0);
        encoder.set_bytes(15, &1u32);
        if let Some((scales, scales_offset)) = &scales {
            encoder.set_input_buffer(16, Some(scales), *scales_offset);
        }
        // Weights [rows][64] and activations [32][64] in half plus 32 route
        // offsets, or the f32 result tile, whichever is larger.
        let staging = (METAL4_TILE_ROWS * 64 + METAL4_ROUTE_TILE * 64) * 2 + METAL4_ROUTE_TILE * 4;
        let result = METAL4_TILE_ROWS * METAL4_ROUTE_TILE * 4;
        encoder.set_threadgroup_memory_length(0, staging.max(result));
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: tile_capacity,
                height: out_dim.div_ceil(METAL4_TILE_ROWS),
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: METAL4_TILE_ROWS * 2,
                height: 1,
                depth: 1,
            },
        );
        return Ok(wrap(output));
    }

    encoder.set_compute_pipeline_state(&pipeline(device, matrix)?);
    encoder.set_input_buffer(0, Some(&weights), weights_offset);
    encoder.set_input_buffer(1, Some(&input_buffer), input_offset);
    encoder.set_output_buffer(2, Some(&output), 0);
    encoder.set_input_buffer(3, Some(&counts), 0);
    encoder.set_input_buffer(4, Some(&route_map), 0);
    let input_row_bytes = in_dim * 4;
    let parameters: [u64; 11] = [
        in_dim as u64,
        row_bytes as u64,
        expert_bytes as u64,
        input_rows as u64,
        tokens as u64,
        4,
        input_row_bytes as u64,
        (input_rows * input_row_bytes) as u64,
        out_dim as u64,
        top_k as u64,
        experts as u64,
    ];
    for (index, value) in parameters.iter().enumerate() {
        encoder.set_bytes(5 + index, value);
    }
    encoder.set_threadgroup_memory_length(0, (8192 + tokens * 4).next_multiple_of(16));
    encoder.dispatch_thread_groups(
        objc2_metal::MTLSize {
            width: tokens.div_ceil(32),
            height: out_dim.div_ceil(64),
            depth: experts,
        },
        objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    if let Some((scales, scales_offset)) = &scales {
        encoder.set_compute_pipeline_state(&pipeline(device, "dinf_raw_scale_routes")?);
        encoder.set_output_buffer(0, Some(&output), 0);
        encoder.set_input_buffer(1, Some(&ids_buffer), ids_offset);
        encoder.set_input_buffer(2, Some(scales), *scales_offset);
        let shape = [
            to_u32(out_dim, "output width")?,
            to_u32(experts, "experts")?,
        ];
        encoder.set_bytes(3, &shape);
        encoder.dispatch_threads(
            objc2_metal::MTLSize {
                width: out_dim,
                height: routes,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
    }
    Ok(wrap(output))
}

/// MXFP8 blocks (36 bytes per 32 weights) on Metal from the GGUF's `F8_E4M3`
/// values and their UE8M0 scales, one per 32 values, rejecting NaN codes.
pub(crate) fn mxfp8_blocks(values: &Tensor, scales: &Tensor) -> anyhow::Result<Tensor> {
    let candle_core::Device::Metal(device) = values.device() else {
        anyhow::bail!("MXFP8 values are not on Metal")
    };
    let count = scales.elem_count();
    anyhow::ensure!(
        values.elem_count() == count * 32,
        "MXFP8 values and scales differ in shape"
    );
    let values = values.flatten_all()?.contiguous()?;
    let scales = scales.flatten_all()?.contiguous()?;
    let (values_buffer, values_offset) = metal_view(&values, DType::U8, "MXFP8 values")?;
    let (scales_buffer, scales_offset) = metal_view(&scales, DType::U8, "MXFP8 scales")?;
    let blocks = exact_buffer(device, count * 36)?;
    let bad = Tensor::zeros(1, DType::U32, values.device())?;
    let (bad_buffer, bad_offset) = metal_view(&bad, DType::U32, "MXFP8 NaN count")?;
    {
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-raw-mxfp8-blocks");
        let encoder: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline(device, "dinf_raw_mxfp8_blocks")?);
        encoder.set_input_buffer(0, Some(&values_buffer), values_offset);
        encoder.set_input_buffer(1, Some(&scales_buffer), scales_offset);
        encoder.set_output_buffer(2, Some(&blocks), 0);
        encoder.set_output_buffer(3, Some(&bad_buffer), bad_offset);
        encoder.set_bytes(4, &to_u32(count, "MXFP8 blocks")?);
        encoder.dispatch_threads(
            objc2_metal::MTLSize {
                width: count,
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
    anyhow::ensure!(
        bad.to_vec1::<u32>()?[0] == 0,
        "MXFP8 weights hold NaN codes"
    );
    Ok(Tensor::from((
        Storage::Metal(MetalStorage::new(
            blocks,
            device.clone(),
            count * 36,
            DType::U8,
        )),
        Shape::from(count * 36),
    )))
}

/// A BF16 expert bank on Metal as F16 bytes (U8, two per weight) in an
/// exact-size buffer, for the indexed F16 kernels: the narrowing the host
/// path made, without host copies.
pub(crate) fn bf16_bank_as_f16_bytes(values: &Tensor) -> anyhow::Result<Tensor> {
    let candle_core::Device::Metal(device) = values.device() else {
        anyhow::bail!("BF16 expert bank is not on Metal")
    };
    let values = values.flatten_all()?.contiguous()?;
    let count = values.elem_count();
    let (input, input_offset) = metal_view(&values, DType::BF16, "BF16 expert bank")?;
    let output = exact_buffer(device, count * 2)?;
    if count > 0 {
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-raw-bf16-to-f16");
        let encoder: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline(device, "dinf_raw_bf16_to_f16")?);
        encoder.set_input_buffer(0, Some(&input), input_offset);
        encoder.set_output_buffer(1, Some(&output), 0);
        encoder.set_bytes(2, &to_u32(count, "BF16 expert weights")?);
        encoder.dispatch_threads(
            objc2_metal::MTLSize {
                width: count,
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
    Ok(Tensor::from((
        Storage::Metal(MetalStorage::new(
            output,
            device.clone(),
            count * 2,
            DType::U8,
        )),
        Shape::from(count * 2),
    )))
}
