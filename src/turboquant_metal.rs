//! Metal implementation of the TurboQuant KV representation.
//!
//! The portable TurboQuant cache owns packed records on the host and expands
//! the complete cache back to F32 on every update.  This module keeps the
//! packed records on Metal instead.  A record contains the packed scalar
//! indices followed by the source vector's F32 norm.  K and V are stored as
//! separate `[1, heads, sequence, record_bytes]` tensors; the record order is
//! head-major so a sequence append is a narrow, regular copy.
//!
//! Quantization and dequantization are deliberately separate from cache
//! ownership.  Callers can validate and construct the new tensors before
//! publishing them into a cache. Public cache operations check compact
//! validation flags immediately. The stage path defers those checks to the
//! end of the forward pass and rejects/rolls back that pass before exposing
//! output when any source, sign, or reconstruction is invalid.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{
    CpuStorage, CustomOp2, CustomOp3, DType, Device, InplaceOp3, Layout, MetalStorage, Result,
    Shape, Storage, Tensor,
};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::MTLCompileOptions;

/// Metal threadgroup scratch is statically sized in the shader.  Attention
/// head widths in supported models are below this limit; larger
/// geometries must use the explicit portable path rather than silently
/// changing precision or algorithm.
pub(crate) const MAX_METAL_HEAD_DIM: usize = 1024;

const TURBOQUANT_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant float tq_centroids_3[8] = {
  -2.1520f, -1.3440f, -0.7560f, -0.2451f,
   0.2451f,  0.7560f,  1.3440f,  2.1520f
};

constant float tq_boundaries_3[7] = {
  -1.7480f, -1.0500f, -0.5006f, 0.0f,
   0.5006f,  1.0500f,  1.7480f
};

constant float tq_centroids_4[16] = {
  -2.69013f, -2.04740f, -1.60586f, -1.24921f,
  -0.93835f, -0.65458f, -0.38699f, -0.12808f,
   0.12808f,  0.38699f,  0.65458f,  0.93835f,
   1.24921f,  1.60586f,  2.04740f,  2.69013f
};

constant float tq_boundaries_4[15] = {
  -2.36876f, -1.82663f, -1.42753f, -1.09378f,
  -0.79647f, -0.52079f, -0.25754f,  0.0f,
   0.25754f,  0.52079f,  0.79647f,  1.09378f,
   1.42753f,  1.82663f,  2.36876f
};

inline uint tq_mask(const uint bits) {
  return bits == 8u ? 0xffu : ((1u << bits) - 1u);
}

inline uint tq_quantize_3(const float value) {
  if (value < tq_boundaries_3[3]) {
    if (value < tq_boundaries_3[1]) {
      return value < tq_boundaries_3[0] ? 0u : 1u;
    }
    return value < tq_boundaries_3[2] ? 2u : 3u;
  }
  if (value < tq_boundaries_3[5]) {
    return value < tq_boundaries_3[4] ? 4u : 5u;
  }
  return value < tq_boundaries_3[6] ? 6u : 7u;
}

inline uint tq_quantize_index(const float value, const uint bits) {
  if (bits == 3u) {
    return tq_quantize_3(value);
  }
  if (bits == 4u) {
    uint index = 0u;
    while (index < 15u && value >= tq_boundaries_4[index]) {
      index += 1u;
    }
    return index;
  }
  // Rust's f32::round rounds halfway cases away from zero, as does the Metal
  // round() intrinsic.  Keep the xor encoding identical to turboquant.rs.
  const float scaled = clamp(value, -4.0f, 4.0f) * (127.0f / 4.0f);
  const int rounded = int(round(scaled));
  return (uint(rounded) & 0xffu) ^ 0x80u;
}

inline float tq_centroid(const uint index, const uint bits) {
  if (bits == 3u) {
    return tq_centroids_3[index & 7u];
  }
  if (bits == 4u) {
    return tq_centroids_4[index & 15u];
  }
  // The CPU encoder stores the signed i8 shifted by 128 (the xor encoding is
  // equivalent to that shift over the signed byte range).  Decode it without
  // relying on implementation-defined uint-to-char conversion.
  return float(int(index & 0xffu) - 128) / (127.0f / 4.0f);
}

inline uint tq_unpack_index(
    device const uchar *record,
    const uint coordinate,
    const uint bits) {
  const uint bit_offset = coordinate * bits;
  const uint byte_offset = bit_offset >> 3;
  const uint shift = bit_offset & 7u;
  uint index = uint(record[byte_offset]) >> shift;
  if (shift + bits > 8u) {
    index |= uint(record[byte_offset + 1u]) << (8u - shift);
  }
  return index & tq_mask(bits);
}

inline void tq_wht_unnormalized(
    threadgroup float *scratch,
    const uint tid,
    const uint head_dim) {
  // All supported Apple SIMD groups are 32 lanes.  WHT butterflies whose
  // span stays below 32 never cross a SIMD-group boundary, so shuffles avoid
  // a threadgroup barrier while preserving the CPU order exactly.  Keep the
  // value in a register until the first cross-group stage.
  float value = scratch[tid];
  for (uint span = 1u; span < head_dim && span < 32u; span <<= 1u) {
    const float other = simd_shuffle_xor(value, ushort(span));
    // H2 is [a + b, a - b]; the high lane therefore computes other - value.
    value = (tid & span) == 0u ? value + other : other - value;
  }
  scratch[tid] = value;
  if (head_dim <= 32u) {
    return;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint span = 32u; span < head_dim; span <<= 1u) {
    const uint pair_width = span << 1u;
    const uint in_pair = tid % pair_width;
    if (in_pair < span) {
      const uint base = (tid / pair_width) * pair_width;
      const float a = scratch[base + in_pair];
      const float b = scratch[base + in_pair + span];
      scratch[base + in_pair] = a + b;
      scratch[base + in_pair + span] = a - b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

inline void tq_wht_normalized(
    threadgroup float *scratch,
    const uint tid,
    const uint head_dim) {
  tq_wht_unnormalized(scratch, tid, head_dim);
  scratch[tid] *= rsqrt(float(head_dim));
}

inline uint tq_read_norm_bits(device const uchar *record, const uint packed_bytes) {
  return uint(record[packed_bytes])
      | (uint(record[packed_bytes + 1u]) << 8u)
      | (uint(record[packed_bytes + 2u]) << 16u)
      | (uint(record[packed_bytes + 3u]) << 24u);
}

kernel void dinf_turboquant_quantize_kv(
    device const float *key [[buffer(0)]],
    device const float *value [[buffer(1)]],
    device const float *signs [[buffer(2)]],
    device uchar *output [[buffer(3)]],
    device atomic_uint *invalid [[buffer(4)]],
    constant uint &heads [[buffer(5)]],
    constant uint &sequence [[buffer(6)]],
    constant uint &head_dim [[buffer(7)]],
    constant uint &bits [[buffer(8)]],
    constant uint &packed_bytes [[buffer(9)]],
    constant uint &record_bytes [[buffer(10)]],
    constant float &max_norm [[buffer(11)]],
    constant uint &output_capacity [[buffer(12)]],
    constant uint &output_start [[buffer(13)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[1024];
  const uint vectors_per_side = heads * sequence;
  const uint side = group / vectors_per_side;
  const uint vector = group - side * vectors_per_side;
  const uint head = vector / sequence;
  const uint token = vector - head * sequence;
  const uint input_base = (head * sequence + token) * head_dim;
  device const float *source = side == 0u ? key : value;
  const float raw = source[input_base + tid];
  const float sign = signs[head * head_dim + tid];
  const bool finite = isfinite(raw) && isfinite(sign)
      && (sign == 1.0f || sign == -1.0f);
  if (!finite) {
    atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
  }
  const float safe = isfinite(raw) ? raw : 0.0f;
  // Find the maximum magnitude before forming squares.  The CPU reference
  // accumulates in f64, and valid API inputs may be as large as 1e20; a
  // direct f32 square would overflow even when the final norm is representable.
  scratch[tid] = fabs(safe);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint span = head_dim >> 1u; span > 0u; span >>= 1u) {
    if (tid < span) {
      scratch[tid] = max(scratch[tid], scratch[tid + span]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float max_abs = scratch[0];
  // Every lane must observe the reduction result before lane 0 reuses
  // scratch[0] for the sum-of-squares reduction.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float scaled = max_abs > 0.0f ? safe / max_abs : 0.0f;
  scratch[tid] = scaled * scaled;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint span = head_dim >> 1u; span > 0u; span >>= 1u) {
    if (tid < span) {
      scratch[tid] += scratch[tid + span];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float norm = max_abs * sqrt(max(scratch[0], 0.0f));
  // As above, make the reduction value visible to every lane before any lane
  // overwrites scratch with its signed normalized coordinate.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!isfinite(norm) || norm > max_norm) {
    atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
    norm = 0.0f;
  }
  // Match turboquant.rs: only norms strictly below the floor are treated as
  // zero, so an exactly-equal norm still takes the normal quantized path.
  const float normalized = norm >= 1.0e-10f ? safe / norm : 0.0f;
  scratch[tid] = normalized * (isfinite(sign) ? sign : 0.0f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tq_wht_unnormalized(scratch, tid, head_dim);
  if (head_dim <= 32u) {
    // The SIMD path intentionally leaves the transformed values in a
    // per-lane scratch slot without a barrier.  Lane zero packs all slots.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0u) {
    const uint output_vector = side * heads * output_capacity
        + head * output_capacity + output_start + token;
    device uchar *record = output + output_vector * record_bytes;
    for (uint i = 0u; i < record_bytes; ++i) {
      record[i] = 0;
    }
    const uint norm_bits = as_type<uint>(norm);
    record[packed_bytes] = uchar(norm_bits & 0xffu);
    record[packed_bytes + 1u] = uchar((norm_bits >> 8u) & 0xffu);
    record[packed_bytes + 2u] = uchar((norm_bits >> 16u) & 0xffu);
    record[packed_bytes + 3u] = uchar(norm_bits >> 24u);
    // Match turboquant.rs: a zero (or numerically tiny) vector has a zero
    // norm and an all-zero packed payload rather than the quantizer's bin for
    // the scalar value zero.
    if (norm >= 1.0e-10f) {
      for (uint coordinate = 0u; coordinate < head_dim; ++coordinate) {
        const uint index = tq_quantize_index(scratch[coordinate], bits);
        const uint bit_offset = coordinate * bits;
        const uint byte_offset = bit_offset >> 3;
        const uint shift = bit_offset & 7u;
        record[byte_offset] |= uchar((index << shift) & 0xffu);
        if (shift + bits > 8u) {
          record[byte_offset + 1u] |= uchar(index >> (8u - shift));
        }
      }
    }
  }
}

kernel void dinf_turboquant_dequantize_kv_f32(
    device const uchar *key [[buffer(0)]],
    device const uchar *value [[buffer(1)]],
    device const float *signs [[buffer(2)]],
    device float *output [[buffer(3)]],
    device atomic_uint *invalid [[buffer(4)]],
    constant uint &heads [[buffer(5)]],
    constant uint &sequence [[buffer(6)]],
    constant uint &head_dim [[buffer(7)]],
    constant uint &bits [[buffer(8)]],
    constant uint &packed_bytes [[buffer(9)]],
    constant uint &record_bytes [[buffer(10)]],
    constant uint &key_head_stride [[buffer(11)]],
    constant uint &value_head_stride [[buffer(12)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[1024];
  const uint vectors_per_side = heads * sequence;
  const uint side = group / vectors_per_side;
  const uint vector = group - side * vectors_per_side;
  const uint head = vector / sequence;
  const uint token = vector - head * sequence;
  device const uchar *source = side == 0u ? key : value;
  const uint head_stride = side == 0u ? key_head_stride : value_head_stride;
  device const uchar *record = source
      + (head * head_stride + token) * record_bytes;
  const float norm = as_type<float>(tq_read_norm_bits(record, packed_bytes));
  if (!isfinite(norm) || norm < 0.0f) {
    atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
  }
  const uint index = tq_unpack_index(record, tid, bits);
  scratch[tid] = tq_centroid(index, bits) * rsqrt(float(head_dim));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tq_wht_normalized(scratch, tid, head_dim);
  const float sign = signs[head * head_dim + tid];
  if (!isfinite(sign) || (sign != 1.0f && sign != -1.0f)) {
    atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
  }
  const float restored = (norm >= 1.0e-10f ? scratch[tid] * sign * norm : 0.0f);
  output[group * head_dim + tid] = isfinite(restored) ? restored : 0.0f;
  if (!isfinite(restored)) {
    atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
  }
}

kernel void dinf_turboquant_dequantize_kv_f16(
    device const uchar *key [[buffer(0)]],
    device const uchar *value [[buffer(1)]],
    device const float *signs [[buffer(2)]],
    device half *output [[buffer(3)]],
    device atomic_uint *invalid [[buffer(4)]],
    constant uint &heads [[buffer(5)]],
    constant uint &sequence [[buffer(6)]],
    constant uint &head_dim [[buffer(7)]],
    constant uint &bits [[buffer(8)]],
    constant uint &packed_bytes [[buffer(9)]],
    constant uint &record_bytes [[buffer(10)]],
    constant uint &key_head_stride [[buffer(11)]],
    constant uint &value_head_stride [[buffer(12)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[1024];
  const uint vectors_per_side = heads * sequence;
  const uint side = group / vectors_per_side;
  const uint vector = group - side * vectors_per_side;
  const uint head = vector / sequence;
  device const uchar *source = side == 0u ? key : value;
  const uint token = vector - head * sequence;
  const uint head_stride = side == 0u ? key_head_stride : value_head_stride;
  device const uchar *record = source
      + (head * head_stride + token) * record_bytes;
  const float norm = as_type<float>(tq_read_norm_bits(record, packed_bytes));
  if (!isfinite(norm) || norm < 0.0f) {
    atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
  }
  const uint index = tq_unpack_index(record, tid, bits);
  scratch[tid] = tq_centroid(index, bits) * rsqrt(float(head_dim));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tq_wht_normalized(scratch, tid, head_dim);
  const float sign = signs[head * head_dim + tid];
  if (!isfinite(sign) || (sign != 1.0f && sign != -1.0f)) {
    atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
  }
  const float restored = (norm >= 1.0e-10f ? scratch[tid] * sign * norm : 0.0f);
  const half converted = half(isfinite(restored) ? restored : 0.0f);
  output[group * head_dim + tid] = converted;
  if (!isfinite(restored) || !isfinite(float(converted))) {
    atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
  }
}

// Dequantize only the scalar centroid and norm.  The inverse WHT is deferred
// to the attention query/output rotation so long cached vectors do not each
// pay a threadgroup transform.
kernel void dinf_turboquant_dequantize_rotated_kv_f32(
    device const uchar *key [[buffer(0)]],
    device const uchar *value [[buffer(1)]],
    device float *output [[buffer(2)]],
    device atomic_uint *invalid [[buffer(3)]],
    constant uint &heads [[buffer(4)]],
    constant uint &sequence [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant uint &bits [[buffer(7)]],
    constant uint &packed_bytes [[buffer(8)]],
    constant uint &record_bytes [[buffer(9)]],
    constant uint &key_head_stride [[buffer(10)]],
    constant uint &value_head_stride [[buffer(11)]],
    uint gid [[thread_position_in_grid]]) {
  const uint group = gid / head_dim;
  const uint tid = gid - group * head_dim;
  const uint vectors_per_side = heads * sequence;
  const uint side = group / vectors_per_side;
  const uint vector = group - side * vectors_per_side;
  const uint head = vector / sequence;
  const uint token = vector - head * sequence;
  device const uchar *source = side == 0u ? key : value;
  const uint head_stride = side == 0u ? key_head_stride : value_head_stride;
  device const uchar *record = source
      + (head * head_stride + token) * record_bytes;
  const float norm = as_type<float>(tq_read_norm_bits(record, packed_bytes));
  if (!isfinite(norm) || norm < 0.0f) {
    atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
  }
  const uint index = tq_unpack_index(record, tid, bits);
  const float restored = norm >= 1.0e-10f
      ? tq_centroid(index, bits) * norm * rsqrt(float(head_dim))
      : 0.0f;
  output[group * head_dim + tid] = isfinite(restored) ? restored : 0.0f;
  if (!isfinite(restored)) {
    atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
  }
}

kernel void dinf_turboquant_dequantize_rotated_kv_f16(
    device const uchar *key [[buffer(0)]],
    device const uchar *value [[buffer(1)]],
    device half *output [[buffer(2)]],
    device atomic_uint *invalid [[buffer(3)]],
    constant uint &heads [[buffer(4)]],
    constant uint &sequence [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant uint &bits [[buffer(7)]],
    constant uint &packed_bytes [[buffer(8)]],
    constant uint &record_bytes [[buffer(9)]],
    constant uint &key_head_stride [[buffer(10)]],
    constant uint &value_head_stride [[buffer(11)]],
    uint gid [[thread_position_in_grid]]) {
  const uint group = gid / head_dim;
  const uint tid = gid - group * head_dim;
  const uint vectors_per_side = heads * sequence;
  const uint side = group / vectors_per_side;
  const uint vector = group - side * vectors_per_side;
  const uint head = vector / sequence;
  const uint token = vector - head * sequence;
  device const uchar *source = side == 0u ? key : value;
  const uint head_stride = side == 0u ? key_head_stride : value_head_stride;
  device const uchar *record = source
      + (head * head_stride + token) * record_bytes;
  const float norm = as_type<float>(tq_read_norm_bits(record, packed_bytes));
  if (!isfinite(norm) || norm < 0.0f) {
    atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
  }
  const uint index = tq_unpack_index(record, tid, bits);
  const float restored = norm >= 1.0e-10f
      ? tq_centroid(index, bits) * norm * rsqrt(float(head_dim))
      : 0.0f;
  const half converted = half(isfinite(restored) ? restored : 0.0f);
  output[group * head_dim + tid] = converted;
  if (!isfinite(restored) || !isfinite(float(converted))) {
    atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
  }
}

// The qualified 8-bit profile has four-byte-aligned records. Process two
// four-coordinate vectors per lane, sharing the norm load and validation.
// The 3D grid avoids integer division/modulo for every cached coordinate.
kernel void dinf_turboquant_dequantize_rotated_kv8_f16_vector(
    device const uchar *key [[buffer(0)]],
    device const uchar *value [[buffer(1)]],
    device half *output [[buffer(2)]],
    device atomic_uint *invalid [[buffer(3)]],
    constant uint &heads [[buffer(4)]],
    constant uint &sequence [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant uint &bits [[buffer(7)]],
    constant uint &packed_bytes [[buffer(8)]],
    constant uint &record_bytes [[buffer(9)]],
    constant uint &key_head_stride [[buffer(10)]],
    constant uint &value_head_stride [[buffer(11)]],
    uint3 position [[thread_position_in_grid]]) {
  const uint coordinate = position.x * 8u;
  const uint token = position.y;
  const uint side = position.z >= heads ? 1u : 0u;
  const uint head = position.z - side * heads;
  device const uchar *source = side == 0u ? key : value;
  const uint stride = side == 0u ? key_head_stride : value_head_stride;
  device const uchar *record = source + (head * stride + token) * record_bytes;
  // Record width and the checked view offset are multiples of four here.
  const float norm = *reinterpret_cast<device const float *>(record + packed_bytes);
  bool invalid_value = !isfinite(norm) || norm < 0.0f;
  #pragma unroll
  for (uint part = 0u; part < 2u; ++part) {
    const uint offset = coordinate + part * 4u;
    const uchar4 indices = *reinterpret_cast<device const uchar4 *>(record + offset);
    const float4 centroid = (float4(indices) - 128.0f) / (127.0f / 4.0f);
    const float4 restored = norm >= 1.0e-10f
        ? centroid * norm * rsqrt(float(head_dim)) : float4(0.0f);
    const half4 converted = half4(select(float4(0.0f), restored, isfinite(restored)));
    *reinterpret_cast<device half4 *>(output
        + (position.z * sequence + token) * head_dim + offset) = converted;
    invalid_value |= !all(isfinite(restored)) || !all(isfinite(float4(converted)));
  }
  if (invalid_value) {
    atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
  }
}

kernel void dinf_turboquant_rotate_attention_f32(
    device const float *input [[buffer(0)]],
    device const float *signs [[buffer(1)]],
    device float *output [[buffer(2)]],
    device atomic_uint *invalid [[buffer(3)]],
    constant uint &heads [[buffer(4)]],
    constant uint &sequence [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant uint &inverse [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[1024];
  const uint vector = group;
  const uint input_base = vector * head_dim;
  const float raw = input[input_base + tid];
  const float sign = signs[tid];
  const bool raw_finite = isfinite(raw);
  const bool sign_valid = isfinite(sign) && (sign == 1.0f || sign == -1.0f);
  if (!raw_finite || !sign_valid) {
    atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
  }
  const float safe_sign = sign_valid ? sign : 0.0f;
  const float safe = raw_finite ? raw : 0.0f;
  scratch[tid] = inverse == 0u ? safe * safe_sign : safe;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tq_wht_normalized(scratch, tid, head_dim);
  const float restored = inverse == 0u ? scratch[tid] : scratch[tid] * safe_sign;
  output[input_base + tid] = isfinite(restored) ? restored : 0.0f;
  if (!isfinite(restored)) {
    atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
  }
}

// Identical F32 rotation followed by the same F16 rounding used by SDPA's
// ordinary query conversion, fused into the output write.
kernel void dinf_turboquant_rotate_query_f32_f16(
    device const float *input [[buffer(0)]],
    device const float *signs [[buffer(1)]],
    device half *output [[buffer(2)]],
    device atomic_uint *invalid [[buffer(3)]],
    constant uint &heads [[buffer(4)]],
    constant uint &sequence [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant uint &inverse [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[1024];
  const uint input_base = group * head_dim;
  const float raw = input[input_base + tid];
  const float sign = signs[tid];
  const bool raw_finite = isfinite(raw);
  const bool sign_valid = isfinite(sign) && (sign == 1.0f || sign == -1.0f);
  if (!raw_finite || !sign_valid) {
    atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
  }
  const float safe_sign = sign_valid ? sign : 0.0f;
  const float safe = raw_finite ? raw : 0.0f;
  scratch[tid] = inverse == 0u ? safe * safe_sign : safe;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tq_wht_normalized(scratch, tid, head_dim);
  const float restored = inverse == 0u ? scratch[tid] : scratch[tid] * safe_sign;
  const half converted = half(isfinite(restored) ? restored : 0.0f);
  output[input_base + tid] = converted;
  if (!isfinite(restored) || !isfinite(float(converted))) {
    atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
  }
}

kernel void dinf_turboquant_rotate_attention_f16(
    device const half *input [[buffer(0)]],
    device const float *signs [[buffer(1)]],
    device half *output [[buffer(2)]],
    device atomic_uint *invalid [[buffer(3)]],
    constant uint &heads [[buffer(4)]],
    constant uint &sequence [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant uint &inverse [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[1024];
  const uint vector = group;
  const uint input_base = vector * head_dim;
  const float raw = float(input[input_base + tid]);
  const float sign = signs[tid];
  const bool raw_finite = isfinite(raw);
  const bool sign_valid = isfinite(sign) && (sign == 1.0f || sign == -1.0f);
  if (!raw_finite || !sign_valid) {
    atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
  }
  const float safe_sign = sign_valid ? sign : 0.0f;
  const float safe = raw_finite ? raw : 0.0f;
  scratch[tid] = inverse == 0u ? safe * safe_sign : safe;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tq_wht_normalized(scratch, tid, head_dim);
  const float restored = inverse == 0u ? scratch[tid] : scratch[tid] * safe_sign;
  const half converted = half(isfinite(restored) ? restored : 0.0f);
  output[input_base + tid] = converted;
  if (!isfinite(restored) || !isfinite(float(converted))) {
    atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
  }
}

"#;

#[derive(Clone)]
struct TurboQuantPipelines {
    quantize: ComputePipeline,
    dequantize_f32: ComputePipeline,
    dequantize_f16: ComputePipeline,
    dequantize_rotated_f32: ComputePipeline,
    dequantize_rotated_f16: ComputePipeline,
    dequantize_rotated8_f16x4: ComputePipeline,
    rotate_f32: ComputePipeline,
    rotate_query_f32_f16: ComputePipeline,
    rotate_f16: ComputePipeline,
}

static PIPELINES: OnceLock<Mutex<HashMap<DeviceId, TurboQuantPipelines>>> = OnceLock::new();

fn pipelines(device: &candle_core::MetalDevice) -> Result<TurboQuantPipelines> {
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let device_id = device.id();
    if let Some(pipelines) = cache.lock().unwrap().get(&device_id).cloned() {
        return Ok(pipelines);
    }
    let options = MTLCompileOptions::new();
    // The kernels intentionally validate NaN/Inf before publishing a cache
    // update.  Fast-math may legally fold those predicates away.
    #[allow(deprecated)]
    options.setFastMathEnabled(false);
    let library = device
        .metal_device()
        .new_library_with_source(TURBOQUANT_MSL, Some(&options))
        .map_err(candle_core::Error::wrap)?;
    let function = |name: &str| {
        library
            .get_function(name, None)
            .map_err(candle_core::Error::wrap)
    };
    let compiled = TurboQuantPipelines {
        quantize: device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function("dinf_turboquant_quantize_kv")?)
            .map_err(candle_core::Error::wrap)?,
        dequantize_f32: device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function(
                "dinf_turboquant_dequantize_kv_f32",
            )?)
            .map_err(candle_core::Error::wrap)?,
        dequantize_f16: device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function(
                "dinf_turboquant_dequantize_kv_f16",
            )?)
            .map_err(candle_core::Error::wrap)?,
        dequantize_rotated_f32: device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function(
                "dinf_turboquant_dequantize_rotated_kv_f32",
            )?)
            .map_err(candle_core::Error::wrap)?,
        dequantize_rotated_f16: device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function(
                "dinf_turboquant_dequantize_rotated_kv_f16",
            )?)
            .map_err(candle_core::Error::wrap)?,
        rotate_f32: device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function(
                "dinf_turboquant_rotate_attention_f32",
            )?)
            .map_err(candle_core::Error::wrap)?,
        dequantize_rotated8_f16x4: device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function(
                "dinf_turboquant_dequantize_rotated_kv8_f16_vector",
            )?)
            .map_err(candle_core::Error::wrap)?,
        rotate_f16: device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function(
                "dinf_turboquant_rotate_attention_f16",
            )?)
            .map_err(candle_core::Error::wrap)?,
        rotate_query_f32_f16: device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function(
                "dinf_turboquant_rotate_query_f32_f16",
            )?)
            .map_err(candle_core::Error::wrap)?,
    };
    let mut cache = cache.lock().unwrap();
    Ok(cache.entry(device_id).or_insert(compiled).clone())
}

fn checked_u32(value: usize, label: &str) -> Result<u32> {
    value
        .try_into()
        .map_err(|_| candle_core::Error::Msg(format!("TurboQuant {label} exceeds Metal u32")))
}

fn checked_product(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| candle_core::Error::Msg(format!("TurboQuant {label} size overflows usize")))
}

fn checked_kernel_product(left: usize, right: usize, label: &str) -> Result<usize> {
    let elements = checked_product(left, right, label)?;
    checked_u32(elements, label)?;
    Ok(elements)
}

fn record_geometry(bits: u8, head_dim: usize) -> Result<(usize, usize)> {
    if !matches!(bits, 3 | 4 | 8) {
        candle_core::bail!("TurboQuant Metal supports only 3, 4, or 8 bits")
    }
    if !head_dim.is_power_of_two() || !(8..=MAX_METAL_HEAD_DIM).contains(&head_dim) {
        candle_core::bail!(
            "TurboQuant Metal head dimension must be a power of two in 8..={MAX_METAL_HEAD_DIM}"
        )
    }
    let packed_bytes = head_dim
        .checked_mul(bits as usize)
        .ok_or_else(|| candle_core::Error::Msg("TurboQuant packed size overflows".into()))?
        .div_ceil(8);
    let record_bytes = packed_bytes
        .checked_add(4)
        .ok_or_else(|| candle_core::Error::Msg("TurboQuant record size overflows".into()))?;
    Ok((packed_bytes, record_bytes))
}

fn validate_contiguous_span(storage: &MetalStorage, layout: &Layout, label: &str) -> Result<()> {
    if !layout.is_contiguous() {
        candle_core::bail!("TurboQuant Metal {label} must be contiguous")
    }
    let (_, stop) = layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg(format!("TurboQuant Metal {label} has no contiguous span"))
    })?;
    let bytes = stop
        .checked_mul(storage.dtype().size_in_bytes())
        .ok_or_else(|| {
            candle_core::Error::Msg(format!("TurboQuant Metal {label} span overflows"))
        })?;
    if bytes > storage.buffer().length() {
        candle_core::bail!("TurboQuant Metal {label} span exceeds its buffer")
    }
    Ok(())
}

/// Validate a packed cache view that may retain a larger physical capacity
/// than its logical sequence length.  Such a view is contiguous within each
/// head but has a head stride larger than `sequence * record_bytes`; passing
/// that stride to the kernel avoids copying the active prefix into a new
/// contiguous allocation after every append.
fn validate_packed_cache_layout(
    storage: &MetalStorage,
    layout: &Layout,
    heads: usize,
    sequence: usize,
    record_bytes: usize,
    label: &str,
) -> Result<u32> {
    if storage.dtype() != DType::U8 {
        candle_core::bail!("TurboQuant Metal {label} must be u8")
    }
    let (batch, actual_heads, actual_sequence, actual_record_bytes) = layout.shape().dims4()?;
    if (batch, actual_heads, actual_sequence, actual_record_bytes)
        != (1, heads, sequence, record_bytes)
    {
        candle_core::bail!("TurboQuant Metal {label} geometry differs")
    }
    let strides = layout.stride();
    if strides.len() != 4
        || strides[3] != 1
        || strides[2] != record_bytes
        || strides[1] < sequence.saturating_mul(record_bytes)
        || !strides[1].is_multiple_of(record_bytes)
    {
        candle_core::bail!("TurboQuant Metal {label} must be head-major with contiguous records")
    }
    let last = layout
        .start_offset()
        .checked_add(
            (heads.saturating_sub(1))
                .checked_mul(strides[1])
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!("TurboQuant Metal {label} span overflows"))
                })?,
        )
        .and_then(|offset| {
            offset.checked_add(
                (sequence.saturating_sub(1))
                    .checked_mul(record_bytes)
                    .and_then(|offset| offset.checked_add(record_bytes.saturating_sub(1)))?,
            )
        })
        .and_then(|offset| offset.checked_add(1))
        .ok_or_else(|| {
            candle_core::Error::Msg(format!("TurboQuant Metal {label} span overflows"))
        })?;
    if last > storage.buffer().length() {
        candle_core::bail!("TurboQuant Metal {label} span exceeds its buffer")
    }
    let head_stride = checked_u32(strides[1] / record_bytes, &format!("{label} head stride"))?;
    let relative_records = (heads.saturating_sub(1))
        .checked_mul(head_stride as usize)
        .and_then(|offset| offset.checked_add(sequence.saturating_sub(1)))
        .and_then(|offset| offset.checked_add(1))
        .ok_or_else(|| {
            candle_core::Error::Msg(format!("TurboQuant Metal {label} index overflows"))
        })?;
    checked_kernel_product(
        relative_records,
        record_bytes,
        &format!("{label} index span"),
    )?;
    Ok(head_stride)
}

/// Maximum number of device validation words a forward-pass scope may defer.
/// A normal stage pass uses one shared word per Metal device; the cap also
/// prevents a malformed caller from turning this thread-local collector into
/// an unbounded allocation.
const MAX_DEFERRED_VALIDATION_FLAGS: usize = 4096;

struct PendingValidationFlag {
    flag: Arc<candle_metal_kernels::metal::Buffer>,
    device: candle_core::MetalDevice,
}

struct ValidationCollector {
    flags: Vec<PendingValidationFlag>,
}

impl ValidationCollector {
    fn new() -> Self {
        Self { flags: Vec::new() }
    }

    /// Return the one validation word shared by all kernels on this device in
    /// the current forward-pass scope. It is initialized once and each kernel
    /// atomically ORs into it, so the stage pays one readback instead of
    /// allocating a Metal buffer for every layer operation.
    fn shared_flag(
        &mut self,
        device: &candle_core::MetalDevice,
    ) -> Result<Arc<candle_metal_kernels::metal::Buffer>> {
        if let Some(pending) = self
            .flags
            .iter()
            .find(|pending| pending.device.id() == device.id())
        {
            return Ok(pending.flag.clone());
        }
        if self.flags.len() >= MAX_DEFERRED_VALIDATION_FLAGS {
            candle_core::bail!(
                "TurboQuant deferred validation exceeded {MAX_DEFERRED_VALIDATION_FLAGS} devices"
            )
        }
        self.flags.try_reserve(1).map_err(|_| {
            candle_core::Error::Msg("TurboQuant deferred validation allocation failed".into())
        })?;
        let flag = device
            .new_buffer_builder()
            .with_data(&[0u32])
            .with_label("dinf-turboquant-forward-validation")
            .build()?;
        self.flags.push(PendingValidationFlag {
            flag: flag.clone(),
            device: device.clone(),
        });
        Ok(flag)
    }

    fn push(
        &mut self,
        flag: Arc<candle_metal_kernels::metal::Buffer>,
        device: &candle_core::MetalDevice,
    ) -> Result<()> {
        // Normal call sites obtain their flag through `shared_flag`, so this
        // is normally a no-op. Keep registration defensive for future kernels
        // that might pass a separately allocated flag: it must still be
        // checked before the scope is accepted.
        if self
            .flags
            .iter()
            .any(|pending| Arc::ptr_eq(&pending.flag, &flag))
        {
            return Ok(());
        }
        if self.flags.len() >= MAX_DEFERRED_VALIDATION_FLAGS {
            candle_core::bail!(
                "TurboQuant deferred validation exceeded {MAX_DEFERRED_VALIDATION_FLAGS} flags"
            )
        }
        self.flags.try_reserve(1).map_err(|_| {
            candle_core::Error::Msg("TurboQuant deferred validation allocation failed".into())
        })?;
        self.flags.push(PendingValidationFlag {
            flag,
            device: device.clone(),
        });
        Ok(())
    }

    fn flush(self) -> Result<()> {
        // All flags are shared Metal buffers created with `with_data`.  One
        // wait per device makes every pending atomic write visible to the CPU;
        // reading their contents directly avoids a blit-and-wait per layer.
        let mut devices = Vec::new();
        devices.try_reserve(self.flags.len()).map_err(|_| {
            candle_core::Error::Msg("TurboQuant deferred device list allocation failed".into())
        })?;
        for pending in &self.flags {
            if !devices
                .iter()
                .any(|(id, _): &(DeviceId, candle_core::MetalDevice)| *id == pending.device.id())
            {
                devices.push((pending.device.id(), pending.device.clone()));
            }
        }
        for (_, device) in devices {
            device.flush_and_wait_current()?;
        }
        for pending in self.flags {
            // A deferred scope may contain quantization, dequantization, and
            // rotation kernels. Report the shared failure at the scope
            // boundary instead of attributing it to whichever layer happened
            // to register the flag first.
            read_validation_flag_contents(&pending.flag, "forward pass")?;
        }
        Ok(())
    }
}

thread_local! {
    static DEFERRED_VALIDATION: RefCell<Option<ValidationCollector>> = const { RefCell::new(None) };
}

struct ValidationScopeGuard {
    active: bool,
}

impl ValidationScopeGuard {
    fn begin() -> Result<Self> {
        let nested = DEFERRED_VALIDATION.with(|scope| scope.borrow().is_some());
        if nested {
            candle_core::bail!("nested TurboQuant deferred-validation scopes are not supported")
        }
        let collector = ValidationCollector::new();
        DEFERRED_VALIDATION.with(|scope| *scope.borrow_mut() = Some(collector));
        Ok(Self { active: true })
    }

    fn take(&mut self) -> Result<ValidationCollector> {
        let collector = DEFERRED_VALIDATION
            .with(|scope| scope.borrow_mut().take())
            .ok_or_else(|| {
                candle_core::Error::Msg("TurboQuant deferred-validation scope was lost".into())
            })?;
        self.active = false;
        Ok(collector)
    }
}

impl Drop for ValidationScopeGuard {
    fn drop(&mut self) {
        if self.active {
            // This is the panic/error cleanup path.  Pending flags are dropped
            // without a readback because the caller must discard the pass's
            // unpublished cache state after an error.
            DEFERRED_VALIDATION.with(|scope| {
                let _ = scope.borrow_mut().take();
            });
        }
    }
}

/// Defer TurboQuant validation readbacks until a complete forward pass ends.
///
/// Public cache operations remain strict: outside this scope every operation
/// waits for and checks its own one-word flag.  The stage path may use this
/// helper to turn many per-layer waits into one wait per Metal device.  An
/// error discards pending flags; a successful closure is returned only after
/// all deferred validation words have been checked.
pub(crate) fn with_deferred_validation<T, F>(operation: F) -> anyhow::Result<T>
where
    F: FnOnce() -> anyhow::Result<T>,
{
    let mut guard = ValidationScopeGuard::begin()
        .map_err(|error| anyhow::anyhow!("failed to begin TurboQuant validation scope: {error}"))?;
    let result = operation();
    let collector = guard
        .take()
        .map_err(|error| anyhow::anyhow!("failed to close TurboQuant validation scope: {error}"))?;
    match result {
        Ok(value) => {
            collector.flush().map_err(|error| {
                anyhow::anyhow!("deferred TurboQuant validation failed: {error}")
            })?;
            Ok(value)
        }
        Err(error) => Err(error),
    }
}

fn read_validation_flag_contents(
    flag: &candle_metal_kernels::metal::Buffer,
    operation: &str,
) -> Result<()> {
    if flag.length() < std::mem::size_of::<u32>() {
        candle_core::bail!("TurboQuant {operation} validation flag is empty")
    }
    let contents = flag.contents();
    if contents.is_null() {
        candle_core::bail!("TurboQuant {operation} validation flag has no CPU mapping")
    }
    // The flag is a shared u32 initialized by `with_data` and only touched by
    // the dispatched atomic ORs.  The collector/device wait above establishes
    // visibility before this read.
    let invalid = unsafe { std::ptr::read_volatile(contents.cast::<u32>()) };
    if invalid != 0 {
        candle_core::bail!("TurboQuant {operation} rejected non-finite or invalid input")
    }
    Ok(())
}

fn read_validation_flag(
    flag: Arc<candle_metal_kernels::metal::Buffer>,
    device: &candle_core::MetalDevice,
    operation: &'static str,
) -> Result<()> {
    let deferred = DEFERRED_VALIDATION.with(|scope| -> Result<bool> {
        let mut scope = scope.borrow_mut();
        if let Some(collector) = scope.as_mut() {
            collector.push(flag.clone(), device)?;
            Ok(true)
        } else {
            Ok(false)
        }
    })?;
    if deferred {
        return Ok(());
    }
    device.flush_and_wait_current()?;
    read_validation_flag_contents(&flag, operation)
}

/// Allocate a validation word for a Metal operation. Deferred forward-pass
/// work shares one zero-initialized word per device; standalone cache APIs
/// retain their strict one-operation buffer and readback behavior.
fn validation_flag(
    device: &candle_core::MetalDevice,
    operation: &'static str,
) -> Result<Arc<candle_metal_kernels::metal::Buffer>> {
    DEFERRED_VALIDATION.with(|scope| -> Result<_> {
        let mut scope = scope.borrow_mut();
        if let Some(collector) = scope.as_mut() {
            collector.shared_flag(device)
        } else {
            device
                .new_buffer_builder()
                .with_data(&[0u32])
                .with_label(operation)
                .build()
        }
    })
}

/// In-place variant used by a capacity-backed packed cache.  Quantization
/// writes only the inactive sequence range selected by `output_start`; the
/// caller publishes the new logical length only after dequantization succeeds.
struct QuantizeKvInto {
    heads: u32,
    sequence: u32,
    head_dim: u32,
    bits: u32,
    packed_bytes: u32,
    record_bytes: u32,
    max_norm: f32,
    output_capacity: u32,
    output_start: u32,
    signs: Tensor,
}

impl InplaceOp3 for QuantizeKvInto {
    fn name(&self) -> &'static str {
        "dinf-metal-turboquant-quantize-kv-into"
    }

    fn cpu_fwd(
        &self,
        _destination: &mut CpuStorage,
        _destination_layout: &Layout,
        _key: &CpuStorage,
        _key_layout: &Layout,
        _value: &CpuStorage,
        _value_layout: &Layout,
    ) -> Result<()> {
        candle_core::bail!("TurboQuant Metal in-place quantization requires Metal tensors")
    }

    fn metal_fwd(
        &self,
        destination: &mut MetalStorage,
        destination_layout: &Layout,
        key: &MetalStorage,
        key_layout: &Layout,
        value: &MetalStorage,
        value_layout: &Layout,
    ) -> Result<()> {
        let (sign_storage, signs_layout) = self.signs.storage_and_layout();
        let signs = match &*sign_storage {
            Storage::Metal(signs) => signs,
            _ => candle_core::bail!("TurboQuant Metal signs must be on Metal"),
        };
        if destination.dtype() != DType::U8 {
            candle_core::bail!("TurboQuant Metal quantization destination must be u8")
        }
        validate_contiguous_span(destination, destination_layout, "quantization destination")?;
        for (label, storage, layout) in [("key", key, key_layout), ("value", value, value_layout)] {
            if storage.dtype() != DType::F32 {
                candle_core::bail!("TurboQuant Metal {label} must be f32")
            }
            validate_contiguous_span(storage, layout, label)?;
        }
        validate_contiguous_span(signs, signs_layout, "signs")?;
        if destination.device().id() != key.device().id()
            || destination.device().id() != value.device().id()
            || destination.device().id() != signs.device().id()
        {
            candle_core::bail!("TurboQuant Metal quantization tensors use different devices")
        }
        let (destination_batch, destination_heads, destination_capacity, destination_record) =
            destination_layout.shape().dims4()?;
        let (key_batch, key_heads, key_sequence, key_dim) = key_layout.shape().dims4()?;
        let (value_batch, value_heads, value_sequence, value_dim) = value_layout.shape().dims4()?;
        let (sign_heads, sign_dim) = signs_layout.shape().dims2()?;
        if (destination_batch, destination_heads, destination_record)
            != (2, key_heads, self.record_bytes as usize)
            || destination_heads != self.heads as usize
            || destination_capacity != self.output_capacity as usize
            || (key_batch, key_heads, key_sequence, key_dim)
                != (value_batch, value_heads, value_sequence, value_dim)
            || key_batch != 1
            || key_sequence != self.sequence as usize
            || sign_heads != key_heads
            || sign_dim != key_dim
            || key_dim != self.head_dim as usize
        {
            candle_core::bail!("TurboQuant Metal in-place quantization geometry differs")
        }
        let end = (self.output_start as usize)
            .checked_add(key_sequence)
            .ok_or_else(|| candle_core::Error::Msg("TurboQuant output range overflows".into()))?;
        if end > destination_capacity {
            candle_core::bail!("TurboQuant Metal quantization output range exceeds capacity")
        }
        let vectors_per_side = checked_product(key_heads, key_sequence, "vector count")?;
        checked_kernel_product(vectors_per_side, key_dim, "input elements")?;
        let groups = checked_product(vectors_per_side, 2, "dispatch groups")?;
        checked_u32(groups, "dispatch groups")?;
        checked_u32(
            checked_product(
                checked_product(2, destination_heads, "destination heads")?,
                checked_product(
                    destination_capacity,
                    self.record_bytes as usize,
                    "destination bytes",
                )?,
                "destination elements",
            )?,
            "destination elements",
        )?;

        let device = key.device();
        let invalid = validation_flag(device, "dinf-turboquant-quantize-into-invalid")?;
        let pipeline = pipelines(device)?.quantize;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-turboquant-quantize-kv-into");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(key.buffer()),
            key_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            1,
            Some(value.buffer()),
            value_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            2,
            Some(signs.buffer()),
            signs_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(
            3,
            Some(destination.buffer()),
            destination_layout.start_offset() * DType::U8.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(4, Some(&invalid), 0);
        encoder_ref.set_bytes(5, &self.heads);
        encoder_ref.set_bytes(6, &self.sequence);
        encoder_ref.set_bytes(7, &self.head_dim);
        encoder_ref.set_bytes(8, &self.bits);
        encoder_ref.set_bytes(9, &self.packed_bytes);
        encoder_ref.set_bytes(10, &self.record_bytes);
        encoder_ref.set_bytes(11, &self.max_norm);
        encoder_ref.set_bytes(12, &self.output_capacity);
        encoder_ref.set_bytes(13, &self.output_start);
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: groups,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: self.head_dim as usize,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        read_validation_flag(invalid, device, "in-place quantization")
    }
}

struct DequantizeKv {
    heads: u32,
    sequence: u32,
    head_dim: u32,
    bits: u32,
    packed_bytes: u32,
    record_bytes: u32,
    output_dtype: DType,
}

impl CustomOp3 for DequantizeKv {
    fn name(&self) -> &'static str {
        "dinf-metal-turboquant-dequantize-kv"
    }

    fn cpu_fwd(
        &self,
        _key: &CpuStorage,
        _key_layout: &Layout,
        _value: &CpuStorage,
        _value_layout: &Layout,
        _signs: &CpuStorage,
        _signs_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("TurboQuant Metal dequantization requires Metal tensors")
    }

    fn metal_fwd(
        &self,
        key: &MetalStorage,
        key_layout: &Layout,
        value: &MetalStorage,
        value_layout: &Layout,
        signs: &MetalStorage,
        signs_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        validate_contiguous_span(signs, signs_layout, "signs")?;
        if key.dtype() != DType::U8
            || value.dtype() != DType::U8
            || signs.dtype() != DType::F32
            || key.device().id() != value.device().id()
            || key.device().id() != signs.device().id()
        {
            candle_core::bail!("TurboQuant Metal dequantization dtypes/devices differ")
        }
        if !matches!(self.output_dtype, DType::F16 | DType::F32) {
            candle_core::bail!("TurboQuant Metal output must be f16 or f32")
        }
        let (key_batch, key_heads, key_sequence, key_record) = key_layout.shape().dims4()?;
        let (value_batch, value_heads, value_sequence, value_record) =
            value_layout.shape().dims4()?;
        let (sign_heads, sign_dim) = signs_layout.shape().dims2()?;
        if (key_batch, key_heads, key_sequence, key_record)
            != (value_batch, value_heads, value_sequence, value_record)
            || key_batch != 1
            || key_heads != self.heads as usize
            || key_sequence != self.sequence as usize
            || key_record != self.record_bytes as usize
            || sign_heads != key_heads
            || sign_dim != self.head_dim as usize
        {
            candle_core::bail!("TurboQuant Metal packed cache geometry differs")
        }
        let key_head_stride = validate_packed_cache_layout(
            key,
            key_layout,
            key_heads,
            key_sequence,
            key_record,
            "packed key",
        )?;
        let value_head_stride = validate_packed_cache_layout(
            value,
            value_layout,
            value_heads,
            value_sequence,
            value_record,
            "packed value",
        )?;
        let vectors_per_side = checked_product(key_heads, key_sequence, "vector count")?;
        let groups = checked_product(vectors_per_side, 2, "dispatch groups")?;
        let output_elements = checked_product(groups, self.head_dim as usize, "output")?;
        checked_u32(groups, "dispatch groups")?;
        checked_u32(output_elements, "output elements")?;

        let device = key.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, self.output_dtype)
            .with_label("dinf-turboquant-dequantized-kv")
            .build()?;
        let invalid = validation_flag(device, "dinf-turboquant-dequantize-invalid")?;
        let pipelines = pipelines(device)?;
        let pipeline = match self.output_dtype {
            DType::F16 => pipelines.dequantize_f16,
            DType::F32 => pipelines.dequantize_f32,
            _ => unreachable!(),
        };
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-turboquant-dequantize-kv");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(key.buffer()),
            key_layout.start_offset() * DType::U8.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            1,
            Some(value.buffer()),
            value_layout.start_offset() * DType::U8.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            2,
            Some(signs.buffer()),
            signs_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(3, Some(&output), 0);
        encoder_ref.set_output_buffer(4, Some(&invalid), 0);
        encoder_ref.set_bytes(5, &self.heads);
        encoder_ref.set_bytes(6, &self.sequence);
        encoder_ref.set_bytes(7, &self.head_dim);
        encoder_ref.set_bytes(8, &self.bits);
        encoder_ref.set_bytes(9, &self.packed_bytes);
        encoder_ref.set_bytes(10, &self.record_bytes);
        encoder_ref.set_bytes(11, &key_head_stride);
        encoder_ref.set_bytes(12, &value_head_stride);
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: groups,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: self.head_dim as usize,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        read_validation_flag(invalid, device, "dequantization")?;
        Ok((
            MetalStorage::new(output, device.clone(), output_elements, self.output_dtype),
            Shape::from((2, key_heads, key_sequence, self.head_dim as usize)),
        ))
    }
}

struct DequantizeRotatedKv {
    heads: u32,
    sequence: u32,
    head_dim: u32,
    bits: u32,
    packed_bytes: u32,
    record_bytes: u32,
    output_dtype: DType,
}

impl CustomOp2 for DequantizeRotatedKv {
    fn name(&self) -> &'static str {
        "dinf-metal-turboquant-dequantize-rotated-kv"
    }

    fn cpu_fwd(
        &self,
        _key: &CpuStorage,
        _key_layout: &Layout,
        _value: &CpuStorage,
        _value_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("TurboQuant Metal rotated dequantization requires Metal tensors")
    }

    fn metal_fwd(
        &self,
        key: &MetalStorage,
        key_layout: &Layout,
        value: &MetalStorage,
        value_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if key.dtype() != DType::U8 || value.dtype() != DType::U8 {
            candle_core::bail!("TurboQuant Metal rotated packed K/V must be u8")
        }
        if key.device().id() != value.device().id() {
            candle_core::bail!("TurboQuant Metal rotated packed K/V use different devices")
        }
        if !matches!(self.output_dtype, DType::F16 | DType::F32) {
            candle_core::bail!("TurboQuant Metal rotated output must be f16 or f32")
        }
        let (key_batch, key_heads, key_sequence, key_record) = key_layout.shape().dims4()?;
        let (value_batch, value_heads, value_sequence, value_record) =
            value_layout.shape().dims4()?;
        if (key_batch, key_heads, key_sequence, key_record)
            != (value_batch, value_heads, value_sequence, value_record)
            || key_batch != 1
            || key_heads != self.heads as usize
            || key_sequence != self.sequence as usize
            || key_record != self.record_bytes as usize
        {
            candle_core::bail!("TurboQuant Metal rotated packed cache geometry differs")
        }
        let key_head_stride = validate_packed_cache_layout(
            key,
            key_layout,
            key_heads,
            key_sequence,
            key_record,
            "rotated packed key",
        )?;
        let value_head_stride = validate_packed_cache_layout(
            value,
            value_layout,
            value_heads,
            value_sequence,
            value_record,
            "rotated packed value",
        )?;
        let vectors_per_side = checked_product(key_heads, key_sequence, "rotated vector count")?;
        let groups = checked_product(vectors_per_side, 2, "rotated dispatch groups")?;
        let output_elements = checked_product(groups, self.head_dim as usize, "rotated output")?;
        checked_u32(groups, "rotated dispatch groups")?;
        checked_u32(output_elements, "rotated output elements")?;

        let device = key.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, self.output_dtype)
            .with_label("dinf-turboquant-dequantized-rotated-kv")
            .build()?;
        let invalid = validation_flag(device, "dinf-turboquant-dequantize-rotated-invalid")?;
        let pipelines = pipelines(device)?;
        let vectorized = self.bits == 8
            && self.output_dtype == DType::F16
            && self.head_dim.is_multiple_of(8)
            && key_layout.start_offset().is_multiple_of(4)
            && value_layout.start_offset().is_multiple_of(4)
            && std::env::var_os("DINF_DISABLE_TURBOQUANT_VECTOR_DEQUANT").is_none();
        let pipeline = match self.output_dtype {
            DType::F16 if vectorized => pipelines.dequantize_rotated8_f16x4,
            DType::F16 => pipelines.dequantize_rotated_f16,
            DType::F32 => pipelines.dequantize_rotated_f32,
            _ => unreachable!(),
        };
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-turboquant-dequantize-rotated-kv");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(key.buffer()),
            key_layout.start_offset() * DType::U8.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            1,
            Some(value.buffer()),
            value_layout.start_offset() * DType::U8.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(2, Some(&output), 0);
        encoder_ref.set_output_buffer(3, Some(&invalid), 0);
        encoder_ref.set_bytes(4, &self.heads);
        encoder_ref.set_bytes(5, &self.sequence);
        encoder_ref.set_bytes(6, &self.head_dim);
        encoder_ref.set_bytes(7, &self.bits);
        encoder_ref.set_bytes(8, &self.packed_bytes);
        encoder_ref.set_bytes(9, &self.record_bytes);
        encoder_ref.set_bytes(10, &key_head_stride);
        encoder_ref.set_bytes(11, &value_head_stride);
        // Neither path needs WHT scratch or barriers. KV8 uses coordinates,
        // tokens, and heads as separate axes to avoid integer division in
        // each lane; the generic fallback retains its flat elementwise grid.
        if vectorized {
            let width = self.head_dim as usize / 8;
            let threads_x = width.min(32);
            let threads_y = pipeline.max_total_threads_per_threadgroup().min(256) / threads_x;
            if threads_y == 0 {
                candle_core::bail!("TurboQuant vector reconstruction exceeds the threadgroup limit")
            }
            encoder_ref.dispatch_threads(
                objc2_metal::MTLSize {
                    width,
                    height: key_sequence,
                    depth: key_heads * 2,
                },
                objc2_metal::MTLSize {
                    width: threads_x,
                    height: threads_y,
                    depth: 1,
                },
            );
        } else {
            encoder_ref.dispatch_threads(
                objc2_metal::MTLSize {
                    width: output_elements,
                    height: 1,
                    depth: 1,
                },
                candle_metal_kernels::utils::get_block_dims(output_elements.min(256), 1, 1),
            );
        }
        drop(encoder);
        read_validation_flag(invalid, device, "rotated dequantization")?;
        Ok((
            MetalStorage::new(output, device.clone(), output_elements, self.output_dtype),
            Shape::from((2, key_heads, key_sequence, self.head_dim as usize)),
        ))
    }
}

struct RotateAttention {
    heads: u32,
    sequence: u32,
    head_dim: u32,
    inverse: u32,
    output_dtype: DType,
}

impl CustomOp2 for RotateAttention {
    fn name(&self) -> &'static str {
        "dinf-metal-turboquant-rotate-attention"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _signs: &CpuStorage,
        _signs_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("TurboQuant Metal attention rotation requires Metal tensors")
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        input_layout: &Layout,
        signs: &MetalStorage,
        signs_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if !matches!(
            (input.dtype(), self.output_dtype),
            (DType::F16, DType::F16) | (DType::F32, DType::F32) | (DType::F32, DType::F16)
        ) || signs.dtype() != DType::F32
        {
            candle_core::bail!("TurboQuant Metal attention rotation dtypes are unsupported")
        }
        if input.device().id() != signs.device().id() {
            candle_core::bail!("TurboQuant Metal attention rotation uses different devices")
        }
        validate_contiguous_span(input, input_layout, "attention rotation input")?;
        validate_contiguous_span(signs, signs_layout, "attention rotation signs")?;
        let (batch, heads, sequence, head_dim) = input_layout.shape().dims4()?;
        let (sign_heads, sign_dim) = signs_layout.shape().dims2()?;
        if batch != 1
            || heads != self.heads as usize
            || sequence != self.sequence as usize
            || head_dim != self.head_dim as usize
            || sign_heads == 0
            || sign_dim != head_dim
        {
            candle_core::bail!("TurboQuant Metal attention rotation geometry differs")
        }
        let groups = checked_product(heads, sequence, "rotation dispatch groups")?;
        let output_elements = checked_product(groups, head_dim, "rotation output")?;
        checked_u32(groups, "rotation dispatch groups")?;
        checked_u32(output_elements, "rotation output elements")?;

        let device = input.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, self.output_dtype)
            .with_label("dinf-turboquant-rotated-attention")
            .build()?;
        let invalid = validation_flag(device, "dinf-turboquant-rotation-invalid")?;
        let pipelines = pipelines(device)?;
        let pipeline = match (input.dtype(), self.output_dtype) {
            (DType::F16, DType::F16) => pipelines.rotate_f16,
            (DType::F32, DType::F32) => pipelines.rotate_f32,
            (DType::F32, DType::F16) => pipelines.rotate_query_f32_f16,
            _ => unreachable!(),
        };
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-turboquant-rotate-attention");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(input.buffer()),
            input_layout.start_offset() * input.dtype().size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            1,
            Some(signs.buffer()),
            signs_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(2, Some(&output), 0);
        encoder_ref.set_output_buffer(3, Some(&invalid), 0);
        encoder_ref.set_bytes(4, &self.heads);
        encoder_ref.set_bytes(5, &self.sequence);
        encoder_ref.set_bytes(6, &self.head_dim);
        encoder_ref.set_bytes(7, &self.inverse);
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: groups,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: self.head_dim as usize,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        read_validation_flag(invalid, device, "attention rotation")?;
        Ok((
            MetalStorage::new(output, device.clone(), output_elements, self.output_dtype),
            Shape::from((1, heads, sequence, head_dim)),
        ))
    }
}

fn validate_signs(signs: &Tensor, heads: usize, head_dim: usize) -> Result<()> {
    if !signs.device().is_metal() || signs.dtype() != DType::F32 || !signs.is_contiguous() {
        candle_core::bail!("TurboQuant Metal signs must be contiguous f32 on Metal")
    }
    if signs.dims() != [heads, head_dim] {
        candle_core::bail!("TurboQuant Metal signs must have shape [heads, head_dim]")
    }
    Ok(())
}

fn normalize_input(input: &Tensor, label: &str) -> Result<Tensor> {
    if !input.device().is_metal() {
        candle_core::bail!("TurboQuant Metal {label} input must be on Metal")
    }
    if !matches!(input.dtype(), DType::F16 | DType::BF16 | DType::F32) {
        candle_core::bail!("TurboQuant Metal {label} input must be f16, bf16, or f32")
    }
    input.to_dtype(DType::F32)?.contiguous()
}

fn tensors_share_storage(left: &Tensor, right: &Tensor) -> bool {
    let (left_storage, _) = left.storage_and_layout();
    let (right_storage, _) = right.storage_and_layout();
    std::ptr::eq(&*left_storage, &*right_storage)
}

/// Quantize a K/V append directly into an inactive range of a preallocated
/// packed destination.  `destination` has shape
/// `[2, heads, capacity, record_bytes]`; `offset` is measured in token slots.
/// Only the new range is written, so a caller can publish its logical length
/// after the operation and leave the prior cache state intact on failure.
pub(crate) fn quantize_kv_into(
    destination: &Tensor,
    key: &Tensor,
    value: &Tensor,
    signs: &Tensor,
    bits: u8,
    output_dtype: DType,
    offset: usize,
) -> Result<()> {
    if !destination.device().is_metal() {
        candle_core::bail!("TurboQuant Metal quantization destination must be on Metal")
    }
    if destination.dtype() != DType::U8 || !destination.is_contiguous() {
        candle_core::bail!("TurboQuant Metal quantization destination must be contiguous u8")
    }
    if !matches!(output_dtype, DType::F16 | DType::F32) {
        candle_core::bail!("TurboQuant Metal output must be f16 or f32")
    }
    let key = normalize_input(key, "key")?;
    let value = normalize_input(value, "value")?;
    if !destination.device().same_device(key.device())
        || !destination.device().same_device(value.device())
        || !destination.device().same_device(signs.device())
    {
        candle_core::bail!("TurboQuant Metal quantization tensors use different devices")
    }
    if tensors_share_storage(destination, &key)
        || tensors_share_storage(destination, &value)
        || tensors_share_storage(destination, signs)
    {
        candle_core::bail!("TurboQuant Metal quantization destination aliases an input")
    }
    let (batch, heads, sequence, head_dim) = key.dims4()?;
    let value_shape = value.dims4()?;
    if value_shape != (batch, heads, sequence, head_dim) || batch != 1 || sequence == 0 {
        candle_core::bail!("TurboQuant Metal K/V input geometry differs")
    }
    let (destination_batch, destination_heads, capacity, record_bytes) = destination.dims4()?;
    let (packed_bytes, expected_record_bytes) = record_geometry(bits, head_dim)?;
    if destination_batch != 2
        || destination_heads != heads
        || expected_record_bytes != record_bytes
        || offset
            .checked_add(sequence)
            .ok_or_else(|| candle_core::Error::Msg("TurboQuant output range overflows".into()))?
            > capacity
    {
        candle_core::bail!("TurboQuant Metal quantization destination geometry differs")
    }
    validate_signs(signs, heads, head_dim)?;
    checked_kernel_product(
        checked_product(2, heads, "destination sides")?,
        checked_product(capacity, record_bytes, "destination records")?,
        "destination elements",
    )?;
    let operation = QuantizeKvInto {
        heads: checked_u32(heads, "heads")?,
        sequence: checked_u32(sequence, "sequence")?,
        head_dim: checked_u32(head_dim, "head dimension")?,
        bits: u32::from(bits),
        packed_bytes: checked_u32(packed_bytes, "packed bytes")?,
        record_bytes: checked_u32(record_bytes, "record bytes")?,
        max_norm: if output_dtype == DType::F16 {
            65_504.0 / 8.0
        } else {
            f32::MAX / 8.0
        },
        output_capacity: checked_u32(capacity, "output capacity")?,
        output_start: checked_u32(offset, "output start")?,
        signs: signs.clone(),
    };
    destination.inplace_op3(&key, &value, &operation)
}

/// Dequantize a packed K/V cache on Metal.
///
/// `output_dtype` may be `F16` for native fused attention or `F32` for the
/// portable attention paths.  Both outputs have shape `[1, heads, sequence,
/// head_dim]`.  The packed tensors are never copied to the CPU.
pub(crate) fn dequantize_kv(
    key_packed: &Tensor,
    value_packed: &Tensor,
    signs: &Tensor,
    bits: u8,
    output_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    if !key_packed.device().is_metal() || !value_packed.device().is_metal() {
        candle_core::bail!("TurboQuant Metal packed cache must be on Metal")
    }
    if !key_packed.device().same_device(value_packed.device()) {
        candle_core::bail!("TurboQuant Metal packed K/V caches use different devices")
    }
    let key_dims = key_packed.dims();
    let value_dims = value_packed.dims();
    if key_dims.len() != 4 || value_dims.len() != 4 {
        candle_core::bail!("TurboQuant Metal packed cache must have rank four")
    }
    let (batch, heads, sequence, record_bytes) = key_packed.dims4()?;
    if value_dims != key_dims || batch != 1 || sequence == 0 {
        candle_core::bail!("TurboQuant Metal packed K/V cache geometry differs")
    }
    let (packed_bytes, expected_record_bytes) = record_geometry(bits, signs.dim(1)?)?;
    if record_bytes != expected_record_bytes {
        candle_core::bail!("TurboQuant Metal packed record width differs")
    }
    validate_signs(signs, heads, signs.dim(1)?)?;
    let head_dim = signs.dim(1)?;
    let groups = checked_product(
        checked_product(heads, sequence, "vector count")?,
        2,
        "groups",
    )?;
    let output_elements = checked_product(groups, head_dim, "output")?;
    let operation = DequantizeKv {
        heads: checked_u32(heads, "heads")?,
        sequence: checked_u32(sequence, "sequence")?,
        head_dim: checked_u32(head_dim, "head dimension")?,
        bits: u32::from(bits),
        packed_bytes: checked_u32(packed_bytes, "packed bytes")?,
        record_bytes: checked_u32(record_bytes, "record bytes")?,
        output_dtype,
    };
    let dequantized = key_packed.apply_op3_no_bwd(value_packed, signs, &operation)?;
    let key = dequantized.narrow(0, 0, 1)?;
    let value = dequantized.narrow(0, 1, 1)?;
    debug_assert_eq!(dequantized.elem_count(), output_elements);
    Ok((key, value))
}

/// Dequantize packed K/V records into TurboQuant's rotated attention basis.
///
/// The output is native F16 and contains `centroid * norm / sqrt(head_dim)`;
/// it intentionally omits the inverse WHT and random-sign undo.  Queries and
/// the final attention output are rotated with [`rotate_attention`] instead.
pub(crate) fn dequantize_kv_rotated(
    key_packed: &Tensor,
    value_packed: &Tensor,
    signs: &Tensor,
    bits: u8,
) -> Result<(Tensor, Tensor)> {
    if !key_packed.device().is_metal() || !value_packed.device().is_metal() {
        candle_core::bail!("TurboQuant Metal rotated cache must be on Metal")
    }
    if !key_packed.device().same_device(value_packed.device()) {
        candle_core::bail!("TurboQuant Metal rotated K/V caches use different devices")
    }
    let key_dims = key_packed.dims();
    let value_dims = value_packed.dims();
    if key_dims.len() != 4 || value_dims.len() != 4 {
        candle_core::bail!("TurboQuant Metal rotated cache must have rank four")
    }
    let (batch, heads, sequence, record_bytes) = key_packed.dims4()?;
    if value_dims != key_dims || batch != 1 || sequence == 0 {
        candle_core::bail!("TurboQuant Metal rotated K/V geometry differs")
    }
    let head_dim = signs.dim(1)?;
    let (packed_bytes, expected_record_bytes) = record_geometry(bits, head_dim)?;
    if record_bytes != expected_record_bytes {
        candle_core::bail!("TurboQuant Metal rotated record width differs")
    }
    // The rotation basis is shared by all KV heads in the rotated path, so
    // callers may provide either one `[1, head_dim]` row or an explicit row
    // for every head.  The normal dequantizer still requires one row per
    // head because it applies the sign during reconstruction.
    if !signs.device().is_metal() || signs.dtype() != DType::F32 || !signs.is_contiguous() {
        candle_core::bail!("TurboQuant Metal rotated signs must be contiguous f32 on Metal")
    }
    let sign_heads = signs.dims().first().copied().unwrap_or(0);
    if (sign_heads != 1 && sign_heads != heads) || signs.dims() != [sign_heads, head_dim] {
        candle_core::bail!(
            "TurboQuant Metal rotated signs must have shape [1, head_dim] or [heads, head_dim]"
        )
    }
    let groups = checked_product(
        checked_product(heads, sequence, "rotated vector count")?,
        2,
        "rotated groups",
    )?;
    let output_elements = checked_product(groups, head_dim, "rotated output")?;
    let operation = DequantizeRotatedKv {
        heads: checked_u32(heads, "rotated heads")?,
        sequence: checked_u32(sequence, "rotated sequence")?,
        head_dim: checked_u32(head_dim, "rotated head dimension")?,
        bits: u32::from(bits),
        packed_bytes: checked_u32(packed_bytes, "rotated packed bytes")?,
        record_bytes: checked_u32(record_bytes, "rotated record bytes")?,
        output_dtype: DType::F16,
    };
    let rotated = key_packed.apply_op2_no_bwd(value_packed, &operation)?;
    let key = rotated.narrow(0, 0, 1)?;
    let value = rotated.narrow(0, 1, 1)?;
    debug_assert_eq!(rotated.elem_count(), output_elements);
    Ok((key, value))
}

/// Apply or undo TurboQuant's orthogonal attention rotation on Metal.
///
/// Forward mode computes `H(signs * x) / sqrt(d)`; inverse mode computes
/// `signs * H(x) / sqrt(d)`.  `signs` may contain one row or repeated rows;
/// the first row is shared across all input heads.
pub(crate) fn rotate_attention(input: &Tensor, signs: &Tensor, inverse: bool) -> Result<Tensor> {
    rotate_attention_impl(input, signs, inverse, input.dtype())
}

/// Preserve the existing F32 rotation then F16 query-rounding semantics in
/// one dispatch, so native attention does not need a separate cast kernel.
pub(crate) fn rotate_attention_native_query(input: &Tensor, signs: &Tensor) -> Result<Tensor> {
    rotate_attention_impl(input, signs, false, DType::F16)
}

fn rotate_attention_impl(
    input: &Tensor,
    signs: &Tensor,
    inverse: bool,
    output_dtype: DType,
) -> Result<Tensor> {
    if !input.device().is_metal() || !signs.device().is_metal() {
        candle_core::bail!("TurboQuant Metal attention rotation requires Metal tensors")
    }
    if !input.device().same_device(signs.device()) {
        candle_core::bail!("TurboQuant Metal attention rotation uses different devices")
    }
    if !matches!(input.dtype(), DType::F16 | DType::F32) {
        candle_core::bail!("TurboQuant Metal attention rotation input must be f16 or f32")
    }
    let input = input.contiguous()?;
    let signs = signs.contiguous()?;
    let (batch, heads, sequence, head_dim) = input.dims4()?;
    if batch != 1 || sequence == 0 {
        candle_core::bail!("TurboQuant Metal attention rotation requires [1, heads, seq, dim]")
    }
    if !head_dim.is_power_of_two() || !(8..=MAX_METAL_HEAD_DIM).contains(&head_dim) {
        candle_core::bail!(
            "TurboQuant Metal attention rotation head dimension must be a power of two in 8..={MAX_METAL_HEAD_DIM}"
        )
    }
    let (sign_heads, sign_dim) = signs.dims2()?;
    if sign_heads == 0 || sign_dim != head_dim {
        candle_core::bail!("TurboQuant Metal attention rotation signs have wrong shape")
    }
    let groups = checked_product(heads, sequence, "rotation groups")?;
    let output_elements = checked_product(groups, head_dim, "rotation output")?;
    let operation = RotateAttention {
        heads: checked_u32(heads, "rotation heads")?,
        sequence: checked_u32(sequence, "rotation sequence")?,
        head_dim: checked_u32(head_dim, "rotation head dimension")?,
        inverse: u32::from(inverse),
        output_dtype,
    };
    let output = input.apply_op2_no_bwd(&signs, &operation)?;
    debug_assert_eq!(output.elem_count(), output_elements);
    Ok(output)
}

/// Compile the TurboQuant Metal kernels during model warmup.
pub(crate) fn precompile(device: &Device) -> anyhow::Result<()> {
    if !device.is_metal() || std::env::var_os("DINF_DISABLE_TURBOQUANT_METAL").is_some() {
        return Ok(());
    }
    let _ = pipelines(device.as_metal_device()?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use candle_core::{DType, Tensor};

    use super::{
        DEFERRED_VALIDATION, checked_kernel_product, rotate_attention, validation_flag,
        with_deferred_validation,
    };

    #[test]
    fn kernel_index_spans_reject_overflow_before_gpu_allocation() {
        assert_eq!(
            checked_kernel_product(u32::MAX as usize, 1, "test").unwrap(),
            u32::MAX as usize
        );
        assert!(checked_kernel_product(u32::MAX as usize, 2, "test").is_err());
        assert!(checked_kernel_product(usize::MAX, 2, "test").is_err());
        // A packed 3-bit output still fits while the dense source exceeds u32.
        let vectors = 1usize << 23;
        assert!(checked_kernel_product(vectors * 2, 196, "packed").is_ok());
        assert!(checked_kernel_product(vectors, 512, "dense").is_err());
        // Each input can fit separately while their concatenated output cannot.
        assert!(checked_kernel_product(1usize << 24, 132, "single cache").is_ok());
        assert!(checked_kernel_product(1usize << 25, 132, "appended cache").is_err());
    }

    #[test]
    fn deferred_scope_rejects_nesting_and_restores_after_error() {
        let nested = with_deferred_validation(|| {
            let nested = with_deferred_validation(|| Ok::<_, anyhow::Error>(()));
            assert!(nested.is_err());
            Ok::<_, anyhow::Error>(())
        });
        assert!(nested.is_ok());

        let failed = with_deferred_validation(|| Err::<(), _>(anyhow::anyhow!("expected")));
        assert!(failed.is_err());
        assert!(with_deferred_validation(|| Ok::<_, anyhow::Error>(())).is_ok());
    }

    #[test]
    fn deferred_scope_restores_after_panic() {
        let panicked = std::panic::catch_unwind(|| {
            let _: anyhow::Result<()> = with_deferred_validation(|| panic!("expected"));
        });
        assert!(panicked.is_err());
        assert!(with_deferred_validation(|| Ok::<_, anyhow::Error>(())).is_ok());
    }

    #[test]
    fn deferred_metal_scope_reuses_one_validation_flag() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let metal = device.as_metal_device()?;
        let input = Tensor::ones((1, 1, 1, 8), DType::F32, &device)?;
        let signs = Tensor::ones((1, 8), DType::F32, &device)?;

        let result = with_deferred_validation(|| {
            let first = validation_flag(metal, "test-first")?;
            let second = validation_flag(metal, "test-second")?;
            assert!(Arc::ptr_eq(&first, &second));
            DEFERRED_VALIDATION.with(|scope| {
                assert_eq!(scope.borrow().as_ref().unwrap().flags.len(), 1);
            });

            rotate_attention(&input, &signs, false)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            rotate_attention(&input, &signs, true)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            DEFERRED_VALIDATION.with(|scope| {
                assert_eq!(scope.borrow().as_ref().unwrap().flags.len(), 1);
            });
            Ok(())
        });
        assert!(result.is_ok(), "shared validation scope failed: {result:?}");
        Ok(())
    }

    #[test]
    fn deferred_metal_validation_poison_is_scope_wide_and_not_sticky() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let valid = Tensor::ones((1, 1, 1, 8), DType::F32, &device)?;
        let signs = Tensor::ones((1, 8), DType::F32, &device)?;
        let invalid = Tensor::from_vec(vec![f32::NAN; 8], (1, 1, 1, 8), &device)?;

        let run = |first: &Tensor, second: &Tensor| {
            with_deferred_validation(|| {
                rotate_attention(first, &signs, false)
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                rotate_attention(second, &signs, false)
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                Ok::<_, anyhow::Error>(())
            })
        };

        // Once an invalid kernel sets the shared atomic word, a later valid
        // kernel cannot clear it before the scope's single readback.
        assert!(run(&invalid, &valid).is_err());
        // A later invalid kernel also remains covered by the same poisoned
        // scope rather than receiving a fresh flag.
        assert!(run(&invalid, &invalid).is_err());
        assert!(run(&valid, &invalid).is_err());

        // The failed scopes dropped their shared buffers; a new scope starts
        // from a fresh zeroed word and accepts valid work.
        assert!(run(&valid, &valid).is_ok());
        Ok(())
    }

    #[test]
    fn standalone_metal_validation_still_rejects_immediately() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let invalid = Tensor::from_vec(vec![f32::NAN; 8], (1, 1, 1, 8), &device)?;
        let signs = Tensor::ones((1, 8), DType::F32, &device)?;
        assert!(rotate_attention(&invalid, &signs, false).is_err());
        Ok(())
    }
}
