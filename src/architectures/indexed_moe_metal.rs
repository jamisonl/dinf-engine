use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, CustomOp3, DType, Layout, MetalStorage, Result, Shape, Tensor,
};
use candle_metal_kernels::metal::{ComputePipeline, Library};
use candle_metal_kernels::source::Source;
use objc2_metal::{MTLCompileOptions, MTLDevice, MTLGPUFamily, MTLLanguageVersion};

const MOE_EPILOGUE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void dinf_silu_mul_f32(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &elements [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < elements) {
    const float value = gate[gid];
    out[gid] = (value / (1.0f + exp(-value))) * up[gid];
  }
}

// Candle's tanh-approximation GELU (`gelu` in unary.metal) times `up`.
kernel void dinf_gelu_mul_f32(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &elements [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < elements) {
    const float x = gate[gid];
    float activated;
    if (x > 5.0f) {
      activated = x;
    } else {
      const float x_cube = (x * x) * x;
      const float alpha = x + 0.044715f * x_cube;
      const float beta = float(M_2_SQRTPI_F * M_SQRT1_2_F) * alpha;
      activated = 0.5f * x * (1.0f + precise::tanh(beta));
    }
    out[gid] = activated * up[gid];
  }
}

kernel void dinf_silu_mul_clamped_f32(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &elements [[buffer(3)]],
    constant float &limit [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < elements) {
    const float gate_value = min(gate[gid], limit);
    const float up_value = clamp(up[gid], -limit, limit);
    out[gid] = (gate_value / (1.0f + exp(-gate_value))) * up_value;
  }
}

kernel void dinf_swiglu_f32(
    device const float *gate_up [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &intermediate_dim [[buffer(2)]],
    constant uint &elements [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < elements) {
    const uint row = gid / intermediate_dim;
    const uint column = gid - row * intermediate_dim;
    const uint row_offset = row * intermediate_dim * 2;
    const float gate = gate_up[row_offset + column];
    const float up = gate_up[row_offset + intermediate_dim + column];
    volatile float activated = gate / (1.0f + exp(-gate));
    out[gid] = activated * up;
  }
}

// Packed `[gate | up]` rows: Candle's tanh-approximation GELU of the gate
// half times the up half (see dinf_gelu_mul_f32).
kernel void dinf_geglu_f32(
    device const float *gate_up [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &intermediate_dim [[buffer(2)]],
    constant uint &elements [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < elements) {
    const uint row = gid / intermediate_dim;
    const uint column = gid - row * intermediate_dim;
    const uint row_offset = row * intermediate_dim * 2;
    const float x = gate_up[row_offset + column];
    float activated;
    if (x > 5.0f) {
      activated = x;
    } else {
      const float x_cube = (x * x) * x;
      const float alpha = x + 0.044715f * x_cube;
      const float beta = float(M_2_SQRTPI_F * M_SQRT1_2_F) * alpha;
      activated = 0.5f * x * (1.0f + precise::tanh(beta));
    }
    out[gid] = activated * gate_up[row_offset + intermediate_dim + column];
  }
}

kernel void dinf_weighted_route_sum_f32(
    device const float *routes [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &top_k [[buffer(3)]],
    constant uint &hidden_dim [[buffer(4)]],
    constant uint &elements [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= elements) {
    return;
  }
  const uint token = gid / hidden_dim;
  const uint column = gid - token * hidden_dim;
  float sum = 0.0f;
  for (uint route = 0; route < top_k; ++route) {
    sum += routes[(token * top_k + route) * hidden_dim + column]
        * weights[token * top_k + route];
  }
  out[gid] = sum;
}

// dinf_weighted_route_sum_f32 with each route weight times its expert's
// output scale, `scales[ids[route]]`.
kernel void dinf_expert_scaled_route_sum_f32(
    device const float *routes [[buffer(0)]],
    device const uint *weights [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &top_k [[buffer(3)]],
    constant uint &hidden_dim [[buffer(4)]],
    constant uint &elements [[buffer(5)]],
    device const uint *ids [[buffer(6)]],
    device const float *scales [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= elements) {
    return;
  }
  const uint token = gid / hidden_dim;
  const uint column = gid - token * hidden_dim;
  float sum = 0.0f;
  for (uint route = 0; route < top_k; ++route) {
    const uint index = token * top_k + route;
    // Weights are f32, read as bits so u32 route tensors can pass them too.
    sum += routes[index * hidden_dim + column]
        * (as_type<float>(weights[index]) * scales[ids[index]]);
  }
  out[gid] = sum;
}

// V4's old path rounds the multiply before adding each route to +0.
// Keep this separate from the existing models' fast weighted sum.
kernel void dinf_ordered_weighted_route_sum_f32(
    device const float *routes [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &top_k [[buffer(3)]],
    constant uint &hidden_dim [[buffer(4)]],
    constant uint &elements [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= elements) return;
  const uint token = gid / hidden_dim;
  const uint column = gid - token * hidden_dim;
  float sum = 0.0f;
  for (uint route = 0; route < top_k; ++route) {
    volatile float weighted = routes[(token * top_k + route) * hidden_dim + column]
        * weights[token * top_k + route];
    sum = sum + weighted;
  }
  out[gid] = sum;
}

kernel void dinf_normalized_route_weights_f32(
    device const float *probabilities [[buffer(0)]],
    device const uint *expert_ids [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &expert_count [[buffer(3)]],
    constant uint &top_k [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    uint token [[thread_position_in_grid]]) {
  if (token >= tokens) {
    return;
  }
  // Match dinf_sanitize_moe_ids without allocating an intermediate ID
  // tensor or adding another command-buffer dispatch.  Since top_k <= 256,
  // a compact bitset covers every deterministic replacement ID.  Valid IDs
  // above 255 can only collide with an earlier unchanged raw ID, so they are
  // checked directly against the input row.
  uint used_low[8] = {0u};
  const uint row_offset = token * top_k;
  // Use a normal accumulator with reassociation disabled at compile time.
  // The volatile loop produced a floor denominator for finite positive rows
  // under concurrent GPU load on M5 Max.
  float weight_sum = 0.0f;
  for (uint route = 0; route < top_k; ++route) {
    const uint offset = row_offset + route;
    uint candidate = expert_ids[offset];
    bool used = candidate >= expert_count;
    if (!used && candidate < 256u) {
      const uint word = candidate >> 5;
      const uint mask = 1u << (candidate & 31u);
      used = (used_low[word] & mask) != 0u;
    } else if (!used) {
      for (uint prior = 0; prior < route && !used; ++prior) {
        used = expert_ids[row_offset + prior] == candidate;
      }
    }
    if (used) {
      for (uint replacement = 0; replacement < top_k; ++replacement) {
        const uint word = replacement >> 5;
        const uint mask = 1u << (replacement & 31u);
        if ((used_low[word] & mask) == 0u) {
          candidate = replacement;
          break;
        }
      }
    }
    if (candidate < 256u) {
      used_low[candidate >> 5] |= 1u << (candidate & 31u);
    }
    const float weight = probabilities[token * expert_count + candidate];
    out[offset] = weight;
    weight_sum += weight;
  }
  const float denominator = max(float(weight_sum), 6.103515625e-5f);
  for (uint route = 0; route < top_k; ++route) {
    const uint offset = token * top_k + route;
    out[offset] /= denominator;
  }
}

// Route IDs normally come from one of the bounded top-k kernels, but this
// operation is also reachable through internal Metal tensors.  Sanitize those
// IDs on-device instead of copying every decode row back to the CPU.  A bad or
// duplicate ID is replaced with the first unused ID in [0, top_k), which is
// guaranteed to exist because the host requires top_k <= expert_count.  Each
// thread owns one token row, so the writes are independent and preserve the
// asynchronous command-buffer chain.
// Float IDs (as packed by the fused top-k route kernels) are accepted only
// when they are exact in-range integers; anything else is treated as invalid
// and replaced below.
inline uint dinf_route_id(uint value, uint) {
  return value;
}
inline uint dinf_route_id(float value, uint expert_count) {
  return value >= 0.0f && value < float(expert_count) && value == floor(value)
      ? uint(value)
      : expert_count;
}

template <typename T>
kernel void dinf_sanitize_moe_ids_t(
    device const T *ids [[buffer(0)]],
    device uint *out [[buffer(1)]],
    constant uint &expert_count [[buffer(2)]],
    constant uint &top_k [[buffer(3)]],
    constant uint &tokens [[buffer(4)]],
    constant uint &ids_row_stride [[buffer(5)]],
    uint token [[thread_position_in_grid]]) {
  if (token >= tokens) {
    return;
  }
  const uint row_offset = token * top_k;
  for (uint route = 0; route < top_k; ++route) {
    uint candidate = dinf_route_id(ids[token * ids_row_stride + route], expert_count);
    bool used = candidate >= expert_count;
    for (uint prior = 0; prior < route && !used; ++prior) {
      used = out[row_offset + prior] == candidate;
    }
    if (used) {
      for (uint replacement = 0; replacement < top_k; ++replacement) {
        bool replacement_used = false;
        for (uint prior = 0; prior < route; ++prior) {
          replacement_used =
              replacement_used || out[row_offset + prior] == replacement;
        }
        if (!replacement_used) {
          candidate = replacement;
          break;
        }
      }
    }
    out[row_offset + route] = candidate;
  }
}
template [[host_name("dinf_sanitize_moe_ids")]] kernel decltype(dinf_sanitize_moe_ids_t<uint>) dinf_sanitize_moe_ids_t<uint>;
template [[host_name("dinf_sanitize_moe_ids_f32")]] kernel decltype(dinf_sanitize_moe_ids_t<float>) dinf_sanitize_moe_ids_t<float>;

"#;

/// Validate routing geometry and, for host-resident test/fallback routes, the
/// selected expert values. Accelerator routes are produced only by this
/// module's bounded top-k kernels or Candle's arg-sort; reading them back here
/// would serialize every MoE projection in the decode hot path.
fn validate_expert_ids(expert_ids: &Tensor, expert_count: usize, operation: &str) -> Result<()> {
    let (tokens, top_k) = expert_ids.dims2()?;
    if expert_ids.dtype() != DType::U32 {
        candle_core::bail!("{operation} requires u32 expert IDs")
    }
    if tokens == 0 || top_k == 0 {
        candle_core::bail!("{operation} requires at least one token and selected expert")
    }
    if expert_count == 0 {
        candle_core::bail!("{operation} cannot select from zero experts")
    }
    if top_k > expert_count {
        candle_core::bail!(
            "{operation} selects {top_k} experts per token, but only {expert_count} experts exist"
        )
    }
    if expert_count > u32::MAX as usize {
        candle_core::bail!(
            "{operation} expert count {expert_count} exceeds the Metal u32 index limit"
        )
    }
    if tokens > u32::MAX as usize || top_k > u32::MAX as usize {
        candle_core::bail!("{operation} route dimensions exceed Metal u32 limits")
    }
    if tokens
        .checked_mul(top_k)
        .is_none_or(|routes| routes > u32::MAX as usize)
    {
        candle_core::bail!("{operation} route count exceeds Metal u32 limits")
    }
    if !expert_ids.device().is_cpu() && top_k > 256 {
        candle_core::bail!(
            "{operation} selects {top_k} experts per token; Metal sanitization supports at most 256"
        )
    }

    if !expert_ids.device().is_cpu() {
        return Ok(());
    }

    let rows = expert_ids.to_vec2::<u32>()?;
    if rows.len() != tokens || rows.iter().any(|row| row.len() != top_k) {
        candle_core::bail!("{operation} expert ID geometry changed during validation")
    }

    let mut seen = HashSet::new();
    for (token, row) in rows.iter().enumerate() {
        seen.clear();
        for (route, &expert) in row.iter().enumerate() {
            if expert >= expert_count as u32 {
                candle_core::bail!(
                    "{operation} expert ID {expert} at token {token}, route {route} is out of range for {expert_count} experts"
                )
            }
            if !seen.insert(expert) {
                candle_core::bail!(
                    "{operation} selects expert {expert} more than once for token {token} (route {route})"
                )
            }
        }
    }
    Ok(())
}

fn sanitize_expert_ids(
    expert_ids: &Tensor,
    expert_count: usize,
    operation: &str,
) -> Result<Tensor> {
    validate_expert_ids(expert_ids, expert_count, operation)?;
    if expert_ids.device().is_cpu() {
        return Ok(expert_ids.clone());
    }
    if !expert_ids.device().is_metal() {
        candle_core::bail!("{operation} requires CPU or Metal expert IDs")
    }
    expert_ids.apply_op1_no_bwd(&SanitizeExpertIds { expert_count })
}

fn checked_u32(value: usize, description: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        candle_core::Error::Msg(format!("{description} {value} exceeds Metal u32 limits"))
    })
}

fn checked_buffer_offset(layout: &Layout, element_size: usize, description: &str) -> Result<usize> {
    layout
        .start_offset()
        .checked_mul(element_size)
        .ok_or_else(|| {
            candle_core::Error::Msg(format!("{description} buffer offset overflows usize"))
        })
}

fn checked_product(lhs: usize, rhs: usize, description: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| candle_core::Error::Msg(format!("{description} overflows usize")))
}

fn route_plan_elements(num_experts: usize, batch: usize) -> Result<usize> {
    let rows = batch
        .checked_add(1)
        .ok_or_else(|| candle_core::Error::Msg("indexed MoE route rows overflow usize".into()))?;
    checked_product(num_experts, rows, "indexed MoE route-plan size")
}

// Candle 0.11's indexed matrix-matrix kernel rebuilds the expert row map in
// every output tile. This wrapper follows llama.cpp's newer two-pass layout:
// compact routes once, then feed the compact map into Candle's quantized MM
// implementation. The quantization formats and math remain Candle's.
const COMPACT_INDEXED_MOE_MSL: &str = r#"
kernel void dinf_compact_moe_routes(
    device const uint *ids [[buffer(0)]],
    device uint *counts [[buffer(1)]],
    device uint *route_map [[buffer(2)]],
    constant uint &num_experts [[buffer(3)]],
    constant uint &batch [[buffer(4)]],
    constant uint &top_k [[buffer(5)]],
    constant uint &ids_row_stride [[buffer(6)]],
    uint expert [[thread_position_in_grid]]) {
  if (expert >= num_experts) {
    return;
  }
  uint count = 0;
  for (uint token = 0; token < batch; ++token) {
    for (uint route = 0; route < top_k; ++route) {
      if (ids[token * ids_row_stride + route] == expert) {
        route_map[expert * batch + count] = token * top_k + route;
        ++count;
      }
    }
  }
  counts[expert] = count;
}

kernel void dinf_compact_moe_routes_atomic(
    device const uint *ids [[buffer(0)]],
    device atomic_uint *counts [[buffer(1)]],
    device uint *route_map [[buffer(2)]],
    constant uint &num_experts [[buffer(3)]],
    constant uint &batch [[buffer(4)]],
    constant uint &top_k [[buffer(5)]],
    constant uint &ids_row_stride [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  for (uint expert = tid; expert < num_experts; expert += threads) {
    atomic_store_explicit(&counts[expert], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_device);

  const uint routes = batch * top_k;
  for (uint flat_route = tid; flat_route < routes; flat_route += threads) {
    const uint token = flat_route / top_k;
    const uint route = flat_route - token * top_k;
    const uint expert = ids[token * ids_row_stride + route];
    if (expert >= num_experts) {
      continue;
    }
    const uint row = atomic_fetch_add_explicit(
        &counts[expert], 1u, memory_order_relaxed);
    route_map[expert * batch + row] = flat_route;
  }
}

kernel void dinf_build_moe_route_tiles(
    device const uint *counts [[buffer(0)]],
    device atomic_uint *tile_count [[buffer(1)]],
    device uint2 *tile_map [[buffer(2)]],
    constant uint &num_experts [[buffer(3)]],
    constant uint &route_tile [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  if (tid == 0) {
    atomic_store_explicit(tile_count, 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_device);

  for (uint expert = tid; expert < num_experts; expert += threads) {
    const uint tiles = (counts[expert] + route_tile - 1u) / route_tile;
    if (tiles == 0) {
      continue;
    }
    const uint first_tile = atomic_fetch_add_explicit(
        tile_count, tiles, memory_order_relaxed);
    for (uint tile = 0; tile < tiles; ++tile) {
      tile_map[first_tile + tile] = uint2(expert, tile * route_tile);
    }
  }
}

template<typename block_q, short nl,
         void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void dinf_kernel_mul_mm_id_compact(
    device const uchar *src0s [[buffer(0)]],
    device const uchar *src1 [[buffer(1)]],
    device float *dst [[buffer(2)]],
    device const uint *counts [[buffer(3)]],
    device const uint *route_map [[buffer(4)]],
    constant int64_t &ne00 [[buffer(5)]],
    constant uint64_t &nb01 [[buffer(6)]],
    constant uint64_t &nb02 [[buffer(7)]],
    constant int64_t &ne11 [[buffer(8)]],
    constant int64_t &ne12 [[buffer(9)]],
    constant uint64_t &nb10 [[buffer(10)]],
    constant uint64_t &nb11 [[buffer(11)]],
    constant uint64_t &nb12 [[buffer(12)]],
    constant int64_t &ne0 [[buffer(13)]],
    constant int64_t &ne1 [[buffer(14)]],
    constant int64_t &ne02 [[buffer(15)]],
    threadgroup uchar *shared_memory [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]) {
  const uint expert = tgpig.z;
  const uint expert_rows = counts[expert];
  const uint first_row = tgpig.x * 32;
  if (first_row >= expert_rows) {
    return;
  }

  threadgroup ushort2 *rowids =
      (threadgroup ushort2 *)(shared_memory + 8192);
  const uint end_row = min(expert_rows, first_row + 32);
  for (uint row = first_row + tiitg; row < end_row; row += 128) {
    const uint flat_route = route_map[expert * ne12 + row];
    rowids[row] = ushort2(flat_route % ne1, flat_route / ne1);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device const uchar *src0 = src0s + expert * nb02;
  kernel_mul_mm_id_impl<block_q, nl, dequantize_func>(
      src0, src1, rowids, dst, ne00, ne02, nb01, nb02, ne11, ne12,
      nb10, nb11, nb12, ne0, expert_rows, ne0 * ne1, shared_memory,
      tgpig, tiitg, sgitg);
}

typedef decltype(dinf_kernel_mul_mm_id_compact<float4x4, 1, dequantize_f32>)
    dinf_compact_mm_id_t;
template [[host_name("dinf_kernel_mul_mm_id_f32_f32")]]
kernel dinf_compact_mm_id_t
    dinf_kernel_mul_mm_id_compact<float4x4, 1, dequantize_f32>;
template [[host_name("dinf_kernel_mul_mm_id_f16_f32")]]
kernel dinf_compact_mm_id_t
    dinf_kernel_mul_mm_id_compact<half4x4, 1, dequantize_f16>;
template [[host_name("dinf_kernel_mul_mm_id_q4_0_f32")]]
kernel dinf_compact_mm_id_t
    dinf_kernel_mul_mm_id_compact<block_q4_0, 2, dequantize_q4_0>;
template [[host_name("dinf_kernel_mul_mm_id_q4_1_f32")]]
kernel dinf_compact_mm_id_t
    dinf_kernel_mul_mm_id_compact<block_q4_1, 2, dequantize_q4_1>;
template [[host_name("dinf_kernel_mul_mm_id_q5_0_f32")]]
kernel dinf_compact_mm_id_t
    dinf_kernel_mul_mm_id_compact<block_q5_0, 2, dequantize_q5_0>;
template [[host_name("dinf_kernel_mul_mm_id_q5_1_f32")]]
kernel dinf_compact_mm_id_t
    dinf_kernel_mul_mm_id_compact<block_q5_1, 2, dequantize_q5_1>;
template [[host_name("dinf_kernel_mul_mm_id_q8_0_f32")]]
kernel dinf_compact_mm_id_t
    dinf_kernel_mul_mm_id_compact<block_q8_0, 2, dequantize_q8_0>;
template [[host_name("dinf_kernel_mul_mm_id_q2_K_f32")]]
kernel dinf_compact_mm_id_t
    dinf_kernel_mul_mm_id_compact<block_q2_K, QK_NL, dequantize_q2_K>;
template [[host_name("dinf_kernel_mul_mm_id_q3_K_f32")]]
kernel dinf_compact_mm_id_t
    dinf_kernel_mul_mm_id_compact<block_q3_K, QK_NL, dequantize_q3_K>;
template [[host_name("dinf_kernel_mul_mm_id_q4_K_f32")]]
kernel dinf_compact_mm_id_t
    dinf_kernel_mul_mm_id_compact<block_q4_K, QK_NL, dequantize_q4_K>;
template [[host_name("dinf_kernel_mul_mm_id_q5_K_f32")]]
kernel dinf_compact_mm_id_t
    dinf_kernel_mul_mm_id_compact<block_q5_K, QK_NL, dequantize_q5_K>;
template [[host_name("dinf_kernel_mul_mm_id_q6_K_f32")]]
kernel dinf_compact_mm_id_t
    dinf_kernel_mul_mm_id_compact<block_q6_K, QK_NL, dequantize_q6_K>;
"#;

// Metal 4 tensor matmul geometry follows llama.cpp's current MUL_MAT_ID
// implementation while retaining dinf's compact route map and output layout.
const METAL4_INDEXED_Q4_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

typedef struct {
    half d;
    uint8_t qs[16];
} block_q4_0;

typedef struct {
    half d;
    half m;
    uint8_t qs[16];
} block_q4_1;

typedef struct {
    half d;
    half m;
    uint8_t qh[4];
    uint8_t qs[16];
} block_q5_1;

// Q5_1 dequantization equivalent to llama.cpp's ggml Metal dequantize_q5_1
// (MIT): `il` 0 yields values 0-15 (low nibbles), 1 yields 16-31 (high
// nibbles), value j taking its fifth bit from qh bit j or 16 + j. The quants
// are read with two 8-byte loads (qs is 8-byte aligned in 24-byte blocks).
void dequantize_q5_1(device const block_q5_1 *xb, short il, thread half4x4 &reg) {
    const float d = xb->d;
    const float m = xb->m;
    const uint2 q0 = ((device const uint2 *)xb->qs)[0];
    const uint2 q1 = ((device const uint2 *)xb->qs)[1];
    const uint4 words = uint4(q0.x, q0.y, q1.x, q1.y);
    const uint qh = *((device const uint32_t *)xb->qh);
    const uint high_bits = il ? (qh >> 16) : qh;
    const uint4 byte_shifts = uint4(0, 8, 16, 24);
    const uint4 bit_shifts = uint4(0, 1, 2, 3);
    FOR_UNROLL (short word = 0; word < 4; ++word) {
        const uint4 bytes = uint4(words[word]) >> byte_shifts;
        const uint4 nibbles = il ? ((bytes >> 4) & 0x0F) : (bytes & 0x0F);
        const uint4 fifth = ((uint4(high_bits >> (word * 4)) >> bit_shifts) & 1) << 4;
        reg[word] = half4(d * float4(nibbles | fifth) + m);
    }
}

// K-quant and Q8_0 blocks and dequantizers follow llama.cpp's ggml Metal
// backend (MIT), matching the dense tensor kernels in tensor_qmatmul_metal.rs.
#define QK_K 256
typedef struct {
    half d;
    int8_t qs[32];
} block_q8_0;

typedef struct {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K/2];
} block_q4_K;

typedef struct {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qh[QK_K/8];
    uint8_t qs[QK_K/2];
} block_q5_K;

typedef struct {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t scales[QK_K/16];
    half d;
} block_q6_K;

void dequantize_q8_0(device const block_q8_0 *xb, short il, thread half4x4 &reg) {
    const half d = xb->d;
    FOR_UNROLL (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = xb->qs[i + 16 * il] * d;
    }
}

static inline uchar2 get_scale_min_k4(int j, int k, device const uchar *q) {
    return j < 4
        ? uchar2{uchar(q[j + k] & 63), uchar(q[j + 4 + k] & 63)}
        : uchar2{
            uchar((q[j + 4 + k] & 0xF) | ((q[j - 4 + k] & 0xC0) >> 2)),
            uchar((q[j + 4 + k] >> 4) | ((q[j + k] & 0xC0) >> 2))
        };
}

void dequantize_q4_K(device const block_q4_K *xb, short il, thread half4x4 &reg) {
    device const uchar *q = xb->qs;
    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4(is, il / 2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.h;
    const float dl = d * sc[0];
    const float ml = float(xb->dmin) * sc[1];
    const ushort mask = il < 2 ? 0x0F : 0xF0;
    FOR_UNROLL (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = dl * (q[i] & mask) - ml;
    }
}

// Chunks `il` (even) and `il + 1` of a Q4_K block, which share one scale
// and one 32-byte span: two 16-byte loads, bit-identical to two
// dequantize_q4_K calls (same as tensor_qmatmul_metal.rs).
void dequantize_q4_K_pair(
    device const block_q4_K *xb,
    short il,
    thread half4x4 &low,
    thread half4x4 &high) {
    device const uchar *q = xb->qs + (il / 4) * 32;
    const short local = il & 3;
    const short is = (il / 4) * 2;
    const uchar2 sc = get_scale_min_k4(is, local / 2, xb->scales);
    const float d = local < 2 ? xb->d : xb->d / 16.h;
    const float dl = d * sc[0];
    const float ml = float(xb->dmin) * sc[1];
    const uint4 packed_low = ((device const uint4 *)q)[0];
    const uint4 packed_high = ((device const uint4 *)q)[1];
    const uint mask = local < 2 ? 0x0F : 0xF0;
    const uint4 shifts = uint4(0, 8, 16, 24);
    FOR_UNROLL (short word = 0; word < 4; ++word) {
        const uint4 lo = (uint4(packed_low[word]) >> shifts) & mask;
        const uint4 hi = (uint4(packed_high[word]) >> shifts) & mask;
        low[word] = half4(dl * float4(lo) - ml);
        high[word] = half4(dl * float4(hi) - ml);
    }
}

void dequantize_q5_K(device const block_q5_K *xb, short il, thread half4x4 &reg) {
    device const uint8_t *q = xb->qs;
    device const uint8_t *qh = xb->qh;
    short is = (il / 4) * 2;
    q = q + 32 * (il / 4) + 16 * (il & 1);
    qh = qh + 16 * (il & 1);
    const uint8_t ul = 1 << (il / 2);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4(is, il / 2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.f;
    const float dl = d * sc[0];
    const float ml = float(xb->dmin) * sc[1];
    const ushort mask = il < 2 ? 0x0F : 0xF0;
    const float qh_value = il < 2 ? 16.f : 256.f;
    FOR_UNROLL (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_value : 0)) - ml;
    }
}

void dequantize_q6_K(device const block_q6_K *xb, short il, thread half4x4 &reg) {
    const half d_all = xb->d;
    device const uint8_t *ql = xb->ql;
    device const uint8_t *qh = xb->qh;
    device const int8_t *scales = xb->scales;
    ql = ql + 64 * (il / 8) + 32 * ((il / 2) & 1) + 16 * (il & 1);
    qh = qh + 32 * (il / 8) + 16 * (il & 1);
    const float sc = scales[(il % 2) + 2 * (il / 2)];
    il = (il / 2) & 3;
    const uint16_t kmask1 = il > 1 ? (il > 2 ? 192 : 48) : (il > 0 ? 12 : 3);
    const uint16_t kmask2 = il > 1 ? 0xF0 : 0x0F;
    const float coef = il > 1 ? 1.f / 16.f : 1.f;
    const float ml = d_all * sc * 32.f;
    const float dl = d_all * sc * coef;
    FOR_UNROLL (int i = 0; i < 16; ++i) {
        const half q = il & 1
            ? ((ql[i] & kmask2) | ((qh[i] & kmask1) << 2))
            : ((ql[i] & kmask2) | ((qh[i] & kmask1) << 4));
        reg[i / 4][i % 4] = dl * q - ml;
    }
}

void dequantize_q4_0_legacy(
    device const block_q4_0 *xb, short il, thread half4x4 &reg) {
    const half d = xb->d;
    FOR_UNROLL (short i = 0; i < 16; ++i) {
        const short q = il == 0 ? (xb->qs[i] & 0x0F) : (xb->qs[i] >> 4);
        reg[i / 4][i % 4] = d * half(q - 8);
    }
}

void dequantize_q4_1_legacy(
    device const block_q4_1 *xb, short il, thread half4x4 &reg) {
    const float d = float(xb->d);
    const float m = float(xb->m);
    FOR_UNROLL (short i = 0; i < 16; ++i) {
        const short q = il == 0 ? (xb->qs[i] & 0x0F) : (xb->qs[i] >> 4);
        reg[i / 4][i % 4] = half(d * float(q) + m);
    }
}

void dequantize_q4_0(device const block_q4_0 *xb, short il, thread half4x4 &reg) {
    device const ushort *qs = ((device const ushort *)xb + 1);
    const float d1 = il == 0 ? float(xb->d) : float(xb->d / 16.h);
    const float d2 = d1 / 256.0f;
    const float md = -8.0f * float(xb->d);
    const ushort mask0 = il == 0 ? 0x000F : 0x00F0;
    const ushort mask1 = mask0 << 8;
    float4x4 values;
    FOR_UNROLL (short i = 0; i < 8; ++i) {
        values[i / 2][2 * (i % 2)] = d1 * float(qs[i] & mask0) + md;
        values[i / 2][2 * (i % 2) + 1] = d2 * float(qs[i] & mask1) + md;
    }
    reg = half4x4(values);
}

void dequantize_q4_1(device const block_q4_1 *xb, short il, thread half4x4 &reg) {
    device const ushort *qs = ((device const ushort *)xb + 2);
    const float d1 = il == 0 ? float(xb->d) : float(xb->d / 16.h);
    const float d2 = d1 / 256.0f;
    const float m = float(xb->m);
    const ushort mask0 = il == 0 ? 0x000F : 0x00F0;
    const ushort mask1 = mask0 << 8;
    float4x4 values;
    FOR_UNROLL (short i = 0; i < 8; ++i) {
        values[i / 2][2 * (i % 2)] = d1 * float(qs[i] & mask0) + m;
        values[i / 2][2 * (i % 2) + 1] = d2 * float(qs[i] & mask1) + m;
    }
    reg = half4x4(values);
}

#ifdef DINF_F16_WEIGHTS
#define DINF_KERNEL_NAME dinf_metal4_indexed_f16
#elif defined(DINF_Q8_0)
#define DINF_BLOCK_TYPE block_q8_0
#define DINF_DEQUANTIZE dequantize_q8_0
#define DINF_KERNEL_NAME dinf_metal4_indexed_q8_0
#elif defined(DINF_Q4_K)
#define DINF_BLOCK_TYPE block_q4_K
#define DINF_DEQUANTIZE dequantize_q4_K
#define DINF_DEQUANTIZE_PAIR dequantize_q4_K_pair
#define DINF_NL 16
#define DINF_KERNEL_NAME dinf_metal4_indexed_q4_K
#elif defined(DINF_Q5_K)
#define DINF_BLOCK_TYPE block_q5_K
#define DINF_DEQUANTIZE dequantize_q5_K
#define DINF_NL 16
#define DINF_KERNEL_NAME dinf_metal4_indexed_q5_K
#elif defined(DINF_Q6_K)
#define DINF_BLOCK_TYPE block_q6_K
#define DINF_DEQUANTIZE dequantize_q6_K
#define DINF_NL 16
#define DINF_KERNEL_NAME dinf_metal4_indexed_q6_K
#elif defined(DINF_Q5_1)
#define DINF_BLOCK_TYPE block_q5_1
#define DINF_DEQUANTIZE dequantize_q5_1
#define DINF_KERNEL_NAME dinf_metal4_indexed_q5_1
#elif defined(DINF_Q4_1)
#define DINF_BLOCK_TYPE block_q4_1
#ifdef DINF_LEGACY_Q4_DEQUANT
#define DINF_DEQUANTIZE dequantize_q4_1_legacy
#else
#define DINF_DEQUANTIZE dequantize_q4_1
#endif
#define DINF_KERNEL_NAME dinf_metal4_indexed_q4_1
#else
#define DINF_BLOCK_TYPE block_q4_0
#ifdef DINF_LEGACY_Q4_DEQUANT
#define DINF_DEQUANTIZE dequantize_q4_0_legacy
#else
#define DINF_DEQUANTIZE dequantize_q4_0
#endif
#define DINF_KERNEL_NAME dinf_metal4_indexed_q4_0
#endif

// Blocks hold 16 * DINF_NL values; dequantizers emit 16-value chunks.
#ifndef DINF_NL
#define DINF_NL 2
#endif

// Half-staged activations with a strict matmul, as llama.cpp's
// kernel_mul_mm_id does. The original Q4_0/Q4_1/F16 kernels keep float
// activations with relaxed MPP precision.
#ifdef DINF_HALF_INPUT
typedef half dinf_input_t;
#define DINF_RELAXED_PRECISION false
#else
typedef float dinf_input_t;
#define DINF_RELAXED_PRECISION true
#endif

#ifdef DINF_TILE_ROWS_256
#define DINF_TILE_ROWS 256
#define DINF_WEIGHT_TILE_BYTES 16384
#define DINF_MATMUL_SIMDGROUPS 16
#elif defined(DINF_TILE_ROWS_128)
#define DINF_TILE_ROWS 128
#define DINF_WEIGHT_TILE_BYTES 8192
#define DINF_MATMUL_SIMDGROUPS 8
#else
#define DINF_TILE_ROWS 64
#define DINF_WEIGHT_TILE_BYTES 4096
#define DINF_MATMUL_SIMDGROUPS 4
#endif

#ifdef DINF_HALF_INPUT
// Half-input expert kernel (Q8_0 and K-quant experts). Activations and
// dequantized weights are staged as half and the matmul is strict, like
// llama.cpp's kernel_mul_mm_id. Each barrier pair stages DINF_K_TILE values of
// K (two dequantized 32-value steps), halving the synchronization per FLOP.
#define DINF_K_TILE 64
#ifndef DINF_ROUTE_TILE
#define DINF_ROUTE_TILE 32
#endif
kernel void DINF_KERNEL_NAME(
    device const uchar *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const uint *counts [[buffer(3)]],
    device const uint *route_map [[buffer(4)]],
    constant uint &in_dim [[buffer(5)]],
    constant uint &out_dim [[buffer(6)]],
    constant uint &batch [[buffer(7)]],
    constant uint &top_k [[buffer(8)]],
    constant uint &input_rows [[buffer(9)]],
    constant uint &weight_row_bytes [[buffer(10)]],
    constant uint &weight_expert_bytes [[buffer(11)]],
    constant uint &relu_squared [[buffer(12)]],
    device const uint *tile_count [[buffer(13)]],
    device const uint2 *tile_map [[buffer(14)]],
    constant uint &use_tile_queue [[buffer(15)]],
    threadgroup uchar *shared_memory [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    constexpr uint tile_rows = DINF_TILE_ROWS;
    constexpr uint tile_routes = DINF_ROUTE_TILE;
    constexpr uint tile_k = DINF_K_TILE;
    constexpr uint chunks = tile_k / 16;          // 16-value dequant chunks per row
    constexpr uint row_threads = 2;               // threads per weight row

    uint expert;
    uint route_start;
    if (use_tile_queue != 0) {
        const uint tile_index = group.x;
        if (tile_index >= tile_count[0]) {
            return;
        }
        const uint2 tile = tile_map[tile_index];
        expert = tile.x;
        route_start = tile.y;
    } else {
        expert = group.z;
        route_start = group.x * tile_routes;
    }
    const uint expert_rows = counts[expert];
    if (route_start >= expert_rows) {
        return;
    }
#ifdef DINF_GEGLU_ROWS
    // Packed [gate | up] expert rows: tile rows 0..half are gate rows and
    // half..tile_rows the matching up rows; the epilogue writes
    // gelu(gate) * up for `half` activation columns.
    constexpr uint half_rows = tile_rows / 2;
    const uint ff = out_dim / 2;
    const uint gate_start = group.y * half_rows;
    const uint valid_rows = min(half_rows, ff - gate_start);
#else
    const uint row_start = group.y * tile_rows;
    const uint valid_rows = min(tile_rows, out_dim - row_start);
#endif
    const uint valid_routes = min(tile_routes, expert_rows - route_start);

    threadgroup half *weight_tile_data = (threadgroup half *)shared_memory;      // [rows][k]
    threadgroup half *input_tile_data = weight_tile_data + tile_rows * tile_k;   // [routes][k]
    threadgroup uint *route_input_offsets =
        (threadgroup uint *)(input_tile_data + tile_routes * tile_k);
    auto weight_tile = tensor(weight_tile_data, dextents<int32_t, 2>(int(tile_k), int(tile_rows)));
    auto input_tile = tensor(input_tile_data, dextents<int32_t, 2>(int(tile_k), int(tile_routes)));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            tile_routes, tile_rows, tile_k, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<DINF_MATMUL_SIMDGROUPS>> matmul;
    auto destination = matmul.get_destination_cooperative_tensor<
        decltype(input_tile), decltype(weight_tile), float>();
    FOR_UNROLL (ushort i = 0; i < destination.get_capacity(); ++i) {
        destination[i] = 0.0f;
    }

    if (tid < valid_routes) {
        const uint flat_route = route_map[expert * batch + route_start + tid];
        const uint token = flat_route / top_k;
        const uint input_row = (flat_route % top_k) % input_rows;
        route_input_offsets[tid] = (token * input_rows + input_row) * in_dim;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint weight_row = tid / row_threads;
    const uint weight_part = tid % row_threads;
#ifdef DINF_GEGLU_ROWS
    const uint pair_row = weight_row % half_rows;
    const bool weight_row_valid = pair_row < valid_rows;
    const uint packed_row = (weight_row < half_rows ? 0u : ff) + gate_start + pair_row;
#else
    const bool weight_row_valid = weight_row < valid_rows;
    const uint packed_row = row_start + weight_row;
#endif
    device const DINF_BLOCK_TYPE *row_weights = (device const DINF_BLOCK_TYPE *)(
        weights + expert * weight_expert_bytes + packed_row * weight_row_bytes);
    // Activation staging: tile_routes * tile_k values, four per thread per pass.
    constexpr uint input_vectors = tile_routes * tile_k / 4;
    for (uint k_start = 0; k_start < in_dim; k_start += tile_k) {
        // Dequantize this row's chunks for the K tile into registers first.
        half4x4 values[chunks / row_threads];
#ifdef DINF_DEQUANTIZE_PAIR
        // Each thread takes adjacent chunks, which share a scale and span.
        static_assert(chunks / row_threads == 2, "paired dequantization needs two chunks");
        {
            const uint local = weight_part * 2;
            if (weight_row_valid && k_start + local * 16 < in_dim) {
                const uint chunk = k_start / 16 + local;
                DINF_DEQUANTIZE_PAIR(
                    row_weights + chunk / DINF_NL, short(chunk % DINF_NL), values[0], values[1]);
            } else {
                values[0] = half4x4(0.h);
                values[1] = half4x4(0.h);
            }
        }
#define DINF_CHUNK_LOCAL(c) (weight_part * 2 + (c))
#else
        FOR_UNROLL (uint c = 0; c < chunks / row_threads; ++c) {
            const uint local = c * row_threads + weight_part;
            if (weight_row_valid && k_start + local * 16 < in_dim) {
                const uint chunk = k_start / 16 + local;
                DINF_DEQUANTIZE(row_weights + chunk / DINF_NL, short(chunk % DINF_NL), values[c]);
            } else {
                values[c] = half4x4(0.h);
            }
        }
#define DINF_CHUNK_LOCAL(c) ((c) * row_threads + weight_part)
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL (uint c = 0; c < chunks / row_threads; ++c) {
            const uint local = DINF_CHUNK_LOCAL(c);
            threadgroup half4 *dst =
                (threadgroup half4 *)(weight_tile_data + weight_row * tile_k + local * 16);
            FOR_UNROLL (short i = 0; i < 4; ++i) {
                dst[i] = values[c][i];
            }
        }
        for (uint v = tid; v < input_vectors; v += threads.x) {
            const uint route = v / (tile_k / 4);
            const uint k4 = (v % (tile_k / 4)) * 4;
            float4 value = 0.0f;
            if (route < valid_routes && k_start + k4 < in_dim) {
                value = *(device const packed_float4 *)(
                    input + route_input_offsets[route] + k_start + k4);
                if (relu_squared != 0) {
                    value = max(value, 0.0f);
                    value *= value;
                }
            }
            *(threadgroup half4 *)(input_tile_data + route * tile_k + k4) = half4(value);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        matmul.run(input_tile, weight_tile, destination);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *result_data = (threadgroup float *)shared_memory;
    auto result = tensor(result_data, dextents<int32_t, 2>(int(tile_rows), int(tile_routes)));
    destination.store(result);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = tid; index < valid_routes * valid_rows; index += threads.x) {
        const uint route = index / valid_rows;
        const uint row = index - route * valid_rows;
        const uint flat_route = route_map[expert * batch + route_start + route];
#ifdef DINF_GEGLU_ROWS
        // Same arithmetic as dinf_geglu_f32 on these f32 results.
        const float x = result_data[route * tile_rows + row];
        const float up = result_data[route * tile_rows + half_rows + row];
        float activated;
        if (x > 5.0f) {
            activated = x;
        } else {
            const float x_cube = (x * x) * x;
            const float alpha = x + 0.044715f * x_cube;
            const float beta = float(M_2_SQRTPI_F * M_SQRT1_2_F) * alpha;
            activated = 0.5f * x * (1.0f + precise::tanh(beta));
        }
        output[flat_route * ff + gate_start + row] = activated * up;
#else
        output[flat_route * out_dim + row_start + row] = result_data[route * tile_rows + row];
#endif
    }
}
#ifdef DINF_GATE_UP
#define DINF_CAT2(a, b) a##b
#define DINF_CAT(a, b) DINF_CAT2(a, b)
// Fused gate and up projections for SwiGLU experts: both weight tiles share
// one activation tile and route table; the epilogue writes
// silu(gate) * up (same formula as dinf_silu_mul_f32) once.
kernel void DINF_CAT(DINF_KERNEL_NAME, _gate_up)(
    device const uchar *gate_weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const uint *counts [[buffer(3)]],
    device const uint *route_map [[buffer(4)]],
    constant uint &in_dim [[buffer(5)]],
    constant uint &out_dim [[buffer(6)]],
    constant uint &batch [[buffer(7)]],
    constant uint &top_k [[buffer(8)]],
    constant uint &input_rows [[buffer(9)]],
    constant uint &weight_row_bytes [[buffer(10)]],
    constant uint &weight_expert_bytes [[buffer(11)]],
    constant uint &relu_squared [[buffer(12)]],
    device const uint *tile_count [[buffer(13)]],
    device const uint2 *tile_map [[buffer(14)]],
    constant uint &use_tile_queue [[buffer(15)]],
    device const uchar *up_weights [[buffer(16)]],
    threadgroup uchar *shared_memory [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    constexpr uint tile_rows = DINF_TILE_ROWS;
    constexpr uint tile_routes = 32;
    constexpr uint tile_k = 32;

    uint expert;
    uint route_start;
    if (use_tile_queue != 0) {
        const uint tile_index = group.x;
        if (tile_index >= tile_count[0]) {
            return;
        }
        const uint2 tile = tile_map[tile_index];
        expert = tile.x;
        route_start = tile.y;
    } else {
        expert = group.z;
        route_start = group.x * tile_routes;
    }
    const uint expert_rows = counts[expert];
    if (route_start >= expert_rows) {
        return;
    }
    const uint row_start = group.y * tile_rows;
    const uint valid_rows = min(tile_rows, out_dim - row_start);
    const uint valid_routes = min(tile_routes, expert_rows - route_start);

    threadgroup half *gate_tile_data = (threadgroup half *)shared_memory;      // [rows][k]
    threadgroup half *up_tile_data = gate_tile_data + tile_rows * tile_k;      // [rows][k]
    threadgroup half *input_tile_data = up_tile_data + tile_rows * tile_k;     // [routes][k]
    threadgroup uint *route_input_offsets =
        (threadgroup uint *)(input_tile_data + tile_routes * tile_k);
    auto gate_tile = tensor(gate_tile_data, dextents<int32_t, 2>(int(tile_k), int(tile_rows)));
    auto up_tile = tensor(up_tile_data, dextents<int32_t, 2>(int(tile_k), int(tile_rows)));
    auto input_tile = tensor(input_tile_data, dextents<int32_t, 2>(int(tile_k), int(tile_routes)));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            tile_routes, tile_rows, tile_k, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<DINF_MATMUL_SIMDGROUPS>> matmul;
    auto gate_acc = matmul.get_destination_cooperative_tensor<
        decltype(input_tile), decltype(gate_tile), float>();
    auto up_acc = matmul.get_destination_cooperative_tensor<
        decltype(input_tile), decltype(up_tile), float>();
    FOR_UNROLL (ushort i = 0; i < gate_acc.get_capacity(); ++i) {
        gate_acc[i] = 0.0f;
        up_acc[i] = 0.0f;
    }

    if (tid < valid_routes) {
        const uint flat_route = route_map[expert * batch + route_start + tid];
        const uint token = flat_route / top_k;
        const uint input_row = (flat_route % top_k) % input_rows;
        route_input_offsets[tid] = (token * input_rows + input_row) * in_dim;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint weight_row = tid / 2;
    const uint weight_part = tid % 2;
    const uint row_offset = expert * weight_expert_bytes + (row_start + weight_row) * weight_row_bytes;
    device const DINF_BLOCK_TYPE *gate_row = (device const DINF_BLOCK_TYPE *)(gate_weights + row_offset);
    device const DINF_BLOCK_TYPE *up_row = (device const DINF_BLOCK_TYPE *)(up_weights + row_offset);
    constexpr uint input_vectors = tile_routes * tile_k / 4;
    for (uint k_start = 0; k_start < in_dim; k_start += tile_k) {
        half4x4 gate_values = half4x4(0.h);
        half4x4 up_values = half4x4(0.h);
        if (weight_row < valid_rows) {
            const uint chunk = k_start / 16 + weight_part;
            DINF_DEQUANTIZE(gate_row + chunk / DINF_NL, short(chunk % DINF_NL), gate_values);
            DINF_DEQUANTIZE(up_row + chunk / DINF_NL, short(chunk % DINF_NL), up_values);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            threadgroup half4 *gate_dst =
                (threadgroup half4 *)(gate_tile_data + weight_row * tile_k + weight_part * 16);
            threadgroup half4 *up_dst =
                (threadgroup half4 *)(up_tile_data + weight_row * tile_k + weight_part * 16);
            FOR_UNROLL (short i = 0; i < 4; ++i) {
                gate_dst[i] = gate_values[i];
                up_dst[i] = up_values[i];
            }
        }
        for (uint v = tid; v < input_vectors; v += threads.x) {
            const uint route = v / (tile_k / 4);
            const uint k4 = (v % (tile_k / 4)) * 4;
            float4 value = 0.0f;
            if (route < valid_routes) {
                value = *(device const packed_float4 *)(
                    input + route_input_offsets[route] + k_start + k4);
            }
            *(threadgroup half4 *)(input_tile_data + route * tile_k + k4) = half4(value);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        matmul.run(input_tile, gate_tile, gate_acc);
        matmul.run(input_tile, up_tile, up_acc);
    }

    FOR_UNROLL (ushort i = 0; i < gate_acc.get_capacity(); ++i) {
        const float g = gate_acc[i];
        gate_acc[i] = (g / (1.0f + exp(-g))) * up_acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *result_data = (threadgroup float *)shared_memory;
    auto result = tensor(result_data, dextents<int32_t, 2>(int(tile_rows), int(tile_routes)));
    gate_acc.store(result);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = tid; index < valid_routes * valid_rows; index += threads.x) {
        const uint route = index / valid_rows;
        const uint row = index - route * valid_rows;
        const uint flat_route = route_map[expert * batch + route_start + route];
        output[flat_route * out_dim + row_start + row] = result_data[route * tile_rows + row];
    }
}
#endif
#else
kernel void DINF_KERNEL_NAME(
    device const uchar *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const uint *counts [[buffer(3)]],
    device const uint *route_map [[buffer(4)]],
    constant uint &in_dim [[buffer(5)]],
    constant uint &out_dim [[buffer(6)]],
    constant uint &batch [[buffer(7)]],
    constant uint &top_k [[buffer(8)]],
    constant uint &input_rows [[buffer(9)]],
    constant uint &weight_row_bytes [[buffer(10)]],
    constant uint &weight_expert_bytes [[buffer(11)]],
    constant uint &relu_squared [[buffer(12)]],
    device const uint *tile_count [[buffer(13)]],
    device const uint2 *tile_map [[buffer(14)]],
    constant uint &use_tile_queue [[buffer(15)]],
    threadgroup uchar *shared_memory [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    constexpr uint tile_rows = DINF_TILE_ROWS;
    constexpr uint tile_routes = 32;
    constexpr uint tile_k = 32;

    uint expert;
    uint route_start;
    if (use_tile_queue != 0) {
        const uint tile_index = group.x;
        if (tile_index >= tile_count[0]) {
            return;
        }
        const uint2 tile = tile_map[tile_index];
        expert = tile.x;
        route_start = tile.y;
    } else {
        expert = group.z;
        route_start = group.x * tile_routes;
    }
    const uint expert_rows = counts[expert];
    if (route_start >= expert_rows) {
        return;
    }
    const uint row_start = group.y * tile_rows;
    const uint valid_rows = min(tile_rows, out_dim - row_start);
    const uint valid_routes = min(tile_routes, expert_rows - route_start);

    threadgroup half *weight_tile_data = (threadgroup half *)shared_memory;
    threadgroup dinf_input_t *input_tile_data =
        (threadgroup dinf_input_t *)(shared_memory + DINF_WEIGHT_TILE_BYTES);
    auto weight_tile = tensor(
        weight_tile_data,
        dextents<int32_t, 2>(int(tile_k), int(tile_rows)));
    auto input_tile = tensor(
        input_tile_data,
        dextents<int32_t, 2>(int(tile_routes), int(tile_k)));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            tile_routes,
            tile_rows,
            tile_k,
            false,
            true,
            DINF_RELAXED_PRECISION,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<DINF_MATMUL_SIMDGROUPS>> matmul;
    auto destination = matmul.get_destination_cooperative_tensor<
        decltype(input_tile), decltype(weight_tile), float>();

#ifdef DINF_PRECOMPUTE_ROUTE_OFFSETS
    threadgroup uint *route_input_offsets =
        (threadgroup uint *)(shared_memory + DINF_WEIGHT_TILE_BYTES
            + tile_routes * tile_k * sizeof(float));
    if (tid < valid_routes) {
        const uint flat_route = route_map[expert * batch + route_start + tid];
        const uint route_slot = flat_route % top_k;
        const uint token = flat_route / top_k;
        const uint input_row = route_slot % input_rows;
        route_input_offsets[tid] = (token * input_rows + input_row) * in_dim;
    }
#endif

    const uint weight_row = tid / 2;
    const short weight_half = short(tid % 2);
    device const uchar *expert_weights = weights + expert * weight_expert_bytes;
    for (uint k_start = 0; k_start < in_dim; k_start += tile_k) {
        half4x4 values;
        if (weight_row < valid_rows) {
#ifdef DINF_F16_WEIGHTS
            device const half *row_weights =
                (device const half *)(
                    expert_weights + (row_start + weight_row) * weight_row_bytes);
            FOR_UNROLL (short index = 0; index < 16; ++index) {
                values[index / 4][index % 4] =
                    row_weights[k_start + weight_half * 16 + index];
            }
#else
            device const DINF_BLOCK_TYPE *row_weights =
                (device const DINF_BLOCK_TYPE *)(
                    expert_weights + (row_start + weight_row) * weight_row_bytes);
            const uint chunk = k_start / 16 + uint(weight_half);
            DINF_DEQUANTIZE(row_weights + chunk / DINF_NL, short(chunk % DINF_NL), values);
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL (short index = 0; index < 16; ++index) {
            weight_tile_data[weight_row * tile_k + weight_half * 16 + index] =
                weight_row < valid_rows ? values[index / 4][index % 4] : 0.h;
        }

        for (uint index = tid; index < tile_routes * tile_k; index += threads.x) {
            const uint route = index / tile_k;
            const uint k = index - route * tile_k;
            float value = 0.0f;
            if (route < valid_routes) {
#ifdef DINF_PRECOMPUTE_ROUTE_OFFSETS
                value = input[route_input_offsets[route] + k_start + k];
#else
                const uint flat_route =
                    route_map[expert * batch + route_start + route];
                const uint route_slot = flat_route % top_k;
                const uint token = flat_route / top_k;
                const uint input_row = route_slot % input_rows;
                value = input[(token * input_rows + input_row) * in_dim + k_start + k];
#endif
                if (relu_squared != 0) {
                    value = max(value, 0.0f);
                    value *= value;
                }
            }
            input_tile_data[index] = dinf_input_t(value);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto input_slice = input_tile.slice(0, 0);
        auto weight_slice = weight_tile.slice(0, 0);
        matmul.run(input_slice, weight_slice, destination);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *result_data = (threadgroup float *)shared_memory;
    auto result = tensor(
        result_data,
        dextents<int32_t, 2>(int(tile_rows), int(tile_routes)));
    destination.store(result);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = tid; index < valid_routes * valid_rows; index += threads.x) {
        const uint route = index / valid_rows;
        const uint row = index - route * valid_rows;
        const uint flat_route = route_map[expert * batch + route_start + route];
        output[flat_route * out_dim + row_start + row] =
            result_data[route * tile_rows + row];
    }
}
#endif
"#;

static EPILOGUE_PIPELINES: OnceLock<Mutex<HashMap<(DeviceId, &'static str), ComputePipeline>>> =
    OnceLock::new();
static COMPACT_MOE_LIBRARIES: OnceLock<Mutex<HashMap<DeviceId, Library>>> = OnceLock::new();
static COMPACT_MOE_PIPELINES: OnceLock<Mutex<HashMap<(DeviceId, &'static str), ComputePipeline>>> =
    OnceLock::new();
type Metal4IndexedQ4PipelineKey = (DeviceId, GgmlDType, usize, bool, bool);
type Metal4IndexedQ4PipelineCache = Mutex<HashMap<Metal4IndexedQ4PipelineKey, ComputePipeline>>;

static METAL4_INDEXED_Q4_PIPELINES: OnceLock<Metal4IndexedQ4PipelineCache> = OnceLock::new();
static METAL4_INDEXED_Q4_LOGGED: OnceLock<()> = OnceLock::new();

fn supports_metal4_indexed_moe(device: &candle_core::MetalDevice) -> bool {
    device
        .metal_device()
        .as_ref()
        .supportsFamily(MTLGPUFamily::Metal4)
}

fn metal4_indexed_q4_enabled(device: &candle_core::MetalDevice, dtype: GgmlDType) -> bool {
    supports_metal4_indexed_moe(device)
        && (matches!(dtype, GgmlDType::Q4_0 | GgmlDType::Q4_1)
            || metal4_indexed_half_input(dtype)
            || (cfg!(test) && dtype == GgmlDType::F16))
}

/// Half-input tiles hold 128 rows of 32 routes. 256 rows exceed threadgroup
/// memory.
const METAL4_INDEXED_HALF_INPUT_TILE_ROWS: usize = 128;

/// Expert types whose Metal 4 kernel stages activations through half.
fn metal4_indexed_half_input(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q5_1 | GgmlDType::Q8_0 | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K
    )
}

fn metal4_indexed_q4_tile_rows(batch: usize) -> usize {
    if batch >= 256 { 256 } else { 128 }
}

fn metal4_indexed_q4_pipeline(
    device: &candle_core::MetalDevice,
    dtype: GgmlDType,
    tile_rows: usize,
) -> Result<ComputePipeline> {
    metal4_indexed_pipeline(device, dtype, tile_rows, false, false)
}

/// Half-input expert kernel over packed `[gate | up]` rows that writes
/// `gelu(gate) * up` (Gemma 4 experts).
fn metal4_indexed_geglu_pipeline(
    device: &candle_core::MetalDevice,
    dtype: GgmlDType,
    tile_rows: usize,
) -> Result<ComputePipeline> {
    if !metal4_indexed_half_input(dtype) {
        candle_core::bail!("packed GEGLU expert kernel does not support {dtype:?}")
    }
    metal4_indexed_pipeline(device, dtype, tile_rows, false, true)
}

/// Fused SwiGLU gate/up variant of the half-input expert kernel.
fn metal4_indexed_gate_up_pipeline(
    device: &candle_core::MetalDevice,
    dtype: GgmlDType,
    tile_rows: usize,
) -> Result<ComputePipeline> {
    if !metal4_indexed_half_input(dtype) {
        candle_core::bail!("fused gate/up expert kernel does not support {dtype:?}")
    }
    metal4_indexed_pipeline(device, dtype, tile_rows, true, false)
}

fn metal4_indexed_pipeline(
    device: &candle_core::MetalDevice,
    dtype: GgmlDType,
    tile_rows: usize,
    gate_up: bool,
    geglu_rows: bool,
) -> Result<ComputePipeline> {
    let pipelines = METAL4_INDEXED_Q4_PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (device.id(), dtype, tile_rows, gate_up, geglu_rows);
    if let Some(pipeline) = pipelines.lock().unwrap().get(&key) {
        return Ok(pipeline.clone());
    }
    let tile_geometry = match tile_rows {
        256 => "#define DINF_TILE_ROWS_256 1\n",
        128 => "#define DINF_TILE_ROWS_128 1\n",
        _ => "",
    };
    let route_offsets = "#define DINF_PRECOMPUTE_ROUTE_OFFSETS 1\n";
    let gate_up_define = match (gate_up, geglu_rows) {
        (true, _) => "#define DINF_GATE_UP 1\n",
        (false, true) => "#define DINF_GEGLU_ROWS 1\n",
        (false, false) => "",
    };
    let (source, function_name) = match dtype {
        GgmlDType::F16 if cfg!(test) => (
            format!(
                "{tile_geometry}{route_offsets}#define DINF_F16_WEIGHTS 1\n{METAL4_INDEXED_Q4_MSL}"
            ),
            "dinf_metal4_indexed_f16",
        ),
        GgmlDType::Q4_0 => (
            format!("{tile_geometry}{route_offsets}{METAL4_INDEXED_Q4_MSL}"),
            "dinf_metal4_indexed_q4_0",
        ),
        GgmlDType::Q4_1 => (
            format!("{tile_geometry}{route_offsets}#define DINF_Q4_1 1\n{METAL4_INDEXED_Q4_MSL}"),
            "dinf_metal4_indexed_q4_1",
        ),
        GgmlDType::Q8_0 => (
            format!(
                "{gate_up_define}{tile_geometry}{route_offsets}#define DINF_HALF_INPUT 1\n#define DINF_Q8_0 1\n{METAL4_INDEXED_Q4_MSL}"
            ),
            "dinf_metal4_indexed_q8_0",
        ),
        GgmlDType::Q5_1 => (
            format!(
                "{gate_up_define}{tile_geometry}{route_offsets}#define DINF_HALF_INPUT 1\n#define DINF_Q5_1 1\n{METAL4_INDEXED_Q4_MSL}"
            ),
            "dinf_metal4_indexed_q5_1",
        ),
        GgmlDType::Q4K => (
            format!(
                "{gate_up_define}{tile_geometry}{route_offsets}#define DINF_HALF_INPUT 1\n#define DINF_Q4_K 1\n{METAL4_INDEXED_Q4_MSL}"
            ),
            "dinf_metal4_indexed_q4_K",
        ),
        GgmlDType::Q5K => (
            format!(
                "{gate_up_define}{tile_geometry}{route_offsets}#define DINF_HALF_INPUT 1\n#define DINF_Q5_K 1\n{METAL4_INDEXED_Q4_MSL}"
            ),
            "dinf_metal4_indexed_q5_K",
        ),
        GgmlDType::Q6K => (
            format!(
                "{gate_up_define}{tile_geometry}{route_offsets}#define DINF_HALF_INPUT 1\n#define DINF_Q6_K 1\n{METAL4_INDEXED_Q4_MSL}"
            ),
            "dinf_metal4_indexed_q6_K",
        ),
        _ => candle_core::bail!("Metal 4 indexed matmul does not support {dtype:?}"),
    };
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version4_0);
    let library = device
        .metal_device()
        .new_library_with_source(&source, Some(&options))
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function(
            &if gate_up {
                format!("{function_name}_gate_up")
            } else {
                function_name.to_string()
            },
            None,
        )
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    pipelines.lock().unwrap().insert(key, pipeline.clone());
    Ok(pipeline)
}

fn epilogue_pipeline(
    device: &candle_core::MetalDevice,
    name: &'static str,
) -> Result<ComputePipeline> {
    let pipelines = EPILOGUE_PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (device.id(), name);
    if let Some(pipeline) = pipelines.lock().unwrap().get(&key) {
        return Ok(pipeline.clone());
    }
    // Preserve the left-to-right routing sum without volatile local storage.
    // Keep the existing math settings for the other epilogue kernels.
    let options = matches!(
        name,
        "dinf_normalized_route_weights_f32" | "dinf_ordered_weighted_route_sum_f32"
    )
    .then(|| {
        let options = MTLCompileOptions::new();
        #[allow(deprecated)]
        options.setFastMathEnabled(false);
        options
    });
    let library = device
        .metal_device()
        .new_library_with_source(MOE_EPILOGUE_MSL, options.as_deref())
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

fn compact_moe_library(device: &candle_core::MetalDevice) -> Result<Library> {
    let libraries = COMPACT_MOE_LIBRARIES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(library) = libraries.lock().unwrap().get(&device.id()) {
        return Ok(library.clone());
    }
    let source = format!(
        "{}\n{}",
        candle_metal_kernels::source::QUANTIZED,
        COMPACT_INDEXED_MOE_MSL
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

fn compact_moe_pipeline(
    device: &candle_core::MetalDevice,
    name: &'static str,
) -> Result<ComputePipeline> {
    let pipelines = COMPACT_MOE_PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (device.id(), name);
    if let Some(pipeline) = pipelines.lock().unwrap().get(&key) {
        return Ok(pipeline.clone());
    }
    let function = compact_moe_library(device)?
        .get_function(name, None)
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    pipelines.lock().unwrap().insert(key, pipeline.clone());
    Ok(pipeline)
}

fn dispatch_compact_routes(
    device: &candle_core::MetalDevice,
    encoder: &candle_metal_kernels::metal::ComputeCommandEncoder,
    num_experts: usize,
    batch: usize,
    top_k: usize,
) -> Result<()> {
    let routes = checked_product(batch, top_k, "compact MoE route count")?;
    let use_atomic = routes >= 64;
    let name = if use_atomic {
        "dinf_compact_moe_routes_atomic"
    } else {
        "dinf_compact_moe_routes"
    };
    encoder.set_compute_pipeline_state(&compact_moe_pipeline(device, name)?);
    if use_atomic {
        let threads = num_experts.max(routes).min(256).next_multiple_of(32);
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
    } else {
        encoder.dispatch_threads(
            objc2_metal::MTLSize {
                width: num_experts,
                height: 1,
                depth: 1,
            },
            candle_metal_kernels::utils::get_block_dims(num_experts.min(256), 1, 1),
        );
    }
    Ok(())
}

/// `activation(gate) * up` for an elementwise activation kernel.
struct GatedMul {
    kernel: &'static str,
}

impl CustomOp2 for GatedMul {
    fn name(&self) -> &'static str {
        "dinf-metal-gated-mul"
    }

    fn cpu_fwd(
        &self,
        _gate: &CpuStorage,
        _gate_layout: &Layout,
        _up: &CpuStorage,
        _up_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused gated multiply is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        gate: &MetalStorage,
        gate_layout: &Layout,
        up: &MetalStorage,
        up_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if gate.dtype() != DType::F32
            || up.dtype() != DType::F32
            || !gate_layout.is_contiguous()
            || !up_layout.is_contiguous()
            || gate_layout.shape() != up_layout.shape()
        {
            candle_core::bail!("fused gated multiply requires matching contiguous f32 tensors")
        }
        if gate.device().id() != up.device().id() {
            candle_core::bail!("fused gated multiply tensors are on different Metal devices")
        }
        let elements = gate_layout.shape().elem_count();
        let elements_u32 = checked_u32(elements, "fused gated multiply element count")?;
        let gate_offset = checked_buffer_offset(
            gate_layout,
            DType::F32.size_in_bytes(),
            "fused gated multiply gate",
        )?;
        let up_offset = checked_buffer_offset(
            up_layout,
            DType::F32.size_in_bytes(),
            "fused gated multiply up",
        )?;
        let device = gate.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label(self.kernel)
            .build()?;
        let pipeline = epilogue_pipeline(device, self.kernel)?;
        let encoder = device.command_encoder()?;
        encoder.set_label(self.kernel);
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(0, Some(gate.buffer()), gate_offset);
        encoder_ref.set_input_buffer(1, Some(up.buffer()), up_offset);
        encoder_ref.set_output_buffer(2, Some(&output), 0);
        encoder_ref.set_bytes(3, &elements_u32);
        encoder_ref.dispatch_threads(
            objc2_metal::MTLSize {
                width: elements,
                height: 1,
                depth: 1,
            },
            candle_metal_kernels::utils::get_block_dims(elements.min(256), 1, 1),
        );
        Ok((
            MetalStorage::new(output, device.clone(), elements, DType::F32),
            gate_layout.shape().clone(),
        ))
    }
}

struct SiluMulClamped {
    limit: f32,
}

impl CustomOp2 for SiluMulClamped {
    fn name(&self) -> &'static str {
        "dinf-metal-silu-mul-clamped"
    }

    fn cpu_fwd(
        &self,
        _gate: &CpuStorage,
        _gate_layout: &Layout,
        _up: &CpuStorage,
        _up_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused clamped SiLU multiply is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        gate: &MetalStorage,
        gate_layout: &Layout,
        up: &MetalStorage,
        up_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if gate.dtype() != DType::F32
            || up.dtype() != DType::F32
            || !gate_layout.is_contiguous()
            || !up_layout.is_contiguous()
            || gate_layout.shape() != up_layout.shape()
            || !self.limit.is_finite()
            || self.limit <= 0.0
        {
            candle_core::bail!(
                "fused clamped SiLU multiply requires matching contiguous f32 tensors and a positive limit"
            )
        }
        if gate.device().id() != up.device().id() {
            candle_core::bail!("fused clamped SiLU multiply tensors are on different Metal devices")
        }
        let elements = gate_layout.shape().elem_count();
        let elements_u32 = checked_u32(elements, "fused clamped SiLU element count")?;
        let gate_offset = checked_buffer_offset(
            gate_layout,
            DType::F32.size_in_bytes(),
            "fused clamped SiLU gate",
        )?;
        let up_offset = checked_buffer_offset(
            up_layout,
            DType::F32.size_in_bytes(),
            "fused clamped SiLU up",
        )?;
        let device = gate.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-silu-mul-clamped")
            .build()?;
        let pipeline = epilogue_pipeline(device, "dinf_silu_mul_clamped_f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-silu-mul-clamped");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(0, Some(gate.buffer()), gate_offset);
        encoder_ref.set_input_buffer(1, Some(up.buffer()), up_offset);
        encoder_ref.set_output_buffer(2, Some(&output), 0);
        encoder_ref.set_bytes(3, &elements_u32);
        encoder_ref.set_bytes(4, &self.limit);
        encoder_ref.dispatch_threads(
            objc2_metal::MTLSize {
                width: elements,
                height: 1,
                depth: 1,
            },
            candle_metal_kernels::utils::get_block_dims(elements.min(256), 1, 1),
        );
        Ok((
            MetalStorage::new(output, device.clone(), elements, DType::F32),
            gate_layout.shape().clone(),
        ))
    }
}

/// `activation(gate) * up` over packed `[..., gate | up]` rows.
struct PackedGated {
    kernel: &'static str,
}

impl CustomOp1 for PackedGated {
    fn name(&self) -> &'static str {
        "dinf-metal-packed-gated"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused packed gated op is only dispatched on Metal")
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        if storage.dtype() != DType::F32 || !layout.is_contiguous() {
            candle_core::bail!("fused packed gated op requires a contiguous f32 tensor")
        }
        let mut output_dims = layout.dims().to_vec();
        let fused_dim = output_dims.last_mut().ok_or_else(|| {
            candle_core::Error::Msg("fused packed gated op requires rank >= 1".into())
        })?;
        if *fused_dim == 0 || *fused_dim % 2 != 0 {
            candle_core::bail!("fused packed gated op requires an even non-zero final dimension")
        }
        let intermediate_dim = *fused_dim / 2;
        let intermediate_dim_u32 = checked_u32(
            intermediate_dim,
            "fused packed gated op intermediate dimension",
        )?;
        *fused_dim = intermediate_dim;
        let output_shape = Shape::from(output_dims);
        let elements = output_shape.elem_count();
        let elements_u32 = checked_u32(elements, "fused packed gated op element count")?;
        let input_offset = checked_buffer_offset(
            layout,
            DType::F32.size_in_bytes(),
            "fused packed gated op input",
        )?;
        let device = storage.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label(self.kernel)
            .build()?;
        let pipeline = epilogue_pipeline(device, self.kernel)?;
        let encoder = device.command_encoder()?;
        encoder.set_label(self.kernel);
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(0, Some(storage.buffer()), input_offset);
        encoder_ref.set_output_buffer(1, Some(&output), 0);
        encoder_ref.set_bytes(2, &intermediate_dim_u32);
        encoder_ref.set_bytes(3, &elements_u32);
        encoder_ref.dispatch_threads(
            objc2_metal::MTLSize {
                width: elements,
                height: 1,
                depth: 1,
            },
            candle_metal_kernels::utils::get_block_dims(elements.min(256), 1, 1),
        );
        Ok((
            MetalStorage::new(output, device.clone(), elements, DType::F32),
            output_shape,
        ))
    }
}

struct WeightedRouteSum {
    ordered: bool,
    /// `(ids, scales)`: multiply each route weight by `scales[ids[route]]`.
    expert_scales: Option<(Tensor, Tensor)>,
}

impl CustomOp2 for WeightedRouteSum {
    fn name(&self) -> &'static str {
        "dinf-metal-weighted-route-sum"
    }

    fn cpu_fwd(
        &self,
        _routes: &CpuStorage,
        _routes_layout: &Layout,
        _weights: &CpuStorage,
        _weights_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused weighted route sum is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        routes: &MetalStorage,
        routes_layout: &Layout,
        weights: &MetalStorage,
        weights_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let weights_dtype_ok = weights.dtype() == DType::F32
            || (self.expert_scales.is_some() && weights.dtype() == DType::U32);
        if routes.dtype() != DType::F32
            || !weights_dtype_ok
            || !routes_layout.is_contiguous()
            || !weights_layout.is_contiguous()
        {
            candle_core::bail!("weighted route sum requires contiguous f32 tensors")
        }
        let (tokens, top_k, hidden_dim) = routes_layout.shape().dims3()?;
        if tokens == 0
            || top_k == 0
            || hidden_dim == 0
            || weights_layout.shape().dims2()? != (tokens, top_k)
        {
            candle_core::bail!("weighted route sum geometry differs")
        }
        if routes.device().id() != weights.device().id() {
            candle_core::bail!("weighted route sum tensors are on different Metal devices")
        }
        let elements = checked_product(tokens, hidden_dim, "weighted route output size")?;
        let elements_u32 = checked_u32(elements, "weighted route element count")?;
        checked_u32(
            checked_product(elements, top_k, "weighted route input size")?,
            "weighted route input size",
        )?;
        let top_k_u32 = checked_u32(top_k, "weighted route top-k")?;
        let hidden_dim_u32 = checked_u32(hidden_dim, "weighted route hidden dimension")?;
        let routes_offset = checked_buffer_offset(
            routes_layout,
            DType::F32.size_in_bytes(),
            "weighted route input",
        )?;
        let weights_offset = checked_buffer_offset(
            weights_layout,
            DType::F32.size_in_bytes(),
            "weighted route weights",
        )?;
        let device = routes.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-weighted-route-sum")
            .build()?;
        let pipeline = epilogue_pipeline(
            device,
            if self.expert_scales.is_some() {
                "dinf_expert_scaled_route_sum_f32"
            } else if self.ordered {
                "dinf_ordered_weighted_route_sum_f32"
            } else {
                "dinf_weighted_route_sum_f32"
            },
        )?;
        let scale_guards = self
            .expert_scales
            .as_ref()
            .map(|(ids, scales)| (ids.storage_and_layout(), scales.storage_and_layout()));
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-weighted-route-sum");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(0, Some(routes.buffer()), routes_offset);
        encoder_ref.set_input_buffer(1, Some(weights.buffer()), weights_offset);
        encoder_ref.set_output_buffer(2, Some(&output), 0);
        encoder_ref.set_bytes(3, &top_k_u32);
        encoder_ref.set_bytes(4, &hidden_dim_u32);
        encoder_ref.set_bytes(5, &elements_u32);
        if let Some(((ids, ids_layout), (scales, scales_layout))) = &scale_guards {
            let (candle_core::Storage::Metal(ids), candle_core::Storage::Metal(scales)) =
                (&**ids, &**scales)
            else {
                candle_core::bail!("expert-scaled route sum operands must be on Metal")
            };
            if ids.dtype() != DType::U32
                || scales.dtype() != DType::F32
                || !ids_layout.is_contiguous()
                || !scales_layout.is_contiguous()
                || ids_layout.shape().dims2()? != (tokens, top_k)
            {
                candle_core::bail!("expert-scaled route sum needs [tokens, top_k] u32 IDs")
            }
            encoder_ref.set_input_buffer(6, Some(ids.buffer()), ids_layout.start_offset() * 4);
            encoder_ref.set_input_buffer(
                7,
                Some(scales.buffer()),
                scales_layout.start_offset() * 4,
            );
        }
        encoder_ref.dispatch_threads(
            objc2_metal::MTLSize {
                width: elements,
                height: 1,
                depth: 1,
            },
            candle_metal_kernels::utils::get_block_dims(elements.min(256), 1, 1),
        );
        Ok((
            MetalStorage::new(output, device.clone(), elements, DType::F32),
            Shape::from((tokens, hidden_dim)),
        ))
    }
}

struct NormalizedRouteWeights;

struct SanitizeExpertIds {
    expert_count: usize,
}

impl CustomOp1 for SanitizeExpertIds {
    fn name(&self) -> &'static str {
        "dinf-metal-sanitize-moe-ids"
    }

    fn cpu_fwd(&self, _ids: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("Metal expert-ID sanitization is only dispatched on Metal")
    }

    fn metal_fwd(&self, ids: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        if !matches!(ids.dtype(), DType::U32 | DType::F32) || !layout.is_contiguous() {
            candle_core::bail!("Metal expert-ID sanitization requires contiguous u32 or f32 IDs")
        }
        let (tokens, top_k) = layout.shape().dims2()?;
        let route_elements = checked_product(tokens, top_k, "Metal expert-ID route count")?;
        checked_product(
            route_elements,
            DType::U32.size_in_bytes(),
            "Metal expert-ID buffer size",
        )?;
        if tokens == 0 || top_k == 0 || self.expert_count == 0 || top_k > self.expert_count {
            candle_core::bail!("invalid Metal expert-ID sanitization geometry")
        }
        let tokens_u32 = checked_u32(tokens, "Metal expert-ID token count")?;
        let top_k_u32 = checked_u32(top_k, "Metal expert-ID top-k")?;
        let expert_count_u32 = checked_u32(self.expert_count, "Metal expert count")?;
        let ids_row_stride_u32 = checked_u32(layout.stride()[0], "Metal expert-ID row stride")?;
        let ids_offset =
            checked_buffer_offset(layout, ids.dtype().size_in_bytes(), "Metal expert-ID input")?;
        let device = ids.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(route_elements, DType::U32)
            .with_label("dinf-sanitize-moe-ids")
            .build()?;
        let pipeline = epilogue_pipeline(
            device,
            if ids.dtype() == DType::F32 {
                "dinf_sanitize_moe_ids_f32"
            } else {
                "dinf_sanitize_moe_ids"
            },
        )?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-sanitize-moe-ids");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(0, Some(ids.buffer()), ids_offset);
        encoder_ref.set_output_buffer(1, Some(&output), 0);
        encoder_ref.set_bytes(2, &expert_count_u32);
        encoder_ref.set_bytes(3, &top_k_u32);
        encoder_ref.set_bytes(4, &tokens_u32);
        encoder_ref.set_bytes(5, &ids_row_stride_u32);
        encoder_ref.dispatch_threads(
            objc2_metal::MTLSize {
                width: tokens,
                height: 1,
                depth: 1,
            },
            candle_metal_kernels::utils::get_block_dims(tokens.min(256), 1, 1),
        );
        Ok((
            MetalStorage::new(output, device.clone(), route_elements, DType::U32),
            layout.shape().clone(),
        ))
    }
}

impl CustomOp2 for NormalizedRouteWeights {
    fn name(&self) -> &'static str {
        "dinf-metal-normalized-route-weights"
    }

    fn cpu_fwd(
        &self,
        _probabilities: &CpuStorage,
        _probabilities_layout: &Layout,
        _expert_ids: &CpuStorage,
        _expert_ids_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused route normalization is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        probabilities: &MetalStorage,
        probabilities_layout: &Layout,
        expert_ids: &MetalStorage,
        expert_ids_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if probabilities.dtype() != DType::F32
            || expert_ids.dtype() != DType::U32
            || !probabilities_layout.is_contiguous()
            || !expert_ids_layout.is_contiguous()
        {
            candle_core::bail!(
                "route normalization requires contiguous f32 probabilities and u32 IDs"
            )
        }
        let (tokens, expert_count) = probabilities_layout.shape().dims2()?;
        let (id_tokens, top_k) = expert_ids_layout.shape().dims2()?;
        if tokens != id_tokens
            || tokens == 0
            || expert_count == 0
            || top_k == 0
            || top_k > expert_count
            || top_k > 256
        {
            candle_core::bail!("route normalization geometry differs")
        }
        if probabilities.device().id() != expert_ids.device().id() {
            candle_core::bail!("route normalization tensors are on different Metal devices")
        }
        let elements = checked_product(tokens, top_k, "normalized route output size")?;
        let expert_count_u32 = checked_u32(expert_count, "normalized route expert count")?;
        let top_k_u32 = checked_u32(top_k, "normalized route top-k")?;
        let tokens_u32 = checked_u32(tokens, "normalized route token count")?;
        let probabilities_offset = checked_buffer_offset(
            probabilities_layout,
            DType::F32.size_in_bytes(),
            "normalized route probabilities",
        )?;
        let expert_ids_offset = checked_buffer_offset(
            expert_ids_layout,
            DType::U32.size_in_bytes(),
            "normalized route expert IDs",
        )?;
        let device = probabilities.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-normalized-route-weights")
            .build()?;
        let pipeline = epilogue_pipeline(device, "dinf_normalized_route_weights_f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-normalized-route-weights");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(0, Some(probabilities.buffer()), probabilities_offset);
        encoder_ref.set_input_buffer(1, Some(expert_ids.buffer()), expert_ids_offset);
        encoder_ref.set_output_buffer(2, Some(&output), 0);
        encoder_ref.set_bytes(3, &expert_count_u32);
        encoder_ref.set_bytes(4, &top_k_u32);
        encoder_ref.set_bytes(5, &tokens_u32);
        encoder_ref.dispatch_threads(
            objc2_metal::MTLSize {
                width: tokens,
                height: 1,
                depth: 1,
            },
            candle_metal_kernels::utils::get_block_dims(tokens.min(256), 1, 1),
        );
        Ok((
            MetalStorage::new(output, device.clone(), elements, DType::F32),
            expert_ids_layout.shape().clone(),
        ))
    }
}

pub(super) fn fused_silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    gate.apply_op2_no_bwd(
        up,
        &GatedMul {
            kernel: "dinf_silu_mul_f32",
        },
    )
}

/// Candle's `NewGelu` (tanh approximation) of `gate`, times `up`.
pub(super) fn fused_gelu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    gate.apply_op2_no_bwd(
        up,
        &GatedMul {
            kernel: "dinf_gelu_mul_f32",
        },
    )
}

pub(super) fn fused_silu_mul_clamped(gate: &Tensor, up: &Tensor, limit: f32) -> Result<Tensor> {
    gate.apply_op2_no_bwd(up, &SiluMulClamped { limit })
}

pub(super) fn fused_swiglu(gate_up: &Tensor) -> Result<Tensor> {
    gate_up.apply_op1_no_bwd(&PackedGated {
        kernel: "dinf_swiglu_f32",
    })
}

/// Candle's `NewGelu` of the gate half times the up half of packed
/// `[..., gate | up]` rows.
pub(super) fn fused_geglu(gate_up: &Tensor) -> Result<Tensor> {
    gate_up.apply_op1_no_bwd(&PackedGated {
        kernel: "dinf_geglu_f32",
    })
}

pub(super) fn fused_weighted_route_sum(routes: &Tensor, weights: &Tensor) -> Result<Tensor> {
    routes.apply_op2_no_bwd(
        weights,
        &WeightedRouteSum {
            ordered: false,
            expert_scales: None,
        },
    )
}

/// [`fused_weighted_route_sum`] with each route weight times its expert's
/// output scale. `ids` are sanitized u32 `[tokens, top_k]` route IDs, which
/// must index `scales`.
pub(super) fn fused_expert_scaled_route_sum(
    routes: &Tensor,
    weights: &Tensor,
    ids: &Tensor,
    scales: &Tensor,
) -> Result<Tensor> {
    routes.apply_op2_no_bwd(
        weights,
        &WeightedRouteSum {
            ordered: false,
            expert_scales: Some((ids.clone(), scales.clone())),
        },
    )
}

/// Sum from +0 in route order, with separate F32 multiply/add rounding.
pub(super) fn fused_ordered_weighted_route_sum(
    routes: &Tensor,
    weights: &Tensor,
) -> Result<Tensor> {
    routes.apply_op2_no_bwd(
        weights,
        &WeightedRouteSum {
            ordered: true,
            expert_scales: None,
        },
    )
}

pub(super) fn fused_normalized_route_weights(
    probabilities: &Tensor,
    expert_ids: &Tensor,
) -> Result<Tensor> {
    let (_, expert_count) = probabilities.dims2()?;
    validate_expert_ids(expert_ids, expert_count, "route normalization")?;
    if !expert_ids.device().is_cpu() && !expert_ids.device().is_metal() {
        candle_core::bail!("route normalization requires CPU or Metal expert IDs")
    }
    probabilities.apply_op2_no_bwd(expert_ids, &NormalizedRouteWeights)
}

pub(super) struct MetalIndexedMoeRoutePlan {
    routes: Tensor,
    batch: usize,
    top_k: usize,
    num_experts: usize,
}

pub(super) struct MetalIndexedMoeRouteStats {
    pub routes: usize,
    pub occupied_experts: usize,
    pub route_tiles: usize,
    pub route_tiles_64: usize,
    pub multi_tile_experts: usize,
    pub max_routes_per_expert: usize,
}

impl MetalIndexedMoeRoutePlan {
    pub(super) fn stats(&self) -> Result<MetalIndexedMoeRouteStats> {
        let plan = self
            .routes
            .narrow(0, 0, self.num_experts)?
            .to_device(&candle_core::Device::Cpu)?
            .to_vec1::<u32>()?;
        let counts = plan.as_slice();
        Ok(MetalIndexedMoeRouteStats {
            routes: counts.iter().map(|count| *count as usize).sum(),
            occupied_experts: counts.iter().filter(|count| **count != 0).count(),
            route_tiles: counts
                .iter()
                .map(|count| (*count as usize).div_ceil(32))
                .sum(),
            route_tiles_64: counts
                .iter()
                .map(|count| (*count as usize).div_ceil(64))
                .sum(),
            multi_tile_experts: counts.iter().filter(|count| **count > 32).count(),
            max_routes_per_expert: counts.iter().copied().max().unwrap_or(0) as usize,
        })
    }
}

struct BuildCompactMoeRoutePlan {
    num_experts: usize,
}

impl CustomOp1 for BuildCompactMoeRoutePlan {
    fn name(&self) -> &'static str {
        "dinf-metal-compact-moe-route-plan"
    }

    fn cpu_fwd(&self, _ids: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("compact MoE route plans are only built on Metal")
    }

    fn metal_fwd(&self, ids: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        if ids.dtype() != DType::U32 || !layout.is_contiguous() {
            candle_core::bail!("compact MoE route plans require contiguous u32 expert IDs")
        }
        let (batch, top_k) = layout.shape().dims2()?;
        let route_count = checked_product(batch, top_k, "compact MoE route count")?;
        if batch == 0
            || top_k == 0
            || route_count > u16::MAX as usize
            || self.num_experts == 0
            || self.num_experts > u32::MAX as usize
            || batch > u32::MAX as usize
            || top_k > u32::MAX as usize
            || layout.stride()[0] > u32::MAX as usize
        {
            candle_core::bail!("invalid compact MoE route plan geometry")
        }
        let output_elements = route_plan_elements(self.num_experts, batch)?;
        let _output_elements_u32 = checked_u32(output_elements, "compact MoE route-plan elements")?;
        checked_product(
            output_elements,
            DType::U32.size_in_bytes(),
            "compact MoE route-plan bytes",
        )?;
        let route_map_offset = checked_product(
            self.num_experts,
            DType::U32.size_in_bytes(),
            "compact MoE route-map offset",
        )?;
        let ids_offset =
            checked_buffer_offset(layout, DType::U32.size_in_bytes(), "compact MoE expert IDs")?;
        let device = ids.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, DType::U32)
            .with_label("dinf-indexed-moe-route-plan")
            .build()?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-indexed-moe-route-plan");
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(0, Some(ids.buffer()), ids_offset);
        encoder_ref.set_output_buffer(1, Some(&output), 0);
        encoder_ref.set_output_buffer(2, Some(&output), route_map_offset);
        encoder_ref.set_bytes(3, &(self.num_experts as u32));
        encoder_ref.set_bytes(4, &(batch as u32));
        encoder_ref.set_bytes(5, &(top_k as u32));
        encoder_ref.set_bytes(6, &(layout.stride()[0] as u32));
        dispatch_compact_routes(device, encoder_ref, self.num_experts, batch, top_k)?;
        Ok((
            MetalStorage::new(output, device.clone(), output_elements, DType::U32),
            Shape::from(output_elements),
        ))
    }
}

#[derive(Clone)]
pub(super) struct MetalIndexedMoeMatMul {
    weights: Tensor,
    dtype: GgmlDType,
    num_experts: usize,
    out_dim: usize,
    in_dim: usize,
    prefer_mv: bool,
    native_vector: bool,
}

impl MetalIndexedMoeMatMul {
    pub(super) fn supports_dtype(dtype: GgmlDType) -> bool {
        compact_mm_kernel_name(dtype).is_ok()
    }

    pub(super) fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub(super) fn from_qtensor(
        packed: &QTensor,
        num_experts: usize,
        out_dim: usize,
        in_dim: usize,
        device: &candle_core::Device,
    ) -> Result<Self> {
        if !device.is_metal() {
            candle_core::bail!("indexed MoE Metal weights require a Metal device")
        }
        if packed.shape().dims() != [num_experts, out_dim, in_dim] {
            candle_core::bail!(
                "packed expert shape {:?} differs from [{num_experts}, {out_dim}, {in_dim}]",
                packed.shape()
            )
        }
        compact_mm_kernel_name(packed.dtype())?;
        if !in_dim.is_multiple_of(packed.dtype().block_size()) {
            candle_core::bail!("packed expert input dimension is not quantization aligned")
        }
        if num_experts == 0 || out_dim == 0 || in_dim == 0 {
            candle_core::bail!("packed expert dimensions must be positive")
        }
        if num_experts > u32::MAX as usize
            || out_dim > u32::MAX as usize
            || in_dim > u32::MAX as usize
        {
            candle_core::bail!("packed expert dimensions exceed Metal u32 kernel limits")
        }
        let raw = packed.data()?.into_owned();
        let raw_len = raw.len();
        let row_blocks = in_dim / packed.dtype().block_size();
        let expected = checked_product(row_blocks, packed.dtype().type_size(), "expert row size")?;
        let expected = checked_product(expected, out_dim, "expert matrix size")?;
        let expected = checked_product(expected, num_experts, "packed expert storage size")?;
        if raw_len != expected {
            candle_core::bail!("packed expert storage has {raw_len} bytes, expected {expected}")
        }
        let weights = Tensor::from_vec(raw, raw_len, device)?;
        let metal = device.as_metal_device()?;
        compact_moe_pipeline(metal, "dinf_compact_moe_routes")?;
        compact_moe_pipeline(metal, "dinf_compact_moe_routes_atomic")?;
        compact_moe_pipeline(metal, "dinf_build_moe_route_tiles")?;
        compact_moe_pipeline(metal, compact_mm_kernel_name(packed.dtype())?)?;
        if metal4_indexed_q4_enabled(metal, packed.dtype()) {
            let rows = if metal4_indexed_half_input(packed.dtype()) {
                METAL4_INDEXED_HALF_INPUT_TILE_ROWS
            } else {
                metal4_indexed_q4_tile_rows(0)
            };
            metal4_indexed_q4_pipeline(metal, packed.dtype(), rows)?;
            METAL4_INDEXED_Q4_LOGGED.get_or_init(|| {
                tracing::info!("Metal 4 indexed Q4 MoE matmul enabled");
            });
        }
        Ok(Self {
            weights,
            dtype: packed.dtype(),
            num_experts,
            out_dim,
            in_dim,
            native_vector: false,
            prefer_mv: packed.dtype() != GgmlDType::Q6K
                && mv_kernel_name(packed.dtype()).is_ok()
                && mv_rows_per_threadgroup(packed.dtype())
                    .is_ok_and(|rows| out_dim.is_multiple_of(rows)),
        })
    }

    /// Match the native indexed vector path for this call only.
    pub(super) fn forward_native_vector(
        &self,
        input: &Tensor,
        expert_ids: &Tensor,
    ) -> Result<Tensor> {
        if !matches!(self.dtype, GgmlDType::Q4K | GgmlDType::Q6K)
            || !self
                .out_dim
                .is_multiple_of(mv_rows_per_threadgroup(self.dtype)?)
        {
            candle_core::bail!("native indexed vector dtype/shape mismatch")
        }
        let native = Self {
            native_vector: true,
            ..self.clone()
        };
        native.forward(input, expert_ids)
    }

    /// Sanitize Metal route IDs once for several projections over the same
    /// experts. Accepts u32 IDs or the f32 IDs packed by the fused top-k
    /// route kernels; returns contiguous u32.
    pub(super) fn sanitize_ids(&self, expert_ids: &Tensor) -> Result<Tensor> {
        let (tokens, top_k) = expert_ids.dims2()?;
        if !expert_ids.device().is_metal()
            || tokens == 0
            || top_k == 0
            || top_k > self.num_experts
            || top_k > 256
        {
            candle_core::bail!("invalid Metal expert IDs for sanitization")
        }
        expert_ids
            .contiguous()?
            .apply_op1_no_bwd(&SanitizeExpertIds {
                expert_count: self.num_experts,
            })
    }

    /// [`Self::forward`] for IDs already returned by [`Self::sanitize_ids`]
    /// for this expert count.
    pub(super) fn forward_sanitized(&self, input: &Tensor, expert_ids: &Tensor) -> Result<Tensor> {
        let (_, top_k) = expert_ids.dims2()?;
        validate_expert_ids(expert_ids, self.num_experts, "indexed MoE")?;
        self.forward_impl(input, expert_ids, top_k, false, false, true)
    }

    /// [`Self::forward_native_vector`] for IDs from [`Self::sanitize_ids`].
    pub(super) fn forward_native_vector_sanitized(
        &self,
        input: &Tensor,
        expert_ids: &Tensor,
    ) -> Result<Tensor> {
        if !matches!(self.dtype, GgmlDType::Q4K | GgmlDType::Q6K)
            || !self
                .out_dim
                .is_multiple_of(mv_rows_per_threadgroup(self.dtype)?)
        {
            candle_core::bail!("native indexed vector dtype/shape mismatch")
        }
        let native = Self {
            native_vector: true,
            ..self.clone()
        };
        native.forward_sanitized(input, expert_ids)
    }

    pub(super) fn forward(&self, input: &Tensor, expert_ids: &Tensor) -> Result<Tensor> {
        let (_, top_k) = expert_ids.dims2()?;
        let expert_ids = sanitize_expert_ids(expert_ids, self.num_experts, "indexed MoE")?;
        self.forward_impl(input, &expert_ids, top_k, false, false, true)
    }

    /// Keep short prompt chunks on the same matrix kernel as long chunks.
    pub(super) fn forward_matrix(&self, input: &Tensor, expert_ids: &Tensor) -> Result<Tensor> {
        let matrix = Self {
            prefer_mv: false,
            native_vector: false,
            ..self.clone()
        };
        matrix.forward(input, expert_ids)
    }

    pub(super) fn prepare_route_plan(
        &self,
        expert_ids: &Tensor,
    ) -> Result<MetalIndexedMoeRoutePlan> {
        let (batch, top_k) = expert_ids.dims2()?;
        if batch == 0
            || top_k == 0
            || expert_ids.dtype() != DType::U32
            || checked_product(batch, top_k, "indexed MoE route count")? > u16::MAX as usize
        {
            candle_core::bail!("invalid indexed MoE expert IDs")
        }
        let expert_ids =
            sanitize_expert_ids(expert_ids, self.num_experts, "indexed MoE route plan")?;
        let routes = expert_ids
            .contiguous()?
            .apply_op1_no_bwd(&BuildCompactMoeRoutePlan {
                num_experts: self.num_experts,
            })?;
        Ok(MetalIndexedMoeRoutePlan {
            routes,
            batch,
            top_k,
            num_experts: self.num_experts,
        })
    }

    pub(super) fn forward_with_route_plan(
        &self,
        input: &Tensor,
        plan: &MetalIndexedMoeRoutePlan,
    ) -> Result<Tensor> {
        self.validate_route_plan(input, plan)?;
        self.forward_impl(input, &plan.routes, plan.top_k, true, false, false)
    }

    /// Compile the packed GEGLU expert kernel ahead of the first prompt.
    pub(super) fn prepare_geglu(&self, device: &candle_core::Device) -> Result<()> {
        let device = device.as_metal_device()?;
        if self.out_dim.is_multiple_of(2)
            && metal4_indexed_half_input(self.dtype)
            && metal4_indexed_q4_enabled(device, self.dtype)
        {
            metal4_indexed_geglu_pipeline(device, self.dtype, METAL4_INDEXED_HALF_INPUT_TILE_ROWS)?;
        }
        Ok(())
    }

    /// Packed `[gate | up]` experts: `gelu(input * gate^T) * (input * up^T)`
    /// in one dispatch on the Metal 4 matrix kernel, or `None` when that
    /// kernel does not apply.
    pub(super) fn forward_geglu_with_route_plan(
        &self,
        input: &Tensor,
        plan: &MetalIndexedMoeRoutePlan,
    ) -> Result<Option<Tensor>> {
        let device = input.device().as_metal_device()?;
        if !self.out_dim.is_multiple_of(2)
            || !metal4_indexed_half_input(self.dtype)
            || !metal4_indexed_q4_enabled(device, self.dtype)
        {
            return Ok(None);
        }
        self.validate_route_plan(input, plan)?;
        let (batch, input_rows, in_dim) = input.dims3()?;
        if in_dim != self.in_dim || input_rows == 0 {
            candle_core::bail!("indexed MoE input geometry differs")
        }
        if checked_product(batch, plan.top_k, "indexed MoE route count")? > u16::MAX as usize {
            candle_core::bail!("indexed MoE batch is too large for Metal row IDs")
        }
        let op = IndexedMoe {
            dtype: self.dtype,
            num_experts: self.num_experts,
            out_dim: self.out_dim,
            in_dim: self.in_dim,
            prefer_mv: false,
            native_vector: false,
            top_k: plan.top_k,
            route_plan: true,
            relu_squared: false,
            fused_up: None,
            geglu_rows: true,
        };
        Ok(Some(self.weights.apply_op3_no_bwd(
            &input.contiguous()?,
            &plan.routes.contiguous()?,
            &op,
        )?))
    }

    /// SwiGLU experts: `silu(input * gate^T) * (input * up^T)` in one fused
    /// dispatch, with `self` as the gate. Returns `None` when this device or
    /// type has no fused kernel.
    pub(super) fn forward_gate_up_silu_with_route_plan(
        &self,
        up: &Self,
        input: &Tensor,
        plan: &MetalIndexedMoeRoutePlan,
    ) -> Result<Option<Tensor>> {
        let device = input.device().as_metal_device()?;
        if up.dtype != self.dtype
            || up.num_experts != self.num_experts
            || up.out_dim != self.out_dim
            || up.in_dim != self.in_dim
            || !metal4_indexed_half_input(self.dtype)
            || !metal4_indexed_q4_enabled(device, self.dtype)
        {
            return Ok(None);
        }
        self.validate_route_plan(input, plan)?;
        let (batch, input_rows, in_dim) = input.dims3()?;
        if in_dim != self.in_dim || input_rows == 0 {
            candle_core::bail!("indexed MoE input geometry differs")
        }
        if checked_product(batch, plan.top_k, "indexed MoE route count")? > u16::MAX as usize {
            candle_core::bail!("indexed MoE batch is too large for Metal row IDs")
        }
        let op = IndexedMoe {
            dtype: self.dtype,
            num_experts: self.num_experts,
            out_dim: self.out_dim,
            in_dim: self.in_dim,
            prefer_mv: false,
            native_vector: false,
            top_k: plan.top_k,
            route_plan: true,
            relu_squared: false,
            fused_up: Some(up.weights.clone()),
            geglu_rows: false,
        };
        Ok(Some(self.weights.apply_op3_no_bwd(
            &input.contiguous()?,
            &plan.routes.contiguous()?,
            &op,
        )?))
    }

    pub(super) fn forward_relu_squared(
        &self,
        input: &Tensor,
        expert_ids: &Tensor,
    ) -> Result<Tensor> {
        let (_, top_k) = expert_ids.dims2()?;
        let expert_ids = sanitize_expert_ids(expert_ids, self.num_experts, "indexed MoE")?;
        self.forward_relu_squared_impl(input, &expert_ids, top_k, false)
    }

    pub(super) fn forward_relu_squared_with_route_plan(
        &self,
        input: &Tensor,
        plan: &MetalIndexedMoeRoutePlan,
    ) -> Result<Tensor> {
        self.validate_route_plan(input, plan)?;
        self.forward_relu_squared_impl(input, &plan.routes, plan.top_k, true)
    }

    fn validate_route_plan(&self, input: &Tensor, plan: &MetalIndexedMoeRoutePlan) -> Result<()> {
        let (batch, _, _) = input.dims3()?;
        let expected_elements = route_plan_elements(self.num_experts, batch)?;
        if batch != plan.batch
            || self.num_experts != plan.num_experts
            || plan.routes.dtype() != DType::U32
            || plan.routes.elem_count() != expected_elements
        {
            candle_core::bail!("indexed MoE route plan geometry differs")
        }
        Ok(())
    }

    fn forward_relu_squared_impl(
        &self,
        input: &Tensor,
        routing: &Tensor,
        top_k: usize,
        route_plan: bool,
    ) -> Result<Tensor> {
        if !route_plan {
            validate_expert_ids(routing, self.num_experts, "indexed MoE")?;
        }
        let (batch, _, _) = input.dims3()?;
        let uses_mv = self.native_vector || (self.prefer_mv && batch < 64);
        let can_fuse = !uses_mv
            && input
                .device()
                .as_metal_device()
                .is_ok_and(|device| metal4_indexed_q4_enabled(device, self.dtype));
        if can_fuse {
            self.forward_impl(input, routing, top_k, route_plan, true, true)
        } else {
            let activated = input.relu()?;
            self.forward_impl(
                &(&activated * &activated)?,
                routing,
                top_k,
                route_plan,
                false,
                true,
            )
        }
    }

    fn forward_impl(
        &self,
        input: &Tensor,
        routing: &Tensor,
        top_k: usize,
        route_plan: bool,
        relu_squared: bool,
        ids_validated: bool,
    ) -> Result<Tensor> {
        let (batch, input_rows, in_dim) = input.dims3()?;
        if in_dim != self.in_dim || input_rows == 0 || top_k == 0 {
            candle_core::bail!("indexed MoE input geometry differs")
        }
        if routing.dtype() != DType::U32 {
            candle_core::bail!("indexed MoE routing data must be u32")
        }
        if route_plan {
            if routing.elem_count() != route_plan_elements(self.num_experts, batch)? {
                candle_core::bail!("indexed MoE route plan geometry differs")
            }
        } else if routing.dims2()? != (batch, top_k) {
            candle_core::bail!("indexed MoE expert ID geometry differs")
        } else if !ids_validated {
            validate_expert_ids(routing, self.num_experts, "indexed MoE")?;
        }
        let max_rows = checked_product(batch, top_k, "indexed MoE route count")?;
        if max_rows > u16::MAX as usize {
            candle_core::bail!("indexed MoE batch is too large for Metal row IDs")
        }
        let op = IndexedMoe {
            dtype: self.dtype,
            num_experts: self.num_experts,
            out_dim: self.out_dim,
            in_dim: self.in_dim,
            prefer_mv: self.prefer_mv,
            native_vector: self.native_vector,
            top_k,
            route_plan,
            relu_squared,
            fused_up: None,
            geglu_rows: false,
        };
        self.weights
            .apply_op3_no_bwd(&input.contiguous()?, &routing.contiguous()?, &op)
    }
}

struct IndexedMoe {
    dtype: GgmlDType,
    num_experts: usize,
    out_dim: usize,
    in_dim: usize,
    prefer_mv: bool,
    native_vector: bool,
    top_k: usize,
    route_plan: bool,
    relu_squared: bool,
    /// Up-projection weights for the fused SwiGLU gate/up kernel; `self`'s
    /// weights are the gate.
    fused_up: Option<Tensor>,
    /// Weights are packed `[gate | up]` rows; write `gelu(gate) * up`, half
    /// as many columns as `out_dim`.
    geglu_rows: bool,
}

impl CustomOp3 for IndexedMoe {
    fn name(&self) -> &'static str {
        "dinf-metal-indexed-quantized-moe"
    }

    fn cpu_fwd(
        &self,
        _weights: &CpuStorage,
        _weights_layout: &Layout,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _ids: &CpuStorage,
        _ids_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("indexed quantized MoE is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        weights: &MetalStorage,
        weights_layout: &Layout,
        input: &MetalStorage,
        input_layout: &Layout,
        routing: &MetalStorage,
        routing_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if weights.dtype() != DType::U8
            || input.dtype() != DType::F32
            || routing.dtype() != DType::U32
            || !weights_layout.is_contiguous()
            || !input_layout.is_contiguous()
            || !routing_layout.is_contiguous()
        {
            candle_core::bail!("indexed MoE requires contiguous u8/f32/u32 Metal inputs")
        }
        let (batch, input_rows, in_dim) = input_layout.shape().dims3()?;
        let top_k = self.top_k;
        if batch > u32::MAX as usize
            || input_rows > u32::MAX as usize
            || top_k > u32::MAX as usize
            || routing_layout.stride()[0] > u32::MAX as usize
        {
            candle_core::bail!("indexed MoE runtime geometry exceeds Metal u32 kernel limits")
        }
        let route_plan_elements = route_plan_elements(self.num_experts, batch)?;
        if in_dim != self.in_dim
            || if self.route_plan {
                routing_layout.shape().elem_count() != route_plan_elements
            } else {
                routing_layout.shape().dims2()? != (batch, top_k)
            }
        {
            candle_core::bail!("indexed MoE Metal input geometry differs")
        }
        let total_ids = checked_product(batch, top_k, "indexed MoE route count")?;
        let batch_u32 = checked_u32(batch, "indexed MoE batch")?;
        let input_rows_u32 = checked_u32(input_rows, "indexed MoE input rows")?;
        let top_k_u32 = checked_u32(top_k, "indexed MoE top-k")?;
        let total_ids_u32 = checked_u32(total_ids, "indexed MoE route count")?;
        let output_dim = if self.geglu_rows {
            self.out_dim / 2
        } else {
            self.out_dim
        };
        let output_shape = Shape::from((batch, top_k, output_dim));
        let output_elements = checked_product(total_ids, output_dim, "indexed MoE output size")?;
        let _output_elements_u32 = checked_u32(output_elements, "indexed MoE output elements")?;
        checked_product(
            output_elements,
            DType::F32.size_in_bytes(),
            "indexed MoE output bytes",
        )?;
        let device = input.device();
        if device.id() != weights.device().id() || device.id() != routing.device().id() {
            candle_core::bail!("indexed MoE tensors are on different Metal devices")
        }
        if self.route_plan && (self.native_vector || (self.prefer_mv && batch < 64)) {
            candle_core::bail!("indexed MoE route plans require compact matrix dispatch")
        }
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, DType::F32)
            .with_label("dinf-indexed-moe")
            .build()?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-indexed-moe");
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(weights.buffer()),
            checked_buffer_offset(
                weights_layout,
                DType::U8.size_in_bytes(),
                "indexed MoE weights",
            )?,
        );
        encoder_ref.set_input_buffer(
            1,
            Some(input.buffer()),
            checked_buffer_offset(
                input_layout,
                DType::F32.size_in_bytes(),
                "indexed MoE input",
            )?,
        );
        encoder_ref.set_output_buffer(2, Some(&output), 0);
        encoder_ref.set_input_buffer(
            3,
            Some(routing.buffer()),
            checked_buffer_offset(
                routing_layout,
                DType::U32.size_in_bytes(),
                "indexed MoE routing",
            )?,
        );

        if self.native_vector || (self.prefer_mv && batch < 64) {
            let pipeline = device
                .kernels()
                .load_pipeline(
                    device.device(),
                    Source::Quantized,
                    mv_kernel_name(self.dtype)?,
                )
                .map_err(candle_core::Error::wrap)?;
            encoder_ref.set_compute_pipeline_state(&pipeline);
            let weight_row_bytes = checked_product(
                self.in_dim / self.dtype.block_size(),
                self.dtype.type_size(),
                "indexed MoE weight-row size",
            )?;
            let weight_expert_bytes =
                checked_product(self.out_dim, weight_row_bytes, "indexed MoE expert size")?;
            if weight_row_bytes > u32::MAX as usize || weight_expert_bytes > u32::MAX as usize {
                candle_core::bail!("indexed MoE weight strides exceed Metal u32 kernel limits")
            }
            let input_element_bytes = DType::F32.size_in_bytes();
            let ids_row_bytes = checked_product(
                routing_layout.stride()[0],
                DType::U32.size_in_bytes(),
                "indexed MoE ID-row stride",
            )?;
            let input_row_bytes = checked_product(
                self.in_dim,
                input_element_bytes,
                "indexed MoE input-row stride",
            )?;
            let input_batch_bytes = checked_product(
                input_rows,
                input_row_bytes,
                "indexed MoE input-batch stride",
            )?;
            let output_row_bytes = checked_product(
                self.out_dim,
                DType::F32.size_in_bytes(),
                "indexed MoE output-row stride",
            )?;
            let parameters = [
                u64::from(top_k_u32),
                u64::from(batch_u32),
                ids_row_bytes as u64,
                self.in_dim as u64,
                self.out_dim as u64,
                self.num_experts as u64,
                self.dtype.type_size() as u64,
                weight_row_bytes as u64,
                weight_expert_bytes as u64,
                self.in_dim as u64,
                u64::from(input_rows_u32),
                u64::from(batch_u32),
                1,
                input_element_bytes as u64,
                input_row_bytes as u64,
                input_batch_bytes as u64,
                self.out_dim as u64,
                u64::from(top_k_u32),
                output_row_bytes as u64,
            ];
            for (index, value) in parameters.iter().enumerate() {
                encoder_ref.set_bytes(4 + index, value);
            }
            encoder_ref.dispatch_thread_groups(
                objc2_metal::MTLSize {
                    width: self.out_dim.div_ceil(mv_rows_per_threadgroup(self.dtype)?),
                    height: 1,
                    depth: total_ids_u32 as usize,
                },
                objc2_metal::MTLSize {
                    width: mv_simdgroups_per_threadgroup(self.dtype)?,
                    height: 32,
                    depth: 1,
                },
            );
        } else {
            let routing_offset = checked_buffer_offset(
                routing_layout,
                DType::U32.size_in_bytes(),
                "indexed MoE routing",
            )?;
            let route_map_elements =
                checked_product(self.num_experts, batch, "indexed MoE route-map size")?;
            let _route_map_elements_u32 =
                checked_u32(route_map_elements, "indexed MoE route-map elements")?;
            let (expert_counts, route_map, counts_offset, route_map_offset) = if self.route_plan {
                let buffer = Arc::new(routing.buffer().clone());
                let route_map_offset = checked_product(
                    self.num_experts,
                    DType::U32.size_in_bytes(),
                    "indexed MoE route-map offset",
                )?;
                (
                    buffer.clone(),
                    buffer,
                    routing_offset,
                    routing_offset
                        .checked_add(route_map_offset)
                        .ok_or_else(|| {
                            candle_core::Error::Msg("indexed MoE route-map offset overflow".into())
                        })?,
                )
            } else {
                let expert_counts = device
                    .new_buffer_builder()
                    .with_size_for(self.num_experts, DType::U32)
                    .with_label("dinf-indexed-moe-counts")
                    .build()?;
                let route_map = device
                    .new_buffer_builder()
                    .with_size_for(route_map_elements, DType::U32)
                    .with_label("dinf-indexed-moe-route-map")
                    .build()?;
                encoder_ref.set_input_buffer(0, Some(routing.buffer()), routing_offset);
                encoder_ref.set_output_buffer(1, Some(&expert_counts), 0);
                encoder_ref.set_output_buffer(2, Some(&route_map), 0);
                encoder_ref.set_bytes(3, &(self.num_experts as u32));
                encoder_ref.set_bytes(4, &(batch as u32));
                encoder_ref.set_bytes(5, &(top_k as u32));
                encoder_ref.set_bytes(6, &(routing_layout.stride()[0] as u32));
                dispatch_compact_routes(device, encoder_ref, self.num_experts, batch, top_k)?;
                (expert_counts, route_map, 0, 0)
            };

            let weight_row_bytes = checked_product(
                self.in_dim / self.dtype.block_size(),
                self.dtype.type_size(),
                "indexed MoE weight-row size",
            )?;
            let weight_expert_bytes =
                checked_product(self.out_dim, weight_row_bytes, "indexed MoE expert size")?;
            let weight_row_bytes_u32 =
                checked_u32(weight_row_bytes, "indexed MoE weight-row stride")?;
            let weight_expert_bytes_u32 =
                checked_u32(weight_expert_bytes, "indexed MoE expert stride")?;
            let use_metal4 = metal4_indexed_q4_enabled(device, self.dtype);
            if self.relu_squared && !use_metal4 {
                candle_core::bail!("indexed MoE ReLU squared fusion requires Metal 4")
            }
            if self.geglu_rows && !use_metal4 {
                candle_core::bail!("indexed MoE GEGLU fusion requires Metal 4")
            }
            if use_metal4 {
                let half_input = metal4_indexed_half_input(self.dtype);
                let tile_rows = if half_input {
                    METAL4_INDEXED_HALF_INPUT_TILE_ROWS
                } else {
                    metal4_indexed_q4_tile_rows(batch)
                };
                let route_tile: usize = 32;
                let tile_capacity = self
                    .num_experts
                    .checked_add(total_ids.div_ceil(route_tile))
                    .ok_or_else(|| {
                        candle_core::Error::Msg("indexed MoE tile capacity overflow".into())
                    })?;
                let tile_count = device
                    .new_buffer_builder()
                    .with_size_for(1, DType::U32)
                    .with_label("dinf-indexed-moe-tile-count")
                    .build()?;
                let tile_map = device
                    .new_buffer_builder()
                    .with_size_for(
                        checked_product(tile_capacity, 2, "indexed MoE tile-map size")?,
                        DType::U32,
                    )
                    .with_label("dinf-indexed-moe-tile-map")
                    .build()?;
                let pipeline = compact_moe_pipeline(device, "dinf_build_moe_route_tiles")?;
                encoder_ref.set_compute_pipeline_state(&pipeline);
                encoder_ref.set_input_buffer(0, Some(&expert_counts), counts_offset);
                encoder_ref.set_output_buffer(1, Some(&tile_count), 0);
                encoder_ref.set_output_buffer(2, Some(&tile_map), 0);
                encoder_ref.set_bytes(3, &(self.num_experts as u32));
                encoder_ref.set_bytes(4, &(route_tile as u32));
                let threads = self.num_experts.clamp(1, 256).next_multiple_of(32);
                encoder_ref.dispatch_thread_groups(
                    objc2_metal::MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    objc2_metal::MTLSize {
                        width: threads,
                        height: 1,
                        depth: 1,
                    },
                );
                let pipeline = if self.fused_up.is_some() {
                    metal4_indexed_gate_up_pipeline(device, self.dtype, tile_rows)?
                } else if self.geglu_rows {
                    metal4_indexed_geglu_pipeline(device, self.dtype, tile_rows)?
                } else {
                    metal4_indexed_q4_pipeline(device, self.dtype, tile_rows)?
                };
                // Keep the up weights' storage guard alive while encoding.
                let fused_up = self.fused_up.as_ref().map(|up| up.storage_and_layout());
                if let Some((storage, layout)) = fused_up.as_ref() {
                    let candle_core::Storage::Metal(up) = &**storage else {
                        candle_core::bail!("fused gate/up weights must be on Metal")
                    };
                    if !layout.is_contiguous() || up.dtype() != DType::U8 {
                        candle_core::bail!("fused gate/up weights must be contiguous u8")
                    }
                    encoder_ref.set_input_buffer(
                        16,
                        Some(up.buffer()),
                        layout.start_offset() * DType::U8.size_in_bytes(),
                    );
                }
                encoder_ref.set_compute_pipeline_state(&pipeline);
                encoder_ref.set_input_buffer(
                    0,
                    Some(weights.buffer()),
                    checked_buffer_offset(
                        weights_layout,
                        DType::U8.size_in_bytes(),
                        "indexed MoE weights",
                    )?,
                );
                encoder_ref.set_input_buffer(
                    1,
                    Some(input.buffer()),
                    checked_buffer_offset(
                        input_layout,
                        DType::F32.size_in_bytes(),
                        "indexed MoE input",
                    )?,
                );
                encoder_ref.set_output_buffer(2, Some(&output), 0);
                encoder_ref.set_input_buffer(3, Some(&expert_counts), counts_offset);
                encoder_ref.set_input_buffer(4, Some(&route_map), route_map_offset);
                let in_dim_u32 = checked_u32(self.in_dim, "indexed MoE input dimension")?;
                let out_dim_u32 = checked_u32(self.out_dim, "indexed MoE output dimension")?;
                encoder_ref.set_bytes(5, &in_dim_u32);
                encoder_ref.set_bytes(6, &out_dim_u32);
                encoder_ref.set_bytes(7, &batch_u32);
                encoder_ref.set_bytes(8, &top_k_u32);
                encoder_ref.set_bytes(9, &input_rows_u32);
                encoder_ref.set_bytes(10, &weight_row_bytes_u32);
                encoder_ref.set_bytes(11, &weight_expert_bytes_u32);
                encoder_ref.set_bytes(12, &u32::from(self.relu_squared));
                encoder_ref.set_input_buffer(13, Some(&tile_count), 0);
                encoder_ref.set_input_buffer(14, Some(&tile_map), 0);
                // The kernels read their tiles from the queue built above.
                encoder_ref.set_bytes(15, &1u32);
                let route_storage =
                    checked_product(32, DType::U32.size_in_bytes(), "indexed MoE route storage")?;
                let tile_storage = checked_product(
                    checked_product(tile_rows, 32, "indexed MoE tile storage")?,
                    DType::F16.size_in_bytes(),
                    "indexed MoE tile storage",
                )?
                .checked_add(checked_product(
                    checked_product(32, 32, "indexed MoE input tile storage")?,
                    DType::F32.size_in_bytes(),
                    "indexed MoE input tile storage",
                )?)
                .and_then(|bytes| bytes.checked_add(route_storage))
                .ok_or_else(|| {
                    candle_core::Error::Msg("indexed MoE tile storage overflow".into())
                })?;
                let result_storage = checked_product(
                    checked_product(tile_rows, route_tile, "indexed MoE result storage")?,
                    DType::F32.size_in_bytes(),
                    "indexed MoE result storage",
                )?;
                let tile_storage = if half_input {
                    // Weights [rows][64] and activations [32][64] in half,
                    // plus 32 route offsets (see the DINF_HALF_INPUT kernel).
                    if self.fused_up.is_some() {
                        // Gate and up tiles [rows][32] plus activations [32][32].
                        (2 * tile_rows * 32 + 32 * 32) * DType::F16.size_in_bytes()
                            + 32 * DType::U32.size_in_bytes()
                    } else {
                        (tile_rows * 64 + route_tile * 64) * DType::F16.size_in_bytes()
                            + route_tile * DType::U32.size_in_bytes()
                    }
                } else {
                    tile_storage
                };
                encoder_ref.set_threadgroup_memory_length(0, tile_storage.max(result_storage));
                encoder_ref.dispatch_thread_groups(
                    objc2_metal::MTLSize {
                        width: tile_capacity,
                        height: if self.geglu_rows {
                            (self.out_dim / 2).div_ceil(tile_rows / 2)
                        } else {
                            self.out_dim.div_ceil(tile_rows)
                        },
                        depth: 1,
                    },
                    objc2_metal::MTLSize {
                        width: tile_rows * 2,
                        height: 1,
                        depth: 1,
                    },
                );
            } else {
                let pipeline = compact_moe_pipeline(device, compact_mm_kernel_name(self.dtype)?)?;
                encoder_ref.set_compute_pipeline_state(&pipeline);
                encoder_ref.set_input_buffer(
                    0,
                    Some(weights.buffer()),
                    checked_buffer_offset(
                        weights_layout,
                        DType::U8.size_in_bytes(),
                        "indexed MoE weights",
                    )?,
                );
                encoder_ref.set_input_buffer(
                    1,
                    Some(input.buffer()),
                    checked_buffer_offset(
                        input_layout,
                        DType::F32.size_in_bytes(),
                        "indexed MoE input",
                    )?,
                );
                encoder_ref.set_output_buffer(2, Some(&output), 0);
                encoder_ref.set_input_buffer(3, Some(&expert_counts), counts_offset);
                encoder_ref.set_input_buffer(4, Some(&route_map), route_map_offset);
                let input_element_bytes = DType::F32.size_in_bytes();
                let input_row_bytes = checked_product(
                    self.in_dim,
                    input_element_bytes,
                    "indexed MoE input-row stride",
                )?;
                let input_batch_bytes = checked_product(
                    input_rows,
                    input_row_bytes,
                    "indexed MoE input-batch stride",
                )?;
                let parameters = [
                    self.in_dim as u64,
                    weight_row_bytes as u64,
                    weight_expert_bytes as u64,
                    input_rows as u64,
                    batch as u64,
                    input_element_bytes as u64,
                    input_row_bytes as u64,
                    input_batch_bytes as u64,
                    self.out_dim as u64,
                    top_k as u64,
                    self.num_experts as u64,
                ];
                for (index, value) in parameters.iter().enumerate() {
                    encoder_ref.set_bytes(5 + index, value);
                }
                let route_bytes = checked_product(
                    batch,
                    std::mem::size_of::<[u16; 2]>(),
                    "indexed MoE threadgroup route storage",
                )?;
                let threadgroup_bytes = 8192usize
                    .checked_add(route_bytes)
                    .ok_or_else(|| {
                        candle_core::Error::Msg("indexed MoE threadgroup storage overflow".into())
                    })?
                    .next_multiple_of(16);
                encoder_ref.set_threadgroup_memory_length(0, threadgroup_bytes);
                encoder_ref.dispatch_thread_groups(
                    objc2_metal::MTLSize {
                        width: batch.div_ceil(32),
                        height: self.out_dim.div_ceil(64),
                        depth: self.num_experts,
                    },
                    objc2_metal::MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
            }
        }

        Ok((
            MetalStorage::new(output, device.clone(), output_elements, DType::F32),
            output_shape,
        ))
    }
}

fn compact_mm_kernel_name(dtype: GgmlDType) -> Result<&'static str> {
    match dtype {
        GgmlDType::F32 => Ok("dinf_kernel_mul_mm_id_f32_f32"),
        GgmlDType::F16 => Ok("dinf_kernel_mul_mm_id_f16_f32"),
        GgmlDType::Q4_0 => Ok("dinf_kernel_mul_mm_id_q4_0_f32"),
        GgmlDType::Q4_1 => Ok("dinf_kernel_mul_mm_id_q4_1_f32"),
        GgmlDType::Q5_0 => Ok("dinf_kernel_mul_mm_id_q5_0_f32"),
        GgmlDType::Q5_1 => Ok("dinf_kernel_mul_mm_id_q5_1_f32"),
        GgmlDType::Q8_0 => Ok("dinf_kernel_mul_mm_id_q8_0_f32"),
        GgmlDType::Q2K => Ok("dinf_kernel_mul_mm_id_q2_K_f32"),
        GgmlDType::Q3K => Ok("dinf_kernel_mul_mm_id_q3_K_f32"),
        GgmlDType::Q4K => Ok("dinf_kernel_mul_mm_id_q4_K_f32"),
        GgmlDType::Q5K => Ok("dinf_kernel_mul_mm_id_q5_K_f32"),
        GgmlDType::Q6K => Ok("dinf_kernel_mul_mm_id_q6_K_f32"),
        other => candle_core::bail!("indexed MoE does not support {other:?}"),
    }
}

fn mv_kernel_name(dtype: GgmlDType) -> Result<&'static str> {
    match dtype {
        GgmlDType::Q4_0 => Ok("kernel_mul_mv_id_q4_0_f32"),
        GgmlDType::Q4_1 => Ok("kernel_mul_mv_id_q4_1_f32"),
        GgmlDType::Q5_0 => Ok("kernel_mul_mv_id_q5_0_f32"),
        GgmlDType::Q5_1 => Ok("kernel_mul_mv_id_q5_1_f32"),
        GgmlDType::Q8_0 => Ok("kernel_mul_mv_id_q8_0_f32"),
        GgmlDType::Q2K => Ok("kernel_mul_mv_id_q2_K_f32"),
        GgmlDType::Q3K => Ok("kernel_mul_mv_id_q3_K_f32"),
        GgmlDType::Q4K => Ok("kernel_mul_mv_id_q4_K_f32"),
        GgmlDType::Q5K => Ok("kernel_mul_mv_id_q5_K_f32"),
        GgmlDType::Q6K => Ok("kernel_mul_mv_id_q6_K_f32"),
        other => candle_core::bail!("indexed MoE vector kernel does not support {other:?}"),
    }
}

fn mv_rows_per_threadgroup(dtype: GgmlDType) -> Result<usize> {
    match dtype {
        // Two SIMD groups produce four Q4, Q5 or Q8_0 rows each.
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1 | GgmlDType::Q8_0 => {
            Ok(8)
        }
        // Two SIMD groups produce two Q5_K rows each.
        GgmlDType::Q5K => Ok(4),
        // Two SIMD groups produce four Q2_K rows or two Q3_K rows each.
        GgmlDType::Q2K => Ok(8),
        GgmlDType::Q3K => Ok(4),
        // Q4_K's kernel handles four rows in a single SIMD group.
        GgmlDType::Q4K => Ok(4),
        // The cached Q6_K vector kernel emits one row per SIMD group.
        GgmlDType::Q6K => Ok(2),
        other => candle_core::bail!("indexed MoE vector geometry does not support {other:?}"),
    }
}

fn mv_simdgroups_per_threadgroup(dtype: GgmlDType) -> Result<usize> {
    match dtype {
        GgmlDType::Q4_0
        | GgmlDType::Q4_1
        | GgmlDType::Q5_0
        | GgmlDType::Q5_1
        | GgmlDType::Q8_0
        | GgmlDType::Q2K
        | GgmlDType::Q3K
        | GgmlDType::Q5K => Ok(2),
        GgmlDType::Q4K => Ok(1),
        GgmlDType::Q6K => Ok(2),
        other => candle_core::bail!("indexed MoE vector geometry does not support {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use anyhow::Context;
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{DType, Device, Tensor};

    use crate::gguf_loader::{LayerRange, load_stage_tensors};

    use super::{
        MetalIndexedMoeMatMul, fused_normalized_route_weights, fused_silu_mul,
        fused_silu_mul_clamped, fused_swiglu, fused_weighted_route_sum, sanitize_expert_ids,
    };

    #[test]
    fn fused_clamped_silu_matches_candle() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let gate = Tensor::new(&[-20.0f32, -1.0, 2.0, 20.0], &device)?;
        let up = Tensor::new(&[-20.0f32, 3.0, -4.0, 20.0], &device)?;
        let actual = fused_silu_mul_clamped(&gate, &up, 10.0)?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        let expected = (gate.clamp(f64::NEG_INFINITY, 10.0)?.silu()? * up.clamp(-10.0, 10.0)?)?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
        Ok(())
    }

    #[test]
    fn fused_swiglu_matches_candle_elementwise() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let intermediate_dim = 512;
        for tokens in [1, 6, 256] {
            let values = (0..tokens * intermediate_dim * 2)
                .map(|index| ((index % 251) as f32 - 125.0) / 29.0)
                .collect::<Vec<_>>();
            let gate_up = Tensor::from_vec(values, (1, tokens, intermediate_dim * 2), &device)?;
            let actual = fused_swiglu(&gate_up)?;
            let gate = gate_up.narrow(candle_core::D::Minus1, 0, intermediate_dim)?;
            let up = gate_up.narrow(candle_core::D::Minus1, intermediate_dim, intermediate_dim)?;
            let expected = (gate.silu()? * up)?;
            let actual = actual.flatten_all()?.to_vec1::<f32>()?;
            let expected = expected.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(actual, expected, "SwiGLU differs for {tokens} tokens");
        }
        Ok(())
    }

    #[test]
    fn fused_gelu_mul_matches_candle_new_gelu() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        // Spans the x > 5 identity branch and large negative inputs.
        let gate = Tensor::from_vec(
            (0..4096)
                .map(|index| ((index % 509) as f32 - 254.0) / 23.0)
                .collect::<Vec<_>>(),
            (2, 1, 2048),
            &device,
        )?;
        let up = Tensor::from_vec(
            (0..4096)
                .map(|index| ((index % 113) as f32 - 56.0) / 17.0)
                .collect::<Vec<_>>(),
            (2, 1, 2048),
            &device,
        )?;
        let actual = super::fused_gelu_mul(&gate, &up)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = (candle_core::Module::forward(&candle_nn::Activation::NewGelu, &gate)?
            * &up)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-6 * expected.abs().max(1.0),
                "GELU multiply differs at {index}: {actual} vs {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn fused_route_normalization_matches_left_to_right_reference() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let probabilities = [
            0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75,
            0.85,
        ];
        let ids = [[7u32, 1, 5, 3], [6, 0, 4, 2]];
        let probabilities = Tensor::new(&probabilities, &device)?.reshape((2, 8))?;
        let ids = Tensor::new(&ids, &device)?;
        let output = fused_normalized_route_weights(&probabilities, &ids)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;
        let probabilities = probabilities.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
        let ids = ids.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
        for token in 0..2 {
            let mut expected = ids[token]
                .iter()
                .map(|expert| probabilities[token][*expert as usize])
                .collect::<Vec<_>>();
            let denominator = expected.iter().copied().sum::<f32>().max(6.103_515_6e-5);
            for weight in &mut expected {
                *weight /= denominator;
            }
            for (actual, expected) in output[token].iter().zip(expected) {
                assert!((*actual - expected).abs() <= 1e-7, "{actual} != {expected}");
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "Metal dispatch microbenchmark"]
    fn benchmark_metal_route_normalization_dispatch() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        const EXPERT_COUNT: usize = 128;
        const TOP_K: usize = 6;
        const DISPATCHES_PER_SAMPLE: usize = 64;
        fn run_dispatch(
            probabilities: &Tensor,
            expert_ids: &Tensor,
            two_dispatches: bool,
        ) -> candle_core::Result<Tensor> {
            if two_dispatches {
                let ids = sanitize_expert_ids(expert_ids, EXPERT_COUNT, "benchmark")?;
                probabilities.apply_op2_no_bwd(&ids, &super::NormalizedRouteWeights)
            } else {
                fused_normalized_route_weights(probabilities, expert_ids)
            }
        }

        for tokens in [1usize, 128] {
            let probabilities = (0..tokens * EXPERT_COUNT)
                .map(|index| ((index * 29 % 997) as f32 + 1.0) / 998.0)
                .collect::<Vec<_>>();
            let expert_ids = (0..tokens * TOP_K)
                .map(|index| ((index * 17 + index / TOP_K * 3) % EXPERT_COUNT) as u32)
                .collect::<Vec<_>>();
            let probabilities = Tensor::from_vec(probabilities, (tokens, EXPERT_COUNT), &device)?;
            let expert_ids = Tensor::from_vec(expert_ids, (tokens, TOP_K), &device)?;

            for two_dispatches in [true, false] {
                for _ in 0..20 {
                    for _ in 0..DISPATCHES_PER_SAMPLE {
                        std::hint::black_box(run_dispatch(
                            &probabilities,
                            &expert_ids,
                            two_dispatches,
                        )?);
                    }
                    device.synchronize()?;
                }

                let mut runs = Vec::with_capacity(101);
                for _ in 0..101 {
                    device.synchronize()?;
                    let started = Instant::now();
                    for _ in 0..DISPATCHES_PER_SAMPLE {
                        std::hint::black_box(run_dispatch(
                            &probabilities,
                            &expert_ids,
                            two_dispatches,
                        )?);
                    }
                    device.synchronize()?;
                    runs.push(
                        started.elapsed().as_secs_f64() * 1_000_000.0
                            / DISPATCHES_PER_SAMPLE as f64,
                    );
                }
                runs.sort_by(f64::total_cmp);
                println!(
                    "metal_route_normalization tokens={tokens} path={}\tmedian_us={:.3}\tp25_us={:.3}\tp75_us={:.3}\tmin_us={:.3}\tmax_us={:.3}",
                    if two_dispatches {
                        "two-dispatch"
                    } else {
                        "fused"
                    },
                    runs[runs.len() / 2],
                    runs[runs.len() / 4],
                    runs[runs.len() * 3 / 4],
                    runs[0],
                    runs[runs.len() - 1],
                );
            }
        }
        Ok(())
    }

    #[test]
    fn indexed_moe_rejects_invalid_route_ids_before_dispatch() -> candle_core::Result<()> {
        let op = MetalIndexedMoeMatMul {
            weights: Tensor::zeros(1, DType::U8, &Device::Cpu)?,
            dtype: GgmlDType::Q2K,
            num_experts: 3,
            out_dim: 1,
            in_dim: 1,
            prefer_mv: false,
            native_vector: false,
        };
        let input = Tensor::zeros((1, 1, 1), DType::F32, &Device::Cpu)?;
        let out_of_range = Tensor::new(&[[0u32, 3]], &Device::Cpu)?;
        let error = op
            .forward(&input, &out_of_range)
            .expect_err("out-of-range expert ID should be rejected");
        let message = error.to_string();
        assert!(
            message.contains("expert ID 3 at token 0, route 1 is out of range"),
            "unexpected error: {message}"
        );

        let probabilities = Tensor::zeros((1, 3), DType::F32, &Device::Cpu)?;
        let error = fused_normalized_route_weights(&probabilities, &out_of_range)
            .expect_err("out-of-range route normalization ID should be rejected");
        let message = error.to_string();
        assert!(
            message.contains("route normalization expert ID 3 at token 0, route 1"),
            "unexpected error: {message}"
        );

        let duplicate = Tensor::new(&[[1u32, 1]], &Device::Cpu)?;
        let error = op
            .prepare_route_plan(&duplicate)
            .err()
            .expect("duplicate expert IDs should be rejected before route-map construction");
        let message = error.to_string();
        assert!(
            message.contains("route plan selects expert 1 more than once for token 0"),
            "unexpected error: {message}"
        );

        let error = op
            .forward_relu_squared(&input, &duplicate)
            .expect_err("duplicate expert IDs should be rejected before activation dispatch");
        let message = error.to_string();
        assert!(
            message.contains("indexed MoE selects expert 1 more than once for token 0"),
            "unexpected error: {message}"
        );
        Ok(())
    }

    #[test]
    fn metal_route_ids_are_sanitized_without_host_readback() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let ids = Tensor::new(&[[0u32, 99, 0, 7], [u32::MAX, 3, 3, 2]], &device)?;
        let sanitized = sanitize_expert_ids(&ids, 8, "test")?
            .to_device(&Device::Cpu)?
            .to_vec2::<u32>()?;
        assert_eq!(sanitized, vec![vec![0, 1, 2, 7], vec![0, 3, 1, 2]]);
        assert!(sanitized.iter().flatten().all(|&expert| expert < 8));
        for row in sanitized {
            let mut unique = row;
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), 4);
        }
        Ok(())
    }

    #[test]
    fn fused_route_normalization_repairs_adversarial_ids_like_sanitizer() -> candle_core::Result<()>
    {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };

        fn check_case(
            device: &Device,
            expert_count: usize,
            tokens: usize,
            top_k: usize,
            raw_ids: Vec<u32>,
            expected_ids: Vec<u32>,
        ) -> candle_core::Result<()> {
            let probability_values = (0..tokens * expert_count)
                .map(|index| ((index * 29 % 997) as f32 + 1.0) / 998.0)
                .collect::<Vec<_>>();
            let probabilities =
                Tensor::from_vec(probability_values.clone(), (tokens, expert_count), device)?;
            let ids = Tensor::from_vec(raw_ids, (tokens, top_k), device)?;
            let sanitized = sanitize_expert_ids(&ids, expert_count, "test")?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<u32>()?;
            assert_eq!(sanitized, expected_ids);

            let actual = fused_normalized_route_weights(&probabilities, &ids)?
                .to_device(&Device::Cpu)?
                .to_vec2::<f32>()?;
            for token in 0..tokens {
                let row_ids = &expected_ids[token * top_k..(token + 1) * top_k];
                let mut expected = row_ids
                    .iter()
                    .map(|&expert| probability_values[token * expert_count + expert as usize])
                    .collect::<Vec<_>>();
                let denominator = expected.iter().copied().sum::<f32>().max(6.103_515_6e-5);
                for weight in &mut expected {
                    *weight /= denominator;
                }
                for (actual, expected) in actual[token].iter().zip(expected) {
                    assert!((*actual - expected).abs() <= 1e-7, "{actual} != {expected}");
                }
            }
            Ok(())
        }

        check_case(
            &device,
            8,
            2,
            4,
            vec![0, 99, 0, 7, u32::MAX, 3, 3, 2],
            vec![0, 1, 2, 7, 0, 3, 1, 2],
        )?;
        // Exercise the path for valid expert IDs above the 256-bit repair
        // set, including repeated high IDs interleaved with replacements.
        check_case(
            &device,
            300,
            1,
            4,
            vec![299, 299, 298, 299],
            vec![299, 0, 298, 1],
        )?;
        // Exercise every bit in the bounded replacement set, including bit
        // 31 of its final word.
        check_case(
            &device,
            256,
            1,
            256,
            vec![u32::MAX; 256],
            (0..256).collect(),
        )
    }

    fn route_stress_case(
        device: &Device,
        tokens: usize,
        expert_count: usize,
        top_k: usize,
        iteration: usize,
    ) -> candle_core::Result<(Tensor, Vec<f32>)> {
        // Poison the rows outside both contiguous views. Generate the live
        // probabilities on-device and queue consumers without a CPU readback.
        let mut probabilities = vec![f32::NAN; (tokens + 2) * expert_count];
        let mut raw_ids = vec![u32::MAX; (tokens + 2) * top_k];
        let mut expected = Vec::with_capacity(tokens * top_k);
        for token in 0..tokens {
            let mut used = vec![false; expert_count];
            let start = (token + 1) * expert_count;
            for expert in 0..expert_count {
                let value = ((expert * 29 + token * 17 + iteration * 13) % 997 + 1) as f32 / 998.;
                probabilities[start + expert] = match iteration % 3 {
                    0 => value,
                    1 => value * 1e-8,
                    _ => 0.,
                };
            }
            let row_start = expected.len();
            for route in 0..top_k {
                let raw = match (route + token + iteration) % 5 {
                    0 => u32::MAX,
                    1 => 0,
                    2 => (expert_count - 1) as u32,
                    3 => expert_count as u32,
                    _ => ((route * 17 + token) % expert_count) as u32,
                };
                raw_ids[(token + 1) * top_k + route] = raw;
                let id = if raw as usize >= expert_count || used[raw as usize] {
                    used.iter().position(|used| !used).unwrap()
                } else {
                    raw as usize
                };
                used[id] = true;
                expected.push(probabilities[start + id]);
            }
            let denominator = expected[row_start..]
                .iter()
                .copied()
                .sum::<f32>()
                .max(6.103_515_6e-5);
            for value in &mut expected[row_start..] {
                *value /= denominator;
            }
        }
        let probabilities = Tensor::from_vec(probabilities, (tokens + 2, expert_count), device)?;
        let probabilities = probabilities.affine(1., 0.)?.narrow(0, 1, tokens)?;
        let ids = Tensor::from_vec(raw_ids, (tokens + 2, top_k), device)?.narrow(0, 1, tokens)?;
        let output = fused_normalized_route_weights(&probabilities, &ids)?;
        // This consumer must observe the finished output in the same encoder.
        Ok((output.affine(1., 0.)?, expected))
    }

    fn check_route_stress_output(
        output: Tensor,
        expected: Vec<f32>,
        case: &str,
    ) -> candle_core::Result<()> {
        let (_, top_k) = output.dims2()?;
        let values = output.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(values.len(), expected.len(), "{case}");
        for (i, (actual, expected)) in values.iter().zip(&expected).enumerate() {
            if !actual.is_finite() || (actual - expected).abs() > 1e-7 {
                let start = i / top_k * top_k;
                let row = &values[start..start + top_k];
                let non_finite = row
                    .iter()
                    .enumerate()
                    .filter_map(|(j, value)| (!value.is_finite()).then_some(j))
                    .collect::<Vec<_>>();
                eprintln!(
                    "{case}, row_start={start} top_k={top_k} non_finite={non_finite:?} row_sum={} first={:?} last={:?}",
                    row.iter().copied().sum::<f32>(),
                    &row[..top_k.min(8)],
                    &row[top_k.saturating_sub(8)..]
                );
            }
            assert!(
                actual.is_finite() && (actual - expected).abs() <= 1e-7,
                "{case}, index {i}: {actual} != {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn metal_route_normalization_chains_poisoned_views_and_repairs_full_rows()
    -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for iteration in 0..3 {
            let mut pending = Vec::new();
            for (tokens, experts, top_k) in [
                (1, 8, 1),
                (2, 8, 4),
                (31, 128, 6),
                (32, 256, 8),
                (33, 300, 32),
                (127, 384, 64),
                (128, 256, 128),
                (129, 256, 256),
                (255, 256, 256),
                (256, 256, 256),
                (257, 300, 256),
            ] {
                let (output, expected) =
                    route_stress_case(&device, tokens, experts, top_k, iteration)?;
                pending.push((
                    output,
                    expected,
                    format!("iteration {iteration}, {tokens}x{experts}, top_k={top_k}"),
                ));
            }
            for (output, expected, case) in pending {
                check_route_stress_output(output, expected, &case)?;
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "Metal route-normalization stress run"]
    fn stress_metal_route_normalization_reused_buffers() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for iteration in 0..256 {
            let mut pending = Vec::new();
            for tokens in [1, 2, 33, 128, 257] {
                let (output, expected) = route_stress_case(&device, tokens, 300, 256, iteration)?;
                pending.push((
                    output,
                    expected,
                    format!("iteration {iteration}, tokens={tokens}"),
                ));
            }
            for (output, expected, case) in pending {
                check_route_stress_output(output, expected, &case)?;
            }
        }
        Ok(())
    }

    #[test]
    fn metal_route_normalization_shared_device_concurrent_waits() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        std::thread::scope(|scope| {
            let handles = (0..4)
                .map(|worker| {
                    let device = device.clone();
                    scope.spawn(move || -> candle_core::Result<()> {
                        for iteration in 0..16 {
                            let tokens = [1, 17, 129][iteration % 3];
                            let (output, expected) =
                                route_stress_case(&device, tokens, 300, 256, iteration + worker)?;
                            device.synchronize()?;
                            check_route_stress_output(
                                output,
                                expected,
                                &format!("worker {worker}, iteration {iteration}"),
                            )?;
                        }
                        Ok(())
                    })
                })
                .collect::<Vec<_>>();
            for handle in handles {
                handle.join().unwrap()?;
            }
            Ok(())
        })
    }

    #[test]
    fn fused_moe_epilogue_matches_scalar_reference() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let tokens = 3;
        let top_k = 4;
        let hidden_dim = 64;
        let values = (0..tokens * top_k * hidden_dim)
            .map(|index| ((index % 41) as f32 - 20.0) / 11.0)
            .collect::<Vec<_>>();
        let up_values = (0..values.len())
            .map(|index| ((index % 29) as f32 - 14.0) / 9.0)
            .collect::<Vec<_>>();
        let route_weights = [0.1f32, 0.2, 0.3, 0.4].repeat(tokens);
        let gate = Tensor::from_vec(values.clone(), (tokens, top_k, hidden_dim), &device)?;
        let up = Tensor::from_vec(up_values.clone(), (tokens, top_k, hidden_dim), &device)?;
        let weights = Tensor::from_vec(route_weights.clone(), (tokens, top_k), &device)?;
        let activated = fused_silu_mul(&gate, &up)?;
        let output = fused_weighted_route_sum(&activated, &weights)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;

        for token in 0..tokens {
            for (column, actual) in output[token].iter().enumerate() {
                let expected = (0..top_k)
                    .map(|route| {
                        let index = (token * top_k + route) * hidden_dim + column;
                        let gate = values[index];
                        gate / (1.0 + (-gate).exp())
                            * up_values[index]
                            * route_weights[token * top_k + route]
                    })
                    .sum::<f32>();
                assert!(
                    (*actual - expected).abs() < 1e-5,
                    "{} != {expected}",
                    actual
                );
            }
        }
        Ok(())
    }

    #[test]
    fn indexed_q4_uses_route_rows_for_down_projection() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for dtype in [
            candle_core::quantized::GgmlDType::Q4_0,
            candle_core::quantized::GgmlDType::Q4_1,
        ] {
            let num_experts = 2;
            let out_dim = 64;
            let in_dim = 256;
            let weights = (0..num_experts * out_dim * in_dim)
                .map(|index| ((index % 29) as f32 - 14.0) / 16.0)
                .collect::<Vec<_>>();
            let weights = Tensor::from_vec(weights, (num_experts, out_dim, in_dim), &Device::Cpu)?;
            let packed = QTensor::quantize(&weights, dtype)?;
            let op = MetalIndexedMoeMatMul::from_qtensor(
                &packed,
                num_experts,
                out_dim,
                in_dim,
                &device,
            )?;
            let input_values = (0..2 * 2 * in_dim)
                .map(|index| ((index % 17) as f32 - 8.0) / 9.0)
                .collect::<Vec<_>>();
            let input = Tensor::from_vec(input_values.clone(), (2, 2, in_dim), &device)?;
            let ids = [[0u32, 1], [1, 0]];
            let output = op
                .forward(&input, &Tensor::new(&ids, &device)?)?
                .to_device(&Device::Cpu)?
                .to_vec3::<f32>()?;
            let dequantized = packed.dequantize(&Device::Cpu)?.to_vec3::<f32>()?;

            for batch in 0..2 {
                for route in 0..2 {
                    let expert = ids[batch][route] as usize;
                    for out in 0..out_dim {
                        let expected = (0..in_dim)
                            .map(|inner| {
                                input_values[(batch * 2 + route) * in_dim + inner]
                                    * dequantized[expert][out][inner]
                            })
                            .sum::<f32>();
                        assert!(
                            (output[batch][route][out] - expected).abs() < 0.05,
                            "{} != {expected}",
                            output[batch][route][out]
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn matrix_route_output_matches_across_prompt_row_counts() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (experts, width, output_dim) = (2, 3072, 64);
        let weights = Tensor::from_vec(
            (0..experts * output_dim * width)
                .map(|i| ((i * 13 % 257) as f32 - 128.) / 1024.)
                .collect::<Vec<_>>(),
            (experts, output_dim, width),
            &Device::Cpu,
        )?;
        for dtype in [GgmlDType::Q2K, GgmlDType::Q3K] {
            let packed = QTensor::quantize(&weights, dtype)?;
            let op =
                MetalIndexedMoeMatMul::from_qtensor(&packed, experts, output_dim, width, &device)?;
            let mut expected = None;
            // Check both shared input rows and per-route down-projection rows.
            for input_rows in [1, 2] {
                let row = (0..input_rows * width)
                    .map(|i| ((i * 11 % 127) as f32 - 63.) / 128.)
                    .collect::<Vec<_>>();
                for rows in [1, 2, 8, 63, 64, 65, 128] {
                    let input = Tensor::from_vec(
                        row.repeat(rows + 1),
                        (rows + 1, input_rows, width),
                        &device,
                    )?
                    .narrow(0, 1, rows)?;
                    let ids = Tensor::from_vec([0u32, 1].repeat(rows), (rows, 2), &device)?;
                    let output = op.forward_matrix(&input, &ids)?.to_vec3::<f32>()?;
                    if rows == 1 {
                        expected = Some(output[0].clone());
                    }
                    for lane in output {
                        assert_eq!(lane, *expected.as_ref().unwrap(), "{dtype:?}, {rows} rows");
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn indexed_q2k_matches_dequantized_reference() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let num_experts = 2;
        let out_dim = 64;
        let in_dim = 256;
        let weights = (0..num_experts * out_dim * in_dim)
            .map(|index| ((index % 29) as f32 - 14.0) / 16.0)
            .collect::<Vec<_>>();
        let weights = Tensor::from_vec(weights, (num_experts, out_dim, in_dim), &Device::Cpu)?;
        let packed = QTensor::quantize(&weights, candle_core::quantized::GgmlDType::Q2K)?;
        let op =
            MetalIndexedMoeMatMul::from_qtensor(&packed, num_experts, out_dim, in_dim, &device)?;
        let input_values = (0..2 * in_dim)
            .map(|index| ((index % 17) as f32 - 8.0) / 9.0)
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(input_values.clone(), (2, 1, in_dim), &device)?;
        let ids = Tensor::new(&[[0u32, 1], [1, 0]], &device)?;
        let output = op.forward(&input, &ids)?.to_device(&Device::Cpu)?;

        let dequantized = packed.dequantize(&Device::Cpu)?.to_vec3::<f32>()?;
        let mut expected = vec![vec![vec![0f32; out_dim]; 2]; 2];
        let ids = [[0usize, 1], [1, 0]];
        for batch in 0..2 {
            for route in 0..2 {
                let expert = ids[batch][route];
                for out in 0..out_dim {
                    expected[batch][route][out] = (0..in_dim)
                        .map(|inner| {
                            input_values[batch * in_dim + inner] * dequantized[expert][out][inner]
                        })
                        .sum();
                }
            }
        }
        let output = output.to_vec3::<f32>()?;
        for (actual_batch, expected_batch) in output.iter().zip(expected) {
            for (actual_route, expected_route) in actual_batch.iter().zip(expected_batch) {
                for (actual, expected) in actual_route.iter().zip(expected_route) {
                    assert!((actual - expected).abs() < 0.05, "{actual} != {expected}");
                }
            }
        }
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].len(), 2);
        assert_eq!(output[0][0].len(), out_dim);
        Ok(())
    }

    #[test]
    fn indexed_q2k_uses_route_rows_for_down_projection() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let num_experts = 2;
        let out_dim = 64;
        let in_dim = 256;
        let weights = (0..num_experts * out_dim * in_dim)
            .map(|index| ((index % 31) as f32 - 15.0) / 17.0)
            .collect::<Vec<_>>();
        let weights = Tensor::from_vec(weights, (num_experts, out_dim, in_dim), &Device::Cpu)?;
        let packed = QTensor::quantize(&weights, candle_core::quantized::GgmlDType::Q2K)?;
        let op =
            MetalIndexedMoeMatMul::from_qtensor(&packed, num_experts, out_dim, in_dim, &device)?;
        let input_values = (0..2 * 2 * in_dim)
            .map(|index| ((index % 19) as f32 - 9.0) / 11.0)
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(input_values.clone(), (2, 2, in_dim), &device)?;
        let ids = [[0u32, 1], [1, 0]];
        let ids_tensor = Tensor::new(&ids, &device)?;
        let output = op.forward(&input, &ids_tensor)?.to_device(&Device::Cpu)?;

        let dequantized = packed.dequantize(&Device::Cpu)?.to_vec3::<f32>()?;
        let output = output.to_vec3::<f32>()?;
        for batch in 0..2 {
            for route in 0..2 {
                let expert = ids[batch][route] as usize;
                for out in 0..out_dim {
                    let expected = (0..in_dim)
                        .map(|inner| {
                            input_values[(batch * 2 + route) * in_dim + inner]
                                * dequantized[expert][out][inner]
                        })
                        .sum::<f32>();
                    assert!(
                        (output[batch][route][out] - expected).abs() < 0.05,
                        "{} != {expected}",
                        output[batch][route][out]
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn indexed_q3k_uses_route_rows_for_down_projection() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let num_experts = 2;
        let out_dim = 64;
        let in_dim = 256;
        let weights = (0..num_experts * out_dim * in_dim)
            .map(|index| ((index % 31) as f32 - 15.0) / 17.0)
            .collect::<Vec<_>>();
        let weights = Tensor::from_vec(weights, (num_experts, out_dim, in_dim), &Device::Cpu)?;
        let packed = QTensor::quantize(&weights, candle_core::quantized::GgmlDType::Q3K)?;
        let op =
            MetalIndexedMoeMatMul::from_qtensor(&packed, num_experts, out_dim, in_dim, &device)?;
        let input_values = (0..2 * 2 * in_dim)
            .map(|index| ((index % 19) as f32 - 9.0) / 11.0)
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(input_values.clone(), (2, 2, in_dim), &device)?;
        let ids = [[0u32, 1], [1, 0]];
        let ids_tensor = Tensor::new(&ids, &device)?;
        let output = op.forward(&input, &ids_tensor)?.to_device(&Device::Cpu)?;

        let dequantized = packed.dequantize(&Device::Cpu)?.to_vec3::<f32>()?;
        let output = output.to_vec3::<f32>()?;
        for batch in 0..2 {
            for route in 0..2 {
                let expert = ids[batch][route] as usize;
                for out in 0..out_dim {
                    let expected = (0..in_dim)
                        .map(|inner| {
                            input_values[(batch * 2 + route) * in_dim + inner]
                                * dequantized[expert][out][inner]
                        })
                        .sum::<f32>();
                    assert!(
                        (output[batch][route][out] - expected).abs() < 0.05,
                        "{} != {expected}",
                        output[batch][route][out]
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn indexed_q4k_uses_route_rows_for_down_projection() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let num_experts = 2;
        let out_dim = 64;
        let in_dim = 256;
        let weights = (0..num_experts * out_dim * in_dim)
            .map(|index| ((index % 31) as f32 - 15.0) / 17.0)
            .collect::<Vec<_>>();
        let weights = Tensor::from_vec(weights, (num_experts, out_dim, in_dim), &Device::Cpu)?;
        let packed = QTensor::quantize(&weights, candle_core::quantized::GgmlDType::Q4K)?;
        let op =
            MetalIndexedMoeMatMul::from_qtensor(&packed, num_experts, out_dim, in_dim, &device)?;
        let input_values = (0..2 * 2 * in_dim)
            .map(|index| ((index % 19) as f32 - 9.0) / 11.0)
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(input_values.clone(), (2, 2, in_dim), &device)?;
        let ids = [[0u32, 1], [1, 0]];
        let ids_tensor = Tensor::new(&ids, &device)?;
        let output = op.forward(&input, &ids_tensor)?.to_device(&Device::Cpu)?;

        let dequantized = packed.dequantize(&Device::Cpu)?.to_vec3::<f32>()?;
        let output = output.to_vec3::<f32>()?;
        for batch in 0..2 {
            for route in 0..2 {
                let expert = ids[batch][route] as usize;
                for out in 0..out_dim {
                    let expected = (0..in_dim)
                        .map(|inner| {
                            input_values[(batch * 2 + route) * in_dim + inner]
                                * dequantized[expert][out][inner]
                        })
                        .sum::<f32>();
                    assert!(
                        (output[batch][route][out] - expected).abs() < 0.05,
                        "{} != {expected}",
                        output[batch][route][out]
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn indexed_q8_0_vector_path_matches_dequantized_reference() -> candle_core::Result<()> {
        indexed_vector_path_matches_dequantized_reference(candle_core::quantized::GgmlDType::Q8_0)
    }

    #[test]
    fn indexed_q5_vector_paths_match_dequantized_reference() -> candle_core::Result<()> {
        use candle_core::quantized::GgmlDType;
        for dtype in [GgmlDType::Q5_0, GgmlDType::Q5_1, GgmlDType::Q5K] {
            indexed_vector_path_matches_dequantized_reference(dtype)?;
        }
        Ok(())
    }

    fn indexed_vector_path_matches_dequantized_reference(
        dtype: candle_core::quantized::GgmlDType,
    ) -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (num_experts, out_dim, in_dim) = (3, 72, 256);
        let weights = (0..num_experts * out_dim * in_dim)
            .map(|index| ((index * 7 % 41) as f32 - 20.0) / 23.0)
            .collect::<Vec<_>>();
        let weights = Tensor::from_vec(weights, (num_experts, out_dim, in_dim), &Device::Cpu)?;
        let packed = QTensor::quantize(&weights, dtype)?;
        let op =
            MetalIndexedMoeMatMul::from_qtensor(&packed, num_experts, out_dim, in_dim, &device)?;
        assert!(
            op.prefer_mv,
            "{dtype:?} decode should use the vector kernel"
        );
        let dequantized = packed.dequantize(&Device::Cpu)?.to_vec3::<f32>()?;
        let ids = [[2u32, 0], [1, 2]];
        let ids_tensor = Tensor::new(&ids, &device)?;
        // Shared input per token (gate/up) and one input per route (down).
        for input_routes in [1, 2] {
            let values = (0..2 * input_routes * in_dim)
                .map(|index| ((index % 29) as f32 - 14.0) / 13.0)
                .collect::<Vec<_>>();
            let input = Tensor::from_vec(values.clone(), (2, input_routes, in_dim), &device)?;
            let output = op.forward(&input, &ids_tensor)?.to_vec3::<f32>()?;
            for batch in 0..2 {
                for route in 0..2 {
                    let expert = ids[batch][route] as usize;
                    let row = batch * input_routes + route.min(input_routes - 1);
                    for out in 0..out_dim {
                        let expected = (0..in_dim)
                            .map(|k| values[row * in_dim + k] * dequantized[expert][out][k])
                            .sum::<f32>();
                        let actual = output[batch][route][out];
                        assert!(
                            (actual - expected).abs() <= 1e-3 * expected.abs().max(1.0),
                            "{dtype:?} routes={input_routes} batch={batch} route={route} out={out}: {actual} != {expected}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn indexed_q2k_batch_matches_lane_local_dispatches() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let num_experts = 16;
        let out_dim = 128;
        let in_dim = 512;
        let batch = 4;
        let top_k = 8;
        let weights = (0..num_experts * out_dim * in_dim)
            .map(|index| ((index % 37) as f32 - 18.0) / 21.0)
            .collect::<Vec<_>>();
        let weights = Tensor::from_vec(weights, (num_experts, out_dim, in_dim), &Device::Cpu)?;
        let packed = QTensor::quantize(&weights, candle_core::quantized::GgmlDType::Q2K)?;
        let op =
            MetalIndexedMoeMatMul::from_qtensor(&packed, num_experts, out_dim, in_dim, &device)?;
        let input = (0..batch * top_k * in_dim)
            .map(|index| ((index % 23) as f32 - 11.0) / 13.0)
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(input, (batch, top_k, in_dim), &device)?;
        let ids = (0..batch * top_k)
            .map(|index| ((index * 7 + index / top_k) % num_experts) as u32)
            .collect::<Vec<_>>();
        let ids = Tensor::from_vec(ids, (batch, top_k), &device)?;
        let batched = op.forward(&input, &ids)?.to_device(&Device::Cpu)?;

        for lane in 0..batch {
            let lane_output = op
                .forward(&input.narrow(0, lane, 1)?, &ids.narrow(0, lane, 1)?)?
                .to_device(&Device::Cpu)?;
            let batched_lane = batched.narrow(0, lane, 1)?;
            let lane_values = lane_output.flatten_all()?.to_vec1::<f32>()?;
            let batched_values = batched_lane.flatten_all()?.to_vec1::<f32>()?;
            for (lane_value, batched_value) in lane_values.iter().zip(batched_values) {
                assert_eq!(*lane_value, batched_value);
            }
        }
        Ok(())
    }

    #[test]
    fn indexed_q2k_compact_batch_matches_dequantized_reference() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let num_experts = 4;
        let out_dim = 64;
        let in_dim = 256;
        let batch = 96;
        let top_k = 2;
        let weights = (0..num_experts * out_dim * in_dim)
            .map(|index| ((index % 31) as f32 - 15.0) / 19.0)
            .collect::<Vec<_>>();
        let weights = Tensor::from_vec(weights, (num_experts, out_dim, in_dim), &Device::Cpu)?;
        let packed = QTensor::quantize(&weights, candle_core::quantized::GgmlDType::Q2K)?;
        let op =
            MetalIndexedMoeMatMul::from_qtensor(&packed, num_experts, out_dim, in_dim, &device)?;
        let input_values = (0..batch * in_dim)
            .map(|index| ((index % 23) as f32 - 11.0) / 17.0)
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(input_values.clone(), (batch, 1, in_dim), &device)?;
        let ids = (0..batch).flat_map(|_| [0u32, 1]).collect::<Vec<_>>();
        let ids_tensor = Tensor::from_vec(ids.clone(), (batch, top_k), &device)?;
        let output = op
            .forward(&input, &ids_tensor)?
            .to_device(&Device::Cpu)?
            .to_vec3::<f32>()?;
        let route_plan = op.prepare_route_plan(&ids_tensor)?;
        let planned_output = op
            .forward_with_route_plan(&input, &route_plan)?
            .to_device(&Device::Cpu)?
            .to_vec3::<f32>()?;
        let dequantized = packed.dequantize(&Device::Cpu)?.to_vec3::<f32>()?;

        for token in 0..batch {
            for route in 0..top_k {
                let expert = ids[token * top_k + route] as usize;
                for out in 0..out_dim {
                    let expected = (0..in_dim)
                        .map(|inner| {
                            input_values[token * in_dim + inner] * dequantized[expert][out][inner]
                        })
                        .sum::<f32>();
                    assert!(
                        (output[token][route][out] - expected).abs() < 0.05,
                        "{} != {expected}",
                        output[token][route][out]
                    );
                    assert_eq!(planned_output[token][route][out], output[token][route][out]);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn indexed_metal4_compact_batch_matches_dequantized_reference() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for dtype in [
            candle_core::quantized::GgmlDType::F16,
            candle_core::quantized::GgmlDType::Q4_0,
            candle_core::quantized::GgmlDType::Q4_1,
            candle_core::quantized::GgmlDType::Q5_1,
            candle_core::quantized::GgmlDType::Q8_0,
            candle_core::quantized::GgmlDType::Q4K,
            candle_core::quantized::GgmlDType::Q5K,
            candle_core::quantized::GgmlDType::Q6K,
        ] {
            let num_experts = 4;
            let out_dim = 64;
            let in_dim = 256;
            let batch = 64;
            let top_k = 2;
            let weights = (0..num_experts * out_dim * in_dim)
                .map(|index| ((index % 31) as f32 - 15.0) / 19.0)
                .collect::<Vec<_>>();
            let weights = Tensor::from_vec(weights, (num_experts, out_dim, in_dim), &Device::Cpu)?;
            let packed = QTensor::quantize(&weights, dtype)?;
            let op = MetalIndexedMoeMatMul::from_qtensor(
                &packed,
                num_experts,
                out_dim,
                in_dim,
                &device,
            )?;
            let input_values = (0..batch * top_k * in_dim)
                .map(|index| ((index % 23) as f32 - 11.0) / 17.0)
                .collect::<Vec<_>>();
            let input = Tensor::from_vec(input_values.clone(), (batch, top_k, in_dim), &device)?;
            let ids = (0..batch)
                .flat_map(|token| [token as u32 % 4, (token as u32 + 1) % 4])
                .collect::<Vec<_>>();
            let ids_tensor = Tensor::from_vec(ids.clone(), (batch, top_k), &device)?;
            let output = op
                .forward(&input, &ids_tensor)?
                .to_device(&Device::Cpu)?
                .to_vec3::<f32>()?;
            let fused_output = op
                .forward_relu_squared(&input, &ids_tensor)?
                .to_device(&Device::Cpu)?
                .to_vec3::<f32>()?;
            let route_plan = op.prepare_route_plan(&ids_tensor)?;
            let planned_output = op
                .forward_with_route_plan(&input, &route_plan)?
                .to_device(&Device::Cpu)?
                .to_vec3::<f32>()?;
            let planned_fused_output = op
                .forward_relu_squared_with_route_plan(&input, &route_plan)?
                .to_device(&Device::Cpu)?
                .to_vec3::<f32>()?;
            let dequantized = packed.dequantize(&Device::Cpu)?.to_vec3::<f32>()?;

            for token in 0..batch {
                for route in 0..top_k {
                    let expert = ids[token * top_k + route] as usize;
                    for out in 0..out_dim {
                        let expected = (0..in_dim)
                            .map(|inner| {
                                input_values[(token * top_k + route) * in_dim + inner]
                                    * dequantized[expert][out][inner]
                            })
                            .sum::<f32>();
                        assert!(
                            (output[token][route][out] - expected).abs() < 0.05,
                            "{} != {expected}",
                            output[token][route][out]
                        );
                        if super::metal4_indexed_half_input(dtype) {
                            // Half-staged operands, float accumulation.
                            let half = |v: f32| half::f16::from_f32(v).to_f32() as f64;
                            let staged = (0..in_dim)
                                .map(|inner| {
                                    half(input_values[(token * top_k + route) * in_dim + inner])
                                        * half(dequantized[expert][out][inner])
                                })
                                .sum::<f64>();
                            let error = (f64::from(output[token][route][out]) - staged).abs();
                            assert!(
                                error <= 2e-5 * staged.abs().max(1.0),
                                "{dtype:?}: {} != half-staged {staged}",
                                output[token][route][out]
                            );
                        }
                        assert_eq!(planned_output[token][route][out], output[token][route][out]);
                        let fused_expected = (0..in_dim)
                            .map(|inner| {
                                let value =
                                    input_values[(token * top_k + route) * in_dim + inner].max(0.0);
                                value * value * dequantized[expert][out][inner]
                            })
                            .sum::<f32>();
                        assert!(
                            (fused_output[token][route][out] - fused_expected).abs() < 0.05,
                            "{} != {fused_expected}",
                            fused_output[token][route][out]
                        );
                        assert_eq!(
                            planned_fused_output[token][route][out],
                            fused_output[token][route][out]
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn fused_gate_up_silu_matches_separate_projections() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for dtype in [
            candle_core::quantized::GgmlDType::Q8_0,
            candle_core::quantized::GgmlDType::Q4K,
            candle_core::quantized::GgmlDType::Q6K,
        ] {
            let (experts, out_dim, in_dim, tokens, top_k) =
                (8usize, 200usize, 512usize, 70usize, 3usize);
            let quantized = |salt: usize| -> candle_core::Result<QTensor> {
                let values = (0..experts * out_dim * in_dim)
                    .map(|i| (((i * 13 + salt) % 47) as f32 - 23.0) / 61.0)
                    .collect::<Vec<_>>();
                QTensor::quantize(
                    &Tensor::from_vec(values, (experts, out_dim, in_dim), &Device::Cpu)?,
                    dtype,
                )
            };
            let gate = MetalIndexedMoeMatMul::from_qtensor(
                &quantized(1)?,
                experts,
                out_dim,
                in_dim,
                &device,
            )?;
            let up = MetalIndexedMoeMatMul::from_qtensor(
                &quantized(5)?,
                experts,
                out_dim,
                in_dim,
                &device,
            )?;
            let ids = (0..tokens)
                .flat_map(|t| (0..top_k).map(move |r| ((t * 5 + r * 3) % experts) as u32))
                .collect::<Vec<_>>();
            let ids = Tensor::from_vec(ids, (tokens, top_k), &device)?;
            let input = Tensor::from_vec(
                (0..tokens * in_dim)
                    .map(|i| ((i % 37) as f32 - 18.0) / 19.0)
                    .collect::<Vec<_>>(),
                (tokens, 1, in_dim),
                &device,
            )?;
            let plan = gate.prepare_route_plan(&ids)?;
            let fused = gate
                .forward_gate_up_silu_with_route_plan(&up, &input, &plan)?
                .expect("fused kernel available for this type");
            let separate = fused_silu_mul(
                &gate.forward_with_route_plan(&input, &plan)?,
                &up.forward_with_route_plan(&input, &plan)?,
            )?;
            let diff = (&fused - &separate)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            let peak = separate.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            assert!(
                diff <= 1e-4 * peak.max(1.0),
                "{dtype:?}: max abs {diff} (peak {peak})"
            );
        }
        Ok(())
    }

    #[test]
    fn f32_route_ids_sanitize_like_u32_ids() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let weights = Tensor::zeros((6, 32, 32), DType::F32, &Device::Cpu)?;
        let op = MetalIndexedMoeMatMul::from_qtensor(
            &QTensor::quantize(&weights, candle_core::quantized::GgmlDType::Q8_0)?,
            6,
            32,
            32,
            &device,
        )?;
        // Valid rows, a duplicate, an out-of-range ID and non-integer/negative floats.
        let rows: [[f32; 3]; 4] = [[0., 5., 2.], [3., 3., 1.], [9., 1., 2.], [1.5, -1., 4.]];
        let float_ids = Tensor::new(&rows, &device)?;
        let from_f32 = op.sanitize_ids(&float_ids)?.to_vec2::<u32>()?;
        assert_eq!(
            from_f32,
            vec![vec![0, 5, 2], vec![3, 0, 1], vec![0, 1, 2], vec![0, 1, 4]]
        );
        let u32_rows = [[0u32, 5, 2], [3, 3, 1], [9, 1, 2]];
        let from_u32 = op
            .sanitize_ids(&Tensor::new(&u32_rows, &device)?)?
            .to_vec2::<u32>()?;
        assert_eq!(from_u32, from_f32[..3].to_vec());
        Ok(())
    }

    #[test]
    #[ignore = "manual Metal timing at Qwen3-30B-A3B expert shapes"]
    fn bench_qwen3_moe_expert_prefill() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (experts, top_k, tokens) = (128usize, 8usize, 512usize);
        for (dtype, out_dim, in_dim, label) in [
            (
                candle_core::quantized::GgmlDType::Q4K,
                768,
                2048,
                "gate/up Q4_K",
            ),
            (
                candle_core::quantized::GgmlDType::Q6K,
                2048,
                768,
                "down Q6_K",
            ),
            (
                candle_core::quantized::GgmlDType::Q8_0,
                768,
                2048,
                "gate/up Q8_0",
            ),
        ] {
            let weights = Tensor::randn(0f32, 0.05, (experts, out_dim, in_dim), &Device::Cpu)?;
            let packed = QTensor::quantize(&weights, dtype)?;
            let op =
                MetalIndexedMoeMatMul::from_qtensor(&packed, experts, out_dim, in_dim, &device)?;
            // Spread routes like a real router: pseudo-random distinct experts.
            let ids = (0..tokens)
                .flat_map(|t| {
                    (0..top_k).map(move |r| ((t * 37 + r * 53 + (t * r) % 11) % experts) as u32)
                })
                .collect::<Vec<_>>();
            let ids = Tensor::from_vec(ids, (tokens, top_k), &device)?;
            let rows = if out_dim == 768 { 1 } else { top_k };
            let input = Tensor::randn(0f32, 1.0, (tokens, rows, in_dim), &device)?;
            for _ in 0..3 {
                op.forward(&input, &ids)?;
            }
            device.synchronize()?;
            let start = std::time::Instant::now();
            for _ in 0..20 {
                op.forward(&input, &ids)?;
            }
            device.synchronize()?;
            let ms = start.elapsed().as_secs_f64() * 1e3 / 20.0;
            let flops = 2.0 * (tokens * top_k * out_dim * in_dim) as f64;
            println!("{label}: {ms:.3} ms, {:.1} TFLOPS", flops / ms / 1e9);
            let plan = op.prepare_route_plan(&ids)?;
            device.synchronize()?;
            let guard = crate::architectures::deepseek41::profile::GpuTimingGuard::enable();
            let start = std::time::Instant::now();
            for _ in 0..20 {
                op.forward_with_route_plan(&input, &plan)?;
            }
            device.synchronize()?;
            let ms = start.elapsed().as_secs_f64() * 1e3 / 20.0;
            let gpu = crate::architectures::deepseek41::profile::gpu_timing_snapshot()
                .gpu_elapsed_ns as f64
                / 1e6
                / 20.0;
            drop(guard);
            println!(
                "{label} (shared route plan): {ms:.3} ms wall, {gpu:.3} ms GPU, {:.1} TFLOPS (GPU)",
                flops / gpu / 1e9
            );
            if out_dim == 768 {
                let up_weights =
                    Tensor::randn(0f32, 0.05, (experts, out_dim, in_dim), &Device::Cpu)?;
                let up = MetalIndexedMoeMatMul::from_qtensor(
                    &QTensor::quantize(&up_weights, dtype)?,
                    experts,
                    out_dim,
                    in_dim,
                    &device,
                )?;
                device.synchronize()?;
                let guard = crate::architectures::deepseek41::profile::GpuTimingGuard::enable();
                for _ in 0..20 {
                    op.forward_with_route_plan(&input, &plan)?;
                    up.forward_with_route_plan(&input, &plan)?;
                }
                device.synchronize()?;
                let separate = crate::architectures::deepseek41::profile::gpu_timing_snapshot()
                    .gpu_elapsed_ns as f64
                    / 1e6
                    / 20.0;
                drop(guard);
                println!("{label} separate gate+up (distinct weights): {separate:.3} ms GPU");
                let guard = crate::architectures::deepseek41::profile::GpuTimingGuard::enable();
                for _ in 0..20 {
                    op.forward_gate_up_silu_with_route_plan(&up, &input, &plan)?;
                }
                device.synchronize()?;
                let gpu = crate::architectures::deepseek41::profile::gpu_timing_snapshot()
                    .gpu_elapsed_ns as f64
                    / 1e6
                    / 20.0;
                drop(guard);
                println!(
                    "{label} fused gate+up+silu: {gpu:.3} ms GPU ({:.1} TFLOPS for both)",
                    2.0 * flops / gpu / 1e9
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires DINF_BENCH_NEMOTRON_GGUF and a large local model"]
    fn benchmark_nemotron_q4_indexed_moe() -> anyhow::Result<()> {
        let path = std::env::var("DINF_BENCH_NEMOTRON_GGUF")
            .context("set DINF_BENCH_NEMOTRON_GGUF to the Nemotron GGUF path")?;
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let stage = load_stage_tensors(
            std::path::Path::new(&path),
            LayerRange::new(3, 4, 52),
            &device,
        )?;
        let up_packed = stage
            .tensors
            .get("blk.3.ffn_up_exps.weight")
            .context("missing packed Nemotron expert up projection")?;
        let down_packed = stage
            .tensors
            .get("blk.3.ffn_down_exps.weight")
            .context("missing packed Nemotron expert down projection")?;
        let (num_experts, expert_dim, hidden_dim) = up_packed.shape().dims3()?;
        let (down_experts, down_hidden, down_expert_dim) = down_packed.shape().dims3()?;
        anyhow::ensure!(
            (down_experts, down_hidden, down_expert_dim) == (num_experts, hidden_dim, expert_dim),
            "Nemotron expert projection shapes do not compose"
        );

        eprintln!(
            "experts={num_experts} hidden={hidden_dim} expert_dim={expert_dim}; building F16 controls"
        );
        let up_f16_packed = QTensor::quantize(
            &up_packed.dequantize(&device)?,
            candle_core::quantized::GgmlDType::F16,
        )?;
        let down_f16_packed = QTensor::quantize(
            &down_packed.dequantize(&device)?,
            candle_core::quantized::GgmlDType::F16,
        )?;
        let up_q4 = MetalIndexedMoeMatMul::from_qtensor(
            up_packed.as_ref(),
            num_experts,
            expert_dim,
            hidden_dim,
            &device,
        )?;
        let down_q4 = MetalIndexedMoeMatMul::from_qtensor(
            down_packed.as_ref(),
            num_experts,
            hidden_dim,
            expert_dim,
            &device,
        )?;
        let up_f16 = MetalIndexedMoeMatMul::from_qtensor(
            &up_f16_packed,
            num_experts,
            expert_dim,
            hidden_dim,
            &device,
        )?;
        let down_f16 = MetalIndexedMoeMatMul::from_qtensor(
            &down_f16_packed,
            num_experts,
            hidden_dim,
            expert_dim,
            &device,
        )?;
        let tokens = std::env::var("DINF_BENCH_MOE_TOKENS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(512);
        let repetitions = std::env::var("DINF_BENCH_MOE_REPETITIONS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(10);
        let top_k = stage.metadata.expert_used_count;
        let input = (0..tokens * hidden_dim)
            .map(|index| ((index % 251) as f32 - 125.0) / 127.0)
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(input, (tokens, 1, hidden_dim), &device)?;
        let ids = if (tokens, top_k, num_experts) == (512, 6, 128) {
            let mut counts = vec![0usize; num_experts];
            counts[0] = 502;
            counts[1..=18].fill(72);
            counts[19..=22].fill(50);
            counts[23..=95].fill(12);
            counts[96..=113].fill(11);
            counts
                .into_iter()
                .enumerate()
                .flat_map(|(expert, count)| std::iter::repeat_n(expert as u32, count))
                .collect::<Vec<_>>()
        } else {
            (0..tokens)
                .flat_map(|token| {
                    (0..top_k).map(move |route| ((token * 17 + route * 23) % num_experts) as u32)
                })
                .collect::<Vec<_>>()
        };
        anyhow::ensure!(ids.len() == tokens * top_k, "benchmark route count differs");
        let ids = Tensor::from_vec(ids, (tokens, top_k), &device)?;

        let route_plan = up_q4.prepare_route_plan(&ids)?;
        let up_output = up_q4.forward_with_route_plan(&input, &route_plan)?;
        let down_output = down_q4.forward_relu_squared_with_route_plan(&up_output, &route_plan)?;
        let up_f16_output = up_f16.forward_with_route_plan(&input, &route_plan)?;
        let down_f16_output =
            down_f16.forward_relu_squared_with_route_plan(&up_output, &route_plan)?;
        device.synchronize()?;
        drop(down_output);
        drop(up_f16_output);
        drop(down_f16_output);

        let measure_pair = |label: &str,
                            q4_op: &MetalIndexedMoeMatMul,
                            f16_op: &MetalIndexedMoeMatMul,
                            input: &Tensor,
                            fuse_relu_squared: bool|
         -> anyhow::Result<()> {
            let mut q4_runs = Vec::with_capacity(repetitions);
            let mut f16_runs = Vec::with_capacity(repetitions);
            for repetition in 0..repetitions {
                let order = if repetition % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                };
                for use_f16 in order {
                    let op = if use_f16 { f16_op } else { q4_op };
                    device.synchronize()?;
                    let started = Instant::now();
                    let output = if fuse_relu_squared {
                        op.forward_relu_squared_with_route_plan(input, &route_plan)?
                    } else {
                        op.forward_with_route_plan(input, &route_plan)?
                    };
                    device.synchronize()?;
                    if use_f16 {
                        f16_runs.push(started.elapsed());
                    } else {
                        q4_runs.push(started.elapsed());
                    }
                    drop(output);
                }
            }
            for (kind, runs) in [("q4", &mut q4_runs), ("f16", &mut f16_runs)] {
                runs.sort_unstable();
                let median = runs[runs.len() / 2];
                println!(
                    "{label}_{kind}\ttokens={tokens}\tmedian_ms={:.3}\truns_ms={}",
                    median.as_secs_f64() * 1_000.0,
                    runs.iter()
                        .map(|run| format!("{:.3}", run.as_secs_f64() * 1_000.0))
                        .collect::<Vec<_>>()
                        .join(",")
                );
            }
            Ok(())
        };
        measure_pair("indexed_up", &up_q4, &up_f16, &input, false)?;
        measure_pair("indexed_down_relu2", &down_q4, &down_f16, &up_output, true)?;
        Ok(())
    }
}
