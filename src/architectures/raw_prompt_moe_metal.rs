//! Prompt MoE over raw expert banks (NVFP4, MXFP8, IQ4_NL) on Metal 4, with
//! a chunk's routes sorted by expert.
//!
//! A router sends most of a chunk's routes to a few experts and a handful to
//! most of the others. Tiles of a fixed 32 routes dequantize a busy expert's
//! weights many times over and leave the quiet experts' tiles mostly empty,
//! and reading each expert's weights once is a large part of the chunk's
//! time. So one kernel sorts the routes by expert and queues each expert's
//! sorted rows by how many it has:
//!
//! - a few routes: a vector kernel, which streams the weights fastest and
//!   needs no matrix unit for so little arithmetic;
//! - more: tile kernels of 16, 32 and up to 128 routes on the matrix units,
//!   which dequantize the weights once for all the routes of a tile.
//!
//! The inputs are gathered once, in half, in sorted order, so the tile
//! kernels read activations straight from device memory, as the dense prompt
//! kernels do, and only the dequantized weights pass through threadgroup
//! memory. The `[gate; up]` kernels apply SiTU in their epilogues and write
//! sorted activations for the `down` kernels, whose sorted rows the weighted
//! route sum reads back through the inverse map. The queues of a bank write
//! disjoint rows, so their dispatches overlap.
//!
//! The tile arithmetic is the unsorted tile kernels': weights and activations
//! in half (the weights exactly), F32 accumulation, and F32 scale, SiTU and
//! route sum. The vector kernel's is decode's, all in F32.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use candle_core::metal_backend::DeviceId;
use candle_core::{DType, MetalStorage, Result, Shape, Storage, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputeCommandEncoder, ComputePipeline, Library};
use objc2_metal::MTLSize;

use super::raw_experts::RawFormat;
use super::raw_experts_metal::{metal_view, to_u32};

/// One tile kernel's shape. A threadgroup takes runs of one expert's sorted
/// routes, `subs` of `sub` routes and then `tails` of `tail`, each run with
/// its own accumulator, so the expert's weights are dequantized once for all
/// of them and a partly filled tile multiplies only the runs it reaches. It
/// covers `rows` weight rows (gate rows with their up rows for the pair
/// kernel), `k` input columns a step, on `simdgroups` of 32 threads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Geometry {
    sub: usize,
    subs: usize,
    tail: usize,
    tails: usize,
    rows: usize,
    k: usize,
    simdgroups: usize,
}

impl Geometry {
    fn threads(self) -> usize {
        self.simdgroups * 32
    }

    /// Routes one tile holds.
    fn capacity(self) -> usize {
        self.sub * self.subs + self.tail * self.tails
    }

    /// Each run's routes and first route within the tile.
    fn runs(self) -> Vec<(usize, usize)> {
        let sizes = std::iter::repeat_n(self.sub, self.subs)
            .chain(std::iter::repeat_n(self.tail, self.tails));
        let mut first = 0;
        sizes
            .map(|size| {
                first += size;
                (size, first - size)
            })
            .collect()
    }
}

/// Tile queues a chunk's experts are split over, besides the tiny one.
const MAX_QUEUES: usize = 4;

/// The tile kernels of each bank by ascending capacity: an expert's routes
/// go to the first kernel that holds what is left of them, or fill a tile of
/// the last. `pair` are the `[gate; up]` kernels with SiTU, `down` the
/// others; kernels at one index share a queue, so they hold the same routes.
///
/// Measured with the official Inkling 2 Mini checkpoint: a
/// thread should dequantize two 16-weight groups a step (more starve the
/// threadgroup, one gives up the paired loads), 64-route runs multiply
/// faster than twice as many 32-route ones, runs of other than 16, 32 or 64
/// routes are much slower, and eight simdgroups a tile are slower than four.
struct Tiles {
    pair: &'static [Geometry],
    down: &'static [Geometry],
}

const PAIR_TILES: [Geometry; 4] = [
    Geometry {
        sub: 16,
        subs: 1,
        tail: 0,
        tails: 0,
        rows: 32,
        k: 64,
        simdgroups: 4,
    },
    Geometry {
        sub: 32,
        subs: 1,
        tail: 0,
        tails: 0,
        rows: 32,
        k: 64,
        simdgroups: 4,
    },
    Geometry {
        sub: 64,
        subs: 1,
        tail: 32,
        tails: 1,
        rows: 32,
        k: 64,
        simdgroups: 4,
    },
    Geometry {
        sub: 64,
        subs: 3,
        tail: 0,
        tails: 0,
        rows: 32,
        k: 64,
        simdgroups: 4,
    },
];
const DOWN_TILES: [Geometry; 4] = [
    Geometry {
        sub: 16,
        subs: 1,
        tail: 0,
        tails: 0,
        rows: 64,
        k: 64,
        simdgroups: 4,
    },
    Geometry {
        sub: 32,
        subs: 1,
        tail: 0,
        tails: 0,
        rows: 64,
        k: 64,
        simdgroups: 4,
    },
    Geometry {
        sub: 64,
        subs: 1,
        tail: 32,
        tails: 1,
        rows: 64,
        k: 64,
        simdgroups: 4,
    },
    Geometry {
        sub: 64,
        subs: 3,
        tail: 0,
        tails: 0,
        rows: 64,
        k: 64,
        simdgroups: 4,
    },
];

/// Experts with at most this many routes in a chunk run the vector kernel:
/// its threads stream a cold expert's weights about 1.6 times as fast as the
/// tile kernels', which matters more than the matrix units for so few
/// routes. Its arithmetic is decode's (weights and activations in F32).
const TINY_ROUTES: usize = 4;

fn tiles() -> Tiles {
    Tiles {
        pair: &PAIR_TILES,
        down: &DOWN_TILES,
    }
}

const SORTED_MSL: &str = r#"
// Queue q's tiles hold up to caps[q] routes and start at tiles[bases[q]].
struct SortedPlanParams {
  uint routes; uint experts; uint queues; uint tiny;
  uint caps[4]; uint bases[4]; uint grid_rows[10];
};
struct SortedTile { uint expert; uint base; uint valid; uint pad; };

// Sorts routes by expert (a count, a prefix sum, then a placement pass), and
// queues each expert's sorted rows as tiles: all of them as one tiny tile
// when there are at most `tiny`; otherwise what is left goes to the first
// queue that holds it, or fills a tile of the last. `inverse` maps a route to
// its sorted row. `tile_counts` holds the four queue lengths and the tiny
// queue's, then the threadgroup grid of each of the ten tile dispatches
// (the queues, then the tiny one, for each bank): tiles by `grid_rows`.
kernel void dinf_sorted_plan(
    device const uint *ids [[buffer(0)]],
    device atomic_uint *cursors [[buffer(1)]],
    device uint *sorted_routes [[buffer(2)]],
    device uint *inverse [[buffer(3)]],
    device uint *tile_counts [[buffer(4)]],
    device SortedTile *tiles [[buffer(5)]],
    device SortedTile *tiny_tiles [[buffer(6)]],
    constant SortedPlanParams &p [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  for (uint e = tid; e < p.experts; e += threads) {
    atomic_store_explicit(&cursors[e], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_device);
  for (uint r = tid; r < p.routes; r += threads) {
    const uint e = ids[r];
    if (e < p.experts) {
      atomic_fetch_add_explicit(&cursors[e], 1u, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (tid == 0) {
    uint base = 0;
    uint counts[4] = {0u, 0u, 0u, 0u};
    uint tiny = 0;
    for (uint e = 0; e < p.experts; ++e) {
      const uint n = atomic_load_explicit(&cursors[e], memory_order_relaxed);
      atomic_store_explicit(&cursors[e], base, memory_order_relaxed);
      uint done = 0;
      if (n > 0 && n <= p.tiny) {
        tiny_tiles[tiny++] = SortedTile{e, base, n, 0u};
        done = n;
      }
      while (done < n) {
        const uint left = n - done;
        uint q = p.queues - 1;
        for (uint c = 0; c + 1 < p.queues; ++c) {
          if (left <= p.caps[c]) {
            q = c;
            break;
          }
        }
        const uint valid = min(left, p.caps[q]);
        tiles[p.bases[q] + counts[q]++] = SortedTile{e, base + done, valid, 0u};
        done += valid;
      }
      base += n;
    }
    for (uint q = 0; q < 4; ++q) {
      tile_counts[q] = counts[q];
    }
    tile_counts[4] = tiny;
    for (uint d = 0; d < 10; ++d) {
      // An empty queue still launches one row of groups, which return.
      tile_counts[5 + d * 3] = max(tile_counts[d % 5], 1u);
      tile_counts[6 + d * 3] = p.grid_rows[d];
      tile_counts[7 + d * 3] = 1u;
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  for (uint r = tid; r < p.routes; r += threads) {
    const uint e = ids[r];
    if (e < p.experts) {
      const uint row = atomic_fetch_add_explicit(&cursors[e], 1u, memory_order_relaxed);
      sorted_routes[row] = r;
      inverse[r] = row;
    } else {
      inverse[r] = 0xffffffffu;
    }
  }
}

struct SortedGatherParams { uint routes; uint width; uint top_k; uint input_rows; };

// out[row] = half(input[token of sorted row]), eight values a thread.
kernel void dinf_sorted_gather_half(
    device const float *input [[buffer(0)]],
    device const uint *sorted_routes [[buffer(1)]],
    device half *out [[buffer(2)]],
    constant SortedGatherParams &p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint k = gid.x * 8;
  if (gid.y >= p.routes || k >= p.width) { return; }
  const uint flat = sorted_routes[gid.y];
  // Rows past the routed ones are unassigned when an ID is out of range.
  if (flat >= p.routes) { return; }
  const uint token = flat / p.top_k;
  const uint row = (flat % p.top_k) % p.input_rows;
  device const float *src = input + (ulong(token) * p.input_rows + row) * p.width + k;
  device half *dst = out + ulong(gid.y) * p.width + k;
  *(device packed_half4 *)dst = half4(*(device const packed_float4 *)src);
  *(device packed_half4 *)(dst + 4) = half4(*(device const packed_float4 *)(src + 4));
}

// Same as dinf_sorted_gather_half when input is already F16.
kernel void dinf_sorted_gather_half_f16(
    device const half *input [[buffer(0)]],
    device const uint *sorted_routes [[buffer(1)]],
    device half *out [[buffer(2)]],
    constant SortedGatherParams &p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint k = gid.x * 8;
  if (gid.y >= p.routes || k >= p.width) { return; }
  const uint flat = sorted_routes[gid.y];
  if (flat >= p.routes) { return; }
  const uint token = flat / p.top_k;
  const uint row = (flat % p.top_k) % p.input_rows;
  device const half *src = input + (ulong(token) * p.input_rows + row) * p.width + k;
  device half *dst = out + ulong(gid.y) * p.width + k;
  *(device packed_half4 *)dst = *(device const packed_half4 *)src;
  *(device packed_half4 *)(dst + 4) = *(device const packed_half4 *)(src + 4);
}

// out[token] = sum over its routes, in route order, of weight * sorted row.
kernel void dinf_sorted_route_sum(
    device const bfloat *routes [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const uint *inverse [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &top_k [[buffer(4)]],
    constant uint &hidden [[buffer(5)]],
    constant uint &tokens [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint column = gid.x * 8;
  if (gid.y >= tokens || column >= hidden) { return; }
  float4 sum0 = 0.0f;
  float4 sum1 = 0.0f;
  for (uint route = 0; route < top_k; ++route) {
    const uint flat = gid.y * top_k + route;
    const uint row = inverse[flat];
    if (row != 0xffffffffu) {
      const float w = weights[flat];
      device const bfloat *src = routes + ulong(row) * hidden + column;
      sum0 += float4(*(device const bfloat4 *)src) * w;
      sum1 += float4(*(device const bfloat4 *)(src + 4)) * w;
    }
  }
  device float *dst = out + ulong(gid.y) * hidden + column;
  *(device packed_float4 *)dst = sum0;
  *(device packed_float4 *)(dst + 4) = sum1;
}

struct SortedTileParams {
  uint in_dim; uint out_dim; uint row_bytes; uint queue;
  ulong expert_bytes; float beta; float linear_beta;
};

// Two adjacent 16-value groups of a row with one load of their scales and
// one or two of their weights, in place of a load per scale and per word:
// the tile threads spend their time on loads, not on the arithmetic. The
// values are the single-group dequantizers'.
#if defined(DINF_NVFP4)
#define DINF_SORTED_PAIR 1
inline half dinf_ue4m3_half(uchar x) {
    return (x == 0x7fu) ? 0.0h : as_type<half>(ushort((x & 0x7fu) << 7)) * half(128.0h);
}
// Groups `il` and `il + 1` of an NVFP4 block, `il` even.
inline void dinf_sorted_dequantize_pair(
    device const block_nvfp4_raw *xb, short il, thread half4x4 &a, thread half4x4 &b) {
    const ushort scales = *(device const ushort *)(xb->scales + il);
    const float s0 = float(dinf_ue4m3_half(uchar(scales & 0xff))) * 32768.0f;
    const float s1 = float(dinf_ue4m3_half(uchar(scales >> 8))) * 32768.0f;
    const uint4 q = *(device const packed_uint4 *)(xb->qs + 8 * il);
    a[0] = half4(raw_fp4_lo(q.x) * s0);
    a[1] = half4(raw_fp4_lo(q.y) * s0);
    a[2] = half4(raw_fp4_hi(q.x) * s0);
    a[3] = half4(raw_fp4_hi(q.y) * s0);
    b[0] = half4(raw_fp4_lo(q.z) * s1);
    b[1] = half4(raw_fp4_lo(q.w) * s1);
    b[2] = half4(raw_fp4_hi(q.z) * s1);
    b[3] = half4(raw_fp4_hi(q.w) * s1);
}
#elif defined(DINF_MXFP8)
#define DINF_SORTED_PAIR 1
// Both groups of an MXFP8 block (`il` is 0).
inline void dinf_sorted_dequantize_pair(
    device const block_mxfp8_raw *xb, short il, thread half4x4 &a, thread half4x4 &b) {
    const uchar sc = xb->scale;
    const uint4 q0 = *(device const packed_uint4 *)(xb->qs);
    const uint4 q1 = *(device const packed_uint4 *)(xb->qs + 16);
    if (sc >= 105 && sc <= 134) {
        const half s = as_type<half>(ushort((uint(sc) - 104u) << 10));
        FOR_UNROLL (short w = 0; w < 4; ++w) {
            const uint v0 = q0[w];
            const uint v1 = q1[w];
            a[w] = half4(raw_e4m3_half(v0), raw_e4m3_half(v0 >> 8),
                         raw_e4m3_half(v0 >> 16), raw_e4m3_half(v0 >> 24)) * s;
            b[w] = half4(raw_e4m3_half(v1), raw_e4m3_half(v1 >> 8),
                         raw_e4m3_half(v1 >> 16), raw_e4m3_half(v1 >> 24)) * s;
        }
    } else {
        const float s = as_type<float>(uint(sc) << 23) * 256.0f;
        FOR_UNROLL (short w = 0; w < 4; ++w) {
            a[w] = half4(raw_e4m3_4(q0[w]) * s);
            b[w] = half4(raw_e4m3_4(q1[w]) * s);
        }
    }
}
#endif

#ifdef DINF_EXPERT_SCALES
#define DINF_SORTED_SCALE_ARG , device const float *sorted_scales [[buffer(6)]]
#define DINF_SORTED_SCALE(e) sorted_scales[e]
#else
#define DINF_SORTED_SCALE_ARG
#define DINF_SORTED_SCALE(e) 1.0f
#endif
"#;

fn tile_name(pair: bool, geometry: Geometry) -> String {
    format!(
        "dinf_sorted_{}_m{}x{}_t{}x{}_n{}_k{}_s{}",
        if pair { "pair" } else { "down" },
        geometry.sub,
        geometry.subs,
        geometry.tail,
        geometry.tails,
        geometry.rows,
        geometry.k,
        geometry.simdgroups
    )
}

/// One tile kernel. The `down` kernel multiplies `rows` weight rows; the
/// pair kernel `rows` gate rows and their up rows of a `[gate; up]` bank
/// against the same activations, and its epilogue writes SiTU of the scaled
/// pair in half, `out_dim / 2` columns a route. Each run of `sub` routes has
/// its own accumulators.
fn tile_source(pair: bool, geometry: Geometry) -> String {
    use std::fmt::Write as _;
    let name = tile_name(pair, geometry);
    let Geometry {
        sub,
        tail,
        rows,
        k,
        simdgroups,
        ..
    } = geometry;
    let matrices = if pair { 2 } else { 1 };
    let mat_n = matrices * rows;
    let threads = simdgroups * 32;
    let (mut declare, mut run, mut finish) = (String::new(), String::new(), String::new());
    finish.push_str("    threadgroup float *s_out = (threadgroup float *)shared;\n");
    for (s, (size, first)) in geometry.runs().into_iter().enumerate() {
        let op = if size == sub { "matmul" } else { "matmul_tail" };
        write!(
            declare,
            "    auto acc{s} = {op}.get_destination_cooperative_tensor<\n        \
             decltype(input_tensor), decltype(tile0), float>();\n    \
             FOR_UNROLL (ushort i = 0; i < acc{s}.get_capacity(); ++i) {{\n        \
             acc{s}[i] = 0.0f;\n    }}\n"
        )
        .unwrap();
        write!(
            run,
            "        if (int(tile.valid) > {first}) {{\n            \
             auto slice{s} = input_tensor.slice(loop_k, {first});\n            \
             {op}.run(slice{s}, tile0, acc{s});\n        }}\n"
        )
        .unwrap();
        if pair {
            write!(
                finish,
                "    if (int(tile.valid) > {first}) {{\n        \
                 auto s_tensor = tensor(s_out, dextents<int32_t, 2>(mat_n, {size}));\n        \
                 acc{s}.store(s_tensor);\n        \
                 threadgroup_barrier(mem_flags::mem_threadgroup);\n        \
                 const int valid_run = min(int(tile.valid) - {first}, {size});\n        \
                 const int vecs = tile_n / 4;\n        \
                 const int total_vec = valid_run * vecs;\n        \
                 const float inv_beta = 1.0f / p.beta;\n        \
                 const float inv_lbeta = 1.0f / p.linear_beta;\n        \
                 const float beta_prod = p.beta * p.linear_beta;\n        \
                 for (int idx = int(tid); idx < total_vec; idx += {threads}) {{\n            \
                 const int r = (idx % vecs) * 4;\n            \
                 const int route_in_run = idx / vecs;\n            \
                 const int row = row_start + r;\n            \
                 if (row + 3 < matrix_rows) {{\n                \
                 const float4 g = *(threadgroup const float4 *)(s_out + route_in_run * mat_n + r) * scale;\n                \
                 const float4 u = *(threadgroup const float4 *)(s_out + route_in_run * mat_n + tile_n + r) * scale;\n                \
                 const float4 gate = tanh(clamp(g * inv_beta, -15.0f, 15.0f))\n                    \
                 / (1.0f + exp(clamp(-g, -80.0f, 80.0f)));\n                \
                 *(device half4 *)(output + (ulong(tile.base) + {first} + route_in_run) * matrix_rows + row) = half4(\n                    \
                 gate * beta_prod * tanh(clamp(u * inv_lbeta, -15.0f, 15.0f)));\n            \
                 }}\n        }}\n        \
                 threadgroup_barrier(mem_flags::mem_threadgroup);\n    }}\n"
            )
            .unwrap();
        } else {
            write!(
                finish,
                "    if (int(tile.valid) > {first}) {{\n        \
                 auto s_tensor = tensor(s_out, dextents<int32_t, 2>(mat_n, {size}));\n        \
                 acc{s}.store(s_tensor);\n        \
                 threadgroup_barrier(mem_flags::mem_threadgroup);\n        \
                 const int valid_run = min(int(tile.valid) - {first}, {size});\n        \
                 const int vecs = tile_n / 4;\n        \
                 const int total_vec = valid_run * vecs;\n        \
                 for (int idx = int(tid); idx < total_vec; idx += {threads}) {{\n            \
                 const int r = (idx % vecs) * 4;\n            \
                 const int route_in_run = idx / vecs;\n            \
                 const int row = row_start + r;\n            \
                 if (row + 3 < matrix_rows) {{\n                \
                 const float4 v = *(threadgroup const float4 *)(s_out + route_in_run * mat_n + r) * scale;\n                \
                 *(device bfloat4 *)(output + (ulong(tile.base) + {first} + route_in_run) * matrix_rows + row) = bfloat4(v);\n            \
                 }}\n        }}\n        \
                 threadgroup_barrier(mem_flags::mem_threadgroup);\n    }}\n"
            )
            .unwrap();
        }
    }
    // The tail runs' operation, when a tile has them.
    let tail_op = if geometry.tails > 0 && tail != sub {
        format!(
            "    mpp::tensor_ops::matmul2d<\n        \
             mpp::tensor_ops::matmul2d_descriptor(\n            \
             {tail}, mat_n, tile_k, false, true, false,\n            \
             mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),\n        \
             execution_simdgroups<{simdgroups}>> matmul_tail;\n"
        )
    } else {
        String::new()
    };
    // A thread's groups come in adjacent pairs of one row.
    let per_thread = matrices * rows * (k / 16) / (simdgroups * 32);
    let paired = u8::from(per_thread.is_multiple_of(2) && (k / 16).is_multiple_of(2));
    let output_type = if pair { "half" } else { "bfloat" };
    let output_tensor = "";
    format!(
        r#"
kernel void {name}(
    device const uchar *weights [[buffer(0)]],
    device half *input [[buffer(1)]],
    device {output_type} *output [[buffer(2)]],
    device const uint *tile_counts [[buffer(3)]],
    device const SortedTile *tiles [[buffer(4)]],
    constant SortedTileParams &p [[buffer(5)]]
    DINF_SORTED_SCALE_ARG,
    threadgroup half *shared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {{
    constexpr int sub_m = {sub};
    constexpr int tile_n = {rows};
    constexpr int mat_n = {mat_n};
    constexpr int tile_k = {k};
    constexpr int row_chunks = tile_k / 16;
    constexpr int matrix_chunks = tile_n * row_chunks;
    constexpr int per_thread = {matrices} * matrix_chunks / ({simdgroups} * 32);
    if (group.x >= tile_counts[p.queue]) {{ return; }}
    const SortedTile tile = tiles[group.x];
    // Weight rows of one matrix, and columns a route writes.
    const int matrix_rows = int(p.out_dim) / {matrices};
    const int row_start = int(group.y) * tile_n;
    auto tile0 = tensor(shared, dextents<int32_t, 2>(tile_k, mat_n));
    auto input_tensor = tensor(
        input + ulong(tile.base) * p.in_dim,
        dextents<int32_t, 2>(int(p.in_dim), int(tile.valid)),
        array<int, 2>({{1, int(p.in_dim)}}));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            sub_m, mat_n, tile_k, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<{simdgroups}>> matmul;
{tail_op}{declare}    device const uchar *expert_weights = weights + ulong(tile.expert) * p.expert_bytes;
    for (int loop_k = 0; loop_k < int(p.in_dim); loop_k += tile_k) {{
#if defined(DINF_SORTED_PAIR) && {paired}
        FOR_UNROLL (int c = 0; c < per_thread; c += 2) {{
            const int unit = int(tid) * per_thread + c;
            const int matrix = unit / matrix_chunks;
            const int local = unit % matrix_chunks;
            const int row = local / row_chunks;
            const int chunk = local % row_chunks;
            half4x4 low = half4x4(0.h);
            half4x4 high = half4x4(0.h);
            if (row_start + row < matrix_rows) {{
                device const DINF_BLOCK_TYPE *row_ptr = (device const DINF_BLOCK_TYPE *)(
                    expert_weights
                    + ulong(matrix * matrix_rows + row_start + row) * p.row_bytes);
                const int position = loop_k / 16 + chunk;
                dinf_sorted_dequantize_pair(
                    row_ptr + position / DINF_NL, short(position % DINF_NL), low, high);
            }}
            threadgroup half4 *dst = (threadgroup half4 *)(
                shared + (matrix * tile_n + row) * tile_k + chunk * 16);
            FOR_UNROLL (short i = 0; i < 4; ++i) {{
                dst[i] = low[i];
                dst[4 + i] = high[i];
            }}
        }}
#else
        FOR_UNROLL (int c = 0; c < per_thread; ++c) {{
            const int unit = int(tid) * per_thread + c;
            const int matrix = unit / matrix_chunks;
            const int local = unit % matrix_chunks;
            const int row = local / row_chunks;
            const int chunk = local % row_chunks;
            half4x4 values = half4x4(0.h);
            if (row_start + row < matrix_rows) {{
                device const DINF_BLOCK_TYPE *row_ptr = (device const DINF_BLOCK_TYPE *)(
                    expert_weights
                    + ulong(matrix * matrix_rows + row_start + row) * p.row_bytes);
                const int position = loop_k / 16 + chunk;
                DINF_DEQUANTIZE(row_ptr + position / DINF_NL, short(position % DINF_NL), values);
            }}
            threadgroup half4 *dst = (threadgroup half4 *)(
                shared + (matrix * tile_n + row) * tile_k + chunk * 16);
            FOR_UNROLL (short i = 0; i < 4; ++i) {{
                dst[i] = values[i];
            }}
        }}
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
{run}        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    const float scale = DINF_SORTED_SCALE(tile.expert);
{output_tensor}{finish}}}
"#
    )
}

type LibraryKey = (DeviceId, RawFormat, bool);
static LIBRARIES: OnceLock<Mutex<HashMap<LibraryKey, Library>>> = OnceLock::new();
static PIPELINES: OnceLock<Mutex<HashMap<(LibraryKey, String), ComputePipeline>>> = OnceLock::new();

fn library(device: &candle_core::MetalDevice, format: RawFormat, scales: bool) -> Result<Library> {
    use objc2_metal::{MTLCompileOptions, MTLLanguageVersion};
    let libraries = LIBRARIES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (device.id(), format, scales);
    if let Some(library) = libraries.lock().unwrap().get(&key) {
        return Ok(library.clone());
    }
    let tiles = tiles();
    let mut kernels = String::new();
    let mut seen = Vec::new();
    for (pair, geometry) in tiles
        .pair
        .iter()
        .map(|g| (true, *g))
        .chain(tiles.down.iter().map(|g| (false, *g)))
    {
        if !seen.contains(&(pair, geometry)) {
            seen.push((pair, geometry));
            kernels.push_str(&tile_source(pair, geometry));
        }
    }
    let source = format!(
        "{}{SORTED_MSL}{kernels}",
        super::raw_experts_metal::metal4_source(format, scales)
    );
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version4_0);
    let library = device
        .metal_device()
        .new_library_with_source(&source, Some(&options))
        .map_err(candle_core::Error::wrap)?;
    libraries.lock().unwrap().insert(key, library.clone());
    Ok(library)
}

fn pipeline(
    device: &candle_core::MetalDevice,
    format: RawFormat,
    scales: bool,
    name: &str,
) -> Result<ComputePipeline> {
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = ((device.id(), format, scales), name.to_string());
    if let Some(pipeline) = pipelines.lock().unwrap().get(&key) {
        return Ok(pipeline.clone());
    }
    let function = library(device, format, scales)?
        .get_function(name, None)
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    pipelines.lock().unwrap().insert(key, pipeline.clone());
    Ok(pipeline)
}

/// A bank as the kernels read it.
pub(crate) struct Bank<'a> {
    pub(crate) format: RawFormat,
    pub(crate) bytes: &'a Tensor,
    pub(crate) scales: Option<&'a Tensor>,
    pub(crate) experts: usize,
    pub(crate) out_dim: usize,
    pub(crate) in_dim: usize,
}

impl Bank<'_> {
    fn block(&self) -> (usize, usize) {
        match self.format {
            RawFormat::Nvfp4 => (64, 36),
            RawFormat::Mxfp8 => (32, 36),
            RawFormat::Iq4Nl => (32, 18),
        }
    }

    fn row_bytes(&self) -> usize {
        let (block, bytes) = self.block();
        self.in_dim / block * bytes
    }
}

/// Whether the sorted kernels take these banks: Metal 4, a `[gate; up]`
/// bank whose halves feed `down`, and widths the tiles and vector loads
/// divide.
fn serves(gate_up: &Bank<'_>, down: &Bank<'_>, device: &candle_core::MetalDevice) -> bool {
    let tiles = tiles();
    let steps = |bank: &Bank<'_>, kernels: &[Geometry]| {
        let (block, _) = bank.block();
        kernels
            .iter()
            .all(|g| bank.in_dim.is_multiple_of(block.max(g.k)))
    };
    std::env::var_os("DINF_RAW_MOE_UNSORTED").is_none()
        && super::raw_experts_metal::uses_metal4(device)
        && gate_up.experts == down.experts
        && gate_up.experts > 0
        && gate_up.out_dim == 2 * down.in_dim
        && steps(gate_up, tiles.pair)
        && steps(down, tiles.down)
        && gate_up.in_dim.is_multiple_of(8)
        && down.in_dim.is_multiple_of(4)
        && down.out_dim.is_multiple_of(8)
}

/// Compile the sorted kernels for `gate_up` and `down` banks of these
/// formats.
pub(crate) fn precompile(
    format: RawFormat,
    scales: bool,
    device: &candle_core::MetalDevice,
) -> Result<()> {
    if !super::raw_experts_metal::uses_metal4(device) {
        return Ok(());
    }
    let tiles = tiles();
    for name in [
        "dinf_sorted_plan",
        "dinf_sorted_gather_half",
        "dinf_sorted_gather_half_f16",
        "dinf_sorted_route_sum",
    ] {
        pipeline(device, format, scales, name)?;
    }
    for (pair, geometry) in tiles
        .pair
        .iter()
        .map(|g| (true, *g))
        .chain(tiles.down.iter().map(|g| (false, *g)))
    {
        pipeline(device, format, scales, &tile_name(pair, geometry))?;
    }
    Ok(())
}

/// The buffers of a pass, all but its result: the plan (cursors, sorted
/// routes, inverse map, queue lengths and tiles), the gathered inputs, the
/// SiTU outputs (F32 rows for the vector kernel's routes, half rows for the
/// tiles') and the `down` outputs. They stay out of Candle's pool, which
/// lends the smallest free buffer that fits a request, however large: there
/// a cache or state tensor that outlives the chunk pins tens of megabytes
/// when it takes one of the large ones, and a small plan buffer that takes
/// one for a moment makes the next large request allocate afresh (an 8K
/// prompt peaked 5.8 GB higher). One set serves every pass on a device, and
/// its lock is held while a pass is encoded, so the passes use it in order.
struct Scratch {
    buffers: [Option<(Arc<Buffer>, usize)>; SCRATCH_SLOTS],
}

const SCRATCH_SLOTS: usize = 10;

static SCRATCH: OnceLock<Mutex<HashMap<DeviceId, Arc<Mutex<Scratch>>>>> = OnceLock::new();

impl Scratch {
    fn of(device: &candle_core::MetalDevice) -> Arc<Mutex<Scratch>> {
        SCRATCH
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap()
            .entry(device.id())
            .or_insert_with(|| {
                Arc::new(Mutex::new(Scratch {
                    buffers: std::array::from_fn(|_| None),
                }))
            })
            .clone()
    }

    /// Buffer `slot` with room for `bytes`.
    fn buffer(
        &mut self,
        device: &candle_core::MetalDevice,
        slot: usize,
        bytes: usize,
    ) -> Result<Arc<Buffer>> {
        if let Some((buffer, capacity)) = &self.buffers[slot]
            && *capacity >= bytes
        {
            return Ok(buffer.clone());
        }
        let buffer = super::raw_experts_metal::exact_buffer(device, bytes)?;
        self.buffers[slot] = Some((buffer.clone(), bytes));
        Ok(buffer)
    }
}

#[repr(C)]
struct PlanParams {
    routes: u32,
    experts: u32,
    queues: u32,
    tiny: u32,
    caps: [u32; MAX_QUEUES],
    bases: [u32; MAX_QUEUES],
    grid_rows: [u32; 2 * (MAX_QUEUES + 1)],
}

#[repr(C)]
struct GatherParams {
    routes: u32,
    width: u32,
    top_k: u32,
    input_rows: u32,
}

#[repr(C)]
struct TinyParams {
    in_dim: u32,
    out_dim: u32,
    top_k: u32,
    has_scales: u32,
    row_bytes: u64,
    expert_bytes: u64,
    beta: f32,
    linear_beta: f32,
}

#[repr(C)]
struct TileParams {
    in_dim: u32,
    out_dim: u32,
    row_bytes: u32,
    queue: u32,
    expert_bytes: u64,
    beta: f32,
    linear_beta: f32,
}

fn grid(width: usize, height: usize) -> MTLSize {
    MTLSize {
        width,
        height,
        depth: 1,
    }
}

/// `sum_r weights[t, r] * down(situ(gate_up(x[t])))` over the experts
/// `ids` `[tokens, top_k]` U32 for `x` `[tokens, in_dim]` F32, giving
/// `[tokens, out_dim]` F32, or `None` when these kernels do not take the
/// banks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward(
    gate_up: &Bank<'_>,
    down: &Bank<'_>,
    x: &Tensor,
    x_f16: Option<&Tensor>,
    ids: &Tensor,
    weights: &Tensor,
    beta: f32,
    linear_beta: f32,
) -> Result<Option<Tensor>> {
    let candle_core::Device::Metal(device) = gate_up.bytes.device() else {
        return Ok(None);
    };
    if !serves(gate_up, down, device) {
        return Ok(None);
    }
    let (tokens, in_dim) = x.dims2()?;
    let (id_tokens, top_k) = ids.dims2()?;
    if in_dim != gate_up.in_dim
        || id_tokens != tokens
        || weights.dims2()? != (tokens, top_k)
        || tokens == 0
        || top_k == 0
    {
        candle_core::bail!(
            "raw prompt MoE input {:?} does not fit ids {:?} and weights {:?}",
            x.dims(),
            ids.dims(),
            weights.dims()
        )
    }
    for bank in [gate_up, down] {
        if bank.bytes.elem_count() != bank.experts * bank.out_dim * bank.row_bytes() {
            candle_core::bail!("raw expert bank holds {} bytes", bank.bytes.elem_count())
        }
    }
    let tiles = tiles();
    let (experts, ffn, hidden) = (gate_up.experts, down.in_dim, down.out_dim);
    let routes = tokens * top_k;
    let x = x.contiguous()?;
    let ids = ids.contiguous()?;
    let weights = weights.contiguous()?;
    let (x_buffer, x_offset) = metal_view(&x, DType::F32, "input")?;
    let (ids_buffer, ids_offset) = metal_view(&ids, DType::U32, "ids")?;
    let (weights_buffer, weights_offset) = metal_view(&weights, DType::F32, "route weights")?;
    let (gate_up_bytes, gate_up_offset) = metal_view(gate_up.bytes, DType::U8, "weights")?;
    let (down_bytes, down_offset) = metal_view(down.bytes, DType::U8, "weights")?;
    let scales = |bank: &Bank<'_>| {
        bank.scales
            .map(|scales| metal_view(scales, DType::F32, "scales"))
            .transpose()
    };
    let (gate_up_scales, down_scales) = (scales(gate_up)?, scales(down)?);

    let buffer = |count: usize, dtype: DType, label: &'static str| {
        device
            .new_buffer_builder()
            .with_size_for(count.max(1), dtype)
            .with_label(label)
            .build()
    };
    let queues = tiles.pair.len();
    let caps: Vec<usize> = tiles.pair.iter().map(|g| g.capacity()).collect();
    if queues == 0
        || queues > MAX_QUEUES
        || tiles.down.len() != queues
        || tiles
            .down
            .iter()
            .zip(&caps)
            .any(|(g, cap)| g.capacity() != *cap)
        || caps[0] == 0
        || caps.windows(2).any(|pair| pair[0] >= pair[1])
    {
        candle_core::bail!("raw prompt MoE tile queues differ between the banks")
    }
    // An expert's last tile goes to one of the queues, its full ones to the
    // last queue.
    let mut bases = [0u32; MAX_QUEUES];
    for (q, base) in bases.iter_mut().enumerate().take(queues) {
        *base = to_u32(q * experts, "tile queue")?;
    }
    let tile_capacity = queues * experts + routes / caps[queues - 1];
    let mut cap_params = [0u32; MAX_QUEUES];
    for (param, cap) in cap_params.iter_mut().zip(&caps) {
        *param = to_u32(*cap, "tile routes")?;
    }
    // The tiny kernels' threadgroups make two gate/up pairs or four rows.
    let mut grid_rows = [1u32; 2 * (MAX_QUEUES + 1)];
    for (bank, (kernels, rows, tiny_rows)) in [
        (&tiles.pair, ffn, ffn.div_ceil(2)),
        (&tiles.down, hidden, hidden.div_ceil(4)),
    ]
    .into_iter()
    .enumerate()
    {
        for (q, geometry) in kernels.iter().enumerate() {
            grid_rows[bank * (MAX_QUEUES + 1) + q] = to_u32(rows.div_ceil(geometry.rows), "rows")?;
        }
        grid_rows[bank * (MAX_QUEUES + 1) + MAX_QUEUES] = to_u32(tiny_rows, "rows")?;
    }
    let tiny = TINY_ROUTES;
    let scratch = Scratch::of(device);
    let mut scratch = scratch.lock().unwrap();
    let cursors = scratch.buffer(device, 4, experts * 4)?;
    let sorted_routes = scratch.buffer(device, 5, routes * 4)?;
    let inverse = scratch.buffer(device, 6, routes * 4)?;
    let tile_counts = scratch.buffer(device, 7, (MAX_QUEUES + 1 + 2 * (MAX_QUEUES + 1) * 3) * 4)?;
    let queued_tiles = scratch.buffer(device, 8, tile_capacity * 16)?;
    let tiny_tiles = scratch.buffer(device, 9, experts * 16)?;
    let gathered = scratch.buffer(device, 0, routes * in_dim * 2)?;
    let activated = scratch.buffer(device, 1, routes * ffn * 4)?;
    let activated_half = scratch.buffer(device, 2, routes * ffn * 2)?;
    let routed = scratch.buffer(device, 3, routes * hidden * 2)?;
    let output = buffer(tokens * hidden, DType::F32, "dinf-sorted-moe")?;

    let shared = |name: &str| pipeline(device, gate_up.format, gate_up.scales.is_some(), name);

    let guard = device.command_encoder()?;
    let encoder: &ComputeCommandEncoder = guard.as_ref();
    encoder.set_compute_pipeline_state(&shared("dinf_sorted_plan")?);
    encoder.set_input_buffer(0, Some(&ids_buffer), ids_offset);
    encoder.set_output_buffer(1, Some(&cursors), 0);
    encoder.set_output_buffer(2, Some(&sorted_routes), 0);
    encoder.set_output_buffer(3, Some(&inverse), 0);
    encoder.set_output_buffer(4, Some(&tile_counts), 0);
    encoder.set_output_buffer(5, Some(&queued_tiles), 0);
    encoder.set_output_buffer(6, Some(&tiny_tiles), 0);
    encoder.set_bytes(
        7,
        &PlanParams {
            routes: to_u32(routes, "routes")?,
            experts: to_u32(experts, "experts")?,
            queues: to_u32(queues, "tile queues")?,
            tiny: to_u32(tiny, "tile routes")?,
            caps: cap_params,
            bases,
            grid_rows,
        },
    );
    encoder.dispatch_thread_groups(
        grid(1, 1),
        grid(routes.max(experts).clamp(32, 1024).next_multiple_of(32), 1),
    );

    drop(guard);

    let x_f16_view = match x_f16 {
        Some(t) if t.dtype() == DType::F16 && t.is_contiguous() && t.dims() == x.dims() => {
            Some(metal_view(t, DType::F16, "half input")?)
        }
        _ => None,
    };
    let guard = device.command_encoder()?;
    let encoder: &ComputeCommandEncoder = guard.as_ref();
    if let Some((xh_buf, xh_off)) = &x_f16_view {
        encoder.set_compute_pipeline_state(&shared("dinf_sorted_gather_half_f16")?);
        encoder.set_input_buffer(0, Some(xh_buf), *xh_off);
    } else {
        encoder.set_compute_pipeline_state(&shared("dinf_sorted_gather_half")?);
        encoder.set_input_buffer(0, Some(&x_buffer), x_offset);
    }
    encoder.set_input_buffer(1, Some(&sorted_routes), 0);
    encoder.set_output_buffer(2, Some(&gathered), 0);
    encoder.set_bytes(
        3,
        &GatherParams {
            routes: to_u32(routes, "routes")?,
            width: to_u32(in_dim, "input width")?,
            top_k: to_u32(top_k, "routes")?,
            input_rows: 1,
        },
    );
    encoder.dispatch_threads(grid(in_dim / 8, routes), grid(32, 8));
    drop(guard);

    let run_tiles = |bank: &Bank<'_>,
                     bytes: &(Buffer, usize),
                     scales: &Option<(Buffer, usize)>,
                     pair: bool,
                     input: &Arc<Buffer>,
                     tiny_input: (&Buffer, usize),
                     out: &Arc<Buffer>,
                     tiny_out: &Arc<Buffer>|
     -> Result<()> {
        let kernels = if pair { &tiles.pair } else { &tiles.down };
        // The fullest tiles first.
        for (queue, geometry) in kernels.iter().copied().enumerate().rev() {
            let chunks = (if pair { 2 } else { 1 }) * geometry.rows * geometry.k / 16;
            if !chunks.is_multiple_of(geometry.threads()) {
                candle_core::bail!("raw prompt MoE tile {geometry:?} does not divide its threads")
            }
            let guard = device.command_encoder()?;
            let encoder: &ComputeCommandEncoder = guard.as_ref();
            encoder.set_compute_pipeline_state(&pipeline(
                device,
                bank.format,
                bank.scales.is_some(),
                &tile_name(pair, geometry),
            )?);
            encoder.set_input_buffer(0, Some(&bytes.0), bytes.1);
            encoder.set_input_buffer(1, Some(input), 0);
            // The queues write disjoint sorted rows, so their dispatches
            // may overlap.
            if queue + 1 == queues {
                encoder.set_output_buffer(2, Some(out), 0);
            } else {
                encoder.set_output_buffer_unordered(2, out, 0);
            }
            encoder.set_input_buffer(3, Some(&tile_counts), 0);
            encoder.set_input_buffer(4, Some(&queued_tiles), bases[queue] as usize * 16);
            encoder.set_bytes(
                5,
                &TileParams {
                    in_dim: to_u32(bank.in_dim, "input width")?,
                    out_dim: to_u32(bank.out_dim, "output width")?,
                    row_bytes: to_u32(bank.row_bytes(), "row bytes")?,
                    queue: queue as u32,
                    expert_bytes: (bank.out_dim * bank.row_bytes()) as u64,
                    beta,
                    linear_beta,
                },
            );
            if let Some((scales, offset)) = scales {
                encoder.set_input_buffer(6, Some(scales), *offset);
            }
            let epilogue_bytes = geometry.sub * (if pair { 2 } else { 1 } * geometry.rows) * 4;
            encoder.set_threadgroup_memory_length(
                0,
                (chunks * 32).max(epilogue_bytes).next_multiple_of(16),
            );
            let dispatch = usize::from(!pair) * (MAX_QUEUES + 1) + queue;
            encoder.dispatch_thread_groups_indirect(
                &tile_counts,
                (MAX_QUEUES + 1 + dispatch * 3) * 4,
                grid(geometry.threads(), 1),
            );
        }
        // The tiny queue: `x` rows by token for the pair, `source`'s sorted
        // rows otherwise.
        let guard = device.command_encoder()?;
        let encoder: &ComputeCommandEncoder = guard.as_ref();
        encoder.set_compute_pipeline_state(&super::raw_experts_metal::pipeline(
            device,
            super::raw_experts_metal::sorted_tiny_kernel(bank.format, pair),
        )?);
        encoder.set_input_buffer(0, Some(&bytes.0), bytes.1);
        encoder.set_input_buffer(1, Some(tiny_input.0), tiny_input.1);
        encoder.set_input_buffer(2, Some(&sorted_routes), 0);
        encoder.set_input_buffer(3, Some(&tile_counts), 0);
        encoder.set_input_buffer(4, Some(&tiny_tiles), 0);
        match scales {
            Some((scales, offset)) => encoder.set_input_buffer(5, Some(scales), *offset),
            // Unread without scales; any bound buffer will do.
            None => encoder.set_input_buffer(5, Some(&tile_counts), 0),
        }
        if Arc::ptr_eq(out, tiny_out) {
            encoder.set_output_buffer_unordered(6, tiny_out, 0);
        } else {
            encoder.set_output_buffer(6, Some(tiny_out), 0);
        }
        encoder.set_bytes(
            7,
            &TinyParams {
                in_dim: to_u32(bank.in_dim, "input width")?,
                out_dim: to_u32(bank.out_dim, "output width")?,
                top_k: to_u32(top_k, "routes")?,
                has_scales: u32::from(scales.is_some()),
                row_bytes: bank.row_bytes() as u64,
                expert_bytes: (bank.out_dim * bank.row_bytes()) as u64,
                beta,
                linear_beta,
            },
        );
        let dispatch = usize::from(!pair) * (MAX_QUEUES + 1) + MAX_QUEUES;
        encoder.dispatch_thread_groups_indirect(
            &tile_counts,
            (MAX_QUEUES + 1 + dispatch * 3) * 4,
            grid(64, 1),
        );
        Ok(())
    };

    run_tiles(
        gate_up,
        &(gate_up_bytes, gate_up_offset),
        &gate_up_scales,
        true,
        &gathered,
        (&x_buffer, x_offset),
        &activated_half,
        &activated,
    )?;

    run_tiles(
        down,
        &(down_bytes, down_offset),
        &down_scales,
        false,
        &activated_half,
        (&activated, 0),
        &routed,
        &routed,
    )?;

    let guard = device.command_encoder()?;
    let encoder: &ComputeCommandEncoder = guard.as_ref();
    encoder.set_compute_pipeline_state(&shared("dinf_sorted_route_sum")?);
    encoder.set_input_buffer(0, Some(&routed), 0);
    encoder.set_input_buffer(1, Some(&weights_buffer), weights_offset);
    encoder.set_input_buffer(2, Some(&inverse), 0);
    encoder.set_output_buffer(3, Some(&output), 0);
    encoder.set_bytes(4, &to_u32(top_k, "routes")?);
    encoder.set_bytes(5, &to_u32(hidden, "output width")?);
    encoder.set_bytes(6, &to_u32(tokens, "tokens")?);
    encoder.dispatch_threads(grid(hidden / 8, tokens), grid(32, 8));
    drop(guard);

    drop(scratch);
    Ok(Some(Tensor::from((
        Storage::Metal(MetalStorage::new(
            output,
            device.clone(),
            tokens * hidden,
            DType::F32,
        )),
        Shape::from((tokens, hidden)),
    ))))
}
