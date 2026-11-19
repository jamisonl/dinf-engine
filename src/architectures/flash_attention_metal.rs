//! Causal and left-windowed prompt attention on Metal 4 tensor operations.
//!
//! Candle's Steel attention runs head dimensions 256 and 512 on small
//! SIMD-matrix tiles, and its cost dominates long prompts. These kernels use
//! the same tensor operations as the quantized matmuls. Each threadgroup owns
//! 32 queries and makes two passes over their keys: the first finds each row's
//! softmax maximum and sum, the second recomputes the scores and accumulates
//! `exp(score - max)` (at most 1, kept in half as Steel does) times values;
//! the rows are divided by their sums in F32 after the store. Recomputing QK
//! costs a third more arithmetic but needs no running rescale of the
//! accumulator. Causal and window masks are computed in the kernel, so no mask
//! buffer exists.
//!
//! A few queries over a long cache (speculative verification) would leave most
//! of each 32-row tile empty and give the GPU one threadgroup per head. There,
//! the query heads that share a key/value head fill one tile together, and each
//! tile's keys are split across threadgroups. Each split stores its rows'
//! unnormalized output with their maximum and sum, and a second kernel merges
//! the splits.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{CpuStorage, CustomOp3, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::{MTLCompileOptions, MTLLanguageVersion};

const FLASH_ATTENTION_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

struct DinfFlashParams {
    uint q_seq;
    uint kv_seq;
    uint q_offset;
    uint window;
    uint gqa;
    float scale;
    uint q_batch;
    uint q_head;
    uint q_row;
    uint k_batch;
    uint k_head;
    uint k_row;
    uint v_batch;
    uint v_head;
    uint v_row;
    uint heads;
    // Query heads that share one tile's rows: 1, or `gqa` when packed.
    uint pack;
    // Threadgroups that divide each tile's keys.
    uint splits;
};

constant constexpr int DINF_BQ = 32;
constant constexpr int DINF_BK = 32;
constant constexpr int DINF_KC = 64;
constant constexpr int DINF_PARTS = 8;
constant constexpr int DINF_COLS = DINF_BK / DINF_PARTS;
constant constexpr float DINF_MASKED = -1.0e30f;

#define DINF_FLASH_SCORES(kb)                                                   \
    {                                                                           \
        for (ushort i = 0; i < scores.get_capacity(); ++i) {                    \
            scores[i] = 0.0f;                                                   \
        }                                                                       \
        for (int kc = 0; kc < D; kc += DINF_KC) {                               \
            auto q_slice = q_t.slice(kc, q_start);                              \
            auto k_slice = k_t.slice(kc, (kb));                                 \
            qk.run(q_slice, k_slice, scores);                                   \
        }                                                                       \
        scores.store(s_t);                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                        \
    }

template <int D>
kernel void dinf_flash_attention(
    device half *q [[buffer(0)]],
    device half *k [[buffer(1)]],
    device half *v [[buffer(2)]],
    device float *o [[buffer(3)]],
    constant DinfFlashParams &p [[buffer(4)]],
    device float2 *row_stats [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float s_data[DINF_BQ * DINF_BK];
    threadgroup half p_data[DINF_BQ * DINF_BK];

    // Rows are the `pack` query heads' queries in order: row r is query
    // r % q_seq of head first_head + r / q_seq.
    const int rows = int(p.pack * p.q_seq);
    const int q_start = int(group.x / p.splits) * DINF_BQ;
    const uint split = group.x % p.splits;
    const uint first_head = group.y * p.pack;
    const uint batch = group.z;
    const uint kv_head = first_head / p.gqa;
    const ulong first_row = (ulong(batch) * p.heads + first_head) * p.q_seq;
    device half *qh = q + ulong(batch) * p.q_batch + ulong(first_head) * p.q_head;
    device half *kh = k + ulong(batch) * p.k_batch + ulong(kv_head) * p.k_head;
    device half *vh = v + ulong(batch) * p.v_batch + ulong(kv_head) * p.v_head;
    // With splits, each row keeps one D-wide partial per split.
    device float *oh = o + (first_row * p.splits + split) * D;

    auto q_t = tensor(qh, dextents<int32_t, 2>(D, rows), array<int, 2>({1, int(p.q_row)}));
    auto k_t = tensor(kh, dextents<int32_t, 2>(D, int(p.kv_seq)), array<int, 2>({1, int(p.k_row)}));
    auto v_t = tensor(vh, dextents<int32_t, 2>(D, int(p.kv_seq)), array<int, 2>({1, int(p.v_row)}));
    auto s_t = tensor(s_data, dextents<int32_t, 2>(DINF_BK, DINF_BQ));
    auto p_t = tensor(p_data, dextents<int32_t, 2>(DINF_BK, DINF_BQ));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            DINF_BQ, DINF_BK, DINF_KC, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<8>> qk;
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            DINF_BQ, D, DINF_BK, false, false, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<8>> pv;
    auto scores = qk.template get_destination_cooperative_tensor<decltype(q_t), decltype(k_t), float>();
    auto output = pv.template get_destination_cooperative_tensor<decltype(p_t), decltype(v_t), float>();
    for (ushort i = 0; i < output.get_capacity(); ++i) {
        output[i] = 0.0f;
    }

    const int row = tid / DINF_PARTS;
    const int part = tid % DINF_PARTS;
    const bool row_valid = q_start + row < rows;
    const int position = int(p.q_offset) + (q_start + row) % int(p.q_seq);

    // Keys this tile can see, in whole blocks. A tile spanning two heads
    // holds every query position.
    const int last_row = min(q_start + DINF_BQ, rows) - 1;
    const bool one_head = q_start / int(p.q_seq) == last_row / int(p.q_seq);
    const int first_position = int(p.q_offset) + (one_head ? q_start % int(p.q_seq) : 0);
    const int last_position =
        int(p.q_offset) + (one_head ? last_row % int(p.q_seq) : int(p.q_seq) - 1);
    int tile_begin = 0;
    if (p.window != 0) {
        tile_begin = max(0, first_position - int(p.window) + 1) / DINF_BK * DINF_BK;
    }
    const int tile_end = min(int(p.kv_seq), last_position + 1);
    // This split's share of the tile's blocks.
    const int blocks = max(0, tile_end - tile_begin + DINF_BK - 1) / DINF_BK;
    const int per_split = (blocks + int(p.splits) - 1) / int(p.splits);
    const int kb_begin = tile_begin + int(split) * per_split * DINF_BK;
    const int kb_end = min(tile_end, kb_begin + per_split * DINF_BK);

    float row_max = DINF_MASKED;
    float row_sum = 0.0f;
    for (int kb = kb_begin; kb < kb_end; kb += DINF_BK) {
        DINF_FLASH_SCORES(kb);
        float values[DINF_COLS];
        bool valid[DINF_COLS];
        float block_max = DINF_MASKED;
        for (int c = 0; c < DINF_COLS; ++c) {
            const int col = part * DINF_COLS + c;
            const int key = kb + col;
            valid[c] = row_valid && key < int(p.kv_seq) && key <= position
                && (p.window == 0 || key + int(p.window) > position);
            values[c] = valid[c] ? s_data[row * DINF_BK + col] * p.scale : DINF_MASKED;
            block_max = max(block_max, values[c]);
        }
        for (ushort offset = 1; offset < DINF_PARTS; offset <<= 1) {
            block_max = max(block_max, simd_shuffle_xor(block_max, offset));
        }
        const float new_max = max(row_max, block_max);
        float block_sum = 0.0f;
        for (int c = 0; c < DINF_COLS; ++c) {
            block_sum += valid[c] ? exp(values[c] - new_max) : 0.0f;
        }
        for (ushort offset = 1; offset < DINF_PARTS; offset <<= 1) {
            block_sum += simd_shuffle_xor(block_sum, offset);
        }
        row_sum = row_sum * exp(row_max - new_max) + block_sum;
        row_max = new_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
    for (int kb = kb_begin; kb < kb_end; kb += DINF_BK) {
        DINF_FLASH_SCORES(kb);
        for (int c = 0; c < DINF_COLS; ++c) {
            const int col = part * DINF_COLS + c;
            const int key = kb + col;
            const bool valid = row_valid && key < int(p.kv_seq) && key <= position
                && (p.window == 0 || key + int(p.window) > position);
            // Unnormalized like Steel: normalized long-row probabilities
            // fall toward half's subnormal range and lose precision.
            const float probability = valid
                ? exp(s_data[row * DINF_BK + col] * p.scale - row_max)
                : 0.0f;
            p_data[row * DINF_BK + col] = half(probability);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto v_slice = v_t.slice(0, kb);
        pv.run(p_t, v_slice, output);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int o_row_stride = int(p.splits) * D;
    auto o_t = tensor(oh, dextents<int32_t, 2>(D, rows), array<int, 2>({1, o_row_stride}));
    auto o_slice = o_t.slice(0, q_start);
    output.store(o_slice);
    if (p.splits > 1) {
        // The merge kernel scales the partials.
        if (row_valid && part == 0) {
            row_stats[(first_row + ulong(q_start + row)) * p.splits + split] =
                float2(row_max, row_sum);
        }
        return;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (row_valid) {
        device float *o_row = oh + ulong(q_start + row) * o_row_stride;
        for (int d = part; d < D; d += DINF_PARTS) {
            o_row[d] *= inv_sum;
        }
    }
}

template [[host_name("dinf_flash_attention_256")]] kernel decltype(dinf_flash_attention<256>) dinf_flash_attention<256>;
template [[host_name("dinf_flash_attention_512")]] kernel decltype(dinf_flash_attention<512>) dinf_flash_attention<512>;

// One thread per output element: weight each split's partial by
// exp(split max - row max) and divide by the weighted sums.
kernel void dinf_flash_attention_merge(
    device const float *partials [[buffer(0)]],
    device const float2 *row_stats [[buffer(1)]],
    device float *o [[buffer(2)]],
    constant uint &elements [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    constant uint &splits [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= elements) {
        return;
    }
    const uint row = index / dim;
    const uint d = index % dim;
    device const float2 *stats = row_stats + ulong(row) * splits;
    float row_max = DINF_MASKED;
    for (uint s = 0; s < splits; ++s) {
        row_max = max(row_max, stats[s].x);
    }
    float sum = 0.0f;
    float value = 0.0f;
    device const float *partial = partials + ulong(row) * splits * dim + d;
    for (uint s = 0; s < splits; ++s) {
        const float weight = stats[s].y > 0.0f ? exp(stats[s].x - row_max) : 0.0f;
        sum += weight * stats[s].y;
        value += weight * partial[ulong(s) * dim];
    }
    o[index] = sum > 0.0f ? value / sum : 0.0f;
}
"#;

const QUERY_TILE: usize = 32;
const THREADS: usize = 256;

#[repr(C)]
#[derive(Clone, Copy)]
struct FlashParams {
    q_seq: u32,
    kv_seq: u32,
    q_offset: u32,
    window: u32,
    gqa: u32,
    scale: f32,
    q_batch: u32,
    q_head: u32,
    q_row: u32,
    k_batch: u32,
    k_head: u32,
    k_row: u32,
    v_batch: u32,
    v_head: u32,
    v_row: u32,
    heads: u32,
    pack: u32,
    splits: u32,
}

/// Threadgroups worth filling the GPU with before keys are split.
const TARGET_THREADGROUPS: usize = 256;
/// Fewest key blocks one split covers.
const MIN_SPLIT_BLOCKS: usize = 4;

/// How many threadgroups divide each tile's keys: enough to give the GPU
/// `TARGET_THREADGROUPS`, but at least `MIN_SPLIT_BLOCKS` blocks each.
fn key_splits(tile_groups: usize, visible_keys: usize) -> usize {
    if tile_groups >= TARGET_THREADGROUPS {
        return 1;
    }
    let blocks = visible_keys.div_ceil(QUERY_TILE);
    TARGET_THREADGROUPS
        .div_ceil(tile_groups)
        .min(blocks / MIN_SPLIT_BLOCKS)
        .max(1)
}

type PipelineKey = (DeviceId, &'static str);
static PIPELINES: OnceLock<Mutex<HashMap<PipelineKey, ComputePipeline>>> = OnceLock::new();

fn pipeline(device: &candle_core::MetalDevice, name: &'static str) -> Result<ComputePipeline> {
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (device.id(), name);
    if let Some(pipeline) = pipelines.lock().unwrap().get(&key) {
        return Ok(pipeline.clone());
    }
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version4_0);
    let library = device
        .metal_device()
        .new_library_with_source(FLASH_ATTENTION_MSL, Some(&options))
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

struct FlashAttention {
    scale: f32,
    window: u32,
}

/// Element strides `(batch, head, row)` of a `[batch, heads, seq, dim]` view
/// whose rows are contiguous.
fn view_strides(layout: &Layout) -> Result<(u32, u32, u32)> {
    let stride = layout.stride();
    if stride.len() != 4 || stride[3] != 1 {
        candle_core::bail!("flash attention needs contiguous head rows")
    }
    let narrow = |value: usize| {
        u32::try_from(value)
            .map_err(|_| candle_core::Error::Msg("flash attention stride exceeds u32".into()))
    };
    Ok((narrow(stride[0])?, narrow(stride[1])?, narrow(stride[2])?))
}

impl CustomOp3 for FlashAttention {
    fn name(&self) -> &'static str {
        "dinf-metal-flash-attention"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("flash attention is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        q: &MetalStorage,
        q_layout: &Layout,
        k: &MetalStorage,
        k_layout: &Layout,
        v: &MetalStorage,
        v_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (batch, heads, q_seq, dim) = q_layout.shape().dims4()?;
        let (k_batch, kv_heads, kv_seq, k_dim) = k_layout.shape().dims4()?;
        if [q.dtype(), k.dtype(), v.dtype()] != [DType::F16; 3]
            || k_layout.shape() != v_layout.shape()
            || batch != k_batch
            || dim != k_dim
            || kv_heads == 0
            || !heads.is_multiple_of(kv_heads)
            || q_seq > kv_seq
        {
            candle_core::bail!("flash attention tensor metadata changed")
        }
        let (q_batch, q_head, q_row) = view_strides(q_layout)?;
        let (k_batch_stride, k_head, k_row) = view_strides(k_layout)?;
        let (v_batch_stride, v_head, v_row) = view_strides(v_layout)?;
        let gqa = heads / kv_heads;
        // Too few queries to fill a tile: the heads sharing a key/value head
        // fill it together, which needs their rows evenly spaced.
        let pack = if q_seq < QUERY_TILE && q_head as usize == q_seq * q_row as usize {
            gqa
        } else {
            1
        };
        let tiles = (pack * q_seq).div_ceil(QUERY_TILE);
        let tile_groups = tiles * (heads / pack) * batch;
        let visible_keys = match self.window {
            0 => kv_seq,
            window => kv_seq.min(window as usize + q_seq),
        };
        let splits = key_splits(tile_groups, visible_keys);
        let params = FlashParams {
            q_seq: q_seq as u32,
            kv_seq: kv_seq as u32,
            q_offset: (kv_seq - q_seq) as u32,
            window: self.window,
            gqa: gqa as u32,
            scale: self.scale,
            q_batch,
            q_head,
            q_row,
            k_batch: k_batch_stride,
            k_head,
            k_row,
            v_batch: v_batch_stride,
            v_head,
            v_row,
            heads: heads as u32,
            pack: pack as u32,
            splits: splits as u32,
        };
        let device = q.device();
        let elements = batch * heads * q_seq * dim;
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-flash-attention")
            .build()?;
        let (partials, row_stats) = if splits > 1 {
            let partials = device
                .new_buffer_builder()
                .with_size_for(elements * splits, DType::F32)
                .with_label("dinf-flash-attention-partials")
                .build()?;
            let row_stats = device
                .new_buffer_builder()
                .with_size_for(elements / dim * splits * 2, DType::F32)
                .with_label("dinf-flash-attention-row-stats")
                .build()?;
            (Some(partials), Some(row_stats))
        } else {
            (None, None)
        };
        let kernel = match dim {
            256 => "dinf_flash_attention_256",
            512 => "dinf_flash_attention_512",
            _ => candle_core::bail!("flash attention head width changed"),
        };
        let attention = pipeline(device, kernel)?;
        {
            let encoder = device.command_encoder()?;
            encoder.set_label("dinf-flash-attention");
            encoder.set_compute_pipeline_state(&attention);
            let e: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
            let half = DType::F16.size_in_bytes();
            e.set_input_buffer(0, Some(q.buffer()), q_layout.start_offset() * half);
            e.set_input_buffer(1, Some(k.buffer()), k_layout.start_offset() * half);
            e.set_input_buffer(2, Some(v.buffer()), v_layout.start_offset() * half);
            e.set_output_buffer(3, Some(partials.as_ref().unwrap_or(&output)), 0);
            e.set_bytes(4, &params);
            // Unused without splits.
            e.set_output_buffer(5, Some(row_stats.as_ref().unwrap_or(&output)), 0);
            e.dispatch_thread_groups(
                objc2_metal::MTLSize {
                    width: tiles * splits,
                    height: heads / pack,
                    depth: batch,
                },
                objc2_metal::MTLSize {
                    width: THREADS,
                    height: 1,
                    depth: 1,
                },
            );
        }
        if let (Some(partials), Some(row_stats)) = (&partials, &row_stats) {
            let merge = pipeline(device, "dinf_flash_attention_merge")?;
            let encoder = device.command_encoder()?;
            encoder.set_label("dinf-flash-attention-merge");
            encoder.set_compute_pipeline_state(&merge);
            let e: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
            e.set_input_buffer(0, Some(partials), 0);
            e.set_input_buffer(1, Some(row_stats), 0);
            e.set_output_buffer(2, Some(&output), 0);
            e.set_bytes(3, &(elements as u32));
            e.set_bytes(4, &(dim as u32));
            e.set_bytes(5, &(splits as u32));
            e.dispatch_thread_groups(
                objc2_metal::MTLSize {
                    width: elements.div_ceil(THREADS),
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: THREADS,
                    height: 1,
                    depth: 1,
                },
            );
        }
        Ok((
            MetalStorage::new(output, device.clone(), elements, DType::F32),
            q_layout.shape().clone(),
        ))
    }
}

/// Causal attention over F16 `[batch, heads, seq, dim]` views, limited to a
/// left `window` when non-zero. Queries are the last `q_seq` positions of the
/// keys. Returns F16, or `None` when these kernels do not apply.
pub(super) fn try_flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    window: usize,
    scale: f32,
) -> Result<Option<Tensor>> {
    let (_, _, q_seq, dim) = q.dims4()?;
    if !matches!(dim, 256 | 512)
        || q_seq < 2
        || window > u32::MAX as usize
        || !super::tensor_qmatmul_metal::supports_tensor_batch_attention(q.device())?
    {
        return Ok(None);
    }
    let output = q.contiguous()?.apply_op3_no_bwd(
        k,
        v,
        &FlashAttention {
            scale,
            window: window as u32,
        },
    )?;
    Ok(Some(output.to_dtype(DType::F16)?))
}

#[cfg(test)]
#[path = "flash_attention_metal_tests.rs"]
mod tests;
