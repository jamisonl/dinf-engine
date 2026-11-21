//! Inkling 2 prompt attention on Metal 4 tensor operations.
//!
//! This is `flash_attention_metal.rs`'s two-pass tile kernel (32 queries by
//! 32 keys, scores recomputed in the second pass) with Inkling's additions:
//! every score gains the per-query, per-head bias for its key's distance,
//! read from a table whose last bias column covers every distance past the
//! relative extent, and a per-query sink logit with a zero value seeds the
//! first split's softmax maximum and sum. Positions are relative to the
//! first cached key, so a trimmed sliding-window cache works unchanged.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{
    CpuStorage, CustomOp3, DType, Layout, MetalStorage, Result, Shape, Storage, Tensor,
};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::{MTLCompileOptions, MTLLanguageVersion};

const INKLING_FLASH_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

struct InklingFlashParams {
    uint q_seq;
    uint kv_seq;
    uint q_offset;
    uint window;
    uint gqa;
    uint q_head;
    uint q_row;
    uint k_head;
    uint k_row;
    uint v_head;
    uint v_row;
    uint heads;
    uint splits;
    uint extent;
    uint bias_stride;
};

constant constexpr int IBQ = 32;
constant constexpr int IBK = 32;
constant constexpr int IKC = 64;
constant constexpr int IPARTS = 8;
constant constexpr int ICOLS = IBK / IPARTS;
constant constexpr float IMASKED = -1.0e30f;

#define INKLING_SCORES(kb)                                                      \
    {                                                                           \
        for (ushort i = 0; i < scores.get_capacity(); ++i) {                    \
            scores[i] = 0.0f;                                                   \
        }                                                                       \
        for (int kc = 0; kc < D; kc += IKC) {                                   \
            auto q_slice = q_t.slice(kc, q_start);                              \
            auto k_slice = k_t.slice(kc, (kb));                                 \
            qk.run(q_slice, k_slice, scores);                                   \
        }                                                                       \
        scores.store(s_t);                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                        \
    }

template <int D>
kernel void inkling_flash_attention(
    device half *q [[buffer(0)]],
    device half *k [[buffer(1)]],
    device half *v [[buffer(2)]],
    device float *o [[buffer(3)]],
    constant InklingFlashParams &p [[buffer(4)]],
    device float2 *row_stats [[buffer(5)]],
    device const float *bias [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float s_data[IBQ * IBK];
    threadgroup half p_data[IBQ * IBK];

    const int rows = int(p.q_seq);
    const int q_start = int(group.x / p.splits) * IBQ;
    const uint split = group.x % p.splits;
    const uint head = group.y;
    const uint kv_head = head / p.gqa;
    const ulong first_row = ulong(head) * p.q_seq;
    device half *qh = q + ulong(head) * p.q_head;
    device half *kh = k + ulong(kv_head) * p.k_head;
    device half *vh = v + ulong(kv_head) * p.v_head;
    device float *oh = o + (first_row * p.splits + split) * D;

    auto q_t = tensor(qh, dextents<int32_t, 2>(D, rows), array<int, 2>({1, int(p.q_row)}));
    auto k_t = tensor(kh, dextents<int32_t, 2>(D, int(p.kv_seq)), array<int, 2>({1, int(p.k_row)}));
    auto v_t = tensor(vh, dextents<int32_t, 2>(D, int(p.kv_seq)), array<int, 2>({1, int(p.v_row)}));
    auto s_t = tensor(s_data, dextents<int32_t, 2>(IBK, IBQ));
    auto p_t = tensor(p_data, dextents<int32_t, 2>(IBK, IBQ));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            IBQ, IBK, IKC, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<8>> qk;
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            IBQ, D, IBK, false, false, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<8>> pv;
    auto scores = qk.template get_destination_cooperative_tensor<decltype(q_t), decltype(k_t), float>();
    auto output = pv.template get_destination_cooperative_tensor<decltype(p_t), decltype(v_t), float>();
    for (ushort i = 0; i < output.get_capacity(); ++i) {
        output[i] = 0.0f;
    }

    const int row = tid / IPARTS;
    const int part = tid % IPARTS;
    const int query = q_start + row;
    const bool row_valid = query < rows;
    const int position = int(p.q_offset) + query;
    device const float *row_bias =
        bias + (ulong(min(query, rows - 1)) * p.heads + head) * p.bias_stride;
    const int extent = int(p.extent);

    const int last_row = min(q_start + IBQ, rows) - 1;
    const int first_position = int(p.q_offset) + q_start;
    const int last_position = int(p.q_offset) + last_row;
    int tile_begin = 0;
    if (p.window != 0) {
        tile_begin = max(0, first_position - int(p.window) + 1) / IBK * IBK;
    }
    const int tile_end = min(int(p.kv_seq), last_position + 1);
    const int blocks = max(0, tile_end - tile_begin + IBK - 1) / IBK;
    const int per_split = (blocks + int(p.splits) - 1) / int(p.splits);
    const int kb_begin = tile_begin + int(split) * per_split * IBK;
    const int kb_end = min(tile_end, kb_begin + per_split * IBK);

    // The sink joins the first split as a logit whose value is zero.
    float row_max = IMASKED;
    float row_sum = 0.0f;
    if (split == 0 && row_valid) {
        row_max = row_bias[extent + 1];
        row_sum = 1.0f;
    }
    for (int kb = kb_begin; kb < kb_end; kb += IBK) {
        INKLING_SCORES(kb);
        float values[ICOLS];
        bool valid[ICOLS];
        float block_max = IMASKED;
        for (int c = 0; c < ICOLS; ++c) {
            const int col = part * ICOLS + c;
            const int key = kb + col;
            valid[c] = row_valid && key < int(p.kv_seq) && key <= position
                && (p.window == 0 || key + int(p.window) > position);
            values[c] = valid[c]
                ? s_data[row * IBK + col] + row_bias[min(position - key, extent)]
                : IMASKED;
            block_max = max(block_max, values[c]);
        }
        for (ushort offset = 1; offset < IPARTS; offset <<= 1) {
            block_max = max(block_max, simd_shuffle_xor(block_max, offset));
        }
        const float new_max = max(row_max, block_max);
        float block_sum = 0.0f;
        for (int c = 0; c < ICOLS; ++c) {
            block_sum += valid[c] ? exp(values[c] - new_max) : 0.0f;
        }
        for (ushort offset = 1; offset < IPARTS; offset <<= 1) {
            block_sum += simd_shuffle_xor(block_sum, offset);
        }
        row_sum = row_sum * exp(row_max - new_max) + block_sum;
        row_max = new_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
    for (int kb = kb_begin; kb < kb_end; kb += IBK) {
        INKLING_SCORES(kb);
        for (int c = 0; c < ICOLS; ++c) {
            const int col = part * ICOLS + c;
            const int key = kb + col;
            const bool valid = row_valid && key < int(p.kv_seq) && key <= position
                && (p.window == 0 || key + int(p.window) > position);
            const float probability = valid
                ? exp(s_data[row * IBK + col] + row_bias[min(position - key, extent)] - row_max)
                : 0.0f;
            p_data[row * IBK + col] = half(probability);
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
        if (row_valid && part == 0) {
            row_stats[(first_row + ulong(query)) * p.splits + split] = float2(row_max, row_sum);
        }
        return;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (row_valid) {
        device float *o_row = oh + ulong(query) * o_row_stride;
        for (int d = part; d < D; d += IPARTS) {
            o_row[d] *= inv_sum;
        }
    }
}

template [[host_name("inkling_flash_attention_128")]] kernel decltype(inkling_flash_attention<128>) inkling_flash_attention<128>;

kernel void inkling_flash_attention_merge(
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
    float row_max = IMASKED;
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
/// Threadgroups worth filling the GPU with before keys are split.
const TARGET_THREADGROUPS: usize = 256;
/// Fewest key blocks one split covers.
const MIN_SPLIT_BLOCKS: usize = 4;

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    q_seq: u32,
    kv_seq: u32,
    q_offset: u32,
    window: u32,
    gqa: u32,
    q_head: u32,
    q_row: u32,
    k_head: u32,
    k_row: u32,
    v_head: u32,
    v_row: u32,
    heads: u32,
    splits: u32,
    extent: u32,
    bias_stride: u32,
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
        .new_library_with_source(INKLING_FLASH_MSL, Some(&options))
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

fn narrow(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        candle_core::Error::Msg(format!("Inkling 2 flash attention {what} exceeds u32"))
    })
}

struct InklingFlash {
    /// `[q_seq, heads, extent + 2]` F32: the bias by distance, then the sink.
    bias: Tensor,
    extent: usize,
    window: usize,
    q_offset: usize,
}

impl CustomOp3 for InklingFlash {
    fn name(&self) -> &'static str {
        "inkling2-flash-attention"
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
        candle_core::bail!("Inkling 2 flash attention is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        q: &MetalStorage,
        ql: &Layout,
        k: &MetalStorage,
        kl: &Layout,
        v: &MetalStorage,
        vl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (q_seq, heads, dim) = ql.shape().dims3()?;
        let (batch, kv_heads, kv_seq, k_dim) = kl.shape().dims4()?;
        let q_stride = ql.stride();
        let k_stride = kl.stride();
        let v_stride = vl.stride();
        if [q.dtype(), k.dtype(), v.dtype()] != [DType::F16; 3]
            || batch != 1
            || kl.shape() != vl.shape()
            || dim != 128
            || k_dim != dim
            || kv_heads == 0
            || heads % kv_heads != 0
            || q_seq > kv_seq
            || q_stride[2] != 1
            || k_stride[3] != 1
            || v_stride[3] != 1
            || self.bias.dims() != [q_seq, heads, self.extent + 2]
            || self.bias.dtype() != DType::F32
            || !self.bias.is_contiguous()
        {
            candle_core::bail!("Inkling 2 flash attention geometry differs")
        }
        let tiles = q_seq.div_ceil(QUERY_TILE);
        let tile_groups = tiles * heads;
        let visible = match self.window {
            0 => kv_seq,
            window => kv_seq.min(window + q_seq),
        };
        let splits = if tile_groups >= TARGET_THREADGROUPS {
            1
        } else {
            TARGET_THREADGROUPS
                .div_ceil(tile_groups)
                .min(visible.div_ceil(QUERY_TILE) / MIN_SPLIT_BLOCKS)
                .max(1)
        };
        let params = Params {
            q_seq: narrow(q_seq, "queries")?,
            kv_seq: narrow(kv_seq, "keys")?,
            q_offset: narrow(self.q_offset, "query offset")?,
            window: narrow(self.window, "window")?,
            gqa: narrow(heads / kv_heads, "group")?,
            q_head: narrow(q_stride[1], "query head stride")?,
            q_row: narrow(q_stride[0], "query row stride")?,
            k_head: narrow(k_stride[1], "key head stride")?,
            k_row: narrow(k_stride[2], "key row stride")?,
            v_head: narrow(v_stride[1], "value head stride")?,
            v_row: narrow(v_stride[2], "value row stride")?,
            heads: narrow(heads, "heads")?,
            splits: narrow(splits, "splits")?,
            extent: narrow(self.extent, "extent")?,
            bias_stride: narrow(self.extent + 2, "bias row")?,
        };
        let (bias_storage, bias_layout) = self.bias.storage_and_layout();
        let Storage::Metal(bias_storage) = &*bias_storage else {
            candle_core::bail!("Inkling 2 flash attention bias must be on Metal")
        };
        let device = q.device();
        if bias_storage.device().id() != device.id() {
            candle_core::bail!("Inkling 2 flash attention tensors are on different devices")
        }
        let elements = heads * q_seq * dim;
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("inkling2-flash-attention")
            .build()?;
        let (partials, row_stats) = if splits > 1 {
            (
                Some(
                    device
                        .new_buffer_builder()
                        .with_size_for(elements * splits, DType::F32)
                        .with_label("inkling2-flash-partials")
                        .build()?,
                ),
                Some(
                    device
                        .new_buffer_builder()
                        .with_size_for(elements / dim * splits * 2, DType::F32)
                        .with_label("inkling2-flash-row-stats")
                        .build()?,
                ),
            )
        } else {
            (None, None)
        };
        let attention = pipeline(device, "inkling_flash_attention_128")?;
        {
            let encoder = device.command_encoder()?;
            encoder.set_label("inkling2-flash-attention");
            encoder.set_compute_pipeline_state(&attention);
            let e: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
            let half = DType::F16.size_in_bytes();
            e.set_input_buffer(0, Some(q.buffer()), ql.start_offset() * half);
            e.set_input_buffer(1, Some(k.buffer()), kl.start_offset() * half);
            e.set_input_buffer(2, Some(v.buffer()), vl.start_offset() * half);
            e.set_output_buffer(3, Some(partials.as_ref().unwrap_or(&output)), 0);
            e.set_bytes(4, &params);
            e.set_output_buffer(5, Some(row_stats.as_ref().unwrap_or(&output)), 0);
            e.set_input_buffer(
                6,
                Some(bias_storage.buffer()),
                bias_layout.start_offset() * DType::F32.size_in_bytes(),
            );
            e.dispatch_thread_groups(
                objc2_metal::MTLSize {
                    width: tiles * splits,
                    height: heads,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: THREADS,
                    height: 1,
                    depth: 1,
                },
            );
        }
        if let (Some(partials), Some(row_stats)) = (&partials, &row_stats) {
            let merge = pipeline(device, "inkling_flash_attention_merge")?;
            let encoder = device.command_encoder()?;
            encoder.set_label("inkling2-flash-attention-merge");
            encoder.set_compute_pipeline_state(&merge);
            let e: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
            e.set_input_buffer(0, Some(partials), 0);
            e.set_input_buffer(1, Some(row_stats), 0);
            e.set_output_buffer(2, Some(&output), 0);
            e.set_bytes(3, &narrow(elements, "elements")?);
            e.set_bytes(4, &narrow(dim, "head width")?);
            e.set_bytes(5, &narrow(splits, "splits")?);
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
            Shape::from((heads, q_seq, dim)),
        ))
    }
}

/// Prompt attention for `q` `[q_seq, heads, 128]` (F16, already scaled)
/// over the `[1, kv_heads, keys, 128]` F16 cache views `k` and `v`, whose
/// first key is `q_offset` positions before the first query. `bias` is
/// `[q_seq, heads, extent + 2]` F32: the bias by distance (column `extent`
/// for every longer distance) and the sink. Returns `[heads, q_seq, 128]`
/// F32, or `None` where these kernels do not run.
pub(super) fn try_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    bias: &Tensor,
    extent: usize,
    window: Option<usize>,
    q_offset: usize,
) -> anyhow::Result<Option<Tensor>> {
    let (q_seq, _, dim) = q.dims3()?;
    if dim != 128
        || q_seq < 2
        || !super::tensor_qmatmul_metal::supports_tensor_batch_attention(q.device())?
    {
        return Ok(None);
    }
    Ok(Some(q.apply_op3_no_bwd(
        k,
        v,
        &InklingFlash {
            bias: bias.contiguous()?,
            extent,
            window: window.unwrap_or(0),
            q_offset,
        },
    )?))
}
