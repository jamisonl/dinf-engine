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
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float s_data[DINF_BQ * DINF_BK];
    threadgroup half p_data[DINF_BQ * DINF_BK];

    const int q_start = int(group.x) * DINF_BQ;
    const uint head = group.y;
    const uint batch = group.z;
    const uint kv_head = head / p.gqa;
    device half *qh = q + ulong(batch) * p.q_batch + ulong(head) * p.q_head;
    device half *kh = k + ulong(batch) * p.k_batch + ulong(kv_head) * p.k_head;
    device half *vh = v + ulong(batch) * p.v_batch + ulong(kv_head) * p.v_head;
    device float *oh = o + (ulong(batch) * p.heads + head) * ulong(p.q_seq) * D;

    auto q_t = tensor(qh, dextents<int32_t, 2>(D, int(p.q_seq)), array<int, 2>({1, int(p.q_row)}));
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
    const bool row_valid = q_start + row < int(p.q_seq);
    const int position = int(p.q_offset) + q_start + row;

    // Keys this tile can see, in whole blocks.
    const int first_position = int(p.q_offset) + q_start;
    const int last_position = min(first_position + DINF_BQ - 1, int(p.q_offset + p.q_seq) - 1);
    int kb_begin = 0;
    if (p.window != 0) {
        kb_begin = max(0, first_position - int(p.window) + 1) / DINF_BK * DINF_BK;
    }
    const int kb_end = min(int(p.kv_seq), last_position + 1);

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

    auto o_t = tensor(oh, dextents<int32_t, 2>(D, int(p.q_seq)), array<int, 2>({1, D}));
    auto o_slice = o_t.slice(0, q_start);
    output.store(o_slice);
    threadgroup_barrier(mem_flags::mem_device);
    if (row_valid) {
        device float *o_row = oh + ulong(q_start + row) * D;
        for (int d = part; d < D; d += DINF_PARTS) {
            o_row[d] *= inv_sum;
        }
    }
}

template [[host_name("dinf_flash_attention_256")]] kernel decltype(dinf_flash_attention<256>) dinf_flash_attention<256>;
template [[host_name("dinf_flash_attention_512")]] kernel decltype(dinf_flash_attention<512>) dinf_flash_attention<512>;
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
}

type PipelineKey = (DeviceId, usize);
static PIPELINES: OnceLock<Mutex<HashMap<PipelineKey, ComputePipeline>>> = OnceLock::new();

fn pipeline(device: &candle_core::MetalDevice, dim: usize) -> Result<ComputePipeline> {
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (device.id(), dim);
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
        .get_function(&format!("dinf_flash_attention_{dim}"), None)
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
        let params = FlashParams {
            q_seq: q_seq as u32,
            kv_seq: kv_seq as u32,
            q_offset: (kv_seq - q_seq) as u32,
            window: self.window,
            gqa: (heads / kv_heads) as u32,
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
        };
        let device = q.device();
        let elements = batch * heads * q_seq * dim;
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-flash-attention")
            .build()?;
        let pipeline = pipeline(device, dim)?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-flash-attention");
        encoder.set_compute_pipeline_state(&pipeline);
        let e: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        let half = DType::F16.size_in_bytes();
        e.set_input_buffer(0, Some(q.buffer()), q_layout.start_offset() * half);
        e.set_input_buffer(1, Some(k.buffer()), k_layout.start_offset() * half);
        e.set_input_buffer(2, Some(v.buffer()), v_layout.start_offset() * half);
        e.set_output_buffer(3, Some(&output), 0);
        e.set_bytes(4, &params);
        e.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: q_seq.div_ceil(QUERY_TILE),
                height: heads,
                depth: batch,
            },
            objc2_metal::MTLSize {
                width: THREADS,
                height: 1,
                depth: 1,
            },
        );
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
    if std::env::var_os("DINF_DISABLE_METAL_FLASH_ATTENTION").is_some()
        || !matches!(dim, 256 | 512)
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
