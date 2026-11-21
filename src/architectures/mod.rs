//! Model-family implementations and shared pipeline/cache primitives.
//!
//! The architecture modules expose stage constructors and model-specific
//! helpers, while this module defines the common stage input/output and cache
//! contracts used by the engine.

mod activation;
mod cpu_math;
mod text_context;
pub use text_context::{TextArithmeticPolicy, TextRequestContext};

#[cfg(feature = "cuda")]
mod decode_sdpa_cuda;
/// DeepSeek V3/R1 multi-head latent-attention stage.
pub mod deepseek2;
/// DeepSeek V4 Flash stage and packed-weight support.
pub mod deepseek4;
pub mod deepseek41;
/// DeepSeek V4 packed matrix-multiplication helper.
pub mod deepseek4_qmatmul;
#[cfg(feature = "cuda")]
mod deepseek_mla_cuda;
#[cfg(target_os = "macos")]
mod deepseek_mla_metal;
#[cfg(feature = "cuda")]
mod dense_qmatmul_cuda;
pub(crate) mod dflash2;
#[cfg(target_os = "macos")]
mod dflash2_conv_metal;
#[cfg(target_os = "macos")]
mod dflash2_selector_metal;
#[cfg(target_os = "macos")]
mod dflash2_topk_metal;
#[cfg(target_os = "macos")]
mod flash_attention_metal;
#[cfg(target_os = "macos")]
mod fused_norm_metal;
#[cfg(target_os = "macos")]
mod fused_route_metal;
/// Gemma 4 text decoder stage.
pub mod gemma4;
#[cfg(feature = "cuda")]
mod gemma4_moe_decode_cuda;
#[cfg(feature = "cuda")]
mod gemma4_norm_fusion_cuda;
pub mod glm5_next;
#[cfg(target_os = "macos")]
mod glm5_next_context_metal;
#[cfg(target_os = "macos")]
mod glm5_next_kda_metal;
#[cfg(target_os = "macos")]
mod glm5_next_sort_metal;
#[cfg(all(test, target_os = "macos"))]
mod hc_bench;
mod hc_post;
#[cfg(target_os = "macos")]
mod hc_post_metal;
mod hc_pre;
#[cfg(target_os = "macos")]
mod hc_pre_metal;
mod hc_sinkhorn;
#[cfg(target_os = "macos")]
mod hc_sinkhorn_metal;
mod indexed_moe;
#[cfg(any(feature = "cuda", test))]
mod indexed_moe_cuda;
#[cfg(any(feature = "cuda", test))]
mod indexed_moe_epilogue_cuda;
#[cfg(target_os = "macos")]
mod indexed_moe_metal;
#[cfg(feature = "cuda")]
mod inkling2_cuda;
#[cfg(target_os = "macos")]
mod inkling2_flash_metal;
pub mod inkling2_loop;
#[cfg(target_os = "macos")]
mod inkling2_metal;
#[cfg(feature = "cuda")]
mod kv_append_cuda;
#[cfg(target_os = "macos")]
mod kv_append_metal;
/// Llama-family decoder stage, including Llama, Mistral, and Qwen2 layouts.
pub mod llama;
pub mod llama4;
pub mod minimax_m2;
/// Muse Spark Open pooled-expert decoder stage.
pub mod muse_spark;
/// Nemotron-H decoder stage.
pub mod nemotron_h;
#[cfg(target_os = "macos")]
mod nemotron_mamba;
#[cfg(any(feature = "cuda", test))]
mod nemotron_mamba_cuda;
/// Phi-3 decoder stage.
pub mod phi3;
/// Process-wide layer profiling switches.
pub mod profiling;
#[cfg(feature = "cuda")]
mod q8_1_epilogue_cuda;
#[cfg(feature = "cuda")]
mod q8_1_memo_cuda;
#[cfg(feature = "cuda")]
mod qkv_decode_cuda;
#[cfg(target_os = "macos")]
mod qkv_layout_metal;
/// Qwen 3.5 decoder stage.
pub mod qwen35;
#[cfg(feature = "cuda")]
mod qwen35_decode_layout_cuda;
#[cfg(feature = "cuda")]
mod qwen35_deltanet_cuda;
#[cfg(any(feature = "cuda", test))]
mod qwen35_deltanet_layout;
#[cfg(target_os = "macos")]
mod qwen35_deltanet_metal;
pub mod qwen35_vision;
/// Qwen 3 Mixture-of-Experts decoder stage.
pub mod qwen3moe;
#[cfg(target_os = "macos")]
mod ragged_metal;
/// Expert banks in GGUF formats Candle lacks (NVFP4, IQ4_NL).
mod raw_experts;
#[cfg(feature = "cuda")]
mod residual_norm_cuda;
#[cfg(target_os = "macos")]
pub(crate) mod sdpa_f16kv_metal;
#[cfg(target_os = "macos")]
mod tensor_qmatmul_metal;
#[cfg(target_os = "macos")]
pub(crate) use tensor_qmatmul_metal::with_speculative_verify;

use std::sync::Arc;

use crate::qmatmul::QMatMul;
use anyhow::{Context, Result};
use candle_core::quantized::k_quants::{BlockQ4K, BlockQ6K};
use candle_core::quantized::{GgmlDType, GgmlType, QTensor};
use candle_core::{D, Device, Module, Tensor};
use half::f16;

use super::gguf_loader::StageTensors;
use super::turboquant::{QuantizedKvCache, TurboQuantConfig};

pub(crate) fn preferred_speculative_block_size(device: &Device, maximum: usize) -> Result<usize> {
    #[cfg(target_os = "macos")]
    {
        tensor_qmatmul_metal::preferred_speculative_block_size(device, maximum).map_err(Into::into)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = device;
        Ok(maximum)
    }
}

pub(crate) fn uses_pre_m5_speculative_policy(device: &Device) -> Result<bool> {
    #[cfg(target_os = "macos")]
    {
        tensor_qmatmul_metal::uses_pre_m5_speculative_policy(device).map_err(Into::into)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = device;
        Ok(false)
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn precompile_metal_kv_append(device: &Device) -> Result<()> {
    kv_append_metal::precompile(device)
}

/// Whether the Metal 4 quantized-matmul path may materialize a second
/// accelerator-side tensor while constructing a GLM stage.  The GGUF memory
/// estimator uses this only to account for construction overlap; the actual
/// architecture path still performs its per-dtype kernel validation.
pub(crate) fn uses_metal_tensor_qmatmul(device: &Device) -> Result<bool> {
    #[cfg(target_os = "macos")]
    {
        tensor_qmatmul_metal::supports_metal_tensor_qmatmul(device).map_err(Into::into)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = device;
        Ok(false)
    }
}

/// Quantized matmul with a narrow Metal fallback for affected single-token K-quants.
pub(super) struct DecodeSafeQMatMul {
    quantized: QMatMul,
    single_token: Option<QMatMul>,
    single_token_matrix: bool,
}

pub(super) fn needs_metal_single_token_fallback(dtype: GgmlDType) -> bool {
    matches!(dtype, GgmlDType::Q4K | GgmlDType::Q6K)
}

impl DecodeSafeQMatMul {
    pub(super) fn from_qtensor(qt: Arc<QTensor>, device: &Device) -> Result<Self> {
        // Candle's Metal GEMV path for these K-quants can return non-finite or
        // materially incorrect values at m=1. Keep GEMM quantized for prefill.
        let single_token = if device.is_metal() && needs_metal_single_token_fallback(qt.dtype()) {
            Some(QMatMul::TensorF16(dequantize_f16_compact(&qt, device)?))
        } else {
            None
        };
        Ok(Self {
            quantized: QMatMul::from_arc(qt)?,
            single_token,
            single_token_matrix: false,
        })
    }

    /// Avoid the dense single-row Metal path on models where it fails too.
    pub(super) fn with_matrix_decode(mut self) -> Self {
        self.single_token_matrix = true;
        self
    }

    pub(super) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.dim(D::Minus2)? == 1 {
            return self.forward_decode(x);
        }
        self.quantized.forward(x).map_err(Into::into)
    }

    /// Apply decode precision to every row, including flattened batch heads.
    /// The row count of `[lanes, hidden]` is not a prefill sequence length.
    pub(super) fn forward_decode(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(single_token) = self.single_token.as_ref() {
            if !self.single_token_matrix || x.dim(D::Minus2)? > 1 {
                return single_token.forward(x).map_err(Into::into);
            }
            // Candle's dense f16 single-row Metal path also returns non-finite
            // values for MiniMax's V projection. Use the matrix path with two
            // identical rows, then keep one. Weights and input precision stay
            // unchanged; this adds no persistent allocation or CPU readback.
            let padded = Tensor::cat(&[x, x], D::Minus2)?;
            let output = single_token
                .forward(&padded)?
                .narrow(D::Minus2, 0, 1)?
                .contiguous()?;
            return Ok(output);
        }
        self.quantized.forward(x).map_err(Into::into)
    }

    #[cfg(feature = "cuda")]
    pub(super) fn quantized(&self) -> &QMatMul {
        &self.quantized
    }
}

fn dequantize_f16_compact(qt: &QTensor, device: &Device) -> Result<Tensor> {
    match qt.dtype() {
        GgmlDType::Q4K => dequantize_blocks_f16::<BlockQ4K>(qt, device),
        GgmlDType::Q6K => dequantize_blocks_f16::<BlockQ6K>(qt, device),
        dtype => anyhow::bail!("compact f16 dequantization is not implemented for {dtype:?}"),
    }
}

fn dequantize_blocks_f16<T: GgmlType>(qt: &QTensor, device: &Device) -> Result<Tensor> {
    const BLOCKS_PER_CHUNK: usize = 4096;

    let data = qt.data()?;
    let bytes_per_block = std::mem::size_of::<T>();
    anyhow::ensure!(
        data.len() % bytes_per_block == 0,
        "quantized tensor byte length is not block aligned"
    );
    let block_count = data.len() / bytes_per_block;
    let elem_count = qt.shape().elem_count();
    anyhow::ensure!(
        block_count.saturating_mul(T::BLCK_SIZE) == elem_count,
        "quantized tensor block count does not match its shape"
    );

    let mut output = Vec::with_capacity(elem_count);
    for bytes in data.chunks(bytes_per_block * BLOCKS_PER_CHUNK) {
        let chunk_blocks = bytes.len() / bytes_per_block;
        let mut blocks = vec![T::zeros(); chunk_blocks];
        // T is Candle's repr(C) GGUF block type. The destination is aligned and
        // sized for exactly this byte chunk before the typed dequantizer reads it.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                blocks.as_mut_ptr().cast::<u8>(),
                bytes.len(),
            );
        }
        let mut values = vec![0f32; chunk_blocks * T::BLCK_SIZE];
        T::to_float(&blocks, &mut values);
        output.extend(values.into_iter().map(f16::from_f32));
    }

    Tensor::from_vec(output, qt.shape().clone(), device).map_err(Into::into)
}

/// Input to a pipeline stage's forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionSpan {
    /// First token position covered by the span.
    pub start: usize,
    /// Number of consecutive token positions in the span.
    pub len: usize,
}

/// Three-axis rotary positions used by Qwen multimodal prompts.
///
/// Text tokens use the same value on every axis. Image tokens use independent
/// temporal, height, and width values. The CPU-side representation is cheap
/// to retain with a request and pass between pipeline stages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MropePositionIds {
    temporal: Vec<u32>,
    height: Vec<u32>,
    width: Vec<u32>,
}

impl MropePositionIds {
    /// Construct an M-RoPE schedule from its temporal, height, and width axes.
    ///
    /// All axes must have the same non-zero length; otherwise an error is
    /// returned.
    pub fn new(temporal: Vec<u32>, height: Vec<u32>, width: Vec<u32>) -> Result<Self> {
        anyhow::ensure!(
            temporal.len() == height.len() && temporal.len() == width.len(),
            "M-RoPE axes must have equal lengths (temporal={}, height={}, width={})",
            temporal.len(),
            height.len(),
            width.len()
        );
        anyhow::ensure!(!temporal.is_empty(), "M-RoPE schedule cannot be empty");
        Ok(Self {
            temporal,
            height,
            width,
        })
    }

    /// Build ordinary one-dimensional text positions on all three axes.
    pub fn text(start_pos: usize, seq_len: usize) -> Result<Self> {
        Self::text_with_delta(start_pos, 0, seq_len)
    }

    /// Build cached decode positions after a multimodal prefill.
    ///
    /// Qwen defines the signed delta as `max(prefill_position) + 1 -
    /// prefill_token_count`; it is commonly negative for image prompts.
    pub fn text_with_delta(start_pos: usize, delta: i64, seq_len: usize) -> Result<Self> {
        anyhow::ensure!(seq_len > 0, "M-RoPE schedule cannot be empty");
        let mut positions = Vec::with_capacity(seq_len);
        for offset in 0..seq_len {
            let logical = start_pos
                .checked_add(offset)
                .context("M-RoPE logical position overflow")?;
            let logical = i64::try_from(logical)
                .map_err(|_| anyhow::anyhow!("M-RoPE logical position exceeds i64"))?;
            let position = logical
                .checked_add(delta)
                .context("M-RoPE decode position overflow")?;
            anyhow::ensure!(position >= 0, "M-RoPE decode position became negative");
            positions.push(
                u32::try_from(position)
                    .map_err(|_| anyhow::anyhow!("M-RoPE position exceeds u32"))?,
            );
        }
        Self::new(positions.clone(), positions.clone(), positions)
    }

    /// Return the number of positions in each axis.
    pub fn len(&self) -> usize {
        self.temporal.len()
    }

    /// Return whether the schedule contains no positions.
    pub fn is_empty(&self) -> bool {
        self.temporal.is_empty()
    }

    /// Return temporal-axis positions in token order.
    pub fn temporal(&self) -> &[u32] {
        &self.temporal
    }

    /// Return height-axis positions in token order.
    pub fn height(&self) -> &[u32] {
        &self.height
    }

    /// Return width-axis positions in token order.
    pub fn width(&self) -> &[u32] {
        &self.width
    }

    /// Return all three axes as borrowed slices in temporal/height/width order.
    pub fn axes(&self) -> [&[u32]; 3] {
        [&self.temporal, &self.height, &self.width]
    }

    /// Signed offset needed to continue decoding after this prefill.
    pub fn decode_delta(&self, logical_end: usize) -> Result<i64> {
        let max_position = self
            .temporal
            .iter()
            .chain(&self.height)
            .chain(&self.width)
            .copied()
            .max()
            .context("M-RoPE schedule cannot be empty")?;
        let logical_end = i64::try_from(logical_end)
            .map_err(|_| anyhow::anyhow!("logical end position exceeds i64"))?;
        Ok(i64::from(max_position) + 1 - logical_end)
    }
}

/// Input accepted by a model stage.
#[derive(Clone)]
pub enum StageInput {
    /// Tokens (head stage receives raw token IDs).
    Tokens(Vec<u32>),
    /// Hidden states from the previous stage.
    Activations {
        /// Activation tensor, usually shaped `[batch, sequence, hidden]`.
        tensor: Tensor,
        /// Image-token blocks that use Gemma 4's local bidirectional mask.
        bidirectional_spans: Vec<AttentionSpan>,
        /// Optional Qwen temporal/height/width rotary schedule.
        positions: Option<MropePositionIds>,
    },
}

impl StageInput {
    /// Wrap a tensor as ordinary activations without spans or M-RoPE positions.
    pub fn activations(tensor: Tensor) -> Self {
        Self::Activations {
            tensor,
            bidirectional_spans: Vec::new(),
            positions: None,
        }
    }

    /// Wrap activations with image-token bidirectional attention spans.
    pub fn activations_with_spans(tensor: Tensor, bidirectional_spans: Vec<AttentionSpan>) -> Self {
        Self::Activations {
            tensor,
            bidirectional_spans,
            positions: None,
        }
    }

    /// Wrap activations with an M-RoPE schedule and no attention spans.
    pub fn positioned_activations(tensor: Tensor, positions: MropePositionIds) -> Self {
        Self::Activations {
            tensor,
            bidirectional_spans: Vec::new(),
            positions: Some(positions),
        }
    }

    /// Wrap activations with both attention spans and an M-RoPE schedule.
    pub fn positioned_activations_with_spans(
        tensor: Tensor,
        bidirectional_spans: Vec<AttentionSpan>,
        positions: MropePositionIds,
    ) -> Self {
        Self::Activations {
            tensor,
            bidirectional_spans,
            positions: Some(positions),
        }
    }
}

/// Output from a pipeline stage's forward pass.
pub enum StageOutput {
    /// Hidden states to pass to the next stage.
    Activations(Tensor),
    /// Logits from the final stage (tail).
    Logits(Tensor),
}

/// Target weights shared with an attached speculative draft model.
///
/// Both values are cheap clones over the target's resident storage. DFlash
/// checkpoints do not carry their own token embedding or output head.
pub struct SpeculativeSharedWeights {
    pub(crate) token_embedding: Arc<QTensor>,
    pub(crate) output: qwen35::Qwen35QMatMul,
}

/// Dense f16 KV storage with spare sequence capacity.
///
/// Candle tensors are immutable at the graph level, but `scatter_set` can update
/// their backing storage in place. Keeping the sequence dimension contiguous
/// lets decode append only the incoming positions instead of rebuilding the
/// entire history with `Tensor::cat` on every token.
pub struct PreallocatedKvCache {
    pub(crate) k: Tensor,
    pub(crate) v: Tensor,
    pub(crate) start: usize,
    pub(crate) len: usize,
    pub(crate) capacity: usize,
    /// Number of lanes in the backing allocation. A one-lane view published
    /// from a decode batch still keeps the whole packed allocation alive.
    pub(crate) allocation_lanes: usize,
}

impl PreallocatedKvCache {
    const MIN_CAPACITY: usize = 64;
    const CAPACITY_ALIGNMENT: usize = 64;

    pub(crate) fn from_f16(k: Tensor, v: Tensor) -> Result<Self> {
        anyhow::ensure!(
            k.dtype() == candle_core::DType::F16,
            "KV key storage must be f16"
        );
        anyhow::ensure!(
            v.dtype() == candle_core::DType::F16,
            "KV value storage must be f16"
        );
        let (k_batch, k_heads, k_len, k_head_dim) = k.dims4()?;
        let (v_batch, v_heads, v_len, v_head_dim) = v.dims4()?;
        anyhow::ensure!(
            (k_batch, k_len) == (v_batch, v_len),
            "KV key/value batch or sequence dimensions differ"
        );
        let capacity = Self::next_capacity(0, k_len)?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        if capacity == k_len {
            return Ok(Self {
                k,
                v,
                start: 0,
                len: k_len,
                capacity,
                allocation_lanes: k_batch,
            });
        }
        let next_k = Tensor::zeros(
            (k_batch, k_heads, capacity, k_head_dim),
            candle_core::DType::F16,
            k.device(),
        )?;
        let next_v = Tensor::zeros(
            (v_batch, v_heads, capacity, v_head_dim),
            candle_core::DType::F16,
            v.device(),
        )?;
        Self::scatter_append(&next_k, &k, 0)?;
        Self::scatter_append(&next_v, &v, 0)?;
        Ok(Self {
            k: next_k,
            v: next_v,
            start: 0,
            len: k_len,
            capacity,
            allocation_lanes: k_batch,
        })
    }

    pub(crate) fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        use candle_core::DType;

        let (k_batch, k_heads, k_incoming, k_head_dim) = k.dims4()?;
        let (v_batch, v_heads, v_incoming, v_head_dim) = v.dims4()?;
        anyhow::ensure!(
            (k_batch, k_incoming) == (v_batch, v_incoming),
            "incoming KV batch or sequence dimensions differ"
        );
        let (stored_k_batch, stored_k_heads, _, stored_k_head_dim) = self.k.dims4()?;
        let (stored_v_batch, stored_v_heads, _, stored_v_head_dim) = self.v.dims4()?;
        anyhow::ensure!(
            (k_batch, k_heads, k_head_dim) == (stored_k_batch, stored_k_heads, stored_k_head_dim),
            "incoming key shape is incompatible with preallocated storage"
        );
        anyhow::ensure!(
            (v_batch, v_heads, v_head_dim) == (stored_v_batch, stored_v_heads, stored_v_head_dim),
            "incoming value shape is incompatible with preallocated storage"
        );
        anyhow::ensure!(k_incoming > 0, "cannot append an empty KV sequence");

        let append_start = self.start.saturating_add(self.len);
        if append_start.saturating_add(k_incoming) <= self.capacity {
            #[cfg(target_os = "macos")]
            if kv_append_metal::try_cast_append_pair(&self.k, &self.v, k, v, append_start)? {
                self.len += k_incoming;
                return Ok(());
            }
            #[cfg(feature = "cuda")]
            if kv_append_cuda::try_cast_append_pair(&self.k, &self.v, k, v, append_start)? {
                self.len += k_incoming;
                return Ok(());
            }

            let k = k.to_dtype(DType::F16)?.contiguous()?;
            let v = v.to_dtype(DType::F16)?.contiguous()?;
            // If the value write fails after the key write, `len` remains
            // unchanged and a retry safely overwrites the same unused slots.
            Self::scatter_append(&self.k, &k, append_start)?;
            Self::scatter_append(&self.v, &v, append_start)?;
            self.len += k_incoming;
            return Ok(());
        }

        // Grow or compact into fresh storage. The old cache remains logically
        // unchanged until both buffers have been populated successfully.
        let k = k.to_dtype(DType::F16)?.contiguous()?;
        let v = v.to_dtype(DType::F16)?.contiguous()?;
        let required = self.len.saturating_add(k_incoming);
        let capacity = if required <= self.capacity {
            self.capacity
        } else {
            Self::next_capacity(self.capacity, required)?
        };
        let next_k = Tensor::zeros(
            (k_batch, k_heads, capacity, k_head_dim),
            DType::F16,
            self.k.device(),
        )?;
        let next_v = Tensor::zeros(
            (v_batch, v_heads, capacity, v_head_dim),
            DType::F16,
            self.v.device(),
        )?;
        if self.len > 0 {
            let (active_k, active_v) = self.active_f16()?;
            Self::scatter_append(&next_k, &active_k.contiguous()?, 0)?;
            Self::scatter_append(&next_v, &active_v.contiguous()?, 0)?;
        }
        Self::scatter_append(&next_k, &k, self.len)?;
        Self::scatter_append(&next_v, &v, self.len)?;

        self.k = next_k;
        self.v = next_v;
        self.start = 0;
        self.len = required;
        self.capacity = capacity;
        self.allocation_lanes = k_batch;
        Ok(())
    }

    /// Append one decode token at a different logical position for each lane.
    ///
    /// The packed allocation is shared by a persistent mixed-position decode
    /// group. All lanes advance together, but their sequence lengths can differ.
    /// Device-side writes update every lane without copying KV history.
    pub(crate) fn append_ragged(
        &mut self,
        k: &Tensor,
        v: &Tensor,
        positions: &[usize],
    ) -> Result<()> {
        use candle_core::DType;

        anyhow::ensure!(self.start == 0, "ragged KV storage must start at zero");
        let (k_batch, k_heads, k_incoming, k_head_dim) = k.dims4()?;
        let (v_batch, v_heads, v_incoming, v_head_dim) = v.dims4()?;
        anyhow::ensure!(
            (k_batch, k_incoming) == (v_batch, v_incoming),
            "incoming KV batch or sequence dimensions differ"
        );
        let (stored_k_batch, stored_k_heads, _, stored_k_head_dim) = self.k.dims4()?;
        let (stored_v_batch, stored_v_heads, _, stored_v_head_dim) = self.v.dims4()?;
        anyhow::ensure!(k_incoming == 1, "ragged KV append is decode-only");
        anyhow::ensure!(
            positions.len() == k_batch,
            "ragged KV position count differs"
        );
        anyhow::ensure!(
            (k_batch, k_heads, k_head_dim) == (stored_k_batch, stored_k_heads, stored_k_head_dim),
            "incoming key shape is incompatible with ragged storage"
        );
        anyhow::ensure!(
            (v_batch, v_heads, v_head_dim) == (stored_v_batch, stored_v_heads, stored_v_head_dim),
            "incoming value shape is incompatible with ragged storage"
        );
        anyhow::ensure!(
            positions.iter().copied().max().unwrap_or(0) == self.len,
            "ragged KV metadata does not match lane positions"
        );
        anyhow::ensure!(
            positions
                .iter()
                .all(|position| *position <= u32::MAX as usize),
            "KV position exceeds scatter index range"
        );

        let required = positions
            .iter()
            .map(|position| position.saturating_add(1))
            .max()
            .unwrap_or(0);

        if required <= self.capacity {
            #[cfg(target_os = "macos")]
            if kv_append_metal::try_cast_append_ragged_pair(&self.k, &self.v, k, v, positions)? {
                self.len = required;
                return Ok(());
            }

            let k = k.to_dtype(DType::F16)?.contiguous()?;
            let v = v.to_dtype(DType::F16)?.contiguous()?;
            let (k_indices, v_indices) = Self::ragged_indices(positions, &k, &v)?;
            self.k.scatter_set(&k_indices, &k, 2)?;
            self.v.scatter_set(&v_indices, &v, 2)?;
            self.len = required;
            return Ok(());
        }

        let k = k.to_dtype(DType::F16)?.contiguous()?;
        let v = v.to_dtype(DType::F16)?.contiguous()?;
        let capacity = Self::next_capacity(self.capacity, required)?;
        let next_k = Tensor::zeros(
            (k_batch, k_heads, capacity, k_head_dim),
            DType::F16,
            self.k.device(),
        )?;
        let next_v = Tensor::zeros(
            (v_batch, v_heads, capacity, v_head_dim),
            DType::F16,
            self.v.device(),
        )?;
        Self::scatter_append(&next_k, &self.k, 0)?;
        Self::scatter_append(&next_v, &self.v, 0)?;
        let (k_indices, v_indices) = Self::ragged_indices(positions, &k, &v)?;
        next_k.scatter_set(&k_indices, &k, 2)?;
        next_v.scatter_set(&v_indices, &v, 2)?;

        self.k = next_k;
        self.v = next_v;
        self.len = required;
        self.capacity = capacity;
        self.allocation_lanes = k_batch;
        Ok(())
    }

    /// The caches and the sequence position at which `incoming` more rows
    /// fit without growing the allocation, for a kernel that writes the
    /// rows itself. Follow a successful write with [`Self::commit_append`].
    #[cfg(feature = "cuda")]
    pub(crate) fn append_slot(&self, incoming: usize) -> Option<(Tensor, Tensor, usize)> {
        let start = self.start.checked_add(self.len)?;
        (incoming > 0 && start.checked_add(incoming)? <= self.capacity)
            .then(|| (self.k.clone(), self.v.clone(), start))
    }

    /// Count `incoming` rows written at [`Self::append_slot`].
    #[cfg(feature = "cuda")]
    pub(crate) fn commit_append(&mut self, incoming: usize) {
        debug_assert!(self.start + self.len + incoming <= self.capacity);
        self.len += incoming;
    }

    /// Make room for `incoming` rows after the active span, growing or
    /// compacting the storage as [`Self::append`] would, and return the key
    /// and value storage (`[batch, heads, capacity, head_dim]` f16) with the
    /// row where the new entries go. The caller writes them in place, then
    /// calls [`Self::commit`].
    #[cfg_attr(not(any(target_os = "macos", feature = "cuda")), allow(dead_code))]
    pub(crate) fn reserve(&mut self, incoming: usize) -> Result<(Tensor, Tensor, usize)> {
        anyhow::ensure!(incoming > 0, "cannot reserve an empty KV sequence");
        if self.start.saturating_add(self.len).saturating_add(incoming) > self.capacity {
            let (batch, k_heads, _, k_head_dim) = self.k.dims4()?;
            let (_, v_heads, _, v_head_dim) = self.v.dims4()?;
            let required = self.len.saturating_add(incoming);
            let capacity = if required <= self.capacity {
                self.capacity
            } else {
                Self::next_capacity(self.capacity, required)?
            };
            let next_k = Tensor::zeros(
                (batch, k_heads, capacity, k_head_dim),
                candle_core::DType::F16,
                self.k.device(),
            )?;
            let next_v = Tensor::zeros(
                (batch, v_heads, capacity, v_head_dim),
                candle_core::DType::F16,
                self.v.device(),
            )?;
            if self.len > 0 {
                let (active_k, active_v) = self.active_f16()?;
                Self::scatter_append(&next_k, &active_k.contiguous()?, 0)?;
                Self::scatter_append(&next_v, &active_v.contiguous()?, 0)?;
            }
            self.k = next_k;
            self.v = next_v;
            self.start = 0;
            self.capacity = capacity;
            self.allocation_lanes = batch;
        }
        Ok((self.k.clone(), self.v.clone(), self.start + self.len))
    }

    /// Count `incoming` rows written in place after [`Self::reserve`] as
    /// active.
    #[cfg_attr(not(any(target_os = "macos", feature = "cuda")), allow(dead_code))]
    pub(crate) fn commit(&mut self, incoming: usize) -> Result<()> {
        anyhow::ensure!(
            self.start + self.len + incoming <= self.capacity,
            "KV commit exceeds the reserved capacity"
        );
        self.len += incoming;
        Ok(())
    }

    pub(crate) fn active_f16(&self) -> Result<(Tensor, Tensor)> {
        Ok((
            self.k.narrow(2, self.start, self.len)?,
            self.v.narrow(2, self.start, self.len)?,
        ))
    }

    fn evict_oldest(&mut self, count: usize) -> Result<()> {
        if count >= self.len {
            return Err(anyhow::anyhow!(
                "cannot evict {count} tokens from cache with {} tokens",
                self.len
            ));
        }
        self.start += count;
        self.len -= count;
        Ok(())
    }

    pub(crate) fn next_capacity(current: usize, required: usize) -> Result<usize> {
        let grown = if current == 0 {
            Self::MIN_CAPACITY
        } else {
            current
                .checked_add(current / 2)
                .ok_or_else(|| anyhow::anyhow!("KV capacity growth overflow"))?
        };
        grown
            .max(required)
            .checked_next_multiple_of(Self::CAPACITY_ALIGNMENT)
            .ok_or_else(|| anyhow::anyhow!("KV capacity alignment overflow"))
    }

    fn scatter_append(dst: &Tensor, src: &Tensor, start: usize) -> Result<()> {
        let (_, _, seq_len, _) = src.dims4()?;
        anyhow::ensure!(
            start.saturating_add(seq_len) <= dst.dim(2)?,
            "KV append exceeds preallocated capacity"
        );
        // Both tensors are contiguous in [batch, heads, sequence, head_dim]
        // order. `slice_set` lowers this append to a strided device copy and
        // avoids allocating, filling, and uploading a sequence-sized scatter
        // index tensor on every layer and decode token.
        dst.slice_set(src, 2, start).map_err(Into::into)
    }

    fn ragged_indices(positions: &[usize], k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch, _, k_seq, _) = k.dims4()?;
        let (v_batch, _, v_seq, _) = v.dims4()?;
        anyhow::ensure!(positions.len() == batch, "ragged KV position count differs");
        anyhow::ensure!(
            (batch, k_seq) == (v_batch, v_seq),
            "ragged KV batch or sequence dimensions differ"
        );
        let position_values = positions
            .iter()
            .map(|position| *position as u32)
            .collect::<Vec<_>>();
        let position_tensor = Tensor::from_vec(position_values, (batch, 1, 1, 1), k.device())?;
        let k_indices = position_tensor.broadcast_as(k.shape())?.contiguous()?;
        let shared_device = k.device().same_device(v.device());
        let v_indices = if shared_device && k.shape() == v.shape() {
            k_indices.clone()
        } else if shared_device {
            position_tensor.broadcast_as(v.shape())?.contiguous()?
        } else {
            let position_values = positions
                .iter()
                .map(|position| *position as u32)
                .collect::<Vec<_>>();
            Tensor::from_vec(position_values, (batch, 1, 1, 1), v.device())?
                .broadcast_as(v.shape())?
                .contiguous()?
        };
        Ok((k_indices, v_indices))
    }
}

/// Per-layer KV cache for autoregressive decoding.
/// Either dense f16 tensor storage, preallocated f16 storage, or TurboQuant.
pub enum KvCache {
    /// Legacy exact-size representation retained for compatibility with
    /// architecture tests and converted on the next cache update.
    F16 {
        /// Key cache tensor, conventionally `[batch, heads, sequence, head_dim]`.
        k: Tensor,
        /// Value cache tensor, conventionally `[batch, heads, sequence, head_dim]`.
        v: Tensor,
    },
    /// Preallocated dense f16 cache with reusable sequence capacity.
    PreallocatedF16(PreallocatedKvCache),
    /// TurboQuant-compressed cache representation.
    Quantized(QuantizedKvCache),
}

impl KvCache {
    /// Evict the oldest `count` token positions from the cache.
    pub fn evict_oldest(&mut self, count: usize) -> Result<()> {
        match self {
            KvCache::F16 { k, v } => {
                let seq = k.dim(2)?;
                if count >= seq {
                    return Err(anyhow::anyhow!(
                        "cannot evict {count} tokens from cache with {seq} tokens"
                    ));
                }
                *k = k.narrow(2, count, seq - count)?;
                *v = v.narrow(2, count, seq - count)?;
            }
            KvCache::PreallocatedF16(cache) => cache.evict_oldest(count)?,
            KvCache::Quantized(qkv) => qkv.evict_oldest(count),
        }
        Ok(())
    }

    fn discard_newest(&mut self, count: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        match self {
            KvCache::F16 { k, v } => {
                let seq = k.dim(2)?;
                anyhow::ensure!(count < seq, "cannot discard all {seq} KV tokens");
                *k = k.narrow(2, 0, seq - count)?;
                *v = v.narrow(2, 0, seq - count)?;
            }
            KvCache::PreallocatedF16(cache) => {
                anyhow::ensure!(
                    count < cache.len,
                    "cannot discard all {} KV tokens",
                    cache.len
                );
                cache.len -= count;
            }
            KvCache::Quantized(_) => {
                anyhow::bail!("speculative rollback does not support quantized KV caches")
            }
        }
        Ok(())
    }
}

/// Conservative token-slot width of a logical cache.
///
/// Packed batch views report the full backing allocation for each lane. This
/// can over-count a shared buffer while every lane remains live, but stays safe
/// when a lane leaves the group and its narrow view keeps that buffer alive.
pub(crate) fn kv_cache_allocation_tokens(cache: &[Option<LayerCache>]) -> Result<(usize, usize)> {
    let mut lane_capacity = 0usize;
    let mut allocation_slots = 0usize;
    for layer in cache {
        if let Some(LayerCache::DeepSeek4(cache)) = layer {
            let slots = cache.allocation_tokens();
            lane_capacity = lane_capacity.max(slots);
            allocation_slots = allocation_slots.max(slots);
            continue;
        }
        if let Some(LayerCache::DeepSeek41(cache)) = layer {
            let slots = cache.allocation_tokens();
            lane_capacity = lane_capacity.max(slots);
            allocation_slots = allocation_slots.max(slots);
            continue;
        }
        if let Some(LayerCache::Inkling2(cache)) = layer {
            let slots = cache.allocation_tokens();
            lane_capacity = lane_capacity.max(slots);
            allocation_slots = allocation_slots.max(slots);
            continue;
        }
        let Some(LayerCache::Attention(kv)) = layer else {
            continue;
        };
        let (layer_capacity, layer_slots) = match kv {
            KvCache::F16 { k, .. } => {
                let capacity = k.dim(2)?;
                (capacity, k.dim(0)?.saturating_mul(capacity))
            }
            KvCache::PreallocatedF16(dense) => (
                dense.capacity,
                dense.allocation_lanes.saturating_mul(dense.capacity),
            ),
            KvCache::Quantized(cache) => (cache.allocation_tokens(), cache.allocation_tokens()),
        };
        lane_capacity = lane_capacity.max(layer_capacity);
        allocation_slots = allocation_slots.max(layer_slots);
    }
    Ok((lane_capacity, allocation_slots))
}

/// DeltaNet recurrent state: fixed-size conv + recurrent state (does NOT grow with sequence).
pub struct DeltaNetCache {
    /// Causal convolution sliding window: [conv_dim, kernel_size]
    pub conv_state: Tensor,
    /// Recurrent state per head: [num_v_heads, key_head_dim, value_head_dim]
    pub recurrent_state: Tensor,
    /// Data used to keep an accepted target verification prefix.
    pub(crate) speculative_state: Option<DeltaNetSpeculativeState>,
}

pub(crate) enum DeltaNetSpeculativeState {
    Snapshots(DeltaNetSnapshots),
    #[cfg(target_os = "macos")]
    Replay(DeltaNetReplay),
}

pub(crate) struct DeltaNetSnapshots {
    /// `[rows, conv_dim, kernel_size]`.
    pub(crate) conv_state: Tensor,
    /// `[rows, heads, key_dim, value_dim]`.
    pub(crate) recurrent_state: Tensor,
}

#[cfg(target_os = "macos")]
pub(crate) struct DeltaNetReplay {
    /// State before the verification block: `[conv_dim, kernel_size]`.
    pub(crate) initial_conv_state: Tensor,
    /// Projected rows before the causal convolution: `[rows, conv_dim]`.
    pub(crate) conv_inputs: Tensor,
    /// State before the verification block: `[heads, key_dim, value_dim]`.
    pub(crate) initial_recurrent_state: Tensor,
    /// Normalized keys: `[rows, heads, key_dim]`.
    pub(crate) keys: Tensor,
    /// Delta-rule innovations: `[rows, heads, value_dim]`.
    pub(crate) deltas: Tensor,
    /// Per-row decay gates: `[rows, heads]`.
    pub(crate) decays: Tensor,
}

/// Mamba-2 SSM state: fixed-size conv + SSM state (does NOT grow with sequence).
pub struct Mamba2Cache {
    /// Causal convolution sliding window: [d_conv, kernel_size]
    pub conv_state: Tensor,
    /// SSM state per head: [n_heads, d_state, head_dim]
    pub ssm_state: Tensor,
}

/// Per-layer cache — either standard KV cache, DeltaNet, or Mamba-2 recurrent state.
pub enum LayerCache {
    /// DeepSeek V4.1 model-wide transactional cache, in slot zero only.
    DeepSeek41(Box<deepseek41::DeepSeek41Cache>),
    /// DeepSeek V4 window and learned compression state.
    DeepSeek4(Box<deepseek4::DeepSeek4Cache>),
    /// Standard attention key/value cache.
    Attention(KvCache),
    /// DeltaNet recurrent state.
    DeltaNet(DeltaNetCache),
    /// Mamba-2 recurrent state.
    Mamba2(Mamba2Cache),
    /// Inkling 2 per-visit KV and convolution state, in slot zero only.
    Inkling2(Box<inkling2_loop::Inkling2Cache>),
}

/// Clone cache metadata for a speculative checkpoint.
///
/// Dense appends only write unused slots, while recurrent kernels return new
/// state tensors. Sharing the old tensor storage therefore preserves the
/// complete pre-pass state without copying the KV history.
pub(crate) fn checkpoint_layer_cache(cache: &LayerCache) -> Result<LayerCache> {
    match cache {
        LayerCache::Inkling2(cache) => Ok(LayerCache::Inkling2(Box::new(cache.checkpoint()?))),
        LayerCache::DeepSeek41(cache) => Ok(LayerCache::DeepSeek41(cache.clone())),
        LayerCache::DeepSeek4(cache) => Ok(LayerCache::DeepSeek4(cache.clone())),
        LayerCache::Attention(KvCache::F16 { k, v }) => Ok(LayerCache::Attention(KvCache::F16 {
            k: k.clone(),
            v: v.clone(),
        })),
        LayerCache::Attention(KvCache::PreallocatedF16(cache)) => Ok(LayerCache::Attention(
            KvCache::PreallocatedF16(PreallocatedKvCache {
                k: cache.k.clone(),
                v: cache.v.clone(),
                start: cache.start,
                len: cache.len,
                capacity: cache.capacity,
                allocation_lanes: cache.allocation_lanes,
            }),
        )),
        LayerCache::Attention(KvCache::Quantized(_)) => {
            anyhow::bail!("speculative checkpoints do not support quantized KV caches")
        }
        LayerCache::DeltaNet(cache) => Ok(LayerCache::DeltaNet(DeltaNetCache {
            conv_state: cache.conv_state.clone(),
            recurrent_state: cache.recurrent_state.clone(),
            speculative_state: None,
        })),
        LayerCache::Mamba2(cache) => Ok(LayerCache::Mamba2(Mamba2Cache {
            conv_state: cache.conv_state.clone(),
            ssm_state: cache.ssm_state.clone(),
        })),
    }
}

/// Copy live state into private storage. A rollback checkpoint alone cannot
/// safely feed two requests which append at the same position.
pub(crate) fn fork_layer_cache(cache: &LayerCache) -> Result<LayerCache> {
    Ok(match cache {
        LayerCache::Inkling2(cache) => LayerCache::Inkling2(Box::new(cache.fork()?)),
        LayerCache::DeepSeek41(cache) => LayerCache::DeepSeek41(cache.clone()),
        // V4 replaces immutable tensor chunks; it never writes shared rows.
        LayerCache::DeepSeek4(cache) => LayerCache::DeepSeek4(cache.clone()),
        LayerCache::Attention(KvCache::F16 { k, v }) => LayerCache::Attention(KvCache::F16 {
            k: copy_persistent_tensor(k)?,
            v: copy_persistent_tensor(v)?,
        }),
        LayerCache::Attention(KvCache::PreallocatedF16(cache)) => {
            let (k, v) = cache.active_f16()?;
            // Keep the batch-compatible format, but copy only live rows. This
            // keeps snapshot bytes/slot charges exact and drops packed peers.
            let allocation_lanes = k.dim(0)?;
            LayerCache::Attention(KvCache::PreallocatedF16(PreallocatedKvCache {
                k: copy_persistent_tensor(&k)?,
                v: copy_persistent_tensor(&v)?,
                start: 0,
                len: cache.len,
                capacity: cache.len,
                allocation_lanes,
            }))
        }
        LayerCache::Attention(KvCache::Quantized(_)) => {
            anyhow::bail!("prompt reuse does not support quantized KV")
        }
        LayerCache::DeltaNet(cache) => LayerCache::DeltaNet(DeltaNetCache {
            conv_state: copy_persistent_tensor(&cache.conv_state)?,
            recurrent_state: copy_persistent_tensor(&cache.recurrent_state)?,
            speculative_state: None,
        }),
        LayerCache::Mamba2(cache) => LayerCache::Mamba2(Mamba2Cache {
            conv_state: copy_persistent_tensor(&cache.conv_state)?,
            ssm_state: copy_persistent_tensor(&cache.ssm_state)?,
        }),
    })
}

/// Allocation size for a persistent F32 state tensor, including Metal buckets.
pub(crate) fn fixed_state_tensor_bytes(dimensions: &[usize], device: &Device) -> Result<usize> {
    let bytes = dimensions
        .iter()
        .try_fold(size_of::<f32>(), |bytes, &dim| {
            bytes
                .checked_mul(dim)
                .context("fixed request state size overflow")
        })?;
    if device.is_metal() {
        bytes
            .checked_next_power_of_two()
            .context("fixed Metal state allocation overflow")
    } else {
        Ok(bytes)
    }
}

fn state_storage_bytes(tensor: &Tensor) -> Result<usize> {
    let (storage, _) = tensor.storage_and_layout();
    let elements = match &*storage {
        candle_core::Storage::Cpu(candle_core::CpuStorage::F32(values)) => values.len(),
        #[cfg(target_os = "macos")]
        candle_core::Storage::Metal(storage) => return Ok(storage.buffer().length()),
        #[cfg(feature = "cuda")]
        candle_core::Storage::Cuda(storage) => storage.as_cuda_slice::<f32>()?.len(),
        _ => anyhow::bail!("fixed request state must use F32 storage"),
    };
    elements
        .checked_mul(size_of::<f32>())
        .context("fixed state storage size overflow")
}

/// Copy a persistent cache view without retaining a pooled temporary buffer.
pub(crate) fn copy_persistent_tensor(tensor: &Tensor) -> Result<Tensor> {
    #[cfg(target_os = "macos")]
    if tensor.device().is_metal() {
        use candle_core::backend::BackendStorage;

        let (storage, layout) = tensor.storage_and_layout();
        let candle_core::Storage::Metal(source) = &*storage else {
            anyhow::bail!("Metal cache tensor has the wrong storage");
        };
        let elements = tensor.elem_count();
        elements
            .checked_mul(tensor.dtype().size_in_bytes())
            .context("persistent cache size overflow")?;
        let device = source.device().clone();
        // Candle's pool can lend a larger buffer than requested. Persistent
        // state needs its own exact allocation to keep its byte limit valid.
        let buffer = device.new_private_buffer(elements, tensor.dtype(), "persistent-cache")?;
        let mut destination =
            candle_core::MetalStorage::new(buffer, device, elements, tensor.dtype());
        source.copy_strided_src(&mut destination, 0, layout)?;
        return Ok(Tensor::from((
            candle_core::Storage::Metal(destination),
            tensor.shape().clone(),
        )));
    }
    tensor.force_contiguous().map_err(Into::into)
}

/// Remove large temporary backing buffers from persistent state views.
pub(crate) fn compact_fixed_layer_cache(cache: &mut LayerCache) -> Result<()> {
    let tensors = match cache {
        LayerCache::DeltaNet(cache) => [&mut cache.conv_state, &mut cache.recurrent_state],
        LayerCache::Mamba2(cache) => [&mut cache.conv_state, &mut cache.ssm_state],
        _ => return Ok(()),
    };
    for tensor in tensors {
        if state_storage_bytes(tensor)? > fixed_state_tensor_bytes(tensor.dims(), tensor.device())?
        {
            *tensor = copy_persistent_tensor(tensor)?;
        }
    }
    Ok(())
}

pub(crate) fn live_layer_cache_bytes(cache: &LayerCache) -> Result<usize> {
    let (a, b) = match cache {
        LayerCache::DeepSeek41(cache) => return Ok(cache.bytes()),
        LayerCache::DeepSeek4(cache) => return Ok(cache.bytes()),
        LayerCache::Inkling2(cache) => return Ok(cache.bytes()),
        LayerCache::Attention(KvCache::F16 { k, v }) => (k.clone(), v.clone()),
        LayerCache::Attention(KvCache::PreallocatedF16(cache)) => cache.active_f16()?,
        LayerCache::Attention(KvCache::Quantized(_)) => {
            anyhow::bail!("prompt reuse does not support quantized KV")
        }
        LayerCache::DeltaNet(cache) => (cache.conv_state.clone(), cache.recurrent_state.clone()),
        LayerCache::Mamba2(cache) => (cache.conv_state.clone(), cache.ssm_state.clone()),
    };
    Ok(a.elem_count()
        .saturating_mul(a.dtype().size_in_bytes())
        .saturating_add(b.elem_count().saturating_mul(b.dtype().size_in_bytes())))
}

/// Keep a verified target prefix without running the target a second time.
pub(crate) fn commit_speculative_layer_cache(
    cache: &mut LayerCache,
    committed_row: usize,
    discarded_rows: usize,
) -> Result<()> {
    match cache {
        LayerCache::DeepSeek41(_) => {
            anyhow::bail!("V4.1 speculative commit is unsupported; restore a complete checkpoint")
        }
        LayerCache::Attention(cache) => cache.discard_newest(discarded_rows),
        LayerCache::DeepSeek4(_) => {
            anyhow::bail!("V4 speculative rollback requires a full checkpoint")
        }
        LayerCache::Inkling2(_) => {
            anyhow::bail!("Inkling 2 speculative rollback requires a full checkpoint")
        }
        LayerCache::DeltaNet(cache) => {
            let speculative_state = cache
                .speculative_state
                .take()
                .context("Qwen DeltaNet verification did not retain rollback states")?;
            match speculative_state {
                DeltaNetSpeculativeState::Snapshots(snapshots) => {
                    anyhow::ensure!(
                        committed_row < snapshots.conv_state.dim(0)?
                            && committed_row < snapshots.recurrent_state.dim(0)?,
                        "Qwen DeltaNet rollback row is out of range"
                    );
                    cache.conv_state = snapshots
                        .conv_state
                        .narrow(0, committed_row, 1)?
                        .squeeze(0)?;
                    cache.recurrent_state = snapshots
                        .recurrent_state
                        .narrow(0, committed_row, 1)?
                        .squeeze(0)?;
                    Ok(())
                }
                #[cfg(target_os = "macos")]
                DeltaNetSpeculativeState::Replay(replay) => {
                    let rows = replay.conv_inputs.dim(0)?;
                    anyhow::ensure!(
                        committed_row < rows
                            && replay.keys.dim(0)? == rows
                            && replay.deltas.dim(0)? == rows
                            && replay.decays.dim(0)? == rows,
                        "Qwen DeltaNet replay row is out of range"
                    );
                    if discarded_rows == 0 {
                        return Ok(());
                    }
                    let replay_len = committed_row + 1;
                    let kernel_size = replay.initial_conv_state.dim(1)?;
                    let history = Tensor::cat(
                        &[
                            &replay.initial_conv_state,
                            &replay.conv_inputs.transpose(0, 1)?,
                        ],
                        1,
                    )?;
                    cache.conv_state = history.narrow(1, replay_len, kernel_size)?.contiguous()?;
                    cache.recurrent_state = qwen35_deltanet_metal::replay(
                        &replay.initial_recurrent_state,
                        &replay.keys,
                        &replay.deltas,
                        &replay.decays,
                        replay_len,
                    )?;
                    Ok(())
                }
            }
        }
        LayerCache::Mamba2(_) => {
            anyhow::bail!("speculative rollback does not support Mamba state")
        }
    }
}

/// Trait for model architectures that can run as pipeline stages.
///
/// Each architecture implements its own attention, FFN, and normalization.
/// The pipeline infrastructure treats this as an opaque forward pass.
pub trait ModelArchitecture: Send + Sync {
    /// Bytes reserved per live request outside its attention KV cache.
    /// Include fixed recurrent state and retained verification state.
    fn fixed_cache_bytes(&self, _verification_rows: usize) -> Result<usize> {
        Ok(0)
    }

    /// Whether independent draft blocks can share target matrix operations.
    fn supports_speculative_batch(&self) -> bool {
        false
    }

    /// Verify equal-sized draft blocks, retaining lane-local rollback state.
    /// Implementations must not publish cache changes on error.
    fn forward_speculative_batch(
        &self,
        _inputs: &[Vec<u32>],
        _caches: &mut [&mut [Option<LayerCache>]],
        _positions: &[usize],
        _layer_ids: &[usize],
    ) -> Result<Vec<SpeculativeBatchOutput>> {
        anyhow::bail!(
            "{} does not support draft verification batches",
            self.family()
        )
    }

    /// Run one model pass.
    ///
    /// `start_pos` is the absolute token position used for RoPE and other
    /// position-aware model math. It is not the retained KV length after
    /// rolling eviction. Attention code must get that length from the updated
    /// cache tensors when it builds masks.
    fn forward(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput>;

    /// Process prompt tokens, including a chunk containing just one token.
    /// Architectures may need different kernels from one-token decode.
    fn forward_prefill(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward(input, cache, seq_len, start_pos)
    }

    /// Advance prompt cache state when the caller will discard the output.
    /// A model may skip its final norm and vocabulary projection, but must run
    /// every decoder layer. The fallback retains the model's prefill kernels,
    /// including special handling for one-token chunks.
    fn forward_prefill_cache_only(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_prefill(input, cache, seq_len, start_pos)
    }

    /// Arithmetic provenance for an explicitly bounded text request.
    /// Unchanged architectures return None and keep their existing paths.
    fn text_arithmetic_policy(&self, _context: TextRequestContext) -> Option<TextArithmeticPolicy> {
        None
    }

    /// Process text with a complete request bound supplied by the caller.
    /// The default preserves the architecture's existing prefill/decode paths.
    #[allow(clippy::too_many_arguments)]
    fn forward_text(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
        context: TextRequestContext,
        prefill: bool,
        cache_only: bool,
    ) -> Result<StageOutput> {
        context.validate(start_pos, seq_len, prefill)?;
        if cache_only {
            anyhow::ensure!(prefill, "cache-only text requires prefill");
            self.forward_prefill_cache_only(input, cache, seq_len, start_pos)
        } else if prefill {
            self.forward_prefill(input, cache, seq_len, start_pos)
        } else {
            self.forward(input, cache, seq_len, start_pos)
        }
    }

    /// Run a target pass while retaining selected transformer-layer inputs.
    fn forward_with_layer_inputs(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
        layer_ids: &[usize],
        all_logits: bool,
    ) -> Result<(StageOutput, Vec<Tensor>)> {
        anyhow::ensure!(
            layer_ids.is_empty() && !all_logits,
            "{} does not expose speculative target state",
            self.family()
        );
        self.forward(input, cache, seq_len, start_pos)
            .map(|output| (output, Vec::new()))
    }

    /// Return the target embedding and output head needed by a draft model.
    fn speculative_shared_weights(&self) -> Option<SpeculativeSharedWeights> {
        None
    }

    /// Embed raw token IDs without running decoder layers. Multimodal head
    /// stages use this to splice vision embeddings into an ordered prompt.
    fn embed_tokens(&self, _tokens: &[u32]) -> Result<Tensor> {
        anyhow::bail!("{} does not expose token embeddings", self.family())
    }

    /// Whether this architecture can execute a solo decode batch.
    fn supports_decode_batch(&self) -> bool {
        false
    }

    /// Whether a split stage can batch independent one-token inputs. This is
    /// separate from solo batching: heads return activations, not logits.
    fn supports_pipeline_decode_batch(&self) -> bool {
        false
    }

    /// Execute independent split-stage decode inputs in the supplied order.
    /// Heads accept one token per lane; other stages accept one flat F32
    /// activation row without multimodal metadata. Each lane owns its cache.
    /// Execution failures require clearing all selected sessions.
    fn forward_pipeline_decode_batch(
        &self,
        _inputs: &[StageInput],
        _caches: &mut [&mut [Option<LayerCache>]],
        _batch_cache: &mut [Option<LayerCache>],
        _positions: &[usize],
    ) -> Result<Vec<StageOutput>> {
        anyhow::bail!(
            "{} does not support pipeline decode batching",
            self.family()
        )
    }

    /// Whether this batch implementation requires preallocated f16 attention
    /// storage. Other cache formats must take scalar steps before batching.
    fn decode_batch_requires_preallocated_attention(&self) -> bool {
        false
    }

    /// Whether decode batching may retain one shared KV allocation across
    /// lanes. Independent-cache implementations avoid the packed-lane charge
    /// during the batch memory preflight; actual allocations are still checked.
    fn decode_batch_uses_packed_cache(&self) -> bool {
        true
    }

    /// Execute one decode token for several independently cached solo sessions.
    /// Implementations must preserve lane order in the returned logits and
    /// report storage requirements before execution. The stage clears selected
    /// sessions after an execution error; only preflight errors can retry scalar.
    fn forward_decode_batch(
        &self,
        _tokens: &[u32],
        _caches: &mut [&mut [Option<LayerCache>]],
        _batch_cache: &mut [Option<LayerCache>],
        _positions: &[usize],
    ) -> Result<Vec<Tensor>> {
        anyhow::bail!("{} does not support decode batching", self.family())
    }

    /// Whether single-token activations passed between stages should use F32.
    ///
    /// Most architectures tolerate F16 activations between stages. Architectures
    /// that are numerically sensitive across stage boundaries can opt in to
    /// F32 for decode while leaving the larger prefill packets in F16.
    fn prefers_f32_decode_activations(&self) -> bool {
        false
    }

    /// Whether multi-token prefill activations passed between stages should
    /// use F32. Most architectures use F16 for the larger prefill payloads, but
    /// architectures with recurrent state carried in the activation (such as
    /// GLM5's four hyperconnection streams) can opt into F32 for both phases.
    fn prefers_f32_prefill_activations(&self) -> bool {
        false
    }

    /// Architecture family name (e.g., "llama", "phi", "gemma").
    fn family(&self) -> &str;

    /// Hidden dimension size.
    fn hidden_dim(&self) -> usize;

    /// Number of layers in this stage.
    fn layer_count(&self) -> usize;

    /// Vocabulary size (only meaningful for tail stages with lm_head).
    fn vocab_size(&self) -> usize;
}

/// Target rows and draft features belonging to one request in a shared pass.
pub struct SpeculativeBatchOutput {
    /// One vocabulary row for each input token, in order.
    pub logits: Tensor,
    /// Requested layer inputs, in layer order and without other requests' rows.
    pub captures: Vec<Tensor>,
}

/// Update the KV cache and return tensors in its native attention dtype.
///
/// Dense caches return zero-copy f16 views into preallocated storage. Quantized
/// caches reconstruct to f16 on Metal and f32 elsewhere. Callers with an f32-only
/// attention implementation should use [`kv_cache_update`] instead.
pub fn kv_cache_update_native(
    kv_cache: &mut Option<KvCache>,
    k: Tensor,
    v: Tensor,
    quant: Option<&TurboQuantConfig>,
) -> Result<(Tensor, Tensor)> {
    match kv_cache {
        Some(KvCache::Quantized(qkv)) => qkv.update_native(&k, &v),
        Some(KvCache::PreallocatedF16(dense)) => {
            dense.append(&k, &v)?;
            dense.active_f16()
        }
        Some(KvCache::F16 { k: ck, v: cv }) => {
            // Build the replacement separately so a failed allocation or
            // scatter leaves the legacy cache available for a scalar retry.
            let mut dense = PreallocatedKvCache::from_f16(ck.clone(), cv.clone())?;
            dense.append(&k, &v)?;
            let (active_k, active_v) = dense.active_f16()?;
            *kv_cache = Some(KvCache::PreallocatedF16(dense));
            Ok((active_k, active_v))
        }
        None => {
            if let Some(cfg) = quant {
                let num_heads = k.dim(1)?;
                let mut qkv = QuantizedKvCache::new(
                    // `cfg` is already a constructed configuration. Preserve
                    // the legacy constructor's larger valid geometries here;
                    // the defensive bound belongs at the untrusted GGUF/API
                    // construction boundary, not while cloning trusted state.
                    Arc::new(cfg.clone()),
                    num_heads,
                );
                let result = qkv.update_native(&k, &v)?;
                *kv_cache = Some(KvCache::Quantized(qkv));
                Ok(result)
            } else {
                let dense = PreallocatedKvCache::from_f16(
                    k.to_dtype(candle_core::DType::F16)?,
                    v.to_dtype(candle_core::DType::F16)?,
                )?;
                let (active_k, active_v) = dense.active_f16()?;
                *kv_cache = Some(KvCache::PreallocatedF16(dense));
                Ok((active_k, active_v))
            }
        }
    }
}

/// Update KV cache and materialize f32 tensors for portable attention paths.
pub fn kv_cache_update(
    kv_cache: &mut Option<KvCache>,
    k: Tensor,
    v: Tensor,
    quant: Option<&TurboQuantConfig>,
) -> Result<(Tensor, Tensor)> {
    use candle_core::DType;

    let (k, v) = kv_cache_update_native(kv_cache, k, v, quant)?;
    if k.dtype() == DType::F32 && v.dtype() == DType::F32 {
        Ok((k, v))
    } else {
        Ok((k.to_dtype(DType::F32)?, v.to_dtype(DType::F32)?))
    }
}

type PreparedQuantizedAttention = (Tensor, Tensor, Tensor, Option<Tensor>);

/// For long single-token Metal decode, attend in the quantizer's orthogonal
/// basis. Only Q and the final output need a WHT; old K/V vectors need a cheap
/// centroid lookup instead of a full inverse transform on every decode step.
/// Short decode/prefill keep the ordinary path to avoid extra dispatches.
pub(crate) fn prepare_quantized_attention(
    kv_cache: &mut Option<KvCache>,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    quant: Option<&TurboQuantConfig>,
) -> Result<PreparedQuantizedAttention> {
    #[cfg(target_os = "macos")]
    if q.device().is_metal()
        && matches!(q.dtype(), candle_core::DType::F16 | candle_core::DType::F32)
        && q.dim(2)? == 1
        && let Some(config) = quant
        && config.bits() == 8
        && config.head_dim() <= crate::turboquant_metal::MAX_METAL_HEAD_DIM
    {
        let previous_len = match kv_cache {
            Some(KvCache::Quantized(cache)) => Some(cache.seq_len()),
            None => Some(0),
            _ => None,
        };
        if let Some(previous_len) = previous_len
            && previous_len
                .checked_add(k.dim(2)?)
                .context("KV attention length overflow")?
                >= 256
        {
            if kv_cache.is_none() {
                *kv_cache = Some(KvCache::Quantized(QuantizedKvCache::new(
                    Arc::new(config.clone()),
                    k.dim(1)?,
                )));
            }
            let Some(KvCache::Quantized(cache)) = kv_cache else {
                unreachable!()
            };
            let (k, v) = cache.update_rotated_native(&k, &v)?;
            let signs = cache.attention_rotation_signs()?;
            // Fuse the F16 conversion only when native SDPA will consume it.
            // Unsupported geometries retain the original query precision for
            // the portable attention fallback.
            let q = if tensor_fused_sdpa_geometry(&q, &k, &v)?.is_some_and(|geometry| {
                metal_sdpa_vector_head_dim_supported(geometry.qk_dim)
                    && geometry.v_dim <= geometry.qk_dim
            }) {
                crate::turboquant_metal::rotate_attention_native_query(&q, &signs)?
            } else {
                crate::turboquant_metal::rotate_attention(&q, &signs, false)?
            };
            return Ok((q, k, v, Some(signs)));
        }
    }
    let (k, v) = kv_cache_update_native(kv_cache, k, v, quant)?;
    Ok((q, k, v, None))
}

pub(crate) fn restore_quantized_attention_output(
    output: Tensor,
    rotation: Option<Tensor>,
) -> Result<Tensor> {
    #[cfg(target_os = "macos")]
    if let Some(signs) = rotation {
        return crate::turboquant_metal::rotate_attention(&output, &signs, true)
            .map_err(Into::into);
    }
    #[cfg(not(target_os = "macos"))]
    let _ = rotation;
    Ok(output)
}

/// Pack equally sized per-session KV allocations into one batch allocation.
/// Architectures can reuse this for uniform-position continuous decode.
pub(crate) fn pack_decode_layer_cache(
    caches: &[&mut [Option<LayerCache>]],
    layer_idx: usize,
) -> Result<Option<KvCache>> {
    anyhow::ensure!(
        caches.iter().all(|cache| layer_idx < cache.len()),
        "decode batch cache is missing layer {layer_idx}"
    );
    let populated = caches[0][layer_idx].is_some();
    anyhow::ensure!(
        caches
            .iter()
            .all(|cache| cache[layer_idx].is_some() == populated),
        "decode batch mixes populated and empty caches"
    );
    if !populated {
        return Ok(None);
    }
    anyhow::ensure!(
        caches.iter().all(|cache| matches!(
            &cache[layer_idx],
            Some(LayerCache::Attention(KvCache::PreallocatedF16(_)))
        )),
        "decode batching currently requires preallocated f16 attention caches"
    );

    let mut keys = Vec::with_capacity(caches.len());
    let mut values = Vec::with_capacity(caches.len());
    let Some(LayerCache::Attention(KvCache::PreallocatedF16(first))) = &caches[0][layer_idx] else {
        unreachable!("cache representation validated above")
    };
    // A compact prompt hit and a cold lane can have the same position but
    // different capacities. Repack live rows with room for the next token.
    // The same path avoids a growth copy for a full, compact prompt snapshot.
    let mut repack = first.start == 0 && first.len == first.capacity;
    for cache in caches {
        let Some(LayerCache::Attention(KvCache::PreallocatedF16(dense))) = &cache[layer_idx] else {
            unreachable!("cache representation validated above")
        };
        anyhow::ensure!(dense.len == first.len, "decode batch KV lengths differ");
        repack |= dense.start != first.start || dense.capacity != first.capacity;
    }
    if repack {
        return pack_ragged_decode_layer_cache(caches, layer_idx, &vec![first.len; caches.len()]);
    }
    for cache in caches {
        let Some(LayerCache::Attention(KvCache::PreallocatedF16(dense))) = &cache[layer_idx] else {
            unreachable!("cache shape validated above")
        };
        anyhow::ensure!(
            dense.start == first.start
                && dense.len == first.len
                && dense.capacity == first.capacity,
            "decode batch KV allocation layouts differ"
        );
        anyhow::ensure!(
            dense.k.dims()[1..] == first.k.dims()[1..]
                && dense.v.dims()[1..] == first.v.dims()[1..],
            "decode batch KV tensor shapes differ"
        );
        keys.push(dense.k.clone());
        values.push(dense.v.clone());
    }
    let key_refs = keys.iter().collect::<Vec<_>>();
    let value_refs = values.iter().collect::<Vec<_>>();
    Ok(Some(KvCache::PreallocatedF16(PreallocatedKvCache {
        k: Tensor::cat(&key_refs, 0)?,
        v: Tensor::cat(&value_refs, 0)?,
        start: first.start,
        len: first.len,
        capacity: first.capacity,
        allocation_lanes: caches.len(),
    })))
}

pub(crate) fn clone_decode_layer_cache(cache: &LayerCache) -> Result<Option<KvCache>> {
    match cache {
        LayerCache::Attention(KvCache::PreallocatedF16(dense)) => {
            Ok(Some(KvCache::PreallocatedF16(PreallocatedKvCache {
                k: dense.k.clone(),
                v: dense.v.clone(),
                start: dense.start,
                len: dense.len,
                capacity: dense.capacity,
                allocation_lanes: dense.allocation_lanes,
            })))
        }
        LayerCache::Attention(KvCache::F16 { k, v }) => Ok(Some(KvCache::F16 {
            k: k.clone(),
            v: v.clone(),
        })),
        LayerCache::Attention(KvCache::Quantized(_)) => {
            anyhow::bail!("persistent decode batch unexpectedly used a quantized KV cache")
        }
        _ => anyhow::bail!("persistent decode batch has non-attention state"),
    }
}

pub(crate) fn publish_decode_layer_cache_views(
    caches: &mut [&mut [Option<LayerCache>]],
    layer_idx: usize,
    batched: &Option<KvCache>,
) -> Result<()> {
    match batched {
        None => {
            for cache in caches {
                cache[layer_idx] = None;
            }
        }
        Some(KvCache::F16 { k, v }) => {
            anyhow::ensure!(
                k.dim(0)? == caches.len() && v.dim(0)? == caches.len(),
                "batched KV cache lane count changed"
            );
            for (lane, cache) in caches.iter_mut().enumerate() {
                cache[layer_idx] = Some(LayerCache::Attention(KvCache::F16 {
                    k: k.narrow(0, lane, 1)?,
                    v: v.narrow(0, lane, 1)?,
                }));
            }
        }
        Some(KvCache::PreallocatedF16(dense)) => {
            anyhow::ensure!(
                dense.k.dim(0)? == caches.len() && dense.v.dim(0)? == caches.len(),
                "batched preallocated KV cache lane count changed"
            );
            for (lane, cache) in caches.iter_mut().enumerate() {
                cache[layer_idx] = Some(LayerCache::Attention(KvCache::PreallocatedF16(
                    PreallocatedKvCache {
                        k: dense.k.narrow(0, lane, 1)?,
                        v: dense.v.narrow(0, lane, 1)?,
                        start: dense.start,
                        len: dense.len,
                        capacity: dense.capacity,
                        allocation_lanes: dense.allocation_lanes,
                    },
                )));
            }
        }
        Some(KvCache::Quantized(_)) => {
            anyhow::bail!("batched decode unexpectedly produced a quantized KV cache")
        }
    }
    Ok(())
}

pub(crate) fn pack_ragged_decode_layer_cache(
    caches: &[&mut [Option<LayerCache>]],
    layer_idx: usize,
    positions: &[usize],
) -> Result<Option<KvCache>> {
    anyhow::ensure!(
        caches.len() == positions.len(),
        "ragged decode lane mismatch"
    );
    anyhow::ensure!(
        caches.iter().all(|cache| layer_idx < cache.len()),
        "decode batch cache is missing layer {layer_idx}"
    );
    let max_position = positions.iter().copied().max().unwrap_or(0);
    if max_position == 0 {
        anyhow::ensure!(
            caches.iter().all(|cache| cache[layer_idx].is_none()),
            "empty ragged decode positions have populated caches"
        );
        return Ok(None);
    }

    let first = caches
        .iter()
        .find_map(|cache| match &cache[layer_idx] {
            Some(LayerCache::Attention(KvCache::PreallocatedF16(dense))) => Some(dense),
            _ => None,
        })
        .ok_or_else(|| {
            anyhow::anyhow!("ragged decode has positions without preallocated KV storage")
        })?;
    let (_, k_heads, _, k_head_dim) = first.k.dims4()?;
    let (_, v_heads, _, v_head_dim) = first.v.dims4()?;
    let required = max_position
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("ragged KV position overflow"))?;
    let capacity = PreallocatedKvCache::next_capacity(0, required)?;
    let packed_k = Tensor::zeros(
        (caches.len(), k_heads, capacity, k_head_dim),
        candle_core::DType::F16,
        first.k.device(),
    )?;
    let packed_v = Tensor::zeros(
        (caches.len(), v_heads, capacity, v_head_dim),
        candle_core::DType::F16,
        first.v.device(),
    )?;

    for (lane, (cache, position)) in caches.iter().zip(positions).enumerate() {
        if *position == 0 {
            anyhow::ensure!(
                cache[layer_idx].is_none(),
                "zero-position ragged lane has a populated cache"
            );
            continue;
        }
        let Some(LayerCache::Attention(KvCache::PreallocatedF16(dense))) = &cache[layer_idx] else {
            anyhow::bail!("ragged decode requires preallocated f16 attention caches")
        };
        anyhow::ensure!(
            dense.len == *position,
            "ragged cache length differs from position"
        );
        anyhow::ensure!(
            dense.k.dim(0)? == 1
                && dense.v.dim(0)? == 1
                && dense.k.dims()[1] == k_heads
                && dense.v.dims()[1] == v_heads
                && dense.k.dim(3)? == k_head_dim
                && dense.v.dim(3)? == v_head_dim,
            "ragged KV tensor shapes differ"
        );
        let (active_k, active_v) = dense.active_f16()?;
        let lane_k = packed_k.narrow(0, lane, 1)?;
        let lane_v = packed_v.narrow(0, lane, 1)?;
        PreallocatedKvCache::scatter_append(&lane_k, &active_k.contiguous()?, 0)?;
        PreallocatedKvCache::scatter_append(&lane_v, &active_v.contiguous()?, 0)?;
    }

    Ok(Some(KvCache::PreallocatedF16(PreallocatedKvCache {
        k: packed_k,
        v: packed_v,
        start: 0,
        len: max_position,
        capacity,
        allocation_lanes: caches.len(),
    })))
}

pub(crate) fn publish_ragged_decode_layer_cache_views(
    caches: &mut [&mut [Option<LayerCache>]],
    layer_idx: usize,
    batched: &Option<KvCache>,
    valid_lengths: &[usize],
) -> Result<()> {
    anyhow::ensure!(
        caches.len() == valid_lengths.len(),
        "ragged publish lane mismatch"
    );
    let Some(KvCache::PreallocatedF16(dense)) = batched else {
        anyhow::bail!("ragged decode did not produce preallocated f16 storage")
    };
    anyhow::ensure!(
        dense.k.dim(0)? == caches.len() && dense.v.dim(0)? == caches.len(),
        "batched ragged KV cache lane count changed"
    );
    for (lane, (cache, valid_len)) in caches.iter_mut().zip(valid_lengths).enumerate() {
        anyhow::ensure!(
            *valid_len <= dense.capacity,
            "ragged KV length exceeds capacity"
        );
        cache[layer_idx] = Some(LayerCache::Attention(KvCache::PreallocatedF16(
            PreallocatedKvCache {
                k: dense.k.narrow(0, lane, 1)?,
                v: dense.v.narrow(0, lane, 1)?,
                start: 0,
                len: *valid_len,
                capacity: dense.capacity,
                allocation_lanes: dense.allocation_lanes,
            },
        )));
    }
    Ok(())
}

/// Use fused CUDA or Metal SDPA directly on native f16 KV-cache views.
pub fn try_fused_f16_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    scale: f32,
) -> Result<Option<Tensor>> {
    let q_dtype = q.dtype();
    let Some(output) = try_fused_f16_sdpa_native(q, k, v, causal, scale)? else {
        return Ok(None);
    };
    if q_dtype == candle_core::DType::F16 {
        Ok(Some(output))
    } else {
        Ok(Some(output.to_dtype(q_dtype)?))
    }
}

/// Retain the native f16 output produced by the selected fused backend.
pub fn try_fused_f16_sdpa_native(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    scale: f32,
) -> Result<Option<Tensor>> {
    if let Some(output) = try_cuda_f16_sdpa_native(q, k, v, causal, scale)? {
        return Ok(Some(output));
    }
    try_metal_f16_sdpa_native(q, k, v, causal, scale)
}

/// One-launch CUDA attention for a single query token over a native F16
/// cache. Unlike [`try_fused_f16_sdpa_native`] it keeps the query in F32 and
/// returns F32, so callers opt in. Returns `None` off CUDA or outside the
/// kernel's geometry.
pub(crate) fn try_cuda_decode_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
) -> Result<Option<Tensor>> {
    #[cfg(feature = "cuda")]
    {
        decode_sdpa_cuda::try_forward(q, k, v, scale)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q, k, v, scale);
        Ok(None)
    }
}

/// [`try_cuda_decode_sdpa`] with the output multiplied by `sigmoid(gate)`,
/// where `gate` is F32 `[batch, heads, 1, v_dim]`.
pub(crate) fn try_cuda_decode_sdpa_gated(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    gate: &Tensor,
    scale: f32,
) -> Result<Option<Tensor>> {
    #[cfg(feature = "cuda")]
    {
        decode_sdpa_cuda::try_forward_gated(q, k, v, Some(gate), scale)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q, k, v, gate, scale);
        Ok(None)
    }
}

/// Compile the single-token CUDA attention kernel ahead of the first request.
pub(crate) fn precompile_cuda_decode_sdpa(device: &candle_core::Device) -> Result<()> {
    #[cfg(feature = "cuda")]
    decode_sdpa_cuda::precompile(device)?;
    #[cfg(not(feature = "cuda"))]
    let _ = device;
    Ok(())
}

/// Use fused CUDA or Metal SDPA for causal attention limited to a left window.
pub fn try_fused_f16_sdpa_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    window: usize,
    scale: f32,
) -> Result<Option<Tensor>> {
    let q_dtype = q.dtype();
    let Some(output) = try_fused_f16_sdpa_windowed_native(q, k, v, window, scale)? else {
        return Ok(None);
    };
    if q_dtype == candle_core::DType::F16 {
        Ok(Some(output))
    } else {
        Ok(Some(output.to_dtype(q_dtype)?))
    }
}

/// Retain the native f16 output from causal, left-windowed fused attention.
pub fn try_fused_f16_sdpa_windowed_native(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    window: usize,
    scale: f32,
) -> Result<Option<Tensor>> {
    if let Some(output) = try_cuda_f16_sdpa_windowed_native(q, k, v, window, scale)? {
        return Ok(Some(output));
    }
    try_metal_f16_sdpa_windowed_native(q, k, v, window, scale)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FusedSdpaGeometry {
    batch: usize,
    q_heads: usize,
    kv_heads: usize,
    q_seq: usize,
    kv_seq: usize,
    qk_dim: usize,
    v_dim: usize,
}

fn fused_sdpa_geometry(q: [usize; 4], k: [usize; 4], v: [usize; 4]) -> Option<FusedSdpaGeometry> {
    let [batch, q_heads, q_seq, qk_dim] = q;
    let [k_batch, kv_heads, kv_seq, k_dim] = k;
    let [v_batch, v_heads, v_seq, v_dim] = v;
    if batch == 0
        || q_heads == 0
        || kv_heads == 0
        || q_seq == 0
        || kv_seq == 0
        || qk_dim == 0
        || v_dim == 0
        || batch != k_batch
        || batch != v_batch
        || q_heads % kv_heads != 0
        || kv_heads != v_heads
        || qk_dim != k_dim
        || kv_seq != v_seq
        || q_seq > kv_seq
    {
        return None;
    }
    Some(FusedSdpaGeometry {
        batch,
        q_heads,
        kv_heads,
        q_seq,
        kv_seq,
        qk_dim,
        v_dim,
    })
}

fn tensor_fused_sdpa_geometry(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<Option<FusedSdpaGeometry>> {
    let q = q.dims4()?;
    let k = k.dims4()?;
    let v = v.dims4()?;
    Ok(fused_sdpa_geometry(
        [q.0, q.1, q.2, q.3],
        [k.0, k.1, k.2, k.3],
        [v.0, v.1, v.2, v.3],
    ))
}

#[cfg(any(feature = "cuda", test))]
fn pad_cuda_flash_attention_values(
    v: &Tensor,
    value_dim: usize,
    query_key_dim: usize,
) -> Result<Tensor> {
    if value_dim == query_key_dim {
        return Ok(v.clone());
    }
    anyhow::ensure!(
        value_dim < query_key_dim,
        "CUDA FlashAttention value dimension exceeds query/key dimension"
    );
    let mut padded_dims = v.dims().to_vec();
    anyhow::ensure!(
        padded_dims.len() == 4 && padded_dims[3] == value_dim,
        "CUDA FlashAttention value tensor geometry changed during padding"
    );
    padded_dims[3] = query_key_dim;
    let padded = Tensor::zeros(padded_dims.as_slice(), v.dtype(), v.device())?;
    padded.slice_set(&v.contiguous()?, 3, 0)?;
    Ok(padded)
}

fn sliding_window_mask_values(q_seq: usize, kv_seq: usize, window: usize) -> Option<Vec<f32>> {
    if q_seq == 0 || q_seq > kv_seq || window == 0 {
        return None;
    }
    let query_offset = kv_seq - q_seq;
    Some(
        (0..q_seq)
            .flat_map(|query| {
                let query = query_offset + query;
                let first_key = query.saturating_sub(window - 1);
                (0..kv_seq).map(move |key| {
                    if (first_key..=query).contains(&key) {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect(),
    )
}

fn causal_mask_values(q_seq: usize, kv_seq: usize) -> Option<Vec<f32>> {
    if q_seq == 0 || q_seq > kv_seq {
        return None;
    }
    let query_offset = kv_seq - q_seq;
    Some(
        (0..q_seq)
            .flat_map(|query| {
                let last_key = query_offset + query;
                (0..kv_seq).map(move |key| {
                    if key <= last_key {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect(),
    )
}

#[cfg(any(feature = "cuda", test))]
fn cuda_flash_attention_compute_supported(compute_major: i32) -> bool {
    compute_major >= 8
}

/// Run FlashAttention v2 over heads-first tensors while retaining f16 output.
///
/// The public Candle kernel consumes `[batch, sequence, heads, head_dim]`, so
/// these transposes are stride-only views. Native GQA support means the f16 KV
/// cache remains at its original head count instead of being expanded to the
/// query-head count.
pub fn try_cuda_f16_sdpa_native(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    scale: f32,
) -> Result<Option<Tensor>> {
    try_cuda_f16_sdpa_native_inner(q, k, v, causal, None, scale)
}

/// Run FlashAttention v2 with an exact causal left window.
pub fn try_cuda_f16_sdpa_windowed_native(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    window: usize,
    scale: f32,
) -> Result<Option<Tensor>> {
    if window == 0 {
        return Ok(None);
    }
    try_cuda_f16_sdpa_native_inner(q, k, v, false, Some(window), scale)
}

fn try_cuda_f16_sdpa_native_inner(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    window: Option<usize>,
    scale: f32,
) -> Result<Option<Tensor>> {
    #[cfg(feature = "cuda")]
    {
        use candle_core::DType;

        if !q.device().is_cuda()
            || !q.device().same_device(k.device())
            || !q.device().same_device(v.device())
            || k.dtype() != DType::F16
            || v.dtype() != DType::F16
        {
            return Ok(None);
        }

        let candle_core::Device::Cuda(device) = q.device() else {
            return Ok(None);
        };
        let Ok((compute_major, _)) = device.cuda_stream().context().compute_capability() else {
            return Ok(None);
        };
        // Candle builds FlashAttention for the target CUDA architecture. Its
        // kernels require Ampere instructions, but newer CUDA generations can
        // run the same path when that target build succeeds.
        if !cuda_flash_attention_compute_supported(compute_major) {
            return Ok(None);
        }

        let Some(geometry) = tensor_fused_sdpa_geometry(q, k, v)? else {
            return Ok(None);
        };
        if geometry.qk_dim > 512
            || geometry.qk_dim % 8 != 0
            || geometry.v_dim > geometry.qk_dim
            || geometry.q_seq > u32::MAX as usize
            || geometry.kv_seq > u32::MAX as usize
        {
            return Ok(None);
        }

        let q = q.to_dtype(DType::F16)?.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v =
            pad_cuda_flash_attention_values(v, geometry.v_dim, geometry.qk_dim)?.transpose(1, 2)?;
        let output = if let Some(window) = window {
            candle_flash_attn::flash_attn_windowed(&q, &k, &v, scale, Some(window - 1), Some(0))?
        } else {
            candle_flash_attn::flash_attn(&q, &k, &v, scale, causal)?
        }
        .transpose(1, 2)?;
        let output = if geometry.v_dim == geometry.qk_dim {
            output
        } else {
            output.narrow(3, 0, geometry.v_dim)?
        };
        Ok(Some(output))
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q, k, v, causal, window, scale);
        Ok(None)
    }
}

/// Use Candle's fused Metal SDPA directly on native f16 KV-cache views.
///
/// Returns `None` when the device, dtype, or attention geometry is unsupported,
/// allowing the caller to retain its existing portable f32 implementation.
fn metal_sdpa_head_dim_supported(head_dim: usize) -> bool {
    matches!(head_dim, 32 | 64 | 72 | 80 | 96 | 128 | 256 | 512)
}

fn metal_sdpa_vector_head_dim_supported(head_dim: usize) -> bool {
    // The Rust dispatcher omits 72/80; their shader instantiations do not make
    // them valid vector kernels (the shader partitions dimensions by 32).
    matches!(head_dim, 32 | 64 | 96 | 128 | 256 | 512)
}

/// Run Candle's fused Metal scaled dot-product attention and restore the
/// query's original output dtype.
///
/// Returns `None` when Metal is unavailable or the tensors have unsupported
/// dtype/geometry, allowing callers to use a portable attention path.
pub fn try_metal_f16_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    scale: f32,
) -> Result<Option<Tensor>> {
    let q_dtype = q.dtype();
    let Some(output) = try_metal_f16_sdpa_native(q, k, v, causal, scale)? else {
        return Ok(None);
    };
    if q_dtype == candle_core::DType::F16 {
        Ok(Some(output))
    } else {
        Ok(Some(output.to_dtype(q_dtype)?))
    }
}

/// Run fused Metal SDPA while retaining its native f16 output.
pub fn try_metal_f16_sdpa_native(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    scale: f32,
) -> Result<Option<Tensor>> {
    try_metal_f16_sdpa_native_inner(q, k, v, causal, None, scale)
}

/// Run fused Metal SDPA with an exact causal left window.
pub fn try_metal_f16_sdpa_windowed_native(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    window: usize,
    scale: f32,
) -> Result<Option<Tensor>> {
    if window == 0 {
        return Ok(None);
    }
    try_metal_f16_sdpa_native_inner(q, k, v, false, Some(window), scale)
}

fn try_metal_f16_sdpa_native_inner(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    window: Option<usize>,
    scale: f32,
) -> Result<Option<Tensor>> {
    use candle_core::DType;

    if !q.device().is_metal()
        || !q.device().same_device(k.device())
        || !q.device().same_device(v.device())
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
    {
        return Ok(None);
    }
    let Some(mut geometry) = tensor_fused_sdpa_geometry(q, k, v)? else {
        return Ok(None);
    };
    if !metal_sdpa_head_dim_supported(geometry.qk_dim) || geometry.v_dim > geometry.qk_dim {
        return Ok(None);
    }
    if geometry.q_seq == 1 && !metal_sdpa_vector_head_dim_supported(geometry.qk_dim) {
        return Ok(None);
    }

    // No query in this chunk can see keys before its first query's window.
    // The single-query Metal kernel ignores masks, so trim its exact window.
    // Keep the stored cache intact; narrow only these read-only attention
    // views. For multiple queries, whole 32-key tiles preserve reduction groups
    // (Candle uses 8, 16, or 32 keys per tile depending on head width).
    let skip = window.map_or(0, |window| {
        metal_windowed_kv_skip(geometry.q_seq, geometry.kv_seq, window)
    });
    let (k, v) = if skip > 0 {
        geometry.kv_seq -= skip;
        (
            k.narrow(2, skip, geometry.kv_seq)?,
            v.narrow(2, skip, geometry.kv_seq)?,
        )
    } else {
        (k.clone(), v.clone())
    };
    let q = q.to_dtype(DType::F16)?;
    let v = if geometry.v_dim == geometry.qk_dim {
        v.clone()
    } else {
        // Candle's Metal kernel reports a distinct value width in its public
        // shape but still loads BD columns. Pad the narrow value head, then
        // remove those zero columns from the fused result.
        // Generic cat can leave a padded cache view noncontiguous. Vector
        // SDPA takes head strides but assumes contiguous key/value rows.
        v.pad_with_zeros(3, 0, geometry.qk_dim - geometry.v_dim)?
            .contiguous()?
    };
    // Metal 4 tensor kernels compute causal and window masks themselves.
    #[cfg(target_os = "macos")]
    if geometry.q_seq > 1
        && geometry.v_dim == geometry.qk_dim
        && (causal || window.is_some())
        && let Some(output) =
            flash_attention_metal::try_flash_attention(&q, &k, &v, window.unwrap_or(0), scale)?
    {
        return Ok(Some(output));
    }

    // The vendored kernel's causal path is bottom-right aligned (query offset
    // kv_seq - q_seq), as cached chunked prefill needs, and skips key blocks
    // past the diagonal. A left window that reaches the first key of the
    // (possibly trimmed) cache for every query is the same causal mask, so
    // only narrower windows need an explicit mask.
    let window_is_causal = window.is_some_and(|window| window >= geometry.kv_seq);
    let mask_values = match window {
        Some(window) if !window_is_causal => {
            sliding_window_mask_values(geometry.q_seq, geometry.kv_seq, window)
        }
        _ => None,
    };
    let mask = if let Some(values) = mask_values {
        Some(
            Tensor::from_vec(values, (1, 1, geometry.q_seq, geometry.kv_seq), q.device())?
                .to_dtype(DType::F16)?
                .broadcast_as((
                    geometry.batch,
                    geometry.q_heads,
                    geometry.q_seq,
                    geometry.kv_seq,
                ))?,
        )
    } else {
        None
    };
    let kernel_causal = (causal || window.is_some()) && mask.is_none();
    let output = candle_nn::ops::sdpa(&q, &k, &v, mask.as_ref(), kernel_causal, scale, 1.0)?;
    if geometry.v_dim == geometry.qk_dim {
        Ok(Some(output))
    } else {
        Ok(Some(output.narrow(3, 0, geometry.v_dim)?))
    }
}

fn metal_windowed_kv_skip(q_seq: usize, kv_seq: usize, window: usize) -> usize {
    if q_seq == 0 || kv_seq < q_seq || window == 0 {
        return 0;
    }
    if q_seq == 1 {
        return kv_seq.saturating_sub(window);
    }
    let needed = window.saturating_add(q_seq - 1);
    kv_seq.saturating_sub(needed) / 32 * 32
}

/// Use one length-aware fused dispatch over a packed ragged KV cache.
pub fn try_fused_f16_ragged_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    valid_lengths: &[usize],
    scale: f32,
) -> Result<Option<Tensor>> {
    if let Some(output) = try_cuda_f16_ragged_sdpa(q, k, v, valid_lengths, scale)? {
        return Ok(Some(output));
    }
    try_metal_f16_ragged_sdpa(q, k, v, valid_lengths, scale)
}

/// Map each dense ragged-cache lane to one paged FlashAttention KV block.
///
/// The cache is stored as `[batch, heads, capacity, head_dim]`. A transpose
/// gives the paged kernel `[blocks, capacity, heads, head_dim]` with no data
/// copy; its explicit strides retain the original head-major layout.
pub fn try_cuda_f16_ragged_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    valid_lengths: &[usize],
    scale: f32,
) -> Result<Option<Tensor>> {
    #[cfg(feature = "cuda")]
    {
        use candle_core::DType;

        if !q.device().is_cuda()
            || !q.device().same_device(k.device())
            || !q.device().same_device(v.device())
            || k.dtype() != DType::F16
            || v.dtype() != DType::F16
        {
            return Ok(None);
        }
        let candle_core::Device::Cuda(device) = q.device() else {
            return Ok(None);
        };
        let Ok((compute_major, _)) = device.cuda_stream().context().compute_capability() else {
            return Ok(None);
        };
        if !cuda_flash_attention_compute_supported(compute_major) {
            return Ok(None);
        }

        let (batch, q_heads, q_seq, q_dim) = q.dims4()?;
        let (k_batch, kv_heads, capacity, k_dim) = k.dims4()?;
        let (v_batch, v_heads, v_capacity, v_dim) = v.dims4()?;
        if batch == 0
            || q_heads == 0
            || kv_heads == 0
            || q_seq != 1
            || batch != k_batch
            || batch != v_batch
            || valid_lengths.len() != batch
            || valid_lengths
                .iter()
                .any(|length| *length == 0 || *length > capacity)
            || q_heads % kv_heads != 0
            || kv_heads != v_heads
            || q_dim != k_dim
            || q_dim != v_dim
            || capacity != v_capacity
            || capacity % 32 != 0
            || q_dim == 0
            || q_dim > 512
            || q_dim % 8 != 0
            || batch > u32::MAX as usize
        {
            return Ok(None);
        }

        let mut cumulative_k = Vec::with_capacity(batch + 1);
        cumulative_k.push(0u32);
        let mut total_k = 0u32;
        for length in valid_lengths {
            let Ok(length) = u32::try_from(*length) else {
                return Ok(None);
            };
            let Some(next) = total_k.checked_add(length) else {
                return Ok(None);
            };
            total_k = next;
            cumulative_k.push(total_k);
        }
        let cumulative_q = (0..=batch as u32).collect::<Vec<_>>();
        let block_table = (0..batch as u32).collect::<Vec<_>>();
        let seqlens_q = Tensor::from_vec(cumulative_q, batch + 1, q.device())?;
        let seqlens_k = Tensor::from_vec(cumulative_k, batch + 1, q.device())?;
        let block_table = Tensor::from_vec(block_table, (batch, 1), q.device())?;
        let q_dtype = q.dtype();
        let q = q.to_dtype(DType::F16)?.squeeze(2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let output = candle_flash_attn::flash_attn_varlen_paged_windowed(
            &q,
            &k,
            &v,
            &seqlens_q,
            &seqlens_k,
            &block_table,
            None,
            1,
            valid_lengths.iter().copied().max().unwrap_or(1),
            scale,
            None,
            None,
            capacity,
            None,
        )?
        .unsqueeze(2)?;
        if q_dtype == DType::F16 {
            Ok(Some(output))
        } else {
            Ok(Some(output.to_dtype(q_dtype)?))
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q, k, v, valid_lengths, scale);
        Ok(None)
    }
}

#[cfg(any(target_os = "macos", test))]
fn metal_ragged_sdpa_launch_supported(simd_width: usize, max_threads: usize) -> bool {
    simd_width == 32 && max_threads >= 1024
}

/// Use a length-aware fused Metal decode kernel over a packed ragged KV cache.
///
/// `k` and `v` contain the common allocation capacity for every lane while
/// `valid_lengths` prevents unwritten slack from participating in attention.
/// Returns `None` off Metal or for unsupported tensor geometry or launch limits.
pub fn try_metal_f16_ragged_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    valid_lengths: &[usize],
    scale: f32,
) -> Result<Option<Tensor>> {
    use candle_core::DType;

    if !q.device().is_metal() || k.dtype() != DType::F16 || v.dtype() != DType::F16 {
        return Ok(None);
    }
    let (q_batch, q_heads, q_seq, q_dim) = q.dims4()?;
    let (k_batch, k_heads, k_capacity, k_dim) = k.dims4()?;
    let (v_batch, v_heads, v_capacity, v_dim) = v.dims4()?;
    if q_batch == 0
        || q_heads == 0
        || k_heads == 0
        || q_seq != 1
        || q_batch != k_batch
        || q_batch != v_batch
        || valid_lengths.len() != q_batch
        || valid_lengths
            .iter()
            .any(|length| *length == 0 || *length > k_capacity || *length > u32::MAX as usize)
        || q_heads % k_heads != 0
        || k_heads != v_heads
        || q_dim != k_dim
        || q_dim != v_dim
        || k_capacity != v_capacity
        || q_dim == 0
        || q_dim > 512
    {
        return Ok(None);
    }

    #[cfg(target_os = "macos")]
    {
        if !ragged_metal::sdpa_supported(q.device().as_metal_device()?, q_dim)? {
            return Ok(None);
        }
        let q_dtype = q.dtype();
        let q = q.to_dtype(DType::F16)?.contiguous()?;
        let output = ragged_metal::f16_ragged_sdpa(&q, k, v, valid_lengths, scale)?;
        if q_dtype == DType::F16 {
            Ok(Some(output))
        } else {
            Ok(Some(output.to_dtype(q_dtype)?))
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = scale;
        Ok(None)
    }
}

/// Apply Q and K RoPE for a decode batch whose lanes have different positions.
///
/// Metal handles both tensors in one dispatch and returns `None` elsewhere so
/// callers can retain the portable lane-wise implementation.
pub fn try_metal_ragged_rope(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &[usize],
    neox: bool,
) -> Result<Option<(Tensor, Tensor)>> {
    use candle_core::DType;

    if !q.device().is_metal()
        || q.dtype() != k.dtype()
        || !matches!(q.dtype(), DType::F16 | DType::F32)
        || cos.dtype() != DType::F32
        || sin.dtype() != DType::F32
    {
        return Ok(None);
    }
    let (batch, q_heads, q_seq, head_dim) = q.dims4()?;
    let (k_batch, k_heads, k_seq, k_dim) = k.dims4()?;
    let (cos_positions, cos_dim) = cos.dims2()?;
    let (sin_positions, sin_dim) = sin.dims2()?;
    if batch == 0
        || q_heads == 0
        || k_heads == 0
        || batch != k_batch
        || q_seq != 1
        || k_seq != 1
        || head_dim == 0
        || head_dim % 2 != 0
        || head_dim != k_dim
        || cos_dim == 0
        || cos_dim * 2 > head_dim
        || sin_dim != cos_dim
        || positions.len() != batch
        || positions.iter().any(|position| {
            *position >= cos_positions
                || *position >= sin_positions
                || *position > u32::MAX as usize
        })
    {
        return Ok(None);
    }

    #[cfg(target_os = "macos")]
    {
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        Ok(Some(ragged_metal::ragged_rope(
            &q, &k, cos, sin, positions, neox,
        )?))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = neox;
        Ok(None)
    }
}

/// Helper: temporarily extract KvCache from a LayerCache slot, run a closure, and put it back.
/// If the slot is None or Attention, works correctly.
///
/// # Panics
///
/// Panics if the slot holds any other cache kind. The check runs before the
/// slot is touched, so the caller's cache is left intact.
pub fn with_kv_cache<F, R>(slot: &mut Option<LayerCache>, f: F) -> R
where
    F: FnOnce(&mut Option<KvCache>) -> R,
{
    assert!(
        matches!(slot, None | Some(LayerCache::Attention(_))),
        "expected an attention cache or an empty slot"
    );
    // Extract existing KvCache if present
    let mut kv = match slot.take() {
        Some(LayerCache::Attention(kv)) => Some(kv),
        _ => None,
    };
    let result = f(&mut kv);
    // Put it back
    *slot = kv.map(LayerCache::Attention);
    result
}

/// Create a model architecture from loaded stage tensors.
///
/// Dispatches based on the `general.architecture` GGUF metadata key.
/// `quant_config` enables TurboQuant KV cache compression when provided.
pub fn from_stage_tensors(
    stage: StageTensors,
    quant_config: Option<Arc<TurboQuantConfig>>,
) -> Result<Box<dyn ModelArchitecture>> {
    match stage.metadata.architecture.as_str() {
        "llama" | "mistral" | "qwen2" => {
            let model = llama::LlamaStage::from_stage_tensors(stage, quant_config)?;
            Ok(Box::new(model))
        }
        "qwen3moe" => {
            let model = qwen3moe::Qwen3MoeStage::from_stage_tensors(stage, quant_config)?;
            Ok(Box::new(model))
        }
        "qwen35" => {
            let model = qwen35::Qwen35Stage::from_stage_tensors(stage, quant_config)?;
            Ok(Box::new(model))
        }
        "phi3" => {
            let model = phi3::Phi3Stage::from_stage_tensors(stage, quant_config)?;
            Ok(Box::new(model))
        }
        "nemotron_h" | "nemotron_h_moe" => {
            let model = nemotron_h::NemotronStage::from_stage_tensors(stage, quant_config)?;
            Ok(Box::new(model))
        }
        "deepseek2" | "glm-dsa" => {
            let model = deepseek2::DeepSeek2Stage::from_stage_tensors(stage, quant_config)?;
            Ok(Box::new(model))
        }
        "deepseek4" => {
            let model = deepseek4::DeepSeek4Stage::from_stage_tensors(stage, quant_config)?;
            Ok(Box::new(model))
        }
        "deepseek41" => {
            let model = deepseek41::DeepSeek41Stage::from_stage_tensors(stage, quant_config)?;
            Ok(Box::new(model))
        }
        "glm5next" | "glm5-next" => {
            let model = glm5_next::Glm5NextStage::from_stage_tensors(stage, quant_config)?;
            Ok(Box::new(model))
        }
        "llama4" => {
            let model = llama4::Llama4Stage::from_stage_tensors(stage, quant_config)?;
            Ok(Box::new(model))
        }
        "gemma4" => {
            let model = gemma4::Gemma4Stage::from_stage_tensors(stage, quant_config)?;
            Ok(Box::new(model))
        }
        "minimax-m2" => {
            let model = minimax_m2::MiniMaxM2Stage::from_stage_tensors(stage, quant_config)?;
            Ok(Box::new(model))
        }
        "muse-spark" => {
            let model = muse_spark::MuseSparkStage::from_stage_tensors(stage, quant_config)?;
            Ok(Box::new(model))
        }
        "inkling2_loop" => {
            let model = inkling2_loop::Inkling2LoopStage::from_stage_tensors(stage, quant_config)?;
            Ok(Box::new(model))
        }
        arch => anyhow::bail!(
            "Unsupported model architecture: '{arch}'. \
             Recognized: llama, mistral, qwen2, phi3, qwen3moe, qwen35, nemotron_h_moe, deepseek2, deepseek4, glm-dsa, glm5next, llama4, gemma4, minimax-m2, muse-spark, inkling2_loop \
             (glm-dsa, muse-spark and inkling2_loop are unvalidated)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;
    use candle_core::{DType, Device, IndexOp};

    #[test]
    fn metal_vector_widths_and_window_trim_length() {
        for width in [32, 64, 96, 128, 256, 512] {
            assert!(metal_sdpa_vector_head_dim_supported(width));
        }
        for width in [72, 80] {
            assert!(!metal_sdpa_vector_head_dim_supported(width));
        }
        for window in [1023, 1024, 1025] {
            let length = 2051 - metal_windowed_kv_skip(1, 2051, window);
            assert_eq!(length, window);
        }
    }

    #[cfg(target_os = "macos")]
    fn sdpa_precision_test_device() -> Result<Option<Device>> {
        match Device::new_metal(0) {
            Ok(device) => Ok(Some(device)),
            Err(error)
                if std::env::var_os("DINF_REQUIRE_METAL_TESTS").is_none()
                    && std::env::var_os("DINF_REQUIRE_METAL").is_none() =>
            {
                eprintln!("Metal unavailable: {error}");
                Ok(None)
            }
            Err(error) => Err(error.into()),
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_native_sdpa_declines_unimplemented_vector_widths() -> Result<()> {
        let Some(device) = sdpa_precision_test_device()? else {
            return Ok(());
        };
        for width in [72, 80] {
            for keys in [1023, 1024] {
                let q = Tensor::zeros((1, 4, 1, width), DType::F32, &device)?;
                let k = Tensor::zeros((1, 2, keys, width), DType::F16, &device)?;
                assert!(try_metal_f16_sdpa_native(&q, &k, &k, false, 0.125)?.is_none());
            }
        }
        device.synchronize()?;
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_native_sdpa_keeps_large_two_pass_partials_in_f32() -> Result<()> {
        let Some(device) = sdpa_precision_test_device()? else {
            return Ok(());
        };
        for (head_dim, value_dim) in [(128, 128), (512, 128), (512, 512)] {
            let q = Tensor::zeros((1, 4, 1, head_dim), DType::F32, &device)?;
            for keys in [1023, 1024, 1025] {
                let k = Tensor::zeros((1, 2, keys, head_dim), DType::F16, &device)?;
                let v = Tensor::full(f16::from_f32(4096.0), (1, 2, keys, value_dim), &device)?;
                let output = try_metal_f16_sdpa_native(&q, &k, &v, false, 0.125)?
                    .expect("supported native vector attention");
                assert_eq!(output.dtype(), DType::F16);
                assert_eq!(output.dims(), &[1, 4, 1, value_dim]);
                assert!(
                    output
                        .flatten_all()?
                        .to_vec1::<f16>()?
                        .iter()
                        .all(|value| value.to_bits() == f16::from_f32(4096.0).to_bits()),
                    "normalized constant V must remain 4096 at D={head_dim}, KV={keys}"
                );
            }
        }
        // F32 full 512 is absent: a multi-query call must stay on the old F16
        // full kernel even when its visible cache crosses the vector threshold.
        let q = Tensor::zeros((1, 2, 2, 512), DType::F16, &device)?;
        let k = Tensor::zeros((1, 1, 1024, 512), DType::F16, &device)?;
        let v = Tensor::full(f16::from_f32(4096.0), (1, 1, 1024, 512), &device)?;
        let output = try_metal_f16_sdpa_native(&q, &k, &v, true, 0.125)?
            .expect("existing F16 full 512 attention");
        assert_eq!(output.dtype(), DType::F16);
        assert!(
            output
                .flatten_all()?
                .to_vec1::<f16>()?
                .iter()
                .all(|value| value.to_f32().is_finite() && (value.to_f32() - 4096.0).abs() <= 2.0)
        );
        device.synchronize()?;
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn flash_rows_do_not_depend_on_prompt_chunking() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        // Gemma 4 geometry: sliding layers 32/16 heads at 256, global 32/4 at
        // 512. A 26-token prompt in one pass, then in chunks of 8, 8, 8, 2.
        for (dim, heads, kv_heads) in [(256usize, 32usize, 16usize), (512, 32, 4)] {
            let prompt = 26usize;
            let value = |i: usize, salt: usize| ((i * 31 + salt) % 97) as f32 / 97.0 - 0.5;
            let tensor = |h: usize, salt: usize| -> Result<Tensor> {
                Ok(Tensor::from_vec(
                    (0..h * prompt * dim)
                        .map(|i| value(i, salt))
                        .collect::<Vec<_>>(),
                    (1, h, prompt, dim),
                    &device,
                )?
                .to_dtype(DType::F16)?)
            };
            let (q, k, v) = (
                tensor(heads, 3)?,
                tensor(kv_heads, 7)?,
                tensor(kv_heads, 11)?,
            );
            for window in [None, Some(1024)] {
                let run = |q: &Tensor, keys: usize| -> Result<Tensor> {
                    let (k, v) = (k.narrow(2, 0, keys)?, v.narrow(2, 0, keys)?);
                    Ok(match window {
                        None => try_metal_f16_sdpa_native(q, &k, &v, true, 1.0)?,
                        Some(w) => try_metal_f16_sdpa_windowed_native(q, &k, &v, w, 1.0)?,
                    }
                    .expect("supported"))
                };
                let all = run(&q, prompt)?.flatten_all()?.to_vec1::<f16>()?;
                let mut start = 0;
                for len in [8usize, 8, 8, 2] {
                    let chunk = run(&q.narrow(2, start, len)?.contiguous()?, start + len)?;
                    for head in 0..heads {
                        let got = chunk.narrow(1, head, 1)?.flatten_all()?.to_vec1::<f16>()?;
                        let base = (head * prompt + start) * dim;
                        let want = &all[base..base + len * dim];
                        let differing = got.iter().zip(want).filter(|(a, b)| a != b).count();
                        assert_eq!(
                            differing, 0,
                            "d{dim} window {window:?} chunk at {start} head {head}: {differing} values differ"
                        );
                    }
                    start += len;
                }
            }
        }
        Ok(())
    }

    /// Chunked prefill (fewer queries than keys) through the kernel's causal
    /// path, plain and with a left window that covers every key, matches a
    /// bottom-right-aligned causal reference; a narrower window still masks.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_native_sdpa_kernel_causal_matches_bottom_right_reference() -> Result<()> {
        for dim in [256usize, 512] {
            metal_native_sdpa_kernel_causal_case(dim)?;
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn metal_native_sdpa_kernel_causal_case(dim: usize) -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (heads, kv_heads, queries, keys) = (4usize, 2usize, 37usize, 90usize);
        let value = |index: usize, salt: usize| ((index * 31 + salt) % 97) as f32 / 97.0 - 0.5;
        let q_values = (0..heads * queries * dim)
            .map(|i| value(i, 3))
            .collect::<Vec<_>>();
        let k_values = (0..kv_heads * keys * dim)
            .map(|i| f16::from_f32(value(i, 7)).to_f32())
            .collect::<Vec<_>>();
        let v_values = (0..kv_heads * keys * dim)
            .map(|i| f16::from_f32(value(i, 11)).to_f32())
            .collect::<Vec<_>>();
        let q = Tensor::from_vec(q_values.clone(), (1, heads, queries, dim), &device)?;
        let k = Tensor::from_vec(k_values.clone(), (1, kv_heads, keys, dim), &device)?
            .to_dtype(DType::F16)?;
        let v = Tensor::from_vec(v_values.clone(), (1, kv_heads, keys, dim), &device)?
            .to_dtype(DType::F16)?;
        let scale = 0.0625f32;
        let reference = |window: Option<usize>| {
            let mut out = vec![0f32; heads * queries * dim];
            for head in 0..heads {
                let kv = head / (heads / kv_heads);
                for query in 0..queries {
                    let position = keys - queries + query;
                    let first = window.map_or(0, |w| (position + 1).saturating_sub(w));
                    let scores = (first..=position)
                        .map(|key| {
                            (0..dim)
                                .map(|d| {
                                    q_values[(head * queries + query) * dim + d]
                                        * k_values[(kv * keys + key) * dim + d]
                                })
                                .sum::<f32>()
                                * scale
                        })
                        .collect::<Vec<_>>();
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let weights = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
                    let total = weights.iter().sum::<f32>();
                    for d in 0..dim {
                        out[(head * queries + query) * dim + d] = (first..=position)
                            .zip(&weights)
                            .map(|(key, w)| w * v_values[(kv * keys + key) * dim + d])
                            .sum::<f32>()
                            / total;
                    }
                }
            }
            out
        };
        let check = |actual: Tensor, expected: Vec<f32>, label: &str| -> Result<()> {
            let actual = actual
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
                assert!((a - e).abs() < 5e-3, "d{dim} {label}[{index}]: {a} vs {e}");
            }
            Ok(())
        };
        check(
            try_metal_f16_sdpa_native(&q, &k, &v, true, scale)?.expect("causal"),
            reference(None),
            "causal",
        )?;
        check(
            try_metal_f16_sdpa_windowed_native(&q, &k, &v, keys, scale)?.expect("wide window"),
            reference(None),
            "wide window",
        )?;
        check(
            try_metal_f16_sdpa_windowed_native(&q, &k, &v, 20, scale)?.expect("narrow window"),
            reference(Some(20)),
            "narrow window",
        )?;
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_native_sdpa_window_trim_preserves_f64_mean_at_two_pass_boundary() -> Result<()> {
        let Some(device) = sdpa_precision_test_device()? else {
            return Ok(());
        };
        let (batch, heads, kv_heads, keys, capacity, base, value_dim) =
            (2, 4, 2, 2051, 2056, 3, 128);
        for window in [1023, 1024, 1025] {
            let mut data = vec![f16::from_f32(-4096.0); batch * kv_heads * capacity * value_dim];
            for lane in 0..batch {
                for head in 0..kv_heads {
                    for key in keys - window..keys {
                        for col in 0..value_dim {
                            let value = 4096.0
                                + ((key % 5) * 4 + head * 32 + lane * 16 + (col % 3) * 4) as f32;
                            data[((lane * kv_heads + head) * capacity + base + key) * value_dim
                                + col] = f16::from_f32(value);
                        }
                    }
                }
            }
            let q = Tensor::zeros((batch, heads, 1, 512), DType::F32, &device)?;
            let k = Tensor::zeros((batch, kv_heads, capacity, 512), DType::F16, &device)?
                .narrow(2, base, keys)?;
            let v = Tensor::from_vec(
                data.clone(),
                (batch, kv_heads, capacity, value_dim),
                &device,
            )?
            .narrow(2, base, keys)?;
            let output = try_metal_f16_sdpa_windowed_native(&q, &k, &v, window, 0.125)?
                .expect("supported narrow-value window");
            assert_eq!(output.dtype(), DType::F16);
            assert_eq!(output.dims(), &[batch, heads, 1, value_dim]);
            let actual = output.flatten_all()?.to_vec1::<f16>()?;
            for lane in 0..batch {
                for head in 0..heads {
                    let kv_head = head / (heads / kv_heads);
                    for col in 0..value_dim {
                        let mean = (keys - window..keys)
                            .map(|key| {
                                data[((lane * kv_heads + kv_head) * capacity + base + key)
                                    * value_dim
                                    + col]
                                    .to_f64()
                            })
                            .sum::<f64>()
                            / window as f64;
                        let expected = f16::from_f64(mean);
                        let actual = actual[(lane * heads + head) * value_dim + col];
                        assert!(actual.is_finite());
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "F64 mean differs at window={window}, lane={lane}, head={head}, col={col}"
                        );
                    }
                }
            }
        }
        device.synchronize()?;
        Ok(())
    }

    #[test]
    fn ragged_metal_attention_requires_complete_32_wide_simd_groups() {
        assert!(metal_ragged_sdpa_launch_supported(32, 1024));
        assert!(metal_ragged_sdpa_launch_supported(32, 2048));
        for (width, threads) in [(16, 1024), (64, 1024), (32, 512), (32, 1023)] {
            assert!(!metal_ragged_sdpa_launch_supported(width, threads));
        }
    }

    fn constant_kv(fill: f32, seq_len: usize) -> (Tensor, Tensor) {
        let k = Tensor::full(fill, (1, 2, seq_len, 4), &Device::Cpu).unwrap();
        (k.clone(), k)
    }

    #[test]
    fn kv_capacity_growth_is_checked_and_keeps_existing_alignment() -> Result<()> {
        for (current, required, expected) in [
            (0, 1, 64),
            (0, 64, 64),
            (0, 65, 128),
            (64, 65, 128),
            (128, 129, 192),
            (128, 300, 320),
        ] {
            assert_eq!(
                PreallocatedKvCache::next_capacity(current, required)?,
                expected
            );
        }
        assert!(PreallocatedKvCache::next_capacity(0, usize::MAX).is_err());
        assert!(PreallocatedKvCache::next_capacity(usize::MAX, usize::MAX).is_err());
        let largest_aligned = usize::MAX - 63;
        assert_eq!(
            PreallocatedKvCache::next_capacity(0, largest_aligned)?,
            largest_aligned
        );
        Ok(())
    }

    #[test]
    fn compact_k_quant_dequantization_matches_candle_f16() -> Result<()> {
        let device = Device::Cpu;
        let values = (0..512)
            .map(|index| ((index as f32 * 0.03125).sin() * 4.0) - 0.5)
            .collect::<Vec<_>>();
        let source = Tensor::from_vec(values, (2, 256), &device)?;

        for dtype in [GgmlDType::Q4K, GgmlDType::Q6K] {
            let quantized = QTensor::quantize(&source, dtype)?;
            let expected = quantized
                .dequantize_f16(&device)?
                .flatten_all()?
                .to_vec1::<f16>()?;
            let actual = dequantize_f16_compact(&quantized, &device)?
                .flatten_all()?
                .to_vec1::<f16>()?;
            assert_eq!(actual, expected, "compact {dtype:?} conversion drifted");
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_decode_safe_projection_matches_cpu_at_minimax_shape() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (rows, width) = (1024, 3072);
        let weights = Tensor::from_vec(
            (0..rows * width)
                .map(|i| ((i * 13 % 257) as f32 - 128.) / 1024.)
                .collect::<Vec<_>>(),
            (rows, width),
            &Device::Cpu,
        )?;
        let cpu_q = QTensor::quantize(&weights, GgmlDType::Q4K)?;
        let expected_weights = cpu_q.dequantize(&Device::Cpu)?.to_dtype(DType::F16)?;
        let q = Arc::new(QTensor::quantize_onto(&weights, GgmlDType::Q4K, &device)?);
        let op = DecodeSafeQMatMul::from_qtensor(q, &device)?.with_matrix_decode();
        let Some(QMatMul::TensorF16(actual_weights)) = &op.single_token else {
            panic!("missing fallback")
        };
        assert_eq!(
            actual_weights.flatten_all()?.to_vec1::<f16>()?,
            expected_weights.flatten_all()?.to_vec1::<f16>()?
        );
        for batch in [1, 2] {
            for seq_len in [1, 4] {
                let input = Tensor::from_vec(
                    (0..batch * seq_len * width)
                        .map(|i| ((i * 11 % 127) as f32 - 63.) / 128.)
                        .collect::<Vec<_>>(),
                    (batch, seq_len, width),
                    &Device::Cpu,
                )?;
                let reference_weights = if seq_len == 1 {
                    expected_weights.to_dtype(DType::F32)?
                } else {
                    cpu_q.dequantize(&Device::Cpu)?
                };
                let expected = QMatMul::Tensor(reference_weights)
                    .forward(&input)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let mut input_gpu = input.to_device(&device)?;
                if seq_len == 1 {
                    // Decode views can start inside a larger backing allocation.
                    let prefix = Tensor::zeros((1, 1, width), DType::F32, &device)?;
                    input_gpu = Tensor::cat(&[prefix, input_gpu], 0)?.narrow(0, 1, batch)?;
                }
                let poison = Tensor::full(f16::NAN, (batch, 2, rows), &device)?;
                device.synchronize()?;
                drop(poison);
                let output = op.forward(&input_gpu)?;
                assert_eq!(output.dims(), &[batch, seq_len, rows]);
                let actual = output.flatten_all()?.to_vec1::<f32>()?;
                if seq_len == 1 {
                    // Output heads flatten `[lanes, 1, hidden]` to two dims.
                    // They must still use the decode weight precision.
                    let head = op.forward_decode(&input_gpu.reshape((batch, width))?)?;
                    assert_eq!(head.dims(), &[batch, rows]);
                    assert_eq!(actual, head.flatten_all()?.to_vec1::<f32>()?);
                }
                assert!(
                    actual.iter().all(|v| v.is_finite()),
                    "non-finite decode-safe output"
                );
                let max_error = actual
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0_f32, f32::max);
                assert!(
                    max_error < 0.01,
                    "batch {batch}, rows {seq_len}: max error {max_error}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn native_kv_update_keeps_dense_views_f16() -> Result<()> {
        let (k, v) = constant_kv(1.0, 3);
        let mut cache = None;
        let (k, v) = kv_cache_update_native(&mut cache, k, v, None)?;
        assert_eq!(k.dtype(), DType::F16);
        assert_eq!(v.dtype(), DType::F16);
        assert_eq!(k.dims(), &[1, 2, 3, 4]);
        assert!(!k.is_contiguous());
        assert!(matches!(cache, Some(KvCache::PreallocatedF16(_))));
        assert!(try_fused_f16_sdpa(&k, &k, &v, false, 1.0)?.is_none());
        assert!(try_metal_f16_sdpa(&k, &k, &v, false, 1.0)?.is_none());
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_quantized_attention_rotation_threshold_and_eviction() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (dim, kv_heads, q_heads) = (64usize, 2usize, 4usize);
        let config = TurboQuantConfig::new(8, dim, 711);

        let values = |start: usize, tokens: usize, heads: usize| {
            (0..heads * tokens * dim)
                .map(|index| {
                    let absolute = start * heads * dim + index;
                    (((absolute * 17 + 11) % 257) as f32 / 257.0 - 0.5) * 0.5
                })
                .collect::<Vec<_>>()
        };
        let k_values = values(0, 255, kv_heads);
        let k_last = values(255, 1, kv_heads);
        let v_values = values(3, 255, kv_heads);
        let v_last = values(258, 1, kv_heads);
        let q_values = values(7, 1, q_heads);

        let k_prefill = Tensor::from_vec(k_values.clone(), (1, kv_heads, 255, dim), &device)?;
        let v_prefill = Tensor::from_vec(v_values.clone(), (1, kv_heads, 255, dim), &device)?;
        let q_prefill = Tensor::zeros((1, q_heads, 255, dim), DType::F16, &device)?;
        let mut cache = None;
        let cache_seq_len = |cache: &Option<KvCache>| match cache.as_ref() {
            Some(KvCache::Quantized(cache)) => cache.seq_len(),
            _ => 0,
        };
        let (_, prepared_k, prepared_v, prefill_rotation) = prepare_quantized_attention(
            &mut cache,
            q_prefill,
            k_prefill,
            v_prefill,
            Some(&config),
        )?;
        assert!(
            prefill_rotation.is_none(),
            "prefill must not rotate attention"
        );
        assert_eq!(prepared_k.dims(), &[1, kv_heads, 255, dim]);
        assert_eq!(prepared_v.dims(), &[1, kv_heads, 255, dim]);
        assert_eq!(cache_seq_len(&cache), 255);

        let k_decode = Tensor::from_vec(k_last.clone(), (1, kv_heads, 1, dim), &device)?;
        let v_decode = Tensor::from_vec(v_last.clone(), (1, kv_heads, 1, dim), &device)?;
        let q_decode = Tensor::from_vec(q_values, (1, q_heads, 1, dim), &device)?;
        let (rotated_q, rotated_k, rotated_v, rotation) =
            prepare_quantized_attention(&mut cache, q_decode, k_decode, v_decode, Some(&config))?;
        let rotation = rotation.context("256-token decode must use rotated attention")?;
        assert_eq!(rotated_q.dtype(), DType::F16);
        assert_eq!(rotated_q.dims(), &[1, q_heads, 1, dim]);
        assert_eq!(rotated_k.dims(), &[1, kv_heads, 256, dim]);
        assert_eq!(rotated_v.dims(), &[1, kv_heads, 256, dim]);
        assert_eq!(
            cache.as_ref().and_then(|cache| match cache {
                KvCache::Quantized(cache) => Some(cache.seq_len()),
                _ => None,
            }),
            Some(256)
        );

        // The prepared K is in the rotated attention basis. Undoing that
        // rotation must recover the same quantized values as the CPU cache
        // built with the cloned, non-default seed.
        let unrotated_k = crate::turboquant_metal::rotate_attention(&rotated_k, &rotation, true)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut cpu_cache = QuantizedKvCache::new(Arc::new(config.clone()), kv_heads);
        let cpu_k_prefill = Tensor::from_vec(k_values, (1, kv_heads, 255, dim), &Device::Cpu)?;
        let cpu_k_last = Tensor::from_vec(k_last, (1, kv_heads, 1, dim), &Device::Cpu)?;
        let cpu_k = Tensor::cat(&[&cpu_k_prefill, &cpu_k_last], 2)?;
        let cpu_v_prefill = Tensor::from_vec(v_values, (1, kv_heads, 255, dim), &Device::Cpu)?;
        let cpu_v_last = Tensor::from_vec(v_last, (1, kv_heads, 1, dim), &Device::Cpu)?;
        let cpu_v = Tensor::cat(&[&cpu_v_prefill, &cpu_v_last], 2)?;
        let (reference_k, _) = cpu_cache.update(&cpu_k, &cpu_v)?;
        let reference_k = reference_k.flatten_all()?.to_vec1::<f32>()?;
        let max_diff = unrotated_k
            .iter()
            .zip(&reference_k)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff <= 0.002,
            "inverse-rotated Metal K differs from CPU reference by {max_diff}"
        );

        let scale = 1.0 / (dim as f32).sqrt();
        let attention =
            try_fused_f16_sdpa_native(&rotated_q, &rotated_k, &rotated_v, false, scale)?
                .context("expected native Metal SDPA for rotated attention")?;
        let restored = restore_quantized_attention_output(attention, Some(rotation))?;
        let restored_values = restored
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(
            restored_values.iter().all(|value| value.is_finite()),
            "restored attention output contains non-finite values"
        );

        let (evict_q, evict_k, evict_v) = (
            Tensor::zeros((1, q_heads, 1, dim), DType::F16, &device)?,
            Tensor::from_vec(values(259, 1, kv_heads), (1, kv_heads, 1, dim), &device)?,
            Tensor::from_vec(values(262, 1, kv_heads), (1, kv_heads, 1, dim), &device)?,
        );
        {
            let Some(KvCache::Quantized(cache)) = cache.as_mut() else {
                anyhow::bail!("expected quantized cache after rotated decode");
            };
            cache.evict_oldest(2);
            assert_eq!(cache.seq_len(), 254);
        }
        let (_, _, _, rotation_255) =
            prepare_quantized_attention(&mut cache, evict_q, evict_k, evict_v, Some(&config))?;
        assert!(
            rotation_255.is_none(),
            "254-token cache plus one token must stay on the ordinary path"
        );
        assert_eq!(cache_seq_len(&cache), 255);
        let (_, _, _, rotation_256) = prepare_quantized_attention(
            &mut cache,
            Tensor::zeros((1, q_heads, 1, dim), DType::F16, &device)?,
            Tensor::from_vec(values(263, 1, kv_heads), (1, kv_heads, 1, dim), &device)?,
            Tensor::from_vec(values(266, 1, kv_heads), (1, kv_heads, 1, dim), &device)?,
            Some(&config),
        )?;
        assert!(
            rotation_256.is_some(),
            "255-token cache plus one token must use rotated attention"
        );
        assert_eq!(cache_seq_len(&cache), 256);

        // BF16 queries are valid inputs to ordinary SDPA but not to the
        // rotation kernel. Keep them on that supported path even after the
        // long-cache threshold; do not append then fail during Q rotation.
        let (q, k, v, rotation) = prepare_quantized_attention(
            &mut cache,
            Tensor::ones((1, q_heads, 1, dim), DType::BF16, &device)?,
            Tensor::ones((1, kv_heads, 1, dim), DType::BF16, &device)?,
            Tensor::ones((1, kv_heads, 1, dim), DType::BF16, &device)?,
            Some(&config),
        )?;
        assert!(rotation.is_none());
        assert_eq!(cache_seq_len(&cache), 257);
        assert!(try_fused_f16_sdpa_native(&q, &k, &v, false, scale)?.is_some());
        Ok(())
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_quantized_attention_preserves_f32_for_portable_geometry() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for dim in [8, 16, 1024] {
            let config = TurboQuantConfig::new(8, dim, 711);
            let mut cache = None;
            let q = Tensor::ones((1, 2, 1, dim), DType::F32, &device)?;
            let (q, k, v, rotation) = prepare_quantized_attention(
                &mut cache,
                q,
                Tensor::ones((1, 1, 256, dim), DType::F32, &device)?,
                Tensor::ones((1, 1, 256, dim), DType::F32, &device)?,
                Some(&config),
            )?;
            assert!(rotation.is_some());
            assert_eq!(q.dtype(), DType::F32);
            assert!(try_fused_f16_sdpa_native(&q, &k, &v, false, 1.0)?.is_none());
        }
        Ok(())
    }

    #[test]
    fn fused_sdpa_geometry_accepts_gqa_and_narrow_values() {
        let geometry = fused_sdpa_geometry([2, 16, 3, 192], [2, 4, 11, 192], [2, 4, 11, 128])
            .expect("valid MLA geometry");
        assert_eq!(geometry.batch, 2);
        assert_eq!(geometry.q_heads, 16);
        assert_eq!(geometry.kv_heads, 4);
        assert_eq!(geometry.q_seq, 3);
        assert_eq!(geometry.kv_seq, 11);
        assert_eq!(geometry.qk_dim, 192);
        assert_eq!(geometry.v_dim, 128);

        assert!(fused_sdpa_geometry([1, 6, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]).is_none());
        assert!(fused_sdpa_geometry([1, 4, 5, 64], [1, 2, 4, 64], [1, 2, 4, 64]).is_none());
        assert!(fused_sdpa_geometry([1, 4, 1, 64], [1, 2, 1, 32], [1, 2, 1, 64]).is_none());
    }

    #[test]
    fn cuda_flash_value_padding_materializes_contiguous_cache_views() -> Result<()> {
        let storage = Tensor::ones((1, 4, 64, 128), DType::F16, &Device::Cpu)?;
        let active = storage.narrow(2, 0, 1)?;
        assert!(!active.is_contiguous());

        let padded = pad_cuda_flash_attention_values(&active, 128, 192)?;
        assert!(padded.is_contiguous());
        assert_eq!(padded.dims4()?, (1, 4, 1, 192));
        assert_eq!(padded.stride()[3], 1);

        let flash_layout = padded.transpose(1, 2)?;
        assert_eq!(flash_layout.dims4()?, (1, 1, 4, 192));
        assert_eq!(flash_layout.stride()[3], 1);

        let values = padded
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for head in values.chunks_exact(192) {
            assert!(head[..128].iter().all(|value| *value == 1.0));
            assert!(head[128..].iter().all(|value| *value == 0.0));
        }
        Ok(())
    }

    #[test]
    fn cuda_flash_attention_accepts_newer_compute_generations() {
        assert!(!cuda_flash_attention_compute_supported(7));
        for major in [8, 9, 10, 12] {
            assert!(cuda_flash_attention_compute_supported(major));
        }
    }

    #[test]
    fn sliding_window_mask_uses_bottom_right_query_alignment() {
        let mask = sliding_window_mask_values(3, 5, 2).expect("valid mask");
        let visible = mask
            .chunks_exact(5)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter_map(|(key, value)| value.is_finite().then_some(key))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(visible, vec![vec![1, 2], vec![2, 3], vec![3, 4]]);
        assert!(sliding_window_mask_values(0, 5, 2).is_none());
        assert!(sliding_window_mask_values(3, 2, 2).is_none());
        assert!(sliding_window_mask_values(3, 5, 0).is_none());
    }

    #[test]
    fn windowed_prefill_trim_preserves_every_visible_key() {
        for q_seq in [2, 3, 31, 128, 129] {
            for kv_seq in [q_seq, q_seq + 31, q_seq + 257] {
                for window in [1, 31, 32, 33, 128, 1024] {
                    let skip = metal_windowed_kv_skip(q_seq, kv_seq, window);
                    assert_eq!(skip % 32, 0);
                    let full = sliding_window_mask_values(q_seq, kv_seq, window).unwrap();
                    let trimmed = sliding_window_mask_values(q_seq, kv_seq - skip, window).unwrap();
                    for (before, after) in full
                        .chunks_exact(kv_seq)
                        .zip(trimmed.chunks_exact(kv_seq - skip))
                    {
                        assert!(before[..skip].iter().all(|v| *v == f32::NEG_INFINITY));
                        assert_eq!(&before[skip..], after);
                    }
                }
            }
        }
        assert_eq!(
            metal_windowed_kv_skip(1, 8192, 128),
            8064,
            "decode trims its exact window"
        );
        assert_eq!(metal_windowed_kv_skip(1, 5, 2), 3);
        assert_eq!(metal_windowed_kv_skip(1, 5, 10), 0);
        assert_eq!(metal_windowed_kv_skip(0, 8, 2), 0);
        assert_eq!(metal_windowed_kv_skip(8, 4, 2), 0);
        assert_eq!(metal_windowed_kv_skip(8, 16, 0), 0);
        assert_eq!(metal_windowed_kv_skip(128, 8192, usize::MAX), 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_single_query_window_excludes_old_values_with_strided_gqa() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for (batch, q_heads, kv_heads) in [(1, 1, 1), (2, 4, 2)] {
            for (kv_seq, window) in [(5, 2), (65, 33), (5, 5), (5, 9)] {
                let q = Tensor::zeros((batch, q_heads, 1, 64), DType::F16, &device)?;
                let shape = (batch, kv_heads, kv_seq + 8, 64);
                let k = Tensor::zeros(shape, DType::F16, &device)?.narrow(2, 3, kv_seq)?;
                let mut values = vec![1.0f32; batch * kv_heads * (kv_seq + 8) * 64];
                let keep = window.min(kv_seq);
                for head in values.chunks_exact_mut((kv_seq + 8) * 64) {
                    head[(3 + kv_seq - keep) * 64..(3 + kv_seq) * 64].fill(0.0);
                }
                let v = Tensor::from_vec(values, shape, &device)?
                    .to_dtype(DType::F16)?
                    .narrow(2, 3, kv_seq)?;
                let actual = try_metal_f16_sdpa_windowed_native(&q, &k, &v, window, 0.125)?
                    .expect("supported geometry");
                let values = actual
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert!(values.iter().all(|value| *value == 0.0), "{values:?}");
            }
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_trimmed_window_matches_full_mask_with_strided_cache() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for dim in [64, 256, 512] {
            for (q_seq, kv_seq, window) in [(3, 321, 33), (128, 1025, 128), (129, 1056, 129)] {
                let data = |shape: (usize, usize, usize, usize), salt: usize| -> Result<Tensor> {
                    let len = shape.0 * shape.1 * shape.2 * shape.3;
                    Ok(Tensor::from_vec(
                        (0..len)
                            .map(|i| (((i * salt) % 251) as f32 - 125.) / 251.)
                            .collect(),
                        shape,
                        &device,
                    )?
                    .to_dtype(DType::F16)?)
                };
                let q = data((2, 4, q_seq, dim), 17)?;
                // Nonzero base offset, spare capacity, and multi-head strides.
                let k = data((2, 2, kv_seq + 31, dim), 29)?.narrow(2, 7, kv_seq)?;
                let v = data((2, 2, kv_seq + 31, dim), 37)?.narrow(2, 7, kv_seq)?;
                let mask = Tensor::from_vec(
                    sliding_window_mask_values(q_seq, kv_seq, window).unwrap(),
                    (1, 1, q_seq, kv_seq),
                    &device,
                )?
                .to_dtype(DType::F16)?
                .broadcast_as((2, 4, q_seq, kv_seq))?;
                let scale = 1. / (dim as f32).sqrt();
                let expected = candle_nn::ops::sdpa(&q, &k, &v, Some(&mask), false, scale, 1.)?;
                let actual = try_metal_f16_sdpa_windowed_native(&q, &k, &v, window, scale)?
                    .expect("supported geometry");
                // The shared helper uses a strict `< tolerance` check. F16
                // values cannot differ by less than this positive F32 limit.
                // Head dims 256 and 512 run the Metal 4 flash kernel, which
                // rounds differently from Steel; dim 64 keeps the exact check.
                let flash = matches!(dim, 256 | 512)
                    && super::tensor_qmatmul_metal::supports_tensor_batch_attention(&device)?;
                assert_close(
                    &actual,
                    &expected,
                    if flash { 2e-3 } else { f32::MIN_POSITIVE },
                    "trimmed vs full window",
                )?;
            }
        }
        Ok(())
    }

    #[test]
    fn causal_mask_uses_bottom_right_query_alignment() {
        let mask = causal_mask_values(3, 5).expect("valid mask");
        let visible = mask
            .chunks_exact(5)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter_map(|(key, value)| value.is_finite().then_some(key))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            visible,
            vec![vec![0, 1, 2], vec![0, 1, 2, 3], vec![0, 1, 2, 3, 4]]
        );
        assert!(causal_mask_values(0, 5).is_none());
        assert!(causal_mask_values(3, 2).is_none());
    }

    #[cfg(any(feature = "cuda", target_os = "macos"))]
    fn portable_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        causal: bool,
        window: Option<usize>,
    ) -> Result<Tensor> {
        let (batch, q_heads, q_seq, q_dim) = q.dims4()?;
        let (_, kv_heads, kv_seq, _) = k.dims4()?;
        let v_dim = v.dim(3)?;
        let n_rep = q_heads / kv_heads;
        let k = k
            .to_dtype(DType::F32)?
            .unsqueeze(2)?
            .expand((batch, kv_heads, n_rep, kv_seq, q_dim))?
            .reshape((batch, q_heads, kv_seq, q_dim))?
            .contiguous()?;
        let v = v
            .to_dtype(DType::F32)?
            .unsqueeze(2)?
            .expand((batch, kv_heads, n_rep, kv_seq, v_dim))?
            .reshape((batch, q_heads, kv_seq, v_dim))?
            .contiguous()?;
        let scale = 1.0 / (q_dim as f32).sqrt();
        let weights = (q.to_dtype(DType::F32)?.matmul(&k.transpose(2, 3)?)? * scale as f64)?;
        let mask = if let Some(window) = window {
            sliding_window_mask_values(q_seq, kv_seq, window)
        } else if causal {
            let offset = kv_seq - q_seq;
            Some(
                (0..q_seq)
                    .flat_map(|query| {
                        (0..kv_seq).map(move |key| {
                            if key <= offset + query {
                                0.0
                            } else {
                                f32::NEG_INFINITY
                            }
                        })
                    })
                    .collect(),
            )
        } else {
            None
        };
        let weights = if let Some(mask) = mask {
            weights.broadcast_add(&Tensor::from_vec(mask, (1, 1, q_seq, kv_seq), q.device())?)?
        } else {
            weights
        };
        candle_nn::ops::softmax_last_dim(&weights)?
            .matmul(&v)
            .map_err(Into::into)
    }

    #[cfg(any(feature = "cuda", target_os = "macos"))]
    fn assert_close(lhs: &Tensor, rhs: &Tensor, limit: f32, context: &str) -> Result<()> {
        let lhs = lhs
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let rhs = rhs
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(lhs.len(), rhs.len(), "{context}: output length differs");
        assert!(
            lhs.iter().chain(&rhs).all(|v| v.is_finite()),
            "{context}: non-finite attention output"
        );
        let max_diff = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < limit, "{context}: max_diff={max_diff}");
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_causal_attention_never_reads_nan_padding_past_kv_length() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for dim in [64, 128, 256, 512] {
            for len in [2usize, 7, 16, 17, 31, 32, 33, 63, 65] {
                let capacity = len.next_multiple_of(32) + 32;
                let q = Tensor::ones((1, 2, len, dim), DType::F16, &device)?;
                let backing = Tensor::from_vec(
                    (0..capacity * dim)
                        .map(|i| {
                            if i / dim < len {
                                ((i * 17 % 127) as f32 - 63.) / 128.
                            } else {
                                f32::NAN
                            }
                        })
                        .collect::<Vec<_>>(),
                    (1, 1, capacity, dim),
                    &device,
                )?
                .to_dtype(DType::F16)?;
                let kv = backing.narrow(2, 0, len)?;
                let actual =
                    try_metal_f16_sdpa_native(&q, &kv, &kv, true, 1.0 / (dim as f32).sqrt())?
                        .expect("supported Metal attention");
                let expected = portable_attention(&q, &kv, &kv, true, None)?;
                assert_close(
                    &actual,
                    &expected,
                    0.005,
                    &format!("causal attention dim={dim} len={len}"),
                )?;
            }
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_windowed_attention_supports_narrow_value_heads() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (batch, q_heads, kv_heads, q_seq, kv_seq, q_dim, v_dim) = (1, 4, 2, 3, 5, 64, 32);
        let q = Tensor::from_vec(
            (0..batch * q_heads * q_seq * q_dim)
                .map(|index| (((index * 17) % 251) as f32 / 251.0 - 0.5) * 0.25)
                .collect::<Vec<_>>(),
            (batch, q_heads, q_seq, q_dim),
            &device,
        )?;
        let k = Tensor::from_vec(
            (0..batch * kv_heads * kv_seq * q_dim)
                .map(|index| (((index * 29) % 241) as f32 / 241.0 - 0.5) * 0.25)
                .collect::<Vec<_>>(),
            (batch, kv_heads, kv_seq, q_dim),
            &device,
        )?
        .to_dtype(DType::F16)?;
        let v = Tensor::from_vec(
            (0..batch * kv_heads * kv_seq * v_dim)
                .map(|index| (((index * 37) % 239) as f32 / 239.0 - 0.5) * 0.25)
                .collect::<Vec<_>>(),
            (batch, kv_heads, kv_seq, v_dim),
            &device,
        )?
        .to_dtype(DType::F16)?;
        let window = 2;
        let scale = 1.0 / (q_dim as f32).sqrt();
        let fused_full = try_metal_f16_sdpa_native(&q, &k, &v, false, scale)?
            .expect("supported Metal SDPA geometry");
        let portable_full = portable_attention(&q, &k, &v, false, None)?;
        assert_close(
            &fused_full,
            &portable_full,
            0.025,
            "Metal narrow-value attention",
        )?;
        let fused_causal = try_metal_f16_sdpa_native(&q, &k, &v, true, scale)?
            .expect("supported Metal SDPA geometry");
        let portable_causal = portable_attention(&q, &k, &v, true, None)?;
        assert_close(
            &fused_causal,
            &portable_causal,
            0.025,
            "Metal cached causal attention",
        )?;
        let fused = try_metal_f16_sdpa_windowed_native(&q, &k, &v, window, scale)?
            .expect("supported Metal SDPA geometry");
        let portable = portable_attention(&q, &k, &v, true, Some(window))?;
        assert_close(
            &fused,
            &portable,
            0.025,
            "Metal windowed narrow-value attention",
        )
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_flash_standard_windowed_and_narrow_value_paths_match_portable() -> Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };

        let batch = 1;
        let q_heads = 4;
        let kv_heads = 2;
        let q_dim = 64;
        for (q_seq, kv_seq, v_dim, causal, window) in [
            (1, 17, 64, false, None),
            (17, 17, 64, true, None),
            (3, 7, 64, true, None),
            (7, 7, 64, true, Some(3)),
            (3, 7, 64, true, Some(2)),
            (5, 5, 32, true, None),
        ] {
            let q_values = (0..batch * q_heads * q_seq * q_dim)
                .map(|index| (((index * 17) % 251) as f32 / 251.0 - 0.5) * 0.25)
                .collect::<Vec<_>>();
            let k_values = (0..batch * kv_heads * kv_seq * q_dim)
                .map(|index| (((index * 29) % 241) as f32 / 241.0 - 0.5) * 0.25)
                .collect::<Vec<_>>();
            let v_values = (0..batch * kv_heads * kv_seq * v_dim)
                .map(|index| (((index * 37) % 239) as f32 / 239.0 - 0.5) * 0.25)
                .collect::<Vec<_>>();
            let q = Tensor::from_vec(q_values, (batch, q_heads, q_seq, q_dim), &device)?;
            let k = Tensor::from_vec(k_values, (batch, kv_heads, kv_seq, q_dim), &device)?
                .to_dtype(DType::F16)?;
            let v = Tensor::from_vec(v_values, (batch, kv_heads, kv_seq, v_dim), &device)?
                .to_dtype(DType::F16)?;
            let scale = 1.0 / (q_dim as f32).sqrt();

            let fused = if let Some(window) = window {
                try_cuda_f16_sdpa_windowed_native(&q, &k, &v, window, scale)?
            } else {
                try_cuda_f16_sdpa_native(&q, &k, &v, causal, scale)?
            };
            let Some(fused) = fused else {
                eprintln!("skipping CUDA FlashAttention test: unsupported CUDA device");
                return Ok(());
            };
            let portable = portable_attention(&q, &k, &v, causal, window)?;
            assert_close(
                &fused,
                &portable,
                0.025,
                &format!(
                    "CUDA FlashAttention diverged for q={q_seq}, kv={kv_seq}, v={v_dim}, causal={causal}, window={window:?}"
                ),
            )?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_paged_ragged_attention_matches_lane_attention() -> Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        let (batch, q_heads, kv_heads, capacity, head_dim) = (3, 4, 2, 64, 64);
        let valid_lengths = [2, 17, 33];
        let q = Tensor::from_vec(
            (0..batch * q_heads * head_dim)
                .map(|index| (((index * 13) % 251) as f32 / 251.0 - 0.5) * 0.25)
                .collect::<Vec<_>>(),
            (batch, q_heads, 1, head_dim),
            &device,
        )?;
        let k = Tensor::from_vec(
            (0..batch * kv_heads * capacity * head_dim)
                .map(|index| (((index * 23) % 241) as f32 / 241.0 - 0.5) * 0.25)
                .collect::<Vec<_>>(),
            (batch, kv_heads, capacity, head_dim),
            &device,
        )?
        .to_dtype(DType::F16)?;
        let v = Tensor::from_vec(
            (0..batch * kv_heads * capacity * head_dim)
                .map(|index| (((index * 31) % 239) as f32 / 239.0 - 0.5) * 0.25)
                .collect::<Vec<_>>(),
            (batch, kv_heads, capacity, head_dim),
            &device,
        )?
        .to_dtype(DType::F16)?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let Some(fused) = try_cuda_f16_ragged_sdpa(&q, &k, &v, &valid_lengths, scale)? else {
            eprintln!("skipping CUDA ragged FlashAttention test: unsupported CUDA device");
            return Ok(());
        };

        let mut lanes = Vec::with_capacity(batch);
        for (lane, valid_len) in valid_lengths.iter().copied().enumerate() {
            lanes.push(portable_attention(
                &q.narrow(0, lane, 1)?,
                &k.narrow(0, lane, 1)?.narrow(2, 0, valid_len)?,
                &v.narrow(0, lane, 1)?.narrow(2, 0, valid_len)?,
                false,
                None,
            )?);
        }
        let lane_refs = lanes.iter().collect::<Vec<_>>();
        let portable = Tensor::cat(&lane_refs, 0)?;
        assert_close(&fused, &portable, 0.025, "CUDA paged ragged attention")
    }

    #[test]
    fn aligned_initial_kv_cache_adopts_f16_storage() -> Result<()> {
        let k = Tensor::zeros((1, 2, 64, 4), DType::F16, &Device::Cpu)?;
        let v = Tensor::zeros((1, 2, 64, 4), DType::F16, &Device::Cpu)?;
        let k_id = k.id();
        let v_id = v.id();
        let cache = PreallocatedKvCache::from_f16(k, v)?;
        assert_eq!(cache.k.id(), k_id);
        assert_eq!(cache.v.id(), v_id);
        assert_eq!(cache.len, 64);
        assert_eq!(cache.capacity, 64);
        Ok(())
    }

    #[test]
    fn preallocated_kv_supports_different_key_and_value_widths() -> Result<()> {
        let k = Tensor::full(1f32, (1, 2, 3, 6), &Device::Cpu)?;
        let v = Tensor::full(2f32, (1, 2, 3, 4), &Device::Cpu)?;
        let mut cache = None;

        let (k, v) = kv_cache_update_native(&mut cache, k, v, None)?;
        assert_eq!(k.dims(), &[1, 2, 3, 6]);
        assert_eq!(v.dims(), &[1, 2, 3, 4]);

        let next_k = Tensor::full(3f32, (1, 2, 1, 6), &Device::Cpu)?;
        let next_v = Tensor::full(4f32, (1, 2, 1, 4), &Device::Cpu)?;
        let (k, v) = kv_cache_update_native(&mut cache, next_k, next_v, None)?;
        assert_eq!(k.dims(), &[1, 2, 4, 6]);
        assert_eq!(v.dims(), &[1, 2, 4, 4]);
        assert!(
            k.i((0, 0, 3, ..))?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?
                .iter()
                .all(|value| *value == 3.0)
        );
        assert!(
            v.i((0, 0, 3, ..))?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?
                .iter()
                .all(|value| *value == 4.0)
        );
        Ok(())
    }

    #[test]
    fn candle_metal_sdpa_gate_accepts_supported_512_wide_heads() {
        assert!(metal_sdpa_head_dim_supported(256));
        assert!(metal_sdpa_head_dim_supported(512));
    }

    #[test]
    fn preallocated_kv_appends_in_place_until_growth() -> Result<()> {
        let (k, v) = constant_kv(1.0, 63);
        let mut cache = None;
        kv_cache_update(&mut cache, k, v, None)?;

        let (storage_id, capacity) = match &cache {
            Some(KvCache::PreallocatedF16(dense)) => {
                assert_eq!(dense.len, 63);
                (dense.k.id(), dense.capacity)
            }
            _ => panic!("expected preallocated f16 cache"),
        };
        assert_eq!(capacity, 64);

        let (k, v) = constant_kv(2.0, 1);
        kv_cache_update(&mut cache, k, v, None)?;
        match &cache {
            Some(KvCache::PreallocatedF16(dense)) => {
                assert_eq!(dense.k.id(), storage_id);
                assert_eq!(dense.len, 64);
                let (active, _) = dense.active_f16()?;
                assert!(
                    active
                        .i((0, 0, 63, ..))?
                        .to_dtype(DType::F32)?
                        .to_vec1::<f32>()?
                        .iter()
                        .all(|value| *value == 2.0)
                );
            }
            _ => panic!("expected preallocated f16 cache"),
        }

        let (k, v) = constant_kv(3.0, 1);
        kv_cache_update(&mut cache, k, v, None)?;
        match &cache {
            Some(KvCache::PreallocatedF16(dense)) => {
                assert_ne!(dense.k.id(), storage_id);
                assert_eq!(dense.len, 65);
                assert_eq!(dense.capacity, 128);
            }
            _ => panic!("expected preallocated f16 cache"),
        }
        Ok(())
    }

    #[test]
    fn preallocated_kv_eviction_is_a_logical_window() -> Result<()> {
        let values = (0..10).map(|value| value as f32).collect::<Vec<_>>();
        let k = Tensor::from_vec(values, (1, 1, 10, 1), &Device::Cpu)?;
        let mut cache = None;
        kv_cache_update(&mut cache, k.clone(), k, None)?;

        let Some(KvCache::PreallocatedF16(dense)) = cache.as_mut() else {
            panic!("expected preallocated f16 cache")
        };
        let storage_id = dense.k.id();
        dense.evict_oldest(3)?;
        assert_eq!(dense.k.id(), storage_id);
        assert_eq!(dense.start, 3);
        assert_eq!(dense.len, 7);
        let (active, _) = dense.active_f16()?;
        assert_eq!(
            active
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            (3..10).map(|value| value as f32).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn preallocated_kv_reserve_keeps_rows_and_compacts_or_grows() -> Result<()> {
        let values = (0..60).map(|value| value as f32).collect::<Vec<_>>();
        let k = Tensor::from_vec(values, (1, 1, 60, 1), &Device::Cpu)?;
        let mut cache = None;
        kv_cache_update(&mut cache, k.clone(), k, None)?;
        let Some(KvCache::PreallocatedF16(dense)) = cache.as_mut() else {
            panic!("expected preallocated f16 cache")
        };
        let rows = |dense: &PreallocatedKvCache| -> Result<Vec<f32>> {
            Ok(dense
                .active_f16()?
                .0
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?)
        };
        // A reservation that fits returns the next unused row of the same storage.
        let storage_id = dense.k.id();
        let (k_cache, _, row) = dense.reserve(4)?;
        assert_eq!((row, k_cache.id(), dense.len), (60, storage_id, 60));
        dense.commit(4)?;
        assert_eq!(dense.len, 64);
        // After eviction, a reservation past the end compacts the live rows.
        dense.evict_oldest(40)?;
        let (_, _, row) = dense.reserve(8)?;
        assert_eq!((dense.start, row, dense.capacity), (0, 24, 64));
        assert_ne!(dense.k.id(), storage_id);
        assert_eq!(
            rows(dense)?,
            (40..60)
                .map(|v| v as f32)
                .chain([0.0; 4])
                .collect::<Vec<_>>()
        );
        dense.commit(8)?;
        // A reservation beyond the capacity grows it.
        let (_, _, row) = dense.reserve(64)?;
        assert_eq!((row, dense.capacity, dense.len), (32, 128, 32));
        assert!(dense.commit(97).is_err());
        Ok(())
    }

    #[test]
    fn ragged_preallocated_kv_grows_and_appends_each_lane_at_its_position() -> Result<()> {
        let k = Tensor::zeros((2, 1, 4, 1), DType::F16, &Device::Cpu)?;
        let v = Tensor::zeros((2, 1, 4, 1), DType::F16, &Device::Cpu)?;
        PreallocatedKvCache::scatter_append(
            &k.narrow(0, 0, 1)?,
            &Tensor::full(1f32, (1, 1, 2, 1), &Device::Cpu)?.to_dtype(DType::F16)?,
            0,
        )?;
        PreallocatedKvCache::scatter_append(
            &k.narrow(0, 1, 1)?,
            &Tensor::full(2f32, (1, 1, 4, 1), &Device::Cpu)?.to_dtype(DType::F16)?,
            0,
        )?;
        PreallocatedKvCache::scatter_append(&v, &k, 0)?;
        let old_id = k.id();
        let mut cache = PreallocatedKvCache {
            k,
            v,
            start: 0,
            len: 4,
            capacity: 4,
            allocation_lanes: 2,
        };
        let incoming = Tensor::from_vec(vec![10f32, 20f32], (2, 1, 1, 1), &Device::Cpu)?;
        cache.append_ragged(&incoming, &incoming, &[2, 4])?;

        assert_ne!(cache.k.id(), old_id);
        assert_eq!(cache.len, 5);
        assert_eq!(cache.capacity, 64);
        assert_eq!(
            cache
                .k
                .narrow(0, 0, 1)?
                .narrow(2, 0, 5)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            vec![1.0, 1.0, 10.0, 0.0, 0.0]
        );
        assert_eq!(
            cache
                .k
                .narrow(0, 1, 1)?
                .narrow(2, 0, 5)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            vec![2.0, 2.0, 2.0, 2.0, 20.0]
        );
        Ok(())
    }

    #[test]
    fn ragged_kv_reuses_equal_shape_scatter_indices() -> Result<()> {
        let k = Tensor::zeros((2, 4, 1, 8), DType::F16, &Device::Cpu)?;
        let v = Tensor::zeros((2, 4, 1, 8), DType::F16, &Device::Cpu)?;
        let (k_indices, v_indices) = PreallocatedKvCache::ragged_indices(&[3, 7], &k, &v)?;
        assert_eq!(k_indices.id(), v_indices.id());

        let narrow_v = Tensor::zeros((2, 4, 1, 4), DType::F16, &Device::Cpu)?;
        let (k_indices, v_indices) = PreallocatedKvCache::ragged_indices(&[3, 7], &k, &narrow_v)?;
        assert_ne!(k_indices.id(), v_indices.id());
        assert_eq!(k_indices.dims(), k.dims());
        assert_eq!(v_indices.dims(), narrow_v.dims());

        assert!(PreallocatedKvCache::ragged_indices(&[3], &k, &v).is_err());
        let single_lane_v = Tensor::zeros((1, 4, 1, 8), DType::F16, &Device::Cpu)?;
        assert!(PreallocatedKvCache::ragged_indices(&[3, 7], &k, &single_lane_v).is_err());
        Ok(())
    }

    #[test]
    #[cfg(target_os = "macos")]
    #[ignore = "manual Metal microbenchmark"]
    fn benchmark_shared_ragged_kv_indices() -> Result<()> {
        use std::time::{Duration, Instant};

        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (batch, heads, capacity, head_dim) = (4usize, 8usize, 64usize, 128usize);
        let positions = [3usize, 11, 27, 63];
        let dst_k = Tensor::zeros((batch, heads, capacity, head_dim), DType::F16, &device)?;
        let dst_v = Tensor::zeros((batch, heads, capacity, head_dim), DType::F16, &device)?;
        let k = Tensor::full(1f32, (batch, heads, 1, head_dim), &device)?.to_dtype(DType::F16)?;
        let v = Tensor::full(2f32, (batch, heads, 1, head_dim), &device)?.to_dtype(DType::F16)?;
        let iterations = 10_000usize;

        let old_indices = |src: &Tensor| -> Result<Tensor> {
            let values = positions
                .iter()
                .map(|position| *position as u32)
                .collect::<Vec<_>>();
            Ok(Tensor::from_vec(values, (batch, 1, 1, 1), src.device())?
                .broadcast_as(src.shape())?
                .contiguous()?)
        };
        let measure = |shared: bool| -> Result<Duration> {
            device.synchronize()?;
            let started = Instant::now();
            for _ in 0..iterations {
                let (k_indices, v_indices) = if shared {
                    PreallocatedKvCache::ragged_indices(&positions, &k, &v)?
                } else {
                    (old_indices(&k)?, old_indices(&v)?)
                };
                dst_k.scatter_set(&k_indices, &k, 2)?;
                dst_v.scatter_set(&v_indices, &v, 2)?;
            }
            device.synchronize()?;
            Ok(started.elapsed())
        };

        measure(false)?;
        measure(true)?;
        let mut old_runs = Vec::with_capacity(5);
        let mut shared_runs = Vec::with_capacity(5);
        for round in 0usize..5 {
            if round.is_multiple_of(2) {
                shared_runs.push(measure(true)?);
                old_runs.push(measure(false)?);
            } else {
                old_runs.push(measure(false)?);
                shared_runs.push(measure(true)?);
            }
        }
        old_runs.sort_unstable();
        shared_runs.sort_unstable();
        let old = old_runs[old_runs.len() / 2];
        let shared = shared_runs[shared_runs.len() / 2];
        eprintln!(
            "RAGGED_KV_INDEX_BENCH iterations={iterations} old_ms={:.3} shared_ms={:.3} speedup={:.3}x",
            old.as_secs_f64() * 1_000.0,
            shared.as_secs_f64() * 1_000.0,
            old.as_secs_f64() / shared.as_secs_f64()
        );
        let written_k = dst_k
            .i((0, 0, positions[0], 0))?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?;
        let written_v = dst_v
            .i((0, 0, positions[0], 0))?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?;
        assert_eq!((written_k, written_v), (1.0, 2.0));
        Ok(())
    }

    #[test]
    fn test_f16_evict_oldest() {
        let device = Device::Cpu;
        // Create F16 KvCache with shape [1, 4, 10, 64]
        let k = Tensor::randn(0.0f32, 1.0, (1, 4, 10, 64), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (1, 4, 10, 64), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        // Get reference to tail 7 tokens
        let k_tail = k.narrow(2, 3, 7).unwrap();

        let mut cache = KvCache::F16 { k, v };
        cache.evict_oldest(3).unwrap();

        match &cache {
            KvCache::F16 { k, v } => {
                assert_eq!(k.dims(), &[1, 4, 7, 64]);
                assert_eq!(v.dims(), &[1, 4, 7, 64]);
                // Values should match the tail
                let k_flat: Vec<f32> = k
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                let k_tail_flat: Vec<f32> = k_tail
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                let max_diff: f32 = k_flat
                    .iter()
                    .zip(k_tail_flat.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_diff < 1e-3,
                    "evicted cache should match tail, max_diff={max_diff}"
                );
            }
            _ => panic!("expected F16 cache"),
        }
    }

    #[test]
    fn test_evict_skips_deltanet() {
        let device = Device::Cpu;
        let k = Tensor::randn(0.0f32, 1.0, (1, 4, 10, 64), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (1, 4, 10, 64), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        let conv_state = Tensor::zeros((32, 4), DType::F32, &device).unwrap();
        let recurrent_state = Tensor::zeros((8, 16, 32), DType::F32, &device).unwrap();
        let deltanet_cache = DeltaNetCache {
            conv_state: conv_state.clone(),
            recurrent_state: recurrent_state.clone(),
            speculative_state: None,
        };

        let mut entries: Vec<Option<LayerCache>> = vec![
            Some(LayerCache::Attention(KvCache::F16 { k, v })),
            Some(LayerCache::DeltaNet(deltanet_cache)),
        ];

        // Evict from attention layers only
        for entry in entries.iter_mut() {
            if let Some(LayerCache::Attention(kv)) = entry {
                let _ = kv.evict_oldest(3);
            }
        }

        // Attention cache should be evicted
        match &entries[0] {
            Some(LayerCache::Attention(KvCache::F16 { k, .. })) => {
                assert_eq!(k.dim(2).unwrap(), 7);
            }
            _ => panic!("expected attention cache"),
        }

        // DeltaNet cache should be untouched
        match &entries[1] {
            Some(LayerCache::DeltaNet(d)) => {
                assert_eq!(d.conv_state.dims(), conv_state.dims());
                assert_eq!(d.recurrent_state.dims(), recurrent_state.dims());
            }
            _ => panic!("expected deltanet cache"),
        }
    }

    fn check_fixed_state_views_release_oversized_storage(device: &Device) -> Result<()> {
        fn state_view(device: &Device, dims: &[usize], strided: bool) -> Result<Tensor> {
            let elements = dims.iter().product::<usize>();
            if strided {
                let last_dim = dims[dims.len() - 1];
                let mut backing_dims = dims.to_vec();
                *backing_dims.last_mut().unwrap() = last_dim * 4;
                let values = (0..elements * 4)
                    .map(|i| i as f32 + 0.25)
                    .collect::<Vec<_>>();
                Ok(Tensor::from_vec(values, backing_dims, device)?.narrow(
                    dims.len() - 1,
                    last_dim,
                    last_dim,
                )?)
            } else {
                let mut backing_dims = vec![4];
                backing_dims.extend_from_slice(dims);
                let values = (0..elements * 4)
                    .map(|i| i as f32 + 0.5)
                    .collect::<Vec<_>>();
                Ok(Tensor::from_vec(values, backing_dims, device)?.i(2)?)
            }
        }

        for strided in [false, true] {
            for deltanet in [false, true] {
                let conv = state_view(device, &[5, 3], strided)?;
                let recurrent = state_view(device, &[2, 3, 5], strided)?;
                assert_eq!(conv.is_contiguous(), !strided);
                assert_eq!(recurrent.is_contiguous(), !strided);
                let original_values = [
                    conv.flatten_all()?.to_vec1::<f32>()?,
                    recurrent.flatten_all()?.to_vec1::<f32>()?,
                ];
                let original_ids = [conv.id(), recurrent.id()];
                let original_storage = [
                    state_storage_bytes(&conv)?,
                    state_storage_bytes(&recurrent)?,
                ];
                for tensor in [&conv, &recurrent] {
                    assert!(
                        state_storage_bytes(tensor)?
                            > fixed_state_tensor_bytes(tensor.dims(), device)?,
                        "fixture must retain oversized storage"
                    );
                }
                let mut cache = if deltanet {
                    LayerCache::DeltaNet(DeltaNetCache {
                        conv_state: conv,
                        recurrent_state: recurrent,
                        speculative_state: None,
                    })
                } else {
                    LayerCache::Mamba2(Mamba2Cache {
                        conv_state: conv,
                        ssm_state: recurrent,
                    })
                };
                compact_fixed_layer_cache(&mut cache)?;
                let states = match &cache {
                    LayerCache::DeltaNet(cache) => [&cache.conv_state, &cache.recurrent_state],
                    LayerCache::Mamba2(cache) => [&cache.conv_state, &cache.ssm_state],
                    _ => unreachable!(),
                };
                let compact_ids = states.map(Tensor::id);
                for (index, tensor) in states.iter().enumerate() {
                    assert_ne!(tensor.id(), original_ids[index], "oversized view must copy");
                    assert_eq!(
                        tensor.flatten_all()?.to_vec1::<f32>()?,
                        original_values[index]
                    );
                    assert!(tensor.is_contiguous());
                    let storage = state_storage_bytes(tensor)?;
                    assert!(storage < original_storage[index]);
                    assert!(storage <= fixed_state_tensor_bytes(tensor.dims(), device)?);
                }
                compact_fixed_layer_cache(&mut cache)?;
                let next_ids = match &cache {
                    LayerCache::DeltaNet(cache) => {
                        [cache.conv_state.id(), cache.recurrent_state.id()]
                    }
                    LayerCache::Mamba2(cache) => [cache.conv_state.id(), cache.ssm_state.id()],
                    _ => unreachable!(),
                };
                assert_eq!(next_ids, compact_ids, "compact state must keep its storage");
            }
        }
        device.synchronize()?;
        Ok(())
    }

    #[test]
    fn fixed_state_views_release_oversized_cpu_storage() -> Result<()> {
        check_fixed_state_views_release_oversized_storage(&Device::Cpu)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn fixed_state_views_release_oversized_metal_storage() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        check_fixed_state_views_release_oversized_storage(&device)
    }

    #[test]
    fn speculative_commit_selects_recurrent_snapshot() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = LayerCache::DeltaNet(DeltaNetCache {
            conv_state: Tensor::zeros((2, 2), DType::F32, &device)?,
            recurrent_state: Tensor::zeros((1, 2, 2), DType::F32, &device)?,
            speculative_state: Some(DeltaNetSpeculativeState::Snapshots(DeltaNetSnapshots {
                conv_state: Tensor::new(
                    &[
                        [[1.0f32, 1.0], [1.0, 1.0]],
                        [[2.0, 2.0], [2.0, 2.0]],
                        [[3.0, 3.0], [3.0, 3.0]],
                    ],
                    &device,
                )?,
                recurrent_state: Tensor::new(
                    &[
                        [[[1.0f32, 1.0], [1.0, 1.0]]],
                        [[[2.0, 2.0], [2.0, 2.0]]],
                        [[[3.0, 3.0], [3.0, 3.0]]],
                    ],
                    &device,
                )?,
            })),
        });

        commit_speculative_layer_cache(&mut cache, 1, 1)?;
        let LayerCache::DeltaNet(cache) = cache else {
            panic!("expected DeltaNet cache");
        };
        assert_eq!(cache.conv_state.to_vec2::<f32>()?, vec![vec![2.0; 2]; 2]);
        assert_eq!(
            cache.recurrent_state.to_vec3::<f32>()?,
            vec![vec![vec![2.0; 2]; 2]]
        );
        assert!(cache.speculative_state.is_none());
        Ok(())
    }

    #[test]
    fn speculative_commit_discards_unaccepted_kv_tail() -> Result<()> {
        let device = Device::Cpu;
        let k = Tensor::zeros((1, 2, 5, 4), DType::F16, &device)?;
        let v = Tensor::zeros((1, 2, 5, 4), DType::F16, &device)?;
        let mut cache = LayerCache::Attention(KvCache::PreallocatedF16(
            PreallocatedKvCache::from_f16(k, v)?,
        ));

        commit_speculative_layer_cache(&mut cache, 1, 3)?;
        let LayerCache::Attention(KvCache::PreallocatedF16(cache)) = cache else {
            panic!("expected preallocated attention cache");
        };
        assert_eq!(cache.len, 2);
        assert_eq!(cache.active_f16()?.0.dim(2)?, 2);
        Ok(())
    }
}
