use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::qmatmul::QMatMul;
use anyhow::{Context, Result};
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, IndexOp, Module, Tensor};

use super::{KvCache, LayerCache, ModelArchitecture, StageInput, StageOutput, with_kv_cache};
use crate::gguf_loader::StageTensors;
use crate::turboquant::TurboQuantConfig;

static ATTN_NORM_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_NS: AtomicU64 = AtomicU64::new(0);
static ATTN_RESIDUAL_NS: AtomicU64 = AtomicU64::new(0);
static FFN_NORM_NS: AtomicU64 = AtomicU64::new(0);
static FFN_NS: AtomicU64 = AtomicU64::new(0);
static FFN_RESIDUAL_NS: AtomicU64 = AtomicU64::new(0);
static FFN_GATE_NS: AtomicU64 = AtomicU64::new(0);
static FFN_UP_NS: AtomicU64 = AtomicU64::new(0);
static FFN_ACTIVATION_NS: AtomicU64 = AtomicU64::new(0);
static FFN_DOWN_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_CALLS: AtomicU64 = AtomicU64::new(0);

fn layer_profile_enabled() -> bool {
    super::profiling::layer_profiling_enabled()
}

fn finish_profile_region(device: &Device, started: Instant, counter: &AtomicU64) -> Result<()> {
    device.synchronize()?;
    counter.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(())
}

fn lm_head_forward(lm_head: &QMatMul, hidden: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Some(logits) = super::dense_qmatmul_cuda::forward_one_to_f32(hidden, lm_head)? {
        return Ok(logits);
    }
    lm_head.forward(hidden).map_err(Into::into)
}

fn prompt_tail_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Option<Tensor>> {
    if k.dim(2)? == 1 {
        return Ok(Some(repeat_kv(v.clone(), q.dim(1)? / k.dim(1)?)?));
    }
    // Select prompt attention math with two queries. The last query sees all
    // real keys; discard the other query. Never append a dummy KV row.
    let queries = Tensor::cat(&[q, q], 2)?;
    super::try_metal_f16_sdpa_native(&queries, k, v, true, scale)?
        .map(|output| output.narrow(2, 1, 1)?.contiguous())
        .transpose()
        .map_err(Into::into)
}

/// Reset the process-local Llama-family layer profiling counters.
pub fn reset_layer_profile() {
    for counter in [
        &ATTN_NORM_NS,
        &ATTN_NS,
        &ATTN_RESIDUAL_NS,
        &FFN_NORM_NS,
        &FFN_NS,
        &FFN_RESIDUAL_NS,
        &FFN_GATE_NS,
        &FFN_UP_NS,
        &FFN_ACTIVATION_NS,
        &FFN_DOWN_NS,
        &PROFILE_CALLS,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

/// Print the process-local Llama-family layer profiling counters to stderr.
pub fn report_layer_profile() {
    let calls = PROFILE_CALLS.load(Ordering::Relaxed).max(1) as f64;
    let ms_per_call =
        |counter: &AtomicU64| counter.load(Ordering::Relaxed) as f64 / calls / 1_000_000.0;
    eprintln!(
        "DINF_LAYER_PROFILE calls={} per_layer_ms={{attn_norm:{:.4},attention:{:.4},attn_residual:{:.4},ffn_norm:{:.4},ffn:{:.4},ffn_residual:{:.4}}}",
        PROFILE_CALLS.load(Ordering::Relaxed),
        ms_per_call(&ATTN_NORM_NS),
        ms_per_call(&ATTN_NS),
        ms_per_call(&ATTN_RESIDUAL_NS),
        ms_per_call(&FFN_NORM_NS),
        ms_per_call(&FFN_NS),
        ms_per_call(&FFN_RESIDUAL_NS),
    );
    eprintln!(
        "DINF_FFN_PROFILE calls={} per_layer_ms={{gate:{:.4},up:{:.4},activation:{:.4},down:{:.4}}}",
        PROFILE_CALLS.load(Ordering::Relaxed),
        ms_per_call(&FFN_GATE_NS),
        ms_per_call(&FFN_UP_NS),
        ms_per_call(&FFN_ACTIVATION_NS),
        ms_per_call(&FFN_DOWN_NS),
    );
}

/// A pipeline stage running the Llama-family architecture.
///
/// Covers Llama, Mistral, Qwen2, and DeepSeek-distill (same GGUF tensor layout).
/// Supports partial layer loading for pipeline parallelism.
pub struct LlamaStage {
    /// Token embedding (head stage only).
    embedding: Option<Tensor>,
    /// Transformer blocks for this stage's layer range.
    blocks: Vec<TransformerBlock>,
    /// Final RMSNorm (tail stage only).
    output_norm: Option<RmsNorm>,
    /// Language model head projection (tail stage only).
    lm_head: Option<QMatMul>,
    /// Whether this is the head stage (has embedding).
    is_head: bool,
    /// Whether this is the tail stage (has output_norm + lm_head).
    is_tail: bool,
    hidden_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    device: Device,
}

struct TransformerBlock {
    attn_norm: RmsNorm,
    attn: Attention,
    ffn_norm: RmsNorm,
    ffn: Mlp,
}

struct Attention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_bias: Option<Tensor>,
    k_bias: Option<Tensor>,
    v_bias: Option<Tensor>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// Number of keys each query may attend to, including itself (0 = full context).
    sliding_window: usize,
    /// Precomputed RoPE cos table: [context_length, head_dim/2]
    rope_cos: Tensor,
    /// Precomputed RoPE sin table: [context_length, head_dim/2]
    rope_sin: Tensor,
    /// RoPE convention: true = NEOX/split-half, false = GGUF Llama/Mistral interleaved.
    rope_is_neox: bool,
    /// TurboQuant KV cache config (None = f16 storage).
    quant_config: Option<Arc<TurboQuantConfig>>,
}

struct Mlp {
    /// Use fixed CPU F32 row tiles only in non-windowed layers.
    canonical_cpu_projections: bool,
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(candle_nn::ops::rms_norm(x, &self.weight, self.eps as f32)?)
    }

    fn forward_residual(&self, residual: &Tensor, update: &Tensor) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        if let Some(result) =
            super::residual_norm_cuda::forward(residual, update, &self.weight, self.eps as f32)?
        {
            return Ok(result);
        }
        let residual = (residual + update)?;
        let normalized = self.forward(&residual)?;
        Ok((residual, normalized))
    }
}

impl Attention {
    fn project(&self, weights: &QMatMul, input: &Tensor) -> Result<Tensor> {
        if self.sliding_window == 0 {
            super::cpu_math::project(weights, input)
        } else {
            weights.forward(input).map_err(Into::into)
        }
    }

    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor, matrix_tail: bool) -> Result<Tensor> {
        let seq_len = q.dim(2)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let fused = if self.sliding_window > 0 {
            super::try_fused_f16_sdpa_windowed_native(q, k, v, self.sliding_window, scale as f32)?
        } else {
            if matrix_tail && let Some(output) = prompt_tail_attention(q, k, v, scale as f32)? {
                return Ok(output);
            }
            super::try_fused_f16_sdpa_native(q, k, v, seq_len > 1, scale as f32)?
        };
        if let Some(output) = fused {
            return Ok(output);
        }
        if self.sliding_window == 0
            && let Some(output) = super::cpu_math::attention(q, k, v, scale)?
        {
            return Ok(output);
        }
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(k.to_dtype(q.dtype())?, n_rep)?;
        let v = repeat_kv(v.to_dtype(q.dtype())?, n_rep)?;
        let weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let weights = if seq_len > 1 || self.sliding_window > 0 {
            let mask =
                create_attention_mask(seq_len, weights.dim(3)?, self.sliding_window, q.device())?
                    .to_dtype(weights.dtype())?;
            weights.broadcast_add(&mask)?
        } else {
            weights
        };
        softmax_last_dim(&weights)?.matmul(&v).map_err(Into::into)
    }

    fn forward(
        &self,
        x: &Tensor,
        kv_cache: &mut Option<KvCache>,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<Tensor> {
        self.forward_pass(x, kv_cache, seq_len, start_pos, false)
    }

    fn forward_pass(
        &self,
        x: &Tensor,
        kv_cache: &mut Option<KvCache>,
        seq_len: usize,
        start_pos: usize,
        prefill: bool,
    ) -> Result<Tensor> {
        let (b_sz, _, _) = x.dims3()?;
        let matrix_tail = prefill && seq_len == 1 && x.device().is_metal();

        #[cfg(feature = "cuda")]
        let (grouped_qkv, qkv_epilogue_applied) = {
            let fused_qkv = if seq_len == 1 && self.rope_is_neox {
                match (
                    self.q_bias.as_ref(),
                    self.k_bias.as_ref(),
                    self.v_bias.as_ref(),
                ) {
                    (Some(q_bias), Some(k_bias), Some(v_bias)) => {
                        super::dense_qmatmul_cuda::forward_qkv(
                            x,
                            [&self.q_proj, &self.k_proj, &self.v_proj],
                            q_bias,
                            k_bias,
                            v_bias,
                            &self.rope_cos,
                            &self.rope_sin,
                            start_pos,
                            self.num_heads,
                            self.num_kv_heads,
                            self.head_dim,
                            self.quant_config.is_none()
                                && !matches!(kv_cache.as_ref(), Some(KvCache::Quantized(_))),
                        )?
                    }
                    _ => None,
                }
            } else {
                None
            };
            if let Some(outputs) = fused_qkv {
                (Some(outputs), true)
            } else {
                let grouped = super::dense_qmatmul_cuda::forward(
                    x,
                    &[&self.q_proj, &self.k_proj, &self.v_proj],
                )?;
                (grouped, false)
            }
        };
        #[cfg(not(feature = "cuda"))]
        let (grouped_qkv, qkv_epilogue_applied): (Option<Vec<Tensor>>, bool) = (None, false);

        let (mut q, mut k, mut v) = if matrix_tail {
            let rows = Tensor::cat(&[x, x], 1)?;
            (
                self.project(&self.q_proj, &rows)?
                    .narrow(1, 0, 1)?
                    .contiguous()?,
                self.project(&self.k_proj, &rows)?
                    .narrow(1, 0, 1)?
                    .contiguous()?,
                self.project(&self.v_proj, &rows)?
                    .narrow(1, 0, 1)?
                    .contiguous()?,
            )
        } else if let Some(mut outputs) = grouped_qkv {
            anyhow::ensure!(outputs.len() == 3, "grouped QKV output count mismatch");
            let v = outputs.pop().unwrap();
            let k = outputs.pop().unwrap();
            let q = outputs.pop().unwrap();
            (q, k, v)
        } else {
            (
                self.project(&self.q_proj, x)?,
                self.project(&self.k_proj, x)?,
                self.project(&self.v_proj, x)?,
            )
        };

        // Apply biases if present (Qwen2, etc.). The fused CUDA QKV path has
        // already applied all three biases in place.
        if !qkv_epilogue_applied {
            if let Some(ref bias) = self.q_bias {
                q = q.broadcast_add(bias)?;
            }
            if let Some(ref bias) = self.k_bias {
                k = k.broadcast_add(bias)?;
            }
            if let Some(ref bias) = self.v_bias {
                v = v.broadcast_add(bias)?;
            }
        }

        // Reshape to [batch, heads, seq_len, head_dim]
        // For seq_len=1 (decode), reshape directly without transpose (avoids contiguous copy).
        // For seq_len>1 (prefill), must transpose since data is interleaved per-position.
        let (q, k, v) = if seq_len == 1 {
            let q = q.reshape((b_sz, self.num_heads, 1, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, 1, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, 1, self.head_dim))?;
            (q, k, v)
        } else {
            #[cfg(target_os = "macos")]
            if let Some(layout) = super::qkv_layout_metal::try_fused_separate_qkv_layout(
                &q,
                &k,
                &v,
                super::qkv_layout_metal::QkvLayoutDimensions {
                    batch: b_sz,
                    seq_len,
                    num_heads: self.num_heads,
                    num_kv_heads: self.num_kv_heads,
                    head_dim: self.head_dim,
                },
            )? {
                layout
            } else {
                self.portable_qkv_layout(q, k, v, b_sz, seq_len)?
            }
            #[cfg(not(target_os = "macos"))]
            self.portable_qkv_layout(q, k, v, b_sz, seq_len)?
        };

        // Apply RoPE via precomputed cos/sin tables. The fused CUDA QKV path
        // has already applied split-half RoPE to Q and K in place.
        let (q, k) = if qkv_epilogue_applied {
            (q, k)
        } else {
            (
                apply_rotary_emb(
                    &q,
                    start_pos,
                    &self.rope_cos,
                    &self.rope_sin,
                    self.rope_is_neox,
                )?,
                apply_rotary_emb(
                    &k,
                    start_pos,
                    &self.rope_cos,
                    &self.rope_sin,
                    self.rope_is_neox,
                )?,
            )
        };

        // Quantized Metal attention can run in the rotated basis. The helper
        // appends K/V to the cache and returns the query/K/V tensors plus an
        // optional inverse-rotation token for the attention output.
        let (q, k, v, rotation) =
            super::prepare_quantized_attention(kv_cache, q, k, v, self.quant_config.as_deref())?;

        let attn_output = self.attend(&q, &k, &v, matrix_tail)?;
        let attn_output = super::restore_quantized_attention_output(attn_output, rotation)?;

        #[cfg(target_os = "macos")]
        let attn_output = if let Some(output) =
            super::qkv_layout_metal::try_fused_attention_output_layout(&attn_output)?
        {
            output
        } else {
            attn_output
                .to_dtype(x.dtype())?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b_sz, seq_len, ()))?
        };
        #[cfg(not(target_os = "macos"))]
        let attn_output =
            attn_output
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b_sz, seq_len, ()))?;
        #[cfg(all(feature = "cuda", not(target_os = "macos")))]
        if let Some(output) =
            super::dense_qmatmul_cuda::forward_one_to_f32(&attn_output, &self.o_proj)?
        {
            return Ok(output);
        }
        #[cfg(not(target_os = "macos"))]
        let attn_output = attn_output.to_dtype(x.dtype())?;
        let output = if matrix_tail {
            let rows = Tensor::cat(&[&attn_output, &attn_output], 1)?;
            self.project(&self.o_proj, &rows)?
                .narrow(1, 0, 1)?
                .contiguous()?
        } else {
            self.project(&self.o_proj, &attn_output)?
        };
        Ok(output)
    }

    fn portable_qkv_layout(
        &self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        batch: usize,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        Ok((q, k, v))
    }

    /// Decode lanes with different absolute positions. Projection and output
    /// kernels remain batched; RoPE, cache append, and SDPA stay lane-local so
    /// no padded KV positions can leak into attention.
    fn forward_ragged_decode(
        &self,
        x: &Tensor,
        caches: &mut [&mut Option<KvCache>],
        positions: &[usize],
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        anyhow::ensure!(seq_len == 1, "ragged attention is decode-only");
        anyhow::ensure!(batch == caches.len(), "ragged attention cache mismatch");
        anyhow::ensure!(
            batch == positions.len(),
            "ragged attention position mismatch"
        );

        let mut q = self.project(&self.q_proj, x)?;
        let mut k = self.project(&self.k_proj, x)?;
        let mut v = self.project(&self.v_proj, x)?;
        if let Some(bias) = self.q_bias.as_ref() {
            q = q.broadcast_add(bias)?;
        }
        if let Some(bias) = self.k_bias.as_ref() {
            k = k.broadcast_add(bias)?;
        }
        if let Some(bias) = self.v_bias.as_ref() {
            v = v.broadcast_add(bias)?;
        }

        let q = q.reshape((batch, self.num_heads, 1, self.head_dim))?;
        let k = k.reshape((batch, self.num_kv_heads, 1, self.head_dim))?;
        let v = v.reshape((batch, self.num_kv_heads, 1, self.head_dim))?;
        let mut outputs = Vec::with_capacity(batch);

        for lane in 0..batch {
            let q = apply_rotary_emb(
                &q.narrow(0, lane, 1)?,
                positions[lane],
                &self.rope_cos,
                &self.rope_sin,
                self.rope_is_neox,
            )?;
            let k = apply_rotary_emb(
                &k.narrow(0, lane, 1)?,
                positions[lane],
                &self.rope_cos,
                &self.rope_sin,
                self.rope_is_neox,
            )?;
            let v = v.narrow(0, lane, 1)?;
            let (k, v) = super::kv_cache_update_native(
                &mut *caches[lane],
                k,
                v,
                self.quant_config.as_deref(),
            )?;
            let output = self.attend(&q, &k, &v, false)?.to_dtype(q.dtype())?;
            outputs.push(output);
        }

        let output_refs = outputs.iter().collect::<Vec<_>>();
        let output = Tensor::cat(&output_refs, 0)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, 1, ()))?;
        self.project(&self.o_proj, &output)
    }

    /// Mixed-position decode over one persistent packed KV allocation.
    ///
    /// RoPE remains lane-specific, while KV append and attention are dispatched
    /// once for the whole batch. The fused kernel reads each lane's valid length
    /// so allocation slack cannot affect the softmax.
    fn forward_packed_ragged_decode(
        &self,
        x: &Tensor,
        cache: &mut Option<KvCache>,
        positions: &[usize],
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        anyhow::ensure!(seq_len == 1, "ragged attention is decode-only");
        anyhow::ensure!(
            batch == positions.len(),
            "ragged attention position mismatch"
        );

        let mut q = self.project(&self.q_proj, x)?;
        let mut k = self.project(&self.k_proj, x)?;
        let mut v = self.project(&self.v_proj, x)?;
        if let Some(bias) = self.q_bias.as_ref() {
            q = q.broadcast_add(bias)?;
        }
        if let Some(bias) = self.k_bias.as_ref() {
            k = k.broadcast_add(bias)?;
        }
        if let Some(bias) = self.v_bias.as_ref() {
            v = v.broadcast_add(bias)?;
        }

        let q = q.reshape((batch, self.num_heads, 1, self.head_dim))?;
        let k = k.reshape((batch, self.num_kv_heads, 1, self.head_dim))?;
        let v = v.reshape((batch, self.num_kv_heads, 1, self.head_dim))?;
        let (q, k) = if let Some(qk) = super::try_metal_ragged_rope(
            &q,
            &k,
            &self.rope_cos,
            &self.rope_sin,
            positions,
            self.rope_is_neox,
        )? {
            qk
        } else {
            let mut q_lanes = Vec::with_capacity(batch);
            let mut k_lanes = Vec::with_capacity(batch);
            for (lane, position) in positions.iter().copied().enumerate() {
                q_lanes.push(apply_rotary_emb(
                    &q.narrow(0, lane, 1)?,
                    position,
                    &self.rope_cos,
                    &self.rope_sin,
                    self.rope_is_neox,
                )?);
                k_lanes.push(apply_rotary_emb(
                    &k.narrow(0, lane, 1)?,
                    position,
                    &self.rope_cos,
                    &self.rope_sin,
                    self.rope_is_neox,
                )?);
            }
            let q_refs = q_lanes.iter().collect::<Vec<_>>();
            let k_refs = k_lanes.iter().collect::<Vec<_>>();
            (Tensor::cat(&q_refs, 0)?, Tensor::cat(&k_refs, 0)?)
        };

        let Some(KvCache::PreallocatedF16(dense)) = cache.as_mut() else {
            anyhow::bail!("packed ragged decode requires preallocated f16 KV storage")
        };
        dense.append_ragged(&k, &v, positions)?;
        let valid_lengths = positions
            .iter()
            .map(|position| position.saturating_add(1))
            .collect::<Vec<_>>();
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let fused = if self.sliding_window == 0 {
            super::try_fused_f16_ragged_sdpa(&q, &dense.k, &dense.v, &valid_lengths, scale as f32)?
        } else {
            None
        };
        let attn_output = if let Some(output) = fused {
            output
        } else {
            let mut outputs = Vec::with_capacity(batch);
            for (lane, valid_len) in valid_lengths.iter().copied().enumerate() {
                let lane_q = q.narrow(0, lane, 1)?;
                let lane_k = dense.k.narrow(0, lane, 1)?.narrow(2, 0, valid_len)?;
                let lane_v = dense.v.narrow(0, lane, 1)?.narrow(2, 0, valid_len)?;
                outputs.push(
                    self.attend(&lane_q, &lane_k, &lane_v, false)?
                        .to_dtype(q.dtype())?,
                );
            }
            let output_refs = outputs.iter().collect::<Vec<_>>();
            Tensor::cat(&output_refs, 0)?
        };

        let output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, 1, ()))?;
        self.project(&self.o_proj, &output)
    }
}

fn silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    // Keep CUDA's exact epilogue: the shared MoE helper uses a different
    // CUDA implementation, so it cannot replace this path unchanged.
    #[cfg(feature = "cuda")]
    if gate.device().is_cuda() {
        return super::indexed_moe_epilogue_cuda::silu_mul_exact(gate, up);
    }
    // Metal stays unfused: full-model controls showed a small Coder gain but
    // slower sampled Llama/Distill runs with the fused SiLU kernel. This also
    // preserves the native dtype and strided-tensor behavior.
    (candle_nn::Activation::Silu.forward(gate)? * up).map_err(Into::into)
}

impl Mlp {
    fn project(&self, weights: &QMatMul, input: &Tensor) -> Result<Tensor> {
        if self.canonical_cpu_projections {
            super::cpu_math::project(weights, input)
        } else {
            weights.forward(input).map_err(Into::into)
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let profile = layer_profile_enabled();
        #[cfg(feature = "cuda")]
        if !profile
            && let Some(activated) =
                super::dense_qmatmul_cuda::silu_gate_up(x, &self.gate_proj, &self.up_proj)?
        {
            if let Some(down) =
                super::dense_qmatmul_cuda::forward_one_to_f32(&activated, &self.down_proj)?
            {
                return Ok(down);
            }
            return self.down_proj.forward(&activated).map_err(Into::into);
        }
        let started = Instant::now();
        #[cfg(feature = "cuda")]
        let grouped_gate_up =
            super::dense_qmatmul_cuda::forward(x, &[&self.gate_proj, &self.up_proj])?;
        #[cfg(not(feature = "cuda"))]
        let grouped_gate_up: Option<Vec<Tensor>> = None;
        let (gate, grouped_up) = if let Some(mut outputs) = grouped_gate_up {
            anyhow::ensure!(outputs.len() == 2, "grouped gate/up output count mismatch");
            let up = outputs.pop().unwrap();
            let gate = outputs.pop().unwrap();
            (gate, Some(up))
        } else {
            (self.project(&self.gate_proj, x)?, None)
        };
        if profile {
            finish_profile_region(x.device(), started, &FFN_GATE_NS)?;
        }
        let started = Instant::now();
        let up = match grouped_up {
            Some(up) => up,
            None => self.project(&self.up_proj, x)?,
        };
        if profile {
            finish_profile_region(x.device(), started, &FFN_UP_NS)?;
        }
        #[cfg(feature = "cuda")]
        if !profile
            && let Some(down) =
                super::dense_qmatmul_cuda::silu_mul_down(&gate, &up, &self.down_proj)?
        {
            return Ok(down);
        }
        let started = Instant::now();
        let activated = silu_mul(&gate, &up)?;
        if profile {
            finish_profile_region(x.device(), started, &FFN_ACTIVATION_NS)?;
        }
        let started = Instant::now();
        let down = self.project(&self.down_proj, &activated)?;
        if profile {
            finish_profile_region(x.device(), started, &FFN_DOWN_NS)?;
        }
        Ok(down)
    }
}

impl TransformerBlock {
    fn forward(
        &self,
        x: &Tensor,
        kv_cache: &mut Option<KvCache>,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<Tensor> {
        self.forward_pass(x, kv_cache, seq_len, start_pos, false)
    }

    fn forward_pass(
        &self,
        x: &Tensor,
        kv_cache: &mut Option<KvCache>,
        seq_len: usize,
        start_pos: usize,
        prefill: bool,
    ) -> Result<Tensor> {
        let profile = layer_profile_enabled();
        if profile {
            PROFILE_CALLS.fetch_add(1, Ordering::Relaxed);
        }

        // Pre-norm attention
        let residual = x;
        let started = Instant::now();
        let x = self.attn_norm.forward(x)?;
        if profile {
            finish_profile_region(x.device(), started, &ATTN_NORM_NS)?;
        }
        let started = Instant::now();
        let x = self
            .attn
            .forward_pass(&x, kv_cache, seq_len, start_pos, prefill)?;
        if profile {
            finish_profile_region(x.device(), started, &ATTN_NS)?;
        }
        let started = Instant::now();
        let x = (residual + x)?;
        if profile {
            finish_profile_region(x.device(), started, &ATTN_RESIDUAL_NS)?;
        }

        // Pre-norm FFN
        let residual = &x;
        let started = Instant::now();
        let ffn_out = self.ffn_norm.forward(&x)?;
        if profile {
            finish_profile_region(ffn_out.device(), started, &FFN_NORM_NS)?;
        }
        let started = Instant::now();
        let ffn_out = if prefill && seq_len == 1 && ffn_out.device().is_metal() {
            let rows = Tensor::cat(&[&ffn_out, &ffn_out], 1)?;
            self.ffn.forward(&rows)?.narrow(1, 0, 1)?.contiguous()?
        } else {
            self.ffn.forward(&ffn_out)?
        };
        if profile {
            finish_profile_region(ffn_out.device(), started, &FFN_NS)?;
        }
        let started = Instant::now();
        let x = (residual + ffn_out)?;
        if profile {
            finish_profile_region(x.device(), started, &FFN_RESIDUAL_NS)?;
        }

        Ok(x)
    }

    fn forward_ragged_decode(
        &self,
        x: &Tensor,
        caches: &mut [&mut Option<KvCache>],
        positions: &[usize],
    ) -> Result<Tensor> {
        let profile = layer_profile_enabled();
        if profile {
            PROFILE_CALLS.fetch_add(1, Ordering::Relaxed);
        }

        let residual = x;
        let started = Instant::now();
        let x = self.attn_norm.forward(x)?;
        if profile {
            finish_profile_region(x.device(), started, &ATTN_NORM_NS)?;
        }
        let started = Instant::now();
        let x = self.attn.forward_ragged_decode(&x, caches, positions)?;
        if profile {
            finish_profile_region(x.device(), started, &ATTN_NS)?;
        }
        let started = Instant::now();
        let x = (residual + x)?;
        if profile {
            finish_profile_region(x.device(), started, &ATTN_RESIDUAL_NS)?;
        }

        let residual = &x;
        let started = Instant::now();
        let ffn_out = self.ffn_norm.forward(&x)?;
        if profile {
            finish_profile_region(ffn_out.device(), started, &FFN_NORM_NS)?;
        }
        let started = Instant::now();
        let ffn_out = self.ffn.forward(&ffn_out)?;
        if profile {
            finish_profile_region(ffn_out.device(), started, &FFN_NS)?;
        }
        let started = Instant::now();
        let x = (residual + ffn_out)?;
        if profile {
            finish_profile_region(x.device(), started, &FFN_RESIDUAL_NS)?;
        }
        Ok(x)
    }

    fn forward_packed_ragged_decode(
        &self,
        x: &Tensor,
        cache: &mut Option<KvCache>,
        positions: &[usize],
    ) -> Result<Tensor> {
        let profile = layer_profile_enabled();
        if profile {
            PROFILE_CALLS.fetch_add(1, Ordering::Relaxed);
        }

        let residual = x;
        let started = Instant::now();
        let x = self.attn_norm.forward(x)?;
        if profile {
            finish_profile_region(x.device(), started, &ATTN_NORM_NS)?;
        }
        let started = Instant::now();
        let x = self
            .attn
            .forward_packed_ragged_decode(&x, cache, positions)?;
        if profile {
            finish_profile_region(x.device(), started, &ATTN_NS)?;
        }
        let started = Instant::now();
        let x = (residual + x)?;
        if profile {
            finish_profile_region(x.device(), started, &ATTN_RESIDUAL_NS)?;
        }

        let residual = &x;
        let started = Instant::now();
        let ffn_out = self.ffn_norm.forward(&x)?;
        if profile {
            finish_profile_region(ffn_out.device(), started, &FFN_NORM_NS)?;
        }
        let started = Instant::now();
        let ffn_out = self.ffn.forward(&ffn_out)?;
        if profile {
            finish_profile_region(ffn_out.device(), started, &FFN_NS)?;
        }
        let started = Instant::now();
        let x = (residual + ffn_out)?;
        if profile {
            finish_profile_region(x.device(), started, &FFN_RESIDUAL_NS)?;
        }
        Ok(x)
    }
}

/// Helper: convert Arc<QTensor> to QMatMul.
fn qtensor_to_qmatmul(qt: Arc<QTensor>) -> Result<QMatMul> {
    QMatMul::from_arc(qt).map_err(Into::into)
}

impl LlamaStage {
    /// Construct a Llama-family stage from its assigned GGUF tensors.
    ///
    /// Supports the Llama, Mistral, and Qwen2 layouts recognized by the
    /// dispatcher. Returns an error when required tensors or metadata are
    /// missing or inconsistent.
    pub fn from_stage_tensors(
        stage: StageTensors,
        quant_config: Option<Arc<TurboQuantConfig>>,
    ) -> Result<Self> {
        let md = &stage.metadata;
        let device = &stage.device;
        let range = &stage.range;
        let tensors = &stage.tensors;

        // Metadata controls the size of the generated context-length RoPE
        // tables. Check it against the still-packed projection shapes before
        // dequantizing anything or allocating those tables. A tiny malformed
        // model must not be able to request arbitrarily wide positional data.
        validate_attention_geometry(&stage)?;

        // Embedding (head only)
        let embedding = if range.is_head() {
            let embd = tensors
                .get("token_embd.weight")
                .context("missing token_embd.weight for head stage")?;
            Some(embd.as_ref().dequantize(device)?)
        } else {
            None
        };

        let rope_freqs = tensors
            .get("rope_freqs.weight")
            .map(|t| t.as_ref().dequantize(device))
            .transpose()?;
        let rope_is_neox = matches!(md.architecture.as_str(), "qwen2");
        #[cfg(feature = "cuda")]
        {
            super::dense_qmatmul_cuda::precompile(device)?;
            super::residual_norm_cuda::precompile(device)?;
            super::indexed_moe_epilogue_cuda::precompile_exact_silu(device)?;
            if rope_is_neox {
                super::qkv_decode_cuda::precompile(device)?;
            }
        }

        // Precompute RoPE cos/sin table for the full context length.
        // Llama 3.1 GGUFs carry rope_freqs.weight with per-dimension scaling
        // factors; ignoring it corrupts even short generations.
        let (rope_cos, rope_sin) = precompute_rope_cos_sin(
            md.head_dim,
            md.rope_theta,
            md.context_length,
            device,
            rope_freqs.as_ref(),
        )?;

        // Transformer blocks
        let mut blocks = Vec::with_capacity(range.layer_count());
        let build_context = BlockBuildContext {
            tensors,
            metadata: md,
            device,
            rope_cos: &rope_cos,
            rope_sin: &rope_sin,
            rope_is_neox,
            quant_config,
        };
        for layer_idx in range.start..range.end {
            let prefix = format!("blk.{layer_idx}");
            let block = build_block(&build_context, &prefix, layer_idx)?;
            blocks.push(block);
        }

        // Output norm + lm_head (tail only)
        let output_norm = if range.is_tail() {
            let norm_w = tensors
                .get("output_norm.weight")
                .context("missing output_norm.weight for tail stage")?;
            Some(RmsNorm {
                weight: norm_w.as_ref().dequantize(device)?,
                eps: md.rms_norm_eps,
            })
        } else {
            None
        };

        let lm_head = if range.is_tail() {
            // Some models share embedding weights for output
            let head = tensors
                .get("output.weight")
                .or_else(|| tensors.get("token_embd.weight"))
                .context("missing output.weight or token_embd.weight for tail stage")?;
            Some(qtensor_to_qmatmul(Arc::clone(head))?)
        } else {
            None
        };

        Ok(Self {
            embedding,
            blocks,
            output_norm,
            lm_head,
            is_head: range.is_head(),
            is_tail: range.is_tail(),
            hidden_dim: md.hidden_dim,
            num_layers: range.layer_count(),
            vocab_size: md.vocab_size,
            device: device.clone(),
        })
    }
}

fn clone_decode_session_cache(cache: &[Option<LayerCache>]) -> Result<Vec<Option<LayerCache>>> {
    cache
        .iter()
        .map(|slot| match slot {
            None => Ok(None),
            Some(layer) => {
                super::clone_decode_layer_cache(layer).map(|cache| cache.map(LayerCache::Attention))
            }
        })
        .collect()
}

impl LlamaStage {
    fn forward_packed_ragged_decode_batch(
        &self,
        tokens: &[u32],
        caches: &mut [&mut [Option<LayerCache>]],
        batch_cache: &mut [Option<LayerCache>],
        positions: &[usize],
    ) -> Result<Vec<Tensor>> {
        let batch = tokens.len();
        let embedding = self.embedding.as_ref().unwrap();
        let token_tensor = Tensor::new(tokens, &self.device)?;
        let mut hidden_states =
            embedding
                .index_select(&token_tensor, 0)?
                .reshape((batch, 1, self.hidden_dim))?;

        let mut next_batch_cache = Vec::with_capacity(self.blocks.len());
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let mut packed = match &batch_cache[layer_idx] {
                Some(cache) => super::clone_decode_layer_cache(cache)?,
                None => super::pack_ragged_decode_layer_cache(caches, layer_idx, positions)?,
            };
            hidden_states =
                block.forward_packed_ragged_decode(&hidden_states, &mut packed, positions)?;
            next_batch_cache.push(packed);
        }

        let normed = self.output_norm.as_ref().unwrap().forward(&hidden_states)?;
        let logits = lm_head_forward(self.lm_head.as_ref().unwrap(), &normed.squeeze(1)?)?;
        let outputs = (0..batch)
            .map(|lane| logits.narrow(0, lane, 1)?.squeeze(0).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;
        let valid_lengths = positions
            .iter()
            .map(|position| position.saturating_add(1))
            .collect::<Vec<_>>();

        for (layer_idx, cache) in next_batch_cache.iter().enumerate() {
            super::publish_ragged_decode_layer_cache_views(
                caches,
                layer_idx,
                cache,
                &valid_lengths,
            )?;
        }
        for (slot, cache) in batch_cache.iter_mut().zip(next_batch_cache) {
            *slot = cache.map(LayerCache::Attention);
        }
        Ok(outputs)
    }

    fn forward_ragged_decode_batch(
        &self,
        tokens: &[u32],
        caches: &mut [&mut [Option<LayerCache>]],
        positions: &[usize],
    ) -> Result<Vec<Tensor>> {
        let batch = tokens.len();
        let embedding = self.embedding.as_ref().unwrap();
        let token_tensor = Tensor::new(tokens, &self.device)?;
        let mut hidden_states =
            embedding
                .index_select(&token_tensor, 0)?
                .reshape((batch, 1, self.hidden_dim))?;

        // Cache metadata is cloned transactionally. Appends may write unused
        // slots in shared backing storage, but original logical lengths remain
        // unchanged unless the entire model forward succeeds.
        let mut next_caches = caches
            .iter()
            .map(|cache| clone_decode_session_cache(cache))
            .collect::<Result<Vec<_>>>()?;

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let mut layer_caches = Vec::with_capacity(batch);
            for cache in &mut next_caches {
                let kv = match cache[layer_idx].take() {
                    Some(LayerCache::Attention(kv)) => Some(kv),
                    None => None,
                    Some(_) => anyhow::bail!("ragged decode encountered non-attention state"),
                };
                layer_caches.push(kv);
            }
            let mut layer_refs = layer_caches.iter_mut().collect::<Vec<_>>();
            hidden_states =
                block.forward_ragged_decode(&hidden_states, &mut layer_refs, positions)?;
            for (cache, kv) in next_caches.iter_mut().zip(layer_caches) {
                cache[layer_idx] = kv.map(LayerCache::Attention);
            }
        }

        let normed = self.output_norm.as_ref().unwrap().forward(&hidden_states)?;
        let logits = lm_head_forward(self.lm_head.as_ref().unwrap(), &normed.squeeze(1)?)?;
        let outputs = (0..batch)
            .map(|lane| logits.narrow(0, lane, 1)?.squeeze(0).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;

        for (target, source) in caches.iter_mut().zip(next_caches) {
            anyhow::ensure!(target.len() == source.len(), "ragged cache layer mismatch");
            for (slot, next) in target.iter_mut().zip(source) {
                *slot = next;
            }
        }
        Ok(outputs)
    }
}

impl LlamaStage {
    fn forward_pass(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
        prefill: bool,
    ) -> Result<StageOutput> {
        let mut hidden_states = match input {
            StageInput::Tokens(tokens) => {
                anyhow::ensure!(self.is_head, "only head stage can receive tokens");
                let embedding = self.embedding.as_ref().unwrap();
                let token_tensor = Tensor::new(tokens.as_slice(), &self.device)?;
                embedding
                    .index_select(&token_tensor, 0)?
                    .reshape((1, seq_len, self.hidden_dim))?
            }
            StageInput::Activations { tensor, .. } => {
                if tensor.dims().len() == 2 {
                    tensor.unsqueeze(0)?
                } else {
                    tensor
                }
            }
        };

        #[cfg(feature = "cuda")]
        let fuse_decode_norms = seq_len == 1 && self.device.is_cuda() && !layer_profile_enabled();
        #[cfg(not(feature = "cuda"))]
        let fuse_decode_norms = false;

        let mut tail_normed = None;
        if fuse_decode_norms {
            // Carry each block's already-normalized input across the layer
            // boundary. This lets residual+RMSNorm replace both the attention
            // residual boundary and the FFN-to-next-layer boundary.
            let mut next_attn_normed = None;
            for (i, block) in self.blocks.iter().enumerate() {
                let normed = if let Some(normed) = next_attn_normed.take() {
                    normed
                } else {
                    block.attn_norm.forward(&hidden_states)?
                };
                let attn_out = with_kv_cache(&mut cache[i], |kv| {
                    block.attn.forward(&normed, kv, seq_len, start_pos)
                })?;
                let (next_hidden_states, normed) =
                    block.ffn_norm.forward_residual(&hidden_states, &attn_out)?;
                hidden_states = next_hidden_states;
                let ffn_out = block.ffn.forward(&normed)?;

                if let Some(next_block) = self.blocks.get(i + 1) {
                    let (next_hidden_states, normed) = next_block
                        .attn_norm
                        .forward_residual(&hidden_states, &ffn_out)?;
                    hidden_states = next_hidden_states;
                    next_attn_normed = Some(normed);
                } else if self.is_tail {
                    let (next_hidden_states, normed) = self
                        .output_norm
                        .as_ref()
                        .unwrap()
                        .forward_residual(&hidden_states, &ffn_out)?;
                    hidden_states = next_hidden_states;
                    tail_normed = Some(normed);
                } else {
                    hidden_states = (hidden_states + ffn_out)?;
                }
            }
        } else {
            // Forward through transformer blocks.
            for (i, block) in self.blocks.iter().enumerate() {
                hidden_states = with_kv_cache(&mut cache[i], |kv| {
                    block.forward_pass(&hidden_states, kv, seq_len, start_pos, prefill)
                })?;
            }
        }

        if self.is_tail {
            // Apply output norm and lm_head
            let normed = if let Some(normed) = tail_normed {
                normed
            } else {
                self.output_norm.as_ref().unwrap().forward(&hidden_states)?
            };
            // Take the last token's hidden state for logit computation
            let last_hidden = if seq_len > 1 {
                normed.i((.., seq_len - 1, ..))?
            } else {
                normed.squeeze(1)?
            };
            let logits = lm_head_forward(self.lm_head.as_ref().unwrap(), &last_hidden)?;
            // Squeeze to 1D [vocab_size] for sampling
            let logits = logits.squeeze(0)?;
            Ok(StageOutput::Logits(logits))
        } else {
            // Pass activations to next stage — squeeze batch dim to [seq, hidden]
            let output = hidden_states.squeeze(0)?; // [seq_len, hidden_dim]
            Ok(StageOutput::Activations(output))
        }
    }
}

impl ModelArchitecture for LlamaStage {
    fn forward(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_pass(input, cache, seq_len, start_pos, false)
    }

    fn forward_prefill(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_pass(input, cache, seq_len, start_pos, true)
    }

    fn supports_decode_batch(&self) -> bool {
        self.is_head
            && self.is_tail
            && self
                .blocks
                .iter()
                .all(|block| block.attn.quant_config.is_none())
    }

    fn decode_batch_requires_preallocated_attention(&self) -> bool {
        true
    }

    fn forward_decode_batch(
        &self,
        tokens: &[u32],
        caches: &mut [&mut [Option<LayerCache>]],
        batch_cache: &mut [Option<LayerCache>],
        positions: &[usize],
    ) -> Result<Vec<Tensor>> {
        anyhow::ensure!(
            self.supports_decode_batch(),
            "Llama decode batching unavailable"
        );
        anyhow::ensure!(!tokens.is_empty(), "decode batch cannot be empty");
        anyhow::ensure!(tokens.len() == caches.len(), "decode batch lane mismatch");
        anyhow::ensure!(
            tokens.len() == positions.len(),
            "decode batch position mismatch"
        );
        anyhow::ensure!(
            batch_cache.len() == self.blocks.len(),
            "persistent decode cache layer mismatch"
        );
        if !positions.iter().all(|position| *position == positions[0]) {
            if self.device.is_metal() {
                return self.forward_packed_ragged_decode_batch(
                    tokens,
                    caches,
                    batch_cache,
                    positions,
                );
            }
            return self.forward_ragged_decode_batch(tokens, caches, positions);
        }
        let start_pos = positions[0];
        let batch = tokens.len();
        let embedding = self.embedding.as_ref().unwrap();
        let token_tensor = Tensor::new(tokens, &self.device)?;
        let mut hidden_states =
            embedding
                .index_select(&token_tensor, 0)?
                .reshape((batch, 1, self.hidden_dim))?;

        let mut next_batch_cache = Vec::with_capacity(self.blocks.len());
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let mut batched_cache = match &batch_cache[layer_idx] {
                Some(cache) => super::clone_decode_layer_cache(cache)?,
                None => super::pack_decode_layer_cache(caches, layer_idx)?,
            };
            hidden_states = block.forward(&hidden_states, &mut batched_cache, 1, start_pos)?;
            next_batch_cache.push(batched_cache);
        }

        let normed = self.output_norm.as_ref().unwrap().forward(&hidden_states)?;
        let last_hidden = normed.squeeze(1)?;
        let logits = lm_head_forward(self.lm_head.as_ref().unwrap(), &last_hidden)?;
        let outputs = (0..batch)
            .map(|lane| logits.narrow(0, lane, 1)?.squeeze(0).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;

        for (layer_idx, cache) in next_batch_cache.iter().enumerate() {
            super::publish_decode_layer_cache_views(caches, layer_idx, cache)?;
        }
        for (slot, cache) in batch_cache.iter_mut().zip(next_batch_cache) {
            *slot = cache.map(LayerCache::Attention);
        }
        Ok(outputs)
    }

    fn family(&self) -> &str {
        "llama"
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn layer_count(&self) -> usize {
        self.num_layers
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// Build a single transformer block from GGUF tensors.
struct BlockBuildContext<'a> {
    tensors: &'a std::collections::HashMap<String, Arc<QTensor>>,
    metadata: &'a crate::gguf_loader::GgufMetadata,
    device: &'a Device,
    rope_cos: &'a Tensor,
    rope_sin: &'a Tensor,
    rope_is_neox: bool,
    quant_config: Option<Arc<TurboQuantConfig>>,
}

fn validate_attention_geometry(stage: &StageTensors) -> Result<()> {
    stage.range.validate()?;
    let md = &stage.metadata;
    anyhow::ensure!(md.hidden_dim > 0, "embedding length must be positive");
    anyhow::ensure!(
        md.rope_scaling_factor.is_finite() && md.rope_scaling_factor > 0.0,
        "RoPE scaling factor must be positive and finite"
    );
    // GGUF frequency tensors already encode the per-dimension scaling (for
    // example, Llama 3.1). This implementation divides by that tensor once.
    // Metadata-only linear/YaRN scaling needs separate math; do not silently
    // run such models with unscaled positions.
    anyhow::ensure!(
        md.rope_scaling_factor == 1.0 || stage.tensors.contains_key("rope_freqs.weight"),
        "unsupported scaled RoPE: {} rope.scaling.factor={} requires rope_freqs.weight; metadata-only scaling is not implemented",
        md.architecture,
        md.rope_scaling_factor
    );
    anyhow::ensure!(
        md.sliding_window_pattern.is_empty()
            || (md.sliding_window_pattern.len() == md.num_layers && md.sliding_window > 0),
        "sliding-window pattern must cover every layer and specify a positive window"
    );
    anyhow::ensure!(
        md.head_dim > 0 && md.head_dim.is_multiple_of(2),
        "{} attention.key_length must be positive and even for RoPE",
        md.architecture
    );
    anyhow::ensure!(
        md.num_kv_heads > 0
            && md.num_attention_heads > 0
            && md.num_attention_heads.is_multiple_of(md.num_kv_heads),
        "{} attention head count must be a positive multiple of the KV head count",
        md.architecture
    );
    let q_width = md
        .num_attention_heads
        .checked_mul(md.head_dim)
        .context("attention query projection width overflow")?;
    let kv_width = md
        .num_kv_heads
        .checked_mul(md.head_dim)
        .context("attention KV projection width overflow")?;
    if let Some(frequencies) = stage.tensors.get("rope_freqs.weight") {
        anyhow::ensure!(
            frequencies.shape().elem_count() == md.head_dim / 2,
            "rope_freqs.weight element count must match half of attention.key_length ({})",
            md.head_dim / 2
        );
    }
    for layer in stage.range.start..stage.range.end {
        for (suffix, expected) in [
            ("attn_q.weight", [q_width, md.hidden_dim]),
            ("attn_k.weight", [kv_width, md.hidden_dim]),
            ("attn_v.weight", [kv_width, md.hidden_dim]),
            ("attn_output.weight", [md.hidden_dim, q_width]),
        ] {
            let name = format!("blk.{layer}.{suffix}");
            let tensor = stage
                .tensors
                .get(&name)
                .with_context(|| format!("missing tensor: {name}"))?;
            anyhow::ensure!(
                tensor.shape().dims() == expected,
                "{name} shape {:?} does not match attention metadata {expected:?}",
                tensor.shape().dims()
            );
        }
    }
    Ok(())
}

fn build_block(
    context: &BlockBuildContext<'_>,
    prefix: &str,
    layer_idx: usize,
) -> Result<TransformerBlock> {
    let BlockBuildContext {
        tensors,
        metadata: md,
        device,
        rope_cos,
        rope_sin,
        rope_is_neox,
        quant_config,
    } = context;
    let get = |name: &str| -> Result<Arc<QTensor>> {
        tensors
            .get(&format!("{prefix}.{name}"))
            .cloned()
            .with_context(|| format!("missing tensor: {prefix}.{name}"))
    };

    let try_get_bias = |name: &str| -> Option<Tensor> {
        tensors
            .get(&format!("{prefix}.{name}"))
            .and_then(|qt| qt.as_ref().dequantize(device).ok())
    };

    let attn_norm = RmsNorm {
        weight: get("attn_norm.weight")?.as_ref().dequantize(device)?,
        eps: md.rms_norm_eps,
    };

    let attn = Attention {
        q_proj: qtensor_to_qmatmul(get("attn_q.weight")?)?,
        k_proj: qtensor_to_qmatmul(get("attn_k.weight")?)?,
        v_proj: qtensor_to_qmatmul(get("attn_v.weight")?)?,
        o_proj: qtensor_to_qmatmul(get("attn_output.weight")?)?,
        q_bias: try_get_bias("attn_q.bias"),
        k_bias: try_get_bias("attn_k.bias"),
        v_bias: try_get_bias("attn_v.bias"),
        num_heads: md.num_attention_heads,
        num_kv_heads: md.num_kv_heads,
        head_dim: md.head_dim,
        sliding_window: if md.sliding_window_pattern.get(layer_idx) == Some(&false) {
            0
        } else {
            md.sliding_window
        },
        rope_cos: (*rope_cos).clone(),
        rope_sin: (*rope_sin).clone(),
        rope_is_neox: *rope_is_neox,
        quant_config: quant_config.clone(),
    };

    let ffn_norm = RmsNorm {
        weight: get("ffn_norm.weight")?.as_ref().dequantize(device)?,
        eps: md.rms_norm_eps,
    };

    let ffn = Mlp {
        canonical_cpu_projections: attn.sliding_window == 0,
        gate_proj: qtensor_to_qmatmul(get("ffn_gate.weight")?)?,
        up_proj: qtensor_to_qmatmul(get("ffn_up.weight")?)?,
        down_proj: qtensor_to_qmatmul(get("ffn_down.weight")?)?,
    };

    Ok(TransformerBlock {
        attn_norm,
        attn,
        ffn_norm,
        ffn,
    })
}

/// Precompute RoPE cos/sin tables for the entire context length.
/// Returns (cos, sin) each of shape [context_length, head_dim/2].
fn precompute_rope_cos_sin(
    head_dim: usize,
    theta: f64,
    context_length: usize,
    device: &Device,
    freq_factors: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;

    let positions = Tensor::arange(0u32, context_length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?;

    let mut angles = positions.matmul(&inv_freq.reshape((1, half_dim))?)?;
    if let Some(freq_factors) = freq_factors {
        let freq_factors = freq_factors.to_dtype(DType::F32)?.reshape((1, half_dim))?;
        angles = angles.broadcast_div(&freq_factors)?;
    }
    let cos = angles.cos()?;
    let sin = angles.sin()?;
    Ok((cos, sin))
}

/// Apply Rotary Position Embedding using precomputed cos/sin tables.
///
/// `cos` and `sin` are [context_length, head_dim/2] — we narrow to the needed positions.
fn apply_rotary_emb(
    x: &Tensor,
    start_pos: usize,
    cos: &Tensor,
    sin: &Tensor,
    rope_is_neox: bool,
) -> Result<Tensor> {
    let (_, _, seq_len, _) = x.dims4()?;
    let cos = cos.narrow(0, start_pos, seq_len)?;
    let sin = sin.narrow(0, start_pos, seq_len)?;
    let x = x.contiguous()?;
    if rope_is_neox {
        candle_nn::rotary_emb::rope(&x, &cos, &sin).map_err(Into::into)
    } else {
        candle_nn::rotary_emb::rope_i(&x, &cos, &sin).map_err(Into::into)
    }
}

/// Repeat KV heads for Grouped-Query Attention.
fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, num_kv_heads, seq_len, head_dim) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, num_kv_heads, n_rep, seq_len, head_dim))?
        .reshape((b, num_kv_heads * n_rep, seq_len, head_dim))?
        .contiguous()
        .map_err(Into::into)
}

/// Softmax on the attention dimension.
fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::softmax_last_dim(x).map_err(Into::into)
}

/// Create a causal attention mask, with an optional left window.
fn create_attention_mask(
    seq_len: usize,
    total_seq_len: usize,
    window: usize,
    device: &Device,
) -> Result<Tensor> {
    let mask = if window > 0 {
        super::sliding_window_mask_values(seq_len, total_seq_len, window)
    } else {
        super::causal_mask_values(seq_len, total_seq_len)
    }
    .context("attention query length must be positive and fit the KV sequence")?;
    Tensor::from_vec(mask, (1, 1, seq_len, total_seq_len), device).map_err(Into::into)
}

#[cfg(all(test, target_os = "macos"))]
fn create_causal_mask(seq_len: usize, total_seq_len: usize, device: &Device) -> Result<Tensor> {
    create_attention_mask(seq_len, total_seq_len, 0, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonwindowed_cpu_mlp_is_independent_of_prompt_chunk_size() -> Result<()> {
        let weight = |out, width, salt| -> Result<QMatMul> {
            Ok(QMatMul::Tensor(Tensor::from_vec(
                (0..out * width)
                    .map(|i| ((i * 73 + salt) % 101) as f32 / 101.0 - 0.5)
                    .collect::<Vec<_>>(),
                (out, width),
                &Device::Cpu,
            )?))
        };
        let mlp = Mlp {
            canonical_cpu_projections: true,
            gate_proj: weight(96, 128, 1)?,
            up_proj: weight(96, 128, 3)?,
            down_proj: weight(128, 96, 7)?,
        };
        let input = Tensor::from_vec(
            (0..37 * 128)
                .map(|i| (i % 37) as f32 / 37.0)
                .collect::<Vec<_>>(),
            (1, 37, 128),
            &Device::Cpu,
        )?;
        let baseline = mlp.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
        for chunk in [1, 5, 7, 32, 37] {
            let parts = (0..37)
                .step_by(chunk)
                .map(|start| mlp.forward(&input.narrow(1, start, chunk.min(37 - start))?))
                .collect::<Result<Vec<_>>>()?;
            let actual = Tensor::cat(&parts, 1)?.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(actual.len(), baseline.len());
            assert!(
                actual
                    .iter()
                    .zip(&baseline)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
        // The disabled policy keeps the exact prior projection, including M1.
        let old = Mlp {
            canonical_cpu_projections: false,
            ..mlp
        };
        let row = input.narrow(1, 0, 1)?;
        let expected = old
            .gate_proj
            .forward(&row)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let actual = old
            .project(&old.gate_proj, &row)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(
            actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        Ok(())
    }

    fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(actual.dtype(), expected.dtype());
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
                "element {index}: {actual} != {expected}"
            );
        }
        Ok(())
    }

    fn check_prompt_tail(device: &Device) -> Result<()> {
        use candle_core::quantized::GgmlDType;

        for dtype in [GgmlDType::Q4K, GgmlDType::Q8_0] {
            for rope_is_neox in [false, true] {
                let projection = |rows: usize, cols: usize, phase: f32| -> Result<QMatMul> {
                    let values = (0..rows * cols)
                        .map(|i| (i as f32 * 0.017 + phase).sin() * 0.05)
                        .collect::<Vec<_>>();
                    let weights = Tensor::from_vec(values, (rows, cols), &Device::Cpu)?;
                    QMatMul::from_arc(Arc::new(QTensor::quantize_onto(&weights, dtype, device)?))
                        .map_err(Into::into)
                };
                let norm = || -> Result<RmsNorm> {
                    Ok(RmsNorm {
                        weight: Tensor::ones(256, DType::F32, device)?,
                        eps: 1e-5,
                    })
                };
                let (rope_cos, rope_sin) = precompute_rope_cos_sin(64, 10_000., 32, device, None)?;
                let stage = LlamaStage {
                    embedding: None,
                    blocks: vec![TransformerBlock {
                        attn_norm: norm()?,
                        attn: Attention {
                            q_proj: projection(256, 256, 0.1)?,
                            k_proj: projection(128, 256, 0.2)?,
                            v_proj: projection(128, 256, 0.3)?,
                            o_proj: projection(256, 256, 0.4)?,
                            q_bias: Some(Tensor::full(0.01f32, 256, device)?),
                            k_bias: Some(Tensor::full(0.02f32, 128, device)?),
                            v_bias: Some(Tensor::full(0.03f32, 128, device)?),
                            num_heads: 4,
                            num_kv_heads: 2,
                            head_dim: 64,
                            sliding_window: 0,
                            rope_cos,
                            rope_sin,
                            rope_is_neox,
                            quant_config: None,
                        },
                        ffn_norm: norm()?,
                        ffn: Mlp {
                            canonical_cpu_projections: false,
                            gate_proj: projection(512, 256, 0.5)?,
                            up_proj: projection(512, 256, 0.6)?,
                            down_proj: projection(256, 512, 0.7)?,
                        },
                    }],
                    output_norm: None,
                    lm_head: None,
                    is_head: false,
                    is_tail: false,
                    hidden_dim: 256,
                    num_layers: 1,
                    vocab_size: 0,
                    device: device.clone(),
                };
                let input = Tensor::from_vec(
                    (0..8 * 256)
                        .map(|i| (i as f32 * 0.23).cos())
                        .collect::<Vec<_>>(),
                    (8, 256),
                    device,
                )?;
                let activations = |out| match out {
                    StageOutput::Activations(tensor) => tensor,
                    _ => panic!("expected middle-stage activations"),
                };
                for len in [1, 2, 7] {
                    let mut full = vec![None];
                    let mut split = vec![None];
                    let expected = activations(stage.forward_prefill(
                        StageInput::activations(input.narrow(0, 0, len)?),
                        &mut full,
                        len,
                        0,
                    )?);
                    if len > 1 {
                        stage.forward_prefill(
                            StageInput::activations(input.narrow(0, 0, len - 1)?),
                            &mut split,
                            len - 1,
                            0,
                        )?;
                    }
                    let actual = activations(stage.forward_prefill(
                        StageInput::activations(input.narrow(0, len - 1, 1)?),
                        &mut split,
                        1,
                        len - 1,
                    )?);
                    assert_close(&actual, &expected.narrow(0, len - 1, 1)?, 2e-5)?;
                    // Only real prompt rows enter KV, including an empty-cache tail.
                    for cache in [&full, &split] {
                        let Some(LayerCache::Attention(KvCache::PreallocatedF16(kv))) = &cache[0]
                        else {
                            panic!("expected dense cache");
                        };
                        assert_eq!(kv.len, len);
                    }
                    let next = || StageInput::activations(input.narrow(0, len, 1).unwrap());
                    let expected = activations(stage.forward(next(), &mut full, 1, len)?);
                    let actual = activations(stage.forward(next(), &mut split, 1, len)?);
                    assert_close(&actual, &expected, 2e-5)?;

                    if device.is_cpu() {
                        let mut decode = vec![None];
                        let mut prefill = vec![None];
                        let expected = activations(stage.forward(next(), &mut decode, 1, 0)?);
                        let actual =
                            activations(stage.forward_prefill(next(), &mut prefill, 1, 0)?);
                        assert_close(&actual, &expected, 0.)?;
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn prompt_tail_preserves_cpu_forward_and_real_cache_rows() -> Result<()> {
        check_prompt_tail(&Device::Cpu)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_prompt_tail_preserves_real_cache_rows_and_following_decode() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        check_prompt_tail(&device)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_prompt_tail_attention_matches_full_prompt_last_row() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for len in [1, 2, 7, 63, 207] {
            let q = Tensor::from_vec(
                (0..4 * len * 128)
                    .map(|i| ((i * 13 % 257) as f32 - 128.) / 64.)
                    .collect::<Vec<_>>(),
                (1, 4, len, 128),
                &device,
            )?;
            let k = Tensor::from_vec(
                (0..2 * len * 128)
                    .map(|i| ((i * 11 % 127) as f32 - 63.) / 128.)
                    .collect::<Vec<_>>(),
                (1, 2, len, 128),
                &device,
            )?
            .to_dtype(DType::F16)?;
            let v = (&k * 0.7)?;
            let scale = 1. / 128f32.sqrt();
            let expected = super::super::try_metal_f16_sdpa_native(&q, &k, &v, len > 1, scale)?
                .context("test attention unavailable")?
                .narrow(2, len - 1, 1)?
                .to_dtype(DType::F32)?;
            let actual =
                prompt_tail_attention(&q.narrow(2, len - 1, 1)?.contiguous()?, &k, &v, scale)?
                    .context("test prompt attention unavailable")?
                    .to_dtype(DType::F32)?;
            assert_close(&actual, &expected, 0.)?;
            assert_eq!(k.dim(2)?, len);
            assert_eq!(v.dim(2)?, len);
        }
        let q = Tensor::zeros((1, 2, 1, 128), DType::F32, &device)?;
        let kv = Tensor::zeros((1, 1, 2, 128), DType::F32, &device)?;
        // Compressed caches expose f32 values and retain the existing fallback.
        assert!(prompt_tail_attention(&q, &kv, &kv, 0.1)?.is_none());
        Ok(())
    }

    fn check_silu_mul(device: &Device) -> Result<()> {
        for dtype in [DType::F32, DType::F16] {
            for (batch, tokens, width) in [(1, 1, 513), (1, 7, 512), (2, 3, 257)] {
                let elements = batch * tokens * width;
                // Separate nonzero offsets exercise the two input bindings.
                let gate = Tensor::from_vec(
                    (0..elements + 7)
                        .map(|i| ((i * 17 % 251) as f32 - 125.0) / 3.1)
                        .collect::<Vec<_>>(),
                    elements + 7,
                    device,
                )?
                .to_dtype(dtype)?
                .narrow(0, 7, elements)?
                .reshape((batch, tokens, width))?;
                let up = Tensor::from_vec(
                    (0..elements + 11)
                        .map(|i| ((i * 31 % 251) as f32 - 125.0) / 29.0)
                        .collect::<Vec<_>>(),
                    elements + 11,
                    device,
                )?
                .to_dtype(dtype)?
                .narrow(0, 11, elements)?
                .reshape((batch, tokens, width))?;
                for (gate, up) in [
                    (gate.clone(), up.clone()),
                    (gate.transpose(1, 2)?, up.transpose(1, 2)?),
                    (gate.transpose(1, 2)?, up.transpose(1, 2)?.contiguous()?),
                ] {
                    let expected = (gate.silu()? * &up)?;
                    assert_close(&silu_mul(&gate, &up)?, &expected, 2e-6)?;
                }
            }
        }
        let gate = Tensor::zeros((1, 8), DType::F32, device)?;
        let wrong_shape = Tensor::zeros((2, 8), DType::F32, device)?;
        let empty = Tensor::zeros((0, 8), DType::F32, device)?;
        assert!(silu_mul(&gate, &wrong_shape).is_err());
        assert!(silu_mul(&gate, &gate.to_dtype(DType::F16)?).is_err());
        assert_close(&silu_mul(&empty, &empty)?, &empty, 0.)?;
        Ok(())
    }

    fn check_mlp(device: &Device) -> Result<()> {
        use candle_core::quantized::GgmlDType;

        for dtype in [GgmlDType::F32, GgmlDType::Q4K, GgmlDType::Q8_0] {
            let projection = |rows: usize, cols: usize, phase: f32| -> Result<QMatMul> {
                let values = (0..rows * cols)
                    .map(|i| (i as f32 * 0.017 + phase).sin() * 0.05)
                    .collect::<Vec<_>>();
                let weights = Tensor::from_vec(values, (rows, cols), &Device::Cpu)?;
                let packed = QTensor::quantize_onto(&weights, dtype, device)?;
                QMatMul::from_arc(Arc::new(packed)).map_err(Into::into)
            };
            let mlp = Mlp {
                canonical_cpu_projections: false,
                gate_proj: projection(512, 256, 0.1)?,
                up_proj: projection(512, 256, 0.7)?,
                down_proj: projection(256, 512, 1.3)?,
            };
            for (batch, tokens) in [(1, 1), (1, 7), (2, 1)] {
                let x = Tensor::from_vec(
                    (0..batch * tokens * 256)
                        .map(|i| (i as f32 * 0.023).cos() * 0.5)
                        .collect::<Vec<_>>(),
                    (batch, tokens, 256),
                    device,
                )?;
                let gate = mlp.gate_proj.forward(&x)?;
                let up = mlp.up_proj.forward(&x)?;
                let expected = mlp.down_proj.forward(&(gate.silu()? * up)?)?;
                assert_close(&mlp.forward(&x)?, &expected, 1e-5)?;
            }
        }
        Ok(())
    }

    #[test]
    fn silu_mul_preserves_cpu_dtypes_and_layouts() -> Result<()> {
        check_silu_mul(&Device::Cpu)
    }

    #[test]
    fn mlp_matches_unfused_cpu() -> Result<()> {
        check_mlp(&Device::Cpu)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn silu_mul_preserves_metal_dtypes_and_layouts() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        check_silu_mul(&device)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mlp_matches_unfused_metal() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        check_mlp(&device)
    }

    fn preallocated_cache(fill: f32, seq_len: usize) -> Option<LayerCache> {
        let k = Tensor::full(fill, (1, 2, seq_len, 4), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let v = k.clone();
        Some(LayerCache::Attention(KvCache::PreallocatedF16(
            crate::architectures::PreallocatedKvCache::from_f16(k, v).unwrap(),
        )))
    }

    #[test]
    fn decode_batch_pack_update_and_unpack_preserves_lanes() -> Result<()> {
        let mut first = vec![preallocated_cache(1.0, 3)];
        let mut second = vec![preallocated_cache(2.0, 3)];
        let mut caches = vec![first.as_mut_slice(), second.as_mut_slice()];
        let mut packed = crate::architectures::pack_decode_layer_cache(&caches, 0)?;
        let incoming_k = Tensor::cat(
            &[
                &Tensor::full(3f32, (1, 2, 1, 4), &Device::Cpu)?,
                &Tensor::full(4f32, (1, 2, 1, 4), &Device::Cpu)?,
            ],
            0,
        )?;
        let incoming_v = incoming_k.clone();
        super::super::kv_cache_update(&mut packed, incoming_k, incoming_v, None)?;
        crate::architectures::publish_decode_layer_cache_views(&mut caches, 0, &packed)?;
        drop(caches);

        assert_eq!(
            crate::architectures::kv_cache_allocation_tokens(&first)?,
            (64, 128)
        );
        assert_eq!(
            crate::architectures::kv_cache_allocation_tokens(&second)?,
            (64, 128)
        );

        for (cache, old_value, new_value) in [(first, 1f32, 3f32), (second, 2f32, 4f32)] {
            let Some(LayerCache::Attention(KvCache::PreallocatedF16(dense))) = &cache[0] else {
                panic!("expected preallocated f16 attention cache")
            };
            let (k, _) = dense.active_f16()?;
            assert_eq!(k.dims(), &[1, 2, 4, 4]);
            let values = k.to_dtype(DType::F32)?;
            for head in 0..2 {
                for position in 0..3 {
                    assert!(
                        values
                            .i((0, head, position, ..))?
                            .to_vec1::<f32>()?
                            .iter()
                            .all(|value| *value == old_value)
                    );
                }
                assert!(
                    values
                        .i((0, head, 3, ..))?
                        .to_vec1::<f32>()?
                        .iter()
                        .all(|value| *value == new_value)
                );
            }
        }
        Ok(())
    }

    #[test]
    fn failed_decode_batch_pack_preserves_session_caches() {
        let mut first = vec![preallocated_cache(1.0, 3)];
        let mismatched = Tensor::zeros((1, 3, 3, 4), DType::F16, &Device::Cpu).unwrap();
        let mut second = vec![Some(LayerCache::Attention(KvCache::PreallocatedF16(
            crate::architectures::PreallocatedKvCache::from_f16(mismatched.clone(), mismatched)
                .unwrap(),
        )))];
        let caches = vec![first.as_mut_slice(), second.as_mut_slice()];

        assert!(crate::architectures::pack_decode_layer_cache(&caches, 0).is_err());
        assert!(first[0].is_some());
        assert!(second[0].is_some());
    }

    #[test]
    fn ragged_decode_pack_append_and_publish_preserves_logical_lengths() -> Result<()> {
        let mut short = vec![preallocated_cache(1.0, 2)];
        let mut long = vec![preallocated_cache(2.0, 4)];
        let mut caches = vec![short.as_mut_slice(), long.as_mut_slice()];
        let mut packed = crate::architectures::pack_ragged_decode_layer_cache(&caches, 0, &[2, 4])?;
        let Some(KvCache::PreallocatedF16(dense)) = packed.as_ref() else {
            panic!("expected packed ragged cache")
        };
        assert_eq!(dense.len, 4);
        assert_eq!(dense.capacity, 64);

        let incoming = Tensor::cat(
            &[
                &Tensor::full(3f32, (1, 2, 1, 4), &Device::Cpu)?,
                &Tensor::full(4f32, (1, 2, 1, 4), &Device::Cpu)?,
            ],
            0,
        )?;
        let Some(KvCache::PreallocatedF16(dense)) = packed.as_mut() else {
            unreachable!()
        };
        dense.append_ragged(&incoming, &incoming, &[2, 4])?;
        crate::architectures::publish_ragged_decode_layer_cache_views(
            &mut caches,
            0,
            &packed,
            &[3, 5],
        )?;
        drop(caches);

        for (cache, expected_len, position, value) in [(short, 3, 2, 3f32), (long, 5, 4, 4f32)] {
            let Some(LayerCache::Attention(KvCache::PreallocatedF16(dense))) = &cache[0] else {
                panic!("expected published ragged cache")
            };
            assert_eq!(dense.len, expected_len);
            let (k, _) = dense.active_f16()?;
            assert!(
                k.i((0, 0, position, ..))?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|actual| *actual == value)
            );
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn zero_copy_f16_metal_sdpa_matches_portable_causal_gqa() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let head_dim = 32;
        let scale = 1.0 / (head_dim as f64).sqrt();

        for (q_seq, kv_seq) in [(7, 7), (3, 7)] {
            let q_len = 4 * q_seq * head_dim;
            let kv_len = 2 * kv_seq * head_dim;
            let q_values = (0..q_len)
                .map(|i| ((i as f32 * 0.013).sin() * 0.25) + 0.01)
                .collect::<Vec<_>>();
            let k_values = (0..kv_len)
                .map(|i| ((i as f32 * 0.017).cos() * 0.20) - 0.02)
                .collect::<Vec<_>>();
            let v_values = (0..kv_len)
                .map(|i| ((i as f32 * 0.019).sin() * 0.30) + 0.03)
                .collect::<Vec<_>>();

            let q = Tensor::from_vec(q_values, (1, 4, q_seq, head_dim), &device)?;
            let k = Tensor::from_vec(k_values, (1, 2, kv_seq, head_dim), &device)?
                .to_dtype(DType::F16)?;
            let v = Tensor::from_vec(v_values, (1, 2, kv_seq, head_dim), &device)?
                .to_dtype(DType::F16)?;
            let dense = crate::architectures::PreallocatedKvCache::from_f16(k, v)?;
            let (k, v) = dense.active_f16()?;

            let fused = crate::architectures::try_metal_f16_sdpa(&q, &k, &v, true, scale as f32)?
                .expect("supported f16 Metal SDPA geometry");

            let repeated_k = repeat_kv(k.to_dtype(DType::F32)?, 2)?;
            let repeated_v = repeat_kv(v.to_dtype(DType::F32)?, 2)?;
            let weights = (q.matmul(&repeated_k.transpose(2, 3)?)? * scale)?;
            let mask = create_causal_mask(q_seq, kv_seq, &device)?;
            let portable = softmax_last_dim(&weights.broadcast_add(&mask)?)?.matmul(&repeated_v)?;

            let fused = fused.flatten_all()?.to_vec1::<f32>()?;
            let portable = portable.flatten_all()?.to_vec1::<f32>()?;
            let max_abs_diff = fused
                .iter()
                .zip(portable.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs_diff < 2e-3,
                "q_seq={q_seq} kv_seq={kv_seq} max_abs_diff={max_abs_diff}"
            );
        }

        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn fused_ragged_metal_rope_matches_lane_local_conventions() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let batch = 2;
        let q_heads = 4;
        let kv_heads = 2;
        let head_dim = 32;
        let positions = [3usize, 17usize];
        let q_len = batch * q_heads * head_dim;
        let k_len = batch * kv_heads * head_dim;
        let q = Tensor::from_vec(
            (0..q_len)
                .map(|i| (i as f32 * 0.013).sin())
                .collect::<Vec<_>>(),
            (batch, q_heads, 1, head_dim),
            &device,
        )?;
        let k = Tensor::from_vec(
            (0..k_len)
                .map(|i| (i as f32 * 0.017).cos())
                .collect::<Vec<_>>(),
            (batch, kv_heads, 1, head_dim),
            &device,
        )?;
        let (cos, sin) = precompute_rope_cos_sin(head_dim, 10_000.0, 32, &device, None)?;

        for neox in [false, true] {
            let (fused_q, fused_k) =
                crate::architectures::try_metal_ragged_rope(&q, &k, &cos, &sin, &positions, neox)?
                    .expect("supported ragged Metal RoPE geometry");
            let mut expected_q = Vec::with_capacity(batch);
            let mut expected_k = Vec::with_capacity(batch);
            for (lane, position) in positions.iter().copied().enumerate() {
                expected_q.push(apply_rotary_emb(
                    &q.narrow(0, lane, 1)?,
                    position,
                    &cos,
                    &sin,
                    neox,
                )?);
                expected_k.push(apply_rotary_emb(
                    &k.narrow(0, lane, 1)?,
                    position,
                    &cos,
                    &sin,
                    neox,
                )?);
            }
            let expected_q = Tensor::cat(&expected_q.iter().collect::<Vec<_>>(), 0)?;
            let expected_k = Tensor::cat(&expected_k.iter().collect::<Vec<_>>(), 0)?;
            for (name, fused, expected) in [("q", fused_q, expected_q), ("k", fused_k, expected_k)]
            {
                let fused = fused.flatten_all()?.to_vec1::<f32>()?;
                let expected = expected.flatten_all()?.to_vec1::<f32>()?;
                let max_abs_diff = fused
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_abs_diff < 1e-6,
                    "{name} neox={neox} max_abs_diff={max_abs_diff}"
                );
            }
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn ragged_metal_sdpa_masks_padding_and_matches_portable_gqa() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let batch = 2;
        let q_heads = 4;
        let kv_heads = 2;
        let head_dim = 32;
        let capacity = 1088;
        let valid_lengths = [3usize, 1025usize];
        let scale = 1.0 / (head_dim as f64).sqrt();

        let q_len = batch * q_heads * head_dim;
        let kv_len = batch * kv_heads * capacity * head_dim;
        let q = Tensor::from_vec(
            (0..q_len)
                .map(|i| (i as f32 * 0.013).sin() * 0.25)
                .collect::<Vec<_>>(),
            (batch, q_heads, 1, head_dim),
            &device,
        )?;
        let mut k_values = (0..kv_len)
            .map(|i| (i as f32 * 0.017).cos() * 0.20)
            .collect::<Vec<_>>();
        let mut v_values = (0..kv_len)
            .map(|i| (i as f32 * 0.019).sin() * 0.30)
            .collect::<Vec<_>>();
        for (lane, valid_len) in valid_lengths.iter().copied().enumerate() {
            for head in 0..kv_heads {
                for position in valid_len..capacity {
                    for dim in 0..head_dim {
                        let index =
                            ((lane * kv_heads + head) * capacity + position) * head_dim + dim;
                        k_values[index] = 50.0;
                        v_values[index] = 100.0;
                    }
                }
            }
        }
        let k = Tensor::from_vec(k_values, (batch, kv_heads, capacity, head_dim), &device)?
            .to_dtype(DType::F16)?;
        let v = Tensor::from_vec(v_values, (batch, kv_heads, capacity, head_dim), &device)?
            .to_dtype(DType::F16)?;

        let fused = crate::architectures::try_metal_f16_ragged_sdpa(
            &q,
            &k,
            &v,
            &valid_lengths,
            scale as f32,
        )?
        .expect("supported ragged Metal geometry");

        for (lane, valid_len) in valid_lengths.into_iter().enumerate() {
            let lane_q = q.narrow(0, lane, 1)?;
            let lane_k = repeat_kv(
                k.narrow(0, lane, 1)?
                    .narrow(2, 0, valid_len)?
                    .to_dtype(DType::F32)?,
                q_heads / kv_heads,
            )?;
            let lane_v = repeat_kv(
                v.narrow(0, lane, 1)?
                    .narrow(2, 0, valid_len)?
                    .to_dtype(DType::F32)?,
                q_heads / kv_heads,
            )?;
            let weights = (lane_q.matmul(&lane_k.transpose(2, 3)?)? * scale)?;
            let portable = softmax_last_dim(&weights)?.matmul(&lane_v)?;
            let fused_values = fused.narrow(0, lane, 1)?.flatten_all()?.to_vec1::<f32>()?;
            let portable_values = portable.flatten_all()?.to_vec1::<f32>()?;
            let max_abs_diff = fused_values
                .iter()
                .zip(portable_values)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs_diff < 5e-3,
                "lane={lane} valid_len={valid_len} max_abs_diff={max_abs_diff}"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "llama_sliding_window_tests.rs"]
mod sliding_window_tests;
