use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Most decoders precompute at least one `[context, head_dim/2]` RoPE pair.
/// Keep their accepted range large enough for the existing 1M-token models
/// while preventing forged metadata from driving unbounded allocations.
const MAX_SUPPORTED_CONTEXT_LENGTH: usize = 1_048_576;
// Scout declares 10M positions. Llama 4 caches at most 8,192 RoPE rows and
// computes later ranges on demand, so accepting its declared limit does not
// allocate a full-context table. Keep this exception bounded and model-specific.
const MAX_LLAMA4_CONTEXT_LENGTH: usize = 10_485_760;

use anyhow::{Context, Result};
use candle_core::Device;
use candle_core::quantized::{GgmlDType, gguf_file};

use crate::deepseek4_gguf::{self, RawGgufContent, RawTensor};
pub use crate::deepseek4_gguf::{DEEPSEEK41_EXPERT_CACHE_BYTES, RawGgufSource};
use crate::gguf_limits::{
    MAX_GGUF_AGGREGATE_HEADER_BYTES, MAX_GGUF_SHARDS, MAX_GGUF_TENSOR_BYTES,
    validate_gguf_header_budget, validate_layer_count,
};

/// A tensor's byte span in one GGUF file. This describes storage, not whether
/// a sparse cache has downloaded or verified those bytes.
pub struct GgufTensorSpan {
    /// Tensor name from the file header.
    pub name: String,
    /// Dimensions in engine order.
    pub shape: Vec<usize>,
    /// GGML dtype name, including formats newer than Candle's reader.
    pub dtype: String,
    /// Inclusive byte start in this file.
    pub start: u64,
    /// Exclusive byte end in this file.
    pub end: u64,
}

/// Read validated tensor spans without loading any model weights.
pub fn read_gguf_tensor_spans(path: &Path) -> Result<Vec<GgufTensorSpan>> {
    let mut file = std::fs::File::open(path)?;
    let file_len = file.metadata()?.len();
    let content = RawGgufContent::read(&mut file)?;
    let mut spans = Vec::with_capacity(content.tensor_infos.len());
    for (name, info) in content.tensor_infos {
        let start = content
            .tensor_data_offset
            .checked_add(info.offset)
            .context("tensor start overflow")?;
        let end = start
            .checked_add(info.byte_len()? as u64)
            .context("tensor end overflow")?;
        anyhow::ensure!(
            end <= file_len,
            "GGUF tensor {name} extends past the end of the file"
        );
        spans.push(GgufTensorSpan {
            name,
            shape: info.shape,
            dtype: format!("{:?}", info.dtype),
            start,
            end,
        });
    }
    spans.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(spans)
}

/// Defines a contiguous range of transformer layers to load.
#[derive(Debug, Clone)]
pub struct LayerRange {
    /// First layer index (inclusive).
    pub start: usize,
    /// Last layer index (exclusive).
    pub end: usize,
    /// Total number of layers in the full model.
    pub total_layers: usize,
}

impl LayerRange {
    /// Construct a half-open layer range using the legacy infallible API.
    ///
    /// This constructor preserves source compatibility and defers validation
    /// until the range is passed to a loader or profiler. Use [`Self::try_new`]
    /// when accepting fallible or untrusted input.
    pub fn new(start: usize, end: usize, total_layers: usize) -> Self {
        Self {
            start,
            end,
            total_layers,
        }
    }

    /// Construct a half-open layer range from fallible or untrusted input.
    pub fn try_new(start: usize, end: usize, total_layers: usize) -> Result<Self> {
        let range = Self {
            start,
            end,
            total_layers,
        };
        range.validate()?;
        Ok(range)
    }

    /// Solo mode using the legacy infallible API.
    ///
    /// Validation is deferred to the loader; use [`Self::try_solo`] for
    /// untrusted input.
    pub fn solo(total_layers: usize) -> Self {
        Self {
            start: 0,
            end: total_layers,
            total_layers,
        }
    }

    /// Construct a full-model range from fallible or untrusted input.
    pub fn try_solo(total_layers: usize) -> Result<Self> {
        Self::try_new(0, total_layers, total_layers)
    }

    /// Validate this range, including values supplied through its public
    /// fields by compatibility callers.
    pub fn validate(&self) -> Result<()> {
        validate_layer_count(self.total_layers, "GGUF layer range total")?;
        anyhow::ensure!(
            self.start < self.end && self.end <= self.total_layers,
            "GGUF layer range {}..{} is outside 0..{}",
            self.start,
            self.end,
            self.total_layers
        );
        Ok(())
    }

    /// Number of transformer layers in this stage.
    ///
    /// Returns zero for a forged or otherwise invalid reversed range. Load and
    /// profiling entry points call [`Self::validate`] and reject such values.
    pub fn layer_count(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Whether this stage includes the embedding layer (first stage).
    pub fn is_head(&self) -> bool {
        self.start == 0
    }

    /// Whether this stage includes the output norm and lm_head (last stage).
    pub fn is_tail(&self) -> bool {
        self.end == self.total_layers
    }

    /// Whether this is the only stage (solo mode).
    pub fn is_solo(&self) -> bool {
        self.is_head() && self.is_tail()
    }

    /// Check whether a given layer index falls within this range.
    pub fn contains(&self, layer: usize) -> bool {
        layer >= self.start && layer < self.end
    }
}

/// Loaded tensors for a pipeline stage, plus model metadata.
pub struct StageTensors {
    /// Layer range represented by these tensors.
    pub range: LayerRange,
    /// Model metadata parsed from the GGUF header.
    pub metadata: GgufMetadata,
    /// Quantized tensors keyed by their GGUF tensor name (Arc for shared ownership).
    pub tensors: HashMap<String, Arc<candle_core::quantized::QTensor>>,
    /// DeepSeek V4/V4.1 tensors kept in their original GGUF byte layout.
    /// V4.1 excludes Engram tables and routed expert banks; use `raw_source`
    /// for bounded reads, and enforce `DEEPSEEK41_EXPERT_CACHE_BYTES` in the graph.
    pub raw_tensors: HashMap<String, Arc<RawTensor>>,
    /// Opened lazy disk tensors and complete typed metadata (V4.1 only).
    /// Full solo Metal loading only; this is not architecture/graph dispatch.
    pub raw_source: Option<Arc<RawGgufSource>>,
    /// Device on which the loaded tensors are materialized.
    pub device: Device,
}

/// Device-specific tensor memory needed to load a model stage.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StageMemoryProfile {
    /// Model identifier associated with this profile.
    pub model_id: String,
    /// Number of executable transformer layers in the model.
    pub total_layers: usize,
    /// Resident device bytes for each layer, including eager source pools
    /// that overlap the constructed weights. CPU stages use this same
    /// component for their resident tensor storage.
    pub layer_gpu_bytes: Vec<u64>,
    /// Peak accelerator bytes temporarily used by one layer's construction or
    /// bounded expert/attention workspace. This is a range peak, not a sum over layers.
    pub layer_gpu_transient_bytes: Vec<u64>,
    /// CPU tensor bytes retained by each layer, plus eager CPU source pools
    /// that overlap accelerator construction.
    /// Streaming architectures put temporary source QTensor bytes in
    /// `layer_cpu_transient_bytes` instead, so a range consumer does not sum
    /// sources which have already been released.
    pub layer_cpu_bytes: Vec<u64>,
    /// Peak extra CPU construction or bounded expert workspace for a stage. This is the
    /// largest source/upload peak needed to materialize one layer. For the
    /// GLM5/MiniMax streaming paths this includes all CPU QTensor sources for that
    /// layer plus its largest owned upload buffer.
    pub layer_cpu_transient_bytes: Vec<u64>,
    /// Accelerator bytes retained by the embedding tensor.
    pub embedding_gpu_bytes: u64,
    /// CPU bytes retained by the embedding tensor.
    pub embedding_cpu_bytes: u64,
    /// Accelerator bytes for the embedding when this is the head stage.
    pub head_embedding_gpu_bytes: Option<u64>,
    /// Accelerator bytes for the embedding when this is the tail stage.
    pub tail_embedding_gpu_bytes: Option<u64>,
    /// CPU bytes for the embedding when this is the head stage.
    pub head_embedding_cpu_bytes: Option<u64>,
    /// CPU bytes for the embedding when this is the tail stage.
    pub tail_embedding_cpu_bytes: Option<u64>,
    /// Peak temporary CPU bytes while constructing the head embedding.
    pub head_embedding_cpu_transient_bytes: Option<u64>,
    /// Peak temporary CPU bytes while constructing the tail embedding.
    pub tail_embedding_cpu_transient_bytes: Option<u64>,
    /// Accelerator bytes retained by the output projection.
    pub output_gpu_bytes: u64,
    /// Peak accelerator bytes temporarily duplicated while constructing the
    /// output tensor.
    pub output_gpu_transient_bytes: u64,
    /// CPU bytes retained by the output projection.
    pub output_cpu_bytes: u64,
    /// Peak temporary CPU bytes while constructing the output projection.
    pub output_cpu_transient_bytes: u64,
    /// Accelerator bytes retained by tensors common to all stages.
    pub common_gpu_bytes: u64,
    /// CPU bytes retained by tensors common to all stages.
    pub common_cpu_bytes: u64,
    /// Peak temporary CPU bytes while constructing common tensors.
    pub common_cpu_transient_bytes: u64,
}

impl StageMemoryProfile {
    /// Return `(peak_accelerator_bytes, retained_cpu_bytes,
    /// transient_cpu_bytes)` for one contiguous stage. The accelerator
    /// component includes the largest temporary source/destination overlap in
    /// the range; CPU construction is likewise charged as a range peak.
    pub fn stage_memory_components(
        &self,
        start: usize,
        end: usize,
        total_layers: usize,
    ) -> Option<(u64, u64, u64)> {
        if start >= end
            || end > total_layers
            || self.total_layers != total_layers
            || self.layer_gpu_bytes.len() != total_layers
            || self.layer_gpu_transient_bytes.len() != total_layers
            || self.layer_cpu_bytes.len() != total_layers
            || self.layer_cpu_transient_bytes.len() != total_layers
        {
            return None;
        }
        let layer_gpu_bytes = self.layer_gpu_bytes[start..end]
            .iter()
            .fold(0u64, |sum, bytes| sum.saturating_add(*bytes));
        let layer_gpu_transient_bytes = self.layer_gpu_transient_bytes[start..end]
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let layer_cpu_bytes = self.layer_cpu_bytes[start..end]
            .iter()
            .fold(0u64, |sum, bytes| sum.saturating_add(*bytes));
        let layer_cpu_transient_bytes = self.layer_cpu_transient_bytes[start..end]
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let boundary_gpu_bytes = self.boundary_gpu_bytes(start, end, total_layers);
        let boundary_cpu_bytes = self.boundary_cpu_bytes(start, end, total_layers);
        let output_gpu_bytes = if end == total_layers {
            self.output_gpu_bytes
        } else {
            0
        };
        let output_cpu_bytes = if end == total_layers {
            self.output_cpu_bytes
        } else {
            0
        };
        let output_gpu_transient_bytes = if end == total_layers {
            self.output_gpu_transient_bytes
        } else {
            0
        };
        let output_cpu_transient_bytes = if end == total_layers {
            self.output_cpu_transient_bytes
        } else {
            0
        };
        let transient_cpu_bytes = layer_cpu_transient_bytes
            .max(self.boundary_cpu_transient_bytes(start, end, total_layers))
            .max(output_cpu_transient_bytes)
            .max(self.common_cpu_transient_bytes);
        let accelerator_construction_bytes =
            layer_gpu_transient_bytes.max(output_gpu_transient_bytes);

        Some((
            layer_gpu_bytes
                .saturating_add(boundary_gpu_bytes)
                .saturating_add(output_gpu_bytes)
                .saturating_add(self.common_gpu_bytes)
                .saturating_add(accelerator_construction_bytes),
            layer_cpu_bytes
                .saturating_add(boundary_cpu_bytes)
                .saturating_add(output_cpu_bytes)
                .saturating_add(self.common_cpu_bytes),
            transient_cpu_bytes,
        ))
    }

    fn boundary_gpu_bytes(&self, start: usize, end: usize, total_layers: usize) -> u64 {
        if self.head_embedding_gpu_bytes.is_none() && self.tail_embedding_gpu_bytes.is_none() {
            return if start == 0 || end == total_layers {
                self.embedding_gpu_bytes
            } else {
                0
            };
        }
        let head = if start == 0 {
            self.head_embedding_gpu_bytes
                .unwrap_or(self.embedding_gpu_bytes)
        } else {
            0
        };
        let tail = if end == total_layers {
            self.tail_embedding_gpu_bytes
                .unwrap_or(self.embedding_gpu_bytes)
        } else {
            0
        };
        // A stage boundary is materialized at either end of the pipeline, not
        // both at once.  Keep this in sync with the protocol-side profile
        // consumer, which uses the larger realized boundary cost for a range.
        head.max(tail)
    }

    fn boundary_cpu_bytes(&self, start: usize, end: usize, total_layers: usize) -> u64 {
        if self.head_embedding_cpu_bytes.is_none() && self.tail_embedding_cpu_bytes.is_none() {
            return if start == 0 || end == total_layers {
                self.embedding_cpu_bytes
            } else {
                0
            };
        }
        let head = if start == 0 {
            self.head_embedding_cpu_bytes
                .unwrap_or(self.embedding_cpu_bytes)
        } else {
            0
        };
        let tail = if end == total_layers {
            self.tail_embedding_cpu_bytes
                .unwrap_or(self.embedding_cpu_bytes)
        } else {
            0
        };
        head.saturating_add(tail)
    }

    fn boundary_cpu_transient_bytes(&self, start: usize, end: usize, total_layers: usize) -> u64 {
        let head = if start == 0 {
            self.head_embedding_cpu_transient_bytes.unwrap_or(0)
        } else {
            0
        };
        let tail = if end == total_layers {
            self.tail_embedding_cpu_transient_bytes.unwrap_or(0)
        } else {
            0
        };
        head.saturating_add(tail)
    }
}

/// Extracted model metadata from the GGUF file header.
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    /// Model architecture name stored in the GGUF header.
    pub architecture: String,
    /// Number of executable transformer trunk layers.
    ///
    /// GLM5-Next GGUFs may advertise one or more physical NextN/MTP blocks
    /// after the ordinary trunk. Those blocks are not part of a pipeline
    /// layer range and are excluded from this count.
    pub num_layers: usize,
    /// Hidden width of the model's transformer states.
    pub hidden_dim: usize,
    /// Number of query attention heads per transformer layer.
    pub num_attention_heads: usize,
    /// Number of key/value attention heads per transformer layer.
    pub num_kv_heads: usize,
    /// Width of each attention head.
    pub head_dim: usize,
    /// Intermediate width of a dense feed-forward layer.
    pub intermediate_dim: usize,
    /// Number of entries in the model vocabulary.
    pub vocab_size: usize,
    /// Epsilon used by RMS normalization layers.
    pub rms_norm_eps: f64,
    /// Muse Spark's post-branch and expert normalization epsilon.
    pub post_norm_eps: f64,
    /// Muse Spark's pooled expert width (0 for other model families).
    pub moe_latent_size: usize,
    /// Scale applied to output logits before softcapping (1 for other families).
    pub logit_scale: f64,
    /// Base period used by rotary positional embeddings.
    pub rope_theta: f64,
    /// Maximum context length advertised by the model.
    pub context_length: usize,
    /// Number of experts (MoE models only, 0 for dense models).
    pub expert_count: usize,
    /// Number of active experts per token (top-k, MoE only).
    pub expert_used_count: usize,
    /// Per-expert FFN intermediate dimension (MoE only).
    pub expert_ffn_dim: usize,
    /// SSM state size / key_head_dim (DeltaNet models only).
    pub ssm_state_size: usize,
    /// SSM convolution kernel width (DeltaNet models only).
    pub ssm_conv_kernel: usize,
    /// SSM group count / num_k_heads (DeltaNet models only).
    pub ssm_group_count: usize,
    /// SSM time step rank / num_v_heads (DeltaNet models only).
    pub ssm_time_step_rank: usize,
    /// SSM inner size / conv_dim (DeltaNet models only).
    pub ssm_inner_size: usize,
    /// RoPE dimension count (for partial RoPE, 0 = full).
    pub rope_dimension_count: usize,
    /// Qwen multimodal RoPE frequency sections (temporal, height, width,
    /// optional trailing zero). Empty for architectures without M-RoPE.
    pub rope_dimension_sections: Vec<usize>,
    /// RoPE scaling factor (1.0 = disabled).
    pub rope_scaling_factor: f64,
    /// Original context length used by scaled RoPE.
    pub rope_scaling_original_context_length: usize,
    /// YaRN logarithmic attention multiplier stored by GGUF converters.
    pub rope_yarn_log_multiplier: f64,
    /// YaRN fast/slow frequency correction boundaries.
    pub rope_yarn_beta_fast: f64,
    /// YaRN slow frequency correction boundary.
    pub rope_yarn_beta_slow: f64,
    /// Sliding-window RoPE dimension count (Gemma 4 only, 0 = use rope_dimension_count).
    pub rope_dimension_count_swa: usize,
    /// Sliding-window attention head dimension (Gemma 4 only, 0 = use head_dim).
    pub head_dim_swa: usize,
    /// Sliding-window RoPE base (0 = use rope_theta).
    pub rope_theta_swa: f64,
    /// Sliding-window attention span (0 = full attention).
    /// Llama 4 uses aligned chunks rather than a rolling window.
    pub sliding_window: usize,
    /// Per-layer sliding-window flags (true = windowed/chunked attention).
    pub sliding_window_pattern: Vec<bool>,
    /// Final logit softcap (Gemma models; 0 = disabled).
    pub final_logit_softcapping: f64,
    /// Interval between full attention layers (e.g. 4 = every 4th layer is attention).
    pub full_attention_interval: usize,
    /// Per-layer KV head counts (empty = uniform). Used by nemotron_h to detect layer types.
    pub head_count_kv_arr: Vec<usize>,
    /// Per-layer FFN lengths (empty = uniform). Used by nemotron_h to detect layer types.
    pub feed_forward_length_arr: Vec<usize>,
    /// Attention scale override (0.0 = use default 1/sqrt(head_dim)).
    pub attention_scale: f64,
    /// Expert weights scale for MoE (nemotron_h: 2.5).
    pub expert_weights_scale: f64,
    /// Whether to normalize MoE expert weights to sum=1.
    pub expert_weights_norm: bool,
    /// Expert routing activation from GGUF (1 = softmax, 2 = sigmoid).
    pub expert_gating_func: usize,
    /// Number of shared experts (MoE models).
    pub shared_expert_count: usize,
    // ===== MLA (Multi-head Latent Attention) — DeepSeek V3/R1 =====
    /// KV compression rank (512 for DeepSeek V3). 0 = standard attention.
    pub kv_lora_rank: usize,
    /// Q compression rank (1536 for DeepSeek V3). 0 = no Q compression.
    pub q_lora_rank: usize,
    /// Non-RoPE Q/K head dim (128 for DeepSeek V3).
    pub qk_nope_head_dim: usize,
    /// RoPE Q/K head dim (64 for DeepSeek V3).
    pub qk_rope_head_dim: usize,
    /// Value head dim (128 for DeepSeek V3).
    pub v_head_dim: usize,
    /// Number of initial dense layers before MoE kicks in.
    pub first_k_dense_replace: usize,
    /// MoE frequency: every N layers (1 = every layer after dense).
    pub moe_layer_freq: usize,
    /// Expert group count for GroupLimitedGreedy routing.
    pub expert_group_count: usize,
    // ===== DeepSeek V4 =====
    /// Output LoRA rank used by the grouped attention output projection.
    pub output_lora_rank: usize,
    /// Number of attention output groups.
    pub output_group_count: usize,
    /// Per-layer KV compression ratio (V4: 0/4/128; V4.1 Maya: 0/1/2).
    pub compress_ratios: Vec<usize>,
    /// Lightning indexer head count.
    pub indexer_head_count: usize,
    /// Lightning indexer head width.
    pub indexer_head_dim: usize,
    /// Number of keys kept by the lightning indexer.
    pub indexer_top_k: usize,
    /// RoPE base used by compressed V4 attention states.
    pub compress_rope_freq_base: f64,
    /// Hyper-connection stream count.
    pub hyperconnection_count: usize,
    /// Sinkhorn iterations for hyper-connections.
    pub hyperconnection_sinkhorn_iterations: usize,
    /// Epsilon used by the hyper-connection Sinkhorn step.
    pub hyperconnection_epsilon: f64,
    /// Number of early layers that route by token-id hash tables.
    pub hash_layer_count: usize,
    // ===== GLM-5.3 Flash / GLM5-Next =====
    /// Kimi Delta Attention head width.
    pub kda_head_dim: usize,
    /// Lower bound used by the KDA decay gate.
    pub kda_gate_lower_bound: f64,
    /// Number of adjacent keys represented by one sparse-index pool.
    pub indexer_kpool: usize,
    /// Whether sparse attention always includes the incomplete key-pool tail.
    pub indexer_kpool_select_tail: bool,
    /// Per-layer upper bound used by routed-expert SwiGLU.
    pub swiglu_clamp_exp: Vec<f64>,
    /// Per-layer upper bound used by shared-expert SwiGLU.
    pub swiglu_clamp_shexp: Vec<f64>,
}

impl GgufMetadata {
    /// Compute KV cache bytes per token for a given layer range.
    ///
    /// Handles hybrid architectures:
    /// - Dense (llama/mistral/qwen2/qwen3moe): all layers have KV cache
    /// - Qwen3.5: only every Nth layer is full attention (rest are DeltaNet, O(1) state)
    /// - Nemotron-H: per-layer arrays determine which layers have attention
    ///
    /// KV cache stores f16 tensors: 2 (K+V) * num_kv_heads * head_dim * 2 bytes per layer per token.
    pub fn kv_bytes_per_token(&self, layer_start: usize, layer_end: usize) -> u64 {
        let layer_start = layer_start.min(self.num_layers);
        let layer_end = layer_end.min(self.num_layers);
        if layer_start >= layer_end {
            return 0;
        }
        if self.architecture == "deepseek41" {
            // Model-wide F32 QDQ carriers: three ratio-two owners and one
            // ratio-one owner. Raw rings/pairs have a separate fixed reserve.
            return [2usize, 8, 14, 20]
                .into_iter()
                .filter(|&owner| (layer_start..layer_end).contains(&owner))
                .map(|owner| if owner == 20 { 2560 } else { 1280 })
                .sum();
        }
        if matches!(self.architecture.as_str(), "glm5next" | "glm5-next") {
            // Each DSA layer stores one f16 latent plus the f16 index key and
            // pool gate. KDA layers use fixed recurrent state, not token KV.
            let bytes_per_dsa_layer = usize_to_u64(self.kv_lora_rank)
                .saturating_add(usize_to_u64(self.indexer_head_dim).saturating_mul(2))
                .saturating_mul(2);
            return ((layer_start..layer_end)
                .filter(|&layer| self.head_count_kv_arr.get(layer).copied().unwrap_or(0) > 0)
                .count() as u64)
                .saturating_mul(bytes_per_dsa_layer);
        }
        if !self.head_count_kv_arr.is_empty() {
            return (layer_start..layer_end)
                .map(|i| {
                    let kv_heads = self
                        .head_count_kv_arr
                        .get(i)
                        .copied()
                        .unwrap_or(self.num_kv_heads);
                    if kv_heads == 0 {
                        0
                    } else {
                        usize_to_u64(kv_heads)
                            .saturating_mul(usize_to_u64(self.head_dim_for_layer(i)))
                            .saturating_mul(4)
                    }
                })
                .fold(0u64, u64::saturating_add);
        }

        let per_layer_bytes = usize_to_u64(self.num_kv_heads)
            .saturating_mul(usize_to_u64(self.head_dim))
            .saturating_mul(4); // K + V, f16 = 2 bytes.
        self.attention_layer_count(layer_start, layer_end)
            .saturating_mul(per_layer_bytes)
    }

    /// KV cache with TurboQuant compression: packed indices + f32 norm per vector.
    pub fn kv_bytes_per_token_quantized(
        &self,
        layer_start: usize,
        layer_end: usize,
        bits: u8,
    ) -> u64 {
        let layer_start = layer_start.min(self.num_layers);
        let layer_end = layer_end.min(self.num_layers);
        if layer_start >= layer_end {
            return 0;
        }
        if self.architecture == "deepseek41" {
            // Reporting must not invent an F16/TurboQuant fallback; construction
            // explicitly rejects TurboQuant for this native cache.
            return self.kv_bytes_per_token(layer_start, layer_end);
        }
        if matches!(self.architecture.as_str(), "glm5next" | "glm5-next") {
            // GLM's asymmetric latent/index cache stays f16.
            return self.kv_bytes_per_token(layer_start, layer_end);
        }
        if !self.head_count_kv_arr.is_empty() {
            return (layer_start..layer_end)
                .map(|i| {
                    let kv_heads = self
                        .head_count_kv_arr
                        .get(i)
                        .copied()
                        .unwrap_or(self.num_kv_heads);
                    if kv_heads == 0 {
                        0
                    } else {
                        let head_dim = self.head_dim_for_layer(i);
                        let packed_per_vec = usize_to_u64(head_dim)
                            .saturating_mul(u64::from(bits))
                            .div_ceil(8)
                            .saturating_add(4);
                        usize_to_u64(kv_heads)
                            .saturating_mul(packed_per_vec)
                            .saturating_mul(2)
                    }
                })
                .fold(0u64, u64::saturating_add);
        }

        let packed_per_vec = usize_to_u64(self.head_dim)
            .saturating_mul(u64::from(bits))
            .div_ceil(8)
            .saturating_add(4); // + f32 norm.
        let per_layer_bytes = usize_to_u64(self.num_kv_heads)
            .saturating_mul(packed_per_vec)
            .saturating_mul(2);
        self.attention_layer_count(layer_start, layer_end)
            .saturating_mul(per_layer_bytes)
    }

    pub(crate) fn head_dim_for_layer(&self, layer_idx: usize) -> usize {
        if self.architecture == "gemma4"
            && self.head_dim_swa > 0
            && self
                .sliding_window_pattern
                .get(layer_idx)
                .copied()
                .unwrap_or(false)
        {
            self.head_dim_swa
        } else {
            self.head_dim
        }
    }

    /// Count attention layers in a range (handles dense, hybrid, and per-layer-array models).
    fn attention_layer_count(&self, layer_start: usize, layer_end: usize) -> u64 {
        if !self.head_count_kv_arr.is_empty() {
            // Per-layer arrays: attention layers have head_count_kv > 0.
            (layer_start..layer_end)
                .filter(|&i| i < self.head_count_kv_arr.len() && self.head_count_kv_arr[i] > 0)
                .count() as u64
        } else if self.full_attention_interval > 0 {
            // Qwen3.5: every Nth layer is full attention
            (layer_start..layer_end)
                .filter(|&i| (i + 1) % self.full_attention_interval == 0)
                .count() as u64
        } else {
            // Dense: all layers have attention
            (layer_end - layer_start) as u64
        }
    }
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

/// Read a string value from GGUF metadata.
fn md_get_str(md: &HashMap<String, gguf_file::Value>, key: &str) -> Result<String> {
    match md.get(key) {
        Some(gguf_file::Value::String(s)) => Ok(s.clone()),
        _ => anyhow::bail!("missing or invalid GGUF metadata key: {key}"),
    }
}

/// Validate the complete GGUF header before handing it to Candle's reader.
/// This includes the metadata values, tensor table, and alignment padding, so
/// a header with individually acceptable counts cannot exceed the aggregate
/// application budget.
#[cfg(test)]
fn validate_gguf_header_counts(path: &Path) -> Result<()> {
    open_validated_gguf(path).map(|_| ())
}

/// Open and validate a GGUF, returning the same descriptor for parsing so a
/// file replaced at the path after validation is never handed to Candle.
fn open_validated_gguf(
    path: &Path,
) -> Result<(std::fs::File, crate::gguf_limits::GgufHeaderStats)> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open GGUF file: {}", path.display()))?;
    let stats = validate_gguf_header_budget(&mut file)
        .map_err(|error| anyhow::anyhow!("invalid GGUF header {}: {error}", path.display()))?;
    Ok((file, stats))
}

fn account_split_header_bytes(total: &mut u64, header_bytes: u64) -> Result<()> {
    *total = total
        .checked_add(header_bytes)
        .context("aggregate GGUF split-header size overflow")?;
    anyhow::ensure!(
        *total <= MAX_GGUF_AGGREGATE_HEADER_BYTES,
        "aggregate GGUF split headers ({total} bytes) exceed application limit ({MAX_GGUF_AGGREGATE_HEADER_BYTES})"
    );
    Ok(())
}

/// Split GGUF files may keep model metadata only in the first shard.
fn read_model_architecture(first_shard: &Path) -> Result<String> {
    let (mut file, _) = open_validated_gguf(first_shard)?;
    match gguf_file::Content::read(&mut file) {
        Ok(content) => md_get_str(&content.metadata, "general.architecture"),
        Err(candle_error) => deepseek4_gguf::read_architecture(first_shard).with_context(|| {
            format!(
                "Failed to parse GGUF header with Candle ({candle_error}) or the DINF V4 reader: {}",
                first_shard.display()
            )
        }),
    }
}

/// Read a u32/u64 value from GGUF metadata as usize.
fn md_get_usize(md: &HashMap<String, gguf_file::Value>, key: &str) -> Result<usize> {
    match md.get(key) {
        Some(gguf_file::Value::U32(v)) => Ok(*v as usize),
        Some(gguf_file::Value::U64(v)) => usize::try_from(*v)
            .with_context(|| format!("GGUF metadata key does not fit usize: {key}")),
        Some(gguf_file::Value::I32(v)) => usize::try_from(*v)
            .with_context(|| format!("GGUF metadata key must be non-negative: {key}")),
        Some(value) => anyhow::bail!(
            "invalid GGUF metadata key {key}: expected an integer, got {:?}",
            value.value_type()
        ),
        None => anyhow::bail!("missing GGUF metadata key: {key}"),
    }
}

/// Read an optional integer while retaining the distinction between an absent
/// key (which may use a compatibility default) and a present malformed key.
fn md_get_usize_or(
    md: &HashMap<String, gguf_file::Value>,
    key: &str,
    default: usize,
) -> Result<usize> {
    match md.get(key) {
        Some(_) => md_get_usize(md, key),
        None => Ok(default),
    }
}

/// Read a scalar integer with compatibility for metadata that stores the
/// same architecture setting as a per-layer integer array. The array itself
/// is validated by the corresponding `md_get_usize_array` call.
fn md_get_usize_or_scalar(
    md: &HashMap<String, gguf_file::Value>,
    key: &str,
    default: usize,
) -> Result<usize> {
    match md.get(key) {
        Some(gguf_file::Value::Array(_)) => Ok(default),
        _ => md_get_usize_or(md, key, default),
    }
}

/// Read an optional structural integer while still rejecting a present value
/// with the wrong type. Optional architecture metadata may be absent on older
/// conversions, but malformed metadata must never silently select a different
/// executable layout.
fn md_get_optional_usize(
    md: &HashMap<String, gguf_file::Value>,
    key: &str,
) -> Result<Option<usize>> {
    match md.get(key) {
        Some(_) => md_get_usize(md, key).map(Some),
        None => Ok(None),
    }
}

/// Read a float value from GGUF metadata.
fn md_get_f64(md: &HashMap<String, gguf_file::Value>, key: &str, default: f64) -> Result<f64> {
    let value = match md.get(key) {
        Some(gguf_file::Value::F32(v)) => *v as f64,
        Some(gguf_file::Value::F64(v)) => *v,
        Some(value) => anyhow::bail!(
            "invalid GGUF metadata key {key}: expected a float, got {:?}",
            value.value_type()
        ),
        None => return Ok(default),
    };
    anyhow::ensure!(
        value.is_finite(),
        "GGUF metadata key {key} must contain a finite float"
    );
    Ok(value)
}

/// Read a finite float or the explicit negative-infinity sentinel used by
/// GLM's unbounded lower decay gate.
fn md_get_f64_or_negative_infinity(
    md: &HashMap<String, gguf_file::Value>,
    key: &str,
    default: f64,
) -> Result<f64> {
    let value = match md.get(key) {
        Some(gguf_file::Value::F32(value)) => f64::from(*value),
        Some(gguf_file::Value::F64(value)) => *value,
        Some(value) => anyhow::bail!(
            "invalid GGUF metadata key {key}: expected a float, got {:?}",
            value.value_type()
        ),
        None => return Ok(default),
    };
    anyhow::ensure!(
        value.is_finite() || value == f64::NEG_INFINITY,
        "GGUF metadata key {key} must contain a finite float or negative infinity"
    );
    Ok(value)
}

/// Read a boolean value from GGUF metadata.
fn md_get_bool(md: &HashMap<String, gguf_file::Value>, key: &str, default: bool) -> Result<bool> {
    match md.get(key) {
        Some(gguf_file::Value::Bool(v)) => Ok(*v),
        Some(value) => anyhow::bail!(
            "invalid GGUF metadata key {key}: expected a boolean, got {:?}",
            value.value_type()
        ),
        None => Ok(default),
    }
}

/// Read an array of integers from GGUF metadata as Vec<usize>.
/// Returns empty vec if key is missing.
fn md_get_usize_array(md: &HashMap<String, gguf_file::Value>, key: &str) -> Result<Vec<usize>> {
    let Some(value) = md.get(key) else {
        return Ok(Vec::new());
    };
    let gguf_file::Value::Array(values) = value else {
        anyhow::bail!(
            "invalid GGUF metadata key {key}: expected an integer array, got {:?}",
            value.value_type()
        );
    };
    values
        .iter()
        .enumerate()
        .map(|(index, value)| match value {
            gguf_file::Value::U32(n) => Ok(*n as usize),
            gguf_file::Value::U64(n) => usize::try_from(*n).with_context(|| {
                format!("GGUF metadata key {key} array value {index} does not fit usize")
            }),
            gguf_file::Value::I32(n) => usize::try_from(*n).with_context(|| {
                format!("GGUF metadata key {key} array value {index} must be non-negative")
            }),
            value => anyhow::bail!(
                "invalid GGUF metadata key {key} array value {index}: expected an integer, got {:?}",
                value.value_type()
            ),
        })
        .collect()
}

/// Read a per-layer integer array while accepting the scalar form used by
/// older converters for uniform values. Uniform scalar metadata is represented
/// by an empty per-layer array because callers fall back to the scalar field.
fn md_get_usize_array_or_scalar(
    md: &HashMap<String, gguf_file::Value>,
    key: &str,
) -> Result<Vec<usize>> {
    match md.get(key) {
        Some(gguf_file::Value::Array(_)) => md_get_usize_array(md, key),
        Some(_) => {
            md_get_usize(md, key)?;
            Ok(Vec::new())
        }
        None => Ok(Vec::new()),
    }
}

/// Read an array of booleans from GGUF metadata.
/// Returns empty vec if key is missing.
fn md_get_bool_array(md: &HashMap<String, gguf_file::Value>, key: &str) -> Result<Vec<bool>> {
    let Some(value) = md.get(key) else {
        return Ok(Vec::new());
    };
    let gguf_file::Value::Array(values) = value else {
        anyhow::bail!(
            "invalid GGUF metadata key {key}: expected a boolean array, got {:?}",
            value.value_type()
        );
    };
    values
        .iter()
        .enumerate()
        .map(|(index, value)| match value {
            gguf_file::Value::Bool(value) => Ok(*value),
            value => anyhow::bail!(
                "invalid GGUF metadata key {key} array value {index}: expected a boolean, got {:?}",
                value.value_type()
            ),
        })
        .collect()
}

fn llama4_attention_metadata(
    md: &HashMap<String, gguf_file::Value>,
    num_layers: usize,
) -> Result<(usize, Vec<bool>)> {
    const WINDOW_KEY: &str = "llama4.attention.sliding_window";
    const PATTERN_KEY: &str = "llama4.attention.sliding_window_pattern";
    // Llama 4's missing key means 8,192-token chunks; explicit zero disables
    // chunking. Reject other spans rather than silently replacing their value.
    let window = md_get_usize_or(md, WINDOW_KEY, 8192)?;
    anyhow::ensure!(
        matches!(window, 0 | 8192),
        "GGUF metadata key {WINDOW_KEY} must be 0 (full attention) or 8192 (chunked attention), got {window}"
    );
    // llama.cpp b8920 set_swa_pattern: 0 means all chunked, 1 means all full,
    // and N > 1 means N-1 chunked layers followed by one full layer. Its scalar
    // loader ignores arrays; reject them here instead of guessing their meaning.
    let period = md_get_usize_or(md, PATTERN_KEY, 4)?;
    anyhow::ensure!(
        u32::try_from(period).is_ok(),
        "GGUF metadata key {PATTERN_KEY} must fit an unsigned 32-bit period, got {period}"
    );
    let pattern = (0..num_layers)
        .map(|layer| window != 0 && (period == 0 || layer % period < period - 1))
        .collect();
    Ok((window, pattern))
}

/// Read an array of floating-point metadata values.
///
/// A single float is accepted as a compatibility alias for a one-element
/// array. Missing keys still produce an empty vector.
fn md_get_f64_array(md: &HashMap<String, gguf_file::Value>, key: &str) -> Result<Vec<f64>> {
    let Some(value) = md.get(key) else {
        return Ok(Vec::new());
    };
    let values: Vec<f64> = match value {
        gguf_file::Value::Array(values) => values
            .iter()
            .enumerate()
            .map(|(index, value)| match value {
                gguf_file::Value::F32(value) => Ok(*value as f64),
                gguf_file::Value::F64(value) => Ok(*value),
                value => anyhow::bail!(
                    "invalid GGUF metadata key {key} array value {index}: expected a float, got {:?}",
                    value.value_type()
                ),
            })
            .collect::<Result<Vec<_>>>()?,
        gguf_file::Value::F32(value) => vec![*value as f64],
        gguf_file::Value::F64(value) => vec![*value],
        value => anyhow::bail!(
            "invalid GGUF metadata key {key}: expected a float array, got {:?}",
            value.value_type()
        ),
    };
    anyhow::ensure!(
        values.iter().all(|value| value.is_finite()),
        "GGUF metadata key {key} must contain only finite floats"
    );
    Ok(values)
}

/// Read the tokenizer vocabulary array length, validating that every entry is
/// a GGUF string instead of silently accepting a malformed array.
fn md_get_string_array_len(
    md: &HashMap<String, gguf_file::Value>,
    key: &str,
) -> Result<Option<usize>> {
    let Some(value) = md.get(key) else {
        return Ok(None);
    };
    let gguf_file::Value::Array(values) = value else {
        anyhow::bail!(
            "invalid GGUF metadata key {key}: expected a string array, got {:?}",
            value.value_type()
        );
    };
    for (index, value) in values.iter().enumerate() {
        anyhow::ensure!(
            matches!(value, gguf_file::Value::String(_)),
            "invalid GGUF metadata key {key} array value {index}: expected a string, got {:?}",
            value.value_type()
        );
    }
    Ok(Some(values.len()))
}

/// Validate the finite F32 values actually used by the RoPE constructors.
/// Positive metadata can still produce an infinite inverse frequency when a
/// sub-unit base underflows during `theta.powf`, or an infinite angle when the
/// resulting frequency is multiplied by the largest cached position.
fn validate_rope_theta_generation(
    theta: f64,
    head_dim: usize,
    context_length: usize,
    key: &str,
) -> Result<()> {
    let theta_f32 = theta as f32;
    anyhow::ensure!(
        theta_f32.is_finite() && theta_f32 > 0.0,
        "GGUF metadata key {key} must be positive and representable as F32"
    );
    let half_dim = head_dim / 2;
    if half_dim == 0 {
        return Ok(());
    }

    // For theta < 1, the largest inverse frequency is the last rotary
    // channel; for theta >= 1 it is the first channel (exactly 1). Checking
    // that endpoint bounds every intermediate channel without iterating over
    // attacker-controlled head dimensions.
    let exponent = (2.0_f32 * ((half_dim - 1) as f32)) / (head_dim as f32);
    let denominator = theta_f32.powf(exponent);
    let endpoint_inverse = 1.0_f32 / denominator;
    anyhow::ensure!(
        denominator.is_finite()
            && denominator > 0.0
            && endpoint_inverse.is_finite()
            && endpoint_inverse > 0.0,
        "GGUF metadata key {key} would produce a non-finite F32 RoPE inverse frequency"
    );
    let max_inverse = if theta_f32 < 1.0 {
        endpoint_inverse
    } else {
        1.0
    };
    let max_position = context_length.saturating_sub(1) as f32;
    anyhow::ensure!(
        (max_position * max_inverse).is_finite(),
        "GGUF metadata key {key} would produce a non-finite F32 RoPE angle at the supported context length"
    );
    Ok(())
}

/// Check dimensions before a decoder builds or applies rotary tables.
pub(crate) fn validate_rope_dimensions(rope_dim: usize, head_dim: usize, key: &str) -> Result<()> {
    anyhow::ensure!(
        rope_dim > 0 && rope_dim.is_multiple_of(2),
        "GGUF metadata key {key} must be positive and even for RoPE"
    );
    anyhow::ensure!(
        rope_dim <= head_dim,
        "GGUF metadata key {key} ({rope_dim}) exceeds attention head dimension ({head_dim})"
    );
    Ok(())
}

/// Parse GGUF metadata to extract model architecture parameters.
pub fn parse_gguf_metadata(md: &HashMap<String, gguf_file::Value>) -> Result<GgufMetadata> {
    if matches!(md.get("general.architecture"), Some(gguf_file::Value::String(a)) if a == "deepseek41")
    {
        return parse_deepseek41_metadata(md);
    }
    parse_canonical_metadata(md)
}

fn parse_canonical_metadata(md: &HashMap<String, gguf_file::Value>) -> Result<GgufMetadata> {
    let architecture = md_get_str(md, "general.architecture")?;
    let prefix = &architecture;

    let physical_num_layers = md_get_usize(md, &format!("{prefix}.block_count"))?;
    validate_layer_count(physical_num_layers, "GGUF physical layer count")?;
    let num_layers = if matches!(architecture.as_str(), "glm5next" | "glm5-next" | "qwen35") {
        // GLM5-Next and some Qwen3.5-family files carry physical NextN/MTP
        // blocks after the ordinary trunk. They are not executable by DINF's
        // pipeline stage and must not become assignable layers.
        let nextn_predict_layers =
            md_get_optional_usize(md, &format!("{prefix}.nextn_predict_layers"))?.unwrap_or(0);
        anyhow::ensure!(
            nextn_predict_layers < physical_num_layers,
            "{prefix}.nextn_predict_layers ({nextn_predict_layers}) must be less than physical block_count ({physical_num_layers})"
        );
        physical_num_layers - nextn_predict_layers
    } else {
        physical_num_layers
    };
    validate_layer_count(num_layers, "GGUF executable layer count")?;
    let hidden_dim = md_get_usize(md, &format!("{prefix}.embedding_length"))?;
    let num_attention_heads = md_get_usize(md, &format!("{prefix}.attention.head_count"))?;
    anyhow::ensure!(num_layers > 0, "GGUF model must contain at least one layer");
    anyhow::ensure!(hidden_dim > 0, "GGUF embedding length must be positive");
    anyhow::ensure!(
        num_attention_heads > 0,
        "GGUF attention head count must be positive"
    );
    let num_kv_heads = md_get_usize_or_scalar(
        md,
        &format!("{prefix}.attention.head_count_kv"),
        num_attention_heads,
    )?;
    let head_dim = md_get_usize_or(
        md,
        &format!("{prefix}.attention.key_length"),
        hidden_dim / num_attention_heads,
    )?;
    let default_intermediate_dim = hidden_dim
        .checked_mul(4)
        .context("GGUF feed-forward dimension overflow")?;
    let intermediate_dim = md_get_usize_or_scalar(
        md,
        &format!("{prefix}.feed_forward_length"),
        default_intermediate_dim,
    )?;
    let vocab_size = {
        let model_vocab_size = md_get_optional_usize(md, &format!("{prefix}.vocab_size"))?;
        let tokenizer_vocab_size = md_get_optional_usize(md, "tokenizer.ggml.vocab_size")?;
        let tokenizer_tokens = md_get_string_array_len(md, "tokenizer.ggml.tokens")?;
        model_vocab_size
            .or(tokenizer_vocab_size)
            .or(tokenizer_tokens)
            .unwrap_or(32000)
    };
    let rms_norm_eps = md_get_f64(
        md,
        &format!("{prefix}.attention.layer_norm_rms_epsilon"),
        1e-5,
    )?;
    let rope_theta = md_get_f64(md, &format!("{prefix}.rope.freq_base"), 10000.0)?;
    let context_length = md_get_usize_or(md, &format!("{prefix}.context_length"), 4096)?;
    anyhow::ensure!(
        rms_norm_eps.is_finite()
            && rms_norm_eps > 0.0
            && (rms_norm_eps as f32).is_finite()
            && (rms_norm_eps as f32) >= f32::MIN_POSITIVE,
        "GGUF metadata key {prefix}.attention.layer_norm_rms_epsilon must be a positive normal F32"
    );
    anyhow::ensure!(
        rope_theta.is_finite()
            && rope_theta > 0.0
            && (rope_theta as f32).is_finite()
            && (rope_theta as f32) > 0.0,
        "GGUF metadata key {prefix}.rope.freq_base must be positive and representable as F32"
    );
    let max_context_length = if architecture == "llama4" {
        MAX_LLAMA4_CONTEXT_LENGTH
    } else {
        MAX_SUPPORTED_CONTEXT_LENGTH
    };
    anyhow::ensure!(
        context_length <= max_context_length,
        "GGUF metadata key {prefix}.context_length ({context_length}) exceeds the supported context length limit ({max_context_length})"
    );
    anyhow::ensure!(num_kv_heads > 0, "GGUF KV head count must be positive");
    anyhow::ensure!(
        head_dim > 0,
        "GGUF attention head dimension must be positive"
    );
    anyhow::ensure!(
        intermediate_dim > 0,
        "GGUF feed-forward dimension must be positive"
    );
    anyhow::ensure!(vocab_size > 0, "GGUF vocabulary size must be positive");
    anyhow::ensure!(
        context_length > 0,
        "GGUF metadata key {prefix}.context_length must be a positive context length"
    );
    validate_rope_theta_generation(
        rope_theta,
        head_dim,
        context_length,
        &format!("{prefix}.rope.freq_base"),
    )?;

    // MoE metadata (optional — 0 for dense models)
    let expert_count = md_get_usize_or(md, &format!("{prefix}.expert_count"), 0)?;
    let expert_used_count = md_get_usize_or(md, &format!("{prefix}.expert_used_count"), 0)?;
    let expert_ffn_dim = md_get_usize_or(md, &format!("{prefix}.expert_feed_forward_length"), 0)?;

    // SSM / DeltaNet metadata (optional — 0 for non-SSM models)
    let ssm_state_size = md_get_usize_or(md, &format!("{prefix}.ssm.state_size"), 0)?;
    let ssm_conv_kernel = md_get_usize_or(md, &format!("{prefix}.ssm.conv_kernel"), 0)?;
    let ssm_group_count = md_get_usize_or(md, &format!("{prefix}.ssm.group_count"), 0)?;
    let ssm_time_step_rank = md_get_usize_or(md, &format!("{prefix}.ssm.time_step_rank"), 0)?;
    let ssm_inner_size = md_get_usize_or(md, &format!("{prefix}.ssm.inner_size"), 0)?;
    let rope_dimension_count = md_get_usize_or(md, &format!("{prefix}.rope.dimension_count"), 0)?;
    let rope_dimension_sections =
        md_get_usize_array(md, &format!("{prefix}.rope.dimension_sections"))?;
    let rope_scaling_factor = md_get_f64(md, &format!("{prefix}.rope.scaling.factor"), 1.0)?;
    let rope_scaling_original_context_length = md_get_usize_or(
        md,
        &format!("{prefix}.rope.scaling.original_context_length"),
        context_length,
    )?;
    let rope_yarn_log_multiplier = md_get_f64(
        md,
        &format!("{prefix}.rope.scaling.yarn_log_multiplier"),
        0.0,
    )?;
    let rope_dimension_count_swa =
        md_get_usize_or(md, &format!("{prefix}.rope.dimension_count_swa"), 0)?;
    let rope_yarn_beta_fast =
        md_get_f64(md, &format!("{prefix}.rope.scaling.yarn_beta_fast"), 32.)?;
    let rope_yarn_beta_slow = md_get_f64(md, &format!("{prefix}.rope.scaling.yarn_beta_slow"), 1.)?;
    let head_dim_swa = md_get_usize_or(md, &format!("{prefix}.attention.key_length_swa"), 0)?;
    let rope_theta_swa = md_get_f64(md, &format!("{prefix}.rope.freq_base_swa"), 0.0)?;
    anyhow::ensure!(
        rope_theta_swa == 0.0
            || (rope_theta_swa > 0.0
                && (rope_theta_swa as f32).is_finite()
                && (rope_theta_swa as f32) >= f32::MIN_POSITIVE),
        "GGUF metadata key {prefix}.rope.freq_base_swa must be zero or a positive normal F32"
    );
    if matches!(architecture.as_str(), "gemma4" | "llama4") && rope_theta_swa != 0.0 {
        validate_rope_theta_generation(
            rope_theta_swa,
            head_dim.max(head_dim_swa),
            context_length,
            &format!("{prefix}.rope.freq_base_swa"),
        )?;
    }
    let (sliding_window, sliding_window_pattern) = if architecture == "llama4" {
        llama4_attention_metadata(md, num_layers)?
    } else {
        (
            md_get_usize_or(md, &format!("{prefix}.attention.sliding_window"), 0)?,
            md_get_bool_array(md, &format!("{prefix}.attention.sliding_window_pattern"))?,
        )
    };
    let final_logit_softcapping =
        md_get_f64(md, &format!("{prefix}.final_logit_softcapping"), 0.0)?;
    let full_attention_interval = {
        let primary = md_get_optional_usize(md, &format!("{prefix}.full_attention_interval"))?;
        let alias =
            md_get_optional_usize(md, &format!("{prefix}.attention.full_attention_interval"))?;
        primary.or(alias).unwrap_or(0)
    };

    // Per-layer arrays (nemotron_h hybrid models)
    let head_count_kv_arr =
        md_get_usize_array_or_scalar(md, &format!("{prefix}.attention.head_count_kv"))?;
    let feed_forward_length_arr =
        md_get_usize_array_or_scalar(md, &format!("{prefix}.feed_forward_length"))?;

    // Additional MoE parameters
    let attention_scale = md_get_f64(md, &format!("{prefix}.attention.scale"), 0.0)?;
    let expert_weights_scale = md_get_f64(md, &format!("{prefix}.expert_weights_scale"), 0.0)?;
    let expert_weights_norm = md_get_bool(md, &format!("{prefix}.expert_weights_norm"), false)?;
    let expert_gating_func = md_get_usize_or(md, &format!("{prefix}.expert_gating_func"), 0)?;
    let shared_expert_count = md_get_usize_or(md, &format!("{prefix}.expert_shared_count"), 0)?;
    let post_norm_eps = md_get_f64(md, &format!("{prefix}.post_norm_epsilon"), 0.0)?;
    let moe_latent_size = md_get_usize_or(md, &format!("{prefix}.moe_latent_size"), 0)?;
    let logit_scale = md_get_f64(md, &format!("{prefix}.logit_scale"), 1.0)?;
    if architecture == "muse-spark" {
        for key in ["post_norm_epsilon", "moe_latent_size", "logit_scale"] {
            anyhow::ensure!(
                md.contains_key(&format!("{prefix}.{key}")),
                "Muse Spark metadata is missing {prefix}.{key}"
            );
        }
        anyhow::ensure!(
            post_norm_eps > 0.0
                && (post_norm_eps as f32).is_finite()
                && (post_norm_eps as f32) >= f32::MIN_POSITIVE,
            "Muse Spark post-normalization epsilon must be positive and representable as F32"
        );
        anyhow::ensure!(
            logit_scale > 0.0 && (logit_scale as f32).is_finite() && (logit_scale as f32) > 0.0,
            "Muse Spark logit scale must be positive and representable as F32"
        );
        anyhow::ensure!(
            moe_latent_size > 0 && moe_latent_size <= hidden_dim,
            "Muse Spark pooled expert width must fit the hidden width"
        );
        anyhow::ensure!(
            expert_count > 0
                && expert_used_count > 0
                && expert_used_count <= expert_count
                && expert_ffn_dim > 0,
            "Muse Spark requires valid routed experts"
        );
        anyhow::ensure!(
            expert_gating_func == 2
                && expert_weights_norm
                && expert_weights_scale > 0.0
                && (expert_weights_scale as f32).is_finite()
                && (expert_weights_scale as f32) > 0.0,
            "Muse Spark requires normalized sigmoid expert routing"
        );
        anyhow::ensure!(
            sliding_window > 0
                && sliding_window <= context_length
                && sliding_window_pattern.len() == num_layers,
            "Muse Spark requires one sliding-window flag per layer and a valid window"
        );
        validate_rope_theta_generation(
            rope_theta_swa,
            head_dim,
            context_length,
            &format!("{prefix}.rope.freq_base_swa"),
        )?;
        anyhow::ensure!(
            rope_theta_swa > 0.0,
            "Muse Spark requires the sliding-window RoPE base"
        );
        validate_rope_dimensions(
            if rope_dimension_count == 0 {
                head_dim
            } else {
                rope_dimension_count
            },
            head_dim,
            &format!("{prefix}.rope.dimension_count"),
        )?;
        anyhow::ensure!(
            shared_expert_count == 0,
            "Muse Spark does not have shared experts"
        );
    }

    // DeepSeek V4 metadata. These keys are absent on older architectures, so
    // the zero values below keep the existing model paths unchanged.
    let output_lora_rank = md_get_usize_or(md, &format!("{prefix}.attention.output_lora_rank"), 0)?;
    let output_group_count =
        md_get_usize_or(md, &format!("{prefix}.attention.output_group_count"), 0)?;
    let compress_ratios = md_get_usize_array(md, &format!("{prefix}.attention.compress_ratios"))?;
    let indexer_head_count =
        md_get_usize_or(md, &format!("{prefix}.attention.indexer.head_count"), 0)?;
    let indexer_head_dim =
        md_get_usize_or(md, &format!("{prefix}.attention.indexer.key_length"), 0)?;
    let indexer_top_k = md_get_usize_or(md, &format!("{prefix}.attention.indexer.top_k"), 0)?;
    let compress_rope_freq_base = md_get_f64(
        md,
        &format!("{prefix}.attention.compress_rope_freq_base"),
        0.0,
    )?;
    anyhow::ensure!(
        compress_rope_freq_base == 0.0
            || (compress_rope_freq_base > 0.0
                && (compress_rope_freq_base as f32).is_finite()
                && (compress_rope_freq_base as f32) >= f32::MIN_POSITIVE),
        "GGUF metadata key {prefix}.attention.compress_rope_freq_base must be zero or a positive normal F32"
    );
    if architecture == "deepseek4"
        && compress_ratios
            .iter()
            .take(num_layers)
            .any(|&ratio| ratio > 0)
    {
        validate_rope_theta_generation(
            compress_rope_freq_base,
            head_dim,
            context_length,
            &format!("{prefix}.attention.compress_rope_freq_base"),
        )?;
        anyhow::ensure!(
            compress_rope_freq_base > 1.0,
            "GGUF metadata key {prefix}.attention.compress_rope_freq_base must be greater than 1 when compressed RoPE is enabled"
        );
    }
    let hyperconnection_count =
        md_get_usize_or(md, &format!("{prefix}.hyper_connection.count"), 0)?;
    let hyperconnection_sinkhorn_iterations = md_get_usize_or(
        md,
        &format!("{prefix}.hyper_connection.sinkhorn_iterations"),
        0,
    )?;
    let hyperconnection_epsilon =
        md_get_f64(md, &format!("{prefix}.hyper_connection.epsilon"), 0.0)?;
    let hash_layer_count = md_get_usize_or(md, &format!("{prefix}.hash_layer_count"), 0)?;

    // GLM5-Next hybrid-attention metadata. The converter omits
    // `kpool_select_tail` when it uses the architecture default, which is true.
    let kda_head_dim = md_get_usize_or(md, &format!("{prefix}.kda.head_dim"), 0)?;
    let kda_gate_lower_bound = md_get_f64_or_negative_infinity(
        md,
        &format!("{prefix}.kda.gate_lower_bound"),
        f64::NEG_INFINITY,
    )?;
    let is_glm5 = matches!(architecture.as_str(), "glm5next" | "glm5-next");
    let indexer_kpool = md_get_usize_or(
        md,
        &format!("{prefix}.attention.indexer.kpool"),
        if is_glm5 { 4 } else { 0 },
    )?;
    let indexer_kpool_select_tail = md_get_bool(
        md,
        &format!("{prefix}.attention.indexer.kpool_select_tail"),
        architecture == "glm5next" || architecture == "glm5-next",
    )?;
    let expand_layer_values = |mut values: Vec<f64>| {
        if values.len() == 1 && num_layers > 1 {
            values.resize(num_layers, values[0]);
        }
        values
    };
    let mut swiglu_clamp_exp =
        expand_layer_values(md_get_f64_array(md, &format!("{prefix}.swiglu_clamp_exp"))?);
    if is_glm5 && swiglu_clamp_exp.is_empty() {
        // Some GLM5 converters omit the per-layer clamp arrays even though
        // the architecture's reference value is 10.0. Keep this default
        // scoped to the known GLM5 format; other models must continue to
        // provide their own arrays when the architecture requires them.
        swiglu_clamp_exp = vec![10.0; num_layers];
    }
    let mut swiglu_clamp_shexp = expand_layer_values(md_get_f64_array(
        md,
        &format!("{prefix}.swiglu_clamp_shexp"),
    )?);
    if swiglu_clamp_shexp.is_empty() {
        swiglu_clamp_shexp = swiglu_clamp_exp.clone();
    }

    // MLA metadata (DeepSeek V3/R1 — 0 for non-MLA models)
    let kv_lora_rank = md_get_usize_or(md, &format!("{prefix}.attention.kv_lora_rank"), 0)?;
    let q_lora_rank = md_get_usize_or(md, &format!("{prefix}.attention.q_lora_rank"), 0)?;
    let qk_rope_head_dim =
        if matches!(architecture.as_str(), "deepseek4" | "deepseek41") || kv_lora_rank > 0 {
            rope_dimension_count
        } else {
            0
        };
    if matches!(architecture.as_str(), "deepseek2" | "glm-dsa") {
        validate_rope_dimensions(
            qk_rope_head_dim,
            head_dim,
            &format!("{prefix}.rope.dimension_count"),
        )?;
    }
    if architecture == "phi3" {
        validate_rope_dimensions(
            if rope_dimension_count > 0 {
                rope_dimension_count
            } else {
                head_dim
            },
            head_dim,
            "phi3.rope.dimension_count",
        )?;
    }
    let qk_nope_head_dim = md_get_usize_or(
        md,
        &format!("{prefix}.attention.key_length_mla"),
        head_dim.saturating_sub(qk_rope_head_dim),
    )?;
    let v_head_dim = {
        let primary = md_get_optional_usize(md, &format!("{prefix}.attention.value_length_mla"))?;
        let alias = md_get_optional_usize(md, &format!("{prefix}.attention.value_length"))?;
        primary.or(alias).unwrap_or(0)
    };
    let first_k_dense_replace =
        md_get_usize_or(md, &format!("{prefix}.leading_dense_block_count"), 0)?;
    anyhow::ensure!(
        architecture != "muse-spark" || first_k_dense_replace == 0,
        "Muse Spark supports the published all-routed layout"
    );
    let moe_layer_freq =
        if md_get_optional_usize(md, &format!("{prefix}.expert_feed_forward_length"))?.is_some() {
            1
        } else {
            0
        };
    let expert_group_count = md_get_usize_or(md, &format!("{prefix}.expert_group_count"), 1)?;

    Ok(GgufMetadata {
        architecture,
        num_layers,
        hidden_dim,
        num_attention_heads,
        num_kv_heads,
        head_dim,
        intermediate_dim,
        vocab_size,
        rms_norm_eps,
        post_norm_eps,
        moe_latent_size,
        logit_scale,
        rope_theta,
        context_length,
        expert_count,
        expert_used_count,
        expert_ffn_dim,
        ssm_state_size,
        ssm_conv_kernel,
        ssm_group_count,
        ssm_time_step_rank,
        ssm_inner_size,
        rope_dimension_count,
        rope_dimension_sections,
        rope_scaling_factor,
        rope_scaling_original_context_length,
        rope_yarn_log_multiplier,
        rope_yarn_beta_fast,
        rope_yarn_beta_slow,
        rope_dimension_count_swa,
        head_dim_swa,
        rope_theta_swa,
        sliding_window,
        sliding_window_pattern,
        final_logit_softcapping,
        full_attention_interval,
        head_count_kv_arr,
        feed_forward_length_arr,
        attention_scale,
        expert_weights_scale,
        expert_weights_norm,
        expert_gating_func,
        shared_expert_count,
        kv_lora_rank,
        q_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        first_k_dense_replace,
        moe_layer_freq,
        expert_group_count,
        output_lora_rank,
        output_group_count,
        compress_ratios,
        indexer_head_count,
        indexer_head_dim,
        indexer_top_k,
        compress_rope_freq_base,
        hyperconnection_count,
        hyperconnection_sinkhorn_iterations,
        hyperconnection_epsilon,
        hash_layer_count,
        kda_head_dim,
        kda_gate_lower_bound,
        indexer_kpool,
        indexer_kpool_select_tail,
        swiglu_clamp_exp,
        swiglu_clamp_shexp,
    })
}

// Maya's published flat typed keys, not the embedded JSON or stale quantization
// label, are authoritative. Normalize names only; retain architecture identity.
fn parse_deepseek41_metadata(md: &HashMap<String, gguf_file::Value>) -> Result<GgufMetadata> {
    use gguf_file::Value;
    if md.contains_key("deepseek41.ple.layers") {
        return parse_coder_metadata(md);
    }
    let mut canonical = HashMap::from([(
        "general.architecture".into(),
        Value::String("deepseek41".into()),
    )]);
    for (flat, target) in [
        ("num_hidden_layers", "block_count"),
        ("hidden_size", "embedding_length"),
        ("num_attention_heads", "attention.head_count"),
        ("num_key_value_heads", "attention.head_count_kv"),
        ("head_dim", "attention.key_length"),
        ("moe_intermediate_size", "feed_forward_length"),
        ("rms_norm_eps", "attention.layer_norm_rms_epsilon"),
        ("rope_theta", "rope.freq_base"),
        ("max_position_embeddings", "context_length"),
        ("n_routed_experts", "expert_count"),
        ("num_experts_per_tok", "expert_used_count"),
        ("moe_intermediate_size", "expert_feed_forward_length"),
        ("q_lora_rank", "attention.q_lora_rank"),
        ("qk_rope_head_dim", "rope.dimension_count"),
        ("o_lora_rank", "attention.output_lora_rank"),
        ("o_groups", "attention.output_group_count"),
        ("compress_ratios", "attention.compress_ratios"),
        ("index_n_heads", "attention.indexer.head_count"),
        ("index_head_dim", "attention.indexer.key_length"),
        ("index_topk", "attention.indexer.top_k"),
        ("compress_rope_theta", "attention.compress_rope_freq_base"),
        ("hc_mult", "hyper_connection.count"),
        ("hc_sinkhorn_iters", "hyper_connection.sinkhorn_iterations"),
        ("hc_eps", "hyper_connection.epsilon"),
        ("n_shared_experts", "expert_shared_count"),
        ("routed_scaling_factor", "expert_weights_scale"),
        ("norm_topk_prob", "expert_weights_norm"),
        ("rope_scaling.factor", "rope.scaling.factor"),
        ("rope_scaling.beta_fast", "rope.scaling.yarn_beta_fast"),
        ("rope_scaling.beta_slow", "rope.scaling.yarn_beta_slow"),
        ("sliding_window", "attention.sliding_window"),
        ("vocab_size", "vocab_size"),
    ] {
        let key = format!("deepseek41.{flat}");
        let value = md
            .get(&key)
            .with_context(|| format!("missing GGUF metadata key: {key}"))?;
        // Maya stores theta as U32, unlike standard GGUF's float freq_base.
        let value = if matches!(flat, "rope_theta" | "compress_rope_theta") {
            match value {
                Value::U32(value) => Value::F64(*value as f64),
                Value::U64(value) if *value <= (1u64 << 53) => Value::F64(*value as f64),
                Value::F32(_) | Value::F64(_) => value.clone(),
                _ => anyhow::bail!("invalid GGUF metadata key {key}: expected numeric theta"),
            }
        } else {
            value.clone()
        };
        canonical.insert(format!("deepseek41.{target}"), value);
    }
    // The released header encodes original context as an integral F32.
    anyhow::ensure!(
        md.contains_key("deepseek41.rope_scaling.original_max_position_embeddings"),
        "missing DeepSeek V4.1 original RoPE context"
    );
    let original = md_get_f64(
        md,
        "deepseek41.rope_scaling.original_max_position_embeddings",
        65536.,
    )?;
    anyhow::ensure!(
        original == 65536.,
        "unsupported DeepSeek V4.1 original RoPE context"
    );
    canonical.insert(
        "deepseek41.rope.scaling.original_context_length".into(),
        Value::U32(original as u32),
    );
    let mut parsed = parse_canonical_metadata(&canonical)?;
    anyhow::ensure!(
        parsed.num_layers == 40
            && parsed.num_kv_heads == 1
            && parsed.hyperconnection_count == 4
            && parsed.hyperconnection_sinkhorn_iterations == 20
            && parsed.expert_count > 0
            && parsed.expert_used_count > 0
            && parsed.expert_used_count <= parsed.expert_count
            && parsed.output_group_count > 0
            && parsed
                .num_attention_heads
                .is_multiple_of(parsed.output_group_count)
            && parsed.output_lora_rank > 0
            && parsed.q_lora_rank > 0
            && parsed.expert_ffn_dim > 0
            && parsed.shared_expert_count == 1
            && parsed.indexer_head_count > 0
            && parsed.indexer_top_k > 0,
        "unsupported DeepSeek V4.1 native layout"
    );
    // The native packed layout is bounded by Maya's geometry. Smaller widths
    // support isolated fixtures; larger/unknown layouts are not an allocation contract.
    anyhow::ensure!(
        parsed.hidden_dim <= 5120
            && parsed.hidden_dim.is_multiple_of(256)
            && parsed.expert_ffn_dim <= 2304
            && parsed.expert_ffn_dim.is_multiple_of(256)
            && parsed.num_attention_heads <= 64
            && parsed.head_dim <= 512
            && parsed.expert_count <= 384
            && parsed.output_group_count <= 8
            && parsed.output_lora_rank <= 1024
            && parsed.q_lora_rank <= 1280
            && parsed.indexer_head_count <= 32
            && parsed.indexer_head_dim <= 128
            && parsed.indexer_top_k == 512
            && parsed.context_length <= 1048576,
        "unsupported DeepSeek V4.1 native dimensions"
    );
    validate_rope_dimensions(
        parsed.qk_rope_head_dim,
        parsed.head_dim,
        "deepseek41.qk_rope_head_dim",
    )?;
    anyhow::ensure!(
        parsed.indexer_head_dim >= parsed.qk_rope_head_dim,
        "invalid DeepSeek V4.1 index dimensions"
    );
    anyhow::ensure!(
        parsed.compress_ratios == [vec![0; 2], vec![2; 18], vec![1; 20]].concat(),
        "unsupported DeepSeek V4.1 compression schedule"
    );
    for (key, expected) in [
        ("kv_source_layer_ids", vec![2, 8, 14, 20]),
        ("index_source_layer_ids", vec![2, 8, 14, 20, 24, 28, 32, 36]),
        ("engram.layer_ids", vec![1, 14]),
        ("engram.rows", vec![384006168, 384016682]),
    ] {
        anyhow::ensure!(
            md_get_usize_array(md, &format!("deepseek41.{key}"))? == expected,
            "unsupported DeepSeek V4.1 {key} schedule"
        );
    }
    for (key, expected) in [
        ("candidate_source_layer_id", 20),
        ("candidate_topk_blocks", 2048),
        ("candidate_block_size", 8),
    ] {
        anyhow::ensure!(
            md_get_usize(md, &format!("deepseek41.{key}"))? == expected,
            "unsupported DeepSeek V4.1 {key}"
        );
    }
    for (key, expected) in [
        ("scoring_func", "sqrtsoftplus"),
        ("hidden_act", "silu"),
        ("topk_method", "noaux_tc"),
        ("engram.encoding", "e4m3_e8m0_32_row264"),
    ] {
        anyhow::ensure!(
            md_get_str(md, &format!("deepseek41.{key}"))? == expected,
            "unsupported DeepSeek V4.1 {key}"
        );
    }
    let epsilon = md_get_f64(md, "deepseek41.hc_eps", 1e-6)?;
    let limit = md_get_f64(md, "deepseek41.swiglu_limit", 10.)?;
    anyhow::ensure!(
        (epsilon - 1e-6).abs() < 1e-12
            && limit == 10.
            && parsed.sliding_window == 128
            && parsed.rope_scaling_factor == 16.
            && parsed.rope_yarn_beta_fast == 32.
            && parsed.rope_yarn_beta_slow == 1.
            && parsed.rope_theta == 10000.
            && parsed.compress_rope_freq_base == 160000.
            && parsed.expert_weights_scale == 1.5
            && parsed.expert_weights_norm,
        "unsupported DeepSeek V4.1 numeric layout"
    );
    parsed.swiglu_clamp_exp = vec![limit; parsed.num_layers];
    parsed.swiglu_clamp_shexp = parsed.swiglu_clamp_exp.clone();
    Ok(parsed)
}

fn parse_coder_metadata(md: &HashMap<String, gguf_file::Value>) -> Result<GgufMetadata> {
    use gguf_file::Value;
    let mut canonical = md.clone();
    let ratios = md_get_usize_array(md, "deepseek41.attention.compress_ratios")?;
    anyhow::ensure!(
        ratios == [vec![0; 2], vec![2; 18], vec![1; 20], vec![0; 3]].concat(),
        "unsupported Coder compression schedule"
    );
    canonical.insert(
        "deepseek41.attention.compress_ratios".into(),
        Value::Array(ratios[..40].iter().map(|&v| Value::U32(v as u32)).collect()),
    );
    canonical.insert("deepseek41.feed_forward_length".into(), Value::U32(2304));
    let parsed = parse_canonical_metadata(&canonical)?;
    anyhow::ensure!(
        parsed.expert_count == 192
            && parsed.expert_used_count == 6
            && parsed.expert_ffn_dim == 2304
            && parsed.context_length == 1048576,
        "unsupported Coder expert/context geometry"
    );
    for (key, expected) in [
        ("attention.kv_source_layers", vec![2, 8, 14, 20]),
        (
            "attention.indexer.source_layers",
            vec![2, 8, 14, 20, 24, 28, 32, 36],
        ),
        ("ple.layers", vec![1, 14]),
    ] {
        anyhow::ensure!(
            md_get_usize_array(md, &format!("deepseek41.{key}"))? == expected,
            "unsupported Coder {key}"
        );
    }
    for (key, expected) in [
        ("attention.indexer.candidate_source_layer", 20),
        ("attention.indexer.candidate_top_k_blocks", 2048),
        ("attention.indexer.candidate_block_size", 8),
        ("expert_gating_func", 4),
    ] {
        anyhow::ensure!(
            md_get_usize(md, &format!("deepseek41.{key}"))? == expected,
            "unsupported Coder {key}"
        );
    }
    anyhow::ensure!(
        parsed.swiglu_clamp_exp == vec![10.; 40] && parsed.swiglu_clamp_shexp == vec![10.; 40],
        "unsupported Coder SwiGLU limits"
    );
    Ok(parsed)
}

/// Read normalized model metadata from a GGUF file without loading weights.
pub fn read_gguf_metadata(path: &Path) -> Result<GgufMetadata> {
    if deepseek4_gguf::is_deepseek4(path).unwrap_or(false) {
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open GGUF: {}", path.display()))?;
        let content = RawGgufContent::read(&mut file)
            .with_context(|| format!("Failed to parse DeepSeek V4 GGUF: {}", path.display()))?;
        return parse_gguf_metadata(&content.metadata);
    }
    let (mut file, _) = open_validated_gguf(path)?;
    let content = gguf_file::Content::read(&mut file)
        .with_context(|| format!("Failed to parse GGUF header: {}", path.display()))?;
    parse_gguf_metadata(&content.metadata)
}

/// Read the GGUF metadata map, including files whose tensor table uses a
/// quant type that Candle does not know yet.
pub fn read_gguf_metadata_map(path: &Path) -> Result<HashMap<String, gguf_file::Value>> {
    let (mut file, _) = open_validated_gguf(path)?;
    match gguf_file::Content::read(&mut file) {
        Ok(content) => Ok(content.metadata),
        Err(candle_error) => {
            // Re-read the validated descriptor, not a fresh open of the path.
            let mut raw_file = file;
            RawGgufContent::read(&mut raw_file)
                .map(|content| content.metadata)
                .with_context(|| {
                    format!(
                        "Failed to parse GGUF metadata with Candle ({candle_error}) or the DINF reader: {}",
                        path.display()
                    )
                })
        }
    }
}

/// Return the number of executable transformer trunk layers advertised by a
/// GGUF metadata map.
///
/// GLM5-Next and some Qwen3.5-family converters include physical NextN/MTP
/// blocks in `<architecture>.block_count` and identify them with
/// `<architecture>.nextn_predict_layers`. Pipeline stages must use the
/// ordinary trunk count so that MTP tensors are never assigned to a stage
/// range. Other architectures retain their historical block-count behavior.
pub fn executable_layer_count(md: &HashMap<String, gguf_file::Value>) -> Result<usize> {
    let architecture = match md.get("general.architecture") {
        Some(gguf_file::Value::String(value)) => Some(value.as_str()),
        Some(value) => anyhow::bail!(
            "invalid GGUF metadata key general.architecture: expected a string, got {:?}",
            value.value_type()
        ),
        None => None,
    };
    if architecture == Some("deepseek41") {
        let key = if md.contains_key("deepseek41.ple.layers") {
            "deepseek41.block_count"
        } else {
            "deepseek41.num_hidden_layers"
        };
        let count = md_get_usize(md, key)?;
        anyhow::ensure!(count == 40, "unsupported DeepSeek V4.1 layer count");
        return Ok(count);
    }

    // Retain known prefixes for older files that omit general.architecture.
    let mut prefixes = architecture.into_iter().collect::<Vec<_>>();
    prefixes.extend([
        "llama",
        "mistral",
        "qwen2",
        "qwen3moe",
        "qwen35",
        "llama4",
        "deepseek4",
        "glm5next",
        "glm5-next",
        "nemotron_h",
        "nemotron_h_moe",
        "phi3",
        "gemma",
        "gemma4",
        "chatglm",
    ]);
    for prefix in prefixes {
        let key = format!("{prefix}.block_count");
        let physical = match md.get(&key) {
            Some(_) => md_get_usize(md, &key)?,
            None => continue,
        };
        validate_layer_count(physical, "GGUF physical layer count")?;

        if matches!(prefix, "glm5next" | "glm5-next" | "qwen35") {
            let nextn_key = format!("{prefix}.nextn_predict_layers");
            let nextn = md_get_optional_usize(md, &nextn_key)?.unwrap_or(0);
            anyhow::ensure!(
                nextn < physical,
                "{nextn_key} ({nextn}) must be less than physical block_count ({physical})"
            );
            let executable = physical - nextn;
            validate_layer_count(executable, "GGUF executable layer count")?;
            return Ok(executable);
        }
        return Ok(physical);
    }
    anyhow::bail!("Cannot determine layer count from GGUF metadata")
}

/// Determine whether a tensor name belongs to a given layer range.
///
/// GGUF tensor naming convention:
/// - `blk.{N}.attn_q.weight` — belongs to layer N
/// - `token_embd.weight` — embedding (head only)
/// - `output_norm.weight`, `output.weight` — output (tail only)
fn tensor_belongs_to_stage(name: &str, range: &LayerRange) -> bool {
    // Block/layer tensors: blk.{N}.*
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some(dot_pos) = rest.find('.')
            && let Ok(layer_idx) = rest[..dot_pos].parse::<usize>()
        {
            return range.contains(layer_idx);
        }
        return false;
    }

    // Embedding tensors — head needs token embeddings, and tail may need them
    // as a tied output head when output.weight is absent.
    if name == "token_embd.weight" || name.starts_with("token_embd.") {
        return range.is_head() || range.is_tail();
    }

    // V4 hyper-connection output tensors belong to the tail stage.
    if name == "output_hc_fn.weight"
        || name == "output_hc_base.weight"
        || name == "output_hc_scale.weight"
    {
        return range.is_tail();
    }

    // Output tensors — only for tail stage
    if name == "output_norm.weight"
        || name == "output.weight"
        || name.starts_with("output_norm.")
        || name.starts_with("output.")
    {
        return range.is_tail();
    }

    // Rope frequency tensors — needed by any stage that has attention layers
    if name == "rope_freqs.weight" || name.starts_with("rope_freqs") {
        return true;
    }

    // Unknown tensor — include it to be safe (usually tokenizer metadata tensors)
    false
}

fn qwen35_cpu_packed_embedding(
    architecture: &str,
    name: &str,
    dtype: GgmlDType,
    is_metal: bool,
) -> bool {
    // Candle's Metal Q4K gather performs part of the scale math in F16.
    // Keep this lookup packed on CPU and transfer only selected F32 rows.
    is_metal && architecture == "qwen35" && name == "token_embd.weight" && dtype == GgmlDType::Q4K
}

fn tensor_loads_on_cpu(architecture: &str, name: &str, dtype: GgmlDType, is_metal: bool) -> bool {
    name.contains("_exps.")
        || (matches!(architecture, "glm5next" | "glm5-next")
            && (name.ends_with(".attn_k_b.weight") || name.ends_with(".attn_v_b.weight")))
        || qwen35_cpu_packed_embedding(architecture, name, dtype, is_metal)
}

/// Given the first shard path, discover all shard paths on disk.
/// If the file doesn't match the split pattern, returns a single-element vec.
fn discover_shard_paths(first_shard: &Path) -> Result<Vec<PathBuf>> {
    let name = match first_shard.file_name().and_then(|n| n.to_str()) {
        Some(n) => n.to_string(),
        None => return Ok(vec![first_shard.to_path_buf()]),
    };

    // Match: ...-00001-of-NNNNN.gguf
    let suffix = "-00001-of-";
    let pos = match name.find(suffix) {
        Some(p) => p,
        None => return Ok(vec![first_shard.to_path_buf()]),
    };
    let after = &name[pos + suffix.len()..];
    let dot = match after.find('.') {
        Some(d) => d,
        None => return Ok(vec![first_shard.to_path_buf()]),
    };
    let n: usize = match after[..dot].parse() {
        Ok(n) if n >= 2 => n,
        _ => return Ok(vec![first_shard.to_path_buf()]),
    };
    anyhow::ensure!(
        n <= MAX_GGUF_SHARDS,
        "GGUF shard count ({n}) exceeds application limit ({MAX_GGUF_SHARDS})"
    );

    let dir = first_shard.parent().unwrap_or(Path::new("."));
    let prefix = &name[..pos + 1]; // up to '-'
    let ext = &after[dot..]; // ".gguf"
    let width = dot;

    let mut paths = Vec::new();
    paths
        .try_reserve_exact(n)
        .map_err(|error| anyhow::anyhow!("GGUF shard path allocation failed: {error}"))?;
    for i in 1..=n {
        paths.push(dir.join(format!(
            "{prefix}{:0>width$}-of-{:0>width$}{ext}",
            i,
            n,
            width = width
        )));
    }
    Ok(paths)
}

/// Reject ambiguous tensor ownership before loading any stage weights. Check
/// every tensor, including layers outside this stage's assigned range, so
/// streaming, whole-stage loading and admission agree on the same model.
fn register_shard_tensors<'a>(
    owners: &mut HashMap<String, usize>,
    shard_index: usize,
    names: impl Iterator<Item = &'a String>,
) -> Result<()> {
    for name in names {
        match owners.entry(name.clone()) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(shard_index);
            }
            std::collections::hash_map::Entry::Occupied(entry) => {
                anyhow::bail!(
                    "duplicate GGUF tensor {name} in shards {} and {}",
                    entry.get() + 1,
                    shard_index + 1,
                );
            }
        }
    }
    Ok(())
}

/// Check the split manifest before loading weights. Older files may omit the
/// descriptor, and gguf-split's merge command leaves split.count set to zero.
/// Later shards need not repeat the model metadata.
struct ShardMetadata {
    shards: usize,
    tensors: usize,
    declared_tensors: Option<usize>,
    require_index: bool,
    architecture: Option<String>,
}

impl ShardMetadata {
    fn new(shards: usize) -> Self {
        Self {
            shards,
            tensors: 0,
            declared_tensors: None,
            require_index: false,
            architecture: None,
        }
    }

    fn record(
        &mut self,
        index: usize,
        metadata: &HashMap<String, gguf_file::Value>,
        tensors: usize,
    ) -> Result<()> {
        let count = split_integer(metadata, "split.count")?;
        if let Some(count) = count {
            anyhow::ensure!(
                count == self.shards || (self.shards == 1 && count == 0),
                "GGUF split.count ({count}) differs from discovered shard count ({})",
                self.shards
            );
        }
        if index == 0 {
            self.require_index = count.is_some_and(|count| count > 1);
            self.architecture = Some(md_get_str(metadata, "general.architecture")?);
        } else if metadata.contains_key("general.architecture") {
            anyhow::ensure!(
                Some(md_get_str(metadata, "general.architecture")?) == self.architecture,
                "GGUF shard {} architecture differs from the first shard",
                index + 1
            );
        }
        let declared_index = split_integer(metadata, "split.no")?;
        anyhow::ensure!(
            declared_index.is_some() || !self.require_index,
            "missing split.no in GGUF shard {}",
            index + 1
        );
        if let Some(declared_index) = declared_index {
            anyhow::ensure!(
                declared_index == index,
                "GGUF split.no ({declared_index}) differs from shard index ({index})"
            );
        }
        if let Some(declared) = split_integer(metadata, "split.tensors.count")? {
            anyhow::ensure!(
                self.declared_tensors
                    .is_none_or(|previous| previous == declared),
                "GGUF split.tensors.count differs between shards"
            );
            self.declared_tensors = Some(declared);
        }
        self.tensors = self
            .tensors
            .checked_add(tensors)
            .context("GGUF split tensor count overflow")?;
        Ok(())
    }

    fn finish(&self) -> Result<()> {
        anyhow::ensure!(
            self.declared_tensors
                .is_none_or(|declared| declared == self.tensors),
            "GGUF split.tensors.count ({:?}) differs from actual tensor count ({})",
            self.declared_tensors,
            self.tensors
        );
        Ok(())
    }
}

fn split_integer(metadata: &HashMap<String, gguf_file::Value>, key: &str) -> Result<Option<usize>> {
    use gguf_file::Value;
    let value = match metadata.get(key) {
        None => return Ok(None),
        Some(Value::U8(value)) => u64::from(*value),
        Some(Value::U16(value)) => u64::from(*value),
        Some(Value::U32(value)) => u64::from(*value),
        Some(Value::U64(value)) => *value,
        Some(Value::I8(value)) => u64::try_from(*value).context("negative GGUF split integer")?,
        Some(Value::I16(value)) => u64::try_from(*value).context("negative GGUF split integer")?,
        Some(Value::I32(value)) => u64::try_from(*value).context("negative GGUF split integer")?,
        Some(Value::I64(value)) => u64::try_from(*value).context("negative GGUF split integer")?,
        Some(value) => anyhow::bail!(
            "invalid GGUF metadata key {key}: expected an integer, got {:?}",
            value.value_type()
        ),
    };
    usize::try_from(value)
        .map(Some)
        .with_context(|| format!("GGUF metadata key does not fit usize: {key}"))
}

fn validate_embedding_vocabulary(metadata: &GgufMetadata, shape: &[usize]) -> Result<()> {
    anyhow::ensure!(
        shape.len() == 2 && shape[0] >= metadata.vocab_size,
        "token_embd.weight shape {shape:?} cannot hold the declared vocabulary of {} tokens",
        metadata.vocab_size
    );
    Ok(())
}

/// Load GGUF tensors for a specific pipeline stage.
///
/// Only loads tensors belonging to the specified layer range, plus embedding
/// (if head) and output norm/lm_head (if tail). This allows each stage to
/// hold only its portion of the model in memory.
///
/// Supports split GGUF files: if `gguf_path` matches the `-00001-of-NNNNN.gguf`
/// pattern, all N shards are opened and tensors loaded from the correct shard.
pub fn load_stage_tensors(
    gguf_path: &Path,
    range: LayerRange,
    device: &Device,
) -> Result<StageTensors> {
    range.validate()?;
    let shard_paths = discover_shard_paths(gguf_path)?;

    let architecture = read_model_architecture(&shard_paths[0])?;
    if architecture == "deepseek41" {
        anyhow::ensure!(
            range.is_solo(),
            "DeepSeek V4.1 requires a full solo layer range; partial stages are unsupported"
        );
        anyhow::ensure!(
            shard_paths.len() == 1
                || (shard_paths.len() == 6 && crate::gsq_coder::is_entry(&shard_paths[0])),
            "DeepSeek V4.1 split GGUF is unsupported"
        );
        return load_stage_tensors_deepseek41(&shard_paths[0], range, device);
    }
    // Candle 0.11 rejects the V4 IQ and MXFP4 dtype IDs while reading the
    // header. Route this family through the byte-preserving reader first.
    if architecture == "deepseek4" {
        return load_stage_tensors_deepseek4(&shard_paths, range, device);
    }

    if shard_paths.len() == 1 {
        return load_stage_tensors_single(gguf_path, range, device);
    }

    tracing::info!("Loading split GGUF model ({} shards)", shard_paths.len());

    // Open all shards and read their headers
    let mut shard_files: Vec<(std::fs::File, gguf_file::Content)> = Vec::new();
    let mut aggregate_header_bytes = 0u64;
    let mut tensor_owners = HashMap::new();
    let mut shard_metadata = ShardMetadata::new(shard_paths.len());
    for (shard_index, path) in shard_paths.iter().enumerate() {
        let (mut f, stats) = open_validated_gguf(path)?;
        let content = gguf_file::Content::read(&mut f)
            .with_context(|| format!("Failed to parse shard header: {}", path.display()))?;
        account_split_header_bytes(
            &mut aggregate_header_bytes,
            stats.parsed_bytes(content.tensor_data_offset),
        )?;
        register_shard_tensors(&mut tensor_owners, shard_index, content.tensor_infos.keys())?;
        shard_metadata.record(shard_index, &content.metadata, content.tensor_infos.len())?;
        shard_files.push((f, content));
    }
    shard_metadata.finish()?;

    // The first shard owns model metadata; later shards may carry only split keys.
    let metadata = parse_gguf_metadata(&shard_files[0].1.metadata)?;
    for (_, content) in &shard_files {
        if let Some(info) = content.tensor_infos.get("token_embd.weight") {
            validate_embedding_vocabulary(&metadata, info.shape.dims())?;
        }
    }

    if range.total_layers != metadata.num_layers {
        anyhow::bail!(
            "Layer range total_layers ({}) doesn't match model num_layers ({})",
            range.total_layers,
            metadata.num_layers
        );
    }

    // Build tensor→shard index map
    let mut tensor_shard: Vec<(String, usize)> = Vec::new();
    for (idx, (_, content)) in shard_files.iter().enumerate() {
        for name in content.tensor_infos.keys() {
            tensor_shard.push((name.clone(), idx));
        }
    }

    // Load tensors for this stage's layer range
    let mut tensors = HashMap::new();
    let has_output_weight = tensor_owners.contains_key("output.weight");
    for (name, shard_idx) in &tensor_shard {
        if !tensor_belongs_to_stage(name, &range) {
            continue;
        }
        if !range.is_head() && has_output_weight && is_token_embedding(name) {
            continue;
        }
        let (ref mut file, ref content) = shard_files[*shard_idx];
        let load_device = if tensor_loads_on_cpu(
            &metadata.architecture,
            name,
            content.tensor_infos[name].ggml_dtype,
            device.is_metal(),
        ) {
            &Device::Cpu
        } else {
            device
        };
        let tensor = content.tensor(file, name, load_device).with_context(|| {
            format!("Failed to load tensor {name} from shard {}", shard_idx + 1)
        })?;
        tensors.insert(name.clone(), Arc::new(tensor));
    }

    tracing::info!(
        "Loaded {} tensors for layers [{}, {}) from {} shards (arch: {}, {} total layers)",
        tensors.len(),
        range.start,
        range.end,
        shard_paths.len(),
        metadata.architecture,
        metadata.num_layers,
    );

    Ok(StageTensors {
        range,
        metadata,
        tensors,
        raw_tensors: HashMap::new(),
        raw_source: None,
        device: device.clone(),
    })
}

/// Reusable GGUF header/index reader for architectures which materialize a
/// stage incrementally. The shard files and tensor tables stay open, while
/// each call to [`Self::load_range`] owns only the requested range's tensors.
/// This avoids reparsing every shard once per streamed layer without bringing
/// the CPU QTensor source pool back to stage scope.
pub(crate) struct StageTensorLoader {
    shard_files: Vec<(std::fs::File, gguf_file::Content)>,
    metadata: GgufMetadata,
    has_output_weight: bool,
}

/// Keep admission accounting aligned with the per-layer loader dispatch.
pub(crate) fn uses_streaming_sources(architecture: &str) -> bool {
    matches!(
        architecture,
        "glm5next" | "glm5-next" | "minimax-m2" | "muse-spark"
    )
}

impl StageTensorLoader {
    pub(crate) fn open(gguf_path: &Path) -> Result<Self> {
        let shard_paths = discover_shard_paths(gguf_path)?;
        let mut shard_files: Vec<(std::fs::File, gguf_file::Content)> =
            Vec::with_capacity(shard_paths.len());
        let mut aggregate_header_bytes = 0u64;
        let mut tensor_owners = HashMap::new();
        let mut shard_metadata = ShardMetadata::new(shard_paths.len());
        for (shard_index, path) in shard_paths.iter().enumerate() {
            let (mut file, stats) = open_validated_gguf(path)?;
            let content = gguf_file::Content::read(&mut file)
                .with_context(|| format!("Failed to parse GGUF header: {}", path.display()))?;
            account_split_header_bytes(
                &mut aggregate_header_bytes,
                stats.parsed_bytes(content.tensor_data_offset),
            )?;
            register_shard_tensors(&mut tensor_owners, shard_index, content.tensor_infos.keys())?;
            shard_metadata.record(shard_index, &content.metadata, content.tensor_infos.len())?;
            shard_files.push((file, content));
        }
        shard_metadata.finish()?;

        let metadata = parse_gguf_metadata(&shard_files[0].1.metadata)?;
        for (_, content) in &shard_files {
            if let Some(info) = content.tensor_infos.get("token_embd.weight") {
                validate_embedding_vocabulary(&metadata, info.shape.dims())?;
            }
        }
        let has_output_weight = shard_files
            .iter()
            .any(|(_, content)| content.tensor_infos.contains_key("output.weight"));
        Ok(Self {
            shard_files,
            metadata,
            has_output_weight,
        })
    }

    pub(crate) fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    pub(crate) fn load_range(
        &mut self,
        range: LayerRange,
        device: &Device,
    ) -> Result<StageTensors> {
        range.validate()?;
        anyhow::ensure!(
            range.total_layers == self.metadata.num_layers,
            "Layer range total_layers ({}) doesn't match model num_layers ({})",
            range.total_layers,
            self.metadata.num_layers
        );

        let mut tensors = HashMap::new();
        for (shard_idx, (file, content)) in self.shard_files.iter_mut().enumerate() {
            let names = content.tensor_infos.keys().cloned().collect::<Vec<_>>();
            for name in names {
                if !tensor_belongs_to_stage(&name, &range) {
                    continue;
                }
                // An untied output head means the tail never retains the
                // token embedding. Avoid materializing that large tensor just
                // to discard it in `from_stage_tensors`.
                if !range.is_head()
                    && range.is_tail()
                    && self.has_output_weight
                    && is_token_embedding(&name)
                {
                    continue;
                }
                let load_device = if tensor_loads_on_cpu(
                    &self.metadata.architecture,
                    &name,
                    content.tensor_infos[&name].ggml_dtype,
                    device.is_metal(),
                ) {
                    &Device::Cpu
                } else {
                    device
                };
                let tensor = content.tensor(file, &name, load_device).with_context(|| {
                    format!("Failed to load tensor {name} from shard {}", shard_idx + 1)
                })?;
                tensors.insert(name, Arc::new(tensor));
            }
        }

        Ok(StageTensors {
            range,
            metadata: self.metadata.clone(),
            tensors,
            raw_tensors: HashMap::new(),
            raw_source: None,
            device: device.clone(),
        })
    }
}

fn validate_deepseek41_tensors(content: &RawGgufContent, metadata: &GgufMetadata) -> Result<()> {
    use deepseek4_gguf::DeepSeek4DType as T;
    if content.metadata.contains_key("deepseek41.ple.layers") {
        anyhow::ensure!(metadata.expert_count == 192, "Coder expert count mismatch");
        return crate::architectures::deepseek41::DeepSeek41Stage::validate_source(content);
    }
    anyhow::ensure!(
        md_get_usize_or(&content.metadata, "split.count", 1)? == 1
            && md_get_usize_or(&content.metadata, "split.no", 0)? == 0,
        "DeepSeek V4.1 split GGUF is unsupported"
    );
    for (name, info) in &content.tensor_infos {
        if let Some(layer) = name
            .strip_prefix("blk.")
            .and_then(|rest| rest.split('.').next())
        {
            anyhow::ensure!(
                layer
                    .parse::<usize>()
                    .is_ok_and(|layer| layer < metadata.num_layers),
                "DeepSeek V4.1 tensor outside layer range: {name}"
            );
        }
        if name.ends_with(".engram_embd.weight") {
            // The exact disk-only table descriptor was already checked in preflight.
            anyhow::ensure!(info.dtype == T::I8, "unsupported Engram dtype");
        } else if name.ends_with("_exps.weight") {
            anyhow::ensure!(
                info.dtype == T::IQ1S
                    && info.shape.len() == 3
                    && info.shape[0] == metadata.expert_count,
                "unsupported DeepSeek V4.1 expert bank: {name}"
            );
            let expected = if name.ends_with("ffn_down_exps.weight") {
                [metadata.hidden_dim, metadata.expert_ffn_dim]
            } else if name.ends_with("ffn_gate_exps.weight") || name.ends_with("ffn_up_exps.weight")
            {
                [metadata.expert_ffn_dim, metadata.hidden_dim]
            } else {
                anyhow::bail!("unsupported DeepSeek V4.1 expert bank name: {name}");
            };
            anyhow::ensure!(
                info.shape[1..] == expected,
                "invalid DeepSeek V4.1 expert bank shape: {name}"
            );
            anyhow::ensure!(
                info.dtype.byte_len(
                    expected[0]
                        .checked_mul(expected[1])
                        .context("expert shape overflow")?
                )? <= deepseek4_gguf::MAX_SOURCE_READ_BYTES,
                "DeepSeek V4.1 expert exceeds bounded source read"
            );
        } else {
            anyhow::ensure!(
                matches!(info.dtype, T::Q8_0 | T::F16 | T::F32),
                "unsupported DeepSeek V4.1 resident tensor dtype: {name}"
            );
        }
    }
    Ok(())
}

/// Request-side native identity, never inferred from a GGUF header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpectedNativeIdentity {
    /// The pinned DeepSeek V4.1 Flash 1017 Maya-XS model.
    Maya,
    /// The pinned six-part DeepSeek V4.1 GSQ-RCO Coder model.
    GsqRcoCoder,
}

/// A verified native GGUF source. The integrity check and the parsed
/// metadata/weights share one retained file descriptor. Construction runs a full
/// SHA-256 pass on a cache miss, or validates a local verification receipt,
/// before parsing any header. Prepared reuse never reopens the launch path.
pub struct PreparedNativeSource {
    source: Arc<RawGgufSource>,
    metadata: GgufMetadata,
}

impl PreparedNativeSource {
    /// Open and verify a local source against the requested model identity.
    pub fn open(path: &Path, expected: ExpectedNativeIdentity) -> Result<Self> {
        let source = match expected {
            ExpectedNativeIdentity::Maya => RawGgufSource::open_maya(path)?,
            ExpectedNativeIdentity::GsqRcoCoder => RawGgufSource::open_coder(path)?,
        };
        Self::from_source(Arc::new(source))
    }

    fn from_source(source: Arc<RawGgufSource>) -> Result<Self> {
        source.require_verified_native()?;
        let metadata = parse_gguf_metadata(&source.content().metadata)?;
        anyhow::ensure!(
            metadata.architecture == "deepseek41",
            "canonical Maya requires deepseek41"
        );
        validate_deepseek41_tensors(source.content(), &metadata)?;
        Ok(Self { source, metadata })
    }

    /// Return parsed metadata after checking the retained source's verification.
    pub fn metadata(&self) -> Result<&GgufMetadata> {
        self.source.require_verified_native()?;
        Ok(&self.metadata)
    }

    /// Read-only verification counters for the retained source; never grants or changes trust.
    pub fn verification_report(&self) -> Result<serde_json::Value> {
        self.source.require_verified_native()?;
        Ok(serde_json::json!({
            "identity": self.source.file_identity(),
            "verification": self.source.verification(),
        }))
    }

    /// Build a tokenizer from the verified source's metadata.
    pub fn tokenizer(&self) -> Result<crate::tokenizer::GgufTokenizer> {
        self.source.require_verified_native()?;
        crate::tokenizer::GgufTokenizer::from_metadata(&self.source.content().metadata)
    }

    /// Estimate stage memory for the verified source on a Metal device.
    pub fn memory_profile(&self, model_id: &str, device: &Device) -> Result<StageMemoryProfile> {
        anyhow::ensure!(
            device.is_metal(),
            "DeepSeek V4.1 native profiling currently requires Metal"
        );
        self.memory_profile_components(model_id, device)
    }

    fn memory_profile_components(
        &self,
        model_id: &str,
        device: &Device,
    ) -> Result<StageMemoryProfile> {
        self.source.require_verified_native()?;
        deepseek41_memory_profile(self.source.content(), model_id, &self.metadata, device)
    }

    pub(crate) fn load_stage_tensors(
        &self,
        range: LayerRange,
        device: &Device,
    ) -> Result<StageTensors> {
        load_stage_tensors_from_maya_source(Arc::clone(&self.source), range, device)
    }
}

fn load_stage_tensors_deepseek41(
    path: &Path,
    range: LayerRange,
    device: &Device,
) -> Result<StageTensors> {
    anyhow::ensure!(
        device.is_metal(),
        "DeepSeek V4.1 native loading currently requires Metal"
    );
    let source = Arc::new(
        if path.file_name().and_then(|s| s.to_str())
            == Some(format!("{}-00001-of-00006.gguf", crate::gsq_coder::PREFIX).as_str())
        {
            RawGgufSource::open_coder(path)?
        } else {
            RawGgufSource::open_maya(path)?
        },
    );
    load_stage_tensors_from_maya_source(source, range, device)
}

#[cfg(all(test, target_os = "macos"))]
thread_local! {
    pub(crate) static MAYA_UPLOAD_PEAK: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn load_stage_tensors_from_maya_source(
    source: Arc<RawGgufSource>,
    range: LayerRange,
    device: &Device,
) -> Result<StageTensors> {
    source.require_verified_native()?;
    anyhow::ensure!(
        device.is_metal(),
        "DeepSeek V4.1 native loading currently requires Metal"
    );
    let metadata = parse_gguf_metadata(&source.content().metadata)?;
    anyhow::ensure!(
        range.is_solo() && range.total_layers == metadata.num_layers,
        "DeepSeek V4.1 requires all 40 layers in one solo stage"
    );
    validate_deepseek41_tensors(source.content(), &metadata)?;
    crate::architectures::deepseek41::DeepSeek41Stage::validate_stage_config(
        &metadata, &range, false, device,
    )?;
    crate::architectures::deepseek41::DeepSeek41Stage::validate_source(source.content())?;
    let mut raw_tensors = HashMap::new();
    for name in source.content().tensor_infos.keys() {
        if name.ends_with("_exps.weight")
            || name.ends_with(".engram_embd.weight")
            || name == "per_layer_token_embd.weight"
        {
            continue;
        }
        let raw = source.load_resident_tensor(name, device)?;
        #[cfg(target_os = "macos")]
        if source.content().tensor_infos[name].dtype == deepseek4_gguf::DeepSeek4DType::Q8_0 {
            deepseek4_gguf::verify_maya_q8_upload(
                &raw,
                &source.content().tensor_infos[name],
                device,
            )
            .with_context(|| format!("ordinary Maya Q8 upload {name}"))?;
        }
        #[cfg(all(test, target_os = "macos"))]
        MAYA_UPLOAD_PEAK.with(|peak| {
            peak.set(
                peak.get().max(
                    device
                        .as_metal_device()
                        .unwrap()
                        .device()
                        .current_allocated_size() as u64,
                ),
            )
        });
        raw_tensors.insert(name.clone(), Arc::new(raw));
    }
    Ok(StageTensors {
        range,
        metadata,
        tensors: HashMap::new(),
        raw_tensors,
        raw_source: Some(source),
        device: device.clone(),
    })
}

fn estimate_deepseek41_profile(
    path: &Path,
    model_id: &str,
    total_layers: usize,
    metadata: &GgufMetadata,
    device: &Device,
) -> Result<StageMemoryProfile> {
    anyhow::ensure!(
        device.is_metal(),
        "DeepSeek V4.1 native profiling currently requires Metal"
    );
    anyhow::ensure!(
        total_layers == metadata.num_layers,
        "DeepSeek V4.1 profile requires all 40 layers"
    );
    let source = if crate::gsq_coder::is_entry(path) {
        RawGgufSource::open_coder(path)?
    } else {
        RawGgufSource::new(std::fs::File::open(path)?)?
    };
    validate_deepseek41_tensors(source.content(), metadata)?;
    deepseek41_memory_profile(source.content(), model_id, metadata, device)
}

fn deepseek41_memory_profile(
    content: &RawGgufContent,
    model_id: &str,
    metadata: &GgufMetadata,
    device: &Device,
) -> Result<StageMemoryProfile> {
    validate_layer_count(metadata.num_layers, "DeepSeek V4.1 profile layers")?;
    let mut profile = StageMemoryProfile {
        model_id: model_id.into(),
        total_layers: metadata.num_layers,
        layer_gpu_bytes: vec![0; metadata.num_layers],
        layer_gpu_transient_bytes: vec![0; metadata.num_layers],
        layer_cpu_bytes: vec![0; metadata.num_layers],
        layer_cpu_transient_bytes: vec![0; metadata.num_layers],
        // Full-solo embedding/output storage is already included in common.
        // Explicit boundaries preserve exact CPU accounting through protocol conversion.
        head_embedding_gpu_bytes: Some(0),
        tail_embedding_gpu_bytes: Some(0),
        head_embedding_cpu_bytes: Some(0),
        tail_embedding_cpu_bytes: Some(0),
        head_embedding_cpu_transient_bytes: Some(0),
        tail_embedding_cpu_transient_bytes: Some(0),
        common_gpu_bytes: DEEPSEEK41_EXPERT_CACHE_BYTES,
        // Source header ownership and independent graph/cache host objects are
        // separate. This bounded allowance never includes disk weight payloads.
        common_cpu_bytes: deepseek41_header_memory_bytes(content)
            + crate::architectures::deepseek41::experts::DEEPSEEK41_MODEL_HOST_BYTES,
        common_cpu_transient_bytes: deepseek4_gguf::MAX_SOURCE_READ_BYTES as u64,
        ..Default::default()
    };
    // Bounded Engram code/scale lookup, IQ1 codebook, and two RoPE frequency
    // families (positions are generated on demand, not a 1M-token table).
    profile.common_gpu_bytes += device_allocation_bytes(128 * 1024 + 2048 * 8 + 2 * 32 * 4, device);
    // HC's plain RMS is a custom op, not an allocated all-ones weight.
    for (name, info) in &content.tensor_infos {
        if name.ends_with("_exps.weight")
            || name.ends_with(".engram_embd.weight")
            || name == "per_layer_token_embd.weight"
        {
            continue;
        }
        let contribution = deepseek4_tensor_memory_accounting(name, info, metadata, device)?;
        // Only verified Maya ordinary Q8 uses exact with_data uploads. Dense
        // copies, construction overlap and every other dtype stay conservative.
        let raw = info.byte_len()? as u64;
        let raw_charge = if info.dtype == deepseek4_gguf::DeepSeek4DType::Q8_0 && device.is_metal()
        {
            deepseek4_gguf::maya_q8_upload_charge(raw)?
        } else {
            device_allocation_bytes(raw, device)
        };
        let retained = contribution.accelerator_bytes - raw + raw_charge;
        profile.common_cpu_transient_bytes = profile
            .common_cpu_transient_bytes
            .max(contribution.cpu_upload_bytes);
        if let Some(layer) = name
            .strip_prefix("blk.")
            .and_then(|rest| rest.split('.').next())
            .and_then(|s| s.parse::<usize>().ok())
        {
            anyhow::ensure!(
                layer < metadata.num_layers,
                "Maya profile tensor outside layer range: {name}"
            );
            profile.layer_gpu_bytes[layer] += retained;
            profile.layer_gpu_transient_bytes[layer] = profile.layer_gpu_transient_bytes[layer]
                .max(contribution.accelerator_transient_bytes);
        } else {
            // Full solo only: common storage counts embedding/output exactly once.
            profile.common_gpu_bytes += retained + contribution.accelerator_transient_bytes;
        }
    }
    Ok(profile)
}

fn deepseek41_header_memory_bytes(content: &RawGgufContent) -> u64 {
    fn value_bytes(value: &gguf_file::Value) -> u64 {
        match value {
            gguf_file::Value::String(s) => s.capacity() as u64,
            gguf_file::Value::Array(values) => {
                (values.capacity() * std::mem::size_of::<gguf_file::Value>()) as u64
                    + values.iter().map(value_bytes).sum::<u64>()
            }
            _ => 0,
        }
    }
    // Include hash buckets/keys, owned Value arrays, and tensor descriptors.
    let metadata = content
        .metadata
        .iter()
        .map(|(key, value)| key.capacity() as u64 + value_bytes(value))
        .sum::<u64>()
        + (content.metadata.capacity().next_power_of_two()
            * (std::mem::size_of::<(String, gguf_file::Value)>() + 1)
            + 16) as u64;
    let tensors = content
        .tensor_infos
        .iter()
        .map(|(key, info)| {
            (key.capacity() + info.shape.capacity() * std::mem::size_of::<usize>()) as u64
        })
        .sum::<u64>()
        + (content.tensor_infos.capacity().next_power_of_two()
            * (std::mem::size_of::<(String, deepseek4_gguf::RawTensorInfo)>() + 1)
            + 16) as u64;
    // One Arc allocation each for source and content; header maps/arrays are
    // owned only by content, not copied into the source.
    metadata
        + tensors
        + std::mem::size_of::<RawGgufSource>() as u64
        + std::mem::size_of::<RawGgufContent>() as u64
        + 4 * std::mem::size_of::<usize>() as u64
}

fn load_stage_tensors_deepseek4(
    shard_paths: &[PathBuf],
    range: LayerRange,
    device: &Device,
) -> Result<StageTensors> {
    tracing::info!("Loading DeepSeek V4 GGUF with the DINF raw tensor reader");

    let mut shard_files: Vec<(std::fs::File, RawGgufContent)> =
        Vec::with_capacity(shard_paths.len());
    let mut aggregate_header_bytes = 0u64;
    let mut tensor_owners = HashMap::new();
    let mut shard_metadata = ShardMetadata::new(shard_paths.len());
    for (shard_index, path) in shard_paths.iter().enumerate() {
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open shard: {}", path.display()))?;
        let content = RawGgufContent::read(&mut file)
            .with_context(|| format!("Failed to parse DeepSeek V4 shard: {}", path.display()))?;
        account_split_header_bytes(&mut aggregate_header_bytes, content.parsed_header_bytes())?;
        register_shard_tensors(&mut tensor_owners, shard_index, content.tensor_infos.keys())?;
        shard_metadata.record(shard_index, &content.metadata, content.tensor_infos.len())?;
        shard_files.push((file, content));
    }
    shard_metadata.finish()?;

    let metadata = parse_gguf_metadata(&shard_files[0].1.metadata)?;
    for (_, content) in &shard_files {
        if let Some(info) = content.tensor_infos.get("token_embd.weight") {
            validate_embedding_vocabulary(&metadata, &info.shape)?;
        }
    }
    anyhow::ensure!(
        metadata.architecture == "deepseek4",
        "DINF V4 reader received architecture {}",
        metadata.architecture
    );
    anyhow::ensure!(
        range.total_layers == metadata.num_layers,
        "Layer range total_layers ({}) doesn't match model num_layers ({})",
        range.total_layers,
        metadata.num_layers
    );

    let mut raw_tensors = HashMap::new();
    for (shard_idx, (file, content)) in shard_files.iter_mut().enumerate() {
        let names = content.tensor_infos.keys().cloned().collect::<Vec<_>>();
        for name in names {
            if !tensor_belongs_to_stage(&name, &range) {
                continue;
            }
            // V4 has an untied output head. Its tail never uses the token
            // embedding, even though ordinary GGUF loaders need it for a
            // possible tied output projection.
            if is_token_embedding(&name) && !range.is_head() {
                continue;
            }
            let tensor = content.load_tensor(file, &name, device).with_context(|| {
                format!(
                    "Failed to load DeepSeek V4 tensor {name} from shard {}",
                    shard_idx + 1
                )
            })?;
            raw_tensors.insert(name, Arc::new(tensor));
        }
    }

    tracing::info!(
        "Loaded {} raw DeepSeek V4 tensors for layers [{}, {}) from {} shards",
        raw_tensors.len(),
        range.start,
        range.end,
        shard_paths.len(),
    );

    Ok(StageTensors {
        range,
        metadata,
        tensors: HashMap::new(),
        raw_tensors,
        raw_source: None,
        device: device.clone(),
    })
}

/// Single-file GGUF loading (original path).
fn load_stage_tensors_single(
    gguf_path: &Path,
    range: LayerRange,
    device: &Device,
) -> Result<StageTensors> {
    let (mut file, _) = open_validated_gguf(gguf_path)?;

    let content =
        gguf_file::Content::read(&mut file).with_context(|| "Failed to parse GGUF file header")?;

    let mut shard_metadata = ShardMetadata::new(1);
    shard_metadata.record(0, &content.metadata, content.tensor_infos.len())?;
    shard_metadata.finish()?;

    let metadata = parse_gguf_metadata(&content.metadata)?;
    if let Some(info) = content.tensor_infos.get("token_embd.weight") {
        validate_embedding_vocabulary(&metadata, info.shape.dims())?;
    }

    if range.total_layers != metadata.num_layers {
        anyhow::bail!(
            "Layer range total_layers ({}) doesn't match model num_layers ({})",
            range.total_layers,
            metadata.num_layers
        );
    }

    let mut tensors = HashMap::new();
    let has_output_weight = content.tensor_infos.contains_key("output.weight");
    for (name, info) in content.tensor_infos.iter() {
        if tensor_belongs_to_stage(name, &range) {
            if !range.is_head() && has_output_weight && is_token_embedding(name) {
                continue;
            }
            let load_device = if tensor_loads_on_cpu(
                &metadata.architecture,
                name,
                info.ggml_dtype,
                device.is_metal(),
            ) {
                &Device::Cpu
            } else {
                device
            };
            let tensor = content
                .tensor(&mut file, name, load_device)
                .with_context(|| format!("Failed to load tensor: {name}"))?;
            tensors.insert(name.clone(), Arc::new(tensor));
        }
    }

    tracing::info!(
        "Loaded {} tensors for layers [{}, {}) of {} (arch: {}, {} total layers)",
        tensors.len(),
        range.start,
        range.end,
        gguf_path.display(),
        metadata.architecture,
        metadata.num_layers,
    );

    Ok(StageTensors {
        range,
        metadata,
        tensors,
        raw_tensors: HashMap::new(),
        raw_source: None,
        device: device.clone(),
    })
}

/// Estimate GPU memory (MB) needed to load a stage's tensors from a GGUF file,
/// WITHOUT actually loading them. Reads only the GGUF header + tensor info table.
///
/// Supports split GGUF files — reads tensor info from all shards.
/// Packed expert tensors (`_exps.`) are loaded through CPU storage first, then
/// copied to the accelerator by the split or indexed MoE implementation.
/// Returns `(gpu_mb, cpu_mb)`.
pub fn estimate_stage_vram_mb(
    gguf_path: &Path,
    range: &LayerRange,
    device: &Device,
) -> Result<(u64, u64)> {
    range.validate()?;
    // Keep this legacy `(gpu_mb, cpu_mb)` API backed by the same exact profile
    // as the newer stage admission path. In particular, this preserves the
    // largest per-layer construction peak instead of summing every source
    // tensor in a multi-layer stage.
    let profile = estimate_stage_memory_profile(gguf_path, "", range.total_layers, device)?;
    let (gpu_bytes, retained_cpu_bytes, transient_cpu_bytes) = profile
        .stage_memory_components(range.start, range.end, range.total_layers)
        .context("exact stage memory profile did not cover the requested range")?;
    const MIB: u64 = 1024 * 1024;
    Ok((
        gpu_bytes.div_ceil(MIB),
        retained_cpu_bytes
            .saturating_add(transient_cpu_bytes)
            .div_ceil(MIB),
    ))
}

/// Return the bytes needed by architecture-owned RoPE tables and the largest
/// temporary construction overlap while those tables are materialized.
///
/// `StageMemoryProfile` has a retained common-memory field and only a CPU
/// common-transient field. The caller keeps CPU construction overlap in that
/// transient field; accelerator construction overlap is folded into the
/// retained GPU total so admission cannot undercount it.
fn generated_rope_memory_bytes(metadata: &GgufMetadata, device: &Device) -> (u64, u64) {
    let (rope_dim, context_length, table_count) = match metadata.architecture.as_str() {
        // Llama-family, Qwen3-MoE, Phi-3, and MiniMax retain one cosine and
        // one sine table for the advertised context.
        "llama" | "mistral" | "qwen2" | "qwen3moe" => {
            (metadata.head_dim, metadata.context_length, 2u64)
        }
        "phi3" => (
            if metadata.rope_dimension_count > 0 {
                metadata.rope_dimension_count
            } else {
                metadata.head_dim
            },
            metadata.context_length,
            2,
        ),
        "minimax-m2" => (
            if metadata.rope_dimension_count > 0 {
                metadata.rope_dimension_count
            } else {
                metadata.head_dim
            },
            metadata.context_length,
            2,
        ),
        "qwen35" => (
            // Qwen3.5 uses a 64-frequency text RoPE by default, even when
            // attention heads are wider.
            if metadata.rope_dimension_count > 0 {
                metadata.rope_dimension_count
            } else {
                64
            },
            metadata.context_length,
            2,
        ),
        "deepseek2" | "glm-dsa" => (metadata.qk_rope_head_dim, metadata.context_length, 2),
        "llama4" => (metadata.head_dim, metadata.context_length.min(8192), 2),
        "deepseek4" => {
            // V4 retains cos, sin, and -sin. A compressed schedule creates a
            // second independent table set; otherwise it clones the first
            // tensors without another allocation.
            let compressed = metadata
                .compress_ratios
                .iter()
                .take(metadata.num_layers)
                .any(|&ratio| ratio > 0);
            (
                metadata.qk_rope_head_dim,
                metadata.context_length,
                if compressed { 6 } else { 3 },
            )
        }
        // Gemma4 computes short RoPE slices per forward pass rather than
        // retaining a context-length table. Nemotron-H and GLM5-Next likewise
        // have no constructor-owned context RoPE table.
        "gemma4" | "nemotron_h" | "nemotron_h_moe" | "glm5next" | "glm5-next" | "muse-spark" => {
            return (0, 0);
        }
        _ => return (0, 0),
    };

    let half_dim = rope_dim / 2;
    let element_bytes = std::mem::size_of::<f32>() as u64;
    let tensor_bytes = |elements: u64| {
        let bytes = elements.saturating_mul(element_bytes);
        if device.is_metal() && bytes != 0 {
            // Candle's Metal storage pool rounds individual allocations to a
            // power-of-two bucket. Admission must include that physical peak,
            // not just the logical tensor length (especially for non-power-of-
            // two context lengths and partial RoPE dimensions).
            bytes.checked_next_power_of_two().unwrap_or(u64::MAX)
        } else {
            bytes
        }
    };
    let table_bytes = tensor_bytes((context_length as u64).saturating_mul(half_dim as u64));
    let persistent_bytes = table_bytes.saturating_mul(table_count);
    // At the final `sin`/`sin_neg` operation, the output tables coexist with
    // the matmul result (`angles`), F32 positions, and the inverse-frequency
    // vector. The initial U32 arange also overlaps its F32 cast; count that
    // source buffer once more so a Metal cast bucket cannot be missed.
    let construction_bytes = table_bytes
        .saturating_add(tensor_bytes(context_length as u64))
        .saturating_add(tensor_bytes(context_length as u64))
        .saturating_add(tensor_bytes(half_dim as u64));
    (persistent_bytes, construction_bytes)
}

fn account_generated_rope_memory(
    profile: &mut StageMemoryProfile,
    metadata: &GgufMetadata,
    device: &Device,
) {
    let (persistent_bytes, construction_bytes) = generated_rope_memory_bytes(metadata, device);
    if device.is_cpu() {
        // On CPU the tables are retained in host memory. Keep the persistent
        // allocation in the retained counter and the construction overlap in
        // the existing common CPU transient field.
        profile.common_cpu_bytes = profile.common_cpu_bytes.saturating_add(persistent_bytes);
        profile.common_cpu_transient_bytes =
            profile.common_cpu_transient_bytes.max(construction_bytes);
    } else {
        // StageMemoryProfile has no accelerator-common transient field. Fold
        // the construction peak into common GPU bytes so admission cannot
        // undercount the overlap while Metal/CUDA RoPE tables are built.
        profile.common_gpu_bytes = profile
            .common_gpu_bytes
            .saturating_add(persistent_bytes)
            .saturating_add(construction_bytes);
    }
}

fn account_qwen3moe_attention_workspace(
    profile: &mut StageMemoryProfile,
    architecture: &str,
    head_dim: usize,
    is_metal: bool,
) {
    if is_metal && architecture == "qwen3moe" && head_dim == 128 {
        // The private F32 attention path enforces this cap before allocating
        // and fences both success and error before dropping its workspace.
        // The Metal pool can retain those buffers into the next phase because
        // it drains unused buffers before the local tensors drop. Charge the
        // attention cap beside the existing construction/expert peak. Range
        // accounting takes the maximum of these layer peaks, not their sum.
        let attention_peak = crate::architectures::qwen3moe::F32_ATTENTION_WORKSPACE_BYTES as u64;
        for peak in &mut profile.layer_gpu_transient_bytes {
            *peak = (*peak).saturating_add(attention_peak);
        }
    }
}

/// Read the GGUF tensor table once and report the device-specific accelerator
/// tensor footprint used when planning pipeline layer splits. Packed experts
/// remain CPU-resident, but Metal and CUDA MoE implementations also copy them
/// to the accelerator, so accelerator profiles include those bytes.
pub fn estimate_stage_memory_profile(
    gguf_path: &Path,
    model_id: &str,
    total_layers: usize,
    device: &Device,
) -> Result<StageMemoryProfile> {
    // Validate before any Vec is sized from the caller-provided layer count.
    validate_layer_count(total_layers, "requested GGUF layer count")?;
    let shard_paths = discover_shard_paths(gguf_path)?;
    let packed_experts_use_accelerator = device.is_metal() || device.is_cuda();
    let metal_tensor_qmatmul_available = crate::architectures::uses_metal_tensor_qmatmul(device)?;
    let architecture = read_model_architecture(&shard_paths[0])?;
    let metadata = read_gguf_metadata(&shard_paths[0])?;
    if architecture == "deepseek41" {
        anyhow::ensure!(
            shard_paths.len() == 1
                || (shard_paths.len() == 6 && crate::gsq_coder::is_entry(&shard_paths[0])),
            "DeepSeek V4.1 split GGUF is unsupported"
        );
        return estimate_deepseek41_profile(
            &shard_paths[0],
            model_id,
            total_layers,
            &metadata,
            device,
        );
    }
    anyhow::ensure!(
        total_layers == metadata.num_layers,
        "requested GGUF layer count ({total_layers}) doesn't match model num_layers ({})",
        metadata.num_layers
    );
    if architecture == "deepseek4" {
        return estimate_stage_memory_profile_deepseek4(
            &shard_paths,
            model_id,
            total_layers,
            &metadata,
            device,
        );
    }
    let mut profile = StageMemoryProfile {
        model_id: model_id.to_string(),
        total_layers,
        layer_gpu_bytes: vec![0; total_layers],
        layer_gpu_transient_bytes: vec![0; total_layers],
        layer_cpu_bytes: vec![0; total_layers],
        layer_cpu_transient_bytes: vec![0; total_layers],
        ..Default::default()
    };
    account_generated_rope_memory(&mut profile, &metadata, device);
    // GLM5 and MiniMax use the streaming loader: CPU sources are retained
    // only while their layer is being constructed, then released before the
    // next layer is read. Keep those bytes in the per-layer transient peak,
    // rather than the retained stage component. Other architectures still
    // use the historical whole-stage source lifetime.
    let stream_cpu_sources = uses_streaming_sources(&architecture);
    let mut layer_cpu_source_bytes = vec![0u64; total_layers];
    let mut layer_cpu_upload_peak = vec![0u64; total_layers];
    let mut layer_accelerator_source_bytes = vec![0u64; total_layers];
    let mut layer_glm_conv_cat_bytes = vec![0u64; total_layers];
    let mut layer_qwen35_grouped_construction =
        vec![Qwen35GroupedConstructionBytes::default(); total_layers];
    let mut cpu_packed_embedding_bytes = 0u64;
    let mut dequantized_embedding_bytes = 0u64;
    let mut tail_embedding_fallback_bytes = 0u64;
    let mut tail_embedding_matmul_bytes = 0u64;
    let mut floating_tied_output = false;
    let mut q2_tied_output_shape = None;
    let mut has_output_weight = false;
    let mut aggregate_header_bytes = 0u64;
    let mut tensor_owners = HashMap::new();
    let mut shard_metadata = ShardMetadata::new(shard_paths.len());

    for (shard_index, path) in shard_paths.iter().enumerate() {
        let (mut file, stats) = open_validated_gguf(path)?;
        let content = gguf_file::Content::read(&mut file)
            .with_context(|| format!("Failed to parse GGUF header: {}", path.display()))?;
        account_split_header_bytes(
            &mut aggregate_header_bytes,
            stats.parsed_bytes(content.tensor_data_offset),
        )?;
        register_shard_tensors(&mut tensor_owners, shard_index, content.tensor_infos.keys())?;
        shard_metadata.record(shard_index, &content.metadata, content.tensor_infos.len())?;

        if let Some(info) = content.tensor_infos.get("token_embd.weight") {
            validate_embedding_vocabulary(&metadata, info.shape.dims())?;
        }
        for (name, info) in content.tensor_infos.iter() {
            has_output_weight |= name == "output.weight";
            let elem_count = info.shape.elem_count();
            let block_size = info.ggml_dtype.block_size();
            let type_size = info.ggml_dtype.type_size();
            let tensor_bytes = ggml_tensor_bytes(elem_count, block_size, type_size)?;
            let contribution = tensor_memory_accounting_with_shape(
                &architecture,
                name,
                tensor_bytes,
                info.shape.dims(),
                info.ggml_dtype,
                packed_experts_use_accelerator,
                device.is_metal(),
                metal_tensor_qmatmul_available,
            )?;
            if contribution.accelerator_bytes == 0
                && contribution.retained_cpu_bytes == 0
                && contribution.accelerator_transient_bytes == 0
                && contribution.cpu_source_bytes == 0
                && contribution.cpu_upload_bytes == 0
            {
                continue;
            }
            let fallback_f16_bytes = decode_fallback_f16_bytes(
                &architecture,
                name,
                info.ggml_dtype,
                elem_count,
                device.is_metal(),
            );
            let realized_tensor_bytes = contribution
                .accelerator_bytes
                .saturating_add(fallback_f16_bytes);

            if let Some(rest) = name.strip_prefix("blk.") {
                if let Some(dot_pos) = rest.find('.')
                    && let Ok(layer_idx) = rest[..dot_pos].parse::<usize>()
                    && let Some(bytes) = profile.layer_gpu_bytes.get_mut(layer_idx)
                {
                    if architecture == "qwen35"
                        && device.is_metal()
                        && metal_tensor_qmatmul_available
                        && qwen35_metal_qmatmul_dtype(info.ggml_dtype)
                        && let Some(construction) =
                            layer_qwen35_grouped_construction.get_mut(layer_idx)
                    {
                        construction.record(name, tensor_bytes);
                    }
                    *bytes = bytes.saturating_add(realized_tensor_bytes);
                    layer_accelerator_source_bytes[layer_idx] = layer_accelerator_source_bytes
                        [layer_idx]
                        .saturating_add(contribution.accelerator_source_bytes);
                    if let Some(transient_bytes) =
                        profile.layer_gpu_transient_bytes.get_mut(layer_idx)
                    {
                        *transient_bytes =
                            (*transient_bytes).max(contribution.accelerator_transient_bytes);
                    }
                    if matches!(architecture.as_str(), "glm5next" | "glm5-next")
                        && (name.ends_with(".ssm_conv1d_q.weight")
                            || name.ends_with(".ssm_conv1d_k.weight")
                            || name.ends_with(".ssm_conv1d_v.weight"))
                        && let Some(cat_bytes) = layer_glm_conv_cat_bytes.get_mut(layer_idx)
                    {
                        // KDA keeps the three realized convolution tensors
                        // alive while concatenating them into the retained
                        // combined weight. Charge that small duplicate peak.
                        *cat_bytes = cat_bytes.saturating_add(contribution.accelerator_bytes);
                    }
                    if stream_cpu_sources {
                        if let Some(source_bytes) = layer_cpu_source_bytes.get_mut(layer_idx) {
                            *source_bytes =
                                source_bytes.saturating_add(contribution.cpu_source_bytes);
                        }
                        if let Some(upload_bytes) = layer_cpu_upload_peak.get_mut(layer_idx) {
                            *upload_bytes = (*upload_bytes).max(contribution.cpu_upload_bytes);
                        }
                        if let Some(retained_bytes) = profile.layer_cpu_bytes.get_mut(layer_idx) {
                            *retained_bytes =
                                retained_bytes.saturating_add(contribution.retained_cpu_bytes);
                        }
                    } else {
                        if let Some(retained_bytes) = profile.layer_cpu_bytes.get_mut(layer_idx) {
                            *retained_bytes = (*retained_bytes)
                                .saturating_add(contribution.retained_cpu_bytes)
                                .saturating_add(contribution.cpu_source_bytes);
                        }
                        if let Some(transient_bytes) =
                            profile.layer_cpu_transient_bytes.get_mut(layer_idx)
                        {
                            *transient_bytes =
                                (*transient_bytes).max(contribution.cpu_upload_bytes);
                        }
                    }
                }
            } else if is_token_embedding(name) {
                if device.is_metal() && info.ggml_dtype == GgmlDType::Q2K {
                    q2_tied_output_shape = Some(info.shape.dims().to_vec());
                }
                if qwen35_cpu_packed_embedding(
                    &architecture,
                    name,
                    info.ggml_dtype,
                    device.is_metal(),
                ) {
                    cpu_packed_embedding_bytes =
                        cpu_packed_embedding_bytes.saturating_add(tensor_bytes);
                }
                profile.embedding_gpu_bytes = profile
                    .embedding_gpu_bytes
                    .saturating_add(contribution.accelerator_bytes);
                // Qwen35 Metal Q4K lookups stay packed on CPU. Other
                // embeddings retain their existing device placement.
                profile.embedding_cpu_bytes = profile
                    .embedding_cpu_bytes
                    .saturating_add(contribution.retained_cpu_bytes)
                    .saturating_add(contribution.cpu_source_bytes);
                dequantized_embedding_bytes =
                    dequantized_embedding_bytes.saturating_add(dequantized_f32_bytes(elem_count));
                tail_embedding_fallback_bytes =
                    tail_embedding_fallback_bytes.saturating_add(fallback_f16_bytes);
                tail_embedding_matmul_bytes = tail_embedding_matmul_bytes.saturating_add(
                    if matches!(
                        info.ggml_dtype,
                        GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::F32
                    ) {
                        floating_tied_output = true;
                        dequantized_f32_bytes(elem_count)
                    } else {
                        contribution.accelerator_bytes
                    },
                );
            } else if name == "output_norm.weight"
                || name == "output.weight"
                || name.starts_with("output_norm.")
                || name.starts_with("output.")
            {
                profile.output_gpu_bytes = profile
                    .output_gpu_bytes
                    .saturating_add(realized_tensor_bytes)
                    .saturating_add(contribution.accelerator_source_bytes);
                profile.output_cpu_bytes = profile
                    .output_cpu_bytes
                    .saturating_add(contribution.retained_cpu_bytes)
                    .saturating_add(contribution.cpu_source_bytes);
                profile.output_gpu_transient_bytes = profile
                    .output_gpu_transient_bytes
                    .max(contribution.accelerator_transient_bytes);
                profile.output_cpu_transient_bytes = profile
                    .output_cpu_transient_bytes
                    .max(contribution.cpu_upload_bytes);
            } else if name == "rope_freqs.weight" || name.starts_with("rope_freqs") {
                profile.common_gpu_bytes = profile
                    .common_gpu_bytes
                    .saturating_add(realized_tensor_bytes)
                    .saturating_add(contribution.accelerator_source_bytes);
                profile.common_cpu_bytes = profile
                    .common_cpu_bytes
                    .saturating_add(contribution.retained_cpu_bytes)
                    .saturating_add(contribution.cpu_source_bytes);
                profile.common_cpu_transient_bytes = profile
                    .common_cpu_transient_bytes
                    .max(contribution.cpu_upload_bytes);
            }
        }
    }

    shard_metadata.finish()?;
    if architecture == "qwen35" && device.is_metal() && metal_tensor_qmatmul_available {
        let mlp_grouped = std::env::var_os("DINF_DISABLE_QWEN35_FUSED_MLP").is_none();
        let grouped_projections =
            std::env::var_os("DINF_DISABLE_QWEN35_GROUPED_PROJECTIONS").is_none();
        let deltanet_grouped = grouped_projections
            && std::env::var_os("DINF_DISABLE_QWEN35_GROUPED_DELTANET").is_none();
        let attention_grouped = grouped_projections
            && std::env::var_os("DINF_DISABLE_QWEN35_GROUPED_ATTENTION_PROJECTIONS").is_none();
        for (layer_idx, construction) in layer_qwen35_grouped_construction.into_iter().enumerate() {
            let (accelerator_peak, host_peak) =
                construction.grouped_peaks(mlp_grouped, deltanet_grouped, attention_grouped);
            profile.layer_gpu_transient_bytes[layer_idx] =
                profile.layer_gpu_transient_bytes[layer_idx].max(accelerator_peak);
            profile.layer_cpu_transient_bytes[layer_idx] =
                profile.layer_cpu_transient_bytes[layer_idx].max(host_peak);
        }
    }

    if stream_cpu_sources {
        for (layer_idx, (transient_bytes, (source_bytes, upload_bytes))) in profile
            .layer_cpu_transient_bytes
            .iter_mut()
            .zip(
                layer_cpu_source_bytes
                    .into_iter()
                    .zip(layer_cpu_upload_peak),
            )
            .enumerate()
        {
            // All source QTensor objects for one layer are loaded together;
            // only the largest owned upload buffer overlaps that source set.
            *transient_bytes = source_bytes.saturating_add(upload_bytes);
            if let Some(gpu_transient_bytes) = profile.layer_gpu_transient_bytes.get_mut(layer_idx)
            {
                *gpu_transient_bytes = (*gpu_transient_bytes).max(
                    layer_glm_conv_cat_bytes
                        .get(layer_idx)
                        .copied()
                        .unwrap_or(0),
                );
            }
        }
    }

    for (layer_idx, source_bytes) in layer_accelerator_source_bytes.into_iter().enumerate() {
        if stream_cpu_sources {
            // The source pool overlaps the largest conversion/concatenation
            // scratch allocation computed above, not just one or the other.
            profile.layer_gpu_transient_bytes[layer_idx] =
                profile.layer_gpu_transient_bytes[layer_idx].saturating_add(source_bytes);
        } else {
            // Eager StageTensors keeps every source until the entire stage
            // constructor returns. Reserve this pool across all selected
            // layers, not merely the largest single tensor/layer.
            profile.layer_gpu_bytes[layer_idx] =
                profile.layer_gpu_bytes[layer_idx].saturating_add(source_bytes);
        }
    }

    let dense_head_embedding =
        !matches!(architecture.as_str(), "qwen35" | "glm5next" | "glm5-next");
    let head_embedding = realized_head_embedding_bytes(
        &architecture,
        profile.embedding_gpu_bytes,
        dequantized_embedding_bytes,
    );
    // The eager head constructor owns the encoded embedding while building
    // its independent dense lookup table. Metal also needs a readback buffer
    // of that encoded size. With no head GPU transient field, reserve both
    // in the head component so large embeddings cannot escape the peak.
    profile.head_embedding_gpu_bytes = Some(head_embedding.saturating_add(
        if packed_experts_use_accelerator && dense_head_embedding {
            profile
                .embedding_gpu_bytes
                .saturating_add(if device.is_metal() {
                    device_allocation_bytes(profile.embedding_gpu_bytes, device)
                } else {
                    0
                })
        } else {
            0
        },
    ));
    profile.head_embedding_cpu_bytes = Some(profile.embedding_cpu_bytes);
    profile.head_embedding_cpu_transient_bytes =
        Some(if device.is_metal() && dense_head_embedding {
            // QMetalStorage::dequantize holds encoded and F32 host vectors
            // together before uploading the dense embedding.
            profile
                .embedding_gpu_bytes
                .saturating_add(dequantized_embedding_bytes)
        } else {
            head_embedding_cpu_transient_bytes(
                &architecture,
                profile.embedding_gpu_bytes,
                packed_experts_use_accelerator,
            )
        });
    if !has_output_weight && floating_tied_output {
        // A floating QMatMul owns an independent F32 output tensor, even when
        // its source is tied to the embedding. Boundary accounting uses
        // max(head, tail) for shared storage, so put this allocation in the
        // output component to count both copies on a solo stage.
        profile.output_gpu_bytes = profile
            .output_gpu_bytes
            .saturating_add(tail_embedding_matmul_bytes);
        if packed_experts_use_accelerator {
            profile.output_gpu_bytes = profile
                .output_gpu_bytes
                .saturating_add(profile.embedding_gpu_bytes);
            if device.is_metal() {
                profile.output_gpu_transient_bytes = profile
                    .output_gpu_transient_bytes
                    .max(profile.embedding_gpu_bytes);
                profile.output_cpu_transient_bytes = profile.output_cpu_transient_bytes.max(
                    profile
                        .embedding_gpu_bytes
                        .saturating_add(tail_embedding_matmul_bytes),
                );
            }
        } else {
            profile.output_cpu_transient_bytes = profile
                .output_cpu_transient_bytes
                .max(profile.embedding_gpu_bytes);
        }
    }
    // An untied lookup never enters the Q2 projection constructor. Defer its
    // shape/cap checks until all shard headers establish that output is tied.
    let q2_tied_output = if has_output_weight {
        None
    } else {
        q2_tied_output_shape
            .as_deref()
            .map(metal_q2_projection_accounting)
            .transpose()?
    };
    if let Some(output) = q2_tied_output {
        account_tied_metal_q2_output(&mut profile, output);
    }
    profile.tail_embedding_gpu_bytes = Some(
        if has_output_weight || floating_tied_output || q2_tied_output.is_some() {
            0
        } else {
            tail_embedding_matmul_bytes.saturating_add(tail_embedding_fallback_bytes)
        },
    );
    profile.tail_embedding_cpu_bytes = Some(profile.embedding_cpu_bytes);
    profile.tail_embedding_cpu_transient_bytes = Some(0);
    if cpu_packed_embedding_bytes > 0 {
        set_qwen35_cpu_embedding_boundaries(
            &mut profile,
            cpu_packed_embedding_bytes,
            has_output_weight,
        );
    }
    account_qwen3moe_attention_workspace(
        &mut profile,
        &metadata.architecture,
        metadata.head_dim,
        device.is_metal(),
    );

    Ok(profile)
}

fn set_qwen35_cpu_embedding_boundaries(
    profile: &mut StageMemoryProfile,
    packed_bytes: u64,
    has_output_weight: bool,
) {
    profile.head_embedding_gpu_bytes = Some(0);
    profile.head_embedding_cpu_bytes = Some(packed_bytes);
    // GGUF loading owns its raw byte buffer while copying into packed CPU
    // storage; reserve that extra buffer even without a dense conversion.
    profile.head_embedding_cpu_transient_bytes = Some(packed_bytes);
    // A tied output keeps a separate packed Metal matmul; the lookup Arc is
    // retained only by head/solo stages. Reserve its CPU source plus one owned
    // upload buffer for a tail-only constructor. This also bounds solo stages
    // conservatively while their lookup Arc remains live.
    profile.tail_embedding_gpu_bytes = Some(if has_output_weight { 0 } else { packed_bytes });
    profile.tail_embedding_cpu_bytes = Some(0);
    profile.tail_embedding_cpu_transient_bytes = Some(if has_output_weight {
        0
    } else {
        packed_bytes.saturating_mul(2)
    });
}

fn estimate_stage_memory_profile_deepseek4(
    shard_paths: &[PathBuf],
    model_id: &str,
    total_layers: usize,
    metadata: &GgufMetadata,
    device: &Device,
) -> Result<StageMemoryProfile> {
    validate_layer_count(total_layers, "requested DeepSeek V4 layer count")?;
    let mut profile = StageMemoryProfile {
        model_id: model_id.to_string(),
        total_layers,
        layer_gpu_bytes: vec![0; total_layers],
        layer_gpu_transient_bytes: vec![0; total_layers],
        layer_cpu_bytes: vec![0; total_layers],
        layer_cpu_transient_bytes: vec![0; total_layers],
        ..Default::default()
    };
    account_generated_rope_memory(&mut profile, metadata, device);
    // V4 creates three norm weights per layer: a 512-wide query norm and
    // one flattened stream norm for each hyper-connection. The tail adds
    // another flattened stream norm beside the output projection.
    let stream_norm_bytes = device_allocation_bytes(
        (metadata.hidden_dim as u64)
            .saturating_mul(metadata.hyperconnection_count as u64)
            .saturating_mul(4),
        device,
    );
    let generated_layer_bytes = device_allocation_bytes(512 * 4, device)
        .saturating_add(stream_norm_bytes.saturating_mul(2));
    profile.layer_gpu_bytes.fill(generated_layer_bytes);
    profile.output_gpu_bytes = stream_norm_bytes;

    let mut aggregate_header_bytes = 0u64;
    let mut tensor_owners = HashMap::new();
    let mut shard_metadata = ShardMetadata::new(shard_paths.len());
    for (shard_index, path) in shard_paths.iter().enumerate() {
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open GGUF file: {}", path.display()))?;
        let content = RawGgufContent::read(&mut file)
            .with_context(|| format!("Failed to parse DeepSeek V4 header: {}", path.display()))?;
        account_split_header_bytes(&mut aggregate_header_bytes, content.parsed_header_bytes())?;
        register_shard_tensors(&mut tensor_owners, shard_index, content.tensor_infos.keys())?;
        shard_metadata.record(shard_index, &content.metadata, content.tensor_infos.len())?;
        if let Some(info) = content.tensor_infos.get("token_embd.weight") {
            validate_embedding_vocabulary(metadata, &info.shape)?;
        }
        for (name, info) in &content.tensor_infos {
            let contribution = deepseek4_tensor_memory_accounting(name, info, metadata, device)?;
            if let Some(rest) = name.strip_prefix("blk.")
                && let Some(dot_pos) = rest.find('.')
                && let Ok(layer_idx) = rest[..dot_pos].parse::<usize>()
                && let Some(layer_bytes) = profile.layer_gpu_bytes.get_mut(layer_idx)
            {
                *layer_bytes = layer_bytes.saturating_add(contribution.accelerator_bytes);
                profile.layer_gpu_transient_bytes[layer_idx] = profile.layer_gpu_transient_bytes
                    [layer_idx]
                    .max(contribution.accelerator_transient_bytes);
                profile.layer_cpu_transient_bytes[layer_idx] =
                    profile.layer_cpu_transient_bytes[layer_idx].max(contribution.cpu_upload_bytes);
            } else if is_token_embedding(name) {
                profile.embedding_gpu_bytes = profile
                    .embedding_gpu_bytes
                    .saturating_add(contribution.accelerator_bytes);
                profile.head_embedding_cpu_transient_bytes = Some(
                    profile
                        .head_embedding_cpu_transient_bytes
                        .unwrap_or(0)
                        .max(contribution.cpu_upload_bytes),
                );
                // There is no embedding GPU transient field. Reserve its
                // readback/cast overlap with the head allocation itself.
                profile.embedding_gpu_bytes = profile
                    .embedding_gpu_bytes
                    .saturating_add(contribution.accelerator_transient_bytes);
            } else if name == "output_norm.weight"
                || name == "output.weight"
                || name.starts_with("output_norm.")
                || name.starts_with("output.")
                || name.starts_with("output_hc_")
            {
                profile.output_gpu_bytes = profile
                    .output_gpu_bytes
                    .saturating_add(contribution.accelerator_bytes);
                profile.output_gpu_transient_bytes = profile
                    .output_gpu_transient_bytes
                    .max(contribution.accelerator_transient_bytes);
                profile.output_cpu_transient_bytes = profile
                    .output_cpu_transient_bytes
                    .max(contribution.cpu_upload_bytes);
            } else if name == "rope_freqs.weight" || name.starts_with("rope_freqs") {
                profile.common_gpu_bytes = profile
                    .common_gpu_bytes
                    .saturating_add(contribution.accelerator_bytes)
                    .saturating_add(contribution.accelerator_transient_bytes);
                profile.common_cpu_transient_bytes = profile
                    .common_cpu_transient_bytes
                    .max(contribution.cpu_upload_bytes);
            }
        }
    }

    shard_metadata.finish()?;
    set_deepseek4_embedding_boundaries(&mut profile);
    Ok(profile)
}

fn set_deepseek4_embedding_boundaries(profile: &mut StageMemoryProfile) {
    profile.head_embedding_gpu_bytes = Some(profile.embedding_gpu_bytes);
    profile.head_embedding_cpu_bytes = Some(profile.embedding_cpu_bytes);
    profile.head_embedding_cpu_transient_bytes.get_or_insert(0);
    // DeepSeek V4 has an untied `output.weight`, so its tail stage does not
    // retain `token_embd.weight`. Reporting the embedding for both boundaries
    // makes a solo stage count the same tensor twice during MM admission.
    profile.tail_embedding_gpu_bytes = Some(0);
    profile.tail_embedding_cpu_bytes = Some(0);
    profile.tail_embedding_cpu_transient_bytes = Some(0);
}

fn device_allocation_bytes(bytes: u64, device: &Device) -> u64 {
    if device.is_metal() && bytes != 0 {
        bytes.checked_next_power_of_two().unwrap_or(u64::MAX)
    } else {
        bytes
    }
}

fn deepseek4_tensor_memory_accounting(
    name: &str,
    info: &deepseek4_gguf::RawTensorInfo,
    metadata: &GgufMetadata,
    device: &Device,
) -> Result<TensorMemoryAccounting> {
    use deepseek4_gguf::DeepSeek4DType;

    let raw_bytes = info.byte_len()? as u64;
    let dense = matches!(
        info.dtype,
        DeepSeek4DType::F32 | DeepSeek4DType::F16 | DeepSeek4DType::BF16 | DeepSeek4DType::I32
    );
    let f32_bytes = if dense {
        dequantized_f32_bytes(info.element_count()?)
    } else {
        0
    };
    // Rank-3 experts and grouped attention projections create one dense
    // tensor per slice. Their raw byte views share the parent allocation.
    let slices = if name.ends_with(".attn_output_a.weight") {
        metadata.output_group_count as u64
    } else if info.shape.len() == 3 {
        info.shape[0] as u64
    } else {
        1
    };
    let dense_device_bytes = if dense && slices > 0 {
        device_allocation_bytes(f32_bytes / slices, device).saturating_mul(slices)
    } else {
        0
    };
    Ok(TensorMemoryAccounting {
        // Dense matmuls retain both forms. Dense vectors and embeddings
        // release raw storage when the eager constructor returns, so count
        // their source pool here too to bound the full construction peak.
        accelerator_bytes: raw_bytes.saturating_add(dense_device_bytes),
        accelerator_transient_bytes: if !device.is_cpu() && dense {
            device_allocation_bytes(raw_bytes, device)
        } else {
            0
        },
        // Loading keeps an owned host upload. Dense decoding also overlaps
        // a byte readback with typed values, then half values with F32.
        cpu_upload_bytes: if device.is_cpu() {
            if dense {
                raw_bytes.saturating_mul(2)
            } else {
                0
            }
        } else if dense {
            raw_bytes.saturating_add(raw_bytes.max(f32_bytes))
        } else {
            raw_bytes
        },
        ..Default::default()
    })
}

fn is_token_embedding(name: &str) -> bool {
    name == "token_embd.weight" || name.starts_with("token_embd.")
}

fn dequantized_f32_bytes(elem_count: usize) -> u64 {
    (elem_count as u64).saturating_mul(std::mem::size_of::<f32>() as u64)
}

fn realized_head_embedding_bytes(
    architecture: &str,
    encoded_bytes: u64,
    dequantized_bytes: u64,
) -> u64 {
    // These native paths retain the GGUF QTensor and decode only selected
    // token rows. Other architectures, including Llama-family CPU stages,
    // materialize a full F32 lookup table after construction.
    if matches!(
        architecture,
        "qwen35" | "glm5next" | "glm5-next" | "muse-spark"
    ) {
        encoded_bytes
    } else {
        dequantized_bytes
    }
}

fn head_embedding_cpu_transient_bytes(
    architecture: &str,
    encoded_bytes: u64,
    uses_accelerator: bool,
) -> u64 {
    // Llama-family constructors dequantize the embedding while the source
    // QTensor remains owned by StageTensors. Qwen35 and GLM5 retain the
    // packed lookup tensor directly and therefore have no such CPU peak.
    if !uses_accelerator
        && !matches!(
            architecture,
            "qwen35" | "glm5next" | "glm5-next" | "muse-spark"
        )
    {
        encoded_bytes
    } else {
        0
    }
}

fn decode_fallback_f16_bytes(
    architecture: &str,
    name: &str,
    dtype: GgmlDType,
    elem_count: usize,
    is_metal: bool,
) -> u64 {
    let uses_decode_fallback = (architecture == "muse-spark"
        && (name == "output.weight"
            || name.ends_with(".attn_q.weight")
            || name.ends_with(".attn_k.weight")
            || name.ends_with(".attn_v.weight")
            || name.ends_with(".attn_output.weight")
            || name.ends_with(".attn_gate.weight")
            || name.ends_with(".ffn_latent_down.weight")
            || name.ends_with(".ffn_latent_up.weight")))
        || matches!(architecture, "llama4" | "minimax-m2")
            && (name == "output.weight"
                || name == "token_embd.weight"
                || name.ends_with(".attn_v.weight"));
    if is_metal && uses_decode_fallback && matches!(dtype, GgmlDType::Q4K | GgmlDType::Q6K) {
        (elem_count as u64).saturating_mul(std::mem::size_of::<half::f16>() as u64)
    } else {
        0
    }
}

#[cfg(test)]
fn tensor_memory_contribution(
    architecture: &str,
    name: &str,
    tensor_bytes: u64,
    packed_experts_use_accelerator: bool,
) -> (u64, u64) {
    let contribution = tensor_memory_accounting(
        architecture,
        name,
        tensor_bytes,
        0,
        GgmlDType::Q4K,
        packed_experts_use_accelerator,
        true,
        true,
    );
    (
        contribution.accelerator_bytes,
        contribution
            .retained_cpu_bytes
            .saturating_add(contribution.cpu_source_bytes)
            .saturating_add(contribution.cpu_upload_bytes),
    )
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct TensorMemoryAccounting {
    /// Final bytes retained by the stage on the accelerator. For a CPU
    /// runtime this is still the generic resident tensor component; the
    /// admission layer folds it into system RAM.
    accelerator_bytes: u64,
    /// Final bytes retained on the host after construction. The estimator may
    /// also add `cpu_source_bytes` to the profile's host component because
    /// stage loading preloads those sources before construction starts.
    retained_cpu_bytes: u64,
    /// One source tensor that can overlap the final accelerator allocation
    /// while a device-side tensor is materialized.
    accelerator_transient_bytes: u64,
    /// Preloaded device sources that overlap the F32 destinations. Eager
    /// loaders retain the whole stage pool; streaming loaders retain a layer.
    accelerator_source_bytes: u64,
    /// CPU source tensors which must be accounted for while one layer is
    /// materialized. The estimator places these in the profile's transient
    /// component for streaming architectures.
    cpu_source_bytes: u64,
    /// One owned host upload buffer. The estimator takes the maximum across
    /// tensors in a layer/output so it does not charge one copy per tensor.
    cpu_upload_bytes: u64,
}

/// Q2's safe Metal projection owns a row-padded packed copy. The source and
/// readback remain live while that copy uploads; the readback uses Candle's
/// power-of-two allocator, while the retained byte tensor has exact size.
fn metal_q2_projection_accounting(shape: &[usize]) -> Result<TensorMemoryAccounting> {
    let [rows, columns] = shape else {
        anyhow::bail!("Metal Q2 projection requires a matrix")
    };
    anyhow::ensure!(
        *rows > 0 && *columns > 0 && columns.is_multiple_of(256),
        "Metal Q2 projection requires nonzero rows and 256-aligned columns"
    );
    let rows = u64::try_from(*rows)?;
    let columns = u64::try_from(*columns)?;
    let row_bytes = (columns / 256)
        .checked_mul(84)
        .context("Metal Q2 row byte overflow")?;
    let original = rows
        .checked_mul(row_bytes)
        .context("Metal Q2 weight byte overflow")?;
    let padded_rows = rows
        .checked_add(7)
        .context("Metal Q2 padded row overflow")?
        / 8
        * 8;
    let padded = padded_rows
        .checked_mul(row_bytes)
        .context("Metal Q2 padded byte overflow")?;
    anyhow::ensure!(
        padded <= 2 * 1024 * 1024 * 1024
            && columns <= i32::MAX as u64
            && padded_rows <= i32::MAX as u64,
        "Metal Q2 packed weights exceed supported bounds"
    );
    let readback = original
        .checked_next_power_of_two()
        .context("Metal Q2 readback byte overflow")?;
    Ok(TensorMemoryAccounting {
        accelerator_bytes: padded,
        accelerator_source_bytes: original,
        accelerator_transient_bytes: readback,
        cpu_upload_bytes: if padded == original {
            original
        } else {
            original.saturating_add(padded)
        },
        ..Default::default()
    })
}

fn account_tied_metal_q2_output(profile: &mut StageMemoryProfile, output: TensorMemoryAccounting) {
    // Head/tail boundaries use max for shared storage. This projection owns
    // its packed copy, so put it in the independent output component.
    profile.output_gpu_bytes = profile
        .output_gpu_bytes
        .saturating_add(output.accelerator_bytes)
        .saturating_add(output.accelerator_source_bytes);
    profile.output_gpu_transient_bytes = profile
        .output_gpu_transient_bytes
        .max(output.accelerator_transient_bytes);
    profile.output_cpu_transient_bytes = profile
        .output_cpu_transient_bytes
        .max(output.cpu_upload_bytes);
}

#[allow(clippy::too_many_arguments)]
fn tensor_memory_accounting_with_shape(
    architecture: &str,
    name: &str,
    tensor_bytes: u64,
    shape: &[usize],
    dtype: GgmlDType,
    uses_accelerator: bool,
    is_metal: bool,
    metal_tensor_qmatmul_available: bool,
) -> Result<TensorMemoryAccounting> {
    let elements = shape
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .context("GGUF tensor element count overflow")?;
    let mut contribution = tensor_memory_accounting(
        architecture,
        name,
        tensor_bytes,
        elements,
        dtype,
        uses_accelerator,
        is_metal,
        metal_tensor_qmatmul_available,
    );
    if !uses_accelerator
        || !is_metal
        || dtype != GgmlDType::Q2K
        || is_token_embedding(name)
        || matches!(architecture, "deepseek4" | "deepseek41")
    {
        return Ok(contribution);
    }
    if name.contains("_exps.") {
        if let [experts, rows, columns] = shape {
            anyhow::ensure!(*experts > 0, "Metal Q2 expert bank is empty");
            let split = metal_q2_projection_accounting(&[*rows, *columns])?;
            anyhow::ensure!(
                split.accelerator_source_bytes.checked_mul(*experts as u64) == Some(tensor_bytes),
                "Metal Q2 expert storage extent differs from shape"
            );
            // Indexed kernels keep the original bank. A split fallback owns
            // one padded copy per expert, but only one upload/readback at a
            // time. Reserve that fallback even when this tensor's dtype has
            // an indexed kernel: a sibling tensor can force the whole bank
            // to split. A prior source can remain pooled until the readback.
            contribution.accelerator_bytes =
                split.accelerator_bytes.saturating_mul(*experts as u64);
            contribution.accelerator_transient_bytes =
                contribution.accelerator_transient_bytes.max(
                    split
                        .accelerator_source_bytes
                        .saturating_mul(2)
                        .saturating_add(split.accelerator_transient_bytes),
                );
            contribution.cpu_upload_bytes =
                contribution.cpu_upload_bytes.max(split.cpu_upload_bytes);
        }
        return Ok(contribution);
    }
    let is_glm = matches!(architecture, "glm5next" | "glm5-next");
    let dense = name.contains("norm.")
        || name.contains("conv1d")
        || (name.ends_with(".ffn_gate_inp.weight") && !is_glm)
        || (is_glm
            && (glm5_dense_tensor(name)
                || name.ends_with(".attn_k_b.weight")
                || name.ends_with(".attn_v_b.weight")))
        || (architecture == "muse-spark"
            && (name.ends_with("_alpha.weight") || name.ends_with("_beta.weight")))
        || (matches!(architecture, "nemotron_h" | "nemotron_h_moe")
            && (name.ends_with(".ssm_a")
                || name.ends_with(".ssm_d")
                || name.ends_with(".ssm_dt.bias")))
        || (architecture == "gemma4" && name.ends_with(".layer_output_scale.weight"));
    if shape.len() == 2 && !dense {
        contribution = metal_q2_projection_accounting(shape)?;
        anyhow::ensure!(
            contribution.accelerator_source_bytes == tensor_bytes,
            "Metal Q2 projection storage extent differs from shape"
        );
    }
    Ok(contribution)
}

#[derive(Clone, Copy, Default)]
struct Qwen35GroupedConstructionBytes {
    mlp_gate: u64,
    mlp_up: u64,
    deltanet_qkv: u64,
    deltanet_gate: u64,
    deltanet_alpha: u64,
    deltanet_beta: u64,
    attention_q: u64,
    attention_k: u64,
    attention_v: u64,
}

impl Qwen35GroupedConstructionBytes {
    fn record(&mut self, name: &str, bytes: u64) {
        let slot = if name.ends_with(".ffn_gate.weight") {
            &mut self.mlp_gate
        } else if name.ends_with(".ffn_up.weight") {
            &mut self.mlp_up
        } else if name.ends_with(".attn_qkv.weight") {
            &mut self.deltanet_qkv
        } else if name.ends_with(".attn_gate.weight") {
            &mut self.deltanet_gate
        } else if name.ends_with(".ssm_alpha.weight") {
            &mut self.deltanet_alpha
        } else if name.ends_with(".ssm_beta.weight") {
            &mut self.deltanet_beta
        } else if name.ends_with(".attn_q.weight") {
            &mut self.attention_q
        } else if name.ends_with(".attn_k.weight") {
            &mut self.attention_k
        } else if name.ends_with(".attn_v.weight") {
            &mut self.attention_v
        } else {
            return;
        };
        *slot = slot.saturating_add(bytes);
    }

    fn grouped_peaks(
        self,
        mlp_grouped: bool,
        deltanet_grouped: bool,
        attention_grouped: bool,
    ) -> (u64, u64) {
        let mlp = mlp_grouped.then(|| self.mlp_gate.saturating_add(self.mlp_up));
        let deltanet_host = deltanet_grouped.then(|| {
            self.deltanet_qkv
                .saturating_add(self.deltanet_gate)
                .max(self.deltanet_alpha.saturating_add(self.deltanet_beta))
        });
        // The two homogeneous groups are built in separate scopes, so the
        // first pair's source tensors are released before the second upload.
        let deltanet_accelerator = deltanet_grouped.then(|| {
            self.deltanet_qkv
                .saturating_add(self.deltanet_gate)
                .max(self.deltanet_alpha.saturating_add(self.deltanet_beta))
        });
        let attention = attention_grouped.then(|| {
            self.attention_q
                .saturating_add(self.attention_k)
                .saturating_add(self.attention_v)
        });
        (
            mlp.unwrap_or(0)
                .max(deltanet_accelerator.unwrap_or(0))
                .max(attention.unwrap_or(0)),
            mlp.unwrap_or(0)
                .max(deltanet_host.unwrap_or(0))
                .max(attention.unwrap_or(0)),
        )
    }
}

fn qwen35_metal_qmatmul_dtype(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
    )
}

/// Return the exact-ish storage/lifetime contribution for one GGUF tensor.
///
/// The legacy stage loader reads a whole stage into QTensor objects before
/// constructing the architecture. GLM's streaming path instead loads one
/// layer map at a time; its Metal tensor matmul copies one ordinary QTensor
/// through a temporary host Vec before creating its final Metal tensor, while
/// packed experts and DSA headwise projections start on the CPU and are
/// uploaded one layer at a time. Those lifetimes are represented separately
/// so the caller can charge `sum(all sources in one layer) + max(one upload)`.
#[allow(clippy::too_many_arguments)]
fn tensor_memory_accounting(
    architecture: &str,
    name: &str,
    tensor_bytes: u64,
    elem_count: usize,
    dtype: GgmlDType,
    uses_accelerator: bool,
    is_metal: bool,
    metal_tensor_qmatmul_available: bool,
) -> TensorMemoryAccounting {
    if architecture == "qwen3moe"
        && name.ends_with("_exps.weight")
        && dtype == GgmlDType::BF16
        && (!uses_accelerator || is_metal)
    {
        // The BF16 expert path shares the original CPU bank and caches selected
        // experts for one layer call. Reserve the F32 fallback even when Metal
        // can keep packed BF16 weights. Gate/up/down have equal element counts;
        // four full F32 banks cover their three caches and one temporary copy.
        // The layer fence keeps these caches from overlapping across layers.
        // Grouped Metal prefill has a separate fixed temporary-storage cap.
        let workspace = dequantized_f32_bytes(elem_count).saturating_mul(4);
        return TensorMemoryAccounting {
            retained_cpu_bytes: tensor_bytes,
            accelerator_transient_bytes: if is_metal {
                workspace.saturating_add(
                    crate::architectures::qwen3moe::BF16_GROUP_WORKSPACE_BYTES as u64,
                )
            } else {
                0
            },
            cpu_upload_bytes: if is_metal {
                tensor_bytes.saturating_add(dequantized_f32_bytes(elem_count))
            } else {
                workspace
            },
            ..Default::default()
        };
    }
    let is_glm5 = matches!(architecture, "glm5next" | "glm5-next");
    let is_glm5_headwise =
        is_glm5 && (name.ends_with(".attn_k_b.weight") || name.ends_with(".attn_v_b.weight"));
    let is_packed_expert = name.contains("_exps.");
    // Packed floating experts can fall back to per-expert Candle QMatMuls,
    // which materialize F32. Reserve that path even when a device may support
    // an indexed F16 kernel: dtype alone does not guarantee that dispatch.
    let floating_experts =
        is_packed_expert && matches!(dtype, GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::F32);
    let headwise_indexed =
        is_glm5_headwise && indexed_headwise_available(is_metal, uses_accelerator, dtype);

    if qwen35_cpu_packed_embedding(architecture, name, dtype, is_metal) {
        return TensorMemoryAccounting {
            retained_cpu_bytes: tensor_bytes,
            ..Default::default()
        };
    }

    if tensor_loads_on_cpu(architecture, name, dtype, is_metal) {
        if !uses_accelerator {
            // CPU-only GLM headwise projections are dequantized to F32 by
            // HeadwiseMatMul::Dense. Quantized experts remain packed after
            // splitting, but floating experts become F32 QMatMuls.
            let retained_cpu_bytes = if is_glm5_headwise || floating_experts {
                dequantized_f32_bytes(elem_count)
            } else {
                tensor_bytes
            };
            return TensorMemoryAccounting {
                retained_cpu_bytes: if is_glm5_headwise && !headwise_indexed {
                    dequantized_f32_bytes(elem_count)
                } else {
                    retained_cpu_bytes
                },
                cpu_source_bytes: if is_glm5_headwise || is_packed_expert {
                    tensor_bytes
                } else {
                    0
                },
                ..Default::default()
            };
        }

        // Both packed experts and GLM DSA headwise projections have a device
        // representation in the Metal/CUDA indexed path. Their CPU QTensor
        // sources remain live until the containing layer is constructed, even
        // on Metal's unified-memory devices, so the source pool must be kept in
        // `cpu_source_bytes` for every accelerator. The extra owned upload is
        // a separate per-layer peak. This is intentionally conservative for
        // backend copies which may be performed by the driver.
        let indexed_device_copy = is_packed_expert || headwise_indexed;
        return TensorMemoryAccounting {
            accelerator_bytes: if is_packed_expert || is_glm5_headwise {
                if (is_glm5_headwise && !headwise_indexed) || floating_experts {
                    dequantized_f32_bytes(elem_count)
                } else {
                    tensor_bytes
                }
            } else {
                0
            },
            cpu_source_bytes: tensor_bytes,
            // The per-expert temporary is no larger than the whole packed
            // tensor. Include its upload/dequantization overlap conservatively.
            accelerator_transient_bytes: if floating_experts {
                if is_metal {
                    tensor_bytes.saturating_mul(2)
                } else {
                    tensor_bytes
                }
            } else {
                0
            },
            cpu_upload_bytes: if is_metal && floating_experts {
                tensor_bytes.saturating_add(dequantized_f32_bytes(elem_count))
            } else if is_metal {
                if indexed_device_copy {
                    // The source is charged separately above. Indexed Metal
                    // construction creates one owned CPU upload buffer.
                    tensor_bytes
                } else {
                    // Dense headwise fallback dequantizes through an F32 CPU
                    // buffer before copying the result to Metal.
                    dequantized_f32_bytes(elem_count)
                }
            } else {
                if is_glm5_headwise && !headwise_indexed {
                    // CUDA dense fallback also dequantizes into a temporary
                    // host F32 buffer before the final device tensor.
                    dequantized_f32_bytes(elem_count)
                } else {
                    // CUDA copies from the preloaded CPU QTensor. Charge one
                    // additional staging-sized buffer conservatively.
                    tensor_bytes
                }
            },
            ..Default::default()
        };
    }

    let is_glm5_dense = is_glm5 && glm5_dense_tensor(name);
    // Muse keeps routing and normalization in F32 even when a future GGUF
    // quantizes these tensors. All other projections retain packed weights.
    let is_muse_dense = architecture == "muse-spark"
        && (name.ends_with(".ffn_gate_inp.weight")
            || name.ends_with("norm.weight")
            || name.ends_with(".ffn_exp_probs_b.bias")
            || name.ends_with("_alpha.weight")
            || name.ends_with("_beta.weight"));
    let is_dense = is_glm5_dense || is_muse_dense;
    // Dense QMatMul and norm construction create a fresh F32 tensor even
    // when the source already uses F32. Keep that source until its eager
    // stage or streamed layer constructor returns. Embeddings have their
    // own boundary accounting; packed experts took the branch above.
    let materializes_f32 = !is_token_embedding(name)
        && matches!(dtype, GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::F32);
    let accelerator_bytes = if is_dense || materializes_f32 {
        // GLM norms, mHC coefficients, KDA state coefficients, and indexer
        // normalizers use `QTensor::dequantize`, whose final Tensor is F32.
        dequantized_f32_bytes(elem_count)
    } else {
        tensor_bytes
    };

    // Ordinary GLM qmatmuls use MetalTensorQMatMul on Metal 4: the source
    // Metal QTensor, an owned CPU Vec from `data()`, and the destination Metal
    // tensor overlap during construction. Dense F32 materialization also
    // overlaps its source QTensor, but does not create that host Vec.
    let mut accelerator_transient_bytes = if uses_accelerator && materializes_f32 {
        // The original source pool is charged separately. Metal additionally
        // blits one encoded tensor before converting it on the host.
        if is_metal { tensor_bytes } else { 0 }
    } else if uses_accelerator && is_dense {
        if is_metal {
            // QMetalStorage::dequantize first blits the packed source into a
            // temporary Metal buffer, then allocates the final F32 tensor.
            tensor_bytes.saturating_mul(2)
        } else {
            // CUDA dequantization keeps the source QTensor live while its
            // final F32 tensor is materialized.
            tensor_bytes
        }
    } else if metal_tensor_qmatmul_available && !is_dense {
        // Metal 4's custom QMatMul copies one ordinary source QTensor through
        // an owned host Vec before allocating the destination Metal tensor.
        tensor_bytes
    } else {
        0
    };
    let qwen35_metal_qmatmul = architecture == "qwen35"
        && is_metal
        && uses_accelerator
        && metal_tensor_qmatmul_available
        && qwen35_metal_qmatmul_dtype(dtype);
    let mut cpu_upload_bytes = if is_metal && uses_accelerator && materializes_f32 {
        // QMetalStorage::dequantize overlaps an encoded readback Vec and an
        // F32 output Vec, before uploading the latter to the device.
        tensor_bytes.saturating_add(dequantized_f32_bytes(elem_count))
    } else if is_metal && uses_accelerator && (is_glm5 || is_muse_dense) {
        if is_dense {
            // QMetalStorage::dequantize materializes an F32 Vec on the host.
            dequantized_f32_bytes(elem_count)
        } else if metal_tensor_qmatmul_available {
            tensor_bytes
        } else {
            0
        }
    } else if qwen35_metal_qmatmul {
        // Qwen3.5 uses the same owned host Vec while copying a packed QTensor
        // into MetalTensorQMatMul. Grouped construction raises this per-tensor
        // value to the sum of each homogeneous group above.
        tensor_bytes
    } else {
        0
    };
    if architecture == "muse-spark" && uses_accelerator {
        let fallback_bytes =
            decode_fallback_f16_bytes(architecture, name, dtype, elem_count, is_metal);
        if fallback_bytes > 0 {
            // DecodeSafe's F16 conversion owns the encoded readback, the full
            // F16 output, and one typed-block/F32 decode chunk at the same time.
            // Its Metal readback also blits the packed tensor into a buffer.
            let chunk_blocks = (elem_count / dtype.block_size()).min(4096) as u64;
            let chunk_bytes =
                chunk_blocks.saturating_mul((dtype.type_size() as u64).saturating_add(
                    (dtype.block_size() as u64).saturating_mul(std::mem::size_of::<f32>() as u64),
                ));
            cpu_upload_bytes = cpu_upload_bytes.max(
                tensor_bytes
                    .saturating_add(fallback_bytes)
                    .saturating_add(chunk_bytes),
            );
            accelerator_transient_bytes = accelerator_transient_bytes.max(tensor_bytes);
        }
    }
    let cpu_source_bytes = if !uses_accelerator && (is_dense || materializes_f32) {
        // On CPU the dequantized F32 destination overlaps the original CPU
        // QTensor until the constructor returns.
        tensor_bytes
    } else {
        0
    };

    TensorMemoryAccounting {
        accelerator_bytes,
        accelerator_transient_bytes,
        accelerator_source_bytes: if uses_accelerator && (materializes_f32 || is_muse_dense) {
            tensor_bytes
        } else {
            0
        },
        cpu_source_bytes,
        cpu_upload_bytes,
        ..Default::default()
    }
}

fn indexed_headwise_available(is_metal: bool, uses_accelerator: bool, dtype: GgmlDType) -> bool {
    if !uses_accelerator {
        return false;
    }
    if is_metal {
        return matches!(
            dtype,
            GgmlDType::F32
                | GgmlDType::F16
                | GgmlDType::Q4_0
                | GgmlDType::Q4_1
                | GgmlDType::Q5_0
                | GgmlDType::Q5_1
                | GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
        ) && std::env::var_os("DINF_DISABLE_INDEXED_MOE_METAL").is_none();
    }

    #[cfg(feature = "cuda")]
    {
        matches!(
            dtype,
            GgmlDType::Q4_0
                | GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
        ) && std::env::var_os("DINF_DISABLE_INDEXED_MOE_CUDA").is_none()
            && (dtype != GgmlDType::Q4_0
                || std::env::var_os("DINF_DISABLE_INDEXED_MOE_Q4_0_CUDA").is_none())
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

#[cfg(test)]
fn tensor_memory_profile_contribution(
    architecture: &str,
    name: &str,
    tensor_bytes: u64,
    packed_experts_use_accelerator: bool,
) -> (u64, u64, u64) {
    let contribution = tensor_memory_accounting(
        architecture,
        name,
        tensor_bytes,
        0,
        GgmlDType::Q4K,
        packed_experts_use_accelerator,
        true,
        true,
    );
    (
        contribution.accelerator_bytes,
        contribution.retained_cpu_bytes,
        contribution
            .cpu_source_bytes
            .saturating_add(contribution.cpu_upload_bytes),
    )
}

fn glm5_dense_tensor(name: &str) -> bool {
    name.ends_with(".attn_norm.weight")
        || name.ends_with(".ffn_norm.weight")
        || name.ends_with(".ssm_norm.weight")
        || name.ends_with(".attn_q_a_norm.weight")
        || name.ends_with(".attn_kv_a_norm.weight")
        || name.ends_with(".indexer.k_norm.weight")
        || name.ends_with(".indexer.k_norm.bias")
        || name.ends_with(".indexer_compressor_ape.weight")
        || name.ends_with(".exp_probs_b.bias")
        || name.ends_with(".ssm_a")
        || name.ends_with(".ssm_dt.bias")
        || name.ends_with(".ssm_conv1d_q.weight")
        || name.ends_with(".ssm_conv1d_k.weight")
        || name.ends_with(".ssm_conv1d_v.weight")
        || name.ends_with(".hc_attn_base.weight")
        || name.ends_with(".hc_attn_scale.weight")
        || name.ends_with(".hc_ffn_base.weight")
        || name.ends_with(".hc_ffn_scale.weight")
        || name == "output_norm.weight"
}

fn ggml_tensor_bytes(elem_count: usize, block_size: usize, type_size: usize) -> Result<u64> {
    anyhow::ensure!(block_size > 0, "GGUF tensor block size must be positive");
    anyhow::ensure!(type_size > 0, "GGUF tensor type size must be positive");
    let block_size =
        NonZeroUsize::new(block_size).context("GGUF tensor block size must be non-zero")?;
    let blocks = elem_count.div_ceil(block_size.get());
    let bytes = blocks
        .checked_mul(type_size)
        .context("GGUF tensor byte length overflow")?;
    let bytes = u64::try_from(bytes).context("GGUF tensor byte length exceeds u64")?;
    anyhow::ensure!(
        bytes <= MAX_GGUF_TENSOR_BYTES,
        "GGUF tensor byte length ({bytes}) exceeds application limit ({MAX_GGUF_TENSOR_BYTES})"
    );
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf_limits::MAX_MODEL_LAYERS;
    use crate::gguf_limits::{
        MAX_GGUF_HEADER_BYTES, MAX_GGUF_TENSOR_BYTES, MAX_GGUF_TENSOR_COUNT,
        MAX_GGUF_TENSOR_ELEMENTS,
    };
    use std::io::{Seek, Write};

    #[test]
    fn metal_q2_accounting_bounds_padding_and_readback() -> Result<()> {
        let aligned = metal_q2_projection_accounting(&[8, 256])?;
        assert_eq!(aligned.accelerator_bytes, 672);
        assert_eq!(aligned.accelerator_source_bytes, 672);
        assert_eq!(aligned.accelerator_transient_bytes, 1024);
        assert_eq!(aligned.cpu_upload_bytes, 672);
        let tail = metal_q2_projection_accounting(&[9, 256])?;
        assert_eq!(tail.accelerator_bytes, 1344);
        assert_eq!(tail.accelerator_source_bytes, 756);
        assert_eq!(tail.accelerator_transient_bytes, 1024);
        assert_eq!(tail.cpu_upload_bytes, 756 + 1344);
        for shape in [
            vec![256],
            vec![0, 256],
            vec![8, 257],
            vec![usize::MAX, 256],
            vec![32_000_000, 256],
        ] {
            assert!(metal_q2_projection_accounting(&shape).is_err());
        }
        Ok(())
    }

    #[test]
    fn metal_q2_accounting_selects_only_common_projections() -> Result<()> {
        let expected = metal_q2_projection_accounting(&[9, 256])?;
        for (architecture, name) in [
            ("llama", "blk.0.attn_q.weight"),
            ("glm5next", "blk.0.ffn_gate_inp.weight"),
            ("glm5-next", "blk.0.ffn_gate_inp.weight"),
            ("qwen35", "blk.0.ssm_alpha.weight"),
            ("deepseek2", "blk.0.attn_kv_b.weight"),
        ] {
            assert_eq!(
                tensor_memory_accounting_with_shape(
                    architecture,
                    name,
                    756,
                    &[9, 256],
                    GgmlDType::Q2K,
                    true,
                    true,
                    false
                )?,
                expected
            );
        }
        // These paths dequantize, gather, use indexed kernels, or use raw V4
        // weights; their prior accounting must remain unchanged.
        for (architecture, name, accelerator, metal, dtype) in [
            ("llama", "blk.0.attn_q.weight", false, false, GgmlDType::Q2K),
            ("llama", "blk.0.attn_q.weight", true, false, GgmlDType::Q2K),
            ("llama", "blk.0.attn_q.weight", true, true, GgmlDType::Q4K),
            ("qwen35", "token_embd.weight", true, true, GgmlDType::Q2K),
            (
                "minimax-m2",
                "blk.0.ffn_gate_inp.weight",
                true,
                true,
                GgmlDType::Q2K,
            ),
            (
                "llama4",
                "blk.0.ffn_gate_inp.weight",
                true,
                true,
                GgmlDType::Q2K,
            ),
            (
                "glm5next",
                "blk.0.attn_k_b.weight",
                true,
                true,
                GgmlDType::Q2K,
            ),
            (
                "glm5next",
                "blk.0.attn_v_b.weight",
                true,
                true,
                GgmlDType::Q2K,
            ),
            (
                "glm5next",
                "blk.0.attn_q_a_norm.weight",
                true,
                true,
                GgmlDType::Q2K,
            ),
            (
                "muse-spark",
                "blk.0.attn_alpha.weight",
                true,
                true,
                GgmlDType::Q2K,
            ),
            ("nemotron_h", "blk.0.ssm_a", true, true, GgmlDType::Q2K),
            ("nemotron_h_moe", "blk.0.ssm_d", true, true, GgmlDType::Q2K),
            (
                "gemma4",
                "blk.0.layer_output_scale.weight",
                true,
                true,
                GgmlDType::Q2K,
            ),
            (
                "deepseek4",
                "blk.0.attn_q.weight",
                true,
                true,
                GgmlDType::Q2K,
            ),
            (
                "deepseek41",
                "blk.0.attn_q.weight",
                true,
                true,
                GgmlDType::Q2K,
            ),
        ] {
            let bytes = ggml_tensor_bytes(9 * 256, dtype.block_size(), dtype.type_size())?;
            assert_eq!(
                tensor_memory_accounting_with_shape(
                    architecture,
                    name,
                    bytes,
                    &[9, 256],
                    dtype,
                    accelerator,
                    metal,
                    false
                )?,
                tensor_memory_accounting(
                    architecture,
                    name,
                    bytes,
                    9 * 256,
                    dtype,
                    accelerator,
                    metal,
                    false
                ),
                "{architecture} {name}"
            );
        }
        assert!(
            tensor_memory_accounting_with_shape(
                "llama",
                "blk.0.attn_q.weight",
                755,
                &[9, 256],
                GgmlDType::Q2K,
                true,
                true,
                false
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn metal_q2_accounting_covers_split_expert_fallback() -> Result<()> {
        for experts in [1, 3] {
            let bytes = experts as u64 * 756;
            let contribution = tensor_memory_accounting_with_shape(
                "minimax-m2",
                "blk.0.ffn_up_exps.weight",
                bytes,
                &[experts, 9, 256],
                GgmlDType::Q2K,
                true,
                true,
                false,
            )?;
            assert_eq!(contribution.accelerator_bytes, experts as u64 * 1344);
            assert_eq!(contribution.cpu_source_bytes, bytes);
            assert_eq!(contribution.accelerator_source_bytes, 0);
            assert_eq!(contribution.accelerator_transient_bytes, 2 * 756 + 1024);
            assert_eq!(contribution.cpu_upload_bytes, bytes.max(756 + 1344));
        }
        assert!(
            tensor_memory_accounting_with_shape(
                "minimax-m2",
                "blk.0.ffn_up_exps.weight",
                0,
                &[0, 9, 256],
                GgmlDType::Q2K,
                true,
                true,
                false
            )
            .is_err()
        );
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_q2_profile_covers_eager_streamed_and_tied_sources() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let dir = tempfile::tempdir()?;
        let mut profiles = Vec::new();
        for architecture in ["llama", "minimax-m2"] {
            for untied in [false, true] {
                let path = dir.path().join(format!("{architecture}-{untied}.gguf"));
                write_dense_profile_fixture_with_geometry(&path, untied, GgmlDType::Q2K, 32, 9, 2)?;
                if architecture != "llama" {
                    // Reuse the same tiny tensor table for the streaming
                    // profile; only its architecture metadata differs.
                    let mut file = std::fs::File::open(&path)?;
                    let content = gguf_file::Content::read(&mut file)?;
                    let tensors = content
                        .tensor_infos
                        .keys()
                        .map(|name| {
                            Ok((name.clone(), content.tensor(&mut file, name, &Device::Cpu)?))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let metadata = content
                        .metadata
                        .into_iter()
                        .map(|(key, value)| {
                            if key == "general.architecture" {
                                (key, gguf_file::Value::String(architecture.into()))
                            } else {
                                (
                                    key.replacen("llama.", &format!("{architecture}."), 1),
                                    value,
                                )
                            }
                        })
                        .collect::<Vec<_>>();
                    let metadata = metadata
                        .iter()
                        .map(|(key, value)| (key.as_str(), value))
                        .collect::<Vec<_>>();
                    let tensors = tensors
                        .iter()
                        .map(|(name, value)| (name.as_str(), value))
                        .collect::<Vec<_>>();
                    drop(file);
                    gguf_file::write(&mut std::fs::File::create(&path)?, &metadata, &tensors)?;
                }
                let profile = estimate_stage_memory_profile(&path, architecture, 2, &device)?;
                // Seven aligned Q2 projections and two F32 norms per layer.
                let source_pool = 7 * 21_504 + 2 * 1024;
                let (resident, scratch) = if architecture == "llama" {
                    (2 * source_pool, 32_768)
                } else {
                    (source_pool, source_pool + 32_768)
                };
                assert_eq!(profile.layer_gpu_bytes, vec![resident; 2]);
                assert_eq!(profile.layer_gpu_transient_bytes, vec![scratch; 2]);
                assert_eq!(profile.layer_cpu_transient_bytes, vec![21_504; 2]);
                assert_eq!(profile.tail_embedding_gpu_bytes, Some(0));
                // The nine-row output owns sixteen rows; the packed lookup
                // cannot hide that second copy through max(head, tail).
                assert_eq!(profile.output_gpu_bytes, 2 * 1024 + 756 + 1344);
                assert_eq!(profile.output_gpu_transient_bytes, 1024);
                assert_eq!(profile.output_cpu_transient_bytes, 756 + 1344);
                let stage_peak = profile.stage_memory_components(0, 2, 2).unwrap().0;
                let boundary = profile.boundary_gpu_bytes(0, 2, 2);
                assert_eq!(
                    stage_peak - boundary - profile.output_gpu_bytes - profile.common_gpu_bytes,
                    2 * resident + scratch
                );
                profiles.push(profile);
            }
        }
        assert_eq!(profiles[0].output_gpu_bytes, profiles[1].output_gpu_bytes);
        assert_eq!(profiles[2].output_gpu_bytes, profiles[3].output_gpu_bytes);
        Ok(())
    }

    #[test]
    fn packed_experts_count_retained_storage_and_cpu_construction_source() {
        let bytes = 4 * 1024 * 1024;
        assert_eq!(
            tensor_memory_contribution("glm5next", "blk.0.ffn_up_exps.weight", bytes, true),
            (bytes, bytes * 2)
        );
        assert_eq!(
            tensor_memory_contribution("glm5next", "blk.0.ffn_up_exps.weight", bytes, false),
            (0, bytes * 2)
        );
        assert_eq!(
            tensor_memory_profile_contribution(
                "glm5next",
                "blk.0.ffn_up_exps.weight",
                bytes,
                false,
            ),
            (0, bytes, bytes)
        );
    }

    #[test]
    fn ordinary_tensors_keep_device_storage_and_metal_upload_staging() {
        let bytes = 2 * 1024 * 1024;
        assert_eq!(
            tensor_memory_contribution("glm5next", "blk.0.attn_q.weight", bytes, true),
            (bytes, bytes)
        );
        assert_eq!(
            tensor_memory_contribution("glm5next", "blk.0.attn_q.weight", bytes, false),
            (bytes, 0)
        );
    }

    #[test]
    fn glm_headwise_attention_projection_has_final_accelerator_storage() {
        let bytes = 3 * 1024 * 1024;
        assert_eq!(
            tensor_memory_contribution("glm5next", "blk.0.attn_k_b.weight", bytes, true),
            (bytes, bytes * 2)
        );
        assert_eq!(
            tensor_memory_contribution("glm5next", "blk.0.attn_v_b.weight", bytes, true),
            (bytes, bytes * 2)
        );
        // The same names are ordinary device tensors for other architectures.
        assert_eq!(
            tensor_memory_contribution("qwen3", "blk.0.attn_k_b.weight", bytes, true),
            (bytes, 0)
        );
    }

    #[test]
    fn accelerator_cpu_profile_separates_retained_and_upload_peak() {
        let bytes = 3 * 1024 * 1024;
        assert_eq!(
            tensor_memory_profile_contribution("glm5next", "blk.0.attn_k_b.weight", bytes, true,),
            (bytes, 0, bytes * 2)
        );
        assert_eq!(
            tensor_memory_profile_contribution("glm5next", "blk.0.attn_k_b.weight", bytes, false,),
            (0, 0, bytes)
        );
    }

    #[test]
    fn glm_cpu_sources_and_uploads_are_accounted_for_streamed_layers() {
        let bytes = 7 * 1024 * 1024;
        let metal = tensor_memory_accounting(
            "glm5next",
            "blk.0.ffn_up_exps.weight",
            bytes,
            1,
            GgmlDType::Q2K,
            true,
            true,
            true,
        );
        assert_eq!(metal.accelerator_bytes, bytes);
        assert_eq!(metal.cpu_source_bytes, bytes);
        assert_eq!(metal.cpu_upload_bytes, bytes);

        let cuda = tensor_memory_accounting(
            "glm5next",
            "blk.0.ffn_up_exps.weight",
            bytes,
            1,
            GgmlDType::Q2K,
            true,
            false,
            false,
        );
        assert_eq!(cuda.accelerator_bytes, bytes);
        assert_eq!(cuda.cpu_source_bytes, bytes);
        assert_eq!(cuda.cpu_upload_bytes, bytes);
    }

    #[test]
    fn glm_metal_qmatmul_accounts_source_overlap_and_one_host_upload() {
        let bytes = 11 * 1024 * 1024;
        let contribution = tensor_memory_accounting(
            "glm5next",
            "blk.0.attn_q.weight",
            bytes,
            1,
            GgmlDType::Q4K,
            true,
            true,
            true,
        );
        assert_eq!(contribution.accelerator_bytes, bytes);
        assert_eq!(contribution.accelerator_transient_bytes, bytes);
        assert_eq!(contribution.cpu_source_bytes, 0);
        assert_eq!(contribution.cpu_upload_bytes, bytes);
    }

    #[test]
    fn qwen35_metal_qmatmul_accounts_host_upload_staging() {
        let bytes = 19_353_600;
        let contribution = tensor_memory_accounting(
            "qwen35",
            "blk.0.ffn_down.weight",
            bytes,
            1,
            GgmlDType::Q6K,
            true,
            true,
            true,
        );
        assert_eq!(contribution.accelerator_bytes, bytes);
        assert_eq!(contribution.accelerator_transient_bytes, bytes);
        assert_eq!(contribution.cpu_upload_bytes, bytes);
    }

    #[test]
    fn qwen35_grouped_construction_uses_group_sums_without_changing_retained_bytes() {
        let mut construction = Qwen35GroupedConstructionBytes::default();
        for (name, bytes) in [
            ("blk.0.ffn_gate.weight", 19_353_600),
            ("blk.0.ffn_up.weight", 19_353_600),
            ("blk.0.attn_qkv.weight", 17_203_200),
            ("blk.0.attn_gate.weight", 8_601_600),
            ("blk.0.ssm_alpha.weight", 87_040),
            ("blk.0.ssm_beta.weight", 87_040),
            ("blk.3.attn_q.weight", 17_203_200),
            ("blk.3.attn_k.weight", 2_150_400),
            ("blk.3.attn_v.weight", 2_150_400),
        ] {
            construction.record(name, bytes);
        }

        assert_eq!(
            construction.grouped_peaks(true, true, true),
            (38_707_200, 38_707_200)
        );
        assert_eq!(
            construction.grouped_peaks(false, true, false),
            (25_804_800, 25_804_800)
        );
        assert_eq!(
            construction.grouped_peaks(false, false, true),
            (21_504_000, 21_504_000)
        );
    }

    #[test]
    fn glm_dense_tensor_profiles_realized_f32_and_source_overlap() {
        let encoded = 3 * 1024 * 1024;
        let elements = 2_000;
        let contribution = tensor_memory_accounting(
            "glm5next",
            "blk.0.attn_norm.weight",
            encoded,
            elements,
            GgmlDType::Q4K,
            true,
            true,
            true,
        );
        assert_eq!(
            contribution.accelerator_bytes,
            dequantized_f32_bytes(elements)
        );
        assert_eq!(contribution.accelerator_transient_bytes, encoded * 2);
        assert_eq!(
            contribution.cpu_upload_bytes,
            dequantized_f32_bytes(elements)
        );
    }

    #[test]
    fn streaming_layer_sources_are_transient_and_upload_is_single_peak() {
        let mut profile = StageMemoryProfile {
            total_layers: 2,
            layer_gpu_bytes: vec![100, 200],
            layer_gpu_transient_bytes: vec![10, 20],
            layer_cpu_bytes: vec![30, 40],
            layer_cpu_transient_bytes: vec![50, 60],
            head_embedding_gpu_bytes: Some(0),
            tail_embedding_gpu_bytes: Some(0),
            head_embedding_cpu_bytes: Some(0),
            tail_embedding_cpu_bytes: Some(0),
            head_embedding_cpu_transient_bytes: Some(0),
            tail_embedding_cpu_transient_bytes: Some(0),
            ..Default::default()
        };
        // A streaming architecture keeps retained layer bytes separate from
        // the source/upload peak for the layer currently being materialized.
        assert_eq!(
            profile.stage_memory_components(0, 2, 2),
            Some((200 + 100 + 20, 30 + 40, 60))
        );
        profile.layer_cpu_transient_bytes[0] += 70;
        profile.layer_cpu_transient_bytes[1] += 80;
        assert_eq!(
            profile.stage_memory_components(0, 2, 2),
            Some((200 + 100 + 20, 30 + 40, 60 + 80))
        );
    }

    #[test]
    fn qwen3moe_attention_workspace_is_one_stage_peak() {
        let attention_peak = crate::architectures::qwen3moe::F32_ATTENTION_WORKSPACE_BYTES as u64;
        let larger_expert_peak = attention_peak * 3;
        let mut profile = StageMemoryProfile {
            total_layers: 3,
            layer_gpu_bytes: vec![100, 200, 300],
            layer_gpu_transient_bytes: vec![10, larger_expert_peak, 30],
            layer_cpu_bytes: vec![0; 3],
            layer_cpu_transient_bytes: vec![0; 3],
            ..Default::default()
        };
        account_qwen3moe_attention_workspace(&mut profile, "qwen3moe", 128, true);
        assert_eq!(
            profile.layer_gpu_transient_bytes,
            [
                10 + attention_peak,
                larger_expert_peak + attention_peak,
                30 + attention_peak,
            ]
        );
        assert_eq!(
            profile.stage_memory_components(0, 3, 3),
            Some((600 + larger_expert_peak + attention_peak, 0, 0))
        );
        assert_eq!(
            profile.stage_memory_components(0, 1, 3),
            Some((100 + 10 + attention_peak, 0, 0))
        );
    }

    #[test]
    fn qwen3moe_attention_workspace_only_charges_the_metal_head128_path() {
        for (architecture, head_dim, is_metal) in [
            ("qwen3moe", 128, false),
            ("qwen3moe", 64, true),
            ("llama", 128, true),
            ("qwen35", 128, true),
        ] {
            let mut profile = StageMemoryProfile {
                layer_gpu_transient_bytes: vec![13, 29],
                ..Default::default()
            };
            account_qwen3moe_attention_workspace(&mut profile, architecture, head_dim, is_metal);
            assert_eq!(profile.layer_gpu_transient_bytes, [13, 29]);
        }
    }

    #[test]
    fn qwen35_q4k_cpu_lookup_policy_is_device_dtype_and_name_specific() {
        for architecture in ["qwen35", "qwen3", "llama", "minimax-m2"] {
            for name in [
                "token_embd.weight",
                "output.weight",
                "token_embd.bias",
                "blk.0.attn_q.weight",
            ] {
                for dtype in [
                    GgmlDType::Q4K,
                    GgmlDType::Q6K,
                    GgmlDType::Q8_0,
                    GgmlDType::F32,
                ] {
                    for is_metal in [false, true] {
                        let expected = architecture == "qwen35"
                            && name == "token_embd.weight"
                            && dtype == GgmlDType::Q4K
                            && is_metal;
                        assert_eq!(
                            tensor_loads_on_cpu(architecture, name, dtype, is_metal),
                            expected
                        );
                    }
                }
            }
        }
        // The existing packed expert and GLM headwise policies still apply.
        assert!(tensor_loads_on_cpu(
            "llama",
            "blk.0.ffn_up_exps.weight",
            GgmlDType::Q4K,
            false
        ));
        assert!(tensor_loads_on_cpu(
            "glm5next",
            "blk.0.attn_k_b.weight",
            GgmlDType::F16,
            true
        ));
        let contribution = tensor_memory_accounting(
            "qwen35",
            "token_embd.weight",
            1152,
            2048,
            GgmlDType::Q4K,
            true,
            true,
            true,
        );
        assert_eq!(contribution.retained_cpu_bytes, 1152);
        assert_eq!(contribution.accelerator_bytes, 0);
        assert_eq!(contribution.cpu_source_bytes, 0);
        assert_eq!(contribution.cpu_upload_bytes, 0);
    }

    #[test]
    fn qwen35_cpu_lookup_profile_counts_tied_head_without_duplicate_lookup() {
        for has_output in [false, true] {
            let mut profile = StageMemoryProfile::default();
            set_qwen35_cpu_embedding_boundaries(&mut profile, 1152, has_output);
            assert_eq!(profile.boundary_gpu_bytes(0, 1, 2), 0);
            assert_eq!(profile.boundary_cpu_bytes(0, 1, 2), 1152);
            assert_eq!(profile.boundary_cpu_bytes(1, 2, 2), 0);
            assert_eq!(profile.boundary_cpu_bytes(0, 2, 2), 1152);
            let tail_gpu = if has_output { 0 } else { 1152 };
            let upload = if has_output { 0 } else { 2304 };
            assert_eq!(profile.boundary_gpu_bytes(1, 2, 2), tail_gpu);
            assert_eq!(profile.boundary_gpu_bytes(0, 2, 2), tail_gpu);
            assert_eq!(profile.boundary_cpu_transient_bytes(0, 1, 2), 1152);
            assert_eq!(profile.boundary_cpu_transient_bytes(1, 2, 2), upload);
            assert_eq!(profile.boundary_cpu_transient_bytes(0, 2, 2), 1152 + upload);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn qwen35_q4k_lookup_loads_on_cpu_in_single_split_and_streaming_paths() -> Result<()> {
        use candle_core::{DType, Tensor, quantized::QTensor};
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let directory = tempfile::tempdir()?;
        let values = (0..8 * 256)
            .map(|i| (i as f32 * 0.37).sin() * 0.23)
            .collect::<Vec<_>>();
        let source = QTensor::quantize(
            &Tensor::from_vec(values, (8, 256), &Device::Cpu)?,
            GgmlDType::Q4K,
        )?;
        let norm = QTensor::quantize(
            &Tensor::ones(256, DType::F32, &Device::Cpu)?,
            GgmlDType::F32,
        )?;
        let ids = Tensor::new(&[7u32, 0, 7], &Device::Cpu)?;
        let expected = source.embedding(&ids)?.flatten_all()?.to_vec1::<f32>()?;
        let packed = source.storage_size_in_bytes() as u64;
        let metadata = [
            (
                "general.architecture",
                gguf_file::Value::String("qwen35".into()),
            ),
            ("qwen35.block_count", gguf_file::Value::U32(2)),
            ("qwen35.embedding_length", gguf_file::Value::U32(256)),
            ("qwen35.attention.head_count", gguf_file::Value::U32(4)),
            ("qwen35.attention.head_count_kv", gguf_file::Value::U32(4)),
            ("qwen35.attention.key_length", gguf_file::Value::U32(64)),
            ("qwen35.context_length", gguf_file::Value::U32(32)),
            ("qwen35.vocab_size", gguf_file::Value::U32(8)),
        ];
        let metadata = metadata.iter().map(|(k, v)| (*k, v)).collect::<Vec<_>>();
        for has_output in [false, true] {
            let mut embedding_tensors = vec![("token_embd.weight", &source)];
            if has_output {
                embedding_tensors.push(("output.weight", &source));
            }
            let single = directory.path().join(format!("single-{has_output}.gguf"));
            let first = directory
                .path()
                .join(format!("split-{has_output}-00001-of-00002.gguf"));
            let second = directory
                .path()
                .join(format!("split-{has_output}-00002-of-00002.gguf"));
            gguf_file::write(
                &mut std::fs::File::create(&single)?,
                &metadata,
                &embedding_tensors,
            )?;
            gguf_file::write(
                &mut std::fs::File::create(&first)?,
                &metadata,
                &[("output_norm.weight", &norm)],
            )?;
            gguf_file::write(
                &mut std::fs::File::create(&second)?,
                &metadata,
                &embedding_tensors,
            )?;
            for path in [&single, &first] {
                let profile = estimate_stage_memory_profile(path, "qwen-cpu-lookup", 2, &device)?;
                assert_eq!(profile.head_embedding_cpu_bytes, Some(packed));
                assert_eq!(profile.head_embedding_gpu_bytes, Some(0));
                assert_eq!(profile.tail_embedding_cpu_bytes, Some(0));
                assert_eq!(
                    profile.tail_embedding_gpu_bytes,
                    Some(if has_output { 0 } else { packed })
                );
                let mut streaming = StageTensorLoader::open(path)?;
                for range in [
                    LayerRange::new(0, 1, 2),
                    LayerRange::new(1, 2, 2),
                    LayerRange::solo(2),
                ] {
                    let direct = load_stage_tensors(path, range.clone(), &device)?;
                    let streamed = streaming.load_range(range.clone(), &device)?;
                    for stage in [direct, streamed] {
                        let lookup = stage.tensors.get("token_embd.weight");
                        if range.is_head() || !has_output {
                            let lookup = lookup.expect("head or tied tail needs embedding");
                            assert!(lookup.device().is_cpu());
                            assert_eq!(lookup.dtype(), GgmlDType::Q4K);
                            assert_eq!(lookup.storage_size_in_bytes() as u64, packed);
                            let actual = lookup.embedding(&ids)?.flatten_all()?.to_vec1::<f32>()?;
                            assert!(
                                actual
                                    .iter()
                                    .zip(&expected)
                                    .all(|(a, b)| a.to_bits() == b.to_bits())
                            );
                        } else {
                            assert!(lookup.is_none());
                        }
                        if range.is_tail() && has_output {
                            assert!(stage.tensors["output.weight"].device().is_metal());
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn quantized_embedding_profile_uses_realized_f32_size() {
        assert_eq!(dequantized_f32_bytes(202_048 * 5_120), 4_137_943_040);
    }

    #[test]
    fn quantized_row_lookup_profiles_keep_packed_embedding_storage() {
        assert_eq!(realized_head_embedding_bytes("qwen35", 256, 1024), 256);
        assert_eq!(realized_head_embedding_bytes("glm5next", 256, 1024), 256);
        assert_eq!(realized_head_embedding_bytes("llama", 256, 1024), 1024);
        assert_eq!(realized_head_embedding_bytes("glm5-next", 256, 1024), 256);
    }

    #[test]
    fn deepseek4_profile_counts_embedding_only_at_the_head() {
        let mut profile = StageMemoryProfile {
            embedding_gpu_bytes: 768 * 1024 * 1024,
            ..Default::default()
        };

        set_deepseek4_embedding_boundaries(&mut profile);

        assert_eq!(
            profile.head_embedding_gpu_bytes,
            Some(profile.embedding_gpu_bytes)
        );
        assert_eq!(profile.tail_embedding_gpu_bytes, Some(0));
    }

    #[test]
    fn metal_decode_fallback_counts_retained_f16_weights() {
        let elements = 202_048 * 5_120;
        let expected = (elements * std::mem::size_of::<half::f16>()) as u64;
        assert_eq!(
            decode_fallback_f16_bytes("llama4", "output.weight", GgmlDType::Q6K, elements, true,),
            expected
        );
        assert_eq!(
            decode_fallback_f16_bytes(
                "minimax-m2",
                "blk.3.attn_v.weight",
                GgmlDType::Q4K,
                4096,
                true,
            ),
            8192
        );
    }

    #[test]
    fn muse_decode_fallback_counts_host_conversion_peak() {
        let elements = 256 * 8192;
        let dtype = GgmlDType::Q4K;
        let encoded = (elements / dtype.block_size() * dtype.type_size()) as u64;
        let contribution = tensor_memory_accounting(
            "muse-spark",
            "output.weight",
            encoded,
            elements,
            dtype,
            true,
            true,
            false,
        );
        let output_f16 = (elements * std::mem::size_of::<half::f16>()) as u64;
        let chunk = (4096 * (dtype.type_size() + dtype.block_size() * 4)) as u64;
        assert_eq!(contribution.accelerator_bytes, encoded);
        assert_eq!(contribution.accelerator_transient_bytes, encoded);
        assert_eq!(contribution.cpu_upload_bytes, encoded + output_f16 + chunk);

        let no_fallback = tensor_memory_accounting(
            "muse-spark",
            "output.weight",
            encoded,
            elements,
            GgmlDType::Q8_0,
            true,
            true,
            false,
        );
        assert_eq!(no_fallback.cpu_upload_bytes, 0);
        assert_eq!(no_fallback.accelerator_transient_bytes, 0);
    }

    #[test]
    fn decode_fallback_memory_is_arch_dtype_and_device_specific() {
        for (architecture, dtype, is_metal) in [
            ("llama", GgmlDType::Q6K, true),
            ("llama4", GgmlDType::Q2K, true),
            ("llama4", GgmlDType::Q6K, false),
        ] {
            assert_eq!(
                decode_fallback_f16_bytes(
                    architecture,
                    "blk.0.attn_v.weight",
                    dtype,
                    4096,
                    is_metal,
                ),
                0
            );
        }
        assert_eq!(
            decode_fallback_f16_bytes("llama4", "blk.0.attn_q.weight", GgmlDType::Q6K, 4096, true,),
            0
        );
    }

    #[test]
    fn test_deepseek2_mla_dimension_fallbacks() {
        let md = HashMap::from([
            (
                "general.architecture".to_string(),
                gguf_file::Value::String("deepseek2".to_string()),
            ),
            (
                "deepseek2.block_count".to_string(),
                gguf_file::Value::U32(27),
            ),
            (
                "deepseek2.embedding_length".to_string(),
                gguf_file::Value::U32(2048),
            ),
            (
                "deepseek2.attention.head_count".to_string(),
                gguf_file::Value::U32(16),
            ),
            (
                "deepseek2.attention.key_length".to_string(),
                gguf_file::Value::U32(192),
            ),
            (
                "deepseek2.rope.dimension_count".to_string(),
                gguf_file::Value::U32(64),
            ),
            (
                "deepseek2.attention.kv_lora_rank".to_string(),
                gguf_file::Value::U32(512),
            ),
            (
                "deepseek2.attention.value_length".to_string(),
                gguf_file::Value::U32(128),
            ),
        ]);

        let parsed = parse_gguf_metadata(&md).unwrap();
        assert_eq!(parsed.qk_nope_head_dim, 128);
        assert_eq!(parsed.qk_rope_head_dim, 64);
        assert_eq!(parsed.v_head_dim, 128);
    }

    #[test]
    fn mla_metadata_and_profiles_reject_invalid_rotary_dimensions() -> Result<()> {
        for architecture in ["deepseek2", "glm-dsa"] {
            let mut metadata = HashMap::from([(
                "general.architecture".to_string(),
                gguf_file::Value::String(architecture.to_string()),
            )]);
            for (key, value) in [
                ("block_count", 2),
                ("embedding_length", 8),
                ("attention.head_count", 2),
                ("attention.key_length", 4),
                ("attention.kv_lora_rank", 2),
                ("context_length", 4),
                ("rope.dimension_count", 2),
            ] {
                metadata.insert(
                    format!("{architecture}.{key}"),
                    gguf_file::Value::U32(value),
                );
            }
            let rotary_key = format!("{architecture}.rope.dimension_count");
            assert_eq!(parse_gguf_metadata(&metadata)?.qk_rope_head_dim, 2);
            let dir = tempfile::tempdir()?;
            let path = dir.path().join("invalid-rotary.gguf");
            for rope_dim in [None, Some(0), Some(1), Some(3), Some(6)] {
                match rope_dim {
                    Some(value) => {
                        metadata.insert(rotary_key.clone(), gguf_file::Value::U32(value));
                    }
                    None => {
                        metadata.remove(&rotary_key);
                    }
                }
                let error = parse_gguf_metadata(&metadata).unwrap_err().to_string();
                assert!(error.contains(&rotary_key), "{error}");
                let values = metadata
                    .iter()
                    .map(|(key, value)| (key.as_str(), value))
                    .collect::<Vec<_>>();
                gguf_file::write(&mut std::fs::File::create(&path)?, &values, &[])?;
                let error = estimate_stage_memory_profile(&path, "invalid-rotary", 2, &Device::Cpu)
                    .unwrap_err()
                    .to_string();
                assert!(error.contains(&rotary_key), "{error}");
            }
        }
        Ok(())
    }

    #[test]
    fn phi3_metadata_and_profiles_validate_partial_rotary_dimensions() -> Result<()> {
        let mut metadata = HashMap::from([(
            "general.architecture".to_string(),
            gguf_file::Value::String("phi3".to_string()),
        )]);
        for (key, value) in [
            ("block_count", 2),
            ("embedding_length", 8),
            ("attention.head_count", 2),
            ("attention.key_length", 4),
            ("context_length", 4),
        ] {
            metadata.insert(format!("phi3.{key}"), gguf_file::Value::U32(value));
        }
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("phi3-rotary.gguf");
        for rope_dim in [0, 1, 2, 3, 4, 6] {
            metadata.insert(
                "phi3.rope.dimension_count".to_string(),
                gguf_file::Value::U32(rope_dim),
            );
            let values = metadata
                .iter()
                .map(|(key, value)| (key.as_str(), value))
                .collect::<Vec<_>>();
            gguf_file::write(&mut std::fs::File::create(&path)?, &values, &[])?;
            if matches!(rope_dim, 0 | 2 | 4) {
                let parsed = parse_gguf_metadata(&metadata)?;
                assert_eq!(parsed.rope_dimension_count, rope_dim as usize);
                let profile = estimate_stage_memory_profile(&path, "phi3-rotary", 2, &Device::Cpu)?;
                assert_eq!(
                    profile.common_cpu_bytes,
                    if rope_dim == 2 { 32 } else { 64 }
                );
            } else {
                let error = parse_gguf_metadata(&metadata).unwrap_err().to_string();
                assert!(error.contains("phi3.rope.dimension_count"), "{error}");
                let error = estimate_stage_memory_profile(&path, "phi3-rotary", 2, &Device::Cpu)
                    .unwrap_err()
                    .to_string();
                assert!(error.contains("phi3.rope.dimension_count"), "{error}");
            }
        }
        Ok(())
    }

    #[test]
    fn test_expert_gating_function_metadata() {
        let md = HashMap::from([
            (
                "general.architecture".to_string(),
                gguf_file::Value::String("minimax-m2".to_string()),
            ),
            (
                "minimax-m2.block_count".to_string(),
                gguf_file::Value::U32(62),
            ),
            (
                "minimax-m2.embedding_length".to_string(),
                gguf_file::Value::U32(3072),
            ),
            (
                "minimax-m2.attention.head_count".to_string(),
                gguf_file::Value::U32(48),
            ),
            (
                "minimax-m2.expert_gating_func".to_string(),
                gguf_file::Value::U32(2),
            ),
        ]);

        assert_eq!(parse_gguf_metadata(&md).unwrap().expert_gating_func, 2);
    }

    fn minimal_llama_metadata() -> HashMap<String, gguf_file::Value> {
        HashMap::from([
            (
                "general.architecture".into(),
                gguf_file::Value::String("llama".into()),
            ),
            ("llama.block_count".into(), gguf_file::Value::U32(1)),
            ("llama.embedding_length".into(), gguf_file::Value::U32(4096)),
            (
                "llama.attention.head_count".into(),
                gguf_file::Value::U32(32),
            ),
        ])
    }

    fn write_cpu_embedding_profile_fixture(path: &Path, include_output_weight: bool) -> Result<()> {
        write_dense_profile_fixture(path, include_output_weight, GgmlDType::Q4K)
    }

    fn write_dense_profile_fixture(
        path: &Path,
        include_output_weight: bool,
        projection_dtype: GgmlDType,
    ) -> Result<()> {
        write_dense_profile_fixture_with_context(path, include_output_weight, projection_dtype, 32)
    }

    fn write_dense_profile_fixture_with_context(
        path: &Path,
        include_output_weight: bool,
        projection_dtype: GgmlDType,
        context_length: u32,
    ) -> Result<()> {
        write_dense_profile_fixture_with_geometry(
            path,
            include_output_weight,
            projection_dtype,
            context_length,
            64,
            1,
        )
    }

    fn write_dense_profile_fixture_with_geometry(
        path: &Path,
        include_output_weight: bool,
        projection_dtype: GgmlDType,
        context_length: u32,
        vocab_size: u32,
        total_layers: u32,
    ) -> Result<()> {
        use candle_core::quantized::QTensor;

        let metadata = [
            (
                "general.architecture",
                gguf_file::Value::String("llama".into()),
            ),
            ("llama.block_count", gguf_file::Value::U32(total_layers)),
            ("llama.embedding_length", gguf_file::Value::U32(256)),
            ("llama.attention.head_count", gguf_file::Value::U32(4)),
            ("llama.attention.head_count_kv", gguf_file::Value::U32(4)),
            ("llama.attention.key_length", gguf_file::Value::U32(64)),
            ("llama.feed_forward_length", gguf_file::Value::U32(256)),
            (
                "llama.context_length",
                gguf_file::Value::U32(context_length),
            ),
            ("llama.vocab_size", gguf_file::Value::U32(vocab_size)),
        ];
        let metadata = metadata
            .iter()
            .map(|(key, value)| (*key, value))
            .collect::<Vec<_>>();
        let mut tensors = Vec::new();
        for (name, shape, dtype) in [
            (
                "token_embd.weight",
                vec![vocab_size as usize, 256],
                projection_dtype,
            ),
            ("output_norm.weight", vec![256], GgmlDType::F32),
            ("blk.0.attn_norm.weight", vec![256], GgmlDType::F32),
            ("blk.0.ffn_norm.weight", vec![256], GgmlDType::F32),
            ("blk.0.attn_q.weight", vec![256, 256], projection_dtype),
            ("blk.0.attn_k.weight", vec![256, 256], projection_dtype),
            ("blk.0.attn_v.weight", vec![256, 256], projection_dtype),
            ("blk.0.attn_output.weight", vec![256, 256], projection_dtype),
            ("blk.0.ffn_gate.weight", vec![256, 256], projection_dtype),
            ("blk.0.ffn_up.weight", vec![256, 256], projection_dtype),
            ("blk.0.ffn_down.weight", vec![256, 256], projection_dtype),
        ] {
            let tensor =
                candle_core::Tensor::ones(shape.as_slice(), candle_core::DType::F32, &Device::Cpu)?;
            tensors.push((name.to_string(), QTensor::quantize(&tensor, dtype)?));
        }
        for layer in 1..total_layers {
            let copies = tensors
                .iter()
                .filter(|(name, _)| name.starts_with("blk.0."))
                .map(|(name, source)| {
                    let dense = candle_core::Tensor::ones(
                        source.shape(),
                        candle_core::DType::F32,
                        &Device::Cpu,
                    )?;
                    Ok((
                        name.replacen("blk.0.", &format!("blk.{layer}."), 1),
                        QTensor::quantize(&dense, source.dtype())?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            tensors.extend(copies);
        }
        if include_output_weight {
            let tensor = candle_core::Tensor::ones(
                (vocab_size as usize, 256),
                candle_core::DType::F32,
                &Device::Cpu,
            )?;
            tensors.push((
                "output.weight".to_string(),
                QTensor::quantize(&tensor, projection_dtype)?,
            ));
        }
        let tensor_refs = tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        gguf_file::write(&mut std::fs::File::create(path)?, &metadata, &tensor_refs)?;
        Ok(())
    }

    #[test]
    fn cpu_dense_embedding_profile_uses_realized_f32_size_for_tied_and_untied_output() -> Result<()>
    {
        let dir = tempfile::tempdir()?;
        for (label, include_output_weight) in [("tied", false), ("untied", true)] {
            let path = dir.path().join(format!("{label}.gguf"));
            write_cpu_embedding_profile_fixture(&path, include_output_weight)?;

            let profile = estimate_stage_memory_profile(&path, "cpu-profile", 1, &Device::Cpu)?;
            let encoded_bytes = profile.embedding_gpu_bytes;
            let realized_bytes = profile.head_embedding_gpu_bytes.unwrap();
            assert!(encoded_bytes < realized_bytes);
            assert_eq!(realized_bytes, 64 * 256 * std::mem::size_of::<f32>() as u64);
            assert_eq!(
                profile.head_embedding_cpu_transient_bytes,
                Some(encoded_bytes)
            );
            assert_eq!(
                profile.tail_embedding_gpu_bytes,
                Some(if include_output_weight {
                    0
                } else {
                    encoded_bytes
                })
            );

            let components = profile
                .stage_memory_components(0, 1, 1)
                .expect("one-layer profile should have exact stage components");
            assert!(components.0 >= realized_bytes);
            assert_eq!(components.2, encoded_bytes);

            let stage = load_stage_tensors(&path, LayerRange::solo(1), &Device::Cpu)?;
            let embedding = stage
                .tensors
                .get("token_embd.weight")
                .expect("fixture embedding")
                .as_ref()
                .dequantize(&Device::Cpu)?;
            assert_eq!(
                embedding.elem_count() * embedding.dtype().size_in_bytes(),
                realized_bytes as usize
            );
            crate::architectures::llama::LlamaStage::from_stage_tensors(stage, None)?;
        }
        Ok(())
    }

    #[test]
    fn stage_memory_profile_accounts_for_generated_rope_peak() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let short_path = dir.path().join("short-context.gguf");
        let long_path = dir.path().join("long-context.gguf");
        write_dense_profile_fixture_with_context(&short_path, false, GgmlDType::F32, 32)?;
        write_dense_profile_fixture_with_context(&long_path, false, GgmlDType::F32, 65_536)?;

        let short = estimate_stage_memory_profile(&short_path, "rope-short", 1, &Device::Cpu)?;
        let long = estimate_stage_memory_profile(&long_path, "rope-long", 1, &Device::Cpu)?;
        let short_metadata = read_gguf_metadata(&short_path)?;
        let long_metadata = read_gguf_metadata(&long_path)?;
        let expected_delta = (65_536u64 - 32) * (64 / 2) * 4 * 2;

        assert!(expected_delta > 0);
        assert_eq!(
            generated_rope_memory_bytes(&long_metadata, &Device::Cpu).0
                - generated_rope_memory_bytes(&short_metadata, &Device::Cpu).0,
            expected_delta
        );
        assert_eq!(
            long.common_cpu_bytes - short.common_cpu_bytes,
            expected_delta
        );
        let expected_short_construction = (32u64 * (64 / 2) * 4) + (2 * (32 * 4)) + ((64 / 2) * 4);
        let expected_long_construction =
            (65_536u64 * (64 / 2) * 4) + (2 * (65_536 * 4)) + ((64 / 2) * 4);
        assert_eq!(
            short.common_cpu_transient_bytes,
            expected_short_construction
        );
        assert_eq!(long.common_cpu_transient_bytes, expected_long_construction);
        let short_components = short
            .stage_memory_components(0, 1, 1)
            .expect("short profile should have exact stage components");
        let long_components = long
            .stage_memory_components(0, 1, 1)
            .expect("long profile should have exact stage components");
        assert!(long_components.1 > short_components.1);
        Ok(())
    }

    #[test]
    fn generated_rope_memory_inventory_matches_architecture_constructors() {
        let mut metadata = make_metadata(4, 2, 64);
        metadata.context_length = 1024;
        metadata.qk_rope_head_dim = 32;
        let retained_table_architectures = [
            "llama",
            "mistral",
            "qwen2",
            "qwen3moe",
            "phi3",
            "minimax-m2",
            "qwen35",
            "deepseek2",
            "glm-dsa",
            "llama4",
            "deepseek4",
        ];
        for architecture in retained_table_architectures {
            metadata.architecture = architecture.into();
            metadata.compress_ratios.clear();
            assert!(
                generated_rope_memory_bytes(&metadata, &Device::Cpu).0 > 0,
                "expected retained RoPE tables for {architecture}"
            );
        }

        for architecture in ["gemma4", "nemotron_h", "nemotron_h_moe", "glm5next"] {
            metadata.architecture = architecture.into();
            assert_eq!(generated_rope_memory_bytes(&metadata, &Device::Cpu), (0, 0));
        }

        metadata.architecture = "deepseek4".into();
        metadata.compress_ratios = vec![0; metadata.num_layers];
        let uncompressed = generated_rope_memory_bytes(&metadata, &Device::Cpu).0;
        metadata.compress_ratios[0] = 4;
        let compressed = generated_rope_memory_bytes(&metadata, &Device::Cpu).0;
        assert!(compressed > uncompressed);
    }

    #[test]
    fn metal_rope_memory_accounts_for_storage_buckets_and_cast_overlap() -> Result<()> {
        let Ok(device) = Device::new_metal(0) else {
            // CPU-only builders still compile and run the metadata suite.
            return Ok(());
        };
        let mut metadata = make_metadata(1, 2, 64);
        metadata.architecture = "llama".into();
        metadata.context_length = 1_000;
        let (persistent, construction) = generated_rope_memory_bytes(&metadata, &device);

        let round_bucket = |elements: u64| (elements * 4).next_power_of_two();
        let table = round_bucket(1_000 * 32);
        let positions = round_bucket(1_000);
        let inverse_frequency = round_bucket(32);
        assert_eq!(persistent, table * 2);
        assert_eq!(
            construction,
            table + positions * 2 + inverse_frequency,
            "construction peak must include U32 source plus F32 position cast"
        );
        assert!(persistent > 1_000 * 32 * 4 * 2);
        Ok(())
    }

    fn check_dense_projection_profile(device: &Device) -> Result<()> {
        use crate::qmatmul::QMatMul;

        fn matmul_bytes(tensor: &Arc<candle_core::quantized::QTensor>) -> Result<u64> {
            Ok(match QMatMul::from_arc(Arc::clone(tensor))? {
                QMatMul::Tensor(t) | QMatMul::TensorF16(t) => {
                    (t.elem_count() * t.dtype().size_in_bytes()) as u64
                }
                QMatMul::QTensor(t) => t.storage_size_in_bytes() as u64,
                #[cfg(target_os = "macos")]
                QMatMul::MetalQ2(t) => t.packed_size_bytes() as u64,
            })
        }

        let dir = tempfile::tempdir()?;
        for dtype in [
            GgmlDType::Q4K,
            GgmlDType::F16,
            GgmlDType::BF16,
            GgmlDType::F32,
        ] {
            for untied in [false, true] {
                let path = dir.path().join(format!("{dtype:?}-{untied}.gguf"));
                write_dense_profile_fixture(&path, untied, dtype)?;
                let profile = estimate_stage_memory_profile(&path, "dense-profile", 1, device)?;
                let stage = load_stage_tensors(&path, LayerRange::solo(1), device)?;
                let projection = &stage.tensors["blk.0.attn_q.weight"];
                let floating = matches!(dtype, GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::F32);
                let layer_source = 2 * 256 * 4
                    + if floating {
                        7 * projection.storage_size_in_bytes() as u64
                    } else {
                        0
                    };
                let actual_layer = 7 * matmul_bytes(projection)? + 2 * 256 * 4;
                assert_eq!(
                    profile.layer_gpu_bytes[0],
                    actual_layer + if device.is_cpu() { 0 } else { layer_source }
                );
                assert_eq!(
                    profile.layer_cpu_bytes[0],
                    if device.is_cpu() { layer_source } else { 0 }
                );
                let output_name = if untied {
                    "output.weight"
                } else {
                    "token_embd.weight"
                };
                let output = &stage.tensors[output_name];
                let actual_output = matmul_bytes(output)?;
                if floating {
                    // The eager stage still owns all encoded tensors while
                    // its independent dense destinations coexist. Derive
                    // this lower bound from loaded storage and materialized
                    // QMatMuls, rather than estimator rules.
                    let source_pool = stage
                        .tensors
                        .values()
                        .map(|tensor| tensor.storage_size_in_bytes() as u64)
                        .sum::<u64>();
                    let dense_destinations = actual_layer + 64 * 256 * 4 + 256 * 4 + actual_output;
                    let (gpu, cpu, scratch) = profile.stage_memory_components(0, 1, 1).unwrap();
                    assert!(
                        gpu.saturating_add(cpu).saturating_add(scratch)
                            >= source_pool + dense_destinations,
                        "{device:?} {dtype:?} source pool and dense destinations overlap"
                    );
                }
                if untied {
                    let source = if !device.is_cpu() {
                        256 * 4
                            + if floating {
                                output.storage_size_in_bytes() as u64
                            } else {
                                0
                            }
                    } else {
                        0
                    };
                    assert_eq!(profile.output_gpu_bytes, 256 * 4 + actual_output + source);
                    assert_eq!(
                        profile.output_cpu_bytes,
                        if device.is_cpu() {
                            256 * 4
                                + if floating {
                                    output.storage_size_in_bytes() as u64
                                } else {
                                    0
                                }
                        } else {
                            0
                        }
                    );
                    assert_eq!(profile.tail_embedding_gpu_bytes, Some(0));
                } else {
                    if matches!(dtype, GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::F32) {
                        assert_eq!(profile.tail_embedding_gpu_bytes, Some(0));
                        let source = if device.is_cpu() {
                            0
                        } else {
                            256 * 4 + output.storage_size_in_bytes() as u64
                        };
                        assert_eq!(profile.output_gpu_bytes, 256 * 4 + actual_output + source);
                        let components = profile.stage_memory_components(0, 1, 1).unwrap();
                        assert!(
                            components.0 >= actual_layer + 256 * 4 + 64 * 256 * 4 + actual_output
                        );
                    } else {
                        assert_eq!(profile.tail_embedding_gpu_bytes, Some(actual_output));
                    }
                }
                assert_eq!(
                    profile.head_embedding_gpu_bytes,
                    Some(
                        64 * 256 * 4
                            + if device.is_cpu() {
                                0
                            } else {
                                profile.embedding_gpu_bytes
                                    + if device.is_metal() {
                                        device_allocation_bytes(profile.embedding_gpu_bytes, device)
                                    } else {
                                        0
                                    }
                            }
                    )
                );
                crate::architectures::llama::LlamaStage::from_stage_tensors(stage, None)?;
                device.synchronize()?;
            }
        }
        Ok(())
    }

    #[test]
    fn dense_projection_profile_matches_materialized_cpu_weights() -> Result<()> {
        check_dense_projection_profile(&Device::Cpu)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn dense_projection_profile_matches_materialized_metal_weights() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        check_dense_projection_profile(&device)
    }

    fn check_dense_embedding_source_peak(device: &Device) -> Result<()> {
        use crate::architectures::{ModelArchitecture, StageInput, StageOutput, llama::LlamaStage};

        let dir = tempfile::tempdir()?;
        for dtype in [GgmlDType::F32, GgmlDType::F16, GgmlDType::BF16] {
            let path = dir.path().join(format!("large-embedding-{dtype:?}.gguf"));
            write_dense_profile_fixture_with_geometry(&path, true, dtype, 32, 512, 2)?;
            let profile = estimate_stage_memory_profile(&path, "embedding-peak", 2, device)?;
            let mut head_output = None;
            for range in [
                LayerRange::solo(2),
                LayerRange::new(0, 1, 2),
                LayerRange::new(1, 2, 2),
            ] {
                let stage = load_stage_tensors(&path, range.clone(), device)?;
                assert_eq!(
                    stage.tensors.contains_key("token_embd.weight"),
                    range.is_head()
                );
                let source_pool = stage
                    .tensors
                    .values()
                    .map(|t| t.storage_size_in_bytes() as u64)
                    .sum::<u64>();
                let dense_weights = stage
                    .tensors
                    .values()
                    .map(|t| dequantized_f32_bytes(t.shape().elem_count()))
                    .sum::<u64>();
                let (gpu, cpu, scratch) = profile
                    .stage_memory_components(range.start, range.end, 2)
                    .unwrap();
                if device.is_cpu() {
                    assert!(gpu + cpu + scratch >= source_pool + dense_weights);
                } else {
                    let conversion_readback = if device.is_metal() {
                        stage
                            .tensors
                            .values()
                            .map(|t| t.storage_size_in_bytes() as u64)
                            .max()
                            .unwrap()
                    } else {
                        0
                    };
                    assert!(
                        gpu >= source_pool + dense_weights + conversion_readback,
                        "eager embedding source must coexist with output conversion scratch"
                    );
                }
                if range.is_head() && !device.is_cpu() {
                    let embedding = &stage.tensors["token_embd.weight"];
                    let encoded = embedding.storage_size_in_bytes() as u64;
                    let decoded = dequantized_f32_bytes(embedding.shape().elem_count());
                    assert!(profile.head_embedding_gpu_bytes.unwrap() >= encoded + decoded);
                    if device.is_metal() {
                        assert!(
                            profile.head_embedding_cpu_transient_bytes.unwrap()
                                >= encoded + decoded
                        );
                    }
                }
                let model = LlamaStage::from_stage_tensors(stage, None)?;
                let input = if range.is_head() {
                    StageInput::Tokens(vec![0, 1])
                } else {
                    StageInput::activations(head_output.take().expect("head activations"))
                };
                let mut cache = std::iter::repeat_with(|| None).take(2).collect::<Vec<_>>();
                match model.forward(input, &mut cache, 2, 0)? {
                    StageOutput::Logits(logits) => {
                        assert_eq!(logits.dims(), [512]);
                        assert!(
                            logits
                                .to_dtype(candle_core::DType::F32)?
                                .to_vec1::<f32>()?
                                .iter()
                                .all(|v| v.is_finite())
                        );
                    }
                    StageOutput::Activations(activations) => head_output = Some(activations),
                }
                device.synchronize()?;
            }
            let mut streaming = StageTensorLoader::open(&path)?;
            let tail = streaming.load_range(LayerRange::new(1, 2, 2), device)?;
            assert!(!tail.tensors.contains_key("token_embd.weight"));
        }
        Ok(())
    }

    #[test]
    fn dense_embedding_profile_covers_full_and_split_cpu_source_peaks() -> Result<()> {
        check_dense_embedding_source_peak(&Device::Cpu)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn dense_embedding_profile_covers_full_and_split_metal_source_peaks() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        check_dense_embedding_source_peak(&device)
    }

    fn write_deepseek4_dense_profile_fixture(path: &Path, dtype: GgmlDType) -> Result<()> {
        use candle_core::quantized::QTensor;

        let mut metadata = vec![(
            "general.architecture".to_owned(),
            gguf_file::Value::String("deepseek4".into()),
        )];
        for (key, value) in [
            ("block_count", 2),
            ("embedding_length", 512),
            ("attention.head_count", 1),
            ("attention.key_length", 512),
            ("attention.value_length", 512),
            ("context_length", 8),
            ("vocab_size", 4),
            ("feed_forward_length", 8),
            ("expert_count", 2),
            ("expert_used_count", 1),
            ("expert_feed_forward_length", 2),
            ("expert_shared_count", 1),
            ("rope.dimension_count", 64),
            ("attention.q_lora_rank", 1),
            ("attention.output_lora_rank", 1),
            ("attention.output_group_count", 1),
            ("attention.sliding_window", 128),
            ("hyper_connection.count", 4),
            ("hyper_connection.sinkhorn_iterations", 2),
        ] {
            metadata.push((format!("deepseek4.{key}"), gguf_file::Value::U32(value)));
        }
        metadata.extend([
            (
                "deepseek4.attention.compress_ratios".into(),
                gguf_file::Value::Array(vec![gguf_file::Value::U32(0); 2]),
            ),
            (
                "deepseek4.expert_weights_scale".into(),
                gguf_file::Value::F32(1.),
            ),
            (
                "deepseek4.expert_weights_norm".into(),
                gguf_file::Value::Bool(true),
            ),
            (
                "deepseek4.hyper_connection.epsilon".into(),
                gguf_file::Value::F32(1e-6),
            ),
        ]);
        let mut specs = vec![
            ("token_embd.weight".to_owned(), vec![4, 512]),
            ("output.weight".to_owned(), vec![4, 512]),
            ("output_norm.weight".to_owned(), vec![512]),
            ("output_hc_fn.weight".to_owned(), vec![4, 2048]),
            ("output_hc_base.weight".to_owned(), vec![4]),
            ("output_hc_scale.weight".to_owned(), vec![1]),
        ];
        for layer in 0..2 {
            for (name, shape) in [
                ("attn_norm", vec![512]),
                ("attn_q_a", vec![1, 512]),
                ("attn_q_a_norm", vec![1]),
                ("attn_q_b", vec![512, 1]),
                ("attn_kv", vec![512, 512]),
                ("attn_kv_a_norm", vec![512]),
                ("attn_output_a", vec![1, 512]),
                ("attn_output_b", vec![512, 1]),
                ("attn_sinks", vec![1]),
                ("hc_attn_fn", vec![24, 2048]),
                ("hc_attn_base", vec![24]),
                ("hc_attn_scale", vec![3]),
                ("ffn_norm", vec![512]),
                ("hc_ffn_fn", vec![24, 2048]),
                ("hc_ffn_base", vec![24]),
                ("hc_ffn_scale", vec![3]),
                ("ffn_gate_inp", vec![2, 512]),
                ("ffn_gate_shexp", vec![2, 512]),
                ("ffn_up_shexp", vec![2, 512]),
                ("ffn_down_shexp", vec![512, 2]),
                ("ffn_gate_exps", vec![2, 2, 512]),
                ("ffn_up_exps", vec![2, 2, 512]),
                ("ffn_down_exps", vec![2, 512, 2]),
            ] {
                specs.push((format!("blk.{layer}.{name}.weight"), shape));
            }
        }
        let mut tensors = Vec::new();
        for (name, shape) in specs {
            let value = if name.contains("norm") || name.contains("scale") {
                1f32
            } else if name.contains("base") || name.contains("sinks") {
                0.
            } else {
                0.01
            };
            let dense = candle_core::Tensor::full(value, shape, &Device::Cpu)?;
            tensors.push((name, QTensor::quantize(&dense, dtype)?));
        }
        let metadata_refs = metadata
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect::<Vec<_>>();
        let tensor_refs = tensors
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect::<Vec<_>>();
        gguf_file::write(
            &mut std::fs::File::create(path)?,
            &metadata_refs,
            &tensor_refs,
        )?;
        Ok(())
    }

    fn check_deepseek4_dense_profile(device: &Device) -> Result<()> {
        use crate::architectures::{
            ModelArchitecture, StageInput, StageOutput, deepseek4::DeepSeek4Stage,
        };

        let dir = tempfile::tempdir()?;
        for dtype in [GgmlDType::F32, GgmlDType::F16, GgmlDType::BF16] {
            let path = dir.path().join(format!("v4-{dtype:?}.gguf"));
            write_deepseek4_dense_profile_fixture(&path, dtype)?;
            let profile = estimate_stage_memory_profile(&path, "v4-profile", 2, device)?;
            let mut split_input = None;
            for range in [
                LayerRange::solo(2),
                LayerRange::new(0, 1, 2),
                LayerRange::new(1, 2, 2),
            ] {
                let stage = load_stage_tensors(&path, range.clone(), device)?;
                assert_eq!(
                    stage.raw_tensors.contains_key("token_embd.weight"),
                    range.is_head()
                );
                assert_eq!(
                    stage.raw_tensors.contains_key("output_hc_fn.weight"),
                    range.is_tail()
                );
                let raw_sources = stage
                    .raw_tensors
                    .values()
                    .map(|t| t.byte_len() as u64)
                    .sum::<u64>();
                let mut dense_destinations = 0u64;
                let mut tensor_footprints = HashMap::new();
                for (name, tensor) in &stage.raw_tensors {
                    let dense = tensor
                        .to_dense(&Device::Cpu)?
                        .to_dtype(candle_core::DType::F32)?;
                    let decoded = (dense.elem_count() * dense.dtype().size_in_bytes()) as u64;
                    dense_destinations += decoded;
                    tensor_footprints.insert(name.as_str(), tensor.byte_len() as u64 + decoded);
                }
                // Check the resident components on their own, so a large
                // conversion scratch estimate cannot hide an omitted norm
                // or an omitted hyper-connection output weight.
                for layer in range.start..range.end {
                    let prefix = format!("blk.{layer}.");
                    let weights = tensor_footprints
                        .iter()
                        .filter(|(name, _)| name.starts_with(&prefix))
                        .map(|(_, bytes)| *bytes)
                        .sum::<u64>();
                    let expected = weights + (512 + 2 * 4 * 512) * 4;
                    assert!(profile.layer_gpu_bytes[layer] >= expected);
                    if device.is_cpu() {
                        assert_eq!(profile.layer_gpu_bytes[layer], expected);
                    }
                }
                if range.is_tail() {
                    let expected = tensor_footprints
                        .iter()
                        .filter(|(name, _)| name.starts_with("output"))
                        .map(|(_, bytes)| *bytes)
                        .sum::<u64>()
                        + 4 * 512 * 4;
                    assert!(profile.output_gpu_bytes >= expected);
                    if device.is_cpu() {
                        assert_eq!(profile.output_gpu_bytes, expected);
                    }
                }
                let generated_norms = (range.layer_count() as u64 * (512 + 2 * 4 * 512)
                    + u64::from(range.is_tail()) * 4 * 512)
                    * 4;
                let (gpu, cpu, scratch) = profile
                    .stage_memory_components(range.start, range.end, 2)
                    .unwrap();
                assert!(
                    gpu.saturating_add(cpu).saturating_add(scratch)
                        >= raw_sources + dense_destinations + generated_norms,
                    "{device:?} {dtype:?} {range:?}: raw sources, decoded weights, and generated norms overlap"
                );
                let model = DeepSeek4Stage::from_stage_tensors(stage, None)?;
                let input = if range.is_head() {
                    StageInput::Tokens(vec![0, 1])
                } else {
                    StageInput::activations(split_input.take().expect("head activations"))
                };
                let mut cache = std::iter::repeat_with(|| None).take(2).collect::<Vec<_>>();
                match model.forward(input, &mut cache, 2, 0)? {
                    StageOutput::Logits(logits) => {
                        assert_eq!(logits.dims(), [4]);
                        assert!(
                            logits
                                .to_vec1::<f32>()?
                                .iter()
                                .all(|value| value.is_finite())
                        );
                    }
                    StageOutput::Activations(activations) => split_input = Some(activations),
                }
                device.synchronize()?;
            }
        }
        Ok(())
    }

    #[test]
    fn deepseek4_dense_profile_covers_cpu_sources_destinations_and_boundaries() -> Result<()> {
        check_deepseek4_dense_profile(&Device::Cpu)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn deepseek4_dense_profile_covers_metal_sources_destinations_and_boundaries() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        check_deepseek4_dense_profile(&device)
    }

    #[test]
    fn floating_projection_accounting_reserves_conversion_sources_and_scratch() {
        for dtype in [GgmlDType::F16, GgmlDType::BF16, GgmlDType::F32] {
            let source_bytes = if dtype == GgmlDType::F32 { 1024 } else { 512 };
            for (accelerator, metal) in [(false, false), (true, false), (true, true)] {
                let contribution = tensor_memory_accounting(
                    "llama",
                    "blk.0.attn_q.weight",
                    source_bytes,
                    256,
                    dtype,
                    accelerator,
                    metal,
                    false,
                );
                assert_eq!(contribution.accelerator_bytes, 1024);
                assert_eq!(
                    contribution.cpu_source_bytes,
                    if accelerator { 0 } else { source_bytes }
                );
                assert_eq!(
                    contribution.accelerator_source_bytes,
                    if accelerator { source_bytes } else { 0 }
                );
                assert_eq!(
                    contribution.accelerator_transient_bytes,
                    if metal { source_bytes } else { 0 }
                );
                assert_eq!(
                    contribution.cpu_upload_bytes,
                    if metal { source_bytes + 1024 } else { 0 }
                );
            }
            // Packed lookup embeddings are handled separately at boundaries.
            let embedding = tensor_memory_accounting(
                "qwen35",
                "token_embd.weight",
                source_bytes,
                256,
                dtype,
                true,
                true,
                true,
            );
            assert_eq!(embedding.accelerator_bytes, source_bytes);
        }
    }

    #[test]
    fn floating_expert_profiles_cover_split_f32_fallback() -> Result<()> {
        use crate::qmatmul::QMatMul;
        use candle_core::quantized::QTensor;
        for dtype in [GgmlDType::F16, GgmlDType::BF16, GgmlDType::F32] {
            let source = QTensor::quantize(
                &candle_core::Tensor::ones((4, 8), candle_core::DType::F32, &Device::Cpu)?,
                dtype,
            )?;
            let bytes = source.storage_size_in_bytes() as u64;
            let QMatMul::Tensor(realized) = QMatMul::from_arc(Arc::new(source))? else {
                panic!("floating expert must materialize F32");
            };
            let final_bytes = (realized.elem_count() * realized.dtype().size_in_bytes()) as u64;
            for (accelerator, metal) in [(false, false), (true, false), (true, true)] {
                let contribution = tensor_memory_accounting(
                    "qwen3moe",
                    "blk.0.ffn_up_exps.weight",
                    bytes,
                    32,
                    dtype,
                    accelerator,
                    metal,
                    false,
                );
                if dtype == GgmlDType::BF16 && (!accelerator || metal) {
                    assert_eq!(contribution.cpu_source_bytes, 0);
                    assert_eq!(contribution.retained_cpu_bytes, bytes);
                    assert_eq!(contribution.accelerator_bytes, 0);
                    assert_eq!(
                        contribution.accelerator_transient_bytes,
                        if metal {
                            final_bytes * 4
                                + crate::architectures::qwen3moe::BF16_GROUP_WORKSPACE_BYTES as u64
                        } else {
                            0
                        }
                    );
                    assert_eq!(
                        contribution.cpu_upload_bytes,
                        if metal {
                            bytes + final_bytes
                        } else {
                            final_bytes * 4
                        }
                    );
                } else {
                    assert_eq!(contribution.cpu_source_bytes, bytes);
                    assert_eq!(
                        contribution.accelerator_bytes + contribution.retained_cpu_bytes,
                        final_bytes
                    );
                }
            }
        }
        Ok(())
    }

    fn split_tensor_fixture(
        architecture: &str,
        names: [&str; 2],
    ) -> Result<(tempfile::TempDir, PathBuf)> {
        use candle_core::quantized::QTensor;
        let dir = tempfile::tempdir()?;
        let metadata = [
            (
                "general.architecture".to_string(),
                gguf_file::Value::String(architecture.to_string()),
            ),
            (
                format!("{architecture}.block_count"),
                gguf_file::Value::U32(2),
            ),
            (
                format!("{architecture}.embedding_length"),
                gguf_file::Value::U32(4),
            ),
            (
                format!("{architecture}.attention.head_count"),
                gguf_file::Value::U32(1),
            ),
            (
                format!("{architecture}.vocab_size"),
                gguf_file::Value::U32(2),
            ),
        ];
        let metadata = metadata
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect::<Vec<_>>();
        for (index, name) in names.into_iter().enumerate() {
            let tensor =
                candle_core::Tensor::from_vec(vec![index as f32 + 1.; 8], (2, 4), &Device::Cpu)?;
            let quant = QTensor::quantize(&tensor, GgmlDType::F32)?;
            let path = dir
                .path()
                .join(format!("model-{:05}-of-00002.gguf", index + 1));
            gguf_file::write(
                &mut std::fs::File::create(path)?,
                &metadata,
                &[(name, &quant)],
            )?;
        }
        let first = dir.path().join("model-00001-of-00002.gguf");
        Ok((dir, first))
    }

    #[test]
    fn split_gguf_rejects_duplicate_tensors_before_stage_loading_and_admission() -> Result<()> {
        // Reject duplicates even if they are outside this stage's layer range.
        for name in ["token_embd.weight", "blk.1.attn_norm.weight"] {
            let (_dir, path) = split_tensor_fixture("llama", [name, name])?;
            let range = LayerRange::new(0, 1, 2);
            let errors = [
                load_stage_tensors(&path, range, &Device::Cpu).err(),
                StageTensorLoader::open(&path).err(),
                estimate_stage_memory_profile(&path, "duplicate", 2, &Device::Cpu).err(),
            ];
            for error in errors {
                let error = error
                    .expect("duplicate shard tensors must fail")
                    .to_string();
                assert!(error.contains(name), "{error}");
                assert!(error.contains("shards 1 and 2"), "{error}");
            }
        }
        Ok(())
    }

    #[test]
    fn single_file_duplicates_fail_before_loading_or_memory_admission() -> Result<()> {
        use candle_core::quantized::QTensor;
        let dir = tempfile::tempdir()?;
        let one = QTensor::quantize(
            &candle_core::Tensor::ones((2, 4), candle_core::DType::F32, &Device::Cpu)?,
            GgmlDType::F32,
        )?;
        for architecture in ["llama", "deepseek4"] {
            for duplicate_metadata in [false, true] {
                let path = dir
                    .path()
                    .join(format!("{architecture}-{duplicate_metadata}.gguf"));
                let mut metadata = vec![
                    (
                        "general.architecture".to_owned(),
                        gguf_file::Value::String(architecture.into()),
                    ),
                    (
                        format!("{architecture}.block_count"),
                        gguf_file::Value::U32(2),
                    ),
                    (
                        format!("{architecture}.embedding_length"),
                        gguf_file::Value::U32(4),
                    ),
                    (
                        format!("{architecture}.attention.head_count"),
                        gguf_file::Value::U32(1),
                    ),
                ];
                let mut tensors = vec![("blk.1.attn_norm.weight", &one)];
                let duplicate_name = if duplicate_metadata {
                    let key = format!("{architecture}.block_count");
                    metadata.push((key.clone(), gguf_file::Value::U32(3)));
                    key
                } else {
                    tensors.push(("blk.1.attn_norm.weight", &one));
                    "blk.1.attn_norm.weight".to_owned()
                };
                let refs = metadata
                    .iter()
                    .map(|(key, value)| (key.as_str(), value))
                    .collect::<Vec<_>>();
                gguf_file::write(&mut std::fs::File::create(&path)?, &refs, &tensors)?;
                for error in [
                    read_gguf_metadata(&path).err(),
                    load_stage_tensors(&path, LayerRange::new(0, 1, 2), &Device::Cpu).err(),
                    estimate_stage_memory_profile(&path, "duplicate", 2, &Device::Cpu).err(),
                    StageTensorLoader::open(&path).err(),
                ] {
                    let error = format!("{:#}", error.expect("single-file duplicate must fail"));
                    assert!(error.contains("duplicate"), "{error}");
                    assert!(error.contains(&duplicate_name), "{error}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn split_deepseek4_rejects_duplicate_tensors_before_loading_and_admission() -> Result<()> {
        let name = "blk.1.attn_norm.weight";
        let (_dir, path) = split_tensor_fixture("deepseek4", [name, name])?;
        let errors = [
            load_stage_tensors(&path, LayerRange::new(0, 1, 2), &Device::Cpu).err(),
            estimate_stage_memory_profile(&path, "duplicate", 2, &Device::Cpu).err(),
        ];
        for error in errors {
            let error = error
                .expect("duplicate raw shard tensors must fail")
                .to_string();
            assert!(error.contains(name), "{error}");
            assert!(error.contains("shards 1 and 2"), "{error}");
        }
        Ok(())
    }

    #[test]
    fn split_gguf_unique_tensors_keep_their_shard_ownership() -> Result<()> {
        for architecture in ["llama", "deepseek4"] {
            let (_dir, path) = split_tensor_fixture(
                architecture,
                ["blk.0.attn_norm.weight", "blk.1.attn_norm.weight"],
            )?;
            let stage = load_stage_tensors(&path, LayerRange::solo(2), &Device::Cpu)?;
            if architecture == "deepseek4" {
                assert_eq!(stage.raw_tensors.len(), 2);
            } else {
                assert_eq!(stage.tensors.len(), 2);
                let mut loader = StageTensorLoader::open(&path)?;
                for layer in 0..2 {
                    let stage =
                        loader.load_range(LayerRange::new(layer, layer + 1, 2), &Device::Cpu)?;
                    assert_eq!(stage.tensors.len(), 1);
                    let tensor = stage
                        .tensors
                        .values()
                        .next()
                        .unwrap()
                        .dequantize(&Device::Cpu)?;
                    assert_eq!(
                        tensor.flatten_all()?.to_vec1::<f32>()?,
                        vec![layer as f32 + 1.; 8]
                    );
                }
            }
            let profile = estimate_stage_memory_profile(&path, "unique", 2, &Device::Cpu)?;
            if architecture == "deepseek4" {
                assert!(profile.layer_gpu_bytes.iter().all(|bytes| *bytes >= 64));
            } else {
                assert_eq!(profile.layer_gpu_bytes, [32, 32]);
            }
        }
        Ok(())
    }

    fn split_descriptor_fixture(
        architecture: &str,
        edits: &[(usize, &str, Option<gguf_file::Value>)],
    ) -> Result<(tempfile::TempDir, PathBuf)> {
        use candle_core::quantized::QTensor;
        use gguf_file::Value as V;
        let dir = tempfile::tempdir()?;
        for index in 0..2 {
            let mut metadata = HashMap::from([
                ("split.no".to_owned(), V::U16(index as u16)),
                ("split.count".to_owned(), V::U16(2)),
                ("split.tensors.count".to_owned(), V::I32(2)),
            ]);
            // Match gguf-split: only the first shard repeats model metadata.
            if index == 0 {
                metadata.extend([
                    (
                        "general.architecture".into(),
                        V::String(architecture.into()),
                    ),
                    (format!("{architecture}.block_count"), V::U32(2)),
                    (format!("{architecture}.embedding_length"), V::U32(4)),
                    (format!("{architecture}.attention.head_count"), V::U32(1)),
                ]);
            }
            for (shard, key, value) in edits {
                if *shard == index {
                    match value {
                        Some(value) => {
                            metadata.insert((*key).into(), value.clone());
                        }
                        None => {
                            metadata.remove(*key);
                        }
                    }
                }
            }
            let refs = metadata
                .iter()
                .map(|(key, value)| (key.as_str(), value))
                .collect::<Vec<_>>();
            let tensor =
                candle_core::Tensor::from_vec(vec![index as f32 + 1.; 8], (2, 4), &Device::Cpu)?;
            let quant = QTensor::quantize(&tensor, GgmlDType::F32)?;
            let name = format!("blk.{index}.attn_norm.weight");
            gguf_file::write(
                &mut std::fs::File::create(
                    dir.path()
                        .join(format!("model-{:05}-of-00002.gguf", index + 1)),
                )?,
                &refs,
                &[(name.as_str(), &quant)],
            )?;
        }
        let first = dir.path().join("model-00001-of-00002.gguf");
        Ok((dir, first))
    }

    #[test]
    fn split_descriptors_reject_wrong_files_before_weights_or_admission() -> Result<()> {
        use gguf_file::Value as V;
        for architecture in ["llama", "deepseek4"] {
            for (edits, message) in [
                (vec![(0, "split.no", Some(V::U16(1)))], "split.no"),
                (vec![(1, "split.no", Some(V::U16(0)))], "split.no"),
                (vec![(1, "split.no", None)], "split.no"),
                (vec![(0, "split.count", Some(V::U16(3)))], "split.count"),
                (vec![(1, "split.count", Some(V::U16(0)))], "split.count"),
                (
                    vec![
                        (0, "split.tensors.count", Some(V::I32(3))),
                        (1, "split.tensors.count", Some(V::I32(3))),
                    ],
                    "actual tensor count",
                ),
                (
                    vec![(1, "split.tensors.count", Some(V::I32(3)))],
                    "between shards",
                ),
                (vec![(1, "split.no", Some(V::I32(-1)))], "negative"),
                (
                    vec![(1, "split.count", Some(V::String("2".into())))],
                    "expected an integer",
                ),
                (
                    vec![(1, "general.architecture", Some(V::String("phi3".into())))],
                    "architecture differs",
                ),
            ] {
                let (_dir, path) = split_descriptor_fixture(architecture, &edits)?;
                let mut errors = vec![
                    load_stage_tensors(&path, LayerRange::new(0, 1, 2), &Device::Cpu).err(),
                    estimate_stage_memory_profile(&path, "bad-split", 2, &Device::Cpu).err(),
                ];
                if architecture == "llama" {
                    errors.push(StageTensorLoader::open(&path).err());
                }
                for error in errors {
                    let error = format!("{:#}", error.expect("bad split metadata must fail"));
                    assert!(error.contains(message), "{message}: {error}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn split_descriptors_allow_first_only_model_metadata_and_optional_repeats() -> Result<()> {
        for architecture in ["llama", "deepseek4"] {
            for edits in [
                vec![],
                vec![(1, "split.count", None), (1, "split.tensors.count", None)],
            ] {
                let (_dir, path) = split_descriptor_fixture(architecture, &edits)?;
                let stage = load_stage_tensors(&path, LayerRange::solo(2), &Device::Cpu)?;
                assert_eq!(stage.tensors.len() + stage.raw_tensors.len(), 2);
                estimate_stage_memory_profile(&path, "valid-split", 2, &Device::Cpu)?;
                if architecture == "llama" {
                    let mut loader = StageTensorLoader::open(&path)?;
                    assert_eq!(
                        loader
                            .load_range(LayerRange::new(1, 2, 2), &Device::Cpu)?
                            .tensors
                            .len(),
                        1
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn split_descriptors_allow_merged_count_zero_but_reject_a_renamed_shard() -> Result<()> {
        use gguf_file::Value as V;
        for architecture in ["llama", "deepseek4"] {
            let (dir, path) = split_descriptor_fixture(architecture, &[])?;
            let single = dir.path().join("renamed.gguf");
            std::fs::copy(path, &single)?;
            assert!(load_stage_tensors(&single, LayerRange::solo(2), &Device::Cpu).is_err());
            assert!(estimate_stage_memory_profile(&single, "renamed", 2, &Device::Cpu).is_err());
            let (dir, path) = split_descriptor_fixture(
                architecture,
                &[
                    (0, "split.count", Some(V::U16(0))),
                    (0, "split.tensors.count", Some(V::I32(1))),
                ],
            )?;
            let single = dir.path().join("merged.gguf");
            std::fs::copy(path, &single)?;
            load_stage_tensors(&single, LayerRange::new(0, 1, 2), &Device::Cpu)?;
            estimate_stage_memory_profile(&single, "merged", 2, &Device::Cpu)?;
            if architecture == "llama" {
                StageTensorLoader::open(&single)?;
            }
        }
        Ok(())
    }

    #[test]
    fn embedding_vocabulary_bounds_precede_backend_loading() -> Result<()> {
        use candle_core::{DType, Tensor, quantized::QTensor};
        use gguf_file::Value as V;
        for architecture in ["llama", "deepseek4"] {
            for split in [false, true] {
                for shape in [vec![2, 4], vec![8]] {
                    let dir = tempfile::tempdir()?;
                    let path = dir.path().join(if split {
                        "model-00001-of-00002.gguf"
                    } else {
                        "model.gguf"
                    });
                    let metadata = [
                        (
                            "general.architecture".into(),
                            V::String(architecture.into()),
                        ),
                        (format!("{architecture}.block_count"), V::U32(2)),
                        (format!("{architecture}.embedding_length"), V::U32(4)),
                        (format!("{architecture}.attention.head_count"), V::U32(1)),
                        (format!("{architecture}.vocab_size"), V::U32(3)),
                    ];
                    let refs = metadata
                        .iter()
                        .map(|(key, value)| (key.as_str(), value))
                        .collect::<Vec<_>>();
                    let embedding = QTensor::quantize(
                        &Tensor::zeros(shape.as_slice(), DType::F32, &Device::Cpu)?,
                        GgmlDType::F32,
                    )?;
                    gguf_file::write(
                        &mut std::fs::File::create(&path)?,
                        &refs,
                        &[("token_embd.weight", &embedding)],
                    )?;
                    if split {
                        gguf_file::write(
                            &mut std::fs::File::create(
                                dir.path().join("model-00002-of-00002.gguf"),
                            )?,
                            &[],
                            &[],
                        )?;
                    }
                    let mut errors = vec![
                        load_stage_tensors(&path, LayerRange::solo(2), &Device::Cpu).err(),
                        estimate_stage_memory_profile(&path, "bad-embedding", 2, &Device::Cpu)
                            .err(),
                    ];
                    if architecture == "llama" {
                        errors.push(StageTensorLoader::open(&path).err());
                    }
                    for error in errors {
                        let error = error
                            .expect("embedding must cover every admitted ID")
                            .to_string();
                        assert!(error.contains("declared vocabulary"), "{error}");
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn embedding_vocabulary_allows_padding_tied_heads_and_partial_stages() -> Result<()> {
        use candle_core::{DType, Tensor, quantized::QTensor};
        use gguf_file::Value as V;
        for architecture in ["llama", "deepseek4"] {
            for untied in [false, true] {
                let dir = tempfile::tempdir()?;
                let path = dir.path().join("model-00001-of-00002.gguf");
                let metadata = [
                    (
                        "general.architecture".into(),
                        V::String(architecture.into()),
                    ),
                    (format!("{architecture}.block_count"), V::U32(2)),
                    (format!("{architecture}.embedding_length"), V::U32(4)),
                    (format!("{architecture}.attention.head_count"), V::U32(1)),
                    (format!("{architecture}.vocab_size"), V::U32(3)),
                ];
                let refs = metadata
                    .iter()
                    .map(|(key, value)| (key.as_str(), value))
                    .collect::<Vec<_>>();
                let embedding = QTensor::quantize(
                    &Tensor::ones((4, 4), DType::F32, &Device::Cpu)?,
                    GgmlDType::F32,
                )?;
                gguf_file::write(
                    &mut std::fs::File::create(&path)?,
                    &refs,
                    &[("token_embd.weight", &embedding)],
                )?;
                let output = if untied {
                    vec![("output.weight", &embedding)]
                } else {
                    vec![]
                };
                gguf_file::write(
                    &mut std::fs::File::create(dir.path().join("model-00002-of-00002.gguf"))?,
                    &[],
                    &output,
                )?;
                for range in [
                    LayerRange::solo(2),
                    LayerRange::new(0, 1, 2),
                    LayerRange::new(1, 2, 2),
                ] {
                    load_stage_tensors(&path, range.clone(), &Device::Cpu)?;
                    if architecture == "llama" {
                        StageTensorLoader::open(&path)?.load_range(range, &Device::Cpu)?;
                    }
                }
                estimate_stage_memory_profile(&path, "padded-embedding", 2, &Device::Cpu)?;
            }
        }
        Ok(())
    }

    #[test]
    fn memory_profiles_reject_layer_counts_that_loading_rejects() -> Result<()> {
        for architecture in ["llama", "deepseek4"] {
            let (_dir, path) = split_tensor_fixture(
                architecture,
                ["blk.0.attn_norm.weight", "blk.1.attn_norm.weight"],
            )?;
            assert_eq!(read_gguf_metadata(&path)?.num_layers, 2);
            for total_layers in [1, 3] {
                let range = LayerRange::solo(total_layers);
                assert!(load_stage_tensors(&path, range.clone(), &Device::Cpu).is_err());
                for result in [
                    estimate_stage_memory_profile(&path, "mismatch", total_layers, &Device::Cpu)
                        .map(|_| ()),
                    estimate_stage_vram_mb(&path, &range, &Device::Cpu).map(|_| ()),
                ] {
                    let error =
                        result.expect_err("profile must use the model's complete layer count");
                    assert!(
                        error.to_string().contains("doesn't match model num_layers"),
                        "{error}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn yarn_beta_metadata_preserves_defaults_and_overrides() -> Result<()> {
        let mut md = minimal_llama_metadata();
        let parsed = parse_gguf_metadata(&md)?;
        assert_eq!(
            (parsed.rope_yarn_beta_fast, parsed.rope_yarn_beta_slow),
            (32., 1.)
        );
        md.insert(
            "llama.rope.scaling.yarn_beta_fast".into(),
            gguf_file::Value::F32(16.),
        );
        md.insert(
            "llama.rope.scaling.yarn_beta_slow".into(),
            gguf_file::Value::F32(2.),
        );
        let parsed = parse_gguf_metadata(&md)?;
        assert_eq!(
            (parsed.rope_yarn_beta_fast, parsed.rope_yarn_beta_slow),
            (16., 2.)
        );
        md.insert(
            "llama.rope.scaling.yarn_beta_slow".into(),
            gguf_file::Value::F32(f32::NAN),
        );
        assert!(parse_gguf_metadata(&md).is_err());
        Ok(())
    }

    #[test]
    fn tensor_span_inspection_reports_absolute_bytes_without_loading_weights() -> Result<()> {
        use candle_core::quantized::QTensor;
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("spans.gguf");
        let tensor = candle_core::Tensor::from_vec(
            (0..8).map(|i| i as f32).collect(),
            (2, 4),
            &Device::Cpu,
        )?;
        let quant = QTensor::quantize(&tensor, GgmlDType::F32)?;
        gguf_file::write(
            &mut std::fs::File::create(&path)?,
            &[],
            &[("z", &quant), ("a", &quant)],
        )?;
        let spans = read_gguf_tensor_spans(&path)?;
        assert_eq!(
            spans.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            ["a", "z"]
        );
        let bytes = std::fs::read(&path)?;
        for span in spans {
            assert_eq!(span.shape, [2, 4]);
            assert_eq!(span.dtype, "F32");
            assert_eq!(span.end - span.start, 32);
            let values = bytes[span.start as usize..span.end as usize]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(values, (0..8).map(|i| i as f32).collect::<Vec<_>>());
        }
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)?
            .set_len((bytes.len() - 1) as u64)?;
        assert!(read_gguf_tensor_spans(&path).is_err());
        Ok(())
    }

    #[test]
    fn rejects_present_metadata_with_wrong_scalar_or_array_type() {
        let cases = [
            (
                "llama.context_length",
                gguf_file::Value::String("4096".into()),
            ),
            ("llama.expert_weights_norm", gguf_file::Value::U32(1)),
            ("llama.rope.dimension_sections", gguf_file::Value::U32(64)),
            ("llama.rope.freq_base", gguf_file::Value::Bool(true)),
        ];
        for (key, value) in cases {
            let mut metadata = minimal_llama_metadata();
            metadata.insert(key.into(), value);
            let error = parse_gguf_metadata(&metadata).unwrap_err().to_string();
            assert!(error.contains(key), "{error}");
        }

        let metadata = minimal_llama_metadata();
        let parsed = parse_gguf_metadata(&metadata).unwrap();
        assert_eq!(parsed.kda_gate_lower_bound, f64::NEG_INFINITY);

        let mut metadata = minimal_llama_metadata();
        metadata.insert(
            "llama.kda.gate_lower_bound".into(),
            gguf_file::Value::F64(f64::NEG_INFINITY),
        );
        assert_eq!(
            parse_gguf_metadata(&metadata).unwrap().kda_gate_lower_bound,
            f64::NEG_INFINITY
        );

        metadata.insert(
            "llama.kda.gate_lower_bound".into(),
            gguf_file::Value::F64(f64::INFINITY),
        );
        assert!(parse_gguf_metadata(&metadata).is_err());
    }

    #[test]
    fn rejects_malformed_or_nonfinite_metadata_array_values() {
        let mut metadata = minimal_llama_metadata();
        metadata.insert(
            "llama.attention.head_count_kv".into(),
            gguf_file::Value::Array(vec![
                gguf_file::Value::U32(8),
                gguf_file::Value::String("bad".into()),
            ]),
        );
        let error = parse_gguf_metadata(&metadata).unwrap_err().to_string();
        assert!(error.contains("head_count_kv array value 1"), "{error}");

        let mut metadata = minimal_llama_metadata();
        metadata.insert(
            "llama.swiglu_clamp_exp".into(),
            gguf_file::Value::Array(vec![gguf_file::Value::F32(f32::NAN)]),
        );
        let error = parse_gguf_metadata(&metadata).unwrap_err().to_string();
        assert!(error.contains("swiglu_clamp_exp"), "{error}");
        assert!(error.contains("finite"), "{error}");

        let mut metadata = minimal_llama_metadata();
        metadata.insert(
            "llama.attention.sliding_window_pattern".into(),
            gguf_file::Value::Array(vec![gguf_file::Value::Bool(true), gguf_file::Value::U32(1)]),
        );
        let error = parse_gguf_metadata(&metadata).unwrap_err().to_string();
        assert!(
            error.contains("sliding_window_pattern array value 1"),
            "{error}"
        );

        // Negative infinity is the documented KDA lower-bound sentinel and is
        // covered by the scalar metadata test above. Other non-finite values
        // remain invalid.
    }

    #[test]
    fn rejects_negative_and_zero_structural_metadata() {
        let mut md = HashMap::from([
            (
                "general.architecture".to_string(),
                gguf_file::Value::String("llama".to_string()),
            ),
            ("llama.block_count".to_string(), gguf_file::Value::I32(-1)),
            (
                "llama.embedding_length".to_string(),
                gguf_file::Value::U32(4096),
            ),
            (
                "llama.attention.head_count".to_string(),
                gguf_file::Value::U32(32),
            ),
        ]);
        assert!(
            parse_gguf_metadata(&md)
                .unwrap_err()
                .to_string()
                .contains("non-negative")
        );

        md.insert("llama.block_count".to_string(), gguf_file::Value::U32(1));
        md.insert(
            "llama.attention.head_count".to_string(),
            gguf_file::Value::U32(0),
        );
        assert!(
            parse_gguf_metadata(&md)
                .unwrap_err()
                .to_string()
                .contains("head count must be positive")
        );
    }

    #[test]
    fn rejects_semantically_invalid_rope_norm_and_context_metadata() {
        let cases = [
            (
                "llama.attention.layer_norm_rms_epsilon",
                gguf_file::Value::F64(0.0),
            ),
            (
                "llama.attention.layer_norm_rms_epsilon",
                gguf_file::Value::F64(-1.0),
            ),
            (
                "llama.attention.layer_norm_rms_epsilon",
                gguf_file::Value::F64(1e-300),
            ),
            (
                "llama.attention.layer_norm_rms_epsilon",
                gguf_file::Value::F64(1e-40),
            ),
            ("llama.rope.freq_base", gguf_file::Value::F64(0.0)),
            ("llama.rope.freq_base", gguf_file::Value::F64(-1.0)),
            ("llama.rope.freq_base", gguf_file::Value::F64(1e300)),
            ("llama.rope.freq_base", gguf_file::Value::F64(1e-300)),
            ("llama.rope.freq_base", gguf_file::Value::F64(1e-40)),
            (
                "llama.context_length",
                gguf_file::Value::U64((MAX_SUPPORTED_CONTEXT_LENGTH + 1) as u64),
            ),
            (
                "llama.context_length",
                gguf_file::Value::U64(usize::MAX as u64),
            ),
        ];
        for (key, value) in cases {
            let mut metadata = minimal_llama_metadata();
            metadata.insert(key.into(), value);
            let error = parse_gguf_metadata(&metadata).unwrap_err().to_string();
            assert!(error.contains(key), "{key}: {error}");
        }
    }

    #[test]
    fn accepts_existing_one_million_token_context_limit() {
        let mut metadata = minimal_llama_metadata();
        metadata.insert(
            "llama.context_length".into(),
            gguf_file::Value::U64(MAX_SUPPORTED_CONTEXT_LENGTH as u64),
        );
        let parsed = parse_gguf_metadata(&metadata).unwrap();
        assert_eq!(parsed.context_length, MAX_SUPPORTED_CONTEXT_LENGTH);
    }

    #[test]
    fn llama4_attention_defaults_and_scalar_periods_match_chunked_contract() -> Result<()> {
        let mut metadata: HashMap<_, _> = minimal_llama_metadata()
            .into_iter()
            .map(|(key, value)| (key.replace("llama.", "llama4."), value))
            .collect();
        metadata.insert(
            "general.architecture".into(),
            gguf_file::Value::String("llama4".into()),
        );
        metadata.insert("llama4.block_count".into(), gguf_file::Value::U32(8));
        let parsed = parse_gguf_metadata(&metadata)?;
        assert_eq!(parsed.sliding_window, 8192);
        assert_eq!(
            parsed.sliding_window_pattern,
            [true, true, true, false, true, true, true, false]
        );

        for (period, expected) in [
            (0, vec![true; 8]),
            (1, vec![false; 8]),
            (2, vec![true, false, true, false, true, false, true, false]),
            (4, vec![true, true, true, false, true, true, true, false]),
            (u32::MAX, vec![true; 8]),
        ] {
            metadata.insert(
                "llama4.attention.sliding_window_pattern".into(),
                gguf_file::Value::U32(period),
            );
            assert_eq!(
                parse_gguf_metadata(&metadata)?.sliding_window_pattern,
                expected
            );
        }

        metadata.insert(
            "llama4.attention.sliding_window".into(),
            gguf_file::Value::U32(0),
        );
        let parsed = parse_gguf_metadata(&metadata)?;
        assert_eq!(parsed.sliding_window, 0);
        assert_eq!(parsed.sliding_window_pattern, vec![false; 8]);
        metadata.insert(
            "llama4.attention.sliding_window".into(),
            gguf_file::Value::U32(8192),
        );
        assert_eq!(parse_gguf_metadata(&metadata)?.sliding_window, 8192);

        // Other model families retain their missing-key defaults and strict
        // boolean-array contract.
        let ordinary = parse_gguf_metadata(&minimal_llama_metadata())?;
        assert_eq!(ordinary.sliding_window, 0);
        assert!(ordinary.sliding_window_pattern.is_empty());
        Ok(())
    }

    #[test]
    fn llama4_attention_rejects_unsupported_sizes_and_patterns() {
        use gguf_file::Value as V;
        let cases = [
            ("llama4.attention.sliding_window", V::U32(4096)),
            ("llama4.attention.sliding_window", V::U64(u64::MAX)),
            ("llama4.attention.sliding_window", V::I32(-1)),
            ("llama4.attention.sliding_window", V::Bool(false)),
            ("llama4.attention.sliding_window_pattern", V::I32(-1)),
            (
                "llama4.attention.sliding_window_pattern",
                V::U64(u32::MAX as u64 + 1),
            ),
            (
                "llama4.attention.sliding_window_pattern",
                V::Array(vec![V::Bool(true), V::Bool(false)]),
            ),
            ("llama4.attention.sliding_window_pattern", V::F32(4.0)),
        ];
        for (key, value) in cases {
            let metadata = HashMap::from([(key.to_string(), value)]);
            let error = llama4_attention_metadata(&metadata, 2)
                .unwrap_err()
                .to_string();
            assert!(error.contains(key), "{key}: {error}");
        }
    }

    #[test]
    fn llama4_scout_context_preserves_declared_limit_and_bounded_rope_memory() -> Result<()> {
        let mut metadata: HashMap<_, _> = minimal_llama_metadata()
            .into_iter()
            .map(|(key, value)| (key.replace("llama.", "llama4."), value))
            .collect();
        metadata.insert(
            "general.architecture".into(),
            gguf_file::Value::String("llama4".into()),
        );
        metadata.insert(
            "llama4.context_length".into(),
            gguf_file::Value::U64(MAX_LLAMA4_CONTEXT_LENGTH as u64),
        );
        let parsed = parse_gguf_metadata(&metadata)?;
        assert_eq!(parsed.context_length, 10_485_760);
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("scout-context.gguf");
        let values: Vec<_> = metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
        gguf_file::write(&mut std::fs::File::create(&path)?, &values, &[])?;
        assert_eq!(read_gguf_metadata(&path)?.context_length, 10_485_760);
        let profile =
            estimate_stage_memory_profile(&path, "scout-context", parsed.num_layers, &Device::Cpu)?;
        // Two F32 tables each hold half a head for only the initial 8,192 rows.
        assert_eq!(profile.common_cpu_bytes, 8192 * parsed.head_dim as u64 * 4);
        let mut bounded = parsed.clone();
        bounded.context_length = 8192;
        assert_eq!(
            generated_rope_memory_bytes(&parsed, &Device::Cpu),
            generated_rope_memory_bytes(&bounded, &Device::Cpu)
        );
        for invalid in [MAX_LLAMA4_CONTEXT_LENGTH as u64 + 1, u64::MAX] {
            metadata.insert(
                "llama4.context_length".into(),
                gguf_file::Value::U64(invalid),
            );
            let error = parse_gguf_metadata(&metadata).unwrap_err().to_string();
            assert!(error.contains("llama4.context_length"), "{error}");
        }
        let mut ordinary = minimal_llama_metadata();
        ordinary.insert(
            "llama.context_length".into(),
            gguf_file::Value::U64(MAX_LLAMA4_CONTEXT_LENGTH as u64),
        );
        let error = parse_gguf_metadata(&ordinary).unwrap_err().to_string();
        assert!(error.contains("llama.context_length"), "{error}");
        Ok(())
    }

    #[test]
    fn accepts_safe_subunit_rope_theta_but_rejects_nonfinite_generated_values() {
        let mut metadata = minimal_llama_metadata();
        metadata.insert("llama.rope.freq_base".into(), gguf_file::Value::F64(0.5));
        assert!(parse_gguf_metadata(&metadata).is_ok());
        assert!(
            validate_rope_theta_generation(
                0.5,
                128,
                MAX_SUPPORTED_CONTEXT_LENGTH,
                "test.rope.freq_base"
            )
            .is_ok()
        );
        let error = validate_rope_theta_generation(
            1e-40,
            128,
            MAX_SUPPORTED_CONTEXT_LENGTH,
            "test.rope.freq_base",
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("test.rope.freq_base"));
        assert!(error.contains("non-finite"));
    }

    #[test]
    fn accepts_zero_optional_rope_sentinels() {
        let mut metadata = minimal_llama_metadata();
        metadata.insert(
            "llama.rope.freq_base_swa".into(),
            gguf_file::Value::F64(0.0),
        );
        metadata.insert(
            "llama.attention.compress_rope_freq_base".into(),
            gguf_file::Value::F64(0.0),
        );
        let parsed = parse_gguf_metadata(&metadata).unwrap();
        assert_eq!(parsed.rope_theta_swa, 0.0);
        assert_eq!(parsed.compress_rope_freq_base, 0.0);
    }

    #[test]
    fn rejects_invalid_optional_rope_values() {
        let cases = [
            ("llama.rope.freq_base_swa", gguf_file::Value::F64(-1.0)),
            ("llama.rope.freq_base_swa", gguf_file::Value::F64(1e300)),
            ("llama.rope.freq_base_swa", gguf_file::Value::F64(1e-40)),
            (
                "llama.attention.compress_rope_freq_base",
                gguf_file::Value::F64(-1.0),
            ),
            (
                "llama.attention.compress_rope_freq_base",
                gguf_file::Value::F64(1e-300),
            ),
            (
                "llama.attention.compress_rope_freq_base",
                gguf_file::Value::F64(1e-40),
            ),
        ];
        for (key, value) in cases {
            let mut metadata = minimal_llama_metadata();
            metadata.insert(key.into(), value);
            let error = parse_gguf_metadata(&metadata).unwrap_err().to_string();
            assert!(error.contains(key), "{key}: {error}");
        }
    }

    #[test]
    fn rejects_physical_and_executable_layer_counts_above_application_limit() {
        let values = HashMap::from([
            (
                "general.architecture".into(),
                gguf_file::Value::String("llama".into()),
            ),
            (
                "llama.block_count".into(),
                gguf_file::Value::U32((MAX_MODEL_LAYERS as u32) + 1),
            ),
            ("llama.embedding_length".into(), gguf_file::Value::U32(4096)),
            (
                "llama.attention.head_count".into(),
                gguf_file::Value::U32(32),
            ),
        ]);

        let parse_error = parse_gguf_metadata(&values).unwrap_err().to_string();
        assert!(parse_error.contains("physical layer count"));
        let count_error = executable_layer_count(&values).unwrap_err().to_string();
        assert!(count_error.contains("physical layer count"));
    }

    #[test]
    fn rejects_zero_and_oversized_requested_profile_layer_counts() {
        assert!(validate_layer_count(0, "test layer count").is_err());
        assert!(validate_layer_count(MAX_MODEL_LAYERS + 1, "test layer count").is_err());
        assert!(validate_layer_count(MAX_MODEL_LAYERS, "test layer count").is_ok());
    }

    #[test]
    fn preflight_rejects_huge_header_counts_before_candle_parse() {
        let path = std::env::temp_dir().join(format!(
            "dinf-gguf-count-limit-{}-{}.gguf",
            std::process::id(),
            std::thread::current()
                .name()
                .unwrap_or("test")
                .replace(':', "_")
        ));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&(MAX_GGUF_TENSOR_COUNT + 1).to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();

        let error = validate_gguf_header_counts(&path).unwrap_err().to_string();
        assert!(error.contains("tensor count"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn preflight_budgets_parsed_metadata_values_not_only_disk_bytes() {
        fn byte_arrays(path: &std::path::Path, arrays: u64) {
            let mut file = std::fs::File::create(path).unwrap();
            file.write_all(b"GGUF").unwrap();
            file.write_all(&3u32.to_le_bytes()).unwrap();
            file.write_all(&0u64.to_le_bytes()).unwrap(); // tensors
            file.write_all(&arrays.to_le_bytes()).unwrap(); // metadata records
            for index in 0..arrays {
                let key = format!("k{index}");
                file.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
                file.write_all(key.as_bytes()).unwrap();
                file.write_all(&9u32.to_le_bytes()).unwrap(); // array
                file.write_all(&0u32.to_le_bytes()).unwrap(); // of u8
                file.write_all(&crate::gguf_limits::MAX_GGUF_ARRAY_ELEMENTS.to_le_bytes())
                    .unwrap();
                file.write_all(&vec![0u8; crate::gguf_limits::MAX_GGUF_ARRAY_ELEMENTS as usize])
                    .unwrap();
            }
            // Pad to the default 32-byte alignment where tensor data starts.
            let length = file.metadata().unwrap().len();
            file.set_len(length.div_ceil(32) * 32).unwrap();
        }
        let path = std::env::temp_dir().join(format!(
            "dinf-gguf-value-budget-{}-{}.gguf",
            std::process::id(),
            std::thread::current().name().unwrap_or("test").replace(':', "_")
        ));
        // ~9 MiB on disk, but ~300 MiB once every byte becomes a Value.
        byte_arrays(&path, 9);
        let error = validate_gguf_header_counts(&path).unwrap_err().to_string();
        assert!(error.contains("metadata values"), "{error}");

        byte_arrays(&path, 7);
        let (_, stats) = open_validated_gguf(&path).unwrap();
        let elements = crate::gguf_limits::MAX_GGUF_ARRAY_ELEMENTS;
        assert_eq!(stats.metadata_values, 7 * elements + 7);
        let disk = std::fs::metadata(&path).unwrap().len();
        assert!(stats.parsed_bytes(disk) > 16 * disk, "split budgets must see parsed size");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn preflight_rejects_aggregate_header_budget_before_candle_parse() {
        let path = std::env::temp_dir().join(format!(
            "dinf-gguf-header-budget-{}-{}.gguf",
            std::process::id(),
            std::thread::current()
                .name()
                .unwrap_or("test")
                .replace(':', "_")
        ));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(b"GGUF").unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();
        file.write_all(&1u64.to_le_bytes()).unwrap();
        file.write_all(&1u64.to_le_bytes()).unwrap();
        file.write_all(b"x").unwrap();
        file.write_all(&8u32.to_le_bytes()).unwrap(); // string metadata value
        file.write_all(&MAX_GGUF_HEADER_BYTES.to_le_bytes())
            .unwrap();
        // Keep the sparse backing file long enough that the budget—not the
        // physical EOF check—is the rejection reason.
        file.set_len(MAX_GGUF_HEADER_BYTES + 4096).unwrap();
        drop(file);

        let error = validate_gguf_header_counts(&path).unwrap_err().to_string();
        assert!(error.contains("application limit"));
        std::fs::remove_file(path).unwrap();
    }

    fn write_single_tensor_v3(
        path: &Path,
        dimensions: &[u64],
        tensor_type: u32,
        tensor_offset: u64,
        file_len: u64,
    ) {
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(b"GGUF").unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&1u64.to_le_bytes()).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();
        file.write_all(&1u64.to_le_bytes()).unwrap();
        file.write_all(b"x").unwrap();
        file.write_all(&(dimensions.len() as u32).to_le_bytes())
            .unwrap();
        for dimension in dimensions {
            file.write_all(&dimension.to_le_bytes()).unwrap();
        }
        file.write_all(&tensor_type.to_le_bytes()).unwrap();
        file.write_all(&tensor_offset.to_le_bytes()).unwrap();
        let minimum_len = file.stream_position().unwrap();
        file.set_len(file_len.max(minimum_len)).unwrap();
    }

    #[test]
    fn preflight_rejects_oversized_and_zero_tensor_dimensions() {
        let path = std::env::temp_dir().join(format!(
            "dinf-gguf-tensor-shape-{}-{}.gguf",
            std::process::id(),
            std::thread::current()
                .name()
                .unwrap_or("test")
                .replace(':', "_")
        ));
        write_single_tensor_v3(&path, &[MAX_GGUF_TENSOR_ELEMENTS + 1], 0, 0, 128);
        let error = validate_gguf_header_counts(&path).unwrap_err().to_string();
        assert!(error.contains("element count"), "{error}");

        write_single_tensor_v3(&path, &[0], 0, 0, 128);
        let error = validate_gguf_header_counts(&path).unwrap_err().to_string();
        assert!(error.contains("dimensions must be positive"), "{error}");

        write_single_tensor_v3(&path, &[MAX_GGUF_TENSOR_BYTES / 4 + 1], 0, 0, 128);
        let error = validate_gguf_header_counts(&path).unwrap_err().to_string();
        assert!(error.contains("tensor byte length"), "{error}");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn preflight_rejects_invalid_quant_blocks_and_out_of_file_tensor_data() {
        let path = std::env::temp_dir().join(format!(
            "dinf-gguf-tensor-extent-{}-{}.gguf",
            std::process::id(),
            std::thread::current()
                .name()
                .unwrap_or("test")
                .replace(':', "_")
        ));
        write_single_tensor_v3(&path, &[31], 2, 0, 128);
        let error = validate_gguf_header_counts(&path).unwrap_err().to_string();
        assert!(error.contains("block size"), "{error}");

        write_single_tensor_v3(&path, &[32], 2, 128, 128);
        let error = validate_gguf_header_counts(&path).unwrap_err().to_string();
        assert!(error.contains("beyond the end"), "{error}");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn aggregate_split_header_budget_is_checked() {
        let mut total = MAX_GGUF_AGGREGATE_HEADER_BYTES;
        assert!(account_split_header_bytes(&mut total, 1).is_err());
    }

    #[test]
    fn rejects_oversized_split_shard_count_before_path_allocation() {
        let path = Path::new("model-00001-of-04097.gguf");
        let error = discover_shard_paths(path).unwrap_err().to_string();
        assert!(error.contains("shard count"));
    }

    #[test]
    fn test_layer_range_solo() {
        let r = LayerRange::solo(32);
        assert!(r.is_head());
        assert!(r.is_tail());
        assert!(r.is_solo());
        assert_eq!(r.layer_count(), 32);
        assert!(r.contains(0));
        assert!(r.contains(31));
        assert!(!r.contains(32));
    }

    #[test]
    fn fallible_layer_ranges_reject_invalid_and_forged_values() {
        assert!(LayerRange::try_new(2, 2, 4).is_err());
        assert!(LayerRange::try_new(2, 5, 4).is_err());
        assert!(LayerRange::try_solo(0).is_err());
        assert!(LayerRange::try_solo(MAX_MODEL_LAYERS + 1).is_err());
        assert!(
            LayerRange {
                start: 3,
                end: 2,
                total_layers: 4,
            }
            .validate()
            .is_err()
        );
        assert_eq!(
            LayerRange {
                start: 3,
                end: 2,
                total_layers: 4,
            }
            .layer_count(),
            0
        );
    }

    #[test]
    fn test_layer_range_head() {
        let r = LayerRange::new(0, 16, 32);
        assert!(r.is_head());
        assert!(!r.is_tail());
        assert!(!r.is_solo());
        assert!(r.contains(0));
        assert!(r.contains(15));
        assert!(!r.contains(16));
    }

    #[test]
    fn test_layer_range_tail() {
        let r = LayerRange::new(16, 32, 32);
        assert!(!r.is_head());
        assert!(r.is_tail());
        assert!(r.contains(16));
        assert!(r.contains(31));
        assert!(!r.contains(15));
    }

    #[test]
    fn test_tensor_belongs_to_stage() {
        let head = LayerRange::new(0, 16, 32);
        let tail = LayerRange::new(16, 32, 32);

        // Embedding — head for token lookup, tail for tied output heads.
        assert!(tensor_belongs_to_stage("token_embd.weight", &head));
        assert!(tensor_belongs_to_stage("token_embd.weight", &tail));

        // Output — tail only
        assert!(!tensor_belongs_to_stage("output_norm.weight", &head));
        assert!(tensor_belongs_to_stage("output_norm.weight", &tail));
        assert!(tensor_belongs_to_stage("output.weight", &tail));

        // Block tensors
        assert!(tensor_belongs_to_stage("blk.0.attn_q.weight", &head));
        assert!(tensor_belongs_to_stage("blk.15.attn_q.weight", &head));
        assert!(!tensor_belongs_to_stage("blk.16.attn_q.weight", &head));
        assert!(tensor_belongs_to_stage("blk.16.attn_q.weight", &tail));
        assert!(tensor_belongs_to_stage("blk.31.ffn_down.weight", &tail));
    }

    #[test]
    fn qwen35_nextn_mtp_block_is_outside_the_executable_trunk() -> Result<()> {
        let values = HashMap::from([
            (
                "general.architecture".into(),
                gguf_file::Value::String("qwen35".into()),
            ),
            ("qwen35.block_count".into(), gguf_file::Value::U32(65)),
            (
                "qwen35.nextn_predict_layers".into(),
                gguf_file::Value::U32(1),
            ),
        ]);
        assert_eq!(executable_layer_count(&values)?, 64);
        Ok(())
    }

    #[test]
    fn glm5_nextn_mtp_block_is_outside_the_executable_trunk() {
        let head = LayerRange::new(0, 11, 45);
        let tail = LayerRange::new(11, 45, 45);

        assert!(tensor_belongs_to_stage("blk.44.attn_output.weight", &tail));
        assert!(!tensor_belongs_to_stage("blk.45.attn_output.weight", &tail));
        assert!(!tensor_belongs_to_stage(
            "blk.45.nextn.eh_proj.weight",
            &tail
        ));
        assert!(!tensor_belongs_to_stage(
            "blk.45.nextn.eh_proj.weight",
            &head
        ));
    }

    fn make_metadata(num_layers: usize, num_kv_heads: usize, head_dim: usize) -> GgufMetadata {
        GgufMetadata {
            architecture: "llama".into(),
            num_layers,
            hidden_dim: 4096,
            num_attention_heads: 32,
            num_kv_heads,
            head_dim,
            intermediate_dim: 11008,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
            post_norm_eps: 0.0,
            moe_latent_size: 0,
            logit_scale: 1.0,
            rope_theta: 10000.0,
            context_length: 4096,
            expert_count: 0,
            expert_used_count: 0,
            expert_ffn_dim: 0,
            ssm_state_size: 0,
            ssm_conv_kernel: 0,
            ssm_group_count: 0,
            ssm_time_step_rank: 0,
            ssm_inner_size: 0,
            rope_dimension_count: 0,
            rope_dimension_sections: Vec::new(),
            rope_scaling_factor: 1.0,
            rope_scaling_original_context_length: 4096,
            rope_yarn_log_multiplier: 0.0,
            rope_yarn_beta_fast: 32.,
            rope_yarn_beta_slow: 1.,
            rope_dimension_count_swa: 0,
            head_dim_swa: 0,
            rope_theta_swa: 0.0,
            sliding_window: 0,
            sliding_window_pattern: Vec::new(),
            final_logit_softcapping: 0.0,
            full_attention_interval: 0,
            head_count_kv_arr: Vec::new(),
            feed_forward_length_arr: Vec::new(),
            attention_scale: 0.0,
            expert_weights_scale: 0.0,
            expert_weights_norm: false,
            expert_gating_func: 0,
            shared_expert_count: 0,
            kv_lora_rank: 0,
            q_lora_rank: 0,
            qk_nope_head_dim: 0,
            qk_rope_head_dim: 0,
            v_head_dim: 0,
            first_k_dense_replace: 0,
            moe_layer_freq: 0,
            expert_group_count: 1,
            output_lora_rank: 0,
            output_group_count: 0,
            compress_ratios: Vec::new(),
            indexer_head_count: 0,
            indexer_head_dim: 0,
            indexer_top_k: 0,
            compress_rope_freq_base: 0.0,
            hyperconnection_count: 0,
            hyperconnection_sinkhorn_iterations: 0,
            hyperconnection_epsilon: 0.0,
            hash_layer_count: 0,
            kda_head_dim: 0,
            kda_gate_lower_bound: f64::NEG_INFINITY,
            indexer_kpool: 0,
            indexer_kpool_select_tail: false,
            swiglu_clamp_exp: Vec::new(),
            swiglu_clamp_shexp: Vec::new(),
        }
    }

    #[test]
    fn parses_glm5_next_hybrid_metadata() -> Result<()> {
        let mut values = HashMap::new();
        values.insert(
            "general.architecture".into(),
            gguf_file::Value::String("glm5next".into()),
        );
        let mut insert_u32 = |key: &str, value: u32| {
            values.insert(key.to_string(), gguf_file::Value::U32(value));
        };
        insert_u32("glm5next.block_count", 45);
        insert_u32("glm5next.embedding_length", 4096);
        insert_u32("glm5next.attention.head_count", 64);
        insert_u32("glm5next.attention.key_length", 512);
        insert_u32("glm5next.feed_forward_length", 12288);
        insert_u32("glm5next.vocab_size", 154880);
        insert_u32("glm5next.context_length", 1048576);
        insert_u32("glm5next.expert_count", 288);
        insert_u32("glm5next.expert_used_count", 8);
        insert_u32("glm5next.expert_feed_forward_length", 2048);
        insert_u32("glm5next.ssm.conv_kernel", 4);
        insert_u32("glm5next.kda.head_dim", 128);
        insert_u32("glm5next.attention.q_lora_rank", 1536);
        insert_u32("glm5next.attention.kv_lora_rank", 512);
        insert_u32("glm5next.attention.key_length_mla", 256);
        insert_u32("glm5next.attention.value_length_mla", 256);
        insert_u32("glm5next.attention.indexer.head_count", 32);
        insert_u32("glm5next.attention.indexer.key_length", 128);
        insert_u32("glm5next.attention.indexer.top_k", 2048);
        insert_u32("glm5next.attention.indexer.kpool", 4);
        insert_u32("glm5next.hyper_connection.count", 4);
        insert_u32("glm5next.hyper_connection.sinkhorn_iterations", 20);
        values.insert(
            "glm5next.kda.gate_lower_bound".into(),
            gguf_file::Value::F32(-5.0),
        );
        values.insert(
            "glm5next.attention.head_count_kv".into(),
            gguf_file::Value::Array(
                (0..45)
                    .map(|layer| gguf_file::Value::U32(u32::from(layer % 4 == 3)))
                    .collect(),
            ),
        );
        values.insert(
            "glm5next.swiglu_clamp_exp".into(),
            gguf_file::Value::Array((0..45).map(|_| gguf_file::Value::F32(10.0)).collect()),
        );

        let metadata = parse_gguf_metadata(&values)?;
        assert_eq!(metadata.architecture, "glm5next");
        assert_eq!(metadata.kda_head_dim, 128);
        assert_eq!(metadata.kda_gate_lower_bound, -5.0);
        assert_eq!(metadata.indexer_kpool, 4);
        assert!(metadata.indexer_kpool_select_tail);
        assert_eq!(metadata.head_count_kv_arr[3], 1);
        assert_eq!(metadata.swiglu_clamp_exp, vec![10.0; 45]);
        assert_eq!(metadata.swiglu_clamp_shexp, vec![10.0; 45]);
        assert_eq!(metadata.kv_bytes_per_token(0, 45), 11 * 1_536);
        assert_eq!(metadata.kv_bytes_per_token(0, 4), 1_536);
        assert_eq!(metadata.kv_bytes_per_token_quantized(0, 45, 4), 11 * 1_536);
        Ok(())
    }

    #[test]
    fn glm5_nextn_physical_block_count_maps_to_executable_trunk_count() -> Result<()> {
        let mut values = HashMap::new();
        values.insert(
            "general.architecture".into(),
            gguf_file::Value::String("glm5next".into()),
        );
        values.insert("glm5next.block_count".into(), gguf_file::Value::U32(46));
        values.insert(
            "glm5next.nextn_predict_layers".into(),
            gguf_file::Value::U32(1),
        );
        values.insert(
            "glm5next.embedding_length".into(),
            gguf_file::Value::U32(4096),
        );
        values.insert(
            "glm5next.attention.head_count".into(),
            gguf_file::Value::U32(64),
        );
        values.insert(
            "glm5next.attention.key_length".into(),
            gguf_file::Value::U32(512),
        );
        let metadata = parse_gguf_metadata(&values)?;
        assert_eq!(metadata.num_layers, 45);
        assert_eq!(executable_layer_count(&values)?, 45);
        Ok(())
    }

    #[test]
    fn glm5_missing_legacy_defaults_are_scoped_to_known_architecture() -> Result<()> {
        let mut values = HashMap::new();
        values.insert(
            "general.architecture".into(),
            gguf_file::Value::String("glm5next".into()),
        );
        values.insert("glm5next.block_count".into(), gguf_file::Value::U32(46));
        values.insert(
            "glm5next.nextn_predict_layers".into(),
            gguf_file::Value::U32(1),
        );
        values.insert(
            "glm5next.embedding_length".into(),
            gguf_file::Value::U32(4096),
        );
        values.insert(
            "glm5next.attention.head_count".into(),
            gguf_file::Value::U32(64),
        );
        let metadata = parse_gguf_metadata(&values)?;
        assert_eq!(metadata.indexer_kpool, 4);
        assert_eq!(metadata.swiglu_clamp_exp, vec![10.0; 45]);
        assert_eq!(metadata.swiglu_clamp_shexp, vec![10.0; 45]);

        Ok(())
    }

    #[test]
    fn executable_layer_count_preserves_legacy_glm5_artifacts() -> Result<()> {
        let mut values = HashMap::new();
        values.insert(
            "general.architecture".into(),
            gguf_file::Value::String("glm5next".into()),
        );
        values.insert("glm5next.block_count".into(), gguf_file::Value::U32(45));
        assert_eq!(executable_layer_count(&values)?, 45);
        Ok(())
    }

    #[test]
    fn glm5_rejects_malformed_nextn_layer_metadata() {
        let values = HashMap::from([
            (
                "general.architecture".into(),
                gguf_file::Value::String("glm5next".into()),
            ),
            ("glm5next.block_count".into(), gguf_file::Value::U32(46)),
            (
                "glm5next.nextn_predict_layers".into(),
                gguf_file::Value::String("one".into()),
            ),
            (
                "glm5next.embedding_length".into(),
                gguf_file::Value::U32(4096),
            ),
            (
                "glm5next.attention.head_count".into(),
                gguf_file::Value::U32(64),
            ),
        ]);

        assert!(parse_gguf_metadata(&values).is_err());
        assert!(executable_layer_count(&values).is_err());
    }

    #[test]
    fn test_kv_bytes_per_token_dense() {
        // Llama 8B: 32 layers, 8 KV heads, 128 head_dim, f16
        let md = make_metadata(32, 8, 128);
        // per layer = 2 (K+V) * 8 * 128 * 2 (f16) = 4096 bytes
        // 32 layers = 131072 bytes = 128 KB/token
        assert_eq!(md.kv_bytes_per_token(0, 32), 32 * 4096);

        // Half the layers (pipeline split)
        assert_eq!(md.kv_bytes_per_token(0, 16), 16 * 4096);
        assert_eq!(md.kv_bytes_per_token(16, 32), 16 * 4096);
    }

    #[test]
    fn kv_memory_estimates_saturate_instead_of_wrapping() {
        let md = make_metadata(1, usize::MAX, usize::MAX);
        assert_eq!(md.kv_bytes_per_token(0, 1), u64::MAX);
        assert_eq!(md.kv_bytes_per_token_quantized(0, 1, u8::MAX), u64::MAX);

        // Public callers can ask about an invalid range; estimation clamps it
        // to the declared model rather than iterating or subtracting forever.
        assert_eq!(md.kv_bytes_per_token(usize::MAX, usize::MAX), 0);
        assert_eq!(md.kv_bytes_per_token(0, usize::MAX), u64::MAX);
    }

    #[test]
    fn tensor_memory_estimate_rejects_overflow_and_byte_limit() {
        assert!(ggml_tensor_bytes(usize::MAX, 1, 2).is_err());
        #[cfg(target_pointer_width = "64")]
        assert!(ggml_tensor_bytes(MAX_GGUF_TENSOR_BYTES as usize + 1, 1, 1).is_err());
        assert_eq!(ggml_tensor_bytes(256, 256, 144).unwrap(), 144);
    }

    #[test]
    fn test_kv_bytes_per_token_hybrid_qwen35() {
        // Qwen3.5: 28 layers, every 4th is attention (layers 3,7,11,15,19,23,27)
        let mut md = make_metadata(28, 8, 128);
        md.full_attention_interval = 4;
        // 7 attention layers out of 28, f16
        let kv = md.kv_bytes_per_token(0, 28);
        assert_eq!(kv, 7 * 2 * 8 * 128 * 2);
    }

    #[test]
    fn test_kv_bytes_per_token_nemotron() {
        // Nemotron-H: 52 layers, only 6 have KV heads > 0
        let mut md = make_metadata(52, 2, 128);
        md.head_count_kv_arr = (0..52)
            .map(|i| {
                if [10, 20, 30, 36, 42, 48].contains(&i) {
                    2
                } else {
                    0
                }
            })
            .collect();
        let kv = md.kv_bytes_per_token(0, 52);
        // 6 attention layers * 2 (K+V) * 2 (kv_heads) * 128 (head_dim) * 2 (f16) = 6144
        assert_eq!(kv, 6 * 2 * 2 * 128 * 2);

        // Only layers 0-25: contains layers 10, 20 → 2 attention layers
        let kv_partial = md.kv_bytes_per_token(0, 26);
        assert_eq!(kv_partial, 2 * 2 * 2 * 128 * 2);
    }

    #[test]
    fn test_kv_bytes_per_token_gemma4_variable_heads() {
        // Gemma 4: 5 sliding-window layers followed by 1 global layer.
        let mut md = make_metadata(6, 16, 512);
        md.architecture = "gemma4".into();
        md.head_dim_swa = 256;
        md.sliding_window_pattern = vec![true, true, true, true, true, false];
        md.head_count_kv_arr = vec![16, 16, 16, 16, 16, 4];

        let swa_bytes = 2 * 16 * 256 * 2;
        let global_bytes = 2 * 4 * 512 * 2;
        assert_eq!(md.kv_bytes_per_token(0, 6), 5 * swa_bytes + global_bytes);
        assert_eq!(md.kv_bytes_per_token(5, 6), global_bytes);
    }
    #[test]
    fn deepseek41_flat_metadata_is_canonical_and_ignores_stale_json_quantization() -> Result<()> {
        use crate::deepseek4_gguf::fixtures::maya_metadata;
        let mut md = maya_metadata(false);
        md.insert(
            "deepseek41.config".into(),
            gguf_file::Value::String("{\"num_hidden_layers\":43}".into()),
        );
        md.insert(
            "deepseek41.quantization".into(),
            gguf_file::Value::String("IQ2_XXS gate/up; Q2_K down".into()),
        );
        let m = parse_gguf_metadata(&md)?;
        assert_eq!(m.architecture, "deepseek41");
        assert_eq!(
            (
                m.num_layers,
                m.hidden_dim,
                m.num_attention_heads,
                m.head_dim,
                m.num_kv_heads
            ),
            (40, 5120, 64, 512, 1)
        );
        assert_eq!(
            (m.expert_count, m.expert_used_count, m.expert_ffn_dim),
            (384, 6, 2304)
        );
        assert_eq!(
            (m.q_lora_rank, m.output_group_count, m.output_lora_rank),
            (1280, 8, 1024)
        );
        assert_eq!(
            (
                m.hyperconnection_count,
                m.hyperconnection_sinkhorn_iterations
            ),
            (4, 20)
        );
        assert!((m.hyperconnection_epsilon - 1e-6).abs() < 1e-12);
        assert!((m.rms_norm_eps - 1e-20).abs() < 1e-26);
        assert_eq!(
            (
                m.sliding_window,
                m.indexer_head_count,
                m.indexer_head_dim,
                m.indexer_top_k
            ),
            (128, 32, 128, 512)
        );
        assert_eq!(
            (m.context_length, m.rope_scaling_original_context_length),
            (1048576, 65536)
        );
        assert_eq!(
            (
                m.rope_theta,
                m.compress_rope_freq_base,
                m.rope_scaling_factor
            ),
            (10000., 160000., 16.)
        );
        assert_eq!((m.rope_yarn_beta_fast, m.rope_yarn_beta_slow), (32., 1.));
        assert_eq!(
            m.compress_ratios,
            [vec![0; 2], vec![2; 18], vec![1; 20]].concat()
        );
        assert_eq!(m.swiglu_clamp_exp, vec![10.; 40]);
        assert_eq!(executable_layer_count(&md)?, 40);
        Ok(())
    }

    #[test]
    fn deepseek41_present_malformed_flat_keys_and_unsupported_schedules_fail() {
        use crate::deepseek4_gguf::fixtures::maya_metadata;
        use gguf_file::Value as V;
        for (key, value) in [
            ("hidden_size", V::String("5120".into())),
            ("num_hidden_layers", V::U32(43)),
            ("q_lora_rank", V::U32(0)),
            ("o_groups", V::U32(7)),
            ("hc_eps", V::F32(f32::NAN)),
            ("rms_norm_eps", V::F32(-1.)),
            ("sliding_window", V::Bool(true)),
            ("index_head_dim", V::U32(32)),
            (
                "rope_scaling.original_max_position_embeddings",
                V::String("65536".into()),
            ),
            ("hidden_size", V::U64(u64::MAX / 8)),
            ("n_routed_experts", V::U32(385)),
            ("rope_theta", V::String("10000".into())),
            ("rope_scaling.beta_slow", V::F32(0.)),
            ("routed_scaling_factor", V::F32(-1.)),
            ("compress_ratios", V::Array(vec![V::U32(2); 40])),
            ("kv_source_layer_ids", V::Array(vec![V::U32(3), V::U32(20)])),
            (
                "index_source_layer_ids",
                V::Array(vec![V::String("2".into())]),
            ),
            ("engram.rows", V::Array(vec![V::U32(1), V::U32(2)])),
            ("candidate_block_size", V::U32(16)),
            ("scoring_func", V::String("sigmoid".into())),
            ("norm_topk_prob", V::U32(1)),
            ("swiglu_limit", V::F32(f32::INFINITY)),
        ] {
            let mut md = maya_metadata(false);
            md.insert(format!("deepseek41.{key}"), value);
            assert!(parse_gguf_metadata(&md).is_err(), "{key}");
        }
        let mut missing = maya_metadata(false);
        missing.remove("deepseek41.hidden_size");
        assert!(parse_gguf_metadata(&missing).is_err());
    }

    #[test]
    fn deepseek41_partial_and_wrong_device_rejected_before_tensor_allocation() -> Result<()> {
        use crate::deepseek4_gguf::fixtures::{maya_metadata, sparse};
        let (file, _) = sparse(&maya_metadata(true), &[]);
        let partial = load_stage_tensors(file.path(), LayerRange::new(0, 20, 40), &Device::Cpu)
            .err()
            .unwrap();
        assert!(partial.to_string().contains("full solo"));
        let cpu = load_stage_tensors(file.path(), LayerRange::solo(40), &Device::Cpu)
            .err()
            .unwrap();
        assert!(cpu.to_string().contains("requires Metal"));
        assert!(
            estimate_stage_memory_profile(file.path(), "wrong-device", 40, &Device::Cpu)
                .unwrap_err()
                .to_string()
                .contains("requires Metal")
        );
        Ok(())
    }

    #[test]
    fn deepseek41_tensor_layout_rejects_unknown_resident_and_wrong_expert_formats() -> Result<()> {
        use crate::deepseek4_gguf::fixtures::{maya_metadata, sparse};
        let md = maya_metadata(true);
        let m = parse_gguf_metadata(&md)?;
        for (name, shape, dtype) in [
            ("blk.0.ffn_gate_exps.weight", vec![2, 256, 256], 8),
            ("blk.0.ffn_gate_exps.weight", vec![3, 256, 256], 19),
            ("blk.0.ffn_gate_exps.weight", vec![2, 512, 256], 19),
            ("blk.0.unknown_exps.weight", vec![2, 256, 256], 19),
            ("blk.40.attn_norm.weight", vec![256], 0),
            ("blk.0.attn_q.weight", vec![256, 256], 2),
        ] {
            let (mut file, _) = sparse(&md, &[(name, &shape, dtype, 0)]);
            let content = RawGgufContent::read(file.as_file_mut())?;
            assert!(
                validate_deepseek41_tensors(&content, &m).is_err(),
                "{name} {dtype}"
            );
        }
        for (key, value) in [("split.count", 2), ("split.no", 1)] {
            let mut split_md = md.clone();
            split_md.insert(key.into(), gguf_file::Value::U32(value));
            let (mut file, _) = sparse(&split_md, &[]);
            let content = RawGgufContent::read(file.as_file_mut())?;
            assert!(
                validate_deepseek41_tensors(&content, &m)
                    .unwrap_err()
                    .to_string()
                    .contains("split GGUF")
            );
        }
        Ok(())
    }

    #[test]
    fn deepseek41_profile_reserves_one_cache_not_disk_tables_banks_or_dead_gap() -> Result<()> {
        use crate::deepseek4_gguf::fixtures::{maya_metadata, sparse};
        let md = maya_metadata(true);
        let m = parse_gguf_metadata(&md)?;
        let ordinary: &[(&str, &[usize], u32, u64)] = &[
            ("token_embd.weight", &[4, 256], 8, 0),
            ("blk.0.attn_norm.weight", &[256], 1, 2048),
        ];
        let mut complete = ordinary.to_vec();
        complete.extend([
            ("blk.0.ffn_gate_exps.weight", &[2, 256, 256][..], 19, 4096),
            (
                "blk.1.engram_embd.weight",
                &[384006168, 264][..],
                24,
                1 << 36,
            ),
            (
                "blk.14.engram_embd.weight",
                &[384016682, 264][..],
                24,
                1 << 38,
            ),
        ]);
        let (mut file, _) = sparse(&md, ordinary);
        let ordinary_header = RawGgufContent::read(file.as_file_mut())?;
        let (mut full, _) = sparse(&md, &complete);
        let full_header = RawGgufContent::read(full.as_file_mut())?;
        let a = deepseek41_memory_profile(&ordinary_header, "a", &m, &Device::Cpu)?;
        let b = deepseek41_memory_profile(&full_header, "b", &m, &Device::Cpu)?;
        assert_eq!(
            a.stage_memory_components(0, 40, 40).unwrap().0,
            b.stage_memory_components(0, 40, 40).unwrap().0
        );
        assert_eq!(a.layer_gpu_bytes, b.layer_gpu_bytes);
        assert_eq!(
            b.common_gpu_bytes,
            DEEPSEEK41_EXPERT_CACHE_BYTES + 128 * 1024 + 2048 * 8 + 2 * 32 * 4 + 1088
        );
        assert_eq!(b.common_cpu_transient_bytes, 16 * 1024 * 1024);
        let host_bytes = crate::architectures::deepseek41::experts::DEEPSEEK41_MODEL_HOST_BYTES;
        assert_eq!(
            b.common_cpu_bytes,
            deepseek41_header_memory_bytes(&full_header) + host_bytes,
        );
        assert_eq!(
            a.common_cpu_bytes,
            deepseek41_header_memory_bytes(&ordinary_header) + host_bytes,
        );
        // Adding disk-only descriptors charges only their owned header metadata,
        // not banks/tables or another copy of the model-host allowance.
        assert_eq!(
            b.common_cpu_bytes - a.common_cpu_bytes,
            deepseek41_header_memory_bytes(&full_header)
                - deepseek41_header_memory_bytes(&ordinary_header),
        );
        assert!(host_bytes >= 578_560);
        assert_eq!(
            b.stage_memory_components(0, 40, 40).unwrap().1,
            deepseek41_header_memory_bytes(&full_header) + host_bytes,
        );
        assert_eq!(b.layer_cpu_bytes, vec![0; 40]); // unified RAM is not counted twice
        for boundary in [
            b.head_embedding_gpu_bytes,
            b.tail_embedding_gpu_bytes,
            b.head_embedding_cpu_bytes,
            b.tail_embedding_cpu_bytes,
            b.head_embedding_cpu_transient_bytes,
            b.tail_embedding_cpu_transient_bytes,
        ] {
            assert_eq!(boundary, Some(0));
        }
        // Same descriptor ownership/capacities, larger disk-only payloads and
        // gaps: retained CPU/GPU accounting must stay identical.
        let mut relocated_header = full_header;
        relocated_header.tensor_data_offset += 1 << 20;
        for (name, info) in &mut relocated_header.tensor_infos {
            if name.ends_with("_exps.weight") || name.ends_with(".engram_embd.weight") {
                info.shape[0] *= 2;
                info.offset += 1 << 39;
            }
        }
        let relocated =
            deepseek41_memory_profile(&relocated_header, "relocated", &m, &Device::Cpu)?;
        assert_eq!(
            b.stage_memory_components(0, 40, 40),
            relocated.stage_memory_components(0, 40, 40),
        );
        assert!(b.stage_memory_components(0, 41, 40).is_none());
        assert!(b.stage_memory_components(0, 40, 41).is_none());
        Ok(())
    }

    #[test]
    fn deepseek41_profile_rejects_invalid_layer_counts_and_descriptor_indices() -> Result<()> {
        use crate::deepseek4_gguf::fixtures::{maya_metadata, sparse};
        let (mut file, _) = sparse(
            &maya_metadata(true),
            &[("blk.40.attn_norm.weight", &[256], 0, 0)],
        );
        let content = RawGgufContent::read(file.as_file_mut())?;
        let mut metadata = parse_gguf_metadata(&content.metadata)?;
        assert!(deepseek41_memory_profile(&content, "bad-index", &metadata, &Device::Cpu).is_err());
        for count in [0, usize::MAX] {
            metadata.num_layers = count;
            assert!(
                deepseek41_memory_profile(&content, "bad-count", &metadata, &Device::Cpu).is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn deepseek41_exact_source_rejects_oversized_norm_and_unknown_resident() -> Result<()> {
        use crate::architectures::deepseek41::DeepSeek41Stage;
        use crate::deepseek4_gguf::fixtures::{maya_metadata, sparse};
        use crate::deepseek4_gguf::{DeepSeek4DType as T, RawTensorInfo};
        use gguf_file::Value as V;

        // Descriptor-only fixture: no resident tensor allocation or bank read.
        // A valid baseline ensures each negative case fails for its mutation,
        // rather than for a missing unrelated required descriptor.
        let (mut file, _) = sparse(&maya_metadata(false), &[]);
        let mut content = RawGgufContent::read(file.as_file_mut())?;
        let mut add = |name: String, dtype: T, shape: &[usize]| {
            content.tensor_infos.insert(
                name,
                RawTensorInfo {
                    dtype,
                    shape: shape.to_vec(),
                    offset: 0,
                },
            );
        };
        add("token_embd.weight".into(), T::Q8_0, &[129280, 5120]);
        add("output.weight".into(), T::Q8_0, &[129280, 5120]);
        add("output_norm.weight".into(), T::F32, &[5120]);
        for l in 0..40 {
            let p = format!("blk.{l}.");
            for side in ["attn", "ffn"] {
                add(format!("{p}hc_{side}_fn.weight"), T::F16, &[24, 20480]);
                add(format!("{p}hc_{side}_scale.weight"), T::F32, &[3]);
                add(format!("{p}hc_{side}_base.weight"), T::F32, &[24]);
                add(format!("{p}{side}_norm.weight"), T::F32, &[5120]);
            }
            for (name, shape) in [
                ("attn_q_a.weight", [1280, 5120]),
                ("attn_q_b.weight", [32768, 1280]),
                ("attn_kv.weight", [512, 5120]),
                ("attn_output_a.weight", [8192, 4096]),
                ("attn_output_b.weight", [5120, 8192]),
                ("ffn_gate_shexp.weight", [2304, 5120]),
                ("ffn_up_shexp.weight", [2304, 5120]),
                ("ffn_down_shexp.weight", [5120, 2304]),
            ] {
                add(format!("{p}{name}"), T::Q8_0, &shape);
            }
            for (name, width) in [
                ("attn_q_a_norm.weight", 1280),
                ("attn_kv_a_norm.weight", 512),
                ("attn_sinks.weight", 64),
                ("exp_probs_b.bias", 384),
            ] {
                add(format!("{p}{name}"), T::F32, &[width]);
            }
            add(format!("{p}ffn_gate_inp.weight"), T::F32, &[384, 5120]);
            for part in ["gate", "up", "down"] {
                let shape = if part == "down" {
                    [384, 5120, 2304]
                } else {
                    [384, 2304, 5120]
                };
                add(format!("{p}ffn_{part}_exps.weight"), T::IQ1S, &shape);
            }
            if [2, 8, 14, 20].contains(&l) {
                add(
                    format!("{p}attn_compressor_kv.weight"),
                    T::Q8_0,
                    &[512, 5120],
                );
                add(format!("{p}attn_compressor_norm.weight"), T::F32, &[512]);
                add(format!("{p}indexer.attn_k.weight"), T::F16, &[128, 512]);
                add(format!("{p}indexer.k_norm.weight"), T::F32, &[128]);
                if l != 20 {
                    add(
                        format!("{p}attn_compressor_gate.weight"),
                        T::Q8_0,
                        &[512, 5120],
                    );
                }
            }
            if [2, 8, 14, 20, 24, 28, 32, 36].contains(&l) {
                add(
                    format!("{p}indexer.attn_q_b.weight"),
                    T::Q8_0,
                    &[4096, 1280],
                );
                add(format!("{p}indexer.proj.weight"), T::F16, &[32, 5120]);
            }
            let rows = match l {
                1 => Some(384006168),
                14 => Some(384016682),
                _ => None,
            };
            if let Some(rows) = rows {
                add(format!("{p}engram_embd.weight"), T::I8, &[rows, 264]);
                add(format!("{p}engram_kv.weight"), T::Q8_0, &[25600, 6144]);
                add(format!("{p}engram_q_norm.weight"), T::F32, &[4, 5120]);
                add(format!("{p}engram_k_norm.weight"), T::F32, &[4, 5120]);
            }
        }
        let primes = [
            16000057, 16000079, 16000081, 16000097, 16000121, 16000129, 16000133, 16000183,
            16000189, 16000207, 16000211, 16000253, 16000277, 16000289, 16000307, 16000321,
            16000339, 16000381, 16000393, 16000399, 16000403, 16000409, 16000447, 16000463,
            16000477, 16000487, 16000499, 16000507, 16000511, 16000573, 16000609, 16000627,
            16000667, 16000669, 16000693, 16000697, 16000711, 16000729, 16000759, 16000769,
            16000781, 16000799, 16000813, 16000819, 16000841, 16000877, 16000879, 16000889,
        ];
        for (key, value) in [
            ("engram.compressed_vocab_size", V::U32(99092)),
            ("engram.pad_id", V::U32(2)),
            (
                "engram.token_map",
                V::Array((0..129280).map(|id| V::U32(id % 99092)).collect()),
            ),
            (
                "engram.primes",
                V::Array(primes.into_iter().map(V::U32).collect()),
            ),
            ("engram.multipliers", V::Array(vec![V::U64(1); 8])),
        ] {
            content.metadata.insert(format!("deepseek41.{key}"), value);
        }
        DeepSeek41Stage::validate_source(&content)?;
        content
            .tensor_infos
            .get_mut("output_norm.weight")
            .unwrap()
            .shape = vec![5120, 1_000_000];
        assert!(DeepSeek41Stage::validate_source(&content).is_err());
        content
            .tensor_infos
            .get_mut("output_norm.weight")
            .unwrap()
            .shape = vec![5120];
        content.tensor_infos.insert(
            "unknown_resident.weight".into(),
            RawTensorInfo {
                dtype: T::F32,
                shape: vec![5120],
                offset: 0,
            },
        );
        assert!(DeepSeek41Stage::validate_source(&content).is_err());
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn deepseek41_metal_loader_rejects_unpinned_artifacts_before_resident_upload() -> Result<()> {
        use crate::deepseek4_gguf::fixtures::{maya_metadata, sparse};
        let device = Device::new_metal(0)?;
        for (name, shape) in [
            ("output_norm.weight", vec![5120, 1_000_000]),
            ("unknown_resident.weight", vec![5120]),
        ] {
            let (file, _) = sparse(&maya_metadata(false), &[(name, &shape, 0, 0)]);
            assert!(load_stage_tensors(file.path(), LayerRange::solo(40), &device).is_err());
        }
        Ok(())
    }
}

#[cfg(test)]
mod maya_identity_cpu_tests {
    use super::*;
    use crate::deepseek4_gguf::fixtures::{maya_metadata, sha_passes, sparse, verified_source};
    use sha2::{Digest, Sha256};
    use std::fs::File;
    use std::io::{Seek, SeekFrom, Write};

    fn fixture() -> Result<(tempfile::NamedTempFile, u64, Vec<u8>)> {
        let mut md = maya_metadata(false);
        md.insert(
            "tokenizer.ggml.tokens".into(),
            gguf_file::Value::Array(vec![
                gguf_file::Value::String("original".into()),
                gguf_file::Value::String("other".into()),
            ]),
        );
        let (mut file, data) = sparse(&md, &[("output_norm.weight", &[2], 0, 0)]);
        let payload = [1f32.to_le_bytes(), 2f32.to_le_bytes()].concat();
        file.as_file_mut().seek(SeekFrom::Start(data))?;
        file.write_all(&payload)?;
        let bytes = std::fs::read(file.path())?;
        Ok((file, data, bytes))
    }

    #[test]
    fn expected_identity_precedes_corrupt_and_other_architecture_headers_maya_identity_cpu()
    -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("launch.gguf");
        let before = sha_passes();
        for corrupt in [false, true] {
            if corrupt {
                std::fs::write(&path, b"not a GGUF header")?;
            } else {
                let md = [(
                    "general.architecture",
                    gguf_file::Value::String("llama".into()),
                )];
                let values = md
                    .iter()
                    .map(|(key, value)| (*key, value))
                    .collect::<Vec<_>>();
                gguf_file::write(&mut File::create(&path)?, &values, &[])?;
                assert!(
                    matches!(read_gguf_metadata_map(&path)?.get("general.architecture"),
                    Some(gguf_file::Value::String(architecture)) if architecture == "llama")
                );
            }
            let error = PreparedNativeSource::open(&path, ExpectedNativeIdentity::Maya)
                .err()
                .expect("expected Maya must not dispatch by the header");
            assert!(
                error.to_string().contains("Maya artifact length mismatch"),
                "{error:#}"
            );
        }
        assert_eq!(
            sha_passes(),
            before,
            "wrong length must not scan an artifact"
        );
        Ok(())
    }

    #[test]
    fn injected_digest_precedes_header_parsing_maya_identity_cpu() -> Result<()> {
        let mut file = tempfile::NamedTempFile::new()?;
        file.write_all(b"corrupt header")?;
        let error = verified_source(File::open(file.path())?, 14, &[0; 32])
            .err()
            .expect("bad SHA must fail before parsing corrupt bytes");
        assert!(
            error.to_string().contains("Maya artifact SHA256 mismatch"),
            "{error:#}"
        );
        Ok(())
    }

    #[test]
    fn unverified_source_rejected_at_all_production_gates_maya_identity_cpu() -> Result<()> {
        let (file, _, _) = fixture()?;
        let source = Arc::new(RawGgufSource::new(File::open(file.path())?)?);
        let error = PreparedNativeSource::from_source(Arc::clone(&source))
            .err()
            .expect("unverified source must not prepare");
        assert!(
            error
                .to_string()
                .contains("lacks pinned artifact SHA256 proof")
        );
        // Even an internal wrapper cannot bypass the independently checked gates.
        let prepared = PreparedNativeSource {
            metadata: parse_gguf_metadata(&source.content().metadata)?,
            source,
        };
        let errors = [
            prepared.metadata().err(),
            prepared.tokenizer().err(),
            prepared
                .memory_profile_components("fixture", &Device::Cpu)
                .err(),
            prepared
                .load_stage_tensors(LayerRange::solo(40), &Device::Cpu)
                .err(),
            crate::PipelineStage::from_prepared_native(
                &prepared,
                LayerRange::solo(40),
                &Device::Cpu,
                None,
            )
            .err(),
        ];
        for error in errors {
            assert!(
                error
                    .expect("production gate must reject unverified source")
                    .to_string()
                    .contains("lacks pinned artifact SHA256 proof")
            );
        }
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn retained_tokenizer_profile_and_tensor_reuse_without_rehash_maya_identity_cpu() -> Result<()>
    {
        use std::os::unix::fs::symlink;
        let (file, _, bytes) = fixture()?;
        let dir = tempfile::tempdir()?;
        let launch = dir.path().join("launch.gguf");
        let replacement = dir.path().join("replacement.gguf");
        symlink(file.path(), &launch)?;
        let before = sha_passes();
        let sha: [u8; 32] = Sha256::digest(&bytes).into();
        let prepared = PreparedNativeSource::from_source(Arc::new(verified_source(
            File::open(&launch)?,
            bytes.len() as u64,
            &sha,
        )?))?;
        let profile = prepared
            .memory_profile_components("fixture", &Device::Cpu)?
            .stage_memory_components(0, 40, 40);
        std::fs::write(
            &replacement,
            b"replacement cannot supply a header or tokenizer",
        )?;
        std::fs::remove_file(&launch)?;
        symlink(&replacement, &launch)?;
        for _ in 0..2 {
            assert_eq!(prepared.metadata()?.architecture, "deepseek41");
            assert_eq!(prepared.metadata()?.num_layers, 40);
            assert_eq!(prepared.tokenizer()?.decode(&[0]), "original");
            assert_eq!(
                prepared
                    .memory_profile_components("fixture", &Device::Cpu)?
                    .stage_memory_components(0, 40, 40),
                profile
            );
            let tensor = prepared
                .source
                .load_resident_tensor("output_norm.weight", &Device::Cpu)?;
            assert_eq!(
                tensor.bytes().to_vec1::<u8>()?,
                [1f32.to_le_bytes(), 2f32.to_le_bytes()].concat()
            );
            let error = crate::PipelineStage::from_prepared_native(
                &prepared,
                LayerRange::solo(40),
                &Device::Cpu,
                None,
            )
            .err()
            .expect("Maya graph must reject CPU before allocation");
            assert!(error.to_string().contains("Metal"), "{error:#}");
        }
        assert_eq!(
            sha_passes(),
            before + 1,
            "prepared reuse must not perform another full SHA"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn ordinary_path_replacement_fails_closed_on_source_identity_change_maya_identity_cpu()
    -> Result<()> {
        let (file, _, bytes) = fixture()?;
        let sha: [u8; 32] = Sha256::digest(&bytes).into();
        let prepared = PreparedNativeSource::from_source(Arc::new(verified_source(
            File::open(file.path())?,
            bytes.len() as u64,
            &sha,
        )?))?;
        let replacement = tempfile::NamedTempFile::new()?;
        std::fs::rename(replacement.path(), file.path())?;
        for error in [
            prepared.metadata().err(),
            prepared.tokenizer().err(),
            prepared
                .memory_profile_components("fixture", &Device::Cpu)
                .err(),
            prepared
                .source
                .read_tensor_bytes("output_norm.weight", 0, 8)
                .err(),
        ] {
            assert!(
                error
                    .expect("unlinked retained inode must fail identity checks")
                    .to_string()
                    .contains("source identity/length/timestamps changed")
            );
        }
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn maya_cached_prepared_cpu() -> Result<()> {
        use crate::deepseek4_gguf::fixtures::cached_source;
        let (_fixture, _, bytes) = fixture()?;
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("prepared.gguf");
        std::fs::write(&path, &bytes)?;
        let sha: [u8; 32] = Sha256::digest(&bytes).into();
        let before = sha_passes();
        let cold = PreparedNativeSource::from_source(Arc::new(cached_source(
            &path,
            bytes.len() as u64,
            sha,
        )?))?;
        assert_eq!(
            cold.source.verification().unwrap().full_read_bytes,
            bytes.len() as u64
        );
        for _ in 0..2 {
            let warm = PreparedNativeSource::from_source(Arc::new(cached_source(
                &path,
                bytes.len() as u64,
                sha,
            )?))?;
            let stats = warm.source.verification().unwrap();
            assert!(stats.sample_hit);
            assert_eq!((stats.full_passes, stats.full_read_bytes), (0, 0));
            assert_eq!(stats.sample_bytes, bytes.len() as u64);
            assert_eq!(warm.source.file_identity(), cold.source.file_identity());
            assert_eq!(
                format!("{:?}", warm.metadata()?),
                format!("{:?}", cold.metadata()?)
            );
            assert_eq!(warm.tokenizer()?.decode(&[0]), "original");
            assert_eq!(
                warm.source.read_tensor_bytes("output_norm.weight", 0, 8)?,
                [1f32.to_le_bytes(), 2f32.to_le_bytes()].concat()
            );
        }
        assert_eq!(sha_passes(), before + 1);
        // This cfg(test) pin receipt cannot satisfy the production constructor.
        assert!(PreparedNativeSource::open(&path, ExpectedNativeIdentity::Maya).is_err());
        expected_identity_precedes_corrupt_and_other_architecture_headers_maya_identity_cpu()?;
        unverified_source_rejected_at_all_production_gates_maya_identity_cpu()?;
        retained_tokenizer_profile_and_tensor_reuse_without_rehash_maya_identity_cpu()?;
        ordinary_path_replacement_fails_closed_on_source_identity_change_maya_identity_cpu()?;
        eprintln!(
            "prepared CPU fixture: one full pass/{} bytes, two warm hits/0 full bytes",
            bytes.len()
        );
        Ok(())
    }
}
