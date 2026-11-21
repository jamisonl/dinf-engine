#![warn(missing_docs)]

//! Embeddable GGUF inference on Candle.
//!
//! The root API provides model loading and text generation. The public modules
//! expose the lower-level stage, cache, tokenizer, and sampling types needed
//! to split a model across optional pipeline stages.
//!
//! # Example
//!
//! ```no_run
//! use candle_core::Device;
//! use dinf_engine::{Engine, GenerationParams};
//!
//! # fn main() -> anyhow::Result<()> {
//! let mut engine = Engine::load("model.gguf", Device::Cpu)?;
//! let output = engine.generate(
//!     "Hello",
//!     &GenerationParams {
//!         max_tokens: 32,
//!         temperature: 0.0,
//!         ..GenerationParams::default()
//!     },
//! )?;
//! println!("{}", output.text);
//! # Ok(())
//! # }
//! ```

mod api;
/// Model architecture implementations and shared stage/cache primitives.
/// Prompt tokens processed per forward pass when a caller does not bound
/// prefill. Matches llama.cpp's default micro-batch: large enough to keep the
/// quantized matmul tiles full, small enough that one pass's temporary
/// buffers stay bounded (Metal releases them only at a device wait).
pub(crate) const DEFAULT_PREFILL_CHUNK_TOKENS: usize = 512;

pub mod architectures;
/// Continuous-batching and speculative-decoding session helpers.
pub mod continuous;
/// Graph-captured single-token decode steps on CUDA.
#[cfg(feature = "cuda")]
pub mod cuda_graph;

/// Mark a layer boundary in a decode step. While the step is being recorded
/// as CUDA graphs, this is where a recorded segment is launched.
#[inline]
pub(crate) fn decode_layer_boundary(layer_index: usize) {
    #[cfg(feature = "cuda")]
    cuda_graph::segment_boundary(layer_index);
    #[cfg(not(feature = "cuda"))]
    let _ = layer_index;
}
pub(crate) mod deepseek4_gguf;
mod device_stages;
/// Native image diffusion, independent of autoregressive generation sessions.
pub mod diffusion;
mod gguf_limits;
/// GGUF metadata parsing, stage loading, and memory estimation.
pub mod gguf_loader;
pub(crate) mod gsq_coder;
/// Inference throughput and latency aggregation.
pub mod metrics;
/// Runtime options for a loaded stage.
pub mod options;
mod prompt_cache;
mod qmatmul;
/// Token sampling controls and deterministic request-scoped random sampling.
pub mod sampling;
#[cfg(feature = "cuda")]
mod sampling_cuda;
#[cfg(target_os = "macos")]
mod sampling_metal;
mod speculative_batch;
/// GGUF tokenization, chat templates, and streamed text decoding.
pub mod tokenizer;
/// Request-owned constraints for complete, model-native tool calls.
pub mod tool_constraints;
/// TurboQuant-compressed key/value cache support.
pub mod turboquant;
#[cfg(target_os = "macos")]
mod turboquant_metal;
/// Pinned GGUF artifact verification and sampled file fingerprinting.
pub mod verified_file;
/// Vision projector loading and image embedding support.
pub mod vision;

pub use api::{Engine, EngineBuilder, GenerationOutput};
pub use gguf_loader::{GgufMetadata, LayerRange, StageMemoryProfile, read_gguf_metadata};
pub use options::{EngineOptions, ProfileOptions, SpeculativeOptions};
pub use prompt_cache::RESERVED_PROMPT_SCOPE_PREFIX;
pub use sampling::SamplingParams as GenerationParams;

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use candle_core::Device;
use sha2::{Digest, Sha256};

fn hash_cache_token_ids(tokens: &[u32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"dinf-cache-fingerprint-v1");
    hasher.update((tokens.len() as u64).to_be_bytes());
    for token in tokens {
        hasher.update(token.to_be_bytes());
    }
    hasher.finalize().into()
}

/// Run tests of process-wide counters in a fresh test process. A mutex among
/// the observing tests cannot exclude other tests that change those counters.
#[cfg(test)]
pub(crate) fn test_in_subprocess(name: &str) -> bool {
    let name = name
        .strip_prefix(concat!(module_path!(), "::"))
        .unwrap_or(name);
    const CHILD: &str = "DINF_ENGINE_ISOLATED_TEST";
    if std::env::var(CHILD).as_deref() == Ok(name) {
        return false;
    }
    let output = std::process::Command::new(std::env::current_exe().expect("test executable"))
        .args(["--exact", name, "--nocapture"])
        .env(CHILD, name)
        .output()
        .expect("start isolated test");
    assert!(
        output.status.success(),
        "isolated test {name} failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("1 passed"),
        "isolated test {name} did not run:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    print!("{}", String::from_utf8_lossy(&output.stdout));
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    true
}

/// Metal tests skip on platforms without Metal.
#[cfg(all(test, not(target_os = "macos")))]
pub(crate) fn test_metal_device() -> Option<Device> {
    None
}

#[cfg(all(test, target_os = "macos"))]
pub(crate) fn test_metal_device() -> Option<Device> {
    let required = std::env::var_os("DINF_REQUIRE_METAL_TESTS").is_some();
    if candle_metal_kernels::metal::Device::all().is_empty() {
        assert!(
            !required,
            "DINF_REQUIRE_METAL_TESTS is set, but no Metal device is available"
        );
        static SKIP_NOTICE: std::sync::Once = std::sync::Once::new();
        SKIP_NOTICE.call_once(|| {
            eprintln!("Metal device unavailable; Metal correctness tests are being skipped");
        });
        None
    } else {
        match Device::new_metal(0) {
            Ok(device) => Some(device),
            Err(error) if required => {
                panic!("DINF_REQUIRE_METAL_TESTS is set, but Metal initialization failed: {error}")
            }
            Err(error) => {
                static SKIP_NOTICE: std::sync::Once = std::sync::Once::new();
                SKIP_NOTICE.call_once(|| {
                    eprintln!("Metal initialization failed; Metal tests are skipped: {error}");
                });
                None
            }
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
pub(crate) fn test_cuda_device() -> Option<Device> {
    let required = std::env::var_os("DINF_REQUIRE_CUDA_TESTS").is_some();
    match Device::new_cuda(0) {
        Ok(device) => Some(device),
        Err(error) if required => {
            panic!("DINF_REQUIRE_CUDA_TESTS is set, but CUDA initialization failed: {error}")
        }
        Err(error) => {
            static SKIP_NOTICE: std::sync::Once = std::sync::Once::new();
            SKIP_NOTICE.call_once(|| {
                eprintln!(
                    "CUDA device unavailable; CUDA correctness tests are being skipped: {error}"
                );
            });
            None
        }
    }
}

use architectures::{
    AttentionSpan, LayerCache, ModelArchitecture, MropePositionIds, StageInput, StageOutput,
    TextArithmeticPolicy, TextRequestContext,
};
use sampling::{SamplingHistory, SamplingParams, SamplingRng};
use turboquant::TurboQuantConfig;

struct EmbeddedPrompt {
    embeddings: candle_core::Tensor,
    bidirectional_spans: Vec<AttentionSpan>,
    positions: Option<MropePositionIds>,
}

#[derive(Clone, Copy)]
enum ForwardPass<'a> {
    Prefill,
    PrefillCacheOnly,
    TailPrefillCacheOnly,
    Text {
        context: TextRequestContext,
        prefill: bool,
        cache_only: bool,
        tail_fence: bool,
    },
    Standard {
        layer_ids: &'a [usize],
        all_logits: bool,
    },
}

impl ForwardPass<'_> {
    fn run(
        self,
        arch: &dyn ModelArchitecture,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<(StageOutput, Vec<candle_core::Tensor>)> {
        let run = || match self {
            Self::Prefill => arch
                .forward_prefill(input, cache, seq_len, start_pos)
                .map(|output| (output, Vec::new())),
            Self::PrefillCacheOnly | Self::TailPrefillCacheOnly => arch
                .forward_prefill_cache_only(input, cache, seq_len, start_pos)
                .map(|output| (output, Vec::new())),
            Self::Text {
                context,
                prefill,
                cache_only,
                ..
            } => arch
                .forward_text(
                    input, cache, seq_len, start_pos, context, prefill, cache_only,
                )
                .map(|output| (output, Vec::new())),
            Self::Standard {
                layer_ids,
                all_logits,
            } => arch
                .forward_with_layer_inputs(input, cache, seq_len, start_pos, layer_ids, all_logits),
        };
        // Quantized Metal layers validate together after the forward pass.
        // Any failed check reaches the existing stage rollback/clear path;
        // no output or cache accounting is committed before validation.
        #[cfg(target_os = "macos")]
        {
            crate::turboquant_metal::with_deferred_validation(run)
        }
        #[cfg(not(target_os = "macos"))]
        {
            run()
        }
    }

    fn finish(self, device: &Device) -> Result<()> {
        if matches!(
            self,
            Self::TailPrefillCacheOnly
                | Self::Text {
                    tail_fence: true,
                    ..
                }
        ) {
            // The caller will send a progress acknowledgement without reading
            // scores. Complete cache writes and surface device errors first.
            device
                .synchronize()
                .context("completing tail cache-only prefill")?;
        }
        Ok(())
    }
}

#[derive(Default)]
struct DecodeProfile {
    forward: Duration,
    penalty: Duration,
    sampling: Duration,
    callback: Duration,
    tokens: usize,
}

static NEXT_SESSION_CACHE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Default)]
struct KvTokenBudget {
    retained: AtomicUsize,
    allocated: AtomicUsize,
}

impl KvTokenBudget {
    fn total(&self) -> usize {
        self.retained.load(Ordering::Acquire)
    }

    fn add(&self, count: usize) {
        self.retained
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |total| {
                total.checked_add(count)
            })
            .expect("KV token budget overflow");
    }

    fn release(&self, count: usize) {
        if count == 0 {
            return;
        }
        self.retained
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |total| {
                total.checked_sub(count)
            })
            .expect("KV token budget underflow");
    }

    fn replace(&self, previous: usize, next: usize) {
        if next >= previous {
            self.add(next - previous);
        } else {
            self.release(previous - next);
        }
    }

    fn allocated(&self) -> usize {
        self.allocated.load(Ordering::Acquire)
    }

    fn add_allocated(&self, count: usize) {
        self.allocated
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |total| {
                total.checked_add(count)
            })
            .expect("KV allocation budget overflow");
    }

    fn release_allocated(&self, count: usize) {
        if count == 0 {
            return;
        }
        self.allocated
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |total| {
                total.checked_sub(count)
            })
            .expect("KV allocation budget underflow");
    }

    fn replace_allocated(&self, previous: usize, next: usize) {
        if next >= previous {
            self.add_allocated(next - previous);
        } else {
            self.release_allocated(previous - next);
        }
    }
}

struct PersistentDecodeBatch {
    layer_caches: Vec<Option<LayerCache>>,
    revisions: Vec<u64>,
}

impl DecodeProfile {
    fn report(&self, synchronized: bool) {
        let n = self.tokens.max(1) as f64;
        eprintln!(
            "DINF_DECODE_PROFILE tokens={} sync_after_forward={} forward_ms={:.3} penalty_ms={:.3} sampling_ms={:.3} callback_ms={:.3} per_token_ms={{forward:{:.3},penalty:{:.3},sampling:{:.3},callback:{:.3}}}",
            self.tokens,
            synchronized,
            self.forward.as_secs_f64() * 1e3,
            self.penalty.as_secs_f64() * 1e3,
            self.sampling.as_secs_f64() * 1e3,
            self.callback.as_secs_f64() * 1e3,
            self.forward.as_secs_f64() * 1e3 / n,
            self.penalty.as_secs_f64() * 1e3 / n,
            self.sampling.as_secs_f64() * 1e3 / n,
            self.callback.as_secs_f64() * 1e3 / n,
        );
    }
}

/// Position of a stage within the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StagePosition {
    /// Single stage, all layers.
    Solo,
    /// First stage: has embedding, sends activations downstream.
    Head,
    /// Middle stage: receives and sends activations.
    Middle,
    /// Last stage: has lm_head, produces tokens.
    Tail,
}

enum DecodeBatchInputs<'a> {
    Solo(&'a [u32]),
    Pipeline(&'a [StageInput]),
}

fn resolve_speculative_block_size(
    trained_rows: usize,
    preferred_rows: usize,
    max_draft_tokens: Option<usize>,
) -> Result<usize> {
    anyhow::ensure!(
        trained_rows >= 2,
        "DFlash block must include an anchor and a draft token"
    );
    let rows = preferred_rows.clamp(2, trained_rows);
    match max_draft_tokens {
        None => Ok(rows),
        Some(max) => {
            anyhow::ensure!(max > 0, "DFlash draft token limit must be positive");
            Ok(rows.min(
                max.checked_add(1)
                    .context("DFlash draft token limit overflow")?,
            ))
        }
    }
}

/// A pipeline stage wrapping a model architecture with its layer range.
pub struct PipelineStage {
    arch: Box<dyn ModelArchitecture>,
    speculative: Option<architectures::dflash2::DFlash2Draft>,
    speculative_cache: Option<architectures::dflash2::DFlash2Cache>,
    speculative_draft_max_tokens: Option<usize>,
    position: StagePosition,
    cache: Vec<Option<LayerCache>>,
    hidden_dim: usize,
    vocab_size: usize,
    context_length: usize,
    device: Device,
    /// KV cache cost per token for this stage's layer range (bytes).
    kv_bytes_per_token: u64,
    /// Peak device bytes per token used to size the shared KV budget.
    ///
    /// Includes packed-cache overlap and dense attention reconstructions;
    /// native Metal reconstructs F16 while portable paths reconstruct F32.
    kv_budget_bytes_per_token: u64,
    fixed_cache_bytes: usize,
    /// Absolute token position for the next forward pass.
    ///
    /// This keeps increasing when rolling eviction drops old cache entries so
    /// RoPE never reuses a position.
    next_position: usize,
    /// Rotary offset after positioned image inputs, including public prefill.
    mrope_decode_delta: Option<i64>,
    text_arithmetic: Option<TextArithmeticPolicy>,
    /// Number of token positions retained in the rolling cache.
    kv_cache_tokens: usize,
    /// Token IDs represented by the reusable, non-evicted stage cache.
    ///
    /// A caller-provided prefix length is only an optimization hint.  We
    /// must also prove that the local cache contains the same token prefix
    /// before skipping prefill.  This is deliberately kept separate from the
    /// prompt cache: the latter owns forkable session snapshots, while this
    /// identity describes the single cache owned by this stage.
    cache_token_ids: Vec<u32>,
    /// Digest of `cache_token_ids`, used to compare cache identity between
    /// pipeline stages.
    ///
    /// A length alone cannot distinguish two prompts that happen to have the
    /// same retained-token count.  Callers compare digests before computing
    /// any activation for a request that reuses the cache.
    cache_fingerprint: Option<[u8; 32]>,
    /// Sequence capacity available to this lane before its cache must grow.
    kv_cache_lane_capacity_tokens: usize,
    /// Conservative token-slot width of the live cache allocation.
    kv_cache_allocation_tokens: usize,
    fixed_cache_allocation_tokens: usize,
    /// TurboQuant config (None = f16 KV cache).
    #[allow(dead_code)]
    quant_config: Option<Arc<TurboQuantConfig>>,
    /// Maximum KV cache token capacity. When set, evicts oldest tokens
    /// before forward() would exceed this limit.
    max_kv_tokens: Option<usize>,
    /// Shared retained-token count for the stage cache and every session cache
    /// created from this stage.
    kv_token_budget: Arc<KvTokenBudget>,
    /// Maximum resident KV token slots across the stage and all session caches.
    ///
    /// The caller-provided headroom is split in half: one half bounds resident
    /// cache allocations and one half remains free for copy-on-grow and batch
    /// packing, where old and new buffers overlap for one forward pass.
    max_total_kv_tokens: Option<usize>,
    /// Full allocation headroom before the copy reserve is applied.
    max_total_kv_headroom_tokens: Option<usize>,
    /// Dense F16 caches use the multi-lane preallocated accounting path.
    /// Quantized caches have their own capacity-aware admission policy.
    preallocated_kv_cache: bool,
    /// Lookup only: sessions own packed KV tensors so dropping a group frees
    /// its storage along with the memory budget charged to those sessions.
    decode_batch_caches: HashMap<Vec<u64>, Weak<PersistentDecodeBatch>>,
    prompt_cache: prompt_cache::PromptCache,
    options: EngineOptions,
    /// State for graph-captured single-token decode steps.
    #[cfg(feature = "cuda")]
    cuda_decode_graph: cuda_graph::DecodeGraph,
    /// This stage's count among the stages on its GPU. Declared last so it
    /// is dropped after the stage's device buffers.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    device_stages: device_stages::StageRegistration,
}

/// Mutable inference state owned by one logical generation session.
///
/// Model weights remain in [`PipelineStage`], while KV/recurrent caches live in
/// this value.  Keeping the two separate lets a scheduler interleave multiple
/// sessions on one loaded model without allowing their cache state to mix.
pub struct PipelineStageSessionState {
    cache: Vec<Option<LayerCache>>,
    speculative_cache: Option<architectures::dflash2::DFlash2Cache>,
    next_position: usize,
    kv_cache_tokens: usize,
    kv_cache_lane_capacity_tokens: usize,
    kv_cache_allocation_tokens: usize,
    fixed_cache_allocation_tokens: usize,
    kv_token_budget: Arc<KvTokenBudget>,
    decode_batch_cache: Option<Arc<PersistentDecodeBatch>>,
    cache_id: u64,
    revision: u64,
    // Qwen image prompts can use rotary positions below the logical count.
    // Carry that offset with the cache when a new text turn resumes it.
    mrope_decode_delta: Option<i64>,
    text_arithmetic: Option<TextArithmeticPolicy>,
    // External embeddings are not identified by prompt token IDs. Preserve
    // that fact across turns so text-only prompt reuse cannot alias this state.
    has_embedded_prompt: bool,
    // A text stop discarded pending speculative output. The next generation
    // must rebuild its full visible prompt instead of reusing lookahead state.
    resume_invalidated: bool,
}

struct PipelineStageSessionCheckpoint {
    cache: Vec<Option<LayerCache>>,
    next_position: usize,
    kv_cache_tokens: usize,
    kv_cache_lane_capacity_tokens: usize,
    kv_cache_allocation_tokens: usize,
    fixed_cache_allocation_tokens: usize,
    mrope_decode_delta: Option<i64>,
    text_arithmetic: Option<TextArithmeticPolicy>,
    has_embedded_prompt: bool,
    resume_invalidated: bool,
}

/// One target verification pass held until the caller decides how much of the
/// proposed prefix to commit.
pub(crate) struct SpeculativeVerification {
    pub(crate) proposed: Vec<u32>,
    pub(crate) logits: candle_core::Tensor,
    pub(crate) draft_time: Duration,
    pub(crate) target_started: Instant,
    inputs: Vec<u32>,
    captures: Vec<candle_core::Tensor>,
    checkpoint: PipelineStageSessionCheckpoint,
}

impl PipelineStageSessionState {
    fn clear(&mut self) {
        self.decode_batch_cache = None;
        for entry in &mut self.cache {
            *entry = None;
        }
        if let Some(cache) = self.speculative_cache.as_mut() {
            cache.clear();
        }
        self.kv_token_budget.release(self.kv_cache_tokens);
        self.kv_token_budget.release_allocated(
            self.kv_cache_allocation_tokens + self.fixed_cache_allocation_tokens,
        );
        self.next_position = 0;
        self.kv_cache_tokens = 0;
        self.kv_cache_lane_capacity_tokens = 0;
        self.kv_cache_allocation_tokens = 0;
        self.fixed_cache_allocation_tokens = 0;
        self.mrope_decode_delta = None;
        self.text_arithmetic = None;
        self.has_embedded_prompt = false;
        self.resume_invalidated = false;
        self.revision = self.revision.wrapping_add(1);
    }
}

impl Drop for PipelineStageSessionState {
    fn drop(&mut self) {
        self.kv_token_budget.release(self.kv_cache_tokens);
        self.kv_token_budget.release_allocated(
            self.kv_cache_allocation_tokens + self.fixed_cache_allocation_tokens,
        );
    }
}

impl Drop for PipelineStage {
    fn drop(&mut self) {
        self.kv_token_budget.release(self.kv_cache_tokens);
        self.kv_token_budget.release_allocated(
            self.kv_cache_allocation_tokens + self.fixed_cache_allocation_tokens,
        );
    }
}

impl PipelineStage {
    /// Load a pipeline stage from a GGUF file for a given layer range.
    ///
    /// `kv_quant_bits` enables the quality-qualified 8-bit TurboQuant profile.
    /// Other bit widths and unsupported head geometry fail explicitly; a
    /// requested quantization mode is never silently replaced with F16.
    pub fn from_gguf(
        gguf_path: &Path,
        range: LayerRange,
        device: &Device,
        kv_quant_bits: Option<u8>,
    ) -> Result<Self> {
        Self::from_gguf_source(Some(gguf_path), None, range, device, kv_quant_bits)
    }

    /// Load only from the retained, canonically verified native descriptor.
    /// The caller must admit its memory profile before resident allocation.
    pub fn from_prepared_native(
        source: &gguf_loader::PreparedNativeSource,
        range: LayerRange,
        device: &Device,
        kv_quant_bits: Option<u8>,
    ) -> Result<Self> {
        Self::from_gguf_source(None, Some(source), range, device, kv_quant_bits)
    }

    fn from_gguf_source(
        gguf_path: Option<&Path>,
        prepared: Option<&gguf_loader::PreparedNativeSource>,
        range: LayerRange,
        device: &Device,
        kv_quant_bits: Option<u8>,
    ) -> Result<Self> {
        let path = || gguf_path.context("prepared native source cannot reopen a path");
        // Count this stage on its GPU before it allocates there. This waits
        // for a decode step that another stage is recording on the GPU.
        let device_stages = device_stages::StageRegistration::register(device);
        // cudarc attaches ordering events to every buffer it allocates, for
        // use across streams. The engine issues all work for a device on one
        // stream, and a captured decode graph cannot carry those events, so
        // turn them off before any weight is allocated.
        #[cfg(feature = "cuda")]
        let cuda_events_off = match device {
            Device::Cuda(cuda) => {
                if cuda.is_event_tracking() {
                    // SAFETY: One stream per device; see above.
                    unsafe { cuda.disable_event_tracking() };
                }
                true
            }
            _ => false,
        };
        if let Some(bits) = kv_quant_bits {
            turboquant::validate_inference_bits(bits)?;
        }
        // GLM5 and MiniMax sources are streamed one layer at a time by their
        // architecture loaders. Other architectures retain the established
        // whole-stage tensor-map path.
        let metadata = if let Some(source) = prepared {
            source.metadata()?.clone()
        } else {
            gguf_loader::read_gguf_metadata(path()?)?
        };
        if metadata.architecture == "deepseek4" {
            architectures::deepseek4::DeepSeek4Stage::validate_stage_config(
                &metadata,
                &range,
                kv_quant_bits.is_some(),
            )?;
        }
        if metadata.architecture == "deepseek41" {
            architectures::deepseek41::DeepSeek41Stage::validate_stage_config(
                &metadata,
                &range,
                kv_quant_bits.is_some(),
                device,
            )?;
        }
        let is_glm5 = matches!(metadata.architecture.as_str(), "glm5next" | "glm5-next");
        let is_minimax = metadata.architecture == "minimax-m2";
        let is_muse = metadata.architecture == "muse-spark";
        range.validate()?;
        anyhow::ensure!(
            !(kv_quant_bits.is_some() && (is_glm5 || metadata.architecture == "deepseek2")),
            "{} latent caches do not support TurboQuant; use the default F16 cache",
            metadata.architecture
        );
        let head_dim = metadata.head_dim;
        let quant_config = kv_quant_bits
            .map(|bits| TurboQuantConfig::try_new(bits, head_dim, 42).map(Arc::new))
            .transpose()?;
        if quant_config.is_some() {
            for layer in range.start..range.end {
                anyhow::ensure!(
                    metadata.head_dim_for_layer(layer) == head_dim,
                    "TurboQuant does not support varying attention head dimensions in one stage"
                );
            }
            #[cfg(target_os = "macos")]
            crate::turboquant_metal::precompile(device)?;
        }
        // Candle retains dropped Metal buffers until a device wait. Release
        // the previous model before allocating another stage: waiting only
        // after the new weights arrive can exceed unified memory capacity.
        device
            .synchronize()
            .context("releasing unused device buffers before model load")?;
        let stage_tensors = if let Some(source) = prepared {
            Some(source.load_stage_tensors(range.clone(), device)?)
        } else if gguf_loader::uses_streaming_sources(&metadata.architecture) {
            None
        } else {
            Some(gguf_loader::load_stage_tensors(
                path()?,
                range.clone(),
                device,
            )?)
        };
        let position = if range.is_solo() {
            StagePosition::Solo
        } else if range.is_head() {
            StagePosition::Head
        } else if range.is_tail() {
            StagePosition::Tail
        } else {
            StagePosition::Middle
        };

        let preallocated_kv_cache = quant_config.is_none();

        let dense_f16_kv_bytes_per_token = metadata.kv_bytes_per_token(range.start, range.end);
        let kv_bytes_per_token = if let Some(config) = quant_config.as_ref() {
            metadata.kv_bytes_per_token_quantized(range.start, range.end, config.bits())
        } else {
            dense_f16_kv_bytes_per_token
        };
        let kv_budget_bytes_per_token = if quant_config.is_some() {
            // Include old/new packed overlap and the dense attention outputs.
            // Deferred Metal validation keeps these buffers alive through the
            // forward pass. Packed storage alone is not a peak-memory budget.
            #[allow(unused_mut)] // Metal can use its native F16 reconstruction.
            let mut dense_copies = if device.is_cpu() { 2 } else { 5 };
            #[cfg(target_os = "macos")]
            if device.is_metal() && head_dim <= crate::turboquant_metal::MAX_METAL_HEAD_DIM {
                dense_copies = 1;
            }
            kv_bytes_per_token
                .saturating_mul(2)
                .saturating_add(dense_f16_kv_bytes_per_token.saturating_mul(dense_copies))
        } else {
            dense_f16_kv_bytes_per_token
        };

        if let Some(config) = quant_config.as_ref() {
            let bits = config.bits();
            tracing::info!(
                "TurboQuant {bits}-bit KV cache enabled (head_dim={head_dim}, {kv_bytes_per_token} bytes/token)"
            );
        }

        let arch: Box<dyn ModelArchitecture> = if is_glm5 {
            Box::new(
                architectures::glm5_next::Glm5NextStage::from_streaming_gguf(
                    path()?,
                    range.clone(),
                    device,
                    quant_config.clone(),
                )?,
            )
        } else if is_minimax {
            Box::new(
                architectures::minimax_m2::MiniMaxM2Stage::from_streaming_gguf(
                    path()?,
                    range.clone(),
                    device,
                    quant_config.clone(),
                )?,
            )
        } else if is_muse {
            Box::new(
                architectures::muse_spark::MuseSparkStage::from_streaming_gguf(
                    path()?,
                    range.clone(),
                    device,
                    quant_config.clone(),
                )?,
            )
        } else {
            architectures::from_stage_tensors(
                stage_tensors.expect("non-streaming stage tensors must be loaded"),
                quant_config.clone(),
            )?
        };
        let layer_count = arch.layer_count();
        let hidden_dim = arch.hidden_dim();
        let vocab_size = arch.vocab_size();
        let fixed_cache_bytes = arch.fixed_cache_bytes(0)?;

        Ok(Self {
            arch,
            speculative: None,
            speculative_cache: None,
            speculative_draft_max_tokens: None,
            position,
            cache: (0..layer_count).map(|_| None).collect(),
            hidden_dim,
            vocab_size,
            context_length: metadata.context_length,
            device: device.clone(),
            kv_bytes_per_token,
            kv_budget_bytes_per_token: kv_budget_bytes_per_token.max(1),
            fixed_cache_bytes,
            next_position: 0,
            mrope_decode_delta: None,
            text_arithmetic: None,
            kv_cache_tokens: 0,
            cache_token_ids: Vec::new(),
            cache_fingerprint: None,
            kv_cache_lane_capacity_tokens: 0,
            kv_cache_allocation_tokens: 0,
            fixed_cache_allocation_tokens: 0,
            quant_config,
            max_kv_tokens: None,
            kv_token_budget: Arc::new(KvTokenBudget::default()),
            max_total_kv_tokens: None,
            max_total_kv_headroom_tokens: None,
            preallocated_kv_cache,
            decode_batch_caches: std::collections::HashMap::new(),
            prompt_cache: prompt_cache::PromptCache::default(),
            options: EngineOptions::default(),
            #[cfg(feature = "cuda")]
            cuda_decode_graph: if cuda_events_off {
                cuda_graph::DecodeGraph::event_free()
            } else {
                cuda_graph::DecodeGraph::default()
            },
            device_stages,
        })
    }

    /// Attach a DFlash 2 GGUF to a full Qwen target stage.
    ///
    /// The draft shares the target embedding and output head. Split-range and
    /// quantized-KV stages keep using the normal decode path for now.
    /// Attach or replace it before creating session states, including idle ones.
    pub fn attach_speculative_draft(&mut self, gguf_path: &Path) -> Result<()> {
        anyhow::ensure!(
            self.position == StagePosition::Solo,
            "DFlash 2 currently requires a solo target stage"
        );
        anyhow::ensure!(
            self.quant_config.is_none(),
            "DFlash 2 does not yet support quantized target KV caches"
        );
        anyhow::ensure!(
            self.next_position == 0
                && self.total_kv_cache_tokens() == 0
                && self.total_kv_cache_allocation_tokens() == 0
                && Arc::strong_count(&self.kv_token_budget) == 1,
            "attach the speculative draft before target inference starts or session states are created"
        );
        let shared = self.arch.speculative_shared_weights().ok_or_else(|| {
            anyhow::anyhow!("{} cannot host a DFlash 2 draft", self.arch.family())
        })?;
        let draft =
            architectures::dflash2::DFlash2Draft::from_gguf(gguf_path, &self.device, shared)?;
        anyhow::ensure!(
            draft
                .target_layers()
                .iter()
                .all(|layer| *layer < self.arch.layer_count()),
            "DFlash target layer IDs exceed the loaded target"
        );
        let fixed_cache_bytes = self
            .arch
            .fixed_cache_bytes(self.speculative_reservation_rows(draft.block_size()))?
            .checked_add(draft.cache_budget_bytes()?)
            .context("fixed target/draft cache size overflow")?;
        self.fixed_cache_bytes = fixed_cache_bytes;
        // A replacement drafter must never reuse the previous drafter's state.
        self.prompt_cache.entries.clear();
        self.speculative_cache = Some(draft.new_cache());
        self.speculative = Some(draft);
        Ok(())
    }

    /// Whether a speculative draft model is attached to this stage.
    pub fn speculative_decoding_enabled(&self) -> bool {
        self.speculative.is_some()
    }

    /// Cap the number of tokens DFlash may propose in one pass, excluding the
    /// target's anchor token. Configure this after attaching the draft and
    /// before inference. The draft's trained limit and device preference can
    /// select fewer tokens. A diagnostic block-size override obeys this cap.
    pub fn set_speculative_draft_max_tokens(&mut self, max: usize) -> Result<()> {
        anyhow::ensure!(max > 0, "DFlash draft token limit must be positive");
        let rows = max
            .checked_add(1)
            .context("DFlash draft token limit overflow")?;
        let draft = self
            .speculative
            .as_ref()
            .context("attach a DFlash draft before setting its token limit")?;
        anyhow::ensure!(
            self.next_position == 0
                && self.total_kv_cache_tokens() == 0
                && self.total_kv_cache_allocation_tokens() == 0,
            "set the DFlash draft token limit before inference starts"
        );
        let fixed_cache_bytes = self
            .arch
            .fixed_cache_bytes(rows.min(draft.block_size()))?
            .checked_add(draft.cache_budget_bytes()?)
            .context("fixed target/draft cache size overflow")?;
        self.speculative_draft_max_tokens = Some(max);
        self.fixed_cache_bytes = fixed_cache_bytes;
        Ok(())
    }

    /// Maximum proposed tokens per DFlash pass, excluding the anchor. The
    /// device preference may choose a smaller pass at runtime.
    pub fn speculative_draft_max_tokens(&self) -> Option<usize> {
        self.speculative.as_ref().map(|draft| {
            self.speculative_draft_max_tokens
                .unwrap_or(draft.block_size() - 1)
                .min(draft.block_size() - 1)
        })
    }

    /// Runtime options this stage applies to requests.
    pub fn options(&self) -> &EngineOptions {
        &self.options
    }

    /// Replace this stage's runtime options.
    ///
    /// The new options apply from the next call. A
    /// [`GenerationSession`](crate::continuous::GenerationSession) already in
    /// progress reads the speculative policy at each step, so set options
    /// before starting requests.
    ///
    /// # Errors
    ///
    /// Returns an error, leaving the current options in place, when
    /// [`EngineOptions::validate`] rejects `options`.
    pub fn set_options(&mut self, options: EngineOptions) -> Result<()> {
        options.validate()?;
        self.options = options;
        Ok(())
    }

    fn speculative_reservation_rows(&self, trained_rows: usize) -> usize {
        self.speculative_draft_max_tokens
            .map_or(trained_rows, |max| (max + 1).min(trained_rows))
    }

    pub(crate) fn session_speculative_ready(&self, state: &PipelineStageSessionState) -> bool {
        self.speculative.is_some()
            && !state.has_embedded_prompt
            && state.mrope_decode_delta.is_none()
            && state
                .speculative_cache
                .as_ref()
                .is_some_and(|cache| cache.next_position() == state.next_position)
    }

    pub(crate) fn speculative_rows_fit_context(
        &self,
        state: &PipelineStageSessionState,
        rows: usize,
    ) -> bool {
        self.speculative.as_ref().is_some_and(|draft| {
            state
                .next_position
                .checked_add(rows)
                .is_some_and(|end| end <= self.context_length && end <= draft.context_length())
        })
    }

    /// Verification must see the same old context as the next scalar token.
    /// Speculative rows are tentative, so they must never force eviction.
    fn preflight_speculative_session(
        &mut self,
        state: &PipelineStageSessionState,
        rows: usize,
    ) -> Result<bool> {
        anyhow::ensure!(
            Arc::ptr_eq(&state.kv_token_budget, &self.kv_token_budget)
                && state.cache.len() == self.cache.len(),
            "draft cache belongs to another stage"
        );
        if !self.speculative_rows_fit_context(state, rows) {
            return Ok(false);
        }
        let desired = state
            .kv_cache_tokens
            .checked_add(rows)
            .context("draft KV length overflow")?;
        if self.max_kv_tokens.is_some_and(|max| desired > max) {
            return Ok(false);
        }
        let allocated = self.predicted_allocation_for_cache(
            desired,
            false,
            state.kv_cache_lane_capacity_tokens,
            state.kv_cache_allocation_tokens,
        )?;
        let additional = allocated
            .saturating_sub(state.kv_cache_allocation_tokens)
            .max(rows)
            .saturating_add(
                self.fixed_cache_budget_tokens()
                    .saturating_sub(state.fixed_cache_allocation_tokens),
            );
        self.release_prompt_cache_under_pressure(additional);
        let Some(max) = self.max_total_kv_tokens else {
            return Ok(true);
        };
        let retained = self
            .total_kv_cache_tokens()
            .checked_add(rows)
            .context("draft KV length overflow")?;
        let resident = self
            .total_kv_cache_allocation_tokens()
            .checked_sub(state.kv_cache_allocation_tokens)
            .and_then(|n| n.checked_sub(state.fixed_cache_allocation_tokens))
            .and_then(|n| n.checked_add(allocated))
            .and_then(|n| n.checked_add(self.fixed_cache_budget_tokens()))
            .context("draft allocation accounting overflow")?;
        Ok(retained <= max && resident <= max)
    }

    pub(crate) fn speculative_block_size(&self) -> Option<usize> {
        self.speculative.as_ref().map(|draft| draft.block_size())
    }

    pub(crate) fn preferred_speculative_block_size(&self) -> Result<Option<usize>> {
        self.speculative
            .as_ref()
            .map(|draft| {
                architectures::preferred_speculative_block_size(&self.device, draft.block_size())
            })
            .transpose()
    }

    pub(crate) fn effective_speculative_block_size(&self) -> Result<Option<usize>> {
        let Some(max) = self.speculative_block_size() else {
            return Ok(None);
        };
        let preferred = self.preferred_speculative_block_size()?.unwrap();
        resolve_speculative_block_size(max, preferred, self.speculative_draft_max_tokens).map(Some)
    }

    pub(crate) fn uses_pre_m5_speculative_policy(&self) -> Result<bool> {
        architectures::uses_pre_m5_speculative_policy(&self.device)
    }

    /// Run a forward pass through this stage.
    /// For prompt chunks, use [`Self::prefill`], even when the chunk has one token.
    ///
    /// `start_pos` is the absolute position of the first incoming token. It
    /// must equal [`PipelineStage::next_position`]. `kv_cache_tokens` tracks
    /// only the retained attention window and can be smaller after eviction.
    ///
    /// If `max_kv_tokens` is set, this evicts the smallest old prefix needed
    /// before the pass. Eviction never changes `start_pos`.
    pub fn forward(
        &mut self,
        input: StageInput,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_pass(
            input,
            seq_len,
            start_pos,
            ForwardPass::Standard {
                layer_ids: &[],
                all_logits: false,
            },
        )
        .map(|(output, _)| output)
    }

    /// Process a prompt chunk with the same cache limits as [`Self::forward`].
    /// Use this even for a one-token prompt tail; use `forward` for decode.
    pub fn prefill(
        &mut self,
        input: StageInput,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_pass(input, seq_len, start_pos, ForwardPass::Prefill)
            .map(|(output, _)| output)
    }

    /// Process a prompt chunk with the complete text request bound.
    /// Unlike `prefill`, this opts supported architectures into request-wide
    /// row arithmetic. Each partition must receive the same context.
    pub fn prefill_text(
        &mut self,
        input: StageInput,
        seq_len: usize,
        start_pos: usize,
        context: TextRequestContext,
    ) -> Result<StageOutput> {
        self.forward_pass(
            input,
            seq_len,
            start_pos,
            ForwardPass::Text {
                context,
                prefill: true,
                cache_only: false,
                tail_fence: false,
            },
        )
        .map(|(output, _)| output)
    }

    /// Decode after an explicitly bounded text prompt.
    pub fn forward_text(
        &mut self,
        input: StageInput,
        seq_len: usize,
        start_pos: usize,
        context: TextRequestContext,
    ) -> Result<StageOutput> {
        self.forward_pass(
            input,
            seq_len,
            start_pos,
            ForwardPass::Text {
                context,
                prefill: false,
                cache_only: false,
                tail_fence: false,
            },
        )
        .map(|(output, _)| output)
    }

    pub(crate) fn desired_text_arithmetic(
        &self,
        context: TextRequestContext,
    ) -> Option<TextArithmeticPolicy> {
        self.arch.text_arithmetic_policy(context)
    }

    /// Drop a legacy or incompatible request cache using the normal charged
    /// clear path. Returns true when its existing causal prefix remains valid.
    pub fn prepare_text_request(&mut self, context: TextRequestContext) -> bool {
        let valid = TextArithmeticPolicy::compatible(
            self.text_arithmetic,
            self.desired_text_arithmetic(context),
            self.next_position,
        );
        if !valid {
            self.clear_cache();
        }
        self.text_arithmetic = self.desired_text_arithmetic(context);
        valid
    }

    fn forward_pass(
        &mut self,
        input: StageInput,
        seq_len: usize,
        start_pos: usize,
        pass: ForwardPass<'_>,
    ) -> Result<(StageOutput, Vec<candle_core::Tensor>)> {
        let policy = if let ForwardPass::Text {
            context, prefill, ..
        } = pass
        {
            context.validate(start_pos, seq_len, prefill)?;
            self.desired_text_arithmetic(context)
        } else {
            None
        };
        anyhow::ensure!(
            TextArithmeticPolicy::compatible(self.text_arithmetic, policy, self.next_position),
            "text arithmetic changed on a nonempty cache; prepare the full request first"
        );
        let result = self.forward_pass_inner(input, seq_len, start_pos, pass);
        if result.is_ok() {
            self.text_arithmetic = policy;
        }
        result
    }

    fn forward_pass_inner(
        &mut self,
        input: StageInput,
        seq_len: usize,
        start_pos: usize,
        pass: ForwardPass<'_>,
    ) -> Result<(StageOutput, Vec<candle_core::Tensor>)> {
        anyhow::ensure!(seq_len > 0, "forward sequence length must be positive");
        self.validate_token_input(&input, seq_len)?;
        anyhow::ensure!(
            start_pos == self.next_position,
            "non-contiguous forward position: expected {}, received {start_pos}",
            self.next_position
        );
        let next_position = start_pos
            .checked_add(seq_len)
            .ok_or_else(|| anyhow::anyhow!("absolute token position overflow"))?;
        if self.arch.family() == "deepseek41" {
            anyhow::ensure!(
                next_position <= self.context_length,
                "DeepSeek V4.1 context limit exceeded"
            );
            anyhow::ensure!(
                matches!(&input, StageInput::Tokens(tokens)
                if tokens.len() == seq_len && tokens.iter().all(|&id| (id as usize) < self.vocab_size)),
                "Maya accepts scalar text tokens only"
            );
            if let ForwardPass::Standard {
                layer_ids,
                all_logits,
            } = pass
            {
                anyhow::ensure!(
                    layer_ids.is_empty() && !all_logits,
                    "Maya speculative capture is unsupported"
                );
            }
        }
        let mrope_decode_delta = match &input {
            StageInput::Activations {
                positions: Some(positions),
                ..
            } => {
                anyhow::ensure!(
                    positions.len() == seq_len,
                    "M-RoPE position count differs from sequence length"
                );
                let delta = positions.decode_delta(next_position)?;
                match self.mrope_decode_delta {
                    // A chunk may not contain the largest coordinate of an
                    // earlier image chunk. Keep the maximum across all axes.
                    Some(previous) => {
                        let rows =
                            i64::try_from(seq_len).context("M-RoPE sequence length exceeds i64")?;
                        Some(
                            delta.max(
                                previous
                                    .checked_sub(rows)
                                    .context("M-RoPE decode delta overflow")?,
                            ),
                        )
                    }
                    None => Some(delta),
                }
            }
            _ => self.mrope_decode_delta,
        };

        let desired = self.kv_cache_tokens.saturating_add(seq_len);
        let additional = self
            .predicted_allocation_tokens(desired, false)?
            .saturating_sub(self.kv_cache_allocation_tokens)
            .max(seq_len)
            .saturating_add(
                self.fixed_cache_budget_tokens()
                    .saturating_sub(self.fixed_cache_allocation_tokens),
            );
        self.release_prompt_cache_under_pressure(additional);
        let evict = self.required_eviction(seq_len)?;
        if evict > 0 {
            tracing::debug!(
                retained_tokens = self.kv_cache_tokens,
                total_retained_tokens = self.total_kv_cache_tokens(),
                absolute_position = start_pos,
                "KV eviction: evicting {evict} tokens before appending {seq_len} tokens"
            );
            if evict == self.kv_cache_tokens {
                tracing::debug!(
                    "KV capacity requires replacing the complete cache with the incoming sequence"
                );
                let result = self.forward_with_replacement_cache(
                    input,
                    seq_len,
                    start_pos,
                    next_position,
                    pass,
                );
                if result.is_ok() {
                    self.mrope_decode_delta = mrope_decode_delta;
                }
                return result;
            }
            self.try_evict_oldest(evict)?;
        }

        let retained_after = self
            .kv_cache_tokens
            .checked_add(seq_len)
            .ok_or_else(|| anyhow::anyhow!("KV cache length overflow"))?;
        let failure_checkpoint = if self.arch.family() == "deepseek41" {
            Some(
                self.cache
                    .iter()
                    .map(|slot| {
                        slot.as_ref()
                            .map(architectures::checkpoint_layer_cache)
                            .transpose()
                    })
                    .collect::<Result<Vec<_>>>()?,
            )
        } else {
            None
        };
        let result = pass.run(
            self.arch.as_ref(),
            input,
            &mut self.cache,
            seq_len,
            start_pos,
        );
        match result {
            Ok(output) => {
                if let Err(error) = self
                    .cache
                    .iter_mut()
                    .flatten()
                    .try_for_each(architectures::compact_fixed_layer_cache)
                {
                    self.restore_forward_failure(failure_checkpoint);
                    return Err(error.context("could not compact fixed request state"));
                }
                let (lane_capacity_tokens, allocation_tokens) =
                    match architectures::kv_cache_allocation_tokens(&self.cache) {
                        Ok(tokens) => tokens,
                        Err(error) => {
                            self.restore_forward_failure(failure_checkpoint);
                            return Err(error.context("could not measure KV cache allocation"));
                        }
                    };
                if let Err(error) = self.ensure_allocation_fits(allocation_tokens) {
                    self.restore_forward_failure(failure_checkpoint);
                    return Err(error);
                }
                if let Err(error) = pass.finish(&self.device) {
                    self.restore_forward_failure(failure_checkpoint);
                    return Err(error);
                }
                let previous_allocation = self.kv_cache_allocation_tokens;
                self.next_position = next_position;
                self.mrope_decode_delta = mrope_decode_delta;
                self.kv_cache_tokens = retained_after;
                self.kv_cache_lane_capacity_tokens = lane_capacity_tokens;
                self.kv_cache_allocation_tokens = allocation_tokens;
                self.kv_token_budget.add(seq_len);
                self.kv_token_budget
                    .replace_allocated(previous_allocation, allocation_tokens);
                self.charge_fixed_cache();
                Ok(output)
            }
            Err(error) => {
                // Architectures update layer caches one at a time. Drop the
                // active cache after a failed pass so partial allocations
                // cannot escape aggregate accounting or be reused.
                self.restore_forward_failure(failure_checkpoint);
                Err(error)
            }
        }
    }

    fn restore_forward_failure(&mut self, checkpoint: Option<Vec<Option<LayerCache>>>) {
        if let Some(cache) = checkpoint {
            self.cache = cache;
        } else {
            self.clear_cache();
        }
    }

    fn required_eviction(&self, seq_len: usize) -> Result<usize> {
        let desired = self
            .kv_cache_tokens
            .checked_add(seq_len)
            .ok_or_else(|| anyhow::anyhow!("KV cache length overflow"))?;
        let lane_capacity = self.max_kv_tokens.unwrap_or(usize::MAX);
        anyhow::ensure!(
            seq_len <= lane_capacity,
            "prompt chunk contains {seq_len} tokens but KV capacity is {lane_capacity}; split or shorten the prompt"
        );

        let total = self.total_kv_cache_tokens();
        anyhow::ensure!(
            total >= self.kv_cache_tokens,
            "aggregate KV accounting is smaller than the active cache"
        );
        let other_sessions = total - self.kv_cache_tokens;
        let aggregate_capacity = match self.max_total_kv_tokens {
            Some(max) => {
                let available = max.checked_sub(other_sessions).ok_or_else(|| {
                    anyhow::anyhow!(
                        "other sessions retain {other_sessions} KV tokens, above the aggregate capacity {max}"
                    )
                })?;
                anyhow::ensure!(
                    seq_len <= available,
                    "aggregate KV capacity {max} has {available} tokens available after other sessions; incoming sequence needs {seq_len}"
                );
                available
            }
            None => usize::MAX,
        };
        let retained_capacity = lane_capacity.min(aggregate_capacity);
        let mut eviction = desired.saturating_sub(retained_capacity);

        if let Some(max) = self.max_total_kv_tokens {
            let total_allocated = self.kv_token_budget.allocated();
            anyhow::ensure!(
                total_allocated >= self.kv_cache_allocation_tokens,
                "aggregate KV allocation accounting is smaller than the active cache"
            );
            let other_allocated = total_allocated
                .checked_sub(self.kv_cache_allocation_tokens)
                .and_then(|bytes| bytes.checked_sub(self.fixed_cache_allocation_tokens))
                .context("fixed cache allocation accounting underflow")?;
            let available_allocated = max.checked_sub(other_allocated)
                .and_then(|available| available.checked_sub(self.fixed_cache_budget_tokens()))
                .ok_or_else(|| {
                anyhow::anyhow!(
                    "resident allocation limit {max} cannot fit fixed request state after {other_allocated} slots reserved by other sessions"
                )
            })?;

            let mut retained_after = desired.saturating_sub(eviction);
            let mut replacing = eviction == self.kv_cache_tokens && eviction > 0;
            let mut allocation_after =
                self.predicted_allocation_tokens(retained_after, replacing)?;

            // Spare capacity can keep a lane decoding after its logical token
            // count has room to grow but the shared allocation budget does not.
            // Roll that lane instead of growing its backing buffers.
            if allocation_after > available_allocated
                && self.kv_cache_allocation_tokens > 0
                && retained_after > self.kv_cache_lane_capacity_tokens
            {
                let extra = retained_after - self.kv_cache_lane_capacity_tokens;
                let remaining = self.kv_cache_tokens.saturating_sub(eviction);
                if extra <= remaining {
                    eviction += extra;
                    retained_after -= extra;
                    replacing = eviction == self.kv_cache_tokens;
                    allocation_after =
                        self.predicted_allocation_tokens(retained_after, replacing)?;
                }
            }
            anyhow::ensure!(
                allocation_after <= available_allocated,
                "resident KV allocation limit {max} has {available_allocated} slots available after other sessions; this forward needs {allocation_after}"
            );
        }

        let resulting_total = total
            .checked_sub(eviction)
            .and_then(|retained| retained.checked_add(seq_len))
            .ok_or_else(|| anyhow::anyhow!("aggregate KV cache length overflow"))?;
        if let Some(max) = self.max_total_kv_tokens {
            anyhow::ensure!(
                resulting_total <= max,
                "aggregate KV capacity {max} would be exceeded by this forward"
            );
        }
        anyhow::ensure!(
            eviction == 0 || self.arch.family() != "deepseek41",
            "Maya history cannot be evicted; context/cache budget exhausted"
        );
        Ok(eviction)
    }

    fn predicted_allocation_tokens(&self, retained_after: usize, replacing: bool) -> Result<usize> {
        self.predicted_allocation_for_cache(
            retained_after,
            replacing,
            self.kv_cache_lane_capacity_tokens,
            self.kv_cache_allocation_tokens,
        )
    }

    fn predicted_allocation_for_cache(
        &self,
        retained_after: usize,
        replacing: bool,
        lane_capacity_tokens: usize,
        allocation_tokens: usize,
    ) -> Result<usize> {
        if self.arch.family() == "deepseek41" {
            return Ok(
                architectures::deepseek41::DeepSeek41Cache::predicted_allocation_tokens(
                    retained_after,
                    &self.device,
                )?
                .max(allocation_tokens),
            );
        }
        if retained_after == 0 {
            return Ok(0);
        }
        if !self.preallocated_kv_cache {
            #[cfg(target_os = "macos")]
            if self.device.is_metal()
                && self.quant_config.as_ref().is_some_and(|config| {
                    config.head_dim() <= crate::turboquant_metal::MAX_METAL_HEAD_DIM
                })
            {
                return turboquant::metal_cache_capacity(
                    if replacing { 0 } else { allocation_tokens },
                    retained_after,
                );
            }
            if self.quant_config.is_some() {
                return turboquant::cpu_cache_capacity(
                    if replacing { 0 } else { allocation_tokens },
                    retained_after,
                );
            }
            return Ok(retained_after);
        }
        let current = if replacing { 0 } else { lane_capacity_tokens };
        if current == 0 || retained_after > current {
            architectures::PreallocatedKvCache::next_capacity(current, retained_after)
        } else {
            Ok(allocation_tokens)
        }
    }

    fn fixed_cache_budget_tokens(&self) -> usize {
        self.fixed_cache_bytes.div_ceil(
            usize::try_from(self.kv_budget_bytes_per_token)
                .unwrap_or(usize::MAX)
                .max(1),
        )
    }

    fn charge_fixed_cache(&mut self) {
        let fixed = self.fixed_cache_budget_tokens();
        self.kv_token_budget
            .replace_allocated(self.fixed_cache_allocation_tokens, fixed);
        self.fixed_cache_allocation_tokens = fixed;
    }

    /// Resident and verification state reserved for each live request.
    pub fn fixed_cache_budget_bytes(&self) -> usize {
        self.fixed_cache_bytes
    }

    fn ensure_allocation_fits(&self, allocation_after: usize) -> Result<()> {
        let Some(max) = self.max_total_kv_tokens else {
            return Ok(());
        };
        let total = self.kv_token_budget.allocated();
        anyhow::ensure!(
            total >= self.kv_cache_allocation_tokens,
            "aggregate KV allocation accounting is smaller than the active cache"
        );
        let required = total
            .checked_sub(self.kv_cache_allocation_tokens)
            .and_then(|other| other.checked_sub(self.fixed_cache_allocation_tokens))
            .and_then(|other| other.checked_add(self.fixed_cache_budget_tokens()))
            .and_then(|other| other.checked_add(allocation_after))
            .ok_or_else(|| anyhow::anyhow!("aggregate KV allocation length overflow"))?;
        anyhow::ensure!(
            required <= max,
            "resident KV allocation limit {max} would be exceeded by {required} slots"
        );
        Ok(())
    }

    /// Run a forward pass with fresh attention history and retained recurrence,
    /// publishing the replacement only if the architecture succeeds. This is
    /// used when a chunk evicts every cached token: mutating each old
    /// layer in place would either reject the equality boundary or expose a
    /// partially rebuilt cache after a failed forward pass.
    fn forward_with_replacement_cache(
        &mut self,
        input: StageInput,
        seq_len: usize,
        start_pos: usize,
        next_position: usize,
        pass: ForwardPass<'_>,
    ) -> Result<(StageOutput, Vec<candle_core::Tensor>)> {
        anyhow::ensure!(
            self.arch.family() != "deepseek41"
                && !self
                    .cache
                    .iter()
                    .any(|c| matches!(c, Some(LayerCache::DeepSeek41(_)))),
            "Maya history cannot be replaced by outer eviction"
        );
        anyhow::ensure!(
            !self
                .cache
                .iter()
                .any(|c| matches!(c, Some(LayerCache::Inkling2(_)))),
            "Inkling 2 visit state cannot be replaced by outer eviction"
        );
        // Evict attention history, not the fixed recurrence which summarizes
        // the whole preceding sequence. Recurrent forwards create new state
        // tensors, so these immutable clones also keep failure rollback safe.
        let replacement = self
            .cache
            .iter()
            .map(|entry| match entry {
                Some(cache @ (LayerCache::DeltaNet(_) | LayerCache::Mamba2(_))) => {
                    architectures::checkpoint_layer_cache(cache).map(Some)
                }
                _ => Ok(None),
            })
            .collect::<Result<Vec<_>>>()?;
        let mut previous_cache = std::mem::replace(&mut self.cache, replacement);

        match pass.run(
            self.arch.as_ref(),
            input,
            &mut self.cache,
            seq_len,
            start_pos,
        ) {
            Ok(output) => {
                if let Err(error) = self
                    .cache
                    .iter_mut()
                    .flatten()
                    .try_for_each(architectures::compact_fixed_layer_cache)
                {
                    std::mem::swap(&mut self.cache, &mut previous_cache);
                    return Err(error.context("could not compact replacement request state"));
                }
                let (lane_capacity_tokens, allocation_tokens) =
                    match architectures::kv_cache_allocation_tokens(&self.cache) {
                        Ok(tokens) => tokens,
                        Err(error) => {
                            std::mem::swap(&mut self.cache, &mut previous_cache);
                            return Err(error.context("could not measure replacement KV cache"));
                        }
                    };
                if let Err(error) = self.ensure_allocation_fits(allocation_tokens) {
                    std::mem::swap(&mut self.cache, &mut previous_cache);
                    return Err(error);
                }
                if let Err(error) = pass.finish(&self.device) {
                    std::mem::swap(&mut self.cache, &mut previous_cache);
                    return Err(error);
                }
                let previous_tokens = self.kv_cache_tokens;
                let previous_allocation = self.kv_cache_allocation_tokens;
                self.next_position = next_position;
                self.kv_cache_tokens = seq_len;
                self.kv_cache_lane_capacity_tokens = lane_capacity_tokens;
                self.kv_cache_allocation_tokens = allocation_tokens;
                self.kv_token_budget.replace(previous_tokens, seq_len);
                self.kv_token_budget
                    .replace_allocated(previous_allocation, allocation_tokens);
                self.charge_fixed_cache();
                tracing::debug!(
                    cache_tokens = self.kv_cache_tokens,
                    absolute_position = self.next_position,
                    "KV cache replacement completed"
                );
                Ok(output)
            }
            Err(error) => {
                std::mem::swap(&mut self.cache, &mut previous_cache);
                tracing::warn!(%error, "KV cache replacement failed; retained the previous cache");
                Err(error)
            }
        }
    }

    /// Allocate empty cache state for an independent inference session.
    pub fn new_session_state(&self) -> PipelineStageSessionState {
        PipelineStageSessionState {
            cache: (0..self.cache.len()).map(|_| None).collect(),
            speculative_cache: self.speculative.as_ref().map(|draft| draft.new_cache()),
            next_position: 0,
            kv_cache_tokens: 0,
            kv_cache_lane_capacity_tokens: 0,
            kv_cache_allocation_tokens: 0,
            fixed_cache_allocation_tokens: 0,
            kv_token_budget: self.kv_token_budget.clone(),
            decode_batch_cache: None,
            cache_id: NEXT_SESSION_CACHE_ID.fetch_add(1, Ordering::Relaxed),
            revision: 0,
            mrope_decode_delta: None,
            text_arithmetic: None,
            has_embedded_prompt: false,
            resume_invalidated: false,
        }
    }

    /// Run a forward pass using cache state belonging to one session.
    /// Use [`Self::prefill_with_session`] for prompt chunks.
    ///
    /// The loaded architecture and weights stay resident in this stage. Cache
    /// ownership is swapped only for the duration of the call and restored even
    /// when the model forward returns an error.
    pub fn forward_with_session(
        &mut self,
        state: &mut PipelineStageSessionState,
        input: StageInput,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_with_session_pass(
            state,
            input,
            seq_len,
            start_pos,
            ForwardPass::Standard {
                layer_ids: &[],
                all_logits: false,
            },
        )
        .map(|(output, _)| output)
    }

    /// Process a prompt chunk using cache state belonging to one session.
    pub fn prefill_with_session(
        &mut self,
        state: &mut PipelineStageSessionState,
        input: StageInput,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_with_session_pass(state, input, seq_len, start_pos, ForwardPass::Prefill)
            .map(|(output, _)| output)
    }

    /// Check a retained session against the whole next text prompt.
    /// A false result means the cache was cleared and the caller must report
    /// zero reused tokens and rebuild from position zero.
    pub fn prepare_text_request_with_session(
        &mut self,
        state: &mut PipelineStageSessionState,
        context: TextRequestContext,
    ) -> Result<bool> {
        anyhow::ensure!(
            Arc::ptr_eq(&state.kv_token_budget, &self.kv_token_budget),
            "session cache belongs to a different pipeline stage"
        );
        let valid = TextArithmeticPolicy::compatible(
            state.text_arithmetic,
            self.desired_text_arithmetic(context),
            state.next_position,
        );
        if !valid {
            self.clear_session_state(state);
        }
        state.text_arithmetic = self.desired_text_arithmetic(context);
        Ok(valid)
    }

    /// Process a prompt chunk with the full text request bounds and session cache.
    pub fn prefill_text_with_session(
        &mut self,
        state: &mut PipelineStageSessionState,
        input: StageInput,
        seq_len: usize,
        start_pos: usize,
        context: TextRequestContext,
    ) -> Result<StageOutput> {
        self.forward_with_session_pass(
            state,
            input,
            seq_len,
            start_pos,
            ForwardPass::Text {
                context,
                prefill: true,
                cache_only: false,
                tail_fence: false,
            },
        )
        .map(|(output, _)| output)
    }

    /// Decode text with the same request bounds that created the session cache.
    pub fn forward_text_with_session(
        &mut self,
        state: &mut PipelineStageSessionState,
        input: StageInput,
        seq_len: usize,
        start_pos: usize,
        context: TextRequestContext,
    ) -> Result<StageOutput> {
        self.forward_with_session_pass(
            state,
            input,
            seq_len,
            start_pos,
            ForwardPass::Text {
                context,
                prefill: false,
                cache_only: false,
                tail_fence: false,
            },
        )
        .map(|(output, _)| output)
    }

    /// Fill a solo-stage text cache without returning scores.
    pub fn prefill_text_cache_only_with_session(
        &mut self,
        state: &mut PipelineStageSessionState,
        input: StageInput,
        seq_len: usize,
        start_pos: usize,
        context: TextRequestContext,
    ) -> Result<()> {
        anyhow::ensure!(
            self.position == StagePosition::Solo,
            "cache-only prefill requires a solo stage"
        );
        self.forward_with_session_pass(
            state,
            input,
            seq_len,
            start_pos,
            ForwardPass::Text {
                context,
                prefill: true,
                cache_only: true,
                tail_fence: false,
            },
        )
        .map(|_| ())
    }

    /// Fill a tail-stage text cache and complete pending device work.
    pub fn prefill_text_tail_cache_only_with_session(
        &mut self,
        state: &mut PipelineStageSessionState,
        input: StageInput,
        seq_len: usize,
        start_pos: usize,
        context: TextRequestContext,
    ) -> Result<()> {
        anyhow::ensure!(
            self.position == StagePosition::Tail,
            "tail cache-only prefill requires a tail stage"
        );
        self.forward_with_session_pass(
            state,
            input,
            seq_len,
            start_pos,
            ForwardPass::Text {
                context,
                prefill: true,
                cache_only: true,
                tail_fence: true,
            },
        )
        .map(|_| ())
    }

    /// Advance a nonfinal prompt chunk on a solo stage without returning scores.
    /// This keeps the usual session cache limits, state swap, and failure cleanup.
    /// Use [`Self::prefill_with_session`] when logits are needed for sampling or
    /// a prompt-cache checkpoint. Non-solo stages are rejected before mutation.
    pub fn prefill_cache_only_with_session(
        &mut self,
        state: &mut PipelineStageSessionState,
        input: StageInput,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            self.position == StagePosition::Solo,
            "cache-only prefill requires a solo stage"
        );
        self.forward_with_session_pass(
            state,
            input,
            seq_len,
            start_pos,
            ForwardPass::PrefillCacheOnly,
        )
        .map(|_| ())
    }

    /// Advance a nonfinal prompt chunk on a tail stage without returning scores.
    /// This keeps the usual session cache limits and failure cleanup. Device
    /// work completes before success, so the caller can acknowledge progress
    /// without reading logits. Use [`Self::prefill_with_session`] for the final
    /// chunk that supplies the first token's scores.
    pub fn prefill_tail_cache_only_with_session(
        &mut self,
        state: &mut PipelineStageSessionState,
        input: StageInput,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            self.position == StagePosition::Tail,
            "tail cache-only prefill requires a tail stage"
        );
        self.forward_with_session_pass(
            state,
            input,
            seq_len,
            start_pos,
            ForwardPass::TailPrefillCacheOnly,
        )
        .map(|_| ())
    }

    fn forward_with_session_pass(
        &mut self,
        state: &mut PipelineStageSessionState,
        input: StageInput,
        seq_len: usize,
        start_pos: usize,
        pass: ForwardPass<'_>,
    ) -> Result<(StageOutput, Vec<candle_core::Tensor>)> {
        #[cfg(feature = "cuda")]
        if self.cuda_graph_step_eligible(&input, seq_len) {
            return self.forward_with_session_graph(state, input, seq_len, start_pos, pass);
        }
        self.forward_with_session_direct(state, input, seq_len, start_pos, pass)
    }

    /// Whether this step may run as one captured CUDA graph: a single text
    /// token on CUDA, with the option on and no earlier capture failure. The
    /// stage must also be the only one on its GPU, which is checked when the
    /// step takes its recording turn.
    #[cfg(feature = "cuda")]
    fn cuda_graph_step_eligible(&self, input: &StageInput, seq_len: usize) -> bool {
        self.options.cuda_graph_decode
            && seq_len == 1
            && self.device.is_cuda()
            && matches!(input, StageInput::Tokens(_))
            && self.speculative.is_none()
            && !self.options.profile.decode
            && !architectures::profiling::layer_profiling_enabled()
            && self.cuda_decode_graph.usable()
    }

    /// Run one decode step with its launches recorded and run as one graph.
    /// If the capture fails, the session is rolled back and the step repeats
    /// with direct launches; when that succeeds, graph capture stays off for
    /// this stage.
    #[cfg(feature = "cuda")]
    fn forward_with_session_graph(
        &mut self,
        state: &mut PipelineStageSessionState,
        input: StageInput,
        seq_len: usize,
        start_pos: usize,
        pass: ForwardPass<'_>,
    ) -> Result<(StageOutput, Vec<candle_core::Tensor>)> {
        // Another stage on this GPU could issue work into the recording, so
        // only a sole stage records; the turn also holds back a stage being
        // loaded on the GPU until this step is submitted.
        let Some(turn) = self.device_stages.begin_recording() else {
            return self.forward_with_session_direct(state, input, seq_len, start_pos, pass);
        };
        // Checkpoints do not cover every cache kind; those steps run direct.
        let Ok(checkpoint) = self.checkpoint_session_state(state) else {
            drop(turn);
            return self.forward_with_session_direct(state, input, seq_len, start_pos, pass);
        };
        let retry_input = input.clone();
        let device = self.device.clone();
        let mut graph = std::mem::take(&mut self.cuda_decode_graph);
        let outcome = graph.run(&device, || {
            self.forward_with_session_direct(state, input, seq_len, start_pos, pass)
        });
        self.cuda_decode_graph = graph;
        drop(turn);
        let graph_error = match outcome {
            Ok(output) => return Ok(output),
            Err(error) => error,
        };
        self.restore_session_checkpoint(state, checkpoint);
        let direct = self.forward_with_session_direct(state, retry_input, seq_len, start_pos, pass);
        if direct.is_ok() {
            tracing::warn!(
                "CUDA graph decode disabled for this stage after a failed capture: {graph_error:#}"
            );
            self.cuda_decode_graph.disable();
        }
        direct
    }

    /// Counters for graph-captured decode steps on this stage.
    #[cfg(feature = "cuda")]
    pub fn cuda_graph_decode_stats(&self) -> cuda_graph::DecodeGraphStats {
        self.cuda_decode_graph.stats()
    }

    fn forward_with_session_direct(
        &mut self,
        state: &mut PipelineStageSessionState,
        input: StageInput,
        seq_len: usize,
        start_pos: usize,
        pass: ForwardPass<'_>,
    ) -> Result<(StageOutput, Vec<candle_core::Tensor>)> {
        self.validate_token_input(&input, seq_len)?;
        anyhow::ensure!(
            state.cache.len() == self.cache.len(),
            "session cache has {} layers, stage has {}",
            state.cache.len(),
            self.cache.len()
        );
        anyhow::ensure!(
            Arc::ptr_eq(&state.kv_token_budget, &self.kv_token_budget),
            "session cache belongs to a different pipeline stage"
        );
        self.detach_decode_batch(state);
        std::mem::swap(&mut self.cache, &mut state.cache);
        std::mem::swap(&mut self.next_position, &mut state.next_position);
        std::mem::swap(&mut self.mrope_decode_delta, &mut state.mrope_decode_delta);
        std::mem::swap(&mut self.text_arithmetic, &mut state.text_arithmetic);
        std::mem::swap(&mut self.kv_cache_tokens, &mut state.kv_cache_tokens);
        std::mem::swap(
            &mut self.kv_cache_lane_capacity_tokens,
            &mut state.kv_cache_lane_capacity_tokens,
        );
        std::mem::swap(
            &mut self.kv_cache_allocation_tokens,
            &mut state.kv_cache_allocation_tokens,
        );
        std::mem::swap(
            &mut self.fixed_cache_allocation_tokens,
            &mut state.fixed_cache_allocation_tokens,
        );
        let uses_embeddings = matches!(&input, StageInput::Activations { .. });
        let result = self.forward_pass(input, seq_len, start_pos, pass);
        std::mem::swap(&mut self.cache, &mut state.cache);
        std::mem::swap(&mut self.next_position, &mut state.next_position);
        std::mem::swap(&mut self.mrope_decode_delta, &mut state.mrope_decode_delta);
        std::mem::swap(&mut self.text_arithmetic, &mut state.text_arithmetic);
        std::mem::swap(&mut self.kv_cache_tokens, &mut state.kv_cache_tokens);
        std::mem::swap(
            &mut self.kv_cache_lane_capacity_tokens,
            &mut state.kv_cache_lane_capacity_tokens,
        );
        std::mem::swap(
            &mut self.kv_cache_allocation_tokens,
            &mut state.kv_cache_allocation_tokens,
        );
        std::mem::swap(
            &mut self.fixed_cache_allocation_tokens,
            &mut state.fixed_cache_allocation_tokens,
        );
        if result.is_ok() {
            state.has_embedded_prompt |= uses_embeddings;
            state.resume_invalidated = false;
            state.revision = state.revision.wrapping_add(1);
        } else if state.next_position == 0 {
            state.mrope_decode_delta = None;
        }
        result
    }

    fn checkpoint_session_state(
        &self,
        state: &PipelineStageSessionState,
    ) -> Result<PipelineStageSessionCheckpoint> {
        anyhow::ensure!(
            Arc::ptr_eq(&state.kv_token_budget, &self.kv_token_budget),
            "session cache belongs to a different pipeline stage"
        );
        let cache = state
            .cache
            .iter()
            .map(|entry| {
                entry
                    .as_ref()
                    .map(architectures::checkpoint_layer_cache)
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(PipelineStageSessionCheckpoint {
            cache,
            next_position: state.next_position,
            kv_cache_tokens: state.kv_cache_tokens,
            kv_cache_lane_capacity_tokens: state.kv_cache_lane_capacity_tokens,
            kv_cache_allocation_tokens: state.kv_cache_allocation_tokens,
            fixed_cache_allocation_tokens: state.fixed_cache_allocation_tokens,
            mrope_decode_delta: state.mrope_decode_delta,
            text_arithmetic: state.text_arithmetic,
            has_embedded_prompt: state.has_embedded_prompt,
            resume_invalidated: state.resume_invalidated,
        })
    }

    fn restore_session_checkpoint(
        &mut self,
        state: &mut PipelineStageSessionState,
        checkpoint: PipelineStageSessionCheckpoint,
    ) {
        self.detach_decode_batch(state);
        state
            .kv_token_budget
            .replace(state.kv_cache_tokens, checkpoint.kv_cache_tokens);
        state.kv_token_budget.replace_allocated(
            state.kv_cache_allocation_tokens + state.fixed_cache_allocation_tokens,
            checkpoint.kv_cache_allocation_tokens + checkpoint.fixed_cache_allocation_tokens,
        );
        state.cache = checkpoint.cache;
        state.next_position = checkpoint.next_position;
        state.kv_cache_tokens = checkpoint.kv_cache_tokens;
        state.kv_cache_lane_capacity_tokens = checkpoint.kv_cache_lane_capacity_tokens;
        state.kv_cache_allocation_tokens = checkpoint.kv_cache_allocation_tokens;
        state.fixed_cache_allocation_tokens = checkpoint.fixed_cache_allocation_tokens;
        state.mrope_decode_delta = checkpoint.mrope_decode_delta;
        state.text_arithmetic = checkpoint.text_arithmetic;
        state.has_embedded_prompt = checkpoint.has_embedded_prompt;
        state.resume_invalidated = checkpoint.resume_invalidated;
        state.revision = state.revision.wrapping_add(1);
    }

    /// Prefill the target and feed the selected target rows into DFlash.
    pub(crate) fn speculative_prefill_with_session(
        &mut self,
        state: &mut PipelineStageSessionState,
        input: StageInput,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        let layer_ids = self
            .speculative
            .as_ref()
            .context("no speculative draft is attached")?
            .target_layers()
            .to_vec();
        let checkpoint = self.checkpoint_session_state(state)?;
        let draft_checkpoint = state.speculative_cache.clone();
        let result = (|| {
            let (output, captures) = self.forward_with_session_pass(
                state,
                input,
                seq_len,
                start_pos,
                ForwardPass::Standard {
                    layer_ids: &layer_ids,
                    all_logits: false,
                },
            )?;
            let draft = self.speculative.as_ref().unwrap();
            let cache = state
                .speculative_cache
                .as_mut()
                .context("session has no DFlash cache")?;
            draft.inject(cache, &captures, start_pos)?;
            Ok(output)
        })();
        if result.is_err() {
            // Target features are local to this pass; neither cache may keep
            // their tokens when the draft rejects or cannot store them.
            self.restore_session_checkpoint(state, checkpoint);
            state.speculative_cache = draft_checkpoint;
        }
        result
    }

    /// Draft and verify one block. The target state remains tentative until
    /// `commit_speculative_verification` or `abort_speculative_verification`.
    /// Return `None` before drafting when the block would exceed context or
    /// require cache eviction, so the caller can run the next scalar token.
    pub(crate) fn begin_speculative_verification(
        &mut self,
        state: &mut PipelineStageSessionState,
        anchor: u32,
        block_size: usize,
    ) -> Result<Option<SpeculativeVerification>> {
        if !self.preflight_speculative_session(state, block_size)? {
            return Ok(None);
        }
        let start_pos = state.next_position;
        let draft = self
            .speculative
            .as_ref()
            .context("no speculative draft is attached")?;
        let draft_cache = state
            .speculative_cache
            .as_ref()
            .context("session has no DFlash cache")?;
        let draft_started = Instant::now();
        let proposed = draft.draft(draft_cache, anchor, start_pos, block_size)?;
        let draft_time = draft_started.elapsed();
        anyhow::ensure!(
            proposed.len() + 1 == block_size,
            "DFlash draft block length changed"
        );
        let layer_ids = draft.target_layers().to_vec();
        let mut inputs = Vec::with_capacity(proposed.len() + 1);
        inputs.push(anchor);
        inputs.extend_from_slice(&proposed);
        let target_started = Instant::now();
        let checkpoint = self.checkpoint_session_state(state)?;
        let verify = || {
            self.forward_with_session_pass(
                state,
                StageInput::Tokens(inputs.clone()),
                inputs.len(),
                start_pos,
                ForwardPass::Standard {
                    layer_ids: &layer_ids,
                    all_logits: true,
                },
            )
        };
        #[cfg(target_os = "macos")]
        let result = crate::architectures::with_speculative_verify(verify);
        #[cfg(not(target_os = "macos"))]
        let result = {
            let mut verify = verify;
            verify()
        };
        let (output, captures) = match result {
            Ok(result) => result,
            Err(error) => {
                self.restore_session_checkpoint(state, checkpoint);
                return Err(error.context("DFlash target verification failed"));
            }
        };
        let StageOutput::Logits(logits) = output else {
            self.restore_session_checkpoint(state, checkpoint);
            anyhow::bail!("DFlash target verification returned activations");
        };
        if logits.dims() != [inputs.len(), self.vocab_size] {
            self.restore_session_checkpoint(state, checkpoint);
            anyhow::bail!("DFlash target verification logit shape changed");
        }
        Ok(Some(SpeculativeVerification {
            proposed,
            logits,
            draft_time,
            target_started,
            inputs,
            captures,
            checkpoint,
        }))
    }

    /// Keep the accepted part of the verified target block and refresh
    /// DFlash's injected cache from the target rows already computed.
    pub(crate) fn commit_speculative_verification(
        &mut self,
        state: &mut PipelineStageSessionState,
        verification: SpeculativeVerification,
        accepted_drafts: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            accepted_drafts <= verification.proposed.len(),
            "cannot commit more DFlash tokens than were proposed"
        );
        let replay_len = accepted_drafts + 1;
        let total_rows = verification.inputs.len();
        let discarded_rows = total_rows - replay_len;
        let committed_row = replay_len - 1;
        let start_pos = verification.checkpoint.next_position;
        let draft_checkpoint = state.speculative_cache.clone();
        let checkpoint = verification.checkpoint;

        let result = (|| -> Result<()> {
            for cache in state.cache.iter_mut().flatten() {
                architectures::commit_speculative_layer_cache(
                    cache,
                    committed_row,
                    discarded_rows,
                )?;
                architectures::compact_fixed_layer_cache(cache)?;
            }

            let previous_tokens = state.kv_cache_tokens;
            let retained_tokens = previous_tokens
                .checked_sub(discarded_rows)
                .context("DFlash rollback exceeds the target cache")?;
            let next_position = state
                .next_position
                .checked_sub(discarded_rows)
                .context("DFlash rollback exceeds the target position")?;
            anyhow::ensure!(
                next_position == start_pos + replay_len,
                "DFlash target rollback ended at the wrong position"
            );
            let (lane_capacity_tokens, allocation_tokens) =
                architectures::kv_cache_allocation_tokens(&state.cache)?;
            state
                .kv_token_budget
                .replace(previous_tokens, retained_tokens);
            state
                .kv_token_budget
                .replace_allocated(state.kv_cache_allocation_tokens, allocation_tokens);
            state.next_position = next_position;
            state.kv_cache_tokens = retained_tokens;
            state.kv_cache_lane_capacity_tokens = lane_capacity_tokens;
            state.kv_cache_allocation_tokens = allocation_tokens;
            state.revision = state.revision.wrapping_add(1);

            let captures = verification
                .captures
                .iter()
                .map(|capture| capture.narrow(0, 0, replay_len))
                .collect::<candle_core::Result<Vec<_>>>()?;
            let draft = self
                .speculative
                .as_ref()
                .context("no speculative draft is attached")?;
            let cache = state
                .speculative_cache
                .as_mut()
                .context("session has no DFlash cache")?;
            draft.inject(cache, &captures, start_pos)?;
            Ok(())
        })();

        if let Err(error) = result {
            self.restore_session_checkpoint(state, checkpoint);
            state.speculative_cache = draft_checkpoint;
            return Err(error);
        }
        Ok(())
    }

    pub(crate) fn abort_speculative_verification(
        &mut self,
        state: &mut PipelineStageSessionState,
        verification: SpeculativeVerification,
    ) {
        self.restore_session_checkpoint(state, verification.checkpoint);
    }

    /// Drop all cache data held by a logical session.
    pub fn clear_session_state(&mut self, state: &mut PipelineStageSessionState) {
        self.detach_decode_batch(state);
        state.clear();
    }

    fn detach_decode_batch(&mut self, state: &mut PipelineStageSessionState) {
        state.decode_batch_cache = None;
        self.decode_batch_caches.retain(|cache_ids, cache| {
            cache.strong_count() > 0 && !cache_ids.contains(&state.cache_id)
        });
    }

    /// Number of token positions currently cached for a logical session.
    pub fn session_kv_cache_tokens(&self, state: &PipelineStageSessionState) -> usize {
        state.kv_cache_tokens
    }

    /// Absolute token position expected by a logical session's next pass.
    pub fn session_next_position(&self, state: &PipelineStageSessionState) -> usize {
        state.next_position
    }

    /// Whether this solo stage's architecture supports batched decode.
    pub fn supports_decode_batch(&self) -> bool {
        self.position == StagePosition::Solo && self.arch.supports_decode_batch()
    }

    /// Split-stage batching is a distinct capability. It does not enable
    /// multi-request scheduling or change admission on another stage.
    pub fn supports_pipeline_decode_batch(&self) -> bool {
        self.position != StagePosition::Solo && self.arch.supports_pipeline_decode_batch()
    }

    /// Whether uniform-position groups can reuse packed KV storage. When
    /// false, the scheduler should batch all eligible positions together.
    pub fn decode_batch_uses_packed_cache(&self) -> bool {
        self.arch.decode_batch_uses_packed_cache()
    }

    /// Whether one more token can be decoded in a batch at the given retained
    /// cache length without crossing the configured per-session limit.
    pub fn can_decode_batch_at(&self, retained_cache_tokens: usize) -> bool {
        self.supports_decode_batch()
            && self
                .max_kv_tokens
                .is_none_or(|max| retained_cache_tokens.saturating_add(1) <= max)
    }

    fn can_decode_batch_with_session(&self, state: &PipelineStageSessionState) -> bool {
        // Packed and ragged batch cache code uses storage offsets as its
        // position metadata. Once eviction makes those offsets differ from
        // absolute RoPE positions, use the scalar path.
        state.mrope_decode_delta.is_none()
            && state.next_position == state.kv_cache_tokens
            && (self.supports_decode_batch() || self.supports_pipeline_decode_batch())
            && self
                .max_kv_tokens
                .is_none_or(|max| state.kv_cache_tokens.saturating_add(1) <= max)
            && (!self.arch.decode_batch_requires_preallocated_attention()
                || state.cache.iter().all(|layer| {
                    matches!(
                        layer,
                        None | Some(architectures::LayerCache::Attention(
                            architectures::KvCache::PreallocatedF16(_)
                        ))
                    )
                }))
    }

    /// Execute a decode batch while keeping every lane's cache independently
    /// owned. Architectures may use a faster packed path for uniform positions
    /// and a ragged path for mixed positions.
    ///
    /// Validation errors leave sessions unchanged. Once model execution starts,
    /// an error clears every selected session: in-place cache writes may have
    /// partly completed, so callers must prefill again before retrying.
    pub fn forward_decode_batch_with_sessions(
        &mut self,
        states: &mut [&mut PipelineStageSessionState],
        tokens: &[u32],
    ) -> Result<Vec<candle_core::Tensor>> {
        self.forward_batch_with_sessions(states, DecodeBatchInputs::Solo(tokens))?
            .into_iter()
            .map(|output| match output {
                StageOutput::Logits(logits) => Ok(logits),
                StageOutput::Activations(_) => unreachable!("batch output kind was checked"),
            })
            .collect()
    }

    /// Run one decode step for independent split-stage sessions. Heads take
    /// one token per input and return activations; middle/tail stages take one
    /// flat F32 activation row per input. Tail stages return logits. No wait is
    /// added to form a batch. Callers must keep each output paired with its
    /// request when passing activations between stages.
    ///
    /// Uses the same cache budget and execution-error rules as solo batching.
    /// Validation leaves sessions intact; execution errors clear every selected
    /// session. The caller must abort those requests, not retry stale tokens.
    pub fn forward_pipeline_decode_batch_with_sessions(
        &mut self,
        states: &mut [&mut PipelineStageSessionState],
        inputs: &[StageInput],
    ) -> Result<Vec<StageOutput>> {
        self.forward_batch_with_sessions(states, DecodeBatchInputs::Pipeline(inputs))
    }

    fn forward_batch_with_sessions(
        &mut self,
        states: &mut [&mut PipelineStageSessionState],
        inputs: DecodeBatchInputs<'_>,
    ) -> Result<Vec<StageOutput>> {
        match &inputs {
            DecodeBatchInputs::Solo(tokens) => {
                anyhow::ensure!(
                    self.supports_decode_batch(),
                    "decode batching is unsupported"
                );
                anyhow::ensure!(states.len() == tokens.len(), "decode batch lane mismatch");
                self.validate_token_ids(tokens)?;
            }
            DecodeBatchInputs::Pipeline(inputs) => {
                anyhow::ensure!(
                    self.supports_pipeline_decode_batch(),
                    "pipeline decode batching is unsupported"
                );
                anyhow::ensure!(
                    states.len() == inputs.len(),
                    "pipeline decode batch lane mismatch"
                );
                for input in *inputs {
                    match (self.position, input) {
                        (StagePosition::Head, StageInput::Tokens(tokens)) => anyhow::ensure!(
                            tokens.len() == 1 && (tokens[0] as usize) < self.vocab_size,
                            "pipeline decode batch needs one valid head token"
                        ),
                        (
                            StagePosition::Middle | StagePosition::Tail,
                            StageInput::Activations {
                                tensor,
                                bidirectional_spans,
                                positions,
                            },
                        ) => anyhow::ensure!(
                            tensor.dims() == [1, self.hidden_dim]
                                && tensor.dtype() == candle_core::DType::F32
                                && tensor.device().same_device(&self.device)
                                && bidirectional_spans.is_empty()
                                && positions.is_none(),
                            "pipeline decode batch needs one flat F32 activation row without position metadata"
                        ),
                        _ => anyhow::bail!("pipeline decode batch input does not match stage role"),
                    }
                }
            }
        }
        anyhow::ensure!(!states.is_empty(), "decode batch cannot be empty");
        // Batch packing can keep several lane allocations alive. Drop optional
        // snapshots first if there is not room for the next batch allocation.
        let additional = states
            .iter()
            .map(|state| state.kv_cache_lane_capacity_tokens.saturating_add(64))
            .fold(0usize, usize::saturating_add)
            .saturating_mul(states.len())
            .saturating_add(states.iter().fold(0usize, |total, state| {
                total.saturating_add(
                    self.fixed_cache_budget_tokens()
                        .saturating_sub(state.fixed_cache_allocation_tokens),
                )
            }));
        self.release_prompt_cache_under_pressure(additional);
        let positions = states
            .iter()
            .map(|state| state.next_position)
            .collect::<Vec<_>>();
        if states
            .iter()
            .any(|state| !self.can_decode_batch_with_session(state))
        {
            anyhow::bail!(
                "decode batch requires per-session KV eviction or scalar cache preparation"
            );
        }
        anyhow::ensure!(
            states
                .iter()
                .all(|state| Arc::ptr_eq(&state.kv_token_budget, &self.kv_token_budget)),
            "decode batch contains a session from another pipeline stage"
        );
        let previous_allocated = states.iter().try_fold(0usize, |total, state| {
            total
                .checked_add(state.kv_cache_allocation_tokens)
                .and_then(|total| total.checked_add(state.fixed_cache_allocation_tokens))
                .ok_or_else(|| anyhow::anyhow!("decode batch KV allocation overflow"))
        })?;
        let fixed_per_state = self.fixed_cache_budget_tokens();
        let fixed_selected = fixed_per_state
            .checked_mul(states.len())
            .context("decode batch fixed state allocation overflow")?;
        // Check even without a configured budget; success below advances
        // every lane by one and must never wrap its absolute position.
        let required_lane_tokens = positions
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("decode batch KV position overflow"))?;
        if let Some(max) = self.max_total_kv_tokens {
            let required = self
                .total_kv_cache_tokens()
                .checked_add(states.len())
                .ok_or_else(|| anyhow::anyhow!("aggregate KV cache length overflow"))?;
            anyhow::ensure!(
                required <= max,
                "decode batch requires per-session KV eviction"
            );

            let total_allocated = self.kv_token_budget.allocated();
            anyhow::ensure!(
                total_allocated >= previous_allocated,
                "aggregate KV allocation accounting is smaller than the decode batch"
            );
            let target_selected = if self.arch.decode_batch_uses_packed_cache() {
                let current_lane_capacity = states
                    .iter()
                    .map(|state| state.kv_cache_lane_capacity_tokens)
                    .max()
                    .unwrap_or(0);
                let target_lane_capacity = if self.preallocated_kv_cache {
                    if current_lane_capacity == 0 || required_lane_tokens > current_lane_capacity {
                        architectures::PreallocatedKvCache::next_capacity(
                            current_lane_capacity,
                            required_lane_tokens,
                        )?
                    } else {
                        current_lane_capacity
                    }
                } else {
                    required_lane_tokens
                };
                // A narrow lane view keeps the whole packed allocation alive. Charge
                // each lane for that full allocation so regrouping cannot make old
                // packed storage disappear from the budget before it is freed.
                let target_per_state = target_lane_capacity
                    .checked_mul(states.len())
                    .ok_or_else(|| anyhow::anyhow!("decode batch KV allocation overflow"))?;
                target_per_state
                    .checked_mul(states.len())
                    .ok_or_else(|| anyhow::anyhow!("decode batch KV allocation overflow"))?
            } else {
                states.iter().try_fold(0usize, |total, state| {
                    let required = state
                        .next_position
                        .checked_add(1)
                        .ok_or_else(|| anyhow::anyhow!("decode batch KV position overflow"))?;
                    let capacity = state.kv_cache_lane_capacity_tokens;
                    let target = if self.preallocated_kv_cache {
                        if capacity == 0 || required > capacity {
                            architectures::PreallocatedKvCache::next_capacity(capacity, required)?
                        } else {
                            capacity
                        }
                    } else {
                        required
                    };
                    total
                        .checked_add(target)
                        .ok_or_else(|| anyhow::anyhow!("decode batch KV allocation overflow"))
                })?
            };
            let required_allocated = total_allocated
                .checked_sub(previous_allocated)
                .and_then(|other| other.checked_add(target_selected))
                .and_then(|other| other.checked_add(fixed_selected))
                .ok_or_else(|| anyhow::anyhow!("decode batch KV allocation overflow"))?;
            anyhow::ensure!(
                required_allocated <= max,
                "decode batch needs {required_allocated} resident KV slots but the limit is {max}; retry with scalar decode"
            );
        }
        let cache_ids = states
            .iter()
            .map(|state| state.cache_id)
            .collect::<Vec<_>>();
        let revisions = states
            .iter()
            .map(|state| state.revision)
            .collect::<Vec<_>>();
        let reusable = self
            .decode_batch_caches
            .remove(&cache_ids)
            .and_then(|cache| cache.upgrade());
        for state in states.iter_mut() {
            state.decode_batch_cache = None;
        }
        // Drop dead lookup entries without retaining model buffers. Normally
        // all owners of a reusable group are the selected sessions above.
        // If another owner remains, repack from lane views rather than mutate
        // shared metadata.
        self.decode_batch_caches
            .retain(|_, cache| cache.strong_count() > 0);
        let mut persistent = reusable
            .filter(|cache| cache.revisions == revisions)
            .and_then(|cache| Arc::try_unwrap(cache).ok())
            .unwrap_or_else(|| PersistentDecodeBatch {
                layer_caches: (0..self.cache.len()).map(|_| None).collect(),
                revisions: revisions.clone(),
            });
        let mut caches = states
            .iter_mut()
            .map(|state| state.cache.as_mut_slice())
            .collect::<Vec<_>>();
        let result = match inputs {
            DecodeBatchInputs::Solo(tokens) => self
                .arch
                .forward_decode_batch(
                    tokens,
                    &mut caches,
                    &mut persistent.layer_caches,
                    &positions,
                )
                .map(|outputs| outputs.into_iter().map(StageOutput::Logits).collect()),
            DecodeBatchInputs::Pipeline(inputs) => self.arch.forward_pipeline_decode_batch(
                inputs,
                &mut caches,
                &mut persistent.layer_caches,
                &positions,
            ),
        };
        match result {
            Ok(outputs) => {
                if outputs.len() != states.len() {
                    self.discard_decode_batch_states(states);
                    anyhow::bail!(
                        "decode batch output count differs: expected {}, received {}",
                        states.len(),
                        outputs.len()
                    );
                }
                // Split callers must never send logits as hidden states (or
                // sample an activation row). Check before publishing positions.
                let valid_outputs = outputs.iter().all(|output| match (self.position, output) {
                    (StagePosition::Solo, StageOutput::Logits(_)) => true,
                    (StagePosition::Tail, StageOutput::Logits(tensor)) => {
                        tensor.dims() == [self.vocab_size]
                            && tensor.dtype() == candle_core::DType::F32
                            && tensor.device().same_device(&self.device)
                    }
                    (
                        StagePosition::Head | StagePosition::Middle,
                        StageOutput::Activations(tensor),
                    ) => {
                        tensor.dims() == [1, self.hidden_dim]
                            && tensor.dtype() == candle_core::DType::F32
                            && tensor.device().same_device(&self.device)
                    }
                    _ => false,
                });
                if !valid_outputs {
                    self.discard_decode_batch_states(states);
                    anyhow::bail!(
                        "pipeline decode batch returned an invalid output kind, shape, dtype, or device"
                    );
                }
                for state in states.iter_mut() {
                    for cache in state.cache.iter_mut().flatten() {
                        if let Err(error) = architectures::compact_fixed_layer_cache(cache) {
                            self.discard_decode_batch_states(states);
                            return Err(error.context("could not compact decode batch state"));
                        }
                    }
                }
                let measured = states
                    .iter()
                    .map(|state| architectures::kv_cache_allocation_tokens(&state.cache))
                    .collect::<Result<Vec<_>>>();
                let measured = match measured {
                    Ok(measured) => measured,
                    Err(error) => {
                        self.discard_decode_batch_states(states);
                        return Err(error.context("could not measure decode batch KV allocation"));
                    }
                };
                let next_allocated =
                    match measured
                        .iter()
                        .try_fold(fixed_selected, |total, (_, slots)| {
                            total.checked_add(*slots).ok_or_else(|| {
                                anyhow::anyhow!("decode batch KV allocation overflow")
                            })
                        }) {
                        Ok(total) => total,
                        Err(error) => {
                            self.discard_decode_batch_states(states);
                            return Err(error);
                        }
                    };
                if let Some(max) = self.max_total_kv_tokens {
                    let required = match self
                        .kv_token_budget
                        .allocated()
                        .checked_sub(previous_allocated)
                        .and_then(|other| other.checked_add(next_allocated))
                    {
                        Some(required) => required,
                        None => {
                            self.discard_decode_batch_states(states);
                            anyhow::bail!("decode batch KV allocation overflow");
                        }
                    };
                    if required > max {
                        self.discard_decode_batch_states(states);
                        anyhow::bail!(
                            "decode batch exceeded the resident KV allocation limit after execution"
                        );
                    }
                }
                self.kv_token_budget.add(states.len());
                self.kv_token_budget
                    .replace_allocated(previous_allocated, next_allocated);
                for ((state, position), (lane_capacity, allocation_tokens)) in
                    states.iter_mut().zip(&positions).zip(measured)
                {
                    state.next_position = position + 1;
                    state.kv_cache_tokens += 1;
                    state.kv_cache_lane_capacity_tokens = lane_capacity;
                    state.kv_cache_allocation_tokens = allocation_tokens;
                    state.fixed_cache_allocation_tokens = fixed_per_state;
                    state.resume_invalidated = false;
                    state.revision = state.revision.wrapping_add(1);
                }
                self.decode_batch_caches.retain(|existing_ids, _| {
                    existing_ids
                        .iter()
                        .all(|existing| !cache_ids.contains(existing))
                });
                persistent.revisions = states.iter().map(|state| state.revision).collect();
                let persistent = Arc::new(persistent);
                for state in states {
                    state.decode_batch_cache = Some(persistent.clone());
                }
                self.decode_batch_caches
                    .insert(cache_ids, Arc::downgrade(&persistent));
                Ok(outputs)
            }
            Err(error) => {
                // A failed layer can leave either lane views or their packed
                // backing partly updated. Neither is safe for a scalar retry.
                self.discard_decode_batch_states(states);
                Err(error)
            }
        }
    }

    fn discard_decode_batch_states(&mut self, states: &mut [&mut PipelineStageSessionState]) {
        for state in states {
            self.clear_session_state(state);
        }
    }

    /// Clear all layer caches (KV cache, DeltaNet state, etc.).
    pub fn clear_cache(&mut self) {
        self.prompt_cache.entries.clear();
        self.kv_token_budget.release(self.kv_cache_tokens);
        self.kv_token_budget.release_allocated(
            self.kv_cache_allocation_tokens + self.fixed_cache_allocation_tokens,
        );
        self.clear_cache_storage();
        if let Some(cache) = self.speculative_cache.as_mut() {
            cache.clear();
        }
        self.next_position = 0;
        self.mrope_decode_delta = None;
        self.text_arithmetic = None;
        self.kv_cache_tokens = 0;
        self.cache_token_ids.clear();
        self.cache_fingerprint = None;
        self.kv_cache_lane_capacity_tokens = 0;
        self.kv_cache_allocation_tokens = 0;
        self.fixed_cache_allocation_tokens = 0;
    }

    /// Validate a caller-provided prefix hint against the stage-local
    /// cache and return the number of tokens that may be skipped safely.
    ///
    /// Prefix lengths are advisory.  Reuse is allowed only when the cache is
    /// at the exact absolute position requested, has not evicted any of the
    /// prefix, and the locally recorded token IDs match byte-for-byte.  Any
    /// mismatch clears the stage cache and forces a cold prefill, which keeps
    /// a stale cached prefix from changing model output.
    pub fn validate_cached_prefix(&mut self, tokens: &[u32], requested: usize) -> usize {
        if requested == 0 {
            if self.next_position != 0 || self.kv_cache_tokens != 0 {
                self.clear_cache();
            }
            return 0;
        }

        let valid = requested < tokens.len()
            && self.next_position == requested
            && self.kv_cache_tokens == requested
            && self.cache_token_ids.len() >= requested
            && self.cache_token_ids[..requested] == tokens[..requested];
        if valid {
            return requested;
        }

        tracing::debug!(
            requested,
            prompt_tokens = tokens.len(),
            next_position = self.next_position,
            retained_tokens = self.kv_cache_tokens,
            recorded_tokens = self.cache_token_ids.len(),
            "rejecting unverified KV prefix; restarting with a cold prefill"
        );
        self.clear_cache();
        0
    }

    /// Record the token identity represented by a successfully completed
    /// stage cache.  The sampled final token is intentionally truncated when
    /// it has not yet been fed through the model; `next_position` is the
    /// source of truth for the reusable cache boundary.
    pub fn record_cache_identity(&mut self, prompt_tokens: &[u32], generated: &[u32]) {
        let cache_len = self.next_position;
        let total_len = prompt_tokens.len().saturating_add(generated.len());
        if cache_len == 0 || cache_len > total_len {
            self.cache_token_ids.clear();
            self.cache_fingerprint = None;
            return;
        }

        let mut identity = Vec::with_capacity(cache_len);
        identity.extend_from_slice(prompt_tokens);
        identity.extend_from_slice(generated);
        identity.truncate(cache_len);
        self.cache_fingerprint = Some(hash_cache_token_ids(&identity));
        if self.kv_cache_tokens == cache_len {
            self.cache_token_ids = identity;
        } else {
            // A rolling cache can still complete successfully, but it is not
            // eligible for prefix reuse because earlier token IDs were
            // evicted.  Keep only the full-sequence digest for this request's
            // completion fingerprint.
            self.cache_token_ids.clear();
        }
    }

    /// Return the identity digest for the current reusable cache, if complete.
    pub fn cache_fingerprint(&self) -> Option<[u8; 32]> {
        self.cache_fingerprint
    }

    /// Invalidate only the reusable token identity while retaining the live
    /// KV state for the current request.  This is used for embedding prompts:
    /// their token count may match the text-token vector, but the image/audio
    /// tensor content is not represented by token IDs and must never be
    /// advertised as a reusable text prefix.
    pub fn invalidate_cache_identity(&mut self) {
        self.cache_token_ids.clear();
        self.cache_fingerprint = None;
    }

    /// Check whether this stage's cache matches a digest of the expected
    /// prefix, without needing the prefix token IDs.
    ///
    /// The stage must have retained the complete absolute prefix and must have
    /// recorded a digest at the end of its previous clean request.  Stages
    /// that cannot prove this are forced through the cold path.
    pub fn cache_matches_fingerprint(&self, cache_len: usize, fingerprint: &[u8; 32]) -> bool {
        cache_len > 0
            && self.next_position == cache_len
            && self.kv_cache_tokens == cache_len
            && self.cache_fingerprint == Some(*fingerprint)
    }

    /// Record a cache digest computed by the first stage after a request
    /// completed cleanly.
    ///
    /// Stages that receive activations rather than token IDs cannot compute
    /// the digest themselves; the digest plus exact position/retention checks
    /// identify the cached prefix.  Record it only after every stage finished
    /// the request successfully.
    pub fn record_cache_fingerprint(&mut self, cache_len: usize, fingerprint: [u8; 32]) {
        if cache_len > 0 && self.next_position == cache_len && self.kv_cache_tokens == cache_len {
            self.cache_fingerprint = Some(fingerprint);
            if self.cache_token_ids.len() > cache_len {
                self.cache_token_ids.truncate(cache_len);
            }
        } else {
            self.cache_token_ids.clear();
            self.cache_fingerprint = None;
        }
    }

    /// Whether this stage has a complete, non-evicted token identity for the
    /// current cache.  Pipeline stages use this when deciding whether they
    /// can report a reusable prefix at request completion.
    pub fn has_cache_identity(&self) -> bool {
        self.next_position > 0
            && self.kv_cache_tokens == self.next_position
            && self.cache_token_ids.len() == self.next_position
            && self.cache_fingerprint.is_some()
    }

    fn clear_cache_storage(&mut self) {
        self.decode_batch_caches.clear();
        for entry in self.cache.iter_mut() {
            *entry = None;
        }
    }

    /// Run warmup forward passes to pre-compile Metal/CUDA shaders.
    ///
    /// This eliminates the ~12s cold-start penalty on the first real inference
    /// by triggering shader compilation for both prefill and decode tensor shapes.
    /// Forces GPU synchronization to ensure compilation completes before returning.
    pub fn warmup(&mut self) -> Result<()> {
        let uses_activations = matches!(self.position, StagePosition::Middle | StagePosition::Tail);
        let hidden_dim = self.hidden_dim;
        let device = self.device.clone();

        #[cfg(target_os = "macos")]
        architectures::precompile_metal_kv_append(&device)?;

        #[cfg(target_os = "macos")]
        if matches!(self.position, StagePosition::Solo | StagePosition::Tail) {
            sampling_metal::precompile(&device)?;
        }

        let make_input = |seq_len: usize| -> Result<StageInput> {
            if uses_activations {
                let dummy = candle_core::Tensor::zeros(
                    (seq_len, hidden_dim),
                    candle_core::DType::F32,
                    &device,
                )?;
                Ok(StageInput::activations(dummy))
            } else {
                Ok(StageInput::Tokens(vec![0u32; seq_len]))
            }
        };

        // Warmup with a short prefill to compile prefill shaders
        let input_prefill = make_input(4)?;
        let output = self.forward(input_prefill, 4, 0)?;
        self.device.synchronize()?;
        drop(output);

        // Warmup with single-token decode to compile decode-specific shaders
        let input_decode = make_input(1)?;
        let output = self.forward(input_decode, 1, 4)?;
        self.device.synchronize()?;
        drop(output);

        // Clear KV cache so real inference starts fresh
        self.clear_cache();
        Ok(())
    }

    /// Position occupied by this stage in the model pipeline.
    pub fn position(&self) -> StagePosition {
        self.position
    }

    /// Model hidden-state width carried by this stage.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Model vocabulary size reported by the loaded architecture.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Check token IDs before a request can change a cache or reach a gather kernel.
    fn validate_token_ids(&self, tokens: &[u32]) -> Result<()> {
        if let Some(id) = tokens.iter().find(|&&id| id as usize >= self.vocab_size) {
            anyhow::bail!("token id {id} is outside the model vocabulary");
        }
        Ok(())
    }

    fn validate_token_input(&self, input: &StageInput, seq_len: usize) -> Result<()> {
        if let StageInput::Tokens(tokens) = input {
            anyhow::ensure!(
                tokens.len() == seq_len,
                "token count {} differs from sequence length {seq_len}",
                tokens.len()
            );
            self.validate_token_ids(tokens)?;
        }
        Ok(())
    }

    fn validate_sampling_params(&self, params: &SamplingParams) -> Result<()> {
        params.validate()?;
        if let Some(constraint) = &params.tool_constraint {
            constraint.validate_vocab_size(self.vocab_size)?;
        }
        if let Some(token_id) = params
            .logit_bias
            .keys()
            .find(|&&id| (id as usize) >= self.vocab_size)
        {
            anyhow::bail!("logit bias token id {token_id} is outside the model vocabulary");
        }
        Ok(())
    }

    /// Normalized architecture family name.
    pub fn family(&self) -> &str {
        self.arch.family()
    }

    /// Device that owns this stage's model tensors.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Embed token IDs using the head stage's token embedding table.
    pub fn embed_tokens(&self, tokens: &[u32]) -> Result<candle_core::Tensor> {
        self.validate_token_ids(tokens)?;
        self.arch.embed_tokens(tokens)
    }

    /// Whether single-token activations passed between stages should be F32.
    pub fn prefers_f32_decode_activations(&self) -> bool {
        self.arch.prefers_f32_decode_activations()
    }

    /// Whether multi-token prefill activations passed between stages should
    /// use F32. This is separate from the decode preference because the default
    /// prefill format remains F16 for size, while architectures carrying
    /// numerically sensitive recurrent state may require F32 in both phases.
    pub fn prefers_f32_prefill_activations(&self) -> bool {
        self.arch.prefers_f32_prefill_activations()
    }

    /// KV cache cost per token for this stage's layer range (bytes).
    /// Used for reporting to MM so it can compute context headroom.
    pub fn kv_bytes_per_token(&self) -> u64 {
        self.kv_bytes_per_token
    }

    /// Per-slot budget multiplier used for cache admission, before the stage's
    /// additional copy reserve. This is not a measurement of peak residency.
    pub fn kv_budget_bytes_per_token(&self) -> u64 {
        self.kv_budget_bytes_per_token
    }

    /// Number of token positions currently in the KV cache.
    pub fn kv_cache_tokens(&self) -> usize {
        self.kv_cache_tokens
    }

    /// Absolute token position expected by the next forward pass.
    pub fn next_position(&self) -> usize {
        self.next_position
    }

    /// Reject a request before prefill when its prompt and output allowance
    /// exceed the model's position limit. This does not replace KV admission.
    pub fn validate_context_budget(&self, prompt_tokens: usize, max_tokens: usize) -> Result<()> {
        anyhow::ensure!(
            prompt_tokens
                .checked_add(max_tokens)
                .is_some_and(|end| end <= self.context_length),
            "request context exceeds model limit: {prompt_tokens} prompt + {max_tokens} completion > {}",
            self.context_length
        );
        Ok(())
    }

    /// Maximum token position span declared by this stage's model.
    pub fn context_length(&self) -> usize {
        self.context_length
    }

    /// Total retained tokens across the stage cache and every live session.
    pub fn total_kv_cache_tokens(&self) -> usize {
        self.kv_token_budget.total()
    }

    /// Resident attention and fixed request state in KV-budget token units.
    pub fn total_kv_cache_allocation_tokens(&self) -> usize {
        self.kv_token_budget.allocated()
    }

    /// Resident half of the configured KV headroom, after the copy reserve.
    pub fn max_resident_kv_allocation_tokens(&self) -> Option<usize> {
        self.max_total_kv_tokens
    }

    /// Total configured KV allocation headroom, including copy reserve, in
    /// token-slot units.
    pub fn max_total_kv_allocation_headroom_tokens(&self) -> Option<usize> {
        self.max_total_kv_headroom_tokens
    }

    /// Set the maximum KV cache token capacity. Enables automatic eviction.
    pub fn set_max_kv_tokens(&mut self, max: usize) {
        if self.max_kv_tokens != Some(max) {
            self.clear_prompt_cache();
        }
        self.max_kv_tokens = Some(max);
    }

    /// Set total KV allocation headroom shared by the stage and all sessions.
    ///
    /// Half remains available for resident KV and fixed request state. The other half is held
    /// for copy-on-grow, compaction, and batch packing because those paths keep
    /// old and new buffers live during one forward pass.
    pub fn set_max_total_kv_tokens(&mut self, max: usize) -> Result<()> {
        anyhow::ensure!(
            max >= 2,
            "aggregate KV allocation headroom must hold at least two token slots"
        );
        let resident_limit = max / 2;
        let retained = self.total_kv_cache_tokens();
        let allocated = self.total_kv_cache_allocation_tokens();
        anyhow::ensure!(
            retained <= resident_limit && allocated <= resident_limit,
            "cannot set resident KV allocation limit to {resident_limit} with {retained} tokens retained and {allocated} slots allocated"
        );
        self.max_total_kv_tokens = Some(resident_limit);
        self.max_total_kv_headroom_tokens = Some(max);
        self.configure_prompt_cache_headroom(resident_limit);
        Ok(())
    }

    /// Evict the oldest `count` token positions from all attention layer caches.
    /// DeltaNet and Mamba-2 caches are fixed-size and unaffected.
    pub fn evict_oldest(&mut self, count: usize) {
        if self.arch.family() == "deepseek41"
            || self
                .cache
                .iter()
                .any(|c| matches!(c, Some(LayerCache::DeepSeek41(_))))
        {
            if count > 0 {
                tracing::warn!("Maya outer eviction rejected; request state preserved");
            }
            return;
        }
        if let Err(error) = self.try_evict_oldest(count) {
            // The public API cannot return the error. Make the documented
            // fallback true for every failure, including validation failures
            // that occur before `try_evict_oldest` touches a layer.
            self.clear_cache();
            tracing::warn!(%error, "KV eviction failed; cache reset to prevent partial reuse");
        }
    }

    fn try_evict_oldest(&mut self, count: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            self.arch.family() != "deepseek41"
                && !self
                    .cache
                    .iter()
                    .any(|c| matches!(c, Some(LayerCache::DeepSeek41(_)))),
            "Maya history cannot be evicted; request state preserved"
        );
        anyhow::ensure!(
            count <= self.kv_cache_tokens,
            "cannot evict {count} tokens from cache with {} retained tokens",
            self.kv_cache_tokens
        );
        if count == self.kv_cache_tokens {
            let allocation_tokens = self.kv_cache_allocation_tokens;
            self.decode_batch_caches.clear();
            for entry in &mut self.cache {
                if !matches!(entry, Some(LayerCache::DeltaNet(_) | LayerCache::Mamba2(_))) {
                    *entry = None;
                }
            }
            self.kv_cache_tokens = 0;
            self.kv_cache_lane_capacity_tokens = 0;
            self.kv_cache_allocation_tokens = 0;
            self.kv_token_budget.release(count);
            self.kv_token_budget.release_allocated(allocation_tokens);
            return Ok(());
        }
        anyhow::ensure!(
            !self
                .cache
                .iter()
                .any(|entry| matches!(entry, Some(LayerCache::DeepSeek4(_)))),
            "V4 compressed history cannot be partially evicted; start a new request within the cache budget"
        );
        anyhow::ensure!(
            !self
                .cache
                .iter()
                .any(|entry| matches!(entry, Some(LayerCache::Inkling2(_)))),
            "Inkling 2 visit state cannot be partially evicted; start a new request within the cache budget"
        );
        let mut eviction_error = None;
        for entry in self.cache.iter_mut() {
            if let Some(LayerCache::Attention(kv)) = entry
                && let Err(error) = kv.evict_oldest(count)
            {
                eviction_error = Some(error);
                break;
            }
            // DeltaNet and Mamba2 caches are fixed-size — skip
        }
        if let Some(error) = eviction_error {
            self.clear_cache();
            return Err(error.context("attention KV eviction failed"));
        }
        self.kv_cache_tokens -= count;
        self.kv_token_budget.release(count);
        Ok(())
    }

    /// Run full autoregressive generation (solo mode only).
    ///
    /// `stop_ids` is a list of token IDs that signal end of generation (e.g. EOS, EOT).
    /// `skip_prefix` — absolute number of prompt tokens already processed in a
    /// previous turn. It must match [`PipelineStage::next_position`] and an
    /// exact, non-evicted token identity; a rolling-evicted cache is not safe
    /// to reuse for a caller prefix hint.
    /// When > 0, prefill only `tokens[skip_prefix..]` at that absolute position.
    /// If `skip_prefix > tokens.len()`, cache is cleared and prefill starts from 0 (overflow).
    /// For pipeline mode, the caller orchestrates the loop across stages.
    pub fn generate(
        &mut self,
        tokens: Vec<u32>,
        params: &SamplingParams,
        stop_ids: &[u32],
        skip_prefix: usize,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        self.generate_inner(tokens, None, params, stop_ids, skip_prefix, &mut on_token)
    }

    /// Generate from decoder-ready prompt embeddings. `tokens` must describe
    /// the same decoder positions and remain available for repeat penalties.
    pub fn generate_with_embeddings(
        &mut self,
        tokens: Vec<u32>,
        embeddings: candle_core::Tensor,
        params: &SamplingParams,
        stop_ids: &[u32],
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        self.generate_inner(
            tokens,
            Some(EmbeddedPrompt {
                embeddings,
                bidirectional_spans: Vec::new(),
                positions: None,
            }),
            params,
            stop_ids,
            0,
            &mut on_token,
        )
    }

    /// Generate from prompt embeddings with image-token attention spans.
    pub fn generate_with_embeddings_and_spans(
        &mut self,
        tokens: Vec<u32>,
        embeddings: candle_core::Tensor,
        bidirectional_spans: Vec<AttentionSpan>,
        params: &SamplingParams,
        stop_ids: &[u32],
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        self.generate_inner(
            tokens,
            Some(EmbeddedPrompt {
                embeddings,
                bidirectional_spans,
                positions: None,
            }),
            params,
            stop_ids,
            0,
            &mut on_token,
        )
    }

    /// Generate from decoder-ready embeddings using Qwen T/H/W rotary
    /// positions. Decode tokens continue with the signed delta derived from
    /// the prefill schedule.
    pub fn generate_with_positioned_embeddings(
        &mut self,
        tokens: Vec<u32>,
        embeddings: candle_core::Tensor,
        positions: MropePositionIds,
        params: &SamplingParams,
        stop_ids: &[u32],
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        self.generate_inner(
            tokens,
            Some(EmbeddedPrompt {
                embeddings,
                bidirectional_spans: Vec::new(),
                positions: Some(positions),
            }),
            params,
            stop_ids,
            0,
            &mut on_token,
        )
    }

    /// Positioned generation with explicit image-token attention spans.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_positioned_embeddings_and_spans(
        &mut self,
        tokens: Vec<u32>,
        embeddings: candle_core::Tensor,
        positions: MropePositionIds,
        bidirectional_spans: Vec<AttentionSpan>,
        params: &SamplingParams,
        stop_ids: &[u32],
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        self.generate_inner(
            tokens,
            Some(EmbeddedPrompt {
                embeddings,
                bidirectional_spans,
                positions: Some(positions),
            }),
            params,
            stop_ids,
            0,
            &mut on_token,
        )
    }

    fn generate_inner(
        &mut self,
        tokens: Vec<u32>,
        embedded_prompt: Option<EmbeddedPrompt>,
        params: &SamplingParams,
        stop_ids: &[u32],
        skip_prefix: usize,
        on_token: &mut impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        anyhow::ensure!(
            self.position == StagePosition::Solo,
            "generation is only available in solo mode"
        );
        self.validate_sampling_params(params)?;
        self.validate_context_budget(tokens.len(), params.max_tokens)?;
        anyhow::ensure!(!tokens.is_empty(), "generation prompt must contain a token");
        self.validate_token_ids(&tokens)?;

        let profile_enabled = self.options.profile.decode;
        let sync_after_forward = self.options.profile.sync_after_forward;
        let layer_profile_enabled = architectures::profiling::layer_profiling_enabled();
        let mut profile = DecodeProfile::default();
        if layer_profile_enabled {
            architectures::deepseek2::reset_layer_profile();
            architectures::deepseek4::reset_layer_profile();
            architectures::gemma4::reset_layer_profile();
            architectures::llama::reset_layer_profile();
            architectures::minimax_m2::reset_layer_profile();
            architectures::nemotron_h::reset_layer_profile();
            architectures::phi3::reset_layer_profile();
            architectures::qwen35::reset_layer_profile();
        }

        let mut generated = Vec::new();
        let mut sampling_history = SamplingHistory::new(&tokens, params);
        let prompt_len = tokens.len();
        let mut mrope_decode_delta = None;

        if let Some(prompt) = embedded_prompt.as_ref() {
            anyhow::ensure!(
                prompt.embeddings.dim(0)? == prompt_len,
                "prompt token and embedding counts differ"
            );
            if let Some(positions) = prompt.positions.as_ref() {
                anyhow::ensure!(
                    positions.len() == prompt_len,
                    "M-RoPE schedule has {} positions for {prompt_len} prompt tokens",
                    positions.len()
                );
                mrope_decode_delta = Some(positions.decode_delta(prompt_len)?);
            }
            self.clear_cache();
        }

        let text_context = if embedded_prompt.is_none() {
            let context = TextRequestContext::new(prompt_len)?;
            self.prepare_text_request(context);
            Some(context)
        } else {
            None
        };

        // A caller-provided prefix length is only a hint.  Validate it against
        // the exact local token identity before skipping any prefill.
        let effective_skip = if embedded_prompt.is_none() {
            self.validate_cached_prefix(&tokens, skip_prefix)
        } else {
            0
        };
        if embedded_prompt.is_none() && effective_skip > 0 {
            mrope_decode_delta = self.mrope_decode_delta;
        }

        let forward_start = Instant::now();
        let output = match embedded_prompt {
            Some(prompt) => match prompt.positions {
                Some(positions) => self.prefill(
                    StageInput::positioned_activations_with_spans(
                        prompt.embeddings,
                        prompt.bidirectional_spans,
                        positions,
                    ),
                    prompt_len,
                    0,
                )?,
                None => self.prefill(
                    StageInput::activations_with_spans(
                        prompt.embeddings,
                        prompt.bidirectional_spans,
                    ),
                    prompt_len,
                    0,
                )?,
            },
            None => self.prefill_text_prompt(
                &tokens[effective_skip..],
                effective_skip,
                mrope_decode_delta,
            )?,
        };
        if sync_after_forward {
            self.device.synchronize()?;
        }
        profile.forward += forward_start.elapsed();
        let logits = match output {
            StageOutput::Logits(l) => l,
            _ => anyhow::bail!("expected logits from solo stage"),
        };

        let sampling_start = Instant::now();
        let mut rng = params
            .seed
            .map_or_else(SamplingRng::from_entropy, SamplingRng::seeded);
        let mut token = sampling_history.sample(&logits, params, &mut rng)?;
        profile.sampling += sampling_start.elapsed();
        profile.tokens += 1;
        generated.push(token);
        sampling_history.push(token);

        let callback_start = Instant::now();
        let keep_going = on_token(token);
        profile.callback += callback_start.elapsed();
        if (stop_ids.contains(&token) && generated.len() >= params.min_tokens) || !keep_going {
            if profile_enabled {
                profile.report(sync_after_forward);
            }
            if layer_profile_enabled {
                architectures::deepseek2::report_layer_profile();
                architectures::deepseek4::report_layer_profile();
                architectures::gemma4::report_layer_profile();
                architectures::llama::report_layer_profile();
                architectures::minimax_m2::report_layer_profile();
                architectures::nemotron_h::report_layer_profile();
                architectures::phi3::report_layer_profile();
                architectures::qwen35::report_layer_profile();
            }
            self.record_cache_identity(&tokens, &generated);
            return Ok(generated);
        }

        // Decode: one token at a time
        for _ in 1..params.max_tokens {
            let pos = prompt_len + generated.len() - 1;
            let forward_start = Instant::now();
            let input = if let Some(delta) = mrope_decode_delta {
                let embeddings = self.embed_tokens(&[token])?;
                let positions = MropePositionIds::text_with_delta(pos, delta, 1)?;
                StageInput::positioned_activations(embeddings, positions)
            } else {
                StageInput::Tokens(vec![token])
            };
            let output = match text_context {
                Some(context) => self.forward_text(input, 1, pos, context)?,
                None => self.forward(input, 1, pos)?,
            };
            if sync_after_forward {
                self.device.synchronize()?;
            }
            profile.forward += forward_start.elapsed();
            let logits = match output {
                StageOutput::Logits(l) => l,
                _ => anyhow::bail!("expected logits from solo stage"),
            };
            let sampling_start = Instant::now();
            let next_token = sampling_history.sample(&logits, params, &mut rng)?;
            profile.sampling += sampling_start.elapsed();
            profile.tokens += 1;

            generated.push(next_token);
            sampling_history.push(next_token);

            let callback_start = Instant::now();
            let keep_going = on_token(next_token);
            profile.callback += callback_start.elapsed();
            if (stop_ids.contains(&next_token) && generated.len() >= params.min_tokens)
                || !keep_going
            {
                break;
            }
            token = next_token;
        }

        if profile_enabled {
            profile.report(sync_after_forward);
        }
        if layer_profile_enabled {
            architectures::deepseek2::report_layer_profile();
            architectures::deepseek4::report_layer_profile();
            architectures::gemma4::report_layer_profile();
            architectures::llama::report_layer_profile();
            architectures::minimax_m2::report_layer_profile();
            architectures::nemotron_h::report_layer_profile();
            architectures::phi3::report_layer_profile();
            architectures::qwen35::report_layer_profile();
        }

        self.record_cache_identity(&tokens, &generated);
        Ok(generated)
    }

    fn prefill_text_prompt(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        mrope_decode_delta: Option<i64>,
    ) -> Result<StageOutput> {
        let chunk_tokens = self
            .options
            .prefill_chunk_tokens
            .unwrap_or(DEFAULT_PREFILL_CHUNK_TOKENS);
        let context = TextRequestContext::new(
            start_pos
                .checked_add(tokens.len())
                .context("text prompt end overflow")?,
        )?;
        let result = (|| {
            let mut position = start_pos;
            for chunk in tokens.chunks(chunk_tokens) {
                let end = position
                    .checked_add(chunk.len())
                    .context("text prefill position overflow")?;
                let input = match mrope_decode_delta {
                    Some(delta) => StageInput::positioned_activations(
                        self.embed_tokens(chunk)?,
                        MropePositionIds::text_with_delta(position, delta, chunk.len())?,
                    ),
                    None => StageInput::Tokens(chunk.to_vec()),
                };
                let complete = end - start_pos == tokens.len();
                let pass = ForwardPass::Text {
                    context,
                    prefill: true,
                    cache_only: !complete,
                    tail_fence: false,
                };
                let (output, _) = self.forward_pass(input, chunk.len(), position, pass)?;
                if complete {
                    return Ok(output);
                }
                drop(output);
                // Metal retains dropped buffers until a device wait. Release
                // each chunk's temporary tensors before queuing the next.
                self.device
                    .synchronize()
                    .context("completing text prefill chunk")?;
                position = end;
            }
            anyhow::bail!("text prefill prompt must contain a token")
        })();
        if result.is_err() {
            // Earlier chunks may already have advanced the cache. Never
            // expose a partial prompt as reusable state after an error.
            self.clear_cache();
        }
        result
    }
}

#[cfg(test)]
mod session_state_tests {
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use anyhow::Result;
    use candle_core::{DType, Device, Tensor};

    use super::architectures::{
        KvCache, LayerCache, ModelArchitecture, MropePositionIds, StageInput, StageOutput,
    };
    use super::sampling::SamplingParams;
    use super::{KvTokenBudget, PipelineStage, StagePosition};
    use crate::continuous::GenerationSession;

    struct CacheCountingArchitecture;

    #[test]
    fn draft_token_cap_excludes_anchor_and_bounds_preferred_rows() -> Result<()> {
        for (preferred, cap, expected) in [
            (8, None, 8),
            (8, Some(7), 8),
            (8, Some(3), 4),
            (8, Some(1), 2),
            (7, Some(7), 7),
            (7, None, 7),
            (usize::MAX, Some(100), 8),
            (0, Some(7), 2),
        ] {
            assert_eq!(
                super::resolve_speculative_block_size(8, preferred, cap)?,
                expected
            );
        }
        assert!(super::resolve_speculative_block_size(1, 8, Some(7)).is_err());
        assert!(super::resolve_speculative_block_size(8, 8, Some(0)).is_err());
        assert!(super::resolve_speculative_block_size(8, 8, Some(usize::MAX)).is_err());
        Ok(())
    }

    #[test]
    fn request_context_limits_reject_before_cache_mutation() -> Result<()> {
        let mut stage = test_stage();
        stage.context_length = 16;
        assert!(stage.validate_context_budget(8, 8).is_ok());
        assert!(stage.validate_context_budget(9, 8).is_err());
        assert!(stage.validate_context_budget(usize::MAX, 1).is_err());
        let params = SamplingParams {
            max_tokens: 8,
            ..Default::default()
        };
        let mut session =
            crate::continuous::GenerationSession::new(&stage, vec![0; 9], params.clone(), vec![]);
        assert!(session.step(&mut stage).is_err());
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        assert_eq!(stage.next_position(), 0);
        assert!(
            stage
                .generate(vec![0; 9], &params, &[], 0, |_| false)
                .is_err()
        );
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn caller_prefix_requires_exact_non_evicted_token_identity() -> Result<()> {
        let mut stage = test_stage();
        stage.prefill(StageInput::Tokens(vec![1, 2]), 2, 0)?;
        stage.record_cache_identity(&[1, 2], &[0]);

        assert_eq!(stage.validate_cached_prefix(&[1, 2, 3], 2), 2);
        assert_eq!(stage.next_position(), 2);
        assert_eq!(stage.kv_cache_tokens(), 2);

        assert_eq!(stage.validate_cached_prefix(&[1, 9, 3], 2), 0);
        assert_eq!(stage.next_position(), 0);
        assert_eq!(stage.kv_cache_tokens(), 0);
        assert!(!stage.has_cache_identity());
        Ok(())
    }

    #[test]
    fn caller_prefix_rejects_evicted_cache_even_when_identity_matches() -> Result<()> {
        let mut stage = test_stage();
        stage.set_max_kv_tokens(1);
        stage.prefill(StageInput::Tokens(vec![1]), 1, 0)?;
        stage.forward(StageInput::Tokens(vec![2]), 1, 1)?;
        stage.record_cache_identity(&[1, 2], &[]);

        assert_eq!(stage.validate_cached_prefix(&[1, 2, 3], 2), 0);
        assert_eq!(stage.next_position(), 0);
        assert_eq!(stage.kv_cache_tokens(), 0);
        Ok(())
    }

    #[test]
    fn completion_identity_survives_rolling_eviction_without_admitting_reuse() -> Result<()> {
        let mut stage = test_stage();
        stage.set_max_kv_tokens(1);
        stage.prefill(StageInput::Tokens(vec![1]), 1, 0)?;
        stage.forward(StageInput::Tokens(vec![2]), 1, 1)?;
        stage.record_cache_identity(&[1, 2, 3], &[]);

        assert!(stage.cache_fingerprint().is_some());
        assert!(!stage.has_cache_identity());
        let fingerprint = stage.cache_fingerprint().unwrap();
        assert!(!stage.cache_matches_fingerprint(2, &fingerprint));
        Ok(())
    }

    #[test]
    fn embedding_prompt_can_invalidate_a_text_shaped_cache_identity() -> Result<()> {
        let mut stage = test_stage();
        stage.prefill(StageInput::Tokens(vec![1, 2]), 2, 0)?;
        stage.record_cache_identity(&[1, 2], &[]);
        assert!(stage.cache_fingerprint().is_some());

        // The embedding tensor may have exactly two rows, but those rows are
        // not represented by the text IDs and must not be published as a
        // reusable prefix.
        stage.invalidate_cache_identity();
        assert!(stage.cache_fingerprint().is_none());
        assert!(!stage.has_cache_identity());
        Ok(())
    }

    #[test]
    fn draft_token_cap_rejects_invalid_values_without_changing_state() {
        let mut stage = test_stage();
        let bytes = stage.fixed_cache_budget_bytes();
        for cap in [0, usize::MAX, 7] {
            assert!(stage.set_speculative_draft_max_tokens(cap).is_err());
            assert!(stage.speculative_draft_max_tokens.is_none());
            assert_eq!(stage.speculative_draft_max_tokens(), None);
            assert_eq!(stage.fixed_cache_budget_bytes(), bytes);
        }
    }
    struct SplitCountingArchitecture {
        broken_output: bool,
    }

    impl ModelArchitecture for SplitCountingArchitecture {
        fn forward(
            &self,
            input: StageInput,
            cache: &mut [Option<LayerCache>],
            len: usize,
            pos: usize,
        ) -> Result<StageOutput> {
            CacheCountingArchitecture.forward(input, cache, len, pos)
        }
        fn supports_pipeline_decode_batch(&self) -> bool {
            true
        }
        fn decode_batch_uses_packed_cache(&self) -> bool {
            false
        }
        fn forward_pipeline_decode_batch(
            &self,
            inputs: &[StageInput],
            caches: &mut [&mut [Option<LayerCache>]],
            _: &mut [Option<LayerCache>],
            positions: &[usize],
        ) -> Result<Vec<StageOutput>> {
            inputs
                .iter()
                .zip(caches)
                .zip(positions)
                .map(|((input, cache), &pos)| {
                    let output = self.forward(input.clone(), cache, 1, pos)?;
                    if self.broken_output {
                        Ok(StageOutput::Activations(Tensor::zeros(
                            (1, 1),
                            DType::F32,
                            &Device::Cpu,
                        )?))
                    } else {
                        Ok(output)
                    }
                })
                .collect()
        }
        fn family(&self) -> &str {
            "split-counting-test"
        }
        fn hidden_dim(&self) -> usize {
            1
        }
        fn layer_count(&self) -> usize {
            1
        }
        fn vocab_size(&self) -> usize {
            2
        }
    }

    #[test]
    fn split_batch_checks_budget_before_writes_and_rejects_wrong_output_role() -> Result<()> {
        let mut stage = test_stage();
        stage.vocab_size = 2; // SplitCountingArchitecture returns two logits.
        stage.position = StagePosition::Tail;
        stage.arch = Box::new(SplitCountingArchitecture {
            broken_output: false,
        });
        stage.set_max_total_kv_tokens(8)?; // Four resident slots plus copy reserve.
        let input =
            || StageInput::activations(Tensor::zeros((1, 1), DType::F32, &Device::Cpu).unwrap());
        let mut first = stage.new_session_state();
        let mut second = stage.new_session_state();
        let mut other = stage.new_session_state();
        stage.forward_with_session(&mut other, input(), 1, 0)?;
        stage.forward_pipeline_decode_batch_with_sessions(
            &mut [&mut first, &mut second],
            &[input(), input()],
        )?;
        assert_eq!(stage.total_kv_cache_tokens(), 3);
        assert!(
            stage
                .forward_pipeline_decode_batch_with_sessions(
                    &mut [&mut first, &mut second],
                    &[input(), input()]
                )
                .is_err()
        );
        assert_eq!(first.next_position, 1);
        assert_eq!(second.next_position, 1);
        assert_eq!(stage.total_kv_cache_tokens(), 3);
        // Give execution room, then fail AFTER both lane caches advance.
        stage.set_max_total_kv_tokens(16)?;
        stage.arch = Box::new(SplitCountingArchitecture {
            broken_output: true,
        });
        let error = stage
            .forward_pipeline_decode_batch_with_sessions(
                &mut [&mut first, &mut second],
                &[input(), input()],
            )
            .err()
            .unwrap();
        assert!(error.to_string().contains("invalid output"));
        assert_eq!(first.next_position, 0);
        assert_eq!(second.next_position, 0);
        assert_eq!(stage.total_kv_cache_tokens(), 1);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 1);
        assert_eq!(other.next_position, 1);
        assert!(first.cache.iter().all(Option::is_none));
        assert!(second.cache.iter().all(Option::is_none));
        stage.clear_session_state(&mut other);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    struct PassRecordingArchitecture {
        seen: Arc<Mutex<Vec<(bool, usize, usize)>>>,
    }

    impl ModelArchitecture for PassRecordingArchitecture {
        fn forward(
            &self,
            input: StageInput,
            cache: &mut [Option<LayerCache>],
            seq_len: usize,
            start_pos: usize,
        ) -> Result<StageOutput> {
            self.seen.lock().unwrap().push((false, seq_len, start_pos));
            CacheCountingArchitecture.forward(input, cache, seq_len, start_pos)
        }

        fn forward_prefill(
            &self,
            input: StageInput,
            cache: &mut [Option<LayerCache>],
            seq_len: usize,
            start_pos: usize,
        ) -> Result<StageOutput> {
            self.seen.lock().unwrap().push((true, seq_len, start_pos));
            CacheCountingArchitecture.forward(input, cache, seq_len, start_pos)
        }

        fn family(&self) -> &str {
            "pass-recording-test"
        }
        fn hidden_dim(&self) -> usize {
            1
        }
        fn layer_count(&self) -> usize {
            1
        }
        fn vocab_size(&self) -> usize {
            2
        }
    }
    struct CacheOnlyRecordingArchitecture {
        // Whether a chunk skipped its head, its length, and its position.
        seen: Arc<Mutex<Vec<(bool, usize, usize)>>>,
    }

    impl ModelArchitecture for CacheOnlyRecordingArchitecture {
        fn forward(
            &self,
            input: StageInput,
            cache: &mut [Option<LayerCache>],
            seq_len: usize,
            start_pos: usize,
        ) -> Result<StageOutput> {
            self.seen.lock().unwrap().push((false, seq_len, start_pos));
            CacheCountingArchitecture.forward(input, cache, seq_len, start_pos)
        }

        fn forward_prefill_cache_only(
            &self,
            input: StageInput,
            cache: &mut [Option<LayerCache>],
            seq_len: usize,
            start_pos: usize,
        ) -> Result<StageOutput> {
            self.seen.lock().unwrap().push((true, seq_len, start_pos));
            CacheCountingArchitecture.forward(input, cache, seq_len, start_pos)?;
            // No logits exist on this path. The session must not sample them.
            Ok(StageOutput::Activations(Tensor::zeros(
                (seq_len, 1),
                candle_core::DType::F32,
                &Device::Cpu,
            )?))
        }

        fn family(&self) -> &str {
            "cache-only-test"
        }
        fn hidden_dim(&self) -> usize {
            1
        }
        fn layer_count(&self) -> usize {
            1
        }
        fn vocab_size(&self) -> usize {
            2
        }
    }

    struct OversizedBatchCacheArchitecture;
    struct RecurrentCacheArchitecture {
        mamba: bool,
    }
    struct WrongBatchOutputArchitecture {
        extra: bool,
    }
    struct PreallocatedCacheArchitecture;
    struct QuantizedCacheArchitecture;

    struct PositionRecordingArchitecture {
        seen: Arc<Mutex<Vec<Option<MropePositionIds>>>>,
    }

    impl ModelArchitecture for PositionRecordingArchitecture {
        fn forward(
            &self,
            input: StageInput,
            _cache: &mut [Option<LayerCache>],
            seq_len: usize,
            _start_pos: usize,
        ) -> Result<StageOutput> {
            let positions = match input {
                StageInput::Tokens(tokens) => {
                    anyhow::ensure!(tokens.len() == seq_len, "test token length mismatch");
                    None
                }
                StageInput::Activations {
                    tensor, positions, ..
                } => {
                    anyhow::ensure!(tensor.dim(0)? == seq_len, "test embedding length mismatch");
                    positions
                }
            };
            self.seen.lock().unwrap().push(positions);
            Ok(StageOutput::Logits(Tensor::new(
                &[0.0f32, 1.0],
                &Device::Cpu,
            )?))
        }

        fn embed_tokens(&self, tokens: &[u32]) -> Result<Tensor> {
            Tensor::from_vec(
                tokens.iter().map(|token| *token as f32).collect::<Vec<_>>(),
                (tokens.len(), 1),
                &Device::Cpu,
            )
            .map_err(Into::into)
        }

        fn family(&self) -> &str {
            "position-recording-test"
        }

        fn hidden_dim(&self) -> usize {
            1
        }

        fn layer_count(&self) -> usize {
            0
        }

        fn vocab_size(&self) -> usize {
            2
        }
    }

    impl ModelArchitecture for CacheCountingArchitecture {
        fn forward(
            &self,
            input: StageInput,
            cache: &mut [Option<LayerCache>],
            seq_len: usize,
            start_pos: usize,
        ) -> Result<StageOutput> {
            let fail_after_cache_update = matches!(
                &input,
                StageInput::Tokens(tokens) if tokens.first() == Some(&255)
            );
            let mut values = match &cache[0] {
                Some(LayerCache::Attention(KvCache::F16 { k, .. })) => {
                    k.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?
                }
                None => Vec::new(),
                Some(_) => anyhow::bail!("unexpected cache type"),
            };
            let incoming = match input {
                StageInput::Tokens(tokens) => {
                    anyhow::ensure!(tokens.len() == seq_len, "test token length mismatch");
                    tokens.into_iter().map(|token| token as f32).collect()
                }
                StageInput::Activations { .. } => vec![0.0; seq_len],
            };
            values.extend(incoming);
            let total = values.len();
            let k = Tensor::from_vec(values.clone(), (1, 1, total, 1), &Device::Cpu)?;
            let v = Tensor::from_vec(values, (1, 1, total, 1), &Device::Cpu)?;
            cache[0] = Some(LayerCache::Attention(KvCache::F16 { k, v }));
            anyhow::ensure!(!fail_after_cache_update, "injected forward failure");
            Ok(StageOutput::Logits(Tensor::new(
                &[start_pos as f32, (start_pos + seq_len) as f32],
                &Device::Cpu,
            )?))
        }

        fn family(&self) -> &str {
            "test"
        }

        fn supports_decode_batch(&self) -> bool {
            true
        }

        fn forward_decode_batch(
            &self,
            tokens: &[u32],
            caches: &mut [&mut [Option<LayerCache>]],
            _batch_cache: &mut [Option<LayerCache>],
            positions: &[usize],
        ) -> Result<Vec<Tensor>> {
            tokens
                .iter()
                .zip(caches.iter_mut())
                .zip(positions)
                .map(|((token, cache), position)| {
                    let StageOutput::Logits(logits) =
                        self.forward(StageInput::Tokens(vec![*token]), cache, 1, *position)?
                    else {
                        anyhow::bail!("expected logits")
                    };
                    Ok(logits)
                })
                .collect()
        }

        fn hidden_dim(&self) -> usize {
            1
        }

        fn layer_count(&self) -> usize {
            1
        }

        fn vocab_size(&self) -> usize {
            2
        }
    }

    struct IndependentBatchCacheArchitecture {
        preallocated: bool,
    }

    impl ModelArchitecture for IndependentBatchCacheArchitecture {
        fn forward(
            &self,
            input: StageInput,
            cache: &mut [Option<LayerCache>],
            seq_len: usize,
            start_pos: usize,
        ) -> Result<StageOutput> {
            if self.preallocated {
                PreallocatedCacheArchitecture.forward(input, cache, seq_len, start_pos)
            } else {
                CacheCountingArchitecture.forward(input, cache, seq_len, start_pos)
            }
        }
        fn forward_decode_batch(
            &self,
            tokens: &[u32],
            caches: &mut [&mut [Option<LayerCache>]],
            _batch_cache: &mut [Option<LayerCache>],
            positions: &[usize],
        ) -> Result<Vec<Tensor>> {
            tokens
                .iter()
                .zip(caches.iter_mut())
                .zip(positions)
                .map(|((&token, cache), &position)| {
                    match self.forward(StageInput::Tokens(vec![token]), cache, 1, position)? {
                        StageOutput::Logits(logits) => Ok(logits),
                        _ => anyhow::bail!("expected logits"),
                    }
                })
                .collect()
        }
        fn supports_decode_batch(&self) -> bool {
            true
        }
        fn decode_batch_uses_packed_cache(&self) -> bool {
            false
        }
        fn family(&self) -> &str {
            "independent-batch-test"
        }
        fn hidden_dim(&self) -> usize {
            1
        }
        fn layer_count(&self) -> usize {
            1
        }
        fn vocab_size(&self) -> usize {
            2
        }
    }

    impl ModelArchitecture for WrongBatchOutputArchitecture {
        fn forward(
            &self,
            input: StageInput,
            cache: &mut [Option<LayerCache>],
            seq_len: usize,
            start_pos: usize,
        ) -> Result<StageOutput> {
            CacheCountingArchitecture.forward(input, cache, seq_len, start_pos)
        }

        fn forward_decode_batch(
            &self,
            tokens: &[u32],
            caches: &mut [&mut [Option<LayerCache>]],
            batch_cache: &mut [Option<LayerCache>],
            positions: &[usize],
        ) -> Result<Vec<Tensor>> {
            let mut outputs = CacheCountingArchitecture.forward_decode_batch(
                tokens,
                caches,
                batch_cache,
                positions,
            )?;
            if self.extra {
                outputs.push(outputs[0].clone());
            } else {
                outputs.pop();
            }
            Ok(outputs)
        }

        fn family(&self) -> &str {
            "wrong-batch-output-test"
        }
        fn supports_decode_batch(&self) -> bool {
            true
        }
        fn hidden_dim(&self) -> usize {
            1
        }
        fn layer_count(&self) -> usize {
            1
        }
        fn vocab_size(&self) -> usize {
            2
        }
    }

    impl ModelArchitecture for OversizedBatchCacheArchitecture {
        fn forward(
            &self,
            input: StageInput,
            cache: &mut [Option<LayerCache>],
            seq_len: usize,
            start_pos: usize,
        ) -> Result<StageOutput> {
            CacheCountingArchitecture.forward(input, cache, seq_len, start_pos)
        }

        fn family(&self) -> &str {
            "oversized-batch-test"
        }

        fn supports_decode_batch(&self) -> bool {
            true
        }

        fn forward_decode_batch(
            &self,
            tokens: &[u32],
            caches: &mut [&mut [Option<LayerCache>]],
            _batch_cache: &mut [Option<LayerCache>],
            _positions: &[usize],
        ) -> Result<Vec<Tensor>> {
            for cache in caches {
                let k = Tensor::zeros((1, 1, 6, 1), DType::F16, &Device::Cpu)?;
                let v = Tensor::zeros((1, 1, 6, 1), DType::F16, &Device::Cpu)?;
                cache[0] = Some(LayerCache::Attention(KvCache::F16 { k, v }));
            }
            tokens
                .iter()
                .map(|token| Tensor::new(&[*token as f32], &Device::Cpu))
                .collect::<candle_core::Result<Vec<_>>>()
                .map_err(Into::into)
        }

        fn hidden_dim(&self) -> usize {
            1
        }

        fn layer_count(&self) -> usize {
            1
        }

        fn vocab_size(&self) -> usize {
            2
        }
    }

    impl ModelArchitecture for PreallocatedCacheArchitecture {
        fn supports_decode_batch(&self) -> bool {
            true
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
            let mut packed = super::architectures::pack_decode_layer_cache(caches, 0)?;
            let incoming = Tensor::from_vec(
                tokens.iter().map(|token| *token as f32).collect::<Vec<_>>(),
                (tokens.len(), 1, 1, 1),
                &Device::Cpu,
            )?;
            super::architectures::kv_cache_update_native(
                &mut packed,
                incoming.clone(),
                incoming,
                None,
            )?;
            super::architectures::publish_decode_layer_cache_views(caches, 0, &packed)?;
            batch_cache[0] = packed.map(LayerCache::Attention);
            positions
                .iter()
                .map(|position| {
                    Tensor::new(&[*position as f32, (*position + 1) as f32], &Device::Cpu)
                        .map_err(Into::into)
                })
                .collect()
        }

        fn forward(
            &self,
            input: StageInput,
            cache: &mut [Option<LayerCache>],
            seq_len: usize,
            start_pos: usize,
        ) -> Result<StageOutput> {
            let values = match input {
                StageInput::Tokens(tokens) => {
                    anyhow::ensure!(tokens.len() == seq_len, "test token length mismatch");
                    tokens.into_iter().map(|token| token as f32).collect()
                }
                StageInput::Activations { .. } => vec![0.0; seq_len],
            };
            let incoming = Tensor::from_vec(values, (1, 1, seq_len, 1), &Device::Cpu)?;
            let mut kv = match cache[0].take() {
                Some(LayerCache::Attention(kv)) => Some(kv),
                None => None,
                Some(_) => anyhow::bail!("unexpected cache type"),
            };
            super::architectures::kv_cache_update_native(
                &mut kv,
                incoming.clone(),
                incoming,
                None,
            )?;
            cache[0] = kv.map(LayerCache::Attention);
            Ok(StageOutput::Logits(Tensor::new(
                &[start_pos as f32, (start_pos + seq_len) as f32],
                &Device::Cpu,
            )?))
        }

        fn family(&self) -> &str {
            "preallocated-test"
        }

        fn hidden_dim(&self) -> usize {
            1
        }

        fn layer_count(&self) -> usize {
            1
        }

        fn vocab_size(&self) -> usize {
            2
        }
    }

    impl ModelArchitecture for QuantizedCacheArchitecture {
        fn forward(
            &self,
            input: StageInput,
            cache: &mut [Option<LayerCache>],
            seq_len: usize,
            _start_pos: usize,
        ) -> Result<StageOutput> {
            match input {
                StageInput::Tokens(tokens) => {
                    anyhow::ensure!(tokens.len() == seq_len, "test token length mismatch");
                }
                StageInput::Activations { tensor, .. } => {
                    anyhow::ensure!(tensor.dim(0)? == seq_len, "test activation length mismatch");
                }
            }
            let config = super::turboquant::TurboQuantConfig::new(8, 64, 42);
            let incoming = Tensor::zeros((1, 2, seq_len, 64), DType::F32, &Device::Cpu)?;
            let mut kv = match cache[0].take() {
                Some(LayerCache::Attention(kv)) => Some(kv),
                None => None,
                Some(_) => anyhow::bail!("unexpected cache type"),
            };
            super::architectures::kv_cache_update(
                &mut kv,
                incoming.clone(),
                incoming,
                Some(&config),
            )?;
            cache[0] = kv.map(LayerCache::Attention);
            Ok(StageOutput::Logits(Tensor::zeros(
                2,
                DType::F32,
                &Device::Cpu,
            )?))
        }

        fn family(&self) -> &str {
            "quantized-test"
        }

        fn hidden_dim(&self) -> usize {
            64
        }

        fn layer_count(&self) -> usize {
            1
        }

        fn vocab_size(&self) -> usize {
            2
        }
    }

    pub(super) fn test_stage() -> PipelineStage {
        // These cache tests store token IDs as visible cache values. The
        // fixture admits 256 IDs and returns two diagnostic scores. ID 255
        // triggers a failure after a write, so request checks must admit it.
        PipelineStage {
            arch: Box::new(CacheCountingArchitecture),
            speculative: None,
            speculative_cache: None,
            speculative_draft_max_tokens: None,
            position: StagePosition::Solo,
            cache: vec![None],
            hidden_dim: 1,
            vocab_size: 256,
            context_length: usize::MAX,
            device: Device::Cpu,
            kv_bytes_per_token: 4,
            kv_budget_bytes_per_token: 4,
            fixed_cache_bytes: 0,
            next_position: 0,
            mrope_decode_delta: None,
            text_arithmetic: None,
            kv_cache_tokens: 0,
            cache_token_ids: Vec::new(),
            cache_fingerprint: None,
            kv_cache_lane_capacity_tokens: 0,
            kv_cache_allocation_tokens: 0,
            fixed_cache_allocation_tokens: 0,
            quant_config: None,
            max_kv_tokens: None,
            kv_token_budget: Arc::new(KvTokenBudget::default()),
            max_total_kv_tokens: None,
            max_total_kv_headroom_tokens: None,
            preallocated_kv_cache: false,
            decode_batch_caches: std::collections::HashMap::new(),
            prompt_cache: crate::prompt_cache::PromptCache::default(),
            options: crate::EngineOptions::default(),
            #[cfg(feature = "cuda")]
            cuda_decode_graph: crate::cuda_graph::DecodeGraph::default(),
            device_stages: crate::device_stages::StageRegistration::uncounted(),
        }
    }

    struct RequestArithmeticArchitecture;

    impl ModelArchitecture for RequestArithmeticArchitecture {
        fn forward(
            &self,
            input: StageInput,
            cache: &mut [Option<LayerCache>],
            len: usize,
            start: usize,
        ) -> Result<StageOutput> {
            CacheCountingArchitecture.forward(input, cache, len, start)
        }
        fn text_arithmetic_policy(
            &self,
            context: super::TextRequestContext,
        ) -> Option<super::TextArithmeticPolicy> {
            Some(super::TextArithmeticPolicy::phi_k(context))
        }
        fn forward_text(
            &self,
            input: StageInput,
            cache: &mut [Option<LayerCache>],
            len: usize,
            start: usize,
            context: super::TextRequestContext,
            prefill: bool,
            _cache_only: bool,
        ) -> Result<StageOutput> {
            context.validate(start, len, prefill)?;
            let StageInput::Tokens(tokens) = input else {
                anyhow::bail!("test text tokens required")
            };
            let cutoff = super::TextArithmeticPolicy::phi_k(context).m4_prefix_end();
            let values = tokens
                .into_iter()
                .enumerate()
                .map(|(i, t)| {
                    if t == 255 {
                        t
                    } else {
                        t + u32::from(start + i >= cutoff)
                    }
                })
                .collect();
            CacheCountingArchitecture.forward(StageInput::Tokens(values), cache, len, start)
        }
        fn family(&self) -> &str {
            "request-arithmetic-test"
        }
        fn hidden_dim(&self) -> usize {
            1
        }
        fn layer_count(&self) -> usize {
            1
        }
        fn vocab_size(&self) -> usize {
            2
        }
    }

    struct DeepRequestArithmeticArchitecture;

    impl ModelArchitecture for DeepRequestArithmeticArchitecture {
        fn forward(
            &self,
            input: StageInput,
            cache: &mut [Option<LayerCache>],
            len: usize,
            start: usize,
        ) -> Result<StageOutput> {
            CacheCountingArchitecture.forward(input, cache, len, start)
        }
        fn text_arithmetic_policy(
            &self,
            context: super::TextRequestContext,
        ) -> Option<super::TextArithmeticPolicy> {
            Some(super::TextArithmeticPolicy::deep_cpu_q8(context))
        }
        fn forward_text(
            &self,
            input: StageInput,
            cache: &mut [Option<LayerCache>],
            len: usize,
            start: usize,
            context: super::TextRequestContext,
            prefill: bool,
            _cache_only: bool,
        ) -> Result<StageOutput> {
            context.validate(start, len, prefill)?;
            let StageInput::Tokens(tokens) = input else {
                anyhow::bail!("test text tokens required")
            };
            let policy = super::TextArithmeticPolicy::deep_cpu_q8(context);
            let values = tokens
                .into_iter()
                .enumerate()
                .map(|(i, t)| t + u32::from(policy.native_q1_split_at(start + i) == Some(true)))
                .collect();
            CacheCountingArchitecture.forward(StageInput::Tokens(values), cache, len, start)
        }
        fn family(&self) -> &str {
            "deep-request-arithmetic-test"
        }
        fn hidden_dim(&self) -> usize {
            1
        }
        fn layer_count(&self) -> usize {
            1
        }
        fn vocab_size(&self) -> usize {
            2
        }
    }

    #[test]
    fn deep_request_context_chunks_interleave_and_clear_keep_policy() -> Result<()> {
        for end in [257, 258, 265, 290] {
            let context = super::TextRequestContext::new(end)?;
            for chunk in [1, 5, 7, 32, 128, usize::MAX] {
                let mut stage = test_stage();
                stage.arch = Box::new(DeepRequestArithmeticArchitecture);
                let mut state = stage.new_session_state();
                let mut other = stage.new_session_state();
                stage.prepare_text_request_with_session(&mut state, context)?;
                for start in (0..end).step_by(chunk.min(end)) {
                    let rows = chunk.min(end - start);
                    stage.prefill_text_with_session(
                        &mut state,
                        StageInput::Tokens(vec![0; rows]),
                        rows,
                        start,
                        context,
                    )?;
                }
                let checkpoint = stage.checkpoint_session_state(&state)?;
                let other_context = super::TextRequestContext::new(3)?;
                stage.prepare_text_request_with_session(&mut other, other_context)?;
                stage.prefill_text_with_session(
                    &mut other,
                    StageInput::Tokens(vec![0; 3]),
                    3,
                    0,
                    other_context,
                )?;
                stage.forward_text_with_session(
                    &mut state,
                    StageInput::Tokens(vec![0]),
                    1,
                    end,
                    context,
                )?;
                let expected: Vec<_> = (0..=end)
                    .map(|p| {
                        u8::from(p >= 256 && (p >= end || end % 8 == 1 && p == end - 1)) as f32
                    })
                    .collect();
                assert_eq!(arithmetic_keys(&state)?, expected);
                assert_eq!(arithmetic_keys(&other)?, vec![0.; 3]);
                stage.restore_session_checkpoint(&mut state, checkpoint);
                assert_eq!(state.next_position, end);
                assert_eq!(
                    state.text_arithmetic,
                    Some(super::TextArithmeticPolicy::deep_cpu_q8(context))
                );
                assert!(
                    stage
                        .forward_with_session(&mut state, StageInput::Tokens(vec![0]), 1, end)
                        .is_err()
                );
                stage.clear_session_state(&mut state);
                stage.clear_session_state(&mut other);
                assert_eq!(stage.total_kv_cache_tokens(), 0);
                assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
            }
        }
        Ok(())
    }

    #[test]
    fn deep_request_context_reuse_stops_before_changed_attention_row() -> Result<()> {
        let old = super::TextRequestContext::new(257)?;
        let next = super::TextRequestContext::new(258)?;
        let mut stage = test_stage();
        stage.arch = Box::new(DeepRequestArithmeticArchitecture);
        let mut state = stage.new_session_state();
        stage.prepare_text_request_with_session(&mut state, old)?;
        stage.prefill_text_with_session(
            &mut state,
            StageInput::Tokens(vec![0; 256]),
            256,
            0,
            old,
        )?;
        let checkpoint = stage.checkpoint_session_state(&state)?;
        let charge = stage.total_kv_cache_allocation_tokens();
        assert!(stage.prepare_text_request_with_session(&mut state, next)?);
        assert_eq!(state.next_position, 256);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), charge);
        stage.restore_session_checkpoint(&mut state, checkpoint);
        stage.prefill_text_with_session(&mut state, StageInput::Tokens(vec![0]), 1, 256, old)?;
        assert_eq!(arithmetic_keys(&state)?[256], 1.);
        assert!(!stage.prepare_text_request_with_session(&mut state, next)?);
        assert_eq!(state.next_position, 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        stage.prefill_text_with_session(
            &mut state,
            StageInput::Tokens(vec![0; 258]),
            258,
            0,
            next,
        )?;
        assert_eq!(arithmetic_keys(&state)?, vec![0.; 258]);
        stage.clear_session_state(&mut state);
        Ok(())
    }

    pub(super) fn arithmetic_test_stage() -> PipelineStage {
        let mut stage = test_stage();
        stage.arch = Box::new(RequestArithmeticArchitecture);
        stage
    }

    fn arithmetic_keys(state: &super::PipelineStageSessionState) -> Result<Vec<f32>> {
        match &state.cache[0] {
            Some(LayerCache::Attention(KvCache::F16 { k, .. })) => {
                Ok(k.flatten_all()?.to_vec1::<f32>()?)
            }
            None => Ok(Vec::new()),
            _ => anyhow::bail!("test F32 key storage required"),
        }
    }

    #[test]
    fn request_context_cache_rows_are_identical_at_arbitrary_chunks() -> Result<()> {
        let context = super::TextRequestContext::new(137)?;
        let mut baseline = None;
        for size in [1, 5, 7, 32, 128, usize::MAX] {
            let mut stage = arithmetic_test_stage();
            let mut state = stage.new_session_state();
            assert!(stage.prepare_text_request_with_session(&mut state, context)?);
            let mut start = 0;
            while start < 137 {
                let rows = size.min(137 - start);
                stage.prefill_text_with_session(
                    &mut state,
                    StageInput::Tokens(vec![0; rows]),
                    rows,
                    start,
                    context,
                )?;
                start += rows;
            }
            for position in 137..140 {
                stage.forward_text_with_session(
                    &mut state,
                    StageInput::Tokens(vec![0]),
                    1,
                    position,
                    context,
                )?;
            }
            let values = arithmetic_keys(&state)?;
            assert_eq!(&values[..136], vec![0.0; 136]);
            assert_eq!(&values[136..], vec![1.0; 4]);
            if let Some(expected) = &baseline {
                assert_eq!(expected, &values)
            } else {
                baseline = Some(values)
            };
            let before = state.next_position;
            assert!(
                stage
                    .forward_with_session(&mut state, StageInput::Tokens(vec![0]), 1, 140)
                    .is_err()
            );
            assert_eq!(state.next_position, before);
            stage.clear_session_state(&mut state);
            assert_eq!(stage.total_kv_cache_tokens(), 0);
            assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
            assert_eq!(state.text_arithmetic, None);
        }
        Ok(())
    }

    #[test]
    fn request_context_partial_decode_reuse_and_checkpoint_accounting() -> Result<()> {
        let mut stage = arithmetic_test_stage();
        let mut state = stage.new_session_state();
        let old = super::TextRequestContext::new(137)?;
        let next = super::TextRequestContext::new(140)?;
        stage.prepare_text_request_with_session(&mut state, old)?;
        stage.prefill_text_with_session(
            &mut state,
            StageInput::Tokens(vec![0; 113]),
            113,
            0,
            old,
        )?;
        let checkpoint = stage.checkpoint_session_state(&state)?;
        let charge = stage.total_kv_cache_allocation_tokens();
        // Both policies used M4 before the stored unaligned position113.
        assert!(stage.prepare_text_request_with_session(&mut state, next)?);
        assert_eq!(state.next_position, 113);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), charge);
        stage.prefill_text_with_session(
            &mut state,
            StageInput::Tokens(vec![0; 27]),
            27,
            113,
            next,
        )?;
        stage.restore_session_checkpoint(&mut state, checkpoint);
        assert_eq!(state.next_position, 113);
        assert_eq!(
            state.text_arithmetic,
            Some(super::TextArithmeticPolicy::phi_k(old))
        );
        stage.prefill_text_with_session(
            &mut state,
            StageInput::Tokens(vec![0; 24]),
            24,
            113,
            old,
        )?;
        stage.forward_text_with_session(&mut state, StageInput::Tokens(vec![0]), 1, 137, old)?;
        // The old tail and decoded row are M1; the next prompt needs M4 there.
        assert!(!stage.prepare_text_request_with_session(&mut state, next)?);
        assert_eq!(state.next_position, 0);
        assert!(state.cache.iter().all(Option::is_none));
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        stage.prefill_text_with_session(
            &mut state,
            StageInput::Tokens(vec![0; 140]),
            140,
            0,
            next,
        )?;
        assert_eq!(arithmetic_keys(&state)?, vec![0.0; 140]);
        assert!(
            stage
                .forward_text_with_session(&mut state, StageInput::Tokens(vec![255]), 1, 140, next)
                .is_err()
        );
        assert_eq!(state.next_position, 0);
        assert_eq!(state.text_arithmetic, None);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    impl ModelArchitecture for RecurrentCacheArchitecture {
        fn forward(
            &self,
            input: StageInput,
            cache: &mut [Option<LayerCache>],
            seq_len: usize,
            _start_pos: usize,
        ) -> Result<StageOutput> {
            let StageInput::Tokens(tokens) = input else {
                anyhow::bail!("recurrent test requires tokens");
            };
            anyhow::ensure!(tokens.len() == seq_len, "test token length mismatch");
            let previous = recurrent_value(&cache[0])?;
            let failed = tokens.contains(&255);
            let total = previous
                + tokens
                    .iter()
                    .map(|token| if *token == 255 { 1.0 } else { *token as f32 })
                    .sum::<f32>();
            let state = Tensor::from_vec(vec![total], (1, 1, 1), &Device::Cpu)?;
            let conv = Tensor::from_vec(vec![total], (1, 1), &Device::Cpu)?;
            cache[0] = Some(if self.mamba {
                LayerCache::Mamba2(crate::architectures::Mamba2Cache {
                    conv_state: conv,
                    ssm_state: state,
                })
            } else {
                LayerCache::DeltaNet(crate::architectures::DeltaNetCache {
                    conv_state: conv,
                    recurrent_state: state,
                    speculative_state: None,
                })
            });
            let mut values = match &cache[1] {
                Some(LayerCache::Attention(KvCache::F16 { k, .. })) => {
                    k.flatten_all()?.to_vec1::<f32>()?
                }
                None => Vec::new(),
                _ => anyhow::bail!("wrong attention test state"),
            };
            values.extend(tokens.iter().map(|token| *token as f32));
            let tensor = Tensor::from_vec(values.clone(), (1, 1, values.len(), 1), &Device::Cpu)?;
            cache[1] = Some(LayerCache::Attention(KvCache::F16 {
                k: tensor.clone(),
                v: tensor,
            }));
            anyhow::ensure!(!failed, "injected recurrent failure after cache writes");
            Ok(StageOutput::Logits(Tensor::new(
                &[total, 0.0],
                &Device::Cpu,
            )?))
        }

        fn family(&self) -> &str {
            "recurrent-test"
        }
        fn hidden_dim(&self) -> usize {
            1
        }
        fn layer_count(&self) -> usize {
            2
        }
        fn vocab_size(&self) -> usize {
            2
        }
    }

    fn recurrent_value(cache: &Option<LayerCache>) -> Result<f32> {
        let (conv, state) = match cache {
            Some(LayerCache::DeltaNet(cache)) => (&cache.conv_state, &cache.recurrent_state),
            Some(LayerCache::Mamba2(cache)) => (&cache.conv_state, &cache.ssm_state),
            None => return Ok(0.0),
            _ => anyhow::bail!("wrong recurrent test state"),
        };
        let value = state.flatten_all()?.to_vec1::<f32>()?[0];
        assert_eq!(conv.flatten_all()?.to_vec1::<f32>()?, [value]);
        Ok(value)
    }

    pub(super) fn recurrent_stage(mamba: bool) -> PipelineStage {
        let mut stage = test_stage();
        stage.arch = Box::new(RecurrentCacheArchitecture { mamba });
        stage.cache = vec![None, None];
        stage.fixed_cache_bytes = 8;
        stage
    }

    fn attention_values(cache: &Option<LayerCache>) -> Result<Vec<f32>> {
        let Some(LayerCache::Attention(KvCache::F16 { k, .. })) = cache else {
            anyhow::bail!("missing attention test cache");
        };
        Ok(k.flatten_all()?.to_vec1::<f32>()?)
    }

    #[test]
    fn full_rolling_eviction_keeps_both_recurrent_state_types() -> Result<()> {
        for mamba in [false, true] {
            let mut stage = recurrent_stage(mamba);
            stage.set_max_kv_tokens(1);
            stage.set_max_total_kv_tokens(6)?; // Two fixed slots and one KV slot.
            for (position, token, expected) in [(0, 1, 1.0), (1, 2, 3.0), (2, 3, 6.0)] {
                let output = stage.forward(StageInput::Tokens(vec![token]), 1, position)?;
                assert_eq!(logits(output)[0], expected, "mamba={mamba}");
                assert_eq!(recurrent_value(&stage.cache[0])?, expected);
                assert_eq!(attention_values(&stage.cache[1])?, [token as f32]);
                assert_eq!(stage.next_position(), position + 1);
                assert_eq!(stage.total_kv_cache_tokens(), 1);
                assert_eq!(stage.total_kv_cache_allocation_tokens(), 3);
                assert_eq!(stage.fixed_cache_allocation_tokens, 2);
            }
            stage.clear_cache();
            assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        }
        Ok(())
    }

    #[test]
    fn full_prompt_chunk_replacement_keeps_recurrence_and_drops_old_attention() -> Result<()> {
        for mamba in [false, true] {
            let mut stage = recurrent_stage(mamba);
            stage.set_max_kv_tokens(2);
            stage.set_max_total_kv_tokens(8)?;
            stage.prefill(StageInput::Tokens(vec![1, 2]), 2, 0)?;
            let output = stage.prefill(StageInput::Tokens(vec![3, 4]), 2, 2)?;
            assert_eq!(logits(output)[0], 10.0, "mamba={mamba}");
            assert_eq!(recurrent_value(&stage.cache[0])?, 10.0);
            assert_eq!(attention_values(&stage.cache[1])?, [3.0, 4.0]);
            assert_eq!(stage.next_position(), 4);
            assert_eq!(stage.total_kv_cache_tokens(), 2);
            assert_eq!(stage.total_kv_cache_allocation_tokens(), 4);
        }
        Ok(())
    }

    #[test]
    fn public_full_eviction_keeps_fixed_recurrence_and_its_charge() -> Result<()> {
        for mamba in [false, true] {
            let mut stage = recurrent_stage(mamba);
            stage.forward(StageInput::Tokens(vec![1, 2]), 2, 0)?;
            stage.evict_oldest(2);
            assert_eq!(recurrent_value(&stage.cache[0])?, 3.0);
            assert!(stage.cache[1].is_none());
            assert_eq!(stage.next_position(), 2);
            assert_eq!(stage.total_kv_cache_tokens(), 0);
            assert_eq!(stage.total_kv_cache_allocation_tokens(), 2);
            let output = stage.forward(StageInput::Tokens(vec![3]), 1, 2)?;
            assert_eq!(logits(output)[0], 6.0, "mamba={mamba}");
            assert_eq!(stage.total_kv_cache_allocation_tokens(), 3);
        }
        Ok(())
    }

    #[test]
    fn failed_full_replacement_restores_both_recurrent_states_and_budget() -> Result<()> {
        for mamba in [false, true] {
            let mut stage = recurrent_stage(mamba);
            stage.set_max_kv_tokens(1);
            stage.set_max_total_kv_tokens(6)?;
            stage.forward(StageInput::Tokens(vec![1]), 1, 0)?;
            assert!(stage.forward(StageInput::Tokens(vec![255]), 1, 1).is_err());
            assert_eq!(recurrent_value(&stage.cache[0])?, 1.0);
            assert_eq!(attention_values(&stage.cache[1])?, [1.0]);
            assert_eq!(stage.next_position(), 1);
            assert_eq!(stage.total_kv_cache_tokens(), 1);
            assert_eq!(stage.total_kv_cache_allocation_tokens(), 3);
            let output = stage.forward(StageInput::Tokens(vec![2]), 1, 1)?;
            assert_eq!(logits(output)[0], 3.0, "mamba={mamba}");
        }
        Ok(())
    }

    #[test]
    fn session_full_replacement_keeps_recurrence_and_rolls_back_only_failed_lane() -> Result<()> {
        for mamba in [false, true] {
            let mut stage = recurrent_stage(mamba);
            stage.set_max_kv_tokens(1);
            stage.set_max_total_kv_tokens(12)?;
            let mut first = stage.new_session_state();
            let mut peer = stage.new_session_state();
            stage.forward_with_session(&mut first, StageInput::Tokens(vec![1]), 1, 0)?;
            stage.forward_with_session(&mut peer, StageInput::Tokens(vec![10]), 1, 0)?;
            assert!(
                stage
                    .forward_with_session(&mut first, StageInput::Tokens(vec![255]), 1, 1)
                    .is_err()
            );
            assert_eq!(recurrent_value(&first.cache[0])?, 1.0);
            assert_eq!(attention_values(&first.cache[1])?, [1.0]);
            assert_eq!(first.next_position, 1);
            assert_eq!(recurrent_value(&peer.cache[0])?, 10.0);
            assert_eq!(peer.next_position, 1);
            assert_eq!(stage.total_kv_cache_allocation_tokens(), 6);
            let output =
                stage.forward_with_session(&mut first, StageInput::Tokens(vec![2]), 1, 1)?;
            assert_eq!(logits(output)[0], 3.0, "mamba={mamba}");
            assert_eq!(attention_values(&first.cache[1])?, [2.0]);
            assert_eq!(stage.next_position(), 0);
            assert_eq!(stage.total_kv_cache_tokens(), 2);
            drop(first);
            assert_eq!(stage.total_kv_cache_allocation_tokens(), 3);
            let output =
                stage.forward_with_session(&mut peer, StageInput::Tokens(vec![20]), 1, 1)?;
            assert_eq!(logits(output)[0], 30.0, "mamba={mamba}");
            stage.clear_session_state(&mut peer);
            assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        }
        Ok(())
    }

    #[test]
    fn fixed_request_state_rejects_a_new_lane_before_allocating() -> Result<()> {
        let mut stage = test_stage();
        stage.fixed_cache_bytes = 256;
        // 64 slots of fixed state plus one attention token exactly fit.
        stage.set_max_total_kv_tokens(130)?;
        let mut first = stage.new_session_state();
        stage.forward_with_session(&mut first, StageInput::Tokens(vec![1]), 1, 0)?;
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 65);
        assert_eq!(first.fixed_cache_allocation_tokens, 64);

        let mut second = stage.new_session_state();
        assert!(
            stage
                .forward_with_session(&mut second, StageInput::Tokens(vec![2]), 1, 0)
                .is_err()
        );
        assert_eq!(second.next_position, 0);
        assert!(second.cache.iter().all(Option::is_none));
        assert_eq!(second.fixed_cache_allocation_tokens, 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 65);

        stage.clear_session_state(&mut first);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        stage.forward_with_session(&mut second, StageInput::Tokens(vec![2]), 1, 0)?;
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 65);
        drop(second);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn fixed_state_cold_batch_checks_budget_before_allocating() -> Result<()> {
        let mut stage = test_stage();
        stage.vocab_size = 2; // SplitCountingArchitecture returns two logits.
        stage.position = StagePosition::Tail;
        stage.arch = Box::new(SplitCountingArchitecture {
            broken_output: false,
        });
        stage.fixed_cache_bytes = 32;
        stage.set_max_total_kv_tokens(34)?; // Seventeen resident slots.
        let input =
            || StageInput::activations(Tensor::zeros((1, 1), DType::F32, &Device::Cpu).unwrap());
        let mut first = stage.new_session_state();
        let mut second = stage.new_session_state();
        assert!(
            stage
                .forward_pipeline_decode_batch_with_sessions(
                    &mut [&mut first, &mut second],
                    &[input(), input()],
                )
                .is_err()
        );
        assert!(first.cache.iter().all(Option::is_none));
        assert!(second.cache.iter().all(Option::is_none));
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);

        stage.set_max_total_kv_tokens(40)?;
        stage.forward_pipeline_decode_batch_with_sessions(
            &mut [&mut first, &mut second],
            &[input(), input()],
        )?;
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 18);
        assert_eq!(first.fixed_cache_allocation_tokens, 8);
        assert_eq!(second.fixed_cache_allocation_tokens, 8);
        stage.forward_pipeline_decode_batch_with_sessions(
            &mut [&mut first, &mut second],
            &[input(), input()],
        )?;
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 20);
        drop(first);
        drop(second);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn fixed_state_is_charged_once_across_decode_and_checkpoint_restore() -> Result<()> {
        let mut stage = test_stage();
        stage.fixed_cache_bytes = 32;
        stage.set_max_total_kv_tokens(256)?;
        let mut state = stage.new_session_state();
        stage.forward_with_session(&mut state, StageInput::Tokens(vec![1]), 1, 0)?;
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 9);
        let checkpoint = stage.checkpoint_session_state(&state)?;
        stage.forward_with_session(&mut state, StageInput::Tokens(vec![2]), 1, 1)?;
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 10);
        assert_eq!(state.fixed_cache_allocation_tokens, 8);
        stage.restore_session_checkpoint(&mut state, checkpoint);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 9);
        drop(state);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn draft_attachment_rejects_live_session_state_before_loading() -> Result<()> {
        let mut stage = test_stage();
        let mut state = stage.new_session_state();
        stage.forward_with_session(&mut state, StageInput::Tokens(vec![1]), 1, 0)?;
        assert_eq!(stage.next_position(), 0);
        let error = stage
            .attach_speculative_draft(std::path::Path::new("missing-draft.gguf"))
            .unwrap_err();
        assert!(error.to_string().contains("before target inference starts"));
        Ok(())
    }

    #[test]
    fn draft_attachment_rejects_idle_session_state_before_loading() {
        let mut stage = test_stage();
        let state = stage.new_session_state();
        let error = stage
            .attach_speculative_draft(std::path::Path::new("missing-draft.gguf"))
            .unwrap_err();
        assert!(error.to_string().contains("session states are created"));
        assert_eq!(state.next_position, 0);
        assert!(state.cache.iter().all(Option::is_none));
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
    }

    #[test]
    fn fixed_state_prompt_forks_charge_and_release_their_own_reservation() -> Result<()> {
        let mut stage = test_stage();
        stage.fixed_cache_bytes = 32;
        stage.set_max_total_kv_tokens(4096)?;
        let mut request = GenerationSession::new(
            &stage,
            vec![0, 1],
            SamplingParams {
                max_tokens: 1,
                ..Default::default()
            },
            vec![],
        );
        request.enable_prompt_cache("owner".into());
        request.step(&mut stage)?;
        request.publish_prompt_cache(&mut stage);
        request.discard_cache(&mut stage);
        let before = stage.total_kv_cache_allocation_tokens();
        assert!(before > 8);
        let (fork, _, _) = stage.reuse_prompt("owner", &[0, 1], true)?.unwrap();
        assert_eq!(fork.fixed_cache_allocation_tokens, 8);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), before + 10);
        drop(fork);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), before);
        stage.clear_prompt_cache();
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    pub(super) fn preallocated_test_stage() -> PipelineStage {
        let mut stage = test_stage();
        stage.arch = Box::new(PreallocatedCacheArchitecture);
        stage.preallocated_kv_cache = true;
        stage
    }

    #[test]
    fn quantized_prediction_retains_evicted_backing_until_replacement() -> Result<()> {
        let mut stage = test_stage();
        stage.quant_config = Some(Arc::new(crate::turboquant::TurboQuantConfig::new(
            8, 64, 42,
        )));
        stage.kv_cache_allocation_tokens = 64;
        stage.kv_token_budget.add_allocated(64);
        assert_eq!(stage.predicted_allocation_tokens(4, false)?, 64);
        assert_eq!(stage.predicted_allocation_tokens(65, false)?, 128);
        assert_eq!(stage.predicted_allocation_tokens(4, true)?, 64);
        Ok(())
    }

    #[test]
    fn quantized_cpu_budget_keeps_evicted_capacity_across_sessions() -> Result<()> {
        let mut stage = test_stage();
        stage.arch = Box::new(QuantizedCacheArchitecture);
        stage.hidden_dim = 64;
        stage.quant_config = Some(Arc::new(crate::turboquant::TurboQuantConfig::new(
            8, 64, 42,
        )));
        stage.set_max_total_kv_tokens(8192)?;

        stage.forward(StageInput::Tokens(vec![1; 4096]), 4096, 0)?;
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 4096);

        stage.evict_oldest(4000);
        stage.forward(StageInput::Tokens(vec![1]), 1, 4096)?;
        assert_eq!(stage.kv_cache_tokens(), 97);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 4096);

        let mut second = stage.new_session_state();
        let error = match stage.forward_with_session(&mut second, StageInput::Tokens(vec![2]), 1, 0)
        {
            Ok(_) => anyhow::bail!("a second session overcommitted retained CPU capacity"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("resident KV allocation"));
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 4096);

        stage.clear_cache();
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        stage.forward_with_session(&mut second, StageInput::Tokens(vec![2]), 1, 0)?;
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 64);
        drop(second);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn deferred_quant_validation_rejects_output_and_rolls_back_stage_state() -> Result<()> {
        use crate::turboquant::TurboQuantConfig;
        struct QuantizedFailureArchitecture {
            device: Device,
        }
        impl ModelArchitecture for QuantizedFailureArchitecture {
            fn forward(
                &self,
                input: StageInput,
                cache: &mut [Option<LayerCache>],
                seq_len: usize,
                _start_pos: usize,
            ) -> Result<StageOutput> {
                let invalid =
                    matches!(input, StageInput::Tokens(ref ids) if ids.first() == Some(&255));
                for (index, layer) in cache.iter_mut().enumerate() {
                    let config = Arc::new(TurboQuantConfig::new(8, 64, 42));
                    let layer = layer.get_or_insert_with(|| {
                        LayerCache::Attention(KvCache::Quantized(
                            crate::turboquant::QuantizedKvCache::new(config, 1),
                        ))
                    });
                    let LayerCache::Attention(KvCache::Quantized(cache)) = layer else {
                        anyhow::bail!("unexpected test cache")
                    };
                    let value = if invalid && index == 1 { f32::NAN } else { 1.0 };
                    let tensor = Tensor::from_vec(
                        vec![value; seq_len * 64],
                        (1, 1, seq_len, 64),
                        &self.device,
                    )?;
                    cache.update_native(&tensor, &tensor)?;
                }
                // Deliberately finite: the deferred flags must independently
                // reject the request, not rely on a later logits scan.
                Ok(StageOutput::Logits(Tensor::ones(
                    2,
                    DType::F32,
                    &self.device,
                )?))
            }
            fn family(&self) -> &str {
                "quant-validation-test"
            }
            fn hidden_dim(&self) -> usize {
                64
            }
            fn layer_count(&self) -> usize {
                2
            }
            fn vocab_size(&self) -> usize {
                2
            }
        }
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let make_stage = || {
            let mut stage = test_stage();
            stage.arch = Box::new(QuantizedFailureArchitecture {
                device: device.clone(),
            });
            stage.device = device.clone();
            stage.cache = vec![None, None];
            stage.quant_config = Some(Arc::new(TurboQuantConfig::new(8, 64, 42)));
            stage
        };
        let mut stage = make_stage();
        stage.forward(StageInput::Tokens(vec![1]), 1, 0)?;
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 64);
        assert_eq!(stage.predicted_allocation_tokens(63, false)?, 64);
        assert_eq!(stage.predicted_allocation_tokens(65, false)?, 128);
        assert!(stage.forward(StageInput::Tokens(vec![255]), 1, 1).is_err());
        assert_eq!(stage.next_position(), 0);
        assert_eq!(stage.kv_cache_tokens(), 0);
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        assert!(stage.cache.iter().all(Option::is_none));
        stage.forward(StageInput::Tokens(vec![1]), 1, 0)?;

        // Full-cache replacement preserves the old request after failed
        // validation, just as it does for an architecture execution failure.
        let mut stage = make_stage();
        stage.max_kv_tokens = Some(1);
        stage.forward(StageInput::Tokens(vec![1]), 1, 0)?;
        assert!(stage.forward(StageInput::Tokens(vec![255]), 1, 1).is_err());
        assert_eq!(stage.next_position(), 1);
        assert_eq!(stage.kv_cache_tokens(), 1);
        assert_eq!(stage.total_kv_cache_tokens(), 1);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 64);
        assert!(stage.cache.iter().all(Option::is_some));
        stage.forward(StageInput::Tokens(vec![1]), 1, 1)?;
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 64);

        // Admission reserves aligned physical storage, not just live tokens.
        // A too-small budget rejects before any layer mutates its cache; a
        // full lane rolls within its existing capacity until growth is allowed.
        let mut stage = make_stage();
        stage.set_max_total_kv_tokens(64)?;
        assert!(stage.forward(StageInput::Tokens(vec![1]), 1, 0).is_err());
        assert_eq!(stage.next_position(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        stage.set_max_total_kv_tokens(128)?;
        stage.forward(StageInput::Tokens(vec![1; 20]), 20, 0)?;
        stage.forward(StageInput::Tokens(vec![1; 45]), 45, 20)?;
        assert_eq!(stage.next_position(), 65);
        assert_eq!(stage.kv_cache_tokens(), 64);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 64);
        stage.forward(StageInput::Tokens(vec![1]), 1, 65)?;
        assert_eq!(stage.next_position(), 66);
        assert_eq!(stage.kv_cache_tokens(), 64);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 64);
        stage.set_max_total_kv_tokens(256)?;
        stage.forward(StageInput::Tokens(vec![1]), 1, 66)?;
        assert_eq!(stage.next_position(), 67);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 128);
        Ok(())
    }

    #[test]
    fn one_token_prompt_chunks_use_prefill_but_sampled_tokens_use_decode() -> Result<()> {
        use crate::continuous::GenerationSession;
        for budget in [1, 2, 128] {
            let seen = Arc::new(Mutex::new(Vec::new()));
            let mut stage = test_stage();
            stage.arch = Box::new(PassRecordingArchitecture { seen: seen.clone() });
            let params = SamplingParams {
                temperature: 0.,
                max_tokens: 3,
                ..Default::default()
            };
            let mut session = GenerationSession::new(&stage, vec![0; 3], params, vec![]);
            while !session.is_finished() {
                session.step_with_prefill_budget(&mut stage, budget)?;
            }
            let calls = seen.lock().unwrap();
            let mut position = 0;
            for &(prefill, len, start) in calls.iter() {
                assert_eq!(start, position);
                assert_eq!(prefill, start < 3);
                assert_eq!(len, if prefill { budget.min(3 - start) } else { 1 });
                position += len;
            }
            assert_eq!(position, 5);
            assert_eq!(stage.total_kv_cache_tokens(), 5);
            assert_eq!(
                stage.next_position(),
                0,
                "session must not move the default cache"
            );
            session.discard_cache(&mut stage);
            assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        }
        Ok(())
    }

    #[test]
    fn cache_only_prefill_keeps_sampling_and_shared_checkpoint_heads() -> Result<()> {
        use crate::continuous::GenerationSession;
        for scope in [false, true] {
            let seen = Arc::new(Mutex::new(Vec::new()));
            let mut stage = test_stage();
            stage.arch = Box::new(CacheOnlyRecordingArchitecture { seen: seen.clone() });
            stage.set_prompt_cache_limits(1_000_000, 4, Duration::from_secs(60));
            let params = SamplingParams {
                temperature: 0.7,
                seed: Some(42),
                max_tokens: 2,
                ..Default::default()
            };
            let mut session = GenerationSession::new(&stage, vec![0; 257], params.clone(), vec![]);
            session.set_shared_prompt_prefix(192);
            if scope {
                session.enable_prompt_cache("owner".into());
            }
            assert!(session.step_with_prefill_budget(&mut stage, 128)?.is_none());
            assert!(session.step_with_prefill_budget(&mut stage, 128)?.is_none());
            assert!(session.generated().is_empty());
            assert!(session.step_with_prefill_budget(&mut stage, 128)?.is_some());
            session.step(&mut stage)?;
            let expected = if scope {
                vec![
                    (true, 128, 0),
                    (false, 64, 128),
                    (true, 64, 192),
                    (false, 1, 256),
                    (false, 1, 257),
                ]
            } else {
                vec![
                    (true, 128, 0),
                    (true, 128, 128),
                    (false, 1, 256),
                    (false, 1, 257),
                ]
            };
            assert_eq!(*seen.lock().unwrap(), expected);
            session.publish_prompt_cache(&mut stage);
            session.discard_cache(&mut stage);
            if scope {
                let mut reused = GenerationSession::new(&stage, vec![0; 192], params, vec![]);
                reused.enable_prompt_cache("owner".into());
                let calls = seen.lock().unwrap().len();
                reused.step(&mut stage)?;
                assert_eq!(reused.reused_prompt_tokens(), 192);
                assert_eq!(
                    seen.lock().unwrap().len(),
                    calls,
                    "checkpoint must have valid logits"
                );
                reused.discard_cache(&mut stage);
            }
            stage.clear_cache();
            assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        }
        Ok(())
    }

    #[test]
    fn cache_only_prefill_failure_clears_failed_lane_and_keeps_other_cache() -> Result<()> {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut stage = test_stage();
        stage.arch = Box::new(CacheOnlyRecordingArchitecture { seen });
        stage.prefill(StageInput::Tokens(vec![0]), 1, 0)?;
        let mut session = stage.new_session_state();
        stage.prefill_cache_only_with_session(&mut session, StageInput::Tokens(vec![0]), 1, 0)?;
        assert_eq!(session.next_position, 1);
        assert_eq!(session.revision, 1);
        assert!(
            stage
                .prefill_cache_only_with_session(&mut session, StageInput::Tokens(vec![255]), 1, 1)
                .is_err()
        );
        assert_eq!(session.next_position, 0);
        assert_eq!(session.revision, 1);
        assert_eq!(stage.next_position(), 1);
        assert_eq!(stage.total_kv_cache_tokens(), 1);
        stage.clear_session_state(&mut session);
        stage.clear_cache();
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn tail_cache_only_prefill_rejects_other_roles_before_mutation() -> Result<()> {
        for role in [
            StagePosition::Solo,
            StagePosition::Head,
            StagePosition::Middle,
        ] {
            let seen = Arc::new(Mutex::new(Vec::new()));
            let mut stage = test_stage();
            stage.arch = Box::new(CacheOnlyRecordingArchitecture { seen: seen.clone() });
            stage.position = role;
            let mut session = stage.new_session_state();
            stage.prefill_with_session(&mut session, StageInput::Tokens(vec![0]), 1, 0)?;
            let calls = seen.lock().unwrap().len();
            let error = stage
                .prefill_tail_cache_only_with_session(
                    &mut session,
                    StageInput::Tokens(vec![1]),
                    1,
                    1,
                )
                .unwrap_err();
            assert!(error.to_string().contains("requires a tail stage"));
            assert_eq!(seen.lock().unwrap().len(), calls);
            assert_eq!(session.next_position, 1);
            assert_eq!(session.revision, 1);
            assert_eq!(stage.total_kv_cache_tokens(), 1);
            assert_eq!(stage.total_kv_cache_allocation_tokens(), 1);
            assert_eq!(stage.next_position(), 0);
            stage.clear_session_state(&mut session);
        }
        Ok(())
    }

    #[test]
    fn tail_cache_only_prefill_keeps_budgets_and_clears_only_failed_session() -> Result<()> {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut stage = test_stage();
        stage.arch = Box::new(CacheOnlyRecordingArchitecture { seen: seen.clone() });
        stage.position = StagePosition::Tail;
        stage.set_max_total_kv_tokens(6)?;
        let mut first = stage.new_session_state();
        let mut second = stage.new_session_state();
        stage.prefill_tail_cache_only_with_session(
            &mut first,
            StageInput::Tokens(vec![0]),
            1,
            0,
        )?;
        stage.prefill_tail_cache_only_with_session(
            &mut second,
            StageInput::Tokens(vec![0, 1]),
            2,
            0,
        )?;
        assert_eq!(*seen.lock().unwrap(), [(true, 1, 0), (true, 2, 0)]);
        assert_eq!(stage.total_kv_cache_tokens(), 3);
        let mut rejected = stage.new_session_state();
        assert!(
            stage
                .prefill_tail_cache_only_with_session(
                    &mut rejected,
                    StageInput::Tokens(vec![0]),
                    1,
                    0,
                )
                .is_err()
        );
        assert_eq!(
            seen.lock().unwrap().len(),
            2,
            "budget rejection precedes model work"
        );
        assert_eq!(rejected.revision, 0);
        assert!(rejected.cache.iter().all(Option::is_none));
        stage.set_max_total_kv_tokens(16)?;
        assert!(
            stage
                .prefill_tail_cache_only_with_session(
                    &mut first,
                    StageInput::Tokens(vec![255]),
                    1,
                    1,
                )
                .is_err()
        );
        assert_eq!(first.next_position, 0);
        assert_eq!(first.revision, 1);
        assert!(first.cache.iter().all(Option::is_none));
        assert_eq!(second.next_position, 2);
        assert_eq!(stage.total_kv_cache_tokens(), 2);
        assert_eq!(stage.next_position(), 0);
        stage.clear_session_state(&mut second);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn tail_cache_only_prefill_replacement_keeps_old_state_after_failure() -> Result<()> {
        let mut stage = test_stage();
        stage.position = StagePosition::Tail;
        stage.max_kv_tokens = Some(1);
        let mut session = stage.new_session_state();
        stage.prefill_tail_cache_only_with_session(
            &mut session,
            StageInput::Tokens(vec![0]),
            1,
            0,
        )?;
        assert!(
            stage
                .prefill_tail_cache_only_with_session(
                    &mut session,
                    StageInput::Tokens(vec![255]),
                    1,
                    1,
                )
                .is_err()
        );
        assert_eq!(session.next_position, 1);
        assert_eq!(session.revision, 1);
        assert_eq!(stage.total_kv_cache_tokens(), 1);
        stage.prefill_tail_cache_only_with_session(
            &mut session,
            StageInput::Tokens(vec![1]),
            1,
            1,
        )?;
        assert_eq!(session.next_position, 2);
        assert_eq!(stage.session_kv_cache_tokens(&session), 1);
        stage.clear_session_state(&mut session);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn tail_cache_only_prefill_completes_metal_work_before_success() -> Result<()> {
        struct RetiringArchitecture {
            device: Device,
            retired: Arc<Mutex<Option<std::sync::Weak<candle_metal_kernels::metal::Buffer>>>>,
        }
        impl ModelArchitecture for RetiringArchitecture {
            fn forward(
                &self,
                input: StageInput,
                cache: &mut [Option<LayerCache>],
                len: usize,
                pos: usize,
            ) -> Result<StageOutput> {
                CacheCountingArchitecture.forward(input, cache, len, pos)
            }
            fn forward_prefill_cache_only(
                &self,
                input: StageInput,
                cache: &mut [Option<LayerCache>],
                len: usize,
                pos: usize,
            ) -> Result<StageOutput> {
                self.forward(input, cache, len, pos)?;
                let output = Tensor::zeros((len, 1), DType::F32, &self.device)?;
                let buffer = self
                    .device
                    .as_metal_device()?
                    .new_buffer_with_data(&[0u8; 4096])?;
                let retired = Arc::downgrade(&buffer);
                drop(buffer);
                assert!(retired.upgrade().is_some());
                *self.retired.lock().unwrap() = Some(retired);
                Ok(StageOutput::Activations(output))
            }
            fn family(&self) -> &str {
                "retiring-tail-test"
            }
            fn hidden_dim(&self) -> usize {
                1
            }
            fn layer_count(&self) -> usize {
                1
            }
            fn vocab_size(&self) -> usize {
                2
            }
        }
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let retired = Arc::new(Mutex::new(None));
        let mut stage = test_stage();
        stage.position = StagePosition::Tail;
        stage.device = device.clone();
        stage.arch = Box::new(RetiringArchitecture {
            device,
            retired: retired.clone(),
        });
        let mut state = stage.new_session_state();
        stage.prefill_tail_cache_only_with_session(
            &mut state,
            StageInput::Tokens(vec![0]),
            1,
            0,
        )?;
        assert!(
            retired
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .upgrade()
                .is_none()
        );
        assert_eq!(state.next_position, 1);
        assert_eq!(state.revision, 1);
        stage.clear_session_state(&mut state);
        Ok(())
    }

    #[test]
    fn one_token_cached_prompt_tail_still_uses_prefill() -> Result<()> {
        use crate::continuous::GenerationSession;
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut stage = test_stage();
        stage.arch = Box::new(PassRecordingArchitecture { seen: seen.clone() });
        stage.set_prompt_cache_limits(1024, 4, Duration::from_secs(60));
        let params = SamplingParams {
            temperature: 0.,
            max_tokens: 2,
            ..Default::default()
        };
        let mut cold = GenerationSession::new(&stage, vec![0, 1], params.clone(), vec![]);
        cold.enable_prompt_cache("owner".into());
        while !cold.is_finished() {
            cold.step(&mut stage)?;
        }
        cold.publish_prompt_cache(&mut stage);
        cold.discard_cache(&mut stage);
        seen.lock().unwrap().clear();
        let mut warm = GenerationSession::new(&stage, vec![0, 1, 0], params, vec![]);
        warm.enable_prompt_cache("owner".into());
        while !warm.is_finished() {
            warm.step(&mut stage)?;
        }
        assert_eq!(*seen.lock().unwrap(), [(true, 1, 2), (false, 1, 3)]);
        warm.discard_cache(&mut stage);
        stage.clear_cache();
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn prefill_replacement_and_failure_keep_cache_accounting() -> Result<()> {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut stage = test_stage();
        stage.arch = Box::new(PassRecordingArchitecture { seen: seen.clone() });
        stage.max_kv_tokens = Some(1);
        stage.prefill(StageInput::Tokens(vec![1]), 1, 0)?;
        assert!(stage.prefill(StageInput::Tokens(vec![255]), 1, 1).is_err());
        assert_eq!(stage.next_position(), 1);
        assert_eq!(stage.total_kv_cache_tokens(), 1);
        stage.prefill(StageInput::Tokens(vec![2]), 1, 1)?;
        assert_eq!(stage.next_position(), 2);
        assert_eq!(stage.total_kv_cache_tokens(), 1);
        stage.forward(StageInput::Tokens(vec![3]), 1, 2)?;
        assert_eq!(
            *seen.lock().unwrap(),
            [(true, 1, 0), (true, 1, 1), (true, 1, 1), (false, 1, 2)]
        );
        stage.clear_cache();
        let mut session = stage.new_session_state();
        assert!(
            stage
                .prefill_with_session(&mut session, StageInput::Tokens(vec![255]), 1, 0)
                .is_err()
        );
        assert_eq!(session.next_position, 0);
        assert_eq!(session.revision, 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        stage.prefill_with_session(&mut session, StageInput::Tokens(vec![0]), 1, 0)?;
        assert_eq!(session.revision, 1);
        assert_eq!(session.next_position, 1);
        assert_eq!(stage.next_position(), 0);
        drop(session);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    fn position_recording_stage(seen: Arc<Mutex<Vec<Option<MropePositionIds>>>>) -> PipelineStage {
        let mut stage = test_stage();
        stage.arch = Box::new(PositionRecordingArchitecture { seen });
        stage.cache.clear();
        stage
    }

    type BoundaryCalls = Arc<Mutex<Vec<(usize, usize, bool)>>>;

    struct SpeculativeBoundaryArchitecture {
        context_length: usize,
        forwards: BoundaryCalls,
        capture_width: usize,
    }

    impl SpeculativeBoundaryArchitecture {
        fn run(
            &self,
            input: StageInput,
            cache: &mut [Option<LayerCache>],
            seq_len: usize,
            start_pos: usize,
            all_logits: bool,
        ) -> Result<StageOutput> {
            anyhow::ensure!(
                start_pos + seq_len <= self.context_length,
                "test rotary context exhausted"
            );
            let StageInput::Tokens(tokens) = input else {
                anyhow::bail!("boundary test requires tokens");
            };
            anyhow::ensure!(tokens.len() == seq_len, "test token count differs");
            self.forwards
                .lock()
                .unwrap()
                .push((seq_len, start_pos, all_logits));
            let mut values = match &cache[0] {
                Some(LayerCache::Attention(KvCache::F16 { k, .. })) => {
                    k.flatten_all()?.to_vec1::<f32>()?
                }
                None => Vec::new(),
                Some(_) => anyhow::bail!("unexpected boundary cache"),
            };
            let mut rows = Vec::with_capacity(seq_len * 3);
            for token in tokens {
                values.push(token as f32);
                // An old prompt token affects every target row. Losing it
                // before verification changes the next token immediately.
                if values.contains(&2.0) {
                    rows.extend([0.0f32, 10.0, -100.0]);
                } else {
                    rows.extend([10.0f32, 0.0, -100.0]);
                }
            }
            let retained = values.len();
            let tensor = Tensor::from_vec(values, (1, 1, retained, 1), &Device::Cpu)?;
            cache[0] = Some(LayerCache::Attention(KvCache::F16 {
                k: tensor.clone(),
                v: tensor,
            }));
            let logits = Tensor::from_vec(rows, (seq_len, 3), &Device::Cpu)?;
            Ok(StageOutput::Logits(if all_logits {
                logits
            } else {
                logits.get(seq_len - 1)?
            }))
        }
    }

    impl ModelArchitecture for SpeculativeBoundaryArchitecture {
        fn forward(
            &self,
            input: StageInput,
            cache: &mut [Option<LayerCache>],
            seq_len: usize,
            start_pos: usize,
        ) -> Result<StageOutput> {
            self.run(input, cache, seq_len, start_pos, false)
        }

        fn forward_with_layer_inputs(
            &self,
            input: StageInput,
            cache: &mut [Option<LayerCache>],
            seq_len: usize,
            start_pos: usize,
            layer_ids: &[usize],
            all_logits: bool,
        ) -> Result<(StageOutput, Vec<Tensor>)> {
            let output = self.run(input, cache, seq_len, start_pos, all_logits)?;
            let captures = layer_ids
                .iter()
                .map(|_| Tensor::zeros((seq_len, self.capture_width), DType::F32, &Device::Cpu))
                .collect::<candle_core::Result<Vec<_>>>()?;
            Ok((output, captures))
        }

        fn family(&self) -> &str {
            "speculative-boundary-test"
        }
        fn hidden_dim(&self) -> usize {
            64
        }
        fn layer_count(&self) -> usize {
            2
        }
        fn vocab_size(&self) -> usize {
            3
        }
    }

    pub(super) fn speculative_boundary_stage(
        target_context: usize,
        draft_context: Option<usize>,
    ) -> Result<(PipelineStage, BoundaryCalls)> {
        let forwards = Arc::new(Mutex::new(Vec::new()));
        let mut stage = test_stage();
        stage.arch = Box::new(SpeculativeBoundaryArchitecture {
            context_length: target_context,
            forwards: Arc::clone(&forwards),
            capture_width: 64,
        });
        stage.context_length = target_context;
        stage.hidden_dim = 64;
        stage.vocab_size = 3;
        stage.cache = vec![None, None];
        stage.kv_budget_bytes_per_token = 1024;
        if let Some(context) = draft_context {
            let draft = crate::architectures::dflash2::DFlash2Draft::test_zero_draft(2, context)?;
            stage.fixed_cache_bytes = draft.cache_budget_bytes()?;
            stage.speculative_cache = Some(draft.new_cache());
            stage.speculative = Some(draft);
        }
        Ok((stage, forwards))
    }

    #[test]
    fn speculative_prefill_injection_error_restores_both_caches() -> Result<()> {
        let (mut stage, calls) = speculative_boundary_stage(32, Some(32))?;
        let mut state = stage.new_session_state();
        stage.speculative_prefill_with_session(&mut state, StageInput::Tokens(vec![2]), 1, 0)?;
        let old_values = session_cached_values(&state);
        let old_allocated = stage.total_kv_cache_allocation_tokens();
        let old_draft_bytes = state.speculative_cache.as_ref().unwrap().live_bytes();
        // Fail in draft.inject after the real target pass appended its token.
        stage.arch = Box::new(SpeculativeBoundaryArchitecture {
            context_length: 32,
            forwards: calls,
            capture_width: 63,
        });
        let error = stage
            .speculative_prefill_with_session(&mut state, StageInput::Tokens(vec![0]), 1, 1)
            .err()
            .expect("bad captured feature must reject draft injection");
        assert!(error.to_string().contains("feature shape"));
        assert_eq!(state.next_position, 1);
        assert_eq!(state.kv_cache_tokens, 1);
        assert_eq!(session_cached_values(&state), old_values);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), old_allocated);
        assert_eq!(state.speculative_cache.as_ref().unwrap().next_position(), 1);
        assert_eq!(
            state.speculative_cache.as_ref().unwrap().live_bytes(),
            old_draft_bytes
        );
        assert!(stage.session_speculative_ready(&state));
        assert_eq!(
            stage.next_position, 0,
            "request failure changed the stage cache"
        );
        stage.clear_session_state(&mut state);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    fn boundary_params(max_tokens: usize) -> SamplingParams {
        SamplingParams {
            temperature: 0.8,
            top_k: 1,
            repetition_penalty: 1.1,
            frequency_penalty: 0.1,
            max_tokens,
            min_tokens: max_tokens,
            // Top-k one keeps this synthetic run deterministic while None
            // lets it reach draft context/budget preflight.
            seed: None,
            ..SamplingParams::default()
        }
    }

    fn finish_boundary_session(
        stage: &mut PipelineStage,
        session: &mut crate::continuous::GenerationSession,
    ) -> Result<()> {
        while !session.is_finished() {
            session.step(stage)?;
        }
        Ok(())
    }

    #[test]
    fn speculative_window_fallback_preserves_target_context_and_sampled_tokens() -> Result<()> {
        use crate::continuous::GenerationSession;
        let (mut plain, _) = speculative_boundary_stage(32, None)?;
        let (mut drafted, forwards) = speculative_boundary_stage(32, Some(32))?;
        plain.set_max_kv_tokens(4);
        drafted.set_max_kv_tokens(4);
        let mut reference =
            GenerationSession::new(&plain, vec![2, 0, 0], boundary_params(4), vec![]);
        let mut session =
            GenerationSession::new(&drafted, vec![2, 0, 0], boundary_params(4), vec![]);
        finish_boundary_session(&mut plain, &mut reference)?;
        finish_boundary_session(&mut drafted, &mut session)?;
        assert_eq!(session.generated(), reference.generated());
        assert_eq!(session.speculative_stats().passes, 0);
        assert!(forwards.lock().unwrap().iter().all(|(_, _, all)| !all));
        assert_eq!(
            session.next_position(&drafted),
            reference.next_position(&plain)
        );
        assert_eq!(
            session.kv_cache_tokens(&drafted),
            reference.kv_cache_tokens(&plain)
        );
        session.discard_cache(&mut drafted);
        reference.discard_cache(&mut plain);
        assert_eq!(drafted.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn speculative_context_tail_uses_only_needed_rows() -> Result<()> {
        use crate::continuous::GenerationSession;
        for (target_context, draft_context) in [(5, 32), (32, 4)] {
            let (mut plain, _) = speculative_boundary_stage(target_context, None)?;
            let (mut drafted, forwards) =
                speculative_boundary_stage(target_context, Some(draft_context))?;
            let mut reference =
                GenerationSession::new(&plain, vec![2, 0, 0], boundary_params(2), vec![]);
            let mut session =
                GenerationSession::new(&drafted, vec![2, 0, 0], boundary_params(2), vec![]);
            finish_boundary_session(&mut plain, &mut reference)?;
            finish_boundary_session(&mut drafted, &mut session)?;
            assert_eq!(session.generated(), reference.generated());
            assert_eq!(session.speculative_stats().passes, 0);
            assert_eq!(*forwards.lock().unwrap(), [(3, 0, false), (1, 3, false)]);
            session.discard_cache(&mut drafted);
            assert_eq!(drafted.total_kv_cache_allocation_tokens(), 0);
        }
        Ok(())
    }

    #[test]
    fn speculative_budget_fallback_keeps_peer_state_and_sampled_tokens() -> Result<()> {
        use crate::continuous::GenerationSession;
        let (mut plain, _) = speculative_boundary_stage(32, None)?;
        plain.set_max_kv_tokens(4);
        let mut reference =
            GenerationSession::new(&plain, vec![2, 0, 0], boundary_params(4), vec![]);
        finish_boundary_session(&mut plain, &mut reference)?;

        let (mut drafted, _) = speculative_boundary_stage(32, Some(32))?;
        let mut peer = drafted.new_session_state();
        drafted.forward_with_session(&mut peer, StageInput::Tokens(vec![0; 2]), 2, 0)?;
        let mut session =
            GenerationSession::new(&drafted, vec![2, 0, 0], boundary_params(4), vec![]);
        session.step(&mut drafted)?;
        drafted.set_max_total_kv_tokens((drafted.total_kv_cache_allocation_tokens() + 1) * 2)?;
        finish_boundary_session(&mut drafted, &mut session)?;
        assert_eq!(session.generated(), reference.generated());
        assert_eq!(session.speculative_stats().passes, 0);
        assert_eq!(peer.next_position, 2);
        assert_eq!(session_cached_values(&peer), [0.0, 0.0]);
        session.discard_cache(&mut drafted);
        drafted.clear_session_state(&mut peer);
        assert_eq!(drafted.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn speculative_target_context_overflow_fails_before_prefill() -> Result<()> {
        use crate::continuous::GenerationSession;
        let (mut plain, _) = speculative_boundary_stage(4, None)?;
        let (mut drafted, forwards) = speculative_boundary_stage(4, Some(32))?;
        let mut reference =
            GenerationSession::new(&plain, vec![2, 0, 0], boundary_params(4), vec![]);
        let mut session =
            GenerationSession::new(&drafted, vec![2, 0, 0], boundary_params(4), vec![]);
        assert!(session.step(&mut drafted).is_err());
        assert!(reference.step(&mut plain).is_err());
        assert!(session.generated().is_empty());
        assert!(reference.generated().is_empty());
        assert!(forwards.lock().unwrap().is_empty());
        assert_eq!(drafted.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn speculative_context_fallback_can_outlive_the_draft() -> Result<()> {
        use crate::continuous::GenerationSession;
        for prompt in [vec![2, 0, 0], vec![2, 0, 0, 0, 0]] {
            let (mut plain, _) = speculative_boundary_stage(16, None)?;
            let (mut drafted, _) = speculative_boundary_stage(16, Some(4))?;
            let mut reference =
                GenerationSession::new(&plain, prompt.clone(), boundary_params(4), vec![]);
            let mut session = GenerationSession::new(&drafted, prompt, boundary_params(4), vec![]);
            finish_boundary_session(&mut plain, &mut reference)?;
            finish_boundary_session(&mut drafted, &mut session)?;
            assert_eq!(session.generated(), reference.generated());
            assert_eq!(session.speculative_stats().passes, 0);
            session.discard_cache(&mut drafted);
            assert_eq!(drafted.total_kv_cache_allocation_tokens(), 0);
        }
        Ok(())
    }

    #[test]
    fn speculative_room_keeps_drafting_until_only_one_output_remains() -> Result<()> {
        use crate::continuous::GenerationSession;
        let (mut plain, _) = speculative_boundary_stage(32, None)?;
        let (mut drafted, forwards) = speculative_boundary_stage(32, Some(32))?;
        let mut params = boundary_params(3);
        params.seed = None;
        let mut reference = GenerationSession::new(&plain, vec![2], params.clone(), vec![]);
        let mut session = GenerationSession::new(&drafted, vec![2], params, vec![]);
        finish_boundary_session(&mut plain, &mut reference)?;
        finish_boundary_session(&mut drafted, &mut session)?;
        assert_eq!(session.generated(), reference.generated());
        assert_eq!(session.speculative_stats().passes, 1);
        assert_eq!(
            forwards
                .lock()
                .unwrap()
                .iter()
                .filter(|(_, _, all)| *all)
                .count(),
            1
        );
        assert_eq!(session.next_position(&drafted), 3);
        session.discard_cache(&mut drafted);
        assert_eq!(drafted.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    fn logits(output: StageOutput) -> Vec<f32> {
        let StageOutput::Logits(logits) = output else {
            panic!("expected logits")
        };
        logits.to_vec1().unwrap()
    }

    fn cached_values(stage: &PipelineStage) -> Vec<f32> {
        let Some(LayerCache::Attention(KvCache::F16 { k, .. })) = &stage.cache[0] else {
            return Vec::new();
        };
        k.flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    fn session_cached_values(state: &super::PipelineStageSessionState) -> Vec<f32> {
        let Some(LayerCache::Attention(KvCache::F16 { k, .. })) = &state.cache[0] else {
            return Vec::new();
        };
        k.flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    #[test]
    fn interleaved_sessions_keep_independent_cache_positions() {
        let mut stage = test_stage();
        let mut first = stage.new_session_state();
        let mut second = stage.new_session_state();

        stage
            .forward_with_session(&mut first, StageInput::Tokens(vec![1, 2]), 2, 0)
            .unwrap();
        stage
            .forward_with_session(&mut second, StageInput::Tokens(vec![3]), 1, 0)
            .unwrap();
        stage
            .forward_with_session(&mut first, StageInput::Tokens(vec![4]), 1, 2)
            .unwrap();

        assert_eq!(stage.session_kv_cache_tokens(&first), 3);
        assert_eq!(stage.session_kv_cache_tokens(&second), 1);
        assert_eq!(stage.session_next_position(&first), 3);
        assert_eq!(stage.session_next_position(&second), 1);
        assert_eq!(stage.kv_cache_tokens(), 0);
        assert_eq!(stage.next_position(), 0);
    }

    #[test]
    fn batched_decode_advances_each_session_cache_once() {
        let mut stage = test_stage();
        let mut first = stage.new_session_state();
        let mut second = stage.new_session_state();
        stage
            .forward_with_session(&mut first, StageInput::Tokens(vec![1, 2]), 2, 0)
            .unwrap();
        stage
            .forward_with_session(&mut second, StageInput::Tokens(vec![3, 4]), 2, 0)
            .unwrap();

        let mut sessions = vec![&mut first, &mut second];
        let logits = stage
            .forward_decode_batch_with_sessions(&mut sessions, &[5, 6])
            .unwrap();
        assert_eq!(logits.len(), 2);
        assert_eq!(stage.session_kv_cache_tokens(&first), 3);
        assert_eq!(stage.session_kv_cache_tokens(&second), 3);
        assert_eq!(stage.decode_batch_caches.len(), 1);

        let mut sessions = vec![&mut first, &mut second];
        stage
            .forward_decode_batch_with_sessions(&mut sessions, &[7, 8])
            .unwrap();
        assert_eq!(stage.decode_batch_caches.len(), 1);
        stage.clear_cache();
        assert!(stage.decode_batch_caches.is_empty());
        stage
            .forward_with_session(&mut first, StageInput::Tokens(vec![9]), 1, 4)
            .unwrap();
        assert!(stage.decode_batch_caches.is_empty());
        assert_eq!(stage.kv_cache_tokens(), 0);
    }

    #[test]
    fn failed_decode_batch_discards_partial_caches_and_keeps_other_sessions() {
        for reuse in [false, true] {
            let mut stage = test_stage();
            let mut first = stage.new_session_state();
            let mut second = stage.new_session_state();
            let mut other = stage.new_session_state();
            for state in [&mut first, &mut second, &mut other] {
                stage
                    .forward_with_session(state, StageInput::Tokens(vec![1, 2]), 2, 0)
                    .unwrap();
            }
            if reuse {
                stage
                    .forward_decode_batch_with_sessions(&mut [&mut first, &mut second], &[3, 4])
                    .unwrap();
            }
            let error = stage
                .forward_decode_batch_with_sessions(&mut [&mut first, &mut second], &[5, 255])
                .unwrap_err();
            assert!(error.to_string().contains("injected forward failure"));
            for state in [&first, &second] {
                assert_eq!(state.next_position, 0, "reuse={reuse}");
                assert_eq!(state.kv_cache_tokens, 0);
                assert_eq!(state.kv_cache_allocation_tokens, 0);
                assert!(state.cache.iter().all(Option::is_none));
            }
            assert!(stage.decode_batch_caches.is_empty());
            assert_eq!(session_cached_values(&other), vec![1., 2.]);
            assert_eq!(stage.total_kv_cache_tokens(), 2);
            assert_eq!(stage.total_kv_cache_allocation_tokens(), 2);
            // A new prefill must work after the failed batch.
            stage
                .forward_with_session(&mut first, StageInput::Tokens(vec![7]), 1, 0)
                .unwrap();
            assert_eq!(session_cached_values(&first), vec![7.]);
        }
    }

    #[test]
    fn wrong_decode_batch_output_count_discards_mutated_sessions() {
        for extra in [false, true] {
            let mut stage = test_stage();
            stage.arch = Box::new(WrongBatchOutputArchitecture { extra });
            let mut first = stage.new_session_state();
            let mut second = stage.new_session_state();
            let error = stage
                .forward_decode_batch_with_sessions(&mut [&mut first, &mut second], &[1, 2])
                .unwrap_err();
            assert!(error.to_string().contains("output count"));
            assert_eq!(stage.total_kv_cache_tokens(), 0);
            assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
            assert!(first.cache.iter().all(Option::is_none));
            assert!(second.cache.iter().all(Option::is_none));
            assert!(stage.decode_batch_caches.is_empty());
        }
    }

    #[test]
    fn decode_batch_position_overflow_fails_before_execution_without_a_budget() {
        let mut stage = test_stage();
        let mut state = stage.new_session_state();
        state.next_position = usize::MAX;
        state.kv_cache_tokens = usize::MAX;
        state.kv_token_budget.add(usize::MAX);
        let error = stage
            .forward_decode_batch_with_sessions(&mut [&mut state], &[1])
            .unwrap_err();
        assert!(error.to_string().contains("position overflow"));
        assert!(state.cache.iter().all(Option::is_none));
        assert_eq!(state.next_position, usize::MAX);
        drop(state);
        assert_eq!(stage.total_kv_cache_tokens(), 0);
    }

    #[test]
    fn dropped_decode_groups_do_not_accumulate_in_stage_cache() {
        let mut stage = test_stage();
        for _ in 0..3 {
            let mut first = stage.new_session_state();
            let mut second = stage.new_session_state();
            stage
                .forward_decode_batch_with_sessions(&mut [&mut first, &mut second], &[1, 2])
                .unwrap();
            assert_eq!(stage.decode_batch_caches.len(), 1);
            drop((first, second));
            assert_eq!(stage.total_kv_cache_tokens(), 0);
            assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        }
    }

    #[test]
    fn packed_cache_lifetime_follows_sessions_not_the_stage() {
        let mut stage = test_stage();
        let mut first = stage.new_session_state();
        let mut second = stage.new_session_state();
        stage
            .forward_decode_batch_with_sessions(&mut [&mut first, &mut second], &[1, 2])
            .unwrap();
        let packed = stage.decode_batch_caches.values().next().unwrap().clone();
        assert_eq!(packed.strong_count(), 2);
        drop(first);
        assert_eq!(packed.strong_count(), 1);
        assert_eq!(stage.total_kv_cache_tokens(), 1);
        drop(second);
        assert!(packed.upgrade().is_none());
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
    }

    #[test]
    fn regrouped_sessions_release_each_packed_cache_when_its_last_owner_leaves() {
        let mut stage = test_stage();
        let mut first = stage.new_session_state();
        let mut second = stage.new_session_state();
        let mut third = stage.new_session_state();
        stage
            .forward_decode_batch_with_sessions(&mut [&mut first, &mut second], &[1, 2])
            .unwrap();
        let old_group = stage.decode_batch_caches.values().next().unwrap().clone();
        stage
            .forward_decode_batch_with_sessions(&mut [&mut second, &mut third], &[3, 4])
            .unwrap();
        let new_group = stage.decode_batch_caches.values().next().unwrap().clone();
        assert_eq!(stage.decode_batch_caches.len(), 1);
        assert_eq!(old_group.strong_count(), 1);
        assert_eq!(new_group.strong_count(), 2);
        drop(first);
        assert!(old_group.upgrade().is_none());
        stage.clear_session_state(&mut second);
        assert_eq!(new_group.strong_count(), 1);
        assert_eq!(session_cached_values(&third), vec![4.]);
        drop(third);
        assert!(new_group.upgrade().is_none());
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
    }

    #[test]
    fn restarting_a_cached_generation_releases_its_packed_group() {
        let mut stage = test_stage();
        let mut first = stage.new_session_state();
        let mut second = stage.new_session_state();
        stage
            .forward_decode_batch_with_sessions(&mut [&mut first, &mut second], &[1, 2])
            .unwrap();
        let packed = stage.decode_batch_caches.values().next().unwrap().clone();
        drop(second);
        let generation = crate::continuous::GenerationSession::with_cache(
            first,
            vec![0],
            SamplingParams::default(),
            vec![],
            1,
        );
        assert!(packed.upgrade().is_none());
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        assert_eq!(generation.into_cache().next_position, 0);
    }

    #[test]
    fn invalid_batch_admission_keeps_live_group_but_scalar_decode_detaches_it() {
        let mut stage = test_stage();
        let mut first = stage.new_session_state();
        let mut second = stage.new_session_state();
        stage
            .forward_decode_batch_with_sessions(&mut [&mut first, &mut second], &[1, 2])
            .unwrap();
        let packed = stage.decode_batch_caches.values().next().unwrap().clone();
        stage.set_max_kv_tokens(1);
        assert!(
            stage
                .forward_decode_batch_with_sessions(&mut [&mut first, &mut second], &[3, 4])
                .is_err()
        );
        assert_eq!(packed.strong_count(), 2);
        stage
            .forward_with_session(&mut first, StageInput::Tokens(vec![3]), 1, 1)
            .unwrap();
        assert_eq!(packed.strong_count(), 1);
        stage.clear_session_state(&mut second);
        assert!(packed.upgrade().is_none());
        assert_eq!(stage.total_kv_cache_tokens(), 1);
    }

    #[test]
    fn continuous_batch_failure_is_terminal_after_execution() {
        use crate::continuous::{DecodeBatchStepError, GenerationSession};
        let mut stage = test_stage();
        let params = SamplingParams {
            temperature: 0.,
            max_tokens: 3,
            ..Default::default()
        };
        let mut first = GenerationSession::new(&stage, vec![0], params.clone(), vec![]);
        let mut second = GenerationSession::new(&stage, vec![0], params, vec![]);
        first.step(&mut stage).unwrap();
        second.step(&mut stage).unwrap();
        stage.arch = Box::new(WrongBatchOutputArchitecture { extra: false });
        assert!(matches!(
            GenerationSession::step_decode_batch(&mut stage, &mut [&mut first, &mut second]),
            Err(DecodeBatchStepError::Terminal(_))
        ));
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        assert_eq!(first.generated(), &[1]);
        assert_eq!(second.generated(), &[1]);
    }

    #[test]
    fn continuous_batch_admission_failure_can_retry_scalar_without_losing_prompt() {
        use crate::continuous::{DecodeBatchStepError, GenerationSession};
        let mut stage = test_stage();
        stage.set_max_total_kv_tokens(8).unwrap();
        let params = SamplingParams {
            temperature: 0.,
            max_tokens: 3,
            ..Default::default()
        };
        let mut first = GenerationSession::new(&stage, vec![0], params.clone(), vec![]);
        let mut second = GenerationSession::new(&stage, vec![0], params, vec![]);
        first.step(&mut stage).unwrap();
        second.step(&mut stage).unwrap();
        assert!(matches!(
            GenerationSession::step_decode_batch(&mut stage, &mut [&mut first, &mut second]),
            Err(DecodeBatchStepError::RetryScalar(_))
        ));
        assert_eq!(stage.total_kv_cache_tokens(), 2);
        first.step(&mut stage).unwrap();
        second.step(&mut stage).unwrap();
        assert_eq!(stage.total_kv_cache_tokens(), 4);
    }

    #[test]
    fn ragged_decode_advances_each_lane_from_its_own_position() {
        let mut stage = test_stage();
        let mut short = stage.new_session_state();
        let mut long = stage.new_session_state();
        stage
            .forward_with_session(&mut short, StageInput::Tokens(vec![1, 2]), 2, 0)
            .unwrap();
        stage
            .forward_with_session(&mut long, StageInput::Tokens(vec![1, 2, 3, 4]), 4, 0)
            .unwrap();

        let mut sessions = vec![&mut short, &mut long];
        let logits = stage
            .forward_decode_batch_with_sessions(&mut sessions, &[5, 6])
            .unwrap();
        assert_eq!(logits.len(), 2);
        assert_eq!(stage.session_kv_cache_tokens(&short), 3);
        assert_eq!(stage.session_kv_cache_tokens(&long), 5);
        assert_eq!(stage.decode_batch_caches.len(), 1);

        let mut sessions = vec![&mut short, &mut long];
        stage
            .forward_decode_batch_with_sessions(&mut sessions, &[7, 8])
            .unwrap();
        assert_eq!(stage.session_kv_cache_tokens(&short), 4);
        assert_eq!(stage.session_kv_cache_tokens(&long), 6);
        assert_eq!(stage.decode_batch_caches.len(), 1);
    }

    #[test]
    fn decode_batch_falls_back_when_lane_requires_kv_eviction() {
        let mut stage = test_stage();
        stage.set_max_kv_tokens(2);
        assert!(stage.supports_decode_batch());
        assert!(stage.can_decode_batch_at(1));
        assert!(!stage.can_decode_batch_at(2));
    }

    #[test]
    fn prompt_larger_than_kv_capacity_is_rejected_without_allocating_cache() {
        let mut stage = test_stage();
        stage.set_max_kv_tokens(2);

        let Err(error) = stage.forward(StageInput::Tokens(vec![1, 2, 3]), 3, 0) else {
            panic!("oversized prompt unexpectedly succeeded");
        };
        let error = error.to_string();

        assert!(error.contains("KV capacity is 2"));
        assert_eq!(stage.kv_cache_tokens(), 0);
    }

    #[test]
    fn automatic_eviction_replaces_the_full_cache_at_the_equality_boundary() {
        let mut stage = test_stage();
        stage.set_max_kv_tokens(4);
        stage.forward(StageInput::Tokens(vec![1, 2]), 2, 0).unwrap();

        stage
            .forward(StageInput::Tokens(vec![3, 4, 5, 6]), 4, 2)
            .unwrap();

        assert_eq!(stage.kv_cache_tokens(), 4);
        let Some(LayerCache::Attention(KvCache::F16 { k, .. })) = &stage.cache[0] else {
            panic!("expected replacement attention cache")
        };
        assert_eq!(k.dim(2).unwrap(), 4);
    }

    #[test]
    fn failed_full_cache_replacement_restores_the_previous_cache_atomically() {
        let mut stage = test_stage();
        stage.set_max_kv_tokens(4);
        stage.forward(StageInput::Tokens(vec![1, 2]), 2, 0).unwrap();

        let Err(error) = stage.forward(StageInput::Tokens(vec![255, 4, 5, 6]), 4, 2) else {
            panic!("injected replacement failure unexpectedly succeeded")
        };
        assert!(error.to_string().contains("injected forward failure"));
        assert_eq!(stage.kv_cache_tokens(), 2);
        assert_eq!(stage.next_position(), 2);
        let Some(LayerCache::Attention(KvCache::F16 { k, .. })) = &stage.cache[0] else {
            panic!("expected original attention cache")
        };
        assert_eq!(k.dim(2).unwrap(), 2);

        // The restored cache remains usable at its original position.
        stage.forward(StageInput::Tokens(vec![7]), 1, 2).unwrap();
        assert_eq!(stage.kv_cache_tokens(), 3);
    }

    #[test]
    fn failed_incremental_forward_releases_partial_cache_from_aggregate_budget() {
        let mut stage = test_stage();
        stage.set_max_total_kv_tokens(16).unwrap();
        stage.forward(StageInput::Tokens(vec![1, 2]), 2, 0).unwrap();

        let Err(error) = stage.forward(StageInput::Tokens(vec![255]), 1, 2) else {
            panic!("injected incremental failure unexpectedly succeeded")
        };

        assert!(error.to_string().contains("injected forward failure"));
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        assert_eq!(stage.kv_cache_tokens(), 0);
        assert_eq!(stage.next_position(), 0);
        assert!(stage.cache[0].is_none());
    }

    #[test]
    fn automatic_eviction_below_equality_keeps_the_remaining_window() {
        let mut stage = test_stage();
        stage.set_max_kv_tokens(4);
        stage
            .forward(StageInput::Tokens(vec![1, 2, 3]), 3, 0)
            .unwrap();

        stage.forward(StageInput::Tokens(vec![4, 5]), 2, 3).unwrap();

        assert_eq!(stage.kv_cache_tokens(), 4);
        let Some(LayerCache::Attention(KvCache::F16 { k, .. })) = &stage.cache[0] else {
            panic!("expected attention cache")
        };
        assert_eq!(k.dim(2).unwrap(), 4);
    }

    #[test]
    fn repeated_overflow_evicts_only_the_needed_prefix_and_keeps_absolute_positions() {
        let mut capped = test_stage();
        capped.set_max_kv_tokens(4);
        let mut reference = test_stage();
        let chunks = [
            (0, vec![1, 2, 3]),
            (3, vec![4, 5]),
            (5, vec![6]),
            (6, vec![7, 8]),
        ];
        let mut all_tokens = Vec::new();

        for (start_pos, tokens) in chunks {
            let capped_output = capped
                .forward(StageInput::Tokens(tokens.clone()), tokens.len(), start_pos)
                .unwrap();
            let reference_output = reference
                .forward(StageInput::Tokens(tokens.clone()), tokens.len(), start_pos)
                .unwrap();
            all_tokens.extend(tokens.into_iter().map(|token| token as f32));

            assert_eq!(logits(capped_output), logits(reference_output));
            assert_eq!(capped.next_position(), reference.next_position());
            assert_eq!(capped.next_position(), all_tokens.len());
            assert_eq!(capped.kv_cache_tokens(), all_tokens.len().min(4));
            assert_eq!(
                cached_values(&capped),
                all_tokens[all_tokens.len().saturating_sub(4)..]
            );
        }

        assert_eq!(capped.next_position(), 8);
        assert_eq!(capped.kv_cache_tokens(), 4);
        assert_eq!(cached_values(&capped), vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn non_contiguous_absolute_position_is_rejected_without_mutating_cache() {
        let mut stage = test_stage();
        stage.forward(StageInput::Tokens(vec![1, 2]), 2, 0).unwrap();

        let Err(error) = stage.forward(StageInput::Tokens(vec![3]), 1, 1) else {
            panic!("non-contiguous position unexpectedly succeeded")
        };
        let error = error.to_string();

        assert!(error.contains("expected 2, received 1"));
        assert_eq!(stage.next_position(), 2);
        assert_eq!(stage.kv_cache_tokens(), 2);
        assert_eq!(cached_values(&stage), vec![1.0, 2.0]);
    }

    #[test]
    fn session_repeated_overflow_keeps_absolute_position_separate_from_cache_length() {
        let mut stage = test_stage();
        stage.set_max_kv_tokens(3);
        let mut session = stage.new_session_state();

        stage
            .forward_with_session(&mut session, StageInput::Tokens(vec![1, 2, 3]), 3, 0)
            .unwrap();
        for position in 3..8 {
            stage
                .forward_with_session(
                    &mut session,
                    StageInput::Tokens(vec![position as u32 + 1]),
                    1,
                    position,
                )
                .unwrap();
            assert_eq!(stage.session_next_position(&session), position + 1);
            assert_eq!(stage.session_kv_cache_tokens(&session), 3);
        }

        assert_eq!(stage.session_next_position(&session), 8);
        assert_eq!(stage.session_kv_cache_tokens(&session), 3);
        assert!(!stage.can_decode_batch_with_session(&session));
    }

    #[test]
    fn aggregate_budget_counts_global_cache_and_every_live_session() {
        let mut stage = test_stage();
        stage.set_max_total_kv_tokens(20).unwrap();
        stage.forward(StageInput::Tokens(vec![1, 2]), 2, 0).unwrap();

        let mut sessions = (0..4)
            .map(|_| stage.new_session_state())
            .collect::<Vec<_>>();
        for (lane, session) in sessions.iter_mut().enumerate() {
            let first = 10 + lane as u32 * 2;
            stage
                .forward_with_session(session, StageInput::Tokens(vec![first, first + 1]), 2, 0)
                .unwrap();
        }

        assert_eq!(stage.kv_cache_tokens(), 2);
        assert_eq!(stage.total_kv_cache_tokens(), 10);
        assert!(sessions.iter().all(|session| {
            stage.session_kv_cache_tokens(session) == 2 && stage.session_next_position(session) == 2
        }));

        let mut blocked = stage.new_session_state();
        let Err(error) =
            stage.forward_with_session(&mut blocked, StageInput::Tokens(vec![99]), 1, 0)
        else {
            panic!("aggregate-full stage admitted another session")
        };
        assert!(error.to_string().contains("aggregate KV capacity 10"));
        assert_eq!(stage.total_kv_cache_tokens(), 10);
        assert_eq!(stage.session_kv_cache_tokens(&blocked), 0);
        assert_eq!(stage.session_next_position(&blocked), 0);

        drop(sessions.pop());
        assert_eq!(stage.total_kv_cache_tokens(), 8);
        stage
            .forward_with_session(&mut blocked, StageInput::Tokens(vec![99, 100]), 2, 0)
            .unwrap();
        assert_eq!(stage.total_kv_cache_tokens(), 10);
    }

    #[test]
    fn aggregate_budget_charges_aligned_cache_capacity_before_allocating() {
        let mut stage = preallocated_test_stage();
        stage.set_max_total_kv_tokens(128).unwrap();
        let mut first = stage.new_session_state();
        let mut blocked = stage.new_session_state();

        stage
            .forward_with_session(&mut first, StageInput::Tokens(vec![1]), 1, 0)
            .unwrap();
        assert_eq!(stage.session_kv_cache_tokens(&first), 1);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 64);

        let Err(error) =
            stage.forward_with_session(&mut blocked, StageInput::Tokens(vec![2]), 1, 0)
        else {
            panic!("aligned cache allocation unexpectedly exceeded the resident limit")
        };
        assert!(
            error
                .to_string()
                .contains("resident KV allocation limit 64")
        );
        assert_eq!(stage.session_kv_cache_tokens(&blocked), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 64);

        stage.clear_session_state(&mut first);
        stage
            .forward_with_session(&mut blocked, StageInput::Tokens(vec![2]), 1, 0)
            .unwrap();
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 64);
    }

    #[test]
    fn copy_reserve_keeps_rolling_decode_inside_allocation_headroom() {
        let mut stage = preallocated_test_stage();
        stage.set_max_total_kv_tokens(128).unwrap();
        let mut session = stage.new_session_state();
        let prompt = (0..64).collect::<Vec<u32>>();

        stage
            .forward_with_session(&mut session, StageInput::Tokens(prompt), 64, 0)
            .unwrap();
        stage
            .forward_with_session(&mut session, StageInput::Tokens(vec![64]), 1, 64)
            .unwrap();

        assert_eq!(stage.session_next_position(&session), 65);
        assert_eq!(stage.session_kv_cache_tokens(&session), 64);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 64);
        assert_eq!(stage.max_resident_kv_allocation_tokens(), Some(64));
        assert_eq!(stage.max_total_kv_allocation_headroom_tokens(), Some(128));
    }

    #[test]
    fn aggregate_budget_rejects_large_prompt_without_mutating_active_session() {
        let mut stage = test_stage();
        stage.set_max_total_kv_tokens(16).unwrap();
        let mut sessions = (0..4)
            .map(|_| stage.new_session_state())
            .collect::<Vec<_>>();
        for (lane, session) in sessions.iter_mut().enumerate() {
            let first = 1 + lane as u32 * 2;
            stage
                .forward_with_session(session, StageInput::Tokens(vec![first, first + 1]), 2, 0)
                .unwrap();
        }
        let before = session_cached_values(&sessions[0]);

        let Err(error) = stage.forward_with_session(
            &mut sessions[0],
            StageInput::Tokens(vec![20, 21, 22]),
            3,
            2,
        ) else {
            panic!("oversized aggregate prompt unexpectedly succeeded")
        };

        assert!(error.to_string().contains("incoming sequence needs 3"));
        assert_eq!(stage.total_kv_cache_tokens(), 8);
        assert_eq!(stage.session_kv_cache_tokens(&sessions[0]), 2);
        assert_eq!(stage.session_next_position(&sessions[0]), 2);
        assert_eq!(session_cached_values(&sessions[0]), before);
    }

    #[test]
    fn aggregate_full_sessions_keep_decoding_by_rolling_their_own_prefixes() {
        let mut stage = test_stage();
        stage.set_max_total_kv_tokens(16).unwrap();
        let mut sessions = (0..4)
            .map(|_| stage.new_session_state())
            .collect::<Vec<_>>();
        for (lane, session) in sessions.iter_mut().enumerate() {
            let first = 1 + lane as u32 * 20;
            stage
                .forward_with_session(session, StageInput::Tokens(vec![first, first + 1]), 2, 0)
                .unwrap();
        }

        for position in 2..7 {
            for (lane, session) in sessions.iter_mut().enumerate() {
                let token = 100 + lane as u32 * 20 + position as u32;
                stage
                    .forward_with_session(session, StageInput::Tokens(vec![token]), 1, position)
                    .unwrap();
                assert_eq!(stage.total_kv_cache_tokens(), 8);
                assert_eq!(stage.session_kv_cache_tokens(session), 2);
                assert_eq!(stage.session_next_position(session), position + 1);
            }
        }

        for (lane, session) in sessions.iter().enumerate() {
            let expected_last = 100 + lane as u32 * 20 + 6;
            assert_eq!(
                session_cached_values(session),
                vec![(expected_last - 1) as f32, expected_last as f32]
            );
        }
    }

    #[test]
    fn incompatible_decode_cache_rejects_before_resetting_any_lane() -> Result<()> {
        let mut stage = preallocated_test_stage();
        let mut first = stage.new_session_state();
        stage.forward_with_session(&mut first, StageInput::Tokens(vec![2, 3]), 2, 0)?;
        let mut second = stage.new_session_state();
        stage.forward_with_session(&mut second, StageInput::Tokens(vec![4, 5]), 2, 0)?;
        let Some(LayerCache::Attention(KvCache::PreallocatedF16(cache))) = &second.cache[0] else {
            panic!("expected preallocated storage");
        };
        let (k, v) = cache.active_f16()?;
        let before = (
            first.revision,
            second.revision,
            stage.total_kv_cache_allocation_tokens(),
        );
        second.cache[0] = Some(LayerCache::Attention(KvCache::F16 { k, v }));
        assert!(!stage.can_decode_batch_with_session(&second));
        let error = stage
            .forward_decode_batch_with_sessions(&mut [&mut first, &mut second], &[6, 7])
            .unwrap_err();
        assert!(error.to_string().contains("scalar cache preparation"));
        assert_eq!(first.next_position, 2);
        assert_eq!(second.next_position, 2);
        assert_eq!(
            (
                first.revision,
                second.revision,
                stage.total_kv_cache_allocation_tokens()
            ),
            before
        );
        stage.forward_with_session(&mut second, StageInput::Tokens(vec![7]), 1, 2)?;
        assert!(stage.can_decode_batch_with_session(&second));
        drop((first, second));
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn aggregate_full_decode_batch_rejects_before_any_lane_allocates() {
        let mut stage = test_stage();
        stage.set_max_total_kv_tokens(8).unwrap();
        let mut first = stage.new_session_state();
        let mut second = stage.new_session_state();
        stage
            .forward_with_session(&mut first, StageInput::Tokens(vec![1, 2]), 2, 0)
            .unwrap();
        stage
            .forward_with_session(&mut second, StageInput::Tokens(vec![3, 4]), 2, 0)
            .unwrap();
        let first_before = session_cached_values(&first);
        let second_before = session_cached_values(&second);

        let mut sessions = vec![&mut first, &mut second];
        let Err(error) = stage.forward_decode_batch_with_sessions(&mut sessions, &[5, 6]) else {
            panic!("aggregate-full decode batch unexpectedly succeeded")
        };

        assert!(
            error
                .to_string()
                .contains("requires per-session KV eviction")
        );
        assert_eq!(stage.total_kv_cache_tokens(), 4);
        assert_eq!(stage.session_next_position(&first), 2);
        assert_eq!(stage.session_next_position(&second), 2);
        assert_eq!(session_cached_values(&first), first_before);
        assert_eq!(session_cached_values(&second), second_before);
    }

    #[test]
    fn independent_decode_batch_charges_only_each_lanes_own_cache() {
        for preallocated in [false, true] {
            let mut stage = test_stage();
            stage.arch = Box::new(IndependentBatchCacheArchitecture { preallocated });
            stage.preallocated_kv_cache = preallocated;
            stage
                .set_max_total_kv_tokens(if preallocated { 256 } else { 16 })
                .unwrap();
            let mut first = stage.new_session_state();
            let mut second = stage.new_session_state();
            stage
                .forward_with_session(&mut first, StageInput::Tokens(vec![1]), 1, 0)
                .unwrap();
            stage
                .forward_with_session(&mut second, StageInput::Tokens(vec![2, 3, 4]), 3, 0)
                .unwrap();
            stage
                .forward_decode_batch_with_sessions(&mut [&mut first, &mut second], &[5, 6])
                .unwrap();
            assert_eq!(stage.total_kv_cache_tokens(), 6);
            assert_eq!(stage.session_next_position(&first), 2);
            assert_eq!(stage.session_next_position(&second), 4);
            let total_allocated = stage.kv_token_budget.allocated();
            assert_eq!(
                total_allocated,
                first.kv_cache_allocation_tokens + second.kv_cache_allocation_tokens
            );
            assert!(total_allocated <= stage.max_total_kv_tokens.unwrap());
            if !preallocated {
                stage
                    .forward_decode_batch_with_sessions(&mut [&mut first, &mut second], &[7, 8])
                    .unwrap();
                let before = session_cached_values(&first);
                assert!(
                    stage
                        .forward_decode_batch_with_sessions(
                            &mut [&mut first, &mut second],
                            &[9, 10]
                        )
                        .is_err()
                );
                assert_eq!(session_cached_values(&first), before);
            }
            drop(first);
            drop(second);
            assert_eq!(stage.kv_token_budget.allocated(), 0);
            assert_eq!(stage.total_kv_cache_tokens(), 0);
        }
    }

    #[test]
    fn oversized_decode_batch_result_clears_mutated_states_and_budget() {
        let mut stage = test_stage();
        stage.arch = Box::new(OversizedBatchCacheArchitecture);
        stage.set_max_total_kv_tokens(20).unwrap();
        let mut first = stage.new_session_state();
        let mut second = stage.new_session_state();
        stage
            .forward_with_session(&mut first, StageInput::Tokens(vec![1]), 1, 0)
            .unwrap();
        stage
            .forward_with_session(&mut second, StageInput::Tokens(vec![2]), 1, 0)
            .unwrap();
        assert_eq!(stage.total_kv_cache_tokens(), 2);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 2);

        let mut sessions = vec![&mut first, &mut second];
        let Err(error) = stage.forward_decode_batch_with_sessions(&mut sessions, &[3, 4]) else {
            panic!("oversized decode batch unexpectedly succeeded")
        };

        assert!(
            error
                .to_string()
                .contains("exceeded the resident KV allocation limit after execution")
        );
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        assert_eq!(stage.session_kv_cache_tokens(&first), 0);
        assert_eq!(stage.session_kv_cache_tokens(&second), 0);
        assert_eq!(stage.session_next_position(&first), 0);
        assert_eq!(stage.session_next_position(&second), 0);
        assert!(first.cache.iter().all(Option::is_none));
        assert!(second.cache.iter().all(Option::is_none));
    }

    #[test]
    fn aggregate_cap_cannot_drop_below_live_retained_total() {
        let mut stage = test_stage();
        stage.forward(StageInput::Tokens(vec![1, 2]), 2, 0).unwrap();
        let mut session = stage.new_session_state();
        stage
            .forward_with_session(&mut session, StageInput::Tokens(vec![3, 4]), 2, 0)
            .unwrap();

        assert!(stage.set_max_total_kv_tokens(0).is_err());
        assert!(stage.set_max_total_kv_tokens(3).is_err());
        assert!(stage.set_max_total_kv_tokens(7).is_err());
        stage.set_max_total_kv_tokens(8).unwrap();
        assert_eq!(stage.total_kv_cache_tokens(), 4);
        assert_eq!(stage.max_resident_kv_allocation_tokens(), Some(4));
    }

    #[test]
    fn public_invalid_eviction_resets_cache_before_reporting_failure() {
        let mut stage = test_stage();
        stage.forward(StageInput::Tokens(vec![1, 2]), 2, 0).unwrap();

        stage.evict_oldest(3);

        assert_eq!(stage.kv_cache_tokens(), 0);
        assert!(stage.cache[0].is_none());
    }

    #[test]
    fn public_full_eviction_clears_cache_at_equality_boundary() {
        let mut stage = test_stage();
        stage.forward(StageInput::Tokens(vec![1, 2]), 2, 0).unwrap();

        stage.evict_oldest(2);

        assert_eq!(stage.kv_cache_tokens(), 0);
        assert_eq!(stage.next_position(), 2);
        assert!(stage.cache[0].is_none());

        stage.forward(StageInput::Tokens(vec![3]), 1, 2).unwrap();
        assert_eq!(stage.kv_cache_tokens(), 1);
        assert_eq!(stage.next_position(), 3);
        assert_eq!(cached_values(&stage), vec![3.0]);
    }

    #[test]
    fn v4_partial_eviction_refuses_to_discard_compressed_history() {
        let mut stage = test_stage();
        stage.forward(StageInput::Tokens(vec![1, 2]), 2, 0).unwrap();
        stage.cache[0] = Some(LayerCache::DeepSeek4(Box::default()));
        let allocation = stage.kv_cache_allocation_tokens;
        assert!(stage.try_evict_oldest(1).is_err());
        assert_eq!(stage.kv_cache_tokens(), 2);
        assert_eq!(stage.kv_cache_allocation_tokens, allocation);
        assert!(matches!(stage.cache[0], Some(LayerCache::DeepSeek4(_))));
        stage.try_evict_oldest(2).unwrap();
        assert!(stage.cache[0].is_none());
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        assert_eq!(stage.kv_cache_allocation_tokens, 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_load_releases_retired_metal_weight_buffers_first() -> Result<()> {
        use candle_core::quantized::gguf_file::{self, Value};
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("missing-weights.gguf");
        // Valid metadata, deliberately missing weights. The load boundary
        // must drain the old allocation even when the new load then fails.
        let metadata = [
            ("general.architecture", Value::String("llama".into())),
            ("llama.block_count", Value::U32(1)),
            ("llama.embedding_length", Value::U32(64)),
            ("llama.attention.head_count", Value::U32(1)),
        ];
        let metadata = metadata.iter().map(|(k, v)| (*k, v)).collect::<Vec<_>>();
        gguf_file::write(&mut std::fs::File::create(&path)?, &metadata, &[])?;
        let buffer = device
            .as_metal_device()?
            .new_buffer_with_data(&[0u8; 4096])?;
        let retired = Arc::downgrade(&buffer);
        drop(buffer);
        assert!(
            retired.upgrade().is_some(),
            "Candle should still own this retired buffer"
        );
        assert!(
            PipelineStage::from_gguf(&path, crate::LayerRange::solo(1), &device, None).is_err()
        );
        assert!(
            retired.upgrade().is_none(),
            "model loading retained the old weight buffer"
        );
        Ok(())
    }

    #[test]
    fn positioned_cache_checkpoint_clear_and_batch_guards_keep_metadata() -> Result<()> {
        use crate::continuous::GenerationSession;
        let mut stage = test_stage();
        let mut session = GenerationSession::new_with_positioned_embeddings(
            &stage,
            vec![0; 4],
            Tensor::zeros((4, 1), DType::F32, &Device::Cpu)?,
            MropePositionIds::new(vec![0, 1, 1, 2], vec![0, 1, 2, 2], vec![0, 2, 1, 2])?,
            SamplingParams {
                temperature: 0.0,
                max_tokens: 1,
                ..Default::default()
            },
            vec![],
        )?;
        session.step(&mut stage)?;
        let mut cache = session.into_cache();
        assert_eq!(cache.mrope_decode_delta, Some(-1));
        let before = stage.total_kv_cache_allocation_tokens();
        assert!(
            stage
                .forward_decode_batch_with_sessions(&mut [&mut cache], &[1])
                .is_err()
        );
        assert_eq!(cache.next_position, 4);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), before);
        let checkpoint = stage.checkpoint_session_state(&cache)?;
        stage.clear_session_state(&mut cache);
        assert_eq!(cache.mrope_decode_delta, None);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        stage.restore_session_checkpoint(&mut cache, checkpoint);
        assert_eq!(cache.mrope_decode_delta, Some(-1));
        assert_eq!(stage.total_kv_cache_allocation_tokens(), before);
        stage.clear_session_state(&mut cache);
        let mut ordinary = GenerationSession::with_cache(
            cache,
            vec![0],
            SamplingParams {
                temperature: 0.0,
                max_tokens: 1,
                ..Default::default()
            },
            vec![],
            0,
        );
        ordinary.step(&mut stage)?;
        let mut ordinary_cache = ordinary.into_cache();
        assert_eq!(ordinary_cache.mrope_decode_delta, None);
        stage.forward_decode_batch_with_sessions(&mut [&mut ordinary_cache], &[1])?;
        stage.clear_session_state(&mut ordinary_cache);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn positioned_session_resume_keeps_decode_delta_and_text_chunks() -> Result<()> {
        use crate::continuous::GenerationSession;

        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut stage = position_recording_stage(Arc::clone(&seen));
        let positions =
            MropePositionIds::new(vec![0, 1, 1, 2], vec![0, 1, 2, 2], vec![0, 2, 1, 2])?;
        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: 2,
            ..SamplingParams::default()
        };
        let mut first = GenerationSession::new_with_positioned_embeddings(
            &stage,
            vec![0; 4],
            Tensor::zeros((4, 1), DType::F32, &Device::Cpu)?,
            positions.clone(),
            params.clone(),
            vec![],
        )?;
        first.step(&mut stage)?;
        first.step(&mut stage)?;
        let cache = first.into_cache();
        assert_eq!(cache.next_position, 5);

        // The last output was not consumed. Include it and two new text
        // tokens after the five positions already stored in the cache.
        let mut resumed =
            GenerationSession::with_cache(cache, vec![0, 0, 0, 0, 1, 1, 0, 0], params, vec![], 5);
        assert!(resumed.step_with_prefill_budget(&mut stage, 1)?.is_none());
        assert!(resumed.step_with_prefill_budget(&mut stage, 1)?.is_none());
        assert!(resumed.step_with_prefill_budget(&mut stage, 1)?.is_some());
        resumed.step(&mut stage)?;
        let recorded = seen.lock().unwrap();
        assert_eq!(recorded[0].as_ref(), Some(&positions));
        assert_eq!(recorded.len(), 6);
        for (recorded, expected) in recorded[1..].iter().zip(3..=7) {
            assert_eq!(
                recorded.as_ref(),
                Some(&MropePositionIds::text(expected, 1)?)
            );
        }
        drop(recorded);
        resumed.discard_cache(&mut stage);
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn audit_direct_generation_rejects_bias_ids_before_model_work() -> Result<()> {
        let mut stage = test_stage();
        stage.forward(StageInput::Tokens(vec![0]), 1, 0)?;
        let old_tokens = stage.total_kv_cache_tokens();
        let old_allocation = stage.total_kv_cache_allocation_tokens();
        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: 1,
            logit_bias: [(stage.vocab_size() as u32, 1.0)].into_iter().collect(),
            ..Default::default()
        };
        assert!(
            stage
                .generate(vec![0, 1], &params, &[], 1, |_| true)
                .is_err()
        );
        assert_eq!(stage.next_position(), 1);
        assert_eq!(stage.total_kv_cache_tokens(), old_tokens);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), old_allocation);
        assert!(
            stage
                .generate_with_embeddings(
                    vec![0, 1],
                    Tensor::zeros((2, 1), DType::F32, &Device::Cpu)?,
                    &params,
                    &[],
                    |_| true,
                )
                .is_err()
        );
        assert_eq!(stage.next_position(), 1);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), old_allocation);
        stage.generate(
            vec![0, 1],
            &SamplingParams {
                temperature: 0.0,
                max_tokens: 1,
                ..Default::default()
            },
            &[],
            1,
            |_| true,
        )?;
        assert_eq!(stage.next_position(), 2);
        Ok(())
    }

    #[test]
    fn audit_direct_positioned_generation_resume_keeps_rotary_offset() -> Result<()> {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut stage = position_recording_stage(Arc::clone(&seen));
        let positions =
            MropePositionIds::new(vec![0, 1, 1, 2], vec![0, 1, 2, 2], vec![0, 2, 1, 2])?;
        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: 2,
            ..Default::default()
        };
        stage.generate_with_positioned_embeddings(
            vec![0; 4],
            Tensor::zeros((4, 1), DType::F32, &Device::Cpu)?,
            positions.clone(),
            &params,
            &[],
            |_| true,
        )?;
        assert_eq!(stage.next_position(), 5);
        stage.generate(vec![0, 0, 0, 0, 1, 1, 0, 0], &params, &[], 5, |_| true)?;
        {
            let recorded = seen.lock().unwrap();
            assert_eq!(recorded[0].as_ref(), Some(&positions));
            assert_eq!(recorded[1].as_ref(), Some(&MropePositionIds::text(3, 1)?));
            assert_eq!(recorded[2].as_ref(), Some(&MropePositionIds::text(4, 3)?));
            assert_eq!(recorded[3].as_ref(), Some(&MropePositionIds::text(7, 1)?));
        }
        stage.clear_cache();
        stage.generate(vec![0, 0], &params, &[], 0, |_| true)?;
        assert!(seen.lock().unwrap()[4..].iter().all(Option::is_none));
        Ok(())
    }

    #[test]
    fn audit_positioned_public_prefill_keeps_offset_for_resumed_generation() -> Result<()> {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut stage = position_recording_stage(Arc::clone(&seen));
        let mut cache = stage.new_session_state();
        let positions =
            MropePositionIds::new(vec![0, 1, 1, 2], vec![0, 1, 2, 2], vec![0, 2, 1, 2])?;
        stage.prefill_with_session(
            &mut cache,
            StageInput::positioned_activations(
                Tensor::zeros((4, 1), DType::F32, &Device::Cpu)?,
                positions,
            ),
            4,
            0,
        )?;
        let mut resumed = GenerationSession::with_cache(
            cache,
            vec![0; 6],
            SamplingParams {
                temperature: 0.0,
                max_tokens: 2,
                ..Default::default()
            },
            vec![],
            4,
        );
        resumed.step(&mut stage)?;
        resumed.step(&mut stage)?;
        let recorded = seen.lock().unwrap();
        assert_eq!(recorded[1].as_ref(), Some(&MropePositionIds::text(3, 2)?));
        assert_eq!(recorded[2].as_ref(), Some(&MropePositionIds::text(5, 1)?));
        Ok(())
    }

    #[test]
    fn positioned_public_prefill_tracks_prior_axes_and_keeps_sessions_separate() -> Result<()> {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut stage = position_recording_stage(Arc::clone(&seen));
        let mut first = stage.new_session_state();
        let mut second = stage.new_session_state();
        for (positions, start) in [
            (
                MropePositionIds::new(vec![0, 1], vec![0, 8], vec![0, 1])?,
                0,
            ),
            (MropePositionIds::text(2, 2)?, 2),
        ] {
            stage.prefill_with_session(
                &mut first,
                StageInput::positioned_activations(
                    Tensor::zeros((2, 1), DType::F32, &Device::Cpu)?,
                    positions,
                ),
                2,
                start,
            )?;
            assert_eq!(stage.mrope_decode_delta, None);
        }
        assert_eq!(first.mrope_decode_delta, Some(5));
        stage.prefill_with_session(
            &mut second,
            StageInput::positioned_activations(
                Tensor::zeros((2, 1), DType::F32, &Device::Cpu)?,
                MropePositionIds::text(10, 2)?,
            ),
            2,
            0,
        )?;
        assert_eq!(second.mrope_decode_delta, Some(10));
        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: 1,
            ..Default::default()
        };
        let mut a = GenerationSession::with_cache(first, vec![0; 5], params.clone(), vec![], 4);
        let mut b = GenerationSession::with_cache(second, vec![0; 3], params, vec![], 2);
        a.step(&mut stage)?;
        b.step(&mut stage)?;
        {
            let recorded = seen.lock().unwrap();
            assert_eq!(recorded[3].as_ref(), Some(&MropePositionIds::text(9, 1)?));
            assert_eq!(recorded[4].as_ref(), Some(&MropePositionIds::text(12, 1)?));
        }
        assert_eq!(stage.mrope_decode_delta, None);
        a.discard_cache(&mut stage);
        b.discard_cache(&mut stage);
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        Ok(())
    }

    #[test]
    fn positioned_public_prefill_offset_survives_eviction_and_resets_after_failure() -> Result<()> {
        let mut stage = test_stage();
        stage.set_max_kv_tokens(4);
        let mut cache = stage.new_session_state();
        stage.prefill_with_session(
            &mut cache,
            StageInput::positioned_activations(
                Tensor::zeros((4, 1), DType::F32, &Device::Cpu)?,
                MropePositionIds::new(vec![0, 1, 1, 2], vec![0, 1, 2, 2], vec![0, 2, 1, 2])?,
            ),
            4,
            0,
        )?;
        stage.forward_with_session(
            &mut cache,
            StageInput::positioned_activations(
                Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?,
                MropePositionIds::text(3, 1)?,
            ),
            1,
            4,
        )?;
        assert_eq!(cache.mrope_decode_delta, Some(-1));
        assert_eq!(cache.next_position, 5);
        assert_eq!(cache.kv_cache_tokens, 4);
        assert!(
            stage
                .forward_with_session(
                    &mut cache,
                    StageInput::positioned_activations(
                        Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?,
                        MropePositionIds::text(4, 2)?
                    ),
                    1,
                    5
                )
                .is_err()
        );
        assert_eq!(cache.next_position, 5);
        assert_eq!(cache.mrope_decode_delta, Some(-1));
        assert!(
            stage
                .forward_decode_batch_with_sessions(&mut [&mut cache], &[1])
                .is_err()
        );
        assert_eq!(cache.next_position, 5);
        assert!(
            stage
                .forward_with_session(&mut cache, StageInput::Tokens(vec![255]), 1, 5)
                .is_err()
        );
        assert_eq!(cache.mrope_decode_delta, None);
        assert_eq!(cache.next_position, 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        stage.set_max_kv_tokens(1);
        stage.prefill_with_session(
            &mut cache,
            StageInput::positioned_activations(
                Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?,
                MropePositionIds::text(5, 1)?,
            ),
            1,
            0,
        )?;
        assert!(
            stage
                .forward_with_session(&mut cache, StageInput::Tokens(vec![255]), 1, 1)
                .is_err()
        );
        assert_eq!(cache.mrope_decode_delta, Some(5));
        assert_eq!(cache.next_position, 1);
        stage.forward_with_session(
            &mut cache,
            StageInput::positioned_activations(
                Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?,
                MropePositionIds::text(6, 1)?,
            ),
            1,
            1,
        )?;
        assert_eq!(cache.mrope_decode_delta, Some(5));
        stage.clear_session_state(&mut cache);
        assert_eq!(cache.mrope_decode_delta, None);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn positioned_generation_retains_qwen_decode_delta() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut stage = position_recording_stage(Arc::clone(&seen));
        let prefill_positions =
            MropePositionIds::new(vec![0, 1, 1, 2], vec![0, 1, 2, 2], vec![0, 2, 1, 2]).unwrap();
        let embeddings = Tensor::zeros((4, 1), DType::F32, &Device::Cpu).unwrap();
        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: 3,
            ..SamplingParams::default()
        };

        let generated = stage
            .generate_with_positioned_embeddings(
                vec![0, 0, 0, 0],
                embeddings,
                prefill_positions.clone(),
                &params,
                &[],
                |_| true,
            )
            .unwrap();

        assert_eq!(generated, vec![1, 1, 1]);
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 3);
        assert_eq!(seen[0].as_ref(), Some(&prefill_positions));
        assert_eq!(seen[1].as_ref().unwrap().temporal(), &[3]);
        assert_eq!(seen[1].as_ref().unwrap().height(), &[3]);
        assert_eq!(seen[1].as_ref().unwrap().width(), &[3]);
        assert_eq!(seen[2].as_ref().unwrap().temporal(), &[4]);
        assert_eq!(seen[2].as_ref().unwrap().height(), &[4]);
        assert_eq!(seen[2].as_ref().unwrap().width(), &[4]);
    }
}

#[cfg(test)]
mod deepseek41_session_tests;

#[cfg(test)]
mod token_admission_tests;
