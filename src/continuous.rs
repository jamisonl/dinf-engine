//! Session-isolated continuous generation scheduling.
//!
//! This module owns request lifecycle and cache isolation. Callers can advance
//! decode lanes or bounded text-prompt chunks, keeping scheduler turns explicit
//! without changing request-owned sampling, cancellation, or streaming state.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use candle_core::DType;
#[cfg(target_os = "macos")]
use candle_core::IndexOp;

use super::architectures::{
    AttentionSpan, ImageEmbeddingSpan, ImageSpanKey, MropePositionIds, StageInput, StageOutput,
};
use super::sampling::{SamplingHistory, SamplingParams, SamplingRng};
use super::{PipelineStage, PipelineStageSessionState, StagePosition, TextRequestContext};

#[cfg(test)]
#[path = "continuous_draft_tests.rs"]
mod draft_sampling_tests;

const SPECULATIVE_SPEED_WINDOW: usize = 8;
const SPECULATIVE_SCALAR_PROBES: usize = 3;
const PRE_M5_SPECULATIVE_MAX_PASSES: usize = 1;

/// Compare useful tokens per wall-clock second, not acceptance percentage.
/// Use the median of three scalar decodes to limit the effect of one slow
/// launch, and require a full verification window plus 5% slowdown before
/// falling back. Waiting in the scheduler never enters either timer. The
/// elapsed-time policy stays opt-in while mixed scalar/draft numerics are
/// under test; the default retains the prior acceptance-based thresholds.
#[derive(Default)]
struct SpeculativeSpeedMonitor {
    scalar: Vec<Duration>,
    recent: VecDeque<(usize, Duration)>,
    total_tokens: usize,
    total_time: Duration,
    /// Committed tokens per pass, in tenths, below which the acceptance
    /// policy falls back: over the recent window, and over the whole request
    /// after 32 passes. Zero means the DFlash defaults (1.5 and 2.5).
    floor_tenths: (usize, usize),
}

impl SpeculativeSpeedMonitor {
    fn needs_probe(&self) -> bool {
        self.scalar.len() < SPECULATIVE_SCALAR_PROBES
    }

    fn record_scalar(&mut self, elapsed: Duration) {
        if self.needs_probe() && !elapsed.is_zero() {
            self.scalar.push(elapsed);
            self.scalar.sort_unstable();
        }
    }

    fn record_pass(&mut self, tokens: usize, elapsed: Duration) {
        self.total_tokens += tokens;
        self.total_time += elapsed;
        self.recent.push_back((tokens, elapsed));
        if self.recent.len() > SPECULATIVE_SPEED_WINDOW {
            self.recent.pop_front();
        }
    }

    fn scalar_token_time(&self) -> Option<Duration> {
        (!self.needs_probe()).then(|| self.scalar[SPECULATIVE_SCALAR_PROBES / 2])
    }

    fn should_fallback(&self) -> bool {
        let Some(scalar) = self.scalar_token_time() else {
            return false;
        };
        if self.recent.len() < SPECULATIVE_SPEED_WINDOW {
            return false;
        }
        let tokens: usize = self.recent.iter().map(|(tokens, _)| tokens).sum();
        let elapsed: Duration = self.recent.iter().map(|(_, elapsed)| *elapsed).sum();
        // A brief low-acceptance interval must not disable a draft that is
        // still faster overall. Require both recent and cumulative slowdown.
        elapsed.as_secs_f64() > scalar.as_secs_f64() * tokens as f64 * 1.05
            && self.total_time.as_secs_f64()
                > scalar.as_secs_f64() * self.total_tokens as f64 * 1.05
    }

    fn acceptance_should_fallback(&self, passes: usize) -> bool {
        let (recent, overall) = match self.floor_tenths {
            (0, 0) => (15, 25),
            floor => floor,
        };
        if passes < 32 {
            self.recent.len() >= SPECULATIVE_SPEED_WINDOW
                && self.recent.iter().map(|(tokens, _)| tokens).sum::<usize>() * 10
                    < SPECULATIVE_SPEED_WINDOW * recent
        } else {
            self.total_tokens.saturating_mul(10) < passes.saturating_mul(overall)
        }
    }

    fn fallback_decision(
        &self,
        pre_m5: bool,
        passes: usize,
        disabled: bool,
        elapsed_time_policy: bool,
    ) -> (bool, bool) {
        if disabled {
            return (false, false);
        }
        (
            pre_m5 && passes >= PRE_M5_SPECULATIVE_MAX_PASSES,
            if elapsed_time_policy {
                self.should_fallback()
            } else {
                self.acceptance_should_fallback(passes)
            },
        )
    }
}

/// Result of advancing one generation lane by one token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationStep {
    /// Token selected by the session for this step.
    pub token: u32,
    /// Whether this token completed the session because it matched a stop ID
    /// or the configured token limit.
    pub finished: bool,
}

/// Timing and acceptance counters collected by speculative decoding.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpeculativeStats {
    /// Number of target verification passes performed.
    pub passes: usize,
    /// Number of tokens proposed by the draft model.
    pub drafted_tokens: usize,
    /// Number of draft tokens accepted by the target model.
    pub accepted_tokens: usize,
    /// Number of tokens committed to the session, including fallback tokens.
    pub committed_tokens: usize,
    /// Time spent producing draft tokens.
    pub draft_time: Duration,
    /// Target verification and sampling time, including the logits readback
    /// that waits for queued target GPU work to finish.
    pub target_time: Duration,
    /// Host time spent committing verified tokens to the cache. Asynchronous
    /// draft-cache updates finish during the next draft readback and count
    /// toward that pass's draft time; no extra GPU fence is inserted here.
    pub commit_time: Duration,
    /// Wall-clock time spent in speculative passes.
    pub wall_time: Duration,
    /// Whether this session disabled speculative decoding after a fallback.
    pub fell_back: bool,
}

/// Failure while advancing a decode batch.
///
/// `RetryScalar` means the caller may retry the affected sessions one at a
/// time. `Terminal` means the batch result is not safe to retry and the
/// sessions should be failed.
#[derive(Debug)]
pub enum DecodeBatchStepError {
    /// The batch failed before execution and left caches intact; retry scalar steps.
    RetryScalar(anyhow::Error),
    /// The batch produced an unrecoverable error.
    Terminal(anyhow::Error),
}

impl std::fmt::Display for DecodeBatchStepError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RetryScalar(error) | Self::Terminal(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for DecodeBatchStepError {}

/// Autoregressive state for one independently cached request.
pub struct GenerationSession {
    cache: PipelineStageSessionState,
    prompt_tokens: Vec<u32>,
    prompt_embeddings: Option<candle_core::Tensor>,
    prompt_image_spans: Vec<ImageEmbeddingSpan>,
    /// Content-addressed image span keys for 1D-causal multimodal prompt caching
    /// (populated only when every span carries a SHA-256 digest and M-RoPE is absent).
    prompt_image_keys: Option<Vec<ImageSpanKey>>,
    has_prompt_images: bool,
    prompt_bidirectional_spans: Vec<AttentionSpan>,
    prompt_positions: Option<MropePositionIds>,
    /// Signed Qwen position delta retained for every cached decode token.
    mrope_decode_delta: Option<i64>,
    sampling_history: SamplingHistory,
    generated: Vec<u32>,
    params: SamplingParams,
    stop_ids: Vec<u32>,
    effective_skip: usize,
    next_input_token: Option<u32>,
    pending_tokens: VecDeque<u32>,
    speculative_speed: SpeculativeSpeedMonitor,
    speculative_disabled: bool,
    speculative_stats: SpeculativeStats,
    finished: bool,
    rng: SamplingRng,
    prompt_cache_scope: Option<String>,
    prompt_cache_tickets: Vec<std::sync::Arc<()>>,
    shared_prompt_prefix: Option<usize>,
    reused_prompt_tokens: usize,
    // Set only after a successful fenced text-input step. Any next step
    // invalidates it before work, including validation and device failures.
    completed_input_cursor: Option<usize>,
    input_prefix_trusted: bool,
    // Vocabulary size for which `prompt_tokens` has already been validated,
    // avoiding an O(N_prompt) rescan on every decode step.
    validated_prompt_vocab: Option<usize>,
}

impl GenerationSession {
    /// Create a session with an empty stage cache and entropy-seeded sampling.
    ///
    /// The prompt tokens are retained as the initial context and `stop_ids`
    /// are checked against subsequently sampled tokens.
    pub fn new(
        stage: &PipelineStage,
        prompt_tokens: Vec<u32>,
        params: SamplingParams,
        stop_ids: Vec<u32>,
    ) -> Self {
        let rng = params
            .seed
            .map_or_else(SamplingRng::from_entropy, SamplingRng::seeded);
        Self::with_cache_and_rng(
            stage.new_session_state(),
            prompt_tokens,
            params,
            stop_ids,
            0,
            rng,
        )
    }

    /// Start a fresh lane from decoder-ready prompt embeddings.
    pub fn new_with_embeddings(
        stage: &PipelineStage,
        prompt_tokens: Vec<u32>,
        prompt_embeddings: candle_core::Tensor,
        params: SamplingParams,
        stop_ids: Vec<u32>,
    ) -> Result<Self> {
        Self::new_with_embeddings_and_spans(
            stage,
            prompt_tokens,
            prompt_embeddings,
            Vec::new(),
            params,
            stop_ids,
        )
    }

    /// Start a fresh lane from decoder-ready embeddings and attention spans.
    ///
    /// Returns an error when the first tensor dimension does not equal the
    /// number of prompt tokens.
    pub fn new_with_embeddings_and_spans(
        stage: &PipelineStage,
        prompt_tokens: Vec<u32>,
        prompt_embeddings: candle_core::Tensor,
        bidirectional_spans: Vec<AttentionSpan>,
        params: SamplingParams,
        stop_ids: Vec<u32>,
    ) -> Result<Self> {
        anyhow::ensure!(
            prompt_embeddings.dim(0)? == prompt_tokens.len(),
            "prompt token and embedding counts differ"
        );
        let mut session = Self::new(stage, prompt_tokens, params, stop_ids);
        session.prompt_embeddings = Some(prompt_embeddings);
        session.prompt_bidirectional_spans = bidirectional_spans;
        Ok(session)
    }

    /// Start a fresh lane from decoder-ready embeddings and a Qwen T/H/W
    /// rotary schedule. The derived signed delta is retained for decode.
    pub fn new_with_positioned_embeddings(
        stage: &PipelineStage,
        prompt_tokens: Vec<u32>,
        prompt_embeddings: candle_core::Tensor,
        positions: MropePositionIds,
        params: SamplingParams,
        stop_ids: Vec<u32>,
    ) -> Result<Self> {
        Self::new_with_positioned_embeddings_and_spans(
            stage,
            prompt_tokens,
            prompt_embeddings,
            positions,
            Vec::new(),
            params,
            stop_ids,
        )
    }

    /// Start a fresh lane from embeddings, attention spans, and an M-RoPE
    /// schedule. Returns an error when token, embedding, or position counts
    /// disagree, or when the schedule cannot provide a decode delta.
    pub fn new_with_positioned_embeddings_and_spans(
        stage: &PipelineStage,
        prompt_tokens: Vec<u32>,
        prompt_embeddings: candle_core::Tensor,
        positions: MropePositionIds,
        bidirectional_spans: Vec<AttentionSpan>,
        params: SamplingParams,
        stop_ids: Vec<u32>,
    ) -> Result<Self> {
        anyhow::ensure!(
            prompt_embeddings.dim(0)? == prompt_tokens.len(),
            "prompt token and embedding counts differ"
        );
        anyhow::ensure!(
            positions.len() == prompt_tokens.len(),
            "M-RoPE schedule has {} positions for {} prompt tokens",
            positions.len(),
            prompt_tokens.len()
        );
        let decode_delta = positions.decode_delta(prompt_tokens.len())?;
        let mut session = Self::new(stage, prompt_tokens, params, stop_ids);
        session.prompt_embeddings = Some(prompt_embeddings);
        session.prompt_bidirectional_spans = bidirectional_spans;
        session.prompt_positions = Some(positions);
        session.mrope_decode_delta = Some(decode_delta);
        Ok(session)
    }

    /// Start a fresh lane with sparse encoded image spans embedded lazily
    /// during chunked prefill.
    pub fn new_with_image_spans(
        stage: &PipelineStage,
        prompt_tokens: Vec<u32>,
        image_spans: Vec<ImageEmbeddingSpan>,
        params: SamplingParams,
        stop_ids: Vec<u32>,
    ) -> Result<Self> {
        validate_image_spans(stage, prompt_tokens.len(), &image_spans)?;
        let mut keys = Vec::with_capacity(image_spans.len());
        let mut all_digests = !image_spans.is_empty();
        for span in &image_spans {
            match span.span_key()? {
                Some(key) => keys.push(key),
                None => {
                    all_digests = false;
                    break;
                }
            }
        }
        let mut session = Self::new(stage, prompt_tokens, params, stop_ids);
        session.prompt_image_spans = image_spans;
        session.prompt_image_keys = all_digests.then_some(keys);
        session.has_prompt_images = true;
        Ok(session)
    }

    /// Start a fresh lane with sparse encoded image spans and a Qwen T/H/W
    /// rotary schedule, prefilled in chunks.
    pub fn new_with_positioned_image_spans(
        stage: &PipelineStage,
        prompt_tokens: Vec<u32>,
        image_spans: Vec<ImageEmbeddingSpan>,
        positions: MropePositionIds,
        params: SamplingParams,
        stop_ids: Vec<u32>,
    ) -> Result<Self> {
        validate_image_spans(stage, prompt_tokens.len(), &image_spans)?;
        anyhow::ensure!(
            positions.len() == prompt_tokens.len(),
            "M-RoPE schedule has {} positions for {} prompt tokens",
            positions.len(),
            prompt_tokens.len()
        );
        let decode_delta = positions.decode_delta(prompt_tokens.len())?;
        let mut session = Self::new(stage, prompt_tokens, params, stop_ids);
        session.prompt_image_spans = image_spans;
        session.has_prompt_images = true;
        session.prompt_positions = Some(positions);
        session.mrope_decode_delta = Some(decode_delta);
        Ok(session)
    }

    /// Create a generation lane with deterministic sampling.
    pub fn new_seeded(
        stage: &PipelineStage,
        prompt_tokens: Vec<u32>,
        mut params: SamplingParams,
        stop_ids: Vec<u32>,
        seed: u64,
    ) -> Self {
        // The seed must be visible to the exact-path guards, which keep a
        // seeded request off approximate batched speculative verification.
        params.seed = Some(seed);
        Self::with_cache_and_rng(
            stage.new_session_state(),
            prompt_tokens,
            params,
            stop_ids,
            0,
            SamplingRng::seeded(seed),
        )
    }

    /// Resume generation with cache retained from an earlier turn.
    ///
    /// `skip_prefix` is the absolute count of prompt tokens already processed,
    /// not the retained cache length after rolling eviction.
    /// Positioned image caches retain their rotary offset for appended text.
    /// A text stop that discarded speculative lookahead invalidates the cache;
    /// the next turn then processes its whole prompt regardless of this hint.
    pub fn with_cache(
        cache: PipelineStageSessionState,
        prompt_tokens: Vec<u32>,
        params: SamplingParams,
        stop_ids: Vec<u32>,
        skip_prefix: usize,
    ) -> Self {
        let rng = params
            .seed
            .map_or_else(SamplingRng::from_entropy, SamplingRng::seeded);
        Self::with_cache_and_rng(cache, prompt_tokens, params, stop_ids, skip_prefix, rng)
    }

    fn with_cache_and_rng(
        mut cache: PipelineStageSessionState,
        prompt_tokens: Vec<u32>,
        params: SamplingParams,
        stop_ids: Vec<u32>,
        skip_prefix: usize,
        rng: SamplingRng,
    ) -> Self {
        let effective_skip = if !cache.resume_invalidated && skip_prefix < prompt_tokens.len() {
            skip_prefix
        } else {
            cache.clear();
            0
        };
        let mrope_decode_delta = cache.mrope_decode_delta;
        Self {
            cache,
            sampling_history: SamplingHistory::new(&prompt_tokens, &params),
            prompt_tokens,
            prompt_embeddings: None,
            prompt_image_spans: Vec::new(),
            prompt_image_keys: None,
            has_prompt_images: false,
            prompt_bidirectional_spans: Vec::new(),
            prompt_positions: None,
            mrope_decode_delta,
            generated: Vec::new(),
            params,
            stop_ids,
            effective_skip,
            next_input_token: None,
            pending_tokens: VecDeque::new(),
            speculative_speed: SpeculativeSpeedMonitor::default(),
            speculative_disabled: false,
            speculative_stats: SpeculativeStats::default(),
            finished: false,
            rng,
            prompt_cache_scope: None,
            prompt_cache_tickets: Vec::new(),
            shared_prompt_prefix: None,
            reused_prompt_tokens: 0,
            completed_input_cursor: None,
            input_prefix_trusted: effective_skip == 0,
            validated_prompt_vocab: None,
        }
    }

    /// Advance this session by exactly one sampled token.
    pub fn step(&mut self, stage: &mut PipelineStage) -> Result<GenerationStep> {
        self.step_with_prefill_budget(stage, usize::MAX)?
            .context("unbounded generation step did not produce a token")
    }

    /// Advance decode once, or process at most `max_prefill_tokens` text tokens.
    /// `None` means prompt processing made progress without sampling a token.
    /// Embedded/multimodal prompts retain their unsplit path: their attention
    /// spans and position IDs cannot in general be cut at a text boundary.
    /// A partial step waits for GPU completion before yielding to the caller.
    pub fn step_with_prefill_budget(
        &mut self,
        stage: &mut PipelineStage,
        max_prefill_tokens: usize,
    ) -> Result<Option<GenerationStep>> {
        self.completed_input_cursor = None;
        anyhow::ensure!(
            stage.position() == StagePosition::Solo,
            "continuous generation currently requires a solo stage"
        );
        anyhow::ensure!(!self.finished, "generation session is already finished");
        // Validate all request-controlled sampling fields before prefill or
        // decode can mutate this session's cache. A sampling error after the
        // forward pass would otherwise leave callers with a partial session
        // that must be discarded before retrying.
        self.validate_sampling_params(stage)?;
        if self.validated_prompt_vocab != Some(stage.vocab_size()) {
            stage.validate_token_ids(&self.prompt_tokens)?;
            self.validated_prompt_vocab = Some(stage.vocab_size());
        }
        stage.validate_context_budget(self.prompt_tokens.len(), self.params.max_tokens)?;
        anyhow::ensure!(
            max_prefill_tokens > 0,
            "prefill token budget must be positive"
        );

        if let Some(token) = self.pending_tokens.pop_front() {
            return Ok(Some(self.accept_token(token)));
        }

        let first_step = self.next_input_token.is_none();
        if self.is_text_prefilling() {
            return self.advance_text_prefill(stage, max_prefill_tokens);
        }

        let speculative_eligible = self.speculative_for_input(stage, 1);
        let policy = stage.options().speculative;
        let measure_scalar = speculative_eligible
            && policy.elapsed_time_fallback
            && policy.fallback
            && self.next_input_token.is_some()
            && self.speculative_speed.needs_probe();
        if let Some(anchor) = self.next_input_token
            && speculative_eligible
            && !measure_scalar
            && let Some(step) = self.step_speculative(stage, anchor)?
        {
            return Ok(Some(step));
        }

        let scalar_started = Instant::now();
        let (input, seq_len, start_pos) = if let Some(token) = self.next_input_token {
            let start_pos = self.prompt_tokens.len() + self.generated.len() - 1;
            let input = if let Some(delta) = self.mrope_decode_delta {
                let embeddings = stage.embed_tokens(&[token])?;
                StageInput::positioned_activations(
                    embeddings,
                    MropePositionIds::text_with_delta(start_pos, delta, 1)?,
                )
            } else {
                StageInput::Tokens(vec![token])
            };
            (input, 1, start_pos)
        } else if let Some(embeddings) = self.prompt_embeddings.take() {
            anyhow::ensure!(
                self.effective_skip == 0,
                "embedded prompts cannot reuse a cached text prefix"
            );
            let seq_len = embeddings.dim(0)?;
            let input = match self.prompt_positions.take() {
                Some(positions) => StageInput::positioned_activations_with_spans(
                    embeddings,
                    std::mem::take(&mut self.prompt_bidirectional_spans),
                    positions,
                ),
                None => StageInput::activations_with_spans(
                    embeddings,
                    std::mem::take(&mut self.prompt_bidirectional_spans),
                ),
            };
            (input, seq_len, 0)
        } else {
            anyhow::bail!("generation session has no decode token or prompt embeddings");
        };

        // Scalar probes still capture target features and inject them into
        // the draft cache. Its position must stay aligned before drafting.
        // Image prompts and M-RoPE decode rows feed it too.
        let output = if speculative_eligible {
            stage.speculative_prefill_with_session(&mut self.cache, input, seq_len, start_pos)?
        } else if first_step {
            stage.prefill_with_session(&mut self.cache, input, seq_len, start_pos)?
        } else if matches!(&input, StageInput::Tokens(_))
            && !self.cache.has_embedded_prompt
            && self.mrope_decode_delta.is_none()
        {
            stage.forward_text_with_session(
                &mut self.cache,
                input,
                seq_len,
                start_pos,
                TextRequestContext::new(self.prompt_tokens.len())?,
            )?
        } else {
            stage.forward_with_session(&mut self.cache, input, seq_len, start_pos)?
        };
        let logits = match output {
            StageOutput::Logits(logits) => logits,
            StageOutput::Activations(_) => anyhow::bail!("solo stage returned activations"),
        };
        self.cache.mrope_decode_delta = self.mrope_decode_delta;
        if first_step
            && let Some(scope) = self.prompt_cache_scope.as_deref()
            && let Some(ticket) = stage.capture_prompt(
                scope,
                &self.prompt_tokens,
                !self.speculative_disabled,
                &self.cache,
                &logits,
                false,
            )?
        {
            self.prompt_cache_tickets.push(ticket);
        }
        let token = self.sample_logits(&logits)?;
        if measure_scalar {
            self.speculative_speed
                .record_scalar(scalar_started.elapsed());
        }
        Ok(Some(self.accept_token(token)))
    }

    /// Whether a prompt still needs its first sampled token via chunked prefill.
    pub fn is_text_prefilling(&self) -> bool {
        self.next_input_token.is_none() && !self.finished && self.prompt_embeddings.is_none()
    }

    fn advance_text_prefill(
        &mut self,
        stage: &mut PipelineStage,
        max_tokens: usize,
    ) -> Result<Option<GenerationStep>> {
        let context = if !self.has_prompt_images
            && !self.cache.has_embedded_prompt
            && self.mrope_decode_delta.is_none()
        {
            Some(TextRequestContext::new(self.prompt_tokens.len())?)
        } else {
            None
        };
        if let Some(context) = context
            && !stage.prepare_text_request_with_session(&mut self.cache, context)?
        {
            // A retained decode prefix may use a different arithmetic rule
            // from this prompt. Rebuild it and report no skipped tokens.
            self.effective_skip = 0;
            self.reused_prompt_tokens = 0;
            self.input_prefix_trusted = true;
        }
        if self.effective_skip == 0
            && let Some(scope) = self.prompt_cache_scope.as_deref()
        {
            if self.has_prompt_images {
                if let Some(image_keys) = self.prompt_image_keys.as_deref()
                    && self.prompt_positions.is_none()
                    && self.mrope_decode_delta.is_none()
                {
                    if let Some((cache, logits, reused)) = stage.reuse_multimodal_prompt(
                        scope,
                        &self.prompt_tokens,
                        image_keys,
                        self.speculative_eligible(stage),
                    )? {
                        self.cache = cache;
                        self.effective_skip = reused;
                        self.reused_prompt_tokens = reused;
                        self.prune_consumed_image_spans(reused);
                        if reused == self.prompt_tokens.len() {
                            let logits =
                                logits.context("exact prompt cache hit requires logits")?;
                            let token = self.sample_logits(&logits)?;
                            return Ok(Some(self.accept_token(token)));
                        }
                    }
                } else {
                    let first_image_start =
                        self.prompt_image_spans.first().map_or(0, |span| span.start);
                    let pre_image_reusable = first_image_start > 0
                        && self.prompt_positions.as_ref().is_none_or(|positions| {
                            positions.is_identity_prefix(first_image_start)
                        });
                    if pre_image_reusable
                        && let Some((cache, reused)) = stage.reuse_prompt_prefix(
                            scope,
                            &self.prompt_tokens[..first_image_start],
                            self.speculative_eligible(stage),
                        )?
                    {
                        self.cache = cache;
                        self.effective_skip = reused;
                        self.reused_prompt_tokens = reused;
                    }
                }
            } else if let Some((mut cache, logits, reused)) =
                stage.reuse_prompt(scope, &self.prompt_tokens, !self.speculative_disabled)?
            {
                if let Some(context) = context {
                    anyhow::ensure!(
                        stage.prepare_text_request_with_session(&mut cache, context)?,
                        "prompt snapshot arithmetic changed after admission"
                    );
                }
                self.cache = cache;
                self.effective_skip = reused;
                self.reused_prompt_tokens = reused;
                if reused == self.prompt_tokens.len() {
                    let logits = logits.context("exact prompt cache hit requires logits")?;
                    let token = self.sample_logits(&logits)?;
                    return Ok(Some(self.accept_token(token)));
                }
            }
        }

        let mut remaining = max_tokens;
        while remaining > 0 {
            let start = self.effective_skip;
            let mut end = start
                .saturating_add(remaining.min(crate::DEFAULT_PREFILL_CHUNK_TOKENS))
                .min(self.prompt_tokens.len());
            if self.prompt_cache_scope.is_some()
                && let Some(boundary) = self.shared_prompt_prefix
                && boundary > start
            {
                end = end.min(boundary);
            }
            if !self.prompt_image_spans.is_empty() {
                let Some(aligned_end) =
                    self.align_chunk_end_for_image_spans(start, end, remaining < max_tokens)?
                else {
                    break;
                };
                end = aligned_end;
            }
            anyhow::ensure!(end > start, "prompt adds no tokens beyond cached prefix");
            let len = end - start;
            let input = self.build_prefill_chunk_input(stage, start, end)?;
            let complete = end == self.prompt_tokens.len();
            let shared = self.shared_prompt_prefix == Some(end);
            let needs_logits = complete || (shared && self.prompt_cache_scope.is_some());
            let speculative_eligible = self.speculative_for_input(stage, len);
            // Partial chunks need only decoder cache state. Keep real logits
            // at sampling/checkpoint boundaries, and leave draft captures on
            // their existing path so target/draft positions stay aligned.
            if !needs_logits && !speculative_eligible {
                if let Some(context) = context {
                    stage.prefill_text_cache_only_with_session(
                        &mut self.cache,
                        input,
                        len,
                        start,
                        context,
                    )?;
                } else {
                    stage.prefill_cache_only_with_session(&mut self.cache, input, len, start)?;
                }
                self.effective_skip = end;
                self.prune_consumed_image_spans(end);
                remaining = remaining.saturating_sub(len);
                if remaining > 0 {
                    // Metal retains a pass's temporary buffers until a device
                    // wait; release them before queuing the next chunk.
                    stage.device().synchronize()?;
                }
                continue;
            }
            let output = if speculative_eligible {
                stage.speculative_prefill_with_session(&mut self.cache, input, len, start)?
            } else if let Some(context) = context {
                stage.prefill_text_with_session(&mut self.cache, input, len, start, context)?
            } else {
                stage.prefill_with_session(&mut self.cache, input, len, start)?
            };
            let StageOutput::Logits(logits) = output else {
                anyhow::bail!("solo stage returned activations")
            };
            let can_capture_text = !self.has_prompt_images && !self.cache.has_embedded_prompt;
            let can_capture_multimodal =
                (complete || shared) && self.has_prompt_images && self.prompt_image_keys.is_some();
            if (complete || shared)
                && (can_capture_text || can_capture_multimodal)
                && self.mrope_decode_delta.is_none()
                && let Some(scope) = self.prompt_cache_scope.as_deref()
            {
                let captured_images: Vec<crate::architectures::ImageSpanKey> =
                    if can_capture_multimodal {
                        self.prompt_image_keys
                            .as_deref()
                            .unwrap_or(&[])
                            .iter()
                            .filter(|key| key.start.saturating_add(key.len) <= end)
                            .cloned()
                            .collect()
                    } else {
                        Vec::new()
                    };
                if let Some(ticket) = stage.capture_prompt_with_images(
                    scope,
                    &self.prompt_tokens[..end],
                    &captured_images,
                    !self.speculative_disabled,
                    &self.cache,
                    &logits,
                    shared,
                )? {
                    self.prompt_cache_tickets.push(ticket);
                }
            }
            self.effective_skip = end;
            self.prune_consumed_image_spans(end);
            remaining = remaining.saturating_sub(len);
            if complete {
                self.cache.mrope_decode_delta = self.mrope_decode_delta;
                self.prompt_positions = None;
                let token = self.sample_logits(&logits)?;
                // Only a reserved scope that may retain a cancelled prefix
                // needs an extra full-device completion barrier here. Other
                // requests keep their existing sampling/error ordering.
                if !self.has_prompt_images
                    && self
                        .prompt_cache_scope
                        .as_deref()
                        .is_some_and(|scope| stage.reserved_prompt_scope_registered(scope))
                {
                    stage.device().synchronize()?;
                    self.completed_input_cursor = Some(end);
                }
                return Ok(Some(self.accept_token(token)));
            }
        }
        // Bound completed work, not just command submission. This also exposes
        // GPU failures before a sibling lane's token leaves this scheduler tick.
        stage.device().synchronize()?;
        self.completed_input_cursor = (!self.has_prompt_images).then_some(self.effective_skip);
        Ok(None)
    }

    fn align_chunk_end_for_image_spans(
        &self,
        start: usize,
        mut end: usize,
        made_progress: bool,
    ) -> Result<Option<usize>> {
        for span in &self.prompt_image_spans {
            if !span.bidirectional {
                continue;
            }
            let span_end = span.end()?;
            if span_end <= start {
                continue;
            }
            if span.start >= end {
                break;
            }
            if span_end > end {
                if span.start > start {
                    // Stop right before the bidirectional image span so the next
                    // chunk starts cleanly at its first token.
                    end = span.start;
                } else if made_progress {
                    // Yield after earlier chunks in this budget so the oversized
                    // bidirectional span starts cleanly on the next tick.
                    return Ok(None);
                } else {
                    // This chunk starts at the bidirectional image span and no
                    // work has run yet in this tick; process the whole image in
                    // one chunk so its local mask stays intact.
                    end = span_end;
                }
                break;
            }
        }
        Ok(Some(end))
    }

    fn build_prefill_chunk_input(
        &self,
        stage: &PipelineStage,
        start: usize,
        end: usize,
    ) -> Result<StageInput> {
        let len = end - start;
        let tokens = &self.prompt_tokens[start..end];
        let overlapping = self
            .prompt_image_spans
            .iter()
            .filter_map(|span| match span.end() {
                Ok(span_end) if span_end > start && span.start < end => Some(Ok((span, span_end))),
                Ok(_) => None,
                Err(error) => Some(Err(error)),
            })
            .collect::<Result<Vec<_>>>()?;

        if overlapping.is_empty() {
            if let Some(positions) = self.prompt_positions.as_ref() {
                if positions.is_identity_prefix(end) {
                    return Ok(StageInput::Tokens(tokens.to_vec()));
                }
                return Ok(StageInput::positioned_activations(
                    stage.embed_tokens(tokens)?,
                    positions.slice(start, len)?,
                ));
            }
            if let Some(delta) = self.mrope_decode_delta {
                return Ok(StageInput::positioned_activations(
                    stage.embed_tokens(tokens)?,
                    MropePositionIds::text_with_delta(start, delta, len)?,
                ));
            }
            return Ok(StageInput::Tokens(tokens.to_vec()));
        }

        let mut pieces = Vec::with_capacity(overlapping.len() * 2 + 1);
        let mut bidirectional_spans = Vec::new();
        let mut cursor = start;
        for (span, span_end) in overlapping {
            if span.start > cursor {
                pieces.push(stage.embed_tokens(&self.prompt_tokens[cursor..span.start])?);
            }
            let ov_start = span.start.max(start);
            let ov_end = span_end.min(end);
            let ov_len = ov_end - ov_start;
            if span.bidirectional {
                anyhow::ensure!(
                    ov_start == span.start && ov_end == span_end,
                    "bidirectional image span [{}, {}) was split across prefill chunk [{}, {})",
                    span.start,
                    span_end,
                    start,
                    end
                );
                bidirectional_spans.push(AttentionSpan {
                    start: span.start,
                    len: ov_len,
                });
            }
            pieces.push(span.embeddings.narrow(0, ov_start - span.start, ov_len)?);
            cursor = ov_end;
        }
        if cursor < end {
            pieces.push(stage.embed_tokens(&self.prompt_tokens[cursor..end])?);
        }
        let embeddings = if pieces.len() == 1 {
            pieces.pop().expect("single piece")
        } else {
            let refs = pieces.iter().collect::<Vec<_>>();
            candle_core::Tensor::cat(&refs, 0)?
        };
        anyhow::ensure!(
            embeddings.dim(0)? == len,
            "prefill chunk embeddings had {} rows; expected {len}",
            embeddings.dim(0)?
        );
        if let Some(positions) = self.prompt_positions.as_ref() {
            Ok(StageInput::positioned_activations_with_spans(
                embeddings,
                bidirectional_spans,
                positions.slice(start, len)?,
            ))
        } else {
            Ok(StageInput::activations_with_spans(
                embeddings,
                bidirectional_spans,
            ))
        }
    }

    fn prune_consumed_image_spans(&mut self, consumed_end: usize) {
        if !self.prompt_image_spans.is_empty() {
            self.prompt_image_spans
                .retain(|span| span.end().is_ok_and(|span_end| span_end > consumed_end));
        }
    }

    fn speculative_eligible(&self, stage: &PipelineStage) -> bool {
        !self.speculative_disabled
            && (self.params.seed.is_none() && self.params.temperature > 0.0
                || stage.options().speculative.allow_approximate)
            && stage.session_speculative_ready(&self.cache)
            && !self.finished
    }

    fn speculative_for_input(&mut self, stage: &PipelineStage, rows: usize) -> bool {
        if !self.speculative_eligible(stage) {
            return false;
        }
        if !stage.speculative_rows_fit_context(&self.cache, rows) {
            // A target may have a larger context than its drafter. Continue
            // target generation without trying to inject rows past draft RoPE.
            self.speculative_disabled = true;
            self.speculative_stats.fell_back = true;
            return false;
        }
        true
    }

    fn step_speculative(
        &mut self,
        stage: &mut PipelineStage,
        anchor: u32,
    ) -> Result<Option<GenerationStep>> {
        self.params.validate()?;
        let pass_started = Instant::now();
        let remaining = self.params.max_tokens.saturating_sub(self.generated.len());
        let mut block_size = stage
            .effective_speculative_block_size()?
            .context("DFlash block size is missing")?;
        if stage.speculative_allows_short_block() {
            if remaining < 2 {
                return Ok(None);
            }
            block_size = block_size.min(remaining);
        } else if remaining < block_size {
            return Ok(None);
        }
        let pre_m5 = stage.uses_pre_m5_speculative_policy()?;
        let draft_sampling = if stage.speculative_supports_rejection_sampling()
            && self.params.temperature > 0.0
            && self.params.seed.is_none()
            && !self.params.has_penalty_controls()
            && self.params.tool_constraint.is_none()
            && std::env::var("DINF_DSPARK_REJECTION_SAMPLING").as_deref() != Ok("0")
        {
            let mut step_u = [0.0f32; 8];
            for u in &mut step_u {
                *u = self.rng.next_f32();
            }
            Some(crate::architectures::speculative_draft::DraftSampling {
                temperature: self.params.temperature as f32,
                top_k: self.params.top_k,
                top_p: self.params.top_p as f32,
                step_u,
            })
        } else {
            None
        };
        let Some(verification) = stage.begin_speculative_verification_with_sampling(
            &mut self.cache,
            anchor,
            block_size,
            draft_sampling.as_ref(),
        )?
        else {
            return Ok(None);
        };
        self.finish_speculative(stage, anchor, verification, Some(pass_started), pre_m5)
            .map(Some)
    }

    fn finish_speculative(
        &mut self,
        stage: &mut PipelineStage,
        anchor: u32,
        verification: super::SpeculativeVerification,
        pass_started: Option<Instant>,
        pre_m5: bool,
    ) -> Result<GenerationStep> {
        #[cfg(test)]
        draft_sampling_tests::capture_target_rows(&verification.logits, true);
        let proposed_count = verification.proposed.len();
        // A rejected pass restores target/draft state. Keep its random draws
        // tentative too, and publish them only after cache commit succeeds.
        let mut rng = self.rng.clone();
        let sampled = (|| -> Result<(Vec<u32>, usize, SamplingHistory)> {
            let mut outputs = Vec::with_capacity(proposed_count + 1);
            // Only penalized requests copy a map of distinct token counts.
            // No full prompt/history clone and no per-row history scan.
            let mut history = self.sampling_history.clone();
            let mut accepted = 0usize;
            let mut rejected = false;
            let mut stopped_early = false;
            let greedy = if Self::can_use_greedy_shortcut(&self.params) {
                Some(Self::greedy_verification_tokens(&verification.logits)?)
            } else {
                None
            };
            // Sample only visited rows (including the rejection/bonus row),
            // so batching readback never consumes extra RNG draws.
            let mut rows = if greedy.is_none() && !verification.logits.device().is_cpu() {
                Some(
                    verification
                        .logits
                        .to_dtype(DType::F32)?
                        .to_vec2::<f32>()?
                        .into_iter(),
                )
            } else {
                None
            };

            for (index, proposed) in verification.proposed.iter().copied().enumerate() {
                let (token, step_accepted) = if let Some(greedy) = greedy.as_ref() {
                    let tok = greedy[index];
                    (tok, tok == proposed)
                } else if let Some(dists) = verification.draft_distributions.as_ref() {
                    let row = if let Some(rows) = &mut rows {
                        rows.next().context("missing verification row")?
                    } else {
                        verification
                            .logits
                            .get(index)?
                            .to_dtype(DType::F32)?
                            .to_vec1::<f32>()?
                    };
                    crate::sampling::verify_rejection_sample_host(
                        &row,
                        proposed,
                        &dists[index],
                        &self.params,
                        &mut rng,
                    )?
                } else if let Some(rows) = &mut rows {
                    let tok = history.sample_host(
                        rows.next().context("missing verification row")?,
                        &self.params,
                        &mut rng,
                    )?;
                    (tok, tok == proposed)
                } else {
                    let tok =
                        history.sample(&verification.logits.get(index)?, &self.params, &mut rng)?;
                    (tok, tok == proposed)
                };
                outputs.push(token);
                history.push(token);
                if step_accepted {
                    accepted += 1;
                } else {
                    rejected = true;
                }
                if self.would_finish_after(token, outputs.len()) {
                    stopped_early = true;
                    break;
                }
                if rejected {
                    break;
                }
            }

            if !rejected && !stopped_early && accepted == proposed_count {
                let token = if let Some(greedy) = greedy.as_ref() {
                    greedy[proposed_count]
                } else if let Some(rows) = &mut rows {
                    history.sample_host(
                        rows.next().context("missing bonus row")?,
                        &self.params,
                        &mut rng,
                    )?
                } else {
                    history.sample(
                        &verification.logits.get(proposed_count)?,
                        &self.params,
                        &mut rng,
                    )?
                };
                outputs.push(token);
            }
            anyhow::ensure!(!outputs.is_empty(), "DFlash verification produced no token");
            Ok((outputs, accepted, history))
        })();

        let (outputs, accepted, history) = match sampled {
            Ok(result) => result,
            Err(error) => {
                stage.abort_speculative_verification(&mut self.cache, verification);
                return Err(error);
            }
        };
        let target_time = verification.target_started.elapsed();
        tracing::debug!(
            anchor,
            proposed = ?verification.proposed,
            target = ?outputs,
            accepted,
            draft_ms = verification.draft_time.as_secs_f64() * 1_000.0,
            target_ms = target_time.as_secs_f64() * 1_000.0,
            "DFlash verification"
        );

        // The last emitted token stays pending as the next target input. Every
        // earlier output was an accepted proposal and belongs in the replay.
        let accepted_to_replay = outputs.len() - 1;
        let draft_time = verification.draft_time;
        let commit_started = Instant::now();
        if let Err(error) =
            stage.commit_speculative_verification(&mut self.cache, verification, accepted_to_replay)
        {
            self.pending_tokens.clear();
            return Err(error.context("could not commit DFlash verification"));
        }
        self.rng = rng;
        // Each verified row was sampled under the tool grammar after the rows
        // before it, so the tentative matcher has consumed exactly `outputs`.
        // Token counts are recorded later, as the queued outputs are accepted.
        self.sampling_history.adopt_tool_state(history);
        self.speculative_stats.passes += 1;
        self.speculative_stats.drafted_tokens += proposed_count;
        self.speculative_stats.accepted_tokens += accepted;
        self.speculative_stats.committed_tokens += outputs.len();
        self.speculative_stats.draft_time += draft_time;
        self.speculative_stats.target_time += target_time;
        let commit_time = commit_started.elapsed();
        self.speculative_stats.commit_time += commit_time;
        let wall_time =
            pass_started.map_or(draft_time + target_time + commit_time, |t| t.elapsed());
        self.speculative_speed.record_pass(outputs.len(), wall_time);
        let policy = stage.options().speculative;
        let elapsed_time_policy = policy.elapsed_time_fallback;
        self.speculative_speed.floor_tenths = stage.speculative_fallback_floor_tenths();
        let (reached_hardware_limit, adaptive_fallback) = self.speculative_speed.fallback_decision(
            pre_m5,
            self.speculative_stats.passes,
            !policy.fallback,
            elapsed_time_policy,
        );
        if reached_hardware_limit || adaptive_fallback {
            self.speculative_disabled = true;
            self.speculative_stats.fell_back = true;
            tracing::debug!(
                total_passes = self.speculative_stats.passes,
                hardware_limit = reached_hardware_limit,
                adaptive_fallback,
                elapsed_time_policy,
                scalar_ms_per_token = self
                    .speculative_speed
                    .scalar_token_time()
                    .map(|time| time.as_secs_f64() * 1_000.0),
                "DFlash disabled for this generation session"
            );
        }
        self.speculative_stats.wall_time += wall_time;
        self.pending_tokens.extend(outputs);
        let token = self
            .pending_tokens
            .pop_front()
            .context("DFlash output queue is empty")?;
        Ok(self.accept_token(token))
    }

    fn sample_logits(&mut self, logits: &candle_core::Tensor) -> Result<u32> {
        #[cfg(test)]
        draft_sampling_tests::capture_target_rows(logits, false);
        self.sampling_history
            .sample(logits, &self.params, &mut self.rng)
    }

    fn validate_sampling_params(&self, stage: &PipelineStage) -> Result<()> {
        stage.validate_sampling_params(&self.params)
    }

    fn greedy_verification_tokens(logits: &candle_core::Tensor) -> Result<Vec<u32>> {
        #[cfg(target_os = "macos")]
        {
            let rows = (0..logits.dim(0)?)
                .map(|index| logits.i(index))
                .collect::<candle_core::Result<Vec<_>>>()?;
            if let Some(tokens) = crate::sampling_metal::try_greedy_argmax_batch(&rows)? {
                return Ok(tokens);
            }
        }
        // Also validate all logits on non-Metal devices and when the fused
        // path is disabled. Unselected NaN/Inf values must not pass silently.
        logits
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?
            .iter()
            .map(|row| super::sampling::checked_first_argmax(row))
            .collect()
    }

    fn can_use_greedy_shortcut(params: &SamplingParams) -> bool {
        params.temperature <= 0.0
            && params.tool_constraint.is_none()
            && params.repetition_penalty == 1.0
            && params.frequency_penalty == 0.0
            && params.presence_penalty == 0.0
            && params.logit_bias.is_empty()
    }

    #[cfg(target_os = "macos")]
    fn can_use_batched_metal_greedy(params: &SamplingParams) -> bool {
        Self::can_use_greedy_shortcut(params) && params.validate().is_ok()
    }

    fn would_finish_after(&self, token: u32, added: usize) -> bool {
        let generated = self.generated.len().saturating_add(added);
        (self.stop_ids.contains(&token) && generated >= self.params.min_tokens)
            || generated >= self.params.max_tokens
    }

    fn accept_token(&mut self, token: u32) -> GenerationStep {
        self.generated.push(token);
        self.sampling_history.push(token);
        self.next_input_token = Some(token);
        self.finished = (self.stop_ids.contains(&token)
            && self.generated.len() >= self.params.min_tokens)
            || self.generated.len() >= self.params.max_tokens;
        if self.finished {
            self.pending_tokens.clear();
        }

        GenerationStep {
            token,
            finished: self.finished,
        }
    }

    /// Position used to group this lane into a kernel-level decode batch.
    pub fn decode_batch_position(&self, stage: &PipelineStage) -> Option<usize> {
        (self.mrope_decode_delta.is_none()
            && !self.speculative_eligible(stage)
            && self.pending_tokens.is_empty()
            && stage.can_decode_batch_with_session(&self.cache)
            && self.next_input_token.is_some()
            && !self.finished)
            .then_some(self.cache.next_position)
    }

    /// Ready for a shared draft check, without waiting for another request.
    /// The optional elapsed-time fallback uses scalar probes, so it retains
    /// the existing separate-check path until shared-cost accounting qualifies.
    pub fn can_speculative_batch(&self, stage: &PipelineStage) -> bool {
        // The qualified shared path accepts exactly eight verification rows.
        let rows = 8;
        // The shared check verifies token rows at text positions.
        stage.supports_speculative_batch()
            && self.mrope_decode_delta.is_none()
            && !self.cache.has_embedded_prompt
            && self.speculative_eligible(stage)
            && self.next_input_token.is_some()
            && self.pending_tokens.is_empty()
            && !stage.options().speculative.elapsed_time_fallback
            && rows <= self.params.max_tokens.saturating_sub(self.generated.len())
            && stage.speculative_rows_fit_context(&self.cache, rows)
    }

    /// Advance ready draft lanes together. Each lane samples, commits, and
    /// reports its own result. Preflight failure permits separate checks;
    /// execution failure must not retry sampled tokens or partial GPU work.
    pub fn step_speculative_batch(
        stage: &mut PipelineStage,
        sessions: &mut [&mut GenerationSession],
    ) -> std::result::Result<Vec<Result<GenerationStep>>, DecodeBatchStepError> {
        use DecodeBatchStepError::RetryScalar;
        if !(2..=4).contains(&sessions.len())
            || !sessions.iter().all(|s| s.can_speculative_batch(stage))
        {
            return Err(RetryScalar(anyhow::anyhow!("ineligible draft batch")));
        }
        for session in sessions.iter() {
            session.params.validate().map_err(RetryScalar)?;
        }
        let rows = stage
            .effective_speculative_block_size()
            .map_err(RetryScalar)?
            .unwrap();
        let pre_m5 = stage
            .uses_pre_m5_speculative_policy()
            .map_err(RetryScalar)?;
        let anchors = sessions
            .iter()
            .map(|s| s.next_input_token.unwrap())
            .collect::<Vec<_>>();
        let verifications = stage.begin_speculative_batch(
            &mut sessions
                .iter_mut()
                .map(|s| &mut s.cache)
                .collect::<Vec<_>>(),
            &anchors,
            rows,
        )?;
        Ok(sessions
            .iter_mut()
            .zip(anchors)
            .zip(verifications)
            .map(|((s, anchor), verification)| {
                s.finish_speculative(stage, anchor, verification, None, pre_m5)
            })
            .collect())
    }

    /// Advance compatible lanes with one tensor-batched model forward.
    pub fn step_decode_batch(
        stage: &mut PipelineStage,
        sessions: &mut [&mut GenerationSession],
    ) -> std::result::Result<Vec<GenerationStep>, DecodeBatchStepError> {
        if sessions.is_empty() {
            return Err(DecodeBatchStepError::RetryScalar(anyhow::anyhow!(
                "decode batch cannot be empty"
            )));
        }
        if !sessions
            .iter()
            .all(|session| session.decode_batch_position(stage).is_some())
        {
            return Err(DecodeBatchStepError::RetryScalar(anyhow::anyhow!(
                "decode batch contains an ineligible lane"
            )));
        }
        let tokens = sessions
            .iter_mut()
            .map(|session| session.next_input_token.unwrap())
            .collect::<Vec<_>>();
        let mut states = sessions
            .iter_mut()
            .map(|session| &mut session.cache)
            .collect::<Vec<_>>();
        let revisions = states
            .iter()
            .map(|state| state.revision)
            .collect::<Vec<_>>();
        let logits = match stage.forward_decode_batch_with_sessions(&mut states, &tokens) {
            Ok(logits) => logits,
            Err(error) => {
                // Only preflight failures allow scalar fallback. Execution
                // failures reset the selected caches and advance revisions;
                // retrying those lanes would continue without their prompts.
                return Err(
                    if states
                        .iter()
                        .zip(revisions)
                        .all(|(state, revision)| state.revision == revision)
                    {
                        DecodeBatchStepError::RetryScalar(error)
                    } else {
                        DecodeBatchStepError::Terminal(error)
                    },
                );
            }
        };
        drop(states);
        if logits.len() != sessions.len() {
            return Err(DecodeBatchStepError::Terminal(anyhow::anyhow!(
                "decode batch output mismatch"
            )));
        }
        #[cfg(target_os = "macos")]
        let batched_greedy = if sessions
            .iter()
            .all(|session| Self::can_use_batched_metal_greedy(&session.params))
        {
            crate::sampling_metal::try_greedy_argmax_batch(&logits)
                .map_err(DecodeBatchStepError::Terminal)?
        } else {
            None
        };
        #[cfg(not(target_os = "macos"))]
        let batched_greedy: Option<Vec<u32>> = None;

        let tokens = match batched_greedy {
            Some(tokens) => tokens,
            None => sessions
                .iter_mut()
                .zip(&logits)
                .map(|(session, logits)| session.sample_logits(logits))
                .collect::<Result<Vec<_>>>()
                .map_err(DecodeBatchStepError::Terminal)?,
        };
        if tokens.len() != sessions.len() {
            return Err(DecodeBatchStepError::Terminal(anyhow::anyhow!(
                "decode batch sampled token count mismatch"
            )));
        }
        Ok(sessions
            .iter_mut()
            .zip(tokens)
            .map(|(session, token)| session.accept_token(token))
            .collect())
    }

    /// Return the tokens generated so far, excluding the prompt.
    ///
    /// The returned slice is owned by this session and remains valid until
    /// the next mutable operation on it.
    pub fn generated(&self) -> &[u32] {
        &self.generated
    }

    /// Return whether this session has reached a stop condition.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Mark a session complete after the caller detected a stop sequence in
    /// decoded text. The sampler has already committed the token
    /// that completed the sequence; clearing pending speculative output keeps
    /// the scheduler from exposing any tokens after the stop boundary.
    /// Clear a cache with pending speculative output so a later turn rebuilds
    /// its visible prompt without retaining hidden lookahead state.
    pub fn finish_text_stop(&mut self) {
        self.finished = true;
        if !self.pending_tokens.is_empty() {
            self.cache.clear();
            self.cache.resume_invalidated = true;
        }
        self.pending_tokens.clear();
    }

    /// Return a copy of the speculative-decoding counters collected so far.
    pub fn speculative_stats(&self) -> SpeculativeStats {
        self.speculative_stats
    }

    /// Apply the caller's speculative-decoding policy before this request starts.
    ///
    /// A greedy or seeded request decodes without its draft unless `stage`'s
    /// [`SpeculativeOptions::allow_approximate`](crate::SpeculativeOptions::allow_approximate)
    /// is set.
    pub fn set_speculative_decoding_enabled(&mut self, stage: &PipelineStage, enabled: bool) {
        // Batched target checks can differ numerically from scalar decoding.
        // A seed or greedy request needs the scalar path until parity holds.
        let exact = self.params.seed.is_some() || self.params.temperature <= 0.0;
        let approximate = stage.options().speculative.allow_approximate;
        self.speculative_disabled = !enabled || (exact && !approximate);
        self.speculative_stats.fell_back |= enabled && exact && !approximate;
    }

    /// Return the number of prompt/decode tokens currently retained in the
    /// stage cache for this session.
    pub fn kv_cache_tokens(&self, stage: &PipelineStage) -> usize {
        stage.session_kv_cache_tokens(&self.cache)
    }

    /// Return the next absolute position that this session will decode.
    pub fn next_position(&self, stage: &PipelineStage) -> usize {
        stage.session_next_position(&self.cache)
    }

    /// Return a terminal prefix length only when this text session can prove
    /// that every retained KV position belongs to its exact token prompt.
    /// Embedded/M-RoPE prompts intentionally do not publish this metadata:
    /// their tensor content is not represented by the token IDs alone.
    pub fn terminal_cache_tokens(&self, stage: &PipelineStage) -> Option<usize> {
        if self.prompt_embeddings.is_some()
            || self.has_prompt_images
            || self.cache.has_embedded_prompt
            || self.mrope_decode_delta.is_some()
            || self.cache.resume_invalidated
        {
            return None;
        }
        let next_position = self.next_position(stage);
        (next_position > 0
            && self.kv_cache_tokens(stage) == next_position
            && next_position
                <= self
                    .prompt_tokens
                    .len()
                    .saturating_add(self.generated.len()))
        .then_some(next_position)
    }

    /// Consume the session and return its stage cache for reuse. If a text stop
    /// discarded speculative lookahead, the next turn rebuilds the full prompt.
    /// The cache also retains the rotary offset from a positioned image prompt.
    pub fn into_cache(self) -> PipelineStageSessionState {
        self.cache
    }

    /// Enable text prompt reuse within a trusted caller's scope. Derive the
    /// scope from a trusted identity, never from unauthenticated request input.
    /// Monolithic embedded prompts never reuse it; sparse-image prompts may
    /// read a cached text prefix before the first image without publishing new
    /// snapshots.
    pub fn enable_prompt_cache(&mut self, scope: String) {
        if self.generated.is_empty()
            && self.prompt_embeddings.is_none()
            && !self.cache.has_embedded_prompt
            && (self.mrope_decode_delta.is_none() || self.has_prompt_images)
            && !scope.is_empty()
            && scope.len() <= 128
        {
            self.prompt_cache_scope = Some(scope);
        }
    }

    /// Save a shared system/tool prefix before processing the variable suffix.
    /// This is a token boundary in the actual encoded prompt, not a text length.
    pub fn set_shared_prompt_prefix(&mut self, tokens: usize) {
        if self.generated.is_empty()
            && self.prompt_embeddings.is_none()
            && !self.has_prompt_images
            && !self.cache.has_embedded_prompt
            && self.mrope_decode_delta.is_none()
            && tokens >= 64
            && tokens < self.prompt_tokens.len()
        {
            self.shared_prompt_prefix = Some(tokens);
        }
    }

    /// Number of prompt tokens skipped by this request's stage-local cache hit.
    pub fn reused_prompt_tokens(&self) -> usize {
        self.reused_prompt_tokens
    }

    /// Publish the prompt snapshot only after successful response delivery.
    pub fn publish_prompt_cache(&mut self, stage: &mut PipelineStage) {
        if self.finished {
            for ticket in self.prompt_cache_tickets.drain(..) {
                stage.publish_prompt(&ticket);
            }
        }
    }

    /// Best-effort retention of the input prefix after an explicit caller
    /// cancellation in a still-registered reserved scope (see
    /// [`crate::RESERVED_PROMPT_SCOPE_PREFIX`]). The caller must exclude
    /// shutdown, failed work and any tick-wide fatal error. Timeouts and generic
    /// discard/error/drop paths must not call this. A retained prefix may exist
    /// briefly before the scope is revoked; revocation drains and purges it.
    pub fn retain_cancelled_input_prefix(
        &mut self,
        stage: &mut PipelineStage,
        scope: &str,
    ) -> bool {
        // Captured snapshots contain only original input, independently copied
        // before decoding. Keep them on explicit cancellation even after the live
        // cache has advanced into generated text. The caller fences failed work.
        let mut retained_snapshot = false;
        if self.input_prefix_trusted
            && !self.cache.resume_invalidated
            && self.prompt_cache_scope.as_deref() == Some(scope)
            && stage.reserved_prompt_scope_registered(scope)
            && self.prompt_embeddings.is_none()
            && !self.has_prompt_images
            && self.prompt_positions.is_none()
            && self.prompt_bidirectional_spans.is_empty()
            && self.mrope_decode_delta.is_none()
            && self.cache.mrope_decode_delta.is_none()
            && self.cache.speculative_cache.is_none()
        {
            for ticket in self.prompt_cache_tickets.drain(..) {
                retained_snapshot |= stage.publish_prompt(&ticket);
            }
        }
        self.prompt_cache_tickets.clear();
        stage.prune_prompt_cache();
        let cursor = self.cache.next_position;
        let eligible = !retained_snapshot
            && self.input_prefix_trusted
            && self.completed_input_cursor == Some(cursor)
            && cursor > 0
            && cursor == self.cache.kv_cache_tokens
            && cursor == self.effective_skip
            && cursor <= self.prompt_tokens.len()
            && !self.cache.resume_invalidated
            && self.prompt_embeddings.is_none()
            && !self.has_prompt_images
            && self.prompt_positions.is_none()
            && self.prompt_bidirectional_spans.is_empty()
            && self.mrope_decode_delta.is_none()
            && self.cache.mrope_decode_delta.is_none()
            && self.cache.speculative_cache.is_none()
            && self.prompt_cache_scope.as_deref() == Some(scope)
            && stage.reserved_prompt_scope_registered(scope);
        let retained = eligible
            && stage.retain_cancelled_input(
                scope,
                &self.prompt_tokens[..cursor],
                !self.speculative_disabled,
                &mut self.cache,
            );
        self.completed_input_cursor = None;
        self.discard_cache(stage);
        retained || retained_snapshot
    }

    /// Release all model-side cache allocations held by this session.
    pub fn discard_cache(&mut self, stage: &mut PipelineStage) {
        self.prompt_cache_tickets.clear();
        stage.prune_prompt_cache();
        stage.clear_session_state(&mut self.cache);
    }
}

fn validate_image_spans(
    stage: &PipelineStage,
    prompt_len: usize,
    image_spans: &[ImageEmbeddingSpan],
) -> Result<()> {
    anyhow::ensure!(prompt_len > 0, "generation prompt must contain a token");
    anyhow::ensure!(
        !image_spans.is_empty(),
        "image prompt must contain at least one image span"
    );
    let hidden_dim = stage.hidden_dim();
    let mut prior_end = 0usize;
    for span in image_spans {
        let (rows, width) = span.embeddings.dims2()?;
        anyhow::ensure!(rows > 0, "image embedding span cannot be empty");
        anyhow::ensure!(
            width == hidden_dim,
            "image embedding width {width} does not match decoder width {hidden_dim}"
        );
        let end = span
            .start
            .checked_add(rows)
            .context("image embedding span overflow")?;
        anyhow::ensure!(
            span.start >= prior_end && end <= prompt_len,
            "image embedding spans must be sorted, non-overlapping, and within prompt bounds"
        );
        prior_end = end;
    }
    Ok(())
}

/// Fair FIFO admission plus round-robin tick scheduling.
pub struct ContinuousScheduler<T> {
    waiting: VecDeque<T>,
    active: VecDeque<T>,
    max_active: usize,
}

impl<T> ContinuousScheduler<T> {
    /// Create a scheduler with at least one active lane allowed at a time.
    pub fn new(max_active: usize) -> Self {
        Self {
            waiting: VecDeque::new(),
            active: VecDeque::new(),
            max_active: max_active.max(1),
        }
    }

    /// Enqueue an item in FIFO order and admit it when capacity is available.
    pub fn push(&mut self, item: T) {
        self.waiting.push_back(item);
        self.admit();
    }

    /// Advance one scheduler tick and return all currently active items.
    ///
    /// Returned items are removed from the active queue; callers that want an
    /// item to participate in a later tick must call [`Self::requeue`].
    pub fn next_tick(&mut self) -> Vec<T> {
        self.admit();
        self.active.drain(..).collect()
    }

    /// Put an item back into the active queue for a subsequent tick.
    pub fn requeue(&mut self, item: T) {
        self.active.push_back(item);
    }

    /// Return the number of items currently active in the scheduler.
    pub fn active_len(&self) -> usize {
        self.active.len()
    }

    /// Return the number of items waiting for an active slot.
    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    /// Iterate over active items followed by waiting items.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.active.iter().chain(self.waiting.iter())
    }

    /// Return whether the scheduler has no active or waiting items.
    pub fn is_empty(&self) -> bool {
        self.active.is_empty() && self.waiting.is_empty()
    }

    fn admit(&mut self) {
        while self.active.len() < self.max_active {
            let Some(item) = self.waiting.pop_front() else {
                break;
            };
            self.active.push_back(item);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::time::Duration;

    use super::{ContinuousScheduler, GenerationSession, SpeculativeSpeedMonitor};
    use candle_core::{Device, Tensor};

    use crate::architectures::{ImageEmbeddingSpan, MropePositionIds, StageInput};
    use crate::sampling::{
        SamplingHistory, SamplingParams, SamplingRng, apply_sampling_penalties,
        sample_token_with_rng,
    };

    fn speed_monitor(scalar_ms: [u64; 3]) -> SpeculativeSpeedMonitor {
        let mut monitor = SpeculativeSpeedMonitor::default();
        for time in scalar_ms {
            monitor.record_scalar(Duration::from_millis(time));
        }
        monitor
    }

    #[test]
    fn scheduler_bounds_active_lanes_and_admits_on_completion() {
        let mut scheduler = ContinuousScheduler::new(2);
        scheduler.push(1);
        scheduler.push(2);
        scheduler.push(3);
        assert_eq!(scheduler.active_len(), 2);
        assert_eq!(scheduler.waiting_len(), 1);

        let mut tick = scheduler.next_tick();
        assert_eq!(tick, vec![1, 2]);
        scheduler.requeue(tick.remove(0));

        let tick = scheduler.next_tick();
        assert_eq!(tick, vec![1, 3]);
        assert_eq!(scheduler.waiting_len(), 0);
    }

    #[test]
    fn zero_capacity_still_allows_one_lane() {
        let mut scheduler = ContinuousScheduler::new(0);
        scheduler.push("session");
        assert_eq!(scheduler.next_tick(), vec!["session"]);
    }

    #[test]
    fn constrained_drafting_matches_scalar_and_advances_the_tool_grammar() -> anyhow::Result<()> {
        let (mut plain, _) = crate::session_state_tests::speculative_boundary_stage(32, None)?;
        let (mut drafted, _) =
            crate::session_state_tests::speculative_boundary_stage(32, Some(32))?;
        let params = SamplingParams {
            temperature: 0.8,
            max_tokens: 9,
            min_tokens: 9,
            seed: None,
            tool_constraint: Some(
                crate::tool_constraints::ToolConstraint::test_required_lookup(3)?,
            ),
            ..Default::default()
        };
        let mut reference = GenerationSession::new(&plain, vec![2], params.clone(), vec![]);
        let mut session = GenerationSession::new(&drafted, vec![2], params, vec![]);
        reference.rng = SamplingRng::seeded(7);
        session.rng = SamplingRng::seeded(7);
        assert!(session.speculative_eligible(&drafted));
        while !reference.is_finished() {
            assert_eq!(reference.step(&mut plain)?, session.step(&mut drafted)?);
        }
        // The grammar permits the single call once, then only EOS. A session
        // grammar left behind by a draft pass would allow the call again.
        assert_eq!(session.generated(), reference.generated());
        assert_eq!(session.generated(), &[0, 1, 1, 1, 1, 1, 1, 1, 1]);
        assert!(
            session.speculative_stats().passes > 1,
            "the constrained request must run several draft passes"
        );
        session.discard_cache(&mut drafted);
        reference.discard_cache(&mut plain);
        Ok(())
    }

    #[test]
    fn committed_draft_pass_advances_the_session_tool_grammar() -> anyhow::Result<()> {
        let (mut stage, _) = crate::session_state_tests::speculative_boundary_stage(32, Some(32))?;
        // Bias the call token away so the scalar first step emits prose; the
        // call then first appears inside a verified draft pass.
        let params = SamplingParams {
            temperature: 0.8,
            max_tokens: 8,
            seed: None,
            logit_bias: [(0, -100.0), (2, 100.0)].into_iter().collect(),
            tool_constraint: Some(
                crate::tool_constraints::ToolConstraint::test_required_lookup(3)?,
            ),
            ..Default::default()
        };
        let mut session = GenerationSession::new(&stage, vec![2], params.clone(), vec![]);
        session.rng = SamplingRng::seeded(42);
        assert!(session.speculative_eligible(&stage));
        assert_eq!(session.step(&mut stage)?.token, 2);
        let anchor = session.next_input_token.unwrap();
        let mut verification = stage
            .begin_speculative_verification(&mut session.cache, anchor, 2)?
            .expect("real draft verification must execute");
        // Row 0 strongly prefers the call; any later row may only end.
        verification.logits = Tensor::from_vec(
            vec![1000.0f32, -1000.0, -1000.0, 0.0, 0.0, 0.0],
            (2, 3),
            &Device::Cpu,
        )?;
        let step = session.finish_speculative(&mut stage, anchor, verification, None, false)?;
        assert_eq!(step.token, 0);
        // The session grammar must now be past the call: only EOS remains.
        // A grammar left before the call would pick prose or the call here.
        let mut rng = SamplingRng::seeded(1);
        assert_eq!(
            session
                .sampling_history
                .sample_host(vec![200.0, 0.0, 9.0], &params, &mut rng)?,
            1
        );
        session.discard_cache(&mut stage);
        Ok(())
    }

    #[test]
    fn speculative_window_fallback_preserves_rng_and_penalty_history() -> anyhow::Result<()> {
        let (mut plain, _) = crate::session_state_tests::speculative_boundary_stage(32, None)?;
        let (mut drafted, _) =
            crate::session_state_tests::speculative_boundary_stage(32, Some(32))?;
        plain.set_max_kv_tokens(4);
        drafted.set_max_kv_tokens(4);
        let params = SamplingParams {
            temperature: 2.5,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.1,
            frequency_penalty: 0.15,
            presence_penalty: 0.1,
            max_tokens: 8,
            min_tokens: 8,
            seed: None,
            ..Default::default()
        };
        let mut reference = GenerationSession::new(&plain, vec![2, 0, 0], params.clone(), vec![]);
        let mut session = GenerationSession::new(&drafted, vec![2, 0, 0], params.clone(), vec![]);
        // Fix the test RNG without setting the request's exact-sampling flag:
        // this must reach draft preflight, not the seeded scalar policy.
        reference.rng = SamplingRng::seeded(42);
        session.rng = SamplingRng::seeded(42);
        assert!(session.speculative_eligible(&drafted));
        while !reference.is_finished() {
            assert_eq!(reference.step(&mut plain)?, session.step(&mut drafted)?);
        }
        assert_eq!(session.generated(), reference.generated());
        assert_eq!(session.speculative_stats().passes, 0);
        assert_eq!(
            session
                .cache
                .speculative_cache
                .as_ref()
                .unwrap()
                .next_position(),
            session.cache.next_position
        );
        // Broad scores expose extra RNG draws or wrong history updates that
        // strongly peaked model scores could hide during the request.
        for _ in 0..128 {
            let scores = vec![0.2, -0.1, 0.4];
            assert_eq!(
                reference.sampling_history.sample_host(
                    scores.clone(),
                    &params,
                    &mut reference.rng
                )?,
                session
                    .sampling_history
                    .sample_host(scores, &params, &mut session.rng)?,
            );
        }
        session.discard_cache(&mut drafted);
        reference.discard_cache(&mut plain);
        assert_eq!(drafted.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    fn speculative_error_restores_sampling(commit_error: bool) -> anyhow::Result<()> {
        let (mut stage, calls) =
            crate::session_state_tests::speculative_boundary_stage(32, Some(32))?;
        let params = SamplingParams {
            temperature: 0.8,
            top_k: 1,
            max_tokens: 8,
            seed: None,
            ..Default::default()
        };
        let mut session = GenerationSession::new(&stage, vec![2], params.clone(), vec![]);
        session.rng = SamplingRng::seeded(42);
        assert!(session.speculative_eligible(&stage));
        session.step(&mut stage)?;
        let position = session.cache.next_position;
        let allocated = stage.total_kv_cache_allocation_tokens();
        let generated = session.generated.clone();
        let mut expected_rng = session.rng.clone();
        let anchor = session.next_input_token.unwrap();
        let mut verification = stage
            .begin_speculative_verification(&mut session.cache, anchor, 2)?
            .expect("real draft verification must execute");
        assert!(calls.lock().unwrap().iter().any(|(_, _, all)| *all));
        if commit_error {
            // A malformed captured feature reaches draft injection only after
            // sampling and target-cache commit have both run.
            verification.captures[0] =
                Tensor::zeros((2, 63), candle_core::DType::F32, &Device::Cpu)?;
        } else {
            // Accept row zero so the later nonfinite row fails after one draw.
            let mut scores = vec![-100.0f32; 6];
            scores[verification.proposed[0] as usize] = 10.0;
            scores[3] = f32::NAN;
            verification.logits = Tensor::from_vec(scores, (2, 3), &Device::Cpu)?;
        }
        let error = session
            .finish_speculative(&mut stage, anchor, verification, None, false)
            .unwrap_err();
        assert!(error.to_string().contains(if commit_error {
            "could not commit"
        } else {
            "finite"
        }));
        assert_eq!(session.cache.next_position, position);
        assert_eq!(session.cache.kv_cache_tokens, position);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), allocated);
        assert_eq!(
            session
                .cache
                .speculative_cache
                .as_ref()
                .unwrap()
                .next_position(),
            position
        );
        assert_eq!(session.generated, generated);
        assert!(session.pending_tokens.is_empty());
        assert_eq!(session.speculative_stats().passes, 0);
        // Errors remain errors, not a promise to retry a failed device. But a
        // rolled-back pass must not spend request-owned random draws either.
        let broad = SamplingParams {
            top_k: 0,
            top_p: 1.0,
            ..params
        };
        let scores = Tensor::new(&[0.2f32, -0.1, 0.4], &Device::Cpu)?;
        for _ in 0..128 {
            assert_eq!(
                sample_token_with_rng(&scores, &broad, &mut session.rng)?,
                sample_token_with_rng(&scores, &broad, &mut expected_rng)?,
                "rolled-back verification consumed sampling randomness"
            );
        }
        session.discard_cache(&mut stage);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn speculative_sampling_error_restores_rng_and_caches() -> anyhow::Result<()> {
        speculative_error_restores_sampling(false)
    }

    #[test]
    fn speculative_commit_error_restores_rng_and_caches() -> anyhow::Result<()> {
        speculative_error_restores_sampling(true)
    }

    #[test]
    fn speculative_success_commits_only_visited_rng_draws() -> anyhow::Result<()> {
        let (mut stage, _) = crate::session_state_tests::speculative_boundary_stage(32, Some(32))?;
        let params = SamplingParams {
            temperature: 0.8,
            top_k: 1,
            seed: None,
            max_tokens: 3,
            ..Default::default()
        };
        let mut session = GenerationSession::new(&stage, vec![0], params.clone(), vec![]);
        session.rng = SamplingRng::seeded(42);
        session.step(&mut stage)?;
        let mut expected_rng = session.rng.clone();
        session.step(&mut stage)?;
        assert_eq!(session.speculative_stats().passes, 1);
        assert_eq!(session.speculative_stats().committed_tokens, 2);
        let scores = Tensor::new(&[10.0f32, 0.0, -100.0], &Device::Cpu)?;
        for _ in 0..2 {
            sample_token_with_rng(&scores, &params, &mut expected_rng)?;
        }
        session.step(&mut stage)?; // Draining queued output spends no draw.
        let broad = SamplingParams {
            top_k: 0,
            top_p: 1.0,
            ..params
        };
        let scores = Tensor::new(&[0.2f32, -0.1, 0.4], &Device::Cpu)?;
        for _ in 0..128 {
            assert_eq!(
                sample_token_with_rng(&scores, &broad, &mut session.rng)?,
                sample_token_with_rng(&scores, &broad, &mut expected_rng)?
            );
        }
        session.discard_cache(&mut stage);
        Ok(())
    }

    #[test]
    fn exact_seeded_request_keeps_scalar_draft_fallback() -> anyhow::Result<()> {
        let (mut plain, _) = crate::session_state_tests::speculative_boundary_stage(32, None)?;
        let (mut drafted, calls) =
            crate::session_state_tests::speculative_boundary_stage(32, Some(32))?;
        let params = SamplingParams {
            temperature: 2.5,
            top_k: 0,
            top_p: 1.0,
            seed: Some(42),
            max_tokens: 8,
            ..Default::default()
        };
        let mut reference = GenerationSession::new(&plain, vec![2], params.clone(), vec![]);
        let mut session = GenerationSession::new(&drafted, vec![2], params, vec![]);
        session.set_speculative_decoding_enabled(&drafted, true);
        while !reference.is_finished() {
            assert_eq!(reference.step(&mut plain)?, session.step(&mut drafted)?);
        }
        assert_eq!(session.speculative_stats().passes, 0);
        assert!(session.speculative_stats().fell_back);
        assert!(calls.lock().unwrap().iter().all(|(_, _, all)| !all));
        reference.discard_cache(&mut plain);
        session.discard_cache(&mut drafted);

        // An explicit `new_seeded` seed is exact too, even with no params seed.
        let unseeded = SamplingParams {
            seed: None,
            ..params_for_seeded()
        };
        let mut reference =
            GenerationSession::new_seeded(&plain, vec![2], unseeded.clone(), vec![], 42);
        let mut session = GenerationSession::new_seeded(&drafted, vec![2], unseeded, vec![], 42);
        session.set_speculative_decoding_enabled(&drafted, true);
        while !reference.is_finished() {
            assert_eq!(reference.step(&mut plain)?, session.step(&mut drafted)?);
        }
        assert_eq!(session.speculative_stats().passes, 0);
        assert!(session.speculative_stats().fell_back);
        reference.discard_cache(&mut plain);
        session.discard_cache(&mut drafted);
        Ok(())
    }

    #[test]
    fn allow_approximate_option_lets_a_seeded_request_draft() -> anyhow::Result<()> {
        let (mut drafted, _) =
            crate::session_state_tests::speculative_boundary_stage(32, Some(32))?;
        drafted.set_options(crate::EngineOptions {
            speculative: crate::SpeculativeOptions {
                allow_approximate: true,
                ..Default::default()
            },
            ..Default::default()
        })?;
        let params = SamplingParams {
            seed: Some(42),
            ..params_for_seeded()
        };
        let mut session = GenerationSession::new(&drafted, vec![2], params, vec![]);
        session.set_speculative_decoding_enabled(&drafted, true);
        while !session.is_finished() {
            session.step(&mut drafted)?;
        }
        assert!(session.speculative_stats().passes > 0);
        session.discard_cache(&mut drafted);
        Ok(())
    }

    fn params_for_seeded() -> SamplingParams {
        SamplingParams {
            temperature: 2.5,
            top_k: 0,
            top_p: 1.0,
            max_tokens: 8,
            ..Default::default()
        }
    }

    #[test]
    fn actual_draft_stop_rebuilds_visible_prefix_and_preserves_prompt_snapshot()
    -> anyhow::Result<()> {
        let (mut stage, calls) =
            crate::session_state_tests::speculative_boundary_stage(32, Some(32))?;
        stage.set_prompt_cache_limits(1 << 20, 4, Duration::from_secs(60));
        let params = SamplingParams {
            temperature: 0.8,
            top_k: 1,
            seed: None,
            max_tokens: 4,
            ..Default::default()
        };
        let mut session = GenerationSession::new(&stage, vec![0, 0], params.clone(), vec![]);
        session.enable_prompt_cache("draft-owner".into());
        session.step(&mut stage)?;
        session.step(&mut stage)?;
        assert_eq!(session.speculative_stats().passes, 1);
        assert_eq!(session.pending_tokens.len(), 1);
        assert_eq!(session.cache.next_position, 4);
        session.finish_text_stop();
        assert!(session.cache.resume_invalidated);
        assert_eq!(session.cache.next_position, 0);
        assert_eq!(
            session
                .cache
                .speculative_cache
                .as_ref()
                .unwrap()
                .next_position(),
            0
        );
        session.publish_prompt_cache(&mut stage);
        // Only the earlier original-input snapshot survives the hidden suffix.
        assert_eq!(
            stage.prompt_cache_match_tokens("draft-owner", &[0, 0]),
            (0, 2)
        );
        let mut resumed = GenerationSession::with_cache(
            session.into_cache(),
            vec![0; 5],
            params.clone(),
            vec![],
            4,
        );
        let first_new_call = calls.lock().unwrap().len();
        resumed.step(&mut stage)?;
        assert_eq!(calls.lock().unwrap()[first_new_call], (5, 0, false));
        resumed.discard_cache(&mut stage);
        // Forking the real target+draft snapshot must permit a new draft pass.
        let mut warm = GenerationSession::new(&stage, vec![0, 0], params, vec![]);
        warm.enable_prompt_cache("draft-owner".into());
        let before = calls.lock().unwrap().len();
        warm.step(&mut stage)?;
        assert_eq!(warm.reused_prompt_tokens(), 2);
        assert_eq!(calls.lock().unwrap().len(), before);
        assert!(stage.session_speculative_ready(&warm.cache));
        warm.step(&mut stage)?;
        assert_eq!(warm.speculative_stats().passes, 1);
        warm.discard_cache(&mut stage);
        stage.clear_prompt_cache();
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn retained_wrong_arithmetic_rebuilds_before_using_skip_prefix() -> anyhow::Result<()> {
        use crate::{TextArithmeticPolicy, TextRequestContext};
        let mut stage = crate::session_state_tests::test_stage();
        let prompt = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let params = SamplingParams {
            temperature: 0.7,
            max_tokens: 4,
            min_tokens: 4,
            seed: Some(42),
            ..Default::default()
        };
        let mut cache = stage.new_session_state();
        stage.prefill_with_session(&mut cache, StageInput::Tokens(vec![0, 1]), 2, 0)?;
        cache.text_arithmetic = Some(TextArithmeticPolicy::phi_k(TextRequestContext::new(8)?));
        let mut resumed =
            GenerationSession::with_cache(cache, prompt.clone(), params.clone(), vec![], 2);
        assert_eq!(resumed.effective_skip, 2);
        assert!(resumed.step_with_prefill_budget(&mut stage, 1)?.is_none());
        // A cold replay starts at zero, rather than appending at the stale
        // skip hint. The old charged cache has been released.
        assert_eq!(resumed.next_position(&stage), 1);
        assert_eq!(resumed.effective_skip, 1);
        assert_eq!(resumed.reused_prompt_tokens(), 0);
        assert_eq!(resumed.cache.text_arithmetic, None);
        while !resumed.is_finished() {
            resumed.step_with_prefill_budget(&mut stage, 3)?;
        }
        let actual = resumed.generated().to_vec();
        resumed.discard_cache(&mut stage);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        let mut fresh = GenerationSession::new(&stage, prompt, params, vec![]);
        while !fresh.is_finished() {
            fresh.step(&mut stage)?;
        }
        assert_eq!(actual, fresh.generated());
        fresh.discard_cache(&mut stage);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn bounded_prefill_preserves_tokens_rng_and_positions() -> anyhow::Result<()> {
        let mut stage = crate::session_state_tests::test_stage();
        let prompt = (0..137).map(|i| i % 2).collect::<Vec<_>>();
        let params = SamplingParams {
            temperature: 0.8,
            repetition_penalty: 1.1,
            frequency_penalty: 0.1,
            max_tokens: 12,
            min_tokens: 12,
            seed: Some(42),
            ..Default::default()
        };
        let mut reference = GenerationSession::new(&stage, prompt.clone(), params.clone(), vec![]);
        while !reference.is_finished() {
            reference.step(&mut stage)?;
        }
        reference.discard_cache(&mut stage);
        for budget in [1, 7, 64, 128] {
            let mut session =
                GenerationSession::new(&stage, prompt.clone(), params.clone(), vec![]);
            let mut processed = 0;
            while session.is_text_prefilling() {
                let step = session.step_with_prefill_budget(&mut stage, budget)?;
                processed = (processed + budget).min(prompt.len());
                assert_eq!(session.cache.next_position, processed);
                assert_eq!(step.is_some(), processed == prompt.len());
                assert_eq!(session.generated().len(), usize::from(step.is_some()));
                assert_eq!(
                    session.decode_batch_position(&stage).is_some(),
                    step.is_some()
                );
            }
            while !session.is_finished() {
                assert!(
                    session
                        .step_with_prefill_budget(&mut stage, budget)?
                        .is_some()
                );
            }
            assert_eq!(
                session.generated(),
                reference.generated(),
                "budget {budget}"
            );
            session.discard_cache(&mut stage);
            assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        }
        Ok(())
    }

    #[test]
    fn bounded_prefill_rejects_zero_without_mutating_state() -> anyhow::Result<()> {
        let mut stage = crate::session_state_tests::test_stage();
        let mut session =
            GenerationSession::new(&stage, vec![0; 12], SamplingParams::default(), vec![]);
        assert!(session.step_with_prefill_budget(&mut stage, 0).is_err());
        assert_eq!(session.cache.next_position, 0);
        assert!(session.generated().is_empty());
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn invalid_sampling_rejects_before_prefill_mutates_cache() -> anyhow::Result<()> {
        let mut stage = crate::session_state_tests::test_stage();
        for (field, top_p, repetition_penalty) in [
            ("top_p", f64::NAN, 1.0),
            ("top_p", 0.0, 1.0),
            ("repetition penalty", 0.9, 1e300),
            ("repetition penalty", 0.9, 1e-300),
        ] {
            let invalid = SamplingParams {
                top_p,
                repetition_penalty,
                ..SamplingParams::default()
            };
            let mut session = GenerationSession::new(&stage, vec![0, 1], invalid, vec![]);
            let error = session.step(&mut stage).unwrap_err();
            assert!(error.to_string().contains(field), "{error:#}");
            assert_eq!(session.kv_cache_tokens(&stage), 0);
            assert_eq!(session.next_position(&stage), 0);
            assert!(session.generated().is_empty());
            assert_eq!(stage.total_kv_cache_tokens(), 0);
            assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        }

        let invalid_bias = SamplingParams {
            logit_bias: [(stage.vocab_size() as u32, 1.0)].into_iter().collect(),
            ..SamplingParams::default()
        };
        let mut bias_session = GenerationSession::new(&stage, vec![0, 1], invalid_bias, vec![]);
        let error = bias_session.step(&mut stage).unwrap_err();
        assert!(error.to_string().contains("outside the model vocabulary"));
        assert_eq!(bias_session.kv_cache_tokens(&stage), 0);
        assert_eq!(bias_session.next_position(&stage), 0);
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        Ok(())
    }

    #[test]
    fn valid_sampling_controls_still_prefill_and_sample() -> anyhow::Result<()> {
        let mut stage = crate::session_state_tests::test_stage();
        let params = SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
            max_tokens: 1,
            seed: Some(7),
            ..SamplingParams::default()
        };
        let mut session = GenerationSession::new(&stage, vec![0, 1], params, vec![]);
        let step = session.step(&mut stage)?;
        assert!(step.finished);
        assert_eq!(session.generated().len(), 1);
        assert_eq!(session.next_position(&stage), 2);
        session.discard_cache(&mut stage);
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        Ok(())
    }

    #[test]
    fn text_stop_finishes_session_and_discards_pending_tokens() {
        let stage = crate::session_state_tests::test_stage();
        let mut session =
            GenerationSession::new(&stage, vec![0; 4], SamplingParams::default(), vec![]);
        session.pending_tokens.extend([7, 8, 9]);

        session.finish_text_stop();

        assert!(session.is_finished());
        assert!(session.pending_tokens.is_empty());
    }

    #[test]
    fn text_stop_with_lookahead_rebuilds_visible_prompt_and_releases_only_its_cache()
    -> anyhow::Result<()> {
        let mut stage = crate::session_state_tests::preallocated_test_stage();
        stage.fixed_cache_bytes = 32;
        let mut peer = stage.new_session_state();
        stage.prefill_with_session(&mut peer, StageInput::Tokens(vec![0]), 1, 0)?;
        let peer_charge = peer.kv_cache_allocation_tokens + peer.fixed_cache_allocation_tokens;
        let params = SamplingParams {
            max_tokens: 3,
            seed: Some(42),
            repetition_penalty: 1.05,
            ..Default::default()
        };
        let mut session = GenerationSession::new(
            &stage,
            vec![0; 4],
            SamplingParams {
                max_tokens: 8,
                ..params.clone()
            },
            vec![],
        );
        let anchor = session.step(&mut stage)?.token;
        // A verified block consumes the anchor and accepted future tokens,
        // then exposes one token while the rest await later scheduler ticks.
        stage.forward_with_session(
            &mut session.cache,
            StageInput::Tokens(vec![anchor, 0, 1]),
            3,
            4,
        )?;
        session.accept_token(0);
        session.pending_tokens.extend([1, 0]);
        let stale_skip = session.next_position(&stage);
        assert_eq!(stale_skip, 7);
        assert!(stage.total_kv_cache_allocation_tokens() > peer_charge);

        session.finish_text_stop();
        session.finish_text_stop(); // Repeated stop notification keeps invalidation.
        assert_eq!(session.next_position(&stage), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), peer_charge);
        assert_eq!(peer.next_position, 1);
        let mut visible_prompt = vec![0; 4];
        visible_prompt.extend_from_slice(session.generated());
        visible_prompt.extend([1; 4]);
        let cache = session.into_cache();
        assert!(cache.resume_invalidated);
        let mut resumed = GenerationSession::with_cache(
            cache,
            visible_prompt.clone(),
            params.clone(),
            vec![],
            stale_skip,
        );
        assert_eq!(resumed.effective_skip, 0);
        assert!(!resumed.cache.resume_invalidated);
        let mut reference = GenerationSession::new(&stage, visible_prompt, params, vec![]);
        while !reference.is_finished() {
            let expected = reference.step(&mut stage)?;
            let actual = resumed.step(&mut stage)?;
            assert_eq!(actual.token, expected.token);
            assert_eq!(actual.finished, expected.finished);
            assert_eq!(
                resumed.next_position(&stage),
                reference.next_position(&stage)
            );
        }
        reference.discard_cache(&mut stage);
        resumed.discard_cache(&mut stage);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), peer_charge);
        stage.clear_session_state(&mut peer);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn text_stop_without_lookahead_keeps_prefix_resumable() -> anyhow::Result<()> {
        let mut stage = crate::session_state_tests::test_stage();
        let params = SamplingParams {
            max_tokens: 2,
            ..Default::default()
        };
        let mut session = GenerationSession::new(&stage, vec![0; 4], params.clone(), vec![]);
        session.step(&mut stage)?;
        session.finish_text_stop();
        assert_eq!(session.next_position(&stage), 4);
        assert!(!session.cache.resume_invalidated);
        let mut resumed = GenerationSession::with_cache(
            session.into_cache(),
            vec![0, 0, 0, 0, 1, 0],
            params,
            vec![],
            4,
        );
        assert_eq!(resumed.effective_skip, 4);
        resumed.step(&mut stage)?;
        assert_eq!(resumed.next_position(&stage), 6);
        assert_eq!(stage.total_kv_cache_tokens(), 6);
        resumed.discard_cache(&mut stage);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn ordinary_resume_position_mismatch_still_rejects_without_clearing_cache() -> anyhow::Result<()>
    {
        let mut stage = crate::session_state_tests::test_stage();
        let mut session =
            GenerationSession::new(&stage, vec![0; 4], SamplingParams::default(), vec![]);
        session.step(&mut stage)?;
        let mut resumed = GenerationSession::with_cache(
            session.into_cache(),
            vec![0; 6],
            SamplingParams::default(),
            vec![],
            3,
        );
        assert_eq!(resumed.effective_skip, 3);
        assert!(resumed.step(&mut stage).is_err());
        assert_eq!(resumed.next_position(&stage), 4);
        assert_eq!(stage.total_kv_cache_tokens(), 4);
        resumed.discard_cache(&mut stage);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn bounded_prefill_keeps_embedded_prompts_unsplit() -> anyhow::Result<()> {
        let mut stage = crate::session_state_tests::test_stage();
        let embeddings = Tensor::zeros((70, 1), candle_core::DType::F32, &Device::Cpu)?;
        let mut session = GenerationSession::new_with_embeddings(
            &stage,
            vec![0; 70],
            embeddings,
            SamplingParams::default(),
            vec![],
        )?;
        assert!(!session.is_text_prefilling());
        assert!(session.step_with_prefill_budget(&mut stage, 1)?.is_some());
        assert_eq!(session.cache.next_position, 70);
        session.discard_cache(&mut stage);
        Ok(())
    }

    #[test]
    fn embedded_prompt_never_claims_a_token_only_cache_identity() -> anyhow::Result<()> {
        let mut stage = crate::session_state_tests::test_stage();
        let params = SamplingParams {
            max_tokens: 1,
            temperature: 0.0,
            ..Default::default()
        };
        let mut session = GenerationSession::new_with_embeddings(
            &stage,
            vec![1; 2],
            Tensor::zeros((2, 1), candle_core::DType::F32, &Device::Cpu)?,
            params.clone(),
            vec![],
        )?;
        session.step(&mut stage)?;
        assert_eq!(session.terminal_cache_tokens(&stage), None);

        let mut resumed =
            GenerationSession::with_cache(session.into_cache(), vec![1, 1, 0], params, vec![], 2);
        resumed.step(&mut stage)?;
        assert_eq!(resumed.terminal_cache_tokens(&stage), None);
        resumed.discard_cache(&mut stage);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        let mut fresh = GenerationSession::with_cache(
            resumed.into_cache(),
            vec![1],
            SamplingParams {
                max_tokens: 1,
                ..Default::default()
            },
            vec![],
            0,
        );
        fresh.step(&mut stage)?;
        assert_eq!(fresh.terminal_cache_tokens(&stage), Some(1));
        Ok(())
    }

    #[test]
    fn resumed_embedded_prompt_cannot_poison_text_prompt_reuse() -> anyhow::Result<()> {
        use crate::architectures::{KvCache, LayerCache};

        let mut stage = crate::session_state_tests::test_stage();
        let params = SamplingParams {
            max_tokens: 1,
            temperature: 0.0,
            ..Default::default()
        };
        let mut image_session = GenerationSession::new_with_embeddings(
            &stage,
            vec![1; 2],
            Tensor::zeros((2, 1), candle_core::DType::F32, &Device::Cpu)?,
            params.clone(),
            vec![],
        )?;
        image_session.step(&mut stage)?;
        let prompt = vec![1, 1, 0];
        let mut resumed = GenerationSession::with_cache(
            image_session.into_cache(),
            prompt.clone(),
            params.clone(),
            vec![],
            2,
        );
        resumed.enable_prompt_cache("owner".into());
        resumed.step(&mut stage)?;
        resumed.publish_prompt_cache(&mut stage);
        resumed.discard_cache(&mut stage);

        let mut text_session = GenerationSession::new(&stage, prompt, params, vec![]);
        text_session.enable_prompt_cache("owner".into());
        text_session.step(&mut stage)?;
        assert_eq!(text_session.reused_prompt_tokens(), 0);
        let Some(LayerCache::Attention(KvCache::F16 { k, .. })) = &text_session.cache.cache[0]
        else {
            anyhow::bail!("missing text attention cache");
        };
        assert_eq!(k.flatten_all()?.to_vec1::<f32>()?, vec![1.0, 1.0, 0.0]);
        text_session.discard_cache(&mut stage);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn bounded_prefill_resumes_an_existing_session_prefix() -> anyhow::Result<()> {
        let mut stage = crate::session_state_tests::test_stage();
        let mut cache = stage.new_session_state();
        stage.forward_with_session(&mut cache, StageInput::Tokens(vec![0; 10]), 10, 0)?;
        let mut session = GenerationSession::with_cache(
            cache,
            vec![0; 30],
            SamplingParams::default(),
            vec![],
            10,
        );
        assert!(session.step_with_prefill_budget(&mut stage, 8)?.is_none());
        assert_eq!(session.cache.next_position, 18);
        assert!(session.step_with_prefill_budget(&mut stage, 8)?.is_none());
        assert_eq!(session.cache.next_position, 26);
        assert!(session.step_with_prefill_budget(&mut stage, 8)?.is_some());
        assert_eq!(session.cache.next_position, 30);
        session.discard_cache(&mut stage);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn failed_prefill_chunk_does_not_publish_its_checkpoint() -> anyhow::Result<()> {
        let mut stage = crate::session_state_tests::test_stage();
        let mut prompt = vec![0; 80];
        prompt[64] = 255; // Test architecture injects a failure after writing its cache.
        let mut session = GenerationSession::new(&stage, prompt, SamplingParams::default(), vec![]);
        session.enable_prompt_cache("alice".into());
        session.set_shared_prompt_prefix(64);
        assert!(session.step_with_prefill_budget(&mut stage, 64)?.is_none());
        assert!(stage.prompt_cache_bytes() > 0);
        assert!(session.step_with_prefill_budget(&mut stage, 8).is_err());
        assert!(session.generated().is_empty());
        session.publish_prompt_cache(&mut stage);
        assert!(stage.reuse_prompt("alice", &vec![0; 64], true)?.is_none());
        session.discard_cache(&mut stage);
        assert_eq!(stage.prompt_cache_bytes(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn speed_fallback_requires_probes_and_a_full_window() {
        let mut monitor = SpeculativeSpeedMonitor::default();
        for _ in 0..8 {
            monitor.record_pass(3, Duration::from_millis(150));
        }
        assert!(!monitor.should_fallback());
        for _ in 0..2 {
            monitor.record_scalar(Duration::from_millis(40));
        }
        assert!(!monitor.should_fallback());
        monitor.record_scalar(Duration::ZERO);
        assert!(monitor.needs_probe());
        monitor.record_scalar(Duration::from_millis(40));
        assert!(monitor.should_fallback());

        let mut monitor = speed_monitor([40; 3]);
        for _ in 0..7 {
            monitor.record_pass(3, Duration::from_millis(150));
            assert!(!monitor.should_fallback());
        }
        monitor.record_pass(3, Duration::from_millis(150));
        assert!(monitor.should_fallback());
    }

    #[test]
    fn speed_fallback_uses_time_not_acceptance_and_ignores_one_probe_outlier() {
        let mut slow = speed_monitor([40, 400, 40]);
        let mut fast = speed_monitor([40; 3]);
        assert_eq!(slow.scalar_token_time(), Some(Duration::from_millis(40)));
        // Both average 3 committed tokens, above the old 2.5-token cutoff.
        for _ in 0..8 {
            slow.record_pass(3, Duration::from_millis(150));
            fast.record_pass(3, Duration::from_millis(90));
        }
        assert!(slow.should_fallback());
        assert!(!fast.should_fallback());
        for _ in 0..8 {
            slow.record_pass(3, Duration::from_millis(90));
        }
        assert!(!slow.should_fallback());
        assert_eq!(slow.recent.len(), 8);
    }

    #[test]
    fn speed_fallback_tolerates_small_timing_noise() {
        let mut monitor = speed_monitor([40; 3]);
        for _ in 0..8 {
            monitor.record_pass(2, Duration::from_millis(83));
        }
        assert!(!monitor.should_fallback());
    }

    #[test]
    fn speed_fallback_keeps_a_draft_with_a_net_time_saving() {
        let mut monitor = speed_monitor([40; 3]);
        for _ in 0..16 {
            monitor.record_pass(4, Duration::from_millis(80));
        }
        for _ in 0..8 {
            monitor.record_pass(1, Duration::from_millis(80));
        }
        assert!(!monitor.should_fallback());
        for _ in 0..40 {
            monitor.record_pass(1, Duration::from_millis(80));
        }
        assert!(monitor.should_fallback());
    }

    #[test]
    fn pre_m5_fallback_caps_speculation_after_one_pass() {
        let monitor = speed_monitor([40; 3]);
        assert_eq!(
            monitor.fallback_decision(true, 1, false, true),
            (true, false)
        );
        assert_eq!(
            monitor.fallback_decision(true, 0, false, true),
            (false, false)
        );
    }

    #[test]
    fn fallback_override_bypasses_pre_m5_hardware_limit() {
        let mut monitor = speed_monitor([40; 3]);
        for _ in 0..8 {
            monitor.record_pass(1, Duration::from_millis(150));
        }
        for speed_enabled in [true, false] {
            assert_eq!(
                monitor.fallback_decision(true, 8, false, speed_enabled),
                (true, true)
            );
            assert_eq!(
                monitor.fallback_decision(true, 8, true, speed_enabled),
                (false, false)
            );
            assert_eq!(
                monitor.fallback_decision(false, 8, true, speed_enabled),
                (false, false)
            );
        }
    }

    #[test]
    fn default_policy_keeps_existing_acceptance_thresholds() {
        for (tokens, passes, expected) in [
            (1, 7, false),
            (1, 8, true),
            (2, 31, false),
            (2, 32, true),
            (3, 32, false),
        ] {
            let mut monitor = SpeculativeSpeedMonitor::default();
            for _ in 0..passes {
                monitor.record_pass(tokens, Duration::from_millis(1));
            }
            assert_eq!(
                monitor.fallback_decision(false, passes, false, false),
                (false, expected)
            );
        }
    }

    #[test]
    fn greedy_speculation_shortcut_respects_all_logit_controls() {
        let base = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 1.0,
            ..Default::default()
        };
        assert!(GenerationSession::can_use_greedy_shortcut(&base));

        let frequency = SamplingParams {
            frequency_penalty: 0.1,
            ..base.clone()
        };
        assert!(!GenerationSession::can_use_greedy_shortcut(&frequency));

        let presence = SamplingParams {
            presence_penalty: 0.1,
            ..base.clone()
        };
        assert!(!GenerationSession::can_use_greedy_shortcut(&presence));

        let bias = SamplingParams {
            logit_bias: BTreeMap::from([(1, 1.0)]),
            ..base
        };
        assert!(!GenerationSession::can_use_greedy_shortcut(&bias));
    }

    fn check_greedy_verification(device: &Device) -> anyhow::Result<()> {
        let logits = Tensor::from_vec(vec![2.0f32, 2.0, 1.0, 1.0, 3.0, 3.0], (2, 3), device)?;
        assert_eq!(
            GenerationSession::greedy_verification_tokens(&logits)?,
            vec![0, 1]
        );
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let logits =
                Tensor::from_vec(vec![2.0f32, 1.0, 0.0, 1.0, 3.0, invalid], (2, 3), device)?;
            assert!(GenerationSession::greedy_verification_tokens(&logits).is_err());
        }
        Ok(())
    }

    #[test]
    fn greedy_verification_validates_every_row_on_cpu() -> anyhow::Result<()> {
        check_greedy_verification(&Device::Cpu)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn greedy_verification_validates_every_row_on_metal() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        check_greedy_verification(&device)
    }

    #[test]
    fn sampled_block_readback_preserves_visited_rows_and_rng() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let params = SamplingParams {
            temperature: 0.8,
            repetition_penalty: 1.2,
            frequency_penalty: 0.1,
            ..Default::default()
        };
        let values = vec![1.0f32, 1.2, -0.1, 0.8, -0.2, 1.3, f32::NAN, 0.0, 0.0];
        let logits = Tensor::from_vec(values, (3, 3), &device)?;
        let mut history = SamplingHistory::new(&[0, 1, 0], &params);
        let mut legacy_history = history.clone();
        let mut batched_rng = SamplingRng::seeded(42);
        let mut scalar_rng = SamplingRng::seeded(42);
        // Rejected/unused rows must neither be sampled nor draw RNG values.
        for (index, row) in logits.to_vec2::<f32>()?.into_iter().take(2).enumerate() {
            let expected = legacy_history.sample(&logits.get(index)?, &params, &mut scalar_rng)?;
            let actual = history.sample_host(row, &params, &mut batched_rng)?;
            assert_eq!(actual, expected);
            legacy_history.push(expected);
            history.push(actual);
        }
        for _ in 0..16 {
            assert_eq!(
                history.sample(&logits.get(0)?, &params, &mut batched_rng)?,
                legacy_history.sample(&logits.get(0)?, &params, &mut scalar_rng)?
            );
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "manual release-mode Metal sampling readback benchmark"]
    fn benchmark_speculative_sampling_readback() -> anyhow::Result<()> {
        use std::hint::black_box;
        use std::time::Instant;
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let vocab = 248_320;
        let logits = Tensor::from_vec(
            (0..8 * vocab)
                .map(|index| (index % 4093) as f32 / 4093.0)
                .collect::<Vec<_>>(),
            (8, vocab),
            &device,
        )?;
        let params = SamplingParams::default();
        let history = SamplingHistory::new(&[], &params);
        for visited in [1, 3, 8] {
            let measure = |batched: bool| -> anyhow::Result<_> {
                let mut rng = SamplingRng::seeded(42);
                device.synchronize()?;
                let started = Instant::now();
                for _ in 0..16 {
                    if batched {
                        for row in logits.to_vec2::<f32>()?.into_iter().take(visited) {
                            black_box(history.sample_host(row, &params, &mut rng)?);
                        }
                    } else {
                        for index in 0..visited {
                            black_box(history.sample(&logits.get(index)?, &params, &mut rng)?);
                        }
                    }
                }
                Ok(started.elapsed())
            };
            measure(false)?;
            measure(true)?;
            let mut scalar = Vec::new();
            let mut batch = Vec::new();
            for round in 0..5 {
                if round % 2 == 0 {
                    scalar.push(measure(false)?);
                    batch.push(measure(true)?);
                } else {
                    batch.push(measure(true)?);
                    scalar.push(measure(false)?);
                }
            }
            scalar.sort_unstable();
            batch.sort_unstable();
            eprintln!(
                "SPEC_READBACK_BENCH visited={visited} scalar_ms={:.3} batch_ms={:.3} speedup={:.2}x",
                scalar[2].as_secs_f64() * 1000.0 / 16.0,
                batch[2].as_secs_f64() * 1000.0 / 16.0,
                scalar[2].as_secs_f64() / batch[2].as_secs_f64()
            );
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn batched_metal_greedy_requires_valid_unpenalized_parameters() {
        let valid = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 1.0,
            ..Default::default()
        };
        assert!(GenerationSession::can_use_batched_metal_greedy(&valid));

        let mut invalid_top_p = valid.clone();
        invalid_top_p.top_p = f64::NAN;
        assert!(!GenerationSession::can_use_batched_metal_greedy(
            &invalid_top_p
        ));
        let mut invalid_temperature = valid.clone();
        invalid_temperature.temperature = f64::NEG_INFINITY;
        assert!(!GenerationSession::can_use_batched_metal_greedy(
            &invalid_temperature
        ));
        let mut penalized = valid.clone();
        penalized.presence_penalty = 0.25;
        assert!(!GenerationSession::can_use_batched_metal_greedy(&penalized));
        let mut sampled = valid;
        sampled.temperature = 0.8;
        assert!(!GenerationSession::can_use_batched_metal_greedy(&sampled));
    }

    #[test]
    fn borrowed_history_sampling_matches_seeded_reference() {
        let params = SamplingParams {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 4,
            repetition_penalty: 1.2,
            frequency_penalty: 0.15,
            presence_penalty: 0.25,
            logit_bias: BTreeMap::from([(4, 0.3)]),
            seed: Some(42),
            ..Default::default()
        };
        let history = [1, 1, 3, 5];
        let logits =
            Tensor::from_vec(vec![0.4f32, 1.2, -0.8, 0.7, 1.0, -0.1], 6, &Device::Cpu).unwrap();

        let adjusted = apply_sampling_penalties(&logits, &history, &params).unwrap();
        let mut expected_rng = SamplingRng::seeded(42);
        let expected = (0..8)
            .map(|_| sample_token_with_rng(&adjusted, &params, &mut expected_rng).unwrap())
            .collect::<Vec<_>>();

        let mut actual_rng = SamplingRng::seeded(42);
        let actual = (0..8)
            .map(|_| {
                SamplingHistory::new(&history, &params)
                    .sample(&logits, &params, &mut actual_rng)
                    .unwrap()
            })
            .collect::<Vec<_>>();

        assert_eq!(actual, expected);
    }

    #[test]
    fn cancelled_input_rejects_uncommitted_invalid_and_non_text_boundaries() -> anyhow::Result<()> {
        for case in [
            "before",
            "error",
            "cursor",
            "logical",
            "invalidated",
            "embedded",
            "mrope",
            "scope",
            "untrusted",
        ] {
            let mut stage = crate::session_state_tests::test_stage();
            let scope = "reserved:eligibility";
            assert!(stage.register_reserved_prompt_scope(scope.into()));
            let mut prompt = vec![0; 12];
            if case == "error" {
                prompt[8] = 255;
            }
            let mut session = GenerationSession::new(
                &stage,
                prompt,
                SamplingParams {
                    temperature: 0.,
                    seed: Some(42),
                    max_tokens: 3,
                    ..Default::default()
                },
                vec![],
            );
            session.enable_prompt_cache(scope.into());
            if case != "before" {
                assert!(session.step_with_prefill_budget(&mut stage, 8)?.is_none());
            }
            match case {
                "error" => assert!(session.step_with_prefill_budget(&mut stage, 2).is_err()),
                "cursor" => session.effective_skip -= 1,
                // Change only the committed marker, not charged KV ownership.
                "logical" => session.completed_input_cursor = Some(7),
                "invalidated" => session.cache.resume_invalidated = true,
                "embedded" => {
                    session.prompt_embeddings = Some(Tensor::zeros(
                        (12, 1),
                        candle_core::DType::F32,
                        &Device::Cpu,
                    )?)
                }
                "mrope" => session.mrope_decode_delta = Some(0),
                "scope" => session.prompt_cache_scope = Some("other".into()),
                "untrusted" => {
                    let cache = std::mem::replace(&mut session.cache, stage.new_session_state());
                    session = GenerationSession::with_cache(
                        cache,
                        vec![0; 12],
                        SamplingParams {
                            temperature: 0.,
                            seed: Some(42),
                            max_tokens: 3,
                            ..Default::default()
                        },
                        vec![],
                        8,
                    );
                    session.enable_prompt_cache(scope.into());
                    assert!(session.step_with_prefill_budget(&mut stage, 2)?.is_none());
                }
                _ => {}
            }
            assert!(
                !session.retain_cancelled_input_prefix(&mut stage, scope),
                "{case}"
            );
            session.publish_prompt_cache(&mut stage);
            assert_eq!(stage.prompt_cache_bytes(), 0, "{case}");
            assert_eq!(stage.total_kv_cache_allocation_tokens(), 0, "{case}");
        }
        Ok(())
    }

    #[test]
    fn cancelled_decode_preserves_copied_input_without_generated_tail() -> anyhow::Result<()> {
        for revoked in [false, true] {
            let mut stage = crate::session_state_tests::recurrent_stage(false);
            let scope = "reserved:cancelled-decode";
            assert!(stage.register_reserved_prompt_scope(scope.into()));
            let mut session = GenerationSession::new(
                &stage,
                vec![1, 1],
                SamplingParams {
                    temperature: 0.,
                    seed: Some(42),
                    max_tokens: 8,
                    ..Default::default()
                },
                vec![],
            );
            session.enable_prompt_cache(scope.into());
            for _ in 0..3 {
                session.step(&mut stage)?;
            }
            assert_eq!(session.generated().len(), 3);
            assert_eq!(stage.prompt_cache_match_tokens(scope, &[1, 1]), (0, 0));
            if revoked {
                stage.revoke_reserved_prompt_scope(scope);
            }
            assert_eq!(
                session.retain_cancelled_input_prefix(&mut stage, scope),
                !revoked
            );
            assert_eq!(
                stage.prompt_cache_match_tokens(scope, &[1, 1]),
                if revoked { (0, 0) } else { (0, 2) }
            );
            assert_eq!(
                stage.prompt_cache_match_tokens("reserved:other", &[1, 1]),
                (0, 0)
            );
            if !revoked {
                let (cache, logits, cursor) = stage.reuse_prompt(scope, &[1, 1], true)?.unwrap();
                assert_eq!(cursor, 2);
                assert_eq!(cache.next_position, 2);
                assert!(logits.is_some());
                drop(cache);
            }
            stage.revoke_reserved_prompt_scope(scope);
            assert_eq!(stage.prompt_cache_bytes(), 0);
            assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        }
        Ok(())
    }

    #[test]
    fn failed_decode_cannot_publish_previously_captured_input() -> anyhow::Result<()> {
        let mut stage = crate::session_state_tests::test_stage();
        let scope = "reserved:failed-after-capture";
        assert!(stage.register_reserved_prompt_scope(scope.into()));
        let mut session = GenerationSession::new(
            &stage,
            vec![0; 12],
            SamplingParams {
                temperature: 0.,
                seed: Some(42),
                max_tokens: 8,
                ..Default::default()
            },
            vec![],
        );
        session.enable_prompt_cache(scope.into());
        session.step(&mut stage)?;
        assert!(stage.prompt_cache_bytes() > 0);
        session.next_input_token = Some(255);
        assert!(session.step(&mut stage).is_err());
        assert!(!session.retain_cancelled_input_prefix(&mut stage, scope));
        session.publish_prompt_cache(&mut stage);
        assert_eq!(stage.prompt_cache_match_tokens(scope, &[0; 12]), (0, 0));
        stage.revoke_reserved_prompt_scope(scope);
        assert_eq!(stage.prompt_cache_bytes(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        Ok(())
    }

    #[test]
    fn cancelled_input_rejects_draft_owned_state_without_leaking() -> anyhow::Result<()> {
        let (mut stage, _) = crate::session_state_tests::speculative_boundary_stage(32, Some(32))?;
        let scope = "reserved:synthetic-draft";
        assert!(stage.register_reserved_prompt_scope(scope.into()));
        let mut session = GenerationSession::new(
            &stage,
            vec![0; 12],
            SamplingParams {
                seed: Some(42),
                max_tokens: 3,
                ..Default::default()
            },
            vec![],
        );
        session.enable_prompt_cache(scope.into());
        assert!(session.step_with_prefill_budget(&mut stage, 8)?.is_none());
        assert!(session.cache.speculative_cache.is_some());
        assert!(!session.retain_cancelled_input_prefix(&mut stage, scope));
        assert_eq!(stage.prompt_cache_bytes(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn cancelled_input_completion_retains_only_original_forwarded_ids() -> anyhow::Result<()> {
        use crate::architectures::StageOutput;
        let mut stage = crate::session_state_tests::recurrent_stage(false);
        let scope = "reserved:input-complete";
        assert!(stage.register_reserved_prompt_scope(scope.into()));
        let mut session = GenerationSession::new(
            &stage,
            vec![1, 1],
            SamplingParams {
                temperature: 0.,
                seed: Some(42),
                max_tokens: 3,
                ..Default::default()
            },
            vec![],
        );
        session.enable_prompt_cache(scope.into());
        session.step(&mut stage)?;
        assert_eq!(session.generated(), [0]);
        assert_eq!(session.next_position(&stage), 2);
        assert!(!session.is_finished());
        assert!(session.retain_cancelled_input_prefix(&mut stage, scope));
        assert!(!session.is_finished());
        assert_eq!(
            stage.prompt_cache_match_tokens(scope, &[1, 1, 0, 1]),
            (0, 2)
        );
        assert_eq!(
            stage.prompt_cache_match_tokens(scope, &[1, 0, 0, 1]),
            (0, 0)
        );
        let (mut resumed, logits, cursor) =
            stage.reuse_prompt(scope, &[1, 1, 0, 1], true)?.unwrap();
        assert!(logits.is_some());
        assert_eq!(cursor, 2);
        let StageOutput::Logits(warm) =
            stage.prefill_with_session(&mut resumed, StageInput::Tokens(vec![0, 1]), 2, cursor)?
        else {
            unreachable!()
        };
        let mut cold = stage.new_session_state();
        let StageOutput::Logits(reference) =
            stage.prefill_with_session(&mut cold, StageInput::Tokens(vec![1, 1, 0, 1]), 4, 0)?
        else {
            unreachable!()
        };
        let warm = warm.flatten_all()?.to_vec1::<f32>()?;
        assert!(warm.iter().all(|x| x.is_finite()));
        assert_eq!(warm, reference.flatten_all()?.to_vec1::<f32>()?);
        eprintln!(
            "cancelled input completion proof: input=2 sampled=0 cursor={cursor} warm_cold_logits={warm:?}"
        );
        drop((resumed, cold));
        stage.revoke_reserved_prompt_scope(scope);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn sparse_image_prompt_chunks_prefill_and_reuses_pre_image_text_cache() -> anyhow::Result<()> {
        let mut stage = crate::session_state_tests::test_stage();
        let scope = "user:multimodal-prefix";
        let prefix = vec![0u32; 6];
        let params = SamplingParams {
            temperature: 0.0,
            seed: Some(7),
            max_tokens: 3,
            ..Default::default()
        };

        let mut warm = GenerationSession::new(&stage, prefix.clone(), params.clone(), vec![]);
        warm.enable_prompt_cache(scope.into());
        while !warm.is_finished() {
            warm.step(&mut stage)?;
        }
        warm.publish_prompt_cache(&mut stage);
        warm.discard_cache(&mut stage);

        let image_rows = candle_core::Tensor::from_vec(
            vec![0.25f32; 2 * stage.hidden_dim()],
            (2, stage.hidden_dim()),
            stage.device(),
        )?;
        let mut tokens = prefix;
        tokens.extend_from_slice(&[0, 0, 1, 1]);
        let mut session = GenerationSession::new_with_image_spans(
            &stage,
            tokens.clone(),
            vec![ImageEmbeddingSpan {
                start: 6,
                embeddings: image_rows.clone(),
                bidirectional: false,
                digest: None,
            }],
            params.clone(),
            vec![],
        )?;
        session.enable_prompt_cache(scope.into());
        assert!(session.is_text_prefilling());

        // First budgeted step reuses the 6-token cached text prefix and processes
        // only the 2-token image chunk without sampling yet.
        assert!(session.step_with_prefill_budget(&mut stage, 2)?.is_none());
        assert_eq!(session.reused_prompt_tokens(), 6);
        assert_eq!(session.next_position(&stage), 8);
        assert!(session.prompt_image_spans.is_empty());

        // Second budgeted step processes the trailing 2 text tokens and samples.
        let first = session
            .step_with_prefill_budget(&mut stage, 2)?
            .expect("final prefill chunk should sample");
        let mut chunked_tokens = vec![first.token];
        while !session.is_finished() {
            chunked_tokens.push(session.step(&mut stage)?.token);
        }
        assert_eq!(session.terminal_cache_tokens(&stage), None);
        session.publish_prompt_cache(&mut stage);
        assert_eq!(stage.prompt_cache_match_tokens(scope, &tokens), (0, 6));
        session.discard_cache(&mut stage);

        // Compare against monolithic embeddings on a cold session.
        let full_embeddings = candle_core::Tensor::cat(
            &[
                &stage.embed_tokens(&tokens[..6])?,
                &image_rows,
                &stage.embed_tokens(&tokens[8..])?,
            ],
            0,
        )?;
        let mut monolithic = GenerationSession::new_with_embeddings(
            &stage,
            tokens,
            full_embeddings,
            params,
            vec![],
        )?;
        let mut monolithic_tokens = Vec::new();
        while !monolithic.is_finished() {
            monolithic_tokens.push(monolithic.step(&mut stage)?.token);
        }
        monolithic.discard_cache(&mut stage);
        assert_eq!(chunked_tokens, monolithic_tokens);
        Ok(())
    }

    #[test]
    fn sparse_bidirectional_image_span_never_splits_across_prefill_chunks() -> anyhow::Result<()> {
        let mut stage = crate::session_state_tests::test_stage();
        let params = SamplingParams {
            temperature: 0.0,
            seed: Some(11),
            max_tokens: 2,
            ..Default::default()
        };
        let image_rows = candle_core::Tensor::from_vec(
            vec![0.5f32; 6 * stage.hidden_dim()],
            (6, stage.hidden_dim()),
            stage.device(),
        )?;
        let tokens = vec![0u32; 14];
        let mut session = GenerationSession::new_with_image_spans(
            &stage,
            tokens,
            vec![ImageEmbeddingSpan {
                start: 5,
                embeddings: image_rows,
                bidirectional: true,
                digest: None,
            }],
            params,
            vec![],
        )?;

        // Budget 3: [0..3)
        assert!(session.step_with_prefill_budget(&mut stage, 3)?.is_none());
        assert_eq!(session.next_position(&stage), 3);

        // Budget 3: snaps to [3..5) so the bidirectional span [5..11) is not split.
        assert!(session.step_with_prefill_budget(&mut stage, 3)?.is_none());
        assert_eq!(session.next_position(&stage), 5);

        // Budget 3: starts at the bidirectional span, so processes all of [5..11) atomically.
        assert!(session.step_with_prefill_budget(&mut stage, 3)?.is_none());
        assert_eq!(session.next_position(&stage), 11);
        assert!(session.prompt_image_spans.is_empty());

        // Budget 3: [11..14) finishes prefill and samples.
        assert!(session.step_with_prefill_budget(&mut stage, 3)?.is_some());
        assert_eq!(session.next_position(&stage), 14);
        session.discard_cache(&mut stage);
        Ok(())
    }

    #[test]
    fn sparse_positioned_image_prompt_chunks_and_sets_mrope_decode_delta() -> anyhow::Result<()> {
        let mut stage = crate::session_state_tests::test_stage();
        let params = SamplingParams {
            temperature: 0.0,
            seed: Some(13),
            max_tokens: 2,
            ..Default::default()
        };
        let image_rows = candle_core::Tensor::from_vec(
            vec![0.1f32; 4 * stage.hidden_dim()],
            (4, stage.hidden_dim()),
            stage.device(),
        )?;
        let tokens = vec![0u32; 10];
        let positions = MropePositionIds::new(
            vec![0, 1, 2, 3, 4, 4, 4, 4, 6, 7],
            vec![0, 1, 2, 3, 4, 4, 5, 5, 6, 7],
            vec![0, 1, 2, 3, 4, 5, 4, 5, 6, 7],
        )?;
        let mut session = GenerationSession::new_with_positioned_image_spans(
            &stage,
            tokens,
            vec![ImageEmbeddingSpan {
                start: 4,
                embeddings: image_rows,
                bidirectional: false,
                digest: None,
            }],
            positions,
            params,
            vec![],
        )?;

        while session.is_text_prefilling() {
            let _ = session.step_with_prefill_budget(&mut stage, 3)?;
        }
        assert_eq!(session.cache.mrope_decode_delta, Some(-2));
        assert!(session.prompt_positions.is_none());
        assert_eq!(session.generated().len(), 1);
        session.step(&mut stage)?;
        assert_eq!(session.generated().len(), 2);
        session.discard_cache(&mut stage);
        Ok(())
    }

    #[test]
    fn sparse_image_prompt_reuses_draft_prefix_and_runs_speculative_decode() -> anyhow::Result<()> {
        let (mut stage, _) = crate::session_state_tests::speculative_boundary_stage(32, Some(32))?;
        let scope = "user:multimodal-draft";
        let prefix = vec![0u32, 0, 1, 1];
        let params = SamplingParams {
            temperature: 0.7,
            seed: None,
            max_tokens: 4,
            ..Default::default()
        };

        // Warm a draft-enabled text-only prefix into the prompt cache.
        let mut warm = GenerationSession::new(&stage, prefix.clone(), params.clone(), vec![]);
        warm.set_speculative_decoding_enabled(&stage, true);
        warm.enable_prompt_cache(scope.into());
        while !warm.is_finished() {
            warm.step(&mut stage)?;
        }
        assert!(warm.speculative_stats().passes > 0);
        warm.publish_prompt_cache(&mut stage);
        warm.discard_cache(&mut stage);

        // Multimodal session with plain 1D causal positions (e.g. Inkling 2):
        // reuses the pre-image draft prefix, injects image + trailing text into
        // the draft cache, and runs speculative decode after prefill.
        let image_rows = candle_core::Tensor::from_vec(
            vec![0.25f32; 2 * stage.hidden_dim()],
            (2, stage.hidden_dim()),
            stage.device(),
        )?;
        let mut tokens = prefix.clone();
        tokens.extend_from_slice(&[0, 0, 1, 1]);
        let mut session = GenerationSession::new_with_image_spans(
            &stage,
            tokens.clone(),
            vec![ImageEmbeddingSpan {
                start: 4,
                embeddings: image_rows.clone(),
                bidirectional: false,
                digest: None,
            }],
            params.clone(),
            vec![],
        )?;
        session.set_speculative_decoding_enabled(&stage, true);
        session.enable_prompt_cache(scope.into());

        // First chunk reuses the 4-token text prefix with draft cache intact,
        // then prefills the 2-token image span into both target and draft.
        assert!(session.step_with_prefill_budget(&mut stage, 2)?.is_none());
        assert_eq!(session.reused_prompt_tokens(), 4);
        assert_eq!(session.next_position(&stage), 6);
        assert!(stage.session_speculative_ready(&session.cache));
        assert_eq!(
            session
                .cache
                .speculative_cache
                .as_ref()
                .map(|cache| cache.next_position()),
            Some(6)
        );

        // Finish prefill and decode: DFlash/DSpark must run speculative passes.
        while !session.is_finished() {
            let _ = session.step_with_prefill_budget(&mut stage, 2)?;
        }
        assert!(
            session.speculative_stats().passes > 0,
            "expected speculative passes after multimodal prefill"
        );
        assert!(session.speculative_stats().committed_tokens > 0);
        // Multimodal session still must not publish to the text-only prompt cache.
        assert_eq!(session.terminal_cache_tokens(&stage), None);
        session.publish_prompt_cache(&mut stage);
        assert_eq!(stage.prompt_cache_match_tokens(scope, &tokens), (0, 4));
        session.discard_cache(&mut stage);

        // Positioned M-RoPE multimodal sessions (e.g. Qwen3.5) reuse the
        // pre-image text prefix and resume speculative decoding after the
        // image, verifying at text positions shifted by the decode delta.
        let positions = MropePositionIds::new(
            vec![0, 1, 2, 3, 4, 4, 5, 6],
            vec![0, 1, 2, 3, 4, 4, 5, 6],
            vec![0, 1, 2, 3, 4, 5, 5, 6],
        )?;
        let mut mrope = GenerationSession::new_with_positioned_image_spans(
            &stage,
            tokens,
            vec![ImageEmbeddingSpan {
                start: 4,
                embeddings: image_rows.clone(),
                bidirectional: false,
                digest: None,
            }],
            positions,
            params.clone(),
            vec![],
        )?;
        mrope.set_speculative_decoding_enabled(&stage, true);
        mrope.enable_prompt_cache(scope.into());
        while !mrope.is_finished() {
            let _ = mrope.step_with_prefill_budget(&mut stage, 2)?;
        }
        assert_eq!(mrope.reused_prompt_tokens(), 4);
        assert!(mrope.speculative_stats().passes > 0);
        mrope.discard_cache(&mut stage);
        stage.clear_prompt_cache();
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);

        // If only a non-draft text prefix is cached, a draft-enabled multimodal
        // session skips it rather than dropping its speculative cache.
        let mut scalar_warm = GenerationSession::new(&stage, prefix, params.clone(), vec![]);
        scalar_warm.set_speculative_decoding_enabled(&stage, false);
        scalar_warm.enable_prompt_cache(scope.into());
        while !scalar_warm.is_finished() {
            scalar_warm.step(&mut stage)?;
        }
        assert_eq!(scalar_warm.speculative_stats().passes, 0);
        scalar_warm.publish_prompt_cache(&mut stage);
        scalar_warm.discard_cache(&mut stage);

        let mut cold_draft = GenerationSession::new_with_image_spans(
            &stage,
            vec![0u32, 0, 1, 1, 0, 0, 1, 1],
            vec![ImageEmbeddingSpan {
                start: 4,
                embeddings: image_rows,
                bidirectional: false,
                digest: None,
            }],
            params,
            vec![],
        )?;
        cold_draft.set_speculative_decoding_enabled(&stage, true);
        cold_draft.enable_prompt_cache(scope.into());
        while !cold_draft.is_finished() {
            let _ = cold_draft.step_with_prefill_budget(&mut stage, 4)?;
        }
        assert_eq!(cold_draft.reused_prompt_tokens(), 0);
        assert!(cold_draft.speculative_stats().passes > 0);
        cold_draft.discard_cache(&mut stage);
        stage.clear_prompt_cache();
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn multimodal_prompt_cache_reuses_images_and_rejects_mismatched_digest() -> anyhow::Result<()> {
        let (mut stage, _) = crate::session_state_tests::speculative_boundary_stage(64, Some(64))?;
        let scope = "multimodal-digest-cache";
        let params = SamplingParams {
            temperature: 0.7,
            max_tokens: 4,
            ..Default::default()
        };
        let img1 = candle_core::Tensor::zeros(
            (2, 64),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )?;
        let img2 =
            candle_core::Tensor::ones((2, 64), candle_core::DType::F32, &candle_core::Device::Cpu)?;
        let digest1 = [0x11u8; 32];
        let digest2 = [0x22u8; 32];
        let digest_other = [0x99u8; 32];

        // Turn 1: [0,0,1,1, IMG1(2), 1,1] (8 tokens), publishes image-keyed snapshot.
        let mut turn1 = GenerationSession::new_with_image_spans(
            &stage,
            vec![0u32, 0, 1, 1, 0, 0, 1, 1],
            vec![ImageEmbeddingSpan {
                start: 4,
                embeddings: img1.clone(),
                bidirectional: false,
                digest: Some(digest1),
            }],
            params.clone(),
            vec![],
        )?;
        turn1.set_speculative_decoding_enabled(&stage, true);
        turn1.enable_prompt_cache(scope.into());
        while !turn1.is_finished() {
            let _ = turn1.step_with_prefill_budget(&mut stage, 4)?;
        }
        assert_eq!(turn1.reused_prompt_tokens(), 0);
        assert!(turn1.speculative_stats().passes > 0);
        turn1.publish_prompt_cache(&mut stage);
        turn1.discard_cache(&mut stage);

        // Turn 2: appends IMG2 + suffix after Turn 1's 8 tokens -> reuses all 8 tokens!
        let mut turn2 = GenerationSession::new_with_image_spans(
            &stage,
            vec![0u32, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            vec![
                ImageEmbeddingSpan {
                    start: 4,
                    embeddings: img1.clone(),
                    bidirectional: false,
                    digest: Some(digest1),
                },
                ImageEmbeddingSpan {
                    start: 8,
                    embeddings: img2.clone(),
                    bidirectional: false,
                    digest: Some(digest2),
                },
            ],
            params.clone(),
            vec![],
        )?;
        turn2.set_speculative_decoding_enabled(&stage, true);
        turn2.enable_prompt_cache(scope.into());
        while !turn2.is_finished() {
            let _ = turn2.step_with_prefill_budget(&mut stage, 4)?;
        }
        assert_eq!(turn2.reused_prompt_tokens(), 8);
        assert!(turn2.speculative_stats().passes > 0);
        turn2.discard_cache(&mut stage);

        // Mismatched image digest at start=4 must NOT reuse Turn 1's KV state.
        let mut mismatch = GenerationSession::new_with_image_spans(
            &stage,
            vec![0u32, 0, 1, 1, 0, 0, 1, 1, 1],
            vec![ImageEmbeddingSpan {
                start: 4,
                embeddings: img1,
                bidirectional: false,
                digest: Some(digest_other),
            }],
            params.clone(),
            vec![],
        )?;
        mismatch.set_speculative_decoding_enabled(&stage, true);
        mismatch.enable_prompt_cache(scope.into());
        while !mismatch.is_finished() {
            let _ = mismatch.step_with_prefill_budget(&mut stage, 4)?;
        }
        assert_eq!(mismatch.reused_prompt_tokens(), 0);
        mismatch.discard_cache(&mut stage);

        // Plain-text prompt with identical token IDs must NOT reuse Turn 1's multimodal KV state.
        let mut plain =
            GenerationSession::new(&stage, vec![0u32, 0, 1, 1, 0, 0, 1, 1, 1], params, vec![]);
        plain.set_speculative_decoding_enabled(&stage, true);
        plain.enable_prompt_cache(scope.into());
        while !plain.is_finished() {
            let _ = plain.step_with_prefill_budget(&mut stage, 4)?;
        }
        assert_eq!(plain.reused_prompt_tokens(), 0);
        plain.discard_cache(&mut stage);
        stage.clear_prompt_cache();
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }
}
