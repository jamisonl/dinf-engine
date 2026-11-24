use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use candle_core::{DType, Tensor};

/// Sampling parameters for token generation.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Softmax temperature. Values less than or equal to zero select the
    /// greedy argmax path; positive values must be finite.
    pub temperature: f64,
    /// Nucleus-sampling probability cutoff in the exclusive/inclusive range
    /// `(0, 1]`. A value of `1.0` disables nucleus truncation.
    pub top_p: f64,
    /// Maximum number of candidates retained before top-p filtering. Zero
    /// means that the full vocabulary is considered.
    pub top_k: usize,
    /// Multiplicative penalty applied once to each previously seen token.
    /// Must stay finite and positive when converted to F32. `1.0` disables
    /// the penalty.
    pub repetition_penalty: f64,
    /// Additive penalty applied per occurrence of a previously seen token.
    /// Valid values are in `[-2, 2]`.
    pub frequency_penalty: f64,
    /// Additive penalty applied once to each previously seen token. Valid
    /// values are in `[-2, 2]`.
    pub presence_penalty: f64,
    /// Additive per-token logit adjustments. Keys must be in the model
    /// vocabulary at sampling time; values are bounded to `[-100, 100]` and
    /// at most 4096 entries are accepted.
    pub logit_bias: BTreeMap<u32, f32>,
    /// Request-owned deterministic sampling seed. `None` uses an entropy-like
    /// process-local seed.
    pub seed: Option<u64>,
    /// Maximum number of tokens to generate. Must be positive.
    pub max_tokens: usize,
    /// Minimum number of generated tokens before a stop token may terminate
    /// generation. Must not exceed [`SamplingParams::max_tokens`].
    pub min_tokens: usize,
    /// Request-owned tool grammar. Generation must retain one SamplingHistory
    /// so the matcher advances once for each token selected by the model.
    pub tool_constraint: Option<Arc<crate::tool_constraints::ToolConstraint>>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            logit_bias: BTreeMap::new(),
            seed: None,
            max_tokens: 2048,
            min_tokens: 0,
            tool_constraint: None,
        }
    }
}

const MAX_LOGIT_BIAS_ENTRIES: usize = 4_096;

fn validate_repetition_penalty(penalty: f64) -> Result<()> {
    let penalty_f32 = penalty as f32;
    anyhow::ensure!(
        penalty.is_finite() && penalty_f32.is_finite() && penalty_f32 > 0.0,
        "repetition penalty must stay finite and positive at single precision"
    );
    Ok(())
}

impl SamplingParams {
    /// Validate values used by the sampling and generation APIs.
    ///
    /// Temperature values at or below zero intentionally select greedy
    /// decoding. All other floating-point parameters must be finite, and
    /// `top_p`, penalties, token limits, and logit-bias entries must satisfy
    /// the bounds documented on their fields.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.temperature.is_finite(), "temperature must be finite");
        anyhow::ensure!(
            self.top_p.is_finite() && self.top_p > 0.0 && self.top_p <= 1.0,
            "top_p must be finite and between 0 (exclusive) and 1"
        );
        self.validate_penalties()?;
        anyhow::ensure!(self.max_tokens > 0, "max_tokens must be positive");
        anyhow::ensure!(
            self.min_tokens <= self.max_tokens,
            "min_tokens must not exceed max_tokens"
        );
        Ok(())
    }

    fn validate_penalties(&self) -> Result<()> {
        validate_repetition_penalty(self.repetition_penalty)?;
        anyhow::ensure!(
            self.frequency_penalty.is_finite() && (-2.0..=2.0).contains(&self.frequency_penalty),
            "frequency penalty must be finite and between -2 and 2"
        );
        anyhow::ensure!(
            self.presence_penalty.is_finite() && (-2.0..=2.0).contains(&self.presence_penalty),
            "presence penalty must be finite and between -2 and 2"
        );
        anyhow::ensure!(
            self.logit_bias.len() <= MAX_LOGIT_BIAS_ENTRIES,
            "logit bias must not contain more than {MAX_LOGIT_BIAS_ENTRIES} entries"
        );
        anyhow::ensure!(
            self.logit_bias
                .values()
                .all(|bias| bias.is_finite() && bias.abs() <= 100.0),
            "logit bias values must be finite and between -100 and 100"
        );
        Ok(())
    }

    pub(crate) fn has_penalty_controls(&self) -> bool {
        self.has_history_penalties() || !self.logit_bias.is_empty()
    }

    fn has_history_penalties(&self) -> bool {
        self.repetition_penalty != 1.0
            || self.frequency_penalty != 0.0
            || self.presence_penalty != 0.0
    }
}

/// Request-owned token counts for repetition, presence and frequency penalties.
/// Build once from the prompt and record each emitted token with [`Self::push`].
/// Parameters must retain the same penalty controls and tool constraint for the
/// lifetime of this history. With history penalties disabled, this allocates no
/// count map.
#[derive(Clone)]
pub struct SamplingHistory {
    counts: Option<HashMap<u32, usize>>,
    tool_constraint: Option<Arc<crate::tool_constraints::ToolConstraint>>,
    tool_matcher: Option<RefCell<llguidance::Matcher>>,
}

impl std::fmt::Debug for SamplingHistory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SamplingHistory")
            .field("counts", &self.counts)
            .field("tool_constraint", &self.tool_matcher.is_some())
            .finish()
    }
}

impl SamplingHistory {
    /// Count prompt tokens only when a history-dependent penalty needs them.
    pub fn new(tokens: &[u32], params: &SamplingParams) -> Self {
        let mut history = Self {
            counts: params.has_history_penalties().then(HashMap::new),
            tool_constraint: params.tool_constraint.clone(),
            tool_matcher: params
                .tool_constraint
                .as_ref()
                .map(|constraint| RefCell::new(constraint.matcher())),
        };
        if history.counts.is_some() {
            for &token in tokens {
                history.push(token);
            }
        }
        history
    }

    /// Take the tool-grammar state from a tentative copy that sampled ahead,
    /// such as a committed speculative verification. Token counts stay with
    /// `self`; callers record those with [`Self::push`] as tokens are emitted.
    pub(crate) fn adopt_tool_state(&mut self, tentative: SamplingHistory) {
        debug_assert_eq!(
            self.tool_matcher.is_some(),
            tentative.tool_matcher.is_some(),
            "tool constraint changed between sampling histories"
        );
        self.tool_matcher = tentative.tool_matcher;
    }

    /// Record one emitted token. Disabled histories do no work.
    pub fn push(&mut self, token: u32) {
        if let Some(counts) = &mut self.counts {
            *counts.entry(token).or_default() += 1;
        }
    }

    /// Sample with cached counts and one logits readback, preserving the
    /// candidate order and RNG draws of the slice-based sampling API.
    pub fn sample(
        &self,
        logits: &Tensor,
        params: &SamplingParams,
        rng: &mut SamplingRng,
    ) -> Result<u32> {
        self.validate_tool_constraint(params, logits.elem_count())?;
        if !params.has_penalty_controls() && self.tool_matcher.is_none() {
            return sample_token_with_rng(logits, params, rng);
        }
        params.validate()?;
        self.sample_penalized_validated(logits, params, rng)
    }

    fn validate_tool_constraint(&self, params: &SamplingParams, vocab_size: usize) -> Result<()> {
        let unchanged = match (&self.tool_constraint, &params.tool_constraint) {
            (None, None) => true,
            (Some(original), Some(current)) => Arc::ptr_eq(original, current),
            _ => false,
        };
        anyhow::ensure!(
            unchanged,
            "tool constraint cannot change after sampling history was created"
        );
        if let Some(constraint) = &self.tool_constraint {
            constraint.validate_vocab_size(vocab_size)?;
        }
        Ok(())
    }

    fn sample_penalized_validated(
        &self,
        logits: &Tensor,
        params: &SamplingParams,
        rng: &mut SamplingRng,
    ) -> Result<u32> {
        let logits = logits.to_dtype(DType::F32)?.flatten_all()?;
        self.sample_host_validated(logits.to_vec1::<f32>()?, params, rng)
    }

    pub(crate) fn sample_host(
        &self,
        logits: Vec<f32>,
        params: &SamplingParams,
        rng: &mut SamplingRng,
    ) -> Result<u32> {
        params.validate()?;
        self.sample_host_validated(logits, params, rng)
    }

    fn sample_host_validated(
        &self,
        mut logits: Vec<f32>,
        params: &SamplingParams,
        rng: &mut SamplingRng,
    ) -> Result<u32> {
        self.validate_tool_constraint(params, logits.len())?;
        anyhow::ensure!(
            !params.has_history_penalties() || self.counts.is_some(),
            "history penalties cannot be enabled after sampling history was created"
        );
        anyhow::ensure!(!logits.is_empty(), "logits must contain at least one value");
        apply_sampling_penalties_with_counts(&mut logits, self.counts.as_ref(), params)?;
        if let Some(matcher) = &self.tool_matcher {
            anyhow::ensure!(
                logits.iter().all(|logit| logit.is_finite()),
                "logits must be finite"
            );
            let mut matcher = matcher.borrow_mut();
            let mask = crate::tool_constraints::mask(&mut matcher)?;
            anyhow::ensure!(
                mask.len() == logits.len(),
                "tool constraint vocabulary has {} tokens, logits vocabulary has {}",
                mask.len(),
                logits.len()
            );
            let candidates = logits
                .into_iter()
                .enumerate()
                .filter(|(id, _)| mask.is_allowed(*id as u32))
                .collect::<Vec<_>>();
            anyhow::ensure!(!candidates.is_empty(), "tool grammar permits no token");
            let token = if params.temperature <= 0.0 {
                let mut best = candidates[0];
                for candidate in &candidates[1..] {
                    if candidate.1 > best.1 {
                        best = *candidate;
                    }
                }
                u32::try_from(best.0)?
            } else {
                sample_positive_candidates(candidates, params, rng)?
            };
            crate::tool_constraints::commit(&mut matcher, token)?;
            return Ok(token);
        }
        if params.temperature <= 0.0 {
            checked_first_argmax(&logits)
        } else {
            sample_positive_from_logits_vec(logits, params, rng)
        }
    }
}

/// Apply repetition penalty to logits for previously generated tokens.
pub fn apply_repetition_penalty(
    logits: &Tensor,
    prev_tokens: &[u32],
    penalty: f64,
) -> Result<Tensor> {
    validate_repetition_penalty(penalty)?;
    if penalty == 1.0 || prev_tokens.is_empty() {
        return Ok(logits.clone());
    }
    let device = logits.device();
    let mut logits_vec = logits.to_vec1::<f32>()?;

    // Repetition penalty is presence-based: applying it once per occurrence
    // exponentially over-penalizes tokens that happened to repeat.
    let mut seen = HashMap::with_capacity(prev_tokens.len());
    for &token_id in prev_tokens {
        if seen.insert(token_id, ()).is_some() {
            continue;
        }
        let idx = token_id as usize;
        if idx < logits_vec.len() {
            if logits_vec[idx] > 0.0 {
                logits_vec[idx] /= penalty as f32;
            } else {
                logits_vec[idx] *= penalty as f32;
            }
        }
    }

    Tensor::from_vec(logits_vec, logits.shape(), device).map_err(Into::into)
}

/// Apply all request-scoped logit transformations before temperature and
/// top-k/top-p filtering. Repetition penalty is retained as a multiplicative
/// legacy control; OpenAI's presence/frequency penalties are additive.
pub fn apply_sampling_penalties(
    logits: &Tensor,
    prev_tokens: &[u32],
    params: &SamplingParams,
) -> Result<Tensor> {
    params.validate_penalties()?;
    if params.repetition_penalty == 1.0
        && params.frequency_penalty == 0.0
        && params.presence_penalty == 0.0
        && params.logit_bias.is_empty()
    {
        return Ok(logits.clone());
    }
    let device = logits.device();
    let mut logits_vec = logits.to_vec1::<f32>()?;
    apply_sampling_penalties_to_vec(&mut logits_vec, prev_tokens, params)?;

    Tensor::from_vec(logits_vec, logits.shape(), device).map_err(Into::into)
}

fn apply_sampling_penalties_to_vec(
    logits_vec: &mut [f32],
    prev_tokens: &[u32],
    params: &SamplingParams,
) -> Result<()> {
    let history = SamplingHistory::new(prev_tokens, params);
    apply_sampling_penalties_with_counts(logits_vec, history.counts.as_ref(), params)
}

fn apply_sampling_penalties_with_counts(
    logits_vec: &mut [f32],
    counts: Option<&HashMap<u32, usize>>,
    params: &SamplingParams,
) -> Result<()> {
    for &token_id in params.logit_bias.keys() {
        anyhow::ensure!(
            (token_id as usize) < logits_vec.len(),
            "logit bias token id {token_id} is outside the model vocabulary"
        );
    }
    for (&token_id, count) in counts.into_iter().flatten() {
        let idx = token_id as usize;
        let Some(logit) = logits_vec.get_mut(idx) else {
            continue;
        };
        if params.repetition_penalty != 1.0 {
            if *logit > 0.0 {
                *logit /= params.repetition_penalty as f32;
            } else {
                *logit *= params.repetition_penalty as f32;
            }
        }
        *logit -= params.presence_penalty as f32;
        *logit -= params.frequency_penalty as f32 * (*count as f32);
    }
    for (&token_id, &bias) in &params.logit_bias {
        if let Some(logit) = logits_vec.get_mut(token_id as usize) {
            *logit += bias;
        }
    }
    Ok(())
}

/// Sample a token from logits using temperature, top-k, and top-p (nucleus) sampling.
pub fn sample_token(logits: &Tensor, params: &SamplingParams) -> Result<u32> {
    thread_local! {
        static RNG: std::cell::RefCell<SamplingRng> =
            std::cell::RefCell::new(SamplingRng::from_entropy());
    }
    RNG.with(|rng| sample_token_with_rng(logits, params, &mut rng.borrow_mut()))
}

/// Small, request-owned pseudo-random generator for reproducible sampling.
/// Callers that model a generation session should retain one instance across
/// every token instead of reseeding on each draw.
#[derive(Debug, Clone)]
pub struct SamplingRng {
    state: u64,
}

impl SamplingRng {
    /// Create a deterministic generator from `seed`.
    ///
    /// This generator is intended for reproducible sampling, not for
    /// cryptographic use. A seed of zero is mapped to a nonzero internal
    /// state so that the xorshift sequence cannot become permanently zero.
    pub fn seeded(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    /// Create a process-local generator seeded from a counter, current time,
    /// and process ID. Use [`SamplingRng::seeded`] when reproducibility is
    /// required; this source is not intended as cryptographic entropy.
    pub fn from_entropy() -> Self {
        static NEXT_SEED: AtomicU64 = AtomicU64::new(0x243f_6a88_85a3_08d3);
        let counter = NEXT_SEED.fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed);
        let time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos() as u64);
        Self::seeded(counter ^ time.rotate_left(17) ^ u64::from(std::process::id()))
    }

    pub(crate) fn next_f32(&mut self) -> f32 {
        // xorshift64*: fast and sufficient for non-cryptographic token sampling.
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        let value = x.wrapping_mul(0x2545_f491_4f6c_dd1d);
        ((value >> 40) as f32) / ((1u32 << 24) as f32)
    }
}

/// Sample using an explicit request-owned RNG.
pub fn sample_token_with_rng(
    logits: &Tensor,
    params: &SamplingParams,
    rng: &mut SamplingRng,
) -> Result<u32> {
    anyhow::ensure!(
        params.tool_constraint.is_none(),
        "tool constraints require a persistent SamplingHistory"
    );
    // Validate request data before touching the logits device. This keeps
    // malformed requests deterministic and avoids doing potentially expensive
    // accelerator readback for parameters that can never be used.
    params.validate()?;
    let logits = logits.to_dtype(DType::F32)?;
    let logits = logits.flatten_all()?;
    anyhow::ensure!(
        logits.elem_count() > 0,
        "logits must contain at least one value"
    );

    // Keep greedy decoding on-device. Returning the argmax already requires a
    // scalar synchronization; copying the entire vocabulary first is very
    // expensive for large vocabularies. The self-difference reduction is zero
    // exactly for finite values and NaN for NaN or either infinity, so it
    // validates the complete tensor with one scalar readback.
    if params.temperature <= 0.0 {
        #[cfg(target_os = "macos")]
        if logits.device().is_metal() && logits.is_contiguous() {
            return crate::sampling_metal::greedy_argmax(&logits);
        }
        #[cfg(feature = "cuda")]
        if let Some(token) = crate::sampling_cuda::try_greedy_argmax(&logits)? {
            return Ok(token);
        }
        let finite_witness = logits.sub(&logits)?.sum_all()?.to_scalar::<f32>()?;
        anyhow::ensure!(finite_witness == 0.0, "logits must be finite");
        let token = logits.argmax(0)?.to_scalar::<u32>()?;
        let selected = logits.get(token as usize)?.to_scalar::<f32>()?;
        anyhow::ensure!(selected.is_finite(), "greedy argmax logit must be finite");
        return Ok(token);
    }

    let logits_vec = logits.to_vec1::<f32>()?;
    sample_positive_from_logits_vec(logits_vec, params, rng)
}

/// Apply request-scoped penalties and sample directly from one logits
/// readback.
///
/// The usual two-step sequence (`apply_sampling_penalties` followed by
/// `sample_token_with_rng`) downloads the full vocabulary twice on an
/// accelerator: once to apply penalties and again after rebuilding the
/// adjusted tensor. This helper keeps the adjusted values in the first host
/// buffer and feeds that buffer directly into the unchanged sampler, so
/// non-greedy penalized decoding performs only one full-vocabulary readback.
/// The seeded RNG sequence and candidate ordering are identical to the
/// two-step API. Penalized greedy decoding also reuses that host buffer for a
/// checked first-index argmax instead of uploading the adjusted vocabulary;
/// unpenalized greedy decoding retains the fused on-device reduction.
pub fn sample_token_with_penalties_with_rng(
    logits: &Tensor,
    prev_tokens: &[u32],
    params: &SamplingParams,
    rng: &mut SamplingRng,
) -> Result<u32> {
    anyhow::ensure!(
        params.tool_constraint.is_none(),
        "tool constraints require a persistent SamplingHistory"
    );
    if !params.has_penalty_controls() {
        return sample_token_with_rng(logits, params, rng);
    }
    params.validate()?;
    SamplingHistory::new(prev_tokens, params).sample_penalized_validated(logits, params, rng)
}

/// Return the first maximum from a finite host-side logits buffer.
///
/// Candle and the fused Metal greedy reduction both resolve equal maxima to
/// the lowest token index. Validate every value before returning so a
/// non-finite unselected logit cannot be silently ignored.
pub(crate) fn checked_first_argmax(logits: &[f32]) -> Result<u32> {
    anyhow::ensure!(!logits.is_empty(), "logits must contain at least one value");
    let mut best_index = 0usize;
    let mut best_logit = f32::NEG_INFINITY;
    for (index, &logit) in logits.iter().enumerate() {
        anyhow::ensure!(logit.is_finite(), "logits must be finite");
        if logit > best_logit {
            best_logit = logit;
            best_index = index;
        }
    }
    u32::try_from(best_index).map_err(|_| anyhow::anyhow!("argmax token index exceeds u32"))
}

fn sample_positive_from_logits_vec(
    mut logits_vec: Vec<f32>,
    params: &SamplingParams,
    rng: &mut SamplingRng,
) -> Result<u32> {
    let mut max_logit = f32::NEG_INFINITY;
    for &logit in &logits_vec {
        anyhow::ensure!(logit.is_finite(), "logits must be finite");
        if logit > max_logit {
            max_logit = logit;
        }
    }
    // Fast path for large vocabularies when top_k is disabled: sort only the
    // high-probability head instead of O(V log V) over 200K tokens.
    if params.top_k == 0 && logits_vec.len() > 2048 && params.temperature > 1e-6 {
        let inv_t = 1.0 / params.temperature;
        let head_span = (20.0 * params.temperature) as f32;
        let head_cutoff = max_logit - head_span;
        let zero_cutoff = max_logit - (40.0 * params.temperature) as f32;
        let mut head: Vec<(usize, f32)> = Vec::with_capacity(256);
        let mut tail_sum = 0.0f64;
        for (id, &logit) in logits_vec.iter().enumerate() {
            if logit >= head_cutoff {
                head.push((id, logit));
            } else if logit >= zero_cutoff {
                tail_sum += ((f64::from(logit) - f64::from(max_logit)) * inv_t).exp();
            }
        }
        if !head.is_empty() && head.len() <= logits_vec.len() / 4 {
            head.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            let mut probs: Vec<f64> = head
                .iter()
                .map(|(_, logit)| ((f64::from(*logit) - f64::from(max_logit)) * inv_t).exp())
                .collect();
            let head_sum: f64 = probs.iter().sum();
            let total_sum = head_sum + tail_sum;
            if total_sum > 0.0 {
                if params.top_p < 1.0 {
                    if head_sum / total_sum >= params.top_p {
                        for p in &mut probs {
                            *p /= total_sum;
                        }
                        let mut cumulative = 0.0f64;
                        let mut cutoff_idx = probs.len();
                        for (i, &prob) in probs.iter().enumerate() {
                            cumulative += prob;
                            if cumulative >= params.top_p {
                                cutoff_idx = i + 1;
                                break;
                            }
                        }
                        probs.truncate(cutoff_idx);
                        head.truncate(cutoff_idx);
                        let kept_sum: f64 = probs.iter().sum();
                        if kept_sum > 0.0 {
                            for p in &mut probs {
                                *p /= kept_sum;
                            }
                        }
                        let r = f64::from(rng.next_f32());
                        let mut cumulative = 0.0f64;
                        for ((token_id, _), &p) in head.iter().zip(&probs) {
                            cumulative += p;
                            if r < cumulative {
                                return Ok(*token_id as u32);
                            }
                        }
                        if let Some(((id, _), _)) =
                            head.iter().zip(&probs).rfind(|(_, p)| **p > 0.0)
                        {
                            return Ok(*id as u32);
                        }
                    }
                } else {
                    let r = f64::from(rng.next_f32());
                    let target = r * total_sum;
                    if target < head_sum {
                        let mut cumulative = 0.0f64;
                        for ((token_id, _), &p) in head.iter().zip(&probs) {
                            cumulative += p;
                            if target < cumulative {
                                return Ok(*token_id as u32);
                            }
                        }
                        if let Some(((id, _), _)) =
                            head.iter().zip(&probs).rfind(|(_, p)| **p > 0.0)
                        {
                            return Ok(*id as u32);
                        }
                    }
                    // Rare tail draw: consume no extra RNG and finish over the sorted tail.
                    let mut tail: Vec<(usize, f32)> = logits_vec
                        .drain(..)
                        .enumerate()
                        .filter(|(_, logit)| *logit < head_cutoff)
                        .collect();
                    tail.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                    let mut cumulative = head_sum;
                    for (token_id, logit) in &tail {
                        let p = ((f64::from(*logit) - f64::from(max_logit)) * inv_t).exp();
                        cumulative += p;
                        if target < cumulative {
                            return Ok(*token_id as u32);
                        }
                    }
                    return Ok(tail.first().map_or(head[0].0, |(id, _)| *id) as u32);
                }
            }
        }
    }
    let candidates: Vec<(usize, f32)> = logits_vec.drain(..).enumerate().collect();
    sample_positive_candidates(candidates, params, rng)
}

/// Compute the normalized target distribution `(token_id, prob)` for positive
/// temperature sampling (`top_k`, `top_p`, `temperature`).
fn positive_target_distribution(
    logits: &[f32],
    params: &SamplingParams,
) -> Result<Vec<(u32, f64)>> {
    anyhow::ensure!(!logits.is_empty(), "logits must contain at least one value");
    let mut max_logit = f32::NEG_INFINITY;
    for &logit in logits {
        anyhow::ensure!(logit.is_finite(), "logits must be finite");
        if logit > max_logit {
            max_logit = logit;
        }
    }
    let inv_t = 1.0 / params.temperature;
    // Fast head-filtered path for large vocabularies when top_k == 0 and top_p < 1.0.
    if params.top_k == 0
        && params.top_p < 1.0
        && logits.len() > 2048
        && params.temperature > 1e-6
    {
        let head_cutoff = max_logit - (20.0 * params.temperature) as f32;
        let zero_cutoff = max_logit - (40.0 * params.temperature) as f32;
        let mut head: Vec<(usize, f32)> = Vec::with_capacity(256);
        let mut tail_sum = 0.0f64;
        for (id, &logit) in logits.iter().enumerate() {
            if logit >= head_cutoff {
                head.push((id, logit));
            } else if logit >= zero_cutoff {
                tail_sum += ((f64::from(logit) - f64::from(max_logit)) * inv_t).exp();
            }
        }
        if !head.is_empty() && head.len() <= logits.len() / 4 {
            head.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            let mut probs: Vec<f64> = head
                .iter()
                .map(|(_, logit)| ((f64::from(*logit) - f64::from(max_logit)) * inv_t).exp())
                .collect();
            let head_sum: f64 = probs.iter().sum();
            let total_sum = head_sum + tail_sum;
            if total_sum > 0.0 && head_sum / total_sum >= params.top_p {
                let mut cumulative = 0.0f64;
                let mut cutoff_idx = probs.len();
                for (i, prob) in probs.iter_mut().enumerate() {
                    *prob /= total_sum;
                    cumulative += *prob;
                    if cumulative >= params.top_p {
                        cutoff_idx = i + 1;
                        break;
                    }
                }
                head.truncate(cutoff_idx);
                probs.truncate(cutoff_idx);
                let kept_sum: f64 = probs.iter().sum();
                if kept_sum > 0.0 {
                    return Ok(head
                        .into_iter()
                        .zip(probs)
                        .map(|((id, _), p)| (id as u32, p / kept_sum))
                        .collect());
                }
            }
        }
    }

    let mut candidates: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    if params.top_k > 0 && params.top_k < candidates.len() {
        let k = params.top_k;
        candidates.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        candidates.truncate(k);
    }
    if params.top_p < 1.0 {
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    }
    let mut probs: Vec<f64> = candidates
        .iter()
        .map(|(_, logit)| ((f64::from(*logit) - f64::from(max_logit)) * inv_t).exp())
        .collect();
    let sum: f64 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }
    if params.top_p < 1.0 {
        let mut cumulative = 0.0f64;
        let mut cutoff_idx = probs.len();
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if cumulative >= params.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }
        candidates.truncate(cutoff_idx);
        probs.truncate(cutoff_idx);
        let kept_sum: f64 = probs.iter().sum();
        if kept_sum > 0.0 {
            for p in &mut probs {
                *p /= kept_sum;
            }
        }
    }
    Ok(candidates
        .into_iter()
        .zip(probs)
        .filter(|(_, p)| *p > 0.0)
        .map(|((id, _), p)| (id as u32, p))
        .collect())
}

/// Exact Leviathan et al. (2023) speculative rejection sampling step.
///
/// Given target logits, a `proposed` token sampled from sparse draft
/// distribution `draft_dist` (`(token_id, q_prob)` pairs summing to 1), either
/// accepts `proposed` with probability `min(1, p(proposed) / q(proposed))` or
/// rejects it and samples from the exact residual `normalize(max(0, p - q))`.
/// The output distribution is mathematically identical to sampling from the
/// target distribution `p` for any valid (even truncated) `draft_dist`.
pub(crate) fn verify_rejection_sample_host(
    logits: &[f32],
    proposed: u32,
    draft_dist: &[(u32, f32)],
    params: &SamplingParams,
    rng: &mut SamplingRng,
) -> Result<(u32, bool)> {
    params.validate()?;
    anyhow::ensure!(
        params.temperature > 0.0,
        "rejection sampling requires positive temperature"
    );
    let target_dist = positive_target_distribution(logits, params)?;
    anyhow::ensure!(!target_dist.is_empty(), "target distribution is empty");
    let p_d = target_dist
        .iter()
        .find(|&&(id, _)| id == proposed)
        .map_or(0.0, |&(_, p)| p);
    let q_d = draft_dist
        .iter()
        .find(|&&(id, _)| id == proposed)
        .map_or(0.0, |&(_, q)| f64::from(q));
    if q_d > 0.0 && p_d > 0.0 {
        if p_d >= q_d || f64::from(rng.next_f32()) * q_d < p_d {
            return Ok((proposed, true));
        }
    }
    let mut residual: Vec<(u32, f64)> = Vec::with_capacity(target_dist.len());
    let mut residual_sum = 0.0f64;
    for &(id, p) in &target_dist {
        let q = draft_dist
            .iter()
            .find(|&&(qid, _)| qid == id)
            .map_or(0.0, |&(_, q)| f64::from(q));
        let r = (p - q).max(0.0);
        if r > 0.0 {
            residual.push((id, r));
            residual_sum += r;
        }
    }
    let (pool, total) = if residual_sum > 0.0 {
        (&residual, residual_sum)
    } else {
        let t: f64 = target_dist.iter().map(|&(_, p)| p).sum();
        (&target_dist, t)
    };
    let target = f64::from(rng.next_f32()) * total;
    let mut cumulative = 0.0f64;
    for &(id, w) in pool {
        cumulative += w;
        if target < cumulative {
            return Ok((id, false));
        }
    }
    Ok((pool.last().expect("non-empty distribution").0, false))
}

fn sample_positive_candidates(
    mut candidates: Vec<(usize, f32)>,
    params: &SamplingParams,
    rng: &mut SamplingRng,
) -> Result<u32> {
    let vocab_size = candidates.len();

    // Temperature is monotonic, so select and sort on the original logits.
    // Subtract the maximum before dividing in f64 below; this stays stable
    // even for a tiny positive temperature and never forms a positive
    // infinity through an f32 reciprocal.
    let candidate_count = if params.top_k > 0 {
        params.top_k.min(vocab_size)
    } else {
        vocab_size
    };
    if candidate_count < vocab_size {
        candidates.select_nth_unstable_by(candidate_count, |a, b| b.1.total_cmp(&a.1));
        candidates.truncate(candidate_count);
    }
    candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    // Softmax
    let max_logit = candidates
        .first()
        .map(|(_, logit)| *logit)
        .unwrap_or(f32::NEG_INFINITY);
    // A large vocabulary can lose enough mass in F32 sums to reach the
    // rounding fallback even when its last token has zero softmax weight.
    // Keep probabilities and their running totals at double precision.
    let mut probs: Vec<f64> = candidates
        .iter()
        .map(|(_, logit)| ((f64::from(*logit) - f64::from(max_logit)) / params.temperature).exp())
        .collect();
    let sum: f64 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }

    // Top-P (nucleus) filtering
    if params.top_p < 1.0 {
        let mut cumulative = 0.0f64;
        let mut cutoff_idx = probs.len();
        for (i, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if cumulative >= params.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }
        probs.truncate(cutoff_idx);
        candidates.truncate(cutoff_idx);

        // Renormalize
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        }
    }

    // Weighted random sampling
    let r = f64::from(rng.next_f32());
    let mut cumulative = 0.0f64;
    for ((token_id, _), &p) in candidates.iter().zip(&probs) {
        cumulative += p;
        if r < cumulative {
            return Ok(*token_id as u32);
        }
    }

    // Rounding must never make a zero-weight tail token reachable.
    candidates
        .iter()
        .zip(&probs)
        .rfind(|(_, probability)| **probability > 0.0)
        .map(|((idx, _), _)| *idx as u32)
        .ok_or_else(|| anyhow::anyhow!("sampling produced no candidates"))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use candle_core::Device;

    #[test]
    fn cached_history_matches_growing_history_and_seeded_rng() -> Result<()> {
        for temperature in [0.0, 0.8, 1e-300] {
            let params = SamplingParams {
                temperature,
                top_k: 4,
                repetition_penalty: 1.2,
                frequency_penalty: -0.15,
                presence_penalty: 0.25,
                logit_bias: BTreeMap::from([(4, 0.3)]),
                ..Default::default()
            };
            let mut tokens = vec![1, 1, 3, 5, u32::MAX];
            let mut history = SamplingHistory::new(&tokens, &params);
            let mut reference_rng = SamplingRng::seeded(42);
            let mut cached_rng = SamplingRng::seeded(42);
            for step in 0..64 {
                let logits = Tensor::from_vec(
                    vec![0.4f32, 1.2, -0.8, step as f32 / 100.0, 1.0, -0.1],
                    6,
                    &Device::Cpu,
                )?;
                let expected = sample_token_with_penalties_with_rng(
                    &logits,
                    &tokens,
                    &params,
                    &mut reference_rng,
                )?;
                let actual = history.sample(&logits, &params, &mut cached_rng)?;
                assert_eq!(actual, expected, "step {step}, temperature {temperature}");
                tokens.push(actual);
                history.push(actual);
            }
            assert_eq!(cached_rng.state, reference_rng.state);
        }
        Ok(())
    }

    #[test]
    fn bias_only_history_skips_counts_and_rejects_late_history_penalties() -> Result<()> {
        let mut params = SamplingParams {
            logit_bias: BTreeMap::from([(1, 0.5)]),
            ..Default::default()
        };
        let mut history = SamplingHistory::new(&[1; 4096], &params);
        history.push(2);
        assert!(history.counts.is_none());
        let mut rng = SamplingRng::seeded(42);
        history.sample_host(vec![1.0, 2.0], &params, &mut rng)?;
        params.frequency_penalty = 0.1;
        assert!(
            history
                .sample_host(vec![1.0, 2.0], &params, &mut rng)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn speculative_history_copy_cannot_leak_rejected_tokens() {
        let params = SamplingParams {
            frequency_penalty: 0.2,
            ..Default::default()
        };
        let history = SamplingHistory::new(&[1, 1, 3], &params);
        let mut tentative = history.clone();
        tentative.push(1);
        tentative.push(2);
        assert_eq!(history.counts.unwrap(), HashMap::from([(1, 2), (3, 1)]));
        assert_eq!(
            tentative.counts.unwrap(),
            HashMap::from([(1, 3), (2, 1), (3, 1)])
        );
    }

    #[test]
    #[ignore = "manual release-mode long-context sampler benchmark"]
    fn benchmark_cached_sampling_history() -> Result<()> {
        use std::hint::black_box;
        use std::time::Instant;
        let params = SamplingParams {
            temperature: 0.8,
            repetition_penalty: 1.1,
            frequency_penalty: 0.1,
            ..Default::default()
        };
        let tokens = (0..131_072).map(|i| (i % 8192) as u32).collect::<Vec<_>>();
        let history = SamplingHistory::new(&tokens, &params);
        let logits = Tensor::from_vec(
            (0..248_320)
                .map(|i| (i % 997) as f32 / 997.0)
                .collect::<Vec<_>>(),
            248_320,
            &Device::Cpu,
        )?;
        let measure = |cached: bool| -> Result<_> {
            let mut rng = SamplingRng::seeded(42);
            let started = Instant::now();
            for _ in 0..128 {
                black_box(if cached {
                    history.sample(&logits, &params, &mut rng)?
                } else {
                    sample_token_with_penalties_with_rng(&logits, &tokens, &params, &mut rng)?
                });
            }
            Ok(started.elapsed())
        };
        measure(false)?;
        measure(true)?;
        let mut old = Vec::new();
        let mut cached = Vec::new();
        for round in 0..5 {
            if round % 2 == 0 {
                old.push(measure(false)?);
                cached.push(measure(true)?);
            } else {
                cached.push(measure(true)?);
                old.push(measure(false)?);
            }
        }
        old.sort_unstable();
        cached.sort_unstable();
        eprintln!(
            "CACHED_HISTORY_BENCH tokens={} vocab=248320 old_ms={:.3} cached_ms={:.3} speedup={:.2}x",
            tokens.len(),
            old[2].as_secs_f64() * 1000.0 / 128.0,
            cached[2].as_secs_f64() * 1000.0 / 128.0,
            old[2].as_secs_f64() / cached[2].as_secs_f64()
        );
        Ok(())
    }

    #[test]
    fn test_greedy_sampling() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![1.0f32, 5.0, 2.0, 0.5], (4,), &device).unwrap();
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let token = sample_token(&logits, &params).unwrap();
        assert_eq!(token, 1); // index of max value (5.0)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn greedy_sampling_uses_validated_metal_reduction() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let logits = Tensor::from_vec(vec![-4.0f32, 7.0, 7.0, -2.0, 1.0], 5, &device)?;
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        assert_eq!(sample_token(&logits, &params)?, 1);
        Ok(())
    }

    #[test]
    fn sampling_params_validate_rejects_invalid_bounds() {
        type InvalidParam = (&'static str, fn(&mut SamplingParams));
        let invalid: &[InvalidParam] = &[
            ("temperature", |params: &mut SamplingParams| {
                params.temperature = f64::NAN;
            }),
            ("temperature", |params: &mut SamplingParams| {
                params.temperature = f64::INFINITY;
            }),
            ("top_p", |params: &mut SamplingParams| {
                params.top_p = 0.0;
            }),
            ("top_p", |params: &mut SamplingParams| {
                params.top_p = f64::NAN;
            }),
            ("repetition penalty", |params: &mut SamplingParams| {
                params.repetition_penalty = 0.0;
            }),
            ("repetition penalty", |params: &mut SamplingParams| {
                params.repetition_penalty = f64::INFINITY;
            }),
            ("repetition penalty", |params: &mut SamplingParams| {
                params.repetition_penalty = 1e300;
            }),
            ("repetition penalty", |params: &mut SamplingParams| {
                params.repetition_penalty = 1e-300;
            }),
            ("frequency penalty", |params: &mut SamplingParams| {
                params.frequency_penalty = 2.01;
            }),
            ("presence penalty", |params: &mut SamplingParams| {
                params.presence_penalty = -2.01;
            }),
            ("logit bias", |params: &mut SamplingParams| {
                params.logit_bias.insert(0, f32::NAN);
            }),
            ("logit bias", |params: &mut SamplingParams| {
                params.logit_bias.insert(0, 100.01);
            }),
            ("max_tokens", |params: &mut SamplingParams| {
                params.max_tokens = 0;
            }),
            ("min_tokens", |params: &mut SamplingParams| {
                params.min_tokens = params.max_tokens + 1;
            }),
        ];

        for (field, mutate) in invalid {
            let mut params = SamplingParams::default();
            mutate(&mut params);
            let error = params.validate().unwrap_err().to_string();
            assert!(
                error.contains(field),
                "expected {field} validation error, got {error}"
            );
        }

        let mut too_many_biases = SamplingParams::default();
        too_many_biases
            .logit_bias
            .extend((0..=MAX_LOGIT_BIAS_ENTRIES as u32).map(|token_id| (token_id, 0.0)));
        let error = too_many_biases.validate().unwrap_err().to_string();
        assert!(error.contains("more than 4096 entries"));
    }

    #[test]
    fn sample_token_rejects_empty_logits_instead_of_returning_zero() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(Vec::<f32>::new(), (0,), &device).unwrap();
        let error = sample_token(&logits, &SamplingParams::default()).unwrap_err();
        assert!(error.to_string().contains("at least one value"));
    }

    #[test]
    fn sample_token_rejects_nonfinite_logits() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![1.0f32, f32::NAN], (2,), &device).unwrap();
        let error = sample_token(&logits, &SamplingParams::default()).unwrap_err();
        assert!(error.to_string().contains("logits must be finite"));
    }

    #[test]
    fn greedy_sampling_rejects_nonfinite_selected_logit() {
        let logits = Tensor::from_vec(vec![1.0f32, f32::INFINITY], (2,), &Device::Cpu).unwrap();
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        assert!(sample_token(&logits, &params).is_err());
    }

    #[test]
    fn greedy_sampling_rejects_nonfinite_unselected_logit() {
        let logits = Tensor::from_vec(vec![f32::NAN, 1.0], (2,), &Device::Cpu).unwrap();
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let error = sample_token(&logits, &params).unwrap_err().to_string();
        assert!(error.contains("logits must be finite"), "{error}");
    }

    #[test]
    fn penalty_transform_validates_only_penalty_controls() {
        let logits = Tensor::from_vec(vec![1.0f32, 2.0], (2,), &Device::Cpu).unwrap();
        let params = SamplingParams {
            top_p: f64::NAN,
            max_tokens: 0,
            ..SamplingParams::default()
        };
        assert_eq!(
            apply_sampling_penalties(&logits, &[], &params)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1.0, 2.0]
        );
    }

    #[test]
    fn tiny_positive_temperature_remains_numerically_stable() {
        let logits = Tensor::from_vec(vec![1.0f32, 2.0, 0.0], (3,), &Device::Cpu).unwrap();
        let params = SamplingParams {
            temperature: f64::MIN_POSITIVE,
            top_k: 0,
            top_p: 1.0,
            ..Default::default()
        };
        let mut rng = SamplingRng::seeded(7);
        assert_eq!(
            sample_token_with_rng(&logits, &params, &mut rng).unwrap(),
            1
        );
    }

    #[test]
    fn rounded_probability_sum_cannot_select_an_underflowed_token() -> Result<()> {
        let mut logits = vec![0.0; 1000];
        logits.push(-1000.0);
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            ..Default::default()
        };
        // This draw exceeds the F32 sum of 1000 probabilities of 0.001.
        // The last token has zero softmax weight, so rounding must not pick it.
        let token =
            sample_positive_from_logits_vec(logits, &params, &mut SamplingRng::seeded(231_432))?;
        assert!(token < 1000, "sampled a token with zero softmax weight");
        Ok(())
    }

    #[test]
    fn nucleus_cutoff_keeps_f64_precision() -> Result<()> {
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.500_000_000_1,
            ..Default::default()
        };
        // Half of the mass is below this cutoff, so both tokens must remain.
        assert_eq!(
            sample_positive_from_logits_vec(vec![0.0, 0.0], &params, &mut SamplingRng::seeded(2),)?,
            1
        );
        Ok(())
    }

    #[test]
    fn test_repetition_penalty() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![1.0f32, 5.0, 2.0, 0.5], (4,), &device).unwrap();
        let penalized = apply_repetition_penalty(&logits, &[1], 2.0).unwrap();
        let vals = penalized.to_vec1::<f32>().unwrap();
        assert_eq!(vals[0], 1.0); // unchanged
        assert_eq!(vals[1], 2.5); // 5.0 / 2.0
        assert_eq!(vals[2], 2.0); // unchanged
    }

    #[test]
    fn repetition_penalty_is_applied_once_per_unique_token() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![1.0f32, 8.0], (2,), &device).unwrap();
        let penalized = apply_repetition_penalty(&logits, &[1, 1, 1], 2.0).unwrap();
        assert_eq!(penalized.to_vec1::<f32>().unwrap(), vec![1.0, 4.0]);
    }

    #[test]
    fn repetition_penalty_rejects_loss_of_f32_range_before_reading_logits() {
        let logits = Tensor::from_vec(Vec::<f32>::new(), 0, &Device::Cpu).unwrap();
        for penalty in [1e300, 1e-300] {
            for history in [&[][..], &[0][..]] {
                let error = apply_repetition_penalty(&logits, history, penalty)
                    .unwrap_err()
                    .to_string();
                assert!(error.contains("repetition penalty"), "{error}");
            }
            let params = SamplingParams {
                repetition_penalty: penalty,
                ..Default::default()
            };
            let error = sample_token_with_penalties_with_rng(
                &logits,
                &[0],
                &params,
                &mut SamplingRng::seeded(7),
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains("repetition penalty"), "{error}");
        }
    }

    #[test]
    fn additive_penalties_and_logit_bias_use_token_counts() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &device).unwrap();
        let params = SamplingParams {
            frequency_penalty: 0.5,
            presence_penalty: 0.75,
            logit_bias: BTreeMap::from([(1, 2.0)]),
            ..Default::default()
        };
        let adjusted = apply_sampling_penalties(&logits, &[1, 1, 2], &params)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(adjusted, vec![1.0, 2.25, 1.75]);
    }

    #[test]
    fn logit_bias_rejects_out_of_vocabulary_ids() {
        let device = Device::Cpu;
        let logits = Tensor::zeros((3,), DType::F32, &device).unwrap();
        let params = SamplingParams {
            logit_bias: BTreeMap::from([(3, 1.0)]),
            ..Default::default()
        };
        let error = apply_sampling_penalties(&logits, &[], &params).unwrap_err();
        assert!(error.to_string().contains("outside the model vocabulary"));
    }

    #[test]
    fn seeded_sampling_is_reproducible() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0], (4,), &device).unwrap();
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            ..Default::default()
        };
        let mut left = SamplingRng::seeded(42);
        let mut right = SamplingRng::seeded(42);
        let left = (0..16)
            .map(|_| sample_token_with_rng(&logits, &params, &mut left).unwrap())
            .collect::<Vec<_>>();
        let right = (0..16)
            .map(|_| sample_token_with_rng(&logits, &params, &mut right).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(left, right);
    }

    #[test]
    fn combined_penalized_sampling_matches_legacy_seeded_sequence() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(
            vec![0.4f32, 1.2, -0.8, 0.7, 1.0, -0.1, 0.2, 0.9],
            (8,),
            &device,
        )
        .unwrap();
        let history = [1, 1, 3, 5];
        let params = SamplingParams {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 4,
            repetition_penalty: 1.2,
            frequency_penalty: 0.15,
            presence_penalty: 0.25,
            logit_bias: BTreeMap::from([(4, 0.3)]),
            ..Default::default()
        };

        let adjusted = apply_sampling_penalties(&logits, &history, &params).unwrap();
        let mut legacy_rng = SamplingRng::seeded(42);
        let legacy = (0..32)
            .map(|_| sample_token_with_rng(&adjusted, &params, &mut legacy_rng).unwrap())
            .collect::<Vec<_>>();

        let mut combined_rng = SamplingRng::seeded(42);
        let combined = (0..32)
            .map(|_| {
                sample_token_with_penalties_with_rng(&logits, &history, &params, &mut combined_rng)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(combined, legacy);
    }

    #[test]
    fn combined_penalized_sampling_rejects_nonfinite_logits() {
        let logits = Tensor::from_vec(vec![1.0f32, f32::NAN], (2,), &Device::Cpu).unwrap();
        let params = SamplingParams {
            temperature: 0.8,
            repetition_penalty: 1.2,
            ..Default::default()
        };
        let error = sample_token_with_penalties_with_rng(
            &logits,
            &[1],
            &params,
            &mut SamplingRng::seeded(7),
        )
        .unwrap_err();
        assert!(error.to_string().contains("logits must be finite"));
    }

    #[test]
    fn checked_argmax_matches_candle_for_randomized_finite_logits() -> anyhow::Result<()> {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for length in [1usize, 2, 3, 31, 257, 1_009] {
            for _ in 0..16 {
                let values = (0..length)
                    .map(|_| {
                        state ^= state >> 12;
                        state ^= state << 25;
                        state ^= state >> 27;
                        let bits = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
                        ((bits >> 40) as i32 - (1 << 23)) as f32 / 65_536.0
                    })
                    .collect::<Vec<_>>();
                let reference = Tensor::from_vec(values.clone(), length, &Device::Cpu)?
                    .argmax(0)?
                    .to_scalar::<u32>()?;
                assert_eq!(checked_first_argmax(&values)?, reference);
            }
        }
        Ok(())
    }

    #[test]
    fn checked_argmax_uses_first_tie_and_handles_all_negative_logits() {
        assert_eq!(checked_first_argmax(&[-9.0, -2.0, -2.0, -7.0]).unwrap(), 1);
        assert_eq!(checked_first_argmax(&[-f32::MAX]).unwrap(), 0);
    }

    #[test]
    fn checked_argmax_rejects_every_nonfinite_class() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let error = checked_first_argmax(&[10.0, invalid, 1.0])
                .unwrap_err()
                .to_string();
            assert!(error.contains("logits must be finite"), "{error}");
        }
    }

    #[test]
    fn penalized_greedy_reuses_host_logits_with_history_and_bias() -> anyhow::Result<()> {
        let logits = Tensor::from_vec(vec![4.0f32, 5.0, 3.0, 2.0], 4, &Device::Cpu)?;
        let history = [0, 1, 1];
        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 2.0,
            frequency_penalty: 0.5,
            presence_penalty: 0.25,
            logit_bias: BTreeMap::from([(2, 4.0)]),
            ..Default::default()
        };
        let token = sample_token_with_penalties_with_rng(
            &logits,
            &history,
            &params,
            &mut SamplingRng::seeded(7),
        )?;
        assert_eq!(token, 2);

        let adjusted = apply_sampling_penalties(&logits, &history, &params)?;
        assert_eq!(
            token,
            sample_token_with_rng(&adjusted, &params, &mut SamplingRng::seeded(7))?
        );
        Ok(())
    }

    #[test]
    fn penalized_greedy_matches_legacy_for_randomized_logits_and_histories() -> anyhow::Result<()> {
        let mut state = 0xd1b5_4a32_d192_ed03u64;
        for round in 0..32u32 {
            let logits = (0..257)
                .map(|_| {
                    state ^= state >> 12;
                    state ^= state << 25;
                    state ^= state >> 27;
                    let bits = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
                    ((bits >> 40) as i32 - (1 << 23)) as f32 / 131_072.0
                })
                .collect::<Vec<_>>();
            let history = (0..64)
                .map(|index| (index * 37 + round * 11) % 257)
                .collect::<Vec<_>>();
            let params = SamplingParams {
                temperature: 0.0,
                repetition_penalty: 1.0 + f64::from(round % 5) * 0.1,
                frequency_penalty: f64::from((round % 7) as i32 - 3) * 0.1,
                presence_penalty: f64::from((round % 5) as i32 - 2) * 0.15,
                logit_bias: BTreeMap::from([
                    ((round * 17) % 257, 0.75),
                    ((round * 29 + 3) % 257, -0.5),
                ]),
                ..Default::default()
            };
            let logits = Tensor::from_vec(logits, 257, &Device::Cpu)?;
            let adjusted = apply_sampling_penalties(&logits, &history, &params)?;
            let expected = sample_token_with_rng(
                &adjusted,
                &params,
                &mut SamplingRng::seeded(u64::from(round)),
            )?;
            let actual = sample_token_with_penalties_with_rng(
                &logits,
                &history,
                &params,
                &mut SamplingRng::seeded(u64::from(round)),
            )?;
            assert_eq!(actual, expected, "round {round}");
        }
        Ok(())
    }

    #[test]
    fn penalized_greedy_rejects_nonfinite_raw_or_adjusted_logits() -> anyhow::Result<()> {
        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 2.0,
            ..Default::default()
        };
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let logits = Tensor::from_vec(vec![3.0f32, invalid, 2.0], 3, &Device::Cpu)?;
            let error = sample_token_with_penalties_with_rng(
                &logits,
                &[0],
                &params,
                &mut SamplingRng::seeded(7),
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains("logits must be finite"), "{error}");
        }

        let overflow_params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: f64::from(f32::from_bits(1)),
            ..Default::default()
        };
        let logits = Tensor::from_vec(vec![1.0f32, 0.0], 2, &Device::Cpu)?;
        let error = sample_token_with_penalties_with_rng(
            &logits,
            &[0],
            &overflow_params,
            &mut SamplingRng::seeded(7),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("logits must be finite"), "{error}");
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "manual Metal microbenchmark"]
    fn benchmark_combined_penalized_sampling_metal() -> anyhow::Result<()> {
        use std::hint::black_box;
        use std::time::{Duration, Instant};

        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let vocab = 248_320usize;
        let logits = Tensor::from_vec(
            (0..vocab)
                .map(|index| ((index % 4096) as f32 - 2048.0) / 128.0)
                .collect::<Vec<_>>(),
            (vocab,),
            &device,
        )?;
        let history = [1, 1, 3, 5, 8, 13, 21];
        let params = SamplingParams {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            repetition_penalty: 1.2,
            frequency_penalty: 0.15,
            presence_penalty: 0.25,
            ..Default::default()
        };
        let iterations = 64usize;
        let measure = |combined: bool| -> anyhow::Result<Duration> {
            let mut rng = SamplingRng::seeded(42);
            device.synchronize()?;
            let started = Instant::now();
            for _ in 0..iterations {
                let token = if combined {
                    sample_token_with_penalties_with_rng(&logits, &history, &params, &mut rng)?
                } else {
                    let adjusted = apply_sampling_penalties(&logits, &history, &params)?;
                    sample_token_with_rng(&adjusted, &params, &mut rng)?
                };
                black_box(token);
            }
            device.synchronize()?;
            Ok(started.elapsed())
        };

        measure(false)?;
        measure(true)?;
        let mut legacy_runs = Vec::with_capacity(5);
        let mut combined_runs = Vec::with_capacity(5);
        for round in 0..5 {
            if round % 2 == 0 {
                combined_runs.push(measure(true)?);
                legacy_runs.push(measure(false)?);
            } else {
                legacy_runs.push(measure(false)?);
                combined_runs.push(measure(true)?);
            }
        }
        legacy_runs.sort_unstable();
        combined_runs.sort_unstable();
        let legacy = legacy_runs[legacy_runs.len() / 2];
        let combined = combined_runs[combined_runs.len() / 2];
        eprintln!(
            "PENALIZED_SAMPLING_METAL_BENCH vocab={vocab} iterations={iterations} legacy_ms={:.3} combined_ms={:.3} speedup={:.3}x",
            legacy.as_secs_f64() * 1_000.0,
            combined.as_secs_f64() * 1_000.0,
            legacy.as_secs_f64() / combined.as_secs_f64()
        );
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "manual optimized Metal microbenchmark; run with --release"]
    fn benchmark_penalized_greedy_host_argmax_metal() -> anyhow::Result<()> {
        use std::hint::black_box;
        use std::time::{Duration, Instant};

        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let vocab = 248_320usize;
        let logits = Tensor::from_vec(
            (0..vocab)
                .map(|index| ((index % 4096) as f32 - 2048.0) / 128.0)
                .collect::<Vec<_>>(),
            (vocab,),
            &device,
        )?;
        let history = [1, 1, 3, 5, 8, 13, 21];
        let params = SamplingParams {
            temperature: 0.0,
            top_p: 0.95,
            top_k: 40,
            repetition_penalty: 1.2,
            frequency_penalty: 0.15,
            presence_penalty: 0.25,
            ..Default::default()
        };
        let iterations = 64usize;
        let adjusted = apply_sampling_penalties(&logits, &history, &params)?;
        let expected = sample_token_with_rng(&adjusted, &params, &mut SamplingRng::seeded(42))?;
        let actual = sample_token_with_penalties_with_rng(
            &logits,
            &history,
            &params,
            &mut SamplingRng::seeded(42),
        )?;
        assert_eq!(actual, expected);

        let measure = |combined: bool| -> anyhow::Result<Duration> {
            let mut rng = SamplingRng::seeded(42);
            device.synchronize()?;
            let started = Instant::now();
            for _ in 0..iterations {
                let token = if combined {
                    sample_token_with_penalties_with_rng(&logits, &history, &params, &mut rng)?
                } else {
                    let adjusted = apply_sampling_penalties(&logits, &history, &params)?;
                    sample_token_with_rng(&adjusted, &params, &mut rng)?
                };
                black_box(token);
            }
            device.synchronize()?;
            Ok(started.elapsed())
        };

        measure(false)?;
        measure(true)?;
        let mut legacy_runs = Vec::with_capacity(5);
        let mut combined_runs = Vec::with_capacity(5);
        for round in 0..5 {
            if round % 2 == 0 {
                combined_runs.push(measure(true)?);
                legacy_runs.push(measure(false)?);
            } else {
                legacy_runs.push(measure(false)?);
                combined_runs.push(measure(true)?);
            }
        }
        legacy_runs.sort_unstable();
        combined_runs.sort_unstable();
        let legacy = legacy_runs[legacy_runs.len() / 2];
        let combined = combined_runs[combined_runs.len() / 2];
        eprintln!(
            "PENALIZED_GREEDY_METAL_BENCH vocab={vocab} iterations={iterations} legacy_ms={:.3} host_argmax_ms={:.3} speedup={:.3}x",
            legacy.as_secs_f64() * 1_000.0,
            combined.as_secs_f64() * 1_000.0,
            legacy.as_secs_f64() / combined.as_secs_f64()
        );
        Ok(())
    }

    #[test]
    fn test_top_k_one_selects_best_token() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![1.0f32, 5.0, 2.0, 0.5], (4,), &device).unwrap();
        let params = SamplingParams {
            temperature: 0.7,
            top_k: 1,
            top_p: 1.0,
            ..Default::default()
        };
        assert_eq!(sample_token(&logits, &params).unwrap(), 1);
    }

    #[test]
    fn tiny_temperatures_preserve_the_best_candidate() {
        let logits = Tensor::from_vec(vec![1.0f32, 5.0, 2.0, 0.5], (4,), &Device::Cpu).unwrap();
        for temperature in [1e-40, 1e-300, f64::from_bits(1)] {
            for top_k in [0, 1, 2, 40] {
                let params = SamplingParams {
                    temperature,
                    top_k,
                    top_p: 1.0,
                    ..Default::default()
                };
                let mut rng = SamplingRng::seeded(42);
                assert_eq!(
                    sample_token_with_rng(&logits, &params, &mut rng).unwrap(),
                    1
                );
            }
        }
    }

    #[test]
    fn tiny_temperatures_still_sample_tied_maxima() {
        let logits = Tensor::from_vec(vec![5.0f32, 1.0, 5.0], (3,), &Device::Cpu).unwrap();
        let params = SamplingParams {
            temperature: f64::from_bits(1),
            top_k: 0,
            top_p: 1.0,
            ..Default::default()
        };
        let mut rng = SamplingRng::seeded(42);
        let samples = (0..64)
            .map(|_| sample_token_with_rng(&logits, &params, &mut rng).unwrap())
            .collect::<HashSet<_>>();
        assert_eq!(samples, HashSet::from([0, 2]));
    }

    #[test]
    fn rejection_sampling_preserves_exact_target_distribution() {
        // Target logits over 6 tokens (with top_p = 0.92 excluding token 5).
        let target_logits = vec![2.4f32, 1.9, 1.3, 0.7, 0.1, -3.0];
        let params = SamplingParams {
            temperature: 0.7,
            top_k: 0,
            top_p: 0.92,
            ..Default::default()
        };
        let exact_p = positive_target_distribution(&target_logits, &params).unwrap();
        // Deliberately mismatched and truncated draft distribution, including mass
        // on token 5 which is outside the target's top_p nucleus.
        let draft_q: Vec<(u32, f32)> = vec![(0, 0.25), (1, 0.35), (2, 0.15), (5, 0.25)];
        let mut rng = SamplingRng::seeded(12345);
        let trials = 120_000usize;
        let mut counts = vec![0usize; target_logits.len()];
        for _ in 0..trials {
            let u = rng.next_f32();
            let mut cum = 0.0f32;
            let mut proposed = draft_q.last().unwrap().0;
            for &(id, q) in &draft_q {
                cum += q;
                if u < cum {
                    proposed = id;
                    break;
                }
            }
            let (tok, _) =
                verify_rejection_sample_host(&target_logits, proposed, &draft_q, &params, &mut rng)
                    .unwrap();
            counts[tok as usize] += 1;
        }
        // Token 5 is outside target's top_p nucleus and must NEVER be emitted.
        assert_eq!(counts[5], 0);
        for &(id, p) in &exact_p {
            let empirical = counts[id as usize] as f64 / trials as f64;
            assert!(
                (empirical - p).abs() < 0.006,
                "token {id}: empirical={empirical:.4}, exact={p:.4}"
            );
        }
    }
}
