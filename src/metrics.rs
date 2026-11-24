//! Rolling inference-performance metrics.
//!
//! The public types in this module collect bounded, recency-weighted rates
//! from completed requests and per-stage service intervals. They are
//! observational only and do not affect inference or scheduling directly.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
struct WeightedSample {
    value: f64,
    weight: f64,
    observed_at: Instant,
}

/// Lightweight struct accumulating timing data from actual inference runs.
/// Updated once per completed inference, not per-token.
pub struct InferenceMetrics {
    decode_samples: VecDeque<WeightedSample>,
    prefill_tok_s_samples: VecDeque<WeightedSample>,
    ttft_samples: VecDeque<WeightedSample>,
    // Whole-model-equivalent throughput measured from local stage service only.
    stage_decode_samples: VecDeque<WeightedSample>,
    stage_prefill_samples: VecDeque<WeightedSample>,
    inference_count: u64,
    tokens_served: u64,
    last_completed_at: Option<Instant>,
}

const MAX_SAMPLES: usize = 16;
const SAMPLE_HALF_LIFE_SECS: f64 = 20.0 * 60.0;
const DECODE_WEIGHT_TOKENS: usize = 128;
const PREFILL_WEIGHT_TOKENS: usize = 512;

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceMetrics {
    /// Create an empty metrics accumulator.
    pub fn new() -> Self {
        Self {
            decode_samples: VecDeque::with_capacity(MAX_SAMPLES),
            prefill_tok_s_samples: VecDeque::with_capacity(MAX_SAMPLES),
            ttft_samples: VecDeque::with_capacity(MAX_SAMPLES),
            stage_decode_samples: VecDeque::with_capacity(MAX_SAMPLES),
            stage_prefill_samples: VecDeque::with_capacity(MAX_SAMPLES),
            inference_count: 0,
            tokens_served: 0,
            last_completed_at: None,
        }
    }

    /// Drop model-specific rate samples while keeping process-wide counters.
    pub fn reset_rates(&mut self) {
        self.decode_samples.clear();
        self.prefill_tok_s_samples.clear();
        self.ttft_samples.clear();
        self.stage_decode_samples.clear();
        self.stage_prefill_samples.clear();
    }

    /// Record one completed inference.
    /// `prompt_tokens` is the number of tokens in the prompt (for prefill tok/s).
    /// A zero `prefill_duration` adds no prefill-rate sample; callers pass it
    /// when the prompt time measures something other than prompt throughput.
    /// `ttft` is time-to-first-token (None if not measured, e.g. non-head pipeline stages).
    pub fn record(
        &mut self,
        tokens_generated: usize,
        prompt_tokens: usize,
        decode_duration: Duration,
        prefill_duration: Duration,
        ttft: Option<Duration>,
    ) {
        self.record_request(
            tokens_generated,
            prompt_tokens,
            decode_duration,
            prefill_duration,
            ttft,
            true,
        );
    }

    /// End-to-end split request timing must not become a local GPU service rate.
    pub fn record_pipeline_request(
        &mut self,
        tokens_generated: usize,
        prompt_tokens: usize,
        decode_duration: Duration,
        prefill_duration: Duration,
        ttft: Option<Duration>,
    ) {
        self.record_request(
            tokens_generated,
            prompt_tokens,
            decode_duration,
            prefill_duration,
            ttft,
            false,
        );
    }

    fn record_request(
        &mut self,
        tokens_generated: usize,
        prompt_tokens: usize,
        decode_duration: Duration,
        prefill_duration: Duration,
        ttft: Option<Duration>,
        solo: bool,
    ) {
        self.last_completed_at = Some(Instant::now());
        self.inference_count += 1;
        self.tokens_served = self
            .tokens_served
            .saturating_add(tokens_generated as u64)
            .saturating_add(prompt_tokens as u64);

        // Decode tok/s excludes the first token (its latency is prefill/TTFT).
        // For 1-token generations, decode_duration is ~0 and would produce infinite tok/s.
        let decode_tokens = tokens_generated.saturating_sub(1);
        if decode_tokens > 0 && decode_duration.as_secs_f64() > 0.0 {
            let tok_s = decode_tokens as f64 / decode_duration.as_secs_f64();
            let weight = token_weight(decode_tokens, DECODE_WEIGHT_TOKENS);
            push_sample(&mut self.decode_samples, tok_s, weight);
            // A solo run covers the complete model, so its local throughput is
            // already a whole-model-equivalent stage measurement.
            if solo {
                push_sample(&mut self.stage_decode_samples, tok_s, weight);
            }
        }

        if prompt_tokens > 0 && prefill_duration.as_secs_f64() > 0.0 {
            let prefill_tok_s = prompt_tokens as f64 / prefill_duration.as_secs_f64();
            let weight = token_weight(prompt_tokens, PREFILL_WEIGHT_TOKENS);
            push_sample(&mut self.prefill_tok_s_samples, prefill_tok_s, weight);
            if solo {
                push_sample(&mut self.stage_prefill_samples, prefill_tok_s, weight);
            }
        }

        if let Some(t) = ttft {
            let ttft_ms = t.as_secs_f64() * 1000.0;
            if ttft_ms > 0.0 {
                let weight = token_weight(prompt_tokens.max(1), PREFILL_WEIGHT_TOKENS);
                push_sample(&mut self.ttft_samples, ttft_ms, weight);
            }
        }
    }

    /// Record local service timings for one pipeline stage.
    ///
    /// Stage rates are normalized by the fraction of model layers assigned to
    /// this stage. This produces a whole-model-equivalent rate that can be
    /// compared across different layer splits, excluding time spent waiting
    /// on other stages.
    pub fn record_stage(
        &mut self,
        decode_tokens: usize,
        prefill_tokens: usize,
        decode_service: Duration,
        prefill_service: Duration,
        layer_count: usize,
        total_layers: usize,
    ) {
        self.last_completed_at = Some(Instant::now());
        self.inference_count += 1;
        self.tokens_served = self
            .tokens_served
            .saturating_add(decode_tokens as u64)
            .saturating_add(prefill_tokens as u64);

        self.record_stage_service(
            decode_tokens,
            prefill_tokens,
            decode_service,
            prefill_service,
            layer_count,
            total_layers,
        );
    }

    /// Ready-work timings exclude waits on other stages and do not count
    /// another finished request. Stages can report service before a long
    /// request ends.
    pub fn record_stage_service(
        &mut self,
        decode_tokens: usize,
        prefill_tokens: usize,
        decode_service: Duration,
        prefill_service: Duration,
        layer_count: usize,
        total_layers: usize,
    ) {
        if layer_count == 0 || total_layers == 0 {
            return;
        }
        let layer_fraction = layer_count as f64 / total_layers as f64;

        if decode_tokens > 0 && decode_service.as_secs_f64() > 0.0 {
            let stage_tok_s = decode_tokens as f64 / decode_service.as_secs_f64();
            let equivalent_tok_s = stage_tok_s * layer_fraction;
            let weight = token_weight(decode_tokens, DECODE_WEIGHT_TOKENS);
            push_sample(&mut self.stage_decode_samples, equivalent_tok_s, weight);
        }

        if prefill_tokens > 0 && prefill_service.as_secs_f64() > 0.0 {
            let stage_tok_s = prefill_tokens as f64 / prefill_service.as_secs_f64();
            let equivalent_tok_s = stage_tok_s * layer_fraction;
            let weight = token_weight(prefill_tokens, PREFILL_WEIGHT_TOKENS);
            push_sample(&mut self.stage_prefill_samples, equivalent_tok_s, weight);
        }
    }

    /// When the most recent request finished, if any has.
    pub fn last_completed_at(&self) -> Option<Instant> {
        self.last_completed_at
    }

    /// Returns (avg_decode_tok_s, avg_prefill_tok_s, avg_ttft_ms, inference_count).
    pub fn snapshot(&self) -> MetricsSnapshot {
        let now = Instant::now();
        let (decode_tok_s, decode_sample_weight) = weighted_summary(&self.decode_samples, now);
        let (prefill_tok_s, prefill_sample_weight) =
            weighted_summary(&self.prefill_tok_s_samples, now);
        let (avg_ttft_ms, _) = weighted_summary(&self.ttft_samples, now);
        let (stage_decode_tok_s, stage_decode_sample_weight) =
            weighted_summary(&self.stage_decode_samples, now);
        let (stage_prefill_tok_s, stage_prefill_sample_weight) =
            weighted_summary(&self.stage_prefill_samples, now);

        MetricsSnapshot {
            decode_tok_s,
            prefill_tok_s,
            avg_ttft_ms,
            inference_count: self.inference_count,
            tokens_served: self.tokens_served,
            decode_sample_weight,
            prefill_sample_weight,
            stage_decode_tok_s,
            stage_prefill_tok_s,
            stage_decode_sample_weight,
            stage_prefill_sample_weight,
        }
    }
}

/// A point-in-time view of request and stage performance metrics.
///
/// Rates are recency-weighted tok/s values; zero means that no usable sample
/// has been recorded. Sample weights indicate how much recent data contributes
/// to the corresponding rate and are not token counts.
#[derive(Debug, Clone, Copy)]
pub struct MetricsSnapshot {
    /// Whole-model decode throughput in tokens per second.
    pub decode_tok_s: f64,
    /// Whole-model prefill throughput in tokens per second.
    pub prefill_tok_s: f64,
    /// Recency-weighted average time to first token, in milliseconds.
    pub avg_ttft_ms: f64,
    /// Number of completed inference records, including records without rate
    /// samples.
    pub inference_count: u64,
    /// Total prompt and generated tokens recorded.
    pub tokens_served: u64,
    /// Effective recency/size weight of decode-rate samples.
    pub decode_sample_weight: f64,
    /// Effective recency/size weight of prefill-rate samples.
    pub prefill_sample_weight: f64,
    /// Whole-model-equivalent decode rate for pipeline stage service.
    pub stage_decode_tok_s: f64,
    /// Whole-model-equivalent prefill rate for pipeline stage service.
    pub stage_prefill_tok_s: f64,
    /// Effective recency/size weight of stage decode samples.
    pub stage_decode_sample_weight: f64,
    /// Effective recency/size weight of stage prefill samples.
    pub stage_prefill_sample_weight: f64,
}

fn token_weight(tokens: usize, full_weight_tokens: usize) -> f64 {
    tokens.min(full_weight_tokens) as f64 / full_weight_tokens as f64
}

fn push_sample(q: &mut VecDeque<WeightedSample>, value: f64, weight: f64) {
    if !value.is_finite() || value <= 0.0 || !weight.is_finite() || weight <= 0.0 {
        return;
    }
    if q.len() >= MAX_SAMPLES {
        q.pop_front();
    }
    q.push_back(WeightedSample {
        value,
        weight,
        observed_at: Instant::now(),
    });
}

fn weighted_summary(q: &VecDeque<WeightedSample>, now: Instant) -> (f64, f64) {
    if q.is_empty() {
        return (0.0, 0.0);
    }

    // Winsorize the tails once enough history exists so a thermal event or a
    // layer-load outlier cannot dominate the routing estimate.
    let mut values: Vec<f64> = q.iter().map(|sample| sample.value).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let trim = if values.len() >= 8 {
        values.len() / 8
    } else {
        0
    };
    let lower = values[trim];
    let upper = values[values.len() - 1 - trim];

    let mut weighted_sum = 0.0;
    let mut effective_weight = 0.0;
    for sample in q {
        let age_secs = now
            .checked_duration_since(sample.observed_at)
            .unwrap_or_default()
            .as_secs_f64();
        let recency = 0.5_f64.powf(age_secs / SAMPLE_HALF_LIFE_SECS);
        let weight = sample.weight * recency;
        weighted_sum += sample.value.clamp(lower, upper) * weight;
        effective_weight += weight;
    }

    if effective_weight > 0.0 {
        (weighted_sum / effective_weight, effective_weight)
    } else {
        (0.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_snapshot() {
        let m = InferenceMetrics::new();
        let s = m.snapshot();
        assert_eq!(s.decode_tok_s, 0.0);
        assert_eq!(s.prefill_tok_s, 0.0);
        assert_eq!(s.avg_ttft_ms, 0.0);
        assert_eq!(s.inference_count, 0);
        assert_eq!(s.tokens_served, 0);
        assert_eq!(s.decode_sample_weight, 0.0);
        assert_eq!(s.stage_decode_tok_s, 0.0);
        assert_eq!(s.stage_decode_sample_weight, 0.0);
    }

    #[test]
    fn test_single_record() {
        let mut m = InferenceMetrics::new();
        m.record(
            10,
            128,
            Duration::from_secs(1),
            Duration::from_millis(200),
            Some(Duration::from_millis(250)),
        );
        let s = m.snapshot();
        // 10 tokens generated, but first token is prefill cost → 9 decode tokens / 1s
        assert!((s.decode_tok_s - 9.0).abs() < 0.01);
        assert!((s.prefill_tok_s - 640.0).abs() < 1.0); // 128 / 0.2
        assert!((s.avg_ttft_ms - 250.0).abs() < 0.1);
        assert_eq!(s.inference_count, 1);
        assert_eq!(s.tokens_served, 138);
    }

    #[test]
    fn test_rolling_window() {
        let mut m = InferenceMetrics::new();
        for i in 0..20 {
            m.record(
                10 + i,
                100,
                Duration::from_secs(1),
                Duration::from_millis(100),
                Some(Duration::from_millis(150)),
            );
        }
        assert_eq!(m.decode_samples.len(), 16);
        assert_eq!(m.prefill_tok_s_samples.len(), 16);
        assert_eq!(m.ttft_samples.len(), 16);
        assert_eq!(m.stage_decode_samples.len(), 16);
        assert_eq!(m.stage_prefill_samples.len(), 16);
        assert_eq!(m.inference_count, 20);
        let s = m.snapshot();
        assert!(s.decode_tok_s > 10.0);
    }

    #[test]
    fn reset_rates_keeps_cumulative_counts() {
        let mut m = InferenceMetrics::new();
        m.record(10, 20, Duration::from_secs(1), Duration::from_secs(1), None);
        m.reset_rates();
        let snapshot = m.snapshot();
        assert_eq!(snapshot.inference_count, 1);
        assert_eq!(snapshot.tokens_served, 30);
        assert_eq!(snapshot.decode_tok_s, 0.0);
        assert_eq!(snapshot.prefill_tok_s, 0.0);
    }

    #[test]
    fn split_end_to_end_rates_do_not_claim_local_gpu_service() {
        let mut metrics = InferenceMetrics::new();
        metrics.record_pipeline_request(
            11,
            20,
            Duration::from_secs(1),
            Duration::from_secs(2),
            Some(Duration::from_secs(2)),
        );
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.inference_count, 1);
        assert_eq!(snapshot.tokens_served, 31);
        assert_eq!(snapshot.decode_tok_s, 10.);
        assert_eq!(snapshot.stage_decode_tok_s, 0.);
        assert_eq!(snapshot.stage_prefill_tok_s, 0.);
    }

    #[test]
    fn test_no_ttft_for_non_head() {
        let mut m = InferenceMetrics::new();
        m.record(10, 0, Duration::from_secs(1), Duration::ZERO, None);
        let s = m.snapshot();
        // 10 tokens generated → 9 decode tokens / 1s
        assert!((s.decode_tok_s - 9.0).abs() < 0.01);
        assert_eq!(s.prefill_tok_s, 0.0);
        assert_eq!(s.avg_ttft_ms, 0.0);
    }

    #[test]
    fn zero_prefill_time_keeps_ttft_but_adds_no_prefill_rate() {
        let mut m = InferenceMetrics::new();
        assert_eq!(m.last_completed_at(), None);
        let before = Instant::now();
        // A prompt that started on an idle GPU: its 388 ms is the clock ramp.
        m.record(
            58,
            48,
            Duration::from_millis(570),
            Duration::ZERO,
            Some(Duration::from_millis(388)),
        );
        let s = m.snapshot();
        assert_eq!(s.prefill_tok_s, 0.0);
        assert_eq!(s.stage_prefill_tok_s, 0.0);
        assert!((s.avg_ttft_ms - 388.0).abs() < 0.01);
        assert!(s.decode_tok_s > 0.0);
        assert!(m.last_completed_at().is_some_and(|at| at >= before));
    }

    #[test]
    fn test_long_decode_samples_carry_more_weight() {
        let mut m = InferenceMetrics::new();
        // One decoded token at 100 tok/s should barely move a full-weight run.
        m.record(2, 0, Duration::from_millis(10), Duration::ZERO, None);
        // 128 decoded tokens at 50 tok/s.
        m.record(129, 0, Duration::from_millis(2560), Duration::ZERO, None);

        let s = m.snapshot();
        assert!(
            s.decode_tok_s > 50.0 && s.decode_tok_s < 51.0,
            "short sample should have little influence: {}",
            s.decode_tok_s
        );
        assert!(s.decode_sample_weight > 1.0 && s.decode_sample_weight < 1.01);
    }

    #[test]
    fn test_stage_rate_is_normalized_to_whole_model() {
        let mut m = InferenceMetrics::new();
        m.record_stage(
            128,
            512,
            Duration::from_secs(1),
            Duration::from_secs(2),
            8,
            32,
        );

        let s = m.snapshot();
        assert!((s.stage_decode_tok_s - 32.0).abs() < 0.01);
        assert!((s.stage_prefill_tok_s - 64.0).abs() < 0.01);
        assert!((s.stage_decode_sample_weight - 1.0).abs() < 0.01);
        assert_eq!(s.decode_tok_s, 0.0);
        assert_eq!(s.inference_count, 1);
        assert_eq!(s.tokens_served, 640);
    }

    #[test]
    fn test_mature_window_winsorizes_outlier() {
        let mut m = InferenceMetrics::new();
        for _ in 0..7 {
            m.record(129, 0, Duration::from_millis(1280), Duration::ZERO, None);
        }
        m.record(129, 0, Duration::from_millis(13), Duration::ZERO, None);

        let s = m.snapshot();
        assert!(
            (s.decode_tok_s - 100.0).abs() < 0.1,
            "outlier should be winsorized: {}",
            s.decode_tok_s
        );
    }
}
