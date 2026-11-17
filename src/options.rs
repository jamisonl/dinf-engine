//! Runtime options for a loaded stage.
//!
//! Earlier versions read these choices from `DINF_*` environment variables
//! on every call. They are now ordinary values that a caller sets on an
//! [`EngineBuilder`](crate::EngineBuilder) or a
//! [`PipelineStage`](crate::PipelineStage), so two stages in one process can
//! differ and nothing changes behind the caller's back. The engine reads no
//! environment variables for them.
//!
//! Layer profiling is process-wide because its counters are; see
//! [`architectures::profiling`](crate::architectures::profiling).

use anyhow::{Result, ensure};

/// Largest accepted [`EngineOptions::prefill_chunk_tokens`].
pub const MAX_PREFILL_CHUNK_TOKENS: usize = 4096;

/// Options that change how a loaded stage runs requests.
///
/// The default is the engine's normal behavior.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineOptions {
    /// Speculative decoding policy for requests with an attached draft.
    pub speculative: SpeculativeOptions,
    /// Prompt tokens per prefill chunk in the stage's own generation calls
    /// (`generate` and its variants). `None` uses the default of 512. Must be
    /// between 1 and [`MAX_PREFILL_CHUNK_TOKENS`]. Schedulers that drive
    /// [`GenerationSession`](crate::continuous::GenerationSession) choose
    /// their own prefill budget per step instead.
    pub prefill_chunk_tokens: Option<usize>,
    /// Timing output for the stage's own generation calls.
    pub profile: ProfileOptions,
    /// Run each single-token decode step on CUDA as a few captured graphs
    /// instead of issuing its kernel launches one by one. Results are the
    /// same; only the driver overhead changes. It applies to stages loaded
    /// from GGUF, and has no effect off CUDA, with an attached draft or
    /// while profiling. Default `true`.
    ///
    /// While a step is being recorded, everything issued on the device's
    /// stream is recorded with it, so the engine records only when that is
    /// safe. A stage records only while it is the sole stage loaded on its
    /// GPU; with a second stage there, such as the stages of a local
    /// pipeline, every stage uses direct launches. A stage being loaded
    /// waits for a recording on its GPU to finish. A stage also turns
    /// recording off for itself if a capture ever fails.
    ///
    /// The engine cannot see device work that is not a stage. If another
    /// thread runs its own tensors on the same `Device` while a stage is
    /// generating, set this to `false`.
    pub cuda_graph_decode: bool,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            speculative: SpeculativeOptions::default(),
            prefill_chunk_tokens: None,
            profile: ProfileOptions::default(),
            cuda_graph_decode: true,
        }
    }
}

impl EngineOptions {
    /// Check that every value is in range.
    ///
    /// # Errors
    ///
    /// Returns an error when `prefill_chunk_tokens` is zero or larger than
    /// [`MAX_PREFILL_CHUNK_TOKENS`].
    pub fn validate(&self) -> Result<()> {
        if let Some(tokens) = self.prefill_chunk_tokens {
            ensure!(
                (1..=MAX_PREFILL_CHUNK_TOKENS).contains(&tokens),
                "prefill_chunk_tokens must be between 1 and {MAX_PREFILL_CHUNK_TOKENS}, got {tokens}"
            );
        }
        Ok(())
    }
}

/// Speculative decoding policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeOptions {
    /// Let greedy and seeded requests use speculative decoding.
    ///
    /// Batched target checks can differ numerically from scalar decoding, so
    /// by default a request that must be reproducible (temperature 0 or a
    /// seed) decodes without its draft. Setting this changes those requests'
    /// output. Default `false`.
    pub allow_approximate: bool,
    /// Stop speculating for the rest of a request when it stops paying off:
    /// after the pre-M5 pass limit, or when the adaptive check says scalar
    /// decoding is faster. Default `true`.
    pub fallback: bool,
    /// Base the adaptive check on measured time per token, with occasional
    /// scalar probes, instead of the draft acceptance rate. Requests using it
    /// are not offered shared verification. Default `false`.
    pub elapsed_time_fallback: bool,
    /// Let ready draft lanes share one target verification pass where the
    /// model and draft support it. Not yet qualified for every model.
    /// Default `false`.
    pub shared_verification: bool,
}

impl Default for SpeculativeOptions {
    fn default() -> Self {
        Self {
            allow_approximate: false,
            fallback: true,
            elapsed_time_fallback: false,
            shared_verification: false,
        }
    }
}

/// Timing output for a stage's own generation calls.
///
/// Reports go to standard error.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProfileOptions {
    /// Print forward, penalty, sampling and callback time after each
    /// generation call.
    pub decode: bool,
    /// Wait for the device after each forward pass so `decode` timings
    /// include queued device work. This slows generation.
    pub sync_after_forward: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_engine_without_options() {
        let options = EngineOptions::default();
        assert!(!options.speculative.allow_approximate);
        assert!(options.speculative.fallback);
        assert!(!options.speculative.elapsed_time_fallback);
        assert!(!options.speculative.shared_verification);
        assert_eq!(options.prefill_chunk_tokens, None);
        assert_eq!(options.profile, ProfileOptions::default());
        assert!(options.cuda_graph_decode);
        options.validate().unwrap();
    }

    #[test]
    fn prefill_chunk_tokens_must_be_in_range() {
        for tokens in [1, 512, MAX_PREFILL_CHUNK_TOKENS] {
            let options = EngineOptions {
                prefill_chunk_tokens: Some(tokens),
                ..Default::default()
            };
            options.validate().unwrap();
        }
        for tokens in [0, MAX_PREFILL_CHUNK_TOKENS + 1] {
            let options = EngineOptions {
                prefill_chunk_tokens: Some(tokens),
                ..Default::default()
            };
            assert!(options.validate().is_err());
        }
    }
}
