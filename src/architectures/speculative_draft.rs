//! The block drafters a target stage can host, behind one interface.

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};

use super::ModelArchitecture;
use super::dflash2::{DFlash2Cache, DFlash2Draft};
pub(crate) use super::dspark::{DSparkCache, DSparkDraft, DraftProposal, DraftSampling};

/// A loaded drafter: DFlash 2 (Qwen targets) or DSpark (Inkling 2 Mini).
pub(crate) enum SpeculativeDraft {
    DFlash2(DFlash2Draft),
    DSpark(DSparkDraft),
}

/// Per-request drafter state, matching its [`SpeculativeDraft`].
#[derive(Clone)]
pub(crate) enum SpeculativeCache {
    DFlash2(DFlash2Cache),
    DSpark(DSparkCache),
}

impl From<DFlash2Draft> for SpeculativeDraft {
    fn from(draft: DFlash2Draft) -> Self {
        Self::DFlash2(draft)
    }
}

impl SpeculativeDraft {
    /// Load a `dflash` GGUF as the drafter it holds.
    pub(crate) fn from_gguf(
        path: &Path,
        device: &Device,
        target: &dyn ModelArchitecture,
    ) -> Result<Self> {
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("failed to open draft GGUF: {}", path.display()))?;
        crate::gguf_limits::validate_gguf_header_budget(&mut file)
            .with_context(|| format!("failed to validate draft GGUF: {}", path.display()))?;
        let content = gguf_file::Content::read(&mut file)
            .with_context(|| format!("failed to parse draft GGUF: {}", path.display()))?;
        if super::dspark::is_dspark(&content) {
            return Ok(Self::DSpark(DSparkDraft::from_gguf(path, device, target)?));
        }
        let shared = target
            .speculative_shared_weights()
            .ok_or_else(|| anyhow::anyhow!("{} cannot host a DFlash 2 draft", target.family()))?;
        Ok(Self::DFlash2(DFlash2Draft::from_gguf(
            path, device, shared,
        )?))
    }

    pub(crate) fn target_layers(&self) -> &[usize] {
        match self {
            Self::DFlash2(draft) => draft.target_layers(),
            Self::DSpark(draft) => draft.target_layers(),
        }
    }

    pub(crate) fn block_size(&self) -> usize {
        match self {
            Self::DFlash2(draft) => draft.block_size(),
            Self::DSpark(draft) => draft.block_size(),
        }
    }

    /// Drafted tokens per pass when the stage sets no cap. Both drafters use
    /// their full block; on Metal, DSpark trims each block dynamically by its
    /// Markov-head token probabilities so uncertain tokens verify cheaply while
    /// high-confidence runs commit up to 8 tokens in one pass.
    pub(crate) fn default_draft_tokens(&self) -> Option<usize> {
        match self {
            Self::DFlash2(_) => None,
            #[cfg(target_os = "macos")]
            Self::DSpark(_) => None,
            #[cfg(not(target_os = "macos"))]
            Self::DSpark(_) => Some(2),
        }
    }

    /// Committed tokens per pass, in tenths, under which drafting is slower
    /// than plain decoding: over a recent window and over a whole request.
    /// DFlash 2's Qwen targets need 1.5 and 2.5; a 3-row Inkling 2
    /// verification costs about 1.75 scalar tokens with its drafter.
    pub(crate) fn fallback_floor_tenths(&self) -> (usize, usize) {
        match self {
            Self::DFlash2(_) => (15, 25),
            Self::DSpark(_) => (13, 18),
        }
    }

    /// Whether the drafter can safely propose a shorter final block when fewer
    /// than `block_size` tokens remain in the request.
    pub(crate) fn allows_short_block(&self) -> bool {
        matches!(self, Self::DSpark(_))
    }

    /// Whether this drafter supports GPU-side sampled distributions for exact
    /// Leviathan speculative rejection sampling on positive-temperature decode.
    pub(crate) fn supports_rejection_sampling(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            matches!(self, Self::DSpark(_))
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    pub(crate) fn context_length(&self) -> usize {
        match self {
            Self::DFlash2(draft) => draft.context_length(),
            Self::DSpark(draft) => draft.context_length(),
        }
    }

    pub(crate) fn cache_budget_bytes(&self) -> Result<usize> {
        match self {
            Self::DFlash2(draft) => draft.cache_budget_bytes(),
            Self::DSpark(draft) => draft.cache_budget_bytes(),
        }
    }

    pub(crate) fn new_cache(&self) -> SpeculativeCache {
        match self {
            Self::DFlash2(draft) => SpeculativeCache::DFlash2(draft.new_cache()),
            Self::DSpark(draft) => SpeculativeCache::DSpark(draft.new_cache()),
        }
    }

    /// Feed committed target rows (one tensor per target feature) to the drafter.
    pub(crate) fn inject(
        &self,
        cache: &mut SpeculativeCache,
        features: &[Tensor],
        start_pos: usize,
    ) -> Result<()> {
        match (self, cache) {
            (Self::DFlash2(draft), SpeculativeCache::DFlash2(cache)) => {
                draft.inject(cache, features, start_pos)
            }
            (Self::DSpark(draft), SpeculativeCache::DSpark(cache)) => {
                draft.inject(cache, features, start_pos)
            }
            _ => anyhow::bail!("draft cache belongs to another drafter"),
        }
    }

    /// Propose `block_size - 1` tokens after `anchor`.
    pub(crate) fn draft(
        &self,
        cache: &SpeculativeCache,
        anchor: u32,
        start_pos: usize,
        block_size: usize,
        target: &dyn ModelArchitecture,
    ) -> Result<Vec<u32>> {
        Ok(self
            .draft_with_sampling(cache, anchor, start_pos, block_size, target, None)?
            .tokens)
    }

    /// Propose up to `block_size - 1` tokens after `anchor`, optionally
    /// sampling from the draft's sparse distribution for exact rejection
    /// sampling verification.
    pub(crate) fn draft_with_sampling(
        &self,
        cache: &SpeculativeCache,
        anchor: u32,
        start_pos: usize,
        block_size: usize,
        target: &dyn ModelArchitecture,
        sampling: Option<&DraftSampling>,
    ) -> Result<DraftProposal> {
        match (self, cache) {
            (Self::DFlash2(draft), SpeculativeCache::DFlash2(cache)) => Ok(DraftProposal {
                tokens: draft.draft(cache, anchor, start_pos, block_size)?,
                distributions: None,
            }),
            (Self::DSpark(draft), SpeculativeCache::DSpark(cache)) => {
                draft.draft_with_sampling(cache, anchor, start_pos, block_size, target, sampling)
            }
            _ => anyhow::bail!("draft cache belongs to another drafter"),
        }
    }
}

impl SpeculativeCache {
    pub(crate) fn fork(&self) -> Result<Self> {
        Ok(match self {
            Self::DFlash2(cache) => Self::DFlash2(cache.fork()?),
            Self::DSpark(cache) => Self::DSpark(cache.fork()?),
        })
    }

    pub(crate) fn live_bytes(&self) -> usize {
        match self {
            Self::DFlash2(cache) => cache.live_bytes(),
            Self::DSpark(cache) => cache.live_bytes(),
        }
    }

    pub(crate) fn clear(&mut self) {
        match self {
            Self::DFlash2(cache) => cache.clear(),
            Self::DSpark(cache) => cache.clear(),
        }
    }

    pub(crate) fn next_position(&self) -> usize {
        match self {
            Self::DFlash2(cache) => cache.next_position(),
            Self::DSpark(cache) => cache.next_position(),
        }
    }
}
