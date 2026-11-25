//! Stage-local prompt reuse. Entries never cross a loaded stage or scope.
//!
//! Scopes prefixed with [`RESERVED_PROMPT_SCOPE_PREFIX`] are reserved scopes:
//! they must be registered explicitly before entries can be matched or
//! published, and revoking the scope drops every entry it owns.

use std::collections::VecDeque;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use anyhow::Result;
use candle_core::Tensor;

use crate::{
    KvTokenBudget, PipelineStage, PipelineStageSessionState, TextArithmeticPolicy,
    TextRequestContext,
    architectures::{self, ImageSpanKey},
};

pub(crate) struct PromptEntry {
    scope: String,
    tokens: Vec<u32>,
    image_keys: Vec<ImageSpanKey>,
    draft: bool,
    shared_prefix: bool,
    state: PipelineStageSessionState,
    logits: Option<Tensor>,
    bytes: usize,
    last_used: Instant,
    pending: Option<Weak<()>>,
    // Session accounting already charges attention storage. Snapshots also
    // retain fixed recurrent/draft state, logits, and the token identity.
    extra_allocation_slots: usize,
    budget: Arc<KvTokenBudget>,
}

impl Drop for PromptEntry {
    fn drop(&mut self) {
        self.budget.release_allocated(self.extra_allocation_slots);
    }
}

pub(crate) struct PromptCache {
    pub(crate) entries: VecDeque<PromptEntry>,
    max_bytes: usize,
    automatic_bytes: bool,
    // Caller-chosen byte ceiling for the automatic limit; `None` scales it
    // from admitted KV memory alone.
    byte_ceiling: Option<usize>,
    max_entries: usize,
    ttl: Duration,
    // Idle lifetime of entries in reserved scopes. Their owner revokes them
    // explicitly, so they outlive a pause between turns of one conversation.
    reserved_ttl: Duration,
    // Registered reserved scopes. Only trusted caller code should register
    // these. Removing membership fences active/queued requests without
    // tombstones.
    reserved_scopes: Vec<String>,
}

/// Largest automatic snapshot byte limit, reached only on machines whose
/// admitted KV memory is at least four times this.
const AUTOMATIC_PROMPT_CACHE_CEILING_BYTES: usize = 16 * 1024 * 1024 * 1024;

/// Prefix for prompt-cache scopes that must be registered with
/// [`PipelineStage::register_reserved_prompt_scope`] before use.
pub const RESERVED_PROMPT_SCOPE_PREFIX: &str = "reserved:";

impl Default for PromptCache {
    fn default() -> Self {
        Self {
            entries: VecDeque::new(),
            max_bytes: 512 * 1024 * 1024,
            automatic_bytes: true,
            byte_ceiling: None,
            max_entries: 4,
            ttl: Duration::from_secs(300),
            reserved_ttl: Duration::from_secs(3600),
            reserved_scopes: Vec::new(),
        }
    }
}

impl PromptCache {
    fn ttl_for(&self, scope: &str) -> Duration {
        if scope.starts_with(RESERVED_PROMPT_SCOPE_PREFIX) {
            self.reserved_ttl
        } else {
            self.ttl
        }
    }
}

/// Returns true iff every image span in `query` starting before `cached_len`
/// ends at or before `cached_len` and matches `cached` in exact order and digest.
fn prefix_image_keys_match(
    query: &[ImageSpanKey],
    cached: &[ImageSpanKey],
    cached_len: usize,
) -> bool {
    let mut matched = 0usize;
    for key in query {
        if key.start >= cached_len {
            break;
        }
        if key.start.saturating_add(key.len) > cached_len {
            return false;
        }
        if cached.get(matched) != Some(key) {
            return false;
        }
        matched += 1;
    }
    matched == cached.len()
}

impl PipelineStage {
    /// Register one fresh, in-memory reserved scope. The caller must never
    /// reuse a revoked scope identifier. This registry is bounded independently
    /// of KV.
    pub fn register_reserved_prompt_scope(&mut self, scope: String) -> bool {
        if !scope.starts_with(RESERVED_PROMPT_SCOPE_PREFIX)
            || self.prompt_cache.reserved_scopes.len() >= 4
            || self.prompt_cache.reserved_scopes.contains(&scope)
        {
            return false;
        }
        self.prompt_cache.reserved_scopes.push(scope);
        true
    }

    /// Return whether this stage still has the reserved scope registered.
    pub fn reserved_prompt_scope_registered(&self, scope: &str) -> bool {
        self.prompt_cache
            .reserved_scopes
            .iter()
            .any(|registered| registered == scope)
    }

    fn prompt_scope_allowed(&self, scope: &str) -> bool {
        !scope.starts_with(RESERVED_PROMPT_SCOPE_PREFIX)
            || self.reserved_prompt_scope_registered(scope)
    }

    /// Remove scope membership and pending/published snapshots. This alone is
    /// not a complete purge: the caller must first stop enqueueing and drain
    /// matching active, scheduler-waiting and queued work at a completion
    /// boundary. Membership checks remain a second barrier against publication.
    pub fn revoke_reserved_prompt_scope(&mut self, scope: &str) {
        self.prompt_cache
            .reserved_scopes
            .retain(|registered| registered != scope);
        self.prompt_cache
            .entries
            .retain(|entry| entry.scope != scope);
    }

    /// Inspect published exact-prefix matches without copying GPU state,
    /// refreshing TTL/LRU, or reserving memory. Returns (plain, draft).
    /// Hints can race with eviction; `reuse_prompt` remains authoritative.
    pub fn prompt_cache_match_tokens(&self, scope: &str, tokens: &[u32]) -> (usize, usize) {
        if !self.prompt_scope_allowed(scope) {
            return (0, 0);
        }
        let Ok(context) = TextRequestContext::new(tokens.len()) else {
            return (0, 0);
        };
        let desired = self.desired_text_arithmetic(context);
        let matches = [false, true].map(|draft| {
            self.prompt_cache
                .entries
                .iter()
                .filter(|entry| {
                    entry.pending.is_none()
                        && entry.last_used.elapsed() < self.prompt_cache.ttl_for(scope)
                        && entry.scope == scope
                        && entry.draft == draft
                        && tokens.starts_with(&entry.tokens)
                        && TextArithmeticPolicy::compatible(
                            entry.state.text_arithmetic,
                            desired,
                            entry.tokens.len(),
                        )
                        && (entry.logits.is_some() || tokens.len() > entry.tokens.len())
                        && self.can_fork_prompt_state(&entry.state)
                })
                .max_by_key(|entry| entry.tokens.len())
                .map_or(0, |entry| entry.tokens.len())
        });
        (matches[0], matches[1])
    }

    /// Drop stage-local prompt snapshots without changing active session state.
    pub fn clear_prompt_cache(&mut self) {
        self.prompt_cache.entries.clear();
    }

    /// Bound stage-local prompt snapshots, including draft and recurrent state.
    /// Zero bytes or entries disables reuse. Changing limits drops old entries.
    pub fn set_prompt_cache_limits(&mut self, max_bytes: usize, max_entries: usize, ttl: Duration) {
        self.prompt_cache = PromptCache {
            entries: VecDeque::new(),
            max_bytes,
            automatic_bytes: false,
            byte_ceiling: None,
            max_entries,
            ttl,
            reserved_ttl: ttl,
            reserved_scopes: std::mem::take(&mut self.prompt_cache.reserved_scopes),
        };
    }

    /// Effective snapshot byte ceiling; aggregate allocation admission remains
    /// authoritative for active state, copies and every retained snapshot.
    pub fn prompt_cache_max_bytes(&self) -> usize {
        self.prompt_cache.max_bytes
    }

    /// Choose the snapshot byte ceiling for the automatic limit, or `None`
    /// to scale it from admitted KV memory alone. A chosen ceiling may use up
    /// to half of the admitted resident KV bytes; aggregate allocation
    /// admission still bounds every retained snapshot. Explicit
    /// [`Self::set_prompt_cache_limits`] values are left unchanged.
    pub fn set_prompt_cache_byte_ceiling(&mut self, ceiling: Option<usize>) {
        self.prompt_cache.byte_ceiling = ceiling;
        if !self.prompt_cache.automatic_bytes {
            return;
        }
        match (self.max_total_kv_tokens, ceiling) {
            (Some(resident), _) => self.configure_prompt_cache_headroom(resident),
            (None, Some(ceiling)) => self.prompt_cache.max_bytes = ceiling,
            (None, None) => self.prompt_cache.max_bytes = PromptCache::default().max_bytes,
        }
        while self.prompt_cache_bytes() > self.prompt_cache.max_bytes {
            self.prompt_cache.entries.pop_front();
        }
    }

    pub(crate) fn configure_prompt_cache_headroom(&mut self, resident_tokens: usize) {
        if !self.prompt_cache.automatic_bytes {
            return;
        }
        // Scale with the machine: at most a quarter of admitted resident KV
        // bytes (one eighth of total old/new allocation headroom), up to
        // 16 GiB. A caller-chosen ceiling may use up to half. Never infer free
        // RAM here: the caller's model-memory admission owns that.
        let resident_bytes = resident_tokens
            .saturating_mul(usize::try_from(self.kv_budget_bytes_per_token).unwrap_or(usize::MAX));
        self.prompt_cache.max_bytes = match self.prompt_cache.byte_ceiling {
            Some(ceiling) => ceiling.min(resident_bytes / 2),
            None => (resident_bytes / 4).min(AUTOMATIC_PROMPT_CACHE_CEILING_BYTES),
        };
        while self.prompt_cache_bytes() > self.prompt_cache.max_bytes {
            self.prompt_cache.entries.pop_front();
        }
    }

    /// Expire published snapshots by idle TTL and pending snapshots when their
    /// request ticket dies. Reserved scope membership gates both lifetimes.
    pub fn prune_prompt_cache(&mut self) {
        let (ttl, reserved_ttl) = (self.prompt_cache.ttl, self.prompt_cache.reserved_ttl);
        let scopes = &self.prompt_cache.reserved_scopes;
        self.prompt_cache.entries.retain(|entry| {
            let ttl = if entry.scope.starts_with(RESERVED_PROMPT_SCOPE_PREFIX) {
                reserved_ttl
            } else {
                ttl
            };
            (!entry.scope.starts_with(RESERVED_PROMPT_SCOPE_PREFIX)
                || scopes.iter().any(|scope| scope == &entry.scope))
                && match &entry.pending {
                    Some(ticket) => ticket.strong_count() > 0,
                    None => entry.last_used.elapsed() < ttl,
                }
        });
    }

    /// Bytes retained by prompt snapshots, including main, draft, and logits.
    pub fn prompt_cache_bytes(&self) -> usize {
        self.prompt_cache
            .entries
            .iter()
            .map(|entry| entry.bytes)
            .sum()
    }

    /// Free reusable state before it could force an active request to evict KV.
    pub(crate) fn release_prompt_cache_under_pressure(&mut self, additional_slots: usize) {
        self.prune_prompt_cache();
        if let Some(max) = self.max_total_kv_tokens {
            while !self.prompt_cache.entries.is_empty()
                && self
                    .total_kv_cache_allocation_tokens()
                    .max(self.total_kv_cache_tokens())
                    .saturating_add(additional_slots)
                    > max
            {
                self.prompt_cache.entries.pop_front();
            }
        }
    }

    fn can_fork_prompt_state(&self, state: &PipelineStageSessionState) -> bool {
        self.can_fork_prompt_state_with_embedded(state, false)
    }

    fn can_fork_prompt_state_with_embedded(
        &self,
        state: &PipelineStageSessionState,
        allow_embedded: bool,
    ) -> bool {
        Arc::ptr_eq(&state.kv_token_budget, &self.kv_token_budget)
            && (allow_embedded || !state.has_embedded_prompt)
            && state.mrope_decode_delta.is_none()
            && state.next_position > 0
            && state.next_position == state.kv_cache_tokens
            && self.max_total_kv_tokens.is_none_or(|max| {
                self.total_kv_cache_allocation_tokens()
                    .max(self.total_kv_cache_tokens())
                    .saturating_add(state.kv_cache_tokens)
                    .saturating_add(state.fixed_cache_allocation_tokens)
                    <= max / 2
            })
    }

    fn fork_prompt_state(
        &self,
        state: &PipelineStageSessionState,
    ) -> Result<PipelineStageSessionState> {
        // Evicted snapshots and finished sessions can leave retired Metal
        // buffers in Candle's pool. Free them before copying another state;
        // waiting at the next forward pass comes after these allocations.
        if self.device.is_metal() {
            self.device.synchronize()?;
        }
        let mut fork = self.new_session_state();
        fork.cache = state
            .cache
            .iter()
            .map(|layer| {
                layer
                    .as_ref()
                    .map(architectures::fork_layer_cache)
                    .transpose()
            })
            .collect::<Result<_>>()?;
        fork.speculative_cache = state
            .speculative_cache
            .as_ref()
            .map(|cache| cache.fork())
            .transpose()?;
        let (capacity, allocated) = architectures::kv_cache_allocation_tokens(&fork.cache)?;
        // Charge only after every copy succeeds; Drop releases exactly this charge.
        fork.kv_token_budget.replace(0, state.kv_cache_tokens);
        fork.kv_token_budget
            .replace_allocated(0, allocated + state.fixed_cache_allocation_tokens);
        fork.next_position = state.next_position;
        fork.kv_cache_tokens = state.kv_cache_tokens;
        fork.kv_cache_lane_capacity_tokens = capacity;
        fork.kv_cache_allocation_tokens = allocated;
        fork.fixed_cache_allocation_tokens = state.fixed_cache_allocation_tokens;
        fork.mrope_decode_delta = state.mrope_decode_delta;
        fork.has_embedded_prompt = state.has_embedded_prompt;
        fork.text_arithmetic = state.text_arithmetic;
        Ok(fork)
    }

    pub(crate) fn reuse_prompt(
        &mut self,
        scope: &str,
        tokens: &[u32],
        draft: bool,
    ) -> Result<Option<(PipelineStageSessionState, Option<Tensor>, usize)>> {
        self.prune_prompt_cache();
        if !self.prompt_scope_allowed(scope) {
            return Ok(None);
        }
        let Ok(context) = TextRequestContext::new(tokens.len()) else {
            return Ok(None);
        };
        let desired = self.desired_text_arithmetic(context);
        let index = self
            .prompt_cache
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.pending.is_none()
                    && entry.scope == scope
                    && entry.image_keys.is_empty()
                    && entry.draft == draft
                    && tokens.starts_with(&entry.tokens)
                    && TextArithmeticPolicy::compatible(
                        entry.state.text_arithmetic,
                        desired,
                        entry.tokens.len(),
                    )
                    && (entry.logits.is_some() || tokens.len() > entry.tokens.len())
                    && self.can_fork_prompt_state_with_embedded(&entry.state, false)
            })
            .max_by_key(|(_, entry)| entry.tokens.len())
            .map(|(index, _)| index);
        let Some(index) = index else {
            // A miss costs a full prefill. Report how close the best published
            // snapshot in this scope came, so a diverging prompt can be told
            // apart from an evicted or expired one.
            if let Some(restored) = self.try_restore_from_disk(scope, tokens, &[], draft)? {
                return Ok(Some(restored));
            }
            let published = self.prompt_cache.entries.iter().filter(|entry| {
                entry.pending.is_none() && entry.scope == scope && entry.draft == draft
            });
            let (snapshots, shared_tokens, snapshot_tokens) =
                published.fold((0usize, 0usize, 0usize), |(count, shared, len), entry| {
                    let common = entry
                        .tokens
                        .iter()
                        .zip(tokens)
                        .take_while(|(cached, wanted)| cached == wanted)
                        .count();
                    if count == 0 || common > shared {
                        (count + 1, common, entry.tokens.len())
                    } else {
                        (count + 1, shared, len)
                    }
                });
            tracing::info!(
                prompt_tokens = tokens.len(),
                draft_decoding = draft,
                snapshots,
                shared_tokens,
                snapshot_tokens,
                "Continuous prompt cache miss"
            );
            return Ok(None);
        };
        let entry = &self.prompt_cache.entries[index];
        let state = self.fork_prompt_state(&entry.state)?;
        let logits = entry.logits.clone();
        let reused = entry.tokens.len();
        let mut entry = self
            .prompt_cache
            .entries
            .remove(index)
            .expect("entry exists");
        entry.last_used = Instant::now();
        self.prompt_cache.entries.push_back(entry);
        tracing::info!(
            reused_prompt_tokens = reused,
            draft_decoding = draft,
            "Continuous prompt cache hit"
        );
        Ok(Some((state, logits, reused)))
    }

    /// Reuse the longest cached plain-text prefix strictly before the first
    /// image of a multimodal prompt. When `draft` is true, require a cached
    /// entry with a position-aligned speculative draft cache so post-image
    /// text decode can keep drafting; when `draft` is false, any matching
    /// prefix is eligible and any forked draft cache is dropped.
    pub(crate) fn reuse_prompt_prefix(
        &mut self,
        scope: &str,
        tokens: &[u32],
        draft: bool,
    ) -> Result<Option<(PipelineStageSessionState, usize)>> {
        self.prune_prompt_cache();
        if !self.prompt_scope_allowed(scope) || tokens.is_empty() {
            return Ok(None);
        }
        let Ok(context) = TextRequestContext::new(tokens.len()) else {
            return Ok(None);
        };
        if self.desired_text_arithmetic(context).is_some() {
            return Ok(None);
        }
        let index =
            self.prompt_cache
                .entries
                .iter()
                .enumerate()
                .filter(|(_, entry)| {
                    entry.pending.is_none()
                        && entry.scope == scope
                        && entry.image_keys.is_empty()
                        && entry.state.text_arithmetic.is_none()
                        && tokens.starts_with(&entry.tokens)
                        && self.can_fork_prompt_state_with_embedded(&entry.state, false)
                        && (!draft
                            || (entry.draft
                                && entry.state.speculative_cache.as_ref().is_some_and(|cache| {
                                    cache.next_position() == entry.tokens.len()
                                })))
                })
                .max_by_key(|(_, entry)| (entry.tokens.len(), !entry.draft))
                .map(|(index, _)| index);
        let Some(index) = index else {
            if let Some((state, _logits, reused)) =
                self.try_restore_from_disk(scope, tokens, &[], draft)?
            {
                return Ok(Some((state, reused)));
            }
            return Ok(None);
        };
        let entry = &self.prompt_cache.entries[index];
        let mut state = self.fork_prompt_state(&entry.state)?;
        if !draft {
            state.speculative_cache = None;
        }
        let reused = entry.tokens.len();
        let mut entry = self
            .prompt_cache
            .entries
            .remove(index)
            .expect("entry exists");
        entry.last_used = Instant::now();
        self.prompt_cache.entries.push_back(entry);
        tracing::info!(
            reused_prompt_tokens = reused,
            draft_decoding = draft,
            "Continuous prompt cache hit (multimodal text prefix)"
        );
        Ok(Some((state, reused)))
    }

    /// Reuse the longest cached prefix (plain-text or image-keyed) matching a
    /// 1D-causal multimodal prompt's token IDs and exact image span digests.
    pub(crate) fn reuse_multimodal_prompt(
        &mut self,
        scope: &str,
        tokens: &[u32],
        image_keys: &[ImageSpanKey],
        draft: bool,
    ) -> Result<Option<(PipelineStageSessionState, Option<Tensor>, usize)>> {
        self.prune_prompt_cache();
        if !self.prompt_scope_allowed(scope) || tokens.is_empty() {
            return Ok(None);
        }
        let Ok(context) = TextRequestContext::new(tokens.len()) else {
            return Ok(None);
        };
        if self.desired_text_arithmetic(context).is_some() {
            return Ok(None);
        }
        let index = self
            .prompt_cache
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.pending.is_none()
                    && entry.scope == scope
                    && entry.state.text_arithmetic.is_none()
                    && tokens.starts_with(&entry.tokens)
                    && prefix_image_keys_match(image_keys, &entry.image_keys, entry.tokens.len())
                    && (entry.logits.is_some() || tokens.len() > entry.tokens.len())
                    && self.can_fork_prompt_state_with_embedded(
                        &entry.state,
                        !entry.image_keys.is_empty(),
                    )
                    && (!draft
                        || (entry.draft
                            && entry
                                .state
                                .speculative_cache
                                .as_ref()
                                .is_some_and(|cache| cache.next_position() == entry.tokens.len())))
            })
            .max_by_key(|(_, entry)| (entry.tokens.len(), !entry.draft))
            .map(|(index, _)| index);
        let Some(index) = index else {
            if let Some(restored) = self.try_restore_from_disk(scope, tokens, image_keys, draft)? {
                return Ok(Some(restored));
            }
            if tokens.len() >= 1024 {
                let (snapshots, shared_tokens, snapshot_tokens, image_match) = self
                    .prompt_cache
                    .entries
                    .iter()
                    .filter(|entry| entry.pending.is_none() && entry.scope == scope)
                    .fold(
                        (0usize, 0usize, 0usize, false),
                        |(count, best_shared, best_len, best_img), entry| {
                            let common = entry
                                .tokens
                                .iter()
                                .zip(tokens)
                                .take_while(|(a, b)| a == b)
                                .count();
                            let img_ok = prefix_image_keys_match(
                                image_keys,
                                &entry.image_keys,
                                entry.tokens.len(),
                            );
                            if count == 0 || common > best_shared {
                                (count + 1, common, entry.tokens.len(), img_ok)
                            } else {
                                (count + 1, best_shared, best_len, best_img)
                            }
                        },
                    );
                tracing::info!(
                    prompt_tokens = tokens.len(),
                    prompt_images = image_keys.len(),
                    draft_decoding = draft,
                    snapshots,
                    shared_tokens,
                    snapshot_tokens,
                    image_match,
                    "Continuous prompt cache miss (multimodal)"
                );
            }
            return Ok(None);
        };
        let entry = &self.prompt_cache.entries[index];
        let mut state = self.fork_prompt_state(&entry.state)?;
        if !draft {
            state.speculative_cache = None;
        }
        let logits = entry.logits.clone();
        let reused = entry.tokens.len();
        let reused_images = entry.image_keys.len();
        let mut entry = self
            .prompt_cache
            .entries
            .remove(index)
            .expect("entry exists");
        entry.last_used = Instant::now();
        self.prompt_cache.entries.push_back(entry);
        tracing::info!(
            reused_prompt_tokens = reused,
            reused_images,
            draft_decoding = draft,
            "Continuous prompt cache hit (multimodal)"
        );
        Ok(Some((state, logits, reused)))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture_prompt(
        &mut self,
        scope: &str,
        tokens: &[u32],
        draft: bool,
        state: &PipelineStageSessionState,
        logits: &Tensor,
        shared_prefix: bool,
    ) -> Result<Option<Arc<()>>> {
        self.capture_prompt_with_images(scope, tokens, &[], draft, state, logits, shared_prefix)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture_prompt_with_images(
        &mut self,
        scope: &str,
        tokens: &[u32],
        image_keys: &[ImageSpanKey],
        draft: bool,
        state: &PipelineStageSessionState,
        logits: &Tensor,
        shared_prefix: bool,
    ) -> Result<Option<Arc<()>>> {
        self.prune_prompt_cache();
        let allow_embedded = !image_keys.is_empty();
        if !self.prompt_scope_allowed(scope)
            || self.prompt_cache.max_entries == 0
            || self.prompt_cache.max_bytes == 0
            || !self.can_fork_prompt_state_with_embedded(state, allow_embedded)
            || state.next_position != tokens.len()
            || (draft
                && state
                    .speculative_cache
                    .as_ref()
                    .is_some_and(|cache| cache.next_position() != state.next_position))
        {
            return Ok(None);
        }
        let mut bytes = tokens
            .len()
            .saturating_mul(size_of::<u32>())
            .saturating_add(scope.len())
            .saturating_add(
                logits
                    .elem_count()
                    .saturating_mul(logits.dtype().size_in_bytes()),
            )
            .saturating_add(
                state
                    .speculative_cache
                    .as_ref()
                    .map_or(0, |cache| cache.live_bytes()),
            );
        let mut attention_bytes = 0usize;
        let mut fixed_state_bytes = state
            .speculative_cache
            .as_ref()
            .map_or(0, |cache| cache.live_bytes());
        for layer in state.cache.iter().flatten() {
            // Unsupported cache formats remain a cold prefill, not a request error.
            let Ok(layer_bytes) = architectures::live_layer_cache_bytes(layer) else {
                return Ok(None);
            };
            bytes = bytes.saturating_add(layer_bytes);
            if matches!(layer, architectures::LayerCache::Attention(_)) {
                attention_bytes = attention_bytes.saturating_add(layer_bytes);
            } else {
                fixed_state_bytes = fixed_state_bytes.saturating_add(layer_bytes);
            }
        }
        if bytes > self.prompt_cache.max_bytes {
            if tokens.len() >= 1024 {
                tracing::info!(
                    prompt_tokens = tokens.len(),
                    bytes,
                    max_bytes = self.prompt_cache.max_bytes,
                    "Prompt snapshot rejected: exceeds cache byte limit"
                );
            }
            return Ok(None);
        }
        while self.prompt_cache.entries.len() >= self.prompt_cache.max_entries
            || self.prompt_cache_bytes().saturating_add(bytes) > self.prompt_cache.max_bytes
        {
            // Do not let a one-off question evict its shared system/tool prefix.
            // Qwen may only fit one of these snapshots under the byte limit.
            let victim = self.prompt_cache.entries.iter().position(|entry| {
                shared_prefix
                    || !entry.shared_prefix
                    || entry.scope != scope
                    || entry.draft != draft
                    || !tokens.starts_with(&entry.tokens)
            });
            let Some(victim) = victim else {
                if tokens.len() >= 1024 {
                    tracing::info!(
                        prompt_tokens = tokens.len(),
                        bytes,
                        max_bytes = self.prompt_cache.max_bytes,
                        "Prompt snapshot rejected: no evictable victim"
                    );
                }
                return Ok(None);
            };
            self.prompt_cache.entries.remove(victim);
        }
        let extra_allocation_slots = bytes
            .saturating_sub(attention_bytes)
            .saturating_sub(fixed_state_bytes)
            .div_ceil(
                usize::try_from(self.kv_budget_bytes_per_token)
                    .unwrap_or(usize::MAX)
                    .max(1),
            );
        if self.max_total_kv_tokens.is_some_and(|max| {
            self.total_kv_cache_allocation_tokens()
                .saturating_add(state.kv_cache_tokens)
                .saturating_add(state.fixed_cache_allocation_tokens)
                .saturating_add(extra_allocation_slots)
                > max / 2
        }) {
            if tokens.len() >= 1024 {
                tracing::info!(
                    prompt_tokens = tokens.len(),
                    allocated = self.total_kv_cache_allocation_tokens(),
                    kv_cache_tokens = state.kv_cache_tokens,
                    max_total_kv_tokens = self.max_total_kv_tokens.unwrap_or(0),
                    "Prompt snapshot rejected: exceeds half of KV token budget"
                );
            }
            return Ok(None);
        }
        let snapshot = self.fork_prompt_state(state)?;
        let logits = architectures::copy_persistent_tensor(logits)?;
        // Control cannot interleave with this exclusive stage borrow. Keep the
        // fence at the actual insert as well as at capture admission.
        if !self.prompt_scope_allowed(scope) {
            return Ok(None);
        }
        self.maybe_persist_snapshot_to_disk(scope, draft, tokens, image_keys, &snapshot, &logits);
        let ticket = Arc::new(());
        self.kv_token_budget
            .replace_allocated(0, extra_allocation_slots);
        self.prompt_cache.entries.push_back(PromptEntry {
            scope: scope.to_owned(),
            tokens: tokens.to_vec(),
            image_keys: image_keys.to_vec(),
            draft,
            shared_prefix,
            state: snapshot,
            logits: Some(logits),
            bytes,
            last_used: Instant::now(),
            pending: Some(Arc::downgrade(&ticket)),
            extra_allocation_slots,
            budget: self.kv_token_budget.clone(),
        });
        Ok(Some(ticket))
    }

    fn disk_cache_model_key(&self) -> String {
        format!(
            "{}_l{}_h{}_v{}_d{}",
            self.family(),
            self.cache.len(),
            self.hidden_dim,
            self.vocab_size,
            usize::from(self.speculative.is_some()),
        )
    }

    fn maybe_persist_snapshot_to_disk(
        &self,
        scope: &str,
        draft: bool,
        tokens: &[u32],
        image_keys: &[ImageSpanKey],
        snapshot: &PipelineStageSessionState,
        logits: &Tensor,
    ) {
        if tokens.len() < crate::disk_kv_cache::min_tokens() {
            return;
        }
        let Some(dir) = crate::disk_kv_cache::cache_dir(&self.disk_cache_model_key()) else {
            return;
        };
        if let Err(error) = crate::disk_kv_cache::save_snapshot_async(
            dir,
            crate::disk_kv_cache::max_disk_bytes(),
            scope.to_owned(),
            draft,
            tokens.to_vec(),
            image_keys.to_vec(),
            &snapshot.cache,
            snapshot.speculative_cache.as_ref(),
            logits,
            snapshot.next_position,
            snapshot.kv_cache_tokens,
            snapshot.kv_cache_lane_capacity_tokens,
            snapshot.fixed_cache_allocation_tokens,
            snapshot.has_embedded_prompt,
        ) {
            tracing::warn!(%error, "Could not schedule disk KV snapshot save");
        }
    }

    fn try_restore_from_disk(
        &mut self,
        scope: &str,
        tokens: &[u32],
        image_keys: &[ImageSpanKey],
        draft: bool,
    ) -> Result<Option<(PipelineStageSessionState, Option<Tensor>, usize)>> {
        if tokens.len() < crate::disk_kv_cache::min_tokens() {
            return Ok(None);
        }
        let Some(dir) = crate::disk_kv_cache::cache_dir(&self.disk_cache_model_key()) else {
            return Ok(None);
        };
        let Some(restored) = crate::disk_kv_cache::load_best_snapshot(
            &dir,
            scope,
            tokens,
            image_keys,
            draft,
            &self.device,
        )?
        else {
            return Ok(None);
        };
        let reused = restored.tokens.len();
        let mut state = self.new_session_state();
        let kv_alloc = restored
            .kv_cache_lane_capacity_tokens
            .max(restored.kv_cache_tokens);
        self.kv_token_budget.replace(0, restored.kv_cache_tokens);
        self.kv_token_budget.replace_allocated(
            0,
            kv_alloc.saturating_add(restored.fixed_cache_allocation_tokens),
        );
        state.cache = restored.cache;
        state.speculative_cache = restored.speculative_cache;
        state.next_position = restored.next_position;
        state.kv_cache_tokens = restored.kv_cache_tokens;
        state.kv_cache_lane_capacity_tokens = restored.kv_cache_lane_capacity_tokens;
        state.kv_cache_allocation_tokens = kv_alloc;
        state.fixed_cache_allocation_tokens = restored.fixed_cache_allocation_tokens;
        state.has_embedded_prompt = restored.has_embedded_prompt;
        Ok(Some((state, Some(restored.logits), reused)))
    }

    /// Transfer a fenced input boundary for an explicit caller cancellation in
    /// a reserved scope, never a generated tail. The session validates
    /// provenance; this layer rechecks registration, admission and charges.
    pub(crate) fn retain_cancelled_input(
        &mut self,
        scope: &str,
        tokens: &[u32],
        draft: bool,
        state: &mut PipelineStageSessionState,
    ) -> bool {
        self.prune_prompt_cache();
        if !self.reserved_prompt_scope_registered(scope)
            || self.prompt_cache.max_entries == 0
            || self.prompt_cache.max_bytes == 0
            || state.resume_invalidated
            || state.speculative_cache.is_some()
            || !self.can_fork_prompt_state(state)
            || state.next_position != tokens.len()
        {
            return false;
        }
        let metadata_bytes = tokens
            .len()
            .saturating_mul(size_of::<u32>())
            .saturating_add(scope.len());
        let mut bytes = metadata_bytes;
        for layer in state.cache.iter().flatten() {
            let Ok(live) = architectures::live_layer_cache_bytes(layer) else {
                return false;
            };
            bytes = bytes.saturating_add(live);
        }
        if bytes > self.prompt_cache.max_bytes {
            return false;
        }
        while self.prompt_cache.entries.len() >= self.prompt_cache.max_entries
            || self.prompt_cache_bytes().saturating_add(bytes) > self.prompt_cache.max_bytes
        {
            let victim = self.prompt_cache.entries.iter().position(|entry| {
                !entry.shared_prefix
                    || entry.scope != scope
                    || entry.draft != draft
                    || !tokens.starts_with(&entry.tokens)
            });
            let Some(victim) = victim else { return false };
            self.prompt_cache.entries.remove(victim);
        }
        let extra_allocation_slots = metadata_bytes.div_ceil(
            usize::try_from(self.kv_budget_bytes_per_token)
                .unwrap_or(usize::MAX)
                .max(1),
        );
        // KV and fixed-state charges move with the state. Only identity is new.
        if self.max_total_kv_tokens.is_some_and(|max| {
            self.total_kv_cache_allocation_tokens()
                .saturating_add(extra_allocation_slots)
                > max / 2
        }) || !self.reserved_prompt_scope_registered(scope)
        {
            return false;
        }
        let snapshot = std::mem::replace(state, self.new_session_state());
        self.kv_token_budget
            .replace_allocated(0, extra_allocation_slots);
        self.prompt_cache.entries.push_back(PromptEntry {
            scope: scope.to_owned(),
            tokens: tokens.to_vec(),
            image_keys: Vec::new(),
            draft,
            shared_prefix: false,
            state: snapshot,
            logits: None,
            bytes,
            last_used: Instant::now(),
            pending: None,
            extra_allocation_slots,
            budget: self.kv_token_budget.clone(),
        });
        true
    }

    pub(crate) fn publish_prompt(&mut self, ticket: &Arc<()>) -> bool {
        self.prune_prompt_cache();
        for entry in &mut self.prompt_cache.entries {
            if entry
                .pending
                .as_ref()
                .is_some_and(|pending| pending.ptr_eq(&Arc::downgrade(ticket)))
            {
                entry.pending = None;
                entry.last_used = Instant::now();
                tracing::debug!(
                    prompt_tokens = entry.tokens.len(),
                    bytes = entry.bytes,
                    "Retained continuous prompt cache"
                );
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architectures::{KvCache, LayerCache, StageInput, StageOutput};
    use crate::continuous::GenerationSession;
    use crate::sampling::SamplingParams;
    use crate::session_state_tests::{preallocated_test_stage, test_stage};

    include!("prompt_cache_model_audit_tests.rs");

    #[test]
    fn automatic_snapshot_admission_tracks_bounded_headroom_and_explicit_disable() -> Result<()> {
        use candle_core::{DType, Device};
        let mut stage = test_stage();
        stage.kv_budget_bytes_per_token = 1;
        stage.vocab_size = 256;
        stage.set_max_total_kv_tokens(8192)?;
        let mut state = stage.new_session_state();
        let row = Tensor::ones((1, 1, 1, 1), DType::F16, &Device::Cpu)?;
        state.cache[0] = Some(LayerCache::Attention(KvCache::F16 {
            k: row.clone(),
            v: row,
        }));
        state.next_position = 1;
        state.kv_cache_tokens = 1;
        state.kv_cache_lane_capacity_tokens = 1;
        state.kv_cache_allocation_tokens = 1;
        state.kv_token_budget.add(1);
        state.kv_token_budget.replace_allocated(0, 1);
        let logits = Tensor::zeros(256, DType::F32, &Device::Cpu)?;
        // Slot headroom alone fits this snapshot. Its bytes exceed the share
        // reserved for optional snapshots on the smaller admitted stage.
        assert!(stage.can_fork_prompt_state(&state));
        assert!(
            stage
                .capture_prompt("owner", &[0], false, &state, &logits, false)?
                .is_none()
        );
        stage.set_max_total_kv_tokens(32768)?;
        let ticket = stage
            .capture_prompt("owner", &[0], false, &state, &logits, false)?
            .expect("larger admitted headroom permits the same snapshot");
        stage.publish_prompt(&ticket);
        assert_eq!(stage.prompt_cache_match_tokens("owner", &[0]), (1, 0));
        assert!(stage.prompt_cache_bytes() <= stage.prompt_cache_max_bytes());
        stage.set_prompt_cache_limits(0, 4, Duration::from_secs(60));
        stage.set_max_total_kv_tokens(65536)?;
        assert!(
            stage
                .capture_prompt("owner", &[0], false, &state, &logits, false)?
                .is_none()
        );
        stage.clear_session_state(&mut state);
        assert_eq!(stage.prompt_cache_bytes(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn audit_prompt_snapshot_logits_use_exact_private_storage() -> Result<()> {
        use candle_core::{DType, Tensor};
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let mut stage = test_stage();
        stage.device = device.clone();
        stage.vocab_size = 3;
        stage.set_prompt_cache_limits(25, 1, Duration::from_secs(60));
        let mut state = stage.new_session_state();
        let row = Tensor::ones((1, 1, 1, 1), DType::F16, &device)?;
        state.cache[0] = Some(LayerCache::Attention(KvCache::F16 {
            k: row.clone(),
            v: row,
        }));
        state.next_position = 1;
        state.kv_cache_tokens = 1;
        state.kv_cache_lane_capacity_tokens = 1;
        state.kv_cache_allocation_tokens = 1;
        state.kv_token_budget.add(1);
        state.kv_token_budget.replace_allocated(0, 1);
        let logits = Tensor::new(&[0.0f32, 1.0, -1.0], &device)?;
        let ticket = stage
            .capture_prompt("owner", &[0], false, &state, &logits, false)?
            .expect("small snapshot fits its byte limit");
        {
            let entry = stage.prompt_cache.entries.back().unwrap();
            let (storage, _) = entry.logits.as_ref().unwrap().storage_and_layout();
            let candle_core::Storage::Metal(storage) = &*storage else {
                panic!("expected Metal logits");
            };
            assert_eq!(
                storage.buffer().length(),
                12,
                "snapshot must not retain pooled padding"
            );
            assert!(entry.bytes >= storage.buffer().length() + 4 + "owner".len());
            assert!(entry.bytes <= 25);
            assert_eq!(
                entry.logits.as_ref().unwrap().to_vec1::<f32>()?,
                [0.0, 1.0, -1.0]
            );
        }
        stage.publish_prompt(&ticket);
        stage.clear_prompt_cache();
        stage.clear_session_state(&mut state);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        device.synchronize()?;
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn prompt_fork_releases_retired_metal_buffers_before_copying() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let mut stage = test_stage();
        stage.device = device;
        // No model forward is needed: this isolates allocation ordering at
        // the cache-copy boundary, including an empty state.
        let state = stage.new_session_state();
        let buffer = stage
            .device
            .as_metal_device()?
            .new_buffer_with_data(&[0u8; 4096])?;
        let retired = Arc::downgrade(&buffer);
        drop(buffer);
        assert!(retired.upgrade().is_some());
        let fork = stage.fork_prompt_state(&state)?;
        assert!(retired.upgrade().is_none());
        assert_eq!(fork.next_position, state.next_position);
        Ok(())
    }

    #[test]
    fn automatic_byte_limit_scales_with_admitted_kv_and_honors_a_chosen_ceiling() -> Result<()> {
        const GIB: usize = 1024 * 1024 * 1024;
        let mut stage = test_stage();
        stage.kv_budget_bytes_per_token = 1024 * 1024;
        // 8 GiB resident: a quarter of it.
        stage.set_max_total_kv_tokens(2 * 8 * 1024)?;
        assert_eq!(stage.prompt_cache_max_bytes(), 2 * GIB);
        // 128 GiB resident: the automatic ceiling binds.
        stage.set_max_total_kv_tokens(2 * 128 * 1024)?;
        assert_eq!(stage.prompt_cache_max_bytes(), 16 * GIB);
        // A chosen ceiling replaces it, up to half of resident memory.
        stage.set_prompt_cache_byte_ceiling(Some(GIB));
        assert_eq!(stage.prompt_cache_max_bytes(), GIB);
        stage.set_prompt_cache_byte_ceiling(Some(1024 * GIB));
        assert_eq!(stage.prompt_cache_max_bytes(), 64 * GIB);
        stage.set_max_total_kv_tokens(2 * 8 * 1024)?;
        assert_eq!(stage.prompt_cache_max_bytes(), 4 * GIB);
        stage.set_prompt_cache_byte_ceiling(None);
        assert_eq!(stage.prompt_cache_max_bytes(), 2 * GIB);
        // Explicit limits are not automatic and stay as set.
        stage.set_prompt_cache_limits(7, 4, Duration::from_secs(60));
        stage.set_prompt_cache_byte_ceiling(Some(GIB));
        assert_eq!(stage.prompt_cache_max_bytes(), 7);
        Ok(())
    }

    #[test]
    fn reserved_scope_snapshots_outlive_the_shared_idle_ttl() -> Result<()> {
        let mut stage = test_stage();
        let owned = "reserved:owned";
        assert!(stage.register_reserved_prompt_scope(owned.into()));
        for scope in [owned, "pool"] {
            let mut cold = generation(&stage, scope, &[0, 1]);
            finish(&mut stage, &mut cold)?;
        }
        let idle = |stage: &mut PipelineStage, by: Duration| {
            for entry in &mut stage.prompt_cache.entries {
                entry.last_used = Instant::now() - by;
            }
        };
        // A pause longer than the shared TTL keeps the owned conversation warm.
        let shared_ttl = stage.prompt_cache.ttl;
        idle(&mut stage, shared_ttl + Duration::from_secs(1));
        assert_eq!(stage.prompt_cache_match_tokens("pool", &[0, 1, 1]), (0, 0));
        assert_eq!(stage.prompt_cache_match_tokens(owned, &[0, 1, 1]), (0, 2));
        assert!(stage.reuse_prompt("pool", &[0, 1, 1], true)?.is_none());
        assert!(stage.reuse_prompt(owned, &[0, 1, 1], true)?.is_some());
        let reserved_ttl = stage.prompt_cache.reserved_ttl;
        idle(&mut stage, reserved_ttl + Duration::from_secs(1));
        assert_eq!(stage.prompt_cache_match_tokens(owned, &[0, 1, 1]), (0, 0));
        assert!(stage.reuse_prompt(owned, &[0, 1, 1], true)?.is_none());
        assert_eq!(stage.prompt_cache_bytes(), 0);
        Ok(())
    }

    fn generation(stage: &PipelineStage, scope: &str, tokens: &[u32]) -> GenerationSession {
        let mut generation = GenerationSession::new(
            stage,
            tokens.to_vec(),
            SamplingParams {
                temperature: 0.0,
                max_tokens: 3,
                seed: Some(42),
                ..SamplingParams::default()
            },
            vec![],
        );
        generation.enable_prompt_cache(scope.into());
        generation
    }

    fn finish(stage: &mut PipelineStage, generation: &mut GenerationSession) -> Result<()> {
        while !generation.is_finished() {
            generation.step(stage)?;
        }
        generation.publish_prompt_cache(stage);
        generation.discard_cache(stage);
        Ok(())
    }

    #[test]
    fn prompt_lookup_falls_back_to_longest_prefix_that_fits_fork_headroom() -> Result<()> {
        let mut stage = crate::session_state_tests::recurrent_stage(false);
        for prompt in [&[0, 1][..], &[0, 1, 0, 1][..]] {
            let mut session = generation(&stage, "alice", prompt);
            finish(&mut stage, &mut session)?;
        }
        assert_eq!(stage.prompt_cache.entries.len(), 2);
        let short = &stage.prompt_cache.entries[0].state;
        let available = stage
            .total_kv_cache_allocation_tokens()
            .max(stage.total_kv_cache_tokens())
            + short.kv_cache_tokens
            + short.fixed_cache_allocation_tokens;
        stage.max_total_kv_tokens = Some(available * 2);
        assert!(stage.can_fork_prompt_state(&stage.prompt_cache.entries[0].state));
        assert!(!stage.can_fork_prompt_state(&stage.prompt_cache.entries[1].state));
        let prompt = [0, 1, 0, 1, 0];
        assert_eq!(stage.prompt_cache_match_tokens("alice", &prompt), (0, 2));
        let (mut warm, _, reused) = stage
            .reuse_prompt("alice", &prompt, true)?
            .expect("a smaller matching snapshot fits the existing budget");
        assert_eq!(reused, 2);
        // Remove the artificial headroom boundary before comparing actual
        // continuation logits; lookup must not change recurrence or RNG state.
        stage.max_total_kv_tokens = None;
        let StageOutput::Logits(warm_logits) = stage.prefill_with_session(
            &mut warm,
            StageInput::Tokens(prompt[reused..].to_vec()),
            prompt.len() - reused,
            reused,
        )?
        else {
            unreachable!()
        };
        let mut cold = stage.new_session_state();
        let StageOutput::Logits(cold_logits) = stage.prefill_with_session(
            &mut cold,
            StageInput::Tokens(prompt.to_vec()),
            prompt.len(),
            0,
        )?
        else {
            unreachable!()
        };
        assert_eq!(warm_logits.to_vec1::<f32>()?, cold_logits.to_vec1::<f32>()?);
        eprintln!(
            "headroom fallback proof: cached=2,4 reused={reused} new_prefill=3 logits={:?}",
            warm_logits.to_vec1::<f32>()?
        );
        drop((warm, cold));
        stage.clear_prompt_cache();
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn reserved_prompt_revoke_purges_pending_and_published_without_republication() -> Result<()> {
        let mut stage = test_stage();
        let alice = "reserved:alice";
        let bob = "reserved:bob";
        assert!(stage.register_reserved_prompt_scope(alice.into()));
        assert!(stage.register_reserved_prompt_scope(bob.into()));
        let mut cold = generation(&stage, alice, &[0, 1]);
        finish(&mut stage, &mut cold)?;
        let mut warm = generation(&stage, alice, &[0, 1]);
        finish(&mut stage, &mut warm)?;
        assert_eq!(warm.reused_prompt_tokens(), 2);
        assert_eq!(warm.generated(), cold.generated());
        for (scope, tokens, draft) in [
            (bob, vec![0, 1], true),
            (alice, vec![1, 1], true),
            (alice, vec![0], true),
            (alice, vec![0, 1], false),
        ] {
            assert!(stage.reuse_prompt(scope, &tokens, draft)?.is_none());
        }
        assert!(test_stage().reuse_prompt(alice, &[0, 1], true)?.is_none());
        let mut unrelated = generation(&stage, bob, &[0, 0]);
        finish(&mut stage, &mut unrelated)?;
        // One active request has a pending snapshot; another was queued before
        // revocation and will only reach its first capture after acknowledgment.
        let mut active = generation(&stage, alice, &[1, 0]);
        active.step(&mut stage)?;
        let mut queued = generation(&stage, alice, &[1, 1]);
        let expired = Instant::now() - stage.prompt_cache.ttl - Duration::from_secs(1);
        for entry in &mut stage.prompt_cache.entries {
            if entry.scope == alice && entry.pending.is_some() {
                entry.last_used = expired;
            }
        }
        stage.prune_prompt_cache();
        assert!(
            stage
                .prompt_cache
                .entries
                .iter()
                .any(|entry| entry.scope == alice && entry.pending.is_some())
        );
        let bob_bytes: usize = stage
            .prompt_cache
            .entries
            .iter()
            .filter(|entry| entry.scope == bob)
            .map(|entry| entry.bytes)
            .sum();
        assert!(bob_bytes > 0);
        let allocated = stage.total_kv_cache_allocation_tokens();
        stage.revoke_reserved_prompt_scope(alice);
        assert!(!stage.reserved_prompt_scope_registered(alice));
        assert!(
            stage
                .prompt_cache
                .entries
                .iter()
                .all(|entry| entry.scope != alice)
        );
        assert_eq!(stage.prompt_cache_bytes(), bob_bytes);
        assert!(stage.total_kv_cache_allocation_tokens() < allocated);
        assert!(stage.reuse_prompt(bob, &[0, 0], true)?.is_some());
        finish(&mut stage, &mut active)?;
        finish(&mut stage, &mut queued)?;
        assert_eq!(queued.reused_prompt_tokens(), 0);
        assert!(
            stage
                .prompt_cache
                .entries
                .iter()
                .all(|entry| entry.scope != alice)
        );
        assert_eq!(stage.prompt_cache_bytes(), bob_bytes);
        stage.revoke_reserved_prompt_scope(bob);
        assert_eq!(stage.prompt_cache_bytes(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn reserved_prompt_drain_releases_owned_kv_tickets_and_preserves_active_peer() -> Result<()> {
        let mut stage = test_stage();
        let alice = "reserved:alice";
        let bob = "reserved:bob";
        assert!(stage.register_reserved_prompt_scope(alice.into()));
        assert!(stage.register_reserved_prompt_scope(bob.into()));
        let mut peer = generation(&stage, bob, &[0, 0]);
        peer.step(&mut stage)?;
        let peer_allocation = stage.total_kv_cache_allocation_tokens();
        let peer_bytes = stage.prompt_cache_bytes();
        let peer_position = peer.next_position(&stage);
        let peer_output = peer.generated().to_vec();
        let mut completed = generation(&stage, alice, &[0, 1]);
        finish(&mut stage, &mut completed)?;
        let mut active = generation(&stage, alice, &[1, 0]);
        active.step(&mut stage)?;
        let mut waiting = generation(&stage, alice, &[1, 1]);
        waiting.step(&mut stage)?;
        assert!(stage.total_kv_cache_allocation_tokens() > peer_allocation);
        // Runtime revoke drains prepared lanes/tickets before removing scope
        // snapshots. This order must not prune another active owner's ticket.
        active.discard_cache(&mut stage);
        waiting.discard_cache(&mut stage);
        stage.revoke_reserved_prompt_scope(alice);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), peer_allocation);
        assert_eq!(stage.prompt_cache_bytes(), peer_bytes);
        assert_eq!(peer.next_position(&stage), peer_position);
        assert_eq!(peer.generated(), peer_output.as_slice());
        assert!(
            stage
                .prompt_cache
                .entries
                .iter()
                .all(|entry| entry.scope == bob)
        );
        assert!(
            stage
                .prompt_cache
                .entries
                .iter()
                .any(|entry| entry.pending.is_some())
        );
        active.publish_prompt_cache(&mut stage);
        waiting.publish_prompt_cache(&mut stage);
        assert!(
            stage
                .prompt_cache
                .entries
                .iter()
                .all(|entry| entry.scope == bob)
        );
        finish(&mut stage, &mut peer)?;
        assert!(stage.reuse_prompt(bob, &[0, 0], true)?.is_some());
        stage.revoke_reserved_prompt_scope(bob);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn reserved_prompt_scopes_are_bounded_and_do_not_outlive_revocation() {
        let mut stage = test_stage();
        for index in 0..4 {
            assert!(stage.register_reserved_prompt_scope(format!("reserved:{index}")));
        }
        assert!(!stage.register_reserved_prompt_scope("reserved:overflow".into()));
        stage.revoke_reserved_prompt_scope("reserved:1");
        assert!(!stage.reserved_prompt_scope_registered("reserved:1"));
        assert!(stage.register_reserved_prompt_scope("reserved:fresh".into()));
        assert!(!stage.reserved_prompt_scope_registered("reserved:1"));
        assert!(stage.reserved_prompt_scope_registered("reserved:0"));
    }

    #[test]
    fn prompt_cache_requires_success_and_trusted_scope() -> Result<()> {
        let mut stage = test_stage();
        let mut first = generation(&stage, "alice", &[0, 1]);
        first.step(&mut stage)?;
        assert!(stage.reuse_prompt("alice", &[0, 1], true)?.is_none());
        first.publish_prompt_cache(&mut stage); // unfinished: no publication
        assert!(stage.reuse_prompt("alice", &[0, 1], true)?.is_none());
        finish(&mut stage, &mut first)?;
        for (scope, tokens, draft) in [
            ("bob", vec![0, 1], true),
            ("alice", vec![1, 1], true),
            ("alice", vec![0], true),
            ("alice", vec![0, 1], false),
        ] {
            assert!(stage.reuse_prompt(scope, &tokens, draft)?.is_none());
        }
        let mut repeat = generation(&stage, "alice", &[0, 1]);
        repeat.step(&mut stage)?;
        assert_eq!(repeat.reused_prompt_tokens(), 2);
        assert_eq!(repeat.generated(), &first.generated()[..1]);
        finish(&mut stage, &mut repeat)?;
        assert_eq!(first.generated(), repeat.generated());
        assert!(test_stage().reuse_prompt("alice", &[0, 1], true)?.is_none());
        stage.clear_cache();
        assert_eq!(stage.prompt_cache_bytes(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn cache_probe_only_reads_live_published_owner_and_mode_matches() -> Result<()> {
        let mut stage = test_stage();
        let mut first = generation(&stage, "alice", &[0, 1]);
        first.step(&mut stage)?;
        assert_eq!(stage.prompt_cache_match_tokens("alice", &[0, 1]), (0, 0));
        finish(&mut stage, &mut first)?;
        let before_bytes = stage.prompt_cache_bytes();
        let before_allocation = stage.total_kv_cache_allocation_tokens();
        let before_use = stage.prompt_cache.entries.back().unwrap().last_used;
        assert_eq!(stage.prompt_cache_match_tokens("alice", &[0, 1, 1]), (0, 2));
        for (scope, tokens) in [
            ("bob", vec![0, 1]),
            ("alice", vec![1, 1]),
            ("alice", vec![0]),
        ] {
            assert_eq!(stage.prompt_cache_match_tokens(scope, &tokens), (0, 0));
        }
        assert_eq!(stage.prompt_cache_bytes(), before_bytes);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), before_allocation);
        assert_eq!(
            stage.prompt_cache.entries.back().unwrap().last_used,
            before_use
        );
        let max = stage.max_total_kv_tokens;
        stage.max_total_kv_tokens = Some(1);
        assert_eq!(stage.prompt_cache_match_tokens("alice", &[0, 1]), (0, 0));
        stage.max_total_kv_tokens = max;
        stage.prompt_cache.entries.back_mut().unwrap().last_used =
            Instant::now() - stage.prompt_cache.ttl;
        assert_eq!(stage.prompt_cache_match_tokens("alice", &[0, 1]), (0, 0));
        stage.clear_prompt_cache();
        assert_eq!(stage.prompt_cache_match_tokens("alice", &[0, 1]), (0, 0));
        Ok(())
    }

    #[test]
    fn prompt_cache_carries_arithmetic_and_rejects_a_wrong_policy() -> Result<()> {
        let mut stage = test_stage();
        let prompt = [0, 1, 0, 1, 0, 1, 0, 1];
        let mut first = generation(&stage, "alice", &prompt);
        finish(&mut stage, &mut first)?;
        let mut warm = generation(&stage, "alice", &prompt);
        finish(&mut stage, &mut warm)?;
        assert_eq!(warm.reused_prompt_tokens(), prompt.len());
        assert_eq!(warm.generated(), first.generated());

        // The toy stage uses the ordinary policy. A snapshot from a Phi
        // arithmetic context cannot become an ordinary-policy cache hit.
        let policy = Some(TextArithmeticPolicy::phi_k(TextRequestContext::new(8)?));
        stage
            .prompt_cache
            .entries
            .back_mut()
            .unwrap()
            .state
            .text_arithmetic = policy;
        let fork = stage.fork_prompt_state(&stage.prompt_cache.entries.back().unwrap().state)?;
        assert_eq!(fork.text_arithmetic, policy);
        drop(fork);
        let bytes = stage.prompt_cache_bytes();
        let allocation = stage.total_kv_cache_allocation_tokens();
        assert_eq!(stage.prompt_cache_match_tokens("alice", &prompt), (0, 0));
        assert!(stage.reuse_prompt("alice", &prompt, true)?.is_none());
        assert_eq!(stage.prompt_cache_bytes(), bytes);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), allocation);

        let mut cold = generation(&stage, "alice", &prompt);
        finish(&mut stage, &mut cold)?;
        assert_eq!(cold.reused_prompt_tokens(), 0);
        assert_eq!(cold.generated(), first.generated());
        stage.clear_cache();
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn prompt_cache_reuses_only_rows_with_compatible_request_arithmetic() -> Result<()> {
        let mut stage = crate::session_state_tests::arithmetic_test_stage();
        let prompt = vec![0; 137];
        let mut first = generation(&stage, "alice", &prompt);
        first.set_shared_prompt_prefix(113);
        finish(&mut stage, &mut first)?;
        let mut warm = generation(&stage, "alice", &prompt);
        finish(&mut stage, &mut warm)?;
        assert_eq!(warm.reused_prompt_tokens(), 137);
        assert_eq!(warm.generated(), first.generated());

        // Row 136 used M1 under the old request. Only the shorter snapshot
        // remains valid when the new request moves that row into the M4 span.
        let longer = vec![0; 140];
        assert_eq!(stage.prompt_cache_match_tokens("alice", &longer), (0, 113));
        let mut partial = generation(&stage, "alice", &longer);
        assert!(partial.step_with_prefill_budget(&mut stage, 1)?.is_none());
        assert_eq!(partial.reused_prompt_tokens(), 113);
        assert_eq!(partial.next_position(&stage), 114);
        let mut state = partial.into_cache();
        assert_eq!(keys(&state)?, vec![0.; 114]);
        assert_eq!(
            state.text_arithmetic,
            Some(TextArithmeticPolicy::phi_k(TextRequestContext::new(140)?))
        );
        stage.clear_session_state(&mut state);
        stage.clear_prompt_cache();

        // A decoded suffix also belongs to the old request's arithmetic.
        // Matching token IDs do not permit skipping those incompatible rows.
        let old = TextRequestContext::new(137)?;
        let mut decoded = stage.new_session_state();
        stage.prefill_text_with_session(&mut decoded, StageInput::Tokens(prompt), 137, 0, old)?;
        stage.forward_text_with_session(&mut decoded, StageInput::Tokens(vec![0]), 1, 137, old)?;
        assert_eq!(keys(&decoded)?, [vec![0.; 136], vec![1.; 2]].concat());
        let mut cold = GenerationSession::with_cache(
            decoded,
            longer,
            SamplingParams {
                temperature: 0.0,
                max_tokens: 3,
                seed: Some(42),
                ..SamplingParams::default()
            },
            vec![],
            138,
        );
        cold.enable_prompt_cache("alice".into());
        assert!(cold.step_with_prefill_budget(&mut stage, 1)?.is_none());
        assert_eq!(cold.reused_prompt_tokens(), 0);
        assert_eq!(cold.next_position(&stage), 1);
        let mut rebuilt = cold.into_cache();
        assert_eq!(keys(&rebuilt)?, vec![0.]);
        assert_eq!(
            rebuilt.text_arithmetic,
            Some(TextArithmeticPolicy::phi_k(TextRequestContext::new(140)?))
        );
        stage.clear_session_state(&mut rebuilt);
        stage.clear_prompt_cache();
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn prompt_cache_cancel_error_and_drop_release_pending_state() -> Result<()> {
        let mut stage = test_stage();
        let mut cancelled = generation(&stage, "alice", &[0, 1]);
        cancelled.step(&mut stage)?;
        assert!(stage.prompt_cache_bytes() > 0);
        stage.prompt_cache.entries.back_mut().unwrap().last_used =
            Instant::now() - stage.prompt_cache.ttl - Duration::from_secs(1);
        cancelled.discard_cache(&mut stage);
        cancelled.publish_prompt_cache(&mut stage);
        assert_eq!(stage.prompt_cache_bytes(), 0);
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        assert_eq!(stage.prompt_cache_match_tokens("alice", &[0, 1]), (0, 0));
        let mut dropped = generation(&stage, "alice", &[0, 1]);
        dropped.step(&mut stage)?;
        stage.prompt_cache.entries.back_mut().unwrap().last_used =
            Instant::now() - stage.prompt_cache.ttl - Duration::from_secs(1);
        drop(dropped);
        stage.prune_prompt_cache();
        assert_eq!(stage.prompt_cache_bytes(), 0);
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        let mut invalid = vec![0; 64];
        invalid.push(255); // Valid token that triggers the fixture's forward error.
        let mut failed = generation(&stage, "alice", &invalid);
        failed.set_shared_prompt_prefix(64);
        assert!(failed.step_with_prefill_budget(&mut stage, 64)?.is_none());
        assert!(stage.prompt_cache_bytes() > 0);
        stage.prompt_cache.entries.back_mut().unwrap().last_used =
            Instant::now() - stage.prompt_cache.ttl - Duration::from_secs(1);
        assert!(failed.step(&mut stage).is_err());
        failed.publish_prompt_cache(&mut stage);
        assert!(
            stage
                .prompt_cache
                .entries
                .iter()
                .all(|entry| entry.pending.is_some())
        );
        assert_eq!(stage.prompt_cache_match_tokens("alice", &invalid), (0, 0));
        assert!(stage.reuse_prompt("alice", &invalid, true)?.is_none());
        failed.discard_cache(&mut stage);
        failed.publish_prompt_cache(&mut stage);
        assert_eq!(stage.prompt_cache_bytes(), 0);
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn prompt_cache_preserves_request_sampling_rules() -> Result<()> {
        let mut stage = test_stage();
        let mut original = generation(&stage, "alice", &[1, 1]);
        finish(&mut stage, &mut original)?;
        let params = SamplingParams {
            max_tokens: 8,
            seed: Some(7),
            temperature: 0.9,
            top_p: 1.0,
            top_k: 0,
            frequency_penalty: 1.0,
            presence_penalty: 0.5,
            logit_bias: [(0, 1.0)].into_iter().collect(),
            ..SamplingParams::default()
        };
        let mut warm = GenerationSession::new(&stage, vec![1, 1], params.clone(), vec![]);
        warm.enable_prompt_cache("alice".into());
        let mut cold = GenerationSession::new(&stage, vec![1, 1], params, vec![]);
        finish(&mut stage, &mut warm)?;
        finish(&mut stage, &mut cold)?;
        assert_eq!(warm.reused_prompt_tokens(), 2);
        assert_eq!(warm.generated(), cold.generated());
        Ok(())
    }

    fn keys(state: &PipelineStageSessionState) -> Result<Vec<f32>> {
        let k = match state.cache[0].as_ref().unwrap() {
            LayerCache::Attention(KvCache::F16 { k, .. }) => k.clone(),
            LayerCache::Attention(KvCache::PreallocatedF16(cache)) => cache.active_f16()?.0,
            _ => unreachable!(),
        };
        Ok(k.flatten_all()?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1()?)
    }

    #[test]
    fn prompt_cache_forks_private_preallocated_storage_for_concurrent_appends() -> Result<()> {
        let mut stage = preallocated_test_stage();
        let mut state = stage.new_session_state();
        let StageOutput::Logits(logits) =
            stage.forward_with_session(&mut state, StageInput::Tokens(vec![2, 3]), 2, 0)?
        else {
            unreachable!()
        };
        let ticket = stage
            .capture_prompt("alice", &[2, 3], false, &state, &logits, false)?
            .unwrap();
        stage.publish_prompt(&ticket);
        let (mut a, _, _) = stage.reuse_prompt("alice", &[2, 3, 4], false)?.unwrap();
        let (mut b, _, _) = stage.reuse_prompt("alice", &[2, 3, 5], false)?.unwrap();
        stage.forward_with_session(&mut a, StageInput::Tokens(vec![4]), 1, 2)?;
        stage.forward_with_session(&mut b, StageInput::Tokens(vec![5]), 1, 2)?;
        stage.forward_with_session(&mut state, StageInput::Tokens(vec![6]), 1, 2)?;
        assert_eq!(keys(&a)?, vec![2., 3., 4.]);
        assert_eq!(keys(&b)?, vec![2., 3., 5.]);
        assert_eq!(keys(&state)?, vec![2., 3., 6.]);
        let (cached, _, _) = stage.reuse_prompt("alice", &[2, 3], false)?.unwrap();
        assert_eq!(keys(&cached)?, vec![2., 3.]);
        drop((a, b, cached, state));
        stage.clear_cache();
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn prompt_hits_keep_compact_batch_storage_and_mix_with_cold_lanes() -> Result<()> {
        for len in [2, 64, 65] {
            for warm_peer in [false, true] {
                let mut stage = preallocated_test_stage();
                stage.set_max_total_kv_tokens(4096)?;
                let prompt = vec![2; len];
                let mut cold = stage.new_session_state();
                let StageOutput::Logits(logits) = stage.forward_with_session(
                    &mut cold,
                    StageInput::Tokens(prompt.clone()),
                    len,
                    0,
                )?
                else {
                    unreachable!()
                };
                let ticket = stage
                    .capture_prompt("alice", &prompt, false, &cold, &logits, false)?
                    .unwrap();
                stage.publish_prompt(&ticket);
                let (mut a, _, _) = stage.reuse_prompt("alice", &prompt, false)?.unwrap();
                let mut b = if warm_peer {
                    drop(cold);
                    stage.reuse_prompt("alice", &prompt, false)?.unwrap().0
                } else {
                    cold
                };
                let Some(LayerCache::Attention(KvCache::PreallocatedF16(cache))) = &a.cache[0]
                else {
                    panic!("prompt hit lost the batch-compatible cache format");
                };
                assert_eq!(
                    (cache.len, cache.capacity, cache.allocation_lanes),
                    (len, len, 1)
                );
                assert_eq!(a.kv_cache_allocation_tokens, len);
                stage.forward_decode_batch_with_sessions(&mut [&mut a, &mut b], &[3, 4])?;
                assert_eq!(a.next_position, len + 1);
                assert_eq!(b.next_position, len + 1);
                assert_eq!(keys(&a)?, [vec![2.; len], vec![3.]].concat());
                assert_eq!(keys(&b)?, [vec![2.; len], vec![4.]].concat());
                // Capturing a packed lane must not retain the peer allocation.
                let compact = stage.fork_prompt_state(&a)?;
                assert_eq!(compact.kv_cache_allocation_tokens, len + 1);
                stage.forward_with_session(&mut a, StageInput::Tokens(vec![5]), 1, len + 1)?;
                assert_eq!(keys(&compact)?, [vec![2.; len], vec![3.]].concat());
                let (original, _, _) = stage.reuse_prompt("alice", &prompt, false)?.unwrap();
                assert_eq!(keys(&original)?, vec![2.; len]);
                drop((a, b, compact, original));
                stage.clear_cache();
                assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
            }
        }
        Ok(())
    }

    #[test]
    fn prompt_cache_extension_and_batch_reuse() -> Result<()> {
        let mut stage = test_stage();
        let mut first = generation(&stage, "alice", &[0, 1]);
        first.set_speculative_decoding_enabled(&stage, false);
        finish(&mut stage, &mut first)?;
        let mut a = generation(&stage, "alice", &[0, 1, 0]);
        let mut b = generation(&stage, "alice", &[0, 1, 1]);
        for g in [&mut a, &mut b] {
            g.set_speculative_decoding_enabled(&stage, false);
            g.step(&mut stage)?;
            assert_eq!(g.reused_prompt_tokens(), 2);
        }
        GenerationSession::step_decode_batch(&mut stage, &mut [&mut a, &mut b])?;
        let ca = a.into_cache();
        let cb = b.into_cache();
        assert_eq!(&keys(&ca)?[..3], &[0., 1., 0.]);
        assert_eq!(&keys(&cb)?[..3], &[0., 1., 1.]);
        drop((ca, cb));
        stage.prune_prompt_cache();
        stage.clear_cache();
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        Ok(())
    }

    #[test]
    fn prompt_cache_enforces_lru_bytes_ttl_and_pressure() -> Result<()> {
        let mut stage = test_stage();
        stage.set_prompt_cache_limits(1024, 2, Duration::from_secs(60));
        for tokens in [&[0, 0][..], &[0, 1]] {
            let mut g = generation(&stage, "alice", tokens);
            finish(&mut stage, &mut g)?;
        }
        drop(stage.reuse_prompt("alice", &[0, 0], true)?);
        let mut g = generation(&stage, "alice", &[1, 1]);
        finish(&mut stage, &mut g)?;
        assert!(stage.reuse_prompt("alice", &[0, 1], true)?.is_none());
        assert!(stage.reuse_prompt("alice", &[0, 0], true)?.is_some());
        assert!(stage.prompt_cache_bytes() > 0);
        assert!(stage.total_kv_cache_allocation_tokens() > 0);
        assert!(
            stage
                .prompt_cache
                .entries
                .iter()
                .all(|entry| entry.pending.is_none())
        );
        for entry in &mut stage.prompt_cache.entries {
            entry.last_used -= Duration::from_secs(61);
        }
        stage.prune_prompt_cache();
        assert_eq!(stage.prompt_cache_bytes(), 0);
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        stage.set_prompt_cache_limits(1, 2, Duration::from_secs(60));
        let mut g = generation(&stage, "alice", &[0, 0]);
        finish(&mut stage, &mut g)?;
        assert_eq!(stage.prompt_cache_bytes(), 0);
        stage.set_prompt_cache_limits(1024, 2, Duration::from_secs(60));
        stage.set_max_total_kv_tokens(128)?;
        let mut g = generation(&stage, "alice", &[0, 0]);
        finish(&mut stage, &mut g)?;
        assert!(stage.prompt_cache_bytes() > 0);
        stage.release_prompt_cache_under_pressure(64);
        assert_eq!(stage.prompt_cache_bytes(), 0);
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        Ok(())
    }

    #[test]
    fn prompt_cache_handoff_clear_keeps_active_state_and_releases_snapshot_charge() -> Result<()> {
        let mut stage = test_stage();
        let mut first = generation(&stage, "alice", &[0, 1]);
        finish(&mut stage, &mut first)?;
        let mut active = generation(&stage, "alice", &[0, 1]);
        active.step(&mut stage)?;
        let before = stage.total_kv_cache_allocation_tokens();
        stage.clear_prompt_cache();
        assert_eq!(stage.prompt_cache_bytes(), 0);
        assert!(stage.total_kv_cache_allocation_tokens() < before);
        assert_eq!(
            stage.total_kv_cache_tokens(),
            active.kv_cache_tokens(&stage)
        );
        finish(&mut stage, &mut active)?;
        assert_eq!(active.generated(), first.generated());
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        let mut next = generation(&stage, "alice", &[0, 1]);
        finish(&mut stage, &mut next)?;
        assert!(stage.prompt_cache_bytes() > 0);
        stage.set_max_kv_tokens(1);
        assert_eq!(stage.prompt_cache_bytes(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn shared_prefix_serves_different_questions_without_tail_pollution() -> Result<()> {
        let mut stage = preallocated_test_stage();
        stage.set_prompt_cache_limits(1000, 4, Duration::from_secs(60));
        let mut prompt = vec![0; 64];
        prompt.push(1);
        let mut first = generation(&stage, "alice", &prompt);
        first.set_shared_prompt_prefix(64);
        finish(&mut stage, &mut first)?;
        // The full prompt would exceed the shared byte budget. Keep the shared
        // checkpoint, not the question or any generated token.
        assert_eq!(stage.prompt_cache.entries.len(), 1);
        assert_eq!(stage.prompt_cache.entries[0].tokens, vec![0; 64]);
        prompt[64] = 0;
        let mut next = generation(&stage, "alice", &prompt);
        next.set_shared_prompt_prefix(64);
        next.step(&mut stage)?;
        assert_eq!(next.reused_prompt_tokens(), 64);
        let state = next.into_cache();
        assert_eq!(keys(&state)?, vec![0.; 65]);
        drop(state);
        stage.prune_prompt_cache();
        assert_eq!(stage.prompt_cache.entries.len(), 1);
        let mut changed_system = prompt;
        changed_system[63] = 1;
        assert!(
            stage
                .reuse_prompt("alice", &changed_system, true)?
                .is_none()
        );
        assert!(stage.reuse_prompt("bob", &vec![0; 65], true)?.is_none());
        stage.clear_cache();
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn shared_prefix_live_pending_outlives_ttl_until_successful_publication() -> Result<()> {
        let mut stage = test_stage();
        let prompt = vec![0; 80];
        let mut session = generation(&stage, "alice", &prompt);
        session.set_shared_prompt_prefix(64);
        for _ in 0..2 {
            assert!(session.step_with_prefill_budget(&mut stage, 32)?.is_none());
        }
        assert!(!session.is_finished());
        assert_eq!(stage.prompt_cache.entries.len(), 1);
        let expired = Instant::now() - stage.prompt_cache.ttl - Duration::from_secs(1);
        stage.prompt_cache.entries[0].last_used = expired;
        let bytes = stage.prompt_cache_bytes();
        let allocated = stage.total_kv_cache_allocation_tokens();
        assert!(bytes > 0);
        stage.prune_prompt_cache();
        assert_eq!(stage.prompt_cache_bytes(), bytes);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), allocated);
        session.publish_prompt_cache(&mut stage);
        assert_eq!(stage.prompt_cache.entries[0].last_used, expired);
        assert!(stage.prompt_cache.entries[0].pending.is_some());
        assert!(stage.reuse_prompt("alice", &prompt, true)?.is_none());
        assert_eq!(stage.prompt_cache_match_tokens("alice", &prompt), (0, 0));

        while !session.is_finished() {
            session.step_with_prefill_budget(&mut stage, 17)?;
        }
        let before_publication = Instant::now();
        session.publish_prompt_cache(&mut stage);
        let shared = stage
            .prompt_cache
            .entries
            .iter()
            .find(|entry| entry.shared_prefix)
            .expect("shared snapshot survives");
        assert!(shared.pending.is_none());
        assert!(shared.last_used >= before_publication);
        assert!(shared.last_used.elapsed() < stage.prompt_cache.ttl);
        session.discard_cache(&mut stage);

        let mut changed = vec![0; 81];
        changed[64] = 1;
        assert_eq!(stage.prompt_cache_match_tokens("alice", &changed), (0, 64));
        let mut next = generation(&stage, "alice", &changed);
        next.set_shared_prompt_prefix(64);
        assert!(next.step_with_prefill_budget(&mut stage, 8)?.is_none());
        assert_eq!(next.reused_prompt_tokens(), 64);
        assert_eq!(next.kv_cache_tokens(&stage), 72);
        let state = next.into_cache();
        let mut expected = vec![0.; 72];
        expected[64] = 1.;
        assert_eq!(keys(&state)?, expected);
        drop(state);
        stage.clear_prompt_cache();
        assert_eq!(stage.prompt_cache_bytes(), 0);
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn shared_prefix_cancellation_does_not_publish_either_snapshot() -> Result<()> {
        let mut stage = test_stage();
        let mut session = generation(&stage, "alice", &vec![0; 70]);
        session.set_shared_prompt_prefix(64);
        session.step(&mut stage)?;
        assert_eq!(stage.prompt_cache.entries.len(), 2);
        assert!(stage.reuse_prompt("alice", &vec![0; 70], true)?.is_none());
        session.discard_cache(&mut stage);
        assert_eq!(stage.prompt_cache_bytes(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn chunked_shared_prefix_stays_pending_until_success() -> Result<()> {
        let mut stage = test_stage();
        let mut session = generation(&stage, "alice", &vec![0; 80]);
        session.set_shared_prompt_prefix(64);
        for processed in [32, 64] {
            assert!(session.step_with_prefill_budget(&mut stage, 32)?.is_none());
            assert_eq!(session.kv_cache_tokens(&stage), processed);
            assert!(session.generated().is_empty());
            session.publish_prompt_cache(&mut stage);
            assert!(stage.reuse_prompt("alice", &vec![0; 80], true)?.is_none());
        }
        assert_eq!(stage.prompt_cache.entries.len(), 1);
        session.discard_cache(&mut stage);
        assert_eq!(stage.prompt_cache_bytes(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);

        let mut completed = generation(&stage, "alice", &vec![0; 80]);
        completed.set_shared_prompt_prefix(64);
        while !completed.is_finished() {
            completed.step_with_prefill_budget(&mut stage, 17)?;
        }
        completed.publish_prompt_cache(&mut stage);
        completed.discard_cache(&mut stage);
        let mut changed = vec![0; 81];
        changed[64] = 1;
        let mut next = generation(&stage, "alice", &changed);
        next.set_shared_prompt_prefix(64);
        assert!(next.step_with_prefill_budget(&mut stage, 8)?.is_none());
        assert_eq!(next.reused_prompt_tokens(), 64);
        assert_eq!(next.kv_cache_tokens(&stage), 72);
        next.discard_cache(&mut stage);
        stage.clear_cache();
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn cancelled_input_resumes_strict_extension_with_cold_logits_and_seed_parity() -> Result<()> {
        let mut stage = crate::session_state_tests::recurrent_stage(false);
        let scope = "reserved:synthetic";
        assert!(stage.register_reserved_prompt_scope(scope.into()));
        let prompt: Vec<u32> = (0..137).map(|i| i % 2).collect();
        let mut stopped = generation(&stage, scope, &prompt);
        for _ in 0..3 {
            assert!(stopped.step_with_prefill_budget(&mut stage, 32)?.is_none());
        }
        let before = stage.total_kv_cache_allocation_tokens();
        assert!(stopped.retain_cancelled_input_prefix(&mut stage, scope));
        assert!(!stopped.is_finished());
        assert_eq!(stopped.next_position(&stage), 0);
        let entry = stage.prompt_cache.entries.back().unwrap();
        assert!(entry.logits.is_none());
        assert_eq!(entry.tokens, prompt[..96]);
        assert_eq!(
            stage.total_kv_cache_allocation_tokens(),
            before + entry.extra_allocation_slots
        );
        assert_eq!(
            stage.prompt_cache_match_tokens(scope, &prompt[..96]),
            (0, 0)
        );
        assert!(stage.reuse_prompt(scope, &prompt[..96], true)?.is_none());
        assert_eq!(stage.prompt_cache_match_tokens(scope, &prompt), (0, 96));
        for (owner, tokens, draft) in [
            (scope, vec![1; 137], true),
            ("reserved:other", prompt.clone(), true),
            (scope, prompt.clone(), false),
        ] {
            assert!(stage.reuse_prompt(owner, &tokens, draft)?.is_none());
        }
        assert!(test_stage().reuse_prompt(scope, &prompt, true)?.is_none());
        let (mut resumed, logits, skipped) = stage.reuse_prompt(scope, &prompt, true)?.unwrap();
        assert!(logits.is_none());
        assert_eq!(skipped, 96);
        let StageOutput::Logits(warm) = stage.prefill_with_session(
            &mut resumed,
            StageInput::Tokens(prompt[skipped..].to_vec()),
            41,
            skipped,
        )?
        else {
            unreachable!()
        };
        let mut cold = stage.new_session_state();
        let StageOutput::Logits(reference) = stage.prefill_with_session(
            &mut cold,
            StageInput::Tokens(prompt.clone()),
            prompt.len(),
            0,
        )?
        else {
            unreachable!()
        };
        let warm = warm.flatten_all()?.to_vec1::<f32>()?;
        assert!(warm.iter().all(|x| x.is_finite()));
        assert_eq!(warm, reference.flatten_all()?.to_vec1::<f32>()?);
        drop((resumed, cold));
        let params = SamplingParams {
            temperature: 100.0,
            seed: Some(42),
            max_tokens: 8,
            repetition_penalty: 1.2,
            ..SamplingParams::default()
        };
        let mut resumed = GenerationSession::new(&stage, prompt.clone(), params.clone(), vec![]);
        resumed.enable_prompt_cache(scope.into());
        let mut cold = GenerationSession::new(&stage, prompt, params, vec![]);
        while !resumed.is_finished() {
            resumed.step_with_prefill_budget(&mut stage, 17)?;
        }
        while !cold.is_finished() {
            cold.step(&mut stage)?;
        }
        assert_eq!(resumed.reused_prompt_tokens(), 96);
        assert_eq!(resumed.generated().len(), 8);
        assert_eq!(resumed.generated(), cold.generated());
        eprintln!(
            "cancelled input proof: cursor=96 prompt=137 finite_logits={} max_abs_diff=0 seed=42 tokens={:?}",
            warm.len(),
            resumed.generated()
        );
        resumed.discard_cache(&mut stage);
        cold.discard_cache(&mut stage);
        stage.revoke_reserved_prompt_scope(scope);
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn cancelled_input_repeated_stop_retains_longest_authoritative_reuse_with_seed_parity()
    -> Result<()> {
        let mut stage = crate::session_state_tests::recurrent_stage(false);
        let scope = "reserved:repeated-stop";
        assert!(stage.register_reserved_prompt_scope(scope.into()));
        let prompt: Vec<u32> = (0..137).map(|i| i % 2).collect();
        let mut first = generation(&stage, scope, &prompt);
        for _ in 0..3 {
            assert!(first.step_with_prefill_budget(&mut stage, 32)?.is_none());
        }
        assert!(first.retain_cancelled_input_prefix(&mut stage, scope));
        assert_eq!(stage.prompt_cache_match_tokens(scope, &prompt), (0, 96));

        let mut second = generation(&stage, scope, &prompt);
        assert!(second.step_with_prefill_budget(&mut stage, 17)?.is_none());
        assert_eq!(second.reused_prompt_tokens(), 96);
        assert_eq!(second.next_position(&stage), 113);
        assert!(second.generated().is_empty());
        assert!(second.retain_cancelled_input_prefix(&mut stage, scope));
        assert_eq!(stage.prompt_cache_match_tokens(scope, &prompt), (0, 113));
        let entry = stage.prompt_cache.entries.back().unwrap();
        assert_eq!(entry.tokens, prompt[..113]);
        assert!(entry.logits.is_none());
        assert!(entry.pending.is_none());

        let params = SamplingParams {
            temperature: 100.0,
            seed: Some(42),
            max_tokens: 8,
            repetition_penalty: 1.2,
            ..SamplingParams::default()
        };
        let mut resumed = GenerationSession::new(&stage, prompt.clone(), params.clone(), vec![]);
        resumed.enable_prompt_cache(scope.into());
        let mut cold = GenerationSession::new(&stage, prompt, params, vec![]);
        while !resumed.is_finished() {
            resumed.step_with_prefill_budget(&mut stage, 17)?;
        }
        while !cold.is_finished() {
            cold.step(&mut stage)?;
        }
        assert_eq!(resumed.reused_prompt_tokens(), 113);
        assert_eq!(resumed.generated().len(), 8);
        assert_eq!(resumed.generated(), cold.generated());
        eprintln!(
            "cancelled input repeated cancel proof: first=96 new_chunk=17 longest=113 prompt=137 seed=42 tokens={:?}",
            resumed.generated()
        );
        resumed.discard_cache(&mut stage);
        cold.discard_cache(&mut stage);
        stage.revoke_reserved_prompt_scope(scope);
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn cancelled_input_revocation_races_purge_tickets_and_preserve_peer() -> Result<()> {
        for revoke_first in [false, true] {
            let mut stage = test_stage();
            let alice = "reserved:alice";
            let bob = "reserved:bob";
            assert!(stage.register_reserved_prompt_scope(alice.into()));
            assert!(stage.register_reserved_prompt_scope(bob.into()));
            let mut peer = generation(&stage, bob, &[1, 0]);
            finish(&mut stage, &mut peer)?;
            let peer_bytes = stage.prompt_cache_bytes();
            let peer_charge = stage.total_kv_cache_allocation_tokens();
            let mut stopped = generation(&stage, alice, &vec![0; 100]);
            stopped.set_shared_prompt_prefix(64);
            assert!(stopped.step_with_prefill_budget(&mut stage, 72)?.is_none());
            assert!(
                stage
                    .prompt_cache
                    .entries
                    .iter()
                    .any(|e| e.scope == alice && e.pending.is_some())
            );
            if revoke_first {
                stage.revoke_reserved_prompt_scope(alice);
            }
            assert_eq!(
                stopped.retain_cancelled_input_prefix(&mut stage, alice),
                !revoke_first
            );
            stopped.publish_prompt_cache(&mut stage);
            assert!(
                stage
                    .prompt_cache
                    .entries
                    .iter()
                    .all(|e| e.pending.is_none())
            );
            stage.revoke_reserved_prompt_scope(alice);
            assert!(!stopped.retain_cancelled_input_prefix(&mut stage, alice));
            assert_eq!(stage.prompt_cache_bytes(), peer_bytes);
            assert_eq!(stage.total_kv_cache_allocation_tokens(), peer_charge);
            assert!(stage.reuse_prompt(bob, &[1, 0], true)?.is_some());
            assert_eq!(
                stage.prompt_cache_match_tokens(alice, &vec![0; 101]),
                (0, 0)
            );
            stage.revoke_reserved_prompt_scope(bob);
            assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        }
        Ok(())
    }

    #[test]
    fn cancelled_input_forks_own_preallocated_tails_and_release_once() -> Result<()> {
        let mut stage = preallocated_test_stage();
        let scope = "reserved:forks";
        assert!(stage.register_reserved_prompt_scope(scope.into()));
        let mut stopped = generation(&stage, scope, &[2, 3, 4]);
        assert!(stopped.step_with_prefill_budget(&mut stage, 2)?.is_none());
        assert!(stopped.retain_cancelled_input_prefix(&mut stage, scope));
        let (mut a, _, _) = stage.reuse_prompt(scope, &[2, 3, 4], true)?.unwrap();
        let (mut b, _, _) = stage.reuse_prompt(scope, &[2, 3, 5], true)?.unwrap();
        stage.forward_with_session(&mut a, StageInput::Tokens(vec![4]), 1, 2)?;
        stage.forward_with_session(&mut b, StageInput::Tokens(vec![5]), 1, 2)?;
        assert_eq!(keys(&a)?, [2., 3., 4.]);
        assert_eq!(keys(&b)?, [2., 3., 5.]);
        assert_eq!(keys(&stage.prompt_cache.entries[0].state)?, [2., 3.]);
        stage.revoke_reserved_prompt_scope(scope);
        assert_eq!(keys(&a)?, [2., 3., 4.]);
        drop((a, b, stopped));
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn cancelled_input_limits_ttl_pressure_and_shared_prefix_protection() -> Result<()> {
        let mut stage = test_stage();
        let scope = "reserved:limits";
        assert!(stage.register_reserved_prompt_scope(scope.into()));
        stage.set_prompt_cache_limits(4096, 2, Duration::from_secs(300));
        for first in [0, 1, 2] {
            let mut stopped = generation(&stage, scope, &[first, 0, 0]);
            assert!(stopped.step_with_prefill_budget(&mut stage, 2)?.is_none());
            assert!(stopped.retain_cancelled_input_prefix(&mut stage, scope));
        }
        assert_eq!(stage.prompt_cache.entries.len(), 2);
        assert!(stage.reuse_prompt(scope, &[0, 0, 0], true)?.is_none());
        stage.prompt_cache.entries[0].last_used = Instant::now() - Duration::from_secs(301);
        stage.prune_prompt_cache();
        assert_eq!(stage.prompt_cache.entries.len(), 1);
        let mut rejected = generation(&stage, scope, &[0, 0, 0]);
        assert!(rejected.step_with_prefill_budget(&mut stage, 2)?.is_none());
        stage.max_total_kv_tokens = Some(1);
        assert!(!rejected.retain_cancelled_input_prefix(&mut stage, scope));
        stage.release_prompt_cache_under_pressure(1);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        stage.max_total_kv_tokens = None;
        stage.set_prompt_cache_limits(4096, 1, Duration::from_secs(300));
        let mut shared = generation(&stage, scope, &vec![0; 80]);
        shared.set_shared_prompt_prefix(64);
        finish(&mut stage, &mut shared)?;
        assert!(stage.prompt_cache.entries[0].shared_prefix);
        let baseline = stage.total_kv_cache_allocation_tokens();
        let mut stopped = generation(&stage, scope, &vec![0; 81]);
        assert!(stopped.step_with_prefill_budget(&mut stage, 8)?.is_none());
        assert_eq!(stopped.reused_prompt_tokens(), 64);
        assert!(!stopped.retain_cancelled_input_prefix(&mut stage, scope));
        assert_eq!(stage.total_kv_cache_allocation_tokens(), baseline);
        stage.set_prompt_cache_limits(1, 4, Duration::from_secs(300));
        let mut oversized = generation(&stage, scope, &[0, 1, 0]);
        assert!(oversized.step_with_prefill_budget(&mut stage, 2)?.is_none());
        assert!(!oversized.retain_cancelled_input_prefix(&mut stage, scope));
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }
}
