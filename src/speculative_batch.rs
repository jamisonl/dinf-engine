//! Budgeted, rollback-safe target verification for independent sessions.

use super::*;
use continuous::DecodeBatchStepError::{self, RetryScalar, Terminal};

impl PipelineStage {
    /// Whether this stage can share target work for draft-enabled requests.
    /// Opt-in until model-level correctness and speed checks qualify the path.
    pub fn supports_speculative_batch(&self) -> bool {
        self.position == StagePosition::Solo
            && self.speculative.is_some()
            && self.effective_speculative_block_size().ok().flatten() == Some(8)
            && self.arch.supports_speculative_batch()
            && std::env::var("DINF_ENABLE_SPECULATIVE_BATCH").as_deref() == Ok("1")
    }

    pub(crate) fn begin_speculative_batch(
        &mut self,
        states: &mut [&mut PipelineStageSessionState],
        anchors: &[u32],
        block_size: usize,
    ) -> std::result::Result<Vec<SpeculativeVerification>, DecodeBatchStepError> {
        self.preflight_speculative_batch(states, anchors, block_size)
            .map_err(RetryScalar)?;
        let mut checkpoints = states
            .iter()
            .map(|s| self.checkpoint_session_state(s))
            .collect::<Result<Vec<_>>>()
            .map_err(RetryScalar)?;
        let draft = self.speculative.as_ref().unwrap();
        let layer_ids = draft.target_layers().to_vec();
        let mut inputs = Vec::with_capacity(states.len());
        let mut draft_times = Vec::with_capacity(states.len());
        for (state, &anchor) in states.iter().zip(anchors) {
            let started = Instant::now();
            let proposed = draft
                .draft(
                    state.speculative_cache.as_ref().unwrap(),
                    anchor,
                    state.next_position,
                    block_size,
                )
                .map_err(Terminal)?;
            draft_times.push(started.elapsed());
            if proposed.len() + 1 != block_size {
                return Err(Terminal(anyhow::anyhow!(
                    "DFlash draft block length changed"
                )));
            }
            inputs.push(std::iter::once(anchor).chain(proposed).collect::<Vec<_>>());
        }
        for state in states.iter_mut() {
            self.detach_decode_batch(state);
        }
        let positions = states.iter().map(|s| s.next_position).collect::<Vec<_>>();
        let target_started = Instant::now();
        let result = (|| -> Result<_> {
            let outputs = self.arch.forward_speculative_batch(
                &inputs,
                &mut states
                    .iter_mut()
                    .map(|s| s.cache.as_mut_slice())
                    .collect::<Vec<_>>(),
                &positions,
                &layer_ids,
            )?;
            anyhow::ensure!(
                outputs.len() == states.len(),
                "draft batch output count mismatch"
            );
            for output in &outputs {
                anyhow::ensure!(
                    output.logits.dims() == [block_size, self.vocab_size]
                        && output.captures.len() == layer_ids.len()
                        && output
                            .captures
                            .iter()
                            .all(|c| c.dims() == [block_size, self.hidden_dim]),
                    "draft batch output shape mismatch"
                );
            }
            for state in states.iter_mut() {
                for cache in state.cache.iter_mut().flatten() {
                    architectures::compact_fixed_layer_cache(cache)?;
                }
            }
            let allocations = states
                .iter()
                .map(|s| architectures::kv_cache_allocation_tokens(&s.cache))
                .collect::<Result<Vec<_>>>()?;
            let mut total = self.kv_token_budget.allocated();
            for (state, &(_, allocated)) in states.iter().zip(&allocations) {
                total = total
                    .checked_sub(state.kv_cache_allocation_tokens)
                    .and_then(|n| n.checked_add(allocated))
                    .context("draft batch allocation overflow")?;
            }
            anyhow::ensure!(
                self.max_total_kv_tokens.is_none_or(|max| total <= max),
                "draft batch exceeds resident KV limit"
            );
            for (state, (capacity, allocated)) in states.iter_mut().zip(allocations) {
                state.kv_token_budget.add(block_size);
                state
                    .kv_token_budget
                    .replace_allocated(state.kv_cache_allocation_tokens, allocated);
                state.kv_cache_tokens += block_size;
                state.next_position += block_size;
                state.kv_cache_lane_capacity_tokens = capacity;
                state.kv_cache_allocation_tokens = allocated;
                state.revision = state.revision.wrapping_add(1);
            }
            Ok(outputs)
        })();
        let outputs = match result {
            Ok(outputs) => outputs,
            Err(error) => {
                for (state, checkpoint) in states.iter_mut().zip(checkpoints.drain(..)) {
                    self.restore_session_checkpoint(state, checkpoint);
                }
                return Err(Terminal(
                    error.context("shared DFlash target verification failed"),
                ));
            }
        };
        Ok(outputs
            .into_iter()
            .zip(inputs)
            .zip(draft_times)
            .zip(checkpoints)
            .map(
                |(((output, inputs), draft_time), checkpoint)| SpeculativeVerification {
                    proposed: inputs[1..].to_vec(),
                    inputs,
                    logits: output.logits,
                    captures: output.captures,
                    draft_time,
                    target_started,
                    checkpoint,
                },
            )
            .collect())
    }

    fn preflight_speculative_batch(
        &mut self,
        states: &[&mut PipelineStageSessionState],
        anchors: &[u32],
        rows: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            self.supports_speculative_batch(),
            "draft batching unavailable"
        );
        anyhow::ensure!(
            states
                .iter()
                .all(|state| self.session_speculative_ready(state)),
            "draft cache is not ready"
        );
        anyhow::ensure!(
            states
                .iter()
                .all(|state| self.speculative_rows_fit_context(state, rows)),
            "draft batch exceeds target or draft context"
        );
        self.preflight_speculative_budget(states, anchors, rows)
    }

    fn preflight_speculative_budget(
        &mut self,
        states: &[&mut PipelineStageSessionState],
        anchors: &[u32],
        rows: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            (2..=4).contains(&states.len())
                && states.len() == anchors.len()
                && rows == 8
                && anchors.iter().all(|&t| (t as usize) < self.vocab_size),
            "invalid draft batch geometry"
        );
        let mut additional = 0usize;
        for state in states {
            anyhow::ensure!(
                Arc::ptr_eq(&state.kv_token_budget, &self.kv_token_budget)
                    && state.cache.len() == self.cache.len(),
                "draft cache belongs to another stage"
            );
            state
                .next_position
                .checked_add(rows)
                .context("draft position overflow")?;
            let desired = state
                .kv_cache_tokens
                .checked_add(rows)
                .context("draft KV length overflow")?;
            anyhow::ensure!(
                self.max_kv_tokens.is_none_or(|max| desired <= max),
                "draft batch requires lane-local KV eviction"
            );
            let allocated =
                if self.preallocated_kv_cache && desired > state.kv_cache_lane_capacity_tokens {
                    architectures::PreallocatedKvCache::next_capacity(
                        state.kv_cache_lane_capacity_tokens,
                        desired,
                    )?
                } else if self.preallocated_kv_cache {
                    state.kv_cache_allocation_tokens
                } else {
                    desired
                };
            additional = additional
                .checked_add(allocated.saturating_sub(state.kv_cache_allocation_tokens))
                .context("draft allocation overflow")?;
        }
        self.release_prompt_cache_under_pressure(additional);
        let allocated = self
            .kv_token_budget
            .allocated()
            .checked_add(additional)
            .context("draft allocation overflow")?;
        let retained = self
            .kv_token_budget
            .total()
            .checked_add(rows * states.len())
            .context("draft KV overflow")?;
        anyhow::ensure!(
            self.max_total_kv_tokens
                .is_none_or(|max| allocated <= max && retained <= max),
            "draft batch requires aggregate KV eviction"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_budget_rejects_without_advancing_or_charging_sessions() -> Result<()> {
        let mut stage = crate::session_state_tests::test_stage();
        stage.vocab_size = 2; // This test exercises a real binary-vocabulary bound.
        let mut a = stage.new_session_state();
        let mut b = stage.new_session_state();
        stage.forward_with_session(&mut a, architectures::StageInput::Tokens(vec![0; 4]), 4, 0)?;
        stage.forward_with_session(&mut b, architectures::StageInput::Tokens(vec![1; 5]), 5, 0)?;
        stage.preflight_speculative_budget(&[&mut a, &mut b], &[0, 1], 8)?;
        stage.set_max_kv_tokens(12);
        assert!(
            stage
                .preflight_speculative_budget(&[&mut a, &mut b], &[0, 1], 8)
                .is_err()
        );
        stage.set_max_kv_tokens(64);
        stage.set_max_total_kv_tokens(48)?; // 9 existing + 16 incoming > 24 resident.
        assert!(
            stage
                .preflight_speculative_budget(&[&mut a, &mut b], &[0, 1], 8)
                .is_err()
        );
        assert_eq!(stage.total_kv_cache_tokens(), 9);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 9);
        assert_eq!((a.next_position, b.next_position), (4, 5));
        stage.set_max_total_kv_tokens(50)?;
        stage.preflight_speculative_budget(&[&mut a, &mut b], &[0, 1], 8)?;
        let foreign = crate::session_state_tests::test_stage();
        let mut other = foreign.new_session_state();
        assert!(
            stage
                .preflight_speculative_budget(&[&mut a, &mut other], &[0, 1], 8)
                .is_err()
        );
        assert!(
            stage
                .preflight_speculative_budget(&[&mut a, &mut b], &[0, 2], 8)
                .is_err()
        );
        assert!(
            stage
                .preflight_speculative_budget(&[&mut a, &mut b], &[0, 1], 4)
                .is_err()
        );
        assert!(
            stage
                .preflight_speculative_budget(&[&mut a, &mut b], &[0], 8)
                .is_err()
        );
        b.next_position = usize::MAX - 1;
        assert!(
            stage
                .preflight_speculative_budget(&[&mut a, &mut b], &[0, 1], 8)
                .is_err()
        );
        drop((a, b));
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn batch_budget_checks_preallocated_growth_not_just_used_tokens() -> Result<()> {
        let mut stage = crate::session_state_tests::preallocated_test_stage();
        let mut a = stage.new_session_state();
        let mut b = stage.new_session_state();
        stage.forward_with_session(
            &mut a,
            architectures::StageInput::Tokens(vec![0; 60]),
            60,
            0,
        )?;
        stage.forward_with_session(
            &mut b,
            architectures::StageInput::Tokens(vec![1; 60]),
            60,
            0,
        )?;
        let before = stage.total_kv_cache_allocation_tokens();
        stage.set_max_total_kv_tokens(before * 2)?;
        assert!(
            stage
                .preflight_speculative_budget(&[&mut a, &mut b], &[0, 1], 8)
                .is_err()
        );
        assert_eq!(stage.total_kv_cache_allocation_tokens(), before);
        assert_eq!(stage.total_kv_cache_tokens(), 120);
        Ok(())
    }

    #[test]
    fn batch_budget_reuses_spare_slots_without_an_extra_charge() -> Result<()> {
        let mut stage = crate::session_state_tests::preallocated_test_stage();
        let mut a = stage.new_session_state();
        let mut b = stage.new_session_state();
        stage.forward_with_session(&mut a, architectures::StageInput::Tokens(vec![0; 4]), 4, 0)?;
        stage.forward_with_session(&mut b, architectures::StageInput::Tokens(vec![1; 5]), 5, 0)?;
        let before = stage.total_kv_cache_allocation_tokens();
        stage.set_max_total_kv_tokens(before * 2)?;
        stage.preflight_speculative_budget(&[&mut a, &mut b], &[0, 1], 8)?;
        assert_eq!(stage.total_kv_cache_allocation_tokens(), before);
        assert_eq!(stage.total_kv_cache_tokens(), 9);
        Ok(())
    }
}
