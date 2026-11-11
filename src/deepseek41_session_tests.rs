//! Exercise the real session/budget wrapper with a deterministic state machine;
//! no model weights, filesystem artifacts or accelerator allocation required.
use super::*;
use crate::architectures::deepseek41::{
    DYNAMIC_BYTES_PER_TOKEN, DeepSeek41Cache, FIXED_CACHE_BYTES,
};
use candle_core::{DType, Tensor};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct MayaStateMachine {
    fail_after_write: Arc<AtomicBool>,
    forward_calls: Arc<AtomicUsize>,
}

impl ModelArchitecture for MayaStateMachine {
    fn forward(
        &self,
        input: StageInput,
        slots: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_calls.fetch_add(1, Ordering::Relaxed);
        let StageInput::Tokens(tokens) = input else {
            anyhow::bail!("tokens required");
        };
        let mut cache = match &slots[0] {
            Some(LayerCache::DeepSeek41(c)) => (**c).clone(),
            None => DeepSeek41Cache::default(),
            _ => anyhow::bail!("wrong state machine cache"),
        };
        anyhow::ensure!(cache.position == start_pos, "wrong state machine position");
        for id in tokens {
            let tail = cache.history.compressed_tail();
            cache.history = architectures::deepseek41::engram::History::from_compressed([
                id as i32, tail[0], tail[1],
            ])?;
        }
        cache.position += seq_len;
        slots[0] = Some(LayerCache::DeepSeek41(Box::new(cache)));
        // Deliberately less transactional than the real graph: the outer engine
        // must still restore a checkpoint on post-write execution failure.
        anyhow::ensure!(
            !self.fail_after_write.load(Ordering::Relaxed),
            "injected execution failure"
        );
        Ok(StageOutput::Logits(Tensor::new(&[1f32, 0.], &Device::Cpu)?))
    }

    fn family(&self) -> &str {
        "deepseek41"
    }
    fn hidden_dim(&self) -> usize {
        5120
    }
    fn layer_count(&self) -> usize {
        40
    }
    fn vocab_size(&self) -> usize {
        129280
    }
}

fn stage() -> (PipelineStage, Arc<AtomicBool>) {
    let fail = Arc::new(AtomicBool::new(false));
    let mut stage = session_state_tests::test_stage();
    stage.arch = Box::new(MayaStateMachine {
        fail_after_write: Arc::clone(&fail),
        forward_calls: Arc::new(AtomicUsize::new(0)),
    });
    stage.cache = (0..40).map(|_| None).collect();
    stage.vocab_size = 129280;
    stage.context_length = 64;
    stage.kv_bytes_per_token = DYNAMIC_BYTES_PER_TOKEN as u64;
    stage.kv_budget_bytes_per_token = DYNAMIC_BYTES_PER_TOKEN as u64;
    stage.fixed_cache_bytes = FIXED_CACHE_BYTES;
    (stage, fail)
}

fn history(cache: &[Option<LayerCache>]) -> [i32; 3] {
    let Some(LayerCache::DeepSeek41(c)) = &cache[0] else {
        panic!("missing Maya cache");
    };
    c.history.compressed_tail()
}

#[test]
fn deepseek41_native_fixed_session_charge_rejects_before_mutation() -> Result<()> {
    let (mut s, fail) = stage();
    let calls = Arc::new(AtomicUsize::new(0));
    s.arch = Box::new(MayaStateMachine {
        fail_after_write: fail,
        forward_calls: Arc::clone(&calls),
    });
    let fixed = FIXED_CACHE_BYTES.div_ceil(DYNAMIC_BYTES_PER_TOKEN);
    // The existing resident/copy split halves total headroom. One unit short
    // cannot admit fixed state alone, before the minimum one dynamic unit.
    s.set_max_total_kv_tokens(2 * (fixed - 1))?;
    let mut lane = s.new_session_state();
    let error = s
        .forward_with_session(&mut lane, StageInput::Tokens(vec![7]), 1, 0)
        .err()
        .context("insufficient fixed reserve was admitted")?;
    assert!(
        error.to_string().contains("fixed request state"),
        "{error:#}"
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert!(lane.cache.iter().all(Option::is_none));
    assert_eq!(
        (
            lane.next_position,
            lane.kv_cache_tokens,
            lane.fixed_cache_allocation_tokens
        ),
        (0, 0, 0)
    );
    assert_eq!(
        (
            s.total_kv_cache_tokens(),
            s.total_kv_cache_allocation_tokens()
        ),
        (0, 0)
    );
    s.set_max_total_kv_tokens(2 * (fixed + 1))?;
    s.forward_with_session(&mut lane, StageInput::Tokens(vec![7]), 1, 0)?;
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(lane.fixed_cache_allocation_tokens, fixed);
    assert_eq!(s.total_kv_cache_allocation_tokens(), fixed + 1);
    let mut other = s.new_session_state();
    assert!(
        s.forward_with_session(&mut other, StageInput::Tokens(vec![8]), 1, 0)
            .is_err()
    );
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert!(other.cache.iter().all(Option::is_none));
    assert_eq!(history(&lane.cache), [7, -1, -1]);
    assert_eq!(s.total_kv_cache_allocation_tokens(), fixed + 1);
    s.clear_session_state(&mut lane);
    assert_eq!(s.total_kv_cache_allocation_tokens(), 0);
    Ok(())
}

#[test]
fn deepseek41_full_partial_and_public_outer_eviction_preserve_state() -> Result<()> {
    let (mut s, _) = stage();
    s.forward(StageInput::Tokens(vec![3, 4]), 2, 0)?;
    let allocation = s.total_kv_cache_allocation_tokens();
    for count in [1, 2, 3] {
        assert!(s.try_evict_oldest(count).is_err());
        s.evict_oldest(count);
        assert_eq!(s.next_position(), 2);
        assert_eq!(s.kv_cache_tokens(), 2);
        assert_eq!(s.total_kv_cache_allocation_tokens(), allocation);
        assert_eq!(history(&s.cache), [4, 3, -1]);
    }
    s.set_max_kv_tokens(2);
    assert!(s.forward(StageInput::Tokens(vec![5]), 1, 2).is_err());
    // Equality used to bypass try_evict_oldest and replace the entire cache.
    assert!(s.forward(StageInput::Tokens(vec![5, 6]), 2, 2).is_err());
    assert_eq!(history(&s.cache), [4, 3, -1]);
    assert_eq!(s.total_kv_cache_allocation_tokens(), allocation);
    Ok(())
}

#[test]
fn deepseek41_interleaved_sessions_checkpoint_fork_rollback_and_reset() -> Result<()> {
    let (mut s, fail) = stage();
    let mut a = s.new_session_state();
    let mut b = s.new_session_state();
    s.forward_with_session(&mut a, StageInput::Tokens(vec![3, 4]), 2, 0)?;
    s.forward_with_session(&mut b, StageInput::Tokens(vec![7]), 1, 0)?;
    let checkpoint = s.checkpoint_session_state(&a)?;
    let mut fork = s.new_session_state();
    let mut fork_checkpoint = s.checkpoint_session_state(&a)?;
    fork_checkpoint.cache = a
        .cache
        .iter()
        .map(|slot| {
            slot.as_ref()
                .map(architectures::fork_layer_cache)
                .transpose()
        })
        .collect::<Result<Vec<_>>>()?;
    s.restore_session_checkpoint(&mut fork, fork_checkpoint);
    let before = s.total_kv_cache_allocation_tokens();
    fail.store(true, Ordering::Relaxed);
    assert!(
        s.forward_with_session(&mut a, StageInput::Tokens(vec![5, 6]), 2, 2)
            .is_err()
    );
    assert_eq!(a.next_position, 2);
    assert_eq!(history(&a.cache), [4, 3, -1]);
    assert_eq!(history(&b.cache), [7, -1, -1]);
    assert_eq!(s.total_kv_cache_allocation_tokens(), before);
    fail.store(false, Ordering::Relaxed);
    s.forward_with_session(&mut a, StageInput::Tokens(vec![5]), 1, 2)?;
    s.forward_with_session(&mut fork, StageInput::Tokens(vec![9]), 1, 2)?;
    assert_eq!(history(&a.cache), [5, 4, 3]);
    assert_eq!(history(&fork.cache), [9, 4, 3]);
    s.restore_session_checkpoint(&mut a, checkpoint);
    assert_eq!(a.next_position, 2);
    assert_eq!(history(&a.cache), [4, 3, -1]);
    s.forward_with_session(&mut a, StageInput::Tokens(vec![5]), 1, 2)?;
    assert_eq!(history(&a.cache), [5, 4, 3]);
    s.clear_session_state(&mut a);
    assert!(a.cache.iter().all(Option::is_none));
    assert_eq!(a.next_position, 0);
    s.forward_with_session(&mut a, StageInput::Tokens(vec![6]), 1, 0)?;
    assert_eq!(history(&a.cache), [6, -1, -1]);
    s.clear_session_state(&mut a);
    s.clear_session_state(&mut b);
    s.clear_session_state(&mut fork);
    assert_eq!(s.total_kv_cache_allocation_tokens(), 0);
    Ok(())
}

#[test]
fn deepseek41_context_budget_and_unsupported_input_errors_are_preflight_only() -> Result<()> {
    let (mut s, _) = stage();
    s.forward(StageInput::Tokens(vec![3, 4]), 2, 0)?;
    s.context_length = 2;
    let before = s.total_kv_cache_allocation_tokens();
    assert!(s.forward(StageInput::Tokens(vec![5]), 1, 2).is_err());
    s.context_length = 64;
    assert!(s.forward(StageInput::Tokens(vec![129280]), 1, 2).is_err());
    assert!(
        s.forward(
            StageInput::activations(Tensor::zeros((1, 5120), DType::F32, &Device::Cpu)?),
            1,
            2
        )
        .is_err()
    );
    assert!(
        s.forward_pass(
            StageInput::Tokens(vec![5]),
            1,
            2,
            ForwardPass::Standard {
                layer_ids: &[0],
                all_logits: false
            }
        )
        .is_err()
    );
    s.set_max_total_kv_tokens((before + 1) * 2)?;
    assert!(s.forward(StageInput::Tokens(vec![5, 6]), 2, 2).is_err());
    assert_eq!(s.next_position(), 2);
    assert_eq!(history(&s.cache), [4, 3, -1]);
    assert_eq!(s.total_kv_cache_allocation_tokens(), before);
    Ok(())
}

#[test]
fn native_tile_cpu_public_prefix_baseline_survives_tail_and_error_reset() -> Result<()> {
    use crate::continuous::GenerationSession;
    use crate::sampling::SamplingParams;
    use std::time::Duration;
    let (mut s, fail) = stage();
    s.context_length = 10_000;
    // Prompt-cache forks have their own stricter half-resident admission gate.
    // Keep that and the engine's independent copy-headroom split unchanged.
    let checkpoint_slots = FIXED_CACHE_BYTES.div_ceil(DYNAMIC_BYTES_PER_TOKEN) + 8195;
    s.set_max_total_kv_tokens(8 * checkpoint_slots)?;
    s.set_prompt_cache_limits(
        2 * (FIXED_CACHE_BYTES + 8195 * DYNAMIC_BYTES_PER_TOKEN),
        1,
        Duration::from_secs(3600),
    );
    let prompt: Vec<u32> = (0..8195).map(|i| (i % 17) as u32).collect();
    let params = SamplingParams {
        temperature: 0.,
        seed: Some(42),
        max_tokens: 1,
        ..Default::default()
    };
    let scope = "public-synthetic-prefix-baseline-cpu";
    let mut prefix = GenerationSession::new(&s, prompt[..8192].to_vec(), params.clone(), vec![]);
    prefix.enable_prompt_cache(scope.into());
    prefix.set_speculative_decoding_enabled(false);
    while !prefix.is_finished() {
        prefix.step_with_prefill_budget(&mut s, 128)?;
    }
    assert_eq!(prefix.generated().len(), 1);
    assert_eq!(prefix.next_position(&s), 8192); // sample never forwarded
    assert!(
        s.prompt_cache_bytes() > 0,
        "capture missing: logical={} allocated={} state={}",
        s.total_kv_cache_tokens(),
        s.total_kv_cache_allocation_tokens(),
        prefix.next_position(&s)
    );
    prefix.publish_prompt_cache(&mut s);
    prefix.discard_cache(&mut s);
    drop(prefix);
    let baseline = (
        s.total_kv_cache_tokens(),
        s.total_kv_cache_allocation_tokens(),
    );
    let fixed = FIXED_CACHE_BYTES.div_ceil(DYNAMIC_BYTES_PER_TOKEN);
    assert_eq!(baseline.0, 8192);
    assert!(baseline.1 >= fixed);
    for inject_error in [false, true] {
        let mut arm = GenerationSession::new(&s, prompt.clone(), params.clone(), vec![]);
        arm.set_speculative_decoding_enabled(false);
        arm.enable_prompt_cache(scope.into());
        assert!(arm.step_with_prefill_budget(&mut s, 1)?.is_none());
        assert_eq!(arm.reused_prompt_tokens(), 8192);
        assert_eq!(arm.next_position(&s), 8193);
        let mut cache = arm.into_cache();
        assert_eq!(cache.fixed_cache_allocation_tokens, fixed);
        let admitted = s.total_kv_cache_allocation_tokens();
        assert!(admitted >= baseline.1 + fixed);
        fail.store(inject_error, Ordering::Relaxed);
        let result = s.prefill_with_session(
            &mut cache,
            StageInput::Tokens(prompt[8193..].to_vec()),
            2,
            8193,
        );
        assert_eq!(result.is_err(), inject_error);
        if inject_error {
            assert_eq!(cache.next_position, 8193);
            assert_eq!(s.total_kv_cache_allocation_tokens(), admitted);
        } else {
            assert_eq!(cache.next_position, 8195);
        }
        fail.store(false, Ordering::Relaxed);
        s.clear_session_state(&mut cache);
        assert_eq!(
            (
                s.total_kv_cache_tokens(),
                s.total_kv_cache_allocation_tokens()
            ),
            baseline
        );
    }
    s.clear_prompt_cache();
    s.set_prompt_cache_limits(0, 0, Duration::ZERO);
    assert_eq!(
        (
            s.total_kv_cache_tokens(),
            s.total_kv_cache_allocation_tokens()
        ),
        (0, 0)
    );
    Ok(())
}

#[cfg(target_os = "macos")]
struct NativeMayaStateMachine {
    device: Device,
    fail_after_write: Arc<AtomicBool>,
    forward_calls: Arc<AtomicUsize>,
}

#[cfg(target_os = "macos")]
impl ModelArchitecture for NativeMayaStateMachine {
    fn forward(
        &self,
        input: StageInput,
        slots: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_calls.fetch_add(1, Ordering::Relaxed);
        let StageInput::Tokens(tokens) = input else {
            anyhow::bail!("tokens required");
        };
        anyhow::ensure!(tokens.len() == seq_len, "native fixture sequence mismatch");
        let mut cache = match &slots[0] {
            Some(LayerCache::DeepSeek41(c)) => (**c).clone(),
            None => DeepSeek41Cache::default(),
            _ => anyhow::bail!("wrong native fixture cache"),
        };
        anyhow::ensure!(
            cache.position() == start_pos,
            "native fixture position mismatch",
        );
        for id in tokens {
            // Position one fits even before the fix. Position two introduces
            // oversized pooled views into newly published owner rows.
            let oversized = cache.position() != 0;
            cache.advance_native_fixture(id as f32, &self.device, oversized)?;
        }
        // Match the real graph's fence before publishing persistent GPU state.
        self.device.synchronize()?;
        slots[0] = Some(LayerCache::DeepSeek41(Box::new(cache)));
        anyhow::ensure!(
            !self.fail_after_write.load(Ordering::Relaxed),
            "injected native post-write failure",
        );
        Ok(StageOutput::Logits(Tensor::new(&[1f32, 0.], &Device::Cpu)?))
    }

    fn family(&self) -> &str {
        "deepseek41"
    }
    fn hidden_dim(&self) -> usize {
        5120
    }
    fn layer_count(&self) -> usize {
        40
    }
    fn vocab_size(&self) -> usize {
        129280
    }
}

#[cfg(target_os = "macos")]
#[test]
fn deepseek41_metal_native_boundary_admission_rollback_and_fork() -> Result<()> {
    let device = Device::new_metal(0)?;
    let (mut s, fail) = stage();
    s.device = device.clone();
    s.arch = Box::new(NativeMayaStateMachine {
        device: device.clone(),
        fail_after_write: Arc::clone(&fail),
        forward_calls: Arc::new(AtomicUsize::new(0)),
    });
    let predicted = DeepSeek41Cache::predicted_allocation_tokens(2, &device)?;
    assert_eq!(predicted, 2);
    let fixed = s.fixed_cache_budget_tokens();
    s.set_max_total_kv_tokens(2 * (fixed + predicted))?;
    s.forward(StageInput::Tokens(vec![1]), 1, 0)?;
    let before = s.total_kv_cache_allocation_tokens();
    let snapshot = architectures::checkpoint_layer_cache(s.cache[0].as_ref().unwrap())?;
    let mut fork = architectures::fork_layer_cache(&snapshot)?;

    fail.store(true, Ordering::Relaxed);
    let error = s
        .forward(StageInput::Tokens(vec![2]), 1, 1)
        .err()
        .expect("injected failure");
    assert!(
        error
            .to_string()
            .contains("injected native post-write failure"),
        "{error:#}",
    );
    assert_eq!(s.next_position(), 1);
    assert_eq!(s.total_kv_cache_allocation_tokens(), before);
    let Some(LayerCache::DeepSeek41(restored)) = &s.cache[0] else {
        panic!("native rollback lost cache");
    };
    assert_eq!(restored.position(), 1);
    assert!(restored.compressed[0].pending.is_some());
    assert_eq!(
        restored.raw[0].as_ref().unwrap().to_vec2::<f32>()?,
        vec![vec![1.; 512]],
    );

    fail.store(false, Ordering::Relaxed);
    s.forward(StageInput::Tokens(vec![2]), 1, 1)?;
    assert_eq!(s.next_position(), 2);
    assert_eq!(s.total_kv_cache_allocation_tokens(), fixed + predicted);
    let Some(LayerCache::DeepSeek41(committed)) = &s.cache[0] else {
        panic!("native commit lost cache");
    };
    assert_eq!(committed.allocation_tokens(), 2);
    assert_eq!(
        committed.raw[0].as_ref().unwrap().to_vec2::<f32>()?,
        vec![vec![1.; 512], vec![2.; 512]],
    );
    let LayerCache::DeepSeek41(snapshot) = snapshot else {
        panic!("wrong checkpoint type");
    };
    let LayerCache::DeepSeek41(ref mut fork) = fork else {
        panic!("wrong fork type");
    };
    fork.advance_native_fixture(9., &device, true)?;
    device.synchronize()?;
    assert_eq!(snapshot.position(), 1);
    assert_eq!(
        snapshot.raw[0].as_ref().unwrap().to_vec2::<f32>()?,
        vec![vec![1.; 512]],
    );
    assert_eq!(
        snapshot.compressed[0]
            .pending
            .as_ref()
            .unwrap()
            .0
            .to_vec2::<f32>()?,
        vec![vec![1.; 512]],
    );
    assert_eq!(
        fork.raw[0].as_ref().unwrap().to_vec2::<f32>()?,
        vec![vec![1.; 512], vec![9.; 512]],
    );
    assert_eq!(
        committed.compressed[3].index.chunks[0].to_vec2::<f32>()?,
        vec![vec![1.; 128], vec![2.; 128]],
    );
    assert_eq!(
        fork.compressed[3].index.chunks[0].to_vec2::<f32>()?,
        vec![vec![1.; 128], vec![9.; 128]],
    );
    eprintln!(
        "native boundary committed position=2 fixed_units={fixed} predicted_units={predicted} resident_units={}",
        s.total_kv_cache_allocation_tokens(),
    );
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn deepseek41_metal_long_index_page_admission_rollback_and_fork() -> Result<()> {
    let device = Device::new_metal(0)?;
    // The retired predictor rounded both carriers at 256 rows. Keep it local
    // to this regression: it must demonstrably undercharge real Metal backing.
    let old_prediction = |position: usize| {
        let mut bytes = 0usize;
        for ratio in [2, 2, 2, 1] {
            let rows = position / ratio;
            for width in [512, 128] {
                let tail = rows % 256 * width * 4;
                bytes += rows / 256 * 256 * width * 4
                    + if tail == 0 {
                        0
                    } else {
                        tail.next_power_of_two()
                    };
            }
        }
        position.max(bytes.div_ceil(DYNAMIC_BYTES_PER_TOKEN))
    };
    let last_row = |tensor: &Tensor| -> Result<Vec<f32>> {
        Ok(tensor
            .narrow(0, tensor.dim(0)? - 1, 1)?
            .to_vec2::<f32>()?
            .remove(0))
    };
    let mut strict_undercharges = 0;
    for position in [1281, 1537, 2047, 2048, 2049, 2561, 3073, 4095, 4096, 4097] {
        let (mut s, fail) = stage();
        let calls = Arc::new(AtomicUsize::new(0));
        s.device = device.clone();
        s.context_length = 8192;
        s.arch = Box::new(NativeMayaStateMachine {
            device: device.clone(),
            fail_after_write: Arc::clone(&fail),
            forward_calls: Arc::clone(&calls),
        });
        let interior = matches!(position, 1281 | 1537 | 2561 | 3073);
        // Two native appends round every owner tail at interior cases; the
        // exact private seed itself still fits the old/new midpoint budget.
        let seq_len = if interior { 2 } else { 1 };
        let start = position - seq_len;
        let seed = DeepSeek41Cache::seeded_native_fixture(start, &device)?;
        device.synchronize()?;
        let before_report = seed.frontier_report()?;
        let before_units = seed.allocation_tokens();
        s.cache[0] = Some(LayerCache::DeepSeek41(Box::new(seed)));
        s.next_position = start;
        s.kv_cache_tokens = start;
        s.kv_cache_lane_capacity_tokens = start;
        s.kv_cache_allocation_tokens = before_units;
        s.kv_token_budget.add(start);
        s.kv_token_budget.add_allocated(before_units);
        s.charge_fixed_cache();
        let fixed = s.fixed_cache_budget_tokens();
        let before = s.total_kv_cache_allocation_tokens();
        let predicted = DeepSeek41Cache::predicted_allocation_tokens(position, &device)?;
        let old = old_prediction(position);
        let mut tight_resident = None;
        if interior {
            let resident = fixed + old + (predicted - old) / 2;
            assert!(fixed + old < resident && resident < fixed + predicted);
            // Preserve the production half-resident/half-copy-headroom split.
            s.set_max_total_kv_tokens(2 * resident)?;
            let error = s
                .forward(StageInput::Tokens(vec![2; seq_len]), seq_len, start)
                .err()
                .context("old/new midpoint admitted the long-index append")?;
            assert!(
                error.to_string().contains("resident KV allocation")
                    || error.to_string().contains("Maya history cannot be evicted"),
                "{error:#}",
            );
            assert_eq!(
                calls.load(Ordering::Relaxed),
                0,
                "native forward ran before rejection"
            );
            assert_eq!(s.next_position(), start);
            assert_eq!(s.total_kv_cache_tokens(), start);
            assert_eq!(s.total_kv_cache_allocation_tokens(), before);
            let Some(LayerCache::DeepSeek41(unchanged)) = &s.cache[0] else {
                panic!("admission rejection lost cache");
            };
            assert_eq!(unchanged.frontier_report()?, before_report);
            assert_eq!(last_row(unchanged.raw[0].as_ref().unwrap())?, vec![1.; 512]);
            for owner in &unchanged.compressed {
                assert_eq!(last_row(owner.index.chunks.last().unwrap())?, vec![1.; 128]);
            }
            tight_resident = Some(resident);
        }
        s.set_max_total_kv_tokens(2 * (fixed + predicted))?;
        let snapshot = if position == 4096 {
            Some(architectures::checkpoint_layer_cache(
                s.cache[0].as_ref().unwrap(),
            )?)
        } else {
            None
        };
        if let Some(snapshot) = &snapshot {
            let LayerCache::DeepSeek41(checkpoint) = snapshot else {
                unreachable!()
            };
            assert_eq!(checkpoint.compressed[0].index.len(), 2047);
            assert!(checkpoint.compressed[0].pending.is_some());
            fail.store(true, Ordering::Relaxed);
            let error = s
                .forward(StageInput::Tokens(vec![2]), 1, position - 1)
                .err()
                .context("page-closing post-write failure was not injected")?;
            assert!(
                error
                    .to_string()
                    .contains("injected native post-write failure"),
                "{error:#}"
            );
            assert_eq!(calls.load(Ordering::Relaxed), 1);
            assert_eq!(s.next_position(), position - 1);
            assert_eq!(s.total_kv_cache_tokens(), position - 1);
            assert_eq!(s.total_kv_cache_allocation_tokens(), before);
            let Some(LayerCache::DeepSeek41(restored)) = &s.cache[0] else {
                panic!("page rollback lost cache");
            };
            assert_eq!(restored.frontier_report()?, before_report);
            assert_eq!(last_row(restored.raw[0].as_ref().unwrap())?, vec![1.; 512]);
            for owner in &restored.compressed[..3] {
                assert_eq!(owner.index.len(), 2047);
                assert_eq!(last_row(owner.index.chunks.last().unwrap())?, vec![1.; 128]);
                assert_eq!(last_row(&owner.pending.as_ref().unwrap().0)?, vec![1.; 512]);
            }
            fail.store(false, Ordering::Relaxed);
        }
        s.forward(StageInput::Tokens(vec![2; seq_len]), seq_len, start)?;
        assert_eq!(
            calls.load(Ordering::Relaxed),
            if snapshot.is_some() { 2 } else { 1 }
        );
        assert_eq!(s.next_position(), position);
        let Some(LayerCache::DeepSeek41(committed)) = &s.cache[0] else {
            panic!("long-index commit lost cache");
        };
        committed.validate_frontier()?;
        let mut dynamic_bytes = 0;
        for owner in &committed.compressed {
            for tensor in owner.kv.chunks.iter().chain(&owner.index.chunks) {
                let (storage, _) = tensor.storage_and_layout();
                let candle_core::Storage::Metal(storage) = &*storage else {
                    panic!("fixture must retain native Metal storage");
                };
                dynamic_bytes += storage.buffer().length();
            }
        }
        let actual = position.max(dynamic_bytes.div_ceil(DYNAMIC_BYTES_PER_TOKEN));
        assert_eq!(committed.allocation_tokens(), actual);
        assert!(
            actual <= predicted,
            "position {position}: actual {actual} > predicted {predicted}"
        );
        assert_eq!(s.total_kv_cache_allocation_tokens(), fixed + actual);
        if actual > old {
            strict_undercharges += 1;
        }
        if interior {
            assert!(
                actual > old,
                "position {position}: fixture did not expose old undercharge"
            );
        }
        if let Some(snapshot) = snapshot {
            let mut fork = architectures::fork_layer_cache(&snapshot)?;
            let LayerCache::DeepSeek41(ref mut fork) = fork else {
                unreachable!()
            };
            fork.advance_native_fixture(9., &device, true)?;
            // Close the half-rate owners' first index2048 page, then carry an
            // odd token beyond it. The committed and checkpoint branches live.
            fork.advance_native_fixture(10., &device, true)?;
            device.synchronize()?;
            fork.validate_frontier()?;
            let LayerCache::DeepSeek41(checkpoint) = snapshot else {
                unreachable!()
            };
            assert_eq!(checkpoint.position(), 4095);
            assert_eq!(checkpoint.frontier_report()?, before_report);
            assert_eq!(fork.position(), 4097);
            assert_eq!(committed.position(), 4096);
            for owner in 0..3 {
                assert_eq!(checkpoint.compressed[owner].index.len(), 2047);
                assert_eq!(
                    last_row(checkpoint.compressed[owner].index.chunks.last().unwrap())?,
                    vec![1.; 128]
                );
                assert_eq!(
                    last_row(&checkpoint.compressed[owner].pending.as_ref().unwrap().0)?,
                    vec![1.; 512]
                );
                assert_eq!(committed.compressed[owner].index.len(), 2048);
                assert!(committed.compressed[owner].pending.is_none());
                assert_eq!(
                    last_row(committed.compressed[owner].index.chunks.last().unwrap())?,
                    vec![2.; 128]
                );
                assert_eq!(fork.compressed[owner].index.len(), 2048);
                assert_eq!(
                    last_row(fork.compressed[owner].index.chunks.last().unwrap())?,
                    vec![9.; 128]
                );
                assert_eq!(
                    last_row(&fork.compressed[owner].pending.as_ref().unwrap().0)?,
                    vec![10.; 512]
                );
            }
            assert_eq!(
                last_row(checkpoint.raw[0].as_ref().unwrap())?,
                vec![1.; 512]
            );
            assert_eq!(last_row(committed.raw[0].as_ref().unwrap())?, vec![2.; 512]);
            assert_eq!(last_row(fork.raw[0].as_ref().unwrap())?, vec![10.; 512]);
            assert!(
                fork.allocation_tokens()
                    <= DeepSeek41Cache::predicted_allocation_tokens(4097, &device)?
            );
        }
        eprintln!(
            "long-index admission position={position} actual_units={actual} dynamic_bytes={dynamic_bytes} old_units={old} predicted_units={predicted} fixed_units={fixed} tight_resident={tight_resident:?} calls={} rollback_fork={}",
            calls.load(Ordering::Relaxed),
            position == 4096,
        );
    }
    assert!(
        strict_undercharges > 0,
        "no real retained allocation exceeded the retired predictor"
    );
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn deepseek41_metal_cancelled_input_native_forks_match_cold_and_revoke() -> Result<()> {
    use crate::continuous::GenerationSession;
    use crate::sampling::SamplingParams;
    let device = Device::new_metal(0)?;
    let (mut s, fail) = stage();
    s.device = device.clone();
    s.context_length = 1024;
    s.arch = Box::new(NativeMayaStateMachine {
        device: device.clone(),
        fail_after_write: fail,
        forward_calls: Arc::new(AtomicUsize::new(0)),
    });
    let checkpoint_slots = FIXED_CACHE_BYTES.div_ceil(DYNAMIC_BYTES_PER_TOKEN) + s.context_length;
    // Stopped session + retained checkpoint + two forks must fit the existing
    // half-resident fork gate, on top of the independent half-copy split.
    s.set_max_total_kv_tokens(4 * 4 * checkpoint_slots)?;
    s.set_prompt_cache_limits(
        2 * (FIXED_CACHE_BYTES + s.context_length * DYNAMIC_BYTES_PER_TOKEN),
        1,
        std::time::Duration::from_secs(3600),
    );
    let scope = "reserved:public-native-cancel";
    assert!(s.register_reserved_prompt_scope(scope.into()));
    let prompt: Vec<u32> = (0..131).map(|i| i % 17).collect();
    let params = SamplingParams {
        temperature: 0.,
        seed: Some(42),
        max_tokens: 1,
        ..Default::default()
    };
    let mut stopped = GenerationSession::new(&s, prompt.clone(), params.clone(), vec![]);
    stopped.enable_prompt_cache(scope.into());
    stopped.set_speculative_decoding_enabled(false);
    assert!(stopped.step_with_prefill_budget(&mut s, 128)?.is_none());
    assert_eq!(stopped.next_position(&s), 128);
    let charge = s.total_kv_cache_allocation_tokens();
    assert!(stopped.retain_cancelled_input_prefix(&mut s, scope));
    assert_eq!(s.total_kv_cache_allocation_tokens(), charge + 1);
    assert_eq!(s.prompt_cache_match_tokens(scope, &prompt[..128]), (0, 0));
    assert_eq!(s.prompt_cache_match_tokens(scope, &prompt), (128, 0));
    let (mut a, no_logits, skipped) = s.reuse_prompt(scope, &prompt, false)?.unwrap();
    let mut other = prompt.clone();
    other[128..].copy_from_slice(&[31, 32, 33]);
    let (mut b, _, _) = s.reuse_prompt(scope, &other, false)?.unwrap();
    assert!(no_logits.is_none());
    assert_eq!(skipped, 128);
    let StageOutput::Logits(warm) =
        s.prefill_with_session(&mut a, StageInput::Tokens(prompt[128..].to_vec()), 3, 128)?
    else {
        unreachable!()
    };
    s.prefill_with_session(&mut b, StageInput::Tokens(other[128..].to_vec()), 3, 128)?;
    let mut cold = s.new_session_state();
    s.prefill_cache_only_with_session(
        &mut cold,
        StageInput::Tokens(prompt[..128].to_vec()),
        128,
        0,
    )?;
    let StageOutput::Logits(reference) = s.prefill_with_session(
        &mut cold,
        StageInput::Tokens(prompt[128..].to_vec()),
        3,
        128,
    )?
    else {
        unreachable!()
    };
    let warm = warm.to_vec1::<f32>()?;
    assert!(warm.iter().all(|x| x.is_finite()));
    assert_eq!(warm, reference.to_vec1::<f32>()?);
    let cache = |state: &PipelineStageSessionState| {
        let Some(LayerCache::DeepSeek41(cache)) = &state.cache[0] else {
            panic!("missing native cache")
        };
        cache.clone()
    };
    let a_cache = cache(&a);
    let b_cache = cache(&b);
    let cold_cache = cache(&cold);
    assert_eq!(a_cache.position(), 131);
    assert_eq!(
        a_cache.raw[0].as_ref().unwrap().to_vec2::<f32>()?,
        cold_cache.raw[0].as_ref().unwrap().to_vec2::<f32>()?
    );
    assert_ne!(
        a_cache.raw[0].as_ref().unwrap().to_vec2::<f32>()?,
        b_cache.raw[0].as_ref().unwrap().to_vec2::<f32>()?
    );
    assert_eq!(
        a_cache.compressed[0]
            .pending
            .as_ref()
            .unwrap()
            .0
            .to_vec2::<f32>()?,
        cold_cache.compressed[0]
            .pending
            .as_ref()
            .unwrap()
            .0
            .to_vec2::<f32>()?
    );
    assert_ne!(
        a_cache.compressed[0]
            .pending
            .as_ref()
            .unwrap()
            .0
            .to_vec2::<f32>()?,
        b_cache.compressed[0]
            .pending
            .as_ref()
            .unwrap()
            .0
            .to_vec2::<f32>()?
    );
    assert_eq!(
        a_cache.compressed[3].index.chunks[0].to_vec2::<f32>()?,
        cold_cache.compressed[3].index.chunks[0].to_vec2::<f32>()?
    );
    assert_ne!(
        a_cache.compressed[3].index.chunks[0].to_vec2::<f32>()?,
        b_cache.compressed[3].index.chunks[0].to_vec2::<f32>()?
    );
    // Four charged fixed-state owners exhaust fork/copy headroom. A miss
    // must preserve the immutable entry for admission after peers release.
    assert!(s.reuse_prompt(scope, &prompt, false)?.is_none());
    assert_eq!(s.prompt_cache_match_tokens(scope, &prompt), (0, 0));
    drop((a_cache, b_cache, cold_cache, a, b, cold));
    let (unchanged, _, _) = s.reuse_prompt(scope, &prompt, false)?.unwrap();
    assert_eq!(cache(&unchanged).position(), 128);
    drop(unchanged);
    let mut resumed = GenerationSession::new(&s, prompt.clone(), params.clone(), vec![]);
    resumed.enable_prompt_cache(scope.into());
    resumed.set_speculative_decoding_enabled(false);
    resumed.step_with_prefill_budget(&mut s, 128)?;
    let mut cold = GenerationSession::new(&s, prompt, params, vec![]);
    cold.set_speculative_decoding_enabled(false);
    while !cold.is_finished() {
        cold.step_with_prefill_budget(&mut s, 128)?;
    }
    assert_eq!(resumed.reused_prompt_tokens(), 128);
    assert_eq!(resumed.generated(), cold.generated());
    eprintln!(
        "native synthetic Maya cancellation: chunk=128 cursor=128 prompt=131 finite_logits=2 max_abs_diff=0 tokens={:?} raw/page/carry=independent",
        resumed.generated()
    );
    resumed.discard_cache(&mut s);
    cold.discard_cache(&mut s);
    s.revoke_reserved_prompt_scope(scope);
    drop(stopped);
    device.synchronize()?;
    assert_eq!(s.total_kv_cache_tokens(), 0);
    assert_eq!(s.total_kv_cache_allocation_tokens(), 0);
    Ok(())
}
