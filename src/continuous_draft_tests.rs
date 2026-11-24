//! Sampling and lane-state checks with a real small DFlash and a causal target.
//! These test session control; Qwen's separate batch tests check target math.

use super::*;
use crate::architectures::{KvCache, LayerCache, ModelArchitecture, SpeculativeBatchOutput};
use candle_core::{Device, Tensor};
use std::sync::{Arc, Mutex};

// Enabled only by the ignored full-target test. Keep the actual rows if a
// paired stream fails, so its report can separate target math from sampling.
thread_local! {
    static TARGET_ROWS: std::cell::RefCell<Option<Vec<(Tensor, bool)>>> = const { std::cell::RefCell::new(None) };
}

pub(super) fn capture_target_rows(logits: &Tensor, verification: bool) {
    TARGET_ROWS.with_borrow_mut(|rows| {
        if let Some(rows) = rows {
            rows.push((logits.clone(), verification));
        }
    });
}

#[cfg(target_os = "macos")]
fn begin_capture() {
    TARGET_ROWS.with_borrow_mut(|rows| *rows = Some(Vec::new()));
}

#[cfg(target_os = "macos")]
fn take_capture() -> Vec<(Tensor, bool)> {
    TARGET_ROWS.with_borrow_mut(|rows| rows.take().unwrap())
}

#[derive(Clone, Copy, Default)]
enum Fault {
    #[default]
    None,
    Target,
    Sample,
}

#[derive(Default)]
struct Calls {
    draft_rows: usize,
    shared: usize,
    fault: Fault,
    stop_at: Option<usize>,
}

struct CausalTarget(Arc<Mutex<Calls>>);

fn cache_tokens(cache: &[Option<LayerCache>]) -> Result<Vec<f32>> {
    match &cache[0] {
        Some(LayerCache::Attention(KvCache::F16 { k, .. })) => {
            Ok(k.flatten_all()?.to_vec1::<f32>()?)
        }
        None => Ok(Vec::new()),
        Some(_) => anyhow::bail!("unexpected test cache"),
    }
}

impl CausalTarget {
    fn run(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start: usize,
        all: bool,
    ) -> Result<StageOutput> {
        let StageInput::Tokens(tokens) = input else {
            anyhow::bail!("test requires tokens")
        };
        anyhow::ensure!(tokens.len() == seq_len && start + seq_len <= 128);
        let mut past = cache_tokens(cache)?;
        anyhow::ensure!(past.len() == start, "target lost its causal prefix");
        let mut scores = Vec::with_capacity(seq_len * 3);
        for token in tokens {
            past.push(token as f32);
            let sum = past.iter().sum::<f32>();
            // Every row depends on its full causal prefix. The zero-prefixed
            // case also gives a certain all-accepted block under top-k one.
            let forced_stop = self.0.lock().unwrap().stop_at;
            let row = if let Some(stop) = forced_stop {
                if past.len() - 1 == stop {
                    [-1.0, 2.0, -2.0]
                } else {
                    [2.0, -1.0, -2.0]
                }
            } else if past[0] == 0.0 {
                [1.2, -0.2 + sum * 0.03, 0.1]
            } else {
                [0.15 + sum * 0.013, (past.len() % 5) as f32 * 0.17, -0.2]
            };
            scores.extend(row);
        }
        let tensor = Tensor::from_vec(past, (1, 1, start + seq_len, 1), &Device::Cpu)?;
        cache[0] = Some(LayerCache::Attention(KvCache::F16 {
            k: tensor.clone(),
            v: tensor,
        }));
        if all {
            self.0.lock().unwrap().draft_rows += seq_len;
        }
        let logits = Tensor::from_vec(scores, (seq_len, 3), &Device::Cpu)?;
        Ok(StageOutput::Logits(if all {
            logits
        } else {
            logits.get(seq_len - 1)?
        }))
    }
}

impl ModelArchitecture for CausalTarget {
    fn forward(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start: usize,
    ) -> Result<StageOutput> {
        self.run(input, cache, seq_len, start, false)
    }

    fn forward_with_layer_inputs(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start: usize,
        layers: &[usize],
        all: bool,
    ) -> Result<(StageOutput, Vec<Tensor>)> {
        let output = self.run(input, cache, seq_len, start, all)?;
        let captures = layers
            .iter()
            .map(|_| Tensor::zeros((seq_len, 64), DType::F32, &Device::Cpu))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok((output, captures))
    }

    fn supports_speculative_batch(&self) -> bool {
        true
    }

    fn forward_speculative_batch(
        &self,
        inputs: &[Vec<u32>],
        caches: &mut [&mut [Option<LayerCache>]],
        positions: &[usize],
        layers: &[usize],
    ) -> Result<Vec<SpeculativeBatchOutput>> {
        self.0.lock().unwrap().shared += 1;
        let fault = self.0.lock().unwrap().fault;
        let mut outputs = Vec::new();
        for (lane, ((input, cache), &start)) in inputs.iter().zip(caches).zip(positions).enumerate()
        {
            let (output, captures) = self.forward_with_layer_inputs(
                StageInput::Tokens(input.clone()),
                cache,
                input.len(),
                start,
                layers,
                true,
            )?;
            let StageOutput::Logits(mut logits) = output else {
                unreachable!()
            };
            if lane == 1 {
                match fault {
                    // Deliberately fail after two cache writes: the outer
                    // transaction must restore every lane, even on error.
                    Fault::Target => anyhow::bail!("injected shared target failure"),
                    Fault::Sample => {
                        logits = Tensor::full(f32::NAN, (input.len(), 3), &Device::Cpu)?;
                    }
                    Fault::None => (),
                }
            }
            outputs.push(SpeculativeBatchOutput { logits, captures });
        }
        Ok(outputs)
    }

    fn family(&self) -> &str {
        "causal-draft-test"
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

fn stage(draft: bool) -> Result<(PipelineStage, Arc<Mutex<Calls>>)> {
    let (mut stage, _) = crate::session_state_tests::speculative_boundary_stage(128, None)?;
    let calls = Arc::new(Mutex::new(Calls::default()));
    stage.arch = Box::new(CausalTarget(Arc::clone(&calls)));
    if draft {
        let draft: crate::architectures::speculative_draft::SpeculativeDraft =
            crate::architectures::dflash2::DFlash2Draft::test_zero_draft(8, 128)?.into();
        stage.fixed_cache_bytes = draft.cache_budget_bytes()?;
        stage.speculative_cache = Some(draft.new_cache());
        stage.speculative = Some(draft);
    }
    Ok((stage, calls))
}

fn params(profile: usize) -> SamplingParams {
    SamplingParams {
        temperature: 0.85,
        top_k: if profile == 0 {
            1
        } else if profile == 1 {
            2
        } else {
            0
        },
        top_p: if profile == 2 { 0.68 } else { 1.0 },
        repetition_penalty: if profile == 3 { 1.3 } else { 1.0 },
        frequency_penalty: if profile == 3 { 0.15 } else { 0.0 },
        presence_penalty: if profile == 3 { 0.2 } else { 0.0 },
        max_tokens: 24,
        min_tokens: 24,
        seed: None,
        ..Default::default()
    }
}

fn session(
    stage: &PipelineStage,
    prompt: Vec<u32>,
    profile: usize,
    seed: u64,
) -> GenerationSession {
    let mut session = GenerationSession::new(stage, prompt, params(profile), vec![]);
    // The public request remains unseeded, so it must execute the draft path.
    // Pair only its private test RNG with the scalar reference's RNG.
    session.rng = SamplingRng::seeded(seed);
    session
}

fn same_committed_state(a: &GenerationSession, b: &GenerationSession) -> Result<()> {
    assert!(b.pending_tokens.is_empty());
    assert_eq!(a.generated, b.generated);
    assert_eq!(a.cache.next_position, b.cache.next_position);
    assert_eq!(a.cache.kv_cache_tokens, b.cache.kv_cache_tokens);
    assert_eq!(cache_tokens(&a.cache.cache)?, cache_tokens(&b.cache.cache)?);
    let draft_position = b.cache.speculative_cache.as_ref().unwrap().next_position();
    if b.speculative_disabled {
        assert!(b.speculative_stats().fell_back);
        assert!(draft_position <= b.cache.next_position);
    } else {
        assert_eq!(draft_position, b.cache.next_position);
    }
    // Check the next draws and penalty history without altering either lane.
    let (mut x, mut y) = (a.rng.clone(), b.rng.clone());
    let broad = SamplingParams {
        top_k: 0,
        top_p: 1.0,
        ..a.params.clone()
    };
    for _ in 0..32 {
        assert_eq!(
            a.sampling_history
                .sample_host(vec![0.3, -0.1, 0.2], &broad, &mut x)?,
            b.sampling_history
                .sample_host(vec![0.3, -0.1, 0.2], &broad, &mut y)?,
            "a draft pass spent an extra random draw or changed penalty history"
        );
    }
    Ok(())
}

#[test]
fn draft_sampling_matches_target_for_paired_streams() -> Result<()> {
    let (mut scalar, _) = stage(false)?;
    let (mut drafted, calls) = stage(true)?;
    let mut passes = 0;
    let mut accepted = 0;
    let mut rejected = 0;
    for profile in 0..4 {
        for seed in 1..=24 {
            let prompt = vec![(profile % 3) as u32, 1, 2];
            let mut a = session(&scalar, prompt.clone(), profile, seed);
            let mut b = session(&drafted, prompt, profile, seed);
            assert!(b.speculative_eligible(&drafted));
            while !a.is_finished() {
                assert_eq!(a.step(&mut scalar)?, b.step(&mut drafted)?);
                if b.pending_tokens.is_empty() {
                    same_committed_state(&a, &b)?;
                }
            }
            let stats = b.speculative_stats();
            passes += stats.passes;
            accepted += stats.accepted_tokens;
            rejected += stats.drafted_tokens - stats.accepted_tokens;
            a.discard_cache(&mut scalar);
            b.discard_cache(&mut drafted);
        }
    }
    assert!(passes > 96 && accepted > 100 && rejected > 100);
    assert!(calls.lock().unwrap().draft_rows >= passes * 8);
    assert_eq!(scalar.total_kv_cache_allocation_tokens(), 0);
    assert_eq!(drafted.total_kv_cache_allocation_tokens(), 0);
    Ok(())
}

#[test]
fn each_draft_rejection_position_and_bonus_stop_matches_scalar() -> Result<()> {
    for row in 0..8 {
        let (mut scalar, scalar_calls) = stage(false)?;
        let (mut drafted, draft_calls) = stage(true)?;
        scalar_calls.lock().unwrap().stop_at = Some(row + 1);
        draft_calls.lock().unwrap().stop_at = Some(row + 1);
        let mut a = session(&scalar, vec![0], 0, 42);
        let mut b = session(&drafted, vec![0], 0, 42);
        for s in [&mut a, &mut b] {
            s.params.min_tokens = 0;
            s.stop_ids = vec![1];
        }
        while !a.is_finished() {
            assert_eq!(a.step(&mut scalar)?, b.step(&mut drafted)?);
        }
        assert_eq!(b.speculative_stats().passes, 1);
        assert_eq!(b.speculative_stats().accepted_tokens, row.min(7));
        same_committed_state(&a, &b)?;
        a.discard_cache(&mut scalar);
        b.discard_cache(&mut drafted);
        assert_eq!(drafted.total_kv_cache_allocation_tokens(), 0);
    }
    Ok(())
}

/// A drafted stage with shared verification turned on. Other stages keep
/// the default options, so these tests no longer need a child process.
fn shared_stage() -> Result<(PipelineStage, Arc<Mutex<Calls>>)> {
    let (mut stage, calls) = stage(true)?;
    stage.set_options(crate::EngineOptions {
        speculative: crate::SpeculativeOptions {
            shared_verification: true,
            ..Default::default()
        },
        ..Default::default()
    })?;
    Ok((stage, calls))
}

#[test]
fn shared_draft_lanes_match_separate_target_streams() -> Result<()> {
    for lanes in [2, 4] {
        let (mut shared, calls) = shared_stage()?;
        let (mut scalar, _) = stage(false)?;
        let mut a = (0..lanes)
            .map(|i| session(&scalar, vec![i as u32 % 3; i + 1], i, 17 + i as u64))
            .collect::<Vec<_>>();
        let mut b = (0..lanes)
            .map(|i| session(&shared, vec![i as u32 % 3; i + 1], i, 17 + i as u64))
            .collect::<Vec<_>>();
        while b.iter().any(|s| !s.is_finished()) {
            let eligible = b
                .iter()
                .filter(|s| s.can_speculative_batch(&shared))
                .count();
            let mut results = Vec::new();
            if eligible >= 2 {
                let mut selected = b
                    .iter_mut()
                    .enumerate()
                    .filter(|(_, s)| s.can_speculative_batch(&shared))
                    .collect::<Vec<_>>();
                let indices = selected.iter().map(|(i, _)| *i).collect::<Vec<_>>();
                let mut refs = selected
                    .iter_mut()
                    .map(|(_, s)| &mut **s)
                    .collect::<Vec<_>>();
                let steps = GenerationSession::step_speculative_batch(&mut shared, &mut refs)
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                results.extend(indices.into_iter().zip(steps));
            } else {
                for (i, lane) in b.iter_mut().enumerate().filter(|(_, s)| !s.is_finished()) {
                    results.push((i, lane.step(&mut shared)));
                }
            }
            for (i, result) in results {
                assert_eq!(result?, a[i].step(&mut scalar)?);
                if b[i].pending_tokens.is_empty() {
                    same_committed_state(&a[i], &b[i])?;
                }
            }
        }
        assert!(calls.lock().unwrap().shared >= 2);
        assert!(b.iter().all(|s| s.speculative_stats().passes > 0));
        for lane in &mut a {
            lane.discard_cache(&mut scalar);
        }
        for lane in &mut b {
            lane.discard_cache(&mut shared);
        }
        assert_eq!(shared.total_kv_cache_allocation_tokens(), 0);
    }
    Ok(())
}

#[test]
fn shared_draft_errors_restore_only_uncommitted_lanes() -> Result<()> {
    for fault in [Fault::Target, Fault::Sample] {
        let (mut shared, calls) = shared_stage()?;
        let (mut scalar, _) = stage(false)?;
        let mut a = session(&scalar, vec![0], 0, 42);
        let mut lanes = [
            session(&shared, vec![0], 0, 42),
            session(&shared, vec![1, 2], 3, 91),
        ];
        assert_eq!(a.step(&mut scalar)?, lanes[0].step(&mut shared)?);
        lanes[1].step(&mut shared)?;
        let before = lanes
            .iter()
            .map(|s| {
                (
                    s.cache.next_position,
                    cache_tokens(&s.cache.cache).unwrap(),
                    s.rng.clone(),
                    s.generated.clone(),
                )
            })
            .collect::<Vec<_>>();
        calls.lock().unwrap().fault = fault;
        let result = GenerationSession::step_speculative_batch(
            &mut shared,
            &mut lanes.iter_mut().collect::<Vec<_>>(),
        );
        let failed_lanes = if matches!(fault, Fault::Target) {
            assert!(matches!(result, Err(DecodeBatchStepError::Terminal(_))));
            0..2
        } else {
            let results = result.map_err(|e| anyhow::anyhow!("{e:?}"))?;
            assert_eq!(*results[0].as_ref().unwrap(), a.step(&mut scalar)?);
            assert!(results[1].is_err());
            while !lanes[0].pending_tokens.is_empty() {
                assert_eq!(lanes[0].step(&mut shared)?, a.step(&mut scalar)?);
            }
            same_committed_state(&a, &lanes[0])?;
            1..2
        };
        for i in failed_lanes {
            assert_eq!(lanes[i].cache.next_position, before[i].0);
            assert_eq!(cache_tokens(&lanes[i].cache.cache)?, before[i].1);
            assert_eq!(lanes[i].generated, before[i].3);
            assert!(lanes[i].pending_tokens.is_empty());
            assert_eq!(
                lanes[i]
                    .cache
                    .speculative_cache
                    .as_ref()
                    .unwrap()
                    .next_position(),
                before[i].0
            );
            let (mut got, mut expected) = (lanes[i].rng.clone(), before[i].2.clone());
            let broad = SamplingParams {
                top_k: 0,
                top_p: 1.0,
                ..lanes[i].params.clone()
            };
            for _ in 0..32 {
                assert_eq!(
                    lanes[i].sampling_history.sample_host(
                        vec![0.2, 0.1, -0.1],
                        &broad,
                        &mut got
                    )?,
                    lanes[i].sampling_history.sample_host(
                        vec![0.2, 0.1, -0.1],
                        &broad,
                        &mut expected
                    )?
                );
            }
        }
        // A target or sampling error ends this attempt. Do not silently retry
        // the failed lane; release its target and draft state explicitly.
        for lane in &mut lanes {
            lane.discard_cache(&mut shared);
        }
        a.discard_cache(&mut scalar);
        assert_eq!(shared.total_kv_cache_allocation_tokens(), 0);
    }
    Ok(())
}

#[test]
fn shared_draft_cancel_keeps_other_lane_and_rebuilds_visible_prefix() -> Result<()> {
    let (mut shared, calls) = shared_stage()?;
    let (mut scalar, _) = stage(false)?;
    shared.set_prompt_cache_limits(1 << 20, 4, Duration::from_secs(60));
    let mut a = session(&shared, vec![0], 0, 42);
    let mut b = session(&shared, vec![0, 0], 0, 43);
    let mut b_reference = session(&scalar, vec![0, 0], 0, 43);
    a.enable_prompt_cache("shared-cancel".into());
    a.step(&mut shared)?;
    assert_eq!(b.step(&mut shared)?, b_reference.step(&mut scalar)?);
    let steps = GenerationSession::step_speculative_batch(&mut shared, &mut [&mut a, &mut b])
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    assert!(steps[0].is_ok());
    assert_eq!(*steps[1].as_ref().unwrap(), b_reference.step(&mut scalar)?);
    assert_eq!(calls.lock().unwrap().shared, 1);
    assert_eq!(a.pending_tokens.len(), 7);
    let b_cache = cache_tokens(&b.cache.cache)?;
    let b_position = b.cache.next_position;
    a.finish_text_stop();
    assert!(a.cache.resume_invalidated);
    assert_eq!(a.cache.next_position, 0);
    assert_eq!(
        a.cache.speculative_cache.as_ref().unwrap().next_position(),
        0
    );
    assert_eq!(cache_tokens(&b.cache.cache)?, b_cache);
    assert_eq!(b.cache.next_position, b_position);
    a.publish_prompt_cache(&mut shared);
    assert_eq!(
        shared.prompt_cache_match_tokens("shared-cancel", &[0]),
        (0, 1)
    );
    while !b.is_finished() {
        assert_eq!(b.step(&mut shared)?, b_reference.step(&mut scalar)?);
    }
    same_committed_state(&b_reference, &b)?;
    b.discard_cache(&mut shared);
    b_reference.discard_cache(&mut scalar);

    let mut visible = vec![0];
    visible.extend(a.generated());
    visible.push(1);
    let mut resumed = GenerationSession::with_cache(
        a.into_cache(),
        visible.clone(),
        params(3),
        vec![],
        visible.len() - 1,
    );
    resumed.rng = SamplingRng::seeded(77);
    let mut reference = session(&scalar, visible, 3, 77);
    while !reference.is_finished() {
        assert_eq!(resumed.step(&mut shared)?, reference.step(&mut scalar)?);
    }
    same_committed_state(&reference, &resumed)?;
    assert!(resumed.speculative_stats().passes > 0);
    resumed.discard_cache(&mut shared);
    reference.discard_cache(&mut scalar);
    let mut warm = session(&shared, vec![0], 0, 42);
    warm.enable_prompt_cache("shared-cancel".into());
    warm.step(&mut shared)?;
    assert_eq!(warm.reused_prompt_tokens(), 1);
    assert!(shared.session_speculative_ready(&warm.cache));
    warm.step(&mut shared)?;
    assert!(warm.speculative_stats().passes > 0);
    warm.discard_cache(&mut shared);
    shared.clear_prompt_cache();
    assert_eq!(shared.total_kv_cache_allocation_tokens(), 0);
    Ok(())
}

/// Runs only with the caller's already-admitted target/draft files and device.
#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires the admitted full target and direct draft; no downloads"]
fn full_target_shared_draft_paired_streams() -> Result<()> {
    use crate::{LayerRange, read_gguf_metadata, tokenizer::GgufTokenizer};
    use std::path::PathBuf;

    let target = PathBuf::from(std::env::var("DINF_TEST_SHARED_TARGET")?);
    let draft = PathBuf::from(std::env::var("DINF_TEST_SHARED_DRAFT")?);
    let report_path = PathBuf::from(std::env::var("DINF_TEST_SHARED_REPORT")?);
    anyhow::ensure!(!report_path.exists(), "fresh report required");
    let metadata = read_gguf_metadata(&target)?;
    let tokenizer = GgufTokenizer::from_gguf(&target)?;
    let device = Device::new_metal(0)?;
    {
        let mut stage = PipelineStage::from_gguf(
            &target,
            LayerRange::try_solo(metadata.num_layers)?,
            &device,
            None,
        )?;
        stage.attach_speculative_draft(&draft)?;
        stage.set_options(crate::EngineOptions {
            speculative: crate::SpeculativeOptions {
                shared_verification: true,
                ..Default::default()
            },
            ..Default::default()
        })?;
        anyhow::ensure!(
            stage.supports_speculative_batch(),
            "shared path did not admit"
        );
        let texts = [
            "Write a Rust function to add two integers.",
            "List three small prime numbers and explain why each is prime.",
            "Give a short example of a JSON object with two string fields.",
            "Explain how to merge sorted lists in plain words. Include one example.",
        ];
        let prompts = texts
            .iter()
            .map(|s| tokenizer.encode_chat(&[("user".into(), (*s).into())]))
            .collect::<Result<Vec<_>>>()?;
        let mut report = serde_json::json!({"passed":false,"target":target,"draft":draft,"lanes":[],"tokens_per_lane":24,"approximate":false,"checks":"paired sampled tokens/RNG/positions and owned cache cleanup; conditional on the tested target rows"});
        let result = (|| -> Result<()> {
            for count in [2, 4] {
                let mut a = Vec::new();
                let mut b = Vec::new();
                let mut last_rows: Vec<Option<(Tensor, usize)>> = vec![None; count];
                for (i, prompt) in prompts.iter().take(count).enumerate() {
                    let mut p = params(i);
                    p.top_k = [40, 8, 0, 40][i];
                    let mut plain =
                        GenerationSession::new(&stage, prompt.clone(), p.clone(), vec![]);
                    let mut drafted = GenerationSession::new(&stage, prompt.clone(), p, vec![]);
                    plain.set_speculative_decoding_enabled(&stage, false);
                    plain.rng = SamplingRng::seeded(413 + i as u64);
                    drafted.rng = SamplingRng::seeded(413 + i as u64);
                    a.push(plain);
                    b.push(drafted);
                }
                let mut shared_passes = 0;
                while b.iter().any(|s| !s.is_finished()) {
                    let eligible = b.iter().filter(|s| s.can_speculative_batch(&stage)).count();
                    let mut results = Vec::new();
                    if eligible >= 2 {
                        let mut selected = b
                            .iter_mut()
                            .enumerate()
                            .filter(|(_, s)| s.can_speculative_batch(&stage))
                            .collect::<Vec<_>>();
                        let indices = selected.iter().map(|(i, _)| *i).collect::<Vec<_>>();
                        let starts = selected
                            .iter()
                            .map(|(_, s)| s.generated.len())
                            .collect::<Vec<_>>();
                        let mut refs = selected
                            .iter_mut()
                            .map(|(_, s)| &mut **s)
                            .collect::<Vec<_>>();
                        begin_capture();
                        let steps =
                            GenerationSession::step_speculative_batch(&mut stage, &mut refs)
                                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                        let rows = take_capture();
                        anyhow::ensure!(
                            rows.len() == indices.len() && rows.iter().all(|(_, all)| *all)
                        );
                        for ((&i, start), (tensor, _)) in indices.iter().zip(starts).zip(rows) {
                            last_rows[i] = Some((tensor, start));
                        }
                        shared_passes += 1;
                        results.extend(indices.into_iter().zip(steps));
                    } else {
                        for (i, lane) in b.iter_mut().enumerate().filter(|(_, s)| !s.is_finished())
                        {
                            let start = lane.generated.len();
                            begin_capture();
                            let step = lane.step(&mut stage);
                            let rows = take_capture();
                            if let Some((tensor, all)) = rows.into_iter().last() {
                                last_rows[i] = Some((
                                    if all {
                                        tensor
                                    } else {
                                        tensor.reshape((1, stage.vocab_size()))?
                                    },
                                    start,
                                ));
                            }
                            results.push((i, step));
                        }
                    }
                    for (i, step) in results {
                        let actual = step?;
                        let token_index = a[i].generated.len();
                        let common_prefix = a[i].generated.clone();
                        begin_capture();
                        let expected = a[i].step(&mut stage)?;
                        let scalar_rows = take_capture();
                        if actual != expected {
                            let (row, start) =
                                last_rows[i].as_ref().context("missing actual target row")?;
                            let actual_scores = row
                                .get(token_index - start)?
                                .to_dtype(DType::F32)?
                                .flatten_all()?
                                .to_vec1::<f32>()?;
                            let expected_scores = scalar_rows
                                .last()
                                .context("missing scalar target row")?
                                .0
                                .to_dtype(DType::F32)?
                                .flatten_all()?
                                .to_vec1::<f32>()?;
                            let max_abs = actual_scores
                                .iter()
                                .zip(&expected_scores)
                                .map(|(x, y)| (x - y).abs())
                                .fold(0.0f32, f32::max);
                            let rms = (actual_scores
                                .iter()
                                .zip(&expected_scores)
                                .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
                                .sum::<f64>()
                                / actual_scores.len() as f64)
                                .sqrt();
                            report["first_mismatch"] = serde_json::json!({"lane_count":count,"lane":i,"output_row":token_index,"prompt":prompts[i],"common_generated_prefix":common_prefix,"params":format!("{:?}",b[i].params),"actual_token":actual.token,"expected_token":expected.token,"actual_position":b[i].cache.next_position,"expected_position":a[i].cache.next_position,"target_logits_max_abs":max_abs,"target_logits_rms":rms,"actual_logits":actual_scores,"expected_logits":expected_scores});
                            anyhow::bail!(
                                "paired full-target token mismatch at lane {i}, output {token_index}; saved target rows"
                            );
                        }
                        if b[i].pending_tokens.is_empty() {
                            anyhow::ensure!(a[i].cache.next_position == b[i].cache.next_position);
                            let (mut x, mut y) = (a[i].rng.clone(), b[i].rng.clone());
                            let broad = SamplingParams {
                                top_k: 0,
                                top_p: 1.0,
                                ..b[i].params.clone()
                            };
                            for _ in 0..32 {
                                anyhow::ensure!(
                                    a[i].sampling_history.sample_host(
                                        vec![0.2, 0.1, -0.1],
                                        &broad,
                                        &mut x
                                    )? == b[i].sampling_history.sample_host(
                                        vec![0.2, 0.1, -0.1],
                                        &broad,
                                        &mut y
                                    )?,
                                    "paired RNG/history mismatch"
                                );
                            }
                        }
                    }
                }
                anyhow::ensure!(
                    shared_passes > 0 && b.iter().all(|s| s.speculative_stats().passes > 0),
                    "no actual shared draft execution"
                );
                report["lanes"].as_array_mut().unwrap().push(serde_json::json!({"count":count,"shared_passes":shared_passes,"outputs":b.iter().map(|s| s.generated.clone()).collect::<Vec<_>>(),"draft_passes":b.iter().map(|s| s.speculative_stats().passes).collect::<Vec<_>>(),"accepted":b.iter().map(|s| s.speculative_stats().accepted_tokens).collect::<Vec<_>>(),"fell_back":b.iter().map(|s| s.speculative_stats().fell_back).collect::<Vec<_>>() }));
                for lane in &mut a {
                    lane.discard_cache(&mut stage);
                }
                for lane in &mut b {
                    lane.discard_cache(&mut stage);
                }
                stage.device().synchronize()?;
                anyhow::ensure!(
                    stage.total_kv_cache_tokens() == 0
                        && stage.total_kv_cache_allocation_tokens() == 0
                );
            }
            Ok(())
        })();
        report["passed"] = serde_json::json!(result.is_ok());
        report["error"] = serde_json::json!(result.as_ref().err().map(|e| format!("{e:#}")));
        std::fs::write(report_path, serde_json::to_vec_pretty(&report)?)?;
        result
    }
}
