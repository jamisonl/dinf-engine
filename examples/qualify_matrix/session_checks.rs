//! Checks request state independently from the teacher-forced oracle rows.
use anyhow::{Result, ensure};
use dinf_engine::PipelineStage;
use dinf_engine::continuous::GenerationSession;
use dinf_engine::sampling::SamplingParams;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::Duration;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Options {
    pub decode_tokens: usize,
    pub prefill_chunk: usize,
    pub comparison_prefill_chunk: usize,
    pub cache_max_bytes: usize,
}

impl Options {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (8..=16).contains(&self.decode_tokens),
            "internal decode count must be 8..16"
        );
        ensure!(
            (1..=4096).contains(&self.prefill_chunk)
                && (1..=4096).contains(&self.comparison_prefill_chunk),
            "invalid internal prefill chunk"
        );
        ensure!(
            self.prefill_chunk != self.comparison_prefill_chunk,
            "internal chunks must differ"
        );
        ensure!(
            self.cache_max_bytes > 0 && self.cache_max_bytes <= 8usize * 1024 * 1024 * 1024,
            "invalid prompt-cache byte bound"
        );
        Ok(())
    }
}

pub(super) fn clean(stage: &mut PipelineStage) -> Result<()> {
    stage.clear_cache();
    stage.clear_prompt_cache();
    stage.device().synchronize()?;
    ensure!(
        stage.total_kv_cache_tokens() == 0
            && stage.total_kv_cache_allocation_tokens() == 0
            && stage.prompt_cache_bytes() == 0,
        "cache allocations remain after cleanup"
    );
    Ok(())
}

fn finish(
    stage: &mut PipelineStage,
    session: &mut GenerationSession,
    prompt: usize,
    count: usize,
    publish: bool,
) -> Result<(Vec<u32>, usize)> {
    ensure!(
        session.is_finished() && session.generated().len() == count,
        "short internal generation"
    );
    ensure!(
        session.next_position(stage) == prompt + count - 1,
        "internal cache position differs"
    );
    let tokens = session.generated().to_vec();
    let reused = session.reused_prompt_tokens();
    if publish {
        session.publish_prompt_cache(stage);
    }
    session.discard_cache(stage);
    stage.device().synchronize()?;
    Ok((tokens, reused))
}

fn run(
    stage: &mut PipelineStage,
    prompt: &[u32],
    params: &SamplingParams,
    budget: usize,
    scope: Option<&str>,
    publish: bool,
) -> Result<(Vec<u32>, usize)> {
    let mut session = GenerationSession::new(stage, prompt.to_vec(), params.clone(), vec![]);
    if let Some(scope) = scope {
        session.enable_prompt_cache(scope.into());
    }
    let result = (|| {
        let mut steps = 0;
        while !session.is_finished() {
            let next = session.step_with_prefill_budget(stage, budget)?;
            ensure!(
                next.is_some() || session.generated().is_empty(),
                "partial prefill emitted a token"
            );
            steps += 1;
            ensure!(
                steps <= prompt.len() + params.max_tokens,
                "session failed to make progress"
            );
        }
        finish(
            stage,
            &mut session,
            prompt.len(),
            params.max_tokens,
            publish,
        )
    })();
    if result.is_err() {
        session.discard_cache(stage);
    }
    result
}

fn interleaved(
    stage: &mut PipelineStage,
    prompt: &[u32],
    other: &[u32],
    params: &SamplingParams,
    budget: usize,
) -> Result<Vec<u32>> {
    let mut target = GenerationSession::new(stage, prompt.to_vec(), params.clone(), vec![]);
    let mut noise = GenerationSession::new(stage, other.to_vec(), params.clone(), vec![]);
    let result = (|| {
        let mut cancelled = false;
        let mut ticks = 0;
        while !target.is_finished() {
            target.step_with_prefill_budget(stage, budget)?;
            if !cancelled {
                noise.step_with_prefill_budget(stage, budget)?;
                if noise.generated().len() == 2 {
                    noise.discard_cache(stage);
                    cancelled = true;
                }
            }
            ticks += 1;
            ensure!(
                ticks <= prompt.len() + params.max_tokens,
                "interleaved session stalled"
            );
        }
        ensure!(cancelled, "noise request never reached decode cancellation");
        Ok(finish(stage, &mut target, prompt.len(), params.max_tokens, false)?.0)
    })();
    target.discard_cache(stage);
    noise.discard_cache(stage);
    result
}

fn record(checks: &mut Vec<Value>, name: &str, expected: &[u32], actual: &[u32], extra: Value) {
    let first = expected
        .iter()
        .zip(actual)
        .position(|(a, b)| a != b)
        .or_else(|| (expected.len() != actual.len()).then_some(expected.len().min(actual.len())));
    checks.push(
        json!({"name":name,"passed":!expected.is_empty() && first.is_none(),
        "first_mismatch":first,"actual_tokens":actual,"details":extra}),
    );
}

pub(super) fn qualify(
    stage: &mut PipelineStage,
    prompt: &[u32],
    other: &[u32],
    options: &Options,
    report: &mut Value,
) -> Result<()> {
    ensure!(
        prompt.len() >= 2 && prompt != other,
        "internal checks require two distinct complete prompts"
    );
    options.validate()?;
    stage.set_prompt_cache_limits(options.cache_max_bytes, 2, Duration::from_secs(3600));
    let primary = options.prefill_chunk.min(prompt.len() - 1);
    let secondary = options.comparison_prefill_chunk.min(prompt.len() - 1);
    let mut modes = Vec::new();
    for (mode, temperature) in [("greedy", 0.0), ("seeded_penalties", 0.7)] {
        clean(stage)?;
        let params = SamplingParams {
            temperature,
            max_tokens: options.decode_tokens,
            seed: Some(42),
            repetition_penalty: if temperature == 0.0 { 1.0 } else { 1.1 },
            frequency_penalty: if temperature == 0.0 { 0.0 } else { 0.1 },
            presence_penalty: if temperature == 0.0 { 0.0 } else { 0.1 },
            ..Default::default()
        };
        let baseline = run(stage, prompt, &params, usize::MAX, None, false)?.0;
        let mut checks = Vec::new();
        run(stage, other, &params, options.prefill_chunk, None, false)?;
        let replay = run(stage, prompt, &params, primary, None, false)?.0;
        record(
            &mut checks,
            "replay_after_other_request_and_chunked_prefill",
            &baseline,
            &replay,
            json!({"prefill_chunk":primary}),
        );
        let mixed = interleaved(stage, prompt, other, &params, primary)?;
        record(
            &mut checks,
            "interleaved_decode_cancellation",
            &baseline,
            &mixed,
            json!({"cancel_after_tokens":2}),
        );
        let mut cancelled = GenerationSession::new(stage, prompt.to_vec(), params.clone(), vec![]);
        let step = cancelled.step_with_prefill_budget(stage, secondary);
        let cancelled_position = cancelled.next_position(stage);
        let cancelled_tokens = cancelled.kv_cache_tokens(stage);
        let cancelled_allocations = stage.total_kv_cache_allocation_tokens();
        cancelled.discard_cache(stage);
        stage.device().synchronize()?;
        ensure!(
            step?.is_none() && cancelled.generated().is_empty(),
            "prefill cancellation emitted a token"
        );
        ensure!(
            cancelled_position == secondary && cancelled_tokens > 0 && cancelled_allocations > 0,
            "prefill cancellation did not retain partial request state"
        );
        ensure!(
            cancelled.next_position(stage) == 0
                && cancelled.kv_cache_tokens(stage) == 0
                && stage.total_kv_cache_tokens() == 0
                && stage.total_kv_cache_allocation_tokens() == 0
                && stage.prompt_cache_bytes() == 0,
            "prefill cancellation retained request cache state"
        );
        // Replay directly: a separate cleanup must not hide a failed discard.
        let replay = run(stage, prompt, &params, secondary, None, false)?.0;
        record(
            &mut checks,
            "replay_after_partial_prefill_cancellation",
            &baseline,
            &replay,
            json!({"prefill_chunk":secondary}),
        );

        let scope = "matrix-owner";
        let (cold, reused) = run(stage, prompt, &params, primary, Some(scope), true)?;
        let retained = stage.prompt_cache_bytes();
        record(
            &mut checks,
            "prompt_cache_publish",
            &baseline,
            &cold,
            json!({"reused_prompt_tokens":reused,"retained_bytes":retained}),
        );
        checks.last_mut().unwrap()["passed"] =
            json!(checks.last().unwrap()["passed"] == true && reused == 0 && retained > 0);
        let (warm, reused) = run(stage, prompt, &params, primary, Some(scope), false)?;
        record(
            &mut checks,
            "prompt_cache_reuse",
            &baseline,
            &warm,
            json!({"reused_prompt_tokens":reused}),
        );
        checks.last_mut().unwrap()["passed"] =
            json!(checks.last().unwrap()["passed"] == true && reused == prompt.len());
        let (foreign, reused) = run(
            stage,
            prompt,
            &params,
            primary,
            Some("matrix-other-owner"),
            false,
        )?;
        record(
            &mut checks,
            "prompt_cache_owner_isolation",
            &baseline,
            &foreign,
            json!({"reused_prompt_tokens":reused}),
        );
        checks.last_mut().unwrap()["passed"] =
            json!(checks.last().unwrap()["passed"] == true && reused == 0);
        clean(stage)?;
        modes.push(json!({"mode":mode,"baseline_tokens":baseline,"checks":checks}));
        // Retain finished modes even if a later mode encounters a runtime error.
        report["modes"] = json!(modes);
    }
    report["valid"] = json!(true);
    report["passed"] = json!(modes.iter().all(|m| {
        m["checks"]
            .as_array()
            .unwrap()
            .iter()
            .all(|c| c["passed"] == true)
    }));
    Ok(())
}
