//! Fixed-length text correctness checks using only dinf-engine and local GGUFs.
//!
//! cargo run -p dinf-engine --release --example qualify_text -- MODEL.gguf metal 16
//!
//! Usage: MODEL.gguf cpu|metal [MAX_TOKENS] [PREFILL_CHUNK]. No downloads or drafts.
//! Stop IDs are disabled; these checks use one full model on one device.
//! Passing covers this artifact, prompt and backend, not general model quality.
//! `--smoke` uses four greedy tokens and three target runs with one model load.
//! It checks chunked replay and cancellation; it does not replace numeric checks.
//! Use `MODEL.gguf cpu|metal --contracts` for short split-stage and tool checks.
//! `--tool-contract` runs only the tool check.
//! `--text-split-contract` checks only context-bound text splits.

#[path = "qualify_text/contracts.rs"]
mod contracts;

use std::path::Path;

use anyhow::{Context, Result, ensure};
use candle_core::Device;
use dinf_engine::continuous::GenerationSession;
use dinf_engine::sampling::SamplingParams;
use dinf_engine::tokenizer::GgufTokenizer;
use dinf_engine::{LayerRange, PipelineStage, read_gguf_metadata};
use serde_json::{Value, json};

const DEFAULT_PREFILL_CHUNK: usize = 8;
const SMOKE_TOKENS: usize = 4;

fn compare(label: &str, expected: &[u32], actual: &[u32]) -> Result<()> {
    let mismatch = expected.iter().zip(actual).position(|(a, b)| a != b);
    ensure!(
        !expected.is_empty() && expected.len() == actual.len() && mismatch.is_none(),
        "{label}: token mismatch at {:?}; expected {expected:?}, got {actual:?}",
        mismatch.or_else(
            || (expected.len() != actual.len()).then_some(actual.len().min(expected.len()))
        )
    );
    Ok(())
}

fn clean(stage: &PipelineStage) -> Result<()> {
    stage.device().synchronize()?;
    ensure!(
        stage.total_kv_cache_allocation_tokens() == 0,
        "cache allocations remain"
    );
    ensure!(stage.total_kv_cache_tokens() == 0, "cache tokens remain");
    Ok(())
}

fn finish(
    stage: &mut PipelineStage,
    session: &mut GenerationSession,
    prompt: usize,
    count: usize,
) -> Result<Vec<u32>> {
    ensure!(
        session.is_finished() && session.generated().len() == count,
        "short generation"
    );
    ensure!(
        session.next_position(stage) == prompt + count - 1,
        "wrong cache position"
    );
    let tokens = session.generated().to_vec();
    session.discard_cache(stage);
    clean(stage)?;
    Ok(tokens)
}

fn run(
    stage: &mut PipelineStage,
    prompt: &[u32],
    params: &SamplingParams,
    budget: usize,
) -> Result<Vec<u32>> {
    let mut session = GenerationSession::new(stage, prompt.to_vec(), params.clone(), vec![]);
    let mut partial_steps = 0;
    while !session.is_finished() {
        // Sampling rejects empty or non-finite logits on every emitted row.
        if session.step_with_prefill_budget(stage, budget)?.is_none() {
            ensure!(
                session.generated().is_empty(),
                "partial prefill emitted a token"
            );
            partial_steps += 1;
        }
    }
    ensure!(
        partial_steps == prompt.len().div_ceil(budget) - 1,
        "wrong prefill step count"
    );
    finish(stage, &mut session, prompt.len(), params.max_tokens)
}

fn interleaved_cancel(
    stage: &mut PipelineStage,
    prompt: &[u32],
    other: &[u32],
    params: &SamplingParams,
) -> Result<Vec<u32>> {
    let mut target = GenerationSession::new(stage, prompt.to_vec(), params.clone(), vec![]);
    let mut noise = Some(GenerationSession::new(
        stage,
        other.to_vec(),
        params.clone(),
        vec![],
    ));
    while !target.is_finished() {
        target.step(stage)?;
        if let Some(session) = noise.as_mut() {
            session.step(stage)?;
            if session.generated().len() == 2 {
                session.discard_cache(stage);
                noise = None;
            }
        }
    }
    ensure!(
        noise.is_none(),
        "interleaved request never reached cancellation"
    );
    finish(stage, &mut target, prompt.len(), params.max_tokens)
}

fn smoke_runs(stage: &mut PipelineStage, prompt: &[u32], other: &[u32]) -> Result<Vec<Value>> {
    let params = SamplingParams {
        temperature: 0.0,
        max_tokens: SMOKE_TOKENS,
        seed: Some(42),
        ..Default::default()
    };
    let baseline = run(stage, prompt, &params, usize::MAX).context("smoke baseline")?;
    compare(
        "smoke chunked prefill",
        &baseline,
        &run(stage, prompt, &params, DEFAULT_PREFILL_CHUNK)?,
    )?;
    let mut cancelled = GenerationSession::new(stage, prompt.to_vec(), params.clone(), vec![]);
    ensure!(
        cancelled
            .step_with_prefill_budget(stage, DEFAULT_PREFILL_CHUNK)?
            .is_none(),
        "smoke prefill cancellation emitted a token"
    );
    cancelled.discard_cache(stage);
    clean(stage)?;
    compare(
        "smoke replay after prefill cancellation with interleaved decode cancellation",
        &baseline,
        &interleaved_cancel(stage, prompt, other, &params)?,
    )?;
    Ok(vec![json!({
        "mode": "greedy",
        "tokens": baseline,
        "checks": ["chunked_prefill", "after_prefill_cancel", "interleaved_decode_cancel"]
    })])
}

fn qualify(
    path: &Path,
    device: &Device,
    backend: &str,
    max_tokens: usize,
    prefill_chunk: usize,
    smoke: bool,
) -> Result<Value> {
    ensure!(max_tokens >= 4, "MAX_TOKENS must be at least 4");
    ensure!(prefill_chunk > 0, "PREFILL_CHUNK must be positive");
    let metadata = read_gguf_metadata(path)?;
    let tokenizer = GgufTokenizer::from_gguf(path)?;
    let prompt = tokenizer.encode_chat(&[(
        "user".into(),
        "Write a Rust function to merge two sorted slices. Include tests.".into(),
    )])?;
    let other = tokenizer.encode_chat(&[(
        "user".into(),
        "Name three oceans and explain how they differ.".into(),
    )])?;
    ensure!(
        prompt.len() > prefill_chunk,
        "prompt is too short for chunk and cancellation checks"
    );
    ensure!(
        prompt
            .len()
            .max(other.len())
            .checked_add(max_tokens)
            .is_some_and(|n| n <= metadata.context_length),
        "test exceeds model context"
    );
    let mut stage = PipelineStage::from_gguf(
        path,
        LayerRange::try_solo(metadata.num_layers)?,
        device,
        None,
    )?;
    let mut runs = Vec::new();
    let modes: &[(&str, f64)] = if smoke {
        eprintln!("Checking basic smoke: {}", path.display());
        runs = smoke_runs(&mut stage, &prompt, &other)?;
        &[]
    } else {
        &[("greedy", 0.0), ("seeded_penalties", 0.7)]
    };
    for &(mode, temperature) in modes {
        eprintln!("Checking {mode}: {}", path.display());
        let params = SamplingParams {
            temperature,
            max_tokens,
            seed: Some(42),
            repetition_penalty: if temperature == 0.0 { 1.0 } else { 1.1 },
            frequency_penalty: if temperature == 0.0 { 0.0 } else { 0.1 },
            presence_penalty: if temperature == 0.0 { 0.0 } else { 0.1 },
            ..Default::default()
        };
        let baseline = run(&mut stage, &prompt, &params, usize::MAX).context("unsplit baseline")?;
        run(&mut stage, &other, &params, usize::MAX).context("other request")?;
        compare(
            "after other request",
            &baseline,
            &run(&mut stage, &prompt, &params, usize::MAX)?,
        )?;
        compare(
            "interleaved cancellation",
            &baseline,
            &interleaved_cancel(&mut stage, &prompt, &other, &params)?,
        )?;
        compare(
            "chunked prefill",
            &baseline,
            &run(&mut stage, &prompt, &params, prefill_chunk)?,
        )?;
        let mut cancelled = GenerationSession::new(&stage, prompt.clone(), params.clone(), vec![]);
        ensure!(
            cancelled
                .step_with_prefill_budget(&mut stage, prefill_chunk)?
                .is_none(),
            "prefill cancellation emitted a token"
        );
        cancelled.discard_cache(&mut stage);
        clean(&stage)?;
        compare(
            "after prefill cancellation",
            &baseline,
            &run(&mut stage, &prompt, &params, usize::MAX)?,
        )?;
        runs.push(json!({"mode": mode, "tokens": baseline, "checks": ["after_other_request", "interleaved_decode_cancel", "chunked_prefill", "after_prefill_cancel"]}));
    }
    drop(stage);
    device.synchronize()?;
    let mut report = json!({"passed": true, "path": path, "architecture": metadata.architecture, "backend": backend,
        "prompt_tokens": prompt.len(), "max_tokens": max_tokens, "prefill_chunk_tokens": prefill_chunk,
        "split_stages": 1, "draft": false, "runs": runs});
    if smoke {
        report["profile"] = json!("basic-v1");
        report["scope"] =
            json!("short greedy state checks; no independent numeric or answer-quality claim");
    }
    Ok(report)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .context("usage: qualify_text MODEL.gguf cpu|metal [MAX_TOKENS] [PREFILL_CHUNK]")?;
    let backend = args.next().context("missing backend: cpu or metal")?;
    let next = args.next();
    let contracts = matches!(
        next.as_deref(),
        Some("--contracts" | "--tool-contract" | "--text-split-contract")
    );
    let text_split_contract = next.as_deref() == Some("--text-split-contract");
    let split_contract = next.as_deref() == Some("--contracts");
    let smoke = next.as_deref() == Some("--smoke");
    ensure!(
        !(contracts || smoke) || args.len() == 0,
        "named profiles take no further arguments"
    );
    let max_tokens = next
        .filter(|_| !(contracts || smoke))
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(if smoke { SMOKE_TOKENS } else { 16 });
    let prefill_chunk = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(DEFAULT_PREFILL_CHUNK);
    ensure!(prefill_chunk > 0, "PREFILL_CHUNK must be positive");
    ensure!(args.next().is_none(), "too many arguments");
    let result = (|| {
        let device = match backend.as_str() {
            "cpu" => Device::Cpu,
            #[cfg(target_os = "macos")]
            "metal" => Device::new_metal(0)?,
            _ => anyhow::bail!("backend {backend:?} is unavailable in this build"),
        };
        if text_split_contract {
            return contracts::qualify_text_split(Path::new(&path), &device, &backend);
        }
        if contracts {
            return contracts::qualify(Path::new(&path), &device, &backend, split_contract);
        }
        qualify(
            Path::new(&path),
            &device,
            &backend,
            max_tokens,
            prefill_chunk,
            smoke,
        )
    })();
    match result {
        Ok(report) => println!("{report}"),
        Err(error) => {
            println!(
                "{}",
                json!({"passed": false, "path": path, "backend": backend, "prefill_chunk_tokens": prefill_chunk, "error": format!("{error:#}")})
            );
            return Err(error);
        }
    }
    Ok(())
}
