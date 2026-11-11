//! Release qualification for the opt-in 8-bit TurboQuant KV-cache profile.
//!
//! This is a selected regression gate, not a universal model-quality claim.
//! It compares an uncompressed F16 cache with the production 8-bit profile on
//! five fixed prompts using teacher-forced next-token histories, including
//! approximately 1K, 4K and 8K contexts. The prompts,
//! step count, and thresholds are deliberately explicit so a result is
//! reproducible and reviewable; they do not establish quality for every model,
//! prompt, context length, sampler, or backend.
//!
//! Usage:
//!
//! ```text
//! cargo run -p dinf-engine --release --example qualify_kv -- MODEL.gguf [min-speed-ratio]
//! ```
//!
//! The optional ratio requires quantized decode throughput to be at least that
//! fraction of the F16 baseline.  It is intentionally opt-in because backend
//! and hardware throughput vary substantially.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device};
use dinf_engine::architectures::{StageInput, StageOutput, TextRequestContext};
use dinf_engine::tokenizer::GgufTokenizer;
use dinf_engine::{LayerRange, PipelineStage, read_gguf_metadata};

const STEPS: usize = 64;
const MIN_ARGMAX_PERCENT: usize = 95;
const MAX_MEAN_KL: f64 = 0.02;
const MAX_PROMPT_KL: f64 = 0.10;
const LONG_PREFILL_CHUNK: usize = 128;

struct Prompt {
    label: &'static str,
    tokens: Vec<u32>,
}

struct ReferencePrompt {
    prompt: Prompt,
    rows: Vec<Vec<f32>>,
    next_tokens: Vec<u32>,
    decode_seconds: f64,
}

struct PromptResult {
    label: &'static str,
    prompt_tokens: usize,
    matched: usize,
    steps: usize,
    mean_kl: f64,
    max_kl: f64,
    decode_seconds: f64,
}

fn default_device() -> Result<Device> {
    #[cfg(target_os = "macos")]
    {
        Device::new_metal(0).context("Metal is required for this Mac qualification")
    }
    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    {
        Device::new_cuda(0).context("CUDA is required for this CUDA-enabled qualification")
    }
    #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
    {
        Ok(Device::Cpu)
    }
}

fn scores(output: StageOutput) -> Result<Vec<f32>> {
    let StageOutput::Logits(logits) = output else {
        bail!("qualification requires a solo stage that returns logits")
    };
    let flat = logits
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    ensure!(!flat.is_empty(), "model returned empty logits");
    ensure!(
        flat.iter().all(|value| value.is_finite()),
        "model returned non-finite logits"
    );
    Ok(flat)
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(left_idx, left), (right_idx, right)| {
            left.total_cmp(right).then_with(|| right_idx.cmp(left_idx))
        })
        .map(|(index, _)| index)
        .expect("scores() rejects empty logits")
}

fn logsumexp(values: &[f32]) -> f64 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    values
        .iter()
        .map(|value| (f64::from(*value) - max).exp())
        .sum::<f64>()
        .ln()
        + max
}

fn kl(reference: &[f32], actual: &[f32]) -> Result<f64> {
    ensure!(
        reference.len() == actual.len(),
        "baseline and quantized vocabulary sizes differ"
    );
    let reference_z = logsumexp(reference);
    let actual_z = logsumexp(actual);
    Ok(reference
        .iter()
        .zip(actual)
        .map(|(reference, actual)| {
            let reference_logp = f64::from(*reference) - reference_z;
            let actual_logp = f64::from(*actual) - actual_z;
            reference_logp.exp() * (reference_logp - actual_logp)
        })
        .sum())
}

fn make_prompts(tokenizer: &GgufTokenizer) -> Result<Vec<Prompt>> {
    let short = tokenizer.encode_chat(&[(
        "user".to_string(),
        "Write a Rust function to merge two sorted slices. Include tests.".to_string(),
    )])?;
    let explanation = tokenizer.encode_chat(&[(
        "user".to_string(),
        "Explain why the sky is blue and why sunsets are red.".to_string(),
    )])?;
    let mut prompts = vec![
        Prompt {
            label: "rust",
            tokens: short,
        },
        Prompt {
            label: "explanation",
            tokens: explanation,
        },
    ];
    for (label, repetitions) in [("context-1k", 48), ("context-4k", 192), ("context-8k", 384)] {
        let context = "A cache stores key and value vectors for each attention head. Each request owns independent positions and cached tokens. ".repeat(repetitions);
        prompts.push(Prompt {
            label,
            tokens: tokenizer.encode_chat(&[(
                "user".to_string(),
                format!("{context}\nSummarize this in three short points."),
            )])?,
        });
    }
    Ok(prompts)
}

fn prefill(stage: &mut PipelineStage, tokens: &[u32]) -> Result<Vec<f32>> {
    ensure!(!tokens.is_empty(), "qualification prompt is empty");
    stage.clear_cache();
    let context = TextRequestContext::new(tokens.len())?;
    stage.prepare_text_request(context);
    // Preserve the existing three-prompt gate's prefill path. Bound the new
    // longer contexts, including GPU completion between chunks, like serving.
    let chunk_size = if tokens.len() > 2048 {
        LONG_PREFILL_CHUNK
    } else {
        tokens.len()
    };
    let mut row = Vec::new();
    for (chunk, start) in tokens
        .chunks(chunk_size)
        .zip((0..tokens.len()).step_by(chunk_size))
    {
        row = scores(stage.prefill_text(
            StageInput::Tokens(chunk.to_vec()),
            chunk.len(),
            start,
            context,
        )?)?;
    }
    Ok(row)
}

fn required_matches(decisions: usize) -> Result<usize> {
    decisions
        .checked_mul(MIN_ARGMAX_PERCENT)
        .map(|percent| percent.div_ceil(100))
        .context("qualification decision count overflow")
}

fn collect_reference(
    stage: &mut PipelineStage,
    prompts: Vec<Prompt>,
    device: &Device,
) -> Result<Vec<ReferencePrompt>> {
    let mut references = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        eprintln!(
            "F16: {} ({} prompt tokens)",
            prompt.label,
            prompt.tokens.len()
        );
        let mut row = prefill(stage, &prompt.tokens)?;
        let mut rows = Vec::with_capacity(STEPS);
        let mut next_tokens = Vec::with_capacity(STEPS);
        let mut decode_seconds = 0.0;
        for index in 0..STEPS {
            let current = std::mem::take(&mut row);
            let next = argmax(&current) as u32;
            rows.push(current);
            next_tokens.push(next);
            if index + 1 < STEPS {
                let started = Instant::now();
                row = scores(stage.forward_text(
                    StageInput::Tokens(vec![next]),
                    1,
                    prompt.tokens.len() + index,
                    TextRequestContext::new(prompt.tokens.len())?,
                )?)?;
                decode_seconds += started.elapsed().as_secs_f64();
            }
        }
        device.synchronize()?;
        references.push(ReferencePrompt {
            prompt,
            rows,
            next_tokens,
            decode_seconds,
        });
    }
    Ok(references)
}

fn compare_quantized(
    stage: &mut PipelineStage,
    references: &[ReferencePrompt],
    device: &Device,
) -> Result<Vec<PromptResult>> {
    let mut results = Vec::with_capacity(references.len());
    for reference in references {
        eprintln!(
            "KV8: {} ({} prompt tokens)",
            reference.prompt.label,
            reference.prompt.tokens.len()
        );
        let mut row = prefill(stage, &reference.prompt.tokens)?;
        let mut matched = 0usize;
        let mut sum_kl = 0.0f64;
        let mut max_kl = 0.0f64;
        let mut decode_seconds = 0.0;
        for index in 0..STEPS {
            let selected = argmax(&row) as u32;
            matched += usize::from(selected == reference.next_tokens[index]);
            let divergence = kl(&reference.rows[index], &row)?;
            ensure!(divergence.is_finite(), "non-finite KL divergence");
            sum_kl += divergence;
            max_kl = max_kl.max(divergence);
            if index + 1 < STEPS {
                let started = Instant::now();
                row = scores(stage.forward_text(
                    StageInput::Tokens(vec![reference.next_tokens[index]]),
                    1,
                    reference.prompt.tokens.len() + index,
                    TextRequestContext::new(reference.prompt.tokens.len())?,
                )?)?;
                decode_seconds += started.elapsed().as_secs_f64();
            }
        }
        device.synchronize()?;
        results.push(PromptResult {
            label: reference.prompt.label,
            prompt_tokens: reference.prompt.tokens.len(),
            matched,
            steps: STEPS,
            mean_kl: sum_kl / STEPS as f64,
            max_kl,
            decode_seconds,
        });
    }
    Ok(results)
}

fn parse_args() -> Result<(PathBuf, Option<f64>)> {
    let mut args = std::env::args_os().skip(1);
    let Some(path) = args.next() else {
        bail!("usage: qualify_kv MODEL.gguf [min-speed-ratio]")
    };
    let min_speed_ratio = args
        .next()
        .map(|value| {
            value
                .to_str()
                .context("min-speed-ratio is not valid UTF-8")?
                .parse::<f64>()
                .context("min-speed-ratio must be a positive number")
        })
        .transpose()?;
    ensure!(
        args.next().is_none(),
        "usage: qualify_kv MODEL.gguf [min-speed-ratio]"
    );
    if let Some(ratio) = min_speed_ratio {
        ensure!(
            ratio.is_finite() && ratio > 0.0,
            "min-speed-ratio must be positive"
        );
    }
    Ok((PathBuf::from(path), min_speed_ratio))
}

fn main() {
    let mut errors = Vec::new();
    let result = (|| -> Result<()> {
        let (path, min_speed_ratio) = parse_args()?;
        let device = default_device()?;
        let metadata = read_gguf_metadata(Path::new(&path))?;
        let tokenizer = GgufTokenizer::from_gguf(Path::new(&path))?;
        let prompts = make_prompts(&tokenizer)?;
        ensure!(
            metadata.num_layers > 0,
            "model metadata contains no executable layers"
        );
        for prompt in &prompts {
            let required = prompt
                .tokens
                .len()
                .checked_add(STEPS - 1)
                .context("qualification context length overflow")?;
            ensure!(
                required <= metadata.context_length,
                "qualification prompt {} needs {required} positions but the model supports {}; long-context cases are not silently skipped",
                prompt.label,
                metadata.context_length
            );
        }

        let mut reference_stage = PipelineStage::from_gguf(
            Path::new(&path),
            LayerRange::solo(metadata.num_layers),
            &device,
            None,
        )?;
        reference_stage.warmup()?;
        let references = collect_reference(&mut reference_stage, prompts, &device)?;
        let reference_bytes = reference_stage.kv_bytes_per_token();
        let reference_budget = reference_stage.kv_budget_bytes_per_token();
        let reference_rates = references
            .iter()
            .map(|reference| (STEPS - 1) as f64 / reference.decode_seconds)
            .collect::<Vec<_>>();
        drop(reference_stage);
        device.synchronize()?;

        let mut quantized_stage = PipelineStage::from_gguf(
            Path::new(&path),
            LayerRange::solo(metadata.num_layers),
            &device,
            Some(8),
        )?;
        quantized_stage.warmup()?;
        let quantized_bytes = quantized_stage.kv_bytes_per_token();
        let quantized_budget = quantized_stage.kv_budget_bytes_per_token();
        let quantized_results = compare_quantized(&mut quantized_stage, &references, &device)?;
        drop(quantized_stage);

        let total_matched = quantized_results
            .iter()
            .map(|result| result.matched)
            .sum::<usize>();
        let total_steps = quantized_results
            .iter()
            .map(|result| result.steps)
            .sum::<usize>();
        let mean_kl = quantized_results
            .iter()
            .map(|result| result.mean_kl * result.steps as f64)
            .sum::<f64>()
            / total_steps as f64;
        let max_prompt_kl = quantized_results
            .iter()
            .map(|result| result.max_kl)
            .fold(0.0, f64::max);
        let quantized_rates = quantized_results
            .iter()
            .map(|result| (STEPS - 1) as f64 / result.decode_seconds)
            .collect::<Vec<_>>();
        let min_observed_speed_ratio = reference_rates
            .iter()
            .zip(&quantized_rates)
            .map(|(reference, quantized)| quantized / reference)
            .fold(f64::INFINITY, f64::min);

        println!("TurboQuant KV qualification (selected gate; not a universal model guarantee)");
        println!(
            "model={} layers={} device={device:?}",
            path.display(),
            metadata.num_layers
        );
        println!("storage_bytes_per_token={{f16:{reference_bytes}, q8:{quantized_bytes}}}");
        println!("admission_bytes_per_token={{f16:{reference_budget}, q8:{quantized_budget}}}");
        for ((result, reference_rate), quantized_rate) in quantized_results
            .iter()
            .zip(&reference_rates)
            .zip(&quantized_rates)
        {
            println!(
                "prompt={} context_tokens={} argmax={}/{} mean_kl={:.6} max_kl={:.6} f16_tok_s={:.2} q8_tok_s={:.2} speed_ratio={:.3}",
                result.label,
                result.prompt_tokens,
                result.matched,
                result.steps,
                result.mean_kl,
                result.max_kl,
                reference_rate,
                quantized_rate,
                quantized_rate / reference_rate,
            );
        }
        println!(
            "aggregate_argmax={total_matched}/{total_steps} mean_kl={mean_kl:.6} max_prompt_kl={max_prompt_kl:.6} min_speed_ratio={min_observed_speed_ratio:.3}"
        );

        let min_argmax_matches = required_matches(total_steps)?;
        if total_matched < min_argmax_matches {
            bail!(
                "aggregate argmax gate failed: {total_matched}/{total_steps} < {min_argmax_matches}/{total_steps}"
            );
        }
        if mean_kl > MAX_MEAN_KL {
            bail!("mean KL gate failed: {mean_kl:.6} > {MAX_MEAN_KL:.6}");
        }
        if max_prompt_kl > MAX_PROMPT_KL {
            bail!("per-prompt max KL gate failed: {max_prompt_kl:.6} > {MAX_PROMPT_KL:.6}");
        }
        if let Some(min_speed_ratio) = min_speed_ratio
            && min_observed_speed_ratio < min_speed_ratio
        {
            bail!(
                "optional speed gate failed: {min_observed_speed_ratio:.3} < {min_speed_ratio:.3}"
            );
        }
        Ok(())
    })();

    if let Err(error) = result {
        errors.push(format!("ERROR: {error:#}"));
    }
    if errors.is_empty() {
        println!("PASS: selected 8-bit KV quality gates passed");
    } else {
        for error in errors {
            eprintln!("{error}");
        }
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_gate_scales_without_weakening_its_percentage() -> Result<()> {
        assert_eq!(required_matches(3 * 64)?, 183);
        assert_eq!(required_matches(5 * 64)?, 304);
        assert_eq!(required_matches(1)?, 1);
        assert!(required_matches(usize::MAX).is_err());
        Ok(())
    }
}
