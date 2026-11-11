use super::{LayerCache, ModelArchitecture, StageInput, StageOutput};
use crate::gguf_loader::GgufMetadata;
use crate::sampling::{SamplingHistory, SamplingParams, SamplingRng};
use crate::tokenizer::GgufTokenizer;
use anyhow::{Context, Result};
use candle_core::Device;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub(super) enum Fusion {
    Sinkhorn,
    Post,
    Pre,
    ReuseProjection,
}

pub(super) trait HcBenchmark: ModelArchitecture {
    fn set_hc_fusion(&mut self, kind: Fusion, fused: bool);
    fn check_hc_cache(&self, cache: &[Option<LayerCache>], position: usize) -> Result<()>;
}

struct Observation {
    logits: Vec<Vec<u32>>,
    tokens: Vec<u32>,
    ttft: f64,
    decode_seconds: f64,
}

fn run<M: HcBenchmark>(
    model: &mut M,
    device: &Device,
    md: &GgufMetadata,
    prompt: &[u32],
    fused: bool,
    params: &SamplingParams,
    kind: Fusion,
) -> Result<Observation> {
    model.set_hc_fusion(kind, fused);
    let mut cache = (0..md.num_layers).map(|_| None).collect::<Vec<_>>();
    let mut history = SamplingHistory::new(prompt, params);
    let mut rng = SamplingRng::seeded(params.seed.context("benchmark requires a seed")?);
    let mut result = Observation {
        logits: Vec::new(),
        tokens: Vec::new(),
        ttft: 0.,
        decode_seconds: 0.,
    };
    let mut last = Vec::new();
    device.synchronize()?;
    let started = Instant::now();
    for (i, chunk) in prompt.chunks(32).enumerate() {
        let StageOutput::Logits(logits) = model.forward_prefill(
            StageInput::Tokens(chunk.to_vec()),
            &mut cache,
            chunk.len(),
            i * 32,
        )?
        else {
            anyhow::bail!("missing prefill logits");
        };
        last = logits.to_vec1::<f32>()?;
        result
            .logits
            .push(last.iter().map(|v| v.to_bits()).collect());
    }
    for index in 0..params.max_tokens {
        if index > 0 {
            let StageOutput::Logits(logits) = model.forward(
                StageInput::Tokens(vec![result.tokens[index - 1]]),
                &mut cache,
                1,
                prompt.len() + index - 1,
            )?
            else {
                anyhow::bail!("missing decode logits");
            };
            last = logits.to_vec1::<f32>()?;
            result
                .logits
                .push(last.iter().map(|v| v.to_bits()).collect());
        }
        anyhow::ensure!(
            last.len() == md.vocab_size && last.iter().all(|v| v.is_finite()),
            "invalid logits"
        );
        let token = history.sample_host(std::mem::take(&mut last), params, &mut rng)?;
        history.push(token);
        result.tokens.push(token);
        if index == 0 {
            result.ttft = started.elapsed().as_secs_f64();
        }
    }
    result.decode_seconds = started.elapsed().as_secs_f64() - result.ttft;
    model.check_hc_cache(&cache, prompt.len() + params.max_tokens - 1)?;
    drop(cache);
    device.synchronize()?;
    Ok(result)
}

pub(super) fn compare<M: HcBenchmark>(
    mut model: M,
    md: &GgufMetadata,
    tokenizer: &GgufTokenizer,
    device: &Device,
    chat: bool,
    kind: Fusion,
) -> Result<()> {
    anyhow::ensure!(
        std::env::var_os("DINF_PROFILE_LAYERS").is_none(),
        "disable synchronized profiling for throughput measurements"
    );
    let family = model.family().to_owned();
    let encode = |text: &str| {
        if chat {
            tokenizer.encode_chat(&[("user".into(), text.into())])
        } else {
            tokenizer.encode(text)
        }
    };
    let prompts = [
        (
            "code",
            "Write a Rust function that merges two sorted slices into a sorted vector. Preserve duplicates, use no unsafe code, and include unit tests for empty inputs and repeated values. Explain its time complexity.",
        ),
        (
            "math",
            "Solve this step by step: A store cuts a price by 20 percent, then raises the new price by 25 percent. The final price is 120 dollars. What was the original price? Verify your answer and explain why the two changes cancel here.",
        ),
        (
            "prose",
            "Explain how to plan a small vegetable garden for a beginner. Give a clear sequence of steps, discuss sunlight and watering, and note three common mistakes. Use plain English and keep the advice practical.",
        ),
    ];
    let base_params = SamplingParams {
        temperature: 0.,
        max_tokens: 32,
        min_tokens: 32,
        seed: Some(42),
        repetition_penalty: 1.05,
        frequency_penalty: 0.1,
        presence_penalty: 0.1,
        ..Default::default()
    };
    let warm = encode("Write a Rust function to add two numbers.")?;
    let warm_params = SamplingParams {
        max_tokens: 4,
        min_tokens: 4,
        ..base_params.clone()
    };
    eprintln!("HC_BENCH kind={kind:?}");
    let a = run(&mut model, device, md, &warm, false, &warm_params, kind)?;
    let b = run(&mut model, device, md, &warm, true, &warm_params, kind)?;
    anyhow::ensure!(
        a.logits == b.logits && a.tokens == b.tokens,
        "warmup differs"
    );
    for (name, text) in prompts {
        let prompt = encode(text)?;
        anyhow::ensure!(
            prompt.len() + base_params.max_tokens <= md.context_length,
            "context overflow"
        );
        for temperature in [0., 0.7] {
            let params = SamplingParams {
                temperature,
                ..base_params.clone()
            };
            let mut expected: Option<Observation> = None;
            let mut times = [Vec::new(), Vec::new()];
            let mut ttfts = [Vec::new(), Vec::new()];
            for repeat in 0..2 {
                for fused in [repeat % 2, 1 - repeat % 2] {
                    let result = run(&mut model, device, md, &prompt, fused == 1, &params, kind)?;
                    if let Some(expected) = &expected {
                        anyhow::ensure!(
                            expected.logits == result.logits,
                            "{name} temperature={temperature} fused={fused} logits differ"
                        );
                        anyhow::ensure!(expected.tokens == result.tokens, "{name} tokens differ");
                    }
                    times[fused].push(result.decode_seconds);
                    ttfts[fused].push(result.ttft);
                    eprintln!(
                        "HC_RUN model={family} name={name} temperature={temperature} repeat={repeat} fused={fused} prompt_tokens={} output_tokens={} ttft_s={:.4} decode_s={:.4} decode_tok_s={:.4} exact_logits=true",
                        prompt.len(),
                        result.tokens.len(),
                        result.ttft,
                        result.decode_seconds,
                        (result.tokens.len() - 1) as f64 / result.decode_seconds
                    );
                    if expected.is_none() {
                        expected = Some(result);
                    }
                }
            }
            let scalar: f64 = times[0].iter().sum();
            let fused: f64 = times[1].iter().sum();
            eprintln!(
                "HC_SUMMARY model={family} name={name} temperature={temperature} scalar_decode_s={:?} fused_decode_s={:?} scalar_ttft_s={:?} fused_ttft_s={:?} speedup={:.4} tokens={:?}",
                times[0],
                times[1],
                ttfts[0],
                ttfts[1],
                scalar / fused,
                expected.as_ref().unwrap().tokens
            );
        }
    }
    Ok(())
}
