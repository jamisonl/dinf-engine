//! Full-weight score and throughput check without retained prompt caches.
use super::*;
use crate::gguf_loader::read_gguf_metadata;
use crate::sampling::{SamplingHistory, SamplingParams, SamplingRng};
use crate::tokenizer::GgufTokenizer;
use std::time::Instant;

struct Run {
    scores: Vec<Vec<Vec<u32>>>,
    tokens: Vec<Vec<u32>>,
    seconds: f64,
}

fn run(
    model: &Glm5NextStage,
    prompts: &[Vec<u32>],
    params: &SamplingParams,
    batch: bool,
) -> Result<Run> {
    let lanes = prompts.len();
    eprintln!(
        "GLM_BATCH_BEGIN lanes={lanes} batch={batch} outputs={}",
        params.max_tokens
    );
    let mut caches = (0..lanes)
        .map(|_| (0..model.blocks.len()).map(|_| None).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let mut positions = prompts.iter().map(Vec::len).collect::<Vec<_>>();
    let mut histories = prompts
        .iter()
        .map(|p| SamplingHistory::new(p, params))
        .collect::<Vec<_>>();
    let mut rngs = (0..lanes)
        .map(|lane| SamplingRng::seeded(42 + lane as u64 * 17))
        .collect::<Vec<_>>();
    let mut tokens = Vec::new();
    for lane in 0..lanes {
        let mut last = Vec::new();
        for (i, chunk) in prompts[lane].chunks(16).enumerate() {
            let StageOutput::Logits(logits) = model.forward_prefill(
                StageInput::Tokens(chunk.to_vec()),
                &mut caches[lane],
                chunk.len(),
                i * 16,
            )?
            else {
                anyhow::bail!("missing prefill logits");
            };
            last = logits.to_vec1::<f32>()?;
        }
        let token = histories[lane].sample_host(last, params, &mut rngs[lane])?;
        histories[lane].push(token);
        tokens.push(vec![token]);
        eprintln!(
            "GLM_BATCH_PREFILLED lane={lane} prompt_tokens={}",
            prompts[lane].len()
        );
    }
    model.device.synchronize()?;
    let mut packed = (0..model.blocks.len()).map(|_| None).collect::<Vec<_>>();
    let mut scores = Vec::new();
    let started = Instant::now();
    for _ in 0..params.max_tokens - 1 {
        let input = tokens
            .iter()
            .map(|ts| *ts.last().unwrap())
            .collect::<Vec<_>>();
        let outputs = if batch {
            model.forward_decode_batch(
                &input,
                &mut caches.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>(),
                &mut packed,
                &positions,
            )?
        } else {
            (0..lanes)
                .map(|lane| {
                    match model.forward(
                        StageInput::Tokens(vec![input[lane]]),
                        &mut caches[lane],
                        1,
                        positions[lane],
                    )? {
                        StageOutput::Logits(logits) => Ok(logits),
                        _ => anyhow::bail!("missing logits"),
                    }
                })
                .collect::<Result<Vec<_>>>()?
        };
        let mut step = Vec::new();
        for (lane, output) in outputs.iter().enumerate() {
            let values = output.to_vec1::<f32>()?;
            anyhow::ensure!(
                values.len() == model.vocab && values.iter().all(|v| v.is_finite()),
                "invalid scores"
            );
            step.push(values.iter().map(|v| v.to_bits()).collect());
            let token = histories[lane].sample_host(values, params, &mut rngs[lane])?;
            histories[lane].push(token);
            tokens[lane].push(token);
            positions[lane] += 1;
        }
        scores.push(step);
    }
    model.device.synchronize()?;
    let seconds = started.elapsed().as_secs_f64();
    assert!(packed.iter().all(Option::is_none));
    for lane in 0..lanes {
        for (slot, block) in caches[lane].iter().zip(&model.blocks) {
            model.check_decode_cache(slot, block, positions[lane])?;
        }
    }
    drop(caches);
    model.device.synchronize()?;
    Ok(Run {
        scores,
        tokens,
        seconds,
    })
}

#[test]
#[ignore = "loads full GLM weights; stop other models first"]
fn full_glm_decode_batch_scores_and_speed() -> Result<()> {
    let path = std::path::PathBuf::from(std::env::var("DINF_GLM5_FULL_GGUF")?);
    let md = read_gguf_metadata(&path)?;
    let tokenizer = GgufTokenizer::from_gguf(&path)?;
    let device = crate::test_metal_device().expect("full GLM batch benchmark requires Metal");
    let mut model =
        Glm5NextStage::from_streaming_gguf(&path, LayerRange::solo(md.num_layers), &device, None)?;
    model.decode_batch_enabled = true;
    let prompts = [
        "Write a Rust function to merge two sorted slices, preserving duplicates. Include tests for empty inputs and negative values, then explain its time and memory costs.",
        "A price falls by 20 percent and then rises by 25 percent. The final price is 120 dollars. Find the original price and verify each step.",
        "Explain how to plan a small vegetable garden. Discuss sunlight, water, and three common mistakes. Use plain English.",
        "Return JSON describing two edge cases for a CSV parser: quoted commas and empty final fields. Include an example and expected output for each.",
    ].iter().map(|text| tokenizer.encode_chat(&[("user".into(), (*text).into())])).collect::<Result<Vec<_>>>()?;
    let lanes = match std::env::var("DINF_GLM5_BENCH_LANES") {
        Ok(value) => value.parse::<usize>()?,
        Err(std::env::VarError::NotPresent) => 2,
        Err(error) => return Err(error.into()),
    };
    anyhow::ensure!((2..=4).contains(&lanes), "benchmark lanes must be 2..4");
    let prompts = &prompts[..lanes];
    let base = SamplingParams {
        max_tokens: 32,
        min_tokens: 32,
        seed: Some(42),
        repetition_penalty: 1.05,
        frequency_penalty: 0.1,
        presence_penalty: 0.1,
        ..Default::default()
    };
    let warm = SamplingParams {
        max_tokens: 3,
        min_tokens: 3,
        temperature: 0.,
        ..base.clone()
    };
    let a = run(&model, prompts, &warm, false)?;
    let b = run(&model, prompts, &warm, true)?;
    anyhow::ensure!(
        a.scores == b.scores && a.tokens == b.tokens,
        "warmup differs"
    );
    for temperature in [0., 0.7] {
        let params = SamplingParams {
            temperature,
            ..base.clone()
        };
        let mut expected: Option<Run> = None;
        let mut seconds = [0., 0.];
        for repeat in 0..2 {
            for batch in [repeat % 2, 1 - repeat % 2] {
                let result = run(&model, prompts, &params, batch == 1)?;
                if let Some(expected) = &expected {
                    anyhow::ensure!(
                        result.tokens == expected.tokens,
                        "token mismatch temperature={temperature} batch={batch}"
                    );
                    anyhow::ensure!(
                        result.scores == expected.scores,
                        "score bits differ temperature={temperature} batch={batch}"
                    );
                }
                seconds[batch] += result.seconds;
                eprintln!(
                    "GLM_BATCH_RUN lanes={lanes} temperature={temperature} repeat={repeat} batch={batch} total_tok_s={:.4} seconds={:.6} exact_scores=true",
                    ((params.max_tokens - 1) * lanes) as f64 / result.seconds,
                    result.seconds
                );
                if expected.is_none() {
                    expected = Some(result);
                }
            }
        }
        eprintln!(
            "GLM_BATCH_SUMMARY lanes={lanes} temperature={temperature} scalar_tok_s={:.4} batch_tok_s={:.4} speedup={:.4}",
            (2 * (params.max_tokens - 1) * lanes) as f64 / seconds[0],
            (2 * (params.max_tokens - 1) * lanes) as f64 / seconds[1],
            seconds[0] / seconds[1]
        );
    }
    Ok(())
}
