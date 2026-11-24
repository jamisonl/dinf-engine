//! How closely model B's next-token distributions follow model A's, for
//! judging a requantized file against its source.
//!
//! cargo run --release --example compare_logits -- A.gguf B.gguf STEPS PROMPT...
//!
//! Each PROMPT is formatted as a user chat turn. A continues it greedily for
//! STEPS tokens; B is fed the same tokens. Prints the mean, 99th percentile
//! and largest per-token KL(A || B) over all rows, and how often both pick the
//! same top token. Both models stay loaded, on Metal on macOS and the CPU
//! elsewhere.
//!
//! A PROMPT of the form `@FILE` is that file's text. With A.gguf given twice
//! the model loads once, and `DINF_COMPARE_B_ENV=NAME=VALUE,...` sets those
//! variables for B's passes only: one kernel path against another on the same
//! weights, for the engine switches that are read on every call.
use anyhow::{Context, Result};
use candle_core::{DType, Device};
use dinf_engine::architectures::{StageInput, StageOutput, TextRequestContext};
use dinf_engine::tokenizer::GgufTokenizer;
use dinf_engine::{LayerRange, PipelineStage, read_gguf_metadata};

fn logits(output: &StageOutput) -> Result<Vec<f32>> {
    let StageOutput::Logits(tensor) = output else {
        anyhow::bail!("expected logits")
    };
    Ok(tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?)
}

fn log_softmax(values: &[f32]) -> Vec<f64> {
    let max = values.iter().fold(f32::MIN, |m, &v| m.max(v)) as f64;
    let sum: f64 = values.iter().map(|&v| ((v as f64) - max).exp()).sum();
    let log_sum = sum.ln() + max;
    values.iter().map(|&v| v as f64 - log_sum).collect()
}

fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b })
        .0 as u32
}

struct Run {
    stage: PipelineStage,
}

impl Run {
    /// Logits after the prompt and after each of `steps - 1` tokens, feeding
    /// `forced` tokens, or this model's own greedy picks where `forced` ends.
    fn rows(
        &mut self,
        prompt: &[u32],
        forced: &[u32],
        steps: usize,
    ) -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
        let context = TextRequestContext::new(prompt.len())?;
        let mut session = self.stage.new_session_state();
        self.stage
            .prepare_text_request_with_session(&mut session, context)?;
        let mut output = self.stage.prefill_text_with_session(
            &mut session,
            StageInput::Tokens(prompt.to_vec()),
            prompt.len(),
            0,
            context,
        )?;
        let mut rows = vec![logits(&output)?];
        let mut fed = Vec::new();
        for i in 0..steps - 1 {
            let token = forced
                .get(i)
                .copied()
                .unwrap_or_else(|| argmax(rows.last().unwrap()));
            fed.push(token);
            output = self.stage.forward_text_with_session(
                &mut session,
                StageInput::Tokens(vec![token]),
                1,
                prompt.len() + i,
                context,
            )?;
            rows.push(logits(&output)?);
        }
        Ok((fed, rows))
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let a_path = std::path::PathBuf::from(args.next().context("A")?);
    let b_path = std::path::PathBuf::from(args.next().context("B")?);
    let steps: usize = args.next().context("STEPS")?.parse()?;
    let prompts: Vec<String> = args.collect();
    #[cfg(target_os = "macos")]
    let device = Device::new_metal(0)?;
    #[cfg(not(target_os = "macos"))]
    let device = Device::Cpu;
    let tokenizer = GgufTokenizer::from_gguf(&a_path)?;
    let md = read_gguf_metadata(&a_path)?;
    let load = |path: &std::path::Path| -> Result<Run> {
        Ok(Run {
            stage: PipelineStage::from_gguf(
                path,
                LayerRange::try_solo(md.num_layers)?,
                &device,
                None,
            )?,
        })
    };
    let b_env: Vec<(String, String)> = std::env::var("DINF_COMPARE_B_ENV")
        .unwrap_or_default()
        .split(',')
        .filter_map(|pair| pair.split_once('='))
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .collect();
    let set_b_env = |on: bool| {
        for (name, value) in &b_env {
            // No other thread reads the environment while a pass runs.
            unsafe {
                if on {
                    std::env::set_var(name, value);
                } else {
                    std::env::remove_var(name);
                }
            }
        }
    };
    let mut a = load(&a_path)?;
    let mut b = if a_path == b_path {
        None
    } else {
        Some(load(&b_path)?)
    };
    let (mut kls, mut agree, mut total) = (Vec::new(), 0usize, 0usize);
    for prompt in &prompts {
        let prompt = match prompt.strip_prefix('@') {
            Some(file) => std::fs::read_to_string(file).context("reading prompt file")?,
            None => prompt.clone(),
        };
        let tokens = tokenizer.encode_chat(&[("user".into(), prompt)])?;
        // A's greedy continuation, then B on the same tokens.
        let (forced, ra) = a.rows(&tokens, &[], steps)?;
        set_b_env(true);
        let rb = b.as_mut().unwrap_or(&mut a).rows(&tokens, &forced, steps);
        set_b_env(false);
        let (_, rb) = rb?;
        for (la, lb) in ra.iter().zip(&rb) {
            let (pa, pb) = (log_softmax(la), log_softmax(lb));
            let kl: f64 = pa.iter().zip(&pb).map(|(x, y)| x.exp() * (x - y)).sum();
            kls.push(kl);
            agree += usize::from(argmax(la) == argmax(lb));
            total += 1;
        }
    }
    kls.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mean = kls.iter().sum::<f64>() / kls.len() as f64;
    println!(
        "rows {total}: mean KL {mean:.5}, p99 {:.4}, max {:.4}, top-1 agreement {agree}/{total} ({:.1}%)",
        kls[(kls.len() * 99 / 100).min(kls.len() - 1)],
        kls.last().unwrap(),
        agree as f64 * 100.0 / total as f64
    );
    Ok(())
}
