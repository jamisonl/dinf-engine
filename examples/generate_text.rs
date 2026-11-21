//! Greedy text from a raw prompt, for looking at a model's output by eye.
//!
//! cargo run --release --example generate_text -- MODEL.gguf cpu|metal PROMPT [MAX_TOKENS] [PREFILL_CHUNK]
//!
//! PROMPT is encoded as given, so it may spell the model's control tokens
//! (its chat format) directly; no chat template is applied. Decoding stops at
//! the tokenizer's stop IDs or after MAX_TOKENS (default 64). The output is
//! printed with control tokens, followed by prefill and decode timing.

use std::time::Instant;

use anyhow::{Context, Result, ensure};
use candle_core::Device;
use dinf_engine::continuous::GenerationSession;
use dinf_engine::sampling::SamplingParams;
use dinf_engine::tokenizer::GgufTokenizer;
use dinf_engine::{LayerRange, PipelineStage, read_gguf_metadata};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = std::path::PathBuf::from(args.next().context("missing MODEL.gguf")?);
    let backend = args.next().context("missing backend: cpu or metal")?;
    let prompt = args.next().context("missing PROMPT")?;
    let max_tokens = args
        .next()
        .map_or(Ok(64), |v| v.parse())
        .context("MAX_TOKENS")?;
    let prefill_chunk = args
        .next()
        .map_or(Ok(usize::MAX), |v| v.parse())
        .context("PREFILL_CHUNK")?;
    ensure!(max_tokens > 0 && prefill_chunk > 0);
    let device = match backend.as_str() {
        "cpu" => Device::Cpu,
        "metal" => Device::new_metal(0)?,
        #[cfg(feature = "cuda")]
        "cuda" => Device::new_cuda(0)?,
        other => anyhow::bail!("unknown backend {other}"),
    };

    let metadata = read_gguf_metadata(&path)?;
    let tokenizer = GgufTokenizer::from_gguf(&path)?;
    let tokens = tokenizer.encode(&prompt)?;
    ensure!(!tokens.is_empty(), "prompt produced no tokens");
    eprintln!(
        "prompt: {} tokens {:?}",
        tokens.len(),
        &tokens[..tokens.len().min(48)]
    );

    let load = Instant::now();
    let mut stage = PipelineStage::from_gguf(
        &path,
        LayerRange::try_solo(metadata.num_layers)?,
        &device,
        None,
    )?;
    device.synchronize()?;
    eprintln!("load {:.1}s", load.elapsed().as_secs_f64());

    let params = SamplingParams {
        temperature: 0.0,
        max_tokens,
        ..Default::default()
    };
    let mut session =
        GenerationSession::new(&stage, tokens.clone(), params, tokenizer.stop_ids.clone());
    let start = Instant::now();
    let mut first = None;
    while !session.is_finished() {
        let Some(step) = session.step_with_prefill_budget(&mut stage, prefill_chunk)? else {
            continue;
        };
        if first.is_none() {
            device.synchronize()?;
            first = Some(start.elapsed().as_secs_f64());
        }
        if step.finished {
            break;
        }
    }
    device.synchronize()?;
    let total = start.elapsed().as_secs_f64();
    let generated = session.generated().to_vec();
    session.discard_cache(&mut stage);

    println!("{}", tokenizer.decode(&generated));
    let prefill = first.unwrap_or(total);
    let decoded = generated.len().saturating_sub(1);
    eprintln!(
        "ids {generated:?}\nprefill {:.2}s ({:.1} tok/s), decode {decoded} tokens in {:.2}s ({:.1} tok/s)",
        prefill,
        tokens.len() as f64 / prefill.max(1e-9),
        total - prefill,
        decoded as f64 / (total - prefill).max(1e-9),
    );
    Ok(())
}
