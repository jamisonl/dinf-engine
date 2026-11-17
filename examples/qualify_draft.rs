//! Full target + DFlash checks. Usage: TARGET.gguf DRAFT.gguf cpu|metal [TOKENS=64].
//! Admit both models before running. Exact requests must retain the scalar path.
//! An unseeded sampled request separately proves draft execution and cleanup.
//! Set DINF_ALLOW_APPROXIMATE_SPECULATIVE=1 only for the opt-in greedy diagnostic;
//! this example turns it into `SpeculativeOptions::allow_approximate`.

use std::path::Path;

use anyhow::{Result, ensure};
use candle_core::Device;
use dinf_engine::continuous::{GenerationSession, SpeculativeStats};
use dinf_engine::sampling::SamplingParams;
use dinf_engine::tokenizer::GgufTokenizer;
use dinf_engine::{
    EngineOptions, LayerRange, PipelineStage, SpeculativeOptions, read_gguf_metadata,
};
use serde_json::json;

fn clean(stage: &PipelineStage) -> Result<()> {
    stage.device().synchronize()?;
    ensure!(stage.total_kv_cache_tokens() == 0, "retained cache tokens");
    ensure!(
        stage.total_kv_cache_allocation_tokens() == 0,
        "retained cache allocations"
    );
    Ok(())
}

fn run(
    stage: &mut PipelineStage,
    prompt: &[u32],
    params: &SamplingParams,
    enabled: bool,
    budget: usize,
) -> Result<(Vec<u32>, SpeculativeStats)> {
    let mut session = GenerationSession::new(stage, prompt.to_vec(), params.clone(), vec![]);
    session.set_speculative_decoding_enabled(stage, enabled);
    while !session.is_finished() {
        session.step_with_prefill_budget(stage, budget)?;
    }
    let tokens = session.generated().to_vec();
    let stats = session.speculative_stats();
    ensure!(tokens.len() == params.max_tokens, "short generation");
    ensure!(
        session.next_position(stage) == prompt.len() + tokens.len() - 1,
        "wrong cache position"
    );
    ensure!(
        stats.accepted_tokens <= stats.drafted_tokens,
        "invalid draft counters"
    );
    ensure!(
        stats.committed_tokens <= tokens.len(),
        "too many committed tokens"
    );
    session.discard_cache(stage);
    clean(stage)?;
    Ok((tokens, stats))
}

fn cancel(stage: &mut PipelineStage, prompt: &[u32], params: &SamplingParams) -> Result<()> {
    let mut session = GenerationSession::new(stage, prompt.to_vec(), params.clone(), vec![]);
    session.set_speculative_decoding_enabled(stage, true);
    while session.generated().len() < 3 {
        session.step(stage)?;
    }
    ensure!(!session.is_finished(), "cancel request already finished");
    session.discard_cache(stage);
    clean(stage)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        (3..=4).contains(&args.len()),
        "usage: TARGET DRAFT cpu|metal [TOKENS=64]"
    );
    let count: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(64);
    ensure!(
        count >= 32,
        "use at least 32 tokens to exercise draft verification"
    );
    let device = match args[2].as_str() {
        "cpu" => Device::Cpu,
        #[cfg(target_os = "macos")]
        "metal" => Device::new_metal(0)?,
        _ => anyhow::bail!("backend unavailable"),
    };
    let model = Path::new(&args[0]);
    let metadata = read_gguf_metadata(model)?;
    let tokenizer = GgufTokenizer::from_gguf(model)?;
    let prompt = tokenizer.encode_chat(&[(
        "user".into(),
        "Write a Rust function to merge two sorted slices. Include tests.".into(),
    )])?;
    ensure!(
        prompt.len() + count <= metadata.context_length,
        "request exceeds context"
    );
    let mut stage = PipelineStage::from_gguf(
        model,
        LayerRange::try_solo(metadata.num_layers)?,
        &device,
        None,
    )?;
    stage.attach_speculative_draft(Path::new(&args[1]))?;
    ensure!(stage.speculative_decoding_enabled(), "draft did not attach");
    let approximate = std::env::var("DINF_ALLOW_APPROXIMATE_SPECULATIVE").as_deref() == Ok("1");
    stage.set_options(EngineOptions {
        speculative: SpeculativeOptions {
            allow_approximate: approximate,
            ..Default::default()
        },
        ..Default::default()
    })?;
    let greedy = SamplingParams {
        temperature: 0.0,
        max_tokens: count,
        ..Default::default()
    };
    let (baseline, _) = run(&mut stage, &prompt, &greedy, false, usize::MAX)?;
    let (enabled, greedy_stats) = run(&mut stage, &prompt, &greedy, true, usize::MAX)?;
    if approximate {
        ensure!(greedy_stats.passes > 0, "opt-in check never used the draft");
    } else {
        ensure!(
            greedy_stats.passes == 0 && greedy_stats.fell_back,
            "exact request did not use scalar fallback"
        );
    }
    ensure!(
        baseline == enabled,
        "enabled greedy tokens differ from scalar tokens"
    );
    cancel(&mut stage, &prompt, &greedy)?;
    let (replay, _) = run(&mut stage, &prompt, &greedy, true, usize::MAX)?;
    ensure!(baseline == replay, "cancel changed greedy replay");
    let (chunked, _) = run(&mut stage, &prompt, &greedy, true, 8)?;
    ensure!(baseline == chunked, "chunked prefill changed greedy tokens");
    if !approximate {
        let seeded = SamplingParams {
            temperature: 0.7,
            seed: Some(42),
            max_tokens: 16,
            ..Default::default()
        };
        let (scalar, _) = run(&mut stage, &prompt, &seeded, false, usize::MAX)?;
        let (enabled, stats) = run(&mut stage, &prompt, &seeded, true, usize::MAX)?;
        ensure!(
            scalar == enabled && stats.passes == 0 && stats.fell_back,
            "seeded request did not retain exact scalar fallback"
        );
    }
    let sampled = SamplingParams {
        temperature: 0.7,
        seed: None,
        max_tokens: count,
        repetition_penalty: 1.1,
        frequency_penalty: 0.1,
        presence_penalty: 0.1,
        ..Default::default()
    };
    let (_, sampled_stats) = run(&mut stage, &prompt, &sampled, true, usize::MAX)?;
    ensure!(
        sampled_stats.passes > 0 && sampled_stats.drafted_tokens > 0,
        "sampled check never used the draft"
    );
    cancel(&mut stage, &prompt, &sampled)?;
    let (after_sampled_cancel, _) = run(&mut stage, &prompt, &greedy, false, usize::MAX)?;
    ensure!(
        baseline == after_sampled_cancel,
        "draft cancellation changed scalar replay"
    );
    println!(
        "{}",
        json!({"passed":true,"target":args[0],"draft":args[1],"backend":args[2],
        "tokens":count,"approximate_greedy_opt_in":approximate,"greedy_tokens":baseline,
        "greedy_draft_passes":greedy_stats.passes,"sampled_draft_passes":sampled_stats.passes,
        "sampled_drafted_tokens":sampled_stats.drafted_tokens,"sampled_accepted_tokens":sampled_stats.accepted_tokens,
        "checks":["greedy_policy","greedy_replay","chunked_prefill","cancel_cleanup","sampled_draft_execution"],
        "independent_draft_quality_oracle":false})
    );
    Ok(())
}
