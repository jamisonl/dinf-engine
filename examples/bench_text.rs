//! Prefill and decode throughput for one GGUF model.
//!
//! cargo run --release --example bench_text -- MODEL.gguf cpu|metal [PROMPT_TOKENS] [DECODE_TOKENS] [REPEATS] [PREFILL_CHUNK]
//!
//! The prompt is a fixed, deterministic token sequence of the requested length.
//! Greedy decoding runs for exactly DECODE_TOKENS with stop IDs disabled. Each
//! repeat reports prefill time (prompt processing through the first sampled
//! token) and steady-state decode time for the remaining tokens.
//!
//! DINF_BENCH_PROMPT_FILE=FILE takes the prompt from that file's text, cycled
//! to PROMPT_TOKENS, instead of the repeated filler sentence. MoE routing
//! follows the text, so real text spreads over more experts than the filler.
//!
//! DINF_BENCH_DRAFT=DRAFT.gguf attaches a DFlash or DSpark draft and reports its pass
//! statistics; DINF_BENCH_DRAFT_MAX caps the drafted tokens per pass. Greedy
//! requests keep the exact scalar path unless
//! DINF_ALLOW_APPROXIMATE_SPECULATIVE=1 is also set, which this example turns
//! into `SpeculativeOptions::allow_approximate`.
//!
//! DINF_BENCH_MEM=1 prints the Metal allocation (`currentAllocatedSize`)
//! after load, warm-up, each prefill and each decode.
//!
//! DINF_PROFILE_LAYERS=1 turns on layer profiling and prints the Qwen3.5
//! layer profile for prefill and decode.
//!
//! DINF_BENCH_TOOLS=1 sends one `save_file` tool with the request and samples
//! under the engine's tool-call grammar (temperature 0.7). The
//! first repeat prints its output so the call syntax can be checked;
//! DINF_BENCH_TOOLS_PROMPT replaces the user request.

use std::time::Instant;

use anyhow::{Context, Result, ensure};
use candle_core::Device;
use dinf_engine::continuous::GenerationSession;
use dinf_engine::sampling::SamplingParams;
use dinf_engine::tokenizer::GgufTokenizer;
use dinf_engine::{
    EngineOptions, LayerRange, PipelineStage, SpeculativeOptions, read_gguf_metadata,
};

fn report_memory(device: &Device, phase: &str) {
    if std::env::var_os("DINF_BENCH_MEM").is_none() {
        return;
    }
    #[cfg(target_os = "macos")]
    if let Ok(metal) = device.as_metal_device() {
        let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
        // SAFETY: proc_pid_rusage fills the caller's rusage_info_v2.
        let footprint = unsafe {
            let mut info: libc::rusage_info_v4 = std::mem::zeroed();
            let ok = libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V4,
                (&mut info as *mut libc::rusage_info_v4).cast(),
            ) == 0;
            ok.then_some((
                info.ri_phys_footprint as usize,
                info.ri_lifetime_max_phys_footprint as usize,
            ))
        };
        let (footprint, max_footprint) = footprint.unwrap_or((0, 0));
        eprintln!(
            "mem {phase}: metal_allocated={:.0} MiB footprint={:.0} MiB max_footprint={:.0} MiB recommended_max={:.0} MiB",
            mib(metal.device().current_allocated_size()),
            mib(footprint),
            mib(max_footprint),
            mib(metal.device().recommended_max_working_set_size())
        );
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (device, phase);
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = std::path::PathBuf::from(args.next().context("missing MODEL.gguf")?);
    let backend = args.next().context("missing backend: cpu or metal")?;
    let parse = |value: Option<String>, default: usize| -> Result<usize> {
        value.map_or(Ok(default), |v| v.parse().context("invalid number"))
    };
    let prompt_tokens = parse(args.next(), 512)?;
    let decode_tokens = parse(args.next(), 128)?;
    let repeats = parse(args.next(), 3)?;
    // Optional per-step prefill budget; the default processes the prompt in one step.
    let prefill_chunk = parse(args.next(), usize::MAX)?;
    ensure!(prompt_tokens > 0 && decode_tokens > 1 && repeats > 0);

    let device = match backend.as_str() {
        "cpu" => Device::Cpu,
        "metal" => Device::new_metal(0)?,
        #[cfg(feature = "cuda")]
        "cuda" => Device::new_cuda(0)?,
        other => anyhow::bail!("unknown backend {other}"),
    };
    let metadata = read_gguf_metadata(&path)?;
    let tokenizer = GgufTokenizer::from_gguf(&path)?;
    let seed = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog while engineers measure \
         inference throughput on quantized transformer weights. ",
    )?;
    ensure!(!seed.is_empty(), "seed text produced no tokens");
    // DINF_BENCH_DRAFT=DRAFT.gguf attaches a DFlash draft. Repeated filler
    // would inflate acceptance, so draft runs use a real chat request padded
    // with filler only in front to reach PROMPT_TOKENS.
    let draft = std::env::var_os("DINF_BENCH_DRAFT").map(std::path::PathBuf::from);
    let tools = std::env::var_os("DINF_BENCH_TOOLS").is_some();
    let mut tool_constraint = None;
    let prompt: Vec<u32> = if tools {
        use dinf_engine::tokenizer::{ChatTemplateMessage, ChatTemplateOptions, ChatTemplateTool};
        let request = std::env::var("DINF_BENCH_TOOLS_PROMPT").unwrap_or_else(|_| {
            "Write a Rust module that merges two sorted slices, with unit tests covering empty \
             inputs and duplicates, and save it to src/merge.rs with the save_file tool."
                .into()
        });
        let messages = [ChatTemplateMessage::new("user", request)];
        let options = ChatTemplateOptions::new([ChatTemplateTool::new("save_file")
            .with_parameters_json(
                r#"{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"],"additionalProperties":false}"#,
            )]);
        tool_constraint = dinf_engine::tool_constraints::ToolConstraint::for_chat(
            &tokenizer, &messages, &options,
        )?;
        ensure!(
            tool_constraint.is_some(),
            "tool request produced no grammar"
        );
        let request = tokenizer.encode_chat_with_tool_options(&messages, &options)?;
        let pad = prompt_tokens.saturating_sub(request.len());
        seed.iter()
            .copied()
            .cycle()
            .take(pad)
            .chain(request)
            .collect()
    } else if draft.is_some() {
        let request = tokenizer.encode_chat(&[(
            "user".into(),
            "Write a Rust function to merge two sorted slices, then explain its complexity \
             and add unit tests covering empty inputs and duplicates."
                .into(),
        )])?;
        let pad = prompt_tokens.saturating_sub(request.len());
        seed.iter()
            .copied()
            .cycle()
            .take(pad)
            .chain(request)
            .collect()
    } else if let Some(file) = std::env::var_os("DINF_BENCH_PROMPT_FILE") {
        let text = std::fs::read_to_string(&file).context("reading DINF_BENCH_PROMPT_FILE")?;
        let tokens = tokenizer.encode(&text)?;
        ensure!(!tokens.is_empty(), "prompt file produced no tokens");
        tokens.iter().copied().cycle().take(prompt_tokens).collect()
    } else {
        seed.iter().copied().cycle().take(prompt_tokens).collect()
    };
    let prompt_tokens = prompt.len();
    // DINF_BENCH_PRINT_PROMPT=1 prints the prompt text, to give another
    // implementation the same input.
    if std::env::var_os("DINF_BENCH_PRINT_PROMPT").is_some() {
        eprintln!("--- prompt ---\n{}\n--- end ---", tokenizer.decode(&prompt));
    }

    let load = Instant::now();
    let mut stage = PipelineStage::from_gguf(
        &path,
        LayerRange::try_solo(metadata.num_layers)?,
        &device,
        None,
    )?;
    if let Some(draft) = &draft {
        stage.attach_speculative_draft(draft)?;
        ensure!(stage.speculative_decoding_enabled(), "draft did not attach");
        if let Some(max) = std::env::var_os("DINF_BENCH_DRAFT_MAX") {
            stage.set_speculative_draft_max_tokens(max.to_str().context("draft max")?.parse()?)?;
        }
        stage.set_options(EngineOptions {
            speculative: SpeculativeOptions {
                allow_approximate: std::env::var("DINF_ALLOW_APPROXIMATE_SPECULATIVE").as_deref()
                    == Ok("1"),
                // DINF_BENCH_DRAFT_FALLBACK=0 keeps drafting for the whole
                // request, to measure the draft alone.
                fallback: std::env::var("DINF_BENCH_DRAFT_FALLBACK").as_deref() != Ok("0"),
                ..Default::default()
            },
            ..Default::default()
        })?;
    }
    // DINF_CUDA_GRAPH_DECODE=0 issues CUDA decode launches one by one
    // instead of as captured graphs, which is the default.
    if std::env::var_os("DINF_CUDA_GRAPH_DECODE").is_some_and(|value| value == "0") {
        let mut options = stage.options().clone();
        options.cuda_graph_decode = false;
        stage.set_options(options)?;
    }
    device.synchronize()?;
    eprintln!("load {:.1}s", load.elapsed().as_secs_f64());
    report_memory(&device, "load");

    // DINF_BENCH_TEMPERATURE overrides the temperature either way.
    let temperature = match std::env::var("DINF_BENCH_TEMPERATURE") {
        Ok(value) => value.parse().context("invalid DINF_BENCH_TEMPERATURE")?,
        Err(_) if tools => 0.7,
        Err(_) => 0.0,
    };
    let params = SamplingParams {
        temperature,
        max_tokens: decode_tokens,
        tool_constraint,
        ..Default::default()
    };
    // Warm-up: compile pipelines and touch weights.
    {
        let mut warm = GenerationSession::new(
            &stage,
            prompt[..prompt.len().min(32)].to_vec(),
            SamplingParams {
                max_tokens: 4,
                ..params.clone()
            },
            vec![],
        );
        while !warm.is_finished() {
            warm.step(&mut stage)?;
        }
        warm.discard_cache(&mut stage);
    }
    device.synchronize()?;
    report_memory(&device, "warm-up");

    // DINF_PROFILE_REGIONS=1 times named regions of the production CUDA
    // decode path without changing which code runs.
    let regions = std::env::var_os("DINF_PROFILE_REGIONS").is_some();
    let profile = std::env::var_os("DINF_PROFILE_LAYERS").is_some();
    dinf_engine::architectures::profiling::set_layer_profiling(profile);
    let mut pp = Vec::new();
    let mut tg = Vec::new();
    for _ in 0..repeats {
        let mut session = GenerationSession::new(&stage, prompt.clone(), params.clone(), vec![]);
        session.set_speculative_decoding_enabled(&stage, draft.is_some());
        if profile {
            dinf_engine::architectures::qwen35::reset_layer_profile();
            dinf_engine::architectures::gemma4::reset_layer_profile();
        }
        let prefill_timing = std::env::var_os("DINF_BENCH_GPU_TIMING")
            .map(|_| dinf_engine::architectures::deepseek41::profile::GpuTimingGuard::enable());
        if regions {
            dinf_engine::architectures::profiling::reset_regions();
            dinf_engine::architectures::profiling::set_region_profiling(true);
        }
        let start = Instant::now();
        while session
            .step_with_prefill_budget(&mut stage, prefill_chunk)?
            .is_none()
        {}
        device.synchronize()?;
        let prefill = start.elapsed().as_secs_f64();
        report_memory(&device, "prefill");
        if regions {
            dinf_engine::architectures::profiling::set_region_profiling(false);
            eprintln!("prefill regions ({prompt_tokens} tokens):");
            dinf_engine::architectures::profiling::report_regions();
        }
        if prefill_timing.is_some() {
            let timing = dinf_engine::architectures::deepseek41::profile::gpu_timing_snapshot();
            eprintln!(
                "prefill gpu: command_buffers={} gpu_busy_ms={:.1} wall_ms={:.1}",
                timing.drained_buffers,
                timing.gpu_elapsed_ns as f64 / 1e6,
                prefill * 1e3
            );
        }
        drop(prefill_timing);
        if profile {
            eprintln!("prefill profile:");
            dinf_engine::architectures::qwen35::report_layer_profile();
            dinf_engine::architectures::gemma4::report_layer_profile();
            dinf_engine::architectures::qwen35::reset_layer_profile();
            dinf_engine::architectures::gemma4::reset_layer_profile();
        }
        let gpu_timing = std::env::var_os("DINF_BENCH_GPU_TIMING")
            .map(|_| dinf_engine::architectures::deepseek41::profile::GpuTimingGuard::enable());
        if regions {
            dinf_engine::architectures::profiling::reset_regions();
            dinf_engine::architectures::profiling::set_region_profiling(true);
        }
        let start = Instant::now();
        while !session.is_finished() {
            session.step(&mut stage)?;
        }
        device.synchronize()?;
        let decode = start.elapsed().as_secs_f64();
        report_memory(&device, "decode");
        if gpu_timing.is_some() {
            let timing = dinf_engine::architectures::deepseek41::profile::gpu_timing_snapshot();
            eprintln!(
                "decode gpu: command_buffers={} per_token={:.1} gpu_busy_ms_per_token={:.3} wall_ms_per_token={:.3}",
                timing.drained_buffers,
                timing.drained_buffers as f64 / (decode_tokens - 1) as f64,
                timing.gpu_elapsed_ns as f64 / 1e6 / (decode_tokens - 1) as f64,
                decode * 1e3 / (decode_tokens - 1) as f64
            );
        }
        drop(gpu_timing);
        if regions {
            dinf_engine::architectures::profiling::set_region_profiling(false);
            eprintln!("decode regions ({} tokens):", decode_tokens - 1);
            dinf_engine::architectures::profiling::report_regions();
        }
        if profile {
            eprintln!("decode profile:");
            dinf_engine::architectures::qwen35::report_layer_profile();
            dinf_engine::architectures::gemma4::report_layer_profile();
        }
        ensure!(
            session.generated().len() == decode_tokens,
            "short generation"
        );
        if draft.is_some() {
            let stats = session.speculative_stats();
            eprintln!(
                "draft passes={} drafted={} accepted={} ({:.1}%) tokens/pass={:.2} draft_ms={:.0} target_ms={:.0} commit_ms={:.0} fell_back={}",
                stats.passes,
                stats.drafted_tokens,
                stats.accepted_tokens,
                100.0 * stats.accepted_tokens as f64 / stats.drafted_tokens.max(1) as f64,
                stats.committed_tokens as f64 / stats.passes.max(1) as f64,
                stats.draft_time.as_secs_f64() * 1e3,
                stats.target_time.as_secs_f64() * 1e3,
                stats.commit_time.as_secs_f64() * 1e3,
                stats.fell_back
            );
        }
        // DINF_BENCH_PRINT_TOKENS=1 prints the first repeat's token IDs, to
        // compare greedy runs with and without a draft.
        if std::env::var_os("DINF_BENCH_PRINT_TOKENS").is_some() && pp.is_empty() {
            eprintln!("tokens {:?}", session.generated());
        }
        if tools && pp.is_empty() {
            eprintln!(
                "--- output ---\n{}\n--- end ---",
                tokenizer.decode(session.generated())
            );
        }
        session.discard_cache(&mut stage);
        pp.push(prompt_tokens as f64 / prefill);
        tg.push((decode_tokens - 1) as f64 / decode);
        eprintln!(
            "pp{prompt_tokens} {:.1} tok/s  tg{decode_tokens} {:.2} tok/s",
            pp.last().unwrap(),
            tg.last().unwrap()
        );
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    #[cfg(feature = "cuda")]
    {
        let stats = stage.cuda_graph_decode_stats();
        if stats.steps > 0 {
            let per_step = |ns: u64| ns as f64 / stats.steps as f64 / 1e6;
            eprintln!(
                "cuda graph decode: {} steps, {} segments, {} instantiations, {:.0} nodes per step; per step: record {:.3} ms, submit {:.3} ms",
                stats.steps,
                stats.segments,
                stats.instantiations,
                stats.nodes as f64 / stats.steps as f64,
                per_step(stats.record_ns),
                per_step(stats.submit_ns)
            );
        }
    }
    println!(
        "{{\"model\":{:?},\"backend\":{backend:?},\"pp_tokens\":{prompt_tokens},\"tg_tokens\":{decode_tokens},\"pp_tok_s\":{:.2},\"tg_tok_s\":{:.3}}}",
        path,
        mean(&pp),
        mean(&tg)
    );
    Ok(())
}
