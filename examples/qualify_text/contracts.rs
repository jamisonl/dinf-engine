//! Short real-model checks. These do not replace independent numeric checks.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use dinf_engine::architectures::{StageInput, StageOutput, TextRequestContext};
use dinf_engine::continuous::GenerationSession;
use dinf_engine::sampling::SamplingParams;
use dinf_engine::tokenizer::{
    ChatTemplateMessage, ChatTemplateOptions, ChatTemplateTool, ChatTemplateToolChoice,
    GgufTokenizer,
};
use dinf_engine::tool_constraints::ToolConstraint;
use dinf_engine::{LayerRange, PipelineStage, read_gguf_metadata};
use serde_json::{Value, json};

const ROWS: usize = 8;

fn pipeline_rows(
    path: &Path,
    device: &Device,
    prompt: &[u32],
    layers: usize,
    count: usize,
    teacher: Option<&[u32]>,
) -> Result<(Vec<Vec<f32>>, Vec<u32>)> {
    ensure!(layers >= count && count > 0, "not enough layers to split");
    let mut stages = (0..count)
        .map(|index| {
            PipelineStage::from_gguf(
                path,
                LayerRange::try_new(index * layers / count, (index + 1) * layers / count, layers)?,
                device,
                None,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mut states = stages
        .iter()
        .map(PipelineStage::new_session_state)
        .collect::<Vec<_>>();
    let mut rows = Vec::new();
    let mut tokens = Vec::new();
    for row in 0..ROWS {
        let input_tokens = if row == 0 {
            prompt.to_vec()
        } else {
            vec![teacher.unwrap_or(&tokens)[row - 1]]
        };
        let position = if row == 0 { 0 } else { prompt.len() + row - 1 };
        let seq_len = input_tokens.len();
        let mut input = StageInput::Tokens(input_tokens);
        let mut logits = None;
        for (index, (stage, state)) in stages.iter_mut().zip(&mut states).enumerate() {
            let output = if row == 0 {
                stage.prefill_with_session(state, input, seq_len, position)?
            } else {
                stage.forward_with_session(state, input, seq_len, position)?
            };
            ensure!(
                stage.session_next_position(state) == position + seq_len,
                "stage {index} lost its cache position"
            );
            input = match output {
                StageOutput::Activations(tensor) => {
                    ensure!(index + 1 < count, "tail returned activations");
                    // Exercise the flat F32 inter-stage representation with fresh storage.
                    let dims = tensor.dims().to_vec();
                    let flat = tensor
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    ensure!(flat.iter().all(|v| v.is_finite()), "invalid activation");
                    StageInput::activations(Tensor::from_vec(flat, dims.as_slice(), device)?)
                }
                StageOutput::Logits(tensor) => {
                    ensure!(index + 1 == count, "non-tail returned logits");
                    logits = Some(
                        tensor
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                    );
                    StageInput::Tokens(Vec::new())
                }
            };
        }
        let logits = logits.context("missing tail logits")?;
        ensure!(
            !logits.is_empty() && logits.iter().all(|v| v.is_finite()),
            "invalid logits"
        );
        let token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .context("empty vocabulary")?
            .0;
        tokens.push(u32::try_from(token)?);
        rows.push(logits);
    }
    for (stage, state) in stages.iter_mut().zip(&mut states) {
        stage.clear_session_state(state);
        super::clean(stage)?;
    }
    drop(states);
    drop(stages);
    device.synchronize()?;
    Ok((rows, tokens))
}

fn tool_run(
    stage: &mut PipelineStage,
    tokenizer: &GgufTokenizer,
    prompt: &[u32],
    params: &SamplingParams,
    interleave: bool,
) -> Result<Vec<u32>> {
    let mut session = GenerationSession::new(
        stage,
        prompt.to_vec(),
        params.clone(),
        tokenizer.stop_token_ids().to_vec(),
    );
    let mut other = interleave.then(|| {
        GenerationSession::new(
            stage,
            prompt.to_vec(),
            params.clone(),
            tokenizer.stop_token_ids().to_vec(),
        )
    });
    while !session.is_finished() {
        session.step(stage)?;
        if let Some(noise) = other.as_mut() {
            noise.step(stage)?;
            if noise.generated().len() == 2 || noise.is_finished() {
                noise.discard_cache(stage);
                other = None;
            }
        }
        // The caller stops at a complete envelope. The grammar permits
        // trailing whitespace and does not promise that the model emits EOS.
        if tokenizer
            .decode(session.generated())
            .trim_end()
            .ends_with("</tool_call>")
        {
            break;
        }
    }
    ensure!(other.is_none(), "second request did not reach cancellation");
    let tokens = session.generated().to_vec();
    let visible = tokens
        .iter()
        .copied()
        .filter(|token| !tokenizer.is_stop_token(*token))
        .collect::<Vec<_>>();
    let text = tokenizer.decode(&visible);
    let body = text
        .trim()
        .strip_prefix("<tool_call>")
        .and_then(|body| body.strip_suffix("</tool_call>"))
        .with_context(|| format!("incomplete tool call: {text:?}"))?;
    let call: Value = serde_json::from_str(body)?;
    ensure!(
        call == json!({"name":"lookup", "arguments":{"n":7}}),
        "wrong tool call: {call}"
    );
    session.discard_cache(stage);
    super::clean(stage)?;
    Ok(tokens)
}

pub(super) fn qualify(
    path: &Path,
    device: &Device,
    backend: &str,
    split_contract: bool,
) -> Result<Value> {
    let metadata = read_gguf_metadata(path)?;
    let tokenizer = GgufTokenizer::from_gguf(path)?;
    let prompt = tokenizer.encode_chat(&[("user".into(), "Count from one to eight.".into())])?;
    ensure!(
        prompt.len() + ROWS <= metadata.context_length,
        "split test exceeds context"
    );
    let mut splits = Vec::new();
    if split_contract {
        let (expected, teacher) =
            pipeline_rows(path, device, &prompt, metadata.num_layers, 1, None)?;
        for count in [2, 3] {
            let (actual, tokens) = pipeline_rows(
                path,
                device,
                &prompt,
                metadata.num_layers,
                count,
                Some(&teacher),
            )?;
            ensure!(expected.len() == actual.len(), "split row count changed");
            for (row, (left, right)) in expected.iter().zip(&actual).enumerate() {
                ensure!(
                    left.len() == right.len()
                        && left
                            .iter()
                            .zip(right)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{count}-stage logits differ at row {row}"
                );
            }
            super::compare("split top tokens", &teacher, &tokens)?;
            splits.push(json!({"stages":count,"rows":ROWS,"logits_bit_exact":true,
            "stage_handoff":"fresh F32 host values","cache_positions_and_release":true}));
            eprintln!("{}", splits.last().context("missing split report")?);
        }
    }
    let messages = [ChatTemplateMessage::new(
        "user",
        "Call lookup with n equal to 999.",
    )];
    let options = ChatTemplateOptions::new([
        ChatTemplateTool::new("lookup").with_parameters_json(
            r#"{"type":"object","properties":{"n":{"type":"integer","const":7}},"required":["n"],"additionalProperties":false}"#,
        ),
    ])
    .with_tool_choice(ChatTemplateToolChoice::Named("lookup".into()))
    .with_parallel_tool_calls(false)
    .with_native_tool_formats(false);
    let tool_prompt = tokenizer.encode_chat_with_tool_options(&messages, &options)?;
    let params = SamplingParams {
        temperature: 0.0,
        max_tokens: 96,
        tool_constraint: ToolConstraint::for_chat(&tokenizer, &messages, &options)?,
        ..Default::default()
    };
    ensure!(params.tool_constraint.is_some(), "missing tool constraint");
    ensure!(
        tool_prompt.len() + params.max_tokens <= metadata.context_length,
        "tool test exceeds context"
    );
    let mut stage = PipelineStage::from_gguf(
        path,
        LayerRange::try_solo(metadata.num_layers)?,
        device,
        None,
    )?;
    let baseline = tool_run(&mut stage, &tokenizer, &tool_prompt, &params, false)?;
    let replay = tool_run(&mut stage, &tokenizer, &tool_prompt, &params, true)?;
    super::compare("tool request isolation", &baseline, &replay)?;
    drop(stage);
    device.synchronize()?;
    Ok(
        json!({"passed":true,"path":path,"backend":backend,"architecture":metadata.architecture,
        "scope":"one short prompt/history, scalar contiguous splits, JSON tool envelope; no network or split batching",
        "split_prompt_tokens":prompt.len(),"split_checks":splits,
        "tool":{"prompt_tokens":tool_prompt.len(),"generated_tokens":baseline.len(),
            "required_name_and_schema":true,"interleaved_cancel_replay_exact":true,
            "caller_stop_at_complete_envelope":true}}),
    )
}

fn pipeline_text_rows(
    path: &Path,
    device: &Device,
    prompt: &[u32],
    layers: usize,
    count: usize,
    teacher: Option<&[u32]>,
    context: TextRequestContext,
) -> Result<(Vec<Vec<f32>>, Vec<u32>)> {
    ensure!(layers >= count && count > 0, "not enough layers to split");
    ensure!(
        context.prompt_end() == prompt.len(),
        "wrong full prompt context"
    );
    let mut stages = (0..count)
        .map(|index| {
            PipelineStage::from_gguf(
                path,
                LayerRange::try_new(index * layers / count, (index + 1) * layers / count, layers)?,
                device,
                None,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mut states = stages
        .iter()
        .map(PipelineStage::new_session_state)
        .collect::<Vec<_>>();
    // Every stage receives the same complete prompt end before any cache use.
    // These are fresh sessions, so no prefix is reused even if prepare clears.
    for (stage, state) in stages.iter_mut().zip(&mut states) {
        stage.prepare_text_request_with_session(state, context)?;
        ensure!(
            stage.session_next_position(state) == 0,
            "fresh cache is not empty"
        );
    }
    let mut rows = Vec::new();
    let mut tokens = Vec::new();
    for row in 0..ROWS {
        let input_tokens = if row == 0 {
            prompt.to_vec()
        } else {
            vec![teacher.unwrap_or(&tokens)[row - 1]]
        };
        let position = if row == 0 { 0 } else { prompt.len() + row - 1 };
        let seq_len = input_tokens.len();
        let mut input = StageInput::Tokens(input_tokens);
        let mut logits = None;
        for (index, (stage, state)) in stages.iter_mut().zip(&mut states).enumerate() {
            let output = if row == 0 {
                stage.prefill_text_with_session(state, input, seq_len, position, context)?
            } else {
                stage.forward_text_with_session(state, input, seq_len, position, context)?
            };
            ensure!(
                stage.session_next_position(state) == position + seq_len,
                "stage {index} lost its cache position"
            );
            input = match output {
                StageOutput::Activations(tensor) => {
                    ensure!(index + 1 < count, "tail returned activations");
                    // Exercise the flat F32 inter-stage representation with fresh storage.
                    let dims = tensor.dims().to_vec();
                    let flat = tensor
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    ensure!(flat.iter().all(|v| v.is_finite()), "invalid activation");
                    StageInput::activations(Tensor::from_vec(flat, dims.as_slice(), device)?)
                }
                StageOutput::Logits(tensor) => {
                    ensure!(index + 1 == count, "non-tail returned logits");
                    logits = Some(
                        tensor
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                    );
                    StageInput::Tokens(Vec::new())
                }
            };
        }
        let logits = logits.context("missing tail logits")?;
        ensure!(
            !logits.is_empty() && logits.iter().all(|v| v.is_finite()),
            "invalid logits"
        );
        let token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .context("empty vocabulary")?
            .0;
        tokens.push(u32::try_from(token)?);
        rows.push(logits);
    }
    for (stage, state) in stages.iter_mut().zip(&mut states) {
        stage.clear_session_state(state);
        super::clean(stage)?;
    }
    drop(states);
    drop(stages);
    device.synchronize()?;
    Ok((rows, tokens))
}

/// Check only the request-context text APIs, without the tool-generation case.
pub(super) fn qualify_text_split(path: &Path, device: &Device, backend: &str) -> Result<Value> {
    let metadata = read_gguf_metadata(path)?;
    let tokenizer = GgufTokenizer::from_gguf(path)?;
    let prompt = tokenizer.encode_chat(&[("user".into(), "Count from one to eight.".into())])?;
    ensure!(
        prompt.len() + ROWS <= metadata.context_length,
        "split test exceeds context"
    );
    let context = TextRequestContext::new(prompt.len())?;
    let (expected, teacher) =
        pipeline_text_rows(path, device, &prompt, metadata.num_layers, 1, None, context)?;
    let mut splits = Vec::new();
    for count in [2, 3] {
        let (actual, tokens) = pipeline_text_rows(
            path,
            device,
            &prompt,
            metadata.num_layers,
            count,
            Some(&teacher),
            context,
        )?;
        ensure!(expected.len() == actual.len(), "split row count changed");
        for (row, (left, right)) in expected.iter().zip(&actual).enumerate() {
            ensure!(
                left.len() == right.len()
                    && left
                        .iter()
                        .zip(right)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{count}-stage text logits differ at row {row}"
            );
        }
        super::compare("text split top tokens", &teacher, &tokens)?;
        splits.push(json!({"stages":count,"rows":ROWS,"logits_bit_exact":true,
            "stage_handoff":"fresh F32 host values","cache_positions_and_release":true,
            "api":"TextRequestContext prepare/prefill_text/forward_text"}));
        eprintln!("{}", splits.last().context("missing split report")?);
    }
    Ok(json!({"passed":true,"path":path,"backend":backend,
        "architecture":metadata.architecture,
        "scope":"one short prompt/history, request-context scalar contiguous splits; no raw API, tool, network or split batching check",
        "prompt_end":context.prompt_end(),"split_prompt_tokens":prompt.len(),
        "split_checks":splits}))
}
