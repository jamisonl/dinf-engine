//! Load one local model once and check every pinned corpus/reference case.
//!
//! qualify_matrix --prepare MODEL.gguf CORPUS.json PREPARED.json
//! qualify_matrix MODEL.gguf cpu|metal SUITE.json NEW_OUTPUT_DIRECTORY
//!
//! Prompts stay complete. All cases use 128 fixed natural teacher tokens,
//! identical across this checkpoint's weight formats and backends. Each case
//! retains raw F32LE engine logits, including numeric failures. The fixed gates
//! are >=95% argmax, mean KL <=0.02, max row KL <=0.1. NLL is reported separately.
//! The default extended-24 profile retains every case. The explicit smoke-6
//! profile runs numeric cases before state checks and may stop on a numeric
//! failure. Unrun work has no pass verdict. No weights download or draft attach.

#[path = "qualify_matrix/common.rs"]
mod common;
#[path = "qualify_matrix/coverage.rs"]
mod coverage;
#[path = "qualify_matrix/metrics.rs"]
mod metrics;
#[path = "qualify_matrix/prepare.rs"]
mod prepare;
#[path = "qualify_matrix/session_checks.rs"]
mod session_checks;

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device};
use common::{
    CheckedFile, ModelFile, Pin, json_file, sha256, valid_hash, valid_name, write_json_new,
};
use dinf_engine::architectures::{StageInput, StageOutput, TextRequestContext};
use dinf_engine::tokenizer::GgufTokenizer;
use dinf_engine::{LayerRange, PipelineStage, read_gguf_metadata};
use prepare::{Corpus, Prepared, PreparedCase, ROWS};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Suite {
    schema_version: u32,
    suite_id: String,
    backend: String,
    #[serde(default)]
    coverage_profile: coverage::Profile,
    #[serde(default)]
    fail_fast: bool,
    prepared: Pin,
    corpus: Pin,
    expected_binary_sha256: String,
    expected_token_identity_sha256: String,
    source_pins: Vec<Pin>,
    cases: Vec<Case>,
    internal_cases: Vec<String>,
    internal: session_checks::Options,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    reference: Pin,
    logits_sha256: String,
}

// The producer may add runtime evidence fields. Every field that controls
// model inputs, tokenization proof, raw bytes or numeric scope is required.
#[derive(Deserialize)]
struct Reference {
    schema_version: u32,
    source: String,
    backend: String,
    case_id: String,
    context_tokens: usize,
    micro_batch_tokens: usize,
    prepared_sha256: String,
    producer_source_sha256: String,
    oracle_raw_sha256: String,
    tokenization_verified: bool,
    model_sha256: String,
    model_files: Vec<ModelFile>,
    vocab_size: usize,
    logits_file: String,
    prompt_tokens: Vec<u32>,
    continuation_tokens: Vec<u32>,
    teacher_forcing_tokens: Vec<u32>,
}

struct ReadyCase {
    reference: Reference,
    json: CheckedFile,
    raw: CheckedFile,
}

fn validate_internal_cases(selected: &[String], cases: &[Case]) -> Result<()> {
    ensure!(
        !selected.is_empty(),
        "internal case selection must not be empty"
    );
    let available: HashSet<_> = cases.iter().map(|case| case.id.as_str()).collect();
    let mut seen = HashSet::new();
    for id in selected {
        ensure!(
            valid_name(id) && available.contains(id.as_str()) && seen.insert(id),
            "invalid, unknown or duplicate internal case: {id}"
        );
    }
    Ok(())
}

fn initial_internal_report(required: bool) -> Value {
    if required {
        json!({"required":true,"attempted":false,"status":"not_run",
            "valid":false,"passed":false,"modes":[]})
    } else {
        json!({"required":false,"attempted":false,"status":"not_selected",
            "valid":null,"passed":null,"modes":[]})
    }
}

fn compiled_sources() -> BTreeMap<String, String> {
    [
        (
            "examples/qualify_matrix.rs",
            include_bytes!("qualify_matrix.rs").as_slice(),
        ),
        (
            "examples/qualify_matrix/coverage.rs",
            include_bytes!("qualify_matrix/coverage.rs").as_slice(),
        ),
        (
            "examples/qualify_matrix/common.rs",
            include_bytes!("qualify_matrix/common.rs").as_slice(),
        ),
        (
            "examples/qualify_matrix/metrics.rs",
            include_bytes!("qualify_matrix/metrics.rs").as_slice(),
        ),
        (
            "examples/qualify_matrix/prepare.rs",
            include_bytes!("qualify_matrix/prepare.rs").as_slice(),
        ),
        (
            "examples/qualify_matrix/session_checks.rs",
            include_bytes!("qualify_matrix/session_checks.rs").as_slice(),
        ),
    ]
    .into_iter()
    .map(|(name, bytes)| (name.into(), format!("{:x}", Sha256::digest(bytes))))
    .collect()
}

fn dispatch_overrides(names: impl IntoIterator<Item = String>) -> Vec<String> {
    names
        .into_iter()
        .filter(|name| {
            ["DINF_", "CANDLE_", "GGML_", "GGUF_"]
                .iter()
                .any(|prefix| name.starts_with(prefix))
                || (name.starts_with("DYLD_") && !name.starts_with("DYLD_PRINT_"))
                || matches!(name.as_str(), "LD_PRELOAD" | "LD_LIBRARY_PATH")
        })
        .collect()
}

fn read_row(file: &mut impl Read, bytes: &mut [u8]) -> Result<Vec<f32>> {
    file.read_exact(bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn validate_reference(
    reference: &Reference,
    case: &PreparedCase,
    prepared: &Prepared,
    prepared_hash: &str,
    backend: &str,
    raw_hash: &str,
) -> Result<()> {
    ensure!(
        reference.schema_version == 1 && reference.source.starts_with("llama.cpp"),
        "unexpected oracle producer"
    );
    ensure!(
        reference.tokenization_verified,
        "oracle did not verify independent tokenization"
    );
    ensure!(
        valid_hash(&reference.producer_source_sha256),
        "invalid producer source hash"
    );
    ensure!(
        reference.backend == backend,
        "oracle and engine backends differ"
    );
    ensure!(
        reference.micro_batch_tokens == 8,
        "oracle must use the fixed 8-token micro-batch policy"
    );
    ensure!(
        reference.case_id == case.id && reference.context_tokens == case.context_tokens,
        "oracle case/context differs"
    );
    ensure!(
        reference.prepared_sha256 == prepared_hash,
        "oracle used another prepared fixture"
    );
    ensure!(
        reference.model_sha256 == prepared.model_sha256
            && reference.model_files == prepared.model_files,
        "oracle model identity differs"
    );
    ensure!(
        reference.vocab_size == prepared.vocab_size,
        "oracle vocabulary differs"
    );
    ensure!(
        reference.prompt_tokens == case.prompt_tokens
            && reference.teacher_forcing_tokens == case.teacher_forcing_tokens,
        "oracle prompt/teacher history differs"
    );
    ensure!(
        reference.continuation_tokens.len() == ROWS
            && reference
                .continuation_tokens
                .iter()
                .all(|&t| (t as usize) < prepared.vocab_size),
        "oracle must contain 128 valid argmax IDs"
    );
    ensure!(
        valid_name(&reference.logits_file)
            && reference.oracle_raw_sha256 == raw_hash
            && valid_hash(raw_hash),
        "oracle raw name/hash differs"
    );
    Ok(())
}

fn check_oracle_rows(raw: &Path, reference: &Reference) -> Result<()> {
    let row_bytes = reference
        .vocab_size
        .checked_mul(4)
        .context("row byte overflow")?;
    let expected = row_bytes.checked_mul(ROWS).context("raw byte overflow")?;
    ensure!(
        raw.metadata()?.len() == expected as u64,
        "oracle raw size differs"
    );
    let mut file = BufReader::new(File::open(raw)?);
    let mut bytes = vec![0; row_bytes];
    for (row, &expected_top) in reference.continuation_tokens.iter().enumerate() {
        let values = read_row(&mut file, &mut bytes)?;
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "oracle non-finite at row {row}"
        );
        let top = values
            .iter()
            .enumerate()
            .fold(0, |best, (i, x)| if *x > values[best] { i } else { best });
        ensure!(
            top == expected_top as usize,
            "oracle argmax/row differs at {row}"
        );
    }
    Ok(())
}

fn actual_values(output: StageOutput, vocab: usize) -> Result<Vec<f32>> {
    let StageOutput::Logits(tensor) = output else {
        anyhow::bail!("expected full-vocabulary logits")
    };
    let values = tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    ensure!(values.len() == vocab, "actual vocabulary dimension differs");
    Ok(values)
}

fn reference_case(
    stage: &mut PipelineStage,
    case: &PreparedCase,
    ready: &ReadyCase,
    chunk: usize,
    out: &Path,
    report: &mut Value,
) -> Result<()> {
    let reference = &ready.reference;
    let actual_path = out.join(format!("{}.actual.f32", case.id));
    let mut actual_file = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&actual_path)?,
    );
    report["actual_logits_file"] = json!(actual_path);
    let mut expected_file = BufReader::new(File::open(&ready.raw.path)?);
    let mut bytes = vec![0; reference.vocab_size * 4];
    let context = TextRequestContext::new(case.prompt_tokens.len())?;
    stage.prepare_text_request(context);
    let mut output = None;
    for (i, tokens) in case.prompt_tokens.chunks(chunk).enumerate() {
        output = Some(stage.prefill_text(
            StageInput::Tokens(tokens.to_vec()),
            tokens.len(),
            i * chunk,
            context,
        )?);
    }
    let mut output = output.context("empty prompt")?;
    let mut rows = Vec::new();
    for (row, &teacher) in case.teacher_forcing_tokens.iter().enumerate() {
        let actual = actual_values(output, reference.vocab_size)?;
        let raw: Vec<_> = actual
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        actual_file.write_all(&raw)?;
        actual_file.flush()?;
        let expected = read_row(&mut expected_file, &mut bytes)?;
        let metric = metrics::compare(row, &expected, &actual, teacher)?;
        ensure!(
            metric.reference_top == reference.continuation_tokens[row] as usize,
            "oracle changed after preflight"
        );
        rows.push(metric);
        report["completed_rows"] = json!(rows.len());
        // A later execution error keeps all completed row metrics as well as raw bytes.
        report["comparisons"] = json!(rows);
        if row + 1 == ROWS {
            break;
        }
        output = stage.forward_text(
            StageInput::Tokens(vec![teacher]),
            1,
            case.prompt_tokens.len() + row,
            context,
        )?;
    }
    actual_file.flush()?;
    actual_file.get_ref().sync_all()?;
    let matches = rows
        .iter()
        .filter(|r| r.actual_top == r.reference_top)
        .count();
    let mean_kl = rows.iter().map(|r| r.kl).sum::<f64>() / ROWS as f64;
    let max_kl = rows.iter().map(|r| r.kl).fold(0.0, f64::max);
    report["valid"] = json!(true);
    report["passed"] = json!(metrics::passed(matches, rows.len(), mean_kl, max_kl));
    report["matching_argmax"] = json!(matches);
    report["mean_kl"] = json!(mean_kl);
    report["max_kl"] = json!(max_kl);
    report["reference_mean_nll"] =
        json!(rows.iter().map(|r| r.reference_nll).sum::<f64>() / ROWS as f64);
    report["actual_mean_nll"] = json!(rows.iter().map(|r| r.actual_nll).sum::<f64>() / ROWS as f64);
    report["mean_nll_delta"] = json!(rows.iter().map(|r| r.nll_delta).sum::<f64>() / ROWS as f64);
    report["actual_logits_sha256"] = json!(sha256(&actual_path)?);
    report["actual_logits_bytes"] = json!(actual_path.metadata()?.len());
    report["prefill_chunk_tokens"] = json!(chunk);
    Ok(())
}

fn run(
    model: &Path,
    backend: &str,
    suite_path: &Path,
    out: &Path,
    report: &mut Value,
    state_only: bool,
) -> Result<()> {
    let overrides = dispatch_overrides(
        std::env::vars_os().map(|(name, _)| name.to_string_lossy().into_owned()),
    );
    ensure!(
        overrides.is_empty(),
        "qualification requires default dispatch; remove environment overrides: {overrides:?}"
    );
    let suite: Suite = json_file(suite_path)?;
    ensure!(
        !state_only || suite.coverage_profile == coverage::Profile::Smoke6ShortState,
        "state-only requires smoke-6-short-state"
    );
    ensure!(
        suite.schema_version == 1 && valid_name(&suite.suite_id),
        "invalid suite identity"
    );
    ensure!(
        suite.backend == backend && matches!(backend, "cpu" | "metal"),
        "suite backend differs"
    );
    suite.internal.validate()?;
    validate_internal_cases(&suite.internal_cases, &suite.cases)?;
    ensure!(
        !suite.source_pins.is_empty() && suite.source_pins.len() <= 4096,
        "source pins are required"
    );
    let base = suite_path.parent().unwrap_or(Path::new("."));
    let suite_hash = sha256(suite_path)?;
    let suite_pin = CheckedFile::check(
        Path::new("."),
        &Pin {
            path: suite_path.to_owned(),
            sha256: suite_hash,
        },
    )?;
    let binary = CheckedFile::check(
        Path::new("."),
        &Pin {
            path: std::env::current_exe()?,
            sha256: suite.expected_binary_sha256.clone(),
        },
    )?;
    let prepared_pin = CheckedFile::check(base, &suite.prepared)?;
    let corpus_pin = CheckedFile::check(base, &suite.corpus)?;
    let prepared: Prepared = json_file(&prepared_pin.path)?;
    let corpus: Corpus = json_file(&corpus_pin.path)?;
    prepare::validate_corpus(&corpus)?;
    ensure!(
        prepared.schema_version == 1
            && prepared.rows == ROWS
            && prepared.suite_id == suite.suite_id
            && corpus.suite_id == suite.suite_id,
        "prepared/corpus suite differs"
    );
    ensure!(
        prepared.corpus_sha256 == corpus_pin.sha256,
        "prepared corpus hash differs"
    );
    ensure!(
        valid_hash(&suite.expected_token_identity_sha256)
            && prepared.token_identity_sha256 == suite.expected_token_identity_sha256
            && prepare::token_identity(&prepared.cases)? == suite.expected_token_identity_sha256,
        "token identity differs from the shared checkpoint fixture"
    );
    ensure!(
        prepared.cases.len() == corpus.cases.len(),
        "prepared fixture must cover every corpus case once"
    );
    let selected_indices = coverage::selected_indices(
        suite.coverage_profile,
        suite.fail_fast,
        &suite
            .cases
            .iter()
            .map(|c| c.id.as_str())
            .collect::<Vec<_>>(),
        &prepared
            .cases
            .iter()
            .map(|c| c.id.as_str())
            .collect::<Vec<_>>(),
        &suite
            .internal_cases
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
    )?;
    let selected: Vec<_> = selected_indices
        .iter()
        .map(|&i| &prepared.cases[i])
        .collect();
    if suite.coverage_profile.is_smoke() {
        ensure!(
            suite.internal.decode_tokens == 8
                && suite.internal.prefill_chunk == 128
                && suite.internal.comparison_prefill_chunk == 32
                && suite.internal.cache_max_bytes == 4usize * 1024 * 1024 * 1024,
            "smoke profiles require unchanged state and prefill settings"
        );
    }
    ensure!(
        !prepared.model_files.is_empty(),
        "complete model manifest required"
    );
    let model = std::path::absolute(model)?;
    let parent = model.parent().context("model parent")?;
    let model_pins: Vec<_> = prepared
        .model_files
        .iter()
        .map(|f| {
            let path = parent.join(&f.name).canonicalize()?;
            Ok(CheckedFile {
                identity: common::stamp(&path)?,
                path,
                sha256: f.sha256.clone(),
            })
        })
        .collect::<Result<_>>()?;
    common::validate_model_files(
        &model,
        &prepared.model_sha256,
        Some(&prepared.model_files),
        true,
    )?;
    for pin in &model_pins {
        pin.recheck(false)?;
    }
    let metadata = read_gguf_metadata(&model)?;
    ensure!(
        metadata.architecture == prepared.architecture
            && metadata.vocab_size == prepared.vocab_size
            && metadata.context_length == prepared.model_context_length,
        "prepared model metadata differs"
    );
    let tokenizer = GgufTokenizer::from_gguf(&model)?;
    let sources: Vec<_> = suite
        .source_pins
        .iter()
        .map(|pin| CheckedFile::check(base, pin))
        .collect::<Result<_>>()?;
    for (name, hash) in compiled_sources() {
        ensure!(
            sources
                .iter()
                .any(|pin| pin.path.ends_with(&name) && pin.sha256 == hash),
            "source pins do not bind the compiled helper: {name}"
        );
    }
    // Validate every supplied text/token row, including unscored rows in a
    // reused 24-case fixture. Oracle JSON bytes keep their original fixture hash.
    for (fixture, text) in prepared.cases.iter().zip(&corpus.cases) {
        ensure!(fixture.id == text.id, "prepared/corpus case order differs");
        prepare::validate_case(fixture, metadata.vocab_size, metadata.context_length)?;
        ensure!(
            prepare::render_case(&tokenizer, text)? == *fixture,
            "prepared text/tokens differ from current tokenizer and exact corpus"
        );
    }
    if suite.coverage_profile.is_smoke() {
        for case in &selected {
            if suite
                .coverage_profile
                .state_ids()
                .contains(&case.id.as_str())
            {
                coverage::smoke_noise(&prepared.cases, case)?;
            }
        }
    }
    let mut ready = Vec::new();
    for (case, fixture) in suite.cases.iter().zip(&selected) {
        let json = CheckedFile::check(base, &case.reference)?;
        let reference: Reference = json_file(&json.path)?;
        validate_reference(
            &reference,
            fixture,
            &prepared,
            &prepared_pin.sha256,
            backend,
            &case.logits_sha256,
        )?;
        ensure!(
            sources
                .iter()
                .any(|pin| pin.sha256 == reference.producer_source_sha256),
            "producer source hash is absent from source pins"
        );
        let raw = CheckedFile::check(
            json.path.parent().unwrap(),
            &Pin {
                path: reference.logits_file.clone().into(),
                sha256: case.logits_sha256.clone(),
            },
        )?;
        check_oracle_rows(&raw.path, &reference)?;
        ready.push(ReadyCase {
            reference,
            json,
            raw,
        });
    }
    report["suite_id"] = json!(suite.suite_id);
    report["coverage_profile"] = json!(suite.coverage_profile.name());
    report["fail_fast"] = json!(suite.fail_fast);
    report["execution_policy"] = json!(if state_only {
        "state_only"
    } else if suite.coverage_profile.is_smoke() {
        "numeric_then_state"
    } else {
        "case_then_state"
    });
    if state_only {
        report["scope"] = json!(
            "state-only diagnostic; no numeric forwards or numeric acceptance; any combined result must retain separate numeric and state origins"
        );
    }
    report["selected_case_ids"] = json!(suite.cases.iter().map(|c| &c.id).collect::<Vec<_>>());
    report["prepared_case_count"] = json!(prepared.cases.len());
    report["backend"] = json!(backend);
    report["architecture"] = json!(metadata.architecture);
    report["model_files"] = json!(prepared.model_files);
    report["token_identity_sha256"] = json!(prepared.token_identity_sha256);
    report["input_pins"] = json!({"suite":suite_pin,"binary":binary,"prepared":prepared_pin,"corpus":corpus_pin,"sources":sources,"model_files":model_pins,
        "references":ready.iter().map(|c| json!({"json":c.json,"raw":c.raw})).collect::<Vec<_>>()});
    report["compiled_helper_sources"] = json!(compiled_sources());
    report["oracle_micro_batch_tokens"] = json!(8);
    report["limits"] = json!({"rows":ROWS,"min_argmax_fraction":0.95,"mean_kl":0.02,"max_kl":0.1,"nll_gate":null,"kl_direction":"reference || actual"});
    report["internal_options"] = json!(suite.internal);
    report["internal_case_ids"] = json!(suite.internal_cases);
    report["runtime_dispatch_overrides"] = json!(overrides);
    let device = match backend {
        "cpu" => Device::Cpu,
        #[cfg(target_os = "macos")]
        "metal" => Device::new_metal(0)?,
        _ => anyhow::bail!("backend unavailable in this build"),
    };
    let mut stage = PipelineStage::from_gguf(
        &model,
        LayerRange::try_solo(metadata.num_layers)?,
        &device,
        None,
    )?;
    report["model_loads"] = json!(1);
    let mut results = Vec::new();
    let mut blocked = None;
    if state_only {
        results = coverage::run_state_only(&mut stage, &suite, &prepared, &selected, out)?;
    } else if suite.coverage_profile.is_smoke() {
        results = coverage::run_smoke(
            &mut stage, &suite, &prepared, &selected, &ready, out, report,
        )?;
    } else {
        for (case, oracle) in selected.iter().zip(&ready) {
            let internal_required = suite.internal_cases.contains(&case.id);
            let internal = initial_internal_report(internal_required);
            if let Some(error) = &blocked {
                results.push(json!({"id":case.id,"attempted":false,"valid":false,"passed":false,"status":"not_run_after_cleanup_error","error":error,"internal":internal}));
                continue;
            }
            let started = Instant::now();
            let mut result = json!({"id":case.id,"domain":case.domain,"context_tokens":case.context_tokens,
            "prompt_tokens":case.prompt_tokens.len(),"rows":ROWS,"attempted":true,"valid":false,"passed":false,
            "reference":{"valid":false,"passed":false,"completed_rows":0},"internal":internal});
            eprintln!(
                "Checking {}: {} prompt tokens, context {}",
                case.id,
                case.prompt_tokens.len(),
                case.context_tokens
            );
            stage.set_max_kv_tokens(case.context_tokens);
            if let Err(error) = reference_case(
                &mut stage,
                case,
                oracle,
                suite.internal.prefill_chunk,
                out,
                &mut result["reference"],
            ) {
                result["reference"]["error"] = json!(format!("{error:#}"));
            }
            let cleanup = session_checks::clean(&mut stage);
            if let Err(error) = cleanup {
                blocked = Some(format!("{error:#}"));
                result["cleanup_error"] = json!(blocked);
            } else if internal_required {
                result["internal"]["attempted"] = json!(true);
                let other = prepared
                    .cases
                    .iter()
                    .filter(|other| {
                        other.prompt_tokens != case.prompt_tokens
                            && other.prompt_tokens.len() + suite.internal.decode_tokens
                                <= case.context_tokens
                    })
                    .min_by_key(|other| other.prompt_tokens.len());
                if let Some(other) = other {
                    if let Err(error) = session_checks::qualify(
                        &mut stage,
                        &case.prompt_tokens,
                        &other.prompt_tokens,
                        &suite.internal,
                        &mut result["internal"],
                    ) {
                        result["internal"]["error"] = json!(format!("{error:#}"));
                    }
                } else {
                    result["internal"]["error"] =
                        json!("no distinct complete noise prompt fits this context");
                }
                result["internal"]["status"] = json!(if result["internal"]["valid"] == true {
                    "complete"
                } else {
                    "error"
                });
                if let Err(error) = session_checks::clean(&mut stage) {
                    blocked = Some(format!("{error:#}"));
                    result["cleanup_error"] = json!(blocked);
                }
            }
            result["valid"] = json!(
                blocked.is_none()
                    && result["reference"]["valid"] == true
                    && (!internal_required || result["internal"]["valid"] == true)
            );
            result["passed"] = json!(
                result["valid"] == true
                    && result["reference"]["passed"] == true
                    && (!internal_required || result["internal"]["passed"] == true)
            );
            result["seconds"] = json!(started.elapsed().as_secs_f64());
            // Hash even partial raw output from an execution failure.
            let raw_path = out.join(format!("{}.actual.f32", case.id));
            if raw_path.exists() {
                result["reference"]["actual_logits_sha256"] = json!(sha256(&raw_path)?);
                result["reference"]["actual_logits_bytes"] = json!(raw_path.metadata()?.len());
            }
            write_json_new(&out.join(format!("{}.result.json", case.id)), &result)?;
            results.push(result);
            report["cases"] = json!(results);
        }
    }
    drop(stage);
    device.synchronize()?;
    report["cases"] = json!(results);
    for pin in [&suite_pin, &binary, &prepared_pin, &corpus_pin]
        .into_iter()
        .chain(&sources)
    {
        pin.recheck(true)?;
    }
    for pin in &model_pins {
        pin.recheck(false)?;
    }
    for case in &ready {
        case.json.recheck(true)?;
        case.raw.recheck(true)?;
    }
    report["post_run_pins_verified"] = json!(true);
    if state_only {
        coverage::summarize_state_only(&results, report);
    } else if suite.coverage_profile.is_smoke() {
        coverage::summarize(suite.coverage_profile, &results, suite.fail_fast, report);
    } else {
        report["valid"] = json!(
            results.len() == prepared.cases.len() && results.iter().all(|r| r["valid"] == true)
        );
        report["passed"] =
            json!(report["valid"] == true && results.iter().all(|r| r["passed"] == true));
        report["counts"] = json!({"planned":prepared.cases.len(),"attempted":results.iter().filter(|r| r["attempted"] == true).count(),
        "passed":results.iter().filter(|r| r["passed"] == true).count(),"failed":results.iter().filter(|r| r["passed"] != true).count()});
        report["internal_counts"] = json!({"planned":suite.internal_cases.len(),
        "attempted":results.iter().filter(|r| r["internal"]["attempted"] == true).count(),
        "passed":results.iter().filter(|r| r["internal"]["required"] == true && r["internal"]["passed"] == true).count(),
        "failed":results.iter().filter(|r| r["internal"]["required"] == true && r["internal"]["passed"] != true).count()});
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() == 4 && args[0] == "--prepare" {
        return prepare::prepare(
            Path::new(&args[1]),
            Path::new(&args[2]),
            Path::new(&args[3]),
        );
    }
    let state_only = args.first().is_some_and(|arg| arg == "--state-only");
    if state_only {
        args.remove(0);
    }
    ensure!(
        args.len() == 4,
        "usage: qualify_matrix --prepare MODEL CORPUS.json NEW_PREPARED.json; or [--state-only] MODEL cpu|metal SUITE.json NEW_OUT_DIR"
    );
    let out = PathBuf::from(&args[3]);
    std::fs::create_dir(&out).context("output directory must be new and its parent must exist")?;
    let mut report = json!({"schema_version":1,"valid":false,"passed":false,"cases":[],"scope":"same-backend fixed-history numeric and internal request-state checks; no task-quality claim"});
    let result = run(
        Path::new(&args[0]),
        &args[1],
        Path::new(&args[2]),
        &out,
        &mut report,
        state_only,
    );
    if let Err(error) = &result {
        report["valid"] = json!(false);
        report["passed"] = json!(false);
        report["error"] = json!(format!("{error:#}"));
    }
    let path = out.join("report.json");
    write_json_new(&path, &report)?;
    println!(
        "{}",
        json!({"valid":report["valid"],"passed":report["passed"],"report":path,"counts":report["counts"],"error":report["error"]})
    );
    result?;
    ensure!(
        report["passed"] == true,
        "matrix contains failed or incomplete cases; see {}",
        path.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_selection_requires_known_unique_nonempty_case_ids() -> Result<()> {
        let cases = ["conversation-512", "json-2048", "retrieval-4096"].map(|id| Case {
            id: id.into(),
            reference: Pin {
                path: "unused.json".into(),
                sha256: "0".repeat(64),
            },
            logits_sha256: "0".repeat(64),
        });
        validate_internal_cases(&["json-2048".into()], &cases)?;
        validate_internal_cases(
            &cases.iter().map(|c| c.id.clone()).collect::<Vec<_>>(),
            &cases,
        )?;
        for invalid in [
            vec![],
            vec!["missing".into()],
            vec!["../json-2048".into()],
            vec!["json-2048".into(), "json-2048".into()],
        ] {
            assert!(validate_internal_cases(&invalid, &cases).is_err());
        }
        Ok(())
    }

    #[test]
    fn unselected_internal_checks_have_no_pass_or_validity_verdict() {
        let report = initial_internal_report(false);
        assert_eq!(
            report,
            json!({"required":false,"attempted":false,"status":"not_selected",
            "valid":null,"passed":null,"modes":[]})
        );
        let required = initial_internal_report(true);
        assert_eq!(required["required"], true);
        assert_eq!(required["attempted"], false);
        assert_eq!(required["valid"], false);
        assert_eq!(required["passed"], false);
    }

    #[test]
    fn raw_reader_rejects_short_rows_and_preserves_float_bits() -> Result<()> {
        let mut bytes = [0; 8];
        let original = [1.0f32, -0.0f32];
        let raw: Vec<_> = original.iter().flat_map(|x| x.to_le_bytes()).collect();
        let values = read_row(&mut raw.as_slice(), &mut bytes)?;
        assert_eq!(values[0].to_bits(), original[0].to_bits());
        assert_eq!(values[1].to_bits(), original[1].to_bits());
        assert!(read_row(&mut &raw[..7], &mut bytes).is_err());
        Ok(())
    }

    #[test]
    fn dispatch_guard_allows_trace_but_rejects_hidden_math_overrides() {
        let names = [
            "PATH",
            "DYLD_PRINT_LIBRARIES",
            "DYLD_PRINT_RPATHS",
            "DINF_DISABLE_NATIVE",
            "CANDLE_METAL_COMPUTE_PIPELINE",
            "LD_PRELOAD",
        ];
        assert_eq!(
            dispatch_overrides(names.into_iter().map(str::to_owned)),
            [
                "DINF_DISABLE_NATIVE",
                "CANDLE_METAL_COMPUTE_PIPELINE",
                "LD_PRELOAD"
            ]
        );
    }

    #[test]
    fn dispatch_guard_rejects_library_search_and_injection_overrides() {
        let names = [
            "LD_LIBRARY_PATH",
            "DYLD_LIBRARY_PATH",
            "DYLD_FALLBACK_LIBRARY_PATH",
            "DYLD_INSERT_LIBRARIES",
            "DYLD_FRAMEWORK_PATH",
        ];
        assert_eq!(
            dispatch_overrides(names.into_iter().map(str::to_owned)),
            names
        );
    }

    #[test]
    fn output_receipts_never_overwrite_prior_evidence() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("report.json");
        write_json_new(&path, &json!({"passed":false}))?;
        let before = std::fs::read(&path)?;
        assert!(write_json_new(&path, &json!({"passed":true})).is_err());
        assert_eq!(std::fs::read(path)?, before);
        Ok(())
    }

    #[test]
    fn reference_binding_rejects_backend_history_and_fixture_drift() -> Result<()> {
        let hash = "0".repeat(64);
        let case = PreparedCase {
            id: "case".into(),
            domain: "code".into(),
            context_tokens: 512,
            rendered_prompt_utf8: "full prompt".into(),
            continuation_utf8: "text".into(),
            prompt_original_ids: vec![0, 1],
            prompt_tokens: vec![0, 1],
            continuation_original_ids: vec![2; 128],
            teacher_forcing_tokens: vec![2; 128],
            prompt_add_special: true,
            continuation_add_special: false,
            parse_special: true,
        };
        let model_files = vec![ModelFile {
            name: "model.gguf".into(),
            size_bytes: 32,
            sha256: hash.clone(),
        }];
        let prepared = Prepared {
            schema_version: 1,
            suite_id: "suite".into(),
            rows: 128,
            corpus_sha256: hash.clone(),
            model_sha256: hash.clone(),
            model_files: model_files.clone(),
            architecture: "llama".into(),
            vocab_size: 3,
            model_context_length: 4096,
            token_identity_sha256: prepare::token_identity(std::slice::from_ref(&case))?,
            preparation_binary_sha256: hash.clone(),
            preparation_source_sha256: BTreeMap::new(),
            cases: vec![case.clone()],
        };
        let value = json!({"schema_version":1,"source":"llama.cpp test","backend":"cpu","case_id":"case",
            "context_tokens":512,"micro_batch_tokens":8,"prepared_sha256":hash,"producer_source_sha256":hash,"oracle_raw_sha256":hash,
            "tokenization_verified":true,"model_sha256":hash,"model_files":model_files,"vocab_size":3,
            "logits_file":"case.f32","prompt_tokens":[0,1],"continuation_tokens":vec![0;128],"teacher_forcing_tokens":vec![2;128]});
        let reference: Reference = serde_json::from_value(value.clone())?;
        validate_reference(&reference, &case, &prepared, &hash, "cpu", &hash)?;
        for (key, changed) in [
            ("backend", json!("metal")),
            ("tokenization_verified", json!(false)),
            ("prepared_sha256", json!("1".repeat(64))),
            ("teacher_forcing_tokens", json!(vec![0; 128])),
            ("logits_file", json!("../other.f32")),
            ("context_tokens", json!(4096)),
            ("micro_batch_tokens", json!(128)),
            ("micro_batch_tokens", json!(0)),
        ] {
            let mut bad = value.clone();
            bad[key] = changed;
            assert!(
                validate_reference(
                    &serde_json::from_value(bad)?,
                    &case,
                    &prepared,
                    &hash,
                    "cpu",
                    &hash
                )
                .is_err(),
                "{key}"
            );
        }
        let mut legacy = value;
        legacy.as_object_mut().unwrap().remove("micro_batch_tokens");
        assert!(
            serde_json::from_value::<Reference>(legacy).is_err(),
            "a legacy oracle without an explicit micro-batch cannot qualify"
        );
        // Argmax IDs intentionally differ from fixed natural teacher IDs.
        assert_ne!(
            reference.continuation_tokens,
            reference.teacher_forcing_tokens
        );
        Ok(())
    }
}
