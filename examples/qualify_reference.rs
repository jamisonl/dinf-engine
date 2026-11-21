//! Compare model logits with a saved reference using the same token history.
//! See fixtures/llama_reference.cpp and fixtures/text-coverage/README.md for
//! the reference format.
//! Split references add an ordered `model_files` list of `{name, size_bytes,
//! sha256}` for every canonical five-digit GGUF shard. Every file is hashed
//! before loading weights; keep the files immutable throughout the check.
//! Optional `teacher_forcing_tokens` selects the history independently of each
//! row's recorded argmax. An optional final CLI path writes actual F32LE rows
//! with create_new and flushes each row, including when comparison later fails.
//! Optional `second_reference_logits_file` names logits for the same rows from
//! a second run of the reference implementation, for example its other
//! backend. Some models are sensitive enough that two backends of the same
//! implementation differ by more than the fixed limits. With a second
//! reference, a row may differ from the first reference by as much as the
//! second one does, and the mean by as much as the second one's mean, where
//! that exceeds the fixed limit. The fixed limits stay the floor.
//! DINF_QUALIFY_PREFILL_CHUNK optionally bounds engine prefill tokens per call;
//! it must be a positive integer. The default processes the whole prompt.
//! Optional reference `context_tokens` also sets the stage KV limit. Without
//! it, legacy fixtures retain the model context admission and default KV policy.

use anyhow::{Context, Result, ensure};
use candle_core::quantized::gguf_file::Value;
use candle_core::{DType, Device};
use dinf_engine::architectures::{StageInput, StageOutput, TextRequestContext};
use dinf_engine::gguf_loader::read_gguf_metadata_map;
use dinf_engine::tokenizer::GgufTokenizer;
use dinf_engine::{LayerRange, PipelineStage, read_gguf_metadata};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ModelFile {
    name: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Deserialize)]
struct Reference {
    source: String,
    backend: String,
    model_sha256: String,
    #[serde(default)]
    model_files: Option<Vec<ModelFile>>,
    vocab_size: usize,
    #[serde(default)]
    context_tokens: Option<usize>,
    logits_file: String,
    #[serde(default)]
    second_reference_logits_file: Option<String>,
    #[serde(default)]
    second_reference_backend: Option<String>,
    prompt_tokens: Vec<u32>,
    continuation_tokens: Vec<u32>,
    #[serde(default)]
    teacher_forcing_tokens: Option<Vec<u32>>,
}

fn validate_reference_context(reference: &Reference, model_context: usize) -> Result<()> {
    let limit = reference.context_tokens.unwrap_or(model_context);
    ensure!(
        limit > 0 && limit <= model_context,
        "invalid reference context limit"
    );
    ensure!(
        reference
            .prompt_tokens
            .len()
            .checked_add(reference.continuation_tokens.len())
            .is_some_and(|tokens| tokens <= limit),
        "reference exceeds context limit"
    );
    Ok(())
}

fn teacher_history(reference: &Reference) -> Result<&[u32]> {
    ensure!(
        !reference.prompt_tokens.is_empty() && !reference.continuation_tokens.is_empty(),
        "empty reference"
    );
    let teacher = reference
        .teacher_forcing_tokens
        .as_deref()
        .unwrap_or(&reference.continuation_tokens);
    ensure!(
        teacher.len() == reference.continuation_tokens.len(),
        "teacher-forcing length differs from logit rows"
    );
    ensure!(
        reference
            .prompt_tokens
            .iter()
            .chain(&reference.continuation_tokens)
            .chain(teacher)
            .all(|&id| (id as usize) < reference.vocab_size),
        "reference token outside vocabulary"
    );
    Ok(teacher)
}

fn create_actual_logits(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?,
    ))
}

fn write_actual_logits(writer: &mut impl Write, values: &[f32]) -> Result<()> {
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    writer.write_all(&bytes)?;
    writer.flush()?;
    Ok(())
}

fn parse_prefill_chunk(value: Option<&str>) -> Result<Option<usize>> {
    value
        .map(|value| {
            let chunk: usize = value
                .parse()
                .context("invalid DINF_QUALIFY_PREFILL_CHUNK")?;
            ensure!(chunk > 0, "DINF_QUALIFY_PREFILL_CHUNK must be positive");
            Ok(chunk)
        })
        .transpose()
}

fn valid_hash(hash: &str) -> bool {
    hash.len() == 64
        && hash
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_.-".contains(&c))
}

fn split_integer(metadata: &std::collections::HashMap<String, Value>, key: &str) -> Result<u64> {
    let value = match metadata.get(key) {
        Some(Value::U8(v)) => u64::from(*v),
        Some(Value::U16(v)) => u64::from(*v),
        Some(Value::U32(v)) => u64::from(*v),
        Some(Value::U64(v)) => *v,
        Some(Value::I8(v)) if *v >= 0 => *v as u64,
        Some(Value::I16(v)) if *v >= 0 => *v as u64,
        Some(Value::I32(v)) if *v >= 0 => *v as u64,
        Some(Value::I64(v)) if *v >= 0 => *v as u64,
        _ => anyhow::bail!("missing or invalid {key}"),
    };
    Ok(value)
}

fn validate_model_files(model: &Path, entry_hash: &str, files: Option<&[ModelFile]>) -> Result<()> {
    ensure!(valid_hash(entry_hash), "invalid entry checksum");
    let entry_metadata = read_gguf_metadata_map(model)?;
    let is_split = entry_metadata.keys().any(|key| key.starts_with("split."));
    let entry_name = model
        .file_name()
        .and_then(|name| name.to_str())
        .context("invalid entry name")?;
    // The engine also discovers shards from the filename. Never allow that
    // path to read files that a legacy, entry-only fixture did not hash.
    let discovered_split = entry_name
        .split_once("-00001-of-")
        .and_then(|(_, tail)| tail.split_once('.'))
        .and_then(|(count, _)| count.parse::<usize>().ok())
        .is_some_and(|count| count >= 2);
    ensure!(
        !discovered_split || is_split,
        "split-style filename requires split metadata and a manifest"
    );
    let Some(files) = files else {
        ensure!(
            !is_split,
            "split model requires a complete model_files manifest"
        );
        ensure!(
            sha256(model)? == entry_hash,
            "reference model checksum mismatch"
        );
        return Ok(());
    };
    ensure!(
        (1..=128).contains(&files.len()),
        "invalid model_files count"
    );
    ensure!(
        files[0].name == entry_name && files[0].sha256 == entry_hash,
        "manifest entry differs from model/model_sha256"
    );
    ensure!(
        is_split == (files.len() > 1),
        "manifest count disagrees with split metadata"
    );
    let suffix = format!("-00001-of-{:05}.gguf", files.len());
    let prefix = if is_split {
        entry_name
            .strip_suffix(&suffix)
            .filter(|prefix| !prefix.is_empty() && !prefix.contains("-00001-of-"))
            .context("split entry must use canonical PREFIX-00001-of-NNNNN.gguf naming")?
    } else {
        ""
    };
    let parent = model
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let canonical_parent = parent.canonicalize()?;
    let mut tensor_count = 0u64;
    let mut declared_count = None;
    for (index, file) in files.iter().enumerate() {
        ensure!(
            valid_name(&file.name) && valid_hash(&file.sha256) && file.size_bytes > 0,
            "invalid shard name, size or checksum at index {index}"
        );
        if is_split {
            ensure!(
                file.name == format!("{prefix}-{:05}-of-{:05}.gguf", index + 1, files.len()),
                "missing, reordered or noncanonical shard at index {index}"
            );
        }
        let path = parent.join(&file.name);
        ensure!(
            path.canonicalize()?.parent() == Some(canonical_parent.as_path()),
            "shard resolves outside the model directory"
        );
        ensure!(
            path.is_file() && path.metadata()?.len() == file.size_bytes,
            "shard size differs: {}",
            file.name
        );
        let metadata = read_gguf_metadata_map(&path)?;
        if is_split {
            ensure!(
                metadata
                    .keys()
                    .filter(|key| key.starts_with("split."))
                    .all(|key| matches!(
                        key.as_str(),
                        "split.no" | "split.count" | "split.tensors.count"
                    )),
                "unknown split metadata in {}",
                file.name
            );
            ensure!(
                split_integer(&metadata, "split.no")? == index as u64
                    && split_integer(&metadata, "split.count")? == files.len() as u64,
                "split index/count differs in {}",
                file.name
            );
            let count = split_integer(&metadata, "split.tensors.count")?;
            ensure!(
                count > 0 && declared_count.is_none_or(|previous| previous == count),
                "split tensor total differs in {}",
                file.name
            );
            declared_count = Some(count);
            // The bounded metadata reader above has parsed the whole header.
            // GGUF v2/v3 place the actual local tensor count at bytes8..16.
            let mut header = [0u8; 16];
            File::open(&path)?.read_exact(&mut header)?;
            ensure!(
                &header[..4] == b"GGUF"
                    && matches!(u32::from_le_bytes(header[4..8].try_into()?), 2 | 3),
                "split reference requires GGUF v2/v3"
            );
            tensor_count = tensor_count
                .checked_add(u64::from_le_bytes(header[8..16].try_into()?))
                .context("split tensor count overflow")?;
        } else {
            ensure!(
                !metadata.keys().any(|key| key.starts_with("split.")),
                "unexpected split metadata"
            );
        }
        ensure!(
            sha256(&path)? == file.sha256,
            "shard checksum differs: {}",
            file.name
        );
    }
    ensure!(
        !is_split || declared_count == Some(tensor_count),
        "actual split tensor total differs"
    );
    Ok(())
}

fn log_normalizer(values: &[f32]) -> (f64, f64) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum = values
        .iter()
        .map(|&v| (f64::from(v) - max).exp())
        .sum::<f64>();
    (max, sum.ln())
}

fn compare(reference: &[f32], actual: &[f32]) -> Result<(usize, usize, f64, f64, f64)> {
    ensure!(
        !reference.is_empty() && reference.len() == actual.len(),
        "logit shape mismatch"
    );
    ensure!(
        reference.iter().chain(actual).all(|v| v.is_finite()),
        "non-finite logits"
    );
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold(0, |best, (i, x)| if *x > v[best] { i } else { best })
    };
    let (mr, zr) = log_normalizer(reference);
    let (ma, za) = log_normalizer(actual);
    let mut kl = 0.0;
    let mut max_abs: f64 = 0.0;
    let mut squares = 0.0;
    for (&r, &a) in reference.iter().zip(actual) {
        let (r, a) = (f64::from(r), f64::from(a));
        let log_r = (r - mr) - zr;
        let log_a = (a - ma) - za;
        kl += log_r.exp() * (log_r - log_a);
        max_abs = max_abs.max((r - a).abs());
        squares += (r - a).powi(2);
    }
    Ok((
        argmax(reference),
        argmax(actual),
        kl.max(0.0),
        max_abs,
        (squares / actual.len() as f64).sqrt(),
    ))
}

/// Fixed limits on a comparison: these are regression bounds, not proof of
/// task accuracy.
const MIN_ARGMAX_FRACTION: f64 = 0.95;
const MEAN_KL_LIMIT: f64 = 0.02;
const MAX_KL_LIMIT: f64 = 0.1;

/// The mean limit and each row's limit. `spread` is each row's divergence
/// between the reference and a second run of the same implementation; a
/// limit rises to it and never drops below the fixed one.
fn kl_limits(rows: usize, spread: Option<&[f64]>) -> (f64, Vec<f64>) {
    match spread {
        Some(spread) if !spread.is_empty() => (
            MEAN_KL_LIMIT.max(spread.iter().sum::<f64>() / spread.len() as f64),
            spread.iter().map(|kl| MAX_KL_LIMIT.max(*kl)).collect(),
        ),
        _ => (MEAN_KL_LIMIT, vec![MAX_KL_LIMIT; rows]),
    }
}

/// Whether a comparison passes: enough matching top tokens, the mean
/// divergence within its limit, and every row within its own.
fn comparison_passes(matches: usize, kl: &[f64], mean_limit: f64, row_limits: &[f64]) -> bool {
    let rows = kl.len();
    rows > 0
        && rows == row_limits.len()
        && matches as f64 / rows as f64 >= MIN_ARGMAX_FRACTION
        && kl.iter().sum::<f64>() / rows as f64 <= mean_limit
        && kl.iter().zip(row_limits).all(|(kl, limit)| kl <= limit)
}

fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn run(
    model: &Path,
    backend: &str,
    reference_path: &Path,
    actual_logits_path: Option<&Path>,
) -> Result<()> {
    let reference: Reference =
        serde_json::from_reader(BufReader::new(File::open(reference_path)?))?;
    validate_model_files(
        model,
        &reference.model_sha256,
        reference.model_files.as_deref(),
    )?;
    let metadata = read_gguf_metadata(model)?;
    let teacher = teacher_history(&reference)?;
    // Inkling 2 pads its output head; logits cover the unpadded vocabulary.
    let logits_vocab = metadata
        .inkling2_loop
        .as_ref()
        .map_or(metadata.vocab_size, |inkling| inkling.vocab_size_unpadded);
    ensure!(
        reference.vocab_size == logits_vocab,
        "reference vocabulary differs from model"
    );
    validate_reference_context(&reference, metadata.context_length)?;
    let path = reference_path
        .parent()
        .unwrap_or(Path::new("."))
        .join(&reference.logits_file);
    let mut file = BufReader::new(File::open(path)?);
    let row_bytes = reference
        .vocab_size
        .checked_mul(4)
        .context("row size overflow")?;
    let expected_bytes = row_bytes
        .checked_mul(reference.continuation_tokens.len())
        .context("reference size overflow")?;
    ensure!(
        file.get_ref().metadata()?.len() == expected_bytes as u64,
        "reference logits length mismatch"
    );
    let mut second_file = match &reference.second_reference_logits_file {
        Some(name) => {
            let path = reference_path.parent().unwrap_or(Path::new(".")).join(name);
            let file = BufReader::new(File::open(path)?);
            ensure!(
                file.get_ref().metadata()?.len() == expected_bytes as u64,
                "second reference logits length mismatch"
            );
            Some(file)
        }
        None => None,
    };
    let chunk_env = match std::env::var("DINF_QUALIFY_PREFILL_CHUNK") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error.into()),
    };
    let prefill_chunk = parse_prefill_chunk(chunk_env.as_deref())?;
    let mut actual_logits = actual_logits_path.map(create_actual_logits).transpose()?;
    let device = match backend {
        "cpu" => Device::Cpu,
        #[cfg(target_os = "macos")]
        "metal" => Device::new_metal(0)?,
        #[cfg(feature = "cuda")]
        "cuda" => Device::new_cuda(0)?,
        _ => anyhow::bail!("backend is unavailable"),
    };
    let mut stage = PipelineStage::from_gguf(
        model,
        LayerRange::try_solo(metadata.num_layers)?,
        &device,
        None,
    )?;
    // DINF_CUDA_GRAPH_DECODE=0 issues CUDA decode launches one by one
    // instead of as captured graphs, which is the default.
    if std::env::var_os("DINF_CUDA_GRAPH_DECODE").is_some_and(|value| value == "0") {
        let mut options = stage.options().clone();
        options.cuda_graph_decode = false;
        stage.set_options(options)?;
    }
    if let Some(limit) = reference.context_tokens {
        stage.set_max_kv_tokens(limit);
    }
    let chunk_size = prefill_chunk.unwrap_or(reference.prompt_tokens.len());
    let context = TextRequestContext::new(reference.prompt_tokens.len())?;
    // A session, as generation uses, so the comparison covers the same decode
    // path, including graph-captured steps when they are enabled.
    let mut session = stage.new_session_state();
    stage.prepare_text_request_with_session(&mut session, context)?;
    let mut prefill_output = None;
    for (index, tokens) in reference.prompt_tokens.chunks(chunk_size).enumerate() {
        prefill_output = Some(stage.prefill_text_with_session(
            &mut session,
            StageInput::Tokens(tokens.to_vec()),
            tokens.len(),
            index * chunk_size,
            context,
        )?);
    }
    let mut output = prefill_output.context("empty prefill")?;
    let mut results = Vec::new();
    let mut bytes = vec![0; row_bytes];
    let mut matches = 0;
    let mut row_kl = Vec::new();
    let mut spread = Vec::new();
    for (row, &token) in reference.continuation_tokens.iter().enumerate() {
        let StageOutput::Logits(tensor) = output else {
            anyhow::bail!("expected logits")
        };
        let actual = tensor
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        ensure!(
            actual.len() == reference.vocab_size,
            "actual vocabulary dimension differs"
        );
        if let Some(writer) = actual_logits.as_mut() {
            // Write before numerical checks so a failed row remains available
            // for diagnosis; the comparison still rejects non-finite values.
            write_actual_logits(writer, &actual)?;
        }
        file.read_exact(&mut bytes)?;
        let expected: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let (reference_top, actual_top, kl, max_abs, rms) = compare(&expected, &actual)?;
        ensure!(
            reference_top == token as usize,
            "reference token does not match its logits at row {row}"
        );
        matches += usize::from(reference_top == actual_top);
        row_kl.push(kl);
        let mut result = json!({"row": row, "reference_top": reference_top, "actual_top": actual_top, "kl": kl, "max_abs": max_abs, "rms": rms});
        if let Some(second_file) = second_file.as_mut() {
            second_file.read_exact(&mut bytes)?;
            let second: Vec<_> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let (_, second_top, second_kl, _, _) = compare(&expected, &second)?;
            spread.push(second_kl);
            result["second_reference_top"] = json!(second_top);
            result["second_reference_kl"] = json!(second_kl);
        }
        results.push(result);
        if row + 1 == reference.continuation_tokens.len() {
            break;
        }
        output = stage.forward_text_with_session(
            &mut session,
            StageInput::Tokens(vec![teacher[row]]),
            1,
            reference.prompt_tokens.len() + row,
            context,
        )?;
    }
    let rows = results.len();
    let mean_kl = row_kl.iter().sum::<f64>() / rows as f64;
    let max_kl = row_kl.iter().copied().fold(0.0, f64::max);
    let spread = second_file.is_some().then_some(spread);
    let (mean_limit, row_limits) = kl_limits(rows, spread.as_deref());
    for (result, limit) in results.iter_mut().zip(&row_limits) {
        result["kl_limit"] = json!(limit);
    }
    let second_reference = spread.as_ref().map(|spread| {
        json!({
            "logits_file": reference.second_reference_logits_file,
            "backend": reference.second_reference_backend,
            "mean_kl": spread.iter().sum::<f64>() / spread.len().max(1) as f64,
            "max_kl": spread.iter().copied().fold(0.0, f64::max),
        })
    });
    #[cfg(feature = "cuda")]
    let graph_steps = stage.cuda_graph_decode_stats().steps;
    #[cfg(not(feature = "cuda"))]
    let graph_steps = 0u64;
    // The fixed limits match the selected-model distribution gates used by
    // qualify_kv. These are regression bounds, not proof of task accuracy or
    // universal model support.
    let passed = comparison_passes(matches, &row_kl, mean_limit, &row_limits);
    println!(
        "{}",
        json!({"passed": passed, "backend": backend, "source": reference.source,
        "reference_backend": reference.backend, "model_sha256": reference.model_sha256,
        "explicit_max_kv_tokens": reference.context_tokens,
        "model_context_tokens": metadata.context_length,
        "model_files": reference.model_files,
        "teacher_forcing_tokens": reference.teacher_forcing_tokens,
        "actual_logits_file": actual_logits_path.map(|path| path.display().to_string()),
        "prefill_chunk_tokens": prefill_chunk,
        "cuda_graph_decode_steps": graph_steps,
        "rows": rows, "matching_argmax": matches, "mean_kl": mean_kl, "max_kl": max_kl,
        "limits": {"min_argmax_fraction": MIN_ARGMAX_FRACTION, "mean_kl": mean_limit,
            "max_kl": MAX_KL_LIMIT, "max_kl_is_per_row": spread.is_some()},
        "second_reference": second_reference, "comparisons": results})
    );
    ensure!(passed, "reference comparison failed");
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if (2..=3).contains(&args.len()) && args[0] == "--emit-prompt" {
        let tokenizer = GgufTokenizer::from_gguf(Path::new(&args[1]))?;
        // An optional text file replaces the built-in user message.
        let message = match args.get(2) {
            Some(path) => std::fs::read_to_string(path)?,
            None => "Write a Rust function to merge two sorted slices. Include tests.".into(),
        };
        let tokens = tokenizer.encode_chat(&[("user".into(), message)])?;
        println!(
            "{}",
            tokens
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        );
        return Ok(());
    }
    ensure!(
        (3..=4).contains(&args.len()),
        "usage: qualify_reference MODEL cpu|metal|cuda REFERENCE.json [ACTUAL.f32]; or --emit-prompt MODEL [MESSAGE.txt]"
    );
    run(
        Path::new(&args[0]),
        &args[1],
        Path::new(&args[2]),
        args.get(3).map(Path::new),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;
    use candle_core::quantized::{GgmlDType, QTensor, gguf_file};
    use std::io::{Seek, SeekFrom, Write};

    #[test]
    fn fixed_limits_apply_without_a_second_reference() {
        let (mean_limit, row_limits) = kl_limits(3, None);
        assert_eq!((mean_limit, row_limits.as_slice()), (0.02, &[0.1; 3][..]));
        assert!(comparison_passes(
            3,
            &[0.01, 0.02, 0.03],
            mean_limit,
            &row_limits
        ));
        // One row over the fixed row limit.
        assert!(!comparison_passes(
            3,
            &[0.0, 0.0, 0.11],
            mean_limit,
            &row_limits
        ));
        // Mean over the fixed mean limit with every row inside its own.
        assert!(!comparison_passes(
            3,
            &[0.03, 0.03, 0.03],
            mean_limit,
            &row_limits
        ));
        // Too few matching top tokens.
        assert!(!comparison_passes(
            2,
            &[0.0, 0.0, 0.0],
            mean_limit,
            &row_limits
        ));
        assert!(!comparison_passes(0, &[], mean_limit, &[]));
    }

    #[test]
    fn a_second_reference_raises_limits_only_where_it_differs_more() {
        let spread = [0.001, 0.25, 0.05];
        let (mean_limit, row_limits) = kl_limits(3, Some(&spread));
        assert_eq!(row_limits, [0.1, 0.25, 0.1]);
        assert!((mean_limit - (0.001 + 0.25 + 0.05) / 3.0).abs() < 1e-12);
        // The unstable row may differ as much as the second reference does.
        assert!(comparison_passes(
            3,
            &[0.0, 0.2, 0.0],
            mean_limit,
            &row_limits
        ));
        // A stable row still has the fixed limit.
        assert!(!comparison_passes(
            3,
            &[0.2, 0.0, 0.0],
            mean_limit,
            &row_limits
        ));
        // Nor may the unstable row exceed the second reference.
        assert!(!comparison_passes(
            3,
            &[0.0, 0.26, 0.0],
            mean_limit,
            &row_limits
        ));
        // A quiet second reference leaves the fixed limits in place.
        assert_eq!(kl_limits(2, Some(&[0.001, 0.002])), (0.02, vec![0.1, 0.1]));
    }

    fn token_reference(teacher: Option<serde_json::Value>) -> Result<Reference> {
        let mut value = json!({"source":"test","backend":"cpu","model_sha256":"0".repeat(64),
            "vocab_size":4,"logits_file":"test.f32","prompt_tokens":[0],"continuation_tokens":[1,2]});
        if let Some(teacher) = teacher {
            value["teacher_forcing_tokens"] = teacher;
        }
        Ok(serde_json::from_value(value)?)
    }

    #[test]
    fn explicit_context_limit_preserves_legacy_and_checks_complete_history() -> Result<()> {
        let mut reference = token_reference(None)?;
        assert_eq!(reference.context_tokens, None);
        validate_reference_context(&reference, 3)?;
        assert!(validate_reference_context(&reference, 2).is_err());
        for limit in [0, 2, 5] {
            reference.context_tokens = Some(limit);
            assert!(validate_reference_context(&reference, 4).is_err());
        }
        for limit in [3, 4] {
            reference.context_tokens = Some(limit);
            validate_reference_context(&reference, 4)?;
        }
        Ok(())
    }

    #[test]
    fn optional_teacher_history_keeps_argmax_integrity_separate() -> Result<()> {
        let legacy = token_reference(None)?;
        assert_eq!(teacher_history(&legacy)?, [1, 2]);
        let forced = token_reference(Some(json!([3, 0])))?;
        assert_eq!(teacher_history(&forced)?, [3, 0]);
        assert_eq!(forced.continuation_tokens, [1, 2]);
        let (argmax, _, _, _, _) = compare(&[0.0, 5.0, 0.0, 0.0], &[0.0, 5.0, 0.0, 0.0])?;
        assert_eq!(argmax, forced.continuation_tokens[0] as usize);
        assert_ne!(argmax, teacher_history(&forced)?[0] as usize);
        for teacher in [json!([]), json!([1]), json!([0, 1, 2]), json!([0, 4])] {
            assert!(teacher_history(&token_reference(Some(teacher))?).is_err());
        }
        assert!(token_reference(Some(json!([-1, 0]))).is_err());
        assert!(token_reference(Some(json!([0.5, 0]))).is_err());
        Ok(())
    }

    #[test]
    fn actual_logits_stream_is_little_endian_flushed_and_never_overwrites() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("actual.f32");
        let mut output = create_actual_logits(&path)?;
        write_actual_logits(&mut output, &[1.0, -0.0])?;
        assert_eq!(std::fs::read(&path)?, [0, 0, 128, 63, 0, 0, 0, 128]);
        let nonfinite = f32::from_bits(0x7fc0_1234);
        write_actual_logits(&mut output, &[nonfinite])?;
        assert!(compare(&[0.0], &[nonfinite]).is_err());
        let bytes = std::fs::read(&path)?;
        assert_eq!(&bytes[8..], &nonfinite.to_le_bytes());
        assert!(create_actual_logits(&path).is_err());
        assert_eq!(std::fs::read(&path)?, bytes);
        Ok(())
    }

    #[test]
    fn prefill_chunk_defaults_to_whole_prompt_and_rejects_invalid_budgets() -> Result<()> {
        assert_eq!(parse_prefill_chunk(None)?, None);
        assert_eq!(parse_prefill_chunk(Some("128"))?, Some(128));
        for value in ["0", "-1", "1.5", "128x", "", "184467440737095516160"] {
            assert!(parse_prefill_chunk(Some(value)).is_err(), "{value}");
        }
        Ok(())
    }

    fn write_shard(path: &Path, index: u32, declared: u64) -> Result<()> {
        let metadata = [
            ("split.no", Value::U32(index)),
            ("split.count", Value::U32(3)),
            ("split.tensors.count", Value::U64(declared)),
        ];
        let metadata: Vec<_> = metadata.iter().map(|(k, v)| (*k, v)).collect();
        let tensor =
            QTensor::quantize(&Tensor::new(&[1.0f32, 2.0], &Device::Cpu)?, GgmlDType::F32)?;
        let name = format!("weight{index}");
        let tensors = if index == 0 {
            vec![]
        } else {
            vec![(name.as_str(), &tensor)]
        };
        gguf_file::write(&mut File::create(path)?, &metadata, &tensors)?;
        Ok(())
    }

    fn split_fixture() -> Result<(tempfile::TempDir, Vec<ModelFile>)> {
        let directory = tempfile::tempdir()?;
        let mut files = Vec::new();
        for index in 0..3 {
            let name = format!("fixture-{:05}-of-00003.gguf", index + 1);
            let path = directory.path().join(&name);
            write_shard(&path, index, 2)?;
            files.push(ModelFile {
                name,
                size_bytes: path.metadata()?.len(),
                sha256: sha256(&path)?,
            });
        }
        Ok((directory, files))
    }

    #[test]
    fn split_reference_accepts_metadata_only_entry_and_hashes_nonentry_shards() -> Result<()> {
        let (directory, files) = split_fixture()?;
        let entry = directory.path().join(&files[0].name);
        validate_model_files(&entry, &files[0].sha256, Some(&files))?;
        assert!(validate_model_files(&entry, &files[0].sha256, None).is_err());
        let last = directory.path().join(&files[2].name);
        let mut file = std::fs::OpenOptions::new().write(true).open(&last)?;
        file.seek(SeekFrom::End(-1))?;
        file.write_all(&[1])?;
        drop(file);
        let error = validate_model_files(&entry, &files[0].sha256, Some(&files)).unwrap_err();
        assert!(error.to_string().contains("checksum differs"), "{error:#}");
        Ok(())
    }

    #[test]
    fn split_reference_rejects_missing_reordered_and_traversal_entries() -> Result<()> {
        let (directory, files) = split_fixture()?;
        let entry = directory.path().join(&files[0].name);
        let mut reordered = files.clone();
        reordered.swap(1, 2);
        assert!(validate_model_files(&entry, &files[0].sha256, Some(&reordered)).is_err());
        assert!(validate_model_files(&entry, &files[0].sha256, Some(&files[..2])).is_err());
        for name in [
            "../fixture-00002-of-00003.gguf",
            "/fixture-00002-of-00003.gguf",
            "sub/fixture-00002-of-00003.gguf",
        ] {
            let mut changed = files.clone();
            changed[1].name = name.into();
            assert!(validate_model_files(&entry, &files[0].sha256, Some(&changed)).is_err());
        }
        std::fs::remove_file(directory.path().join(&files[1].name))?;
        assert!(validate_model_files(&entry, &files[0].sha256, Some(&files)).is_err());
        Ok(())
    }

    #[test]
    fn split_reference_rejects_wrong_size_hash_and_tensor_total() -> Result<()> {
        let (directory, files) = split_fixture()?;
        let entry = directory.path().join(&files[0].name);
        let mut wrong = files.clone();
        wrong[1].size_bytes += 1;
        assert!(validate_model_files(&entry, &files[0].sha256, Some(&wrong)).is_err());
        wrong = files.clone();
        wrong[1].sha256 = "not-a-sha256".into();
        assert!(validate_model_files(&entry, &files[0].sha256, Some(&wrong)).is_err());
        wrong = files.clone();
        for (index, file) in wrong.iter_mut().enumerate() {
            let path = directory.path().join(&file.name);
            write_shard(&path, index as u32, 3)?;
            file.sha256 = sha256(&path)?;
        }
        let error = validate_model_files(&entry, &wrong[0].sha256, Some(&wrong)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("actual split tensor total differs"),
            "{error:#}"
        );
        Ok(())
    }

    #[test]
    fn split_reference_rejects_malformed_or_inconsistent_split_metadata() -> Result<()> {
        let (directory, mut files) = split_fixture()?;
        let entry = directory.path().join(&files[0].name);
        let other = directory.path().join(&files[1].name);
        for (key, value) in [
            ("split.count", Value::F32(3.0)),
            ("split.no", Value::I32(-1)),
            ("split.tensors.count", Value::U64(3)),
            ("split.unknown", Value::U32(1)),
        ] {
            let mut metadata = vec![
                ("split.no", Value::U32(1)),
                ("split.count", Value::U32(3)),
                ("split.tensors.count", Value::U64(2)),
            ];
            metadata.retain(|(name, _)| *name != key);
            metadata.push((key, value));
            let refs: Vec<_> = metadata.iter().map(|(k, v)| (*k, v)).collect();
            gguf_file::write(&mut File::create(&other)?, &refs, &[])?;
            files[1].size_bytes = other.metadata()?.len();
            files[1].sha256 = sha256(&other)?;
            assert!(
                validate_model_files(&entry, &files[0].sha256, Some(&files)).is_err(),
                "{key}"
            );
        }
        Ok(())
    }

    #[test]
    fn legacy_unsplit_reference_still_accepts_entry_checksum_only() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("legacy.gguf");
        gguf_file::write(&mut File::create(&path)?, &[], &[])?;
        validate_model_files(&path, &sha256(&path)?, None)?;
        let reference: Reference = serde_json::from_value(json!({"source":"old", "backend":"cpu",
            "model_sha256":sha256(&path)?,"vocab_size":1,"logits_file":"old.f32",
            "prompt_tokens":[0],"continuation_tokens":[0]}))?;
        assert!(reference.model_files.is_none());
        assert!(
            serde_json::from_value::<ModelFile>(
                json!({"name":"a","size_bytes":1,"sha256":"a","extra":true})
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn reference_rejects_unlisted_filename_discovery_and_ambiguous_prefixes() -> Result<()> {
        let directory = tempfile::tempdir()?;
        for name in ["fixture-00001-of-00003.gguf", "fixture-00001-of-3.other"] {
            let path = directory.path().join(name);
            gguf_file::write(&mut File::create(&path)?, &[], &[])?;
            assert!(validate_model_files(&path, &sha256(&path)?, None).is_err());
        }
        let (directory, mut files) = split_fixture()?;
        for file in &mut files {
            let name = format!("other-00001-of-00002.{name}", name = file.name);
            std::fs::rename(
                directory.path().join(&file.name),
                directory.path().join(&name),
            )?;
            file.name = name;
        }
        let entry = directory.path().join(&files[0].name);
        assert!(validate_model_files(&entry, &files[0].sha256, Some(&files)).is_err());
        Ok(())
    }

    #[test]
    fn reference_metrics_match_known_distributions() -> Result<()> {
        let (r, a, kl, _, _) = compare(&[0.0, 0.0], &[0.0, 3.0f32.ln()])?;
        assert_eq!((r, a), (0, 1));
        assert!((kl - (4.0f64 / 3.0).ln() / 2.0).abs() < 1e-7);
        let (_, _, shifted_kl, _, _) = compare(&[1000.0, 1000.0], &[999.0, 999.0])?;
        assert!(shifted_kl < 1e-12);
        let (_, _, large_shift_kl, _, _) = compare(&[1e20, 1e20], &[0.0, 0.0])?;
        assert!(large_shift_kl < 1e-12);
        assert!(compare(&[f32::NAN], &[0.0]).is_err());
        assert!(compare(&[], &[]).is_err());
        Ok(())
    }
}
