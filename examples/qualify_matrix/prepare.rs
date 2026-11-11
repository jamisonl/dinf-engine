use super::common::{ModelFile, json_file, model_manifest, sha256, valid_name, write_json_new};
use anyhow::{Context, Result, ensure};
use dinf_engine::read_gguf_metadata;
use dinf_engine::tokenizer::GgufTokenizer;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

pub(super) const ROWS: usize = 128;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Corpus {
    pub schema_version: u32,
    pub suite_id: String,
    pub rows: usize,
    pub cases: Vec<CorpusCase>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CorpusCase {
    pub id: String,
    pub domain: String,
    pub context_tokens: usize,
    pub messages: Vec<Message>,
    pub continuation_text: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Prepared {
    pub schema_version: u32,
    pub suite_id: String,
    pub rows: usize,
    pub corpus_sha256: String,
    pub model_sha256: String,
    pub model_files: Vec<ModelFile>,
    pub architecture: String,
    pub vocab_size: usize,
    pub model_context_length: usize,
    pub token_identity_sha256: String,
    pub preparation_binary_sha256: String,
    pub preparation_source_sha256: BTreeMap<String, String>,
    pub cases: Vec<PreparedCase>,
}

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct PreparedCase {
    pub id: String,
    pub domain: String,
    pub context_tokens: usize,
    pub rendered_prompt_utf8: String,
    pub continuation_utf8: String,
    pub prompt_original_ids: Vec<u32>,
    pub prompt_tokens: Vec<u32>,
    pub continuation_original_ids: Vec<u32>,
    pub teacher_forcing_tokens: Vec<u32>,
    pub prompt_add_special: bool,
    pub continuation_add_special: bool,
    pub parse_special: bool,
}

pub(super) fn token_identity(cases: &[PreparedCase]) -> Result<String> {
    use sha2::{Digest, Sha256};
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(cases)?)))
}

pub(super) fn validate_corpus(corpus: &Corpus) -> Result<()> {
    ensure!(
        corpus.schema_version == 1 && corpus.rows == ROWS,
        "expected schema 1 and 128 rows"
    );
    ensure!(valid_name(&corpus.suite_id), "invalid suite ID");
    ensure!(
        (1..=64).contains(&corpus.cases.len()),
        "expected 1..64 cases"
    );
    let mut ids = HashSet::new();
    for case in &corpus.cases {
        ensure!(
            valid_name(&case.id) && ids.insert(&case.id),
            "invalid or duplicate case ID"
        );
        ensure!(!case.domain.is_empty(), "empty domain");
        ensure!(
            [512, 2048, 4096].contains(&case.context_tokens),
            "context must be 512, 2048 or 4096"
        );
        ensure!(
            !case.messages.is_empty() && case.messages.len() <= 128,
            "invalid message count"
        );
        ensure!(
            case.messages.last().is_some_and(|m| m.role == "user"),
            "last message must be user"
        );
        ensure!(
            case.messages.iter().all(|m| matches!(
                m.role.as_str(),
                "system" | "user" | "assistant"
            ) && !m.content.is_empty()),
            "invalid role or empty message"
        );
        ensure!(!case.continuation_text.is_empty(), "empty continuation");
    }
    Ok(())
}

pub(super) fn validate_case(case: &PreparedCase, vocab: usize, model_context: usize) -> Result<()> {
    ensure!(
        valid_name(&case.id) && !case.domain.is_empty(),
        "invalid prepared case identity"
    );
    ensure!(
        [512, 2048, 4096].contains(&case.context_tokens),
        "invalid context budget"
    );
    ensure!(
        case.prompt_add_special && !case.continuation_add_special && case.parse_special,
        "unexpected tokenizer flags"
    );
    ensure!(
        !case.rendered_prompt_utf8.is_empty() && !case.continuation_utf8.is_empty(),
        "empty prepared text"
    );
    ensure!(
        case.prompt_tokens.len() >= 2 && case.prompt_tokens == case.prompt_original_ids,
        "prompt was shortened or changed"
    );
    ensure!(
        case.teacher_forcing_tokens.len() == ROWS
            && case
                .continuation_original_ids
                .starts_with(&case.teacher_forcing_tokens),
        "teacher must be the first 128 continuation tokens"
    );
    ensure!(
        case.context_tokens <= model_context
            && case
                .prompt_tokens
                .len()
                .checked_add(ROWS)
                .is_some_and(|n| n <= case.context_tokens),
        "complete prompt plus 128 rows exceeds context; shortening is forbidden"
    );
    ensure!(
        case.prompt_tokens
            .iter()
            .chain(&case.continuation_original_ids)
            .all(|&id| (id as usize) < vocab),
        "prepared token outside vocabulary"
    );
    Ok(())
}

fn stripped_continuation(tokenizer: &GgufTokenizer, text: &str) -> Result<Vec<u32>> {
    // Public encode() prepends the model's configured BOS. Prove that prefix
    // from the empty input, rather than guessing from a model family name.
    let prefix = tokenizer.encode("")?;
    ensure!(
        prefix.is_empty() || prefix == [tokenizer.bos_id],
        "unexpected empty-input tokenization"
    );
    let encoded = tokenizer.encode(text)?;
    Ok(encoded
        .strip_prefix(prefix.as_slice())
        .context("continuation lacks expected BOS prefix")?
        .to_vec())
}

pub(super) fn render_case(tokenizer: &GgufTokenizer, case: &CorpusCase) -> Result<PreparedCase> {
    let messages: Vec<_> = case
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();
    let rendered_prompt_utf8 = tokenizer.format_chat(&messages);
    let prompt_tokens = tokenizer.encode(&rendered_prompt_utf8)?;
    ensure!(
        prompt_tokens == tokenizer.encode_chat(&messages)?,
        "rendered prompt differs from native chat encoding"
    );
    let continuation_original_ids = stripped_continuation(tokenizer, &case.continuation_text)?;
    ensure!(
        continuation_original_ids.len() >= ROWS,
        "{} continuation has fewer than 128 tokens",
        case.id
    );
    Ok(PreparedCase {
        id: case.id.clone(),
        domain: case.domain.clone(),
        context_tokens: case.context_tokens,
        rendered_prompt_utf8,
        continuation_utf8: case.continuation_text.clone(),
        prompt_original_ids: prompt_tokens.clone(),
        prompt_tokens,
        teacher_forcing_tokens: continuation_original_ids[..ROWS].to_vec(),
        continuation_original_ids,
        prompt_add_special: true,
        continuation_add_special: false,
        parse_special: true,
    })
}

pub(super) fn prepare(model: &Path, corpus_path: &Path, output: &Path) -> Result<()> {
    ensure!(!output.exists(), "output already exists");
    // Keep the caller's filename: split discovery and the manifest use logical
    // GGUF names even when an unsplit HF-cache entry is a symlink to a blob.
    let model = std::path::absolute(model)?;
    let corpus: Corpus = json_file(corpus_path)?;
    validate_corpus(&corpus)?;
    let corpus_sha256 = sha256(corpus_path)?;
    let metadata = read_gguf_metadata(&model)?;
    let tokenizer = GgufTokenizer::from_gguf(&model)?;
    let mut cases = Vec::new();
    for case in &corpus.cases {
        let prepared =
            render_case(&tokenizer, case).with_context(|| format!("prepare {}", case.id))?;
        validate_case(&prepared, metadata.vocab_size, metadata.context_length)?;
        cases.push(prepared);
    }
    // Hash the complete model only after the inexpensive corpus checks pass.
    let model_files = model_manifest(&model)?;
    let result = Prepared {
        schema_version: 1,
        suite_id: corpus.suite_id,
        rows: ROWS,
        corpus_sha256,
        model_sha256: model_files[0].sha256.clone(),
        model_files,
        architecture: metadata.architecture,
        vocab_size: metadata.vocab_size,
        model_context_length: metadata.context_length,
        token_identity_sha256: token_identity(&cases)?,
        preparation_binary_sha256: sha256(&std::env::current_exe()?)?,
        preparation_source_sha256: super::compiled_sources(),
        cases,
    };
    ensure!(
        sha256(corpus_path)? == result.corpus_sha256,
        "corpus changed during preparation"
    );
    write_json_new(output, &result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case() -> PreparedCase {
        PreparedCase {
            id: "test".into(),
            domain: "test".into(),
            context_tokens: 512,
            rendered_prompt_utf8: "complete chat".into(),
            continuation_utf8: "natural text".into(),
            prompt_original_ids: vec![1; 384],
            prompt_tokens: vec![1; 384],
            continuation_original_ids: vec![2; 129],
            teacher_forcing_tokens: vec![2; 128],
            prompt_add_special: true,
            continuation_add_special: false,
            parse_special: true,
        }
    }

    #[test]
    fn context_admission_never_shortens_prompt_or_teacher() -> Result<()> {
        let original = case();
        validate_case(&original, 3, 4096)?;
        let mut too_long = original.clone();
        too_long.prompt_tokens.push(1);
        too_long.prompt_original_ids.push(1);
        assert!(validate_case(&too_long, 3, 4096).is_err());
        let mut shortened = original.clone();
        shortened.prompt_tokens.pop();
        assert!(validate_case(&shortened, 3, 4096).is_err());
        let mut changed = original.clone();
        changed.teacher_forcing_tokens[127] = 1;
        assert!(validate_case(&changed, 3, 4096).is_err());
        assert!(validate_case(&original, 3, 511).is_err());
        assert!(validate_case(&original, 2, 4096).is_err());
        Ok(())
    }

    #[test]
    fn corpus_rejects_duplicate_cases_and_old_cap_policy() -> Result<()> {
        let value = serde_json::json!({"schema_version":1,"suite_id":"s","rows":128,"cases":[
            {"id":"x","domain":"code","context_tokens":512,"messages":[{"role":"user","content":"x"}],"continuation_text":"y"}]});
        let corpus: Corpus = serde_json::from_value(value.clone())?;
        validate_corpus(&corpus)?;
        let mut duplicate = value.clone();
        duplicate["cases"]
            .as_array_mut()
            .unwrap()
            .push(value["cases"][0].clone());
        assert!(validate_corpus(&serde_json::from_value(duplicate)?).is_err());
        let mut cap = value;
        cap["cases"][0]["context_policy"] = "cap_to_model_limit_minus_rows".into();
        assert!(serde_json::from_value::<Corpus>(cap).is_err());
        Ok(())
    }
}
