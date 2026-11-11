use std::collections::HashMap;

use candle_core::quantized::gguf_file::Value;
use dinf_engine::tokenizer::GgufTokenizer;

fn metadata() -> HashMap<String, Value> {
    HashMap::from([
        (
            "tokenizer.ggml.tokens".into(),
            Value::Array(
                ["<s>", "</s>", "a", "Ġ", "b"]
                    .into_iter()
                    .map(|s| Value::String(s.into()))
                    .collect(),
            ),
        ),
        ("tokenizer.ggml.model".into(), Value::String("gpt2".into())),
        ("tokenizer.ggml.bos_token_id".into(), Value::U32(0)),
        ("tokenizer.ggml.eos_token_id".into(), Value::U32(1)),
        ("tokenizer.ggml.add_bos_token".into(), Value::Bool(false)),
    ])
}

#[test]
fn empty_merges_preserve_gpt2_byte_encoding() {
    let mut metadata = metadata();
    metadata.insert("tokenizer.ggml.merges".into(), Value::Array(vec![]));
    let tokenizer = GgufTokenizer::from_metadata(&metadata).unwrap();
    assert_eq!(tokenizer.encode("a b").unwrap(), [2, 3, 4]);
    assert_eq!(tokenizer.decode(&[2, 3, 4]), "a b");
}

#[test]
fn special_token_ids_reject_wrapping_and_out_of_vocab_values() {
    for key in ["tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"] {
        for value in [
            Value::I32(-1),
            Value::I64(-1),
            Value::U64((1u64 << 32) + 1),
            Value::I64((1i64 << 32) + 1),
            Value::U32(5),
            Value::String("1".into()),
        ] {
            let mut metadata = metadata();
            metadata.insert(key.into(), value);
            assert!(GgufTokenizer::from_metadata(&metadata).is_err(), "{key}");
        }
        for value in [Value::U32(1), Value::I32(1), Value::U64(1), Value::I64(1)] {
            let mut metadata = metadata();
            metadata.insert(key.into(), value);
            assert!(GgufTokenizer::from_metadata(&metadata).is_ok(), "{key}");
        }
    }
}

#[test]
fn malformed_token_types_cannot_shift_special_token_ids() {
    for value in [
        Value::U32(1),
        Value::Array(vec![Value::I32(1); 4]),
        Value::Array(vec![
            Value::I32(1),
            Value::String("bad".into()),
            Value::I32(4),
            Value::I32(1),
            Value::I32(1),
        ]),
        Value::Array(vec![Value::U64(1u64 << 32); 5]),
    ] {
        let mut metadata = metadata();
        metadata.insert("tokenizer.ggml.token_type".into(), value);
        assert!(GgufTokenizer::from_metadata(&metadata).is_err());
    }
}

#[test]
fn malformed_merges_cannot_change_the_tokenizer_algorithm() {
    for value in [Value::U32(1), Value::Array(vec![Value::U32(1)])] {
        let mut metadata = metadata();
        metadata.insert("tokenizer.ggml.merges".into(), value);
        assert!(GgufTokenizer::from_metadata(&metadata).is_err());
    }
}

#[test]
fn incomplete_scores_cannot_change_sentencepiece_merge_order() {
    let mut metadata = metadata();
    metadata.insert("tokenizer.ggml.model".into(), Value::String("llama".into()));
    metadata.insert(
        "tokenizer.ggml.tokens".into(),
        Value::Array(
            ["<s>", "</s>", "▁", "a", "b", "▁a", "ab"]
                .into_iter()
                .map(|s| Value::String(s.into()))
                .collect(),
        ),
    );
    let scores = [-1.0, -1.0, -1.0, -1.0, -1.0, -2.0, -3.0];
    metadata.insert(
        "tokenizer.ggml.scores".into(),
        Value::Array(scores.into_iter().map(Value::F32).collect()),
    );
    let tokenizer = GgufTokenizer::from_metadata(&metadata).unwrap();
    // The score for "▁a" (-2) wins over "ab" (-3).
    assert_eq!(tokenizer.encode("ab").unwrap(), [5, 4]);

    for length in [6, 0, 8] {
        metadata.insert(
            "tokenizer.ggml.scores".into(),
            Value::Array((0..length).map(|i| Value::F32(scores[i % 7])).collect()),
        );
        // A missing score used to default to zero, incorrectly making "ab" win.
        if let Ok(tokenizer) = GgufTokenizer::from_metadata(&metadata) {
            panic!(
                "accepted {length} scores, encoding 'ab' as {:?}",
                tokenizer.encode("ab")
            );
        }
    }
}

#[test]
fn malformed_add_bos_cannot_insert_unrequested_tokens() {
    let mut metadata = metadata();
    assert_eq!(
        GgufTokenizer::from_metadata(&metadata)
            .unwrap()
            .encode("a")
            .unwrap(),
        [2]
    );
    metadata.remove("tokenizer.ggml.add_bos_token");
    // A missing GPT-2 setting follows GGUF's default: no automatic BOS.
    assert_eq!(
        GgufTokenizer::from_metadata(&metadata)
            .unwrap()
            .encode("a")
            .unwrap(),
        [2]
    );
    metadata.insert("tokenizer.ggml.add_bos_token".into(), Value::Bool(true));
    assert_eq!(
        GgufTokenizer::from_metadata(&metadata)
            .unwrap()
            .encode("a")
            .unwrap(),
        [0, 2]
    );
    for value in [
        Value::U32(0),
        Value::String("false".into()),
        Value::F32(0.0),
    ] {
        metadata.insert("tokenizer.ggml.add_bos_token".into(), value);
        if let Ok(tokenizer) = GgufTokenizer::from_metadata(&metadata) {
            panic!(
                "accepted malformed BOS setting, encoding 'a' as {:?}",
                tokenizer.encode("a")
            );
        }
    }
}
