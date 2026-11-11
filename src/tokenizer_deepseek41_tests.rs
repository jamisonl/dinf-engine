//! Official fixture attribution and MIT notice: tokenizer_deepseek41.rs and
//! tests/fixtures/deepseek41-1017-goldens.json.
use super::*;
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    thinking_mode: String,
    reasoning_effort: Option<String>,
    #[serde(default)]
    tools: Vec<Tool>,
    messages: Vec<Message>,
    expected: String,
}
#[derive(Deserialize)]
struct Tool {
    function: Function,
}
#[derive(Deserialize)]
struct Function {
    name: String,
    description: Option<String>,
    parameters: Option<Json>,
    arguments: Option<String>,
}
#[derive(Deserialize)]
struct Message {
    role: String,
    content: Value,
    reasoning_content: Option<String>,
    tool_call_id: Option<String>,
    #[serde(default)]
    tool_calls: Vec<Tool>,
    #[serde(default)]
    tools: Vec<Tool>,
    task: Option<String>,
}
fn tools(input: &[Tool]) -> Vec<ChatTemplateTool> {
    input
        .iter()
        .map(|tool| ChatTemplateTool {
            name: tool.function.name.clone(),
            description: tool.function.description.clone(),
            parameters_json: tool.function.parameters.as_ref().map(Json::render),
        })
        .collect()
}
fn options(effort: &str) -> ChatTemplateOptions {
    ChatTemplateOptions::new([]).with_reasoning_effort(Some(effort.into()))
}

#[test]
fn tokenizer_deepseek41_five_official_format_goldens() {
    let fixtures: Fixture = serde_json::from_str(include_str!(
        "../tests/fixtures/deepseek41-1017-goldens.json"
    ))
    .unwrap();
    for (index, case) in fixtures.cases.into_iter().enumerate() {
        let messages = case
            .messages
            .iter()
            .map(|source| {
                // Official image preprocessing is format-only here. Native admission
                // rejects images; no vision processor or inference is advertised.
                let content = if let Some(text) = source.content.as_str() {
                    text.to_owned()
                } else {
                    source
                        .content
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|block| {
                            if block["type"] == "text" {
                                block["text"].as_str().unwrap()
                            } else {
                                "<｜deepseek_image｜>"
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n\n")
                };
                let mut message = ChatTemplateMessage::new(&source.role, content);
                message.reasoning_content = source.reasoning_content.clone();
                message.tool_call_id = source.tool_call_id.clone();
                message.tool_calls = source
                    .tool_calls
                    .iter()
                    .map(|call| {
                        ChatTemplateToolCall::new(
                            &call.function.name,
                            call.function.arguments.as_deref().unwrap(),
                        )
                    })
                    .collect();
                message
            })
            .collect::<Vec<_>>();
        let decorations: HashMap<_, _> = case
            .messages
            .iter()
            .enumerate()
            .map(|(index, message)| {
                (
                    index,
                    Decoration {
                        tools: tools(&message.tools),
                        task: message.task.clone(),
                    },
                )
            })
            .collect();
        let options = ChatTemplateOptions::new(tools(&case.tools)).with_reasoning_effort(
            if case.thinking_mode == "chat" {
                Some("none".into())
            } else {
                case.reasoning_effort
            },
        );
        let prompt = if decorations
            .values()
            .all(|decoration| decoration.tools.is_empty() && decoration.task.is_none())
        {
            format_chat(&messages, &options)
        } else {
            format_prepared(&messages, &options, &decorations)
        };
        assert_eq!(
            prompt.unwrap(),
            case.expected,
            "official golden {}",
            index + 1
        );
    }
}

#[test]
fn tokenizer_deepseek41_literal_prefix_efforts_and_errors() {
    let messages = [ChatTemplateMessage::new("user", "hello")];
    assert_eq!(
        format_chat(&messages, &options("none")).unwrap(),
        "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>"
    );
    assert_eq!(
        format_chat(&messages, &ChatTemplateOptions::new([])).unwrap(),
        "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>hello<｜Assistant｜><think>"
    );
    assert_eq!(
        format_chat(
            &[
                ChatTemplateMessage::new("system", "You are a helpful assistant."),
                ChatTemplateMessage::new("user", "hello"),
            ],
            &options("none"),
        )
        .unwrap(),
        "<｜begin▁of▁sentence｜><｜System｜>You are a helpful assistant.<｜User｜>hello<｜Assistant｜></think>"
    );
    for (effort, budget) in [("low", 50), ("high", 75), ("max", 100)] {
        let prompt = format_chat(&messages, &options(effort)).unwrap();
        assert!(prompt.starts_with(&format!(
            "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: {budget} (range 1-100,"
        )));
        assert!(prompt.ends_with("<｜Assistant｜><think>"));
    }
    for effort in ["medium", "xhigh", "0", "101", "off"] {
        assert!(format_chat(&messages, &options(effort)).is_err());
    }
    for name in ["ns::tool", "tool", "ns::tool.name"] {
        assert!(valid_name(name));
    }
    for name in ["ns::a::b", "::tool", "ns::", "a\"", "a:b"] {
        assert!(!valid_name(name));
    }
    assert!(
        render_call(&ChatTemplateToolCall::new(
            "f",
            r#"{"x":"</｜DSML｜ parameter>"}"#
        ))
        .is_err()
    );
    assert!(render_call(&ChatTemplateToolCall::new("f", r#"{"bad\nkey":"x"}"#)).is_err());
    assert!(render_call(&ChatTemplateToolCall::new("f", r#"{"x":1,"x":2}"#)).is_err());
}

#[test]
fn tokenizer_deepseek41_tool_cycles_order_and_reasoning_retention() {
    let messages = [
        ChatTemplateMessage::new("user", "old"),
        ChatTemplateMessage::new("assistant", "old answer").with_reasoning_content("old reasoning"),
        ChatTemplateMessage::new("developer", "mid system"),
        ChatTemplateMessage::new("assistant", "summary")
            .with_reasoning_content("current reasoning")
            .with_tool_calls([
                ChatTemplateToolCall::new(
                    "ns::first",
                    r#"{"text":"  a\n>; && <div>雪</div>  ","n":2,"v":[true,null,{"x":"\\\""}]}"#,
                )
                .with_id("a"),
                ChatTemplateToolCall::new("second", "{}").with_id("b"),
            ]),
        ChatTemplateMessage::new("tool", "B").with_tool_call_id("b"),
        ChatTemplateMessage::new("user", "text position"),
        ChatTemplateMessage::new("tool", "A").with_tool_call_id("a"),
    ];
    let prompt = format_chat(&messages[..5], &options("high")).unwrap();
    assert!(!prompt.contains("old reasoning"));
    assert!(prompt.contains("<｜System｜>mid system<｜Assistant｜><think>current reasoning</think>summary\n\n<｜DSML｜ calls>"));
    assert!(prompt.ends_with("<｜User｜><tool_result>B</tool_result><｜Assistant｜><think>"));
    let prompt = format_chat(&messages, &options("high")).unwrap();
    assert!(prompt.ends_with("<｜User｜><tool_result>A</tool_result>\n\ntext position\n\n<tool_result>B</tool_result><｜Assistant｜><think>"));
    assert!(!prompt.contains("current reasoning")); // ordinary user text starts a new turn
    let with_tools = ChatTemplateOptions::new([ChatTemplateTool::new("ns::first")]);
    assert!(
        format_chat(&messages, &with_tools)
            .unwrap()
            .contains("old reasoning")
    );
}

#[test]
fn tokenizer_deepseek41_joyai_isolated_split_goldens() {
    for (text, expected) in [
        ("中文abcかなカナZ", vec!["中文", "abc", "かなカナ", "Z"]),
        (
            " 1234567890 １２３４",
            vec![" ", "123", "456", "789", "0", " ", "１２３", "４"],
        ),
        (
            "foo.bar++;\r\n    baz",
            vec!["foo", ".bar", "++;\r\n", "   ", " baz"],
        ),
        (
            "e\u{301}\u{a0}word\u{2003}\n\tend  ",
            vec!["e\u{301}", "\u{a0}word", "\u{2003}\n", "\tend", "  "],
        ),
    ] {
        assert_eq!(pretokenize(text).unwrap(), expected, "{text:?}");
    }
}

fn metadata(tokens: Vec<String>, merges: Vec<String>) -> HashMap<String, gguf_file::Value> {
    use gguf_file::Value as G;
    // Maya's GGUF keeps the five non-special AddedTokens at type 1.
    // Synthetic controls are explicit; never infer GGUF types from spelling.
    let token_types = tokens
        .iter()
        .map(|token| {
            G::I32(match token.as_str() {
                "<｜begin▁of▁sentence｜>"
                | "<｜end▁of▁sentence｜>"
                | "<｜System｜>"
                | "<｜User｜>"
                | "<｜Assistant｜>" => 3,
                _ => 1,
            })
        })
        .collect();
    let empty_merges = merges.is_empty();
    let mut metadata = HashMap::from([
        ("tokenizer.ggml.token_type".into(), G::Array(token_types)),
        (
            "general.architecture".into(),
            G::String("deepseek41".into()),
        ),
        ("tokenizer.ggml.model".into(), G::String("gpt2".into())),
        ("tokenizer.ggml.pre".into(), G::String("joyai-llm".into())),
        (
            "tokenizer.ggml.tokens".into(),
            G::Array(tokens.into_iter().map(G::String).collect()),
        ),
        (
            "tokenizer.ggml.merges".into(),
            G::Array(merges.into_iter().map(G::String).collect()),
        ),
        ("tokenizer.ggml.bos_token_id".into(), G::U32(0)),
        ("tokenizer.ggml.eos_token_id".into(), G::U32(1)),
        ("tokenizer.ggml.add_bos_token".into(), G::Bool(true)),
    ]);
    if empty_merges {
        metadata.remove("tokenizer.ggml.merges");
    }
    metadata
}

#[test]
fn tokenizer_deepseek41_bos_once_and_incremental_prefix() {
    let mut tokens = vec![
        "<｜begin▁of▁sentence｜>".into(),
        "<｜end▁of▁sentence｜>".into(),
        "<｜System｜>".into(),
        "<｜User｜>".into(),
        "<｜Assistant｜>".into(),
        "<think>".into(),
        "</think>".into(),
        "｜DSML｜".into(),
    ];
    tokens.extend(build_gpt2_byte_to_unicode().values().map(char::to_string));
    let tokenizer = GgufTokenizer::from_metadata(&metadata(tokens, Vec::new())).unwrap();
    assert_eq!(tokenizer.chat_style, ChatStyle::DeepSeekV41);
    let prompt = tokenizer
        .format_chat_with_tool_options(
            &[ChatTemplateMessage::new("user", "hello")],
            &options("none"),
        )
        .unwrap();
    let encoded = tokenizer.encode(&prompt).unwrap();
    assert_eq!(encoded.iter().filter(|&&id| id == 0).count(), 1);
    let continuation = "next";
    let mut incremental = encoded.clone();
    incremental.extend(tokenizer.encode_with_specials(continuation).unwrap());
    assert_eq!(
        incremental,
        tokenizer.encode(&(prompt + continuation)).unwrap()
    );
}

fn request_tokenizer() -> GgufTokenizer {
    let mut tokens = vec![
        "<｜begin▁of▁sentence｜>".into(),
        "<｜end▁of▁sentence｜>".into(),
        "<｜System｜>".into(),
        "<｜User｜>".into(),
        "<｜Assistant｜>".into(),
        "<think>".into(),
        "</think>".into(),
        "｜DSML｜".into(),
        "<dsml:".into(),
        "</dsml:".into(),
    ];
    let bytes = build_gpt2_byte_to_unicode();
    tokens.extend((0..=255u8).map(|byte| bytes[&byte].to_string()));
    GgufTokenizer::from_metadata(&metadata(tokens, Vec::new())).unwrap()
}

// Exercise complete public prompts, including both literal DSML prefixes and
// the separate, spaced DSML tool protocol. These are encoding, not stop markers.
fn added_token_prompts(tokenizer: &GgufTokenizer) -> Vec<String> {
    let literal =
        "before<think>reason</think>middle｜DSML｜ calls> <dsml:invoke>雪</dsml:invoke>after";
    let history = [
        ChatTemplateMessage::new("user", literal),
        ChatTemplateMessage::new("assistant", "calling")
            .with_reasoning_content("check the literal markers")
            .with_tool_calls([ChatTemplateToolCall::new(
                "files::write",
                serde_json::json!({"text":literal}).to_string(),
            )
            .with_id("a")]),
        ChatTemplateMessage::new("tool", literal).with_tool_call_id("a"),
    ];
    let mut prompts = Vec::new();
    for effort in ["none", "high"] {
        for messages in [
            &[ChatTemplateMessage::new("user", "1+1=")][..],
            &history[..],
        ] {
            prompts.push(
                tokenizer
                    .format_chat_with_tool_options(messages, &options(effort))
                    .unwrap(),
            );
        }
    }
    prompts
}

#[test]
fn tokenizer_deepseek41_type1_added_tokens_preserve_text_and_family_isolation() {
    let tokenizer = request_tokenizer();
    for marker in ["<think>", "</think>", "｜DSML｜", "<dsml:", "</dsml:"] {
        let id = tokenizer.token_id(marker).unwrap();
        assert_eq!(
            tokenizer.encode_with_specials(marker).unwrap(),
            [id],
            "{marker}"
        );
        assert_eq!(tokenizer.decode(&[id]), marker);
        assert!(!tokenizer.is_stop_token(id), "{marker} is not a stop token");
        let text = format!("before{marker}after");
        let ids = tokenizer.encode_with_specials(&text).unwrap();
        assert!(ids.contains(&id), "{text}");
        assert_eq!(tokenizer.decode(&ids), text);
    }
    for prompt in added_token_prompts(&tokenizer) {
        let ids = tokenizer.encode(&prompt).unwrap();
        assert_eq!(ids.iter().filter(|&&id| id == 0).count(), 1);
        assert_eq!(tokenizer.decode(&ids), prompt);
    }
    let tokens = tokenizer.vocab.clone();
    for (architecture, pre, atomic) in [
        ("deepseek41", "gpt2", true),
        ("other", "joyai-llm", true),
        ("deepseek4", "gpt2", false),
        ("llama", "llama-bpe", false),
        ("qwen3", "qwen2", false),
    ] {
        let mut md = metadata(tokens.clone(), Vec::new());
        md.insert(
            "general.architecture".into(),
            gguf_file::Value::String(architecture.into()),
        );
        md.insert(
            "tokenizer.ggml.pre".into(),
            gguf_file::Value::String(pre.into()),
        );
        let family = GgufTokenizer::from_metadata(&md).unwrap();
        for marker in ["<think>", "</think>", "｜DSML｜", "<dsml:", "</dsml:"] {
            let ids = family.encode_with_specials(marker).unwrap();
            if atomic {
                assert_eq!(
                    ids,
                    [family.token_id(marker).unwrap()],
                    "{architecture}/{pre}: {marker}"
                );
            } else {
                assert_eq!(
                    ids,
                    family.encode_plain(marker).unwrap(),
                    "{architecture}/{pre}: {marker}"
                );
                assert!(!ids.contains(&family.token_id(marker).unwrap()));
            }
            assert_eq!(family.decode(&ids), marker);
        }
    }
}

#[test]
fn tokenizer_deepseek41_public_encoding_preserves_literal_image_and_think_text() {
    let tokenizer = request_tokenizer();
    for text in [
        "<image>",
        "</image>",
        "<｜deepseek_image｜>",
        "/think /no_think\n<div>雪</div>",
    ] {
        let plain = [("user".into(), text.into())];
        let messages = [ChatTemplateMessage::new("user", text)];
        let prompt = tokenizer
            .format_chat_with_tool_options(&messages, &ChatTemplateOptions::new([]))
            .unwrap();
        assert!(prompt.contains(text));
        assert!(prompt.ends_with("<｜Assistant｜><think>"));
        assert_eq!(
            tokenizer.encode_chat(&plain).unwrap(),
            tokenizer.encode(&prompt).unwrap()
        );
    }
    let invalid = [("unknown".into(), "text".into())];
    assert!(tokenizer.encode_chat(&invalid).is_err());
    let invalid_history = [ChatTemplateMessage::new("assistant", "").with_tool_calls([
        ChatTemplateToolCall::new("f", r#"{"text":"</｜DSML｜ parameter>"}"#),
    ])];
    assert!(
        tokenizer
            .encode_chat_with_tool_options(&invalid_history, &ChatTemplateOptions::new([]))
            .is_err()
    );
}

#[test]
fn tokenizer_deepseek41_tool_choice_controls_declarations_and_constraint() {
    use crate::tool_constraints::ToolConstraint;

    let tokenizer = request_tokenizer();
    let messages = [
        ChatTemplateMessage::new("user", "write"),
        ChatTemplateMessage::new("assistant", "")
            .with_reasoning_content("history reasoning")
            .with_tool_calls([
                ChatTemplateToolCall::new("files::write", r#"{"text":"old"}"#).with_id("a"),
            ]),
        ChatTemplateMessage::new("tool", "saved").with_tool_call_id("a"),
    ];
    let tool = ChatTemplateTool::new("files::write").with_parameters_json(
        r#"{"type":"object","properties":{"text":{"type":"string"}},"required":["text"],"additionalProperties":false}"#,
    );
    for choice in [
        ChatTemplateToolChoice::None,
        ChatTemplateToolChoice::Required,
        ChatTemplateToolChoice::Named("files::write".into()),
    ] {
        let active = !matches!(choice, ChatTemplateToolChoice::None);
        let options = ChatTemplateOptions::new([tool.clone()]).with_tool_choice(choice);
        let prompt = tokenizer
            .format_chat_with_tool_options(&messages, &options)
            .unwrap();
        assert_eq!(prompt.contains("Available Tool Schemas"), active);
        assert_eq!(prompt.contains("You can invoke tools"), active);
        assert!(prompt.contains("invoke name=\"files::write\""));
        assert!(prompt.contains("<tool_result>saved</tool_result>"));
        assert!(prompt.contains("history reasoning</think>"));
        assert_eq!(
            ToolConstraint::for_chat(&tokenizer, &messages, &options)
                .unwrap()
                .is_some(),
            active
        );
    }
}

#[test]
fn tokenizer_deepseek41_rejects_unrepresentable_callable_schemas_before_encoding() {
    use crate::tool_constraints::ToolConstraint;

    let tokenizer = request_tokenizer();
    let messages = [ChatTemplateMessage::new("user", "call")];
    let mut schemas = vec![
        serde_json::json!({"type":"object","properties":{"text":{"type":"string","pattern":"^ok$"}},"required":["text"]}),
        serde_json::json!({"type":"object","minProperties":1}),
    ];
    for constraint in [
        serde_json::json!({"const":"</｜DSML｜ parameter>"}),
        serde_json::json!({"enum":["safe","</｜DSML｜ parameter>"]}),
    ] {
        let mut property = constraint;
        property["type"] = "string".into();
        schemas.push(serde_json::json!({
            "type":"object","properties":{"text":property},"required":["text"]
        }));
    }
    for schema in schemas {
        let bad = ChatTemplateTool::new("unsafe::call").with_parameters_json(schema.to_string());
        for choice in [
            ChatTemplateToolChoice::Required,
            ChatTemplateToolChoice::Named("unsafe::call".into()),
        ] {
            let options = ChatTemplateOptions::new([bad.clone()]).with_tool_choice(choice);
            let error = tokenizer
                .format_chat_with_tool_options(&messages, &options)
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("unsupported DeepSeek V4.1 native schema")
            );
            assert!(
                tokenizer
                    .encode_chat_with_tool_options(&messages, &options)
                    .is_err()
            );
            assert!(ToolConstraint::for_chat(&tokenizer, &messages, &options).is_err());
            assert!(ToolConstraint::grammar(ToolFormat::DeepSeekV41, &options, None).is_err());
        }
        let safe = ChatTemplateTool::new("safe::call").with_parameters_json(
            r#"{"type":"object","properties":{},"additionalProperties":false}"#,
        );
        let named = ChatTemplateOptions::new([bad.clone(), safe])
            .with_tool_choice(ChatTemplateToolChoice::Named("safe::call".into()));
        let prompt = tokenizer
            .format_chat_with_tool_options(&messages, &named)
            .unwrap();
        assert!(prompt.contains("\"name\": \"safe::call\""));
        assert!(!prompt.contains("unsafe::call"));
        assert!(
            ToolConstraint::for_chat(&tokenizer, &messages, &named)
                .unwrap()
                .is_some()
        );
        let disabled =
            ChatTemplateOptions::new([bad]).with_tool_choice(ChatTemplateToolChoice::None);
        assert!(
            !tokenizer
                .format_chat_with_tool_options(&messages, &disabled)
                .unwrap()
                .contains("Available Tool Schemas")
        );
        assert!(
            ToolConstraint::for_chat(&tokenizer, &messages, &disabled)
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn tokenizer_deepseek41_efforts_preserve_literals_and_tool_cycle_framing() {
    let tokenizer = request_tokenizer();
    let literal = "/think /no_think\n<image>HTML</image>\n雪 && >;";
    let messages = [
        ChatTemplateMessage::new("user", literal),
        ChatTemplateMessage::new("assistant", "")
            .with_reasoning_content("call reasoning")
            .with_tool_calls([ChatTemplateToolCall::new(
                "files::write",
                serde_json::json!({"text":literal}).to_string(),
            )
            .with_id("a")]),
        ChatTemplateMessage::new("tool", literal).with_tool_call_id("a"),
    ];
    let tool = ChatTemplateTool::new("files::write").with_parameters_json(
        r#"{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}"#,
    );
    for effort in ["none", "low", "high", "max"] {
        for tools in [Vec::new(), vec![tool.clone()]] {
            let options =
                ChatTemplateOptions::new(tools).with_reasoning_effort(Some(effort.into()));
            let prompt = tokenizer
                .format_chat_with_tool_options(&messages, &options)
                .unwrap();
            assert_eq!(prompt.matches(literal).count(), 3);
            let close = if effort == "none" {
                "</think>"
            } else {
                "<think>"
            };
            assert!(prompt.ends_with(&format!("<｜Assistant｜>{close}")));
            if effort == "none" {
                assert!(!prompt.contains("Reasoning Effort:"));
                assert!(!prompt.contains("call reasoning"));
            } else {
                let budget = match effort {
                    "low" => 50,
                    "high" => 75,
                    "max" => 100,
                    _ => unreachable!(),
                };
                assert!(prompt.contains(&format!("Reasoning Effort: {budget} (range 1-100,")));
                assert!(prompt.contains("<think>call reasoning</think>"));
            }
            let encoded = tokenizer
                .encode_chat_with_tool_options(&messages, &options)
                .unwrap();
            assert_eq!(encoded.iter().filter(|&&id| id == 0).count(), 1);
        }
    }
    for effort in ["medium", "xhigh", "50", "75", "100", "arbitrary"] {
        for tools in [Vec::new(), vec![tool.clone()]] {
            let options =
                ChatTemplateOptions::new(tools).with_reasoning_effort(Some(effort.into()));
            assert!(
                tokenizer
                    .encode_chat_with_tool_options(&messages, &options)
                    .is_err()
            );
        }
    }
}

/// Directory holding the external tokenizer oracle fixture:
/// `maya-xs-header.json` (GGUF header metadata dumped as JSON) and
/// `official-encoding-1017/tokenizer.json` (the upstream Hugging Face tokenizer).
const TOKENIZER_FIXTURE_ENV: &str = "DINF_DEEPSEEK41_TOKENIZER_FIXTURE";

fn tokenizer_fixture_dir() -> Option<std::path::PathBuf> {
    std::env::var_os(TOKENIZER_FIXTURE_ENV).map(std::path::PathBuf::from)
}

// Read the actual pinned GGUF header, independently of the synthetic fixture's
// type assignment and the production AddedToken recognition.
fn pinned_gguf_metadata(dir: &std::path::Path) -> HashMap<String, gguf_file::Value> {
    use gguf_file::Value as G;
    let path = dir.join("maya-xs-header.json");
    let header: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let source = header["metadata"].as_object().unwrap();
    assert_eq!(
        source["general.source.revision"],
        "1bae0e1088993b9b82de08f59de5be5a34e42170"
    );
    let mut metadata = HashMap::new();
    for key in [
        "general.architecture",
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
    ] {
        metadata.insert(key.into(), G::String(source[key].as_str().unwrap().into()));
    }
    for key in ["tokenizer.ggml.tokens", "tokenizer.ggml.merges"] {
        metadata.insert(
            key.into(),
            G::Array(
                source[key]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| G::String(value.as_str().unwrap().into()))
                    .collect(),
            ),
        );
    }
    metadata.insert(
        "tokenizer.ggml.token_type".into(),
        G::Array(
            source["tokenizer.ggml.token_type"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| G::I32(value.as_i64().unwrap().try_into().unwrap()))
                .collect(),
        ),
    );
    for key in ["tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"] {
        metadata.insert(
            key.into(),
            G::U32(source[key].as_u64().unwrap().try_into().unwrap()),
        );
    }
    metadata.insert(
        "tokenizer.ggml.add_bos_token".into(),
        G::Bool(source["tokenizer.ggml.add_bos_token"].as_bool().unwrap()),
    );
    metadata
}

// Development oracle only: the production path uses GGUF vocab/merges, not a sidecar.
#[test]
#[ignore = "requires external tokenizer fixture; set DINF_DEEPSEEK41_TOKENIZER_FIXTURE"]
fn tokenizer_deepseek41_official_oracle() {
    let Some(dir) = tokenizer_fixture_dir() else {
        eprintln!("skipping: {TOKENIZER_FIXTURE_ENV} is not set");
        return;
    };
    let path = dir.join("official-encoding-1017/tokenizer.json");
    let official = tokenizers::Tokenizer::from_file(&path).unwrap();
    let source: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let metadata = pinned_gguf_metadata(&dir);
    let tokenizer = GgufTokenizer::from_metadata(&metadata).unwrap();
    assert_eq!(official.get_vocab_size(true), tokenizer.vocab_size());
    for (token, id) in official.get_vocab(true) {
        assert_eq!(
            tokenizer.token_id(&token),
            Some(id),
            "GGUF vocabulary: {token}"
        );
    }
    let official_merges = source["model"]["merges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|pair| {
            if let Some(text) = pair.as_str() {
                text.to_owned()
            } else {
                format!(
                    "{} {}",
                    pair[0].as_str().unwrap(),
                    pair[1].as_str().unwrap()
                )
            }
        })
        .collect::<Vec<_>>();
    let gguf_merges = metadata["tokenizer.ggml.merges"]
        .to_vec()
        .unwrap()
        .iter()
        .map(|value| value.to_string().unwrap().clone())
        .collect::<Vec<_>>();
    assert_eq!(gguf_merges, official_merges);
    let types = metadata["tokenizer.ggml.token_type"].to_vec().unwrap();
    for (marker, id) in [
        ("<think>", 128821),
        ("</think>", 128822),
        ("｜DSML｜", 128825),
        ("<dsml:", 128840),
        ("</dsml:", 128841),
    ] {
        assert!(
            matches!(types[id], gguf_file::Value::I32(1)),
            "GGUF type: {marker}"
        );
        assert!(
            !source["model"]["vocab"]
                .as_object()
                .unwrap()
                .contains_key(marker)
        );
        let added = source["added_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .find(|token| token["content"] == marker)
            .unwrap();
        assert_eq!(added["id"], id);
        assert_eq!(added["special"], false);
    }
    let prompt = tokenizer
        .format_chat_with_tool_options(
            &[ChatTemplateMessage::new("user", "1+1=")],
            &options("none"),
        )
        .unwrap();
    assert_eq!(
        prompt,
        "<｜begin▁of▁sentence｜><｜User｜>1+1=<｜Assistant｜></think>"
    );
    let expected = [0, 128803, 19, 13, 19, 31, 128804, 128822];
    assert_eq!(
        official.encode(prompt.as_str(), false).unwrap().get_ids(),
        expected
    );
    assert_eq!(
        tokenizer.encode(&prompt).unwrap(),
        expected,
        "native smoke prompt"
    );
    for (token, id) in [
        ("<｜begin▁of▁sentence｜>", 0),
        ("<｜end▁of▁sentence｜>", 1),
        ("<｜System｜>", 128799),
        ("<｜User｜>", 128803),
        ("<｜Assistant｜>", 128804),
        ("<think>", 128821),
        ("</think>", 128822),
        ("｜DSML｜", 128825),
        ("<dsml:", 128840),
        ("</dsml:", 128841),
        ("<｜latest_reminder｜>", 128828),
        ("<｜deepseek_image｜>", 129264),
    ] {
        assert_eq!(tokenizer.encode_with_specials(token).unwrap(), [id]);
    }
    let mut texts = vec![
        "中文abcかなカナZ".to_owned(),
        "1234567890 １２３４ ١٢٣٤٥٦٧".to_owned(),
        "foo.bar++;\r\n    baz >;\n".to_owned(),
        "e\u{301}\u{a0}word\u{2003}\n\tend  ".to_owned(),
        "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜><think>".to_owned(),
        "｜DSML｜ calls>\n<｜DSML｜ parameter name=\"x\">雪".to_owned(),
        "<think></think>｜DSML｜<dsml:</dsml:".to_owned(),
        "ordinary<think>reason</think>text｜DSML｜ <dsml:invoke>雪</dsml:invoke>end".to_owned(),
    ];
    texts.extend(added_token_prompts(&tokenizer));
    for text in &texts {
        let mut expected = official
            .encode(text.as_str(), false)
            .unwrap()
            .get_ids()
            .to_vec();
        if !text.starts_with("<｜begin▁of▁sentence｜>") {
            expected.insert(0, 0);
        }
        let actual = tokenizer.encode(text).unwrap();
        assert_eq!(actual, expected, "{text:?}");
        assert_eq!(actual.iter().filter(|&&id| id == 0).count(), 1);
        assert_eq!(
            tokenizer.decode(&actual),
            if text.starts_with("<｜begin▁of▁sentence｜>") {
                text.clone()
            } else {
                format!("<｜begin▁of▁sentence｜>{text}")
            }
        );
        assert_eq!(
            tokenizer.decode(&actual),
            official.decode(&actual, false).unwrap()
        );
    }
    for marker in ["<think>", "</think>", "｜DSML｜", "<dsml:", "</dsml:"] {
        let id = tokenizer.token_id(marker).unwrap();
        assert_eq!(tokenizer.decode(&[id]), marker);
        assert!(!tokenizer.is_stop_token(id));
    }
    println!("Native smoke prompt IDs: {expected:?}");
    println!(
        "JoyAI oracle: {} adversarial/full prompts and 12 atomic markers match pinned official token IDs; vocab={}; GGUF types=1 for all five non-special AddedTokens",
        texts.len(),
        tokenizer.vocab_size()
    );
}
