//! Generate complete tool envelopes without changing sampled text afterward.
//!
//! LLGuidance owns the byte grammar and token trie. Each generation owns a
//! matcher. A model may emit prose, reasoning, and Markdown; after an actual
//! tool opener, only the declared syntax and its matching close remain valid.

use std::fmt;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use llguidance::{Matcher, ParserFactory, api::TopLevelGrammar};
use serde_json::{Value, json};
use toktrie::{ApproximateTokEnv, SimpleVob, TokEnv, TokRxInfo, TokTrie};

use crate::tokenizer::{
    ChatTemplateMessage, ChatTemplateOptions, ChatTemplateTool, ChatTemplateToolChoice,
    GgufTokenizer, ToolFormat,
};

/// A compiled tool grammar shared by requests with the same prompt policy.
/// Its prototype never advances; each sampling history gets its own matcher.
pub struct ToolConstraint {
    prototype: Mutex<Matcher>,
    vocab_size: usize,
}

impl fmt::Debug for ToolConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolConstraint")
            .field("vocab_size", &self.vocab_size)
            .finish_non_exhaustive()
    }
}

impl ToolConstraint {
    /// Build a constraint for the exact syntax selected by prompt formatting.
    /// An empty or disabled tool set needs no tool matcher.
    pub fn for_chat(
        tokenizer: &GgufTokenizer,
        messages: &[ChatTemplateMessage],
        options: &ChatTemplateOptions,
    ) -> Result<Option<Arc<Self>>> {
        if options.tools.is_empty() || matches!(options.tool_choice, ChatTemplateToolChoice::None) {
            return Ok(None);
        }
        let format = tokenizer.selected_tool_format(messages, options);
        let prompt = tokenizer.format_chat_with_tool_options(messages, options)?;
        let implicit_reasoning = if prompt.ends_with("<think>\n") || prompt.ends_with("<think>") {
            Some("</think>")
        } else if prompt.ends_with("<|channel>thought\n") {
            Some("<channel|>")
        } else {
            None
        };
        let grammar = Self::grammar(format, options, implicit_reasoning)?;
        let factory = tokenizer.tool_parser_factory.get_or_init(|| {
            Self::build_factory(tokenizer)
                .map(Arc::new)
                .map_err(|error| format!("{error:#}"))
        });
        let factory = factory
            .as_ref()
            .map_err(|error| anyhow::anyhow!(error.clone()))?;
        let matcher = Matcher::new(factory.create_parser(TopLevelGrammar::from_lark(grammar)));
        anyhow::ensure!(
            !matcher.is_error(),
            "tool grammar: {}",
            matcher.get_error().unwrap_or_default()
        );
        Ok(Some(Arc::new(Self {
            prototype: Mutex::new(matcher),
            vocab_size: tokenizer.vocab_size(),
        })))
    }

    fn build_factory(tokenizer: &GgufTokenizer) -> Result<ParserFactory> {
        let words = (0..tokenizer.vocab_size())
            .map(|id| {
                let mut bytes = tokenizer.decode_token_bytes(id as u32);
                if tokenizer.is_stop_token(id as u32) {
                    // EOS has printable source text in GGUF vocabularies. It
                    // must not match that text inside a JSON string or code.
                    bytes.insert(0, TokTrie::SPECIAL_TOKEN_MARKER);
                }
                bytes
            })
            .collect::<Vec<_>>();
        let info = TokRxInfo::new(u32::try_from(words.len())?, tokenizer.eos_id);
        let trie = TokTrie::from(&info, &words).with_eos_tokens(tokenizer.stop_token_ids());
        // We do not retokenize, splice, or fast-forward. An approximate env
        // uses exactly these decoded bytes while the model samples every ID.
        let env: TokEnv = Arc::new(ApproximateTokEnv::new(trie));
        let mut factory = ParserFactory::new_simple(&env)?;
        factory.quiet();
        Ok(factory)
    }

    /// Required `lookup` call over a synthetic vocabulary: token 0 is the
    /// whole call, token 1 is EOS, and every other token is unusable text.
    #[cfg(test)]
    pub(crate) fn test_required_lookup(vocab_size: usize) -> Result<Arc<Self>> {
        let options = ChatTemplateOptions::new([ChatTemplateTool::new("lookup")
            .with_parameters_json(
                r#"{"type":"object","properties":{},"additionalProperties":false}"#,
            )])
        .with_tool_choice(ChatTemplateToolChoice::Required);
        let grammar = ToolConstraint::grammar(ToolFormat::Json, &options, None)?;
        let mut words = vec![
            br#"<tool_call>{"name":"lookup","arguments":{}}</tool_call>"#.to_vec(),
            vec![TokTrie::SPECIAL_TOKEN_MARKER, b'E'],
        ];
        words.extend((2..vocab_size).map(|id| format!("unused{id}").into_bytes()));
        let trie = TokTrie::from(&TokRxInfo::new(vocab_size as u32, 1), &words);
        let env: TokEnv = Arc::new(ApproximateTokEnv::new(trie));
        let mut factory = ParserFactory::new_simple(&env)?;
        factory.quiet();
        let matcher = Matcher::new(factory.create_parser(TopLevelGrammar::from_lark(grammar)));
        anyhow::ensure!(!matcher.is_error(), "{:?}", matcher.get_error());
        Ok(Arc::new(Self {
            prototype: Mutex::new(matcher),
            vocab_size,
        }))
    }

    pub(crate) fn matcher(&self) -> Matcher {
        self.prototype
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .deep_clone()
    }

    pub(crate) fn validate_vocab_size(&self, vocab_size: usize) -> Result<()> {
        anyhow::ensure!(
            self.vocab_size == vocab_size,
            "tool constraint vocabulary has {} tokens, model vocabulary has {vocab_size}",
            self.vocab_size
        );
        Ok(())
    }

    /// Build the production grammar. Exposed for one-off, model-free checks.
    pub fn grammar(
        format: ToolFormat,
        options: &ChatTemplateOptions,
        implicit_reasoning: Option<&str>,
    ) -> Result<String> {
        if format == ToolFormat::DeepSeekV41 {
            crate::tokenizer::validate_deepseek41_tool_schemas(options)?;
        }
        let tools = options
            .tools
            .iter()
            .filter(|tool| match &options.tool_choice {
                ChatTemplateToolChoice::None if format == ToolFormat::DeepSeekV41 => false,
                ChatTemplateToolChoice::Named(name) => &tool.name == name,
                _ => true,
            })
            .collect::<Vec<_>>();
        anyhow::ensure!(!tools.is_empty(), "tool constraint has no callable tools");
        if format == ToolFormat::Atem {
            return atem_grammar(&tools, options);
        }
        let mut grammar = String::from("%llguidance {\"no_forcing\":true}\nWS: /[ \\t\\r\\n]*/\n");
        grammar.push_str(&excluding_terminal(
            "TEXT",
            &[
                "</tool_call",
                "<function=",
                "</function",
                "<parameter=",
                "</parameter",
                "<arg_key>",
                "</arg_key>",
                "<arg_value>",
                "</arg_value>",
                "</｜DSML｜",
            ],
        ));
        let mut items = vec![
            "call".to_owned(),
            "thought".to_owned(),
            "gemma_thought".to_owned(),
        ];
        grammar.push_str("thought_head[lazy]: TEXT \"<think>\"\nthought: thought_head thought_tail\nthought_tail[lazy]: THOUGHT_TEXT \"</think>\"\n");
        grammar.push_str("gemma_head[lazy]: TEXT \"<|channel>thought\\n\"\ngemma_thought: gemma_head gemma_tail\ngemma_tail[lazy]: GEMMA_TEXT \"<channel|>\"\n");
        grammar.push_str(&excluding_terminal("THOUGHT_TEXT", &["</think>"]));
        grammar.push_str(&excluding_terminal("GEMMA_TEXT", &["<channel|>"]));
        // Trigger once, before choosing the Markdown run length. Independent
        // lazy heads can let a later tool opener win inside a fenced example.
        grammar.push_str("code_tick_head[lazy]: TEXT \"`\"\n");
        let mut tick_codes = Vec::new();
        for (kind, low, consumed, branches) in [('`', 1usize, 1usize, &mut tick_codes)] {
            for count in low..=16 {
                let marker = kind.to_string().repeat(count);
                let id = format!(
                    "code_{}_{}",
                    if kind == '`' { "tick" } else { "tilde" },
                    count
                );
                let body = id.to_ascii_uppercase();
                branches.push(id.clone());
                grammar.push_str(&format!(
                    "{id}: {} /[^{kind}]/ {id}_tail\n{id}_tail[lazy]: {body} {}\n",
                    literal(&kind.to_string().repeat(count - consumed))?,
                    literal(&marker)?
                ));
                grammar.push_str(&excluding_terminal(&body, &[&marker]));
            }
        }
        items.push("code_tick".into());
        grammar.push_str(&format!(
            "code_tick: code_tick_head ({})\n",
            tick_codes.join(" | ")
        ));
        let prose_items = items
            .iter()
            .filter(|item| item.as_str() != "call")
            .cloned()
            .collect::<Vec<_>>()
            .join(" | ");
        // Once a real call closes, end the turn. A parallel request permits a
        // bounded run of calls, separated only by framing whitespace.
        let dsml = matches!(format, ToolFormat::DeepSeek | ToolFormat::DeepSeekV41);
        let opener = if dsml { "<｜DSML｜" } else { "<tool_call" };
        let calls = if options.parallel_tool_calls && !dsml {
            format!("call_body (WS {} call_body){{0,15}}", literal(opener)?)
        } else {
            "call_body".to_owned()
        };
        let end = if matches!(
            options.tool_choice,
            ChatTemplateToolChoice::Required | ChatTemplateToolChoice::Named(_)
        ) {
            format!("call_head {calls} WS")
        } else {
            format!("(TEXT | call_head {calls} WS)")
        };
        let rest = format!("({prose_items})* {end}");
        if let Some(close) = implicit_reasoning {
            grammar.push_str(&excluding_terminal("INITIAL_REASONING", &[close]));
            let unfinished = if matches!(
                options.tool_choice,
                ChatTemplateToolChoice::Required | ChatTemplateToolChoice::Named(_)
            ) {
                ""
            } else {
                "INITIAL_REASONING | "
            };
            grammar.push_str(&format!("initial_closed[lazy]: INITIAL_REASONING {}\nstart: {unfinished}initial_closed rest\nrest: {rest}\n", literal(close)?));
        } else {
            grammar.push_str(&format!("start: {rest}\n"));
        }
        match format {
            ToolFormat::Json => {
                let alternatives = tools.iter().map(|tool| Ok(json!({
                    "type":"object",
                    "properties": {"name":{"const":tool.name}, "arguments":{"allOf":[{"type":"object"}, parameter_schema(tool)?]}},
                    "required":["name","arguments"],
                    "additionalProperties":false
                }))).collect::<Result<Vec<_>>>()?;
                grammar.push_str("call_head[lazy]: TEXT \"<tool_call\"\n");
                grammar.push_str(&format!(
                    "call_body: \">\" WS %json {} WS \"</tool_call>\"\n",
                    serde_json::to_string(&json!({"anyOf":alternatives}))?
                ));
            }
            ToolFormat::Qwen | ToolFormat::Glm | ToolFormat::DeepSeek | ToolFormat::DeepSeekV41 => {
                grammar.push_str(&native_grammar(
                    format,
                    &tools,
                    options.parallel_tool_calls,
                )?);
            }
            ToolFormat::Atem => unreachable!("recipient grammar handled above"),
        }
        Ok(grammar)
    }
}

/// Advance only after the sampler picks an allowed token. No bytes get added.
pub(crate) fn commit(matcher: &mut Matcher, token: u32) -> Result<()> {
    // LLGuidance returns an EOS-only mask after normal completion, but cannot
    // consume another token once stopped. The caller may still need EOS to
    // meet min_tokens or may handle stop IDs outside the sampler.
    if matcher.is_stopped() {
        let allowed = mask(matcher)?;
        anyhow::ensure!(
            (token as usize) < allowed.len() && allowed.is_allowed(token),
            "completed tool grammar rejected sampled token"
        );
        return Ok(());
    }
    matcher
        .consume_token(token)
        .context("tool grammar rejected sampled token")
}

/// Compute the allowed IDs before temperature, top-k, and top-p filtering.
pub(crate) fn mask(matcher: &mut Matcher) -> Result<SimpleVob> {
    matcher
        .compute_mask_or_eos()
        .context("tool grammar could not advance")
}

fn literal(text: &str) -> Result<String> {
    Ok(serde_json::to_string(text)?)
}

fn excluding_terminal(name: &str, markers: &[&str]) -> String {
    let markers = markers
        .iter()
        .map(|marker| regex::escape(marker).replace('/', "\\/"))
        .collect::<Vec<_>>()
        .join("|");
    format!("{name}: /(?s:.*)/ & ~/(?s:.*)(?:{markers})(?s:.*)/\n")
}

fn native_grammar(
    format: ToolFormat,
    tools: &[&ChatTemplateTool],
    parallel: bool,
) -> Result<String> {
    let mut grammar = String::new();
    let dsml_space = if format == ToolFormat::DeepSeekV41 {
        " "
    } else {
        ""
    };
    let dsml_parameter_end = format!("</｜DSML｜{dsml_space}parameter>");
    let reserved = [
        "<tool_call",
        "</tool_call",
        "<function=",
        "</function",
        "<parameter=",
        "</parameter",
        "<arg_key>",
        "</arg_key>",
        "<arg_value>",
        "</arg_value>",
        "<｜DSML｜",
        "</｜DSML｜",
    ];
    // String parameters carry raw UTF-8 text. Reserved native controls cannot
    // appear as file text in that format. The prompt switches to JSON when
    // history needs such strings. Nested JSON strings may escape '<' as \u003c.
    if format == ToolFormat::Atem {
        grammar.push_str(&excluding_terminal(
            "RAW_NATIVE",
            &["</atem:parameter>", "<|start|>", "<|eom|>", "<|eot|>"],
        ));
    } else if format == ToolFormat::DeepSeekV41 {
        grammar.push_str(&excluding_terminal("RAW_NATIVE", &[&dsml_parameter_end]));
    } else {
        grammar.push_str(&excluding_terminal("RAW_NATIVE", &reserved));
    }
    grammar.push_str("JSON_STRING: /\"(?:[^\"\\\\\\x00-\\x1f<]|\\\\(?:[\"\\\\\\/bfnrt]|u[0-9a-fA-F]{4}))*\"/\nJSON_NUMBER: /-?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?/\njson_value: JSON_STRING | JSON_NUMBER | \"true\" | \"false\" | \"null\" | json_object | json_array\njson_object: \"{\" WS (JSON_STRING WS \":\" WS json_value (WS \",\" WS JSON_STRING WS \":\" WS json_value)*)? WS \"}\"\njson_array: \"[\" WS (json_value (WS \",\" WS json_value)*)? WS \"]\"\n");
    let mut calls = Vec::new();
    for (index, tool) in tools.iter().enumerate() {
        anyhow::ensure!(
            !tool.name.contains(['<', '>', '\0', '"']),
            "tool name cannot enter native syntax"
        );
        let schema = parameter_schema(tool)?;
        let required = schema
            .get("required")
            .and_then(Value::as_array)
            .map(|values| values.iter().filter_map(Value::as_str).collect::<Vec<_>>())
            .unwrap_or_default();
        let mut params = Vec::new();
        if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
            for (param_index, (key, value)) in properties.iter().enumerate() {
                anyhow::ensure!(
                    !key.contains(['<', '>', '\0', '"'])
                        && (format != ToolFormat::DeepSeekV41
                            || (!key.is_empty() && !key.chars().any(char::is_control))),
                    "tool parameter cannot enter native syntax"
                );
                let id = format!("param_{index}_{param_index}");
                params.push(if required.contains(&key.as_str()) {
                    id.clone()
                } else {
                    format!("{id}?")
                });
                let string = value.get("type").is_some_and(|ty| ty == "string");
                let value_id = format!("VALUE_{index}_{param_index}");
                if string {
                    grammar.push_str(&raw_string_rule(&value_id, value)?);
                } else {
                    let mut value_schema = value.clone();
                    if let Some(value_schema) = value_schema.as_object_mut() {
                        for definitions in ["$defs", "definitions"] {
                            if let Some(definitions_value) = schema.get(definitions) {
                                value_schema
                                    .insert(definitions.to_owned(), definitions_value.clone());
                            }
                        }
                    }
                    grammar.push_str(&format!(
                        "{}: %json {}\n",
                        value_id.to_ascii_lowercase(),
                        serde_json::to_string(&value_schema)?
                    ));
                }
                let body = if string {
                    value_id.clone()
                } else {
                    value_id.to_ascii_lowercase()
                };
                if string {
                    let close = match format {
                        ToolFormat::Qwen => "</parameter>",
                        ToolFormat::Glm => "</arg_value>",
                        ToolFormat::DeepSeek => "</｜DSML｜parameter>",
                        ToolFormat::DeepSeekV41 => "</｜DSML｜ parameter>",
                        ToolFormat::Atem => "</atem:parameter>",
                        ToolFormat::Json => unreachable!(),
                    };
                    let tail = format!("{id}_tail");
                    let ws = if format == ToolFormat::Qwen { "WS" } else { "" };
                    grammar.push_str(&format!("{tail}[lazy]: {body} {ws} {}\n", literal(close)?));
                    match format {
                        ToolFormat::Qwen => grammar.push_str(&format!(
                            "{id}: {} WS {tail} WS\n",
                            literal(&format!("<parameter={key}>"))?
                        )),
                        ToolFormat::Glm => grammar.push_str(&format!(
                            "{id}: \"<arg_key>\" {} \"</arg_key>\" \"<arg_value>\" {tail}\n",
                            literal(key)?
                        )),
                        ToolFormat::DeepSeek | ToolFormat::DeepSeekV41 => {
                            grammar.push_str(&format!(
                                "{id}: {} {tail} WS\n",
                                literal(&format!(
                                    "<｜DSML｜{dsml_space}parameter name=\"{key}\" string=\"true\">"
                                ))?
                            ))
                        }
                        ToolFormat::Atem => grammar.push_str(&format!(
                            "{id}: {} {tail} WS\n",
                            literal(&format!("<atem:parameter name=\"{key}\">"))?
                        )),
                        ToolFormat::Json => unreachable!(),
                    }
                    continue;
                }
                match format {
                    ToolFormat::Qwen => grammar.push_str(&format!("{id}: {} WS {body} WS \"</parameter>\" WS\n", literal(&format!("<parameter={key}>"))?)),
                    ToolFormat::Glm => grammar.push_str(&format!("{id}: \"<arg_key>\" {} \"</arg_key>\" \"<arg_value>\" {body} \"</arg_value>\"\n", literal(key)?)),
                    ToolFormat::DeepSeek | ToolFormat::DeepSeekV41 => grammar.push_str(&format!("{id}: {} {body} {} WS\n", literal(&format!("<｜DSML｜{dsml_space}parameter name=\"{key}\" string=\"{string}\">"))?, literal(&dsml_parameter_end)?)),
                    ToolFormat::Atem => grammar.push_str(&format!("{id}: {} WS {body} WS \"</atem:parameter>\" WS\n", literal(&format!("<atem:parameter name=\"{key}\">"))?)),
                    ToolFormat::Json => unreachable!(),
                }
            }
        }
        let parameters = params.join(" ");
        let id = format!("native_{index}");
        calls.push(id.clone());
        match format {
            ToolFormat::Qwen => grammar.push_str(&format!(
                "{id}: {} WS {parameters} \"</function>\" WS \"</tool_call>\"\n",
                literal(&format!("<function={}>", tool.name))?
            )),
            ToolFormat::Glm => grammar.push_str(&format!(
                "{id}: {} {parameters} \"</tool_call>\"\n",
                literal(&tool.name)?
            )),
            ToolFormat::DeepSeek | ToolFormat::DeepSeekV41 => grammar.push_str(&format!(
                "{id}: {} WS {parameters} {} WS\n",
                literal(&format!(
                    "<｜DSML｜{dsml_space}invoke name=\"{}\">",
                    tool.name
                ))?,
                literal(&format!("</｜DSML｜{dsml_space}invoke>"))?
            )),
            ToolFormat::Atem => grammar.push_str(&format!(
                "{id}: {} WS {parameters} \"</atem:invoke>\" WS\n",
                literal(&format!("<atem:invoke name=\"{}\">", tool.name))?
            )),
            ToolFormat::Json => unreachable!(),
        }
    }
    if format == ToolFormat::Atem {
        return Ok(grammar);
    }
    if matches!(format, ToolFormat::DeepSeek | ToolFormat::DeepSeekV41) {
        let block = if format == ToolFormat::DeepSeekV41 {
            " calls"
        } else {
            "tool_calls"
        };
        grammar.push_str("call_head[lazy]: TEXT \"<｜DSML｜\"\n");
        grammar.push_str(&format!(
            "call_body: {} WS ({}){} {}\n",
            literal(&format!("{block}>"))?,
            calls.join(" | "),
            if parallel { "{1,16}" } else { "" },
            literal(&format!("</｜DSML｜{block}>"))?
        ));
    } else {
        grammar.push_str("call_head[lazy]: TEXT \"<tool_call\"\n");
        grammar.push_str(&format!("call_body: \">\" WS ({})\n", calls.join(" | ")));
    }
    Ok(grammar)
}

/// Muse addresses each message before its body. Reasoning and commentary can
/// contain tool examples; only an addressed tool message enters the call grammar.
fn atem_grammar(tools: &[&ChatTemplateTool], options: &ChatTemplateOptions) -> Result<String> {
    let mut grammar = String::from("%llguidance {\"no_forcing\":true}\nWS: /[ \\t\\r\\n]*/\n");
    grammar.push_str(&native_grammar(ToolFormat::Atem, tools, false)?);
    grammar.push_str(&excluding_terminal(
        "MUSE_BODY",
        &["<|start|>", "<|eom|>", "<|eot|>"],
    ));
    grammar.push_str("muse_next: \"<|eom|><|start|>assistant\"\n");
    grammar.push_str("muse_self: \" to=self<|message|>\" muse_body_tail \"<|start|>assistant\"\nmuse_commentary: \" to=commentary<|message|>\" muse_body_tail \"<|start|>assistant\"\nmuse_body_tail[lazy]: MUSE_BODY \"<|eom|>\"\nmuse_user: \" to=user<|message|>\" MUSE_BODY\n");
    let mut calls = Vec::new();
    for (index, tool) in tools.iter().enumerate() {
        let id = format!("muse_call_{index}");
        grammar.push_str(&format!(
            "{id}: ({} | \" to=tool<|message|>\") WS \"<atem:function_calls>\" WS native_{index} \"</atem:function_calls>\" WS\n",
            literal(&format!(" to={}<|message|>", tool.name))?
        ));
        calls.push(id);
    }
    grammar.push_str(&format!("muse_call: {}\n", calls.join(" | ")));
    grammar.push_str(if options.parallel_tool_calls {
        "muse_calls: muse_call (muse_next muse_call){0,15}\n"
    } else {
        "muse_calls: muse_call\n"
    });
    let end = if matches!(
        options.tool_choice,
        ChatTemplateToolChoice::Required | ChatTemplateToolChoice::Named(_)
    ) {
        "muse_calls"
    } else {
        "(muse_user | muse_calls)"
    };
    grammar.push_str(&format!("start: (muse_self | muse_commentary)* {end}\n"));
    Ok(grammar)
}

fn parameter_schema(tool: &ChatTemplateTool) -> Result<Value> {
    Ok(tool
        .parameters_json
        .as_deref()
        .map(serde_json::from_str::<Value>)
        .transpose()?
        .unwrap_or_else(|| json!({"type":"object"})))
}

fn raw_string_rule(id: &str, schema: &Value) -> Result<String> {
    if let Some(value) = schema.get("const").and_then(Value::as_str) {
        return Ok(format!("{id}: {}\n", literal(value)?));
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        let values = values
            .iter()
            .map(|value| {
                let value = value
                    .as_str()
                    .context("native string enum contains non-string value")?;
                literal(value)
            })
            .collect::<Result<Vec<_>>>()?;
        anyhow::ensure!(!values.is_empty(), "native string enum is empty");
        return Ok(format!("{id}: {}\n", values.join(" | ")));
    }
    let min = schema.get("minLength").and_then(Value::as_u64).unwrap_or(0);
    let max = schema.get("maxLength").and_then(Value::as_u64);
    anyhow::ensure!(
        max.is_none_or(|max| max >= min),
        "native string bounds are invalid"
    );
    anyhow::ensure!(
        schema.get("pattern").is_none(),
        "native patterned strings require JSON syntax"
    );
    let bounds = if let Some(max) = max {
        format!("{{{min},{max}}}")
    } else {
        format!("{{{min},}}")
    };
    Ok(format!("{id}: RAW_NATIVE & /(?s:.{bounds})/\n"))
}

#[cfg(test)]
mod sampling_tests {
    use super::*;
    use crate::continuous::GenerationSession;
    use crate::sampling::{SamplingHistory, SamplingParams, SamplingRng};
    use candle_core::{Device, Tensor};

    fn constraint(vocab_size: usize) -> Result<Arc<ToolConstraint>> {
        ToolConstraint::test_required_lookup(vocab_size)
    }

    #[test]
    fn constrained_sampling_rejects_a_different_logits_vocabulary() -> Result<()> {
        let params = SamplingParams {
            temperature: 0.0,
            tool_constraint: Some(constraint(2)?),
            ..Default::default()
        };
        for width in [64, 1, 3] {
            let history = SamplingHistory::new(&[], &params);
            let logits = Tensor::zeros(width, candle_core::DType::F32, &Device::Cpu)?;
            let error = history
                .sample(&logits, &params, &mut SamplingRng::seeded(42))
                .expect_err("a tool constraint must match the logits vocabulary");
            assert!(error.to_string().contains("vocabulary"), "{error:#}");
        }
        Ok(())
    }

    #[test]
    fn sampling_history_rejects_changed_tool_constraints() -> Result<()> {
        let logits = Tensor::new(&[0.0f32, 10.0], &Device::Cpu)?;
        for original in [None, Some(constraint(2)?)] {
            let params = SamplingParams {
                temperature: 0.0,
                frequency_penalty: 0.1,
                tool_constraint: original.clone(),
                ..Default::default()
            };
            let history = SamplingHistory::new(&[0], &params);
            let mut changed = params.clone();
            changed.tool_constraint = Some(constraint(2)?);
            assert!(
                history
                    .sample(&logits, &changed, &mut SamplingRng::seeded(42))
                    .is_err()
            );
            if original.is_some() {
                changed.tool_constraint = None;
                assert!(
                    history
                        .sample(&logits, &changed, &mut SamplingRng::seeded(42))
                        .is_err()
                );
            }
            // Rejected changes must leave the original grammar ready to sample.
            assert_eq!(
                history.sample(&logits, &params, &mut SamplingRng::seeded(42))?,
                if original.is_some() { 0 } else { 1 }
            );
        }
        Ok(())
    }

    #[test]
    fn constrained_generation_rejects_a_different_vocabulary_before_prefill() -> Result<()> {
        let mut stage = crate::session_state_tests::test_stage();
        let params = SamplingParams {
            max_tokens: 1,
            tool_constraint: Some(constraint(3)?),
            ..Default::default()
        };
        let error = stage
            .generate(vec![0], &params, &[1], 0, |_| true)
            .unwrap_err();
        assert!(error.to_string().contains("vocabulary"), "{error:#}");
        assert_eq!(stage.next_position(), 0);
        let mut session = GenerationSession::new(&stage, vec![0], params, vec![1]);
        let error = session.step(&mut stage).unwrap_err();
        assert!(error.to_string().contains("vocabulary"), "{error:#}");
        assert_eq!(session.next_position(&stage), 0);
        assert!(session.generated().is_empty());
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn constrained_session_honors_min_tokens_after_eos() -> Result<()> {
        let mut stage = crate::session_state_tests::test_stage();
        stage.vocab_size = 2; // Match the two-token grammar and test logits.
        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: 4,
            min_tokens: 4,
            tool_constraint: Some(constraint(2)?),
            ..Default::default()
        };
        let mut session = GenerationSession::new(&stage, vec![0], params, vec![1]);
        for token in [0, 1, 1, 1] {
            assert_eq!(session.step(&mut stage)?.token, token);
        }
        assert!(session.is_finished());
        assert_eq!(session.generated(), &[0, 1, 1, 1]);
        session.discard_cache(&mut stage);
        assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
        Ok(())
    }

    #[test]
    fn completed_constraint_accepts_only_its_eos_tokens() -> Result<()> {
        let mut matcher = constraint(2)?.matcher();
        assert!(mask(&mut matcher)?.is_allowed(0));
        commit(&mut matcher, 0)?;
        assert!(mask(&mut matcher)?.is_allowed(1));
        commit(&mut matcher, 1)?;
        assert!(matcher.is_stopped());
        assert!(commit(&mut matcher, 0).is_err());
        assert!(commit(&mut matcher, u32::MAX).is_err());
        commit(&mut matcher, 1)?;

        let mut invalid = constraint(2)?.matcher();
        assert!(commit(&mut invalid, 1).is_err());
        assert!(commit(&mut invalid, 1).is_err());
        Ok(())
    }
}

#[cfg(test)]
mod deepseek41_tests {
    use super::*;

    fn accepts(grammar: &str, text: &str) -> bool {
        let mut words = (0u8..=255).map(|byte| vec![byte]).collect::<Vec<_>>();
        words.push(vec![TokTrie::SPECIAL_TOKEN_MARKER, b'E']);
        let trie = TokTrie::from(&TokRxInfo::new(257, 256), &words);
        let env: TokEnv = Arc::new(ApproximateTokEnv::new(trie));
        let mut factory = ParserFactory::new_simple(&env).unwrap();
        factory.quiet();
        let mut matcher =
            Matcher::new(factory.create_parser(TopLevelGrammar::from_lark(grammar.to_owned())));
        assert!(!matcher.is_error(), "{:?}", matcher.get_error());
        for token in text.bytes().map(u32::from).chain([256]) {
            if matcher.compute_mask_or_eos().is_err() || matcher.consume_token(token).is_err() {
                return false;
            }
        }
        true
    }

    #[test]
    fn deepseek41_dsml_grammar_constrains_names_types_and_parallel_calls() {
        let tool = ChatTemplateTool::new("files::write").with_parameters_json(
            r#"{"type":"object","properties":{"content":{"type":"string"},"count":{"type":"integer"}},"required":["content","count"],"additionalProperties":false}"#,
        );
        let options =
            ChatTemplateOptions::new([tool]).with_tool_choice(ChatTemplateToolChoice::Required);
        let grammar =
            ToolConstraint::grammar(ToolFormat::DeepSeekV41, &options, Some("</think>")).unwrap();
        let invoke = "<｜DSML｜ invoke name=\"files::write\"><｜DSML｜ parameter name=\"content\" string=\"true\"> first\n<div>雪 && >;</div><｜DSML｜tool_calls></｜DSML｜ parameter><｜DSML｜ parameter name=\"count\" string=\"false\">2</｜DSML｜ parameter></｜DSML｜ invoke>";
        let single = format!("reason</think><｜DSML｜ calls>{invoke}</｜DSML｜ calls>");
        assert!(accepts(&grammar, &single));
        assert!(accepts(
            &grammar,
            &format!("reason</think><｜DSML｜ calls>{invoke}\n{invoke}</｜DSML｜ calls>")
        ));
        for invalid in [
            single.replace("files::write", "unknown"),
            single.replace(">2</", ">\"two\"</"),
            single.replace(" calls>", "tool_calls>"),
            single.replace(" parameter>", "parameter>"),
            single.trim_end_matches("</｜DSML｜ calls>").to_owned(),
        ] {
            assert!(!accepts(&grammar, &invalid), "{invalid}");
        }
        let serial = options.with_parallel_tool_calls(false);
        let grammar =
            ToolConstraint::grammar(ToolFormat::DeepSeekV41, &serial, Some("</think>")).unwrap();
        assert!(!accepts(
            &grammar,
            &format!("reason</think><｜DSML｜ calls>{invoke}{invoke}</｜DSML｜ calls>")
        ));
    }

    #[test]
    fn deepseek41_dsml_grammar_keeps_escaped_json_controls_as_argument_data() {
        let tool = ChatTemplateTool::new("lookup").with_parameters_json(
            r#"{"type":"object","properties":{"payload":{"type":"object","properties":{"items":{"type":"array","items":{"type":"string"}}},"required":["items"],"additionalProperties":false}},"required":["payload"],"additionalProperties":false}"#,
        );
        let options =
            ChatTemplateOptions::new([tool]).with_tool_choice(ChatTemplateToolChoice::Required);
        let grammar = ToolConstraint::grammar(ToolFormat::DeepSeekV41, &options, None).unwrap();
        let payload =
            json!({"items":["</｜DSML｜ calls>", "</｜DSML｜ parameter>", "雪\\\"", "line\nnext"]});
        let text = format!(
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"lookup\"><｜DSML｜ parameter name=\"payload\" string=\"false\">{payload}</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>"
        );
        assert!(accepts(&grammar, &text));
        let bad = "<｜DSML｜ calls><｜DSML｜ invoke name=\"lookup\"><｜DSML｜ parameter name=\"payload\" string=\"false\">{\"items\":[2]}</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>";
        assert!(!accepts(&grammar, bad));
    }

    #[test]
    fn deepseek41_dsml_required_and_named_keep_representable_string_and_json_values() {
        let raw = "/think /no_think\n<image>雪 && >;</image>\n</tool_call><｜DSML｜tool_calls>";
        let payload = json!({"items":["</｜DSML｜ parameter>", "line\nnext", "雪\\\""]});
        for keyword in ["const", "enum"] {
            let mut property = json!({"type":"string"});
            property[keyword] = if keyword == "const" {
                json!(raw)
            } else {
                json!([raw])
            };
            let schema = json!({
                "type":"object",
                "properties":{
                    "content":property,
                    "payload":{"type":"object","properties":{
                        "items":{"type":"array","items":{"type":"string"}}
                    },"required":["items"],"additionalProperties":false}
                },
                "required":["content","payload"],
                "additionalProperties":false
            });
            let tool =
                ChatTemplateTool::new("files::write").with_parameters_json(schema.to_string());
            for choice in [
                ChatTemplateToolChoice::Required,
                ChatTemplateToolChoice::Named("files::write".into()),
            ] {
                let options = ChatTemplateOptions::new([tool.clone()])
                    .with_tool_choice(choice)
                    .with_parallel_tool_calls(false);
                let grammar =
                    ToolConstraint::grammar(ToolFormat::DeepSeekV41, &options, None).unwrap();
                let invoke = format!(
                    "<｜DSML｜ invoke name=\"files::write\"><｜DSML｜ parameter name=\"content\" string=\"true\">{raw}</｜DSML｜ parameter><｜DSML｜ parameter name=\"payload\" string=\"false\">{payload}</｜DSML｜ parameter></｜DSML｜ invoke>"
                );
                let call = format!("<｜DSML｜ calls>{invoke}</｜DSML｜ calls>");
                assert!(accepts(&grammar, &call));
                assert!(!accepts(
                    &grammar,
                    &call.replace("files::write", "other::write")
                ));
                assert!(!accepts(
                    &grammar,
                    &format!("<｜DSML｜ calls>{invoke}{invoke}</｜DSML｜ calls>")
                ));
                assert!(!accepts(&grammar, "plain answer"));
            }
        }
    }

    #[test]
    fn deepseek41_dsml_open_empty_root_allows_only_schema_valid_empty_arguments() {
        for schema in [
            r#"{"type":"object"}"#,
            r#"{"type":"object","properties":{}}"#,
        ] {
            let options = ChatTemplateOptions::new([
                ChatTemplateTool::new("empty").with_parameters_json(schema)
            ])
            .with_tool_choice(ChatTemplateToolChoice::Required);
            let grammar = ToolConstraint::grammar(ToolFormat::DeepSeekV41, &options, None).unwrap();
            assert!(accepts(
                &grammar,
                "<｜DSML｜ calls><｜DSML｜ invoke name=\"empty\"></｜DSML｜ invoke></｜DSML｜ calls>"
            ));
        }
        let options = ChatTemplateOptions::new([ChatTemplateTool::new("nonempty")
            .with_parameters_json(r#"{"type":"object","minProperties":1}"#)])
        .with_tool_choice(ChatTemplateToolChoice::Required);
        assert!(ToolConstraint::grammar(ToolFormat::DeepSeekV41, &options, None).is_err());
    }
}
