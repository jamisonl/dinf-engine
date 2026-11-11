//! Text-only DeepSeek V4.1 1017 encoding and JoyAI pretokenization.
//!
//! Ported from deepseek-ai/DeepSeek-V4.1-Flash-1017 encoding/encoding.py at
//! 1bae0e1088993b9b82de08f59de5be5a34e42170. MIT license:
//! Copyright (c) 2023 DeepSeek
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in
//! all copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! SOFTWARE.

use super::*;
use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use std::fmt::{self, Write as _};
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{
    OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, SplitDelimiterBehavior,
};

pub(super) fn pretokenize(text: &str) -> Result<Vec<String>> {
    static SPLITS: LazyLock<Sequence> = LazyLock::new(|| {
        Sequence::new(
            [
                r"\p{N}{1,3}",
                r"[一-龥぀-ゟ゠-ヿ]+",
                r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##,
            ]
            .into_iter()
            .map(|pattern| {
                Split::new(
                    SplitPattern::Regex(pattern.into()),
                    SplitDelimiterBehavior::Isolated,
                    false,
                )
                .expect("pinned JoyAI split")
                .into()
            })
            .collect(),
        )
    });
    let mut input = PreTokenizedString::from(text);
    SPLITS
        .pre_tokenize(&mut input)
        .map_err(|error| anyhow::anyhow!("JoyAI split: {error}"))?;
    Ok(input
        .get_splits(OffsetReferential::Original, OffsetType::Byte)
        .into_iter()
        .map(|(piece, _, _)| piece.to_owned())
        .collect())
}

// Preserve input object order independently of serde_json's map feature flags.
#[derive(Debug)]
enum Json {
    Scalar(serde_json::Value),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}
impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct Ordered;
        impl<'de> Visitor<'de> for Ordered {
            type Value = Json;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("JSON")
            }
            fn visit_bool<E: serde::de::Error>(self, v: bool) -> std::result::Result<Json, E> {
                Ok(Json::Scalar(v.into()))
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> std::result::Result<Json, E> {
                Ok(Json::Scalar(v.into()))
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> std::result::Result<Json, E> {
                Ok(Json::Scalar(v.into()))
            }
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> std::result::Result<Json, E> {
                Ok(Json::Scalar(serde_json::json!(v)))
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> std::result::Result<Json, E> {
                Ok(Json::Scalar(v.into()))
            }
            fn visit_unit<E: serde::de::Error>(self) -> std::result::Result<Json, E> {
                Ok(Json::Scalar(serde_json::Value::Null))
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Json, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = seq.next_element()? {
                    values.push(value);
                }
                Ok(Json::Array(values))
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Json, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = map.next_entry()? {
                    values.push(value);
                }
                Ok(Json::Object(values))
            }
        }
        deserializer.deserialize_any(Ordered)
    }
}
impl Json {
    fn render(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Self::Scalar(value) => write!(out, "{value}").unwrap(),
            Self::Array(values) => {
                out.push('[');
                for (index, value) in values.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    value.write(out);
                }
                out.push(']');
            }
            Self::Object(values) => {
                out.push('{');
                for (index, (key, value)) in values.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&serde_json::to_string(key).unwrap());
                    out.push_str(": ");
                    value.write(out);
                }
                out.push('}');
            }
        }
    }
}

pub(super) fn valid_name(name: &str) -> bool {
    let mut parts = name.split("::");
    let valid = |part: &str| {
        !part.is_empty()
            && part
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    };
    valid(parts.next().unwrap_or_default())
        && parts.next().is_none_or(valid)
        && parts.next().is_none()
}

const TOOLS: &str = r##"## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜ calls>" block like the following:

<｜DSML｜ calls>
<｜DSML｜ invoke name="$TOOL_NAME">
<｜DSML｜ parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜ parameter>
...
</｜DSML｜ invoke>
<｜DSML｜ invoke name="$TOOL_NAME2">
...
</｜DSML｜ invoke>
</｜DSML｜ calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

"##;

fn render_tools(tools: &[ChatTemplateTool]) -> Result<String> {
    let mut out = TOOLS.to_owned();
    for (index, tool) in tools.iter().enumerate() {
        if index > 0 {
            out.push('\n');
        }
        out.push_str("{\"name\": ");
        out.push_str(&serde_json::to_string(&tool.name)?);
        if let Some(description) = &tool.description {
            out.push_str(", \"description\": ");
            out.push_str(&serde_json::to_string(description)?);
        }
        if let Some(schema) = &tool.parameters_json {
            let schema: Json = serde_json::from_str(schema)?;
            out.push_str(", \"parameters\": ");
            schema.write(&mut out);
        }
        out.push('}');
    }
    out.push_str("\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n");
    Ok(out)
}

fn render_call(call: &ChatTemplateToolCall) -> Result<String> {
    anyhow::ensure!(valid_name(&call.name), "invalid DSML tool name");
    let mut arguments: Json = serde_json::from_str(&call.arguments_json)
        .unwrap_or_else(|_| Json::Scalar(call.arguments_json.clone().into()));
    if let Json::Scalar(serde_json::Value::String(value)) = &arguments
        && let Ok(parsed) = serde_json::from_str(value)
    {
        arguments = parsed;
    }
    let arguments = match arguments {
        Json::Object(arguments) => arguments,
        _ => vec![(
            "arguments".into(),
            Json::Scalar(call.arguments_json.clone().into()),
        )],
    };
    let mut out = format!("<｜DSML｜ invoke name=\"{}\">\n", call.name);
    let mut keys = std::collections::HashSet::with_capacity(arguments.len());
    for (index, (key, value)) in arguments.iter().enumerate() {
        anyhow::ensure!(
            !key.is_empty()
                && !key.contains(['<', '>', '"'])
                && !key.chars().any(char::is_control)
                && keys.insert(key),
            "invalid or duplicate DSML parameter name"
        );
        if index > 0 {
            out.push('\n');
        }
        let rendered;
        let (string, value) = match value {
            Json::Scalar(serde_json::Value::String(value)) => (true, value.as_str()),
            value => {
                rendered = value.render();
                (false, rendered.as_str())
            }
        };
        // Only this raw-string delimiter is reserved; HTML and other dialects are data.
        anyhow::ensure!(
            !string || !value.contains("</｜DSML｜ parameter>"),
            "DSML string contains its closing delimiter"
        );
        write!(
            out,
            "<｜DSML｜ parameter name=\"{key}\" string=\"{string}\">{value}</｜DSML｜ parameter>"
        )
        .unwrap();
    }
    out.push_str("\n</｜DSML｜ invoke>");
    Ok(out)
}

struct Turn<'a> {
    message: &'a ChatTemplateMessage,
    blocks: Vec<&'a ChatTemplateMessage>,
    source_index: usize,
}

pub(super) fn format_chat(
    messages: &[ChatTemplateMessage],
    options: &ChatTemplateOptions,
) -> Result<String> {
    // Images are typed request content and rejected at native admission. Tag
    // spellings in text are ordinary data, not evidence of an image payload.
    validate_deepseek41_tool_schemas(options)?;
    let decorations = HashMap::new();
    format_prepared(messages, options, &decorations)
}

/// The prepared renderer also permits preprocessed official image placeholders.
#[derive(Default)]
struct Decoration {
    tools: Vec<ChatTemplateTool>,
    task: Option<String>,
}

fn format_prepared(
    messages: &[ChatTemplateMessage],
    options: &ChatTemplateOptions,
    decorations: &HashMap<usize, Decoration>,
) -> Result<String> {
    let thinking = options.reasoning_effort.as_deref() != Some("none");
    let budget = match options.reasoning_effort.as_deref() {
        None | Some("high") | Some("none") => 75,
        Some("low") => 50,
        Some("max") => 100,
        Some(effort) => anyhow::bail!("unsupported DeepSeek V4.1 effort {effort:?}"),
    };
    let tools = match &options.tool_choice {
        ChatTemplateToolChoice::None => &[],
        ChatTemplateToolChoice::Named(name) => options
            .tools
            .iter()
            .find(|tool| tool.name == *name)
            .map(std::slice::from_ref)
            .unwrap_or_default(),
        _ => options.tools.as_slice(),
    };
    let mut turns: Vec<Turn<'_>> = Vec::new();
    let initial_system = (!tools.is_empty()
        && !messages
            .first()
            .is_some_and(|message| matches!(message.role.as_str(), "system" | "developer")))
    .then(|| ChatTemplateMessage::new("system", ""));
    if let Some(message) = &initial_system {
        turns.push(Turn {
            message,
            blocks: Vec::new(),
            source_index: usize::MAX,
        });
    }
    for (source_index, message) in messages.iter().enumerate() {
        let user = matches!(message.role.as_str(), "user" | "tool" | "function");
        if user
            && turns.last().is_some_and(|turn| {
                !turn.blocks.is_empty()
                    && decorations
                        .get(&turn.source_index)
                        .is_none_or(|decoration| decoration.task.is_none())
            })
        {
            turns.last_mut().unwrap().blocks.push(message);
        } else {
            turns.push(Turn {
                message,
                blocks: if user { vec![message] } else { Vec::new() },
                source_index,
            });
        }
    }
    let last_real_user = turns.iter().rposition(|turn| {
        turn.blocks.iter().any(|message| message.role == "user")
            || (matches!(turn.message.role.as_str(), "system" | "developer")
                && turn.source_index > 0
                && turn.source_index != usize::MAX)
    });
    let retain = !options.tools.is_empty()
        || decorations
            .values()
            .any(|decoration| !decoration.tools.is_empty());
    let mut out = String::from("<｜begin▁of▁sentence｜>");
    let mut call_order: HashMap<&str, usize> = HashMap::new();
    for (index, turn) in turns.iter().enumerate() {
        let message = turn.message;
        let system = matches!(message.role.as_str(), "system" | "developer");
        if index == 0 && (thinking || system) {
            out.push_str("<｜System｜>");
        }
        if index == 0 && thinking {
            write!(out, "Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n").unwrap();
        }
        if system {
            if index > 0 {
                out.push_str("<｜System｜>");
            }
            out.push_str(&message.content);
            let tools = decorations
                .get(&turn.source_index)
                .map(|decoration| decoration.tools.as_slice())
                .filter(|tools| !tools.is_empty())
                .unwrap_or(if index == 0 { tools } else { &[] });
            if !tools.is_empty() {
                out.push_str("\n\n");
                out.push_str(&render_tools(tools)?);
            }
        } else if !turn.blocks.is_empty() {
            out.push_str("<｜User｜>");
            let mut results = turn
                .blocks
                .iter()
                .copied()
                .filter(|message| message.role != "user")
                .collect::<Vec<_>>();
            results.sort_by_key(|message| {
                message
                    .tool_call_id
                    .as_deref()
                    .and_then(|id| call_order.get(id))
                    .copied()
                    .unwrap_or(0)
            });
            let mut results = results.into_iter();
            for (block_index, block) in turn.blocks.iter().enumerate() {
                if block_index > 0 {
                    out.push_str("\n\n");
                }
                if block.role == "user" {
                    out.push_str(&block.content);
                } else {
                    let result = results.next().unwrap();
                    out.push_str("<tool_result>");
                    out.push_str(&result.content);
                    out.push_str("</tool_result>");
                }
            }
        } else if message.role == "assistant" {
            let previous_task = index > 0
                && decorations
                    .get(&turns[index - 1].source_index)
                    .is_some_and(|decoration| decoration.task.is_some());
            if thinking
                && !previous_task
                && (retain || last_real_user.is_none_or(|last| index > last))
            {
                out.push_str(message.reasoning_content.as_deref().unwrap_or_default());
                out.push_str("</think>");
            }
            out.push_str(&message.content);
            if !message.tool_calls.is_empty() {
                call_order.clear();
                out.push_str("\n\n<｜DSML｜ calls>\n");
                for (call_index, call) in message.tool_calls.iter().enumerate() {
                    if let Some(id) = call.id.as_deref() {
                        call_order.insert(id, call_index);
                    }
                    if call_index > 0 {
                        out.push('\n');
                    }
                    out.push_str(&render_call(call)?);
                }
                out.push_str("\n</｜DSML｜ calls>");
            }
            out.push_str("<｜end▁of▁sentence｜>");
        } else if message.role == "latest_reminder" {
            out.push_str("<｜latest_reminder｜>");
            out.push_str(&message.content);
        } else {
            anyhow::bail!("unsupported DeepSeek V4.1 role {:?}", message.role);
        }
        let next_assistant = turns.get(index + 1).is_none_or(|next| {
            matches!(next.message.role.as_str(), "assistant" | "latest_reminder")
        });
        if next_assistant
            && let Some(task) = decorations
                .get(&turn.source_index)
                .and_then(|decoration| decoration.task.as_deref())
        {
            anyhow::ensure!(
                matches!(
                    task,
                    "action" | "query" | "authority" | "domain" | "title" | "read_url"
                ),
                "invalid DeepSeek task"
            );
            if task == "action" {
                out.push_str("<｜Assistant｜>");
                out.push_str(if thinking { "<think>" } else { "</think>" });
            }
            write!(out, "<｜{task}｜>").unwrap();
            continue;
        }
        if next_assistant && (!turn.blocks.is_empty() || (system && index > 0)) {
            out.push_str("<｜Assistant｜>");
            out.push_str(
                if thinking && (retain || last_real_user.is_none_or(|last| index >= last)) {
                    "<think>"
                } else {
                    "</think>"
                },
            );
        }
    }
    anyhow::ensure!(
        out.len() <= MAX_CHAT_TEMPLATE_PROMPT_BYTES,
        "DeepSeek V4.1 prompt exceeds size limit"
    );
    Ok(out)
}

#[cfg(test)]
#[path = "tokenizer_deepseek41_tests.rs"]
mod tests;
