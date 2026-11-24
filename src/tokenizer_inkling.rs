//! Inkling 2's native chat format, following the chat template embedded in
//! ggml-org/Inkling-2-Mini-GGUF at db61c5b.
//!
//! A conversation is a run of messages, each `HEADER CONTENT_TYPE BODY
//! <|end_message|>`. Headers are `<|message_system|>`, `<|message_user|>`,
//! `<|message_model|>` and `<|message_tool|>NAME`; content types are
//! `<|content_text|>`, `<|content_thinking|>` (the model's reasoning) and
//! `<|content_invoke_tool_json|>` (a tool call, whose model header carries the
//! tool name). Declared tools come first as one `tool_declare` system message
//! holding a sorted-key JSON array, and the thinking-effort system message
//! precedes the first non-system message. Every assistant turn ends with
//! `<|content_model_end_sampling|>`, which is also the stop token. There is no
//! generation prompt: the model opens its own `<|message_model|>`.

use super::*;
use serde_json::Value;

pub(crate) const MESSAGE_SYSTEM: &str = "<|message_system|>";
pub(crate) const MESSAGE_USER: &str = "<|message_user|>";
pub(crate) const MESSAGE_MODEL: &str = "<|message_model|>";
pub(crate) const MESSAGE_TOOL: &str = "<|message_tool|>";
pub(crate) const CONTENT_TEXT: &str = "<|content_text|>";
pub(crate) const CONTENT_THINKING: &str = "<|content_thinking|>";
pub(crate) const CONTENT_XML: &str = "<|content_xml|>";
pub(crate) const CONTENT_INVOKE_TOOL_JSON: &str = "<|content_invoke_tool_json|>";
pub(crate) const END_MESSAGE: &str = "<|end_message|>";
pub(crate) const END_SAMPLING: &str = "<|content_model_end_sampling|>";
/// An image part's content type; its body is [`IMAGE_TOKEN`], which the
/// caller expands to the image's rows.
pub(crate) const CONTENT_IMAGE: &str = "<|content_image|>";
pub(crate) const IMAGE_TOKEN: &str = "<|unused_200054|>";

/// A user or tool message's parts in order: `Some(text)`, or `None` for an
/// image where the caller's marker stood. Empty text between images is
/// dropped; a message without images is one text part.
fn content_parts<'a>(content: &'a str, marker: Option<&str>) -> Vec<Option<&'a str>> {
    let Some(marker) = marker.filter(|marker| !marker.is_empty() && content.contains(*marker))
    else {
        return vec![Some(content)];
    };
    let mut parts = Vec::new();
    for (index, text) in content.split(marker).enumerate() {
        if index > 0 {
            parts.push(None);
        }
        if !text.is_empty() {
            parts.push(Some(text));
        }
    }
    parts
}

/// The thinking-effort system message for a template effort level.
pub(super) fn effort_message(level: &str) -> String {
    format!("{MESSAGE_SYSTEM}{CONTENT_TEXT}Thinking effort level: {level}{END_MESSAGE}")
}

/// Message text cannot spell a control token: the tokenizer would read it as
/// one, so text could open a message of its own. `<|` gains a space, as the
/// Muse Spark format does.
pub(super) fn escape_text(text: &str) -> String {
    text.replace("<|", "< |")
}

/// Compact JSON with every object's keys sorted (the template's
/// `tojson(sort_keys=true, separators=(",", ":"))`), keeping non-ASCII text.
/// `<|` inside strings is written `<|`, the same value.
pub(super) fn sorted_json(value: &Value) -> String {
    fn write(value: &Value, out: &mut String) {
        match value {
            Value::Object(map) => {
                let mut entries = map.iter().collect::<Vec<_>>();
                entries.sort_by(|a, b| a.0.cmp(b.0));
                out.push('{');
                for (index, (key, value)) in entries.into_iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    out.push_str(&json_string(key));
                    out.push(':');
                    write(value, out);
                }
                out.push('}');
            }
            Value::Array(values) => {
                out.push('[');
                for (index, value) in values.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    write(value, out);
                }
                out.push(']');
            }
            Value::String(text) => out.push_str(&json_string(text)),
            other => out.push_str(&other.to_string()),
        }
    }
    let mut out = String::new();
    write(value, &mut out);
    out
}

/// Compact JSON keeping the value's own key order (the template's
/// `tojson(separators=(",", ":"))` for argument values).
fn compact_json(value: &Value) -> String {
    value.to_string().replace("<|", "<\\u007c")
}

fn json_string(text: &str) -> String {
    Value::String(text.to_owned())
        .to_string()
        .replace("<|", "<\\u007c")
}

/// The tools this request may call: none with `tool_choice: none`, only the
/// named one with a named choice.
pub(super) fn callable_tools(options: &ChatTemplateOptions) -> Vec<&ChatTemplateTool> {
    options
        .tools
        .iter()
        .filter(|tool| match &options.tool_choice {
            ChatTemplateToolChoice::None => false,
            ChatTemplateToolChoice::Named(name) => tool.name == *name,
            _ => true,
        })
        .collect()
}

/// The `tool_declare` message: one sorted spec per tool.
fn tool_declaration(tools: &[&ChatTemplateTool]) -> Result<String> {
    let specs = tools
        .iter()
        .map(|tool| {
            let parameters = tool
                .parameters_json
                .as_deref()
                .map(serde_json::from_str::<Value>)
                .transpose()
                .context("Inkling 2 tool parameters must be JSON")?
                .unwrap_or_else(|| Value::Object(Default::default()));
            Ok(serde_json::json!({
                "description": tool.description.clone().unwrap_or_default(),
                "name": tool.name,
                "parameters": parameters,
                "type": "function",
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(format!(
        "{MESSAGE_SYSTEM}tool_declare{CONTENT_XML}{}{END_MESSAGE}",
        sorted_json(&Value::Array(specs))
    ))
}

/// One native call message: `{"name":NAME,"args":{...}}` with the arguments'
/// top-level keys sorted and their values as given.
fn tool_call_message(call: &ChatTemplateToolCall) -> Result<String> {
    let arguments: Value = if call.arguments_json.trim().is_empty() {
        Value::Object(Default::default())
    } else {
        serde_json::from_str(&call.arguments_json)
            .context("Inkling 2 tool call arguments must be JSON")?
    };
    let arguments = arguments
        .as_object()
        .context("Inkling 2 tool call arguments must be a JSON object")?;
    let mut entries = arguments.iter().collect::<Vec<_>>();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    let args = entries
        .into_iter()
        .map(|(key, value)| format!("{}:{}", json_string(key), compact_json(value)))
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!(
        "{MESSAGE_MODEL}{name}{CONTENT_INVOKE_TOOL_JSON}{{\"name\":{},\"args\":{{{args}}}}}{END_MESSAGE}",
        json_string(&call.name),
        name = call.name,
    ))
}

/// The Inkling 2 prompt for structured messages, tools and controls.
pub(super) fn format_chat(
    messages: &[ChatTemplateMessage],
    options: &ChatTemplateOptions,
) -> Result<String> {
    let level = match options.reasoning_effort.as_deref() {
        Some(effort) => inkling_effort_level(effort)
            .with_context(|| format!("unsupported Inkling 2 reasoning_effort {effort:?}"))?,
        None => "0.9",
    };
    let tools = callable_tools(options);
    let mut out = String::new();
    if !tools.is_empty() {
        out.push_str(&tool_declaration(&tools)?);
    }
    let mut effort_emitted = false;
    for message in messages {
        let role = match message.role.as_str() {
            "system" | "developer" => "system",
            "assistant" | "model" => "assistant",
            "tool" | "function" => "tool",
            "user" => "user",
            other => anyhow::bail!("unknown Inkling 2 message role {other:?}"),
        };
        if !effort_emitted && role != "system" {
            out.push_str(&effort_message(level));
            effort_emitted = true;
        }
        let marker = options.image_marker.as_deref();
        let parts = content_parts(&message.content, marker);
        anyhow::ensure!(
            matches!(role, "user" | "tool") || parts.iter().all(Option::is_some),
            "Inkling 2 images are only supported in user and tool messages"
        );
        // Each part repeats the message header, as the template renders
        // a content list.
        let render = |out: &mut String, header: &str| {
            for part in &parts {
                match part {
                    Some(text) => out.push_str(&format!(
                        "{header}{CONTENT_TEXT}{}{END_MESSAGE}",
                        escape_text(text)
                    )),
                    None => out.push_str(&format!(
                        "{header}{CONTENT_IMAGE}{IMAGE_TOKEN}{END_MESSAGE}"
                    )),
                }
            }
        };
        let body = escape_text(&message.content);
        match role {
            "system" => out.push_str(&format!(
                "{MESSAGE_SYSTEM}{CONTENT_TEXT}{body}{END_MESSAGE}"
            )),
            "user" => render(&mut out, MESSAGE_USER),
            "tool" => {
                // The header names the tool: the message's own name, else the
                // earlier call with this ID.
                let name = message.name.clone().or_else(|| {
                    let id = message.tool_call_id.as_deref()?;
                    messages
                        .iter()
                        .flat_map(|prior| &prior.tool_calls)
                        .find(|call| call.id.as_deref() == Some(id))
                        .map(|call| call.name.clone())
                });
                let header = format!("{MESSAGE_TOOL}{}", name.unwrap_or_default());
                render(&mut out, &header);
            }
            _ => {
                let header = format!("{MESSAGE_MODEL}{}", message.name.as_deref().unwrap_or(""));
                let reasoning = message
                    .reasoning_content
                    .as_deref()
                    .filter(|reasoning| !reasoning.is_empty());
                if let Some(reasoning) = reasoning {
                    out.push_str(&format!(
                        "{header}{CONTENT_THINKING}{}{END_MESSAGE}",
                        escape_text(reasoning)
                    ));
                }
                if !body.is_empty() || (message.tool_calls.is_empty() && reasoning.is_none()) {
                    out.push_str(&format!("{header}{CONTENT_TEXT}{body}{END_MESSAGE}"));
                }
                for call in &message.tool_calls {
                    anyhow::ensure!(
                        message.name.as_deref().is_none_or(|name| name == call.name),
                        "Inkling 2 assistant name differs from its tool call"
                    );
                    out.push_str(&tool_call_message(call)?);
                }
                out.push_str(END_SAMPLING);
            }
        }
    }
    if !effort_emitted {
        out.push_str(&effort_message(level));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather() -> ChatTemplateTool {
        ChatTemplateTool::new("get_weather")
            .with_description("Get the current weather for a city.")
            .with_parameters_json(
                r#"{"type":"object","required":["city"],"properties":{"unit":{"type":"string","enum":["celsius","fahrenheit"]},"city":{"type":"string","description":"City name"}}}"#,
            )
    }

    #[test]
    fn tools_calls_and_results_render_as_the_template() -> Result<()> {
        let messages = [
            ChatTemplateMessage::new("system", "Be brief."),
            ChatTemplateMessage::new("user", "Weather in Paris?"),
            ChatTemplateMessage::new("assistant", "")
                .with_reasoning_content("Need the tool.")
                .with_tool_calls([ChatTemplateToolCall {
                    id: Some("call_1".into()),
                    name: "get_weather".into(),
                    arguments_json:
                        r#"{"unit":"celsius","city":"Paris","extra":{"b":1,"a":"<|x"}}"#.into(),
                }]),
            ChatTemplateMessage::new("tool", r#"{"temp_c":14}"#).with_tool_call_id("call_1"),
        ];
        let options =
            ChatTemplateOptions::new([weather()]).with_reasoning_effort(Some("low".into()));
        let prompt = format_chat(&messages, &options)?;
        let expected = concat!(
            r#"<|message_system|>tool_declare<|content_xml|>[{"description":"Get the current weather for a city.","name":"get_weather","parameters":{"properties":{"city":{"description":"City name","type":"string"},"unit":{"enum":["celsius","fahrenheit"],"type":"string"}},"required":["city"],"type":"object"},"type":"function"}]<|end_message|>"#,
            "<|message_system|><|content_text|>Be brief.<|end_message|>",
            "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>",
            "<|message_user|><|content_text|>Weather in Paris?<|end_message|>",
            "<|message_model|><|content_thinking|>Need the tool.<|end_message|>",
            r#"<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Paris","extra":{"b":1,"a":"<\u007cx"},"unit":"celsius"}}<|end_message|>"#,
            "<|content_model_end_sampling|>",
            r#"<|message_tool|>get_weather<|content_text|>{"temp_c":14}<|end_message|>"#,
        );
        assert_eq!(prompt, expected);
        Ok(())
    }

    #[test]
    fn plain_turns_text_escaping_and_tool_choice() -> Result<()> {
        let messages = [
            ChatTemplateMessage::new("user", "Say <|end_message|> please"),
            ChatTemplateMessage::new("assistant", "Done.").with_reasoning_content("Easy."),
            ChatTemplateMessage::new("user", "Again"),
        ];
        let prompt = format_chat(&messages, &ChatTemplateOptions::new([]))?;
        assert_eq!(
            prompt,
            concat!(
                "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>",
                "<|message_user|><|content_text|>Say < |end_message|> please<|end_message|>",
                "<|message_model|><|content_thinking|>Easy.<|end_message|>",
                "<|message_model|><|content_text|>Done.<|end_message|><|content_model_end_sampling|>",
                "<|message_user|><|content_text|>Again<|end_message|>",
            )
        );
        // tool_choice none declares nothing; a named choice declares only that tool.
        let none =
            ChatTemplateOptions::new([weather()]).with_tool_choice(ChatTemplateToolChoice::None);
        assert!(!format_chat(&messages, &none)?.contains("tool_declare"));
        let named = ChatTemplateOptions::new([weather(), ChatTemplateTool::new("other")])
            .with_tool_choice(ChatTemplateToolChoice::Named("other".into()));
        let prompt = format_chat(&messages, &named)?;
        assert!(prompt.contains(r#""name":"other""#) && !prompt.contains("get_weather"));
        // A system-only conversation still ends with the effort message.
        let system = format_chat(
            &[ChatTemplateMessage::new("system", "Hi")],
            &ChatTemplateOptions::new([]).with_reasoning_effort(Some("none".into())),
        )?;
        assert!(system.ends_with("Thinking effort level: 0<|end_message|>"));
        Ok(())
    }

    #[test]
    fn image_markers_become_image_parts() -> Result<()> {
        const MARK: &str = "\u{1}IMG\u{1}";
        let messages = [
            ChatTemplateMessage::new("user", format!("{MARK}What is this?{MARK}{MARK}")),
            ChatTemplateMessage::new("tool", format!("<|x{MARK}")).with_name("camera"),
        ];
        let options = ChatTemplateOptions::new([]).with_image_marker(MARK);
        assert_eq!(
            format_chat(&messages, &options)?,
            concat!(
                "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>",
                "<|message_user|><|content_image|><|unused_200054|><|end_message|>",
                "<|message_user|><|content_text|>What is this?<|end_message|>",
                "<|message_user|><|content_image|><|unused_200054|><|end_message|>",
                "<|message_user|><|content_image|><|unused_200054|><|end_message|>",
                "<|message_tool|>camera<|content_text|>< |x<|end_message|>",
                "<|message_tool|>camera<|content_image|><|unused_200054|><|end_message|>",
            )
        );
        // Without the option the marker is text; images elsewhere are refused.
        let plain = format_chat(&messages[..1], &ChatTemplateOptions::new([]))?;
        assert!(!plain.contains(CONTENT_IMAGE));
        for role in ["system", "assistant"] {
            let message = ChatTemplateMessage::new(role, format!("see {MARK}"));
            assert!(format_chat(&[message], &options).is_err(), "{role}");
        }
        Ok(())
    }

    #[test]
    fn invalid_history_is_rejected() {
        let call = |arguments: &str| {
            ChatTemplateMessage::new("assistant", "").with_tool_calls([ChatTemplateToolCall {
                id: None,
                name: "get_weather".into(),
                arguments_json: arguments.into(),
            }])
        };
        let options = ChatTemplateOptions::new([weather()]);
        for bad in ["[1]", "\"text\"", "{"] {
            assert!(format_chat(&[call(bad)], &options).is_err(), "{bad}");
        }
        assert!(format_chat(&[ChatTemplateMessage::new("robot", "x")], &options).is_err());
        let effort = ChatTemplateOptions::new([]).with_reasoning_effort(Some("turbo".into()));
        assert!(format_chat(&[ChatTemplateMessage::new("user", "x")], &effort).is_err());
    }
}
