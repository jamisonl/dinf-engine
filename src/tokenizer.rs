//! GGUF vocabulary, BPE, and model-family chat-template handling.
//!
//! [`GgufTokenizer`](crate::tokenizer::GgufTokenizer) builds a bounded tokenizer from GGUF metadata and keeps
//! model-specific prompt framing at the tokenizer boundary. The tool-aware
//! message types are lightweight engine representations of OpenAI-style chat
//! metadata; they do not perform tool execution.
//!
//! DeepSeek V4.1/JoyAI also matches `<think>`, `</think>`, `｜DSML｜`,
//! `<dsml:` and `</dsml:` atomically from the GGUF vocabulary. Upstream
//! defines these as non-special AddedTokens even though GGUF stores type 1;
//! atomic encoding does not make them stop tokens or discard their decoded text.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::io::{Seek, SeekFrom};
use std::path::Path;
use std::sync::LazyLock;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use unicode_normalization::UnicodeNormalization;

use crate::deepseek4_gguf::RawGgufContent;
use crate::gguf_limits::validate_gguf_header_budget;

#[path = "tokenizer_deepseek41.rs"]
mod deepseek41;

#[derive(Debug)]
struct MergeNode {
    piece: String,
    previous: Option<usize>,
    next: Option<usize>,
    generation: u32,
    alive: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MergeCandidate {
    /// Larger values are merged first.
    priority: u32,
    /// Original piece position. This remains a stable proxy for the current
    /// left-to-right order because a merge always retains its left node.
    left: usize,
    right: usize,
    left_generation: u32,
    right_generation: u32,
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            // The scan implementation chose the first pair on ties. Reverse
            // the position comparison so BinaryHeap returns the leftmost one.
            .then_with(|| other.left.cmp(&self.left))
            .then_with(|| other.right.cmp(&self.right))
            .then_with(|| other.left_generation.cmp(&self.left_generation))
            .then_with(|| other.right_generation.cmp(&self.right_generation))
    }
}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Convert a selectable SentencePiece score to an integer key whose ordering
/// is the same as the score's numeric ordering. Negative zero is canonicalized
/// so it ties with positive zero, matching `f32::gt` in the former scan loop.
fn sentencepiece_score_priority(score: f32) -> Option<u32> {
    if score.is_nan() || score == f32::NEG_INFINITY {
        return None;
    }
    let score = if score == 0.0 { 0.0 } else { score };
    let bits = score.to_bits();
    Some(if bits & (1 << 31) == 0 {
        bits ^ (1 << 31)
    } else {
        !bits
    })
}

/// Merge adjacent pieces according to a stable priority function.
///
/// The previous implementation rescanned and compacted the entire piece
/// vector after every merge. Keeping adjacency in a linked arena and only
/// refreshing the two affected boundaries reduces selection from quadratic to
/// O(n log n), while the original position tie-break preserves byte-for-byte
/// tokenization behavior.
fn merge_adjacent_pieces(
    pieces: Vec<String>,
    mut priority_for: impl FnMut(&str, &str) -> Option<u32>,
) -> Vec<String> {
    if pieces.len() < 2 {
        return pieces;
    }

    let piece_count = pieces.len();
    let mut nodes = pieces
        .into_iter()
        .enumerate()
        .map(|(index, piece)| MergeNode {
            piece,
            previous: index.checked_sub(1),
            next: (index + 1 < piece_count).then_some(index + 1),
            generation: 0,
            alive: true,
        })
        .collect::<Vec<_>>();

    let mut candidates = BinaryHeap::new();
    let push_candidate =
        |left: usize,
         right: usize,
         nodes: &[MergeNode],
         candidates: &mut BinaryHeap<MergeCandidate>,
         priority_for: &mut dyn FnMut(&str, &str) -> Option<u32>| {
            if let Some(priority) = priority_for(&nodes[left].piece, &nodes[right].piece) {
                candidates.push(MergeCandidate {
                    priority,
                    left,
                    right,
                    left_generation: nodes[left].generation,
                    right_generation: nodes[right].generation,
                });
            }
        };

    for left in 0..nodes.len() - 1 {
        push_candidate(left, left + 1, &nodes, &mut candidates, &mut priority_for);
    }

    while let Some(candidate) = candidates.pop() {
        let left = candidate.left;
        let right = candidate.right;
        if !nodes[left].alive
            || !nodes[right].alive
            || nodes[left].next != Some(right)
            || nodes[left].generation != candidate.left_generation
            || nodes[right].generation != candidate.right_generation
        {
            continue;
        }

        let previous = nodes[left].previous;
        let next = nodes[right].next;
        let right_piece = std::mem::take(&mut nodes[right].piece);
        nodes[left].piece.push_str(&right_piece);
        nodes[left].next = next;
        nodes[left].generation = nodes[left].generation.wrapping_add(1);
        nodes[right].alive = false;
        nodes[right].generation = nodes[right].generation.wrapping_add(1);
        if let Some(next) = next {
            nodes[next].previous = Some(left);
        }

        if let Some(previous) = previous {
            push_candidate(previous, left, &nodes, &mut candidates, &mut priority_for);
        }
        if let Some(next) = next {
            push_candidate(left, next, &nodes, &mut candidates, &mut priority_for);
        }
    }

    nodes
        .into_iter()
        .filter_map(|node| node.alive.then_some(node.piece))
        .collect()
}

/// BPE tokenizer model type (determined from GGUF metadata).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerModel {
    /// SentencePiece BPE — score-based merges (Llama 2, Mistral, etc.)
    SentencePiece,
    /// GPT-2 BPE — ordered merge list (Llama 3, Qwen, etc.)
    Gpt2,
    /// Gemma 4 BPE — ordered merge list over raw UTF-8 text with ▁ space normalization.
    Gemma4,
}

/// Chat template style (detected from vocabulary).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatStyle {
    /// Llama 3 style: <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n...
    Llama3,
    /// Llama 4 style: <|header_start|>user<|header_end|>\n\n...<|eot|>
    Llama4,
    /// Muse Spark recipient headers and separate message/turn endings.
    MuseSpark,
    /// Phi-3 style: <|user|>\n...\n<|end|>\n<|assistant|>\n
    Phi3,
    /// Classic instruct: `[INST] user [/INST] assistant`
    Instruct,
    /// Mistral instruct: fold an optional system prompt into the first `[INST]` turn.
    Mistral,
    /// ChatML: <|im_start|>system\n...<|im_end|>
    ChatML,
    /// Qwen 3 family ChatML with explicit thinking-mode prompt control.
    Qwen3,
    /// GLM-5.3 role tokens with the GLM mask/start prefix.
    Glm5,
    /// Gemma 4 turn format: `<bos><|turn>user\n...<turn|>\n<|turn>model\n`
    Gemma4,
    /// Nemotron-H ChatML with explicit thinking-mode prompt control.
    Nemotron,
    /// DeepSeek-R1 format: `<｜User｜>...<｜Assistant｜><think>\n`
    DeepSeekR1,
    /// DeepSeek V4 format: `<｜User｜>...<｜Assistant｜></think>` by default.
    DeepSeekV4,
    /// DeepSeek V4.1 1017, with numeric effort and spaced DSML tags.
    DeepSeekV41,
    /// DeepSeek-V2 format: User: ...\n\nAssistant:
    DeepSeek,
    /// MiniMax M2 format: ]~b]system / ]~b]user / ]~b]ai turns.
    MiniMax,
    /// No chat template — just concatenate messages
    Raw,
}

/// A chat message with the OpenAI tool-call metadata needed by model-native
/// prompt templates.
///
/// This is deliberately an engine-side type.  Caller-facing APIs and
/// serialization formats can evolve independently while the engine keeps one bounded,
/// model-aware representation at the tokenizer boundary.  `content` is the
/// visible text (an empty string is valid for an assistant message containing
/// only tool calls); tool arguments remain JSON text because that is the
/// representation used by the OpenAI wire format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTemplateMessage {
    /// Message role, such as `system`, `user`, `assistant`, or `tool`.
    pub role: String,
    /// Visible message text. Empty text is valid when metadata carries the
    /// message's meaning, such as an assistant tool call.
    pub content: String,
    /// Optional speaker or function name supplied to the template.
    pub name: Option<String>,
    /// Identifier linking a tool result to the assistant call it answers.
    pub tool_call_id: Option<String>,
    /// Function calls emitted by an assistant message.
    pub tool_calls: Vec<ChatTemplateToolCall>,
    /// Earlier reasoning in this assistant turn, kept separate from its answer.
    pub reasoning_content: Option<String>,
}

impl ChatTemplateMessage {
    /// Create a message with no name, tool-result ID, or tool calls.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            reasoning_content: None,
        }
    }

    /// Attach assistant tool calls to this message, replacing any existing
    /// calls.
    pub fn with_tool_calls(
        mut self,
        tool_calls: impl IntoIterator<Item = ChatTemplateToolCall>,
    ) -> Self {
        self.tool_calls = tool_calls.into_iter().collect();
        self
    }

    /// Attach a template-visible speaker or function name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Attach the tool-call ID answered by this message.
    pub fn with_tool_call_id(mut self, tool_call_id: impl Into<String>) -> Self {
        self.tool_call_id = Some(tool_call_id.into());
        self
    }

    /// Retain reasoning while a tool turn is still in progress.
    pub fn with_reasoning_content(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning_content = Some(reasoning.into());
        self
    }
}

/// A function declaration supplied to a model's chat template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTemplateTool {
    /// Function name exposed to the model.
    pub name: String,
    /// Optional human-readable function description.
    pub description: Option<String>,
    /// A serialized JSON Schema object.  Keeping this as text avoids making
    /// the engine depend on any particular API crate's representation and
    /// preserves the OpenAI distinction between omitted and empty schemas.
    pub parameters_json: Option<String>,
}

impl ChatTemplateTool {
    /// Create a function declaration with no description or parameter schema.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            parameters_json: None,
        }
    }

    /// Set the human-readable function description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the serialized JSON Schema for function parameters.
    pub fn with_parameters_json(mut self, parameters_json: impl Into<String>) -> Self {
        self.parameters_json = Some(parameters_json.into());
        self
    }
}

/// A previous assistant function invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTemplateToolCall {
    /// Optional call ID used to correlate a tool result with the call.
    pub id: Option<String>,
    /// Function name requested by the assistant.
    pub name: String,
    /// JSON-encoded function arguments, as required by OpenAI Chat
    /// Completions.  The tokenizer accepts malformed historical arguments and
    /// quotes them when a template requires a JSON value; it never lets them
    /// escape into a control delimiter.
    pub arguments_json: String,
}

/// Tool selection semantics passed to a model chat template.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ChatTemplateToolChoice {
    #[default]
    /// Do not request a tool call.
    None,
    /// Let the model decide whether to call one of the supplied tools.
    Auto,
    /// Require the model to call a tool.
    Required,
    /// Require a call to the named function.
    Named(String),
}

impl ChatTemplateToolChoice {
    fn requires_tool(&self) -> bool {
        matches!(self, Self::Required | Self::Named(_))
    }
}

/// Tool rendering controls independent of any request serialization format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTemplateOptions {
    /// Function declarations available to the model.
    pub tools: Vec<ChatTemplateTool>,
    /// Whether tool use is disabled, automatic, required, or named.
    pub tool_choice: ChatTemplateToolChoice,
    /// Whether the template may represent multiple tool calls in one turn.
    pub parallel_tool_calls: bool,
    /// Explicit model-native effort; unsupported levels produce an error.
    pub reasoning_effort: Option<String>,
    /// False keeps the JSON envelope for a caller that cannot read the
    /// model's native tool syntax.
    pub native_tool_formats: bool,
}

impl ChatTemplateOptions {
    /// Create options for a tool set, defaulting to `None` for an empty set
    /// and `Auto` otherwise.
    pub fn new(tools: impl IntoIterator<Item = ChatTemplateTool>) -> Self {
        let tools = tools.into_iter().collect::<Vec<_>>();
        Self {
            tool_choice: if tools.is_empty() {
                ChatTemplateToolChoice::None
            } else {
                ChatTemplateToolChoice::Auto
            },
            tools,
            parallel_tool_calls: true,
            reasoning_effort: None,
            native_tool_formats: true,
        }
    }

    /// Set the tool-selection policy.
    pub fn with_tool_choice(mut self, tool_choice: ChatTemplateToolChoice) -> Self {
        self.tool_choice = tool_choice;
        self
    }

    /// Enable or disable parallel tool-call rendering.
    pub fn with_parallel_tool_calls(mut self, parallel_tool_calls: bool) -> Self {
        self.parallel_tool_calls = parallel_tool_calls;
        self
    }

    /// Set the model's reasoning effort.
    pub fn with_reasoning_effort(mut self, effort: Option<String>) -> Self {
        self.reasoning_effort = effort;
        self
    }

    /// Opt in to native tool syntax when the response reader supports it.
    pub fn with_native_tool_formats(mut self, enabled: bool) -> Self {
        self.native_tool_formats = enabled;
        self
    }
}

impl ChatTemplateToolCall {
    /// Create a function call with no correlation ID.
    pub fn new(name: impl Into<String>, arguments_json: impl Into<String>) -> Self {
        Self {
            id: None,
            name: name.into(),
            arguments_json: arguments_json.into(),
        }
    }

    /// Attach the call's correlation ID.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }
}

const MAX_CHAT_TEMPLATE_MESSAGES: usize = 4_096;
const MAX_CHAT_TEMPLATE_TOOLS: usize = 128;
const MAX_CHAT_TEMPLATE_TOOL_CALLS: usize = 128;
const MAX_CHAT_TEMPLATE_TOOL_NAME_BYTES: usize = 64;
const MAX_CHAT_TEMPLATE_TOOL_ID_BYTES: usize = 256;
const MAX_CHAT_TEMPLATE_DESCRIPTION_BYTES: usize = 16 * 1024;
const MAX_CHAT_TEMPLATE_SCHEMA_BYTES: usize = 256 * 1024;
const MAX_CHAT_TEMPLATE_ARGUMENT_BYTES: usize = 256 * 1024;
const MAX_CHAT_TEMPLATE_PROMPT_BYTES: usize = 8 * 1024 * 1024;

/// Build the GPT-2 unicode-to-byte reverse mapping.
///
/// GPT-2 maps all 256 byte values to printable Unicode characters:
/// - Printable bytes (33-126, 161-172, 174-255) map to their Unicode code point
/// - Non-printable bytes (0-32, 127-160, 173) map to U+0100..U+0143
fn build_gpt2_unicode_to_byte() -> HashMap<char, u8> {
    let mut map = HashMap::new();

    // Printable ranges: byte value == code point
    for b in 33u16..=126 {
        map.insert(char::from(b as u8), b as u8);
    }
    for b in 161u16..=172 {
        map.insert(char::from_u32(b as u32).unwrap(), b as u8);
    }
    for b in 174u16..=255 {
        map.insert(char::from_u32(b as u32).unwrap(), b as u8);
    }

    // Non-printable bytes: mapped sequentially starting at U+0100
    let mut n = 0u32;
    for b in 0u16..=255 {
        let is_printable =
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        if !is_printable {
            let ch = char::from_u32(256 + n).unwrap();
            map.insert(ch, b as u8);
            n += 1;
        }
    }
    map
}

fn build_gpt2_byte_to_unicode() -> HashMap<u8, char> {
    build_gpt2_unicode_to_byte()
        .into_iter()
        .map(|(ch, byte)| (byte, ch))
        .collect()
}

/// A BPE tokenizer extracted from GGUF metadata.
///
/// Supports both SentencePiece (score-based) and GPT-2 (merge-list) BPE.
pub struct GgufTokenizer {
    /// Vocabulary: token_id → token string
    vocab: Vec<String>,
    /// Reverse mapping: token string → token_id
    vocab_to_id: HashMap<String, u32>,
    /// SentencePiece scores (empty for GPT-2 BPE)
    scores: Vec<f32>,
    /// GPT-2 merge ranks: (left, right) → rank (lower = merge first)
    merge_ranks: HashMap<(String, String), u32>,
    /// Prefer a whole pretokenized vocabulary entry before applying BPE merges.
    ignore_merges: bool,
    /// MiniMax's case-aware word splits and NFC normalization.
    minimax_pretokenizer: bool,
    /// GLM's cl100k-style split; spaces never join number tokens.
    glm_pretokenizer: bool,
    /// JoyAI's three sequential isolated splits, before byte encoding.
    joyai_pretokenizer: bool,
    /// DeepSeek V1/V2's ordered isolated regex splits.
    deepseek_llm_pretokenizer: bool,
    /// Qwen's single-digit, whitespace and combining-mark split rules.
    qwen_pretokenizer: Option<QwenPretokenizer>,
    /// Control and added tokens matched atomically before normal BPE encoding.
    special_tokens: Vec<(String, u32)>,
    /// Optional leading-space marker used by Gemma-style BPE vocabularies.
    space_marker: Option<char>,
    /// Which BPE variant
    model: TokenizerModel,
    /// Detected chat style
    pub chat_style: ChatStyle,
    /// Whether the embedded Qwen template opens the default assistant turn with `<think>\n`.
    qwen_default_think_prefix: bool,
    /// The embedded template closes thinking unless the caller enables it.
    qwen_default_closed_thinking: bool,
    /// The embedded ChatML template supplies Qwen's default system message.
    qwen_chatml_default_system: bool,
    /// Whether the embedded Qwen template adds its default xhigh reasoning instruction.
    qwen_default_xhigh_reasoning: bool,
    /// Qwen 3.5 and later use function/parameter tags instead of JSON calls.
    qwen_xml_tools: bool,
    /// Beginning-of-sequence token ID
    pub bos_id: u32,
    /// End-of-sequence token ID
    pub eos_id: u32,
    /// Additional stop token IDs (model-specific)
    pub stop_ids: Vec<u32>,
    /// Whether to prepend BOS on encode
    add_bos: bool,
    /// Byte fallback tokens: byte_value → token_id (for <0xHH> tokens)
    byte_tokens: HashMap<u8, u32>,
    /// GPT-2 unicode→byte reverse mapping (only populated for GPT-2 BPE)
    gpt2_unicode_to_byte: HashMap<char, u8>,
    /// GPT-2 byte→unicode mapping used before applying merge ranks.
    gpt2_byte_to_unicode: HashMap<u8, char>,
    /// Build the decoded token trie once for this loaded tokenizer.
    pub(crate) tool_parser_factory:
        std::sync::OnceLock<std::result::Result<std::sync::Arc<llguidance::ParserFactory>, String>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QwenPretokenizer {
    V2,
    V35,
}

const QWEN_XHIGH_REASONING_INSTRUCTION: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const QWEN_DEFAULT_SYSTEM: &str =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

fn thinking_control_value(style: ChatStyle, content: &str) -> Option<bool> {
    match style {
        ChatStyle::Qwen3 if content.contains("/no_think") => Some(false),
        ChatStyle::Qwen3 | ChatStyle::Nemotron if content.contains("/think") => Some(true),
        ChatStyle::Nemotron if content.contains("/no_think") => Some(false),
        _ => None,
    }
}

fn strip_thinking_controls(content: &str) -> String {
    content
        .replace("/no_think", "")
        .replace("/think", "")
        .trim()
        .to_string()
}

/// Conservatively recognize the Qwen3-family generation-prompt branch that
/// opens a thinking response. Other Qwen templates retain their formatting.
fn qwen_template_appends_default_think_prefix(template: &str) -> bool {
    let Some(generation_start) = template.rfind("add_generation_prompt") else {
        return false;
    };
    let generation = &template[generation_start..];

    let prefix_start = [
        "'<think>\\n'",
        "\"<think>\\n\"",
        "'<think>\n'",
        "\"<think>\n\"",
    ]
    .into_iter()
    .filter_map(|literal| generation.find(literal))
    .min();
    let Some(prefix_start) = prefix_start else {
        return false;
    };

    let before_prefix = &generation[..prefix_start];
    let Some(else_start) = before_prefix.rfind("{%") else {
        return false;
    };
    let Some(else_end) = before_prefix[else_start + 2..].find("%}") else {
        return false;
    };
    let else_tag = before_prefix[else_start + 2..else_start + 2 + else_end]
        .trim()
        .trim_matches('-')
        .trim();
    if else_tag != "else" {
        return false;
    }

    let before_else = &before_prefix[..else_start];
    let Some(if_start) = before_else.rfind("{%") else {
        return false;
    };
    let Some(if_end) = before_else[if_start + 2..].find("%}") else {
        return false;
    };
    let if_tag = before_else[if_start + 2..if_start + 2 + if_end]
        .trim()
        .trim_matches('-')
        .trim();
    let Some(condition) = if_tag.strip_prefix("if ") else {
        return false;
    };
    let compact_condition = condition.split_whitespace().collect::<String>();
    if compact_condition != "enable_thinkingisdefinedandenable_thinkingisfalse" {
        return false;
    }

    let assistant_prefix = &generation[..if_start];
    assistant_prefix.contains("<|im_start|>assistant\\n")
        || assistant_prefix.contains("<|im_start|>assistant\n")
}

/// Some GGUF publishers reverse the primary Qwen template's default. Recognize
/// that explicit branch rather than choosing a default from the model name.
fn qwen_template_defaults_to_closed_thinking(template: &str) -> bool {
    let Some(start) = template.rfind("add_generation_prompt") else {
        return false;
    };
    let generation = &template[start..];
    let Some(open) = [
        "'<think>\\n'",
        "\"<think>\\n\"",
        "'<think>\n'",
        "\"<think>\n\"",
    ]
    .into_iter()
    .filter_map(|literal| generation.find(literal))
    .min() else {
        return false;
    };
    let before = &generation[..open];
    let Some(statement) = before.rfind("{%") else {
        return false;
    };
    let Some(end) = before[statement + 2..].find("%}") else {
        return false;
    };
    let condition = before[statement + 2..statement + 2 + end]
        .trim()
        .trim_matches('-')
        .split_whitespace()
        .collect::<String>();
    if condition != "ifenable_thinkingisdefinedandenable_thinkingistrue" {
        return false;
    }
    let after = &generation[open..];
    let Some(next) = after.find("{%") else {
        return false;
    };
    let Some(end) = after[next + 2..].find("%}") else {
        return false;
    };
    let next_tag = after[next + 2..next + 2 + end]
        .trim()
        .trim_matches('-')
        .trim();
    next_tag == "else"
        && (before[..statement].contains("<|im_start|>assistant\\n")
            || before[..statement].contains("<|im_start|>assistant\n"))
        && [
            "'<think>\\n\\n</think>\\n\\n'",
            "\"<think>\\n\\n</think>\\n\\n\"",
            "'<think>\n\n</think>\n\n'",
            "\"<think>\n\n</think>\n\n\"",
        ]
        .iter()
        .any(|literal| after[next + 2 + end..].contains(literal))
}

fn qwen_template_has_default_system(template: &str) -> bool {
    let escaped = format!("<|im_start|>system\\n{QWEN_DEFAULT_SYSTEM}<|im_end|>\\n");
    let literal = format!("<|im_start|>system\n{QWEN_DEFAULT_SYSTEM}<|im_end|>\n");
    // Recognize the published no-system branch, not just the presence of its
    // text elsewhere in a custom template. Its if/else bodies have no nested
    // statements, and the else branch emits this complete literal directly.
    for value in [escaped, literal] {
        for quoted in [format!("'{value}'"), format!("\"{value}\"")] {
            for (start, _) in template.match_indices(&quoted) {
                let before = &template[..start];
                let Some(expression) = before.rfind("{{") else {
                    continue;
                };
                if !before[expression + 2..]
                    .trim()
                    .trim_matches('-')
                    .trim()
                    .is_empty()
                {
                    continue;
                }
                let after = template[start + quoted.len()..].trim_start();
                if !after.starts_with("}}") && !after.starts_with("-}}") {
                    continue;
                }
                let Some((before_else, else_tag)) = previous_jinja_statement(&before[..expression])
                else {
                    continue;
                };
                if else_tag != "else" {
                    continue;
                }
                let Some((_, if_tag)) = previous_jinja_statement(before_else) else {
                    continue;
                };
                let condition = if_tag
                    .split_whitespace()
                    .collect::<String>()
                    .replace('"', "'");
                if condition == "ifmessages[0]['role']=='system'" {
                    return true;
                }
            }
        }
    }
    false
}

fn previous_jinja_statement(text: &str) -> Option<(&str, &str)> {
    let start = text.rfind("{%")?;
    let end = text[start + 2..].find("%}")?;
    Some((
        &text[..start],
        text[start + 2..start + 2 + end]
            .trim()
            .trim_matches('-')
            .trim(),
    ))
}

/// Recognize the Qwen3.8 template's default xhigh system instruction.
fn qwen_template_defaults_to_xhigh_reasoning(template: &str) -> bool {
    template.contains("reasoning_effort")
        && (template.contains("default('xhigh')") || template.contains("default(\"xhigh\")"))
        && template.contains(QWEN_XHIGH_REASONING_INSTRUCTION)
}

/// Reproduce Qwen3-family handling of assistant history. Completed reasoning is
/// omitted from earlier turns, while assistant messages after the latest user
/// query preserve it for in-progress tool turns.
fn qwen3_assistant_history(content: &str, preserve_thinking: bool) -> String {
    let content = content.trim();
    let Some(first_close) = content.find("</think>") else {
        return if preserve_thinking {
            format!("<think>\n\n</think>\n\n{content}")
        } else {
            content.to_string()
        };
    };
    let last_close = content.rfind("</think>").unwrap_or(first_close);
    let visible = content[last_close + "</think>".len()..].trim_start_matches('\n');
    if !preserve_thinking {
        return visible.to_string();
    }

    let reasoning = content[..first_close]
        .trim_end_matches('\n')
        .rsplit("<think>")
        .next()
        .unwrap_or_default()
        .trim_start_matches('\n')
        .trim();
    format!("<think>\n{reasoning}\n</think>\n\n{visible}")
}

impl GgufTokenizer {
    /// Extract tokenizer from a GGUF file's metadata.
    pub fn from_gguf(path: &Path) -> Result<Self> {
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open GGUF: {}", path.display()))?;
        validate_gguf_header_budget(&mut file)
            .with_context(|| format!("invalid GGUF header: {}", path.display()))?;
        let metadata = match gguf_file::Content::read(&mut file) {
            Ok(content) => content.metadata,
            Err(candle_error) => {
                file.seek(SeekFrom::Start(0))?;
                RawGgufContent::read(&mut file)
                    .with_context(|| {
                        format!(
                            "Failed to parse GGUF header with Candle ({candle_error}) or the DINF reader"
                        )
                    })?
                    .metadata
            }
        };
        Self::from_metadata(&metadata)
    }

    /// Build tokenizer from parsed GGUF metadata.
    pub fn from_metadata(md: &HashMap<String, gguf_file::Value>) -> Result<Self> {
        // Extract vocabulary tokens
        let tokens_val = md
            .get("tokenizer.ggml.tokens")
            .ok_or_else(|| anyhow::anyhow!("missing tokenizer.ggml.tokens"))?;
        let tokens_arr = tokens_val
            .to_vec()
            .map_err(|e| anyhow::anyhow!("tokenizer.ggml.tokens: {e}"))?;
        let vocab: Vec<String> = tokens_arr
            .iter()
            .map(gguf_string_value)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("tokenizer vocab extraction: {e}"))?;

        let token_types: Vec<i32> = if let Some(types_val) = md.get("tokenizer.ggml.token_type") {
            let types = types_val
                .to_vec()
                .context("tokenizer.ggml.token_type must be an array")?;
            anyhow::ensure!(
                types.len() == vocab.len(),
                "tokenizer.ggml.token_type length must match the vocabulary"
            );
            types
                .iter()
                .map(|value| {
                    let value = match value {
                        gguf_file::Value::I32(n) => return Ok(*n),
                        gguf_file::Value::I64(n) => i32::try_from(*n),
                        gguf_file::Value::U32(n) => i32::try_from(*n),
                        gguf_file::Value::U64(n) => i32::try_from(*n),
                        _ => anyhow::bail!("tokenizer.ggml.token_type entries must be integers"),
                    };
                    value.context("tokenizer.ggml.token_type entry exceeds I32")
                })
                .collect::<Result<_>>()?
        } else {
            Vec::new()
        };

        // Extract scores (SentencePiece) — may be absent for GPT-2 models
        let scores: Vec<f32> = if let Some(scores_val) = md.get("tokenizer.ggml.scores") {
            let scores_arr = scores_val
                .to_vec()
                .map_err(|e| anyhow::anyhow!("tokenizer.ggml.scores: {e}"))?;
            anyhow::ensure!(
                scores_arr.len() == vocab.len(),
                "tokenizer.ggml.scores length must match the vocabulary"
            );
            scores_arr
                .iter()
                .map(|v| v.to_f32())
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| anyhow::anyhow!("tokenizer score extraction: {e}"))?
        } else {
            Vec::new()
        };

        // Determine BPE model type
        let model_str = md
            .get("tokenizer.ggml.model")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_default();
        let architecture = md
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_default();
        let chat_template = md
            .get("tokenizer.chat_template")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_default();
        let tokenizer_pre = md
            .get("tokenizer.ggml.pre")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_default();
        let glm_pretokenizer = matches!(architecture.as_str(), "glm5next" | "glm5-next");

        // Build reverse lookup
        let mut vocab_to_id = HashMap::with_capacity(vocab.len());
        for (id, token) in vocab.iter().enumerate() {
            vocab_to_id.insert(token.clone(), id as u32);
        }

        // Extract merges (GPT-2/Gemma BPE). Some Gemma merges contain literal
        // leading-space tokens, so the separating space must be found with
        // vocabulary context rather than `split_once`.
        let mut merge_ranks = HashMap::new();
        let model = if let Some(merges_val) = md.get("tokenizer.ggml.merges") {
            let merges_arr = merges_val
                .to_vec()
                .context("tokenizer.ggml.merges must be an array")?;
            for (rank, merge_val) in merges_arr.iter().enumerate() {
                let merge_str = gguf_string_value(merge_val)
                    .context("tokenizer.ggml.merges entries must be strings")?;
                if let Some((left, right)) = parse_merge_pair(&merge_str, &vocab_to_id) {
                    merge_ranks.insert((left, right), rank as u32);
                }
            }
            if model_str == "gemma4" {
                TokenizerModel::Gemma4
            } else if model_str != "gpt2" && merge_ranks.is_empty() {
                TokenizerModel::SentencePiece
            } else {
                TokenizerModel::Gpt2
            }
        } else if model_str == "gemma4" {
            TokenizerModel::Gemma4
        } else if model_str == "gpt2" {
            TokenizerModel::Gpt2
        } else {
            TokenizerModel::SentencePiece
        };

        let mut special_tokens = Vec::new();
        for (id, token) in vocab.iter().enumerate() {
            let token_type = token_types.get(id).copied().unwrap_or(1);
            if matches!(token_type, 3 | 4)
                || looks_like_special_token(token)
                || (glm_pretokenizer
                    && matches!(
                        token.as_str(),
                        "[gMASK]" | "<sop>" | "<eop>" | "<think>" | "</think>"
                    ))
                // Non-special official AddedTokens remain GGUF type 1 and
                // are absent from the BPE vocabulary. Scope their recognition
                // to this family; decoding and stop IDs stay independent.
                || ((architecture == "deepseek41" || tokenizer_pre == "joyai-llm")
                    && matches!(
                        token.as_str(),
                        "<think>" | "</think>" | "｜DSML｜" | "<dsml:" | "</dsml:"
                    ))
            {
                special_tokens.push((token.clone(), id as u32));
            }
        }
        special_tokens.sort_by(|(a, _), (b, _)| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        let space_marker = if model_str == "gemma4" && vocab_to_id.contains_key("▁") {
            Some('▁')
        } else {
            None
        };
        let ignore_merges = tokenizer_pre == "pixtral";
        let minimax_pretokenizer = tokenizer_pre == "minimax-m2";

        // Extract special token IDs
        let bos_id = metadata_token_id(md, "tokenizer.ggml.bos_token_id", vocab.len())?
            .unwrap_or_else(|| {
                if glm_pretokenizer {
                    vocab_to_id.get("[gMASK]").copied().unwrap_or(1)
                } else {
                    1
                }
            });

        let eos_id =
            metadata_token_id(md, "tokenizer.ggml.eos_token_id", vocab.len())?.unwrap_or(2);

        let mut add_bos = md
            .get("tokenizer.ggml.add_bos_token")
            .map(|value| {
                value
                    .to_bool()
                    .context("tokenizer.ggml.add_bos_token must be a boolean")
            })
            .transpose()?
            .unwrap_or(match model {
                TokenizerModel::SentencePiece | TokenizerModel::Gemma4 => true,
                // GPT-2 BPE defaults to no BOS. These presets explicitly
                // enable it in llama.cpp before the GGUF flag override.
                TokenizerModel::Gpt2 => matches!(
                    tokenizer_pre.as_str(),
                    "llama3"
                        | "llama-v3"
                        | "llama-bpe"
                        | "falcon3"
                        | "falcon-h1"
                        | "pixtral"
                        | "midm-2.0"
                        | "lfm2"
                        | "jina-v5-nano"
                        | "tekken"
                        | "chameleon"
                ),
            });
        // llama.cpp overrides Gemma4 to always add BOS for correct prompt parity.
        if model == TokenizerModel::Gemma4 {
            add_bos = true;
        }

        // Build byte fallback map (<0x00> through <0xFF>)
        let mut byte_tokens = HashMap::new();
        for byte_val in 0u16..=255 {
            let hex_token = format!("<0x{:02X}>", byte_val);
            if let Some(&id) = vocab_to_id.get(&hex_token) {
                byte_tokens.insert(byte_val as u8, id);
            }
        }

        // Build GPT-2 reverse mapping if needed
        let gpt2_unicode_to_byte = if model == TokenizerModel::Gpt2 {
            build_gpt2_unicode_to_byte()
        } else {
            HashMap::new()
        };
        let gpt2_byte_to_unicode = if model == TokenizerModel::Gpt2 {
            build_gpt2_byte_to_unicode()
        } else {
            HashMap::new()
        };

        // Detect chat style from vocabulary
        if architecture == "muse-spark" {
            anyhow::ensure!(
                model == TokenizerModel::Gpt2 && tokenizer_pre == "gpt-4o",
                "Muse Spark requires the native gpt-4o BPE tokenizer"
            );
            anyhow::ensure!(
                ["<|start|>", "<|message|>", "<|eom|>", "<|eot|>"]
                    .iter()
                    .all(|token| vocab_to_id.contains_key(*token)),
                "Muse Spark vocabulary lacks recipient control tokens"
            );
        }
        let chat_style = if architecture == "muse-spark" {
            ChatStyle::MuseSpark
        } else if matches!(architecture.as_str(), "glm5next" | "glm5-next")
            && vocab_to_id.contains_key("[gMASK]")
            && vocab_to_id.contains_key("<sop>")
            && vocab_to_id.contains_key("<|user|>")
            && vocab_to_id.contains_key("<|assistant|>")
        {
            ChatStyle::Glm5
        } else if architecture == "deepseek41" {
            ChatStyle::DeepSeekV41
        } else if architecture == "deepseek4" {
            ChatStyle::DeepSeekV4
        } else if vocab_to_id.contains_key("<｜User｜>")
            && vocab_to_id.contains_key("<｜Assistant｜>")
            && vocab_to_id.contains_key("<｜end▁of▁sentence｜>")
        {
            ChatStyle::DeepSeekR1
        } else if vocab_to_id.contains_key("<|begin_of_text|>")
            && vocab_to_id.contains_key("<|start_header_id|>")
        {
            ChatStyle::Llama3
        } else if vocab_to_id.contains_key("<|header_start|>")
            && vocab_to_id.contains_key("<|header_end|>")
            && vocab_to_id.contains_key("<|eot|>")
        {
            ChatStyle::Llama4
        } else if model_str == "gemma4"
            && vocab_to_id.contains_key("<|turn>")
            && vocab_to_id.contains_key("<turn|>")
        {
            ChatStyle::Gemma4
        } else if architecture.starts_with("nemotron_h")
            && vocab_to_id.contains_key("<|im_start|>")
            && vocab_to_id.contains_key("<|im_end|>")
        {
            ChatStyle::Nemotron
        } else if matches!(architecture.as_str(), "qwen3moe" | "qwen35")
            && vocab_to_id.contains_key("<|im_start|>")
            && vocab_to_id.contains_key("<|im_end|>")
        {
            ChatStyle::Qwen3
        } else if vocab_to_id.contains_key("<|im_start|>") && vocab_to_id.contains_key("<|im_end|>")
        {
            ChatStyle::ChatML
        } else if architecture == "deepseek2" {
            ChatStyle::DeepSeek
        } else if architecture == "minimax-m2"
            && vocab_to_id.contains_key("]~b]")
            && vocab_to_id.contains_key("[e~[")
        {
            ChatStyle::MiniMax
        } else if vocab_to_id.contains_key("<|user|>")
            && vocab_to_id.contains_key("<|assistant|>")
            && vocab_to_id.contains_key("<|end|>")
        {
            ChatStyle::Phi3
        } else if vocab_to_id.contains_key("[INST]")
            && chat_template.contains("Only user and assistant roles are supported")
            && !chat_template.contains("<<SYS>>")
        {
            ChatStyle::Mistral
        } else if vocab_to_id.contains_key("[INST]") {
            ChatStyle::Instruct
        } else {
            ChatStyle::Raw
        };
        let qwen_default_think_prefix = chat_style == ChatStyle::Qwen3
            && qwen_template_appends_default_think_prefix(&chat_template);
        let qwen_default_closed_thinking = chat_style == ChatStyle::Qwen3
            && qwen_template_defaults_to_closed_thinking(&chat_template);
        let qwen_chatml_default_system =
            chat_style == ChatStyle::ChatML && qwen_template_has_default_system(&chat_template);
        let qwen_default_xhigh_reasoning = chat_style == ChatStyle::Qwen3
            && qwen_template_defaults_to_xhigh_reasoning(&chat_template);

        // Collect additional stop token IDs based on chat style
        let mut stop_ids = vec![eos_id];
        let stop_tokens: &[&str] = match chat_style {
            ChatStyle::Llama3 => &["<|eot_id|>", "<|end_of_text|>"],
            ChatStyle::Llama4 => &["<|eot|>", "<|end_of_text|>"],
            ChatStyle::MuseSpark => &["<|eot|>", "<|end_of_text|>"],
            ChatStyle::Phi3 => &["<|end|>"],
            ChatStyle::ChatML => &["<|im_end|>", "<|endoftext|>"],
            ChatStyle::Qwen3 => &["<|im_end|>", "<|endoftext|>"],
            ChatStyle::Glm5 => &["<|endoftext|>", "<|user|>", "<|observation|>"],
            ChatStyle::Gemma4 => &["<turn|>", "<eos>"],
            ChatStyle::Nemotron => &["<|im_end|>", "<|endoftext|>"],
            ChatStyle::DeepSeekR1 | ChatStyle::DeepSeekV4 | ChatStyle::DeepSeekV41 => {
                &["<｜end▁of▁sentence｜>"]
            }
            ChatStyle::DeepSeek => &[],
            ChatStyle::MiniMax => &["[e~["],
            ChatStyle::Mistral => &["</s>"],
            ChatStyle::Instruct => &["[/INST]", "</s>"],
            ChatStyle::Raw => &[],
        };
        for &tok in stop_tokens {
            if let Some(&id) = vocab_to_id.get(tok)
                && !stop_ids.contains(&id)
            {
                stop_ids.push(id);
            }
        }
        if chat_style == ChatStyle::MuseSpark {
            // EOM joins reasoning, commentary and tool messages in one turn.
            if let Some(eom) = vocab_to_id.get("<|eom|>") {
                stop_ids.retain(|id| id != eom);
            }
        }

        tracing::info!(
            "Tokenizer loaded: {} tokens, model={:?}, chat={:?}, bos={}, eos={}, stops={:?}",
            vocab.len(),
            model,
            chat_style,
            bos_id,
            eos_id,
            stop_ids
        );

        Ok(Self {
            vocab,
            vocab_to_id,
            scores,
            merge_ranks,
            ignore_merges,
            minimax_pretokenizer,
            glm_pretokenizer,
            joyai_pretokenizer: tokenizer_pre == "joyai-llm",
            deepseek_llm_pretokenizer: tokenizer_pre == "deepseek-llm",
            qwen_pretokenizer: match tokenizer_pre.as_str() {
                "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" => Some(QwenPretokenizer::V2),
                "qwen35" => Some(QwenPretokenizer::V35),
                _ => None,
            },
            special_tokens,
            space_marker,
            model,
            chat_style,
            qwen_default_think_prefix,
            qwen_default_closed_thinking,
            qwen_chatml_default_system,
            qwen_default_xhigh_reasoning,
            qwen_xml_tools: chat_template.contains("<function="),
            bos_id,
            eos_id,
            stop_ids,
            add_bos,
            byte_tokens,
            gpt2_unicode_to_byte,
            gpt2_byte_to_unicode,
            tool_parser_factory: std::sync::OnceLock::new(),
        })
    }

    /// Encode a text string into token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        if self.add_bos
            && !(self.chat_style == ChatStyle::DeepSeekV41
                && text.starts_with("<｜begin▁of▁sentence｜>"))
        {
            tokens.push(self.bos_id);
        }
        tokens.extend(self.encode_with_specials(text)?);
        Ok(tokens)
    }

    fn encode_plain(&self, text: &str) -> Result<Vec<u32>> {
        match self.model {
            TokenizerModel::SentencePiece => self.encode_sentencepiece(text),
            TokenizerModel::Gpt2 => self.encode_gpt2(text),
            TokenizerModel::Gemma4 => self.encode_gemma4(text),
        }
    }

    fn encode_with_specials(&self, text: &str) -> Result<Vec<u32>> {
        if self.special_tokens.is_empty() || text.is_empty() {
            return self.encode_plain(text);
        }

        let mut ids = Vec::new();
        let mut plain = String::new();
        let mut pos = 0;
        while pos < text.len() {
            if let Some((len, id)) = self.match_special_at(text, pos) {
                if !plain.is_empty() {
                    ids.extend(self.encode_plain(&plain)?);
                    plain.clear();
                }
                ids.push(id);
                pos += len;
                if self.special_token_rstrips_whitespace(id) {
                    while pos < text.len() {
                        let ch = text[pos..].chars().next().unwrap();
                        if !ch.is_ascii_whitespace() {
                            break;
                        }
                        pos += ch.len_utf8();
                    }
                }
            } else {
                let ch = text[pos..].chars().next().unwrap();
                plain.push(ch);
                pos += ch.len_utf8();
            }
        }

        if !plain.is_empty() {
            ids.extend(self.encode_plain(&plain)?);
        }
        Ok(ids)
    }

    fn special_token_rstrips_whitespace(&self, id: u32) -> bool {
        if self.chat_style != ChatStyle::Phi3 {
            return false;
        }

        !matches!(
            self.vocab.get(id as usize).map(String::as_str),
            Some("<unk>" | "<s>" | "<|endoftext|>")
        )
    }

    fn match_special_at(&self, text: &str, pos: usize) -> Option<(usize, u32)> {
        self.special_tokens.iter().find_map(|(token, id)| {
            if !token.is_empty() && text[pos..].starts_with(token) {
                Some((token.len(), *id))
            } else {
                None
            }
        })
    }

    /// Encode chat messages using the appropriate template.
    pub fn encode_chat(&self, messages: &[(String, String)]) -> Result<Vec<u32>> {
        if matches!(
            self.chat_style,
            ChatStyle::DeepSeekV41 | ChatStyle::MuseSpark
        ) {
            let messages = messages
                .iter()
                .map(|(role, content)| ChatTemplateMessage::new(role, content))
                .collect::<Vec<_>>();
            return self.encode_chat_with_tool_options(&messages, &ChatTemplateOptions::new([]));
        }
        let text = self.format_chat(messages);
        self.encode(&text)
    }

    /// Find a stable system/developer prefix in the actual encoded prompt.
    /// Re-encode a prompt with an empty user turn, then compare token IDs: text
    /// lengths cannot locate a safe boundary when tokenizer merges differ.
    pub fn shared_chat_prefix_len(
        &self,
        messages: &[(String, String)],
        tokens: &[u32],
    ) -> Option<usize> {
        let mut prefix = messages
            .iter()
            .take_while(|(role, _)| matches!(role.as_str(), "system" | "developer"))
            .cloned()
            .collect::<Vec<_>>();
        if prefix.is_empty() {
            return None;
        }
        prefix.push(("user".into(), String::new()));
        let candidate = self.encode_chat(&prefix).ok()?;
        shared_token_prefix_len(tokens, &candidate)
    }

    /// As above, including the exact tool definitions and selection rules.
    pub fn shared_tool_prefix_len(
        &self,
        messages: &[ChatTemplateMessage],
        options: &ChatTemplateOptions,
        tokens: &[u32],
    ) -> Option<usize> {
        let mut prefix = messages
            .iter()
            .take_while(|message| matches!(message.role.as_str(), "system" | "developer"))
            .cloned()
            .collect::<Vec<_>>();
        if prefix.is_empty() && options.tools.is_empty() {
            return None;
        }
        prefix.push(ChatTemplateMessage::new("user", ""));
        let candidate = self.encode_chat_with_tool_options(&prefix, options).ok()?;
        shared_token_prefix_len(tokens, &candidate)
    }

    /// Encode a chat with OpenAI-compatible function declarations and tool
    /// history.  The legacy [`Self::encode_chat`] path is intentionally kept
    /// separate so ordinary prompts retain byte-for-byte formatting.
    pub fn encode_chat_with_tools(
        &self,
        messages: &[ChatTemplateMessage],
        tools: &[ChatTemplateTool],
    ) -> Result<Vec<u32>> {
        let text = self.format_chat_with_tools(messages, tools)?;
        self.encode(&text)
    }

    /// Encode a conversation with full tool-selection semantics.
    pub fn encode_chat_with_tool_options(
        &self,
        messages: &[ChatTemplateMessage],
        options: &ChatTemplateOptions,
    ) -> Result<Vec<u32>> {
        let text = self.format_chat_with_tool_options(messages, options)?;
        self.encode(&text)
    }

    /// Look up a vocabulary token and return its ID, if present.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.vocab_to_id.get(token).copied()
    }

    /// Format chat messages into a prompt string.
    /// Format plain messages with the model's exact generation prefix.
    ///
    /// Legacy infallible API: panics on invalid native message roles/history.
    /// Request handlers must use [`Self::encode_chat`] or
    /// [`Self::format_chat_with_tool_options`] to propagate formatting errors.
    pub fn format_chat(&self, messages: &[(String, String)]) -> String {
        self.format_chat_with_thinking_state(messages, None, None)
    }

    /// Tool prompts pass controls read from the original message text. Do not
    /// read or strip controls again after adding tool JSON to those messages.
    fn format_chat_with_thinking_state(
        &self,
        messages: &[(String, String)],
        thinking_state: Option<bool>,
        reasoning_effort: Option<&str>,
    ) -> String {
        match self.chat_style {
            ChatStyle::DeepSeekV41 => {
                let messages = messages
                    .iter()
                    .map(|(role, content)| ChatTemplateMessage::new(role, content))
                    .collect::<Vec<_>>();
                deepseek41::format_chat(
                    &messages,
                    &ChatTemplateOptions::new([])
                        .with_reasoning_effort(reasoning_effort.map(str::to_owned)),
                )
                .expect("plain DeepSeek V4.1 chat requires valid text messages")
            }
            ChatStyle::MuseSpark => {
                let messages = messages
                    .iter()
                    .map(|(role, content)| {
                        let message = ChatTemplateMessage::new(role, content);
                        if matches!(role.as_str(), "tool" | "function") {
                            message.with_name("tool")
                        } else {
                            message
                        }
                    })
                    .collect::<Vec<_>>();
                // Legacy plain-chat callers cannot carry tool metadata.
                // Their valid roles have no fallible argument conversion.
                format_muse_chat(
                    &messages,
                    &ChatTemplateOptions::new([])
                        .with_reasoning_effort(reasoning_effort.map(str::to_owned)),
                    self.add_bos,
                )
                .expect("plain Muse chat requires supported message roles")
            }
            ChatStyle::Llama3 => {
                let mut out = if self.add_bos {
                    String::new()
                } else {
                    "<|begin_of_text|>".to_string()
                };

                let mut message_start = 0;
                let system_message = if let Some((role, content)) = messages.first() {
                    if role == "system" {
                        message_start = 1;
                        Some(content.trim())
                    } else {
                        None
                    }
                } else {
                    None
                };

                out.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
                out.push_str("Cutting Knowledge Date: December 2023\n");
                out.push_str(&format!(
                    "Today Date: {}\n\n",
                    chrono::Local::now().format("%d %b %Y")
                ));
                if let Some(content) = system_message {
                    out.push_str(content);
                }
                out.push_str("<|eot_id|>");

                for (role, content) in messages.iter().skip(message_start) {
                    out.push_str(&format!(
                        "<|start_header_id|>{role}<|end_header_id|>\n\n{}<|eot_id|>",
                        content.trim()
                    ));
                }
                // Add the assistant header to prompt generation
                out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                out
            }
            ChatStyle::Llama4 => {
                let mut out = if self.add_bos {
                    String::new()
                } else {
                    "<|begin_of_text|>".to_string()
                };
                for (role, content) in messages {
                    out.push_str(&format!(
                        "<|header_start|>{role}<|header_end|>\n\n{}<|eot|>",
                        content.trim()
                    ));
                }
                out.push_str("<|header_start|>assistant<|header_end|>\n\n");
                out
            }
            ChatStyle::Phi3 => {
                let mut out = String::new();
                for (role, content) in messages {
                    match role.as_str() {
                        "system" => {
                            out.push_str(&format!("<|system|>\n{}<|end|>\n", content.trim()));
                        }
                        "user" => {
                            out.push_str(&format!("<|user|>\n{}<|end|>\n", content.trim()));
                        }
                        "assistant" => {
                            out.push_str(&format!("<|assistant|>\n{}<|end|>\n", content.trim()));
                        }
                        _ => out.push_str(content.trim()),
                    }
                }
                out.push_str("<|assistant|>\n");
                out
            }
            ChatStyle::ChatML => {
                let mut out = String::new();
                if self.qwen_chatml_default_system
                    && messages.first().is_none_or(|(role, _)| role != "system")
                {
                    out.push_str(&format!(
                        "<|im_start|>system\n{QWEN_DEFAULT_SYSTEM}<|im_end|>\n"
                    ));
                }
                for (role, content) in messages {
                    out.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
                }
                out.push_str("<|im_start|>assistant\n");
                out
            }
            ChatStyle::Qwen3 => {
                let mut enable_thinking =
                    thinking_state.unwrap_or(!self.qwen_default_closed_thinking);
                if thinking_state.is_none() {
                    for (role, content) in messages {
                        if matches!(role.as_str(), "user" | "system")
                            && let Some(value) = thinking_control_value(self.chat_style, content)
                        {
                            enable_thinking = value;
                        }
                    }
                }

                let strip_controls = |content: &str| {
                    if thinking_state.is_some() {
                        content.to_string()
                    } else {
                        strip_thinking_controls(content)
                    }
                };
                let last_query_index = messages
                    .iter()
                    .rposition(|(role, content)| {
                        role == "user"
                            && !(content.trim().starts_with("<tool_response>")
                                && content.trim().ends_with("</tool_response>"))
                    })
                    .unwrap_or(usize::MAX);
                let mut out = String::new();
                let reasoning_instruction = if self.qwen_default_xhigh_reasoning && enable_thinking
                {
                    match reasoning_effort.unwrap_or("xhigh") {
                        "low" => {
                            "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration."
                        }
                        "medium" => "",
                        _ => QWEN_XHIGH_REASONING_INSTRUCTION,
                    }
                } else {
                    ""
                };
                let combines_first_system = !reasoning_instruction.is_empty()
                    && messages.first().is_some_and(|(role, _)| role == "system");
                if !reasoning_instruction.is_empty() {
                    out.push_str("<|im_start|>system\n");
                    out.push_str(reasoning_instruction);
                    if let Some((_, content)) =
                        messages.first().filter(|(role, _)| role == "system")
                    {
                        let content = strip_controls(content);
                        if !content.is_empty() {
                            out.push_str("\n\n");
                            out.push_str(&content);
                        }
                    }
                    out.push_str("<|im_end|>\n");
                }
                for (index, (role, content)) in messages.iter().enumerate() {
                    if combines_first_system && index == 0 {
                        continue;
                    }
                    let content = if role == "user" || role == "system" {
                        strip_controls(content)
                    } else if role == "assistant"
                        && (self.qwen_default_think_prefix || self.qwen_default_closed_thinking)
                    {
                        qwen3_assistant_history(
                            content,
                            self.qwen_default_xhigh_reasoning || index > last_query_index,
                        )
                    } else {
                        content.to_string()
                    };
                    out.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
                }
                out.push_str("<|im_start|>assistant\n");
                if !enable_thinking {
                    out.push_str("<think>\n\n</think>\n\n");
                } else if self.qwen_default_think_prefix || self.qwen_default_closed_thinking {
                    out.push_str("<think>\n");
                }
                out
            }
            ChatStyle::Glm5 => {
                // GLM's source tokenizer has no implicit BOS. Include the mask
                // here unless the GGUF explicitly inserts it during encoding.
                let mut out = if self.add_bos {
                    "<sop>".to_string()
                } else {
                    "[gMASK]<sop>".to_string()
                };
                out.push_str("<|system|>Reasoning Effort: ");
                out.push_str(match reasoning_effort.unwrap_or("max") {
                    "low" => "Low",
                    "high" => "High",
                    _ => "Max",
                });
                for (role, content) in messages {
                    match role.as_str() {
                        "system" => {
                            out.push_str("<|system|>");
                            out.push_str(content);
                        }
                        "user" => {
                            out.push_str("<|user|>");
                            out.push_str(content);
                        }
                        "assistant" => {
                            out.push_str("<|assistant|><think>");
                            if let Some((reasoning, _)) = content.split_once("</think>") {
                                out.push_str(
                                    reasoning.rsplit("<think>").next().unwrap_or(reasoning),
                                );
                                out.push_str("</think>");
                                out.push_str(content.rsplit("</think>").next().unwrap_or_default());
                            } else {
                                out.push_str("</think>");
                                out.push_str(content);
                            }
                        }
                        "tool" => {
                            out.push_str("<|observation|>");
                            out.push_str(content);
                        }
                        _ => {
                            out.push_str("<|user|>");
                            out.push_str(content.trim());
                            out.push('\n');
                        }
                    }
                }
                out.push_str("<|assistant|><think>");
                out
            }
            ChatStyle::Gemma4 => {
                let bos = self
                    .vocab
                    .get(self.bos_id as usize)
                    .map(String::as_str)
                    .unwrap_or("<bos>");
                let mut out = if self.add_bos {
                    String::new()
                } else {
                    bos.to_string()
                };
                for (role, content) in messages {
                    let role = if role == "assistant" {
                        "model"
                    } else {
                        role.as_str()
                    };
                    out.push_str(&format!("<|turn>{role}\n{content}<turn|>\n"));
                }
                out.push_str("<|turn>model\n<|channel>thought\n<channel|>");
                out
            }
            ChatStyle::Nemotron => {
                let mut enable_thinking = thinking_state.unwrap_or(true);
                if thinking_state.is_none() {
                    for (role, content) in messages {
                        if matches!(role.as_str(), "user" | "system")
                            && let Some(value) = thinking_control_value(self.chat_style, content)
                        {
                            enable_thinking = value;
                        }
                    }
                }

                let strip_controls = |content: &str| {
                    if thinking_state.is_some() {
                        content.to_string()
                    } else {
                        strip_thinking_controls(content)
                    }
                };
                let mut out = String::new();
                for (role, content) in messages {
                    let content = if role == "user" || role == "system" {
                        strip_controls(content)
                    } else {
                        content.to_string()
                    };
                    out.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
                }
                out.push_str("<|im_start|>assistant\n");
                if enable_thinking {
                    out.push_str("<think>\n");
                } else {
                    out.push_str("<think></think>");
                }
                out
            }
            ChatStyle::DeepSeekV4 => {
                let thinking = messages.iter().any(|(role, content)| {
                    (role == "user" || role == "system") && content.contains("/think")
                });
                let mut out = if self.add_bos {
                    String::new()
                } else {
                    self.vocab
                        .get(self.bos_id as usize)
                        .cloned()
                        .unwrap_or_else(|| "<｜begin▁of▁sentence｜>".to_string())
                };

                for (role, content) in messages {
                    match role.as_str() {
                        "system" => out.push_str(content.trim()),
                        "user" => {
                            out.push_str("<｜User｜>");
                            out.push_str(content);
                        }
                        "assistant" => {
                            out.push_str("<｜Assistant｜>");
                            if content.starts_with("<think>") {
                                out.push_str(content);
                            } else {
                                out.push_str(if thinking { "<think>" } else { "</think>" });
                                out.push_str(content);
                            }
                            out.push_str("<｜end▁of▁sentence｜>");
                        }
                        _ => out.push_str(content),
                    }
                }
                out.push_str("<｜Assistant｜>");
                if thinking {
                    out.push_str("<think>");
                } else {
                    out.push_str("</think>");
                }
                out
            }
            ChatStyle::DeepSeekR1 => {
                let mut out = if self.add_bos {
                    String::new()
                } else {
                    self.vocab
                        .get(self.bos_id as usize)
                        .cloned()
                        .unwrap_or_else(|| "<｜begin▁of▁sentence｜>".to_string())
                };

                let mut first_system = true;
                for (_, system_prompt) in messages.iter().filter(|(role, _)| role == "system") {
                    if !first_system {
                        out.push_str("\n\n");
                    }
                    out.push_str(system_prompt);
                    first_system = false;
                }

                for (role, content) in messages {
                    match role.as_str() {
                        "system" => {}
                        "user" => {
                            out.push_str("<｜User｜>");
                            out.push_str(content);
                        }
                        "assistant" => {
                            let final_answer = content
                                .rsplit_once("</think>")
                                .map(|(_, answer)| answer)
                                .unwrap_or(content);
                            out.push_str("<｜Assistant｜>");
                            out.push_str(final_answer);
                            out.push_str("<｜end▁of▁sentence｜>");
                        }
                        _ => out.push_str(content),
                    }
                }
                out.push_str("<｜Assistant｜><think>\n");
                out
            }
            ChatStyle::DeepSeek => {
                let mut out = String::new();
                for (role, content) in messages {
                    match role.as_str() {
                        "system" => {
                            out.push_str(content.trim());
                            out.push_str("\n\n");
                        }
                        "user" => {
                            out.push_str("User: ");
                            out.push_str(content.trim());
                            out.push_str("\n\n");
                        }
                        "assistant" => {
                            out.push_str("Assistant: ");
                            out.push_str(content.trim());
                            if let Some(eos) = self.vocab.get(self.eos_id as usize) {
                                out.push_str(eos);
                            }
                        }
                        _ => out.push_str(content.trim()),
                    }
                }
                out.push_str("Assistant:");
                out
            }
            ChatStyle::MiniMax => {
                let mut out = if self.add_bos {
                    String::new()
                } else {
                    self.vocab
                        .get(self.bos_id as usize)
                        .cloned()
                        .unwrap_or_else(|| "]~!b[".to_string())
                };

                let (system, start) = match messages.first() {
                    Some((role, content)) if role == "system" => (content.trim(), 1),
                    _ => (
                        "You are a helpful assistant. Your name is MiniMax-M2.5 and is built by MiniMax.",
                        0,
                    ),
                };
                out.push_str("]~b]system\n");
                out.push_str(system);
                out.push_str("[e~[\n");

                for (role, content) in messages.iter().skip(start) {
                    match role.as_str() {
                        "user" => {
                            out.push_str("]~b]user\n");
                            out.push_str(content);
                            out.push_str("[e~[\n");
                        }
                        "assistant" => {
                            let final_answer = content
                                .rsplit_once("</think>")
                                .map(|(_, answer)| answer.trim_start_matches('\n'))
                                .unwrap_or(content);
                            out.push_str("]~b]ai\n");
                            out.push_str(final_answer);
                            out.push_str("[e~[\n");
                        }
                        _ => out.push_str(content),
                    }
                }
                out.push_str("]~b]ai\n<think>\n");
                out
            }
            ChatStyle::Mistral => {
                let mut out = String::new();
                let (mut system, start) = match messages.first() {
                    Some((role, content)) if role == "system" => (Some(content.trim()), 1),
                    _ => (None, 0),
                };

                for (role, content) in messages.iter().skip(start) {
                    match role.as_str() {
                        "user" => {
                            out.push_str("[INST] ");
                            if let Some(system) = system.take() {
                                out.push_str(system);
                                out.push('\n');
                            }
                            out.push_str(content.trim());
                            out.push_str(" [/INST]");
                        }
                        "assistant" => {
                            out.push_str(content.trim());
                            if let Some(eos) = self.vocab.get(self.eos_id as usize) {
                                out.push_str(eos);
                            }
                        }
                        _ => out.push_str(content.trim()),
                    }
                }
                out
            }
            ChatStyle::Instruct => {
                let mut out = String::new();
                let mut prev_was_user = false;
                for (role, content) in messages {
                    match role.as_str() {
                        "system" => {
                            out.push_str(&format!("<<SYS>>\n{content}\n<</SYS>>\n\n"));
                        }
                        "user" => {
                            out.push_str(&format!("[INST] {content} [/INST]"));
                            prev_was_user = true;
                        }
                        "assistant" => {
                            if prev_was_user {
                                out.push_str(&format!(" {content} "));
                            } else {
                                out.push_str(content);
                            }
                            prev_was_user = false;
                        }
                        _ => {
                            out.push_str(content);
                            prev_was_user = false;
                        }
                    }
                }
                out
            }
            ChatStyle::Raw => messages
                .iter()
                .map(|(_, content)| content.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    /// Format a tool-capable conversation using the family turn formatter and
    /// the common JSON tool envelope.  No-tool conversations with no metadata
    /// delegate to the original formatter, which is an important compatibility
    /// guarantee for existing cached prompts and multimodal callers.
    pub fn format_chat_with_tools(
        &self,
        messages: &[ChatTemplateMessage],
        tools: &[ChatTemplateTool],
    ) -> Result<String> {
        let options = ChatTemplateOptions::new(tools.to_vec());
        self.format_chat_with_tool_options(messages, &options)
    }

    /// Format a conversation with function declarations, history, and the
    /// OpenAI tool-selection controls.  Native tool syntax and turn/reasoning framing follow the model family.
    /// Older families retain the bounded JSON call envelope.
    pub fn format_chat_with_tool_options(
        &self,
        messages: &[ChatTemplateMessage],
        options: &ChatTemplateOptions,
    ) -> Result<String> {
        validate_chat_template_input(
            messages,
            &options.tools,
            self.chat_style == ChatStyle::DeepSeekV41,
        )?;
        if let Some(effort) = options.reasoning_effort.as_deref() {
            let supported = match self.chat_style {
                ChatStyle::Glm5 => matches!(effort, "low" | "high" | "max"),
                ChatStyle::DeepSeekV41 => matches!(effort, "none" | "low" | "high" | "max"),
                ChatStyle::MuseSpark => matches!(effort, "low" | "medium" | "high" | "xhigh"),
                ChatStyle::Qwen3 if self.qwen_default_xhigh_reasoning => {
                    matches!(effort, "none" | "low" | "medium" | "xhigh")
                }
                ChatStyle::Qwen3 | ChatStyle::Nemotron => effort == "none",
                _ => false,
            };
            anyhow::ensure!(
                supported,
                "reasoning_effort {effort:?} is not supported by this model"
            );
        }
        if let ChatTemplateToolChoice::Named(name) = &options.tool_choice {
            anyhow::ensure!(
                name.len() <= MAX_CHAT_TEMPLATE_TOOL_NAME_BYTES
                    && if self.chat_style == ChatStyle::DeepSeekV41 {
                        deepseek41::valid_name(name)
                    } else {
                        !name.is_empty()
                            && name.bytes().all(|byte| {
                                byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')
                            })
                    },
                "named tool choice is invalid"
            );
            anyhow::ensure!(
                options.tools.iter().any(|tool| tool.name == *name),
                "named tool choice references an undeclared tool"
            );
        }
        if options.tool_choice.requires_tool() {
            anyhow::ensure!(
                !options.tools.is_empty(),
                "a required tool choice needs at least one declared tool"
            );
        }
        let call_count = messages
            .iter()
            .map(|message| message.tool_calls.len())
            .sum::<usize>();
        anyhow::ensure!(
            call_count <= MAX_CHAT_TEMPLATE_TOOL_CALLS,
            "tool-capable chat exceeds {MAX_CHAT_TEMPLATE_TOOL_CALLS} tool calls"
        );
        if self.chat_style == ChatStyle::MuseSpark {
            return format_muse_chat(messages, options, self.add_bos);
        }
        if self.chat_style == ChatStyle::DeepSeekV41 {
            return deepseek41::format_chat(messages, options);
        }
        if options.tools.is_empty()
            && options.reasoning_effort.is_none()
            && matches!(options.tool_choice, ChatTemplateToolChoice::None)
            && messages.iter().all(|message| {
                message.tool_calls.is_empty()
                    && message.tool_call_id.is_none()
                    && message.name.is_none()
                    && message.reasoning_content.is_none()
                    && !matches!(message.role.as_str(), "developer" | "tool" | "function")
            })
        {
            let tuples = messages
                .iter()
                .map(|message| (message.role.clone(), message.content.clone()))
                .collect::<Vec<_>>();
            return Ok(self.format_chat(&tuples));
        }
        let format = self.selected_tool_format(messages, options);
        let with_default_system;
        let messages = if self.qwen_chatml_default_system
            && messages
                .first()
                .is_none_or(|message| message.role != "system")
        {
            with_default_system =
                std::iter::once(ChatTemplateMessage::new("system", QWEN_DEFAULT_SYSTEM))
                    .chain(messages.iter().cloned())
                    .collect::<Vec<_>>();
            with_default_system.as_slice()
        } else {
            messages
        };
        let (messages, mut thinking_state) = normalize_tool_messages(
            messages,
            options,
            self.chat_style,
            format,
            !self.qwen_default_closed_thinking,
        )?;
        if options.reasoning_effort.as_deref() == Some("none") {
            thinking_state = Some(false);
        }
        Ok(self.format_chat_with_thinking_state(
            &messages,
            thinking_state,
            options.reasoning_effort.as_deref(),
        ))
    }

    fn tool_format(&self) -> ToolFormat {
        match self.chat_style {
            ChatStyle::Glm5 => ToolFormat::Glm,
            ChatStyle::DeepSeekV4 => ToolFormat::DeepSeek,
            ChatStyle::DeepSeekV41 => ToolFormat::DeepSeekV41,
            ChatStyle::MuseSpark => ToolFormat::Atem,
            ChatStyle::Qwen3 if self.qwen_xml_tools => ToolFormat::Qwen,
            _ => ToolFormat::Json,
        }
    }

    /// Report the syntax used by this exact prompt. Native history containing
    /// control tags needs JSON escaping, so the model's default alone is not
    /// enough to select a response reader or generation constraint.
    /// V4.1 always uses official DSML: its fallible formatter and grammar
    /// builder reject unrepresentable callable schemas instead of downgrading.
    pub fn selected_tool_format(
        &self,
        messages: &[ChatTemplateMessage],
        options: &ChatTemplateOptions,
    ) -> ToolFormat {
        if self.chat_style == ChatStyle::MuseSpark {
            return ToolFormat::Atem;
        }
        if self.chat_style == ChatStyle::DeepSeekV41 {
            return ToolFormat::DeepSeekV41;
        }
        let format = if options.native_tool_formats {
            self.tool_format()
        } else {
            ToolFormat::Json
        };
        if native_history_needs_json(messages, format)
            || (format != ToolFormat::Json
                && options
                    .tools
                    .iter()
                    .filter(|tool| match &options.tool_choice {
                        ChatTemplateToolChoice::Named(name) => tool.name == *name,
                        _ => true,
                    })
                    .any(|tool| !native_tool_schema_safe(tool)))
        {
            ToolFormat::Json
        } else {
            format
        }
    }
}

fn muse_escape(value: &str) -> String {
    value
        .replace("<|", "< |")
        .replace("<image>", "< image>")
        .replace("<video>", "< video>")
        .replace("<audio>", "< audio>")
}

fn muse_attribute(value: &str) -> String {
    muse_escape(value)
        .replace(['\r', '\n'], " ")
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Match the released Muse Spark template's native recipients and ATEM history.
/// Its default xhigh effort is a model instruction, not a token budget.
fn format_muse_chat(
    messages: &[ChatTemplateMessage],
    options: &ChatTemplateOptions,
    add_bos: bool,
) -> Result<String> {
    let mut parts = vec![
        "You are a helpful AI assistant.".to_owned(),
        "Knowledge cutoff: 2026-01-04.".to_owned(),
        format!("Current date: {}.", chrono::Local::now().format("%Y-%m-%d")),
    ];
    let first_system = messages
        .first()
        .filter(|m| matches!(m.role.as_str(), "system" | "developer"));
    let instructions = first_system.map(|m| {
        muse_escape(&m.content)
            .replace("Reasoning effort", "Reasoning strength")
            .replace("Reasoning Effort", "Reasoning Strength")
            .replace("reasoning effort", "reasoning strength")
            .replace("REASONING EFFORT", "REASONING STRENGTH")
    });
    if let Some(text) = instructions.as_deref().filter(|s| !s.is_empty()) {
        parts.push(text.to_owned());
    }
    let tools = options
        .tools
        .iter()
        .filter(|tool| match &options.tool_choice {
            ChatTemplateToolChoice::None => false,
            ChatTemplateToolChoice::Named(name) => tool.name == *name,
            _ => true,
        })
        .collect::<Vec<_>>();
    if !tools.is_empty() {
        let mut definitions = "In this environment you have access to a set of tools you can use to answer the user's question.\n\nYou can invoke a function by writing a \"<atem:function_calls>\" block like the following as part of your reply to the user:\n<atem:function_calls>\n<atem:invoke name=\"$FUNCTION_NAME\">\n<atem:parameter name=\"$PARAMETER_NAME\">$PARAMETER_VALUE</atem:parameter>\n...\n</atem:invoke>\n</atem:function_calls>\n\nString and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.\nHere are the functions available in JSONSchema format:\n// Tool metadata\n{\"name\": \"\", \"description\": \"\"}\n// Function schemas".to_owned();
        for tool in &tools {
            let schema: serde_json::Value = tool
                .parameters_json
                .as_deref()
                .map(serde_json::from_str)
                .transpose()?
                .unwrap_or_else(|| serde_json::json!({}));
            definitions.push_str(&format!(
                "\n{{\"name\": {}, \"description\": {}, \"parameters\": {}}}",
                muse_escape(&muse_json(&serde_json::Value::String(tool.name.clone()))),
                muse_escape(&muse_json(&serde_json::Value::String(
                    tool.description.clone().unwrap_or_default()
                ))),
                muse_escape(&muse_json(&schema))
            ));
        }
        definitions.push_str("\n\nHere are examples of how to call functions in the tool set. To make parallel tool calls, emit each call as its own message (one tool call per message) in consecutive messages, as shown below:\n(If the tool namespace is not specified, invoke the function directly as `example_function_name_1` rather than `example_tool_name_1.example_function_name_1`)\n\nto=example_tool_name_1.example_function_name_1\n\n<atem:function_calls>\n<atem:invoke name=\"example_tool_name_1.example_function_name_1\">\n<atem:parameter name=\"example_parameter_1\">value_1</atem:parameter>\n</atem:invoke>\n</atem:function_calls>\n\nto=example_tool_name_2.example_function_name_2\n\n<atem:function_calls>\n<atem:invoke name=\"example_tool_name_2.example_function_name_2\">\n<atem:parameter name=\"example_parameter_2\">This is the value for the second parameter\nthat can span\n\"multiple\" lines\n</atem:parameter>\n<atem:parameter name=\"example_parameter_3\">value_3</atem:parameter>\n</atem:invoke>\n</atem:function_calls>");
        parts.push(definitions);
    }
    if !instructions
        .as_deref()
        .unwrap_or_default()
        .to_ascii_lowercase()
        .contains("reasoning strength:")
    {
        let strength = match options.reasoning_effort.as_deref().unwrap_or("xhigh") {
            "low" => 8,
            "medium" => 32,
            "high" => 256,
            "xhigh" => 512,
            other => anyhow::bail!("unsupported Muse Spark reasoning_effort {other:?}"),
        };
        parts.push(format!("Reasoning strength: {strength}."));
    }
    let mut recipients = "Use the appropriate recipient for each message:\n\n- \"self\": private reasoning and tool planning.\n- \"commentary\": user-visible intermediate updates while the assistant will continue working, including messages sent to the user before or between tool calls.\n".to_owned();
    if !tools.is_empty() {
        recipients.push_str("- \"tool\" or \"[tool name]\": the name of the tool to call.\n");
    }
    recipients.push_str("- \"user\": messages that end the assistant turn, such as a completed response or clarification question that waits for the user's reply; do not use for pre-tool updates or partial responses.\n\n# Valid recipients: \"self\", \"commentary\", ");
    if let Some(tool) = tools.first() {
        recipients.push_str(&format!("\"[func_name] (e.g., {})\", ", tool.name));
    }
    recipients.push_str("\"user\".");
    parts.push(recipients);
    if options.tool_choice.requires_tool() {
        parts.push("Call a tool before completing this turn.".into());
    }
    if !tools.is_empty() && !options.parallel_tool_calls {
        parts.push("Call at most one tool in this turn.".into());
    }
    let mut out = if add_bos {
        String::new()
    } else {
        "<|begin_of_text|>".to_owned()
    };
    out.push_str("<|start|>system<|message|>");
    out.push_str(&parts.join("\n\n"));
    out.push_str("<|eot|>");
    for (index, message) in messages
        .iter()
        .enumerate()
        .skip(usize::from(first_system.is_some()))
    {
        let body = muse_escape(&message.content);
        match message.role.as_str() {
            "system" | "developer" | "user" => {
                out.push_str(&format!(
                    "<|start|>{}<|message|>{body}<|eot|>",
                    message.role
                ));
            }
            "assistant" => {
                if let Some(reasoning) = message
                    .reasoning_content
                    .as_deref()
                    .filter(|r| !r.is_empty())
                {
                    out.push_str(&format!(
                        "<|start|>assistant to=self<|message|>{}<|eom|>",
                        muse_escape(reasoning)
                    ));
                }
                if message.tool_calls.is_empty() {
                    if !body.is_empty()
                        || message
                            .reasoning_content
                            .as_deref()
                            .is_none_or(str::is_empty)
                    {
                        out.push_str(&format!(
                            "<|start|>assistant to=user<|message|>{body}<|eot|>"
                        ));
                    }
                } else {
                    if !body.is_empty() {
                        out.push_str(&format!(
                            "<|start|>assistant to=commentary<|message|>{body}<|eom|>"
                        ));
                    }
                    for (call_index, call) in message.tool_calls.iter().enumerate() {
                        let arguments: serde_json::Value = serde_json::from_str(
                            &call.arguments_json,
                        )
                        .context("Muse Spark tool history arguments must be JSON objects")?;
                        let arguments = arguments
                            .as_object()
                            .context("Muse Spark tool history arguments must be JSON objects")?;
                        out.push_str(&format!("<|start|>assistant to={}<|message|><atem:function_calls>\n<atem:invoke name=\"{}\">\n", call.name, muse_attribute(&call.name)));
                        for (key, value) in arguments {
                            let value = value
                                .as_str()
                                .map(str::to_owned)
                                .unwrap_or_else(|| muse_json(value));
                            // ATEM has no string escape for its own closing tag.
                            // Refuse an ambiguous replay instead of changing file text.
                            anyhow::ensure!(
                                !value.contains("</atem:parameter>"),
                                "Muse Spark history has an ambiguous ATEM parameter delimiter"
                            );
                            out.push_str(&format!(
                                "<atem:parameter name=\"{}\">{}</atem:parameter>\n",
                                muse_attribute(key),
                                muse_escape(&value)
                            ));
                        }
                        out.push_str("</atem:invoke>\n</atem:function_calls>");
                        let continue_message = call_index + 1 < message.tool_calls.len()
                            || messages
                                .get(index + 1)
                                .is_some_and(|m| m.role == "assistant");
                        out.push_str(if continue_message {
                            "<|eom|>"
                        } else {
                            "<|eot|>"
                        });
                    }
                }
            }
            "tool" | "function" => {
                let name = message
                    .name
                    .as_deref()
                    .or_else(|| {
                        message.tool_call_id.as_deref().and_then(|id| {
                            messages[..index]
                                .iter()
                                .rev()
                                .flat_map(|m| &m.tool_calls)
                                .find(|call| call.id.as_deref() == Some(id))
                                .map(|call| call.name.as_str())
                        })
                    })
                    .context("Muse Spark tool result needs a name or matching tool_call_id")?;
                out.push_str(&format!("<|start|>tool {name}<|message|><tool_output name=\"{}\">\n{body}\n</tool_output><|eot|>", muse_attribute(name)));
            }
            role => anyhow::bail!("unsupported Muse Spark chat role {role:?}"),
        }
    }
    out.push_str("<|start|>assistant");
    anyhow::ensure!(
        out.len() <= MAX_CHAT_TEMPLATE_PROMPT_BYTES,
        "Muse Spark prompt exceeds the size limit"
    );
    Ok(out)
}

// Jinja's `tojson` uses sorted keys, spaces and HTML-safe strings.
fn muse_json(value: &serde_json::Value) -> String {
    use serde_json::Value;
    match value {
        Value::Array(values) => format!(
            "[{}]",
            values.iter().map(muse_json).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(k, v)| format!("{}: {}", muse_json(&Value::String(k.clone())), muse_json(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::String(_) => {
            let encoded = serde_json::to_string(value)
                .unwrap()
                .replace('&', "\\u0026")
                .replace('<', "\\u003c")
                .replace('>', "\\u003e")
                .replace('\'', "\\u0027");
            let mut ascii = String::new();
            for character in encoded.chars() {
                if character.is_ascii() {
                    ascii.push(character);
                } else {
                    let mut units = [0u16; 2];
                    for unit in character.encode_utf16(&mut units) {
                        ascii.push_str(&format!("\\u{unit:04x}"));
                    }
                }
            }
            ascii
        }
        _ => value.to_string(),
    }
}

fn validate_chat_template_input(
    messages: &[ChatTemplateMessage],
    tools: &[ChatTemplateTool],
    deepseek41: bool,
) -> Result<()> {
    anyhow::ensure!(
        !messages.is_empty(),
        "tool-capable chat must contain at least one message"
    );
    anyhow::ensure!(
        messages.len() <= MAX_CHAT_TEMPLATE_MESSAGES,
        "tool-capable chat exceeds {MAX_CHAT_TEMPLATE_MESSAGES} messages"
    );
    anyhow::ensure!(
        tools.len() <= MAX_CHAT_TEMPLATE_TOOLS,
        "tool-capable chat exceeds {MAX_CHAT_TEMPLATE_TOOLS} tools"
    );

    let mut total_bytes = 0usize;
    for message in messages {
        anyhow::ensure!(
            !message.role.trim().is_empty(),
            "chat message role is empty"
        );
        anyhow::ensure!(message.role.len() <= 64, "chat message role is too long");
        anyhow::ensure!(
            !message.role.contains('\0'),
            "chat message role contains NUL"
        );
        anyhow::ensure!(
            message.content.len() <= MAX_CHAT_TEMPLATE_ARGUMENT_BYTES,
            "chat message content is too large"
        );
        anyhow::ensure!(
            !message.content.contains('\0'),
            "chat message content contains NUL"
        );
        if let Some(reasoning) = message.reasoning_content.as_deref() {
            anyhow::ensure!(
                message.role == "assistant"
                    && reasoning.len() <= MAX_CHAT_TEMPLATE_PROMPT_BYTES
                    && !reasoning.contains('\0'),
                "invalid assistant reasoning"
            );
            total_bytes = total_bytes.saturating_add(reasoning.len());
        }
        if let Some(name) = message.name.as_deref() {
            anyhow::ensure!(
                name.len() <= MAX_CHAT_TEMPLATE_TOOL_NAME_BYTES
                    && !name.is_empty()
                    && name.bytes().all(|byte| {
                        byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')
                    }),
                "chat message name is invalid"
            );
        }
        if let Some(id) = message.tool_call_id.as_deref() {
            anyhow::ensure!(
                id.len() <= MAX_CHAT_TEMPLATE_TOOL_ID_BYTES && !id.is_empty(),
                "tool call ID is too long or empty"
            );
            anyhow::ensure!(!id.contains('\0'), "tool call ID contains NUL");
        }
        for call in &message.tool_calls {
            anyhow::ensure!(
                call.name.len() <= MAX_CHAT_TEMPLATE_TOOL_NAME_BYTES
                    && if deepseek41 {
                        deepseek41::valid_name(&call.name)
                    } else {
                        !call.name.is_empty()
                            && call.name.bytes().all(|byte| {
                                byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')
                            })
                    },
                "tool call name is invalid"
            );
            if let Some(id) = call.id.as_deref() {
                anyhow::ensure!(
                    id.len() <= MAX_CHAT_TEMPLATE_TOOL_ID_BYTES && !id.is_empty(),
                    "tool call ID is too long or empty"
                );
                anyhow::ensure!(!id.contains('\0'), "tool call ID contains NUL");
            }
            anyhow::ensure!(
                call.arguments_json.len() <= MAX_CHAT_TEMPLATE_ARGUMENT_BYTES,
                "tool call arguments are too large"
            );
            anyhow::ensure!(
                !call.arguments_json.contains('\0'),
                "tool call arguments contain NUL"
            );
            total_bytes = total_bytes
                .saturating_add(call.name.len())
                .saturating_add(call.arguments_json.len())
                .saturating_add(call.id.as_deref().map_or(0, str::len));
        }
        total_bytes = total_bytes
            .saturating_add(message.role.len())
            .saturating_add(message.content.len())
            .saturating_add(message.tool_call_id.as_deref().map_or(0, str::len));
    }

    for tool in tools {
        anyhow::ensure!(
            tool.name.len() <= MAX_CHAT_TEMPLATE_TOOL_NAME_BYTES
                && if deepseek41 {
                    deepseek41::valid_name(&tool.name)
                } else {
                    !tool.name.is_empty()
                        && tool
                            .name
                            .bytes()
                            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
                },
            "tool name is invalid"
        );
        if let Some(description) = tool.description.as_deref() {
            anyhow::ensure!(
                description.len() <= MAX_CHAT_TEMPLATE_DESCRIPTION_BYTES,
                "tool description is too large"
            );
            anyhow::ensure!(!description.contains('\0'), "tool description contains NUL");
        }
        if let Some(schema) = tool.parameters_json.as_deref() {
            anyhow::ensure!(
                schema.len() <= MAX_CHAT_TEMPLATE_SCHEMA_BYTES,
                "tool parameter schema is too large"
            );
            let value: serde_json::Value =
                serde_json::from_str(schema).context("tool parameter schema is not valid JSON")?;
            anyhow::ensure!(value.is_object(), "tool parameter schema must be an object");
            total_bytes = total_bytes.saturating_add(schema.len());
        }
        total_bytes = total_bytes
            .saturating_add(tool.name.len())
            .saturating_add(tool.description.as_deref().map_or(0, str::len));
    }
    anyhow::ensure!(
        total_bytes <= MAX_CHAT_TEMPLATE_PROMPT_BYTES,
        "tool-capable chat exceeds the prompt size limit"
    );
    Ok(())
}

fn tool_definition_json(tool: &ChatTemplateTool) -> Result<String> {
    let mut function = serde_json::Map::new();
    function.insert("name".into(), serde_json::Value::String(tool.name.clone()));
    if let Some(description) = tool.description.as_deref() {
        function.insert(
            "description".into(),
            serde_json::Value::String(description.to_owned()),
        );
    }
    let parameters = tool
        .parameters_json
        .as_deref()
        .map(serde_json::from_str)
        .transpose()
        .context("tool parameter schema is not valid JSON")?
        .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));
    function.insert("parameters".into(), parameters);
    Ok(escape_tool_envelope_json(serde_json::to_string(
        &serde_json::json!({
            "type": "function",
            "function": function,
        }),
    )?))
}

fn tool_arguments_value(arguments: &str) -> serde_json::Value {
    serde_json::from_str(arguments)
        .unwrap_or_else(|_| serde_json::Value::String(arguments.to_owned()))
}

fn tool_call_json(call: &ChatTemplateToolCall) -> Result<String> {
    let mut value = serde_json::Map::new();
    if let Some(id) = call.id.as_deref() {
        value.insert("id".into(), serde_json::Value::String(id.to_owned()));
    }
    value.insert("name".into(), serde_json::Value::String(call.name.clone()));
    value.insert(
        "arguments".into(),
        tool_arguments_value(&call.arguments_json),
    );
    Ok(escape_tool_envelope_json(serde_json::to_string(&value)?))
}

/// Escape markup delimiters inside JSON that is embedded in one of the
/// tokenizer's XML-like tool envelopes.  JSON permits literal `<` and `>`
/// characters in strings, but leaving them raw would let an untrusted tool
/// description, argument, or result terminate the surrounding envelope and
/// inject a second control message into the model prompt.  `\u003c` and
/// `\u003e` are decoded back to the original characters by any JSON parser.
fn escape_tool_envelope_json(json: String) -> String {
    json.replace('<', "\\u003c").replace('>', "\\u003e")
}

fn inject_tool_instruction(
    messages: &[ChatTemplateMessage],
    instruction: &str,
) -> Vec<ChatTemplateMessage> {
    let mut messages = messages.to_vec();
    if let Some(first) = messages.first_mut()
        && matches!(first.role.as_str(), "system" | "developer")
    {
        if first.content.is_empty() {
            first.content.push_str(instruction);
        } else {
            first.content.push_str("\n\n");
            first.content.push_str(instruction);
        }
    } else {
        messages.insert(0, ChatTemplateMessage::new("system", instruction));
    }
    messages
}

fn shared_token_prefix_len(tokens: &[u32], candidate: &[u32]) -> Option<usize> {
    let count = tokens
        .iter()
        .zip(candidate)
        .take_while(|(a, b)| a == b)
        .count();
    (count >= 64 && count < tokens.len()).then_some(count)
}

/// Call markup selected from the model's source template and request history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFormat {
    /// A JSON call object inside tool_call tags.
    Json,
    /// Qwen function and parameter tags inside tool_call tags.
    Qwen,
    /// GLM named arguments inside tool_call tags.
    Glm,
    /// DeepSeek's DSML tool_calls, invoke and parameter tags.
    DeepSeek,
    /// DeepSeek V4.1's spaced calls, invoke and parameter tags.
    DeepSeekV41,
    /// Muse Spark's ATEM calls, one recipient message per call.
    Atem,
}

fn contains_native_control(text: &str) -> bool {
    [
        "<tool_call",
        "</tool_call",
        "<tool_response",
        "</tool_response",
        "<tool_result",
        "</tool_result",
        "<think",
        "</think",
        "<function=",
        "</function",
        "<parameter=",
        "</parameter",
        "<arg_key",
        "</arg_key",
        "<arg_value",
        "</arg_value",
        "<|",
        "</|",
        "<｜",
        "</｜",
    ]
    .iter()
    .any(|marker| text.contains(marker))
}

fn native_history_needs_json(messages: &[ChatTemplateMessage], format: ToolFormat) -> bool {
    if format == ToolFormat::Json {
        return false;
    }
    messages.iter().any(|message| {
        contains_native_control(&message.content)
            || message.tool_calls.iter().any(|call| {
                let arguments = serde_json::from_str::<serde_json::Value>(&call.arguments_json)
                    .ok()
                    .filter(serde_json::Value::is_object)
                    .unwrap_or_else(|| serde_json::json!({"arguments":call.arguments_json}));
                arguments.as_object().unwrap().iter().any(|(key, value)| {
                    key.contains(['<', '>', '\0'])
                        || (format == ToolFormat::DeepSeek && key.contains('"'))
                        || value_contains_native_control(value)
                })
            })
    })
}

/// Reject schemas that cannot be represented by the official raw-string DSML
/// spelling. Only callable tools matter; inactive declarations remain history.
pub(crate) fn validate_deepseek41_tool_schemas(options: &ChatTemplateOptions) -> Result<()> {
    for tool in options
        .tools
        .iter()
        .filter(|tool| match &options.tool_choice {
            ChatTemplateToolChoice::None => false,
            ChatTemplateToolChoice::Named(name) => tool.name == *name,
            _ => true,
        })
    {
        anyhow::ensure!(
            native_tool_schema_safe_for(tool, true),
            "unsupported DeepSeek V4.1 native schema for callable tool {:?}: \
             requires fixed parameter names and representable raw strings \
             (no patterns or DSML closing delimiter)",
            tool.name
        );
    }
    Ok(())
}

// Native fields distinguish raw strings from JSON through the declared type.
// A finite field order cannot express an open object or a string/JSON union.
// Use the escaped JSON call format whenever that distinction is not exact.
fn native_tool_schema_safe(tool: &ChatTemplateTool) -> bool {
    native_tool_schema_safe_for(tool, false)
}

fn native_tool_schema_safe_for(tool: &ChatTemplateTool, deepseek41: bool) -> bool {
    let Some(schema) = tool
        .parameters_json
        .as_deref()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
    else {
        return deepseek41 && tool.parameters_json.is_none();
    };
    let Some(root) = schema.as_object() else {
        return false;
    };
    if root.get("type").and_then(serde_json::Value::as_str) != Some("object")
        || (!deepseek41
            && root.get("additionalProperties") != Some(&serde_json::Value::Bool(false)))
        || root.keys().any(|key| {
            !matches!(
                key.as_str(),
                "type"
                    | "properties"
                    | "required"
                    | "additionalProperties"
                    | "description"
                    | "title"
                    | "$schema"
                    | "default"
                    | "examples"
            )
        })
    {
        return false;
    }
    let Some(properties) = root
        .get("properties")
        .and_then(serde_json::Value::as_object)
    else {
        return root
            .get("required")
            .is_none_or(|required| required.as_array().is_some_and(Vec::is_empty));
    };
    if root.get("required").is_some_and(|required| {
        required.as_array().is_none_or(|keys| {
            keys.iter().any(|key| {
                key.as_str()
                    .is_none_or(|name| !properties.contains_key(name))
            })
        })
    }) {
        return false;
    }
    properties.iter().all(|(name, property)| {
        if name.is_empty()
            || name.contains(['<', '>', '\0', '"', '\n', '\r'])
            || (deepseek41 && name.chars().any(char::is_control))
        {
            return false;
        }
        let nonstring = |kind: &str| {
            matches!(
                kind,
                "object" | "array" | "number" | "integer" | "boolean" | "null"
            )
        };
        match property.get("type") {
            Some(serde_json::Value::String(kind)) if kind == "string" => {
                property.as_object().is_some_and(|schema| {
                    schema.keys().all(|key| {
                        matches!(
                            key.as_str(),
                            "type"
                                | "const"
                                | "enum"
                                | "minLength"
                                | "maxLength"
                                | "description"
                                | "title"
                                | "default"
                                | "examples"
                                | "deprecated"
                                | "readOnly"
                                | "writeOnly"
                        )
                    })
                }) && !((property.get("const").is_some() || property.get("enum").is_some())
                    && (property.get("minLength").is_some() || property.get("maxLength").is_some()))
                    && ["minLength", "maxLength"].iter().all(|key| {
                        property
                            .get(*key)
                            .is_none_or(|value| value.as_u64().is_some())
                    })
                    && (!deepseek41
                        || (property
                            .get("const")
                            .is_none_or(serde_json::Value::is_string)
                            && property.get("enum").is_none_or(|values| {
                                values.as_array().is_some_and(|values| {
                                    !values.is_empty()
                                        && values.iter().all(serde_json::Value::is_string)
                                })
                            })
                            && !(property.get("const").is_some()
                                && property.get("enum").is_some())
                            && property
                                .get("maxLength")
                                .and_then(serde_json::Value::as_u64)
                                .is_none_or(|max| {
                                    max >= property
                                        .get("minLength")
                                        .and_then(serde_json::Value::as_u64)
                                        .unwrap_or(0)
                                })))
                    && !["const", "enum"].iter().any(|key| {
                        property.get(*key).is_some_and(|value| {
                            if deepseek41 {
                                value
                                    .as_str()
                                    .is_some_and(|text| text.contains("</｜DSML｜ parameter>"))
                                    || value.as_array().is_some_and(|values| {
                                        values.iter().any(|value| {
                                            value.as_str().is_some_and(|text| {
                                                text.contains("</｜DSML｜ parameter>")
                                            })
                                        })
                                    })
                            } else {
                                value_contains_native_control(value)
                            }
                        })
                    })
            }
            Some(serde_json::Value::String(kind)) => nonstring(kind),
            Some(serde_json::Value::Array(kinds)) => {
                !kinds.is_empty()
                    && kinds
                        .iter()
                        .all(|kind| kind.as_str().is_some_and(nonstring))
            }
            _ => false,
        }
    })
}

fn value_contains_native_control(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::String(text) => contains_native_control(text),
        serde_json::Value::Array(values) => values.iter().any(value_contains_native_control),
        serde_json::Value::Object(values) => values.iter().any(|(key, value)| {
            contains_native_control(key) || value_contains_native_control(value)
        }),
        _ => false,
    }
}

fn native_tool_call(call: &ChatTemplateToolCall, format: ToolFormat) -> Result<String> {
    if format == ToolFormat::Json {
        return Ok(format!("<tool_call>{}</tool_call>", tool_call_json(call)?));
    }
    // Legacy clients can replay a failed call with malformed arguments and
    // a tool error. Keep that text as data so the model can recover; only new
    // generated calls must satisfy the response boundary's argument checks.
    let arguments: serde_json::Value = serde_json::from_str(&call.arguments_json)
        .ok()
        .filter(serde_json::Value::is_object)
        .unwrap_or_else(|| serde_json::json!({"arguments":call.arguments_json}));
    let arguments = arguments.as_object().unwrap();
    let mut out = match format {
        ToolFormat::Qwen => format!("<tool_call>\n<function={}>\n", call.name),
        ToolFormat::Glm => format!("<tool_call>{}", call.name),
        ToolFormat::DeepSeek => format!("<｜DSML｜invoke name=\"{}\">\n", call.name),
        ToolFormat::Json | ToolFormat::Atem | ToolFormat::DeepSeekV41 => unreachable!(),
    };
    for (key, value) in arguments {
        anyhow::ensure!(
            !key.contains(['<', '>', '\0']),
            "tool argument name contains a control delimiter"
        );
        let is_string = value.is_string();
        let value = if let Some(value) = value.as_str() {
            anyhow::ensure!(
                !contains_native_control(value),
                "native tool string contains a control tag; use the JSON format"
            );
            value.to_owned()
        } else {
            escape_tool_envelope_json(value.to_string())
        };
        match format {
            ToolFormat::DeepSeek => {
                anyhow::ensure!(!key.contains('"'), "DSML argument name contains a quote");
                out.push_str(&format!("<｜DSML｜parameter name=\"{key}\" string=\"{is_string}\">{value}</｜DSML｜parameter>\n"));
            }
            ToolFormat::Qwen => {
                out.push_str(&format!("<parameter={key}>\n{value}\n</parameter>\n"))
            }
            ToolFormat::Glm => out.push_str(&format!(
                "<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"
            )),
            ToolFormat::Json | ToolFormat::Atem | ToolFormat::DeepSeekV41 => unreachable!(),
        }
    }
    if format == ToolFormat::Qwen {
        out.push_str("</function>\n");
    }
    out.push_str(if format == ToolFormat::DeepSeek {
        "</｜DSML｜invoke>"
    } else {
        "</tool_call>"
    });
    Ok(out)
}

// The native templates associate a result by call order. Public IDs remain
// intact in the protocol; they need not become tokens in the native prompt.
fn order_tool_results(messages: Vec<ChatTemplateMessage>) -> Vec<ChatTemplateMessage> {
    let mut ordered = Vec::with_capacity(messages.len());
    let mut source = messages.into_iter().peekable();
    while let Some(message) = source.next() {
        if message.role == "assistant" && !message.tool_calls.is_empty() {
            let calls = message.tool_calls.clone();
            ordered.push(message);
            let mut results = Vec::new();
            while source.peek().is_some_and(|m| m.role == "tool") {
                results.push(source.next().unwrap());
            }
            results.sort_by_key(|result| {
                calls
                    .iter()
                    .position(|call| call.id == result.tool_call_id)
                    .unwrap_or(usize::MAX)
            });
            ordered.extend(results);
        } else {
            ordered.push(message);
        }
    }
    ordered
}

fn canonical_tool_instruction(options: &ChatTemplateOptions, format: ToolFormat) -> Result<String> {
    let definitions = options
        .tools
        .iter()
        .map(|tool| {
            let json = tool_definition_json(tool)?;
            if matches!(format, ToolFormat::Glm | ToolFormat::DeepSeek) {
                let value: serde_json::Value = serde_json::from_str(&json)?;
                Ok(serde_json::to_string(&value["function"])?)
            } else {
                Ok(json)
            }
        })
        .collect::<Result<Vec<_>>>()?
        .join("\n");
    let choice = match &options.tool_choice {
        ChatTemplateToolChoice::None => "Do not call tools; answer the user directly.".to_owned(),
        ChatTemplateToolChoice::Auto => {
            "You may call zero or more tools when they help answer the user.".to_owned()
        }
        ChatTemplateToolChoice::Required => {
            "You must call at least one tool before answering the user.".to_owned()
        }
        ChatTemplateToolChoice::Named(name) => {
            format!("You must call the named tool `{name}` before answering the user.")
        }
    };
    let parallel = if options.parallel_tool_calls {
        "Multiple tool calls may be emitted in one turn."
    } else {
        "Emit at most one tool call in a turn."
    };
    let end_turn = if options.parallel_tool_calls {
        "End the turn after the calls."
    } else {
        "If you call a tool, emit exactly one call and end the turn. Wait for its result before another call."
    };
    if format != ToolFormat::Json {
        let example = match format {
            ToolFormat::DeepSeek => {
                "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"function_name\">\n<｜DSML｜parameter name=\"argument_name\" string=\"true\">value</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
            }
            ToolFormat::Qwen => {
                "<tool_call>\n<function=function_name>\n<parameter=argument_name>\nvalue\n</parameter>\n</function>\n</tool_call>"
            }
            ToolFormat::Glm => {
                "<tool_call>function_name<arg_key>argument_name</arg_key><arg_value>value</arg_value></tool_call>"
            }
            ToolFormat::Json | ToolFormat::Atem | ToolFormat::DeepSeekV41 => unreachable!(),
        };
        let closing = match format {
            ToolFormat::Qwen => {
                "Every <parameter=...> needs its own </parameter>. Close the last parameter before </function>, then close </tool_call>. Never omit a closing tag."
            }
            ToolFormat::Glm => {
                "Every <arg_key> and <arg_value> needs its matching closing tag. Close the last argument value before </tool_call>."
            }
            ToolFormat::DeepSeek => {
                "Every invoke must be inside <｜DSML｜tool_calls></｜DSML｜tool_calls>. Close every parameter, then the invoke, then the outer tool_calls block. Never emit an invoke without the outer block."
            }
            ToolFormat::Json | ToolFormat::Atem | ToolFormat::DeepSeekV41 => unreachable!(),
        };
        let types = if format == ToolFormat::DeepSeek {
            "Set string=\"true\" for strings and string=\"false\" for other values."
        } else {
            ""
        };
        let final_answer = if format == ToolFormat::Qwen
            && matches!(options.tool_choice, ChatTemplateToolChoice::None)
        {
            "\nClose </think> before the visible final answer. Do not put the final answer inside <think>."
        } else {
            ""
        };
        return Ok(format!(
            "# Tools\n\n{choice}\n{parallel}\n\n<tools>\n{definitions}\n</tools>\n\nWhen calling a function, use this exact format:\n{example}\n{closing}\nInclude all required arguments. Use JSON for arrays, objects, numbers, booleans and null. Write string values as raw text with actual newlines. Do not wrap strings in JSON quotes or add JSON backslash escapes. {types} {end_turn}{final_answer}"
        ));
    }
    let json_strings = concat!(
        r#"Tool calls must be valid JSON. Inside JSON string values, encode newlines as \n, quotes as \", and backslashes as \\. Never put an unescaped newline or quote inside a JSON string."#,
        "\nClose the arguments object with }, then the outer call object with }, before </tool_call>. After these braces, emit only </tool_call>; do not add native function, parameter, or invoke closing tags. Tags inside JSON string values are file text."
    );
    if options.tools.is_empty() {
        return Ok(format!(
            "# Tools\n\n{choice}\n{parallel}\n\nTool calls, when allowed, use one JSON object inside <tool_call></tool_call> tags:\n<tool_call>{{\"name\": \"function-name\", \"arguments\": {{}}}}</tool_call>\n{json_strings}\n{end_turn}"
        ));
    }
    Ok(format!(
        "# Tools\n\n{choice}\n{parallel}\n\nAvailable function declarations are enclosed in <tools></tools>:\n<tools>\n{definitions}\n</tools>\n\nWhen calling a function, emit one JSON object inside <tool_call></tool_call> tags. The name must be declared and arguments must be a JSON object:\n<tool_call>{{\"name\": \"function-name\", \"arguments\": {{}}}}</tool_call>\n{json_strings}\n{end_turn}"
    ))
}

fn tool_response_json(message: &ChatTemplateMessage) -> Result<String> {
    let mut response = serde_json::Map::new();
    if let Some(id) = message.tool_call_id.as_deref() {
        response.insert(
            "tool_call_id".into(),
            serde_json::Value::String(id.to_owned()),
        );
    }
    if let Some(name) = message.name.as_deref() {
        response.insert("name".into(), serde_json::Value::String(name.to_owned()));
    }
    response.insert(
        "content".into(),
        serde_json::Value::String(message.content.clone()),
    );
    Ok(escape_tool_envelope_json(serde_json::to_string(&response)?))
}

type NormalizedToolMessages = (Vec<(String, String)>, Option<bool>);

fn normalize_tool_messages(
    messages: &[ChatTemplateMessage],
    options: &ChatTemplateOptions,
    chat_style: ChatStyle,
    format: ToolFormat,
    default_thinking: bool,
) -> Result<NormalizedToolMessages> {
    let mut source = messages.to_vec();
    if format == ToolFormat::Json {
        let mut ids = HashMap::new();
        let mut next = 0;
        for message in &mut source {
            for call in &mut message.tool_calls {
                if let Some(id) = call.id.as_mut() {
                    let short = format!("call_{next}");
                    next += 1;
                    ids.insert(id.clone(), short.clone());
                    *id = short;
                }
            }
            if let Some(id) = message.tool_call_id.as_mut()
                && let Some(short) = ids.get(id)
            {
                *id = short.clone();
            }
        }
    }

    let mut thinking_state =
        matches!(chat_style, ChatStyle::Qwen3 | ChatStyle::Nemotron).then_some(default_thinking);
    if let Some(enable_thinking) = thinking_state.as_mut() {
        // Read controls before adding declarations or wrapping names/results.
        // Tool data may contain these strings as part of URLs or other text.
        for message in &mut source {
            if matches!(message.role.as_str(), "system" | "developer" | "user") {
                if let Some(value) = thinking_control_value(chat_style, &message.content) {
                    *enable_thinking = value;
                }
                message.content = strip_thinking_controls(&message.content);
            }
        }
    }
    let has_instruction =
        !options.tools.is_empty() || !matches!(options.tool_choice, ChatTemplateToolChoice::None);
    if has_instruction {
        let instruction = canonical_tool_instruction(options, format)?;
        source = inject_tool_instruction(&source, &instruction);
    }

    let source = order_tool_results(source);
    let messages = source
        .into_iter()
        .map(|message| {
            if matches!(message.role.as_str(), "tool" | "function") {
                // GLM has a native observation turn that belongs to tool
                // results. Preserve it so the model does not interpret a
                // result as a fresh user request. Other existing formatters
                // understand user turns more consistently, so retain the
                // portable fallback there.
                let role = if matches!(chat_style, ChatStyle::Glm5) {
                    "tool"
                } else {
                    "user"
                };
                return Ok((
                    role.to_owned(),
                    format!(
                        "{open}{}{close}",
                        if format == ToolFormat::Json {
                            tool_response_json(&message)?
                        } else {
                            message.content.clone()
                        },
                        open = if format == ToolFormat::DeepSeek {
                            "<tool_result>"
                        } else {
                            "<tool_response>"
                        },
                        close = if format == ToolFormat::DeepSeek {
                            "</tool_result>"
                        } else {
                            "</tool_response>"
                        },
                    ),
                    true,
                ));
            }

            let mut content = message.content;
            if let Some(reasoning) = message.reasoning_content {
                content = if chat_style == ChatStyle::Gemma4 {
                    format!("<|channel>thought\n{reasoning}<channel|>{content}")
                } else {
                    format!("<think>\n{reasoning}\n</think>\n\n{content}")
                };
            }
            if let Some(name) = message.name.as_deref() {
                content = format!(
                    "<named_message>{}</named_message>",
                    escape_tool_envelope_json(serde_json::to_string(&serde_json::json!({
                        "name": name,
                        "content": content,
                    }))?)
                );
            }
            if message.role == "assistant" && !message.tool_calls.is_empty() {
                if format == ToolFormat::DeepSeek {
                    content.push_str("\n<｜DSML｜tool_calls>");
                }
                for call in &message.tool_calls {
                    content.push('\n');
                    content.push_str(&native_tool_call(call, format)?);
                }
                if format == ToolFormat::DeepSeek {
                    content.push_str("\n</｜DSML｜tool_calls>");
                }
            }
            let role = if message.role == "developer" {
                "system".to_owned()
            } else {
                message.role
            };
            Ok((role, content, false))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut grouped: Vec<(String, String)> = Vec::new();
    let mut previous_tool = false;
    for (role, content, tool) in messages {
        if tool
            && previous_tool
            && format != ToolFormat::Json
            && grouped.last().is_some_and(|message| message.0 == role)
        {
            grouped.last_mut().unwrap().1.push_str(&content);
        } else {
            grouped.push((role, content));
        }
        previous_tool = tool;
    }
    Ok((grouped, thinking_state))
}

impl GgufTokenizer {
    /// SentencePiece BPE encoding (score-based merges).
    ///
    /// Algorithm:
    /// 1. Replace spaces with ▁ (U+2581), prepend ▁
    /// 2. Split into individual characters (as strings)
    /// 3. Repeatedly merge the pair with the highest score
    fn encode_sentencepiece(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        // Prepend space marker and replace spaces
        let processed = format!("▁{}", text.replace(' ', "▁"));
        let pieces = merge_adjacent_pieces(
            processed.chars().map(|c| c.to_string()).collect(),
            |left, right| {
                let mut merged = String::with_capacity(left.len() + right.len());
                merged.push_str(left);
                merged.push_str(right);
                self.vocab_to_id.get(&merged).and_then(|&id| {
                    let score = self.scores.get(id as usize).copied().unwrap_or(0.0);
                    sentencepiece_score_priority(score)
                })
            },
        );

        // Convert pieces to token IDs, falling back to byte tokens for unknowns
        let mut ids = Vec::new();
        for piece in &pieces {
            self.append_piece_ids(piece, &mut ids)?;
        }
        Ok(ids)
    }

    /// GPT-2 BPE encoding (merge-rank-based).
    ///
    /// Algorithm:
    /// 1. Pre-tokenize by splitting on whitespace boundaries
    /// 2. For each word, split into characters
    /// 3. Repeatedly merge the pair with the lowest rank
    fn encode_gpt2(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_ids = Vec::new();
        let nfc_text;
        let text = if self.minimax_pretokenizer && !text.is_ascii() {
            nfc_text = text.nfc().collect::<String>();
            nfc_text.as_str()
        } else {
            text
        };
        let normalized;
        let text = if let Some(marker) = self.space_marker {
            normalized = text.replace(' ', &marker.to_string());
            normalized.as_str()
        } else {
            text
        };

        // Pre-tokenize: split into words, keeping leading spaces with the following word
        let words = if self.joyai_pretokenizer {
            deepseek41::pretokenize(text)?
        } else if self.deepseek_llm_pretokenizer {
            pretokenize_deepseek_llm(text)
        } else if self.space_marker.is_some() {
            vec![text.to_string()]
        } else if self.minimax_pretokenizer || self.chat_style == ChatStyle::MuseSpark {
            pretokenize_minimax(text)
        } else if self.glm_pretokenizer {
            pretokenize_glm(text)
        } else if let Some(preset) = self.qwen_pretokenizer {
            pretokenize_qwen(text, preset)
        } else {
            self.pretokenize(text)
        };

        for word in &words {
            let encoded = if self.space_marker.is_some() {
                word.clone()
            } else {
                self.gpt2_byte_encode(word)
            };
            if self.ignore_merges
                && let Some(&id) = self.vocab_to_id.get(&encoded)
            {
                all_ids.push(id);
                continue;
            }
            let pieces = merge_adjacent_pieces(
                encoded.chars().map(|c| c.to_string()).collect(),
                |left, right| {
                    self.merge_ranks
                        .get(&(left.to_owned(), right.to_owned()))
                        .copied()
                        // The previous `rank < u32::MAX` comparison ignored
                        // MAX, so retain that edge-case behavior.
                        .filter(|&rank| rank < u32::MAX)
                        .map(|rank| u32::MAX - rank)
                },
            );

            // Convert pieces to token IDs
            for piece in &pieces {
                self.append_piece_ids(piece, &mut all_ids)?;
            }
        }

        Ok(all_ids)
    }

    /// Gemma 4 BPE encoding.
    ///
    /// llama.cpp treats Gemma4 as SPM-style BPE: normalize spaces to ▁, split
    /// only around newline runs, and run merge-rank BPE over raw UTF-8 chars.
    fn encode_gemma4(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let normalized = text.replace(' ', "▁");
        let mut all_ids = Vec::new();
        let mut parts = Vec::new();
        let mut current = String::new();
        let mut current_is_newline: Option<bool> = None;

        for ch in normalized.chars() {
            let is_newline = ch == '\n';
            if current_is_newline == Some(is_newline) || current_is_newline.is_none() {
                current.push(ch);
                current_is_newline = Some(is_newline);
            } else {
                parts.push(std::mem::take(&mut current));
                current.push(ch);
                current_is_newline = Some(is_newline);
            }
        }
        if !current.is_empty() {
            parts.push(current);
        }

        for part in parts {
            if let Some(&id) = self.vocab_to_id.get(&part) {
                all_ids.push(id);
                continue;
            }

            let pieces = merge_adjacent_pieces(
                part.chars().map(|c| c.to_string()).collect(),
                |left, right| {
                    self.merge_ranks
                        .get(&(left.to_owned(), right.to_owned()))
                        .copied()
                        .filter(|&rank| rank < u32::MAX)
                        .map(|rank| u32::MAX - rank)
                },
            );

            for piece in pieces {
                self.append_piece_ids(&piece, &mut all_ids)?;
            }
        }

        Ok(all_ids)
    }

    fn append_piece_ids(&self, piece: &str, ids: &mut Vec<u32>) -> Result<()> {
        if let Some(&id) = self.vocab_to_id.get(piece) {
            ids.push(id);
            return Ok(());
        }
        for byte in piece.bytes() {
            if let Some(id) = self.byte_tokens.get(&byte) {
                ids.push(*id);
            } else if let Some(unk) = self.vocab_to_id.get("<unk>") {
                // Make unknown input explicit rather than silently dropping a
                // byte and changing the prompt without an observable token.
                ids.push(*unk);
            } else {
                anyhow::bail!("tokenizer cannot represent byte 0x{byte:02x} without data loss");
            }
        }
        Ok(())
    }

    fn gpt2_byte_encode(&self, text: &str) -> String {
        let mut out = String::new();
        for &byte in text.as_bytes() {
            if let Some(&ch) = self.gpt2_byte_to_unicode.get(&byte) {
                out.push(ch);
            } else {
                out.push(byte as char);
            }
        }
        out
    }

    /// Llama/GPT-style pre-tokenization for GPT-2 BPE.
    /// This mirrors the common tiktoken split points closely enough for GGUF
    /// prompt parity: whitespace attaches to following words/punctuation,
    /// newline runs are separate, and long digit strings split in groups of 3.
    fn pretokenize(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut parts = Vec::new();
        let mut i = 0;

        while i < chars.len() {
            let start = i;

            if chars[i] == '\'' {
                let rest: String = chars[i..chars.len().min(i + 4)].iter().collect();
                let lower = rest.to_ascii_lowercase();
                let mut matched_suffix = false;
                for suffix in ["'re", "'ve", "'ll", "'s", "'t", "'m", "'d"] {
                    if lower.starts_with(suffix) {
                        i += suffix.chars().count();
                        parts.push(chars[start..i].iter().collect());
                        matched_suffix = true;
                        break;
                    }
                }
                if matched_suffix {
                    continue;
                }
            }

            if is_newline(chars[i]) || starts_space_then_newline(&chars, i) {
                while i < chars.len() && is_space_not_newline(chars[i]) {
                    i += 1;
                }
                while i < chars.len() && is_newline(chars[i]) {
                    i += 1;
                }
                parts.push(chars[start..i].iter().collect());
                continue;
            }

            let mut token_start = i;
            if chars[i] == ' ' && i + 1 < chars.len() && !chars[i + 1].is_whitespace() {
                i += 1;
                token_start = i;
            }

            if i + 1 < chars.len()
                && !chars[i].is_whitespace()
                && !chars[i].is_alphabetic()
                && !chars[i].is_numeric()
                && !is_newline(chars[i])
                && chars[i + 1].is_alphabetic()
            {
                i += 2;
                while i < chars.len() && chars[i].is_alphabetic() {
                    i += 1;
                }
                parts.push(chars[start..i].iter().collect());
                continue;
            }

            if i < chars.len() && chars[i].is_alphabetic() {
                i += 1;
                while i < chars.len() && chars[i].is_alphabetic() {
                    i += 1;
                }
                parts.push(chars[start..i].iter().collect());
                continue;
            }

            if i < chars.len() && chars[i].is_numeric() {
                let mut n = 0;
                while i < chars.len() && chars[i].is_numeric() && n < 3 {
                    i += 1;
                    n += 1;
                }
                parts.push(chars[start..i].iter().collect());
                continue;
            }

            if i < chars.len() && !chars[i].is_whitespace() {
                i += 1;
                while i < chars.len()
                    && !chars[i].is_whitespace()
                    && !chars[i].is_alphabetic()
                    && !chars[i].is_numeric()
                {
                    i += 1;
                }
                while i < chars.len() && is_newline(chars[i]) {
                    i += 1;
                }
                parts.push(chars[start..i].iter().collect());
                continue;
            }

            if token_start != start {
                parts.push(chars[start..token_start].iter().collect());
                continue;
            }

            while i < chars.len() && chars[i].is_whitespace() && !is_newline(chars[i]) {
                i += 1;
            }
            parts.push(chars[start..i].iter().collect());
        }

        parts
    }

    /// Decode a single token ID to a string fragment.
    pub fn decode_token(&self, token_id: u32) -> String {
        String::from_utf8_lossy(&self.decode_token_bytes(token_id)).into_owned()
    }

    pub(crate) fn decode_token_bytes(&self, token_id: u32) -> Vec<u8> {
        let id = token_id as usize;
        if id >= self.vocab.len() {
            return Vec::new();
        }

        let token_str = &self.vocab[id];

        // Check for byte tokens: <0xHH>
        if token_str.starts_with("<0x")
            && token_str.ends_with('>')
            && token_str.len() == 6
            && let Ok(byte_val) = u8::from_str_radix(&token_str[3..5], 16)
        {
            return vec![byte_val];
        }

        match self.model {
            TokenizerModel::SentencePiece => {
                // ▁ → space
                token_str.replace('▁', " ").into_bytes()
            }
            TokenizerModel::Gpt2 => {
                let token_text;
                let token_str = if let Some(marker) = self.space_marker {
                    token_text = token_str.replace(marker, " ");
                    token_text.as_str()
                } else {
                    token_str.as_str()
                };
                // Reverse GPT-2 unicode→byte mapping
                let mut bytes = Vec::new();
                for ch in token_str.chars() {
                    if let Some(&byte) = self.gpt2_unicode_to_byte.get(&ch) {
                        bytes.push(byte);
                    } else {
                        let mut buf = [0; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
                bytes
            }
            TokenizerModel::Gemma4 => token_str.replace('▁', " ").into_bytes(),
        }
    }

    /// Decode a sequence of token IDs to a string.
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in token_ids {
            bytes.extend(self.decode_token_bytes(id));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Decode one token for a streamed response, retaining incomplete UTF-8 bytes.
    pub fn decode_stream_token(&self, token_id: u32, pending: &mut Vec<u8>) -> String {
        pending.extend(self.decode_token_bytes(token_id));
        drain_utf8(pending, false)
    }

    /// Flush bytes left at the end of a streamed response.
    pub fn finish_decode_stream(&self, pending: &mut Vec<u8>) -> String {
        drain_utf8(pending, true)
    }

    /// Check if a token ID is a stop token.
    pub fn is_stop_token(&self, token_id: u32) -> bool {
        self.stop_ids.contains(&token_id)
    }

    /// Get all stop token IDs.
    pub fn stop_token_ids(&self) -> &[u32] {
        &self.stop_ids
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

fn drain_utf8(pending: &mut Vec<u8>, flush: bool) -> String {
    let mut output = String::new();
    while !pending.is_empty() {
        match std::str::from_utf8(pending) {
            Ok(text) => {
                output.push_str(text);
                pending.clear();
            }
            Err(error) => {
                let valid = error.valid_up_to();
                if valid > 0 {
                    output.push_str(std::str::from_utf8(&pending[..valid]).unwrap());
                    pending.drain(..valid);
                    continue;
                }
                if let Some(invalid_len) = error.error_len() {
                    output.push('\u{fffd}');
                    pending.drain(..invalid_len);
                } else if flush {
                    output.push_str(&String::from_utf8_lossy(pending));
                    pending.clear();
                } else {
                    break;
                }
            }
        }
    }
    output
}

fn metadata_token_id(
    metadata: &HashMap<String, gguf_file::Value>,
    key: &str,
    vocab_size: usize,
) -> Result<Option<u32>> {
    let Some(value) = metadata.get(key) else {
        return Ok(None);
    };
    let id = match value {
        gguf_file::Value::U32(n) => *n,
        gguf_file::Value::I32(n) => {
            u32::try_from(*n).with_context(|| format!("{key} exceeds U32"))?
        }
        gguf_file::Value::U64(n) => {
            u32::try_from(*n).with_context(|| format!("{key} exceeds U32"))?
        }
        gguf_file::Value::I64(n) => {
            u32::try_from(*n).with_context(|| format!("{key} exceeds U32"))?
        }
        _ => anyhow::bail!("{key} must be an integer"),
    };
    anyhow::ensure!(
        (id as usize) < vocab_size,
        "{key} is outside the vocabulary"
    );
    Ok(Some(id))
}

fn gguf_string_value(value: &gguf_file::Value) -> candle_core::Result<String> {
    match value {
        gguf_file::Value::String(s) => Ok(s.clone()),
        v => candle_core::bail!("not a string {v:?}"),
    }
}

fn looks_like_special_token(token: &str) -> bool {
    matches!(
        token,
        "[INST]" | "[/INST]" | "<<SYS>>" | "<</SYS>>" | "<s>" | "</s>" | "<bos>" | "<eos>"
    ) || (token.starts_with('<')
        && token.ends_with('>')
        && (token.contains('|') || token.contains('｜')))
}

fn parse_merge_pair(merge: &str, vocab_to_id: &HashMap<String, u32>) -> Option<(String, String)> {
    for (idx, ch) in merge.char_indices() {
        if ch != ' ' || idx == 0 {
            continue;
        }
        let right_start = idx + ch.len_utf8();
        if right_start >= merge.len() {
            continue;
        }
        let left = &merge[..idx];
        let right = &merge[right_start..];
        if vocab_to_id.contains_key(left) && vocab_to_id.contains_key(right) {
            return Some((left.to_string(), right.to_string()));
        }
    }
    None
}

/// MiniMax-M2's o200k-style split from its tokenizer.json. Keep this separate
/// from the legacy splitter so other model families retain their token IDs.
fn pretokenize_minimax(text: &str) -> Vec<String> {
    static PATTERN: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(concat!(
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|(?P<space>\s+)"
        )).expect("valid MiniMax pretokenizer regex")
    });
    pretokenize_regex(text, &PATTERN)
}

/// Split from zai-org/GLM-5.3-Flash tokenizer.json. Its number branch has no
/// optional space; allowing one admits merged vocabulary entries that the
/// source tokenizer never emits (for example `Ġ2` instead of `Ġ`, `2`).
fn pretokenize_glm(text: &str) -> Vec<String> {
    static PATTERN: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(concat!(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}",
            r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|(?P<space>\s+)"
        ))
        .expect("valid GLM pretokenizer regex")
    });
    pretokenize_regex(text, &PATTERN)
}

/// Qwen's tokenizer.json split, also recorded in llama.cpp 15fa3c49's
/// QWEN2/QWEN35 presets. Digits never take a leading space; indentation leaves
/// its final whitespace character with the next word. Qwen3.5 includes Unicode
/// combining marks in word runs, while Qwen2 does not.
fn pretokenize_qwen(text: &str, preset: QwenPretokenizer) -> Vec<String> {
    static V2: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(concat!(
            r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
            r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}",
            r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|(?P<space>\s+)"
        ))
        .expect("valid Qwen2 pretokenizer regex")
    });
    static V35: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(concat!(
            r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
            r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}",
            r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|(?P<space>\s+)"
        ))
        .expect("valid Qwen3.5 pretokenizer regex")
    });
    pretokenize_regex(
        text,
        match preset {
            QwenPretokenizer::V2 => &V2,
            QwenPretokenizer::V35 => &V35,
        },
    )
}

/// DeepSeek-V2's `deepseek-llm` preset applies these six isolated splits in
/// order (llama.cpp 15fa3c49). Keep unmatched text between matches; joining it
/// to a neighboring split would admit BPE merges that the source forbids.
fn pretokenize_deepseek_llm(text: &str) -> Vec<String> {
    static PATTERNS: LazyLock<Vec<regex::Regex>> = LazyLock::new(|| {
        [
            r"[\r\n]",
            r"\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
            r"\s?[!-/:-~！-／：-～‘-‟　-。]+",
            r"\s+$",
            r"[一-龥ࠀ-一가-퟿]+",
            r"\p{N}+",
        ].into_iter().map(|pattern| regex::Regex::new(pattern).expect("valid DeepSeek LLM split")).collect()
    });
    let mut parts = vec![text.to_owned()];
    for pattern in PATTERNS.iter() {
        let mut next = Vec::new();
        for part in &parts {
            let mut start = 0;
            for matched in pattern.find_iter(part) {
                if start < matched.start() {
                    next.push(part[start..matched.start()].to_owned());
                }
                next.push(matched.as_str().to_owned());
                start = matched.end();
            }
            if start < part.len() {
                next.push(part[start..].to_owned());
            }
        }
        parts = next;
    }
    parts
}

fn pretokenize_regex(text: &str, pattern: &regex::Regex) -> Vec<String> {
    let mut parts = Vec::new();
    let mut offset = 0;
    while offset < text.len() {
        let captures = pattern
            .captures_at(text, offset)
            .expect("pretokenizer pattern covers all Unicode characters");
        let matched = captures.get(0).unwrap();
        debug_assert_eq!(matched.start(), offset);
        let mut end = matched.end();
        // The source's final alternatives are `\s+(?!\S)|\s+`. Rust's regex
        // engine has no lookahead. Before non-whitespace, the first branch
        // leaves one whitespace character for the next word's optional prefix.
        if let Some(space) = captures.name("space")
            && end < text.len()
        {
            let last = space.as_str().chars().next_back().unwrap();
            if space.len() > last.len_utf8() {
                end -= last.len_utf8();
            }
        }
        parts.push(text[offset..end].to_owned());
        offset = end;
    }
    parts
}

fn is_newline(ch: char) -> bool {
    ch == '\n' || ch == '\r'
}

fn is_space_not_newline(ch: char) -> bool {
    ch.is_whitespace() && !is_newline(ch)
}

fn starts_space_then_newline(chars: &[char], pos: usize) -> bool {
    if pos >= chars.len() || !is_space_not_newline(chars[pos]) {
        return false;
    }
    let mut i = pos;
    while i < chars.len() && is_space_not_newline(chars[i]) {
        i += 1;
    }
    i < chars.len() && is_newline(chars[i])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct TestRng(u64);

    impl TestRng {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (self.0 >> 32) as u32
        }

        fn index(&mut self, upper: usize) -> usize {
            self.next_u32() as usize % upper
        }
    }

    fn reference_sentencepiece_merge(
        mut pieces: Vec<String>,
        scores: &HashMap<String, f32>,
    ) -> Vec<String> {
        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_index = usize::MAX;
            for index in 0..pieces.len().saturating_sub(1) {
                let merged = format!("{}{}", pieces[index], pieces[index + 1]);
                if let Some(&score) = scores.get(&merged)
                    && score > best_score
                {
                    best_score = score;
                    best_index = index;
                }
            }
            if best_index == usize::MAX {
                return pieces;
            }
            let right = pieces.remove(best_index + 1);
            pieces[best_index].push_str(&right);
        }
    }

    fn reference_rank_merge(
        mut pieces: Vec<String>,
        ranks: &HashMap<(String, String), u32>,
    ) -> Vec<String> {
        loop {
            let mut best_rank = u32::MAX;
            let mut best_index = usize::MAX;
            for index in 0..pieces.len().saturating_sub(1) {
                if let Some(&rank) = ranks.get(&(pieces[index].clone(), pieces[index + 1].clone()))
                    && rank < best_rank
                {
                    best_rank = rank;
                    best_index = index;
                }
            }
            if best_index == usize::MAX {
                return pieces;
            }
            let right = pieces.remove(best_index + 1);
            pieces[best_index].push_str(&right);
        }
    }

    #[test]
    fn heap_sentencepiece_merges_match_reference_on_random_inputs() {
        let alphabet = ["a", "b", "c", "▁"];
        let score_choices = [
            f32::NEG_INFINITY,
            -10.0,
            -1.0,
            -0.0,
            0.0,
            1.0,
            1.0,
            10.0,
            f32::INFINITY,
            f32::NAN,
        ];
        let mut rng = TestRng(0x5eed_5eed_cafe_f00d);

        for case in 0..300 {
            let len = 1 + rng.index(12);
            let pieces = (0..len)
                .map(|_| alphabet[rng.index(alphabet.len())].to_string())
                .collect::<Vec<_>>();
            let mut scores = HashMap::new();
            for start in 0..len {
                for end in start + 2..=len {
                    if rng.index(3) != 0 {
                        continue;
                    }
                    let merged = pieces[start..end].concat();
                    scores.insert(merged, score_choices[rng.index(score_choices.len())]);
                }
            }

            let expected = reference_sentencepiece_merge(pieces.clone(), &scores);
            let actual = merge_adjacent_pieces(pieces, |left, right| {
                let merged = format!("{left}{right}");
                scores
                    .get(&merged)
                    .copied()
                    .and_then(sentencepiece_score_priority)
            });
            assert_eq!(actual, expected, "random SentencePiece case {case}");
        }
    }

    #[test]
    fn heap_gpt2_and_gemma_rank_merges_match_reference_on_random_inputs() {
        let alphabet = ["a", "b", "c", "▁", "\n"];
        let mut rng = TestRng(0xc001_d00d_1234_5678);

        for case in 0..300 {
            let len = 1 + rng.index(12);
            let pieces = (0..len)
                .map(|_| alphabet[rng.index(alphabet.len())].to_string())
                .collect::<Vec<_>>();
            let mut ranks = HashMap::new();
            for start in 0..len {
                for middle in start + 1..len {
                    for end in middle + 1..=len {
                        if rng.index(5) != 0 {
                            continue;
                        }
                        let rank = if rng.index(20) == 0 {
                            u32::MAX
                        } else {
                            rng.next_u32() % 8
                        };
                        ranks.insert(
                            (pieces[start..middle].concat(), pieces[middle..end].concat()),
                            rank,
                        );
                    }
                }
            }

            let expected = reference_rank_merge(pieces.clone(), &ranks);
            let actual = merge_adjacent_pieces(pieces, |left, right| {
                ranks
                    .get(&(left.to_owned(), right.to_owned()))
                    .copied()
                    .filter(|&rank| rank < u32::MAX)
                    .map(|rank| u32::MAX - rank)
            });
            assert_eq!(actual, expected, "random rank-BPE case {case}");
        }
    }

    #[test]
    fn heap_rank_merges_scale_to_long_inputs() {
        const PIECE_COUNT: usize = 16_384;
        let mut ranks = HashMap::new();
        let mut piece = "a".to_string();
        for rank in 0..PIECE_COUNT.ilog2() {
            ranks.insert((piece.clone(), piece.clone()), rank);
            piece = piece.repeat(2);
        }

        let merged = merge_adjacent_pieces(vec!["a".to_string(); PIECE_COUNT], |left, right| {
            ranks
                .get(&(left.to_owned(), right.to_owned()))
                .copied()
                .map(|rank| u32::MAX - rank)
        });

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].len(), PIECE_COUNT);
    }

    /// Build a minimal test tokenizer with known vocabulary.
    fn test_tokenizer_sp() -> GgufTokenizer {
        // Minimal SentencePiece-style vocab
        let vocab = vec![
            "<unk>".to_string(),  // 0
            "<s>".to_string(),    // 1 (BOS)
            "</s>".to_string(),   // 2 (EOS)
            "▁".to_string(),      // 3
            "▁H".to_string(),     // 4
            "ello".to_string(),   // 5
            "▁world".to_string(), // 6
            "▁he".to_string(),    // 7
            "ll".to_string(),     // 8
            "o".to_string(),      // 9
            "H".to_string(),      // 10
            "e".to_string(),      // 11
            "l".to_string(),      // 12
            "w".to_string(),      // 13
            "r".to_string(),      // 14
            "d".to_string(),      // 15
        ];
        // Higher score = merge first
        let scores = vec![
            0.0,  // <unk>
            0.0,  // <s>
            0.0,  // </s>
            -1.0, // ▁
            -2.0, // ▁H
            -3.0, // ello
            -4.0, // ▁world
            -1.5, // ▁he
            -2.5, // ll
            -0.5, // o
            -0.1, // H
            -0.2, // e
            -0.3, // l
            -0.4, // w
            -0.5, // r
            -0.6, // d
        ];

        let mut vocab_to_id = HashMap::new();
        for (id, tok) in vocab.iter().enumerate() {
            vocab_to_id.insert(tok.clone(), id as u32);
        }

        GgufTokenizer {
            vocab,
            vocab_to_id,
            scores,
            merge_ranks: HashMap::new(),
            ignore_merges: false,
            minimax_pretokenizer: false,
            glm_pretokenizer: false,
            joyai_pretokenizer: false,
            deepseek_llm_pretokenizer: false,
            qwen_pretokenizer: None,
            special_tokens: Vec::new(),
            space_marker: None,
            model: TokenizerModel::SentencePiece,
            chat_style: ChatStyle::Raw,
            qwen_default_think_prefix: false,
            qwen_default_closed_thinking: false,
            qwen_chatml_default_system: false,
            qwen_default_xhigh_reasoning: false,
            qwen_xml_tools: false,
            bos_id: 1,
            eos_id: 2,
            stop_ids: vec![2],
            add_bos: true,
            byte_tokens: HashMap::new(),
            gpt2_unicode_to_byte: HashMap::new(),
            gpt2_byte_to_unicode: HashMap::new(),
            tool_parser_factory: std::sync::OnceLock::new(),
        }
    }

    #[test]
    fn shared_chat_prefix_uses_real_tokens_and_excludes_the_question() {
        for style in [
            ChatStyle::Qwen3,
            ChatStyle::Gemma4,
            ChatStyle::Llama3,
            ChatStyle::Glm5,
        ] {
            let mut tok = test_tokenizer_sp();
            tok.chat_style = style;
            let system = "Hello world ".repeat(64);
            let a = vec![
                ("system".into(), system.clone()),
                ("user".into(), "Hello".into()),
            ];
            let b = vec![("system".into(), system), ("user".into(), "world".into())];
            let ta = tok.encode_chat(&a).unwrap();
            let tb = tok.encode_chat(&b).unwrap();
            let boundary = tok.shared_chat_prefix_len(&a, &ta).unwrap();
            assert_eq!(&ta[..boundary], &tb[..boundary], "{style:?}");
            assert!(boundary >= 64 && boundary < ta.len());
            let question_only = vec![("user".into(), "Hello".into())];
            assert_eq!(tok.shared_chat_prefix_len(&question_only, &ta), None);
        }
    }

    #[test]
    fn shared_tool_prefix_binds_the_rendered_tool_definitions() {
        let mut tok = test_tokenizer_sp();
        tok.chat_style = ChatStyle::Qwen3;
        let options = ChatTemplateOptions::new([
            ChatTemplateTool::new("Hello").with_description("Hello world ".repeat(64))
        ]);
        let a = vec![ChatTemplateMessage::new("user", "Hello")];
        let b = vec![ChatTemplateMessage::new("user", "world")];
        let ta = tok.encode_chat_with_tool_options(&a, &options).unwrap();
        let tb = tok.encode_chat_with_tool_options(&b, &options).unwrap();
        let boundary = tok.shared_tool_prefix_len(&a, &options, &ta).unwrap();
        assert_eq!(&ta[..boundary], &tb[..boundary]);
        let changed = ChatTemplateOptions::new([
            ChatTemplateTool::new("world").with_description("world Hello ".repeat(64))
        ]);
        let tc = tok.encode_chat_with_tool_options(&b, &changed).unwrap();
        assert_ne!(&ta[..boundary.min(tc.len())], &tc[..boundary.min(tc.len())]);
    }

    fn qwen3_tokenizer_from_template(template: &str) -> GgufTokenizer {
        chatml_tokenizer_from_template("qwen35", template)
    }

    fn chatml_tokenizer_from_template(architecture: &str, template: &str) -> GgufTokenizer {
        let vocab = ["<s>", "</s>", "<|im_start|>", "<|im_end|>"];
        let mut metadata = HashMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            gguf_file::Value::Array(
                vocab
                    .iter()
                    .map(|token| gguf_file::Value::String((*token).to_string()))
                    .collect(),
            ),
        );
        metadata.insert(
            "tokenizer.ggml.token_type".to_string(),
            gguf_file::Value::Array(vocab.iter().map(|_| gguf_file::Value::I32(4)).collect()),
        );
        metadata.insert(
            "tokenizer.ggml.model".to_string(),
            gguf_file::Value::String("gpt2".to_string()),
        );
        metadata.insert(
            "general.architecture".to_string(),
            gguf_file::Value::String(architecture.to_string()),
        );
        metadata.insert(
            "tokenizer.chat_template".to_string(),
            gguf_file::Value::String(template.to_string()),
        );
        metadata.insert(
            "tokenizer.ggml.bos_token_id".to_string(),
            gguf_file::Value::U32(0),
        );
        metadata.insert(
            "tokenizer.ggml.eos_token_id".to_string(),
            gguf_file::Value::U32(1),
        );
        GgufTokenizer::from_metadata(&metadata).unwrap()
    }

    #[test]
    fn test_decode_sentencepiece() {
        let tok = test_tokenizer_sp();
        // "▁H" + "ello" should decode to " Hello"
        let text = tok.decode(&[4, 5]);
        assert_eq!(text, " Hello");
    }

    #[test]
    fn test_decode_world() {
        let tok = test_tokenizer_sp();
        let text = tok.decode(&[6]);
        assert_eq!(text, " world");
    }

    #[test]
    fn test_stop_token_detection() {
        let tok = test_tokenizer_sp();
        assert!(tok.is_stop_token(2));
        assert!(!tok.is_stop_token(1));
        assert!(!tok.is_stop_token(42));
    }

    #[test]
    fn test_byte_token_decode() {
        let mut tok = test_tokenizer_sp();
        // Add a byte token
        tok.vocab.push("<0x41>".to_string()); // 'A' = 0x41
        let id = (tok.vocab.len() - 1) as u32;
        tok.vocab_to_id.insert("<0x41>".to_string(), id);
        tok.byte_tokens.insert(0x41, id);

        let text = tok.decode_token(id);
        assert_eq!(text, "A");
    }

    #[test]
    fn test_chat_format_llama3() {
        let mut tok = test_tokenizer_sp();
        tok.chat_style = ChatStyle::Llama3;

        let messages = vec![
            ("system".to_string(), "You are helpful.".to_string()),
            ("user".to_string(), "Hi".to_string()),
        ];
        let formatted = tok.format_chat(&messages);
        assert!(!formatted.starts_with("<|begin_of_text|>"));
        assert!(formatted.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(formatted.contains("Cutting Knowledge Date: December 2023"));
        assert!(formatted.contains("Today Date:"));
        assert!(formatted.contains("You are helpful."));
        assert!(formatted.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_chat_format_llama3_adds_bos_when_requested_by_template() {
        let mut tok = test_tokenizer_sp();
        tok.chat_style = ChatStyle::Llama3;
        tok.add_bos = false;

        let messages = vec![("user".to_string(), "Hi".to_string())];
        let formatted = tok.format_chat(&messages);
        assert!(formatted.starts_with("<|begin_of_text|>"));
        assert!(formatted.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(formatted.contains("<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"));
    }

    #[test]
    fn test_chat_format_llama4() {
        let mut tok = test_tokenizer_sp();
        tok.chat_style = ChatStyle::Llama4;

        let messages = vec![("user".to_string(), "Hi".to_string())];
        let formatted = tok.format_chat(&messages);
        assert_eq!(
            formatted,
            "<|header_start|>user<|header_end|>\n\nHi<|eot|><|header_start|>assistant<|header_end|>\n\n"
        );
    }

    #[test]
    fn test_chat_format_phi3() {
        let mut tok = test_tokenizer_sp();
        tok.chat_style = ChatStyle::Phi3;

        let messages = vec![("user".to_string(), "Hi".to_string())];
        let formatted = tok.format_chat(&messages);
        assert_eq!(formatted, "<|user|>\nHi<|end|>\n<|assistant|>\n");
    }

    #[test]
    fn test_chat_format_mistral_folds_system_into_first_user_turn() {
        let mut tok = test_tokenizer_sp();
        tok.chat_style = ChatStyle::Mistral;

        let messages = vec![
            (
                "system".to_string(),
                "Follow the format exactly.".to_string(),
            ),
            ("user".to_string(), "Return MISTRAL_OK.".to_string()),
        ];
        assert_eq!(
            tok.format_chat(&messages),
            "[INST] Follow the format exactly.\nReturn MISTRAL_OK. [/INST]"
        );
    }

    #[test]
    fn test_phi3_chat_encoding_right_strips_special_token_whitespace() {
        let mut tok = test_tokenizer_sp();
        tok.chat_style = ChatStyle::Phi3;

        let user = tok.vocab.len() as u32;
        tok.vocab.push("<|user|>".to_string());
        tok.vocab_to_id.insert("<|user|>".to_string(), user);
        let end = tok.vocab.len() as u32;
        tok.vocab.push("<|end|>".to_string());
        tok.vocab_to_id.insert("<|end|>".to_string(), end);
        let assistant = tok.vocab.len() as u32;
        tok.vocab.push("<|assistant|>".to_string());
        tok.vocab_to_id
            .insert("<|assistant|>".to_string(), assistant);
        tok.special_tokens = vec![
            ("<|assistant|>".to_string(), assistant),
            ("<|user|>".to_string(), user),
            ("<|end|>".to_string(), end),
        ];

        let tokens = tok
            .encode_chat(&[("user".to_string(), "Hello".to_string())])
            .unwrap();

        assert_eq!(tokens, vec![tok.bos_id, user, 4, 11, 8, 9, end, assistant]);
    }

    #[test]
    fn test_chat_format_chatml() {
        let mut tok = test_tokenizer_sp();
        tok.chat_style = ChatStyle::ChatML;

        let messages = vec![("user".to_string(), "Hello".to_string())];
        let formatted = tok.format_chat(&messages);
        assert!(formatted.starts_with("<|im_start|>user\nHello<|im_end|>"));
        assert!(formatted.ends_with("<|im_start|>assistant\n"));
    }

    fn sample_tool_messages() -> Vec<ChatTemplateMessage> {
        vec![
            ChatTemplateMessage::new("system", "You are a helpful assistant."),
            ChatTemplateMessage::new("user", "What is the weather in Paris?"),
            ChatTemplateMessage::new("assistant", "").with_tool_calls([ChatTemplateToolCall::new(
                "get_weather",
                r#"{"city":"Paris"}"#,
            )
            .with_id("call-1")]),
            ChatTemplateMessage::new("tool", r#"{"temperature":18}"#).with_tool_call_id("call-1"),
            ChatTemplateMessage::new("user", "And tomorrow?"),
        ]
    }

    fn sample_tool_options() -> ChatTemplateOptions {
        ChatTemplateOptions::new([ChatTemplateTool::new("get_weather")
            .with_description("Get the current weather.")
            .with_parameters_json(r#"{"type":"object","properties":{"city":{"type":"string"}},"additionalProperties":false}"#)])
    }

    fn assert_json_envelope(prompt: &str, tag: &str, expected: serde_json::Value) {
        let opening = format!("<{tag}>");
        let closing = format!("</{tag}>");
        assert!(
            prompt.split(&opening).skip(1).any(|suffix| {
                suffix.split_once(&closing).is_some_and(|(body, _)| {
                    serde_json::from_str::<serde_json::Value>(body)
                        .is_ok_and(|actual| actual == expected)
                })
            }),
            "missing {tag} envelope {expected} in {prompt}"
        );
    }

    #[test]
    fn tool_formatter_preserves_plain_prompt_bytes() {
        let mut tok = test_tokenizer_sp();
        tok.chat_style = ChatStyle::ChatML;
        let plain = vec![
            ChatTemplateMessage::new("system", "You are helpful."),
            ChatTemplateMessage::new("user", "Hello"),
        ];
        let tuples = plain
            .iter()
            .map(|message| (message.role.clone(), message.content.clone()))
            .collect::<Vec<_>>();
        let options = ChatTemplateOptions::new([]);
        assert_eq!(
            tok.format_chat_with_tool_options(&plain, &options).unwrap(),
            tok.format_chat(&tuples)
        );
    }

    #[test]
    fn tool_formatter_preserves_names_without_tools() {
        let mut tok = test_tokenizer_sp();
        tok.chat_style = ChatStyle::Qwen3;
        let messages = [ChatTemplateMessage::new("user", "Hello").with_name("alice")];
        let formatted = tok
            .format_chat_with_tool_options(&messages, &ChatTemplateOptions::new([]))
            .unwrap();

        assert_json_envelope(
            &formatted,
            "named_message",
            serde_json::json!({"content":"Hello","name":"alice"}),
        );
    }

    #[test]
    fn tool_formatter_preserves_named_user_thinking_controls() {
        for style in [ChatStyle::Qwen3, ChatStyle::Nemotron] {
            let mut tok = test_tokenizer_sp();
            tok.chat_style = style;
            let messages =
                [ChatTemplateMessage::new("user", "/no_think\nHello").with_name("alice")];
            let formatted = tok
                .format_chat_with_tool_options(&messages, &ChatTemplateOptions::new([]))
                .unwrap();

            assert_json_envelope(
                &formatted,
                "named_message",
                serde_json::json!({"content":"Hello","name":"alice"}),
            );
            let suffix = if style == ChatStyle::Qwen3 {
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            } else {
                "<|im_start|>assistant\n<think></think>"
            };
            assert!(formatted.ends_with(suffix));
        }
    }

    #[test]
    fn tool_results_preserve_thinking_control_text() {
        for style in [ChatStyle::Qwen3, ChatStyle::Nemotron] {
            let mut tok = test_tokenizer_sp();
            tok.chat_style = style;
            tok.qwen_default_think_prefix = true;
            for (user, result, thinking) in [
                (
                    "Find the guide.",
                    "https://example.com/no_think/guide",
                    true,
                ),
                (
                    "/no_think\nFind the guide.",
                    "https://example.com/think/guide",
                    false,
                ),
            ] {
                let messages = [
                    ChatTemplateMessage::new("user", user),
                    ChatTemplateMessage::new("assistant", "").with_tool_calls([
                        ChatTemplateToolCall::new("lookup", "{}").with_id("call-1"),
                    ]),
                    ChatTemplateMessage::new("tool", result).with_tool_call_id("call-1"),
                ];
                let formatted = tok
                    .format_chat_with_tool_options(&messages, &ChatTemplateOptions::new([]))
                    .unwrap();

                assert_json_envelope(
                    &formatted,
                    "tool_response",
                    serde_json::json!({"content":result,"tool_call_id":"call_0"}),
                );
                let suffix = match (style, thinking) {
                    (_, true) => "<|im_start|>assistant\n<think>\n",
                    (ChatStyle::Qwen3, false) => "<|im_start|>assistant\n<think>\n\n</think>\n\n",
                    (_, false) => "<|im_start|>assistant\n<think></think>",
                };
                assert!(formatted.ends_with(suffix));
            }
        }
    }

    #[test]
    fn tool_definitions_preserve_thinking_control_text() {
        for style in [ChatStyle::Qwen3, ChatStyle::Nemotron] {
            let mut tok = test_tokenizer_sp();
            tok.chat_style = style;
            tok.qwen_default_think_prefix = true;
            let options = ChatTemplateOptions::new([
                ChatTemplateTool::new("lookup")
                    .with_description("https://example.com/no_think/guide")
                    .with_parameters_json(
                        r#"{"type":"object","properties":{"path":{"type":"string","default":"/no_think/guide"}}}"#,
                    ),
            ]);
            let messages = [ChatTemplateMessage::new("user", "Find the guide.")];
            let formatted = tok
                .format_chat_with_tool_options(&messages, &options)
                .unwrap();

            assert!(formatted.contains(&tool_definition_json(&options.tools[0]).unwrap()));
            assert!(formatted.contains("<|im_start|>user\nFind the guide.<|im_end|>"));
            assert!(formatted.ends_with("<|im_start|>assistant\n<think>\n"));
        }
    }

    #[test]
    fn deepseek_r1_preserves_system_developer_and_tool_instructions() {
        let mut tok = test_tokenizer_sp();
        tok.chat_style = ChatStyle::DeepSeekR1;
        let messages = [
            ChatTemplateMessage::new("system", "Follow the system policy."),
            ChatTemplateMessage::new("developer", "Keep the answer concise."),
            ChatTemplateMessage::new("user", "Find the guide."),
        ];
        let options = sample_tool_options()
            .with_tool_choice(ChatTemplateToolChoice::Required)
            .with_parallel_tool_calls(false);
        let formatted = tok
            .format_chat_with_tool_options(&messages, &options)
            .unwrap();

        assert!(formatted.starts_with("Follow the system policy.\n\n# Tools"));
        assert!(formatted.contains(&tool_definition_json(&options.tools[0]).unwrap()));
        assert!(formatted.contains("You must call at least one tool"));
        assert!(formatted.contains("Emit at most one tool call"));
        assert!(formatted.contains(
            "\n\nKeep the answer concise.<｜User｜>Find the guide.<｜Assistant｜><think>\n"
        ));
    }

    #[test]
    fn tool_formatter_uses_native_or_json_envelopes() {
        for style in [
            ChatStyle::Qwen3,
            ChatStyle::ChatML,
            ChatStyle::Nemotron,
            ChatStyle::MiniMax,
            ChatStyle::Glm5,
            ChatStyle::Gemma4,
            ChatStyle::DeepSeekV4,
            ChatStyle::DeepSeekR1,
        ] {
            let mut tok = test_tokenizer_sp();
            tok.chat_style = style;
            let formatted = tok
                .format_chat_with_tool_options(
                    &sample_tool_messages(),
                    &sample_tool_options()
                        .with_tool_choice(ChatTemplateToolChoice::Required)
                        .with_parallel_tool_calls(false),
                )
                .unwrap();
            if style == ChatStyle::DeepSeekV4 {
                assert!(formatted.contains("<｜DSML｜invoke name=\"get_weather\">"));
                assert!(formatted.contains(
                    "<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>"
                ));
                assert!(formatted.contains(r#"<tool_result>{"temperature":18}</tool_result>"#));
            } else if style == ChatStyle::Glm5 {
                assert!(formatted.contains("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"));
                assert!(formatted.contains(r#"<tool_response>{"temperature":18}</tool_response>"#));
            } else {
                assert_json_envelope(
                    &formatted,
                    "tool_call",
                    serde_json::json!({"arguments":{"city":"Paris"},"id":"call_0","name":"get_weather"}),
                );
                assert_json_envelope(
                    &formatted,
                    "tool_response",
                    serde_json::json!({"content":"{\"temperature\":18}","tool_call_id":"call_0"}),
                );
            }
            assert!(!formatted.contains("<minimax:tool_call>"));
            assert!(!formatted.contains("<|tool_call>call:"));
            assert!(!formatted.contains("<TOOLCALL>"));
        }
    }

    #[test]
    fn tool_choice_and_parallel_semantics_are_injected_and_validated() {
        let tok = test_tokenizer_sp();
        let messages = [ChatTemplateMessage::new("user", "Use the tool.")];
        let none = sample_tool_options().with_tool_choice(ChatTemplateToolChoice::None);
        let none_formatted = tok.format_chat_with_tool_options(&messages, &none).unwrap();
        assert!(none_formatted.contains("Do not call tools"));

        let auto_formatted = tok
            .format_chat_with_tool_options(&messages, &sample_tool_options())
            .unwrap();
        assert!(auto_formatted.contains("zero or more tools"));

        let required_formatted = tok
            .format_chat_with_tool_options(
                &messages,
                &sample_tool_options().with_tool_choice(ChatTemplateToolChoice::Required),
            )
            .unwrap();
        assert!(required_formatted.contains("must call at least one tool"));

        let required = sample_tool_options()
            .with_tool_choice(ChatTemplateToolChoice::Named("get_weather".into()))
            .with_parallel_tool_calls(false);
        let formatted = tok
            .format_chat_with_tool_options(&messages, &required)
            .unwrap();
        assert!(formatted.contains("must call the named tool `get_weather`"));
        assert!(formatted.contains("Emit at most one tool call"));

        let missing =
            ChatTemplateOptions::new([]).with_tool_choice(ChatTemplateToolChoice::Required);
        let error = tok
            .format_chat_with_tool_options(&messages, &missing)
            .unwrap_err()
            .to_string();
        assert!(error.contains("at least one declared tool"));
    }

    #[test]
    fn developer_and_function_messages_use_openai_roles() {
        let mut tok = test_tokenizer_sp();
        tok.chat_style = ChatStyle::ChatML;
        let messages = [
            ChatTemplateMessage::new("developer", "Keep the answer concise."),
            ChatTemplateMessage::new("function", r#"{"ok":true}"#).with_tool_call_id("call-1"),
        ];
        let formatted = tok
            .format_chat_with_tool_options(&messages, &ChatTemplateOptions::new([]))
            .unwrap();

        assert!(formatted.contains("<|im_start|>system\nKeep the answer concise."));
        assert!(formatted.contains("<|im_start|>user\n<tool_response>"));
        assert_json_envelope(
            &formatted,
            "tool_response",
            serde_json::json!({"content":"{\"ok\":true}","tool_call_id":"call-1"}),
        );
    }

    #[test]
    fn glm_tool_results_use_observation_turns() {
        let mut tok = test_tokenizer_sp();
        tok.chat_style = ChatStyle::Glm5;
        let messages = [
            ChatTemplateMessage::new("user", "Call the weather function."),
            ChatTemplateMessage::new("assistant", "").with_tool_calls([ChatTemplateToolCall::new(
                "get_weather",
                r#"{"city":"Paris"}"#,
            )
            .with_id("call-1")]),
            ChatTemplateMessage::new("tool", "18 C").with_tool_call_id("call-1"),
        ];
        let formatted = tok
            .format_chat_with_tool_options(
                &messages,
                &ChatTemplateOptions::new([ChatTemplateTool::new("get_weather")]),
            )
            .unwrap();

        assert!(formatted.contains("<|observation|><tool_response>"));
        assert!(!formatted.contains("<|user|><tool_response>"));
    }

    #[test]
    fn tool_envelope_json_escapes_markup_delimiters() {
        let mut tok = test_tokenizer_sp();
        tok.chat_style = ChatStyle::ChatML;
        let messages = [
            ChatTemplateMessage::new("user", "Use lookup."),
            ChatTemplateMessage::new("assistant", "").with_tool_calls([ChatTemplateToolCall::new(
                "lookup",
                r#"{"text":"</tool_call>"}"#,
            )]),
            ChatTemplateMessage::new("tool", "</tool_response>"),
        ];
        let options = ChatTemplateOptions::new([
            ChatTemplateTool::new("lookup").with_description("</tools><tool_call>")
        ]);
        let formatted = tok
            .format_chat_with_tool_options(&messages, &options)
            .unwrap();

        assert!(formatted.contains(r#"\u003c/tools\u003e\u003ctool_call\u003e"#));
        assert!(formatted.contains(r#"\u003c/tool_call\u003e"#));
        assert!(formatted.contains(r#"\u003c/tool_response\u003e"#));
    }

    #[test]
    fn test_chat_format_qwen3_no_think() {
        let mut tok = test_tokenizer_sp();
        tok.chat_style = ChatStyle::Qwen3;

        let messages = vec![("user".to_string(), "/no_think\nHello".to_string())];
        let formatted = tok.format_chat(&messages);
        assert_eq!(
            formatted,
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn test_chat_format_qwen3_defaults_to_thinking() {
        let mut tok = test_tokenizer_sp();
        tok.chat_style = ChatStyle::Qwen3;

        let messages = vec![("user".to_string(), "Hello".to_string())];
        let formatted = tok.format_chat(&messages);
        assert_eq!(
            formatted,
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_qwen3_family_template_opens_default_thinking_turn() {
        let template = r#"
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}
"#;
        let tok = qwen3_tokenizer_from_template(template);

        assert_eq!(tok.chat_style, ChatStyle::Qwen3);
        assert_eq!(
            tok.format_chat(&[("user".to_string(), "Hello".to_string())]),
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        assert_eq!(
            tok.format_chat(&[("user".to_string(), "/no_think\nHello".to_string())]),
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn qwen35_published_closed_thinking_default_and_controls() -> Result<()> {
        // The selected GGUF reverses the primary HF template's default.
        let template = r#"
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is true %}
        {{- '<think>\n' }}
    {%- else %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
"#;
        let tokenizer = qwen3_tokenizer_from_template(template);
        assert!(tokenizer.qwen_default_closed_thinking);
        assert!(!tokenizer.qwen_default_think_prefix);
        let messages = [("user".into(), "Hello".into())];
        let prefix = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
        let closed = format!("{prefix}<think>\n\n</think>\n\n");
        let open = format!("{prefix}<think>\n");
        assert_eq!(tokenizer.format_chat(&messages), closed);
        assert_eq!(
            tokenizer.format_chat_with_thinking_state(&messages, Some(false), None),
            closed
        );
        assert_eq!(
            tokenizer.format_chat_with_thinking_state(&messages, Some(true), None),
            open
        );
        assert_eq!(
            tokenizer.format_chat(&[("user".into(), "/think\nHello".into())]),
            open
        );
        assert_eq!(
            tokenizer.format_chat(&[("user".into(), "/no_think\nHello".into())]),
            closed
        );
        let options = ChatTemplateOptions::new([ChatTemplateTool::new("lookup")]);
        let default_tools = tokenizer.format_chat_with_tool_options(
            &[ChatTemplateMessage::new("user", "Hello")],
            &options,
        )?;
        assert!(default_tools.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        let thinking_tools = tokenizer.format_chat_with_tool_options(
            &[ChatTemplateMessage::new("user", "/think\nHello")],
            &options,
        )?;
        assert!(thinking_tools.ends_with("<|im_start|>assistant\n<think>\n"));
        Ok(())
    }

    #[test]
    fn qwen_chatml_uses_embedded_default_system_only_when_missing() -> Result<()> {
        let template = format!(
            r#"
{{%- if messages[0]['role'] == 'system' %}}
    {{{{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}}}
{{%- else %}}
    {{{{- '<|im_start|>system\n{QWEN_DEFAULT_SYSTEM}<|im_end|>\n' }}}}
{{%- endif %}}
"#
        );
        let tokenizer = chatml_tokenizer_from_template("qwen2", &template);
        assert!(tokenizer.qwen_chatml_default_system);
        assert_eq!(tokenizer.chat_style, ChatStyle::ChatML);
        let user = [("user".into(), "Hello".into())];
        let expected = format!(
            "<|im_start|>system\n{QWEN_DEFAULT_SYSTEM}<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(tokenizer.format_chat(&user), expected);
        assert_eq!(
            tokenizer.format_chat(&[
                ("system".into(), "Custom".into()),
                ("user".into(), "Hello".into())
            ]),
            "<|im_start|>system\nCustom<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
        let tools = tokenizer.format_chat_with_tool_options(
            &[ChatTemplateMessage::new("user", "Hello")],
            &ChatTemplateOptions::new([ChatTemplateTool::new("lookup")]),
        )?;
        assert!(tools.starts_with(&format!("<|im_start|>system\n{QWEN_DEFAULT_SYSTEM}")));
        assert_eq!(tools.matches(QWEN_DEFAULT_SYSTEM).count(), 1);
        let ordinary = chatml_tokenizer_from_template("qwen2", "plain ChatML");
        assert!(!ordinary.qwen_chatml_default_system);
        assert_eq!(
            ordinary.format_chat(&user),
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
        Ok(())
    }

    #[test]
    fn qwen_default_system_requires_the_no_system_else_branch() {
        let literal = format!("'<|im_start|>system\\n{QWEN_DEFAULT_SYSTEM}<|im_end|>\\n'");
        for template in [
            // A custom template can use the same text for an explicit system
            // message while leaving the no-system branch empty.
            format!(
                "{{% if messages[0]['role'] == 'system' %}}{{{{ {literal} }}}}{{% else %}}{{{{ '' }}}}{{% endif %}}"
            ),
            format!(
                "{{% if messages[0]['role'] != 'system' %}}{{{{ '' }}}}{{% else %}}{{{{ {literal} }}}}{{% endif %}}"
            ),
            // A literal mentioned outside the branch is not its default.
            format!(
                "{{% if messages[0]['role'] == 'system' %}}{{{{ '' }}}}{{% else %}}{{{{ '' }}}}{{% endif %}}{{{{ {literal} }}}}"
            ),
        ] {
            let tokenizer = chatml_tokenizer_from_template("qwen2", &template);
            assert!(!tokenizer.qwen_chatml_default_system);
            assert_eq!(
                tokenizer.format_chat(&[("user".into(), "Hello".into())]),
                "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
            );
        }
    }

    #[test]
    fn test_qwen38_template_adds_default_xhigh_reasoning_instruction() {
        let template = format!(
            r#"
{{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}}
{{%- set reasoning_instructions = '{QWEN_XHIGH_REASONING_INSTRUCTION}' %}}
{{%- if add_generation_prompt %}}
    {{{{- '<|im_start|>assistant\n' }}}}
    {{%- if enable_thinking is defined and enable_thinking is false %}}
        {{{{- '<think>\n\n</think>\n\n' }}}}
    {{%- else %}}
        {{{{- '<think>\n' }}}}
    {{%- endif %}}
{{%- endif %}}
"#
        );
        let tok = qwen3_tokenizer_from_template(&template);

        assert!(tok.qwen_default_xhigh_reasoning);
        assert_eq!(
            tok.format_chat(&[("user".to_string(), "Hello".to_string())]),
            format!(
                "<|im_start|>system\n{QWEN_XHIGH_REASONING_INSTRUCTION}<|im_end|>\n\
<|im_start|>user\nHello<|im_end|>\n\
<|im_start|>assistant\n<think>\n"
            )
        );
        assert_eq!(
            tok.format_chat(&[
                ("system".to_string(), "Be concise.".to_string()),
                ("user".to_string(), "Hello".to_string()),
            ]),
            format!(
                "<|im_start|>system\n{QWEN_XHIGH_REASONING_INSTRUCTION}\n\nBe concise.<|im_end|>\n\
<|im_start|>user\nHello<|im_end|>\n\
<|im_start|>assistant\n<think>\n"
            )
        );
        assert_eq!(
            tok.format_chat(&[("user".to_string(), "/no_think\nHello".to_string())]),
            "<|im_start|>user\nHello<|im_end|>\n\
<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn test_qwen3_family_template_strips_completed_reasoning_from_history() {
        let template = r#"
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}
"#;
        let tok = qwen3_tokenizer_from_template(template);
        let messages = vec![
            ("user".to_string(), "First question".to_string()),
            (
                "assistant".to_string(),
                "<think>\nprivate reasoning\n</think>\n\nFirst answer".to_string(),
            ),
            ("user".to_string(), "Second question".to_string()),
        ];

        assert_eq!(
            tok.format_chat(&messages),
            "<|im_start|>user\nFirst question<|im_end|>\n\
<|im_start|>assistant\nFirst answer<|im_end|>\n\
<|im_start|>user\nSecond question<|im_end|>\n\
<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn qwen3_family_preserves_reasoning_after_latest_user_for_tool_turns() {
        assert_eq!(
            qwen3_assistant_history(
                "<think>\ncurrent reasoning\n</think>\n\n<tool_call>call</tool_call>",
                true,
            ),
            "<think>\ncurrent reasoning\n</think>\n\n<tool_call>call</tool_call>"
        );
        assert_eq!(
            qwen3_assistant_history("<tool_call>call</tool_call>", true),
            "<think>\n\n</think>\n\n<tool_call>call</tool_call>"
        );
    }

    #[test]
    fn qwen_template_without_default_think_prefix_is_unchanged() {
        let template = r#"
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"#;
        let tok = qwen3_tokenizer_from_template(template);

        assert!(!tok.qwen_default_think_prefix);
        assert_eq!(
            tok.format_chat(&[("user".to_string(), "Hello".to_string())]),
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_chat_format_gemma4_uses_vocab_bos() {
        let mut tok = test_tokenizer_sp();
        tok.chat_style = ChatStyle::Gemma4;
        tok.add_bos = false;

        let messages = vec![("user".to_string(), "Hello".to_string())];
        let formatted = tok.format_chat(&messages);
        assert!(formatted.starts_with("<s><|turn>user\n"));
        assert!(formatted.contains("<|turn>user\nHello<turn|>\n"));
        assert!(formatted.ends_with("<|turn>model\n<|channel>thought\n<channel|>"));
    }

    #[test]
    fn test_chat_format_nemotron_no_think() {
        let mut tok = test_tokenizer_sp();
        tok.chat_style = ChatStyle::Nemotron;

        let messages = vec![("user".to_string(), "/no_think\nHi".to_string())];
        let formatted = tok.format_chat(&messages);
        assert_eq!(
            formatted,
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>"
        );
    }

    #[test]
    fn test_chat_format_nemotron_defaults_to_thinking() {
        let mut tok = test_tokenizer_sp();
        tok.chat_style = ChatStyle::Nemotron;

        let messages = vec![
            ("system".to_string(), "Be concise.".to_string()),
            ("user".to_string(), "Hi".to_string()),
        ];
        let formatted = tok.format_chat(&messages);
        assert_eq!(
            formatted,
            "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn test_chat_format_deepseek2() {
        let mut tok = test_tokenizer_sp();
        tok.chat_style = ChatStyle::DeepSeek;

        let messages = vec![
            ("system".to_string(), "Be concise.".to_string()),
            ("user".to_string(), "Hi".to_string()),
        ];
        let formatted = tok.format_chat(&messages);
        assert_eq!(formatted, "Be concise.\n\nUser: Hi\n\nAssistant:");
    }

    #[test]
    fn test_chat_format_minimax() {
        let mut tok = test_tokenizer_sp();
        tok.chat_style = ChatStyle::MiniMax;

        let messages = vec![
            ("system".to_string(), "Be concise.".to_string()),
            ("user".to_string(), "Hi".to_string()),
        ];
        assert_eq!(
            tok.format_chat(&messages),
            "]~b]system\nBe concise.[e~[\n]~b]user\nHi[e~[\n]~b]ai\n<think>\n"
        );

        let user_only = vec![("user".to_string(), "Hi".to_string())];
        assert!(tok.format_chat(&user_only).starts_with(
            "]~b]system\nYou are a helpful assistant. Your name is MiniMax-M2.5 and is built by MiniMax.[e~[\n"
        ));
    }

    #[test]
    fn test_chat_format_deepseek_r1_uses_reasoning_prefix() {
        let mut tok = test_tokenizer_sp();
        tok.chat_style = ChatStyle::DeepSeekR1;

        let messages = vec![
            ("system".to_string(), "Be concise.".to_string()),
            ("user".to_string(), "What is 2 + 2?".to_string()),
            ("assistant".to_string(), "scratch work</think>4".to_string()),
            ("user".to_string(), "And 3 + 3?".to_string()),
        ];
        let formatted = tok.format_chat(&messages);
        assert_eq!(
            formatted,
            "Be concise.<｜User｜>What is 2 + 2?<｜Assistant｜>4<｜end▁of▁sentence｜><｜User｜>And 3 + 3?<｜Assistant｜><think>\n"
        );
    }

    #[test]
    fn test_chat_format_deepseek_r1_includes_bos_when_tokenizer_does_not() {
        let mut tok = test_tokenizer_sp();
        tok.chat_style = ChatStyle::DeepSeekR1;
        tok.add_bos = false;

        let messages = vec![("user".to_string(), "Hi".to_string())];
        let formatted = tok.format_chat(&messages);
        assert_eq!(formatted, "<s><｜User｜>Hi<｜Assistant｜><think>\n");
    }

    #[test]
    fn test_special_tokens_encode_atomically() {
        let mut tok = test_tokenizer_sp();
        let turn = tok.vocab.len() as u32;
        tok.vocab.push("<|turn>".to_string());
        tok.vocab_to_id.insert("<|turn>".to_string(), turn);
        let end_turn = tok.vocab.len() as u32;
        tok.vocab.push("<turn|>".to_string());
        tok.vocab_to_id.insert("<turn|>".to_string(), end_turn);
        tok.special_tokens = vec![
            ("<|turn>".to_string(), turn),
            ("<turn|>".to_string(), end_turn),
        ];
        tok.add_bos = false;

        let tokens = tok.encode("<|turn>user\nHi<turn|>").unwrap();
        assert_eq!(tokens.first().copied(), Some(turn));
        assert!(tokens.contains(&end_turn));
    }

    #[test]
    fn test_deepseek_r1_markers_are_special_tokens() {
        assert!(looks_like_special_token("<｜User｜>"));
        assert!(looks_like_special_token("<｜end▁of▁sentence｜>"));
    }

    #[test]
    fn test_deepseek_r1_style_detected_from_vocab() {
        let vocab = [
            "<｜begin▁of▁sentence｜>",
            "<｜end▁of▁sentence｜>",
            "<｜User｜>",
            "<｜Assistant｜>",
            "<think>",
            "</think>",
        ];
        let mut metadata = HashMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            gguf_file::Value::Array(
                vocab
                    .iter()
                    .map(|token| gguf_file::Value::String((*token).to_string()))
                    .collect(),
            ),
        );
        metadata.insert(
            "tokenizer.ggml.token_type".to_string(),
            gguf_file::Value::Array(vocab.iter().map(|_| gguf_file::Value::I32(4)).collect()),
        );
        metadata.insert(
            "tokenizer.ggml.model".to_string(),
            gguf_file::Value::String("llama".to_string()),
        );
        metadata.insert(
            "general.architecture".to_string(),
            gguf_file::Value::String("qwen2".to_string()),
        );
        metadata.insert(
            "tokenizer.ggml.bos_token_id".to_string(),
            gguf_file::Value::U32(0),
        );
        metadata.insert(
            "tokenizer.ggml.eos_token_id".to_string(),
            gguf_file::Value::U32(1),
        );

        let tokenizer = GgufTokenizer::from_metadata(&metadata).unwrap();
        assert_eq!(tokenizer.chat_style, ChatStyle::DeepSeekR1);
        assert_eq!(tokenizer.stop_token_ids(), &[1]);
    }

    #[test]
    fn test_deepseek_v4_uses_no_think_generation_prefix() {
        let vocab = [
            "<｜begin▁of▁sentence｜>",
            "<｜end▁of▁sentence｜>",
            "<｜User｜>",
            "<｜Assistant｜>",
            "<think>",
            "</think>",
            "Hi",
        ];
        let mut metadata = HashMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            gguf_file::Value::Array(
                vocab
                    .iter()
                    .map(|token| gguf_file::Value::String((*token).to_string()))
                    .collect(),
            ),
        );
        metadata.insert(
            "tokenizer.ggml.token_type".to_string(),
            gguf_file::Value::Array(vocab.iter().map(|_| gguf_file::Value::I32(4)).collect()),
        );
        metadata.insert(
            "tokenizer.ggml.model".to_string(),
            gguf_file::Value::String("gpt2".to_string()),
        );
        metadata.insert(
            "general.architecture".to_string(),
            gguf_file::Value::String("deepseek4".to_string()),
        );
        metadata.insert(
            "tokenizer.ggml.bos_token_id".to_string(),
            gguf_file::Value::U32(0),
        );
        metadata.insert(
            "tokenizer.ggml.eos_token_id".to_string(),
            gguf_file::Value::U32(1),
        );
        metadata.insert(
            "tokenizer.ggml.add_bos_token".to_string(),
            gguf_file::Value::Bool(false),
        );

        let tokenizer = GgufTokenizer::from_metadata(&metadata).unwrap();
        assert_eq!(tokenizer.chat_style, ChatStyle::DeepSeekV4);
        assert_eq!(
            tokenizer.format_chat(&[("user".to_string(), "Hi".to_string())]),
            "<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜></think>"
        );
    }

    #[test]
    fn test_glm5_uses_mask_start_and_reasoning_prefix() {
        let vocab = [
            "[gMASK]",
            "<sop>",
            "<|endoftext|>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|observation|>",
            "Hi",
        ];
        let mut metadata = HashMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            gguf_file::Value::Array(
                vocab
                    .iter()
                    .map(|token| gguf_file::Value::String((*token).to_string()))
                    .collect(),
            ),
        );
        metadata.insert(
            "tokenizer.ggml.token_type".to_string(),
            gguf_file::Value::Array(vocab.iter().map(|_| gguf_file::Value::I32(4)).collect()),
        );
        metadata.insert(
            "tokenizer.ggml.model".to_string(),
            gguf_file::Value::String("gpt2".to_string()),
        );
        metadata.insert(
            "general.architecture".to_string(),
            gguf_file::Value::String("glm5next".to_string()),
        );
        metadata.insert(
            "tokenizer.ggml.bos_token_id".to_string(),
            gguf_file::Value::U32(0),
        );
        metadata.insert(
            "tokenizer.ggml.eos_token_id".to_string(),
            gguf_file::Value::U32(2),
        );

        let tokenizer = GgufTokenizer::from_metadata(&metadata).unwrap();
        assert_eq!(tokenizer.chat_style, ChatStyle::Glm5);
        assert_eq!(
            tokenizer.format_chat(&[("user".to_string(), "Hi".to_string())]),
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hi<|assistant|><think>"
        );
        assert_eq!(
            tokenizer.format_chat(&[
                ("user".to_string(), "Hi".to_string()),
                ("assistant".to_string(), "Hello".to_string()),
                ("user".to_string(), "Again".to_string()),
            ]),
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hi<|assistant|><think></think>Hello<|user|>Again<|assistant|><think>"
        );
        assert_eq!(tokenizer.stop_token_ids(), &[2, 4, 6]);
    }

    #[test]
    fn glm_missing_bos_and_normal_typed_control_tokens_match_source() {
        let bytes = build_gpt2_byte_to_unicode();
        let mut vocab: Vec<_> = (0..=255u8).map(|b| bytes[&b].to_string()).collect();
        vocab.extend(
            [
                "[gMASK]",
                "<sop>",
                "<think>",
                "</think>",
                "<|user|>",
                "<|assistant|>",
                "<|system|>",
                "<|endoftext|>",
                "<|observation|>",
                "Ġ2",
            ]
            .into_iter()
            .map(str::to_owned),
        );
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                gguf_file::Value::String("glm5next".into()),
            ),
            (
                "tokenizer.ggml.model".into(),
                gguf_file::Value::String("gpt2".into()),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                gguf_file::Value::Array(vocab.into_iter().map(gguf_file::Value::String).collect()),
            ),
            (
                "tokenizer.ggml.merges".into(),
                gguf_file::Value::Array(vec![gguf_file::Value::String("Ġ 2".into())]),
            ),
            (
                "tokenizer.ggml.eos_token_id".into(),
                gguf_file::Value::U32(263),
            ),
        ]);
        let tok = GgufTokenizer::from_metadata(&metadata).unwrap();
        assert!(!tok.add_bos);
        assert_eq!(tok.encode("").unwrap(), Vec::<u32>::new());
        assert_eq!(tok.encode(" 2 + 2").unwrap(), vec![32, 50, 32, 43, 32, 50]);
        assert_eq!(
            tok.encode("[gMASK]<sop><think></think>").unwrap(),
            vec![256, 257, 258, 259]
        );
        let ids = tok.encode_chat(&[("user".into(), "Hi".into())]).unwrap();
        assert_eq!(&ids[..2], &[256, 257]);
        assert_eq!(&ids[ids.len() - 2..], &[261, 258]);
        let history =
            tok.format_chat(&[("assistant".into(), "<think>Reason.</think>Answer".into())]);
        assert!(history.contains("<|assistant|><think>Reason.</think>Answer<|assistant|><think>"));
    }

    #[test]
    fn glm_splits_match_source_without_normalizing_text() {
        let cases: &[(&str, &[&str])] = &[
            (
                "What is 2 + 2?",
                &["What", " is", " ", "2", " +", " ", "2", "?"],
            ),
            (
                "1234567 ١٢٣٤ ½²",
                &["123", "456", "7", " ", "١٢٣", "٤", " ", "½²"],
            ),
            (
                "I'M can't We'd I'Ll",
                &["I", "'M", " can", "'t", " We", "'d", " I", "'Ll"],
            ),
            (
                "cafe\u{301} e\u{301}lan",
                &["cafe", "\u{301}", " e", "\u{301}lan"],
            ),
            (
                "  abc \t\t b\r\n \n x ///\r\ny  \t",
                &[
                    " ", " abc", " \t\t", " b", "\r\n \n", " x", " ///\r\n", "y", "  \t",
                ],
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(pretokenize_glm(input), *expected);
            assert_eq!(pretokenize_glm(input).concat(), *input);
        }
    }

    #[test]
    fn test_mistral_style_detected_from_embedded_template() {
        let vocab = ["<s>", "</s>", "[INST]", "[/INST]"];
        let mut metadata = HashMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            gguf_file::Value::Array(
                vocab
                    .iter()
                    .map(|token| gguf_file::Value::String((*token).to_string()))
                    .collect(),
            ),
        );
        metadata.insert(
            "tokenizer.ggml.model".to_string(),
            gguf_file::Value::String("llama".to_string()),
        );
        metadata.insert(
            "general.architecture".to_string(),
            gguf_file::Value::String("llama".to_string()),
        );
        metadata.insert(
            "tokenizer.chat_template".to_string(),
            gguf_file::Value::String(
                "[INST] Only user and assistant roles are supported! [/INST]".to_string(),
            ),
        );
        metadata.insert(
            "tokenizer.ggml.bos_token_id".to_string(),
            gguf_file::Value::U32(0),
        );
        metadata.insert(
            "tokenizer.ggml.eos_token_id".to_string(),
            gguf_file::Value::U32(1),
        );

        let tokenizer = GgufTokenizer::from_metadata(&metadata).unwrap();
        assert_eq!(tokenizer.chat_style, ChatStyle::Mistral);
        assert_eq!(tokenizer.stop_token_ids(), &[1]);
    }

    #[test]
    fn test_minimax_style_detected_from_role_marker() {
        let vocab = ["]~!b[", "]~b]", "[e~[", "<think>"];
        let mut metadata = HashMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            gguf_file::Value::Array(
                vocab
                    .iter()
                    .map(|token| gguf_file::Value::String((*token).to_string()))
                    .collect(),
            ),
        );
        metadata.insert(
            "tokenizer.ggml.token_type".to_string(),
            gguf_file::Value::Array(vocab.iter().map(|_| gguf_file::Value::I32(4)).collect()),
        );
        metadata.insert(
            "tokenizer.ggml.model".to_string(),
            gguf_file::Value::String("gpt2".to_string()),
        );
        metadata.insert(
            "general.architecture".to_string(),
            gguf_file::Value::String("minimax-m2".to_string()),
        );
        metadata.insert(
            "tokenizer.ggml.bos_token_id".to_string(),
            gguf_file::Value::U32(0),
        );
        metadata.insert(
            "tokenizer.ggml.eos_token_id".to_string(),
            gguf_file::Value::U32(2),
        );

        let tokenizer = GgufTokenizer::from_metadata(&metadata).unwrap();
        assert_eq!(tokenizer.chat_style, ChatStyle::MiniMax);
        assert_eq!(tokenizer.stop_token_ids(), &[2]);
        assert_eq!(tokenizer.encode("]~b]").unwrap(), vec![1]);
    }

    #[test]
    fn minimax_splits_match_reference() {
        // Expected pieces from MiniMaxAI/MiniMax-M2.5 tokenizer.json at
        // f710177d938eff80b684d42c5aa84b382612f21f, HF tokenizers 0.22.2.
        let cases: &[(&str, &[&str])] = &[
            ("", &[]),
            (
                "MiniMax twoSum XMLHttpRequest HTTPServer",
                &[
                    "Mini",
                    "Max",
                    " two",
                    "Sum",
                    " XMLHttp",
                    "Request",
                    " HTTPServer",
                ],
            ),
            ("I'M can't We'd I'Ll", &["I'M", " can't", " We'd", " I'Ll"]),
            ("  abc", &[" ", " abc"]),
            (" \t\u{2003}x", &[" \t", "\u{2003}x"]),
            ("abc \r\n \n  def", &["abc", " \r\n \n", " ", " def"]),
            (
                "a\t\t b\r\n \n x ///\r\ny  \t",
                &["a", "\t\t", " b", "\r\n \n", " x", " ///\r\n", "y", "  \t"],
            ),
            ("abc///\r\n//x", &["abc", "///\r\n//", "x"]),
            (
                "1234567 ١٢٣٤ ½²",
                &["123", "456", "7", " ", "١٢٣", "٤", " ", "½²"],
            ),
            ("𝔘nicode \u{0345}शब्द", &["𝔘nicode", " \u{0345}शब्द"]),
            (
                "cafe\u{301} e\u{301}lan 한글 日本語",
                &["cafe\u{301}", " e\u{301}lan", " 한글", " 日本語"],
            ),
        ];
        for &(text, expected) in cases {
            let actual = pretokenize_minimax(text);
            assert_eq!(actual, expected, "{text:?}");
            assert_eq!(actual.concat(), text, "split lost input");
        }
    }

    #[test]
    fn minimax_metadata_enables_nfc_without_changing_other_tokenizers() {
        // A byte-only vocabulary isolates normalization from BPE merge ranks.
        let bytes = build_gpt2_byte_to_unicode();
        let mut vocab: Vec<_> = (0..=255_u8).map(|b| bytes[&b].to_string()).collect();
        vocab.push("]~b]".into());
        let mut metadata = HashMap::from([
            (
                "tokenizer.ggml.tokens".into(),
                gguf_file::Value::Array(vocab.into_iter().map(gguf_file::Value::String).collect()),
            ),
            (
                "tokenizer.ggml.model".into(),
                gguf_file::Value::String("gpt2".into()),
            ),
            (
                "general.architecture".into(),
                gguf_file::Value::String("minimax-m2".into()),
            ),
            (
                "tokenizer.ggml.add_bos_token".into(),
                gguf_file::Value::Bool(false),
            ),
            (
                "tokenizer.ggml.token_type".into(),
                gguf_file::Value::Array(
                    (0..257)
                        .map(|i| gguf_file::Value::I32(if i == 256 { 4 } else { 1 }))
                        .collect(),
                ),
            ),
        ]);
        let text = "cafe\u{301}]~b]A\u{30a}ngstro\u{308}m";
        let legacy = GgufTokenizer::from_metadata(&metadata).unwrap();
        let original = legacy.encode(text).unwrap();
        assert!(!legacy.minimax_pretokenizer);
        assert_eq!(legacy.decode(&original), text);
        metadata.insert(
            "tokenizer.ggml.pre".into(),
            gguf_file::Value::String("minimax-m2".into()),
        );
        let minimax = GgufTokenizer::from_metadata(&metadata).unwrap();
        assert!(minimax.minimax_pretokenizer);
        let actual = minimax.encode(text).unwrap();
        assert_ne!(actual, original);
        assert_eq!(minimax.decode(&actual), "café]~b]Ångström");
        assert!(actual.contains(&256), "special token must remain atomic");
        assert_eq!(
            minimax.encode("plain ASCII").unwrap(),
            legacy.encode("plain ASCII").unwrap()
        );
    }

    #[test]
    fn test_architecture_specific_chat_styles_precede_chatml() {
        let vocab = ["<s>", "</s>", "<|im_start|>", "<|im_end|>"];

        for (architecture, expected) in [
            ("qwen35", ChatStyle::Qwen3),
            ("qwen3moe", ChatStyle::Qwen3),
            ("nemotron_h_moe", ChatStyle::Nemotron),
        ] {
            let mut metadata = HashMap::new();
            metadata.insert(
                "tokenizer.ggml.tokens".to_string(),
                gguf_file::Value::Array(
                    vocab
                        .iter()
                        .map(|token| gguf_file::Value::String((*token).to_string()))
                        .collect(),
                ),
            );
            metadata.insert(
                "tokenizer.ggml.token_type".to_string(),
                gguf_file::Value::Array(vocab.iter().map(|_| gguf_file::Value::I32(4)).collect()),
            );
            metadata.insert(
                "tokenizer.ggml.model".to_string(),
                gguf_file::Value::String("gpt2".to_string()),
            );
            metadata.insert(
                "general.architecture".to_string(),
                gguf_file::Value::String(architecture.to_string()),
            );
            metadata.insert(
                "tokenizer.ggml.bos_token_id".to_string(),
                gguf_file::Value::U32(0),
            );
            metadata.insert(
                "tokenizer.ggml.eos_token_id".to_string(),
                gguf_file::Value::U32(1),
            );

            let tokenizer = GgufTokenizer::from_metadata(&metadata).unwrap();
            assert_eq!(tokenizer.chat_style, expected, "{architecture}");
        }
    }

    #[test]
    fn test_encode_adds_bos() {
        let tok = test_tokenizer_sp();
        let tokens = tok.encode("").unwrap();
        // BOS only for empty string (encode_sentencepiece returns empty for "")
        assert_eq!(tokens[0], 1); // BOS
    }

    #[test]
    fn test_gpt2_decode() {
        let map = build_gpt2_unicode_to_byte();
        // Ġ (U+0120) should map to byte 32 = space
        assert_eq!(map.get(&'\u{0120}'), Some(&32u8));
        // Ċ (U+010A) should map to byte 10 = newline
        assert_eq!(map.get(&'\u{010A}'), Some(&10u8));
        // Regular ASCII chars map to themselves
        assert_eq!(map.get(&'A'), Some(&65u8));
        assert_eq!(map.get(&'z'), Some(&122u8));
    }

    #[test]
    fn test_gpt2_decode_preserves_literal_space() {
        let mut vocab_to_id = HashMap::new();
        vocab_to_id.insert(" hello".to_string(), 0);
        let tok = GgufTokenizer {
            vocab: vec![" hello".to_string()],
            vocab_to_id,
            scores: Vec::new(),
            merge_ranks: HashMap::new(),
            ignore_merges: false,
            minimax_pretokenizer: false,
            glm_pretokenizer: false,
            joyai_pretokenizer: false,
            deepseek_llm_pretokenizer: false,
            qwen_pretokenizer: None,
            special_tokens: Vec::new(),
            space_marker: None,
            model: TokenizerModel::Gpt2,
            chat_style: ChatStyle::Raw,
            qwen_default_think_prefix: false,
            qwen_default_closed_thinking: false,
            qwen_chatml_default_system: false,
            qwen_default_xhigh_reasoning: false,
            qwen_xml_tools: false,
            bos_id: 0,
            eos_id: 0,
            stop_ids: vec![0],
            add_bos: false,
            byte_tokens: HashMap::new(),
            gpt2_unicode_to_byte: build_gpt2_unicode_to_byte(),
            gpt2_byte_to_unicode: build_gpt2_byte_to_unicode(),
            tool_parser_factory: std::sync::OnceLock::new(),
        };

        assert_eq!(tok.decode(&[0]), " hello");
        assert_eq!(tok.decode_token(0), " hello");
    }

    #[test]
    fn test_gpt2_ignore_merges_prefers_whole_pretoken() {
        let vocab = ["a", "b", "c", "ab", "abc"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let vocab_to_id = vocab
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as u32))
            .collect();
        let tok = GgufTokenizer {
            vocab,
            vocab_to_id,
            scores: Vec::new(),
            merge_ranks: HashMap::from([(("a".to_string(), "b".to_string()), 0)]),
            ignore_merges: true,
            minimax_pretokenizer: false,
            glm_pretokenizer: false,
            joyai_pretokenizer: false,
            deepseek_llm_pretokenizer: false,
            qwen_pretokenizer: None,
            special_tokens: Vec::new(),
            space_marker: None,
            model: TokenizerModel::Gpt2,
            chat_style: ChatStyle::Raw,
            qwen_default_think_prefix: false,
            qwen_default_closed_thinking: false,
            qwen_chatml_default_system: false,
            qwen_default_xhigh_reasoning: false,
            qwen_xml_tools: false,
            bos_id: 0,
            eos_id: 0,
            stop_ids: vec![0],
            add_bos: false,
            byte_tokens: HashMap::new(),
            gpt2_unicode_to_byte: build_gpt2_unicode_to_byte(),
            gpt2_byte_to_unicode: build_gpt2_byte_to_unicode(),
            tool_parser_factory: std::sync::OnceLock::new(),
        };

        assert_eq!(tok.encode("abc").unwrap(), vec![4]);
    }

    #[test]
    fn test_stream_decode_buffers_split_utf8_codepoint() {
        let byte_to_unicode = build_gpt2_byte_to_unicode();
        let encode_bytes = |bytes: &[u8]| {
            bytes
                .iter()
                .map(|byte| byte_to_unicode[byte])
                .collect::<String>()
        };
        let vocab = vec![encode_bytes(&[0xf0, 0x9f]), encode_bytes(&[0x98, 0x80])];
        let vocab_to_id = vocab
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as u32))
            .collect();
        let tok = GgufTokenizer {
            vocab,
            vocab_to_id,
            scores: Vec::new(),
            merge_ranks: HashMap::new(),
            ignore_merges: false,
            minimax_pretokenizer: false,
            glm_pretokenizer: false,
            joyai_pretokenizer: false,
            deepseek_llm_pretokenizer: false,
            qwen_pretokenizer: None,
            special_tokens: Vec::new(),
            space_marker: None,
            model: TokenizerModel::Gpt2,
            chat_style: ChatStyle::Raw,
            qwen_default_think_prefix: false,
            qwen_default_closed_thinking: false,
            qwen_chatml_default_system: false,
            qwen_default_xhigh_reasoning: false,
            qwen_xml_tools: false,
            bos_id: 0,
            eos_id: 0,
            stop_ids: vec![0],
            add_bos: false,
            byte_tokens: HashMap::new(),
            gpt2_unicode_to_byte: build_gpt2_unicode_to_byte(),
            gpt2_byte_to_unicode: byte_to_unicode,
            tool_parser_factory: std::sync::OnceLock::new(),
        };

        let mut pending = Vec::new();
        assert_eq!(tok.decode_stream_token(0, &mut pending), "");
        assert_eq!(tok.decode_stream_token(1, &mut pending), "😀");
        assert!(pending.is_empty());
        assert_eq!(tok.decode(&[0, 1]), "😀");
    }

    #[test]
    fn test_parse_merge_pair_with_leading_space_token() {
        let mut vocab_to_id = HashMap::new();
        vocab_to_id.insert(" ".to_string(), 0);
        vocab_to_id.insert("w".to_string(), 1);
        vocab_to_id.insert(" w".to_string(), 2);

        let pair = parse_merge_pair("  w", &vocab_to_id);
        assert_eq!(pair, Some((" ".to_string(), "w".to_string())));
    }

    #[test]
    fn test_pretokenize() {
        let tok = GgufTokenizer {
            vocab: Vec::new(),
            vocab_to_id: HashMap::new(),
            scores: Vec::new(),
            merge_ranks: HashMap::new(),
            ignore_merges: false,
            minimax_pretokenizer: false,
            glm_pretokenizer: false,
            joyai_pretokenizer: false,
            deepseek_llm_pretokenizer: false,
            qwen_pretokenizer: None,
            special_tokens: Vec::new(),
            space_marker: None,
            model: TokenizerModel::Gpt2,
            chat_style: ChatStyle::Raw,
            qwen_default_think_prefix: false,
            qwen_default_closed_thinking: false,
            qwen_chatml_default_system: false,
            qwen_default_xhigh_reasoning: false,
            qwen_xml_tools: false,
            bos_id: 0,
            eos_id: 0,
            stop_ids: vec![0],
            add_bos: false,
            byte_tokens: HashMap::new(),
            gpt2_unicode_to_byte: HashMap::new(),
            gpt2_byte_to_unicode: HashMap::new(),
            tool_parser_factory: std::sync::OnceLock::new(),
        };

        let words = tok.pretokenize("hello twoSum(nums, target) 2026\n\nok!");
        assert_eq!(
            words,
            vec![
                "hello", " twoSum", "(nums", ",", " target", ")", " 202", "6", "\n\n", "ok", "!"
            ]
        );
    }

    #[test]
    fn qwen_splits_indentation_json_digits_and_combining_marks() {
        let cases: &[(&str, &[&str])] = &[
            (
                "    let x = 12;\n",
                &["   ", " let", " x", " =", " ", "1", "2", ";\n"],
            ),
            (
                "{\n  \"x\": 123\n}\n",
                &[
                    "{\n", " ", " \"", "x", "\":", " ", "1", "2", "3", "\n", "}\n",
                ],
            ),
            ("\tword  next\r\n", &["\tword", " ", " next", "\r\n"]),
            ("12 ٣٤ ½²", &["1", "2", " ", "٣", "٤", " ", "½", "²"]),
            ("I'M we've", &["I", "'M", " we", "'ve"]),
            (" \n\n    x  ", &[" \n\n", "   ", " x", "  "]),
        ];
        for &(text, expected) in cases {
            for preset in [QwenPretokenizer::V2, QwenPretokenizer::V35] {
                let actual = pretokenize_qwen(text, preset);
                assert_eq!(actual, expected, "{preset:?}: {text:?}");
                assert_eq!(actual.concat(), text);
            }
        }
        assert_eq!(
            pretokenize_qwen("e\u{301}lan", QwenPretokenizer::V2),
            ["e", "\u{301}lan"]
        );
        assert_eq!(
            pretokenize_qwen("e\u{301}lan", QwenPretokenizer::V35),
            ["e\u{301}lan"]
        );
    }

    #[test]
    fn qwen_metadata_blocks_cross_split_bpe_merges_without_changing_legacy() -> Result<()> {
        // These valid merge entries make the old whitespace and digit splits
        // observably wrong. A byte-only vocabulary would hide this defect.
        let bytes = build_gpt2_byte_to_unicode();
        let mut vocab: Vec<_> = (0..=255u8).map(|b| bytes[&b].to_string()).collect();
        vocab.extend(["ĠĠ", "ĠĠĠ", "ĠĠĠĠ", "Ġa", "12", "Ġ1"].map(str::to_owned));
        let mut metadata = HashMap::from([
            (
                "general.architecture".into(),
                gguf_file::Value::String("qwen2".into()),
            ),
            (
                "tokenizer.ggml.model".into(),
                gguf_file::Value::String("gpt2".into()),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                gguf_file::Value::Array(vocab.into_iter().map(gguf_file::Value::String).collect()),
            ),
            (
                "tokenizer.ggml.merges".into(),
                gguf_file::Value::Array(
                    ["Ġ Ġ", "ĠĠ Ġ", "ĠĠ ĠĠ", "Ġ a", "1 2", "Ġ 1"]
                        .into_iter()
                        .map(|s| gguf_file::Value::String(s.into()))
                        .collect(),
                ),
            ),
            (
                "tokenizer.ggml.add_bos_token".into(),
                gguf_file::Value::Bool(false),
            ),
        ]);
        let legacy = GgufTokenizer::from_metadata(&metadata)?;
        assert_eq!(legacy.encode("    a")?, [258, 97]);
        assert_eq!(legacy.encode("12 12")?, [260, 32, 260]);
        for preset in ["qwen2", "qwen35", "deepseek-r1-qwen"] {
            metadata.insert(
                "tokenizer.ggml.pre".into(),
                gguf_file::Value::String(preset.into()),
            );
            let tokenizer = GgufTokenizer::from_metadata(&metadata)?;
            assert_eq!(tokenizer.encode("    a")?, [257, 259], "{preset}");
            assert_eq!(tokenizer.encode("12 12")?, [49, 50, 32, 49, 50], "{preset}");
            assert_eq!(tokenizer.decode(&tokenizer.encode("    a 12")?), "    a 12");
        }
        metadata.insert(
            "tokenizer.ggml.pre".into(),
            gguf_file::Value::String("deepseek-llm".into()),
        );
        let deepseek = GgufTokenizer::from_metadata(&metadata)?;
        assert_eq!(deepseek.encode("    a")?, [257, 259]);
        assert_eq!(deepseek.encode("12 12")?, [260, 32, 260]);
        Ok(())
    }

    #[test]
    fn deepseek_llm_preserves_ordered_newline_and_word_boundaries() {
        let cases: &[(&str, &[&str])] = &[
            (
                "    let x = 12;\n",
                &["   ", " let", " x", " =", " ", "12", ";", "\n"],
            ),
            (
                "{\n  \"x\": 123\n}\n",
                &[
                    "{", "\n", " ", " \"", "x", "\":", " ", "123", "\n", "}", "\n",
                ],
            ),
            (
                "café 中文123\r\n",
                &["caf", "é", " ", "中文", "123", "\r", "\n"],
            ),
            ("\tword  next  ", &["\tword", " ", " next", "  "]),
        ];
        for &(text, expected) in cases {
            let actual = pretokenize_deepseek_llm(text);
            assert_eq!(actual, expected, "{text:?}");
            assert_eq!(actual.concat(), text);
        }
    }

    #[test]
    fn bos_defaults_follow_tokenizer_model_and_presets_with_explicit_override() -> Result<()> {
        let mut metadata = HashMap::from([
            (
                "general.architecture".into(),
                gguf_file::Value::String("qwen35".into()),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                gguf_file::Value::Array(
                    ["<s>", "</s>", "a"]
                        .into_iter()
                        .map(|s| gguf_file::Value::String(s.into()))
                        .collect(),
                ),
            ),
            (
                "tokenizer.ggml.bos_token_id".into(),
                gguf_file::Value::U32(0),
            ),
            (
                "tokenizer.ggml.eos_token_id".into(),
                gguf_file::Value::U32(1),
            ),
        ]);
        for (model, preset, default) in [
            ("gpt2", "qwen35", false),
            ("gpt2", "qwen2", false),
            ("gpt2", "deepseek-llm", false),
            ("gpt2", "default", false),
            ("gpt2", "llama-bpe", true),
            ("gpt2", "pixtral", true),
            ("gpt2", "tekken", true),
            ("llama", "default", true),
        ] {
            metadata.insert(
                "tokenizer.ggml.model".into(),
                gguf_file::Value::String(model.into()),
            );
            metadata.insert(
                "tokenizer.ggml.pre".into(),
                gguf_file::Value::String(preset.into()),
            );
            metadata.remove("tokenizer.ggml.add_bos_token");
            let tokenizer = GgufTokenizer::from_metadata(&metadata)?;
            assert_eq!(tokenizer.add_bos, default, "{model}/{preset}");
            assert_eq!(
                tokenizer.encode("")?,
                if default { vec![0] } else { vec![] }
            );
            for explicit in [true, false] {
                metadata.insert(
                    "tokenizer.ggml.add_bos_token".into(),
                    gguf_file::Value::Bool(explicit),
                );
                assert_eq!(
                    GgufTokenizer::from_metadata(&metadata)?.add_bos,
                    explicit,
                    "{model}/{preset}"
                );
            }
        }
        Ok(())
    }
    #[test]
    fn native_tools_keep_types_reasoning_and_hide_public_ids() {
        let messages = vec![
            ChatTemplateMessage::new("user", "Look up stock."),
            ChatTemplateMessage::new("assistant", "")
                .with_reasoning_content("Need the stock before the total.")
                .with_tool_calls([ChatTemplateToolCall::new(
                    "lookup",
                    r#"{"sku":"00123","count":2}"#,
                )
                .with_id("random-public-id")]),
            ChatTemplateMessage::new("tool", r#"{"stock":7}"#)
                .with_tool_call_id("random-public-id"),
        ];
        for style in [ChatStyle::Qwen3, ChatStyle::Glm5] {
            let mut tokenizer = test_tokenizer_sp();
            tokenizer.chat_style = style;
            tokenizer.qwen_default_think_prefix = true;
            tokenizer.qwen_xml_tools = true;
            let prompt = tokenizer
                .format_chat_with_tool_options(
                    &messages,
                    &ChatTemplateOptions::new([ChatTemplateTool::new("lookup")
                        .with_parameters_json(r#"{"type":"object","properties":{"sku":{"type":"string"},"count":{"type":"integer"}},"additionalProperties":false}"#)]),
                )
                .unwrap();
            assert!(prompt.contains("Need the stock before the total."));
            assert!(!prompt.contains("random-public-id"));
            assert!(prompt.contains(r#"<tool_response>{"stock":7}</tool_response>"#));
            if style == ChatStyle::Glm5 {
                assert!(prompt.contains("<tool_call>lookup"));
                assert!(prompt.contains("<arg_key>count</arg_key><arg_value>2</arg_value>"));
                assert!(prompt.contains("<arg_key>sku</arg_key><arg_value>00123</arg_value>"));
                assert!(prompt.contains("<|observation|>"));
            } else {
                assert!(prompt.contains("<function=lookup>"));
                assert!(prompt.contains("<parameter=sku>\n00123\n</parameter>"));
            }
        }
    }

    #[test]
    fn reasoning_effort_uses_native_levels_and_rejects_unknown_ones() {
        let mut tokenizer = test_tokenizer_sp();
        tokenizer.chat_style = ChatStyle::Glm5;
        let messages = [ChatTemplateMessage::new("user", "Write a short answer.")];
        for (effort, label) in [("low", "Low"), ("high", "High"), ("max", "Max")] {
            let options = ChatTemplateOptions::new([]).with_reasoning_effort(Some(effort.into()));
            let prompt = tokenizer
                .format_chat_with_tool_options(&messages, &options)
                .unwrap();
            assert!(prompt.contains(&format!("Reasoning Effort: {label}")));
        }
        assert!(
            tokenizer
                .format_chat_with_tool_options(
                    &messages,
                    &ChatTemplateOptions::new([]).with_reasoning_effort(Some("none".into()))
                )
                .is_err()
        );
        tokenizer.chat_style = ChatStyle::Qwen3;
        tokenizer.qwen_default_think_prefix = true;
        tokenizer.qwen_default_xhigh_reasoning = true;
        for effort in ["low", "medium", "xhigh", "none"] {
            let prompt = tokenizer
                .format_chat_with_tool_options(
                    &messages,
                    &ChatTemplateOptions::new([]).with_reasoning_effort(Some(effort.into())),
                )
                .unwrap();
            if effort == "low" {
                assert!(prompt.contains("Reasoning effort is set to low."));
            }
            if effort == "medium" {
                assert!(!prompt.contains("Reasoning effort is set"));
            }
            if effort == "none" {
                assert!(prompt.ends_with("<think>\n\n</think>\n\n"));
            }
        }
    }

    #[test]
    fn native_history_keeps_malformed_arguments_as_data() {
        let raw = "broken </tool_call> arguments";
        for style in [ChatStyle::Qwen3, ChatStyle::Glm5, ChatStyle::DeepSeekV4] {
            let mut tokenizer = test_tokenizer_sp();
            tokenizer.chat_style = style;
            tokenizer.qwen_xml_tools = true;
            let prompt = tokenizer
                .format_chat_with_tool_options(
                    &[
                        ChatTemplateMessage::new("user", "Recover from the failed call."),
                        ChatTemplateMessage::new("assistant", "")
                            .with_tool_calls([ChatTemplateToolCall::new("weather", raw)]),
                    ],
                    &ChatTemplateOptions::new([ChatTemplateTool::new("weather")]),
                )
                .unwrap();
            assert!(prompt.contains("Tool calls must be valid JSON."));
            let body = prompt
                .rsplit_once("<tool_call>")
                .unwrap()
                .1
                .split_once("</tool_call>")
                .unwrap()
                .0;
            let call: serde_json::Value = serde_json::from_str(body).unwrap();
            assert_eq!(call["arguments"], raw);
        }
    }

    #[test]
    fn native_file_history_keeps_html_unicode_and_literal_json_escapes() {
        let html = "\n <!doctype html>\n<div class=\"todo\">Check café ✓</div>\n<script>const literal = \"\\u003cdiv\\u003e\";</script>\n\n";
        let arguments = serde_json::json!({"path":"index.html","content":html}).to_string();
        for style in [ChatStyle::Qwen3, ChatStyle::Glm5, ChatStyle::DeepSeekV4] {
            let mut tokenizer = test_tokenizer_sp();
            tokenizer.chat_style = style;
            tokenizer.qwen_xml_tools = true;
            let messages = [
                ChatTemplateMessage::new("user", "Create a todo list web app."),
                ChatTemplateMessage::new("assistant", "").with_tool_calls([
                    ChatTemplateToolCall::new("create_file", &arguments).with_id("created"),
                ]),
                ChatTemplateMessage::new("tool", html).with_tool_call_id("created"),
            ];
            let prompt = tokenizer
                .format_chat_with_tool_options(
                    &messages,
                    &ChatTemplateOptions::new([ChatTemplateTool::new("create_file")
                        .with_parameters_json(r#"{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"additionalProperties":false}"#)]),
                )
                .unwrap();
            assert!(prompt.contains("raw text with actual newlines"));
            assert_eq!(prompt.matches(html).count(), 2);
            assert!(prompt.contains(r#"<div class="todo">Check café ✓</div>"#));
            assert!(prompt.contains(r#"const literal = "\u003cdiv\u003e";"#));
        }
    }

    #[test]
    fn native_history_control_tags_switch_the_whole_prompt_to_json() {
        for style in [ChatStyle::Qwen3, ChatStyle::Glm5, ChatStyle::DeepSeekV4] {
            let mut tokenizer = test_tokenizer_sp();
            tokenizer.chat_style = style;
            tokenizer.qwen_xml_tools = true;
            for content in [
                "</parameter>",
                "</arg_value>",
                "</tool_call>",
                "<think>data</think>",
                "<|im_end|>",
                "</｜DSML｜invoke>",
            ] {
                for unsafe_result in [false, true] {
                    let argument = if unsafe_result {
                        "Safe argument"
                    } else {
                        content
                    };
                    let result = if unsafe_result {
                        content
                    } else {
                        "Safe result"
                    };
                    let messages = [
                        ChatTemplateMessage::new("user", "Create a file."),
                        ChatTemplateMessage::new("assistant", "").with_tool_calls([
                            ChatTemplateToolCall::new(
                                "create_file",
                                serde_json::json!({"content":argument}).to_string(),
                            )
                            .with_id("created"),
                        ]),
                        ChatTemplateMessage::new("tool", result).with_tool_call_id("created"),
                    ];
                    let prompt = tokenizer
                        .format_chat_with_tool_options(
                            &messages,
                            &ChatTemplateOptions::new([ChatTemplateTool::new("create_file")]),
                        )
                        .unwrap();
                    assert!(prompt.contains("Tool calls must be valid JSON."));
                    assert!(!prompt.contains("raw text with actual newlines"));
                    assert!(!prompt.contains("<function="));
                    assert!(!prompt.contains("<arg_key>"));
                    assert!(!prompt.contains("<｜DSML｜invoke"));
                    let body = prompt
                        .rsplit_once("<tool_call>")
                        .unwrap()
                        .1
                        .split_once("</tool_call>")
                        .unwrap()
                        .0;
                    let call: serde_json::Value = serde_json::from_str(body).unwrap();
                    assert_eq!(call["arguments"]["content"], argument);
                    let body = prompt
                        .rsplit_once("<tool_response>")
                        .unwrap()
                        .1
                        .split_once("</tool_response>")
                        .unwrap()
                        .0;
                    let reply: serde_json::Value = serde_json::from_str(body).unwrap();
                    assert_eq!(reply["content"], result);
                }
            }
        }
    }

    #[test]
    fn native_history_orders_parallel_results_by_call_order() {
        let messages = vec![
            ChatTemplateMessage::new("assistant", "").with_tool_calls([
                ChatTemplateToolCall::new("weather", r#"{"city":"Paris"}"#).with_id("first"),
                ChatTemplateToolCall::new("weather", r#"{"city":"Rome"}"#).with_id("second"),
            ]),
            ChatTemplateMessage::new("tool", "Rome result").with_tool_call_id("second"),
            ChatTemplateMessage::new("tool", "Paris result").with_tool_call_id("first"),
        ];
        let ordered = order_tool_results(messages);
        assert_eq!(ordered[1].content, "Paris result");
        assert_eq!(ordered[2].content, "Rome result");
    }

    #[test]
    fn legacy_readers_keep_json_calls_on_native_models() {
        for style in [ChatStyle::Qwen3, ChatStyle::Glm5, ChatStyle::DeepSeekV4] {
            let mut tokenizer = test_tokenizer_sp();
            tokenizer.chat_style = style;
            tokenizer.qwen_xml_tools = true;
            let prompt = tokenizer
                .format_chat_with_tool_options(
                    &sample_tool_messages(),
                    &sample_tool_options().with_native_tool_formats(false),
                )
                .unwrap();
            assert!(prompt.contains("one JSON object inside <tool_call></tool_call>"));
            assert!(!prompt.contains("<function="));
            assert!(!prompt.contains("<arg_key>"));
            assert!(!prompt.contains("<｜DSML｜invoke"));
            assert!(prompt.contains("\\\"arguments\\\"") || prompt.contains("\"arguments\""));
        }
    }

    #[test]
    fn file_tool_prompts_use_only_the_selected_formats_string_rules() {
        let options = ChatTemplateOptions::new([ChatTemplateTool::new("create_file")
            .with_parameters_json(
                r#"{"type":"object","required":["path","content"],"properties":{"path":{"type":"string"},"content":{"type":"string"}},"additionalProperties":false}"#,
            )])
        .with_parallel_tool_calls(false);
        let messages = [ChatTemplateMessage::new(
            "user",
            "Create a todo list web app.",
        )];
        for style in [ChatStyle::Qwen3, ChatStyle::Glm5] {
            let mut tokenizer = test_tokenizer_sp();
            tokenizer.chat_style = style;
            tokenizer.qwen_xml_tools = true;
            for native in [false, true] {
                let prompt = tokenizer
                    .format_chat_with_tool_options(
                        &messages,
                        &options.clone().with_native_tool_formats(native),
                    )
                    .unwrap();
                assert!(prompt.contains("emit exactly one call and end the turn"));
                assert!(prompt.contains("Wait for its result before another call."));
                if native {
                    assert!(prompt.contains("raw text with actual newlines"));
                    assert!(prompt.contains("Do not wrap strings in JSON quotes"));
                    assert!(!prompt.contains("Inside JSON string values"));
                    let closing = if style == ChatStyle::Qwen3 {
                        "Every <parameter=...> needs its own </parameter>."
                    } else {
                        "Every <arg_key> and <arg_value> needs its matching closing tag."
                    };
                    assert!(prompt.contains(closing));
                } else {
                    assert!(prompt.contains("Tool calls must be valid JSON."));
                    assert!(
                        prompt.contains(r#"newlines as \n, quotes as \", and backslashes as \\"#)
                    );
                    assert!(!prompt.contains("raw text with actual newlines"));
                }
            }
        }
    }

    #[test]
    fn json_tool_guidance_has_exact_escape_bytes_and_object_closure() {
        let expected = "Tool calls must be valid JSON. Inside JSON string values, encode newlines as \\n, quotes as \\\", and backslashes as \\\\. Never put an unescaped newline or quote inside a JSON string.";
        let closure = "Close the arguments object with }, then the outer call object with }, before </tool_call>. After these braces, emit only </tool_call>; do not add native function, parameter, or invoke closing tags. Tags inside JSON string values are file text.";
        for options in [sample_tool_options(), ChatTemplateOptions::new([])] {
            let instruction = canonical_tool_instruction(
                &options.with_parallel_tool_calls(false),
                ToolFormat::Json,
            )
            .unwrap();
            let line = instruction
                .lines()
                .find(|line| line.starts_with("Tool calls must be valid JSON."))
                .unwrap();
            assert_eq!(line.as_bytes(), expected.as_bytes());
            let newline = line
                .split_once("newlines as ")
                .unwrap()
                .1
                .split_once(", quotes as ")
                .unwrap()
                .0;
            let quote = line
                .split_once("quotes as ")
                .unwrap()
                .1
                .split_once(", and backslashes as ")
                .unwrap()
                .0;
            let backslash = line
                .split_once("backslashes as ")
                .unwrap()
                .1
                .split_once(". Never ")
                .unwrap()
                .0;
            assert_eq!(newline.as_bytes(), &[92, 110]);
            assert_eq!(quote.as_bytes(), &[92, 34]);
            assert_eq!(backslash.as_bytes(), &[92, 92]);
            // The exact examples must decode to the intended file characters,
            // not to a backslash followed by another printable character.
            for (escape, character) in [(newline, "\n"), (quote, "\""), (backslash, "\\")] {
                assert_eq!(
                    serde_json::from_str::<String>(&format!("\"{escape}\"")).unwrap(),
                    character
                );
            }
            let line = instruction
                .lines()
                .find(|line| line.starts_with("Close the arguments object"))
                .unwrap();
            assert_eq!(line, closure);
            for native_tag in ["</function>", "</parameter>", "</invoke>", "</arg_value>"] {
                assert!(!instruction.contains(native_tag));
            }
        }
    }

    #[test]
    fn portable_file_tool_history_round_trips_multiline_html() {
        let html = "<!doctype html>\n<div class=\"todo\">One \"quoted\" task</div>\n<script>\nconst file = \"C:\\\\notes\";\nconst markers = \"</tool_call><think>literal</think>\";\n</script>\n";
        let arguments = serde_json::json!({"path":"index.html","content":html}).to_string();
        let mut tokenizer = test_tokenizer_sp();
        tokenizer.chat_style = ChatStyle::Qwen3;
        tokenizer.qwen_xml_tools = true;
        let messages = [
            ChatTemplateMessage::new("user", "Create a todo list web app."),
            ChatTemplateMessage::new("assistant", "")
                .with_tool_calls([ChatTemplateToolCall::new("create_file", arguments)]),
        ];
        let prompt = tokenizer
            .format_chat_with_tool_options(
                &messages,
                &ChatTemplateOptions::new([ChatTemplateTool::new("create_file")])
                    .with_native_tool_formats(false)
                    .with_parallel_tool_calls(false),
            )
            .unwrap();
        let body = prompt
            .rsplit_once("<tool_call>")
            .unwrap()
            .1
            .split_once("</tool_call>")
            .unwrap()
            .0;
        let call: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(call["name"], "create_file");
        assert_eq!(call["arguments"]["path"], "index.html");
        assert_eq!(call["arguments"]["content"], html);
        assert!(body.contains(r#"\n"#));
        assert!(body.contains(r#"\"todo\""#));
        assert!(!body.contains('\n'));
        assert!(!body.contains("<think>"));
    }

    #[test]
    fn changing_public_call_ids_does_not_change_model_prompt() {
        for style in [
            ChatStyle::Qwen3,
            ChatStyle::Glm5,
            ChatStyle::DeepSeekV4,
            ChatStyle::Nemotron,
        ] {
            let mut tokenizer = test_tokenizer_sp();
            tokenizer.chat_style = style;
            tokenizer.qwen_xml_tools = true;
            let first = sample_tool_messages();
            let mut second = first.clone();
            for message in &mut second {
                for call in &mut message.tool_calls {
                    call.id = Some("call_another_random_id".into());
                }
                if message.tool_call_id.is_some() {
                    message.tool_call_id = Some("call_another_random_id".into());
                }
            }
            assert_eq!(
                tokenizer
                    .format_chat_with_tool_options(&first, &sample_tool_options())
                    .unwrap(),
                tokenizer
                    .format_chat_with_tool_options(&second, &sample_tool_options())
                    .unwrap(),
                "{style:?}"
            );
        }
    }
}

#[cfg(test)]
mod muse_artifact_tests {
    use super::*;
    use serde_json::Value;

    /// Compares the public GGUF vocabulary against an independent HF BPE
    /// reader and the embedded Jinja template. No model tensors are needed.
    #[test]
    #[ignore = "requires DINF_MUSE_TEST_GGUF and DINF_MUSE_TEST_REFERENCE"]
    fn muse_gguf_reference_matches_template_and_ids() {
        let gguf = std::env::var("DINF_MUSE_TEST_GGUF").expect("set DINF_MUSE_TEST_GGUF");
        let reference =
            std::env::var("DINF_MUSE_TEST_REFERENCE").expect("set DINF_MUSE_TEST_REFERENCE");
        let tokenizer = GgufTokenizer::from_gguf(Path::new(&gguf)).unwrap();
        assert_eq!(tokenizer.chat_style, ChatStyle::MuseSpark);
        assert!(
            !tokenizer
                .stop_ids
                .contains(&tokenizer.token_id("<|eom|>").unwrap())
        );
        assert!(
            tokenizer
                .stop_ids
                .contains(&tokenizer.token_id("<|eot|>").unwrap())
        );
        let reference: Value = serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
        for (index, case) in reference["cases"].as_array().unwrap().iter().enumerate() {
            let messages = case["messages"]
                .as_array()
                .unwrap()
                .iter()
                .map(|message| {
                    let mut native = ChatTemplateMessage::new(
                        message["role"].as_str().unwrap(),
                        message["content"].as_str().unwrap_or_default(),
                    );
                    native.reasoning_content =
                        message["reasoning_content"].as_str().map(str::to_owned);
                    native.tool_call_id = message["tool_call_id"].as_str().map(str::to_owned);
                    if let Some(calls) = message["tool_calls"].as_array() {
                        native.tool_calls = calls
                            .iter()
                            .map(|call| {
                                let mut native = ChatTemplateToolCall::new(
                                    call["function"]["name"].as_str().unwrap(),
                                    call["function"]["arguments"].to_string(),
                                );
                                native.id = call["id"].as_str().map(str::to_owned);
                                native
                            })
                            .collect();
                    }
                    native
                })
                .collect::<Vec<_>>();
            let tools = case["tools"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|tool| {
                    ChatTemplateTool::new(tool["function"]["name"].as_str().unwrap())
                        .with_description(tool["function"]["description"].as_str().unwrap())
                        .with_parameters_json(tool["function"]["parameters"].to_string())
                })
                .collect::<Vec<_>>();
            let options = ChatTemplateOptions::new(tools)
                .with_reasoning_effort(case["reasoning_effort"].as_str().map(str::to_owned));
            let date_before = chrono::Local::now().date_naive();
            let prompt = tokenizer
                .format_chat_with_tool_options(&messages, &options)
                .unwrap();
            let date_after = chrono::Local::now().date_naive();
            let expected = case["expected_prompt"]
                .as_str()
                .unwrap()
                .strip_prefix("<|begin_of_text|>")
                .unwrap();
            // The saved Jinja oracle includes its generation date. Check the
            // live date, then hold it fixed for exact prompt and token parity.
            let (prefix, dated) = expected.split_once("\n\nCurrent date: ").unwrap();
            let fixture_date = dated.split_once(".\n\n").unwrap().0;
            chrono::NaiveDate::parse_from_str(fixture_date, "%Y-%m-%d").unwrap();
            let prompt = [date_before, date_after]
                .into_iter()
                .find_map(|date| {
                    let live = format!("{prefix}\n\nCurrent date: {date}.\n\n");
                    prompt.strip_prefix(&live).map(|suffix| {
                        format!("{prefix}\n\nCurrent date: {fixture_date}.\n\n{suffix}")
                    })
                })
                .expect("Muse template must render the current date");
            assert_eq!(prompt, expected, "template case {index}");
            let expected_ids = case["expected_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect::<Vec<_>>();
            assert_eq!(
                tokenizer.encode(&prompt).unwrap(),
                expected_ids,
                "template IDs case {index}"
            );
        }
        for sample in reference["samples"].as_array().unwrap() {
            let expected = sample["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect::<Vec<_>>();
            assert_eq!(
                tokenizer.encode(sample["text"].as_str().unwrap()).unwrap(),
                expected,
                "sample {}",
                sample["text"]
            );
        }
        let user = [ChatTemplateMessage::new("user", "Hi")];
        let legacy = tokenizer.format_chat_with_thinking_state(
            &[("user".into(), "Hi".into())],
            None,
            Some("low"),
        );
        assert!(legacy.contains("Reasoning strength: 8."));
        for unsupported in ["none", "minimal", "max"] {
            assert!(
                tokenizer
                    .format_chat_with_tool_options(
                        &user,
                        &ChatTemplateOptions::new([])
                            .with_reasoning_effort(Some(unsupported.into()))
                    )
                    .is_err()
            );
        }
    }
}
