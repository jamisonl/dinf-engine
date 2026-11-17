use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{Result, ensure};
use candle_core::Device;

use crate::gguf_loader::{GgufMetadata, LayerRange, read_gguf_metadata};
use crate::options::EngineOptions;
use crate::sampling::SamplingParams;
use crate::tokenizer::{ChatTemplateMessage, ChatTemplateOptions, GgufTokenizer};
use crate::{PipelineStage, StagePosition};

/// A loaded GGUF model stage and its tokenizer.
///
/// `Engine` is the convenience, single-stage API. It owns the model weights,
/// tokenizer, and stage cache. The high-level text methods require a
/// [`StagePosition::Solo`](crate::StagePosition::Solo) stage; use
/// [`Engine::builder`] with no layer range (or [`Engine::load`]) for that
/// mode. For split-range (pipeline) execution, use the lower-level
/// [`PipelineStage`](crate::PipelineStage) API instead.
/// These convenience generation methods decode one token at a time. Use
/// [`crate::continuous::GenerationSession`] to execute an attached DFlash draft.
pub struct Engine {
    path: PathBuf,
    metadata: GgufMetadata,
    range: LayerRange,
    tokenizer: GgufTokenizer,
    stage: PipelineStage,
}

/// Options used to load an [`Engine`].
///
/// A builder does not read the GGUF file until [`EngineBuilder::build`] is
/// called, so it is inexpensive to construct and pass through configuration
/// code. Unless [`EngineBuilder::layers`] is supplied, the builder loads every
/// executable model layer and produces a solo engine.
pub struct EngineBuilder {
    path: PathBuf,
    device: Device,
    layers: Option<Range<usize>>,
    kv_quant_bits: Option<u8>,
    max_kv_tokens: Option<usize>,
    options: EngineOptions,
}

/// Tokens and decoded text returned by a generation call.
///
/// `tokens` contains the sampled token IDs, including a sampled stop token if
/// generation ended because of one. `text` is the UTF-8-lossy decoded text of
/// non-stop tokens; model control/stop tokens are not appended to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationOutput {
    /// Sampled token IDs in generation order.
    pub tokens: Vec<u32>,
    /// Decoded text produced by the generated non-stop tokens.
    pub text: String,
}

impl Engine {
    /// Load a full GGUF model on `device`.
    ///
    /// This is equivalent to `Engine::builder(path, device).build()` with
    /// default KV-cache settings. The GGUF metadata, tokenizer, and model
    /// weights are read before this method returns.
    ///
    /// # Errors
    ///
    /// Returns an error when the GGUF file is missing or malformed, the
    /// architecture or device is unsupported, or model tensors cannot be
    /// loaded.
    pub fn load(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        Self::builder(path, device).build()
    }

    /// Start configuring a full model or a contiguous stage before loading it.
    ///
    /// This method performs no file or device work. Call builder methods and
    /// then [`EngineBuilder::build`] to load the model.
    pub fn builder(path: impl AsRef<Path>, device: Device) -> EngineBuilder {
        EngineBuilder::new(path, device)
    }

    /// Return the GGUF path supplied when this engine was built.
    ///
    /// The path is retained as supplied; it is not guaranteed to be absolute
    /// or canonicalized.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Return normalized metadata parsed from the GGUF header.
    ///
    /// This metadata describes the executable model layout, including the
    /// effective layer count used for GLM5-Next files that contain an MTP
    /// block.
    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    /// Return the contiguous layer range loaded by this engine.
    ///
    /// A range covering all layers is a solo engine. A partial range is useful
    /// to callers using the lower-level pipeline API, but high-level
    /// [`Engine::generate`] methods reject it.
    pub fn layer_range(&self) -> &LayerRange {
        &self.range
    }

    /// Borrow the tokenizer associated with this model.
    ///
    /// The tokenizer is parsed from the same GGUF source as the model and can
    /// be used to inspect or reproduce prompt encoding and streamed decoding.
    pub fn tokenizer(&self) -> &GgufTokenizer {
        &self.tokenizer
    }

    /// Borrow the loaded pipeline stage for read-only inspection.
    ///
    /// Use [`Engine::stage_mut`] when lower-level cache or execution controls
    /// must be changed.
    pub fn stage(&self) -> &PipelineStage {
        &self.stage
    }

    /// Borrow the loaded pipeline stage mutably.
    ///
    /// This exposes lower-level controls such as warmup, cache limits, and
    /// speculative-draft attachment. Callers are responsible for preserving
    /// the stage's cache-position invariants when mixing these operations with
    /// high-level generation.
    pub fn stage_mut(&mut self) -> &mut PipelineStage {
        &mut self.stage
    }

    /// Generate text from a raw prompt. Each call starts with an empty cache.
    ///
    /// The prompt is encoded using the model's GGUF tokenizer, then generated
    /// autoregressively according to `params`. The returned text excludes
    /// tokenizer stop/control tokens.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty/invalid prompt, a non-solo engine, invalid
    /// sampling parameters, or a model/device execution failure. In
    /// particular, `params.max_tokens` must be positive.
    pub fn generate(&mut self, prompt: &str, params: &SamplingParams) -> Result<GenerationOutput> {
        let tokens = self.tokenizer.encode(prompt)?;
        self.generate_tokens(tokens, params, |_| true)
    }

    /// Generate from a model chat template. Each call starts with an empty cache.
    ///
    /// Each tuple is `(role, content)`. The tokenizer selects the chat format
    /// embedded in the GGUF vocabulary/template and handles model-specific
    /// BOS, turn, and reasoning markers.
    ///
    /// # Errors
    ///
    /// Returns the same generation and model/device errors as
    /// [`Engine::generate`], plus tokenizer/template errors.
    pub fn generate_chat(
        &mut self,
        messages: &[(String, String)],
        params: &SamplingParams,
    ) -> Result<GenerationOutput> {
        let tokens = self.tokenizer.encode_chat(messages)?;
        self.generate_tokens(tokens, params, |_| true)
    }

    /// Generate from a chat conversation carrying OpenAI-compatible tool
    /// declarations, tool history, and selection semantics.
    ///
    /// `messages` may contain assistant tool calls and tool-result messages;
    /// `options` supplies function declarations, tool choice, and parallel
    /// call policy. Tool arguments and schemas are carried as JSON text at the
    /// tokenizer boundary. This method only renders the prompt; it does not
    /// execute tools or parse tool calls from the generated output.
    ///
    /// # Errors
    ///
    /// Returns an error when message/tool bounds or tool-choice references are
    /// invalid, when the model template cannot render the conversation, or
    /// when generation fails. It also inherits the non-solo and
    /// `max_tokens > 0` requirements of [`Engine::generate`].
    pub fn generate_chat_with_tools(
        &mut self,
        messages: &[ChatTemplateMessage],
        options: &ChatTemplateOptions,
        params: &SamplingParams,
    ) -> Result<GenerationOutput> {
        let tokens = self
            .tokenizer
            .encode_chat_with_tool_options(messages, options)?;
        self.generate_tokens(tokens, params, |_| true)
    }

    /// Generate from token IDs and report each decoded text fragment.
    ///
    /// `tokens` are the already-tokenized prompt and must be non-empty. The
    /// callback receives each decoded, non-stop fragment as soon as it becomes
    /// valid UTF-8. It may return `false` to stop further model execution;
    /// returning `true` continues generation. A token that completes a
    /// multi-byte UTF-8 sequence may cause the callback to receive an empty
    /// fragment. Any final buffered fragment is appended to the returned
    /// [`GenerationOutput`] and is offered to the callback once more; its
    /// return value is ignored because generation has already ended.
    ///
    /// Stop-token IDs are still present in `GenerationOutput::tokens`, but
    /// are not decoded or passed to `on_text`.
    ///
    /// # Errors
    ///
    /// Returns an error when the engine is not a solo stage, `tokens` is empty
    /// or contains an ID outside the model vocabulary,
    /// `params.max_tokens` is zero, sampling parameters are invalid, or model
    /// execution/decoding fails. The stage cache is cleared before execution
    /// begins.
    pub fn generate_tokens(
        &mut self,
        tokens: Vec<u32>,
        params: &SamplingParams,
        mut on_text: impl FnMut(&str) -> bool,
    ) -> Result<GenerationOutput> {
        ensure!(
            self.stage.position() == StagePosition::Solo,
            "text generation requires an engine loaded with all model layers"
        );
        ensure!(!tokens.is_empty(), "prompt must contain at least one token");
        self.stage.validate_token_ids(&tokens)?;
        self.stage.validate_sampling_params(params)?;
        self.stage
            .validate_context_budget(tokens.len(), params.max_tokens)?;

        self.stage.clear_cache();
        let stop_ids = self.tokenizer.stop_token_ids().to_vec();
        let mut pending = Vec::new();
        let mut text = String::new();
        let generated = self.stage.generate(tokens, params, &stop_ids, 0, |token| {
            if stop_ids.contains(&token) {
                return true;
            }
            let fragment = self.tokenizer.decode_stream_token(token, &mut pending);
            text.push_str(&fragment);
            on_text(&fragment)
        })?;
        let fragment = self.tokenizer.finish_decode_stream(&mut pending);
        text.push_str(&fragment);
        if !fragment.is_empty() {
            let _ = on_text(&fragment);
        }
        Ok(GenerationOutput {
            tokens: generated,
            text,
        })
    }
}

impl EngineBuilder {
    /// Create a lazy engine configuration for `path` and `device`.
    ///
    /// No GGUF data is read and no tensors are allocated until
    /// [`EngineBuilder::build`] is called. The default configuration loads all
    /// executable layers, uses the device's normal KV representation, and has
    /// no explicit token-capacity limit.
    pub fn new(path: impl AsRef<Path>, device: Device) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            device,
            layers: None,
            kv_quant_bits: None,
            max_kv_tokens: None,
            options: EngineOptions::default(),
        }
    }

    /// Load only the given half-open transformer layer range.
    ///
    /// `start..end` is checked against the model's executable layer count at
    /// build time. A range beginning at zero includes the head/embedding, and
    /// a range ending at the total layer count includes the tail/output head.
    /// A partial range makes the resulting engine suitable for lower-level
    /// pipeline calls, not the high-level text-generation methods.
    ///
    /// # Errors
    ///
    /// [`EngineBuilder::build`] returns an error if the range is empty or lies
    /// outside the model.
    pub fn layers(mut self, layers: Range<usize>) -> Self {
        self.layers = Some(layers);
        self
    }

    /// Store KV vectors with the quality-qualified 8-bit TurboQuant profile.
    ///
    /// Validation is deferred until [`EngineBuilder::build`]. Only the 8-bit
    /// profile is quality-qualified for inference. Unsupported attention head
    /// dimensions fail explicitly instead of silently using an F16 cache.
    /// Power-of-two head dimensions above 16,384 are rejected at load time to
    /// bound allocations derived from model metadata.
    /// TurboQuant is currently incompatible with an attached DFlash draft.
    ///
    /// # Errors
    ///
    /// [`EngineBuilder::build`] returns an error for any bit width other than 8.
    pub fn kv_quant_bits(mut self, bits: u8) -> Self {
        self.kv_quant_bits = Some(bits);
        self
    }

    /// Limit the number of token positions retained in attention caches.
    ///
    /// When a forward pass would exceed `max`, the oldest attention positions
    /// are evicted before the pass. Absolute model positions continue to
    /// increase, so eviction does not reuse rotary positions. Recurrent
    /// (DeltaNet/Mamba) state is not reduced by this setting.
    ///
    /// # Errors
    ///
    /// [`EngineBuilder::build`] returns an error when `max` is zero.
    pub fn max_kv_tokens(mut self, max: usize) -> Self {
        self.max_kv_tokens = Some(max);
        self
    }

    /// Set the runtime options the loaded stage applies to requests.
    ///
    /// Validation is deferred until [`EngineBuilder::build`]. The options can
    /// be changed later with [`PipelineStage::set_options`] through
    /// [`Engine::stage_mut`].
    ///
    /// # Errors
    ///
    /// [`EngineBuilder::build`] returns an error when
    /// [`EngineOptions::validate`] rejects `options`.
    pub fn options(mut self, options: EngineOptions) -> Self {
        self.options = options;
        self
    }

    /// Read the GGUF header/tokenizer and materialize the configured stage.
    ///
    /// This is the first builder operation that performs I/O and allocates
    /// model tensors. It returns an engine only after metadata, tokenizer, KV
    /// settings, layer range, and architecture construction have all
    /// succeeded.
    ///
    /// # Errors
    ///
    /// Returns an error for an unreadable or malformed GGUF, an invalid layer
    /// range, unsupported KV quantization settings, tokenizer/architecture
    /// incompatibility, missing tensors, or device allocation/kernel errors.
    pub fn build(self) -> Result<Engine> {
        if let Some(bits) = self.kv_quant_bits {
            crate::turboquant::validate_inference_bits(bits)?;
        }
        if let Some(max) = self.max_kv_tokens {
            ensure!(max > 0, "max_kv_tokens must be positive");
        }
        self.options.validate()?;

        let metadata = read_gguf_metadata(&self.path)?;
        let total_layers = metadata.num_layers;
        let range = match self.layers {
            Some(layers) => {
                ensure!(
                    layers.start < layers.end && layers.end <= total_layers,
                    "layer range {}..{} is outside model range 0..{}",
                    layers.start,
                    layers.end,
                    total_layers
                );
                LayerRange::new(layers.start, layers.end, total_layers)
            }
            None => LayerRange::solo(total_layers),
        };
        let tokenizer = GgufTokenizer::from_gguf(&self.path)?;
        let mut stage =
            PipelineStage::from_gguf(&self.path, range.clone(), &self.device, self.kv_quant_bits)?;
        if let Some(max) = self.max_kv_tokens {
            stage.set_max_kv_tokens(max);
        }
        stage.set_options(self.options)?;

        Ok(Engine {
            path: self.path,
            metadata,
            range,
            tokenizer,
            stage,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architectures::StageInput;
    use candle_core::quantized::gguf_file::Value;
    use std::collections::{BTreeMap, HashMap};

    #[test]
    fn invalid_generation_controls_preserve_the_existing_stage_cache() -> Result<()> {
        let values = HashMap::from([
            ("general.architecture".into(), Value::String("llama".into())),
            ("llama.block_count".into(), Value::U32(1)),
            ("llama.embedding_length".into(), Value::U32(2)),
            ("llama.attention.head_count".into(), Value::U32(1)),
            ("llama.context_length".into(), Value::U32(16)),
            (
                "tokenizer.ggml.tokens".into(),
                Value::Array(vec![
                    Value::String("a".into()),
                    Value::String("</s>".into()),
                ]),
            ),
            ("tokenizer.ggml.bos_token_id".into(), Value::U32(0)),
            ("tokenizer.ggml.eos_token_id".into(), Value::U32(1)),
        ]);
        let mut engine = Engine {
            path: "test.gguf".into(),
            metadata: crate::gguf_loader::parse_gguf_metadata(&values)?,
            range: LayerRange::solo(1),
            tokenizer: GgufTokenizer::from_metadata(&values)?,
            stage: crate::session_state_tests::test_stage(),
        };
        engine.stage.vocab_size = 2;
        engine.stage.prefill(StageInput::Tokens(vec![0]), 1, 0)?;
        let allocated = engine.stage.total_kv_cache_allocation_tokens();
        let params = SamplingParams {
            max_tokens: 1,
            logit_bias: BTreeMap::from([(2, 1.0)]),
            ..Default::default()
        };
        let error = engine
            .generate_tokens(vec![0], &params, |_| true)
            .unwrap_err();
        assert!(error.to_string().contains("outside the model vocabulary"));
        assert_eq!(engine.stage.next_position(), 1);
        assert_eq!(engine.stage.total_kv_cache_allocation_tokens(), allocated);
        let error = engine
            .generate_tokens(
                vec![0],
                &SamplingParams {
                    max_tokens: usize::MAX,
                    ..Default::default()
                },
                |_| true,
            )
            .unwrap_err();
        assert!(error.to_string().contains("request context"));
        assert_eq!(engine.stage.next_position(), 1);
        assert_eq!(engine.stage.total_kv_cache_allocation_tokens(), allocated);
        for token in [2, u32::MAX] {
            let error = engine
                .generate_tokens(
                    vec![0, token],
                    &SamplingParams {
                        max_tokens: 1,
                        ..Default::default()
                    },
                    |_| panic!("invalid prompt must not reach the callback"),
                )
                .unwrap_err();
            assert!(error.to_string().contains("outside the model vocabulary"));
            assert_eq!(engine.stage.next_position(), 1);
            assert_eq!(engine.stage.total_kv_cache_allocation_tokens(), allocated);
        }
        Ok(())
    }
}
