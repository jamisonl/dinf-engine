//! Native Candle implementation of the Qwen-Image 2.1 single-stream DiT.
//!
//! The layout and arithmetic in this module follow
//! `diffusers/models/transformers/transformer_qwenimage21.py`.  In
//! particular, text and image tokens share one sequence, text is causal,
//! each image block is bidirectional, and (when enabled) text/conditioning
//! image modulation is taken from the extra `t = 0` row.  This is intentionally
//! separate from the GGUF autoregressive model traits.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "image_mpp_attention.rs"]
mod image_mpp_attention;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "rope_math.rs"]
mod rope_math;

use std::fmt;

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{Module, RmsNorm, VarBuilder, rms_norm};
use serde::Deserialize;

use super::attention::attention;
use super::linear::{Linear, linear_no_bias};

// Time and modulation projections use the native BF16 matmul contract on supported Apple
// devices. Other types, hosts, shapes and allocation sizes retain the old path.
fn conditioning_linear(linear: &Linear, input: &Tensor) -> Result<Tensor> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let Some(output) = super::text_encoder::try_bf16_linear(input, linear.weight())? {
        return Ok(output);
    }
    linear.forward(input)
}

const IMAGE_TOKENS_PER_SLOT: usize = 4;
const ROPE_THETA: f64 = 10_000.0;
const TIMESTEP_MAX_PERIOD: f64 = 10_000.0;
const TIMESTEP_FACTOR: f64 = 1_000.0;

// Torch's BF16 text projection evaluates tanh-GELU in F32 before rounding.
// Candle's BF16 unary kernel rounds intermediate products to BF16 instead.
fn text_projection_gelu(input: &Tensor) -> Result<Tensor> {
    if input.device().is_metal() && input.dtype() == DType::BF16 {
        return candle_nn::Activation::GeluPytorchTanh
            .forward(&input.to_dtype(DType::F32)?)?
            .to_dtype(DType::BF16);
    }
    candle_nn::Activation::GeluPytorchTanh.forward(input)
}

/// Configuration serialized by the Qwen-Image 2.1 transformer checkpoint.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct TransformerConfig {
    /// Spatial patch edge used by the transformer input and output heads.
    pub patch_size: usize,
    /// Number of channels in each input latent patch.
    pub in_channels: usize,
    /// Optional output channel count; defaults to `in_channels`.
    pub out_channels: Option<usize>,
    /// Number of transformer blocks.
    pub num_layers: usize,
    /// Per-head attention width.
    pub attention_head_dim: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Width of the text encoder hidden state.
    pub context_in_dim: usize,
    /// Expansion ratio used by the image-token SwiGLU.
    pub mlp_ratio: usize,
    /// Rotary position dimensions for frame, height, and width axes.
    #[serde(alias = "axes_dim")]
    pub axes_dims_rope: [usize; 3],
    /// Epsilon used by RMS and layer normalization.
    pub eps: f64,
    /// Whether to append the zero-timestep conditioning row.
    pub causal_condition: bool,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            patch_size: 1,
            in_channels: 64,
            out_channels: Some(64),
            num_layers: 32,
            attention_head_dim: 128,
            num_attention_heads: 32,
            context_in_dim: 4096,
            mlp_ratio: 3,
            axes_dims_rope: [16, 56, 56],
            eps: 1e-6,
            causal_condition: true,
        }
    }
}

impl TransformerConfig {
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.patch_size > 0,
            "Qwen Image patch_size must be non-zero"
        );
        anyhow::ensure!(
            self.in_channels > 0,
            "Qwen Image in_channels must be non-zero"
        );
        anyhow::ensure!(
            self.num_layers > 0,
            "Qwen Image num_layers must be non-zero"
        );
        anyhow::ensure!(
            self.attention_head_dim > 0,
            "Qwen Image attention_head_dim must be non-zero"
        );
        anyhow::ensure!(
            self.num_attention_heads > 0,
            "Qwen Image num_attention_heads must be non-zero"
        );
        anyhow::ensure!(
            self.context_in_dim > 0,
            "Qwen Image context_in_dim must be non-zero"
        );
        anyhow::ensure!(self.mlp_ratio > 0, "Qwen Image mlp_ratio must be non-zero");
        let patch_area = self
            .patch_size
            .checked_mul(self.patch_size)
            .ok_or_else(|| anyhow::anyhow!("Qwen Image patch area overflows usize"))?;
        let input_dim = self
            .in_channels
            .checked_mul(patch_area)
            .ok_or_else(|| anyhow::anyhow!("Qwen Image input dimension overflows usize"))?;
        let output_dim = self
            .out_channels()
            .checked_mul(patch_area)
            .ok_or_else(|| anyhow::anyhow!("Qwen Image output dimension overflows usize"))?;
        anyhow::ensure!(input_dim > 0, "Qwen Image input dimension must be non-zero");
        anyhow::ensure!(
            output_dim > 0,
            "Qwen Image output dimension must be non-zero"
        );
        anyhow::ensure!(
            self.eps.is_finite() && (self.eps as f32).is_finite() && (self.eps as f32) > 0.0,
            "Qwen Image eps must be positive and finite in F32"
        );
        let head_dim = self
            .num_attention_heads
            .checked_mul(self.attention_head_dim)
            .ok_or_else(|| anyhow::anyhow!("Qwen Image inner dimension overflows usize"))?;
        let rope_dim = self.axes_dims_rope.iter().try_fold(0_usize, |sum, dim| {
            sum.checked_add(*dim)
                .ok_or_else(|| anyhow::anyhow!("Qwen Image RoPE dimension overflows usize"))
        })?;
        let mlp_dim = head_dim
            .checked_mul(self.mlp_ratio)
            .ok_or_else(|| anyhow::anyhow!("Qwen Image MLP dimension overflows usize"))?;
        anyhow::ensure!(
            rope_dim == self.attention_head_dim,
            "Qwen Image RoPE dimensions ({rope_dim}) must equal attention head dimension ({})",
            self.attention_head_dim
        );
        for (axis, dim) in self.axes_dims_rope.iter().enumerate() {
            anyhow::ensure!(
                *dim > 0 && dim % 2 == 0,
                "Qwen Image RoPE axis {axis} must be a positive even dimension"
            );
        }
        anyhow::ensure!(head_dim > 0, "Qwen Image inner dimension must be non-zero");
        anyhow::ensure!(mlp_dim > 0, "Qwen Image MLP dimension must be non-zero");
        Ok(())
    }

    fn out_channels(&self) -> usize {
        self.out_channels.unwrap_or(self.in_channels)
    }
}

/// One `(frame, height, width)` image block in latent-token units.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageShape {
    /// Number of frames in the image block (Qwen Image currently requires one).
    pub frame: usize,
    /// Height in latent-token units.
    pub height: usize,
    /// Width in latent-token units.
    pub width: usize,
}

impl ImageShape {
    /// Construct an image block shape.
    pub const fn new(frame: usize, height: usize, width: usize) -> Self {
        Self {
            frame,
            height,
            width,
        }
    }

    fn token_count(self) -> anyhow::Result<usize> {
        self.frame
            .checked_mul(self.height)
            .and_then(|value| value.checked_mul(self.width))
            .ok_or_else(|| anyhow::anyhow!("Qwen Image shape token count overflow"))
    }
}

/// Which side of the diffusion denoising cache a forward pass executes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvCacheMode {
    /// Run the full joint sequence and retain the prefix K/V tensors.
    Extract,
    /// Run only target-image queries against the retained prefix K/V tensors.
    Cached,
}

/// Per-block prefix KV cache used by the causal-condition denoising loop.
#[derive(Debug, Default)]
pub struct TransformerKvCache {
    layers: Vec<LayerKvCache>,
}

#[derive(Debug, Default)]
struct LayerKvCache {
    key: Option<Tensor>,
    value: Option<Tensor>,
}

impl TransformerKvCache {
    /// Create an empty cache with one slot per transformer block.
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| LayerKvCache::default()).collect(),
        }
    }

    /// Remove all retained prefix keys and values.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.key = None;
            layer.value = None;
        }
    }

    /// Return whether every transformer block has a retained prefix.
    pub fn is_populated(&self) -> bool {
        !self.layers.is_empty()
            && self
                .layers
                .iter()
                .all(|layer| layer.key.is_some() && layer.value.is_some())
    }
}

/// The Qwen-Image 2.1 single-stream transformer.
pub struct QwenImageTransformer {
    config: TransformerConfig,
    inner_dim: usize,
    time_linear_1: Linear,
    time_linear_2: Linear,
    text_norm: ZeroCenterRmsNorm,
    text_in: Linear,
    text_out: Linear,
    image_in: Linear,
    modulation: Linear,
    blocks: Vec<TransformerBlock>,
    norm_out: AdaptiveLayerNorm,
    proj_out: Linear,
}

impl fmt::Debug for QwenImageTransformer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenImageTransformer")
            .field("config", &self.config)
            .field("inner_dim", &self.inner_dim)
            .field("blocks", &self.blocks.len())
            .finish_non_exhaustive()
    }
}

impl QwenImageTransformer {
    /// Load the transformer from a safetensors `VarBuilder`.
    ///
    /// The builder is expected to be rooted at the transformer component (or
    /// at the checkpoint root when the caller passes the corresponding
    /// `transformer` prefix).  Qwen Image 2.1 has no linear biases.
    pub fn load(vb: VarBuilder<'_>, config: TransformerConfig) -> anyhow::Result<Self> {
        config.validate()?;
        anyhow::ensure!(
            matches!(vb.dtype(), DType::BF16 | DType::F32),
            "Qwen Image transformer requires BF16 or F32 weights, got {:?}",
            vb.dtype()
        );
        let inner_dim = config
            .num_attention_heads
            .checked_mul(config.attention_head_dim)
            .ok_or_else(|| anyhow::anyhow!("Qwen Image inner dimension overflow"))?;
        let linear = |name: &str, input: usize, output: usize| {
            linear_no_bias(input, output, vb.pp(name))
                .map_err(|error| anyhow::anyhow!("loading Qwen Image {name}: {error}"))
        };

        let time_linear_1 = linear("time_text_embed.timestep_embedder.linear_1", 256, inner_dim)?;
        let time_linear_2 = linear(
            "time_text_embed.timestep_embedder.linear_2",
            inner_dim,
            inner_dim,
        )?;
        let text_norm =
            ZeroCenterRmsNorm::load(config.context_in_dim, config.eps, vb.pp("txt_in.text_norm"))?;
        let text_in = linear("txt_in.in_layer", config.context_in_dim, inner_dim)?;
        let text_out = linear("txt_in.out_layer", inner_dim, inner_dim)?;
        let image_in = linear(
            "img_in",
            config.in_channels * config.patch_size * config.patch_size,
            inner_dim,
        )?;
        let modulation = linear("modulation.1", inner_dim, 4 * inner_dim)?;
        let blocks = (0..config.num_layers)
            .map(|index| {
                TransformerBlock::load(
                    vb.pp("transformer_blocks").pp(index),
                    inner_dim,
                    config.num_attention_heads,
                    config.attention_head_dim,
                    config.mlp_ratio,
                    config.eps,
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let norm_out =
            AdaptiveLayerNorm::load(inner_dim, inner_dim, config.eps, vb.pp("norm_out"))?;
        let proj_out = linear(
            "proj_out",
            inner_dim,
            config.patch_size * config.patch_size * config.out_channels(),
        )?;
        Ok(Self {
            config,
            inner_dim,
            time_linear_1,
            time_linear_2,
            text_norm,
            text_in,
            text_out,
            image_in,
            modulation,
            blocks,
            norm_out,
            proj_out,
        })
    }

    /// Return the serialized transformer configuration.
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Return the number of transformer blocks.
    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Execute a Qwen Image 2.1 forward pass.
    ///
    /// `hidden_states` is `[B, image_tokens, in_channels]`; it contains
    /// condition-image tokens first and target-image tokens last.  `image_mask`
    /// labels the corresponding pre-expansion joint sequence: each `true`
    /// entry is one 2x2 image slot and therefore consumes four rows from
    /// `hidden_states`; `false` entries are text/placeholder positions.
    /// `image_shapes` lists condition blocks followed by the target block.
    /// All batch items must share this layout, matching the upstream model.
    ///
    /// The output has the full joint sequence for `Extract`/uncached passes,
    /// and target-image rows only for `Cached` passes, just like the reference
    /// implementation.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: &Tensor,
        image_shapes: &[ImageShape],
        image_mask: &[bool],
        encoder_hidden_states_mask: Option<&Tensor>,
        mut kv_cache: Option<&mut TransformerKvCache>,
        kv_cache_mode: Option<KvCacheMode>,
    ) -> Result<Tensor> {
        let (batch, _image_tokens, input_channels) = hidden_states.dims3()?;
        let (encoder_batch, encoder_tokens, context_dim) = encoder_hidden_states.dims3()?;
        if batch != encoder_batch
            || input_channels
                != self.config.in_channels * self.config.patch_size * self.config.patch_size
            || context_dim != self.config.context_in_dim
            || !matches!(hidden_states.dtype(), DType::BF16 | DType::F32)
            || hidden_states.dtype() != encoder_hidden_states.dtype()
            || hidden_states.dtype() != timestep.dtype()
        {
            candle_core::bail!(
                "invalid Qwen Image transformer input shapes or precision: hidden={:?}, encoder={:?}, timestep={:?}",
                hidden_states.dims(),
                encoder_hidden_states.dims(),
                timestep.dims()
            );
        }
        if image_shapes.is_empty() {
            candle_core::bail!("Qwen Image requires a target image shape");
        }
        if let Some(cache) = kv_cache.as_deref() {
            if cache.layers.len() != self.blocks.len() {
                candle_core::bail!(
                    "Qwen Image KV cache has {} layers, expected {}",
                    cache.layers.len(),
                    self.blocks.len()
                );
            }
            if !self.config.causal_condition {
                candle_core::bail!("Qwen Image KV cache requires causal_condition=true");
            }
        }
        if kv_cache_mode.is_some() && kv_cache.is_none() {
            candle_core::bail!("Qwen Image cache mode requires a KV cache");
        }
        if kv_cache_mode == Some(KvCacheMode::Extract) && !self.config.causal_condition {
            candle_core::bail!("Qwen Image cache extraction requires causal_condition=true");
        }
        let target_tokens = image_shapes
            .last()
            .copied()
            .ok_or_else(|| candle_core::Error::Msg("missing target shape".into()))?
            .token_count()
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        if target_tokens % IMAGE_TOKENS_PER_SLOT != 0 {
            candle_core::bail!("Qwen Image target token count must be divisible by 4");
        }
        if image_mask.len() != encoder_tokens + target_tokens / IMAGE_TOKENS_PER_SLOT {
            candle_core::bail!(
                "Qwen Image mask length {} does not equal encoder tokens {} + target slots {}",
                image_mask.len(),
                encoder_tokens,
                target_tokens / IMAGE_TOKENS_PER_SLOT
            );
        }

        let projected_image = self.image_in.forward(hidden_states)?;
        let projected_text = self.text_out.forward(&text_projection_gelu(
            &self
                .text_in
                .forward(&self.text_norm.forward(encoder_hidden_states)?)?,
        )?)?;
        let target_slots = target_tokens / IMAGE_TOKENS_PER_SLOT;
        let target_zero = Tensor::zeros(
            (batch, target_slots, self.inner_dim),
            projected_text.dtype(),
            projected_text.device(),
        )?;
        let pre_expanded = Tensor::cat(&[&projected_text, &target_zero], 1)?;
        let (joint_hidden_states, image_pad_mask) =
            expand_joint_sequence(&pre_expanded, &projected_image, image_mask)?;
        let (image_ids, target_token_mask) = build_token_metadata(&image_pad_mask, image_shapes)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        let (frame_pos, height_pos, width_pos) =
            build_rope_positions(&image_pad_mask, image_shapes)
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        let (rope_cos, rope_sin) = build_rope_for_dtype(
            &frame_pos,
            &height_pos,
            &width_pos,
            &self.config.axes_dims_rope,
            projected_image.dtype(),
            projected_image.device(),
        )?;
        let timestep = timestep.reshape((batch,))?;
        let mut timestep = timestep.to_dtype(hidden_states.dtype())?;
        if self.config.causal_condition {
            let zero = Tensor::zeros(1, hidden_states.dtype(), hidden_states.device())?;
            timestep = Tensor::cat(&[&timestep, &zero], 0)?;
        }
        let temb = self.time_embedding(&timestep)?;
        let modulation = conditioning_linear(
            &self.modulation,
            &candle_nn::Activation::Silu.forward(&temb)?,
        )?;
        let (prefix_len, target_token_mask) = target_prefix(&target_token_mask)?;
        let key_valid = encoder_key_validity(
            encoder_hidden_states_mask,
            image_mask,
            &image_pad_mask,
            batch,
            encoder_tokens,
        )?;

        let cached = kv_cache_mode == Some(KvCacheMode::Cached);
        let (work_hidden, work_rope_cos, work_rope_sin, work_target_mask) = if cached {
            if !kv_cache
                .as_deref()
                .is_some_and(TransformerKvCache::is_populated)
            {
                candle_core::bail!("Qwen Image cached forward requires populated KV cache");
            }
            (
                joint_hidden_states.narrow(
                    1,
                    prefix_len,
                    joint_hidden_states.dim(1)? - prefix_len,
                )?,
                rope_cos.narrow(0, prefix_len, rope_cos.dim(0)? - prefix_len)?,
                rope_sin.narrow(0, prefix_len, rope_sin.dim(0)? - prefix_len)?,
                target_token_mask[prefix_len..].to_vec(),
            )
        } else {
            (
                joint_hidden_states,
                rope_cos,
                rope_sin,
                target_token_mask.clone(),
            )
        };
        let mut block_segments = Vec::new();
        if !cached {
            block_segments = prefix_segments(&image_ids, prefix_len);
        }
        let mut hidden = work_hidden;
        // Modulation is shared by every block at this timestep. Build the
        // selected rows and nonlinear gates once, not 32 times per step.
        let modulation =
            BlockModulation::new(&modulation, &work_target_mask, batch, hidden.dim(1)?)?;
        let query_start_offset = if cached { prefix_len } else { 0 };
        for (index, block) in self.blocks.iter().enumerate() {
            let layer_cache = kv_cache
                .as_deref_mut()
                .map(|cache| &mut cache.layers[index]);
            hidden = block.forward(
                &hidden,
                &modulation,
                &work_rope_cos,
                &work_rope_sin,
                if cached { None } else { Some(&block_segments) },
                key_valid.as_deref(),
                layer_cache,
                kv_cache_mode,
                prefix_len,
                query_start_offset,
            )?;
        }
        let output = self.norm_out.forward(&hidden, &temb, &work_target_mask)?;
        self.proj_out.forward(&output)
    }

    fn time_embedding(&self, timestep: &Tensor) -> Result<Tensor> {
        let projection = timestep_embedding(timestep, 256, TIMESTEP_FACTOR, TIMESTEP_MAX_PERIOD)?;
        let projection = projection.to_dtype(self.time_linear_1.weight().dtype())?;
        let projection = conditioning_linear(&self.time_linear_1, &projection)?;
        let projection = candle_nn::Activation::Silu.forward(&projection)?;
        conditioning_linear(&self.time_linear_2, &projection)
    }
}

struct TransformerBlock {
    attn: AttentionBlock,
    norm1: NoAffineLayerNorm,
    norm2: NoAffineLayerNorm,
    mlp: SwiGlu,
}

struct BlockModulation {
    scale1: Tensor,
    gate1: Tensor,
    scale2: Tensor,
    gate2: Tensor,
}

impl BlockModulation {
    fn new(params: &Tensor, target_mask: &[bool], batch: usize, sequence: usize) -> Result<Self> {
        let chunks = params.chunk(4, D::Minus1)?;
        let row = |index| select_modulation_rows(&chunks[index], target_mask, batch, sequence);
        Ok(Self {
            scale1: (row(0)? + 1.0)?,
            gate1: row(1)?.tanh()?,
            scale2: (row(2)? + 1.0)?,
            gate2: row(3)?.tanh()?,
        })
    }
}

fn gated_residual(hidden: &Tensor, gate: &Tensor, update: &Tensor) -> Result<Tensor> {
    #[cfg(target_os = "macos")]
    if hidden.device().is_metal() && super::fused_metal::enabled() {
        return super::fused_metal::gated_residual(hidden, gate, update);
    }
    hidden + gate.broadcast_mul(update)?
}

impl TransformerBlock {
    fn load(
        vb: VarBuilder<'_>,
        dim: usize,
        heads: usize,
        head_dim: usize,
        mlp_ratio: usize,
        eps: f64,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            attn: AttentionBlock::load(vb.pp("attn"), dim, heads, head_dim, eps)?,
            norm1: NoAffineLayerNorm::new(dim, eps),
            norm2: NoAffineLayerNorm::new(dim, eps),
            mlp: SwiGlu::load(vb.pp("img_mlp"), dim, dim * mlp_ratio)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden: &Tensor,
        modulation: &BlockModulation,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        segments: Option<&[(usize, usize, bool)]>,
        key_valid: Option<&[Vec<bool>]>,
        layer_cache: Option<&mut LayerKvCache>,
        cache_mode: Option<KvCacheMode>,
        prefix_len: usize,
        query_start_offset: usize,
    ) -> Result<Tensor> {
        let modulated = self.norm1.scaled(hidden, &modulation.scale1)?;
        let attn = self.attn.forward(
            &modulated,
            rope_cos,
            rope_sin,
            segments,
            key_valid,
            layer_cache,
            cache_mode,
            prefix_len,
            query_start_offset,
        )?;
        let hidden = gated_residual(hidden, &modulation.gate1, &attn)?;
        let modulated = self.norm2.scaled(&hidden, &modulation.scale2)?;
        let mlp = self.mlp.forward(&modulated)?;
        let hidden = gated_residual(&hidden, &modulation.gate2, &mlp)?;
        if hidden.dtype() == DType::F32 || hidden.dtype() == DType::BF16 {
            Ok(hidden)
        } else {
            candle_core::bail!(
                "Qwen Image transformer does not support {:?}",
                hidden.dtype()
            )
        }
    }
}

struct AttentionBlock {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    heads: usize,
    head_dim: usize,
}

fn qk_rms_norm(norm: &RmsNorm, input: &Tensor) -> Result<Tensor> {
    // Both modules use the same F32 inverse and BF16 weight/cast contract.
    super::text_encoder::text_rms_norm(norm, input)
}

impl AttentionBlock {
    fn load(
        vb: VarBuilder<'_>,
        dim: usize,
        heads: usize,
        head_dim: usize,
        eps: f64,
    ) -> anyhow::Result<Self> {
        let inner = heads
            .checked_mul(head_dim)
            .ok_or_else(|| anyhow::anyhow!("Qwen Image attention dimension overflow"))?;
        Ok(Self {
            to_q: linear_no_bias(dim, inner, vb.pp("to_q"))?,
            to_k: linear_no_bias(dim, inner, vb.pp("to_k"))?,
            to_v: linear_no_bias(dim, inner, vb.pp("to_v"))?,
            to_out: linear_no_bias(inner, dim, vb.pp("to_out.0"))?,
            norm_q: rms_norm(head_dim, eps, vb.pp("norm_q"))?,
            norm_k: rms_norm(head_dim, eps, vb.pp("norm_k"))?,
            heads,
            head_dim,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        input: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        segments: Option<&[(usize, usize, bool)]>,
        key_valid: Option<&[Vec<bool>]>,
        layer_cache: Option<&mut LayerKvCache>,
        cache_mode: Option<KvCacheMode>,
        prefix_len: usize,
        query_start_offset: usize,
    ) -> Result<Tensor> {
        let (batch, sequence, _) = input.dims3()?;
        let query = self
            .to_q
            .forward(input)?
            .reshape((batch, sequence, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let key = self
            .to_k
            .forward(input)?
            .reshape((batch, sequence, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let value = self
            .to_v
            .forward(input)?
            .reshape((batch, sequence, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let query = qk_rms_norm(&self.norm_q, &query)?;
        let key = qk_rms_norm(&self.norm_k, &key)?;
        let query = apply_rope(&query, rope_cos, rope_sin)?;
        let key = apply_rope(&key, rope_cos, rope_sin)?;
        let (query, key, value, key_valid, query_start) = match cache_mode {
            Some(KvCacheMode::Cached) => {
                let cache = layer_cache.as_deref().ok_or_else(|| {
                    candle_core::Error::Msg("missing Qwen Image layer cache".into())
                })?;
                let cached_key = cache.key.as_ref().ok_or_else(|| {
                    candle_core::Error::Msg("missing Qwen Image cached key".into())
                })?;
                let cached_value = cache.value.as_ref().ok_or_else(|| {
                    candle_core::Error::Msg("missing Qwen Image cached value".into())
                })?;
                if cached_key.dim(2)? != prefix_len || cached_value.dim(2)? != prefix_len {
                    candle_core::bail!(
                        "Qwen Image cached prefix length does not match the current layout"
                    );
                }
                let key = Tensor::cat(&[cached_key, &key], 2)?;
                let value = Tensor::cat(&[cached_value, &value], 2)?;
                let key_valid =
                    key_valid.map(|valid| valid.iter().map(|row| row.to_vec()).collect::<Vec<_>>());
                (query, key, value, key_valid, prefix_len)
            }
            Some(KvCacheMode::Extract) => {
                let cache = layer_cache.ok_or_else(|| {
                    candle_core::Error::Msg("missing Qwen Image layer cache".into())
                })?;
                cache.key = Some(key.narrow(2, 0, prefix_len)?.copy()?);
                cache.value = Some(value.narrow(2, 0, prefix_len)?.copy()?);
                (
                    query,
                    key,
                    value,
                    key_valid.map(|valid| valid.to_vec()),
                    query_start_offset,
                )
            }
            None => (
                query,
                key,
                value,
                key_valid.map(|valid| valid.to_vec()),
                query_start_offset,
            ),
        };

        let attended = if let Some(segments) = segments {
            let mut outputs = Vec::with_capacity(segments.len() + 1);
            for &(start, end, is_text) in segments {
                let q = query.narrow(2, start, end - start)?;
                let k = key.narrow(2, 0, end)?;
                let v = value.narrow(2, 0, end)?;
                outputs.push(attend_segment(
                    &q,
                    &k,
                    &v,
                    is_text,
                    start,
                    key_valid.as_deref(),
                )?);
            }
            let target = query.narrow(2, prefix_len, query.dim(2)? - prefix_len)?;
            outputs.push(attend_segment(
                &target,
                &key,
                &value,
                false,
                prefix_len,
                key_valid.as_deref(),
            )?);
            Tensor::cat(&outputs, 2)?
        } else {
            attend_segment(
                &query,
                &key,
                &value,
                false,
                query_start,
                key_valid.as_deref(),
            )?
        };
        let attended = attended.transpose(1, 2)?.contiguous()?.reshape((
            batch,
            attended.dim(2)?,
            self.heads * self.head_dim,
        ))?;
        self.to_out.forward(&attended)
    }
}

struct SwiGlu {
    gate: Linear,
    proj: Linear,
    out: Linear,
}

impl SwiGlu {
    fn load(vb: VarBuilder<'_>, input: usize, hidden: usize) -> anyhow::Result<Self> {
        Ok(Self {
            proj: linear_no_bias(input, hidden, vb.pp("proj"))?,
            out: linear_no_bias(hidden, input, vb.pp("out"))?,
            gate: linear_no_bias(input, hidden, vb.pp("gate_layer"))?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let gate = denoiser_gate_silu(&self.gate.forward(input)?)?;
        let value = self.proj.forward(input)?;
        self.out.forward(&(gate * value)?)
    }
}

fn denoiser_gate_silu(input: &Tensor) -> Result<Tensor> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let Some(output) = super::text_encoder::try_bf16_gate_silu(input)? {
        return Ok(output);
    }
    candle_nn::Activation::Silu.forward(input)
}

struct ZeroCenterRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl ZeroCenterRmsNorm {
    fn load(dim: usize, eps: f64, vb: VarBuilder<'_>) -> anyhow::Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let dtype = input.dtype();
        let input_f32 = input.to_dtype(DType::F32)?;
        let dim = input.dim(D::Minus1)? as f64;
        let mean_square = (input_f32.sqr()?.sum_keepdim(D::Minus1)? / dim)?;
        let normalized = input_f32.broadcast_div(&(&mean_square + self.eps)?.sqrt()?)?;
        let one = Tensor::ones(self.weight.dims(), DType::F32, input.device())?;
        let scale = self.weight.to_dtype(DType::F32)?.broadcast_add(&one)?;
        normalized.broadcast_mul(&scale)?.to_dtype(dtype)
    }
}

struct NoAffineLayerNorm {
    eps: f64,
}

impl NoAffineLayerNorm {
    fn new(_dim: usize, eps: f64) -> Self {
        Self { eps }
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let dtype = input.dtype();
        let input_f32 = input.to_dtype(DType::F32)?;
        let dim = input.dim(D::Minus1)? as f64;
        let mean = (input_f32.sum_keepdim(D::Minus1)? / dim)?;
        let centered = input_f32.broadcast_sub(&mean)?;
        let variance = (centered.sqr()?.sum_keepdim(D::Minus1)? / dim)?;
        centered
            .broadcast_div(&(&variance + self.eps)?.sqrt()?)?
            .to_dtype(dtype)
    }

    fn scaled(&self, input: &Tensor, scale: &Tensor) -> Result<Tensor> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if let Some(output) =
            super::fused_metal::try_native_norm_scale(input, scale, self.eps as f32)?
        {
            return Ok(output);
        }
        #[cfg(target_os = "macos")]
        if input.device().is_metal() && super::fused_metal::enabled() {
            return super::fused_metal::norm_scale(input, scale, self.eps as f32);
        }
        self.forward(input)?.broadcast_mul(scale)
    }
}

struct AdaptiveLayerNorm {
    linear: Linear,
    eps: f64,
}

impl AdaptiveLayerNorm {
    fn load(
        dim: usize,
        conditioning_dim: usize,
        eps: f64,
        vb: VarBuilder<'_>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            linear: linear_no_bias(conditioning_dim, dim, vb.pp("linear"))?,
            eps,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        conditioning: &Tensor,
        target_mask: &[bool],
    ) -> Result<Tensor> {
        let scale = conditioning_linear(
            &self.linear,
            &candle_nn::Activation::Silu.forward(conditioning)?,
        )?;
        let scale = select_modulation_rows(&scale, target_mask, input.dim(0)?, input.dim(1)?)?;
        NoAffineLayerNorm::new(input.dim(D::Minus1)?, self.eps).scaled(input, &(&scale + 1.0)?)
    }
}

fn timestep_embedding(
    timestep: &Tensor,
    dim: usize,
    factor: f64,
    max_period: f64,
) -> Result<Tensor> {
    if !dim.is_multiple_of(2) {
        candle_core::bail!("Qwen Image timestep embedding dimension must be even");
    }
    let half = dim / 2;
    let frequencies = if timestep.device().is_metal() && timestep.dtype() == DType::BF16 {
        // The BF16 checkpoint builds this table with separate F32 operations.
        // Preserve that order before the projection is rounded to BF16.
        let coefficient = (-max_period.ln()) as f32;
        (0..half)
            .map(|index| ((coefficient * index as f32) / half as f32).exp())
            .collect::<Vec<_>>()
    } else {
        (0..half)
            .map(|index| (-max_period.ln() * index as f64 / half as f64).exp() as f32)
            .collect::<Vec<_>>()
    };
    let frequencies = Tensor::from_vec(frequencies, half, timestep.device())?;
    // The upstream implementation scales the timestep before multiplying by
    // the frequency table.  Keep that order explicit: it matters for BF16
    // callers because the promotion to F32 happens at this boundary.
    let timestep = (timestep.to_dtype(DType::F32)? * factor)?;
    let args = timestep
        .unsqueeze(1)?
        .broadcast_mul(&frequencies.unsqueeze(0)?)?;
    Tensor::cat(&[&args.cos()?, &args.sin()?], D::Minus1)
}

fn select_modulation_rows(
    params: &Tensor,
    target_mask: &[bool],
    batch: usize,
    sequence: usize,
) -> Result<Tensor> {
    let dim = params.dim(D::Minus1)?;
    if params.dim(0)? == batch {
        return params.unsqueeze(1)?.broadcast_as((batch, sequence, dim));
    }
    if params.dim(0)? != batch + 1 || target_mask.len() != sequence {
        candle_core::bail!("invalid Qwen Image modulation rows");
    }
    let real = params.narrow(0, 0, batch)?.unsqueeze(1)?;
    let zero = params.narrow(0, batch, 1)?.unsqueeze(1)?;
    let mask = target_mask
        .iter()
        .map(|target| if *target { 1_f32 } else { 0_f32 })
        .collect::<Vec<_>>();
    let mask =
        Tensor::from_vec(mask, (1, sequence, 1), params.device())?.to_dtype(params.dtype())?;
    let real = real.broadcast_as((batch, sequence, dim))?;
    let zero = zero.broadcast_as((batch, sequence, dim))?;
    let inverse_mask = ((&mask * -1.0)? + 1.0)?;
    real.broadcast_mul(&mask)?
        .broadcast_add(&zero.broadcast_mul(&inverse_mask)?)
}

fn expand_joint_sequence(
    pre_expanded: &Tensor,
    image_tokens: &Tensor,
    image_mask: &[bool],
) -> Result<(Tensor, Vec<bool>)> {
    let (_, sequence, _) = pre_expanded.dims3()?;
    if sequence != image_mask.len() {
        candle_core::bail!("Qwen Image pre-expansion sequence and image mask differ");
    }
    let image_count = image_mask.iter().filter(|value| **value).count();
    let image_token_count = image_count
        .checked_mul(IMAGE_TOKENS_PER_SLOT)
        .ok_or_else(|| candle_core::Error::Msg("Qwen Image token count overflows usize".into()))?;
    if image_tokens.dim(1)? != image_token_count {
        candle_core::bail!(
            "Qwen Image hidden state count {} does not match {} image slots",
            image_tokens.dim(1)?,
            image_count
        );
    }
    let mut chunks = Vec::with_capacity(sequence);
    let mut image_cursor = 0;
    let expanded_capacity = sequence.checked_add(image_token_count).ok_or_else(|| {
        candle_core::Error::Msg("Qwen Image sequence length overflows usize".into())
    })?;
    let mut expanded_mask = Vec::with_capacity(expanded_capacity);
    let mut index = 0;
    while index < sequence {
        let is_image = image_mask[index];
        let mut end = index + 1;
        while end < sequence && image_mask[end] == is_image {
            end += 1;
        }
        if is_image {
            let count = (end - index) * IMAGE_TOKENS_PER_SLOT;
            chunks.push(image_tokens.narrow(1, image_cursor, count)?);
            expanded_mask.extend(std::iter::repeat_n(true, count));
            image_cursor += count;
        } else {
            chunks.push(pre_expanded.narrow(1, index, end - index)?);
            expanded_mask.extend(std::iter::repeat_n(false, end - index));
        }
        index = end;
    }
    Ok((Tensor::cat(&chunks, 1)?, expanded_mask))
}

fn build_token_metadata(
    image_mask: &[bool],
    shapes: &[ImageShape],
) -> anyhow::Result<(Vec<i64>, Vec<bool>)> {
    anyhow::ensure!(
        !shapes.is_empty(),
        "Qwen Image requires at least one image block"
    );
    let image_positions = image_mask
        .iter()
        .enumerate()
        .filter_map(|(index, is_image)| is_image.then_some(index))
        .collect::<Vec<_>>();
    let expected = shapes.iter().try_fold(0_usize, |sum, shape| {
        let length = shape.token_count()?;
        sum.checked_add(length)
            .ok_or_else(|| anyhow::anyhow!("Qwen Image shape token count overflows usize"))
    })?;
    anyhow::ensure!(
        expected == image_positions.len(),
        "Qwen Image shapes account for {expected} image tokens but mask marks {}",
        image_positions.len()
    );
    let mut image_ids = vec![-1_i64; image_mask.len()];
    let mut cursor = 0_usize;
    for (block, shape) in shapes.iter().enumerate() {
        let length = shape.token_count()?;
        let end = cursor
            .checked_add(length)
            .ok_or_else(|| anyhow::anyhow!("Qwen Image image position index overflows usize"))?;
        anyhow::ensure!(
            end <= image_positions.len(),
            "Qwen Image shape has more image tokens than its mask"
        );
        for position in &image_positions[cursor..end] {
            image_ids[*position] = block as i64;
        }
        cursor = end;
    }
    let target_len = shapes.last().unwrap().token_count()?;
    let mut target = vec![false; image_mask.len()];
    for position in &image_positions[image_positions.len() - target_len..] {
        target[*position] = true;
    }
    Ok((image_ids, target))
}

fn target_prefix(mask: &[bool]) -> Result<(usize, Vec<bool>)> {
    let first = mask
        .iter()
        .position(|value| *value)
        .ok_or_else(|| candle_core::Error::Msg("Qwen Image target mask is empty".into()))?;
    if !mask[first..].iter().all(|value| *value) {
        candle_core::bail!("Qwen Image target image must be the final contiguous block");
    }
    Ok((first, mask.to_vec()))
}

fn prefix_segments(image_ids: &[i64], prefix_len: usize) -> Vec<(usize, usize, bool)> {
    let mut segments = Vec::new();
    let mut start = 0;
    while start < prefix_len {
        let id = image_ids[start];
        let mut end = start + 1;
        while end < prefix_len && image_ids[end] == id {
            end += 1;
        }
        segments.push((start, end, id < 0));
        start = end;
    }
    segments
}

fn build_rope_positions(
    image_mask: &[bool],
    shapes: &[ImageShape],
) -> anyhow::Result<(Vec<i64>, Vec<i64>, Vec<i64>)> {
    let image_positions = image_mask
        .iter()
        .enumerate()
        .filter_map(|(index, is_image)| is_image.then_some(index))
        .collect::<Vec<_>>();
    let mut frame = Vec::with_capacity(image_mask.len());
    let mut position = 0_i64;
    let mut cursor = 0_usize;
    let mut image_cursor = 0_usize;
    let mut height = vec![0_i64; image_mask.len()];
    let mut width = vec![0_i64; image_mask.len()];
    for shape in shapes {
        anyhow::ensure!(
            shape.frame == 1,
            "Qwen Image 2.1 transformer currently requires frame=1"
        );
        let block_len = shape.token_count()?;
        let block_end = image_cursor
            .checked_add(block_len)
            .ok_or_else(|| anyhow::anyhow!("Qwen Image image position index overflows usize"))?;
        anyhow::ensure!(
            block_end <= image_positions.len(),
            "Qwen Image shape has more image tokens than its mask"
        );
        let block_start = image_mask[cursor..]
            .iter()
            .position(|value| *value)
            .map(|offset| cursor + offset)
            .ok_or_else(|| anyhow::anyhow!("Qwen Image shape has no corresponding image block"))?;
        frame.extend((0..block_start - cursor).map(|offset| position + offset as i64));
        position += (block_start - cursor) as i64;
        let block_positions = &image_positions[image_cursor..block_end];
        frame.extend(std::iter::repeat_n(position, block_len));
        let image_height = (-(shape.height as i64 - (shape.height / 2) as i64)
            ..(shape.height / 2) as i64)
            .flat_map(|value| std::iter::repeat_n(value, shape.width))
            .collect::<Vec<_>>();
        let width_values = (-(shape.width as i64 - (shape.width / 2) as i64)
            ..(shape.width / 2) as i64)
            .collect::<Vec<_>>();
        let image_width = (0..shape.height)
            .flat_map(|_| width_values.iter().copied())
            .collect::<Vec<_>>();
        anyhow::ensure!(
            image_height.len() == block_len && image_width.len() == block_len,
            "Qwen Image RoPE image shape does not match its frame/height/width token count"
        );
        for (index, token) in block_positions.iter().enumerate() {
            height[*token] = image_height[index];
            width[*token] = image_width[index];
        }
        image_cursor += block_len;
        cursor = block_start + block_len;
        position += shape.height.max(shape.width) as i64;
    }
    if cursor < image_mask.len() {
        frame.extend((0..image_mask.len() - cursor).map(|offset| position + offset as i64));
    }
    anyhow::ensure!(
        frame.len() == image_mask.len(),
        "Qwen Image RoPE frame positions length {} != sequence length {}",
        frame.len(),
        image_mask.len()
    );
    // Text positions advance along all three axes in the reference
    // implementation.  Image tokens overwrite their height/width axes above.
    for (index, is_image) in image_mask.iter().copied().enumerate() {
        if !is_image {
            height[index] = frame[index];
            width[index] = frame[index];
        }
    }
    Ok((frame, height, width))
}

// BF16 Metal follows the native model's CPU-F32 frequency/table contract.
// This is a precision choice, independent of the optional fusion speed flag.
fn build_rope_for_dtype(
    frame: &[i64],
    height: &[i64],
    width: &[i64],
    axes: &[usize; 3],
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if dtype == DType::BF16 && device.is_metal() {
        return build_rope_native_bf16(frame, height, width, axes, device);
    }
    let _ = dtype;
    build_rope(frame, height, width, axes, device)
}

// Each F32 table is at most 32 MiB (enough for the supported 4096-square
// image plus 4096 text tokens at head width 128). Check dimensions and the native table's
// position range before allocating either host table or a frequency vector.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn native_rope_table_size(sequence: usize, axes: &[usize; 3]) -> Result<usize> {
    if sequence == 0 || axes.iter().any(|dim| *dim == 0 || dim % 2 != 0) {
        candle_core::bail!("invalid native BF16 RoPE dimensions");
    }
    let half_dim = axes.iter().try_fold(0_usize, |sum, dim| {
        sum.checked_add(dim / 2)
            .ok_or_else(|| candle_core::Error::Msg("native BF16 RoPE width overflows".into()))
    })?;
    let count = sequence
        .checked_mul(half_dim)
        .filter(|count| *count <= 8 * 1024 * 1024)
        .ok_or_else(|| candle_core::Error::Msg("native BF16 RoPE table exceeds 32 MiB".into()))?;
    Ok(count)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn build_rope_native_bf16(
    frame: &[i64],
    height: &[i64],
    width: &[i64],
    axes: &[usize; 3],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    if frame.len() != height.len() || frame.len() != width.len() {
        candle_core::bail!("Qwen Image RoPE position axes differ in length");
    }
    let count = native_rope_table_size(frame.len(), axes)?;
    if [frame, height, width]
        .iter()
        .flat_map(|positions| positions.iter())
        .any(|position| !(-1024..8192).contains(position))
    {
        candle_core::bail!("native BF16 RoPE position is outside [-1024, 8191]");
    }
    let half_dim = count / frame.len();
    let mut cos = vec![0.0_f32; count];
    let mut sin = vec![0.0_f32; count];
    let mut axis_offset = 0;
    for (positions, &dim) in [frame, height, width].iter().zip(axes) {
        let frequencies = rope_math::frequencies(ROPE_THETA as f32, dim)
            .ok_or_else(|| candle_core::Error::Msg("native BF16 RoPE frequency domain".into()))?;
        for (row, position) in positions.iter().enumerate() {
            for (column, frequency) in frequencies.iter().enumerate() {
                let angle = *position as f32 * frequency;
                let index = row * half_dim + axis_offset + column;
                // The pinned CPU polar path uses Apple's paired sin/cos API.
                let (sine, cosine) = rope_math::sin_cos(angle);
                cos[index] = cosine;
                sin[index] = sine;
            }
        }
        axis_offset += dim / 2;
    }
    Ok((
        Tensor::from_vec(cos, (frame.len(), half_dim), device)?,
        Tensor::from_vec(sin, (frame.len(), half_dim), device)?,
    ))
}

fn build_rope(
    frame: &[i64],
    height: &[i64],
    width: &[i64],
    axes: &[usize; 3],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    if frame.len() != height.len() || frame.len() != width.len() {
        candle_core::bail!("Qwen Image RoPE position axes differ in length");
    }
    let mut cos_axes = Vec::with_capacity(3);
    let mut sin_axes = Vec::with_capacity(3);
    for (positions, &dim) in [frame, height, width].iter().zip(axes) {
        let pos = Tensor::new(positions.to_vec(), device)?
            .to_dtype(DType::F32)?
            .unsqueeze(1)?;
        let frequencies = (0..dim / 2)
            .map(|index| (1.0 / ROPE_THETA.powf((2 * index) as f64 / dim as f64)) as f32)
            .collect::<Vec<_>>();
        let frequency = Tensor::from_vec(frequencies, dim / 2, device)?.unsqueeze(0)?;
        let angles = pos.broadcast_mul(&frequency)?;
        cos_axes.push(angles.cos()?);
        sin_axes.push(angles.sin()?);
    }
    Ok((
        Tensor::cat(&cos_axes.iter().collect::<Vec<_>>(), 1)?,
        Tensor::cat(&sin_axes.iter().collect::<Vec<_>>(), 1)?,
    ))
}

fn apply_rope(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if input.dtype() == DType::BF16 && input.device().is_metal() {
        // Always pair native CPU-F32 tables with the native complex-multiply
        // rounding order, even when optional diffusion fusion is disabled.
        return super::fused_metal::rope_native_bf16(input, cos, sin);
    }
    #[cfg(target_os = "macos")]
    if input.device().is_metal() && super::fused_metal::enabled() {
        return super::fused_metal::rope(input, cos, sin);
    }
    apply_rope_reference(input, cos, sin)
}

pub(super) fn apply_rope_reference(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (batch, heads, sequence, dim) = input.dims4()?;
    if cos.dims() != [sequence, dim / 2] || sin.dims() != [sequence, dim / 2] {
        candle_core::bail!("Qwen Image RoPE tensor has incompatible shape");
    }
    let dtype = input.dtype();
    let input = input
        .to_dtype(DType::F32)?
        .reshape((batch, heads, sequence, dim / 2, 2))?;
    let real = input.narrow(4, 0, 1)?.squeeze(4)?;
    let imag = input.narrow(4, 1, 1)?.squeeze(4)?;
    let cos = cos.reshape((1, 1, sequence, dim / 2))?;
    let sin = sin.reshape((1, 1, sequence, dim / 2))?;
    let out_real = real
        .broadcast_mul(&cos)?
        .broadcast_sub(&imag.broadcast_mul(&sin)?)?;
    let out_imag = real
        .broadcast_mul(&sin)?
        .broadcast_add(&imag.broadcast_mul(&cos)?)?;
    Tensor::cat(&[&out_real.unsqueeze(4)?, &out_imag.unsqueeze(4)?], 4)?
        .reshape((batch, heads, sequence, dim))?
        .to_dtype(dtype)
}

fn encoder_key_validity(
    encoder_mask: Option<&Tensor>,
    image_mask: &[bool],
    expanded_mask: &[bool],
    batch: usize,
    encoder_tokens: usize,
) -> Result<Option<Vec<Vec<bool>>>> {
    let Some(mask) = encoder_mask else {
        return Ok(None);
    };
    let dims = mask.dims();
    if dims != [batch, encoder_tokens] {
        candle_core::bail!("Qwen Image encoder mask must be [batch, encoder_tokens]");
    }
    let values: Vec<Vec<bool>> = match mask.dtype() {
        DType::U8 => mask
            .to_vec2::<u8>()?
            .into_iter()
            .map(|row| row.into_iter().map(|v| v != 0).collect())
            .collect(),
        DType::U32 => mask
            .to_vec2::<u32>()?
            .into_iter()
            .map(|row| row.into_iter().map(|v| v != 0).collect())
            .collect(),
        DType::I64 => mask
            .to_vec2::<i64>()?
            .into_iter()
            .map(|row| row.into_iter().map(|v| v != 0).collect())
            .collect(),
        DType::F32 => mask
            .to_vec2::<f32>()?
            .into_iter()
            .map(|row| row.into_iter().map(|v| v != 0.0).collect())
            .collect(),
        DType::BF16 => mask
            .to_vec2::<half::bf16>()?
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|v| v != half::bf16::from_f32(0.0))
                    .collect()
            })
            .collect(),
        dtype => candle_core::bail!("unsupported Qwen Image encoder mask dtype {dtype:?}"),
    };
    if encoder_tokens > 0
        && values
            .iter()
            .any(|row| !row.iter().any(|is_valid| *is_valid))
    {
        candle_core::bail!("Qwen Image encoder mask cannot wholly mask conditioning tokens");
    }
    let mut valid = vec![vec![true; expanded_mask.len()]; batch];
    // Fill text positions by row; image and target slots remain valid keys.
    for (batch_index, row) in valid.iter_mut().enumerate() {
        let mut cursor = 0;
        for (index, &is_image) in image_mask.iter().enumerate() {
            let repeats = if is_image { IMAGE_TOKENS_PER_SLOT } else { 1 };
            if index < encoder_tokens && !is_image {
                row[cursor] = values[batch_index][index];
            }
            cursor += repeats;
        }
    }
    Ok(Some(valid))
}

fn attend_segment(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    query_start: usize,
    key_valid: Option<&[Vec<bool>]>,
) -> Result<Tensor> {
    if key_valid.is_none() {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if let Some(output) = image_mpp_attention::try_attention(q, k, v, causal, query_start)? {
            return Ok(output);
        }
        return attention(q, k, v, causal, query_start);
    }
    masked_attention(q, k, v, causal, query_start, key_valid.unwrap())
}

fn masked_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    query_start: usize,
    key_valid: &[Vec<bool>],
) -> Result<Tensor> {
    let dtype = q.dtype();
    let (batch, heads, queries, dim) = q.dims4()?;
    let keys = k.dim(2)?;
    if key_valid.len() != batch || key_valid.iter().any(|row| row.len() < keys) {
        candle_core::bail!("invalid Qwen Image key mask");
    }
    let q = q
        .to_dtype(DType::F32)?
        .reshape((batch * heads, queries, dim))?;
    let k = k
        .to_dtype(DType::F32)?
        .reshape((batch * heads, keys, dim))?;
    let v = v
        .to_dtype(DType::F32)?
        .reshape((batch * heads, keys, dim))?;
    if keys == 0 {
        candle_core::bail!("Qwen Image attention requires at least one key");
    }
    let mut scores = (q.matmul(&k.transpose(1, 2)?)? * (1.0 / (dim as f64).sqrt()))?;
    let mut mask = Vec::with_capacity(batch * heads * queries * keys);
    let mut query_valid = Vec::with_capacity(batch * heads * queries);
    for row in key_valid.iter().take(batch) {
        for _ in 0..heads {
            for query_index in 0..queries {
                let row_start = mask.len();
                let mut has_valid_key = false;
                for (key_index, &key_is_valid) in row.iter().take(keys).enumerate() {
                    let allowed =
                        key_is_valid && (!causal || key_index <= query_start + query_index);
                    has_valid_key |= allowed;
                    mask.push(if allowed { 0.0_f32 } else { f32::NEG_INFINITY });
                }
                if !has_valid_key {
                    // Softmax over all -inf is NaN.  Give the row a finite
                    // placeholder score, then zero its output below.  This
                    // matches SDPA's zero result for a fully masked query
                    // and keeps left-padded conditioning finite.
                    mask[row_start] = 0.0;
                }
                query_valid.push(if has_valid_key { 1.0_f32 } else { 0.0 });
            }
        }
    }
    let mask = Tensor::from_vec(mask, (batch * heads, queries, keys), q.device())?;
    scores = scores.broadcast_add(&mask)?;
    let probabilities = candle_nn::ops::softmax_last_dim(&scores)?;
    let query_valid = Tensor::from_vec(query_valid, (batch * heads, queries, 1), q.device())?;
    probabilities
        .matmul(&v)?
        .broadcast_mul(&query_valid)?
        .reshape((batch, heads, queries, dim))?
        .to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn bf16_qk_norm_keeps_inverse_scale_in_f32() -> Result<()> {
        let values = [0.125_f32, 0.375, 0.625, 0.875].repeat(32);
        let weights = [0.75_f32, 1.0, 1.25, 1.5].repeat(32);
        let eps = 1e-6_f32;
        let mean_square = values.iter().map(|x| x * x).sum::<f32>() / values.len() as f32;
        let inverse = (mean_square + eps).sqrt().recip();
        let round = |x| half::bf16::from_f32(x).to_f32();
        // Scalar reference: normalize in F32, round to BF16, then apply the
        // learned weight and round again. Inputs and weights are exact BF16.
        let expected = values
            .iter()
            .zip(&weights)
            .map(|(&x, &weight)| round(round(x * inverse) * weight))
            .collect::<Vec<_>>();
        let early_rounded = values
            .iter()
            .zip(&weights)
            .map(|(&x, &weight)| round(round(x * round(inverse)) * weight))
            .collect::<Vec<_>>();
        assert_ne!(
            expected, early_rounded,
            "fixture must detect early rounding"
        );
        let devices = std::iter::once(Device::Cpu);
        #[cfg(target_os = "macos")]
        let devices = devices.chain(crate::test_metal_device());
        for device in devices {
            let input =
                Tensor::from_slice(&values, (1, 1, 1, 128), &device)?.to_dtype(DType::BF16)?;
            let weight = Tensor::from_slice(&weights, 128, &device)?.to_dtype(DType::BF16)?;
            let actual = qk_rms_norm(&RmsNorm::new(weight, f64::from(eps)), &input)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(actual, expected, "BF16 Q/K RMSNorm on {device:?}");
        }
        Ok(())
    }

    #[test]
    fn normalization_epsilon_must_survive_f32_conversion() -> Result<()> {
        for eps in [1e-100, 1e300] {
            let config = TransformerConfig {
                eps,
                ..Default::default()
            };
            assert!(
                config.validate().is_err(),
                "accepted invalid F32 epsilon {eps}"
            );
        }
        let config = TransformerConfig::default();
        config.validate().map_err(candle_core::Error::wrap)?;
        let zeros = Tensor::zeros((1, 4), DType::F32, &Device::Cpu)?;
        assert!(
            NoAffineLayerNorm::new(4, config.eps)
                .forward(&zeros)?
                .to_vec2::<f32>()?[0]
                .iter()
                .all(|value| value.is_finite())
        );
        Ok(())
    }

    fn tiny_model(device: &Device, dtype: DType) -> Result<QwenImageTransformer> {
        let config = TransformerConfig {
            patch_size: 1,
            in_channels: 4,
            out_channels: Some(4),
            num_layers: 1,
            // 32 is one of the dimensions supported by the Metal attention
            // fast path.  Keep the fixture on that path so the BF16 test is
            // representative of the production model.
            attention_head_dim: 32,
            num_attention_heads: 2,
            context_in_dim: 4,
            mlp_ratio: 2,
            axes_dims_rope: [8, 12, 12],
            eps: 1e-6,
            causal_condition: true,
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, device);
        let model = QwenImageTransformer::load(vb, config).map_err(|error| {
            candle_core::Error::Msg(format!("tiny Qwen Image model load failed: {error}"))
        })?;
        let mut vars = varmap.data().lock().unwrap();
        let mut names = vars.keys().cloned().collect::<Vec<_>>();
        names.sort();
        for (var_index, name) in names.iter().enumerate() {
            let variable = vars.get_mut(name).expect("sorted VarMap key disappeared");
            let values = (0..variable.as_tensor().elem_count())
                .map(|index| {
                    let phase = (index + 1 + var_index * 3) as f32;
                    0.02 * phase.sin()
                })
                .collect::<Vec<_>>();
            let tensor =
                Tensor::from_vec(values, variable.shape().clone(), device)?.to_dtype(dtype)?;
            variable.set(&tensor)?;
        }
        drop(vars);
        Ok(model)
    }

    fn test_inputs(device: &Device, dtype: DType) -> Result<(Tensor, Tensor, Tensor)> {
        let hidden = Tensor::from_vec(
            (0..16)
                .map(|index| (index as f32 * 0.11).cos())
                .collect::<Vec<_>>(),
            (1, 4, 4),
            device,
        )?
        .to_dtype(dtype)?;
        let encoder = Tensor::from_vec(
            (0..12)
                .map(|index| (index as f32 * 0.09).sin())
                .collect::<Vec<_>>(),
            (1, 3, 4),
            device,
        )?
        .to_dtype(dtype)?;
        let timestep = Tensor::new(&[0.37_f32], device)?.to_dtype(dtype)?;
        Ok((hidden, encoder, timestep))
    }

    fn assert_finite(label: &str, values: &[f32]) {
        assert!(
            values.iter().all(|value| value.is_finite()),
            "{label} contains a non-finite value"
        );
    }

    #[test]
    fn coalesced_image_slots_preserve_mixed_runs_and_batch_order() -> Result<()> {
        let devices = std::iter::once(Device::Cpu);
        #[cfg(target_os = "macos")]
        let devices = devices.chain(crate::test_metal_device());
        for device in devices {
            for mask in [
                vec![false, false, true, true, false, true],
                vec![true; 7],
                vec![false; 5],
            ] {
                let sequence = mask.len();
                let images = mask.iter().filter(|v| **v).count() * IMAGE_TOKENS_PER_SLOT;
                // Metal does not allocate zero-byte buffers. The no-image
                // helper case is covered on CPU; real requests contain images.
                if images == 0 && device.is_metal() {
                    continue;
                }
                let text = Tensor::from_vec(
                    (0..2 * sequence * 3).map(|v| v as f32).collect::<Vec<_>>(),
                    (2, sequence, 3),
                    &device,
                )?;
                let image = Tensor::from_vec(
                    (0..2 * images * 3)
                        .map(|v| v as f32 + 1000.0)
                        .collect::<Vec<_>>(),
                    (2, images, 3),
                    &device,
                )?;
                let (got, expanded) = expand_joint_sequence(&text, &image, &mask)?;
                let mut chunks = Vec::new();
                let mut cursor = 0;
                let mut expected_mask = Vec::new();
                for (i, is_image) in mask.iter().enumerate() {
                    if *is_image {
                        chunks.push(image.narrow(1, cursor, 4)?);
                        cursor += 4;
                        expected_mask.extend([true; 4]);
                    } else {
                        chunks.push(text.narrow(1, i, 1)?);
                        expected_mask.push(false);
                    }
                }
                assert_eq!(expanded, expected_mask);
                assert_eq!(
                    got.flatten_all()?.to_vec1::<f32>()?,
                    Tensor::cat(&chunks, 1)?.flatten_all()?.to_vec1::<f32>()?
                );
            }
        }
        Ok(())
    }

    fn dense_block_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        image_ids: &[i64],
    ) -> Result<Tensor> {
        let (_, _, sequence, dim) = q.dims4()?;
        let q = q.flatten(0, 1)?.to_dtype(DType::F32)?;
        let k = k.flatten(0, 1)?.to_dtype(DType::F32)?;
        let v = v.flatten(0, 1)?.to_dtype(DType::F32)?;
        let mut scores = (q.matmul(&k.transpose(1, 2)?)? * (1.0 / (dim as f64).sqrt()))?;
        let mut mask = Vec::with_capacity(sequence * sequence);
        for query in 0..sequence {
            for key in 0..sequence {
                let query_image = image_ids[query];
                let allowed = if query_image < 0 {
                    key <= query
                } else {
                    image_ids[key] == query_image || key < query && image_ids[key] < query_image
                };
                mask.push(if allowed { 0.0 } else { f32::NEG_INFINITY });
            }
        }
        scores = scores.broadcast_add(&Tensor::from_vec(
            mask,
            (1, sequence, sequence),
            q.device(),
        )?)?;
        candle_nn::ops::softmax_last_dim(&scores)?
            .matmul(&v)?
            .reshape(q.dims())
    }

    #[test]
    fn block_causal_segments_match_dense_reference() -> Result<()> {
        let device = Device::Cpu;
        let sequence = 11;
        let q = Tensor::from_vec(
            (0..sequence * 4)
                .map(|index| (index as f32 * 0.17).sin())
                .collect::<Vec<_>>(),
            (1, 1, sequence, 4),
            &device,
        )?;
        let k = Tensor::from_vec(
            (0..sequence * 4)
                .map(|index| (index as f32 * 0.13).cos())
                .collect::<Vec<_>>(),
            (1, 1, sequence, 4),
            &device,
        )?;
        let v = Tensor::from_vec(
            (0..sequence * 4)
                .map(|index| (index as f32 * 0.07).sin())
                .collect::<Vec<_>>(),
            (1, 1, sequence, 4),
            &device,
        )?;
        let image_ids = [-1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1];
        let segments = prefix_segments(&image_ids, 7);
        let mut pieces = Vec::new();
        for &(start, end, is_text) in &segments {
            pieces.push(attend_segment(
                &q.narrow(2, start, end - start)?,
                &k.narrow(2, 0, end)?,
                &v.narrow(2, 0, end)?,
                is_text,
                start,
                None,
            )?);
        }
        pieces.push(attend_segment(&q.narrow(2, 7, 4)?, &k, &v, false, 7, None)?);
        let actual = Tensor::cat(&pieces, 2)?.to_dtype(DType::F32)?;
        let expected = dense_block_attention(&q, &k, &v, &image_ids)?;
        let actual_values = actual.flatten_all()?.to_vec1::<f32>()?;
        let expected_values = expected.flatten_all()?.to_vec1::<f32>()?;
        assert_finite("segmented attention actual", &actual_values);
        assert_finite("segmented attention expected", &expected_values);
        let max_error = actual_values
            .iter()
            .zip(expected_values)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_error < 1e-5,
            "segmented attention max error {max_error}"
        );
        Ok(())
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn native_rope_tables_check_bounds_and_preserve_cpu_f32() -> Result<()> {
        assert_eq!(
            native_rope_table_size(131_072, &[16, 56, 56])?,
            8 * 1024 * 1024
        );
        assert!(native_rope_table_size(131_073, &[16, 56, 56]).is_err());
        assert!(native_rope_table_size(usize::MAX, &[16, 56, 56]).is_err());
        for axes in [[0, 2, 2], [3, 2, 2], [usize::MAX - 1; 3]] {
            assert!(native_rope_table_size(1, &axes).is_err());
        }
        assert!(native_rope_table_size(0, &[2, 2, 2]).is_err());
        let positions = [-1024, -1, 0, 8191];
        let axes = [16, 56, 56];
        let (old_cos, old_sin) =
            build_rope(&positions, &positions, &positions, &axes, &Device::Cpu)?;
        for dtype in [DType::F32, DType::BF16] {
            let (cos, sin) = build_rope_for_dtype(
                &positions,
                &positions,
                &positions,
                &axes,
                dtype,
                &Device::Cpu,
            )?;
            assert_eq!(cos.to_vec2::<f32>()?, old_cos.to_vec2::<f32>()?);
            assert_eq!(sin.to_vec2::<f32>()?, old_sin.to_vec2::<f32>()?);
        }
        for position in [-1025, 8192] {
            assert!(build_rope_native_bf16(&[position], &[0], &[0], &axes, &Device::Cpu).is_err());
        }
        assert!(build_rope_native_bf16(&[0], &[], &[0], &axes, &Device::Cpu).is_err());
        let (cos, sin) =
            build_rope_native_bf16(&positions, &positions, &positions, &axes, &Device::Cpu)?;
        assert!(
            cos.to_vec2::<f32>()?
                .iter()
                .flatten()
                .all(|x| x.is_finite())
        );
        assert!(
            sin.to_vec2::<f32>()?
                .iter()
                .flatten()
                .all(|x| x.is_finite())
        );
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (old_cos, old_sin) = build_rope(&positions, &positions, &positions, &axes, &device)?;
        let (cos, sin) = build_rope_for_dtype(
            &positions,
            &positions,
            &positions,
            &axes,
            DType::F32,
            &device,
        )?;
        assert_eq!(cos.to_vec2::<f32>()?, old_cos.to_vec2::<f32>()?);
        assert_eq!(sin.to_vec2::<f32>()?, old_sin.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn rope_fixture_matches_pair_stride_and_text_axes() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let (cos, _) = build_rope(&[1], &[0], &[0], &[4, 2, 2], &device)?;
        let values = cos.to_vec2::<f32>()?;
        assert!((values[0][0] - 1.0_f32.cos()).abs() < 1e-6);
        assert!((values[0][1] - 0.01_f32.cos()).abs() < 1e-6);
        assert!((values[0][2] - 1.0).abs() < 1e-6);
        assert!((values[0][3] - 1.0).abs() < 1e-6);

        let (frame, height, width) =
            build_rope_positions(&[false, false, true], &[ImageShape::new(1, 1, 1)])?;
        assert_eq!(frame, [0, 1, 2]);
        assert_eq!(height[..2], frame[..2]);
        assert_eq!(width[..2], frame[..2]);
        Ok(())
    }

    #[test]
    fn extract_and_cached_target_outputs_match() -> Result<()> {
        let device = Device::Cpu;
        let model = tiny_model(&device, DType::F32)?;
        let (hidden, encoder, timestep) = test_inputs(&device, DType::F32)?;
        let shapes = [ImageShape::new(1, 2, 2)];
        let image_mask = [false, false, false, true];
        let mut cache = TransformerKvCache::new(model.num_layers());
        let extracted = model.forward(
            &hidden,
            &encoder,
            &timestep,
            &shapes,
            &image_mask,
            None,
            Some(&mut cache),
            Some(KvCacheMode::Extract),
        )?;
        let cached = model.forward(
            &hidden,
            &encoder,
            &timestep,
            &shapes,
            &image_mask,
            None,
            Some(&mut cache),
            Some(KvCacheMode::Cached),
        )?;
        let expected = extracted.narrow(1, 3, 4)?.flatten_all()?.to_vec1::<f32>()?;
        let actual = cached.flatten_all()?.to_vec1::<f32>()?;
        assert_finite("cached target expected", &expected);
        assert_finite("cached target actual", &actual);
        let max_error = expected
            .iter()
            .zip(actual)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_error < 1e-5, "cached target max error {max_error}");
        Ok(())
    }

    #[test]
    fn tiny_model_matches_independent_reference_fixture() -> Result<()> {
        // Expected values were generated independently with PyTorch 1.13.1
        // from the same sorted-name sine weight recipe and the upstream
        // Qwen-Image equations (F32, one block, two heads of width 32).  The
        // fixture is intentionally kept here as numbers so this regression
        // does not depend on Python or on a runtime reference implementation.
        let device = Device::Cpu;
        let model = tiny_model(&device, DType::F32)?;
        let (hidden, encoder, timestep) = test_inputs(&device, DType::F32)?;
        let shapes = [ImageShape::new(1, 2, 2)];
        let image_mask = [false, false, false, true];
        let output = model.forward(
            &hidden,
            &encoder,
            &timestep,
            &shapes,
            &image_mask,
            None,
            None,
            None,
        )?;
        let actual = output.narrow(1, 3, 4)?.flatten_all()?.to_vec1::<f32>()?;
        // Keep the externally generated decimal precision; rounding these
        // literals would weaken the independent-reference regression.
        #[allow(clippy::excessive_precision)]
        let expected = [
            0.0027400255,
            -0.0081772804,
            -0.0091486648,
            0.0010073408,
            0.0013417378,
            -0.0098347086,
            -0.0090493374,
            0.0027426258,
            -0.0012798496,
            -0.0125918146,
            -0.0085885413,
            0.0058608465,
            -0.0104735941,
            -0.0178081952,
            -0.0034829490,
            0.0150785614,
        ];
        assert_finite("independent Qwen Image fixture actual", &actual);
        assert_finite("independent Qwen Image fixture expected", &expected);
        let max_error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_error < 2e-5,
            "independent Qwen Image fixture max error {max_error}"
        );
        Ok(())
    }

    #[test]
    fn encoder_mask_and_cache_edge_cases_are_bounded() -> Result<()> {
        let device = Device::Cpu;
        let model = tiny_model(&device, DType::F32)?;
        let (hidden, encoder, timestep) = test_inputs(&device, DType::F32)?;
        let shapes = [ImageShape::new(1, 2, 2)];
        let image_mask = [false, false, false, true];
        let encoder_mask = Tensor::zeros((1, 3), DType::F32, &device)?;
        let masked_error = match model.forward(
            &hidden,
            &encoder,
            &timestep,
            &shapes,
            &image_mask,
            Some(&encoder_mask),
            None,
            None,
        ) {
            Ok(_) => panic!("all-masked conditioning should fail closed"),
            Err(error) => error.to_string(),
        };
        assert!(
            masked_error.contains("cannot wholly mask conditioning"),
            "unexpected all-masked conditioning error: {masked_error}"
        );

        let left_padded_mask = Tensor::new(&[[0.0_f32, 1.0, 1.0]], &device)?;
        let left_padded = model.forward(
            &hidden,
            &encoder,
            &timestep,
            &shapes,
            &image_mask,
            Some(&left_padded_mask),
            None,
            None,
        )?;
        let left_padded_values = left_padded.flatten_all()?.to_vec1::<f32>()?;
        assert_finite("left-padded conditioning", &left_padded_values);

        assert!(
            model
                .forward(
                    &hidden,
                    &encoder,
                    &timestep,
                    &shapes,
                    &image_mask,
                    None,
                    None,
                    Some(KvCacheMode::Cached),
                )
                .is_err()
        );
        let mut wrong_layer_count = TransformerKvCache::new(0);
        assert!(
            model
                .forward(
                    &hidden,
                    &encoder,
                    &timestep,
                    &shapes,
                    &image_mask,
                    None,
                    Some(&mut wrong_layer_count),
                    None,
                )
                .is_err()
        );
        let mut empty_cache = TransformerKvCache::new(model.num_layers());
        assert!(
            model
                .forward(
                    &hidden,
                    &encoder,
                    &timestep,
                    &shapes,
                    &image_mask,
                    None,
                    Some(&mut empty_cache),
                    Some(KvCacheMode::Cached),
                )
                .is_err()
        );
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn extract_and_cached_target_outputs_match_bf16_metal() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let model = tiny_model(&device, DType::BF16)?;
        let (hidden, encoder, timestep) = test_inputs(&device, DType::BF16)?;
        let shapes = [ImageShape::new(1, 2, 2)];
        let image_mask = [false, false, false, true];
        let mut cache = TransformerKvCache::new(model.num_layers());
        let extracted = model.forward(
            &hidden,
            &encoder,
            &timestep,
            &shapes,
            &image_mask,
            None,
            Some(&mut cache),
            Some(KvCacheMode::Extract),
        )?;
        let cached = model.forward(
            &hidden,
            &encoder,
            &timestep,
            &shapes,
            &image_mask,
            None,
            Some(&mut cache),
            Some(KvCacheMode::Cached),
        )?;
        let expected = extracted
            .narrow(1, 3, 4)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let actual = cached
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_finite("cached BF16 target expected", &expected);
        assert_finite("cached BF16 target actual", &actual);
        let max_error = expected
            .iter()
            .zip(actual)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_error < 0.02, "cached BF16 target max error {max_error}");

        // Compare the same deterministic BF16 fixture through Candle's CPU
        // reference attention and Metal's native attention implementation.
        let cpu_device = Device::Cpu;
        // Candle's CPU backend intentionally does not implement BF16 GEMM;
        // compare Metal BF16 against the F32 reference path instead.
        let cpu_model = tiny_model(&cpu_device, DType::F32)?;
        let (cpu_hidden, cpu_encoder, cpu_timestep) = test_inputs(&cpu_device, DType::F32)?;
        let mut cpu_cache = TransformerKvCache::new(cpu_model.num_layers());
        let cpu_extracted = cpu_model.forward(
            &cpu_hidden,
            &cpu_encoder,
            &cpu_timestep,
            &shapes,
            &image_mask,
            None,
            Some(&mut cpu_cache),
            Some(KvCacheMode::Extract),
        )?;
        let cpu_values = cpu_extracted
            .narrow(1, 3, 4)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let metal_values = extracted
            .narrow(1, 3, 4)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_finite("BF16 CPU target", &cpu_values);
        assert_finite("BF16 Metal target", &metal_values);
        let max_device_error = cpu_values
            .iter()
            .zip(metal_values)
            .map(|(cpu, metal)| (cpu - metal).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_device_error < 0.05,
            "BF16 CPU/Metal target max error {max_device_error}"
        );
        Ok(())
    }
}

#[cfg(test)]
mod text_projection_gelu_tests {
    use super::*;

    #[test]
    fn text_projection_gelu_preserves_cpu_f32_path() -> Result<()> {
        let input = Tensor::from_vec(vec![-2.0_f32, -0.5, 0.0, 1.0, 4.0], 5, &Device::Cpu)?;
        let expected = candle_nn::Activation::GeluPytorchTanh
            .forward(&input)?
            .to_vec1::<f32>()?;
        assert_eq!(text_projection_gelu(&input)?.to_vec1::<f32>()?, expected);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn text_projection_gelu_bf16_metal_f64_bound_and_f16_fallback() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let values = (0..259)
            .flat_map(|i| [((i as f32) - 129.0) / 32.0, 0.0])
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(values, (259, 2), &device)?
            .to_dtype(DType::BF16)?
            .narrow(0, 1, 257)?
            .narrow(1, 0, 1)?;
        let output = text_projection_gelu(&input)?;
        assert_eq!(output.dtype(), DType::BF16);
        assert_eq!(output.dims(), input.dims());
        let actual = output
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let old = candle_nn::Activation::GeluPytorchTanh
            .forward(&input)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut error = 0.0;
        let mut old_error = 0.0;
        for (i, (&actual, &old)) in actual.iter().zip(&old).enumerate() {
            let x = (i as f64 - 128.0) / 32.0;
            let expected = 0.5
                * x
                * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh());
            let delta = f64::from(actual) - expected;
            assert!(delta.abs() <= 2e-6 + 0.004 * expected.abs());
            error += delta * delta;
            old_error += (f64::from(old) - expected).powi(2);
        }
        assert!(
            error < 0.8 * old_error,
            "F32 GELU should reduce local BF16 rounding error"
        );
        let half = input.to_dtype(DType::F16)?;
        let expected = candle_nn::Activation::GeluPytorchTanh
            .forward(&half)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(
            text_projection_gelu(&half)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            expected
        );
        Ok(())
    }
}

#[cfg(test)]
mod timestep_contract_tests {
    use super::*;

    fn old_projection(timestep: &Tensor, dim: usize) -> Result<Tensor> {
        let half = dim / 2;
        let values = (0..half)
            .map(|i| (-10000_f64.ln() * i as f64 / half as f64).exp() as f32)
            .collect::<Vec<_>>();
        let table = Tensor::from_vec(values, half, timestep.device())?;
        let scaled = (timestep.to_dtype(DType::F32)? * 1000.)?;
        let args = scaled.unsqueeze(1)?.broadcast_mul(&table.unsqueeze(0)?)?;
        Tensor::cat(&[&args.cos()?, &args.sin()?], D::Minus1)
    }

    #[test]
    fn timestep_contract_keeps_cpu_paths() -> Result<()> {
        for dtype in [DType::F32, DType::BF16] {
            let input = Tensor::new(&[0_f32, 0.125, 0.37, 1.], &Device::Cpu)?.to_dtype(dtype)?;
            for dim in [16, 32, 256] {
                assert_eq!(
                    timestep_embedding(&input, dim, 1000., 10000.)?.to_vec2::<f32>()?,
                    old_projection(&input, dim)?.to_vec2::<f32>()?
                );
            }
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn timestep_contract_bf16_metal_views_f64_and_fallback() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let padded = Tensor::from_vec(
            vec![9_f32, 9., 0., 9., 0.125, 9., 0.37, 9., 1., 9., 9., 9.],
            (6, 2),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let input = padded.narrow(0, 1, 4)?.narrow(1, 0, 1)?.squeeze(1)?;
        let values = input.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        for dim in [16, 32, 256] {
            let output = timestep_embedding(&input, dim, 1000., 10000.)?;
            assert_eq!(output.dtype(), DType::F32);
            assert_eq!(output.dims(), &[4, dim]);
            let actual = output.to_vec2::<f32>()?;
            assert_eq!(
                actual,
                timestep_embedding(&input.contiguous()?, dim, 1000., 10000.)?.to_vec2::<f32>()?
            );
            let rounded = output
                .to_dtype(DType::BF16)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            for (row, &time) in values.iter().enumerate() {
                for i in 0..dim / 2 {
                    let angle = f64::from(time)
                        * 1000.
                        * (-10000_f64.ln() * i as f64 / (dim / 2) as f64).exp();
                    for (column, exact) in [(i, angle.cos()), (i + dim / 2, angle.sin())] {
                        assert!((f64::from(actual[row][column]) - exact).abs() <= 1.2e-4);
                        assert!((f64::from(rounded[row][column]) - exact).abs() <= 0.0041);
                    }
                }
            }
        }
        for dtype in [DType::F32, DType::F16] {
            let input = input.to_dtype(dtype)?;
            assert_eq!(
                timestep_embedding(&input, 256, 1000., 10000.)?.to_vec2::<f32>()?,
                old_projection(&input, 256)?.to_vec2::<f32>()?
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod time_mlp_contract_tests {
    use super::*;
    use std::collections::HashMap;

    fn projection(device: &Device, dtype: DType) -> Result<Linear> {
        let values = (0..13 * 32)
            .map(|i| ((i * 17 % 97) as f32 - 48.) / 64.)
            .collect::<Vec<_>>();
        let weight = Tensor::from_vec(values, (13, 32), device)?.to_dtype(dtype)?;
        linear_no_bias(
            32,
            13,
            VarBuilder::from_tensors(HashMap::from([("weight".into(), weight)]), dtype, device),
        )
    }

    fn input(device: &Device, dtype: DType) -> Result<Tensor> {
        Tensor::from_vec(
            (0..4 * 32)
                .map(|i| ((i * 13 % 67) as f32 - 33.) / 32.)
                .collect::<Vec<_>>(),
            (32, 4),
            device,
        )?
        .to_dtype(dtype)?
        .narrow(1, 1, 2)?
        .t()
    }

    #[test]
    fn keeps_cpu_and_f32_paths() -> Result<()> {
        let devices = std::iter::once(Device::Cpu);
        #[cfg(target_os = "macos")]
        let devices = devices.chain(crate::test_metal_device());
        for device in devices {
            for dtype in [DType::F32, DType::BF16] {
                if device.is_metal() && dtype == DType::BF16 {
                    continue;
                }
                let linear = projection(&device, dtype)?;
                let x = input(&device, dtype)?;
                match (conditioning_linear(&linear, &x), linear.forward(&x)) {
                    (Ok(actual), Ok(expected)) => assert_eq!(
                        actual.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                        expected.to_dtype(DType::F32)?.to_vec2::<f32>()?
                    ),
                    (Err(actual), Err(expected)) => {
                        assert_eq!(actual.to_string(), expected.to_string())
                    }
                    _ => panic!("time projection changed fallback success/error behavior"),
                }
            }
        }
        Ok(())
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn bf16_two_row_strided_input_matches_f64() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let linear = projection(&device, DType::BF16)?;
        let x = input(&device, DType::BF16)?;
        let out = conditioning_linear(&linear, &x)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        let a = x.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let w = linear.weight().to_dtype(DType::F32)?.to_vec2::<f32>()?;
        for (row, actual) in a.iter().zip(out.iter()) {
            for (weight, &value) in w.iter().zip(actual.iter()) {
                let expected = row
                    .iter()
                    .zip(weight)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum::<f64>();
                assert!((f64::from(value) - expected).abs() <= 0.004 * expected.abs() + 1e-6);
            }
        }
        Ok(())
    }
}
