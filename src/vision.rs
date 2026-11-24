//! Pure Rust vision encoders for GGUF multimodal projectors.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use crate::qmatmul::QMatMul;
use anyhow::{Context, Result};
use candle_core::quantized::{QTensor, gguf_file};
use candle_core::{D, DType, Device, IndexOp, Module, Tensor};
use candle_nn::{LayerNorm, RmsNorm};

use crate::gguf_limits::{validate_gguf_header_budget, validate_layer_count};

mod inkling2;
pub use inkling2::{Inkling2ImagePlan, Inkling2VisionEncoder};

const LLAMA4_PROJECTOR_TYPE: &str = "llama4";
const GEMMA4_PROJECTOR_TYPE: &str = "gemma4v";
const GEMMA4_POOLING_KERNEL: usize = 3;
const GEMMA4_MAX_IMAGE_TOKENS: usize = 280;
const GEMMA4_ROPE_THETA: f64 = 100.0;
// Bound metadata-driven image and RoPE tables before loading any weights.
const VISION_MAX_IMAGE_PIXELS: usize = 64 * 1024 * 1024;
const LLAMA4_MAX_PATCHES: usize = 4_096;
const VISION_MAX_HIDDEN_DIM: usize = 16_384;
const VISION_MAX_DECODER_DIM: usize = 65_536;
const VISION_MAX_HEAD_DIM: usize = 1_024;

struct Linear {
    weight: QMatMul,
    bias: Option<Tensor>,
}

impl Linear {
    fn load(
        tensors: &HashMap<String, Arc<QTensor>>,
        weight_name: &str,
        bias_name: Option<&str>,
        device: &Device,
    ) -> Result<Self> {
        let weight = tensors
            .get(weight_name)
            .with_context(|| format!("vision projector is missing {weight_name}"))?;
        let bias = bias_name
            .map(|name| {
                tensors
                    .get(name)
                    .with_context(|| format!("vision projector is missing {name}"))?
                    .dequantize(device)
                    .map_err(anyhow::Error::from)
            })
            .transpose()?;
        Ok(Self {
            weight: QMatMul::from_arc(Arc::clone(weight))?,
            bias,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = self.weight.forward(input)?;
        match self.bias.as_ref() {
            Some(bias) => output.broadcast_add(bias).map_err(Into::into),
            None => Ok(output),
        }
    }
}

struct VisionAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    output: Linear,
    heads: usize,
    head_dim: usize,
    rope_cos: Tensor,
    rope_sin: Tensor,
}

impl VisionAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden) = input.dims3()?;
        anyhow::ensure!(
            hidden == self.heads * self.head_dim,
            "vision attention width does not match its head layout"
        );

        let q = self
            .q
            .forward(input)?
            .reshape((batch, seq_len, self.heads, self.head_dim))?;
        let k = self
            .k
            .forward(input)?
            .reshape((batch, seq_len, self.heads, self.head_dim))?;
        let v = self
            .v
            .forward(input)?
            .reshape((batch, seq_len, self.heads, self.head_dim))?;
        let q = apply_vision_rope(&q, &self.rope_cos, &self.rope_sin)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = apply_vision_rope(&k, &self.rope_cos, &self.rope_sin)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scores =
            (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? / (self.head_dim as f64).sqrt())?;
        let probabilities = candle_nn::ops::softmax_last_dim(&scores)?;
        let output = probabilities
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, hidden))?;
        self.output.forward(&output)
    }
}

struct VisionMlp {
    up: Linear,
    down: Linear,
}

impl VisionMlp {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.down.forward(&self.up.forward(input)?.gelu_erf()?)
    }
}

struct VisionBlock {
    norm1: LayerNorm,
    attention: VisionAttention,
    norm2: LayerNorm,
    mlp: VisionMlp,
}

impl VisionBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let attention = self.attention.forward(&self.norm1.forward(input)?)?;
        let hidden = (input + attention)?;
        let mlp = self.mlp.forward(&self.norm2.forward(&hidden)?)?;
        (hidden + mlp).map_err(Into::into)
    }
}

/// Llama 4's ViT and decoder projector loaded from `mmproj-F16.gguf`.
pub struct Llama4VisionEncoder {
    patch_embedding: Linear,
    class_embedding: Tensor,
    position_embedding: Tensor,
    pre_norm: LayerNorm,
    blocks: Vec<VisionBlock>,
    post_norm: LayerNorm,
    adapter_up: Linear,
    adapter_down: Linear,
    decoder_projection: Linear,
    image_size: usize,
    patch_size: usize,
    hidden_dim: usize,
    decoder_dim: usize,
    merge_factor: usize,
    device: Device,
}

impl Llama4VisionEncoder {
    /// Load a Llama 4 vision projector from a GGUF file onto `device`.
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let mut file = File::open(path)
            .with_context(|| format!("could not open vision projector '{}'", path.display()))?;
        validate_gguf_header_budget(&mut file)
            .with_context(|| format!("could not validate vision projector '{}'", path.display()))?;
        let content = gguf_file::Content::read(&mut file)
            .with_context(|| format!("could not parse vision projector '{}'", path.display()))?;

        let architecture = metadata_string(&content.metadata, "general.architecture")?;
        anyhow::ensure!(
            architecture == "clip",
            "vision projector architecture is '{architecture}', not 'clip'"
        );
        let projector_type = metadata_string(&content.metadata, "clip.projector_type")?;
        anyhow::ensure!(
            projector_type == LLAMA4_PROJECTOR_TYPE,
            "vision projector type '{projector_type}' is not supported"
        );
        let image_size = metadata_usize(&content.metadata, "clip.vision.image_size")?;
        let patch_size = metadata_usize(&content.metadata, "clip.vision.patch_size")?;
        let hidden_dim = metadata_usize(&content.metadata, "clip.vision.embedding_length")?;
        let decoder_dim = metadata_usize(&content.metadata, "clip.vision.projection_dim")?;
        let feed_forward_dim =
            metadata_usize(&content.metadata, "clip.vision.feed_forward_length")?;
        let block_count = metadata_usize(&content.metadata, "clip.vision.block_count")?;
        let heads = metadata_usize(&content.metadata, "clip.vision.attention.head_count")?;
        let merge_factor = metadata_usize(&content.metadata, "clip.vision.projector.scale_factor")?;
        let norm_eps = metadata_f64(
            &content.metadata,
            "clip.vision.attention.layer_norm_epsilon",
        )?;

        anyhow::ensure!(image_size > 0, "vision image size must be positive");
        anyhow::ensure!(patch_size > 0, "vision patch size must be positive");
        let image_pixels = image_size
            .checked_mul(image_size)
            .context("vision image pixel count overflow")?;
        anyhow::ensure!(
            image_pixels <= VISION_MAX_IMAGE_PIXELS,
            "vision image pixel count exceeds {VISION_MAX_IMAGE_PIXELS}"
        );
        anyhow::ensure!(
            (1..=VISION_MAX_HIDDEN_DIM).contains(&hidden_dim),
            "vision hidden width must be in 1..={VISION_MAX_HIDDEN_DIM}"
        );
        anyhow::ensure!(
            (1..=VISION_MAX_DECODER_DIM).contains(&decoder_dim),
            "vision decoder width must be in 1..={VISION_MAX_DECODER_DIM}"
        );
        validate_layer_count(block_count, "vision block count")?;
        anyhow::ensure!(heads > 0, "vision head count must be positive");
        anyhow::ensure!(
            norm_eps.is_finite() && (norm_eps as f32).is_finite() && (norm_eps as f32) > 0.0,
            "vision layer norm epsilon must be finite, positive, and representable in F32"
        );
        anyhow::ensure!(
            image_size % patch_size == 0,
            "vision image size is not divisible by its patch size"
        );
        let patches_per_side = image_size / patch_size;
        let patch_count = patches_per_side
            .checked_mul(patches_per_side)
            .context("vision patch count overflow")?;
        anyhow::ensure!(
            patch_count <= LLAMA4_MAX_PATCHES,
            "vision patch count exceeds {LLAMA4_MAX_PATCHES}"
        );
        anyhow::ensure!(
            merge_factor > 0 && patches_per_side % merge_factor == 0,
            "vision patch grid is not divisible by its merge factor"
        );
        anyhow::ensure!(
            hidden_dim % heads == 0
                && (4..=VISION_MAX_HEAD_DIM).contains(&(hidden_dim / heads))
                && (hidden_dim / heads) % 4 == 0,
            "vision head width cannot use Llama 4 two-axis RoPE"
        );
        let merge_area = merge_factor
            .checked_mul(merge_factor)
            .context("vision merge area overflow")?;
        let adapter_width = hidden_dim
            .checked_mul(merge_area)
            .context("vision adapter width overflow")?;
        let expected_feed_forward_dim = hidden_dim
            .checked_mul(4)
            .context("vision feed-forward width overflow")?;
        anyhow::ensure!(
            feed_forward_dim == expected_feed_forward_dim,
            "unsupported Llama 4 vision feed-forward width"
        );

        tracing::info!(
            path = %path.display(),
            image_size,
            patch_size,
            hidden_dim,
            decoder_dim,
            block_count,
            "Loading pure Candle Llama 4 vision projector"
        );
        let mut tensors = HashMap::with_capacity(content.tensor_infos.len());
        for name in content.tensor_infos.keys() {
            let tensor = content
                .tensor(&mut file, name, device)
                .with_context(|| format!("could not load vision tensor {name}"))?;
            tensors.insert(name.clone(), Arc::new(tensor));
        }

        let tensor = |name: &str| -> Result<Tensor> {
            tensors
                .get(name)
                .with_context(|| format!("vision projector is missing {name}"))?
                .dequantize(device)
                .map_err(Into::into)
        };
        let norm = |prefix: &str| -> Result<LayerNorm> {
            Ok(LayerNorm::new(
                tensor(&format!("{prefix}.weight"))?,
                tensor(&format!("{prefix}.bias"))?,
                norm_eps,
            ))
        };

        let rope_cos_sin =
            build_vision_rope(patches_per_side, hidden_dim / heads, 10_000.0, device)?;
        let mut blocks = Vec::with_capacity(block_count);
        for index in 0..block_count {
            let prefix = format!("v.blk.{index}");
            let attention = VisionAttention {
                q: Linear::load(
                    &tensors,
                    &format!("{prefix}.attn_q.weight"),
                    Some(&format!("{prefix}.attn_q.bias")),
                    device,
                )?,
                k: Linear::load(
                    &tensors,
                    &format!("{prefix}.attn_k.weight"),
                    Some(&format!("{prefix}.attn_k.bias")),
                    device,
                )?,
                v: Linear::load(
                    &tensors,
                    &format!("{prefix}.attn_v.weight"),
                    Some(&format!("{prefix}.attn_v.bias")),
                    device,
                )?,
                output: Linear::load(
                    &tensors,
                    &format!("{prefix}.attn_out.weight"),
                    Some(&format!("{prefix}.attn_out.bias")),
                    device,
                )?,
                heads,
                head_dim: hidden_dim / heads,
                rope_cos: rope_cos_sin.0.clone(),
                rope_sin: rope_cos_sin.1.clone(),
            };
            blocks.push(VisionBlock {
                norm1: norm(&format!("{prefix}.ln1"))?,
                attention,
                norm2: norm(&format!("{prefix}.ln2"))?,
                mlp: VisionMlp {
                    up: Linear::load(
                        &tensors,
                        &format!("{prefix}.ffn_up.weight"),
                        Some(&format!("{prefix}.ffn_up.bias")),
                        device,
                    )?,
                    down: Linear::load(
                        &tensors,
                        &format!("{prefix}.ffn_down.weight"),
                        Some(&format!("{prefix}.ffn_down.bias")),
                        device,
                    )?,
                },
            });
        }

        let adapter_up = Linear::load(&tensors, "mm.model.mlp.1.weight", None, device)?;
        let adapter_down = Linear::load(&tensors, "mm.model.mlp.2.weight", None, device)?;
        let decoder_projection = Linear::load(&tensors, "mm.model.fc.weight", None, device)?;
        validate_linear_shape(&tensors, "mm.model.mlp.1.weight", 4_096, adapter_width)?;
        validate_linear_shape(&tensors, "mm.model.mlp.2.weight", 4_096, 4_096)?;
        validate_linear_shape(&tensors, "mm.model.fc.weight", decoder_dim, 4_096)?;

        Ok(Self {
            patch_embedding: Linear::load(&tensors, "v.patch_embd.weight", None, device)?,
            class_embedding: tensor("v.class_embd")?,
            position_embedding: tensor("v.position_embd.weight")?,
            pre_norm: norm("v.pre_ln")?,
            blocks,
            post_norm: norm("v.post_ln")?,
            adapter_up,
            adapter_down,
            decoder_projection,
            image_size,
            patch_size,
            hidden_dim,
            decoder_dim,
            merge_factor,
            device: device.clone(),
        })
    }

    /// Return the square input side length required by the projector.
    pub fn image_size(&self) -> usize {
        self.image_size
    }

    /// Return the language-model embedding width produced by the projector.
    pub fn decoder_dim(&self) -> usize {
        self.decoder_dim
    }

    /// Return the number of language-model tokens produced for one image.
    pub fn image_tokens(&self) -> usize {
        let side = self.image_size / self.patch_size / self.merge_factor;
        side * side
    }

    /// Encode one normalized RGB image in CHW order. Values must use the
    /// projector's `[-1, 1]` range and exact square image size.
    pub fn encode(&self, image: &Tensor) -> Result<Tensor> {
        anyhow::ensure!(
            image.device().same_device(&self.device),
            "vision image and projector are on different devices"
        );
        anyhow::ensure!(
            image.dims() == [3, self.image_size, self.image_size],
            "vision image must have shape [3, {}, {}]",
            self.image_size,
            self.image_size
        );
        let patches_per_side = self.image_size / self.patch_size;
        let patch_values = image
            .unfold(1, self.patch_size, self.patch_size)?
            .unfold(2, self.patch_size, self.patch_size)?
            .permute((1, 2, 0, 3, 4))?
            .contiguous()?
            .reshape((
                1,
                patches_per_side * patches_per_side,
                3 * self.patch_size * self.patch_size,
            ))?;
        let mut hidden = self.patch_embedding.forward(&patch_values)?;
        anyhow::ensure!(
            hidden.dims3()? == (1, patches_per_side * patches_per_side, self.hidden_dim),
            "vision patch embedding returned the wrong shape"
        );
        let class = self.class_embedding.reshape((1, 1, self.hidden_dim))?;
        hidden = Tensor::cat(&[&hidden, &class], 1)?;
        hidden = hidden.broadcast_add(&self.position_embedding.unsqueeze(0)?)?;
        hidden = self.pre_norm.forward(&hidden)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        hidden = self.post_norm.forward(&hidden)?;
        hidden = hidden.narrow(1, 0, patches_per_side * patches_per_side)?;
        hidden = pixel_merge(
            &hidden,
            patches_per_side,
            self.hidden_dim,
            self.merge_factor,
        )?;
        hidden = self.adapter_up.forward(&hidden)?.gelu_erf()?;
        hidden = self.adapter_down.forward(&hidden)?.gelu_erf()?;
        let projected = self.decoder_projection.forward(&hidden)?;
        anyhow::ensure!(
            projected.dims3()? == (1, self.image_tokens(), self.decoder_dim),
            "vision projector returned the wrong decoder embedding shape"
        );
        projected.squeeze(0).map_err(Into::into)
    }
}

struct Gemma4VisionAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    output: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    v_norm: RmsNorm,
    heads: usize,
    head_dim: usize,
}

impl Gemma4VisionAttention {
    fn forward(&self, input: &Tensor, grid_width: usize, grid_height: usize) -> Result<Tensor> {
        let (batch, seq_len, hidden_dim) = input.dims3()?;
        anyhow::ensure!(
            seq_len == grid_width * grid_height,
            "Gemma 4 vision grid does not match its patch count"
        );
        anyhow::ensure!(
            hidden_dim == self.heads * self.head_dim,
            "Gemma 4 vision attention width does not match its head layout"
        );

        let q = self.q_norm.forward(&self.q.forward(input)?.reshape((
            batch,
            seq_len,
            self.heads,
            self.head_dim,
        ))?)?;
        let k = self.k_norm.forward(&self.k.forward(input)?.reshape((
            batch,
            seq_len,
            self.heads,
            self.head_dim,
        ))?)?;
        let v = self.v_norm.forward(&self.v.forward(input)?.reshape((
            batch,
            seq_len,
            self.heads,
            self.head_dim,
        ))?)?;
        let (cos, sin) = build_gemma4_vision_rope(
            grid_width,
            grid_height,
            self.head_dim,
            GEMMA4_ROPE_THETA,
            q.dtype(),
            q.device(),
        )?;
        let q = apply_gemma4_vision_rope(&q, &cos, &sin)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = apply_gemma4_vision_rope(&k, &cos, &sin)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Gemma 4 vision attention uses a scale of 1.0 after Q/K RMSNorm.
        let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let probabilities =
            candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(v.dtype())?;
        let output = probabilities
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, hidden_dim))?;
        self.output.forward(&output)
    }
}

struct Gemma4VisionMlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Gemma4VisionMlp {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::Activation::NewGelu.forward(&self.gate.forward(input)?)?;
        self.down.forward(&(gate * self.up.forward(input)?)?)
    }
}

struct Gemma4VisionBlock {
    input_norm: RmsNorm,
    attention: Gemma4VisionAttention,
    attention_post_norm: RmsNorm,
    pre_ffn_norm: RmsNorm,
    mlp: Gemma4VisionMlp,
    ffn_post_norm: RmsNorm,
}

impl Gemma4VisionBlock {
    fn forward(&self, input: &Tensor, grid_width: usize, grid_height: usize) -> Result<Tensor> {
        let attention =
            self.attention
                .forward(&self.input_norm.forward(input)?, grid_width, grid_height)?;
        let hidden = (input + self.attention_post_norm.forward(&attention)?)?;
        let mlp = self.mlp.forward(&self.pre_ffn_norm.forward(&hidden)?)?;
        (hidden + self.ffn_post_norm.forward(&mlp)?).map_err(Into::into)
    }
}

/// Gemma 4's dynamic-resolution vision tower and language projector.
pub struct Gemma4VisionEncoder {
    patch_embedding: Tensor,
    position_embedding: Tensor,
    blocks: Vec<Gemma4VisionBlock>,
    std_bias: Tensor,
    std_scale: Tensor,
    projector_norm: RmsNorm,
    decoder_projection: Linear,
    patch_size: usize,
    hidden_dim: usize,
    decoder_dim: usize,
    position_count: usize,
    device: Device,
}

impl Gemma4VisionEncoder {
    /// Load a Gemma 4 vision projector from a GGUF file onto `device`.
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let mut file = File::open(path)
            .with_context(|| format!("could not open vision projector '{}'", path.display()))?;
        validate_gguf_header_budget(&mut file)
            .with_context(|| format!("could not validate vision projector '{}'", path.display()))?;
        let content = gguf_file::Content::read(&mut file)
            .with_context(|| format!("could not parse vision projector '{}'", path.display()))?;

        let architecture = metadata_string(&content.metadata, "general.architecture")?;
        anyhow::ensure!(
            architecture == "clip",
            "vision projector architecture is '{architecture}', not 'clip'"
        );
        let projector_type = metadata_string(&content.metadata, "clip.vision.projector_type")?;
        anyhow::ensure!(
            projector_type == GEMMA4_PROJECTOR_TYPE,
            "vision projector type '{projector_type}' is not supported as Gemma 4"
        );
        let patch_size = metadata_usize(&content.metadata, "clip.vision.patch_size")?;
        let hidden_dim = metadata_usize(&content.metadata, "clip.vision.embedding_length")?;
        let decoder_dim = metadata_usize(&content.metadata, "clip.vision.projection_dim")?;
        let feed_forward_dim =
            metadata_usize(&content.metadata, "clip.vision.feed_forward_length")?;
        let block_count = metadata_usize(&content.metadata, "clip.vision.block_count")?;
        let heads = metadata_usize(&content.metadata, "clip.vision.attention.head_count")?;
        let norm_eps = metadata_f64(
            &content.metadata,
            "clip.vision.attention.layer_norm_epsilon",
        )?;

        anyhow::ensure!(patch_size > 0, "Gemma 4 vision patch size must be positive");
        let patch_area = patch_size
            .checked_mul(patch_size)
            .context("Gemma 4 vision patch area overflow")?;
        let patch_input_dim = patch_area
            .checked_mul(3)
            .context("Gemma 4 vision patch input width overflow")?;
        let max_image_pixels = GEMMA4_MAX_IMAGE_TOKENS
            .checked_mul(GEMMA4_POOLING_KERNEL.pow(2))
            .and_then(|patches| patches.checked_mul(patch_area))
            .context("Gemma 4 vision image pixel budget overflow")?;
        anyhow::ensure!(
            max_image_pixels <= VISION_MAX_IMAGE_PIXELS,
            "Gemma 4 vision image pixel budget exceeds {VISION_MAX_IMAGE_PIXELS}"
        );
        anyhow::ensure!(
            (1..=VISION_MAX_HIDDEN_DIM).contains(&hidden_dim),
            "Gemma 4 vision hidden width must be in 1..={VISION_MAX_HIDDEN_DIM}"
        );
        anyhow::ensure!(
            (1..=VISION_MAX_DECODER_DIM).contains(&decoder_dim),
            "Gemma 4 vision decoder width must be in 1..={VISION_MAX_DECODER_DIM}"
        );
        validate_layer_count(block_count, "Gemma 4 vision block count")?;
        anyhow::ensure!(heads > 0, "Gemma 4 vision head count must be positive");
        anyhow::ensure!(
            norm_eps.is_finite() && (norm_eps as f32).is_finite() && (norm_eps as f32) > 0.0,
            "Gemma 4 vision layer norm epsilon must be finite, positive, and representable in F32"
        );
        anyhow::ensure!(
            hidden_dim % heads == 0
                && (4..=VISION_MAX_HEAD_DIM).contains(&(hidden_dim / heads))
                && (hidden_dim / heads) % 4 == 0,
            "Gemma 4 vision head width cannot use two-axis RoPE"
        );
        anyhow::ensure!(
            feed_forward_dim == 4_304,
            "unsupported Gemma 4 vision feed-forward width {feed_forward_dim}"
        );

        tracing::info!(
            path = %path.display(),
            patch_size,
            hidden_dim,
            decoder_dim,
            block_count,
            max_image_tokens = GEMMA4_MAX_IMAGE_TOKENS,
            "Loading pure Candle Gemma 4 vision projector"
        );
        let mut tensors = HashMap::with_capacity(content.tensor_infos.len());
        for name in content.tensor_infos.keys() {
            let tensor = content
                .tensor(&mut file, name, device)
                .with_context(|| format!("could not load vision tensor {name}"))?;
            tensors.insert(name.clone(), Arc::new(tensor));
        }

        let tensor = |name: &str| -> Result<Tensor> {
            tensors
                .get(name)
                .with_context(|| format!("vision projector is missing {name}"))?
                .dequantize(device)
                .map_err(Into::into)
        };
        let norm = |name: &str| -> Result<RmsNorm> { Ok(RmsNorm::new(tensor(name)?, norm_eps)) };
        let position_embedding = tensor("v.position_embd.weight")?;
        anyhow::ensure!(
            position_embedding.dims().len() == 3
                && position_embedding.dim(0)? == 2
                && position_embedding.dim(2)? == hidden_dim,
            "Gemma 4 position embedding has the wrong shape"
        );
        let position_count = position_embedding.dim(1)?;
        let patch_embedding =
            tensor("v.patch_embd.weight")?.reshape((hidden_dim, patch_input_dim))?;
        let std_bias = tensor("v.std_bias")?.to_dtype(DType::F32)?;
        let std_scale = tensor("v.std_scale")?.to_dtype(DType::F32)?;
        let projector_norm = RmsNorm::new(
            Tensor::ones(hidden_dim, position_embedding.dtype(), device)?,
            norm_eps,
        );

        let mut blocks = Vec::with_capacity(block_count);
        for index in 0..block_count {
            let prefix = format!("v.blk.{index}");
            blocks.push(Gemma4VisionBlock {
                input_norm: norm(&format!("{prefix}.ln1.weight"))?,
                attention: Gemma4VisionAttention {
                    q: Linear::load(&tensors, &format!("{prefix}.attn_q.weight"), None, device)?,
                    k: Linear::load(&tensors, &format!("{prefix}.attn_k.weight"), None, device)?,
                    v: Linear::load(&tensors, &format!("{prefix}.attn_v.weight"), None, device)?,
                    output: Linear::load(
                        &tensors,
                        &format!("{prefix}.attn_out.weight"),
                        None,
                        device,
                    )?,
                    q_norm: norm(&format!("{prefix}.attn_q_norm.weight"))?,
                    k_norm: norm(&format!("{prefix}.attn_k_norm.weight"))?,
                    v_norm: RmsNorm::new(
                        Tensor::ones(hidden_dim / heads, position_embedding.dtype(), device)?,
                        norm_eps,
                    ),
                    heads,
                    head_dim: hidden_dim / heads,
                },
                attention_post_norm: norm(&format!("{prefix}.attn_post_norm.weight"))?,
                pre_ffn_norm: norm(&format!("{prefix}.ln2.weight"))?,
                mlp: Gemma4VisionMlp {
                    gate: Linear::load(
                        &tensors,
                        &format!("{prefix}.ffn_gate.weight"),
                        None,
                        device,
                    )?,
                    up: Linear::load(&tensors, &format!("{prefix}.ffn_up.weight"), None, device)?,
                    down: Linear::load(
                        &tensors,
                        &format!("{prefix}.ffn_down.weight"),
                        None,
                        device,
                    )?,
                },
                ffn_post_norm: norm(&format!("{prefix}.ffn_post_norm.weight"))?,
            });
        }

        validate_linear_shape(
            &tensors,
            "mm.input_projection.weight",
            decoder_dim,
            hidden_dim,
        )?;
        Ok(Self {
            patch_embedding,
            position_embedding,
            blocks,
            std_bias,
            std_scale,
            projector_norm,
            decoder_projection: Linear::load(&tensors, "mm.input_projection.weight", None, device)?,
            patch_size,
            hidden_dim,
            decoder_dim,
            position_count,
            device: device.clone(),
        })
    }

    /// Return the language-model embedding width produced by the projector.
    pub fn decoder_dim(&self) -> usize {
        self.decoder_dim
    }

    /// Return the maximum number of language-model tokens emitted for an image.
    pub fn max_image_tokens(&self) -> usize {
        GEMMA4_MAX_IMAGE_TOKENS
    }

    /// Return the aspect-ratio-preserving image size used by the Gemma 4 processor.
    pub fn target_image_size(&self, width: usize, height: usize) -> Result<(usize, usize)> {
        anyhow::ensure!(width > 0 && height > 0, "image has zero width or height");
        let max_patches = GEMMA4_MAX_IMAGE_TOKENS * GEMMA4_POOLING_KERNEL.pow(2);
        let target_pixels = max_patches
            .checked_mul(self.patch_size.pow(2))
            .context("Gemma 4 image pixel budget overflow")?;
        let source_pixels = width
            .checked_mul(height)
            .context("Gemma 4 source image dimensions overflow")?;
        let factor = (target_pixels as f64 / source_pixels as f64).sqrt();
        let side_multiple = GEMMA4_POOLING_KERNEL * self.patch_size;
        let mut target_height =
            ((factor * height as f64) / side_multiple as f64).floor() as usize * side_multiple;
        let mut target_width =
            ((factor * width as f64) / side_multiple as f64).floor() as usize * side_multiple;
        anyhow::ensure!(
            target_height > 0 || target_width > 0,
            "image aspect ratio cannot fit the Gemma 4 patch grid"
        );
        let max_side = GEMMA4_MAX_IMAGE_TOKENS * side_multiple;
        if target_height == 0 {
            target_height = side_multiple;
            target_width = (width / height).saturating_mul(side_multiple).min(max_side);
        } else if target_width == 0 {
            target_width = side_multiple;
            target_height = (height / width).saturating_mul(side_multiple).min(max_side);
        }
        anyhow::ensure!(
            target_width > 0 && target_height > 0,
            "image aspect ratio cannot fit the Gemma 4 patch grid"
        );
        let resized_pixels = target_width
            .checked_mul(target_height)
            .context("Gemma 4 resized image dimensions overflow")?;
        anyhow::ensure!(
            resized_pixels <= target_pixels,
            "Gemma 4 resized image exceeds its patch budget"
        );
        Ok((target_width, target_height))
    }

    /// Return the token count after resizing an image to the supported grid.
    pub fn image_tokens_for_size(&self, width: usize, height: usize) -> Result<usize> {
        let (width, height) = self.target_image_size(width, height)?;
        Ok((width / self.patch_size / GEMMA4_POOLING_KERNEL)
            * (height / self.patch_size / GEMMA4_POOLING_KERNEL))
    }

    /// Encode one resized RGB image in CHW order with values in `[0, 1]`.
    pub fn encode(&self, image: &Tensor) -> Result<Tensor> {
        anyhow::ensure!(
            image.device().same_device(&self.device),
            "vision image and projector are on different devices"
        );
        let (channels, height, width) = image.dims3()?;
        anyhow::ensure!(
            channels == 3,
            "Gemma 4 vision image must have three channels"
        );
        let side_multiple = self.patch_size * GEMMA4_POOLING_KERNEL;
        anyhow::ensure!(
            width % side_multiple == 0 && height % side_multiple == 0,
            "Gemma 4 vision image sides must be divisible by {side_multiple}"
        );
        let grid_width = width / self.patch_size;
        let grid_height = height / self.patch_size;
        let image_tokens =
            (grid_width / GEMMA4_POOLING_KERNEL) * (grid_height / GEMMA4_POOLING_KERNEL);
        anyhow::ensure!(
            image_tokens <= GEMMA4_MAX_IMAGE_TOKENS,
            "Gemma 4 image emits {image_tokens} tokens; the limit is {GEMMA4_MAX_IMAGE_TOKENS}"
        );
        anyhow::ensure!(
            grid_width <= self.position_count && grid_height <= self.position_count,
            "Gemma 4 image grid exceeds its learned position table"
        );

        // GGUF stores the source HWC patch projection as a convolution weight,
        // so its flattened input uses channel-first patch order.
        let patches = image
            .to_dtype(self.position_embedding.dtype())?
            .unfold(1, self.patch_size, self.patch_size)?
            .unfold(2, self.patch_size, self.patch_size)?
            .permute((1, 2, 0, 3, 4))?
            .contiguous()?
            .reshape((
                1,
                grid_width * grid_height,
                3 * self.patch_size * self.patch_size,
            ))?;
        let patches = ((patches * 2.0)? - 1.0)?;
        let mut hidden = patches
            .reshape((
                grid_width * grid_height,
                3 * self.patch_size * self.patch_size,
            ))?
            .matmul(&self.patch_embedding.t()?)?
            .reshape((1, grid_width * grid_height, self.hidden_dim))?;

        let x_positions = (0..grid_width * grid_height)
            .map(|index| (index % grid_width) as u32)
            .collect::<Vec<_>>();
        let y_positions = (0..grid_width * grid_height)
            .map(|index| (index / grid_width) as u32)
            .collect::<Vec<_>>();
        let x_positions = Tensor::from_vec(x_positions, grid_width * grid_height, &self.device)?;
        let y_positions = Tensor::from_vec(y_positions, grid_width * grid_height, &self.device)?;
        let x_embedding = self
            .position_embedding
            .i(0)?
            .index_select(&x_positions, 0)?;
        let y_embedding = self
            .position_embedding
            .i(1)?
            .index_select(&y_positions, 0)?;
        hidden = hidden.broadcast_add(&(x_embedding + y_embedding)?.unsqueeze(0)?)?;

        for block in &self.blocks {
            hidden = block.forward(&hidden, grid_width, grid_height)?;
        }
        hidden = spatial_average_pool(
            &hidden,
            grid_width,
            grid_height,
            self.hidden_dim,
            GEMMA4_POOLING_KERNEL,
        )?;
        hidden = (hidden.to_dtype(DType::F32)? * (self.hidden_dim as f64).sqrt())?;
        hidden = hidden.broadcast_sub(&self.std_bias)?;
        hidden = hidden.broadcast_mul(&self.std_scale)?;
        hidden = hidden.to_dtype(self.position_embedding.dtype())?;
        hidden = self.projector_norm.forward(&hidden)?;
        let projected = self.decoder_projection.forward(&hidden)?;
        anyhow::ensure!(
            projected.dims3()? == (1, image_tokens, self.decoder_dim),
            "Gemma 4 vision projector returned the wrong decoder embedding shape"
        );
        projected.squeeze(0).map_err(Into::into)
    }
}

fn spatial_average_pool(
    input: &Tensor,
    grid_width: usize,
    grid_height: usize,
    hidden_dim: usize,
    factor: usize,
) -> Result<Tensor> {
    let batch = input.dim(0)?;
    let output_width = grid_width / factor;
    let output_height = grid_height / factor;
    input
        .reshape((
            batch,
            output_height,
            factor,
            output_width,
            factor,
            hidden_dim,
        ))?
        .permute((0, 1, 3, 2, 4, 5))?
        .contiguous()?
        .reshape((
            batch,
            output_height * output_width,
            factor * factor,
            hidden_dim,
        ))?
        .mean(2)
        .map_err(Into::into)
}

fn build_gemma4_vision_rope(
    grid_width: usize,
    grid_height: usize,
    head_dim: usize,
    theta: f64,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    anyhow::ensure!(
        head_dim.is_multiple_of(4),
        "Gemma 4 vision head width must divide by four"
    );
    let axis_dim = head_dim / 2;
    let frequencies = (0..axis_dim)
        .step_by(2)
        .map(|index| 1.0_f32 / theta.powf(index as f64 / axis_dim as f64) as f32)
        .collect::<Vec<_>>();
    let mut cos = Vec::with_capacity(grid_width * grid_height * head_dim);
    let mut sin = Vec::with_capacity(cos.capacity());
    for y in 0..grid_height {
        for x in 0..grid_width {
            for position in [x, y] {
                let axis_cos = frequencies
                    .iter()
                    .map(|frequency| (position as f32 * frequency).cos())
                    .collect::<Vec<_>>();
                let axis_sin = frequencies
                    .iter()
                    .map(|frequency| (position as f32 * frequency).sin())
                    .collect::<Vec<_>>();
                cos.extend_from_slice(&axis_cos);
                cos.extend_from_slice(&axis_cos);
                sin.extend_from_slice(&axis_sin);
                sin.extend_from_slice(&axis_sin);
            }
        }
    }
    let shape = (1, grid_width * grid_height, 1, head_dim);
    Ok((
        Tensor::from_vec(cos, shape, device)?.to_dtype(dtype)?,
        Tensor::from_vec(sin, shape, device)?.to_dtype(dtype)?,
    ))
}

fn apply_gemma4_vision_rope(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let head_dim = input.dim(D::Minus1)?;
    let axis_dim = head_dim / 2;
    let half_axis = axis_dim / 2;
    let mut rotated_axes = Vec::with_capacity(2);
    for axis in 0..2 {
        let values = input.narrow(D::Minus1, axis * axis_dim, axis_dim)?;
        let first = values.narrow(D::Minus1, 0, half_axis)?;
        let second = values.narrow(D::Minus1, half_axis, half_axis)?;
        rotated_axes.push(Tensor::cat(&[&second.neg()?, &first], D::Minus1)?);
    }
    let rotated_refs = rotated_axes.iter().collect::<Vec<_>>();
    let rotated = Tensor::cat(&rotated_refs, D::Minus1)?;
    Ok((input.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?)?)
}

fn validate_linear_shape(
    tensors: &HashMap<String, Arc<QTensor>>,
    name: &str,
    output: usize,
    input: usize,
) -> Result<()> {
    let shape = tensors
        .get(name)
        .with_context(|| format!("vision projector is missing {name}"))?
        .shape()
        .dims();
    anyhow::ensure!(
        shape == [output, input],
        "vision tensor {name} has shape {shape:?}, expected [{output}, {input}]"
    );
    Ok(())
}

fn pixel_merge(
    input: &Tensor,
    patches_per_side: usize,
    hidden_dim: usize,
    factor: usize,
) -> Result<Tensor> {
    let batch = input.dim(0)?;
    anyhow::ensure!(
        factor > 0 && patches_per_side.is_multiple_of(factor),
        "vision patch grid is not divisible by its merge factor"
    );
    let merged_side = patches_per_side / factor;
    let first_width = hidden_dim
        .checked_mul(factor)
        .context("vision merge width overflow")?;
    let merged_width = first_width
        .checked_mul(factor)
        .context("vision merge width overflow")?;
    let merged_count = merged_side
        .checked_mul(merged_side)
        .context("vision merged patch count overflow")?;
    input
        .reshape((batch, patches_per_side, patches_per_side, hidden_dim))?
        .reshape((batch, patches_per_side, merged_side, first_width))?
        .permute((0, 2, 1, 3))?
        .contiguous()?
        .reshape((batch, merged_side, merged_side, merged_width))?
        .permute((0, 2, 1, 3))?
        .contiguous()?
        .reshape((batch, merged_count, merged_width))
        .map_err(Into::into)
}

fn build_vision_rope(
    patches_per_side: usize,
    head_dim: usize,
    theta: f64,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    anyhow::ensure!(
        (4..=VISION_MAX_HEAD_DIM).contains(&head_dim) && head_dim.is_multiple_of(4),
        "invalid vision RoPE head width"
    );
    let patch_count = patches_per_side
        .checked_mul(patches_per_side)
        .context("vision RoPE patch count overflow")?;
    anyhow::ensure!(
        (1..=LLAMA4_MAX_PATCHES).contains(&patch_count),
        "vision RoPE patch count must be in 1..={LLAMA4_MAX_PATCHES}"
    );
    let positions = patch_count
        .checked_add(1)
        .context("vision RoPE position count overflow")?;
    let elements = positions
        .checked_mul(head_dim / 2)
        .context("vision RoPE table size overflow")?;
    let axis_dim = head_dim / 2;
    let frequencies = (0..axis_dim)
        .step_by(2)
        .map(|index| 1.0_f32 / theta.powf(index as f64 / axis_dim as f64) as f32)
        .collect::<Vec<_>>();
    let mut cos = Vec::new();
    cos.try_reserve_exact(elements)
        .context("could not reserve vision RoPE cosine table")?;
    let mut sin = Vec::new();
    sin.try_reserve_exact(elements)
        .context("could not reserve vision RoPE sine table")?;
    for row in 0..patches_per_side {
        for column in 0..patches_per_side {
            for position in [column + 1, row + 1] {
                for frequency in &frequencies {
                    let angle = position as f32 * frequency;
                    cos.push(angle.cos());
                    sin.push(angle.sin());
                }
            }
        }
    }
    cos.extend(std::iter::repeat_n(1.0, head_dim / 2));
    sin.extend(std::iter::repeat_n(0.0, head_dim / 2));
    Ok((
        Tensor::from_vec(cos, (1, positions, 1, head_dim / 2), device)?,
        Tensor::from_vec(sin, (1, positions, 1, head_dim / 2), device)?,
    ))
}

fn apply_vision_rope(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = input.dtype();
    let input = input.to_dtype(DType::F32)?;
    let (batch, seq_len, heads, head_dim) = input.dims4()?;
    let pairs = input.reshape((batch, seq_len, heads, head_dim / 2, 2))?;
    let even = pairs.i((.., .., .., .., 0))?;
    let odd = pairs.i((.., .., .., .., 1))?;
    let rotated_even = (even.broadcast_mul(cos)? - odd.broadcast_mul(sin)?)?;
    let rotated_odd = (even.broadcast_mul(sin)? + odd.broadcast_mul(cos)?)?;
    Tensor::stack(&[&rotated_even, &rotated_odd], D::Minus1)?
        .reshape((batch, seq_len, heads, head_dim))?
        .to_dtype(dtype)
        .map_err(Into::into)
}

fn metadata_string(metadata: &HashMap<String, gguf_file::Value>, key: &str) -> Result<String> {
    metadata
        .get(key)
        .with_context(|| format!("vision projector metadata is missing {key}"))?
        .to_string()
        .cloned()
        .map_err(Into::into)
}

fn metadata_usize(metadata: &HashMap<String, gguf_file::Value>, key: &str) -> Result<usize> {
    let value = metadata
        .get(key)
        .with_context(|| format!("vision projector metadata is missing {key}"))?;
    let value = match value {
        gguf_file::Value::U8(value) => *value as u64,
        gguf_file::Value::U16(value) => *value as u64,
        gguf_file::Value::U32(value) => *value as u64,
        gguf_file::Value::U64(value) => *value,
        gguf_file::Value::I8(value) if *value >= 0 => *value as u64,
        gguf_file::Value::I16(value) if *value >= 0 => *value as u64,
        gguf_file::Value::I32(value) if *value >= 0 => *value as u64,
        gguf_file::Value::I64(value) if *value >= 0 => *value as u64,
        _ => anyhow::bail!("vision projector metadata {key} is not a positive integer"),
    };
    usize::try_from(value).context("vision projector metadata exceeds this platform")
}

fn metadata_f64(metadata: &HashMap<String, gguf_file::Value>, key: &str) -> Result<f64> {
    match metadata
        .get(key)
        .with_context(|| format!("vision projector metadata is missing {key}"))?
    {
        gguf_file::Value::F32(value) => Ok(f64::from(*value)),
        gguf_file::Value::F64(value) => Ok(*value),
        _ => anyhow::bail!("vision projector metadata {key} is not a float"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama4_metadata_fixture(key: &str, value: gguf_file::Value) -> tempfile::NamedTempFile {
        let mut metadata = vec![
            (
                "general.architecture",
                gguf_file::Value::String("clip".into()),
            ),
            (
                "clip.projector_type",
                gguf_file::Value::String("llama4".into()),
            ),
            ("clip.vision.image_size", gguf_file::Value::U64(4)),
            ("clip.vision.patch_size", gguf_file::Value::U64(1)),
            ("clip.vision.embedding_length", gguf_file::Value::U64(8)),
            ("clip.vision.projection_dim", gguf_file::Value::U64(8)),
            ("clip.vision.feed_forward_length", gguf_file::Value::U64(32)),
            ("clip.vision.block_count", gguf_file::Value::U64(1)),
            ("clip.vision.attention.head_count", gguf_file::Value::U64(1)),
            (
                "clip.vision.projector.scale_factor",
                gguf_file::Value::U64(2),
            ),
            (
                "clip.vision.attention.layer_norm_epsilon",
                gguf_file::Value::F64(1e-5),
            ),
        ];
        metadata
            .iter_mut()
            .find(|(name, _)| *name == key)
            .unwrap()
            .1 = value;
        let metadata = metadata
            .iter()
            .map(|(name, value)| (*name, value))
            .collect::<Vec<_>>();
        let mut file = tempfile::NamedTempFile::new().unwrap();
        gguf_file::write(file.as_file_mut(), &metadata, &[]).unwrap();
        file
    }

    #[test]
    fn llama4_loader_rejects_invalid_geometry_before_loading_weights() {
        for (key, value, message) in [
            (
                "clip.vision.attention.head_count",
                gguf_file::Value::U64(0),
                "head count must be positive",
            ),
            (
                "clip.vision.image_size",
                gguf_file::Value::U64(u64::MAX),
                "image pixel count overflow",
            ),
            (
                "clip.vision.image_size",
                gguf_file::Value::U64(8_193),
                "image pixel count exceeds",
            ),
            (
                "clip.vision.image_size",
                gguf_file::Value::U64(65),
                "patch count exceeds",
            ),
            (
                "clip.vision.embedding_length",
                gguf_file::Value::U64(0),
                "hidden width must be",
            ),
            (
                "clip.vision.projection_dim",
                gguf_file::Value::U64(0),
                "decoder width must be",
            ),
            (
                "clip.vision.block_count",
                gguf_file::Value::U64(u64::MAX),
                "block count",
            ),
            (
                "clip.vision.projector.scale_factor",
                gguf_file::Value::U64(0),
                "merge factor",
            ),
            (
                "clip.vision.feed_forward_length",
                gguf_file::Value::U64(0),
                "feed-forward width",
            ),
            (
                "clip.vision.attention.layer_norm_epsilon",
                gguf_file::Value::F64(f64::NAN),
                "epsilon must be",
            ),
            (
                "clip.vision.attention.layer_norm_epsilon",
                gguf_file::Value::F64(f64::from_bits(1)),
                "epsilon must be",
            ),
            (
                "clip.vision.attention.layer_norm_epsilon",
                gguf_file::Value::F64(f64::MAX),
                "epsilon must be",
            ),
        ] {
            let fixture = llama4_metadata_fixture(key, value);
            let error = match Llama4VisionEncoder::load(fixture.path(), &Device::Cpu) {
                Ok(_) => panic!("accepted invalid {key}"),
                Err(error) => error.to_string(),
            };
            assert!(error.contains(message), "{key}: {error}");
        }
    }

    fn gemma4_metadata_fixture(key: &str, value: gguf_file::Value) -> tempfile::NamedTempFile {
        use candle_core::quantized::GgmlDType;

        let mut metadata = vec![
            (
                "general.architecture",
                gguf_file::Value::String("clip".into()),
            ),
            (
                "clip.vision.projector_type",
                gguf_file::Value::String("gemma4v".into()),
            ),
            ("clip.vision.patch_size", gguf_file::Value::U64(1)),
            ("clip.vision.embedding_length", gguf_file::Value::U64(4)),
            ("clip.vision.projection_dim", gguf_file::Value::U64(4)),
            (
                "clip.vision.feed_forward_length",
                gguf_file::Value::U64(4304),
            ),
            ("clip.vision.block_count", gguf_file::Value::U64(1)),
            ("clip.vision.attention.head_count", gguf_file::Value::U64(1)),
            (
                "clip.vision.attention.layer_norm_epsilon",
                gguf_file::Value::F64(1e-6),
            ),
        ];
        metadata
            .iter_mut()
            .find(|(name, _)| *name == key)
            .unwrap()
            .1 = value;
        // These four small tensors reached the old block-count allocation.
        // No layer weights are needed to prove metadata fails before that point.
        let tensors = [
            (
                "v.position_embd.weight",
                Tensor::zeros((2, 1, 4), DType::F32, &Device::Cpu).unwrap(),
            ),
            (
                "v.patch_embd.weight",
                Tensor::zeros((4, 3), DType::F32, &Device::Cpu).unwrap(),
            ),
            (
                "v.std_bias",
                Tensor::zeros(4, DType::F32, &Device::Cpu).unwrap(),
            ),
            (
                "v.std_scale",
                Tensor::ones(4, DType::F32, &Device::Cpu).unwrap(),
            ),
        ]
        .into_iter()
        .map(|(name, tensor)| (name, QTensor::quantize(&tensor, GgmlDType::F32).unwrap()))
        .collect::<Vec<_>>();
        let metadata = metadata
            .iter()
            .map(|(name, value)| (*name, value))
            .collect::<Vec<_>>();
        let tensors = tensors
            .iter()
            .map(|(name, tensor)| (*name, tensor))
            .collect::<Vec<_>>();
        let mut file = tempfile::NamedTempFile::new().unwrap();
        gguf_file::write(file.as_file_mut(), &metadata, &tensors).unwrap();
        file
    }

    #[test]
    fn gemma4_loader_rejects_invalid_metadata_without_panicking() {
        for (key, value, message) in [
            (
                "clip.vision.patch_size",
                gguf_file::Value::U64(0),
                "patch size",
            ),
            (
                "clip.vision.patch_size",
                gguf_file::Value::U64(1 << 32),
                "patch area overflow",
            ),
            (
                "clip.vision.patch_size",
                gguf_file::Value::U64(164),
                "image pixel budget exceeds",
            ),
            (
                "clip.vision.embedding_length",
                gguf_file::Value::U64(0),
                "hidden width",
            ),
            (
                "clip.vision.embedding_length",
                gguf_file::Value::U64(u64::MAX),
                "hidden width",
            ),
            (
                "clip.vision.projection_dim",
                gguf_file::Value::U64(0),
                "decoder width",
            ),
            (
                "clip.vision.projection_dim",
                gguf_file::Value::U64(u64::MAX),
                "decoder width",
            ),
            (
                "clip.vision.block_count",
                gguf_file::Value::U64(u64::MAX),
                "block count",
            ),
            (
                "clip.vision.block_count",
                gguf_file::Value::U64(0),
                "block count",
            ),
            (
                "clip.vision.attention.head_count",
                gguf_file::Value::U64(0),
                "head count",
            ),
            (
                "clip.vision.embedding_length",
                gguf_file::Value::U64(2048),
                "head width",
            ),
            (
                "clip.vision.attention.layer_norm_epsilon",
                gguf_file::Value::F64(f64::NAN),
                "epsilon",
            ),
            (
                "clip.vision.attention.layer_norm_epsilon",
                gguf_file::Value::F64(f64::INFINITY),
                "epsilon",
            ),
            (
                "clip.vision.attention.layer_norm_epsilon",
                gguf_file::Value::F64(0.),
                "epsilon",
            ),
            (
                "clip.vision.attention.layer_norm_epsilon",
                gguf_file::Value::F64(f64::from_bits(1)),
                "epsilon",
            ),
            (
                "clip.vision.attention.layer_norm_epsilon",
                gguf_file::Value::F64(f64::MAX),
                "epsilon",
            ),
        ] {
            let fixture = gemma4_metadata_fixture(key, value);
            let error = match Gemma4VisionEncoder::load(fixture.path(), &Device::Cpu) {
                Ok(_) => panic!("accepted invalid {key}"),
                Err(error) => error.to_string(),
            };
            assert!(error.contains(message), "{key}: {error}");
        }
    }

    #[test]
    fn llama4_rope_and_merge_reject_overflow_and_zero_factors() {
        assert!(build_vision_rope(usize::MAX, 8, 10_000.0, &Device::Cpu).is_err());
        assert!(build_vision_rope(65, 8, 10_000.0, &Device::Cpu).is_err());
        assert!(build_vision_rope(2, 0, 10_000.0, &Device::Cpu).is_err());
        let input = Tensor::zeros((1, 4, 8), DType::F32, &Device::Cpu).unwrap();
        assert!(pixel_merge(&input, 2, 8, 0).is_err());
        assert!(pixel_merge(&input, 2, usize::MAX, 2).is_err());
    }

    #[test]
    fn pixel_merge_preserves_llama4_channel_order() {
        let input = Tensor::arange(0f32, 16f32, &Device::Cpu)
            .unwrap()
            .reshape((1, 4, 4))
            .unwrap();
        let output = pixel_merge(&input, 2, 4, 2).unwrap();
        assert_eq!(output.dims(), &[1, 1, 16]);
        assert_eq!(
            output.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            (0..16).map(|value| value as f32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn two_axis_rope_keeps_class_token_unchanged() {
        let (cos, sin) = build_vision_rope(2, 8, 10_000.0, &Device::Cpu).unwrap();
        let input = Tensor::ones((1, 5, 1, 8), DType::F32, &Device::Cpu).unwrap();
        let output = apply_vision_rope(&input, &cos, &sin).unwrap();
        assert_eq!(
            output.i((0, 4, 0)).unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0; 8]
        );
    }

    #[test]
    fn gemma4_two_axis_rope_keeps_origin_unchanged() {
        let (cos, sin) =
            build_gemma4_vision_rope(2, 2, 8, 100.0, DType::F32, &Device::Cpu).unwrap();
        let input = Tensor::ones((1, 4, 1, 8), DType::F32, &Device::Cpu).unwrap();
        let output = apply_gemma4_vision_rope(&input, &cos, &sin).unwrap();
        assert_eq!(
            output.i((0, 0, 0)).unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0; 8]
        );
        assert_ne!(
            output.i((0, 1, 0)).unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0; 8]
        );
    }

    #[test]
    fn spatial_pool_uses_row_major_three_by_three_blocks() {
        let input = Tensor::arange(0f32, 36f32, &Device::Cpu)
            .unwrap()
            .reshape((1, 36, 1))
            .unwrap();
        let output = spatial_average_pool(&input, 6, 6, 1, 3).unwrap();
        assert_eq!(output.dims(), &[1, 4, 1]);
        assert_eq!(
            output.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![7.0, 10.0, 25.0, 28.0]
        );
    }

    #[test]
    #[ignore = "requires an external Llama 4 mmproj GGUF"]
    fn external_llama4_projector_smoke() {
        let path = std::env::var("DINF_LLAMA4_PROJECTOR")
            .expect("set DINF_LLAMA4_PROJECTOR to mmproj-F16.gguf");
        #[cfg(target_os = "macos")]
        let Some(device) = crate::test_metal_device() else {
            return;
        };
        #[cfg(not(target_os = "macos"))]
        let device = Device::Cpu;
        let encoder = Llama4VisionEncoder::load(Path::new(&path), &device).unwrap();
        let image = Tensor::zeros(
            (3, encoder.image_size(), encoder.image_size()),
            DType::F32,
            &device,
        )
        .unwrap();
        let output = encoder.encode(&image).unwrap();
        device.synchronize().unwrap();
        assert_eq!(
            output.dims(),
            &[encoder.image_tokens(), encoder.decoder_dim()]
        );
        let embeddings = output
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            embeddings.iter().all(|value| value.is_finite()),
            "Llama 4 projector returned non-finite embeddings"
        );
    }

    #[test]
    #[ignore = "requires an external Gemma 4 mmproj GGUF"]
    fn external_gemma4_projector_smoke() {
        let path = std::env::var("DINF_GEMMA4_PROJECTOR")
            .expect("set DINF_GEMMA4_PROJECTOR to the Gemma 4 mmproj GGUF");
        #[cfg(target_os = "macos")]
        let Some(device) = crate::test_metal_device() else {
            return;
        };
        #[cfg(not(target_os = "macos"))]
        let device = Device::Cpu;
        let encoder = Gemma4VisionEncoder::load(Path::new(&path), &device).unwrap();
        let (width, height) = encoder.target_image_size(256, 256).unwrap();
        assert_eq!((width, height), (768, 768));
        let image = Tensor::zeros((3, height, width), DType::F32, &device).unwrap();
        let output = encoder.encode(&image).unwrap();
        device.synchronize().unwrap();
        assert_eq!(output.dims(), &[256, encoder.decoder_dim()]);
        let embeddings = output
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            embeddings.iter().all(|value| value.is_finite()),
            "Gemma 4 projector returned non-finite embeddings"
        );
    }
}
