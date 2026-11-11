//! Qwen3.5/Qwen3.6/Qwen3.8 vision tower for `qwen3vl_merger` mmproj files.
//!
//! The 27B conditional-generation checkpoints use an auxiliary GGUF rather
//! than storing vision weights in the text-model GGUF.  A still image is
//! duplicated across the two temporal Conv3D slices, encoded by the ViT, and
//! merged in 2x2 spatial groups into text-width feature rows. Those rows are
//! consumed through [`super::StageInput::positioned_activations`].

use std::fs::File;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use crate::qmatmul::QMatMul;
use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig};
use image::codecs::gif::GifDecoder;
use image::imageops::FilterType;
use image::{AnimationDecoder, ImageDecoder};

use crate::gguf_limits::validate_gguf_header_budget;

/// One merged Qwen image token covers this many source pixels per axis.
///
/// Keep this synchronized with the caller's image-token reservation: callers
/// should reserve at most 4,096 image tokens per image, so accepting a projector
/// with a smaller effective stride would under-reserve prompt capacity.
const SUPPORTED_IMAGE_TOKEN_STRIDE: usize = 32;
const DEFAULT_MIN_PIXELS: usize = 64 * SUPPORTED_IMAGE_TOKEN_STRIDE.pow(2);
const DEFAULT_MAX_PIXELS: usize = 4096 * SUPPORTED_IMAGE_TOKEN_STRIDE.pow(2);
const MAX_ENCODED_IMAGE_BYTES: usize = 16 * 1024 * 1024;
const MAX_SOURCE_PIXELS: usize = 64 * 1024 * 1024;
const MAX_ASPECT_RATIO: f64 = 200.0;
const PORTABLE_ATTENTION_QUERY_CHUNK: usize = 128;

/// RGB pixels accepted by the vision encoder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionImage {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Interleaved, row-major RGB8 pixels.
    pub pixels: Vec<u8>,
}

impl VisionImage {
    /// Construct an RGB8 image after validating dimensions and byte length.
    ///
    /// Returns an error for zero dimensions, oversized source images, or a
    /// pixel buffer whose length is not `width * height * 3`.
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> Result<Self> {
        anyhow::ensure!(width > 0 && height > 0, "image dimensions must be positive");
        let pixel_count = width
            .checked_mul(height)
            .context("image dimensions overflow")?;
        anyhow::ensure!(
            pixel_count <= MAX_SOURCE_PIXELS,
            "image has {pixel_count} pixels; limit is {MAX_SOURCE_PIXELS}"
        );
        let expected = pixel_count
            .checked_mul(3)
            .context("image RGB byte count overflow")?;
        anyhow::ensure!(
            pixels.len() == expected,
            "RGB image has {} bytes, expected {expected}",
            pixels.len()
        );
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Decode a bounded still GIF, PNG, JPEG, or WebP image.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        anyhow::ensure!(!bytes.is_empty(), "image payload is empty");
        anyhow::ensure!(
            bytes.len() <= MAX_ENCODED_IMAGE_BYTES,
            "encoded image is {} bytes; limit is {MAX_ENCODED_IMAGE_BYTES}",
            bytes.len()
        );
        let format = image::guess_format(bytes).context("could not determine image format")?;
        anyhow::ensure!(
            matches!(
                format,
                image::ImageFormat::Gif
                    | image::ImageFormat::Png
                    | image::ImageFormat::Jpeg
                    | image::ImageFormat::WebP
            ),
            "unsupported image format {format:?}; use GIF, PNG, JPEG, or WebP"
        );
        // Animated formats have ambiguous frame/timestamp semantics in Qwen.
        // Accept only still inputs until the video position contract is wired.
        if format == image::ImageFormat::Png {
            anyhow::ensure!(
                !png_has_animation_chunk(bytes),
                "animated PNG images are not supported"
            );
        }
        if format == image::ImageFormat::WebP {
            anyhow::ensure!(
                !webp_has_animation_chunk(bytes),
                "animated WebP images are not supported"
            );
        }

        if format == image::ImageFormat::Gif {
            return decode_still_gif(bytes);
        }

        let reader = image::ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .context("could not inspect image")?;
        let mut decoder = reader.into_decoder().context("could not inspect image")?;
        let (width, height) = decoder.dimensions();
        let pixel_count = usize::try_from(width)?
            .checked_mul(usize::try_from(height)?)
            .context("image dimensions overflow")?;
        anyhow::ensure!(
            pixel_count <= MAX_SOURCE_PIXELS,
            "image has {pixel_count} pixels; limit is {MAX_SOURCE_PIXELS}"
        );
        let orientation = decoder
            .orientation()
            .context("could not read image orientation")?;
        let mut image =
            image::DynamicImage::from_decoder(decoder).context("could not decode image")?;
        image.apply_orientation(orientation);
        let image = composite_onto_white(image);
        Self::new(
            image.width() as usize,
            image.height() as usize,
            image.into_raw(),
        )
    }
}

fn decode_still_gif(bytes: &[u8]) -> Result<VisionImage> {
    let decoder = GifDecoder::new(Cursor::new(bytes)).context("could not inspect GIF image")?;
    let (width, height) = decoder.dimensions();
    let pixel_count = usize::try_from(width)?
        .checked_mul(usize::try_from(height)?)
        .context("image dimensions overflow")?;
    anyhow::ensure!(
        pixel_count <= MAX_SOURCE_PIXELS,
        "image has {pixel_count} pixels; limit is {MAX_SOURCE_PIXELS}"
    );

    let mut frames = decoder.into_frames();
    let frame = frames
        .next()
        .transpose()
        .context("could not decode GIF image")?
        .context("GIF image contains no frames")?;
    anyhow::ensure!(
        frames
            .next()
            .transpose()
            .context("could not inspect GIF animation")?
            .is_none(),
        "animated GIF images are not supported"
    );
    let image = composite_onto_white(image::DynamicImage::ImageRgba8(frame.into_buffer()));
    VisionImage::new(
        image.width() as usize,
        image.height() as usize,
        image.into_raw(),
    )
}

fn composite_onto_white(image: image::DynamicImage) -> image::RgbImage {
    let rgba = image.into_rgba8();
    let mut rgb = image::RgbImage::new(rgba.width(), rgba.height());
    for (target, source) in rgb.pixels_mut().zip(rgba.pixels()) {
        let alpha = u16::from(source[3]);
        for channel in 0..3 {
            let value = u16::from(source[channel]) * alpha + 255_u16 * (255_u16 - alpha);
            target[channel] = ((value + 127) / 255) as u8;
        }
    }
    rgb
}

fn png_has_animation_chunk(bytes: &[u8]) -> bool {
    if !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return false;
    }
    let mut offset = 8usize;
    while let Some(header_end) = offset.checked_add(8).filter(|end| *end <= bytes.len()) {
        let length = u32::from_be_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        if &bytes[offset + 4..header_end] == b"acTL" {
            return true;
        }
        let Some(next) = header_end
            .checked_add(length)
            .and_then(|end| end.checked_add(4))
            .filter(|end| *end <= bytes.len())
        else {
            break;
        };
        offset = next;
    }
    false
}

fn webp_has_animation_chunk(bytes: &[u8]) -> bool {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WEBP" {
        return false;
    }
    let mut offset = 12usize;
    while let Some(header_end) = offset.checked_add(8).filter(|end| *end <= bytes.len()) {
        let kind = &bytes[offset..offset + 4];
        if kind == b"ANIM" || kind == b"ANMF" {
            return true;
        }
        let length = u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]) as usize;
        let Some(next) = header_end
            .checked_add(length)
            .and_then(|end| end.checked_add(length % 2))
            .filter(|end| *end <= bytes.len())
        else {
            break;
        };
        offset = next;
    }
    false
}

/// Post-preprocessing patch geometry for one image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionGrid {
    /// Number of patch rows before spatial merging.
    pub patch_rows: usize,
    /// Number of patch columns before spatial merging.
    pub patch_cols: usize,
    /// Number of merged rows emitted to the text decoder.
    pub merged_rows: usize,
    /// Number of merged columns emitted to the text decoder.
    pub merged_cols: usize,
}

impl VisionGrid {
    /// Return the number of decoder tokens represented by this grid.
    pub fn image_tokens(self) -> usize {
        self.merged_rows * self.merged_cols
    }
}

/// Projected image features and their M-RoPE geometry.
pub struct VisionEmbedding {
    /// `[image_tokens, text_hidden_dim]`.
    pub embeddings: Tensor,
    /// Image geometry corresponding to the embedding rows.
    pub grid: VisionGrid,
}

struct Linear {
    weight: QMatMul,
    bias: Tensor,
}

impl Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let output = match cpu_w8a32_projection(&self.weight, x)? {
            Some(output) => output,
            None => self.weight.forward(x)?,
        };
        let bias = self.bias.to_dtype(output.dtype())?;
        output.broadcast_add(&bias).map_err(Into::into)
    }
}

// Keep Q8 vision weights packed between calls, but retain F32 activations.
// The CPU reference profile uses the same exact widening. Other paths decline.
fn cpu_w8a32_projection(weight: &QMatMul, x: &Tensor) -> Result<Option<Tensor>> {
    use candle_core::quantized::GgmlDType;
    if !cfg!(all(target_os = "macos", target_arch = "aarch64"))
        || !x.device().is_cpu()
        || x.dtype() != DType::F32
    {
        return Ok(None);
    }
    let QMatMul::QTensor(weight) = weight else {
        return Ok(None);
    };
    if !weight.device().is_cpu() || weight.dtype() != GgmlDType::Q8_0 {
        return Ok(None);
    }
    let [columns, width] = weight.shape().dims() else {
        return Ok(None);
    };
    let rows = match *x.dims() {
        [rows, k] | [1, rows, k] if k == *width && rows > 0 => rows,
        _ => return Ok(None),
    };
    anyhow::ensure!(
        columns
            .checked_mul(*width)
            .and_then(|n| n.checked_mul(4))
            .is_some_and(|n| n <= 128 * 1024 * 1024),
        "vision temporary F32 weight bound"
    );
    for count in [rows.checked_mul(*width), rows.checked_mul(*columns)] {
        anyhow::ensure!(
            count
                .and_then(|n| n.checked_mul(4))
                .is_some_and(|n| n <= 256 * 1024 * 1024),
            "vision F32 projection input/output bound"
        );
    }
    // Q8 scales/codes widen exactly. This temporary drops after the projection.
    let dense = weight.dequantize(x.device())?;
    Ok(Some(QMatMul::Tensor(dense).forward(x)?))
}

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let mean = x.mean_keepdim(D::Minus1)?;
        let centered = x.broadcast_sub(&mean)?;
        let variance = (&centered * &centered)?.mean_keepdim(D::Minus1)?;
        let normalized = centered.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        normalized
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?
            .to_dtype(dtype)
            .map_err(Into::into)
    }
}

struct VisionBlock {
    ln1: LayerNorm,
    qkv: Linear,
    attn_out: Linear,
    ln2: LayerNorm,
    ffn_up: Linear,
    ffn_down: Linear,
}

/// Loaded `clip.projector_type = qwen3vl_merger` vision sidecar.
pub struct Qwen35VisionEncoder {
    patch_embed: Conv2d,
    patch_bias: Tensor,
    position_embeddings: Vec<f32>,
    position_side: usize,
    blocks: Vec<VisionBlock>,
    merger_norm: LayerNorm,
    merger_up: Linear,
    merger_down: Linear,
    device: Device,
    hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,
    patch_size: usize,
    merge_size: usize,
    projection_dim: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
}

impl Qwen35VisionEncoder {
    /// Load and validate a llama.cpp-compatible Qwen3VL merger GGUF.
    pub fn from_gguf(path: &Path, device: &Device) -> Result<Self> {
        let mut file = File::open(path)
            .with_context(|| format!("failed to open vision projector {}", path.display()))?;
        validate_gguf_header_budget(&mut file)
            .with_context(|| format!("failed to validate vision projector {}", path.display()))?;
        let content = gguf_file::Content::read(&mut file)
            .with_context(|| format!("failed to parse vision projector {}", path.display()))?;
        let metadata = &content.metadata;
        anyhow::ensure!(
            metadata_string(metadata, "general.architecture").as_deref() == Some("clip"),
            "vision projector architecture must be 'clip'"
        );
        anyhow::ensure!(
            metadata_string(metadata, "clip.projector_type").as_deref() == Some("qwen3vl_merger"),
            "unsupported vision projector type; expected qwen3vl_merger"
        );
        validate_projector_activation(metadata)?;
        if let Some(flags) = metadata_bool_array(metadata, "clip.vision.is_deepstack_layers") {
            anyhow::ensure!(
                flags.iter().all(|enabled| !enabled),
                "Qwen vision deep-stack projectors are not supported by the input-only 27B path"
            );
        }

        let hidden_dim = metadata_usize(metadata, "clip.vision.embedding_length")?;
        let intermediate_dim = metadata_usize(metadata, "clip.vision.feed_forward_length")?;
        let block_count = metadata_usize(metadata, "clip.vision.block_count")?;
        let num_heads = metadata_usize(metadata, "clip.vision.attention.head_count")?;
        let patch_size = metadata_usize(metadata, "clip.vision.patch_size")?;
        let merge_size = metadata_usize(metadata, "clip.vision.spatial_merge_size")?;
        let projection_dim = metadata_usize(metadata, "clip.vision.projection_dim")?;
        let eps =
            metadata_f64(metadata, "clip.vision.attention.layer_norm_epsilon").unwrap_or(1e-6);
        anyhow::ensure!(
            eps.is_finite() && eps > 0.0,
            "vision layer-norm epsilon must be positive"
        );
        anyhow::ensure!(
            num_heads > 0 && hidden_dim > 0 && hidden_dim % num_heads == 0,
            "invalid vision attention geometry"
        );
        anyhow::ensure!(
            intermediate_dim > 0 && projection_dim > 0,
            "vision projection dimensions must be positive"
        );
        validate_patch_geometry(patch_size, merge_size)?;
        anyhow::ensure!(
            block_count > 0,
            "vision projector has no transformer blocks"
        );
        let head_dim = hidden_dim / num_heads;
        anyhow::ensure!(
            head_dim % 4 == 0,
            "vision head dimension must be divisible by four"
        );

        let patch0 = load_dense(&content, &mut file, "v.patch_embd.weight", device)?;
        let patch1 = load_dense(&content, &mut file, "v.patch_embd.weight.1", device)?;
        anyhow::ensure!(
            patch0.dims4()? == (hidden_dim, 3, patch_size, patch_size),
            "unexpected first temporal patch-kernel shape {:?}",
            patch0.dims()
        );
        anyhow::ensure!(
            patch1.shape() == patch0.shape(),
            "temporal patch kernels differ"
        );
        let activation_dtype = patch0.dtype();
        let patch_embed = Conv2d::new(
            (&patch0 + &patch1)?,
            None,
            Conv2dConfig {
                stride: patch_size,
                ..Default::default()
            },
        );
        let patch_bias = load_dense(&content, &mut file, "v.patch_embd.bias", device)?
            .to_dtype(activation_dtype)?;
        anyhow::ensure!(
            patch_bias.dims1()? == hidden_dim,
            "vision patch bias has the wrong width"
        );

        let position_tensor =
            load_dense(&content, &mut file, "v.position_embd.weight", &Device::Cpu)?;
        let (position_count, position_width) = position_tensor.dims2()?;
        anyhow::ensure!(
            position_count > 0 && position_width == hidden_dim,
            "vision absolute-position width {position_width} does not match hidden width {hidden_dim}"
        );
        let position_side = (position_count as f64).sqrt() as usize;
        anyhow::ensure!(
            position_side * position_side == position_count,
            "vision absolute-position table is not square"
        );
        let position_embeddings = position_tensor.flatten_all()?.to_vec1::<f32>()?;

        let mut blocks = Vec::with_capacity(block_count);
        for index in 0..block_count {
            let prefix = format!("v.blk.{index}");
            blocks.push(VisionBlock {
                ln1: load_layer_norm(
                    &content,
                    &mut file,
                    &format!("{prefix}.ln1"),
                    hidden_dim,
                    eps,
                    device,
                )?,
                qkv: load_linear(
                    &content,
                    &mut file,
                    &format!("{prefix}.attn_qkv"),
                    hidden_dim * 3,
                    hidden_dim,
                    activation_dtype,
                    device,
                )?,
                attn_out: load_linear(
                    &content,
                    &mut file,
                    &format!("{prefix}.attn_out"),
                    hidden_dim,
                    hidden_dim,
                    activation_dtype,
                    device,
                )?,
                ln2: load_layer_norm(
                    &content,
                    &mut file,
                    &format!("{prefix}.ln2"),
                    hidden_dim,
                    eps,
                    device,
                )?,
                ffn_up: load_linear(
                    &content,
                    &mut file,
                    &format!("{prefix}.ffn_up"),
                    intermediate_dim,
                    hidden_dim,
                    activation_dtype,
                    device,
                )?,
                ffn_down: load_linear(
                    &content,
                    &mut file,
                    &format!("{prefix}.ffn_down"),
                    hidden_dim,
                    intermediate_dim,
                    activation_dtype,
                    device,
                )?,
            });
        }

        let merger_width = hidden_dim
            .checked_mul(merge_size)
            .and_then(|width| width.checked_mul(merge_size))
            .context("vision merger width overflow")?;
        let merger_norm =
            load_layer_norm(&content, &mut file, "v.post_ln", hidden_dim, eps, device)?;
        let merger_up = load_linear(
            &content,
            &mut file,
            "mm.0",
            merger_width,
            merger_width,
            activation_dtype,
            device,
        )?;
        let merger_down = load_linear(
            &content,
            &mut file,
            "mm.2",
            projection_dim,
            merger_width,
            activation_dtype,
            device,
        )?;
        let image_mean =
            metadata_f32_triplet(metadata, "clip.vision.image_mean").unwrap_or([0.5, 0.5, 0.5]);
        let image_std =
            metadata_f32_triplet(metadata, "clip.vision.image_std").unwrap_or([0.5, 0.5, 0.5]);
        anyhow::ensure!(
            image_mean.iter().all(|value| value.is_finite())
                && image_std
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.0),
            "vision image normalization metadata is invalid"
        );

        tracing::info!(
            projector = %path.display(),
            block_count,
            hidden_dim,
            projection_dim,
            patch_size,
            merge_size,
            "Loaded Qwen3VL vision projector"
        );
        Ok(Self {
            patch_embed,
            patch_bias,
            position_embeddings,
            position_side,
            blocks,
            merger_norm,
            merger_up,
            merger_down,
            device: device.clone(),
            hidden_dim,
            num_heads,
            head_dim,
            patch_size,
            merge_size,
            projection_dim,
            image_mean,
            image_std,
        })
    }

    /// Return the decoder hidden width produced by the vision projector.
    pub fn projection_dim(&self) -> usize {
        self.projection_dim
    }

    /// Compute the decoder-token count without decoding pixels or running the
    /// vision tower. Request admission uses this before doing expensive work.
    pub fn image_tokens_for_dimensions(&self, width: usize, height: usize) -> Result<usize> {
        Ok(self.resized_geometry(width, height)?.2.image_tokens())
    }

    /// Encode one still image to decoder-width feature rows.
    pub fn encode(&self, image: &VisionImage) -> Result<VisionEmbedding> {
        let (width, height, grid) = self.resized_geometry(image.width, image.height)?;
        let patch_rows = grid.patch_rows;
        let patch_cols = grid.patch_cols;
        let raw = image::RgbImage::from_raw(
            u32::try_from(image.width)?,
            u32::try_from(image.height)?,
            image.pixels.clone(),
        )
        .context("invalid RGB image storage")?;
        let resized = image::imageops::resize(
            &raw,
            u32::try_from(width)?,
            u32::try_from(height)?,
            FilterType::CatmullRom,
        );

        let mut channels = vec![0f32; 3 * width * height];
        for (index, pixel) in resized.pixels().enumerate() {
            for channel in 0..3 {
                channels[channel * width * height + index] = (f32::from(pixel[channel]) / 255.0
                    - self.image_mean[channel])
                    / self.image_std[channel];
            }
        }
        let input = Tensor::from_vec(channels, (1, 3, height, width), &self.device)?
            .to_dtype(self.patch_embed.weight().dtype())?;
        let mut hidden = self.patch_embed.forward(&input)?;
        let patch_bias = self.patch_bias.to_dtype(hidden.dtype())?;
        hidden = hidden.broadcast_add(&patch_bias.reshape((1, self.hidden_dim, 1, 1))?)?;
        hidden =
            hidden
                .permute((0, 2, 3, 1))?
                .reshape((1, patch_rows * patch_cols, self.hidden_dim))?;

        let order = spatial_merge_order(patch_rows, patch_cols, self.merge_size);
        let order_tensor = Tensor::new(order.as_slice(), &self.device)?;
        hidden = hidden.index_select(&order_tensor, 1)?;
        let absolute = self
            .interpolated_position_embeddings(patch_rows, patch_cols, &order)?
            .to_dtype(hidden.dtype())?;
        hidden = hidden.broadcast_add(&absolute.unsqueeze(0)?)?;
        let (rope_cos, rope_sin) = self.visual_rope(patch_rows, patch_cols, &order)?;

        for block in &self.blocks {
            hidden = block.forward(&hidden, &rope_cos, &rope_sin, self.num_heads, self.head_dim)?;
        }
        hidden = self.merger_norm.forward(&hidden)?;
        let merged_width = self.hidden_dim * self.merge_size * self.merge_size;
        let merged = hidden.reshape((
            patch_rows * patch_cols / (self.merge_size * self.merge_size),
            merged_width,
        ))?;
        let merged = self.merger_up.forward(&merged)?;
        let merged = candle_nn::Activation::Gelu.forward(&merged)?;
        let embeddings = self.merger_down.forward(&merged)?;
        anyhow::ensure!(
            embeddings.dims2()? == (grid.image_tokens(), self.projection_dim),
            "vision projector output shape mismatch"
        );
        Ok(VisionEmbedding { embeddings, grid })
    }

    fn resized_geometry(
        &self,
        source_width: usize,
        source_height: usize,
    ) -> Result<(usize, usize, VisionGrid)> {
        let factor = self
            .patch_size
            .checked_mul(self.merge_size)
            .context("vision resize factor overflow")?;
        let (width, height) = smart_resize_dimensions(
            source_width,
            source_height,
            factor,
            DEFAULT_MIN_PIXELS,
            DEFAULT_MAX_PIXELS,
        )?;
        let patch_rows = height / self.patch_size;
        let patch_cols = width / self.patch_size;
        anyhow::ensure!(
            patch_rows.is_multiple_of(self.merge_size)
                && patch_cols.is_multiple_of(self.merge_size),
            "resized image patch grid is not merge-aligned"
        );
        let grid = VisionGrid {
            patch_rows,
            patch_cols,
            merged_rows: patch_rows / self.merge_size,
            merged_cols: patch_cols / self.merge_size,
        };
        Ok((width, height, grid))
    }

    fn interpolated_position_embeddings(
        &self,
        rows: usize,
        cols: usize,
        order: &[u32],
    ) -> Result<Tensor> {
        let mut output = Vec::with_capacity(rows * cols * self.hidden_dim);
        for &flat in order {
            let flat = flat as usize;
            let row = flat / cols;
            let col = flat % cols;
            let source_y = align_corners_coordinate(row, rows, self.position_side);
            let source_x = align_corners_coordinate(col, cols, self.position_side);
            let y0 = source_y.floor() as usize;
            let x0 = source_x.floor() as usize;
            let y1 = (y0 + 1).min(self.position_side - 1);
            let x1 = (x0 + 1).min(self.position_side - 1);
            let wy = source_y - y0 as f64;
            let wx = source_x - x0 as f64;
            for channel in 0..self.hidden_dim {
                let at = |y: usize, x: usize| {
                    self.position_embeddings
                        [(y * self.position_side + x) * self.hidden_dim + channel]
                        as f64
                };
                let top = at(y0, x0) * (1.0 - wx) + at(y0, x1) * wx;
                let bottom = at(y1, x0) * (1.0 - wx) + at(y1, x1) * wx;
                output.push((top * (1.0 - wy) + bottom * wy) as f32);
            }
        }
        Tensor::from_vec(output, (rows * cols, self.hidden_dim), &self.device).map_err(Into::into)
    }

    fn visual_rope(&self, rows: usize, cols: usize, order: &[u32]) -> Result<(Tensor, Tensor)> {
        let quarter = self.head_dim / 4;
        let inverse = (0..quarter)
            .map(|index| 1.0f64 / 10_000f64.powf((2 * index) as f64 / (self.head_dim / 2) as f64))
            .collect::<Vec<_>>();
        let mut cos = Vec::with_capacity(order.len() * self.head_dim);
        let mut sin = Vec::with_capacity(order.len() * self.head_dim);
        for &flat in order {
            let row = (flat as usize / cols) as f64;
            let col = (flat as usize % cols) as f64;
            let mut angles = Vec::with_capacity(self.head_dim / 2);
            angles.extend(inverse.iter().map(|frequency| row * frequency));
            angles.extend(inverse.iter().map(|frequency| col * frequency));
            for _ in 0..2 {
                cos.extend(angles.iter().map(|angle| angle.cos() as f32));
                sin.extend(angles.iter().map(|angle| angle.sin() as f32));
            }
        }
        Ok((
            Tensor::from_vec(cos, (1, 1, rows * cols, self.head_dim), &self.device)?,
            Tensor::from_vec(sin, (1, 1, rows * cols, self.head_dim), &self.device)?,
        ))
    }
}

impl VisionBlock {
    fn forward(
        &self,
        x: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let (_, tokens, hidden_dim) = x.dims3()?;
        let normalized = self.ln1.forward(x)?;
        let qkv = self.qkv.forward(&normalized)?;
        let query = qkv
            .narrow(2, 0, hidden_dim)?
            .reshape((1, tokens, num_heads, head_dim))?
            .transpose(1, 2)?;
        let key = qkv
            .narrow(2, hidden_dim, hidden_dim)?
            .reshape((1, tokens, num_heads, head_dim))?
            .transpose(1, 2)?;
        let value = qkv
            .narrow(2, hidden_dim * 2, hidden_dim)?
            .reshape((1, tokens, num_heads, head_dim))?
            .transpose(1, 2)?;
        let query = apply_visual_rope(&query, rope_cos, rope_sin)?;
        let key = apply_visual_rope(&key, rope_cos, rope_sin)?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let attended = if let Some(attended) = super::try_fused_f16_sdpa(
            &query.to_dtype(DType::F16)?,
            &key.to_dtype(DType::F16)?,
            &value.to_dtype(DType::F16)?,
            false,
            scale as f32,
        )? {
            attended.to_dtype(x.dtype())?
        } else {
            portable_attention(&query, &key, &value, scale)?
        };
        let attended = attended
            .transpose(1, 2)?
            .contiguous()?
            .reshape((1, tokens, hidden_dim))?;
        let x = (x + self.attn_out.forward(&attended)?)?;
        let mlp = self.ffn_up.forward(&self.ln2.forward(&x)?)?;
        let mlp = candle_nn::Activation::GeluPytorchTanh.forward(&mlp)?;
        (x + self.ffn_down.forward(&mlp)?).map_err(Into::into)
    }
}

fn portable_attention(query: &Tensor, key: &Tensor, value: &Tensor, scale: f64) -> Result<Tensor> {
    let query_tokens = query.dim(2)?;
    anyhow::ensure!(query_tokens > 0, "vision attention requires image tokens");
    let key_transposed = key.transpose(2, 3)?;
    let mut chunks = Vec::with_capacity(query_tokens.div_ceil(PORTABLE_ATTENTION_QUERY_CHUNK));
    for start in (0..query_tokens).step_by(PORTABLE_ATTENTION_QUERY_CHUNK) {
        let len = PORTABLE_ATTENTION_QUERY_CHUNK.min(query_tokens - start);
        let query_chunk = query.narrow(2, start, len)?;
        // Match the reference implementation's fp32 softmax, then return to
        // the activation dtype for the value matmul. Chunking query rows avoids
        // materializing the full O(tokens^2) score matrix on portable backends.
        let weights = (query_chunk.matmul(&key_transposed)? * scale)?.to_dtype(DType::F32)?;
        let probabilities = candle_nn::ops::softmax_last_dim(&weights)?.to_dtype(value.dtype())?;
        chunks.push(probabilities.matmul(value)?);
    }
    let chunk_refs = chunks.iter().collect::<Vec<_>>();
    Tensor::cat(&chunk_refs, 2).map_err(Into::into)
}

fn apply_visual_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let head_dim = x.dim(D::Minus1)?;
    anyhow::ensure!(head_dim % 2 == 0, "visual RoPE head width must be even");
    let half = head_dim / 2;
    let first = x.narrow(D::Minus1, 0, half)?;
    let second = x.narrow(D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[&second.affine(-1.0, 0.0)?, &first], D::Minus1)?;
    let cos = cos.to_dtype(x.dtype())?;
    let sin = sin.to_dtype(x.dtype())?;
    (x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?).map_err(Into::into)
}

/// Official Qwen smart-resize, with a conservative llama.cpp-compatible
/// default maximum chosen by the caller.
pub fn smart_resize_dimensions(
    width: usize,
    height: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    anyhow::ensure!(width > 0 && height > 0, "image dimensions must be positive");
    anyhow::ensure!(factor > 0, "resize factor must be positive");
    anyhow::ensure!(
        min_pixels > 0 && min_pixels <= max_pixels,
        "invalid pixel limits"
    );
    let aspect = width.max(height) as f64 / width.min(height) as f64;
    anyhow::ensure!(
        aspect <= MAX_ASPECT_RATIO,
        "image aspect ratio {aspect:.1} exceeds {MAX_ASPECT_RATIO}"
    );
    let round_factor =
        |value: usize| ((value as f64 / factor as f64).round() as usize).max(1) * factor;
    let mut resized_width = round_factor(width);
    let mut resized_height = round_factor(height);
    let area = resized_width.saturating_mul(resized_height);
    if area > max_pixels {
        let beta = (width as f64 * height as f64 / max_pixels as f64).sqrt();
        resized_width = ((width as f64 / beta / factor as f64).floor() as usize).max(1) * factor;
        resized_height = ((height as f64 / beta / factor as f64).floor() as usize).max(1) * factor;
    } else if area < min_pixels {
        let beta = (min_pixels as f64 / (width as f64 * height as f64)).sqrt();
        resized_width = ((width as f64 * beta / factor as f64).ceil() as usize).max(1) * factor;
        resized_height = ((height as f64 * beta / factor as f64).ceil() as usize).max(1) * factor;
    }
    anyhow::ensure!(
        resized_width.saturating_mul(resized_height) <= max_pixels,
        "could not resize image within the pixel limit"
    );
    Ok((resized_width, resized_height))
}

fn spatial_merge_order(rows: usize, cols: usize, merge: usize) -> Vec<u32> {
    let mut order = Vec::with_capacity(rows * cols);
    for outer_row in 0..rows / merge {
        for outer_col in 0..cols / merge {
            for inner_row in 0..merge {
                for inner_col in 0..merge {
                    order.push(
                        ((outer_row * merge + inner_row) * cols + outer_col * merge + inner_col)
                            as u32,
                    );
                }
            }
        }
    }
    order
}

fn align_corners_coordinate(target: usize, target_size: usize, source_size: usize) -> f64 {
    if target_size <= 1 {
        0.0
    } else {
        target as f64 * (source_size - 1) as f64 / (target_size - 1) as f64
    }
}

fn load_dense(
    content: &gguf_file::Content,
    file: &mut File,
    name: &str,
    device: &Device,
) -> Result<Tensor> {
    content
        .tensor(file, name, device)
        .with_context(|| format!("missing or invalid vision tensor {name}"))?
        .dequantize(device)
        .with_context(|| format!("could not dequantize vision tensor {name}"))
}

fn load_linear(
    content: &gguf_file::Content,
    file: &mut File,
    prefix: &str,
    output_dim: usize,
    input_dim: usize,
    activation_dtype: DType,
    device: &Device,
) -> Result<Linear> {
    let weight_name = format!("{prefix}.weight");
    let weight = content
        .tensor(file, &weight_name, device)
        .with_context(|| format!("missing or invalid vision tensor {weight_name}"))?;
    anyhow::ensure!(
        weight.shape().dims() == [output_dim, input_dim],
        "vision tensor {weight_name} has shape {:?}, expected [{output_dim}, {input_dim}]",
        weight.shape().dims()
    );
    let bias_name = format!("{prefix}.bias");
    let bias = load_dense(content, file, &bias_name, device)?.to_dtype(activation_dtype)?;
    anyhow::ensure!(
        bias.dims1()? == output_dim,
        "vision tensor {bias_name} has the wrong width"
    );
    Ok(Linear {
        weight: QMatMul::from_arc(Arc::new(weight))?,
        bias,
    })
}

fn load_layer_norm(
    content: &gguf_file::Content,
    file: &mut File,
    prefix: &str,
    width: usize,
    eps: f64,
    device: &Device,
) -> Result<LayerNorm> {
    let weight = load_dense(content, file, &format!("{prefix}.weight"), device)?;
    let bias = load_dense(content, file, &format!("{prefix}.bias"), device)?;
    anyhow::ensure!(
        weight.shape() == bias.shape() && weight.dims1()? == width,
        "vision layer-norm {prefix} parameters have the wrong shape"
    );
    Ok(LayerNorm { weight, bias, eps })
}

fn metadata_string(
    metadata: &std::collections::HashMap<String, gguf_file::Value>,
    key: &str,
) -> Option<String> {
    metadata.get(key)?.to_string().ok().cloned()
}

fn metadata_usize(
    metadata: &std::collections::HashMap<String, gguf_file::Value>,
    key: &str,
) -> Result<usize> {
    let value = metadata
        .get(key)
        .with_context(|| format!("vision projector metadata is missing {key}"))?;
    let value = match value {
        gguf_file::Value::U8(value) => *value as u64,
        gguf_file::Value::I8(value) => u64::try_from(*value)?,
        gguf_file::Value::U16(value) => *value as u64,
        gguf_file::Value::I16(value) => u64::try_from(*value)?,
        gguf_file::Value::U32(value) => *value as u64,
        gguf_file::Value::I32(value) => u64::try_from(*value)?,
        gguf_file::Value::U64(value) => *value,
        gguf_file::Value::I64(value) => u64::try_from(*value)?,
        _ => anyhow::bail!("vision projector metadata {key} is not an integer"),
    };
    usize::try_from(value).with_context(|| format!("vision projector metadata {key} is too large"))
}

fn metadata_f64(
    metadata: &std::collections::HashMap<String, gguf_file::Value>,
    key: &str,
) -> Option<f64> {
    match metadata.get(key)? {
        gguf_file::Value::F32(value) => Some(f64::from(*value)),
        gguf_file::Value::F64(value) => Some(*value),
        _ => None,
    }
}

fn validate_projector_activation(
    metadata: &std::collections::HashMap<String, gguf_file::Value>,
) -> Result<()> {
    anyhow::ensure!(
        matches!(
            metadata.get("clip.use_gelu"),
            Some(gguf_file::Value::Bool(true))
        ),
        "Qwen vision projector must declare clip.use_gelu = true"
    );
    Ok(())
}

fn validate_patch_geometry(patch_size: usize, merge_size: usize) -> Result<()> {
    anyhow::ensure!(patch_size > 0, "vision patch size must be positive");
    anyhow::ensure!(merge_size == 2, "only Qwen 2x2 spatial merge is supported");
    anyhow::ensure!(
        patch_size.checked_mul(merge_size) == Some(SUPPORTED_IMAGE_TOKEN_STRIDE),
        "Qwen vision projector effective patch stride must be {SUPPORTED_IMAGE_TOKEN_STRIDE} pixels"
    );
    Ok(())
}

fn metadata_bool_array(
    metadata: &std::collections::HashMap<String, gguf_file::Value>,
    key: &str,
) -> Option<Vec<bool>> {
    let gguf_file::Value::Array(values) = metadata.get(key)? else {
        return None;
    };
    values
        .iter()
        .map(|value| match value {
            gguf_file::Value::Bool(value) => Some(*value),
            _ => None,
        })
        .collect()
}

fn metadata_f32_triplet(
    metadata: &std::collections::HashMap<String, gguf_file::Value>,
    key: &str,
) -> Option<[f32; 3]> {
    let gguf_file::Value::Array(values) = metadata.get(key)? else {
        return None;
    };
    if values.len() != 3 {
        return None;
    }
    let mut output = [0f32; 3];
    for (index, value) in values.iter().enumerate() {
        output[index] = match value {
            gguf_file::Value::F32(value) => *value,
            gguf_file::Value::F64(value) => *value as f32,
            _ => return None,
        };
    }
    Some(output)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::qmatmul::QMatMul;
    use candle_core::quantized::gguf_file;
    use candle_core::{DType, Device, Tensor};
    use image::codecs::gif::GifEncoder;
    use image::{Delay, Frame, Rgba, RgbaImage};

    use super::{
        Linear, VisionImage, apply_visual_rope, composite_onto_white, png_has_animation_chunk,
        portable_attention, smart_resize_dimensions, spatial_merge_order, validate_patch_geometry,
        validate_projector_activation, webp_has_animation_chunk,
    };

    #[test]
    fn projector_activation_requires_explicit_gelu_metadata() {
        let mut metadata = HashMap::new();
        assert!(validate_projector_activation(&metadata).is_err());

        metadata.insert("clip.use_gelu".to_string(), gguf_file::Value::Bool(false));
        assert!(validate_projector_activation(&metadata).is_err());

        metadata.insert(
            "clip.use_gelu".to_string(),
            gguf_file::Value::String("true".to_string()),
        );
        assert!(validate_projector_activation(&metadata).is_err());

        metadata.insert("clip.use_gelu".to_string(), gguf_file::Value::Bool(true));
        assert!(validate_projector_activation(&metadata).is_ok());
    }

    #[test]
    fn projector_geometry_preserves_the_admitted_image_token_bound() {
        assert!(validate_patch_geometry(16, 2).is_ok());
        assert!(validate_patch_geometry(8, 2).is_err());
        assert!(validate_patch_geometry(32, 1).is_err());
        assert!(validate_patch_geometry(0, 2).is_err());
    }

    #[test]
    fn portable_attention_chunks_without_changing_global_attention() {
        let device = Device::Cpu;
        let query = Tensor::zeros((1, 1, 129, 2), DType::F32, &device).unwrap();
        let key = Tensor::zeros((1, 1, 129, 2), DType::F32, &device).unwrap();
        let values = (0..129)
            .flat_map(|index| [index as f32, (index * 2) as f32])
            .collect::<Vec<_>>();
        let value = Tensor::from_vec(values, (1, 1, 129, 2), &device).unwrap();

        let output = portable_attention(&query, &key, &value, 1.0).unwrap();
        let output = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(output.len(), 129 * 2);
        for pair in output.chunks_exact(2) {
            assert!((pair[0] - 64.0).abs() < 1e-4);
            assert!((pair[1] - 128.0).abs() < 2e-4);
        }
    }

    #[test]
    fn linear_casts_f32_bias_to_f16_activation_dtype() {
        let device = Device::Cpu;
        let weight = Tensor::from_slice(&[1f32, 0.0, 0.0, 1.0], (2, 2), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let linear = Linear {
            weight: QMatMul::Tensor(weight),
            bias: Tensor::from_slice(&[0.5f32, -0.5], 2, &device).unwrap(),
        };
        let input = Tensor::from_slice(&[1f32, 2.0], (1, 2), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        let output = linear.forward(&input).unwrap();

        assert_eq!(output.dtype(), DType::F16);
        assert_eq!(
            output
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1.5, 1.5]
        );
    }

    #[test]
    fn visual_rope_casts_f32_tables_to_f16_activation_dtype() {
        let device = Device::Cpu;
        let input = Tensor::from_slice(&[1f32, 2.0, 3.0, 4.0], (1, 1, 1, 4), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let cos = Tensor::zeros((1, 1, 1, 4), DType::F32, &device).unwrap();
        let sin = Tensor::ones((1, 1, 1, 4), DType::F32, &device).unwrap();

        let output = apply_visual_rope(&input, &cos, &sin).unwrap();

        assert_eq!(output.dtype(), DType::F16);
        assert_eq!(
            output
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![-3.0, -4.0, 1.0, 2.0]
        );
    }

    #[test]
    fn smart_resize_aligns_and_bounds_qwen_images() {
        assert_eq!(
            smart_resize_dimensions(640, 480, 32, 65_536, 4_194_304).unwrap(),
            (640, 480)
        );
        let (width, height) =
            smart_resize_dimensions(10_000, 5_000, 32, 65_536, 4_194_304).unwrap();
        assert_eq!(width % 32, 0);
        assert_eq!(height % 32, 0);
        assert!(width * height <= 4_194_304);
        let (width, height) = smart_resize_dimensions(64, 64, 32, 65_536, 4_194_304).unwrap();
        assert_eq!((width, height), (256, 256));
    }

    #[test]
    fn smart_resize_rejects_extreme_aspect_ratios() {
        assert!(smart_resize_dimensions(1, 1000, 32, 65_536, 4_194_304).is_err());
    }

    #[test]
    fn patch_order_makes_each_merge_tile_contiguous() {
        assert_eq!(
            spatial_merge_order(4, 4, 2),
            vec![0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
        );
    }

    #[test]
    fn rgb_storage_is_strictly_validated() {
        assert!(VisionImage::new(2, 2, vec![0; 12]).is_ok());
        assert!(VisionImage::new(2, 2, vec![0; 11]).is_err());
        assert!(VisionImage::new(0, 2, Vec::new()).is_err());
    }

    #[test]
    fn transparent_pixels_are_composited_onto_white() {
        let rgba = image::RgbaImage::from_raw(2, 1, vec![10, 20, 30, 0, 40, 50, 60, 255]).unwrap();

        let rgb = composite_onto_white(image::DynamicImage::ImageRgba8(rgba));

        assert_eq!(rgb.into_raw(), vec![255, 255, 255, 40, 50, 60]);
    }

    fn encode_gif(frames: &[[u8; 4]]) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut encoder = GifEncoder::new(&mut bytes);
            for rgba in frames {
                let image = RgbaImage::from_pixel(1, 1, Rgba(*rgba));
                encoder
                    .encode_frame(Frame::from_parts(
                        image,
                        0,
                        0,
                        Delay::from_numer_denom_ms(10, 1),
                    ))
                    .unwrap();
            }
        }
        bytes
    }

    #[test]
    fn static_gif_is_decoded_and_composited_onto_white() {
        let gif = encode_gif(&[[10, 20, 30, 0]]);

        let image = VisionImage::decode(&gif).unwrap();

        assert_eq!((image.width, image.height), (1, 1));
        assert_eq!(image.pixels, vec![255, 255, 255]);
    }

    #[test]
    fn animated_inputs_are_rejected_before_decode() {
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend_from_slice(&8u32.to_be_bytes());
        png.extend_from_slice(b"acTL\0\0\0\0\0\0\0\0\0\0\0\0");
        assert!(VisionImage::decode(&png).is_err());
        let mut webp = b"RIFF\x10\0\0\0WEBP".to_vec();
        webp.extend_from_slice(b"ANIM\0\0\0\0");
        assert!(VisionImage::decode(&webp).is_err());

        let gif = encode_gif(&[[10, 20, 30, 255], [40, 50, 60, 255]]);
        let error = VisionImage::decode(&gif).unwrap_err();
        assert!(error.to_string().contains("animated GIF"));
    }

    #[test]
    fn animation_markers_inside_chunk_data_are_not_false_positives() {
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend_from_slice(&4u32.to_be_bytes());
        png.extend_from_slice(b"tEXtacTL\0\0\0\0");
        assert!(!png_has_animation_chunk(&png));

        let mut webp = b"RIFF\x10\0\0\0WEBP".to_vec();
        webp.extend_from_slice(b"VP8 \x04\0\0\0ANIM");
        assert!(!webp_has_animation_chunk(&webp));
    }
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
mod cpu_w8a32_tests {
    use super::*;
    use candle_core::quantized::{GgmlDType, QTensor};
    #[test]
    fn cpu_q8_projection_preserves_f32_inputs_views_and_chunks() -> Result<()> {
        let d = Device::Cpu;
        let w = Tensor::from_vec(
            (0..256)
                .map(|i| ((i * 7 % 53) as f32 - 26.) / 17.)
                .collect::<Vec<_>>(),
            (8, 32),
            &d,
        )?;
        let q = QTensor::quantize(&w, GgmlDType::Q8_0)?;
        let wide = q.dequantize(&d)?.to_vec2::<f32>()?;
        let q = QMatMul::QTensor(Arc::new(q));
        let input = Tensor::from_vec(
            (0..11 * 35)
                .map(|i| ((i * 13 % 97) as f32 - 48.) / 31.)
                .collect::<Vec<_>>(),
            (11, 35),
            &d,
        )?;
        let x = input.narrow(0, 1, 9)?.narrow(1, 2, 32)?;
        let all = cpu_w8a32_projection(&q, &x)?.unwrap().to_vec2::<f32>()?;
        assert_eq!(
            all,
            cpu_w8a32_projection(&q, &x.contiguous()?)?
                .unwrap()
                .to_vec2::<f32>()?
        );
        let xf = x.to_vec2::<f32>()?;
        for (r, output) in all.iter().enumerate() {
            for (c, &actual) in output.iter().enumerate() {
                let exact: f64 = wide[c]
                    .iter()
                    .zip(&xf[r])
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                let bound: f64 = wide[c]
                    .iter()
                    .zip(&xf[r])
                    .map(|(&a, &b)| (f64::from(a) * f64::from(b)).abs())
                    .sum();
                assert!((f64::from(actual) - exact).abs() <= 2e-6 + 2e-6 * bound);
            }
        }
        for chunk in [1, 2, 4, 8] {
            let mut got = Vec::new();
            for start in (0..9).step_by(chunk) {
                got.extend(
                    cpu_w8a32_projection(
                        &q,
                        &x.narrow(0, start, chunk.min(9 - start))?.unsqueeze(0)?,
                    )?
                    .unwrap()
                    .squeeze(0)?
                    .to_vec2::<f32>()?,
                );
            }
            for (r, output) in got.iter().enumerate() {
                for (c, &actual) in output.iter().enumerate() {
                    let exact: f64 = wide[c]
                        .iter()
                        .zip(&xf[r])
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum();
                    let bound: f64 = wide[c]
                        .iter()
                        .zip(&xf[r])
                        .map(|(&a, &b)| (f64::from(a) * f64::from(b)).abs())
                        .sum();
                    assert!((f64::from(actual) - exact).abs() <= 2e-6 + 2e-6 * bound);
                }
            }
        }
        Ok(())
    }
    #[test]
    fn cpu_q8_projection_declines_other_types_and_checks_before_allocation() -> Result<()> {
        let d = Device::Cpu;
        let w = Tensor::ones((8, 32), DType::F32, &d)?;
        let q = QMatMul::QTensor(Arc::new(QTensor::quantize(&w, GgmlDType::Q8_0)?));
        let x = Tensor::ones((1, 32), DType::F32, &d)?;
        assert!(cpu_w8a32_projection(&QMatMul::Tensor(w.clone()), &x)?.is_none());
        let q4 = QMatMul::QTensor(Arc::new(QTensor::quantize(&w, GgmlDType::Q4_0)?));
        assert!(cpu_w8a32_projection(&q4, &x)?.is_none());
        assert!(cpu_w8a32_projection(&q, &x.to_dtype(DType::F64)?)?.is_none());
        assert!(cpu_w8a32_projection(&q, &x.broadcast_as((2, 1, 32))?)?.is_none());
        assert!(cpu_w8a32_projection(&q, &x.broadcast_as((2_097_153, 32))?).is_err());
        Ok(())
    }
}
