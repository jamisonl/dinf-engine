//! Inkling 2's vision tower (projector type `inkling2v`).
//!
//! A ViT over 20-pixel patches: a class token, learned position embeddings
//! (a 32 x 32 grid, bilinearly resized to the image) and two-axis RoPE inside
//! attention. Pre-norm LayerNorm blocks scale each branch by a learned vector
//! before its residual add. After the last block the class token is dropped,
//! 2 x 2 patches are concatenated, and a two-layer projector with exact GELU
//! maps them to the decoder width: one decoder embedding per 40 x 40 pixels.
//!
//! Images are not resized unless they exceed [`Inkling2VisionEncoder::max_image_tokens`];
//! they are padded on the right and bottom to whole 40-pixel tiles with a raw
//! value of -1, normalized like the image (as the reference does).

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::LayerNorm;

use super::{metadata_f64, metadata_string, metadata_usize};
use crate::gguf_limits::{validate_gguf_header_budget, validate_layer_count};

const PROJECTOR_TYPE: &str = "inkling2v";
const ROPE_THETA: f64 = 10_000.0;
/// Raw pixel value of the padding, before normalization.
const PAD_VALUE: f32 = -1.0;
/// Query chunk size for ViT self-attention so max-size images (up to 4,097
/// tokens) bound score/softmax buffers to ~270 MB instead of >2 GB.
const ATTN_QUERY_CHUNK: usize = 512;

struct Dense {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Dense {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        let y = if x.device().is_cuda() && self.weight.dtype() == DType::BF16 {
            crate::architectures::inkling2_cuda::bf16_matmul(&x.contiguous()?, &self.weight)?
        } else {
            x.matmul(&self.weight.t()?)?
        };
        #[cfg(not(feature = "cuda"))]
        let y = x.matmul(&self.weight.t()?)?;
        Ok(match &self.bias {
            Some(bias) => y.broadcast_add(bias)?,
            None => y,
        })
    }
}

struct Block {
    ln1: LayerNorm,
    q: Dense,
    k: Dense,
    v: Dense,
    out: Dense,
    ls1: Tensor,
    ln2: LayerNorm,
    up: Dense,
    down: Dense,
    ls2: Tensor,
}

/// How one image is prepared: its size after any downscale, and the padded
/// size the tower sees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Inkling2ImagePlan {
    /// Width the image is encoded at, after any downscale.
    pub width: usize,
    /// Height the image is encoded at, after any downscale.
    pub height: usize,
    /// Width after padding to whole tiles.
    pub padded_width: usize,
    /// Height after padding to whole tiles.
    pub padded_height: usize,
    /// Decoder embeddings the image becomes.
    pub tokens: usize,
}

/// See [`Inkling2VisionEncoder::plan`]: `tile` pixels per side of one
/// decoder embedding, at most `max_tokens` of them.
fn plan_image(
    width: usize,
    height: usize,
    tile: usize,
    max_tokens: usize,
) -> Result<Inkling2ImagePlan> {
    anyhow::ensure!(width > 0 && height > 0, "image has no pixels");
    let align_up = |x: usize| x.div_ceil(tile) * tile;
    let max_pixels = max_tokens * tile * tile;
    let (width, height) = if align_up(width) * align_up(height) > max_pixels {
        let round = |x: f64| ((x / tile as f64).round() as usize).max(1) * tile;
        let (mut w, mut h) = (round(width as f64), round(height as f64));
        if w * h > max_pixels {
            let beta = ((height as f64 * width as f64) / max_pixels as f64).sqrt();
            let floor = |x: f64| ((x / tile as f64).floor() as usize).max(1) * tile;
            h = floor(height as f64 / beta);
            w = floor(width as f64 / beta);
        }
        (w, h)
    } else {
        (width, height)
    };
    let (padded_width, padded_height) = (align_up(width), align_up(height));
    let tokens = (padded_width / tile) * (padded_height / tile);
    anyhow::ensure!(
        (1..=max_tokens).contains(&tokens),
        "image needs {tokens} tokens; the limit is {max_tokens}"
    );
    Ok(Inkling2ImagePlan {
        width,
        height,
        padded_width,
        padded_height,
        tokens,
    })
}

/// The `inkling2v` vision tower and projector.
pub struct Inkling2VisionEncoder {
    /// `[hidden, 3 * patch * patch]`, channel-major patches.
    patch_embedding: Dense,
    /// The learned grid, row-major `[grid * grid][hidden]`, on the host for resizing.
    positions: Vec<f32>,
    /// Class embedding plus its position embedding, `[1, hidden]`.
    class: Tensor,
    pre_ln: LayerNorm,
    blocks: Vec<Block>,
    proj_in: Dense,
    proj_out: Dense,
    hidden: usize,
    heads: usize,
    patch: usize,
    merge: usize,
    grid: usize,
    decoder_dim: usize,
    mean: [f32; 3],
    std: [f32; 3],
    max_image_tokens: usize,
    device: Device,
}

fn read_array_f32(
    metadata: &std::collections::HashMap<String, gguf_file::Value>,
    key: &str,
) -> Result<[f32; 3]> {
    let value = metadata
        .get(key)
        .with_context(|| format!("vision projector is missing {key}"))?;
    let gguf_file::Value::Array(values) = value else {
        anyhow::bail!("vision projector {key} is not an array")
    };
    anyhow::ensure!(
        values.len() == 3,
        "vision projector {key} must have three values"
    );
    let mut out = [0f32; 3];
    for (slot, value) in out.iter_mut().zip(values) {
        *slot = value
            .to_f32()
            .with_context(|| format!("vision projector {key} must hold numbers"))?;
        anyhow::ensure!(slot.is_finite(), "vision projector {key} must be finite");
    }
    Ok(out)
}

impl Inkling2VisionEncoder {
    /// Load an `inkling2v` projector onto `device`; at most
    /// `max_image_tokens` decoder embeddings per image.
    pub fn load(path: &Path, device: &Device, max_image_tokens: usize) -> Result<Self> {
        let mut file = File::open(path)
            .with_context(|| format!("could not open vision projector '{}'", path.display()))?;
        validate_gguf_header_budget(&mut file)
            .with_context(|| format!("could not validate vision projector '{}'", path.display()))?;
        let content = gguf_file::Content::read(&mut file)
            .with_context(|| format!("could not parse vision projector '{}'", path.display()))?;
        let md = &content.metadata;
        anyhow::ensure!(
            metadata_string(md, "general.architecture")? == "clip",
            "vision projector architecture is not 'clip'"
        );
        let projector_type = metadata_string(md, "clip.vision.projector_type")?;
        anyhow::ensure!(
            projector_type == PROJECTOR_TYPE,
            "vision projector type '{projector_type}' is not {PROJECTOR_TYPE}"
        );
        let image_size = metadata_usize(md, "clip.vision.image_size")?;
        let patch = metadata_usize(md, "clip.vision.patch_size")?;
        let hidden = metadata_usize(md, "clip.vision.embedding_length")?;
        let ffn = metadata_usize(md, "clip.vision.feed_forward_length")?;
        let decoder_dim = metadata_usize(md, "clip.vision.projection_dim")?;
        let block_count = metadata_usize(md, "clip.vision.block_count")?;
        let heads = metadata_usize(md, "clip.vision.attention.head_count")?;
        let merge = metadata_usize(md, "clip.vision.spatial_merge_size")?;
        let eps = metadata_f64(md, "clip.vision.attention.layer_norm_epsilon")?;
        let mean = read_array_f32(md, "clip.vision.image_mean")?;
        let std = read_array_f32(md, "clip.vision.image_std")?;
        validate_layer_count(block_count, "vision block count")?;
        anyhow::ensure!(
            patch > 0 && merge > 0 && image_size % patch == 0,
            "vision patch geometry is invalid"
        );
        anyhow::ensure!(
            heads > 0 && hidden % heads == 0 && (hidden / heads) % 4 == 0,
            "vision head width cannot use two-axis RoPE"
        );
        anyhow::ensure!(
            (1..=16_384).contains(&hidden) && (1..=65_536).contains(&decoder_dim),
            "vision widths are out of range"
        );
        anyhow::ensure!(
            eps.is_finite() && eps > 0.0 && std.iter().all(|s| *s != 0.0),
            "vision normalization parameters are invalid"
        );
        anyhow::ensure!(max_image_tokens > 0, "an image needs at least one token");
        let grid = image_size / patch;

        // Keep 2-D BF16 projection weights in BF16 on CUDA to halve resident
        // VRAM (~637 MB instead of ~1.27 GB) and run them on Tensor Cores via
        // cuBLAS gemm_ex with F32 accumulation and output.
        let keep_bf16 = cfg!(feature = "cuda") && device.is_cuda();
        let mut shaped_with_bf16 = |name: &str, shape: &[usize], allow_bf16: bool| -> Result<Tensor> {
            let qt = content
                .tensor(&mut file, name, &Device::Cpu)
                .with_context(|| format!("vision projector is missing {name}"))?;
            anyhow::ensure!(
                qt.shape().dims() == shape,
                "vision tensor {name} has shape {:?}, expected {shape:?}",
                qt.shape().dims()
            );
            if allow_bf16 && keep_bf16 && qt.dtype() == candle_core::quantized::GgmlDType::BF16 {
                return Ok(Tensor::from_raw_buffer(
                    &qt.data()?,
                    DType::BF16,
                    shape,
                    device,
                )?);
            }
            Ok(qt
                .dequantize(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .to_device(device)?)
        };
        let patch_embedding = Dense {
            weight: shaped_with_bf16("v.patch_embd.weight", &[hidden, 3, patch, patch], true)?
                .reshape((hidden, 3 * patch * patch))?
                .contiguous()?,
            bias: None,
        };
        let positions = shaped_with_bf16("v.position_embd.weight", &[grid * grid, hidden], false)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let class = (shaped_with_bf16("v.class_embd", &[hidden], false)?
            + shaped_with_bf16("v.class_position_embd", &[hidden], false)?)?
        .reshape((1, hidden))?;
        let layer_norm = |load: &mut dyn FnMut(&str, &[usize], bool) -> Result<Tensor>,
                          name: &str|
         -> Result<LayerNorm> {
            Ok(LayerNorm::new(
                load(&format!("{name}.weight"), &[hidden], false)?,
                load(&format!("{name}.bias"), &[hidden], false)?,
                eps,
            ))
        };
        let dense = |load: &mut dyn FnMut(&str, &[usize], bool) -> Result<Tensor>,
                     name: &str,
                     out: usize,
                     inn: usize,
                     bias: bool|
         -> Result<Dense> {
            Ok(Dense {
                weight: load(&format!("{name}.weight"), &[out, inn], true)?,
                bias: bias
                    .then(|| load(&format!("{name}.bias"), &[out], false))
                    .transpose()?,
            })
        };
        let pre_ln = layer_norm(&mut shaped_with_bf16, "v.pre_ln")?;
        let blocks = (0..block_count)
            .map(|layer| {
                let b = format!("v.blk.{layer}");
                Ok(Block {
                    ln1: layer_norm(&mut shaped_with_bf16, &format!("{b}.ln1"))?,
                    q: dense(
                        &mut shaped_with_bf16,
                        &format!("{b}.attn_q"),
                        hidden,
                        hidden,
                        true,
                    )?,
                    k: dense(
                        &mut shaped_with_bf16,
                        &format!("{b}.attn_k"),
                        hidden,
                        hidden,
                        true,
                    )?,
                    v: dense(
                        &mut shaped_with_bf16,
                        &format!("{b}.attn_v"),
                        hidden,
                        hidden,
                        true,
                    )?,
                    out: dense(
                        &mut shaped_with_bf16,
                        &format!("{b}.attn_out"),
                        hidden,
                        hidden,
                        true,
                    )?,
                    ls1: shaped_with_bf16(&format!("{b}.ls1.weight"), &[hidden], false)?,
                    ln2: layer_norm(&mut shaped_with_bf16, &format!("{b}.ln2"))?,
                    up: dense(
                        &mut shaped_with_bf16,
                        &format!("{b}.ffn_up"),
                        ffn,
                        hidden,
                        true,
                    )?,
                    down: dense(
                        &mut shaped_with_bf16,
                        &format!("{b}.ffn_down"),
                        hidden,
                        ffn,
                        true,
                    )?,
                    ls2: shaped_with_bf16(&format!("{b}.ls2.weight"), &[hidden], false)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let merged = hidden * merge * merge;
        let proj_in = dense(&mut shaped_with_bf16, "mm.1", merged, merged, false)?;
        let proj_out = dense(&mut shaped_with_bf16, "mm.2", decoder_dim, merged, false)?;
        tracing::info!(
            path = %path.display(),
            hidden,
            blocks = block_count,
            decoder_dim,
            max_image_tokens,
            "Loaded Inkling 2 vision projector"
        );
        Ok(Self {
            patch_embedding,
            positions,
            class,
            pre_ln,
            blocks,
            proj_in,
            proj_out,
            hidden,
            heads,
            patch,
            merge,
            grid,
            decoder_dim,
            mean,
            std,
            max_image_tokens,
            device: device.clone(),
        })
    }

    /// Width of the decoder embeddings the projector emits.
    pub fn decoder_dim(&self) -> usize {
        self.decoder_dim
    }

    /// Most decoder embeddings one image may become.
    pub fn max_image_tokens(&self) -> usize {
        self.max_image_tokens
    }

    /// Pixels per side of one decoder embedding.
    fn tile(&self) -> usize {
        self.patch * self.merge
    }

    /// The size an image of `width` x `height` is encoded at. Larger images
    /// are downscaled, keeping their aspect ratio, to fit the token cap.
    pub fn plan(&self, width: usize, height: usize) -> Result<Inkling2ImagePlan> {
        plan_image(width, height, self.tile(), self.max_image_tokens)
    }

    /// Encode RGB bytes (`width * height * 3`, row-major) already at the
    /// planned size into `[plan.tokens, decoder_dim]` F32 embeddings.
    pub fn encode(&self, rgb: &[u8], plan: &Inkling2ImagePlan) -> Result<Tensor> {
        let (w, h) = (plan.width, plan.height);
        anyhow::ensure!(rgb.len() == w * h * 3, "image bytes do not match its size");
        let p = self.patch;
        let (cols, rows) = (plan.padded_width / p, plan.padded_height / p);
        let patches = cols * rows;
        // Normalized patches, channel-major within each patch, the patches
        // row-major over the padded image.
        let pad = |c: usize| (PAD_VALUE - self.mean[c] * 255.0) / (self.std[c] * 255.0);
        let mut pixels = vec![0f32; patches * 3 * p * p];
        for py in 0..rows {
            for px in 0..cols {
                let base = (py * cols + px) * 3 * p * p;
                for c in 0..3 {
                    let (mean, std) = (self.mean[c] * 255.0, self.std[c] * 255.0);
                    for ky in 0..p {
                        let y = py * p + ky;
                        for kx in 0..p {
                            let x = px * p + kx;
                            let value = if x < w && y < h {
                                (f32::from(rgb[(y * w + x) * 3 + c]) - mean) / std
                            } else {
                                pad(c)
                            };
                            pixels[base + c * p * p + ky * p + kx] = value;
                        }
                    }
                }
            }
        }
        let pixels = Tensor::from_vec(pixels, (patches, 3 * p * p), &self.device)?;
        let embedded = self.patch_embedding.forward(&pixels)?;
        let positions = Tensor::from_vec(
            self.resized_positions(cols, rows),
            (patches, self.hidden),
            &self.device,
        )?;
        let tokens = Tensor::cat(&[&self.class, &(embedded + positions)?], 0)?;
        let mut x = self.pre_ln.forward(&tokens)?;

        let (cos, sin) = self.rope_tables(cols, rows)?;
        for block in &self.blocks {
            x = self.block(block, &x, &cos, &sin)?;
        }

        // Drop the class token, then merge 2 x 2 patches: top-left,
        // top-right, bottom-left, bottom-right, the tiles row-major.
        let m = self.merge;
        let x = x
            .narrow(0, 1, patches)?
            .reshape((rows / m, m, cols / m, m, self.hidden))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?
            .reshape(((rows / m) * (cols / m), m * m * self.hidden))?;
        let x = self.proj_in.forward(&x)?.gelu_erf()?;
        let out = self.proj_out.forward(&x)?;
        anyhow::ensure!(
            out.dim(0)? == plan.tokens,
            "vision tower emitted the wrong row count"
        );
        Ok(out)
    }

    /// The learned grid resized to `cols` x `rows` patches, as ggml's
    /// half-pixel bilinear interpolation (clamped at the edges).
    fn resized_positions(&self, cols: usize, rows: usize) -> Vec<f32> {
        let (g, hidden) = (self.grid, self.hidden);
        if cols == g && rows == g {
            return self.positions.clone();
        }
        let sample = |i: usize, out: usize| -> (usize, usize, f32) {
            let x = (i as f32 + 0.5) * g as f32 / out as f32 - 0.5;
            let floor = x.floor();
            let x0 = (floor as i64).clamp(0, g as i64 - 1) as usize;
            let x1 = (floor as i64 + 1).clamp(0, g as i64 - 1) as usize;
            let dx = (x - x0 as f32).clamp(0.0, 1.0);
            (x0, x1, dx)
        };
        let mut out = vec![0f32; cols * rows * hidden];
        for y in 0..rows {
            let (y0, y1, dy) = sample(y, rows);
            for x in 0..cols {
                let (x0, x1, dx) = sample(x, cols);
                let at = |yy: usize, xx: usize| &self.positions[(yy * g + xx) * hidden..][..hidden];
                let (a, b, c, d) = (at(y0, x0), at(y0, x1), at(y1, x0), at(y1, x1));
                let row = &mut out[(y * cols + x) * hidden..][..hidden];
                for e in 0..hidden {
                    let top = a[e] + (b[e] - a[e]) * dx;
                    let bottom = c[e] + (d[e] - c[e]) * dx;
                    row[e] = top + (bottom - top) * dy;
                }
            }
        }
        out
    }

    /// Two-axis RoPE: the first half of each head turns with the column, the
    /// second with the row, adjacent pairs at base^(-2i / (head / 2)). The
    /// class token sits at (0, 0) and patches at (x + 1, y + 1). Returns
    /// cos and sin as `[tokens, 1, head / 2]`, one angle per pair.
    fn rope_tables(&self, cols: usize, rows: usize) -> Result<(Tensor, Tensor)> {
        let head = self.hidden / self.heads;
        let half = head / 2;
        let pairs = half / 2;
        let freqs = (0..pairs)
            .map(|i| ROPE_THETA.powf(-(2.0 * i as f64) / half as f64))
            .collect::<Vec<_>>();
        let tokens = cols * rows + 1;
        let (mut cos, mut sin) = (
            Vec::with_capacity(tokens * half),
            Vec::with_capacity(tokens * half),
        );
        for t in 0..tokens {
            let (pw, ph) = if t == 0 {
                (0.0, 0.0)
            } else {
                (((t - 1) % cols + 1) as f64, ((t - 1) / cols + 1) as f64)
            };
            for axis in [pw, ph] {
                for f in &freqs {
                    let angle = axis * f;
                    cos.push(angle.cos() as f32);
                    sin.push(angle.sin() as f32);
                }
            }
        }
        Ok((
            Tensor::from_vec(cos, (tokens, 1, half), &self.device)?,
            Tensor::from_vec(sin, (tokens, 1, half), &self.device)?,
        ))
    }

    fn rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        // `[tokens, heads, head]` as adjacent pairs.
        let (tokens, heads, head) = x.dims3()?;
        let pairs = x.reshape((tokens, heads, head / 2, 2))?;
        let even = pairs.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
        let odd = pairs.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;
        let rotated_even = (even.broadcast_mul(cos)? - odd.broadcast_mul(sin)?)?;
        let rotated_odd = (even.broadcast_mul(sin)? + odd.broadcast_mul(cos)?)?;
        Ok(Tensor::stack(&[rotated_even, rotated_odd], D::Minus1)?
            .reshape((tokens, heads, head))?)
    }

    fn block(&self, block: &Block, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let tokens = x.dim(0)?;
        let head = self.hidden / self.heads;
        let h = block.ln1.forward(x)?;
        let split = |t: Tensor| t.reshape((tokens, self.heads, head));
        let q = self.rope(&split(block.q.forward(&h)?)?, cos, sin)?;
        let k = self.rope(&split(block.k.forward(&h)?)?, cos, sin)?;
        let v = split(block.v.forward(&h)?)?;
        let (q, k, v) = (
            q.transpose(0, 1)?.contiguous()?,
            k.transpose(0, 1)?.contiguous()?,
            v.transpose(0, 1)?.contiguous()?,
        );
        let attended = attend_chunked(&q, &k, &v, head, ATTN_QUERY_CHUNK)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((tokens, self.hidden))?;
        let x = (x + block.out.forward(&attended)?.broadcast_mul(&block.ls1)?)?;
        let h = block.ln2.forward(&x)?;
        let h = block.down.forward(&block.up.forward(&h)?.gelu_erf()?)?;
        Ok((x + h.broadcast_mul(&block.ls2)?)?)
    }
}

/// Full self-attention `[heads, tokens, head]`, chunked over queries so large
/// images never materialize a full `[heads, tokens, tokens]` score table.
fn attend_chunked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head: usize,
    chunk_size: usize,
) -> Result<Tensor> {
    let tokens = q.dim(1)?;
    let scale = (head as f64).sqrt();
    let k_t = k.t()?;
    if tokens <= chunk_size {
        let scores = (q.matmul(&k_t)? / scale)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        return Ok(weights.matmul(v)?);
    }
    let mut chunks = Vec::with_capacity(tokens.div_ceil(chunk_size));
    for start in (0..tokens).step_by(chunk_size) {
        let len = chunk_size.min(tokens - start);
        let q_chunk = q.narrow(1, start, len)?.contiguous()?;
        let scores = (q_chunk.matmul(&k_t)? / scale)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        chunks.push(weights.matmul(v)?);
    }
    let refs = chunks.iter().collect::<Vec<_>>();
    Ok(Tensor::cat(&refs, 1)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunked_attention_matches_single_pass() -> Result<()> {
        let (heads, tokens, head) = (4, 19, 8);
        let values = |offset: f32| {
            (0..heads * tokens * head)
                .map(|i| ((i as f32 * 0.13 + offset).sin()) * 0.5)
                .collect::<Vec<_>>()
        };
        let q = Tensor::from_vec(values(0.0), (heads, tokens, head), &Device::Cpu)?;
        let k = Tensor::from_vec(values(1.7), (heads, tokens, head), &Device::Cpu)?;
        let v = Tensor::from_vec(values(3.1), (heads, tokens, head), &Device::Cpu)?;
        let whole = attend_chunked(&q, &k, &v, head, tokens)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let chunked = attend_chunked(&q, &k, &v, head, 6)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(whole.len(), chunked.len());
        let max_diff = whole
            .iter()
            .zip(&chunked)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-6, "max diff {max_diff}");
        Ok(())
    }

    #[test]
    fn small_images_keep_their_size_and_pad_to_tiles() -> Result<()> {
        let plan = plan_image(130, 90, 40, 1024)?;
        assert_eq!(
            (
                plan.width,
                plan.height,
                plan.padded_width,
                plan.padded_height,
                plan.tokens
            ),
            (130, 90, 160, 120, 12)
        );
        let plan = plan_image(640, 488, 40, 1024)?;
        assert_eq!((plan.width, plan.height, plan.tokens), (640, 488, 16 * 13));
        let plan = plan_image(1, 1, 40, 1024)?;
        assert_eq!((plan.padded_width, plan.tokens), (40, 1));
        assert!(plan_image(0, 10, 40, 1024).is_err());
        Ok(())
    }

    #[test]
    fn large_images_shrink_to_the_cap_keeping_their_shape() -> Result<()> {
        for (width, height) in [(4032, 3024), (1281, 1281), (2000, 900), (60_000, 100)] {
            let plan = plan_image(width, height, 40, 1024)?;
            assert!(plan.tokens <= 1024, "{width}x{height}: {plan:?}");
            assert_eq!((plan.width % 40, plan.height % 40), (0, 0), "{plan:?}");
            assert_eq!(
                (plan.padded_width, plan.padded_height),
                (plan.width, plan.height)
            );
            if height >= 200 {
                let (before, after) = (
                    width as f64 / height as f64,
                    plan.width as f64 / plan.height as f64,
                );
                assert!(
                    (after / before - 1.0).abs() < 0.15,
                    "{width}x{height}: {plan:?}"
                );
            }
        }
        // 1280 x 1280 is exactly the cap and is not resized.
        assert_eq!(plan_image(1280, 1280, 40, 1024)?.width, 1280);
        Ok(())
    }
}
