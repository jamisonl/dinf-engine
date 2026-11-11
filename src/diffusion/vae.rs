//! Qwen-Image-2.1's single-image VAE.
//!
//! The implementation follows the upstream
//! [`AutoencoderKLQwenImage21`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py)
//! module.  The upstream model is written as a causal 3-D VAE so that the
//! same weights can also process video.  Qwen-Image-2.1 is an image model,
//! and this module deliberately exposes the exact `T = 1` path: all temporal
//! convolutions are spatial 2-D convolutions and the first causal chunk skips
//! the temporal cache convolution.  The temporal/spatial folding in the
//! residual shortcut paths is retained, since it changes the first-frame
//! values even when there is only one frame.
//!
//! Public tensor shapes are image-shaped, rather than the internal upstream
//! video-shaped tensors:
//!
//! * `encode`: `[B, in_channels, H, W]` -> `[B, 2 * z_dim, H/16, W/16]`.
//!   The first `z_dim` channels are the posterior mean and the second half
//!   are log-variance moments, matching Diffusers' diagonal Gaussian.
//! * `encode_mode`: same input -> `[B, z_dim, H/16, W/16]`, selecting the
//!   deterministic posterior mode.
//! * `decode`: `[B, z_dim, H, W]` -> `[B, out_channels, H*16, W*16]`.
//!   Latents passed to `decode` are the *raw VAE latents*, after reversing
//!   the diffusion-pipeline normalization.  Use `denormalize_latents` when
//!   consuming normalized diffusion latents.
//!
//! The VAE itself does not apply `latents_mean`/`latents_std`; Diffusers does
//! that around the VAE.  `normalize_latents` and `denormalize_latents` are
//! provided to make that boundary explicit.

use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};
use serde::Deserialize;

use super::attention::attention;

const RMS_EPS: f64 = 1e-12;
const CONV_WORKSPACE_BYTES: usize = 256 * 1024 * 1024;
// The released Qwen-Image-2.1 VAE uses two residual blocks per stage. Keep a
// generous finite ceiling for compatible variants so a malformed config cannot
// turn model loading into an unbounded allocation.
const MAX_RES_BLOCKS: usize = 64;
// `dim_mult` includes one entry for each VAE level. The encoder and decoder
// resample between adjacent levels, so five levels are required for the
// supported 16x spatial factor (four 2x resampling stages).
const REQUIRED_DIM_MULT_LEVELS: usize = 5;

/// Candle's Metal convolution expands its input into an im2col workspace.
/// At native 2K this can exceed 40GB for a single VAE convolution. Split
/// output rows, including each row's complete input halo. This preserves the
/// convolution and its precision; it does not tile/blend the decoded image.
fn bounded_conv2d(conv: &Conv2d, x: &Tensor, workspace_bytes: usize) -> Result<Tensor> {
    let (batch, channels, height, width) = x.dims4()?;
    let (_, _, kh, kw) = conv.weight().dims4()?;
    let config = conv.config();
    // VAE convolutions pad explicitly before entering this helper.
    if config.padding != 0
        || config.dilation != 1
        || config.groups != 1
        || height < kh
        || width < kw
        || config.stride == 0
    {
        return conv.forward(x);
    }
    let rows = (height - kh) / config.stride + 1;
    let columns = (width - kw) / config.stride + 1;
    let bytes_per_row = batch
        .saturating_mul(columns)
        .saturating_mul(channels)
        .saturating_mul(kh)
        .saturating_mul(kw)
        .saturating_mul(x.dtype().size_in_bytes());
    let tile_rows = (workspace_bytes / bytes_per_row.max(1)).max(1);
    if rows <= tile_rows {
        return conv.forward(x);
    }
    if x.device().is_metal() {
        x.device().synchronize()?;
    }
    let mut outputs = Vec::new();
    for start in (0..rows).step_by(tile_rows) {
        let count = tile_rows.min(rows - start);
        let patch = x
            .narrow(2, start * config.stride, (count - 1) * config.stride + kh)?
            .contiguous()?;
        outputs.push(conv.forward(&patch)?);
        drop(patch);
        // Let the allocator release the previous im2col buffer before the
        // next tile. Merely dropping its Tensor still leaves queued GPU work.
        if x.device().is_metal() {
            x.device().synchronize()?;
        }
    }
    Tensor::cat(&outputs, 2)
}

/// Configuration from Qwen-Image-2.1's `vae/config.json`.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct PublicVaeConfig {
    /// Base width of the encoder.
    pub base_dim: usize,
    /// Base width of the decoder.
    pub decoder_base_dim: Option<usize>,
    /// Number of latent channels.
    pub z_dim: usize,
    /// Channel multipliers for the encoder and decoder stages.
    pub dim_mult: Vec<usize>,
    /// Number of residual blocks per encoder stage.
    pub num_res_blocks: usize,
    /// Spatial scales at which the upstream VAE would add attention blocks.
    pub attn_scales: Vec<f32>,
    /// Upstream spelling is intentionally `temperal_downsample`.
    #[serde(alias = "temporal_downsample")]
    pub temperal_downsample: Vec<bool>,
    /// Inference dropout; the released model sets this to zero.
    pub dropout: f64,
    /// Per-channel VAE latent mean used by the surrounding diffusion pipeline.
    pub latents_mean: Vec<f32>,
    /// Per-channel VAE latent standard deviation used by the surrounding pipeline.
    pub latents_std: Vec<f32>,
    /// Whether the residual down/up blocks are used.
    pub is_residual: bool,
    /// Number of channels in the VAE input image.
    pub in_channels: usize,
    /// Number of channels emitted by the VAE (four RGBA channels here).
    pub out_channels: usize,
    /// Optional patching is `null` for the released Qwen-Image-2.1 VAE.
    pub patch_size: Option<usize>,
    /// Video temporal compression factor from the upstream config.
    pub scale_factor_temporal: Option<usize>,
    /// Image spatial compression factor from the upstream config.
    pub scale_factor_spatial: Option<usize>,
}

// These are learned channel statistics; values such as -0.5236 must not be
// rewritten as mathematical constants by clippy's readability lint.
#[allow(clippy::approx_constant)]
impl Default for PublicVaeConfig {
    fn default() -> Self {
        Self {
            base_dim: 96,
            decoder_base_dim: Some(144),
            z_dim: 64,
            dim_mult: vec![1, 2, 4, 8, 8],
            num_res_blocks: 2,
            attn_scales: Vec::new(),
            temperal_downsample: vec![false, true, true, true],
            dropout: 0.,
            latents_mean: vec![
                0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873,
                -1.7177, -0.6510, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.0210, 0.0472, 0.9320,
                2.0087, 2.4954, -0.1391, -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286,
                -1.4518, -1.9036, -1.9955, -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751,
                -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.3680, 0.4809, -0.4433, 0.3764,
                0.5729, -2.0595, 1.0960, -1.3260, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026,
                1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
            ],
            latents_std: vec![
                3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910, 3.0785, 3.6500,
                3.9308, 3.0904, 2.8778, 3.7675, 3.7320, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443,
                2.9249, 3.9454, 3.0988, 4.2489, 3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.2830,
                3.5694, 4.2467, 3.9037, 3.2947, 5.0770, 3.5075, 3.2700, 3.4767, 2.8063, 5.1125,
                3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875, 3.9624, 4.0168,
                3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959, 3.8747, 3.7608, 3.5735, 3.1490,
                3.7662, 3.6746, 3.4563, 3.8161,
            ],
            is_residual: true,
            in_channels: 4,
            out_channels: 4,
            patch_size: None,
            scale_factor_temporal: Some(8),
            scale_factor_spatial: Some(16),
        }
    }
}

impl PublicVaeConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.base_dim == 0 || self.z_dim == 0 || self.in_channels == 0 || self.out_channels == 0
        {
            candle_core::bail!("Qwen-Image VAE channel dimensions must be positive")
        }
        let decoder_base_dim = self.decoder_base_dim.unwrap_or(self.base_dim);
        if decoder_base_dim == 0 {
            candle_core::bail!("Qwen-Image VAE decoder_base_dim must be positive")
        }
        if self.num_res_blocks == 0 || self.num_res_blocks > MAX_RES_BLOCKS {
            candle_core::bail!(
                "Qwen-Image VAE num_res_blocks must be in 1..={MAX_RES_BLOCKS} (got {})",
                self.num_res_blocks
            )
        }
        if self.dim_mult.is_empty() {
            candle_core::bail!("Qwen-Image VAE dim_mult must not be empty")
        }
        if self.dim_mult.len() != REQUIRED_DIM_MULT_LEVELS {
            candle_core::bail!(
                "Qwen-Image VAE dim_mult must contain exactly {REQUIRED_DIM_MULT_LEVELS} levels for the supported 16x spatial factor (got {})",
                self.dim_mult.len()
            )
        }
        if self.z_dim > usize::MAX / 2
            || self
                .dim_mult
                .iter()
                .any(|&mult| mult == 0 || self.base_dim.checked_mul(mult).is_none())
            || self
                .dim_mult
                .iter()
                .any(|&mult| decoder_base_dim.checked_mul(mult).is_none())
        {
            candle_core::bail!("Qwen-Image VAE channel dimensions overflow or contain zero")
        }
        if self.temperal_downsample.len() != self.dim_mult.len().saturating_sub(1) {
            candle_core::bail!(
                "Qwen-Image VAE temperal_downsample has length {}, expected {}",
                self.temperal_downsample.len(),
                self.dim_mult.len().saturating_sub(1)
            )
        }
        if self.latents_mean.len() != self.z_dim || self.latents_std.len() != self.z_dim {
            candle_core::bail!(
                "Qwen-Image VAE latent statistics must have z_dim={} entries (got mean={}, std={})",
                self.z_dim,
                self.latents_mean.len(),
                self.latents_std.len()
            )
        }
        if self.latents_mean.iter().any(|value| !value.is_finite()) {
            candle_core::bail!("Qwen-Image VAE latents_mean must be finite")
        }
        if self
            .latents_std
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.)
        {
            candle_core::bail!("Qwen-Image VAE latents_std must be finite and positive")
        }
        if self.scale_factor_spatial != Some(16) {
            candle_core::bail!(
                "Qwen-Image-2.1 single-image VAE requires scale_factor_spatial=16 (got {:?})",
                self.scale_factor_spatial
            )
        }
        if self.patch_size.is_some_and(|patch| patch != 1) {
            candle_core::bail!("Qwen-Image-2.1 VAE only supports patch_size=null or 1")
        }
        if !self.is_residual {
            candle_core::bail!("only the released residual Qwen-Image VAE topology is supported")
        }
        Ok(())
    }
}

struct CausalConv2d {
    conv: Conv2d,
    padding: usize,
}

impl CausalConv2d {
    fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let conv = candle_nn::conv2d(
            in_channels,
            out_channels,
            kernel_size,
            Conv2dConfig {
                padding: 0,
                stride,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
            vb,
        )?;
        Ok(Self { conv, padding })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // The upstream layer is a Conv2d wrapped in a singleton temporal
        // dimension.  Keep that dimension around in the VAE graph so the
        // temporal folding shortcuts remain identical to the reference
        // implementation, but perform the actual convolution on rank-4
        // `[B*T,C,H,W]` tensors.  This implementation is intentionally the
        // image specialization, so a caller cannot accidentally get a
        // silently incorrect multi-frame result.
        let had_temporal = match x.rank() {
            4 => false,
            5 => {
                if x.dim(2)? != 1 {
                    candle_core::bail!(
                        "Qwen-Image VAE T=1 specialization received {} frames",
                        x.dim(2)?
                    )
                }
                true
            }
            rank => {
                candle_core::bail!("Qwen-Image VAE convolution expects rank 4 or 5, got {rank}")
            }
        };
        let x = if had_temporal {
            x.squeeze(2)?
        } else {
            x.clone()
        };
        let x = if self.padding == 0 {
            x
        } else {
            pad_2d(&x, self.padding, self.padding, self.padding, self.padding)?
        };
        let x = if x.device().is_metal() {
            bounded_conv2d(&self.conv, &x, CONV_WORKSPACE_BYTES)?
        } else {
            self.conv.forward(&x)?
        };
        if had_temporal { x.unsqueeze(2) } else { Ok(x) }
    }
}

struct ChannelRmsNorm {
    gamma: Tensor,
    channels: usize,
}

impl ChannelRmsNorm {
    fn load(vb: VarBuilder, channels: usize, images: bool) -> Result<Self> {
        // The released checkpoint preserves the source module's parameter
        // shape: image attention uses [C,1,1], while residual/norm-out paths
        // use [C,1,1,1].  Asking VarBuilder for the exact shape matters for
        // strict safetensor backends; all forms are reshaped for broadcasting
        // at execution time below.
        let shape = if images {
            vec![channels, 1, 1]
        } else {
            vec![channels, 1, 1, 1]
        };
        Ok(Self {
            gamma: vb.get(shape, "gamma")?,
            channels,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x_f32 = match dtype {
            DType::F16 | DType::BF16 => x.to_dtype(DType::F32)?,
            _ => x.clone(),
        };
        let denom = x_f32.sqr()?.sum_keepdim(1)?.sqrt()?.maximum(RMS_EPS)?;
        let scale = (self.channels as f64).sqrt();
        let normalized = (x_f32.broadcast_div(&denom)? * scale)?;
        let normalized = normalized.to_dtype(dtype)?;
        let gamma = match normalized.rank() {
            4 => self.gamma.reshape((1, self.channels, 1, 1))?,
            5 => self.gamma.reshape((1, self.channels, 1, 1, 1))?,
            rank => candle_core::bail!("Qwen-Image VAE RMS norm expects rank 4 or 5, got {rank}"),
        };
        normalized.broadcast_mul(&gamma)
    }
}

struct ResidualBlock {
    norm1: ChannelRmsNorm,
    conv1: CausalConv2d,
    norm2: ChannelRmsNorm,
    conv2: CausalConv2d,
    shortcut: Option<CausalConv2d>,
}

impl ResidualBlock {
    fn load(vb: VarBuilder, in_dim: usize, out_dim: usize) -> Result<Self> {
        Ok(Self {
            norm1: ChannelRmsNorm::load(vb.pp("norm1"), in_dim, false)?,
            conv1: CausalConv2d::load(vb.pp("conv1"), in_dim, out_dim, 3, 1, 1)?,
            norm2: ChannelRmsNorm::load(vb.pp("norm2"), out_dim, false)?,
            conv2: CausalConv2d::load(vb.pp("conv2"), out_dim, out_dim, 3, 1, 1)?,
            shortcut: (in_dim != out_dim)
                .then(|| CausalConv2d::load(vb.pp("conv_shortcut"), in_dim, out_dim, 1, 1, 0))
                .transpose()?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shortcut = match &self.shortcut {
            Some(shortcut) => shortcut.forward(x)?,
            None => x.clone(),
        };
        let x = candle_nn::ops::silu(&self.norm1.forward(x)?)?;
        let x = self.conv1.forward(&x)?;
        let x = candle_nn::ops::silu(&self.norm2.forward(&x)?)?;
        let x = self.conv2.forward(&x)?;
        x + shortcut
    }
}

struct AttentionBlock {
    norm: ChannelRmsNorm,
    to_qkv: Conv2d,
    proj: Conv2d,
    channels: usize,
}

impl AttentionBlock {
    fn load(vb: VarBuilder, channels: usize) -> Result<Self> {
        Ok(Self {
            norm: ChannelRmsNorm::load(vb.pp("norm"), channels, true)?,
            to_qkv: candle_nn::conv2d(
                channels,
                channels * 3,
                1,
                Conv2dConfig::default(),
                vb.pp("to_qkv"),
            )?,
            proj: candle_nn::conv2d(
                channels,
                channels,
                1,
                Conv2dConfig::default(),
                vb.pp("proj"),
            )?,
            channels,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = x.clone();
        let (batch, _, frames, height, width) = x.dims5()?;
        if frames != 1 {
            candle_core::bail!(
                "Qwen-Image VAE attention T=1 specialization received {} frames",
                frames
            )
        }
        let x = x.permute((0, 2, 1, 3, 4))?.contiguous()?.reshape((
            batch * frames,
            self.channels,
            height,
            width,
        ))?;
        let x = self.norm.forward(&x)?;
        let qkv = self
            .to_qkv
            .forward(&x)?
            .reshape((batch * frames, 1, self.channels * 3, height * width))?
            .permute((0, 1, 3, 2))?
            .contiguous()?;
        let pieces = qkv.chunk(3, 3)?;
        let q = &pieces[0];
        let k = &pieces[1];
        let v = &pieces[2];
        let attended = attention(q, k, v, false, 0)?;
        let attended = attended
            .squeeze(1)?
            .permute((0, 2, 1))?
            .contiguous()?
            .reshape((batch * frames, self.channels, height, width))?;
        let attended = self.proj.forward(&attended)?;
        let attended = attended
            .reshape((batch, frames, self.channels, height, width))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?;
        attended + identity
    }
}

struct MidBlock {
    first: ResidualBlock,
    attention: AttentionBlock,
    second: ResidualBlock,
}

impl MidBlock {
    fn load(vb: VarBuilder, channels: usize) -> Result<Self> {
        Ok(Self {
            first: ResidualBlock::load(vb.pp("resnets").pp("0"), channels, channels)?,
            attention: AttentionBlock::load(vb.pp("attentions").pp("0"), channels)?,
            second: ResidualBlock::load(vb.pp("resnets").pp("1"), channels, channels)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.first.forward(x)?;
        let x = self.attention.forward(&x)?;
        self.second.forward(&x)
    }
}

struct ResidualDownBlock {
    resnets: Vec<ResidualBlock>,
    downsampler: Option<CausalConv2d>,
    in_dim: usize,
    out_dim: usize,
    factor_t: usize,
    factor_s: usize,
}

impl ResidualDownBlock {
    fn load(
        vb: VarBuilder,
        in_dim: usize,
        out_dim: usize,
        num_res_blocks: usize,
        temporal_downsample: bool,
        down_flag: bool,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_res_blocks);
        let mut current = in_dim;
        for index in 0..num_res_blocks {
            resnets.push(ResidualBlock::load(
                vb.pp("resnets").pp(index.to_string()),
                current,
                out_dim,
            )?);
            current = out_dim;
        }
        let factor_t = if temporal_downsample { 2 } else { 1 };
        let factor_s = if down_flag { 2 } else { 1 };
        let downsampler = down_flag
            .then(|| {
                CausalConv2d::load(
                    vb.pp("downsampler").pp("resample").pp("1"),
                    out_dim,
                    out_dim,
                    3,
                    2,
                    0,
                )
            })
            .transpose()?;
        Ok(Self {
            resnets,
            downsampler,
            in_dim,
            out_dim,
            factor_t,
            factor_s,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shortcut = avg_down_2d(x, self.in_dim, self.out_dim, self.factor_t, self.factor_s)?;
        let mut x = x.clone();
        for resnet in &self.resnets {
            x = resnet.forward(&x)?;
        }
        if let Some(downsampler) = &self.downsampler {
            x = downsample_2d(downsampler, &x)?;
        }
        x + shortcut
    }
}

struct Encoder {
    conv_in: CausalConv2d,
    down_blocks: Vec<ResidualDownBlock>,
    mid_block: MidBlock,
    norm_out: ChannelRmsNorm,
    conv_out: CausalConv2d,
}

impl Encoder {
    fn load(vb: VarBuilder, config: &PublicVaeConfig) -> Result<Self> {
        let dims: Vec<usize> = std::iter::once(1)
            .chain(config.dim_mult.iter().copied())
            .map(|mult| config.base_dim * mult)
            .collect();
        let conv_in = CausalConv2d::load(vb.pp("conv_in"), config.in_channels, dims[0], 3, 1, 1)?;
        let mut down_blocks = Vec::with_capacity(config.dim_mult.len());
        for index in 0..config.dim_mult.len() {
            down_blocks.push(ResidualDownBlock::load(
                vb.pp("down_blocks").pp(index.to_string()),
                dims[index],
                dims[index + 1],
                config.num_res_blocks,
                if index + 1 < config.dim_mult.len() {
                    config.temperal_downsample[index]
                } else {
                    false
                },
                index + 1 < config.dim_mult.len(),
            )?);
        }
        let last_dim = *dims.last().expect("dim_mult is validated non-empty");
        Ok(Self {
            conv_in,
            down_blocks,
            mid_block: MidBlock::load(vb.pp("mid_block"), last_dim)?,
            norm_out: ChannelRmsNorm::load(vb.pp("norm_out"), last_dim, false)?,
            conv_out: CausalConv2d::load(vb.pp("conv_out"), last_dim, config.z_dim * 2, 3, 1, 1)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_in.forward(x)?;
        for block in &self.down_blocks {
            x = block.forward(&x)?;
        }
        x = self.mid_block.forward(&x)?;
        x = candle_nn::ops::silu(&self.norm_out.forward(&x)?)?;
        self.conv_out.forward(&x)
    }
}

struct ResidualUpBlock {
    resnets: Vec<ResidualBlock>,
    upsampler: Option<SpatialUpsampler>,
    in_dim: usize,
    out_dim: usize,
    factor_t: usize,
    factor_s: usize,
}

impl ResidualUpBlock {
    fn load(
        vb: VarBuilder,
        in_dim: usize,
        out_dim: usize,
        num_res_blocks: usize,
        temporal_upsample: bool,
        up_flag: bool,
    ) -> Result<Self> {
        let resnet_count = num_res_blocks.checked_add(1).ok_or_else(|| {
            candle_core::Error::Msg("Qwen-Image VAE residual block count overflows usize".into())
        })?;
        let mut resnets = Vec::with_capacity(resnet_count);
        let mut current = in_dim;
        for index in 0..resnet_count {
            resnets.push(ResidualBlock::load(
                vb.pp("resnets").pp(index.to_string()),
                current,
                out_dim,
            )?);
            current = out_dim;
        }
        let factor_t = if temporal_upsample { 2 } else { 1 };
        let factor_s = if up_flag { 2 } else { 1 };
        let upsampler = up_flag
            .then(|| SpatialUpsampler::load(vb.pp("upsampler"), out_dim, out_dim))
            .transpose()?;
        Ok(Self {
            resnets,
            upsampler,
            in_dim,
            out_dim,
            factor_t,
            factor_s,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shortcut = self
            .upsampler
            .is_some()
            .then(|| dup_up_2d(x, self.in_dim, self.out_dim, self.factor_t, self.factor_s))
            .transpose()?;
        let mut x = x.clone();
        for resnet in &self.resnets {
            x = resnet.forward(&x)?;
        }
        if let Some(upsampler) = &self.upsampler {
            x = upsampler.forward(&x)?;
        }
        match shortcut {
            Some(shortcut) => x + shortcut,
            None => Ok(x),
        }
    }
}

struct SpatialUpsampler {
    conv: CausalConv2d,
}

impl SpatialUpsampler {
    fn load(vb: VarBuilder, in_dim: usize, out_dim: usize) -> Result<Self> {
        Ok(Self {
            conv: CausalConv2d::load(vb.pp("resample").pp("1"), in_dim, out_dim, 3, 1, 1)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, channels, frames, height, width) = x.dims5()?;
        if frames != 1 {
            candle_core::bail!(
                "Qwen-Image VAE upsampling T=1 specialization received {} frames",
                frames
            )
        }
        let flat = x.permute((0, 2, 1, 3, 4))?.contiguous()?.reshape((
            batch * frames,
            channels,
            height,
            width,
        ))?;
        // The Python reference explicitly promotes nearest-exact interpolation
        // to float32 and restores the input dtype before its convolution.
        let flat = flat
            .to_dtype(DType::F32)?
            .upsample_nearest2d(height * 2, width * 2)?
            .to_dtype(x.dtype())?;
        let flat = self.conv.forward(&flat)?;
        let out_channels = flat.dim(1)?;
        flat.reshape((batch, frames, out_channels, height * 2, width * 2))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()
    }
}

struct Decoder {
    conv_in: CausalConv2d,
    mid_block: MidBlock,
    up_blocks: Vec<ResidualUpBlock>,
    norm_out: ChannelRmsNorm,
    conv_out: CausalConv2d,
}

impl Decoder {
    fn load(vb: VarBuilder, config: &PublicVaeConfig) -> Result<Self> {
        let decoder_base_dim = config.decoder_base_dim.unwrap_or(config.base_dim);
        let mut dims: Vec<usize> = Vec::with_capacity(config.dim_mult.len() + 1);
        dims.push(decoder_base_dim * config.dim_mult[config.dim_mult.len() - 1]);
        for mult in config.dim_mult.iter().rev() {
            dims.push(decoder_base_dim * mult);
        }
        let conv_in = CausalConv2d::load(vb.pp("conv_in"), config.z_dim, dims[0], 3, 1, 1)?;
        let mid_block = MidBlock::load(vb.pp("mid_block"), dims[0])?;
        let temporal_upsample: Vec<bool> =
            config.temperal_downsample.iter().rev().copied().collect();
        let mut up_blocks = Vec::with_capacity(config.dim_mult.len().saturating_sub(1));
        for index in 0..dims.len() - 1 {
            // The source decoder has one residual block at every level and
            // upsamples only the first `dim_mult.len() - 1` levels.  The last
            // block changes 288 -> 144 channels without a spatial upsample.
            let up_flag = index + 1 < config.dim_mult.len();
            up_blocks.push(ResidualUpBlock::load(
                vb.pp("up_blocks").pp(index.to_string()),
                dims[index],
                dims[index + 1],
                config.num_res_blocks,
                if up_flag {
                    temporal_upsample[index]
                } else {
                    false
                },
                up_flag,
            )?);
        }
        let last_dim = *dims.last().expect("dim_mult is validated non-empty");
        Ok(Self {
            conv_in,
            mid_block,
            up_blocks,
            norm_out: ChannelRmsNorm::load(vb.pp("norm_out"), last_dim, false)?,
            conv_out: CausalConv2d::load(
                vb.pp("conv_out"),
                last_dim,
                config.out_channels,
                3,
                1,
                1,
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_in.forward(x)?;
        x = self.mid_block.forward(&x)?;
        for block in &self.up_blocks {
            x = block.forward(&x)?;
        }
        x = candle_nn::ops::silu(&self.norm_out.forward(&x)?)?;
        self.conv_out.forward(&x)
    }
}

/// The Qwen-Image-2.1 VAE, restricted to the exact single-image (`T = 1`) path.
pub struct QwenImageVae {
    config: PublicVaeConfig,
    encoder: Option<Encoder>,
    quant_conv: Option<CausalConv2d>,
    post_quant_conv: CausalConv2d,
    decoder: Decoder,
}

impl QwenImageVae {
    /// Load Qwen-Image-2.1 VAE weights from the `vae/` safetensors namespace.
    pub fn load(vb: VarBuilder, config: PublicVaeConfig) -> Result<Self> {
        validate_vb_dtype(&vb)?;
        config.validate()?;
        let encoder = Encoder::load(vb.pp("encoder"), &config)?;
        let quant_conv = CausalConv2d::load(
            vb.pp("quant_conv"),
            config.z_dim * 2,
            config.z_dim * 2,
            1,
            1,
            0,
        )?;
        let (post_quant_conv, decoder) = Self::load_decoder_parts(vb, &config)?;
        Ok(Self {
            config,
            encoder: Some(encoder),
            quant_conv: Some(quant_conv),
            post_quant_conv,
            decoder,
        })
    }

    /// Load only the decoder-side VAE weights.
    ///
    /// Text-to-image generation never calls `encode`, so this constructor
    /// avoids retaining the encoder/quantizer subtree.
    /// The supplied [`VarBuilder`] may therefore be backed by a decoder-only
    /// namespace or by a lazy snapshot containing the full VAE namespace.
    pub fn load_decoder(vb: VarBuilder, config: PublicVaeConfig) -> Result<Self> {
        validate_vb_dtype(&vb)?;
        config.validate()?;
        let (post_quant_conv, decoder) = Self::load_decoder_parts(vb, &config)?;
        Ok(Self {
            config,
            encoder: None,
            quant_conv: None,
            post_quant_conv,
            decoder,
        })
    }

    fn load_decoder_parts(
        vb: VarBuilder,
        config: &PublicVaeConfig,
    ) -> Result<(CausalConv2d, Decoder)> {
        let post_quant_conv = CausalConv2d::load(
            vb.pp("post_quant_conv"),
            config.z_dim,
            config.z_dim,
            1,
            1,
            0,
        )?;
        let decoder = Decoder::load(vb.pp("decoder"), config)?;
        Ok((post_quant_conv, decoder))
    }

    /// Return the validated model configuration.
    pub fn config(&self) -> &PublicVaeConfig {
        &self.config
    }

    /// Number of channels in one raw VAE latent.
    pub fn z_dim(&self) -> usize {
        self.config.z_dim
    }

    /// Spatial image-to-latent compression factor (16 for Qwen-Image-2.1).
    pub fn spatial_factor(&self) -> usize {
        self.config.scale_factor_spatial.unwrap_or(16)
    }

    /// Per-channel mean used by the surrounding diffusion pipeline.
    pub fn latents_mean(&self) -> &[f32] {
        &self.config.latents_mean
    }

    /// Per-channel standard deviation used by the surrounding diffusion pipeline.
    pub fn latents_std(&self) -> &[f32] {
        &self.config.latents_std
    }

    /// Encode `[B, in_channels, H, W]` into `[B, 2*z_dim, H/16, W/16]` moments.
    ///
    /// The first `z_dim` channels are the posterior mean and the second half
    /// are log-variance moments.  This is the raw VAE output; call
    /// [`Self::normalize_latents`] on the posterior mean before feeding it to
    /// the diffusion transformer.
    pub fn encode(&self, image: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = image.dims4()?;
        if channels != self.config.in_channels {
            candle_core::bail!(
                "Qwen-Image VAE encode expected {} channels, got {}",
                self.config.in_channels,
                channels
            )
        }
        check_spatial_divisibility(height, width, self.spatial_factor())?;
        let image = image.unsqueeze(2)?;
        let encoder = self.encoder.as_ref().ok_or_else(|| {
            candle_core::Error::Msg(
                "Qwen-Image VAE was loaded decoder-only; encode is unavailable".to_string(),
            )
        })?;
        let encoded = encoder.forward(&image)?;
        let quant_conv = self.quant_conv.as_ref().ok_or_else(|| {
            candle_core::Error::Msg(
                "Qwen-Image VAE was loaded decoder-only; encode is unavailable".to_string(),
            )
        })?;
        let moments = quant_conv.forward(&encoded)?;
        moments.squeeze(2)?.reshape((
            batch,
            self.config.z_dim * 2,
            height / self.spatial_factor(),
            width / self.spatial_factor(),
        ))
    }

    /// Encode and select the deterministic posterior mode `[B, z_dim, H/16, W/16]`.
    pub fn encode_mode(&self, image: &Tensor) -> Result<Tensor> {
        self.encode(image)?.narrow(1, 0, self.config.z_dim)
    }

    /// Decode raw VAE latents `[B, z_dim, h, w]` into clamped RGBA images
    /// `[B, out_channels, h*16, w*16]`.
    ///
    /// The input must be denormalized VAE latents.  Diffusion latents are
    /// normally normalized as `(raw - latents_mean) / latents_std`; use
    /// [`Self::denormalize_latents`] first when starting from that form.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = latents.dims4()?;
        if channels != self.config.z_dim {
            candle_core::bail!(
                "Qwen-Image VAE decode expected {} latent channels, got {}",
                self.config.z_dim,
                channels
            )
        }
        let latents = self.post_quant_conv.forward(&latents.unsqueeze(2)?)?;
        let decoded = self.decoder.forward(&latents)?.squeeze(2)?;
        let decoded = clamp_to_unit(&decoded)?;
        decoded.reshape((
            batch,
            self.config.out_channels,
            height * self.spatial_factor(),
            width * self.spatial_factor(),
        ))
    }

    /// Normalize raw VAE latents for the diffusion transformer.
    pub fn normalize_latents(&self, latents: &Tensor) -> Result<Tensor> {
        self.latent_affine(latents, false)
    }

    /// Reverse the diffusion-pipeline normalization before [`Self::decode`].
    pub fn denormalize_latents(&self, latents: &Tensor) -> Result<Tensor> {
        self.latent_affine(latents, true)
    }

    fn latent_affine(&self, latents: &Tensor, denormalize: bool) -> Result<Tensor> {
        let channels = latents.dim(1)?;
        if channels != self.config.z_dim {
            candle_core::bail!(
                "Qwen-Image latent affine expected {} channels, got {}",
                self.config.z_dim,
                channels
            )
        }
        let shape = match latents.rank() {
            4 => vec![1, self.config.z_dim, 1, 1],
            5 => vec![1, self.config.z_dim, 1, 1, 1],
            rank => candle_core::bail!("Qwen-Image latent affine expects rank 4 or 5, got {rank}"),
        };
        let mean = Tensor::from_vec(
            self.config.latents_mean.clone(),
            shape.clone(),
            latents.device(),
        )?
        .to_dtype(latents.dtype())?;
        let scale = Tensor::from_vec(self.config.latents_std.clone(), shape, latents.device())?
            .to_dtype(latents.dtype())?;
        if denormalize {
            latents.broadcast_mul(&scale)?.broadcast_add(&mean)
        } else {
            latents.broadcast_sub(&mean)?.broadcast_div(&scale)
        }
    }
}

fn validate_vb_dtype(vb: &VarBuilder) -> Result<()> {
    if !matches!(vb.dtype(), DType::BF16 | DType::F32) {
        candle_core::bail!(
            "Qwen-Image VAE requires BF16 or F32 weights; refusing {:?}",
            vb.dtype()
        )
    }
    Ok(())
}

fn check_spatial_divisibility(height: usize, width: usize, factor: usize) -> Result<()> {
    if height == 0 || width == 0 || !height.is_multiple_of(factor) || !width.is_multiple_of(factor)
    {
        candle_core::bail!(
            "Qwen-Image VAE spatial dimensions must be positive multiples of {} (got {}x{})",
            factor,
            height,
            width
        )
    }
    Ok(())
}

/// Clamp on-device to the reference `[-1, 1]` range while preserving NaNs.
fn clamp_to_unit(x: &Tensor) -> Result<Tensor> {
    let upper = x.ones_like()?;
    let lower = upper.neg()?;
    // Explicit selection preserves NaNs: Metal's MAX/MIN ternaries treat a
    // NaN comparison as false and would silently replace it with a bound.
    let below = x.lt(&lower)?;
    let lower_clamped = below.where_cond(&lower, x)?;
    let above = lower_clamped.gt(&upper)?;
    above.where_cond(&upper, &lower_clamped)
}

fn pad_2d(x: &Tensor, left: usize, right: usize, top: usize, bottom: usize) -> Result<Tensor> {
    x.pad_with_zeros(2, top, bottom)?
        .pad_with_zeros(3, left, right)
}

fn downsample_2d(conv: &CausalConv2d, x: &Tensor) -> Result<Tensor> {
    let (batch, channels, frames, height, width) = x.dims5()?;
    if frames != 1 || !height.is_multiple_of(2) || !width.is_multiple_of(2) {
        candle_core::bail!("invalid Qwen-Image spatial downsample shape {:?}", x.dims())
    }
    let flat = x.permute((0, 2, 1, 3, 4))?.contiguous()?.reshape((
        batch * frames,
        channels,
        height,
        width,
    ))?;
    let flat = pad_2d(&flat, 0, 1, 0, 1)?;
    let flat = conv.forward(&flat)?;
    let out_channels = flat.dim(1)?;
    flat.reshape((batch, frames, out_channels, height / 2, width / 2))?
        .permute((0, 2, 1, 3, 4))?
        .contiguous()
}

fn avg_down_2d(
    x: &Tensor,
    in_channels: usize,
    out_channels: usize,
    factor_t: usize,
    factor_s: usize,
) -> Result<Tensor> {
    let (batch, channels, frames, height, width) = x.dims5()?;
    if channels != in_channels
        || frames != 1
        || !height.is_multiple_of(factor_s)
        || !width.is_multiple_of(factor_s)
    {
        candle_core::bail!("invalid Qwen-Image average-downsample shape {:?}", x.dims())
    }
    let x = if factor_t == 2 {
        x.pad_with_zeros(2, 1, 0)?
    } else {
        x.clone()
    };
    let temporal = x.dim(2)?;
    let h = height / factor_s;
    let w = width / factor_s;
    let grouped = x
        .reshape(&[
            batch,
            channels,
            temporal / factor_t,
            factor_t,
            h,
            factor_s,
            w,
            factor_s,
        ])?
        .permute(&[0, 1, 3, 5, 7, 2, 4, 6][..])?
        .contiguous()?
        .reshape(&[
            batch,
            channels * factor_t * factor_s * factor_s,
            temporal / factor_t,
            h,
            w,
        ])?;
    let grouped_channels = channels * factor_t * factor_s * factor_s;
    if !grouped_channels.is_multiple_of(out_channels) {
        candle_core::bail!("invalid Qwen-Image average-downsample channel geometry")
    }
    let group_size = grouped_channels / out_channels;
    grouped
        .reshape(&[batch, out_channels, group_size, temporal / factor_t, h, w])?
        .mean(2)
}

fn dup_up_2d(
    x: &Tensor,
    in_channels: usize,
    out_channels: usize,
    factor_t: usize,
    factor_s: usize,
) -> Result<Tensor> {
    let (batch, channels, frames, height, width) = x.dims5()?;
    if channels != in_channels || frames != 1 {
        candle_core::bail!("invalid Qwen-Image duplicate-upsample shape {:?}", x.dims())
    }
    let factor = factor_t * factor_s * factor_s;
    if !(out_channels * factor).is_multiple_of(in_channels) {
        candle_core::bail!("invalid Qwen-Image duplicate-upsample channel geometry")
    }
    let repeats = out_channels * factor / in_channels;
    let repeated = x.unsqueeze(2)?.repeat((1, 1, repeats, 1, 1, 1))?.reshape((
        batch,
        channels * repeats,
        frames,
        height,
        width,
    ))?;
    let expanded = repeated
        .reshape(&[
            batch,
            out_channels,
            factor_t,
            factor_s,
            factor_s,
            frames,
            height,
            width,
        ])?
        .permute(&[0, 1, 5, 2, 6, 3, 7, 4][..])?
        .contiguous()?
        .reshape(&[
            batch,
            out_channels,
            frames * factor_t,
            height * factor_s,
            width * factor_s,
        ])?;
    expanded.narrow(2, factor_t - 1, 1)
}

#[cfg(test)]
#[path = "vae_tests.rs"]
mod vae_tests;
