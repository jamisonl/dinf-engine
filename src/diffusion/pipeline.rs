//! Staged, text-to-image execution for a local Qwen Image 2.1 snapshot.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use tokenizers::Tokenizer;

use super::scheduler::{FlowMatchEuler, SchedulerConfig};
use super::text_encoder::{TextEncoder, TextEncoderConfig};
use super::transformer::{
    ImageShape, KvCacheMode, QwenImageTransformer, TransformerConfig, TransformerKvCache,
};
use super::vae::{PublicVaeConfig, QwenImageVae};
use super::weights::{load_component, read_json};

const SYSTEM_PREFIX: &str =
    "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
const MAX_PROMPT_BYTES: usize = 32 * 1024;
const MAX_PROMPT_TOKENS: usize = 4096;

/// Native research-runtime controls. Sampler controls are internal and are not
/// additions to the public OpenAI Images request schema.
#[derive(Clone, Debug)]
pub struct ImageGenerationOptions {
    /// Output width, a multiple of 32 in 32..=4096.
    pub width: u32,
    /// Output height, a multiple of 32 in 32..=4096.
    pub height: u32,
    /// Euler steps (the reference default is 40).
    pub steps: usize,
    /// Request-local Rust RNG seed; not bit-identical to a PyTorch RNG seed.
    pub seed: u64,
    /// Promote all weights/activations to F32. Never reduce precision.
    pub force_f32: bool,
    /// Reuse the exact static conditioning K/V between denoising steps.
    pub cache_conditioning: bool,
}

impl Default for ImageGenerationOptions {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            steps: 40,
            seed: 0,
            force_f32: false,
            cache_conditioning: true,
        }
    }
}

/// Progress events for callers that report generation progress.
#[derive(Clone, Copy, Debug)]
pub enum DiffusionProgress {
    /// Loading the native condition encoder.
    LoadingTextEncoder,
    /// Computing prompt conditioning.
    EncodingPrompt,
    /// Loading the native diffusion transformer.
    LoadingTransformer,
    /// A denoising iteration completed.
    Denoising {
        /// Completed iterations.
        completed: usize,
        /// Total iterations.
        total: usize,
    },
    /// Loading the VAE decoder.
    LoadingDecoder,
    /// Decoding latents into RGBA pixels.
    Decoding,
    /// Encoding the decoded pixels as PNG.
    EncodingImage,
}

/// A complete, in-memory PNG result. No public URL or persistence is implied.
pub struct GeneratedImage {
    /// PNG-encoded RGBA bytes.
    pub png: Vec<u8>,
    /// Output width.
    pub width: u32,
    /// Output height.
    pub height: u32,
    /// Request-local seed, for repeating this Rust runtime's noise.
    pub seed: u64,
}

/// Native single-image research runner. A snapshot must contain the released
/// processor, text_encoder, transformer, scheduler and vae directories.
///
/// Components are loaded sequentially and released at stage boundaries. This
/// is not layer/disk offloading and makes no claim of fitting in 12 GB. Keep
/// snapshot files immutable while generating. Weight authenticity/licensing
/// must be verified by the operator; this runner never executes remote code.
pub struct QwenImagePipeline {
    snapshot: PathBuf,
    device: Device,
    tokenizer: Tokenizer,
    text_config: TextEncoderConfig,
    transformer_config: TransformerConfig,
    vae_config: PublicVaeConfig,
    scheduler_config: SchedulerConfig,
}

impl QwenImagePipeline {
    /// Inspect local configurations/tokenizer without loading tensor weights.
    pub fn from_snapshot(snapshot: impl AsRef<Path>, device: Device) -> Result<Self> {
        let snapshot = snapshot.as_ref().to_owned();
        let transformer_path = snapshot.join("transformer/config.json");
        let vae_path = snapshot.join("vae/config.json");
        let text_path = snapshot.join("text_encoder/config.json");
        let scheduler_path = snapshot.join("scheduler/scheduler_config.json");
        check_class(&transformer_path, "QwenImage21Transformer2DModel")?;
        check_class(&vae_path, "AutoencoderKLQwenImage21")?;
        check_class(&scheduler_path, "FlowMatchEulerDiscreteScheduler")?;
        let text_value: serde_json::Value = read_json(&text_path)?;
        ensure!(
            text_value["model_type"] == "qwen3_vl",
            "snapshot requires the Qwen3-VL condition encoder"
        );
        let text_config: TextEncoderConfig =
            serde_json::from_value(text_value).context("Qwen3-VL config")?;
        let transformer_config: TransformerConfig = read_json(&transformer_path)?;
        let vae_config: PublicVaeConfig = read_json(&vae_path)?;
        let scheduler_config: SchedulerConfig = read_json(&scheduler_path)?;
        text_config.text_config.validate()?;
        transformer_config.validate()?;
        vae_config.validate()?;
        FlowMatchEuler::new(&scheduler_config, 40, 4096)?;
        ensure!(
            transformer_config.patch_size == 1
                && transformer_config.in_channels == vae_config.z_dim
                && transformer_config
                    .out_channels
                    .unwrap_or(transformer_config.in_channels)
                    == vae_config.z_dim
                && vae_config.scale_factor_spatial == Some(16)
                && vae_config.out_channels == 4
                && text_config.text_config.hidden_size == transformer_config.context_in_dim,
            "incompatible diffusion component dimensions"
        );
        let tokenizer_path = snapshot.join("processor/tokenizer.json");
        ensure!(
            std::fs::metadata(&tokenizer_path)?.len() <= 64 * 1024 * 1024,
            "oversized tokenizer metadata"
        );
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
        // No truncation/padding: never silently discard user prompt tokens.
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("configure tokenizer: {e}"))?;
        tokenizer.with_padding(None);
        Ok(Self {
            snapshot,
            device,
            tokenizer,
            text_config,
            transformer_config,
            vae_config,
            scheduler_config,
        })
    }

    /// Validate a text-to-image request, including tokenizer limits, without
    /// loading tensor weights or reserving a generation slot.
    pub fn validate_request(&self, prompt: &str, options: &ImageGenerationOptions) -> Result<()> {
        validate_options(prompt, options)?;
        prompt_tokens(&self.tokenizer, prompt)?;
        FlowMatchEuler::new(
            &self.scheduler_config,
            options.steps,
            options.width as usize / 16 * (options.height as usize / 16),
        )?;
        Ok(())
    }

    /// Generate one image, checking cancellation between stages and steps.
    /// Cancellation returns an error, never a partially denoised success.
    /// Calls are synchronous; async callers should run them on a bounded
    /// blocking thread pool.
    pub fn generate(
        &self,
        prompt: &str,
        options: &ImageGenerationOptions,
        cancelled: &AtomicBool,
        mut progress: impl FnMut(DiffusionProgress),
    ) -> Result<GeneratedImage> {
        validate_options(prompt, options)?;
        check_cancelled(cancelled)?;
        let (ids, drop_index) = prompt_tokens(&self.tokenizer, prompt)?;
        let h = options.height as usize / 16;
        let w = options.width as usize / 16;
        let schedule = FlowMatchEuler::new(&self.scheduler_config, options.steps, h * w)?;
        // Candle's portable CPU matmul does not accept BF16. Diagnostic CPU
        // execution explicitly promotes components to F32; Metal keeps BF16.
        let force_f32 = options.force_f32 || self.device.is_cpu();

        progress(DiffusionProgress::LoadingTextEncoder);
        let conditioning = {
            let vb = load_component(&self.snapshot.join("text_encoder"), &self.device, force_f32)?;
            let encoder = TextEncoder::load(vb, self.text_config.clone())?;
            check_cancelled(cancelled)?;
            progress(DiffusionProgress::EncodingPrompt);
            let ids = Tensor::from_vec(ids.clone(), (1, ids.len()), &self.device)?;
            let hidden = encoder.forward(&ids)?;
            let conditioning = hidden
                .narrow(1, drop_index, hidden.dim(1)? - drop_index)?
                .contiguous()?;
            self.device.synchronize()?;
            conditioning
        };
        // Metal's allocator releases unused pooled buffers on synchronization.
        // Fence after the encoder has been dropped, not only while it is live,
        // so the next component does not inherit its multi-GB weight residency.
        self.device.synchronize()?;
        check_cancelled(cancelled)?;

        progress(DiffusionProgress::LoadingTransformer);
        let latents = {
            // Never downcast the encoder's conditioning if it was promoted.
            let vb = load_component(
                &self.snapshot.join("transformer"),
                &self.device,
                force_f32 || conditioning.dtype() == DType::F32,
            )?;
            let dtype = vb.dtype();
            let transformer = QwenImageTransformer::load(vb, self.transformer_config.clone())?;
            let conditioning = conditioning.to_dtype(dtype)?;
            let channels = self.transformer_config.in_channels;
            let mut latents = initial_noise(options.seed, channels, h, w, dtype, &self.device)?;
            let mut mask = vec![false; conditioning.dim(1)?];
            mask.extend(std::iter::repeat_n(true, h * w / 4));
            let shapes = [ImageShape::new(1, h, w)];
            let mut cache = TransformerKvCache::new(transformer.num_layers());
            progress(DiffusionProgress::Denoising {
                completed: 0,
                total: schedule.steps(),
            });
            for step in 0..schedule.steps() {
                check_cancelled(cancelled)?;
                // The reference rounds the training timestep to the latent
                // dtype before dividing by 1000.
                let timestep = (Tensor::new(&[schedule.timestep(step)?], &self.device)?
                    .to_dtype(dtype)?
                    / 1000.)?;
                let mode = if options.cache_conditioning && self.transformer_config.causal_condition
                {
                    Some(if step == 0 {
                        KvCacheMode::Extract
                    } else {
                        KvCacheMode::Cached
                    })
                } else {
                    None
                };
                let prediction = transformer.forward(
                    &latents,
                    &conditioning,
                    &timestep,
                    &shapes,
                    &mask,
                    None,
                    mode.map(|_| &mut cache),
                    mode,
                )?;
                let prediction = prediction.narrow(1, prediction.dim(1)? - h * w, h * w)?;
                latents = schedule.step(step, &latents, &prediction)?;
                // Fence before progress/cancel and before releasing a stage's
                // weights; GPU work must not accumulate unboundedly.
                self.device.synchronize()?;
                progress(DiffusionProgress::Denoising {
                    completed: step + 1,
                    total: schedule.steps(),
                });
            }
            latents
                .transpose(1, 2)?
                .reshape((1, channels, h, w))?
                .contiguous()?
        };
        drop(conditioning);
        self.device.synchronize()?;
        check_cancelled(cancelled)?;

        progress(DiffusionProgress::LoadingDecoder);
        let vb = load_component(
            &self.snapshot.join("vae"),
            &self.device,
            force_f32 || latents.dtype() == DType::F32,
        )?;
        let dtype = vb.dtype();
        let vae = QwenImageVae::load_decoder(vb, self.vae_config.clone())?;
        check_cancelled(cancelled)?;
        progress(DiffusionProgress::Decoding);
        let raw_latents = vae.denormalize_latents(&latents.to_dtype(dtype)?)?;
        let decoded = vae.decode(&raw_latents)?;
        self.device.synchronize()?;
        check_cancelled(cancelled)?;
        progress(DiffusionProgress::EncodingImage);
        let png = encode_rgba_png(&decoded, options.width, options.height)?;
        Ok(GeneratedImage {
            png,
            width: options.width,
            height: options.height,
            seed: options.seed,
        })
    }
}

fn check_class(path: &Path, class: &str) -> Result<()> {
    let value: serde_json::Value = read_json(path)?;
    ensure!(
        value["_class_name"] == class,
        "unsupported component class in {} (expected {class})",
        path.display()
    );
    Ok(())
}

fn validate_options(prompt: &str, options: &ImageGenerationOptions) -> Result<()> {
    ensure!(
        !prompt.trim().is_empty() && prompt.len() <= MAX_PROMPT_BYTES,
        "prompt must contain text and be at most {MAX_PROMPT_BYTES} UTF-8 bytes"
    );
    for reserved in [
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
    ] {
        ensure!(
            !prompt.contains(reserved),
            "vision special tokens require the image-edit conditioning path, which is not implemented yet"
        );
    }
    ensure!(
        [options.width, options.height]
            .iter()
            .all(|v| (32..=4096).contains(v) && v % 32 == 0),
        "image dimensions must be multiples of 32 in 32..=4096"
    );
    ensure!(
        (2..=1000).contains(&options.steps),
        "steps must be in 2..=1000"
    );
    Ok(())
}

fn check_cancelled(cancelled: &AtomicBool) -> Result<()> {
    ensure!(
        !cancelled.load(Ordering::Relaxed),
        "image generation cancelled"
    );
    Ok(())
}

fn prompt_tokens(tokenizer: &Tokenizer, prompt: &str) -> Result<(Vec<u32>, usize)> {
    // The published processor's system-only template emits precisely this
    // prefix. The full prompt is intentionally raw, not apply_chat_template.
    let system = tokenizer
        .encode(SYSTEM_PREFIX, true)
        .map_err(|e| anyhow::anyhow!("tokenize system: {e}"))?;
    let rendered =
        format!("{SYSTEM_PREFIX}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
    let encoded = tokenizer
        .encode(rendered, true)
        .map_err(|e| anyhow::anyhow!("tokenize prompt: {e}"))?;
    let ids = encoded.get_ids().to_vec();
    ensure!(
        ids.len() <= MAX_PROMPT_TOKENS,
        "encoded prompt exceeds the native runner's {MAX_PROMPT_TOKENS}-token limit"
    );
    ensure!(
        ids.starts_with(system.get_ids()) && ids.len() > system.len(),
        "tokenizer does not match Qwen Image's system prefix"
    );
    Ok((ids, system.len()))
}

fn initial_noise(
    seed: u64,
    channels: usize,
    height: usize,
    width: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mut rng = StdRng::seed_from_u64(seed);
    let values = (0..channels * height * width)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect::<Vec<f32>>();
    // Match the reference's C,H,W noise layout before packing into S,C.
    Ok(
        Tensor::from_vec(values, (1, channels, height * width), device)?
            .to_dtype(dtype)?
            .transpose(1, 2)?
            .contiguous()?,
    )
}

fn encode_rgba_png(decoded: &Tensor, width: u32, height: u32) -> Result<Vec<u8>> {
    ensure!(
        decoded.dims() == [1, 4, height as usize, width as usize],
        "VAE returned unexpected image dimensions"
    );
    let values = decoded
        .to_dtype(DType::F32)?
        .permute((0, 2, 3, 1))?
        .flatten_all()?
        .to_vec1::<f32>()?;
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "VAE returned non-finite pixels"
    );
    let bytes = values
        .into_iter()
        .map(|value| (((value / 2. + 0.5).clamp(0., 1.)) * 255.).round_ties_even() as u8)
        .collect::<Vec<_>>();
    let rgba =
        image::RgbaImage::from_raw(width, height, bytes).context("invalid VAE RGBA output")?;
    let mut png = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(rgba).write_to(&mut png, image::ImageFormat::Png)?;
    Ok(png.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_request_local_and_pack_roundtrips() -> Result<()> {
        let a = initial_noise(42, 4, 2, 2, DType::F32, &Device::Cpu)?;
        let b = initial_noise(42, 4, 2, 2, DType::F32, &Device::Cpu)?;
        let other = initial_noise(43, 4, 2, 2, DType::F32, &Device::Cpu)?;
        assert_eq!(
            a.flatten_all()?.to_vec1::<f32>()?,
            b.flatten_all()?.to_vec1::<f32>()?
        );
        assert_ne!(
            a.flatten_all()?.to_vec1::<f32>()?,
            other.flatten_all()?.to_vec1::<f32>()?
        );
        let unpacked = a.transpose(1, 2)?.reshape((1, 4, 2, 2))?;
        assert_eq!(unpacked.dims(), &[1, 4, 2, 2]);
        Ok(())
    }

    #[test]
    fn output_is_rgba_png_and_rejects_nonfinite() -> Result<()> {
        let x = Tensor::new(&[[[[-1_f32]], [[0.]], [[1.]], [[1.]]]], &Device::Cpu)?;
        let png = encode_rgba_png(&x, 1, 1)?;
        let decoded = image::load_from_memory(&png)?.to_rgba8();
        assert_eq!(decoded.get_pixel(0, 0).0, [0, 128, 255, 255]);
        assert!(
            encode_rgba_png(&Tensor::full(f32::NAN, (1, 4, 1, 1), &Device::Cpu)?, 1, 1).is_err()
        );
        Ok(())
    }

    #[test]
    fn invalid_work_is_rejected_before_loading() {
        let defaults = ImageGenerationOptions::default();
        assert!(validate_options("", &defaults).is_err());
        assert!(validate_options("<|image_pad|>", &defaults).is_err());
        assert!(
            validate_options(
                "a cat",
                &ImageGenerationOptions {
                    width: 1025,
                    ..defaults.clone()
                }
            )
            .is_err()
        );
        assert!(validate_options("a cat", &defaults).is_ok());
        assert!(check_cancelled(&AtomicBool::new(true)).is_err());
    }
}

#[cfg(test)]
#[path = "text_reference_tests.rs"]
mod text_reference_tests;
