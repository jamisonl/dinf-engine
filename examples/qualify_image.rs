//! Full Qwen Image pipeline replay, cache and cancellation checks.
//! Usage: SNAPSHOT cpu|metal OUTPUT_DIR [STEPS=4]. Admit component memory first.
//! Uses 512x512 output. This checks repeatability, not image quality against an
//! independent implementation; a Rust seed does not match PyTorch's noise.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, ensure};
use candle_core::Device;
use dinf_engine::diffusion::{DiffusionProgress, ImageGenerationOptions, QwenImagePipeline};
use serde_json::json;
use sha2::{Digest, Sha256};

const PROMPT: &str = "A red ceramic teapot on a blue table, with a yellow wall behind it.";

fn spatially_nonconstant(image: &image::RgbaImage) -> bool {
    image
        .pixels()
        .next()
        .is_some_and(|first| image.pixels().any(|pixel| pixel.0[..3] != first.0[..3]))
}

fn render(
    pipeline: &QwenImagePipeline,
    options: &ImageGenerationOptions,
    path: &Path,
) -> Result<Vec<u8>> {
    let cancelled = AtomicBool::new(false);
    let mut completed = Vec::new();
    let result = pipeline.generate(PROMPT, options, &cancelled, |progress| {
        if let DiffusionProgress::Denoising { completed: n, .. } = progress {
            completed.push(n);
        }
    })?;
    ensure!(
        completed == (0..=options.steps).collect::<Vec<_>>(),
        "missing denoising steps"
    );
    ensure!(
        (result.width, result.height, result.seed) == (512, 512, options.seed),
        "wrong result metadata"
    );
    // The pipeline rejects non-finite VAE outputs before PNG conversion.
    let image = image::load_from_memory(&result.png)?.to_rgba8();
    ensure!(image.dimensions() == (512, 512), "wrong PNG dimensions");
    ensure!(
        spatially_nonconstant(&image),
        "spatially constant RGB image"
    );
    std::fs::write(path, result.png)?;
    Ok(image.into_raw())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        (3..=4).contains(&args.len()),
        "usage: SNAPSHOT cpu|metal OUTPUT_DIR [STEPS=4]"
    );
    let device = match args[1].as_str() {
        "cpu" => Device::Cpu,
        #[cfg(target_os = "macos")]
        "metal" => Device::new_metal(0)?,
        _ => anyhow::bail!("backend unavailable"),
    };
    let output = Path::new(&args[2]);
    ensure!(!output.exists(), "output directory already exists");
    let mut options = ImageGenerationOptions {
        width: 512,
        height: 512,
        steps: args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(4),
        seed: 42,
        ..Default::default()
    };
    ensure!(
        options.steps >= 2,
        "cache qualification needs at least two denoising steps"
    );
    let pipeline = QwenImagePipeline::from_snapshot(&args[0], device.clone())?;
    let transformer_config: serde_json::Value = serde_json::from_reader(std::fs::File::open(
        Path::new(&args[0]).join("transformer/config.json"),
    )?)?;
    ensure!(
        transformer_config["causal_condition"] == true,
        "this cache qualification requires causal conditioning"
    );
    pipeline.validate_request(PROMPT, &options)?;
    std::fs::create_dir_all(output)?;
    let baseline = render(&pipeline, &options, &output.join("baseline.png"))?;
    let replay = render(&pipeline, &options, &output.join("replay.png"))?;
    ensure!(baseline == replay, "seeded replay changed pixels");
    options.cache_conditioning = false;
    let uncached = render(&pipeline, &options, &output.join("uncached.png"))?;
    ensure!(baseline == uncached, "conditioning cache changed pixels");
    options.cache_conditioning = true;
    let cancelled = AtomicBool::new(false);
    let mut cancelled_after_step = false;
    let result = pipeline.generate(PROMPT, &options, &cancelled, |progress| {
        if matches!(progress, DiffusionProgress::Denoising { completed: 1, .. }) {
            cancelled_after_step = true;
            cancelled.store(true, Ordering::Relaxed);
        }
    });
    ensure!(cancelled_after_step, "cancel check never reached denoising");
    ensure!(
        result
            .err()
            .is_some_and(|e| e.to_string().contains("cancelled")),
        "cancelled generation did not return the cancellation error"
    );
    device.synchronize()?;
    let after_cancel = render(&pipeline, &options, &output.join("after-cancel.png"))?;
    ensure!(
        baseline == after_cancel,
        "cancelled request changed later pixels"
    );
    let report = json!({"passed":true,"snapshot":args[0],"backend":args[1],
        "width":512,"height":512,"steps":options.steps,"seed":options.seed,
        "checks":["seeded_replay","conditioning_cache","mid_denoise_cancel","replay_after_cancel"],
        "pixel_sha256":format!("{:x}",Sha256::digest(&baseline)),
        "independent_quality_oracle":false});
    std::fs::write(
        output.join("result.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_check_rejects_a_solid_color_and_alpha_only_changes() {
        let mut image = image::RgbaImage::from_pixel(2, 1, image::Rgba([255, 0, 0, 255]));
        assert!(!spatially_nonconstant(&image));
        image.put_pixel(1, 0, image::Rgba([255, 0, 0, 0]));
        assert!(!spatially_nonconstant(&image));
        image.put_pixel(1, 0, image::Rgba([0, 0, 255, 255]));
        assert!(spatially_nonconstant(&image));
    }
}
