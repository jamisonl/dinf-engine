//! Synthetic snapshot integration tests; these do not qualify real image quality.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use anyhow::Result;
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::{Init, VarBuilder, var_builder::SimpleBackend};
use serde_json::json;
use tokenizers::{
    Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::WhitespaceSplit,
};

use super::text_encoder::TextEncoder;
use super::transformer::QwenImageTransformer;
use super::vae::QwenImageVae;
use super::{DiffusionProgress, ImageGenerationOptions, QwenImagePipeline};

#[derive(Clone)]
struct RecordingFixture {
    tensors: Arc<Mutex<HashMap<String, Tensor>>>,
    nonzero: bool,
}

impl RecordingFixture {
    fn zeros() -> Self {
        Self {
            tensors: Arc::new(Mutex::new(HashMap::new())),
            nonzero: false,
        }
    }

    fn nonzero() -> Self {
        Self {
            tensors: Arc::new(Mutex::new(HashMap::new())),
            nonzero: true,
        }
    }
}

impl SimpleBackend for RecordingFixture {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _: Init,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let value = if self.nonzero {
            let values = (0..shape.elem_count())
                .map(|index| {
                    let phase = (index.wrapping_mul(7) % 19) as f32 - 9.0;
                    if name.contains("norm") || name.ends_with("gamma") {
                        1.0 + phase * 0.01
                    } else {
                        phase * 0.002
                    }
                })
                .collect::<Vec<_>>();
            Tensor::from_vec(values, shape.dims().to_vec(), device)?.to_dtype(dtype)?
        } else {
            Tensor::zeros(shape, dtype, device)?
        };
        self.tensors
            .lock()
            .unwrap()
            .insert(name.into(), value.clone());
        Ok(value)
    }
    fn get_unchecked(&self, name: &str, _: DType, _: &Device) -> candle_core::Result<Tensor> {
        candle_core::bail!("fixture must declare the shape of {name}")
    }
    fn contains_tensor(&self, _: &str) -> bool {
        true
    }
}

fn record_component(
    root: &Path,
    name: &str,
    config: serde_json::Value,
    nonzero: bool,
    load: impl FnOnce(VarBuilder<'_>, serde_json::Value) -> Result<()>,
) -> Result<()> {
    let path = root.join(name);
    std::fs::create_dir_all(&path)?;
    std::fs::write(path.join("config.json"), serde_json::to_vec(&config)?)?;
    let backend = if nonzero {
        RecordingFixture::nonzero()
    } else {
        RecordingFixture::zeros()
    };
    let vb = VarBuilder::from_backend(Box::new(backend.clone()), DType::BF16, Device::Cpu);
    load(vb, config)?;
    candle_core::safetensors::save(
        &*backend.tensors.lock().unwrap(),
        path.join("model.safetensors"),
    )?;
    Ok(())
}

fn snapshot() -> Result<tempfile::TempDir> {
    snapshot_with_weights(false)
}

fn nonzero_snapshot() -> Result<tempfile::TempDir> {
    snapshot_with_weights(true)
}

fn snapshot_with_weights(nonzero: bool) -> Result<tempfile::TempDir> {
    let dir = tempfile::tempdir()?;
    record_component(
        dir.path(),
        "text_encoder",
        json!({
            "model_type":"qwen3_vl", "text_config":{
                "attention_bias":false,"head_dim":8,"hidden_act":"silu","hidden_size":32,
                "intermediate_size":64,"max_position_embeddings":128,"num_attention_heads":4,
                "num_hidden_layers":1,"num_key_value_heads":1,"rms_norm_eps":1e-6,
                "rope_theta":5000000,"vocab_size":8
            }
        }),
        nonzero,
        |vb, config| {
            TextEncoder::load(vb, serde_json::from_value(config)?)?;
            Ok(())
        },
    )?;
    record_component(
        dir.path(),
        "transformer",
        json!({
            "_class_name":"QwenImage21Transformer2DModel","patch_size":1,"in_channels":4,"out_channels":4,
            "num_layers":1,"attention_head_dim":32,"num_attention_heads":1,"context_in_dim":32,
            "mlp_ratio":2,"axes_dims_rope":[8,12,12],"eps":1e-6,"causal_condition":true
        }),
        nonzero,
        |vb, config| {
            QwenImageTransformer::load(vb, serde_json::from_value(config)?)?;
            Ok(())
        },
    )?;
    record_component(
        dir.path(),
        "vae",
        json!({
            "_class_name":"AutoencoderKLQwenImage21","base_dim":2,"decoder_base_dim":2,"z_dim":4,
            "num_res_blocks":1,"latents_mean":[0.,0.,0.,0.],"latents_std":[1.,1.,1.,1.]
        }),
        nonzero,
        |vb, config| {
            QwenImageVae::load_decoder(vb, serde_json::from_value(config)?)?;
            Ok(())
        },
    )?;
    std::fs::create_dir_all(dir.path().join("scheduler"))?;
    std::fs::write(
        dir.path().join("scheduler/scheduler_config.json"),
        serde_json::to_vec(&json!({
            "_class_name":"FlowMatchEulerDiscreteScheduler","num_train_timesteps":1000,"shift":1.,
            "use_dynamic_shifting":true,"base_image_seq_len":256,"max_image_seq_len":8192,
            "base_shift":0.5,"max_shift":0.9,"shift_terminal":0.02,"time_shift_type":"exponential",
            "invert_sigmas":false,"stochastic_sampling":false,"use_karras_sigmas":false,
            "use_exponential_sigmas":false,"use_beta_sigmas":false
        }))?,
    )?;
    std::fs::create_dir_all(dir.path().join("processor"))?;
    let words = WordLevel::builder()
        .unk_token("[UNK]".into())
        .vocab([("[UNK]".to_string(), 0)].into_iter().collect())
        .build()
        .map_err(|e| anyhow::anyhow!("fixture tokenizer: {e}"))?;
    let mut tokenizer = Tokenizer::new(words);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer
        .save(dir.path().join("processor/tokenizer.json"), false)
        .map_err(|e| anyhow::anyhow!("fixture tokenizer save: {e}"))?;
    Ok(dir)
}

fn tiny_options() -> ImageGenerationOptions {
    ImageGenerationOptions {
        width: 32,
        height: 32,
        steps: 2,
        seed: 17,
        ..Default::default()
    }
}

fn check_pipeline(device: Device) -> Result<()> {
    let snapshot = snapshot()?;
    let pipeline = QwenImagePipeline::from_snapshot(snapshot.path(), device)?;
    let options = tiny_options();
    pipeline.validate_request("a red kite", &options)?;
    let cancelled = AtomicBool::new(false);
    let mut steps = Vec::new();
    let result = pipeline.generate("a red kite", &options, &cancelled, |progress| {
        if let DiffusionProgress::Denoising { completed, .. } = progress {
            steps.push(completed);
        }
    })?;
    assert_eq!(steps, [0, 1, 2]);
    let pixels = image::load_from_memory(&result.png)?.to_rgba8();
    assert_eq!(pixels.dimensions(), (32, 32));
    // The entirely zero-weight fixture must decode to zero before the VAE
    // [-1,1] -> byte conversion. This is deliberately not a generated image.
    assert!(pixels.pixels().all(|pixel| pixel.0 == [128, 128, 128, 128]));
    let result = pipeline.generate("a red kite", &options, &cancelled, |progress| {
        if matches!(progress, DiffusionProgress::Denoising { completed: 1, .. }) {
            cancelled.store(true, Ordering::Relaxed);
        }
    });
    assert!(result.err().unwrap().to_string().contains("cancelled"));
    Ok(())
}

fn render_nonzero(device: Device) -> Result<Vec<[u8; 4]>> {
    let snapshot = nonzero_snapshot()?;
    let pipeline = QwenImagePipeline::from_snapshot(snapshot.path(), device)?;
    let cancelled = AtomicBool::new(false);
    let result = pipeline.generate("a red kite", &tiny_options(), &cancelled, |_| {})?;
    assert!(!result.png.is_empty());
    let pixels = image::load_from_memory(&result.png)?.to_rgba8();
    assert_eq!(pixels.dimensions(), (32, 32));

    // `generate` rejects non-finite decoder values before encoding.  These
    // finite PNG samples additionally ensure that the pattern fixture did not
    // collapse to the old all-zero/all-128 output.
    let rgb_values = pixels
        .pixels()
        .flat_map(|pixel| pixel.0[..3].iter().copied())
        .collect::<Vec<_>>();
    let min = *rgb_values.iter().min().expect("non-empty image");
    let max = *rgb_values.iter().max().expect("non-empty image");
    assert!(max > min, "nonzero fixture produced a constant RGB image");
    assert!(
        pixels
            .pixels()
            .any(|pixel| { pixel.0[0] != 128 || pixel.0[1] != 128 || pixel.0[2] != 128 })
    );

    Ok(pixels.pixels().map(|pixel| pixel.0).collect())
}

#[test]
fn staged_snapshot_to_png_cpu_and_cancellation() -> Result<()> {
    check_pipeline(Device::Cpu)
}

#[test]
fn nonzero_snapshot_to_png_cpu_is_finite_and_nonconstant() -> Result<()> {
    let _ = render_nonzero(Device::Cpu)?;
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn staged_bf16_snapshot_to_png_metal_and_cancellation() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    check_pipeline(device)
}

#[cfg(target_os = "macos")]
#[test]
fn nonzero_snapshot_cpu_f32_matches_metal_bf16() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    let cpu = render_nonzero(Device::Cpu)?;
    let metal = render_nonzero(device)?;
    assert_eq!(cpu.len(), metal.len());

    // The comparison is after 8-bit PNG quantization. On the test M1 Pro,
    // the maximum BF16/F32 difference is one byte level; allow one additional
    // level for reduction-order differences on other Metal generations.
    const MAX_PIXEL_DELTA: u8 = 2;
    for (index, (cpu, metal)) in cpu.iter().zip(metal).enumerate() {
        for channel in 0..4 {
            let delta = cpu[channel].abs_diff(metal[channel]);
            assert!(
                delta <= MAX_PIXEL_DELTA,
                "CPU/Metal pixel mismatch at {index}, channel {channel}: {} vs {} (delta {delta})",
                cpu[channel],
                metal[channel]
            );
        }
    }
    Ok(())
}
