use super::{MAX_RES_BLOCKS, PublicVaeConfig, QwenImageVae, avg_down_2d, dup_up_2d};
use candle_core::{DType, Device, Result, Shape, Tensor};
use candle_nn::{Init, VarBuilder, var_builder::SimpleBackend};

struct FixedBackend;

impl SimpleBackend for FixedBackend {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _: Init,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let value = if name.ends_with("gamma") {
            1_f32
        } else if name.ends_with("bias") {
            0.005_f32
        } else {
            0.01_f32
        };
        Tensor::full(value, shape, device)?.to_dtype(dtype)
    }

    fn get_unchecked(&self, _name: &str, _dtype: DType, _device: &Device) -> Result<Tensor> {
        candle_core::bail!("fixed test backend requires a requested tensor shape")
    }

    fn contains_tensor(&self, _name: &str) -> bool {
        true
    }
}

fn fixed_vb(dtype: DType, device: &Device) -> VarBuilder<'static> {
    VarBuilder::from_backend(Box::new(FixedBackend), dtype, device.clone())
}

fn decode_fixed(device: &Device, dtype: DType) -> Result<Tensor> {
    let vae = QwenImageVae::load_decoder(fixed_vb(dtype, device), tiny_config())?;
    let latent = Tensor::full(0.2_f32, (1, 1, 1, 1), device)?.to_dtype(dtype)?;
    vae.decode(&latent)
}

#[cfg(target_os = "macos")]
fn trace_decode_fixed(device: &Device, dtype: DType) -> Result<Vec<(&'static str, Tensor)>> {
    let vae = QwenImageVae::load_decoder(fixed_vb(dtype, device), tiny_config())?;
    let latent = Tensor::full(0.2_f32, (1, 1, 1, 1), device)?.to_dtype(dtype)?;
    let mut trace = Vec::new();
    let mut x = vae.post_quant_conv.forward(&latent.unsqueeze(2)?)?;
    trace.push(("post_quant", x.clone()));
    x = vae.decoder.conv_in.forward(&x)?;
    trace.push(("conv_in", x.clone()));
    x = vae.decoder.mid_block.first.forward(&x)?;
    trace.push(("mid_first", x.clone()));
    x = vae.decoder.mid_block.attention.forward(&x)?;
    trace.push(("mid_attention", x.clone()));
    x = vae.decoder.mid_block.second.forward(&x)?;
    trace.push(("mid_second", x.clone()));
    for (index, block) in vae.decoder.up_blocks.iter().enumerate() {
        x = block.forward(&x)?;
        trace.push((
            match index {
                0 => "up_0",
                1 => "up_1",
                2 => "up_2",
                3 => "up_3",
                _ => "up_final",
            },
            x.clone(),
        ));
    }
    x = candle_nn::ops::silu(&vae.decoder.norm_out.forward(&x)?)?;
    trace.push(("norm_out", x.clone()));
    x = vae.decoder.conv_out.forward(&x)?;
    trace.push(("conv_out", x.clone()));
    trace.push(("clamp", super::clamp_to_unit(&x)?));
    Ok(trace)
}

fn tiny_config() -> PublicVaeConfig {
    // Keep the released five-stage topology (and therefore the 16x spatial
    // compression) while making every convolution small enough for a unit
    // test.  VarBuilder::zeros materializes only the requested tensors.
    PublicVaeConfig {
        base_dim: 1,
        decoder_base_dim: Some(1),
        z_dim: 1,
        latents_mean: vec![0.],
        latents_std: vec![1.],
        ..Default::default()
    }
}

fn check_bounded_convolution(device: &Device) -> Result<()> {
    for batch in [1, 2] {
        for (kernel, stride) in [(1, 1), (3, 1), (3, 2)] {
            let input = Tensor::from_vec(
                (0..batch * 3 * 19 * 13)
                    .map(|i| (i as f32 * 0.17).sin())
                    .collect::<Vec<_>>(),
                (batch, 3, 19, 13),
                device,
            )?;
            let conv = candle_nn::conv2d(
                3,
                5,
                kernel,
                candle_nn::Conv2dConfig {
                    stride,
                    ..Default::default()
                },
                fixed_vb(DType::F32, device),
            )?;
            let full = candle_core::Module::forward(&conv, &input)?;
            let tiled = super::bounded_conv2d(&conv, &input, 800)?;
            assert_eq!(full.dims(), tiled.dims());
            let full = full.flatten_all()?.to_vec1::<f32>()?;
            let tiled = tiled.flatten_all()?.to_vec1::<f32>()?;
            assert!(full.iter().zip(tiled).all(|(a, b)| (a - b).abs() < 1e-5));
        }
    }
    Ok(())
}

#[test]
fn bounded_convolution_matches_whole_image_with_halo_stride_and_tail() -> Result<()> {
    check_bounded_convolution(&Device::Cpu)
}

#[cfg(target_os = "macos")]
#[test]
fn bounded_convolution_matches_whole_image_metal() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    check_bounded_convolution(&device)
}

#[test]
fn released_latent_statistics_have_exact_channel_count() {
    let config = PublicVaeConfig::default();
    assert_eq!(config.latents_mean.len(), 64);
    assert_eq!(config.latents_std.len(), 64);
    assert_eq!(config.latents_std[38], 2.8063);
    assert_eq!(config.latents_std[63], 3.8161);
}

#[test]
fn residual_block_count_is_bounded_before_model_allocation() -> Result<()> {
    let mut config = PublicVaeConfig::default();
    for count in [0, MAX_RES_BLOCKS + 1, usize::MAX - 1, usize::MAX] {
        config.num_res_blocks = count;
        let error = config.validate().unwrap_err().to_string();
        assert!(
            error.contains("num_res_blocks"),
            "unexpected validation error for {count}: {error}"
        );
        // Both public entry points must reject the config before even trying
        // to access weights. An empty backend keeps this regression bounded.
        for decoder_only in [false, true] {
            let vb = VarBuilder::from_tensors(
                std::collections::HashMap::new(),
                DType::F32,
                &Device::Cpu,
            );
            let result = if decoder_only {
                QwenImageVae::load_decoder(vb, config.clone())
            } else {
                QwenImageVae::load(vb, config.clone())
            };
            let error = result
                .err()
                .expect("invalid block count must fail")
                .to_string();
            assert!(error.contains("num_res_blocks"), "{error}");
        }
    }

    for count in [1, 2, MAX_RES_BLOCKS] {
        config.num_res_blocks = count;
        config.validate()?;
    }
    Ok(())
}

#[test]
fn spatial_topology_is_bounded_before_model_allocation() -> Result<()> {
    let device = Device::Cpu;
    let mut config = tiny_config();
    for levels in [1, 6] {
        config.dim_mult = vec![1; levels];
        config.temperal_downsample = vec![false; levels - 1];
        let validation_error = config.validate().unwrap_err().to_string();
        assert!(validation_error.contains("dim_mult"), "{validation_error}");

        // Both public constructors must reject the unsupported topology
        // before asking the backend for any weight shape.
        for decoder_only in [false, true] {
            let vb =
                VarBuilder::from_tensors(std::collections::HashMap::new(), DType::F32, &device);
            let result = if decoder_only {
                QwenImageVae::load_decoder(vb, config.clone())
            } else {
                QwenImageVae::load(vb, config.clone())
            };
            let error = result
                .err()
                .expect("unsupported spatial topology must fail")
                .to_string();
            assert!(error.contains("dim_mult"), "{error}");
        }
    }

    // The released five-level topology remains valid and is covered by the
    // full encode/decode shape test below.
    config = tiny_config();
    config.validate()?;
    Ok(())
}

#[test]
fn t1_encode_decode_shapes_and_decoder_only_loading() -> Result<()> {
    let device = Device::Cpu;
    let image = Tensor::zeros((1, 4, 16, 16), DType::F32, &device)?;
    let vae = QwenImageVae::load(VarBuilder::zeros(DType::F32, &device), tiny_config())?;

    let moments = vae.encode(&image)?;
    assert_eq!(moments.dims(), &[1, 2, 1, 1]);
    let latent = vae.encode_mode(&image)?;
    assert_eq!(latent.dims(), &[1, 1, 1, 1]);
    let decoded = vae.decode(&latent)?;
    assert_eq!(decoded.dims(), &[1, 4, 16, 16]);
    assert!(
        decoded
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|v| *v == 0.)
    );

    let decoder =
        QwenImageVae::load_decoder(VarBuilder::zeros(DType::F32, &device), tiny_config())?;
    assert_eq!(decoder.decode(&latent)?.dims(), &[1, 4, 16, 16]);
    assert!(decoder.encode(&image).is_err());
    Ok(())
}

#[test]
fn loader_refuses_precision_reduction_and_affine_roundtrips() -> Result<()> {
    let device = Device::Cpu;
    assert!(
        QwenImageVae::load_decoder(VarBuilder::zeros(DType::F16, &device), tiny_config(),).is_err()
    );

    let mut config = tiny_config();
    config.latents_mean = vec![2.];
    config.latents_std = vec![4.];
    let vae = QwenImageVae::load_decoder(VarBuilder::zeros(DType::F32, &device), config)?;
    let raw = Tensor::new(&[[[[10_f32]]]], &device)?;
    let normalized = vae.normalize_latents(&raw)?;
    assert_eq!(normalized.flatten_all()?.to_vec1::<f32>()?, vec![2.]);
    assert_eq!(
        vae.denormalize_latents(&normalized)?
            .flatten_all()?
            .to_vec1::<f32>()?,
        raw.flatten_all()?.to_vec1::<f32>()?
    );
    Ok(())
}

#[test]
fn temporal_shortcuts_preserve_t1_padding_and_duplication() -> Result<()> {
    let device = Device::Cpu;
    let source = Tensor::from_vec(vec![1_f32, 2., 3., 4.], (1, 1, 1, 2, 2), &device)?;
    let averaged = avg_down_2d(&source, 1, 1, 2, 2)?;
    assert_eq!(averaged.dims(), &[1, 1, 1, 1, 1]);
    assert_eq!(averaged.flatten_all()?.to_vec1::<f32>()?, vec![1.25]);

    let duplicated = dup_up_2d(&Tensor::full(7_f32, (1, 1, 1, 1, 1), &device)?, 1, 1, 2, 2)?;
    assert_eq!(duplicated.dims(), &[1, 1, 1, 2, 2]);
    assert_eq!(duplicated.flatten_all()?.to_vec1::<f32>()?, vec![7.; 4]);
    Ok(())
}

#[test]
fn decoder_clamp_preserves_reference_unit_range() -> Result<()> {
    let values = Tensor::new(&[[-2_f32, -0.5, 0.5, 2.]], &Device::Cpu)?;
    let clamped = super::clamp_to_unit(&values)?;
    assert_eq!(clamped.to_vec2::<f32>()?, &[[-1., -0.5, 0.5, 1.]]);
    Ok(())
}

#[test]
fn decoder_clamp_preserves_nan_cpu() -> Result<()> {
    let values = Tensor::new(&[f32::NAN, -2., 0.5, 2.], &Device::Cpu)?;
    let clamped = super::clamp_to_unit(&values)?.to_vec1::<f32>()?;
    assert!(clamped[0].is_nan());
    assert_eq!(&clamped[1..], &[-1., 0.5, 1.]);
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn decoder_clamp_preserves_nan_metal() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    let values = Tensor::new(&[f32::NAN, -2., 0.5, 2.], &device)?;
    let clamped = super::clamp_to_unit(&values)?.to_vec1::<f32>()?;
    assert!(clamped[0].is_nan());
    assert_eq!(&clamped[1..], &[-1., 0.5, 1.]);
    Ok(())
}

#[test]
fn tiny_decoder_cpu_precision_paths_are_nonzero_and_finite() -> Result<()> {
    let decoded = decode_fixed(&Device::Cpu, DType::F32)?;
    assert_eq!(decoded.dims(), &[1, 4, 16, 16]);
    let values = decoded.flatten_all()?.to_vec1::<f32>()?;
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values.iter().any(|value| value.abs() > 1e-6));
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn tiny_decoder_metal_f32_and_bf16_match_cpu_reference() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    let cpu = decode_fixed(&Device::Cpu, DType::F32)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let metal = decode_fixed(&device, DType::F32)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert!(metal.iter().any(|value| value.abs() > 1e-6));
    let max_error = max_abs_error(&cpu, &metal);
    assert!(max_error < 1e-5, "Metal F32 VAE max error {max_error}");

    // Candle's CPU backend does not implement BF16 matmul, so the BF16
    // device result is checked against the same fixed-weight F32 reference.
    let metal_bf16 = decode_fixed(&device, DType::BF16)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert!(metal_bf16.iter().all(|value| value.is_finite()));
    assert!(metal_bf16.iter().any(|value| value.abs() > 1e-6));
    let bf16_error = max_abs_error(&cpu, &metal_bf16);
    assert!(
        bf16_error < 0.002,
        "Metal BF16 VAE vs CPU F32 max error {bf16_error}"
    );
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn tiny_decoder_metal_f32_stages_match_cpu_reference() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    let cpu = trace_decode_fixed(&Device::Cpu, DType::F32)?;
    let metal = trace_decode_fixed(&device, DType::F32)?;
    for ((cpu_name, cpu_tensor), (metal_name, metal_tensor)) in cpu.iter().zip(metal.iter()) {
        assert_eq!(cpu_name, metal_name);
        let cpu_values = cpu_tensor
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let metal_values = metal_tensor
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(
            cpu_values.iter().all(|value| value.is_finite()),
            "CPU VAE stage {cpu_name} contains non-finite values"
        );
        assert!(
            metal_values.iter().all(|value| value.is_finite()),
            "Metal VAE stage {metal_name} contains non-finite values"
        );
        let max_error = max_abs_error(&cpu_values, &metal_values);
        assert!(
            max_error < 1e-5,
            "Metal VAE stage {cpu_name} {:?} max error {max_error}",
            cpu_tensor.dims()
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn max_abs_error(expected: &[f32], actual: &[f32]) -> f32 {
    assert_eq!(expected.len(), actual.len());
    assert!(expected.iter().all(|value| value.is_finite()));
    assert!(actual.iter().all(|value| value.is_finite()));
    expected
        .iter()
        .zip(actual)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0_f32, f32::max)
}
