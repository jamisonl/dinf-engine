use super::*;

const WIDTH: usize = 64;

fn attention(device: &Device, window: usize) -> Result<Attention> {
    let zeros = Tensor::zeros((WIDTH, WIDTH), DType::F32, device)?;
    let identity = Tensor::from_vec(
        (0..WIDTH * WIDTH)
            .map(|i| {
                if i / WIDTH == i % WIDTH {
                    1.0f32
                } else {
                    0.0f32
                }
            })
            .collect::<Vec<_>>(),
        (WIDTH, WIDTH),
        device,
    )?;
    let (rope_cos, rope_sin) = precompute_rope_cos_sin(WIDTH, 10_000., 32, device, None)?;
    Ok(Attention {
        q_proj: QMatMul::Tensor(zeros.clone()),
        k_proj: QMatMul::Tensor(zeros),
        v_proj: QMatMul::Tensor(identity.clone()),
        o_proj: QMatMul::Tensor(identity),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: WIDTH,
        sliding_window: window,
        rope_cos,
        rope_sin,
        rope_is_neox: false,
        quant_config: None,
    })
}

fn input(values: &[f32], device: &Device) -> Result<Tensor> {
    Tensor::from_vec(
        values.iter().flat_map(|v| [*v; WIDTH]).collect::<Vec<_>>(),
        (1, values.len(), WIDTH),
        device,
    )
    .map_err(Into::into)
}

fn assert_rows(actual: &Tensor, values: &[f32]) -> Result<()> {
    let rows = actual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(rows.len(), values.len() * WIDTH);
    for (row, expected) in rows.chunks(WIDTH).zip(values) {
        for actual in row {
            assert!((actual - expected).abs() < 0.03, "{actual} != {expected}");
        }
    }
    Ok(())
}

fn check_windowed_attention(device: &Device) -> Result<()> {
    // Zero Q/K makes the independent answer the mean of exactly the visible values.
    // Values before the left edge must not affect either prefill or later decode.
    let attn = attention(device, 2)?;
    let values = [2., 4., 8., 16., 32., 64.];
    let expected = [2., 3., 6., 12., 24., 48.];
    let x = input(&values, device)?;
    assert_rows(&attn.forward(&x, &mut None, values.len(), 0)?, &expected)?;
    for chunks in [&[2, 3, 1][..], &[1, 1, 1, 1, 1, 1][..], &[5, 1][..]] {
        let mut cache = None;
        let mut offset = 0;
        for &len in chunks {
            let out =
                attn.forward_pass(&x.narrow(1, offset, len)?, &mut cache, len, offset, true)?;
            assert_rows(&out, &expected[offset..offset + len])?;
            offset += len;
        }
    }

    // Equal-length lanes use the regular attention path with a batch dimension.
    let batch = Tensor::cat(&[&x, &(x.clone() * 2.)?], 0)?;
    let out = attn.forward(&batch, &mut None, values.len(), 0)?;
    assert_rows(&out.narrow(0, 0, 1)?, &expected)?;
    assert_rows(&out.narrow(0, 1, 1)?, &expected.map(|v| v * 2.))?;

    // Different-length lanes use both the separate and packed cache paths.
    for packed in [false, true] {
        let mut first = None;
        let mut second = None;
        attn.forward(&input(&values[..3], device)?, &mut first, 3, 0)?;
        attn.forward(&input(&values[..5], device)?, &mut second, 5, 0)?;
        let next = Tensor::cat(&[&input(&[16.], device)?, &input(&[64.], device)?], 0)?;
        let out = if packed {
            let mut first = [first.map(LayerCache::Attention)];
            let mut second = [second.map(LayerCache::Attention)];
            let lanes: Vec<&mut [Option<LayerCache>]> = vec![&mut first, &mut second];
            let mut cache = super::super::pack_ragged_decode_layer_cache(&lanes, 0, &[3, 5])?;
            attn.forward_packed_ragged_decode(&next, &mut cache, &[3, 5])?
        } else {
            attn.forward_ragged_decode(&next, &mut [&mut first, &mut second], &[3, 5])?
        };
        assert_rows(&out, &[12., 48.])?;
    }
    Ok(())
}

#[test]
fn sliding_window_matches_reference_for_prefill_decode_and_batches_cpu() -> Result<()> {
    check_windowed_attention(&Device::Cpu)
}

#[cfg(target_os = "macos")]
#[test]
fn sliding_window_matches_reference_for_prefill_decode_and_batches_metal() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    check_windowed_attention(&device)
}

#[test]
fn gguf_sliding_window_reaches_each_selected_layer() -> Result<()> {
    use crate::gguf_loader::{LayerRange, parse_gguf_metadata};
    use candle_core::quantized::{GgmlDType, gguf_file::Value};
    use std::collections::HashMap;

    for architecture in ["llama", "mistral", "qwen2"] {
        let mut metadata = HashMap::from([(
            "general.architecture".to_owned(),
            Value::String(architecture.into()),
        )]);
        for (suffix, value) in [
            ("block_count", 3),
            ("embedding_length", WIDTH),
            ("attention.head_count", 1),
            ("context_length", 16),
            ("attention.sliding_window", 2),
        ] {
            metadata.insert(format!("{architecture}.{suffix}"), Value::U32(value as u32));
        }
        let metadata = parse_gguf_metadata(&metadata)?;
        for (pattern, expected_window) in [(vec![], 2), (vec![true, false, true], 0)] {
            let mut metadata = metadata.clone();
            metadata.sliding_window_pattern = pattern;
            metadata.rope_scaling_factor = 4.0;
            let mut tensors = HashMap::new();
            tensors.insert(
                "rope_freqs.weight".to_owned(),
                Arc::new(QTensor::quantize(
                    &Tensor::full(2.0f32, WIDTH / 2, &Device::Cpu)?,
                    GgmlDType::F32,
                )?),
            );
            for name in ["attn_norm.weight", "ffn_norm.weight"] {
                tensors.insert(
                    format!("blk.1.{name}"),
                    Arc::new(QTensor::quantize(
                        &Tensor::ones(WIDTH, DType::F32, &Device::Cpu)?,
                        GgmlDType::F32,
                    )?),
                );
            }
            for name in [
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ] {
                tensors.insert(
                    format!("blk.1.{name}"),
                    Arc::new(QTensor::quantize(
                        &Tensor::zeros((WIDTH, WIDTH), DType::F32, &Device::Cpu)?,
                        GgmlDType::F32,
                    )?),
                );
            }
            let stage = LlamaStage::from_stage_tensors(
                StageTensors {
                    range: LayerRange::try_new(1, 2, 3)?,
                    metadata,
                    tensors,
                    raw_tensors: HashMap::new(),
                    raw_source: None,
                    device: Device::Cpu,
                },
                None,
            )?;
            assert_eq!(stage.blocks[0].attn.sliding_window, expected_window);
            // The tensor supplies the actual per-dimension scaling. Do not
            // also divide by the metadata factor and apply scaling twice.
            let angle_cos = stage.blocks[0]
                .attn
                .rope_cos
                .i((1, 0))?
                .to_scalar::<f32>()?;
            assert!((angle_cos - 0.5f32.cos()).abs() < 1e-6);
        }
    }
    Ok(())
}

#[test]
fn metadata_only_rope_scaling_fails_before_loading_weights() -> Result<()> {
    use crate::gguf_loader::{LayerRange, parse_gguf_metadata};
    use candle_core::quantized::gguf_file::Value;
    use std::collections::HashMap;

    for architecture in ["llama", "mistral", "qwen2"] {
        let mut values = HashMap::from([(
            "general.architecture".to_owned(),
            Value::String(architecture.into()),
        )]);
        for (suffix, value) in [
            ("block_count", 3),
            ("embedding_length", WIDTH),
            ("attention.head_count", 1),
        ] {
            values.insert(format!("{architecture}.{suffix}"), Value::U32(value as u32));
        }
        values.insert(
            format!("{architecture}.rope.scaling.factor"),
            Value::F32(4.0),
        );
        let stage = StageTensors {
            range: LayerRange::try_new(1, 2, 3)?,
            metadata: parse_gguf_metadata(&values)?,
            tensors: HashMap::new(),
            raw_tensors: HashMap::new(),
            raw_source: None,
            device: Device::Cpu,
        };
        let error = LlamaStage::from_stage_tensors(stage, None)
            .err()
            .expect("unsupported RoPE scaling must fail");
        assert!(
            error.to_string().contains("unsupported scaled RoPE"),
            "{error}"
        );
    }
    Ok(())
}
