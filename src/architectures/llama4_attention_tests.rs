use super::*;

fn values(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite(), "non-finite value at {index}");
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: {actual} != {expected}, tolerance {tolerance}"
        );
    }
}

// A scalar f64 dot product and softmax, independent of the tensor masks,
// slicing and native attention path. Input K/V already have cache precision.
fn reference_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    start_pos: usize,
    chunk_size: usize,
) -> Result<Vec<f32>> {
    let (batch, heads, rows, width) = q.dims4()?;
    let (_, kv_heads, keys, _) = k.dims4()?;
    let cache_start = start_pos + rows - keys;
    let q = values(q)?;
    let k = values(k)?;
    let v = values(v)?;
    let mut output = Vec::with_capacity(q.len());
    for lane in 0..batch {
        for head in 0..heads {
            let kv_head = head / (heads / kv_heads);
            for row in 0..rows {
                let position = start_pos + row;
                let visible = (0..keys)
                    .filter(|key| {
                        let key_pos = cache_start + key;
                        key_pos <= position
                            && (chunk_size == 0 || key_pos / chunk_size == position / chunk_size)
                    })
                    .collect::<Vec<_>>();
                let q_offset = ((lane * heads + head) * rows + row) * width;
                let scores = visible
                    .iter()
                    .map(|key| {
                        let k_offset = ((lane * kv_heads + kv_head) * keys + key) * width;
                        (0..width)
                            .map(|col| f64::from(q[q_offset + col]) * f64::from(k[k_offset + col]))
                            .sum::<f64>()
                            / (width as f64).sqrt()
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights = scores
                    .iter()
                    .map(|score| (score - max).exp())
                    .collect::<Vec<_>>();
                let denom = weights.iter().sum::<f64>();
                for col in 0..width {
                    output.push(
                        (visible
                            .iter()
                            .zip(&weights)
                            .map(|(key, weight)| {
                                let offset = ((lane * kv_heads + kv_head) * keys + key) * width;
                                weight * f64::from(v[offset + col])
                            })
                            .sum::<f64>()
                            / denom) as f32,
                    );
                }
            }
        }
    }
    Ok(output)
}

fn check_attention_boundaries(device: &Device) -> Result<()> {
    let (batch, heads, kv_heads, rows, width) = (2, 4, 2, 7, 32);
    // Include both a full prefix and an already-evicted cache with a nonzero
    // absolute origin. Queries cross 8191/8192; the small case spans two edges.
    for (start_pos, prefix, chunk_size) in [(3, 3, 4), (8189, 8189, 8192), (8189, 13, 8192)] {
        let keys = prefix + rows;
        let q = Tensor::from_vec(
            (0..batch * heads * rows * width)
                .map(|i| ((i * 17 % 53) as f32 - 26.) / 31.)
                .collect(),
            (batch, heads, rows, width),
            device,
        )?;
        let k = Tensor::from_vec(
            (0..batch * kv_heads * keys * width)
                .map(|i| ((i * 11 % 59) as f32 - 29.) / 32.)
                .collect(),
            (batch, kv_heads, keys, width),
            device,
        )?
        .to_dtype(DType::F16)?;
        let v = Tensor::from_vec(
            (0..batch * kv_heads * keys * width)
                .map(|i| ((i * 7 % 47) as f32 - 23.) / 16.)
                .collect(),
            (batch, kv_heads, keys, width),
            device,
        )?
        .to_dtype(DType::F16)?;
        for window in [0, chunk_size] {
            let expected = reference_attention(&q, &k, &v, start_pos, window)?;
            let expected = Tensor::from_vec(expected, q.shape(), &Device::Cpu)?;
            let tolerance = if device.is_cpu() { 2e-6 } else { 2e-3 };
            let actual = cached_attention(&q, &k, &v, start_pos, window)?;
            assert_close(&values(&actual)?, &values(&expected)?, tolerance);
            for chunks in [&[1, 1, 1, 1, 1, 1, 1][..], &[2, 3, 2][..], &[4, 3][..]] {
                let mut offset = 0;
                for &len in chunks {
                    let actual = cached_attention(
                        &q.narrow(2, offset, len)?,
                        &k.narrow(2, 0, prefix + offset + len)?,
                        &v.narrow(2, 0, prefix + offset + len)?,
                        start_pos + offset,
                        window,
                    )?;
                    assert_close(
                        &values(&actual)?,
                        &values(&expected.narrow(2, offset, len)?)?,
                        tolerance,
                    );
                    offset += len;
                }
            }
        }
        #[cfg(target_os = "macos")]
        if device.is_metal() {
            assert!(
                super::super::try_metal_f16_sdpa_native(
                    &q,
                    &k,
                    &v,
                    true,
                    1.0 / (width as f32).sqrt()
                )?
                .is_some()
            );
        }
    }
    Ok(())
}

fn check_nope_temperature(device: &Device) -> Result<()> {
    let q = Tensor::ones((2, 3, 5, 32), DType::F32, device)?;
    for start_pos in [0, 8189, 16381] {
        let actual = apply_nope_temperature(&q, start_pos)?;
        let expected = (0..2 * 3 * 5 * 32)
            .map(|i| {
                let position = start_pos + i / 32 % 5;
                let block = (position + 1) / 8192;
                (1.0 + (block as f64 + 1.0).ln() * f64::from(0.1f32)) as f32
            })
            .collect::<Vec<_>>();
        assert_close(&values(&actual)?, &expected, 0.0);
        for (offset, len) in [(0, 2), (2, 1), (3, 2)] {
            let piece = apply_nope_temperature(&q.narrow(2, offset, len)?, start_pos + offset)?;
            assert_close(
                &values(&piece)?,
                &values(&actual.narrow(2, offset, len)?)?,
                0.0,
            );
        }
    }
    Ok(())
}

fn check_cache_append(device: &Device) -> Result<()> {
    const WIDTH: usize = 32;
    let identity = Tensor::from_vec(
        (0..WIDTH * WIDTH)
            .map(|i| f32::from(i / WIDTH == i % WIDTH))
            .collect::<Vec<_>>(),
        (WIDTH, WIDTH),
        device,
    )?;
    let zeros = Tensor::zeros((WIDTH, WIDTH), DType::F32, device)?;
    let (rope_cos, rope_sin) = precompute_rope_cos_sin(WIDTH, 500_000., 8, device, None)?;
    let attention = Attention {
        q_proj: QMatMul::Tensor(zeros.clone()),
        k_proj: QMatMul::Tensor(zeros),
        v_proj: DecodeSafeQMatMul::from_qtensor(
            Arc::new(QTensor::quantize(&identity, GgmlDType::F32)?),
            device,
        )?,
        o_proj: QMatMul::Tensor(identity),
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: WIDTH,
        rope_cos,
        rope_sin,
        rope_theta: 500_000.,
        rope_freqs: None,
        use_rope: true,
        chunk_size: 8192,
        qk_norm: None,
        quant_config: None,
    };
    let rows = [2.0f32, 4., 8., 16., 32., 64.];
    let x = Tensor::from_vec(
        rows.iter().flat_map(|v| [*v; WIDTH]).collect::<Vec<_>>(),
        (1, rows.len(), WIDTH),
        device,
    )?;
    // Absolute origin8189 tests an evicted prefix without allocating8192 rows.
    // After the boundary, the query must exclude every key in the old block.
    let expected = [2., 3., 14. / 3., 16., 24., 112. / 3.];
    for chunks in [&[6][..], &[2, 3, 1][..], &[1, 1, 1, 1, 1, 1][..]] {
        let mut cache = None;
        let mut offset = 0;
        for &len in chunks {
            let output =
                attention.forward(&x.narrow(1, offset, len)?, &mut cache, len, 8189 + offset)?;
            let expected = expected[offset..offset + len]
                .iter()
                .flat_map(|value| [*value; WIDTH])
                .collect::<Vec<_>>();
            assert_close(
                &values(&output)?,
                &expected,
                if device.is_cpu() { 1e-5 } else { 0.04 },
            );
            offset += len;
        }
    }
    Ok(())
}

fn check_dynamic_rope_reference(device: &Device) -> Result<()> {
    let width = 8;
    let start_pos = 8190;
    let theta = 500_000.0f64;
    let input = (0..4 * width)
        .map(|i| (i as f32 - 15.) / 19.)
        .collect::<Vec<_>>();
    let x = Tensor::from_vec(input.clone(), (1, 1, 4, width), device)?;
    let (cos, sin) = precompute_rope_cos_sin(width, theta, 8192, device, None)?;
    let actual = apply_rotary_emb(&x, start_pos, &cos, &sin, theta, None)?;
    let mut expected = vec![0f32; input.len()];
    for row in 0..4 {
        for pair in 0..width / 2 {
            let angle = (start_pos + row) as f64 / theta.powf((2 * pair) as f64 / width as f64);
            let (sin, cos) = angle.sin_cos();
            let offset = row * width + pair * 2;
            let (a, b) = (f64::from(input[offset]), f64::from(input[offset + 1]));
            expected[offset] = (a * cos - b * sin) as f32;
            expected[offset + 1] = (a * sin + b * cos) as f32;
        }
    }
    assert_close(&values(&actual)?, &expected, 5e-5);
    for offset in 0..4 {
        let scalar = apply_rotary_emb(
            &x.narrow(2, offset, 1)?,
            start_pos + offset,
            &cos,
            &sin,
            theta,
            None,
        )?;
        assert_close(
            &values(&scalar)?,
            &values(&actual.narrow(2, offset, 1)?)?,
            1e-6,
        );
    }
    Ok(())
}

fn check_nope_forward(device: &Device) -> Result<()> {
    let width = 32;
    let identity = Tensor::from_vec(
        (0..width * width)
            .map(|i| f32::from(i / width == i % width))
            .collect::<Vec<_>>(),
        (width, width),
        device,
    )?;
    let (rope_cos, rope_sin) = precompute_rope_cos_sin(width, 500_000., 8, device, None)?;
    let attention = Attention {
        q_proj: QMatMul::Tensor(identity.clone()),
        k_proj: QMatMul::Tensor(identity.clone()),
        v_proj: DecodeSafeQMatMul::from_qtensor(
            Arc::new(QTensor::quantize(&identity, GgmlDType::F32)?),
            device,
        )?,
        o_proj: QMatMul::Tensor(identity),
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: width,
        rope_cos,
        rope_sin,
        rope_theta: 500_000.,
        rope_freqs: None,
        use_rope: false,
        chunk_size: 0,
        qk_norm: None,
        quant_config: None,
    };
    let input = (0..3 * width)
        .map(|i| (i / width + 1) as f32 * 0.25 + (i % width) as f32 / 128.)
        .collect::<Vec<_>>();
    let x = Tensor::from_vec(input.clone(), (1, 3, width), device)?;
    let kv = x.reshape((1, 1, 3, width))?.to_dtype(DType::F16)?;
    let scaled = input
        .iter()
        .enumerate()
        .map(|(i, value)| {
            let position = 8190 + i / width;
            let scale =
                (1.0 + (((position + 1) / 8192 + 1) as f64).ln() * f64::from(0.1f32)) as f32;
            value * scale
        })
        .collect::<Vec<_>>();
    let q = Tensor::from_vec(scaled, (1, 1, 3, width), device)?;
    let expected = reference_attention(&q, &kv, &kv, 8190, 0)?;
    let unscaled = reference_attention(&x.reshape((1, 1, 3, width))?, &kv, &kv, 8190, 0)?;
    assert!(
        expected
            .iter()
            .zip(unscaled)
            .any(|(a, b)| (a - b).abs() > 0.002)
    );
    for chunks in [&[3][..], &[1, 1, 1][..], &[2, 1][..]] {
        let mut cache = None;
        let mut offset = 0;
        for &len in chunks {
            let actual =
                attention.forward(&x.narrow(1, offset, len)?, &mut cache, len, 8190 + offset)?;
            assert_close(
                &values(&actual)?,
                &expected[offset * width..(offset + len) * width],
                if device.is_cpu() { 1e-6 } else { 1e-3 },
            );
            offset += len;
        }
    }
    Ok(())
}

#[test]
fn chunked_attention_matches_independent_reference_cpu() -> Result<()> {
    check_attention_boundaries(&Device::Cpu)
}

#[test]
fn nope_temperature_matches_absolute_positions_cpu() -> Result<()> {
    check_nope_temperature(&Device::Cpu)
}

#[test]
fn attention_cache_append_crosses_chunk_boundary_cpu() -> Result<()> {
    check_cache_append(&Device::Cpu)
}

#[test]
fn dynamic_rope_matches_scalar_reference_across_cache_boundary_cpu() -> Result<()> {
    check_dynamic_rope_reference(&Device::Cpu)
}

#[test]
fn nope_forward_applies_temperature_before_cache_attention_cpu() -> Result<()> {
    check_nope_forward(&Device::Cpu)
}

#[test]
fn disabled_chunking_keeps_rope_on_last_layer() {
    // Pinned llama.cpp15fa3c493 says explicit window0 means "always use rope",
    // but its modulo-by-n_layer expression accidentally skips the last layer.
    // Follow that metadata intent, including the last layer, rather than copy
    // the conflicting expression. Ordinary Scout retains every-fourth NoPE.
    for layer in 0..48 {
        assert!(uses_rope(layer, 0));
        assert_eq!(uses_rope(layer, 8192), layer % 4 != 3);
    }
}

#[test]
fn invalid_cache_positions_fail_before_slicing() -> Result<()> {
    let q = Tensor::zeros((1, 1, 2, 32), DType::F32, &Device::Cpu)?;
    assert!(cached_attention(&q, &q, &q, usize::MAX, 8192).is_err());
    let kv = Tensor::zeros((1, 1, 3, 32), DType::F32, &Device::Cpu)?;
    assert!(cached_attention(&q, &kv, &kv, 0, 8192).is_err());
    assert!(cached_attention(&kv, &q, &q, 0, 8192).is_err());
    assert!(apply_nope_temperature(&q, usize::MAX).is_err());
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn chunked_attention_and_temperature_boundaries_metal() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    check_attention_boundaries(&device)?;
    check_nope_temperature(&device)?;
    check_cache_append(&device)?;
    check_dynamic_rope_reference(&device)?;
    check_nope_forward(&device)
}

fn rope_scalar_reference(
    input: &[f32],
    width: usize,
    start_pos: usize,
    theta: f64,
    factors: &[f32],
) -> Vec<f32> {
    let mut expected = vec![0.0; input.len()];
    for (row, input) in input.chunks_exact(width).enumerate() {
        for (pair, &factor) in factors.iter().enumerate() {
            let angle = (start_pos + row) as f64
                / theta.powf((2 * pair) as f64 / width as f64)
                / f64::from(factor);
            let (sin, cos) = angle.sin_cos();
            let (a, b) = (f64::from(input[2 * pair]), f64::from(input[2 * pair + 1]));
            expected[row * width + 2 * pair] = (a * cos - b * sin) as f32;
            expected[row * width + 2 * pair + 1] = (a * sin + b * cos) as f32;
        }
    }
    expected
}

fn check_scaled_rope(device: &Device) -> Result<()> {
    // Powers of two give exact f32 inverse frequencies, so this test can
    // compare the rotations to independent f64 trig at a strict tolerance.
    let width = 8;
    let factors = [1.0, 1.0, 16.0, 16.0];
    let freq_factors = Tensor::new(&factors, device)?;
    let input = (0..4 * width)
        .map(|i| (i as f32 - 15.) / 19.)
        .collect::<Vec<_>>();
    let x = Tensor::from_vec(input.clone(), (1, 1, 4, width), device)?;
    for theta in [65_536., 4096.] {
        let (cos, sin) = precompute_rope_cos_sin(width, theta, 8192, device, Some(&freq_factors))?;
        // Cached-only, mixed cached/dynamic and entirely dynamic positions.
        for start_pos in [53, 8189, 8193] {
            let expected = rope_scalar_reference(&input, width, start_pos, theta, &factors);
            let unscaled = rope_scalar_reference(&input, width, start_pos, theta, &[1.; 4]);
            assert!(
                expected
                    .iter()
                    .zip(&unscaled)
                    .any(|(a, b)| (a - b).abs() > 0.03)
            );
            let actual = apply_rotary_emb(&x, start_pos, &cos, &sin, theta, Some(&freq_factors))?;
            assert_close(&values(&actual)?, &expected, 2e-6);
            for offset in 0..4 {
                let scalar = apply_rotary_emb(
                    &x.narrow(2, offset, 1)?,
                    start_pos + offset,
                    &cos,
                    &sin,
                    theta,
                    Some(&freq_factors),
                )?;
                assert_close(
                    &values(&scalar)?,
                    &expected[offset * width..(offset + 1) * width],
                    2e-6,
                );
            }
        }
    }
    Ok(())
}

#[test]
fn scaled_rope_matches_independent_reference_cpu() -> Result<()> {
    check_scaled_rope(&Device::Cpu)
}

#[cfg(target_os = "macos")]
#[test]
fn scaled_rope_matches_independent_reference_metal() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    check_scaled_rope(&device)
}

fn rope_test_metadata() -> Result<crate::gguf_loader::GgufMetadata> {
    use candle_core::quantized::gguf_file::Value;
    crate::gguf_loader::parse_gguf_metadata(&HashMap::from([
        (
            "general.architecture".into(),
            Value::String("llama4".into()),
        ),
        ("llama4.block_count".into(), Value::U32(6)),
        ("llama4.embedding_length".into(), Value::U32(8)),
        ("llama4.attention.head_count".into(), Value::U32(1)),
        ("llama4.context_length".into(), Value::U32(8193)),
    ]))
}

#[test]
fn invalid_rope_factors_fail_before_weight_allocation() -> Result<()> {
    use crate::gguf_loader::LayerRange;
    for (shape, factors) in [
        (vec![3], vec![1., 1., 16.]),
        (vec![1, 4], vec![1., 1., 16., 16.]),
        (vec![4], vec![1., 0., 16., 16.]),
        (vec![4], vec![1., -1., 16., 16.]),
        (vec![4], vec![1., f32::NAN, 16., 16.]),
        (vec![4], vec![1., f32::INFINITY, 16., 16.]),
    ] {
        let tensor = Tensor::from_vec(factors, shape, &Device::Cpu)?;
        let tensors = HashMap::from([(
            "rope_freqs.weight".to_owned(),
            Arc::new(QTensor::quantize(&tensor, GgmlDType::F32)?),
        )]);
        let stage = StageTensors {
            range: LayerRange::try_new(0, 1, 6)?,
            metadata: rope_test_metadata()?,
            tensors,
            raw_tensors: HashMap::new(),
            raw_source: None,
            device: Device::Cpu,
        };
        let error = Llama4Stage::from_stage_tensors(stage, None)
            .err()
            .expect("invalid factors must fail");
        assert!(error.to_string().contains("rope_freqs.weight"), "{error}");
    }
    assert!(load_rope_freqs(&HashMap::new(), 8, &Device::Cpu)?.is_none());
    Ok(())
}

#[test]
fn middle_stage_keeps_rope_factors_for_both_attention_schedules() -> Result<()> {
    use crate::gguf_loader::LayerRange;
    let factors = [1., 1., 16., 16.];
    let matrix = Arc::new(QTensor::quantize(
        &Tensor::ones((8, 8), DType::F32, &Device::Cpu)?,
        GgmlDType::F32,
    )?);
    let norm = Arc::new(QTensor::quantize(
        &Tensor::ones(8, DType::F32, &Device::Cpu)?,
        GgmlDType::F32,
    )?);
    let mut tensors = HashMap::from([(
        "rope_freqs.weight".to_owned(),
        Arc::new(QTensor::quantize(
            &Tensor::new(&factors, &Device::Cpu)?,
            GgmlDType::F32,
        )?),
    )]);
    for layer in 1..5 {
        for name in ["attn_norm", "ffn_norm"] {
            tensors.insert(format!("blk.{layer}.{name}.weight"), Arc::clone(&norm));
        }
        for name in [
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ] {
            tensors.insert(format!("blk.{layer}.{name}.weight"), Arc::clone(&matrix));
        }
    }
    let mut metadata = rope_test_metadata()?;
    metadata.rope_theta = 65_536.;
    metadata.rope_theta_swa = 4096.;
    // RoPE/NoPE and chunked/full schedules are independent. Exercise both
    // base frequencies in a stage that does not own embeddings or the head.
    metadata.sliding_window_pattern = vec![false, true, false, true, false, false];
    let stage = Llama4Stage::from_stage_tensors(
        StageTensors {
            range: LayerRange::try_new(1, 5, 6)?,
            metadata,
            tensors,
            raw_tensors: HashMap::new(),
            raw_source: None,
            device: Device::Cpu,
        },
        None,
    )?;
    let input = (0..32).map(|i| (i as f32 - 15.) / 19.).collect::<Vec<_>>();
    let x = Tensor::from_vec(input.clone(), (1, 1, 4, 8), &Device::Cpu)?;
    for (index, block) in stage.blocks.iter().enumerate() {
        let Llama4Block::Dense(block) = block else {
            panic!("expected dense fixture")
        };
        let attn = &block.attn;
        assert_eq!(
            values(attn.rope_freqs.as_ref().expect("stage must retain factors"))?,
            factors
        );
        assert_eq!(
            attn.rope_theta,
            if [0, 2].contains(&index) {
                4096.
            } else {
                65_536.
            }
        );
        for start_pos in [53, 8189] {
            let actual = apply_rotary_emb(
                &x,
                start_pos,
                &attn.rope_cos,
                &attn.rope_sin,
                attn.rope_theta,
                attn.rope_freqs.as_ref(),
            )?;
            let expected = rope_scalar_reference(&input, 8, start_pos, attn.rope_theta, &factors);
            assert_close(&values(&actual)?, &expected, 2e-6);
        }
    }
    Ok(())
}
