use super::*;
use candle_core::quantized::GgmlDType;

fn weights(rows: usize, cols: usize, salt: usize, device: &Device) -> Result<Arc<QTensor>> {
    let values = (0..rows * cols)
        .map(|i| {
            let n = (i + salt)
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            ((n ^ (n >> 13)) % 1009) as f32 / 1009.0 * 0.08 - 0.04
        })
        .collect::<Vec<_>>();
    let source = Tensor::from_vec(values, (rows, cols), &Device::Cpu)?;
    Ok(Arc::new(QTensor::quantize_onto(
        &source,
        GgmlDType::Q4K,
        device,
    )?))
}

fn matrix(rows: usize, cols: usize, salt: usize, device: &Device) -> Result<Qwen35QMatMul> {
    Qwen35QMatMul::from_qtensor(weights(rows, cols, salt, device)?, device)
}

fn norm(width: usize, device: &Device) -> Result<RmsNorm> {
    Ok(RmsNorm {
        weight: Tensor::ones(width, DType::F32, device)?,
        eps: 1e-6,
    })
}

fn mlp(device: &Device) -> Result<Mlp> {
    Ok(Mlp {
        input_projections: MlpInputProjections::Separate {
            gate: matrix(256, 256, 1, device)?,
            up: matrix(256, 256, 2, device)?,
        },
        down_proj: matrix(256, 256, 3, device)?,
    })
}

fn stage(device: &Device) -> Result<Qwen35Stage> {
    let dn = DeltaNetBlock {
        attn_norm: norm(256, device)?,
        projections: DeltaNetProjections {
            qkv_gate: HomogeneousProjectionPair::Separate {
                first: matrix(1024, 256, 4, device)?,
                second: matrix(512, 256, 5, device)?,
            },
            alpha_beta: HomogeneousProjectionPair::Separate {
                first: matrix(4, 256, 6, device)?,
                second: matrix(4, 256, 7, device)?,
            },
        },
        conv1d_weight: (Tensor::ones((1024, 4), DType::F32, device)? * 0.25)?,
        a_log: (Tensor::ones(4, DType::F32, device)? * -0.5)?,
        dt_bias: Tensor::zeros(4, DType::F32, device)?,
        ssm_norm: norm(128, device)?,
        out_proj: matrix(256, 512, 8, device)?,
        ffn_norm: norm(256, device)?,
        ffn: mlp(device)?,
        num_k_heads: 2,
        num_v_heads: 4,
        key_head_dim: 128,
        value_head_dim: 128,
        key_dim: 256,
        value_dim: 512,
        conv_dim: 1024,
        conv_kernel: 4,
    };
    let (rope_cos, rope_sin) = precompute_rope_cos_sin(32, 10000., 128, device)?;
    let attention = GatedAttentionBlock {
        attn_norm: norm(256, device)?,
        input_projections: AttentionInputProjections::Separate {
            q: matrix(512, 256, 9, device)?,
            k: matrix(128, 256, 10, device)?,
            v: matrix(128, 256, 11, device)?,
        },
        o_proj: matrix(256, 256, 12, device)?,
        q_norm: norm(64, device)?,
        k_norm: norm(64, device)?,
        ffn_norm: norm(256, device)?,
        ffn: mlp(device)?,
        rope_cos,
        rope_sin,
        mrope_sections: None,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 64,
        rope_dims: 32,
        quant_config: None,
    };
    Ok(Qwen35Stage {
        embedding: Some(weights(64, 256, 13, device)?),
        blocks: vec![
            Qwen35Block::DeltaNet(dn),
            Qwen35Block::GatedAttention(attention),
        ],
        output_norm: Some(norm(256, device)?),
        lm_head: Some(matrix(64, 256, 14, device)?),
        is_head: true,
        is_tail: true,
        hidden_dim: 256,
        num_layers: 2,
        vocab_size: 64,
        layer_indices: vec![0, 1],
        full_attention_interval: 2,
        device: device.clone(),
    })
}

fn close(actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims());
    let actual = actual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let expected = expected
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut max_error = 0f32;
    for (a, b) in actual.iter().zip(&expected) {
        assert!(a.is_finite() && b.is_finite());
        max_error = max_error.max((a - b).abs());
    }
    assert!(
        max_error <= tolerance,
        "max error {max_error} > {tolerance}"
    );
    Ok(())
}

fn dense(kv: &KvCache) -> Result<(Tensor, Tensor)> {
    match kv {
        KvCache::F16 { k, v } => Ok((k.clone(), v.clone())),
        KvCache::PreallocatedF16(cache) => cache.active_f16(),
        KvCache::Quantized(_) => anyhow::bail!("unexpected quantized cache"),
    }
}

fn same_cache(
    actual: &[Option<LayerCache>],
    expected: &[Option<LayerCache>],
    tol: f32,
) -> Result<()> {
    for (a, b) in actual.iter().zip(expected) {
        match (a, b) {
            (None, None) => {}
            (Some(LayerCache::DeltaNet(a)), Some(LayerCache::DeltaNet(b))) => {
                close(&a.conv_state, &b.conv_state, tol)?;
                close(&a.recurrent_state, &b.recurrent_state, tol)?;
                assert!(a.speculative_state.is_none());
            }
            (Some(LayerCache::Attention(a)), Some(LayerCache::Attention(b))) => {
                let (ak, av) = dense(a)?;
                let (bk, bv) = dense(b)?;
                close(&ak, &bk, tol)?;
                close(&av, &bv, tol)?;
            }
            _ => anyhow::bail!("cache types differ"),
        }
    }
    Ok(())
}

fn copy_cache(cache: &[Option<LayerCache>]) -> Result<Vec<Option<LayerCache>>> {
    cache
        .iter()
        .map(|slot| {
            slot.as_ref()
                .map(super::super::checkpoint_layer_cache)
                .transpose()
        })
        .collect()
}

fn check_cache_only_prefill(device: &Device) -> Result<()> {
    let stage = stage(device)?;
    let mut expected = vec![None, None];
    let mut actual = vec![None, None];
    for (start, tokens) in [(0, vec![1, 2, 3]), (3, vec![4]), (4, vec![5, 6])] {
        let input = StageInput::Tokens(tokens.clone());
        stage.forward_prefill(input, &mut expected, tokens.len(), start)?;
        let output = stage.forward_prefill_cache_only(
            StageInput::Tokens(tokens.clone()),
            &mut actual,
            tokens.len(),
            start,
        )?;
        assert!(matches!(output, StageOutput::Activations(_)));
        same_cache(&actual, &expected, 0.)?;
    }
    let StageOutput::Logits(expected) =
        stage.forward_prefill(StageInput::Tokens(vec![7]), &mut expected, 1, 6)?
    else {
        panic!("no logits")
    };
    let StageOutput::Logits(actual) =
        stage.forward_prefill(StageInput::Tokens(vec![7]), &mut actual, 1, 6)?
    else {
        panic!("no logits")
    };
    close(&actual, &expected, 0.)
}

#[test]
fn cache_only_prefill_preserves_qwen_state_and_logits() -> Result<()> {
    check_cache_only_prefill(&Device::Cpu)
}

#[test]
#[cfg(target_os = "macos")]
fn metal_cache_only_prefill_preserves_qwen_state_and_logits() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    check_cache_only_prefill(&device)
}

fn check_decode(device: &Device) -> Result<()> {
    let stage = stage(device)?;
    let tolerance = 1e-4;
    for initial_positions in [[3, 3, 3], [0, 3, 7]] {
        let mut positions = initial_positions;
        let mut scalar = Vec::new();
        let mut batched = Vec::new();
        for (lane, &len) in positions.iter().enumerate() {
            let mut cache = vec![None, None];
            if len > 0 {
                let prompt = (0..len).map(|i| (i + lane * 9) as u32 % 64).collect();
                stage.forward(StageInput::Tokens(prompt), &mut cache, len, 0)?;
            }
            batched.push(copy_cache(&cache)?);
            scalar.push(cache);
        }
        let mut packed = vec![None, None];
        // Reorder, remove, then rejoin requests without touching idle state.
        for lanes in [vec![0, 1, 2], vec![2, 0], vec![1, 2, 0], vec![1]] {
            let tokens = lanes
                .iter()
                .map(|&lane| (positions[lane] + lane * 11) as u32 % 64)
                .collect::<Vec<_>>();
            let before = batched
                .iter()
                .map(|cache| copy_cache(cache))
                .collect::<Result<Vec<_>>>()?;
            let mut selected = lanes
                .iter()
                .map(|&lane| copy_cache(&batched[lane]))
                .collect::<Result<Vec<_>>>()?;
            let outputs = stage.forward_decode_batch(
                &tokens,
                &mut selected
                    .iter_mut()
                    .map(Vec::as_mut_slice)
                    .collect::<Vec<_>>(),
                &mut packed,
                &lanes
                    .iter()
                    .map(|&lane| positions[lane])
                    .collect::<Vec<_>>(),
            )?;
            assert!(packed.iter().all(Option::is_none));
            for (row, (&lane, next)) in lanes.iter().zip(selected).enumerate() {
                let StageOutput::Logits(expected) = stage.forward(
                    StageInput::Tokens(vec![tokens[row]]),
                    &mut scalar[lane],
                    1,
                    positions[lane],
                )?
                else {
                    panic!("expected logits")
                };
                close(&outputs[row], &expected, tolerance)?;
                same_cache(&next, &scalar[lane], tolerance)?;
                batched[lane] = next;
                positions[lane] += 1;
            }
            for lane in 0..3 {
                if !lanes.contains(&lane) {
                    same_cache(&batched[lane], &before[lane], 0.)?;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn qwen_decode_batch_cpu_keeps_independent_state() -> Result<()> {
    check_decode(&Device::Cpu)
}

fn check_speculative_batch(device: &Device) -> Result<()> {
    let stage = stage(device)?;
    for lanes in [2, 4] {
        for rows in [8] {
            let mut scalar = Vec::new();
            for lane in 0..lanes {
                let mut cache = vec![None, None];
                stage.forward(
                    StageInput::Tokens(vec![lane as u32; lane + 2]),
                    &mut cache,
                    lane + 2,
                    0,
                )?;
                scalar.push(cache);
            }
            let mut batch = scalar
                .iter()
                .map(|c| copy_cache(c))
                .collect::<Result<Vec<_>>>()?;
            let positions = (0..lanes).map(|lane| lane + 2).collect::<Vec<_>>();
            let inputs = (0..lanes)
                .map(|lane| (0..rows).map(|row| (lane * 9 + row) as u32).collect())
                .collect::<Vec<Vec<u32>>>();
            let outputs = stage.forward_speculative_batch(
                &inputs,
                &mut batch.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>(),
                &positions,
                &[0, 1],
            )?;
            for (lane, output) in outputs.iter().enumerate() {
                let (StageOutput::Logits(expected), captures) = stage.forward_with_layer_inputs(
                    StageInput::Tokens(inputs[lane].clone()),
                    &mut scalar[lane],
                    rows,
                    positions[lane],
                    &[0, 1],
                    true,
                )?
                else {
                    unreachable!()
                };
                close(&output.logits, &expected, 2e-4)?;
                for (a, b) in output.captures.iter().zip(captures) {
                    close(a, &b, 2e-4)?;
                }
                // Each request commits a different prefix of its own block.
                let accepted_row = if lane == lanes - 1 { rows - 1 } else { lane };
                for cache in [&mut batch[lane], &mut scalar[lane]] {
                    for slot in cache.iter_mut().flatten() {
                        super::super::commit_speculative_layer_cache(
                            slot,
                            accepted_row,
                            rows - accepted_row - 1,
                        )?;
                    }
                }
                same_cache(&batch[lane], &scalar[lane], 2e-4)?;
                let token = (lane + 31) as u32;
                let position = positions[lane] + accepted_row + 1;
                let StageOutput::Logits(a) = stage.forward(
                    StageInput::Tokens(vec![token]),
                    &mut batch[lane],
                    1,
                    position,
                )?
                else {
                    unreachable!()
                };
                let StageOutput::Logits(b) = stage.forward(
                    StageInput::Tokens(vec![token]),
                    &mut scalar[lane],
                    1,
                    position,
                )?
                else {
                    unreachable!()
                };
                close(&a, &b, 2e-4)?;
                same_cache(&batch[lane], &scalar[lane], 2e-4)?;
            }
        }
    }
    Ok(())
}

#[test]
fn qwen_speculative_batch_cpu_preserves_rollback_and_captures() -> Result<()> {
    check_speculative_batch(&Device::Cpu)
}

#[cfg(target_os = "macos")]
#[test]
fn qwen_speculative_batch_metal_preserves_rollback_and_captures() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    check_speculative_batch(&device)
}

#[test]
fn qwen_speculative_batch_errors_keep_all_lane_state() -> Result<()> {
    let mut stage = stage(&Device::Cpu)?;
    let mut a = vec![None, None];
    let mut b = vec![None, None];
    stage.forward(StageInput::Tokens(vec![1, 2]), &mut a, 2, 0)?;
    stage.forward(StageInput::Tokens(vec![3]), &mut b, 1, 0)?;
    let before_a = copy_cache(&a)?;
    let before_b = copy_cache(&b)?;
    for (inputs, positions, layers) in [
        (vec![], vec![], vec![]),
        (vec![vec![4, 5], vec![6]], vec![2, 1], vec![0]),
        (vec![vec![4; 8], vec![6; 8]], vec![2], vec![0]),
        (vec![vec![4; 8], vec![6; 8]], vec![2, 1], vec![1, 0]),
        (vec![vec![4; 8], vec![6; 8]], vec![2, 1], vec![2]),
    ] {
        assert!(
            stage
                .forward_speculative_batch(&inputs, &mut [&mut a, &mut b], &positions, &layers)
                .is_err()
        );
        same_cache(&a, &before_a, 0.)?;
        same_cache(&b, &before_b, 0.)?;
    }
    stage.lm_head = Some(matrix(64, 512, 0, &Device::Cpu)?);
    assert!(
        stage
            .forward_speculative_batch(
                &[vec![4; 8], vec![6; 8]],
                &mut [&mut a, &mut b],
                &[2, 1],
                &[0, 1]
            )
            .is_err()
    );
    same_cache(&a, &before_a, 0.)?;
    same_cache(&b, &before_b, 0.)?;
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn qwen_decode_batch_metal_keeps_independent_state() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    check_decode(&device)
}

#[test]
fn qwen_decode_batch_error_does_not_publish_partial_state() -> Result<()> {
    let mut stage = stage(&Device::Cpu)?;
    let mut first = vec![None, None];
    let mut second = vec![None, None];
    stage.forward(StageInput::Tokens(vec![1, 2]), &mut first, 2, 0)?;
    stage.forward(StageInput::Tokens(vec![3]), &mut second, 1, 0)?;
    let first_before = copy_cache(&first)?;
    let second_before = copy_cache(&second)?;
    // Fail at the head, after both recurrent and KV layers ran.
    stage.lm_head = Some(matrix(64, 512, 0, &Device::Cpu)?);
    let mut packed = copy_cache(&first)?;
    let packed_before = copy_cache(&packed)?;
    assert!(
        stage
            .forward_decode_batch(
                &[4, 5],
                &mut [&mut first, &mut second],
                &mut packed,
                &[2, 1]
            )
            .is_err()
    );
    same_cache(&first, &first_before, 0.)?;
    same_cache(&second, &second_before, 0.)?;
    same_cache(&packed, &packed_before, 0.)?;
    Ok(())
}

#[test]
fn qwen_decode_batch_rejects_bad_geometry_and_cache_types() -> Result<()> {
    let stage = stage(&Device::Cpu)?;
    let mut first = vec![None, None];
    let mut packed = vec![None, None];
    assert!(
        stage
            .forward_decode_batch(&[], &mut [], &mut packed, &[])
            .is_err()
    );
    assert!(
        stage
            .forward_decode_batch(&[1], &mut [], &mut packed, &[0])
            .is_err()
    );
    assert!(
        stage
            .forward_decode_batch(&[1], &mut [&mut first], &mut packed, &[])
            .is_err()
    );
    assert!(
        stage
            .forward_decode_batch(&[1], &mut [&mut first[..1]], &mut packed, &[0])
            .is_err()
    );
    stage.forward(StageInput::Tokens(vec![1]), &mut first, 1, 0)?;
    first.swap(0, 1);
    assert!(
        stage
            .forward_decode_batch(&[2], &mut [&mut first], &mut packed, &[1])
            .is_err()
    );
    assert!(matches!(first[0], Some(LayerCache::Attention(_))));
    assert!(packed.iter().all(Option::is_none));
    Ok(())
}

#[test]
fn qwen_decode_batch_excludes_split_stages_and_quantized_kv() -> Result<()> {
    let mut stage = stage(&Device::Cpu)?;
    assert!(stage.supports_decode_batch());
    assert!(!stage.decode_batch_uses_packed_cache());
    stage.is_tail = false;
    assert!(!stage.supports_decode_batch());
    stage.is_tail = true;
    stage.is_head = false;
    assert!(!stage.supports_decode_batch());
    stage.is_head = true;
    let Qwen35Block::GatedAttention(attention) = &mut stage.blocks[1] else {
        unreachable!()
    };
    attention.quant_config = Some(Arc::new(TurboQuantConfig::new(8, 64, 42)));
    assert!(!stage.supports_decode_batch());
    Ok(())
}
