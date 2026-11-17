use super::*;

fn matrix(rows: usize, width: usize, seed: usize, device: &Device) -> Result<DeepSeek4QMatMul> {
    let bytes = (0..rows * width)
        .flat_map(|i| ((((i * 17 + seed * 7) % 71) as f32 - 35.) / 300.).to_le_bytes())
        .collect::<Vec<_>>();
    DeepSeek4QMatMul::new_on_device(
        Arc::new(RawTensor::new(
            vec![rows, width],
            DeepSeek4DType::F32,
            Tensor::from_slice(&bytes, bytes.len(), device)?,
        )?),
        device,
    )
}

fn mix(seed: usize, device: &Device) -> Result<HcMix> {
    Ok(HcMix {
        projection: matrix(24, 32, seed, device)?,
        base: Tensor::zeros(24, DType::F32, device)?,
        scale: Tensor::ones(3, DType::F32, device)?,
        norm: V4RmsNorm::ones(32, 1e-6, device)?,
        stream_count: 4,
        hidden_dim: 8,
        sinkhorn_iterations: 2,
        eps: 1e-6,
        fused_sinkhorn: true,
        fused_post: true,
        fused_pre: true,
    })
}

#[cfg(target_os = "macos")]
#[test]
fn hc_fusion_preserves_stream_mixing_and_iteration_fallback() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    let mut hc = mix(17, &device)?;
    let bits = |x: Tensor| -> Result<Vec<u32>> {
        Ok(x.flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    };
    for rows in [1, 3, 8, 32] {
        let residual = Tensor::from_vec(
            (0..rows * 32)
                .map(|i| ((i * 13 % 31) as f32 - 15.) / 7.3)
                .collect::<Vec<_>>(),
            (1, rows, 4, 8),
            &device,
        )?;
        for iterations in [1, 2, 20, 128, 129] {
            hc.sinkhorn_iterations = iterations;
            hc.fused_sinkhorn = false;
            hc.fused_post = false;
            hc.fused_pre = false;
            let (update, post, comb) = hc.pre(&residual)?;
            let expected = bits(hc.post(&update, &residual, &post, &comb)?)?;
            hc.fused_sinkhorn = true;
            hc.fused_post = true;
            hc.fused_pre = true;
            let (update2, post2, comb2) = hc.pre(&residual)?;
            assert_eq!(
                bits(hc.post(&update2, &residual, &post2, &comb2)?)?,
                expected
            );
            assert_eq!(bits(update2)?, bits(update)?);
            assert_eq!(bits(post2)?, bits(post)?);
            assert_eq!(bits(comb2)?, bits(comb)?);
        }
    }
    Ok(())
}

fn fixture(device: &Device) -> Result<DeepSeek4Stage> {
    let mut blocks = Vec::new();
    for (layer, ratio) in [0, 4, 128].into_iter().enumerate() {
        let experts = (0..4)
            .map(|expert| {
                Ok(V4Expert {
                    gate: matrix(16, 8, expert + layer, device)?,
                    up: matrix(16, 8, expert + layer + 5, device)?,
                    down: matrix(8, 16, expert + layer + 8, device)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        blocks.push(V4Block {
            attention: attention::tests::fixture(ratio, device)?,
            attn_norm: V4RmsNorm::ones(8, 1e-6, device)?,
            attn_hc: mix(layer, device)?,
            ffn_norm: V4RmsNorm::ones(8, 1e-6, device)?,
            ffn_hc: mix(layer + 7, device)?,
            moe: V4Moe {
                router: matrix(4, 8, layer, device)?,
                hash_selected_experts: if layer == 0 {
                    Some(Tensor::new(
                        &[[0f32, 2.], [3., 1.], [1., 0.], [2., 3.]],
                        device,
                    )?)
                } else {
                    None
                },
                experts,
                shared_gate: matrix(16, 8, 11, device)?,
                shared_up: matrix(16, 8, 12, device)?,
                shared_down: matrix(8, 16, 13, device)?,
                bias: Some(Tensor::new(&[0.1f32, -0.2, 0.3, -0.1], device)?),
                active_experts: 2,
                scale: 1.2,
                normalize_weights: true,
                clamp: 1.,
                shared_clamp: 0.5,
                group_experts: true,
            },
        });
    }
    Ok(DeepSeek4Stage {
        embedding: Some(V4Embedding::Dense(Tensor::from_vec(
            (0..32)
                .map(|i| ((i * 13 % 31) as f32 - 15.) / 20.)
                .collect(),
            (4, 8),
            device,
        )?)),
        blocks,
        output_norm: Some(V4RmsNorm::ones(8, 1e-6, device)?),
        lm_head: Some(matrix(4, 8, 17, device)?),
        output_hc: Some(HcHead {
            projection: matrix(4, 32, 19, device)?,
            base: Tensor::zeros(4, DType::F32, device)?,
            scale: Tensor::ones(1, DType::F32, device)?,
            norm: V4RmsNorm::ones(32, 1e-6, device)?,
            stream_count: 4,
            hidden_dim: 8,
            eps: 1e-6,
        }),
        is_head: true,
        is_tail: true,
        model_hidden_dim: 8,
        stage_hidden_dim: 32,
        stream_count: 4,
        num_layers: 3,
        vocab_size: 4,
        context_length: 512,
        decode_batch_enabled: true,
        device: device.clone(),
    })
}

fn clone_cache(cache: &[Option<LayerCache>]) -> Result<Vec<Option<LayerCache>>> {
    cache
        .iter()
        .map(|slot| {
            slot.as_ref()
                .map(super::super::super::checkpoint_layer_cache)
                .transpose()
        })
        .collect()
}

fn empty_cache() -> Vec<Option<LayerCache>> {
    (0..3).map(|_| None).collect()
}

fn scalar(
    model: &DeepSeek4Stage,
    token: u32,
    cache: &mut [Option<LayerCache>],
    pos: usize,
) -> Result<Tensor> {
    match model.forward(StageInput::Tokens(vec![token]), cache, 1, pos)? {
        StageOutput::Logits(logits) => Ok(logits),
        _ => anyhow::bail!("expected logits"),
    }
}

fn exact(a: &Tensor, b: &Tensor) -> Result<()> {
    let a = a.to_vec1::<f32>()?;
    let b = b.to_vec1::<f32>()?;
    assert!(a.iter().all(|v| v.is_finite()));
    assert_eq!(a, b);
    Ok(())
}

fn check_batch(device: &Device) -> Result<()> {
    let model = fixture(device)?;
    assert!(model.supports_decode_batch());
    assert!(!model.decode_batch_uses_packed_cache());
    let mut caches = (0..3).map(|_| empty_cache()).collect::<Vec<_>>();
    // Fork right before both compressed-pool boundaries and the window roll.
    let mut prefix = empty_cache();
    for pos in 0..127 {
        scalar(&model, (pos % 4) as u32, &mut prefix, pos)?;
        if pos == 2 {
            caches[0] = clone_cache(&prefix)?;
        }
        if pos == 4 {
            caches[1] = clone_cache(&prefix)?;
        }
    }
    caches[2] = clone_cache(&prefix)?;
    let saved = clone_cache(&prefix)?;
    let mut expected = caches
        .iter()
        .map(|c| clone_cache(c))
        .collect::<Result<Vec<_>>>()?;
    let mut positions = [3, 5, 127];
    let mut batch_cache = clone_cache(&prefix)?;
    for step in 0..5 {
        let tokens = [step % 4, (step + 1) % 4, (step + 2) % 4];
        let logits = model.forward_decode_batch(
            &tokens,
            &mut caches.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>(),
            &mut batch_cache,
            &positions,
        )?;
        assert!(batch_cache.iter().all(Option::is_none));
        for lane in 0..3 {
            exact(
                &logits[lane],
                &scalar(&model, tokens[lane], &mut expected[lane], positions[lane])?,
            )?;
            positions[lane] += 1;
        }
    }
    // Scalar continuation checks the published batch state, not only logits.
    for lane in 0..3 {
        exact(
            &scalar(&model, 2, &mut caches[lane], positions[lane])?,
            &scalar(&model, 2, &mut expected[lane], positions[lane])?,
        )?;
    }
    // Saved prompt state must still be usable after forks diverge.
    exact(
        &scalar(&model, 1, &mut prefix, 127)?,
        &scalar(&model, 1, &mut clone_cache(&saved)?, 127)?,
    )?;
    // A single-lane batch and a fresh request use the same path safely.
    let mut fresh = empty_cache();
    let output = model.forward_decode_batch(&[3], &mut [&mut fresh], &mut batch_cache, &[0])?;
    exact(&output[0], &scalar(&model, 3, &mut empty_cache(), 0)?)?;
    Ok(())
}

#[test]
fn v4_decode_batch_matches_scalar_cpu() -> Result<()> {
    check_batch(&Device::Cpu)
}

#[cfg(target_os = "macos")]
#[test]
fn v4_decode_batch_matches_scalar_metal() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    check_batch(&device)
}

#[test]
fn v4_decode_batch_errors_leave_all_caches_intact() -> Result<()> {
    let mut model = fixture(&Device::Cpu)?;
    let mut first = empty_cache();
    scalar(&model, 1, &mut first, 0)?;
    let saved = clone_cache(&first)?;
    let mut second = clone_cache(&first)?;
    let mut packed = clone_cache(&first)?;
    for (tokens, positions) in [
        (vec![], vec![]),
        (vec![0], vec![1]),
        (vec![0, 1], vec![1]),
        (vec![0, 4], vec![1, 1]),
        (vec![0, 1], vec![1, 512]),
        (vec![0, 1], vec![1, 2]),
    ] {
        assert!(
            model
                .forward_decode_batch(
                    &tokens,
                    &mut [&mut first, &mut second],
                    &mut packed,
                    &positions
                )
                .is_err()
        );
    }
    let mut empty = empty_cache();
    assert!(
        model
            .forward_decode_batch(&[0, 1], &mut [&mut first, &mut empty], &mut packed, &[1, 1])
            .is_err()
    );
    let mut wrong = empty_cache();
    wrong[0] = Some(LayerCache::Attention(super::super::super::KvCache::F16 {
        k: Tensor::zeros((1, 1, 1, 8), DType::F32, &Device::Cpu)?,
        v: Tensor::zeros((1, 1, 1, 8), DType::F32, &Device::Cpu)?,
    }));
    assert!(
        model
            .forward_decode_batch(&[0, 1], &mut [&mut first, &mut wrong], &mut packed, &[1, 1])
            .is_err()
    );
    assert!(
        model
            .forward_decode_batch(
                &[0, 1],
                &mut [&mut first, &mut second[..2]],
                &mut packed,
                &[1, 1]
            )
            .is_err()
    );
    assert!(
        model
            .forward_decode_batch(
                &[0, 1],
                &mut [&mut first, &mut second],
                &mut packed[..2],
                &[1, 1]
            )
            .is_err()
    );
    // Force a late failure after all attention caches have advanced privately.
    let head = model.lm_head.take();
    assert!(
        model
            .forward_decode_batch(
                &[0, 1],
                &mut [&mut first, &mut second],
                &mut packed,
                &[1, 1]
            )
            .is_err()
    );
    model.lm_head = head;
    for cache in [&mut first, &mut second, &mut packed] {
        exact(
            &scalar(&model, 2, cache, 1)?,
            &scalar(&model, 2, &mut clone_cache(&saved)?, 1)?,
        )?;
    }
    model.decode_batch_enabled = false;
    assert!(!model.supports_decode_batch());
    assert!(
        model
            .forward_decode_batch(&[0], &mut [&mut empty], &mut packed, &[0])
            .is_err()
    );
    model.decode_batch_enabled = true;
    model.is_tail = false;
    assert!(!model.supports_decode_batch());
    model.is_tail = true;
    model.is_head = false;
    assert!(!model.supports_decode_batch());
    Ok(())
}
