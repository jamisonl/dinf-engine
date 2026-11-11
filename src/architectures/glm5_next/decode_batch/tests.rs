use super::*;

#[cfg(target_os = "macos")]
#[test]
fn glm_decode_projections_match_scalar_bits() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    for dtype in [
        GgmlDType::F32,
        GgmlDType::F16,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4_0,
        GgmlDType::Q4K,
        GgmlDType::Q6K,
        GgmlDType::Q8_0,
    ] {
        let weights = Arc::new(QTensor::quantize_onto(
            &values(256, 512, 42, &Device::Cpu)?,
            dtype,
            &device,
        )?);
        let matrix = GlmQMatMul::from_qtensor(weights, &device)?;
        let source = values(9, 512, 23, &device)?.reshape((1, 9, 512))?;
        for count in [1, 2, 3, 4, 8] {
            let input = source.narrow(1, 1, count)?;
            for input in [
                input.clone(),
                input.squeeze(0)?,
                input.transpose(1, 2)?.contiguous()?.transpose(1, 2)?,
            ] {
                let actual = matrix.forward_decode_rows(&input)?;
                for row in 0..count {
                    let expected =
                        matrix.forward(&input.narrow(D::Minus2, row, 1)?.contiguous()?)?;
                    assert_eq!(
                        bits(&actual.narrow(D::Minus2, row, 1)?)?,
                        bits(&expected)?,
                        "dtype {dtype:?}, count {count}, row {row}"
                    );
                }
            }
        }
        for shape in [
            vec![512],
            vec![1, 1, 1, 512],
            vec![1, 0, 512],
            vec![1, 2, 256],
        ] {
            assert!(
                matrix
                    .forward_decode_rows(&Tensor::zeros(shape, DType::F32, &device)?)
                    .is_err()
            );
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn glm_decode_indexed_experts_and_shared_projections_match_scalar_bits() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    for dtype in [GgmlDType::Q2K, GgmlDType::Q3K, GgmlDType::Q4K] {
        let packed = Arc::new(QTensor::quantize(
            &values(4 * 256, 256, 27, &Device::Cpu)?.reshape((4, 256, 256))?,
            dtype,
        )?);
        let indexed = || IndexedMoeMatMul::from_qtensor(&packed, 4, 256, 256, &device);
        let shared_weight = Arc::new(QTensor::quantize_onto(
            &values(256, 256, 31, &Device::Cpu)?,
            dtype,
            &device,
        )?);
        let shared = GlmQMatMul::from_qtensor(shared_weight, &device)?;
        let moe = MoeLayer {
            gate: matrix(4, 256, 29, &device)?,
            bias: Tensor::new(&[0.1f32, -0.2, 0.3, -0.1], &device)?,
            experts: MoeExperts::Indexed(Box::new(IndexedExperts {
                gate: indexed()?,
                up: indexed()?,
                down: indexed()?,
                clamp: 0.5,
            })),
            active: 2,
            scale: 1.2,
            shared: DenseMlp {
                gate: shared.clone(),
                up: shared.clone(),
                down: shared,
                clamp: 0.5,
            },
        };
        let source = values(67, 256, 11, &device)?.reshape((1, 67, 256))?;
        for rows in [1, 2, 3, 4, 8, 33, 65] {
            let input = source.narrow(1, 1, rows)?;
            let actual = moe.forward_decode_rows(&input)?;
            for row in 0..rows {
                let expected = moe.forward(&input.narrow(1, row, 1)?, false)?;
                assert_eq!(
                    bits(&actual.narrow(1, row, 1)?)?,
                    bits(&expected)?,
                    "{dtype:?}, {rows} rows, lane {row}"
                );
            }
        }
    }
    Ok(())
}

fn values(rows: usize, width: usize, seed: usize, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(
        (0..rows * width)
            .map(|i| (((i * 17 + seed * 7) % 71) as f32 - 35.) / 300.)
            .collect::<Vec<_>>(),
        (rows, width),
        device,
    )?)
}

fn matrix(rows: usize, width: usize, seed: usize, device: &Device) -> Result<GlmQMatMul> {
    Ok(GlmQMatMul::Candle(QMatMul::Tensor(values(
        rows, width, seed, device,
    )?)))
}

fn mix(seed: usize, device: &Device) -> Result<HcMix> {
    Ok(HcMix {
        projection: matrix(24, 32, seed, device)?,
        base: Tensor::zeros(24, DType::F32, device)?,
        scale: Tensor::new(&[0.7f32, 0.8, 0.6], device)?,
        norm: RmsNorm::ones(32, 1e-6, device)?,
        streams: 4,
        hidden: 8,
        sinkhorn_iterations: 2,
        eps: 1e-6,
        fused_sinkhorn: true,
        fused_post: true,
        fused_pre: true,
        reuse_projection: true,
    })
}

fn dense(seed: usize, device: &Device) -> Result<DenseMlp> {
    Ok(DenseMlp {
        gate: matrix(16, 8, seed, device)?,
        up: matrix(16, 8, seed + 1, device)?,
        down: matrix(8, 16, seed + 2, device)?,
        clamp: 0.5,
    })
}

fn fixture(device: &Device) -> Result<Glm5NextStage> {
    let kda = KdaAttention {
        q: matrix(8, 8, 1, device)?,
        k: matrix(8, 8, 2, device)?,
        v: matrix(8, 8, 3, device)?,
        out: matrix(8, 8, 4, device)?,
        f_a: matrix(4, 8, 5, device)?,
        f_b: matrix(8, 4, 6, device)?,
        beta: matrix(2, 8, 7, device)?,
        a: Tensor::ones(2, DType::F32, device)?,
        dt_bias: Tensor::zeros(8, DType::F32, device)?,
        g_a: matrix(4, 8, 8, device)?,
        g_b: matrix(8, 4, 9, device)?,
        output_norm: RmsNorm::ones(4, 1e-6, device)?,
        conv_weight: Tensor::full(0.2f32, (24, 3), device)?,
        heads: 2,
        head_dim: 4,
        inner: 8,
        kernel: 3,
        gate_lower_bound: -5.,
        norm_eps: 1e-6,
    };
    let dsa = DsaAttention {
        q_a: matrix(4, 8, 11, device)?,
        q_a_norm: RmsNorm::ones(4, 1e-6, device)?,
        q_b: matrix(8, 4, 12, device)?,
        kv_a: matrix(4, 8, 13, device)?,
        kv_a_norm: RmsNorm::ones(4, 1e-6, device)?,
        k_b: HeadwiseMatMul::Dense(values(8, 4, 14, device)?.reshape((2, 4, 4))?),
        v_b: HeadwiseMatMul::Dense(values(8, 4, 15, device)?.reshape((2, 4, 4))?),
        out: matrix(8, 8, 16, device)?,
        index_q: matrix(8, 4, 17, device)?,
        index_k: matrix(4, 8, 18, device)?,
        index_k_norm: LayerNorm {
            weight: Tensor::ones(4, DType::F32, device)?,
            bias: Tensor::zeros(4, DType::F32, device)?,
            eps: 1e-6,
        },
        index_weight: matrix(2, 8, 19, device)?,
        pool_gate: matrix(4, 8, 20, device)?,
        pool_ape: Tensor::zeros((2, 4), DType::F32, device)?,
        heads: 2,
        q_head_dim: 4,
        value_head_dim: 4,
        q_rank: 4,
        kv_rank: 4,
        index_heads: 2,
        index_dim: 4,
        index_top_k: 4,
        kpool: 2,
    };
    let experts = (0..4)
        .map(|i| {
            Ok(MoeExpert {
                gate: matrix(16, 8, i + 21, device)?,
                up: matrix(16, 8, i + 25, device)?,
                down: matrix(8, 16, i + 29, device)?,
                clamp: 0.5,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut blocks = Vec::new();
    for (i, (attention, ffn)) in [
        (
            GlmAttention::Kda(Box::new(kda)),
            GlmFfn::Dense(dense(31, device)?),
        ),
        (
            GlmAttention::Dsa(Box::new(dsa)),
            GlmFfn::Moe(MoeLayer {
                gate: matrix(4, 8, 32, device)?,
                bias: Tensor::new(&[0.1f32, -0.2, 0.3, -0.1], device)?,
                experts: MoeExperts::Split(experts),
                active: 2,
                scale: 1.2,
                shared: dense(33, device)?,
            }),
        ),
    ]
    .into_iter()
    .enumerate()
    {
        blocks.push(GlmBlock {
            attn_norm: RmsNorm::ones(8, 1e-6, device)?,
            attention,
            attn_hc: mix(i, device)?,
            ffn_norm: RmsNorm::ones(8, 1e-6, device)?,
            ffn,
            ffn_hc: mix(i + 7, device)?,
        });
    }
    Ok(Glm5NextStage {
        embedding: Some(Arc::new(QTensor::quantize_onto(
            &values(16, 8, 37, &Device::Cpu)?,
            GgmlDType::F32,
            device,
        )?)),
        blocks,
        output_norm: Some(RmsNorm::ones(8, 1e-6, device)?),
        output: Some(matrix(16, 8, 38, device)?),
        is_head: true,
        is_tail: true,
        hidden: 8,
        stage_hidden: 32,
        vocab: 16,
        context_length: 128,
        decode_batch_enabled: true,
        device: device.clone(),
    })
}

fn split_fixture(device: &Device) -> Result<(Glm5NextStage, Vec<crate::PipelineStage>)> {
    let mut full = fixture(device)?;
    full.blocks.extend(fixture(device)?.blocks);
    let mut head = fixture(device)?;
    head.blocks.truncate(1);
    head.is_tail = false;
    head.output = None;
    head.output_norm = None;
    let mut middle = fixture(device)?;
    middle.blocks.reverse();
    middle.is_head = false;
    middle.is_tail = false;
    middle.embedding = None;
    middle.output = None;
    middle.output_norm = None;
    let mut tail = fixture(device)?;
    tail.blocks.remove(0);
    tail.is_head = false;
    tail.embedding = None;
    let stages = [head, middle, tail]
        .into_iter()
        .zip([
            crate::StagePosition::Head,
            crate::StagePosition::Middle,
            crate::StagePosition::Tail,
        ])
        .map(|(model, position)| {
            let mut stage = crate::session_state_tests::test_stage();
            stage.cache = (0..model.blocks.len()).map(|_| None).collect();
            stage.arch = Box::new(model);
            stage.position = position;
            stage.hidden_dim = 32;
            stage.vocab_size = 16;
            stage.device = device.clone();
            stage.preallocated_kv_cache = true;
            stage
        })
        .collect();
    Ok((full, stages))
}

fn split_check(device: &Device) -> Result<()> {
    let (full, mut stages) = split_fixture(device)?;
    let mut states = stages
        .iter()
        .map(|stage| {
            (0..4)
                .map(|_| stage.new_session_state())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut expected = (0..4)
        .map(|_| (0..4).map(|_| None).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let mut positions = [1, 3, 4, 7];
    for lane in 0..4 {
        for pos in 0..positions[lane] {
            let token = (pos % 16) as u32;
            let mut input = StageInput::Tokens(vec![token]);
            for (stage, states) in stages.iter_mut().zip(&mut states) {
                let output = stage.prefill_with_session(&mut states[lane], input, 1, pos)?;
                input = match output {
                    StageOutput::Activations(x) => StageInput::activations(x),
                    StageOutput::Logits(x) => {
                        let StageOutput::Logits(reference) = full.forward_prefill(
                            StageInput::Tokens(vec![token]),
                            &mut expected[lane],
                            1,
                            pos,
                        )?
                        else {
                            unreachable!()
                        };
                        assert_eq!(bits(&x)?, bits(&reference)?);
                        StageInput::Tokens(vec![])
                    }
                };
            }
        }
    }
    for count in [4, 3, 2, 1, 4] {
        let mut inputs = (0..count)
            .map(|lane| StageInput::Tokens(vec![(positions[lane] % 16) as u32]))
            .collect::<Vec<_>>();
        // Reverse lane order independently each step. This also gives cached
        // requests different absolute positions at every split boundary.
        inputs.reverse();
        for (stage, states) in stages.iter_mut().zip(&mut states) {
            assert!(stage.supports_pipeline_decode_batch());
            assert!(!stage.supports_decode_batch());
            let outputs = stage.forward_pipeline_decode_batch_with_sessions(
                &mut states[..count].iter_mut().rev().collect::<Vec<_>>(),
                &inputs,
            )?;
            inputs = outputs
                .into_iter()
                .enumerate()
                .map(|(index, output)| match output {
                    StageOutput::Activations(x) => Ok(StageInput::activations(x)),
                    StageOutput::Logits(x) => {
                        let lane = count - 1 - index;
                        let reference = scalar(
                            &full,
                            &mut expected[lane],
                            (positions[lane] % 16) as u32,
                            positions[lane],
                        )?;
                        assert_eq!(
                            bits(&x)?,
                            bits(&reference)?,
                            "lane {lane} position {}",
                            positions[lane]
                        );
                        Ok(StageInput::Tokens(vec![]))
                    }
                })
                .collect::<Result<Vec<_>>>()?;
        }
        for position in &mut positions[..count] {
            *position += 1;
        }
    }
    // Cancelling a lane must not discard another lane. Resume scalar on the
    // survivor, then confirm every stage's aggregate charge drops on cleanup.
    for (stage, states) in stages.iter_mut().zip(&mut states) {
        stage.clear_session_state(&mut states[0]);
    }
    let mut input = StageInput::Tokens(vec![2]);
    for (stage, states) in stages.iter_mut().zip(&mut states) {
        input = match stage.forward_with_session(&mut states[1], input, 1, positions[1])? {
            StageOutput::Activations(x) => StageInput::activations(x),
            StageOutput::Logits(x) => {
                assert_eq!(
                    bits(&x)?,
                    bits(&scalar(&full, &mut expected[1], 2, positions[1])?)?
                );
                StageInput::Tokens(vec![])
            }
        };
        for state in states {
            stage.clear_session_state(state);
        }
        assert_eq!(stage.total_kv_cache_tokens(), 0);
        assert_eq!(stage.kv_token_budget.allocated(), 0);
    }
    Ok(())
}

#[test]
fn glm_pipeline_decode_batch_matches_full_cpu() -> Result<()> {
    split_check(&Device::Cpu)
}

#[cfg(target_os = "macos")]
#[test]
fn glm_pipeline_decode_batch_matches_full_metal() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    split_check(&device)
}

#[test]
fn glm_pipeline_batch_rejects_bad_inputs_and_keeps_budget() -> Result<()> {
    let (_, mut stages) = split_fixture(&Device::Cpu)?;
    let mut foreign = stages[0].new_session_state();
    let stage = &mut stages[1];
    stage.set_max_total_kv_tokens(512)?;
    let mut first = stage.new_session_state();
    let mut second = stage.new_session_state();
    let valid = || -> Result<StageInput> {
        Ok(StageInput::activations(Tensor::zeros(
            (1, 32),
            DType::F32,
            &Device::Cpu,
        )?))
    };
    for bad in [
        StageInput::Tokens(vec![1]),
        StageInput::activations(Tensor::zeros((2, 32), DType::F32, &Device::Cpu)?),
        StageInput::activations(Tensor::zeros((1, 32), DType::F16, &Device::Cpu)?),
        StageInput::activations_with_spans(
            Tensor::zeros((1, 32), DType::F32, &Device::Cpu)?,
            vec![crate::architectures::AttentionSpan { start: 0, len: 1 }],
        ),
    ] {
        assert!(
            stage
                .forward_pipeline_decode_batch_with_sessions(
                    &mut [&mut first, &mut second],
                    &[valid()?, bad]
                )
                .is_err()
        );
        assert_eq!(stage.session_next_position(&first), 0);
        assert_eq!(stage.total_kv_cache_tokens(), 0);
    }
    assert!(
        stage
            .forward_pipeline_decode_batch_with_sessions(
                &mut [&mut first, &mut foreign],
                &[valid()?, valid()?]
            )
            .is_err()
    );
    stage.forward_pipeline_decode_batch_with_sessions(
        &mut [&mut first, &mut second],
        &[valid()?, valid()?],
    )?;
    assert_eq!(stage.total_kv_cache_tokens(), 2);
    assert_eq!(stage.kv_token_budget.allocated(), 128);
    assert!(
        stage
            .forward_pipeline_decode_batch_with_sessions(&mut [&mut first], &[])
            .is_err()
    );
    assert_eq!(stage.session_next_position(&first), 1);
    // A real execution error (missing output head) must clear all selected
    // state and charges, including in-place appends to existing DSA storage.
    let mut broken = fixture(&Device::Cpu)?;
    broken.blocks.reverse();
    broken.is_head = false;
    broken.is_tail = true;
    broken.output = None;
    stage.arch = Box::new(broken);
    stage.position = crate::StagePosition::Tail;
    assert!(
        stage
            .forward_pipeline_decode_batch_with_sessions(
                &mut [&mut first, &mut second],
                &[valid()?, valid()?]
            )
            .is_err()
    );
    assert_eq!(stage.total_kv_cache_tokens(), 0);
    assert_eq!(stage.kv_token_budget.allocated(), 0);
    assert_eq!(stage.session_next_position(&first), 0);
    assert_eq!(stage.session_next_position(&second), 0);
    Ok(())
}

fn empty() -> Vec<Option<LayerCache>> {
    vec![None, None]
}

fn fork(cache: &[Option<LayerCache>]) -> Result<Vec<Option<LayerCache>>> {
    cache
        .iter()
        .map(|slot| {
            slot.as_ref()
                .map(super::super::super::fork_layer_cache)
                .transpose()
        })
        .collect()
}

fn scalar(
    model: &Glm5NextStage,
    cache: &mut [Option<LayerCache>],
    token: u32,
    pos: usize,
) -> Result<Tensor> {
    match model.forward(StageInput::Tokens(vec![token]), cache, 1, pos)? {
        StageOutput::Logits(logits) => Ok(logits),
        _ => anyhow::bail!("missing logits"),
    }
}

fn bits(x: &Tensor) -> Result<Vec<u32>> {
    let values = x.flatten_all()?.to_vec1::<f32>()?;
    anyhow::ensure!(values.iter().all(|v| v.is_finite()), "non-finite output");
    Ok(values.into_iter().map(f32::to_bits).collect())
}

fn tail_prefill_model(device: &Device) -> Result<Glm5NextStage> {
    let mut model = fixture(device)?;
    model.is_head = false;
    model.embedding = None;
    for block in &mut model.blocks {
        if let GlmAttention::Dsa(attention) = &mut block.attention {
            // One head keeps this small dense Metal fixture's prompt input
            // contiguous; its narrow two-head form has unsupported strides.
            // A tiny indexed F32 substitute also changed scores between two
            // identical full-score runs. Keep the control on stable geometry.
            attention.heads = 1;
            attention.q_b = matrix(4, 4, 12, device)?;
            attention.k_b = HeadwiseMatMul::Dense(values(4, 4, 14, device)?.reshape((1, 4, 4))?);
            attention.v_b = HeadwiseMatMul::Dense(values(4, 4, 15, device)?.reshape((1, 4, 4))?);
            attention.out = matrix(8, 4, 16, device)?;
        }
    }
    Ok(model)
}

fn tail_prefill_fixture(device: &Device) -> Result<crate::PipelineStage> {
    let model = tail_prefill_model(device)?;
    let mut stage = crate::session_state_tests::test_stage();
    stage.cache = empty();
    stage.fixed_cache_bytes = model.fixed_cache_bytes(1)?;
    stage.arch = Box::new(model);
    stage.position = crate::StagePosition::Tail;
    stage.hidden_dim = 32;
    stage.vocab_size = 16;
    stage.device = device.clone();
    Ok(stage)
}

fn prompt_cache_bits(state: &crate::PipelineStageSessionState) -> Result<Vec<Vec<u32>>> {
    state
        .cache
        .iter()
        .map(|slot| match slot {
            Some(LayerCache::DeltaNet(cache)) => {
                let mut values = bits(&cache.conv_state)?;
                values.extend(bits(&cache.recurrent_state)?);
                Ok(values)
            }
            Some(LayerCache::Attention(KvCache::PreallocatedF16(cache))) => {
                let mut values = bits(
                    &cache
                        .k
                        .narrow(2, cache.start, cache.len)?
                        .to_dtype(DType::F32)?,
                )?;
                values.extend(bits(
                    &cache
                        .v
                        .narrow(2, cache.start, cache.len)?
                        .to_dtype(DType::F32)?,
                )?);
                Ok(values)
            }
            Some(LayerCache::Attention(KvCache::F16 { k, v })) => {
                let mut values = bits(&k.to_dtype(DType::F32)?)?;
                values.extend(bits(&v.to_dtype(DType::F32)?)?);
                Ok(values)
            }
            _ => anyhow::bail!("missing GLM prompt cache"),
        })
        .collect()
}

fn tail_cache_only_check(device: &Device, cache_only: bool) -> Result<()> {
    let mut actual = tail_prefill_fixture(device)?;
    let mut reference = tail_prefill_fixture(device)?;
    let mut states = [actual.new_session_state(), actual.new_session_state()];
    let mut expected = [reference.new_session_state(), reference.new_session_state()];
    for (step, (lane, len, final_chunk)) in [
        (0, 1, false),
        (1, 2, false),
        (0, 2, false),
        (1, 1, false),
        (0, 1, true),
        (1, 2, true),
    ]
    .into_iter()
    .enumerate()
    {
        let pos = expected[lane].next_position;
        let input = StageInput::activations(values(len, 32, step + 39, device)?);
        let StageOutput::Logits(scores) =
            reference.prefill_with_session(&mut expected[lane], input.clone(), len, pos)?
        else {
            anyhow::bail!("reference tail omitted scores")
        };
        let expected_bits = bits(&scores)?;
        if final_chunk {
            let StageOutput::Logits(scores) =
                actual.prefill_with_session(&mut states[lane], input, len, pos)?
            else {
                anyhow::bail!("final tail chunk omitted scores")
            };
            assert_eq!(bits(&scores)?, expected_bits, "final lane {lane}");
        } else if cache_only {
            actual.prefill_tail_cache_only_with_session(&mut states[lane], input, len, pos)?;
        } else {
            let StageOutput::Logits(scores) =
                actual.prefill_with_session(&mut states[lane], input, len, pos)?
            else {
                anyhow::bail!("control omitted scores")
            };
            assert_eq!(
                bits(&scores)?,
                expected_bits,
                "full-score control lane {lane} step {step}"
            );
        }
        assert_eq!(states[lane].next_position, expected[lane].next_position);
        assert_eq!(states[lane].revision, expected[lane].revision);
        assert_eq!(
            prompt_cache_bits(&states[lane])?,
            prompt_cache_bits(&expected[lane])?
        );
        assert_eq!(
            actual.total_kv_cache_tokens(),
            reference.total_kv_cache_tokens()
        );
        assert_eq!(
            actual.total_kv_cache_allocation_tokens(),
            reference.total_kv_cache_allocation_tokens()
        );
        assert_eq!(
            actual.next_position(),
            0,
            "default cache must stay separate"
        );
    }
    // Cancelling one request must leave its peer's next decode unchanged.
    actual.clear_session_state(&mut states[0]);
    reference.clear_session_state(&mut expected[0]);
    let input = StageInput::activations(values(1, 32, 49, device)?);
    let pos = states[1].next_position;
    let StageOutput::Logits(scores) =
        actual.forward_with_session(&mut states[1], input.clone(), 1, pos)?
    else {
        anyhow::bail!("tail decode omitted scores")
    };
    let StageOutput::Logits(reference_scores) =
        reference.forward_with_session(&mut expected[1], input, 1, pos)?
    else {
        anyhow::bail!("reference decode omitted scores")
    };
    assert_eq!(bits(&scores)?, bits(&reference_scores)?);
    actual.clear_session_state(&mut states[1]);
    reference.clear_session_state(&mut expected[1]);
    assert_eq!(actual.total_kv_cache_allocation_tokens(), 0);
    assert_eq!(reference.total_kv_cache_allocation_tokens(), 0);

    // Missing tail weights prove that cache-only never runs either head step.
    let mut model = tail_prefill_model(device)?;
    model.output_norm = None;
    model.output = None;
    let mut cache = empty();
    for (len, pos) in [(1, 0), (2, 1)] {
        let output = model.forward_prefill_cache_only(
            StageInput::activations(values(len, 32, 52, device)?),
            &mut cache,
            len,
            pos,
        )?;
        assert!(matches!(output, StageOutput::Activations(_)));
        assert!(cache.iter().all(Option::is_some));
    }
    Ok(())
}

#[test]
fn glm_tail_cache_only_prefill_matches_full_scores_cpu() -> Result<()> {
    for cache_only in [false, true] {
        tail_cache_only_check(&Device::Cpu, cache_only)?;
    }
    Ok(())
}

fn tail_cache_only_failure_check(device: &Device) -> Result<()> {
    let mut stage = tail_prefill_fixture(device)?;
    let mut first = stage.new_session_state();
    let mut peer = stage.new_session_state();
    for state in [&mut first, &mut peer] {
        stage.prefill_tail_cache_only_with_session(
            state,
            StageInput::activations(values(1, 32, 63, device)?),
            1,
            0,
        )?;
    }
    let peer_bits = prompt_cache_bits(&peer)?;
    let mut broken = tail_prefill_model(device)?;
    let mut ffn = dense(64, device)?;
    ffn.gate = matrix(16, 7, 65, device)?;
    broken.blocks[1].ffn = GlmFfn::Dense(ffn);
    stage.arch = Box::new(broken);
    assert!(
        stage
            .prefill_tail_cache_only_with_session(
                &mut first,
                StageInput::activations(values(1, 32, 66, device)?),
                1,
                1,
            )
            .is_err()
    );
    assert_eq!(first.next_position, 0);
    assert_eq!(first.revision, 1);
    assert!(first.cache.iter().all(Option::is_none));
    assert_eq!(peer.next_position, 1);
    assert_eq!(prompt_cache_bits(&peer)?, peer_bits);
    assert_eq!(stage.total_kv_cache_tokens(), 1);
    assert_eq!(stage.next_position(), 0);
    stage.clear_session_state(&mut peer);
    assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
    Ok(())
}

#[test]
fn glm_tail_cache_only_prefill_failure_keeps_peer_cpu() -> Result<()> {
    tail_cache_only_failure_check(&Device::Cpu)
}

#[cfg(target_os = "macos")]
#[test]
fn glm_tail_cache_only_prefill_matches_full_scores_metal() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    for cache_only in [false, true] {
        tail_cache_only_check(&device, cache_only)?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn glm_tail_cache_only_prefill_failure_keeps_peer_metal() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    tail_cache_only_failure_check(&device)
}

fn check(device: &Device) -> Result<()> {
    let model = fixture(device)?;
    assert!(model.supports_decode_batch());
    assert!(!model.decode_batch_uses_packed_cache());
    let mut prefix = empty();
    scalar(&model, &mut prefix, 1, 0)?;
    let saved = fork(&prefix)?;
    let mut caches = (0..4).map(|_| fork(&prefix)).collect::<Result<Vec<_>>>()?;
    let mut positions = [1, 3, 4, 7];
    for lane in 0..4 {
        for pos in 1..positions[lane] {
            scalar(&model, &mut caches[lane], (lane + pos) as u32 % 16, pos)?;
        }
    }
    let mut expected = caches.iter().map(|c| fork(c)).collect::<Result<Vec<_>>>()?;
    let mut packed = fork(&prefix)?;
    for count in [4, 3, 2, 1, 4] {
        // Positions cross the sparse-index threshold and complete pool tails.
        let tokens = (0..count)
            .map(|i| (positions[i] % 16) as u32)
            .collect::<Vec<_>>();
        let outputs = model.forward_decode_batch(
            &tokens,
            &mut caches[..count]
                .iter_mut()
                .map(Vec::as_mut_slice)
                .collect::<Vec<_>>(),
            &mut packed,
            &positions[..count],
        )?;
        assert!(packed.iter().all(Option::is_none));
        for lane in 0..count {
            assert_eq!(
                bits(&outputs[lane])?,
                bits(&scalar(
                    &model,
                    &mut expected[lane],
                    tokens[lane],
                    positions[lane]
                )?)?
            );
            positions[lane] += 1;
        }
    }
    for lane in 0..4 {
        assert_eq!(
            bits(&scalar(&model, &mut caches[lane], 2, positions[lane])?)?,
            bits(&scalar(&model, &mut expected[lane], 2, positions[lane])?)?
        );
    }
    assert_eq!(
        bits(&scalar(&model, &mut prefix, 2, 1)?)?,
        bits(&scalar(&model, &mut fork(&saved)?, 2, 1)?)?
    );
    let outputs = model.forward_decode_batch(&[3], &mut [&mut empty()], &mut packed, &[0])?;
    assert_eq!(
        bits(&outputs[0])?,
        bits(&scalar(&model, &mut empty(), 3, 0)?)?
    );
    Ok(())
}

#[test]
fn glm_decode_batch_matches_scalar_cpu() -> Result<()> {
    check(&Device::Cpu)
}

#[cfg(target_os = "macos")]
#[test]
fn glm_decode_batch_matches_scalar_metal() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    check(&device)
}

fn errors(device: &Device) -> Result<()> {
    let mut model = fixture(device)?;
    let mut first = empty();
    scalar(&model, &mut first, 1, 0)?;
    let saved = fork(&first)?;
    let mut second = fork(&first)?;
    let mut packed = fork(&first)?;
    for (tokens, positions) in [
        (vec![], vec![]),
        (vec![0], vec![1]),
        (vec![0, 1], vec![1]),
        (vec![0, 16], vec![1, 1]),
        (vec![0, 1], vec![1, 128]),
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
    assert!(
        model
            .forward_decode_batch(
                &[0, 1],
                &mut [&mut first, &mut empty()],
                &mut packed,
                &[1, 1]
            )
            .is_err()
    );
    assert!(
        model
            .forward_decode_batch(
                &[0, 1],
                &mut [&mut first, &mut second[..1]],
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
                &mut packed[..1],
                &[1, 1]
            )
            .is_err()
    );
    let mut wrong = fork(&saved)?;
    wrong.swap(0, 1);
    assert!(
        model
            .forward_decode_batch(&[0, 1], &mut [&mut first, &mut wrong], &mut packed, &[1, 1])
            .is_err()
    );
    for dtype in [DType::F16, DType::F32] {
        let mut wrong = fork(&saved)?;
        let Some(LayerCache::DeltaNet(state)) = &mut wrong[0] else {
            unreachable!()
        };
        state.recurrent_state = Tensor::zeros(
            if dtype == DType::F32 {
                (2, 4, 3)
            } else {
                (2, 4, 4)
            },
            dtype,
            device,
        )?;
        assert!(
            model
                .forward_decode_batch(&[0, 1], &mut [&mut first, &mut wrong], &mut packed, &[1, 1])
                .is_err()
        );
    }
    // Preflight errors leave all requests and saved state unchanged.
    for cache in [&first, &second, &packed] {
        assert_eq!(
            bits(&scalar(&model, &mut fork(cache)?, 2, 1)?)?,
            bits(&scalar(&model, &mut fork(&saved)?, 2, 1)?)?
        );
    }
    // A late execution error must discard every selected state, not allow a
    // caller to continue with only some layers/lanes advanced.
    let head = model.output.take();
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
    model.output = head;
    for cache in [&mut first, &mut second, &mut packed] {
        assert!(cache.iter().all(Option::is_none));
        *cache = fork(&saved)?;
        scalar(&model, cache, 2, 1)?;
    }
    // The first continuation grew DSA storage. Fail again while appending
    // into existing capacity, not just while allocating a replacement.
    let saved = fork(&first)?;
    let head = model.output.take();
    assert!(
        model
            .forward_decode_batch(
                &[0, 1],
                &mut [&mut first, &mut second],
                &mut packed,
                &[2, 2]
            )
            .is_err()
    );
    model.output = head;
    for cache in [&mut first, &mut second, &mut packed] {
        assert!(cache.iter().all(Option::is_none));
        *cache = fork(&saved)?;
        assert_eq!(
            bits(&scalar(&model, cache, 2, 2)?)?,
            bits(&scalar(&model, &mut fork(&saved)?, 2, 2)?)?
        );
    }
    model.decode_batch_enabled = false;
    assert!(!model.supports_decode_batch());
    assert!(
        model
            .forward_decode_batch(&[0], &mut [&mut empty()], &mut packed, &[0])
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

#[test]
fn glm_decode_batch_errors_follow_session_contract_cpu() -> Result<()> {
    errors(&Device::Cpu)
}

#[cfg(target_os = "macos")]
#[test]
fn glm_decode_batch_errors_follow_session_contract_metal() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    errors(&device)
}
