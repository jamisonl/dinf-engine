use super::*;

fn put(
    weights: &mut RawMap,
    name: &str,
    shape: &[usize],
    values: Vec<f32>,
    device: &Device,
) -> Result<()> {
    let bytes: Vec<u8> = values.into_iter().flat_map(f32::to_le_bytes).collect();
    weights.insert(
        name.into(),
        Arc::new(RawTensor::new(
            shape.to_vec(),
            DeepSeek4DType::F32,
            Tensor::from_vec(bytes.clone(), bytes.len(), device)?,
        )?),
    );
    Ok(())
}

pub(in super::super) fn fixture(ratio: usize, device: &Device) -> Result<V4Attention> {
    fixture_with_metadata(ratio, device, |_| {})
}

fn fixture_with_metadata(
    ratio: usize,
    device: &Device,
    change: impl FnOnce(&mut GgufMetadata),
) -> Result<V4Attention> {
    let mut md = super::super::tests::test_metadata();
    md.hidden_dim = 8;
    md.q_lora_rank = 4;
    md.output_lora_rank = 4;
    md.compress_ratios = vec![ratio];
    md.context_length = 512;
    md.sliding_window = 5;
    md.indexer_head_count = 2;
    md.indexer_head_dim = 64;
    md.indexer_top_k = 2;
    change(&mut md);
    let mut weights = RawMap::new();
    let mut add = |name: &str, shape: &[usize], constant: Option<f32>| -> Result<()> {
        let data = (0..shape.iter().product())
            .map(|i| constant.unwrap_or_else(|| (((i * 17 + name.len()) % 71) as f32 - 35.) / 100.))
            .collect();
        put(
            &mut weights,
            &format!("blk.0.{name}.weight"),
            shape,
            data,
            device,
        )
    };
    for (name, shape) in [
        ("attn_q_a", vec![4, 8]),
        ("attn_q_b", vec![512, 4]),
        ("attn_kv", vec![512, 8]),
        ("attn_output_a", vec![4, 512]),
        ("attn_output_b", vec![8, 4]),
    ] {
        add(name, &shape, None)?;
    }
    add("attn_q_a_norm", &[4], Some(1.))?;
    add("attn_kv_a_norm", &[512], Some(1.))?;
    add("attn_sinks", &[1], Some(0.3))?;
    if ratio > 0 {
        for (prefix, dim) in [("attn_compressor", 512), ("indexer_compressor", 64)] {
            if dim == 64 && ratio != 4 {
                continue;
            }
            let width = dim * if ratio == 4 { 2 } else { 1 };
            add(&format!("{prefix}_kv"), &[width, 8], None)?;
            add(&format!("{prefix}_gate"), &[width, 8], None)?;
            add(&format!("{prefix}_ape"), &[ratio, width], None)?;
            add(&format!("{prefix}_norm"), &[dim], Some(1.))?;
        }
        if ratio == 4 {
            add("indexer.attn_q_b", &[128, 4], None)?;
            add("indexer.proj", &[2, 8], None)?;
        }
    }
    let rope = precompute_rope(64, 10_000., 512, device, None)?;
    V4Attention::new(&weights, "blk.0", &md, 0, &rope, device)
}

fn input(len: usize, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(
        (0..len * 8)
            .map(|i| (((i * 11 + i / 8 * 7) % 59) as f32 - 29.) / 40.)
            .collect(),
        (1, len, 8),
        device,
    )?)
}

fn close(a: &Tensor, b: &Tensor, tolerance: f32) -> Result<()> {
    assert_eq!(a.dims(), b.dims());
    for (a, b) in a
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .zip(b.flatten_all()?.to_vec1::<f32>()?)
    {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() < tolerance,
            "{a} != {b}; tolerance {tolerance}"
        );
    }
    Ok(())
}

fn state(slot: &Option<LayerCache>) -> &DeepSeek4Cache {
    let Some(LayerCache::DeepSeek4(cache)) = slot else {
        panic!("wrong cache")
    };
    cache
}

fn check_chunks_and_forks(device: &Device) -> Result<()> {
    let mut rows = Rows::default();
    for i in 0..65 {
        rows.push(Tensor::new(&[[i as f32, i as f32 + 1.]], device)?)?;
    }
    let forked = rows.clone();
    for i in 65..130 {
        rows.push(Tensor::new(&[[i as f32, i as f32 + 1.]], device)?)?;
    }
    assert_eq!(forked.len, 65);
    let expected_bytes = 65 * 2 * 4;
    #[cfg(target_os = "macos")]
    let expected_bytes = if device.is_metal() {
        // The pool may lend a larger free buffer to either chunk.
        forked
            .chunks
            .iter()
            .map(|tensor| {
                let (storage, _) = tensor.storage_and_layout();
                let candle_core::Storage::Metal(storage) = &*storage else {
                    panic!("expected Metal chunk");
                };
                storage.buffer().length()
            })
            .sum()
    } else {
        expected_bytes
    };
    assert_eq!(forked.bytes(), expected_bytes);
    assert_eq!(rows.chunks.len(), 3);
    assert_eq!(forked.tensor()?.to_vec2::<f32>()?[64], [64., 65.]);
    assert_eq!(rows.tensor()?.to_vec2::<f32>()?[129], [129., 130.]);
    for (ratio, len) in [(0, 17), (4, 17), (128, 129)] {
        let attn = fixture(ratio, device)?;
        let x = input(len + 1, device)?;
        let mut full = None;
        let expected = attn.forward(&x.narrow(1, 0, len)?, &mut full, len, 0)?;
        let mut chunked = None;
        let mut parts = Vec::new();
        let mut pos = 0;
        for chunk in [3, 1, 5, len - 9] {
            parts.push(attn.forward(&x.narrow(1, pos, chunk)?, &mut chunked, chunk, pos)?);
            pos += chunk;
        }
        close(&expected, &Tensor::cat(&parts, 1)?, 2e-3)?;
        assert_eq!(
            state(&full).pool.completed.len,
            len.checked_div(ratio).unwrap_or(0)
        );
        assert_eq!(state(&full).window.as_ref().unwrap().dim(0)?, 5);
        // Chunk boundaries can retain different Metal allocation buckets
        // even though the cache has the same rows and produces the same output.
        if device.is_cpu() {
            assert_eq!(state(&full).bytes(), state(&chunked).bytes());
        }
        // Fork at a partial block, then advance the original. Fork and checkpoint
        // must retain their own last chunk, window and unfinished pool values.
        let saved_window = state(&full).window.as_ref().unwrap().to_dtype(DType::F32)?;
        let saved_bytes = state(&full).bytes();
        let mut fork = Some(crate::architectures::fork_layer_cache(
            full.as_ref().unwrap(),
        )?);
        let mut checkpoint = Some(crate::architectures::checkpoint_layer_cache(
            full.as_ref().unwrap(),
        )?);
        let next = x.narrow(1, len, 1)?;
        let wanted = attn.forward(&next, &mut full, 1, len)?;
        assert_eq!(state(&fork).position, len);
        assert_eq!(state(&fork).bytes(), saved_bytes);
        close(
            &saved_window,
            &state(&fork).window.as_ref().unwrap().to_dtype(DType::F32)?,
            1e-7,
        )?;
        close(&wanted, &attn.forward(&next, &mut fork, 1, len)?, 2e-3)?;
        close(
            &wanted,
            &attn.forward(&next, &mut checkpoint, 1, len)?,
            2e-3,
        )?;
        assert!(attn.forward(&next, &mut fork, 1, len).is_err());
        assert!(attn.forward(&next, &mut None, 1, len).is_err());
    }
    Ok(())
}

fn check_window(device: &Device) -> Result<()> {
    let attn = fixture(0, device)?;
    let x = input(11, device)?;
    let changed = Tensor::cat(
        &[x.narrow(1, 0, 6)?.affine(-3., 1.)?, x.narrow(1, 6, 5)?],
        1,
    )?;
    let a = attn.forward(&x, &mut None, 11, 0)?;
    let b = attn.forward(&changed, &mut None, 11, 0)?;
    close(&a.narrow(1, 10, 1)?, &b.narrow(1, 10, 1)?, 1e-5)?;
    let short = attn.forward(&x.narrow(1, 0, 7)?, &mut None, 7, 0)?;
    close(&a.narrow(1, 0, 7)?, &short, 2e-3)
}

fn check_pool_reference(device: &Device) -> Result<()> {
    for ratio in [4, 128] {
        let attn = fixture(ratio, device)?;
        let c = attn.compressor.as_ref().unwrap();
        let width = c.dim * if ratio == 4 { 2 } else { 1 };
        let mut pool = PoolState::default();
        let ape = c.ape.to_vec2::<f32>()?;
        let value = |p: usize, d: usize| ((p * 5 + d * 7) % 37) as f32 / 19. - 1.;
        let score = |p: usize, d: usize| ((p * 3 + d * 11) % 23) as f32 / 17. - 0.5;
        for p in 0..ratio * 2 + 1 {
            c.push(
                &Tensor::from_vec(
                    (0..width).map(|d| value(p, d)).collect(),
                    (1, width),
                    device,
                )?,
                &Tensor::from_vec(
                    (0..width).map(|d| score(p, d)).collect(),
                    (1, width),
                    device,
                )?,
                p,
                &mut pool,
                &attn.rope,
                DType::F32,
            )?;
            assert_eq!(pool.completed.len, (p + 1) / ratio);
            if (p + 1) % ratio != 0 {
                continue;
            }
            let start = p + 1 - ratio;
            // Keep the reference in F64 so it is independent of the device's F32
            // softmax/reduction order. Metal's F32 path measured a 1.32e-5 absolute
            // difference on the final ratio-128 block; CPU remains checked at 1e-5.
            let mut pooled = vec![0f64; c.dim];
            for (d, result) in pooled.iter_mut().enumerate() {
                let mut sum = 0f64;
                for t in if ratio == 4 {
                    start.saturating_sub(ratio)
                } else {
                    start
                }..=p
                {
                    let column = if ratio == 4 && t >= start {
                        c.dim + d
                    } else {
                        d
                    };
                    let weight =
                        (f64::from(score(t, column)) + f64::from(ape[t % ratio][column])).exp();
                    *result += f64::from(value(t, column)) * weight;
                    sum += weight;
                }
                *result /= sum;
            }
            let scale = (pooled.iter().map(|v| v * v).sum::<f64>() / c.dim as f64 + 1e-6)
                .sqrt()
                .recip();
            pooled.iter_mut().for_each(|v| *v *= scale);
            // Independent interleaved rotation at the first position of the block.
            for d in (c.dim - 64..c.dim).step_by(2) {
                let angle = start as f64 / 10_000f64.powf((d - (c.dim - 64)) as f64 / 64.);
                let (sin, cos) = angle.sin_cos();
                let (a, b) = (pooled[d], pooled[d + 1]);
                pooled[d] = a * cos - b * sin;
                pooled[d + 1] = a * sin + b * cos;
            }
            let block = pool.completed.len - 1;
            let actual = pool
                .completed
                .tensor()?
                .narrow(0, block, 1)?
                .to_dtype(DType::F32)?;
            assert_eq!(
                actual.dims(),
                &[1, c.dim],
                "pool reference shape mismatch ratio={ratio} block={block}"
            );
            let actual = actual.to_vec2::<f32>()?;
            assert_eq!(actual.len(), 1);
            let tolerance = if device.is_metal() { 2e-5 } else { 1e-5 };
            for (index, (&actual, &expected)) in actual[0].iter().zip(&pooled).enumerate() {
                let actual = f64::from(actual);
                assert!(
                    actual.is_finite() && expected.is_finite(),
                    "pool reference non-finite ratio={ratio} block={block} index={index}: actual={actual} expected={expected}"
                );
                let difference = (actual - expected).abs();
                assert!(
                    difference < tolerance,
                    "pool reference mismatch ratio={ratio} block={block} index={index}: actual={actual} expected={expected} difference={difference} tolerance={tolerance}"
                );
            }
        }
        assert_eq!(pool.values.as_ref().unwrap().dim(0)?, 1);
    }
    Ok(())
}

#[test]
fn v4_attention_state_cpu() -> Result<()> {
    check_pool_reference(&Device::Cpu)?;
    check_window(&Device::Cpu)?;
    check_chunks_and_forks(&Device::Cpu)
}

#[test]
fn index_selection_keeps_best_scores_and_stable_ties() -> Result<()> {
    assert_eq!(select_index_keys(&[0., 5., 2., 5., -1.], 2)?, [1, 3]);
    assert_eq!(select_index_keys(&[5., 5., 5., 4.], 2)?, [0, 1]);
    assert_eq!(select_index_keys(&[1., 0.], 4)?, [0, 1]);
    assert!(select_index_keys(&[f32::NAN], 1).is_err());
    assert!(select_index_keys(&[f32::INFINITY], 1).is_err());
    assert!(select_index_keys(&[1.], 0).is_err());
    Ok(())
}

#[test]
fn v4_indexer_rejects_invalid_geometry_before_building_projections() {
    for (heads, dim, message) in [
        (2, usize::MAX, "pooling width overflow"),
        (usize::MAX, 64, "query width overflow"),
        (2, 65, "index query shape"),
        (3, 64, "index query shape"),
    ] {
        let error = fixture_with_metadata(4, &Device::Cpu, |md| {
            md.indexer_head_count = heads;
            md.indexer_head_dim = dim;
        })
        .err()
        .expect("invalid index geometry must fail loading");
        assert!(error.to_string().contains(message), "{error}");
    }
    let error = Compressor::new(
        &RawMap::new(),
        "missing",
        &super::super::tests::test_metadata(),
        4,
        usize::MAX,
        &Device::Cpu,
    )
    .err()
    .expect("pooling width must be checked before looking up tensors");
    assert!(error.to_string().contains("pooling width overflow"));
}

fn check_shared_storage_bytes(device: &Device) -> Result<()> {
    for dtype in [DType::F16, DType::F32] {
        check_shared_storage_bytes_dtype(device, dtype)?;
    }
    Ok(())
}

fn check_shared_storage_bytes_dtype(device: &Device, dtype: DType) -> Result<()> {
    let backing = Tensor::zeros((5, 512), dtype, device)?;
    let cache = DeepSeek4Cache {
        window: Some(backing.narrow(0, 0, 1)?),
        pool: PoolState {
            values: Some(backing.narrow(0, 1, 1)?),
            scores: Some(backing.narrow(0, 1, 1)?),
            ..Default::default()
        },
        ..Default::default()
    };
    let (storage, _) = backing.storage_and_layout();
    let expected = match &*storage {
        #[cfg(target_os = "macos")]
        candle_core::Storage::Metal(storage) => storage.buffer().length(),
        #[cfg(feature = "cuda")]
        candle_core::Storage::Cuda(storage) => match dtype {
            DType::F16 => storage.as_cuda_slice::<half::f16>()?.len() * 2,
            DType::F32 => storage.as_cuda_slice::<f32>()?.len() * 4,
            _ => unreachable!(),
        },
        _ => 5 * 512 * dtype.size_in_bytes(),
    };
    assert_eq!(
        cache.bytes(),
        expected,
        "shared backing must be charged once"
    );
    let layer = LayerCache::DeepSeek4(Box::new(cache));
    for saved in [
        crate::architectures::fork_layer_cache(&layer)?,
        crate::architectures::checkpoint_layer_cache(&layer)?,
    ] {
        assert_eq!(
            crate::architectures::live_layer_cache_bytes(&saved)?,
            expected
        );
    }
    Ok(())
}

#[test]
fn v4_cache_shared_cpu_views_charge_the_complete_backing_once() -> Result<()> {
    check_shared_storage_bytes(&Device::Cpu)?;
    let mut values = Vec::with_capacity(32);
    values.extend([0f32; 4]);
    let expected = values.capacity() * std::mem::size_of::<f32>();
    let backing = Tensor::from_vec(values, (2, 2), &Device::Cpu)?;
    let cache = DeepSeek4Cache {
        window: Some(backing.narrow(0, 0, 1)?),
        ..Default::default()
    };
    assert_eq!(cache.bytes(), expected, "a view keeps unused Vec capacity");
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn v4_cache_shared_cuda_views_charge_the_complete_backing_once() -> Result<()> {
    let Some(device) = crate::test_cuda_device() else {
        return Ok(());
    };
    check_shared_storage_bytes(&device)
}

#[cfg(target_os = "macos")]
#[test]
fn v4_cache_metal_bytes_cover_buckets_and_shared_views() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    check_shared_storage_bytes(&device)?;
    for rows in [3, 65] {
        let mut attention = fixture(128, &device)?;
        attention.window = 128;
        let x = input(rows + 1, &device)?;
        let mut cache = None;
        attention.forward(&x.narrow(1, 0, rows)?, &mut cache, rows, 0)?;
        let current = state(&cache);
        let retained = [
            current.window.as_ref().unwrap(),
            current.pool.values.as_ref().unwrap(),
            current.pool.scores.as_ref().unwrap(),
        ];
        let actual = retained
            .iter()
            .map(|tensor| {
                let (storage, _) = tensor.storage_and_layout();
                let candle_core::Storage::Metal(storage) = &*storage else {
                    panic!("expected Metal cache")
                };
                storage.buffer().length()
            })
            .sum::<usize>();
        let logical = retained
            .iter()
            .map(|tensor| tensor.elem_count() * tensor.dtype().size_in_bytes())
            .sum::<usize>();
        assert!(
            actual > logical,
            "fixture must exercise Metal bucket padding"
        );
        assert_eq!(current.bytes(), actual);
        assert!(current.allocation_tokens() * 2048 >= actual);
        let saved = crate::architectures::checkpoint_layer_cache(cache.as_ref().unwrap())?;
        assert_eq!(
            crate::architectures::live_layer_cache_bytes(&saved)?,
            actual
        );
        attention.forward(&x.narrow(1, rows, 1)?, &mut cache, 1, rows)?;
        assert_eq!(
            crate::architectures::live_layer_cache_bytes(&saved)?,
            actual
        );
        device.synchronize()?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn v4_metal_resident_budget_rejects_the_actual_retained_buffers() -> Result<()> {
    struct AttentionStage(V4Attention);
    impl ModelArchitecture for AttentionStage {
        fn forward(
            &self,
            input: StageInput,
            cache: &mut [Option<LayerCache>],
            seq_len: usize,
            start_pos: usize,
        ) -> Result<StageOutput> {
            let StageInput::Activations { tensor, .. } = input else {
                anyhow::bail!("attention fixture requires activations")
            };
            Ok(StageOutput::Activations(self.0.forward(
                &tensor,
                &mut cache[0],
                seq_len,
                start_pos,
            )?))
        }
        fn family(&self) -> &str {
            "deepseek4"
        }
        fn hidden_dim(&self) -> usize {
            8
        }
        fn layer_count(&self) -> usize {
            1
        }
        fn vocab_size(&self) -> usize {
            4
        }
    }
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    let mut attention = fixture(128, &device)?;
    attention.window = 128;
    let mut stage = crate::session_state_tests::test_stage();
    stage.arch = Box::new(AttentionStage(attention));
    stage.position = crate::StagePosition::Middle;
    stage.hidden_dim = 8;
    stage.vocab_size = 4;
    stage.device = device.clone();
    stage.kv_bytes_per_token = 2048;
    stage.kv_budget_bytes_per_token = 2048;
    stage.preallocated_kv_cache = true;
    // Half of the headroom bounds resident storage: 200 slots. The former
    // logical byte count was 163 slots; these buffers actually occupy 320.
    stage.set_max_total_kv_tokens(400)?;
    let error = stage
        .prefill(StageInput::activations(input(65, &device)?), 65, 0)
        .err()
        .expect("resident Metal cache must exceed the limit");
    assert!(
        error
            .to_string()
            .contains("resident KV allocation limit 200"),
        "{error}"
    );
    assert_eq!(stage.total_kv_cache_tokens(), 0);
    assert_eq!(stage.total_kv_cache_allocation_tokens(), 0);
    assert!(stage.cache.iter().all(Option::is_none));
    device.synchronize()?;
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn v4_attention_state_metal() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    check_pool_reference(&device)?;
    check_window(&device)?;
    check_chunks_and_forks(&device)
}
