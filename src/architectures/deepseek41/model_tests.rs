//! Native graph/state regressions without allocating model banks or 72 GiB.
//! See adjacent NOTICE and LICENSE-MIT.
use super::cache::{CHUNK_ROWS, OWNERS, Rows};
use super::*;

fn row(width: usize, value: f32, d: &Device) -> Result<Tensor> {
    Ok(Tensor::full(value, (1, width), d)?)
}

// Exercise the same immutable append, pair pooling and frontier invariants used
// by the graph, with synthetic projections rather than enormous weight banks.
fn advance(c: &mut DeepSeek41Cache, value: f32) -> Result<()> {
    let pos = c.position;
    c.validate_position(pos, 1, 1_048_576)?;
    for l in 0..40 {
        c.append_raw(l, row(512, value, &Device::Cpu)?)?;
    }
    for owner in 0..4 {
        let cache = &mut c.compressed[owner];
        let kv = row(512, value, &Device::Cpu)?;
        let score = if owner == 3 {
            None
        } else {
            Some(row(512, value / 10.0, &Device::Cpu)?)
        };
        if let Some((latent, rope_position)) = attention::pool_projected(cache, kv, score, pos)? {
            assert_eq!(rope_position, if owner == 3 { pos } else { pos - 1 });
            cache.kv.append(latent)?;
            cache.index.append(row(128, value, &Device::Cpu)?)?;
        }
    }
    c.position += 1;
    Ok(())
}

#[test]
fn deepseek41_owner_and_refresh_schedule_is_model_wide() {
    for l in 0..40 {
        let expected = match l {
            0..=1 => None,
            2..=7 => Some(0),
            8..=13 => Some(1),
            14..=19 => Some(2),
            _ => Some(3),
        };
        assert_eq!(cache::owner(l), expected);
    }
    assert_eq!(
        (0..40).filter(|&l| cache::refresh(l)).collect::<Vec<_>>(),
        [2, 8, 14, 20, 24, 28, 32, 36]
    );
    assert_eq!(OWNERS, [2, 8, 14, 20]);
}

#[test]
fn deepseek41_position_and_pending_frontiers_reject_corruption() -> Result<()> {
    let mut c = DeepSeek41Cache::default();
    assert!(c.validate_position(1, 1, 10).is_err());
    assert!(c.validate_position(0, 0, 10).is_err());
    assert!(c.validate_position(0, usize::MAX, 10).is_err());
    advance(&mut c, 1.)?;
    assert!(c.validate_position(1, 1, 1).is_err());
    c.validate_position(1, 1, 10)?;
    c.compressed[0].pending = None;
    assert!(c.validate_position(1, 1, 10).is_err());
    Ok(())
}

#[test]
fn deepseek41_odd_chunk_fork_and_completed_prefix_are_causal() -> Result<()> {
    let mut a = DeepSeek41Cache::default();
    advance(&mut a, 1.)?;
    assert_eq!(a.compressed[0].kv.len(), 0);
    assert_eq!(a.compressed[3].kv.len(), 1);
    let snapshot = a.clone();
    let mut fork = snapshot.clone();
    advance(&mut a, 2.)?;
    advance(&mut fork, 8.)?;
    assert_eq!(snapshot.position(), 1);
    assert!(snapshot.compressed[0].pending.is_some());
    assert_ne!(
        a.compressed[0].kv.chunks[0].to_vec2::<f32>()?,
        fork.compressed[0].kv.chunks[0].to_vec2::<f32>()?
    );
    let mut replay = snapshot.clone();
    advance(&mut replay, 2.)?;
    assert_eq!(
        a.compressed[0].kv.chunks[0].to_vec2::<f32>()?,
        replay.compressed[0].kv.chunks[0].to_vec2::<f32>()?
    );
    assert_eq!(a.position(), 2);
    assert_eq!(fork.position(), 2);
    a.validate_position(2, 1, 10)?;
    let mut reset = DeepSeek41Cache::default();
    advance(&mut reset, 2.)?;
    assert_eq!(reset.compressed[0].kv.len(), 0);
    Ok(())
}

#[test]
fn deepseek41_chunk_append_does_not_mutate_checkpoint_storage() -> Result<()> {
    let mut r = Rows::default();
    for i in 0..CHUNK_ROWS {
        r.append(row(2, i as f32, &Device::Cpu)?)?;
    }
    let checkpoint = r.clone();
    r.append(row(2, 999., &Device::Cpu)?)?;
    assert_eq!(r.chunks.len(), 2);
    assert_eq!(checkpoint.chunks.len(), 1);
    assert_eq!(checkpoint.chunks[0].to_vec2::<f32>()?[255], [255., 255.]);
    let fork = r.clone();
    r.append(row(2, 777., &Device::Cpu)?)?;
    assert_eq!(fork.chunks[1].to_vec2::<f32>()?, [[999., 999.]]);
    assert_eq!(r.chunks[1].to_vec2::<f32>()?, [[999., 999.], [777., 777.]]);
    Ok(())
}

#[test]
fn deepseek41_raw_ring_is_bounded_and_forks_keep_old_rows() -> Result<()> {
    let mut c = DeepSeek41Cache::default();
    for i in 0..128 {
        c.append_raw(0, row(512, i as f32, &Device::Cpu)?)?;
    }
    let checkpoint = c.clone();
    for i in 128..600 {
        c.append_raw(0, row(512, i as f32, &Device::Cpu)?)?;
    }
    assert_eq!(c.raw[0].as_ref().unwrap().dims(), [128, 512]);
    assert_eq!(c.raw[0].as_ref().unwrap().to_vec2::<f32>()?[0][0], 472.);
    assert_eq!(
        checkpoint.raw[0].as_ref().unwrap().to_vec2::<f32>()?[0][0],
        0.
    );
    assert_eq!(c.bytes(), 128 * 512 * 4);
    Ok(())
}

#[test]
fn deepseek41_budget_uses_same_model_wide_f32_units() -> Result<()> {
    let md = crate::gguf_loader::parse_gguf_metadata(
        &crate::deepseek4_gguf::fixtures::maya_metadata(false),
    )?;
    assert_eq!(md.kv_bytes_per_token(0, 40), DYNAMIC_BYTES_PER_TOKEN as u64);
    assert_eq!(md.kv_bytes_per_token(0, 2), 0);
    assert_eq!(md.kv_bytes_per_token(2, 20), 3840);
    assert_eq!(md.kv_bytes_per_token(20, 40), 2560);
    assert_eq!(md.kv_bytes_per_token_quantized(0, 40, 4), 6400);
    let mut c = DeepSeek41Cache::default();
    advance(&mut c, 1.)?;
    advance(&mut c, 2.)?;
    let dynamic = (3 + 2) * (512 + 128) * 4;
    assert_eq!(dynamic, 2 * DYNAMIC_BYTES_PER_TOKEN);
    assert_eq!(c.allocation_tokens(), 2);
    assert_eq!(
        DeepSeek41Cache::predicted_allocation_tokens(2, &Device::Cpu)?,
        c.allocation_tokens(),
    );
    assert!(c.bytes() > dynamic);
    Ok(())
}

#[test]
fn deepseek41_cache_bytes_deduplicate_views_and_charge_retained_capacity() -> Result<()> {
    let tensor = row(512, 1., &Device::Cpu)?;
    let mut c = DeepSeek41Cache::default();
    c.raw[0] = Some(tensor.clone());
    c.raw[1] = Some(tensor.narrow(1, 0, 128)?);
    assert_eq!(c.bytes(), 512 * 4);
    Ok(())
}

#[test]
fn deepseek41_hc_new_token_always_starts_from_first_stream() -> Result<()> {
    let residual = Tensor::new(
        &[[1f32, 2.], [9., 10.], [15., 16.], [20., 21.]],
        &Device::Cpu,
    )?;
    let old = Tensor::new(&[0f32, 0., 0., 1.], &Device::Cpu)?;
    assert_eq!(
        math::hc_collapse(&residual, &old)?.to_vec1::<f32>()?,
        [20., 21.]
    );
    for _ in 0..2 {
        assert_eq!(
            math::hc_collapse(&residual, &initial_pre(&Device::Cpu)?)?.to_vec1::<f32>()?,
            [1., 2.]
        );
    }
    Ok(())
}

#[test]
fn deepseek41_sink_is_an_extra_logit_with_no_value() -> Result<()> {
    let q = Tensor::zeros((64, 512), DType::F32, &Device::Cpu)?;
    let values = Tensor::ones((1, 512), DType::F32, &Device::Cpu)?;
    let sinks = Tensor::zeros(64, DType::F32, &Device::Cpu)?;
    let heads = attention::heads(&q, &values, &sinks)?.to_vec2::<f32>()?;
    assert_eq!(heads, vec![vec![0.5; 512]; 64]);
    Ok(())
}

fn check_selection(d: &Device) -> Result<()> {
    // A partial last run and exact ties exercise the source bitonic/merge path.
    for n in [1, 31, 513, 1025, 4099, 16384, 16385, 32769] {
        let values: Vec<f32> = (0..n).map(|i| ((i * 37) % 73) as f32 - 20.).collect();
        let cpu = Tensor::new(values.as_slice(), &Device::Cpu)?;
        let expected = ops::topk(&cpu, n.min(512))?.to_vec1::<f32>()?;
        let actual = ops::topk(&cpu.to_device(d)?, n.min(512))?.to_vec1::<f32>()?;
        assert_eq!(actual, expected, "topk n={n}");
        let mut seen = std::collections::HashSet::new();
        for &id in &actual {
            assert!(seen.insert(id as usize));
        }
    }
    Ok(())
}

#[test]
fn deepseek41_selection_cpu_ties_and_partial_runs() -> Result<()> {
    check_selection(&Device::Cpu)
}

#[test]
#[cfg(target_os = "macos")]
fn deepseek41_metal_topk_gather_embedding_and_candidate_policy() -> Result<()> {
    let d = Device::new_metal(0).context("native graph test requires Metal device 0")?;
    check_selection(&d)?;
    let mut rows = Rows::default();
    for i in 0..259 {
        rows.append(row(128, i as f32, &d)?)?;
    }
    let ids = Tensor::new(&[258f32, 0., 256., 255.], &d)?;
    assert_eq!(
        ops::gather(&ids, &rows.chunks, 128, rows.len())?.to_vec2::<f32>()?,
        vec![
            vec![258.; 128],
            vec![0.; 128],
            vec![256.; 128],
            vec![255.; 128]
        ]
    );
    let mut packed = Vec::new();
    for token in 0..3u8 {
        for _ in 0..2 {
            packed.extend_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());
            packed.extend([token + 2; 32]);
        }
    }
    let bytes = Tensor::new(packed.as_slice(), &d)?;
    assert_eq!(
        ops::embedding(&bytes, 2, 64)?.to_vec2::<f32>()?,
        [vec![2.; 64]]
    );
    check_candidates(&d)?;
    d.synchronize()?;
    Ok(())
}

fn check_candidates(d: &Device) -> Result<()> {
    let n = 16385;
    let scores: Vec<f32> = (0..n)
        .map(|i| if i == n - 1 { -100. } else { i as f32 })
        .collect();
    let t = Tensor::new(scores.as_slice(), d)?;
    let mut selection = Selection::default();
    attention::select_scores(&t, 20, &mut selection)?;
    // The newest block is pinned even though its key score is the lowest.
    let low_newest = Tensor::new(
        (0..n)
            .map(|i| if i == n - 1 { 1e6 } else { -(i as f32) })
            .collect::<Vec<_>>(),
        d,
    )?;
    let ids = attention::select_scores(&low_newest, 24, &mut selection)?.to_vec1::<f32>()?;
    assert_eq!(ids[0], (n - 1) as f32);
    // At this width only one block is excluded: block0, not newest block2048.
    assert!(!ids.iter().any(|&i| i < 8.));
    assert!(attention::select_scores(&t, 24, &mut Selection::default()).is_err());
    Ok(())
}

#[test]
fn deepseek41_candidate_source_pins_newest_and_filters_later_refresh() -> Result<()> {
    check_candidates(&Device::Cpu)
}

#[test]
fn deepseek41_full_cache_checkpoint_and_attention_only_rejection_preserve_slot() -> Result<()> {
    let mut c = DeepSeek41Cache::default();
    advance(&mut c, 1.)?;
    let mut slot = Some(LayerCache::DeepSeek41(Box::new(c)));
    let checkpoint = super::super::checkpoint_layer_cache(slot.as_ref().unwrap())?;
    assert!(super::super::commit_speculative_layer_cache(slot.as_mut().unwrap(), 0, 1).is_err());
    let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        super::super::with_kv_cache(&mut slot, |_| ());
    }));
    assert!(rejected.is_err());
    let Some(LayerCache::DeepSeek41(current)) = slot else {
        panic!("lost slot");
    };
    let LayerCache::DeepSeek41(checkpoint) = checkpoint else {
        panic!("wrong checkpoint");
    };
    assert_eq!(current.position(), checkpoint.position());
    assert_eq!(current.bytes(), checkpoint.bytes());
    Ok(())
}

// Deliberately retain a strided, nonzero-offset view of a pooled temporary.
// Padding differs from the row so compaction must honor both strides and offset.
#[cfg(target_os = "macos")]
fn oversized_metal_row(width: usize, value: f32, device: &Device) -> Result<Tensor> {
    let mut values = vec![-999f32; width * 16];
    for i in 0..width {
        values[i * 16 + 3] = value;
    }
    let view = Tensor::from_vec(values, (width, 16), device)?
        .t()?
        .narrow(0, 3, 1)?;
    assert!(metal_backing_bytes(&view) > (width * 4).next_power_of_two());
    Ok(view)
}

#[cfg(target_os = "macos")]
fn metal_backing_bytes(tensor: &Tensor) -> usize {
    let (storage, _) = tensor.storage_and_layout();
    let candle_core::Storage::Metal(storage) = &*storage else {
        panic!("native fixture must retain Metal storage");
    };
    storage.buffer().length()
}

#[cfg(target_os = "macos")]
fn metal_backing_key(tensor: &Tensor) -> usize {
    let (storage, _) = tensor.storage_and_layout();
    let candle_core::Storage::Metal(storage) = &*storage else {
        panic!("native fixture must retain Metal storage");
    };
    std::ptr::from_ref(storage.buffer().as_ref()) as *const () as usize
}

#[cfg(target_os = "macos")]
#[test]
fn deepseek41_metal_bounded_retention_keeps_fitting_backing_and_sealed_chunks() -> Result<()> {
    let device = Device::new_metal(0)?;
    let exact = super::super::copy_persistent_tensor(&row(512, 7., &device)?)?;
    let key = metal_backing_key(&exact);
    let mut rows = Rows::default();
    rows.append(exact.clone())?;
    assert_eq!(metal_backing_key(&rows.chunks[0]), key);
    let mut cache = DeepSeek41Cache::default();
    cache.append_raw(0, exact.clone())?;
    assert_eq!(metal_backing_key(cache.raw[0].as_ref().unwrap()), key);
    attention::pool_projected(&mut cache.compressed[0], exact.clone(), Some(exact), 0)?;
    let (kv, score) = cache.compressed[0].pending.as_ref().unwrap();
    assert_eq!(metal_backing_key(kv), key);
    assert_eq!(metal_backing_key(score), key);
    assert_eq!(cache.bytes(), metal_backing_bytes(kv));
    // No copy for a nonzero-offset view whose backing fits the rounded bound.
    let backing = super::super::copy_persistent_tensor(&Tensor::full(3f32, (4, 128), &device)?)?;
    let view = backing.narrow(0, 1, 3)?;
    let retained = cache::retain_bounded(view)?;
    assert_eq!(metal_backing_key(&retained), metal_backing_key(&backing));
    assert_eq!(retained.to_vec2::<f32>()?, vec![vec![3.; 128]; 3]);
    // All later appends use oversized pooled views, including the replacement
    // tail and the first row of the next immutable chunk.
    for i in 1..255 {
        rows.append(oversized_metal_row(512, i as f32, &device)?)?;
    }
    let partial = rows.clone();
    rows.append(oversized_metal_row(512, 255., &device)?)?;
    let sealed = rows.clone();
    let sealed_key = metal_backing_key(&sealed.chunks[0]);
    let mut fork = sealed.clone();
    rows.append(oversized_metal_row(512, 256., &device)?)?;
    fork.append(oversized_metal_row(512, 999., &device)?)?;
    device.synchronize()?;
    assert_eq!(partial.len(), 255);
    assert_eq!(
        partial.chunks[0].to_vec2::<f32>()?.last().unwrap(),
        &vec![254.; 512],
    );
    assert_eq!(sealed.len(), 256);
    assert_eq!(
        sealed.chunks[0].to_vec2::<f32>()?.last().unwrap(),
        &vec![255.; 512],
    );
    assert_eq!(metal_backing_key(&rows.chunks[0]), sealed_key);
    assert_eq!(metal_backing_key(&fork.chunks[0]), sealed_key);
    assert_eq!(rows.chunks[1].to_vec2::<f32>()?, vec![vec![256.; 512]]);
    assert_eq!(fork.chunks[1].to_vec2::<f32>()?, vec![vec![999.; 512]]);
    Ok(())
}

#[cfg(target_os = "macos")]
impl DeepSeek41Cache {
    /// Bounded native backing at a long frontier without replaying every layer.
    /// Distinct exact private seed pages leave rounding to the native appends.
    pub(crate) fn seeded_native_fixture(position: usize, device: &Device) -> Result<Self> {
        anyhow::ensure!(
            (1..=4096).contains(&position),
            "bounded native fixture frontier"
        );
        let tensor = |rows, width, value| -> Result<Tensor> {
            super::super::copy_persistent_tensor(&Tensor::full(value, (rows, width), device)?)
        };
        let pages = |len: usize, width, page_rows| -> Result<Rows> {
            let chunks = (0..len)
                .step_by(page_rows)
                .map(|start| tensor((len - start).min(page_rows), width, 1f32))
                .collect::<Result<Vec<_>>>()?;
            Ok(Rows::fixture_pages(page_rows, chunks, len))
        };
        let mut cache = Self::default();
        for raw in &mut cache.raw {
            *raw = Some(tensor(position.min(128), 512, 1f32)?);
        }
        for (owner, compressed) in cache.compressed.iter_mut().enumerate() {
            let len = position / if owner == 3 { 1 } else { 2 };
            compressed.kv = pages(len, 512, CHUNK_ROWS)?;
            compressed.index = pages(len, 128, cache::INDEX_PAGE_ROWS)?;
            if owner != 3 && position % 2 == 1 {
                compressed.pending = Some((tensor(1, 512, 1f32)?, tensor(1, 512, 1f32 / 16.)?));
            }
        }
        cache.position = position;
        cache.validate_frontier()?;
        Ok(cache)
    }

    // Shared with the real PipelineStage admission regression. Only projections
    // are synthetic: raw retention, pending pooling and chunk append are native.
    pub(crate) fn advance_native_fixture(
        &mut self,
        value: f32,
        device: &Device,
        oversized: bool,
    ) -> Result<()> {
        let projection = |width, value| {
            if oversized {
                oversized_metal_row(width, value, device)
            } else {
                super::super::copy_persistent_tensor(&row(width, value, device)?)
            }
        };
        let pos = self.position;
        self.validate_position(pos, 1, 1_048_576)?;
        for layer in 0..40 {
            self.append_raw(layer, projection(512, value)?)?;
        }
        for (owner, compressed) in self.compressed.iter_mut().enumerate() {
            let kv = projection(512, value)?;
            // The no-pooling BF16 kernel requires contiguous input; the
            // retained index/raw/pending paths still receive strided views.
            let kv = if owner == 3 { kv.contiguous()? } else { kv };
            let score = if owner == 3 {
                None
            } else {
                Some(projection(512, value / 16.0)?)
            };
            if let Some((latent, _)) = attention::pool_projected(compressed, kv, score, pos)? {
                compressed.kv.append(latent)?;
                compressed.index.append(projection(128, value)?)?;
            }
        }
        self.position += 1;
        Ok(())
    }
}

#[cfg(target_os = "macos")]
#[test]
fn deepseek41_metal_cache_admission_charges_retained_dynamic_buffers() -> Result<()> {
    let device = Device::new_metal(0).context("cache admission test requires Metal device 0")?;
    let mut cache = DeepSeek41Cache::default();
    for position in 1..=513 {
        let value = (position % 8) as f32;
        cache.advance_native_fixture(value, &device, true)?;
        if ![1, 2, 3, 127, 128, 129, 255, 256, 257, 513].contains(&position) {
            continue;
        }
        device.synchronize()?;
        cache.validate_position(position, 1, 1_048_576)?;
        let mut dynamic = 0;
        for compressed in &cache.compressed {
            for tensor in compressed.kv.chunks.iter().chain(&compressed.index.chunks) {
                let bytes = metal_backing_bytes(tensor);
                assert!(
                    bytes <= (tensor.elem_count() * 4).next_power_of_two(),
                    "oversized dynamic backing at position {position}: {bytes}",
                );
                dynamic += bytes;
            }
        }
        let predicted = DeepSeek41Cache::predicted_allocation_tokens(position, &device)?;
        assert_eq!(
            cache.allocation_tokens(),
            position.max(dynamic.div_ceil(DYNAMIC_BYTES_PER_TOKEN)),
            "actual accounting at position {position}",
        );
        assert!(
            cache.allocation_tokens() <= predicted,
            "position {position}: retained {} units ({dynamic} bytes), predicted {predicted}",
            cache.allocation_tokens(),
        );
        let mut fixed = 0;
        for tensor in cache.raw.iter().flatten().chain(
            cache
                .compressed
                .iter()
                .flat_map(|c| c.pending.iter().flat_map(|(a, b)| [a, b])),
        ) {
            let bytes = metal_backing_bytes(tensor);
            assert!(
                bytes <= (tensor.elem_count() * 4).next_power_of_two(),
                "oversized fixed backing at position {position}: {bytes}",
            );
            fixed += bytes;
        }
        assert!(fixed <= FIXED_CACHE_BYTES);
        assert_eq!(cache.bytes(), dynamic + fixed);
        for raw in cache.raw.iter().flatten() {
            let values = raw.to_vec2::<f32>()?;
            assert_eq!(values.last().unwrap(), &vec![value; 512]);
            let first = ((position.saturating_sub(128) + 1) % 8) as f32;
            assert_eq!(values[0], vec![first; 512]);
        }
        for compressed in &cache.compressed {
            if let Some(index) = compressed.index.chunks.last() {
                let last = if compressed.pending.is_some() {
                    position - 1
                } else {
                    position
                };
                assert_eq!(
                    index.to_vec2::<f32>()?.last().unwrap(),
                    &vec![(last % 8) as f32; 128],
                );
            }
            if let Some((kv, score)) = &compressed.pending {
                assert_eq!(kv.to_vec2::<f32>()?, vec![vec![value; 512]]);
                assert_eq!(score.to_vec2::<f32>()?, vec![vec![value / 16.0; 512]]);
            }
        }
        eprintln!(
            "native retention position={position} dynamic_bytes={dynamic} actual_units={} predicted_units={predicted} fixed_bytes={fixed}",
            cache.allocation_tokens(),
        );
    }
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
#[allow(
    clippy::excessive_precision,
    reason = "Preserve exact F32 inputs that detect an unwanted BF16 rounding step."
)]
fn deepseek41_metal_embedding_preserves_f32_initial_residual() -> Result<()> {
    let device =
        Device::new_metal(0).context("embedding precision test requires Metal device 0")?;
    // F16 scale * int8 is exactly F32, but not BF16. Repeating the four
    // initial streams must not introduce the later projection BF16 boundary.
    let mut packed = Vec::with_capacity(2 * 34);
    for _ in 0..2 {
        packed.extend_from_slice(&0x355bu16.to_le_bytes());
        packed.extend_from_slice(&[3u8; 32]);
    }
    let bytes = Tensor::new(packed.as_slice(), &device)?;
    let residual = initial_residual(&bytes, 0, 64)?;
    assert_eq!(
        residual.to_vec2::<f32>()?,
        vec![vec![1.004150390625; 64]; 4]
    );
    assert_eq!(
        math::bf16(&residual)?.to_vec2::<f32>()?,
        vec![vec![1.0078125; 64]; 4],
    );
    device.synchronize()?;
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
#[allow(
    clippy::excessive_precision,
    reason = "Preserve exact F32 inputs that detect an unwanted BF16 rounding step."
)]
fn deepseek41_metal_resident_q8_projects_f32_across_prefill_rows() -> Result<()> {
    let device =
        Device::new_metal(0).context("resident projection test requires Metal device 0")?;
    let mut packed = Vec::new();
    packed.extend_from_slice(&0x355bu16.to_le_bytes());
    packed.extend_from_slice(&[3u8; 32]);
    let raw = RawTensor::new(
        vec![1, 32],
        crate::deepseek4_gguf::DeepSeek4DType::Q8_0,
        Tensor::new(packed.as_slice(), &device)?,
    )?;
    let weights = Weights::from([("output.weight".into(), Arc::new(raw))]);
    let projection = weights::linear(&weights, "output.weight", &device)?;
    let mut input = vec![0f32; 3 * 32];
    for row in 0..3 {
        input[row * 32] = 1.;
    }
    let input = Tensor::from_vec(input, (3, 32), &device)?;
    assert_eq!(
        projection.forward(&input)?.to_vec2::<f32>()?,
        vec![vec![1.004150390625]; 3],
    );
    device.synchronize()?;
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
fn deepseek41_metal_public_stage_cannot_bypass_verified_source_gate() -> Result<()> {
    let device = Device::new_metal(0).context("source integrity test requires Metal device 0")?;
    let metadata = crate::deepseek4_gguf::fixtures::maya_metadata(false);
    let (file, _) = crate::deepseek4_gguf::fixtures::sparse(&metadata, &[]);
    let source = crate::deepseek4_gguf::RawGgufSource::new(std::fs::File::open(file.path())?)?;
    let stage = StageTensors {
        range: LayerRange::solo(40),
        metadata: crate::gguf_loader::parse_gguf_metadata(&metadata)?,
        tensors: Default::default(),
        raw_tensors: Default::default(),
        raw_source: Some(Arc::new(source)),
        device,
    };
    DeepSeek41Stage::validate_stage_config(&stage.metadata, &stage.range, false, &stage.device)?;
    let error = DeepSeek41Stage::from_stage_tensors(stage, None)
        .err()
        .context("unverified public StageTensors unexpectedly constructed a graph")?;
    // A missing-weight error would mean the public entry point bypassed the
    // integrity guard and reached graph construction on an unverified source.
    assert!(
        error.to_string().contains("SHA256 proof"),
        "expected source-integrity rejection before weight construction: {error:#}",
    );
    Ok(())
}

// The original four-dispatch graph is the numerical oracle for future fusion:
// routed activations stay F32; shared inputs and mid retain BF16 boundaries.
#[test]
#[cfg(target_os = "macos")]
#[allow(
    clippy::excessive_precision,
    clippy::approx_constant,
    reason = "These fixed activation inputs probe F32 and BF16 boundaries; the near-pi value is test data, not a formula constant."
)]
fn deepseek41_metal_swiglu_clamps_and_precision_match_four_op_reference() -> Result<()> {
    let device = Device::new_metal(0).context("SwiGLU precision test requires Metal device 0")?;
    let gates = [
        -f32::MAX,
        -1000.,
        -100.,
        -90.,
        -88.,
        -20.,
        -10.000001,
        -10.,
        -9.999999,
        -1.004150390625,
        -1.,
        -f32::MIN_POSITIVE,
        -f32::from_bits(1),
        -0.,
        0.,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        0.0001,
        0.1,
        1.,
        1.004150390625,
        3.1415927,
        9.999999,
        10.,
        10.000001,
        20.,
        100.,
        1000.,
        f32::MAX,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NAN,
        f32::from_bits(0xffc00001),
    ];
    let ups = [
        -f32::MAX,
        -10.000001,
        -10.,
        -9.999999,
        -1.004150390625,
        -1.,
        -f32::MIN_POSITIVE,
        -f32::from_bits(1),
        -0.,
        0.,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        0.1,
        1.,
        1.004150390625,
        9.999999,
        10.,
        10.000001,
        f32::MAX,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NAN,
    ];
    let mut gate_cells = Vec::new();
    let mut up_cells = Vec::new();
    for gate in gates {
        for up in ups {
            gate_cells.push(gate);
            up_cells.push(up);
        }
    }
    let gate = Tensor::new(gate_cells.as_slice(), &device)?;
    let up = Tensor::new(up_cells.as_slice(), &device)?;
    for shared in [false, true] {
        let (gate, up) = if shared {
            (math::bf16(&gate)?, math::bf16(&up)?)
        } else {
            (gate.clone(), up.clone())
        };
        let clamped_gate = gate.clamp(f64::NEG_INFINITY, 10.0)?;
        let clamped_up = up.clamp(-10.0, 10.0)?;
        let expected = candle_nn::ops::silu(&clamped_gate)?.mul(&clamped_up)?;
        let actual = swiglu(&gate, &up)?;
        let (expected, actual) = if shared {
            (math::bf16(&expected)?, math::bf16(&actual)?)
        } else {
            (expected, actual)
        };
        let expected = expected.to_vec1::<f32>()?;
        let actual = actual.to_vec1::<f32>()?;
        for (i, (&old, &new)) in expected.iter().zip(&actual).enumerate() {
            assert_eq!(
                new.to_bits(),
                old.to_bits(),
                "shared={shared} gate={:?} up={:?}",
                gate_cells[i],
                up_cells[i]
            );
        }
        if !shared {
            assert!(
                actual
                    .iter()
                    .any(|&x| x.is_finite()
                        && x.to_bits() != half::bf16::from_f32(x).to_f32().to_bits()),
                "fixture must detect accidental BF16 rounding of routed intermediates"
            );
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn deepseek41_metal_grouped_routed_owner_cache_and_error_fence() -> Result<()> {
    grouped_routed_owner_cache_and_error_fence(false)
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "test-only routed-post path; run serially (lease/error/bucket checks)"]
fn routed_post_metal_owner_cache_and_error_fence() -> Result<()> {
    grouped_routed_owner_cache_and_error_fence(true)
}

#[cfg(target_os = "macos")]
fn grouped_routed_owner_cache_and_error_fence(post: bool) -> Result<()> {
    use crate::deepseek4_gguf::{DeepSeek4DType, RawGgufSource, fixtures};
    use std::io::{Seek, SeekFrom, Write};
    let device = Device::new_metal(0)?;
    assert!(
        profile::grouped_iq1_simd_enabled(),
        "native grouped proof requires SIMD"
    );
    let matrix_bytes = 2_304_000u64;
    let bank_bytes = 384 * matrix_bytes;
    let specs = [
        ("blk.0.ffn_gate_exps.weight", vec![384, 2304, 5120], 19, 0),
        (
            "blk.0.ffn_up_exps.weight",
            vec![384, 2304, 5120],
            19,
            bank_bytes,
        ),
        (
            "blk.0.ffn_down_exps.weight",
            vec![384, 5120, 2304],
            19,
            2 * bank_bytes,
        ),
    ];
    let specs = specs
        .iter()
        .map(|(n, s, t, o)| (*n, s.as_slice(), *t, *o))
        .collect::<Vec<_>>();
    let (mut file, data) = fixtures::sparse(&Default::default(), &specs);
    for part in 0..3 {
        for expert in 0..6 {
            let mut packed = vec![0u8; matrix_bytes as usize];
            for (i, block) in packed.chunks_exact_mut(50).enumerate() {
                block[..2].copy_from_slice(
                    &half::f16::from_f32((1 + (i + expert * 3 + part * 7) % 19) as f32 / 4096.)
                        .to_le_bytes(),
                );
                for (j, byte) in block[2..].iter_mut().enumerate() {
                    *byte = ((i * 13 + expert * 31 + part * 53 + j * 7) % 256) as u8;
                }
            }
            file.seek(SeekFrom::Start(
                data + part as u64 * bank_bytes + expert as u64 * matrix_bytes,
            ))?;
            file.write_all(&packed)?;
        }
    }
    file.flush()?;
    let source = Arc::new(RawGgufSource::new(std::fs::File::open(file.path())?)?);
    let mut cache = ExpertCache::new(source, device.clone())?;
    let route = math::RouterSelection {
        indices: [5, 1, 3, 0, 4, 2],
        weights: [0.5, 0.375, 0.25, 0.1875, 0.125, 0.0625],
    };
    let mut reference = None;
    let mut charge = 0;
    for (arm, grouped) in [false, true, true].into_iter().enumerate() {
        // Post comparison never uses the older ungrouped gate/up baseline.
        let grouped = post || grouped;
        let _post = RoutedPostBatchGuard::set(post && arm > 0);
        let input = Tensor::from_vec(
            (0..5120)
                .map(|i| (i % 37) as f32 / 8192. - 0.002)
                .collect::<Vec<_>>(),
            (1, 5120),
            &device,
        )?;
        let selected = cache.prepare(0, &route.indices)?;
        let pin = Arc::downgrade(selected.get(5).unwrap());
        let mut owners = MoeLeases {
            selected: Some(selected),
            input: Some(input),
            gate_up: None,
            ..Default::default()
        };
        let output = routed_experts(&mut owners, &route, grouped, post)?;
        if grouped {
            let backing = owners.gate_up.as_ref().unwrap();
            assert_eq!(backing.dims(), [12, 2304]);
            assert_eq!(
                metal_backing_key(backing),
                metal_backing_key(&backing.narrow(0, 11, 1)?)
            );
            assert!(
                metal_backing_bytes(backing) + metal_backing_bytes(owners.input.as_ref().unwrap())
                    <= 160 * 1024
            );
        }
        if post && arm > 0 {
            assert_eq!(owners.routed_post.tensors.last().unwrap().dims(), [6, 5120]);
            assert!(
                cache::retained_bytes(owners.routed_post.tensors())
                    <= profile::TILE_ROUTED_POST_BYTES
            );
            // This FFN-only check is NOT a combined tile-owner admission proof.
            for tensor in owners.tensors() {
                assert!(
                    metal_backing_bytes(tensor)
                        <= super::super::fixed_state_tensor_bytes(tensor.dims(), &device)?
                );
            }
            // Queue dependent copies of EVERY exposed activation/down owner,
            // then reject a malformed down input while those operands are live.
            let pending = owners
                .routed_post
                .tensors
                .iter()
                .map(super::super::copy_persistent_tensor)
                .collect::<Result<Vec<_>>>()?;
            let matrices = route
                .indices
                .map(|id| owners.selected.as_ref().unwrap().get(id).unwrap().down());
            let bad = owners.routed_post.tensors[3].narrow(1, 0, 2303)?;
            assert!(Linear::forward_grouped_iq1_down(&bad, &matrices).is_err());
            let pressure = Tensor::full(999f32, (6, 5120), &device)?;
            assert!(pin.strong_count() >= 2);
            device.synchronize()?; // Existing error-fence boundary in qualifier.
            for (source, copy) in owners.routed_post.tensors.iter().zip(&pending) {
                let bits = |t: &Tensor| -> Result<Vec<u32>> {
                    t.flatten_all()?
                        .to_vec1::<f32>()?
                        .into_iter()
                        .map(|v| {
                            ensure!(v.is_finite());
                            Ok(v.to_bits())
                        })
                        .collect()
                };
                assert_eq!(bits(source)?, bits(copy)?);
            }
            drop(pressure);
        }
        // Pressure the same pooled size while owners protect queued operands.
        let pressure = Tensor::full(999f32, (12, 2304), &device)?;
        // The production next-router readback is the retirement boundary.
        let router = Tensor::zeros(384, DType::F32, &device)?;
        assert_eq!(router.to_vec1::<f32>()?, vec![0.; 384]);
        let bits = output
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|v| {
                assert!(v.is_finite());
                v.to_bits()
            })
            .collect::<Vec<_>>();
        if let Some(expected) = &reference {
            assert_eq!(&bits, expected);
        } else {
            assert!(bits.iter().any(|&v| f32::from_bits(v) != 0.));
            reference = Some(bits);
        }
        drop(pressure);
        let stats = cache.stats();
        if charge == 0 {
            charge = stats.charged_bytes;
        }
        assert_eq!(stats.charged_bytes, charge);
        assert_eq!((stats.entries, stats.misses, stats.evictions), (6, 6, 0));
        drop(owners);
        assert_eq!(pin.upgrade().unwrap().gate().dtype(), DeepSeek4DType::IQ1S);
    }
    assert_eq!(cache.stats().hits, 12);
    if post {
        let _post = RoutedPostBatchGuard::set(true);
        let input = Tensor::from_vec(
            (0..5120)
                .map(|i| (i % 37) as f32 / 8192. - 0.002)
                .collect::<Vec<_>>(),
            (1, 5120),
            &device,
        )?;
        let mut decode = MoeLeases {
            selected: Some(cache.prepare(0, &route.indices)?),
            input: Some(input),
            ..Default::default()
        };
        let out = routed_experts(&mut decode, &route, true, false)?;
        assert!(
            decode.routed_post.tensors.is_empty(),
            "decode must not select post batching"
        );
        let bits = out
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|v| {
                assert!(v.is_finite());
                v.to_bits()
            })
            .collect::<Vec<_>>();
        assert_eq!(Some(&bits), reference.as_ref());
        device.synchronize()?;
    }
    let mut owners = MoeLeases {
        selected: Some(cache.prepare(0, &route.indices)?),
        input: Some(Tensor::zeros((1, 5120), DType::F16, &device)?),
        gate_up: None,
        ..Default::default()
    };
    let lease = Arc::downgrade(owners.selected.as_ref().unwrap().get(5).unwrap());
    let _post = RoutedPostBatchGuard::set(post);
    assert!(routed_experts(&mut owners, &route, true, post).is_err());
    assert!(owners.input.is_some() && owners.selected.is_some());
    assert!(lease.strong_count() >= 2);
    device.synchronize()?; // Same mandatory final/error fence, before owners drop.
    drop(owners);
    assert_eq!(lease.strong_count(), 1);
    assert_eq!(cache.stats().charged_bytes, charge);
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "test-only routed-post path; run serially (six-row activation exactness)"]
#[allow(
    clippy::excessive_precision,
    reason = "Preserve exact F32 inputs that detect an unwanted BF16 rounding step."
)]
fn routed_post_metal_interleaved_activation_finite_bits() -> Result<()> {
    let d = Device::new_metal(0)?;
    let edge = [
        -1000.,
        -90.,
        -20.,
        -10.000001,
        -10.,
        -1.004150390625,
        -f32::MIN_POSITIVE,
        -f32::from_bits(1),
        -0.,
        0.,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        0.1,
        1.004150390625,
        9.999999,
        10.,
        10.000001,
        1000.,
    ];
    let mut data = vec![999f32; 3];
    for slot in 0..6 {
        for part in 0..2 {
            for i in 0..2304 {
                data.push(edge[(i * 7 + slot * 11 + part * 5) % edge.len()]);
            }
        }
    }
    let len = data.len();
    let backing = Tensor::from_vec(data, len, &d)?
        .narrow(0, 3, 12 * 2304)?
        .reshape((12, 2304))?;
    let expected = (0..6)
        .map(|slot| {
            swiglu(
                &backing.narrow(0, 2 * slot, 1)?,
                &backing.narrow(0, 2 * slot + 1, 1)?,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mut owners = RoutedPostLeases::default();
    let actual = routed_post_swiglu(&backing, &mut owners)?;
    let pressure = Tensor::full(999f32, (6, 2304), &d)?;
    d.synchronize()?;
    let mut non_bf16 = false;
    for (slot, expected_row) in expected[..6].iter().enumerate() {
        let old = expected_row.flatten_all()?.to_vec1::<f32>()?;
        let new = actual.narrow(0, slot, 1)?.flatten_all()?.to_vec1::<f32>()?;
        for (i, (&a, &b)) in old.iter().zip(&new).enumerate() {
            ensure!(a.is_finite() && b.is_finite());
            assert_eq!(a.to_bits(), b.to_bits(), "route={slot} column={i}");
            non_bf16 |= b.to_bits() != math::bf16_scalar(b).to_bits();
        }
    }
    assert!(
        non_bf16,
        "fixture must reject an inserted routed BF16 round"
    );
    assert_eq!(actual.dims(), [6, 2304]);
    assert!(actual.is_contiguous());
    drop(pressure);
    Ok(())
}

#[test]
#[ignore = "test-only routed-post selector checks"]
fn routed_post_guard_defaults_off_and_restores_nested_error() -> Result<()> {
    assert!(!ROUTED_POST_BATCH.with(|v| v.get()));
    {
        let _on = RoutedPostBatchGuard::set(true);
        let error: Result<()> = (|| {
            let _off = RoutedPostBatchGuard::set(false);
            assert!(!ROUTED_POST_BATCH.with(|v| v.get()));
            anyhow::bail!("qualifier error")
        })();
        assert!(error.is_err());
        assert!(ROUTED_POST_BATCH.with(|v| v.get()));
    }
    assert!(!ROUTED_POST_BATCH.with(|v| v.get()));
    Ok(())
}

// These filters are CPU-only: no Device::new_metal and no model/source loads.
#[test]
fn native_tile_cpu_default_request_and_scalar_environment_policy() -> Result<()> {
    use std::ffi::OsStr;
    assert!(!profile::scalar_prefill_opt_out(None));
    for value in ["", "1", "7", "false", "00", " 0", "0 "] {
        assert!(
            !profile::scalar_prefill_opt_out(Some(OsStr::new(value))),
            "{value:?}"
        );
    }
    assert!(profile::scalar_prefill_opt_out(Some(OsStr::new("0"))));
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        assert!(!profile::scalar_prefill_opt_out(Some(OsStr::from_bytes(
            b"\xff"
        ))));
        assert!(!profile::scalar_prefill_opt_out(Some(OsStr::from_bytes(
            b"0\xff"
        ))));
    }
    let scalar = profile::prefill_tile_plan(1, 5291, 7)?;
    assert_eq!(
        (
            scalar.requested,
            scalar.effective,
            scalar.scratch_bound_bytes
        ),
        (1, 1, 0)
    );
    let before = (
        profile::requested_prefill_tile(),
        profile::prefill_planner_limit(),
        profile::prefill_validation_enabled(),
    );
    assert!(profile::set_prefill_tile(profile::MAX_PREFILL_TILE + 1).is_err());
    assert!(profile::set_prefill_tile(0).is_err());
    assert_eq!(
        (
            profile::requested_prefill_tile(),
            profile::prefill_planner_limit(),
            profile::prefill_validation_enabled()
        ),
        before
    );
    Ok(())
}

#[test]
fn native_tile_cpu_prefix_clips_partial_chunks_before_scores_topk_and_gather() -> Result<()> {
    let d = &Device::Cpu;
    let mut keys = Rows::default();
    let mut values = Rows::default();
    for i in 0..520 {
        // Future rows dominate if even the final partial chunk leaks.
        keys.append(row(2, if i >= 257 { 1e6 } else { i as f32 }, d)?)?;
        values.append(row(3, i as f32, d)?)?;
    }
    let q = Tensor::ones((32, 2), DType::F32, d)?;
    let weights = Tensor::ones((32, 1), DType::F32, d)?;
    for n in [1, 255, 256, 257, 511, 512, 513] {
        let prefix = keys.prefix(n)?;
        let sizes = prefix
            .chunks()
            .map(|c| c.unwrap().dims()[0])
            .collect::<Vec<_>>();
        assert_eq!(sizes.iter().sum::<usize>(), n);
        assert_eq!(*sizes.last().unwrap(), (n - 1) % 256 + 1);
        let scores = attention::score_prefix(&q, &weights, &prefix)?;
        assert_eq!(scores.dims(), [n]);
        let score_values = scores.to_vec1::<f32>()?;
        assert_eq!(score_values[0], 0.);
        if n == 257 {
            assert_eq!(score_values[256], 256.);
        }
        let mut scalar_prefix = Rows::default();
        for i in 0..n {
            scalar_prefix.append(row(2, if i >= 257 { 1e6 } else { i as f32 }, d)?)?;
        }
        let scalar_scores =
            attention::score_prefix(&q, &weights, &scalar_prefix.prefix(n)?)?.to_vec1::<f32>()?;
        assert_eq!(
            score_values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            scalar_scores
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let ids = attention::select_scores(&scores, 2, &mut Selection::default())?;
        let selected = ids.to_vec1::<f32>()?;
        assert!(selected.iter().all(|&i| i >= 0. && i < n as f32));
        let gathered = values.prefix(n)?.gather(&ids, 3)?.to_vec2::<f32>()?;
        for (id, value) in selected.iter().zip(gathered) {
            assert_eq!(value, vec![*id; 3]);
        }
        if n == 257 {
            assert_eq!(selected[0], 256.);
        }
    }
    assert!(keys.prefix(521).is_err());
    let bad = Tensor::new(&[257f32], d)?;
    assert!(values.prefix(257)?.gather(&bad, 3).is_err());
    assert_eq!(keys.prefix(0)?.chunks().count(), 0);
    Ok(())
}

#[test]
fn native_tile_cpu_ratio_frontiers_pending_carry_and_transactional_error() -> Result<()> {
    // Scalar oracle crosses an odd boundary. The candidate executes owner-major
    // publishes, then later consumers see only their own absolute frontiers.
    let mut committed = DeepSeek41Cache::default();
    advance(&mut committed, 1.)?;
    let before = committed.frontier_report()?;
    let end = profile::MAX_PREFILL_TILE + 1;
    let mut oracle = committed.clone();
    for value in 2..=end {
        advance(&mut oracle, value as f32)?;
    }
    let mut candidate = committed.clone();
    for l in 0..40 {
        for pos in 1..end {
            candidate.append_raw(l, row(512, (pos + 1) as f32, &Device::Cpu)?)?;
        }
    }
    for owner in 0..4 {
        let c = &mut candidate.compressed[owner];
        for pos in 1..end {
            let value = (pos + 1) as f32;
            let score = if owner == 3 {
                None
            } else {
                Some(row(512, value / 10., &Device::Cpu)?)
            };
            if let Some((kv, _)) =
                attention::pool_projected(c, row(512, value, &Device::Cpu)?, score, pos)?
            {
                c.kv.append(kv)?;
                c.index.append(row(128, value, &Device::Cpu)?)?;
            }
        }
        for pos in 1..end {
            let visible = cache::visible_rows(owner, pos);
            assert_eq!(visible, if owner == 3 { pos + 1 } else { pos.div_ceil(2) });
            let prefix = c.index.prefix(visible)?;
            let actual = Tensor::cat(
                &prefix.chunks().collect::<candle_core::Result<Vec<_>>>()?,
                0,
            )?
            .to_vec2::<f32>()?;
            assert_eq!(
                actual.last().unwrap()[0],
                if owner == 3 {
                    (pos + 1) as f32
                } else {
                    (visible * 2) as f32
                }
            );
        }
        assert_eq!(c.pending.is_some(), owner != 3);
        assert_eq!(
            c.kv.chunks[0].to_vec2::<f32>()?,
            oracle.compressed[owner].kv.chunks[0].to_vec2::<f32>()?
        );
    }
    // Mid-tile error: discard all the private frontiers, never publish them.
    assert!(candidate.validate_frontier().is_err()); // Position is still 1.
    assert_eq!(committed.frontier_report()?, before);
    candidate.position = end;
    candidate.validate_frontier()?;
    assert_eq!(candidate.frontier_report()?, oracle.frontier_report()?);
    for pos in [0, 1, 2, 3, 510, 511, 512, 1_048_575] {
        assert_eq!(cache::visible_rows(0, pos), pos.div_ceil(2));
        assert_eq!(cache::visible_rows(3, pos), pos + 1);
    }
    Ok(())
}

#[test]
fn native_tile_cpu_selection_masks_are_per_token_and_causal() -> Result<()> {
    // Two current-token selections with different visible lengths and block0
    // scores. Layer20 pins its own newest block, never a later tile row's block.
    let mut selections: [Selection; 2] = std::array::from_fn(|_| Selection::default());
    let mut expected = Vec::new();
    for (r, n) in [16385usize, 16401].into_iter().enumerate() {
        let scores = Tensor::new(
            (0..n)
                .map(|i| {
                    if i / 8 == r {
                        -1e6
                    } else if i / 8 == 0 {
                        1e6
                    } else {
                        i as f32
                    }
                })
                .collect::<Vec<_>>(),
            &Device::Cpu,
        )?;
        let ids = attention::select_scores(&scores, 20, &mut selections[r])?;
        let gathered_ids = ids.to_vec1::<f32>()?;
        assert!(gathered_ids.iter().all(|&id| id < n as f32));
        let reverse = Tensor::new(
            (0..n).map(|i| -(i as f32)).collect::<Vec<_>>(),
            &Device::Cpu,
        )?;
        expected
            .push(attention::select_scores(&reverse, 24, &mut selections[r])?.to_vec1::<f32>()?);
    }
    let n = 16385;
    let reverse = Tensor::new(
        (0..n).map(|i| -(i as f32)).collect::<Vec<_>>(),
        &Device::Cpu,
    )?;
    assert_eq!(
        attention::select_scores(&reverse, 28, &mut selections[0])?.to_vec1::<f32>()?,
        expected[0]
    );
    assert!(!expected[0].iter().any(|&id| id < 8.));
    assert!(expected[1].iter().any(|&id| id < 8.));
    assert_ne!(expected[0], expected[1]);
    Ok(())
}

#[test]
fn native_tile_cpu_adaptive_pooled_bound_chunks_tails_and_context_rejection() -> Result<()> {
    use profile::{
        MAX_PREFILL_TILE, prefill_tile_plan_with_limit as plan, prefill_tile_scratch as scratch,
    };
    let old = 16 * 1024 * 1024;
    assert_eq!(plan(7, 5291, 7, old)?.effective, 1);
    for end in 8193..=9780 {
        assert_eq!(plan(7, end - 7, 7, old)?.effective, 1);
    }
    assert_eq!(plan(7, 5291, 7, FIXED_CACHE_BYTES)?.effective, 7);
    for end in [
        32768, 32769, 64852, 65536, 65537, 131072, 131073, 262144, 262145, 524288, 524289,
        1_048_576,
    ] {
        let expected = plan(MAX_PREFILL_TILE, 0, end, FIXED_CACHE_BYTES)?.effective;
        assert!(scratch(expected, end) <= FIXED_CACHE_BYTES);
        if expected < MAX_PREFILL_TILE {
            assert!(scratch(expected + 1, end) > FIXED_CACHE_BYTES);
        }
    }
    assert_eq!(
        plan(MAX_PREFILL_TILE, 1_048_575, 1, FIXED_CACHE_BYTES)?.reason,
        "single_token_decode_or_tail"
    );
    assert_eq!(
        plan(MAX_PREFILL_TILE, 8192, 3, FIXED_CACHE_BYTES)?.reason,
        "input_tail"
    );
    assert_eq!(
        plan(1, 0, 1_048_576, FIXED_CACHE_BYTES)?.scratch_bound_bytes,
        0
    );
    // Production scheduler chunk128, odd tails, and exact 1M all use the same
    // whole-forward end and preserve every input row in its original order.
    for end in [1, 3, 127, 128, 5298, 8192, 8193, 9780, 65536, 1_048_576] {
        for requested in 1..=MAX_PREFILL_TILE {
            for limit in [1, old, FIXED_CACHE_BYTES] {
                let count = end.min(128);
                let p = plan(requested, end - count, count, limit)?;
                assert!((1..=requested.min(count)).contains(&p.effective));
                if p.effective == 1 {
                    assert_eq!(p.scratch_bound_bytes, 0);
                } else {
                    assert!(p.scratch_bound_bytes <= limit);
                    if p.effective < requested.min(count) {
                        assert!(scratch(p.effective + 1, end) > limit);
                    }
                }
                let rows: Vec<_> = (0..count).collect();
                assert_eq!(
                    rows.chunks(p.effective)
                        .flatten()
                        .copied()
                        .collect::<Vec<_>>(),
                    rows
                );
                assert!(
                    rows.chunks(p.effective)
                        .all(|r| r.len() <= MAX_PREFILL_TILE)
                );
            }
        }
    }
    for (requested, start, count) in [
        (0, 0, 1),
        (MAX_PREFILL_TILE + 1, 0, 1),
        (MAX_PREFILL_TILE, 0, 0),
        (MAX_PREFILL_TILE, usize::MAX, 32),
        (MAX_PREFILL_TILE, 1_048_576, 1),
    ] {
        assert!(plan(requested, start, count, FIXED_CACHE_BYTES).is_err());
    }
    assert!(plan(32, 0, 32, 0).is_err());
    assert!(plan(32, 0, 32, FIXED_CACHE_BYTES + 1).is_err());
    assert!(
        DeepSeek41Cache::default()
            .validate_position(0, 33, 32)
            .is_err()
    );
    Ok(())
}

#[test]
fn native_tile_cpu_index_tail_prediction_and_long_plan_stay_admitted() -> Result<()> {
    let _packet_off = Iq1PacketGuard::set(false);
    // A larger immutable index tail is rounded per kind, not a second copy of
    // all index rows. The default256 KV gather geometry remains unchanged.
    assert_eq!(
        DeepSeek41Cache::predicted_dynamic_bytes(64852, true)?,
        416_546_816
    );
    assert_eq!(
        DeepSeek41Cache::predicted_dynamic_bytes(1_048_576, true)?,
        6_710_886_400
    );
    assert_eq!(
        DeepSeek41Cache::predicted_dynamic_bytes(64852, false)?,
        64852 * 6400
    );
    let plan = profile::prefill_tile_plan(128, 64852, 128)?;
    assert_eq!(plan.effective, 128);
    assert!(plan.scratch_bound_bytes <= FIXED_CACHE_BYTES);
    let full = profile::prefill_tile_plan(128, 1_048_448, 128)?;
    assert!(full.effective < 128 && full.scratch_bound_bytes <= FIXED_CACHE_BYTES);
    assert!(profile::prefill_tile_scratch(full.effective + 1, full.end) > FIXED_CACHE_BYTES);
    assert!(DeepSeek41Cache::predicted_dynamic_bytes(1_048_577, true).is_err());
    Ok(())
}

#[test]
fn native_tile_cpu_router_row_order_exact_top6_and_union_dedup() -> Result<()> {
    let bias: Vec<f32> = (0..384).map(|i| (i % 5) as f32 * 0.01).collect();
    let rows: Vec<Vec<f32>> = (0..profile::MAX_PREFILL_TILE)
        .map(|r| {
            (0..384)
                .map(|i| ((i * 37 + (r % 3) * 19) % 67) as f32 / 8. - 4.)
                .collect()
        })
        .collect();
    let projected = rows
        .iter()
        .map(|r| Tensor::new(r.as_slice(), &Device::Cpu)?.reshape((1, 384)))
        .collect::<candle_core::Result<Vec<_>>>()?;
    let values = Tensor::cat(&projected, 0)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let routes = tile_routes(&values, &bias, profile::MAX_PREFILL_TILE)?;
    for (r, route) in routes.iter().enumerate() {
        assert_eq!(*route, math::router_top6(&rows[r], Some(&bias))?);
    }
    let union = route_union(&routes);
    let mut expected = Vec::new();
    for route in &routes {
        for id in route.indices {
            if !expected.contains(&id) {
                expected.push(id);
            }
        }
    }
    assert_eq!(union, expected);
    assert!(union.len() <= profile::MAX_PREFILL_TILE * 6);
    assert!(union.len() < profile::MAX_PREFILL_TILE * 6); // Shared routes.
    assert!(
        tile_routes(
            &values[..values.len() - 1],
            &bias,
            profile::MAX_PREFILL_TILE
        )
        .is_err()
    );
    assert!(
        tile_routes(
            &vec![0.; (profile::MAX_PREFILL_TILE + 1) * 384],
            &bias,
            profile::MAX_PREFILL_TILE + 1
        )
        .is_err()
    );
    let mut nonfinite = values;
    nonfinite[384 + 5] = f32::NAN;
    assert!(tile_routes(&nonfinite, &bias, profile::MAX_PREFILL_TILE).is_err());
    Ok(())
}

#[test]
fn native_tile_cpu_engram_history_and_duplicate_hash_rows_replay_exactly() -> Result<()> {
    use engram::*;
    let primes = [
        [
            16000057, 16000079, 16000081, 16000097, 16000121, 16000129, 16000133, 16000183,
            16000189, 16000207, 16000211, 16000253, 16000277, 16000289, 16000307, 16000321,
            16000339, 16000381, 16000393, 16000399, 16000403, 16000409, 16000447, 16000463,
        ],
        [
            16000477, 16000487, 16000499, 16000507, 16000511, 16000573, 16000609, 16000627,
            16000667, 16000669, 16000693, 16000697, 16000711, 16000729, 16000759, 16000769,
            16000781, 16000799, 16000813, 16000819, 16000841, 16000877, 16000879, 16000889,
        ],
    ];
    let layout = Layout::new(LayoutMetadata {
        architecture: ARCHITECTURE.into(),
        encoding: ENCODING.into(),
        vocab_size: VOCAB_SIZE,
        compressed_vocab_size: COMPRESSED_VOCAB_SIZE,
        pad_id: 2,
        layer_ids: LAYER_IDS,
        rows: TABLE_ROWS.map(u64::from),
        token_map: (0..VOCAB_SIZE)
            .map(|id| id % COMPRESSED_VOCAB_SIZE)
            .collect(),
        primes,
        multipliers: [[1, 3, 5, 7], [11, 13, 17, 19]],
        tables: std::array::from_fn(|i| TableGeometry {
            name: format!("blk.{}.engram_embd.weight", LAYER_IDS[i]),
            dtype: 24,
            dimensions: [ROW_BYTES as u64, TABLE_ROWS[i] as u64],
        }),
    })?;
    let tokens: Vec<_> = [4, 4, 4, 4, 4, 4, 4, 8, 5, 8, 5, 8, 5, 8, 5]
        .into_iter()
        .cycle()
        .take(65)
        .collect();
    let whole = layout.hash(&History::default(), &tokens, None)?;
    assert_eq!(whole.rows[3], whole.rows[4]);
    for tile in 1..=profile::MAX_PREFILL_TILE {
        let mut history = History::default();
        let mut actual = Vec::new();
        for chunk in tokens.chunks(tile) {
            let batch = layout.hash(&history, chunk, None)?;
            actual.extend(batch.rows);
            history = batch.next_history;
        }
        assert_eq!(actual, whole.rows);
        assert_eq!(history, whole.next_history);
        let before = history.clone();
        assert!(layout.hash(&history, &[4, -1], None).is_err());
        assert_eq!(history, before);
    }
    Ok(())
}

#[test]
fn native_tile_cpu_backing_alias_dedup_and_exact_hc_owner_views() -> Result<()> {
    let pattern = [
        0.0f32,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc1_2345),
        0.125,
        -3.5,
        1.0,
    ];
    let values: Vec<_> = pattern.into_iter().cycle().take(64).collect();
    let capacity_bytes = values.capacity() * 4;
    let backing = Tensor::from_vec(values.clone(), 64, &Device::Cpu)?;
    let owner = math::HcSplit::from_backing(backing.narrow(0, 8, 24)?)?;
    let mut rows = vec![TileRow::new(
        backing.narrow(0, 0, 32)?.reshape((4, 8))?,
        backing.narrow(0, 4, 4)?,
    )?];
    rows[0].set_ffn(owner)?;
    rows[0].set_input(backing.narrow(0, 16, 8)?.reshape((1, 8))?)?;
    rows[0].selection.selected[0] = Some(backing.narrow(0, 32, 8)?);
    rows[0].selection.candidate_mask = Some(backing.narrow(0, 40, 8)?);
    rows[0].selection.retain_bounded()?;
    let mut leases = MoeLeases::default();
    leases.retain_tile_inputs(&rows)?;
    leases.tile_gate_up[0] = Some(backing.narrow(0, 48, 16)?);
    assert_eq!(tile_owner_bytes(&rows, &leases), capacity_bytes);
    // CPU identity is preserved even for a logically smaller view. Deduplication
    // must use storage, not Tensor IDs, offsets, shapes or logical byte lengths.
    assert_eq!(
        cache::retained_bytes(std::iter::once(&rows[0].previous)),
        capacity_bytes
    );
    let hc = rows[0].ffn.as_ref().unwrap();
    for (view, range) in [(hc.pre(), 8..12), (hc.post(), 12..16), (hc.comb(), 16..32)] {
        let actual: Vec<_> = view
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect();
        let expected: Vec<_> = values[range].iter().map(|x| x.to_bits()).collect();
        assert_eq!(actual, expected);
        assert_eq!(
            cache::retained_bytes([hc.backing(), view].into_iter()),
            capacity_bytes
        );
    }
    // A distinct allocation is charged even when its values are identical.
    let other = Tensor::from_vec(values, 64, &Device::Cpu)?;
    assert_eq!(
        cache::retained_bytes([&backing, &other].into_iter()),
        2 * capacity_bytes
    );
    Ok(())
}

#[test]
fn native_tile_cpu_backing_interval_handoff_and_ffn_update() -> Result<()> {
    let d = &Device::Cpu;
    let mut rows = (0..profile::MAX_PREFILL_TILE)
        .map(|r| TileRow::new(Tensor::full(r as f32 / 128., (4, 8), d)?, initial_pre(d)?))
        .collect::<Result<Vec<_>>>()?;
    let mut leases = MoeLeases::default();
    for interval in 0..4 {
        let mut expected = Vec::new();
        let out = Tensor::full((interval + 1) as f32 / 16., 8, d)?;
        for (r, row) in rows.iter_mut().enumerate() {
            let split = math::HcSplit::from_backing(Tensor::full((r + 1) as f32 / 256., 24, d)?)?;
            expected.push(math::hc_expand(&out, &row.residual, &split)?.to_vec2::<f32>()?);
            row.set_ffn(split)?;
            row.set_input(Tensor::full((interval + r) as f32, (1, 8), d)?)?;
        }
        let bytes_before = tile_owner_bytes(&rows, &leases);
        // The preceding lease is retired only at the simulated successful
        // CPU router readback; current-row input then becomes the new lease.
        let router = Tensor::zeros((profile::MAX_PREFILL_TILE, 384), DType::F32, d)?;
        assert_eq!(
            router.to_vec2::<f32>()?,
            vec![vec![0.; 384]; profile::MAX_PREFILL_TILE]
        );
        leases = MoeLeases::default();
        leases.retain_tile_inputs(&rows)?;
        assert!(tile_owner_bytes(&rows, &leases) <= bytes_before);
        for (r, row) in rows.iter_mut().enumerate() {
            row.finish_ffn(&out)?;
            assert_eq!(row.residual.to_vec2::<f32>()?, expected[r]);
            assert_eq!(
                row.previous.to_vec1::<f32>()?,
                vec![(r + 1) as f32 / 256.; 4]
            );
            assert!(row.ffn.is_none() && row.input.is_none());
            assert_eq!(
                leases.tile_inputs[r].to_vec2::<f32>()?,
                vec![vec![(interval + r) as f32; 8]]
            );
        }
        assert_eq!(leases.tile_inputs.len(), profile::MAX_PREFILL_TILE);
    }
    Ok(())
}

#[test]
fn native_tile_cpu_backing_exact_comparison_bound_and_rejection() -> Result<()> {
    let plan = profile::prefill_tile_plan_with_limit(32, 0, 9780, FIXED_CACHE_BYTES)?;
    let bound = profile::tile_owner_comparison_bound(plan)?;
    assert_eq!(plan.effective, 32);
    assert_eq!(
        bound + profile::TILE_RAW_PENDING_BYTES + 32 * profile::TILE_ACTIVE_SCRATCH_BYTES,
        plan.scratch_bound_bytes
    );
    assert_eq!(profile::check_tile_owner_bytes(bound, plan)?, bound);
    assert!(profile::check_tile_owner_bytes(bound + 1, plan).is_err());
    assert!(
        profile::tile_owner_comparison_bound(profile::prefill_tile_plan(7, 1_048_575, 1)?).is_err()
    );
    Ok(())
}

// Native-only filter: never included in native_tile_cpu/backing CPU commands.
#[cfg(target_os = "macos")]
#[test]
fn native_tile_metal_backing_rows_hc_selection_leases_and_error_lifetime() -> Result<()> {
    use anyhow::Context;

    fn values(n: usize) -> Vec<f32> {
        [
            0.0f32,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7fc1_2345),
            0.125,
            -3.5,
            1.0,
        ]
        .into_iter()
        .cycle()
        .take(n)
        .collect()
    }
    fn oversized(dims: &[usize], device: &Device) -> Result<Tensor> {
        let n: usize = dims.iter().product();
        let mut padded = vec![-999f32; n * 16];
        padded[3..3 + n].copy_from_slice(&values(n));
        let view = Tensor::from_vec(padded, n * 16, device)
            .context("allocate oversized offset carrier")?
            .narrow(0, 3, n)
            .context("slice oversized offset carrier")?
            .reshape(dims)
            .context("reshape oversized offset carrier")?;
        assert!(metal_backing_bytes(&view) > (n * 4).next_power_of_two());
        Ok(view)
    }
    fn oversized_hc(packed: Vec<f32>, device: &Device) -> Result<math::HcSplit> {
        assert_eq!(packed.len(), 24);
        let mut padded = vec![-999f32; 24 * 16];
        padded[..24].copy_from_slice(&packed);
        // HC validation precedes retention: keep this input valid but oversized.
        let carrier = Tensor::from_vec(padded, 24 * 16, device)
            .context("allocate oversized HC carrier")?
            .narrow(0, 0, 24)
            .context("slice zero-offset HC carrier")?;
        assert_eq!(carrier.dims(), &[24]);
        assert_eq!(carrier.dtype(), DType::F32);
        assert!(carrier.is_contiguous());
        assert_eq!(carrier.layout().start_offset(), 0);
        assert!(metal_backing_bytes(&carrier) > (24usize * 4).next_power_of_two());
        math::HcSplit::from_backing(carrier).context("validate oversized HC carrier")
    }
    fn assert_bits(tensor: &Tensor) -> Result<()> {
        let actual: Vec<_> = tensor
            .flatten_all()
            .context("flatten retained carrier")?
            .to_vec1::<f32>()
            .context("read retained carrier")?
            .into_iter()
            .map(f32::to_bits)
            .collect();
        let expected: Vec<_> = values(tensor.elem_count())
            .into_iter()
            .map(f32::to_bits)
            .collect();
        assert_eq!(actual, expected);
        Ok(())
    }
    fn assert_bounded(
        rows: &[TileRow],
        leases: &MoeLeases,
        plan: profile::PrefillTilePlan,
    ) -> Result<()> {
        for tensor in rows
            .iter()
            .flat_map(TileRow::tensors)
            .chain(leases.tensors())
        {
            assert!(
                metal_backing_bytes(tensor)
                    <= super::super::fixed_state_tensor_bytes(tensor.dims(), tensor.device())
                        .context("calculate retained carrier bound")?
            );
        }
        profile::check_tile_owner_bytes(tile_owner_bytes(rows, leases), plan)
            .context("check tile owner bound")?;
        Ok(())
    }
    let d = Device::new_metal(0).context("create fixture Metal device")?;
    // Prime a larger pool bucket, then FORCE oversized carriers: offset views
    // for rows/inputs, zero-offset contiguous packed HC inputs.
    let prime = Tensor::full(-999f32, 1 << 20, &d).context("prime Metal pool")?;
    drop(prime);
    let plan = profile::prefill_tile_plan(7, 5291, 7).context("plan fixture tile")?;
    let mut rows = Vec::new();
    let mut leases = MoeLeases::default();
    for _ in 0..2 {
        rows.push(
            TileRow::new(oversized(&[4, 5120], &d)?, oversized(&[4], &d)?)
                .context("retain initial row residual and previous pre")?,
        );
        let row = rows.last_mut().unwrap();
        row.set_ffn(oversized_hc(values(24), &d)?)
            .context("retain initial packed HC")?;
        row.set_input(oversized(&[1, 5120], &d)?)
            .context("retain row input")?;
        for owner in 0..4 {
            row.selection.selected[owner] = Some(
                oversized_metal_row(512, owner as f32, &d)
                    .with_context(|| format!("construct selected owner {owner}"))?
                    .squeeze(0)
                    .context("squeeze selected owner")?,
            );
        }
        row.selection.candidate_mask = Some(
            oversized_metal_row(2048, 1., &d)
                .context("construct candidate mask")?
                .squeeze(0)
                .context("squeeze candidate mask")?,
        );
        row.selection
            .retain_bounded()
            .context("retain selection views")?;
        // Preceding interval owners are distinct from the new row inputs.
        leases.tile_inputs.push(
            cache::retain_bounded(oversized(&[1, 5120], &d)?)
                .context("retain preceding input lease")?,
        );
        leases.tile_gate_up.push(Some(
            cache::retain_bounded(oversized(&[12, 2304], &d)?)
                .context("retain preceding gate-up lease")?,
        ));
        assert_bounded(&rows, &leases, plan).context("check initial row and lease bounds")?;
    }
    // Same production row update helper, while the previous interval is live.
    rows[0]
        .set_residual(oversized(&[4, 5120], &d)?)
        .context("replace oversized residual")?;
    assert_bounded(&rows, &leases, plan).context("check replaced residual bound")?;
    let fitting =
        super::super::copy_persistent_tensor(&rows[0].residual).context("copy fitting residual")?;
    let fitting_key = metal_backing_key(&fitting);
    rows[0]
        .set_residual(fitting)
        .context("retain fitting residual")?;
    assert_eq!(metal_backing_key(&rows[0].residual), fitting_key);
    let hc = rows[0].ffn.as_ref().unwrap();
    let carrier_key = metal_backing_key(hc.backing());
    assert_eq!(metal_backing_key(hc.pre()), carrier_key);
    assert_eq!(metal_backing_key(hc.post()), carrier_key);
    assert_eq!(metal_backing_key(hc.comb()), carrier_key);
    let fitting_hc = retain_tile_hc(hc.clone()).context("retain fitting HC without copying")?;
    assert_eq!(metal_backing_key(fitting_hc.backing()), carrier_key);
    drop(fitting_hc);
    // Existing successful router readback retires ONLY the preceding interval.
    let router = Tensor::zeros(384, DType::F32, &d).context("allocate router readback")?;
    assert_eq!(
        router
            .to_vec1::<f32>()
            .context("read router retirement boundary")?,
        vec![0.; 384]
    );
    for row in &rows {
        assert_bits(&row.residual).context("check retained residual bits")?;
        assert_bits(&row.previous).context("check retained previous-pre bits")?;
        assert_bits(row.ffn.as_ref().unwrap().backing()).context("check retained HC bits")?;
        assert_bits(row.input.as_ref().unwrap()).context("check retained input bits")?;
        for (owner, ids) in row.selection.selected.iter().enumerate() {
            assert_eq!(
                ids.as_ref()
                    .unwrap()
                    .to_vec1::<f32>()
                    .with_context(|| format!("read retained selected owner {owner}"))?,
                vec![owner as f32; 512]
            );
        }
        assert_eq!(
            row.selection
                .candidate_mask
                .as_ref()
                .unwrap()
                .to_vec1::<f32>()
                .context("read retained candidate mask")?,
            vec![1.; 2048]
        );
    }
    for tensor in leases.tensors() {
        assert_bits(tensor).context("check preceding lease bits")?;
    }
    leases = MoeLeases::default();
    leases
        .retain_tile_inputs(&rows)
        .context("retain current input leases")?;
    for gate_up in &mut leases.tile_gate_up {
        *gate_up = Some(
            cache::retain_bounded(oversized(&[12, 2304], &d)?)
                .context("retain current gate-up lease")?,
        );
    }
    leases.shared_input = Some(
        cache::retain_bounded(oversized(&[rows.len(), 5120], &d)?)
            .context("retain joined shared-expert input")?,
    );
    leases.shared_output = Some(
        cache::retain_bounded(oversized(&[rows.len(), 5120], &d)?)
            .context("retain joined shared-expert output")?,
    );
    assert_eq!(
        metal_backing_key(&leases.tile_inputs[0]),
        metal_backing_key(rows[0].input.as_ref().unwrap())
    );
    let naive: usize = rows
        .iter()
        .flat_map(TileRow::tensors)
        .chain(leases.tensors())
        .map(metal_backing_bytes)
        .sum();
    assert_eq!(
        naive - tile_owner_bytes(&rows, &leases),
        leases
            .tile_inputs
            .iter()
            .map(metal_backing_bytes)
            .sum::<usize>()
    );
    assert_bounded(&rows, &leases, plan).context("check shared input lease bounds")?;
    // Queue an exact dependent copy, then trigger a real shape error. Inputs
    // remain owned through the same mandatory outer error fence, without a
    // source-owner side list. Source temporaries above have already been dropped.
    let pending = super::super::copy_persistent_tensor(&leases.tile_inputs[0])
        .context("queue dependent input copy")?;
    let bad_out = Tensor::zeros(3, DType::F32, &d).context("allocate invalid FFN output")?;
    assert!(rows[0].finish_ffn(&bad_out).is_err());
    assert!(rows[0].ffn.is_some() && rows[0].input.is_some());
    let pressure = Tensor::full(999f32, (12, 2304), &d).context("allocate error-path pressure")?;
    assert_bounded(&rows, &leases, plan).context("check error-path owner bounds")?;
    d.synchronize()
        .context("synchronize error lifetime boundary")?;
    drop(pressure);
    assert_bits(&pending).context("check dependent copy after error fence")?;
    for tensor in leases.tensors() {
        assert_bits(tensor).context("check leases after error fence")?;
    }
    // Successful GPU update: use finite HC values, compare the unchanged
    // expansion with the original operation, then bound the next-pre view.
    rows[0]
        .set_residual(
            oversized_metal_row(5120, 0.25, &d)
                .context("construct finite strided residual")?
                .broadcast_as((4, 5120))
                .context("broadcast finite strided residual")?,
        )
        .context("retain finite broadcast residual")?;
    rows[0]
        .set_ffn(oversized_hc(vec![0.125; 24], &d)?)
        .context("retain finite packed HC")?;
    let block = Tensor::full(0.25f32, 5120, &d).context("allocate finite block output")?;
    let reference = math::hc_expand(&block, &rows[0].residual, rows[0].ffn.as_ref().unwrap())
        .context("compute finite HC reference from retained residual")?;
    rows[0]
        .finish_ffn(&block)
        .context("finish finite FFN update")?;
    assert_bounded(&rows, &leases, plan).context("check successful update owner bounds")?;
    d.synchronize()
        .context("synchronize successful FFN update")?;
    let bits = |t: &Tensor| -> Result<Vec<u32>> {
        Ok(t.flatten_all()
            .context("flatten finite output")?
            .to_vec1::<f32>()
            .context("read finite output")?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    };
    assert_eq!(bits(&rows[0].residual)?, bits(&reference)?);
    assert_eq!(
        rows[0]
            .previous
            .to_vec1::<f32>()
            .context("read next previous pre")?,
        vec![0.125; 4]
    );
    assert!(rows[0].ffn.is_none() && rows[0].input.is_none());
    drop(leases);
    drop(rows);
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn attention_phase_metal_whole_tile_forced_loans_and_error_owners() -> Result<()> {
    let _packet_off = Iq1PacketGuard::set(false);
    use serde_json::json;
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            ATTENTION_PHASED.with(|v| v.set(false));
        }
    }
    let _restore = Restore;
    let d = Device::new_metal(0)?;
    let forced = |rows: usize, width: usize, value: f32| -> Result<Tensor> {
        let n = rows * width;
        let source = Tensor::full(value, n * 4 + 3, &d)?;
        let view = source.narrow(0, 3, n)?.reshape((rows, width))?;
        assert!(metal_backing_bytes(&view) > (n * 4).next_power_of_two());
        Ok(view)
    };
    let mut reports = Vec::new();
    for frontier in [2048, 64852] {
        ATTENTION_PHASED.with(|v| v.set(true));
        let plan = profile::prefill_tile_plan(32, frontier, 128)?;
        assert!(plan.scratch_bound_bytes <= FIXED_CACHE_BYTES);
        let mut rows = Vec::new();
        // A preceding router interval must survive all attention phases.
        let mut leases = MoeLeases {
            input: Some(cache::retain_bounded(forced(1, 5120, 0.75)?)?),
            ..MoeLeases::default()
        };
        let preceding = metal_backing_key(leases.input.as_ref().unwrap());
        for _ in 0..plan.effective {
            rows.push(TileRow::new(forced(4, 5120, 0.25)?, initial_pre(&d)?)?);
            let packed = forced(1, 24, 0.125)?.reshape(24)?;
            leases
                .attention
                .splits
                .push(retain_tile_hc(math::HcSplit::from_backing(packed)?)?);
            leases
                .attention
                .inputs
                .push(cache::retain_bounded(forced(1, 5120, 0.5)?)?);
        }
        let mut phase_bytes = Vec::new();
        for (phase, widths) in [
            (1, vec![5120, 1280, 512, 32768]),
            (2, vec![8192]),
            (3, vec![8192, 5120]),
        ] {
            for width in widths {
                if phase == 2 {
                    for _ in 0..plan.effective {
                        leases
                            .attention
                            .lows
                            .push(cache::retain_bounded(forced(1, width, 0.25)?)?);
                    }
                } else {
                    leases.attention.pin(forced(plan.effective, width, 0.25)?)?;
                }
            }
            let bytes = tile_owner_bytes(&rows, &leases);
            profile::check_tile_owner_bytes(bytes, plan)?;
            let phase_charge = plan.effective * 512 * 1024;
            assert!(cache::retained_bytes(leases.attention.tensors()) <= phase_charge);
            assert_eq!(metal_backing_key(leases.input.as_ref().unwrap()), preceding);
            for t in leases.attention.tensors() {
                assert!(
                    metal_backing_bytes(t) <= super::super::fixed_state_tensor_bytes(t.dims(), &d)?
                );
            }
            // Queue dependent work, force a real mid-phase shape error, apply
            // pressure, then use the same mandatory error fence before release.
            let source = leases.attention.carriers.last().unwrap();
            let queued = super::super::copy_persistent_tensor(source)?;
            assert!(source.reshape(3).is_err());
            let pressure = Tensor::full(99f32, (plan.effective, 32768), &d)?;
            d.synchronize()?;
            assert!(
                queued
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|v| v.to_bits() == 0.25f32.to_bits())
            );
            drop(pressure);
            phase_bytes.push(json!({"phase":phase,"deduplicated_tile_owner_bytes":bytes,
                "phase_owner_bytes":cache::retained_bytes(leases.attention.tensors()),
                "phase_charge_bytes":phase_charge,"error_fence_exact":true}));
        }
        d.synchronize()?;
        drop(leases);
        drop(rows);
        reports.push(json!({"frontier":frontier,"tile":plan.effective,
            "scratch_bytes":plan.scratch_bound_bytes,"phases":phase_bytes}));
    }
    println!(
        "attention owner evidence: {}",
        serde_json::to_string_pretty(&json!({"forced_oversized_offset_backings":true,
            "previous_interval_preserved":true,"reports":reports}))?
    );
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn attention_v2_metal_full_tile_prior_interval_short_slice_and_loan_rejection() -> Result<()> {
    let d = Device::new_metal(0)?;
    let _candidate = AttentionV2Guard::set(true);
    let _engram = EngramV2Guard::set(true);
    let _packet = Iq1PacketGuard::set(true);
    let fitting = |height: usize, width: usize| -> Result<Tensor> {
        let n = height * width;
        let bucket = (n * 4).next_power_of_two();
        let source = Tensor::from_vec(vec![0.25f32; bucket / 4], bucket / 4, &d)?;
        let view = source.narrow(0, 0, n)?.reshape((height, width))?;
        ensure!(
            metal_backing_bytes(&view) == bucket,
            "forced fitting loan required"
        );
        Ok(view)
    };
    let plan = profile::prefill_tile_plan_with_limit(32, 2048, 33, FIXED_CACHE_BYTES)?;
    assert!(plan.effective > 1);
    let committed = DeepSeek41Cache::seeded_native_fixture(2048, &d)?;
    let frontier = committed.frontier_report()?;
    let mut candidate = committed.clone();
    let mut leases = MoeLeases::default();
    let mut rows = Vec::new();
    for _ in 0..plan.effective {
        rows.push(TileRow::new(fitting(4, 5120)?, fitting(1, 4)?.reshape(4)?)?);
        leases.tile_inputs.push(fitting(1, 5120)?);
        leases.tile_gate_up.push(Some(fitting(12, 2304)?));
    }
    // Full preceding interval survives a short final slice, including joined
    // Engram and attention projection owners; no logical-size substitution.
    for width in [6144, 25600, 5120, 1280, 512, 32768, 8192, 5120] {
        leases.attention.pin(fitting(plan.effective, width)?)?;
    }
    let prior = tile_owner_bytes(&rows, &leases);
    let q = fitting(plan.effective, 4096)?.reshape((plan.effective, 32, 128))?;
    let weights = fitting(plan.effective, 32)?;
    let pages = candidate.compressed[3].index.chunks.clone();
    leases.attention.batch_cache.extend(
        candidate
            .compressed
            .iter()
            .flat_map(|c| c.kv.chunks.iter().chain(c.index.chunks.iter()))
            .cloned(),
    );
    let visible: Vec<_> = (0..plan.effective)
        .map(|r| 2048 - plan.effective + r + 1)
        .collect();
    let output =
        ops::index_scores_many(&q, &weights, &pages, &visible, &mut leases.attention.batch)?;
    rows.truncate(1); // prior full carriers remain live
    let full_owned = tile_owner_bytes(&rows, &leases);
    assert!(full_owned >= prior - (plan.effective - 1) * (131072 + 16));
    assert!(full_owned <= profile::tile_owner_comparison_bound(plan)?);
    let dynamic = cache::retained_bytes(leases.attention.batch_cache.iter());
    assert!(dynamic <= DeepSeek41Cache::predicted_dynamic_bytes(2048, true)?);
    assert!(
        full_owned + profile::TILE_RAW_PENDING_BYTES + dynamic
            <= plan.scratch_bound_bytes + DeepSeek41Cache::predicted_dynamic_bytes(2048, true)?
    );
    // Actual scorer results and causal invisible suffix, not a carrier-only test.
    d.synchronize()?;
    let scores = output.to_vec2::<f32>()?;
    for (r, row) in scores.iter().enumerate() {
        assert!(row[..visible[r]].iter().all(|&v| (v - 4.0).abs() < 1e-5));
        assert!(row[visible[r]..].iter().all(|&v| v == f32::NEG_INFINITY));
    }
    let source = fitting(plan.effective * 16, 512)?;
    let oversized = source
        .narrow(0, 0, plan.effective)?
        .reshape((plan.effective, 512))?;
    let key = metal_backing_key(&oversized);
    assert!(leases.attention.pin(oversized).is_err());
    assert!(
        leases
            .attention
            .batch
            .iter()
            .any(|t| metal_backing_key(t) == key)
    );
    // Candidate mutation followed by a genuine malformed append is rolled back.
    candidate.compressed[3]
        .index
        .append_many(fitting(1, 128)?, &mut leases.attention.batch)?;
    assert!(
        candidate.compressed[3]
            .index
            .append_many(fitting(1, 127)?, &mut leases.attention.batch)
            .is_err()
    );
    drop(candidate);
    d.synchronize()?; // existing error fence, not a normal-path CPU wait
    assert_eq!(committed.frontier_report()?, frontier);
    println!(
        "prefill owner evidence: {}",
        serde_json::to_string_pretty(&serde_json::json!({
            "effective_tile":plan.effective,"short_slice":1,"prior_full_interval_bytes":prior,
            "actual_unique_scratch_bytes":full_owned,"comparison_bound_bytes":profile::tile_owner_comparison_bound(plan)?,
            "actual_unique_dynamic_bytes":dynamic,"dynamic_bound_bytes":DeepSeek41Cache::predicted_dynamic_bytes(2048,true)?,
            "scratch_plus_fixed_plus_dynamic_bytes":full_owned+profile::TILE_RAW_PENDING_BYTES+dynamic,
            "admitted_scratch_plus_dynamic_bytes":plan.scratch_bound_bytes+DeepSeek41Cache::predicted_dynamic_bytes(2048,true)?,
            "oversized_source_retained_until_error_fence":true,"rollback_frontier_unchanged":true,
            "proof_scope":"generated full-tile scorer/owner interval; not a 40-layer whole-model stage proof"
        }))?
    );
    drop(leases);
    Ok(())
}
