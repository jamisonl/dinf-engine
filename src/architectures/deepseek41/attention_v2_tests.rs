//! Causal/cache fixtures for the test-only AttentionV2 prefill path.
//! Synthetic data only: no model loading, subprocesses or environment mutation.
use super::cache::{Compressed, Rows};
use super::*;

fn flat(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.flatten_all()?.to_vec1::<f32>()?)
}

#[test]
fn attention_v2_pooling_odd_carry_and_input_boundaries() -> Result<()> {
    let d = &Device::Cpu;
    for count in [1, 2, 25, 31, 32, 128] {
        for base in [0usize, 1] {
            let mut cache = Compressed::default();
            if base == 1 {
                cache.pending = Some((
                    Tensor::full(0f32, (1, 2), d)?,
                    Tensor::zeros((1, 2), DType::F32, d)?,
                ));
            }
            let checkpoint = cache.clone();
            let values: Vec<f32> = (base..base + count)
                .flat_map(|p| [p as f32, 2. * p as f32])
                .collect();
            let kv = Tensor::from_vec(values, (count, 2), d)?;
            let gate = Tensor::zeros((count, 2), DType::F32, d)?;
            let mut pins = Vec::new();
            let (rows, positions) =
                attention::pool_projected_tile(&mut cache, &kv, Some(&gate), base, &mut pins)?;
            let expected_positions: Vec<_> = (0..base + count)
                .step_by(2)
                .filter(|p| p + 1 < base + count)
                .collect();
            assert_eq!(positions, expected_positions, "base{base}/input{count}");
            for (row, &position) in rows.iter().zip(&positions) {
                // Equal gates give an arithmetic mean, independent of pooling implementation.
                assert_eq!(
                    flat(row)?,
                    [position as f32 + 0.5, 2. * position as f32 + 1.]
                );
            }
            assert_eq!(cache.pending.is_some(), (base + count) % 2 == 1);
            if let Some((last, _)) = &cache.pending {
                let pos = (base + count - 1) as f32;
                assert_eq!(flat(last)?, [pos, 2. * pos]);
            }
            // A fork/rollback does not consume its parent's incoming pair.
            assert_eq!(checkpoint.pending.is_some(), base == 1);
            if let Some((old, _)) = &checkpoint.pending {
                assert_eq!(flat(old)?, [0., 0.]);
            }
            let (single, single_positions) = attention::pool_projected_tile(
                &mut Compressed::default(),
                &kv,
                None,
                base,
                &mut Vec::new(),
            )?;
            assert_eq!(single_positions, (base..base + count).collect::<Vec<_>>());
            assert_eq!(flat(&Tensor::cat(&single, 0)?)?, flat(&kv)?);
        }
    }
    // A bad gate must not consume or overwrite a valid incoming odd carry.
    let carry = Tensor::full(7f32, (1, 2), d)?;
    let mut cache = Compressed {
        pending: Some((carry.clone(), carry)),
        ..Compressed::default()
    };
    assert!(
        attention::pool_projected_tile(
            &mut cache,
            &Tensor::zeros((2, 2), DType::F32, d)?,
            Some(&Tensor::zeros((2, 3), DType::F32, d)?),
            1,
            &mut Vec::new()
        )
        .is_err()
    );
    assert_eq!(flat(&cache.pending.unwrap().0)?, [7., 7.]);
    Ok(())
}

fn bulk_pages(d: &Device) -> Result<()> {
    for geometry in [256, 2048] {
        for start in [geometry - 1, geometry, geometry + 1] {
            let data = Tensor::from_vec(
                (0..start * 2).map(|i| i as f32).collect::<Vec<_>>(),
                (start, 2),
                d,
            )?;
            let chunks = (0..start)
                .step_by(geometry)
                .map(|s| data.narrow(0, s, geometry.min(start - s)))
                .collect::<candle_core::Result<Vec<_>>>()?;
            let original = Rows::fixture_pages(geometry, chunks, start);
            let mut candidate = original.clone();
            let mut pins = Vec::new();
            candidate.append_many(Tensor::full(999f32, (32, 2), d)?, &mut pins)?;
            let held = Tensor::cat(&original.chunks, 0)?;
            assert_eq!(flat(&held)?, flat(&data)?);
            let joined = flat(&Tensor::cat(&candidate.chunks, 0)?)?;
            assert_eq!(&joined[..start * 2], flat(&data)?.as_slice());
            assert_eq!(&joined[start * 2..], &[999f32; 64]);
            assert_eq!(candidate.len(), start + 32);
            assert_eq!(original.len(), start);
            for page in candidate.chunks.iter().take(candidate.chunks.len() - 1) {
                assert_eq!(page.dim(0)?, geometry);
            }
            let frontier = candidate.len();
            assert!(
                candidate
                    .append_many(Tensor::zeros((2, 3), DType::F32, d)?, &mut pins)
                    .is_err()
            );
            assert_eq!(candidate.len(), frontier);
            assert_eq!(flat(&Tensor::cat(&candidate.chunks, 0)?)?, joined);
        }
    }
    Ok(())
}

#[test]
fn attention_v2_bulk_publication_keeps_forks_and_page_frontiers() -> Result<()> {
    bulk_pages(&Device::Cpu)
}

fn fused_causality(d: &Device) -> Result<()> {
    for n in [2047usize, 2048, 2049] {
        // Nonzero offsets in Q, weights and pages exercise the full storage span.
        let q = Tensor::ones((3, 32, 128), DType::F32, d)?.narrow(0, 1, 2)?;
        let w = Tensor::ones((3, 32), DType::F32, d)?.narrow(0, 1, 2)?;
        let values: Vec<f32> = (0..n + 1)
            .flat_map(|i| std::iter::repeat_n((i % 7) as f32 - 3., 128))
            .collect();
        let data = Tensor::from_vec(values, (n + 1, 128), d)?;
        let pages: Vec<_> = (0..n)
            .step_by(2048)
            .map(|s| data.narrow(0, s + 1, 2048.min(n - s)))
            .collect::<candle_core::Result<_>>()?;
        let visible = [n - 1, n];
        let mut pins = Vec::new();
        let output = ops::index_scores_many(&q, &w, &pages, &visible, &mut pins)?;
        let actual = output.to_vec2::<f32>()?;
        for r in 0..2 {
            for (k, &value) in actual[r][..n].iter().enumerate() {
                let expected = if k < visible[r] {
                    64. * (((k + 1) % 7) as f32 - 3.).max(0.)
                } else {
                    f32::NEG_INFINITY
                };
                assert_eq!(value, expected, "row{r}/key{k}/N{n}");
            }
        }
        // Every future key is NaN. Earlier scoring/topk is strictly unaffected.
        let mut poisoned = pages.clone();
        let last = poisoned.pop().unwrap();
        let count = last.dim(0)?;
        let poison = Tensor::full(f32::NAN, (1, 128), d)?;
        poisoned.push(if count == 1 {
            poison
        } else {
            Tensor::cat(&[&last.narrow(0, 0, count - 1)?, &poison], 0)?
        });
        let other = ops::index_scores_many(&q, &w, &poisoned, &visible, &mut pins)?;
        let a = output.narrow(0, 0, 1)?.flatten_all()?.narrow(0, 0, n - 1)?;
        let b = other.narrow(0, 0, 1)?.flatten_all()?.narrow(0, 0, n - 1)?;
        assert_eq!(flat(&a)?, flat(&b)?);
        assert_eq!(flat(&ops::topk(&a, 512)?)?, flat(&ops::topk(&b, 512)?)?);
    }
    Ok(())
}

#[test]
fn attention_v2_cpu_multiquery_future_poison_and_page_edges() -> Result<()> {
    fused_causality(&Device::Cpu)
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires a Metal device; run serially (causal/page/allocation checks)"]
fn attention_v2_metal_multiquery_future_poison_and_page_edges() -> Result<()> {
    let d = Device::new_metal(0)?;
    bulk_pages(&d)?;
    fused_causality(&d)?;
    d.synchronize()?;
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires a Metal device; numeric bounds against an F64 oracle"]
fn attention_v2_metal_score_f64_error_gate() -> Result<()> {
    let d = Device::new_metal(0)?;
    let (t, n) = (3usize, 257usize);
    let q: Vec<f32> = (0..t * 4096)
        .map(|i| ((i * 37 % 127) as f32 - 63.) / 31.)
        .collect();
    let w: Vec<f32> = (0..t * 32)
        .map(|i| ((i * 11 % 31) as f32 - 15.) / 16.)
        .collect();
    let k: Vec<f32> = (0..n * 128)
        .map(|i| ((i * 19 % 131) as f32 - 65.) / 33.)
        .collect();
    let queries = Tensor::from_vec(q.clone(), (t, 32, 128), &d)?;
    let weights = Tensor::from_vec(w.clone(), (t, 32), &d)?;
    let page = Tensor::from_vec(k.clone(), (n, 128), &d)?;
    let visible = [1, 128, n];
    let mut pins = Vec::new();
    let output = ops::index_scores_many(
        &queries,
        &weights,
        std::slice::from_ref(&page),
        &visible,
        &mut pins,
    )?;
    let mut leases = MoeLeases::default();
    leases.attention.batch = pins;
    leases.attention.index_pages.push(page);
    // Remove every caller-local GPU input/output owner before the normal
    // readback fence; only the interval leases own these scoring resources.
    drop(queries);
    drop(weights);
    drop(output);
    let actual = leases
        .attention
        .batch
        .last()
        .context("missing score output owner")?
        .to_vec2::<f32>()?;
    for row in 0..t {
        for key in 0..visible[row] {
            let contributions: Vec<f64> = (0..32)
                .map(|head| {
                    let dot: f64 = q[(row * 32 + head) * 128..(row * 32 + head + 1) * 128]
                        .iter()
                        .zip(&k[key * 128..(key + 1) * 128])
                        .map(|(&a, &b)| a as f64 * b as f64)
                        .sum();
                    dot.max(0.) * w[row * 32 + head] as f64 / 64.
                })
                .collect();
            let expected: f64 = contributions.iter().sum();
            let scale: f64 = contributions.iter().map(|v| v.abs()).sum();
            let error = (actual[row][key] as f64 - expected).abs();
            assert!(
                actual[row][key].is_finite() && error <= 2e-5 + 2e-4 * scale,
                "row{row}/key{key}: error{error}, scale{scale}"
            );
        }
        assert!(
            actual[row][visible[row]..]
                .iter()
                .all(|&x| x == f32::NEG_INFINITY)
        );
    }
    drop(leases);
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires a Metal device; run serially (bulk allocation and preflight-error rollback)"]
fn attention_v2_metal_bulk_rounded_ledger_and_error_rollback() -> Result<()> {
    let d = Device::new_metal(0)?;
    let _selector = AttentionV2Guard::set(true);
    for start in [2047usize, 2048, 2049] {
        let committed = DeepSeek41Cache::seeded_native_fixture(start, &d)?;
        let before = committed.frontier_report()?;
        let mut candidate = committed.clone();
        let mut leases = MoeLeases::default();
        let end = start + 32;
        for (owner, compressed) in candidate.compressed.iter_mut().enumerate() {
            let ratio = if owner == 3 { 1 } else { 2 };
            let count = end / ratio - start / ratio;
            compressed.kv.append_many(
                Tensor::full(17f32, (count, 512), &d)?,
                &mut leases.attention.batch,
            )?;
            compressed.index.append_many(
                Tensor::full(19f32, (count, 128), &d)?,
                &mut leases.attention.batch,
            )?;
            assert_eq!(compressed.kv.len(), end / ratio);
            assert_eq!(compressed.index.len(), end / ratio);
            leases
                .attention
                .batch_cache
                .extend(compressed.kv.chunks.iter().cloned());
            leases
                .attention
                .batch_cache
                .extend(compressed.index.chunks.iter().cloned());
        }
        let dynamic = cache::retained_bytes(leases.attention.batch_cache.iter());
        assert!(dynamic <= DeepSeek41Cache::predicted_dynamic_bytes(end, true)?);
        assert!(
            candidate.allocation_tokens() <= DeepSeek41Cache::predicted_allocation_tokens(end, &d)?
        );
        let extra = tile_owner_bytes(&[], &leases);
        assert!(extra <= profile::attention_v2_bytes(32, end));
        // Genuine failure after earlier GPU cache-copy work, not a fabricated
        // Err or a mutated old assertion. Partial candidate is never committed.
        assert!(
            candidate.compressed[3]
                .index
                .append_many(
                    Tensor::zeros((1, 127), DType::F32, &d)?,
                    &mut leases.attention.batch
                )
                .is_err()
        );
        drop(candidate);
        d.synchronize()?; // Stand-in for the existing outer final/error fence.
        assert_eq!(committed.frontier_report()?, before);
        for owner in &committed.compressed {
            assert!(flat(&owner.index.chunks[0])?.iter().all(|&v| v == 1.));
            assert!(flat(&owner.kv.chunks[0])?.iter().all(|&v| v == 1.));
        }
        drop(leases);
    }
    Ok(())
}

#[test]
fn attention_v2_candidate_blocks_clip_before_masks_and_keep_row_source() -> Result<()> {
    // Different rows cross both the 16384 threshold and the partial8 block.
    // A deliberately enormous future score cannot enter the earlier row's mask.
    for n in [16384usize, 16385, 16391, 16392, 16393] {
        let scores = Tensor::from_vec(
            (0..n + 8)
                .map(|i| if i >= n { 1e30 } else { i as f32 })
                .collect::<Vec<_>>(),
            n + 8,
            &Device::Cpu,
        )?;
        let clipped = scores.narrow(0, 0, n)?;
        let mut row = Selection::default();
        let ids = attention::select_scores(&clipped, 20, &mut row)?;
        assert_eq!(
            flat(&ids)?,
            (n - 512..n).rev().map(|i| i as f32).collect::<Vec<_>>()
        );
        if n > 16384 {
            let mask = flat(row.candidate_mask.as_ref().unwrap())?;
            assert_eq!(mask.len(), n.div_ceil(8));
            assert_eq!(*mask.last().unwrap(), 0.); // Newest partial block stays pinned.
            // Later refresh owns its score projection, but consumes THIS row's
            // layer20 mask: boost an excluded old block and it must stay absent.
            let excluded = mask.iter().position(|v| *v == f32::NEG_INFINITY).unwrap();
            let later: Vec<f32> = (0..n)
                .map(|i| if i / 8 == excluded { 1e20 } else { i as f32 })
                .collect();
            for layer in [24, 28, 32, 36] {
                let selected = attention::select_scores(
                    &Tensor::from_vec(later.clone(), n, &Device::Cpu)?,
                    layer,
                    &mut row,
                )?;
                assert!(
                    flat(&selected)?
                        .iter()
                        .all(|&id| id as usize / 8 != excluded)
                );
            }
        } else {
            assert!(row.candidate_mask.is_none());
        }
    }
    // Wide tied scores retain stable ID ordering across the new score carrier.
    let mut row = Selection::default();
    let ties = Tensor::ones(16385, DType::F32, &Device::Cpu)?;
    assert_eq!(
        flat(&attention::select_scores(&ties, 20, &mut row)?)?,
        (0..512).map(|i| i as f32).collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn attention_v2_invalid_preflight_and_unique_backing_ledger() -> Result<()> {
    let d = &Device::Cpu;
    let q = Tensor::ones((2, 32, 128), DType::F32, d)?;
    let w = Tensor::ones((2, 32), DType::F32, d)?;
    let page = Tensor::ones((5, 128), DType::F32, d)?;
    let bad = Tensor::ones((1, 127), DType::F32, d)?;
    let mut pins = Vec::new();
    assert!(ops::index_scores_many(&q, &w, &[page.clone(), bad], &[5, 6], &mut pins).is_err());
    assert!(pins.is_empty());
    assert!(
        ops::index_scores_many(&q, &w, std::slice::from_ref(&page), &[5, 6], &mut pins).is_err()
    );
    assert!(pins.is_empty());
    let strided = Tensor::ones((2, 128, 32), DType::F32, d)?.transpose(1, 2)?;
    assert!(
        ops::index_scores_many(
            &strided,
            &w,
            std::slice::from_ref(&page),
            &[0, 5],
            &mut pins
        )
        .is_err()
    );
    assert!(pins.is_empty());
    let zero = ops::index_scores_many(&q, &w, std::slice::from_ref(&page), &[0, 5], &mut pins)?
        .to_vec2::<f32>()?;
    assert_eq!(zero[0], [f32::NEG_INFINITY; 5]);
    assert_eq!(zero[1], [64f32; 5]);
    let mut leases = MoeLeases::default();
    leases.attention.batch_cache.push(page.clone());
    leases.attention.batch.extend([page.clone(), page]);
    assert_eq!(tile_owner_bytes(&[], &leases), 0); // Dynamic backing charged once, elsewhere.
    leases
        .attention
        .batch
        .extend([q.clone(), q.clone(), w.clone()]);
    assert_eq!(
        tile_owner_bytes(&[], &leases),
        cache::retained_bytes([&q, &w].into_iter())
    );
    let plan_with_limit =
        || profile::prefill_tile_plan_with_limit(32, 64852, 128, FIXED_CACHE_BYTES);
    let old = plan_with_limit()?;
    {
        let _candidate = AttentionV2Guard::set(true);
        let plan = plan_with_limit()?;
        assert!(plan.effective <= old.effective);
        assert!(plan.scratch_bound_bytes <= FIXED_CACHE_BYTES);
        assert!(profile::attention_v2_bytes(plan.effective, plan.end) > 0);
        // The full admitted plan, not a shortlastslice length, owns the bound.
        assert!(
            profile::prefill_tile_scratch(plan.effective + 1, plan.end) > plan.scratch_bound_bytes
                || plan.effective == 32
        );
        let _off = AttentionV2Guard::set(false);
        assert_eq!(profile::attention_v2_bytes(32, plan.end), 0);
    }
    assert_eq!(plan_with_limit()?, old);
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn attention_v2_metal_malformed_storage_preflight_acquires_no_encoder() -> Result<()> {
    let d = Device::new_metal(0)?;
    let other = Device::new_metal(0)?;
    let q = Tensor::ones((2, 32, 128), DType::F32, &d)?;
    let w = Tensor::ones((2, 32), DType::F32, &d)?;
    let page = Tensor::ones((4, 128), DType::F32, &d)?;
    let short = d
        .as_metal_device()?
        .new_private_buffer(1, DType::F32, "score-short-fixture")?;
    let malformed = Tensor::from((
        candle_core::Storage::Metal(candle_core::MetalStorage::new(
            short,
            d.as_metal_device()?.clone(),
            8192,
            DType::F32,
        )),
        candle_core::Shape::from((2, 32, 128)),
    ));
    let before = ops::INDEX_MANY_ENCODERS.with(|v| v.get());
    let mut pins = Vec::new();
    assert!(
        ops::index_scores_many(
            &malformed,
            &w,
            std::slice::from_ref(&page),
            &[3, 4],
            &mut pins
        )
        .is_err()
    );
    let other_w = Tensor::ones((2, 32), DType::F32, &other)?;
    let other_page = Tensor::ones((4, 128), DType::F32, &other)?;
    assert!(
        ops::index_scores_many(
            &q,
            &other_w,
            std::slice::from_ref(&page),
            &[3, 4],
            &mut pins
        )
        .is_err()
    );
    assert!(ops::index_scores_many(&q, &w, &[other_page], &[3, 4], &mut pins).is_err());
    assert!(pins.is_empty());
    assert_eq!(ops::INDEX_MANY_ENCODERS.with(|v| v.get()), before);
    d.synchronize()?;
    other.synchronize()?;
    Ok(())
}
