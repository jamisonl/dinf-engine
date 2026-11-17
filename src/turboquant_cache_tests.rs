use super::*;

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Storage, Tensor};
use std::sync::Arc;
use std::time::Instant;

const HEADS: usize = 2;
const HEAD_DIM: usize = 64;

fn input(device: &Device, token: usize, sequence: usize, salt: usize) -> Result<Tensor> {
    let values = (0..HEADS * sequence * HEAD_DIM)
        .map(|index| {
            let head = index / (sequence * HEAD_DIM);
            let offset = index % (sequence * HEAD_DIM);
            let position = offset / HEAD_DIM;
            let coordinate = offset % HEAD_DIM;
            let value = (token + position) * 37 + head * 19 + coordinate * 11 + salt;
            (value % 251) as f32 / 251.0 - 0.5
        })
        .collect::<Vec<_>>();
    Ok(Tensor::from_vec(
        values,
        (1, HEADS, sequence, HEAD_DIM),
        device,
    )?)
}

fn update(
    cache: &mut QuantizedKvCache,
    device: &Device,
    token: usize,
    sequence: usize,
) -> Result<(Tensor, Tensor)> {
    let k = input(device, token, sequence, 7)?;
    let v = input(device, token, sequence, 113)?;
    let output = cache.update(&k, &v)?;
    assert_eq!(output.0.dtype(), DType::F32);
    assert_eq!(output.1.dtype(), DType::F32);
    Ok(output)
}

fn buffer_pointer(tensor: &Tensor) -> usize {
    let (storage, _) = tensor.storage_and_layout();
    let Storage::Metal(storage) = &*storage else {
        panic!("expected Metal tensor storage")
    };
    let pointer = storage.buffer().contents() as usize;
    assert_ne!(
        pointer, 0,
        "Metal buffer has no CPU-visible contents pointer"
    );
    pointer
}

fn buffer_bytes(tensor: &Tensor) -> usize {
    let (storage, _) = tensor.storage_and_layout();
    let Storage::Metal(storage) = &*storage else {
        panic!("expected Metal tensor storage")
    };
    storage.buffer().length()
}

fn layout(cache: &QuantizedKvCache) -> Result<(usize, usize, usize)> {
    let state = cache
        .metal
        .as_ref()
        .context("expected a preallocated Metal cache")?;
    let (batch, heads, capacity, record_bytes) = state.k.dims4()?;
    ensure!(
        batch == 1,
        "packed K view has unexpected batch size {batch}"
    );
    ensure!(
        heads == HEADS,
        "packed K view has unexpected head count {heads}"
    );
    ensure!(
        capacity >= cache.seq_len(),
        "packed capacity is below sequence length"
    );
    ensure!(
        capacity.is_multiple_of(64),
        "packed capacity is not aligned: {capacity}"
    );
    ensure!(state.start.saturating_add(cache.seq_len()) <= capacity);
    ensure!(state.v.dims() == state.k.dims(), "packed K/V views differ");
    assert_eq!(
        state.backing.dims(),
        &[2, HEADS, capacity, record_bytes],
        "packed backing geometry changed"
    );
    assert_eq!(
        buffer_bytes(&state.backing),
        2 * HEADS * capacity * record_bytes,
        "persistent packed backing must not use a rounded allocator bucket"
    );
    Ok((buffer_pointer(&state.k), capacity, state.start))
}

#[test]
fn packed_metal_allocations_preserve_the_physical_kv8_storage_saving() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    let (heads, dim) = (8, 128);
    for capacity in [1024, 1536, 2048] {
        let mut cache = QuantizedKvCache::new(Arc::new(TurboQuantConfig::new(8, dim, 42)), heads);
        let input = Tensor::ones((1, heads, capacity, dim), DType::F32, &device)?;
        drop(cache.update_native(&input, &input)?);
        let state = cache.metal.as_ref().context("expected a native cache")?;
        let packed_bytes = buffer_bytes(&state.backing);
        let dense_bytes = 2 * heads * capacity * dim * DType::F16.size_in_bytes();
        assert_eq!(packed_bytes, 2 * heads * capacity * (dim + 4));
        assert!(
            packed_bytes * 19 < dense_bytes * 10,
            "KV8 must save at least 1.9x in backing storage"
        );
        drop(input);
        drop(cache);
        device.synchronize()?;
    }
    Ok(())
}

#[test]
fn exact_metal_backing_starts_zeroed_and_cpu_migration_stays_exact() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    let mut cache = QuantizedKvCache::new(Arc::new(TurboQuantConfig::new(8, HEAD_DIM, 42)), HEADS);
    let backing = cache.allocate_metal_backing(64, &device)?;
    assert_eq!(buffer_bytes(&backing), 2 * HEADS * 64 * (HEAD_DIM + 4));
    assert!(
        backing
            .flatten_all()?
            .to_vec1::<u8>()?
            .into_iter()
            .all(|byte| byte == 0)
    );
    drop(backing);

    update(&mut cache, &Device::Cpu, 0, 8)?;
    update(&mut cache, &device, 8, 1)?;
    let (_, capacity, start) = layout(&cache)?;
    assert_eq!((capacity, start), (64, 0));
    assert_eq!(cache.seq_len(), 9);
    Ok(())
}

#[test]
fn malformed_config_is_rejected_before_metal_cache_allocation() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    for (bits, width) in [(8, 64), (4, 128), (2, 64), (4, 32)] {
        let mut config = TurboQuantConfig::new(4, 64, 42);
        config.bits = bits;
        config.head_dim = width;
        let mut cache = QuantizedKvCache::new(Arc::new(config), 1);
        let input = Tensor::ones((1, 1, 1, width), DType::F32, &device)?;
        assert!(cache.update(&input, &input).is_err());
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.allocation_tokens(), 0);
        assert!(cache.metal.is_none());
    }
    Ok(())
}

fn assert_matches_cpu(
    gpu: &QuantizedKvCache,
    cpu: &QuantizedKvCache,
    device: &Device,
) -> Result<()> {
    let (gpu_k, gpu_v) = gpu.dequantize_all(device)?;
    let (cpu_k, cpu_v) = cpu.dequantize_all_cpu(&Device::Cpu)?;
    assert_eq!(gpu_k.dims(), cpu_k.dims());
    assert_eq!(gpu_v.dims(), cpu_v.dims());
    for (label, actual, expected) in [("K", gpu_k, cpu_k), ("V", gpu_v, cpu_v)] {
        let actual = actual.flatten_all()?.to_vec1::<f32>()?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            actual.iter().all(|value| value.is_finite())
                && expected.iter().all(|value| value.is_finite()),
            "{label} reference contains a non-finite value"
        );
        assert!(
            max_error <= 2e-3,
            "{label} Metal/CPU reconstruction differs by {max_error}"
        );
    }
    Ok(())
}

fn offset_rotation_input(
    device: &Device,
    dtype: DType,
    heads: usize,
    sequence: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let elements = heads * sequence * head_dim;
    let prefix = 11;
    let values = (0..elements)
        .map(|index| {
            let value = (index * 29 + heads * 17 + sequence * 13 + head_dim) % 257;
            value as f32 / 257.0 - 0.5
        })
        .collect::<Vec<_>>();
    let mut backing_values = vec![91.0f32; prefix];
    backing_values.extend_from_slice(&values);
    backing_values.extend([92.0f32; 7]);
    let backing =
        Tensor::from_vec(backing_values, (prefix + elements + 7,), device)?.to_dtype(dtype)?;
    let view = backing
        .narrow(0, prefix, elements)?
        .reshape((1, heads, sequence, head_dim))?;
    assert!(view.is_contiguous());
    assert!(view.layout().start_offset() > 0);
    Ok(view)
}

fn rotation_signs(device: &Device, heads: usize, head_dim: usize) -> Result<Tensor> {
    let config = TurboQuantConfig::new(8, head_dim, 711);
    Ok(Tensor::from_vec(
        config.signs.repeat(heads),
        (heads, head_dim),
        device,
    )?)
}

fn attention_input(
    device: &Device,
    heads: usize,
    sequence: usize,
    head_dim: usize,
    salt: usize,
) -> Result<Tensor> {
    let values = (0..heads * sequence * head_dim)
        .map(|index| {
            let value = (index * 37 + salt * 19 + head_dim * 11) % 251;
            value as f32 / 251.0 - 0.5
        })
        .collect::<Vec<_>>();
    Ok(Tensor::from_vec(
        values,
        (1, heads, sequence, head_dim),
        device,
    )?)
}

fn scalar_rotated_from_packed(
    packed: &Tensor,
    heads: usize,
    sequence: usize,
    head_dim: usize,
) -> Result<Vec<half::f16>> {
    let packed = packed
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<u8>()?;
    let record_bytes = head_dim + 4;
    assert_eq!(
        packed.len(),
        heads * sequence * record_bytes,
        "packed record geometry changed"
    );
    let scale = (head_dim as f32).sqrt();
    let mut expected = Vec::with_capacity(heads * sequence * head_dim);
    for head in 0..heads {
        for token in 0..sequence {
            let record = &packed[(head * sequence + token) * record_bytes
                ..(head * sequence + token + 1) * record_bytes];
            let norm_bytes: [u8; 4] = record[head_dim..head_dim + 4]
                .try_into()
                .expect("packed norm suffix has four bytes");
            let norm = f32::from_le_bytes(norm_bytes);
            for &index in &record[..head_dim] {
                let value = if norm < 1.0e-10 {
                    0.0
                } else {
                    super::centroid(8, index) * norm / scale
                };
                expected.push(half::f16::from_f32(value));
            }
        }
    }
    Ok(expected)
}

fn assert_rotated_dequant_matches_scalar(
    actual: &Tensor,
    packed: &Tensor,
    heads: usize,
    sequence: usize,
    head_dim: usize,
    label: &str,
) -> Result<()> {
    let actual = actual
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<half::f16>()?;
    let expected = scalar_rotated_from_packed(packed, heads, sequence, head_dim)?;
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label} output length changed"
    );
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label} differs at element {index}: actual={} expected={}",
            actual.to_f32(),
            expected.to_f32()
        );
    }
    Ok(())
}

fn packed_records_with_norm(
    device: &Device,
    heads: usize,
    sequence: usize,
    head_dim: usize,
    norm: f32,
) -> Result<Tensor> {
    let record_bytes = head_dim + 4;
    let mut bytes = vec![0u8; heads * sequence * record_bytes];
    for record in bytes.chunks_exact_mut(record_bytes) {
        record[head_dim..].copy_from_slice(&norm.to_le_bytes());
    }
    Ok(Tensor::from_vec(
        bytes,
        (1, heads, sequence, record_bytes),
        device,
    )?)
}

#[test]
fn rotated_dequant_vectorized_matches_scalar_for_supported_widths_and_offset_windows() -> Result<()>
{
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    const HEADS: usize = 2;
    const INITIAL_TOKENS: usize = 64;
    const EVICTED_TOKENS: usize = 7;
    for head_dim in [8, 32, 64, 128, 256, 512, 1024] {
        let config = Arc::new(TurboQuantConfig::new(8, head_dim, 711));
        let mut cache = QuantizedKvCache::new(config, HEADS);
        drop(cache.update_rotated_native(
            &attention_input(&device, HEADS, INITIAL_TOKENS, head_dim, 17)?,
            &attention_input(&device, HEADS, INITIAL_TOKENS, head_dim, 113)?,
        )?);
        cache.evict_oldest(EVICTED_TOKENS);
        let (packed_k, packed_v, capacity, start) = {
            let state = cache
                .metal
                .as_ref()
                .context("expected packed Metal cache")?;
            (
                state.k.clone(),
                state.v.clone(),
                state.k.dim(2)?,
                state.start,
            )
        };
        assert!(capacity > cache.seq_len());
        assert!(start > 0);
        assert!(start + cache.seq_len() <= capacity);
        let signs = cache.attention_rotation_signs()?;
        for window in [1, 7, 31, 33] {
            let key_window = packed_k.narrow(2, start, window)?;
            let value_window = packed_v.narrow(2, start, window)?;
            assert!(key_window.layout().start_offset() > 0);
            let (actual_k, actual_v) = crate::turboquant_metal::dequantize_kv_rotated(
                &key_window,
                &value_window,
                &signs,
                8,
            )?;
            assert_rotated_dequant_matches_scalar(
                &actual_k,
                &key_window,
                HEADS,
                window,
                head_dim,
                &format!("K head_dim={head_dim} window={window}"),
            )?;
            assert_rotated_dequant_matches_scalar(
                &actual_v,
                &value_window,
                HEADS,
                window,
                head_dim,
                &format!("V head_dim={head_dim} window={window}"),
            )?;
        }
        for tiny_norm in [0.0, 0.5e-10] {
            let tiny_key = packed_records_with_norm(&device, HEADS, 1, head_dim, tiny_norm)?;
            let tiny_value = packed_records_with_norm(&device, HEADS, 1, head_dim, tiny_norm)?;
            let (actual_k, actual_v) =
                crate::turboquant_metal::dequantize_kv_rotated(&tiny_key, &tiny_value, &signs, 8)?;
            assert_rotated_dequant_matches_scalar(
                &actual_k,
                &tiny_key,
                HEADS,
                1,
                head_dim,
                &format!("tiny K head_dim={head_dim} norm={tiny_norm}"),
            )?;
            assert_rotated_dequant_matches_scalar(
                &actual_v,
                &tiny_value,
                HEADS,
                1,
                head_dim,
                &format!("tiny V head_dim={head_dim} norm={tiny_norm}"),
            )?;
        }
    }
    Ok(())
}

#[test]
fn rotated_dequant_rejects_nonfinite_norm_and_f16_overflow() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 64;
    const SEQUENCE: usize = 7;
    let signs = rotation_signs(&device, HEADS, HEAD_DIM)?;
    let finite = packed_records_with_norm(&device, HEADS, SEQUENCE, HEAD_DIM, 1.0)?;
    let invalid_norm = packed_records_with_norm(&device, HEADS, SEQUENCE, HEAD_DIM, f32::NAN)?;
    assert!(
        crate::turboquant_metal::dequantize_kv_rotated(&invalid_norm, &finite, &signs, 8,).is_err(),
        "non-finite packed norm was accepted"
    );
    let negative_norm = packed_records_with_norm(&device, HEADS, SEQUENCE, HEAD_DIM, -1.0)?;
    assert!(
        crate::turboquant_metal::dequantize_kv_rotated(&negative_norm, &finite, &signs, 8,)
            .is_err(),
        "negative K norm was accepted"
    );
    assert!(
        crate::turboquant_metal::dequantize_kv_rotated(&finite, &negative_norm, &signs, 8,)
            .is_err(),
        "negative V norm was accepted"
    );
    let infinite_norm =
        packed_records_with_norm(&device, HEADS, SEQUENCE, HEAD_DIM, f32::INFINITY)?;
    assert!(
        crate::turboquant_metal::dequantize_kv_rotated(&infinite_norm, &finite, &signs, 8,)
            .is_err(),
        "infinite K norm was accepted"
    );
    assert!(
        crate::turboquant_metal::dequantize_kv_rotated(&finite, &infinite_norm, &signs, 8,)
            .is_err(),
        "infinite V norm was accepted"
    );
    // 1e6 remains finite in F32 after multiplication by a centroid, but the
    // rotated F16 output cannot represent the result and must fail closed.
    let overflowing_norm = packed_records_with_norm(&device, HEADS, SEQUENCE, HEAD_DIM, 1.0e6)?;
    assert!(
        crate::turboquant_metal::dequantize_kv_rotated(&overflowing_norm, &finite, &signs, 8,)
            .is_err(),
        "packed dequantization accepted an F16-overflowing norm"
    );
    Ok(())
}

#[test]
#[ignore = "bounded Metal performance probe; run explicitly"]
fn rotated_dequant_attention_microbench() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    const HEAD_DIM: usize = 128;
    const KV_HEADS: usize = 2;
    const Q_HEADS: usize = 12;
    const ITERATIONS: usize = 20;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    for sequence in [1024, 4096, 8192] {
        let mut cache =
            QuantizedKvCache::new(Arc::new(TurboQuantConfig::new(8, HEAD_DIM, 711)), KV_HEADS);
        drop(cache.update_rotated_native(
            &attention_input(&device, KV_HEADS, sequence, HEAD_DIM, 17)?,
            &attention_input(&device, KV_HEADS, sequence, HEAD_DIM, 113)?,
        )?);
        let (packed_k, packed_v) = {
            let state = cache
                .metal
                .as_ref()
                .context("expected packed Metal cache")?;
            (
                state.k.narrow(2, state.start, cache.seq_len())?,
                state.v.narrow(2, state.start, cache.seq_len())?,
            )
        };
        let signs = cache.attention_rotation_signs()?;
        let query = crate::turboquant_metal::rotate_attention_native_query(
            &attention_input(&device, Q_HEADS, 1, HEAD_DIM, 227)?,
            &signs,
        )?;
        let (warm_k, warm_v) =
            crate::turboquant_metal::dequantize_kv_rotated(&packed_k, &packed_v, &signs, 8)?;
        let _ = crate::architectures::try_fused_f16_sdpa_native(
            &query, &warm_k, &warm_v, false, scale,
        )?
        .context("native Metal attention unavailable")?;
        drop((warm_k, warm_v));
        device.synchronize()?;

        let started = Instant::now();
        let elements = crate::turboquant_metal::with_deferred_validation(|| {
            let mut elements = 0usize;
            for _ in 0..ITERATIONS {
                let (dense_k, dense_v) = crate::turboquant_metal::dequantize_kv_rotated(
                    &packed_k, &packed_v, &signs, 8,
                )?;
                let output = crate::architectures::try_fused_f16_sdpa_native(
                    &query, &dense_k, &dense_v, false, scale,
                )?
                .context("native Metal attention unavailable")?;
                elements = elements.saturating_add(output.elem_count());
                device.synchronize()?;
            }
            Ok(elements)
        })?;
        std::hint::black_box(elements);
        let seconds = started.elapsed().as_secs_f64();
        eprintln!(
            "rotated_dequant_attention_microbench sequence={sequence} tok_s={:.2}",
            ITERATIONS as f64 / seconds
        );
    }
    Ok(())
}

#[test]
fn preallocated_metal_cache_reuses_storage_until_compaction_is_needed() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    let config = Arc::new(TurboQuantConfig::new(8, HEAD_DIM, 42));
    let mut cache = QuantizedKvCache::new(config, HEADS);

    update(&mut cache, &device, 0, 8)?;
    let (initial_buffer, initial_capacity, initial_start) = layout(&cache)?;
    assert_eq!(initial_capacity, 64);
    assert_eq!(initial_start, 0);

    update(&mut cache, &device, 8, 1)?;
    let (append_buffer, append_capacity, append_start) = layout(&cache)?;
    assert_eq!(
        append_buffer, initial_buffer,
        "in-capacity append allocated a buffer"
    );
    assert_eq!(append_capacity, initial_capacity);
    assert_eq!(append_start, 0);

    cache.evict_oldest(2);
    let (evicted_buffer, evicted_capacity, evicted_start) = layout(&cache)?;
    assert_eq!(
        evicted_buffer, initial_buffer,
        "eviction copied packed storage"
    );
    assert_eq!(evicted_capacity, initial_capacity);
    assert_eq!(evicted_start, 2);

    // There is still unused tail capacity, so the next append must keep the
    // same allocation and nonzero logical start.
    update(&mut cache, &device, 9, 1)?;
    let (tail_buffer, tail_capacity, tail_start) = layout(&cache)?;
    assert_eq!(tail_buffer, initial_buffer);
    assert_eq!(tail_capacity, initial_capacity);
    assert_eq!(tail_start, evicted_start);

    // Fill the physical allocation, evict a prefix, then append past its end.
    // Compaction is required only at this point and publishes a fresh buffer.
    let mut full = QuantizedKvCache::new(Arc::new(TurboQuantConfig::new(8, HEAD_DIM, 42)), HEADS);
    update(&mut full, &device, 0, 64)?;
    let (full_buffer, full_capacity, _) = layout(&full)?;
    assert_eq!(full_capacity, 64);
    full.evict_oldest(3);
    assert_eq!(layout(&full)?.0, full_buffer);
    update(&mut full, &device, 64, 1)?;
    let (compacted_buffer, compacted_capacity, compacted_start) = layout(&full)?;
    assert_ne!(
        compacted_buffer, full_buffer,
        "compaction reused an exhausted view"
    );
    assert_eq!(compacted_capacity, full_capacity);
    assert_eq!(compacted_start, 0);
    Ok(())
}

#[test]
fn preallocated_metal_cache_rejects_failed_append_without_publishing_state() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    let config = Arc::new(TurboQuantConfig::new(8, HEAD_DIM, 42));

    // A validation failure that targets the inactive tail must not alter the
    // retained prefix, metadata, or backing identity.
    let mut cache = QuantizedKvCache::new(config.clone(), HEADS);
    let (before_k, before_v) = update(&mut cache, &device, 0, 63)?;
    let before_layout = layout(&cache)?;
    for bad_on_key in [true, false] {
        let good = Tensor::ones((1, HEADS, 1, HEAD_DIM), DType::F32, &device)?;
        let bad = Tensor::from_vec(
            vec![f32::NAN; HEADS * HEAD_DIM],
            (1, HEADS, 1, HEAD_DIM),
            &device,
        )?;
        let result = if bad_on_key {
            cache.update(&bad, &good)
        } else {
            cache.update(&good, &bad)
        };
        assert!(
            result.is_err(),
            "accepted a non-finite append on the inactive tail"
        );
        assert_eq!(cache.seq_len(), 63);
        assert_eq!(layout(&cache)?, before_layout);
        let (after_k, after_v) = cache.dequantize_all(&device)?;
        for (before, after) in [(&before_k, &after_k), (&before_v, &after_v)] {
            let before = before.flatten_all()?.to_vec1::<f32>()?;
            let after = after.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(before, after, "failed append changed the retained prefix");
        }
    }

    // A failure that would require growth must not publish the newly allocated
    // backing tensor either.
    let mut growth = QuantizedKvCache::new(config, HEADS);
    let (before_k, before_v) = update(&mut growth, &device, 0, 64)?;
    let before_layout = layout(&growth)?;
    let bad = Tensor::from_vec(
        vec![f32::INFINITY; HEADS * HEAD_DIM],
        (1, HEADS, 1, HEAD_DIM),
        &device,
    )?;
    let good = Tensor::zeros((1, HEADS, 1, HEAD_DIM), DType::F32, &device)?;
    assert!(growth.update(&bad, &good).is_err());
    assert_eq!(growth.seq_len(), 64);
    assert_eq!(layout(&growth)?, before_layout);
    let (after_k, after_v) = growth.dequantize_all(&device)?;
    for (before, after) in [(&before_k, &after_k), (&before_v, &after_v)] {
        assert_eq!(
            before.flatten_all()?.to_vec1::<f32>()?,
            after.flatten_all()?.to_vec1::<f32>()?,
            "failed growth changed the published prefix"
        );
    }
    Ok(())
}

#[test]
fn preallocated_metal_cache_compaction_and_growth_match_cpu_reference() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    let config = Arc::new(TurboQuantConfig::new(8, HEAD_DIM, 42));
    let mut gpu = QuantizedKvCache::new(config.clone(), HEADS);
    let mut cpu = QuantizedKvCache::new(config, HEADS);
    let gpu_input = input(&device, 0, 64, 7)?;
    let cpu_input = input(&Device::Cpu, 0, 64, 7)?;
    gpu.update(&gpu_input, &input(&device, 0, 64, 113)?)?;
    cpu.update(&cpu_input, &input(&Device::Cpu, 0, 64, 113)?)?;
    let before = layout(&gpu)?;
    gpu.evict_oldest(5);
    cpu.evict_oldest(5);
    assert_eq!(
        layout(&gpu)?.0,
        before.0,
        "eviction copied the backing buffer"
    );
    assert_eq!(gpu.seq_len(), cpu.seq_len());

    // The tail is exhausted after eviction, so this append compacts the
    // retained prefix into a fresh same-capacity backing allocation.
    let gpu_next = input(&device, 64, 1, 7)?;
    let gpu_next_v = input(&device, 64, 1, 113)?;
    let cpu_next = input(&Device::Cpu, 64, 1, 7)?;
    let cpu_next_v = input(&Device::Cpu, 64, 1, 113)?;
    gpu.update(&gpu_next, &gpu_next_v)?;
    cpu.update(&cpu_next, &cpu_next_v)?;
    let compacted = layout(&gpu)?;
    assert_ne!(compacted.0, before.0);
    assert_eq!(compacted.1, 64);
    assert_eq!(compacted.2, 0);
    assert_matches_cpu(&gpu, &cpu, &device)?;

    // Crossing the physical capacity grows to the aligned next allocation.
    let gpu_more = input(&device, 65, 9, 7)?;
    let gpu_more_v = input(&device, 65, 9, 113)?;
    let cpu_more = input(&Device::Cpu, 65, 9, 7)?;
    let cpu_more_v = input(&Device::Cpu, 65, 9, 113)?;
    gpu.update(&gpu_more, &gpu_more_v)?;
    cpu.update(&cpu_more, &cpu_more_v)?;
    let grown = layout(&gpu)?;
    assert!(grown.1 >= gpu.seq_len());
    assert_eq!(grown.1 % 64, 0);
    assert!(grown.1 > compacted.1);
    assert_matches_cpu(&gpu, &cpu, &device)
}

#[test]
fn preallocated_metal_cache_stress_append_evict_matches_cpu_reference() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    let config = Arc::new(TurboQuantConfig::new(8, HEAD_DIM, 711));
    let mut gpu = QuantizedKvCache::new(config.clone(), HEADS);
    let mut cpu = QuantizedKvCache::new(config, HEADS);

    for token in 0..96 {
        update(&mut gpu, &device, token, 1)?;
        update(&mut cpu, &Device::Cpu, token, 1)?;
        if token % 3 == 2 && gpu.seq_len() > 2 {
            gpu.evict_oldest(2);
            cpu.evict_oldest(2);
        }
        if token % 12 == 11 {
            assert_eq!(gpu.seq_len(), cpu.seq_len());
            assert_matches_cpu(&gpu, &cpu, &device)?;
            let (_, capacity, start) = layout(&gpu)?;
            assert!(capacity >= gpu.seq_len());
            assert!(start + gpu.seq_len() <= capacity);
        }
    }
    assert_eq!(gpu.seq_len(), cpu.seq_len());
    assert_matches_cpu(&gpu, &cpu, &device)
}

#[test]
fn native_query_rotation_matches_f16_reference_for_offset_views() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    for (head_dim, heads, sequence) in [(8, 2, 3), (64, 3, 2), (128, 4, 3), (1024, 2, 1)] {
        let signs = rotation_signs(&device, heads, head_dim)?;
        for dtype in [DType::F32, DType::F16] {
            let input = offset_rotation_input(&device, dtype, heads, sequence, head_dim)?;
            let expected = crate::turboquant_metal::rotate_attention(&input, &signs, false)?
                .to_dtype(DType::F16)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let actual = crate::turboquant_metal::rotate_attention_native_query(&input, &signs)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(
                actual, expected,
                "native query rotation differs for dtype={dtype:?} heads={heads} sequence={sequence} head_dim={head_dim}"
            );
        }
    }
    Ok(())
}

#[test]
fn native_query_rotation_rejects_non_finite_inputs_and_signs() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    let head_dim = 64;
    let heads = 2;
    let signs = rotation_signs(&device, heads, head_dim)?;
    let nan = Tensor::from_vec(
        vec![f32::NAN; heads * head_dim],
        (1, heads, 1, head_dim),
        &device,
    )?;
    assert!(
        crate::turboquant_metal::rotate_attention_native_query(&nan, &signs).is_err(),
        "native query rotation accepted a NaN input"
    );

    // The source and rotated F32 values are finite here, but the DC Walsh-
    // Hadamard coefficient is sqrt(64) * 10_000 = 80_000 and cannot be represented
    // by the native F16 query output. The fused path must reject that cast
    // rather than return an infinity.
    let all_positive_signs = Tensor::ones((heads, head_dim), DType::F32, &device)?;
    let f16_overflow = Tensor::full(10_000.0f32, (1, heads, 1, head_dim), &device)?;
    assert!(
        crate::turboquant_metal::rotate_attention_native_query(&f16_overflow, &all_positive_signs)
            .is_err(),
        "native query rotation accepted a finite F32 value that overflowed F16"
    );

    let overflow = Tensor::full(f32::MAX, (1, heads, 1, head_dim), &device)?;
    assert!(
        crate::turboquant_metal::rotate_attention_native_query(&overflow, &signs).is_err(),
        "native query rotation accepted an overflowing F32 input"
    );

    let invalid_signs = Tensor::zeros((heads, head_dim), DType::F32, &device)?;
    let finite = Tensor::ones((1, heads, 1, head_dim), DType::F32, &device)?;
    assert!(
        crate::turboquant_metal::rotate_attention_native_query(&finite, &invalid_signs).is_err(),
        "native query rotation accepted a zero sign"
    );
    Ok(())
}
