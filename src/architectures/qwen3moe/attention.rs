//! Keep native-cache attention inputs unchanged while retaining F32 partials.

use anyhow::Result;
use candle_core::{DType, Tensor};

// The query's temporary half staging, widened Q/K/V and F32 output overlap.
// Include an explicit mask and all three vector partial/sum/max buffers, then
// double for Metal allocation rounding (also covering CPU mask upload storage).
fn workspace_bytes(geometry: &super::super::FusedSdpaGeometry, causal: bool) -> Option<usize> {
    let query = geometry
        .batch
        .checked_mul(geometry.q_heads)?
        .checked_mul(geometry.q_seq)?
        .checked_mul(geometry.qk_dim)?;
    let cache = geometry
        .batch
        .checked_mul(geometry.kv_heads)?
        .checked_mul(geometry.kv_seq)?
        .checked_mul(geometry.qk_dim)?;
    let mut bytes = query.checked_mul(10)?.checked_add(cache.checked_mul(8)?)?;
    if causal && geometry.q_seq > 1 && geometry.q_seq != geometry.kv_seq {
        bytes = bytes.checked_add(
            geometry
                .q_seq
                .checked_mul(geometry.kv_seq)?
                .checked_mul(4)?,
        )?;
    }
    if geometry.q_seq == 1 && geometry.kv_seq >= 1024 {
        // Candle 0.11 uses 32 partial blocks, with output width plus sum/max.
        let partials = geometry
            .batch
            .checked_mul(geometry.q_heads)?
            .checked_mul(32)?
            .checked_mul(geometry.qk_dim.checked_add(2)?)?
            .checked_mul(4)?;
        bytes = bytes.checked_add(partials)?;
    }
    bytes.checked_mul(2)
}

// Preallocate the exact mask length. A flat_map collector can double the
// host Vec capacity, exceeding the upload space charged in workspace_bytes.
fn causal_mask_values_exact(q_seq: usize, kv_seq: usize) -> Option<Vec<f32>> {
    if q_seq == 0 || q_seq > kv_seq {
        return None;
    }
    let mut values = vec![f32::NEG_INFINITY; q_seq.checked_mul(kv_seq)?];
    for (query, row) in values.chunks_exact_mut(kv_seq).enumerate() {
        row[..kv_seq - q_seq + query + 1].fill(0.0);
    }
    Some(values)
}

/// Restrict this path to the native F16 cache and the admitted Qwen3 MoE head
/// geometry. Other devices, cache types and geometries keep the old path.
pub(super) fn try_metal_f32_sdpa_from_native_f16(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    scale: f32,
) -> Result<Option<Tensor>> {
    if !q.device().is_metal()
        || !q.device().same_device(k.device())
        || !q.device().same_device(v.device())
        || !matches!(q.dtype(), DType::F16 | DType::F32)
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
        || [q, k, v].iter().any(|tensor| tensor.rank() != 4)
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Ok(None);
    }
    let Some(geometry) = super::super::tensor_fused_sdpa_geometry(q, k, v)? else {
        return Ok(None);
    };
    if geometry.qk_dim != 128 || geometry.v_dim != 128 {
        return Ok(None);
    }
    // Decode reads the F16 cache directly with F32 math and output: same
    // values as the widened path below, without widening or a device wait.
    #[cfg(target_os = "macos")]
    if geometry.q_seq == 1
        && let Some(output) = super::super::sdpa_f16kv_metal::try_decode(q, k, v, scale)?
    {
        return Ok(Some(output));
    }
    // Causal prompt chunks: tiled attention on the F16 cache, causal bound
    // in-kernel, no widened copies or host mask.
    #[cfg(target_os = "macos")]
    if causal
        && geometry.q_seq > 1
        && let Some(output) = super::super::sdpa_f16kv_metal::try_prefill(q, k, v, scale)?
    {
        return Ok(Some(output));
    }
    if workspace_bytes(&geometry, causal)
        .is_none_or(|bytes| bytes > super::F32_ATTENTION_WORKSPACE_BYTES)
    {
        return Ok(None);
    }
    // Preserve the current query-input rounding. Widening the F16 cache is
    // lossless. Candle's F32 SDPA avoids both the F16 unnormalized partials
    // in its two-pass vector kernel and the final F16 output conversion.
    let mut temporaries = Vec::with_capacity(5);
    let result = (|| -> Result<Tensor> {
        let half_q = q.to_dtype(DType::F16)?;
        temporaries.push(half_q.clone());
        let q = half_q.to_dtype(DType::F32)?;
        temporaries.push(q.clone());
        let k = k.to_dtype(DType::F32)?;
        temporaries.push(k.clone());
        let v = v.to_dtype(DType::F32)?;
        temporaries.push(v.clone());
        let mask = if causal && geometry.q_seq > 1 && geometry.q_seq != geometry.kv_seq {
            // Cached queries align at the bottom right. Candle's built-in
            // causal flag aligns at the top left; decode needs no mask.
            let values = causal_mask_values_exact(geometry.q_seq, geometry.kv_seq)
                .expect("validated attention geometry");
            let mask =
                Tensor::from_vec(values, (1, 1, geometry.q_seq, geometry.kv_seq), q.device())?
                    .broadcast_as((
                        geometry.batch,
                        geometry.q_heads,
                        geometry.q_seq,
                        geometry.kv_seq,
                    ))?;
            temporaries.push(mask.clone());
            Some(mask)
        } else {
            None
        };
        Ok(candle_nn::ops::sdpa(
            &q,
            &k,
            &v,
            mask.as_ref(),
            causal && mask.is_none(),
            scale,
            1.0,
        )?)
    })();
    // Bound queued storage to one call, including when an upload/dispatch
    // fails. Keep every widened tensor alive until that success/error fence.
    let synchronized = q.device().synchronize();
    drop(temporaries);
    let output = result?;
    synchronized?;
    Ok(Some(output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn native_f32_attention_mask_has_exact_capacity_and_cached_alignment() {
        for (queries, keys) in [(1, 1), (3, 5), (17, 33), (2049, 4097)] {
            let values = causal_mask_values_exact(queries, keys).unwrap();
            assert_eq!(values.len(), queries * keys);
            assert_eq!(values.capacity(), values.len());
            for (query, row) in values.chunks_exact(keys).enumerate() {
                let visible = keys - queries + query + 1;
                assert!(row[..visible].iter().all(|value| *value == 0.0));
                assert!(
                    row[visible..]
                        .iter()
                        .all(|value| *value == f32::NEG_INFINITY)
                );
            }
            if queries <= 17 {
                assert_eq!(
                    values,
                    super::super::super::causal_mask_values(queries, keys).unwrap()
                );
            }
        }
        assert!(causal_mask_values_exact(0, 1).is_none());
        assert!(causal_mask_values_exact(2, 1).is_none());
        assert!(causal_mask_values_exact(2, usize::MAX).is_none());
    }

    #[test]
    fn native_f32_attention_workspace_admits_4k_and_rejects_overflow() {
        for rows in [1, 128, 4095, 4096] {
            let geometry = super::super::super::fused_sdpa_geometry(
                [1, 32, rows, 128],
                [1, 4, 4096, 128],
                [1, 4, 4096, 128],
            )
            .unwrap();
            assert!(
                workspace_bytes(&geometry, true).unwrap()
                    <= super::super::F32_ATTENTION_WORKSPACE_BYTES
            );
        }
        let geometry = super::super::super::fused_sdpa_geometry(
            [1, 32, 4096, 128],
            [1, 4, 8192, 128],
            [1, 4, 8192, 128],
        )
        .unwrap();
        assert!(
            workspace_bytes(&geometry, true).unwrap() > super::super::F32_ATTENTION_WORKSPACE_BYTES
        );
        let geometry = super::super::super::fused_sdpa_geometry(
            [usize::MAX, 32, 1, 128],
            [usize::MAX, 4, 4096, 128],
            [usize::MAX, 4, 4096, 128],
        )
        .unwrap();
        assert_eq!(workspace_bytes(&geometry, true), None);
    }

    #[test]
    fn native_f32_attention_leaves_cpu_on_existing_path() -> Result<()> {
        let q = Tensor::zeros((1, 4, 1, 128), DType::F32, &Device::Cpu)?;
        let k = Tensor::zeros((1, 2, 8, 128), DType::F16, &Device::Cpu)?;
        assert!(try_metal_f32_sdpa_from_native_f16(&q, &k, &k, false, 0.125)?.is_none());
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn metal_device() -> Result<Option<Device>> {
        match Device::new_metal(0) {
            Ok(device) => Ok(Some(device)),
            Err(error)
                if std::env::var_os("DINF_REQUIRE_METAL_TESTS").is_none()
                    && std::env::var_os("DINF_REQUIRE_METAL").is_none() =>
            {
                eprintln!("Metal unavailable: {error}");
                Ok(None)
            }
            Err(error) => Err(error.into()),
        }
    }

    #[cfg(target_os = "macos")]
    fn values(tensor: &Tensor) -> Result<Vec<f32>> {
        Ok(tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
    }

    #[cfg(target_os = "macos")]
    fn reference(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool, scale: f32) -> Result<Vec<f32>> {
        let (batch, heads, rows, width) = q.dims4()?;
        let (_, kv_heads, keys, _) = k.dims4()?;
        // Match only the stated input precision, then perform independent
        // scalar F64 dot products, exponentials and weighted sums.
        let q = values(&q.to_device(&Device::Cpu)?.to_dtype(DType::F16)?)?;
        let k = values(&k.to_device(&Device::Cpu)?)?;
        let v = values(&v.to_device(&Device::Cpu)?)?;
        let mut output = Vec::with_capacity(batch * heads * rows * width);
        for lane in 0..batch {
            for head in 0..heads {
                let kv_head = head / (heads / kv_heads);
                for row in 0..rows {
                    let visible = if causal { keys - rows + row + 1 } else { keys };
                    let q_offset = ((lane * heads + head) * rows + row) * width;
                    let scores = (0..visible)
                        .map(|key| {
                            let k_offset = ((lane * kv_heads + kv_head) * keys + key) * width;
                            (0..width)
                                .map(|col| {
                                    f64::from(q[q_offset + col]) * f64::from(k[k_offset + col])
                                })
                                .sum::<f64>()
                                * f64::from(scale)
                        })
                        .collect::<Vec<_>>();
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let weights = scores
                        .iter()
                        .map(|score| (score - max).exp())
                        .collect::<Vec<_>>();
                    let sum = weights.iter().sum::<f64>();
                    for col in 0..width {
                        output.push(
                            (weights
                                .iter()
                                .enumerate()
                                .map(|(key, weight)| {
                                    let offset = ((lane * kv_heads + kv_head) * keys + key) * width;
                                    weight * f64::from(v[offset + col])
                                })
                                .sum::<f64>()
                                / sum) as f32,
                        );
                    }
                }
            }
        }
        Ok(output)
    }

    #[cfg(target_os = "macos")]
    fn patterned(
        shape: (usize, usize, usize, usize),
        salt: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let count = shape.0 * shape.1 * shape.2 * shape.3;
        Ok(Tensor::from_vec(
            (0..count)
                .map(|i| (((i * 17 + salt) % 101) as f32 - 50.0) / 53.0)
                .collect::<Vec<_>>(),
            shape,
            device,
        )?)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_f32_attention_keeps_large_constant_values_at_two_pass_boundary() -> Result<()> {
        let Some(device) = metal_device()? else {
            return Ok(());
        };
        let q = Tensor::zeros((1, 32, 1, 128), DType::F32, &device)?;
        for keys in [1023, 1024, 1025] {
            let k = Tensor::zeros((1, 4, keys, 128), DType::F16, &device)?;
            let v = Tensor::full(4096.0f32, (1, 4, keys, 128), &device)?.to_dtype(DType::F16)?;
            let output = try_metal_f32_sdpa_from_native_f16(&q, &k, &v, false, 0.125)?
                .expect("admitted native F16 cache");
            assert_eq!(output.dtype(), DType::F32);
            for value in values(&output)? {
                assert!(
                    value.is_finite() && (value - 4096.0).abs() <= 0.001,
                    "constant attention at KV={keys}: {value}"
                );
            }
        }
        device.synchronize()?;
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_f32_attention_matches_f64_gqa_and_cached_causal_reference() -> Result<()> {
        let Some(device) = metal_device()? else {
            return Ok(());
        };
        for (batch, heads, kv_heads, rows, keys, causal) in [
            (2, 4, 2, 5, 5, true),
            (2, 4, 2, 3, 19, true),
            (1, 32, 4, 3, 1025, true),
            (1, 4, 2, 1, 1023, false),
            (1, 4, 2, 1, 1024, false),
            (1, 4, 2, 1, 1025, false),
            (1, 4, 2, 3, 19, false),
        ] {
            // Nonzero-offset KV views exercise native cache strides too.
            let q = patterned((batch, heads, rows, 128), 3, &device)?;
            let k = patterned((batch, kv_heads, keys + 2, 128), 7, &device)?
                .to_dtype(DType::F16)?
                .narrow(2, 1, keys)?;
            let v = patterned((batch, kv_heads, keys + 2, 128), 11, &device)?
                .to_dtype(DType::F16)?
                .narrow(2, 1, keys)?;
            let scale = 1.0 / 128f32.sqrt();
            let expected = reference(&q, &k, &v, causal, scale)?;
            let actual = try_metal_f32_sdpa_from_native_f16(&q, &k, &v, causal, scale)?
                .expect("admitted geometry");
            let rounded_q = try_metal_f32_sdpa_from_native_f16(
                &q.to_dtype(DType::F16)?,
                &k,
                &v,
                causal,
                scale,
            )?
            .expect("F16 query admitted");
            let actual = values(&actual)?;
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                values(&rounded_q)?
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                "preserve the original F16 query input"
            );
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    actual.is_finite() && (actual - expected).abs() <= 2e-5,
                    "GQA/cached causal reference {rows}/{keys} index{index}: {actual} != {expected}"
                );
            }
        }
        device.synchronize()?;
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_f32_attention_rejects_other_devices_dtypes_and_geometry() -> Result<()> {
        let Some(device) = metal_device()? else {
            return Ok(());
        };
        let q = Tensor::zeros((1, 4, 1, 128), DType::F32, &device)?;
        let k = Tensor::zeros((1, 2, 8, 128), DType::F16, &device)?;
        for (q, k, v, scale) in [
            (q.to_device(&Device::Cpu)?, k.clone(), k.clone(), 0.125),
            (q.clone(), k.to_device(&Device::Cpu)?, k.clone(), 0.125),
            (q.to_dtype(DType::BF16)?, k.clone(), k.clone(), 0.125),
            (q.clone(), k.to_dtype(DType::F32)?, k.clone(), 0.125),
            (q.clone(), k.clone(), k.to_dtype(DType::F32)?, 0.125),
            (q.narrow(1, 0, 3)?, k.clone(), k.clone(), 0.125),
            (
                q.narrow(3, 0, 64)?,
                k.narrow(3, 0, 64)?,
                k.narrow(3, 0, 64)?,
                0.125,
            ),
            (q.clone(), k.clone(), k.narrow(3, 0, 64)?, 0.125),
            (q.squeeze(2)?, k.clone(), k.clone(), 0.125),
            (q.clone(), k.clone(), k.clone(), f32::NAN),
            (q.clone(), k.clone(), k.clone(), f32::INFINITY),
        ] {
            assert!(try_metal_f32_sdpa_from_native_f16(&q, &k, &v, false, scale)?.is_none());
        }
        device.synchronize()?;
        Ok(())
    }
}
