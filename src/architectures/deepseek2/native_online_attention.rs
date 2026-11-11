//! Request-wide native attention order for each absolute causal query.
//!
//! Arithmetic follows ggml_compute_forward_flash_attn_ext_f16_one_chunk
//! in the pinned llama.cpp CPU reference (MIT, Copyright (c) 2023-2025
//! The ggml authors). Prefill groups use one prefix; decode and a one-row
//! final prompt batch use eight native parts. The full request policy, not
//! the submitted chunk size or worker count, chooses each row.
//! Q/K/V must already contain the exact widened F16 values, and the F32 post-dot scale.

#[path = "native_split_attention.rs"]
mod split;

pub fn row(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    scale: f32,
) -> Result<(), &'static str> {
    let dim = q.len();
    let value_dim = 128;
    if !scale.is_finite()
        || scale <= 0.0
        || dim != 192
        || out.len() != value_dim
        || k.is_empty()
        || k.len() / dim != v.len() / value_dim
        || !v.len().is_multiple_of(value_dim)
        || !k.len().is_multiple_of(dim)
    {
        return Err("native online attention extent");
    }
    let mut sum = 0.0f32;
    let mut maximum = f32::NEG_INFINITY;
    out.fill(0.0);
    for (key, value) in k.chunks_exact(dim).zip(v.chunks_exact(value_dim)) {
        let mut score = 0.0f32;
        for (&a, &b) in key.iter().zip(q) {
            // Products of two finite F16 values are exact in F32. Keep the
            // scalar dimension order of the repaired native QK dot.
            score += a * b;
        }
        score *= scale;
        score += 0.0; // Native visible additive mask.
        let (rescale, probability) = if score > maximum {
            let old = maximum;
            maximum = score;
            let rescale = (old - maximum).exp();
            for x in out.iter_mut() {
                *x *= rescale;
            }
            (rescale, 1.0)
        } else {
            (1.0, (score - maximum).exp())
        };
        for (acc, &x) in out.iter_mut().zip(value) {
            *acc = x.mul_add(probability, *acc);
        }
        sum = sum.mul_add(rescale, probability);
    }
    let inverse = if sum == 0.0 { 0.0 } else { 1.0 / sum };
    for x in out.iter_mut() {
        *x *= inverse;
    }
    if !out.iter().all(|x| x.is_finite()) {
        return Err("non-finite native online output");
    }
    Ok(())
}

/// Deep's native F16 cache path only. Whole rows belong to one worker;
/// no sum crosses workers or changes with the size of the submitted chunk.
#[cfg(test)]
pub fn attention(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    v: &candle_core::Tensor,
    scale: f32,
) -> anyhow::Result<Option<candle_core::Tensor>> {
    attention_with_policy(q, k, v, scale, 0, None)
}

/// Native attention family from the full request and absolute query position.
/// A truncated prefix or unsupported geometry keeps the prior unsplit order.
pub fn attention_with_policy(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    v: &candle_core::Tensor,
    scale: f32,
    start_pos: usize,
    policy: Option<crate::architectures::TextArithmeticPolicy>,
) -> anyhow::Result<Option<candle_core::Tensor>> {
    use anyhow::Context;
    use candle_core::{DType, Storage, Tensor};
    if !q.device().is_cpu()
        || !q.device().same_device(k.device())
        || !q.device().same_device(v.device())
        || q.dtype() != DType::F32
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
        || [q, k, v].iter().any(|x| x.rank() != 4)
    {
        return Ok(None);
    }
    let (batch, heads, queries, dim) = q.dims4()?;
    let (kb, kh, keys, kd) = k.dims4()?;
    if batch == 0
        || heads == 0
        || queries == 0
        || dim != 192
        || kb != batch
        || kh == 0
        || !heads.is_multiple_of(kh)
        || keys < queries
        || kd != dim
        || v.dims() != [batch, kh, keys, 128]
    {
        return Ok(None);
    }
    let split_policy = policy.filter(|_| {
        batch == 1 && heads == 16 && kh == 16 && start_pos.checked_add(queries) == Some(keys)
    });
    let q = q.contiguous()?;
    let k = k.to_dtype(DType::F32)?.contiguous()?;
    let v = v.to_dtype(DType::F32)?.contiguous()?;
    let (qs, ql) = q.storage_and_layout();
    let (ks, kl) = k.storage_and_layout();
    let (vs, vl) = v.storage_and_layout();
    let values = |s: &Storage, l: &candle_core::Layout| -> anyhow::Result<(usize, usize)> {
        anyhow::ensure!(matches!(s, Storage::Cpu(_)), "Deep online CPU storage");
        l.contiguous_offsets()
            .context("Deep online contiguous span")
    };
    let (qa, qz) = values(&qs, ql)?;
    let (ka, kz) = values(&ks, kl)?;
    let (va, vz) = values(&vs, vl)?;
    let Storage::Cpu(qs) = &*qs else {
        unreachable!()
    };
    let Storage::Cpu(ks) = &*ks else {
        unreachable!()
    };
    let Storage::Cpu(vs) = &*vs else {
        unreachable!()
    };
    let qv = qs
        .as_slice::<f32>()?
        .get(qa..qz)
        .context("Deep online Q span")?;
    let kv = ks
        .as_slice::<f32>()?
        .get(ka..kz)
        .context("Deep online K span")?;
    let vv = vs
        .as_slice::<f32>()?
        .get(va..vz)
        .context("Deep online V span")?;
    let rows = batch
        .checked_mul(heads)
        .and_then(|x| x.checked_mul(queries))
        .context("Deep online rows overflow")?;
    let total = rows
        .checked_mul(128)
        .context("Deep online output overflow")?;
    let workers = candle_core::utils::get_num_threads().max(1).min(rows);
    let chunk = rows.div_ceil(workers);
    let slots: Vec<_> = (0..workers).map(|_| std::sync::Mutex::new(None)).collect();
    candle_core::cpu::kernels::par_for_each(workers, |worker| {
        let begin = worker * chunk;
        let end = (begin + chunk).min(rows);
        let result = (|| -> anyhow::Result<Vec<f32>> {
            let mut out = vec![0.0f32; end.saturating_sub(begin) * 128];
            for index in begin..end {
                let group = index / queries;
                let visible = keys - queries + index % queries + 1;
                let kv_group = (group / heads) * kh + (group % heads) / (heads / kh);
                let start = kv_group * keys * dim;
                let value_start = kv_group * keys * 128;
                let query = &qv[index * dim..(index + 1) * dim];
                let keys = &kv[start..start + visible * dim];
                let values = &vv[value_start..value_start + visible * 128];
                let output = &mut out[(index - begin) * 128..(index - begin + 1) * 128];
                // In an untruncated prefix, visible-1 is the absolute query row.
                // Only the source-proved 256-padded physical range is admitted.
                if (257..=4096).contains(&visible)
                    && split_policy.and_then(|p| p.native_q1_split_at(visible - 1)) == Some(true)
                {
                    split::split_row(
                        query,
                        keys,
                        values,
                        output,
                        scale,
                        visible.div_ceil(256) * 256,
                    )
                } else {
                    row(query, keys, values, output, scale)
                }
                .map_err(anyhow::Error::msg)?;
            }
            Ok(out)
        })();
        *slots[worker].lock().expect("new Deep online worker lock") = Some(result);
    });
    let mut output = Vec::with_capacity(total);
    for slot in slots {
        output.extend(
            slot.into_inner()
                .map_err(|_| anyhow::anyhow!("Deep online lock poisoned"))?
                .context("Deep online worker did not finish")??,
        );
    }
    anyhow::ensure!(output.len() == total, "Deep online output extent");
    Ok(Some(Tensor::from_vec(
        output,
        (batch, heads, queries, 128),
        q.device(),
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    fn bits(t: &Tensor) -> anyhow::Result<Vec<u32>> {
        Ok(t.flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .map(|x| x.to_bits())
            .collect())
    }
    #[test]
    fn deep_online_unequal_width_views_and_chunks() -> anyhow::Result<()> {
        let (b, h, q, n) = (2, 16, 5, 19);
        let query = Tensor::from_vec(
            (0..b * h * (q + 2) * 192)
                .map(|i| ((i * 19 % 113) as f32 - 56.) / 83.)
                .collect::<Vec<_>>(),
            (b, h, q + 2, 192),
            &Device::Cpu,
        )?
        .to_dtype(DType::F16)?
        .to_dtype(DType::F32)?
        .narrow(2, 1, q)?;
        let keys = Tensor::from_vec(
            (0..b * h * (n + 2) * 192)
                .map(|i| ((i * 31 % 127) as f32 - 63.) / 79.)
                .collect::<Vec<_>>(),
            (b, h, n + 2, 192),
            &Device::Cpu,
        )?
        .to_dtype(DType::F16)?
        .narrow(2, 1, n)?;
        let vals = Tensor::from_vec(
            (0..b * h * (n + 2) * 128)
                .map(|i| ((i * 53 % 131) as f32 - 65.) / 71.)
                .collect::<Vec<_>>(),
            (b, h, n + 2, 128),
            &Device::Cpu,
        )?
        .to_dtype(DType::F16)?
        .narrow(2, 1, n)?;
        assert!(!query.is_contiguous());
        assert!(!keys.is_contiguous());
        assert!(!vals.is_contiguous());
        let scale = f32::from_bits(0x3deaf30c);
        let whole = attention(&query, &keys, &vals, scale)?.unwrap();
        assert_eq!(whole.dims(), &[b, h, q, 128]);
        for chunk in [1, 2, 4] {
            let mut parts = Vec::new();
            for at in (0..q).step_by(chunk) {
                let len = chunk.min(q - at);
                parts.push(
                    attention(
                        &query.narrow(2, at, len)?,
                        &keys.narrow(2, 0, n - q + at + len)?,
                        &vals.narrow(2, 0, n - q + at + len)?,
                        scale,
                    )?
                    .unwrap(),
                );
            }
            assert_eq!(bits(&whole)?, bits(&Tensor::cat(&parts, 2)?)?);
        }
        assert_eq!(
            bits(&whole)?,
            bits(
                &attention(
                    &query.contiguous()?,
                    &keys.contiguous()?,
                    &vals.contiguous()?,
                    scale
                )?
                .unwrap()
            )?
        );
        Ok(())
    }
    #[test]
    fn deep_online_declines_other_geometry_and_cache() -> anyhow::Result<()> {
        let q = Tensor::ones((1, 16, 1, 192), DType::F32, &Device::Cpu)?;
        let k = Tensor::ones((1, 16, 3, 192), DType::F16, &Device::Cpu)?;
        let v = Tensor::ones((1, 16, 3, 128), DType::F16, &Device::Cpu)?;
        assert!(attention(&q, &k, &v.to_dtype(DType::F32)?, 1.)?.is_none());
        assert!(attention(&q, &k, &k, 1.)?.is_none());
        assert!(attention(&q.to_dtype(DType::F16)?, &k, &v, 1.)?.is_none());
        assert!(row(&[0.; 192], &[0.; 192], &[0.; 128], &mut [0.; 128], f32::NAN).is_err());
        Ok(())
    }
    #[test]
    fn deep_online_gqa_unequal_value_matches_f64() -> anyhow::Result<()> {
        let (b, h, hk, q, n, dk, dv) = (2, 4, 2, 3, 17, 192, 128);
        let query = Tensor::from_vec(
            (0..b * h * q * dk)
                .map(|i| ((i * 73 % 509) as f32 - 254.) / 811.)
                .collect::<Vec<_>>(),
            (b, h, q, dk),
            &Device::Cpu,
        )?
        .to_dtype(DType::F16)?
        .to_dtype(DType::F32)?;
        let keys = Tensor::from_vec(
            (0..b * hk * n * dk)
                .map(|i| ((i * 17 % 503) as f32 - 251.) / 769.)
                .collect::<Vec<_>>(),
            (b, hk, n, dk),
            &Device::Cpu,
        )?
        .to_dtype(DType::F16)?;
        let vals = Tensor::from_vec(
            (0..b * hk * n * dv)
                .map(|i| ((i * 31 % 499) as f32 - 249.) / 751.)
                .collect::<Vec<_>>(),
            (b, hk, n, dv),
            &Device::Cpu,
        )?
        .to_dtype(DType::F16)?;
        let scale = f32::from_bits(0x3deaf30c);
        let got = attention(&query, &keys, &vals, scale)?
            .unwrap()
            .flatten_all()?
            .to_vec1::<f32>()?;
        let qs = query.flatten_all()?.to_vec1::<f32>()?;
        let ks = keys.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let vs = vals.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        for batch in 0..b {
            for head in 0..h {
                for qi in 0..q {
                    let group = batch * hk + head / (h / hk);
                    let visible = n - q + qi + 1;
                    let queryrow = (batch * h + head) * q + qi;
                    let mut scores = Vec::new();
                    for key in 0..visible {
                        let mut score = 0f64;
                        for dim in 0..dk {
                            score += f64::from(qs[queryrow * dk + dim])
                                * f64::from(ks[(group * n + key) * dk + dim]);
                        }
                        scores.push(score * f64::from(scale));
                    }
                    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let probs = scores
                        .iter()
                        .map(|x| (x - maximum).exp())
                        .collect::<Vec<_>>();
                    let sum = probs.iter().sum::<f64>();
                    for dim in 0..dv {
                        let expected = (0..visible)
                            .map(|key| {
                                probs[key] / sum * f64::from(vs[(group * n + key) * dv + dim])
                            })
                            .sum::<f64>();
                        assert!((f64::from(got[queryrow * dv + dim]) - expected).abs() <= 2e-5);
                    }
                }
            }
        }
        Ok(())
    }
    #[test]
    fn deep_split_fixed512_offset_and_f64() -> anyhow::Result<()> {
        let n = 415;
        let h = 16;
        let dk = 192;
        let dv = 128;
        let data = |count: usize, salt: usize| {
            (0..count)
                .map(|i| half::f16::from_f32(((i * salt % 127) as f32 - 63.) / 97.).to_f32())
                .collect::<Vec<_>>()
        };
        let q =
            Tensor::from_vec(data(h * 2 * dk, 17), (1, h, 2, dk), &Device::Cpu)?.narrow(2, 1, 1)?;
        let k = Tensor::from_vec(data(h * (n + 2) * dk, 31), (1, h, n + 2, dk), &Device::Cpu)?
            .to_dtype(DType::F16)?
            .narrow(2, 1, n)?;
        let v = Tensor::from_vec(data(h * (n + 2) * dv, 43), (1, h, n + 2, dv), &Device::Cpu)?
            .to_dtype(DType::F16)?
            .narrow(2, 1, n)?;
        let scale = f32::from_bits(0x3deaf30e);
        let got = attention_with_policy(
            &q,
            &k,
            &v,
            scale,
            n - 1,
            Some(crate::architectures::TextArithmeticPolicy::deep_cpu_q8(
                crate::architectures::TextRequestContext::new(290)?,
            )),
        )?
        .unwrap();
        assert_eq!(
            bits(&got)?,
            bits(
                &attention_with_policy(
                    &q.contiguous()?,
                    &k.contiguous()?,
                    &v.contiguous()?,
                    scale,
                    n - 1,
                    Some(crate::architectures::TextArithmeticPolicy::deep_cpu_q8(
                        crate::architectures::TextRequestContext::new(290)?
                    )),
                )?
                .unwrap()
            )?
        );
        let qs = q.flatten_all()?.to_vec1::<f32>()?;
        let ks = k.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let vs = v.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let out = got.flatten_all()?.to_vec1::<f32>()?;
        for head in 0..h {
            let mut own = [0f32; 128];
            split::split_row(
                &qs[head * dk..(head + 1) * dk],
                &ks[head * n * dk..(head + 1) * n * dk],
                &vs[head * n * dv..(head + 1) * n * dv],
                &mut own,
                scale,
                512,
            )
            .unwrap();
            assert!(
                own.iter()
                    .zip(&out[head * dv..(head + 1) * dv])
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            let scores = (0..n)
                .map(|i| {
                    (0..dk)
                        .map(|d| {
                            f64::from(qs[head * dk + d]) * f64::from(ks[(head * n + i) * dk + d])
                        })
                        .sum::<f64>()
                        * f64::from(scale)
                })
                .collect::<Vec<_>>();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let probs = scores.iter().map(|x| (x - max).exp()).collect::<Vec<_>>();
            let sum = probs.iter().sum::<f64>();
            for d in 0..dv {
                let expected = (0..n)
                    .map(|i| probs[i] / sum * f64::from(vs[(head * n + i) * dv + d]))
                    .sum::<f64>();
                assert!((f64::from(out[head * dv + d]) - expected).abs() <= 2e-5);
            }
        }
        Ok(())
    }
    #[test]
    fn deep_split_declines_other_shapes_and_keeps_unsplit() -> anyhow::Result<()> {
        for (b, h, q, n) in [
            (1, 16, 2, 415),
            (2, 16, 1, 415),
            (1, 4, 1, 415),
            (1, 16, 1, 256),
            (1, 16, 1, 513),
        ] {
            let query = Tensor::ones((b, h, q, 192), DType::F32, &Device::Cpu)?;
            let key = Tensor::ones((b, h, n, 192), DType::F16, &Device::Cpu)?;
            let value = Tensor::ones((b, h, n, 128), DType::F16, &Device::Cpu)?;
            assert_eq!(
                bits(&attention(&query, &key, &value, 0.1)?.unwrap())?,
                bits(
                    &attention_with_policy(
                        &query,
                        &key,
                        &value,
                        0.1,
                        n,
                        Some(crate::architectures::TextArithmeticPolicy::deep_cpu_q8(
                            crate::architectures::TextRequestContext::new(290)?
                        ))
                    )?
                    .unwrap()
                )?
            );
        }
        assert!(
            split::split_row(
                &[0.; 191],
                &vec![0.; 257 * 192],
                &vec![0.; 257 * 128],
                &mut [0.; 128],
                0.1,
                512,
            )
            .is_err()
        );
        assert!(
            split::split_row(
                &[0.; 192],
                &vec![0.; 257 * 192],
                &vec![0.; 257 * 128 - 1],
                &mut [0.; 128],
                0.1,
                512,
            )
            .is_err()
        );
        Ok(())
    }
    #[test]
    fn deep_request_split_matches_absolute_rows_for_every_chunk() -> anyhow::Result<()> {
        use crate::architectures::{TextArithmeticPolicy, TextRequestContext};
        let make = |count: usize, salt: usize| {
            (0..count)
                .map(|i| half::f16::from_f32(((i * salt % 211) as f32 - 105.) / 313.).to_f32())
                .collect::<Vec<_>>()
        };
        // Cross threshold, tail1 and decode, then larger padded-key boundaries.
        for (prompt, start, count) in [
            (257, 248, 24),
            (258, 248, 24),
            (265, 248, 24),
            (290, 280, 16),
            (513, 508, 8),
            (1793, 1788, 8),
            (3298, 3294, 8),
            (4093, 4088, 8),
        ] {
            let n = start + count;
            let policy = TextArithmeticPolicy::deep_cpu_q8(TextRequestContext::new(prompt)?);
            let q = Tensor::from_vec(
                make(16 * (count + 2) * 192, 17),
                (1, 16, count + 2, 192),
                &Device::Cpu,
            )?
            .narrow(2, 1, count)?;
            let k = Tensor::from_vec(
                make(16 * (n + 2) * 192, 31),
                (1, 16, n + 2, 192),
                &Device::Cpu,
            )?
            .to_dtype(DType::F16)?
            .narrow(2, 1, n)?;
            let v = Tensor::from_vec(
                make(16 * (n + 2) * 128, 43),
                (1, 16, n + 2, 128),
                &Device::Cpu,
            )?
            .to_dtype(DType::F16)?
            .narrow(2, 1, n)?;
            let scale = f32::from_bits(0x3deaf30e);
            let whole = attention_with_policy(&q, &k, &v, scale, start, Some(policy))?.unwrap();
            for chunk in [1, 5, 7, 32, 128] {
                let mut parts = Vec::new();
                for at in (0..count).step_by(chunk) {
                    let len = chunk.min(count - at);
                    parts.push(
                        attention_with_policy(
                            &q.narrow(2, at, len)?,
                            &k.narrow(2, 0, start + at + len)?,
                            &v.narrow(2, 0, start + at + len)?,
                            scale,
                            start + at,
                            Some(policy),
                        )?
                        .unwrap(),
                    );
                }
                assert_eq!(
                    bits(&whole)?,
                    bits(&Tensor::cat(&parts, 2)?)?,
                    "prompt={prompt} start={start} chunk={chunk}"
                );
            }
            // Each absolute row independently matches the intended native family.
            let qs = q.flatten_all()?.to_vec1::<f32>()?;
            let ks = k.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let vs = v.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let got = whole.flatten_all()?.to_vec1::<f32>()?;
            for at in 0..count {
                let visible = start + at + 1;
                let mut expected = [0f32; 128];
                if policy.native_q1_split_at(start + at) == Some(true) {
                    split::split_row(
                        &qs[at * 192..(at + 1) * 192],
                        &ks[..visible * 192],
                        &vs[..visible * 128],
                        &mut expected,
                        scale,
                        visible.div_ceil(256) * 256,
                    )
                    .unwrap();
                } else {
                    row(
                        &qs[at * 192..(at + 1) * 192],
                        &ks[..visible * 192],
                        &vs[..visible * 128],
                        &mut expected,
                        scale,
                    )
                    .unwrap();
                }
                assert!(
                    expected
                        .iter()
                        .zip(&got[at * 128..(at + 1) * 128])
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
            // An explicit start that does not match the complete cache declines
            // split without changing the old unsplit result.
            assert_eq!(
                bits(&attention(&q, &k, &v, scale)?.unwrap())?,
                bits(&attention_with_policy(&q, &k, &v, scale, start + 1, Some(policy))?.unwrap())?
            );
        }
        Ok(())
    }
}
