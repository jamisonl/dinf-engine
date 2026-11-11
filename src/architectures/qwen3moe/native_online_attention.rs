//! One fixed online attention order for every absolute causal query.
//!
//! Arithmetic follows ggml_compute_forward_flash_attn_ext_f16_one_chunk
//! in the pinned llama.cpp CPU reference (MIT, Copyright (c) 2023-2025
//! The ggml authors). This port uses one complete visible prefix per row;
//! it never changes the reduction order with prompt chunk or worker count.
//! Q/K/V must already contain the exact widened F16 values, and the F32 post-dot scale.

#[path = "native_split_attention.rs"]
mod native_split_attention;

pub fn row(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    scale: f32,
) -> Result<(), &'static str> {
    let dim = q.len();
    if !scale.is_finite()
        || scale <= 0.0
        || dim != 128
        || out.len() != dim
        || k.is_empty()
        || k.len() != v.len()
        || !k.len().is_multiple_of(dim)
    {
        return Err("native online attention extent");
    }
    let mut sum = 0.0f32;
    let mut maximum = f32::NEG_INFINITY;
    out.fill(0.0);
    for (key, value) in k.chunks_exact(dim).zip(v.chunks_exact(dim)) {
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

/// Qwen's native F16 cache path only. Whole rows belong to one worker;
/// no sum crosses workers or changes with the size of the submitted chunk.
pub fn attention(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    v: &candle_core::Tensor,
    scale: f32,
) -> anyhow::Result<Option<candle_core::Tensor>> {
    attention_rows(q, k, v, scale, 0, None)
}

/// Apply the request-wide native attention family to each absolute row.
pub fn attention_rows(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    v: &candle_core::Tensor,
    scale: f32,
    start: usize,
    arithmetic: Option<crate::architectures::TextArithmeticPolicy>,
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
        || dim != 128
        || kb != batch
        || kh == 0
        || !heads.is_multiple_of(kh)
        || keys < queries
        || kd != dim
        || v.dims() != k.dims()
    {
        return Ok(None);
    }
    let phase = arithmetic.filter(|p| {
        p.native_q1_split_at(start).is_some()
            && batch == 1
            && heads == 32
            && kh == 4
            && start.checked_add(queries) == Some(keys)
            && keys <= 4096
    });
    let q = q.contiguous()?;
    let k = k.to_dtype(DType::F32)?.contiguous()?;
    let v = v.to_dtype(DType::F32)?.contiguous()?;
    let (qs, ql) = q.storage_and_layout();
    let (ks, kl) = k.storage_and_layout();
    let (vs, vl) = v.storage_and_layout();
    let values = |s: &Storage, l: &candle_core::Layout| -> anyhow::Result<(usize, usize)> {
        anyhow::ensure!(matches!(s, Storage::Cpu(_)), "Qwen online CPU storage");
        l.contiguous_offsets()
            .context("Qwen online contiguous span")
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
        .context("Qwen online Q span")?;
    let kv = ks
        .as_slice::<f32>()?
        .get(ka..kz)
        .context("Qwen online K span")?;
    let vv = vs
        .as_slice::<f32>()?
        .get(va..vz)
        .context("Qwen online V span")?;
    let expected_q = batch
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(queries))
        .and_then(|n| n.checked_mul(dim));
    let expected_kv = batch
        .checked_mul(kh)
        .and_then(|n| n.checked_mul(keys))
        .and_then(|n| n.checked_mul(dim));
    anyhow::ensure!(
        expected_q == Some(qv.len())
            && expected_kv == Some(kv.len())
            && expected_kv == Some(vv.len()),
        "Qwen online logical extents"
    );
    let rows = batch
        .checked_mul(heads)
        .and_then(|x| x.checked_mul(queries))
        .context("Qwen online rows overflow")?;
    let total = rows
        .checked_mul(dim)
        .context("Qwen online output overflow")?;
    let workers = candle_core::utils::get_num_threads().max(1).min(rows);
    let chunk = rows.div_ceil(workers);
    let slots: Vec<_> = (0..workers).map(|_| std::sync::Mutex::new(None)).collect();
    candle_core::cpu::kernels::par_for_each(workers, |worker| {
        let begin = worker * chunk;
        let end = (begin + chunk).min(rows);
        let result = (|| -> anyhow::Result<Vec<f32>> {
            let mut out = vec![0.0f32; end.saturating_sub(begin) * dim];
            for index in begin..end {
                let group = index / queries;
                let visible = keys - queries + index % queries + 1;
                let kv_group = (group / heads) * kh + (group % heads) / (heads / kh);
                let start = kv_group * keys * dim;
                let query = &qv[index * dim..(index + 1) * dim];
                let key = &kv[start..start + visible * dim];
                let value = &vv[start..start + visible * dim];
                let output = &mut out[(index - begin) * dim..(index - begin + 1) * dim];
                if phase.is_some_and(|p| p.native_q1_split_at(visible - 1) == Some(true)) {
                    native_split_attention::split_row(
                        query,
                        key,
                        value,
                        output,
                        scale,
                        visible.div_ceil(256) * 256,
                    )
                } else {
                    row(query, key, value, output, scale)
                }
                .map_err(anyhow::Error::msg)?;
            }
            Ok(out)
        })();
        *slots[worker].lock().expect("new Qwen online worker lock") = Some(result);
    });
    let mut output = Vec::with_capacity(total);
    for slot in slots {
        output.extend(
            slot.into_inner()
                .map_err(|_| anyhow::anyhow!("Qwen online lock poisoned"))?
                .context("Qwen online worker did not finish")??,
        );
    }
    anyhow::ensure!(output.len() == total, "Qwen online output extent");
    Ok(Some(Tensor::from_vec(output, q.shape(), q.device())?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    #[test]
    fn online_gqa_batches_and_f64() -> anyhow::Result<()> {
        let (b, h, hk, n, m, d) = (2, 8, 1, 17, 3, 128);
        let data = |len: usize, s: usize| {
            (0..len)
                .map(|i| {
                    half::f16::from_f32(((i * 139 + s * 431) % 1009) as f32 / 719. - 0.7).to_f32()
                })
                .collect::<Vec<_>>()
        };
        let qv = data(b * h * m * d, 1);
        let kv = data(b * hk * n * d, 2);
        let vv = data(b * hk * n * d, 3);
        let q = Tensor::from_vec(qv.clone(), (b, h, m, d), &Device::Cpu)?;
        let k = Tensor::from_vec(kv.clone(), (b, hk, n, d), &Device::Cpu)?.to_dtype(DType::F16)?;
        let v = Tensor::from_vec(vv.clone(), (b, hk, n, d), &Device::Cpu)?.to_dtype(DType::F16)?;
        let scale = 1f32 / (d as f32).sqrt();
        let output = attention(&q, &k, &v, scale)?
            .unwrap()
            .flatten_all()?
            .to_vec1::<f32>()?;
        for batch in 0..b {
            for head in 0..h {
                for query in 0..m {
                    let visible = n - m + query + 1;
                    let qi = ((batch * h + head) * m + query) * d;
                    let ki = (batch * hk + head / (h / hk)) * n * d;
                    let mut scores = (0..visible)
                        .map(|key| {
                            (0..d)
                                .map(|i| f64::from(qv[qi + i]) * f64::from(kv[ki + key * d + i]))
                                .sum::<f64>()
                                * f64::from(scale)
                        })
                        .collect::<Vec<_>>();
                    let mx = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    for s in &mut scores {
                        *s = (*s - mx).exp();
                    }
                    let sum = scores.iter().sum::<f64>();
                    for i in 0..d {
                        let expected = scores
                            .iter()
                            .enumerate()
                            .map(|(key, p)| p / sum * f64::from(vv[ki + key * d + i]))
                            .sum::<f64>();
                        assert!((f64::from(output[qi + i]) - expected).abs() <= 2e-5);
                    }
                }
            }
        }
        Ok(())
    }
    #[test]
    fn online_row_rejects_invalid_inputs() {
        let q = vec![0f32; 128];
        let k = vec![0f32; 128];
        let mut y = vec![0f32; 128];
        assert!(row(&q, &k, &k, &mut y, f32::NAN).is_err());
        assert!(row(&q, &k, &k, &mut y, 0.).is_err());
        assert!(row(&q, &k, &k[..127], &mut y, 1.).is_err());
    }
}

#[cfg(test)]
mod request_attention_tests {
    use super::*;
    use crate::architectures::{TextArithmeticPolicy, TextRequestContext};
    use candle_core::{DType, Device, Tensor};

    fn values(n: usize, salt: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                half::f16::from_f32(((i * 139 + salt * 431) % 1009) as f32 / 719.0 - 0.7).to_f32()
            })
            .collect()
    }
    fn bits(t: &Tensor) -> anyhow::Result<Vec<u32>> {
        Ok(t.flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }

    #[test]
    fn request_split_views_chunks_and_absolute_phase_match() -> anyhow::Result<()> {
        let (h, hk, m, d, start) = (32, 4, 24, 128, 250);
        let keys = start + m;
        let q = Tensor::from_vec(values(2 * h * m * d, 1), (2, h, m, d), &Device::Cpu)?
            .narrow(0, 1, 1)?;
        let k = Tensor::from_vec(values(2 * hk * keys * d, 2), (2, hk, keys, d), &Device::Cpu)?
            .to_dtype(DType::F16)?
            .narrow(0, 1, 1)?;
        let v = Tensor::from_vec(values(2 * hk * keys * d, 3), (2, hk, keys, d), &Device::Cpu)?
            .to_dtype(DType::F16)?
            .narrow(0, 1, 1)?;
        assert!(q.layout().start_offset() > 0);
        assert!(k.layout().start_offset() > 0);
        let qv = q.flatten_all()?.to_vec1::<f32>()?;
        let kv = k.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let vv = v.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let scale = 1f32 / 128f32.sqrt();
        for prompt_end in [257, 258, 265] {
            let policy = TextArithmeticPolicy::qwen_cpu_k_native_attention(
                TextRequestContext::new(prompt_end)?,
            );
            let whole = attention_rows(&q, &k, &v, scale, start, Some(policy))?.unwrap();
            let mut expected = vec![0f32; h * m * d];
            for head in 0..h {
                for query in 0..m {
                    let pos = start + query;
                    let visible = pos + 1;
                    let qi = (head * m + query) * d;
                    let ki = (head / (h / hk)) * keys * d;
                    let qa = &qv[qi..qi + d];
                    let ka = &kv[ki..ki + visible * d];
                    let va = &vv[ki..ki + visible * d];
                    let out = &mut expected[qi..qi + d];
                    // Independent native batch schedule, without consulting the policy selector.
                    let split =
                        pos >= 256 && (pos >= prompt_end || prompt_end - (pos / 8 * 8) == 1);
                    if split {
                        native_split_attention::split_row(
                            qa,
                            ka,
                            va,
                            out,
                            scale,
                            visible.div_ceil(256) * 256,
                        )
                        .map_err(anyhow::Error::msg)?;
                    } else {
                        row(qa, ka, va, out, scale).map_err(anyhow::Error::msg)?;
                    }
                }
            }
            assert_eq!(
                bits(&whole)?,
                expected.into_iter().map(f32::to_bits).collect::<Vec<_>>()
            );
            for size in [1, 5, 7, 32, 128] {
                let mut parts = Vec::new();
                let mut offset = 0;
                while offset < m {
                    let count = size.min(m - offset);
                    let qpart = q.narrow(2, offset, count)?;
                    let kpart = k.narrow(2, 0, start + offset + count)?;
                    let vpart = v.narrow(2, 0, start + offset + count)?;
                    parts.push(
                        attention_rows(
                            &qpart,
                            &kpart,
                            &vpart,
                            scale,
                            start + offset,
                            Some(policy),
                        )?
                        .unwrap(),
                    );
                    offset += count;
                }
                assert_eq!(
                    bits(&Tensor::cat(&parts, 2)?)?,
                    bits(&whole)?,
                    "prompt={prompt_end}, chunk={size}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn request_split_unsupported_layouts_keep_legacy_attention() -> anyhow::Result<()> {
        let scale = 1f32 / 128f32.sqrt();
        let policy =
            TextArithmeticPolicy::qwen_cpu_k_native_attention(TextRequestContext::new(257)?);
        for (b, h, hk) in [(2, 32, 4), (1, 8, 1), (1, 32, 4)] {
            let q = Tensor::from_vec(values(b * h * 128, 1), (b, h, 1, 128), &Device::Cpu)?;
            let k = Tensor::from_vec(
                values(b * hk * 257 * 128, 2),
                (b, hk, 257, 128),
                &Device::Cpu,
            )?
            .to_dtype(DType::F16)?;
            let v = Tensor::from_vec(
                values(b * hk * 257 * 128, 3),
                (b, hk, 257, 128),
                &Device::Cpu,
            )?
            .to_dtype(DType::F16)?;
            let start = if b == 1 && h == 32 { 1000 } else { 256 };
            assert_eq!(
                bits(&attention_rows(&q, &k, &v, scale, start, Some(policy))?.unwrap())?,
                bits(&attention(&q, &k, &v, scale)?.unwrap())?
            );
            assert_eq!(
                bits(&attention_rows(&q, &k, &v, scale, start, None)?.unwrap())?,
                bits(&attention(&q, &k, &v, scale)?.unwrap())?
            );
            assert!(
                attention_rows(&q.to_dtype(DType::F16)?, &k, &v, scale, start, Some(policy))?
                    .is_none()
            );
            assert!(
                attention_rows(&q, &k.to_dtype(DType::F32)?, &v, scale, start, Some(policy))?
                    .is_none()
            );
        }
        Ok(())
    }

    #[test]
    fn split_row_fixed_f64_bound_and_extent_guards() -> anyhow::Result<()> {
        let q = values(128, 1);
        let k = values(257 * 128, 2);
        let v = values(257 * 128, 3);
        let scale = 1f32 / 128f32.sqrt();
        let mut out = vec![0f32; 128];
        native_split_attention::split_row(&q, &k, &v, &mut out, scale, 512)
            .map_err(anyhow::Error::msg)?;
        let mut scores: Vec<_> = k
            .chunks_exact(128)
            .map(|key| {
                key.iter()
                    .zip(&q)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum::<f64>()
                    * f64::from(scale)
            })
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for s in &mut scores {
            *s = (*s - max).exp();
        }
        let sum = scores.iter().sum::<f64>();
        for (i, &actual) in out.iter().enumerate() {
            let expected = scores
                .iter()
                .enumerate()
                .map(|(n, p)| p / sum * f64::from(v[n * 128 + i]))
                .sum::<f64>();
            assert!((f64::from(actual) - expected).abs() <= 2e-5);
        }
        for physical in [0, 256, 511, 513, 768, 4352] {
            assert!(
                native_split_attention::split_row(&q, &k, &v, &mut out, scale, physical).is_err()
            );
        }
        assert!(
            native_split_attention::split_row(&q[..127], &k, &v, &mut out, scale, 512).is_err()
        );
        assert!(
            native_split_attention::split_row(&q, &k[..k.len() - 1], &v, &mut out, scale, 512)
                .is_err()
        );
        assert!(
            native_split_attention::split_row(&q, &k, &v[..128], &mut out, scale, 512).is_err()
        );
        assert!(
            native_split_attention::split_row(&q, &k, &v, &mut out[..127], scale, 512).is_err()
        );
        assert!(native_split_attention::split_row(&q, &[], &[], &mut out, scale, 512).is_err());
        assert!(native_split_attention::split_row(&q, &k, &v, &mut out, f32::NAN, 512).is_err());
        Ok(())
    }
}
