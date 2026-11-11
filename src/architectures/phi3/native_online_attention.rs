//! One fixed online attention order for every absolute causal query.
//!
//! Arithmetic follows ggml_compute_forward_flash_attn_ext_f16_one_chunk
//! in the pinned llama.cpp CPU reference (MIT, Copyright (c) 2023-2025
//! The ggml authors). This port uses one complete visible prefix per row;
//! it never changes the reduction order with prompt chunk or worker count.
//! Q/K/V must already contain the exact widened F16 values, with scaled Q.

pub fn row(q: &[f32], k: &[f32], v: &[f32], out: &mut [f32]) -> Result<(), &'static str> {
    let dim = q.len();
    if dim != 96
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
        score += 0.0; // Native visible additive mask (+0), scale is exactly 1.
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

/// Phi's native F16 cache path only. Whole rows belong to one worker;
/// no sum crosses workers or changes with the size of the submitted chunk.
pub fn attention(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    v: &candle_core::Tensor,
    arithmetic: Option<(usize, super::super::TextArithmeticPolicy)>,
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
        || dim != 96
        || kb != batch
        || kh != heads
        || keys < queries
        || kd != dim
        || v.dims() != k.dims()
    {
        return Ok(None);
    }
    // A retained sliding window cannot reproduce the native full-prefix graph.
    // Preserve the old complete-prefix path when that separate layout is used.
    let native_phase = arithmetic.filter(|(start, policy)| {
        batch == 1
            && heads == 32
            && start.checked_add(queries) == Some(keys)
            && policy.native_q1_split_at(*start).is_some()
    });
    let q = q.contiguous()?;
    let k = k.to_dtype(DType::F32)?.contiguous()?;
    let v = v.to_dtype(DType::F32)?.contiguous()?;
    let (qs, ql) = q.storage_and_layout();
    let (ks, kl) = k.storage_and_layout();
    let (vs, vl) = v.storage_and_layout();
    let values = |s: &Storage, l: &candle_core::Layout| -> anyhow::Result<(usize, usize)> {
        anyhow::ensure!(matches!(s, Storage::Cpu(_)), "Phi online CPU storage");
        l.contiguous_offsets().context("Phi online contiguous span")
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
        .context("Phi online Q span")?;
    let kv = ks
        .as_slice::<f32>()?
        .get(ka..kz)
        .context("Phi online K span")?;
    let vv = vs
        .as_slice::<f32>()?
        .get(va..vz)
        .context("Phi online V span")?;
    let rows = batch
        .checked_mul(heads)
        .and_then(|x| x.checked_mul(queries))
        .context("Phi online rows overflow")?;
    let total = rows
        .checked_mul(dim)
        .context("Phi online output overflow")?;
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
                let start = group * keys * dim;
                let query = &qv[index * dim..(index + 1) * dim];
                let key = &kv[start..start + visible * dim];
                let value = &vv[start..start + visible * dim];
                let output = &mut out[(index - begin) * dim..(index - begin + 1) * dim];
                if native_phase.is_some_and(|(absolute, policy)| {
                    policy.native_q1_split_at(absolute + index % queries) == Some(true)
                }) {
                    let physical = visible
                        .checked_add(255)
                        .context("Phi native physical key overflow")?
                        / 256
                        * 256;
                    super::native_split_attention::row(query, key, value, output, physical)
                        .map_err(anyhow::Error::msg)?;
                } else {
                    row(query, key, value, output).map_err(anyhow::Error::msg)?;
                }
            }
            Ok(out)
        })();
        *slots[worker].lock().expect("new Phi online worker lock") = Some(result);
    });
    let mut output = Vec::with_capacity(total);
    for slot in slots {
        output.extend(
            slot.into_inner()
                .map_err(|_| anyhow::anyhow!("Phi online lock poisoned"))?
                .context("Phi online worker did not finish")??,
        );
    }
    anyhow::ensure!(output.len() == total, "Phi online output extent");
    Ok(Some(Tensor::from_vec(output, q.shape(), q.device())?))
}

#[cfg(test)]
mod request_tests {
    use super::*;
    use crate::architectures::{TextArithmeticPolicy, TextRequestContext};
    use candle_core::{DType, Device, Tensor};
    #[test]
    fn native_request_attention_keeps_views_chunks_and_phase() -> anyhow::Result<()> {
        let dev = Device::Cpu;
        let start = 250;
        let rows = 12;
        let keys = start + rows;
        let q = Tensor::from_vec(
            (0..32 * (rows + 2) * 96)
                .map(|i| ((i % 13) as f32 - 6.) / 64.)
                .collect::<Vec<_>>(),
            (1, 32, rows + 2, 96),
            &dev,
        )?
        .narrow(2, 2, rows)?;
        let k = Tensor::from_vec(
            (0..32 * (keys + 2) * 96)
                .map(|i| ((i % 41) as f32 - 20.) / 128.)
                .collect::<Vec<_>>(),
            (1, 32, keys + 2, 96),
            &dev,
        )?
        .to_dtype(DType::F16)?
        .narrow(2, 1, keys)?;
        let v = Tensor::from_vec(
            (0..32 * (keys + 2) * 96)
                .map(|i| ((i % 29) as f32 - 14.) / 32.)
                .collect::<Vec<_>>(),
            (1, 32, keys + 2, 96),
            &dev,
        )?
        .to_dtype(DType::F16)?
        .narrow(2, 1, keys)?;
        for end in [257, 258] {
            let policy =
                TextArithmeticPolicy::phi_k_native_attention(TextRequestContext::new(end)?);
            let whole = attention(&q, &k, &v, Some((start, policy)))?.unwrap();
            let whole_values = whole.flatten_all()?.to_vec1::<f32>()?;
            let qv = q.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
            let kv = k
                .to_dtype(DType::F32)?
                .contiguous()?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let vv = v
                .to_dtype(DType::F32)?
                .contiguous()?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for h in 0..32 {
                for r in 0..rows {
                    let visible = start + r + 1;
                    let offset = h * keys * 96;
                    let qs = &qv[(h * rows + r) * 96..(h * rows + r + 1) * 96];
                    let ks = &kv[offset..offset + visible * 96];
                    let vs = &vv[offset..offset + visible * 96];
                    let mut expected = [0f32; 96];
                    if policy.native_q1_split_at(start + r) == Some(true) {
                        super::super::native_split_attention::row(
                            qs,
                            ks,
                            vs,
                            &mut expected,
                            visible.div_ceil(256) * 256,
                        )
                        .unwrap();
                    } else {
                        row(qs, ks, vs, &mut expected).unwrap();
                    }
                    assert_eq!(
                        &whole_values[(h * rows + r) * 96..(h * rows + r + 1) * 96],
                        &expected
                    );
                }
            }
            for size in [1usize, 5, 7, 32, 128] {
                let mut pieces = Vec::new();
                for offset in (0..rows).step_by(size) {
                    let count = size.min(rows - offset);
                    let end = start + offset + count;
                    pieces.push(
                        attention(
                            &q.narrow(2, offset, count)?,
                            &k.narrow(2, 0, end)?,
                            &v.narrow(2, 0, end)?,
                            Some((start + offset, policy)),
                        )?
                        .unwrap(),
                    );
                }
                let merged = Tensor::cat(&pieces, 2)?.flatten_all()?.to_vec1::<f32>()?;
                assert_eq!(merged, whole_values);
            }
            // A truncated window retains its old complete-prefix path.
            let kt = k.narrow(2, 10, keys - 10)?;
            let vt = v.narrow(2, 10, keys - 10)?;
            let a = attention(&q, &kt, &vt, Some((start, policy)))?
                .unwrap()
                .flatten_all()?
                .to_vec1::<f32>()?;
            let b = attention(&q, &kt, &vt, None)?
                .unwrap()
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(a, b);
        }
        Ok(())
    }
}
