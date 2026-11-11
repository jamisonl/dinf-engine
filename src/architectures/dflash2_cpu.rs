//! DFlash's pinned macOS ARM CPU arithmetic. Other devices keep their old path.
//! Queries, probabilities and sums stay F32; K/V remain stored as F16.
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
#[path = "dflash2_cpu_attention.rs"]
mod native_attention;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::super::qwen3moe::native_rope_compat as native_rope;

pub(super) fn rope_tables(
    dim: usize,
    base: f32,
    context: usize,
    device: &Device,
) -> Result<Option<(Tensor, Tensor)>> {
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (dim, base, context, device);
        Ok(None)
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        if !device.is_cpu() {
            return Ok(None);
        }
        anyhow::ensure!(
            dim > 0 && dim.is_multiple_of(2) && base.is_finite() && base > 0.0 && context > 0,
            "DFlash CPU rotary table geometry"
        );
        let count = context
            .checked_mul(dim / 2)
            .context("DFlash rotary table extent")?;
        let mut cos = Vec::with_capacity(count);
        let mut sin = Vec::with_capacity(count);
        for p in 0..context {
            let (_, c, s, _) = native_rope::native_rope_table(p, dim, base);
            cos.extend(c);
            sin.extend(s);
        }
        Ok(Some((
            Tensor::from_vec(cos, (context, dim / 2), device)?,
            Tensor::from_vec(sin, (context, dim / 2), device)?,
        )))
    }
}

pub(super) fn rope(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Option<Tensor>> {
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (input, cos, sin);
        Ok(None)
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        if !input.device().is_cpu()
            || !cos.device().is_cpu()
            || !sin.device().is_cpu()
            || input.dtype() != DType::F32
            || cos.dtype() != DType::F32
            || sin.dtype() != DType::F32
            || input.rank() != 4
        {
            return Ok(None);
        }
        let (b, h, q, d) = input.dims4()?;
        if b == 0
            || h == 0
            || q == 0
            || d == 0
            || !d.is_multiple_of(2)
            || cos.dims() != [q, d / 2]
            || sin.dims() != cos.dims()
        {
            return Ok(None);
        }
        let x = input.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let c = cos.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let s = sin.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let total = b
            .checked_mul(h)
            .and_then(|n| n.checked_mul(q))
            .and_then(|n| n.checked_mul(d))
            .context("DFlash rotary extent")?;
        anyhow::ensure!(
            x.len() == total && c.len() == q * (d / 2) && s.len() == c.len(),
            "DFlash rotary logical span"
        );
        let mut out = Vec::with_capacity(total);
        for (row, values) in x.chunks_exact(d).enumerate() {
            let p = row % q;
            out.extend(native_rope::native_rope_rotate(
                values,
                d,
                &c[p * d / 2..(p + 1) * d / 2],
                &s[p * d / 2..(p + 1) * d / 2],
            ));
        }
        Ok(Some(Tensor::from_vec(out, input.shape(), input.device())?))
    }
}

pub(super) fn attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Option<Tensor>> {
    if !cfg!(all(target_os = "macos", target_arch = "aarch64"))
        || !q.device().is_cpu()
        || !k.device().is_cpu()
        || !v.device().is_cpu()
        || q.dtype() != DType::F32
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
        || q.rank() != 4
        || k.rank() != 4
        || v.rank() != 4
        || k.dims() != v.dims()
    {
        return Ok(None);
    }
    let (b, h, queries, dim) = q.dims4()?;
    let (kb, kvh, keys, kdim) = k.dims4()?;
    if b != 1
        || kb != b
        || h != 32
        || kvh != 8
        || dim != 128
        || kdim != dim
        || queries == 0
        || keys < queries
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Ok(None);
    }
    // The native F16 KV cache pads physical key rows to a multiple of256.
    // Masked suffix entries have zero probability. row() keeps the same
    // four-lane softmax grouping while reading only the occupied prefix.
    let _physical_keys = keys.checked_add(255).context("DFlash padded key extent")? / 256 * 256;
    let x = q.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let ks = k
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let vs = v
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let qr = h.checked_mul(queries).context("DFlash query rows")?;
    let kt = kvh
        .checked_mul(keys)
        .and_then(|n| n.checked_mul(dim))
        .context("DFlash cache extent")?;
    let total = qr
        .checked_mul(dim)
        .context("DFlash attention output extent")?;
    anyhow::ensure!(
        x.len() == total && ks.len() == kt && vs.len() == kt,
        "DFlash attention logical spans"
    );
    let mut output = Vec::with_capacity(total);
    for row in 0..qr {
        let head = row / queries;
        let base = (head / (h / kvh)) * keys * dim;
        let (_, _, y) = native_attention::row(
            &x[row * dim..(row + 1) * dim],
            &ks[base..base + keys * dim],
            &vs[base..base + keys * dim],
            scale,
        )
        .map_err(anyhow::Error::msg)?;
        output.extend(y);
    }
    Ok(Some(Tensor::from_vec(output, q.shape(), q.device())?))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unsupported_attention_layouts_decline() -> Result<()> {
        let q = Tensor::zeros((1, 4, 3, 128), DType::F32, &Device::Cpu)?;
        let k = Tensor::zeros((1, 2, 5, 128), DType::F16, &Device::Cpu)?;
        assert!(attention(&q, &k, &k, 0.125)?.is_none());
        let q = Tensor::zeros((1, 32, 3, 128), DType::F32, &Device::Cpu)?;
        assert!(attention(&q, &k.to_dtype(DType::F32)?, &k, 0.125)?.is_none());
        Ok(())
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn attention_offsets_match_small_f64_reference() -> Result<()> {
        let qv: Vec<_> = (0..32 * 3 * 128)
            .map(|i| ((i * 13 % 31) as f32 - 15.0) / 64.0)
            .collect();
        let kv: Vec<_> = (0..8 * 5 * 128)
            .map(|i| ((i * 7 % 29) as f32 - 14.0) / 32.0)
            .collect();
        let vv: Vec<_> = (0..8 * 5 * 128)
            .map(|i| ((i * 11 % 23) as f32 - 11.0) / 32.0)
            .collect();
        let q = Tensor::from_vec(qv.clone(), (1, 32, 3, 128), &Device::Cpu)?;
        let k = Tensor::from_vec(kv.clone(), (1, 8, 5, 128), &Device::Cpu)?.to_dtype(DType::F16)?;
        let v = Tensor::from_vec(vv.clone(), (1, 8, 5, 128), &Device::Cpu)?.to_dtype(DType::F16)?;
        let out = attention(&q, &k, &v, 0.125)?
            .unwrap()
            .flatten_all()?
            .to_vec1::<f32>()?;
        for h in 0..32 {
            for row in 0..3 {
                let scores: Vec<_> = (0..5)
                    .map(|j| {
                        (0..128)
                            .map(|d| {
                                f64::from(qv[(h * 3 + row) * 128 + d])
                                    * f64::from(kv[((h / 4) * 5 + j) * 128 + d])
                            })
                            .sum::<f64>()
                            * 0.125
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut p: Vec<_> = scores.iter().map(|x| (x - max).exp()).collect();
                let sum = p.iter().sum::<f64>();
                p.iter_mut().for_each(|x| *x /= sum);
                for d in 0..128 {
                    let expected = (0..5)
                        .map(|j| p[j] * f64::from(vv[((h / 4) * 5 + j) * 128 + d]))
                        .sum::<f64>();
                    assert!((f64::from(out[(h * 3 + row) * 128 + d]) - expected).abs() < 2e-6);
                }
            }
        }
        let offset = Tensor::cat(
            &[
                &Tensor::zeros((1, 32, 1, 128), DType::F32, &Device::Cpu)?,
                &q,
            ],
            2,
        )?
        .narrow(2, 1, 3)?;
        let actual = attention(&offset, &k, &v, 0.125)?
            .unwrap()
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            out.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        Ok(())
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn rotary_offsets_and_position_bounds() -> Result<()> {
        let (c, s) = rope_tables(128, 1e7, 8, &Device::Cpu)?.unwrap();
        let values: Vec<_> = (0..2 * 3 * 128)
            .map(|i| ((i % 29) as f32 - 14.0) / 16.0)
            .collect();
        let x = Tensor::from_vec(values, (1, 2, 3, 128), &Device::Cpu)?;
        let expected = super::super::apply_rope(&x, 2, &c, &s)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let offset = Tensor::cat(
            &[
                &Tensor::zeros((1, 2, 1, 128), DType::F32, &Device::Cpu)?,
                &x,
            ],
            2,
        )?
        .narrow(2, 1, 3)?;
        let actual = super::super::apply_rope(&offset, 2, &c, &s)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert!(super::super::apply_rope(&x, 6, &c, &s).is_err());
        assert!(super::super::apply_rope(&x, usize::MAX, &c, &s).is_err());
        Ok(())
    }
}
