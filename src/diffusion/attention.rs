//! Bounded attention without the text engine's implicit f16 conversion.

use candle_core::{D, DType, Result, Tensor};

const QUERY_TILE: usize = 32;
const KEY_TILE: usize = 256;

/// Scaled attention on [batch, heads, sequence, head_dim] tensors.
/// `query_start` locates the first query in the key sequence for causal masks.
pub(crate) fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    query_start: usize,
) -> Result<Tensor> {
    let (batch, heads, queries, dim) = q.dims4()?;
    let (kb, kh, keys, kd) = k.dims4()?;
    let (vb, vh, vk, vd) = v.dims4()?;
    if batch == 0
        || heads == 0
        || queries == 0
        || keys == 0
        || dim == 0
        || (kb, kh, kd) != (batch, heads, dim)
        || (vb, vh, vk, vd) != (batch, heads, keys, dim)
        || !matches!(q.dtype(), DType::F32 | DType::BF16)
        || k.dtype() != q.dtype()
        || v.dtype() != q.dtype()
        || !q.device().same_device(k.device())
        || !q.device().same_device(v.device())
        || (causal
            && query_start
                .checked_add(queries)
                .is_none_or(|end| end > keys))
    {
        candle_core::bail!("invalid diffusion attention shapes, precision or causal range");
    }
    let scale = 1.0 / (dim as f64).sqrt();
    // Candle's native Metal kernels support BF16 and F32. Do not use the
    // autoregressive engine's f16 helper: BF16 -> F16 would change the model.
    if q.device().is_metal() && matches!(dim, 32 | 64 | 72 | 80 | 96 | 128 | 256) {
        if !causal || (query_start == 0 && queries == keys) {
            return candle_nn::ops::sdpa(q, k, v, None, causal, scale as f32, 1.0);
        }
        let mut outputs = Vec::new();
        for start in (0..queries).step_by(QUERY_TILE) {
            let rows = QUERY_TILE.min(queries - start);
            let allowed = query_start + start + rows;
            let qt = q.narrow(2, start, rows)?.contiguous()?;
            let kt = k.narrow(2, 0, allowed)?;
            let vt = v.narrow(2, 0, allowed)?;
            // The vector kernel ignores masks; bounding K/V makes its only
            // query exact. Multi-query kernels receive an explicit offset mask.
            let mask = if rows == 1 {
                None
            } else {
                let values = (0..rows)
                    .flat_map(|row| {
                        (0..allowed).map(move |column| {
                            if column <= query_start + start + row {
                                0_f32
                            } else {
                                f32::NEG_INFINITY
                            }
                        })
                    })
                    .collect::<Vec<_>>();
                Some(
                    Tensor::from_vec(values, (1, 1, rows, allowed), q.device())?
                        .to_dtype(q.dtype())?
                        .broadcast_as((batch, heads, rows, allowed))?,
                )
            };
            outputs.push(candle_nn::ops::sdpa(
                &qt,
                &kt,
                &vt,
                mask.as_ref(),
                false,
                scale as f32,
                1.0,
            )?);
        }
        return Tensor::cat(&outputs, 2);
    }
    tiled_attention(q, k, v, causal, query_start, scale)
}

fn tiled_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    query_start: usize,
    scale: f64,
) -> Result<Tensor> {
    let (batch, heads, queries, dim) = q.dims4()?;
    let keys = k.dim(2)?;
    let mut outputs = Vec::new();
    for start in (0..queries).step_by(QUERY_TILE) {
        let rows = QUERY_TILE.min(queries - start);
        let qt = q
            .narrow(2, start, rows)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let mut max = Tensor::full(f32::NEG_INFINITY, (batch, heads, rows, 1), q.device())?;
        let mut sum = Tensor::zeros((batch, heads, rows, 1), DType::F32, q.device())?;
        let mut out = Tensor::zeros((batch, heads, rows, dim), DType::F32, q.device())?;
        let end = if causal {
            (query_start + start + rows).min(keys)
        } else {
            keys
        };
        for key_start in (0..end).step_by(KEY_TILE) {
            let cols = KEY_TILE.min(end - key_start);
            let kt = k
                .narrow(2, key_start, cols)?
                .to_dtype(DType::F32)?
                .contiguous()?;
            let vt = v
                .narrow(2, key_start, cols)?
                .to_dtype(DType::F32)?
                .contiguous()?;
            let mut scores = (qt.matmul(&kt.transpose(2, 3)?)? * scale)?;
            if causal {
                let values = (0..rows)
                    .flat_map(|row| {
                        (0..cols).map(move |col| {
                            if key_start + col <= query_start + start + row {
                                0_f32
                            } else {
                                f32::NEG_INFINITY
                            }
                        })
                    })
                    .collect::<Vec<_>>();
                scores = scores.broadcast_add(&Tensor::from_vec(
                    values,
                    (1, 1, rows, cols),
                    q.device(),
                )?)?;
            }
            let next_max = max.maximum(&scores.max_keepdim(D::Minus1)?)?;
            let rescale = (&max - &next_max)?.exp()?;
            let probability = scores.broadcast_sub(&next_max)?.exp()?;
            sum = (sum * &rescale)?.add(&probability.sum_keepdim(D::Minus1)?)?;
            out = out
                .broadcast_mul(&rescale)?
                .add(&probability.matmul(&vt)?)?;
            max = next_max;
        }
        outputs.push(out.broadcast_div(&sum)?.to_dtype(q.dtype())?);
        // Bound live Metal command buffers as well as Tensor shapes. The
        // 1152-wide VAE fallback otherwise queues all query/key tiles before
        // releasing their temporary K/V and softmax buffers at native 2K.
        if q.device().is_metal() {
            q.device().synchronize()?;
        }
    }
    Tensor::cat(&outputs, 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn check(device: &Device, dtype: DType, dim: usize) -> Result<()> {
        let q = Tensor::from_vec(
            (0..3 * dim)
                .map(|x| (x as f32 * 0.031).sin())
                .collect::<Vec<_>>(),
            (1, 1, 3, dim),
            device,
        )?
        .to_dtype(dtype)?;
        let k = Tensor::from_vec(
            (0..7 * dim)
                .map(|x| (x as f32 * 0.043).cos())
                .collect::<Vec<_>>(),
            (1, 1, 7, dim),
            device,
        )?
        .to_dtype(dtype)?;
        let v = (&k * 0.3)?;
        for (causal, offset) in [(false, 0), (true, 4)] {
            let got = attention(&q, &k, &v, causal, offset)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let expected = tiled_attention(
                &q.to_device(&Device::Cpu)?,
                &k.to_device(&Device::Cpu)?,
                &v.to_device(&Device::Cpu)?,
                causal,
                offset,
                1.0 / (dim as f64).sqrt(),
            )?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
            for (a, b) in got.iter().zip(expected) {
                assert!(
                    (a - b).abs() < if dtype == DType::BF16 { 0.004 } else { 1e-5 },
                    "{a} != {b}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn diffusion_attention_cpu() -> Result<()> {
        check(&Device::Cpu, DType::F32, 32)
    }

    #[test]
    fn tiled_softmax_matches_dense_across_tile_boundaries() -> Result<()> {
        let device = &Device::Cpu;
        let q = Tensor::from_vec(
            (0..35 * 8)
                .map(|i| (i as f32 * 0.11).sin())
                .collect::<Vec<_>>(),
            (1, 1, 35, 8),
            device,
        )?;
        let k = Tensor::from_vec(
            (0..513 * 8)
                .map(|i| (i as f32 * 0.013).cos())
                .collect::<Vec<_>>(),
            (1, 1, 513, 8),
            device,
        )?;
        let v = (&k * 0.7)?;
        for (causal, offset) in [(false, 0), (true, 250)] {
            let mut scores = (q.matmul(&k.transpose(2, 3)?)? / 8_f64.sqrt())?;
            if causal {
                let mask = (0..35)
                    .flat_map(|r| {
                        (0..513).map(move |c| {
                            if c <= offset + r {
                                0_f32
                            } else {
                                f32::NEG_INFINITY
                            }
                        })
                    })
                    .collect::<Vec<_>>();
                scores = scores.add(&Tensor::from_vec(mask, (1, 1, 35, 513), device)?)?;
            }
            let dense = candle_nn::ops::softmax_last_dim(&scores)?
                .matmul(&v)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let tiled = attention(&q, &k, &v, causal, offset)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert!(dense.iter().zip(tiled).all(|(a, b)| (a - b).abs() < 1e-6));
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn diffusion_attention_metal_preserves_precision() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        // Exercise both native SDPA and the tiled fallback used by the VAE.
        for dim in [8, 32, 1152] {
            check(&device, DType::F32, dim)?;
            check(&device, DType::BF16, dim)?;
        }
        Ok(())
    }

    #[test]
    fn diffusion_attention_causal_does_not_see_future() -> Result<()> {
        let q = Tensor::zeros((1, 1, 2, 4), DType::F32, &Device::Cpu)?;
        let k = Tensor::zeros((1, 1, 4, 4), DType::F32, &Device::Cpu)?;
        let v = Tensor::from_vec(
            vec![
                1_f32, 1., 1., 1., 3., 3., 3., 3., 9., 9., 9., 9., 99., 99., 99., 99.,
            ],
            (1, 1, 4, 4),
            &Device::Cpu,
        )?;
        let got = attention(&q, &k, &v, true, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(
            got.iter()
                .zip([2., 2., 2., 2., 13. / 3., 13. / 3., 13. / 3., 13. / 3.])
                .all(|(a, b)| (a - b).abs() < 1e-6)
        );
        assert!(attention(&q, &k, &v, true, 3).is_err());
        Ok(())
    }
}
