use crate::qmatmul::QMatMul;
use anyhow::{Context, Result};
use candle_core::{DType, Module, Tensor};

/// Fix CPU reduction shapes to positions within the retained KV history.
/// Requires a causal, non-windowed history; callers keep windowed paths separate.
/// Full and chunked prefill must use the same floating-point operations: tiny
/// attention changes can alter Q8 activation codes and then routed experts.
/// Fixed query tiles retain matrix throughput while fixing the key extent.
/// Missing queries/keys in a partial tile are zero-filled, and the causal
/// selection removes every future score, including overflowed values.
pub(super) fn attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Option<Tensor>> {
    if !q.device().is_cpu()
        || !q.device().same_device(k.device())
        || !q.device().same_device(v.device())
        || q.dtype() != DType::F32
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
        || [q, k, v].iter().any(|x| x.rank() != 4)
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Ok(None);
    }
    let (batch, heads, queries, dim) = q.dims4()?;
    let (kbatch, kv_heads, keys, kdim) = k.dims4()?;
    if batch == 0
        || heads == 0
        || queries == 0
        || kv_heads == 0
        || keys < queries
        || batch != kbatch
        || !heads.is_multiple_of(kv_heads)
        || !matches!(dim, 96 | 128)
        || kdim != dim
        || v.dims() != k.dims()
    {
        return Ok(None);
    }
    let repeats = heads / kv_heads;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    // With no GQA expansion, 32 queries retain useful CPU GEMM throughput.
    // Keep the established eight-query shape for every other geometry.
    let tile = if dim == 96 && repeats == 1 { 32 } else { 8 };
    let past = keys - queries;
    let mut outputs = Vec::new();
    let mut row = 0;
    while row < queries {
        let position = past + row;
        let tile_start = position / tile * tile;
        let end = tile_start
            .checked_add(tile)
            .context("CPU attention tile overflow")?;
        let offset = position - tile_start;
        let take = (queries - row).min(tile - offset);
        let mut parts = Vec::new();
        if offset > 0 {
            parts.push(Tensor::zeros(
                (batch, heads, offset, dim),
                DType::F32,
                q.device(),
            )?);
        }
        parts.push(q.narrow(2, row, take)?);
        if offset + take < tile {
            parts.push(Tensor::zeros(
                (batch, heads, tile - offset - take, dim),
                DType::F32,
                q.device(),
            )?);
        }
        let grouped_rows = repeats
            .checked_mul(tile)
            .context("CPU attention row overflow")?;
        let query =
            Tensor::cat(&parts, 2)?
                .contiguous()?
                .reshape((batch, kv_heads, grouped_rows, dim))?;
        let prefix = |x: &Tensor| -> candle_core::Result<Tensor> {
            let known = x.narrow(2, 0, keys.min(end))?;
            if keys < end {
                Tensor::cat(
                    &[
                        known,
                        Tensor::zeros((batch, kv_heads, end - keys, dim), DType::F32, q.device())?,
                    ],
                    2,
                )?
                .contiguous()
            } else {
                known.contiguous()
            }
        };
        let keys = prefix(&k)?;
        let values = prefix(&v)?;
        let mask_size = grouped_rows
            .checked_mul(end)
            .context("CPU attention mask overflow")?;
        let mut mask = vec![0u8; mask_size];
        for (i, r) in mask.chunks_exact_mut(end).enumerate() {
            r[..tile_start + i % tile + 1].fill(1);
        }
        let mask = Tensor::from_vec(mask, (1, 1, grouped_rows, end), q.device())?;
        let scores = query.matmul(&keys.transpose(2, 3)?)? * scale;
        let scores = scores?;
        // Select replaces masked infinities/NaNs; adding -inf cannot do that.
        let negative = Tensor::full(f32::NEG_INFINITY, scores.shape(), q.device())?;
        let scores = mask
            .broadcast_as(scores.shape())?
            .contiguous()?
            .where_cond(&scores, &negative)?;
        let probabilities = candle_nn::ops::softmax_last_dim(&scores)?;
        let output = probabilities
            .matmul(&values)?
            .reshape((batch, heads, tile, dim))?;
        outputs.push(output.narrow(2, offset, take)?);
        row += take;
    }
    Ok(Some(Tensor::cat(&outputs, 2)?))
}

/// Keep CPU dense projection reductions independent of prompt chunk size.
/// Every call uses eight rows, padding only the unused rows of its last tile.
/// This avoids the different one-row GEMM path without copying the weights.
/// Quantized matrices and other devices/dtypes keep their existing kernels.
pub(super) fn project(weights: &QMatMul, input: &Tensor) -> Result<Tensor> {
    if !input.device().is_cpu() || input.dtype() != DType::F32 {
        return weights.forward(input).map_err(Into::into);
    }
    let QMatMul::Tensor(dense) = weights else {
        return weights.forward(input).map_err(Into::into);
    };
    if !dense.device().is_cpu() || dense.dtype() != DType::F32 {
        return weights.forward(input).map_err(Into::into);
    }
    let (out_width, in_width) = dense.dims2()?;
    anyhow::ensure!(
        input.rank() >= 2 && input.dim(input.rank() - 1)? == in_width && in_width > 0,
        "CPU projection input and weight shapes differ"
    );
    let rows = input.elem_count() / in_width;
    let mut shape = input.dims().to_vec();
    *shape.last_mut().expect("checked input rank") = out_width;
    if rows == 0 {
        return Tensor::zeros(shape, DType::F32, input.device()).map_err(Into::into);
    }
    const TILE: usize = 8;
    let input = input.contiguous()?.reshape((rows, in_width))?;
    let transposed = dense.t()?;
    let mut outputs = Vec::with_capacity(rows.div_ceil(TILE));
    for row in (0..rows).step_by(TILE) {
        let take = TILE.min(rows - row);
        let tile = input.narrow(0, row, take)?;
        let tile = if take < TILE {
            Tensor::cat(
                &[
                    tile,
                    Tensor::zeros((TILE - take, in_width), DType::F32, input.device())?,
                ],
                0,
            )?
        } else {
            tile
        };
        outputs.push(tile.matmul(&transposed)?.narrow(0, 0, take)?);
    }
    Tensor::cat(&outputs, 0)?.reshape(shape).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_core::quantized::{GgmlDType, QTensor};
    use std::sync::Arc;

    fn data(n: usize, salt: u32, amplitude: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let mut x = (i as u32).wrapping_mul(747796405).wrapping_add(salt);
                x = ((x >> ((x >> 28) + 4)) ^ x).wrapping_mul(277803737);
                x = (x >> 22) ^ x;
                ((x & 0xffffff) as f32 / 8388608.0 - 1.0) * amplitude
            })
            .collect()
    }
    fn values(t: &Tensor) -> Result<Vec<f32>> {
        Ok(t.flatten_all()?.to_vec1()?)
    }
    fn bits(a: &Tensor, b: &Tensor) -> Result<()> {
        let a = values(a)?;
        let b = values(b)?;
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
        Ok(())
    }
    fn chunked(q: &Tensor, k: &Tensor, v: &Tensor, chunk: usize) -> Result<Tensor> {
        let total = q.dim(2)?;
        let scale = 1.0 / (q.dim(3)? as f64).sqrt();
        let mut output = Vec::new();
        for start in (0..total).step_by(chunk) {
            let take = chunk.min(total - start);
            output.push(
                attention(
                    &q.narrow(2, start, take)?,
                    &k.narrow(2, 0, start + take)?,
                    &v.narrow(2, 0, start + take)?,
                    scale,
                )?
                .expect("admitted"),
            );
        }
        Tensor::cat(&output, 2).map_err(Into::into)
    }
    fn expected(q: &[f32], k: &[f32], v: &[f32], position: usize, dim: usize, d: usize) -> f64 {
        let scores = (0..=position)
            .map(|key| {
                (0..dim)
                    .map(|j| q[j] as f64 * k[key * dim + j] as f64)
                    .sum::<f64>()
                    / (dim as f64).sqrt()
            })
            .collect::<Vec<_>>();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let probs = scores.iter().map(|x| (x - max).exp()).collect::<Vec<_>>();
        let sum = probs.iter().sum::<f64>();
        probs
            .iter()
            .enumerate()
            .map(|(key, p)| p / sum * v[key * dim + d] as f64)
            .sum()
    }

    #[test]
    fn head96_native_cache_matches_all_chunk_sizes_and_f64() -> Result<()> {
        let (batch, heads, kv_heads, total, dim) = (2, 4, 2, 137, 96);
        let q = Tensor::from_vec(
            data(batch * heads * total * dim, 17, 1.0),
            (batch, heads, total, dim),
            &Device::Cpu,
        )?;
        let k = Tensor::from_vec(
            data(batch * kv_heads * total * dim, 71, 1.0),
            (batch, kv_heads, total, dim),
            &Device::Cpu,
        )?
        .to_dtype(DType::F16)?;
        let v = Tensor::from_vec(
            data(batch * kv_heads * total * dim, 93, 1.0),
            (batch, kv_heads, total, dim),
            &Device::Cpu,
        )?
        .to_dtype(DType::F16)?;
        let baseline = chunked(&q, &k, &v, total)?;
        for chunk in [1, 5, 7, 32, 128] {
            bits(&baseline, &chunked(&q, &k, &v, chunk)?)?;
        }
        let actual = values(&baseline)?;
        let q = values(&q)?;
        let k = values(&k.to_dtype(DType::F32)?)?;
        let v = values(&v.to_dtype(DType::F32)?)?;
        for b in 0..batch {
            for h in 0..heads {
                for pos in [0, 31, 136] {
                    let qi = ((b * heads + h) * total + pos) * dim;
                    let ki = (b * kv_heads + h / (heads / kv_heads)) * total * dim;
                    for d in 0..dim {
                        let oracle = expected(
                            &q[qi..qi + dim],
                            &k[ki..ki + total * dim],
                            &v[ki..ki + total * dim],
                            pos,
                            dim,
                            d,
                        );
                        assert!((actual[qi + d] as f64 - oracle).abs() <= 2e-5);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn head96_without_gqa_is_chunk_independent_with_partial_32_query_tiles() -> Result<()> {
        let (batch, heads, total, dim) = (2, 2, 137, 96);
        let q = Tensor::from_vec(
            data(batch * heads * total * dim, 17, 1.0),
            (batch, heads, total, dim),
            &Device::Cpu,
        )?;
        let k = Tensor::from_vec(
            data(batch * heads * total * dim, 71, 1.0),
            (batch, heads, total, dim),
            &Device::Cpu,
        )?
        .to_dtype(DType::F16)?;
        let v = Tensor::from_vec(
            data(batch * heads * total * dim, 93, 1.0),
            (batch, heads, total, dim),
            &Device::Cpu,
        )?
        .to_dtype(DType::F16)?;
        let baseline = chunked(&q, &k, &v, total)?;
        for chunk in [1, 5, 7, 8, 31, 32, 33, 128] {
            bits(&baseline, &chunked(&q, &k, &v, chunk)?)?;
        }
        let actual = values(&baseline)?;
        let q = values(&q)?;
        let k = values(&k.to_dtype(DType::F32)?)?;
        let v = values(&v.to_dtype(DType::F32)?)?;
        for b in 0..batch {
            for h in 0..heads {
                for pos in [0, 7, 31, 32, 127, 136] {
                    let qi = ((b * heads + h) * total + pos) * dim;
                    let ki = (b * heads + h) * total * dim;
                    for d in [0, 31, 95] {
                        let oracle = expected(
                            &q[qi..qi + dim],
                            &k[ki..ki + total * dim],
                            &v[ki..ki + total * dim],
                            pos,
                            dim,
                            d,
                        );
                        assert!((actual[qi + d] as f64 - oracle).abs() <= 2e-5);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn head96_prefix_is_independent_of_later_keys_across_gemm_depth_split() -> Result<()> {
        let (total, past, prefix, dim) = (3298, 1600, 1664, 96);
        let q = Tensor::from_vec(
            data((total - past) * dim, 17, 1.0),
            (1, 1, total - past, dim),
            &Device::Cpu,
        )?;
        let k = Tensor::from_vec(data(total * dim, 71, 1.0), (1, 1, total, dim), &Device::Cpu)?
            .to_dtype(DType::F16)?;
        let v = Tensor::from_vec(data(total * dim, 93, 1.0), (1, 1, total, dim), &Device::Cpu)?
            .to_dtype(DType::F16)?;
        let scale = 1.0 / (dim as f64).sqrt();
        let full = attention(&q, &k, &v, scale)?.expect("admitted");
        let short = attention(
            &q.narrow(2, 0, prefix - past)?,
            &k.narrow(2, 0, prefix)?,
            &v.narrow(2, 0, prefix)?,
            scale,
        )?
        .expect("admitted");
        bits(&full.narrow(2, 0, prefix - past)?, &short)?;
        let actual = values(&short)?;
        let q = values(&q)?;
        let k = values(&k.to_dtype(DType::F32)?)?;
        let v = values(&v.to_dtype(DType::F32)?)?;
        for pos in [1648, 1649, 1663] {
            for d in [0, 31, 95] {
                let qi = (pos - past) * dim;
                assert!(
                    (actual[qi + d] as f64 - expected(&q[qi..qi + dim], &k, &v, pos, dim, d)).abs()
                        <= 2e-5
                );
            }
        }
        Ok(())
    }

    #[test]
    fn fixed8_dense_projection_is_chunk_independent_and_matches_f64() -> Result<()> {
        let (rows, width, outputs) = (129, 3072, 96);
        let x = Tensor::from_vec(data(rows * width, 17, 1.0), (1, rows, width), &Device::Cpu)?;
        let w = Tensor::from_vec(
            data(outputs * width, 71, 0.0625),
            (outputs, width),
            &Device::Cpu,
        )?;
        let projection = QMatMul::Tensor(w.clone());
        let baseline = project(&projection, &x)?;
        for chunk in [1, 5, 7, 32, 128, 129] {
            let mut parts = Vec::new();
            for start in (0..rows).step_by(chunk) {
                parts.push(project(
                    &projection,
                    &x.narrow(1, start, chunk.min(rows - start))?,
                )?);
            }
            bits(&baseline, &Tensor::cat(&parts, 1)?)?;
        }
        bits(
            &baseline.narrow(1, 3, 11)?,
            &project(&projection, &x.narrow(1, 3, 11)?)?,
        )?;
        let actual = values(&baseline)?;
        let x = values(&x)?;
        let w = values(&w)?;
        for row in [0, 1, 7, 8, 31, 32, 127, 128] {
            for out in [0, 1, 31, 32, 95] {
                let expected = (0..width)
                    .map(|i| x[row * width + i] as f64 * w[out * width + i] as f64)
                    .sum::<f64>();
                assert!((actual[row * outputs + out] as f64 - expected).abs() <= 2e-5);
            }
        }
        Ok(())
    }

    #[test]
    fn quantized_projection_and_empty_input_keep_contracts() -> Result<()> {
        let x = Tensor::from_vec(data(5 * 32, 17, 1.0), (1, 5, 32), &Device::Cpu)?;
        let w = Tensor::from_vec(data(96 * 32, 71, 0.0625), (96, 32), &Device::Cpu)?;
        let quantized = QMatMul::QTensor(Arc::new(QTensor::quantize(&w, GgmlDType::Q8_0)?));
        bits(&project(&quantized, &x)?, &quantized.forward(&x)?)?;
        let empty = project(
            &QMatMul::Tensor(w),
            &Tensor::zeros((2, 0, 32), DType::F32, &Device::Cpu)?,
        )?;
        assert_eq!(empty.dims(), [2, 0, 96]);
        Ok(())
    }

    #[test]
    fn attention_declines_unsupported_widths_and_cache_dtypes() -> Result<()> {
        for dim in [64, 192] {
            let q = Tensor::zeros((1, 2, 3, dim), DType::F32, &Device::Cpu)?;
            let k = Tensor::zeros((1, 1, 3, dim), DType::F16, &Device::Cpu)?;
            assert!(attention(&q, &k, &k, 1.0)?.is_none());
        }
        let q = Tensor::zeros((1, 2, 3, 96), DType::F32, &Device::Cpu)?;
        let k = Tensor::zeros((1, 1, 3, 96), DType::F16, &Device::Cpu)?;
        assert!(attention(&q.to_dtype(DType::F16)?, &k, &k, 1.0)?.is_none());
        assert!(attention(&q, &k.to_dtype(DType::F32)?, &k, 1.0)?.is_none());
        assert!(attention(&q, &k, &k, f64::NAN)?.is_none());
        Ok(())
    }
}
