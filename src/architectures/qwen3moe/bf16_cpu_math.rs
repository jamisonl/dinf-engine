#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::cpu_math::native_online_attention;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::native_rope_compat;

use crate::qmatmul::QMatMul;
use anyhow::Result;
use candle_core::{DType, Module, Tensor};

/// Admit only actual stored BF16 CPU weights with the default native cache.
pub(super) fn bf16_norm_weight(
    weight: &candle_core::quantized::QTensor,
    native_cache: bool,
) -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && native_cache
        && weight.device().is_cpu()
        && weight.dtype() == candle_core::quantized::GgmlDType::BF16
}
pub(super) fn bf16_norm_projection(projection: &QMatMul, native_cache: bool) -> bool {
    matches!(projection, QMatMul::QTensor(weight) if bf16_norm_weight(weight, native_cache))
}

/// Match the native CPU BF16 model's F32 rotary recurrence and paired trig.
/// Preserve the dense, quantized, custom-cache and other-device paths.
pub(super) fn bf16_rotary(
    projection: &QMatMul,
    input: &Tensor,
    start: usize,
    base: f64,
    context: usize,
    native_cache: bool,
) -> Result<Option<Tensor>> {
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (projection, input, start, base, context, native_cache);
        Ok(None)
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        use anyhow::Context;
        use candle_core::quantized::GgmlDType;
        let QMatMul::QTensor(weight) = projection else {
            return Ok(None);
        };
        if !native_cache
            || !weight.device().is_cpu()
            || weight.dtype() != GgmlDType::BF16
            || !input.device().is_cpu()
            || input.dtype() != DType::F32
            || input.rank() != 4
            || input.dim(3)? != 128
            || base != 1_000_000.0
        {
            return Ok(None);
        }
        let (batch, heads, queries, dim) = input.dims4()?;
        let end = start
            .checked_add(queries)
            .context("CPU rotary position overflow")?;
        anyhow::ensure!(end <= context, "CPU rotary exceeds the original context");
        let width = heads.checked_mul(dim).context("CPU rotary row overflow")?;
        if batch == 0 || heads == 0 || queries == 0 {
            return Ok(Some(Tensor::zeros(
                input.shape(),
                DType::F32,
                input.device(),
            )?));
        }
        let values = input
            .transpose(1, 2)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut output = Vec::with_capacity(values.len());
        for (row, values) in values.chunks_exact(width).enumerate() {
            let position = start + row % queries;
            let (_, cos, sin, _) =
                native_rope_compat::native_rope_table(position, dim, base as f32);
            output.extend(native_rope_compat::native_rope_rotate(
                values, dim, &cos, &sin,
            ));
        }
        Ok(Some(
            Tensor::from_vec(output, (batch, queries, heads, dim), input.device())?
                .transpose(1, 2)?
                .contiguous()?,
        ))
    }
}

// Keep this wrapper so the Qwen projection policy and its existing tests stay
// local; the shared attention helper preserves the prior head128 operations.
#[cfg(any(test, not(all(target_os = "macos", target_arch = "aarch64"))))]
pub(super) fn attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Option<Tensor>> {
    if q.rank() != 4 || q.dim(3)? != 128 {
        return Ok(None);
    }
    super::super::cpu_math::attention(q, k, v, scale)
}

/// Stored BF16 CPU projections use the native F16-cache query precision.
/// Keep the original query untouched when this layout/cache is not admitted.
#[cfg(test)]
pub(super) fn bf16_attention(
    projection: &QMatMul,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    native_cache: bool,
) -> Result<Option<Tensor>> {
    bf16_attention_rows(projection, q, k, v, native_cache, 0, None)
}

pub(super) fn bf16_attention_rows(
    projection: &QMatMul,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    native_cache: bool,
    start: usize,
    arithmetic: Option<crate::architectures::TextArithmeticPolicy>,
) -> Result<Option<Tensor>> {
    let QMatMul::QTensor(weight) = projection else {
        return Ok(None);
    };
    if !native_cache
        || !weight.device().is_cpu()
        || weight.dtype() != candle_core::quantized::GgmlDType::BF16
        || !q.device().is_cpu()
        || q.dtype() != DType::F32
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
        || q.rank() != 4
        || q.dim(3)? != 128
    {
        return Ok(None);
    }
    let query = q.to_dtype(DType::F16)?.to_dtype(DType::F32)?;
    let scale = 1.0_f32 / 128.0_f32.sqrt();
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        native_online_attention::attention_rows(&query, k, v, scale, start, arithmetic)
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (start, arithmetic);
        attention(&query, k, v, f64::from(scale))
    }
}

pub(super) fn project_bf16(
    weights: &QMatMul,
    input: &Tensor,
    native_cache: bool,
) -> Result<Tensor> {
    if let Some(output) = super::bf16::native_projection(weights, input, native_cache)? {
        return Ok(output);
    }
    project(weights, input)
}

/// CPU F32 matrix multiplication changes its reduction path at one input row.
/// Keep each router/dense projection on that path for every prompt chunk.
/// Quantized projections already process each input row independently.
pub(super) fn project(weights: &QMatMul, input: &Tensor) -> Result<Tensor> {
    if input.device().is_cpu()
        && input.dtype() == DType::F32
        && let QMatMul::Tensor(weights) = weights
        && weights.dtype() == DType::F32
    {
        return dense_rows(input, weights);
    }
    weights.forward(input).map_err(Into::into)
}

#[cfg(test)]
pub(super) fn router(input: &Tensor, weights: &Tensor) -> Result<Tensor> {
    if input.device().is_cpu() && input.dtype() == DType::F32 && weights.dtype() == DType::F32 {
        dense_rows(input, weights)
    } else {
        input.matmul(&weights.t()?).map_err(Into::into)
    }
}

fn dense_rows(input: &Tensor, weights: &Tensor) -> Result<Tensor> {
    let (out_width, in_width) = weights.dims2()?;
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
    let flat = input.contiguous()?.reshape((rows, in_width))?;
    let transposed = weights.t()?;
    let mut outputs = Vec::with_capacity(rows);
    for row in 0..rows {
        outputs.push(flat.narrow(0, row, 1)?.matmul(&transposed)?);
    }
    Tensor::cat(&outputs, 0)?.reshape(shape).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architectures::kv_cache_update_native;
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device, IndexOp};

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn bf16_rotary_preserves_views_batches_and_query_chunks() -> Result<()> {
        let weight = QTensor::quantize(
            &Tensor::ones((1, 128), DType::F32, &Device::Cpu)?,
            GgmlDType::BF16,
        )?;
        let projection = QMatMul::QTensor(std::sync::Arc::new(weight));
        let backing = Tensor::from_vec(
            (0..2 * 4 * 9 * 128)
                .map(|i| ((i * 73 % 997) as f32 - 498.0) / 249.0)
                .collect::<Vec<_>>(),
            (2, 4, 9, 128),
            &Device::Cpu,
        )?;
        let view = backing.narrow(2, 1, 7)?;
        assert!(!view.is_contiguous());
        let before = values(&backing)?;
        let expected = bf16_rotary(&projection, &view, 1023, 1_000_000.0, 4096, true)?.unwrap();
        for chunk in [1, 3, 4] {
            let mut output = Vec::new();
            for start in (0..7).step_by(chunk) {
                let len = chunk.min(7 - start);
                output.push(
                    bf16_rotary(
                        &projection,
                        &view.narrow(2, start, len)?,
                        1023 + start,
                        1_000_000.0,
                        4096,
                        true,
                    )?
                    .unwrap(),
                );
            }
            same_bits(&values(&expected)?, &values(&Tensor::cat(&output, 2)?)?);
        }
        same_bits(&before, &values(&backing)?);
        let one = view.narrow(2, 0, 1)?;
        same_bits(
            &values(&one)?,
            &values(&bf16_rotary(&projection, &one, 0, 1_000_000.0, 4096, true)?.unwrap())?,
        );
        Ok(())
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn bf16_rotary_keeps_other_precision_paths_and_checks_context() -> Result<()> {
        let input = Tensor::ones((1, 4, 3, 128), DType::F32, &Device::Cpu)?;
        let dense = Tensor::ones((1, 128), DType::F32, &Device::Cpu)?;
        let bf16 = QMatMul::QTensor(std::sync::Arc::new(QTensor::quantize(
            &dense,
            GgmlDType::BF16,
        )?));
        let quant = QMatMul::QTensor(std::sync::Arc::new(QTensor::quantize(
            &dense,
            GgmlDType::Q8_0,
        )?));
        for projection in [&quant, &QMatMul::Tensor(dense)] {
            assert!(bf16_rotary(projection, &input, 0, 1_000_000.0, 4096, true)?.is_none());
        }
        assert!(bf16_rotary(&bf16, &input, 0, 1_000_000.0, 4096, false)?.is_none());
        assert!(bf16_rotary(&bf16, &input, 0, 10_000.0, 4096, true)?.is_none());
        assert!(
            bf16_rotary(
                &bf16,
                &input.to_dtype(DType::F16)?,
                0,
                1_000_000.0,
                4096,
                true
            )?
            .is_none()
        );
        assert!(bf16_rotary(&bf16, &input, 4094, 1_000_000.0, 4096, true).is_err());
        assert!(bf16_rotary(&bf16, &input, usize::MAX, 1_000_000.0, 4096, true).is_err());
        Ok(())
    }

    fn values(t: &Tensor) -> Result<Vec<f32>> {
        Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
    }

    fn same_bits(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        assert!(
            left.iter()
                .zip(right)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    fn chunks(q: &Tensor, k: &Tensor, v: &Tensor, chunk: usize) -> Result<Vec<f32>> {
        let total = q.dim(2)?;
        let mut cache = None;
        let mut outputs = Vec::new();
        for start in (0..total).step_by(chunk) {
            let len = chunk.min(total - start);
            let (k, v) = kv_cache_update_native(
                &mut cache,
                k.narrow(2, start, len)?,
                v.narrow(2, start, len)?,
                None,
            )?;
            outputs.push(
                attention(&q.narrow(2, start, len)?, &k, &v, 1.0 / 128f64.sqrt())?
                    .expect("admitted fixture"),
            );
        }
        values(&Tensor::cat(&outputs, 2)?)
    }

    #[test]
    fn cpu_attention_is_chunk_independent_with_native_cache() -> Result<()> {
        let (heads, kv_heads, tokens, dim) = (4, 2, 137, 128);
        let data = |n: usize, salt: usize| {
            (0..n)
                .map(|i| (((i * 73 + salt * 37) % 2039) as f32 / 1019.5) - 1.0)
                .collect::<Vec<_>>()
        };
        let q = Tensor::from_vec(
            data(heads * tokens * dim, 1),
            (1, heads, tokens, dim),
            &Device::Cpu,
        )?;
        let k = Tensor::from_vec(
            data(kv_heads * tokens * dim, 2),
            (1, kv_heads, tokens, dim),
            &Device::Cpu,
        )?;
        let v = Tensor::from_vec(
            data(kv_heads * tokens * dim, 3),
            (1, kv_heads, tokens, dim),
            &Device::Cpu,
        )?;
        let baseline = chunks(&q, &k, &v, tokens)?;
        assert!(baseline.iter().all(|x| x.is_finite()));
        for chunk in [1, 5, 7, 32, 128] {
            same_bits(&baseline, &chunks(&q, &k, &v, chunk)?);
        }
        let q = values(&q)?;
        let k = values(&k.to_dtype(DType::F16)?)?;
        let v = values(&v.to_dtype(DType::F16)?)?;
        for position in [0, 31, 136] {
            for head in 0..heads {
                let kv_head = head / (heads / kv_heads);
                let scores = (0..=position)
                    .map(|key| {
                        (0..dim)
                            .map(|d| {
                                q[(head * tokens + position) * dim + d] as f64
                                    * k[(kv_head * tokens + key) * dim + d] as f64
                            })
                            .sum::<f64>()
                            / 128f64.sqrt()
                    })
                    .collect::<Vec<_>>();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let p = scores
                    .into_iter()
                    .map(|x| (x - maximum).exp())
                    .collect::<Vec<_>>();
                let sum = p.iter().sum::<f64>();
                for d in 0..dim {
                    let expected = p
                        .iter()
                        .enumerate()
                        .map(|(key, p)| p / sum * v[(kv_head * tokens + key) * dim + d] as f64)
                        .sum::<f64>();
                    assert!(
                        (baseline[(head * tokens + position) * dim + d] as f64 - expected).abs()
                            <= 2e-5
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn cpu_attention_preserves_batch_and_gqa_causal_means() -> Result<()> {
        let q = Tensor::zeros((2, 4, 9, 128), DType::F32, &Device::Cpu)?;
        let k = Tensor::zeros((2, 2, 9, 128), DType::F16, &Device::Cpu)?;
        let v = Tensor::from_vec(
            (0..2 * 2 * 9 * 128)
                .map(|i| {
                    (100 * (i / (128 * 9 * 2)) + 10 * (i / (128 * 9) % 2) + 2 * (i / 128 % 9))
                        as f32
                })
                .collect::<Vec<_>>(),
            (2, 2, 9, 128),
            &Device::Cpu,
        )?
        .to_dtype(DType::F16)?;
        let expected = chunks(&q, &k, &v, 9)?;
        for chunk in [1, 2, 3, 5, 7, 8, 9] {
            same_bits(&expected, &chunks(&q, &k, &v, chunk)?);
        }
        for (i, actual) in expected.iter().enumerate() {
            let mean =
                (100 * (i / (128 * 9 * 4)) + 10 * ((i / (128 * 9) % 4) / 2) + (i / 128 % 9)) as f32;
            assert!((actual - mean).abs() <= 2e-5);
        }
        Ok(())
    }

    #[test]
    fn cpu_attention_long_history_keeps_prefix_reduction_shape() -> Result<()> {
        // The former AV GEMM split a 3051-key reduction at key 1526. A
        // shorter prefill used one reduction block, despite zero probability
        // for every later key. Keep this regression near that real boundary
        // with two query heads, so it needs only a few MiB of working memory.
        let (past, prefix, total, dim) = (1512, 1536, 3051, 128);
        let data = |n: usize, salt: usize| {
            (0..n)
                .map(|i| (((i * 73 + salt * 37) % 2039) as f32 / 1019.5) - 1.0)
                .collect::<Vec<_>>()
        };
        let q = Tensor::from_vec(
            data(2 * (total - past) * dim, 1),
            (1, 2, total - past, dim),
            &Device::Cpu,
        )?;
        let k = Tensor::from_vec(data(total * dim, 2), (1, 1, total, dim), &Device::Cpu)?
            .to_dtype(DType::F16)?;
        let v = Tensor::from_vec(data(total * dim, 3), (1, 1, total, dim), &Device::Cpu)?
            .to_dtype(DType::F16)?;
        let scale = 1.0 / 128f64.sqrt();
        let full = attention(&q, &k, &v, scale)?.expect("admitted fixture");
        let short = attention(
            &q.narrow(2, 0, prefix - past)?,
            &k.narrow(2, 0, prefix)?,
            &v.narrow(2, 0, prefix)?,
            scale,
        )?
        .expect("admitted fixture");
        same_bits(
            &values(&full.narrow(2, 0, prefix - past)?)?,
            &values(&short)?,
        );
        let q_values = values(&q)?;
        let k_values = values(&k)?;
        let v_values = values(&v)?;
        let actual = values(&short)?;
        for position in [1525, 1526, 1535] {
            for head in 0..2 {
                let query_offset = (head * (total - past) + position - past) * dim;
                let scores = (0..=position)
                    .map(|key| {
                        (0..dim)
                            .map(|d| {
                                q_values[query_offset + d] as f64 * k_values[key * dim + d] as f64
                            })
                            .sum::<f64>()
                            * scale
                    })
                    .collect::<Vec<_>>();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let probabilities = scores
                    .iter()
                    .map(|score| (score - maximum).exp())
                    .collect::<Vec<_>>();
                let sum = probabilities.iter().sum::<f64>();
                for d in 0..dim {
                    let expected = probabilities
                        .iter()
                        .enumerate()
                        .map(|(key, p)| p / sum * v_values[key * dim + d] as f64)
                        .sum::<f64>();
                    let i = (head * (prefix - past) + position - past) * dim + d;
                    assert!((actual[i] as f64 - expected).abs() <= 2e-5);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn cpu_attention_excludes_overflowed_future_scores_and_signed_zero() -> Result<()> {
        let mut q = vec![0f32; 2 * 2 * 128];
        q[0] = 1e38;
        q[2 * 128] = 1e38;
        let q = Tensor::from_vec(q, (1, 2, 2, 128), &Device::Cpu)?;
        let mut k = vec![0f32; 2 * 128];
        k[128] = 65504.0;
        let k = Tensor::from_vec(k, (1, 1, 2, 128), &Device::Cpu)?.to_dtype(DType::F16)?;
        for zero in [false, true] {
            let mut v = vec![-0f32; 2 * 128];
            if zero {
                v[128..].fill(0.0);
            } else {
                v[..128].fill(2.0);
            }
            let v = Tensor::from_vec(v, (1, 1, 2, 128), &Device::Cpu)?.to_dtype(DType::F16)?;
            let full = chunks(&q, &k, &v, 2)?;
            same_bits(&full, &chunks(&q, &k, &v, 1)?);
            for (i, x) in full.iter().enumerate() {
                let expected = if zero {
                    0.0
                } else if (i / 128) % 2 == 0 {
                    2.0
                } else {
                    1.0
                };
                assert_eq!(*x, expected);
            }
        }
        Ok(())
    }

    #[test]
    fn cpu_attention_declines_other_dtypes_and_geometries() -> Result<()> {
        let q = Tensor::zeros((1, 4, 3, 128), DType::F32, &Device::Cpu)?;
        let k = Tensor::zeros((1, 2, 3, 128), DType::F16, &Device::Cpu)?;
        assert!(attention(&q, &k, &k, 0.0)?.is_none());
        assert!(attention(&q.to_dtype(DType::F16)?, &k, &k, 1.0)?.is_none());
        assert!(attention(&q, &k.to_dtype(DType::F32)?, &k, 1.0)?.is_none());
        assert!(attention(&q, &k.narrow(2, 0, 2)?, &k.narrow(2, 0, 2)?, 1.0)?.is_none());
        assert!(
            attention(
                &q.narrow(3, 0, 64)?,
                &k.narrow(3, 0, 64)?,
                &k.narrow(3, 0, 64)?,
                1.0
            )?
            .is_none()
        );
        Ok(())
    }

    #[test]
    fn cpu_dense_rows_match_single_row_math_for_every_chunk() -> Result<()> {
        let (rows, width, outputs) = (37, 2048, 128);
        let input = Tensor::from_vec(
            (0..rows * width)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) / 61.0)
                .collect::<Vec<_>>(),
            (1, rows, width),
            &Device::Cpu,
        )?;
        let weights = Tensor::from_vec(
            (0..outputs * width)
                .map(|i| ((i * 73 % 89) as f32 - 44.0) / 59.0)
                .collect::<Vec<_>>(),
            (outputs, width),
            &Device::Cpu,
        )?;
        let projection = QMatMul::Tensor(weights.clone());
        let full = values(&project(&projection, &input)?)?;
        for chunk in [1, 5, 7, 32, 37] {
            let parts = (0..rows)
                .step_by(chunk)
                .map(|start| {
                    project(
                        &projection,
                        &input.narrow(1, start, chunk.min(rows - start))?,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            same_bits(&full, &values(&Tensor::cat(&parts, 1)?)?);
        }
        let x = values(&input)?;
        let w = values(&weights)?;
        for row in [0, 17, 36] {
            for output in 0..outputs {
                let terms = (0..width)
                    .map(|i| x[row * width + i] as f64 * w[output * width + i] as f64)
                    .collect::<Vec<_>>();
                let expected = terms.iter().sum::<f64>();
                let bound = 2e-6 + 2e-6 * terms.iter().map(|v| v.abs()).sum::<f64>();
                assert!((full[row * outputs + output] as f64 - expected).abs() <= bound);
            }
        }
        same_bits(
            &full,
            &values(&router(&input.reshape((rows, width))?, &weights)?)?,
        );
        let selected = input.i((.., 3..5, ..))?;
        same_bits(
            &values(&project(&projection, &selected)?)?,
            &full[3 * outputs..5 * outputs],
        );
        Ok(())
    }

    #[test]
    fn cpu_quantized_projection_keeps_its_existing_row_path() -> Result<()> {
        let input = Tensor::from_vec(
            (0..5 * 32).map(|i| i as f32 / 37.0).collect::<Vec<_>>(),
            (1, 5, 32),
            &Device::Cpu,
        )?;
        let weights = Tensor::from_vec(
            (0..4 * 32)
                .map(|i| i as f32 / 61.0 - 1.0)
                .collect::<Vec<_>>(),
            (4, 32),
            &Device::Cpu,
        )?;
        let quantized = QMatMul::QTensor(std::sync::Arc::new(QTensor::quantize(
            &weights,
            GgmlDType::Q8_0,
        )?));
        same_bits(
            &values(&project(&quantized, &input)?)?,
            &values(&quantized.forward(&input)?)?,
        );
        Ok(())
    }
}

#[cfg(test)]
mod bf16_query_precision_tests {
    use super::*;
    use candle_core::Device;
    use candle_core::quantized::{GgmlDType, QTensor};
    use std::sync::Arc;
    fn data(n: usize, salt: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 137 + salt * 431) % 1009) as f32 / 1009.0 - 0.5)
            .collect()
    }
    fn bf16_projection() -> Result<QMatMul> {
        Ok(QMatMul::QTensor(Arc::new(QTensor::quantize(
            &Tensor::ones((32, 128), DType::F32, &Device::Cpu)?,
            GgmlDType::BF16,
        )?)))
    }
    fn bits(t: &Tensor) -> Result<Vec<u32>> {
        Ok(t.flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }
    #[test]
    fn cpu_bf16_query_rounding_and_chunks_match_fixed_half_inputs() -> Result<()> {
        let projection = bf16_projection()?;
        let q = Tensor::from_vec(data(2 * 9 * 128, 1), (1, 2, 9, 128), &Device::Cpu)?;
        let k = Tensor::from_vec(data(9 * 128, 2), (1, 1, 9, 128), &Device::Cpu)?
            .to_dtype(DType::F16)?;
        let v = Tensor::from_vec(data(9 * 128, 3), (1, 1, 9, 128), &Device::Cpu)?
            .to_dtype(DType::F16)?;
        let before = bits(&q)?;
        let actual = bf16_attention(&projection, &q, &k, &v, true)?.unwrap();
        let scale = f64::from(1.0_f32 / 128.0_f32.sqrt());
        let half_q = q.to_dtype(DType::F16)?.to_dtype(DType::F32)?;
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let expected = native_online_attention::attention(&half_q, &k, &v, scale as f32)?.unwrap();
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        let expected = attention(&half_q, &k, &v, scale)?.unwrap();
        assert_eq!(bits(&actual)?, bits(&expected)?);
        assert_ne!(
            bits(&actual)?,
            bits(&attention(&q, &k, &v, scale)?.unwrap())?
        );
        assert_eq!(bits(&q)?, before);
        for chunk in [1, 3, 8] {
            let mut rows = Vec::new();
            for start in (0..9).step_by(chunk) {
                let take = chunk.min(9 - start);
                rows.push(
                    bf16_attention(
                        &projection,
                        &q.narrow(2, start, take)?,
                        &k.narrow(2, 0, start + take)?,
                        &v.narrow(2, 0, start + take)?,
                        true,
                    )?
                    .unwrap(),
                );
            }
            assert_eq!(bits(&Tensor::cat(&rows, 2)?)?, bits(&actual)?);
        }
        Ok(())
    }
    #[test]
    fn cpu_bf16_query_declines_dense_and_custom_cache() -> Result<()> {
        let projection = bf16_projection()?;
        let q = Tensor::ones((1, 2, 1, 128), DType::F32, &Device::Cpu)?;
        let k = Tensor::ones((1, 1, 1, 128), DType::F16, &Device::Cpu)?;
        let dense = QMatMul::Tensor(Tensor::ones((32, 128), DType::F32, &Device::Cpu)?);
        assert!(bf16_attention(&dense, &q, &k, &k, true)?.is_none());
        assert!(bf16_attention(&projection, &q, &k, &k, false)?.is_none());
        assert!(bf16_attention(&projection, &q.to_dtype(DType::F16)?, &k, &k, true)?.is_none());
        assert!(bf16_attention(&projection, &q, &k.to_dtype(DType::F32)?, &k, true)?.is_none());
        for dtype in [
            GgmlDType::F32,
            GgmlDType::F16,
            GgmlDType::Q8_0,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
        ] {
            let other = QMatMul::QTensor(Arc::new(QTensor::quantize(
                &Tensor::ones((32, 256), DType::F32, &Device::Cpu)?,
                dtype,
            )?));
            assert!(bf16_attention(&other, &q, &k, &k, true)?.is_none());
        }
        Ok(())
    }
}

#[cfg(test)]
mod bf16_native_policy_tests {
    use super::*;
    use candle_core::{
        Device,
        quantized::{GgmlDType, QTensor},
    };
    use std::sync::Arc;
    #[test]
    fn cpu_bf16_norm_admission_keeps_dense_quant_and_custom_cache() -> Result<()> {
        let w = Tensor::ones((8, 256), DType::F32, &Device::Cpu)?;
        let bf = Arc::new(QTensor::quantize(&w, GgmlDType::BF16)?);
        assert_eq!(
            bf16_norm_weight(&bf, true),
            cfg!(all(target_os = "macos", target_arch = "aarch64"))
        );
        assert!(!bf16_norm_weight(&bf, false));
        assert!(!bf16_norm_projection(&QMatMul::Tensor(w.clone()), true));
        for kind in [
            GgmlDType::F32,
            GgmlDType::F16,
            GgmlDType::Q8_0,
            GgmlDType::Q4K,
            GgmlDType::Q6K,
        ] {
            let q = QMatMul::QTensor(Arc::new(QTensor::quantize(&w, kind)?));
            assert!(!bf16_norm_projection(&q, true));
        }
        Ok(())
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn cpu_bf16_online_batches_gqa_offsets_and_f64() -> Result<()> {
        let (batch, heads, kv_heads, queries, keys, dim) = (2, 4, 2, 5, 17, 128);
        let data = |n: usize, salt: usize| {
            (0..n)
                .map(|i| ((i * 73 + salt * 137) % 997) as f32 / 997. - 0.5)
                .collect::<Vec<_>>()
        };
        let backing = Tensor::from_vec(
            data((batch + 1) * heads * queries * dim, 1),
            (batch + 1, heads, queries, dim),
            &Device::Cpu,
        )?;
        let q = backing.narrow(0, 1, batch)?;
        assert_eq!(q.layout().start_offset(), heads * queries * dim);
        let k = Tensor::from_vec(
            data(batch * kv_heads * keys * dim, 2),
            (batch, kv_heads, keys, dim),
            &Device::Cpu,
        )?
        .to_dtype(DType::F16)?;
        let v = Tensor::from_vec(
            data(batch * kv_heads * keys * dim, 3),
            (batch, kv_heads, keys, dim),
            &Device::Cpu,
        )?
        .to_dtype(DType::F16)?;
        let weight = QMatMul::QTensor(Arc::new(QTensor::quantize(
            &Tensor::ones((8, dim), DType::F32, &Device::Cpu)?,
            GgmlDType::BF16,
        )?));
        let y = bf16_attention(&weight, &q, &k, &v, true)?.unwrap();
        let values = |t: &Tensor| -> Result<Vec<f32>> { Ok(t.flatten_all()?.to_vec1::<f32>()?) };
        let actual = values(&y)?;
        let qv = values(&q.to_dtype(DType::F16)?.to_dtype(DType::F32)?)?;
        let kv = values(&k.to_dtype(DType::F32)?)?;
        let vv = values(&v.to_dtype(DType::F32)?)?;
        let scale = f64::from(1f32 / (dim as f32).sqrt());
        for b in 0..batch {
            for h in 0..heads {
                for t in 0..queries {
                    let qi = ((b * heads + h) * queries + t) * dim;
                    let ki = (b * kv_heads + h / (heads / kv_heads)) * keys * dim;
                    let visible = keys - queries + t + 1;
                    let scores = (0..visible)
                        .map(|j| {
                            (0..dim)
                                .map(|d| f64::from(qv[qi + d]) * f64::from(kv[ki + j * dim + d]))
                                .sum::<f64>()
                                * scale
                        })
                        .collect::<Vec<_>>();
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let prob = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
                    let total = prob.iter().sum::<f64>();
                    for d in 0..dim {
                        let want = (0..visible)
                            .map(|j| prob[j] * f64::from(vv[ki + j * dim + d]))
                            .sum::<f64>()
                            / total;
                        assert!((f64::from(actual[qi + d]) - want).abs() < 2e-5);
                    }
                }
            }
        }
        let bits = |t: &Tensor| -> Result<Vec<u32>> {
            Ok(values(t)?.into_iter().map(f32::to_bits).collect())
        };
        for chunk in [1, 2, 3, 5] {
            let mut pieces = Vec::new();
            for at in (0..queries).step_by(chunk) {
                let take = chunk.min(queries - at);
                pieces.push(
                    bf16_attention(
                        &weight,
                        &q.narrow(2, at, take)?,
                        &k.narrow(2, 0, keys - queries + at + take)?,
                        &v.narrow(2, 0, keys - queries + at + take)?,
                        true,
                    )?
                    .unwrap(),
                );
            }
            assert_eq!(bits(&Tensor::cat(&pieces, 2)?)?, bits(&y)?);
        }
        assert!(
            bf16_attention(&weight, &q, &k.narrow(2, 0, 1)?, &v.narrow(2, 0, 1)?, true)?.is_none()
        );
        Ok(())
    }
}
