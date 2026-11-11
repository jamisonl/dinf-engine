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
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
}

fn same_bits(a: &Tensor, b: &Tensor) -> Result<()> {
    let (a, b) = (values(a)?, values(b)?);
    assert_eq!(a.len(), b.len());
    assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
    Ok(())
}

fn chunked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    past: usize,
    chunk: usize,
    scale: f64,
) -> Result<Tensor> {
    use crate::architectures::kv_cache_update_native;

    let mut cache = None;
    if past > 0 {
        kv_cache_update_native(
            &mut cache,
            k.narrow(2, 0, past)?,
            v.narrow(2, 0, past)?,
            None,
        )?;
    }
    let queries = q.dim(2)?;
    assert_eq!(k.dim(2)?, past + queries);
    let mut output = Vec::new();
    for start in (0..queries).step_by(chunk) {
        let take = chunk.min(queries - start);
        let (keys, vals) = kv_cache_update_native(
            &mut cache,
            k.narrow(2, past + start, take)?,
            v.narrow(2, past + start, take)?,
            None,
        )?;
        output.push(
            attention(&q.narrow(2, start, take)?, &keys, &vals, scale)?
                .expect("native MLA cache admitted"),
        );
    }
    Tensor::cat(&output, 2).map_err(Into::into)
}

fn expected(q: &[f32], k: &[f32], v: &[f32], position: usize, d: usize, scale: f64) -> f64 {
    let scores = (0..=position)
        .map(|key| {
            (0..192)
                .map(|j| q[j] as f64 * k[key * 192 + j] as f64)
                .sum::<f64>()
                * scale
        })
        .collect::<Vec<_>>();
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let probabilities = scores
        .iter()
        .map(|s| (s - maximum).exp())
        .collect::<Vec<_>>();
    let sum = probabilities.iter().sum::<f64>();
    probabilities
        .iter()
        .enumerate()
        .map(|(key, p)| p / sum * v[key * 128 + d] as f64)
        .sum()
}

#[test]
fn mla_sixteen_heads_native_cache_matches_chunks_and_f64() -> Result<()> {
    let (heads, total) = (16, 137);
    // Head layout is a view, as in the portable MLA path.
    let q = Tensor::from_vec(
        data(total * heads * 192, 17, 1.0),
        (1, total, heads, 192),
        &Device::Cpu,
    )?
    .transpose(1, 2)?;
    let k = Tensor::from_vec(
        data(total * heads * 192, 71, 1.0),
        (1, total, heads, 192),
        &Device::Cpu,
    )?
    .transpose(1, 2)?;
    let v = Tensor::from_vec(
        data(total * heads * 128, 93, 1.0),
        (1, total, heads, 128),
        &Device::Cpu,
    )?
    .transpose(1, 2)?;
    let scale = 1.35 / 192f64.sqrt();
    let baseline = chunked(&q, &k, &v, 0, total, scale)?;
    assert_eq!(baseline.dims(), [1, heads, total, 128]);
    for chunk in [1, 5, 7, 32, 128] {
        same_bits(&baseline, &chunked(&q, &k, &v, 0, chunk, scale)?)?;
    }
    let actual = values(&baseline)?;
    let q = values(&q)?;
    let k = values(&k.to_dtype(DType::F16)?)?;
    let v = values(&v.to_dtype(DType::F16)?)?;
    for head in [0, 7, 15] {
        for pos in [0, 31, 136] {
            let qi = (head * total + pos) * 192;
            let ki = head * total * 192;
            let vi = head * total * 128;
            for d in 0..128 {
                let want = expected(
                    &q[qi..qi + 192],
                    &k[ki..ki + total * 192],
                    &v[vi..vi + total * 128],
                    pos,
                    d,
                    scale,
                );
                assert!((actual[(head * total + pos) * 128 + d] as f64 - want).abs() <= 2e-5);
            }
        }
    }
    Ok(())
}

#[test]
fn mla_long_prefix_preserves_offsets_batch_and_gqa() -> Result<()> {
    let (batch, heads, kv_heads, past, total) = (2, 4, 2, 1529, 1572);
    let queries = total - past;
    let q = Tensor::from_vec(
        data(batch * heads * queries * 192, 17, 1.0),
        (batch, heads, queries, 192),
        &Device::Cpu,
    )?;
    let k = Tensor::from_vec(
        data(batch * kv_heads * total * 192, 71, 1.0),
        (batch, kv_heads, total, 192),
        &Device::Cpu,
    )?;
    let v = Tensor::from_vec(
        data(batch * kv_heads * total * 128, 93, 1.0),
        (batch, kv_heads, total, 128),
        &Device::Cpu,
    )?;
    let scale = 0.125;
    let baseline = chunked(&q, &k, &v, past, queries, scale)?;
    for chunk in [1, 5, 7, 8, 32, 128] {
        same_bits(&baseline, &chunked(&q, &k, &v, past, chunk, scale)?)?;
    }
    let actual = values(&baseline)?;
    let q = values(&q)?;
    let k = values(&k.to_dtype(DType::F16)?)?;
    let v = values(&v.to_dtype(DType::F16)?)?;
    for b in 0..batch {
        for head in 0..heads {
            let kh = b * kv_heads + head / (heads / kv_heads);
            for pos in [1529, 1535, 1536, 1571] {
                let qi = ((b * heads + head) * queries + pos - past) * 192;
                for d in [0, 63, 127] {
                    let want = expected(
                        &q[qi..qi + 192],
                        &k[kh * total * 192..(kh + 1) * total * 192],
                        &v[kh * total * 128..(kh + 1) * total * 128],
                        pos,
                        d,
                        scale,
                    );
                    let oi = ((b * heads + head) * queries + pos - past) * 128 + d;
                    assert!((actual[oi] as f64 - want).abs() <= 2e-5);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn mla_admission_stays_at_native_geometry_and_f16_cache() -> Result<()> {
    for (dim, value_dim) in [(96, 96), (128, 128), (192, 96), (192, 192), (256, 128)] {
        let q = Tensor::zeros((1, 2, 3, dim), DType::F32, &Device::Cpu)?;
        let k = Tensor::zeros((1, 2, 3, dim), DType::F16, &Device::Cpu)?;
        let v = Tensor::zeros((1, 2, 3, value_dim), DType::F16, &Device::Cpu)?;
        assert!(attention(&q, &k, &v, 1.0)?.is_none());
    }
    let q = Tensor::zeros((1, 2, 3, 192), DType::F32, &Device::Cpu)?;
    let k = Tensor::zeros((1, 2, 3, 192), DType::F16, &Device::Cpu)?;
    let v = Tensor::zeros((1, 2, 3, 128), DType::F16, &Device::Cpu)?;
    assert!(attention(&q, &k, &v, 1.0)?.is_some());
    assert!(attention(&q, &k.to_dtype(DType::F32)?, &v, 1.0)?.is_none());
    assert!(attention(&q, &k, &v.narrow(2, 0, 2)?, 1.0)?.is_none());
    assert!(attention(&q, &k, &v, f64::NAN)?.is_none());
    Ok(())
}

#[test]
fn cpu_router_and_full_projection_match_chunks_and_f64() -> Result<()> {
    let (rows, width, experts) = (137, 2048, 64);
    let x = Tensor::from_vec(data(rows * width, 17, 1.0), (rows, width), &Device::Cpu)?;
    let w = Tensor::from_vec(
        data(experts * width, 71, 0.03125),
        (experts, width),
        &Device::Cpu,
    )?;
    let baseline = router(&x, &w)?;
    let weights = QMatMul::Tensor(w.clone());
    same_bits(&baseline, &project(&weights, &x)?)?;
    for chunk in [1, 5, 7, 32, 128] {
        let mut parts = Vec::new();
        for start in (0..rows).step_by(chunk) {
            parts.push(router(&x.narrow(0, start, chunk.min(rows - start))?, &w)?);
        }
        same_bits(&baseline, &Tensor::cat(&parts, 0)?)?;
    }
    let (actual, x, w) = (values(&baseline)?, values(&x)?, values(&w)?);
    for row in [0, 7, 8, 31, 32, 127, 136] {
        for expert in [0, 31, 63] {
            let want = (0..width)
                .map(|i| x[row * width + i] as f64 * w[expert * width + i] as f64)
                .sum::<f64>();
            assert!((actual[row * experts + expert] as f64 - want).abs() <= 2e-5);
        }
    }
    Ok(())
}

#[test]
fn declined_quantized_projection_keeps_the_existing_kernel() -> Result<()> {
    let x = Tensor::from_vec(data(17 * 32, 17, 1.0), (1, 17, 32), &Device::Cpu)?;
    // The native Q8 adapter requires a whole four-column group.
    let w = Tensor::from_vec(data(63 * 32, 71, 0.0625), (63, 32), &Device::Cpu)?;
    let weights = QMatMul::QTensor(Arc::new(QTensor::quantize(&w, GgmlDType::Q8_0)?));
    assert!(!super::super::native_cpu::supported(&weights));
    same_bits(&project(&weights, &x)?, &weights.forward(&x)?)?;
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn admitted_q8_projection_matches_exact_integer_golden() -> Result<()> {
    // Both stored scales are exactly one: each block has unique maximum127.
    // These half ties use the native RNE codes proved by the quantizer component.
    let ties = [0.5f32, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5];
    let codes = [0i32, 2, 2, 4, 0, -2, -2, -4];
    let x: Vec<f32> = (0..3 * 32)
        .map(|i| {
            if i % 32 == 0 {
                127.0
            } else {
                ties[(i + i / 32) % 8]
            }
        })
        .collect();
    let w: Vec<f32> = (0..4 * 32)
        .map(|i| {
            if i % 32 == 0 {
                127.0
            } else {
                ((i * 7) % 15) as f32 - 7.0
            }
        })
        .collect();
    let weights = QMatMul::QTensor(Arc::new(QTensor::quantize(
        &Tensor::from_vec(w.clone(), (4, 32), &Device::Cpu)?,
        GgmlDType::Q8_0,
    )?));
    assert!(super::super::native_cpu::supported(&weights));
    let input = Tensor::from_vec(x, (1, 3, 32), &Device::Cpu)?;
    let actual = values(&project(&weights, &input)?)?;
    assert_eq!(actual.len(), 12);
    for row in 0..3 {
        for col in 0..4 {
            let expected: i32 = (0..32)
                .map(|i| {
                    let code = if i == 0 {
                        127
                    } else {
                        codes[(row * 32 + i + row) % 8]
                    };
                    code * w[col * 32 + i] as i32
                })
                .sum();
            // Small integer products and sum are exact in F32; no tolerance.
            assert_eq!(actual[row * 4 + col].to_bits(), (expected as f32).to_bits());
        }
    }
    let legacy = values(&weights.forward(&input)?)?;
    assert!(
        actual
            .iter()
            .zip(legacy)
            .any(|(a, b)| a.to_bits() != b.to_bits())
    );
    Ok(())
}
