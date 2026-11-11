use super::*;
use candle_core::{Device, IndexOp};

fn norm_fixture(rows: usize, width: usize) -> (Vec<f32>, Vec<f32>) {
    let values = (0..rows * width)
        .map(|i| {
            let u = (i as u32)
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            ((u % 8193) as i32 - 4096) as f32 / 4096.0 + (i % 7) as f32 / 1_048_576.0
        })
        .collect();
    let weights = (0..width)
        .map(|i| ((i * 29 % 257) as f32 + 128.0) / 256.0)
        .collect();
    (values, weights)
}

#[test]
fn deep_synthetic_native_norm_rounding_exact() -> Result<()> {
    use sha2::{Digest, Sha256};
    fn hash(values: &[f32]) -> String {
        let mut hash = Sha256::new();
        for value in values {
            hash.update(value.to_le_bytes());
        }
        format!("{:x}", hash.finalize())
    }
    // Non-model inputs. The golden came from the pinned native CPU graph's
    // RMS_NORM followed by MUL, with one thread and F32 epsilon. It checks
    // the generic norm's rounding and operation order without a self-oracle.
    let (values, weights) = norm_fixture(2, 3072);
    assert_eq!(
        hash(&values),
        "21a6e9bdb8c0d268eea6d3b8b52751818f465e09c180bebda6a805cb69e6e3d6"
    );
    assert_eq!(
        hash(&weights),
        "aa013a3f4f3ec1015a46379a58c35f86021dd830019747a8d6abd52bf0c9fb04"
    );
    let input = Tensor::from_vec(values, (2, 3072), &Device::Cpu)?;
    let weight = Tensor::from_vec(weights, 3072, &Device::Cpu)?;
    let output = rms_norm(&input, &weight, f64::from(1e-5_f32))?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(output.len(), 6144);
    assert_eq!(
        hash(&output),
        "f1d5450e41353243cfe7c7448477b545e7c8b0ed6b150599ee72135fa603dc3c"
    );
    Ok(())
}

#[test]
fn deep_synthetic_norm_shape_offsets_chunks_and_f64() -> Result<()> {
    let (rows, width) = (5, 2048);
    let (values, weights) = norm_fixture(rows + 2, width);
    let full = Tensor::from_vec(values, (rows + 2, width), &Device::Cpu)?;
    let input = full.narrow(0, 1, rows)?;
    let mut padded_weights = vec![123.0_f32];
    padded_weights.extend(weights.iter().copied());
    let weight = Tensor::from_vec(padded_weights, width + 1, &Device::Cpu)?.narrow(0, 1, width)?;
    assert_eq!(input.layout().start_offset(), width);
    assert_eq!(weight.layout().start_offset(), 1);
    let epsilon = f64::from(1e-6_f32);
    let output = rms_norm(&input, &weight, epsilon)?;
    assert_eq!(output.dims(), [rows, width]);
    assert_eq!(output.i(0)?.dims(), [width]);
    let bits = |value: &Tensor| -> Result<Vec<u32>> {
        Ok(value
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .map(|v| v.to_bits())
            .collect())
    };
    let expected = bits(&output)?;
    for chunk in [1, 2, 5] {
        let mut pieces = Vec::new();
        for start in (0..rows).step_by(chunk) {
            pieces.push(rms_norm(
                &input.narrow(0, start, chunk.min(rows - start))?,
                &weight,
                epsilon,
            )?);
        }
        assert_eq!(bits(&Tensor::cat(&pieces, 0)?)?, expected);
    }
    let strided = input.t()?.contiguous()?.t()?;
    assert_eq!(bits(&rms_norm(&strided, &weight, epsilon)?)?, expected);
    let inputs = input.to_vec2::<f32>()?;
    let actual = output.to_vec2::<f32>()?;
    assert_eq!(actual.len(), rows);
    for (row, actual) in inputs.iter().zip(&actual) {
        let mean = row.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / width as f64;
        let denominator = (mean + epsilon).sqrt();
        assert_eq!(actual.len(), width);
        for ((&x, &weight), &value) in row.iter().zip(&weights).zip(actual) {
            let reference = f64::from(x) / denominator * f64::from(weight);
            assert!((f64::from(value) - reference).abs() <= 2e-6);
        }
    }
    Ok(())
}
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn deep_q8_offset_and_chunk_bits() -> Result<()> {
    use candle_core::quantized::{GgmlDType, QTensor};
    let w = Tensor::from_vec(
        (0..8 * 64)
            .map(|i| ((i * 7 % 113) as f32 - 56.) / 29.)
            .collect::<Vec<_>>(),
        (8, 64),
        &Device::Cpu,
    )?;
    let w = QMatMul::from_qtensor(QTensor::quantize(&w, GgmlDType::Q8_0)?)?;
    let x = Tensor::from_vec(
        (0..7 * 64)
            .map(|i| ((i * 19 % 137) as f32 - 68.) / 17.)
            .collect::<Vec<_>>(),
        (7, 64),
        &Device::Cpu,
    )?;
    let view = x.narrow(0, 1, 5)?;
    assert_eq!(view.layout().start_offset(), 64);
    let y = project(&w, &view)?.unwrap().to_vec2::<f32>()?;
    assert_eq!(y.len(), 5);
    for (i, values) in y.iter().enumerate() {
        let row = project(&w, &view.i(i)?.unsqueeze(0)?)?
            .unwrap()
            .to_vec2::<f32>()?;
        assert_eq!(
            values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            row[0].iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    assert!(
        project(
            &QMatMul::Tensor(Tensor::zeros((8, 64), DType::F32, &Device::Cpu)?),
            &view
        )?
        .is_none()
    );
    assert!(project(&w, &view.to_dtype(DType::F16)?)?.is_none());
    assert!(project(&w, &view.t()?)?.is_none());
    Ok(())
}
#[test]
fn deep_norm_bad_geometry_rejected() -> Result<()> {
    let x = Tensor::ones((1, 64), DType::F32, &Device::Cpu)?;
    let w = Tensor::ones(32, DType::F32, &Device::Cpu)?;
    assert!(rms_norm(&x, &w, 1e-6).is_err());
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn deep_ffn_admission_requires_all_q8_and_native_cache() -> Result<()> {
    use candle_core::quantized::{GgmlDType, QTensor};
    let q = |rows, cols| -> Result<QMatMul> {
        Ok(QMatMul::from_qtensor(QTensor::quantize(
            &Tensor::ones((rows, cols), DType::F32, &Device::Cpu)?,
            GgmlDType::Q8_0,
        )?)?)
    };
    let g = q(32, 64)?;
    let u = q(32, 64)?;
    let d = q(64, 32)?;
    assert!(admitted_ffn(&g, &u, &d, true));
    assert!(!admitted_ffn(&g, &u, &d, false));
    assert!(!admitted_ffn(&g, &u, &q(64, 64)?, true));
    let dense = QMatMul::Tensor(Tensor::ones((32, 64), DType::F32, &Device::Cpu)?);
    assert!(!admitted_ffn(&dense, &u, &d, true));
    assert!(!admitted_ffn(&g, &dense, &d, true));
    assert!(!admitted_ffn(&g, &u, &dense, true));
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn deep_swiglu_offset_chunk_and_f64() -> Result<()> {
    let width = 1408;
    let g = Tensor::from_vec(
        (0..7 * width)
            .map(|i| ((i * 17 % 97) as f32 - 48.) / 16.)
            .collect::<Vec<_>>(),
        (7, width),
        &Device::Cpu,
    )?;
    let u = Tensor::from_vec(
        (0..7 * width)
            .map(|i| ((i * 13 % 83) as f32 - 41.) / 41.)
            .collect::<Vec<_>>(),
        (7, width),
        &Device::Cpu,
    )?;
    let g = g.narrow(0, 1, 5)?;
    let u = u.narrow(0, 2, 5)?;
    assert_eq!(g.layout().start_offset(), width);
    assert_eq!(u.layout().start_offset(), 2 * width);
    let y = swiglu(&g, &u)?.to_vec2::<f32>()?;
    let gv = g.to_vec2::<f32>()?;
    let uv = u.to_vec2::<f32>()?;
    for row in 0..5 {
        let one = swiglu(&g.i(row)?.unsqueeze(0)?, &u.i(row)?.unsqueeze(0)?)?.to_vec2::<f32>()?;
        assert_eq!(
            y[row].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            one[0].iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        for col in 0..width {
            let x = gv[row][col] as f64;
            let r = x / (1. + (-x).exp()) * uv[row][col] as f64;
            assert!((y[row][col] as f64 - r).abs() <= 2e-6);
        }
    }
    let transposed =
        swiglu(&g.t()?.contiguous()?.t()?, &u.t()?.contiguous()?.t()?)?.to_vec2::<f32>()?;
    assert_eq!(
        y.iter().flatten().map(|x| x.to_bits()).collect::<Vec<_>>(),
        transposed
            .iter()
            .flatten()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn deep_swiglu_rejects_invalid_shape_dtype_values() -> Result<()> {
    let x = Tensor::ones((2, 8), DType::F32, &Device::Cpu)?;
    assert!(swiglu(&x, &x.to_dtype(DType::F16)?).is_err());
    assert!(swiglu(&x, &x.narrow(0, 0, 1)?).is_err());
    let odd = Tensor::ones((2, 7), DType::F32, &Device::Cpu)?;
    assert!(swiglu(&odd, &odd).is_err());
    let nan = Tensor::from_vec(vec![f32::NAN; 16], (2, 8), &Device::Cpu)?;
    assert!(swiglu(&nan, &x).is_err());
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn deep_k_request_chunks_offsets_and_f64() -> Result<()> {
    use candle_core::quantized::{GgmlDType, QTensor};
    for kind in [GgmlDType::Q4K, GgmlDType::Q5K, GgmlDType::Q6K] {
        let (n, k, rows) = (8, 512, 19);
        let w = Tensor::from_vec(
            (0..n * k)
                .map(|i| ((i * 7 % 101) as f32 - 50.) / 32.)
                .collect::<Vec<_>>(),
            (n, k),
            &Device::Cpu,
        )?;
        let w = QTensor::quantize(&w, kind)?;
        let wd = w.dequantize(&Device::Cpu)?.to_vec2::<f32>()?;
        let w = QMatMul::from_qtensor(w)?;
        let full = Tensor::from_vec(
            (0..(rows + 2) * k)
                .map(|i| {
                    if i % 256 == 0 {
                        127.
                    } else {
                        (i % 17) as f32 - 8.
                    }
                })
                .collect::<Vec<_>>(),
            (1, rows + 2, k),
            &Device::Cpu,
        )?;
        let x = full.narrow(1, 1, rows)?;
        assert_eq!(x.layout().start_offset(), k);
        let policy = RowPolicy {
            start: 0,
            seq_len: rows,
            m4_prefix_end: 12,
        };
        let y = project_rows(&w, &x, Some(policy))?.unwrap();
        let bits = |x: &Tensor| -> Result<Vec<u32>> {
            Ok(x.flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .map(|v| v.to_bits())
                .collect())
        };
        for chunk in [1, 5, 7, 32, 128] {
            let mut pieces = Vec::new();
            for start in (0..rows).step_by(chunk) {
                let count = chunk.min(rows - start);
                pieces.push(
                    project_rows(
                        &w,
                        &x.narrow(1, start, count)?,
                        Some(RowPolicy {
                            start,
                            seq_len: count,
                            ..policy
                        }),
                    )?
                    .unwrap(),
                );
            }
            assert_eq!(bits(&y)?, bits(&Tensor::cat(&pieces, 1)?)?);
        }
        let xv = x.squeeze(0)?.to_vec2::<f32>()?;
        let yv = y.squeeze(0)?.to_vec2::<f32>()?;
        assert_eq!(yv.len(), rows);
        for (yr, xr) in yv.iter().zip(&xv) {
            for (value, wr) in yr.iter().zip(&wd) {
                let terms = wr
                    .iter()
                    .zip(xr)
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .collect::<Vec<_>>();
                let sum: f64 = terms.iter().sum();
                let bound = 2e-6 + 2e-6 * terms.iter().map(|v| v.abs()).sum::<f64>();
                assert!((*value as f64 - sum).abs() <= bound);
            }
        }
        assert!(
            project_rows(
                &w,
                &x,
                Some(RowPolicy {
                    seq_len: 0,
                    ..policy
                })
            )
            .is_err()
        );
        assert!(
            project_rows(
                &w,
                &x,
                Some(RowPolicy {
                    start: usize::MAX,
                    ..policy
                })
            )
            .is_err()
        );
        assert!(project_rows(&w, &x.to_dtype(DType::F16)?, Some(policy))?.is_none());
        assert!(!norm_supported(&w, false));
        assert!(norm_supported(&w, true));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn deep_routed_q5_even_blocks_offset_rows_and_f64() -> Result<()> {
    use candle_core::quantized::{GgmlDType, QTensor};
    let (n, k, rows) = (8, 1408, 5);
    let w = Tensor::from_vec(
        (0..n * k)
            .map(|i| ((i * 13 % 79) as f32 - 39.) / 64.)
            .collect::<Vec<_>>(),
        (n, k),
        &Device::Cpu,
    )?;
    let w = QTensor::quantize(&w, GgmlDType::Q5_0)?;
    let wd = w.dequantize(&Device::Cpu)?.to_vec2::<f32>()?;
    let w = QMatMul::from_qtensor(w)?;
    assert!(supported_routed(&w));
    assert!(!supported_quant(&w));
    let full = Tensor::from_vec(
        (0..(rows + 2) * k)
            .map(|i| {
                if i % 32 == 0 {
                    127.
                } else {
                    (i % 17) as f32 - 8.
                }
            })
            .collect::<Vec<_>>(),
        (rows + 2, k),
        &Device::Cpu,
    )?;
    let x = full.narrow(0, 1, rows)?;
    assert_eq!(x.layout().start_offset(), k);
    let y = forward_routed(&w, &x)?.to_vec2::<f32>()?;
    assert_eq!(y.len(), rows);
    for (r, values) in y.iter().enumerate() {
        let one = forward_routed(&w, &x.narrow(0, r, 1)?)?.to_vec2::<f32>()?;
        assert_eq!(
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            one[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let input = x.i(r)?.to_vec1::<f32>()?;
        for (value, weight) in values.iter().zip(&wd) {
            let terms = weight
                .iter()
                .zip(&input)
                .map(|(&a, &b)| a as f64 * b as f64)
                .collect::<Vec<_>>();
            let sum: f64 = terms.iter().sum();
            let bound = 2e-6 + 2e-6 * terms.iter().map(|v| v.abs()).sum::<f64>();
            assert!((*value as f64 - sum).abs() <= bound);
        }
    }
    assert!(project_q5(&w, &x.to_dtype(DType::F16)?)?.is_none());
    let odd = QMatMul::from_qtensor(QTensor::quantize(
        &Tensor::ones((8, 32), DType::F32, &Device::Cpu)?,
        GgmlDType::Q5_0,
    )?)?;
    assert!(!supported_routed(&odd));
    Ok(())
}
