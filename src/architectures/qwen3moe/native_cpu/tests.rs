use super::*;
use candle_core::Device;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use candle_core::IndexOp;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn request_k_rows_preserve_native_families_offsets_and_arbitrary_chunks() -> Result<()> {
    use candle_core::quantized::{GgmlDType, QTensor};
    let bits = |x: Tensor| -> Result<Vec<u32>> {
        Ok(x.flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    };
    let width = 256;
    let rows = 17;
    let x = Tensor::from_vec(
        (0..19 * width)
            .map(|i| ((i * 19 % 137) as f32 - 68.) / 17.)
            .collect::<Vec<_>>(),
        (1, 19, width),
        &Device::Cpu,
    )?
    .narrow(1, 1, rows)?;
    assert_eq!(x.layout().start_offset(), width);
    let original = bits(x.clone())?;
    for dtype in [GgmlDType::Q4K, GgmlDType::Q5K, GgmlDType::Q6K] {
        let w = Tensor::from_vec(
            (0..16 * width)
                .map(|i| ((i * 7 % 113) as f32 - 56.) / 29.)
                .collect::<Vec<_>>(),
            (16, width),
            &Device::Cpu,
        )?;
        let w = QMatMul::from_qtensor(QTensor::quantize(&w, dtype)?)?;
        assert!(supported_k(&w));
        for end in [1usize, 2, 3, 4, 5, 7, 8, 12, 13, 16, 17] {
            let cutoff = end / 4 * 4;
            let whole = bits(
                project_rows(
                    &w,
                    &x,
                    Some(RowPolicy {
                        start: 0,
                        seq_len: rows,
                        m4_prefix_end: cutoff,
                    }),
                )?
                .unwrap(),
            )?;
            for size in [1usize, 5, 7, 32, 128] {
                let mut joined = Vec::new();
                let mut start = 0;
                while start < rows {
                    let n = size.min(rows - start);
                    joined.extend(bits(
                        project_rows(
                            &w,
                            &x.narrow(1, start, n)?,
                            Some(RowPolicy {
                                start,
                                seq_len: n,
                                m4_prefix_end: cutoff,
                            }),
                        )?
                        .unwrap(),
                    )?);
                    start += n;
                }
                assert_eq!(joined, whole, "dtype{dtype:?},end{end},chunk{size}");
            }
            for row in 0..rows {
                let input = x.narrow(1, row, 1)?;
                let expected = if row < cutoff {
                    project(&w, &input)?.unwrap()
                } else {
                    forward_routed(&w, &input)?
                };
                assert_eq!(bits(expected)?, whole[row * 16..(row + 1) * 16]);
            }
        }
        let head = x.narrow(1, 0, 1)?.reshape((1, width))?;
        assert_eq!(
            bits(forward_rows(
                &w,
                &head,
                Some(RowPolicy {
                    start: 0,
                    seq_len: 1,
                    m4_prefix_end: 0
                })
            )?)?,
            bits(forward_routed(&w, &head)?)?
        );
        assert!(
            project_rows(
                &w,
                &x,
                Some(RowPolicy {
                    start: usize::MAX,
                    seq_len: rows,
                    m4_prefix_end: 0
                })
            )
            .is_err()
        );
        assert!(
            project_rows(
                &w,
                &x,
                Some(RowPolicy {
                    start: 0,
                    seq_len: 0,
                    m4_prefix_end: 0
                })
            )
            .is_err()
        );
    }
    let w = Tensor::ones((16, width), DType::F32, &Device::Cpu)?;
    for dtype in [GgmlDType::Q8_0, GgmlDType::Q4_0] {
        let w = QMatMul::from_qtensor(QTensor::quantize(&w, dtype)?)?;
        assert!(!supported_k(&w));
        assert_eq!(
            bits(forward_rows(
                &w,
                &x,
                Some(RowPolicy {
                    start: 0,
                    seq_len: rows,
                    m4_prefix_end: 12
                })
            )?)?,
            bits(forward(&w, &x)?)?
        );
    }
    assert_eq!(bits(x)?, original);
    Ok(())
}
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn qwen_q8_offset_and_chunk_bits() -> Result<()> {
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
    for (i, expected) in y.iter().enumerate() {
        let row = project(&w, &view.i(i)?.unsqueeze(0)?)?
            .unwrap()
            .to_vec2::<f32>()?;
        assert_eq!(
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
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
fn qwen_norm_bad_geometry_rejected() -> Result<()> {
    let x = Tensor::ones((1, 64), DType::F32, &Device::Cpu)?;
    let w = Tensor::ones(32, DType::F32, &Device::Cpu)?;
    assert!(rms_norm(&x, &w, 1e-6).is_err());
    Ok(())
}

#[test]
fn qwen_norm_views_chunks_and_f64() -> Result<()> {
    for width in [128, 2048] {
        let values: Vec<f32> = (0..7 * width)
            .map(|i| ((i * 139 % 997) as f32 - 498.) / 113.)
            .collect();
        let weights: Vec<f32> = (0..width + 2)
            .map(|i| ((i * 37 % 197) as f32 + 1.) / 173.)
            .collect();
        let base = Tensor::from_vec(values, (7, width), &Device::Cpu)?;
        let x = base.narrow(0, 1, 5)?;
        let w = Tensor::from_vec(weights, width + 2, &Device::Cpu)?.narrow(0, 1, width)?;
        let y = rms_norm(&x, &w, 1e-6)?.to_vec2::<f32>()?;
        let xv = x.to_vec2::<f32>()?;
        let wv = w.to_vec1::<f32>()?;
        for r in 0..5 {
            let scale = 1.
                / (xv[r].iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / width as f64 + 1e-6)
                    .sqrt();
            for c in 0..width {
                let expected = f64::from(xv[r][c]) * scale * f64::from(wv[c]);
                assert!((f64::from(y[r][c]) - expected).abs() <= 2e-6 * (1. + expected.abs()));
            }
            let one = rms_norm(&x.narrow(0, r, 1)?, &w, 1e-6)?.to_vec2::<f32>()?;
            assert_eq!(
                y[r].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                one[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        let strided = x.t()?;
        let weight = Tensor::ones(5, DType::F32, &Device::Cpu)?;
        assert_eq!(
            rms_norm(&strided, &weight, 1e-6)?.to_vec2::<f32>()?,
            rms_norm(&strided.contiguous()?, &weight, 1e-6)?.to_vec2::<f32>()?
        );
        assert!(rms_norm(&x, &w, f64::NAN).is_err());
    }
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn routed_swiglu_views_chunks_and_f64() -> Result<()> {
    let width = 768;
    let gate = Tensor::from_vec(
        (0..5 * width)
            .map(|i| ((i * 139 % 997) as f32 - 498.) / 17.)
            .collect::<Vec<_>>(),
        (5, width),
        &Device::Cpu,
    )?
    .narrow(0, 1, 3)?;
    let up = Tensor::from_vec(
        (0..6 * width)
            .map(|i| ((i * 73 % 431) as f32 - 215.) / 19.)
            .collect::<Vec<_>>(),
        (6, width),
        &Device::Cpu,
    )?
    .narrow(0, 2, 3)?;
    assert_eq!(gate.layout().start_offset(), width);
    assert_eq!(up.layout().start_offset(), 2 * width);
    let y = swiglu(&gate, &up)?.to_vec2::<f32>()?;
    let gv = gate.to_vec2::<f32>()?;
    let uv = up.to_vec2::<f32>()?;
    for row in 0..3 {
        let one = swiglu(&gate.narrow(0, row, 1)?, &up.narrow(0, row, 1)?)?.to_vec2::<f32>()?;
        assert_eq!(
            y[row].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            one[0].iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        for col in 0..width {
            let g = f64::from(gv[row][col]);
            let u = f64::from(uv[row][col]);
            let expected = g / (1. + (-g).exp()) * u;
            assert!((f64::from(y[row][col]) - expected).abs() <= 2e-6 * (1. + expected.abs()));
        }
    }
    let strided = Tensor::ones((8, 12), DType::F32, &Device::Cpu)?.t()?;
    assert_eq!(
        swiglu(&strided, &strided)?.to_vec2::<f32>()?,
        swiglu(&strided.contiguous()?, &strided.contiguous()?)?.to_vec2::<f32>()?
    );
    assert!(swiglu(&gate, &up.to_dtype(DType::F16)?).is_err());
    assert!(swiglu(&gate, &up.narrow(0, 0, 1)?).is_err());
    let odd = Tensor::ones((1, 7), DType::F32, &Device::Cpu)?;
    assert!(swiglu(&odd, &odd).is_err());
    let empty = Tensor::zeros((0, 768), DType::F32, &Device::Cpu)?;
    assert_eq!(swiglu(&empty, &empty)?.dims(), [0, 768]);
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn mixed_k_views_chunks_and_routes() -> Result<()> {
    use candle_core::quantized::{GgmlDType, QTensor};
    for dtype in [GgmlDType::Q4K, GgmlDType::Q5K, GgmlDType::Q6K] {
        let w = Tensor::from_vec(
            (0..8 * 256)
                .map(|i| ((i * 71 % 997) as f32 - 498.) / 113.)
                .collect::<Vec<_>>(),
            (8, 256),
            &Device::Cpu,
        )?;
        let w = QMatMul::from_qtensor(QTensor::quantize(&w, dtype)?)?;
        assert!(supported(&w));
        let x = Tensor::from_vec(
            (0..11 * 256)
                .map(|i| ((i * 139 % 1009) as f32 - 504.) / 97.)
                .collect::<Vec<_>>(),
            (11, 256),
            &Device::Cpu,
        )?
        .narrow(0, 1, 9)?;
        assert_eq!(x.layout().start_offset(), 256);
        for routed in [false, true] {
            let full = project_k(&w, &x, routed, None)?.unwrap().to_vec2::<f32>()?;
            for chunk in [1, 2, 4, 8] {
                let mut parts = Vec::new();
                for start in (0..9).step_by(chunk) {
                    parts.extend(
                        project_k(&w, &x.narrow(0, start, chunk.min(9 - start))?, routed, None)?
                            .unwrap()
                            .to_vec2::<f32>()?,
                    );
                }
                assert_eq!(
                    full.iter()
                        .flatten()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    parts
                        .iter()
                        .flatten()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
        assert!(project_k(&w, &x.to_dtype(DType::F16)?, false, None)?.is_none());
        assert!(project_k(&w, &x.t()?, false, None)?.is_none());
    }
    Ok(())
}
