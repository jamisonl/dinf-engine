use super::*;
use candle_core::{DType, Device};

fn raw(
    shape: Vec<usize>,
    dtype: DeepSeek4DType,
    bytes: Vec<u8>,
    device: &Device,
) -> Result<Arc<RawTensor>> {
    let len = bytes.len();
    Ok(Arc::new(RawTensor::new(
        shape,
        dtype,
        Tensor::from_vec(bytes, len, device)?,
    )?))
}

#[test]
fn rejects_rows_that_split_quant_blocks_and_overflowing_shapes() -> Result<()> {
    for (dtype, shape, bytes) in [
        (DeepSeek4DType::Q8_0, vec![2, 16], 34),
        (DeepSeek4DType::Q4K, vec![2, 128], 144),
        (DeepSeek4DType::IQ3XXS, vec![2, 128], 98),
        (DeepSeek4DType::IQ1S, vec![2, 128], 50),
        (DeepSeek4DType::F32, vec![usize::MAX, 2], 0),
    ] {
        assert!(DeepSeek4QMatMul::new(raw(shape, dtype, vec![0; bytes], &Device::Cpu)?).is_err());
    }
    Ok(())
}

#[test]
fn dense_cached_and_uncached_weights_match_for_all_supported_types() -> Result<()> {
    check_dense_types(&Device::Cpu)?;
    #[cfg(target_os = "macos")]
    if let Some(device) = crate::test_metal_device() {
        check_dense_types(&device)?;
    }
    Ok(())
}

fn check_dense_types(device: &Device) -> Result<()> {
    for dtype in [
        DeepSeek4DType::F32,
        DeepSeek4DType::F16,
        DeepSeek4DType::BF16,
        DeepSeek4DType::I32,
    ] {
        let bytes = [1i32, -2, 3, -4, 5, -6]
            .into_iter()
            .flat_map(|v| match dtype {
                DeepSeek4DType::F32 => (v as f32).to_le_bytes().to_vec(),
                DeepSeek4DType::F16 => half::f16::from_f32(v as f32).to_le_bytes().to_vec(),
                DeepSeek4DType::BF16 => half::bf16::from_f32(v as f32).to_le_bytes().to_vec(),
                DeepSeek4DType::I32 => v.to_le_bytes().to_vec(),
                _ => unreachable!(),
            })
            .collect();
        let weights = raw(vec![2, 3], dtype, bytes, device)?;
        for matmul in [
            DeepSeek4QMatMul::new(weights.clone())?,
            DeepSeek4QMatMul::new_on_device(weights, device)?,
        ] {
            let x = Tensor::from_slice(&[2f32, 1., -1.], (1, 3), device)?;
            assert_eq!(
                matmul.forward(&x)?.to_vec2::<f32>()?,
                vec![vec![-3., 3.]],
                "{dtype:?}"
            );
            let x = x.reshape((1, 1, 3))?;
            assert_eq!(
                matmul.forward(&x)?.to_vec3::<f32>()?,
                vec![vec![vec![-3., 3.]]],
                "{dtype:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn rejects_bad_weight_storage_and_bad_inputs() -> Result<()> {
    for shape in [vec![3], vec![1, 1, 3], vec![0, 3], vec![1, 0]] {
        assert!(
            DeepSeek4QMatMul::new(raw(shape, DeepSeek4DType::F32, vec![], &Device::Cpu)?).is_err()
        );
    }
    assert!(
        DeepSeek4QMatMul::new(raw(
            vec![2, 3],
            DeepSeek4DType::F32,
            vec![0; 20],
            &Device::Cpu
        )?)
        .is_err()
    );
    let matmul = DeepSeek4QMatMul::new(raw(
        vec![2, 3],
        DeepSeek4DType::F32,
        vec![0; 24],
        &Device::Cpu,
    )?)?;
    for x in [
        Tensor::zeros(3, DType::F32, &Device::Cpu)?,
        Tensor::zeros((1, 2), DType::F32, &Device::Cpu)?,
        Tensor::zeros((1, 3), DType::F16, &Device::Cpu)?,
    ] {
        assert!(matmul.forward(&x).is_err());
    }
    Ok(())
}

#[test]
fn grouped_v41_geometry_rejects_invalid_groups_shapes_and_overflow_cpu() -> Result<()> {
    for (out, width, groups, dims) in [
        (16, 32, 0, vec![0, 32]),
        (16, 32, 3, vec![3, 32]),
        (24, 32, 8, vec![8, 32]),
        (0, 32, 8, vec![8, 32]),
        (16, 0, 8, vec![8, 0]),
        (16, 16, 8, vec![8, 16]),
        (16, 32, 8, vec![1, 8, 32]),
        (16, 32, 8, vec![8, 64]),
        (16, 32, 8, vec![7, 32]),
        (131_076, 32, 2, vec![2, 32]),
        (usize::MAX - 1, 32, 1, vec![1, 32]),
        (16, usize::MAX - 31, 8, vec![8, usize::MAX - 31]),
        (268_435_456, 32, 134_217_728, vec![134_217_728, 32]),
        (
            (u32::MAX as usize).saturating_add(1),
            32,
            65_536,
            vec![65_536, 32],
        ),
    ] {
        assert!(
            GroupedV41Geometry::new(out, width, &dims, groups).is_err(),
            "out={out} width={width} groups={groups} dims={dims:?}"
        );
    }
    for (out, width, groups) in [(8192, 4096, 8), (48, 1056, 8), (2, 32, 1), (65_536, 32, 1)] {
        let geometry = GroupedV41Geometry::new(out, width, &[groups, width], groups)?;
        assert_eq!(geometry.in_dim as usize, width);
        assert_eq!(geometry.out_dim as usize, out);
        assert_eq!(geometry.rows_per_group as usize, out / groups);
        assert_eq!(geometry.row_bytes as usize, width / 32 * 34);
        assert_eq!(geometry.weight_bytes, out * width / 32 * 34);
        assert_eq!(geometry.input_elements, groups * width);
    }
    Ok(())
}

#[test]
fn grouped_v41_public_contract_and_unsupported_backend_cpu() -> Result<()> {
    let weights = raw(
        vec![16, 32],
        DeepSeek4DType::Q8_0,
        vec![0; 16 * 34],
        &Device::Cpu,
    )?;
    let input = Tensor::zeros((8, 32), DType::F32, &Device::Cpu)?;
    let ordinary = DeepSeek4QMatMul::new(weights.clone())?;
    assert!(
        ordinary
            .forward_grouped_v41(&input, 8)
            .unwrap_err()
            .to_string()
            .contains("opt-in")
    );
    let mat = DeepSeek4QMatMul::new_v41_on_device(weights, &Device::Cpu)?;
    assert!(
        mat.forward_grouped_v41(&input, 8)
            .unwrap_err()
            .to_string()
            .contains("Metal backend")
    );
    for x in [
        Tensor::zeros((8, 32), DType::F16, &Device::Cpu)?,
        Tensor::zeros((8, 32), DType::BF16, &Device::Cpu)?,
        Tensor::zeros((32, 8), DType::F32, &Device::Cpu)?.t()?,
    ] {
        assert!(
            mat.forward_grouped_v41(&x, 8)
                .unwrap_err()
                .to_string()
                .contains("contiguous F32")
        );
    }
    let dense = DeepSeek4QMatMul::new_v41_on_device(
        raw(
            vec![16, 32],
            DeepSeek4DType::F32,
            vec![0; 16 * 32 * 4],
            &Device::Cpu,
        )?,
        &Device::Cpu,
    )?;
    assert!(
        dense
            .forward_grouped_v41(&input, 8)
            .unwrap_err()
            .to_string()
            .contains("Q8_0")
    );
    Ok(())
}

// These fixtures and tests use CPU storage only, never test_metal_device.
#[test]
fn grouped_iq1_geometry_cpu() -> Result<()> {
    for count in [0, 13, usize::MAX] {
        assert!(GroupedIq1Geometry::new(count, &[1, 5120]).is_err());
    }
    for dims in [
        vec![],
        vec![5120],
        vec![1, 1, 5120],
        vec![2, 5120],
        vec![1, 256],
        vec![0, 5120],
        vec![usize::MAX, usize::MAX],
    ] {
        assert!(GroupedIq1Geometry::new(12, &dims).is_err());
    }
    for count in 1..=12 {
        let g = GroupedIq1Geometry::new(count, &[1, 5120])?;
        assert_eq!(g.output_elements * 4, count * 9216);
        assert_eq!(g.weight_bytes, 2_304_000);
        g.validate_weight(&[2304, 5120], DeepSeek4DType::IQ1S, 2_304_000)?;
        for dims in [
            vec![2304],
            vec![1, 2304, 5120],
            vec![5120, 2304],
            vec![4, 256],
            vec![usize::MAX, usize::MAX],
        ] {
            assert!(
                g.validate_weight(&dims, DeepSeek4DType::IQ1S, g.weight_bytes)
                    .is_err()
            );
        }
        for bytes in [0, g.weight_bytes - 1, g.weight_bytes + 1, usize::MAX] {
            assert!(
                g.validate_weight(&[2304, 5120], DeepSeek4DType::IQ1S, bytes)
                    .is_err()
            );
        }
        assert!(
            g.validate_weight(&[2304, 5120], DeepSeek4DType::IQ1M, g.weight_bytes)
                .is_err()
        );
    }
    Ok(())
}

#[test]
fn grouped_iq1_span_cpu() -> Result<()> {
    use candle_core::{Layout, Shape};
    let input = |offset| Layout::new(Shape::from((1, 5120)), vec![5120, 1], offset);
    assert_eq!(grouped_iq1_span(&input(3), DType::F32, 5120, 5123 * 4)?, 12);
    let weight = Layout::new(Shape::from(2_304_000), vec![1], 50);
    assert_eq!(
        grouped_iq1_span(&weight, DType::U8, 2_304_000, 2_304_050)?,
        50
    );
    assert!(grouped_iq1_span(&weight, DType::U8, 2_304_000, 2_304_049).is_err());
    let unaligned = Layout::new(Shape::from(2_304_000), vec![1], 1);
    assert!(
        grouped_iq1_span(&unaligned, DType::U8, 2_304_000, 2_304_001)
            .unwrap_err()
            .to_string()
            .contains("two-byte aligned")
    );
    assert!(grouped_iq1_span(&input(3), DType::F32, 5120, 5123 * 4 - 1).is_err());
    assert!(grouped_iq1_span(&input(usize::MAX), DType::F32, 5120, usize::MAX).is_err());
    assert!(grouped_iq1_span(&input(usize::MAX / 4), DType::F32, 5120, usize::MAX).is_err());
    let strided = Layout::new(Shape::from((1, 5120)), vec![10240, 2], 0);
    assert!(grouped_iq1_span(&strided, DType::F32, 5120, 40960).is_err());
    assert!(grouped_iq1_span(&input(0), DType::F32, 5119, 20480).is_err());
    let overflow = Layout::new(Shape::from((usize::MAX, 2)), vec![2, 1], 0);
    assert!(grouped_iq1_span(&overflow, DType::U8, 5120, usize::MAX).is_err());
    let bytes_overflow = Layout::new(Shape::from(usize::MAX), vec![1], 0);
    assert!(grouped_iq1_span(&bytes_overflow, DType::F32, usize::MAX, usize::MAX).is_err());
    Ok(())
}

// Construct deliberately invalid raw metadata as well as valid native weights.
// This tests the grouped entry point independently of the ordinary constructor.
fn grouped_iq1_test_matrix(
    shape: Vec<usize>,
    dtype: DeepSeek4DType,
    bytes: Tensor,
) -> Result<DeepSeek4QMatMul> {
    Ok(DeepSeek4QMatMul {
        weights: Arc::new(RawTensor::new(shape, dtype, bytes)?),
        dense_weights: None,
        out_dim: 2304,
        in_dim: 5120,
        v41_q8: false,
    })
}

#[test]
fn grouped_iq1_contract_cpu() -> Result<()> {
    let input = Tensor::zeros((1, 5120), DType::F32, &Device::Cpu)?;
    let bytes = Tensor::zeros(2_304_000, DType::U8, &Device::Cpu)?;
    let mat = grouped_iq1_test_matrix(vec![2304, 5120], DeepSeek4DType::IQ1S, bytes.clone())?;
    let error = |x: &Tensor, ms: &[&DeepSeek4QMatMul], text: &str| {
        let err = DeepSeek4QMatMul::forward_grouped_iq1(x, ms).unwrap_err();
        assert!(err.to_string().contains(text), "{err:#}");
    };
    error(&input, &[], "count");
    error(&input, &[&mat; 13], "count");
    // Valid and offset CPU operands reach an explicit no-fallback rejection.
    error(&input, &[&mat; 12], "Metal backend");
    let offset_input = Tensor::zeros(5123, DType::F32, &Device::Cpu)?
        .narrow(0, 3, 5120)?
        .reshape((1, 5120))?;
    let offset_mat = grouped_iq1_test_matrix(
        vec![2304, 5120],
        DeepSeek4DType::IQ1S,
        Tensor::zeros(2_304_050, DType::U8, &Device::Cpu)?.narrow(0, 50, 2_304_000)?,
    )?;
    error(&offset_input, &[&offset_mat], "Metal backend");
    for x in [
        input.reshape(5120)?,
        Tensor::zeros((2, 5120), DType::F32, &Device::Cpu)?,
    ] {
        error(&x, &[&mat], "shape");
    }
    for x in [
        input.to_dtype(DType::F16)?,
        input.to_dtype(DType::BF16)?,
        Tensor::zeros((5120, 2), DType::F32, &Device::Cpu)?
            .narrow(1, 0, 1)?
            .t()?,
    ] {
        error(&x, &[&mat], "contiguous F32");
    }
    for (shape, dtype, storage, text) in [
        (
            vec![1, 2304, 5120],
            DeepSeek4DType::IQ1S,
            bytes.clone(),
            "native geometry",
        ),
        (
            vec![2304, 5120],
            DeepSeek4DType::Q8_0,
            bytes.clone(),
            "IQ1_S",
        ),
        (
            vec![2304, 5120],
            DeepSeek4DType::IQ1S,
            bytes.narrow(0, 0, 2_303_999)?,
            "span",
        ),
        (
            vec![2304, 5120],
            DeepSeek4DType::IQ1S,
            Tensor::zeros((2_304_000, 2), DType::U8, &Device::Cpu)?.narrow(1, 0, 1)?,
            "contiguous packed",
        ),
    ] {
        let bad = grouped_iq1_test_matrix(shape, dtype, storage)?;
        // Check a bad final operand, not just the first binding.
        let mut matrices = [&mat; 12];
        matrices[11] = &bad;
        error(&input, &matrices, text);
    }
    Ok(())
}

#[cfg(target_os = "macos")]
mod metal {
    use super::super::metal::{KernelKind, forward_with_kernel};
    use super::*;

    #[test]
    fn coder_grouped_q5_projection_matches_independent_cpu_rows() -> Result<()> {
        use candle_core::quantized::{GgmlDType, QTensor};
        let device = Device::new_metal(0)?;
        let source = Tensor::from_vec(
            (0..17 * 256)
                .map(|i| ((i * 13 % 61) as f32 - 30.) / 16.)
                .collect::<Vec<_>>(),
            (17, 256),
            &Device::Cpu,
        )?;
        let q = QTensor::quantize(&source, GgmlDType::Q5K)?;
        let reference = q
            .dequantize(&Device::Cpu)?
            .to_dtype(DType::F16)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        let weights = raw(
            vec![17, 256],
            DeepSeek4DType::Q5K,
            q.data()?.to_vec(),
            &device,
        )?
        .slice_rows(1, 16)?;
        let mat = DeepSeek4QMatMul::new_v41_on_device(Arc::new(weights), &device)?;
        let host = (0..8 * 256)
            .map(|i| ((i * 7 % 47) as f32 - 23.) / 17.)
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(host.clone(), (8, 256), &device)?;
        let actual = mat.forward_grouped_packed(&input, 8)?.to_vec2::<f32>()?;
        for group in 0..8 {
            for row in 0..2 {
                let expected: f64 = reference[1 + 2 * group + row]
                    .iter()
                    .zip(&host[group * 256..(group + 1) * 256])
                    .map(|(&w, &x)| f64::from(w) * f64::from(x))
                    .sum();
                assert!((f64::from(actual[0][2 * group + row]) - expected).abs() < 0.002);
            }
        }
        assert!(mat.forward_grouped_packed(&input, 0).is_err());
        device.synchronize()?;
        Ok(())
    }

    const QUANTS: [DeepSeek4DType; 11] = [
        DeepSeek4DType::Q4K,
        DeepSeek4DType::IQ2XXS,
        DeepSeek4DType::Q5K,
        DeepSeek4DType::IQ2XS,
        DeepSeek4DType::IQ3XXS,
        DeepSeek4DType::IQ3S,
        DeepSeek4DType::IQ1M,
        DeepSeek4DType::IQ1S,
        DeepSeek4DType::Q6K,
        DeepSeek4DType::Q8_0,
        DeepSeek4DType::MXFP4,
    ];

    fn q4k_scalar(block: &[u8], i: usize) -> f64 {
        assert_eq!(block.len(), 144);
        let d = f64::from(half::f16::from_le_bytes([block[0], block[1]]).to_f32());
        let m = f64::from(half::f16::from_le_bytes([block[2], block[3]]).to_f32());
        let s = &block[4..16];
        let g = i / 32;
        let (scale, min) = if g < 4 {
            (s[g] & 63, s[g + 4] & 63)
        } else {
            (
                (s[g + 4] & 15) | ((s[g - 4] >> 6) << 4),
                (s[g + 4] >> 4) | ((s[g] >> 6) << 4),
            )
        };
        let q = block[16 + (i / 64) * 32 + i % 32];
        let q = if i % 64 < 32 { q & 15 } else { q >> 4 };
        d * f64::from(scale) * f64::from(q) - m * f64::from(min)
    }

    fn q4k_fixture(width: usize, rows: usize, scale: u16, kind: usize) -> Vec<u8> {
        let mut data = Vec::new();
        for row in 0..rows {
            for block in 0..width / 256 {
                let mut b = [0u8; 144];
                b[..2].copy_from_slice(&scale.to_le_bytes());
                let min_scale = if row % 3 == 0 { 0 } else { scale };
                b[2..4].copy_from_slice(&min_scale.to_le_bytes());
                for (i, v) in b[4..16].iter_mut().enumerate() {
                    *v = if row == 0 {
                        255
                    } else {
                        ((i * 71 + row * 13 + block * 29 + kind * 53) % 256) as u8
                    };
                }
                for (i, v) in b[16..].iter_mut().enumerate() {
                    *v = ((i * 23 + row * 71 + block * 19 + kind * 31) % 256) as u8;
                }
                data.extend(b);
            }
        }
        data
    }

    #[test]
    fn q4k_finite_scale_boundaries_keep_f32_weights() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for scale in [0u16, 1, 15, 0x3ff, 0x400, 0x3c01, 0x7bff, 0x8001, 0xfbff] {
            for width in [256usize, 4096] {
                let n = 5;
                let packed = q4k_fixture(width, n + 1, scale, 0);
                let weights = Arc::new(
                    raw(
                        vec![n + 1, width],
                        DeepSeek4DType::Q4K,
                        packed.clone(),
                        &device,
                    )?
                    .slice_rows(1, n)?,
                );
                let matmul = DeepSeek4QMatMul::new_on_device(weights.clone(), &device)?;
                assert!(Arc::ptr_eq(&matmul.weights, &weights));
                assert!(matmul.dense_weights.is_none());
                let amplitude = if scale & 0x7fff > 0 && scale & 0x7fff < 0x400 {
                    1_048_576.0f32
                } else {
                    0.25
                };
                let tokens = if width == 256 { 256 } else { 3 };
                let host = (0..tokens * width)
                    .map(|i| {
                        if width == 256 {
                            if i / width == i % width {
                                amplitude
                            } else {
                                0.0
                            }
                        } else {
                            amplitude * ((i * 13 + i / width * 7) % 31) as f32 / 16.0 - amplitude
                        }
                    })
                    .collect::<Vec<_>>();
                let tensor = Tensor::from_vec(host.clone(), (tokens, width), &device)?;
                for input in [
                    tensor.reshape((1, tokens, width))?,
                    tensor.t()?.contiguous()?.t()?,
                ] {
                    for kind in [KernelKind::Serial, KernelKind::Simd] {
                        for decode_rows in [false, true] {
                            let actual = super::super::metal::forward_with_mode(
                                weights.bytes(),
                                &input,
                                DeepSeek4DType::Q4K,
                                n,
                                width,
                                kind,
                                decode_rows,
                            )?
                            .flatten_all()?
                            .to_vec1::<f32>()?;
                            for (index, &value) in actual.iter().enumerate() {
                                let token = index / n;
                                let row = index % n + 1;
                                let mut expected = 0.0f64;
                                let mut magnitude = 0.0f64;
                                for column in 0..width {
                                    let block =
                                        &packed[(row * width / 256 + column / 256) * 144..][..144];
                                    let term = q4k_scalar(block, column % 256)
                                        * f64::from(host[token * width + column]);
                                    expected += term;
                                    magnitude += term.abs();
                                }
                                assert!(
                                    value.is_finite()
                                        && (f64::from(value) - expected).abs()
                                            <= 1e-4 + 2e-5 * magnitude,
                                    "Q4_K scale={scale:#x} width={width} {kind:?} rows={decode_rows} index={index} actual={value} expected={expected}"
                                );
                            }
                        }
                    }
                }
            }
        }
        device.synchronize()?;
        Ok(())
    }

    // Numerical fixture from GGML iq1s_grid[256 * high + low], low=0..3,
    // in scalar element order (not Metal's offset/nibble representation).
    // Reference: llama.cpp b72050d03ae14bb3cb5138ecefab3f3f65157d31,
    // ggml/src/ggml-common.h; the scalar rule is dequantize_row_iq1_s in
    // ggml/src/ggml-quants.c. Only numerical fixture data is reproduced here.
    const IQ1S_SCALAR_GRIDS: [[i8; 8]; 32] = [
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [1, -1, -1, -1, -1, -1, -1, -1],
        [0, 0, -1, -1, -1, -1, -1, -1],
        [-1, 1, -1, -1, -1, -1, -1, -1],
        [-1, 0, 1, 0, 1, -1, 0, -1],
        [1, 0, 1, 0, 1, -1, 0, -1],
        [0, 0, -1, 1, 1, -1, 0, -1],
        [0, -1, 0, 1, 1, -1, 0, -1],
        [0, 0, 1, -1, -1, 0, 1, -1],
        [-1, 0, -1, 0, -1, 0, 1, -1],
        [0, 0, -1, 0, -1, 0, 1, -1],
        [1, 0, -1, 0, -1, 0, 1, -1],
        [-1, 0, 1, -1, 1, 0, -1, 0],
        [0, 1, 1, -1, 1, 0, -1, 0],
        [-1, -1, -1, 0, 1, 0, -1, 0],
        [1, -1, -1, 0, 1, 0, -1, 0],
        [0, 1, -1, 0, 0, 0, 0, 0],
        [-1, -1, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, 0],
        [1, -1, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, -1, 0, 1, 0],
        [-1, 0, 0, 0, -1, 0, 1, 0],
        [0, 0, 0, 0, -1, 0, 1, 0],
        [1, 0, 0, 0, -1, 0, 1, 0],
        [1, -1, 0, 1, 0, 0, -1, 1],
        [0, 0, 0, 1, 0, 0, -1, 1],
        [-1, 1, 0, 1, 0, 0, -1, 1],
        [1, 0, 1, 1, 0, 0, -1, 1],
        [1, -1, 0, 1, 1, 0, 0, 1],
        [-1, 0, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0, 1],
        [-1, 0, 1, 1, 1, 0, 0, 1],
    ];

    fn iq1s_scalar_reference(bytes: &[u8]) -> Vec<f32> {
        assert_eq!(bytes.len(), 50);
        let d = half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f32();
        // Index individual scalar elements, independently of Metal's 16-value
        // chunks and independently of the fixture encoder's nested loops.
        (0..256)
            .map(|element| {
                let group = element / 32;
                let h = u16::from_le_bytes([bytes[34 + 2 * group], bytes[35 + 2 * group]]);
                let high = usize::from((h >> (3 * ((element / 8) % 4))) & 7);
                let low = usize::from(bytes[2 + element / 8]);
                assert!(low < 4, "grid is outside the scalar fixture");
                let grid = IQ1S_SCALAR_GRIDS[4 * high + low][element % 8];
                let delta = if h & 0x8000 == 0 { 0.125 } else { -0.125 };
                d * f32::from(2 * ((h >> 12) & 7) + 1) * (f32::from(grid) + delta)
            })
            .collect()
    }

    // Small, independently decoded fixtures. Vary entries, scales and signs
    // to expose row/lane errors; IQ1_S also exercises every high grid bit.
    fn block(dtype: DeepSeek4DType, index: usize) -> Result<(Vec<u8>, Vec<f32>)> {
        let (n, size) = dtype.block_layout().unwrap();
        let mut bytes = vec![0; size];
        let scale = (1 + index % 3) as f32 / 8.;
        let mut values = vec![0.; n];
        match dtype {
            DeepSeek4DType::IQ2XXS => {
                // GGML scalar IQ2_XXS rule, restricted to two known grid
                // entries and sign codes. Vary all eight group scales.
                bytes[..2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
                for g in 0..8 {
                    let sc = (g + index) % 5;
                    let mut signs = (sc as u32) << 28;
                    for j in 0..4 {
                        let code = (g + j + index) % 2;
                        signs |= (code as u32) << (7 * j);
                        let grid = (g + 2 * j + index) % 2;
                        bytes[2 + 8 * g + j] = grid as u8;
                        for k in 0..8 {
                            let sign = if code == 1 && (k == 0 || k == 7) {
                                -1.
                            } else {
                                1.
                            };
                            let value = if grid == 1 && k == 0 { 43. } else { 8. };
                            values[32 * g + 8 * j + k] =
                                scale * (0.5 + sc as f32) * 0.25 * value * sign;
                        }
                    }
                    bytes[6 + 8 * g..10 + 8 * g].copy_from_slice(&signs.to_le_bytes());
                }
            }
            DeepSeek4DType::IQ2XS => {
                bytes[..2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
                for g in 0..16 {
                    let sc = (g + index) % 5;
                    bytes[66 + g / 2] |= (sc as u8) << (4 * (g % 2));
                    for j in 0..2 {
                        let sign_code = (g + j + index) % 2;
                        let grid = (g + 2 * j + index) % 2;
                        let q = ((sign_code as u16) << 9) | grid as u16;
                        let offset = 2 + (2 * g + j) * 2;
                        bytes[offset..offset + 2].copy_from_slice(&q.to_le_bytes());
                        for k in 0..8 {
                            let sign = if sign_code == 1 && (k == 0 || k == 7) {
                                -1.
                            } else {
                                1.
                            };
                            let value = if grid == 1 && k == 0 { 43. } else { 8. };
                            values[16 * g + 8 * j + k] =
                                scale * (0.5 + sc as f32) * 0.25 * value * sign;
                        }
                    }
                }
            }
            DeepSeek4DType::IQ3XXS => {
                bytes[..2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
                for g in 0..8 {
                    let sc = (g + index) % 5;
                    let mut signs = (sc as u32) << 28;
                    for j in 0..4 {
                        let code = (g + j + index) % 2;
                        signs |= (code as u32) << (7 * j);
                        for h in 0..2 {
                            bytes[2 + 8 * g + 2 * j + h] = ((g + j + h + index) % 2) as u8;
                        }
                        for k in 0..8 {
                            let sign = if code == 1 && (k == 0 || k == 7) {
                                -1.
                            } else {
                                1.
                            };
                            let grid = bytes[2 + 8 * g + 2 * j + k / 4];
                            let value = if grid == 1 && k % 4 == 0 { 20. } else { 4. };
                            values[32 * g + 8 * j + k] =
                                scale * (0.5 + sc as f32) * 0.5 * value * sign;
                        }
                    }
                    bytes[66 + 4 * g..70 + 4 * g].copy_from_slice(&signs.to_le_bytes());
                }
            }
            DeepSeek4DType::IQ3S => {
                bytes[..2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
                for g in 0..8 {
                    let sc = (g + index) % 5;
                    bytes[106 + g / 2] |= (sc as u8) << (4 * (g % 2));
                    for j in 0..4 {
                        let signs = 0xa5u8.rotate_left(((g + j + index) % 8) as u32);
                        bytes[74 + 4 * g + j] = signs;
                        for h in 0..2 {
                            bytes[2 + 8 * g + 2 * j + h] = ((g + j + h + index) % 2) as u8;
                        }
                        for k in 0..8 {
                            let sign = if signs & (1 << k) != 0 { -1. } else { 1. };
                            let grid = bytes[2 + 8 * g + 2 * j + k / 4];
                            let value = if grid == 1 && k % 4 == 0 { 3. } else { 1. };
                            values[32 * g + 8 * j + k] = scale * (1 + 2 * sc) as f32 * value * sign;
                        }
                    }
                }
            }
            DeepSeek4DType::IQ1M => {
                let scale_bits = half::f16::from_f32(scale).to_bits();
                for g in 0..4 {
                    let mut sc = ((scale_bits >> (4 * g)) & 0xf) << 12;
                    for j in 0..4 {
                        let code = (g + j + index) % 5;
                        sc |= (code as u16) << (3 * j);
                        let qh = if (g + j + index).is_multiple_of(2) {
                            0x08
                        } else {
                            0x80
                        };
                        bytes[32 + 4 * g + j] = qh;
                        for h in 0..2 {
                            bytes[8 * g + 2 * j + h] = ((g + j + h + index) % 2) as u8;
                        }
                        for k in 0..16 {
                            let negative_delta = if k < 8 {
                                qh & 0x08 != 0
                            } else {
                                qh & 0x80 != 0
                            };
                            let grid = bytes[8 * g + 2 * j + k / 8];
                            let value = if grid == 1 && k % 8 == 0 { 2. } else { 0. };
                            values[64 * g + 16 * j + k] = scale
                                * (2 * code + 1) as f32
                                * (value + if negative_delta { -1.125 } else { -0.875 });
                        }
                    }
                    bytes[48 + 2 * g..50 + 2 * g].copy_from_slice(&sc.to_le_bytes());
                }
            }
            DeepSeek4DType::IQ1S => {
                assert_eq!((n, size), (256, 50));
                bytes[..2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
                for group in 0..8 {
                    let mut h = (((group + index) % 8) as u16) << 12;
                    if !(group / 2 + index).is_multiple_of(2) {
                        h |= 0x8000;
                    }
                    for grid in 0..4 {
                        bytes[2 + 4 * group + grid] = ((grid + index) % 4) as u8;
                        h |= (((group + 3 * grid + index) % 8) as u16) << (3 * grid);
                    }
                    bytes[34 + 2 * group..36 + 2 * group].copy_from_slice(&h.to_le_bytes());
                }
                values = iq1s_scalar_reference(&bytes);
            }
            DeepSeek4DType::Q4K
            | DeepSeek4DType::Q5K
            | DeepSeek4DType::Q6K
            | DeepSeek4DType::Q8_0 => {
                use candle_core::quantized::{GgmlDType, QTensor};
                let source = Tensor::from_vec(
                    (0..n)
                        .map(|i| ((i * 13 + index * 7) % 61) as f32 / 16. - 2.)
                        .collect::<Vec<_>>(),
                    n,
                    &Device::Cpu,
                )?;
                let dtype = if dtype == DeepSeek4DType::Q4K {
                    GgmlDType::Q4K
                } else if dtype == DeepSeek4DType::Q5K {
                    GgmlDType::Q5K
                } else if dtype == DeepSeek4DType::Q6K {
                    GgmlDType::Q6K
                } else {
                    GgmlDType::Q8_0
                };
                let q = QTensor::quantize(&source, dtype)?;
                bytes = q.data()?.to_vec();
                // Q4_K keeps F32 decoded weights; the older paths retain half.
                let decoded = q.dequantize(&Device::Cpu)?;
                values = if dtype == GgmlDType::Q4K {
                    decoded.to_vec1()?
                } else {
                    decoded
                        .to_dtype(DType::F16)?
                        .to_dtype(DType::F32)?
                        .to_vec1()?
                };
            }
            DeepSeek4DType::MXFP4 => {
                let table = [
                    0., 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5, -2., -3., -4., -6.,
                ];
                bytes[0] = 124 + (index % 4) as u8;
                let scale = 2f32.powi(i32::from(bytes[0]) - 128);
                for i in 0..16 {
                    let lo = (i + index) % 16;
                    let hi = (3 * i + index + 1) % 16;
                    bytes[1 + i] = lo as u8 | ((hi as u8) << 4);
                    values[i] = scale * table[lo];
                    values[16 + i] = scale * table[hi];
                }
            }
            _ => unreachable!(),
        }
        assert!(values.iter().any(|v| *v != 0.));
        Ok((bytes, values))
    }

    #[allow(
        clippy::excessive_precision,
        reason = "Exact binary fractions define quantization boundary fixtures."
    )]
    fn grouped_iq1_weights(device: &Device, offset: bool) -> Result<Vec<DeepSeek4QMatMul>> {
        let scales = [
            0.00006103515625,
            0.03125,
            0.125,
            1.0,
            3.0,
            31.0,
            256.0,
            2048.0,
            16384.0,
            65504.0,
            0.375,
            4096.0,
        ];
        (0..12)
            .map(|slot| {
                let prefix = if offset { (slot + 1) * 50 } else { 0 };
                let mut bytes = vec![0x7f; prefix];
                for row in 0..2304 {
                    for ib in 0..20 {
                        let (mut packed, _) =
                            block(DeepSeek4DType::IQ1S, slot * 101 + row * 7 + ib)?;
                        packed[..2].copy_from_slice(
                            &half::f16::from_f32(scales[(slot + row / 17 + ib) % scales.len()])
                                .to_le_bytes(),
                        );
                        bytes.extend_from_slice(&packed);
                    }
                }
                bytes.extend_from_slice(&[0xff; 50]);
                let len = bytes.len();
                let storage = Tensor::from_vec(bytes, len, device)?.narrow(0, prefix, 2_304_000)?;
                assert_eq!(storage.storage_and_layout().1.start_offset(), prefix);
                DeepSeek4QMatMul::new(Arc::new(RawTensor::new(
                    vec![2304, 5120],
                    DeepSeek4DType::IQ1S,
                    storage,
                )?))
            })
            .collect()
    }

    fn grouped_iq1_input(device: &Device, case: usize, offset: bool) -> Result<Tensor> {
        let boundary = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            1.0039063,
            -1.0039063,
            1.0e-20,
            -1.0e-20,
            1.0e20,
            -1.0e20,
            65504.0,
            -65504.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(1),
            -f32::from_bits(1),
        ];
        let prefix = if offset { 3 } else { 0 };
        let mut data = vec![f32::NAN; prefix];
        data.extend((0..5120).map(|i| match case {
            0 => boundary[(i * 7 + i / 32) % boundary.len()],
            1 => {
                if i % 2 == 0 {
                    16_777_216.0
                } else {
                    -16_777_216.0 + (i % 7) as f32
                }
            }
            _ => ((i * 7919 % 104729) as f32 - 52364.0) / 8192.0,
        }));
        data.push(f32::INFINITY);
        let len = data.len();
        let input = Tensor::from_vec(data, len, device)?
            .narrow(0, prefix, 5120)?
            .reshape((1, 5120))?;
        assert_eq!(input.storage_and_layout().1.start_offset(), prefix);
        Ok(input)
    }

    fn grouped_iq1_require_simd() {
        assert!(
            DeepSeek4QMatMul::iq1_simd_enabled(),
            "grouped IQ1 real-Metal qualification requires SIMD enabled"
        );
    }

    fn grouped_iq1_bits(tensor: &Tensor) -> Result<Vec<u32>> {
        Ok(tensor
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }

    #[test]
    fn grouped_iq1_real_metal_bitwise_native_order_offsets_queued() -> Result<()> {
        grouped_iq1_require_simd();
        let device = Device::new_metal(0)?;
        eprintln!(
            "grouped IQ1 native exactness on {device:?}: 165888 finite bit comparisons, 12 distinct matrices, 3 inputs, 2 offset modes"
        );
        for offset in [false, true] {
            let owners = grouped_iq1_weights(&device, offset)?;
            let matrices = owners.iter().collect::<Vec<_>>();
            let inputs = (0..3)
                .map(|case| grouped_iq1_input(&device, case, offset))
                .collect::<Result<Vec<_>>>()?;
            let mut queued = Vec::new();
            for input in &inputs {
                let expected = matrices
                    .iter()
                    .map(|mat| mat.forward(input))
                    .collect::<Result<Vec<_>>>()?;
                let actual = DeepSeek4QMatMul::forward_grouped_iq1(input, &matrices)?;
                queued.push((actual, expected));
            }
            // All input, packed RawTensor/ExpertTriplet-equivalent owners and
            // outputs remain live through this fence. No early-drop guarantee.
            device.synchronize()?;
            for (actual, expected) in &queued {
                assert_eq!(actual.dims(), &[12, 2304]);
                let references = expected
                    .iter()
                    .map(grouped_iq1_bits)
                    .collect::<Result<Vec<_>>>()?;
                for slot in 0..12 {
                    let bits = grouped_iq1_bits(&actual.narrow(0, slot, 1)?)?;
                    assert!(bits.iter().all(|&v| f32::from_bits(v).is_finite()));
                    assert_eq!(
                        bits, references[slot],
                        "grouped IQ1 finite bits, slot={slot}, offset={offset}"
                    );
                    for other in 0..slot {
                        assert_ne!(
                            references[slot], references[other],
                            "fixture must discriminate expert gate/up order {slot} vs {other}"
                        );
                    }
                }
                assert!(actual.narrow(0, 12, 1).is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn grouped_iq1_real_metal_repeated_weights_and_bounded_counts() -> Result<()> {
        grouped_iq1_require_simd();
        let device = Device::new_metal(0)?;
        let owners = grouped_iq1_weights(&device, true)?;
        let input = grouped_iq1_input(&device, 2, true)?;
        let order = [7, 1, 7, 11, 0, 3, 3, 5, 2, 9, 8, 4];
        let matrices = order.map(|i| &owners[i]);
        let expected = matrices
            .iter()
            .map(|mat| mat.forward(&input))
            .collect::<Result<Vec<_>>>()?;
        let queued = [1, 5, 12]
            .into_iter()
            .map(|count| DeepSeek4QMatMul::forward_grouped_iq1(&input, &matrices[..count]))
            .collect::<Result<Vec<_>>>()?;
        device.synchronize()?;
        for actual in &queued {
            for (slot, reference) in expected.iter().enumerate().take(actual.dim(0)?) {
                assert_eq!(
                    grouped_iq1_bits(&actual.narrow(0, slot, 1)?)?,
                    grouped_iq1_bits(reference)?,
                    "repeated binding slot {slot}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn grouped_iq1_real_metal_nonfinite_classification() -> Result<()> {
        grouped_iq1_require_simd();
        let device = Device::new_metal(0)?;
        let owners = grouped_iq1_weights(&device, true)?;
        let matrices = owners.iter().collect::<Vec<_>>();
        for exceptional in [f32::INFINITY, f32::NEG_INFINITY, f32::from_bits(0x7fc01234)] {
            let mut values = vec![0.0f32; 5120];
            values[123] = exceptional;
            let input = Tensor::from_vec(values, (1, 5120), &device)?;
            let expected = matrices
                .iter()
                .map(|m| m.forward(&input))
                .collect::<Result<Vec<_>>>()?;
            let actual = DeepSeek4QMatMul::forward_grouped_iq1(&input, &matrices)?;
            device.synchronize()?;
            let mut positive = false;
            let mut negative = false;
            for (slot, reference) in expected.iter().enumerate() {
                let got = actual.narrow(0, slot, 1)?.to_vec2::<f32>()?;
                let want = reference.to_vec2::<f32>()?;
                for (&a, &b) in got[0].iter().zip(&want[0]) {
                    if exceptional.is_nan() {
                        assert!(a.is_nan() && b.is_nan(), "NaN must propagate");
                    } else {
                        assert!(
                            a.is_infinite() && b.is_infinite(),
                            "single infinite input must propagate"
                        );
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "infinite sign must match separate forward"
                        );
                        positive |= a.is_sign_positive();
                        negative |= a.is_sign_negative();
                    }
                }
            }
            if exceptional.is_infinite() {
                assert!(
                    positive && negative,
                    "fixture must exercise both weight signs"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn grouped_iq1_real_metal_rejects_device_layout_and_dtype() -> Result<()> {
        let device = Device::new_metal(0)?;
        let owners = grouped_iq1_weights(&device, false)?;
        let matrices = owners.iter().collect::<Vec<_>>();
        let input = grouped_iq1_input(&device, 2, false)?;
        for bad in [
            input.to_device(&Device::Cpu)?,
            input.to_dtype(DType::F16)?,
            input.reshape(5120)?,
            Tensor::zeros((5120, 2), DType::F32, &device)?
                .narrow(1, 0, 1)?
                .t()?,
        ] {
            assert!(DeepSeek4QMatMul::forward_grouped_iq1(&bad, &matrices).is_err());
        }
        let other_device = Device::new_metal(0)?;
        assert!(
            !device.same_device(&other_device),
            "fixture needs a distinct Candle Metal device"
        );
        let alien = grouped_iq1_test_matrix(
            vec![2304, 5120],
            DeepSeek4DType::IQ1S,
            Tensor::zeros(2_304_000, DType::U8, &other_device)?,
        )?;
        let mut bad = matrices.clone();
        bad[11] = &alien;
        assert!(
            DeepSeek4QMatMul::forward_grouped_iq1(&input, &bad)
                .unwrap_err()
                .to_string()
                .contains("share one Metal device")
        );
        let strided = grouped_iq1_test_matrix(
            vec![2304, 5120],
            DeepSeek4DType::IQ1S,
            Tensor::zeros((2_304_000, 2), DType::U8, &device)?.narrow(1, 0, 1)?,
        )?;
        bad[11] = &strided;
        assert!(
            DeepSeek4QMatMul::forward_grouped_iq1(&input, &bad)
                .unwrap_err()
                .to_string()
                .contains("contiguous packed")
        );
        // Keep all owners (including rejection fixtures) through completion.
        device.synchronize()?;
        other_device.synchronize()?;
        Ok(())
    }

    #[test]
    fn grouped_iq1_real_metal_explicit_disabled_mode() -> Result<()> {
        let device = Device::new_metal(0)?;
        let owners = grouped_iq1_weights(&device, false)?;
        let matrices = owners.iter().collect::<Vec<_>>();
        let mut basis = vec![0f32; 5120];
        basis[123] = 1.;
        let input = Tensor::from_vec(basis, (1, 5120), &device)?;
        let geometry = GroupedIq1Geometry::new(12, input.dims())?;
        assert!(
            super::super::metal::forward_grouped_iq1_with_mode(&input, &matrices, geometry, false)
                .unwrap_err()
                .to_string()
                .contains("requires SIMD enabled")
        );
        // Explicit serial mode exercises the normal opt-out kernel, without
        // mutating the process environment or claiming env-to-dispatch proof.
        let fallback = forward_with_kernel(
            owners[0].weights.bytes(),
            &input,
            DeepSeek4DType::IQ1S,
            2304,
            5120,
            KernelKind::Serial,
        )?;
        device.synchronize()?;
        let actual = fallback.to_vec2::<f32>()?;
        let packed = owners[0].weights.bytes().to_vec1::<u8>()?;
        for (row, &value) in actual[0].iter().enumerate() {
            let scalar = iq1s_scalar_reference(&packed[row * 1000..row * 1000 + 50])[123];
            assert!(value.is_finite());
            assert_eq!(
                value.to_bits(),
                scalar.to_bits(),
                "disabled serial basis row={row}"
            );
        }
        Ok(())
    }

    #[test]
    fn iq1s_basis_projections_match_independent_scalar_values() -> Result<()> {
        let device = crate::test_metal_device().expect("IQ1_S numerical check requires Metal");
        let (width, out_dim) = (512, 3);
        let mut bytes = Vec::new();
        let mut reference = Vec::new();
        for index in 0..out_dim * 2 {
            let (packed, scalar) = block(DeepSeek4DType::IQ1S, index)?;
            bytes.extend(packed);
            reference.extend(scalar);
        }
        let weights = raw(vec![out_dim, width], DeepSeek4DType::IQ1S, bytes, &device)?;
        let mut basis = vec![0f32; width * width];
        for element in 0..width {
            basis[element * width + element] = 1.;
        }
        let input = Tensor::from_vec(basis, (width, width), &device)?;
        for kind in [KernelKind::Serial, KernelKind::Simd] {
            for decode_rows in [false, true] {
                let actual = super::super::metal::forward_with_mode(
                    weights.bytes(),
                    &input,
                    DeepSeek4DType::IQ1S,
                    out_dim,
                    width,
                    kind,
                    decode_rows,
                )?
                .to_vec2::<f32>()?;
                // These fixture scales/deltas are exactly representable in f16.
                // Basis vectors isolate each decoded value, with no cancellation
                // or loose dot-product tolerance to hide an indexing error.
                for element in 0..width {
                    for row in 0..out_dim {
                        assert_eq!(
                            actual[element][row],
                            reference[row * width + element],
                            "{kind:?} decode_rows={decode_rows} element={element} row={row}"
                        );
                    }
                }
            }
        }
        device.synchronize()?;
        eprintln!("IQ1_S Metal: all four variants exactly match 1536 independent scalar values");
        Ok(())
    }

    #[allow(
        clippy::excessive_precision,
        reason = "Exact binary fractions define quantization boundary fixtures."
    )]
    fn iq1s_precision_fixture(width: usize) -> (Vec<u8>, Vec<f32>) {
        let mut bytes = Vec::new();
        let mut reference = Vec::new();
        for scale in [1.0009765625f32, 4096., 65504.] {
            for block_index in 0..width / 256 {
                let mut packed = vec![0u8; 50];
                packed[..2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
                for group in 0..8 {
                    // Maximum odd scale (15), both delta signs, and all high
                    // grid bits. The first weight is d * 15 * (1 + 1/8).
                    let mut h = 7u16 << 12;
                    if (group + block_index) % 2 != 0 {
                        h |= 0x8000;
                    }
                    for grid in 0..4 {
                        packed[2 + 4 * group + grid] = ((grid + block_index + 1) % 4) as u8;
                        h |= (((group + 3 * grid + block_index) % 8) as u16) << (3 * grid);
                    }
                    packed[34 + 2 * group..36 + 2 * group].copy_from_slice(&h.to_le_bytes());
                }
                reference.extend(iq1s_scalar_reference(&packed));
                bytes.extend(packed);
            }
        }
        (bytes, reference)
    }

    #[test]
    #[allow(
        clippy::excessive_precision,
        reason = "Exact binary fractions define quantization boundary fixtures."
    )]
    fn iq1s_precision_basis_preserves_f32_weights_and_finite_range() -> Result<()> {
        let device = crate::test_metal_device().expect("IQ1_S precision check requires Metal");
        let (width, out_dim) = (512, 3);
        let (bytes, reference) = iq1s_precision_fixture(width);
        assert_eq!(reference[0], 16.8914794921875);
        assert_eq!(half::f16::from_f32(reference[0]).to_f32(), 16.890625);
        assert_eq!(reference[width], 69120.);
        assert_eq!(reference[2 * width], 1105380.);
        let weights = raw(vec![out_dim, width], DeepSeek4DType::IQ1S, bytes, &device)?;
        let mut basis = vec![0f32; width * width];
        for element in 0..width {
            basis[element * width + element] = 1.;
        }
        let input = Tensor::from_vec(basis, (width, width), &device)?;
        for kind in [KernelKind::Serial, KernelKind::Simd] {
            for decode_rows in [false, true] {
                let actual = super::super::metal::forward_with_mode(
                    weights.bytes(),
                    &input,
                    DeepSeek4DType::IQ1S,
                    out_dim,
                    width,
                    kind,
                    decode_rows,
                )?
                .to_vec2::<f32>()?;
                for element in 0..width {
                    for row in 0..out_dim {
                        let value = actual[element][row];
                        assert!(
                            value.is_finite(),
                            "{kind:?} decode_rows={decode_rows} element={element} row={row}: {value}"
                        );
                        assert_eq!(
                            value,
                            reference[row * width + element],
                            "{kind:?} decode_rows={decode_rows} element={element} row={row}"
                        );
                    }
                }
            }
        }
        device.synchronize()?;
        eprintln!(
            "IQ1_S precision Metal: four variants exactly match 1536 F32 scalar weights (6144 comparisons), including weights beyond F16 range"
        );
        Ok(())
    }

    #[test]
    fn iq1s_precision_dots_and_public_dispatch_match_f64_oracle() -> Result<()> {
        let device = crate::test_metal_device().expect("IQ1_S precision check requires Metal");
        let out_dim = 3;
        let mut comparisons = 0;
        let mut max_abs_error = 0f64;
        let mut max_scaled_error = 0f64;
        for width in [256, 1024, 2048] {
            let (bytes, reference) = iq1s_precision_fixture(width);
            let weights = raw(vec![out_dim, width], DeepSeek4DType::IQ1S, bytes, &device)?;
            let matmul = DeepSeek4QMatMul::new(weights.clone())?;
            for tokens in [1, 3] {
                let host = (0..tokens * width)
                    .map(|i| ((i * 13 + 7) % 59) as f32 / 7.1 - 4.)
                    .collect::<Vec<_>>();
                let input = Tensor::from_slice(&host, (tokens, width), &device)?;
                for mode in 0..6 {
                    let output = match mode {
                        0..=3 => super::super::metal::forward_with_mode(
                            weights.bytes(),
                            &input,
                            DeepSeek4DType::IQ1S,
                            out_dim,
                            width,
                            if mode < 2 {
                                KernelKind::Serial
                            } else {
                                KernelKind::Simd
                            },
                            mode % 2 != 0,
                        )?,
                        4 => matmul.forward(&input)?,
                        _ => matmul.forward_decode_rows(&input)?,
                    };
                    let actual = output.to_vec2::<f32>()?;
                    for token in 0..tokens {
                        for row in 0..out_dim {
                            let terms = host[token * width..(token + 1) * width]
                                .iter()
                                .zip(&reference[row * width..(row + 1) * width])
                                .map(|(x, w)| f64::from(*x) * f64::from(*w));
                            let expected: f64 = terms.clone().sum();
                            let magnitude: f64 = terms.map(f64::abs).sum();
                            let value = actual[token][row];
                            assert!(
                                value.is_finite(),
                                "mode={mode} width={width} token={token} row={row}: {value}"
                            );
                            let error = (f64::from(value) - expected).abs();
                            // Bound F32 dot/reduction error against the sum of
                            // absolute products, not a cancellation-sensitive result.
                            assert!(
                                error <= 1e-4 + magnitude * 2e-5,
                                "mode={mode} width={width} token={token} row={row} got={value} expected={expected} error={error}"
                            );
                            comparisons += 1;
                            max_abs_error = max_abs_error.max(error);
                            max_scaled_error = max_scaled_error.max(error / magnitude.max(1.));
                        }
                    }
                }
            }
        }
        device.synchronize()?;
        eprintln!(
            "IQ1_S precision Metal: {comparisons} finite dot outputs match F64 oracle; max_abs_error={max_abs_error} max_magnitude_scaled_error={max_scaled_error}; four forced variants and both public APIs"
        );
        Ok(())
    }

    #[test]
    fn public_matmul_dispatch_matches_the_requested_policy() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for (dtype, width, tokens) in [
            (DeepSeek4DType::Q4K, 256, 1),
            (DeepSeek4DType::Q4K, 4096, 1),
            (DeepSeek4DType::Q4K, 4096, 3),
            (DeepSeek4DType::IQ3XXS, 256, 1),
            (DeepSeek4DType::IQ3XXS, 2048, 1),
            (DeepSeek4DType::IQ3XXS, 2048, 8),
            (DeepSeek4DType::IQ1S, 256, 1),
            (DeepSeek4DType::IQ1S, 2048, 1),
            (DeepSeek4DType::IQ1S, 2048, 8),
        ] {
            let out_dim = 5;
            let mut bytes = Vec::new();
            for i in 0..out_dim * width / 256 {
                bytes.extend(block(dtype, i)?.0);
            }
            let weights = raw(vec![out_dim, width], dtype, bytes, &device)?;
            let matmul = DeepSeek4QMatMul::new(weights.clone())?;
            let input = Tensor::from_vec(
                (0..tokens * width)
                    .map(|i| (i % 17) as f32 / 7.1)
                    .collect::<Vec<_>>(),
                (tokens, width),
                &device,
            )?;
            let enabled = cfg!(target_arch = "aarch64");
            let expected_kind = if enabled && tokens == 1 && width >= 1024 {
                KernelKind::Simd
            } else {
                KernelKind::Serial
            };
            let expected = forward_with_kernel(
                weights.bytes(),
                &input,
                dtype,
                out_dim,
                width,
                expected_kind,
            )?
            .to_vec2::<f32>()?;
            assert_eq!(matmul.forward(&input)?.to_vec2::<f32>()?, expected);
        }
        Ok(())
    }

    #[test]
    fn decode_rows_match_separate_scalar_calls_exactly() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for dtype in QUANTS {
            for width in [256, 1024, 2048] {
                let blocks = width / dtype.block_layout().unwrap().0;
                let mut bytes = Vec::new();
                for i in 0..6 * blocks {
                    bytes.extend(block(dtype, i)?.0);
                }
                let weights = raw(vec![6, width], dtype, bytes, &device)?;
                let weights = Arc::new(weights.slice_rows(1, 5)?);
                let matmul = DeepSeek4QMatMul::new(weights.clone())?;
                for lanes in [1, 2, 3, 4, 8] {
                    let input = Tensor::from_vec(
                        (0..(lanes + 1) * width)
                            .map(|i| ((i * 13 % 59) as f32 - 29.) / 7.1)
                            .collect(),
                        (lanes + 1, width),
                        &device,
                    )?;
                    for input in [
                        input.narrow(0, 1, lanes)?,
                        input.t()?.contiguous()?.t()?.narrow(0, 1, lanes)?,
                    ] {
                        let expected = (0..lanes)
                            .map(|lane| matmul.forward(&input.narrow(0, lane, 1)?))
                            .collect::<Result<Vec<_>>>()?;
                        assert_eq!(
                            matmul.forward_decode_rows(&input)?.to_vec2::<f32>()?,
                            Tensor::cat(&expected, 0)?.to_vec2::<f32>()?,
                            "{dtype:?} width={width} lanes={lanes}"
                        );
                        if matches!(
                            dtype,
                            DeepSeek4DType::Q5K
                                | DeepSeek4DType::IQ1M
                                | DeepSeek4DType::IQ2XXS
                                | DeepSeek4DType::IQ2XS
                        ) {
                            // The row kernels keep decode's sums; tiled
                            // prompt matmuls are checked in prefill_mm tests.
                            let expected = Tensor::cat(&expected, 0)?.to_vec2::<f32>()?;
                            prefill_mm::with_packed_disabled(|| -> Result<()> {
                                assert_eq!(
                                    matmul.forward_prefill_matrix(&input)?.to_vec2::<f32>()?,
                                    expected
                                );
                                assert_eq!(
                                    matmul.forward_attention_prefill(&input)?.to_vec2::<f32>()?,
                                    expected
                                );
                                Ok(())
                            })?;
                        }
                        // Both reduction choices preserve row order with interleaving.
                        for kind in [KernelKind::Serial, KernelKind::Simd] {
                            let expected = forward_with_kernel(
                                weights.bytes(),
                                &input,
                                dtype,
                                5,
                                width,
                                kind,
                            )?;
                            let actual = super::super::metal::forward_with_mode(
                                weights.bytes(),
                                &input,
                                dtype,
                                5,
                                width,
                                kind,
                                true,
                            )?;
                            assert_eq!(actual.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn packed_reuse4_matches_scalar_bitwise_with_tails_and_offsets() -> Result<()> {
        let device = Device::new_metal(0)?;
        for dtype in [
            DeepSeek4DType::Q5K,
            DeepSeek4DType::IQ1M,
            DeepSeek4DType::IQ2XXS,
            DeepSeek4DType::IQ2XS,
        ] {
            for width in [1024, 2304, 5120] {
                let blocks = width / dtype.block_layout().unwrap().0;
                let bytes = (0..6 * blocks)
                    .map(|i| block(dtype, i).map(|v| v.0))
                    .collect::<Result<Vec<_>>>()?
                    .concat();
                let weights =
                    Arc::new(raw(vec![6, width], dtype, bytes, &device)?.slice_rows(1, 5)?);
                let matmul = DeepSeek4QMatMul::new(weights.clone())?;
                for rows in [2, 3, 4, 7, 8, 31, 32, 127, 128] {
                    let input = Tensor::from_vec(
                        (0..(rows + 1) * width)
                            .map(|i| ((i * 13 % 59) as f32 - 29.) / 7.1)
                            .collect::<Vec<_>>(),
                        (rows + 1, width),
                        &device,
                    )?;
                    for input in [
                        input.narrow(0, 1, rows)?,
                        input.t()?.contiguous()?.t()?.narrow(0, 1, rows)?,
                    ] {
                        let expected = Tensor::cat(
                            &(0..rows)
                                .map(|i| {
                                    forward_with_kernel(
                                        weights.bytes(),
                                        &input.narrow(0, i, 1)?,
                                        dtype,
                                        5,
                                        width,
                                        KernelKind::Simd,
                                    )
                                })
                                .collect::<candle_core::Result<Vec<_>>>()?,
                            0,
                        )?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                        let candidate = forward_with_kernel(
                            weights.bytes(),
                            &input,
                            dtype,
                            5,
                            width,
                            KernelKind::Reuse4,
                        )?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                        assert!(
                            candidate
                                .iter()
                                .zip(&expected)
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "{dtype:?} width={width} rows={rows}"
                        );
                        let scoped = prefill_mm::with_packed_disabled(|| {
                            super::super::with_packed_reuse4_enabled(true, || {
                                matmul.forward_prefill_matrix(&input)
                            })
                        })?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                        let automatic = Tensor::cat(
                            &(0..rows)
                                .map(|i| matmul.forward(&input.narrow(0, i, 1)?))
                                .collect::<Result<Vec<_>>>()?,
                            0,
                        )?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                        assert!(
                            scoped
                                .iter()
                                .zip(&automatic)
                                .all(|(a, b)| a.to_bits() == b.to_bits())
                        );
                    }
                }
                let single = Tensor::zeros((1, width), DType::F32, &device)?;
                assert!(
                    forward_with_kernel(
                        weights.bytes(),
                        &single,
                        dtype,
                        5,
                        width,
                        KernelKind::Reuse4
                    )
                    .is_err()
                );
                let large = Tensor::zeros((129, width), DType::F32, &device)?;
                assert!(
                    forward_with_kernel(
                        weights.bytes(),
                        &large,
                        dtype,
                        5,
                        width,
                        KernelKind::Reuse4
                    )
                    .is_err()
                );
            }
        }
        Ok(())
    }

    #[test]
    fn nonzero_packed_matmuls_match_reference_with_slices_and_strides() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for dtype in QUANTS {
            for (blocks, out_dim, tokens) in [
                (1, 1, 1),
                (2, 3, 2),
                (3, 5, 6),
                (31, 7, 3),
                (32, 4, 1),
                (33, 9, 2),
            ] {
                let width = dtype.block_layout().unwrap().0 * blocks;
                let mut bytes = Vec::new();
                let mut reference = Vec::new();
                // One extra row lets us test a packed expert/row view with an offset.
                for i in 0..(out_dim + 1) * blocks {
                    let (b, v) = block(dtype, i)?;
                    bytes.extend(b);
                    reference.extend(v);
                }
                let weights = raw(vec![out_dim + 1, width], dtype, bytes, &device)?;
                let weights = weights.slice_rows(1, out_dim)?;
                let reference = &reference[width..];
                let host = (0..(tokens + 2) * width)
                    .map(|i| (((i * 7 + 3) % 23) as f32 - 11.) / 7.1)
                    .collect::<Vec<_>>();
                let input = Tensor::from_slice(&host, (tokens + 2, width), &device)?;
                let batch = if tokens.is_multiple_of(2) { 2 } else { 1 };
                for x in [
                    input.narrow(0, 1, tokens)?,
                    input
                        .narrow(0, 1, tokens)?
                        .reshape((batch, tokens / batch, width))?,
                    input.t()?.contiguous()?.t()?.narrow(0, 1, tokens)?,
                ] {
                    for kind in [KernelKind::Serial, KernelKind::Simd] {
                        let actual =
                            forward_with_kernel(weights.bytes(), &x, dtype, out_dim, width, kind)?
                                .flatten_all()?
                                .to_vec1::<f32>()?;
                        for token in 0..tokens {
                            for row in 0..out_dim {
                                let terms = host[(token + 1) * width..(token + 2) * width]
                                    .iter()
                                    .zip(&reference[row * width..(row + 1) * width])
                                    .map(|(x, w)| f64::from(*x) * f64::from(*w));
                                let expected: f64 = terms.clone().sum();
                                let magnitude: f64 = terms.map(f64::abs).sum();
                                let error =
                                    (f64::from(actual[token * out_dim + row]) - expected).abs();
                                assert!(
                                    error <= 1e-4 + magnitude * 2e-5,
                                    "{dtype:?} {kind:?} width={width} token={token} row={row} got={} expected={expected} error={error}",
                                    actual[token * out_dim + row]
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn compare_and_bench(weights: &RawTensor, input: &Tensor, label: &str) -> Result<()> {
        let [out_dim, width] = [weights.shape()[0], weights.shape()[1]];
        let device = input.device();
        let call = |kind| {
            forward_with_kernel(
                weights.bytes(),
                input,
                weights.dtype(),
                out_dim,
                width,
                kind,
            )
        };
        let serial = call(KernelKind::Serial)?.flatten_all()?.to_vec1::<f32>()?;
        let simd = call(KernelKind::Simd)?.flatten_all()?.to_vec1::<f32>()?;
        let mut error_sq = 0f64;
        let mut norm_sq = 0f64;
        let mut max_abs = 0f64;
        for (a, b) in serial.iter().zip(&simd) {
            anyhow::ensure!(a.is_finite() && b.is_finite(), "{label}: non-finite output");
            let error = f64::from(*a) - f64::from(*b);
            error_sq += error * error;
            norm_sq += f64::from(*a).powi(2);
            max_abs = max_abs.max(error.abs());
        }
        let relative_l2 = (error_sq / norm_sq.max(1e-30)).sqrt();
        anyhow::ensure!(
            relative_l2 < 2e-4,
            "{label}: relative L2 error {relative_l2}"
        );
        for kind in [KernelKind::Serial, KernelKind::Simd] {
            let outputs = (0..64)
                .map(|_| call(kind))
                .collect::<candle_core::Result<Vec<_>>>()?;
            device.synchronize()?;
            std::hint::black_box(outputs);
        }
        // Report both single-operation latency and queued operation cost.
        // Fencing each tiny kernel alone can hide gains behind wait overhead.
        for queued in [1, 16] {
            let mut times = [Vec::new(), Vec::new()];
            for repeat in 0..21 {
                // Alternate order to reduce thermal/order bias. Include dispatch
                // and a completion fence in both measurements; exclude compilation.
                for index in [repeat % 2, 1 - repeat % 2] {
                    let kind = [KernelKind::Serial, KernelKind::Simd][index];
                    let mut outputs = Vec::with_capacity(queued);
                    let start = std::time::Instant::now();
                    for _ in 0..queued {
                        outputs.push(call(kind)?);
                    }
                    device.synchronize()?;
                    times[index].push(start.elapsed().as_secs_f64() * 1e6 / queued as f64);
                    std::hint::black_box(outputs);
                }
            }
            for samples in &mut times {
                samples.sort_by(f64::total_cmp);
            }
            eprintln!(
                "{label} {:?} shape={:?} input={:?} queued={queued} serial_us={:.2} simd_us={:.2} speedup={:.2} max_abs={max_abs:.7} relative_l2={relative_l2:.8}",
                weights.dtype(),
                weights.shape(),
                input.dims(),
                times[0][10],
                times[1][10],
                times[0][10] / times[1][10]
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "manual synchronized Metal kernel benchmark; not a full-model speed result"]
    fn bench_packed_matmul_kernels() -> Result<()> {
        let device = crate::test_metal_device().expect("benchmark requires Metal");
        for dtype in QUANTS {
            let (block_elements, _) = dtype.block_layout().unwrap();
            let fixtures = (0..24)
                .map(|i| block(dtype, i).map(|(b, _)| b))
                .collect::<Result<Vec<_>>>()?;
            for (width, out_dim) in [(256, 7), (1024, 256), (5120, 256), (5120, 2048)] {
                let mut bytes = Vec::with_capacity(dtype.byte_len(width * out_dim)?);
                for i in 0..width * out_dim / block_elements {
                    bytes.extend_from_slice(&fixtures[i % fixtures.len()]);
                }
                let weights = raw(vec![out_dim, width], dtype, bytes, &device)?;
                for tokens in [1, 8] {
                    let input = Tensor::from_vec(
                        (0..tokens * width)
                            .map(|i| (((i * 17 + 3) % 101) as f32 - 50.) / 63.7)
                            .collect::<Vec<_>>(),
                        (tokens, width),
                        &device,
                    )?;
                    compare_and_bench(&weights, &input, "synthetic")?;
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a local weight shard; dependent expert chain, not full inference"]
    fn bench_real_expert_chain() -> Result<()> {
        use crate::deepseek4_gguf::{RawGgufContent, RawGgufSource};
        let path = std::env::var("DINF_DEEPSEEK4_BENCH_GGUF")?;
        let layer = std::env::var("DINF_DEEPSEEK4_BENCH_LAYER")
            .unwrap_or_else(|_| "0".into())
            .parse::<usize>()?;
        let mut file = std::fs::File::open(&path)?;
        let mut content = RawGgufContent::read(&mut file)?;
        let maya = if matches!(
            content.metadata.get("general.architecture"),
            Some(candle_core::quantized::gguf_file::Value::String(architecture))
                if architecture == "deepseek41"
        ) {
            Some(RawGgufSource::open_maya(std::path::Path::new(&path))?)
        } else {
            None
        };
        eprintln!(
            "expert chain weight source={path} verified_maya={}",
            maya.is_some()
        );
        let device = crate::test_metal_device().expect("benchmark requires Metal");
        let mut matrices = Vec::new();
        for part in ["gate", "up", "down"] {
            let name = format!("blk.{layer}.ffn_{part}_exps.weight");
            if let Some(source) = &maya {
                // Use the bounded slice API; V4.1 whole-bank loads stay forbidden.
                matrices.push(source.read_expert_slice(&name, 0, &device)?);
                continue;
            }
            let info = content
                .tensor_infos
                .get_mut(&name)
                .ok_or_else(|| anyhow::anyhow!("missing {name}"))?;
            anyhow::ensure!(
                info.shape.len() == 3 && info.shape[0] > 0,
                "invalid expert shape"
            );
            // Read only the first complete expert, not the multi-GB expert bank.
            info.shape = info.shape[1..].to_vec();
            anyhow::ensure!(
                info.byte_len()? <= 64 * 1024 * 1024,
                "expert exceeds 64 MiB"
            );
            matrices.push(content.load_tensor(&mut file, &name, &device)?);
        }
        let [gate, up, down] = [&matrices[0], &matrices[1], &matrices[2]];
        let width = gate.shape()[1];
        let initial = Tensor::from_vec(
            (0..width)
                .map(|i| (((i * 17 + 3) % 101) as f32 - 50.) / 63.7)
                .collect::<Vec<_>>(),
            (1, width),
            &device,
        )?;
        let project = |w: &RawTensor, x: &Tensor, kind| {
            forward_with_kernel(w.bytes(), x, w.dtype(), w.shape()[0], w.shape()[1], kind)
        };
        let chain = |kind, fused_activation, steps| -> Result<Tensor> {
            let mut x = initial.clone();
            for _ in 0..steps {
                let g = project(gate, &x, kind)?;
                let u = project(up, &x, kind)?;
                let activated = if fused_activation {
                    crate::architectures::indexed_moe::silu_mul(&g, &u)?
                } else {
                    candle_nn::ops::silu(&g)?.broadcast_mul(&u)?
                };
                let d = project(down, &activated, kind)?;
                // A modest residual keeps values bounded while each expert
                // pass depends on the last; independent calls may overlap in
                // Candle's concurrent Metal encoder and hide latency changes.
                x = (&x + d.affine(0.125, 0.)?)?;
            }
            Ok(x)
        };
        for steps in [1, 8] {
            let reference = chain(KernelKind::Serial, false, steps)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for (name, candidate) in [
                ("fused_activation", KernelKind::Serial),
                ("simd_and_fused", KernelKind::Simd),
            ] {
                let actual = chain(candidate, true, steps)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let error_sq = reference
                    .iter()
                    .zip(&actual)
                    .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                    .sum::<f64>();
                let norm_sq = reference.iter().map(|a| f64::from(*a).powi(2)).sum::<f64>();
                let relative_l2 = (error_sq / norm_sq.max(1e-30)).sqrt();
                anyhow::ensure!(
                    reference.iter().chain(&actual).all(|v| v.is_finite()) && relative_l2 < 2e-4,
                    "expert chain error {relative_l2}"
                );
                for _ in 0..4 {
                    for (kind, fused) in [(KernelKind::Serial, false), (candidate, true)] {
                        std::hint::black_box(chain(kind, fused, steps)?);
                        device.synchronize()?;
                    }
                }
                let mut times = [Vec::new(), Vec::new()];
                for repeat in 0..21 {
                    for i in [repeat % 2, 1 - repeat % 2] {
                        let start = std::time::Instant::now();
                        let output = chain([KernelKind::Serial, candidate][i], i == 1, steps)?;
                        device.synchronize()?;
                        times[i].push(start.elapsed().as_secs_f64() * 1e6 / steps as f64);
                        std::hint::black_box(output);
                    }
                }
                for values in &mut times {
                    values.sort_by(f64::total_cmp);
                }
                eprintln!(
                    "expert layer={layer} gate={:?}{:?} up={:?} down={:?}{:?} steps={steps} candidate={name} serial_us={:.2} candidate_us={:.2} speedup={:.2} relative_l2={relative_l2:.8}",
                    gate.dtype(),
                    gate.shape(),
                    up.dtype(),
                    down.dtype(),
                    down.shape(),
                    times[0][10],
                    times[1][10],
                    times[0][10] / times[1][10]
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires DINF_DEEPSEEK4_BENCH_GGUF pointing to a local verified shard"]
    fn bench_real_packed_matrix_slices() -> Result<()> {
        use crate::deepseek4_gguf::RawGgufContent;
        use std::io::{Read, Seek, SeekFrom};
        let path = std::env::var("DINF_DEEPSEEK4_BENCH_GGUF")?;
        eprintln!("Weight source: {path}");
        let mut file = std::fs::File::open(path)?;
        let content = RawGgufContent::read(&mut file)?;
        let device = crate::test_metal_device().expect("benchmark requires Metal");
        let mut names = content.tensor_infos.keys().collect::<Vec<_>>();
        names.sort();
        let mut seen = std::collections::HashSet::new();
        for name in names {
            let info = &content.tensor_infos[name];
            if info.shape.len() < 2 || !QUANTS.contains(&info.dtype) || seen.contains(&info.dtype) {
                continue;
            }
            let width = *info.shape.last().unwrap();
            let rows = info.shape[info.shape.len() - 2];
            let out_dim = rows.min(512);
            if width == 0 || out_dim == 0 {
                continue;
            }
            let start_row = rows - out_dim;
            let row_bytes = info.dtype.byte_len(width)?;
            let byte_len = row_bytes.checked_mul(out_dim).unwrap();
            anyhow::ensure!(
                byte_len <= 64 * 1024 * 1024,
                "benchmark slice exceeds 64 MiB"
            );
            let start = content
                .tensor_data_offset
                .checked_add(info.offset)
                .and_then(|v| v.checked_add((row_bytes * start_row) as u64))
                .unwrap();
            file.seek(SeekFrom::Start(start))?;
            let mut bytes = vec![0; byte_len];
            file.read_exact(&mut bytes)?;
            let weights = raw(vec![out_dim, width], info.dtype, bytes, &device)?;
            for tokens in [1, 8] {
                let input = Tensor::from_vec(
                    (0..tokens * width)
                        .map(|i| (((i * 17 + 3) % 101) as f32 - 50.) / 63.7)
                        .collect::<Vec<_>>(),
                    (tokens, width),
                    &device,
                )?;
                compare_and_bench(&weights, &input, name)?;
            }
            seen.insert(info.dtype);
        }
        anyhow::ensure!(!seen.is_empty(), "no supported packed matrices in shard");
        Ok(())
    }
}

#[cfg(target_os = "macos")]
mod v41_precision {
    use super::*;

    fn matrix(
        device: &Device,
        rows: usize,
        width: usize,
        bytes: Vec<u8>,
    ) -> Result<DeepSeek4QMatMul> {
        DeepSeek4QMatMul::new_v41_on_device(
            raw(vec![rows, width], DeepSeek4DType::Q8_0, bytes, device)?,
            device,
        )
    }

    fn block(scale: u16, q: i8) -> Vec<u8> {
        let mut bytes = scale.to_le_bytes().to_vec();
        bytes.extend([q as u8; 32]);
        bytes
    }

    #[test]
    #[allow(
        clippy::excessive_precision,
        reason = "Exact binary fractions define quantization boundary fixtures."
    )]
    fn v41_q8_preserves_bf16_boundary_and_cancellation() -> Result<()> {
        let device = Device::new_metal(0)?;
        let bytes = block(0x355b, 3);
        let weights = raw(vec![1, 32], DeepSeek4DType::Q8_0, bytes, &device)?;
        let old = DeepSeek4QMatMul::new_on_device(weights.clone(), &device)?;
        let new = DeepSeek4QMatMul::new_v41_on_device(weights, &device)?;
        let mut x = vec![0f32; 32];
        x[0] = 1.0;
        let x = Tensor::from_vec(x, (1, 32), &device)?;
        let y = new.forward(&x)?;
        assert_eq!(y.dtype(), DType::F32);
        let value = y.to_vec2::<f32>()?[0][0];
        let legacy = old.forward(&x)?.to_vec2::<f32>()?[0][0];
        assert_eq!(value, 1.004150390625);
        assert_eq!(legacy, 1.00390625);
        assert_eq!(half::bf16::from_f32(value).to_f32(), 1.0078125);
        assert_eq!(half::bf16::from_f32(legacy).to_f32(), 1.0);

        let bytes = [block(0x3c01, 3), block(0x3c00, -3)].concat();
        let weights = raw(vec![1, 64], DeepSeek4DType::Q8_0, bytes, &device)?;
        let old = DeepSeek4QMatMul::new(weights.clone())?;
        let new = DeepSeek4QMatMul::new_v41_on_device(weights, &device)?;
        let mut x = vec![0f32; 64];
        x[0] = 1.0;
        x[32] = 1.0;
        let x = Tensor::from_vec(x, (1, 64), &device)?;
        assert_eq!(new.forward(&x)?.to_vec2::<f32>()?[0][0], 0.0029296875);
        assert_eq!(old.forward(&x)?.to_vec2::<f32>()?[0][0], 0.00390625);
        device.synchronize()?;
        Ok(())
    }

    #[test]
    #[allow(
        clippy::excessive_precision,
        reason = "Exact binary fractions define quantization boundary fixtures."
    )]
    fn v41_q8_scale_boundaries_and_large_head() -> Result<()> {
        let device = Device::new_metal(0)?;
        let scales = [
            0x0000u16, 0x0001, 0x03ff, 0x0400, 0x355b, 0x3c01, 0x7bff, 0xbc01,
        ];
        let bytes = scales.iter().flat_map(|s| block(*s, 127)).collect();
        let mat = matrix(&device, scales.len(), 32, bytes)?;
        let mut x = vec![0f32; 32];
        x[0] = 1.0;
        let x = Tensor::from_vec(x, (1, 32), &device)?;
        let y = mat.forward(&x)?.to_vec2::<f32>()?;
        for (i, scale) in scales.iter().enumerate() {
            assert_eq!(y[0][i], half::f16::from_bits(*scale).to_f32() * 127.0);
        }
        // Block 39 belongs to SIMD group 4 in NSG8, exercising its upper half.
        let mat = matrix(&device, 65_537, 1280, block(0x355b, 3).repeat(65_537 * 40))?;
        let mut head_input = vec![0f32; 1280];
        head_input[0] = 1.0;
        head_input[1279] = 2.0;
        let x = Tensor::from_vec(head_input, (1, 1280), &device)?;
        let y = mat.forward(&x)?.to_vec2::<f32>()?;
        assert!(y[0].iter().all(|v| *v == 3.012451171875));
        device.synchronize()?;
        Ok(())
    }

    #[test]
    fn v41_q8_real_widths_independent_rows_match_f64_dot() -> Result<()> {
        let device = Device::new_metal(0)?;
        for width in [1280, 5120, 6144] {
            let mut bytes = Vec::new();
            let mut dense = Vec::new();
            for row in 0..3 {
                for b in 0..width / 32 {
                    let scale = half::f16::from_bits(0x3000 + ((b * 73 + row * 29) % 2048) as u16);
                    bytes.extend(scale.to_le_bytes());
                    for i in 0..32 {
                        let q = ((b * 17 + i * 13 + row * 7) % 255) as i16 - 127;
                        bytes.push(q as i8 as u8);
                        dense.push(f64::from(scale.to_f32()) * f64::from(q));
                    }
                }
            }
            let mat = matrix(&device, 3, width, bytes)?;
            let values = (0..6 * width)
                .map(|i| ((i * 31 % 257) as f32 - 128.0) / 127.0)
                .collect::<Vec<_>>();
            let input = Tensor::from_slice(&values, (2, 3, width), &device)?;
            let actual = mat.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
            let flat = input.reshape((6, width))?;
            for token in 0..6 {
                let single = mat
                    .forward_decode_rows(&flat.narrow(0, token, 1)?)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert_eq!(&actual[token * 3..token * 3 + 3], single.as_slice());
                for row in 0..3 {
                    let terms = (0..width)
                        .map(|i| dense[row * width + i] * f64::from(values[token * width + i]));
                    let expected: f64 = terms.clone().sum();
                    let magnitude: f64 = terms.map(f64::abs).sum();
                    let error = (f64::from(actual[token * 3 + row]) - expected).abs();
                    assert!(error <= magnitude * 2.0e-6 + 1.0e-6, "{width}: {error}");
                }
            }
        }
        device.synchronize()?;
        Ok(())
    }

    #[test]
    #[allow(
        clippy::excessive_precision,
        reason = "Exact binary fractions define quantization boundary fixtures."
    )]
    fn v41_q8_views_and_invalid_inputs() -> Result<()> {
        let device = Device::new_metal(0)?;
        let payload = block(0x355b, 3);
        let storage = Tensor::from_vec([vec![0u8; 34], payload].concat(), 68, &device)?;
        let weights = Arc::new(RawTensor::new(
            vec![1, 32],
            DeepSeek4DType::Q8_0,
            storage.narrow(0, 34, 34)?,
        )?);
        let mat = DeepSeek4QMatMul::new_v41_on_device(weights, &device)?;
        let input = Tensor::ones((4, 32), DType::F32, &device)?;
        let view = input.narrow(0, 2, 1)?;
        assert_eq!(
            mat.forward(&view)?.to_vec2::<f32>()?,
            vec![vec![32.1328125]]
        );
        let transposed = Tensor::ones((32, 2), DType::F32, &device)?.t()?;
        assert_eq!(
            mat.forward(&transposed)?.to_vec2::<f32>()?,
            vec![vec![32.1328125]; 2]
        );
        for x in [
            Tensor::zeros(32, DType::F32, &device)?,
            Tensor::zeros((1, 64), DType::F32, &device)?,
            Tensor::zeros((1, 32), DType::F16, &device)?,
            Tensor::zeros((0, 32), DType::F32, &device)?,
            Tensor::zeros((1, 32), DType::F32, &Device::Cpu)?,
        ] {
            assert!(mat.forward(&x).is_err());
        }
        device.synchronize()?;
        Ok(())
    }
}

#[cfg(target_os = "macos")]
mod grouped_v41 {
    use super::*;

    #[test]
    #[allow(
        clippy::excessive_precision,
        reason = "Exact binary fractions define quantization boundary fixtures."
    )]
    fn grouped_v41_q8_matches_eight_forwards_bitwise_with_offset_views() -> Result<()> {
        let device = Device::new_metal(0)?;
        let mut comparisons = 0;
        for (rows_per_group, width) in [(6, 1056), (1024, 4096)] {
            let groups = 8;
            let out = groups * rows_per_group;
            let blocks = width / 32;
            let row_bytes = blocks * 34;
            // One poison row before the matrix exercises the packed byte offset.
            // Largest fixture is 35,655,936 bytes, not a dense weight matrix.
            let mut bytes = vec![0xa5; (out + 1) * row_bytes];
            for row in 0..out {
                let group = row / rows_per_group;
                let local_row = row % rows_per_group;
                for ib in 0..blocks {
                    let offset = (row + 1) * row_bytes + ib * 34;
                    let scales = [0x355bu16, 0x3c01, 0x7bff, 0xfbff, 0x0001, 0xbc01];
                    let scale = match local_row {
                        0 => {
                            if ib == 0 {
                                0x355b
                            } else {
                                0
                            }
                        }
                        1 => match ib {
                            0 => 0x3c01,
                            1 => 0x3c00,
                            _ => 0,
                        },
                        2 => match ib {
                            0 | 1 => 0x7bff,
                            2 => 0x3c01,
                            _ => 0,
                        },
                        _ => scales[(group + local_row + ib) % scales.len()],
                    };
                    bytes[offset..offset + 2].copy_from_slice(&scale.to_le_bytes());
                    for i in 0..32 {
                        let q = match local_row {
                            0 => {
                                if i == 0 {
                                    (group + 3) as i8
                                } else {
                                    0
                                }
                            }
                            1 => match (i, ib) {
                                (0, 0) => 3,
                                (0, _) => -3,
                                _ => 0,
                            },
                            2 => match (i, ib) {
                                (0, 0) => 127,
                                (0, 1) => -127,
                                (0, _) => 1,
                                _ => 0,
                            },
                            _ => ((group * 19 + local_row * 13 + ib * 7 + i * 11) % 255) as i8,
                        };
                        bytes[offset + 2 + i] = q as u8;
                    }
                }
            }
            let backing = raw(vec![out + 1, width], DeepSeek4DType::Q8_0, bytes, &device)?;
            let weights = Arc::new(backing.slice_rows(1, out)?);
            let mat = DeepSeek4QMatMul::new_v41_on_device(weights.clone(), &device)?;
            let mut host = (0..(groups + 1) * width)
                .map(|i| ((i * 31 + i / width * 17) % 257) as f32 / 127.0 - 1.0)
                .collect::<Vec<_>>();
            for group in 0..groups {
                // These two entries make the boundary/cancellation rows exact
                // while all other entries exercise the full reduction tree.
                host[(group + 1) * width] = (group + 1) as f32;
                host[(group + 1) * width + 32] = (group + 1) as f32;
            }
            let input =
                Tensor::from_vec(host, (groups + 1, width), &device)?.narrow(0, 1, groups)?;
            let output = mat.forward_grouped_v41(&input, groups)?;
            assert_eq!(output.dims(), &[1, out]);
            assert_eq!(output.dtype(), DType::F32);
            assert!(output.is_contiguous());
            let actual = output.to_vec2::<f32>()?.remove(0);
            for group in 0..groups {
                let reference = DeepSeek4QMatMul::new_v41_on_device(
                    Arc::new(weights.slice_rows(group * rows_per_group, rows_per_group)?),
                    &device,
                )?
                .forward(&input.narrow(0, group, 1)?)?
                .to_vec2::<f32>()?;
                for row in 0..rows_per_group {
                    let value = actual[group * rows_per_group + row];
                    assert!(value.is_finite());
                    assert_eq!(
                        value.to_bits(),
                        reference[0][row].to_bits(),
                        "width={width} group={group} row={row}"
                    );
                    comparisons += 1;
                }
                let first = actual[group * rows_per_group];
                assert_eq!(
                    first,
                    half::f16::from_bits(0x355b).to_f32() * ((group + 3) * (group + 1)) as f32
                );
                assert_eq!(
                    actual[group * rows_per_group + 1],
                    0.0029296875 * (group + 1) as f32
                );
                for earlier in 0..group {
                    assert_ne!(first.to_bits(), actual[earlier * rows_per_group].to_bits());
                }
            }
            assert_eq!(actual[0], 1.004150390625);
            assert_eq!(half::bf16::from_f32(actual[0]).to_f32(), 1.0078125);
        }
        device.synchronize()?;
        eprintln!(
            "grouped V4.1 Q8 Metal: {comparisons} bitwise F32 comparisons against \
             16 independent forward calls; full 8192x4096 plus 48x1056, eight \
             distinct groups, finite large scales/cancellation, BF16 boundary, \
             offset weights/input"
        );
        Ok(())
    }

    #[test]
    fn grouped_v41_q8_rejects_wrong_groups_inputs_and_devices() -> Result<()> {
        let device = Device::new_metal(0)?;
        let weights = raw(
            vec![16, 32],
            DeepSeek4DType::Q8_0,
            vec![0; 16 * 34],
            &device,
        )?;
        let mat = DeepSeek4QMatMul::new_v41_on_device(weights, &device)?;
        let input = Tensor::zeros((8, 32), DType::F32, &device)?;
        for groups in [0, 3, 16] {
            assert!(mat.forward_grouped_v41(&input, groups).is_err());
        }
        for x in [
            Tensor::zeros(256, DType::F32, &device)?,
            Tensor::zeros((1, 8, 32), DType::F32, &device)?,
            Tensor::zeros((8, 64), DType::F32, &device)?,
            Tensor::zeros((8, 32), DType::F16, &device)?,
            Tensor::zeros((8, 32), DType::BF16, &device)?,
            Tensor::zeros((32, 8), DType::F32, &device)?.t()?,
        ] {
            assert!(mat.forward_grouped_v41(&x, 8).is_err());
        }
        let cpu_input = Tensor::zeros((8, 32), DType::F32, &Device::Cpu)?;
        assert!(
            mat.forward_grouped_v41(&cpu_input, 8)
                .unwrap_err()
                .to_string()
                .contains("share a device")
        );
        let cpu_weights = DeepSeek4QMatMul::new_v41_on_device(
            raw(
                vec![16, 32],
                DeepSeek4DType::Q8_0,
                vec![0; 16 * 34],
                &Device::Cpu,
            )?,
            &Device::Cpu,
        )?;
        assert!(
            cpu_weights
                .forward_grouped_v41(&input, 8)
                .unwrap_err()
                .to_string()
                .contains("share a device")
        );
        let odd = DeepSeek4QMatMul::new_v41_on_device(
            raw(
                vec![24, 32],
                DeepSeek4DType::Q8_0,
                vec![0; 24 * 34],
                &device,
            )?,
            &device,
        )?;
        assert!(
            odd.forward_grouped_v41(&input, 8)
                .unwrap_err()
                .to_string()
                .contains("even")
        );
        device.synchronize()?;
        Ok(())
    }
}

#[cfg(target_os = "macos")]
#[test]
fn v41_q8_rows_seq4_exact_production_shapes_offsets_and_tails() -> Result<()> {
    use super::metal;
    let device = Device::new_metal(0)?;
    let mut comparisons = 0;
    let mut timings = Vec::new();
    for (width, out) in [
        (5120, 2304),
        (2304, 5120),
        (5120, 7),
        (1280, 32768),
        (8192, 5120),
        (5120, 1280),
        (5120, 512),
    ] {
        let row_bytes = width / 32 * 34;
        let mut bytes = vec![0xa5u8; (out + 1) * row_bytes];
        for row in 0..out {
            for ib in 0..width / 32 {
                let base = (row + 1) * row_bytes + ib * 34;
                let scale = [0x355bu16, 0xbc01, 0x0001, 0x7bff, 0xfbff][(row + ib) % 5];
                bytes[base..base + 2].copy_from_slice(&scale.to_le_bytes());
                for i in 0..32 {
                    bytes[base + 2 + i] =
                        (((row * 7 + ib * 17 + i * 13) % 255) as i16 - 127) as i8 as u8;
                }
            }
        }
        let storage = Tensor::from_vec(bytes, (out + 1) * row_bytes, &device)?;
        let weights = storage.narrow(0, row_bytes, out * row_bytes)?;
        for rows in [1, 2, 3, 7, 8, 9, 26, 27, 28, 31, 32] {
            let values: Vec<_> = (0..(rows + 1) * width)
                .map(|i| {
                    let scale = [1.0e-20f32, 1.1, 1.0e20, 0.03125][(i / 32) % 4];
                    (((i * 31 + 7) % 257) as f32 - 128.0) * scale
                })
                .collect();
            let storage = Tensor::from_vec(values, (rows + 1, width), &device)?;
            let input = storage.narrow(0, 1, rows)?;
            let reference = metal::forward_with_kernel(
                &weights,
                &input,
                DeepSeek4DType::Q8_0,
                out,
                width,
                metal::KernelKind::V41Q8Nsg4,
            )?;
            // Scalar rows deliberately remain on the original kernel.
            let kind = if rows == 1 {
                metal::KernelKind::V41Q8Nsg4
            } else {
                metal::KernelKind::V41Q8RowsSeq4
            };
            let actual = metal::forward_with_kernel(
                &weights,
                &input,
                DeepSeek4DType::Q8_0,
                out,
                width,
                kind,
            )?;
            // Both offset-view consumers are queued on one device before either
            // readback; no parallel GPU execution or extra fence is introduced.
            let reference = reference.flatten_all()?.to_vec1::<f32>()?;
            let actual = actual.flatten_all()?.to_vec1::<f32>()?;
            assert!(reference.iter().chain(&actual).all(|v| v.is_finite()));
            assert_eq!(
                reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "width{width} out{out} rows{rows}"
            );
            comparisons += actual.len();
            if rows == 32 && out > 7 {
                for kind in [
                    metal::KernelKind::V41Q8Nsg4,
                    metal::KernelKind::V41Q8RowsSeq4,
                    metal::KernelKind::V41Q8RowsSeq4,
                    metal::KernelKind::V41Q8Nsg4,
                ] {
                    device.synchronize()?;
                    let start = std::time::Instant::now();
                    for _ in 0..20 {
                        let output = metal::forward_with_kernel(
                            &weights,
                            &input,
                            DeepSeek4DType::Q8_0,
                            out,
                            width,
                            kind,
                        )?;
                        device.synchronize()?;
                        std::hint::black_box(output);
                    }
                    timings.push(serde_json::json!({"width":width,"out":out,"rows":rows,"kind":format!("{kind:?}"),"seconds_20":start.elapsed().as_secs_f64()}));
                }
            }
        }
        let wrong_width = Tensor::zeros((2, width - 1), DType::F32, &device)?;
        assert!(
            metal::forward_with_kernel(
                &weights,
                &wrong_width,
                DeepSeek4DType::Q8_0,
                out,
                width,
                metal::KernelKind::V41Q8RowsSeq4
            )
            .is_err()
        );
        let too_many = Tensor::zeros((33, width), DType::F32, &device)?;
        assert!(
            metal::forward_with_kernel(
                &weights,
                &too_many,
                DeepSeek4DType::Q8_0,
                out,
                width,
                metal::KernelKind::V41Q8RowsSeq4
            )
            .is_err()
        );
        let strided = Tensor::zeros((width, 2), DType::F32, &device)?.t()?;
        assert!(
            metal::forward_with_kernel(
                &weights,
                &strided,
                DeepSeek4DType::Q8_0,
                out,
                width,
                metal::KernelKind::V41Q8RowsSeq4
            )
            .is_err()
        );
        let odd = Tensor::zeros(out * row_bytes + 1, DType::U8, &device)?.narrow(
            0,
            1,
            out * row_bytes,
        )?;
        let input = Tensor::zeros((2, width), DType::F32, &device)?;
        assert!(
            metal::forward_with_kernel(
                &odd,
                &input,
                DeepSeek4DType::Q8_0,
                out,
                width,
                metal::KernelKind::V41Q8RowsSeq4
            )
            .is_err()
        );
    }
    println!(
        "Q8 attention rows evidence: {}",
        serde_json::to_string_pretty(
            &serde_json::json!({"finite_exact_comparisons":comparisons,"offsets":true,"partial_output_rows":7,"rows":[1,2,3,7,8,9,26,27,28,31,32],"timings":timings,"scope":"public synthetic packed Q8 and wide finite F32 cancellation; per-call fences in component timing only"})
        )?
    );
    Ok(())
}

/// Router scores choose experts, so batched dense decode rows must keep each
/// row's single-row GEMV bits.
#[test]
fn dense_decode_rows_match_single_row_calls_bitwise() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    let (rows, k, n) = (37usize, 5120usize, 192usize);
    let bytes = (0..n * k)
        .flat_map(|i| {
            half::bf16::from_f32(((i * 104_729 % 1999) as f32 / 999.0 - 1.0) * 0.01).to_le_bytes()
        })
        .collect::<Vec<_>>();
    let matmul = DeepSeek4QMatMul::new_on_device(
        raw(vec![n, k], DeepSeek4DType::BF16, bytes, &device)?,
        &device,
    )?;
    let x = Tensor::from_vec(
        (0..rows * k)
            .map(|i| (i * 7919 % 2003) as f32 / 1001.0 - 1.0)
            .collect::<Vec<_>>(),
        (rows, k),
        &device,
    )?;
    let batched = matmul
        .forward_decode_rows(&x)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for r in 0..rows {
        let single = matmul
            .forward(&x.narrow(0, r, 1)?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(
            single
                .iter()
                .zip(&batched[r * n..(r + 1) * n])
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "row {r} differs from its single-row GEMV"
        );
    }
    Ok(())
}
