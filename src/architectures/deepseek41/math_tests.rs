use super::*;
use candle_core::Device;

const FORMATS: [CacheFormat; 3] = [
    CacheFormat::RawKvFp8E8M0,
    CacheFormat::IndexQkFp4E8M0,
    CacheFormat::CompressedKvFp4E4M3,
];

fn tensor<const N: usize>(data: &[f32], dims: [usize; N], device: &Device) -> Result<Tensor> {
    Tensor::from_vec(data.to_vec(), dims.as_slice(), device)
}

fn values(x: &Tensor) -> Result<Vec<f32>> {
    x.flatten_all()?.to_vec1()
}

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance,
            "element {i}: actual {a:?}, expected {b:?}, tolerance {tolerance}"
        );
    }
}

// Independent oracle: double-precision distances to adjacent BF16 values,
// rather than the production integer-bias implementation.
fn round64(x: f64) -> f32 {
    let x = x as f32;
    let lo = x.to_bits() & 0xffff_0000;
    let hi = lo + 0x1_0000;
    let a = f32::from_bits(lo);
    let b = f32::from_bits(hi);
    let da = (f64::from(x) - f64::from(a)).abs();
    let db = (f64::from(x) - f64::from(b)).abs();
    if db < da || (da == db && (hi >> 16) & 1 == 0) {
        b
    } else {
        a
    }
}

#[test]
fn bf16_bits_and_f32_carrier() -> Result<()> {
    let cases = [
        (0x0000_0000, 0x0000_0000),
        (0x8000_0000, 0x8000_0000),
        (0x3f80_8000, 0x3f80_0000),
        (0x3f81_8000, 0x3f82_0000),
        (0xbf80_8000, 0xbf80_0000),
        (0x0000_8000, 0x0000_0000),
        (0x0001_8000, 0x0002_0000),
        (0x7f80_0000, 0x7f80_0000),
        (0xff80_0000, 0xff80_0000),
        (0x7fc1_2345, 0x7fc1_0000),
    ];
    for (input, expected) in cases {
        assert_eq!(bf16_scalar(f32::from_bits(input)).to_bits(), expected);
    }
    let x = tensor(&[70_000.0, 1.003_906_3, 1.011_718_8], [1, 3], &Device::Cpu)?;
    let y = bf16(&x)?;
    assert_eq!(y.dtype(), DType::F32);
    assert_eq!(y.dims(), &[1, 3]);
    assert_eq!(values(&y)?, [70_144.0, 1.0, 1.015_625]);
    Ok(())
}

#[test]
fn quantization_literal_ties_saturation_and_tiny() -> Result<()> {
    // Raw scale exactly 1. All tie inputs survive initial BF16 rounding.
    let mut raw = vec![0.0; 32];
    raw[..8].copy_from_slice(&[
        0.000_976_562_5,
        0.002_929_687_5,
        1.0625,
        1.1875,
        432.0,
        -432.0,
        -0.0,
        448.0,
    ]);
    let x = tensor(&raw, [32], &Device::Cpu)?;
    let y = values(&quantize_dequantize(&x, CacheFormat::RawKvFp8E8M0)?)?;
    close(
        &y[..8],
        &[0.0, 0.003_906_25, 1.0, 1.25, 448.0, -448.0, -0.0, 448.0],
        0.0,
    );
    assert_eq!(y[6].to_bits(), (-0.0f32).to_bits());
    assert_eq!(nearest_e4m3(1000.0), 448.0);
    assert_eq!(nearest_e4m3(0.000_976_562_5), 0.0);
    assert_eq!(nearest_e4m3(0.002_929_687_5), 0.003_906_25);

    let ties = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0];
    let expected = [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0];
    for format in &FORMATS[1..] {
        let mut input = vec![0.0; format.block()];
        input[..8].copy_from_slice(&ties);
        let x = tensor(&input, [input.len()], &Device::Cpu)?;
        let y = values(&quantize_dequantize(&x, *format)?)?;
        assert_eq!(&y[..8], &expected);
    }
    assert_eq!(nearest_e2m1(1000.0), 6.0);
    assert_eq!(quant_scale(0.0, 1).to_bits(), 0x3480_0000); // 2^-22
    assert_eq!(quant_scale(0.0, 2).to_bits(), 0x0080_0000); // 2^-126
    assert_eq!(quant_scale(0.0, 3), 0.001_953_125);
    let tiny = f32::from_bits(0x0080_0000);
    let x = tensor(&[tiny, -tiny, 0.0], [3, 1], &Device::Cpu)?;
    let y = values(&quantize_dequantize(&x, CacheFormat::IndexQkFp4E8M0)?)?;
    assert_eq!(y, [tiny, -tiny, 0.0]);

    // Amax AFTER BF16: 6.001 rounds down to 6, retaining scale 1, not 2.
    let x = tensor(&[6.001, 0.75, -0.75], [3], &Device::Cpu)?;
    assert_eq!(
        values(&quantize_dequantize(&x, CacheFormat::IndexQkFp4E8M0)?)?,
        [6.0, 1.0, -1.0]
    );
    // E4M3 scale nearest even, not pow2ceil: 6 * 1.1875 -> scale 1.25.
    let x = tensor(&[7.125, 1.875, -1.875], [3], &Device::Cpu)?;
    assert_eq!(
        values(&quantize_dequantize(&x, CacheFormat::CompressedKvFp4E4M3)?)?,
        [7.5, 1.875, -1.875]
    );
    let x = tensor(&[10_000.0, -10_000.0, 448.0], [3], &Device::Cpu)?;
    assert_eq!(
        values(&quantize_dequantize(&x, CacheFormat::CompressedKvFp4E4M3)?)?,
        [2688.0, -2688.0, 448.0]
    );
    Ok(())
}

fn e4_levels() -> Vec<f64> {
    (0..127)
        .map(|code| {
            let e = code / 8;
            let m = code % 8;
            if e == 0 {
                f64::from(m) / 512.0
            } else {
                (1.0 + f64::from(m) / 8.0) * 2.0f64.powi(e - 7)
            }
        })
        .collect()
}

fn nearest64(x: f64, levels: &[f64]) -> f64 {
    let mut best = 0;
    for i in 1..levels.len() {
        let a = (x - levels[i]).abs();
        let b = (x - levels[best]).abs();
        if a < b || (a == b && i % 2 == 0 && best % 2 != 0) {
            best = i;
        }
    }
    levels[best]
}

// Brute-force level enumeration, double-precision distances/log2/ceil.
fn quant_reference(input: &[f32], width: usize, format: CacheFormat) -> Vec<f32> {
    let mut output = Vec::new();
    let scale_levels = e4_levels();
    let levels = if format == CacheFormat::RawKvFp8E8M0 {
        e4_levels()
    } else {
        vec![0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    };
    for row in input.chunks(width) {
        for block in row.chunks(format.block()) {
            let rounded: Vec<f32> = block.iter().map(|&v| round64(f64::from(v))).collect();
            let max = rounded
                .iter()
                .map(|x| f64::from(x.abs()))
                .fold(0.0, f64::max);
            let scale = match format {
                CacheFormat::RawKvFp8E8M0 => 2.0f64.powf((max.max(1.0e-4) / 448.0).log2().ceil()),
                CacheFormat::IndexQkFp4E8M0 => {
                    2.0f64.powf((max.max(7.052966104933725e-38) / 6.0).log2().ceil())
                }
                CacheFormat::CompressedKvFp4E4M3 => {
                    nearest64(max.max(0.01171875) / 6.0, &scale_levels)
                }
            };
            for v in rounded {
                let q = nearest64(f64::from(v.abs()) / scale, &levels).copysign(f64::from(v));
                output.push(round64(q * scale));
            }
        }
    }
    output
}

fn row_fixture(width: usize) -> Vec<f32> {
    (0..3 * width)
        .map(|i| ((i * 73 % 211) as f32 - 105.0) / 31.0)
        .collect()
}

#[test]
fn quantization_independent_oracle_rows_and_rotary_tail() -> Result<()> {
    for width in [1, 7, 17, 33, 128, 512] {
        let data = row_fixture(width);
        let x = tensor(&data, [3, width], &Device::Cpu)?;
        for format in FORMATS {
            let y = values(&quantize_dequantize(&x, format)?)?;
            close(&y, &quant_reference(&data, width, format), 0.0);
        }
    }
    Ok(())
}

#[test]
fn rms_double_precision_and_epsilon() -> Result<()> {
    for width in [1, 3, 35] {
        let data: Vec<f32> = (0..2 * width).map(|i| (i as f32 - 0.7) * 1.0e-11).collect();
        let weight: Vec<f32> = (0..width).map(|i| 0.4 + i as f32 / 11.0).collect();
        let x = tensor(&data, [2, width], &Device::Cpu)?;
        let w = tensor(&weight, [width], &Device::Cpu)?;
        let y = values(&rms_norm(&x, &w, RMS_EPS)?)?;
        let mut expected = Vec::new();
        for row in data.chunks(width) {
            let mean = row.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / width as f64;
            for (i, &v) in row.iter().enumerate() {
                expected.push(round64(
                    f64::from(v) / (mean + f64::from(RMS_EPS)).sqrt() * f64::from(weight[i]),
                ));
            }
        }
        close(&y, &expected, 0.0);
    }
    Ok(())
}

#[test]
fn rope_source_goldens_and_layer_families() -> Result<()> {
    let raw = Rope::for_layer(0)?;
    assert_eq!(raw.frequencies, Rope::for_layer(1)?.frequencies);
    for layer in 3..40 {
        assert_eq!(
            Rope::for_layer(2)?.frequencies,
            Rope::for_layer(layer)?.frequencies
        );
    }
    assert_eq!(raw.frequencies[0], 1.0);
    assert_eq!(raw.frequencies[16], 0.01);
    let compressed = Rope::for_layer(2)?;
    assert!((compressed.frequencies[16] - 0.002_265_625).abs() < 3.0e-10);
    for width in [128, 512] {
        let mut data = vec![1.003_906_3; width];
        for pair in 0..32 {
            data[width - 64 + 2 * pair] = 1.0;
            data[width - 63 + 2 * pair] = 0.0;
        }
        let x = tensor(&data, [1, width], &Device::Cpu)?;
        // Source equations: cos(7),sin(7); raw lane16 pos100 gives theta=1.
        // Compressed lane16 pos256 gives theta=.58 after YaRN interpolation.
        let y = values(&raw.apply(&x, 7, false, false)?)?;
        assert_eq!(&y[width - 64..width - 62], &[0.753_906_25, 0.65625]);
        assert_eq!(y[0], data[0]);
        let y = values(&raw.apply(&x, 100, false, false)?)?;
        assert_eq!(&y[width - 32..width - 30], &[0.539_062_5, 0.839_843_75]);
        let y = values(&compressed.apply(&x, 256, false, false)?)?;
        assert_eq!(&y[width - 32..width - 30], &[0.835_937_5, 0.546_875]);
        let inverse = values(&raw.apply(&x, 7, true, true)?)?;
        assert_eq!(inverse[0], 1.0);
        assert_eq!(&inverse[width - 64..width - 62], &[0.753_906_25, -0.65625]);
    }
    Ok(())
}

fn hc_reference(residual: &[f32], fc: &[f32], scale: &[f32], base: &[f32]) -> Vec<f32> {
    let n = residual.len();
    let mean = residual.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / n as f64;
    let rms = (mean + f64::from(RMS_EPS)).sqrt();
    let z: Vec<f64> = fc
        .chunks(n)
        .enumerate()
        .map(|(i, row)| {
            let dot = row
                .iter()
                .zip(residual)
                .map(|(&w, &x)| f64::from(w) * f64::from(x) / rms)
                .sum::<f64>();
            let part = if i < 4 {
                0
            } else if i < 8 {
                1
            } else {
                2
            };
            dot * f64::from(scale[part]) + f64::from(base[i])
        })
        .collect();
    let eps = f64::from(HC_EPS);
    let mut out = vec![0.0; 24];
    for i in 0..4 {
        out[i] = (1.0 / (1.0 + (-z[i]).exp()) + eps) as f32;
        out[4 + i] = (2.0 / (1.0 + (-z[4 + i]).exp())) as f32;
    }
    let mut matrix = [[0.0f64; 4]; 4];
    for r in 0..4 {
        let peak = z[8 + 4 * r..12 + 4 * r]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        for c in 0..4 {
            matrix[r][c] = (z[8 + 4 * r + c] - peak).exp();
        }
        let sum: f64 = matrix[r].iter().sum();
        for v in &mut matrix[r] {
            *v = *v / sum + eps;
        }
    }
    for iteration in 0..20 {
        if iteration > 0 {
            for row in &mut matrix {
                let sum = row.iter().sum::<f64>() + eps;
                for v in row {
                    *v /= sum;
                }
            }
        }
        for c in 0..4 {
            let sum = matrix.iter().map(|r| r[c]).sum::<f64>() + eps;
            for row in &mut matrix {
                row[c] /= sum;
            }
        }
    }
    for r in 0..4 {
        for c in 0..4 {
            out[8 + 4 * r + c] = matrix[r][c] as f32;
        }
    }
    out
}

fn hc_inputs(width: usize) -> (Vec<f32>, Vec<f32>, [f32; 3], Vec<f32>) {
    let residual = (0..4 * width).map(|i| (i as f32 - 4.3) / 7.0).collect();
    let fc = (0..24 * 4 * width)
        .map(|i| ((i * 17 % 31) as f32 - 15.0) / 91.0)
        .collect();
    let base = (0..24)
        .map(|i| ((i * 13 % 29) as f32 - 14.0) / 3.0)
        .collect();
    (residual, fc, [0.7, 1.1, 0.9], base)
}

fn split_fixture(width: usize, device: &Device) -> Result<(Tensor, HcSplit)> {
    let (r, fc, scale, base) = hc_inputs(width);
    let residual = tensor(&r, [4, width], device)?;
    let split = hc_split(
        &residual,
        &tensor(&fc, [24, 4 * width], device)?,
        &tensor(&scale, [3], device)?,
        &tensor(&base, [24], device)?,
    )?;
    Ok((residual, split))
}

#[test]
fn hc_double_precision_previous_pre_and_transposed_expand() -> Result<()> {
    let device = Device::Cpu;
    for width in [1, 3, 35] {
        let (r, fc, scale, base) = hc_inputs(width);
        let (residual, split) = split_fixture(width, &device)?;
        let expected = hc_reference(&r, &fc, &scale, &base);
        close(&values(&split.packed)?, &expected, 8.0e-7);
        let previous = tensor(&[1.0, 0.0, 0.0, 0.0], [4], &device)?;
        let collapsed = hc_collapse(&residual, &previous)?;
        let first: Vec<f32> = r[..width].iter().map(|&x| round64(f64::from(x))).collect();
        close(&values(&collapsed)?, &first, 0.0);
        assert_ne!(
            values(&collapsed)?,
            values(&hc_collapse(&residual, split.pre())?)?
        );
        let block: Vec<f32> = (0..width).map(|i| (i as f32 - 1.3) / 11.0).collect();
        let expanded = hc_expand(&tensor(&block, [width], &device)?, &residual, &split)?;
        let mut gold = vec![0.0; 4 * width];
        for dest in 0..4 {
            for i in 0..width {
                let mut sum = f64::from(block[i]) * f64::from(expected[4 + dest]);
                for src in 0..4 {
                    sum += f64::from(expected[8 + 4 * src + dest]) * f64::from(r[src * width + i]);
                }
                gold[dest * width + i] = round64(sum);
            }
        }
        close(&values(&expanded)?, &gold, 0.0);
    }
    Ok(())
}

fn gate_reference(r: &[f32], kv: &[f32], q: &[f32], k: &[f32], width: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for stream in 0..4 {
        let offset = stream * width;
        let (mut h2, mut k2, mut dot) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..width {
            let h = f64::from(r[offset + i]);
            let key = f64::from(round64(f64::from(kv[offset + i])));
            h2 += h * h;
            k2 += key * key;
            dot += h * f64::from(q[offset + i]) * f64::from(k[offset + i]) * key;
        }
        dot /= (h2 / width as f64 + f64::from(RMS_EPS)).sqrt()
            * (k2 / width as f64 + f64::from(RMS_EPS)).sqrt()
            * (width as f64).sqrt();
        let gate = 1.0 / (1.0 + (-dot.abs().max(1.0e-6).sqrt().copysign(dot)).exp());
        for i in 0..width {
            out.push(round64(
                f64::from(r[offset + i]) + gate * f64::from(round64(f64::from(kv[4 * width + i]))),
            ));
        }
    }
    out
}

fn gate_inputs(width: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let kv = (0..5 * width)
        .map(|i| ((i * 13 % 23) as f32 - 10.4) / 19.0)
        .collect();
    let q = (0..4 * width).map(|i| 0.5 + i as f32 / 83.0).collect();
    let k = (0..4 * width).map(|i| -0.7 + i as f32 / 37.0).collect();
    (kv, q, k)
}

#[test]
fn engram_gate_double_precision_and_zero_floor() -> Result<()> {
    for width in [1, 3, 35] {
        let r: Vec<f32> = (0..4 * width).map(|i| (i as f32 - 3.9) / 17.0).collect();
        let (kv, q, k) = gate_inputs(width);
        let d = Device::Cpu;
        let y = engram_add(
            &tensor(&r, [4, width], &d)?,
            &tensor(&kv, [5, width], &d)?,
            &tensor(&q, [4, width], &d)?,
            &tensor(&k, [4, width], &d)?,
            RMS_EPS,
        )?;
        close(&values(&y)?, &gate_reference(&r, &kv, &q, &k, width), 0.0);
    }
    let d = Device::Cpu;
    // Cancellation makes the +sqrt(1e-6) zero-dot gate floor visible even
    // after BF16 output rounding; a plain sigmoid(0) would return zero.
    let r = tensor(&[-500.0; 4], [4, 1], &d)?;
    let kv = tensor(&[0.0, 0.0, 0.0, 0.0, 1000.0], [5, 1], &d)?;
    let w = Tensor::ones((4, 1), DType::F32, &d)?;
    assert_eq!(values(&engram_add(&r, &kv, &w, &w, RMS_EPS)?)?, [0.25; 4]);
    Ok(())
}

#[test]
fn router_extremes_bias_ties_and_errors() -> Result<()> {
    let logits = [-80.0, -30.0, -4.0, 0.0, 20.0, 40.0, 1.0, -2.0];
    let selected = router_top6(&logits, None)?;
    assert_eq!(selected.indices, [5, 4, 6, 3, 7, 2]);
    let probabilities: Vec<f64> = selected
        .indices
        .iter()
        .map(|&i| f64::from(logits[i]).exp().ln_1p().sqrt())
        .collect();
    let sum = probabilities.iter().sum::<f64>();
    for (i, &probability) in probabilities[..6].iter().enumerate() {
        assert!((f64::from(selected.weights[i]) - probability / sum * 1.5).abs() < 2.0e-7);
    }
    let selected = router_top6(&[-80.0; 6], None)?;
    assert!(selected.weights.iter().all(|&w| w > 0.0 && w < 1.0e-10));
    assert_eq!(router_top6(&[0.0; 8], None)?.indices, [0, 1, 2, 3, 4, 5]);
    assert_eq!(router_top6(&[0.0; 7], None)?.indices, [0, 1, 2, 3, 4, 5]);
    // Equal-score ordering is the source network, not a stable index sort.
    assert_eq!(
        router_top6(&[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0], None)?.indices,
        [1, 3, 5, 7, 6, 4]
    );
    let bias = [100.0, 90.0, 80.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let biased = router_top6(&logits, Some(&bias))?;
    assert_eq!(&biased.indices[..3], &[0, 1, 2]);
    assert!(biased.weights[0] > 0.0 && biased.weights[0] < 1.0e-15);
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(router_top6(&[invalid; 6], None).is_err());
        assert!(router_top6(&[0.0; 6], Some(&[invalid; 6])).is_err());
    }
    assert!(router_top6(&[0.0; 5], None).is_err());
    assert!(router_top6(&[0.0; 6], Some(&[0.0; 7])).is_err());
    Ok(())
}

#[test]
fn router_rounds_quotient_before_scaling() -> Result<()> {
    let logits = [21.0f32, 22.0, 23.0, 24.0, 25.0, 40.0];
    let y = router_top6(&logits, None)?;
    assert_eq!(y.indices, [5, 4, 3, 2, 1, 0]);
    let p = y.indices.map(|i| logits[i].sqrt());
    let sum = ((p[0] + p[4]) + p[2]) + ((p[1] + p[5]) + p[3]);
    let mut distinguishes_unrounded_quotient = false;
    for (i, &probability) in p.iter().enumerate() {
        let q = (f64::from(probability) / f64::from(sum)) as f32;
        let expected = (f64::from(q) * 1.5) as f32;
        assert_eq!(y.weights[i].to_bits(), expected.to_bits());
        let fused = (f64::from(probability) / f64::from(sum) * 1.5) as f32;
        distinguishes_unrounded_quotient |= fused.to_bits() != expected.to_bits();
    }
    assert!(distinguishes_unrounded_quotient);
    Ok(())
}

#[test]
fn malformed_shapes_kinds_and_offsets() -> Result<()> {
    let d = Device::Cpu;
    let x = Tensor::ones((2, 3), DType::F32, &d)?;
    let weight = Tensor::ones(3, DType::F32, &d)?;
    for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(rms_norm(&x, &weight, epsilon).is_err());
    }
    assert!(bf16(&x.to_dtype(DType::F16)?).is_err());
    assert!(bf16(&x.to_dtype(DType::BF16)?).is_err());
    assert!(bf16(&x.t()?).is_err());
    assert!(bf16(&Tensor::zeros(0, DType::F32, &d)?).is_err());
    let scalar = Tensor::new(1.0f32, &d)?;
    assert!(quantize_dequantize(&scalar, CacheFormat::RawKvFp8E8M0).is_err());
    assert!(rms_norm(&x, &Tensor::ones(2, DType::F32, &d)?, RMS_EPS).is_err());
    assert!(hc_collapse(&x, &Tensor::ones(4, DType::F32, &d)?).is_err());
    assert!(Rope::for_layer(40).is_err());
    assert!(Rope::for_layer(2)?.apply(&x, 1, false, false).is_err());
    let rope_x = Tensor::zeros(128, DType::F32, &d)?;
    assert!(
        Rope::for_layer(0)?
            .apply(&rope_x, 1_048_576, false, false)
            .is_err()
    );
    let x = tensor(&[99.0, 1.011_718_8, -1.011_718_8, 99.0], [4], &d)?.narrow(0, 1, 2)?;
    assert_eq!(values(&bf16(&x)?)?, [1.015_625, -1.015_625]);
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn real_metal_all_primitives_and_token_pipeline() -> Result<()> {
    let device = Device::new_metal(0)?;
    assert!(device.is_metal());
    eprintln!("V41_MATH_METAL_USED: native CustomOps, no host math fallback");
    for width in [1, 3, 17, 33, 128, 512] {
        let data = row_fixture(width);
        let cpu = tensor(&data, [3, width], &Device::Cpu)?;
        let gpu = cpu.to_device(&device)?;
        let rounded = bf16(&gpu)?;
        assert!(rounded.device().is_metal());
        close(&values(&rounded)?, &values(&bf16(&cpu)?)?, 0.0);
        let weight = Tensor::ones(width, DType::F32, &Device::Cpu)?;
        close(
            &values(&rms_norm(&gpu, &weight.to_device(&device)?, RMS_EPS)?)?,
            &values(&rms_norm(&cpu, &weight, RMS_EPS)?)?,
            0.0,
        );
        for format in FORMATS {
            let quantized = quantize_dequantize(&gpu, format)?;
            assert!(quantized.device().is_metal());
            close(
                &values(&quantized)?,
                &quant_reference(&data, width, format),
                0.0,
            );
        }
        if matches!(width, 128 | 512) {
            for layer in [0, 1, 2, 20, 39] {
                let rope = Rope::for_layer(layer)?;
                for position in [7, 256, 65_537, 1_048_575] {
                    for (inverse, round_all) in [(false, false), (true, true)] {
                        let a = rope.apply(&gpu, position, inverse, round_all)?;
                        let b = rope.apply(&cpu, position, inverse, round_all)?;
                        close(&values(&a)?, &values(&b)?, 0.0);
                        for format in FORMATS {
                            close(
                                &values(&quantize_dequantize(&a, format)?)?,
                                &quant_reference(&values(&b)?, width, format),
                                0.0,
                            );
                        }
                    }
                }
            }
        }
    }
    // Real token composition: split current gates, collapse previous pre,
    // expand current post/comb, then Engram gate/add. No model is involved.
    for width in [1, 3, 35, 128] {
        let (r, fc, scale, base) = hc_inputs(width);
        let make = |d: &Device| -> Result<(Tensor, HcSplit)> {
            let (residual, current) = split_fixture(width, d)?;
            let previous = tensor(&[1.0, 0.0, 0.0, 0.0], [4], d)?;
            let block = hc_collapse(&residual, &previous)?;
            let out = hc_expand(&block, &residual, &current)?;
            Ok((out, current))
        };
        let (cpu, cpu_split) = make(&Device::Cpu)?;
        let (gpu, gpu_split) = make(&device)?;
        close(
            &values(&gpu_split.packed)?,
            &hc_reference(&r, &fc, &scale, &base),
            3.0e-6,
        );
        close(&values(&gpu)?, &values(&cpu)?, 0.0);
        close(&values(gpu_split.pre())?, &values(cpu_split.pre())?, 3.0e-6);
        // A nonzero buffer start offset, distinct from a packed gate view.
        let previous = tensor(&[99.0, 1.0, 0.0, 0.0, 0.0], [5], &device)?.narrow(0, 1, 4)?;
        let collapsed = hc_collapse(&tensor(&r, [4, width], &device)?, &previous)?;
        let first: Vec<f32> = r[..width].iter().map(|&x| round64(f64::from(x))).collect();
        close(&values(&collapsed)?, &first, 0.0);
        let (kv, q, k) = gate_inputs(width);
        let added = engram_add(
            &gpu,
            &tensor(&kv, [5, width], &device)?,
            &tensor(&q, [4, width], &device)?,
            &tensor(&k, [4, width], &device)?,
            RMS_EPS,
        )?;
        close(
            &values(&added)?,
            &gate_reference(&values(&cpu)?, &kv, &q, &k, width),
            0.0,
        );
    }
    // Tiny FP4 normals must not flush; BF16 carriers must not pass through F16.
    let tiny = f32::from_bits(0x0080_0000);
    let x = tensor(&[tiny, -tiny, 0.0], [3, 1], &device)?;
    assert_eq!(
        values(&quantize_dequantize(&x, CacheFormat::IndexQkFp4E8M0)?)?,
        [tiny, -tiny, 0.0]
    );
    let x = tensor(&[70_000.0, 1.011_718_8], [2], &device)?;
    assert_eq!(values(&bf16(&x)?)?, [70_144.0, 1.015_625]);
    let x = tensor(&[99.0, 1.011_718_8, -1.011_718_8, 99.0], [4], &device)?;
    assert_eq!(
        values(&bf16(&x.narrow(0, 1, 2)?)?)?,
        [1.015_625, -1.015_625]
    );
    metal_boundary_fixtures(device)?;
    eprintln!("V41_MATH_METAL_COMPLETED: all eight kernels and token composition");
    Ok(())
}

#[cfg(target_os = "macos")]
fn metal_boundary_fixtures(device: Device) -> Result<()> {
    let x = tensor(&[6.001, 0.75, -0.75], [3], &device)?;
    assert_eq!(
        values(&quantize_dequantize(&x, CacheFormat::IndexQkFp4E8M0)?)?,
        [6.0, 1.0, -1.0]
    );
    let x = tensor(&[7.125, 1.875, -1.875], [3], &device)?;
    assert_eq!(
        values(&quantize_dequantize(&x, CacheFormat::CompressedKvFp4E4M3)?)?,
        [7.5, 1.875, -1.875]
    );
    let x = tensor(&[10_000.0, -10_000.0, 448.0], [3], &device)?;
    assert_eq!(
        values(&quantize_dequantize(&x, CacheFormat::CompressedKvFp4E4M3)?)?,
        [2688.0, -2688.0, 448.0]
    );
    let x = tensor(&[448.0, 432.0, 1.0625, 1.1875], [4], &device)?;
    assert_eq!(
        values(&quantize_dequantize(&x, CacheFormat::RawKvFp8E8M0)?)?,
        [448.0, 448.0, 1.0, 1.25]
    );
    let x = tensor(&[1.0e-11, -2.0e-11, 3.0e-11], [3], &device)?;
    let weight = Tensor::ones(3, DType::F32, &device)?;
    let cpu = x.to_device(&Device::Cpu)?;
    close(
        &values(&rms_norm(&x, &weight, RMS_EPS)?)?,
        &values(&rms_norm(&cpu, &weight.to_device(&Device::Cpu)?, RMS_EPS)?)?,
        0.0,
    );
    let (residual, current) = split_fixture(3, &device)?;
    let initial = tensor(&[1.0, 0.0, 0.0, 0.0], [4], &device)?;
    let block = hc_collapse(&residual, &initial)?;
    let expanded = hc_expand(&block, &residual, &current)?;
    let kv = Tensor::ones((5, 3), DType::F32, &device)?;
    let weight = Tensor::ones((4, 3), DType::F32, &device)?;
    let added = engram_add(&expanded, &kv, &weight, &weight, RMS_EPS)?;
    assert!(added.device().is_metal());
    let (cpu_residual, cpu_current) = split_fixture(3, &Device::Cpu)?;
    let cpu_block = hc_collapse(&cpu_residual, &initial.to_device(&Device::Cpu)?)?;
    let cpu_expanded = hc_expand(&cpu_block, &cpu_residual, &cpu_current)?;
    let expected = engram_add(
        &cpu_expanded,
        &kv.to_device(&Device::Cpu)?,
        &weight.to_device(&Device::Cpu)?,
        &weight.to_device(&Device::Cpu)?,
        RMS_EPS,
    )?;
    close(&values(&added)?, &values(&expected)?, 0.0);
    let mut data = vec![0.0f32; 128];
    data[96] = 1.0;
    let rotated = Rope::for_layer(39)?.apply(&tensor(&data, [128], &device)?, 256, false, false)?;
    assert_eq!(&values(&rotated)?[96..98], &[0.835_937_5, 0.546_875]);
    assert_eq!(
        values(&bf16(&tensor(&[70_000.0, 1.003_906_3], [2], &device)?)?)?,
        [70_144.0, 1.0]
    );
    let residual = tensor(&[-500.0; 4], [4, 1], &device)?;
    let kv = tensor(&[0.0, 0.0, 0.0, 0.0, 1000.0], [5, 1], &device)?;
    let weight = Tensor::ones((4, 1), DType::F32, &device)?;
    assert_eq!(
        values(&engram_add(&residual, &kv, &weight, &weight, RMS_EPS)?)?,
        [0.25; 4]
    );
    Ok(())
}

// Literal F32 goldens at saturation and BF16 thresholds, not exp-sigmoid
// goldens or CPU/Metal self-comparisons. Native tanh is not correctly rounded
// like libm at z=-18; avoid claiming cross-device bit parity there.
fn stable_gate_goldens(device: &Device) -> Result<()> {
    #[allow(
        clippy::excessive_precision,
        reason = "Keep the literal F32 reference values and BF16 boundary inputs unchanged."
    )]
    let cases = [
        (-100.0f32, 0.000001f32, 0.0f32, 0.0f32),
        (-20.0, 0.000001, 0.0, 0.0),
        (
            -16.0,
            0.0000011192092896,
            0.0000002384185791015625,
            0.0000002384185791015625,
        ),
        (-0.0078125, 0.4980478883, 0.99609375, 0.99609375),
        (0.0, 0.5000010133, 1.0, 1.0),
        (0.0078125, 0.5019541383, 1.00390625, 1.0),
        (20.0, 1.0000009537, 2.0, 2.0),
        (100.0, 1.0000009537, 2.0, 2.0),
    ];
    let mix = Tensor::zeros(24, DType::F32, device)?;
    let scale = Tensor::ones(3, DType::F32, device)?;
    for (z, pre, post, rounded_post) in cases {
        let mut base = vec![0.0f32; 24];
        base[..8].fill(z);
        let base = tensor(&base, [24], device)?;
        let out =
            Op::new(Kind::HcSplit, 24, 1, Shape::from(vec![24])).apply(&mix, &scale, &base)?;
        let out = values(&out)?;
        close(&out[..4], &[pre; 4], pre.abs() * 1.0e-7 + 1.0e-13);
        close(&out[4..8], &[post; 4], 0.0);
        assert_eq!(half::bf16::from_f32(out[4]).to_f32(), rounded_post, "z={z}");
        assert_eq!(
            half::bf16::from_f32(out[0]),
            half::bf16::from_f32(pre),
            "pre z={z}"
        );
    }
    Ok(())
}

#[test]
fn hc_stable_literal_cpu_goldens() -> Result<()> {
    stable_gate_goldens(&Device::Cpu)
}

#[cfg(target_os = "macos")]
#[test]
fn hc_stable_literal_metal_goldens() -> Result<()> {
    let device = Device::new_metal(0)?;
    stable_gate_goldens(&device)?;
    device.synchronize()
}

fn plain_norm(x: &Tensor) -> Result<Tensor> {
    let (rows, width) = rows_width(x)?;
    let mut op = Op::new(Kind::Rms, width, rows, x.shape().clone());
    op.args.eps = RMS_EPS;
    op.apply(x, x, x)
}

fn real_width_norms(device: &Device) -> Result<()> {
    for width in [128, 512, 1280, 5120, 20480] {
        let data: Vec<f32> = (0..2 * width)
            .map(|i| ((i * 73 % 1021) as f32 - 510.0) / 127.0)
            .collect();
        let weights: Vec<f32> = (0..width).map(|i| 0.5 + (i % 17) as f32 / 16.0).collect();
        let x = tensor(&data, [2, width], device)?;
        let w = tensor(&weights, [width], device)?;
        let plain = values(&plain_norm(&x)?)?;
        let weighted = values(&rms_norm(&x, &w, RMS_EPS)?)?;
        for row in 0..2 {
            let sum = data[row * width..(row + 1) * width]
                .iter()
                .map(|&v| f64::from(v).powi(2))
                .sum::<f64>();
            let scale = (sum / width as f64 + f64::from(RMS_EPS)).sqrt().recip();
            for (col, &weight) in weights.iter().enumerate() {
                let i = row * width + col;
                let expected = f64::from(data[i]) * scale;
                assert!(
                    (f64::from(plain[i]) - expected).abs() <= expected.abs() * 4.0e-7 + 1.0e-8,
                    "plain width={width} col={col}"
                );
                let expected = expected * f64::from(weight);
                assert!(
                    (f64::from(weighted[i]) - expected).abs() <= expected.abs() / 256.0 + 1.0e-7,
                    "weighted width={width} col={col}"
                );
            }
        }
    }
    Ok(())
}

// Source-layout golden: the first float4 contributes exactly 1; every other
// float4 contributes 2^-22. All lane and shared sums are exact in F32.
// A 32-scalar-lane reduction instead loses the small squares in lane zero.
fn source_layout_norm_goldens(device: &Device) -> Result<()> {
    for (width, threads) in [
        (128, 32),
        (512, 128),
        (1280, 320),
        (5120, 1024),
        (20480, 1024),
    ] {
        assert_eq!(rms_threads(width), threads);
        let mut row = vec![1.0f32 / 4096.0; width];
        row[..4].copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        let source_sum = 1.0 + (width - 4) as f32 / 16_777_216.0;
        assert_eq!(rms_sum(&row), source_sum);
        let scale = 1.0 / (source_sum / width as f32 + RMS_EPS).sqrt();
        // Nonzero, non-float4-aligned buffer offset must not change the tree.
        let storage = tensor(&[vec![99.0], row.clone()].concat(), [width + 1], device)?;
        let x = storage.narrow(0, 1, width)?.reshape((1, width))?;
        let y = values(&plain_norm(&x)?)?;
        assert!((y[0] - scale).abs() <= scale * 2.0e-7);
        for i in 1..width {
            assert!((y[i] - row[i] * scale).abs() <= (row[i] * scale).abs() * 2.0e-7);
        }
        if width >= 5120 {
            let mut scalar_lanes = [0.0f32; 32];
            for (i, &v) in row.iter().enumerate() {
                scalar_lanes[i % 32] += v * v;
            }
            let old_scale = 1.0 / (sum_simd(scalar_lanes) / width as f32 + RMS_EPS).sqrt();
            #[allow(
                clippy::excessive_precision,
                reason = "These fixed values distinguish BF16 rounding across the two reduction trees."
            )]
            let rounding_edges = [(1.00390525f32, 1.0f32), (1.01171775, 1.0078125)];
            for (target, golden) in rounding_edges {
                let mut weights = vec![1.0f32; width];
                weights[0] = target / scale;
                assert_ne!(
                    half::bf16::from_f32(old_scale * weights[0]).to_f32(),
                    golden
                );
                let w = tensor(&weights, [width], device)?;
                assert_eq!(values(&rms_norm(&x, &w, RMS_EPS)?)?[0], golden);
            }
        }
    }
    Ok(())
}

#[test]
fn rms_real_width_f64_and_source_layout_cpu() -> Result<()> {
    real_width_norms(&Device::Cpu)?;
    source_layout_norm_goldens(&Device::Cpu)
}

#[cfg(target_os = "macos")]
#[test]
fn rms_real_width_f64_and_source_layout_metal() -> Result<()> {
    let device = Device::new_metal(0)?;
    real_width_norms(&device)?;
    source_layout_norm_goldens(&device)?;
    // HC flat residual has width 4*5120. Its plain normalization must stay
    // F32 until projection, rather than going through weighted/BF16 norm.
    let (r, fc, scale, base) = hc_inputs(5120);
    let split = hc_split(
        &tensor(&r, [4, 5120], &device)?,
        &tensor(&fc, [24, 20480], &device)?,
        &tensor(&scale, [3], &device)?,
        &tensor(&base, [24], &device)?,
    )?;
    close(
        &values(&split.packed)?,
        &hc_reference(&r, &fc, &scale, &base),
        4.0e-6,
    );
    device.synchronize()
}
