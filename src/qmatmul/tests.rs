use super::QMatMul;
use candle_core::quantized::{GgmlDType, QMatMul as CandleQMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use std::sync::Arc;

fn exact(a: &Tensor, b: &Tensor) -> Result<()> {
    assert_eq!(a.dims(), b.dims());
    let a = a.flatten_all()?.to_vec1::<f32>()?;
    let b = b.flatten_all()?.to_vec1::<f32>()?;
    assert!(a.iter().all(|x| x.is_finite()) && b.iter().all(|x| x.is_finite()));
    assert!(a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()));
    Ok(())
}

#[test]
fn cpu_projection_variants_keep_candle_outputs() -> Result<()> {
    let weights = Tensor::from_vec(
        (0..16 * 256)
            .map(|i| (i % 37) as f32 / 32. - 0.5)
            .collect::<Vec<_>>(),
        (16, 256),
        &Device::Cpu,
    )?;
    let input = Tensor::from_vec(
        (0..3 * 256)
            .map(|i| (i % 23) as f32 / 16. - 0.625)
            .collect::<Vec<_>>(),
        (3, 256),
        &Device::Cpu,
    )?;
    for kind in [
        GgmlDType::F32,
        GgmlDType::F16,
        GgmlDType::BF16,
        GgmlDType::Q2K,
        GgmlDType::Q4K,
        GgmlDType::Q8_0,
    ] {
        let qt = Arc::new(QTensor::quantize(&weights, kind)?);
        exact(
            &QMatMul::from_arc(Arc::clone(&qt))?.forward(&input)?,
            &CandleQMatMul::from_arc(qt)?.forward(&input)?,
        )?;
    }
    exact(
        &QMatMul::Tensor(weights.clone()).forward(&input)?,
        &CandleQMatMul::Tensor(weights.clone()).forward(&input)?,
    )?;
    let half = weights.to_dtype(DType::F16)?;
    exact(
        &QMatMul::TensorF16(half.clone()).forward(&input)?,
        &CandleQMatMul::TensorF16(half).forward(&input)?,
    )?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn packed_q2(rows: usize, device: &Device) -> Result<QTensor> {
    let mut bytes = vec![0_u8; rows * 84];
    for (row, b) in bytes.chunks_exact_mut(84).enumerate() {
        b[..16].fill(0x10 | ((row % 7 + 1) as u8));
        b[16..80].fill(0x55);
        b[80..82].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        b[82..84].copy_from_slice(&half::f16::from_f32(0.125).to_le_bytes());
    }
    candle_core::quantized::ggml_file::qtensor_from_ggml(
        GgmlDType::Q2K,
        &bytes,
        vec![rows, 256],
        device,
    )
}

#[cfg(target_os = "macos")]
#[test]
fn metal_q2_covers_tails_views_batches_and_dropped_neighbors() -> anyhow::Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    let input = Tensor::from_vec(
        (0..256)
            .map(|i| (i % 13) as f32 / 16. - 0.375)
            .collect::<Vec<_>>(),
        (1, 256),
        &device,
    )?;
    let host = input.to_device(&Device::Cpu)?.to_vec2::<f32>()?.remove(0);
    for rows in [1, 7, 8, 9, 15, 16, 31, 32, 33] {
        let qt = Arc::new(packed_q2(rows, &device)?);
        let projection = QMatMul::from_arc(Arc::clone(&qt))?;
        let QMatMul::MetalQ2(safe) = &projection else {
            anyhow::bail!("Q2 test requires packed dispatch")
        };
        assert_eq!(safe.original_size_bytes(), rows * 84);
        assert_eq!(safe.packed_size_bytes(), rows.div_ceil(8) * 8 * 84);
        assert_eq!(
            safe.padding_size_bytes(),
            (rows.div_ceil(8) * 8 - rows) * 84
        );
        let expected = projection.forward(&input)?;
        let actual = expected.flatten_all()?.to_vec1::<f32>()?;
        let dense = packed_q2(rows, &Device::Cpu)?
            .dequantize(&Device::Cpu)?
            .to_vec2::<f32>()?;
        for (actual, w) in actual.iter().zip(&dense) {
            let mut sum = 0.;
            let mut sumabs = 0.;
            for (&x, &w) in host.iter().zip(w) {
                let v = f64::from(x) * f64::from(w);
                sum += v;
                sumabs += v.abs();
            }
            assert!(actual.is_finite() && (f64::from(*actual) - sum).abs() <= 1e-4 + 2e-5 * sumabs);
        }
        let offset = Tensor::cat(&[&input, &input], 0)?.narrow(0, 1, 1)?;
        let strided = Tensor::stack(&[&input, &input], 2)?
            .narrow(2, 0, 1)?
            .squeeze(2)?;
        exact(&projection.forward(&offset)?, &expected)?;
        exact(&projection.forward(&strided)?, &expected)?;
        let batch = Tensor::cat(&[&input.unsqueeze(0)?, &input.unsqueeze(0)?], 0)?;
        let output = projection.forward(&batch)?;
        for lane in 0..2 {
            exact(&output.narrow(0, lane, 1)?.squeeze(0)?, &expected)?;
        }
        // The independent Candle oracle here uses only its matrix kernel.
        let prompt = Tensor::cat(&[&input, &input, &input], 0)?;
        exact(
            &projection.forward(&prompt)?,
            &CandleQMatMul::from_arc(Arc::clone(&qt))?.forward(&prompt)?,
        )?;
        let mut held = Vec::new();
        for i in 0usize..128 {
            let x = input.affine(1., 0.)?;
            let neighbor = projection.forward(&x)?;
            let sum = neighbor.sqr()?.sum_all()?;
            drop((neighbor, sum));
            device.synchronize()?;
            let output = projection.forward(&x)?;
            if i.is_multiple_of(2) {
                held.push(output);
            } else {
                exact(&output, &expected)?;
            }
        }
        for output in held {
            exact(&output, &expected)?;
        }
        assert!(QMatMul::QTensor(qt).forward(&input).is_err());
        assert!(projection.forward(&input.to_dtype(DType::F16)?).is_err());
        assert!(projection.forward(&input.flatten_all()?).is_err());
    }
    Ok(())
}

#[test]
fn metal_quantized_projection_respects_offset_prompt_views() -> anyhow::Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    let weights = Tensor::from_vec(
        (0..64 * 256)
            .map(|i| (i % 29) as f32 / 32. - 0.4)
            .collect::<Vec<_>>(),
        (64, 256),
        &device,
    )?;
    // Three blocks of five rows; the projection sees the middle and last.
    let groups = Tensor::from_vec(
        (0..3 * 5 * 256)
            .map(|i| (i % 41) as f32 / 32. - 0.6)
            .collect::<Vec<_>>(),
        (3, 5, 256),
        &device,
    )?;
    for kind in [GgmlDType::Q8_0, GgmlDType::Q4K] {
        let projection = QMatMul::from_arc(Arc::new(QTensor::quantize(&weights, kind)?))?;
        for group in 1..3 {
            let view = groups.i(group)?;
            assert_ne!(view.layout().start_offset(), 0);
            let fresh = view.force_contiguous()?;
            exact(&projection.forward(&view)?, &projection.forward(&fresh)?)?;
            let row = view.narrow(0, 2, 1)?;
            exact(
                &projection.forward(&row)?,
                &projection.forward(&row.force_contiguous()?)?,
            )?;
        }
    }
    Ok(())
}
