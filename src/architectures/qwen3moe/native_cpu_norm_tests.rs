use super::*;
use candle_core::Device;
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
