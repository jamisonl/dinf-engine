use anyhow::Result;
use candle_core::{DType, Tensor};

/// Numerically stable softplus: log(1 + exp(x)).
pub(super) fn softplus(x: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    // Half-precision exp overflows before the linear branch begins.
    let input = if matches!(dtype, DType::F16 | DType::BF16) {
        x.to_dtype(DType::F32)?
    } else {
        x.clone()
    };
    let x = &input;
    // Correct for rounding in 1 + exp(x): log(sum) * exp(x)/(sum - 1)
    // keeps the negative tail, including values for which sum rounds to 1.
    // Clamp only the positive end to avoid overflow in the unused branch.
    let exponent = x.clamp(f64::NEG_INFINITY, 20.0)?.exp()?;
    let sum = (&exponent + 1.0)?;
    let one = sum.ones_like()?;
    let rounded_to_one = sum.eq(&one)?;
    let denominator = rounded_to_one.where_cond(&one, &(&sum - 1.0)?)?;
    let corrected = (sum.log()? * exponent.broadcast_div(&denominator)?)?;
    let small = rounded_to_one.where_cond(&exponent, &corrected)?;
    let large = x.ge(&(x.ones_like()? * 20.0)?)?;
    large
        .where_cond(x, &small)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn check_softplus(device: &Device) -> Result<()> {
        let values = [
            -80.0f32, -30., -20., -17., -15., -10., -5., -1., 0., 1., 19., 20., 25., 80.,
        ];
        let actual = softplus(&Tensor::new(&values, device)?)?.to_vec1::<f32>()?;
        for (&x, &actual) in values.iter().zip(&actual) {
            let expected = f64::from(x).exp().ln_1p();
            assert!(actual.is_finite());
            let relative = (f64::from(actual) - expected).abs() / expected;
            assert!(
                relative < 2e-6,
                "softplus({x})={actual}, reference={expected}, relative={relative}"
            );
        }
        // Half-precision inputs must not overflow exp(x) before the linear
        // branch starts; softplus(19) is finite in both input formats.
        let values = [-15.0f32, -10., -1., 0., 10., 12., 19., 20., 25.];
        for dtype in [DType::F16, DType::BF16] {
            let input = Tensor::new(&values, device)?.to_dtype(dtype)?;
            let actual = softplus(&input)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            for (&x, actual) in values.iter().zip(actual) {
                let expected = f64::from(x).exp().ln_1p();
                let expected = match dtype {
                    DType::F16 => half::f16::from_f64(expected).to_f32(),
                    DType::BF16 => half::bf16::from_f64(expected).to_f32(),
                    _ => unreachable!(),
                };
                assert_eq!(actual, expected, "{dtype:?} softplus({x})");
            }
        }
        Ok(())
    }

    #[test]
    fn softplus_preserves_negative_tail_cpu() -> Result<()> {
        check_softplus(&Device::Cpu)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn softplus_preserves_negative_tail_metal() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        check_softplus(&device)
    }
}
