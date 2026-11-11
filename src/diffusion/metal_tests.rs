//! Regression for the rank > 4 strided-reduction path used by the image VAE.

use candle_core::{DType, Device, Result, Tensor};

#[test]
fn metal_strided_reductions_cover_every_axis_through_rank_seven() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    for rank in 3..=7 {
        let dims = (0..rank)
            .map(|axis| if axis % 2 == 0 { 2 } else { 3 })
            .collect::<Vec<_>>();
        let count = dims.iter().product::<usize>();
        // Exact binary fractions make these small sums exact in F32/BF16;
        // indexing defects cannot hide behind a rounding tolerance.
        let values = (0..count)
            .map(|index| ((index * 7 % 17) as f32 - 8.) * 0.125)
            .collect::<Vec<_>>();
        for dtype in [DType::F32, DType::BF16] {
            let cpu =
                Tensor::from_vec(values.clone(), dims.clone(), &Device::Cpu)?.to_dtype(dtype)?;
            let metal = cpu.to_device(&device)?;
            for axis in 0..rank {
                for (name, expected, actual) in [
                    ("sum", cpu.sum(axis)?, metal.sum(axis)?),
                    ("min", cpu.min(axis)?, metal.min(axis)?),
                    ("max", cpu.max(axis)?, metal.max(axis)?),
                ] {
                    let expected = expected
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let actual = actual
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    assert!(actual.iter().all(|value| value.is_finite()));
                    assert_eq!(
                        actual, expected,
                        "{dtype:?} rank {rank}, axis {axis}, {name}"
                    );
                }
            }
        }
    }
    Ok(())
}
