//! Source-only routed-post qualifiers. Explicit --ignored authorization required.
use super::*;
use candle_core::{DType, Device, Layout, Shape};

fn unchecked(dims: Vec<usize>, dtype: DeepSeek4DType, bytes: Tensor) -> Result<DeepSeek4QMatMul> {
    Ok(DeepSeek4QMatMul {
        weights: Arc::new(RawTensor::new(dims, dtype, bytes)?),
        dense_weights: None,
        out_dim: 5120,
        in_dim: 2304,
        v41_q8: false,
    })
}

#[test]
#[ignore = "source preparation only; explicit CPU guard qualification required"]
fn routed_post_down_guards_cpu() -> Result<()> {
    let g = GroupedIq1Geometry::down(6, &[6, 2304])?;
    for count in [0, 1, 5, 7, 12, usize::MAX] {
        assert!(GroupedIq1Geometry::down(count, &[6, 2304]).is_err());
    }
    for dims in [
        vec![],
        vec![13824],
        vec![1, 13824],
        vec![6, 2303],
        vec![1, 6, 2304],
    ] {
        assert!(GroupedIq1Geometry::down(6, &dims).is_err());
    }
    g.validate_weight(&[5120, 2304], DeepSeek4DType::IQ1S, 2_304_000)?;
    for (dims, dtype, bytes) in [
        (vec![2304, 5120], DeepSeek4DType::IQ1S, 2_304_000),
        (vec![5120, 2304], DeepSeek4DType::Q8_0, 2_304_000),
        (vec![5120, 2304], DeepSeek4DType::IQ1S, 2_303_999),
    ] {
        assert!(g.validate_weight(&dims, dtype, bytes).is_err());
    }
    let layout = |offset| Layout::new(Shape::from((6, 2304)), vec![2304, 1], offset);
    assert_eq!(
        grouped_iq1_span(&layout(3), DType::F32, 13824, 13827 * 4)?,
        12
    );
    assert!(grouped_iq1_span(&layout(3), DType::F32, 13824, 13827 * 4 - 1).is_err());
    for offset in [usize::MAX, usize::MAX / 4] {
        assert!(grouped_iq1_span(&layout(offset), DType::F32, 13824, usize::MAX).is_err());
    }
    let overflow = Layout::new(Shape::from((usize::MAX, 2)), vec![2, 1], 0);
    assert!(grouped_iq1_span(&overflow, DType::F32, 13824, usize::MAX).is_err());
    let packed = |offset| Layout::new(Shape::from(2_304_000), vec![1], offset);
    assert_eq!(
        grouped_iq1_span(&packed(50), DType::U8, 2_304_000, 2_304_050)?,
        50
    );
    assert!(grouped_iq1_span(&packed(1), DType::U8, 2_304_000, 2_304_001).is_err());
    assert!(grouped_iq1_span(&packed(50), DType::U8, 2_304_000, 2_304_049).is_err());
    let x = Tensor::zeros((6, 2304), DType::F32, &Device::Cpu)?;
    let bytes = Tensor::zeros(2_304_000, DType::U8, &Device::Cpu)?;
    let mat = unchecked(vec![5120, 2304], DeepSeek4DType::IQ1S, bytes.clone())?;
    assert!(
        DeepSeek4QMatMul::forward_grouped_iq1_down(&x, &[&mat; 6])
            .unwrap_err()
            .to_string()
            .contains("Metal backend")
    );
    for bad in [
        x.to_dtype(DType::F16)?,
        x.to_dtype(DType::BF16)?,
        Tensor::zeros((6, 4608), DType::F32, &Device::Cpu)?.narrow(1, 0, 2304)?,
    ] {
        assert!(DeepSeek4QMatMul::forward_grouped_iq1_down(&bad, &[&mat; 6]).is_err());
    }
    for bad in [
        unchecked(vec![5120, 2304], DeepSeek4DType::Q8_0, bytes.clone())?,
        unchecked(vec![2304, 5120], DeepSeek4DType::IQ1S, bytes.clone())?,
        unchecked(
            vec![5120, 2304],
            DeepSeek4DType::IQ1S,
            bytes.narrow(0, 0, 2_303_999)?,
        )?,
        unchecked(
            vec![5120, 2304],
            DeepSeek4DType::IQ1S,
            Tensor::zeros((2_304_000, 2), DType::U8, &Device::Cpu)?.narrow(1, 0, 1)?,
        )?,
    ] {
        let mut ms = [&mat; 6];
        ms[5] = &bad;
        assert!(DeepSeek4QMatMul::forward_grouped_iq1_down(&x, &ms).is_err());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
#[allow(
    clippy::excessive_precision,
    reason = "Exact binary fractions define quantization boundary fixtures."
)]
fn weights(device: &Device, offset: bool) -> Result<Vec<DeepSeek4QMatMul>> {
    (0..6)
        .map(|slot| {
            let prefix = if offset { 50 * (slot + 1) } else { 0 };
            let mut data = vec![0xff; prefix];
            for row in 0..5120 {
                for block in 0..9 {
                    let mut packed = [0u8; 50];
                    let scale = [0.00006103515625, 0.03125, 0.375, 1., 3., 31.]
                        [(slot + row / 17 + block) % 6];
                    packed[..2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
                    for (j, b) in packed[2..].iter_mut().enumerate() {
                        *b = ((slot * 53 + row * 17 + block * 31 + j * 7) % 256) as u8;
                    }
                    data.extend_from_slice(&packed);
                }
            }
            data.extend_from_slice(&[0xff; 50]);
            let len = data.len();
            let bytes = Tensor::from_vec(data, len, device)?.narrow(0, prefix, 2_304_000)?;
            DeepSeek4QMatMul::new(Arc::new(RawTensor::new(
                vec![5120, 2304],
                DeepSeek4DType::IQ1S,
                bytes,
            )?))
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn finite_bits(t: &Tensor) -> Result<Vec<u32>> {
    t.flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(|v| {
            anyhow::ensure!(v.is_finite(), "nonfinite routed down qualifier output");
            Ok(v.to_bits())
        })
        .collect()
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "source preparation only; exclusive Metal exact-bit qualification required"]
fn routed_post_down_six_distinct_rows_offsets_cancellation_order() -> Result<()> {
    anyhow::ensure!(
        DeepSeek4QMatMul::iq1_simd_enabled(),
        "requires current SIMD forward policy"
    );
    let d = Device::new_metal(0)?;
    for offset in [false, true] {
        let owners = weights(&d, offset)?;
        for case in 0..3 {
            let prefix = if offset { 3 } else { 0 };
            let mut values = vec![999f32; prefix];
            for slot in 0..6 {
                for i in 0..2304 {
                    let v = match case {
                        0 => {
                            if i == 17 + slot * 113 {
                                1. + slot as f32 / 8.
                            } else {
                                0.
                            }
                        }
                        1 => {
                            (if (i + slot) % 2 == 0 { 1. } else { -1. })
                                * (if i % 7 == 0 { 1e12 } else { 1e-12 })
                                * (slot + 1) as f32
                        }
                        _ => (((i * 37 + slot * 71) % 257) as f32 - 128.) / (slot + 7) as f32,
                    };
                    values.push(v);
                }
            }
            let len = values.len();
            let input = Tensor::from_vec(values, len, &d)?
                .narrow(0, prefix, 6 * 2304)?
                .reshape((6, 2304))?;
            for (order, binding, distinct_fixture) in [
                ([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], true),
                ([5, 1, 3, 0, 4, 2], [5, 1, 3, 0, 4, 2], true),
                // Six distinct inputs, repeated expert AND packed-buffer aliases.
                ([0, 1, 2, 3, 4, 5], [5, 1, 5, 0, 1, 2], false),
                ([5, 1, 3, 0, 4, 2], [2, 1, 0, 5, 1, 5], false),
                // Exact duplicate weight+input pairs must produce equal outputs.
                ([0, 1, 0, 3, 1, 5], [5, 1, 5, 0, 1, 2], false),
            ] {
                let rows = order
                    .map(|slot| input.narrow(0, slot, 1))
                    .into_iter()
                    .collect::<candle_core::Result<Vec<_>>>()?;
                let ordered = Tensor::cat(&rows, 0)?;
                // Also exercise the original nonzero activation offset directly.
                let x = if order == [0, 1, 2, 3, 4, 5] {
                    &input
                } else {
                    &ordered
                };
                let ms = binding.map(|slot| &owners[slot]);
                let expected = ms
                    .iter()
                    .enumerate()
                    .map(|(slot, m)| m.forward(&x.narrow(0, slot, 1)?))
                    .collect::<Result<Vec<_>>>()?;
                let actual = DeepSeek4QMatMul::forward_grouped_iq1_down(x, &ms)?;
                let pressure = Tensor::full(999f32, (6, 5120), &d)?;
                d.synchronize()?; // Qualifier boundary only; never integration.
                assert_eq!(actual.dims(), [6, 5120]);
                for slot in 0..6 {
                    let bits = finite_bits(&actual.narrow(0, slot, 1)?)?;
                    assert_eq!(
                        bits,
                        finite_bits(&expected[slot])?,
                        "case={case} offset={offset} order={order:?} binding={binding:?} slot={slot}"
                    );
                    if distinct_fixture {
                        for earlier in 0..slot {
                            assert_ne!(
                                finite_bits(&expected[slot])?,
                                finite_bits(&expected[earlier])?,
                                "distinct fixture outputs must distinguish route rows"
                            );
                        }
                    } else {
                        for earlier in 0..slot {
                            if order[earlier] == order[slot] && binding[earlier] == binding[slot] {
                                assert_eq!(
                                    bits,
                                    finite_bits(&expected[earlier])?,
                                    "duplicate input and packed weight must agree"
                                );
                            }
                        }
                    }
                }
                drop(pressure);
            }
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "source preparation only; exclusive Metal rejection qualification required"]
fn routed_post_down_rejects_foreign_device_unaligned_and_disabled() -> Result<()> {
    let d = Device::new_metal(0)?;
    let owners = weights(&d, false)?;
    let x = Tensor::zeros((6, 2304), DType::F32, &d)?;
    let ms = owners.iter().collect::<Vec<_>>();
    let other = Device::new_metal(0)?;
    assert!(!d.same_device(&other));
    for bad in [
        unchecked(
            vec![5120, 2304],
            DeepSeek4DType::IQ1S,
            Tensor::zeros(2_304_000, DType::U8, &other)?,
        )?,
        unchecked(
            vec![5120, 2304],
            DeepSeek4DType::IQ1S,
            Tensor::zeros(2_304_001, DType::U8, &d)?.narrow(0, 1, 2_304_000)?,
        )?,
    ] {
        let mut rejected = ms.clone();
        rejected[5] = &bad;
        assert!(DeepSeek4QMatMul::forward_grouped_iq1_down(&x, &rejected).is_err());
    }
    assert!(DeepSeek4QMatMul::forward_grouped_iq1_down(&x.to_device(&Device::Cpu)?, &ms).is_err());
    assert!(
        super::metal::forward_grouped_iq1_with_mode(
            &x,
            &ms,
            GroupedIq1Geometry::down(6, x.dims())?,
            false
        )
        .is_err()
    );
    d.synchronize()?;
    Ok(())
}
