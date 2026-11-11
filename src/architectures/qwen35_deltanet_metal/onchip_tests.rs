use super::*;

struct Fixture {
    state: Tensor,
    conv: Tensor,
    alpha: Tensor,
    beta: Tensor,
    a: Tensor,
    dt: Tensor,
    rows: usize,
    key_heads: usize,
    value_heads: usize,
}

fn tensor_with_offset(device: &Device, len: usize, salt: usize, scale: f32) -> Result<Tensor> {
    let values = (0..len + 7)
        .map(|index| (((index * 17 + salt) % 53) as f32 - 26.) * scale)
        .collect::<Vec<_>>();
    Tensor::from_vec(values, len + 7, device)?.narrow(0, 7, len)
}

impl Fixture {
    fn new(device: &Device, rows: usize, key_heads: usize, value_heads: usize) -> Result<Self> {
        let conv_dim = key_heads * 256 + value_heads * 128;
        Ok(Self {
            state: tensor_with_offset(device, value_heads * 128 * 128, 3, 0.001)?,
            conv: tensor_with_offset(device, rows * conv_dim, 7, 0.03)?,
            // Include large positive/negative gates, exercising both softplus branches.
            alpha: tensor_with_offset(device, rows * value_heads, 11, 1.2)?,
            beta: tensor_with_offset(device, rows * value_heads, 19, 1.2)?,
            a: Tensor::full(-0.7f32, value_heads + 7, device)?.narrow(0, 7, value_heads)?,
            dt: tensor_with_offset(device, value_heads, 23, 0.01)?,
            rows,
            key_heads,
            value_heads,
        })
    }

    fn run(&self, onchip: bool, record_replay: bool) -> Result<Tensor> {
        self.scan(&self.state, 0, self.rows, onchip, record_replay)
    }

    fn scan(
        &self,
        state: &Tensor,
        start: usize,
        rows: usize,
        onchip: bool,
        record_replay: bool,
    ) -> Result<Tensor> {
        let conv_dim = self.key_heads * 256 + self.value_heads * 128;
        let conv = self.conv.narrow(0, start * conv_dim, rows * conv_dim)?;
        let alpha = self
            .alpha
            .narrow(0, start * self.value_heads, rows * self.value_heads)?;
        let beta = self
            .beta
            .narrow(0, start * self.value_heads, rows * self.value_heads)?;
        state.apply_op3_no_bwd(
            &conv,
            &alpha,
            &DeltaNetPrefill {
                beta_raw: &beta,
                a: &self.a,
                dt_bias: &self.dt,
                seq_len: rows,
                n_k_heads: self.key_heads,
                n_v_heads: self.value_heads,
                key_dim: 128,
                value_dim: 128,
                record_replay,
                onchip_state: onchip,
            },
        )
    }
}

// Metal can round differently when a state load changes address space, even
// with the same scalar reduction order. Keep this tighter than the existing
// recurrent-reference tolerance (2e-5); real-model token parity is a separate gate.
fn assert_parity(actual: &Tensor, expected: &Tensor, label: &str) -> Result<()> {
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(actual.len(), expected.len(), "{label} size");
    let mut max_error = 0_f32;
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "{label}[{index}] nonfinite"
        );
        let error = (actual - expected).abs();
        max_error = max_error.max(error);
        assert!(
            error <= 2e-6,
            "{label}[{index}]: {actual} != {expected} (error {error})"
        );
    }
    eprintln!("onchip parity {label}: max_abs_error={max_error:e}");
    Ok(())
}

#[test]
fn onchip_prefill_outputs_state_and_replay() -> anyhow::Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    anyhow::ensure!(
        onchip_prefill_pipeline(device.as_metal_device()?)?.is_some(),
        "on-chip pipeline unavailable"
    );
    for (key_heads, value_heads, lengths) in [
        (2, 4, vec![2, 7, 8, 17, 31, 32, 127, 128, 129, 257]),
        (16, 32, vec![8, 128]),
        (16, 48, vec![8, 129]),
    ] {
        for rows in lengths {
            let fixture = Fixture::new(&device, rows, key_heads, value_heads)?;
            for tape in [false, true] {
                let expected = fixture.run(false, tape)?;
                let actual = fixture.run(true, tape)?;
                assert_parity(
                    &actual,
                    &expected,
                    &format!("heads={value_heads}, rows={rows}, tape={tape}"),
                )?;
            }
        }
    }
    Ok(())
}

#[test]
fn onchip_prefill_split_carry_and_every_replay_prefix() -> anyhow::Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    anyhow::ensure!(
        onchip_prefill_pipeline(device.as_metal_device()?)?.is_some(),
        "on-chip pipeline unavailable"
    );
    let fixture = Fixture::new(&device, 17, 2, 4)?;
    let full = fixture.run(true, true)?;
    let heads = fixture.value_heads;
    let y_count = fixture.rows * heads * 128;
    let state_count = heads * 128 * 128;
    let key_count = fixture.rows * heads * 128;
    let keys =
        full.narrow(0, y_count + state_count, key_count)?
            .reshape((fixture.rows, heads, 128))?;
    let deltas = full
        .narrow(0, y_count + state_count + key_count, key_count)?
        .reshape((fixture.rows, heads, 128))?;
    let decays = full
        .narrow(
            0,
            y_count + state_count + 2 * key_count,
            fixture.rows * heads,
        )?
        .reshape((fixture.rows, heads))?;
    let initial = fixture.state.reshape((heads, 128, 128))?;
    for prefix in 1..=fixture.rows {
        let generic = fixture.scan(&fixture.state, 0, prefix, false, false)?;
        let expected_state = generic.narrow(0, prefix * heads * 128, state_count)?;
        let replayed = replay(&initial, &keys, &deltas, &decays, prefix)?;
        assert_parity(&replayed, &expected_state, "replay prefix")?;
    }
    for prefix in [2, 7, 8, 15] {
        let first = fixture.scan(&fixture.state, 0, prefix, true, false)?;
        let state = first.narrow(0, prefix * heads * 128, state_count)?;
        let last = fixture.scan(&state, prefix, fixture.rows - prefix, true, false)?;
        assert_parity(
            &first.narrow(0, 0, prefix * heads * 128)?,
            &full.narrow(0, 0, prefix * heads * 128)?,
            "split prefix output",
        )?;
        assert_parity(
            &last,
            &full.narrow(
                0,
                prefix * heads * 128,
                (fixture.rows - prefix) * heads * 128 + state_count,
            )?,
            "split suffix and carry",
        )?;
    }
    Ok(())
}

#[test]
#[ignore = "manual synchronized Metal performance comparison"]
fn bench_onchip_prefill() -> anyhow::Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    anyhow::ensure!(
        onchip_prefill_pipeline(device.as_metal_device()?)?.is_some(),
        "on-chip pipeline unavailable"
    );
    for rows in [2, 7, 8, 17, 32, 128, 257] {
        let fixture = Fixture::new(&device, rows, 16, 32)?;
        for _ in 0..2 {
            assert_parity(
                &fixture.run(true, false)?,
                &fixture.run(false, false)?,
                "benchmark parity",
            )?;
        }
        let mut times = [Vec::new(), Vec::new()];
        for round in 0..7 {
            for enabled in if round % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let started = std::time::Instant::now();
                for _ in 0..10 {
                    let result = fixture.run(enabled, false)?;
                    device.synchronize()?;
                    std::hint::black_box(result);
                }
                times[usize::from(enabled)].push(started.elapsed().as_secs_f64() * 1000. / 10.);
            }
        }
        for time in &mut times {
            time.sort_by(f64::total_cmp);
        }
        println!(
            "onchip_prefill rows={rows} generic_ms={:.6} onchip_ms={:.6} speedup={:.3}",
            times[0][3],
            times[1][3],
            times[0][3] / times[1][3]
        );
    }
    Ok(())
}
