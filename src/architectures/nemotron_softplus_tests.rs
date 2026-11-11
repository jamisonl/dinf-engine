use super::*;

fn assert_decay(state: &Tensor, gates: &[f32], a: &[f32], rows: usize) -> Result<()> {
    let values = state.flatten_all()?.to_vec1::<f32>()?;
    let size = values.len() / gates.len();
    for (head, (&gate, &a)) in gates.iter().zip(a).enumerate() {
        let expected = (f64::from(a) * f64::from(gate).exp().ln_1p() * rows as f64).exp();
        for &actual in &values[head * size..(head + 1) * size] {
            let relative = (f64::from(actual) - expected).abs() / expected;
            assert!(
                actual.is_finite() && relative < 2e-5,
                "rows={rows}, gate={gate}: decay={actual}, reference={expected}"
            );
        }
    }
    Ok(())
}

fn check_negative_gate_decay(device: &Device) -> Result<()> {
    let gates = [-30.0f32, -20., -15., -8.];
    let a_values = gates.map(|x| (-1.0 / f64::from(x).exp().ln_1p()) as f32);
    let a = Tensor::new(&a_values, device)?;
    let dt_raw = Tensor::new(&gates, device)?;
    let bias = Tensor::zeros(4, DType::F32, device)?;
    for head_dim in [3, 4] {
        let state = Tensor::ones((4, 4, head_dim), DType::F32, device)?;
        let x = Tensor::zeros((4, head_dim), DType::F32, device)?;
        let bc = Tensor::zeros((2, 4), DType::F32, device)?;
        let (_, actual) = mamba2_ssm_step_batched(Mamba2SsmStep {
            state: &state,
            x: &x,
            b: &bc,
            c: &bc,
            a: &a,
            dt_raw: &dt_raw,
            dt_bias: &bias,
            d: &bias,
            n_group: 2,
        })?;
        assert_decay(&actual, &gates, &a_values, 1)?;

        #[cfg(target_os = "macos")]
        if device.is_metal() {
            let conv_dim = 4 * head_dim + 2 * 2 * 4;
            let conv_state = Tensor::zeros((conv_dim, 4), DType::F32, device)?;
            let conv_weight = Tensor::zeros((conv_dim, 4), DType::F32, device)?;
            let conv_bias = Tensor::zeros(conv_dim, DType::F32, device)?;
            for rows in [1, 3, 7] {
                let input = Tensor::zeros((rows, conv_dim), DType::F32, device)?;
                let dt = Tensor::from_vec(gates.repeat(rows), (rows, 4), device)?;
                let (_, _, actual) = super::super::nemotron_mamba::prefill(
                    &input,
                    &conv_state,
                    &state,
                    &conv_weight,
                    &conv_bias,
                    &dt,
                    &a,
                    &bias,
                    &bias,
                    4,
                    4,
                    head_dim,
                    2,
                    4,
                )?;
                assert_decay(&actual, &gates, &a_values, rows)?;
            }
        }
    }
    Ok(())
}

#[test]
fn nemotron_negative_gate_decay_matches_f64_reference_cpu() -> Result<()> {
    check_negative_gate_decay(&Device::Cpu)
}

#[cfg(target_os = "macos")]
#[test]
fn nemotron_negative_gate_decay_matches_f64_reference_metal() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    check_negative_gate_decay(&device)
}
