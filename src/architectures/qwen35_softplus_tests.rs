use super::*;

#[test]
fn deltanet_negative_gate_decay_matches_f64_reference_cpu() -> Result<()> {
    let device = Device::Cpu;
    let gates = [-30.0f32, -20., -15., -8.];
    let a_values = gates.map(|x| (-1.0 / f64::from(x).exp().ln_1p()) as f32);
    let dt = softplus(&Tensor::new(&gates, &device)?)?;
    let g = (dt * Tensor::new(&a_values, &device)?)?.to_vec1::<f32>()?;
    let state = Tensor::ones((4, 4, 4), DType::F32, &device)?;
    let zero = Tensor::zeros((4, 4), DType::F32, &device)?;
    let (_, state) = deltanet_forward(&zero, &zero, &zero, &[0.5; 4], &g, &state)?;
    check_decay(&state, &gates, &a_values, 1)
}

#[cfg(target_os = "macos")]
#[test]
fn deltanet_negative_gate_decay_matches_f64_reference_metal() -> Result<()> {
    let Some(device) = crate::test_metal_device() else {
        return Ok(());
    };
    const HEADS: usize = 4;
    const DIM: usize = 128;
    let gates = [-30.0f32, -20., -15., -8.];
    // Choose finite A values that expose relative error in dt as a visible
    // decay error. With zero K and V, S' is exactly exp(A*softplus(alpha))*S.
    let a_values = gates.map(|x| (-1.0 / f64::from(x).exp().ln_1p()) as f32);
    let a = Tensor::new(&a_values, &device)?;
    let bias = Tensor::zeros(HEADS, DType::F32, &device)?;
    let state = Tensor::ones((HEADS, DIM, DIM), DType::F32, &device)?;
    let zero = Tensor::zeros((HEADS, DIM), DType::F32, &device)?;
    let (output, actual) = super::super::qwen35_deltanet_metal::step(
        &state,
        &zero,
        &zero,
        &zero,
        &bias,
        &Tensor::new(&gates, &device)?,
        &a,
        &bias,
    )?;
    assert!(
        output
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|x| *x == 0.0)
    );
    check_decay(&actual, &gates, &a_values, 1)?;

    let conv_dim = HEADS * DIM * 3;
    for rows in [1, 3, 7] {
        let qkv = Tensor::zeros((rows, conv_dim), DType::F32, &device)?;
        let conv_state = Tensor::zeros((conv_dim, 4), DType::F32, &device)?;
        let conv_weight = Tensor::zeros((conv_dim, 4), DType::F32, &device)?;
        let alpha = Tensor::from_vec(gates.repeat(rows), (rows, HEADS), &device)?;
        let beta = Tensor::zeros((rows, HEADS), DType::F32, &device)?;
        let gate = Tensor::zeros((1, rows, HEADS * DIM), DType::F32, &device)?;
        let norm = Tensor::ones(DIM, DType::F32, &device)?;
        let (_, _, actual, _, _) = super::super::qwen35_deltanet_metal::prefill(
            &qkv,
            &conv_state,
            &state,
            &conv_weight,
            &alpha,
            &beta,
            &a,
            &bias,
            HEADS,
            HEADS,
            DIM,
            DIM,
            4,
            false,
            &gate,
            &norm,
            1e-5,
            false,
        )?;
        check_decay(&actual, &gates, &a_values, rows)?;
    }
    Ok(())
}

fn check_decay(state: &Tensor, gates: &[f32], a: &[f32], rows: usize) -> Result<()> {
    let values = state.flatten_all()?.to_vec1::<f32>()?;
    let size = values.len() / gates.len();
    for (head, (&gate, &a)) in gates.iter().zip(a).enumerate() {
        let expected = (f64::from(a) * f64::from(gate).exp().ln_1p() * rows as f64).exp();
        for &actual in &values[head * size..(head + 1) * size] {
            let relative = (f64::from(actual) - expected).abs() / expected;
            assert!(
                actual.is_finite() && relative < 2e-5,
                "rows={rows}, gate={gate}: decay={actual}, reference={expected}, relative={relative}"
            );
        }
    }
    Ok(())
}
