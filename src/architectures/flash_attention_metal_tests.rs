use candle_core::{DType, Device, Tensor};

use super::try_flash_attention;

fn metal() -> Option<Device> {
    let device = Device::new_metal(0).ok()?;
    super::super::tensor_qmatmul_metal::supports_tensor_batch_attention(&device)
        .ok()?
        .then_some(device)
}

/// Exact attention in F32 on the host over the same F16 inputs.
fn reference(q: &Tensor, k: &Tensor, v: &Tensor, window: usize, scale: f32) -> Vec<f32> {
    let (batch, heads, q_seq, dim) = q.dims4().unwrap();
    let (_, kv_heads, kv_seq, _) = k.dims4().unwrap();
    let read = |t: &Tensor| {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    let (q, k, v) = (read(q), read(k), read(v));
    let gqa = heads / kv_heads;
    let mut out = vec![0f32; batch * heads * q_seq * dim];
    for b in 0..batch {
        for h in 0..heads {
            let kvh = h / gqa;
            for i in 0..q_seq {
                let position = kv_seq - q_seq + i;
                let qrow = &q[((b * heads + h) * q_seq + i) * dim..][..dim];
                let keys: Vec<usize> = (0..=position)
                    .filter(|&j| window == 0 || j + window > position)
                    .collect();
                let scores: Vec<f32> = keys
                    .iter()
                    .map(|&j| {
                        let krow = &k[((b * kv_heads + kvh) * kv_seq + j) * dim..][..dim];
                        qrow.iter().zip(krow).map(|(a, b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f32 = weights.iter().sum();
                let orow = &mut out[((b * heads + h) * q_seq + i) * dim..][..dim];
                for (&j, w) in keys.iter().zip(&weights) {
                    let vrow = &v[((b * kv_heads + kvh) * kv_seq + j) * dim..][..dim];
                    for (o, x) in orow.iter_mut().zip(vrow) {
                        *o += w / sum * x;
                    }
                }
            }
        }
    }
    out
}

fn inputs(
    device: &Device,
    (batch, heads, kv_heads, q_seq, kv_seq, dim): (usize, usize, usize, usize, usize, usize),
) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
    let q = Tensor::randn(0f32, 1.0, (batch, heads, q_seq, dim), device)?.to_dtype(DType::F16)?;
    // Keys and values live in a larger cache, as the KV cache hands them out.
    let capacity = kv_seq + 37;
    let k = Tensor::randn(0f32, 1.0, (batch, kv_heads, capacity, dim), device)?
        .to_dtype(DType::F16)?
        .narrow(2, 0, kv_seq)?;
    let v = Tensor::randn(0f32, 1.0, (batch, kv_heads, capacity, dim), device)?
        .to_dtype(DType::F16)?
        .narrow(2, 0, kv_seq)?;
    Ok((q, k, v))
}

#[test]
fn flash_attention_matches_reference_for_verify_and_prompt_shapes() -> candle_core::Result<()> {
    let Some(device) = metal() else {
        return Ok(());
    };
    for &(shape, window) in &[
        // Verify-sized: a few queries over a long cache, grouped heads.
        ((1, 24, 4, 8, 3000, 256), 0),
        ((1, 24, 4, 8, 3000, 256), 700),
        ((1, 12, 2, 5, 1100, 256), 0),
        ((2, 8, 2, 3, 300, 256), 0),
        ((1, 4, 1, 16, 2100, 512), 0),
        ((1, 8, 8, 2, 4000, 256), 0),
        // Prompt chunks.
        ((1, 24, 4, 70, 1200, 256), 0),
        ((1, 8, 2, 64, 64, 256), 0),
        ((1, 8, 2, 100, 900, 256), 333),
    ] {
        let (q, k, v) = inputs(&device, shape)?;
        let scale = 1.0 / (shape.5 as f32).sqrt();
        let actual = try_flash_attention(&q, &k, &v, window, scale)?
            .expect("flash attention applies")
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = reference(&q, &k, &v, window, scale);
        let worst = actual
            .iter()
            .zip(&expected)
            .map(|(a, e)| (a - e).abs())
            .fold(0f32, f32::max);
        assert!(worst < 4e-3, "{shape:?} window {window}: max error {worst}");
    }
    Ok(())
}

/// cargo test --release -p dinf-engine flash_attention_verify_timing -- --ignored --nocapture
#[test]
#[ignore = "timing"]
fn flash_attention_verify_timing() -> candle_core::Result<()> {
    let Some(device) = metal() else {
        return Ok(());
    };
    for q_seq in [8, 16, 512] {
        for kv_seq in [1024, 4096, 12288, 30720] {
            let (q, k, v) = inputs(&device, (1, 24, 4, q_seq, kv_seq, 256))?;
            for _ in 0..3 {
                try_flash_attention(&q, &k, &v, 0, 0.0625)?;
            }
            device.synchronize()?;
            let runs = 20;
            let started = std::time::Instant::now();
            for _ in 0..runs {
                try_flash_attention(&q, &k, &v, 0, 0.0625)?;
            }
            device.synchronize()?;
            let each = started.elapsed().as_secs_f64() * 1e3 / runs as f64;
            println!(
                "q_seq {q_seq:4} kv {kv_seq:6}: {each:7.3} ms/layer, {:7.1} ms per pass over 16 layers",
                each * 16.0
            );
        }
    }
    Ok(())
}
