// One fixed online attention order for every absolute causal query.
//
// Arithmetic follows ggml_compute_forward_flash_attn_ext_f16_one_chunk
// in the pinned llama.cpp CPU reference (MIT, Copyright (c) 2023-2025
// The ggml authors). This component ports eight ordered native partials.
// Q/K/V must already contain the exact widened F16 values, and the F32 post-dot scale.

fn partial(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    scale: f32,
) -> Result<(f32, f32), &'static str> {
    let dim = q.len();
    let value_dim = 128;
    if !scale.is_finite()
        || scale <= 0.0
        || dim != 128
        || out.len() != value_dim
        || k.is_empty()
        || k.len() / dim != v.len() / value_dim
        || !v.len().is_multiple_of(value_dim)
        || !k.len().is_multiple_of(dim)
    {
        return Err("native online attention extent");
    }
    let mut sum = 0.0f32;
    let mut maximum = f32::NEG_INFINITY;
    out.fill(0.0);
    for (key, value) in k.chunks_exact(dim).zip(v.chunks_exact(value_dim)) {
        let mut score = 0.0f32;
        for (&a, &b) in key.iter().zip(q) {
            // Products of two finite F16 values are exact in F32. Keep the
            // scalar dimension order of the repaired native QK dot.
            score += a * b;
        }
        score *= scale;
        score += 0.0; // Native visible additive mask.
        let (rescale, probability) = if score > maximum {
            let old = maximum;
            maximum = score;
            let rescale = (old - maximum).exp();
            for x in out.iter_mut() {
                *x *= rescale;
            }
            (rescale, 1.0)
        } else {
            (1.0, (score - maximum).exp())
        };
        for (acc, &x) in out.iter_mut().zip(value) {
            *acc = x.mul_add(probability, *acc);
        }
        sum = sum.mul_add(rescale, probability);
    }
    if !out.iter().all(|x| x.is_finite()) {
        return Err("non-finite native online output");
    }
    Ok((maximum, sum))
}

/// Replay only: the native q1/F16/padded physical/eight-thread split policy.
/// Corresponds to ops.cpp split dispatch8833 and reduce_partials8714.
pub fn split_row(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    scale: f32,
    physical: usize,
) -> Result<(), &'static str> {
    if q.len() != 128
        || out.len() != 128
        || k.is_empty()
        || !k.len().is_multiple_of(128)
        || v.len() != k.len()
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err("native split extent");
    }
    if !(512..=4096).contains(&physical)
        || !physical.is_multiple_of(256)
        || physical != (k.len() / 128).div_ceil(256) * 256
    {
        return Err("native physical split extent");
    }
    let visible = k.len() / 128;
    let mut maximum = f32::NEG_INFINITY;
    let mut sum = 0f32;
    out.fill(0.);
    let width = physical.div_ceil(8);
    for chunk in 0..8 {
        let start = chunk * width;
        let end = ((chunk + 1) * width).min(visible);
        if start >= end {
            continue;
        }
        let mut value = [0f32; 128];
        let (m, s) = partial(
            q,
            &k[start * 128..end * 128],
            &v[start * 128..end * 128],
            &mut value,
            scale,
        )?;
        if s == 0. {
            continue;
        }
        let next = maximum.max(m);
        let old_scale = (maximum - next).exp();
        let new_scale = (m - next).exp();
        for (a, b) in out.iter_mut().zip(value) {
            *a = a.mul_add(old_scale, b * new_scale);
        }
        sum = sum.mul_add(old_scale, s * new_scale);
        maximum = next;
    }
    if sum != 0. {
        let inverse = 1.0 / sum;
        for a in out.iter_mut() {
            *a *= inverse;
        }
    }
    if !out.iter().all(|x| x.is_finite()) {
        return Err("nonfinite split replay");
    }
    Ok(())
}
