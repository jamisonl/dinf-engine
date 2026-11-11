//! Native CPU eight-part online attention sum for Phi's F16 cache.
//!
//! Arithmetic follows the pinned ggml CPU one_chunk/reduce_partials functions.
//! MIT, Copyright (c) 2023-2025 The ggml authors.
//! The request policy selects this order by absolute row, not call shape.

fn partial(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    scale: f32,
) -> Result<(f32, f32), &'static str> {
    let dim = q.len();
    let value_dim = 96;
    if !scale.is_finite()
        || scale <= 0.0
        || dim != 96
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

/// Native eight-part sum order. The caller fixes the physical key count
/// from absolute request position, never from chunk or worker count.
/// Corresponds to ops.cpp split dispatch8833 and reduce_partials8714.
pub fn row(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    physical: usize,
) -> Result<(), &'static str> {
    if q.len() != 96
        || out.len() != 96
        || k.is_empty()
        || k.len() != v.len()
        || !k.len().is_multiple_of(96)
        || physical < 512
        || !physical.is_multiple_of(256)
        || k.len() / 96 > physical
    {
        return Err("native split attention extent");
    }
    let visible = k.len() / 96;
    let width = physical / 8;
    let mut maximum = f32::NEG_INFINITY;
    let mut sum = 0f32;
    out.fill(0.);
    for chunk in 0..8 {
        let start = chunk * width;
        let end = ((chunk + 1) * width).min(visible);
        if start >= end {
            continue;
        }
        let mut value = [0f32; 96];
        let (m, s) = partial(
            q,
            &k[start * 96..end * 96],
            &v[start * 96..end * 96],
            &mut value,
            1.,
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
        let inverse = 1. / sum;
        for a in out.iter_mut() {
            *a *= inverse;
        }
    }
    if !out.iter().all(|x| x.is_finite()) {
        return Err("non-finite native split output");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn split_attention_has_fixed_f64_bound_and_checks_extents() {
        let q: Vec<_> = (0..96).map(|i| ((i % 13) as f32 - 6.) / 64.).collect();
        for visible in [257usize, 511, 513, 1793, 3299] {
            let k: Vec<_> = (0..visible * 96)
                .map(|i| ((i % 41) as f32 - 20.) / 128.)
                .collect();
            let v: Vec<_> = (0..visible * 96)
                .map(|i| ((i % 29) as f32 - 14.) / 32.)
                .collect();
            let physical = visible.div_ceil(256) * 256;
            let mut actual = [0f32; 96];
            row(&q, &k, &v, &mut actual, physical).unwrap();
            let mut scores: Vec<f64> = k
                .chunks_exact(96)
                .map(|key| {
                    key.iter()
                        .zip(&q)
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for p in &mut scores {
                *p = (*p - max).exp();
            }
            let sum: f64 = scores.iter().sum();
            for d in 0..96 {
                let expected: f64 = (0..visible)
                    .map(|j| scores[j] * f64::from(v[j * 96 + d]))
                    .sum::<f64>()
                    / sum;
                assert!((f64::from(actual[d]) - expected).abs() <= 2e-5);
            }
            assert!(row(&q, &k, &v, &mut actual, 256).is_err());
            assert!(row(&q, &k, &v, &mut actual, 513).is_err());
            assert!(row(&q, &k[..k.len() - 1], &v, &mut actual, physical).is_err());
        }
        assert!(row(&q, &[], &[], &mut [0.; 96], 512).is_err());
        assert!(row(&q[..95], &[0.; 96], &[0.; 96], &mut [0.; 96], 512).is_err());
        assert!(row(&q, &[0.; 96], &[0.; 96], &mut [0.; 95], 512).is_err());
    }
}
