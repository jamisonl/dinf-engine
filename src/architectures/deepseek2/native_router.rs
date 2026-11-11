// SPDX-License-Identifier: MIT
// Scalar port of the NEON F32 router dot and ggml_v_expf softmax in
// ggml/llama.cpp commit
// 15fa3c493bfcd040b5f4dcb29e1c998a0846de16. The exp routine in ggml is
// adapted from Arm Limited's optimized routine. This preserves that native
// CPU arithmetic policy; it is not a claim of greater exp accuracy.
//
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#[derive(Clone, Copy)]
struct ExpLane {
    n: f32,
    e: u32,
    k: f32,
    j: f32,
}

fn exp_lane(x: f32) -> ExpLane {
    // Exact F32 encodings of the hexadecimal constants in ggml-cpu/vec.h.
    let r = f32::from_bits(0x4b40_0000); // 0x1.8p23
    let z = x.mul_add(f32::from_bits(0x3fb8_aa3b), r); // 0x1.715476p+0
    let n = z - r;
    let b = (-n).mul_add(f32::from_bits(0x3f31_7200), x); // 0x1.62e4p-1
    let b = (-n).mul_add(f32::from_bits(0x35bf_be8e), b); // 0x1.7f7d1cp-20
    let e = z.to_bits().wrapping_shl(23);
    let k = f32::from_bits(e.wrapping_add(1.0f32.to_bits()));
    let u = b * b;
    let p = f32::from_bits(0x3e2a_af33).mul_add(b, f32::from_bits(0x3eff_fedb));
    let q = f32::from_bits(0x3c07_2010).mul_add(b, f32::from_bits(0x3d2b_9f17));
    let j = q.mul_add(u, p).mul_add(u, f32::from_bits(0x3f7f_fff6) * b);
    ExpLane { n, e, k, j }
}

fn exp_finish(v: ExpLane, any_large_lane: bool) -> f32 {
    if !any_large_lane {
        return v.j.mul_add(v.k, v.k);
    }
    // Preserve the native four-lane branch and unsigned wrap operations.
    let d: u32 = if v.n <= 0.0 { 0x8200_0000 } else { 0 };
    let s1 = f32::from_bits(d.wrapping_add(0x7f00_0000));
    let s2 = f32::from_bits(v.e.wrapping_sub(d));
    if v.n.abs() > 192.0 {
        s1 * s1
    } else if v.n.abs() > 126.0 {
        s2.mul_add(v.j, s2) * s1
    } else {
        v.k.mul_add(v.j, v.k)
    }
}

/// Exact fixed native F32 router dot order; input and weight share full width.
fn dot(weight: &[f32], input: &[f32]) -> f32 {
    assert_eq!(weight.len(), input.len());
    assert!(!input.is_empty() && input.len().is_multiple_of(16));
    let mut acc = [[0.0f32; 4]; 4];
    for (w, x) in weight.chunks_exact(16).zip(input.chunks_exact(16)) {
        for (group, values) in acc.iter_mut().enumerate() {
            for (lane, value) in values.iter_mut().enumerate() {
                let i = group * 4 + lane;
                *value = w[i].mul_add(x[i], *value);
            }
        }
    }
    let lanes: [f32; 4] =
        std::array::from_fn(|i| (acc[0][i] + acc[2][i]) + (acc[1][i] + acc[3][i]));
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}
/// Native NEON router SOFT_MAX order, not flash-attention softmax.
fn softmax(logits: &[f32]) -> Vec<f32> {
    assert!(!logits.is_empty() && logits.len().is_multiple_of(4));
    assert!(logits.iter().all(|x| x.is_finite()));
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out = Vec::with_capacity(logits.len());
    let mut sum = 0.0f64;
    for group in logits.chunks_exact(4) {
        let lanes = [
            exp_lane(group[0] - max),
            exp_lane(group[1] - max),
            exp_lane(group[2] - max),
            exp_lane(group[3] - max),
        ];
        let large = lanes.iter().any(|v| v.n.abs() > 126.0);
        let e: [f32; 4] = std::array::from_fn(|i| exp_finish(lanes[i], large));
        sum += ((e[0] + e[1]) + (e[2] + e[3])) as f64;
        out.extend(e);
    }
    let scale = (1.0f64 / sum) as f32;
    for value in &mut out {
        *value *= scale;
    }
    out
}

pub(super) fn supports_weight(weight: &candle_core::Tensor) -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && weight.device().is_cpu()
        && weight.dtype() == candle_core::DType::F32
        && weight.dims() == [64, 2048]
}

/// Return the native router only after the stage admits actual CPU Q8 experts,
/// a stored F32 router and the native cache. Other paths retain their old math.
pub(super) fn router(
    input: &candle_core::Tensor,
    weight: &candle_core::Tensor,
    admitted: bool,
) -> anyhow::Result<Option<(candle_core::Tensor, candle_core::Tensor)>> {
    use anyhow::Context;
    use candle_core::{DType, Storage, Tensor};
    if !admitted
        || !supports_weight(weight)
        || !input.device().same_device(weight.device())
        || input.dtype() != DType::F32
        || input.rank() != 2
    {
        return Ok(None);
    }
    let (rows, width) = input.dims2()?;
    if rows == 0 || width != 2048 {
        return Ok(None);
    }
    let input = input.contiguous()?;
    let weight = weight.contiguous()?;
    let (xs, xl) = input.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let Storage::Cpu(xs) = &*xs else {
        anyhow::bail!("Deep router input storage");
    };
    let Storage::Cpu(ws) = &*ws else {
        anyhow::bail!("Deep router weight storage");
    };
    let (xa, xz) = xl.contiguous_offsets().context("Deep router input span")?;
    let (wa, wz) = wl.contiguous_offsets().context("Deep router weight span")?;
    let x = xs
        .as_slice::<f32>()?
        .get(xa..xz)
        .context("Deep router input bounds")?;
    let w = ws
        .as_slice::<f32>()?
        .get(wa..wz)
        .context("Deep router weight bounds")?;
    anyhow::ensure!(
        rows.checked_mul(width) == Some(x.len()) && w.len() == 64 * width,
        "Deep router logical extents"
    );
    anyhow::ensure!(
        x.iter().chain(w).all(|v| v.is_finite()),
        "Deep router nonfinite input"
    );
    let workers = candle_core::utils::get_num_threads().max(1).min(rows);
    let chunk = rows.div_ceil(workers);
    let slots: Vec<_> = (0..workers).map(|_| std::sync::Mutex::new(None)).collect();
    candle_core::cpu::kernels::par_for_each(workers, |worker| {
        let start = worker * chunk;
        let end = (start + chunk).min(rows);
        let result = (|| -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
            let mut logits = Vec::with_capacity(end.saturating_sub(start) * 64);
            let mut probabilities = Vec::with_capacity(logits.capacity());
            for row in start..end {
                let input = &x[row * width..(row + 1) * width];
                let values: Vec<_> = w
                    .chunks_exact(width)
                    .map(|weight| dot(weight, input))
                    .collect();
                anyhow::ensure!(
                    values.iter().all(|v| v.is_finite()),
                    "Deep router nonfinite logits"
                );
                probabilities.extend(softmax(&values));
                logits.extend(values);
            }
            Ok((logits, probabilities))
        })();
        *slots[worker].lock().expect("new Deep router worker lock") = Some(result);
    });
    let total = rows
        .checked_mul(64)
        .context("Deep router output overflow")?;
    let mut logits = Vec::with_capacity(total);
    let mut probabilities = Vec::with_capacity(total);
    for slot in slots {
        let (values, probs) = slot
            .into_inner()
            .map_err(|_| anyhow::anyhow!("Deep router lock poisoned"))?
            .context("Deep router worker missing")??;
        logits.extend(values);
        probabilities.extend(probs);
    }
    anyhow::ensure!(
        logits.len() == total && probabilities.len() == total,
        "Deep router output extent"
    );
    Ok(Some((
        Tensor::from_vec(logits, (rows, 64), input.device())?,
        Tensor::from_vec(probabilities, (rows, 64), input.device())?,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn bits(t: &Tensor) -> anyhow::Result<Vec<u32>> {
        Ok(t.flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }

    #[test]
    fn scalar_dot_and_router_softmax_have_fixed_f64_bounds() {
        let x: Vec<_> = (0..2048)
            .map(|i| ((i * 173 % 257) as f32 - 128.) / 257.)
            .collect();
        let mut logits = Vec::new();
        for expert in 0..64 {
            let w: Vec<_> = (0..2048)
                .map(|i| ((i * 79 + expert * 31) % 257) as f32 / 2048. - 0.0625)
                .collect();
            let result = dot(&w, &x);
            let products: Vec<_> = w
                .iter()
                .zip(&x)
                .map(|(w, x)| f64::from(*w) * f64::from(*x))
                .collect();
            let expected: f64 = products.iter().sum();
            let bound = 2e-6 + 2e-6 * products.iter().map(|v| v.abs()).sum::<f64>();
            assert!((f64::from(result) - expected).abs() <= bound);
            logits.push(result);
        }
        let result = softmax(&logits);
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<_> = logits
            .iter()
            .map(|x| (f64::from(*x) - f64::from(max)).exp())
            .collect();
        let sum: f64 = exp.iter().sum();
        for (actual, exact) in result.iter().zip(exp) {
            assert!((f64::from(*actual) - exact / sum).abs() <= 2e-6);
        }
        assert_eq!(softmax(&[0.; 64]), vec![1. / 64.; 64]);
        let mut wide = [0.; 64];
        for (i, x) in wide.iter_mut().enumerate() {
            *x = -(i as f32) * 32.;
        }
        let out = softmax(&wide);
        assert!(out.iter().all(|v| v.is_finite() && *v >= 0.));
        assert!((out.iter().sum::<f32>() - 1.).abs() <= 2e-6);
    }

    #[test]
    fn declined_router_leaves_other_paths_available() -> anyhow::Result<()> {
        let w = Tensor::zeros((64, 2048), DType::F32, &Device::Cpu)?;
        let x = Tensor::zeros((1, 2048), DType::F32, &Device::Cpu)?;
        assert!(router(&x, &w, false)?.is_none());
        assert!(router(&x.to_dtype(DType::F16)?, &w, true)?.is_none());
        assert!(router(&x, &w.to_dtype(DType::F16)?, true)?.is_none());
        assert!(router(&x, &w.narrow(0, 0, 63)?, true)?.is_none());
        assert!(router(&x.narrow(1, 0, 2047)?, &w, true)?.is_none());
        assert!(router(&x.reshape((1, 1, 2048))?, &w, true)?.is_none());
        Ok(())
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn tensor_offsets_strides_and_chunks_keep_exact_row_order() -> anyhow::Result<()> {
        let data = |n: usize, salt: usize| {
            (0..n)
                .map(|i| ((i * 97 + salt * 31) % 257) as f32 / 257. - 0.5)
                .collect::<Vec<_>>()
        };
        let x = Tensor::from_vec(data(11 * 2048, 1), (11, 2048), &Device::Cpu)?.narrow(0, 2, 9)?;
        let w = Tensor::from_vec(data(65 * 2048, 2), (65, 2048), &Device::Cpu)?.narrow(0, 1, 64)?;
        assert!(x.layout().start_offset() > 0 && w.layout().start_offset() > 0);
        let (all_logits, all_probs) = router(&x, &w, true)?.unwrap();
        let xv = x.to_vec2::<f32>()?;
        let wv = w.to_vec2::<f32>()?;
        let expected: Vec<_> = xv
            .iter()
            .flat_map(|x| wv.iter().map(|w| dot(w, x)))
            .collect();
        assert_eq!(
            bits(&all_logits)?,
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let expected_probs: Vec<_> = expected
            .chunks_exact(64)
            .flat_map(softmax)
            .map(f32::to_bits)
            .collect();
        assert_eq!(bits(&all_probs)?, expected_probs);
        for size in [1, 2, 4, 8, 32] {
            let mut logits = Vec::new();
            let mut probs = Vec::new();
            for start in (0..9).step_by(size) {
                let (a, b) = router(&x.narrow(0, start, size.min(9 - start))?, &w, true)?.unwrap();
                logits.extend(bits(&a)?);
                probs.extend(bits(&b)?);
            }
            assert_eq!(logits, bits(&all_logits)?);
            assert_eq!(probs, bits(&all_probs)?);
        }
        let strided = Tensor::stack(&[&x, &x], 2)?.narrow(2, 0, 1)?.squeeze(2)?;
        assert!(!strided.is_contiguous());
        assert_eq!(
            bits(&router(&strided, &w, true)?.unwrap().0)?,
            bits(&all_logits)?
        );
        Ok(())
    }
}
