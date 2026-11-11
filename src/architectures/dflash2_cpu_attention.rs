// SPDX-License-Identifier: MIT
// Sequential F32 FMA dots and a scalar port of ggml_v_expf softmax in
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

type AttentionRow = (Vec<f32>, Vec<f32>, Vec<f32>);

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

/// Noncausal attention with F16 cache values already widened to F32.
/// Q and probabilities stay F32. The dot order matches the requested-F32
/// reference repair; softmax uses the pinned native four-lane exp/sum order.
pub fn row(q: &[f32], k: &[f32], v: &[f32], scale: f32) -> Result<AttentionRow, &'static str> {
    let dim = q.len();
    if dim == 0
        || k.is_empty()
        || !k.len().is_multiple_of(dim)
        || v.len() != k.len()
        || !scale.is_finite()
        || scale <= 0.0
        || !q.iter().chain(k).chain(v).all(|x| x.is_finite())
    {
        return Err("finite attention rows and matching extents required");
    }
    let keys = k.len() / dim;
    let mut scores = Vec::with_capacity(keys);
    for key in k.chunks_exact(dim) {
        let mut sum = 0.0f32;
        for (&a, &b) in key.iter().zip(q) {
            sum = a.mul_add(b, sum);
        }
        scores.push(sum);
    }
    let scaled: Vec<_> = scores.iter().map(|&x| x * scale + 0.0f32).collect();
    let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = Vec::with_capacity(keys);
    let mut sum = 0.0f64;
    for begin in (0..keys).step_by(4) {
        let count = (keys - begin).min(4);
        let lanes: Vec<_> = scaled[begin..begin + count]
            .iter()
            .map(|&x| exp_lane(x - max))
            .collect();
        // Masked native padding has exp(-inf)=0 and takes the large-lane branch.
        let large = count < 4 || lanes.iter().any(|x| x.n.abs() > 126.0);
        let mut e = [0.0f32; 4];
        for (i, lane) in lanes.into_iter().enumerate() {
            e[i] = exp_finish(lane, large);
        }
        sum += f64::from((e[0] + e[1]) + (e[2] + e[3]));
        probs.extend_from_slice(&e[..count]);
    }
    let inv = (1.0f64 / sum) as f32;
    for p in &mut probs {
        *p *= inv;
    }
    let mut output = vec![0.0f32; dim];
    for (d, value) in output.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for key in 0..keys {
            acc = v[key * dim + d].mul_add(probs[key], acc);
        }
        *value = acc + 0.0f32;
    }
    if !scores
        .iter()
        .chain(&probs)
        .chain(&output)
        .all(|x| x.is_finite())
    {
        return Err("nonfinite attention result");
    }
    Ok((scores, probs, output))
}
