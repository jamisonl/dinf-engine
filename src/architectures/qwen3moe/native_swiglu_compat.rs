// SPDX-License-Identifier: MIT
// Scalar port of ggml_v_expf, ggml_v_silu, and the NEON branch of
// ggml_vec_swiglu_f32 in ggml/llama.cpp commit
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

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

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

/// Native ARM NEON SwiGLU policy for complete groups of four F32 values.
/// The Qwen expert adapter admits complete groups of four; no scalar tail.
/// Attention softmax must not use this exp approximation.
pub fn native_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), up.len());
    assert_eq!(gate.len() % 4, 0);
    let mut out = Vec::with_capacity(gate.len());
    for (g, u) in gate.chunks_exact(4).zip(up.chunks_exact(4)) {
        // Native uses zero - x, rather than a sign-bit negation.
        let v = [
            exp_lane(0.0f32 - g[0]),
            exp_lane(0.0f32 - g[1]),
            exp_lane(0.0f32 - g[2]),
            exp_lane(0.0f32 - g[3]),
        ];
        let any_large_lane = v.iter().any(|x| x.n.abs() > 126.0);
        for i in 0..4 {
            let e = exp_finish(v[i], any_large_lane);
            let silu = g[i] / (1.0f32 + e);
            out.push(silu * u[i]);
        }
    }
    out
}
