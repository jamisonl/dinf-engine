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

// YaRN terms: Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
// Same MIT terms above. Arithmetic follows pinned ggml rope_yarn/cache/rotate_pairs.

// DeepSeek's D64 normal YaRN CPU policy. macOS ARM64 only.
// Port of ggml's MIT-licensed rope_yarn/cache/rotate_pairs arithmetic.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[repr(C)]
struct SinCos {
    sin: f32,
    cos: f32,
}
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe extern "C" {
    fn __sincosf_stret(x: f32) -> SinCos;
}
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn rotate(input: &[f32], heads: usize, position: usize) -> Option<Vec<f32>> {
    if heads == 0
        || input.len() != heads.checked_mul(64)?
        || position > i32::MAX as usize
        || input.iter().any(|x| !x.is_finite())
    {
        return None;
    }
    let base = 10000.0f32;
    let scale = 1.0f32 / 40.0;
    let correction = |rotations: f32| {
        64.0 * (4096.0 / (rotations * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln())
    };
    let low = correction(32.0).floor().max(0.0);
    let high = correction(1.0).ceil().min(63.0);
    let magnitude = 0.1f32.mul_add((1.0 / scale).ln(), 1.0);
    let amplitude = (1.0 / magnitude) * magnitude;
    let factor = base.powf(-2.0 / 64.0);
    let mut theta = position as f32;
    let mut out = vec![0.0; input.len()];
    for pair in 0..32 {
        let ramp = 1.0 - (((pair as f32) - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
        let interp = scale * theta;
        let angle = interp.mul_add(1.0 - ramp, theta * ramp);
        // Value argument and repr(C) two-F32 return match the macOS SDK ABI.
        let sc = unsafe { __sincosf_stret(angle) };
        let c = sc.cos * amplitude;
        let s = sc.sin * amplitude;
        for head in 0..heads {
            let at = head * 64 + pair * 2;
            let (x0, x1) = (input[at], input[at + 1]);
            out[at] = x0.mul_add(c, -(x1 * s));
            out[at + 1] = x0.mul_add(s, x1 * c);
        }
        theta *= factor;
    }
    Some(out)
}
