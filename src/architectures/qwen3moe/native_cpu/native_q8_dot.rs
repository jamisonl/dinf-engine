// SPDX-License-Identifier: MIT
// Row-major form of the arithmetic in ggml_gemv_q8_0_4x8_q8_0 and
// ggml_gemm_q8_0_4x8_q8_0 from ggml/llama.cpp
// 15fa3c493bfcd040b5f4dcb29e1c998a0846de16, ggml-cpu/arch/arm/repack.cpp.
// It sums all 32 integer products before each F32 scale/FMA. This preserves
// the pinned native arithmetic order without native repacking or private
// Candle fields. No quantization or source tensor mutation occurs here.
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

use half::f16;

/// One already-quantized Q8_0 activation row times row-major Q8_0 weights.
/// Each block has a little-endian F16 scale followed by 32 signed bytes.
pub fn project_q8_0(
    weight: &[u8],
    activation: &[u8],
    width: usize,
    output: &mut [f32],
) -> Result<(), &'static str> {
    if width == 0 || !width.is_multiple_of(32) || output.is_empty() {
        return Err("positive Q8 dimensions required");
    }
    let row_bytes = (width / 32).checked_mul(34).ok_or("size overflow")?;
    let weight_bytes = row_bytes.checked_mul(output.len()).ok_or("size overflow")?;
    if activation.len() != row_bytes || weight.len() != weight_bytes {
        return Err("Q8 byte extent differs");
    }
    for block in activation.chunks_exact(34).chain(weight.chunks_exact(34)) {
        if !f16::from_bits(u16::from_le_bytes([block[0], block[1]])).is_finite() {
            return Err("nonfinite Q8 scale");
        }
    }
    for (dst, row) in output.iter_mut().zip(weight.chunks_exact(row_bytes)) {
        let mut acc = 0.0f32;
        for (w, a) in row.chunks_exact(34).zip(activation.chunks_exact(34)) {
            // The largest magnitude is 32 * 128 * 128, exact in i32/F32.
            let mut sum = 0i32;
            for (&x, &y) in w[2..].iter().zip(&a[2..]) {
                sum += i32::from(x as i8) * i32::from(y as i8);
            }
            let wd = f16::from_bits(u16::from_le_bytes([w[0], w[1]])).to_f32();
            let ad = f16::from_bits(u16::from_le_bytes([a[0], a[1]])).to_f32();
            let scale = wd * ad;
            acc = (sum as f32).mul_add(scale, acc);
        }
        if !acc.is_finite() {
            return Err("nonfinite projection output");
        }
        *dst = acc;
    }
    Ok(())
}
