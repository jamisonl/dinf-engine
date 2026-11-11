// SPDX-License-Identifier: MIT
// Row-major form of the integer and FMA order in ggml_gemv_q{4,5,6}_K_8x8_q8_K
// from ggml/llama.cpp 15fa3c493bfcd040b5f4dcb29e1c998a0846de16.
// Keep this order for every row, including decode, to preserve chunk invariance.
// This helper does not quantize inputs, select a backend, or mutate weights.
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

#[derive(Clone, Copy, Debug)]
pub enum Kind {
    Q4,
    Q5,
    Q6,
}

impl Kind {
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Q4 => 144,
            Self::Q5 => 176,
            Self::Q6 => 210,
        }
    }
}

fn half(bytes: &[u8], offset: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
}

fn scale_min(bytes: &[u8], group: usize) -> (i32, i32) {
    if group < 4 {
        ((bytes[group] & 63) as i32, (bytes[group + 4] & 63) as i32)
    } else {
        (
            ((bytes[group + 4] & 15) | ((bytes[group - 4] >> 6) << 4)) as i32,
            ((bytes[group + 4] >> 4) | ((bytes[group] >> 6) << 4)) as i32,
        )
    }
}

/// Multiply one encoded Q8_K row by row-major Q4_K, Q5_K, or Q6_K weights.
/// Q4 uses per-32-code FMA and block bias; Q5 applies bias per 64 codes;
/// Q6 uses the whole-block integer sum and separate F32 multiply/add.
pub fn project(
    kind: Kind,
    weight: &[u8],
    activation: &[u8],
    width: usize,
    output: &mut [f32],
) -> Result<(), &'static str> {
    if width == 0 || !width.is_multiple_of(256) || output.is_empty() {
        return Err("positive K-quant dimensions required");
    }
    let blocks = width / 256;
    let bytes = kind.block_bytes();
    let row_bytes = blocks.checked_mul(bytes).ok_or("size overflow")?;
    let expected_weight = row_bytes.checked_mul(output.len()).ok_or("size overflow")?;
    if activation.len() != blocks.checked_mul(292).ok_or("size overflow")?
        || weight.len() != expected_weight
    {
        return Err("K-quant byte extent differs");
    }
    for a in activation.chunks_exact(292) {
        if !f32::from_le_bytes(a[..4].try_into().unwrap()).is_finite() {
            return Err("nonfinite activation scale");
        }
        for group in 0..16 {
            let sum: i32 = a[4 + group * 16..4 + (group + 1) * 16]
                .iter()
                .map(|&x| i32::from(x as i8))
                .sum();
            let offset = 260 + group * 2;
            if sum != i32::from(i16::from_le_bytes([a[offset], a[offset + 1]])) {
                return Err("Q8_K sum differs from its codes");
            }
        }
    }
    for w in weight.chunks_exact(bytes) {
        let finite = match kind {
            Kind::Q6 => half(w, 208).is_finite(),
            _ => half(w, 0).is_finite() && half(w, 2).is_finite(),
        };
        if !finite {
            return Err("nonfinite weight scale");
        }
    }
    for (dst, row) in output.iter_mut().zip(weight.chunks_exact(row_bytes)) {
        let mut acc = 0.0f32;
        for (w, a) in row.chunks_exact(bytes).zip(activation.chunks_exact(292)) {
            let ad = f32::from_le_bytes(a[..4].try_into().unwrap());
            let mut sum = 0i32;
            let wd = match kind {
                Kind::Q4 | Kind::Q5 => {
                    let mut bias = 0i32;
                    let quant_offset = if matches!(kind, Kind::Q4) { 16 } else { 48 };
                    for group in 0..8 {
                        let (scale, minimum) = scale_min(&w[4..16], group);
                        let mut dot = 0i32;
                        let mut input_sum = 0i32;
                        for j in 0..32 {
                            let packed = w[quant_offset + group / 2 * 32 + j];
                            let mut code = if group % 2 == 0 {
                                packed & 15
                            } else {
                                packed >> 4
                            };
                            if matches!(kind, Kind::Q5) {
                                code |= ((w[16 + j] >> group) & 1) << 4;
                            }
                            let input = i32::from(a[4 + group * 32 + j] as i8);
                            dot += i32::from(code) * input;
                            input_sum += input;
                        }
                        acc = ((scale * dot) as f32).mul_add(half(w, 0) * ad, acc);
                        bias += minimum * input_sum;
                        if matches!(kind, Kind::Q5) && group % 2 == 1 {
                            acc = (-(bias as f32)).mul_add(half(w, 2) * ad, acc);
                            bias = 0;
                        }
                    }
                    if matches!(kind, Kind::Q4) {
                        acc = (-(bias as f32)).mul_add(half(w, 2) * ad, acc);
                    }
                    half(w, 0)
                }
                Kind::Q6 => {
                    for h in 0..2 {
                        for j in 0..32 {
                            let low0 = w[h * 64 + j];
                            let low1 = w[h * 64 + j + 32];
                            let high = w[128 + h * 32 + j];
                            let codes = [
                                (low0 & 15) | ((high & 3) << 4),
                                (low1 & 15) | (((high >> 2) & 3) << 4),
                                (low0 >> 4) | (((high >> 4) & 3) << 4),
                                (low1 >> 4) | (((high >> 6) & 3) << 4),
                            ];
                            for (group, code) in codes.into_iter().enumerate() {
                                let scale = i32::from(w[192 + h * 8 + group * 2 + j / 16] as i8);
                                let input = i32::from(a[4 + h * 128 + group * 32 + j] as i8);
                                sum += (i32::from(code) - 32) * scale * input;
                            }
                        }
                    }
                    half(w, 208)
                }
            };
            // Worst-case signed integer sum fits i32, including Q6_K's
            // -128 scale, -32 weight code and -128 activation code.
            if matches!(kind, Kind::Q6) {
                acc += (sum as f32) * (wd * ad);
            }
        }
        if !acc.is_finite() {
            return Err("nonfinite projection output");
        }
        *dst = acc;
    }
    Ok(())
}
