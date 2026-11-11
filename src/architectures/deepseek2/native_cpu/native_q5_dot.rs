// SPDX-License-Identifier: MIT
// Native ARM Q5_0/Q8_0 dot order, ggml-cpu/arch/arm/quants.c,
// llama.cpp 15fa3c493bfcd040b5f4dcb29e1c998a0846de16.
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

/// Q5_0 weights and a native Q8_0 input row. Even block counts only.
pub fn project_q5_0(
    weight: &[u8],
    activation: &[u8],
    width: usize,
    output: &mut [f32],
) -> Result<(), &'static str> {
    if width == 0 || !width.is_multiple_of(64) || output.is_empty() {
        return Err("positive even-block Q5 dimensions required");
    }
    let blocks = width / 32;
    let row_bytes = blocks.checked_mul(22).ok_or("size overflow")?;
    let input_bytes = blocks.checked_mul(34).ok_or("size overflow")?;
    let weight_bytes = row_bytes.checked_mul(output.len()).ok_or("size overflow")?;
    if activation.len() != input_bytes || weight.len() != weight_bytes {
        return Err("Q5 byte extent differs");
    }
    for block in activation.chunks_exact(34).chain(weight.chunks_exact(22)) {
        if !f16::from_bits(u16::from_le_bytes([block[0], block[1]])).is_finite() {
            return Err("nonfinite quant scale");
        }
    }
    for (dst, row) in output.iter_mut().zip(weight.chunks_exact(row_bytes)) {
        let mut sums = [[0.0f32; 4]; 2];
        for block in 0..blocks {
            let w = &row[block * 22..][..22];
            let a = &activation[block * 34..][..34];
            let high = u32::from_le_bytes(w[2..6].try_into().unwrap());
            let scale = f16::from_bits(u16::from_le_bytes([w[0], w[1]])).to_f32()
                * f16::from_bits(u16::from_le_bytes([a[0], a[1]])).to_f32();
            for (lane, acc) in sums[block % 2].iter_mut().enumerate() {
                let mut dot = 0i32;
                for j in lane * 4..lane * 4 + 4 {
                    let lo = i32::from(w[6 + j] & 15) + (((high >> j) & 1) as i32) * 16 - 16;
                    let hi = i32::from(w[6 + j] >> 4) + (((high >> (j + 16)) & 1) as i32) * 16 - 16;
                    dot += lo * i32::from(a[2 + j] as i8) + hi * i32::from(a[18 + j] as i8);
                }
                *acc = (dot as f32).mul_add(scale, *acc);
            }
        }
        let reduce = |v: [f32; 4]| (v[0] + v[1]) + (v[2] + v[3]);
        let value = reduce(sums[0]) + reduce(sums[1]);
        if !value.is_finite() {
            return Err("nonfinite projection output");
        }
        *dst = value;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_extents_and_nonfinite_scales() {
        let mut out = [0.0f32];
        let mut w = [0u8; 44];
        let a = [0u8; 68];
        for width in [0, 1, 32, 65, usize::MAX] {
            assert!(project_q5_0(&w, &a, width, &mut out).is_err());
        }
        assert!(project_q5_0(&w[..43], &a, 64, &mut out).is_err());
        assert!(project_q5_0(&w, &a[..67], 64, &mut out).is_err());
        assert!(project_q5_0(&w, &a, 64, &mut []).is_err());
        assert_eq!(project_q5_0(&w, &a, 64, &mut out), Ok(()));
        assert_eq!(out[0].to_bits(), 0);
        w[..2].copy_from_slice(&f16::NAN.to_bits().to_le_bytes());
        assert!(project_q5_0(&w, &a, 64, &mut out).is_err());
    }
}
