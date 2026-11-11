// SPDX-License-Identifier: MIT
// ARM CPU Q8_0 activation encoding from ggml quantize_row_q8_0, commit
// 15fa3c493bfcd040b5f4dcb29e1c998a0846de16.
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

/// Encode complete Q8_0 blocks with the native ARM nearest-even rounding.
pub fn quantize(input: &[f32]) -> Result<Vec<u8>, &'static str> {
    if input.is_empty() || !input.len().is_multiple_of(32) {
        return Err("positive complete Q8_0 blocks required");
    }
    if input.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite Q8_0 input");
    }
    let size = (input.len() / 32)
        .checked_mul(34)
        .ok_or("Q8_0 size overflow")?;
    let mut result = vec![0u8; size];
    for (row, block) in input.chunks_exact(32).zip(result.chunks_exact_mut(34)) {
        let maximum = row.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = maximum / 127.0;
        let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        let stored_scale = f16::from_f32(scale);
        if !inverse.is_finite() || !stored_scale.is_finite() {
            return Err("Q8_0 scale overflow");
        }
        block[..2].copy_from_slice(&stored_scale.to_bits().to_le_bytes());
        for (j, &value) in row.iter().enumerate() {
            block[2 + j] = (value * inverse).round_ties_even() as i8 as u8;
        }
    }
    Ok(result)
}
