// SPDX-License-Identifier: MIT
// Q8_K activation encoding from ggml quantize_row_q8_K_ref and
// ggml_quantize_mat_q8_K_4x8_generic, commit
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

/// Encode complete Q8_K blocks. Ties use the first maximum by input order
/// and nearest-even integer rounding, as in the native CPU implementation.
pub fn quantize(input: &[f32]) -> Result<Vec<u8>, &'static str> {
    if input.is_empty() || !input.len().is_multiple_of(256) {
        return Err("positive complete Q8_K blocks required");
    }
    if input.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite Q8_K input");
    }
    let size = (input.len() / 256)
        .checked_mul(292)
        .ok_or("Q8_K size overflow")?;
    let mut result = vec![0u8; size];
    for (row, block) in input.chunks_exact(256).zip(result.chunks_exact_mut(292)) {
        let mut maximum = 0.0f32;
        let mut absolute = 0.0f32;
        for &value in row {
            if value.abs() > absolute {
                maximum = value;
                absolute = value.abs();
            }
        }
        if absolute == 0.0 {
            continue;
        }
        let inverse = -127.0f32 / maximum;
        if !inverse.is_finite() {
            return Err("Q8_K inverse scale overflow");
        }
        let scale = 1.0f32 / inverse;
        block[..4].copy_from_slice(&scale.to_le_bytes());
        for (group, values) in row.chunks_exact(16).enumerate() {
            let mut sum = 0i16;
            for (j, &value) in values.iter().enumerate() {
                let code = (value * inverse).round_ties_even().min(127.0) as i8;
                block[4 + group * 16 + j] = code as u8;
                sum += i16::from(code);
            }
            block[260 + group * 2..262 + group * 2].copy_from_slice(&sum.to_le_bytes());
        }
    }
    Ok(result)
}
