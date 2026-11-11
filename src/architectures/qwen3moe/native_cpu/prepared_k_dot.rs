// SPDX-License-Identifier: MIT
// Decode the reviewed GGML K-quant row layout once per projection call.
// Integer sums are exact; retain the proved block bias/FMA order unchanged.
// Source format and arithmetic: ggml 15fa3c493bfcd040b5f4dcb29e1c998a0846de16.
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use super::native_k_dot::Kind;
use half::f16;

const PREPARED_CAP: usize = 256 * 1024 * 1024;

pub struct Block {
    pub codes: [i16; 256],
    pub scales: [i16; 16],
    pub minima: [i16; 8],
    pub d: f32,
    pub dmin: f32,
}

pub struct Prepared {
    pub kind: Kind,
    blocks: Vec<Block>,
    blocks_per_row: usize,
    columns: usize,
}

pub struct InputBlock {
    pub codes: [i16; 256],
    pub sums32: [i32; 8],
    pub d: f32,
}

pub struct Input {
    blocks: Vec<InputBlock>,
}

fn half(b: &[u8], at: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([b[at], b[at + 1]])).to_f32()
}

fn scale_min(b: &[u8], g: usize) -> (i16, i16) {
    if g < 4 {
        ((b[g] & 63) as i16, (b[g + 4] & 63) as i16)
    } else {
        (
            ((b[g + 4] & 15) | ((b[g - 4] >> 6) << 4)) as i16,
            ((b[g + 4] >> 4) | ((b[g] >> 6) << 4)) as i16,
        )
    }
}

impl Prepared {
    pub fn new(kind: Kind, weight: &[u8], width: usize) -> Result<Self, &'static str> {
        if width == 0 || !width.is_multiple_of(256) || weight.is_empty() {
            return Err("positive K-quant dimensions required");
        }
        let blocks_per_row = width / 256;
        let bytes = kind.block_bytes();
        let row_bytes = blocks_per_row.checked_mul(bytes).ok_or("size overflow")?;
        if !weight.len().is_multiple_of(row_bytes) {
            return Err("K-quant weight extent differs");
        }
        let count = weight.len() / bytes;
        let size = count
            .checked_mul(std::mem::size_of::<Block>())
            .ok_or("size overflow")?;
        if size > PREPARED_CAP {
            return Err("prepared K-quant layout exceeds 256 MiB");
        }
        let mut blocks = Vec::with_capacity(count);
        for w in weight.chunks_exact(bytes) {
            let mut b = Block {
                codes: [0; 256],
                scales: [0; 16],
                minima: [0; 8],
                d: 0.,
                dmin: 0.,
            };
            match kind {
                Kind::Q4 | Kind::Q5 => {
                    b.d = half(w, 0);
                    b.dmin = half(w, 2);
                    let offset = if matches!(kind, Kind::Q4) { 16 } else { 48 };
                    for g in 0..8 {
                        (b.scales[g], b.minima[g]) = scale_min(&w[4..16], g);
                        for j in 0..32 {
                            let packed = w[offset + g / 2 * 32 + j];
                            let mut code = if g % 2 == 0 { packed & 15 } else { packed >> 4 };
                            if matches!(kind, Kind::Q5) {
                                code |= ((w[16 + j] >> g) & 1) << 4;
                            }
                            b.codes[g * 32 + j] = i16::from(code);
                        }
                    }
                }
                Kind::Q6 => {
                    b.d = half(w, 208);
                    for (dst, &src) in b.scales.iter_mut().zip(&w[192..208]) {
                        *dst = i16::from(src as i8);
                    }
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
                            for (g, code) in codes.into_iter().enumerate() {
                                b.codes[h * 128 + g * 32 + j] = i16::from(code) - 32;
                            }
                        }
                    }
                }
            }
            if !b.d.is_finite() || !b.dmin.is_finite() {
                return Err("nonfinite weight scale");
            }
            blocks.push(b);
        }
        Ok(Self {
            kind,
            blocks,
            blocks_per_row,
            columns: weight.len() / row_bytes,
        })
    }

    pub fn bytes(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<Block>()
    }
    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn project(
        &self,
        input: &Input,
        column: usize,
        output: &mut [f32],
    ) -> Result<(), &'static str> {
        let end = column.checked_add(output.len()).ok_or("size overflow")?;
        if output.is_empty() || end > self.columns || input.blocks.len() != self.blocks_per_row {
            return Err("prepared projection extent differs");
        }
        let groups = if matches!(self.kind, Kind::Q6) { 16 } else { 8 };
        let group_width = 256 / groups;
        let weights = &self.blocks[column * self.blocks_per_row..end * self.blocks_per_row];
        for (dst, row) in output
            .iter_mut()
            .zip(weights.chunks_exact(self.blocks_per_row))
        {
            let mut acc = 0.0f32;
            for (w, a) in row.iter().zip(&input.blocks) {
                let mut sum = 0i32;
                for g in 0..groups {
                    let lo = g * group_width;
                    let hi = lo + group_width;
                    // Widen before multiplication. Even Q6's whole-block bound
                    // 256 * 32 * 128 * 128 fits i32; integer regrouping is exact.
                    let dot: i32 = w.codes[lo..hi]
                        .iter()
                        .zip(&a.codes[lo..hi])
                        .map(|(&x, &y)| i32::from(x) * i32::from(y))
                        .sum();
                    sum += dot * i32::from(w.scales[g]);
                }
                if !matches!(self.kind, Kind::Q6) {
                    let bias: i32 = w
                        .minima
                        .iter()
                        .zip(&a.sums32)
                        .map(|(&minimum, &s)| i32::from(minimum) * s)
                        .sum();
                    acc = (-(bias as f32)).mul_add(w.dmin * a.d, acc);
                }
                acc = (sum as f32).mul_add(w.d * a.d, acc);
            }
            if !acc.is_finite() {
                return Err("nonfinite projection output");
            }
            *dst = acc;
        }
        Ok(())
    }
}

impl Input {
    pub fn new(encoded: &[u8], width: usize) -> Result<Self, &'static str> {
        if width == 0
            || !width.is_multiple_of(256)
            || encoded.len() != (width / 256).checked_mul(292).ok_or("size overflow")?
        {
            return Err("Q8_K byte extent differs");
        }
        let mut blocks = Vec::with_capacity(width / 256);
        for a in encoded.chunks_exact(292) {
            let mut b = InputBlock {
                codes: [0; 256],
                sums32: [0; 8],
                d: f32::from_le_bytes(a[..4].try_into().unwrap()),
            };
            if !b.d.is_finite() {
                return Err("nonfinite activation scale");
            }
            for (dst, &src) in b.codes.iter_mut().zip(&a[4..260]) {
                *dst = i16::from(src as i8);
            }
            for g in 0..16 {
                let sum: i32 = b.codes[g * 16..(g + 1) * 16]
                    .iter()
                    .map(|&x| i32::from(x))
                    .sum();
                let at = 260 + g * 2;
                if sum != i32::from(i16::from_le_bytes([a[at], a[at + 1]])) {
                    return Err("Q8_K sum differs from its codes");
                }
                b.sums32[g / 2] += sum;
            }
            blocks.push(b);
        }
        Ok(Self { blocks })
    }
}
