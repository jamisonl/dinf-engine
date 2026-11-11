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

// Native ARM BF16 dot policy from ggml vec.cpp / ggml-impl.h at
// 15fa3c493bfcd040b5f4dcb29e1c998a0846de16: BF16 input rounding,
// F32 products, scalar F64 sum, one final F32 cast. No global fast-math.
// The ARM reference has no BF16 SIMD branch; other hosts may use other orders.

pub fn encode(input: &[f32]) -> Vec<u8> {
    input
        .iter()
        .flat_map(|x| {
            let bits = x.to_bits();
            let word = if bits & 0x7fff_ffff > 0x7f80_0000 {
                ((bits >> 16) | 64) as u16
            } else {
                (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
            };
            word.to_le_bytes()
        })
        .collect()
}

fn value(b: &[u8]) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)
}
fn sequential(row: &[u8], input: &[u8]) -> f32 {
    let mut sum = 0.0f64;
    for (w, x) in row.chunks_exact(2).zip(input.chunks_exact(2)) {
        let product = value(w) * value(x);
        sum += f64::from(product);
    }
    sum as f32
}
#[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
fn dot(row: &[u8], input: &[u8], _decoded: &[f32], _width: usize) -> f32 {
    sequential(row, input)
}
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
fn dot(row: &[u8], input: &[u8], decoded: &[f32], width: usize) -> f32 {
    use std::arch::aarch64::*;
    // SAFETY: project validates exactly width*2 bytes for both spans. Each
    // iteration loads sixteen weight bytes and eight checked decoded floats. vld1q_u8
    // supports byte alignment. ARM64 always has these NEON instructions.
    unsafe {
        let mut s0 = vdupq_n_f64(0.0);
        let mut s1 = s0;
        let mut s2 = s0;
        let mut s3 = s0;
        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = a0;
        let blocks = width / 8;
        for block in 0..blocks {
            let w = vreinterpretq_u16_u8(vld1q_u8(row.as_ptr().add(block * 16)));

            let wl = vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(vget_low_u16(w))));
            let wh = vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(vget_high_u16(w))));
            let xl = vld1q_f32(decoded.as_ptr().add(block * 8));
            let xh = vld1q_f32(decoded.as_ptr().add(block * 8 + 4));
            // Keep multiplication in F32 before widening; never F64 FMA.
            let p0 = vmulq_f32(wl, xl);
            let p1 = vmulq_f32(wh, xh);
            s0 = vaddq_f64(s0, vcvt_f64_f32(vget_low_f32(p0)));
            s1 = vaddq_f64(s1, vcvt_f64_f32(vget_high_f32(p0)));
            s2 = vaddq_f64(s2, vcvt_f64_f32(vget_low_f32(p1)));
            s3 = vaddq_f64(s3, vcvt_f64_f32(vget_high_f32(p1)));
            a0 = vaddq_f32(a0, vabsq_f32(p0));
            a1 = vaddq_f32(a1, vabsq_f32(p1));
        }
        let mut absolute = f64::from(vaddvq_f32(vaddq_f32(a0, a1)));
        let mut tail = 0.0f64;
        for at in blocks * 8..width {
            let product = value(&row[at * 2..at * 2 + 2]) * value(&input[at * 2..at * 2 + 2]);
            tail += f64::from(product);
            absolute += f64::from(product.abs());
        }
        let sum = vaddvq_f64(s0) + vaddvq_f64(s1) + vaddvq_f64(s2) + vaddvq_f64(s3) + tail;
        if absolute == 0.0 {
            return 0.0;
        }
        // At width <= 65536, twice the finite positive F32 absolute sum
        // bounds the exact absolute sum (its worst relative error is < 1%).
        // Each F64 order has error <= 2*n*EPSILON*that upper bound. Use
        // their sum as a conservative distance bound. Admit only if BOTH
        // interval endpoints round to the same F32; otherwise retain native
        // scalar order. This also handles cancellation and half-way results.
        if width <= 65536 && absolute.is_finite() && sum.is_finite() {
            let error = (width as f64) * f64::EPSILON * absolute * 8.0;
            let low = (sum - error) as f32;
            let high = (sum + error) as f32;
            if low.to_bits() == high.to_bits() {
                return low;
            }
        }
        sequential(row, input)
    }
}
pub fn project(
    weight: &[u8],
    input: &[u8],
    width: usize,
    output: &mut [f32],
) -> Result<(), &'static str> {
    let row_bytes = width.checked_mul(2).ok_or("BF16 size overflow")?;
    if width == 0
        || input.len() != row_bytes
        || weight.len()
            != output
                .len()
                .checked_mul(row_bytes)
                .ok_or("BF16 size overflow")?
    {
        return Err("BF16 shape mismatch");
    }
    let decoded: Vec<f32> = input.chunks_exact(2).map(value).collect();
    for (row, out) in weight.chunks_exact(row_bytes).zip(output) {
        *out = dot(row, input, &decoded, width);
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bf16_rounding_ties_range_and_specials() {
        let values = [
            f32::from_bits(0x3f808000),
            f32::from_bits(0x3f818000),
            -f32::from_bits(0x3f808000),
            -f32::from_bits(0x3f818000),
            0.0,
            -0.0,
            f32::from_bits(0x0001_0000),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7f80_0001),
        ];
        let words: Vec<_> = encode(&values)
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        assert_eq!(
            words,
            [
                0x3f80, 0x3f82, 0xbf80, 0xbf82, 0, 0x8000, 1, 0x7f80, 0xff80, 0x7fc0
            ]
        );
    }
    #[test]
    fn bf16_double_sum_retains_small_cancellation_term() {
        let w = encode(&[16_777_216.0, 1.0, -16_777_216.0]);
        let x = encode(&[1.0, 1.0, 1.0]);
        let mut out = [0.0];
        project(&w, &x, 3, &mut out).unwrap();
        assert_eq!(out, [1.0]);
        // A sequential F32 sum loses this term.
        assert_eq!((16_777_216.0f32 + 1.0) - 16_777_216.0, 0.0);
        assert!(project(&w, &x, 2, &mut out).is_err());
        assert!(project(&[], &[], 0, &mut out).is_err());
        project(&encode(&[f32::INFINITY]), &encode(&[1.0]), 1, &mut out).unwrap();
        assert_eq!(out[0], f32::INFINITY);
        project(&encode(&[f32::INFINITY]), &encode(&[0.0]), 1, &mut out).unwrap();
        assert!(out[0].is_nan());
    }
}

#[cfg(test)]
mod lane_tests {
    use super::*;
    #[test]
    fn lanes_match_preserved_native_order_for_wide_exponents_and_cancellation() {
        for width in [1, 3, 8, 31, 768, 2048] {
            for salt in 0..16u32 {
                let mut w = Vec::new();
                let mut x = Vec::new();
                for i in 0..width {
                    let a = (i as u32).wrapping_mul(1664525).wrapping_add(salt * 131);
                    let e = if salt < 8 {
                        110 + (a % 25)
                    } else {
                        1 + (a % 250)
                    };
                    w.push(f32::from_bits(
                        ((a & 1) << 31) | (e << 23) | ((a >> 2) & 0x7f) << 16,
                    ));
                    x.push(if i % 3 == 0 { -1.0 } else { 1.0 });
                }
                let w = encode(&w);
                let x = encode(&x);
                let expected = sequential(&w, &x);
                let mut y = [0.0];
                project(&w, &x, width, &mut y).unwrap();
                assert_eq!(y[0].to_bits(), expected.to_bits());
            }
        }
        for w in [
            vec![2f32.powi(100), 1.0, -2f32.powi(100)],
            vec![f32::INFINITY, 0.0, f32::NEG_INFINITY],
            vec![
                f32::from_bits(0x0001_0000),
                -f32::from_bits(0x0001_0000),
                0.0,
            ],
        ] {
            let w = encode(&w);
            let x = encode(&[1.0; 3]);
            let expected = sequential(&w, &x);
            let mut y = [0.0];
            project(&w, &x, 3, &mut y).unwrap();
            assert_eq!(y[0].to_bits(), expected.to_bits());
        }
    }
}
