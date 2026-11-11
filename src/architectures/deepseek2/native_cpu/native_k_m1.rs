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

// Only integer products change here; the caller retains every F32 operation.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot32(
    packed: &[u8],
    input: &[u8],
    high_bits: Option<&[u8]>,
    high_nibble: bool,
    group: usize,
) -> i32 {
    use std::arch::aarch64::*;
    assert_eq!(packed.len(), 32);
    assert_eq!(input.len(), 32);
    assert!(group < 8);
    if let Some(high) = high_bits {
        assert_eq!(high.len(), 32);
    }
    // Every load reads 16 bytes within an extent checked above. Q5 codes are
    // at most31, so reinterpretation as signed bytes preserves their values.
    unsafe {
        let nibble = vdupq_n_u8(15);
        let mut acc = vdupq_n_s32(0);
        for offset in [0, 16] {
            let p = vld1q_u8(packed.as_ptr().add(offset));
            let mut codes = if high_nibble {
                vshrq_n_u8::<4>(p)
            } else {
                vandq_u8(p, nibble)
            };
            if let Some(high) = high_bits {
                let high = vld1q_u8(high.as_ptr().add(offset));
                let shifted = vshlq_u8(high, vdupq_n_s8(-(group as i8)));
                let bit = vshlq_n_u8::<4>(vandq_u8(shifted, vdupq_n_u8(1)));
                codes = vorrq_u8(codes, bit);
            }
            let activation = vld1q_s8(input.as_ptr().add(offset).cast());
            let codes = vreinterpretq_s8_u8(codes);
            let lo = vmull_s8(vget_low_s8(codes), vget_low_s8(activation));
            let hi = vmull_s8(vget_high_s8(codes), vget_high_s8(activation));
            acc = vpadalq_s16(acc, lo);
            acc = vpadalq_s16(acc, hi);
        }
        vaddvq_s32(acc)
    }
}

// Q6 products use signed codes in[-32,31] and one signed scale per16codes.
// Integer sums remain exact; the caller keeps the original F32 operations.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn q6_block_dot(weight: &[u8], input: &[u8]) -> i32 {
    use std::arch::aarch64::*;
    assert_eq!(weight.len(), 210);
    assert_eq!(input.len(), 256);
    let mut sum = 0i32;
    // All loads read16bytes inside the checked weight and input extents.
    // A product fits i16; each pairwise sum widens to i32 before addition.
    unsafe {
        for half in 0..2 {
            for group in 0..4 {
                for segment in 0..2 {
                    let offset = segment * 16;
                    let low = vld1q_u8(weight.as_ptr().add(half * 64 + (group % 2) * 32 + offset));
                    let high = vld1q_u8(weight.as_ptr().add(128 + half * 32 + offset));
                    let nibble = if group < 2 {
                        vandq_u8(low, vdupq_n_u8(15))
                    } else {
                        vshrq_n_u8::<4>(low)
                    };
                    let high = vandq_u8(
                        vshlq_u8(high, vdupq_n_s8(-((group * 2) as i8))),
                        vdupq_n_u8(3),
                    );
                    let codes = vreinterpretq_s8_u8(vsubq_u8(
                        vorrq_u8(nibble, vshlq_n_u8::<4>(high)),
                        vdupq_n_u8(32),
                    ));
                    let activation =
                        vld1q_s8(input.as_ptr().add(half * 128 + group * 32 + offset).cast());
                    let lo = vmull_s8(vget_low_s8(codes), vget_low_s8(activation));
                    let hi = vmull_s8(vget_high_s8(codes), vget_high_s8(activation));
                    let dot = vaddvq_s32(vpadalq_s16(vpadalq_s16(vdupq_n_s32(0), lo), hi));
                    sum += dot * i32::from(weight[192 + half * 8 + group * 2 + segment] as i8);
                }
            }
        }
    }
    sum
}

#[cfg(all(test, target_arch = "aarch64"))]
mod integer_tests {
    use super::*;

    #[test]
    fn q6_codes_scales_and_extremes_match_scalar_products() {
        assert!(std::arch::is_aarch64_feature_detected!("neon"));
        let scalar = |w: &[u8; 210], a: &[u8; 256]| {
            let mut sum = 0i32;
            for h in 0..2 {
                for j in 0..32 {
                    let lo0 = w[h * 64 + j];
                    let lo1 = w[h * 64 + j + 32];
                    let hi = w[128 + h * 32 + j];
                    let codes = [
                        (lo0 & 15) | ((hi & 3) << 4),
                        (lo1 & 15) | (((hi >> 2) & 3) << 4),
                        (lo0 >> 4) | (((hi >> 4) & 3) << 4),
                        (lo1 >> 4) | (((hi >> 6) & 3) << 4),
                    ];
                    for (g, code) in codes.into_iter().enumerate() {
                        sum += (i32::from(code) - 32)
                            * i32::from(w[192 + h * 8 + g * 2 + j / 16] as i8)
                            * i32::from(a[h * 128 + g * 32 + j] as i8);
                    }
                }
            }
            sum
        };
        for seed in 0u16..256 {
            let mut w =
                std::array::from_fn::<_, 210, _>(|i| (seed as u8).wrapping_add((i * 73) as u8));
            for scale in [-128i8, -127, -1, 0, 1, 63, 127] {
                for (i, s) in w[192..208].iter_mut().enumerate() {
                    *s = scale.wrapping_add((i * 13) as i8) as u8;
                }
                for activation in [-128i8, -127, -1, 0, 1, 63, 127] {
                    let a = std::array::from_fn::<_, 256, _>(|i| {
                        activation.wrapping_add((i * 17) as i8) as u8
                    });
                    assert_eq!(unsafe { q6_block_dot(&w, &a) }, scalar(&w, &a));
                }
            }
        }
        let mut w = [0u8; 210];
        w[192..208].fill(128);
        let a = [128u8; 256];
        assert_eq!(unsafe { q6_block_dot(&w, &a) }, -134217728);
    }

    #[test]
    fn every_packed_byte_and_group_matches_signed_scalar_products() {
        assert!(std::arch::is_aarch64_feature_detected!("neon"));
        for seed in 0u16..256 {
            let packed =
                std::array::from_fn::<_, 32, _>(|i| (seed as u8).wrapping_add(i as u8 * 7));
            let high = std::array::from_fn::<_, 32, _>(|i| {
                (seed as u8).wrapping_mul(73).wrapping_add((i * 13) as u8)
            });
            for group in 0..8 {
                for activation in [-128i8, -127, -1, 0, 1, 63, 127] {
                    let input = std::array::from_fn::<_, 32, _>(|i| {
                        activation.wrapping_add((i * 17) as i8) as u8
                    });
                    for q5 in [false, true] {
                        let expected: i32 = (0..32)
                            .map(|i| {
                                let nibble = if group % 2 == 0 {
                                    packed[i] & 15
                                } else {
                                    packed[i] >> 4
                                };
                                let bit = if q5 { ((high[i] >> group) & 1) << 4 } else { 0 };
                                i32::from(nibble | bit) * i32::from(input[i] as i8)
                            })
                            .sum();
                        let actual = unsafe {
                            dot32(&packed, &input, q5.then_some(&high), group % 2 == 1, group)
                        };
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
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
    #[cfg(target_arch = "aarch64")]
    let accelerated = std::arch::is_aarch64_feature_detected!("neon");
    #[cfg(not(target_arch = "aarch64"))]
    let accelerated = false;
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
                        if accelerated {
                            #[cfg(target_arch = "aarch64")]
                            {
                                let packed = &w[quant_offset + group / 2 * 32..][..32];
                                let input = &a[4 + group * 32..][..32];
                                let high = matches!(kind, Kind::Q5).then_some(&w[16..48]);
                                // The feature test and checked 32-byte slices
                                // satisfy dot32's target and memory contract.
                                dot = unsafe { dot32(packed, input, high, group % 2 == 1, group) };
                            }
                            let offset = 260 + group * 4;
                            input_sum = i32::from(i16::from_le_bytes([a[offset], a[offset + 1]]))
                                + i32::from(i16::from_le_bytes([a[offset + 2], a[offset + 3]]));
                        } else {
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
                    if accelerated {
                        #[cfg(target_arch = "aarch64")]
                        {
                            // NEON and exact byte extents were checked above.
                            sum = unsafe { q6_block_dot(w, &a[4..260]) };
                        }
                    } else {
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
                                    let scale =
                                        i32::from(w[192 + h * 8 + group * 2 + j / 16] as i8);
                                    let input = i32::from(a[4 + h * 128 + group * 32 + j] as i8);
                                    sum += (i32::from(code) - 32) * scale * input;
                                }
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
