// Copyright Naoki Shibata and contributors 2010 - 2024.
// Copyright ARM Ltd. 2010 - 2024.
// SPDX-License-Identifier: BSL-1.0
// Adapted from SLEEF 5a1d179df9cf652951b59010a2d2075372d67f68:
// sleefsimdsp.c logkf/expkf/xpowf and df.h's FMA two-float helpers.
// See the shipped source record and Boost Software License notice.
//
// Only the finite positive domain needed by RoPE is admitted. Every ordinary
// operation rounds to F32; mul_add marks exactly the upstream FMA operations.
// The scalar implementation follows one ARM vector lane and has no Torch link.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

#[derive(Clone, Copy)]
struct Pair(f32, f32);

impl Pair {
    fn normalize(self) -> Self {
        let s = self.0 + self.1;
        Self(s, (self.0 - s) + self.1)
    }

    fn scale(self, value: f32) -> Self {
        Self(self.0 * value, self.1 * value)
    }

    // Upstream dfadd: the caller supplies |self| >= |other|.
    fn add(self, other: Self) -> Self {
        let s = self.0 + other.0;
        Self(s, (((self.0 - s) + other.0) + self.1) + other.1)
    }

    fn add2(self, other: Self) -> Self {
        let s = self.0 + other.0;
        let v = s - self.0;
        let t = (self.0 - (s - v)) + (other.0 - v);
        Self(s, t + (self.1 + other.1))
    }

    fn add_scalar(self, other: f32) -> Self {
        let s = self.0 + other;
        let v = s - self.0;
        let t = (self.0 - (s - v)) + (other - v);
        Self(s, t + self.1)
    }

    fn scalar_add(first: f32, other: Self) -> Self {
        let s = first + other.0;
        Self(s, ((first - s) + other.0) + other.1)
    }

    fn two_scalars(first: f32, second: f32) -> Self {
        let s = first + second;
        let v = s - first;
        Self(s, (first - (s - v)) + (second - v))
    }

    fn div(self, other: Self) -> Self {
        let t = 1.0 / other.0;
        let s = self.0 * t;
        let u = t.mul_add(self.0, -s);
        let v = (-other.1).mul_add(t, (-other.0).mul_add(t, 1.0));
        Self(s, s.mul_add(v, self.1.mul_add(t, u)))
    }

    fn square(self) -> Self {
        let s = self.0 * self.0;
        Self(
            s,
            (self.0 + self.0).mul_add(self.1, self.0.mul_add(self.0, -s)),
        )
    }

    fn mul(self, other: Self) -> Self {
        let s = self.0 * other.0;
        Self(
            s,
            self.0.mul_add(
                other.1,
                self.1.mul_add(other.0, self.0.mul_add(other.0, -s)),
            ),
        )
    }

    fn mul_scalar(self, other: f32) -> Self {
        let s = self.0 * other;
        Self(s, self.1.mul_add(other, self.0.mul_add(other, -s)))
    }
}

fn log_positive(value: f32) -> Pair {
    let exponent = (((value * (1.0_f32 / 0.75)).to_bits() >> 23) & 0xff) as i32 - 127;
    // All admitted values and this scaled mantissa are normal and positive.
    let mantissa =
        f32::from_bits((i64::from(value.to_bits()) - (i64::from(exponent) << 23)) as u32);
    let x = Pair::two_scalars(-1.0, mantissa).div(Pair::two_scalars(1.0, mantissa));
    let x2 = x.square();
    let t = 0.240320354700088500976562_f32
        .mul_add(x2.0, 0.285112679004669189453125)
        .mul_add(x2.0, 0.400007992982864379882812);
    let c = Pair(0.66666662693023681640625, 3.69183861259614332084311e-09);
    let s = Pair(0.69314718246459960938, -1.904654323148236017e-09)
        .mul_scalar(exponent as f32)
        .add(x.scale(2.0));
    s.add(x2.mul(x).mul(x2.mul_scalar(t).add2(c)))
}

fn exp_small(value: Pair) -> f32 {
    let u = (value.0 + value.1)
        * 1.442695040888963407359924681001892137426645954152985934135449406931_f32;
    let q = u.round_ties_even() as i32;
    let s = value
        .add_scalar(q as f32 * -0.693145751953125_f32)
        .add_scalar(q as f32 * -1.428606765330187045e-06_f32)
        .normalize();
    let u = 0.00136324646882712841033936_f32
        .mul_add(s.0, 0.00836596917361021041870117)
        .mul_add(s.0, 0.0416710823774337768554688)
        .mul_add(s.0, 0.166665524244308471679688)
        .mul_add(s.0, 0.499999850988388061523438);
    let t = Pair::scalar_add(1.0, s.add(s.square().mul_scalar(u)));
    // The guarded input domain keeps q in [0, 16]; upstream's general ldexp
    // reduces to one exact normal power-of-two multiplication in this range.
    debug_assert!((0..=16).contains(&q));
    (t.0 + t.1) * f32::from_bits(((q + 127) as u32) << 23)
}

pub(super) fn positive_pow(base: f32, exponent: f32) -> Option<f32> {
    if !(1.0..=65536.0).contains(&base) || !(0.0..=1.0).contains(&exponent) {
        return None;
    }
    if exponent == 0.0 || base == 1.0 {
        return Some(1.0);
    }
    Some(exp_small(log_positive(base).mul_scalar(exponent)))
}

// Torch processes two four-lane vectors, then scalar leftovers, per axis.
// Preserve that runtime arithmetic schedule independently of frequency values.
pub(super) fn frequencies(base: f32, dim: usize) -> Option<Vec<f32>> {
    if dim == 0 || !dim.is_multiple_of(2) || dim > 16_777_216 || !(1.0..=65536.0).contains(&base) {
        return None;
    }
    let count = dim / 2;
    let vector_end = count / 8 * 8;
    (0..count)
        .map(|i| {
            let exponent = (2 * i) as f32 / dim as f32;
            let power = if i < vector_end {
                positive_pow(base, exponent)?
            } else {
                base.powf(exponent)
            };
            Some(1.0 / power)
        })
        .collect()
}

// A public C-header wrapper owns the platform ABI. Only macOS ARM64 builds
// compile this module and its shim; angle and output pointers stay local.
unsafe extern "C" {
    fn dinf_rope_sincosf(angle: f32, sine: *mut f32, cosine: *mut f32);
}

pub(super) fn sin_cos(angle: f32) -> (f32, f32) {
    let (mut sine, mut cosine) = (0.0, 0.0);
    // SAFETY: both pointers refer to writable, distinct local F32 values.
    unsafe { dinf_rope_sincosf(angle, &mut sine, &mut cosine) };
    (sine, cosine)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_power_domain_and_f64_bound() {
        for (base, exponent) in [
            (0.0, 0.5),
            (-1.0, 0.5),
            (65537.0, 0.5),
            (f32::NAN, 0.5),
            (f32::INFINITY, 0.5),
            (2.0, -0.1),
            (2.0, 1.1),
            (2.0, f32::NAN),
        ] {
            assert!(positive_pow(base, exponent).is_none());
        }
        for base in [
            1.0_f32, 1.5, 2.0, 3.0, 10.0, 256.0, 1000.0, 10000.0, 65536.0,
        ] {
            for i in 0..=1024 {
                let exponent = i as f32 / 1024.0;
                let actual = positive_pow(base, exponent).unwrap();
                let exact = f64::from(base).powf(f64::from(exponent));
                let rounded = exact as f32;
                let ulp = f64::from(f32::from_bits(rounded.to_bits() + 1)) - f64::from(rounded);
                assert!((f64::from(actual) - exact).abs() <= 1.00001 * ulp);
                assert!(actual.is_finite() && actual > 0.0);
            }
        }
    }

    #[test]
    fn frequency_tails_and_bounds() {
        for dim in [0, 1, 3, 16_777_218, usize::MAX] {
            assert!(frequencies(10000.0, dim).is_none());
        }
        for dim in [2, 14, 16, 18, 56, 128, 258] {
            let values = frequencies(10000.0, dim).unwrap();
            assert_eq!(values.len(), dim / 2);
            let start = values.len() / 8 * 8;
            for (i, value) in values.iter().enumerate().skip(start) {
                let expected = 1.0 / 10000.0_f32.powf((2 * i) as f32 / dim as f32);
                assert_eq!(value.to_bits(), expected.to_bits());
            }
        }
    }
}
