//! Native CPU top-six route normalization on macOS ARM64.
//!
//! The pinned ggml CPU build uses Accelerate vDSP_sve for SUM_ROWS, followed
//! by clamp to 2^-14 and scalar F32 division. Call that public routine rather
//! than guess its sum order. The caller must retain the actual CPU Q8/native
//! cache/F32-router admission; this helper accepts exactly six finite weights.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod platform {
    // The pinned SDK declares vDSP_Stride as ptrdiff_t-sized signed long and
    // vDSP_Length as unsigned long on LP64 macOS; both are 64 bits here.
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn vDSP_sve(input: *const f32, stride: isize, output: *mut f32, count: usize);
    }

    pub(super) fn sum(input: &[f32; 6]) -> f32 {
        let mut result = 0.0;
        // SAFETY: input has exactly six initialized, aligned F32 values; stride
        // is one and count is six. The distinct live result holds one writable
        // F32. The C function keeps neither pointer after returning.
        unsafe { vDSP_sve(input.as_ptr(), 1, &mut result, 6) };
        result
    }
}

pub(super) struct Normalized {
    pub(super) sum: f32,
    pub(super) weights: [f32; 6],
}

pub(super) fn normalize(input: &[f32]) -> Result<Normalized, &'static str> {
    let input: &[f32; 6] = input.try_into().map_err(|_| "native route count")?;
    if !input.iter().all(|x| x.is_finite() && *x >= 0.0) {
        return Err("native route finite nonnegative weights");
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        Err("native route platform")
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let sum = platform::sum(input);
        if !sum.is_finite() || sum < 0.0 {
            return Err("native route nonfinite sum");
        }
        let denominator = sum.max(f32::from_bits(0x3880_0000)); // 2^-14
        let weights = std::array::from_fn(|i| input[i] / denominator);
        if !weights.iter().all(|x| x.is_finite() && *x >= 0.0) {
            return Err("native route nonfinite normalized weight");
        }
        Ok(Normalized { sum, weights })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_wrong_count_and_nonfinite_or_negative_weights() {
        assert!(normalize(&[1.0; 5]).is_err());
        assert!(normalize(&[1.0; 7]).is_err());
        for bad in [-1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let mut values = [1.0; 6];
            values[3] = bad;
            assert!(normalize(&values).is_err());
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn zero_uniform_and_overflow_sum_guard() {
        let zero = normalize(&[0.0; 6]).unwrap();
        assert_eq!(zero.sum.to_bits(), 0.0f32.to_bits());
        assert_eq!(zero.weights, [0.0; 6]);
        let one = normalize(&[1.0; 6]).unwrap();
        assert_eq!(one.sum.to_bits(), 6.0f32.to_bits());
        assert_eq!(
            one.weights.map(f32::to_bits),
            [1.0f32 / 6.0; 6].map(f32::to_bits)
        );
        assert!(normalize(&[f32::MAX; 6]).is_err());
    }
}
