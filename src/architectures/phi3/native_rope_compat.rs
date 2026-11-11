//! Compatibility with the pinned macOS ARM64 GGML CPU D96 RoPE operator.
//! Tested at the manifest positions; this is not a claim of higher accuracy.
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]
// SAFETY: macOS SDK math.h declares __sincosf_stret(float) with this two-float
// C return layout. It takes no pointers and has no caller-owned state. Keep
// this module and all calls restricted to macOS ARM64. The pinned GGML binary
// calls this same libSystem entrypoint; Rust sin_cos uses separate functions.
#[repr(C)]
struct NativeSinCos {
    sin: f32,
    cos: f32,
}
unsafe extern "C" {
    fn __sincosf_stret(x: f32) -> NativeSinCos;
}
// Source-ordered F32 recurrence and paired native trig, as pinned GGML CPU RoPE.
#[inline(never)]
pub fn native_rope_table(
    position: usize,
    dim: usize,
    base: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, f32) {
    let factor = base.powf(-2.0 / (dim as f32));
    let mut theta = position as f32;
    let mut angles = Vec::with_capacity(dim / 2);
    let mut cos = Vec::with_capacity(dim / 2);
    let mut sin = Vec::with_capacity(dim / 2);
    for _ in 0..dim / 2 {
        angles.push(theta);
        let sc = unsafe { __sincosf_stret(theta) };
        cos.push(sc.cos);
        sin.push(sc.sin);
        theta *= factor;
    }
    (angles, cos, sin, factor)
}
#[inline(never)]
pub fn native_rope_rotate(input: &[f32], dim: usize, cos: &[f32], sin: &[f32]) -> Vec<f32> {
    assert!(dim > 0 && dim.is_multiple_of(2) && input.len().is_multiple_of(dim));
    assert_eq!(cos.len(), dim / 2);
    assert_eq!(sin.len(), dim / 2);
    let mut out = vec![0.; input.len()];
    let half = dim / 2;
    for (src, dst) in input.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        for i in 0..half {
            let x0 = src[i];
            let x1 = src[i + half];
            let c = cos[i];
            let s = sin[i];
            dst[i] = x0.mul_add(c, -(x1 * s));
            dst[i + half] = x0.mul_add(s, x1 * c);
        }
    }
    out
}
