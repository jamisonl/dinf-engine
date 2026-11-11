# Vendored gemm-common

This directory contains the published `gemm-common` 0.19.0 crate with four
function attributes added to fix generic AArch64 builds. Its MIT license and
both published Cargo manifests remain unchanged.

- Crate SHA256: `88027625910cc9b1085aaaa1c4bc46bb3a36aad323452b33c25b5e4e7c8e2a3e`.
  This matches the downloaded crate archive and the registry lock entry.
- Source commit: [`86102c5b712737978371ac9ef7a11982f686d7bc`](https://github.com/sarah-quinones/gemm/tree/86102c5b712737978371ac9ef7a11982f686d7bc),
  recorded in `.cargo_vcs_info.json` and still upstream HEAD on 2026-11-04.
- Upstream tracks the same generic AArch64 debug-build failure in
  [issue 31](https://github.com/sarah-quinones/gemm/issues/31). The checked HEAD
  does not contain this fix.

## Local change

`src/simd.rs` adds `#[target_feature(enable = "neon,fp16")]` to the four
unsafe assembly helpers `vmulq_f16`, `vaddq_f16`, `vfmaq_f16`, and
`vfmaq_laneq_f16`. Their vector half-precision instructions need those CPU
features. The scalar half-precision helpers already carry feature attributes.
An `inline` hint alone does not supply the features when the compiler emits a
helper as its own function, which caused eleven `instruction requires:
fullfp16` errors in the generic Linux debug build.

No instruction, arithmetic, public signature, or dispatch rule changed. The
workspace still targets generic AArch64; it does not enable FP16 globally.
[Rust's per-function feature attribute](https://doc.rust-lang.org/reference/attributes/codegen.html#the-target_feature-attribute)
lets these helpers compile while keeping their feature requirement at each
call boundary.

The existing runtime checks remain necessary:

- [`gemm-f16/src/gemm.rs`, lines 814–826](https://github.com/sarah-quinones/gemm/blob/86102c5b712737978371ac9ef7a11982f686d7bc/gemm-f16/src/gemm.rs#L814-L826)
  selects the FP16 kernel only after detecting both Neon and FP16. Without
  FP16 it uses the Neon path with F32 accumulation; without Neon it uses the
  scalar path.
- `NeonFp16` has a private field. Both `MixedSimd::try_new` implementations in
  `src/simd.rs` check Neon and FP16 before creating it.
- With `std`, `feature_detected!` in `src/lib.rs` calls
  `std::arch::is_aarch64_feature_detected!`. The generated FP16 microkernels
  already carry `neon,fp16` attributes.

## Validation

On 2026-11-04, Rust 1.96.1 in an AArch64 Linux container built and ran a small
GEMM client in both debug and release modes with this patch. No `RUSTFLAGS`,
global FP16 feature, or native CPU target was set. Both runs reported
`compiled_fp16=false` and `detected_fp16=true`.

Each run compared every F16 matrix output with an independent F32 dot product
for shapes `(M,N,K)` of `(1,1,1)`, `(3,7,9)`, `(16,32,64)`, `(33,17,65)`, and
`(128,129,128)`. Small integer inputs keep the expected sums exact in F16;
all comparisons passed. These cover full tiles and row, column, and depth
tails. The engine's unpatched generic debug build supplied the failing case.
This host has FP16; runtime execution on an AArch64 CPU without FP16 remains
untested. The fallback claim above comes from source review.

Keep the local patch in engine source bundles. Cargo does not pass a root
`[patch.crates-io]` entry to downstream crate consumers. Remove it only after
a dependency release fixes these helpers and generic AArch64 debug and release
builds pass without a global `+fp16` flag.
