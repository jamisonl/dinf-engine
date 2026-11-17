# Vendored Candle core

This directory is `candle-core` 0.11.0 from crates.io (published crate
checksum
`5ecb245093b0f791b89d3420c3df9c6d49c60ab63ba54db896bf8a3baf486706`). It
retains Candle's declared `MIT OR Apache-2.0` license and the upstream license
text included with the published crate.

The upstream fix carried here is the ZIP 9 API backport from Hugging Face Candle
commit
[`70f72e9cfdbfc570846ca363ead2135394af3e9a`](https://github.com/huggingface/candle/commit/70f72e9cfdbfc570846ca363ead2135394af3e9a),
the fix behind [Candle pull request #4041](https://github.com/huggingface/candle/pull/4041):
the two NPZ filename reads and the Pickle filename iterator now propagate the
fallible ZIP 9 filename API. The vendored manifest raises its ZIP requirement
to `9.0.3`; the root lockfile selects the current compatible `9.1.2` release.
One upstream documentation line has trailing whitespace normalized.

Local changes for the CUDA backend, marked `Local patch` or `local
addition` in the source:

- `src/quantized/cuda.rs`: the indexed expert kernels honor view offsets,
  and one input row whose Q8_1 form was already written skips the
  quantization launch.
- `src/quantized/q8_1_memo.rs` (new): the per-thread table of activations
  that already have a Q8_1 form on the device.

Remove the workspace patch and this directory after upgrading to a released
Candle version that includes the ZIP 9 compatibility fix.
