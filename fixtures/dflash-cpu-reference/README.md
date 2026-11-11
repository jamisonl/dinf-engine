# DFlash CPU reference repair

This fixture repairs a pinned, independent llama.cpp CPU reference used to
check DFlash draft outputs. It does not patch the engine or change any numeric
limit, and it contains no model weights, token data or saved model arrays.

The stock CPU F16 dot path can accumulate in F16 even when the graph requests
F32 precision, and it rounds an existing F32 right operand to F16.
`reference-repair.patch` keeps F16 left storage, keeps the F32 right operand,
and accumulates with F32 `fmaf` when the graph requests F32 precision.
Default-precision operations are unchanged. The DFlash weighted-value graph
also requests F32 precision. The producer runs on the CPU only and reports the
patch hash and precision rule.

## Requirements

- macOS on ARM64 with the active Xcode command-line SDK.
- Python 3.11 or newer (`hashlib.file_digest`).
- llama.cpp revision `de5122d879d00d2895c36be7e0cee94105aeeabd`.
- The matching static-library build pinned in `inputs.json`.

The recipe targets exactly this source and build pair; other revisions,
compiler flags and hosts are not supported. All source and archive inputs must
match their recorded hashes, and a mismatch stops before any output is written.

## Build and run the synthetic proof

Pass an existing upstream tree, its matching CMake build directory, and a new
output directory outside both input trees:

```sh
python3 build.py \
  --upstream /path/to/llama.cpp \
  --base-build /path/to/llama.cpp/build \
  --output /path/to/new-cpu-fixture
```

The script copies the two affected source files and the static libraries,
then builds an unpatched control and a repaired variant. It never modifies the
input trees, downloads code, opens weights or runs a model. Each step has a
180-second limit and the whole script a 300-second limit. Both synthetic graph
runs use two CPU threads and no GPU backend.

Six cases cover exact sums, the F16 overflow case, signed cancellation,
widths 32/128/512, and weighted values with non-half-exact probabilities.
The repaired requested-F32 output must satisfy the F64 forward-error bound
printed in the result, and every default-precision output bit must match the
unpatched control, including nonfinite outputs. The control's `passed` field
only means that it finished; its requested-F32 accuracy check is expected to
fail.

Outputs include `build-result.json`, per-step stdout/stderr, both producer
binaries and `repaired/proof.json`. Use a fresh output directory for each run
and keep failed outputs.

## Use as a model reference

`build.py` does not run models. To use the repaired producer as a reference:

1. Verify the target, draft and derived-copy hashes, and confirm that each
   copied tensor range matches its source (see below).
2. Run the unpatched control on the fixed component input and require all five
   original raw hashes (features, noise embeddings, hidden, logits, lattice).
3. Run the repaired producer with the same rows, block, inputs and tokens.
   Store its reference separately, with the patch hash and precision rule.
4. Run the engine consumer and the unchanged validator against it.

The synthetic bound does not replace the hidden-state, selector, token or KL
gates, and a passing state check does not establish draft quality. The
producer's command-line syntax is documented in its source.

## Verify a derived file

`verify_dflash_copy.py` uses only the Python standard library. It checks the
output of the C++ helper's `--derive` command before that file is passed to
`--produce`. It does not load a model, initialize a backend or run inference.

Use trusted SHA-256 values for the target and draft, record the derived file's
SHA-256, and save the `--derive` stdout as JSON. Derive with absolute paths so
that the manifest paths still resolve. From this directory:

```sh
python3 verify_dflash_copy.py \
  --target /path/to/target.gguf --target-sha256 "$TARGET_SHA256" \
  --draft /path/to/draft.gguf --draft-sha256 "$DRAFT_SHA256" \
  --derived /path/to/derived.gguf --derived-sha256 "$DERIVED_SHA256" \
  --manifest /path/to/derive.json \
  --receipt /path/to/new-copy-receipt.json
```

`--manifest-sha256 "$MANIFEST_SHA256"` additionally pins the manifest. The
receipt path must not exist. Exit status zero and `passed: true` mean all copy
checks passed; keep the receipt with the producer binary, source and runtime
pins and the numerical results.

The checker streams full hashes of all three files and compares every copied
tensor range between source and destination. It also checks:

- stable file identities before, during and after reads;
- the complete ordered tensor set, shapes, dtypes, offsets, byte lengths and
  disjoint in-file ranges;
- exact typed draft metadata, and target embedding/head widths and vocabulary
  sizes;
- that the derived file contains every draft tensor plus exactly the target's
  `token_embd.weight` and `output.weight`.

Supported inputs are unsplit GGUF v3 files with at most 256 MiB of header data
and a derived file of at most 10 GiB. Supported tensor types are F32, F16,
BF16, Q8_0, Q4_K, Q5_K and Q6_K, with block sizes taken from llama.cpp
`de5122d879d00d2895c36be7e0cee94105aeeabd`; other types are rejected. The
checker verifies copied bytes and input identity only, not model quality or
the numerical output of the C++ helper.

Run its unit tests with:

```sh
python3 -m unittest test_verify_dflash_copy
```

## Files and license

- `reference-repair.patch`: the requested-F32 repair.
- `inputs.json`: original and patched source hashes, static libraries,
  compiler arguments and fixture file hashes checked by `build.py`.
- `validation-summary.json`, `preparation.json`: hashes of the tested build
  outputs and the fixture files they were produced from.
- `LICENSE-MIT`: the upstream ggml MIT notice. Keep it with copies of the
  patch or fixture code.

The synthetic proof passes all six repaired cases and preserves every default
output bit. It rebuilds two archive members from the pinned build rather than
the whole upstream build, and it does not qualify a model. Model support and
draft limits are described in the [crate README](../../README.md).
