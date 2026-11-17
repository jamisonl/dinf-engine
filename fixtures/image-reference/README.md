# Qwen Image 2.1 reference checks

These helpers compare the engine's text encoder, denoiser and VAE on real
weights with pinned Diffusers 0.42.0 and PyTorch 2.15.0. Weights, Python
environments and model outputs stay outside the source tree.

Recorded results:

- CPU/F32 and Metal/F32 components, and the CPU F32 four-step/cache run, pass
  their fixed limits.
- The production BF16 Metal text encoder matches native output exactly for
  all 34 hidden rows and the 20-row conditioning suffix.
- The BF16 Metal denoiser matches native output exactly at one step and at
  all four steps in both cache modes, with exact cached/uncached replay. It
  uses 512x512 latents and 32 synthetic conditioning rows, not the text
  check's output.

These checks do not cover end-to-end generation, 40-step schedules, image
quality, longer prompts, other hardware or other kernel sizes. `SOURCE.json`
pins upstream revisions, source hashes, inputs and checkpoint files.
`requirements.lock` pins Python packages and distribution hashes;
`rust/Cargo.lock` pins the engine's dependency versions and vendored patches.

`rust/Cargo.toml.example` expects this directory at
`fixtures/image-reference` inside the crate, with the vendored crates in
`third_party/` at the crate root. The `.example` suffix keeps Cargo from
treating it as a nested package when the crate is packaged.

## Setup

The Python lock was tested with Python 3.15.1 on macOS arm64; other platforms
need their own lock. Set `FIXTURE` to this directory, `SNAPSHOT` to a local
Qwen Image 2.1 snapshot and `WORK` to a new absolute output directory.

```sh
mkdir "$WORK"
python3.15 -m venv "$WORK/venv"
"$WORK/venv/bin/python" -m pip install --require-hashes -r "$FIXTURE/requirements.lock"
cp "$FIXTURE/rust/Cargo.toml.example" "$FIXTURE/rust/Cargo.toml"
cargo build --manifest-path "$FIXTURE/rust/Cargo.toml" --locked --release \
  --target-dir "$WORK/rust-target"
python3 "$FIXTURE/generate_inputs.py" "$WORK/inputs"
"$WORK/venv/bin/python" "$FIXTURE/test_checker.py"
"$WORK/venv/bin/python" "$FIXTURE/test_bf16_checker.py"
```

Copying the manifest is the only change to the source tree; build output goes
to `WORK`. Use the same Cargo profile, features and flags as the engine build
under test (the recorded build used default release settings and no
`RUSTFLAGS`).

Before loading, hash every checkpoint shard and config and compare them, with
stable file identities, against `SOURCE.json`. Record the input, source and
lock, executable and loaded native library hashes with each run. The
comparators check raw hashes, shapes, finite values, input equality and runtime
labels, but labels alone do not prove build or weight identity.

Run environment:

- Clear `DINF_`, `CANDLE_`, `GGML_`, `TORCH_`, `PYTORCH_` and `DYLD_`
  variables, preloads, and Python and library path overrides.
- Set `HF_HUB_OFFLINE=1` and `PYTORCH_ENABLE_MPS_FALLBACK=0`.
- Run one model process at a time with at least 64 GiB of available memory.
- Keep stderr, timing, memory and raw outputs when a run fails.

## F32 components and cache

Use a new output directory for each call. Run `schedule` (no weights) first,
then `denoise_single`, then `vae`. The VAE decodes shared normalized latents to
512x512 RGBA, compared as clamped F32 values before PNG quantization.

```sh
MODE=denoise_single
"$WORK/venv/bin/python" "$FIXTURE/reference.py" \
  "$SNAPSHOT" "$WORK/inputs/input.json" "$MODE" cpu f32 "$WORK/ref-$MODE"
"$WORK/rust-target/release/dinf-image-component-audit" \
  "$SNAPSHOT" "$WORK/inputs/input.json" "$MODE" cpu f32 "$WORK/engine-$MODE"
"$WORK/venv/bin/python" "$FIXTURE/check_results.py" \
  "$WORK/ref-$MODE" "$WORK/engine-$MODE" "$FIXTURE/numeric-gates.json" "$WORK/check-$MODE.json"
```

For Metal/F32, change only the engine backend to `metal` and use a new output
path. Component limits are relative L2 <= 0.01 and cosine >= 0.9999; input
bits, full shapes and finiteness must match, and the scheduler's maximum
absolute error must be <= 1e-6. Outputs are never rescaled or row-selected.
Exports are little-endian F32, including values computed in BF16.

For the four-step check, run `denoise_uncached` and `denoise_cache` separately
in both runtimes and compare matching modes across runtimes with the component
limits. Within each runtime, compare cached and uncached with
`check_results.py ... --cache`, which requires opposite modes and applies
relative L2 <= 1e-5 and cosine >= 0.99999999. Each step saves its full
prediction and next latent. A zero exit status is not numeric acceptance: the
report must have both `valid` and `passed` set to true.

## Production BF16 denoiser

On Metal, the checkpoint's text encoder and transformer run in BF16. The VAE
weights are F32 and the loader keeps them in F32, so the F32 Metal VAE check
already covers its production dtype; do not downcast it. The shared input
values are exactly representable in BF16.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 "$WORK/venv/bin/python" \
  "$FIXTURE/bf16_preflight.py" "$WORK/bf16-preflight.json"
PYTORCH_ENABLE_MPS_FALLBACK=0 "$WORK/venv/bin/python" "$FIXTURE/reference.py" \
  "$SNAPSHOT" "$WORK/inputs/input.json" denoise_single mps bf16 "$WORK/ref-bf16"
"$WORK/rust-target/release/dinf-image-component-audit" \
  "$SNAPSHOT" "$WORK/inputs/input.json" denoise_single metal bf16 "$WORK/engine-bf16"
"$WORK/venv/bin/python" "$FIXTURE/check_bf16_results.py" \
  "$WORK/ref-bf16" "$WORK/engine-bf16" "$FIXTURE/numeric-gates.json" "$WORK/check-bf16.json"
```

The preflight runs nine small native MPS operations without checkpoint files.
The BF16 comparator requires an MPS reference and a Metal engine run, checks
that F32 exports hold BF16 values, and applies the component limits.

### Four-step trajectory

`check_bf16_trajectory.py` checks all four predictions and next latents
against the component limits at every step. It needs separate native MPS and
engine Metal exports for both `denoise_uncached` and `denoise_cache`, all from
the fixed generated inputs, and both schedules must match the recorded
schedule within 1e-6. Within each runtime, cached and uncached exports must be
byte-identical across all 12 arrays (eight step outputs, two inputs and two
schedule arrays). This exact-cache rule applies only to this BF16 profile.

The checker rejects missing or extra steps, wrong modes or runtime labels,
wrong shapes, nonfinite values, non-BF16 values and output files whose names
do not match their keys. It bounds and checks regular files before hashing and
hashes exactly the bytes it compares. The gate file binds each schedule to its
original little-endian F32 hash.

Synthetic self-test (no checkpoint):

```sh
"$WORK/venv/bin/python" "$FIXTURE/test_bf16_trajectory.py"
```

Full comparison, with fresh output directories and the same weights, inputs,
build and environment as above:

```sh
for MODE in denoise_uncached denoise_cache; do
  PYTORCH_ENABLE_MPS_FALLBACK=0 "$WORK/venv/bin/python" "$FIXTURE/reference.py" \
    "$SNAPSHOT" "$WORK/inputs/input.json" "$MODE" mps bf16 "$WORK/ref-bf16-$MODE"
  "$WORK/rust-target/release/dinf-image-component-audit" \
    "$SNAPSHOT" "$WORK/inputs/input.json" "$MODE" metal bf16 "$WORK/engine-bf16-$MODE"
done
"$WORK/venv/bin/python" "$FIXTURE/check_bf16_trajectory.py" \
  "$WORK/ref-bf16-denoise_uncached" "$WORK/ref-bf16-denoise_cache" \
  "$WORK/engine-bf16-denoise_uncached" "$WORK/engine-bf16-denoise_cache" \
  "$FIXTURE/bf16-trajectory-gates.json" "$WORK/inputs/input.json" \
  "$WORK/check-bf16-trajectory.json"
```

## Text conditioning

The same environment, snapshot and provenance rules apply. The fixed prompt
has 34 token IDs with a 14-token system prefix. Both the full 34x4096 hidden
output and the 20x4096 conditioning suffix must meet relative L2 <= 0.01 and
cosine >= 0.9999, and the engine's tokenizer IDs and drop index must match the
reference. The engine hook is test-only and adds no public API.

`text_reference.py prepare` hashes every required text shard, config, index
and tokenizer file. The reference uses Transformers 5.22.1 with eager
attention and no cache.

```sh
"$WORK/venv/bin/python" "$FIXTURE/text_reference.py" prepare \
  "$SNAPSHOT" "$WORK/text-f32-input.json" --compute-mode f32
"$WORK/venv/bin/python" "$FIXTURE/text_reference.py" produce \
  "$SNAPSHOT" "$WORK/text-f32-input.json" "$WORK/text-f32-reference"
DINF_IMAGE_TEXT_REFERENCE="$WORK/text-f32-reference.json" \
DINF_IMAGE_TEXT_SNAPSHOT="$SNAPSHOT" \
DINF_IMAGE_TEXT_ACTUAL_PREFIX="$WORK/text-f32-engine" \
DINF_IMAGE_TEXT_BACKEND=cpu DINF_IMAGE_TEXT_MODE=f32 \
cargo test --release --lib --locked \
  diffusion::pipeline::text_reference_tests::real_image_text_conditioning_matches_reference \
  -- --exact --ignored --nocapture
```

For production BF16, prepare a fresh input with
`--compute-mode production_bf16` (which selects native MPS/BF16) and produce a
fresh reference with `PYTORCH_ENABLE_MPS_FALLBACK=0`. Run the engine test with
fresh paths, `DINF_IMAGE_TEXT_BACKEND=metal` and
`DINF_IMAGE_TEXT_MODE=production_bf16`. The engine uses the production
loader's `force_f32=false` path and asserts the actual builder and hidden
dtypes. Both runtimes export F32LE bytes regardless of compute dtype, and the
test saves raw outputs and a report before failing on a numeric mismatch.

### Output capture

The checkpoint selects `QwenImage21Pipeline` (not the older
`pipeline_qwenimage.py`). In Diffusers 0.42.0,
`pipelines/qwenimage21/pipeline_qwenimage21.py:304-318` hooks the language
model's final norm to return its input, calls the encoder with
`output_hidden_states=True` and selects `hidden_states[-1]`.

In Transformers 5.22.1, `utils/output_capturing.py:234-246,293-307` ties the
last recorded hidden state to `last_hidden_state`, which `Qwen3VLTextModel`
sets to its final norm output (`modeling_qwen3_vl.py:840-843`). With the hook
installed, the encoder's last recorded state and this reference's direct
language output are therefore the same pre-norm tensor; without it, the
recorded state is post-norm.

Keep exact token IDs, full shard hashes, dtype and device records and all
hidden values with each result. This checks numeric conditioning, not prompt
or image quality.
