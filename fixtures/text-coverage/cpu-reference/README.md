# CPU reference profiles

This helper builds an isolated CPU plugin for the pinned llama.cpp reference.
It does not alter installed libraries, engine math, or numeric limits.

| Profile | Build option | Contract |
| --- | --- | --- |
| `ggml-cpu-flash-f32-v2` | Default | F32 sums for requested-F32 flash attention; retain the half-rounded query and F16 K/V. The six-model matrix uses this profile. |
| `ggml-cpu-f32-activations-v3` | `--f32-activations` | Add accurate F32 softplus and tanh-GELU; keep F16 operators unchanged. Separate fixed-token checks use this profile. |
| `ggml-cpu-flash-f32-query-v4` | `--original-f32-query` | Extend v2 to retain the original F32 query in requested-F32 flash attention with F16 keys. |

The two optional flags cannot be combined. Each result must keep its selected
profile and build receipt; a new profile does not change an older verdict.
These reference repairs do not establish engine or model correctness.

The build requires macOS 26+ on arm64 with the `apple_m4` CPU ISA, Apple clang 21, and
the exact recorded ggml 0.10.0 Homebrew runtime. It uses the `apple_m4` ISA,
Accelerate, OpenMP, repack, and LTO. The builder rejects a different runtime,
config, architecture, or source tree. Python 3.11+ is required. This fixture
does not claim Linux or other CPU support.

## Build a plugin

Obtain the source archive in `SOURCE.json`. Set `FIXTURE` to this directory,
`DINF_LLAMA_SOURCE` to the extracted llama.cpp tree, `DINF_GGML_PREFIX` and
`DINF_OPENMP_PREFIX` to the pinned installed prefixes, and `DINF_WORK` to an
external work directory. Use absolute paths and a fresh output directory.

```sh
python3 "$FIXTURE/build_cpu_reference_plugin.py" \
  --ggml-source "$DINF_LLAMA_SOURCE/ggml" \
  --ggml-prefix "$DINF_GGML_PREFIX" \
  --openmp-prefix "$DINF_OPENMP_PREFIX" \
  --output "$DINF_WORK/cpu-v2"
```

Add `--check` to verify inputs without creating output. Select another profile
with its flag and a new output path, such as `cpu-v3` or `cpu-v4`. The builder
checks all 82 source files, applies pinned patches to a copy, builds with one
compiler job, and links the unchanged installed GGML base. It records commands,
source/runtime hashes, and the plugin path in `build-receipt.json`. It never
loads a model or initializes a GPU.

## Build the model reference producer

Set `DINF_LLAMA_PREFIX` to the matching installed llama prefix. From the
crate root, build the suite producer using the plugin receipt:

```sh
python3 fixtures/text-coverage/build_reference.py \
  --llama-source "$DINF_LLAMA_SOURCE" \
  --include-dir "$DINF_LLAMA_PREFIX/include" \
  --include-dir "$DINF_GGML_PREFIX/include" \
  --library-dir "$DINF_LLAMA_PREFIX/lib" \
  --library-dir "$DINF_GGML_PREFIX/lib" \
  --revision 15fa3c493bfcd040b5f4dcb29e1c998a0846de16 \
  --cpu-plugin-build "$DINF_WORK/cpu-v2/build-receipt.json" \
  --out "$DINF_WORK/llama-matrix-reference"
```

The builder runs tests without a backend or model, then loads only the pinned
BLAS and repaired CPU plugins and checks non-system loaded libraries. CPU
inference requires that policy. `GGML_BACKEND_PATH` alone is insufficient:
the stock loader can load an installed CPU backend first.

Pass the resulting `.build.json` to the [suite runner](../README.md#running-a-cell)
with `--reference-build`. The producer and independent checker require the
same CPU policy and library identities before and after each run. The suite
uses 128 submitted prompt tokens, eight-token physical batches, F16 K/V, eight
CPU threads, and fixed teacher IDs. Metal retains its own reference path.

For legacy fixed-token checks, add `--legacy` to the build command and select
the required plugin receipt. That producer keeps 4096-token batches and eight
CPU threads. Run it with the original prompt and teacher files:

```sh
"$DINF_WORK/legacy-reference" "$DINF_MODEL" cpu \
  "$DINF_WORK/prompt.txt" 64 "$DINF_WORK/reference" \
  "$DINF_MODEL_SHA256" - "$DINF_WORK/teacher.txt"
```

Hash the model first; the legacy producer checks its header and records the
supplied hash. Replace `-` with a complete shard manifest for split models.
Use `qualify_reference` with the original limits to compare engine outputs.

## Small component checks

Compile the probes as C++17 against the pinned ggml headers/libraries and
llama.cpp's JSON header. For example:

```sh
clang++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -I "$DINF_GGML_PREFIX/include" -I "$DINF_LLAMA_SOURCE/vendor" \
  "$FIXTURE/flash_probe.cpp" \
  -L "$DINF_GGML_PREFIX/lib" -Wl,-rpath,"$DINF_GGML_PREFIX/lib" \
  -lggml -lggml-base -o "$DINF_WORK/flash_probe"
"$DINF_WORK/flash_probe" \
  "$DINF_WORK/cpu-v2/build/plugins/libggml-cpu-apple_m4.so" \
  "$DINF_WORK/flash-output"
python3 "$FIXTURE/verify_flash_probe.py" \
  --plugin-build "$DINF_WORK/cpu-v2/build-receipt.json" \
  --result-dir "$DINF_WORK/flash-output" \
  --output "$DINF_WORK/flash-review.json"
```

The ten flash cases require six constant outputs to equal 4096 exactly and
four random cases to stay within 2e-6 of the half-query F64 reference.

For v3, compile `activation_probe.cpp` with the same include/library arguments
and run `activation_probe CPU_PLUGIN NEW_OUTPUT` once for stock and once for
v3. Softplus must meet its fixed 2e-7 relative bound. The 65,536-value GELU
check uses `1e-6 + 2e-6 * abs(F64 result)`; compare stock and v3 F16 GELU bytes
exactly. The repair uses `log1pf(expf(x))` below softplus's linear branch and
the existing F32 tanh-GELU function instead of an F16 lookup table.
`F32_ACTIVATIONS.json` pins the patches and source.

For v4, compile `query_probe.cpp` with the same arguments, then run:

```sh
"$DINF_WORK/query_probe" /absolute/v2/libggml-cpu-apple_m4.so "$DINF_WORK/query-control"
"$DINF_WORK/query_probe" /absolute/v4/libggml-cpu-apple_m4.so "$DINF_WORK/query-v4"
python3 "$FIXTURE/verify_query_probe.py" \
  "$DINF_WORK/query-control" "$DINF_WORK/query-v4" "$DINF_WORK/query-review.json"
```

Its nine cases cover one/three queries, 1/3/31/33/257 visible keys, 256/512
padded keys, and grouped heads. DEFAULT outputs must stay byte-identical.
Explicit-F32 output must pass the fixed F64 bound and improve the eight
nontrivial cases by more than ten times. `F32_QUERY.json` pins this profile.
None of these probes loads model weights.

Keep every build receipt and referenced file immutable, and keep failed
results alongside later results. `SOURCE.json` records source revisions,
hashes, patches, and tested runtimes; `LICENSE.ggml` retains the upstream MIT
notice. Supported models are listed in the [crate README](../../../README.md).
