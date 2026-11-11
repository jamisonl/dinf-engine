# HF-contract projector reference

`hf-contract-v1.patch` changes three projector graphs in llama.cpp **b8920**
(commit `15fa3c493bfcd040b5f4dcb29e1c998a0846de16`) so that they follow the
Hugging Face model definitions. The result is a separate reference named
`hf-contract-v1`; it is not stock llama.cpp and does not change dinf-engine.
Keep stock reference results alongside its results.

- Qwen3-VL: align-corners bilinear position interpolation without antialias,
  and erf GELU in the merger.
- Gemma4: RMSNorm before the embedding projection, with no extra norm after it.
- Llama4 Scout: final pixel-shuffle spatial transpose, and erf GELU in the
  tower and adapter.

The Hugging Face references are Transformers commit
`fd3cf0140de03f7619ac95b5cfed05b711ff9e22` for Gemma4 and Scout, with Scout
also checked against upstream's cited commit
`2932f318a20d9e54cc7aea052e040164d85de7d6`.

Comparisons use identical normalized pixel bits and the fixed gates
relative L2 <= 0.01 and cosine >= 0.9999. They check projected embeddings,
not caption quality or every image preprocessing policy, and they do not
assert identical arithmetic across backends.

`SOURCE.json` records the source archive, patch, before/after source hashes,
runtime files and input fixtures. Patch SHA-256:
`98a3f2f65673e691a8dc8c92495c6c7462b429685bb6fd2768cb64ed582391b6`.
`LICENSE.llama.cpp` retains the upstream MIT notice (Copyright (c) 2023-2026
The ggml authors); keep it with the patch.

## Build

Requirements: a C++17 compiler, CMake 3.20 or newer, Git, Python 3, the pinned
llama.cpp source, matching b8920 libllama libraries and headers, and ggml
0.10.0 libraries and headers. The recorded build used macOS arm64 shared
libraries. The CMake paths are parameters; they do not select or install a
runtime. The binary and runtime hashes in `SOURCE.json` identify the recorded
build, so a new build must record its own.

Set `ORACLE` to this directory, `LLAMA_SOURCE` to a fresh llama.cpp checkout,
`LLAMA_PREFIX` and `GGML_PREFIX` to the matching installed prefixes, `BUILD` to
a new build directory and `FIXTURES` to the crate's `fixtures/` directory. Use
absolute paths.

```sh
git -C "$LLAMA_SOURCE" checkout --detach 15fa3c493bfcd040b5f4dcb29e1c998a0846de16
test "$(git -C "$LLAMA_SOURCE" rev-parse HEAD)" = 15fa3c493bfcd040b5f4dcb29e1c998a0846de16
test -z "$(git -C "$LLAMA_SOURCE" status --porcelain)"
git -C "$LLAMA_SOURCE" apply --check "$ORACLE/hf-contract-v1.patch"
git -C "$LLAMA_SOURCE" apply "$ORACLE/hf-contract-v1.patch"
cmake -S "$ORACLE" -B "$BUILD" \
  -DLLAMA_SOURCE="$LLAMA_SOURCE" -DLLAMA_PREFIX="$LLAMA_PREFIX" -DGGML_PREFIX="$GGML_PREFIX"
cmake --build "$BUILD" --target mtmd --parallel 4
c++ -std=c++17 -O2 -Wall -Wextra -Werror \
  '-DVISION_ORACLE_VARIANT="hf-contract-v1"' \
  '-DVISION_ORACLE_PATCH_SHA256="98a3f2f65673e691a8dc8c92495c6c7462b429685bb6fd2768cb64ed582391b6"' \
  -isystem "$LLAMA_SOURCE/tools/mtmd" -isystem "$GGML_PREFIX/include" \
  -isystem "$LLAMA_PREFIX/include" "$FIXTURES/vision_reference.cpp" \
  -L"$BUILD/lib" -L"$LLAMA_PREFIX/lib" -L"$GGML_PREFIX/lib" \
  -Wl,-rpath,"$BUILD/lib" -Wl,-rpath,"$LLAMA_PREFIX/lib" -Wl,-rpath,"$GGML_PREFIX/lib" \
  -lmtmd -lllama -lggml-base -lggml -o "$BUILD/vision_reference_hf_v1"
```

## Run

Before loading a model, run the producer with `--fixture PREFIX 64` and check
the RGB hash against `SOURCE.json`, which also pins the 256 and 768 fixtures.
Record hashes of the producer source, the three patched files, the
executable, the patched libmtmd, libllama, the ggml core and the CPU/Metal
backend modules actually loaded.

On macOS, set `DYLD_PRINT_LIBRARIES=1` in the producer's immediate parent and
keep stderr; require the patched libmtmd in the trace and reject the stock
one. `/usr/bin/time` can strip that variable, so if the run is timed, set it
from a Python child that then calls `os.execv` on the producer. A requested
`metal` backend label is not proof of Metal use; record the compute-buffer and
backend evidence. Clear DINF/CANDLE/GGML dispatch variables and dynamic-library
overrides first.

```sh
"$BUILD/vision_reference_hf_v1" TEXT.gguf MMPROJ.gguf TEXT_SHA256 PROJECTOR_SHA256 cpu PREFIX 64
cargo run --locked --release --example qualify_vision_reference -- \
  TEXT.gguf MMPROJ.gguf cpu PREFIX.json ACTUAL_PREFIX
```

The producer trusts the model hashes passed to it, so hash both complete files
and verify stable file identities before and after each process;
`qualify_vision_reference` checks both hashes itself. Use fixture size 256 for
the Qwen same-grid input or 768 for the Qwen 48x48 native position grid.
Shapes and normalized pixels must still match. Raw F32 outputs are kept for
failing comparisons.

## CPU Q8-weight/F32-activation profile

`f32-gelu-w8a32` is a CPU-only reference profile layered on `hf-contract-v1`.
It keeps stored weights and the original F32 activations:

- Stored Q8_0 and F16 matrices are widened exactly to F32 before each graph
  projection. By default, the Q8 operator quantizes activations to Q8, and the
  F16 operator rounds its right operand to F16 and, on ARM, accumulates mostly
  in F16; neither matches the F32 activation contract.
- F32 GELU operators use F32 tanh-GELU instead of the F16 lookup table. F16
  GELU operators keep the lookup.

Metal keeps the `hf-contract-v1` graph. `CPU_W8A32.json` pins the patches,
the tested plugin and library, and the saved-input F64 component checks that
separate these effects. Widened-weight F32 projection reduced local error by a
factor of 5,010 to 62,005 on the six measured Q8 inputs. No tolerance changed.

On the fixed 256x256 Qwen3.8 Q8 CPU projector input, the profile passes with
relative L2 4.256504229e-6 and cosine 0.999999999991305 against the unchanged
0.01 / 0.9999 limits. The result does not extend to other projectors, image
sizes or caption quality.

### Build the CPU profile

1. Start from the pinned b8920 source and the matching llama/ggml headers and
   libraries described above.
2. Apply `hf-contract-v1.patch`, then `cpu-w8a32-v1.patch` (which touches only
   `tools/mtmd/clip.cpp` and `clip-graph.h`), and build `mtmd` with the same
   CMake commands.
3. Copy `fixtures/vision_reference.cpp` to a fresh work directory, apply
   `cpu-w8a32-producer-v1.patch` to the copy, and compile it with the same
   command, definitions, include directories and library paths as above. The
   patched producer requires the CPU backend and an explicit plugin and
   profile, never calls `ggml_backend_load_all`, and records both choices in
   its JSON output.
4. Build an isolated CPU plugin with F32 GELU, for example with
   `../text-coverage/cpu-reference/build_cpu_reference_plugin.py
   --f32-activations`. That profile also repairs flash attention and softplus,
   which do not occur in this vision graph; keep its own profile name and
   build receipt. A newly built plugin must be validated on its own outputs.

Set these variables only in the producer's clean child environment:

```sh
DINF_VISION_CPU_REFERENCE_PLUGIN=/absolute/path/to/isolated/cpu/plugin
DINF_VISION_CPU_REFERENCE_POLICY=f32-gelu-w8a32
```

Then run the producer and `qualify_vision_reference` commands above with
fixture size 256. The producer reads only the text model's header, not its
weights. Keep the normalized pixel hash, the raw F32 outputs and the explicit
profile name with the result.

Each widened reference matrix is capped at 128 MiB, and the graph can hold
several at once. The recorded run used a 6 GiB process cap for the reference
and 3 GiB for the engine, four threads, at least 24 GiB of available memory
and a 128 MiB swap-growth limit. These bounds apply to this fixture only.
