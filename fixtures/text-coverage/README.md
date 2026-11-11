# Text coverage fixtures

These fixtures compare the engine's text logits with a pinned llama.cpp
reference and check request-state behavior. See
[docs/VALIDATION.md](../../docs/VALIDATION.md) for the overview and pass
limits.

## Files

| File | Purpose |
| --- | --- |
| `models.json` | Pinned model artifacts: repository, revision, size and SHA-256. |
| `corpus.json` | 24 cases: prose, Rust, arithmetic, JSON, extraction, multilingual text, conversation and retrieval at 512, 2048 and 4096-token budgets. |
| `corpus-smoke-6.json` | The six-case subset used by the smoke profiles. |
| `corpus-phi3-boundary-4096.json` | A Phi-3 case that fills the full 4096-token window. |
| `basic-coverage.json` | Every supported checkpoint with its pinned files and recorded request-state results. |
| `llama_matrix_reference.cpp` | The llama.cpp reference producer. |
| `fetch_models.py` | Downloads pinned artifacts and verifies size and hash. |
| `prepare_bf16.py` | Prepares the Qwen2.5-Coder BF16 file (chat-template metadata only; tensor bytes unchanged). |
| `build_engine.py`, `build_reference.py` | Build and pin the engine and reference binaries. |
| `run.py` | Runs one artifact on one backend and keeps all evidence. |
| `verify_results.py` | Independently recomputes every metric from the saved raw logits. |
| `run_basic.py` | Runs the short request-state check across `basic-coverage.json`. |
| `compare_tiers.py`, `summarize.py`, `archive_rows.py` | Reporting and archival helpers. |
| `cpu-reference/` | A repaired llama.cpp CPU attention plugin (see below). |

## What a case checks

Each case renders the same authored messages through each checkpoint's chat
template and tokenizer, then scores 128 fixed continuation tokens. The engine
and the reference must produce identical rendered text and token IDs. The
reference uses an F16 KV cache, 128-token prefill calls and fixed eight-token
physical batches; larger batches in llama.cpp's Metal path round F32 inputs to
lower precision before matrix multiplies.

Profiles:

- `smoke-6-short-state` (recommended): six cases, numeric checks first, then
  12 request-state checks on `conversation-512`. Stops at the first numeric
  failure.
- `smoke-6`: the same six cases with request-state checks on three cases.
- `extended-24`: all 24 cases; does not stop at the first failure.

State checks cover greedy and seeded generation with penalties, replay after
another request, two prefill chunk sizes, cancellation during prefill and
decode, interleaved requests, prompt-cache publication and reuse, and cache
owner isolation.

## Running a cell

Keep model files and results outside the repository.

1. Fetch the pinned weights:

   ```sh
   python3 fixtures/text-coverage/fetch_models.py --output-dir MODELS \
     --artifact ARTIFACT_ID --receipt receipt.json
   ```

2. Build the engine and the reference:

   ```sh
   python3 fixtures/text-coverage/build_engine.py --root "$PWD" --output ENGINE_BUILD
   python3 fixtures/text-coverage/build_reference.py --help
   ```

   The reference pins llama.cpp `15fa3c493bfcd040b5f4dcb29e1c998a0846de16`
   with GGML 0.10.0. CPU runs also need the [CPU reference
   plugin](cpu-reference/README.md): llama.cpp's ARM CPU attention accumulates
   in F16 even when F32 is requested, and the plugin restores F32 sums.

3. Write a JSON map from artifact ID to the GGUF path (for split files, the
   first shard), and install the pinned NumPy into a separate environment:

   ```sh
   python3 -m venv VENV
   VENV/bin/python -m pip install -r fixtures/text-coverage/requirements.txt
   ```

4. Run one artifact on one backend:

   ```sh
   python3 fixtures/text-coverage/run.py \
     --paths paths.json --engine-build ENGINE_BUILD/build.json \
     --reference-build REFERENCE_BUILD.json --identities IDENTITIES \
     --output RESULTS/ARTIFACT-metal --artifact ARTIFACT_ID --backend metal \
     --profile smoke-6-short-state --validation-python VENV/bin/python
   ```

   Every output directory must be new. A cell passes only when its
   `independent-review.json` reports `valid_evidence: true`, `passed: true`
   and `pilot: false`; the exit code alone is not sufficient.

`compare_tiers.py` reports descriptive cross-format scores for a checkpoint
once its full-precision, Q8_0 and Q4_K_M cells all pass.

## Request-state sweep

`run_basic.py` runs the short `basic-v1` check (`qualify_text --smoke`) for
every artifact in `basic-coverage.json`, verifying each file's hash first and
enforcing memory and time limits:

```sh
python3 fixtures/text-coverage/run_basic.py --paths paths.json \
  --binary target/release/examples/qualify_text --output RESULTS \
  --checksum-cache checksums.json
```

Use `--model ID` or `--backend cpu|metal` for a subset.

## Split-stage and tool checks

```sh
cargo run --release --example qualify_text -- MODEL metal --contracts
cargo run --release --example qualify_text -- MODEL cpu --tool-contract
```

`--contracts` compares a single stage with two- and three-stage splits
(eight logit rows must match bitwise) and runs a grammar-constrained tool
call with replay and cancellation. `--tool-contract` runs only the tool part.

## Phi-3 4K boundary case

```sh
cargo run --release --example qualify_matrix -- --prepare "$PHI_GGUF" \
  fixtures/text-coverage/corpus-phi3-boundary-4096.json prepared.json
llama_matrix_reference "$PHI_GGUF" "$BACKEND" prepared.json "ref-$BACKEND"
DINF_QUALIFY_PREFILL_CHUNK=128 cargo run --release --example qualify_reference -- \
  "$PHI_GGUF" "$BACKEND" "ref-$BACKEND/context-edge-4096.json" "actual-$BACKEND.f32"
```
