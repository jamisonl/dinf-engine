# Validation

dinf-engine is validated at three levels: unit and integration tests, numeric
agreement with llama.cpp, and request-state checks on full models.

## Numeric agreement

For each tested checkpoint and backend, the engine and a pinned llama.cpp
build score the same prompt and the same fixed continuation tokens
(teacher forcing). Each scored row compares the full-vocabulary logits. A case
passes when all of these hold:

- the top token matches on at least 95% of rows,
- the mean KL divergence from the reference is at most 0.02,
- the maximum per-row KL divergence is at most 0.1.

The core matrix covers six checkpoints in full precision, Q8_0 and Q4_K_M on
CPU and Metal, using 128-row cases at 512, 2048 and 4096-token context budgets:

| Checkpoint | Full precision | Q8_0 | Q4_K_M |
| --- | --- | --- | --- |
| Qwen2.5-Coder-1.5B-Instruct | BF16 | ✓ | ✓ |
| Mistral-7B-Instruct-v0.3 | BF16 | ✓ | ✓ |
| Phi-3-mini-4k-instruct | F32 | ✓ | ✓ |
| Qwen3.5-4B | BF16 | ✓ | ✓ |
| Qwen3-30B-A3B | BF16 | ✓ | ✓ |
| DeepSeek-V2-Lite-Chat | BF16 | ✓ | ✓ |

Several larger models also pass targeted 64-row comparisons on both backends,
including MiniMax M2.5, Nemotron-3-Nano and Qwen3.6/3.8.

Some heavily quantized or pruned checkpoints are numerically unstable: llama.cpp
disagrees with itself across CPU and Metal by more than the limits above. For
those, the reference's own cross-backend spread is measured first, and
agreement is judged against a well-conditioned checkpoint of the same family.

## Request-state checks

Every supported checkpoint runs a short check on each supported backend:
finite logits, identical tokens with chunked versus whole-prompt prefill,
cancellation during prefill and decode, interleaved requests, cache positions
and cleanup. Draft, vision and image-generation pairs run equivalent checks for
draft fallback, image sensitivity, request isolation, pipeline replay and
cancellation.

These checks measure implementation agreement and state correctness. They do
not grade answer quality.

## Running the checks

The harnesses live in [`examples/`](../examples) and
[`fixtures/`](../fixtures):

```sh
# Short request-state check for one model
cargo run --release --example qualify_text -- model.gguf metal --smoke

# Compare against saved llama.cpp reference logits
cargo run --release --example qualify_reference -- model.gguf metal reference.json

# Vision, draft and image pipelines
cargo run --release --example qualify_vision -- model.gguf projector.gguf metal 4
cargo run --release --example qualify_draft -- model.gguf draft.gguf metal 32
cargo run --release --example qualify_image -- snapshot_dir metal output_dir 2
```

Use `cpu` in place of `metal` for CPU runs. The
[text coverage fixtures](../fixtures/text-coverage/README.md) describe the
corpus, model manifest and reference build. The pinned model files and recorded
results are in
[`fixtures/text-coverage/`](../fixtures/text-coverage).
