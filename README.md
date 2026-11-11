# dinf-engine

A Rust inference engine for GGUF language models on CPU and Apple Metal, with
experimental CUDA support. It loads local GGUF weights, tokenizes prompts,
generates text and chat, and manages request and cache state. Applications
embed it through the high-level `Engine` API or the lower-level stage and
session APIs.

## Features

- Load single-file and split GGUF models; inspect metadata and memory needs.
- Generate plain text, chat, and model-specific tool calls; stream decoded text.
- Greedy and seeded sampling with temperature, top-k, top-p, token bias, and
  repetition, frequency and presence penalties.
- Grammar-constrained tool output.
- Chunked prefill, interleaved requests, cancellation and prompt-cache reuse.
- Pipeline stages: load a contiguous layer range and pass activations between
  stages.
- Optional quantized KV caches.
- Speculative decoding with DFlash2 draft models.
- Image input for supported vision models, and text-to-image generation for
  Qwen Image.

## Supported models

- Llama, Mistral and Qwen2
- Phi-3
- Qwen3 MoE
- Qwen3.5, Qwen3.6 and Qwen3.8
- Gemma 4
- Llama 4
- Nemotron-H and Nemotron 3 Nano
- DeepSeek V2
- DeepSeek V4 Flash (Metal only)
- DeepSeek V4.1 Flash (Metal only; pinned checkpoints)
- MiniMax M2.5
- GLM-5.3 Flash

Vision input is supported for Qwen3.5, Gemma 4 and Llama 4. Speculative
decoding uses DFlash2 draft models, and image generation uses Qwen Image.

Each entry is backed by at least one checkpoint that passed the request-state
checks in [`fixtures/text-coverage/basic-coverage.json`](fixtures/text-coverage/basic-coverage.json).
The engine also contains code paths for Muse Spark Open and the GLM-DSA
variant, but neither has been validated on a complete checkpoint, so they are
not supported.

## Performance

On Apple Silicon, dinf-engine matches or beats llama.cpp on recent Qwen and
Gemma models, both when reading the prompt and when generating text.

## Quick start

Requires Rust 1.96 or newer. Weights are not included; point the engine at a
local GGUF file.

```rust,no_run
use candle_core::Device;
use dinf_engine::{Engine, GenerationParams};

fn main() -> anyhow::Result<()> {
    let mut engine = Engine::load("model.gguf", Device::Cpu)?;
    let params = GenerationParams {
        max_tokens: 32,
        temperature: 0.0,
        ..GenerationParams::default()
    };
    let output = engine.generate("Hello", &params)?;
    println!("{}", output.text);
    Ok(())
}
```

On Apple Silicon, use `Device::new_metal(0)?` to run on the GPU.

- `Engine::builder` sets a layer range and KV-cache limits.
- `generate_chat` applies the model's chat template;
  `generate_chat_with_tools` adds tool schemas.
- `generate_tokens` accepts token IDs and streams decoded text to a callback.
- Speculative decoding runs through `continuous::GenerationSession` with an
  attached draft model.

See the API documentation (`cargo doc --open`) for details.

### Stage and session API

`Engine::generate*`, `PipelineStage::generate` and
`continuous::GenerationSession` manage prompt context automatically. Callers
driving stages directly must:

1. Create one `TextRequestContext::new(full_prompt_token_count)`, including
   any reused prefix, and call `prepare_text_request`.
2. Call `prefill_text` for each prompt chunk and `forward_text` for each decode
   step, with that context and absolute token positions. Use the same context
   on every pipeline stage.
3. If preparation returns `false`, the cache was incompatible and has been
   cleared: rebuild the prompt from position zero.

Stage and architecture APIs may change before a stable release.

## Building

| Backend | Command | Notes |
| --- | --- | --- |
| CPU | `cargo build --release` | Default. |
| Metal | `cargo build --release` on macOS | Enabled automatically on macOS. |
| CUDA | `cargo build --release --features cuda` | Experimental; requires a CUDA toolchain. Not covered by the model validation. |

The repository vendors patched copies of `candle-metal-kernels`, `cudaforge`
and `gemm-common` under `third_party/`, wired in through `[patch.crates-io]`
in `Cargo.toml`. Builds must use these patches.

## Testing

```sh
cargo fmt -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --lib --tests --locked
cargo test --doc --locked
```

On a Mac, set `DINF_REQUIRE_METAL_TESTS=1` so a missing GPU fails the Metal
tests instead of skipping them. Full-model tests are `#[ignore]`d and need
local weights.

Model-level validation compares the engine's logits against llama.cpp on the
same token history and checks request-state behavior such as chunked prefill,
cancellation and cache reuse. See [docs/VALIDATION.md](docs/VALIDATION.md).

## License

MIT; see [LICENSE](LICENSE). Code adapted from other projects keeps its
original license; see [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES). Model
weights have their own terms and are not part of this repository.
