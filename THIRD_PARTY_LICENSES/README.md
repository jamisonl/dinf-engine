# Third-party licenses

dinf-engine's own code is licensed under the MIT license in
[`../LICENSE`](../LICENSE). Some source files, test fixtures and vendored
crates adapt or include third-party code under the licenses below. Each
adapted source file also carries its upstream notice or points to the files
listed here. Model weights are not part of this source and have their own
licenses.

Paths are relative to the crate root.

## License texts in this directory

| File | License | Used by |
| --- | --- | --- |
| `GGML-DS4-MIT.txt` | MIT (ggml authors, ds4.c authors, DeepSeek) | ggml/llama.cpp and ds4.c adaptations; copy of `src/architectures/deepseek41/LICENSE-MIT` |
| `APACHE-2.0.txt` | Apache License 2.0 | Diffusers and MTPLX adaptations; copy of `src/diffusion/LICENSE-APACHE` |
| `LLVM-APACHE-2.0-WITH-LLVM-EXCEPTION.txt` | Apache-2.0 WITH LLVM-exception | libc++ sort adaptation; copy of `src/architectures/qwen3moe/LICENSE-LLVM` |
| `CANDLE-MIT.txt` | MIT (Candle, MIT option of MIT OR Apache-2.0) | Candle kernel adaptation |
| `PYTORCH-BSD-3-CLAUSE.txt` | BSD-3-Clause | PyTorch MPS kernel adaptations |
| `MLX-MIT.txt` | MIT (Apple Inc.) | MLX origin of the PyTorch MPP attention kernel |
| `SLEEF-BOOST-1.0.txt` | BSL-1.0 | SLEEF trigonometric routines |

## Components

| Component | License | Where it is used | License and provenance files |
| --- | --- | --- | --- |
| [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | Metal and CPU kernels adapted in `src/architectures/` (DeepSeek V4 prefill/Q8/IQ1 kernels, `tensor_qmatmul_metal.rs`, `dflash2_cpu_attention.rs`, `phi3/native_*`, `qwen3moe/native_router_metal.metal`) | Header in each file; `GGML-DS4-MIT.txt` |
| [ds4.c](https://github.com/audreyt/ds4) | MIT | DeepSeek V4.1 port in `src/architectures/deepseek41/` | `src/architectures/deepseek41/NOTICE`, `src/architectures/deepseek41/LICENSE-MIT`, `GGML-DS4-MIT.txt` |
| [DeepSeek V4.1 encoding](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | MIT | Tokenizer and chat template port in `src/tokenizer_deepseek41.rs` | Header in that file; `src/architectures/deepseek41/NOTICE` |
| [Hugging Face Diffusers](https://github.com/huggingface/diffusers) | Apache-2.0 | Qwen Image implementation in `src/diffusion/` | `src/diffusion/NOTICE`, `src/diffusion/LICENSE-APACHE`, `APACHE-2.0.txt` |
| [MTPLX](https://github.com/youssofal/mtplx) | Apache-2.0 | Exact-width verification layouts in `src/architectures/tensor_qmatmul_metal.rs` | Header in that file; `APACHE-2.0.txt` |
| [Hugging Face Candle](https://github.com/huggingface/candle) | MIT OR Apache-2.0 (MIT selected) | Q4_K decoder in `src/architectures/deepseek4_qmatmul.rs` | `CANDLE-MIT.txt` |
| [LLVM libc++](https://llvm.org) | Apache-2.0 WITH LLVM-exception | Sort and heap ports in `src/architectures/qwen3moe/` | `src/architectures/qwen3moe/NOTICE-LLVM`, `LICENSE-LLVM`, `SOURCE.json`, `heap-provenance.json`; `LLVM-APACHE-2.0-WITH-LLVM-EXCEPTION.txt` |
| [PyTorch](https://github.com/pytorch/pytorch) | BSD-3-Clause | MPS attention, normalization and softmax kernels in `src/diffusion/` | `PYTORCH-BSD-3-CLAUSE.txt`, `PYTORCH-image-attention-SOURCE.txt`, `PYTORCH-image-norm-SOURCE.json`, `PYTORCH-text-norm-SOURCE.txt`, `src/diffusion/image_mpp_SOURCE.json` |
| [MLX](https://github.com/ml-explore/mlx) | MIT | Origin of the MPP attention body in `src/diffusion/image_mpp_attention.metal` | `MLX-MIT.txt`, `PYTORCH-image-attention-SOURCE.txt` |
| [SLEEF](https://github.com/shibatch/sleef) | BSL-1.0 | Rotary embedding math in `src/diffusion/rope_math.rs` | `SLEEF-BOOST-1.0.txt`, `SLEEF-rope-SOURCE.json` |

## Test fixtures

These files are used only by tests and reference tooling under `fixtures/`
and are not compiled into the library.

| Fixture | Upstream | License files |
| --- | --- | --- |
| `fixtures/llama-prefill-oracle/` | llama.cpp Metal sources | `LICENSE`, `NOTICE`, `SOURCE.json` |
| `fixtures/vision-hf-contract/` | llama.cpp patches | `LICENSE.llama.cpp`, `SOURCE.json` |
| `fixtures/text-coverage/cpu-reference/` | ggml CPU patches | `LICENSE.ggml`, `SOURCE.json` |
| `fixtures/dflash-cpu-reference/` | llama.cpp patch and producers | `LICENSE-MIT`, `inputs.json` |

## Vendored crates

The source release includes patched copies of three crates in `third_party/`.
The root `Cargo.toml` routes the dependency graph to them with
`[patch.crates-io]`.

| Crate | Upstream license | License and provenance files |
| --- | --- | --- |
| `third_party/candle-metal-kernels` (0.11.1) | MIT OR Apache-2.0; used under MIT | `LICENSE-MIT`, `UPSTREAM.md` |
| `third_party/cudaforge` (0.1.6) | MIT OR Apache-2.0 | `LICENSE-APACHE`, `NOTICE`, `SOURCE.json`, `DINF_PATCH.md` |
| `third_party/gemm-common` (0.19.0) | MIT | `LICENSE`, `UPSTREAM.md` |
