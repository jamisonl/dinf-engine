# Engine safety and backend trust boundaries

This document describes the safety assumptions of `dinf-engine`. It is an
inference library, not a sandbox. Run it in a process with only the filesystem,
devices, and privileges required for the models it is intended to serve.

## Input boundary

GGUF files, split shards, projector files, and image inputs are untrusted data.
The engine does not execute code from a model file, but model loading can use
large amounts of host and device memory and can keep decoded model data alive.

The structural checks are split across these paths:

- `src/gguf_limits.rs` performs a bounded preflight. It bounds GGUF
  versions, record counts, nesting, header bytes, tensor dimensions/elements,
  encoded tensor bytes, shard count, and aggregate split-shard headers using
  checked arithmetic and checked file offsets. Record names must be valid
  UTF-8, so lossy decoding cannot merge distinct names. `general.alignment`
  accepts only the 8-, 16-, and 32-bit integer types that both GGUF readers
  honor, so they cannot read tensor data at different offsets. Duplicate
  metadata keys and tensor names are rejected before the downstream parser can
  overwrite them;
  the duplicate index retains bounded offsets rather than untrusted strings.
- `src/gguf_loader.rs` parses known metadata types, rejects malformed arrays
  and non-finite numeric values, validates tensor shapes and layer ranges, and
  rejects duplicate tensor names across shards before loading weights.
  Split-file checks validate declared shard indices, counts, tensor totals,
  and any repeated architecture field. Files without split descriptors and
  merged files with `split.count = 0` remain valid. Embedding headers must
  provide at least the declared number of vocabulary rows; padding is allowed.
  These checks also apply to streaming loads and memory profiles. Regular
  decoder metadata requires positive F32-representable normalization epsilon
  and RoPE bases. Context metadata is capped at 1,048,576 tokens except for
  Llama4, which accepts up to 10,485,760 without changing the declared value.
  Llama4 caches RoPE for at most 8,192 positions and computes later ranges on
  demand; request/cache budgets still apply. Accepting that metadata does not
  qualify inference across the full declared context. Separate vision,
  diffusion, and draft-model parsers have their own checks.
  Llama/Mistral/Qwen2 stage construction also checks attention head geometry
  against the packed Q/K/V/output projection shapes before expanding embeddings
  or generating positional tables. Incompatible metadata fails at construction,
  not after allocating those tables and attempting the first request.
  These stages honor sliding windows in prefill, decode, and both batch paths.
  They reject non-unit RoPE scaling without a `rope_freqs.weight` tensor;
  metadata-only scaling is not implemented. Frequency tensors supply the
  per-dimension scaling once.
  DeepSeek2/GLM-DSA and Phi3 also check rotary widths before table allocation;
  Phi3 checks its fused QKV/output projection shapes at the same boundary.
  Nemotron checks its assigned layer range and SSM dimension products against
  the input, output, and convolution weights before expanding weights or
  allocating state. Unused SSM fields do not affect attention-only ranges.
  Stage profiles include architecture-generated positional tables and their
  construction overlap, including Metal's power-of-two allocation buckets.
  CPU construction overlap remains transient; accelerator construction overlap
  is conservatively charged as common device memory. Stage
  memory profiles require the model's exact layer count, apply the same shard
  check, and include dense CPU embedding expansion and its construction source.
  Qwen35 Q4K embeddings used on Metal stay packed on CPU; lookup transfers
  only selected F32 rows. The single-file, split-shard, and streaming loaders
  apply this policy. Profiles count the packed host table and read overlap;
  tied output heads also retain a separate packed Metal copy and its upload
  overlap. The lookup path does not expand the full embedding table to F32.
  Ordinary FP16/BF16 projections conservatively reserve the F32 fallback,
  including construction sources and Metal conversion scratch, even when a
  scoped CPU helper reads packed values directly. Floating tied output weights
  reserve a separate output allocation rather than shared embedding storage.
  Packed Q2_K Metal projections use the bounded eight-row launch in
  `src/qmatmul/metal_q2.rs`. It pads incomplete weight and output groups,
  checks tensor spans and device geometry, and preserves the existing shader
  math. Construction caps packed weights at 2 GiB; calls cap each input and
  output at 256 MiB. Memory profiles include the packed copy, readback and
  upload overlap, split-expert fallback, and separate tied-output storage.
  Floating expert weights
  generally reserve the F32 fallback even if an indexed kernel can retain
  packed F16. Qwen3 MoE BF16 experts are a scoped exception: the stage keeps
  packed host banks and caches only selected experts for one layer call.
  Admitted CPU BF16 projections retain selected packed weights and read their
  values in the checked dot helper; other CPU paths may use the F32 fallback.
  Metal expands selected BF16 values to F32 without loss. Stored banks keep
  their BF16 precision. Eligible Metal prefill batches
  one-row F32 matrix-vector calls with shared, zero-batch-stride weights, so it
  makes no weight copy per token. Decode uses the same F32 projection path.
  The per-call cache can retain experts used across all chunks. The existing
  F32 fallback reserve covers that cache, separately from grouped work's
  checked 128-token/256 MiB per-chunk scratch bound and input/output storage.
  Each chunk fences its work. After the final fence and cache drop, a second
  Metal drain removes pool-only weight buffers before a successful return.
  Both fences run before propagating calculation errors. Mixed or unaligned
  banks keep the scalar fallback. Qwen3 MoE's 128-wide Metal native-cache attention also
  has a checked 512 MiB F32 workspace cap, which its profile adds to the stage
  peak to account for possible Metal pool overlap. These memory bounds do not
  establish full-model numerical qualification.
  Qwen3 MoE, Phi3, and nonwindowed Llama-family CPU native-cache attention use
  fixed query tiles and checked mask extents so prompt chunk sizes do not
  change reduction shapes. The shared path admits 96- and 128-wide heads;
  Qwen3 MoE retains its 128-wide guard. It groups query heads by their KV head,
  keeps queries in F32, and expands F16 cache values without loss. Boolean
  causal selection excludes future scores even when their calculation
  overflows. Phi3 and nonwindowed Llama-family F32 projections use fixed
  eight-row calls; Qwen3 MoE keeps its one-row projections and router.
  Scoped Phi3, DeepSeek2 and Qwen3 MoE CPU paths also support stored Q8/K
  quantized weights; Qwen3 MoE has a separate packed BF16 path. Dispatch checks
  the actual stored dtype, CPU device, shape and native F16-cache mode before
  using a native-order helper. These checks do not change other architectures
  or windowed Llama math.
  Request-aware paths require the full prompt bounds in `TextRequestContext`.
  Their cache identity records the arithmetic kind, version and prompt-dependent
  policy. Reuse requires a compatible cached prefix; an incompatible prefix
  must be cleared and rebuilt with truthful reuse counts. CPU attention policies
  select the checked eight-part native reduction by absolute query position,
  prompt tail and padded key extent, not by the caller's chunk size. They cover
  the admitted Phi K, Deep Q8/K and Qwen K/BF16 paths with their own head shapes.
  Typed Qwen Q4 Metal requests may use the existing indexed vector kernels for
  actual Q4K gate/up and Q4K/Q6K down weights. The native QKV/cache and request
  guards, expert-ID checks, same-device checks and input/output extent bounds
  still apply. Raw calls without a request policy keep default dispatch, including ordinary
  Q6 matrix dispatch; FULL/Q8 Metal types do not enter this Q4 path. Raw/default
  APIs do not claim the request-aware qualification. Arithmetic and memory
  checks alone do not establish full-model numerical qualification.
  It loads only the requested stage range. Use `LayerRange::try_new` and
  `LayerRange::try_solo` for values crossing an untrusted API boundary.
- `src/deepseek4_gguf.rs` is the raw DeepSeek V4 path. It validates raw tensor
  dtype, shape, byte length, and file extent before retaining packed bytes for
  the architecture-specific kernels. V4 construction checks indexer and
  compressor widths and projection shapes before expanding those weights.
- `src/vision.rs` and `src/architectures/qwen35_vision.rs` validate projector
  metadata, image channel/layout/size constraints, and device placement before
  vision execution. Llama 4 and Gemma 4 bound projector layers, widths, head
  geometry and image budgets before loading weights; their normalization
  epsilon must remain finite and positive when represented in F32.
- `src/diffusion/weights.rs` bounds safetensors headers and checks tensor
  ranges, shapes, shard routing, duplicates and BF16/F32 precision before
  per-tensor reads. The research pipeline uses only local, operator-verified
  immutable snapshots, not remote code. This is not a signed artifact verifier
  or a memory-admission system; component residency may still exceed device RAM.
  VAE configuration validation also bounds the residual-block count before
  either encoder or decoder weights are accessed, and requires the five-level
  topology which implements the supported 16x spatial factor.
  The text encoder and image transformer require normalization epsilon to
  remain finite and positive in F32.

These checks establish structural validity, not authenticity or correctness of
the model. The engine does not verify signatures or provenance, and it does not
check that every byte of a file was actually downloaded. Verify shard hashes
and expected sizes before loading a model obtained from an untrusted source. A
structurally valid file can still contain adversarial weights, an incompatible
conversion, or zero-filled ranges left by an incomplete download.

## Accelerator boundary

Metal's residency set keeps a strong reference to each registered buffer.
Exact, non-pooled cache buffers must leave that set after their last owner and
queued GPU use end. The patched Candle buffer guard shares this lifetime across
clones; compute and blit completion handlers hold it through pending work.
Set changes use one lock because Metal's residency set methods are not thread
safe. See [Apple's residency set contract](https://developer.apple.com/documentation/metal/mtlresidencyset).

The portable path uses Candle operations from safe Rust. Backend-specific
operations are internal `CustomOp` implementations:

- Metal sources and wrappers are in `src/architectures/*_metal.rs` and
  `src/sampling_metal.rs`. They cover fused normalization/routing, QKV and
  attention layouts, DFlash/DeltaNet/KDA scans, indexed MoE, quantized matmul,
  ragged attention, and validated greedy token selection.
- CUDA sources and wrappers are in `src/architectures/*_cuda.rs`. They cover
  quantized matmul, indexed MoE epilogues, MLA/QKV layouts, DeltaNet/Mamba
  scans, residual normalization, and optional FlashAttention integration.
- CUDA FFI launches and device allocations use `unsafe` in the CUDA modules;
  the raw typed block copy used for compact K-quant dequantization is in
  `src/architectures/mod.rs`. The copy is preceded by block-alignment and
  shape/element-count checks, and the destination is Candle's `repr(C)` GGUF
  block type.

Before dispatch, these wrappers validate the invariants relevant to their
kernel geometry: dtype, contiguous layout, shape and element counts, shared
device identity, checked buffer offsets, expert-ID ranges, and launch/grid
limits. Unsupported layouts return an error or use the portable fallback; a
GGUF value cannot select an arbitrary kernel or supply kernel source. Review
the geometry validator beside any new kernel before changing its dispatch
conditions.

Metal pipeline caches and architecture-specific fallback decisions are keyed
by device identity. DFlash selection rejects non-finite logits on both portable
and Metal paths; Metal reads back a one-word validity flag before candidate IDs
can reach embedding lookup, not the full vocabulary.
The DFlash Metal selector falls back to the portable path when token IDs cannot
remain exact in its F32 result buffer or its launch and index ranges do not fit.
Ragged Metal attention requires a 32-thread SIMD width and a 1,024-thread group;
devices without those launch limits use the portable path.
GLM5 pooled-index sorting checks the compiled pipeline's thread limit before
choosing the one-thread-per-column path. Wider F32 rows use MLX's bounded-block
sort, with negated scores for descending order. The Candle bitonic wrapper also
rejects a group above its pipeline limit before encoding any work. Tests compare
wide pool indices with host sorting at both thread and merge boundaries.

Qwen3.5, Nemotron-H, and DeepSeek4 share a softplus helper that preserves small
positive results for negative inputs. These values affect recurrent decay and
routing weights. F16/BF16 inputs compute in F32 before casting back to avoid
intermediate overflow. Qwen and Nemotron Metal kernels use the same rounding
correction; independent scalar tests cover their decay paths. CUDA needs its
own backend test run.

GPU execution is asynchronous. Treat a failed command buffer, CUDA launch, or
device synchronization as an inference failure and restart or isolate the
host process according to its own recovery policy. Do not convert backend
errors into successful output or ignore an error merely because a later
operation completes. The vendored Candle Metal patch in `third_party/` improves
command-buffer error reporting; builds that bypass `[patch.crates-io]` lose it.

## Session cache lifetime and batch failures

Direct and continuous generation validate sampling parameters and logit-bias
vocabulary bounds before executing the model or mutating cache state.
Token inputs also check IDs and sequence lengths before cache eviction,
session detachment, or device lookup. A bad token request leaves the prior
cache intact; this avoids Metal lookup kernels silently clamping invalid IDs.
Sampling accumulates probabilities in F64 and never selects a zero-weight
candidate to make up a rounding gap. Tokenizer loading rejects invalid explicit
BOS/EOS IDs, non-boolean BOS settings, and malformed token-type, score, or merge
arrays before encoding. Present token-type and score arrays must match the
vocabulary length.

Tool constraints must match the model/logits vocabulary. A sampling history
retains one constraint and rejects later addition, removal, or replacement.
After a grammar completes, only its allowed EOS token can extend generation
to meet `min_tokens`; parser errors remain errors. The high-level `Engine`
checks sampling and context limits before clearing any existing stage cache.

Sessions own packed decode caches. The stage keeps only weak lookup entries;
dropping the last session releases the group's cache tensors. The device
allocator may keep freed buffers for reuse, so process/GPU memory need not
fall at once. Each live lane still pays the conservative full-backing-buffer
charge, including after regrouping or when other lanes leave.
V4 caches charge the whole backing storage, including Metal allocation padding,
CPU spare capacity, and storage retained by tensor views. Shared views within
one cache pay that charge once.
Prompt snapshots copy logits into exact-size persistent storage, so pooled
Metal padding cannot exceed the snapshot's byte charge.

Batch validation and admission failures leave caches intact and allow scalar
fallback. After model execution starts, a failure or wrong output count clears
every selected session and invalidates its packed group. The continuous
scheduler treats that as terminal, not as permission to decode without the
lost prompt. A caller that wants to retry must prefill again. Other sessions
keep their state. GPU device failures still need the recovery policy above.

Cache reset paths share one implementation for target/draft state, packed
ownership, positions, revisions, and memory accounting. Capacity growth and
alignment use checked arithmetic; an overflow returns an error before cache
allocation or mutation.
Full attention-window eviction keeps fixed DeltaNet and Mamba state. A text
stop which drops pending speculative output clears the affected session cache;
resuming that session rebuilds the whole visible prompt.
Positioned inputs record their rotary offset in the shared forward/prefill
path, including public stage and session APIs. Chunked input retains the largest
coordinate seen across all three axes. Direct and continuous generation keep
that offset when the next text turn resumes the cache. These sessions use
scalar inference rather than text-only decode batches, draft verification,
or shared prompt snapshots.
Session state also records external embeddings without rotary positions. That
flag survives explicit cache resume and checkpoints, so token IDs alone cannot
publish embedding-derived state into the shared text prompt cache.

DFlash falls back to scalar inference before drafting when a full verification
block would evict old context, exceed target or draft positions, or use more
rows than the remaining output limit. This preserves the context and random
draws of the next scalar token. If a target prompt outgrows the draft context,
the target can continue without the draft.

Draft sampling keeps random draws tentative until cache commit succeeds.
A failed draft injection or verification restores target and draft state;
tests cover sampling failure, commit failure, stop, and prompt reuse. This
state guarantee does not make a failed GPU command safe to retry; the device
failure rule above still applies.

## What to validate before deployment

1. Check model and shard hashes, file sizes, download completeness, and
   available host/device memory outside the engine.
2. Load the exact intended layer range and run a short deterministic prompt on
   the target backend. Compare against the CPU path or a trusted reference for
   representative prefill and single-token decode shapes.
3. Exercise the backend path that production will use. The engine reads no
   environment variables to choose kernels or fallbacks; a fallback runs only
   where the optimized kernel's device, shape or dtype checks reject the input.
   Runtime policy, such as speculative decoding, is set explicitly through
   `EngineOptions`.
4. Keep profiling output and verbose tracing private. They can contain
   prompts, token IDs, local paths and tensor metadata.

## Backend test gates

Run from the repository root:

```sh
# Portable baseline
cargo test --lib --locked
cargo test --test public_api --locked

# Metal: run on a host with a Metal device; absence is a failure, not a skip.
DINF_REQUIRE_METAL_TESTS=1 cargo test --lib --locked -- --nocapture

# CUDA: require device 0 on a CUDA host; otherwise the CUDA cases log one skip.
DINF_REQUIRE_CUDA_TESTS=1 cargo test --no-default-features --features cuda --lib --locked -- --nocapture
```

See [docs/VALIDATION.md](docs/VALIDATION.md) for full-model validation.

Metal correctness tests are located beside the relevant wrappers, including
`fused_norm_metal.rs`, `indexed_moe_metal.rs`, `qkv_layout_metal.rs`,
`tensor_qmatmul_metal.rs`, and `sampling_metal.rs`. CUDA geometry and reference tests are beside the
corresponding `*_cuda.rs` modules. `DINF_REQUIRE_CUDA_TESTS=1` makes a missing
or failed CUDA device-0 initialization fail the test run. Without it, the
helper logs one skip and a CUDA build without executed device tests is only a
compile check.
