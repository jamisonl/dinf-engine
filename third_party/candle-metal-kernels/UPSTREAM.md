# Vendored Candle Metal kernels

This directory is the published `candle-metal-kernels` 0.11.1 crate
(SHA-256 `6e9fdf9f3fb13fb677fec22802f85515e4bdf9a2b8fa153efb5454eced755932`,
upstream commit
[`59f8e90`](https://github.com/huggingface/candle/commit/59f8e903ca80a32ea7f8514b39748fc979e69e82))
with local safety fixes. Both Cargo manifests match the release. The root
`[patch.crates-io]` entry routes every Candle crate to this copy.

## License

Candle is dual-licensed `MIT OR Apache-2.0`. dinf-engine uses this vendored
copy, including its local modifications, under the **MIT** option. The
upstream MIT license text is included as `LICENSE-MIT`, and the crate's
`license` field is unchanged.

## Local patches

| Area | Files | Change | Upstream 0.11.1 |
| --- | --- | --- | --- |
| Concurrent waits | `src/metal/commands.rs`, `src/metal/command_buffer.rs` | Wait for a new queue marker even when another waiter took the in-flight list; keep fences for newer, unsubmitted writers. | `flush_and_wait` can return on an empty snapshot and clears every writer fence. |
| Readback | `src/metal/commands.rs` | `flush_and_wait_current` checks earlier buffers before CPU readback. | Only the newest buffer is checked. |
| Buffer lifetime and residency | `src/metal/buffer.rs`, `src/metal/residency_set.rs`, `src/metal/encoder.rs` | Remove private buffers from the residency set after their last owner drops; keep owners alive until queued work completes; serialize residency changes, including the batch APIs (iterators are consumed before the lock is taken). | Private buffers stay resident after their owners drop. |
| Causal attention bound | `src/metal_src/scaled_dot_product_attention.metal` | Bound the key-block limit by the actual KV block count. | Reads past the valid KV rows. |
| Two-pass vector attention partials | `src/metal_src/scaled_dot_product_attention.metal` | `sdpa_vector_2pass_1` stores its unnormalized partial outputs as float. Covered by the engine's `metal_native_sdpa_*` tests. | Casts them to the input type (F16/BF16) before the float store, so long-context partial sums can overflow or lose precision. |
| Sort launch guard | `src/kernels/sort.rs` | Reject bitonic sort groups above the pipeline's thread limit. | No guard. |
| Q8_0 mat-vec layout | `src/metal_src/quantized.metal`, `src/kernels/quantized.rs` | Dense `kernel_mul_mv_q8_0_f32` computes two rows per SIMD group with four groups per threadgroup, as llama.cpp does; each row's reduction order is unchanged. The indexed (`_id`) kernel keeps the shared implementation. | Four rows per SIMD group, two groups. |
| Head-dim 512 attention tile | `src/metal_src/scaled_dot_product_attention.metal`, `src/kernels/sdpa.rs` | F16 full attention at head dim 512 uses 16-query tiles over two SIMD groups; its Q tile is unpadded to fit 32 KB of threadgroup memory. BF16 keeps 8-query tiles. Covered by the engine's `metal_native_sdpa_*` tests. | Head dim 512 is not instantiated. |
| GPU timing counters | `src/metal/commands.rs` | Opt-in command-buffer timing (`GpuTimingSnapshot`), used by engine diagnostics. Disabled by default. | Not available. |
| Indirect dispatch and unordered outputs | `src/metal/encoder.rs` | `ComputeCommandEncoder::dispatch_thread_groups_indirect` reads threadgroup counts from a buffer that earlier work in the encoder may write, and orders the dispatch after those writers. `set_output_buffer_unordered` binds an output without ordering the dispatch after earlier ones that touch it, for writes the caller knows are disjoint; later readers are still ordered after it. | Not available. |
| Command buffer sizing | `src/metal/commands.rs` | Without `CANDLE_METAL_COMPUTE_PER_BUFFER`, the encoders per command buffer ramp from 8 to 512 (8, 24, 64, 128, 256, then 512) after each wait, so the GPU starts early and steady-state work uses few command buffers. The variable still sets a fixed limit. | A fixed 50 encoders per command buffer. |
| Batched residency commits | `src/metal/residency_set.rs`, `src/metal/buffer.rs`, `src/metal/commands.rs` | Once the set is attached to a queue, the batch add and remove APIs mark it dirty, and it is committed once before the next command buffer is submitted; removal when a buffer's last owner drops still commits at once. A buffer already in the set is not added again. | Every insert or removal commits the set. |
| Encoder hazard tracking | `src/metal/encoder.rs` | Per-barrier read and write sets are small vectors instead of hash sets, and the unused all-inputs set is gone. Barrier placement is unchanged. | Hash sets. |
| Encoder labels | `src/metal/encoder.rs` | `ComputeCommandEncoder::set_label` only labels the encoder with the `debug-labels` feature. | Always labels. |
| Tests | `src/tests.rs` | Regression tests for each fix above. | Upstream tests are kept unchanged. |

All other files, including `src/kernel.rs`, `reduce.metal` and `conv.metal`,
match the published source byte for byte, except for local additions such as
head-dim 512 attention support in `sdpa.rs` and its kernel instantiations. The post-wait error-reporting fix
([`091395a`](https://github.com/huggingface/candle/commit/091395a1dcc61641f57f55b0a3477435448808f2))
is already part of 0.11.1.

Neither the local nor the upstream command code makes a queue failure sticky
for later callers; the local drain returns on its first error. Callers should
treat Metal device failures as terminal.

## Testing

```sh
DINF_REQUIRE_METAL_TESTS=1 cargo test --release --locked \
  --manifest-path third_party/candle-metal-kernels/Cargo.toml --lib
```

The regression tests fail against an unmodified 0.11.1 with only the tests
added, and pass with the local fixes.

## Removal criteria

Remove this vendored copy and its `[patch.crates-io]` entry only when all of
the following hold:

1. A **published** `candle-metal-kernels` release contains each safety fix
   above (concurrent waits, readback, buffer lifetime and residency, attention
   bound, two-pass partials, sort guard). A change merged on `main` is not sufficient, because
   Cargo applies `[patch]` only at the workspace root and downstream consumers
   resolve the registry version.
2. The engine no longer calls local-only timing APIs, or uses an equivalent
   published API. A backend without GPU timestamps must report
   `supported: false` and zero counters rather than substituting host time.
3. The regression tests pass on Metal hardware against the published release.
4. The engine, its tests and examples build and pass in a clean consumer
   project with no path, git or patch overrides, and the resolved dependency
   graph routes every Candle crate to the fixed release.

A published fork is an alternative only if every Candle crate resolves to it;
renaming just the engine's direct dependency leaves `candle-core` on the
unfixed copy.
