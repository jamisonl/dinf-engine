# Vendored cudarc

This directory is `cudarc` 0.19.10 from crates.io (published crate checksum
`7359bf1de037ddada72729c3acec9b040f3468400c6e9e740b01ba0cd2e89562`), the
version Candle 0.11.1 depends on. It keeps cudarc's declared `MIT OR Apache-2.0`
license and the upstream license texts. The published crate's `examples/`
directory and its `[[example]]` manifest entries are omitted.

Local changes, for CUDA graph capture of a decode step and for the cost
of a decode step's many small launches:

- `src/driver/safe/capture_window.rs` (new): a per-thread capture window.
  While one is open for a stream, allocations come from a process-wide pool
  of `cuMemAlloc` blocks, dropped slices are held until the window ends, a
  host upload completes immediately on a helper stream into an idle block,
  and other host copies are errors. The pool keeps blocks released since
  the stream was last idle apart from idle ones, so a window can hold
  several captures while earlier ones still run. The module documents why
  each stock operation is wrong during capture.
- `src/driver/safe/core.rs`: `CudaStream::alloc`, `clone_htod`,
  `memcpy_stod`, `CudaSlice`'s `Drop`, `memcpy_htod` and `memcpy_dtoh`
  consult the window. Without an open window, allocations of up to 4 MiB
  that are not host uploads also come from the pool, because the stream
  allocator costs about a microsecond per allocation and free and a decode
  step makes hundreds. Uploads and larger allocations are unchanged; the
  module documents why uploads are left out. `memcpy_htod` into anything
  but the block just allocated for that upload drains the stream first.
  Pool blocks dropped outside a window return to the pool instead of the
  driver.
- `src/driver/safe/core.rs`: `CudaModule::load_function` keeps the handles
  it has looked up, since callers ask for one before every launch.
  `CudaStream`'s `Drop` drains the stream and returns its pool blocks to
  the driver.
- `src/driver/safe/mod.rs`: exports the module.

Pool memory is returned to the driver only when its stream is dropped.
`capture_window::pool_bytes` reports the pool's size.

Remove the workspace patch and this directory if cudarc gains a stream
capture mode that covers allocation, or if the engine stops capturing
graphs.
