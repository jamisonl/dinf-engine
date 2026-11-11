# dinf CudaForge patch

This directory vendors CudaForge 0.1.6 from crates.io under its declared
MIT-or-Apache-2.0 license. The root Cargo manifest patches Candle's transitive
dependency to this source.

dinf adds reproducible multi-architecture static CUDA builds through
`DINF_CUDA_ARCHS` and its lower-level alias `CUDA_COMPUTE_CAPS`. Every
configured architecture receives native cubin code, the oldest configured
virtual architecture is retained as PTX for forward compatibility, and the
complete architecture set participates in cache invalidation. Remove the root
patch when equivalent behavior is available in a reviewed upstream CudaForge
release.

Single-architecture static builds use a distinct cache key so upstream's
cubin-only objects cannot bypass compilation of the PTX fallback.
