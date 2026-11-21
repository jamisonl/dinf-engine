use anyhow::{Context, Result, ensure};

const Q4_0_BLOCK_SIZE: usize = 32;
const Q4_0_TYPE_SIZE: usize = 18;
/// Q5_0 adds four bytes of fifth bits between the scale and the nibbles.
#[cfg(feature = "cuda")]
const Q5_0_TYPE_SIZE: usize = 22;
/// Q4_K stores 256 elements in a 144-byte super-block.
#[cfg(feature = "cuda")]
const Q4_K_BLOCK_SIZE: usize = 256;
#[cfg(feature = "cuda")]
const Q4_K_TYPE_SIZE: usize = 144;
/// Q8_0 stores a scale and 32 signed bytes.
#[cfg(feature = "cuda")]
const Q8_0_TYPE_SIZE: usize = 34;
#[cfg(feature = "cuda")]
const CUDA_WARP_SIZE: usize = 32;
const CUDA_WARPS_PER_BLOCK: usize = 4;
const CUDA_PROMPT_ROUTE_TILE: usize = 8;
const CUDA_PROMPT_OUTPUT_TILE: usize = 32;
const CUDA_MAX_GRID_Y: usize = 65_535;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Q4IndexedGeometry {
    tokens: usize,
    input_rows: usize,
    top_k: usize,
    num_experts: usize,
    out_dim: usize,
    in_dim: usize,
    routes: usize,
    input_elements: usize,
    output_elements: usize,
    weight_bytes: usize,
    type_size: usize,
}

impl Q4IndexedGeometry {
    /// Geometry for Q4_0 expert banks.
    #[allow(clippy::too_many_arguments)]
    fn new(
        tokens: usize,
        input_rows: usize,
        in_dim: usize,
        id_tokens: usize,
        top_k: usize,
        num_experts: usize,
        out_dim: usize,
    ) -> Result<Self> {
        Self::with_type_size(
            Q4_0_TYPE_SIZE,
            Q4_0_BLOCK_SIZE,
            tokens,
            input_rows,
            in_dim,
            id_tokens,
            top_k,
            num_experts,
            out_dim,
        )
    }

    /// Geometry for a format with `type_size` bytes per `block_elements`.
    #[allow(clippy::too_many_arguments)]
    fn with_type_size(
        type_size: usize,
        block_elements: usize,
        tokens: usize,
        input_rows: usize,
        in_dim: usize,
        id_tokens: usize,
        top_k: usize,
        num_experts: usize,
        out_dim: usize,
    ) -> Result<Self> {
        ensure!(tokens > 0, "CUDA Q4_0 indexed MoE requires tokens");
        ensure!(top_k > 0, "CUDA Q4_0 indexed MoE requires routes");
        ensure!(
            tokens == id_tokens,
            "CUDA Q4_0 indexed MoE token counts differ"
        );
        ensure!(
            input_rows == 1 || input_rows == top_k,
            "CUDA Q4_0 indexed MoE input rows must be one or top-k"
        );
        ensure!(
            num_experts > 0 && top_k <= num_experts,
            "CUDA Q4_0 indexed MoE expert geometry differs"
        );
        ensure!(
            num_experts <= CUDA_MAX_GRID_Y,
            "CUDA Q4_0 indexed MoE expert count exceeds the CUDA grid limit"
        );
        ensure!(out_dim > 0, "CUDA Q4_0 indexed MoE requires output rows");
        ensure!(
            in_dim > 0
                && block_elements.is_multiple_of(Q4_0_BLOCK_SIZE)
                && in_dim.is_multiple_of(block_elements),
            "CUDA indexed MoE input width must be a multiple of {block_elements}"
        );

        let routes = checked_mul(tokens, top_k, "route count")?;
        ensure!(
            routes <= CUDA_MAX_GRID_Y,
            "CUDA Q4_0 indexed MoE route count exceeds the CUDA grid limit"
        );
        let input_elements = checked_mul(
            checked_mul(tokens, input_rows, "input rows")?,
            in_dim,
            "input elements",
        )?;
        let output_elements = checked_mul(routes, out_dim, "output elements")?;
        let blocks_per_row = in_dim / block_elements;
        let weight_bytes = checked_mul(
            checked_mul(
                checked_mul(num_experts, out_dim, "expert output rows")?,
                blocks_per_row,
                "expert Q4_0 blocks",
            )?,
            type_size,
            "expert storage bytes",
        )?;

        for (name, value) in [
            ("tokens", tokens),
            ("input rows", input_rows),
            ("top-k", top_k),
            ("expert count", num_experts),
            ("output width", out_dim),
            ("input width", in_dim),
        ] {
            ensure!(
                value <= i32::MAX as usize,
                "CUDA Q4_0 indexed MoE {name} exceeds the kernel index limit"
            );
        }
        ensure!(
            out_dim.div_ceil(CUDA_WARPS_PER_BLOCK) <= i32::MAX as usize,
            "CUDA Q4_0 indexed MoE output grid exceeds the CUDA limit"
        );

        Ok(Self {
            tokens,
            input_rows,
            top_k,
            num_experts,
            out_dim,
            in_dim,
            routes,
            input_elements,
            output_elements,
            weight_bytes,
            type_size,
        })
    }
}

fn checked_mul(lhs: usize, rhs: usize, label: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .with_context(|| format!("CUDA Q4_0 indexed MoE {label} overflows usize"))
}

/// Decoding of the expert formats Candle lacks, shared by the F32 and
/// tensor-core kernel sources: NVFP4 (GGML type 40, 64 elements in 36 bytes)
/// and IQ4_NL (type 20, Q4_0's block with a nonlinear value table).
#[cfg(feature = "cuda")]
macro_rules! raw_formats_source {
    () => {
        r#"
#define DINF_NVFP4 36
#define DINF_IQ4_NL 19
// In-memory MXFP8: 32 E4M3 weights in a 36-byte block after their UE8M0
// scale byte and three zero bytes.
#define DINF_MXFP8 37

// An E4M3 (FN) byte; the loader rejects the NaN codes.
__device__ __forceinline__ float dinf_e4m3(unsigned int x) {
    const unsigned int exponent = (x >> 3) & 15u;
    const unsigned int mantissa = x & 7u;
    const float magnitude = exponent == 0
        ? (float) mantissa * 0.001953125f
        : __uint_as_float(((exponent + 120u) << 23) | (mantissa << 20));
    return __uint_as_float(__float_as_uint(magnitude) | ((x & 0x80u) << 24));
}

// A UE8M0 scale byte, 2^(x - 127); 0 is the subnormal 2^-127.
__device__ __forceinline__ float dinf_e8m0(unsigned int x) {
    return x == 0 ? __uint_as_float(0x00400000u) : __uint_as_float(x << 23);
}

// ggml's kvalues_iq4nl.
__device__ __constant__ signed char dinf_iq4nl_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// E2M1 values, doubled as in ggml's kvalues_fp4; dinf_ue4m3 halves the scale.
__device__ __constant__ signed char dinf_fp4_values[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// ggml_ue4m3_to_fp32: an unsigned E4M3 scale, halved. 0 and 0x7f read as
// zero and the sign bit is ignored.
// The float is built from bits: (1 + m/8) 2^(e - 7) / 2 has the biased
// exponent e + 119, and the subnormal m 2^-9 / 2 is m 2^-10, both exact.
__device__ __forceinline__ float dinf_ue4m3(unsigned char x) {
    if (x == 0 || x == 0x7f) {
        return 0.0f;
    }
    const unsigned int exponent = (x >> 3) & 15;
    const unsigned int mantissa = x & 7;
    if (exponent == 0) {
        return (float) mantissa * 0.0009765625f;
    }
    return __uint_as_float(((exponent + 119u) << 23) | (mantissa << 20));
}

// Element `inner` (0..63) of the 36-byte NVFP4 block at `packed`: four UE4M3
// scales, one per 16 elements, then 32 bytes of E2M1 pairs. Byte j of a
// 16-element group holds element j in its low nibble and j + 8 in its high one.
__device__ __forceinline__ float dinf_nvfp4_weight(const unsigned char *packed, int inner) {
    const int sub = inner >> 4;
    const int j = inner & 15;
    const unsigned char quant = packed[4 + sub * 8 + (j & 7)];
    const int value = (j & 8) ? (quant >> 4) : (quant & 15);
    return dinf_ue4m3(packed[sub]) * (float) dinf_fp4_values[value];
}
"#
    };
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::{path::PathBuf, sync::OnceLock};

    use anyhow::Context as _;
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::WrapErr;
    use candle_core::cuda_backend::cudarc::driver::{
        CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
    };
    use candle_core::cuda_backend::cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{CpuStorage, CudaStorage, D, DType, Device, Layout, Shape, Storage, Tensor};

    use super::{
        CUDA_MAX_GRID_Y, CUDA_PROMPT_OUTPUT_TILE, CUDA_PROMPT_ROUTE_TILE, CUDA_WARP_SIZE,
        CUDA_WARPS_PER_BLOCK, Q4IndexedGeometry,
    };

    const CUDA_MODULE_NAME: &str = "dinf_indexed_moe_q4_0_v7";
    const RANK_SORT_KERNEL_NAME: &str = "dinf_indexed_moe_rank_sort";
    const RANK_SORT_THREADS: usize = 128;
    /// Routes up to which the quadratic rank sort replaces the general sort.
    const RANK_SORT_MAX_ROUTES: usize = 8192;

    const WMMA_MODULE_NAME: &str = "dinf_indexed_moe_wmma_v6";
    const WMMA_TILES_KERNEL_NAME: &str = "dinf_indexed_moe_wmma_tiles";
    const WMMA_TILES_THREADS: usize = 256;
    /// Routes one block multiplies against a weight tile it dequantizes once.
    const WMMA_ROUTE_TILE: usize = 128;
    const WMMA_GATHER_KERNEL_NAME: &str = "dinf_indexed_moe_gather_f16";
    const WMMA_GATHER_THREADS: usize = 256;
    const WMMA_TILE: usize = 32;
    const WMMA_WARPS: usize = 8;
    /// Output rows one block computes.
    const WMMA_OUTPUT_TILE: usize = 64;

    const MMA_MODULE_NAME: &str = "dinf_raw_moe_mma_v1";
    /// Output rows and threads of one `mma.sync` grouped block.
    const MMA_OUTPUT_TILE: usize = 128;
    const MMA_THREADS: usize = 256;

    /// Exact F32 decode and grouped prompt entry points.
    #[derive(Clone, Copy)]
    struct F32Kernels {
        decode: &'static str,
        prompt: &'static str,
    }

    /// Kernel entry points and block layout for one supported expert format.
    #[derive(Clone, Copy)]
    struct QuantKernels {
        f32: Option<F32Kernels>,
        /// Half-precision tensor-core grouped prompt kernel.
        wmma: Option<&'static str>,
        /// The `mma.sync` grouped prompt kernel (Ampere or newer), used in
        /// place of `wmma` where it loads.
        mma: Option<&'static str>,
        type_size: usize,
        block_elements: usize,
    }

    const Q4_0_KERNELS: QuantKernels = QuantKernels {
        f32: Some(F32Kernels {
            decode: "dinf_indexed_moe_q4_0_f32",
            prompt: "dinf_indexed_moe_q4_0_grouped_f32",
        }),
        wmma: None,
        mma: None,
        type_size: super::Q4_0_TYPE_SIZE,
        block_elements: super::Q4_0_BLOCK_SIZE,
    };
    const Q5_0_KERNELS: QuantKernels = QuantKernels {
        f32: Some(F32Kernels {
            decode: "dinf_indexed_moe_q5_0_f32",
            prompt: "dinf_indexed_moe_q5_0_grouped_f32",
        }),
        wmma: Some("dinf_indexed_moe_q5_0_grouped_wmma"),
        mma: None,
        type_size: super::Q5_0_TYPE_SIZE,
        block_elements: super::Q4_0_BLOCK_SIZE,
    };
    const Q4_K_KERNELS: QuantKernels = QuantKernels {
        f32: None,
        wmma: Some("dinf_indexed_moe_q4_k_grouped_wmma"),
        mma: None,
        type_size: super::Q4_K_TYPE_SIZE,
        block_elements: super::Q4_K_BLOCK_SIZE,
    };
    /// Q5_K stores 256 elements in a 176-byte super-block.
    const Q5_K_KERNELS: QuantKernels = QuantKernels {
        f32: None,
        wmma: Some("dinf_indexed_moe_q5_k_grouped_wmma"),
        mma: None,
        type_size: 176,
        block_elements: super::Q4_K_BLOCK_SIZE,
    };

    const Q8_0_KERNELS: QuantKernels = QuantKernels {
        f32: Some(F32Kernels {
            decode: "dinf_indexed_moe_q8_0_f32",
            prompt: "dinf_indexed_moe_q8_0_grouped_f32",
        }),
        wmma: Some("dinf_indexed_moe_q8_0_grouped_wmma"),
        mma: None,
        type_size: super::Q8_0_TYPE_SIZE,
        block_elements: super::Q4_0_BLOCK_SIZE,
    };

    /// NVFP4 stores 64 elements in 36 bytes; the geometry counts 18 bytes
    /// per 32-element block, as for Q4_0.
    const NVFP4_KERNELS: QuantKernels = QuantKernels {
        f32: Some(F32Kernels {
            decode: "dinf_indexed_moe_nvfp4_f32",
            prompt: "dinf_indexed_moe_nvfp4_grouped_f32",
        }),
        wmma: Some("dinf_indexed_moe_nvfp4_grouped_wmma"),
        mma: Some("dinf_raw_moe_nvfp4_grouped_mma"),
        type_size: super::Q4_0_TYPE_SIZE,
        block_elements: super::Q4_0_BLOCK_SIZE,
    };
    /// IQ4_NL has Q4_0's block layout.
    const IQ4_NL_KERNELS: QuantKernels = QuantKernels {
        f32: Some(F32Kernels {
            decode: "dinf_indexed_moe_iq4_nl_f32",
            prompt: "dinf_indexed_moe_iq4_nl_grouped_f32",
        }),
        wmma: Some("dinf_indexed_moe_iq4_nl_grouped_wmma"),
        mma: Some("dinf_raw_moe_iq4_nl_grouped_mma"),
        type_size: super::Q4_0_TYPE_SIZE,
        block_elements: super::Q4_0_BLOCK_SIZE,
    };

    /// In-memory MXFP8 (see `raw_experts.rs`): 36 bytes per 32 weights.
    const MXFP8_KERNELS: QuantKernels = QuantKernels {
        f32: Some(F32Kernels {
            decode: "dinf_indexed_moe_mxfp8_f32",
            prompt: "dinf_indexed_moe_mxfp8_grouped_f32",
        }),
        wmma: Some("dinf_indexed_moe_mxfp8_grouped_wmma"),
        mma: Some("dinf_raw_moe_mxfp8_grouped_mma"),
        type_size: 36,
        block_elements: super::Q4_0_BLOCK_SIZE,
    };

    /// Expert formats Candle lacks, kept in their GGUF bytes.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum RawExpertFormat {
        Nvfp4,
        Iq4Nl,
        Mxfp8,
    }

    impl RawExpertFormat {
        fn kernels(self) -> QuantKernels {
            match self {
                Self::Nvfp4 => NVFP4_KERNELS,
                Self::Iq4Nl => IQ4_NL_KERNELS,
                Self::Mxfp8 => MXFP8_KERNELS,
            }
        }

        /// Elements per GGUF block.
        fn block(self) -> usize {
            match self {
                Self::Nvfp4 => 64,
                Self::Iq4Nl | Self::Mxfp8 => 32,
            }
        }
    }

    /// The expert weights of one launch: a Candle quantized bank, or raw
    /// GGUF bytes (`[experts, out_dim, in_dim]`) of a format Candle lacks,
    /// with optional F32 per-expert scales that multiply each expert's output.
    #[derive(Clone, Copy)]
    pub(crate) enum ExpertBank<'a> {
        Packed(&'a QTensor),
        Raw {
            format: RawExpertFormat,
            bytes: &'a Tensor,
            scales: Option<&'a Tensor>,
            dims: [usize; 3],
        },
    }

    impl ExpertBank<'_> {
        fn kernels(&self) -> Option<QuantKernels> {
            match self {
                Self::Packed(weights) => quant_kernels(weights.dtype()),
                Self::Raw { format, .. } => Some(format.kernels()),
            }
        }

        fn dims(&self) -> Option<[usize; 3]> {
            match self {
                Self::Packed(weights) => match weights.shape().dims() {
                    &[experts, out_dim, in_dim] => Some([experts, out_dim, in_dim]),
                    _ => None,
                },
                Self::Raw { dims, .. } => Some(*dims),
            }
        }

        fn byte_len(&self) -> usize {
            match self {
                Self::Packed(weights) => weights.storage_size_in_bytes(),
                Self::Raw { bytes, .. } => bytes.elem_count(),
            }
        }

        fn is_on(&self, device: &candle_core::CudaDevice) -> bool {
            let device = Device::Cuda(device.clone());
            match self {
                Self::Packed(weights) => weights.device().same_device(&device),
                Self::Raw { bytes, scales, .. } => {
                    bytes.device().same_device(&device)
                        && scales.is_none_or(|scales| scales.device().same_device(&device))
                }
            }
        }

        /// Run `launch` with the device addresses of the weights and of the
        /// per-expert scales (zero without scales), both held for the launch.
        fn with_addresses<R>(
            &self,
            stream: &CudaStream,
            launch: impl FnOnce(u64, u64) -> candle_core::Result<R>,
        ) -> candle_core::Result<R> {
            match self {
                Self::Packed(weights) => {
                    let (pointer, _guard) = weights.device_ptr_with_guard(stream)?;
                    launch(pointer as u64, 0)
                }
                Self::Raw {
                    bytes,
                    scales,
                    dims,
                    ..
                } => {
                    let (storage, layout) = bytes.storage_and_layout();
                    let Storage::Cuda(storage) = &*storage else {
                        candle_core::bail!("raw expert bytes are not on CUDA")
                    };
                    if bytes.dtype() != DType::U8
                        || !layout.is_contiguous()
                        || layout.start_offset() != 0
                    {
                        candle_core::bail!("raw expert bytes must be a whole contiguous u8 tensor")
                    }
                    let (pointer, _guard) = storage.as_cuda_slice::<u8>()?.device_ptr(stream);
                    let Some(scales) = scales else {
                        return launch(pointer, 0);
                    };
                    let (scale_storage, scale_layout) = scales.storage_and_layout();
                    let Storage::Cuda(scale_storage) = &*scale_storage else {
                        candle_core::bail!("expert scales are not on CUDA")
                    };
                    if scales.dtype() != DType::F32
                        || !scale_layout.is_contiguous()
                        || scale_layout.shape().elem_count() != dims[0]
                    {
                        candle_core::bail!("expert scales must be contiguous f32, one per expert")
                    }
                    let scale_view = checked_view(
                        scale_storage.as_cuda_slice::<f32>()?,
                        scale_layout,
                        dims[0],
                        "expert scales",
                    )?;
                    let (scale_pointer, _scale_guard) = scale_view.device_ptr(stream);
                    launch(pointer, scale_pointer)
                }
            }
        }
    }

    fn quant_kernels(dtype: GgmlDType) -> Option<QuantKernels> {
        match dtype {
            GgmlDType::Q4_0 => Some(Q4_0_KERNELS),
            GgmlDType::Q5_0 => Some(Q5_0_KERNELS),
            GgmlDType::Q8_0 => Some(Q8_0_KERNELS),
            GgmlDType::Q4K => Some(Q4_K_KERNELS),
            GgmlDType::Q5K => Some(Q5_K_KERNELS),
            _ => None,
        }
    }

    /// Whether the exact F32 decode and grouped prompt kernels are the
    /// default CUDA path for `dtype`. Q8_0 has them too, but other models keep
    /// Candle's Q8_0 path unless they opt in through
    /// [`supports_f32_decode`].
    pub(crate) fn supports(dtype: GgmlDType) -> bool {
        matches!(dtype, GgmlDType::Q4_0 | GgmlDType::Q5_0)
    }

    /// Whether [`forward`] can run a single token for `dtype` in one launch.
    pub(crate) fn supports_f32_decode(dtype: GgmlDType) -> bool {
        quant_kernels(dtype).is_some_and(|kernels| kernels.f32.is_some())
    }

    /// Whether the half-precision tensor-core grouped prompt kernel covers
    /// `dtype` and compiled for this toolkit.
    pub(crate) fn supports_grouped_wmma(dtype: GgmlDType) -> bool {
        quant_kernels(dtype).is_some_and(|kernels| kernels.wmma.is_some()) && wmma_ptx().is_ok()
    }

    // A first kernel gathers every route's input row, in sorted route order,
    // into one F16 buffer. A second cuts each expert's run of sorted routes
    // into tiles of at most 128 routes. Then one block owns a tile and 64
    // output rows: for each 32-element step its threads fill a half-precision
    // 64x32 weight tile once, and each of the eight warps takes 16 routes of
    // the tile, loads their inputs once from the gathered buffer and
    // accumulates the product with each 16 rows of the weight tile on tensor
    // cores. Prompt routing is skewed (one expert can take a tenth of the
    // routes), so the tile table spreads a busy expert over several blocks
    // and the shared weight tile saves it most of the dequantization. Inputs
    // and dequantized weights are rounded to F16; sums accumulate in F32.
    const WMMA_SOURCE: &str = concat!(
        r#"
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define DINF_T 32
#define DINF_OUT 64
#define DINF_OUT_CHUNKS (DINF_OUT / 16)
#define DINF_THREADS 256
#define DINF_ROUTE_TILE 128
#define DINF_TABLE_EXPERTS 1024
#define DINF_NO_ROUTE 0xffffffffu
"#,
        raw_formats_source!(),
        r#"

__device__ __forceinline__ int dinf_wmma_lower_bound(
    const unsigned int *sorted_experts, int routes, unsigned int expert) {
    int low = 0;
    int high = routes;
    while (low < high) {
        const int middle = low + (high - low) / 2;
        if (sorted_experts[middle] < expert) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low;
}

__device__ __forceinline__ int dinf_wmma_upper_bound(
    const unsigned int *sorted_experts, int routes, unsigned int expert) {
    int low = 0;
    int high = routes;
    while (low < high) {
        const int middle = low + (high - low) / 2;
        if (sorted_experts[middle] <= expert) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low;
}

// Eight dequantized weights starting at `inner0` (a multiple of eight) of
// 32-element step `step` in one expert row, rounded to F16. FORMAT is the
// GGUF block byte size: 22 is Q5_0, 34 is Q8_0, 144 is Q4_K, 176 is Q5_K;
// DINF_NVFP4 is
// NVFP4, whose values F16 holds exactly, and DINF_IQ4_NL is IQ4_NL. One
// header decode serves the run.
template <int FORMAT>
__device__ __forceinline__ void dinf_tile_weights8(
    const unsigned char *weights,
    unsigned long long expert_row,
    int step,
    int in_dim,
    int inner0,
    __half *out) {
    if (FORMAT == DINF_NVFP4) {
        const unsigned char *packed = weights
            + (expert_row * (unsigned long long) (in_dim / 64)
                + (unsigned long long) (step >> 1)) * 36ull;
        const int sub = ((step & 1) << 1) | (inner0 >> 4);
        const float d = dinf_ue4m3(packed[sub]);
        const unsigned char *quants = packed + 4 + sub * 8;
        const int shift = (inner0 & 8) ? 4 : 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            out[i] = __float2half(d * (float) dinf_fp4_values[(quants[i] >> shift) & 15]);
        }
        return;
    }
    if (FORMAT == DINF_MXFP8) {
        const unsigned char *packed = weights
            + (expert_row * (unsigned long long) (in_dim / 32)
                + (unsigned long long) step) * 36ull;
        const float d = dinf_e8m0(packed[0]);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            out[i] = __float2half(dinf_e4m3(packed[4 + inner0 + i]) * d);
        }
        return;
    }
    if (FORMAT == DINF_IQ4_NL) {
        const unsigned char *packed = weights
            + (expert_row * (unsigned long long) (in_dim / 32)
                + (unsigned long long) step) * 18ull;
        const float d = __half2float(*reinterpret_cast<const __half *>(packed));
        const unsigned char *quants = packed + 2 + (inner0 & 15);
        const int shift = inner0 < 16 ? 0 : 4;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            out[i] = __float2half(d * (float) dinf_iq4nl_values[(quants[i] >> shift) & 15]);
        }
        return;
    }
    if (FORMAT == 176) {
        // Q5_K: Q4_K's scales and nibbles (after 32 bytes of fifth bits), with
        // bit `sub` of qh[l] as the fifth bit of weight l in sub-block `sub`.
        const unsigned char *packed = weights
            + (expert_row * (unsigned long long) (in_dim / 256)
                + (unsigned long long) (step >> 3)) * 176ull;
        const int sub = step & 7;
        const float d = __half2float(*reinterpret_cast<const __half *>(packed));
        const float dmin = __half2float(*reinterpret_cast<const __half *>(packed + 2));
        const unsigned char *scales = packed + 4;
        int scale;
        int minimum;
        if (sub < 4) {
            scale = scales[sub] & 63;
            minimum = scales[sub + 4] & 63;
        } else {
            scale = (scales[sub + 4] & 0xF) | ((scales[sub - 4] >> 6) << 4);
            minimum = (scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4);
        }
        const float step_scale = d * (float) scale;
        const float step_min = dmin * (float) minimum;
        const unsigned char *high = packed + 16 + inner0;
        const unsigned char *quants = packed + 48 + (sub >> 1) * 32 + inner0;
        const int shift = (sub & 1) * 4;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int value = ((quants[i] >> shift) & 15) | (((high[i] >> sub) & 1) << 4);
            out[i] = __float2half(step_scale * (float) value - step_min);
        }
        return;
    }
    if (FORMAT == 144) {
        const unsigned char *packed = weights
            + (expert_row * (unsigned long long) (in_dim / 256)
                + (unsigned long long) (step >> 3)) * 144ull;
        const int sub = step & 7;
        const float d = __half2float(*reinterpret_cast<const __half *>(packed));
        const float dmin = __half2float(*reinterpret_cast<const __half *>(packed + 2));
        const unsigned char *scales = packed + 4;
        int scale;
        int minimum;
        if (sub < 4) {
            scale = scales[sub] & 63;
            minimum = scales[sub + 4] & 63;
        } else {
            scale = (scales[sub + 4] & 0xF) | ((scales[sub - 4] >> 6) << 4);
            minimum = (scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4);
        }
        const float step_scale = d * (float) scale;
        const float step_min = dmin * (float) minimum;
        const unsigned char *quants = packed + 16 + (sub >> 1) * 32 + inner0;
        const int shift = (sub & 1) * 4;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int value = (quants[i] >> shift) & 15;
            out[i] = __float2half(step_scale * (float) value - step_min);
        }
        return;
    }
    const unsigned char *packed = weights
        + (expert_row * (unsigned long long) (in_dim / 32)
            + (unsigned long long) step) * (unsigned long long) FORMAT;
    const float scale = __half2float(*reinterpret_cast<const __half *>(packed));
    if (FORMAT == 34) {
        const signed char *quants = reinterpret_cast<const signed char *>(packed) + 2 + inner0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            out[i] = __float2half(scale * (float) quants[i]);
        }
        return;
    }
    const unsigned int high = (unsigned int) packed[2]
        | ((unsigned int) packed[3] << 8)
        | ((unsigned int) packed[4] << 16)
        | ((unsigned int) packed[5] << 24);
    const unsigned char *quants = packed + 6 + (inner0 & 15);
    const int shift = inner0 < 16 ? 0 : 4;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int low = (quants[i] >> shift) & 15;
        const int value = (low | (int) (((high >> (inner0 + i)) & 1u) << 4)) - 16;
        out[i] = __float2half(scale * (float) value);
    }
}

// Row `s` of `gathered` is the input row of sorted route `s`, rounded to F16.
// Rows at or past `routes` are zero padding so a route tile may read past
// the last route.
extern "C" __global__ void dinf_indexed_moe_gather_f16(
    const float *input,
    const unsigned int *sorted_route_ids,
    __half *gathered,
    long long routes,
    long long padded_routes,
    long long top_k,
    long long input_rows,
    long long in_dim,
    int activation_kind) {
    const long long index = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= padded_routes * in_dim) {
        return;
    }
    const long long sorted = index / in_dim;
    const long long column = index - sorted * in_dim;
    float value = 0.0f;
    if (sorted < routes) {
        const long long route = sorted_route_ids[sorted];
        if (route < routes) {
            const long long row = input_rows == 1 ? route / top_k : route;
            value = input[row * in_dim + column];
            if (activation_kind == 1) {
                value = fmaxf(value, 0.0f);
                value *= value;
            }
        }
    }
    gathered[index] = __float2half(value);
}

// One block. Writes `max_tiles` triples `[expert, first sorted route, routes]`:
// each expert's run of sorted routes cut into tiles of at most
// DINF_ROUTE_TILE, then empty tiles. `max_tiles` is at least
// `num_experts + routes / DINF_ROUTE_TILE`, which bounds the tile count.
extern "C" __global__ void dinf_indexed_moe_wmma_tiles(
    const unsigned int *sorted_experts,
    unsigned int *tiles,
    int routes,
    int num_experts,
    int max_tiles) {
    __shared__ int starts[DINF_TABLE_EXPERTS];
    __shared__ int counts[DINF_TABLE_EXPERTS];
    const int thread = threadIdx.x;
    int tile = 0;
    for (int base = 0; base < num_experts; base += DINF_TABLE_EXPERTS) {
        const int limit = min(num_experts - base, DINF_TABLE_EXPERTS);
        for (int slot = thread; slot < limit; slot += blockDim.x) {
            const unsigned int expert = (unsigned int) (base + slot);
            const int start = dinf_wmma_lower_bound(sorted_experts, routes, expert);
            starts[slot] = start;
            counts[slot] = dinf_wmma_upper_bound(sorted_experts, routes, expert) - start;
        }
        __syncthreads();
        if (thread == 0) {
            for (int slot = 0; slot < limit; ++slot) {
                for (int offset = 0; offset < counts[slot] && tile < max_tiles;
                     offset += DINF_ROUTE_TILE) {
                    tiles[3 * tile] = (unsigned int) (base + slot);
                    tiles[3 * tile + 1] = (unsigned int) (starts[slot] + offset);
                    tiles[3 * tile + 2] =
                        (unsigned int) min(counts[slot] - offset, DINF_ROUTE_TILE);
                    ++tile;
                }
            }
        }
        __syncthreads();
    }
    if (thread == 0) {
        for (; tile < max_tiles; ++tile) {
            tiles[3 * tile] = 0;
            tiles[3 * tile + 1] = 0;
            tiles[3 * tile + 2] = 0;
        }
    }
}

template <int FORMAT>
__device__ __forceinline__ void dinf_indexed_moe_grouped_wmma(
    const __half *gathered,
    const unsigned int *sorted_route_ids,
    const unsigned int *tiles,
    const unsigned char *weights,
    float *output,
    int routes,
    int in_dim,
    int out_dim,
    int num_experts,
    const float *expert_scales) {
    const int warp = threadIdx.y;
    const int thread = warp * 32 + threadIdx.x;
    const int first_row = blockIdx.x * DINF_OUT;
    const unsigned int *tile = tiles + 3ull * blockIdx.y;
    const int expert = (int) tile[0];
    const int segment_start = (int) tile[1];
    const int tile_routes = (int) tile[2];
    if (tile_routes <= 0 || tile_routes > DINF_ROUTE_TILE || expert >= num_experts
        || segment_start < 0 || segment_start > routes - tile_routes) {
        return;
    }
    const int segment_end = segment_start + tile_routes;

    __shared__ unsigned int tile_route[DINF_ROUTE_TILE];
    // Tensor-core loads and stores need 32-byte aligned tiles.
    // [output row][inner], read by tensor cores as a column-major K x N tile.
    __shared__ __align__(32) __half weight_tile[DINF_OUT][DINF_T];
    // One 16x16 result tile per warp, written and scattered by that warp.
    __shared__ __align__(32) float output_tile[DINF_THREADS / 32][16][16];

    const int steps = in_dim / DINF_T;
    const unsigned long long expert_base = (unsigned long long) expert * out_dim;
    // This warp's 16 routes of the tile; warps past the tile only help fill
    // the weight tile.
    const int tile_m = warp * 16;
    const bool active = tile_m < tile_routes;

    if (thread < DINF_ROUTE_TILE) {
        const int sorted_index = segment_start + thread;
        unsigned int route = DINF_NO_ROUTE;
        if (sorted_index < segment_end) {
            const unsigned int candidate = sorted_route_ids[sorted_index];
            if (candidate < (unsigned int) routes) {
                route = candidate;
            }
        }
        tile_route[thread] = route;
    }
    __syncthreads();
    {
        // Rows past the tile belong to other experts or the padding; their
        // products are computed and discarded.
        const __half *tile_input = gathered
            + (unsigned long long) (segment_start + tile_m) * (unsigned long long) in_dim;

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> sum[DINF_OUT_CHUNKS];
#pragma unroll
        for (int chunk = 0; chunk < DINF_OUT_CHUNKS; ++chunk) {
            wmma::fill_fragment(sum[chunk], 0.0f);
        }

        for (int step = 0; step < steps; ++step) {
            {
                // Four threads share a tile row; each fills eight neighbours.
                const int row = thread >> 2;
                const int inner0 = (thread & 3) * 8;
                const int output_row = first_row + row;
                if (output_row < out_dim) {
                    dinf_tile_weights8<FORMAT>(
                        weights, expert_base + output_row, step, in_dim, inner0,
                        &weight_tile[row][inner0]);
                } else {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        weight_tile[row][inner0 + i] = __float2half(0.0f);
                    }
                }
            }
            __syncthreads();

            if (active) {
#pragma unroll
                for (int k = 0; k < DINF_T; k += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                    wmma::load_matrix_sync(a, tile_input + step * DINF_T + k, in_dim);
#pragma unroll
                    for (int chunk = 0; chunk < DINF_OUT_CHUNKS; ++chunk) {
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                        wmma::load_matrix_sync(b, &weight_tile[chunk * 16][k], DINF_T);
                        wmma::mma_sync(sum[chunk], a, b, sum[chunk]);
                    }
                }
            }
            __syncthreads();
        }

        if (active) {
            const int lane = threadIdx.x;
            const float expert_scale = expert_scales != nullptr ? expert_scales[expert] : 1.0f;
#pragma unroll
            for (int chunk = 0; chunk < DINF_OUT_CHUNKS; ++chunk) {
                wmma::store_matrix_sync(
                    &output_tile[warp][0][0], sum[chunk], 16, wmma::mem_row_major);
                __syncwarp();
                for (int item = lane; item < 256; item += 32) {
                    const int row = item >> 4;
                    const int column = item & 15;
                    const unsigned int route = tile_route[tile_m + row];
                    const int output_row = first_row + chunk * 16 + column;
                    if (route != DINF_NO_ROUTE && output_row < out_dim) {
                        output[(unsigned long long) route * out_dim + output_row] =
                            output_tile[warp][row][column] * expert_scale;
                    }
                }
                __syncwarp();
            }
        }
    }
}

#define DINF_WMMA_ENTRY(NAME, FORMAT) \
extern "C" __global__ void NAME( \
    const __half *gathered, const unsigned int *sorted_route_ids, \
    const unsigned int *tiles, const unsigned char *weights, float *output, \
    int routes, int in_dim, int out_dim, int num_experts, const float *expert_scales) { \
    dinf_indexed_moe_grouped_wmma<FORMAT>(gathered, sorted_route_ids, tiles, \
        weights, output, routes, in_dim, out_dim, num_experts, expert_scales); \
}

DINF_WMMA_ENTRY(dinf_indexed_moe_q5_0_grouped_wmma, 22)
DINF_WMMA_ENTRY(dinf_indexed_moe_q8_0_grouped_wmma, 34)
DINF_WMMA_ENTRY(dinf_indexed_moe_q4_k_grouped_wmma, 144)
DINF_WMMA_ENTRY(dinf_indexed_moe_q5_k_grouped_wmma, 176)
DINF_WMMA_ENTRY(dinf_indexed_moe_nvfp4_grouped_wmma, DINF_NVFP4)
DINF_WMMA_ENTRY(dinf_indexed_moe_iq4_nl_grouped_wmma, DINF_IQ4_NL)
DINF_WMMA_ENTRY(dinf_indexed_moe_mxfp8_grouped_wmma, DINF_MXFP8)
"#
    );

    // Each warp computes one output row. Four warps share each input tile,
    // while Q4_0 weights stay in their canonical 18-byte GGUF blocks.
    const CUDA_SOURCE: &str = concat!(
        r#"
#include <cuda_fp16.h>

#define DINF_Q4_BLOCK_SIZE 32
#define DINF_WARP_SIZE 32
"#,
        raw_formats_source!(),
        r#"

// Bytes of one 32-element block, and of an expert row of `blocks` of them.
template <int TYPE_SIZE>
__device__ __forceinline__ unsigned long long dinf_block_bytes() {
    return TYPE_SIZE == DINF_NVFP4 || TYPE_SIZE == DINF_IQ4_NL ? 18ull
        : TYPE_SIZE == DINF_MXFP8 ? 36ull
        : TYPE_SIZE;
}

template <int TYPE_SIZE>
__device__ __forceinline__ unsigned long long dinf_row_bytes(int blocks) {
    return (unsigned long long) blocks * dinf_block_bytes<TYPE_SIZE>();
}

// Start of the GGUF block holding 32-element block `block` of `row`. An NVFP4
// block holds two of them.
template <int TYPE_SIZE>
__device__ __forceinline__ const unsigned char *dinf_block_ptr(
    const unsigned char *row, int block) {
    if (TYPE_SIZE == DINF_NVFP4) {
        return row + (unsigned long long) (block >> 1) * 36ull;
    }
    return row + (unsigned long long) block * dinf_block_bytes<TYPE_SIZE>();
}

// One dequantized weight of 32-element block `block`, whose GGUF block starts
// at `packed`. TYPE_SIZE 18 is Q4_0 (scale, 16 nibble bytes); 22 is Q5_0
// (scale, 32 fifth bits, 16 nibble bytes); 34 is Q8_0 (scale, 32 signed
// bytes); DINF_NVFP4 is NVFP4 (see dinf_nvfp4_weight); DINF_IQ4_NL is
// IQ4_NL (Q4_0's layout through dinf_iq4nl_values).
template <int TYPE_SIZE>
__device__ __forceinline__ float dinf_block_weight(
    const unsigned char *packed, int block, int inner) {
    if (TYPE_SIZE == DINF_NVFP4) {
        return dinf_nvfp4_weight(packed, ((block & 1) << 5) | inner);
    }
    if (TYPE_SIZE == DINF_MXFP8) {
        return dinf_e4m3(packed[4 + inner]) * dinf_e8m0(packed[0]);
    }
    if (TYPE_SIZE == DINF_IQ4_NL) {
        const unsigned char quant = packed[2 + (inner & 15)];
        const int value = inner < 16 ? (quant & 15) : (quant >> 4);
        return __half2float(*reinterpret_cast<const __half *>(packed))
            * (float) dinf_iq4nl_values[value];
    }
    const float scale = __half2float(*reinterpret_cast<const __half *>(packed));
    if (TYPE_SIZE == 34) {
        return scale * (float) reinterpret_cast<const signed char *>(packed)[2 + inner];
    }
    if (TYPE_SIZE == 22) {
        const unsigned int high = (unsigned int) packed[2]
            | ((unsigned int) packed[3] << 8)
            | ((unsigned int) packed[4] << 16)
            | ((unsigned int) packed[5] << 24);
        const unsigned char quant = packed[6 + (inner & 15)];
        const int low = inner < 16 ? (quant & 15) : (quant >> 4);
        const int value = (low | (int) (((high >> inner) & 1u) << 4)) - 16;
        return scale * (float) value;
    }
    const unsigned char quant = packed[2 + (inner & 15)];
    const int value = inner < 16 ? (quant & 15) - 8 : (quant >> 4) - 8;
    return scale * (float) value;
}

template <int TYPE_SIZE>
__device__ __forceinline__ void dinf_indexed_moe_decode(
    const float *input,
    const unsigned int *expert_ids,
    const unsigned char *weights,
    float *output,
    int routes,
    int top_k,
    int input_rows,
    int in_dim,
    int out_dim,
    int num_experts,
    int activation_kind,
    const float *expert_scales) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int route = blockIdx.y;
    const int output_row = blockIdx.x * blockDim.y + warp;
    if (route >= routes) {
        return;
    }

    const unsigned int expert = expert_ids[route];
    const bool valid_output = output_row < out_dim;
    if (expert >= (unsigned int) num_experts) {
        // The output is not cleared beforehand; an unknown expert adds nothing.
        if (lane == 0 && valid_output) {
            output[(unsigned long long) route * out_dim + output_row] = 0.0f;
        }
        return;
    }
    const int input_row = input_rows == 1 ? route / top_k : route;
    const float *input_row_ptr = input + (unsigned long long) input_row * in_dim;
    const int blocks_per_row = in_dim / DINF_Q4_BLOCK_SIZE;
    const int safe_output_row = valid_output ? output_row : 0;
    const unsigned long long expert_row =
        (unsigned long long) expert * out_dim + safe_output_row;
    const unsigned char *weight_row = weights
        + expert_row * dinf_row_bytes<TYPE_SIZE>(blocks_per_row);

    // Each lane reads its own input column of every block; the row is one
    // token's activation and stays in cache across the block's warps. A
    // shared tile would cost two block barriers per 32 columns.
    float sum = 0.0f;
    if (valid_output) {
        for (int block = 0; block < blocks_per_row; ++block) {
            float value = input_row_ptr[block * DINF_Q4_BLOCK_SIZE + lane];
            if (activation_kind == 1) {
                value = fmaxf(value, 0.0f);
                value *= value;
            }
            const unsigned char *packed = dinf_block_ptr<TYPE_SIZE>(weight_row, block);
            sum = fmaf(value, dinf_block_weight<TYPE_SIZE>(packed, block, lane), sum);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0 && valid_output) {
        if (expert_scales != nullptr) {
            sum *= expert_scales[expert];
        }
        output[(unsigned long long) route * out_dim + output_row] = sum;
    }
}

#define DINF_PROMPT_M 8
#define DINF_PROMPT_N 32

__device__ __forceinline__ int dinf_lower_bound_expert(
    const unsigned int *sorted_experts,
    int routes,
    unsigned int expert) {
    int low = 0;
    int high = routes;
    while (low < high) {
        const int middle = low + (high - low) / 2;
        if (sorted_experts[middle] < expert) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low;
}

__device__ __forceinline__ int dinf_upper_bound_expert(
    const unsigned int *sorted_experts,
    int routes,
    unsigned int expert) {
    int low = 0;
    int high = routes;
    while (low < high) {
        const int middle = low + (high - low) / 2;
        if (sorted_experts[middle] <= expert) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low;
}

// Prompt routes are sorted by expert before this launch. One block owns an
// expert, 32 output rows, and eight routes. It expands each 32x32 Q4_0 weight
// tile once, then all eight route warps reuse that tile.
template <int TYPE_SIZE>
__device__ __forceinline__ void dinf_indexed_moe_grouped(
    const float *input,
    const unsigned int *sorted_route_ids,
    const unsigned int *sorted_experts,
    const unsigned char *weights,
    float *output,
    int routes,
    int top_k,
    int input_rows,
    int in_dim,
    int out_dim,
    int num_experts,
    int activation_kind,
    const float *expert_scales) {
    const int lane = threadIdx.x;
    const int route_lane = threadIdx.y;
    const int expert = blockIdx.y;
    const int output_row = blockIdx.x * DINF_PROMPT_N + lane;

    if (expert >= num_experts) {
        return;
    }

    __shared__ int segment_start;
    __shared__ int segment_end;
    __shared__ float input_tile[DINF_PROMPT_M][DINF_Q4_BLOCK_SIZE];
    // Padding keeps lanes on separate banks while they read one K column.
    __shared__ float weight_tile[DINF_PROMPT_N][DINF_Q4_BLOCK_SIZE + 1];

    if (lane == 0 && route_lane == 0) {
        segment_start = dinf_lower_bound_expert(
            sorted_experts, routes, (unsigned int) expert);
        segment_end = dinf_upper_bound_expert(
            sorted_experts, routes, (unsigned int) expert);
    }
    __syncthreads();

    const bool valid_output = output_row < out_dim;
    const int blocks_per_row = in_dim / DINF_Q4_BLOCK_SIZE;

    for (int route_tile_start = segment_start;
         route_tile_start < segment_end;
         route_tile_start += DINF_PROMPT_M) {
        const int sorted_index = route_tile_start + route_lane;
        const bool active_route = sorted_index < segment_end;
        const unsigned int candidate_route =
            active_route ? sorted_route_ids[sorted_index] : 0;
        const bool valid_route =
            active_route && candidate_route < (unsigned int) routes;
        const unsigned int route = valid_route ? candidate_route : 0;
        const int input_row = input_rows == 1 ? route / top_k : route;
        float sum = 0.0f;

        for (int block = 0; block < blocks_per_row; ++block) {
            float input_value = valid_route
                ? input[(unsigned long long) input_row * in_dim
                    + block * DINF_Q4_BLOCK_SIZE + lane]
                : 0.0f;
            if (activation_kind == 1) {
                input_value = fmaxf(input_value, 0.0f);
                input_value *= input_value;
            }
            input_tile[route_lane][lane] = input_value;

            const int thread_index = route_lane * DINF_PROMPT_N + lane;
#pragma unroll
            for (int item = thread_index;
                 item < DINF_PROMPT_N * DINF_Q4_BLOCK_SIZE;
                 item += DINF_PROMPT_M * DINF_PROMPT_N) {
                const int tile_row = item / DINF_Q4_BLOCK_SIZE;
                const int inner = item % DINF_Q4_BLOCK_SIZE;
                const int candidate_output_row = blockIdx.x * DINF_PROMPT_N + tile_row;
                const bool valid_weight = candidate_output_row < out_dim;
                const int safe_output_row = valid_weight ? candidate_output_row : 0;
                const unsigned long long expert_row =
                    (unsigned long long) expert * out_dim + safe_output_row;
                const unsigned char *packed = dinf_block_ptr<TYPE_SIZE>(
                    weights + expert_row * dinf_row_bytes<TYPE_SIZE>(blocks_per_row), block);
                weight_tile[tile_row][inner] = valid_weight
                    ? dinf_block_weight<TYPE_SIZE>(packed, block, inner)
                    : 0.0f;
            }
            __syncthreads();

            if (valid_route && valid_output) {
#pragma unroll
                for (int inner = 0; inner < DINF_Q4_BLOCK_SIZE; ++inner) {
                    sum = fmaf(
                        input_tile[route_lane][inner],
                        weight_tile[lane][inner],
                        sum);
                }
            }
            __syncthreads();
        }

        if (valid_route && valid_output) {
            if (expert_scales != nullptr) {
                sum *= expert_scales[expert];
            }
            output[(unsigned long long) route * out_dim + output_row] = sum;
        }
    }
}

// Stable sort of the routes by expert in one launch. Each route counts the
// routes that precede it (a lower expert, or the same expert and a lower
// route index) and writes itself at that rank: `sorted[rank]` is the route
// and `sorted[routes + rank]` its expert. Quadratic, for prompt-sized inputs.
extern "C" __global__ void dinf_indexed_moe_rank_sort(
    const unsigned int *experts,
    unsigned int *sorted,
    int routes) {
    const int route = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= routes) {
        return;
    }
    const unsigned int expert = experts[route];
    int rank = 0;
    for (int other = 0; other < routes; ++other) {
        const unsigned int other_expert = experts[other];
        rank += (other_expert < expert) || (other_expert == expert && other < route);
    }
    sorted[rank] = (unsigned int) route;
    sorted[routes + rank] = expert;
}

#define DINF_DECODE_ENTRY(NAME, TYPE_SIZE) \
extern "C" __global__ void NAME( \
    const float *input, const unsigned int *expert_ids, const unsigned char *weights, \
    float *output, int routes, int top_k, int input_rows, int in_dim, int out_dim, \
    int num_experts, int activation_kind, const float *expert_scales) { \
    dinf_indexed_moe_decode<TYPE_SIZE>(input, expert_ids, weights, output, routes, top_k, \
        input_rows, in_dim, out_dim, num_experts, activation_kind, expert_scales); \
}

#define DINF_GROUPED_ENTRY(NAME, TYPE_SIZE) \
extern "C" __global__ void NAME( \
    const float *input, const unsigned int *sorted_route_ids, \
    const unsigned int *sorted_experts, const unsigned char *weights, float *output, \
    int routes, int top_k, int input_rows, int in_dim, int out_dim, int num_experts, \
    int activation_kind, const float *expert_scales) { \
    dinf_indexed_moe_grouped<TYPE_SIZE>(input, sorted_route_ids, sorted_experts, weights, \
        output, routes, top_k, input_rows, in_dim, out_dim, num_experts, activation_kind, \
        expert_scales); \
}

DINF_DECODE_ENTRY(dinf_indexed_moe_q4_0_f32, 18)
DINF_DECODE_ENTRY(dinf_indexed_moe_q5_0_f32, 22)
DINF_DECODE_ENTRY(dinf_indexed_moe_q8_0_f32, 34)
DINF_GROUPED_ENTRY(dinf_indexed_moe_q4_0_grouped_f32, 18)
DINF_GROUPED_ENTRY(dinf_indexed_moe_q5_0_grouped_f32, 22)
DINF_GROUPED_ENTRY(dinf_indexed_moe_q8_0_grouped_f32, 34)
// Single-token product for the raw formats: one warp per (route, output
// row), each lane taking 16 weights of the row at a time (an NVFP4 scale
// group or half an IQ4_NL block). A lane decodes a group's scale once,
// reads its nibbles as words and its inputs as float4, and looks the values
// up in a shared table, which distinct nibbles read from distinct banks.
template <int FORMAT>
__device__ __forceinline__ void dinf_raw_moe_decode(
    const float *input,
    const unsigned int *expert_ids,
    const unsigned char *weights,
    float *output,
    int routes,
    int top_k,
    int input_rows,
    int in_dim,
    int out_dim,
    int num_experts,
    const float *expert_scales) {
    __shared__ float values[16];
    if (threadIdx.y == 0 && threadIdx.x < 16) {
        values[threadIdx.x] = FORMAT == DINF_NVFP4
            ? (float) dinf_fp4_values[threadIdx.x]
            : (float) dinf_iq4nl_values[threadIdx.x];
    }
    __syncthreads();
    const int lane = threadIdx.x;
    const int route = blockIdx.y;
    const int output_row = blockIdx.x * blockDim.y + threadIdx.y;
    if (route >= routes || output_row >= out_dim) {
        return;
    }
    const unsigned int expert = expert_ids[route];
    if (expert >= (unsigned int) num_experts) {
        if (lane == 0) {
            output[(unsigned long long) route * out_dim + output_row] = 0.0f;
        }
        return;
    }
    const int input_row = input_rows == 1 ? route / top_k : route;
    const float *x = input + (unsigned long long) input_row * in_dim;
    const unsigned long long row = (unsigned long long) expert * out_dim + output_row;
    float sum = 0.0f;
    if (FORMAT == DINF_NVFP4) {
        // Byte j of a group holds weight j (low nibble) and j + 8 (high).
        const unsigned char *w = weights + row * (unsigned long long) (in_dim / 64) * 36ull;
        const int groups = in_dim / 16;
        for (int g = lane; g < groups; g += 32) {
            const unsigned char *block = w + (g >> 2) * 36;
            const int sub = g & 3;
            const float d = dinf_ue4m3(block[sub]);
            const unsigned int lo = *reinterpret_cast<const unsigned int *>(block + 4 + sub * 8);
            const unsigned int hi = *reinterpret_cast<const unsigned int *>(block + 8 + sub * 8);
            const float4 *xv = reinterpret_cast<const float4 *>(x + g * 16);
            const float4 x0 = xv[0];
            const float4 x1 = xv[1];
            const float4 x2 = xv[2];
            const float4 x3 = xv[3];
            float acc = values[lo & 15] * x0.x;
            acc = fmaf(values[(lo >> 8) & 15], x0.y, acc);
            acc = fmaf(values[(lo >> 16) & 15], x0.z, acc);
            acc = fmaf(values[(lo >> 24) & 15], x0.w, acc);
            acc = fmaf(values[hi & 15], x1.x, acc);
            acc = fmaf(values[(hi >> 8) & 15], x1.y, acc);
            acc = fmaf(values[(hi >> 16) & 15], x1.z, acc);
            acc = fmaf(values[(hi >> 24) & 15], x1.w, acc);
            acc = fmaf(values[(lo >> 4) & 15], x2.x, acc);
            acc = fmaf(values[(lo >> 12) & 15], x2.y, acc);
            acc = fmaf(values[(lo >> 20) & 15], x2.z, acc);
            acc = fmaf(values[(lo >> 28) & 15], x2.w, acc);
            acc = fmaf(values[(hi >> 4) & 15], x3.x, acc);
            acc = fmaf(values[(hi >> 12) & 15], x3.y, acc);
            acc = fmaf(values[(hi >> 20) & 15], x3.z, acc);
            acc = fmaf(values[(hi >> 28) & 15], x3.w, acc);
            sum = fmaf(d, acc, sum);
        }
    } else if (FORMAT == DINF_MXFP8) {
        // A lane takes half a block: sixteen E4M3 bytes under one scale.
        const unsigned char *w = weights + row * (unsigned long long) (in_dim / 32) * 36ull;
        const int units = in_dim / 16;
        for (int u = lane; u < units; u += 32) {
            const unsigned char *block = w + (u >> 1) * 36;
            const unsigned int *bytes =
                reinterpret_cast<const unsigned int *>(block + 4 + (u & 1) * 16);
            const float4 *xv = reinterpret_cast<const float4 *>(x + u * 16);
            float acc = 0.0f;
#pragma unroll
            for (int word = 0; word < 4; ++word) {
                const unsigned int packed = bytes[word];
                const float4 xw = xv[word];
                acc = fmaf(dinf_e4m3(packed & 255u), xw.x, acc);
                acc = fmaf(dinf_e4m3((packed >> 8) & 255u), xw.y, acc);
                acc = fmaf(dinf_e4m3((packed >> 16) & 255u), xw.z, acc);
                acc = fmaf(dinf_e4m3(packed >> 24), xw.w, acc);
            }
            sum = fmaf(dinf_e8m0(block[0]), acc, sum);
        }
    } else {
        // Byte j of a block holds weight j (low nibble) and j + 16 (high).
        // A lane takes half a block: eight bytes, weights h..h+7 and
        // h+16..h+23 for h = 0 or 8, so units match NVFP4's 16 weights.
        const unsigned char *w = weights + row * (unsigned long long) (in_dim / 32) * 18ull;
        const int units = in_dim / 16;
        for (int u = lane; u < units; u += 32) {
            const unsigned char *block = w + (u >> 1) * 18;
            const int half = (u & 1) * 8;
            const float d = __half2float(*reinterpret_cast<const __half *>(block));
            const unsigned short *pairs = reinterpret_cast<const unsigned short *>(block + 2 + half);
            const float4 *low = reinterpret_cast<const float4 *>(x + (u >> 1) * 32 + half);
            const float4 *high = reinterpret_cast<const float4 *>(x + (u >> 1) * 32 + 16 + half);
            const float4 l0 = low[0];
            const float4 l1 = low[1];
            const float4 h0 = high[0];
            const float4 h1 = high[1];
            const unsigned int p0 = pairs[0];
            const unsigned int p1 = pairs[1];
            const unsigned int p2 = pairs[2];
            const unsigned int p3 = pairs[3];
            float acc = values[p0 & 15] * l0.x;
            acc = fmaf(values[(p0 >> 8) & 15], l0.y, acc);
            acc = fmaf(values[p1 & 15], l0.z, acc);
            acc = fmaf(values[(p1 >> 8) & 15], l0.w, acc);
            acc = fmaf(values[p2 & 15], l1.x, acc);
            acc = fmaf(values[(p2 >> 8) & 15], l1.y, acc);
            acc = fmaf(values[p3 & 15], l1.z, acc);
            acc = fmaf(values[(p3 >> 8) & 15], l1.w, acc);
            acc = fmaf(values[(p0 >> 4) & 15], h0.x, acc);
            acc = fmaf(values[(p0 >> 12) & 15], h0.y, acc);
            acc = fmaf(values[(p1 >> 4) & 15], h0.z, acc);
            acc = fmaf(values[(p1 >> 12) & 15], h0.w, acc);
            acc = fmaf(values[(p2 >> 4) & 15], h1.x, acc);
            acc = fmaf(values[(p2 >> 12) & 15], h1.y, acc);
            acc = fmaf(values[(p3 >> 4) & 15], h1.z, acc);
            acc = fmaf(values[(p3 >> 12) & 15], h1.w, acc);
            sum = fmaf(d, acc, sum);
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        if (expert_scales != nullptr) {
            sum *= expert_scales[expert];
        }
        output[(unsigned long long) route * out_dim + output_row] = sum;
    }
}

#define DINF_RAW_DECODE_ENTRY(NAME, FORMAT) \
extern "C" __global__ void NAME( \
    const float *input, const unsigned int *expert_ids, const unsigned char *weights, \
    float *output, int routes, int top_k, int input_rows, int in_dim, int out_dim, \
    int num_experts, int activation_kind, const float *expert_scales) { \
    (void) activation_kind; \
    dinf_raw_moe_decode<FORMAT>(input, expert_ids, weights, output, routes, top_k, \
        input_rows, in_dim, out_dim, num_experts, expert_scales); \
}

DINF_RAW_DECODE_ENTRY(dinf_indexed_moe_nvfp4_f32, DINF_NVFP4)
DINF_GROUPED_ENTRY(dinf_indexed_moe_nvfp4_grouped_f32, DINF_NVFP4)
DINF_RAW_DECODE_ENTRY(dinf_indexed_moe_iq4_nl_f32, DINF_IQ4_NL)
DINF_RAW_DECODE_ENTRY(dinf_indexed_moe_mxfp8_f32, DINF_MXFP8)
DINF_GROUPED_ENTRY(dinf_indexed_moe_mxfp8_grouped_f32, DINF_MXFP8)
DINF_GROUPED_ENTRY(dinf_indexed_moe_iq4_nl_grouped_f32, DINF_IQ4_NL)
"#
    );

    static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    static CUDA_LOGGED: OnceLock<()> = OnceLock::new();

    fn cuda_include_paths() -> Vec<String> {
        let mut paths = Vec::new();
        for variable in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
            let Some(root) = std::env::var_os(variable) else {
                continue;
            };
            let root = PathBuf::from(root);
            let include = if root.join("cuda_fp16.h").is_file() {
                root
            } else {
                root.join("include")
            };
            if include.join("cuda_fp16.h").is_file() {
                let include = include.to_string_lossy().into_owned();
                if !paths.contains(&include) {
                    paths.push(include);
                }
            }
        }
        for candidate in ["/usr/local/cuda/include", "/opt/cuda/include"] {
            let include = PathBuf::from(candidate);
            if include.join("cuda_fp16.h").is_file() {
                let include = include.to_string_lossy().into_owned();
                if !paths.contains(&include) {
                    paths.push(include);
                }
            }
        }
        paths
    }

    fn ptx() -> candle_core::Result<&'static str> {
        match PTX.get_or_init(|| {
            let include_paths = cuda_include_paths();
            if include_paths.is_empty() {
                return Err(
                    "CUDA toolkit include directory containing cuda_fp16.h was not found".into(),
                );
            }
            compile_ptx_with_opts(
                CUDA_SOURCE,
                CompileOptions {
                    include_paths,
                    name: Some("indexed_moe_q4_0.cu".to_string()),
                    use_fast_math: Some(true),
                    ..Default::default()
                },
            )
            .map(|ptx| ptx.to_src())
            .map_err(|error| error.to_string())
        }) {
            Ok(ptx) => Ok(ptx.as_str()),
            Err(error) => candle_core::bail!("failed to compile CUDA Q4_0 indexed MoE: {error}"),
        }
    }

    static WMMA_PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();

    /// Tensor-core kernels need Volta or newer; a toolkit or device that
    /// rejects them leaves the exact F32 and Candle paths in use.
    fn wmma_ptx() -> candle_core::Result<&'static str> {
        match WMMA_PTX.get_or_init(|| {
            let include_paths = cuda_include_paths();
            if include_paths.is_empty() {
                return Err(
                    "CUDA toolkit include directory containing cuda_fp16.h was not found".into(),
                );
            }
            compile_ptx_with_opts(
                WMMA_SOURCE,
                CompileOptions {
                    include_paths,
                    name: Some("indexed_moe_wmma.cu".to_string()),
                    arch: Some("sm_70"),
                    ..Default::default()
                },
            )
            .map(|ptx| ptx.to_src())
            .map_err(|error| error.to_string())
        }) {
            Ok(ptx) => Ok(ptx.as_str()),
            Err(error) => candle_core::bail!("failed to compile CUDA WMMA indexed MoE: {error}"),
        }
    }

    const MMA_SOURCE: &str = concat!(
        r#"
#include <cuda_fp16.h>
"#,
        raw_formats_source!(),
        r#"
// Output rows per block (16 per warp), routes per tile (the tile table's
// DINF_ROUTE_TILE), inner elements per step, and the padded shared row
// width in halves: 40 halves = 20 words, so the eight rows an ldmatrix
// reads land in distinct banks.
#define DINF_MMA_N 128
#define DINF_MMA_M 128
#define DINF_MMA_K 32
#define DINF_MMA_STRIDE 40
#define DINF_MMA_THREADS 256
#define DINF_NO_ROUTE 0xffffffffu

static __device__ __forceinline__ void dinf_mma_16816(float *d, const unsigned int *a, const unsigned int *b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

static __device__ __forceinline__ void dinf_ldmatrix_x4(unsigned int *r, const __half *p) {
    const unsigned int address = (unsigned int) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(address));
}

static __device__ __forceinline__ void dinf_cp_async_16(__half *dst, const __half *src) {
    const unsigned int address = (unsigned int) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(address), "l"(src));
}

// The raw bytes of one thread's 16 weights for one 32-element step:
// output row `expert_row`, weights [16 * half, 16 * half + 16) of the step.
struct DinfRawWeights {
    unsigned int words[4];
    unsigned int scale;
};

template <int FORMAT>
static __device__ __forceinline__ DinfRawWeights dinf_raw_fetch(
    const unsigned char *weights, unsigned long long expert_row, int step, int in_dim, int half) {
    DinfRawWeights raw;
    if (FORMAT == DINF_NVFP4) {
        // 64 weights per 36-byte block: four scales, then 8 bytes per
        // 16-weight group (byte j: weight j low, j + 8 high).
        const unsigned char *block = weights
            + (expert_row * (unsigned long long) (in_dim / 64) + (unsigned long long) (step >> 1)) * 36ull;
        const int group = ((step & 1) << 1) | half;
        const unsigned int *words = reinterpret_cast<const unsigned int *>(block);
        raw.scale = (words[0] >> (8 * group)) & 0xffu;
        raw.words[0] = words[1 + 2 * group];
        raw.words[1] = words[2 + 2 * group];
        raw.words[2] = 0;
        raw.words[3] = 0;
    } else if (FORMAT == DINF_MXFP8) {
        // 32 weights per 36-byte block: the UE8M0 scale, three zero bytes,
        // then the E4M3 bytes.
        const unsigned int *words = reinterpret_cast<const unsigned int *>(
            weights + (expert_row * (unsigned long long) (in_dim / 32) + (unsigned long long) step) * 36ull);
        raw.scale = words[0] & 0xffu;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            raw.words[i] = words[1 + 4 * half + i];
        }
    } else {
        // IQ4_NL: 32 weights per 18-byte block, an f16 scale then 16 bytes
        // (byte j: weight j low, j + 16 high); 2-byte aligned.
        const unsigned short *shorts = reinterpret_cast<const unsigned short *>(
            weights + (expert_row * (unsigned long long) (in_dim / 32) + (unsigned long long) step) * 18ull);
        raw.scale = shorts[0];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            raw.words[i] = (unsigned int) shorts[1 + 2 * i] | ((unsigned int) shorts[2 + 2 * i] << 16);
        }
    }
    return raw;
}

static __device__ __forceinline__ unsigned int dinf_half_pair(__half lo, __half hi) {
    return (unsigned int) __half_as_ushort(lo) | ((unsigned int) __half_as_ushort(hi) << 16);
}

// The 16 weights of `raw` as f16 pairs in order, exactly as
// dinf_tile_weights8 rounds them.
template <int FORMAT>
static __device__ __forceinline__ void dinf_raw_decode(
    const DinfRawWeights &raw, int half, const __half *lut, unsigned int *packed) {
    if (FORMAT == DINF_NVFP4) {
        // fp4 * scale is exact in f16 (at most 6 significant bits). Weights
        // 0..7 are the low nibbles of bytes 0..7, weights 8..15 the high ones.
        const __half scale = __float2half(dinf_ue4m3((unsigned char) raw.scale));
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned int word = raw.words[i >> 1];
            const unsigned int lo = (word >> (16 * (i & 1))) & 0xffu;
            const unsigned int hi = (word >> (16 * (i & 1) + 8)) & 0xffu;
            packed[i] = dinf_half_pair(__hmul(lut[lo & 15u], scale), __hmul(lut[hi & 15u], scale));
            packed[4 + i] = dinf_half_pair(__hmul(lut[lo >> 4], scale), __hmul(lut[hi >> 4], scale));
        }
    } else if (FORMAT == DINF_MXFP8) {
        // An E4M3 byte's low seven bits, shifted up seven, are an f16 of its
        // magnitude times 2^-8 (exponent bias 15 against 7, subnormals
        // included), so one f16 multiply by 2^(e - 127 + 8) is the single
        // rounding of the product. Scales outside f16's normal powers of two
        // take the float path.
        const int exponent = (int) raw.scale - 127 + 8;
        if (raw.scale != 0 && exponent >= -14 && exponent <= 15) {
            const __half2 scale = __half2half2(__ushort_as_half((unsigned short) ((exponent + 15) << 10)));
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const unsigned int word = raw.words[i >> 1] >> (16 * (i & 1));
                const unsigned int bits = ((word & 0x7fu) << 7) | ((word & 0x80u) << 8)
                    | ((word & 0x7f00u) << 15) | ((word & 0x8000u) << 16);
                const __half2 product = __hmul2(*reinterpret_cast<const __half2 *>(&bits), scale);
                packed[i] = *reinterpret_cast<const unsigned int *>(&product);
            }
        } else {
            const float scale = dinf_e8m0(raw.scale);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const unsigned int word = raw.words[i >> 1] >> (16 * (i & 1));
                packed[i] = dinf_half_pair(__float2half(dinf_e4m3(word & 0xffu) * scale),
                                           __float2half(dinf_e4m3((word >> 8) & 0xffu) * scale));
            }
        }
    } else {
        // One f16 rounding of value * scale, as the float product rounds.
        const __half scale = __ushort_as_half((unsigned short) raw.scale);
        const int shift = half ? 4 : 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned int word = raw.words[i >> 1] >> (16 * (i & 1));
            packed[i] = dinf_half_pair(__hmul(lut[(word >> shift) & 15u], scale),
                                       __hmul(lut[(word >> (8 + shift)) & 15u], scale));
        }
    }
}

// Grouped prompt expert products on mma.sync tensor cores for the raw
// formats: block (output tile, route tile) computes DINF_MMA_N output rows
// for one tile of one expert's sorted routes (see dinf_indexed_moe_wmma_tiles).
// Each warp owns 16 output rows for every route of the tile, so all warps
// work however few routes the tile has. Per 32-element step the block
// dequantizes its 128 x 32 weights into shared memory (each thread 16) and
// copies the routes' f16 inputs in with cp.async, both one step ahead of
// the products. Outputs are scattered to route order, times the expert's
// scale when `expert_scales` is set.
template <int FORMAT>
static __device__ __forceinline__ void dinf_raw_moe_grouped_mma(
    const __half *gathered, const unsigned int *sorted_route_ids, const unsigned int *tiles,
    const unsigned char *weights, float *output, int routes, int in_dim, int out_dim,
    int num_experts, const float *expert_scales) {
    __shared__ __align__(16) __half a_tile[2][DINF_MMA_M * DINF_MMA_STRIDE];
    __shared__ __align__(16) __half b_tile[2][DINF_MMA_N * DINF_MMA_STRIDE];
    __shared__ unsigned int tile_route[DINF_MMA_M];
    __shared__ __half lut[16];
    const int thread = threadIdx.x;
    const int warp = thread >> 5;
    const int lane = thread & 31;
    const unsigned int *tile = tiles + 3ull * blockIdx.y;
    const int expert = (int) tile[0];
    const int segment_start = (int) tile[1];
    const int tile_routes = (int) tile[2];
    if (tile_routes <= 0 || tile_routes > DINF_MMA_M || expert >= num_experts
        || segment_start < 0 || segment_start > routes - tile_routes) {
        return;
    }
    if (thread < 16) {
        lut[thread] = __float2half(FORMAT == DINF_NVFP4 ? (float) dinf_fp4_values[thread]
                                                        : (float) dinf_iq4nl_values[thread]);
    }
    if (thread < DINF_MMA_M) {
        unsigned int route = DINF_NO_ROUTE;
        if (thread < tile_routes) {
            const unsigned int candidate = sorted_route_ids[segment_start + thread];
            if (candidate < (unsigned int) routes) {
                route = candidate;
            }
        }
        tile_route[thread] = route;
    }
    // Rows past the tile, up to its last 16, belong to the next expert or
    // the padding; their products are discarded.
    const int m_tiles = (tile_routes + 15) / 16;
    const int a_rows = m_tiles * 16;
    const int first_row = blockIdx.x * DINF_MMA_N;
    const int steps = in_dim / DINF_MMA_K;
    const int w_row = thread >> 1;
    const int w_half = thread & 1;
    const unsigned long long expert_row =
        (unsigned long long) expert * out_dim + min(first_row + w_row, out_dim - 1);
    const __half *a_source = gathered + (unsigned long long) segment_start * in_dim;

    // Two 16-byte chunks per thread cover 128 rows of 32 halves.
    auto copy_inputs = [&](int step, int buffer) {
#pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const int index = pass * DINF_MMA_THREADS + thread;
            const int row = index >> 2;
            const int chunk = index & 3;
            if (row < a_rows) {
                dinf_cp_async_16(a_tile[buffer] + row * DINF_MMA_STRIDE + chunk * 8,
                                 a_source + (unsigned long long) row * in_dim + step * DINF_MMA_K + chunk * 8);
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    auto store_weights = [&](const DinfRawWeights &raw, int buffer) {
        unsigned int packed[8];
        dinf_raw_decode<FORMAT>(raw, w_half, lut, packed);
        uint4 *dst = reinterpret_cast<uint4 *>(b_tile[buffer] + w_row * DINF_MMA_STRIDE + w_half * 16);
        dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
        dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
    };

    float acc[DINF_MMA_M / 16][2][4];
#pragma unroll
    for (int m = 0; m < DINF_MMA_M / 16; ++m) {
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[m][n][i] = 0.0f;
            }
        }
    }

    copy_inputs(0, 0);
    __syncthreads();  // the value table
    store_weights(dinf_raw_fetch<FORMAT>(weights, expert_row, 0, in_dim, w_half), 0);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    for (int step = 0; step < steps; ++step) {
        const int buffer = step & 1;
        const bool more = step + 1 < steps;
        DinfRawWeights next = {};
        if (more) {
            copy_inputs(step + 1, buffer ^ 1);
            next = dinf_raw_fetch<FORMAT>(weights, expert_row, step + 1, in_dim, w_half);
        }
        const __half *a = a_tile[buffer];
        const __half *b = b_tile[buffer] + warp * 16 * DINF_MMA_STRIDE;
#pragma unroll
        for (int k = 0; k < DINF_MMA_K; k += 16) {
            // Rows +0..7 / +8..15 of the warp's outputs by inner +0..7 / +8..15.
            const int matrix = lane >> 3;
            unsigned int bf[4];
            dinf_ldmatrix_x4(bf, b + ((matrix >> 1) * 8 + (lane & 7)) * DINF_MMA_STRIDE + k + (matrix & 1) * 8);
#pragma unroll
            for (int m = 0; m < DINF_MMA_M / 16; ++m) {
                if (m < m_tiles) {
                    // Routes +0..7 / +8..15 by inner +0..7 / +8..15: a0..a3.
                    unsigned int af[4];
                    dinf_ldmatrix_x4(af, a + (m * 16 + (matrix & 1) * 8 + (lane & 7)) * DINF_MMA_STRIDE + k + (matrix >> 1) * 8);
                    dinf_mma_16816(acc[m][0], af, bf);
                    dinf_mma_16816(acc[m][1], af, bf + 2);
                }
            }
        }
        if (more) {
            store_weights(next, buffer ^ 1);
        }
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }

    const float scale = expert_scales != nullptr ? expert_scales[expert] : 1.0f;
    const int group = lane >> 2;
    const int quad = lane & 3;
#pragma unroll
    for (int m = 0; m < DINF_MMA_M / 16; ++m) {
        if (m >= m_tiles) {
            continue;
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int slot = m * 16 + group + 8 * r;
            const unsigned int route = slot < tile_routes ? tile_route[slot] : DINF_NO_ROUTE;
            if (route == DINF_NO_ROUTE) {
                continue;
            }
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                const int column = first_row + warp * 16 + n * 8 + 2 * quad;
                float *o = output + (unsigned long long) route * out_dim + column;
                // Pairs are 8-byte aligned only when every row is.
                if ((out_dim & 1) == 0 && column + 1 < out_dim) {
                    *reinterpret_cast<float2 *>(o) =
                        make_float2(acc[m][n][2 * r] * scale, acc[m][n][2 * r + 1] * scale);
                } else {
                    if (column < out_dim) {
                        o[0] = acc[m][n][2 * r] * scale;
                    }
                    if (column + 1 < out_dim) {
                        o[1] = acc[m][n][2 * r + 1] * scale;
                    }
                }
            }
        }
    }
}

#define DINF_MMA_ENTRY(NAME, FORMAT)                                                          \
    extern "C" __global__ void __launch_bounds__(DINF_MMA_THREADS) NAME(                     \
        const __half *gathered, const unsigned int *sorted_route_ids,                       \
        const unsigned int *tiles, const unsigned char *weights, float *output,             \
        int routes, int in_dim, int out_dim, int num_experts, const float *expert_scales) { \
        dinf_raw_moe_grouped_mma<FORMAT>(gathered, sorted_route_ids, tiles, weights, output,  \
                                         routes, in_dim, out_dim, num_experts, expert_scales); \
    }

DINF_MMA_ENTRY(dinf_raw_moe_nvfp4_grouped_mma, DINF_NVFP4)
DINF_MMA_ENTRY(dinf_raw_moe_iq4_nl_grouped_mma, DINF_IQ4_NL)
DINF_MMA_ENTRY(dinf_raw_moe_mxfp8_grouped_mma, DINF_MXFP8)
"#
    );

    static MMA_PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();

    /// The raw formats' `mma.sync` grouped kernels need `mma.sync` m16n8k16
    /// and `cp.async` (Ampere or newer); elsewhere the WMMA kernels stay.
    fn mma_ptx() -> candle_core::Result<&'static str> {
        match MMA_PTX.get_or_init(|| {
            compile_ptx_with_opts(
                MMA_SOURCE,
                CompileOptions {
                    include_paths: cuda_include_paths(),
                    name: Some("raw_moe_mma.cu".to_string()),
                    arch: Some("sm_80"),
                    ..Default::default()
                },
            )
            .map(|ptx| ptx.to_src())
            .map_err(|error| error.to_string())
        }) {
            Ok(ptx) => Ok(ptx.as_str()),
            Err(error) => candle_core::bail!("failed to compile CUDA mma raw MoE: {error}"),
        }
    }

    /// Whether `name` of the `mma.sync` module loads on `device`.
    fn mma_loads(device: &candle_core::CudaDevice, name: &str) -> bool {
        mma_ptx().is_ok_and(|ptx| {
            device
                .get_or_load_custom_func(name, MMA_MODULE_NAME, ptx)
                .is_ok()
        })
    }

    pub(crate) fn precompile(device: &Device) -> anyhow::Result<()> {
        if !device.is_cuda() {
            return Ok(());
        }
        let device = device.as_cuda_device()?;
        for kernels in [Q4_0_KERNELS, Q5_0_KERNELS, Q8_0_KERNELS]
            .into_iter()
            .filter_map(|k| k.f32)
        {
            device.get_or_load_custom_func(kernels.decode, CUDA_MODULE_NAME, ptx()?)?;
            device.get_or_load_custom_func(kernels.prompt, CUDA_MODULE_NAME, ptx()?)?;
        }
        CUDA_LOGGED.get_or_init(|| {
            tracing::info!("CUDA Q4_0/Q5_0 decode and grouped prompt expert matmul enabled");
        });
        Ok(())
    }

    struct RankSort;

    impl candle_core::CustomOp1 for RankSort {
        fn name(&self) -> &'static str {
            "cuda-indexed-moe-rank-sort"
        }

        fn cpu_fwd(
            &self,
            _experts: &CpuStorage,
            _layout: &Layout,
        ) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("cuda-indexed-moe-rank-sort is only dispatched on CUDA")
        }

        fn cuda_fwd(
            &self,
            experts: &CudaStorage,
            layout: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let routes = layout.shape().elem_count();
            if routes == 0 || routes > RANK_SORT_MAX_ROUTES {
                candle_core::bail!("CUDA rank sort route count is out of range")
            }
            let view = checked_view_u32(
                experts.as_cuda_slice::<u32>()?,
                layout,
                routes,
                "route experts",
            )?;
            let device = experts.device();
            // SAFETY: Ranks are a permutation of `0..routes`, so every
            // element of both halves is written.
            let mut sorted = unsafe { device.alloc::<u32>(2 * routes)? };
            let function =
                device.get_or_load_custom_func(RANK_SORT_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
            let routes_arg = routes as i32;
            let mut builder = function.builder();
            builder.arg(&view).arg(&mut sorted).arg(&routes_arg);
            // SAFETY: The view holds `routes` experts and the output `2 * routes`.
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (routes.div_ceil(RANK_SORT_THREADS) as u32, 1, 1),
                    block_dim: (RANK_SORT_THREADS as u32, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .w()?;
            Ok((
                CudaStorage::wrap_cuda_slice(sorted, device.clone()),
                Shape::from((2, routes)),
            ))
        }
    }

    /// Route indices ordered by expert, and the experts in that order.
    pub(super) fn sort_routes(route_experts: &Tensor) -> anyhow::Result<(Tensor, Tensor)> {
        let routes = route_experts.elem_count();
        if routes <= RANK_SORT_MAX_ROUTES {
            let sorted = route_experts.apply_op1_no_bwd(&RankSort)?;
            return Ok((sorted.get(0)?, sorted.get(1)?));
        }
        let sorted_route_ids = route_experts.arg_sort_last_dim(true)?;
        let sorted_experts = route_experts
            .gather(&sorted_route_ids, D::Minus1)?
            .contiguous()?;
        Ok((sorted_route_ids, sorted_experts))
    }

    /// [`sort_routes`] of a prompt's expert IDs, remembering the last result
    /// on this thread. A layer routes its gate/up and its down projection by
    /// the same IDs tensor, so the second one reuses the first one's sort.
    /// Tensor IDs are never reused, so a hit is always that same tensor.
    fn sorted_routes_of(expert_ids: &Tensor) -> anyhow::Result<(Tensor, Tensor)> {
        thread_local! {
            static LAST: std::cell::RefCell<Option<(candle_core::TensorId, Tensor, Tensor)>> =
                const { std::cell::RefCell::new(None) };
        }
        let id = expert_ids.id();
        let hit = LAST.with(|last| {
            last.borrow()
                .as_ref()
                .filter(|(key, _, _)| *key == id)
                .map(|(_, routes, experts)| (routes.clone(), experts.clone()))
        });
        if let Some(sorted) = hit {
            return Ok(sorted);
        }
        let sorted = sort_routes(&expert_ids.flatten_all()?.contiguous()?)?;
        LAST.with(|last| {
            *last.borrow_mut() = Some((id, sorted.0.clone(), sorted.1.clone()));
        });
        Ok(sorted)
    }

    struct DecodeIndexedQ4<'a> {
        bank: ExpertBank<'a>,
        geometry: Q4IndexedGeometry,
        activation_kind: i32,
    }

    impl candle_core::CustomOp2 for DecodeIndexedQ4<'_> {
        fn name(&self) -> &'static str {
            "cuda-indexed-moe-q4-0"
        }

        fn cpu_fwd(
            &self,
            _input: &CpuStorage,
            _input_layout: &Layout,
            _ids: &CpuStorage,
            _ids_layout: &Layout,
        ) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("cuda-indexed-moe-q4-0 is only dispatched on CUDA")
        }

        fn cuda_fwd(
            &self,
            input: &CudaStorage,
            input_layout: &Layout,
            ids: &CudaStorage,
            ids_layout: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let geometry = self.geometry;
            validate_storage(
                "input",
                input,
                input_layout,
                DType::F32,
                geometry.input_elements,
            )?;
            validate_storage("expert IDs", ids, ids_layout, DType::U32, geometry.routes)?;
            if input.device().id() != ids.device().id() {
                candle_core::bail!("CUDA Q4_0 indexed MoE inputs use different devices")
            }
            let Some(kernels) = self.bank.kernels() else {
                candle_core::bail!("CUDA indexed MoE does not support this expert format")
            };
            if kernels.type_size != geometry.type_size
                || self.bank.dims()
                    != Some([geometry.num_experts, geometry.out_dim, geometry.in_dim])
                || self.bank.byte_len() != geometry.weight_bytes
                || !self.bank.is_on(input.device())
            {
                candle_core::bail!("CUDA Q4_0 indexed MoE weight geometry differs")
            }

            let input_view = checked_view_f32(
                input.as_cuda_slice::<f32>()?,
                input_layout,
                geometry.input_elements,
                "input",
            )?;
            let ids_view = checked_view_u32(
                ids.as_cuda_slice::<u32>()?,
                ids_layout,
                geometry.routes,
                "expert IDs",
            )?;
            let device = input.device();
            let stream = device.cuda_stream();
            // SAFETY: The decode kernel writes every output element,
            // including zeros for an out-of-range expert.
            let mut output = unsafe { device.alloc::<f32>(geometry.output_elements)? };
            let Some(f32_kernels) = kernels.f32 else {
                candle_core::bail!("CUDA indexed MoE has no decode kernel for this expert format")
            };
            let function =
                device.get_or_load_custom_func(f32_kernels.decode, CUDA_MODULE_NAME, ptx()?)?;
            let config = LaunchConfig {
                grid_dim: (
                    geometry.out_dim.div_ceil(CUDA_WARPS_PER_BLOCK) as u32,
                    geometry.routes as u32,
                    1,
                ),
                block_dim: (CUDA_WARP_SIZE as u32, CUDA_WARPS_PER_BLOCK as u32, 1),
                shared_mem_bytes: 0,
            };
            let routes = geometry.routes as i32;
            let top_k = geometry.top_k as i32;
            let input_rows = geometry.input_rows as i32;
            let in_dim = geometry.in_dim as i32;
            let out_dim = geometry.out_dim as i32;
            let num_experts = geometry.num_experts as i32;
            let activation_kind = self.activation_kind;
            self.bank
                .with_addresses(&stream, |weight_address, scale_address| {
                    let mut builder = function.builder();
                    builder
                        .arg(&input_view)
                        .arg(&ids_view)
                        .arg(&weight_address)
                        .arg(&mut output)
                        .arg(&routes)
                        .arg(&top_k)
                        .arg(&input_rows)
                        .arg(&in_dim)
                        .arg(&out_dim)
                        .arg(&num_experts)
                        .arg(&activation_kind)
                        .arg(&scale_address);
                    // SAFETY: Geometry, storage bounds, device identity, and
                    // all kernel index limits were checked before this launch.
                    unsafe { builder.launch(config) }.w()?;
                    Ok(())
                })?;

            Ok((
                CudaStorage::wrap_cuda_slice(output, device.clone()),
                Shape::from((geometry.tokens, geometry.top_k, geometry.out_dim)),
            ))
        }
    }

    struct GroupedPromptIndexedQ4<'a> {
        bank: ExpertBank<'a>,
        geometry: Q4IndexedGeometry,
        activation_kind: i32,
        /// Use the half-precision tensor-core kernel.
        wmma: bool,
    }

    impl candle_core::CustomOp3 for GroupedPromptIndexedQ4<'_> {
        fn name(&self) -> &'static str {
            "cuda-indexed-moe-q4-0-grouped-prompt"
        }

        fn cpu_fwd(
            &self,
            _input: &CpuStorage,
            _input_layout: &Layout,
            _sorted_route_ids: &CpuStorage,
            _sorted_route_ids_layout: &Layout,
            _sorted_experts: &CpuStorage,
            _sorted_experts_layout: &Layout,
        ) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("cuda-indexed-moe-q4-0-grouped-prompt is only dispatched on CUDA")
        }

        fn cuda_fwd(
            &self,
            input: &CudaStorage,
            input_layout: &Layout,
            sorted_route_ids: &CudaStorage,
            sorted_route_ids_layout: &Layout,
            sorted_experts: &CudaStorage,
            sorted_experts_layout: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let geometry = self.geometry;
            validate_storage(
                "input",
                input,
                input_layout,
                DType::F32,
                geometry.input_elements,
            )?;
            validate_storage(
                "sorted route IDs",
                sorted_route_ids,
                sorted_route_ids_layout,
                DType::U32,
                geometry.routes,
            )?;
            validate_storage(
                "sorted experts",
                sorted_experts,
                sorted_experts_layout,
                DType::U32,
                geometry.routes,
            )?;
            if input.device().id() != sorted_route_ids.device().id()
                || input.device().id() != sorted_experts.device().id()
            {
                candle_core::bail!("CUDA Q4_0 grouped prompt inputs use different devices")
            }
            let Some(kernels) = self.bank.kernels() else {
                candle_core::bail!("CUDA indexed MoE does not support this expert format")
            };
            if kernels.type_size != geometry.type_size
                || self.bank.dims()
                    != Some([geometry.num_experts, geometry.out_dim, geometry.in_dim])
                || self.bank.byte_len() != geometry.weight_bytes
                || !self.bank.is_on(input.device())
            {
                candle_core::bail!("CUDA Q4_0 grouped prompt weight geometry differs")
            }

            let input_view = checked_view_f32(
                input.as_cuda_slice::<f32>()?,
                input_layout,
                geometry.input_elements,
                "input",
            )?;
            let sorted_route_ids_view = checked_view_u32(
                sorted_route_ids.as_cuda_slice::<u32>()?,
                sorted_route_ids_layout,
                geometry.routes,
                "sorted route IDs",
            )?;
            let sorted_experts_view = checked_view_u32(
                sorted_experts.as_cuda_slice::<u32>()?,
                sorted_experts_layout,
                geometry.routes,
                "sorted experts",
            )?;
            let device = input.device();
            let stream = device.cuda_stream();
            let mut output = device.alloc_zeros::<f32>(geometry.output_elements)?;
            // Both grouped kernels give one block 32 output rows.
            const _: () = assert!(WMMA_TILE == CUDA_PROMPT_OUTPUT_TILE);
            let grid_dim = (
                geometry.out_dim.div_ceil(CUDA_PROMPT_OUTPUT_TILE) as u32,
                geometry.num_experts as u32,
                1,
            );
            if self.wmma {
                let Some(name) = kernels.wmma else {
                    candle_core::bail!("CUDA indexed MoE has no WMMA kernel for this format")
                };
                // One tile of padding rows lets the last route tile read past
                // the final route. Tensor-core loads need 32-byte aligned
                // rows, which a multiple-of-32 width gives.
                let padded_routes = geometry.routes + WMMA_TILE;
                let gathered_elements = padded_routes
                    .checked_mul(geometry.in_dim)
                    .filter(|elements| elements.div_ceil(WMMA_GATHER_THREADS) <= i32::MAX as usize)
                    .ok_or_else(|| {
                        candle_core::Error::Msg("CUDA WMMA indexed MoE input is too large".into())
                    })?;
                // SAFETY: The gather kernel writes every element, padding
                // included, before the product kernel reads it on this stream.
                let mut gathered = unsafe { device.alloc::<half::f16>(gathered_elements)? };
                let gather = device.get_or_load_custom_func(
                    WMMA_GATHER_KERNEL_NAME,
                    WMMA_MODULE_NAME,
                    wmma_ptx()?,
                )?;
                let (routes_wide, padded_wide) = (geometry.routes as i64, padded_routes as i64);
                let (top_k_wide, input_rows_wide) =
                    (geometry.top_k as i64, geometry.input_rows as i64);
                let in_dim_wide = geometry.in_dim as i64;
                let mut builder = gather.builder();
                builder
                    .arg(&input_view)
                    .arg(&sorted_route_ids_view)
                    .arg(&mut gathered)
                    .arg(&routes_wide)
                    .arg(&padded_wide)
                    .arg(&top_k_wide)
                    .arg(&input_rows_wide)
                    .arg(&in_dim_wide)
                    .arg(&self.activation_kind);
                // SAFETY: Route IDs are bounds-checked in the kernel and the
                // input view holds every row a valid route can name.
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (gathered_elements.div_ceil(WMMA_GATHER_THREADS) as u32, 1, 1),
                        block_dim: (WMMA_GATHER_THREADS as u32, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .w()?;

                let routes = geometry.routes as i32;
                let (in_dim, out_dim) = (geometry.in_dim as i32, geometry.out_dim as i32);
                let num_experts = geometry.num_experts as i32;
                // Every expert with routes has at most one partial tile.
                let max_tiles = geometry.num_experts + geometry.routes / WMMA_ROUTE_TILE;
                if max_tiles > i32::MAX as usize / 3 {
                    candle_core::bail!("CUDA WMMA indexed MoE has too many route tiles")
                }
                // SAFETY: The tile kernel writes all `max_tiles` triples
                // before the product kernel reads them on this stream.
                let mut tiles = unsafe { device.alloc::<u32>(3 * max_tiles)? };
                let tile_table = device.get_or_load_custom_func(
                    WMMA_TILES_KERNEL_NAME,
                    WMMA_MODULE_NAME,
                    wmma_ptx()?,
                )?;
                let max_tiles_arg = max_tiles as i32;
                let mut builder = tile_table.builder();
                builder
                    .arg(&sorted_experts_view)
                    .arg(&mut tiles)
                    .arg(&routes)
                    .arg(&num_experts)
                    .arg(&max_tiles_arg);
                // SAFETY: `sorted_experts` holds `routes` entries and the
                // table has room for the tile bound above.
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (WMMA_TILES_THREADS as u32, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .w()?;

                let mma = kernels.mma.filter(|name| mma_loads(device, name));
                let (function, grid_dim, block_dim) = match mma {
                    Some(name) => (
                        device.get_or_load_custom_func(name, MMA_MODULE_NAME, mma_ptx()?)?,
                        (
                            geometry.out_dim.div_ceil(MMA_OUTPUT_TILE) as u32,
                            max_tiles as u32,
                            1,
                        ),
                        (MMA_THREADS as u32, 1, 1),
                    ),
                    None => (
                        device.get_or_load_custom_func(name, WMMA_MODULE_NAME, wmma_ptx()?)?,
                        (
                            geometry.out_dim.div_ceil(WMMA_OUTPUT_TILE) as u32,
                            max_tiles as u32,
                            1,
                        ),
                        (CUDA_WARP_SIZE as u32, WMMA_WARPS as u32, 1),
                    ),
                };
                self.bank
                    .with_addresses(&stream, |weight_address, scale_address| {
                        let mut builder = function.builder();
                        builder
                            .arg(&gathered)
                            .arg(&sorted_route_ids_view)
                            .arg(&tiles)
                            .arg(&weight_address)
                            .arg(&mut output)
                            .arg(&routes)
                            .arg(&in_dim)
                            .arg(&out_dim)
                            .arg(&num_experts)
                            .arg(&scale_address);
                        // SAFETY: The sort outputs, the tile table, the
                        // gathered rows and the weights stay on this stream;
                        // geometry, storage bounds and launch limits were
                        // checked. The kernels reject a tile outside the
                        // routes and read at most 32 routes past a tile,
                        // which the padding covers.
                        unsafe {
                            builder.launch(LaunchConfig {
                                grid_dim,
                                block_dim,
                                shared_mem_bytes: 0,
                            })
                        }
                        .w()?;
                        Ok(())
                    })?;
                return Ok((
                    CudaStorage::wrap_cuda_slice(output, device.clone()),
                    Shape::from((geometry.tokens, geometry.top_k, geometry.out_dim)),
                ));
            }
            let Some(f32_kernels) = kernels.f32 else {
                candle_core::bail!("CUDA indexed MoE has no F32 kernel for this format")
            };
            let function =
                device.get_or_load_custom_func(f32_kernels.prompt, CUDA_MODULE_NAME, ptx()?)?;
            let config = LaunchConfig {
                grid_dim,
                block_dim: (
                    CUDA_PROMPT_OUTPUT_TILE as u32,
                    CUDA_PROMPT_ROUTE_TILE as u32,
                    1,
                ),
                shared_mem_bytes: 0,
            };
            let routes = geometry.routes as i32;
            let top_k = geometry.top_k as i32;
            let input_rows = geometry.input_rows as i32;
            let in_dim = geometry.in_dim as i32;
            let out_dim = geometry.out_dim as i32;
            let num_experts = geometry.num_experts as i32;
            let activation_kind = self.activation_kind;
            self.bank
                .with_addresses(&stream, |weight_address, scale_address| {
                    let mut builder = function.builder();
                    builder
                        .arg(&input_view)
                        .arg(&sorted_route_ids_view)
                        .arg(&sorted_experts_view)
                        .arg(&weight_address)
                        .arg(&mut output)
                        .arg(&routes)
                        .arg(&top_k)
                        .arg(&input_rows)
                        .arg(&in_dim)
                        .arg(&out_dim)
                        .arg(&num_experts)
                        .arg(&activation_kind)
                        .arg(&scale_address);
                    // SAFETY: The sort outputs and all storage views stay on
                    // this stream. Geometry, storage bounds, and launch limits
                    // were checked.
                    unsafe { builder.launch(config) }.w()?;
                    Ok(())
                })?;

            Ok((
                CudaStorage::wrap_cuda_slice(output, device.clone()),
                Shape::from((geometry.tokens, geometry.top_k, geometry.out_dim)),
            ))
        }
    }

    /// Rows of a quantized product's activation are padded to this many
    /// values, as in Candle's `MATRIX_ROW_PADDING`.
    const Q8_1_ROW_PADDING: usize = 512;
    /// Threads per block of Candle's `quantize_q8_1` kernel.
    const Q8_1_QUANTIZE_BLOCK: usize = 256;

    /// Candle's indexed K-quant expert kernel, launched from the engine.
    ///
    /// The launch is Candle's (`QTensor::indexed_moe_forward`) with two
    /// differences: an input or ID view that starts inside a larger buffer
    /// is read from where it starts, and one input row whose producer
    /// already wrote its Q8_1 form (see `q8_1_memo_cuda`) skips the
    /// quantization launch. The kernels and their arguments are unchanged,
    /// so the output has the same bits.
    struct CandleIndexedKQuant<'a> {
        weights: &'a QTensor,
    }

    fn candle_indexed_kernel(dtype: GgmlDType) -> Option<&'static str> {
        match dtype {
            GgmlDType::Q2K => Some("indexed_moe_forward_q2k_q8_1"),
            GgmlDType::Q3K => Some("indexed_moe_forward_q3k_q8_1"),
            GgmlDType::Q4K => Some("indexed_moe_forward_q4k_q8_1"),
            GgmlDType::Q5K => Some("indexed_moe_forward_q5k_q8_1"),
            GgmlDType::Q6K => Some("indexed_moe_forward_q6k_q8_1"),
            GgmlDType::Q8_0 => Some("indexed_moe_forward_q8_0_q8_1"),
            _ => None,
        }
    }

    impl candle_core::CustomOp2 for CandleIndexedKQuant<'_> {
        fn name(&self) -> &'static str {
            "cuda-indexed-moe-candle-k-quant"
        }

        fn cpu_fwd(
            &self,
            _input: &CpuStorage,
            _input_layout: &Layout,
            _ids: &CpuStorage,
            _ids_layout: &Layout,
        ) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("cuda-indexed-moe-candle-k-quant is only dispatched on CUDA")
        }

        fn cuda_fwd(
            &self,
            input: &CudaStorage,
            input_layout: &Layout,
            ids: &CudaStorage,
            ids_layout: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            use candle_core::cuda_backend::cudarc::driver::DevicePtr;
            use candle_core::cuda_backend::kernels::QUANTIZED;

            let (_, out_dim, in_dim) = self.weights.shape().dims3()?;
            let (batch, input_rows, input_dim) = input_layout.shape().dims3()?;
            let (id_batch, top_k) = ids_layout.shape().dims2()?;
            let Some(kernel) = candle_indexed_kernel(self.weights.dtype()) else {
                candle_core::bail!("CUDA indexed MoE does not support this expert format")
            };
            if input_dim != in_dim
                || id_batch != batch
                || (input_rows != 1 && input_rows != top_k)
                || batch == 0
                || top_k == 0
            {
                candle_core::bail!("CUDA indexed MoE input, ID and weight shapes differ")
            }
            let overflow = || candle_core::Error::Msg("CUDA indexed MoE size overflow".into());
            let total_rows = batch.checked_mul(input_rows).ok_or_else(overflow)?;
            let input_elements = total_rows.checked_mul(in_dim).ok_or_else(overflow)?;
            let routes = batch.checked_mul(top_k).ok_or_else(overflow)?;
            let output_elements = routes.checked_mul(out_dim).ok_or_else(overflow)?;
            let padded = in_dim
                .div_ceil(Q8_1_ROW_PADDING)
                .checked_mul(Q8_1_ROW_PADDING)
                .ok_or_else(overflow)?;
            // 36-byte blocks of 32 values.
            let row_bytes = padded / 32 * 36;
            let quantized_bytes = total_rows.checked_mul(row_bytes).ok_or_else(overflow)?;
            if [out_dim, in_dim, batch, top_k, padded, input_rows]
                .iter()
                .any(|value| *value > i32::MAX as usize)
            {
                return Err(overflow());
            }
            validate_storage("input", input, input_layout, DType::F32, input_elements)?;
            validate_storage("expert IDs", ids, ids_layout, DType::U32, routes)?;
            let device = input.device();
            if device.id() != ids.device().id()
                || !self
                    .weights
                    .device()
                    .same_device(&Device::Cuda(device.clone()))
            {
                candle_core::bail!("CUDA indexed MoE inputs and weights use different devices")
            }
            let input_view = checked_view_f32(
                input.as_cuda_slice::<f32>()?,
                input_layout,
                input_elements,
                "input",
            )?;
            let ids_view = checked_view_u32(
                ids.as_cuda_slice::<u32>()?,
                ids_layout,
                routes,
                "expert IDs",
            )?;
            let stream = device.cuda_stream();

            let published = if total_rows == 1 {
                let (address, _sync) = input_view.device_ptr(&stream);
                crate::architectures::q8_1_memo_cuda::lookup(address, in_dim)
            } else {
                None
            };
            let quantized = match published {
                Some(_) => None,
                None => {
                    // SAFETY: The quantization kernel writes every block of
                    // every row, padding included.
                    let quantized = unsafe { device.alloc::<u8>(quantized_bytes)? };
                    let function = device.get_or_load_func("quantize_q8_1", &QUANTIZED)?;
                    let blocks = padded.div_ceil(Q8_1_QUANTIZE_BLOCK) as u32;
                    let width = in_dim as i32;
                    let padded_width = padded as i32;
                    let mut row = 0;
                    while row < total_rows {
                        let rows = (total_rows - row).min(CUDA_MAX_GRID_Y);
                        let source = input_view.slice(row * in_dim..(row + rows) * in_dim);
                        let destination =
                            quantized.slice(row * row_bytes..(row + rows) * row_bytes);
                        let config = LaunchConfig {
                            grid_dim: (blocks, rows as u32, 1),
                            block_dim: (Q8_1_QUANTIZE_BLOCK as u32, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let mut builder = function.builder();
                        builder
                            .arg(&source)
                            .arg(&destination)
                            .arg(&width)
                            .arg(&padded_width);
                        // SAFETY: Both chunks lie inside their buffers and
                        // the grid covers exactly `rows` padded rows.
                        unsafe { builder.launch(config) }.w()?;
                        row += rows;
                    }
                    Some(quantized)
                }
            };

            let (weight_ptr, _weight_guard) = self.weights.device_ptr_with_guard(&stream)?;
            let weight_address = weight_ptr as u64;
            // SAFETY: The grid has one block per output element and each
            // block writes its element.
            let output = unsafe { device.alloc::<f32>(output_elements)? };
            let function = device.get_or_load_func(kernel, &QUANTIZED)?;
            let config = LaunchConfig {
                grid_dim: (out_dim as u32, batch as u32, top_k as u32),
                block_dim: (CUDA_WARP_SIZE as u32, CUDA_WARPS_PER_BLOCK as u32, 1),
                shared_mem_bytes: 0,
            };
            let n = out_dim as i32;
            let k = in_dim as i32;
            let batch_arg = batch as i32;
            let top_k_arg = top_k as i32;
            let padded_arg = padded as i32;
            let input_rows_arg = input_rows as i32;
            let mut builder = function.builder();
            builder.arg(&weight_address);
            match (&quantized, &published) {
                (Some(quantized), _) => {
                    builder.arg(quantized);
                }
                (None, Some(address)) => {
                    builder.arg(address);
                }
                (None, None) => unreachable!("one of the two forms was prepared above"),
            }
            builder
                .arg(&ids_view)
                .arg(&output)
                .arg(&n)
                .arg(&k)
                .arg(&batch_arg)
                .arg(&top_k_arg)
                .arg(&padded_arg)
                .arg(&input_rows_arg);
            // SAFETY: Shapes, storage bounds and device identity were
            // checked above; the weight guard and the quantized buffer live
            // until the launch is issued.
            unsafe { builder.launch(config) }.w()?;

            Ok((
                CudaStorage::wrap_cuda_slice(output, device.clone()),
                Shape::from((batch, top_k, out_dim)),
            ))
        }
    }

    /// Candle's indexed expert product for the K-quant formats the engine's
    /// own kernels do not cover. See [`CandleIndexedKQuant`].
    pub(crate) fn forward_candle_k_quant(
        input: &Tensor,
        expert_ids: &Tensor,
        weights: &QTensor,
    ) -> anyhow::Result<Tensor> {
        anyhow::ensure!(
            candle_indexed_kernel(weights.dtype()).is_some(),
            "CUDA indexed MoE does not support {:?} experts",
            weights.dtype()
        );
        input
            .apply_op2_no_bwd(expert_ids, &CandleIndexedKQuant { weights })
            .map_err(Into::into)
    }

    fn validate_storage(
        name: &str,
        storage: &CudaStorage,
        layout: &Layout,
        dtype: DType,
        elements: usize,
    ) -> candle_core::Result<()> {
        if storage.dtype() != dtype
            || !layout.is_contiguous()
            || layout.shape().elem_count() != elements
        {
            candle_core::bail!(
                "CUDA Q4_0 indexed MoE {name} must be contiguous {dtype:?} with {elements} elements"
            )
        }
        Ok(())
    }

    fn checked_view_f32<'a>(
        slice: &'a candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
        layout: &Layout,
        elements: usize,
        name: &str,
    ) -> candle_core::Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, f32>> {
        checked_view(slice, layout, elements, name)
    }

    fn checked_view_u32<'a>(
        slice: &'a candle_core::cuda_backend::cudarc::driver::CudaSlice<u32>,
        layout: &Layout,
        elements: usize,
        name: &str,
    ) -> candle_core::Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, u32>> {
        checked_view(slice, layout, elements, name)
    }

    fn checked_view<'a, T>(
        slice: &'a candle_core::cuda_backend::cudarc::driver::CudaSlice<T>,
        layout: &Layout,
        elements: usize,
        name: &str,
    ) -> candle_core::Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, T>> {
        let start = layout.start_offset();
        let end = start
            .checked_add(elements)
            .ok_or_else(|| candle_core::Error::Msg(format!("{name} CUDA view overflow")))?;
        if end > slice.len() {
            candle_core::bail!("{name} CUDA view exceeds its storage")
        }
        Ok(slice.slice(start..end))
    }

    pub(crate) fn forward(
        input: &Tensor,
        expert_ids: &Tensor,
        weights: &QTensor,
        num_experts: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> anyhow::Result<Tensor> {
        forward_impl(
            input,
            expert_ids,
            ExpertBank::Packed(weights),
            num_experts,
            out_dim,
            in_dim,
            0,
            false,
        )
    }

    /// Grouped prompt product on tensor cores. Inputs and dequantized weights
    /// are rounded to F16, so results differ slightly from [`forward`] and an
    /// input beyond the F16 range overflows. Requires more than one token and
    /// [`supports_grouped_wmma`].
    pub(crate) fn forward_grouped_wmma(
        input: &Tensor,
        expert_ids: &Tensor,
        weights: &QTensor,
        num_experts: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> anyhow::Result<Tensor> {
        forward_impl(
            input,
            expert_ids,
            ExpertBank::Packed(weights),
            num_experts,
            out_dim,
            in_dim,
            0,
            true,
        )
    }

    pub(crate) fn forward_relu_squared(
        input: &Tensor,
        expert_ids: &Tensor,
        weights: &QTensor,
        num_experts: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> anyhow::Result<Tensor> {
        forward_impl(
            input,
            expert_ids,
            ExpertBank::Packed(weights),
            num_experts,
            out_dim,
            in_dim,
            1,
            false,
        )
    }

    /// Expert product over raw GGUF bytes: `bytes` holds `[num_experts,
    /// out_dim, in_dim]` in `format` and `scales`, if any, one output scale
    /// per expert. Single tokens use the exact F32 decode kernel; prompts use
    /// the tensor-core grouped kernel where it compiled (inputs and weights
    /// are rounded to F16; NVFP4 weights are exact in it), else the grouped
    /// F32 kernel.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_raw(
        format: RawExpertFormat,
        input: &Tensor,
        expert_ids: &Tensor,
        bytes: &Tensor,
        scales: Option<&Tensor>,
        num_experts: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> anyhow::Result<Tensor> {
        anyhow::ensure!(
            in_dim.is_multiple_of(format.block()),
            "{format:?} expert rows must hold whole blocks"
        );
        let tokens = input.dim(0)?;
        forward_impl(
            input,
            expert_ids,
            ExpertBank::Raw {
                format,
                bytes,
                scales,
                dims: [num_experts, out_dim, in_dim],
            },
            num_experts,
            out_dim,
            in_dim,
            0,
            tokens > 1 && wmma_ptx().is_ok(),
        )
    }

    /// Compile and load the kernels of one raw expert format ahead of use.
    pub(crate) fn precompile_raw(format: RawExpertFormat, device: &Device) -> anyhow::Result<()> {
        if !device.is_cuda() {
            return Ok(());
        }
        let device = device.as_cuda_device()?;
        let kernels = format.kernels();
        if let Some(kernels) = kernels.f32 {
            device.get_or_load_custom_func(kernels.decode, CUDA_MODULE_NAME, ptx()?)?;
            device.get_or_load_custom_func(kernels.prompt, CUDA_MODULE_NAME, ptx()?)?;
        }
        if let Some(name) = kernels.mma {
            // Optional: without it the WMMA kernel serves prompts.
            let _ = mma_loads(device, name);
        }
        if let (Some(name), Ok(wmma)) = (kernels.wmma, wmma_ptx()) {
            device.get_or_load_custom_func(name, WMMA_MODULE_NAME, wmma)?;
            device.get_or_load_custom_func(WMMA_GATHER_KERNEL_NAME, WMMA_MODULE_NAME, wmma)?;
            device.get_or_load_custom_func(WMMA_TILES_KERNEL_NAME, WMMA_MODULE_NAME, wmma)?;
        }
        device.get_or_load_custom_func(RANK_SORT_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_impl(
        input: &Tensor,
        expert_ids: &Tensor,
        bank: ExpertBank<'_>,
        num_experts: usize,
        out_dim: usize,
        in_dim: usize,
        activation_kind: i32,
        wmma: bool,
    ) -> anyhow::Result<Tensor> {
        let (tokens, input_rows, actual_in_dim) = input.dims3()?;
        let (id_tokens, top_k) = expert_ids.dims2()?;
        anyhow::ensure!(
            actual_in_dim == in_dim,
            "CUDA Q4_0 indexed MoE input width differs"
        );
        let kernels = bank
            .kernels()
            .context("CUDA indexed MoE does not support this expert format")?;
        anyhow::ensure!(
            if wmma {
                tokens > 1 && kernels.wmma.is_some()
            } else {
                kernels.f32.is_some()
            },
            "CUDA indexed MoE has no kernel for this expert format and token count"
        );
        let geometry = Q4IndexedGeometry::with_type_size(
            kernels.type_size,
            kernels.block_elements,
            tokens,
            input_rows,
            in_dim,
            id_tokens,
            top_k,
            num_experts,
            out_dim,
        )?;
        anyhow::ensure!(
            input.dtype() == DType::F32,
            "CUDA Q4_0 indexed MoE input must be f32"
        );
        anyhow::ensure!(
            expert_ids.dtype() == DType::U32,
            "CUDA Q4_0 indexed MoE expert IDs must be u32"
        );
        anyhow::ensure!(
            input.device().is_cuda()
                && input.device().same_device(expert_ids.device())
                && bank.is_on(input.device().as_cuda_device()?),
            "CUDA Q4_0 indexed MoE inputs and weights must share one CUDA device"
        );
        anyhow::ensure!(
            bank.dims() == Some([num_experts, out_dim, in_dim])
                && bank.byte_len() == geometry.weight_bytes,
            "CUDA Q4_0 indexed MoE weight geometry differs"
        );
        if tokens == 1 {
            return input
                .apply_op2_no_bwd(
                    expert_ids,
                    &DecodeIndexedQ4 {
                        bank,
                        geometry,
                        activation_kind,
                    },
                )
                .map_err(Into::into);
        }

        // Keep route grouping on the accelerator. Sorting once lets every
        // prompt block reuse an expert weight tile across nearby routes.
        let (sorted_route_ids, sorted_experts) = sorted_routes_of(expert_ids)?;
        crate::architectures::profiling::mark_region(input.device(), "moe.sort");
        input
            .apply_op3_no_bwd(
                &sorted_route_ids,
                &sorted_experts,
                &GroupedPromptIndexedQ4 {
                    bank,
                    geometry,
                    activation_kind,
                    wmma,
                },
            )
            .map_err(Into::into)
    }
}

#[cfg(feature = "cuda")]
pub(super) use cuda::{
    RawExpertFormat, forward, forward_candle_k_quant, forward_grouped_wmma, forward_raw,
    forward_relu_squared, precompile, precompile_raw, supports, supports_f32_decode,
    supports_grouped_wmma,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_geometry_tracks_canonical_expert_stride() -> Result<()> {
        let geometry = Q4IndexedGeometry::new(7, 1, 5120, 7, 6, 128, 2688)?;
        assert_eq!(geometry.routes, 42);
        assert_eq!(geometry.input_elements, 7 * 5120);
        assert_eq!(geometry.output_elements, 42 * 2688);
        assert_eq!(geometry.weight_bytes, 128 * 2688 * (5120 / 32) * 18);
        Ok(())
    }

    #[test]
    fn q4_layout_uses_gguf_block_and_expert_strides() -> Result<()> {
        let geometry = Q4IndexedGeometry::new(3, 1, 64, 3, 2, 4, 35)?;
        let blocks_per_row = geometry.in_dim / Q4_0_BLOCK_SIZE;
        let byte_offset = |expert: usize, output_row: usize, block: usize| {
            ((expert * geometry.out_dim + output_row) * blocks_per_row + block) * Q4_0_TYPE_SIZE
        };
        assert_eq!(byte_offset(0, 0, 0), 0);
        assert_eq!(byte_offset(0, 0, 1), 18);
        assert_eq!(byte_offset(0, 1, 0), 36);
        assert_eq!(byte_offset(1, 0, 0), 35 * 2 * 18);
        assert_eq!(byte_offset(3, 34, 1) + 18, geometry.weight_bytes);
        Ok(())
    }

    #[test]
    fn grouped_prompt_tile_math_preserves_original_route_order() {
        let tokens = 10;
        let top_k = 2;
        let routes = tokens * top_k;
        let num_experts = 3;
        let in_dim = 7;
        let out_dim = 35;
        let ids = (0..tokens)
            .flat_map(|token| [2usize, token % 2])
            .collect::<Vec<_>>();
        let input = (0..routes * in_dim)
            .map(|index| (index as f32 - 30.0) / 19.0)
            .collect::<Vec<_>>();
        let weights = (0..num_experts * out_dim * in_dim)
            .map(|index| ((index * 11 % 37) as f32 - 18.0) / 23.0)
            .collect::<Vec<_>>();

        let mut expected = vec![0.0f32; routes * out_dim];
        for route in 0..routes {
            for output_row in 0..out_dim {
                for inner in 0..in_dim {
                    expected[route * out_dim + output_row] += input[route * in_dim + inner]
                        * weights[(ids[route] * out_dim + output_row) * in_dim + inner];
                }
            }
        }

        let mut sorted_routes = (0..routes).collect::<Vec<_>>();
        sorted_routes.sort_by_key(|&route| ids[route]);
        let mut actual = vec![0.0f32; routes * out_dim];
        for expert in 0..num_experts {
            let expert_routes = sorted_routes
                .iter()
                .copied()
                .filter(|&route| ids[route] == expert)
                .collect::<Vec<_>>();
            for route_tile in expert_routes.chunks(CUDA_PROMPT_ROUTE_TILE) {
                for output_start in (0..out_dim).step_by(CUDA_PROMPT_OUTPUT_TILE) {
                    for &route in route_tile {
                        for output_row in
                            output_start..(output_start + CUDA_PROMPT_OUTPUT_TILE).min(out_dim)
                        {
                            for inner in 0..in_dim {
                                actual[route * out_dim + output_row] += input
                                    [route * in_dim + inner]
                                    * weights[(expert * out_dim + output_row) * in_dim + inner];
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(actual, expected);
        assert!(ids.iter().filter(|&&expert| expert == 2).count() > CUDA_PROMPT_ROUTE_TILE);
    }

    #[test]
    fn q4_geometry_rejects_unsafe_shapes() {
        assert!(Q4IndexedGeometry::new(0, 1, 64, 0, 2, 4, 32).is_err());
        assert!(Q4IndexedGeometry::new(2, 3, 64, 2, 2, 4, 32).is_err());
        assert!(Q4IndexedGeometry::new(2, 1, 63, 2, 2, 4, 32).is_err());
        assert!(Q4IndexedGeometry::new(2, 1, 64, 2, 5, 4, 32).is_err());
        assert!(Q4IndexedGeometry::new(40_000, 1, 64, 40_000, 2, 4, 32).is_err());
    }

    #[cfg(feature = "cuda")]
    mod cuda_tests {
        use candle_core::quantized::{GgmlDType, QTensor};
        use candle_core::{Device, Tensor};

        use super::*;

        #[allow(clippy::too_many_arguments)]
        fn expected(
            input: &[f32],
            input_rows: usize,
            ids: &[u32],
            dequantized: &[Vec<Vec<f32>>],
            tokens: usize,
            top_k: usize,
            out_dim: usize,
            in_dim: usize,
            relu_squared: bool,
        ) -> Vec<f32> {
            let mut output = vec![0.0; tokens * top_k * out_dim];
            for token in 0..tokens {
                for route in 0..top_k {
                    let route_index = token * top_k + route;
                    let input_row = if input_rows == 1 { token } else { route_index };
                    let expert = ids[route_index] as usize;
                    for out in 0..out_dim {
                        output[route_index * out_dim + out] = (0..in_dim)
                            .map(|inner| {
                                let input = input[input_row * in_dim + inner];
                                let input = if relu_squared {
                                    input.max(0.0).powi(2)
                                } else {
                                    input
                                };
                                input * dequantized[expert][out][inner]
                            })
                            .sum();
                    }
                }
            }
            output
        }

        fn assert_close(actual: &[f32], expected: &[f32]) {
            assert_eq!(actual.len(), expected.len());
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                let tolerance = 2e-3 + expected.abs() * 2e-4;
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "CUDA Q4_0 mismatch at {index}: {actual} != {expected}"
                );
            }
        }

        fn test_case(
            route_specific: bool,
            offset_view: bool,
            tokens: usize,
            relu_squared: bool,
        ) -> anyhow::Result<()> {
            test_case_for(
                GgmlDType::Q4_0,
                route_specific,
                offset_view,
                tokens,
                relu_squared,
            )
        }

        fn test_case_for(
            dtype: GgmlDType,
            route_specific: bool,
            offset_view: bool,
            tokens: usize,
            relu_squared: bool,
        ) -> anyhow::Result<()> {
            test_case_with(
                dtype,
                route_specific,
                offset_view,
                tokens,
                relu_squared,
                false,
            )
        }

        /// `wmma` runs the tensor-core kernel on a wider geometry (Q4_K needs
        /// 256-element rows) and allows for its F16 rounding.
        fn test_case_with(
            dtype: GgmlDType,
            route_specific: bool,
            offset_view: bool,
            tokens: usize,
            relu_squared: bool,
            wmma: bool,
        ) -> anyhow::Result<()> {
            let Some(device) = crate::test_cuda_device() else {
                return Ok(());
            };
            let num_experts = 3;
            let top_k = 2;
            let out_dim = 35;
            let in_dim = if wmma { 512 } else { 64 };
            let input_rows = if route_specific { top_k } else { 1 };
            let weight_values = (0..num_experts * out_dim * in_dim)
                .map(|index| ((index * 13 % 41) as f32 - 20.0) / 17.0)
                .collect::<Vec<_>>();
            let weight_tensor =
                Tensor::from_vec(weight_values, (num_experts, out_dim, in_dim), &Device::Cpu)?;
            let packed = QTensor::quantize(&weight_tensor, dtype)?;
            let dequantized = packed.dequantize(&Device::Cpu)?.to_vec3::<f32>()?;
            let packed = candle_core::quantized::ggml_file::qtensor_from_ggml(
                dtype,
                packed.data()?.as_ref(),
                vec![num_experts, out_dim, in_dim],
                &device,
            )?;

            let active_input = (0..tokens * input_rows * in_dim)
                .map(|index| ((index * 7 % 31) as f32 - 15.0) / 11.0)
                .collect::<Vec<_>>();
            let ids = (0..tokens)
                .flat_map(|token| [2u32, (token % 2) as u32])
                .collect::<Vec<_>>();
            let (input, expert_ids) = if offset_view {
                let mut padded_input = vec![99.0; (tokens + 1) * input_rows * in_dim];
                padded_input[input_rows * in_dim..].copy_from_slice(&active_input);
                let input =
                    Tensor::from_vec(padded_input, (tokens + 1, input_rows, in_dim), &device)?
                        .narrow(0, 1, tokens)?;
                let mut padded_ids = vec![0u32; (tokens + 1) * top_k];
                padded_ids[top_k..].copy_from_slice(&ids);
                let expert_ids = Tensor::from_vec(padded_ids, (tokens + 1, top_k), &device)?
                    .narrow(0, 1, tokens)?;
                assert_ne!(input.layout().start_offset(), 0);
                assert_ne!(expert_ids.layout().start_offset(), 0);
                (input, expert_ids)
            } else {
                (
                    Tensor::from_vec(active_input.clone(), (tokens, input_rows, in_dim), &device)?,
                    Tensor::from_vec(ids.clone(), (tokens, top_k), &device)?,
                )
            };
            let output = if wmma {
                assert!(supports_grouped_wmma(dtype));
                forward_grouped_wmma(&input, &expert_ids, &packed, num_experts, out_dim, in_dim)?
            } else if relu_squared {
                forward_relu_squared(&input, &expert_ids, &packed, num_experts, out_dim, in_dim)?
            } else {
                forward(&input, &expert_ids, &packed, num_experts, out_dim, in_dim)?
            }
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
            let reference = expected(
                &active_input,
                input_rows,
                &ids,
                &dequantized,
                tokens,
                top_k,
                out_dim,
                in_dim,
                relu_squared,
            );
            if wmma {
                // F16 inputs and weights: about 5e-4 relative error per term.
                for (index, (actual, expected)) in output.iter().zip(&reference).enumerate() {
                    let tolerance = 6e-2 + expected.abs() * 4e-3;
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "CUDA WMMA mismatch at {index}: {actual} != {expected}"
                    );
                }
                assert_eq!(output.len(), reference.len());
            } else {
                assert_close(&output, &reference);
            }
            Ok(())
        }

        #[test]
        fn cuda_q4_decode_matches_dequantized_reference() -> anyhow::Result<()> {
            test_case(false, false, 1, false)
        }

        #[test]
        fn cuda_q4_grouped_prompt_shared_input_matches_reference() -> anyhow::Result<()> {
            test_case(false, false, 10, false)
        }

        #[test]
        fn cuda_q4_grouped_prompt_repeated_experts_and_offsets_match_reference()
        -> anyhow::Result<()> {
            test_case(true, true, 10, false)
        }

        #[test]
        fn cuda_q5_decode_matches_dequantized_reference() -> anyhow::Result<()> {
            test_case_for(GgmlDType::Q5_0, false, false, 1, false)
        }

        #[test]
        fn cuda_q8_decode_matches_dequantized_reference() -> anyhow::Result<()> {
            assert!(supports_f32_decode(GgmlDType::Q8_0) && !supports(GgmlDType::Q8_0));
            test_case_for(GgmlDType::Q8_0, false, false, 1, false)?;
            test_case_for(GgmlDType::Q8_0, true, true, 1, false)
        }

        #[test]
        fn cuda_q5_grouped_prompt_matches_reference() -> anyhow::Result<()> {
            test_case_for(GgmlDType::Q5_0, false, false, 10, false)?;
            test_case_for(GgmlDType::Q5_0, true, true, 10, true)
        }

        #[test]
        fn cuda_rank_sort_is_a_stable_sort_by_expert() -> anyhow::Result<()> {
            let Some(device) = crate::test_cuda_device() else {
                return Ok(());
            };
            for routes in [1usize, 2, 127, 128, 129, 4096] {
                let experts = (0..routes)
                    .map(|route| ((route * 7919 + 13) % 37) as u32)
                    .collect::<Vec<_>>();
                let mut expected = (0..routes as u32).collect::<Vec<_>>();
                expected.sort_by_key(|route| experts[*route as usize]);
                let (sorted_routes, sorted_experts) = super::super::cuda::sort_routes(
                    &Tensor::from_vec(experts.clone(), routes, &device)?,
                )?;
                assert_eq!(sorted_routes.to_vec1::<u32>()?, expected, "routes {routes}");
                assert_eq!(
                    sorted_experts.to_vec1::<u32>()?,
                    expected
                        .iter()
                        .map(|route| experts[*route as usize])
                        .collect::<Vec<_>>()
                );
            }
            Ok(())
        }

        #[test]
        fn cuda_wmma_grouped_prompt_matches_reference() -> anyhow::Result<()> {
            for dtype in [
                GgmlDType::Q5_0,
                GgmlDType::Q8_0,
                GgmlDType::Q4K,
                GgmlDType::Q5K,
            ] {
                // 70 tokens give one expert more than two 32-route tiles.
                test_case_with(dtype, false, false, 70, false, true)?;
                // 300 tokens give each expert more than one 128-route tile, the
                // last one partial.
                test_case_with(dtype, false, false, 300, false, true)?;
                test_case_with(dtype, true, true, 10, false, true)?;
            }
            Ok(())
        }

        #[test]
        fn cuda_q4_decode_relu_squared_matches_reference() -> anyhow::Result<()> {
            test_case(false, false, 1, true)
        }

        #[test]
        fn cuda_q4_grouped_prompt_relu_squared_matches_reference() -> anyhow::Result<()> {
            test_case(true, true, 10, true)
        }
    }
}
