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

#[cfg(feature = "cuda")]
mod cuda {
    use std::{path::PathBuf, sync::OnceLock};

    use anyhow::Context as _;
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::WrapErr;
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{CpuStorage, CudaStorage, D, DType, Device, Layout, Shape, Tensor};

    use super::{
        CUDA_PROMPT_OUTPUT_TILE, CUDA_PROMPT_ROUTE_TILE, CUDA_WARP_SIZE, CUDA_WARPS_PER_BLOCK,
        Q4IndexedGeometry,
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
        type_size: usize,
        block_elements: usize,
    }

    const Q4_0_KERNELS: QuantKernels = QuantKernels {
        f32: Some(F32Kernels {
            decode: "dinf_indexed_moe_q4_0_f32",
            prompt: "dinf_indexed_moe_q4_0_grouped_f32",
        }),
        wmma: None,
        type_size: super::Q4_0_TYPE_SIZE,
        block_elements: super::Q4_0_BLOCK_SIZE,
    };
    const Q5_0_KERNELS: QuantKernels = QuantKernels {
        f32: Some(F32Kernels {
            decode: "dinf_indexed_moe_q5_0_f32",
            prompt: "dinf_indexed_moe_q5_0_grouped_f32",
        }),
        wmma: Some("dinf_indexed_moe_q5_0_grouped_wmma"),
        type_size: super::Q5_0_TYPE_SIZE,
        block_elements: super::Q4_0_BLOCK_SIZE,
    };
    const Q4_K_KERNELS: QuantKernels = QuantKernels {
        f32: None,
        wmma: Some("dinf_indexed_moe_q4_k_grouped_wmma"),
        type_size: super::Q4_K_TYPE_SIZE,
        block_elements: super::Q4_K_BLOCK_SIZE,
    };

    const Q8_0_KERNELS: QuantKernels = QuantKernels {
        f32: Some(F32Kernels {
            decode: "dinf_indexed_moe_q8_0_f32",
            prompt: "dinf_indexed_moe_q8_0_grouped_f32",
        }),
        wmma: Some("dinf_indexed_moe_q8_0_grouped_wmma"),
        type_size: super::Q8_0_TYPE_SIZE,
        block_elements: super::Q4_0_BLOCK_SIZE,
    };

    fn quant_kernels(dtype: GgmlDType) -> Option<QuantKernels> {
        match dtype {
            GgmlDType::Q4_0 => Some(Q4_0_KERNELS),
            GgmlDType::Q5_0 => Some(Q5_0_KERNELS),
            GgmlDType::Q8_0 => Some(Q8_0_KERNELS),
            GgmlDType::Q4K => Some(Q4_K_KERNELS),
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
    const WMMA_SOURCE: &str = r#"
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
// GGUF block byte size: 22 is Q5_0, 34 is Q8_0, 144 is Q4_K. One header decode
// serves the whole run.
template <int FORMAT>
__device__ __forceinline__ void dinf_tile_weights8(
    const unsigned char *weights,
    unsigned long long expert_row,
    int step,
    int in_dim,
    int inner0,
    __half *out) {
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
    int num_experts) {
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
                            output_tile[warp][row][column];
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
    int routes, int in_dim, int out_dim, int num_experts) { \
    dinf_indexed_moe_grouped_wmma<FORMAT>(gathered, sorted_route_ids, tiles, \
        weights, output, routes, in_dim, out_dim, num_experts); \
}

DINF_WMMA_ENTRY(dinf_indexed_moe_q5_0_grouped_wmma, 22)
DINF_WMMA_ENTRY(dinf_indexed_moe_q8_0_grouped_wmma, 34)
DINF_WMMA_ENTRY(dinf_indexed_moe_q4_k_grouped_wmma, 144)
"#;

    // Each warp computes one output row. Four warps share each input tile,
    // while Q4_0 weights stay in their canonical 18-byte GGUF blocks.
    const CUDA_SOURCE: &str = r#"
#include <cuda_fp16.h>

#define DINF_Q4_BLOCK_SIZE 32
#define DINF_WARP_SIZE 32

// One dequantized weight from a 32-element GGUF block. TYPE_SIZE 18 is Q4_0
// (scale, 16 nibble bytes); 22 is Q5_0 (scale, 32 fifth bits, 16 nibble bytes);
// 34 is Q8_0 (scale, 32 signed bytes).
template <int TYPE_SIZE>
__device__ __forceinline__ float dinf_block_weight(const unsigned char *packed, int inner) {
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
    int activation_kind) {
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
        + expert_row * blocks_per_row * TYPE_SIZE;

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
            const unsigned char *packed =
                weight_row + (unsigned long long) block * TYPE_SIZE;
            sum = fmaf(value, dinf_block_weight<TYPE_SIZE>(packed, lane), sum);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0 && valid_output) {
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
    int activation_kind) {
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
                const unsigned char *packed = weights
                    + (expert_row * blocks_per_row + block) * TYPE_SIZE;
                weight_tile[tile_row][inner] = valid_weight
                    ? dinf_block_weight<TYPE_SIZE>(packed, inner)
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
    int num_experts, int activation_kind) { \
    dinf_indexed_moe_decode<TYPE_SIZE>(input, expert_ids, weights, output, routes, top_k, \
        input_rows, in_dim, out_dim, num_experts, activation_kind); \
}

#define DINF_GROUPED_ENTRY(NAME, TYPE_SIZE) \
extern "C" __global__ void NAME( \
    const float *input, const unsigned int *sorted_route_ids, \
    const unsigned int *sorted_experts, const unsigned char *weights, float *output, \
    int routes, int top_k, int input_rows, int in_dim, int out_dim, int num_experts, \
    int activation_kind) { \
    dinf_indexed_moe_grouped<TYPE_SIZE>(input, sorted_route_ids, sorted_experts, weights, \
        output, routes, top_k, input_rows, in_dim, out_dim, num_experts, activation_kind); \
}

DINF_DECODE_ENTRY(dinf_indexed_moe_q4_0_f32, 18)
DINF_DECODE_ENTRY(dinf_indexed_moe_q5_0_f32, 22)
DINF_DECODE_ENTRY(dinf_indexed_moe_q8_0_f32, 34)
DINF_GROUPED_ENTRY(dinf_indexed_moe_q4_0_grouped_f32, 18)
DINF_GROUPED_ENTRY(dinf_indexed_moe_q5_0_grouped_f32, 22)
DINF_GROUPED_ENTRY(dinf_indexed_moe_q8_0_grouped_f32, 34)
"#;

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
        weights: &'a QTensor,
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
            let Some(kernels) = quant_kernels(self.weights.dtype()) else {
                candle_core::bail!("CUDA indexed MoE does not support this expert format")
            };
            if kernels.type_size != geometry.type_size
                || self.weights.shape().dims()
                    != [geometry.num_experts, geometry.out_dim, geometry.in_dim]
                || self.weights.storage_size_in_bytes() != geometry.weight_bytes
                || !self
                    .weights
                    .device()
                    .same_device(&Device::Cuda(input.device().clone()))
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
            let (weight_ptr, _weight_guard) = self.weights.device_ptr_with_guard(&stream)?;
            let weight_address = weight_ptr as u64;
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
                .arg(&activation_kind);
            // SAFETY: Geometry, storage bounds, device identity, and all kernel
            // index limits were checked before this launch.
            unsafe { builder.launch(config) }.w()?;

            Ok((
                CudaStorage::wrap_cuda_slice(output, device.clone()),
                Shape::from((geometry.tokens, geometry.top_k, geometry.out_dim)),
            ))
        }
    }

    struct GroupedPromptIndexedQ4<'a> {
        weights: &'a QTensor,
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
            let Some(kernels) = quant_kernels(self.weights.dtype()) else {
                candle_core::bail!("CUDA indexed MoE does not support this expert format")
            };
            if kernels.type_size != geometry.type_size
                || self.weights.shape().dims()
                    != [geometry.num_experts, geometry.out_dim, geometry.in_dim]
                || self.weights.storage_size_in_bytes() != geometry.weight_bytes
                || !self
                    .weights
                    .device()
                    .same_device(&Device::Cuda(input.device().clone()))
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
            let (weight_ptr, _weight_guard) = self.weights.device_ptr_with_guard(&stream)?;
            let weight_address = weight_ptr as u64;
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

                let function =
                    device.get_or_load_custom_func(name, WMMA_MODULE_NAME, wmma_ptx()?)?;
                let grid_dim = (
                    geometry.out_dim.div_ceil(WMMA_OUTPUT_TILE) as u32,
                    max_tiles as u32,
                    1,
                );
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
                    .arg(&num_experts);
                // SAFETY: The sort outputs, the tile table, the gathered rows
                // and the weights stay on this stream; geometry, storage
                // bounds and launch limits were checked. The kernel rejects
                // a tile outside the routes and reads at most one 32-route
                // step past a tile, which the padding covers.
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim,
                        block_dim: (CUDA_WARP_SIZE as u32, WMMA_WARPS as u32, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .w()?;
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
                .arg(&activation_kind);
            // SAFETY: The sort outputs and all storage views stay on this
            // stream. Geometry, storage bounds, and launch limits were checked.
            unsafe { builder.launch(config) }.w()?;

            Ok((
                CudaStorage::wrap_cuda_slice(output, device.clone()),
                Shape::from((geometry.tokens, geometry.top_k, geometry.out_dim)),
            ))
        }
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
            weights,
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
            weights,
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
            weights,
            num_experts,
            out_dim,
            in_dim,
            1,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_impl(
        input: &Tensor,
        expert_ids: &Tensor,
        weights: &QTensor,
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
        let kernels = quant_kernels(weights.dtype())
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
                && input.device().same_device(&weights.device()),
            "CUDA Q4_0 indexed MoE inputs and weights must share one CUDA device"
        );
        anyhow::ensure!(
            weights.shape().dims() == [num_experts, out_dim, in_dim]
                && weights.storage_size_in_bytes() == geometry.weight_bytes,
            "CUDA Q4_0 indexed MoE weight geometry differs"
        );
        if tokens == 1 {
            return input
                .apply_op2_no_bwd(
                    expert_ids,
                    &DecodeIndexedQ4 {
                        weights,
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
                    weights,
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
    forward, forward_grouped_wmma, forward_relu_squared, precompile, supports, supports_f32_decode,
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
            for dtype in [GgmlDType::Q5_0, GgmlDType::Q8_0, GgmlDType::Q4K] {
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
