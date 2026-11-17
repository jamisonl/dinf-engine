//! CUDA fast paths for dense quantized projections that share activations.
//!
//! Candle's fast GGUF MMVQ path quantizes the activation to Q8_1 before every
//! quantized matrix-vector product. Attention Q/K/V and SwiGLU gate/up all
//! consume the same activation, so decode can quantize once and fuse compatible
//! Q8_0, Q4_0, Q2_K, and Q4_K grids. Prompt MMQ similarly reuses one tiled
//! Q8_1 activation per required scale layout across the projections. [`forward`]
//! preserves Candle's output shapes and arithmetic.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use crate::qmatmul::QMatMul;
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use candle_core::cuda_backend::cudarc::driver::{CudaSlice, CudaView, DevicePtr};
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::cudarc::driver::{result as cuda_result, sys as cuda_sys};
use candle_core::cuda_backend::cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use candle_core::cuda_backend::{DeviceId, kernels};
use candle_core::op::BackpropOp;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{CpuStorage, CudaStorage, DType, Device, Layout, Shape, Storage, Tensor};

use super::qkv_decode_cuda::{QkvDecodeEpilogue, QkvDecodeOutput};

const MATRIX_ROW_PADDING: usize = 512;
const Q8_1_BLOCK_SIZE: usize = 32;
const Q8_1_TYPE_SIZE: usize = 36;
const CUDA_QUANTIZE_BLOCK_SIZE: usize = 256;
const SILU_QUANT_MODULE_NAME: &str = "dinf_dense_silu_quant_q8_1_v1";
const SILU_QUANT_KERNEL_NAME: &str = "dinf_silu_mul_quant_q8_1_f32";
const RELU_SQUARED_QUANT_KERNEL_NAME: &str = "dinf_relu_squared_quant_q8_1_f32";
const Q8_FUSION_MODULE_NAME: &str = "dinf_dense_q8_0_fusion_v1";
const Q8_GROUP_KERNEL_NAME: &str = "dinf_q8_0_group3_mmvq_f32";
const Q8_SWIGLU_KERNEL_NAME: &str = "dinf_q8_0_swiglu_mmvq_f32";
const KQUANT_FUSION_MODULE_NAME: &str = "dinf_dense_kquant_fusion_v1";
const Q4_0_GROUP_KERNEL_NAME: &str = "dinf_q4_0_group3_mmvq_f32";
const Q4_0_GATED_KERNEL_NAME: &str = "dinf_q4_0_gated_mmvq_f32";
const Q2_K_GROUP_KERNEL_NAME: &str = "dinf_q2_k_group3_mmvq_f32";
const Q2_K_GATED_KERNEL_NAME: &str = "dinf_q2_k_gated_mmvq_f32";
const Q4_K_GROUP_KERNEL_NAME: &str = "dinf_q4_k_group3_mmvq_f32";
const Q4_K_GATED_KERNEL_NAME: &str = "dinf_q4_k_gated_mmvq_f32";
const Q2_K_INDEXED_GATED_KERNEL_NAME: &str = "dinf_q2_k_indexed_gated_mmvq_f32";
const Q4_K_INDEXED_GATED_KERNEL_NAME: &str = "dinf_q4_k_indexed_gated_mmvq_f32";
const MMQ_Q8_1_BLOCK_SIZE: usize = 4 * Q8_1_BLOCK_SIZE;
const MMQ_Q8_1_TYPE_SIZE: usize = 144;
const MMQ_FIXUP_X_MAX: usize = 128;
const MMQ_FIXUP_Y_MAX: usize = 128;
const MMQ_FIXUP_MAX_SMS: usize = 256;

const SILU_QUANT_CUDA_SOURCE: &str = r#"
#define DINF_WARP_SIZE 32
#define DINF_QK8_1 32

typedef struct {
    unsigned short ds[2];
    signed char qs[DINF_QK8_1];
} dinf_block_q8_1;

static __device__ __forceinline__ unsigned short dinf_f32_to_f16(float value) {
    unsigned short result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(value));
    return result;
}

static __device__ __forceinline__ float dinf_warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        value += __shfl_xor_sync(0xffffffff, value, mask, DINF_WARP_SIZE);
    }
    return value;
}

static __device__ __forceinline__ float dinf_warp_max(float value) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        value = fmaxf(
            value,
            __shfl_xor_sync(0xffffffff, value, mask, DINF_WARP_SIZE));
    }
    return value;
}

extern "C" __global__ void dinf_silu_mul_quant_q8_1_f32(
    const float *gate,
    const float *up,
    void *quantized,
    int input_dim,
    int input_dim_padded) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= input_dim_padded) {
        return;
    }

    float value = 0.0f;
    if (index < input_dim) {
        const float gate_value = gate[index];
        // Preserve Candle's SiLU output and following multiply as two f32
        // rounding boundaries before feeding the stock Q8_1 quantizer math.
        const float denominator = __fadd_rn(1.0f, expf(-gate_value));
        const float silu = __fdiv_rn(gate_value, denominator);
        value = __fmul_rn(silu, up[index]);
    }

    float amax = dinf_warp_max(fabsf(value));
    float sum = dinf_warp_sum(value);
    const float scale = amax / 127.0f;
    const signed char quantized_value =
        amax == 0.0f ? 0 : (signed char) roundf(value / scale);

    dinf_block_q8_1 *blocks = (dinf_block_q8_1 *) quantized;
    const int block = index / DINF_QK8_1;
    const int lane = index % DINF_QK8_1;
    blocks[block].qs[lane] = quantized_value;
    if (lane == 0) {
        blocks[block].ds[0] = dinf_f32_to_f16(scale);
        blocks[block].ds[1] = dinf_f32_to_f16(sum);
    }
}

extern "C" __global__ void dinf_relu_squared_quant_q8_1_f32(
    const float *input,
    void *quantized,
    int input_dim,
    int input_dim_padded) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= input_dim_padded) {
        return;
    }

    float value = 0.0f;
    if (index < input_dim) {
        const float positive = fmaxf(input[index], 0.0f);
        value = __fmul_rn(positive, positive);
    }
    float amax = dinf_warp_max(fabsf(value));
    float sum = dinf_warp_sum(value);
    const float scale = amax / 127.0f;
    const signed char quantized_value =
        amax == 0.0f ? 0 : (signed char) roundf(value / scale);

    dinf_block_q8_1 *blocks = (dinf_block_q8_1 *) quantized;
    const int block = index / DINF_QK8_1;
    const int lane = index % DINF_QK8_1;
    blocks[block].qs[lane] = quantized_value;
    if (lane == 0) {
        blocks[block].ds[0] = dinf_f32_to_f16(scale);
        blocks[block].ds[1] = dinf_f32_to_f16(sum);
    }
}
"#;

// Q8_0 decode is bandwidth and launch-latency bound.  The stock MMVQ kernel
// launches once per tensor, even when projections share the same Q8_1 input.
// These kernels retain llama.cpp's four-warp reduction order while selecting
// Q/K/V rows inside one grid and evaluating gate/up together inside one grid.
const Q8_FUSION_CUDA_SOURCE: &str = r#"
#define DINF_WARP_SIZE 32
#define DINF_QK8 32
#define DINF_Q8_0_BYTES 34
#define DINF_Q8_1_BYTES 36

static __device__ __forceinline__ float dinf_q8_half_to_f32(unsigned short value) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(value));
    return result;
}

static __device__ __forceinline__ int dinf_q8_load_i32_unaligned(
    const signed char *values, int byte_offset) {
    const unsigned short *halves =
        (const unsigned short *)(values + byte_offset);
    return (int)((unsigned int)halves[0] | ((unsigned int)halves[1] << 16));
}

static __device__ __forceinline__ int dinf_q8_dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;"
        : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
#else
    const signed char *av = (const signed char *)&a;
    const signed char *bv = (const signed char *)&b;
    return c + av[0] * bv[0] + av[1] * bv[1] +
        av[2] * bv[2] + av[3] * bv[3];
#endif
}

static __device__ __forceinline__ float dinf_q8_warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        value += __shfl_xor_sync(0xffffffff, value, mask, DINF_WARP_SIZE);
    }
    return value;
}

static __device__ __forceinline__ float dinf_q8_partial_dot(
    const unsigned char *weights,
    const unsigned char *activation,
    int row,
    int input_dim) {
    const int tid = DINF_WARP_SIZE * threadIdx.y + threadIdx.x;
    const int blocks_per_row = input_dim / DINF_QK8;
    float sum = 0.0f;

    // Four adjacent threads cover one 32-element quant block.  Each thread
    // issues two dp4a operations, matching the stock Q8_0/Q8_1 MMVQ mapping.
    for (int block = tid / 4; block < blocks_per_row; block += 32) {
        const int chunk = tid & 3;
        const unsigned char *weight_block =
            weights + (row * blocks_per_row + block) * DINF_Q8_0_BYTES;
        const unsigned char *input_block =
            activation + block * DINF_Q8_1_BYTES;
        const signed char *weight_qs = (const signed char *)(weight_block + 2);
        const int *input_qs = (const int *)(input_block + 4);
        int integer_sum = 0;
        integer_sum = dinf_q8_dp4a(
            dinf_q8_load_i32_unaligned(weight_qs, chunk * 8),
            input_qs[chunk * 2], integer_sum);
        integer_sum = dinf_q8_dp4a(
            dinf_q8_load_i32_unaligned(weight_qs, chunk * 8 + 4),
            input_qs[chunk * 2 + 1], integer_sum);
        sum += (float)integer_sum * dinf_q8_half_to_f32(
            *(const unsigned short *)weight_block) * dinf_q8_half_to_f32(
            *(const unsigned short *)input_block);
    }
    return sum;
}

extern "C" __global__ void dinf_q8_0_group3_mmvq_f32(
    unsigned long long weight0_address,
    unsigned long long weight1_address,
    unsigned long long weight2_address,
    const void *activation_ptr,
    float *output,
    int input_dim,
    int rows0,
    int rows1,
    int rows2) {
    const int output_row = (int)blockIdx.x;
    const int rows01 = rows0 + rows1;
    const unsigned char *weights;
    int weight_row;
    if (output_row < rows0) {
        weights = (const unsigned char *)weight0_address;
        weight_row = output_row;
    } else if (output_row < rows01) {
        weights = (const unsigned char *)weight1_address;
        weight_row = output_row - rows0;
    } else {
        weights = (const unsigned char *)weight2_address;
        weight_row = output_row - rows01;
    }

    const unsigned char *activation = (const unsigned char *)activation_ptr;
    float value = dinf_q8_partial_dot(weights, activation, weight_row, input_dim);
    __shared__ float partial[3][DINF_WARP_SIZE];
    if (threadIdx.y > 0) {
        partial[threadIdx.y - 1][threadIdx.x] = value;
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
#pragma unroll
    for (int warp = 0; warp < 3; ++warp) {
        value += partial[warp][threadIdx.x];
    }
    value = dinf_q8_warp_sum(value);
    if (threadIdx.x == 0 && output_row < rows01 + rows2) {
        output[output_row] = value;
    }
}

extern "C" __global__ void dinf_q8_0_swiglu_mmvq_f32(
    unsigned long long gate_address,
    unsigned long long up_address,
    const void *activation_ptr,
    float *output,
    int input_dim,
    int output_dim) {
    const int output_row = (int)blockIdx.x;
    const unsigned char *activation = (const unsigned char *)activation_ptr;
    const float gate_partial = dinf_q8_partial_dot(
        (const unsigned char *)gate_address, activation, output_row, input_dim);
    const float up_partial = dinf_q8_partial_dot(
        (const unsigned char *)up_address, activation, output_row, input_dim);
    __shared__ float partial_gate[3][DINF_WARP_SIZE];
    __shared__ float partial_up[3][DINF_WARP_SIZE];
    float gate = gate_partial;
    float up = up_partial;
    if (threadIdx.y > 0) {
        partial_gate[threadIdx.y - 1][threadIdx.x] = gate;
        partial_up[threadIdx.y - 1][threadIdx.x] = up;
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
#pragma unroll
    for (int warp = 0; warp < 3; ++warp) {
        gate += partial_gate[warp][threadIdx.x];
        up += partial_up[warp][threadIdx.x];
    }
    gate = dinf_q8_warp_sum(gate);
    up = dinf_q8_warp_sum(up);
    if (threadIdx.x == 0 && output_row < output_dim) {
        const float denominator = __fadd_rn(1.0f, expf(-gate));
        const float silu = __fdiv_rn(gate, denominator);
        output[output_row] = __fmul_rn(silu, up);
    }
}
"#;

// Launch-fused decode kernels for the supported non-Q8 quantizations. The
// vector-dot mappings and four-warp reduction order match Candle's GGUF MMVQ
// kernels. Keeping canonical GGUF blocks means model loading stays zero-copy.
const KQUANT_FUSION_CUDA_SOURCE: &str = r#"
#define DINF_WARP_SIZE 32
#define DINF_QK8_1 32
#define DINF_QK_K 256

typedef struct {
    unsigned int ds;
    signed char qs[32];
} dinf_block_q8_1;

typedef struct {
    unsigned short d;
    unsigned char qs[16];
} dinf_block_q4_0;

typedef struct {
    unsigned char scales[16];
    unsigned char qs[64];
    unsigned int dm;
} dinf_block_q2_k;

typedef struct {
    unsigned int dm;
    unsigned char scales[12];
    unsigned char qs[128];
} dinf_block_q4_k;

static __device__ __forceinline__ float dinf_half_to_f32(unsigned short value) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(value));
    return result;
}

static __device__ __forceinline__ float dinf_half2_low(unsigned int value) {
    return dinf_half_to_f32((unsigned short)value);
}

static __device__ __forceinline__ float dinf_half2_high(unsigned int value) {
    return dinf_half_to_f32((unsigned short)(value >> 16));
}

static __device__ __forceinline__ int dinf_load_u32_unaligned(
    const unsigned char *values, int word) {
    const unsigned short *halves =
        (const unsigned short *)(values + 4 * word);
    return (int)((unsigned int)halves[0] | ((unsigned int)halves[1] << 16));
}

static __device__ __forceinline__ int dinf_load_i32_aligned(
    const signed char *values, int word) {
    return ((const int *)values)[word];
}

static __device__ __forceinline__ int dinf_load_u32_aligned(
    const unsigned char *values, int word) {
    return ((const int *)values)[word];
}

static __device__ __forceinline__ int dinf_dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;"
        : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
#else
    const signed char *av = (const signed char *)&a;
    const signed char *bv = (const signed char *)&b;
    return c + av[0] * bv[0] + av[1] * bv[1] +
        av[2] * bv[2] + av[3] * bv[3];
#endif
}

static __device__ __forceinline__ float dinf_warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        value += __shfl_xor_sync(0xffffffff, value, mask, DINF_WARP_SIZE);
    }
    return value;
}

static __device__ __forceinline__ float dinf_q4_0_partial_dot(
    const void *weight_base,
    const dinf_block_q8_1 *activation,
    int row,
    int input_dim) {
    const dinf_block_q4_0 *weights = (const dinf_block_q4_0 *)weight_base;
    const int tid = DINF_WARP_SIZE * threadIdx.y + threadIdx.x;
    const int blocks_per_row = input_dim / 32;
    float sum = 0.0f;
    for (int block = tid / 2; block < blocks_per_row; block += 64) {
        const int iqs = 2 * (tid & 1);
        const dinf_block_q4_0 *weight = weights + row * blocks_per_row + block;
        const dinf_block_q8_1 *input = activation + block;
        int integer_sum = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int packed = dinf_load_u32_unaligned(weight->qs, iqs + i);
            const int low = packed & 0x0f0f0f0f;
            const int high = (packed >> 4) & 0x0f0f0f0f;
            integer_sum = dinf_dp4a(
                low, dinf_load_i32_aligned(input->qs, iqs + i), integer_sum);
            integer_sum = dinf_dp4a(
                high, dinf_load_i32_aligned(input->qs, iqs + i + 4), integer_sum);
        }
        const float d8 = dinf_half2_low(input->ds);
        const float s8 = dinf_half2_high(input->ds);
        sum += dinf_half_to_f32(weight->d) * ((float)integer_sum * d8 - 4.0f * s8);
    }
    return sum;
}

static __device__ __forceinline__ float dinf_q2_k_partial_dot(
    const void *weight_base,
    const dinf_block_q8_1 *activation,
    int row,
    int input_dim) {
    const dinf_block_q2_k *weights = (const dinf_block_q2_k *)weight_base;
    const int tid = DINF_WARP_SIZE * threadIdx.y + threadIdx.x;
    const int blocks_per_row = input_dim / DINF_QK_K;
    float sum = 0.0f;
    for (int block = tid / 16; block < blocks_per_row; block += 8) {
        const int iqs = tid & 15;
        const dinf_block_q2_k *weight = weights + row * blocks_per_row + block;
        const dinf_block_q8_1 *input = activation + block * 8;
        const int input_offset = 4 * (iqs / 8);
        const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
        const unsigned char *scales = weight->scales + scale_offset;
        const int packed = dinf_load_u32_aligned(weight->qs, iqs);
        float sum_d = 0.0f;
        float sum_m = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const dinf_block_q8_1 *input_block = input + input_offset + i;
            const int q8 = dinf_load_i32_aligned(input_block->qs, iqs % 8);
            const float d8 = dinf_half2_low(input_block->ds);
            const int scale = scales[2 * i];
            const int q2 = (packed >> (2 * i)) & 0x03030303;
            sum_d += d8 * (float)(dinf_dp4a(q2, q8, 0) * (scale & 15));
            int minimum = scale >> 4;
            minimum |= minimum << 8;
            minimum |= minimum << 16;
            sum_m += d8 * (float)dinf_dp4a(minimum, q8, 0);
        }
        sum += dinf_half2_low(weight->dm) * sum_d
            - dinf_half2_high(weight->dm) * sum_m;
    }
    return sum;
}

static __device__ __forceinline__ float dinf_q4_k_partial_dot(
    const void *weight_base,
    const dinf_block_q8_1 *activation,
    int row,
    int input_dim) {
    const dinf_block_q4_k *weights = (const dinf_block_q4_k *)weight_base;
    const int tid = DINF_WARP_SIZE * threadIdx.y + threadIdx.x;
    const int blocks_per_row = input_dim / DINF_QK_K;
    float sum = 0.0f;
    for (int block = tid / 16; block < blocks_per_row; block += 8) {
        const int iqs = 2 * (tid & 15);
        const dinf_block_q4_k *weight = weights + row * blocks_per_row + block;
        const dinf_block_q8_1 *input = activation + block * 8;
        const int input_offset = 2 * ((iqs / 2) / 4);
        const int q4_word = (iqs / 2) % 4;
        const int *q4 = (const int *)(weight->qs + 16 * input_offset + 4 * q4_word);
        const int packed0 = q4[0];
        const int packed1 = q4[4];
        const unsigned short *scales = (const unsigned short *)weight->scales;
        unsigned short auxiliary[2];
        const int j = input_offset / 2;
        if (j < 2) {
            auxiliary[0] = scales[j] & 0x3f3f;
            auxiliary[1] = scales[j + 2] & 0x3f3f;
        } else {
            auxiliary[0] = ((scales[j + 2] >> 0) & 0x0f0f)
                | ((scales[j - 2] & 0xc0c0) >> 2);
            auxiliary[1] = ((scales[j + 2] >> 4) & 0x0f0f)
                | ((scales[j] & 0xc0c0) >> 2);
        }
        const unsigned char *scale = (const unsigned char *)auxiliary;
        const unsigned char *minimum = scale + 2;
        float sum_d = 0.0f;
        float sum_m = 0.0f;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const dinf_block_q8_1 *input_block = input + input_offset + i;
            const int *q8 = (const int *)input_block->qs + q4_word;
            const int q0 = i == 0 ? packed0 : packed0 >> 4;
            const int q1 = i == 0 ? packed1 : packed1 >> 4;
            const int low0 = q0 & 0x0f0f0f0f;
            const int low1 = q1 & 0x0f0f0f0f;
            const int dot = dinf_dp4a(low1, q8[4], dinf_dp4a(low0, q8[0], 0));
            const int sum_q8 = dinf_dp4a(
                0x01010101, q8[4], dinf_dp4a(0x01010101, q8[0], 0));
            const float d8 = dinf_half2_low(input_block->ds);
            sum_d += d8 * (float)(dot * scale[i]);
            sum_m += d8 * (float)(sum_q8 * minimum[i]);
        }
        sum += dinf_half2_low(weight->dm) * sum_d
            - dinf_half2_high(weight->dm) * sum_m;
    }
    return sum;
}

template<int KIND>
static __device__ __forceinline__ float dinf_partial_dot(
    const void *weights,
    const dinf_block_q8_1 *activation,
    int row,
    int input_dim);

template<>
__device__ __forceinline__ float dinf_partial_dot<0>(
    const void *weights, const dinf_block_q8_1 *activation, int row, int input_dim) {
    return dinf_q4_0_partial_dot(weights, activation, row, input_dim);
}

template<>
__device__ __forceinline__ float dinf_partial_dot<1>(
    const void *weights, const dinf_block_q8_1 *activation, int row, int input_dim) {
    return dinf_q2_k_partial_dot(weights, activation, row, input_dim);
}

template<>
__device__ __forceinline__ float dinf_partial_dot<2>(
    const void *weights, const dinf_block_q8_1 *activation, int row, int input_dim) {
    return dinf_q4_k_partial_dot(weights, activation, row, input_dim);
}

template<int KIND>
static __device__ __forceinline__ void dinf_group3_body(
    unsigned long long weight0_address,
    unsigned long long weight1_address,
    unsigned long long weight2_address,
    const void *activation_ptr,
    float *output,
    int input_dim,
    int rows0,
    int rows1,
    int rows2) {
    const int output_row = (int)blockIdx.x;
    const int rows01 = rows0 + rows1;
    const void *weights;
    int weight_row;
    if (output_row < rows0) {
        weights = (const void *)weight0_address;
        weight_row = output_row;
    } else if (output_row < rows01) {
        weights = (const void *)weight1_address;
        weight_row = output_row - rows0;
    } else {
        weights = (const void *)weight2_address;
        weight_row = output_row - rows01;
    }
    float value = dinf_partial_dot<KIND>(
        weights, (const dinf_block_q8_1 *)activation_ptr, weight_row, input_dim);
    __shared__ float partial[3][DINF_WARP_SIZE];
    if (threadIdx.y > 0) {
        partial[threadIdx.y - 1][threadIdx.x] = value;
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
#pragma unroll
    for (int warp = 0; warp < 3; ++warp) {
        value += partial[warp][threadIdx.x];
    }
    value = dinf_warp_sum(value);
    if (threadIdx.x == 0 && output_row < rows01 + rows2) {
        output[output_row] = value;
    }
}

template<int KIND>
static __device__ __forceinline__ void dinf_gated_body(
    unsigned long long gate_address,
    unsigned long long up_address,
    const void *activation_ptr,
    float *output,
    int input_dim,
    int output_dim,
    int activation_kind) {
    const int output_row = (int)blockIdx.x;
    const dinf_block_q8_1 *activation = (const dinf_block_q8_1 *)activation_ptr;
    float gate = dinf_partial_dot<KIND>(
        (const void *)gate_address, activation, output_row, input_dim);
    float up = dinf_partial_dot<KIND>(
        (const void *)up_address, activation, output_row, input_dim);
    __shared__ float partial_gate[3][DINF_WARP_SIZE];
    __shared__ float partial_up[3][DINF_WARP_SIZE];
    if (threadIdx.y > 0) {
        partial_gate[threadIdx.y - 1][threadIdx.x] = gate;
        partial_up[threadIdx.y - 1][threadIdx.x] = up;
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
#pragma unroll
    for (int warp = 0; warp < 3; ++warp) {
        gate += partial_gate[warp][threadIdx.x];
        up += partial_up[warp][threadIdx.x];
    }
    gate = dinf_warp_sum(gate);
    up = dinf_warp_sum(up);
    if (threadIdx.x == 0 && output_row < output_dim) {
        float activated;
        if (activation_kind == 1) {
            const float squared = gate * gate;
            const float cubed = squared * gate;
            const float alpha = gate + 0.044715f * cubed;
            activated = 0.5f * gate
                * (1.0f + tanhf(0.7978845608028654f * alpha));
        } else {
            activated = gate / (1.0f + expf(-gate));
        }
        output[output_row] = activated * up;
    }
}

template<int KIND>
static __device__ __forceinline__ void dinf_indexed_gated_body(
    unsigned long long gate_address,
    unsigned long long up_address,
    const void *activation_ptr,
    const unsigned int *expert_ids,
    float *output,
    int input_dim,
    int output_dim,
    int num_experts,
    int top_k,
    int input_rows,
    int activation_stride) {
    const int output_row = (int)blockIdx.x;
    const int route = (int)blockIdx.y;
    const unsigned int expert = expert_ids[route];
    if (expert >= (unsigned int)num_experts) {
        return;
    }
    const int input_row = input_rows == 1 ? route / top_k : route;
    const dinf_block_q8_1 *activation = (const dinf_block_q8_1 *)activation_ptr
        + (unsigned long long)input_row * activation_stride;
    const int blocks_per_row = KIND == 0 ? input_dim / 32 : input_dim / 256;
    const int block_bytes = KIND == 1 ? 84 : 144;
    const unsigned long long expert_bytes =
        (unsigned long long)output_dim * blocks_per_row * block_bytes;
    const void *gate_weights = (const void *)(gate_address + expert * expert_bytes);
    const void *up_weights = (const void *)(up_address + expert * expert_bytes);
    float gate = dinf_partial_dot<KIND>(gate_weights, activation, output_row, input_dim);
    float up = dinf_partial_dot<KIND>(up_weights, activation, output_row, input_dim);
    __shared__ float partial_gate[3][DINF_WARP_SIZE];
    __shared__ float partial_up[3][DINF_WARP_SIZE];
    if (threadIdx.y > 0) {
        partial_gate[threadIdx.y - 1][threadIdx.x] = gate;
        partial_up[threadIdx.y - 1][threadIdx.x] = up;
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
#pragma unroll
    for (int warp = 0; warp < 3; ++warp) {
        gate += partial_gate[warp][threadIdx.x];
        up += partial_up[warp][threadIdx.x];
    }
    gate = dinf_warp_sum(gate);
    up = dinf_warp_sum(up);
    if (threadIdx.x == 0 && output_row < output_dim) {
        output[(unsigned long long)route * output_dim + output_row] =
            (gate / (1.0f + expf(-gate))) * up;
    }
}

#define DINF_GROUP_ENTRY(name, kind)                                           \
extern "C" __global__ void name(                                             \
    unsigned long long w0, unsigned long long w1, unsigned long long w2,       \
    const void *activation, float *output, int input_dim,                      \
    int rows0, int rows1, int rows2) {                                         \
    dinf_group3_body<kind>(w0, w1, w2, activation, output, input_dim,          \
        rows0, rows1, rows2);                                                  \
}

#define DINF_GATED_ENTRY(name, kind)                                           \
extern "C" __global__ void name(                                             \
    unsigned long long gate, unsigned long long up, const void *activation,    \
    float *output, int input_dim, int output_dim, int activation_kind) {       \
    dinf_gated_body<kind>(gate, up, activation, output, input_dim,             \
        output_dim, activation_kind);                                          \
}

#define DINF_INDEXED_GATED_ENTRY(name, kind)                                   \
extern "C" __global__ void name(                                             \
    unsigned long long gate, unsigned long long up, const void *activation,    \
    const unsigned int *expert_ids, float *output, int input_dim,              \
    int output_dim, int num_experts, int top_k, int input_rows,                \
    int activation_stride) {                                                   \
    dinf_indexed_gated_body<kind>(gate, up, activation, expert_ids, output,    \
        input_dim, output_dim, num_experts, top_k, input_rows,                 \
        activation_stride);                                                    \
}

DINF_GROUP_ENTRY(dinf_q4_0_group3_mmvq_f32, 0)
DINF_GATED_ENTRY(dinf_q4_0_gated_mmvq_f32, 0)
DINF_GROUP_ENTRY(dinf_q2_k_group3_mmvq_f32, 1)
DINF_GATED_ENTRY(dinf_q2_k_gated_mmvq_f32, 1)
DINF_INDEXED_GATED_ENTRY(dinf_q2_k_indexed_gated_mmvq_f32, 1)
DINF_GROUP_ENTRY(dinf_q4_k_group3_mmvq_f32, 2)
DINF_GATED_ENTRY(dinf_q4_k_gated_mmvq_f32, 2)
DINF_INDEXED_GATED_ENTRY(dinf_q4_k_indexed_gated_mmvq_f32, 2)
"#;

type PlainLauncher = unsafe extern "C" fn(
    vx: *const c_void,
    vy: *const c_void,
    dst: *mut c_void,
    ncols_x: i32,
    nrows_x: i32,
    stride_col_y: i32,
    stride_col_dst: i32,
    batch_size: i32,
    stream: *mut c_void,
);

type MmqQuantizeLauncher = unsafe extern "C" fn(
    x: *const c_void,
    ids: *const i32,
    vy: *mut c_void,
    type_x: i32,
    ne00: i64,
    s01: i64,
    s02: i64,
    s03: i64,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    ne3: i64,
    stream: *mut c_void,
);

type MmqLauncher = unsafe extern "C" fn(
    tmp_fixup: *mut c_void,
    x: *const c_void,
    y: *const c_void,
    dst: *mut c_void,
    ncols_x: i64,
    nrows_x: i64,
    ncols_y: i64,
    stride_row_x: i64,
    stride_col_dst: i64,
    cc: i32,
    nsm: i32,
    smpbo: i64,
    warp_size: i32,
    stream: *mut c_void,
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MmqDsLayout {
    D4,
    Ds4,
    D2s6,
}

struct WorkspaceSlot {
    slice: CudaSlice<u8>,
    capacity: usize,
}

static WORKSPACES: OnceLock<Mutex<HashMap<DeviceId, WorkspaceSlot>>> = OnceLock::new();
static MMQ_WORKSPACES: OnceLock<Mutex<HashMap<DeviceId, WorkspaceSlot>>> = OnceLock::new();
static MMQ_FIXUP_WORKSPACES: OnceLock<Mutex<HashMap<DeviceId, WorkspaceSlot>>> = OnceLock::new();
static FAST_PATH_LOGGED: OnceLock<()> = OnceLock::new();
static SILU_DOWN_LOGGED: OnceLock<()> = OnceLock::new();
static RELU_SQUARED_DOWN_LOGGED: OnceLock<()> = OnceLock::new();
static Q8_FUSION_LOGGED: OnceLock<()> = OnceLock::new();
static MMQ_PREFILL_LOGGED: OnceLock<()> = OnceLock::new();
static KQUANT_FUSION_LOGGED: OnceLock<()> = OnceLock::new();
static INDEXED_KQUANT_FUSION_LOGGED: OnceLock<()> = OnceLock::new();
static SILU_QUANT_PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
static Q8_FUSION_PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
static KQUANT_FUSION_PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();

fn silu_quant_ptx() -> candle_core::Result<&'static str> {
    match SILU_QUANT_PTX.get_or_init(|| {
        compile_ptx_with_opts(
            SILU_QUANT_CUDA_SOURCE,
            CompileOptions {
                name: Some("dense_silu_quant_q8_1.cu".to_string()),
                use_fast_math: Some(false),
                ..Default::default()
            },
        )
        .map(|ptx| ptx.to_src())
        .map_err(|error| error.to_string())
    }) {
        Ok(ptx) => Ok(ptx.as_str()),
        Err(error) => candle_core::bail!("failed to compile fused CUDA SwiGLU quantizer: {error}"),
    }
}

fn q8_fusion_ptx() -> candle_core::Result<&'static str> {
    match Q8_FUSION_PTX.get_or_init(|| {
        compile_ptx_with_opts(
            Q8_FUSION_CUDA_SOURCE,
            CompileOptions {
                name: Some("dense_q8_0_fusion.cu".to_string()),
                use_fast_math: Some(false),
                ..Default::default()
            },
        )
        .map(|ptx| ptx.to_src())
        .map_err(|error| error.to_string())
    }) {
        Ok(ptx) => Ok(ptx.as_str()),
        Err(error) => candle_core::bail!("failed to compile fused Q8_0 CUDA kernels: {error}"),
    }
}

fn kquant_fusion_ptx() -> candle_core::Result<&'static str> {
    match KQUANT_FUSION_PTX.get_or_init(|| {
        compile_ptx_with_opts(
            KQUANT_FUSION_CUDA_SOURCE,
            CompileOptions {
                name: Some("dense_kquant_fusion.cu".to_string()),
                use_fast_math: Some(false),
                ..Default::default()
            },
        )
        .map(|ptx| ptx.to_src())
        .map_err(|error| error.to_string())
    }) {
        Ok(ptx) => Ok(ptx.as_str()),
        Err(error) => {
            candle_core::bail!("failed to compile fused Q4_0/Q2_K/Q4_K CUDA kernels: {error}")
        }
    }
}

pub(super) fn precompile(device: &Device) -> anyhow::Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    let device = device.as_cuda_device()?;
    device.get_or_load_custom_func(
        SILU_QUANT_KERNEL_NAME,
        SILU_QUANT_MODULE_NAME,
        silu_quant_ptx()?,
    )?;
    device.get_or_load_custom_func(
        RELU_SQUARED_QUANT_KERNEL_NAME,
        SILU_QUANT_MODULE_NAME,
        silu_quant_ptx()?,
    )?;
    device.get_or_load_custom_func(
        Q8_GROUP_KERNEL_NAME,
        Q8_FUSION_MODULE_NAME,
        q8_fusion_ptx()?,
    )?;
    device.get_or_load_custom_func(
        Q8_SWIGLU_KERNEL_NAME,
        Q8_FUSION_MODULE_NAME,
        q8_fusion_ptx()?,
    )?;
    for kernel in [
        Q4_0_GROUP_KERNEL_NAME,
        Q4_0_GATED_KERNEL_NAME,
        Q2_K_GROUP_KERNEL_NAME,
        Q2_K_GATED_KERNEL_NAME,
        Q2_K_INDEXED_GATED_KERNEL_NAME,
        Q4_K_GROUP_KERNEL_NAME,
        Q4_K_GATED_KERNEL_NAME,
        Q4_K_INDEXED_GATED_KERNEL_NAME,
    ] {
        device.get_or_load_custom_func(kernel, KQUANT_FUSION_MODULE_NAME, kquant_fusion_ptx()?)?;
    }
    Ok(())
}

fn pad(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

fn launcher(dtype: GgmlDType) -> Option<PlainLauncher> {
    let launcher: PlainLauncher = match dtype {
        GgmlDType::Q4_0 => kernels::ffi::launch_mmvq_gguf_q4_0_f32_plain,
        GgmlDType::Q4_1 => kernels::ffi::launch_mmvq_gguf_q4_1_f32_plain,
        GgmlDType::Q5_0 => kernels::ffi::launch_mmvq_gguf_q5_0_f32_plain,
        GgmlDType::Q5_1 => kernels::ffi::launch_mmvq_gguf_q5_1_f32_plain,
        GgmlDType::Q8_0 => kernels::ffi::launch_mmvq_gguf_q8_0_f32_plain,
        GgmlDType::Q2K => kernels::ffi::launch_mmvq_gguf_q2_k_f32_plain,
        GgmlDType::Q3K => kernels::ffi::launch_mmvq_gguf_q3_k_f32_plain,
        GgmlDType::Q4K => kernels::ffi::launch_mmvq_gguf_q4_k_f32_plain,
        GgmlDType::Q5K => kernels::ffi::launch_mmvq_gguf_q5_k_f32_plain,
        GgmlDType::Q6K => kernels::ffi::launch_mmvq_gguf_q6_k_f32_plain,
        _ => return None,
    };
    Some(launcher)
}

fn mmq_qk(dtype: GgmlDType) -> Option<usize> {
    match dtype {
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1 | GgmlDType::Q8_0 => {
            Some(32)
        }
        GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K => {
            Some(256)
        }
        _ => None,
    }
}

fn mmq_ds_layout(dtype: GgmlDType) -> Option<MmqDsLayout> {
    match dtype {
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_1 | GgmlDType::Q4K | GgmlDType::Q5K => {
            Some(MmqDsLayout::Ds4)
        }
        GgmlDType::Q5_0 | GgmlDType::Q8_0 | GgmlDType::Q3K | GgmlDType::Q6K => {
            Some(MmqDsLayout::D4)
        }
        GgmlDType::Q2K => Some(MmqDsLayout::D2s6),
        _ => None,
    }
}

fn mmq_quantize_launcher(layout: MmqDsLayout) -> MmqQuantizeLauncher {
    match layout {
        MmqDsLayout::D4 => kernels::ffi::launch_mmq_quantize_q8_1_D4,
        MmqDsLayout::Ds4 => kernels::ffi::launch_mmq_quantize_q8_1_DS4,
        MmqDsLayout::D2s6 => kernels::ffi::launch_mmq_quantize_q8_1_D2S6,
    }
}

fn mmq_launcher(dtype: GgmlDType) -> Option<MmqLauncher> {
    let launcher: MmqLauncher = match dtype {
        GgmlDType::Q4_0 => kernels::ffi::launch_mmq_gguf_q4_0,
        GgmlDType::Q4_1 => kernels::ffi::launch_mmq_gguf_q4_1,
        GgmlDType::Q5_0 => kernels::ffi::launch_mmq_gguf_q5_0,
        GgmlDType::Q5_1 => kernels::ffi::launch_mmq_gguf_q5_1,
        GgmlDType::Q8_0 => kernels::ffi::launch_mmq_gguf_q8_0,
        GgmlDType::Q2K => kernels::ffi::launch_mmq_gguf_q2_k,
        GgmlDType::Q3K => kernels::ffi::launch_mmq_gguf_q3_k,
        GgmlDType::Q4K => kernels::ffi::launch_mmq_gguf_q4_k,
        GgmlDType::Q5K => kernels::ffi::launch_mmq_gguf_q5_k,
        GgmlDType::Q6K => kernels::ffi::launch_mmq_gguf_q6_k,
        _ => return None,
    };
    Some(launcher)
}

fn fused_group_kernel(dtype: GgmlDType) -> Option<(&'static str, &'static str)> {
    match dtype {
        GgmlDType::Q8_0 => Some((Q8_FUSION_MODULE_NAME, Q8_GROUP_KERNEL_NAME)),
        GgmlDType::Q4_0 => Some((KQUANT_FUSION_MODULE_NAME, Q4_0_GROUP_KERNEL_NAME)),
        GgmlDType::Q2K => Some((KQUANT_FUSION_MODULE_NAME, Q2_K_GROUP_KERNEL_NAME)),
        GgmlDType::Q4K => Some((KQUANT_FUSION_MODULE_NAME, Q4_K_GROUP_KERNEL_NAME)),
        _ => None,
    }
}

fn fused_gated_kernel(dtype: GgmlDType) -> Option<(&'static str, &'static str)> {
    match dtype {
        GgmlDType::Q8_0 => Some((Q8_FUSION_MODULE_NAME, Q8_SWIGLU_KERNEL_NAME)),
        GgmlDType::Q4_0 => Some((KQUANT_FUSION_MODULE_NAME, Q4_0_GATED_KERNEL_NAME)),
        GgmlDType::Q2K => Some((KQUANT_FUSION_MODULE_NAME, Q2_K_GATED_KERNEL_NAME)),
        GgmlDType::Q4K => Some((KQUANT_FUSION_MODULE_NAME, Q4_K_GATED_KERNEL_NAME)),
        _ => None,
    }
}

fn fusion_ptx(dtype: GgmlDType) -> candle_core::Result<&'static str> {
    if dtype == GgmlDType::Q8_0 {
        q8_fusion_ptx()
    } else {
        kquant_fusion_ptx()
    }
}

fn ensure_workspace(
    device: &candle_core::CudaDevice,
    bytes: usize,
) -> candle_core::Result<std::sync::MutexGuard<'static, HashMap<DeviceId, WorkspaceSlot>>> {
    ensure_workspace_in(&WORKSPACES, device, bytes)
}

fn ensure_workspace_in(
    workspaces: &'static OnceLock<Mutex<HashMap<DeviceId, WorkspaceSlot>>>,
    device: &candle_core::CudaDevice,
    bytes: usize,
) -> candle_core::Result<std::sync::MutexGuard<'static, HashMap<DeviceId, WorkspaceSlot>>> {
    use std::collections::hash_map::Entry;

    let workspaces = workspaces.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = workspaces.lock().map_err(|_| {
        candle_core::Error::Msg("CUDA grouped MMVQ workspace lock was poisoned".into())
    })?;
    match guard.entry(device.id()) {
        Entry::Occupied(mut entry) => {
            if entry.get().capacity < bytes {
                entry.insert(WorkspaceSlot {
                    slice: unsafe { device.alloc::<u8>(bytes)? },
                    capacity: bytes,
                });
            }
        }
        Entry::Vacant(entry) => {
            entry.insert(WorkspaceSlot {
                slice: unsafe { device.alloc::<u8>(bytes)? },
                capacity: bytes,
            });
        }
    }
    Ok(guard)
}

/// The device address of the Q8_1 form a producer kernel already wrote for
/// the single F32 row at `input`, if there is one.
fn published_q8_1(input: cuda_sys::CUdeviceptr, input_dim: usize, rows: usize) -> Option<u64> {
    if rows != 1 {
        return None;
    }
    super::q8_1_memo_cuda::lookup(input, input_dim)
}

fn checked_f32_view<'a>(
    name: &str,
    storage: &'a CudaStorage,
    layout: &Layout,
    elements: usize,
) -> candle_core::Result<CudaView<'a, f32>> {
    let slice = storage.as_cuda_slice::<f32>()?;
    let start = layout.start_offset();
    let end = start.checked_add(elements).ok_or_else(|| {
        candle_core::Error::Msg(format!("fused CUDA SwiGLU {name} view overflow"))
    })?;
    if end > slice.len() {
        candle_core::bail!("fused CUDA SwiGLU {name} view exceeds its storage")
    }
    Ok(slice.slice(start..end))
}

struct GroupedQMatMul {
    weights: Vec<Arc<QTensor>>,
    rows: Vec<usize>,
    input_dim: usize,
    output_dim: usize,
    batch_size: usize,
    qkv_epilogue: Option<QkvDecodeEpilogue>,
}

impl candle_core::CustomOp1 for GroupedQMatMul {
    fn name(&self) -> &'static str {
        "cuda-grouped-quantized-matmul"
    }

    fn cpu_fwd(
        &self,
        _storage: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda-grouped-quantized-matmul is only dispatched on CUDA")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        if !matches!(storage.dtype(), DType::F16 | DType::F32)
            || !layout.is_contiguous()
            || layout.shape().elem_count() != self.input_dim * self.batch_size
        {
            candle_core::bail!(
                "CUDA grouped quantized matmul requires {} contiguous f16/f32 rows of {} elements",
                self.batch_size,
                self.input_dim,
            )
        }
        let input_start = layout.start_offset();
        let input_end = input_start
            .checked_add(self.input_dim * self.batch_size)
            .ok_or_else(|| candle_core::Error::Msg("grouped MMVQ input view overflow".into()))?;
        let device = storage.device();
        let stream = device.cuda_stream();

        let input_dim_padded = pad(self.input_dim, MATRIX_ROW_PADDING);
        let scratch_bytes = self.batch_size * input_dim_padded / Q8_1_BLOCK_SIZE * Q8_1_TYPE_SIZE;
        let workspace_guard = ensure_workspace(device, scratch_bytes)?;
        let workspace = &workspace_guard
            .get(&device.id())
            .expect("workspace was inserted above")
            .slice;

        let (workspace_ptr, _workspace_sync) = workspace.device_ptr(&stream);
        let mut quantized_address = workspace_ptr;
        let mut output = unsafe { device.alloc::<f32>(self.output_dim * self.batch_size)? };
        let stream_ptr = stream.cu_stream() as *mut c_void;

        match storage.dtype() {
            DType::F16 => {
                let input = storage.as_cuda_slice::<half::f16>()?;
                if input_end > input.len() {
                    candle_core::bail!("grouped MMVQ input view exceeds its CUDA storage")
                }
                let input = input.slice(input_start..input_end);
                let (input_ptr, _input_sync) = input.device_ptr(&stream);
                unsafe {
                    kernels::ffi::launch_mmvq_gguf_quantize_q8_1_f16(
                        input_ptr as *const c_void,
                        workspace_ptr as *mut c_void,
                        self.input_dim as i32,
                        input_dim_padded as i32,
                        self.batch_size as i32,
                        stream_ptr,
                    );
                }
            }
            DType::F32 => {
                let input = storage.as_cuda_slice::<f32>()?;
                if input_end > input.len() {
                    candle_core::bail!("grouped MMVQ input view exceeds its CUDA storage")
                }
                let input = input.slice(input_start..input_end);
                let (input_ptr, _input_sync) = input.device_ptr(&stream);
                // An input whose producer already wrote its Q8_1 form skips
                // the quantization launch.
                match published_q8_1(input_ptr, self.input_dim, self.batch_size) {
                    Some(address) => quantized_address = address,
                    None => unsafe {
                        kernels::ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                            input_ptr as *const c_void,
                            workspace_ptr as *mut c_void,
                            self.input_dim as i32,
                            input_dim_padded as i32,
                            self.batch_size as i32,
                            stream_ptr,
                        );
                    },
                }
            }
            _ => unreachable!("dtype eligibility checked above"),
        }

        // Consecutive projections with one fusable quantization share a
        // launch, up to three at a time; the rest launch one by one. Gemma 4
        // 26B-A4B's K and V are one such run beside a differently quantized Q.
        let stride_col_y = (input_dim_padded / Q8_1_BLOCK_SIZE) as i32;
        let mut output_offset = 0usize;
        let mut first = 0usize;
        while first < self.weights.len() {
            let dtype = self.weights[first].dtype();
            let mut end = first + 1;
            if self.batch_size == 1 && fused_group_kernel(dtype).is_some() {
                while end < self.weights.len()
                    && end - first < 3
                    && self.weights[end].dtype() == dtype
                {
                    end += 1;
                }
            }
            let run_rows = self.rows[first..end].iter().sum::<usize>();
            if end - first >= 2 {
                let mut addresses = Vec::with_capacity(3);
                let mut weight_syncs = Vec::with_capacity(3);
                for weight in &self.weights[first..end] {
                    let (weight_ptr, weight_sync) = weight.device_ptr_with_guard(&stream)?;
                    addresses.push(weight_ptr as u64);
                    weight_syncs.push(weight_sync);
                }
                while addresses.len() < 3 {
                    addresses.push(addresses[0]);
                }
                let (module_name, kernel_name) =
                    fused_group_kernel(dtype).expect("fused dtype was checked above");
                let function =
                    device.get_or_load_custom_func(kernel_name, module_name, fusion_ptx(dtype)?)?;
                let config = LaunchConfig {
                    grid_dim: (run_rows as u32, 1, 1),
                    block_dim: (32, 4, 1),
                    shared_mem_bytes: 0,
                };
                let input_dim = self.input_dim as i32;
                let run = &self.rows[first..end];
                let rows0 = run[0] as i32;
                let rows1 = run[1] as i32;
                let rows2 = run.get(2).copied().unwrap_or(0) as i32;
                let mut run_output = output.slice_mut(output_offset..output_offset + run_rows);
                let mut builder = function.builder();
                builder
                    .arg(&addresses[0])
                    .arg(&addresses[1])
                    .arg(&addresses[2])
                    .arg(&quantized_address)
                    .arg(&mut run_output)
                    .arg(&input_dim)
                    .arg(&rows0)
                    .arg(&rows1)
                    .arg(&rows2);
                // SAFETY: The run's matrices share the validated quantization
                // and geometry, the kernel writes `run_rows` outputs into a
                // view of that length, and the Q8_1 activation stays live.
                unsafe { builder.launch(config) }.w()?;
                drop(weight_syncs);
                if dtype == GgmlDType::Q8_0 {
                    Q8_FUSION_LOGGED.get_or_init(|| {
                        tracing::info!("CUDA fused Q8_0 projection grids enabled");
                    });
                } else {
                    KQUANT_FUSION_LOGGED.get_or_init(|| {
                        tracing::info!("CUDA fused Q4_0/Q2_K/Q4_K projection grids enabled");
                    });
                }
            } else {
                let weight = &self.weights[first];
                let Some(launch) = launcher(dtype) else {
                    candle_core::bail!("grouped MMVQ weight {first} uses unsupported {dtype:?}")
                };
                let (weight_ptr, _weight_sync) = weight.device_ptr_with_guard(&stream)?;
                let (output_ptr, _output_sync) = output.device_ptr(&stream);
                let destination = (output_ptr as usize)
                    .checked_add(output_offset * std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        candle_core::Error::Msg("grouped MMVQ output pointer overflow".into())
                    })? as *mut c_void;
                unsafe {
                    launch(
                        weight_ptr as *const c_void,
                        quantized_address as *const c_void,
                        destination,
                        self.input_dim as i32,
                        run_rows as i32,
                        stride_col_y,
                        self.output_dim as i32,
                        self.batch_size as i32,
                        stream_ptr,
                    );
                }
            }
            output_offset += run_rows;
            first = end;
        }
        debug_assert_eq!(output_offset, self.output_dim);

        let output = match self.qkv_epilogue.as_ref() {
            Some(epilogue) => epilogue.launch(output, device)?,
            None => QkvDecodeOutput::F32(output),
        };

        let mut output_shape = layout.shape().dims().to_vec();
        output_shape.pop();
        output_shape.push(self.output_dim);
        let storage = match output {
            QkvDecodeOutput::F32(output) => CudaStorage::wrap_cuda_slice(output, device.clone()),
            QkvDecodeOutput::F16(output) => CudaStorage::wrap_cuda_slice(output, device.clone()),
        };
        Ok((storage, Shape::from_dims(&output_shape)))
    }
}

struct FusedQ8SwiGlu {
    gate: Arc<QTensor>,
    up: Arc<QTensor>,
    input_dim: usize,
    output_dim: usize,
}

#[derive(Clone, Copy)]
enum GatedActivation {
    Silu = 0,
    Gelu = 1,
}

struct FusedKQuantGated {
    gate: Arc<QTensor>,
    up: Arc<QTensor>,
    input_dim: usize,
    output_dim: usize,
    dtype: GgmlDType,
    activation: GatedActivation,
}

impl candle_core::CustomOp1 for FusedKQuantGated {
    fn name(&self) -> &'static str {
        "cuda-fused-kquant-gated-projection"
    }

    fn cpu_fwd(
        &self,
        _storage: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda-fused-kquant-gated-projection is only dispatched on CUDA")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        if storage.dtype() != DType::F32
            || !layout.is_contiguous()
            || layout.shape().elem_count() != self.input_dim
        {
            candle_core::bail!(
                "fused CUDA K-quant gated projection requires one contiguous f32 row of {} elements",
                self.input_dim
            )
        }
        let device = storage.device();
        let stream = device.cuda_stream();
        let input_dim_padded = pad(self.input_dim, MATRIX_ROW_PADDING);
        let scratch_bytes = input_dim_padded / Q8_1_BLOCK_SIZE * Q8_1_TYPE_SIZE;
        let workspace_guard = ensure_workspace(device, scratch_bytes)?;
        let workspace = &workspace_guard
            .get(&device.id())
            .expect("workspace was inserted above")
            .slice;
        let input = storage.as_cuda_slice::<f32>()?;
        let input_start = layout.start_offset();
        let input_end = input_start + self.input_dim;
        if input_end > input.len() {
            candle_core::bail!("fused CUDA K-quant gated input exceeds its storage")
        }
        let input = input.slice(input_start..input_end);
        let (input_ptr, _input_sync) = input.device_ptr(&stream);
        let (workspace_ptr, _workspace_sync) = workspace.device_ptr(&stream);
        // An input whose producer already wrote its Q8_1 form skips the
        // quantization launch.
        let quantized_address = match published_q8_1(input_ptr, self.input_dim, 1) {
            Some(address) => address,
            None => {
                unsafe {
                    kernels::ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                        input_ptr as *const c_void,
                        workspace_ptr as *mut c_void,
                        self.input_dim as i32,
                        input_dim_padded as i32,
                        1,
                        stream.cu_stream() as *mut c_void,
                    );
                }
                workspace_ptr
            }
        };

        let (gate_ptr, _gate_sync) = self.gate.device_ptr_with_guard(&stream)?;
        let (up_ptr, _up_sync) = self.up.device_ptr_with_guard(&stream)?;
        let gate_address = gate_ptr as u64;
        let up_address = up_ptr as u64;
        let mut output = unsafe { device.alloc::<f32>(self.output_dim)? };
        let (module_name, kernel_name) = fused_gated_kernel(self.dtype)
            .ok_or_else(|| candle_core::Error::Msg("unsupported fused K-quant dtype".into()))?;
        let function =
            device.get_or_load_custom_func(kernel_name, module_name, kquant_fusion_ptx()?)?;
        let config = LaunchConfig {
            grid_dim: (self.output_dim as u32, 1, 1),
            block_dim: (32, 4, 1),
            shared_mem_bytes: 0,
        };
        let input_dim = self.input_dim as i32;
        let output_dim = self.output_dim as i32;
        let activation = self.activation as i32;
        let mut builder = function.builder();
        builder
            .arg(&gate_address)
            .arg(&up_address)
            .arg(&quantized_address)
            .arg(&mut output)
            .arg(&input_dim)
            .arg(&output_dim)
            .arg(&activation);
        // SAFETY: Weight quantization, matrix geometry, activation workspace,
        // output bounds, and device identity were validated by the caller.
        unsafe { builder.launch(config) }.w()?;

        let mut output_shape = layout.shape().dims().to_vec();
        output_shape.pop();
        output_shape.push(self.output_dim);
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from_dims(&output_shape),
        ))
    }
}

struct FusedIndexedKQuantSwiGlu {
    gate: Arc<QTensor>,
    up: Arc<QTensor>,
    dtype: GgmlDType,
    tokens: usize,
    input_rows: usize,
    top_k: usize,
    num_experts: usize,
    input_dim: usize,
    output_dim: usize,
}

impl candle_core::CustomOp2 for FusedIndexedKQuantSwiGlu {
    fn name(&self) -> &'static str {
        "cuda-fused-indexed-kquant-swiglu"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _ids: &CpuStorage,
        _ids_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda-fused-indexed-kquant-swiglu is only dispatched on CUDA")
    }

    fn cuda_fwd(
        &self,
        input_storage: &CudaStorage,
        input_layout: &Layout,
        ids_storage: &CudaStorage,
        ids_layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let input_elements = self.tokens * self.input_rows * self.input_dim;
        let routes = self.tokens * self.top_k;
        if input_storage.dtype() != DType::F32
            || ids_storage.dtype() != DType::U32
            || !input_layout.is_contiguous()
            || !ids_layout.is_contiguous()
            || input_layout.shape().elem_count() != input_elements
            || ids_layout.shape().elem_count() != routes
            || input_storage.device().id() != ids_storage.device().id()
        {
            candle_core::bail!("fused indexed K-quant SwiGLU received invalid CUDA inputs")
        }
        let input_slice = input_storage.as_cuda_slice::<f32>()?;
        let input_start = input_layout.start_offset();
        let input_end = input_start + input_elements;
        if input_end > input_slice.len() {
            candle_core::bail!("fused indexed K-quant input exceeds its storage")
        }
        let ids_slice = ids_storage.as_cuda_slice::<u32>()?;
        let ids_start = ids_layout.start_offset();
        let ids_end = ids_start + routes;
        if ids_end > ids_slice.len() {
            candle_core::bail!("fused indexed K-quant IDs exceed their storage")
        }
        let input = input_slice.slice(input_start..input_end);
        let ids = ids_slice.slice(ids_start..ids_end);
        let device = input_storage.device();
        let stream = device.cuda_stream();
        let input_dim_padded = pad(self.input_dim, MATRIX_ROW_PADDING);
        let activation_rows = self.tokens * self.input_rows;
        let scratch_bytes = activation_rows * input_dim_padded / Q8_1_BLOCK_SIZE * Q8_1_TYPE_SIZE;
        let workspace_guard = ensure_workspace(device, scratch_bytes)?;
        let workspace = &workspace_guard
            .get(&device.id())
            .expect("workspace was inserted above")
            .slice;
        let (input_ptr, _input_sync) = input.device_ptr(&stream);
        let (workspace_ptr, _workspace_sync) = workspace.device_ptr(&stream);
        unsafe {
            kernels::ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                input_ptr as *const c_void,
                workspace_ptr as *mut c_void,
                self.input_dim as i32,
                input_dim_padded as i32,
                activation_rows as i32,
                stream.cu_stream() as *mut c_void,
            );
        }

        let (gate_ptr, _gate_sync) = self.gate.device_ptr_with_guard(&stream)?;
        let (up_ptr, _up_sync) = self.up.device_ptr_with_guard(&stream)?;
        let gate_address = gate_ptr as u64;
        let up_address = up_ptr as u64;
        let mut output = unsafe { device.alloc::<f32>(routes * self.output_dim)? };
        let kernel_name = match self.dtype {
            GgmlDType::Q2K => Q2_K_INDEXED_GATED_KERNEL_NAME,
            GgmlDType::Q4K => Q4_K_INDEXED_GATED_KERNEL_NAME,
            _ => candle_core::bail!("unsupported indexed K-quant fusion dtype"),
        };
        let function = device.get_or_load_custom_func(
            kernel_name,
            KQUANT_FUSION_MODULE_NAME,
            kquant_fusion_ptx()?,
        )?;
        let config = LaunchConfig {
            grid_dim: (self.output_dim as u32, routes as u32, 1),
            block_dim: (32, 4, 1),
            shared_mem_bytes: 0,
        };
        let input_dim = self.input_dim as i32;
        let output_dim = self.output_dim as i32;
        let num_experts = self.num_experts as i32;
        let top_k = self.top_k as i32;
        let input_rows = self.input_rows as i32;
        let activation_stride = (input_dim_padded / Q8_1_BLOCK_SIZE) as i32;
        let mut builder = function.builder();
        builder
            .arg(&gate_address)
            .arg(&up_address)
            .arg(workspace)
            .arg(&ids)
            .arg(&mut output)
            .arg(&input_dim)
            .arg(&output_dim)
            .arg(&num_experts)
            .arg(&top_k)
            .arg(&input_rows)
            .arg(&activation_stride);
        // SAFETY: Route, expert, storage, quantization, and launch geometry
        // bounds were validated before the custom operation was constructed.
        unsafe { builder.launch(config) }.w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from((self.tokens, self.top_k, self.output_dim)),
        ))
    }
}

impl candle_core::CustomOp1 for FusedQ8SwiGlu {
    fn name(&self) -> &'static str {
        "cuda-fused-q8-0-swiglu"
    }

    fn cpu_fwd(
        &self,
        _storage: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda-fused-q8-0-swiglu is only dispatched on CUDA")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        if storage.dtype() != DType::F32
            || !layout.is_contiguous()
            || layout.shape().elem_count() != self.input_dim
        {
            candle_core::bail!(
                "fused CUDA Q8_0 SwiGLU requires one contiguous f32 row of {} elements",
                self.input_dim
            )
        }
        let device = storage.device();
        let stream = device.cuda_stream();
        let input_dim_padded = pad(self.input_dim, MATRIX_ROW_PADDING);
        let scratch_bytes = input_dim_padded / Q8_1_BLOCK_SIZE * Q8_1_TYPE_SIZE;
        let workspace_guard = ensure_workspace(device, scratch_bytes)?;
        let workspace = &workspace_guard
            .get(&device.id())
            .expect("workspace was inserted above")
            .slice;
        let input = storage.as_cuda_slice::<f32>()?;
        let input_start = layout.start_offset();
        let input_end = input_start + self.input_dim;
        if input_end > input.len() {
            candle_core::bail!("fused CUDA Q8_0 SwiGLU input exceeds its storage")
        }
        let input = input.slice(input_start..input_end);
        let (input_ptr, _input_sync) = input.device_ptr(&stream);
        let (workspace_ptr, _workspace_sync) = workspace.device_ptr(&stream);
        // An input whose producer already wrote its Q8_1 form skips the
        // quantization launch.
        let quantized_address = match published_q8_1(input_ptr, self.input_dim, 1) {
            Some(address) => address,
            None => {
                unsafe {
                    kernels::ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                        input_ptr as *const c_void,
                        workspace_ptr as *mut c_void,
                        self.input_dim as i32,
                        input_dim_padded as i32,
                        1,
                        stream.cu_stream() as *mut c_void,
                    );
                }
                workspace_ptr
            }
        };

        let (gate_ptr, _gate_sync) = self.gate.device_ptr_with_guard(&stream)?;
        let (up_ptr, _up_sync) = self.up.device_ptr_with_guard(&stream)?;
        let gate_address = gate_ptr as u64;
        let up_address = up_ptr as u64;
        let mut output = unsafe { device.alloc::<f32>(self.output_dim)? };
        let function = device.get_or_load_custom_func(
            Q8_SWIGLU_KERNEL_NAME,
            Q8_FUSION_MODULE_NAME,
            q8_fusion_ptx()?,
        )?;
        let config = LaunchConfig {
            grid_dim: (self.output_dim as u32, 1, 1),
            block_dim: (32, 4, 1),
            shared_mem_bytes: 0,
        };
        let input_dim = self.input_dim as i32;
        let output_dim = self.output_dim as i32;
        let mut builder = function.builder();
        builder
            .arg(&gate_address)
            .arg(&up_address)
            .arg(&quantized_address)
            .arg(&mut output)
            .arg(&input_dim)
            .arg(&output_dim);
        // SAFETY: Both Q8_0 matrices share the validated geometry, and the
        // complete quantized input and f32 output allocations remain live.
        unsafe { builder.launch(config) }.w()?;

        let mut output_shape = layout.shape().dims().to_vec();
        output_shape.pop();
        output_shape.push(self.output_dim);
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from_dims(&output_shape),
        ))
    }
}

struct FusedSiluDown {
    weight: Arc<QTensor>,
    input_dim: usize,
    output_dim: usize,
}

impl candle_core::CustomOp2 for FusedSiluDown {
    fn name(&self) -> &'static str {
        "cuda-fused-silu-quantized-down-projection"
    }

    fn cpu_fwd(
        &self,
        _gate: &CpuStorage,
        _gate_layout: &Layout,
        _up: &CpuStorage,
        _up_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda-fused-silu-quantized-down-projection is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        gate_storage: &CudaStorage,
        gate_layout: &Layout,
        up_storage: &CudaStorage,
        up_layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let validate = |name: &str, storage: &CudaStorage, layout: &Layout| {
            if storage.dtype() != DType::F32
                || !layout.is_contiguous()
                || layout.shape().elem_count() != self.input_dim
            {
                candle_core::bail!(
                    "fused CUDA SwiGLU {name} must be one contiguous f32 row of {} elements",
                    self.input_dim
                )
            }
            Ok(())
        };
        validate("gate", gate_storage, gate_layout)?;
        validate("up", up_storage, up_layout)?;
        if gate_storage.device().id() != up_storage.device().id() {
            candle_core::bail!("fused CUDA SwiGLU inputs use different devices")
        }

        let gate = checked_f32_view("gate", gate_storage, gate_layout, self.input_dim)?;
        let up = checked_f32_view("up", up_storage, up_layout, self.input_dim)?;
        let device = gate_storage.device();
        let stream = device.cuda_stream();
        let input_dim_padded = pad(self.input_dim, MATRIX_ROW_PADDING);
        let scratch_bytes = input_dim_padded / Q8_1_BLOCK_SIZE * Q8_1_TYPE_SIZE;
        let mut workspace_guard = ensure_workspace(device, scratch_bytes)?;
        let workspace = &mut workspace_guard
            .get_mut(&device.id())
            .expect("workspace was inserted above")
            .slice;

        let function = device.get_or_load_custom_func(
            SILU_QUANT_KERNEL_NAME,
            SILU_QUANT_MODULE_NAME,
            silu_quant_ptx()?,
        )?;
        let config = LaunchConfig {
            grid_dim: (
                input_dim_padded.div_ceil(CUDA_QUANTIZE_BLOCK_SIZE) as u32,
                1,
                1,
            ),
            block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let input_dim = self.input_dim as i32;
        let input_dim_padded_i32 = input_dim_padded as i32;
        let mut builder = function.builder();
        builder
            .arg(&gate)
            .arg(&up)
            .arg(&mut *workspace)
            .arg(&input_dim)
            .arg(&input_dim_padded_i32);
        // SAFETY: Both views, the complete padded destination, and launch
        // bounds were validated above. The kernel initializes every Q8_1 block.
        unsafe { builder.launch(config) }.w()?;

        let Some(launch) = launcher(self.weight.dtype()) else {
            candle_core::bail!(
                "fused CUDA SwiGLU down projection uses unsupported {:?}",
                self.weight.dtype()
            )
        };
        let output = unsafe { device.alloc::<f32>(self.output_dim)? };
        let (workspace_ptr, _workspace_sync) = workspace.device_ptr(&stream);
        let (weight_ptr, _weight_sync) = self.weight.device_ptr_with_guard(&stream)?;
        let (output_ptr, _output_sync) = output.device_ptr(&stream);
        unsafe {
            launch(
                weight_ptr as *const c_void,
                workspace_ptr as *const c_void,
                output_ptr as *mut c_void,
                self.input_dim as i32,
                self.output_dim as i32,
                (input_dim_padded / Q8_1_BLOCK_SIZE) as i32,
                self.output_dim as i32,
                1,
                stream.cu_stream() as *mut c_void,
            );
        }
        drop(_output_sync);

        let mut output_shape = gate_layout.shape().dims().to_vec();
        output_shape.pop();
        output_shape.push(self.output_dim);
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from_dims(&output_shape),
        ))
    }
}

struct FusedReluSquaredDown {
    weight: Arc<QTensor>,
    input_dim: usize,
    output_dim: usize,
}

impl candle_core::CustomOp1 for FusedReluSquaredDown {
    fn name(&self) -> &'static str {
        "cuda-fused-relu-squared-quantized-down-projection"
    }

    fn cpu_fwd(
        &self,
        _storage: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda-fused-relu-squared-down is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        if storage.dtype() != DType::F32
            || !layout.is_contiguous()
            || layout.shape().elem_count() != self.input_dim
        {
            candle_core::bail!(
                "fused CUDA ReLU-squared input must be one contiguous f32 row of {} elements",
                self.input_dim
            )
        }
        let input = checked_f32_view("ReLU-squared input", storage, layout, self.input_dim)?;
        let device = storage.device();
        let stream = device.cuda_stream();
        let input_dim_padded = pad(self.input_dim, MATRIX_ROW_PADDING);
        let scratch_bytes = input_dim_padded / Q8_1_BLOCK_SIZE * Q8_1_TYPE_SIZE;
        let mut workspace_guard = ensure_workspace(device, scratch_bytes)?;
        let workspace = &mut workspace_guard
            .get_mut(&device.id())
            .expect("workspace was inserted above")
            .slice;
        let function = device.get_or_load_custom_func(
            RELU_SQUARED_QUANT_KERNEL_NAME,
            SILU_QUANT_MODULE_NAME,
            silu_quant_ptx()?,
        )?;
        let config = LaunchConfig {
            grid_dim: (
                input_dim_padded.div_ceil(CUDA_QUANTIZE_BLOCK_SIZE) as u32,
                1,
                1,
            ),
            block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let input_dim = self.input_dim as i32;
        let input_dim_padded_i32 = input_dim_padded as i32;
        let mut builder = function.builder();
        builder
            .arg(&input)
            .arg(&mut *workspace)
            .arg(&input_dim)
            .arg(&input_dim_padded_i32);
        // SAFETY: Input and padded Q8_1 output bounds were checked above.
        unsafe { builder.launch(config) }.w()?;

        let launch = launcher(self.weight.dtype()).ok_or_else(|| {
            candle_core::Error::Msg("unsupported ReLU-squared down quantization".into())
        })?;
        let output = unsafe { device.alloc::<f32>(self.output_dim)? };
        let (workspace_ptr, _workspace_sync) = workspace.device_ptr(&stream);
        let (weight_ptr, _weight_sync) = self.weight.device_ptr_with_guard(&stream)?;
        let (output_ptr, output_sync) = output.device_ptr(&stream);
        unsafe {
            launch(
                weight_ptr as *const c_void,
                workspace_ptr as *const c_void,
                output_ptr as *mut c_void,
                self.input_dim as i32,
                self.output_dim as i32,
                (input_dim_padded / Q8_1_BLOCK_SIZE) as i32,
                self.output_dim as i32,
                1,
                stream.cu_stream() as *mut c_void,
            );
        }
        drop(output_sync);
        let mut output_shape = layout.shape().dims().to_vec();
        output_shape.pop();
        output_shape.push(self.output_dim);
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from_dims(&output_shape),
        ))
    }
}

#[derive(Clone, Copy)]
struct MmqDeviceInfo {
    cc: i32,
    nsm: i32,
    smpbo: i64,
    warp_size: i32,
}

fn mmq_device_info(device: &candle_core::CudaDevice) -> MmqDeviceInfo {
    let cuda_device = device.cuda_stream().context().cu_device();
    let attribute =
        |name| unsafe { cuda_result::device::get_attribute(cuda_device, name).unwrap_or_default() };
    let major =
        attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    let minor =
        attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    MmqDeviceInfo {
        cc: major * 100 + minor * 10,
        nsm: attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .max(1),
        smpbo: i64::from(
            attribute(
                cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )
            .max(49152),
        ),
        warp_size: attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE).max(32),
    }
}

/// Run prompt projections after quantizing each required Q8_1 scale layout once.
///
/// Candle's individual MMQ entry point owns its activation workspace, so Q/K/V
/// quantize the same prompt three times and gate/up do it twice. K-quants use
/// three different scale layouts; mixed projection groups are partitioned by
/// layout while retaining their original output order.
fn forward_mmq_grouped(
    input: &Tensor,
    weights: &[Arc<QTensor>],
    rows: &[usize],
    input_dim: usize,
    batch_size: usize,
) -> anyhow::Result<Option<Vec<Tensor>>> {
    if batch_size <= 8
        || input.dtype() != DType::F32
        || !input.is_contiguous()
        || weights.is_empty()
        || weights.len() != rows.len()
        || weights.iter().any(|weight| {
            mmq_launcher(weight.dtype()).is_none()
                || mmq_ds_layout(weight.dtype()).is_none()
                || mmq_qk(weight.dtype()).is_none_or(|qk| !input_dim.is_multiple_of(qk))
        })
    {
        return Ok(None);
    }

    let (storage_guard, layout) = input.storage_and_layout();
    let Storage::Cuda(storage) = &*storage_guard else {
        return Ok(None);
    };
    let Some((input_start, input_end)) = layout.contiguous_offsets() else {
        return Ok(None);
    };
    if input_end - input_start != batch_size * input_dim {
        return Ok(None);
    }
    let input_slice = storage.as_cuda_slice::<f32>()?;
    if input_end > input_slice.len() {
        anyhow::bail!("grouped quantized MMQ input view exceeds its CUDA storage")
    }
    let input_view = input_slice.slice(input_start..input_end);
    let device = storage.device();
    let stream = device.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut c_void;
    let (input_ptr, _input_sync) = input_view.device_ptr(&stream);

    let input_dim_padded = pad(pad(input_dim, MATRIX_ROW_PADDING), MMQ_Q8_1_BLOCK_SIZE);
    let blocks_per_row = input_dim_padded / MMQ_Q8_1_BLOCK_SIZE;
    let workspace_bytes =
        batch_size * blocks_per_row * MMQ_Q8_1_TYPE_SIZE + MMQ_FIXUP_X_MAX * MMQ_Q8_1_TYPE_SIZE;
    let workspace_guard = ensure_workspace_in(&MMQ_WORKSPACES, device, workspace_bytes)?;
    let workspace = &workspace_guard
        .get(&device.id())
        .expect("MMQ workspace was inserted above")
        .slice;
    let (workspace_ptr, _workspace_sync) = workspace.device_ptr(&stream);

    let fixup_bytes =
        MMQ_FIXUP_MAX_SMS * MMQ_FIXUP_X_MAX * MMQ_FIXUP_Y_MAX * std::mem::size_of::<f32>();
    let fixup_guard = ensure_workspace_in(&MMQ_FIXUP_WORKSPACES, device, fixup_bytes)?;
    let fixup = &fixup_guard
        .get(&device.id())
        .expect("MMQ fixup workspace was inserted above")
        .slice;
    let (fixup_ptr, _fixup_sync) = fixup.device_ptr(&stream);

    let device_info = mmq_device_info(device);
    let mut outputs = std::iter::repeat_with(|| None)
        .take(weights.len())
        .collect::<Vec<Option<Tensor>>>();
    for ds_layout in [MmqDsLayout::D4, MmqDsLayout::Ds4, MmqDsLayout::D2s6] {
        if !weights
            .iter()
            .any(|weight| mmq_ds_layout(weight.dtype()) == Some(ds_layout))
        {
            continue;
        }
        unsafe {
            mmq_quantize_launcher(ds_layout)(
                input_ptr as *const c_void,
                std::ptr::null(),
                workspace_ptr as *mut c_void,
                0,
                input_dim as i64,
                input_dim as i64,
                0,
                0,
                input_dim_padded as i64,
                batch_size as i64,
                1,
                1,
                stream_ptr,
            );
        }
        for (index, (weight, output_dim)) in weights.iter().zip(rows.iter().copied()).enumerate() {
            if mmq_ds_layout(weight.dtype()) != Some(ds_layout) {
                continue;
            }
            let qk = mmq_qk(weight.dtype()).expect("MMQ dtype was checked above");
            let launch = mmq_launcher(weight.dtype()).expect("MMQ dtype was checked above");
            let (weight_ptr, weight_sync) = weight.device_ptr_with_guard(&stream)?;
            let output = unsafe { device.alloc::<f32>(batch_size * output_dim)? };
            let (output_ptr, output_sync) = output.device_ptr(&stream);
            unsafe {
                launch(
                    fixup_ptr as *mut c_void,
                    weight_ptr as *const c_void,
                    workspace_ptr as *const c_void,
                    output_ptr as *mut c_void,
                    input_dim as i64,
                    output_dim as i64,
                    batch_size as i64,
                    (input_dim / qk) as i64,
                    output_dim as i64,
                    device_info.cc,
                    device_info.nsm,
                    device_info.smpbo,
                    device_info.warp_size,
                    stream_ptr,
                );
            }
            drop(output_sync);
            drop(weight_sync);

            let mut output_shape = layout.shape().dims().to_vec();
            output_shape.pop();
            output_shape.push(output_dim);
            let output_storage = CudaStorage::wrap_cuda_slice(output, device.clone());
            outputs[index] = Some(Tensor::from_storage(
                Storage::Cuda(output_storage),
                Shape::from_dims(&output_shape),
                BackpropOp::none(),
                false,
            ));
        }
    }
    let outputs = outputs
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| anyhow::anyhow!("grouped MMQ did not produce every projection"))?;
    MMQ_PREFILL_LOGGED.get_or_init(|| {
        tracing::info!("CUDA shared-activation mixed-quant MMQ prefill path enabled");
    });
    Ok(Some(outputs))
}

/// Run quantized projections that share one contiguous CUDA activation.
///
/// Returns `Ok(None)` when the inputs are not eligible, preserving Candle's
/// normal `QMatMul` path for non-CUDA devices, dense weights, and unsupported
/// quantization types. MMVQ batches reuse one Q8_1 vector set; prompt batches
/// reuse one tiled MMQ activation per required quantization scale layout.
pub(super) fn forward(input: &Tensor, weights: &[&QMatMul]) -> anyhow::Result<Option<Vec<Tensor>>> {
    if input.dtype() != DType::F32 || weights.is_empty() {
        return Ok(None);
    }
    forward_impl(input, weights, None)
}

/// Project one f16/f32 decode row with a quantized weight and return f32.
///
/// Candle normally returns f16 from an f16 MMVQ. Attention consumes native f16,
/// but the residual stream is f32, so selecting the f32 MMVQ epilogue avoids a
/// standalone f16-to-f32 conversion without changing the Q8_1 activation or
/// accumulation arithmetic.
pub(super) fn forward_one_to_f32(
    input: &Tensor,
    weight: &QMatMul,
) -> anyhow::Result<Option<Tensor>> {
    let Some(mut outputs) = forward_impl(input, &[weight], None)? else {
        return Ok(None);
    };
    debug_assert_eq!(outputs.len(), 1);
    Ok(outputs.pop())
}

/// Evaluate gate/up MMVQ and SwiGLU in one projection grid.
pub(super) fn silu_gate_up(
    input: &Tensor,
    gate: &QMatMul,
    up: &QMatMul,
) -> anyhow::Result<Option<Tensor>> {
    gated_projection(input, gate, up, GatedActivation::Silu)
}

/// Evaluate K-quant gate/up MMVQ and GeGLU in one projection grid.
pub(super) fn gelu_gate_up(
    input: &Tensor,
    gate: &QMatMul,
    up: &QMatMul,
) -> anyhow::Result<Option<Tensor>> {
    gated_projection(input, gate, up, GatedActivation::Gelu)
}

fn gated_projection(
    input: &Tensor,
    gate: &QMatMul,
    up: &QMatMul,
    activation: GatedActivation,
) -> anyhow::Result<Option<Tensor>> {
    if !input.device().is_cuda() || input.dtype() != DType::F32 || !input.is_contiguous() {
        return Ok(None);
    }
    let input_dim = input.dim(candle_core::D::Minus1)?;
    if input_dim == 0
        || input.elem_count() != input_dim
        || input_dim > (i32::MAX as usize).saturating_sub(MATRIX_ROW_PADDING - 1)
    {
        return Ok(None);
    }
    let (QMatMul::QTensor(gate), QMatMul::QTensor(up)) = (gate, up) else {
        return Ok(None);
    };
    let (gate_rows, gate_cols) = gate.shape().dims2()?;
    let (up_rows, up_cols) = up.shape().dims2()?;
    let dtype = gate.dtype();
    if up.dtype() != dtype
        || fused_gated_kernel(dtype).is_none()
        || matches!(activation, GatedActivation::Gelu) && dtype == GgmlDType::Q8_0
        || gate_rows == 0
        || gate_rows > i32::MAX as usize
        || gate_rows != up_rows
        || gate_cols != input_dim
        || up_cols != input_dim
        || !input.device().same_device(&gate.device())
        || !input.device().same_device(&up.device())
    {
        return Ok(None);
    }

    let output = if dtype == GgmlDType::Q8_0 {
        input.apply_op1_no_bwd(&FusedQ8SwiGlu {
            gate: Arc::clone(gate),
            up: Arc::clone(up),
            input_dim,
            output_dim: gate_rows,
        })?
    } else {
        input.apply_op1_no_bwd(&FusedKQuantGated {
            gate: Arc::clone(gate),
            up: Arc::clone(up),
            input_dim,
            output_dim: gate_rows,
            dtype,
            activation,
        })?
    };
    if dtype == GgmlDType::Q8_0 {
        Q8_FUSION_LOGGED.get_or_init(|| {
            tracing::info!("CUDA fused Q8_0 projection grids enabled");
        });
    } else {
        KQUANT_FUSION_LOGGED.get_or_init(|| {
            tracing::info!("CUDA fused Q4_0/Q2_K/Q4_K gated projections enabled");
        });
    }
    Ok(Some(output))
}

/// Fuse indexed Q2_K/Q4_K gate and up projections for small decode batches.
///
/// Prompt routes keep Candle's grouped WMMA path; the MMVQ fusion is selected
/// only when at most eight expert routes are active.
#[allow(clippy::too_many_arguments)]
pub(super) fn indexed_silu_gate_up(
    input: &Tensor,
    expert_ids: &Tensor,
    gate: &Arc<QTensor>,
    up: &Arc<QTensor>,
    num_experts: usize,
    output_dim: usize,
    input_dim: usize,
) -> anyhow::Result<Option<Tensor>> {
    if !input.device().is_cuda()
        || input.dtype() != DType::F32
        || expert_ids.dtype() != DType::U32
        || !input.is_contiguous()
        || !expert_ids.is_contiguous()
    {
        return Ok(None);
    }
    let (tokens, input_rows, actual_input_dim) = input.dims3()?;
    let (id_tokens, top_k) = expert_ids.dims2()?;
    let routes = tokens.checked_mul(top_k);
    let dtype = gate.dtype();
    let expected_shape = [num_experts, output_dim, input_dim];
    if tokens == 0
        || id_tokens != tokens
        || input_rows != 1 && input_rows != top_k
        || actual_input_dim != input_dim
        || routes.is_none_or(|routes| routes == 0 || routes > 8 || routes > 65_535)
        || !matches!(dtype, GgmlDType::Q2K | GgmlDType::Q4K)
        || up.dtype() != dtype
        || gate.shape().dims() != expected_shape
        || up.shape().dims() != expected_shape
        || output_dim == 0
        || output_dim > i32::MAX as usize
        || input_dim > (i32::MAX as usize).saturating_sub(MATRIX_ROW_PADDING - 1)
        || num_experts == 0
        || num_experts > i32::MAX as usize
        || !input.device().same_device(expert_ids.device())
        || !input.device().same_device(&gate.device())
        || !input.device().same_device(&up.device())
    {
        return Ok(None);
    }
    let output = input.apply_op2_no_bwd(
        expert_ids,
        &FusedIndexedKQuantSwiGlu {
            gate: Arc::clone(gate),
            up: Arc::clone(up),
            dtype,
            tokens,
            input_rows,
            top_k,
            num_experts,
            input_dim,
            output_dim,
        },
    )?;
    INDEXED_KQUANT_FUSION_LOGGED.get_or_init(|| {
        tracing::info!("CUDA fused indexed Q2_K/Q4_K SwiGLU projections enabled");
    });
    Ok(Some(output))
}

/// Compute exact f32 SwiGLU, quantize it to Q8_1, and run the down projection.
///
/// Returns `None` outside single-token CUDA MMVQ so callers retain the normal
/// activation and projection path for prefill and unsupported GGUF types.
pub(super) fn silu_mul_down(
    gate: &Tensor,
    up: &Tensor,
    down: &QMatMul,
) -> anyhow::Result<Option<Tensor>> {
    if !gate.device().is_cuda()
        || gate.dtype() != DType::F32
        || up.dtype() != DType::F32
        || gate.shape() != up.shape()
        || !gate.is_contiguous()
        || !up.is_contiguous()
        || !gate.device().same_device(up.device())
    {
        return Ok(None);
    }
    let input_dim = gate.dim(candle_core::D::Minus1)?;
    if input_dim == 0
        || gate.elem_count() != input_dim
        || input_dim > (i32::MAX as usize).saturating_sub(MATRIX_ROW_PADDING - 1)
    {
        return Ok(None);
    }
    let QMatMul::QTensor(weight) = down else {
        return Ok(None);
    };
    let (output_dim, weight_input_dim) = weight.shape().dims2()?;
    if output_dim == 0
        || output_dim > i32::MAX as usize
        || weight_input_dim != input_dim
        || launcher(weight.dtype()).is_none()
        || !gate.device().same_device(&weight.device())
    {
        return Ok(None);
    }

    let output = gate.apply_op2_no_bwd(
        up,
        &FusedSiluDown {
            weight: Arc::clone(weight),
            input_dim,
            output_dim,
        },
    )?;
    SILU_DOWN_LOGGED.get_or_init(|| {
        tracing::info!("CUDA fused SwiGLU + quantized down projection decode path enabled");
    });
    Ok(Some(output))
}

/// Compute ReLU-squared, quantize it to Q8_1, and run a decode down projection.
pub(super) fn relu_squared_down(input: &Tensor, down: &QMatMul) -> anyhow::Result<Option<Tensor>> {
    if !input.device().is_cuda() || input.dtype() != DType::F32 || !input.is_contiguous() {
        return Ok(None);
    }
    let input_dim = input.dim(candle_core::D::Minus1)?;
    if input_dim == 0
        || input.elem_count() != input_dim
        || input_dim > (i32::MAX as usize).saturating_sub(MATRIX_ROW_PADDING - 1)
    {
        return Ok(None);
    }
    let QMatMul::QTensor(weight) = down else {
        return Ok(None);
    };
    let (output_dim, weight_input_dim) = weight.shape().dims2()?;
    if output_dim == 0
        || output_dim > i32::MAX as usize
        || weight_input_dim != input_dim
        || launcher(weight.dtype()).is_none()
        || !input.device().same_device(&weight.device())
    {
        return Ok(None);
    }
    let output = input.apply_op1_no_bwd(&FusedReluSquaredDown {
        weight: Arc::clone(weight),
        input_dim,
        output_dim,
    })?;
    RELU_SQUARED_DOWN_LOGGED.get_or_init(|| {
        tracing::info!("CUDA fused ReLU-squared + quantized down projection enabled");
    });
    Ok(Some(output))
}

/// Run grouped Q/K/V projections and apply Qwen2 bias + NEOX RoPE.
///
/// This returns `None` unless every part of the combined fast path is eligible,
/// allowing the caller to fall back to grouped projections plus portable
/// epilogues without changing model coverage.
#[allow(clippy::too_many_arguments)]
pub(super) fn forward_qkv(
    input: &Tensor,
    weights: [&QMatMul; 3],
    q_bias: &Tensor,
    k_bias: &Tensor,
    v_bias: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    start_pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    allow_f16_output: bool,
) -> anyhow::Result<Option<Vec<Tensor>>> {
    if input.dtype() != DType::F32 {
        return Ok(None);
    }
    let Some(epilogue) = QkvDecodeEpilogue::try_new(
        q_bias,
        k_bias,
        v_bias,
        cos,
        sin,
        start_pos,
        num_heads,
        num_kv_heads,
        head_dim,
        allow_f16_output,
    )?
    else {
        return Ok(None);
    };
    forward_impl(input, &weights, Some(epilogue))
}

fn forward_impl(
    input: &Tensor,
    weights: &[&QMatMul],
    qkv_epilogue: Option<QkvDecodeEpilogue>,
) -> anyhow::Result<Option<Vec<Tensor>>> {
    if !input.device().is_cuda()
        || !matches!(input.dtype(), DType::F16 | DType::F32)
        || weights.is_empty()
    {
        return Ok(None);
    }

    let input_dim = input.dim(candle_core::D::Minus1)?;
    if input_dim == 0 || input_dim > (i32::MAX as usize).saturating_sub(MATRIX_ROW_PADDING - 1) {
        return Ok(None);
    }
    let input_elements = input.elem_count();
    if !input_elements.is_multiple_of(input_dim) {
        return Ok(None);
    }
    let batch_size = input_elements / input_dim;
    if batch_size == 0 || batch_size > i32::MAX as usize {
        return Ok(None);
    }
    let mut quantized = Vec::with_capacity(weights.len());
    let mut rows = Vec::with_capacity(weights.len());
    let mut output_dim = 0usize;
    for weight in weights {
        let QMatMul::QTensor(weight) = weight else {
            return Ok(None);
        };
        let (weight_rows, weight_cols) = weight.shape().dims2()?;
        if weight_rows == 0
            || weight_rows > i32::MAX as usize
            || weight_cols != input_dim
            || launcher(weight.dtype()).is_none()
            || !input.device().same_device(&weight.device())
        {
            return Ok(None);
        }
        output_dim = output_dim
            .checked_add(weight_rows)
            .ok_or_else(|| anyhow::anyhow!("grouped MMVQ output size overflow"))?;
        quantized.push(Arc::clone(weight));
        rows.push(weight_rows);
    }
    if qkv_epilogue
        .as_ref()
        .is_some_and(|epilogue| rows.as_slice() != epilogue.output_dims())
        || qkv_epilogue.is_some() && batch_size != 1
    {
        return Ok(None);
    }

    if let Some(outputs) = forward_mmq_grouped(input, &quantized, &rows, input_dim, batch_size)? {
        return Ok(Some(outputs));
    }
    if batch_size > 8 {
        return Ok(None);
    }

    let packed = input.apply_op1_no_bwd(&GroupedQMatMul {
        weights: quantized,
        rows: rows.clone(),
        input_dim,
        output_dim,
        batch_size,
        qkv_epilogue,
    })?;
    let mut outputs = Vec::with_capacity(rows.len());
    let mut offset = 0usize;
    for rows in rows {
        outputs.push(packed.narrow(candle_core::D::Minus1, offset, rows)?);
        offset += rows;
    }
    FAST_PATH_LOGGED.get_or_init(|| {
        tracing::info!("CUDA grouped dense MMVQ path enabled");
    });
    Ok(Some(outputs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Module};

    fn values(elements: usize, multiplier: usize) -> Vec<f32> {
        (0..elements)
            .map(|index| ((index * multiplier % 97) as f32 - 48.0) / 29.0)
            .collect()
    }

    fn assert_close(
        actual: &Tensor,
        expected: &Tensor,
        tolerance: f32,
        label: &str,
    ) -> anyhow::Result<()> {
        let actual = actual
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        anyhow::ensure!(actual.len() == expected.len(), "{label} length differs");
        let max_diff = actual
            .iter()
            .zip(expected.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        anyhow::ensure!(
            max_diff <= tolerance,
            "{label} drifted by {max_diff}, tolerance is {tolerance}"
        );
        Ok(())
    }

    fn projection(
        device: &Device,
        dtype: GgmlDType,
        rows: usize,
        input_dim: usize,
        multiplier: usize,
    ) -> anyhow::Result<QMatMul> {
        let source = Tensor::from_vec(
            values(rows * input_dim, multiplier),
            (rows, input_dim),
            &Device::Cpu,
        )?;
        Ok(QMatMul::from_arc(Arc::new(QTensor::quantize_onto(
            &source, dtype, device,
        )?))?)
    }

    #[test]
    fn fused_k_quant_groups_match_individual_mmvq() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        let input_dim = 256;
        let input = Tensor::from_vec(values(input_dim, 11), (1, 1, input_dim), &device)?;
        for dtype in [GgmlDType::Q4_0, GgmlDType::Q2K, GgmlDType::Q4K] {
            let projections = [
                projection(&device, dtype, 128, input_dim, 17)?,
                projection(&device, dtype, 64, input_dim, 23)?,
                projection(&device, dtype, 32, input_dim, 31)?,
            ];
            let expected = projections
                .iter()
                .map(|projection| projection.forward(&input))
                .collect::<candle_core::Result<Vec<_>>>()?;
            let refs = projections.iter().collect::<Vec<_>>();
            let actual = forward(&input, &refs)?.expect("eligible fused K-quant group");
            for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
                assert_close(
                    actual,
                    expected,
                    0.0,
                    &format!("{dtype:?} grouped output {index}"),
                )?;
            }
        }
        Ok(())
    }

    #[test]
    fn fused_k_quant_gates_match_portable_epilogues() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        let input_dim = 256;
        let output_dim = 128;
        let input = Tensor::from_vec(values(input_dim, 11), (1, 1, input_dim), &device)?;
        for dtype in [GgmlDType::Q4_0, GgmlDType::Q2K, GgmlDType::Q4K] {
            let gate = projection(&device, dtype, output_dim, input_dim, 17)?;
            let up = projection(&device, dtype, output_dim, input_dim, 29)?;
            let gate_output = gate.forward(&input)?;
            let up_output = up.forward(&input)?;

            let expected_silu =
                super::super::indexed_moe_epilogue_cuda::silu_mul_exact(&gate_output, &up_output)?;
            let actual_silu =
                silu_gate_up(&input, &gate, &up)?.expect("eligible fused K-quant SwiGLU");
            assert_close(
                &actual_silu,
                &expected_silu,
                1e-6,
                &format!("{dtype:?} fused SwiGLU"),
            )?;

            let expected_gelu =
                (&candle_nn::Activation::NewGelu.forward(&gate_output)? * &up_output)?;
            let actual_gelu =
                gelu_gate_up(&input, &gate, &up)?.expect("eligible fused K-quant GeGLU");
            assert_close(
                &actual_gelu,
                &expected_gelu,
                2e-5,
                &format!("{dtype:?} fused GeGLU"),
            )?;
        }
        Ok(())
    }

    #[test]
    fn mixed_layout_k_quant_prefill_matches_individual_mmq() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        let input_dim = 256;
        let batch_size = 9;
        let projections = [
            projection(&device, GgmlDType::Q4K, 128, input_dim, 17)?,
            projection(&device, GgmlDType::Q6K, 64, input_dim, 29)?,
        ];
        let input = Tensor::from_vec(
            values(batch_size * input_dim, 11),
            (1, batch_size, input_dim),
            &device,
        )?;
        let expected = projections
            .iter()
            .map(|projection| projection.forward(&input))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let refs = projections.iter().collect::<Vec<_>>();
        let actual = forward(&input, &refs)?.expect("eligible mixed-layout shared MMQ");
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_close(
                actual,
                expected,
                0.0,
                &format!("mixed-layout MMQ output {index}"),
            )?;
        }
        Ok(())
    }

    #[test]
    fn fused_relu_squared_down_matches_portable_q4_0_path() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        let input_dim = 256;
        let input = Tensor::from_vec(values(input_dim, 17), (1, 1, input_dim), &device)?;
        let down = projection(&device, GgmlDType::Q4_0, 128, input_dim, 29)?;
        let relu = input.relu()?;
        let expected = down.forward(&relu.sqr()?)?;
        let actual = relu_squared_down(&input, &down)?
            .expect("eligible fused ReLU-squared Q4_0 down projection");
        assert_close(&actual, &expected, 0.0, "fused ReLU-squared down")?;
        Ok(())
    }

    #[test]
    fn grouped_mixed_k_quant_outputs_match_individual_mmvq() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        let input_dim = 256;
        let first_rows = 128;
        let second_rows = 64;
        let first_source = Tensor::from_vec(
            values(first_rows * input_dim, 17),
            (first_rows, input_dim),
            &Device::Cpu,
        )?;
        let second_source = Tensor::from_vec(
            values(second_rows * input_dim, 23),
            (second_rows, input_dim),
            &Device::Cpu,
        )?;
        let first = QMatMul::from_arc(Arc::new(QTensor::quantize_onto(
            &first_source,
            GgmlDType::Q4K,
            &device,
        )?))?;
        let second = QMatMul::from_arc(Arc::new(QTensor::quantize_onto(
            &second_source,
            GgmlDType::Q6K,
            &device,
        )?))?;

        let mut padded_input = vec![0.0f32; input_dim * 2];
        padded_input[input_dim..].copy_from_slice(&values(input_dim, 11));
        let input = Tensor::from_vec(padded_input, (2, 1, input_dim), &device)?.narrow(0, 1, 1)?;
        assert_ne!(input.layout().start_offset(), 0);

        let expected = [first.forward(&input)?, second.forward(&input)?];
        let actual = forward(&input, &[&first, &second])?.expect("eligible grouped MMVQ");
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let actual = actual
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let expected = expected
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(actual, expected, "grouped output {index} drifted");
        }
        Ok(())
    }

    #[test]
    fn fused_q8_group_outputs_match_individual_mmvq() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        let input_dim = 256;
        let row_counts = [128, 64, 32];
        let mut projections = Vec::new();
        for (index, rows) in row_counts.into_iter().enumerate() {
            let source = Tensor::from_vec(
                values(rows * input_dim, 17 + index * 6),
                (rows, input_dim),
                &Device::Cpu,
            )?;
            projections.push(QMatMul::from_arc(Arc::new(QTensor::quantize_onto(
                &source,
                GgmlDType::Q8_0,
                &device,
            )?))?);
        }
        let input = Tensor::from_vec(values(input_dim, 11), (1, 1, input_dim), &device)?;
        let expected = projections
            .iter()
            .map(|projection| projection.forward(&input))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let refs = projections.iter().collect::<Vec<_>>();
        let actual = forward(&input, &refs)?.expect("eligible fused Q8_0 group");
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let actual = actual
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let expected = expected
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(actual, expected, "fused Q8_0 output {index} drifted");
        }
        Ok(())
    }

    #[test]
    fn fused_q8_swiglu_matches_separate_exact_kernels() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        let input_dim = 256;
        let output_dim = 128;
        let projection = |multiplier| -> anyhow::Result<QMatMul> {
            let source = Tensor::from_vec(
                values(output_dim * input_dim, multiplier),
                (output_dim, input_dim),
                &Device::Cpu,
            )?;
            Ok(QMatMul::from_arc(Arc::new(QTensor::quantize_onto(
                &source,
                GgmlDType::Q8_0,
                &device,
            )?))?)
        };
        let gate = projection(17)?;
        let up = projection(29)?;
        let input = Tensor::from_vec(values(input_dim, 11), (1, 1, input_dim), &device)?;
        let expected_gate = gate.forward(&input)?;
        let expected_up = up.forward(&input)?;
        let expected =
            super::super::indexed_moe_epilogue_cuda::silu_mul_exact(&expected_gate, &expected_up)?;
        let actual = silu_gate_up(&input, &gate, &up)?.expect("eligible fused Q8_0 SwiGLU");
        let actual = actual
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual, expected, "fused Q8_0 SwiGLU drifted");
        Ok(())
    }

    #[test]
    fn grouped_q8_prefill_outputs_match_individual_mmq() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        let input_dim = 256;
        let batch_size = 9;
        let row_counts = [128, 64];
        let mut projections = Vec::new();
        for (index, rows) in row_counts.into_iter().enumerate() {
            let source = Tensor::from_vec(
                values(rows * input_dim, 17 + index * 6),
                (rows, input_dim),
                &Device::Cpu,
            )?;
            projections.push(QMatMul::from_arc(Arc::new(QTensor::quantize_onto(
                &source,
                GgmlDType::Q8_0,
                &device,
            )?))?);
        }
        let input = Tensor::from_vec(
            values(batch_size * input_dim, 11),
            (1, batch_size, input_dim),
            &device,
        )?;
        let expected = projections
            .iter()
            .map(|projection| projection.forward(&input))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let refs = projections.iter().collect::<Vec<_>>();
        let actual = forward(&input, &refs)?.expect("eligible shared Q8_0 MMQ");
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let actual = actual
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let expected = expected
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(actual, expected, "shared Q8_0 MMQ output {index} drifted");
        }
        Ok(())
    }

    #[test]
    fn grouped_q8_small_batch_outputs_match_individual_mmvq() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        let input_dim = 256;
        let batch_size = 4;
        let row_counts = [128, 64];
        let mut projections = Vec::new();
        for (index, rows) in row_counts.into_iter().enumerate() {
            let source = Tensor::from_vec(
                values(rows * input_dim, 19 + index * 6),
                (rows, input_dim),
                &Device::Cpu,
            )?;
            projections.push(QMatMul::from_arc(Arc::new(QTensor::quantize_onto(
                &source,
                GgmlDType::Q8_0,
                &device,
            )?))?);
        }
        let input = Tensor::from_vec(
            values(batch_size * input_dim, 13),
            (batch_size, 1, input_dim),
            &device,
        )?;
        let expected = projections
            .iter()
            .map(|projection| projection.forward(&input))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let refs = projections.iter().collect::<Vec<_>>();
        let actual = forward(&input, &refs)?.expect("eligible grouped Q8_0 small batch");
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let actual = actual
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let expected = expected
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(actual, expected, "small-batch Q8_0 output {index} drifted");
        }
        Ok(())
    }

    #[test]
    fn f16_input_f32_output_matches_explicit_cast_then_mmvq() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        let input_dim = 256;
        let output_dim = 128;
        let source = Tensor::from_vec(
            values(output_dim * input_dim, 17),
            (output_dim, input_dim),
            &Device::Cpu,
        )?;
        let projection = QMatMul::from_arc(Arc::new(QTensor::quantize_onto(
            &source,
            GgmlDType::Q4K,
            &device,
        )?))?;

        let mut padded_input = vec![0.0f32; input_dim * 2];
        padded_input[input_dim..].copy_from_slice(&values(input_dim, 11));
        let input = Tensor::from_vec(padded_input, (2, 1, input_dim), &device)?
            .to_dtype(DType::F16)?
            .narrow(0, 1, 1)?;
        assert_ne!(input.layout().start_offset(), 0);

        let expected = projection.forward(&input.to_dtype(DType::F32)?)?;
        let actual = forward_one_to_f32(&input, &projection)?.expect("eligible f16 MMVQ");
        assert_eq!(actual.dtype(), DType::F32);
        let actual = actual
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn fused_silu_quantized_down_matches_separate_exact_kernels() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        let input_dim = 256;
        let output_dim = 128;
        let source = Tensor::from_vec(
            values(output_dim * input_dim, 23),
            (output_dim, input_dim),
            &Device::Cpu,
        )?;
        let down = QMatMul::from_arc(Arc::new(QTensor::quantize_onto(
            &source,
            GgmlDType::Q4K,
            &device,
        )?))?;
        let offset_row = |multiplier: usize| -> candle_core::Result<Tensor> {
            let mut padded = vec![0.0f32; input_dim * 2];
            padded[input_dim..].copy_from_slice(&values(input_dim, multiplier));
            Tensor::from_vec(padded, (2, 1, input_dim), &device)?.narrow(0, 1, 1)
        };
        let gate = offset_row(17)?;
        let up = offset_row(31)?;
        let activated = super::super::indexed_moe_epilogue_cuda::silu_mul_exact(&gate, &up)?;
        let expected = down.forward(&activated)?;
        let actual = silu_mul_down(&gate, &up, &down)?.expect("eligible fused SwiGLU down");

        let actual = actual
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn grouped_qkv_epilogue_matches_bias_then_neox_rope() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        let input_dim = 256;
        let head_dim = 32;
        let num_heads = 2;
        let num_kv_heads = 1;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let q_source = Tensor::from_vec(
            values(q_dim * input_dim, 13),
            (q_dim, input_dim),
            &Device::Cpu,
        )?;
        let k_source = Tensor::from_vec(
            values(kv_dim * input_dim, 19),
            (kv_dim, input_dim),
            &Device::Cpu,
        )?;
        let v_source = Tensor::from_vec(
            values(kv_dim * input_dim, 29),
            (kv_dim, input_dim),
            &Device::Cpu,
        )?;
        let q_proj = QMatMul::from_arc(Arc::new(QTensor::quantize_onto(
            &q_source,
            GgmlDType::Q4K,
            &device,
        )?))?;
        let k_proj = QMatMul::from_arc(Arc::new(QTensor::quantize_onto(
            &k_source,
            GgmlDType::Q6K,
            &device,
        )?))?;
        let v_proj = QMatMul::from_arc(Arc::new(QTensor::quantize_onto(
            &v_source,
            GgmlDType::Q4_0,
            &device,
        )?))?;

        let mut padded_input = vec![0.0f32; input_dim * 2];
        padded_input[input_dim..].copy_from_slice(&values(input_dim, 7));
        let input = Tensor::from_vec(padded_input, (2, 1, input_dim), &device)?.narrow(0, 1, 1)?;

        let offset_tensor = |elements: usize, multiplier: usize| -> candle_core::Result<Tensor> {
            let mut padded = vec![0.0f32; elements * 2];
            padded[elements..].copy_from_slice(&values(elements, multiplier));
            Tensor::from_vec(padded, (2, elements), &device)?
                .narrow(0, 1, 1)?
                .squeeze(0)
        };
        let q_bias = offset_tensor(q_dim, 31)?;
        let k_bias = offset_tensor(kv_dim, 37)?;
        let v_bias = offset_tensor(kv_dim, 41)?;

        let positions = 8;
        let half_dim = head_dim / 2;
        let rope_values = |phase: f32| {
            (0..(positions + 1) * half_dim)
                .map(|index| ((index as f32 + phase) * 0.071).sin())
                .collect::<Vec<_>>()
        };
        let cos = Tensor::from_vec(rope_values(0.25), (positions + 1, half_dim), &device)?
            .narrow(0, 1, positions)?;
        let sin = Tensor::from_vec(rope_values(1.75), (positions + 1, half_dim), &device)?
            .narrow(0, 1, positions)?;
        let start_pos = 5;

        let expected_raw = [
            q_proj.forward(&input)?,
            k_proj.forward(&input)?,
            v_proj.forward(&input)?,
        ];
        let biased_q = expected_raw[0]
            .broadcast_add(&q_bias)?
            .reshape((1, num_heads, 1, head_dim))?;
        let expected_k =
            expected_raw[1]
                .broadcast_add(&k_bias)?
                .reshape((1, num_kv_heads, 1, head_dim))?;
        let expected_v_f32 =
            expected_raw[2]
                .broadcast_add(&v_bias)?
                .reshape((1, num_kv_heads, 1, head_dim))?;
        let cos_row = cos.narrow(0, start_pos, 1)?;
        let sin_row = sin.narrow(0, start_pos, 1)?;
        let expected_q_f32 =
            candle_nn::rotary_emb::rope(&biased_q.contiguous()?, &cos_row, &sin_row)?;
        let expected_k_f32 =
            candle_nn::rotary_emb::rope(&expected_k.contiguous()?, &cos_row, &sin_row)?;
        let expected_q = expected_q_f32.to_dtype(DType::F16)?;
        let expected_k = expected_k_f32.to_dtype(DType::F16)?;
        let expected_v = expected_v_f32.to_dtype(DType::F16)?;

        let mut actual = forward_qkv(
            &input,
            [&q_proj, &k_proj, &v_proj],
            &q_bias,
            &k_bias,
            &v_bias,
            &cos,
            &sin,
            start_pos,
            num_heads,
            num_kv_heads,
            head_dim,
            true,
        )?
        .expect("eligible grouped QKV epilogue");
        let actual_v = actual
            .pop()
            .unwrap()
            .reshape((1, num_kv_heads, 1, head_dim))?;
        let actual_k = actual
            .pop()
            .unwrap()
            .reshape((1, num_kv_heads, 1, head_dim))?;
        let actual_q = actual.pop().unwrap().reshape((1, num_heads, 1, head_dim))?;

        for (name, actual, expected) in [
            ("Q", actual_q, expected_q),
            ("K", actual_k, expected_k),
            ("V", actual_v, expected_v),
        ] {
            let actual = actual
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<half::f16>()?;
            let expected = expected
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<half::f16>()?;
            assert_eq!(actual, expected, "{name} fused QKV f16 epilogue drifted");
        }

        // Quantized KV caches consume f32 projections through their portable
        // attention path. Disallowing f16 output must retain the exact fused
        // bias/RoPE result without disabling grouped MMVQ itself.
        let mut actual = forward_qkv(
            &input,
            [&q_proj, &k_proj, &v_proj],
            &q_bias,
            &k_bias,
            &v_bias,
            &cos,
            &sin,
            start_pos,
            num_heads,
            num_kv_heads,
            head_dim,
            false,
        )?
        .expect("eligible grouped f32 QKV epilogue");
        let actual_v = actual
            .pop()
            .unwrap()
            .reshape((1, num_kv_heads, 1, head_dim))?;
        let actual_k = actual
            .pop()
            .unwrap()
            .reshape((1, num_kv_heads, 1, head_dim))?;
        let actual_q = actual.pop().unwrap().reshape((1, num_heads, 1, head_dim))?;
        for (name, actual, expected) in [
            ("Q", actual_q, expected_q_f32),
            ("K", actual_k, expected_k_f32),
            ("V", actual_v, expected_v_f32),
        ] {
            assert_eq!(actual.dtype(), DType::F32);
            let actual = actual
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let expected = expected
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(actual, expected, "{name} fused QKV f32 epilogue drifted");
        }
        Ok(())
    }
}
