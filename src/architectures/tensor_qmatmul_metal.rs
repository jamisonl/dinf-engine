use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{
    CpuStorage, CustomOp2, CustomOp3, D, DType, Device, Layout, MetalStorage, Result, Shape, Tensor,
};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::{MTLCompileOptions, MTLDevice, MTLGPUFamily, MTLLanguageVersion};

// Metal 4 tensor geometry and K-quant dequantizers are adapted from llama.cpp's
// ggml Metal backend: https://github.com/ggml-org/llama.cpp
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// The exact-width verification layouts are adapted from MTPLX by Youssof
// Altoukhi (Apache-2.0): https://github.com/youssofal/mtplx
const TENSOR_QMATMUL_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    half d;
    uint8_t qs[16];
} block_q4_0;

typedef struct {
    half d;
    half m;
    uint8_t qh[4];
    uint8_t qs[16];
} block_q5_1;

typedef struct {
    half d;
    int8_t qs[32];
} block_q8_0;

typedef struct {
    union {
        struct { half d; half dmin; };
        half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

typedef struct {
    union {
        struct { half d; half dmin; };
        half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K/8];
    uint8_t qs[QK_K/2];
} block_q5_K;

typedef struct {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t scales[QK_K/16];
    half d;
} block_q6_K;

void dequantize_q4_0(device const block_q4_0 *xb, short il, thread half4x4 &reg) {
    const half d = xb->d;
    FOR_UNROLL (short i = 0; i < 16; ++i) {
        const short q = il == 0 ? (xb->qs[i] & 0x0F) : (xb->qs[i] >> 4);
        reg[i / 4][i % 4] = d * half(q - 8);
    }
}

void dequantize_q5_1(device const block_q5_1 *xb, short il, thread half4x4 &reg) {
    device const uint16_t *qs = ((device const uint16_t *)xb + 4);
    const float d = xb->d;
    const float m = xb->m;
    const ushort mask = il ? 0x00F0 : 0x000F;
    const uint32_t qh = *((device const uint32_t *)xb->qh);
    const int x_mv = il ? 4 : 0;
    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ? 0 : 4;
    FOR_UNROLL (int i = 0; i < 8; ++i) {
        const uint8_t xh_0 = ((qh >> (gh_mv + 2 * i)) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2 * i + 1)) << gh_bk) & 0x10;
        const int32_t x0 = (((qs[i] & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);
        reg[i / 2][2 * (i % 2)] = d * x0 + m;
        reg[i / 2][2 * (i % 2) + 1] = d * x1 + m;
    }
}

void dequantize_q8_0(device const block_q8_0 *xb, short il, thread half4x4 &reg) {
    const half d = xb->d;
    FOR_UNROLL (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = xb->qs[i + 16 * il] * d;
    }
}

void dequantize_q8_0_pair(
    device const block_q8_0 *xb,
    short,
    thread half4x4 &low,
    thread half4x4 &high) {
    // Q8_0 blocks are 2-byte aligned: load byte pairs as ushorts.
    const half d = xb->d;
    device const ushort *qs = (device const ushort *)xb->qs;
    FOR_UNROLL (short pair = 0; pair < 8; ++pair) {
        const ushort lo = qs[pair];
        const ushort hi = qs[8 + pair];
        FOR_UNROLL (short byte = 0; byte < 2; ++byte) {
            const short i = 2 * pair + byte;
            low[i / 4][i % 4] = half(as_type<char>(uchar(lo >> (8 * byte)))) * d;
            high[i / 4][i % 4] = half(as_type<char>(uchar(hi >> (8 * byte)))) * d;
        }
    }
}

static inline uchar2 get_scale_min_k4(int j, int k, device const uchar *q) {
    return j < 4
        ? uchar2{uchar(q[j + k] & 63), uchar(q[j + 4 + k] & 63)}
        : uchar2{
            uchar((q[j + 4 + k] & 0xF) | ((q[j - 4 + k] & 0xC0) >> 2)),
            uchar((q[j + 4 + k] >> 4) | ((q[j + k] & 0xC0) >> 2))
        };
}

void dequantize_q4_K(device const block_q4_K *xb, short il, thread half4x4 &reg) {
    device const uchar *q = xb->qs;
    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4(is, il / 2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.h;
    const float min_value = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min_value * sc[1];
    const ushort mask = il < 2 ? 0x0F : 0xF0;
    FOR_UNROLL (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = dl * (q[i] & mask) - ml;
    }
}

// Decode both 16-value halves of one Q4_K sub-block together. Verification
// and bounded prefill can share its scale and minimum calculations.
void dequantize_q4_K_pair(
    device const block_q4_K *xb,
    short il,
    thread half4x4 &low,
    thread half4x4 &high) {
    device const uchar *q = xb->qs + (il / 4) * 32;
    const short local = il & 3;
    const short is = (il / 4) * 2;
    const uchar2 sc = get_scale_min_k4(is, local / 2, xb->scales);
    const float d = local < 2 ? xb->d : xb->d / 16.h;
    const float dl = d * sc[0];
    const float ml = float(xb->dmin) * sc[1];
    // Two 16-byte loads instead of 32 byte loads; qs is 16-byte aligned.
    const uint4 packed_low = ((device const uint4 *)q)[0];
    const uint4 packed_high = ((device const uint4 *)q)[1];
    // Keep the masked high-nibble value (16 * q) so results match
    // dequantize_q4_K bit for bit, including its half-precision d / 16.
    const uint mask = local < 2 ? 0x0F : 0xF0;
    const uint4 shifts = uint4(0, 8, 16, 24);
    FOR_UNROLL (short word = 0; word < 4; ++word) {
        const uint4 lo = (uint4(packed_low[word]) >> shifts) & mask;
        const uint4 hi = (uint4(packed_high[word]) >> shifts) & mask;
        low[word] = half4(dl * float4(lo) - ml);
        high[word] = half4(dl * float4(hi) - ml);
    }
}

void dequantize_q5_K(device const block_q5_K *xb, short il, thread half4x4 &reg) {
    device const uint8_t *q = xb->qs;
    device const uint8_t *qh = xb->qh;
    short is = (il / 4) * 2;
    q = q + 32 * (il / 4) + 16 * (il & 1);
    qh = qh + 16 * (il & 1);
    const uint8_t ul = 1 << (il / 2);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4(is, il / 2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.f;
    const float min_value = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min_value * sc[1];
    const ushort mask = il < 2 ? 0x0F : 0xF0;
    const float qh_value = il < 2 ? 16.f : 256.f;
    FOR_UNROLL (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_value : 0)) - ml;
    }
}

void dequantize_q6_K(device const block_q6_K *xb, short il, thread half4x4 &reg) {
    const half d_all = xb->d;
    device const uint8_t *ql = xb->ql;
    device const uint8_t *qh = xb->qh;
    device const int8_t *scales = xb->scales;
    ql = ql + 64 * (il / 8) + 32 * ((il / 2) & 1) + 16 * (il & 1);
    qh = qh + 32 * (il / 8) + 16 * (il & 1);
    const float sc = scales[(il % 2) + 2 * (il / 2)];
    il = (il / 2) & 3;
    const uint16_t kmask1 = il > 1 ? (il > 2 ? 192 : 48) : (il > 0 ? 12 : 3);
    const uint16_t kmask2 = il > 1 ? 0xF0 : 0x0F;
    const float coef = il > 1 ? 1.f / 16.f : 1.f;
    const float ml = d_all * sc * 32.f;
    const float dl = d_all * sc * coef;
    FOR_UNROLL (int i = 0; i < 16; ++i) {
        const half q = il & 1
            ? ((ql[i] & kmask2) | ((qh[i] & kmask1) << 2))
            : ((ql[i] & kmask2) | ((qh[i] & kmask1) << 4));
        reg[i / 4][i % 4] = dl * q - ml;
    }
}

// Two consecutive 16-value Q6_K groups for the K32 split kernels.
// `il` is even, so both groups share one 32-byte ql and qh span and the
// bit selection; Q6_K blocks are 2-byte aligned, so load ushorts. The
// arithmetic matches dequantize_q6_K exactly.
void dequantize_q6_K_pair(
    device const block_q6_K *xb,
    short il,
    thread half4x4 &low,
    thread half4x4 &high) {
    const half d_all = xb->d;
    device const ushort *ql = (device const ushort *)(xb->ql + 64 * (il / 8) + 32 * ((il / 2) & 1));
    device const ushort *qh = (device const ushort *)(xb->qh + 32 * (il / 8));
    const float sc_low = xb->scales[2 * (il / 2)];
    const float sc_high = xb->scales[1 + 2 * (il / 2)];
    const short group = (il / 2) & 3;
    const ushort kmask1 = group > 1 ? (group > 2 ? 192 : 48) : (group > 0 ? 12 : 3);
    const ushort kmask2 = group > 1 ? 0xF0 : 0x0F;
    const ushort shift = group & 1 ? 2 : 4;
    const float coef = group > 1 ? 1.f / 16.f : 1.f;
    const float ml_low = d_all * sc_low * 32.f;
    const float dl_low = d_all * sc_low * coef;
    const float ml_high = d_all * sc_high * 32.f;
    const float dl_high = d_all * sc_high * coef;
    FOR_UNROLL (short pair = 0; pair < 16; ++pair) {
        const ushort l = ql[pair];
        const ushort h = qh[pair];
        FOR_UNROLL (short byte = 0; byte < 2; ++byte) {
            const ushort lb = (l >> (8 * byte)) & 0xFF;
            const ushort hb = (h >> (8 * byte)) & 0xFF;
            const half q = (lb & kmask2) | ((hb & kmask1) << shift);
            const short i = 2 * pair + byte;
            if (i < 16) {
                low[i / 4][i % 4] = dl_low * q - ml_low;
            } else {
                high[(i - 16) / 4][(i - 16) % 4] = dl_high * q - ml_high;
            }
        }
    }
}

// Two rows per SIMD group for scalar Q6_K. Two SIMD groups therefore emit
// four rows per threadgroup while sharing the same input working set.
kernel void dinf_scalar_q6k_r2_sg2(
    device const block_q6_K *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &k_dim [[buffer(3)]],
    constant uint &out_dim [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr ushort output_rows = 2;
    constexpr uchar mask1 = 0x03;
    constexpr uchar mask2 = 0x0C;
    constexpr uchar mask3 = 0x30;
    constexpr uchar mask4 = 0xC0;
    const uint blocks_per_row = k_dim / QK_K;
    const uint first_row = (group.x * 2 + simd_group) * output_rows;
    const ushort thread_pair = lane / 2;
    const ushort block_parity = lane % 2;
    const ushort input_half = thread_pair / 8;
    const ushort local = thread_pair % 8;
    const ushort local4 = 4 * local;
    const ushort scale_index = 8 * input_half + local4 / 16;
    const ushort input_offset = 128 * input_half + local4;
    const ushort low_offset = 64 * input_half + local4;
    const ushort high_offset = 32 * input_half + local4;
    float sums[output_rows] = {0.0f};
    float input_values[16];

    for (uint block_index = block_parity; block_index < blocks_per_row; block_index += 2) {
        device const float *input_block = input + block_index * QK_K + input_offset;
        FOR_UNROLL (ushort i = 0; i < 4; ++i) {
            input_values[4 * i] = input_block[i];
            input_values[4 * i + 1] = input_block[i + 32];
            input_values[4 * i + 2] = input_block[i + 64];
            input_values[4 * i + 3] = input_block[i + 96];
        }
        FOR_UNROLL (ushort row = 0; row < output_rows; ++row) {
            if (first_row + row < out_dim) {
                device const block_q6_K *weight =
                    weights + (first_row + row) * blocks_per_row + block_index;
                device const uchar *low1 = weight->ql + low_offset;
                device const uchar *low2 = low1 + 32;
                device const uchar *high = weight->qh + high_offset;
                device const int8_t *scales = weight->scales + scale_index;
                float4 dot = 0.0f;
                FOR_UNROLL (ushort i = 0; i < 4; ++i) {
                    dot[0] += input_values[4 * i]
                        * (int8_t((low1[i] & 0xF) | ((high[i] & mask1) << 4)) - 32);
                    dot[1] += input_values[4 * i + 1]
                        * (int8_t((low2[i] & 0xF) | ((high[i] & mask2) << 2)) - 32);
                    dot[2] += input_values[4 * i + 2]
                        * (int8_t((low1[i] >> 4) | (high[i] & mask3)) - 32);
                    dot[3] += input_values[4 * i + 3]
                        * (int8_t((low2[i] >> 4) | ((high[i] & mask4) >> 2)) - 32);
                }
                sums[row] += float(weight->d)
                    * (dot[0] * scales[0] + dot[1] * scales[2]
                        + dot[2] * scales[4] + dot[3] * scales[6]);
            }
        }
    }

    FOR_UNROLL (ushort row = 0; row < output_rows; ++row) {
        const float total = simd_sum(sums[row]);
        if (lane == 0 && first_row + row < out_dim) {
            output[first_row + row] = total;
        }
    }
}

// Single-token Q6_K projection with a residual epilogue. This uses the same
// row mapping and accumulation order as dinf_scalar_q6k_r2_sg2, then adds the
// corresponding residual element before storing. Qwen3.5 uses this for every
// dense FFN down projection, avoiding a second allocation and dispatch.
kernel void dinf_scalar_q6k_residual_r2_sg2(
    device const block_q6_K *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device const float *residual [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &k_dim [[buffer(4)]],
    constant uint &out_dim [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr ushort output_rows = 2;
    constexpr uchar mask1 = 0x03;
    constexpr uchar mask2 = 0x0C;
    constexpr uchar mask3 = 0x30;
    constexpr uchar mask4 = 0xC0;
    const uint blocks_per_row = k_dim / QK_K;
    const uint first_row = (group.x * 2 + simd_group) * output_rows;
    const ushort thread_pair = lane / 2;
    const ushort block_parity = lane % 2;
    const ushort input_half = thread_pair / 8;
    const ushort local = thread_pair % 8;
    const ushort local4 = 4 * local;
    const ushort scale_index = 8 * input_half + local4 / 16;
    const ushort input_offset = 128 * input_half + local4;
    const ushort low_offset = 64 * input_half + local4;
    const ushort high_offset = 32 * input_half + local4;
    float sums[output_rows] = {0.0f};
    float input_values[16];

    for (uint block_index = block_parity; block_index < blocks_per_row; block_index += 2) {
        device const float *input_block = input + block_index * QK_K + input_offset;
        FOR_UNROLL (ushort i = 0; i < 4; ++i) {
            input_values[4 * i] = input_block[i];
            input_values[4 * i + 1] = input_block[i + 32];
            input_values[4 * i + 2] = input_block[i + 64];
            input_values[4 * i + 3] = input_block[i + 96];
        }
        FOR_UNROLL (ushort row = 0; row < output_rows; ++row) {
            if (first_row + row < out_dim) {
                device const block_q6_K *weight =
                    weights + (first_row + row) * blocks_per_row + block_index;
                device const uchar *low1 = weight->ql + low_offset;
                device const uchar *low2 = low1 + 32;
                device const uchar *high = weight->qh + high_offset;
                device const int8_t *scales = weight->scales + scale_index;
                float4 dot = 0.0f;
                FOR_UNROLL (ushort i = 0; i < 4; ++i) {
                    dot[0] += input_values[4 * i]
                        * (int8_t((low1[i] & 0xF) | ((high[i] & mask1) << 4)) - 32);
                    dot[1] += input_values[4 * i + 1]
                        * (int8_t((low2[i] & 0xF) | ((high[i] & mask2) << 2)) - 32);
                    dot[2] += input_values[4 * i + 2]
                        * (int8_t((low1[i] >> 4) | (high[i] & mask3)) - 32);
                    dot[3] += input_values[4 * i + 3]
                        * (int8_t((low2[i] >> 4) | ((high[i] & mask4) >> 2)) - 32);
                }
                sums[row] += float(weight->d)
                    * (dot[0] * scales[0] + dot[1] * scales[2]
                        + dot[2] * scales[4] + dot[3] * scales[6]);
            }
        }
    }

    FOR_UNROLL (ushort row = 0; row < output_rows; ++row) {
        const float total = simd_sum(sums[row]);
        if (lane == 0 && first_row + row < out_dim) {
            output[first_row + row] = total + residual[first_row + row];
        }
    }
}

// Single-token Q8_0 projection with a residual epilogue. The row mapping and
// accumulation order match Candle's kernel_mul_mv_q8_0_f32 exactly, while the
// final store folds in the DeltaNet residual.
kernel void dinf_scalar_q8_0_residual_r4_sg2(
    device const block_q8_0 *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device const float *residual [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &k_dim [[buffer(4)]],
    constant uint &out_dim [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr ushort output_rows = 4;
    constexpr ushort values_per_lane = 8;
    const uint blocks_per_row = k_dim / 32;
    const uint first_row = (group.x * 2 + simd_group) * output_rows;
    const ushort block_start = lane / 4;
    const ushort quant_offset = values_per_lane * (lane % 4);
    device const float *input_block =
        input + block_start * 32 + quant_offset;
    float sums[output_rows] = {0.0f};

    for (uint block = block_start; block < blocks_per_row; block += 8) {
        float input_values[values_per_lane];
        FOR_UNROLL (ushort index = 0; index < values_per_lane; ++index) {
            input_values[index] = input_block[index];
        }
        FOR_UNROLL (ushort row = 0; row < output_rows; ++row) {
            if (first_row + row < out_dim) {
                device const block_q8_0 *weight =
                    weights + (first_row + row) * blocks_per_row + block;
                device const int8_t *quants = weight->qs + quant_offset;
                float block_sum = 0.0f;
                FOR_UNROLL (ushort index = 0; index < values_per_lane; ++index) {
                    block_sum += quants[index] * input_values[index];
                }
                sums[row] += block_sum * float(weight->d);
            }
        }
        input_block += values_per_lane * 32;
    }

    FOR_UNROLL (ushort row = 0; row < output_rows; ++row) {
        const float total = simd_sum(sums[row]);
        if (lane == 0 && first_row + row < out_dim) {
            output[first_row + row] = total + residual[first_row + row];
        }
    }
}

// Qwen3.5 SwiGLU decode packs equal-width Q6_K gate/up matrices row-wise.
// Compute both projections for each intermediate row while the input values
// are live in registers, then apply SiLU and write only the activated value.
// This removes one quantized dispatch and the two full gate/up output buffers.
kernel void dinf_scalar_q6k_swiglu_r2_sg2(
    device const block_q6_K *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &k_dim [[buffer(3)]],
    constant uint &intermediate_dim [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr ushort output_rows = 2;
    constexpr uchar mask1 = 0x03;
    constexpr uchar mask2 = 0x0C;
    constexpr uchar mask3 = 0x30;
    constexpr uchar mask4 = 0xC0;
    const uint blocks_per_row = k_dim / QK_K;
    const uint first_row = (group.x * 2 + simd_group) * output_rows;
    const ushort thread_pair = lane / 2;
    const ushort block_parity = lane % 2;
    const ushort input_half = thread_pair / 8;
    const ushort local = thread_pair % 8;
    const ushort local4 = 4 * local;
    const ushort scale_index = 8 * input_half + local4 / 16;
    const ushort input_offset = 128 * input_half + local4;
    const ushort low_offset = 64 * input_half + local4;
    const ushort high_offset = 32 * input_half + local4;
    float gate_sums[output_rows] = {0.0f};
    float up_sums[output_rows] = {0.0f};
    float input_values[16];

    for (uint block_index = block_parity; block_index < blocks_per_row; block_index += 2) {
        device const float *input_block = input + block_index * QK_K + input_offset;
        FOR_UNROLL (ushort i = 0; i < 4; ++i) {
            input_values[4 * i] = input_block[i];
            input_values[4 * i + 1] = input_block[i + 32];
            input_values[4 * i + 2] = input_block[i + 64];
            input_values[4 * i + 3] = input_block[i + 96];
        }
        FOR_UNROLL (ushort row = 0; row < output_rows; ++row) {
            const uint output_row = first_row + row;
            if (output_row < intermediate_dim) {
                FOR_UNROLL (ushort projection = 0; projection < 2; ++projection) {
                    const uint weight_row = output_row + projection * intermediate_dim;
                    device const block_q6_K *weight =
                        weights + weight_row * blocks_per_row + block_index;
                    device const uchar *low1 = weight->ql + low_offset;
                    device const uchar *low2 = low1 + 32;
                    device const uchar *high = weight->qh + high_offset;
                    device const int8_t *scales = weight->scales + scale_index;
                    float4 dot = 0.0f;
                    FOR_UNROLL (ushort i = 0; i < 4; ++i) {
                        dot[0] += input_values[4 * i]
                            * (int8_t((low1[i] & 0xF) | ((high[i] & mask1) << 4)) - 32);
                        dot[1] += input_values[4 * i + 1]
                            * (int8_t((low2[i] & 0xF) | ((high[i] & mask2) << 2)) - 32);
                        dot[2] += input_values[4 * i + 2]
                            * (int8_t((low1[i] >> 4) | (high[i] & mask3)) - 32);
                        dot[3] += input_values[4 * i + 3]
                            * (int8_t((low2[i] >> 4) | ((high[i] & mask4) >> 2)) - 32);
                    }
                    const float partial = float(weight->d)
                        * (dot[0] * scales[0] + dot[1] * scales[2]
                            + dot[2] * scales[4] + dot[3] * scales[6]);
                    if (projection == 0) {
                        gate_sums[row] += partial;
                    } else {
                        up_sums[row] += partial;
                    }
                }
            }
        }
    }

    FOR_UNROLL (ushort row = 0; row < output_rows; ++row) {
        const float gate = simd_sum(gate_sums[row]);
        const float up = simd_sum(up_sums[row]);
        if (lane == 0 && first_row + row < intermediate_dim) {
            output[first_row + row] = (gate / (1.0f + exp(-gate))) * up;
        }
    }
}

template<
    typename block_q,
    short nl,
    short tile_tokens,
    void (*dequantize)(device const block_q *, short, thread half4x4 &),
    short weight_steps = 1,
    void (*dequantize_pair)(device const block_q *, short, thread half4x4 &, thread half4x4 &) = nullptr>
kernel void dinf_tensor_qmatmul(
    device const uchar *weights [[buffer(0)]],
    device float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &k_dim [[buffer(3)]],
    constant uint &out_dim [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    constant uint &weight_row_bytes [[buffer(6)]],
    constant uint &relu_squared_epilogue [[buffer(7)]],
    threadgroup half *shared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort thread_index [[thread_index_in_threadgroup]]) {
    constexpr int tile_rows = 64;
    constexpr int tile_k = 32;
    constexpr int k_chunks = 2;

    const int row_start = group.y * tile_rows;
    const int token_start = group.x * tile_tokens;
    auto weight_tile = tensor(shared, dextents<int32_t, 2>(tile_k, tile_rows));
    auto input_tensor = tensor(
        input,
        dextents<int32_t, 2>(int(k_dim), int(tokens)),
        array<int, 2>({1, int(k_dim)}));

    // Q8_0 needs strict float-input precision. Relaxed MPP matmul on M5
    // exceeds its F64-reference error bound even with a float destination.
    // Q8_0 prefill tiles use dinf_tensor_qmatmul_half_input instead.
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            tile_tokens,
            tile_rows,
            tile_k,
            false,
            true,
            !is_same<block_q, block_q8_0>::value,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> matmul;
    auto destination = matmul.template get_destination_cooperative_tensor<
        decltype(input_tensor),
        decltype(weight_tile),
        float>();

    // Stage several weight tiles with one pair of barriers, but retain the
    // original 32-value matmul calls in order. A wider matmul would change
    // the reduction. Each tile also keeps the original shared-memory stride.
    for (int loop_k = 0; loop_k < int(k_dim); loop_k += tile_k * weight_steps) {
        const int row = thread_index / k_chunks;
        const int k_chunk = thread_index % k_chunks;
        if constexpr (dequantize_pair != nullptr) {
            FOR_UNROLL (short step = k_chunk; step < weight_steps; step += k_chunks) {
                const int k_pos = loop_k + step * tile_k;
                const int shared_base = step * tile_rows * tile_k + row * tile_k;
                if (row_start + row < int(out_dim)) {
                    device const block_q *row_ptr =
                        (device const block_q *)(weights + weight_row_bytes * (row_start + row));
                    half4x4 low, high;
                    dequantize_pair(row_ptr + k_pos / (16 * nl), (k_pos / 16) % nl, low, high);
                    FOR_UNROLL (short i = 0; i < 16; ++i) {
                        shared[shared_base + i] = low[i / 4][i % 4];
                        shared[shared_base + 16 + i] = high[i / 4][i % 4];
                    }
                } else {
                    FOR_UNROLL (short i = 0; i < tile_k; ++i) {
                        shared[shared_base + i] = 0.h;
                    }
                }
            }
        } else {
            FOR_UNROLL (short step = 0; step < weight_steps; ++step) {
                const int k_pos = loop_k + step * tile_k + k_chunk * 16;
                const int shared_base = step * tile_rows * tile_k + row * tile_k + k_chunk * 16;
                if (row_start + row < int(out_dim)) {
                    const int block_index = k_pos / (16 * nl);
                    const short il = (k_pos / 16) % nl;
                    device const block_q *row_ptr =
                        (device const block_q *)(weights + weight_row_bytes * (row_start + row));
                    half4x4 values;
                    dequantize(row_ptr + block_index, il, values);
                    FOR_UNROLL (short i = 0; i < 16; ++i) {
                        shared[shared_base + i] = values[i / 4][i % 4];
                    }
                } else {
                    FOR_UNROLL (short i = 0; i < 16; ++i) {
                        shared[shared_base + i] = 0.h;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL (short step = 0; step < weight_steps; ++step) {
            auto input_slice = input_tensor.slice(loop_k + step * tile_k, token_start);
            auto step_tile = tensor(shared + step * tile_rows * tile_k,
                dextents<int32_t, 2>(tile_k, tile_rows));
            auto weight_slice = step_tile.slice(0, 0);
            matmul.run(input_slice, weight_slice, destination);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (relu_squared_epilogue != 0) {
        FOR_UNROLL (ushort i = 0; i < destination.get_capacity(); ++i) {
            if (destination.is_valid_element(i)) {
                const float value = max(destination[i], 0.0f);
                destination[i] = value * value;
            }
        }
    }

    auto output_tensor = tensor(
        output,
        dextents<int32_t, 2>(int(out_dim), int(tokens)),
        array<int, 2>({1, int(out_dim)}));
    destination.store(output_tensor.slice(row_start, token_start));
}


// Q8_0 prefill tile with half activations, following llama.cpp's
// kernel_mul_mm: weights and activations are both rounded to half and the
// matmul is strict, so products of the half operands accumulate in float.
// The host converts the activations to half once with dinf_cast_f32_f16.
// Relaxed MPP precision on float input is about as fast but has roughly 3x
// the error of half-rounded activations.
kernel void dinf_cast_f32_f16(
    device const float4 *src [[buffer(0)]],
    device half4 *dst [[buffer(1)]],
    constant uint &count4 [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count4) {
        dst[index] = half4(src[index]);
    }
}

template<
    typename block_q,
    short nl,
    short tile_tokens,
    void (*dequantize)(device const block_q *, short, thread half4x4 &)>
kernel void dinf_tensor_qmatmul_half_input(
    device const uchar *weights [[buffer(0)]],
    device half *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &k_dim [[buffer(3)]],
    constant uint &out_dim [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    constant uint &weight_row_bytes [[buffer(6)]],
    constant uint &relu_squared_epilogue [[buffer(7)]],
    threadgroup half *shared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort thread_index [[thread_index_in_threadgroup]]) {
    constexpr int tile_rows = 64;
    constexpr int tile_k = 32;
    constexpr int k_chunks = 2;

    const int row_start = group.y * tile_rows;
    const int token_start = group.x * tile_tokens;
    auto weight_tile = tensor(shared, dextents<int32_t, 2>(tile_k, tile_rows));
    auto input_tensor = tensor(
        input,
        dextents<int32_t, 2>(int(k_dim), int(tokens)),
        array<int, 2>({1, int(k_dim)}));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            tile_tokens,
            tile_rows,
            tile_k,
            false,
            true,
            false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> matmul;
    auto destination = matmul.template get_destination_cooperative_tensor<
        decltype(input_tensor),
        decltype(weight_tile),
        float>();

    const int row = thread_index / k_chunks;
    const int k_chunk = thread_index % k_chunks;
    device const block_q *row_ptr =
        (device const block_q *)(weights + weight_row_bytes * (row_start + row));
    for (int loop_k = 0; loop_k < int(k_dim); loop_k += tile_k) {
        const int k_pos = loop_k + k_chunk * 16;
        const int shared_base = row * tile_k + k_chunk * 16;
        if (row_start + row < int(out_dim)) {
            half4x4 values;
            dequantize(row_ptr + k_pos / (16 * nl), (k_pos / 16) % nl, values);
            FOR_UNROLL (short i = 0; i < 4; ++i) {
                *(threadgroup half4 *)(shared + shared_base + 4 * i) = values[i];
            }
        } else {
            FOR_UNROLL (short i = 0; i < 4; ++i) {
                *(threadgroup half4 *)(shared + shared_base + 4 * i) = half4(0.h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto input_slice = input_tensor.slice(loop_k, token_start);
        matmul.run(input_slice, weight_tile, destination);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (relu_squared_epilogue != 0) {
        FOR_UNROLL (ushort i = 0; i < destination.get_capacity(); ++i) {
            if (destination.is_valid_element(i)) {
                const float value = max(destination[i], 0.0f);
                destination[i] = value * value;
            }
        }
    }

    auto output_tensor = tensor(
        output,
        dextents<int32_t, 2>(int(out_dim), int(tokens)),
        array<int, 2>({1, int(out_dim)}));
    destination.store(output_tensor.slice(row_start, token_start));
}

// Exact-width speculative verification kernels. The split-K and
// multi-simdgroup layouts follow MTPLX/oMLX's skinny quantized-matmul work,
// adapted here for GGML Q4_K blocks and f32 activations.
template<
    typename block_q,
    short nl,
    void (*dequantize)(device const block_q *, short, thread half4x4 &),
    short m_rows,
    short k_parts>
kernel void dinf_tensor_qmatmul_verify_ksplit(
    device const uchar *weights [[buffer(0)]],
    device float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &k_dim [[buffer(3)]],
    constant uint &out_dim [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    constant uint &weight_row_bytes [[buffer(6)]],
    constant uint &relu_squared_epilogue [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr short rows_per_tile = 4;
    constexpr short chunk_width = 16;
    constexpr short accumulator_count = rows_per_tile * m_rows;

    const int row_start = int(group.x) * rows_per_tile;
    const int chunk_count = int(k_dim) / chunk_width;
    const int chunks_per_part = (chunk_count + int(k_parts) - 1) / int(k_parts);
    const int chunk_begin = int(simdgroup_index) * chunks_per_part;
    const int chunk_end = min(chunk_count, chunk_begin + chunks_per_part);

    float accumulators[accumulator_count];
    FOR_UNROLL (short index = 0; index < accumulator_count; ++index) {
        accumulators[index] = 0.0f;
    }

    for (int chunk = chunk_begin + int(lane); chunk < chunk_end; chunk += 32) {
        const int k_base = chunk * chunk_width;
        const int block_index = chunk / int(nl);
        const short il = short(chunk % int(nl));
        half4x4 quantized[rows_per_tile];

        FOR_UNROLL (short output_row = 0; output_row < rows_per_tile; ++output_row) {
            const int row = row_start + int(output_row);
            if (row < int(out_dim)) {
                device const block_q *row_ptr =
                    (device const block_q *)(weights + weight_row_bytes * row);
                dequantize(row_ptr + block_index, il, quantized[output_row]);
            } else {
                FOR_UNROLL (short k = 0; k < chunk_width; ++k) {
                    quantized[output_row][k / 4][k % 4] = 0.h;
                }
            }
        }

        FOR_UNROLL (short input_row = 0; input_row < m_rows; ++input_row) {
            FOR_UNROLL (short k = 0; k < chunk_width; ++k) {
                const float value = input[int(input_row) * int(k_dim) + k_base + int(k)];
                FOR_UNROLL (short output_row = 0; output_row < rows_per_tile; ++output_row) {
                    accumulators[output_row * m_rows + input_row] +=
                        value * float(quantized[output_row][k / 4][k % 4]);
                }
            }
        }
    }

    FOR_UNROLL (short index = 0; index < accumulator_count; ++index) {
        accumulators[index] = simd_sum(accumulators[index]);
    }

    threadgroup float partials[k_parts * accumulator_count];
    if (lane < accumulator_count) {
        partials[int(simdgroup_index) * accumulator_count + int(lane)] =
            accumulators[int(lane)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index == 0 && lane < accumulator_count) {
        float value = 0.0f;
        FOR_UNROLL (short part = 0; part < k_parts; ++part) {
            value += partials[part * accumulator_count + int(lane)];
        }
        const int output_row = int(lane) / int(m_rows);
        const int input_row = int(lane) - output_row * int(m_rows);
        if (row_start + output_row < int(out_dim) && input_row < int(tokens)) {
            if (relu_squared_epilogue != 0) {
                value = max(value, 0.0f);
                value *= value;
            }
            output[input_row * int(out_dim) + row_start + output_row] = value;
        }
    }
}

template<
    typename block_q,
    short nl,
    void (*dequantize)(device const block_q *, short, thread half4x4 &),
    short m_rows,
    short simdgroups>
kernel void dinf_tensor_qmatmul_verify_msg(
    device const uchar *weights [[buffer(0)]],
    device float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &k_dim [[buffer(3)]],
    constant uint &out_dim [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    constant uint &weight_row_bytes [[buffer(6)]],
    constant uint &relu_squared_epilogue [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr short rows_per_simdgroup = 4;
    constexpr short chunk_width = 16;
    constexpr short accumulator_count = rows_per_simdgroup * m_rows;

    const int row_start =
        (int(group.x) * int(simdgroups) + int(simdgroup_index)) * rows_per_simdgroup;
    if (row_start >= int(out_dim)) {
        return;
    }

    float accumulators[accumulator_count];
    FOR_UNROLL (short index = 0; index < accumulator_count; ++index) {
        accumulators[index] = 0.0f;
    }

    const int chunk_count = int(k_dim) / chunk_width;
    for (int chunk = int(lane); chunk < chunk_count; chunk += 32) {
        const int k_base = chunk * chunk_width;
        const int block_index = chunk / int(nl);
        const short il = short(chunk % int(nl));
        half4x4 quantized[rows_per_simdgroup];

        FOR_UNROLL (short output_row = 0; output_row < rows_per_simdgroup; ++output_row) {
            const int row = row_start + int(output_row);
            if (row < int(out_dim)) {
                device const block_q *row_ptr =
                    (device const block_q *)(weights + weight_row_bytes * row);
                dequantize(row_ptr + block_index, il, quantized[output_row]);
            } else {
                FOR_UNROLL (short k = 0; k < chunk_width; ++k) {
                    quantized[output_row][k / 4][k % 4] = 0.h;
                }
            }
        }

        FOR_UNROLL (short input_row = 0; input_row < m_rows; ++input_row) {
            FOR_UNROLL (short k = 0; k < chunk_width; ++k) {
                const float value = input[int(input_row) * int(k_dim) + k_base + int(k)];
                FOR_UNROLL (short output_row = 0; output_row < rows_per_simdgroup; ++output_row) {
                    accumulators[output_row * m_rows + input_row] +=
                        value * float(quantized[output_row][k / 4][k % 4]);
                }
            }
        }
    }

    FOR_UNROLL (short index = 0; index < accumulator_count; ++index) {
        accumulators[index] = simd_sum(accumulators[index]);
    }

    if (lane < accumulator_count) {
        const int output_row = int(lane) / int(m_rows);
        const int input_row = int(lane) - output_row * int(m_rows);
        if (row_start + output_row < int(out_dim) && input_row < int(tokens)) {
            float value = accumulators[int(lane)];
            if (relu_squared_epilogue != 0) {
                value = max(value, 0.0f);
                value *= value;
            }
            output[input_row * int(out_dim) + row_start + output_row] = value;
        }
    }
}

inline uint dinf_ksplit_output_index(
    uint token,
    uint row,
    uint out_dim,
    uint tokens,
    constant uint *group_starts,
    uint group_count) {
    if (group_count == 0) {
        return token * out_dim + row;
    }
    uint group = 0;
    FOR_UNROLL (short index = 1; index < 4; ++index) {
        group += uint(uint(index) < group_count && row >= group_starts[index]);
    }
    const uint start = group_starts[group];
    const uint width = group_starts[group + 1] - start;
    return start * tokens + token * width + row - start;
}

// Small verification batches do not expose enough output-row work to fill an
// M5-class GPU. Split K across eight simdgroups, then reduce their partials.
template<
    typename block_q,
    short nl,
    void (*dequantize)(device const block_q *, short, thread half4x4 &),
    short tile_tokens>
kernel void dinf_tensor_qmatmul_ksplit(
    device const uchar *weights [[buffer(0)]],
    device float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &k_dim [[buffer(3)]],
    constant uint &out_dim [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    constant uint &weight_row_bytes [[buffer(6)]],
    constant uint &relu_squared_epilogue [[buffer(7)]],
    constant uint *group_starts [[buffer(8)]],
    constant uint &group_count [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort thread_index [[thread_index_in_threadgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int tile_rows = 32;
    constexpr int tile_k = 16;
    constexpr int simdgroups = 8;
    threadgroup half weight_tiles[simdgroups][tile_k * tile_rows];
    threadgroup float partials[simdgroups][tile_tokens * tile_rows];

    const int row_start = int(group.x) * tile_rows;
    const int token_start = int(group.y) * tile_tokens;
    const int k_chunk = int(k_dim) / simdgroups;
    const int k_begin = int(simdgroup_index) * k_chunk;
    const int k_end = k_begin + k_chunk;
    auto input_tensor = tensor(
        input,
        dextents<int32_t, 2>(int(k_dim), int(tokens)),
        array<int, 2>({1, int(k_dim)}));
    auto weight_tile = tensor(
        weight_tiles[simdgroup_index],
        dextents<int32_t, 2>(tile_k, tile_rows));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            tile_tokens,
            tile_rows,
            tile_k,
            false,
            true,
            !is_same<block_q, block_q8_0>::value,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroup> matmul;
    auto destination = matmul.template get_destination_cooperative_tensor<
        decltype(input_tensor),
        decltype(weight_tile),
        float>();

    for (int loop_k = k_begin; loop_k < k_end; loop_k += tile_k) {
        const int row = int(lane);
        if (row_start + row < int(out_dim)) {
            const int block_index = loop_k / (tile_k * int(nl));
            const short il = short((loop_k / tile_k) % int(nl));
            device const block_q *row_ptr =
                (device const block_q *)(weights + weight_row_bytes * (row_start + row));
            half4x4 values;
            dequantize(row_ptr + block_index, il, values);
            FOR_UNROLL (short i = 0; i < tile_k; ++i) {
                weight_tiles[simdgroup_index][row * tile_k + i] =
                    values[i / 4][i % 4];
            }
        } else {
            FOR_UNROLL (short i = 0; i < tile_k; ++i) {
                weight_tiles[simdgroup_index][row * tile_k + i] = 0.h;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto input_slice = input_tensor.slice(loop_k, token_start);
        auto weight_slice = weight_tile.slice(0, 0);
        matmul.run(input_slice, weight_slice, destination);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto partial_tensor = tensor(
        partials[simdgroup_index],
        dextents<int32_t, 2>(tile_rows, tile_tokens),
        array<int, 2>({1, tile_rows}));
    destination.store(partial_tensor.slice(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int index = int(thread_index);
         index < tile_tokens * tile_rows;
         index += simdgroups * 32) {
        const int local_token = index / tile_rows;
        const int token = token_start + local_token;
        const int row = index - local_token * tile_rows;
        if (token < int(tokens) && row_start + row < int(out_dim)) {
            float value = 0.0f;
            FOR_UNROLL (short part = 0; part < simdgroups; ++part) {
                value += partials[part][index];
            }
            if (relu_squared_epilogue != 0) {
                value = max(value, 0.0f);
                value *= value;
            }
            const uint output_index = dinf_ksplit_output_index(
                uint(token), uint(row_start + row), out_dim, tokens, group_starts, group_count);
            output[output_index] = value;
        }
    }
}

template<
    typename block_q,
    short nl,
    void (*dequantize_pair)(
        device const block_q *,
        short,
        thread half4x4 &,
        thread half4x4 &),
    short tile_tokens,
    short split_parts,
    typename input_t = float>
kernel void dinf_tensor_qmatmul_ksplit_k32(
    device const uchar *weights [[buffer(0)]],
    device input_t *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &k_dim [[buffer(3)]],
    constant uint &out_dim [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    constant uint &weight_row_bytes [[buffer(6)]],
    constant uint &relu_squared_epilogue [[buffer(7)]],
    constant uint *group_starts [[buffer(8)]],
    constant uint &group_count [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort thread_index [[thread_index_in_threadgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int tile_rows = 32;
    constexpr int tile_k = 32;
    constexpr int simdgroups = split_parts;
    threadgroup half weight_tiles[simdgroups][tile_k * tile_rows];
    threadgroup float partials[simdgroups][tile_tokens * tile_rows];

    const int row_start = int(group.x) * tile_rows;
    const int token_start = int(group.y) * tile_tokens;
    const int k_chunk = int(k_dim) / simdgroups;
    const int k_begin = int(simdgroup_index) * k_chunk;
    const int k_end = k_begin + k_chunk;
    auto input_tensor = tensor(
        input,
        dextents<int32_t, 2>(int(k_dim), int(tokens)),
        array<int, 2>({1, int(k_dim)}));
    auto weight_tile = tensor(
        weight_tiles[simdgroup_index],
        dextents<int32_t, 2>(tile_k, tile_rows));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            tile_tokens,
            tile_rows,
            tile_k,
            false,
            true,
            !is_same<block_q, block_q8_0>::value && !is_same<input_t, half>::value,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroup> matmul;
    auto destination = matmul.template get_destination_cooperative_tensor<
        decltype(input_tensor),
        decltype(weight_tile),
        float>();

    for (int loop_k = k_begin; loop_k < k_end; loop_k += tile_k) {
        const int row = int(lane);
        if (row_start + row < int(out_dim)) {
            const int block_index = loop_k / (16 * int(nl));
            const short il = short((loop_k / 16) % int(nl));
            device const block_q *row_ptr =
                (device const block_q *)(weights + weight_row_bytes * (row_start + row));
            half4x4 low;
            half4x4 high;
            dequantize_pair(row_ptr + block_index, il, low, high);
            threadgroup half4 *tile_row =
                (threadgroup half4 *)(weight_tiles[simdgroup_index] + row * tile_k);
            FOR_UNROLL (short i = 0; i < 4; ++i) {
                tile_row[i] = low[i];
                tile_row[4 + i] = high[i];
            }
        } else {
            FOR_UNROLL (short i = 0; i < tile_k; ++i) {
                weight_tiles[simdgroup_index][row * tile_k + i] = 0.h;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto input_slice = input_tensor.slice(loop_k, token_start);
        auto weight_slice = weight_tile.slice(0, 0);
        matmul.run(input_slice, weight_slice, destination);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto partial_tensor = tensor(
        partials[simdgroup_index],
        dextents<int32_t, 2>(tile_rows, tile_tokens),
        array<int, 2>({1, tile_rows}));
    destination.store(partial_tensor.slice(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int index = int(thread_index);
         index < tile_tokens * tile_rows;
         index += simdgroups * 32) {
        const int local_token = index / tile_rows;
        const int token = token_start + local_token;
        const int row = index - local_token * tile_rows;
        if (token < int(tokens) && row_start + row < int(out_dim)) {
            float value = 0.0f;
            FOR_UNROLL (short part = 0; part < simdgroups; ++part) {
                value += partials[part][index];
            }
            if (relu_squared_epilogue != 0) {
                value = max(value, 0.0f);
                value *= value;
            }
            const uint output_index = dinf_ksplit_output_index(
                uint(token), uint(row_start + row), out_dim, tokens, group_starts, group_count);
            output[output_index] = value;
        }
    }
}

// Q4_K narrow-batch K-split with 64-value steps. The low and high nibbles of
// one 32-byte qs slice are adjacent 32-value sub-blocks, so each lane loads
// that slice once per step instead of once per nibble.
template<short tile_tokens, short split_parts, typename input_t>
kernel void dinf_tensor_qmatmul_q4k_ksplit_k64(
    device const uchar *weights [[buffer(0)]],
    device input_t *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &k_dim [[buffer(3)]],
    constant uint &out_dim [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    constant uint &weight_row_bytes [[buffer(6)]],
    constant uint &relu_squared_epilogue [[buffer(7)]],
    constant uint *group_starts [[buffer(8)]],
    constant uint &group_count [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort thread_index [[thread_index_in_threadgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int tile_rows = 32;
    constexpr int tile_k = 64;
    constexpr int simdgroups = split_parts;
    // Partials reuse the weight tiles after the K loop, which keeps
    // threadgroup memory low enough for more resident threadgroups.
    threadgroup half weight_tiles[simdgroups][tile_k * tile_rows];
    static_assert(simdgroups * tile_tokens * tile_rows * sizeof(float)
        <= sizeof(weight_tiles), "partials must fit in the weight tiles");
    threadgroup float (*partials)[tile_tokens * tile_rows] =
        (threadgroup float (*)[tile_tokens * tile_rows])&weight_tiles[0][0];

    const int row_start = int(group.x) * tile_rows;
    const int token_start = int(group.y) * tile_tokens;
    const int k_chunk = int(k_dim) / simdgroups;
    const int k_begin = int(simdgroup_index) * k_chunk;
    const int k_end = k_begin + k_chunk;
    auto input_tensor = tensor(
        input,
        dextents<int32_t, 2>(int(k_dim), int(tokens)),
        array<int, 2>({1, int(k_dim)}));
    auto weight_tile = tensor(
        weight_tiles[simdgroup_index],
        dextents<int32_t, 2>(tile_k, tile_rows));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            tile_tokens,
            tile_rows,
            tile_k,
            false,
            true,
            !is_same<input_t, half>::value,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroup> matmul;
    auto destination = matmul.template get_destination_cooperative_tensor<
        decltype(input_tensor),
        decltype(weight_tile),
        float>();

    for (int loop_k = k_begin; loop_k < k_end; loop_k += tile_k) {
        const int row = int(lane);
        const bool valid_row = row_start + row < int(out_dim);
        device const block_q4_K *row_ptr = (device const block_q4_K *)(
            weights + weight_row_bytes * uint(valid_row ? row_start + row : 0));
        threadgroup half4 *tile_row =
            (threadgroup half4 *)(weight_tiles[simdgroup_index] + row * tile_k);
        if (valid_row) {
            device const block_q4_K *block = row_ptr + loop_k / QK_K;
            const short pair = short((loop_k % QK_K) / 64);
            device const uint4 *q = (device const uint4 *)(block->qs + pair * 32);
            const uint4 packed[2] = {q[0], q[1]};
            const uchar2 sc_low = get_scale_min_k4(pair * 2, 0, block->scales);
            const uchar2 sc_high = get_scale_min_k4(pair * 2, 1, block->scales);
            const float d = block->d;
            const float d16 = block->d / 16.h;
            const float dl_low = d * sc_low[0];
            const float ml_low = float(block->dmin) * sc_low[1];
            const float dl_high = d16 * sc_high[0];
            const float ml_high = float(block->dmin) * sc_high[1];
            const uint4 shifts = uint4(0, 8, 16, 24);
            FOR_UNROLL (short half_index = 0; half_index < 2; ++half_index) {
                FOR_UNROLL (short word = 0; word < 4; ++word) {
                    const uint4 bytes = uint4(packed[half_index][word]) >> shifts;
                    tile_row[half_index * 4 + word] =
                        half4(dl_low * float4(bytes & 0x0F) - ml_low);
                    tile_row[8 + half_index * 4 + word] =
                        half4(dl_high * float4(bytes & 0xF0) - ml_high);
                }
            }
        } else {
            FOR_UNROLL (short i = 0; i < tile_k / 4; ++i) {
                tile_row[i] = half4(0.h);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto input_slice = input_tensor.slice(loop_k, token_start);
        auto weight_slice = weight_tile.slice(0, 0);
        matmul.run(input_slice, weight_slice, destination);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto partial_tensor = tensor(
        partials[simdgroup_index],
        dextents<int32_t, 2>(tile_rows, tile_tokens),
        array<int, 2>({1, tile_rows}));
    destination.store(partial_tensor.slice(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int index = int(thread_index);
         index < tile_tokens * tile_rows;
         index += simdgroups * 32) {
        const int local_token = index / tile_rows;
        const int token = token_start + local_token;
        const int out_row = index - local_token * tile_rows;
        if (token < int(tokens) && row_start + out_row < int(out_dim)) {
            float value = 0.0f;
            FOR_UNROLL (short part = 0; part < simdgroups; ++part) {
                value += partials[part][index];
            }
            if (relu_squared_epilogue != 0) {
                value = max(value, 0.0f);
                value *= value;
            }
            const uint output_index = dinf_ksplit_output_index(
                uint(token), uint(row_start + out_row), out_dim, tokens, group_starts, group_count);
            output[output_index] = value;
        }
    }
}

typedef decltype(dinf_tensor_qmatmul_q4k_ksplit_k64<8, 4, half>) q4k_ksplit_k64_t8_h_fn;
typedef decltype(dinf_tensor_qmatmul_q4k_ksplit_k64<16, 4, half>) q4k_ksplit_k64_t16_h_fn;
template [[host_name("dinf_tensor_qmatmul_q4k_ksplit_k64_t8_h")]] kernel q4k_ksplit_k64_t8_h_fn dinf_tensor_qmatmul_q4k_ksplit_k64<8, 4, half>;
template [[host_name("dinf_tensor_qmatmul_q4k_ksplit_k64_t16_h")]] kernel q4k_ksplit_k64_t16_h_fn dinf_tensor_qmatmul_q4k_ksplit_k64<16, 4, half>;

typedef decltype(dinf_tensor_qmatmul_ksplit<block_q4_K, 16, dequantize_q4_K, 16>) q4k_ksplit_t16_fn;
typedef decltype(dinf_tensor_qmatmul_ksplit_k32<block_q4_K, 16, dequantize_q4_K_pair, 16, 4>) q4k_ksplit_k32_t16_fn;
typedef decltype(dinf_tensor_qmatmul_ksplit<block_q8_0, 2, dequantize_q8_0, 8>) q8_0_ksplit_t8_fn;
typedef decltype(dinf_tensor_qmatmul_ksplit_k32<block_q8_0, 2, dequantize_q8_0_pair, 8, 8>) q8_0_ksplit_k32_t8_fn;
typedef decltype(dinf_tensor_qmatmul_ksplit<block_q6_K, 16, dequantize_q6_K, 8>) q6k_ksplit_t8_fn;
typedef decltype(dinf_tensor_qmatmul_verify_ksplit<block_q4_K, 16, dequantize_q4_K, 3, 2>) q4k_verify_ks_m3_fn;
typedef decltype(dinf_tensor_qmatmul_verify_ksplit<block_q4_K, 16, dequantize_q4_K, 4, 2>) q4k_verify_ks_m4_fn;
typedef decltype(dinf_tensor_qmatmul_verify_ksplit<block_q4_K, 16, dequantize_q4_K, 5, 2>) q4k_verify_ks_m5_fn;
typedef decltype(dinf_tensor_qmatmul_verify_ksplit<block_q4_K, 16, dequantize_q4_K, 6, 2>) q4k_verify_ks_m6_fn;
typedef decltype(dinf_tensor_qmatmul_verify_ksplit<block_q4_K, 16, dequantize_q4_K, 7, 2>) q4k_verify_ks_m7_fn;
typedef decltype(dinf_tensor_qmatmul_verify_ksplit<block_q4_K, 16, dequantize_q4_K, 8, 2>) q4k_verify_ks_m8_fn;
typedef decltype(dinf_tensor_qmatmul_verify_msg<block_q4_K, 16, dequantize_q4_K, 3, 8>) q4k_verify_msg_m3_fn;
typedef decltype(dinf_tensor_qmatmul_verify_msg<block_q4_K, 16, dequantize_q4_K, 4, 8>) q4k_verify_msg_m4_fn;
typedef decltype(dinf_tensor_qmatmul_verify_msg<block_q4_K, 16, dequantize_q4_K, 5, 8>) q4k_verify_msg_m5_fn;
typedef decltype(dinf_tensor_qmatmul_verify_msg<block_q4_K, 16, dequantize_q4_K, 6, 8>) q4k_verify_msg_m6_fn;
typedef decltype(dinf_tensor_qmatmul_verify_msg<block_q4_K, 16, dequantize_q4_K, 7, 8>) q4k_verify_msg_m7_fn;
typedef decltype(dinf_tensor_qmatmul_verify_msg<block_q4_K, 16, dequantize_q4_K, 8, 8>) q4k_verify_msg_m8_fn;
typedef decltype(dinf_tensor_qmatmul<block_q4_K, 16, 8, dequantize_q4_K>) q4k_t8_fn;
typedef decltype(dinf_tensor_qmatmul<block_q4_0, 2, 8, dequantize_q4_0>) q4_0_t8_fn;
typedef decltype(dinf_tensor_qmatmul<block_q5_1, 2, 8, dequantize_q5_1>) q5_1_t8_fn;
typedef decltype(dinf_tensor_qmatmul<block_q8_0, 2, 8, dequantize_q8_0>) q8_0_t8_fn;
typedef decltype(dinf_tensor_qmatmul<block_q5_K, 16, 8, dequantize_q5_K>) q5k_t8_fn;
typedef decltype(dinf_tensor_qmatmul<block_q6_K, 16, 8, dequantize_q6_K>) q6k_t8_fn;
typedef decltype(dinf_tensor_qmatmul<block_q4_K, 16, 16, dequantize_q4_K>) q4k_t16_fn;
typedef decltype(dinf_tensor_qmatmul<block_q4_0, 2, 16, dequantize_q4_0>) q4_0_t16_fn;
typedef decltype(dinf_tensor_qmatmul<block_q5_1, 2, 16, dequantize_q5_1>) q5_1_t16_fn;
typedef decltype(dinf_tensor_qmatmul<block_q8_0, 2, 16, dequantize_q8_0>) q8_0_t16_fn;
typedef decltype(dinf_tensor_qmatmul<block_q5_K, 16, 16, dequantize_q5_K>) q5k_t16_fn;
typedef decltype(dinf_tensor_qmatmul<block_q6_K, 16, 16, dequantize_q6_K>) q6k_t16_fn;
typedef decltype(dinf_tensor_qmatmul<block_q4_K, 16, 32, dequantize_q4_K>) q4k_t32_fn;
typedef decltype(dinf_tensor_qmatmul<block_q4_0, 2, 32, dequantize_q4_0>) q4_0_t32_fn;
typedef decltype(dinf_tensor_qmatmul<block_q5_1, 2, 32, dequantize_q5_1>) q5_1_t32_fn;
typedef decltype(dinf_tensor_qmatmul<block_q8_0, 2, 32, dequantize_q8_0>) q8_0_t32_fn;
typedef decltype(dinf_tensor_qmatmul<block_q5_K, 16, 32, dequantize_q5_K>) q5k_t32_fn;
typedef decltype(dinf_tensor_qmatmul<block_q6_K, 16, 32, dequantize_q6_K>) q6k_t32_fn;
typedef decltype(dinf_tensor_qmatmul<block_q4_K, 16, 64, dequantize_q4_K>) q4k_t64_fn;
typedef decltype(dinf_tensor_qmatmul<block_q4_0, 2, 64, dequantize_q4_0>) q4_0_t64_fn;
typedef decltype(dinf_tensor_qmatmul<block_q5_1, 2, 64, dequantize_q5_1>) q5_1_t64_fn;
typedef decltype(dinf_tensor_qmatmul<block_q8_0, 2, 64, dequantize_q8_0>) q8_0_t64_fn;
typedef decltype(dinf_tensor_qmatmul<block_q5_K, 16, 64, dequantize_q5_K>) q5k_t64_fn;
typedef decltype(dinf_tensor_qmatmul<block_q6_K, 16, 64, dequantize_q6_K>) q6k_t64_fn;
typedef decltype(dinf_tensor_qmatmul<block_q4_K, 16, 128, dequantize_q4_K>) q4k_t128_fn;
typedef decltype(dinf_tensor_qmatmul<block_q4_0, 2, 128, dequantize_q4_0>) q4_0_t128_fn;
typedef decltype(dinf_tensor_qmatmul<block_q5_1, 2, 128, dequantize_q5_1>) q5_1_t128_fn;
typedef decltype(dinf_tensor_qmatmul<block_q8_0, 2, 128, dequantize_q8_0>) q8_0_t128_fn;
typedef decltype(dinf_tensor_qmatmul<block_q5_K, 16, 128, dequantize_q5_K>) q5k_t128_fn;
typedef decltype(dinf_tensor_qmatmul<block_q6_K, 16, 128, dequantize_q6_K>) q6k_t128_fn;
typedef decltype(dinf_tensor_qmatmul<block_q4_K, 16, 64, dequantize_q4_K, 4, dequantize_q4_K_pair>) q4k_t64_w4_fn;

template [[host_name("dinf_tensor_qmatmul_q4k_t64_w4")]] kernel q4k_t64_w4_fn dinf_tensor_qmatmul<block_q4_K, 16, 64, dequantize_q4_K, 4, dequantize_q4_K_pair>;

template [[host_name("dinf_tensor_qmatmul_q4k_ksplit_t16")]] kernel q4k_ksplit_t16_fn dinf_tensor_qmatmul_ksplit<block_q4_K, 16, dequantize_q4_K, 16>;
template [[host_name("dinf_tensor_qmatmul_q4k_ksplit_k32_t16")]] kernel q4k_ksplit_k32_t16_fn dinf_tensor_qmatmul_ksplit_k32<block_q4_K, 16, dequantize_q4_K_pair, 16, 4>;
typedef decltype(dinf_tensor_qmatmul_ksplit_k32<block_q6_K, 16, dequantize_q6_K_pair, 8, 8, half>) q6k_ksplit_k32_t8_h_fn;
template [[host_name("dinf_tensor_qmatmul_q6k_ksplit_k32_t8_h")]] kernel q6k_ksplit_k32_t8_h_fn dinf_tensor_qmatmul_ksplit_k32<block_q6_K, 16, dequantize_q6_K_pair, 8, 8, half>;
template [[host_name("dinf_tensor_qmatmul_q8_0_ksplit_t8")]] kernel q8_0_ksplit_t8_fn dinf_tensor_qmatmul_ksplit<block_q8_0, 2, dequantize_q8_0, 8>;
template [[host_name("dinf_tensor_qmatmul_q8_0_ksplit_k32_t8")]] kernel q8_0_ksplit_k32_t8_fn dinf_tensor_qmatmul_ksplit_k32<block_q8_0, 2, dequantize_q8_0_pair, 8, 8>;
template [[host_name("dinf_tensor_qmatmul_q6k_ksplit_t8")]] kernel q6k_ksplit_t8_fn dinf_tensor_qmatmul_ksplit<block_q6_K, 16, dequantize_q6_K, 8>;
template [[host_name("dinf_tensor_qmatmul_q4k_verify_ks_m3")]] kernel q4k_verify_ks_m3_fn dinf_tensor_qmatmul_verify_ksplit<block_q4_K, 16, dequantize_q4_K, 3, 2>;
template [[host_name("dinf_tensor_qmatmul_q4k_verify_ks_m4")]] kernel q4k_verify_ks_m4_fn dinf_tensor_qmatmul_verify_ksplit<block_q4_K, 16, dequantize_q4_K, 4, 2>;
template [[host_name("dinf_tensor_qmatmul_q4k_verify_ks_m5")]] kernel q4k_verify_ks_m5_fn dinf_tensor_qmatmul_verify_ksplit<block_q4_K, 16, dequantize_q4_K, 5, 2>;
template [[host_name("dinf_tensor_qmatmul_q4k_verify_ks_m6")]] kernel q4k_verify_ks_m6_fn dinf_tensor_qmatmul_verify_ksplit<block_q4_K, 16, dequantize_q4_K, 6, 2>;
template [[host_name("dinf_tensor_qmatmul_q4k_verify_ks_m7")]] kernel q4k_verify_ks_m7_fn dinf_tensor_qmatmul_verify_ksplit<block_q4_K, 16, dequantize_q4_K, 7, 2>;
template [[host_name("dinf_tensor_qmatmul_q4k_verify_ks_m8")]] kernel q4k_verify_ks_m8_fn dinf_tensor_qmatmul_verify_ksplit<block_q4_K, 16, dequantize_q4_K, 8, 2>;
template [[host_name("dinf_tensor_qmatmul_q4k_verify_msg_m3")]] kernel q4k_verify_msg_m3_fn dinf_tensor_qmatmul_verify_msg<block_q4_K, 16, dequantize_q4_K, 3, 8>;
template [[host_name("dinf_tensor_qmatmul_q4k_verify_msg_m4")]] kernel q4k_verify_msg_m4_fn dinf_tensor_qmatmul_verify_msg<block_q4_K, 16, dequantize_q4_K, 4, 8>;
template [[host_name("dinf_tensor_qmatmul_q4k_verify_msg_m5")]] kernel q4k_verify_msg_m5_fn dinf_tensor_qmatmul_verify_msg<block_q4_K, 16, dequantize_q4_K, 5, 8>;
template [[host_name("dinf_tensor_qmatmul_q4k_verify_msg_m6")]] kernel q4k_verify_msg_m6_fn dinf_tensor_qmatmul_verify_msg<block_q4_K, 16, dequantize_q4_K, 6, 8>;
template [[host_name("dinf_tensor_qmatmul_q4k_verify_msg_m7")]] kernel q4k_verify_msg_m7_fn dinf_tensor_qmatmul_verify_msg<block_q4_K, 16, dequantize_q4_K, 7, 8>;
template [[host_name("dinf_tensor_qmatmul_q4k_verify_msg_m8")]] kernel q4k_verify_msg_m8_fn dinf_tensor_qmatmul_verify_msg<block_q4_K, 16, dequantize_q4_K, 8, 8>;
template [[host_name("dinf_tensor_qmatmul_q4k_t8")]] kernel q4k_t8_fn dinf_tensor_qmatmul<block_q4_K, 16, 8, dequantize_q4_K>;
template [[host_name("dinf_tensor_qmatmul_q4_0_t8")]] kernel q4_0_t8_fn dinf_tensor_qmatmul<block_q4_0, 2, 8, dequantize_q4_0>;
template [[host_name("dinf_tensor_qmatmul_q5_1_t8")]] kernel q5_1_t8_fn dinf_tensor_qmatmul<block_q5_1, 2, 8, dequantize_q5_1>;
template [[host_name("dinf_tensor_qmatmul_q8_0_t8")]] kernel q8_0_t8_fn dinf_tensor_qmatmul<block_q8_0, 2, 8, dequantize_q8_0>;
template [[host_name("dinf_tensor_qmatmul_q5k_t8")]] kernel q5k_t8_fn dinf_tensor_qmatmul<block_q5_K, 16, 8, dequantize_q5_K>;
template [[host_name("dinf_tensor_qmatmul_q6k_t8")]] kernel q6k_t8_fn dinf_tensor_qmatmul<block_q6_K, 16, 8, dequantize_q6_K>;
template [[host_name("dinf_tensor_qmatmul_q4k_t16")]] kernel q4k_t16_fn dinf_tensor_qmatmul<block_q4_K, 16, 16, dequantize_q4_K>;
template [[host_name("dinf_tensor_qmatmul_q4_0_t16")]] kernel q4_0_t16_fn dinf_tensor_qmatmul<block_q4_0, 2, 16, dequantize_q4_0>;
template [[host_name("dinf_tensor_qmatmul_q5_1_t16")]] kernel q5_1_t16_fn dinf_tensor_qmatmul<block_q5_1, 2, 16, dequantize_q5_1>;
template [[host_name("dinf_tensor_qmatmul_q8_0_t16")]] kernel q8_0_t16_fn dinf_tensor_qmatmul<block_q8_0, 2, 16, dequantize_q8_0>;
template [[host_name("dinf_tensor_qmatmul_q5k_t16")]] kernel q5k_t16_fn dinf_tensor_qmatmul<block_q5_K, 16, 16, dequantize_q5_K>;
template [[host_name("dinf_tensor_qmatmul_q6k_t16")]] kernel q6k_t16_fn dinf_tensor_qmatmul<block_q6_K, 16, 16, dequantize_q6_K>;
template [[host_name("dinf_tensor_qmatmul_q4k_t32")]] kernel q4k_t32_fn dinf_tensor_qmatmul<block_q4_K, 16, 32, dequantize_q4_K>;
template [[host_name("dinf_tensor_qmatmul_q4_0_t32")]] kernel q4_0_t32_fn dinf_tensor_qmatmul<block_q4_0, 2, 32, dequantize_q4_0>;
template [[host_name("dinf_tensor_qmatmul_q5_1_t32")]] kernel q5_1_t32_fn dinf_tensor_qmatmul<block_q5_1, 2, 32, dequantize_q5_1>;
template [[host_name("dinf_tensor_qmatmul_q8_0_t32")]] kernel q8_0_t32_fn dinf_tensor_qmatmul<block_q8_0, 2, 32, dequantize_q8_0>;
template [[host_name("dinf_tensor_qmatmul_q5k_t32")]] kernel q5k_t32_fn dinf_tensor_qmatmul<block_q5_K, 16, 32, dequantize_q5_K>;
template [[host_name("dinf_tensor_qmatmul_q6k_t32")]] kernel q6k_t32_fn dinf_tensor_qmatmul<block_q6_K, 16, 32, dequantize_q6_K>;
template [[host_name("dinf_tensor_qmatmul_q4k_t64")]] kernel q4k_t64_fn dinf_tensor_qmatmul<block_q4_K, 16, 64, dequantize_q4_K>;
template [[host_name("dinf_tensor_qmatmul_q4_0_t64")]] kernel q4_0_t64_fn dinf_tensor_qmatmul<block_q4_0, 2, 64, dequantize_q4_0>;
template [[host_name("dinf_tensor_qmatmul_q5_1_t64")]] kernel q5_1_t64_fn dinf_tensor_qmatmul<block_q5_1, 2, 64, dequantize_q5_1>;
template [[host_name("dinf_tensor_qmatmul_q8_0_t64")]] kernel q8_0_t64_fn dinf_tensor_qmatmul<block_q8_0, 2, 64, dequantize_q8_0>;
template [[host_name("dinf_tensor_qmatmul_q5k_t64")]] kernel q5k_t64_fn dinf_tensor_qmatmul<block_q5_K, 16, 64, dequantize_q5_K>;
template [[host_name("dinf_tensor_qmatmul_q6k_t64")]] kernel q6k_t64_fn dinf_tensor_qmatmul<block_q6_K, 16, 64, dequantize_q6_K>;
template [[host_name("dinf_tensor_qmatmul_q4k_t128")]] kernel q4k_t128_fn dinf_tensor_qmatmul<block_q4_K, 16, 128, dequantize_q4_K>;
template [[host_name("dinf_tensor_qmatmul_q4_0_t128")]] kernel q4_0_t128_fn dinf_tensor_qmatmul<block_q4_0, 2, 128, dequantize_q4_0>;
template [[host_name("dinf_tensor_qmatmul_q5_1_t128")]] kernel q5_1_t128_fn dinf_tensor_qmatmul<block_q5_1, 2, 128, dequantize_q5_1>;
template [[host_name("dinf_tensor_qmatmul_q8_0_t128")]] kernel q8_0_t128_fn dinf_tensor_qmatmul<block_q8_0, 2, 128, dequantize_q8_0>;
template [[host_name("dinf_tensor_qmatmul_q5k_t128")]] kernel q5k_t128_fn dinf_tensor_qmatmul<block_q5_K, 16, 128, dequantize_q5_K>;
template [[host_name("dinf_tensor_qmatmul_q6k_t128")]] kernel q6k_t128_fn dinf_tensor_qmatmul<block_q6_K, 16, 128, dequantize_q6_K>;
typedef decltype(dinf_tensor_qmatmul_half_input<block_q8_0, 2, 64, dequantize_q8_0>) q8_0_h64_fn;
typedef decltype(dinf_tensor_qmatmul_half_input<block_q8_0, 2, 128, dequantize_q8_0>) q8_0_h128_fn;
template [[host_name("dinf_tensor_qmatmul_q8_0_h64")]] kernel q8_0_h64_fn dinf_tensor_qmatmul_half_input<block_q8_0, 2, 64, dequantize_q8_0>;
template [[host_name("dinf_tensor_qmatmul_q8_0_h128")]] kernel q8_0_h128_fn dinf_tensor_qmatmul_half_input<block_q8_0, 2, 128, dequantize_q8_0>;
"#;

type PipelineCache = HashMap<(DeviceId, &'static str), Option<ComputePipeline>>;

static PIPELINES: OnceLock<Mutex<PipelineCache>> = OnceLock::new();
static TILE_OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
static Q4K_VERIFY_MIN_OUT: OnceLock<usize> = OnceLock::new();
static Q4K_VERIFY_MSG_MIN_OUT: OnceLock<usize> = OnceLock::new();
static Q4K_VERIFY_MAX_TOKENS: OnceLock<usize> = OnceLock::new();
static PRE_M5_SCALAR_Q6_R2_ENABLED: OnceLock<bool> = OnceLock::new();
const VERIFY_TILE_TOKENS: usize = 8;
const TINY_TILE_TOKENS: usize = 16;
const SMALL_TILE_TOKENS: usize = 32;
const MEDIUM_TILE_TOKENS: usize = 64;
const LARGE_TILE_TOKENS: usize = 128;
const MIN_TENSOR_TOKENS: usize = 8;
const MIN_Q4K_VERIFY_TOKENS: usize = 3;
const MAX_Q4K_VERIFY_TOKENS: usize = 8;
const DEFAULT_Q4K_VERIFY_MAX_TOKENS: usize = 6;
const MIN_Q4K_KSPLIT_TOKENS: usize = 7;
const DEFAULT_Q4K_VERIFY_MIN_OUT: usize = 4096;
const DEFAULT_Q4K_VERIFY_MSG_MIN_OUT: usize = 100_000;
const Q8_KSPLIT_MIN_OUT: usize = 512;
const Q6_KSPLIT_MIN_OUT: usize = 512;
const Q6_KSPLIT_MAX_OUT: usize = 8192;
const Q4K_VERIFY_K_PARTS: usize = 2;
const Q4K_VERIFY_MSG_SIMDGROUPS: usize = 8;

pub(super) fn supports_shared_verification() -> bool {
    // The optional eight-row F32 Q4 verifier has no multi-block kernel yet.
    // Keep that explicit numerical override on separate request checks.
    q4k_verify_max_tokens() < VERIFY_TILE_TOKENS
        || std::env::var_os("DINF_DISABLE_METAL4_Q4K_VERIFY").is_some()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MetalTensorProfile {
    Default,
    PreM5,
}

impl MetalTensorProfile {
    fn from_device_name(name: &str) -> Self {
        if matches!(apple_soc_generation(name), Some(1..=4)) {
            Self::PreM5
        } else {
            Self::Default
        }
    }

    fn uses_tensor_batch(self, tokens: usize) -> bool {
        // Pre-M5 Apple GPUs implement Metal 4 tensor operations without the
        // per-GPU-core Neural Accelerators introduced with M5. The tensor
        // kernels still win for narrow DFlash verification batches, but lose
        // decisively to Candle's legacy quantized GEMM for longer prompts.
        self != Self::PreM5 || tokens <= VERIFY_TILE_TOKENS
    }

    fn preferred_speculative_block_size(self, maximum: usize) -> usize {
        if self == Self::PreM5 {
            maximum.min(7)
        } else {
            maximum
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::PreM5 => "pre-m5-hybrid",
        }
    }
}

fn apple_soc_generation(name: &str) -> Option<u8> {
    let generation = name.strip_prefix("Apple M")?.split_whitespace().next()?;
    generation.parse().ok()
}

fn metal_tensor_profile(device: &Device) -> Result<MetalTensorProfile> {
    if !device.is_metal() {
        return Ok(MetalTensorProfile::Default);
    }
    let device = device.as_metal_device()?;
    Ok(MetalTensorProfile::from_device_name(
        &device.metal_device().as_ref().name().to_string(),
    ))
}

fn active_pre_m5_tensor_profile(device: &Device) -> Result<bool> {
    Ok(std::env::var_os("DINF_DISABLE_METAL4_QMM").is_none()
        && supports_metal_tensor(device)?
        && uses_pre_m5_decode_profile(device)?)
}

fn uses_pre_m5_decode_profile(device: &Device) -> Result<bool> {
    Ok(device.is_metal() && metal_tensor_profile(device)? == MetalTensorProfile::PreM5)
}

pub(super) fn preferred_speculative_block_size(device: &Device, maximum: usize) -> Result<usize> {
    if active_pre_m5_tensor_profile(device)? {
        Ok(MetalTensorProfile::PreM5.preferred_speculative_block_size(maximum))
    } else {
        Ok(maximum)
    }
}

pub(super) fn uses_pre_m5_speculative_policy(device: &Device) -> Result<bool> {
    uses_pre_m5_decode_profile(device)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn q4k_verify_min_out() -> usize {
    *Q4K_VERIFY_MIN_OUT
        .get_or_init(|| env_usize("DINF_METAL4_Q4K_VERIFY_MIN_OUT", DEFAULT_Q4K_VERIFY_MIN_OUT))
}

fn pre_m5_scalar_q6_r2_enabled() -> bool {
    *PRE_M5_SCALAR_Q6_R2_ENABLED
        .get_or_init(|| std::env::var_os("DINF_DISABLE_METAL_PRE_M5_Q6_R2").is_none())
}

fn q4k_verify_msg_min_out() -> usize {
    *Q4K_VERIFY_MSG_MIN_OUT.get_or_init(|| {
        env_usize(
            "DINF_METAL4_Q4K_VERIFY_MSG_MIN_OUT",
            DEFAULT_Q4K_VERIFY_MSG_MIN_OUT,
        )
    })
}

fn q4k_verify_max_tokens() -> usize {
    (*Q4K_VERIFY_MAX_TOKENS.get_or_init(|| {
        env_usize(
            "DINF_METAL4_Q4K_VERIFY_MAX_TOKENS",
            DEFAULT_Q4K_VERIFY_MAX_TOKENS,
        )
    }))
    .min(MAX_Q4K_VERIFY_TOKENS)
}

fn q4k_verify_kernel_name(tokens: usize, msg: bool) -> Option<&'static str> {
    match (tokens, msg) {
        (3, false) => Some("dinf_tensor_qmatmul_q4k_verify_ks_m3"),
        (4, false) => Some("dinf_tensor_qmatmul_q4k_verify_ks_m4"),
        (5, false) => Some("dinf_tensor_qmatmul_q4k_verify_ks_m5"),
        (6, false) => Some("dinf_tensor_qmatmul_q4k_verify_ks_m6"),
        (7, false) => Some("dinf_tensor_qmatmul_q4k_verify_ks_m7"),
        (8, false) => Some("dinf_tensor_qmatmul_q4k_verify_ks_m8"),
        (3, true) => Some("dinf_tensor_qmatmul_q4k_verify_msg_m3"),
        (4, true) => Some("dinf_tensor_qmatmul_q4k_verify_msg_m4"),
        (5, true) => Some("dinf_tensor_qmatmul_q4k_verify_msg_m5"),
        (6, true) => Some("dinf_tensor_qmatmul_q4k_verify_msg_m6"),
        (7, true) => Some("dinf_tensor_qmatmul_q4k_verify_msg_m7"),
        (8, true) => Some("dinf_tensor_qmatmul_q4k_verify_msg_m8"),
        _ => None,
    }
}

fn tile_tokens(tokens: usize) -> usize {
    if let Some(tile_tokens) = *TILE_OVERRIDE.get_or_init(|| {
        let value = std::env::var_os("DINF_METAL4_QMM_TILE")?;
        let parsed = value.to_string_lossy().parse::<usize>().ok();
        if parsed.is_some_and(|tile| {
            matches!(
                tile,
                VERIFY_TILE_TOKENS
                    | TINY_TILE_TOKENS
                    | SMALL_TILE_TOKENS
                    | MEDIUM_TILE_TOKENS
                    | LARGE_TILE_TOKENS
            )
        }) {
            parsed
        } else {
            tracing::warn!(
                value = %value.to_string_lossy(),
                "ignoring unsupported DINF_METAL4_QMM_TILE; expected 8, 16, 32, 64, or 128"
            );
            None
        }
    }) {
        return tile_tokens;
    }
    match tokens {
        ..=VERIFY_TILE_TOKENS => VERIFY_TILE_TOKENS,
        9..=TINY_TILE_TOKENS => TINY_TILE_TOKENS,
        17..=SMALL_TILE_TOKENS => SMALL_TILE_TOKENS,
        33..=MEDIUM_TILE_TOKENS => MEDIUM_TILE_TOKENS,
        _ => LARGE_TILE_TOKENS,
    }
}

fn kernel_name(dtype: GgmlDType, tile_tokens: usize) -> Result<&'static str> {
    match (dtype, tile_tokens) {
        (GgmlDType::Q4_0, VERIFY_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q4_0_t8"),
        (GgmlDType::Q5_1, VERIFY_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q5_1_t8"),
        (GgmlDType::Q8_0, VERIFY_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q8_0_t8"),
        (GgmlDType::Q4K, VERIFY_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q4k_t8"),
        (GgmlDType::Q5K, VERIFY_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q5k_t8"),
        (GgmlDType::Q6K, VERIFY_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q6k_t8"),
        (GgmlDType::Q4_0, TINY_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q4_0_t16"),
        (GgmlDType::Q5_1, TINY_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q5_1_t16"),
        (GgmlDType::Q8_0, TINY_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q8_0_t16"),
        (GgmlDType::Q4K, TINY_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q4k_t16"),
        (GgmlDType::Q5K, TINY_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q5k_t16"),
        (GgmlDType::Q6K, TINY_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q6k_t16"),
        (GgmlDType::Q4_0, SMALL_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q4_0_t32"),
        (GgmlDType::Q5_1, SMALL_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q5_1_t32"),
        (GgmlDType::Q8_0, SMALL_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q8_0_t32"),
        (GgmlDType::Q4K, SMALL_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q4k_t32"),
        (GgmlDType::Q5K, SMALL_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q5k_t32"),
        (GgmlDType::Q6K, SMALL_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q6k_t32"),
        (GgmlDType::Q4_0, MEDIUM_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q4_0_t64"),
        (GgmlDType::Q5_1, MEDIUM_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q5_1_t64"),
        (GgmlDType::Q8_0, MEDIUM_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q8_0_t64"),
        (GgmlDType::Q4K, MEDIUM_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q4k_t64"),
        (GgmlDType::Q5K, MEDIUM_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q5k_t64"),
        (GgmlDType::Q6K, MEDIUM_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q6k_t64"),
        (GgmlDType::Q4_0, LARGE_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q4_0_t128"),
        (GgmlDType::Q5_1, LARGE_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q5_1_t128"),
        (GgmlDType::Q8_0, LARGE_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q8_0_t128"),
        (GgmlDType::Q4K, LARGE_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q4k_t128"),
        (GgmlDType::Q5K, LARGE_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q5k_t128"),
        (GgmlDType::Q6K, LARGE_TILE_TOKENS) => Ok("dinf_tensor_qmatmul_q6k_t128"),
        (_, tile_tokens)
            if !matches!(
                tile_tokens,
                VERIFY_TILE_TOKENS
                    | TINY_TILE_TOKENS
                    | SMALL_TILE_TOKENS
                    | MEDIUM_TILE_TOKENS
                    | LARGE_TILE_TOKENS
            ) =>
        {
            candle_core::bail!("unsupported Metal tensor token tile {tile_tokens}")
        }
        (other, _) => {
            candle_core::bail!("Metal tensor quantized matmul does not support {other:?}")
        }
    }
}

fn pipeline(
    device: &candle_core::MetalDevice,
    dtype: GgmlDType,
    tile_tokens: usize,
) -> Result<Option<ComputePipeline>> {
    let name = kernel_name(dtype, tile_tokens)?;
    named_pipeline(device, name)
}

fn named_pipeline(
    device: &candle_core::MetalDevice,
    name: &'static str,
) -> Result<Option<ComputePipeline>> {
    let key = (device.id(), name);
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(pipeline) = pipelines.lock().unwrap().get(&key) {
        return Ok(pipeline.clone());
    }
    let compiled = (|| -> Result<ComputePipeline> {
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version4_0);
        let library = device
            .metal_device()
            .new_library_with_source(TENSOR_QMATMUL_MSL, Some(&options))
            .map_err(candle_core::Error::wrap)?;
        let function = library
            .get_function(name, None)
            .map_err(candle_core::Error::wrap)?;
        device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(candle_core::Error::wrap)
    })();
    match compiled {
        Ok(pipeline) => {
            pipelines
                .lock()
                .unwrap()
                .insert(key, Some(pipeline.clone()));
            Ok(Some(pipeline))
        }
        Err(error) => {
            tracing::warn!(
                kernel = name,
                %error,
                "Metal 4 tensor matmul unavailable; using Candle"
            );
            pipelines.lock().unwrap().insert(key, None);
            Ok(None)
        }
    }
}

fn supports_metal_tensor(device: &Device) -> Result<bool> {
    if !device.is_metal() {
        return Ok(false);
    }
    let device = device.as_metal_device()?;
    Ok(device
        .metal_device()
        .as_ref()
        .supportsFamily(MTLGPUFamily::Metal4))
}

pub(crate) fn supports_metal_tensor_qmatmul(device: &Device) -> Result<bool> {
    if std::env::var_os("DINF_DISABLE_METAL4_QMM").is_some() {
        return Ok(false);
    }
    supports_metal_tensor(device)
}

#[derive(Clone)]
pub(crate) struct MetalTensorQMatMul {
    weights: Tensor,
    dtype: GgmlDType,
    out_dim: usize,
    in_dim: usize,
    output_groups: Option<Arc<[usize]>>,
    profile: MetalTensorProfile,
    prefill_weight_reuse: bool,
}

impl MetalTensorQMatMul {
    /// Decode independent rows with the scalar F32 reduction in one GPU
    /// dispatch. Small generic GEMMs round through half and waste tile work.
    pub(super) fn forward_decode_rows(&self, input: &Tensor) -> Result<Tensor> {
        // Pre-M5's Q6 single-row override uses a different reduction. Preserve
        // it per row instead of silently switching math when requests join.
        if self.profile == MetalTensorProfile::PreM5
            && self.dtype == GgmlDType::Q6K
            && pre_m5_scalar_q6_r2_enabled()
            && matches!(input.rank(), 2 | 3)
            && (input.rank() == 2 || input.dim(0)? == 1)
            && input.dim(D::Minus2)? > 1
        {
            let outputs = (0..input.dim(D::Minus2)?)
                .map(|row| self.forward(&input.narrow(D::Minus2, row, 1)?))
                .collect::<Result<Vec<_>>>()?;
            return Tensor::cat(&outputs, D::Minus2);
        }
        self.weights.apply_op2_no_bwd(
            &input.contiguous()?,
            &TensorQMatMul {
                dtype: self.dtype,
                out_dim: self.out_dim,
                in_dim: self.in_dim,
                preferred_tile_tokens: None,
                relu_squared_epilogue: false,
                output_groups: None,
                profile: self.profile,
                prefill_weight_reuse: self.prefill_weight_reuse,
                decode_rows: true,
            },
        )
    }

    pub(super) fn try_from_qtensor(packed: Arc<QTensor>, device: &Device) -> Result<Option<Self>> {
        let profile = metal_tensor_profile(device)?;
        let validation_tile = if profile == MetalTensorProfile::PreM5 {
            VERIFY_TILE_TOKENS
        } else {
            LARGE_TILE_TOKENS
        };
        if !supports_metal_tensor(device)? || kernel_name(packed.dtype(), validation_tile).is_err()
        {
            return Ok(None);
        }
        if pipeline(device.as_metal_device()?, packed.dtype(), validation_tile)?.is_none() {
            return Ok(None);
        }
        if packed.dtype() == GgmlDType::Q4K
            && std::env::var_os("DINF_DISABLE_METAL4_Q4K_KSPLIT").is_none()
        {
            if std::env::var_os("DINF_DISABLE_METAL4_Q4K_K32_KSPLIT").is_none() {
                let _ = named_pipeline(
                    device.as_metal_device()?,
                    "dinf_tensor_qmatmul_q4k_ksplit_k32_t16",
                )?;
            }
            let _ = named_pipeline(
                device.as_metal_device()?,
                "dinf_tensor_qmatmul_q4k_ksplit_t16",
            )?;
        }
        if packed.dtype() == GgmlDType::Q8_0
            && std::env::var_os("DINF_DISABLE_METAL4_Q8_KSPLIT").is_none()
            && std::env::var_os("DINF_DISABLE_METAL4_Q8_K32_KSPLIT").is_none()
        {
            let _ = named_pipeline(
                device.as_metal_device()?,
                "dinf_tensor_qmatmul_q8_0_ksplit_k32_t8",
            )?;
        }
        if profile == MetalTensorProfile::PreM5
            && packed.dtype() == GgmlDType::Q6K
            && pre_m5_scalar_q6_r2_enabled()
        {
            let _ = named_pipeline(device.as_metal_device()?, "dinf_scalar_q6k_r2_sg2")?;
        }
        let (out_dim, in_dim) = packed.shape().dims2()?;
        if profile == MetalTensorProfile::PreM5
            && packed.dtype() == GgmlDType::Q8_0
            && in_dim == 4096
            && matches!(out_dim, 2560 | 4096)
            && std::env::var_os("DINF_DISABLE_FUSED_ADD_RMSNORM").is_none()
            && std::env::var_os("DINF_DISABLE_FUSED_RMSNORM").is_none()
            && std::env::var_os("DINF_DISABLE_QWEN35_FUSED_Q8_SSM_RESIDUAL").is_none()
        {
            let _ = named_pipeline(
                device.as_metal_device()?,
                "dinf_scalar_q8_0_residual_r4_sg2",
            )?;
        }
        if !in_dim.is_multiple_of(packed.dtype().block_size()) {
            return Ok(None);
        }
        let raw = packed.data()?.into_owned();
        let expected = out_dim * in_dim / packed.dtype().block_size() * packed.dtype().type_size();
        if raw.len() != expected {
            candle_core::bail!(
                "packed tensor storage has {} bytes, expected {expected}",
                raw.len()
            )
        }
        let raw_len = raw.len();
        let weights = Tensor::from_vec(raw, raw_len, device)?;
        let dtype = packed.dtype();
        Ok(Some(Self {
            weights,
            dtype,
            out_dim,
            in_dim,
            output_groups: None,
            profile,
            prefill_weight_reuse: std::env::var_os("DINF_DISABLE_METAL4_PREFILL_WEIGHT_REUSE")
                .is_none(),
        }))
    }

    /// Wrap packed GGML rows that are already resident on `device` (for
    /// example a native kernel's copy) without copying them.
    pub(super) fn try_from_raw(
        weights: &Tensor,
        dtype: GgmlDType,
        out_dim: usize,
        in_dim: usize,
        device: &Device,
    ) -> Result<Option<Self>> {
        let profile = metal_tensor_profile(device)?;
        if !supports_metal_tensor(device)?
            || kernel_name(dtype, LARGE_TILE_TOKENS).is_err()
            || pipeline(device.as_metal_device()?, dtype, LARGE_TILE_TOKENS)?.is_none()
            || !in_dim.is_multiple_of(dtype.block_size())
            || weights.dtype() != DType::U8
            || !weights.is_contiguous()
            || !weights.device().same_device(device)
            || weights.elem_count() != out_dim * in_dim / dtype.block_size() * dtype.type_size()
        {
            return Ok(None);
        }
        Ok(Some(Self {
            weights: weights.clone(),
            dtype,
            out_dim,
            in_dim,
            output_groups: None,
            profile,
            prefill_weight_reuse: std::env::var_os("DINF_DISABLE_METAL4_PREFILL_WEIGHT_REUSE")
                .is_none(),
        }))
    }

    pub(super) fn try_from_qtensors(
        packed: &[Arc<QTensor>],
        device: &Device,
    ) -> Result<Option<Self>> {
        let Some(first) = packed.first() else {
            return Ok(None);
        };
        let profile = metal_tensor_profile(device)?;
        let validation_tile = if profile == MetalTensorProfile::PreM5 {
            VERIFY_TILE_TOKENS
        } else {
            LARGE_TILE_TOKENS
        };
        let dtype = first.dtype();
        if packed.len() < 2
            || !supports_metal_tensor(device)?
            || kernel_name(dtype, validation_tile).is_err()
            || packed.iter().any(|tensor| tensor.dtype() != dtype)
        {
            return Ok(None);
        }
        if pipeline(device.as_metal_device()?, dtype, validation_tile)?.is_none() {
            return Ok(None);
        }
        if dtype == GgmlDType::Q4K && std::env::var_os("DINF_DISABLE_METAL4_Q4K_KSPLIT").is_none() {
            if std::env::var_os("DINF_DISABLE_METAL4_Q4K_K32_KSPLIT").is_none() {
                let _ = named_pipeline(
                    device.as_metal_device()?,
                    "dinf_tensor_qmatmul_q4k_ksplit_k32_t16",
                )?;
            }
            let _ = named_pipeline(
                device.as_metal_device()?,
                "dinf_tensor_qmatmul_q4k_ksplit_t16",
            )?;
        }
        if dtype == GgmlDType::Q8_0
            && std::env::var_os("DINF_DISABLE_METAL4_Q8_KSPLIT").is_none()
            && std::env::var_os("DINF_DISABLE_METAL4_Q8_K32_KSPLIT").is_none()
        {
            let _ = named_pipeline(
                device.as_metal_device()?,
                "dinf_tensor_qmatmul_q8_0_ksplit_k32_t8",
            )?;
        }
        if profile == MetalTensorProfile::PreM5
            && dtype == GgmlDType::Q6K
            && pre_m5_scalar_q6_r2_enabled()
        {
            let _ = named_pipeline(device.as_metal_device()?, "dinf_scalar_q6k_r2_sg2")?;
        }

        let mut in_dim = None;
        let mut out_dim = 0usize;
        let mut expected_bytes = 0usize;
        let mut output_groups = Vec::with_capacity(packed.len());
        for tensor in packed {
            let (tensor_out, tensor_in) = tensor.shape().dims2()?;
            if in_dim.is_some_and(|width| width != tensor_in)
                || !tensor_in.is_multiple_of(dtype.block_size())
            {
                return Ok(None);
            }
            in_dim = Some(tensor_in);
            out_dim = out_dim
                .checked_add(tensor_out)
                .ok_or_else(|| candle_core::Error::Msg("grouped output width overflow".into()))?;
            output_groups.push(tensor_out);
            expected_bytes = expected_bytes
                .checked_add(tensor_out * tensor_in / dtype.block_size() * dtype.type_size())
                .ok_or_else(|| candle_core::Error::Msg("grouped weight size overflow".into()))?;
        }
        let in_dim = in_dim.unwrap();
        let mut raw = Vec::with_capacity(expected_bytes);
        for tensor in packed {
            let bytes = tensor.data()?;
            raw.extend_from_slice(bytes.as_ref());
        }
        if raw.len() != expected_bytes {
            candle_core::bail!(
                "grouped packed tensor storage has {} bytes, expected {expected_bytes}",
                raw.len()
            )
        }
        let raw_len = raw.len();
        let weights = Tensor::from_vec(raw, raw_len, device)?;
        Ok(Some(Self {
            weights,
            dtype,
            out_dim,
            in_dim,
            output_groups: Some(output_groups.into()),
            profile,
            prefill_weight_reuse: std::env::var_os("DINF_DISABLE_METAL4_PREFILL_WEIGHT_REUSE")
                .is_none(),
        }))
    }

    pub(super) fn profile_label(&self) -> &'static str {
        self.profile.label()
    }

    pub(super) fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_with_tile(input, None)
    }

    #[cfg(test)]
    pub(super) fn forward_groups(&self, input: &Tensor) -> Result<Option<Vec<Tensor>>> {
        let Some(groups) = self.output_groups.as_deref() else {
            return Ok(None);
        };
        if self.dtype != GgmlDType::Q8_0
            || input.rank() != 3
            || input.dims3()?.0 != 1
            || input.dims3()?.1 != VERIFY_TILE_TOKENS
            || !(2..=4).contains(&groups.len())
            || groups.iter().sum::<usize>() != self.out_dim
            || std::env::var_os("DINF_DISABLE_METAL4_Q8_KSPLIT").is_some()
            || std::env::var_os("DINF_DISABLE_QWEN35_GROUPED_OUTPUT").is_some()
        {
            return Ok(None);
        }
        let output = self.weights.apply_op2_no_bwd(
            &input.contiguous()?,
            &TensorQMatMul {
                dtype: self.dtype,
                out_dim: self.out_dim,
                in_dim: self.in_dim,
                preferred_tile_tokens: None,
                relu_squared_epilogue: false,
                output_groups: self.output_groups.clone(),
                profile: self.profile,
                prefill_weight_reuse: self.prefill_weight_reuse,
                decode_rows: false,
            },
        )?;
        let output = output.flatten_all()?;
        let mut offset = 0usize;
        let mut tensors = Vec::with_capacity(groups.len());
        for &width in groups {
            let elements = VERIFY_TILE_TOKENS * width;
            tensors.push(output.narrow(0, offset, elements)?.reshape((
                1,
                VERIFY_TILE_TOKENS,
                width,
            ))?);
            offset += elements;
        }
        Ok(Some(tensors))
    }

    /// Run one packed homogeneous projection for a single decode token and
    /// return zero-copy views for its logical output groups.
    pub(super) fn forward_single_token_groups(
        &self,
        input: &Tensor,
    ) -> Result<Option<Vec<Tensor>>> {
        let Some(groups) = self.output_groups.as_deref() else {
            return Ok(None);
        };
        let input_dims = input.dims();
        let tokens = input_dims
            .get(input_dims.len().saturating_sub(2))
            .copied()
            .unwrap_or(0);
        let batch = input_dims[..input_dims.len().saturating_sub(2)]
            .iter()
            .product::<usize>();
        if tokens != 1
            || batch != 1
            || !matches!(input.rank(), 2 | 3)
            || groups.iter().sum::<usize>() != self.out_dim
        {
            return Ok(None);
        }

        let output = self.forward(input)?.flatten_all()?;
        let mut offset = 0usize;
        let mut outputs = Vec::with_capacity(groups.len());
        for &width in groups {
            let mut shape = input_dims.to_vec();
            *shape.last_mut().unwrap() = width;
            outputs.push(
                output
                    .narrow(0, offset, width)?
                    .reshape(Shape::from(shape))?,
            );
            offset += width;
        }
        Ok(Some(outputs))
    }

    /// Fuse equal-width packed Q6_K gate/up projections with SwiGLU for a
    /// single decode token on pre-M5 Apple GPUs.
    pub(super) fn forward_swiglu(&self, input: &Tensor) -> Result<Option<Tensor>> {
        let Some(groups) = self.output_groups.as_deref() else {
            return Ok(None);
        };
        let dims = input.dims();
        let tokens = dims.get(dims.len().saturating_sub(2)).copied().unwrap_or(0);
        let batch = dims[..dims.len().saturating_sub(2)]
            .iter()
            .product::<usize>();
        if self.profile != MetalTensorProfile::PreM5
            || self.dtype != GgmlDType::Q6K
            || groups.len() != 2
            || groups[0] != groups[1]
            || groups[0] * 2 != self.out_dim
            || tokens != 1
            || batch != 1
            || !matches!(input.rank(), 2 | 3)
            || !input.device().is_metal()
            || input.dtype() != DType::F32
            || !pre_m5_scalar_q6_r2_enabled()
            || std::env::var_os("DINF_DISABLE_FUSED_SILU_MUL").is_some()
            || std::env::var_os("DINF_DISABLE_QWEN35_FUSED_Q6K_SWIGLU").is_some()
        {
            return Ok(None);
        }
        if named_pipeline(
            input.device().as_metal_device()?,
            "dinf_scalar_q6k_swiglu_r2_sg2",
        )?
        .is_none()
        {
            return Ok(None);
        }
        let output = self.weights.apply_op2_no_bwd(
            &input.contiguous()?,
            &FusedQ6KSwiGlu {
                in_dim: self.in_dim,
                intermediate_dim: groups[0],
            },
        )?;
        Ok(Some(output))
    }

    /// Fuse a single-token Q6_K projection with the following residual add on
    /// pre-M5 Apple GPUs. Unsupported shapes retain the ordinary matmul/add
    /// path so this optimization never changes model compatibility.
    pub(super) fn forward_residual(
        &self,
        input: &Tensor,
        residual: &Tensor,
    ) -> Result<Option<Tensor>> {
        let input_dims = input.dims();
        let residual_dims = residual.dims();
        let tokens = input_dims
            .get(input_dims.len().saturating_sub(2))
            .copied()
            .unwrap_or(0);
        let batch = input_dims[..input_dims.len().saturating_sub(2)]
            .iter()
            .product::<usize>();
        if self.profile != MetalTensorProfile::PreM5
            || self.dtype != GgmlDType::Q6K
            || self.output_groups.is_some()
            || tokens != 1
            || batch != 1
            || !matches!(input.rank(), 2 | 3)
            || residual.rank() != input.rank()
            || input_dims.last().copied() != Some(self.in_dim)
            || residual_dims.last().copied() != Some(self.out_dim)
            || input_dims[..input_dims.len().saturating_sub(1)]
                != residual_dims[..residual_dims.len().saturating_sub(1)]
            || !input.device().is_metal()
            || !residual.device().is_metal()
            || input.dtype() != DType::F32
            || residual.dtype() != DType::F32
            || !pre_m5_scalar_q6_r2_enabled()
            || std::env::var_os("DINF_DISABLE_QWEN35_FUSED_Q6K_FFN_RESIDUAL").is_some()
        {
            return Ok(None);
        }
        if named_pipeline(
            input.device().as_metal_device()?,
            "dinf_scalar_q6k_residual_r2_sg2",
        )?
        .is_none()
        {
            return Ok(None);
        }
        let input = input.contiguous()?;
        let residual = residual.contiguous()?;
        let output = self.weights.apply_op3_no_bwd(
            &input,
            &residual,
            &FusedQ6KResidual {
                in_dim: self.in_dim,
                out_dim: self.out_dim,
            },
        )?;
        Ok(Some(output))
    }

    /// Fuse the Q8_0 DeltaNet output projection with its residual add during
    /// single-token decode. The following RMSNorm remains a separate kernel;
    /// this path is retained only when the combined end-to-end sequence wins.
    pub(super) fn forward_qwen35_ssm_residual(
        &self,
        input: &Tensor,
        residual: &Tensor,
    ) -> Result<Option<Tensor>> {
        let input_dims = input.dims();
        let residual_dims = residual.dims();
        let tokens = input_dims
            .get(input_dims.len().saturating_sub(2))
            .copied()
            .unwrap_or(0);
        let batch = input_dims[..input_dims.len().saturating_sub(2)]
            .iter()
            .product::<usize>();
        if self.profile != MetalTensorProfile::PreM5
            || self.dtype != GgmlDType::Q8_0
            || self.output_groups.is_some()
            || self.in_dim != 4096
            || !matches!(self.out_dim, 2560 | 4096)
            || tokens != 1
            || batch != 1
            || !matches!(input.rank(), 2 | 3)
            || residual.rank() != input.rank()
            || input_dims.last().copied() != Some(self.in_dim)
            || residual_dims.last().copied() != Some(self.out_dim)
            || input_dims[..input_dims.len().saturating_sub(1)]
                != residual_dims[..residual_dims.len().saturating_sub(1)]
            || !input.device().is_metal()
            || !residual.device().is_metal()
            || input.dtype() != DType::F32
            || residual.dtype() != DType::F32
            || std::env::var_os("DINF_DISABLE_FUSED_ADD_RMSNORM").is_some()
            || std::env::var_os("DINF_DISABLE_FUSED_RMSNORM").is_some()
            || std::env::var_os("DINF_DISABLE_QWEN35_FUSED_Q8_SSM_RESIDUAL").is_some()
        {
            return Ok(None);
        }
        if named_pipeline(
            input.device().as_metal_device()?,
            "dinf_scalar_q8_0_residual_r4_sg2",
        )?
        .is_none()
        {
            return Ok(None);
        }
        let input = input.contiguous()?;
        let residual = residual.contiguous()?;
        let output = self.weights.apply_op3_no_bwd(
            &input,
            &residual,
            &FusedQ8Residual {
                in_dim: self.in_dim,
                out_dim: self.out_dim,
            },
        )?;
        Ok(Some(output))
    }

    pub(super) fn forward_relu_squared(&self, input: &Tensor) -> Result<Tensor> {
        let dims = input.dims();
        let tokens = dims.get(dims.len().saturating_sub(2)).copied().unwrap_or(0);
        let batch = dims[..dims.len().saturating_sub(2)]
            .iter()
            .product::<usize>();
        if input.rank() == 3
            && batch == 1
            && self.profile.uses_tensor_batch(tokens)
            && (tokens >= MIN_TENSOR_TOKENS
                || (self.dtype == GgmlDType::Q4K
                    && (tokens >= MIN_Q4K_KSPLIT_TOKENS
                        || ((MIN_Q4K_VERIFY_TOKENS..=q4k_verify_max_tokens()).contains(&tokens)
                            && self.out_dim >= q4k_verify_min_out()))))
        {
            match self.forward_with_tile_and_epilogue(input, None, true) {
                Ok(output) => return Ok(output),
                Err(error) => {
                    tracing::debug!(
                        %error,
                        "fused ReLU-squared Metal matmul unavailable; using unfused fallback"
                    );
                }
            }
        }

        let output = self.forward(input)?;
        let relu = output.relu()?;
        &relu * &relu
    }

    pub(super) fn forward_with_tile(
        &self,
        input: &Tensor,
        preferred_tile_tokens: Option<usize>,
    ) -> Result<Tensor> {
        self.forward_with_tile_and_epilogue(input, preferred_tile_tokens, false)
    }

    fn forward_with_tile_and_epilogue(
        &self,
        input: &Tensor,
        preferred_tile_tokens: Option<usize>,
        relu_squared_epilogue: bool,
    ) -> Result<Tensor> {
        if preferred_tile_tokens.is_some_and(|tile| {
            !matches!(
                tile,
                VERIFY_TILE_TOKENS
                    | TINY_TILE_TOKENS
                    | SMALL_TILE_TOKENS
                    | MEDIUM_TILE_TOKENS
                    | LARGE_TILE_TOKENS
            )
        }) {
            candle_core::bail!("unsupported preferred Metal tensor token tile")
        }
        if !input.device().is_metal()
            || input.dtype() != DType::F32
            || !matches!(input.rank(), 2 | 3)
        {
            candle_core::bail!(
                "Metal tensor quantized matmul requires a rank-2 or rank-3 Metal f32 input"
            )
        }
        self.weights.apply_op2_no_bwd(
            &input.contiguous()?,
            &TensorQMatMul {
                dtype: self.dtype,
                out_dim: self.out_dim,
                in_dim: self.in_dim,
                preferred_tile_tokens,
                relu_squared_epilogue,
                output_groups: None,
                profile: self.profile,
                prefill_weight_reuse: self.prefill_weight_reuse,
                decode_rows: false,
            },
        )
    }
}

struct FusedQ6KSwiGlu {
    in_dim: usize,
    intermediate_dim: usize,
}

impl CustomOp2 for FusedQ6KSwiGlu {
    fn name(&self) -> &'static str {
        "dinf-pre-m5-q6k-swiglu"
    }

    fn cpu_fwd(
        &self,
        _weights: &CpuStorage,
        _weights_layout: &Layout,
        _input: &CpuStorage,
        _input_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused Q6_K SwiGLU is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        weights: &MetalStorage,
        weights_layout: &Layout,
        input: &MetalStorage,
        input_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if weights.dtype() != DType::U8
            || input.dtype() != DType::F32
            || !weights_layout.is_contiguous()
            || !input_layout.is_contiguous()
            || !matches!(input_layout.dims().len(), 2 | 3)
        {
            candle_core::bail!("fused Q6_K SwiGLU requires contiguous u8/f32 inputs")
        }
        let input_dims = input_layout.dims();
        if input_dims[input_dims.len() - 2] != 1
            || input_dims[input_dims.len() - 1] != self.in_dim
            || input_dims[..input_dims.len() - 2].iter().product::<usize>() != 1
        {
            candle_core::bail!("fused Q6_K SwiGLU input geometry differs")
        }
        let expected_weights = 2 * self.intermediate_dim * self.in_dim
            / GgmlDType::Q6K.block_size()
            * GgmlDType::Q6K.type_size();
        if weights_layout.shape().elem_count() != expected_weights
            || weights_layout.start_offset() != 0
            || weights.device().id() != input.device().id()
        {
            candle_core::bail!("fused Q6_K SwiGLU weight geometry differs")
        }
        if self.in_dim > u32::MAX as usize || self.intermediate_dim > u32::MAX as usize {
            candle_core::bail!("fused Q6_K SwiGLU geometry exceeds u32")
        }
        let mut output_dims = input_dims.to_vec();
        *output_dims.last_mut().unwrap() = self.intermediate_dim;
        let output_shape = Shape::from(output_dims);
        let output_elements = output_shape.elem_count();
        let device = input.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, DType::F32)
            .with_label("dinf-pre-m5-q6k-swiglu")
            .build()?;
        let pipeline = named_pipeline(device, "dinf_scalar_q6k_swiglu_r2_sg2")?
            .ok_or_else(|| candle_core::Error::Msg("fused Q6_K SwiGLU unavailable".into()))?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-pre-m5-q6k-swiglu");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(0, Some(weights.buffer()), 0);
        encoder_ref.set_input_buffer(
            1,
            Some(input.buffer()),
            input_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(2, Some(&output), 0);
        encoder_ref.set_bytes(3, &(self.in_dim as u32));
        encoder_ref.set_bytes(4, &(self.intermediate_dim as u32));
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: self.intermediate_dim.div_ceil(4),
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), output_elements, DType::F32),
            output_shape,
        ))
    }
}

struct FusedQ6KResidual {
    in_dim: usize,
    out_dim: usize,
}

impl CustomOp3 for FusedQ6KResidual {
    fn name(&self) -> &'static str {
        "dinf-pre-m5-q6k-residual"
    }

    fn cpu_fwd(
        &self,
        _weights: &CpuStorage,
        _weights_layout: &Layout,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _residual: &CpuStorage,
        _residual_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused Q6_K residual projection is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        weights: &MetalStorage,
        weights_layout: &Layout,
        input: &MetalStorage,
        input_layout: &Layout,
        residual: &MetalStorage,
        residual_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if weights.dtype() != DType::U8
            || input.dtype() != DType::F32
            || residual.dtype() != DType::F32
            || !weights_layout.is_contiguous()
            || !input_layout.is_contiguous()
            || !residual_layout.is_contiguous()
            || !matches!(input_layout.dims().len(), 2 | 3)
            || residual_layout.dims().len() != input_layout.dims().len()
        {
            candle_core::bail!("fused Q6_K residual projection requires contiguous u8/f32 inputs")
        }
        let input_dims = input_layout.dims();
        let mut output_dims = input_dims.to_vec();
        *output_dims.last_mut().unwrap() = self.out_dim;
        if input_dims[input_dims.len() - 2] != 1
            || input_dims[input_dims.len() - 1] != self.in_dim
            || input_dims[..input_dims.len() - 2].iter().product::<usize>() != 1
            || residual_layout.dims() != output_dims.as_slice()
        {
            candle_core::bail!("fused Q6_K residual projection input geometry differs")
        }
        let expected_weights = self
            .out_dim
            .checked_mul(self.in_dim)
            .and_then(|elements| elements.checked_div(GgmlDType::Q6K.block_size()))
            .and_then(|blocks| blocks.checked_mul(GgmlDType::Q6K.type_size()))
            .ok_or_else(|| {
                candle_core::Error::Msg("fused Q6_K residual weight size overflow".into())
            })?;
        if weights_layout.shape().elem_count() != expected_weights
            || weights_layout.start_offset() != 0
            || weights.device().id() != input.device().id()
            || weights.device().id() != residual.device().id()
        {
            candle_core::bail!("fused Q6_K residual projection weight geometry differs")
        }
        if !self.in_dim.is_multiple_of(GgmlDType::Q6K.block_size())
            || self.in_dim > u32::MAX as usize
            || self.out_dim > u32::MAX as usize
        {
            candle_core::bail!("fused Q6_K residual projection geometry exceeds kernel limits")
        }

        let output_shape = Shape::from(output_dims);
        let output_elements = output_shape.elem_count();
        let device = input.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, DType::F32)
            .with_label("dinf-pre-m5-q6k-residual")
            .build()?;
        let pipeline = named_pipeline(device, "dinf_scalar_q6k_residual_r2_sg2")?
            .ok_or_else(|| candle_core::Error::Msg("fused Q6_K residual unavailable".into()))?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-pre-m5-q6k-residual");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(0, Some(weights.buffer()), 0);
        encoder_ref.set_input_buffer(
            1,
            Some(input.buffer()),
            input_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            2,
            Some(residual.buffer()),
            residual_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(3, Some(&output), 0);
        encoder_ref.set_bytes(4, &(self.in_dim as u32));
        encoder_ref.set_bytes(5, &(self.out_dim as u32));
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: self.out_dim.div_ceil(4),
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), output_elements, DType::F32),
            output_shape,
        ))
    }
}

struct FusedQ8Residual {
    in_dim: usize,
    out_dim: usize,
}

impl CustomOp3 for FusedQ8Residual {
    fn name(&self) -> &'static str {
        "dinf-pre-m5-q8-ssm-residual"
    }

    fn cpu_fwd(
        &self,
        _weights: &CpuStorage,
        _weights_layout: &Layout,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _residual: &CpuStorage,
        _residual_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused Q8_0 SSM residual projection is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        weights: &MetalStorage,
        weights_layout: &Layout,
        input: &MetalStorage,
        input_layout: &Layout,
        residual: &MetalStorage,
        residual_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if weights.dtype() != DType::U8
            || input.dtype() != DType::F32
            || residual.dtype() != DType::F32
            || !weights_layout.is_contiguous()
            || !input_layout.is_contiguous()
            || !residual_layout.is_contiguous()
            || !matches!(input_layout.dims().len(), 2 | 3)
            || residual_layout.dims().len() != input_layout.dims().len()
        {
            candle_core::bail!(
                "fused Q8_0 SSM residual projection requires contiguous u8/f32 inputs"
            )
        }
        let input_dims = input_layout.dims();
        let mut output_dims = input_dims.to_vec();
        *output_dims.last_mut().unwrap() = self.out_dim;
        if input_dims[input_dims.len() - 2] != 1
            || input_dims[input_dims.len() - 1] != self.in_dim
            || input_dims[..input_dims.len() - 2].iter().product::<usize>() != 1
            || residual_layout.dims() != output_dims.as_slice()
        {
            candle_core::bail!("fused Q8_0 SSM residual projection input geometry differs")
        }
        let expected_weights = self
            .out_dim
            .checked_mul(self.in_dim)
            .and_then(|elements| elements.checked_div(GgmlDType::Q8_0.block_size()))
            .and_then(|blocks| blocks.checked_mul(GgmlDType::Q8_0.type_size()))
            .ok_or_else(|| {
                candle_core::Error::Msg("fused Q8_0 SSM residual weight size overflow".into())
            })?;
        if weights_layout.shape().elem_count() != expected_weights
            || weights_layout.start_offset() != 0
            || weights.device().id() != input.device().id()
            || weights.device().id() != residual.device().id()
        {
            candle_core::bail!("fused Q8_0 SSM residual projection weight geometry differs")
        }
        if !self.in_dim.is_multiple_of(GgmlDType::Q8_0.block_size())
            || self.in_dim > u32::MAX as usize
            || self.out_dim > u32::MAX as usize
        {
            candle_core::bail!("fused Q8_0 SSM residual projection geometry exceeds kernel limits")
        }

        let output_shape = Shape::from(output_dims);
        let output_elements = output_shape.elem_count();
        let device = input.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, DType::F32)
            .with_label("dinf-pre-m5-q8-ssm-residual")
            .build()?;
        let pipeline = named_pipeline(device, "dinf_scalar_q8_0_residual_r4_sg2")?
            .ok_or_else(|| candle_core::Error::Msg("fused Q8_0 SSM residual unavailable".into()))?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-pre-m5-q8-ssm-residual");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(0, Some(weights.buffer()), 0);
        encoder_ref.set_input_buffer(
            1,
            Some(input.buffer()),
            input_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            2,
            Some(residual.buffer()),
            residual_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(3, Some(&output), 0);
        encoder_ref.set_bytes(4, &(self.in_dim as u32));
        encoder_ref.set_bytes(5, &(self.out_dim as u32));
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: self.out_dim.div_ceil(8),
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), output_elements, DType::F32),
            output_shape,
        ))
    }
}

struct TensorQMatMul {
    dtype: GgmlDType,
    out_dim: usize,
    in_dim: usize,
    preferred_tile_tokens: Option<usize>,
    prefill_weight_reuse: bool,
    relu_squared_epilogue: bool,
    output_groups: Option<Arc<[usize]>>,
    profile: MetalTensorProfile,
    decode_rows: bool,
}

impl CustomOp2 for TensorQMatMul {
    fn name(&self) -> &'static str {
        "dinf-metal4-tensor-qmatmul"
    }

    fn cpu_fwd(
        &self,
        _weights: &CpuStorage,
        _weights_layout: &Layout,
        _input: &CpuStorage,
        _input_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("Metal tensor quantized matmul is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        weights: &MetalStorage,
        weights_layout: &Layout,
        input: &MetalStorage,
        input_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if weights.dtype() != DType::U8
            || input.dtype() != DType::F32
            || !weights_layout.is_contiguous()
            || !input_layout.is_contiguous()
        {
            candle_core::bail!("Metal tensor quantized matmul requires contiguous u8/f32 inputs")
        }
        let input_dims = input_layout.dims();
        if !matches!(input_dims.len(), 2 | 3) {
            candle_core::bail!("Metal tensor quantized matmul input geometry differs")
        }
        let tokens = input_dims[input_dims.len() - 2];
        let in_dim = input_dims[input_dims.len() - 1];
        let batch = input_dims[..input_dims.len() - 2].iter().product::<usize>();
        if in_dim != self.in_dim {
            candle_core::bail!("Metal tensor quantized matmul input geometry differs")
        }
        let expected =
            self.out_dim * self.in_dim / self.dtype.block_size() * self.dtype.type_size();
        if weights_layout.shape().elem_count() != expected || weights_layout.start_offset() != 0 {
            candle_core::bail!("Metal tensor quantized matmul weight geometry differs")
        }
        if weights.device().id() != input.device().id() {
            candle_core::bail!("Metal tensor quantized matmul inputs are on different devices")
        }

        let mut output_dims = input_dims.to_vec();
        *output_dims.last_mut().unwrap() = self.out_dim;
        let output_shape = Shape::from(output_dims);
        let output_elements = output_shape.elem_count();
        let device = input.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, DType::F32)
            .with_label("dinf-metal4-tensor-qmatmul")
            .build()?;
        let tensor_dispatched = !self.decode_rows
            && dispatch_tensor_qmatmul(
                self.dtype,
                self.in_dim,
                self.out_dim,
                batch,
                tokens,
                input_dims.len(),
                self.preferred_tile_tokens,
                self.relu_squared_epilogue,
                self.output_groups.as_deref(),
                self.profile,
                self.prefill_weight_reuse,
                weights,
                input,
                input_layout,
                &output,
            )?;
        if !tensor_dispatched {
            if self.relu_squared_epilogue {
                candle_core::bail!(
                    "fused Metal tensor quantized matmul epilogue requires tensor dispatch"
                )
            }
            dispatch_legacy_qmatmul(
                self.dtype,
                self.in_dim,
                self.out_dim,
                batch,
                tokens,
                self.profile,
                weights,
                input,
                input_layout,
                &output,
                &output_shape,
                self.decode_rows,
            )?;
        }
        Ok((
            MetalStorage::new(output, device.clone(), output_elements, DType::F32),
            output_shape,
        ))
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_tensor_qmatmul(
    dtype: GgmlDType,
    in_dim: usize,
    out_dim: usize,
    batch: usize,
    tokens: usize,
    input_rank: usize,
    preferred_tile_tokens: Option<usize>,
    relu_squared_epilogue: bool,
    output_groups: Option<&[usize]>,
    profile: MetalTensorProfile,
    prefill_weight_reuse: bool,
    weights: &MetalStorage,
    input: &MetalStorage,
    input_layout: &Layout,
    output: &candle_metal_kernels::metal::Buffer,
) -> Result<bool> {
    if input_rank != 3 || batch != 1 {
        return Ok(false);
    }
    if !profile.uses_tensor_batch(tokens) {
        return Ok(false);
    }
    if in_dim > u32::MAX as usize || out_dim > u32::MAX as usize || tokens > u32::MAX as usize {
        candle_core::bail!("Metal tensor quantized matmul geometry exceeds u32")
    }
    // A shared DFlash pass retains the per-block reduction and precision.
    // Extra token tiles change only the dispatch grid, not the dot product.
    let verify_batch = preferred_tile_tokens == Some(VERIFY_TILE_TOKENS)
        && tokens > VERIFY_TILE_TOKENS
        && tokens.is_multiple_of(VERIFY_TILE_TOKENS);
    let kernel_tokens = if verify_batch {
        VERIFY_TILE_TOKENS
    } else {
        tokens
    };
    if dtype == GgmlDType::Q4K
        && (MIN_Q4K_VERIFY_TOKENS..=q4k_verify_max_tokens()).contains(&tokens)
        && out_dim >= q4k_verify_min_out()
        && in_dim.is_multiple_of(GgmlDType::Q4K.block_size())
        && std::env::var_os("DINF_DISABLE_METAL4_Q4K_VERIFY").is_none()
        && dispatch_q4k_verify(
            in_dim,
            out_dim,
            tokens,
            relu_squared_epilogue,
            weights,
            input,
            input_layout,
            output,
        )?
    {
        return Ok(true);
    }
    if dtype == GgmlDType::Q4K
        && (MIN_Q4K_KSPLIT_TOKENS..=TINY_TILE_TOKENS).contains(&kernel_tokens)
        && in_dim.is_multiple_of(4 * 64)
        && std::env::var_os("DINF_DISABLE_METAL4_Q4K_KSPLIT").is_none()
        && std::env::var_os("DINF_DISABLE_METAL4_Q4K_HALF_KSPLIT").is_none()
        && {
            // Narrow batches are bound by the tensor matmul and dequantisation,
            // not bandwidth. Half-staged activations with a strict matmul (as
            // llama.cpp does) and 64-value steps that reuse each qs load are
            // faster than the float-input K32 kernels below.
            let kernel = if kernel_tokens <= VERIFY_TILE_TOKENS {
                "dinf_tensor_qmatmul_q4k_ksplit_k64_t8_h"
            } else {
                "dinf_tensor_qmatmul_q4k_ksplit_k64_t16_h"
            };
            dispatch_ksplit_staged(
                GgmlDType::Q4K,
                kernel,
                4,
                true,
                in_dim,
                out_dim,
                tokens,
                relu_squared_epilogue,
                None,
                weights,
                input,
                input_layout,
                output,
            )?
        }
    {
        return Ok(true);
    }
    if dtype == GgmlDType::Q4K
        && (MIN_Q4K_KSPLIT_TOKENS..=TINY_TILE_TOKENS).contains(&kernel_tokens)
        && in_dim.is_multiple_of(8 * 32)
        && std::env::var_os("DINF_DISABLE_METAL4_Q4K_KSPLIT").is_none()
        && std::env::var_os("DINF_DISABLE_METAL4_Q4K_K32_KSPLIT").is_none()
        && dispatch_ksplit_parts(
            GgmlDType::Q4K,
            "dinf_tensor_qmatmul_q4k_ksplit_k32_t16",
            4,
            in_dim,
            out_dim,
            tokens,
            relu_squared_epilogue,
            None,
            weights,
            input,
            input_layout,
            output,
        )?
    {
        return Ok(true);
    }
    if dtype == GgmlDType::Q4K
        && (MIN_Q4K_KSPLIT_TOKENS..=TINY_TILE_TOKENS).contains(&kernel_tokens)
        && in_dim.is_multiple_of(8 * 16)
        && std::env::var_os("DINF_DISABLE_METAL4_Q4K_KSPLIT").is_none()
        && dispatch_ksplit(
            GgmlDType::Q4K,
            "dinf_tensor_qmatmul_q4k_ksplit_t16",
            in_dim,
            out_dim,
            tokens,
            relu_squared_epilogue,
            None,
            weights,
            input,
            input_layout,
            output,
        )?
    {
        return Ok(true);
    }
    if dtype == GgmlDType::Q8_0
        && kernel_tokens == VERIFY_TILE_TOKENS
        && out_dim >= Q8_KSPLIT_MIN_OUT
        && in_dim.is_multiple_of(8 * 32)
        && std::env::var_os("DINF_DISABLE_METAL4_Q8_KSPLIT").is_none()
        && std::env::var_os("DINF_DISABLE_METAL4_Q8_K32_KSPLIT").is_none()
        && dispatch_ksplit(
            GgmlDType::Q8_0,
            "dinf_tensor_qmatmul_q8_0_ksplit_k32_t8",
            in_dim,
            out_dim,
            tokens,
            relu_squared_epilogue,
            output_groups,
            weights,
            input,
            input_layout,
            output,
        )?
    {
        return Ok(true);
    }
    if dtype == GgmlDType::Q8_0
        && kernel_tokens == VERIFY_TILE_TOKENS
        && out_dim >= Q8_KSPLIT_MIN_OUT
        && in_dim.is_multiple_of(8 * 16)
        && std::env::var_os("DINF_DISABLE_METAL4_Q8_KSPLIT").is_none()
        && dispatch_ksplit(
            GgmlDType::Q8_0,
            "dinf_tensor_qmatmul_q8_0_ksplit_t8",
            in_dim,
            out_dim,
            tokens,
            relu_squared_epilogue,
            output_groups,
            weights,
            input,
            input_layout,
            output,
        )?
    {
        return Ok(true);
    }
    // Q6_K verify rows, including the vocabulary head: half-staged
    // activations, a strict matmul, and paired 2-byte weight loads.
    if dtype == GgmlDType::Q6K
        && (MIN_Q4K_KSPLIT_TOKENS..=VERIFY_TILE_TOKENS).contains(&kernel_tokens)
        && in_dim.is_multiple_of(8 * 32)
        && std::env::var_os("DINF_DISABLE_METAL4_Q6_HALF_KSPLIT").is_none()
        && dispatch_ksplit_staged(
            GgmlDType::Q6K,
            "dinf_tensor_qmatmul_q6k_ksplit_k32_t8_h",
            8,
            true,
            in_dim,
            out_dim,
            tokens,
            relu_squared_epilogue,
            None,
            weights,
            input,
            input_layout,
            output,
        )?
    {
        return Ok(true);
    }
    if dtype == GgmlDType::Q6K
        && kernel_tokens == VERIFY_TILE_TOKENS
        && (Q6_KSPLIT_MIN_OUT..=Q6_KSPLIT_MAX_OUT).contains(&out_dim)
        && in_dim.is_multiple_of(8 * 16)
        && std::env::var_os("DINF_DISABLE_METAL4_Q6_KSPLIT").is_none()
        && dispatch_ksplit(
            GgmlDType::Q6K,
            "dinf_tensor_qmatmul_q6k_ksplit_t8",
            in_dim,
            out_dim,
            tokens,
            relu_squared_epilogue,
            None,
            weights,
            input,
            input_layout,
            output,
        )?
    {
        return Ok(true);
    }
    if output_groups.is_some() {
        return Ok(false);
    }
    if tokens < MIN_TENSOR_TOKENS {
        return Ok(false);
    }
    let tile_tokens = if std::env::var_os("DINF_METAL4_QMM_TILE").is_some() {
        tile_tokens(tokens)
    } else {
        preferred_tile_tokens.unwrap_or_else(|| tile_tokens(tokens))
    };
    dispatch_tensor_pipeline(
        dtype,
        in_dim,
        out_dim,
        batch,
        tokens,
        tile_tokens,
        relu_squared_epilogue,
        prefill_weight_reuse,
        weights,
        input,
        input_layout,
        output,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_tensor_pipeline(
    dtype: GgmlDType,
    in_dim: usize,
    out_dim: usize,
    batch: usize,
    tokens: usize,
    tile_tokens: usize,
    relu_squared_epilogue: bool,
    prefill_weight_reuse: bool,
    weights: &MetalStorage,
    input: &MetalStorage,
    input_layout: &Layout,
    output: &candle_metal_kernels::metal::Buffer,
) -> Result<bool> {
    let device = input.device();
    let mut reuse = prefill_weight_reuse && use_prefill_weight_reuse(dtype, tokens, tile_tokens);
    let staged = if reuse {
        named_pipeline(device, "dinf_tensor_qmatmul_q4k_t64_w4")?
    } else {
        None
    };
    let half_input = if staged.is_none() {
        half_input_pipeline(device, dtype, tile_tokens)?
    } else {
        None
    };
    let pipeline = if let Some(staged) = staged {
        Some(staged)
    } else if let Some(half_input) = half_input.clone() {
        reuse = false;
        Some(half_input)
    } else {
        reuse = false;
        pipeline(device, dtype, tile_tokens)?
    };
    let Some(pipeline) = pipeline else {
        return Ok(false);
    };
    let half_buffer = if half_input.is_some() {
        Some(cast_input_to_half(
            device,
            input,
            input_layout,
            batch * tokens * in_dim,
        )?)
    } else {
        None
    };
    let encoder = device.command_encoder()?;
    encoder.set_label("dinf-metal4-tensor-qmatmul");
    encoder.set_compute_pipeline_state(&pipeline);
    let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
    encoder_ref.set_input_buffer(0, Some(weights.buffer()), 0);
    if let Some(half_buffer) = half_buffer.as_ref() {
        encoder_ref.set_input_buffer(1, Some(half_buffer), 0);
    } else {
        encoder_ref.set_input_buffer(
            1,
            Some(input.buffer()),
            input_layout.start_offset() * DType::F32.size_in_bytes(),
        );
    }
    encoder_ref.set_output_buffer(2, Some(output), 0);
    encoder_ref.set_bytes(3, &(in_dim as u32));
    encoder_ref.set_bytes(4, &(out_dim as u32));
    encoder_ref.set_bytes(5, &(tokens as u32));
    let weight_row_bytes = in_dim / dtype.block_size() * dtype.type_size();
    encoder_ref.set_bytes(6, &(weight_row_bytes as u32));
    encoder_ref.set_bytes(7, &u32::from(relu_squared_epilogue));
    encoder_ref.set_threadgroup_memory_length(
        0,
        64 * 32 * (if reuse { 4 } else { 1 }) * DType::F16.size_in_bytes(),
    );
    encoder_ref.dispatch_thread_groups(
        objc2_metal::MTLSize {
            width: tokens.div_ceil(tile_tokens),
            height: out_dim.div_ceil(64),
            depth: batch,
        },
        objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(true)
}

/// Convert contiguous f32 activations to a pooled half buffer.
fn cast_input_to_half(
    device: &candle_core::MetalDevice,
    input: &MetalStorage,
    input_layout: &Layout,
    count: usize,
) -> Result<Arc<candle_metal_kernels::metal::Buffer>> {
    if !count.is_multiple_of(4) {
        candle_core::bail!("half activation staging requires a multiple of four values");
    }
    let Some(pipeline) = named_pipeline(device, "dinf_cast_f32_f16")? else {
        candle_core::bail!("dinf_cast_f32_f16 is unavailable");
    };
    let output = device.new_buffer(count, DType::F16, "dinf-metal4-half-activations")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("dinf-metal4-half-activations");
    encoder.set_compute_pipeline_state(&pipeline);
    let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
    encoder_ref.set_input_buffer(
        0,
        Some(input.buffer()),
        input_layout.start_offset() * DType::F32.size_in_bytes(),
    );
    encoder_ref.set_output_buffer(1, Some(&output), 0);
    let count4 = count / 4;
    encoder_ref.set_bytes(2, &(count4 as u32));
    encoder_ref.dispatch_thread_groups(
        objc2_metal::MTLSize {
            width: count4.div_ceil(256),
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(output)
}

/// Q8_0 prefill tiles stage activations through half, as llama.cpp does.
fn half_input_pipeline(
    device: &candle_core::MetalDevice,
    dtype: GgmlDType,
    tile_tokens: usize,
) -> Result<Option<ComputePipeline>> {
    if std::env::var_os("DINF_DISABLE_METAL4_HALF_INPUT_QMM").is_some() {
        return Ok(None);
    }
    let name = match (dtype, tile_tokens) {
        (GgmlDType::Q8_0, MEDIUM_TILE_TOKENS) => "dinf_tensor_qmatmul_q8_0_h64",
        (GgmlDType::Q8_0, LARGE_TILE_TOKENS) => "dinf_tensor_qmatmul_q8_0_h128",
        _ => return Ok(None),
    };
    named_pipeline(device, name)
}

fn use_prefill_weight_reuse(dtype: GgmlDType, tokens: usize, tile_tokens: usize) -> bool {
    // Measured on M5 Qwen's bounded prompt passes. A 192-row down projection
    // regressed; do not extend this to larger prompts or eight-row draft tiles.
    dtype == GgmlDType::Q4K && tile_tokens == 64 && (64..=128).contains(&tokens)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_q4k_verify(
    in_dim: usize,
    out_dim: usize,
    tokens: usize,
    relu_squared_epilogue: bool,
    weights: &MetalStorage,
    input: &MetalStorage,
    input_layout: &Layout,
    output: &candle_metal_kernels::metal::Buffer,
) -> Result<bool> {
    let msg = out_dim >= q4k_verify_msg_min_out();
    let name = q4k_verify_kernel_name(tokens, msg);
    let Some(name) = name else {
        return Ok(false);
    };
    let device = input.device();
    let Some(pipeline) = named_pipeline(device, name)? else {
        return Ok(false);
    };
    let encoder = device.command_encoder()?;
    encoder.set_label(if msg {
        "dinf-metal4-q4k-verify-msg"
    } else {
        "dinf-metal4-q4k-verify-ksplit"
    });
    encoder.set_compute_pipeline_state(&pipeline);
    let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
    encoder_ref.set_input_buffer(0, Some(weights.buffer()), 0);
    encoder_ref.set_input_buffer(
        1,
        Some(input.buffer()),
        input_layout.start_offset() * DType::F32.size_in_bytes(),
    );
    encoder_ref.set_output_buffer(2, Some(output), 0);
    encoder_ref.set_bytes(3, &(in_dim as u32));
    encoder_ref.set_bytes(4, &(out_dim as u32));
    encoder_ref.set_bytes(5, &(tokens as u32));
    let weight_row_bytes = in_dim / GgmlDType::Q4K.block_size() * GgmlDType::Q4K.type_size();
    encoder_ref.set_bytes(6, &(weight_row_bytes as u32));
    encoder_ref.set_bytes(7, &u32::from(relu_squared_epilogue));
    let (rows_per_threadgroup, simdgroups) = if msg {
        (4 * Q4K_VERIFY_MSG_SIMDGROUPS, Q4K_VERIFY_MSG_SIMDGROUPS)
    } else {
        (4, Q4K_VERIFY_K_PARTS)
    };
    encoder_ref.dispatch_thread_groups(
        objc2_metal::MTLSize {
            width: out_dim.div_ceil(rows_per_threadgroup),
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: simdgroups * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_ksplit(
    dtype: GgmlDType,
    kernel: &'static str,
    in_dim: usize,
    out_dim: usize,
    tokens: usize,
    relu_squared_epilogue: bool,
    output_groups: Option<&[usize]>,
    weights: &MetalStorage,
    input: &MetalStorage,
    input_layout: &Layout,
    output: &candle_metal_kernels::metal::Buffer,
) -> Result<bool> {
    dispatch_ksplit_parts(
        dtype,
        kernel,
        8,
        in_dim,
        out_dim,
        tokens,
        relu_squared_epilogue,
        output_groups,
        weights,
        input,
        input_layout,
        output,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_ksplit_parts(
    dtype: GgmlDType,
    kernel: &'static str,
    split_parts: usize,
    in_dim: usize,
    out_dim: usize,
    tokens: usize,
    relu_squared_epilogue: bool,
    output_groups: Option<&[usize]>,
    weights: &MetalStorage,
    input: &MetalStorage,
    input_layout: &Layout,
    output: &candle_metal_kernels::metal::Buffer,
) -> Result<bool> {
    dispatch_ksplit_staged(
        dtype,
        kernel,
        split_parts,
        false,
        in_dim,
        out_dim,
        tokens,
        relu_squared_epilogue,
        output_groups,
        weights,
        input,
        input_layout,
        output,
    )
}

/// K-split dispatch; `half_input` stages the activations through half first
/// for kernels instantiated with half input, as llama.cpp does.
#[allow(clippy::too_many_arguments)]
fn dispatch_ksplit_staged(
    dtype: GgmlDType,
    kernel: &'static str,
    split_parts: usize,
    half_input: bool,
    in_dim: usize,
    out_dim: usize,
    tokens: usize,
    relu_squared_epilogue: bool,
    output_groups: Option<&[usize]>,
    weights: &MetalStorage,
    input: &MetalStorage,
    input_layout: &Layout,
    output: &candle_metal_kernels::metal::Buffer,
) -> Result<bool> {
    let device = input.device();
    let Some(pipeline) = named_pipeline(device, kernel)? else {
        return Ok(false);
    };
    let staged = if half_input {
        Some(cast_input_to_half(
            device,
            input,
            input_layout,
            tokens * in_dim,
        )?)
    } else {
        None
    };
    let encoder = device.command_encoder()?;
    encoder.set_label("dinf-metal4-ksplit");
    encoder.set_compute_pipeline_state(&pipeline);
    let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
    encoder_ref.set_input_buffer(0, Some(weights.buffer()), 0);
    if let Some(staged) = staged.as_ref() {
        encoder_ref.set_input_buffer(1, Some(staged), 0);
    } else {
        encoder_ref.set_input_buffer(
            1,
            Some(input.buffer()),
            input_layout.start_offset() * DType::F32.size_in_bytes(),
        );
    }
    encoder_ref.set_output_buffer(2, Some(output), 0);
    encoder_ref.set_bytes(3, &(in_dim as u32));
    encoder_ref.set_bytes(4, &(out_dim as u32));
    encoder_ref.set_bytes(5, &(tokens as u32));
    let weight_row_bytes = in_dim / dtype.block_size() * dtype.type_size();
    encoder_ref.set_bytes(6, &(weight_row_bytes as u32));
    encoder_ref.set_bytes(7, &u32::from(relu_squared_epilogue));
    let mut group_starts = [0u32; 5];
    let group_count = output_groups.map_or(0, |groups| groups.len());
    if let Some(groups) = output_groups {
        if !(2..=4).contains(&groups.len()) || groups.iter().sum::<usize>() != out_dim {
            candle_core::bail!("grouped Metal Q8 output geometry differs")
        }
        let mut start = 0usize;
        for (index, &width) in groups.iter().enumerate() {
            group_starts[index] = start as u32;
            start += width;
        }
        group_starts[groups.len()] = start as u32;
    }
    encoder_ref.set_bytes(8, &group_starts);
    encoder_ref.set_bytes(9, &(group_count as u32));
    let tile = if kernel.ends_with("t16") || kernel.ends_with("t16_h") {
        TINY_TILE_TOKENS
    } else {
        VERIFY_TILE_TOKENS
    };
    encoder_ref.dispatch_thread_groups(
        objc2_metal::MTLSize {
            width: out_dim.div_ceil(32),
            height: tokens.div_ceil(tile),
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: split_parts * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_pre_m5_scalar_q6_r2(
    in_dim: usize,
    out_dim: usize,
    weights: &MetalStorage,
    input: &MetalStorage,
    input_layout: &Layout,
    output: &candle_metal_kernels::metal::Buffer,
) -> Result<bool> {
    if !in_dim.is_multiple_of(GgmlDType::Q6K.block_size()) {
        return Ok(false);
    }
    let device = input.device();
    let Some(pipeline) = named_pipeline(device, "dinf_scalar_q6k_r2_sg2")? else {
        return Ok(false);
    };
    let encoder = device.command_encoder()?;
    encoder.set_label("dinf-scalar-q6k-r2-sg2");
    encoder.set_compute_pipeline_state(&pipeline);
    let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
    encoder_ref.set_input_buffer(0, Some(weights.buffer()), 0);
    encoder_ref.set_input_buffer(
        1,
        Some(input.buffer()),
        input_layout.start_offset() * DType::F32.size_in_bytes(),
    );
    encoder_ref.set_output_buffer(2, Some(output), 0);
    encoder_ref.set_bytes(3, &(in_dim as u32));
    encoder_ref.set_bytes(4, &(out_dim as u32));
    encoder_ref.dispatch_thread_groups(
        objc2_metal::MTLSize {
            width: out_dim.div_ceil(4),
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_legacy_qmatmul(
    dtype: GgmlDType,
    in_dim: usize,
    out_dim: usize,
    batch: usize,
    tokens: usize,
    profile: MetalTensorProfile,
    weights: &MetalStorage,
    input: &MetalStorage,
    input_layout: &Layout,
    output: &candle_metal_kernels::metal::Buffer,
    output_shape: &Shape,
    decode_rows: bool,
) -> Result<()> {
    let packed_ratio = dtype.type_size() as f32 / dtype.block_size() as f32;
    let kernel_dtype = match dtype {
        GgmlDType::Q4_0 => candle_metal_kernels::GgmlDType::Q4_0,
        GgmlDType::Q5_1 => candle_metal_kernels::GgmlDType::Q5_1,
        GgmlDType::Q8_0 => candle_metal_kernels::GgmlDType::Q8_0,
        GgmlDType::Q4K => candle_metal_kernels::GgmlDType::Q4K,
        GgmlDType::Q5K => candle_metal_kernels::GgmlDType::Q5K,
        GgmlDType::Q6K => candle_metal_kernels::GgmlDType::Q6K,
        _ => unreachable!("unsupported Metal tensor quantization"),
    };
    let device = input.device();
    if profile == MetalTensorProfile::PreM5
        && dtype == GgmlDType::Q6K
        && tokens == 1
        && batch == 1
        && pre_m5_scalar_q6_r2_enabled()
        && dispatch_pre_m5_scalar_q6_r2(in_dim, out_dim, weights, input, input_layout, output)?
    {
        return Ok(());
    }
    let encoder = device.command_encoder()?;
    if tokens == 1 || decode_rows {
        for batch_index in 0..batch {
            candle_metal_kernels::call_quantized_matmul_mv_t(
                device.device(),
                &encoder,
                device.kernels(),
                kernel_dtype,
                (1, tokens, out_dim, in_dim),
                input.buffer(),
                (input_layout.start_offset() + batch_index * tokens * in_dim)
                    * DType::F32.size_in_bytes(),
                weights.buffer(),
                batch_index * tokens * out_dim * DType::F32.size_in_bytes(),
                output,
            )
            .map_err(candle_core::MetalError::from)?;
        }
    } else {
        let weight_shape = vec![1, 1, out_dim, in_dim];
        let weight_layout = Layout::contiguous(weight_shape.clone());
        let weight_stride = weight_layout
            .stride()
            .iter()
            .map(|stride| (*stride as f32 * packed_ratio) as usize)
            .collect::<Vec<_>>();
        let input_dims = input_layout.dims();
        let input_shape = [vec![1; 4 - input_dims.len()], input_dims.to_vec()].concat();
        let input_stride = Layout::contiguous(input_shape.clone())
            .stride()
            .iter()
            .map(|stride| stride * DType::F32.size_in_bytes())
            .collect::<Vec<_>>();
        candle_metal_kernels::call_quantized_matmul_mm_t(
            device.device(),
            &encoder,
            device.kernels(),
            kernel_dtype,
            &weight_shape,
            &weight_stride,
            weights.buffer(),
            &input_shape,
            &input_stride,
            input.buffer(),
            input_layout.start_offset() * DType::F32.size_in_bytes(),
            output_shape.dims(),
            0,
            output,
        )
        .map_err(candle_core::MetalError::from)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Module;
    use candle_core::quantized::QMatMul as CandleQMatMul;

    #[test]
    fn prefill_weight_reuse_stays_on_bounded_q4k_tiles() {
        for tokens in [1, 8, 16, 32, 63, 129, 192, 256] {
            assert!(!use_prefill_weight_reuse(GgmlDType::Q4K, tokens, 64));
        }
        for tokens in [64, 65, 96, 127, 128] {
            assert!(use_prefill_weight_reuse(GgmlDType::Q4K, tokens, 64));
            for tile in [8, 16, 32, 128] {
                assert!(!use_prefill_weight_reuse(GgmlDType::Q4K, tokens, tile));
            }
            for dtype in [
                GgmlDType::Q4_0,
                GgmlDType::Q5K,
                GgmlDType::Q6K,
                GgmlDType::Q8_0,
            ] {
                assert!(!use_prefill_weight_reuse(dtype, tokens, 64));
            }
        }
        assert!(!MetalTensorProfile::PreM5.uses_tensor_batch(128));
    }

    #[test]
    fn prefill_weight_reuse_matches_original_with_tails_offsets_and_epilogue() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        if !supports_metal_tensor(&device)? {
            return Ok(());
        }
        assert!(
            named_pipeline(device.as_metal_device()?, "dinf_tensor_qmatmul_q4k_t64_w4")?.is_some()
        );
        for in_dim in [256, 768, 5120] {
            let out_dim = 70; // A partial output tile, unlike the real FFN shapes.
            let weights = Tensor::from_vec(
                (0..out_dim * in_dim)
                    .map(|i| (i as f32 * 0.0137).sin() * 0.2 + (i as f32 * 0.0031).cos() * 0.05)
                    .collect::<Vec<_>>(),
                (out_dim, in_dim),
                &Device::Cpu,
            )?;
            let packed = Arc::new(QTensor::quantize(&weights, GgmlDType::Q4K)?);
            let mut projection = MetalTensorQMatMul::try_from_qtensor(packed, &device)?.unwrap();
            // Exercise the tensor kernel even on pre-M5 test hosts. Production
            // retains their legacy prefill path.
            projection.profile = MetalTensorProfile::Default;
            for tokens in [64, 65, 96, 127, 128] {
                let input = Tensor::from_vec(
                    (0..(tokens + 2) * in_dim)
                        .map(|i| (i as f32 * 0.021).cos() * 0.4)
                        .collect::<Vec<_>>(),
                    (1, tokens + 2, in_dim),
                    &device,
                )?
                .narrow(1, 1, tokens)?;
                for epilogue in [false, true] {
                    projection.prefill_weight_reuse = false;
                    let expected = projection
                        .forward_with_tile_and_epilogue(&input, Some(64), epilogue)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    projection.prefill_weight_reuse = true;
                    let actual = projection
                        .forward_with_tile_and_epilogue(&input, Some(64), epilogue)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    assert!(actual.iter().all(|v| v.is_finite()));
                    assert_eq!(
                        actual, expected,
                        "in={in_dim} tokens={tokens} epilogue={epilogue}"
                    );
                }
            }
        }
        Ok(())
    }

    /// Opt-in decode vs eight-row verify cost per projection on real weights.
    #[test]
    #[ignore = "set DINF_BENCH_QWEN_GGUF to a local Qwen GGUF"]
    fn bench_qwen_verify_projections() -> Result<()> {
        let device = Device::new_metal(0)?;
        let path = std::env::var("DINF_BENCH_QWEN_GGUF")
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        let mut file = std::fs::File::open(path)?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
        let names = std::env::var("DINF_BENCH_TENSORS").unwrap_or_else(|_| {
            "blk.0.attn_qkv.weight,blk.0.attn_gate.weight,blk.0.ssm_out.weight,\
             blk.0.ffn_gate.weight+blk.0.ffn_up.weight,blk.0.ffn_down.weight,\
             blk.1.ffn_down.weight,blk.3.attn_q.weight,blk.3.attn_output.weight"
                .into()
        });
        for group in names.split(',') {
            let packed = group
                .split('+')
                .map(|name| content.tensor(&mut file, name, &Device::Cpu).map(Arc::new))
                .collect::<Result<Vec<_>>>()?;
            let bytes: usize = packed.iter().map(|t| t.storage_size_in_bytes()).sum();
            let projection = if packed.len() == 1 {
                MetalTensorQMatMul::try_from_qtensor(Arc::clone(&packed[0]), &device)?
            } else {
                MetalTensorQMatMul::try_from_qtensors(&packed, &device)?
            }
            .expect("Metal tensor matmul required");
            let mut line = format!("{group} {:?} {:.1}MB", projection.dtype, bytes as f64 / 1e6);
            for tokens in [1usize, 8] {
                let input = Tensor::from_vec(
                    (0..tokens * projection.in_dim)
                        .map(|i| (i as f32 * 0.021).sin() * 0.4)
                        .collect::<Vec<_>>(),
                    (1, tokens, projection.in_dim),
                    &device,
                )?;
                let mut times = Vec::new();
                for _ in 0..7 {
                    device.synchronize()?;
                    let start = std::time::Instant::now();
                    for _ in 0..50 {
                        let _output = projection.forward(&input)?;
                    }
                    device.synchronize()?;
                    times.push(start.elapsed().as_secs_f64() * 1000.0 / 50.0);
                }
                times.sort_by(f64::total_cmp);
                let ms = times[times.len() / 2];
                line += &format!("  t{tokens} {ms:.4}ms {:.0}GB/s", bytes as f64 / ms / 1e6);
            }
            eprintln!("verify_bench {line}");
        }
        Ok(())
    }

    /// Opt-in, real-weight tile comparison. Stop other GPU work before running.
    #[test]
    #[ignore = "set DINF_BENCH_QWEN_GGUF to a local Qwen GGUF"]
    fn bench_qwen_prefill_tiles() -> Result<()> {
        let device = Device::new_metal(0)?;
        assert!(
            std::env::var_os("DINF_METAL4_QMM_TILE").is_none(),
            "unset DINF_METAL4_QMM_TILE before comparing tiles"
        );
        assert_eq!(
            metal_tensor_profile(&device)?,
            MetalTensorProfile::Default,
            "this benchmark measures the M5 tensor-prefill path"
        );
        let path = std::env::var("DINF_BENCH_QWEN_GGUF")
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        let mut file = std::fs::File::open(path)?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
        for names in [
            vec!["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"],
            vec!["blk.0.ffn_down.weight"],
        ] {
            let packed = names
                .iter()
                .map(|name| content.tensor(&mut file, name, &Device::Cpu).map(Arc::new))
                .collect::<Result<Vec<_>>>()?;
            let mut projection = if packed.len() == 1 {
                MetalTensorQMatMul::try_from_qtensor(Arc::clone(&packed[0]), &device)?
            } else {
                MetalTensorQMatMul::try_from_qtensors(&packed, &device)?
            }
            .expect("Metal tensor matmul required");
            let variants = [(32, false), (64, false), (128, false), (64, true)];
            for tokens in [64, 128, 192, 256] {
                let input = Tensor::from_vec(
                    (0..tokens * projection.in_dim)
                        .map(|i| (i as f32 * 0.021).sin() * 0.4)
                        .collect::<Vec<_>>(),
                    (1, tokens, projection.in_dim),
                    &device,
                )?;
                projection.prefill_weight_reuse = false;
                let reference = projection
                    .forward_with_tile(&input, Some(64))?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let mut runs = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                for repeat in 0..7 {
                    for offset in 0..variants.len() {
                        let slot = (repeat + offset) % variants.len();
                        let (tile, reuse) = variants[slot];
                        projection.prefill_weight_reuse = reuse;
                        let actual = projection
                            .forward_with_tile(&input, Some(tile))?
                            .flatten_all()?
                            .to_vec1::<f32>()?;
                        let diff = reference
                            .iter()
                            .zip(&actual)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0f32, f32::max);
                        assert!(actual.iter().chain(&reference).all(|v| v.is_finite()));
                        assert_eq!(actual.len(), reference.len());
                        assert_eq!(
                            diff, 0.0,
                            "{names:?} tokens={tokens} tile={tile} reuse={reuse}"
                        );
                        if repeat == 0 {
                            eprintln!(
                                "tile_check {names:?} {:?} tokens={tokens} tile={tile} reuse={reuse} max_abs={diff}",
                                projection.dtype
                            );
                        }
                        device.synchronize()?;
                        let start = std::time::Instant::now();
                        for _ in 0..20 {
                            let _output = projection.forward_with_tile(&input, Some(tile))?;
                        }
                        device.synchronize()?;
                        runs[slot].push(start.elapsed().as_secs_f64() * 1000.0 / 20.0);
                    }
                }
                for ((tile, reuse), mut times) in variants.into_iter().zip(runs) {
                    times.sort_by(f64::total_cmp);
                    eprintln!(
                        "tile_bench {names:?} {:?} tokens={tokens} tile={tile} reuse={reuse} median_ms={:.4} runs_ms={times:?}",
                        projection.dtype,
                        times[times.len() / 2]
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn independent_decode_rows_match_scalar_f32_with_offsets() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        if !supports_metal_tensor(&device)? {
            return Ok(());
        }
        for dtype in [GgmlDType::Q4K, GgmlDType::Q6K, GgmlDType::Q8_0] {
            let source = Tensor::from_vec(
                (0..520 * 512)
                    .map(|i| (i as f32 * 0.0137).sin() * 0.2)
                    .collect::<Vec<_>>(),
                (520, 512),
                &Device::Cpu,
            )?;
            let packed = Arc::new(QTensor::quantize(&source, dtype)?);
            let projection = MetalTensorQMatMul::try_from_qtensor(packed, &device)?.unwrap();
            let input = Tensor::from_vec(
                (0..9 * 512)
                    .map(|i| (i as f32 * 0.021).cos() * 0.4)
                    .collect::<Vec<_>>(),
                (1, 9, 512),
                &device,
            )?;
            for rows in [2, 3, 4, 8] {
                let input = input.narrow(1, 1, rows)?;
                let actual = projection.forward_decode_rows(&input)?;
                for row in 0..rows {
                    let expected = projection.forward(&input.narrow(1, row, 1)?)?;
                    let actual = actual.narrow(1, row, 1)?.flatten_all()?.to_vec1::<f32>()?;
                    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
                    for (&a, &b) in actual.iter().zip(&expected) {
                        assert!(a.is_finite() && b.is_finite());
                        assert!(
                            (a - b).abs() < 2e-5,
                            "{dtype:?}: rows {rows}, lane {row}: {a} != {b}"
                        );
                    }
                }
            }
            assert!(
                projection
                    .forward_decode_rows(&Tensor::zeros((1, 2, 256), DType::F32, &device)?)
                    .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn apple_soc_generation_parses_m1_without_matching_m10() {
        assert_eq!(apple_soc_generation("Apple M1"), Some(1));
        assert_eq!(apple_soc_generation("Apple M1 Pro"), Some(1));
        assert_eq!(apple_soc_generation("Apple M1 Max"), Some(1));
        assert_eq!(apple_soc_generation("Apple M5"), Some(5));
        assert_eq!(apple_soc_generation("Apple M10 Pro"), Some(10));
        assert_eq!(apple_soc_generation("AMD Radeon Pro"), None);
    }

    fn f64_qmatmul_reference(
        dequantized: &[Vec<f32>],
        input: &[f32],
        tokens: usize,
        in_dim: usize,
    ) -> Vec<f64> {
        let out_dim = dequantized.len();
        let mut reference = vec![0f64; tokens * out_dim];
        for token in 0..tokens {
            for (row, weights) in dequantized.iter().enumerate() {
                let mut sum = 0f64;
                for column in 0..in_dim {
                    sum += f64::from(input[token * in_dim + column]) * f64::from(weights[column]);
                }
                reference[token * out_dim + row] = sum;
            }
        }
        reference
    }

    fn assert_f64_qmatmul_contract(actual: &[f32], reference: &[f64], label: &str) {
        assert_eq!(
            actual.len(),
            reference.len(),
            "{label}: output length differs"
        );
        let mut max_abs = 0f64;
        let mut squared_error = 0f64;
        let mut squared_reference = 0f64;
        for (&actual, &reference) in actual.iter().zip(reference) {
            assert!(
                actual.is_finite() && reference.is_finite(),
                "{label}: non-finite output"
            );
            let error = f64::from(actual) - reference;
            max_abs = max_abs.max(error.abs());
            squared_error += error * error;
            squared_reference += reference * reference;
        }
        let relative_l2 = (squared_error / squared_reference.max(f64::EPSILON)).sqrt();
        // Metal 4 stages quantized weights through half before the strict
        // float-input accumulation. Keep the contract independent of another
        // GPU reduction order: this bound is against the dequantized F64 dot
        // product, with margin over the observed Q8_0 worst case (~9.6e-4).
        assert!(
            max_abs <= 2e-3 && relative_l2 <= 1e-4,
            "{label}: max_abs={max_abs:.6e}, relative_l2={relative_l2:.6e}"
        );
    }

    #[test]
    fn q8_tensor_paths_preserve_float_input_precision() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        if !supports_metal_tensor(&device)? {
            return Ok(());
        }
        // Cover ordinary tensor tiles, K32 split verification, and the K16
        // fallback selected when K is divisible by 128 but not 256. Test both
        // one verification block and four blocks in a shared dispatch.
        for (out_dim, in_dim, kernel) in [
            (520, 512, "dinf_tensor_qmatmul_q8_0_t8"),
            (8448, 512, "dinf_tensor_qmatmul_q8_0_ksplit_k32_t8"),
            (8448, 384, "dinf_tensor_qmatmul_q8_0_ksplit_t8"),
        ] {
            assert!(named_pipeline(device.as_metal_device()?, kernel)?.is_some());
            let source = Tensor::from_vec(
                (0..out_dim * in_dim)
                    .map(|i| (i as f32 * 0.0137).sin() * 0.2)
                    .collect::<Vec<_>>(),
                (out_dim, in_dim),
                &Device::Cpu,
            )?;
            let packed = Arc::new(QTensor::quantize(&source, GgmlDType::Q8_0)?);
            let reference_weights = packed.dequantize(&Device::Cpu)?.to_vec2::<f32>()?;
            let mut projection = MetalTensorQMatMul::try_from_qtensor(packed, &device)?.unwrap();
            // Exercise these kernels on pre-M5 hosts too, regardless of the
            // production policy for longer batches on those devices.
            projection.profile = MetalTensorProfile::Default;
            for tokens in [8, 32] {
                let values = (0..(tokens + 1) * in_dim)
                    .map(|i| (i as f32 * 0.021).cos() * 0.4)
                    .collect::<Vec<_>>();
                let input = Tensor::from_vec(values.clone(), (1, tokens + 1, in_dim), &device)?
                    .narrow(1, 1, tokens)?;
                let actual = projection
                    .forward_with_tile(&input, Some(VERIFY_TILE_TOKENS))?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let reference =
                    f64_qmatmul_reference(&reference_weights, &values[in_dim..], tokens, in_dim);
                assert_f64_qmatmul_contract(
                    &actual,
                    &reference,
                    &format!("Q8_0 {kernel} tokens={tokens}"),
                );
            }
        }
        Ok(())
    }

    #[test]
    fn q8_prefill_tiles_match_half_staged_reference() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        if !supports_metal_tensor(&device)? {
            return Ok(());
        }
        // Prefill tiles stage both operands in half, as llama.cpp's
        // kernel_mul_mm does. Compare against an F64 dot product of the
        // half-rounded weights and activations, so only the float
        // accumulation order may differ.
        let (out_dim, in_dim) = (200, 4096);
        let source = Tensor::from_vec(
            (0..out_dim * in_dim)
                .map(|i| (i as f32 * 0.0137).sin() * 0.2)
                .collect::<Vec<_>>(),
            (out_dim, in_dim),
            &Device::Cpu,
        )?;
        let packed = Arc::new(QTensor::quantize(&source, GgmlDType::Q8_0)?);
        let half_weights = packed
            .dequantize(&Device::Cpu)?
            .to_vec2::<f32>()?
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|w| half::f16::from_f32(w).to_f32())
                    .collect()
            })
            .collect::<Vec<Vec<f32>>>();
        let mut projection = MetalTensorQMatMul::try_from_qtensor(packed, &device)?.unwrap();
        projection.profile = MetalTensorProfile::Default;
        for (tokens, tile) in [(70, MEDIUM_TILE_TOKENS), (129, LARGE_TILE_TOKENS)] {
            let values = (0..tokens * in_dim)
                .map(|i| (i as f32 * 0.021).cos() * 0.4 + (i % 7) as f32 * 1e-3)
                .collect::<Vec<_>>();
            let input = Tensor::from_vec(values.clone(), (1, tokens, in_dim), &device)?;
            let actual = projection
                .forward_with_tile(&input, Some(tile))?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let half_values = values
                .iter()
                .map(|&v| half::f16::from_f32(v).to_f32())
                .collect::<Vec<_>>();
            let reference = f64_qmatmul_reference(&half_weights, &half_values, tokens, in_dim);
            let (mut max_abs, mut squared_error, mut squared_reference) = (0f64, 0f64, 0f64);
            for (&actual, &reference) in actual.iter().zip(&reference) {
                assert!(actual.is_finite());
                let error = f64::from(actual) - reference;
                max_abs = max_abs.max(error.abs());
                squared_error += error * error;
                squared_reference += reference * reference;
            }
            let relative_l2 = (squared_error / squared_reference).sqrt();
            eprintln!("Q8_0 prefill tile={tile}: max_abs={max_abs:e}, relative_l2={relative_l2:e}");
            assert!(
                max_abs <= 1e-4 && relative_l2 <= 1e-6,
                "Q8_0 prefill tile={tile}: max_abs={max_abs:e}, relative_l2={relative_l2:e}"
            );
        }
        Ok(())
    }

    #[test]
    fn shared_verification_tiles_match_separate_blocks() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        if !supports_metal_tensor(&device)? {
            return Ok(());
        }
        for dtype in [GgmlDType::Q4K, GgmlDType::Q6K, GgmlDType::Q8_0] {
            for width in [520, 8448] {
                let source = Tensor::from_vec(
                    (0..width * 512)
                        .map(|i| (i as f32 * 0.0137).sin() * 0.2)
                        .collect::<Vec<_>>(),
                    (width, 512),
                    &Device::Cpu,
                )?;
                let packed = Arc::new(QTensor::quantize(&source, dtype)?);
                let reference_weights = if dtype == GgmlDType::Q8_0 && width == 8448 {
                    Some(packed.dequantize(&Device::Cpu)?.to_vec2::<f32>()?)
                } else {
                    None
                };
                let mut projection =
                    MetalTensorQMatMul::try_from_qtensor(Arc::clone(&packed), &device)?.unwrap();
                let production_profile = projection.profile;
                let input_values = (0..33 * 512)
                    .map(|i| (i as f32 * 0.021).cos() * 0.4)
                    .collect::<Vec<_>>();
                let input = Tensor::from_vec(input_values.clone(), (1, 33, 512), &device)?;
                if production_profile == MetalTensorProfile::PreM5
                    && dtype == GgmlDType::Q8_0
                    && width == 8448
                {
                    // Preserve the actual M1-M4 production split as a
                    // separate contract: long batches use legacy GEMM while
                    // each eight-token verification block uses Metal 4.
                    // These reductions are not required to be bitwise equal,
                    // but both must stay within the independent F64 bound.
                    let production_input = input.narrow(1, 1, 4 * 8)?;
                    let production_shared = projection
                        .forward_with_tile(&production_input, Some(8))?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let mut production_separate = Vec::with_capacity(production_shared.len());
                    for lane in 0..4 {
                        production_separate.extend(
                            projection
                                .forward(&production_input.narrow(1, lane * 8, 8)?)?
                                .flatten_all()?
                                .to_vec1::<f32>()?,
                        );
                    }
                    let Some(dequantized) = reference_weights.as_deref() else {
                        unreachable!("Q8_0 production reference weights must be present");
                    };
                    let reference = f64_qmatmul_reference(
                        dequantized,
                        &input_values[512..(512 + 4 * 8 * 512)],
                        4 * 8,
                        512,
                    );
                    assert_f64_qmatmul_contract(
                        &production_shared,
                        &reference,
                        "Q8_0 pre-M5 shared legacy path",
                    );
                    assert_f64_qmatmul_contract(
                        &production_separate,
                        &reference,
                        "Q8_0 pre-M5 separate Metal 4 path",
                    );
                }
                // This test exercises the Metal 4 tiled kernels themselves.
                // Pre-M5 production intentionally routes batches over eight
                // tokens through Candle's legacy GEMM, so comparing that
                // fallback to the tensor kernel would conflate two valid
                // reduction paths (and is what made the old assertion fail on
                // M1). The fallback has its own parity coverage below.
                projection.profile = MetalTensorProfile::Default;
                for lanes in [2, 3, 4] {
                    let input = input.narrow(1, 1, lanes * 8)?;
                    let shared = projection.forward_with_tile(&input, Some(8))?;
                    let shared_values = shared.flatten_all()?.to_vec1::<f32>()?;
                    let mut separate_values = Vec::with_capacity(shared_values.len());
                    for lane in 0..lanes {
                        let separate = projection
                            .forward(&input.narrow(1, lane * 8, 8)?)?
                            .flatten_all()?
                            .to_vec1::<f32>()?;
                        let shared = shared
                            .narrow(1, lane * 8, 8)?
                            .flatten_all()?
                            .to_vec1::<f32>()?;
                        for (a, b) in shared.iter().zip(&separate) {
                            assert!(
                                a.is_finite() && b.is_finite() && (a - b).abs() < 2e-5,
                                "{dtype:?} width={width} lanes={lanes} lane={lane}: {a} != {b}"
                            );
                        }
                        separate_values.extend(separate);
                    }
                    if lanes == 4 {
                        let Some(dequantized) = reference_weights.as_deref() else {
                            continue;
                        };
                        let reference = f64_qmatmul_reference(
                            dequantized,
                            &input_values[512..(512 + lanes * 8 * 512)],
                            lanes * 8,
                            512,
                        );
                        assert_f64_qmatmul_contract(
                            &shared_values,
                            &reference,
                            &format!("Q8_0 shared width={width} lanes={lanes}"),
                        );
                        assert_f64_qmatmul_contract(
                            &separate_values,
                            &reference,
                            &format!("Q8_0 separate width={width} lanes={lanes}"),
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn pre_m5_profile_keeps_narrow_tensor_batches_and_uses_seven_token_blocks() {
        for name in ["Apple M1 Pro", "Apple M2 Max", "Apple M3", "Apple M4 Ultra"] {
            let profile = MetalTensorProfile::from_device_name(name);
            assert_eq!(profile, MetalTensorProfile::PreM5);
            assert!(profile.uses_tensor_batch(7));
            assert!(profile.uses_tensor_batch(8));
            assert!(!profile.uses_tensor_batch(9));
            assert!(!profile.uses_tensor_batch(128));
            assert_eq!(profile.preferred_speculative_block_size(8), 7);
            assert_eq!(profile.preferred_speculative_block_size(6), 6);
        }

        let m5 = MetalTensorProfile::from_device_name("Apple M5 Max");
        assert_eq!(m5, MetalTensorProfile::Default);
        assert!(m5.uses_tensor_batch(128));
        assert_eq!(m5.preferred_speculative_block_size(8), 8);
    }

    #[test]
    fn pre_m5_scalar_q6k_matches_candle() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        if !supports_metal_tensor(&device)?
            || metal_tensor_profile(&device)? != MetalTensorProfile::PreM5
            || !pre_m5_scalar_q6_r2_enabled()
        {
            return Ok(());
        }

        for (in_dim, out_dim) in [(256, 67), (512, 1024), (768, 1023)] {
            let weights = (0..out_dim * in_dim)
                .map(|index| {
                    let x = index as f32;
                    (x * 0.0137).sin() * 0.2 + (x * 0.0031).cos() * 0.05
                })
                .collect::<Vec<_>>();
            let weights =
                Tensor::from_vec(weights, (out_dim, in_dim), &Device::Cpu)?.to_device(&device)?;
            let packed = Arc::new(QTensor::quantize(&weights, GgmlDType::Q6K)?);
            let candle = CandleQMatMul::from_arc(Arc::clone(&packed))?;
            let optimized = MetalTensorQMatMul::try_from_qtensor(packed, &device)?
                .expect("pre-M5 Metal Q6_K matmul should be available");
            let input = (0..in_dim)
                .map(|index| {
                    let x = index as f32;
                    (x * 0.021).sin() * 0.4 - (x * 0.007).cos() * 0.1
                })
                .collect::<Vec<_>>();
            let input = Tensor::from_vec(input, (1, 1, in_dim), &device)?;

            let expected = candle.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
            let actual = optimized.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
            let mut max_abs = 0f32;
            let mut squared_error = 0f64;
            let mut squared_reference = 0f64;
            for (actual, expected) in actual.iter().zip(&expected) {
                let error = (actual - expected).abs();
                max_abs = max_abs.max(error);
                squared_error += f64::from(error).powi(2);
                squared_reference += f64::from(*expected).powi(2);
            }
            let relative_rmse = (squared_error / squared_reference.max(f64::EPSILON)).sqrt();
            assert!(
                max_abs < 2e-5,
                "Q6_K {out_dim}x{in_dim} maximum error {max_abs} is too high"
            );
            assert!(
                relative_rmse < 1e-6,
                "Q6_K {out_dim}x{in_dim} relative RMSE {relative_rmse} is too high"
            );
        }
        Ok(())
    }

    #[test]
    fn pre_m5_decode_rows_keep_scalar_q6_override() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        if !supports_metal_tensor(&device)? {
            return Ok(());
        }
        let weights = Tensor::from_vec(
            (0..67 * 512)
                .map(|i| (i as f32 * 0.0137).sin() * 0.2)
                .collect::<Vec<_>>(),
            (67, 512),
            &Device::Cpu,
        )?;
        let packed = Arc::new(QTensor::quantize(&weights, GgmlDType::Q6K)?);
        let mut projection = MetalTensorQMatMul::try_from_qtensor(packed, &device)?.unwrap();
        // Exercise the policy and kernel on M5 too; it does not require an
        // older host to check the independent-row fallback's exact math.
        projection.profile = MetalTensorProfile::PreM5;
        let input = Tensor::from_vec(
            (0..5 * 512)
                .map(|i| (i as f32 * 0.021).cos() * 0.4)
                .collect::<Vec<_>>(),
            (1, 5, 512),
            &device,
        )?
        .narrow(1, 1, 4)?;
        for input in [input.clone(), input.squeeze(0)?] {
            let actual = projection.forward_decode_rows(&input)?;
            for row in 0..4 {
                let expected = projection.forward(&input.narrow(D::Minus2, row, 1)?)?;
                let bits = |x: Tensor| -> Result<Vec<u32>> {
                    Ok(x.flatten_all()?
                        .to_vec1::<f32>()?
                        .into_iter()
                        .map(f32::to_bits)
                        .collect())
                };
                assert_eq!(bits(actual.narrow(D::Minus2, row, 1)?)?, bits(expected)?);
            }
        }
        Ok(())
    }

    #[test]
    fn qwen35_q6k_ffn_down_residual_matches_unfused_at_real_geometry() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        if !supports_metal_tensor(&device)?
            || metal_tensor_profile(&device)? != MetalTensorProfile::PreM5
            || !pre_m5_scalar_q6_r2_enabled()
        {
            return Ok(());
        }

        // Exact Qwen3.5-4B FFN down-projection geometry.
        let in_dim = 9216;
        let out_dim = 2560;
        let weights = (0..out_dim * in_dim)
            .map(|index| {
                let row = index / in_dim;
                let column = index % in_dim;
                let mixed = (row * 131 + column * 17) % 4093;
                (mixed as f32 - 2046.0) * (0.000_019 + (row % 7) as f32 * 0.000_001)
            })
            .collect::<Vec<_>>();
        let weights = Tensor::from_vec(weights, (out_dim, in_dim), &Device::Cpu)?;
        let packed = Arc::new(QTensor::quantize(&weights, GgmlDType::Q6K)?);
        let projection = MetalTensorQMatMul::try_from_qtensor(packed, &device)?
            .expect("pre-M5 Metal Q6_K matmul should be available");

        // Narrow the leading dimension so both input bindings exercise a
        // non-zero contiguous buffer offset.
        let input = (0..2 * in_dim)
            .map(|index| {
                let x = index as f32;
                (x * 0.013).sin() * 0.31 - (x * 0.0037).cos() * 0.07
            })
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(input, (2, 1, in_dim), &device)?.narrow(0, 1, 1)?;
        let residual = (0..2 * out_dim)
            .map(|index| match index % 11 {
                0 => 1_000.0,
                1 => -1_000.0,
                2 => f32::EPSILON,
                3 => -f32::EPSILON,
                _ => ((index as f32) * 0.017).sin() * 0.9,
            })
            .collect::<Vec<_>>();
        let residual = Tensor::from_vec(residual, (2, 1, out_dim), &device)?.narrow(0, 1, 1)?;

        let projected = projection.forward(&input)?;
        let expected = (&projected + &residual)?.flatten_all()?.to_vec1::<f32>()?;
        let actual = projection
            .forward_residual(&input, &residual)?
            .expect("real Qwen3.5 FFN geometry should use the fused residual epilogue")
            .flatten_all()?
            .to_vec1::<f32>()?;
        let max_abs = expected
            .iter()
            .zip(&actual)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0f32, f32::max);
        eprintln!("real Qwen3.5 FFN fused residual max_abs={max_abs:.8}");
        assert!(actual.iter().all(|value| value.is_finite()));
        assert!(
            max_abs < 2e-5,
            "fused Q6_K FFN residual differs by {max_abs}"
        );
        Ok(())
    }

    #[test]
    fn qwen35_q8_ssm_residual_matches_unfused_at_real_geometries() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        if !supports_metal_tensor(&device)?
            || metal_tensor_profile(&device)? != MetalTensorProfile::PreM5
        {
            return Ok(());
        }

        let in_dim = 4096;
        for out_dim in [2560usize, 4096usize] {
            let weights = (0..out_dim * in_dim)
                .map(|index| {
                    let row = index / in_dim;
                    let column = index % in_dim;
                    let mixed = (row * 131 + column * 17) % 4093;
                    (mixed as f32 - 2046.0) * (0.000_019 + (row % 7) as f32 * 0.000_001)
                })
                .collect::<Vec<_>>();
            let weights = Tensor::from_vec(weights, (out_dim, in_dim), &Device::Cpu)?;
            let packed = Arc::new(QTensor::quantize(&weights, GgmlDType::Q8_0)?);
            let projection = MetalTensorQMatMul::try_from_qtensor(packed, &device)?
                .expect("pre-M5 Metal Q8_0 matmul should be available");

            for rank3 in [false, true] {
                let input = (0..2 * in_dim)
                    .map(|index| {
                        let x = index as f32;
                        (x * 0.013).sin() * 0.31 - (x * 0.0037).cos() * 0.07
                    })
                    .collect::<Vec<_>>();
                let input = Tensor::from_vec(input, (2, 1, in_dim), &device)?.narrow(0, 1, 1)?;
                let residual = (0..2 * out_dim)
                    .map(|index| match index % 11 {
                        0 => 1_000.0,
                        1 => -1_000.0,
                        2 => f32::EPSILON,
                        3 => -f32::EPSILON,
                        _ => ((index as f32) * 0.017).sin() * 0.9,
                    })
                    .collect::<Vec<_>>();
                let residual =
                    Tensor::from_vec(residual, (2, 1, out_dim), &device)?.narrow(0, 1, 1)?;
                let input = if rank3 { input } else { input.squeeze(0)? };
                let residual = if rank3 {
                    residual
                } else {
                    residual.squeeze(0)?
                };

                let norm_weight = Tensor::from_vec(
                    (0..out_dim)
                        .map(|index| 0.75 + (index % 17) as f32 * 0.015)
                        .collect::<Vec<_>>(),
                    out_dim,
                    &device,
                )?;
                let projected = projection.forward(&input)?;
                let (expected_residual, expected_norm) =
                    crate::architectures::fused_norm_metal::try_fused_add_rms_norm(
                        &residual,
                        &projected,
                        &norm_weight,
                        1e-6,
                    )?
                    .expect("real Qwen3.5 SSM shape should use fused add/RMSNorm");
                let actual_residual = projection
                    .forward_qwen35_ssm_residual(&input, &residual)?
                    .expect("real Qwen3.5 SSM geometry should use the fused residual epilogue");
                let actual_norm = crate::architectures::fused_norm_metal::try_fused_rms_norm(
                    &actual_residual,
                    &norm_weight,
                    1e-6,
                )?
                .expect("real Qwen3.5 SSM shape should use fused RMSNorm");
                let expected = expected_residual.flatten_all()?.to_vec1::<f32>()?;
                let actual = actual_residual.flatten_all()?.to_vec1::<f32>()?;
                let max_abs = expected
                    .iter()
                    .zip(&actual)
                    .map(|(expected, actual)| (expected - actual).abs())
                    .fold(0f32, f32::max);
                eprintln!(
                    "real Qwen3.5 Q8 SSM fused residual {out_dim} rank3={rank3} max_abs={max_abs:.8}"
                );
                assert!(actual.iter().all(|value| value.is_finite()));
                assert!(
                    max_abs < 2e-5,
                    "fused Q8_0 SSM residual differs by {max_abs}"
                );
                let expected_norm = expected_norm.flatten_all()?.to_vec1::<f32>()?;
                let actual_norm = actual_norm.flatten_all()?.to_vec1::<f32>()?;
                let norm_max_abs = expected_norm
                    .iter()
                    .zip(&actual_norm)
                    .map(|(expected, actual)| (expected - actual).abs())
                    .fold(0f32, f32::max);
                assert!(
                    norm_max_abs < 2e-5,
                    "fused Q8_0 SSM residual/RMSNorm differs by {norm_max_abs}"
                );
            }

            let multi_token = Tensor::zeros((1, 2, in_dim), DType::F32, &device)?;
            let multi_residual = Tensor::zeros((1, 2, out_dim), DType::F32, &device)?;
            assert!(
                projection
                    .forward_qwen35_ssm_residual(&multi_token, &multi_residual)?
                    .is_none(),
                "fused Q8_0 SSM residual accepted multi-token input"
            );
        }
        Ok(())
    }

    fn compare_qwen35_single_token_groups(
        device: &Device,
        dtype: GgmlDType,
        in_dim: usize,
        group_widths: &[usize],
    ) -> Result<()> {
        let mut row_base = 0usize;
        let mut packed = Vec::with_capacity(group_widths.len());
        for &out_dim in group_widths {
            let weights = (0..out_dim * in_dim)
                .map(|index| {
                    let row = row_base + index / in_dim;
                    let column = index % in_dim;
                    let mixed = (row * 131 + column * 17) % 4093;
                    (mixed as f32 - 2046.0) * (0.000_019 + (row % 7) as f32 * 0.000_001)
                })
                .collect::<Vec<_>>();
            let weights = Tensor::from_vec(weights, (out_dim, in_dim), &Device::Cpu)?;
            packed.push(Arc::new(QTensor::quantize(&weights, dtype)?));
            row_base += out_dim;
        }
        let mut separate = packed
            .iter()
            .map(|weights| MetalTensorQMatMul::try_from_qtensor(Arc::clone(weights), device))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .map(|projection| projection.expect("single projection should be available"))
            .collect::<Vec<_>>();
        let mut grouped = MetalTensorQMatMul::try_from_qtensors(&packed, device)?
            .expect("grouped projection should be available");

        let input = (0..2 * in_dim)
            .map(|index| {
                let x = index as f32;
                (x * 0.013).sin() * 0.31 - (x * 0.0037).cos() * 0.07
            })
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(input, (2, 1, in_dim), device)?.narrow(0, 1, 1)?;
        let compare = |profile: MetalTensorProfile,
                       separate: &[MetalTensorQMatMul],
                       grouped: &MetalTensorQMatMul|
         -> Result<()> {
            let expected = separate
                .iter()
                .map(|projection| projection.forward(&input))
                .collect::<Result<Vec<_>>>()?;
            let actual = grouped
                .forward_single_token_groups(&input)?
                .expect("single-token grouped output should be available");
            assert_eq!(actual.len(), expected.len());
            for (group, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(actual.shape(), expected.shape());
                let actual = actual.flatten_all()?.to_vec1::<f32>()?;
                let expected = expected.flatten_all()?.to_vec1::<f32>()?;
                let max_abs = actual
                    .iter()
                    .zip(&expected)
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0f32, f32::max);
                eprintln!(
                    "{dtype:?} {} grouped projection {group} max_abs={max_abs:.8}",
                    profile.label()
                );
                assert!(actual.iter().all(|value| value.is_finite()));
                assert!(
                    max_abs < 2e-5,
                    "{dtype:?} {} grouped projection {group} differs by {max_abs}",
                    profile.label()
                );
            }
            Ok(())
        };

        let native_profile = grouped.profile;
        compare(native_profile, &separate, &grouped)?;
        // Pre-M5 and Default select different tile/dispatch policies. Force
        // the latter on older hosts so exact production geometry exercises
        // both supported profiles without allocating a second weight set.
        if native_profile != MetalTensorProfile::Default {
            for projection in &mut separate {
                projection.profile = MetalTensorProfile::Default;
            }
            grouped.profile = MetalTensorProfile::Default;
            compare(MetalTensorProfile::Default, &separate, &grouped)?;
        }
        Ok(())
    }

    #[test]
    fn qwen35_homogeneous_grouped_decode_matches_separate_at_real_geometry() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        if !supports_metal_tensor(&device)? {
            return Ok(());
        }

        // DeltaNet QKV/gate, DeltaNet alpha/beta, and attention Q/K/V.
        compare_qwen35_single_token_groups(&device, GgmlDType::Q6K, 2560, &[8192, 4096])?;
        compare_qwen35_single_token_groups(&device, GgmlDType::Q8_0, 2560, &[32, 32])?;
        compare_qwen35_single_token_groups(&device, GgmlDType::Q6K, 2560, &[8192, 1024, 1024])
    }

    fn compare_with_candle(dtype: GgmlDType, tokens: usize, out_dim: usize) -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        if !supports_metal_tensor(&device)? {
            return Ok(());
        }

        let in_dim = 512;
        let weights = (0..out_dim * in_dim)
            .map(|index| {
                let x = index as f32;
                (x * 0.0137).sin() * 0.2 + (x * 0.0031).cos() * 0.05
            })
            .collect::<Vec<_>>();
        let weights =
            Tensor::from_vec(weights, (out_dim, in_dim), &Device::Cpu)?.to_device(&device)?;
        let packed = Arc::new(QTensor::quantize(&weights, dtype)?);
        let candle = CandleQMatMul::from_arc(Arc::clone(&packed))?;
        let tensor = MetalTensorQMatMul::try_from_qtensor(packed, &device)?
            .expect("Metal 4 tensor matmul should support this dtype");

        let input = (0..tokens * in_dim)
            .map(|index| {
                let x = index as f32;
                (x * 0.021).sin() * 0.4 - (x * 0.007).cos() * 0.1
            })
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(input, (1, tokens, in_dim), &device)?;
        let expected = candle.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
        let actual = tensor.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;

        let mut max_abs = 0f32;
        let mut squared_error = 0f64;
        let mut squared_reference = 0f64;
        for (expected, actual) in expected.iter().zip(&actual) {
            let error = (expected - actual).abs();
            max_abs = max_abs.max(error);
            squared_error += f64::from(error).powi(2);
            squared_reference += f64::from(*expected).powi(2);
        }
        let relative_rmse = (squared_error / squared_reference.max(f64::EPSILON)).sqrt();
        eprintln!(
            "{dtype:?} tokens={tokens} out={out_dim}: max_abs={max_abs:.6}, relative_rmse={relative_rmse:.6}"
        );
        assert!(actual.iter().all(|value| value.is_finite()));
        assert!(
            relative_rmse < 0.005,
            "{dtype:?} tensor matmul relative RMSE {relative_rmse} is too high"
        );
        Ok(())
    }

    fn compare_fused_relu_squared(tokens: usize) -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        if !supports_metal_tensor(&device)? {
            return Ok(());
        }

        let in_dim = 512;
        let out_dim = 70;
        let weights = (0..out_dim * in_dim)
            .map(|index| {
                let x = index as f32;
                (x * 0.0137).sin() * 0.2 + (x * 0.0031).cos() * 0.05
            })
            .collect::<Vec<_>>();
        let weights =
            Tensor::from_vec(weights, (out_dim, in_dim), &Device::Cpu)?.to_device(&device)?;
        let packed = Arc::new(QTensor::quantize(&weights, GgmlDType::Q5_1)?);
        let tensor = MetalTensorQMatMul::try_from_qtensor(packed, &device)?
            .expect("Metal 4 tensor matmul should support Q5_1");
        let input = (0..tokens * in_dim)
            .map(|index| {
                let x = index as f32;
                (x * 0.021).sin() * 0.4 - (x * 0.007).cos() * 0.1
            })
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(input, (1, tokens, in_dim), &device)?;

        let output = tensor.forward(&input)?;
        let relu = output.relu()?;
        let expected = (&relu * &relu)?.flatten_all()?.to_vec1::<f32>()?;
        let actual = tensor
            .forward_relu_squared(&input)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let max_abs = expected
            .iter()
            .zip(&actual)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0f32, f32::max);
        eprintln!("Q5_1 fused ReLU-squared tokens={tokens}: max_abs={max_abs:.8}");
        assert!(actual.iter().all(|value| value.is_finite()));
        assert!(max_abs < 1e-5, "fused ReLU-squared differs by {max_abs}");
        Ok(())
    }

    fn compare_grouped_q8_with_separate(tokens: usize) -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        if !supports_metal_tensor(&device)? {
            return Ok(());
        }

        let in_dim = 512;
        let out_dims = [512, 256, 32, 32];
        let mut packed = Vec::with_capacity(out_dims.len());
        let mut expected = Vec::with_capacity(out_dims.len());
        let mut row_base = 0usize;
        for out_dim in out_dims {
            let weights = (0..out_dim * in_dim)
                .map(|index| {
                    let x = (row_base * in_dim + index) as f32;
                    (x * 0.0137).sin() * 0.2 + (x * 0.0031).cos() * 0.05
                })
                .collect::<Vec<_>>();
            let weights =
                Tensor::from_vec(weights, (out_dim, in_dim), &Device::Cpu)?.to_device(&device)?;
            let tensor = Arc::new(QTensor::quantize(&weights, GgmlDType::Q8_0)?);
            packed.push(Arc::clone(&tensor));
            expected.push(CandleQMatMul::from_arc(tensor)?);
            row_base += out_dim;
        }

        let input = (0..tokens * in_dim)
            .map(|index| {
                let x = index as f32;
                (x * 0.021).sin() * 0.4 - (x * 0.007).cos() * 0.1
            })
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(input, (1, tokens, in_dim), &device)?;
        let expected = expected
            .iter()
            .map(|projection| projection.forward(&input))
            .collect::<Result<Vec<_>>>()?;
        let expected_refs = expected.iter().collect::<Vec<_>>();
        let expected = Tensor::cat(&expected_refs, 2)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let grouped = MetalTensorQMatMul::try_from_qtensors(&packed, &device)?
            .expect("Metal grouped tensor matmul should support Q8_0");
        let actual = grouped
            .forward_groups(&input)?
            .expect("Metal grouped output should support an eight-token Q8_0 input");
        let actual_refs = actual.iter().collect::<Vec<_>>();
        let actual = Tensor::cat(&actual_refs, 2)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let mut squared_error = 0f64;
        let mut squared_reference = 0f64;
        for (expected, actual) in expected.iter().zip(&actual) {
            squared_error += f64::from((expected - actual).abs()).powi(2);
            squared_reference += f64::from(*expected).powi(2);
        }
        let relative_rmse = (squared_error / squared_reference.max(f64::EPSILON)).sqrt();
        eprintln!("grouped Q8_0 tokens={tokens}: relative_rmse={relative_rmse:.6}");
        assert!(actual.iter().all(|value| value.is_finite()));
        assert!(
            relative_rmse < 0.005,
            "grouped Q8_0 relative RMSE {relative_rmse} is too high"
        );
        Ok(())
    }
    #[test]
    fn metal4_q4_0_matches_candle_with_partial_tiles() -> Result<()> {
        compare_with_candle(GgmlDType::Q4_0, 8, 70)?;
        compare_with_candle(GgmlDType::Q4_0, 17, 70)?;
        compare_with_candle(GgmlDType::Q4_0, 33, 70)?;
        compare_with_candle(GgmlDType::Q4_0, 129, 70)
    }

    #[test]
    fn metal4_q5_1_matches_candle_with_partial_tiles() -> Result<()> {
        compare_with_candle(GgmlDType::Q5_1, 8, 70)?;
        compare_with_candle(GgmlDType::Q5_1, 17, 70)?;
        compare_with_candle(GgmlDType::Q5_1, 33, 70)?;
        compare_with_candle(GgmlDType::Q5_1, 129, 70)
    }

    #[test]
    fn metal4_q8_0_matches_candle_with_partial_tiles() -> Result<()> {
        compare_with_candle(GgmlDType::Q8_0, 8, Q8_KSPLIT_MIN_OUT)?;
        compare_with_candle(GgmlDType::Q8_0, 8, 70)?;
        compare_with_candle(GgmlDType::Q8_0, 17, 70)?;
        compare_with_candle(GgmlDType::Q8_0, 33, 70)?;
        compare_with_candle(GgmlDType::Q8_0, 129, 70)
    }

    #[test]
    fn metal4_grouped_q8_0_matches_separate_projections() -> Result<()> {
        compare_grouped_q8_with_separate(8)
    }

    #[test]
    fn metal4_q4k_matches_candle_with_partial_tiles() -> Result<()> {
        compare_with_candle(GgmlDType::Q4K, 3, DEFAULT_Q4K_VERIFY_MIN_OUT)?;
        compare_with_candle(GgmlDType::Q4K, 5, DEFAULT_Q4K_VERIFY_MIN_OUT)?;
        compare_with_candle(GgmlDType::Q4K, 8, DEFAULT_Q4K_VERIFY_MIN_OUT)?;
        compare_with_candle(GgmlDType::Q4K, 7, 70)?;
        compare_with_candle(GgmlDType::Q4K, 8, 70)?;
        compare_with_candle(GgmlDType::Q4K, 16, 70)?;
        compare_with_candle(GgmlDType::Q4K, 17, 70)?;
        compare_with_candle(GgmlDType::Q4K, 32, 70)?;
        compare_with_candle(GgmlDType::Q4K, 33, 70)?;
        compare_with_candle(GgmlDType::Q4K, 64, 70)?;
        compare_with_candle(GgmlDType::Q4K, 129, 70)
    }

    #[test]
    fn metal4_q5k_matches_candle_with_partial_tiles() -> Result<()> {
        compare_with_candle(GgmlDType::Q5K, 8, 70)?;
        compare_with_candle(GgmlDType::Q5K, 17, 70)?;
        compare_with_candle(GgmlDType::Q5K, 33, 70)?;
        compare_with_candle(GgmlDType::Q5K, 129, 70)
    }

    #[test]
    fn metal4_q6k_matches_candle_with_partial_tiles() -> Result<()> {
        compare_with_candle(GgmlDType::Q6K, 8, Q6_KSPLIT_MIN_OUT)?;
        compare_with_candle(GgmlDType::Q6K, 8, 70)?;
        compare_with_candle(GgmlDType::Q6K, 17, 70)?;
        compare_with_candle(GgmlDType::Q6K, 33, 70)?;
        compare_with_candle(GgmlDType::Q6K, 129, 70)
    }

    #[test]
    fn metal4_q5_1_fused_relu_squared_matches_unfused() -> Result<()> {
        compare_fused_relu_squared(1)?;
        compare_fused_relu_squared(8)?;
        compare_fused_relu_squared(17)?;
        compare_fused_relu_squared(33)?;
        compare_fused_relu_squared(129)
    }

    #[test]
    fn raw_buffer_legacy_paths_match_candle() -> Result<()> {
        compare_with_candle(GgmlDType::Q4_0, 1, 70)?;
        compare_with_candle(GgmlDType::Q5_1, 1, 70)?;
        compare_with_candle(GgmlDType::Q8_0, MIN_TENSOR_TOKENS - 1, 70)?;
        compare_with_candle(GgmlDType::Q4K, 1, 70)?;
        compare_with_candle(GgmlDType::Q6K, MIN_TENSOR_TOKENS - 1, 70)
    }
}
