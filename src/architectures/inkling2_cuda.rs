//! Fused CUDA kernels for Inkling 2.
//!
//! The CUDA counterpart of `inkling2_metal.rs`: each kernel replaces a run of
//! small tensor ops in `inkling2_loop.rs`, whose plain versions stay the CPU
//! path and the reference for the tests there. The contracts match the Metal
//! ones, plus an RMS norm and the sigmoid top-k router, which Metal takes
//! from shared kernels. All tensors are contiguous F32 unless noted.

use std::sync::OnceLock;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use candle_core::cuda_backend::cudarc::driver::{
    CudaSlice, CudaStream, CudaView, DevicePtr, LaunchConfig, PushKernelArg, SyncOnDrop,
};
use candle_core::cuda_backend::cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use candle_core::{
    CpuStorage, CudaDevice, CudaStorage, CustomOp1, CustomOp2, CustomOp3, DType, Layout, Result,
    Shape, Tensor,
};

const MODULE: &str = "dinf_inkling2_v1";

const SOURCE: &str = r#"
#include <cuda_fp16.h>

#define DINF_NEG_LARGE -1.0e30f
// NVRTC has no math.h; this is the IEEE bit pattern of -infinity.
#define DINF_NEG_INF __int_as_float(0xff800000)

static __device__ __forceinline__ float dinf_warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, mask);
    }
    return value;
}

// Sum over a block of whole warps, returned to every thread. `partials`
// holds one slot per warp and must not be reused by a later reduction.
static __device__ __forceinline__ float dinf_block_sum(float value, float *partials) {
    const int lane = threadIdx.x & 31;
    const int warps = blockDim.x >> 5;
    value = dinf_warp_sum(value);
    if (lane == 0) {
        partials[threadIdx.x >> 5] = value;
    }
    __syncthreads();
    float total = 0.0f;
    for (int warp = 0; warp < warps; ++warp) {
        total += partials[warp];
    }
    return total;
}

// out[t, c] = scale * (h[t + width - 1, c] + sum_j w[c, j] h[t + j, c]
//                      + sum_lag taps[t, head(c), lag] h[t + width - 1 - lag, c]),
// where h is [past (width - 1 rows); x] and the tap sum is skipped when heads == 0.
extern "C" __global__ void inkling_short_conv(
    const float *history, const float *weight, const float *taps, float *out,
    int tokens, int channels, int width, int heads, float scale) {
    const long long gid = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long) tokens * channels) {
        return;
    }
    const int t = (int) (gid / channels);
    const int c = (int) (gid - (long long) t * channels);
    float acc = history[(long long) (t + width - 1) * channels + c];
    for (int j = 0; j < width; ++j) {
        acc += weight[c * width + j] * history[(long long) (t + j) * channels + c];
    }
    if (heads > 0) {
        const int head = c / (channels / heads);
        const float *row = taps + ((long long) t * heads + head) * width;
        for (int lag = 0; lag < width; ++lag) {
            acc += row[lag] * history[(long long) (t + width - 1 - lag) * channels + c];
        }
    }
    out[gid] = acc * scale;
}

// Static taps with the past rows and the new rows in separate buffers. Rows
// 0..tokens of out are the convolution; rows tokens.. are the next state
// (the last width - 1 rows of [past; x]).
extern "C" __global__ void inkling_short_conv_state(
    const float *x, const float *past, const float *weight, float *out,
    int tokens, int channels, int width, float scale) {
    const int history = width - 1;
    const long long gid = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long) (tokens + history) * channels) {
        return;
    }
    const int t = (int) (gid / channels);
    const int c = (int) (gid - (long long) t * channels);
    if (t >= tokens) {
        // History row t of [past; x] becomes state row t - tokens.
        out[gid] = t < history ? past[(long long) t * channels + c]
                               : x[(long long) (t - history) * channels + c];
        return;
    }
    float acc = x[(long long) t * channels + c];
    for (int j = 0; j < width; ++j) {
        const int row = t + j;
        const float h = row < history ? past[(long long) row * channels + c]
                                      : x[(long long) (row - history) * channels + c];
        acc += weight[c * width + j] * h;
    }
    out[gid] = acc * scale;
}

// Row r (token r / heads, head r % heads): v = x * scale[head] (flag 1),
// out = v * min(threshold / (|v| + 1e-6), 1) * token_scale[token] (flag 2)
// * post_scale. One warp per row.
extern "C" __global__ void inkling_scaled_l2_clip(
    const float *x, const float *scale, const float *token_scale, float *out,
    int rows, int dim, int heads, int flags, float threshold, float post_scale) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= rows) {
        return;
    }
    const bool has_scale = (flags & 1) != 0;
    const float *src = x + (long long) row * dim;
    const float *s = scale + (long long) (row % heads) * dim;
    float sumsq = 0.0f;
    for (int d = lane; d < dim; d += 32) {
        const float v = has_scale ? src[d] * s[d] : src[d];
        sumsq += v * v;
    }
    sumsq = dinf_warp_sum(sumsq);
    float factor = fminf(threshold / (sqrtf(sumsq) + 1.0e-6f), 1.0f);
    if ((flags & 2) != 0) {
        factor *= token_scale[row / heads];
    }
    factor *= post_scale;
    for (int d = lane; d < dim; d += 32) {
        const float v = has_scale ? src[d] * s[d] : src[d];
        out[(long long) row * dim + d] = v * factor;
    }
}

// out[r] = x[r] * rsqrt(mean(x[r]^2) + eps) (* weight when has_weight), one
// block per row.
extern "C" __global__ void inkling_rms_norm(
    const float *x, const float *weight, float *out, int hidden, int has_weight, float eps) {
    __shared__ float partials[32];
    const float *row = x + (long long) blockIdx.x * hidden;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        sum += row[i] * row[i];
    }
    const float inv = rsqrtf(dinf_block_sum(sum, partials) / (float) hidden + eps);
    float *o = out + (long long) blockIdx.x * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float value = row[i] * inv;
        o[i] = has_weight ? value * weight[i] : value;
    }
}

// out[r] = [rms(x[r]) | rms(m[r])], one block per row.
extern "C" __global__ void inkling_rms_pair(
    const float *x, const float *m, float *out, int hidden, float eps) {
    __shared__ float partials_x[32];
    __shared__ float partials_m[32];
    const float *xr = x + (long long) blockIdx.x * hidden;
    const float *mr = m + (long long) blockIdx.x * hidden;
    float sx = 0.0f;
    float sm = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        sx += xr[i] * xr[i];
        sm += mr[i] * mr[i];
    }
    const float ix = rsqrtf(dinf_block_sum(sx, partials_x) / (float) hidden + eps);
    const float im = rsqrtf(dinf_block_sum(sm, partials_m) / (float) hidden + eps);
    float *o = out + (long long) blockIdx.x * 2 * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        o[i] = xr[i] * ix;
        o[hidden + i] = mr[i] * im;
    }
}

// out = (1 - tanh(f)) * x + tanh(g) * m, where each fg row is [f | g].
extern "C" __global__ void inkling_gate_combine(
    const float *fg, const float *x, const float *m, float *out, int rows, int hidden) {
    const long long gid = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long) rows * hidden) {
        return;
    }
    const long long row = gid / hidden;
    const long long i = gid - row * hidden;
    const float forget = tanhf(fg[row * 2 * hidden + i]);
    const float gain = tanhf(fg[row * 2 * hidden + hidden + i]);
    out[gid] = (1.0f - forget) * x[gid] + gain * m[gid];
}

// out = beta tanh(g / beta) sigmoid(g) * linear_beta tanh(u / linear_beta),
// where each gate_up row is [g | u].
extern "C" __global__ void inkling_situ(
    const float *gate_up, float *out, int rows, int ffn, float beta, float linear_beta) {
    const long long gid = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long) rows * ffn) {
        return;
    }
    const long long row = gid / ffn;
    const long long i = gid - row * ffn;
    const float g = gate_up[row * 2 * ffn + i];
    const float u = gate_up[row * 2 * ffn + ffn + i];
    const float gate = beta * tanhf(g / beta) / (1.0f + expf(-g));
    out[gid] = gate * linear_beta * tanhf(u / linear_beta);
}

// sum_k w[k] x[k] for one BF16 row `w` of `cols` (a multiple of 8) by one
// warp, eight weights (16 bytes) per lane step; returned to every lane.
static __device__ __forceinline__ float dinf_bf16_dot(
    const float *x, const unsigned int *w, int cols, int lane) {
    const uint4 *weights = reinterpret_cast<const uint4 *>(w);
    const float4 *xv = reinterpret_cast<const float4 *>(x);
    float sum = 0.0f;
#pragma unroll 3
    for (int c = lane; c < cols / 8; c += 32) {
        const uint4 p = weights[c];
        const float4 a = xv[2 * c];
        const float4 b = xv[2 * c + 1];
        sum = fmaf(__uint_as_float(p.x << 16), a.x, sum);
        sum = fmaf(__uint_as_float(p.x & 0xffff0000u), a.y, sum);
        sum = fmaf(__uint_as_float(p.y << 16), a.z, sum);
        sum = fmaf(__uint_as_float(p.y & 0xffff0000u), a.w, sum);
        sum = fmaf(__uint_as_float(p.z << 16), b.x, sum);
        sum = fmaf(__uint_as_float(p.z & 0xffff0000u), b.y, sum);
        sum = fmaf(__uint_as_float(p.w << 16), b.z, sum);
        sum = fmaf(__uint_as_float(p.w & 0xffff0000u), b.w, sum);
    }
    return dinf_warp_sum(sum);
}

// out[r] = sum_k w[r, k] x[k] for BF16 rows `w` [rows, cols]: one warp per
// row. cols is a multiple of 8.
extern "C" __global__ void inkling_bf16_matvec(
    const float *x, const unsigned int *w, float *out, int rows, int cols) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= rows) {
        return;
    }
    const float sum = dinf_bf16_dot(x, w + (long long) row * (cols / 2), cols, lane);
    if (lane == 0) {
        out[row] = sum;
    }
}

// Up to three products o_j = W_j x_j of one width in one launch, one warp
// per row of the stacked row ranges (segment j has rows_j rows; unused
// segments have none).
extern "C" __global__ void inkling_bf16_matvec3(
    const float *x0, const unsigned int *w0, float *o0, int rows0,
    const float *x1, const unsigned int *w1, float *o1, int rows1,
    const float *x2, const unsigned int *w2, float *o2, int rows2, int cols) {
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const float *x = x0;
    const unsigned int *w = w0;
    float *o = o0;
    if (row >= rows0) {
        row -= rows0;
        x = x1;
        w = w1;
        o = o1;
        if (row >= rows1) {
            row -= rows1;
            x = x2;
            w = w2;
            o = o2;
            if (row >= rows2) {
                return;
            }
        }
    }
    const float sum = dinf_bf16_dot(x, w + (long long) row * (cols / 2), cols, lane);
    if (lane == 0) {
        o[row] = sum;
    }
}

#define DINF_ROUTE_SLOTS 8
#define DINF_ROUTE_MAX_TOP_K 32

// For one token's logits, by one warp: sigmoid scores, the top_k experts by
// score + bias (ties to the lower expert), and their unbiased scores
// normalized and times `scale`, highest selection first. At most 32 *
// DINF_ROUTE_SLOTS experts; every slot index is a compile-time constant, so
// the scores stay in registers. Logits are read through L2 (other blocks
// may have just written them).
static __device__ __forceinline__ void dinf_top_k_token(
    const float *logits, const float *bias, unsigned int *ids, float *weights,
    int experts, int top_k, float scale, int lane) {
    float score[DINF_ROUTE_SLOTS];
    float choice[DINF_ROUTE_SLOTS];
#pragma unroll
    for (int i = 0; i < DINF_ROUTE_SLOTS; ++i) {
        const int expert = lane + 32 * i;
        if (expert < experts) {
            score[i] = 1.0f / (1.0f + expf(-__ldcg(logits + expert)));
            choice[i] = score[i] + bias[expert];
        } else {
            score[i] = 0.0f;
            choice[i] = DINF_NEG_INF;
        }
    }
    float picked_score = 0.0f;
    unsigned int picked_expert = 0;
    float total = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        // This lane's best: slots ascend by expert, so `>` keeps the lower on ties.
        float best = DINF_NEG_INF;
        int best_expert = 0x7fffffff;
        float best_score = 0.0f;
#pragma unroll
        for (int i = 0; i < DINF_ROUTE_SLOTS; ++i) {
            if (choice[i] > best) {
                best = choice[i];
                best_expert = lane + 32 * i;
                best_score = score[i];
            }
        }
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask);
            const int other_expert = __shfl_xor_sync(0xffffffffu, best_expert, mask);
            const float other_score = __shfl_xor_sync(0xffffffffu, best_score, mask);
            if (other > best || (other == best && other_expert < best_expert)) {
                best = other;
                best_expert = other_expert;
                best_score = other_score;
            }
        }
#pragma unroll
        for (int i = 0; i < DINF_ROUTE_SLOTS; ++i) {
            if (lane + 32 * i == best_expert) {
                choice[i] = DINF_NEG_INF;
            }
        }
        total += best_score;
        if (lane == k) {
            picked_score = best_score;
            picked_expert = (unsigned int) best_expert;
        }
    }
    if (lane < top_k) {
        ids[lane] = picked_expert;
        weights[lane] = picked_score / (total + 1e-20f) * scale;
    }
}

// One warp per token: see dinf_top_k_token. ids are U32 [tokens, top_k].
extern "C" __global__ void inkling_sigmoid_top_k(
    const float *logits, const float *bias, unsigned int *ids, float *weights,
    int tokens, int experts, int top_k, float scale) {
    const int token = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (token >= tokens) {
        return;
    }
    dinf_top_k_token(logits + (long long) token * experts, bias, ids + (long long) token * top_k,
                     weights + (long long) token * top_k, experts, top_k, scale, lane);
}

#define DINF_ROUTE_WARPS 8

// The router for a few tokens in one launch: one warp per expert computes
// its F32 logit for every token (x [tokens][hidden], router [experts][hidden],
// hidden a multiple of 4, rows 16-byte aligned) into `logits`; the last
// block to finish (by the counter at `counter`, which it resets) then picks
// each token's routes as inkling_sigmoid_top_k does, one warp per token
// (tokens <= DINF_ROUTE_WARPS).
extern "C" __global__ void inkling_route(
    const float *x, const float *router, const float *bias, float *logits,
    unsigned int *ids, float *weights, unsigned int *counter,
    int tokens, int experts, int hidden, int top_k, float scale) {
    __shared__ int last;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int expert = blockIdx.x * DINF_ROUTE_WARPS + warp;
    if (expert < experts) {
        const float4 *w = reinterpret_cast<const float4 *>(router + (long long) expert * hidden);
        const int quads = hidden / 4;
        for (int t = 0; t < tokens; ++t) {
            const float4 *xr = reinterpret_cast<const float4 *>(x + (long long) t * hidden);
            float sum = 0.0f;
#pragma unroll 6
            for (int i = lane; i < quads; i += 32) {
                const float4 a = xr[i];
                const float4 b = w[i];
                sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
            sum = dinf_warp_sum(sum);
            if (lane == 0) {
                logits[(long long) t * experts + expert] = sum;
            }
        }
    }
    // Publish this block's logits before it counts itself done.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned int done = atomicAdd(counter, 1u);
        last = done == gridDim.x - 1;
    }
    __syncthreads();
    if (!last) {
        return;
    }
    __threadfence();
    if (warp < tokens) {
        dinf_top_k_token(logits + (long long) warp * experts, bias, ids + (long long) warp * top_k,
                         weights + (long long) warp * top_k, experts, top_k, scale, lane);
    }
    if (threadIdx.x == 0) {
        *counter = 0u;
    }
}

#define DINF_ATTN_WARPS 8
// Keys each warp scores per step; their loads and reductions overlap.
#define DINF_ATTN_KEYS 4

// P f16 values from 2P-byte-aligned `src` as floats.
template <int P>
static __device__ __forceinline__ void dinf_load_halves(const __half *src, float *dst) {
    if constexpr (P == 8) {
        const uint4 raw = *reinterpret_cast<const uint4 *>(src);
        const __half2 *h = reinterpret_cast<const __half2 *>(&raw);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(h[i]);
            dst[2 * i] = f.x;
            dst[2 * i + 1] = f.y;
        }
    } else if constexpr (P == 4) {
        const uint2 raw = *reinterpret_cast<const uint2 *>(src);
        const __half2 *h = reinterpret_cast<const __half2 *>(&raw);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const float2 f = __half22float2(h[i]);
            dst[2 * i] = f.x;
            dst[2 * i + 1] = f.y;
        }
    } else {
        const unsigned int raw = *reinterpret_cast<const unsigned int *>(src);
        const float2 f = __half22float2(*reinterpret_cast<const __half2 *>(&raw));
        dst[0] = f.x;
        dst[1] = f.y;
    }
}

// Normalized `acc / total` times the output gate's sigmoid, for the P
// dimensions of lane `lane` of row (token, head).
template <int P>
static __device__ __forceinline__ void dinf_attention_store(
    const float *acc, float total, const float *gate, float *out,
    int token, int head, int heads, int gate_stride, int lane) {
    constexpr int D = 32 * P;
    float *o = out + ((long long) token * heads + head) * D + lane * P;
    const float *g = gate != nullptr ? gate + (long long) token * gate_stride + head * D + lane * P : nullptr;
#pragma unroll
    for (int i = 0; i < P; ++i) {
        float value = acc[i] / total;
        if (g != nullptr) {
            value /= 1.0f + expf(-g[i]);
        }
        o[i] = value;
    }
}

// Pass 1: one block of eight warps per (split, head, token) over keys
// [split * chunk, (split + 1) * chunk); lane l holds dimensions [l P, (l +
// 1) P) of the head (D = 32 P). Queries `q` [tokens][heads][D] are already
// scaled; each `bias` row [tokens][heads] holds the bias by distance (extent
// + 1 columns), then the sink logit. Keys at distance d use bias column
// min(d, extent); keys after the query or, with a window, `window` or more
// back are skipped. Writes the split's running max, sum and unnormalized
// output: [max, sum, acc (D)]; or, with one split, the output itself (sink
// and gate included), as the combine pass would.
template <int P>
static __device__ __forceinline__ void dinf_attention_partial(
    const float *q, const float *bias, const __half *k, const __half *v, const float *gate,
    float *partial, float *out,
    int heads, int kv_heads, int keys, int extent, int window,
    long long kv_head_stride, long long first_query, long long first_key,
    int splits, int chunk, int gate_stride) {
    constexpr int D = 32 * P;
    constexpr int U = DINF_ATTN_KEYS;
    __shared__ float block_max[DINF_ATTN_WARPS];
    __shared__ float block_sum[DINF_ATTN_WARPS];
    __shared__ float block_acc[DINF_ATTN_WARPS][D];
    const int split = blockIdx.x;
    const int head = blockIdx.y;
    const int token = blockIdx.z;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int kv_head = head / (heads / kv_heads);
    const float *query = q + ((long long) token * heads + head) * D + lane * P;
    const float *table = bias + ((long long) token * heads + head) * (extent + 2);
    const __half *kh = k + kv_head * kv_head_stride + lane * P;
    const __half *vh = v + kv_head * kv_head_stride + lane * P;
    float qv[P];
    float acc[P];
#pragma unroll
    for (int i = 0; i < P; ++i) {
        qv[i] = query[i];
        acc[i] = 0.0f;
    }
    float running_max = DINF_NEG_LARGE;
    float running_sum = 0.0f;
    const long long position = first_query + token;
    const int begin = split * chunk;
    const int end = min(keys, begin + chunk);
    for (int base = begin + warp * U; base < end; base += DINF_ATTN_WARPS * U) {
        float dot[U];
        bool valid[U];
        long long distance[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int s = base + u;
            distance[u] = position - (first_key + s);
            valid[u] = s < end && distance[u] >= 0 && (window <= 0 || distance[u] < window);
            dot[u] = 0.0f;
            if (valid[u]) {
                float kv[P];
                dinf_load_halves<P>(kh + (long long) s * D, kv);
#pragma unroll
                for (int i = 0; i < P; ++i) {
                    dot[u] += qv[i] * kv[i];
                }
            }
        }
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
                dot[u] += __shfl_xor_sync(0xffffffffu, dot[u], mask);
            }
        }
        float next_max = running_max;
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (valid[u]) {
                dot[u] += table[distance[u] < extent ? (int) distance[u] : extent];
                next_max = fmaxf(next_max, dot[u]);
            }
        }
        const float correction = expf(running_max - next_max);
        running_sum *= correction;
#pragma unroll
        for (int i = 0; i < P; ++i) {
            acc[i] *= correction;
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (valid[u]) {
                const float weight = expf(dot[u] - next_max);
                running_sum += weight;
                float value[P];
                dinf_load_halves<P>(vh + (long long) (base + u) * D, value);
#pragma unroll
                for (int i = 0; i < P; ++i) {
                    acc[i] += weight * value[i];
                }
            }
        }
        running_max = next_max;
    }
    if (lane == 0) {
        block_max[warp] = running_max;
        block_sum[warp] = running_sum;
    }
#pragma unroll
    for (int i = 0; i < P; ++i) {
        block_acc[warp][lane * P + i] = acc[i];
    }
    __syncthreads();
    if (warp != 0) {
        return;
    }
    // One split: fold in the sink and finish here.
    float split_max = splits == 1 ? table[extent + 1] : DINF_NEG_LARGE;
#pragma unroll
    for (int w = 0; w < DINF_ATTN_WARPS; ++w) {
        split_max = fmaxf(split_max, block_max[w]);
    }
    float scale[DINF_ATTN_WARPS];
    float split_sum = splits == 1 ? expf(table[extent + 1] - split_max) : 0.0f;
#pragma unroll
    for (int w = 0; w < DINF_ATTN_WARPS; ++w) {
        scale[w] = expf(block_max[w] - split_max);
        split_sum += block_sum[w] * scale[w];
    }
    float merged[P];
#pragma unroll
    for (int i = 0; i < P; ++i) {
        merged[i] = 0.0f;
#pragma unroll
        for (int w = 0; w < DINF_ATTN_WARPS; ++w) {
            merged[i] += block_acc[w][lane * P + i] * scale[w];
        }
    }
    if (splits == 1) {
        dinf_attention_store<P>(merged, split_sum, gate, out, token, head, heads, gate_stride, lane);
        return;
    }
    float *o = partial + (((long long) token * heads + head) * splits + split) * (D + 2);
    if (lane == 0) {
        o[0] = split_max;
        o[1] = split_sum;
    }
#pragma unroll
    for (int i = 0; i < P; ++i) {
        o[2 + lane * P + i] = merged[i];
    }
}

// Pass 2: one warp per (head, token) merges the splits with the sink logit,
// then applies the output gate's sigmoid (when `gate` is not null; its rows
// are `gate_stride` apart). out is [tokens][heads][D].
template <int P>
static __device__ __forceinline__ void dinf_attention_combine(
    const float *bias, const float *partial, const float *gate, float *out,
    int heads, int extent, int splits, int gate_stride) {
    constexpr int D = 32 * P;
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    const int lane = threadIdx.x;
    const int stride = D + 2;
    const float sink = bias[((long long) token * heads + head) * (extent + 2) + extent + 1];
    const float *parts = partial + ((long long) token * heads + head) * splits * stride;
    float total_max = sink;
#pragma unroll 4
    for (int s = 0; s < splits; ++s) {
        total_max = fmaxf(total_max, parts[(long long) s * stride]);
    }
    float total = expf(sink - total_max);
    float acc[P];
#pragma unroll
    for (int i = 0; i < P; ++i) {
        acc[i] = 0.0f;
    }
#pragma unroll 4
    for (int s = 0; s < splits; ++s) {
        const float *part = parts + (long long) s * stride;
        const float scale = expf(part[0] - total_max);
        total += part[1] * scale;
#pragma unroll
        for (int i = 0; i < P; ++i) {
            acc[i] += part[2 + lane * P + i] * scale;
        }
    }
    dinf_attention_store<P>(acc, total, gate, out, token, head, heads, gate_stride, lane);
}

#define DINF_ATTENTION(D, P)                                                                   \
    extern "C" __global__ void __launch_bounds__(256) inkling_attention_partial_##D(           \
        const float *q, const float *bias, const __half *k, const __half *v, const float *gate, \
        float *partial, float *out, int heads, int kv_heads, int keys, int extent, int window, \
        long long kv_head_stride, long long first_query, long long first_key,                 \
        int splits, int chunk, int gate_stride) {                                              \
        dinf_attention_partial<P>(q, bias, k, v, gate, partial, out, heads, kv_heads, keys,    \
                                  extent, window, kv_head_stride, first_query, first_key,      \
                                  splits, chunk, gate_stride);                                 \
    }                                                                                          \
    extern "C" __global__ void inkling_attention_combine_##D(                                  \
        const float *bias, const float *partial, const float *gate, float *out,               \
        int heads, int extent, int splits, int gate_stride) {                                  \
        dinf_attention_combine<P>(bias, partial, gate, out, heads, extent, splits, gate_stride); \
    }
DINF_ATTENTION(64, 2)
DINF_ATTENTION(128, 4)
DINF_ATTENTION(256, 8)

// ---- Single-pass kernels: one launch for a run of per-visit steps ----

// Attention over depth for one token per block. Sources are the token's rows
// of b0..b(S-1) (the blocks, then the current state); keys and values are
// the RMS-normalized sources, queries [Q][hidden]. Compile-time counts keep
// the per-thread sums in registers. Flags: 1 = RMS-normalize each output row
// and multiply by norm row q (2: one shared norm row); 4 = group-major
// output [Q][tokens], else [tokens][Q].
template <int S, int Q>
static __device__ __forceinline__ void dinf_mix_one(
    const float *b0, const float *b1, const float *b2, const float *b3,
    const float *b4, const float *b5, const float *b6, const float *b7,
    const float *queries, const float *norm, float *out,
    int tokens, int hidden, float eps, float scale, float norm_eps, int flags) {
    constexpr int SLOTS = S * (Q + 1);
    __shared__ float sums[32][SLOTS];
    __shared__ float weights[Q][S];
    __shared__ float norm_sums[32][Q];
    __shared__ float norm_inv[Q];
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = blockDim.x >> 5;
    const long long row = (long long) token * hidden;
    const float *bases[8] = {b0, b1, b2, b3, b4, b5, b6, b7};
    const float *src[S];
#pragma unroll
    for (int s = 0; s < S; ++s) {
        src[s] = bases[s] + row;
    }
    // Slot s: source s's sum of squares; slot (q + 1) * S + s: its dot
    // product with query q.
    float acc[SLOTS];
#pragma unroll
    for (int i = 0; i < SLOTS; ++i) {
        acc[i] = 0.0f;
    }
    for (int i = tid; i < hidden; i += blockDim.x) {
        float qv[Q];
#pragma unroll
        for (int q = 0; q < Q; ++q) {
            qv[q] = queries[(long long) q * hidden + i];
        }
#pragma unroll
        for (int s = 0; s < S; ++s) {
            const float v = src[s][i];
            acc[s] += v * v;
#pragma unroll
            for (int q = 0; q < Q; ++q) {
                acc[(q + 1) * S + s] += v * qv[q];
            }
        }
    }
#pragma unroll
    for (int i = 0; i < SLOTS; ++i) {
        const float total = dinf_warp_sum(acc[i]);
        if (lane == 0) {
            sums[warp][i] = total;
        }
    }
    __syncthreads();
    if (tid < Q) {
        const int q = tid;
        float inv[S];
        float score[S];
        float best = DINF_NEG_LARGE;
#pragma unroll
        for (int s = 0; s < S; ++s) {
            float sumsq = 0.0f;
            float dot = 0.0f;
            for (int w = 0; w < warps; ++w) {
                sumsq += sums[w][s];
                dot += sums[w][(q + 1) * S + s];
            }
            inv[s] = rsqrtf(sumsq / (float) hidden + eps);
            score[s] = dot * inv[s] * scale;
            best = fmaxf(best, score[s]);
        }
        float total = 0.0f;
#pragma unroll
        for (int s = 0; s < S; ++s) {
            score[s] = expf(score[s] - best);
            total += score[s];
        }
#pragma unroll
        for (int s = 0; s < S; ++s) {
            weights[q][s] = score[s] / total * inv[s];
        }
    }
    __syncthreads();
    const bool normed = (flags & 1) != 0;
    const bool major = (flags & 4) != 0;
    float squares[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        squares[q] = 0.0f;
    }
    for (int i = tid; i < hidden; i += blockDim.x) {
        float values[S];
#pragma unroll
        for (int s = 0; s < S; ++s) {
            values[s] = src[s][i];
        }
#pragma unroll
        for (int q = 0; q < Q; ++q) {
            float v = 0.0f;
#pragma unroll
            for (int s = 0; s < S; ++s) {
                v += weights[q][s] * values[s];
            }
            const long long out_row = major ? (long long) q * tokens + token : (long long) token * Q + q;
            out[out_row * hidden + i] = v;
            squares[q] += v * v;
        }
    }
    if (!normed) {
        return;
    }
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        const float total = dinf_warp_sum(squares[q]);
        if (lane == 0) {
            norm_sums[warp][q] = total;
        }
    }
    __syncthreads();
    if (tid < Q) {
        float total = 0.0f;
        for (int w = 0; w < warps; ++w) {
            total += norm_sums[w][tid];
        }
        norm_inv[tid] = rsqrtf(total / (float) hidden + norm_eps);
    }
    __syncthreads();
    // Each thread rescales only the elements it wrote above.
    for (int i = tid; i < hidden; i += blockDim.x) {
#pragma unroll
        for (int q = 0; q < Q; ++q) {
            const long long out_row = major ? (long long) q * tokens + token : (long long) token * Q + q;
            const int norm_row = (flags & 2) != 0 ? 0 : q;
            out[out_row * hidden + i] *= norm_inv[q] * norm[(long long) norm_row * hidden + i];
        }
    }
}

#define DINF_MIX_ONE(S, Q)                                                              \
    extern "C" __global__ void inkling_depth_mix_one_##S##_##Q(                         \
        const float *b0, const float *b1, const float *b2, const float *b3,             \
        const float *b4, const float *b5, const float *b6, const float *b7,             \
        const float *queries, const float *norm, float *out,                            \
        int tokens, int hidden, float eps, float scale, float norm_eps, int flags) {    \
        dinf_mix_one<S, Q>(b0, b1, b2, b3, b4, b5, b6, b7, queries, norm, out,          \
                           tokens, hidden, eps, scale, norm_eps, flags);                \
    }
#define DINF_MIX_ONE_QUERIES(S) DINF_MIX_ONE(S, 1) DINF_MIX_ONE(S, 2) DINF_MIX_ONE(S, 3) DINF_MIX_ONE(S, 4)
DINF_MIX_ONE_QUERIES(2)
DINF_MIX_ONE_QUERIES(3)
DINF_MIX_ONE_QUERIES(4)
DINF_MIX_ONE_QUERIES(5)
DINF_MIX_ONE_QUERIES(6)
DINF_MIX_ONE_QUERIES(7)
DINF_MIX_ONE_QUERIES(8)

// Rows come in groups of `groups` per token (row r is group r % groups of
// token r / groups). `x` holds `x_groups` rows per token: one row shared by
// the group, or one per member.
static __device__ __forceinline__ long long dinf_x_row(int row, int groups, int x_groups) {
    return x_groups == groups ? row : (long long) (row / groups) * x_groups;
}

// out[r] = [rms(x[r]) | rms(m[r])] over grouped rows, one block per row.
// Flag 2 = the output is group-major [groups][tokens], else token-major.
extern "C" __global__ void inkling_rms_pair_groups(
    const float *x, const float *m, float *out,
    int rows, int hidden, float eps, int groups, int x_groups, int flags) {
    __shared__ float partials_x[32];
    __shared__ float partials_m[32];
    const int row = blockIdx.x;
    const float *xr = x + dinf_x_row(row, groups, x_groups) * hidden;
    const float *mr = m + (long long) row * hidden;
    float sx = 0.0f;
    float sm = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        sx += xr[i] * xr[i];
        sm += mr[i] * mr[i];
    }
    const float ix = rsqrtf(dinf_block_sum(sx, partials_x) / (float) hidden + eps);
    const float im = rsqrtf(dinf_block_sum(sm, partials_m) / (float) hidden + eps);
    const int tokens = rows / groups;
    const long long out_row = (flags & 2) != 0 ? (long long) (row % groups) * tokens + row / groups : row;
    float *o = out + out_row * 2 * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        o[i] = xr[i] * ix;
        o[hidden + i] = mr[i] * im;
    }
}

// out = rms((1 - tanh(f)) * x + tanh(g) * m) * weight[group % weight_rows]
// for token-major row r, one block per row. fg rows are [f | g], group-major
// [groups][tokens] when flag 1 is set; out is group-major when flag 2 is set.
extern "C" __global__ void inkling_gate_combine_norm(
    const float *fg, const float *x, const float *m, const float *weight, float *out,
    int rows, int hidden, float eps, int groups, int x_groups, int flags, int weight_rows) {
    __shared__ float partials[32];
    const int row = blockIdx.x;
    const int tokens = rows / groups;
    const int group = row % groups;
    const int token = row / groups;
    const long long major_row = (long long) group * tokens + token;
    const float *f = fg + ((flags & 1) != 0 ? major_row : row) * 2 * hidden;
    const float *xr = x + dinf_x_row(row, groups, x_groups) * hidden;
    const float *mr = m + (long long) row * hidden;
    float *o = out + ((flags & 2) != 0 ? major_row : row) * hidden;
    float sumsq = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float value = (1.0f - tanhf(f[i])) * xr[i] + tanhf(f[hidden + i]) * mr[i];
        o[i] = value;
        sumsq += value * value;
    }
    const float inv = rsqrtf(dinf_block_sum(sumsq, partials) / (float) hidden + eps);
    const float *w = weight + (long long) (group % weight_rows) * hidden;
    // Each thread rescales only the elements it wrote above.
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        o[i] *= inv * w[i];
    }
}

// The loop gates' down projections for a few tokens: reduced[g][t][r] =
// down[g][r] . [rms(x) | rms(m)] for the gate's input row x and message m,
// both RMS-normalized with `eps`. One block per (r, g, t) makes a single
// pass over x, m and the weight row for both norms and both dot products;
// the inputs stay in cache across blocks. hidden is a multiple of 4 and
// every row 16-byte aligned.
extern "C" __global__ void inkling_gate_down(
    const float *x, const float *m, const float *down, float *reduced,
    int tokens, int hidden, int rank, float eps, int groups, int x_groups) {
    __shared__ float partials[4][32];
    const int r = blockIdx.x;
    const int g = blockIdx.y;
    const int t = blockIdx.z;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = t * groups + g;
    const float4 *xr = reinterpret_cast<const float4 *>(x + dinf_x_row(row, groups, x_groups) * hidden);
    const float4 *mr = reinterpret_cast<const float4 *>(m + (long long) row * hidden);
    const float4 *w = reinterpret_cast<const float4 *>(down + ((long long) g * rank + r) * 2 * hidden);
    const int quads = hidden / 4;
    float sx = 0.0f;
    float sm = 0.0f;
    float dx = 0.0f;
    float dm = 0.0f;
#pragma unroll 4
    for (int i = threadIdx.x; i < quads; i += blockDim.x) {
        const float4 a = xr[i];
        const float4 b = mr[i];
        const float4 wa = w[i];
        const float4 wb = w[quads + i];
        sx += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
        sm += b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
        dx += wa.x * a.x + wa.y * a.y + wa.z * a.z + wa.w * a.w;
        dm += wb.x * b.x + wb.y * b.y + wb.z * b.z + wb.w * b.w;
    }
    sx = dinf_warp_sum(sx);
    sm = dinf_warp_sum(sm);
    dx = dinf_warp_sum(dx);
    dm = dinf_warp_sum(dm);
    if (lane == 0) {
        partials[0][warp] = sx;
        partials[1][warp] = sm;
        partials[2][warp] = dx;
        partials[3][warp] = dm;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int k = 0; k < (int) (blockDim.x >> 5); ++k) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                total[j] += partials[j][k];
            }
        }
        const float ix = rsqrtf(total[0] / (float) hidden + eps);
        const float im = rsqrtf(total[1] / (float) hidden + eps);
        reduced[((long long) g * tokens + t) * rank + r] = total[2] * ix + total[3] * im;
    }
}

#define DINF_GATE_WARPS 8

// The loop gates' forget/gain projections, combine and output norm for a few
// tokens: fg[j] = sum_r reduced[g][t][r] * forget_gain[g][r][j], value =
// (1 - tanh(f)) * x + tanh(g) * m, then RMS-normalized with `norm_eps` and
// times weight[g % weight_rows]. Block (column tile, g, t) covers 32 columns,
// its warps splitting the rank. Each block adds its sum of squares to
// scratch[row] and the last block of a row (by the counter scratch[64 +
// row]) rescales the whole row, then resets both. Output is group-major
// [groups][tokens][hidden].
extern "C" __global__ void inkling_gate_up_norm(
    const float *reduced, const float *forget_gain, const float *x, const float *m,
    const float *weight, float *out, float *scratch,
    int tokens, int hidden, int rank, float norm_eps, int groups, int x_groups, int weight_rows) {
    __shared__ float partial[DINF_GATE_WARPS][2][32];
    __shared__ int last;
    const int g = blockIdx.y;
    const int t = blockIdx.z;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int j = blockIdx.x * 32 + lane;
    const int row = t * groups + g;
    const float *red = reduced + ((long long) g * tokens + t) * rank;
    const float *fg = forget_gain + (long long) g * rank * 2 * hidden;
    float f = 0.0f;
    float gn = 0.0f;
    if (j < hidden) {
#pragma unroll 8
        for (int r = warp; r < rank; r += DINF_GATE_WARPS) {
            const float c = red[r];
            f += c * fg[(long long) r * 2 * hidden + j];
            gn += c * fg[(long long) r * 2 * hidden + hidden + j];
        }
    }
    partial[warp][0][lane] = f;
    partial[warp][1][lane] = gn;
    __syncthreads();
    const long long out_row = (long long) g * tokens + t;
    float *o = out + out_row * hidden;
    if (warp == 0) {
        float sumsq = 0.0f;
        if (j < hidden) {
            f = 0.0f;
            gn = 0.0f;
#pragma unroll
            for (int w = 0; w < DINF_GATE_WARPS; ++w) {
                f += partial[w][0][lane];
                gn += partial[w][1][lane];
            }
            const float xv = x[dinf_x_row(row, groups, x_groups) * hidden + j];
            const float mv = m[(long long) row * hidden + j];
            const float value = (1.0f - tanhf(f)) * xv + tanhf(gn) * mv;
            o[j] = value;
            sumsq = value * value;
        }
        // Publish this block's outputs before it counts itself done.
        __threadfence();
        sumsq = dinf_warp_sum(sumsq);
        __syncwarp();
        if (lane == 0) {
            atomicAdd(scratch + out_row, sumsq);
            __threadfence();
            const unsigned int done = atomicAdd(reinterpret_cast<unsigned int *>(scratch + 64 + out_row), 1u);
            last = done == gridDim.x - 1;
        }
    }
    __syncthreads();
    if (!last) {
        return;
    }
    __threadfence();
    const float total = __ldcg(scratch + out_row);
    const float inv = rsqrtf(total / (float) hidden + norm_eps);
    const float *w = weight + (long long) (g % weight_rows) * hidden;
#pragma unroll 4
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        o[i] = __ldcg(o + i) * inv * w[i];
    }
    if (threadIdx.x == 0) {
        scratch[out_row] = 0.0f;
        reinterpret_cast<unsigned int *>(scratch)[64 + out_row] = 0u;
    }
}

static __device__ __forceinline__ float dinf_kv_history(
    const float *past, const float *x, int stride, int history, int channels, int row, int c, bool has_past) {
    if (row < history) {
        return has_past ? past[(long long) row * channels + c] : 0.0f;
    }
    return x[(long long) (row - history) * stride + c];
}

// The K and V short convolutions with their head norms, one block of `dim`
// threads (rounded up to whole warps) per (head, token, K or V). K is
// L2-clipped per head, V RMS-normalized per head; both are written
// head-major, to `out` [K|V][heads][tokens][dim] or, with flag 4, as f16
// into the caches [heads][capacity][dim] from `row` on. Blocks with token <
// width - 1 also write that row of the next conv state ([K|V][width -
// 1][channels]). Flags: 1 = per-token dynamic taps, 2 = a past state (else
// zeros).
extern "C" __global__ void inkling_kv_finish(
    const float *k_in, const float *v_in, const float *k_past, const float *v_past,
    const float *k_weight, const float *v_weight, const float *k_taps, const float *v_taps,
    float *out, float *state, __half *k_cache, __half *v_cache,
    int tokens, int heads, int dim, int width, int k_stride, int v_stride,
    int k_taps_stride, int v_taps_stride, float scale, float clip, float eps, int flags,
    int capacity, int row) {
    __shared__ float partials[32];
    const int head = blockIdx.x;
    const int t = blockIdx.y;
    const int which = blockIdx.z;
    const bool is_v = which == 1;
    const float *x = is_v ? v_in : k_in;
    const float *past = is_v ? v_past : k_past;
    const float *weight = is_v ? v_weight : k_weight;
    const float *taps = is_v ? v_taps : k_taps;
    const int stride = is_v ? v_stride : k_stride;
    const int taps_stride = is_v ? v_taps_stride : k_taps_stride;
    const int history = width - 1;
    const int channels = heads * dim;
    const bool has_past = (flags & 2) != 0;
    const int d = threadIdx.x;
    const int c = head * dim + d;
    if (t < history && d < dim) {
        state[((long long) which * history + t) * channels + c] =
            dinf_kv_history(past, x, stride, history, channels, tokens + t, c, has_past);
    }
    // Whole blocks leave together, before the block-wide sum.
    if (t >= tokens) {
        return;
    }
    float acc = 0.0f;
    if (d < dim) {
        acc = dinf_kv_history(past, x, stride, history, channels, t + history, c, has_past);
        for (int j = 0; j < width; ++j) {
            acc += weight[c * width + j] * dinf_kv_history(past, x, stride, history, channels, t + j, c, has_past);
        }
        if ((flags & 1) != 0) {
            const float *row = taps + (long long) t * taps_stride + head * width;
            for (int lag = 0; lag < width; ++lag) {
                acc += row[lag] * dinf_kv_history(past, x, stride, history, channels, t + history - lag, c, has_past);
            }
        }
        acc *= scale;
    }
    const float total = dinf_block_sum(acc * acc, partials);
    const float factor = is_v ? rsqrtf(total / (float) dim + eps)
                              : fminf(clip / (sqrtf(total) + 1.0e-6f), 1.0f);
    if (d >= dim) {
        return;
    }
    if ((flags & 4) != 0) {
        __half *cache = is_v ? v_cache : k_cache;
        cache[((long long) head * capacity + row + t) * dim + d] = __float2half(acc * factor);
    } else {
        out[(((long long) which * heads + head) * tokens + t) * dim + d] = acc * factor;
    }
}

#define DINF_QUERY_PREP_TOKENS 8

// From the stacked query-side projection rows: the scaled, clipped queries
// [tokens][heads][dim] and, per (token, head), the attention bias by distance
// (extent + 1 columns, from the clipped relative coefficients times `basis`
// [rel][extent + 1]) followed by the sink logit. One block of
// DINF_QUERY_PREP_TOKENS warps per (head, block of tokens, slice of bias
// columns): warp j handles token j of the block, and each thread keeps its
// basis column in registers across the block's tokens. Slice 0 also writes
// the queries and sinks. R is the relative width rounded up (16 or 32).
// Flag 1 = times token_scale[token] (the log scale).
template <int R>
static __device__ __forceinline__ void dinf_query_prep(
    const float *x, const float *q_scale, const float *rel_scale, const float *token_scale,
    const float *basis, float *q_out, float *bias,
    int tokens, int heads, int dim, int rel, int extent, int x_stride,
    int q_offset, int rel_offset, int sink_offset,
    float q_clip, float rel_clip, float q_post, int flags) {
    __shared__ float rel_rows[DINF_QUERY_PREP_TOKENS][R];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int head = blockIdx.x;
    const int first = blockIdx.y * DINF_QUERY_PREP_TOKENS;
    const int slice = blockIdx.z;
    const int token = first + warp;
    const bool active = token < tokens;
    const float *row = x + (long long) token * x_stride;
    const float ts = active && (flags & 1) != 0 ? token_scale[token] : 1.0f;
    {
        float r = 0.0f;
        if (active && lane < rel) {
            r = row[rel_offset + head * rel + lane] * rel_scale[head * rel + lane];
        }
        const float norm = sqrtf(dinf_warp_sum(r * r));
        if (lane < R) {
            rel_rows[warp][lane] = r * fminf(rel_clip / (norm + 1.0e-6f), 1.0f) * ts;
        }
    }
    __syncthreads();
    const int e = slice * blockDim.x + tid;
    if (e <= extent) {
        float column[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            column[r] = r < rel ? basis[(long long) r * (extent + 1) + e] : 0.0f;
        }
        for (int j = 0; j < DINF_QUERY_PREP_TOKENS && first + j < tokens; ++j) {
            float a = 0.0f;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                a += rel_rows[j][r] * column[r];
            }
            bias[((long long) (first + j) * heads + head) * (extent + 2) + e] = a;
        }
    }
    if (slice != 0 || !active) {
        return;
    }
    float *b = bias + ((long long) token * heads + head) * (extent + 2);
    if (lane == 0) {
        b[extent + 1] = row[sink_offset + head];
    }
    float sumsq = 0.0f;
    for (int d = lane; d < dim; d += 32) {
        const float v = row[q_offset + head * dim + d] * q_scale[head * dim + d];
        sumsq += v * v;
    }
    const float factor = fminf(q_clip / (sqrtf(dinf_warp_sum(sumsq)) + 1.0e-6f), 1.0f) * ts * q_post;
    for (int d = lane; d < dim; d += 32) {
        const float v = row[q_offset + head * dim + d] * q_scale[head * dim + d];
        q_out[((long long) token * heads + head) * dim + d] = v * factor;
    }
}

#define DINF_QUERY_PREP(R)                                                                     \
    extern "C" __global__ void inkling_query_prep_##R(                                         \
        const float *x, const float *q_scale, const float *rel_scale, const float *token_scale, \
        const float *basis, float *q_out, float *bias,                                         \
        int tokens, int heads, int dim, int rel, int extent, int x_stride,                     \
        int q_offset, int rel_offset, int sink_offset,                                         \
        float q_clip, float rel_clip, float q_post, int flags) {                               \
        dinf_query_prep<R>(x, q_scale, rel_scale, token_scale, basis, q_out, bias,             \
                           tokens, heads, dim, rel, extent, x_stride, q_offset, rel_offset,    \
                           sink_offset, q_clip, rel_clip, q_post, flags);                      \
    }
DINF_QUERY_PREP(16)
DINF_QUERY_PREP(32)

// Input row r of the post convolution: x's row, or with flag 1 the
// route-weighted sum of routed rows [tokens][routes][channels].
static __device__ __forceinline__ float dinf_post_input(
    const float *x, const float *route_weights, int x_stride, int routes, int channels,
    int flags, int r, int c) {
    if ((flags & 1) == 0) {
        return x[(long long) r * x_stride + c];
    }
    float sum = 0.0f;
    for (int k = 0; k < routes; ++k) {
        sum += route_weights[r * routes + k] * x[((long long) r * routes + k) * channels + c];
    }
    return sum;
}

// A branch's residual short convolution with static taps, plus the residual
// stream (flag 2). Writes out [tokens][channels] and the next state
// [width - 1][channels]. Flag 4 = a past state (else zeros).
extern "C" __global__ void inkling_post_conv(
    const float *x, const float *route_weights, const float *past, const float *weight,
    const float *residual, float *out, float *state,
    int tokens, int channels, int width, int x_stride, int routes, float scale, int flags) {
    const int history = width - 1;
    const long long gid = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long) (tokens + history) * channels) {
        return;
    }
    const int t = (int) (gid / channels);
    const int c = (int) (gid - (long long) t * channels);
    // Row `row` of [past; inputs].
#define DINF_POST_HISTORY(row)                                                          \
    ((row) < history ? ((flags & 4) != 0 ? past[(long long) (row) * channels + c] : 0.0f) \
                     : dinf_post_input(x, route_weights, x_stride, routes, channels, flags, (row) - history, c))
    if (t >= tokens) {
        const int r = t - tokens;
        state[(long long) r * channels + c] = DINF_POST_HISTORY(tokens + r);
        return;
    }
    float acc = DINF_POST_HISTORY(t + history);
    for (int j = 0; j < width; ++j) {
        acc += weight[c * width + j] * DINF_POST_HISTORY(t + j);
    }
#undef DINF_POST_HISTORY
    acc *= scale;
    if ((flags & 2) != 0) {
        acc += residual[(long long) t * channels + c];
    }
    out[(long long) t * channels + c] = acc;
}

#define DINF_GEMM_TILE 64
#define DINF_GEMM_K 32

// partial[s][g][t][r] = sum over split s of the inner dimension of
// a[g][t][k] * w[g][r][k], for the loop gates' skinny down projections
// (rank columns) over prompt chunks. a is [groups][tokens][inner], w
// [groups][rank][inner], both F32 with 16-byte aligned rows; inner is a
// multiple of DINF_GEMM_K * splits. Block (token tile, rank tile + group
// tiles, split): 64 x 64 outputs, 4 x 4 per thread.
extern "C" __global__ void __launch_bounds__(256) inkling_gemm_nt_split(
    const float *a, const float *w, float *partial,
    int tokens, int rank, int inner, int groups, int rank_tiles) {
    __shared__ float a_s[DINF_GEMM_K][DINF_GEMM_TILE + 4];
    __shared__ float w_s[DINF_GEMM_K][DINF_GEMM_TILE + 4];
    const int first_token = blockIdx.x * DINF_GEMM_TILE;
    const int g = blockIdx.y / rank_tiles;
    const int first_rank = (blockIdx.y % rank_tiles) * DINF_GEMM_TILE;
    const int split = blockIdx.z;
    const int per_split = inner / gridDim.z;
    const int tx = threadIdx.x & 15;
    const int ty = threadIdx.x >> 4;
    const float *a_g = a + (long long) g * tokens * inner;
    const float *w_g = w + (long long) g * rank * inner;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.0f;
        }
    }
    for (int k0 = split * per_split; k0 < (split + 1) * per_split; k0 += DINF_GEMM_K) {
        // 64 rows of 8 float4s for each operand: two of each per thread.
#pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const int index = pass * 256 + threadIdx.x;
            const int row = index >> 3;
            const int quad = index & 7;
            const int token = first_token + row;
            const int r = first_rank + row;
            const float4 av = token < tokens
                ? *reinterpret_cast<const float4 *>(a_g + (long long) token * inner + k0 + quad * 4)
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const float4 wv = r < rank
                ? *reinterpret_cast<const float4 *>(w_g + (long long) r * inner + k0 + quad * 4)
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            a_s[quad * 4 + 0][row] = av.x;
            a_s[quad * 4 + 1][row] = av.y;
            a_s[quad * 4 + 2][row] = av.z;
            a_s[quad * 4 + 3][row] = av.w;
            w_s[quad * 4 + 0][row] = wv.x;
            w_s[quad * 4 + 1][row] = wv.y;
            w_s[quad * 4 + 2][row] = wv.z;
            w_s[quad * 4 + 3][row] = wv.w;
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < DINF_GEMM_K; ++kk) {
            const float4 av = *reinterpret_cast<const float4 *>(&a_s[kk][ty * 4]);
            const float4 wv = *reinterpret_cast<const float4 *>(&w_s[kk][tx * 4]);
            const float as[4] = {av.x, av.y, av.z, av.w};
            const float ws[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] = fmaf(as[i], ws[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }
    float *out = partial + ((long long) split * groups + g) * tokens * rank;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int token = first_token + ty * 4 + i;
        if (token >= tokens) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = first_rank + tx * 4 + j;
            if (r < rank) {
                out[(long long) token * rank + r] = acc[i][j];
            }
        }
    }
}
"#;

/// The tensor-core prompt attention, which needs `mma.sync` and `cp.async`
/// (Ampere or newer); a device or toolkit without them keeps the split-key
/// kernel for prompts.
const FLASH_MODULE: &str = "dinf_inkling2_flash_v1";

const FLASH_SOURCE: &str = r#"
#include <cuda_fp16.h>

#define DINF_NEG_LARGE -1.0e30f

// ---- Prompt attention on tensor cores ----

// Queries per block (16 per warp), keys per tile, and the padded shared row
// width in halves: 136 halves = 68 words, so the eight rows an ldmatrix
// reads land in distinct banks.
#define DINF_FA_WARPS 4
#define DINF_FA_BQ 64
#define DINF_FA_BK 64
#define DINF_FA_D 128
#define DINF_FA_STRIDE (DINF_FA_D + 8)

static __device__ __forceinline__ unsigned int dinf_pack_half2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned int *>(&h);
}

// D += A B for one m16n8k16 tile, F16 inputs and F32 accumulators.
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

static __device__ __forceinline__ void dinf_ldmatrix_x4_trans(unsigned int *r, const __half *p) {
    const unsigned int address = (unsigned int) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(address));
}

// 16 bytes from global to shared, or zeros when `valid` is false.
static __device__ __forceinline__ void dinf_cp_async_16(__half *dst, const __half *src, bool valid) {
    const unsigned int address = (unsigned int) __cvta_generic_to_shared(dst);
    const int bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(address), "l"(src), "r"(bytes));
}

static __device__ __forceinline__ void dinf_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

static __device__ __forceinline__ void dinf_cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

// Rows [first, first + DINF_FA_BK) of one head's f16 keys or values (row
// stride DINF_FA_D) into a padded shared tile, zeros past `keys`.
static __device__ __forceinline__ void dinf_fa_load_tile(
    __half *tile, const __half *rows, int first, int keys) {
    // 16 chunks of 8 halves per row; 128 threads cover 8 rows per pass.
#pragma unroll
    for (int pass = 0; pass < DINF_FA_BK * (DINF_FA_D / 8) / (32 * DINF_FA_WARPS); ++pass) {
        const int index = pass * 32 * DINF_FA_WARPS + threadIdx.x;
        const int row = index / (DINF_FA_D / 8);
        const int chunk = index % (DINF_FA_D / 8);
        const int key = first + row;
        const bool valid = key < keys;
        const __half *src = rows + (long long) (valid ? key : 0) * DINF_FA_D + chunk * 8;
        dinf_cp_async_16(tile + row * DINF_FA_STRIDE + chunk * 8, src, valid);
    }
}

// Attention for a prompt chunk on tensor cores (FlashAttention-2 order),
// with Inkling's relative bias, window and sink. Block (query tile, head):
// four warps of 16 queries each. q [tokens][heads][128] F32 (scaled); bias
// rows [tokens][heads] hold the bias by distance (extent + 1 columns, the
// last for every distance past extent) then the sink logit, which joins the
// softmax with a zero value. k, v: one head's keys at kv_head_stride, rows
// of 128 f16, the first at absolute position first_key. Queries sit at
// first_query + index; keys after the query, or `window` or more back
// (window > 0), are masked. With `gate` (rows gate_stride apart), the output
// is times its sigmoid. out is [tokens][heads][128] F32.
extern "C" __global__ void __launch_bounds__(128) inkling_flash_attention_128(
    const float *q, const float *bias, const __half *k, const __half *v, const float *gate,
    float *out, int tokens, int heads, int kv_heads, int keys, int extent, int window,
    long long kv_head_stride, long long first_query, long long first_key, int gate_stride) {
    __shared__ __align__(16) __half k_tile[DINF_FA_BK * DINF_FA_STRIDE];
    __shared__ __align__(16) __half v_tile[DINF_FA_BK * DINF_FA_STRIDE];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int group = lane >> 2;
    const int quad = lane & 3;
    const int head = blockIdx.y;
    const int kv_head = head / (heads / kv_heads);
    const int tile_first = blockIdx.x * DINF_FA_BQ;
    const int warp_first = tile_first + warp * 16;
    const __half *kh = k + kv_head * kv_head_stride;
    const __half *vh = v + kv_head * kv_head_stride;

    // This thread's two query rows: group and group + 8 of the warp's 16.
    int rows[2];
    long long positions[2];
    const float *tables[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        rows[r] = warp_first + group + 8 * r;
        const int clamped = min(rows[r], tokens - 1);
        positions[r] = first_query + rows[r];
        tables[r] = bias + ((long long) clamped * heads + head) * (extent + 2);
    }

    // The warp's queries as A fragments, one per 16 dimensions.
    unsigned int qa[DINF_FA_D / 16][4];
#pragma unroll
    for (int c = 0; c < DINF_FA_D / 16; ++c) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const bool valid = rows[r] < tokens;
            const float *row = q + ((long long) min(rows[r], tokens - 1) * heads + head) * DINF_FA_D + c * 16 + 2 * quad;
            const float2 lo = valid ? *reinterpret_cast<const float2 *>(row) : make_float2(0.0f, 0.0f);
            const float2 hi = valid ? *reinterpret_cast<const float2 *>(row + 8) : make_float2(0.0f, 0.0f);
            qa[c][r] = dinf_pack_half2(lo.x, lo.y);
            qa[c][2 + r] = dinf_pack_half2(hi.x, hi.y);
        }
    }

    // Key range: the tile's causal end, and its window start when sliding.
    const long long last_row = min(tile_first + DINF_FA_BQ, tokens) - 1;
    const long long first_position = first_query + tile_first;
    const long long last_position = first_query + last_row;
    long long begin = 0;
    if (window > 0) {
        begin = max(0LL, first_position - window + 1 - first_key);
    }
    begin = begin / DINF_FA_BK * DINF_FA_BK;
    const long long end = min((long long) keys, last_position - first_key + 1);

    float row_max[2];
    float row_sum[2];
    float acc[DINF_FA_D / 8][4];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        row_max[r] = tables[r][extent + 1];
        row_sum[r] = 1.0f;
    }
#pragma unroll
    for (int n = 0; n < DINF_FA_D / 8; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[n][i] = 0.0f;
        }
    }

    if (begin < end) {
        dinf_fa_load_tile(k_tile, kh, (int) begin, keys);
        dinf_cp_async_commit();
    }
    for (long long kb = begin; kb < end; kb += DINF_FA_BK) {
        dinf_cp_async_wait_all();
        __syncthreads();
        // V for this tile streams in while the scores are computed.
        dinf_fa_load_tile(v_tile, vh, (int) kb, keys);
        dinf_cp_async_commit();

        // S = Q K^T: eight 8-key tiles, each [2 rows][2 keys] per thread.
        float s[DINF_FA_BK / 8][4];
#pragma unroll
        for (int n = 0; n < DINF_FA_BK / 8; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[n][i] = 0.0f;
            }
        }
#pragma unroll
        for (int c = 0; c < DINF_FA_D / 16; ++c) {
#pragma unroll
            for (int n = 0; n < DINF_FA_BK / 16; ++n) {
                // Matrices: keys +0..7 / +8..15 by dims +0..7 / +8..15.
                const int m = lane >> 3;
                const int key = n * 16 + (m >> 1) * 8 + (lane & 7);
                unsigned int b[4];
                dinf_ldmatrix_x4(b, k_tile + key * DINF_FA_STRIDE + c * 16 + (m & 1) * 8);
                dinf_mma_16816(s[2 * n], qa[c], b);
                dinf_mma_16816(s[2 * n + 1], qa[c], b + 2);
            }
        }

        // Bias, masks and the online softmax for rows group and group + 8.
        float tile_max[2] = {DINF_NEG_LARGE, DINF_NEG_LARGE};
#pragma unroll
        for (int n = 0; n < DINF_FA_BK / 8; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = i >> 1;
                const long long key = kb + n * 8 + 2 * quad + (i & 1);
                const long long distance = positions[r] - (first_key + key);
                const bool valid = rows[r] < tokens && key < keys && distance >= 0
                    && (window <= 0 || distance < window);
                s[n][i] = valid ? s[n][i] + tables[r][distance < extent ? distance : extent] : DINF_NEG_LARGE;
                tile_max[r] = fmaxf(tile_max[r], s[n][i]);
            }
        }
        float correction[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffu, tile_max[r], 1));
            tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffu, tile_max[r], 2));
            const float next_max = fmaxf(row_max[r], tile_max[r]);
            correction[r] = expf(row_max[r] - next_max);
            row_max[r] = next_max;
        }
        float tile_sum[2] = {0.0f, 0.0f};
#pragma unroll
        for (int n = 0; n < DINF_FA_BK / 8; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = i >> 1;
                // Masked scores sit far below the finite running max (seeded
                // by the sink), so their weights underflow to zero.
                s[n][i] = expf(s[n][i] - row_max[r]);
                tile_sum[r] += s[n][i];
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            tile_sum[r] += __shfl_xor_sync(0xffffffffu, tile_sum[r], 1);
            tile_sum[r] += __shfl_xor_sync(0xffffffffu, tile_sum[r], 2);
            row_sum[r] = row_sum[r] * correction[r] + tile_sum[r];
        }
#pragma unroll
        for (int n = 0; n < DINF_FA_D / 8; ++n) {
            acc[n][0] *= correction[0];
            acc[n][1] *= correction[0];
            acc[n][2] *= correction[1];
            acc[n][3] *= correction[1];
        }

        dinf_cp_async_wait_all();
        __syncthreads();
        // The next keys stream in during O += P V.
        if (kb + DINF_FA_BK < end) {
            dinf_fa_load_tile(k_tile, kh, (int) (kb + DINF_FA_BK), keys);
            dinf_cp_async_commit();
        }
#pragma unroll
        for (int c = 0; c < DINF_FA_BK / 16; ++c) {
            // The scores of keys 16c.. as an A fragment, straight from registers.
            const unsigned int pa[4] = {
                dinf_pack_half2(s[2 * c][0], s[2 * c][1]),
                dinf_pack_half2(s[2 * c][2], s[2 * c][3]),
                dinf_pack_half2(s[2 * c + 1][0], s[2 * c + 1][1]),
                dinf_pack_half2(s[2 * c + 1][2], s[2 * c + 1][3]),
            };
#pragma unroll
            for (int n = 0; n < DINF_FA_D / 16; ++n) {
                // Transposed matrices: keys +0..7 / +8..15 by dims +0..7 / +8..15.
                const int m = lane >> 3;
                const int key = c * 16 + (m & 1) * 8 + (lane & 7);
                unsigned int b[4];
                dinf_ldmatrix_x4_trans(b, v_tile + key * DINF_FA_STRIDE + n * 16 + (m >> 1) * 8);
                dinf_mma_16816(acc[2 * n], pa, b);
                dinf_mma_16816(acc[2 * n + 1], pa, b + 2);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        if (rows[r] >= tokens) {
            continue;
        }
        const float inv = 1.0f / row_sum[r];
        const long long base = ((long long) rows[r] * heads + head) * DINF_FA_D;
        const float *g = gate != nullptr ? gate + (long long) rows[r] * gate_stride + head * DINF_FA_D : nullptr;
#pragma unroll
        for (int n = 0; n < DINF_FA_D / 8; ++n) {
            const int d = n * 8 + 2 * quad;
            float a = acc[n][2 * r] * inv;
            float b = acc[n][2 * r + 1] * inv;
            if (g != nullptr) {
                a /= 1.0f + expf(-g[d]);
                b /= 1.0f + expf(-g[d + 1]);
            }
            *reinterpret_cast<float2 *>(out + base + d) = make_float2(a, b);
        }
    }
}
"#;

static FLASH_PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();

fn flash_ptx() -> Result<&'static str> {
    match FLASH_PTX.get_or_init(|| {
        compile_ptx_with_opts(
            FLASH_SOURCE,
            CompileOptions {
                include_paths: cuda_include_paths(),
                name: Some("inkling2_flash.cu".to_string()),
                arch: Some("sm_80"),
                use_fast_math: Some(false),
                ..Default::default()
            },
        )
        .map(|ptx| ptx.to_src())
        .map_err(|error| error.to_string())
    }) {
        Ok(ptx) => Ok(ptx.as_str()),
        Err(error) => {
            candle_core::bail!("failed to compile the Inkling 2 prompt attention: {error}")
        }
    }
}

/// Whether the tensor-core prompt attention compiles and loads on `device`.
fn flash_available(device: &CudaDevice) -> bool {
    flash_ptx().is_ok_and(|ptx| {
        device
            .get_or_load_custom_func("inkling_flash_attention_128", FLASH_MODULE, ptx)
            .is_ok()
    })
}

static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();

fn ptx() -> Result<&'static str> {
    match PTX.get_or_init(|| {
        let include_paths = cuda_include_paths();
        compile_ptx_with_opts(
            SOURCE,
            CompileOptions {
                include_paths,
                name: Some("inkling2.cu".to_string()),
                use_fast_math: Some(false),
                ..Default::default()
            },
        )
        .map(|ptx| ptx.to_src())
        .map_err(|error| error.to_string())
    }) {
        Ok(ptx) => Ok(ptx.as_str()),
        Err(error) => candle_core::bail!("failed to compile the Inkling 2 CUDA kernels: {error}"),
    }
}

/// Toolkit include directories for `cuda_fp16.h`, as the other kernels find them.
fn cuda_include_paths() -> Vec<String> {
    let mut paths = Vec::new();
    for variable in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
        let Some(root) = std::env::var_os(variable) else {
            continue;
        };
        let root = std::path::PathBuf::from(root);
        for include in [root.clone(), root.join("include")] {
            if include.join("cuda_fp16.h").is_file() {
                let include = include.to_string_lossy().into_owned();
                if !paths.contains(&include) {
                    paths.push(include);
                }
            }
        }
    }
    for candidate in ["/usr/local/cuda/include", "/opt/cuda/include"] {
        let include = std::path::PathBuf::from(candidate);
        if include.join("cuda_fp16.h").is_file() {
            paths.push(include.to_string_lossy().into_owned());
        }
    }
    paths
}

/// Compile the kernels and load each entry point ahead of use, so no module
/// load happens inside a captured decode step.
pub(super) fn precompile(device: &candle_core::Device) -> anyhow::Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    let device = device.as_cuda_device()?;
    for name in [
        "inkling_short_conv",
        "inkling_short_conv_state",
        "inkling_scaled_l2_clip",
        "inkling_rms_norm",
        "inkling_rms_pair",
        "inkling_gate_combine",
        "inkling_situ",
        "inkling_sigmoid_top_k",
        "inkling_bf16_matvec",
        "inkling_attention_partial_64",
        "inkling_attention_combine_64",
        "inkling_attention_partial_128",
        "inkling_attention_combine_128",
        "inkling_attention_partial_256",
        "inkling_attention_combine_256",
        "inkling_rms_pair_groups",
        "inkling_gate_combine_norm",
        "inkling_gate_down",
        "inkling_gate_up_norm",
        "inkling_kv_finish",
        "inkling_query_prep_16",
        "inkling_query_prep_32",
        "inkling_post_conv",
        "inkling_bf16_matvec3",
        "inkling_route",
        "inkling_gemm_nt_split",
    ]
    .into_iter()
    .chain(DEPTH_MIX_ONE_KERNELS.into_iter().flatten())
    {
        device.get_or_load_custom_func(name, MODULE, ptx()?)?;
    }
    // The gate norms' scratch must exist before any captured step.
    gate_scratch(device)?;
    // Optional: without it, prompts take the split-key kernel.
    let _ = flash_available(device);
    Ok(())
}

fn cuda_only<T>() -> Result<T> {
    candle_core::bail!("this Inkling 2 kernel only runs on CUDA")
}

fn to_i32(value: usize, what: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| candle_core::Error::Msg(format!("Inkling 2 {what} exceeds the kernel range")))
}

/// A contiguous view of `layout`'s elements.
fn view<'a, T: candle_core::cuda_backend::CudaDType>(
    storage: &'a CudaStorage,
    layout: &Layout,
    what: &str,
) -> Result<CudaView<'a, T>> {
    if !layout.is_contiguous() {
        candle_core::bail!("Inkling 2 {what} must be contiguous")
    }
    let slice = storage.as_cuda_slice::<T>()?;
    let start = layout.start_offset();
    let end = start + layout.shape().elem_count();
    if end > slice.len() {
        candle_core::bail!("Inkling 2 {what} view exceeds its storage")
    }
    Ok(slice.slice(start..end))
}

fn same_device(a: &CudaDevice, b: &CudaStorage) -> Result<()> {
    if a.id() != b.device().id() {
        candle_core::bail!("Inkling 2 kernel inputs are on different devices")
    }
    Ok(())
}

fn grid_1d(elements: usize, threads: usize) -> Result<LaunchConfig> {
    let blocks = elements.div_ceil(threads);
    if blocks > u32::MAX as usize {
        candle_core::bail!("Inkling 2 launch is too large")
    }
    Ok(LaunchConfig {
        grid_dim: (blocks.max(1) as u32, 1, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn output(
    device: &CudaDevice,
    elements: usize,
) -> Result<candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>> {
    // SAFETY: every kernel here writes each element of its output.
    Ok(unsafe { device.alloc::<f32>(elements.max(1))? })
}

struct ShortConv {
    tokens: usize,
    heads: usize,
}

impl ShortConv {
    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        history: &CudaStorage,
        hl: &Layout,
        weight: &CudaStorage,
        wl: &Layout,
        taps: Option<(&CudaStorage, &Layout)>,
    ) -> Result<(CudaStorage, Shape)> {
        let (rows, channels) = hl.shape().dims2()?;
        let (weight_channels, width) = wl.shape().dims2()?;
        let tokens = self.tokens;
        if weight_channels != channels
            || width < 1
            || rows != tokens + width - 1
            || (self.heads > 0 && !channels.is_multiple_of(self.heads))
        {
            candle_core::bail!("Inkling 2 short-conv geometry differs")
        }
        let device = history.device();
        same_device(device, weight)?;
        let history = view::<f32>(history, hl, "conv history")?;
        let weight = view::<f32>(weight, wl, "conv weight")?;
        let taps = match taps {
            Some((storage, layout)) => {
                same_device(device, storage)?;
                if layout.shape().elem_count() != tokens * self.heads * width {
                    candle_core::bail!("Inkling 2 dynamic conv taps differ in shape")
                }
                Some(view::<f32>(storage, layout, "conv taps")?)
            }
            None => None,
        };
        let terms = if self.heads > 0 { 2 * width } else { width };
        let scale = 1.0 / ((terms + 1) as f32).sqrt();
        let elements = tokens * channels;
        let mut out = output(device, elements)?;
        let function = device.get_or_load_custom_func("inkling_short_conv", MODULE, ptx()?)?;
        let (tokens_i, channels_i, width_i, heads_i) = (
            to_i32(tokens, "conv tokens")?,
            to_i32(channels, "conv channels")?,
            to_i32(width, "conv width")?,
            to_i32(self.heads, "conv heads")?,
        );
        let mut builder = function.builder();
        builder.arg(&history).arg(&weight);
        // Without taps the kernel never reads its taps argument.
        match &taps {
            Some(taps) => builder.arg(taps),
            None => builder.arg(&history),
        };
        builder
            .arg(&mut out)
            .arg(&tokens_i)
            .arg(&channels_i)
            .arg(&width_i)
            .arg(&heads_i)
            .arg(&scale);
        // SAFETY: shapes and views were checked; the kernel stays in bounds.
        unsafe { builder.launch(grid_1d(elements, 256)?) }.w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(out, device.clone()),
            Shape::from((tokens, channels)),
        ))
    }
}

impl CustomOp2 for ShortConv {
    fn name(&self) -> &'static str {
        "inkling2-short-conv-cuda"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        cuda_only()
    }
    fn cuda_fwd(
        &self,
        history: &CudaStorage,
        hl: &Layout,
        weight: &CudaStorage,
        wl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        self.launch(history, hl, weight, wl, None)
    }
}

impl CustomOp3 for ShortConv {
    fn name(&self) -> &'static str {
        "inkling2-short-conv-taps-cuda"
    }
    #[allow(clippy::too_many_arguments)]
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        cuda_only()
    }
    #[allow(clippy::too_many_arguments)]
    fn cuda_fwd(
        &self,
        history: &CudaStorage,
        hl: &Layout,
        weight: &CudaStorage,
        wl: &Layout,
        taps: &CudaStorage,
        tl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        self.launch(history, hl, weight, wl, Some((taps, tl)))
    }
}

struct ShortConvState;

impl CustomOp3 for ShortConvState {
    fn name(&self) -> &'static str {
        "inkling2-short-conv-state-cuda"
    }
    #[allow(clippy::too_many_arguments)]
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        cuda_only()
    }
    #[allow(clippy::too_many_arguments)]
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        past: &CudaStorage,
        pl: &Layout,
        weight: &CudaStorage,
        wl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (tokens, channels) = xl.shape().dims2()?;
        let (weight_channels, width) = wl.shape().dims2()?;
        if weight_channels != channels || width < 1 || pl.shape().dims() != [width - 1, channels] {
            candle_core::bail!("Inkling 2 short-conv state geometry differs")
        }
        let device = x.device();
        same_device(device, past)?;
        same_device(device, weight)?;
        let x = view::<f32>(x, xl, "conv input")?;
        let past = view::<f32>(past, pl, "conv state")?;
        let weight = view::<f32>(weight, wl, "conv weight")?;
        let rows = tokens + width - 1;
        let elements = rows * channels;
        let mut out = output(device, elements)?;
        let scale = 1.0 / ((width + 1) as f32).sqrt();
        let function =
            device.get_or_load_custom_func("inkling_short_conv_state", MODULE, ptx()?)?;
        let (tokens_i, channels_i, width_i) = (
            to_i32(tokens, "conv tokens")?,
            to_i32(channels, "conv channels")?,
            to_i32(width, "conv width")?,
        );
        let mut builder = function.builder();
        builder
            .arg(&x)
            .arg(&past)
            .arg(&weight)
            .arg(&mut out)
            .arg(&tokens_i)
            .arg(&channels_i)
            .arg(&width_i)
            .arg(&scale);
        // SAFETY: shapes and views were checked; the kernel stays in bounds.
        unsafe { builder.launch(grid_1d(elements, 256)?) }.w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(out, device.clone()),
            Shape::from((rows, channels)),
        ))
    }
}

/// The residual short convolution of `x` `[tokens, channels]` after `past`
/// `[width - 1, channels]` with static `weight` `[channels, width]`. Returns
/// `[tokens + width - 1, channels]`: the output rows, then the next state.
pub(super) fn short_conv_with_state(x: &Tensor, past: &Tensor, weight: &Tensor) -> Result<Tensor> {
    x.apply_op3_no_bwd(past, weight, &ShortConvState)
}

/// The residual short convolution over `history` (`[past; x]`, `tokens +
/// width - 1` rows) with `weight` `[channels, width]` and optional per-token
/// taps `[tokens, heads * width]`.
pub(super) fn short_conv(
    history: &Tensor,
    weight: &Tensor,
    taps: Option<(&Tensor, usize)>,
    tokens: usize,
) -> Result<Tensor> {
    match taps {
        Some((taps, heads)) => history.apply_op3_no_bwd(weight, taps, &ShortConv { tokens, heads }),
        None => history.apply_op2_no_bwd(weight, &ShortConv { tokens, heads: 0 }),
    }
}

struct ScaledClip {
    heads: usize,
    flags: i32,
    threshold: f32,
    post_scale: f32,
}

impl CustomOp3 for ScaledClip {
    fn name(&self) -> &'static str {
        "inkling2-scaled-l2-clip-cuda"
    }
    #[allow(clippy::too_many_arguments)]
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        cuda_only()
    }
    #[allow(clippy::too_many_arguments)]
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        scale: &CudaStorage,
        sl: &Layout,
        token_scale: &CudaStorage,
        tl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (tokens, heads, dim) = xl.shape().dims3()?;
        if heads != self.heads
            || (self.flags & 1 != 0 && sl.shape().dims() != [heads, dim])
            || (self.flags & 2 != 0 && tl.shape().elem_count() != tokens)
        {
            candle_core::bail!("Inkling 2 clip geometry differs")
        }
        let device = x.device();
        same_device(device, scale)?;
        same_device(device, token_scale)?;
        let rows = tokens * heads;
        let x = view::<f32>(x, xl, "clip input")?;
        let scale = view::<f32>(scale, sl, "clip scale")?;
        let token_scale = view::<f32>(token_scale, tl, "clip token scale")?;
        let mut out = output(device, rows * dim)?;
        let function = device.get_or_load_custom_func("inkling_scaled_l2_clip", MODULE, ptx()?)?;
        let (rows_i, dim_i, heads_i) = (
            to_i32(rows, "clip rows")?,
            to_i32(dim, "clip width")?,
            to_i32(heads, "clip heads")?,
        );
        let mut builder = function.builder();
        builder
            .arg(&x)
            .arg(&scale)
            .arg(&token_scale)
            .arg(&mut out)
            .arg(&rows_i)
            .arg(&dim_i)
            .arg(&heads_i)
            .arg(&self.flags)
            .arg(&self.threshold)
            .arg(&self.post_scale);
        // Four warps per block, one row per warp.
        let config = LaunchConfig {
            grid_dim: (rows.div_ceil(4).max(1) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: shapes and views were checked; the kernel stays in bounds.
        unsafe { builder.launch(config) }.w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(out, device.clone()),
            xl.shape().clone(),
        ))
    }
}

/// `x` `[tokens, heads, dim]` times `scale` `[heads, dim]` (if any), clipped
/// to an L2 norm of `threshold` per head, then times `token_scale[token]`
/// (if any) and `post_scale`.
pub(super) fn scaled_l2_clip(
    x: &Tensor,
    scale: Option<&Tensor>,
    token_scale: Option<&Tensor>,
    threshold: f32,
    post_scale: f32,
) -> Result<Tensor> {
    let (_, heads, _) = x.dims3()?;
    let flags = i32::from(scale.is_some()) | (i32::from(token_scale.is_some()) << 1);
    x.apply_op3_no_bwd(
        scale.unwrap_or(x),
        token_scale.unwrap_or(x),
        &ScaledClip {
            heads,
            flags,
            threshold,
            post_scale,
        },
    )
}

/// Most block states and queries one depth-mix call supports (eight
/// sources including the current row).
pub(super) const DEPTH_MIX_MAX_BLOCKS: usize = 7;
pub(super) const DEPTH_MIX_MAX_QUERIES: usize = 4;

const ROW_THREADS: usize = 256;

struct RmsNorm {
    eps: f32,
    weighted: bool,
}

impl CustomOp2 for RmsNorm {
    fn name(&self) -> &'static str {
        "inkling2-rms-norm-cuda"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        cuda_only()
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        weight: &CudaStorage,
        wl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let hidden = *xl.shape().dims().last().unwrap_or(&0);
        if hidden == 0 || (self.weighted && wl.shape().elem_count() != hidden) {
            candle_core::bail!("Inkling 2 RMS norm geometry differs")
        }
        let rows = xl.shape().elem_count() / hidden;
        let device = x.device();
        same_device(device, weight)?;
        let x = view::<f32>(x, xl, "norm input")?;
        let weight = view::<f32>(weight, wl, "norm weight")?;
        let mut out = output(device, rows * hidden)?;
        let function = device.get_or_load_custom_func("inkling_rms_norm", MODULE, ptx()?)?;
        let hidden_i = to_i32(hidden, "norm width")?;
        let weighted = i32::from(self.weighted);
        let mut builder = function.builder();
        builder
            .arg(&x)
            .arg(&weight)
            .arg(&mut out)
            .arg(&hidden_i)
            .arg(&weighted)
            .arg(&self.eps);
        let config = LaunchConfig {
            grid_dim: (rows.max(1) as u32, 1, 1),
            block_dim: (ROW_THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: shapes and views were checked; the kernel stays in bounds.
        unsafe { builder.launch(config) }.w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(out, device.clone()),
            xl.shape().clone(),
        ))
    }
}

/// RMS norm over the last dimension, times `weight` when given.
pub(super) fn rms_norm(x: &Tensor, weight: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    x.apply_op2_no_bwd(
        weight.unwrap_or(x),
        &RmsNorm {
            eps,
            weighted: weight.is_some(),
        },
    )
}

struct RmsPair {
    eps: f32,
}

impl CustomOp2 for RmsPair {
    fn name(&self) -> &'static str {
        "inkling2-rms-pair-cuda"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        cuda_only()
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        m: &CudaStorage,
        ml: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (rows, hidden) = xl.shape().dims2()?;
        if ml.shape().dims() != [rows, hidden] {
            candle_core::bail!("Inkling 2 gate inputs differ in shape")
        }
        let device = x.device();
        same_device(device, m)?;
        let x = view::<f32>(x, xl, "gate input")?;
        let m = view::<f32>(m, ml, "gate message")?;
        let mut out = output(device, rows * 2 * hidden)?;
        let function = device.get_or_load_custom_func("inkling_rms_pair", MODULE, ptx()?)?;
        let hidden_i = to_i32(hidden, "gate width")?;
        let mut builder = function.builder();
        builder
            .arg(&x)
            .arg(&m)
            .arg(&mut out)
            .arg(&hidden_i)
            .arg(&self.eps);
        let config = LaunchConfig {
            grid_dim: (rows.max(1) as u32, 1, 1),
            block_dim: (ROW_THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: shapes and views were checked; the kernel stays in bounds.
        unsafe { builder.launch(config) }.w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(out, device.clone()),
            Shape::from((rows, 2 * hidden)),
        ))
    }
}

/// `[rms(x) | rms(message)]` per row.
pub(super) fn rms_pair(x: &Tensor, message: &Tensor, eps: f32) -> Result<Tensor> {
    x.apply_op2_no_bwd(message, &RmsPair { eps })
}

struct GateCombine;

impl CustomOp3 for GateCombine {
    fn name(&self) -> &'static str {
        "inkling2-gate-combine-cuda"
    }
    #[allow(clippy::too_many_arguments)]
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        cuda_only()
    }
    #[allow(clippy::too_many_arguments)]
    fn cuda_fwd(
        &self,
        fg: &CudaStorage,
        fl: &Layout,
        x: &CudaStorage,
        xl: &Layout,
        m: &CudaStorage,
        ml: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (rows, hidden) = xl.shape().dims2()?;
        if fl.shape().dims() != [rows, 2 * hidden] || ml.shape().dims() != [rows, hidden] {
            candle_core::bail!("Inkling 2 gate combine geometry differs")
        }
        let device = x.device();
        same_device(device, fg)?;
        same_device(device, m)?;
        let fg = view::<f32>(fg, fl, "gate projections")?;
        let x = view::<f32>(x, xl, "gate input")?;
        let m = view::<f32>(m, ml, "gate message")?;
        let elements = rows * hidden;
        let mut out = output(device, elements)?;
        let function = device.get_or_load_custom_func("inkling_gate_combine", MODULE, ptx()?)?;
        let (rows_i, hidden_i) = (to_i32(rows, "gate rows")?, to_i32(hidden, "gate width")?);
        let mut builder = function.builder();
        builder
            .arg(&fg)
            .arg(&x)
            .arg(&m)
            .arg(&mut out)
            .arg(&rows_i)
            .arg(&hidden_i);
        // SAFETY: shapes and views were checked; the kernel stays in bounds.
        unsafe { builder.launch(grid_1d(elements, 256)?) }.w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(out, device.clone()),
            Shape::from((rows, hidden)),
        ))
    }
}

/// `(1 - tanh(f)) * x + tanh(g) * message` for `forget_gain` rows `[f | g]`.
pub(super) fn gate_combine(forget_gain: &Tensor, x: &Tensor, message: &Tensor) -> Result<Tensor> {
    forget_gain.apply_op3_no_bwd(x, message, &GateCombine)
}

struct Situ {
    beta: f32,
    linear_beta: f32,
}

impl CustomOp1 for Situ {
    fn name(&self) -> &'static str {
        "inkling2-situ-cuda"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        cuda_only()
    }
    fn cuda_fwd(&self, gate_up: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let dims = layout.shape().dims();
        let width = *dims.last().unwrap_or(&0);
        if width == 0 || !width.is_multiple_of(2) {
            candle_core::bail!("Inkling 2 SiTU input must have an even last dimension")
        }
        let ffn = width / 2;
        let rows = layout.shape().elem_count() / width;
        let device = gate_up.device();
        let input = view::<f32>(gate_up, layout, "SiTU input")?;
        let elements = rows * ffn;
        let mut out = output(device, elements)?;
        let function = device.get_or_load_custom_func("inkling_situ", MODULE, ptx()?)?;
        let (rows_i, ffn_i) = (to_i32(rows, "SiTU rows")?, to_i32(ffn, "SiTU width")?);
        let mut builder = function.builder();
        builder
            .arg(&input)
            .arg(&mut out)
            .arg(&rows_i)
            .arg(&ffn_i)
            .arg(&self.beta)
            .arg(&self.linear_beta);
        // SAFETY: shapes and views were checked; the kernel stays in bounds.
        unsafe { builder.launch(grid_1d(elements, 256)?) }.w()?;
        let mut shape = dims.to_vec();
        *shape.last_mut().expect("nonempty shape") = ffn;
        Ok((
            CudaStorage::wrap_cuda_slice(out, device.clone()),
            Shape::from(shape),
        ))
    }
}

/// SiTU over packed `[..., gate | up]` rows.
pub(super) fn situ(gate_up: &Tensor, beta: f32, linear_beta: f32) -> Result<Tensor> {
    gate_up.apply_op1_no_bwd(&Situ { beta, linear_beta })
}

struct Bf16Matvec;

impl CustomOp2 for Bf16Matvec {
    fn name(&self) -> &'static str {
        "inkling2-bf16-matvec-cuda"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        cuda_only()
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        w: &CudaStorage,
        wl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (rows, cols) = wl.shape().dims2()?;
        if xl.shape().elem_count() != cols
            || !cols.is_multiple_of(8)
            || w.dtype() != DType::BF16
            || !wl.is_contiguous()
            || !wl.start_offset().is_multiple_of(8)
            || !xl.start_offset().is_multiple_of(4)
        {
            candle_core::bail!("Inkling 2 BF16 matvec geometry differs")
        }
        let device = x.device();
        same_device(device, w)?;
        let x = view::<f32>(x, xl, "matvec input")?;
        let slice = w.as_cuda_slice::<half::bf16>()?;
        let weights = slice.slice(wl.start_offset()..wl.start_offset() + rows * cols);
        let mut out = output(device, rows)?;
        let function = device.get_or_load_custom_func("inkling_bf16_matvec", MODULE, ptx()?)?;
        let (rows_i, cols_i) = (to_i32(rows, "matvec rows")?, to_i32(cols, "matvec width")?);
        let mut builder = function.builder();
        builder
            .arg(&x)
            .arg(&weights)
            .arg(&mut out)
            .arg(&rows_i)
            .arg(&cols_i);
        let config = LaunchConfig {
            grid_dim: (rows.div_ceil(4).max(1) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: shapes, alignment and views were checked; the kernel stays in bounds.
        unsafe { builder.launch(config) }.w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(out, device.clone()),
            Shape::from((1, rows)),
        ))
    }
}

/// `x` `[1, cols]` F32 times BF16 `weight` `[rows, cols]` transposed: `[1, rows]` F32.
pub(super) fn bf16_matvec(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let x = if x.layout().start_offset().is_multiple_of(4) {
        x.clone()
    } else {
        x.force_contiguous()?
    };
    x.apply_op2_no_bwd(weight, &Bf16Matvec)
}

const ROUTE_MAX_EXPERTS: usize = 256;
const ROUTE_MAX_TOP_K: usize = 32;

struct SigmoidTopK {
    top_k: usize,
    scale: f32,
}

impl CustomOp2 for SigmoidTopK {
    fn name(&self) -> &'static str {
        "inkling2-sigmoid-top-k-cuda"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        cuda_only()
    }
    fn cuda_fwd(
        &self,
        logits: &CudaStorage,
        ll: &Layout,
        bias: &CudaStorage,
        bl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (tokens, experts) = ll.shape().dims2()?;
        if bl.shape().elem_count() != experts
            || experts > ROUTE_MAX_EXPERTS
            || self.top_k == 0
            || self.top_k > ROUTE_MAX_TOP_K
            || self.top_k > experts
        {
            candle_core::bail!("Inkling 2 route geometry is outside the kernel's range")
        }
        let device = logits.device();
        same_device(device, bias)?;
        let logits = view::<f32>(logits, ll, "router logits")?;
        let bias = view::<f32>(bias, bl, "router bias")?;
        let routes = tokens * self.top_k;
        // One buffer: U32 ids, then F32 weights, split by the caller.
        // SAFETY: the kernel writes every id and weight of every token.
        let mut ids = unsafe { device.alloc::<u32>(routes.max(1))? };
        let mut weights = output(device, routes)?;
        let function = device.get_or_load_custom_func("inkling_sigmoid_top_k", MODULE, ptx()?)?;
        let (tokens_i, experts_i, top_k_i) = (
            to_i32(tokens, "route tokens")?,
            to_i32(experts, "route experts")?,
            to_i32(self.top_k, "route top-k")?,
        );
        let mut builder = function.builder();
        builder
            .arg(&logits)
            .arg(&bias)
            .arg(&mut ids)
            .arg(&mut weights)
            .arg(&tokens_i)
            .arg(&experts_i)
            .arg(&top_k_i)
            .arg(&self.scale);
        // Four warps per block, one token per warp.
        let config = LaunchConfig {
            grid_dim: (tokens.div_ceil(4).max(1) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: shapes and views were checked; the kernel stays in bounds.
        unsafe { builder.launch(config) }.w()?;
        // Return the weights; the ids travel through `ROUTE_IDS`.
        ROUTE_IDS.with(|slot| {
            *slot.borrow_mut() = Some(CudaStorage::wrap_cuda_slice(ids, device.clone()));
        });
        Ok((
            CudaStorage::wrap_cuda_slice(weights, device.clone()),
            Shape::from((tokens, self.top_k)),
        ))
    }
}

thread_local! {
    /// The ids of the route the last [`SigmoidTopK`] launch on this thread computed.
    static ROUTE_IDS: std::cell::RefCell<Option<CudaStorage>> = const { std::cell::RefCell::new(None) };
}

/// Sigmoid-scored top-k routing: `logits` `[tokens, experts]` and selection
/// `bias` `[experts]` give U32 ids `[tokens, top_k]` (by score + bias, ties
/// to the lower expert) and their unbiased scores normalized and times `scale`.
pub(super) fn sigmoid_top_k(
    logits: &Tensor,
    bias: &Tensor,
    top_k: usize,
    scale: f32,
) -> Result<(Tensor, Tensor)> {
    let weights = logits.apply_op2_no_bwd(bias, &SigmoidTopK { top_k, scale })?;
    let storage = ROUTE_IDS
        .with(|slot| slot.borrow_mut().take())
        .ok_or_else(|| candle_core::Error::Msg("Inkling 2 route ids are missing".into()))?;
    let ids = Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        weights.shape().clone(),
        candle_core::op::BackpropOp::none(),
        false,
    );
    Ok((ids, weights))
}

/// Fewest queries that run on the tensor-core prompt kernel; fewer split
/// keys across blocks instead.
const FLASH_MIN_TOKENS: usize = 64;
/// Queries per tensor-core prompt block (`DINF_FA_BQ`).
const FLASH_QUERY_TILE: usize = 64;

/// Keys per attention split: four steps of four keys for each of a block's
/// eight warps. More splits keep the GPU busy for a few queries; fewer keep
/// the combine pass short.
const KEYS_PER_SPLIT: usize = 128;
/// Up to this many keys one split per query finishes in the first pass,
/// which costs less than a combine launch.
const SINGLE_SPLIT_KEYS: usize = 256;
/// Most splits per query and head.
const MAX_SPLITS: usize = 1024;
/// Blocks the first pass aims for.
const TARGET_BLOCKS: usize = 4096;

/// Largest query count the fused attention handles. Every query of a chunk
/// reads the same keys, which stay in the GPU caches.
pub(super) const FUSED_ATTENTION_MAX_TOKENS: usize = 512;

/// Attention for up to [`FUSED_ATTENTION_MAX_TOKENS`] queries straight from
/// the f16 cache.
///
/// `q` is `[tokens, heads, head_dim]`, already scaled; `bias` is
/// `[tokens, heads, extent + 2]`: the bias for each distance (column `extent`
/// is every distance past it), then the sink logit. `k`, `v` are the active
/// `[1, kv_heads, keys, head_dim]` f16 cache views, whose first key sits at
/// absolute position `first_key`. With `gate` (`[tokens, heads * head_dim]`
/// logits, whose rows may be slices of wider rows), the output is multiplied
/// by its sigmoid. Returns `[tokens, heads, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn attention(
    q: &Tensor,
    bias: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    extent: usize,
    window: Option<usize>,
    first_query: usize,
    first_key: usize,
    gate: Option<&Tensor>,
) -> Result<Tensor> {
    let device = device_of(q)?;
    let (tokens, heads, q_dim) = q.dims3()?;
    let (batch, kv_heads, keys, cache_dim) = k.dims4()?;
    let kv_strides = k.stride();
    if batch != 1
        || q_dim != head_dim
        || cache_dim != head_dim
        || bias.dims() != [tokens, heads, extent + 2]
        || v.dims() != k.dims()
        || v.stride() != kv_strides
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
        || !matches!(head_dim, 64 | 128 | 256)
        || kv_heads == 0
        || !heads.is_multiple_of(kv_heads)
        || kv_strides[3] != 1
        || kv_strides[2] != head_dim
        || keys == 0
        || tokens == 0
        || tokens > 65_535
        || heads > 65_535
        || gate.is_some_and(|gate| gate.dims() != [tokens, heads * head_dim])
    {
        candle_core::bail!("Inkling 2 fused attention geometry differs")
    }
    let gate_stride = gate.map_or(Ok(0), |gate| row_stride(gate, "attention gate"))?;
    let mut kept = vec![
        dense(q, "attention queries")?,
        dense(bias, "attention bias")?,
        k.clone(),
        v.clone(),
    ];
    kept.extend(gate.cloned());
    let holds = hold(&kept.iter().collect::<Vec<_>>());
    // The key/value reads run from each cache's first active element to the
    // end of the last head's active keys.
    let span = (kv_heads - 1) * kv_strides[1] + keys * head_dim;
    for (storage, layout) in &holds[2..4] {
        let candle_core::Storage::Cuda(storage) = &**storage else {
            return cuda_only();
        };
        if layout.start_offset() + span > storage.as_cuda_slice::<half::f16>()?.len() {
            candle_core::bail!("Inkling 2 attention cache view exceeds its storage")
        }
    }
    let stream = device.cuda_stream();
    let (address, _reads) = addresses(&holds, &device, &stream)?;
    let gate_address = if gate.is_some() { address[4] } else { 0u64 };
    let gate_stride_i = to_i32(gate_stride, "attention gate stride")?;

    if head_dim == 128 && tokens >= FLASH_MIN_TOKENS && flash_available(&device) {
        // Prompt chunks: tensor-core tiles of 64 queries by 64 keys.
        let mut out = output(&device, tokens * heads * head_dim)?;
        let ints = [
            to_i32(tokens, "attention tokens")?,
            to_i32(heads, "attention heads")?,
            to_i32(kv_heads, "attention kv heads")?,
            to_i32(keys, "attention keys")?,
            to_i32(extent, "attention extent")?,
            to_i32(window.unwrap_or(0), "attention window")?,
        ];
        let longs = [kv_strides[1] as i64, first_query as i64, first_key as i64];
        let function = device.get_or_load_custom_func(
            "inkling_flash_attention_128",
            FLASH_MODULE,
            flash_ptx()?,
        )?;
        let mut builder = function.builder();
        builder
            .arg(&address[0])
            .arg(&address[1])
            .arg(&address[2])
            .arg(&address[3])
            .arg(&gate_address)
            .arg(&mut out);
        for value in &ints {
            builder.arg(value);
        }
        for value in &longs {
            builder.arg(value);
        }
        builder.arg(&gate_stride_i);
        let config = LaunchConfig {
            grid_dim: (
                to_u32(tokens.div_ceil(FLASH_QUERY_TILE), "attention tokens")?,
                heads as u32,
                1,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: shapes, strides and the cache span were checked; inputs are held.
        unsafe { builder.launch(config) }.w()?;
        return Ok(wrap(&device, out, (tokens, heads, head_dim)));
    }

    // Enough splits to fill the GPU for a few queries, few for long chunks.
    let wanted = (TARGET_BLOCKS / (tokens * heads)).max(1);
    let splits = if keys <= SINGLE_SPLIT_KEYS {
        1
    } else {
        keys.div_ceil(KEYS_PER_SPLIT)
            .min(wanted)
            .clamp(1, MAX_SPLITS)
    };
    let chunk = keys.div_ceil(splits);
    let splits = keys.div_ceil(chunk);
    let mut partial = if splits > 1 {
        Some(output(&device, tokens * heads * splits * (head_dim + 2))?)
    } else {
        None
    };
    let mut out = output(&device, tokens * heads * head_dim)?;
    let ints = [
        to_i32(heads, "attention heads")?,
        to_i32(kv_heads, "attention kv heads")?,
        to_i32(keys, "attention keys")?,
        to_i32(extent, "attention extent")?,
        to_i32(window.unwrap_or(0), "attention window")?,
    ];
    let (partial_name, combine_name) = match head_dim {
        64 => (
            "inkling_attention_partial_64",
            "inkling_attention_combine_64",
        ),
        128 => (
            "inkling_attention_partial_128",
            "inkling_attention_combine_128",
        ),
        _ => (
            "inkling_attention_partial_256",
            "inkling_attention_combine_256",
        ),
    };
    let longs = [kv_strides[1] as i64, first_query as i64, first_key as i64];
    let (splits_i, chunk_i) = (
        to_i32(splits, "attention splits")?,
        to_i32(chunk, "attention split width")?,
    );
    let null = 0u64;
    let partial_fn = device.get_or_load_custom_func(partial_name, MODULE, ptx()?)?;
    let mut builder = partial_fn.builder();
    builder
        .arg(&address[0])
        .arg(&address[1])
        .arg(&address[2])
        .arg(&address[3])
        .arg(&gate_address);
    match partial.as_mut() {
        Some(partial) => builder.arg(partial),
        None => builder.arg(&null),
    };
    builder.arg(&mut out);
    for value in &ints {
        builder.arg(value);
    }
    for value in &longs {
        builder.arg(value);
    }
    builder.arg(&splits_i).arg(&chunk_i).arg(&gate_stride_i);
    let config = LaunchConfig {
        grid_dim: (splits as u32, heads as u32, tokens as u32),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: shapes, strides and the cache span were checked; inputs are held.
    unsafe { builder.launch(config) }.w()?;

    if let Some(partial) = &partial {
        let combine_fn = device.get_or_load_custom_func(combine_name, MODULE, ptx()?)?;
        let mut builder = combine_fn.builder();
        builder
            .arg(&address[1])
            .arg(partial)
            .arg(&gate_address)
            .arg(&mut out)
            .arg(&ints[0])
            .arg(&ints[3])
            .arg(&splits_i)
            .arg(&gate_stride_i);
        let config = LaunchConfig {
            grid_dim: (heads as u32, tokens as u32, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the partial buffer holds every split of every row.
        unsafe { builder.launch(config) }.w()?;
    }
    Ok(wrap(&device, out, (tokens, heads, head_dim)))
}

// ---- Single-pass launches. These bypass candle's custom ops, which take at
// most three inputs and return one output.

/// Read holds on one launch's input tensors: each storage and layout.
fn hold<'a>(
    tensors: &[&'a Tensor],
) -> Vec<(
    impl std::ops::Deref<Target = candle_core::Storage> + 'a,
    &'a Layout,
)> {
    tensors
        .iter()
        .map(|tensor| tensor.storage_and_layout())
        .collect()
}

/// The address of each held tensor's first element on `device`, and guards
/// that record the reads when they drop after the launch.
fn addresses<'a, G: std::ops::Deref<Target = candle_core::Storage>>(
    holds: &'a [(G, &Layout)],
    device: &CudaDevice,
    stream: &'a CudaStream,
) -> Result<(Vec<u64>, Vec<SyncOnDrop<'a>>)> {
    let mut addresses = Vec::with_capacity(holds.len());
    let mut guards = Vec::with_capacity(holds.len());
    for (storage, layout) in holds {
        let candle_core::Storage::Cuda(storage) = &**storage else {
            return cuda_only();
        };
        same_device(device, storage)?;
        let (base, guard) = match storage.dtype() {
            DType::F32 => storage.as_cuda_slice::<f32>()?.device_ptr(stream),
            DType::F16 => storage.as_cuda_slice::<half::f16>()?.device_ptr(stream),
            DType::U32 => storage.as_cuda_slice::<u32>()?.device_ptr(stream),
            dtype => candle_core::bail!("Inkling 2 kernels take no {dtype:?} inputs"),
        };
        let offset = layout.start_offset() * storage.dtype().size_in_bytes();
        addresses.push(base + offset as u64);
        guards.push(guard);
    }
    Ok((addresses, guards))
}

fn device_of(tensor: &Tensor) -> Result<CudaDevice> {
    match tensor.device() {
        candle_core::Device::Cuda(device) => Ok(device.clone()),
        _ => cuda_only(),
    }
}

/// `tensor` as contiguous F32 (itself when it already is).
fn dense(tensor: &Tensor, what: &str) -> Result<Tensor> {
    if tensor.dtype() != DType::F32 {
        candle_core::bail!("Inkling 2 {what} must be f32")
    }
    tensor.contiguous()
}

/// The row stride (elements) of a 2-D F32 tensor whose rows are contiguous
/// but may be column slices of wider rows.
fn row_stride(tensor: &Tensor, what: &str) -> Result<usize> {
    let (rows, width) = tensor.dims2()?;
    let stride = tensor.stride();
    if tensor.dtype() != DType::F32 || stride[1] != 1 || (rows > 1 && stride[0] < width) {
        candle_core::bail!("Inkling 2 {what} must be f32 with contiguous rows")
    }
    Ok(if rows > 1 { stride[0] } else { width })
}

fn wrap(device: &CudaDevice, slice: CudaSlice<f32>, shape: impl Into<Shape>) -> Tensor {
    Tensor::from((
        candle_core::Storage::Cuda(CudaStorage::wrap_cuda_slice(slice, device.clone())),
        shape.into(),
    ))
}

fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| candle_core::Error::Msg(format!("Inkling 2 {what} exceeds the kernel range")))
}

/// `inkling_depth_mix_one_<sources>_<queries>` by `[blocks - 1][queries - 1]`.
const DEPTH_MIX_ONE_KERNELS: [[&str; DEPTH_MIX_MAX_QUERIES]; DEPTH_MIX_MAX_BLOCKS] = [
    [
        "inkling_depth_mix_one_2_1",
        "inkling_depth_mix_one_2_2",
        "inkling_depth_mix_one_2_3",
        "inkling_depth_mix_one_2_4",
    ],
    [
        "inkling_depth_mix_one_3_1",
        "inkling_depth_mix_one_3_2",
        "inkling_depth_mix_one_3_3",
        "inkling_depth_mix_one_3_4",
    ],
    [
        "inkling_depth_mix_one_4_1",
        "inkling_depth_mix_one_4_2",
        "inkling_depth_mix_one_4_3",
        "inkling_depth_mix_one_4_4",
    ],
    [
        "inkling_depth_mix_one_5_1",
        "inkling_depth_mix_one_5_2",
        "inkling_depth_mix_one_5_3",
        "inkling_depth_mix_one_5_4",
    ],
    [
        "inkling_depth_mix_one_6_1",
        "inkling_depth_mix_one_6_2",
        "inkling_depth_mix_one_6_3",
        "inkling_depth_mix_one_6_4",
    ],
    [
        "inkling_depth_mix_one_7_1",
        "inkling_depth_mix_one_7_2",
        "inkling_depth_mix_one_7_3",
        "inkling_depth_mix_one_7_4",
    ],
    [
        "inkling_depth_mix_one_8_1",
        "inkling_depth_mix_one_8_2",
        "inkling_depth_mix_one_8_3",
        "inkling_depth_mix_one_8_4",
    ],
];

/// Threads per depth-mix block: wide for a few tokens, whose blocks are
/// all the work, narrow when many token blocks fill the GPU.
fn mix_threads(tokens: usize, hidden: usize) -> u32 {
    if tokens > 16 {
        256
    } else {
        ((hidden / 4).div_ceil(32) * 32).clamp(128, 1024) as u32
    }
}

/// [`depth_mix`] in one launch, reading each block in place: `blocks` are
/// `[tokens, hidden]` each, `current` `[tokens, hidden]`, `queries`
/// `[queries, hidden]`. With `norm` (`weight` `[queries or 1, hidden]`, eps),
/// each output row is RMS-normalized and weighted. Returns `[tokens, queries,
/// hidden]`, or `[queries, tokens, hidden]` when `group_major`.
pub(super) fn depth_mix_one(
    blocks: &[Tensor],
    current: &Tensor,
    queries: &Tensor,
    eps: f32,
    norm: Option<(&Tensor, f32)>,
    group_major: bool,
) -> Result<Tensor> {
    let device = device_of(current)?;
    let (tokens, hidden) = current.dims2()?;
    let (query_count, query_hidden) = queries.dims2()?;
    if blocks.is_empty()
        || blocks.len() > DEPTH_MIX_MAX_BLOCKS
        || query_count == 0
        || query_count > DEPTH_MIX_MAX_QUERIES
        || query_hidden != hidden
        || blocks.iter().any(|block| block.dims() != [tokens, hidden])
    {
        candle_core::bail!("Inkling 2 depth-mix geometry differs")
    }
    let (norm_eps, mut flags) = match norm {
        Some((weight, norm_eps)) => {
            let (rows, width) = weight.dims2()?;
            if width != hidden || (rows != 1 && rows != query_count) {
                candle_core::bail!("Inkling 2 depth-mix norm geometry differs")
            }
            (norm_eps, 1 | (i32::from(rows == 1 && query_count > 1) << 1))
        }
        None => (0.0, 0),
    };
    flags |= i32::from(group_major) << 2;
    let kept = [current, queries]
        .into_iter()
        .chain(blocks)
        .chain(norm.map(|(weight, _)| weight))
        .map(|tensor| dense(tensor, "depth-mix input"))
        .collect::<Result<Vec<_>>>()?;
    let holds = hold(&kept.iter().collect::<Vec<_>>());
    let stream = device.cuda_stream();
    let (address, _reads) = addresses(&holds, &device, &stream)?;
    // Source `blocks.len()` is the current row; later ones are never read.
    let mut sources = address[2..2 + blocks.len()].to_vec();
    sources.resize(8, address[0]);
    let norm_address = if norm.is_some() {
        address[address.len() - 1]
    } else {
        address[0]
    };
    let mut out = output(&device, tokens * query_count * hidden)?;
    let name = DEPTH_MIX_ONE_KERNELS[blocks.len() - 1][query_count - 1];
    let function = device.get_or_load_custom_func(name, MODULE, ptx()?)?;
    let (tokens_i, hidden_i) = (to_i32(tokens, "mix tokens")?, to_i32(hidden, "mix width")?);
    let scale = 1.0 / (hidden as f32).sqrt();
    let mut builder = function.builder();
    for source in &sources {
        builder.arg(source);
    }
    builder
        .arg(&address[1])
        .arg(&norm_address)
        .arg(&mut out)
        .arg(&tokens_i)
        .arg(&hidden_i)
        .arg(&eps)
        .arg(&scale)
        .arg(&norm_eps)
        .arg(&flags);
    let config = LaunchConfig {
        grid_dim: (to_u32(tokens.max(1), "mix tokens")?, 1, 1),
        block_dim: (mix_threads(tokens, hidden), 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: every input was checked for shape and held for the launch.
    unsafe { builder.launch(config) }.w()?;
    let shape = if group_major {
        Shape::from((query_count, tokens, hidden))
    } else {
        Shape::from((tokens, query_count, hidden))
    };
    Ok(wrap(&device, out, shape))
}

/// Most tokens the single-pass loop gates take; longer chunks multiply
/// their projections as matrices.
const GATE_FUSED_MAX_TOKENS: usize = 8;
/// Rows of [`gate_scratch`]: each holds one sum and one counter.
const GATE_SCRATCH_ROWS: usize = 64;

/// The address of this device's zeroed scratch for `inkling_gate_up_norm`
/// ([`GATE_SCRATCH_ROWS`] sums, then as many counters) and `inkling_route`
/// (one counter after them), which every launch leaves zeroed.
/// [`precompile`] allocates it before any captured step.
fn gate_scratch(device: &CudaDevice) -> Result<u64> {
    type Scratch = Vec<(candle_core::cuda_backend::DeviceId, CudaSlice<f32>)>;
    static SCRATCH: OnceLock<std::sync::Mutex<Scratch>> = OnceLock::new();
    let mut scratch = SCRATCH
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| candle_core::Error::Msg("Inkling 2 gate scratch lock poisoned".into()))?;
    let stream = device.cuda_stream();
    if let Some((_, slice)) = scratch.iter().find(|(id, _)| *id == device.id()) {
        return Ok(slice.device_ptr(&stream).0);
    }
    // The sums, the counters, then the router's counter.
    let slice = device.alloc_zeros::<f32>(2 * GATE_SCRATCH_ROWS + 1)?;
    let address = slice.device_ptr(&stream).0;
    scratch.push((device.id(), slice));
    Ok(address)
}

/// The Q, K and V loop gates for `x` `[tokens, 1 or 3, hidden]` (one input
/// shared by Q, K and V, or one each) and `messages` `[tokens, 3, hidden]`,
/// with `down` and `forget_gain` `[3, rank, 2 * hidden]`, each gated row then
/// RMS-normalized and times `norm` row `g` (`[3, hidden]`, or `[1, hidden]`
/// for all). `eps` serves both norms. Returns group-major `[3, tokens, hidden]`.
pub(super) fn message_gates(
    x: &Tensor,
    messages: &Tensor,
    down: &Tensor,
    forget_gain: &Tensor,
    eps: f32,
    norm: &Tensor,
) -> Result<Tensor> {
    gates_normed(x, messages, down, forget_gain, eps, norm, eps)
}

/// `groups` loop gates `(1 - tanh(F h)) * x + tanh(G h) * m`, `h = D [rms(x);
/// rms(m)]` (norms with `eps`), each output row then RMS-normalized with
/// `norm_eps` and times `norm` row `g`. `x` is `[tokens, 1 or groups,
/// hidden]`, `messages` `[tokens, groups, hidden]`, `down` and `forget_gain`
/// `[groups, rank, 2 * hidden]`, `norm` `[groups or 1, hidden]`. Returns
/// group-major `[groups, tokens, hidden]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn gates_normed(
    x: &Tensor,
    messages: &Tensor,
    down: &Tensor,
    forget_gain: &Tensor,
    eps: f32,
    norm: &Tensor,
    norm_eps: f32,
) -> Result<Tensor> {
    let device = device_of(x)?;
    let (tokens, x_groups, hidden) = x.dims3()?;
    let (groups, rank, width) = down.dims3()?;
    let (norm_rows, norm_hidden) = norm.dims2()?;
    if groups == 0
        || (x_groups != 1 && x_groups != groups)
        || messages.dims() != [tokens, groups, hidden]
        || width != 2 * hidden
        || forget_gain.dims() != [groups, rank, width]
        || norm_hidden != hidden
        || (norm_rows != 1 && norm_rows != groups)
    {
        candle_core::bail!("Inkling 2 loop gate geometry differs")
    }
    let kept = [x, messages, down, forget_gain, norm]
        .into_iter()
        .map(|tensor| dense(tensor, "loop gate input"))
        .collect::<Result<Vec<_>>>()?;
    let aligned = kept
        .iter()
        .all(|tensor| tensor.layout().start_offset().is_multiple_of(4));
    if tokens == 0
        || tokens > GATE_FUSED_MAX_TOKENS
        || groups * tokens > GATE_SCRATCH_ROWS
        || !hidden.is_multiple_of(4)
        || !aligned
    {
        // Matrix products over every row, normalized around them.
        let joined = rms_pair_groups(&kept[0], &kept[1], eps, groups)?;
        let projected = gemm_nt(&joined, &kept[2])?.matmul(&kept[3])?;
        return gate_combine_norm(
            &projected, &kept[0], &kept[1], &kept[4], groups, norm_eps, true, true,
        );
    }
    let scratch = gate_scratch(&device)?;
    let holds = hold(&kept.iter().collect::<Vec<_>>());
    let stream = device.cuda_stream();
    let (address, _reads) = addresses(&holds, &device, &stream)?;
    let mut reduced = output(&device, groups * tokens * rank)?;
    let (tokens_i, hidden_i, rank_i) = (
        to_i32(tokens, "gate tokens")?,
        to_i32(hidden, "gate width")?,
        to_i32(rank, "gate rank")?,
    );
    let (groups_i, x_groups_i, norm_rows_i) = (
        to_i32(groups, "gate groups")?,
        to_i32(x_groups, "gate inputs")?,
        to_i32(norm_rows, "gate norm rows")?,
    );
    let function = device.get_or_load_custom_func("inkling_gate_down", MODULE, ptx()?)?;
    let mut builder = function.builder();
    builder
        .arg(&address[0])
        .arg(&address[1])
        .arg(&address[2])
        .arg(&mut reduced)
        .arg(&tokens_i)
        .arg(&hidden_i)
        .arg(&rank_i)
        .arg(&eps)
        .arg(&groups_i)
        .arg(&x_groups_i);
    let config = LaunchConfig {
        grid_dim: (
            to_u32(rank, "gate rank")?,
            to_u32(groups, "gate groups")?,
            to_u32(tokens, "gate tokens")?,
        ),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: shapes and 16-byte alignment were checked; inputs are held.
    unsafe { builder.launch(config) }.w()?;

    let mut out = output(&device, groups * tokens * hidden)?;
    let function = device.get_or_load_custom_func("inkling_gate_up_norm", MODULE, ptx()?)?;
    let mut builder = function.builder();
    builder
        .arg(&reduced)
        .arg(&address[3])
        .arg(&address[0])
        .arg(&address[1])
        .arg(&address[4])
        .arg(&mut out)
        .arg(&scratch)
        .arg(&tokens_i)
        .arg(&hidden_i)
        .arg(&rank_i)
        .arg(&norm_eps)
        .arg(&groups_i)
        .arg(&x_groups_i)
        .arg(&norm_rows_i);
    let config = LaunchConfig {
        grid_dim: (
            to_u32(hidden.div_ceil(32), "gate width")?,
            to_u32(groups, "gate groups")?,
            to_u32(tokens, "gate tokens")?,
        ),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: `scratch` holds GATE_SCRATCH_ROWS rows, more than groups * tokens.
    unsafe { builder.launch(config) }.w()?;
    Ok(wrap(&device, out, (groups, tokens, hidden)))
}

/// `[rms(x) | rms(messages)]` for `groups` rows per token, group-major
/// `[groups, tokens, 2 * hidden]`; `x` is `[tokens, 1 or groups, hidden]`.
fn rms_pair_groups(x: &Tensor, messages: &Tensor, eps: f32, groups: usize) -> Result<Tensor> {
    let device = device_of(x)?;
    let (tokens, x_groups, hidden) = x.dims3()?;
    let rows = tokens * groups;
    let holds = hold(&[x, messages]);
    let stream = device.cuda_stream();
    let (address, _reads) = addresses(&holds, &device, &stream)?;
    let mut out = output(&device, rows * 2 * hidden)?;
    let function = device.get_or_load_custom_func("inkling_rms_pair_groups", MODULE, ptx()?)?;
    let (rows_i, hidden_i, groups_i, x_groups_i) = (
        to_i32(rows, "gate rows")?,
        to_i32(hidden, "gate width")?,
        to_i32(groups, "gate groups")?,
        to_i32(x_groups, "gate inputs")?,
    );
    let flags = 2i32;
    let mut builder = function.builder();
    builder
        .arg(&address[0])
        .arg(&address[1])
        .arg(&mut out)
        .arg(&rows_i)
        .arg(&hidden_i)
        .arg(&eps)
        .arg(&groups_i)
        .arg(&x_groups_i)
        .arg(&flags);
    let config = LaunchConfig {
        grid_dim: (to_u32(rows.max(1), "gate rows")?, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: the caller checked x [tokens, 1 or groups, hidden] and messages.
    unsafe { builder.launch(config) }.w()?;
    Ok(wrap(&device, out, (groups, tokens, 2 * hidden)))
}

/// `(1 - tanh(f)) * x + tanh(g) * m` for `forget_gain` rows `[f | g]`, each
/// then RMS-normalized and times `weight` row `group % rows`. `forget_gain`
/// is `[groups, tokens, 2 * hidden]` group-major when
/// `forget_gain_group_major`, else `[tokens * groups, 2 * hidden]`; `x` is
/// `[tokens, 1 or groups, hidden]` and `message` `[tokens, groups, hidden]`.
/// Returns `[groups, tokens, hidden]` when `group_major_out`, else `[tokens,
/// groups, hidden]`.
#[allow(clippy::too_many_arguments)]
fn gate_combine_norm(
    forget_gain: &Tensor,
    x: &Tensor,
    message: &Tensor,
    weight: &Tensor,
    groups: usize,
    eps: f32,
    forget_gain_group_major: bool,
    group_major_out: bool,
) -> Result<Tensor> {
    let device = device_of(x)?;
    let (tokens, x_groups, hidden) = x.dims3()?;
    let (weight_rows, weight_hidden) = weight.dims2()?;
    if groups == 0
        || (x_groups != 1 && x_groups != groups)
        || message.dims() != [tokens, groups, hidden]
        || forget_gain.elem_count() != tokens * groups * 2 * hidden
        || weight_hidden != hidden
        || weight_rows == 0
    {
        candle_core::bail!("Inkling 2 gate-norm geometry differs")
    }
    let kept = [forget_gain, x, message, weight]
        .into_iter()
        .map(|tensor| dense(tensor, "gate-norm input"))
        .collect::<Result<Vec<_>>>()?;
    let holds = hold(&kept.iter().collect::<Vec<_>>());
    let stream = device.cuda_stream();
    let (address, _reads) = addresses(&holds, &device, &stream)?;
    let rows = tokens * groups;
    let mut out = output(&device, rows * hidden)?;
    let function = device.get_or_load_custom_func("inkling_gate_combine_norm", MODULE, ptx()?)?;
    let (rows_i, hidden_i, groups_i, x_groups_i, weight_rows_i) = (
        to_i32(rows, "gate rows")?,
        to_i32(hidden, "gate width")?,
        to_i32(groups, "gate groups")?,
        to_i32(x_groups, "gate inputs")?,
        to_i32(weight_rows, "gate norm rows")?,
    );
    let flags = i32::from(forget_gain_group_major) | (i32::from(group_major_out) << 1);
    let mut builder = function.builder();
    builder
        .arg(&address[0])
        .arg(&address[1])
        .arg(&address[2])
        .arg(&address[3])
        .arg(&mut out)
        .arg(&rows_i)
        .arg(&hidden_i)
        .arg(&eps)
        .arg(&groups_i)
        .arg(&x_groups_i)
        .arg(&flags)
        .arg(&weight_rows_i);
    let config = LaunchConfig {
        grid_dim: (to_u32(rows.max(1), "gate rows")?, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: shapes were checked; inputs are held for the launch.
    unsafe { builder.launch(config) }.w()?;
    let shape = if group_major_out {
        Shape::from((groups, tokens, hidden))
    } else {
        Shape::from((tokens, groups, hidden))
    };
    Ok(wrap(&device, out, shape))
}

/// Where [`kv_finish`] writes the keys and values.
pub(super) enum KvOutput<'a> {
    /// New F32 tensors.
    Fresh,
    /// In place, as f16, into cache storage `[1, heads, capacity, dim]` from
    /// `row` on.
    Cache {
        k: &'a Tensor,
        v: &'a Tensor,
        row: usize,
    },
}

/// Keys and values ready for the cache, and the next conv states.
pub(super) struct KvFinished {
    /// `[1, heads, tokens, dim]`, L2-clipped per head (fresh output only).
    pub(super) k: Option<Tensor>,
    /// `[1, heads, tokens, dim]`, RMS-normalized per head (fresh output only).
    pub(super) v: Option<Tensor>,
    /// `[width - 1, heads * dim]` each.
    pub(super) k_state: Tensor,
    pub(super) v_state: Tensor,
}

/// The K and V short convolutions (`k_lin`, `v_lin` `[tokens, heads * dim]`
/// rows, after `past` `[width - 1, heads * dim]` states or zeros), with
/// static `weight`s `[heads * dim, width]` and optional per-token `taps`
/// `[tokens, heads * width]`, then K clipped to an L2 norm of `clip` and V
/// RMS-normalized per head, in one launch, written to `destination`.
#[allow(clippy::too_many_arguments)]
pub(super) fn kv_finish(
    k_lin: &Tensor,
    v_lin: &Tensor,
    past: Option<(&Tensor, &Tensor)>,
    k_weight: &Tensor,
    v_weight: &Tensor,
    taps: Option<(&Tensor, &Tensor)>,
    heads: usize,
    clip: f32,
    eps: f32,
    destination: KvOutput<'_>,
) -> Result<KvFinished> {
    let device = device_of(k_lin)?;
    let (tokens, channels) = k_lin.dims2()?;
    let (weight_channels, width) = k_weight.dims2()?;
    let dim = channels / heads.max(1);
    if heads == 0
        || heads * dim != channels
        || dim > 1024
        || width < 2
        || v_lin.dims() != [tokens, channels]
        || weight_channels != channels
        || v_weight.dims() != [channels, width]
        || past.is_some_and(|(k, v)| {
            k.dims() != [width - 1, channels] || v.dims() != [width - 1, channels]
        })
        || taps.is_some_and(|(k, v)| {
            k.dims() != [tokens, heads * width] || v.dims() != [tokens, heads * width]
        })
    {
        candle_core::bail!("Inkling 2 key/value finish geometry differs")
    }
    let strides = [
        row_stride(k_lin, "key input")?,
        row_stride(v_lin, "value input")?,
        taps.map_or(Ok(0), |(k, _)| row_stride(k, "key taps"))?,
        taps.map_or(Ok(0), |(_, v)| row_stride(v, "value taps"))?,
    ];
    let mut kept = vec![
        k_lin.clone(),
        v_lin.clone(),
        dense(k_weight, "key conv weight")?,
        dense(v_weight, "value conv weight")?,
    ];
    if let Some((k, v)) = past {
        kept.push(dense(k, "key state")?);
        kept.push(dense(v, "value state")?);
    }
    if let Some((k, v)) = taps {
        kept.push(k.clone());
        kept.push(v.clone());
    }
    let cache = match destination {
        KvOutput::Fresh => None,
        KvOutput::Cache { k, v, row } => {
            let (batch, cache_heads, capacity, cache_dim) = k.dims4()?;
            if batch != 1
                || (cache_heads, cache_dim) != (heads, dim)
                || v.dims() != k.dims()
                || k.dtype() != DType::F16
                || v.dtype() != DType::F16
                || !k.is_contiguous()
                || !v.is_contiguous()
                || row + tokens > capacity
            {
                candle_core::bail!("Inkling 2 key/value cache geometry differs")
            }
            kept.push(k.clone());
            kept.push(v.clone());
            Some((capacity, row))
        }
    };
    let holds = hold(&kept.iter().collect::<Vec<_>>());
    let stream = device.cuda_stream();
    let (address, _reads) = addresses(&holds, &device, &stream)?;
    let (past_k, past_v) = if past.is_some() {
        (address[4], address[5])
    } else {
        (address[0], address[1])
    };
    let taps_at = if past.is_some() { 6 } else { 4 };
    let (taps_k, taps_v) = if taps.is_some() {
        (address[taps_at], address[taps_at + 1])
    } else {
        (address[0], address[1])
    };
    let (k_cache, v_cache) = if cache.is_some() {
        (address[address.len() - 2], address[address.len() - 1])
    } else {
        (0, 0)
    };
    let history = width - 1;
    // In place, the main output is never written.
    let mut out = output(
        &device,
        if cache.is_some() {
            1
        } else {
            2 * tokens * channels
        },
    )?;
    let mut state = output(&device, 2 * history * channels)?;
    let terms = width + if taps.is_some() { width } else { 0 };
    let scale = 1.0 / ((terms + 1) as f32).sqrt();
    let flags = i32::from(taps.is_some())
        | (i32::from(past.is_some()) << 1)
        | (i32::from(cache.is_some()) << 2);
    let (capacity, row) = cache.unwrap_or((0, 0));
    let (capacity_i, row_i) = (
        to_i32(capacity, "cache capacity")?,
        to_i32(row, "cache row")?,
    );
    let (tokens_i, heads_i, dim_i, width_i) = (
        to_i32(tokens, "key/value tokens")?,
        to_i32(heads, "key/value heads")?,
        to_i32(dim, "key/value head width")?,
        to_i32(width, "conv width")?,
    );
    let [k_stride, v_stride, k_taps_stride, v_taps_stride] = [
        to_i32(strides[0], "key stride")?,
        to_i32(strides[1], "value stride")?,
        to_i32(strides[2], "tap stride")?,
        to_i32(strides[3], "tap stride")?,
    ];
    let function = device.get_or_load_custom_func("inkling_kv_finish", MODULE, ptx()?)?;
    let mut builder = function.builder();
    builder
        .arg(&address[0])
        .arg(&address[1])
        .arg(&past_k)
        .arg(&past_v)
        .arg(&address[2])
        .arg(&address[3])
        .arg(&taps_k)
        .arg(&taps_v)
        .arg(&mut out)
        .arg(&mut state)
        .arg(&k_cache)
        .arg(&v_cache)
        .arg(&tokens_i)
        .arg(&heads_i)
        .arg(&dim_i)
        .arg(&width_i)
        .arg(&k_stride)
        .arg(&v_stride)
        .arg(&k_taps_stride)
        .arg(&v_taps_stride)
        .arg(&scale)
        .arg(&clip)
        .arg(&eps)
        .arg(&flags)
        .arg(&capacity_i)
        .arg(&row_i);
    let config = LaunchConfig {
        grid_dim: (
            to_u32(heads, "key/value heads")?,
            to_u32(tokens.max(history), "key/value tokens")?,
            2,
        ),
        block_dim: (to_u32(dim.div_ceil(32) * 32, "key/value head width")?, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: shapes and row strides were checked; inputs are held.
    unsafe { builder.launch(config) }.w()?;
    let (k, v) = if cache.is_some() {
        (None, None)
    } else {
        let main = wrap(&device, out, (2, 1, heads, tokens, dim));
        (Some(main.get(0)?), Some(main.get(1)?))
    };
    let state = wrap(&device, state, (2, history, channels));
    Ok(KvFinished {
        k,
        v,
        k_state: state.get(0)?,
        v_state: state.get(1)?,
    })
}

/// Tokens per query-preparation block (`DINF_QUERY_PREP_TOKENS`).
const QUERY_PREP_TOKENS: usize = 8;

/// Column offsets of the queries, relative coefficients and sink logits in
/// the stacked query-side projection's rows.
pub(super) struct QuerySide {
    pub(super) q: usize,
    pub(super) rel: usize,
    pub(super) sink: usize,
}

/// From the stacked query-side projection `x` `[tokens, width]`: the queries
/// times `q_scale` `[heads, dim]`, clipped to an L2 norm of `q_clip`, times
/// `token_scale` (if any) and `q_post` (`[tokens, heads, dim]`); and the
/// attention bias `[tokens, heads, extent + 2]`, the relative coefficients
/// times `rel_scale` `[heads, rel]`, clipped to `rel_clip` and times
/// `token_scale`, multiplied into `basis` `[rel, extent + 1]`, then the sink.
#[allow(clippy::too_many_arguments)]
pub(super) fn query_prep(
    x: &Tensor,
    columns: &QuerySide,
    heads: usize,
    dim: usize,
    q_scale: &Tensor,
    rel_scale: &Tensor,
    token_scale: Option<&Tensor>,
    basis: &Tensor,
    q_clip: f32,
    rel_clip: f32,
    q_post: f32,
) -> Result<(Tensor, Tensor)> {
    let device = device_of(x)?;
    let (tokens, width) = x.dims2()?;
    let (rel, columns_plus) = basis.dims2()?;
    let extent = columns_plus.saturating_sub(1);
    if columns_plus == 0
        || rel == 0
        || rel > 32
        || heads == 0
        || q_scale.dims() != [heads, dim]
        || rel_scale.dims() != [heads, rel]
        || token_scale.is_some_and(|t| t.elem_count() != tokens)
        || columns.q + heads * dim > width
        || columns.rel + heads * rel > width
        || columns.sink + heads > width
    {
        candle_core::bail!("Inkling 2 query preparation geometry differs")
    }
    let x_stride = row_stride(x, "query-side rows")?;
    let mut kept = vec![
        x.clone(),
        dense(q_scale, "query scale")?,
        dense(rel_scale, "relative scale")?,
        dense(basis, "relative basis")?,
    ];
    if let Some(token_scale) = token_scale {
        kept.push(dense(token_scale, "token scale")?);
    }
    let holds = hold(&kept.iter().collect::<Vec<_>>());
    let stream = device.cuda_stream();
    let (address, _reads) = addresses(&holds, &device, &stream)?;
    let token_scale_address = if token_scale.is_some() {
        address[4]
    } else {
        address[0]
    };
    let threads = 32 * QUERY_PREP_TOKENS;
    let name = if rel <= 16 {
        "inkling_query_prep_16"
    } else {
        "inkling_query_prep_32"
    };
    let mut q_out = output(&device, tokens * heads * dim)?;
    let mut bias = output(&device, tokens * heads * (extent + 2))?;
    let function = device.get_or_load_custom_func(name, MODULE, ptx()?)?;
    let ints = [
        to_i32(tokens, "query tokens")?,
        to_i32(heads, "query heads")?,
        to_i32(dim, "query head width")?,
        to_i32(rel, "relative width")?,
        to_i32(extent, "relative extent")?,
        to_i32(x_stride, "query-side stride")?,
        to_i32(columns.q, "query column")?,
        to_i32(columns.rel, "relative column")?,
        to_i32(columns.sink, "sink column")?,
    ];
    let flags = i32::from(token_scale.is_some());
    let mut builder = function.builder();
    builder
        .arg(&address[0])
        .arg(&address[1])
        .arg(&address[2])
        .arg(&token_scale_address)
        .arg(&address[3])
        .arg(&mut q_out)
        .arg(&mut bias);
    for value in &ints {
        builder.arg(value);
    }
    builder.arg(&q_clip).arg(&rel_clip).arg(&q_post).arg(&flags);
    let config = LaunchConfig {
        grid_dim: (
            to_u32(heads, "query heads")?,
            to_u32(tokens.div_ceil(QUERY_PREP_TOKENS).max(1), "query tokens")?,
            to_u32((extent + 1).div_ceil(threads), "relative extent")?,
        ),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: shapes and column ranges were checked; inputs are held.
    unsafe { builder.launch(config) }.w()?;
    Ok((
        wrap(&device, q_out, (tokens, heads, dim)),
        wrap(&device, bias, (tokens, heads, extent + 2)),
    ))
}

/// Input of [`post_conv`]: a branch output, or routed expert rows to sum.
pub(super) enum PostInput<'a> {
    /// `[tokens, channels]` rows.
    Rows(&'a Tensor),
    /// `routed` `[tokens, routes, channels]` summed with `weights`
    /// `[tokens, routes]`.
    Routes {
        routed: &'a Tensor,
        weights: &'a Tensor,
    },
}

/// A branch's residual short convolution with static `weight`
/// `[channels, width]` after `past` (`[width - 1, channels]`, or zeros),
/// plus `residual` `[tokens, channels]` if given. Returns the output
/// `[tokens, channels]` and the next state `[width - 1, channels]`.
pub(super) fn post_conv(
    input: PostInput<'_>,
    past: Option<&Tensor>,
    weight: &Tensor,
    residual: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let (channels, width) = weight.dims2()?;
    let (tokens, mut kept, x_stride, routes) = match input {
        PostInput::Rows(x) => {
            let (tokens, x_channels) = x.dims2()?;
            if x_channels != channels {
                candle_core::bail!("Inkling 2 post-conv geometry differs")
            }
            (
                tokens,
                vec![x.clone()],
                row_stride(x, "post-conv input")?,
                0,
            )
        }
        PostInput::Routes { routed, weights } => {
            let (tokens, routes, routed_channels) = routed.dims3()?;
            if routed_channels != channels || weights.dims() != [tokens, routes] {
                candle_core::bail!("Inkling 2 routed post-conv geometry differs")
            }
            (
                tokens,
                vec![
                    dense(routed, "routed rows")?,
                    dense(weights, "route weights")?,
                ],
                channels,
                routes,
            )
        }
    };
    if width < 2
        || past.is_some_and(|p| p.dims() != [width - 1, channels])
        || residual.is_some_and(|r| r.dims() != [tokens, channels])
    {
        candle_core::bail!("Inkling 2 post-conv geometry differs")
    }
    let device = device_of(&kept[0])?;
    let routed = routes > 0;
    kept.push(dense(weight, "post-conv weight")?);
    let weight_at = kept.len() - 1;
    if let Some(past) = past {
        kept.push(dense(past, "post-conv state")?);
    }
    if let Some(residual) = residual {
        kept.push(dense(residual, "residual")?);
    }
    let holds = hold(&kept.iter().collect::<Vec<_>>());
    let stream = device.cuda_stream();
    let (address, _reads) = addresses(&holds, &device, &stream)?;
    let route_weights = if routed { address[1] } else { address[0] };
    let past_address = if past.is_some() {
        address[weight_at + 1]
    } else {
        address[0]
    };
    let residual_address = if residual.is_some() {
        address[address.len() - 1]
    } else {
        address[0]
    };
    let history = width - 1;
    let mut out = output(&device, tokens * channels)?;
    let mut state = output(&device, history * channels)?;
    let scale = 1.0 / ((width + 1) as f32).sqrt();
    let flags =
        i32::from(routed) | (i32::from(residual.is_some()) << 1) | (i32::from(past.is_some()) << 2);
    let ints = [
        to_i32(tokens, "post-conv tokens")?,
        to_i32(channels, "post-conv channels")?,
        to_i32(width, "post-conv width")?,
        to_i32(x_stride, "post-conv stride")?,
        to_i32(routes, "routes")?,
    ];
    let function = device.get_or_load_custom_func("inkling_post_conv", MODULE, ptx()?)?;
    let mut builder = function.builder();
    builder
        .arg(&address[0])
        .arg(&route_weights)
        .arg(&past_address)
        .arg(&address[weight_at])
        .arg(&residual_address)
        .arg(&mut out)
        .arg(&mut state);
    for value in &ints {
        builder.arg(value);
    }
    builder.arg(&scale).arg(&flags);
    // SAFETY: shapes and the input row stride were checked; inputs are held.
    unsafe { builder.launch(grid_1d((tokens + history) * channels, 256)?) }.w()?;
    Ok((
        wrap(&device, out, (tokens, channels)),
        wrap(&device, state, (history, channels)),
    ))
}

/// Most tokens [`route`] scores and picks in one launch.
const ROUTE_FUSED_MAX_TOKENS: usize = 8;
/// Warps (experts) per `inkling_route` block (`DINF_ROUTE_WARPS`).
const ROUTE_WARPS: usize = 8;

/// The sigmoid router for `x` `[tokens, hidden]`: logits against `router`
/// `[experts, hidden]` (F32), then [`sigmoid_top_k`]'s routes. Up to
/// [`ROUTE_FUSED_MAX_TOKENS`] tokens run as one launch.
pub(super) fn route(
    x: &Tensor,
    router: &Tensor,
    bias: &Tensor,
    top_k: usize,
    scale: f32,
) -> Result<(Tensor, Tensor)> {
    let device = device_of(x)?;
    let (tokens, hidden) = x.dims2()?;
    let (experts, router_hidden) = router.dims2()?;
    if router_hidden != hidden
        || bias.elem_count() != experts
        || experts > ROUTE_MAX_EXPERTS
        || top_k == 0
        || top_k > ROUTE_MAX_TOP_K
        || top_k > experts
    {
        candle_core::bail!("Inkling 2 route geometry is outside the kernel's range")
    }
    let kept = [
        dense(x, "router input")?,
        dense(router, "router")?,
        dense(bias, "router bias")?,
    ];
    let aligned = kept[..2]
        .iter()
        .all(|t| t.layout().start_offset().is_multiple_of(4));
    if tokens == 0 || tokens > ROUTE_FUSED_MAX_TOKENS || !hidden.is_multiple_of(4) || !aligned {
        let logits = kept[0].matmul(&kept[1].t()?)?;
        return sigmoid_top_k(&logits, &kept[2], top_k, scale);
    }
    // The counter slot after the gate norms' sums and counters.
    let counter = gate_scratch(&device)? + (2 * GATE_SCRATCH_ROWS * 4) as u64;
    let holds = hold(&kept.iter().collect::<Vec<_>>());
    let stream = device.cuda_stream();
    let (address, _reads) = addresses(&holds, &device, &stream)?;
    let mut logits = output(&device, tokens * experts)?;
    // SAFETY: the kernel's last block writes every id and weight.
    let mut ids = unsafe { device.alloc::<u32>(tokens * top_k)? };
    let mut weights = output(&device, tokens * top_k)?;
    let ints = [
        to_i32(tokens, "route tokens")?,
        to_i32(experts, "route experts")?,
        to_i32(hidden, "route width")?,
        to_i32(top_k, "route top-k")?,
    ];
    let function = device.get_or_load_custom_func("inkling_route", MODULE, ptx()?)?;
    let mut builder = function.builder();
    builder
        .arg(&address[0])
        .arg(&address[1])
        .arg(&address[2])
        .arg(&mut logits)
        .arg(&mut ids)
        .arg(&mut weights)
        .arg(&counter);
    for value in &ints {
        builder.arg(value);
    }
    builder.arg(&scale);
    let config = LaunchConfig {
        grid_dim: (
            to_u32(experts.div_ceil(ROUTE_WARPS), "route experts")?,
            1,
            1,
        ),
        block_dim: (32 * ROUTE_WARPS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: shapes and 16-byte alignment were checked; inputs are held,
    // and the counter slot lies inside the device scratch.
    unsafe { builder.launch(config) }.w()?;
    let ids = Tensor::from((
        candle_core::Storage::Cuda(CudaStorage::wrap_cuda_slice(ids, device.clone())),
        Shape::from((tokens, top_k)),
    ));
    Ok((ids, wrap(&device, weights, (tokens, top_k))))
}

/// `W_j x_j` for up to three pairs of one token `x_j` (`[1, cols]` F32) and
/// BF16 rows `W_j` (`[rows_j, cols]`, one `cols` for all), in one launch.
/// Returns each product as `[1, rows_j]`.
pub(super) fn bf16_matvec_many(pairs: &[(&Tensor, &Tensor)]) -> Result<Vec<Tensor>> {
    let Some(((first, first_w), _)) = pairs.split_first() else {
        return Ok(Vec::new());
    };
    let device = device_of(first)?;
    let cols = first_w.dim(1)?;
    if pairs.len() > 3
        || !cols.is_multiple_of(8)
        || pairs.iter().any(|(x, w)| {
            x.elem_count() != cols
                || w.dtype() != DType::BF16
                || w.rank() != 2
                || w.dim(1).ok() != Some(cols)
                || !w.is_contiguous()
                || !w.layout().start_offset().is_multiple_of(8)
        })
    {
        candle_core::bail!("Inkling 2 BF16 matvec geometry differs")
    }
    // The kernel reads x as float4s, so each starts 16-byte aligned.
    let inputs = pairs
        .iter()
        .map(|(x, _)| {
            let x = dense(x, "matvec input")?;
            if x.layout().start_offset().is_multiple_of(4) {
                Ok(x)
            } else {
                x.force_contiguous()
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let mut held_tensors: Vec<&Tensor> = inputs.iter().collect();
    held_tensors.extend(pairs.iter().map(|(_, w)| *w));
    let holds = hold(&held_tensors);
    let stream = device.cuda_stream();
    let (address, _reads) = {
        // Weights are BF16, which `addresses` does not take: read them here.
        let mut addresses = Vec::with_capacity(holds.len());
        let mut guards = Vec::with_capacity(holds.len());
        for (storage, layout) in &holds {
            let candle_core::Storage::Cuda(storage) = &**storage else {
                return cuda_only();
            };
            same_device(&device, storage)?;
            let (base, guard) = match storage.dtype() {
                DType::F32 => storage.as_cuda_slice::<f32>()?.device_ptr(&stream),
                DType::BF16 => storage.as_cuda_slice::<half::bf16>()?.device_ptr(&stream),
                dtype => candle_core::bail!("Inkling 2 matvec takes no {dtype:?} inputs"),
            };
            addresses.push(base + (layout.start_offset() * storage.dtype().size_in_bytes()) as u64);
            guards.push(guard);
        }
        (addresses, guards)
    };
    let rows = pairs
        .iter()
        .map(|(_, w)| w.dim(0))
        .collect::<candle_core::Result<Vec<_>>>()?;
    let mut outs = rows
        .iter()
        .map(|&rows| output(&device, rows))
        .collect::<Result<Vec<_>>>()?;
    let count = pairs.len();
    let function = device.get_or_load_custom_func("inkling_bf16_matvec3", MODULE, ptx()?)?;
    let rows_i = (0..3)
        .map(|j| to_i32(rows.get(j).copied().unwrap_or(0), "matvec rows"))
        .collect::<Result<Vec<_>>>()?;
    let null = 0u64;
    let mut builder = function.builder();
    for (j, out) in outs.iter_mut().enumerate() {
        builder
            .arg(&address[j])
            .arg(&address[count + j])
            .arg(out)
            .arg(&rows_i[j]);
    }
    for rows in &rows_i[count..] {
        builder.arg(&null).arg(&null).arg(&null).arg(rows);
    }
    let cols_i = to_i32(cols, "matvec width")?;
    builder.arg(&cols_i);
    let total: usize = rows.iter().sum();
    let config = LaunchConfig {
        grid_dim: (to_u32(total.div_ceil(4).max(1), "matvec rows")?, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: shapes and alignment were checked; inputs are held.
    unsafe { builder.launch(config) }.w()?;
    Ok(outs
        .into_iter()
        .zip(rows)
        .map(|(out, rows)| wrap(&device, out, (1, rows)))
        .collect())
}

/// Output tile and inner step of `inkling_gemm_nt_split`.
const GEMM_TILE: usize = 64;
const GEMM_K: usize = 32;

/// `a` `[groups, tokens, inner]` times `w` `[groups, rank, inner]`
/// transposed: `[groups, tokens, rank]`. The inner dimension splits across
/// blocks so a skinny product (the loop gates' rank 64) fills the GPU; the
/// partial sums are then added.
pub(super) fn gemm_nt(a: &Tensor, w: &Tensor) -> Result<Tensor> {
    let device = device_of(a)?;
    let (groups, tokens, inner) = a.dims3()?;
    let (w_groups, rank, w_inner) = w.dims3()?;
    let kept = [
        dense(a, "gate product input")?,
        dense(w, "gate product weight")?,
    ];
    let aligned = kept
        .iter()
        .all(|t| t.layout().start_offset().is_multiple_of(4));
    if w_groups != groups
        || w_inner != inner
        || tokens == 0
        || rank == 0
        || !inner.is_multiple_of(GEMM_K)
        || !aligned
    {
        return kept[0].matmul(&kept[1].transpose(1, 2)?);
    }
    let rank_tiles = rank.div_ceil(GEMM_TILE);
    let tiles = tokens.div_ceil(GEMM_TILE) * rank_tiles * groups;
    // About two blocks per SM of a large GPU, in steps that divide the inner chunks.
    let chunks = inner / GEMM_K;
    let mut splits = 256usize.div_ceil(tiles).clamp(1, 16).min(chunks);
    while !chunks.is_multiple_of(splits) {
        splits -= 1;
    }
    let holds = hold(&kept.iter().collect::<Vec<_>>());
    let stream = device.cuda_stream();
    let (address, _reads) = addresses(&holds, &device, &stream)?;
    let mut partial = output(&device, splits * groups * tokens * rank)?;
    let ints = [
        to_i32(tokens, "gate product tokens")?,
        to_i32(rank, "gate product rank")?,
        to_i32(inner, "gate product width")?,
        to_i32(groups, "gate product groups")?,
        to_i32(rank_tiles, "gate product rank")?,
    ];
    let function = device.get_or_load_custom_func("inkling_gemm_nt_split", MODULE, ptx()?)?;
    let mut builder = function.builder();
    builder.arg(&address[0]).arg(&address[1]).arg(&mut partial);
    for value in &ints {
        builder.arg(value);
    }
    let config = LaunchConfig {
        grid_dim: (
            to_u32(tokens.div_ceil(GEMM_TILE), "gate product tokens")?,
            to_u32(rank_tiles * groups, "gate product groups")?,
            to_u32(splits, "gate product splits")?,
        ),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: shapes, the inner split and 16-byte alignment were checked.
    unsafe { builder.launch(config) }.w()?;
    let partial = wrap(&device, partial, (splits, groups, tokens, rank));
    if splits == 1 {
        partial.squeeze(0)
    } else {
        partial.sum(0)
    }
}

/// `x` `[tokens, cols]` (F32) times BF16 rows `w` `[rows, cols]` transposed,
/// as one cuBLAS product with BF16 inputs and F32 accumulation and output
/// (no cast of the result). Returns `[tokens, rows]` F32.
pub(crate) fn bf16_matmul(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    use candle_core::cuda_backend::cudarc::cublas::{result::gemm_ex, sys};
    let device = device_of(x)?;
    let (tokens, cols) = x.dims2()?;
    let (rows, w_cols) = w.dims2()?;
    if w_cols != cols || w.dtype() != DType::BF16 || !w.is_contiguous() {
        candle_core::bail!("Inkling 2 BF16 product geometry differs")
    }
    let xb = x.to_dtype(DType::BF16)?.contiguous()?;
    let (m, n, k) = (
        to_i32(rows, "product rows")?,
        to_i32(tokens, "product tokens")?,
        to_i32(cols, "product width")?,
    );
    let out = output(&device, tokens * rows)?;
    let stream = device.cuda_stream();
    let blas = device.cublas_handle();
    let holds = hold(&[&xb, w]);
    let mut pointers = Vec::with_capacity(2);
    let mut guards = Vec::with_capacity(2);
    for (storage, layout) in &holds {
        let candle_core::Storage::Cuda(storage) = &**storage else {
            return cuda_only();
        };
        same_device(&device, storage)?;
        let (base, guard) = storage.as_cuda_slice::<half::bf16>()?.device_ptr(&stream);
        pointers.push(base + (layout.start_offset() * 2) as u64);
        guards.push(guard);
    }
    let (out_ptr, out_guard) = out.device_ptr(&stream);
    let (alpha, beta) = (1.0f32, 0.0f32);
    // Column-major: out^T [rows, tokens] = w (as [cols, rows])^T x^T.
    // SAFETY: pointers are live device buffers of the checked shapes, on the
    // stream the handle is bound to.
    unsafe {
        gemm_ex(
            *blas.handle(),
            sys::cublasOperation_t::CUBLAS_OP_T,
            sys::cublasOperation_t::CUBLAS_OP_N,
            m,
            n,
            k,
            (&alpha as *const f32).cast(),
            pointers[1] as *const std::ffi::c_void,
            sys::cudaDataType_t::CUDA_R_16BF,
            k,
            pointers[0] as *const std::ffi::c_void,
            sys::cudaDataType_t::CUDA_R_16BF,
            k,
            (&beta as *const f32).cast(),
            out_ptr as *mut std::ffi::c_void,
            sys::cudaDataType_t::CUDA_R_32F,
            m,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    }
    .map_err(|error| candle_core::Error::Msg(format!("Inkling 2 BF16 product: {error:?}")))?;
    drop(out_guard);
    drop(guards);
    Ok(wrap(&device, out, (tokens, rows)))
}
