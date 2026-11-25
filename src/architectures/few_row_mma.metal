// Few-row quantized matrix products (2..16 input rows) on 8x8 simdgroup
// matrices, for Apple GPUs without per-core neural accelerators (M1-M4).
//
// Adapted from llama.cpp a6ac41c75 (ggml/src/ggml-metal/kernels/mul_mv_mma.metal,
// from "metal : few-row MMA mat-mul", ggml-org/llama.cpp#29869):
// https://github.com/ggml-org/llama.cpp/blob/a6ac41c75/ggml/src/ggml-metal/kernels/mul_mv_mma.metal
// kernel_mul_mv_mma_blk (Q8_0), kernel_mul_mv_mma_gen (Q4_K, Q6_K, Q8_0),
// their tile and store helpers, and the Q4_K/Q6_K/Q8_0 dequantizers from
// kernels/dequantize.h. Changes: two-dimensional operands only (no batch
// broadcast), no fused residual, and the simdgroup count is a kernel argument
// is a kernel argument as well as a function constant.
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

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#define QK_K 256
#define QK_NL 16
#define QK8_0 32

typedef struct {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[QK_K/2];
} block_q4_K;

typedef struct {
    uchar ql[QK_K/2];
    uchar qh[QK_K/4];
    char scales[QK_K/16];
    half d;
} block_q6_K;

typedef struct {
    half d;
    char qs[QK8_0];
} block_q8_0;

// Specialized per pipeline, as upstream does: the simdgroup count and the
// input width give the loops compile-time bounds.
constant ushort FC_nsg  [[function_constant(0)]];
constant ulong  FC_ne00 [[function_constant(1)]];

// Host layout: dinf_few_row_mma_args in few_row_mma_metal.rs.
struct dinf_few_row_mma_args {
    int ne00;     // input width (K)
    int ne01;     // weight rows (output width)
    ulong nb01;   // bytes per weight row
    int ne11;     // input rows
    ulong nb11;   // bytes per input row
    int ne0;      // output row width
    int nsg;      // simdgroups per threadgroup, splitting K
};

template<typename T> struct dinf_elem;
template<> struct dinf_elem<float4x4> { typedef float type; };
template<> struct dinf_elem<half4x4>  { typedef half  type; };

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K * xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = (typename dinf_elem<type4x4>::type) (dl * (q[i] & mask) - ml);
    }
}

template <typename type4x4>
void dequantize_q6_K(device const block_q6_K * xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint16_t * ql = (device const uint16_t *)xb->ql;
    device const uint16_t * qh = (device const uint16_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint8_t shr_h = il>2 ? 2 : 0;
    const uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uint8_t shr_l = il>1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        typedef typename dinf_elem<type4x4>::type T;
        reg[i][0] = (T) (dl0 *  ((half)(q & 0xFF))       - ml);
        reg[i][1] = (T) (dl1 * ((float)(q & 0xFF00))     - ml);
        reg[i][2] = (T) (dl2 * ((float)(q & 0xFF0000))   - ml);
        reg[i][3] = (T) (dl3 * ((float)(q & 0xFF000000)) - ml);
    }
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 * xb, short il, thread type4x4 & reg) {
    device const packed_char4 * qs = (device const packed_char4 *) xb->qs;
    const float d = xb->d;

    float4x4 reg_f;
    for (int i = 0; i < 4; ++i) {
        reg_f[i] = float4(qs[4*il + i]) * d;
    }
    reg = (type4x4) reg_f;
}

// a lane of a few-row MMA tile: A fragment row fm, B fragment columns fn and fn + 1
struct mul_mv_mma_tile {
    short fm;
    short fn;
    int   i01;
    int   i11;
};

// the A fragment row and the first B fragment column that lane l holds in an 8x8 simdgroup matrix
inline short mul_mv_mma_lane_fm(ushort l) { return ((l/4) & 4) + ((l/2) % 4); }
inline short mul_mv_mma_lane_fn(ushort l) { return ((l/4) & 2)*2 + (l % 2)*2; }

template<short NT, short RT>
inline mul_mv_mma_tile mul_mv_mma_tile_init(uint3 tgpig, ushort tiisg) {
    mul_mv_mma_tile tile;
    tile.fm  = mul_mv_mma_lane_fm(tiisg);
    tile.fn  = mul_mv_mma_lane_fn(tiisg);
    tile.i01 = tgpig.x*(8*NT);
    tile.i11 = tgpig.y*(8*RT);
    return tile;
}

// the weight row of A fragment row fm in 8-row tile t, clamped to the last row
inline device const char * mul_mv_mma_src0_row(
        thread const mul_mv_mma_tile & tile, constant dinf_few_row_mma_args & args, device const char * src0, short t) {
    const int r = min(tile.i01 + 8*t + tile.fm, args.ne01 - 1);
    return src0 + (uint64_t) r*args.nb01;
}

// the input row of B fragment column fn + e in 8-row tile rt, clamped to the last row
inline device const float * mul_mv_mma_src1_row(
        thread const mul_mv_mma_tile & tile, constant dinf_few_row_mma_args & args, device const char * src1, short rt, short e) {
    const int r = min(tile.i11 + 8*rt + tile.fn + e, args.ne11 - 1);
    return (device const float *) (src1 + (uint64_t) r*args.nb11);
}

constexpr constant static ushort mma_f16_1024_bits = 0x6400;
constexpr constant static half   mma_f16_1024      = 1024.0h;

// the halves 1024 + q for integers q < 1024: exact normal values, unlike the subnormal q*2^-24 that Metal may flush to zero
inline half2 mul_mv_mma_1024_plus(ushort2 q) {
    return as_type<half2>(q | mma_f16_1024_bits);
}

// adds up the K slices of the simdgroups for an 8*NT x 8*RT output tile and writes it
template<short NT, short RT>
inline void mul_mv_mma_store(
        thread float (&acc)[RT][NT][2],
        constant dinf_few_row_mma_args & args,
        device char * dst,
        threadgroup char * shmem,
        thread const mul_mv_mma_tile & tile, ushort tiisg, ushort sgitg) {
    const short NSG = FC_nsg;

    threadgroup float * red = (threadgroup float *) shmem;

    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short t = 0; t < NT; ++t) {
            red[((sgitg*RT + rt)*NT + t)*64 + 2*tiisg + 0] = acc[rt][t][0];
            red[((sgitg*RT + rt)*NT + t)*64 + 2*tiisg + 1] = acc[rt][t][1];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float * dst_f32 = (device float *) dst;

    for (short idx = sgitg*32 + tiisg; idx < RT*NT*64; idx += NSG*32) {
        float sum = 0.0f;
        for (short sg = 0; sg < NSG; ++sg) {
            sum += red[sg*(RT*NT*64) + idx];
        }

        const short rt = idx/(NT*64);
        const short t  = (idx/64) % NT;
        const short l  = (idx % 64)/2;
        const short e  = idx % 2;

        const int r0 = tile.i01 + 8*t  + mul_mv_mma_lane_fm(l);
        const int r1 = tile.i11 + 8*rt + mul_mv_mma_lane_fn(l) + e;

        if (r0 < args.ne01 && r1 < args.ne11) {
            dst_f32[(uint64_t) r1*args.ne0 + r0] = sum;
        }
    }
}

// q8_0: A lane (m, j) holds qs bytes 4*j .. 4*j + 7 (j even); flipping the sign bit of a quant byte gives the unsigned q + 128.
// B lane k = fm holds src1 values b, b + 1 (b0) and b + 4, b + 5 (b1) of a block, b = 8*(k/2) + 2*(k%2)
struct mul_mv_mma_q8_0 {
    typedef block_q8_0 block;
    typedef ushort4    quants;

    // weights per block, and the float2 offset of b1 in a src1 block
    enum { qk = QK8_0, b1 = 2 };

    static short a_off(short fn) { return 1 + 2*fn; }
    static short b_off(short fm) { return 8*(fm/2) + 2*(fm%2); }

    static quants load(device const ushort * qs) { return ushort4(qs[0], qs[1], qs[2], qs[3]); }
    static quants prep(quants q) {
        constexpr ushort sign_bits = 0x8080;
        return q ^ ushort4(sign_bits);
    }
    static float2 prep_b1(float2 v) { return v; }

    static bool b1_step(short s) { return s >= 2; }
    static bool y_step (short s) { return s % 2 != 0; }

    static half2 frag(quants q, short s) {
        constexpr ushort byte_mask = 0x00FF;
        constexpr half   q_bias    = mma_f16_1024 + 128.0h;

        const ushort2 w  = s < 2 ? q.xy : q.zw;
        const ushort2 qq = s % 2 == 0 ? w : w >> 8;
        return mul_mv_mma_1024_plus(qq & byte_mask) - q_bias;
    }
};

template<typename Q, short NT>
inline void load_mma_blk_a(device const ushort * const x[NT], int off, short fn, thread typename Q::quants * q, thread float * d) {
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        device const ushort * qs = x[t] + off;
        q[t] = Q::load(qs);
        d[t] = as_type<half>(*(qs - Q::a_off(fn)));
    }
}

template<typename Q, short RT>
inline void load_mma_blk_b(device const float2 * const y[RT][2], int ib, thread float2 (*b0)[2], thread float2 (*b1)[2]) {
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short e = 0; e < 2; ++e) {
            b0[rt][e] = y[rt][e][ib*(Q::qk/2)];
            b1[rt][e] = y[rt][e][ib*(Q::qk/2) + Q::b1];
        }
    }
}

// few-row mat-mat for 32-weight block types: a threadgroup reads each weight once
// for 8*NT weight rows x 8*RT input rows, and its simdgroups split K
template<short NT, short RT, typename Q>
kernel void kernel_mul_mv_mma_blk(
        constant dinf_few_row_mma_args & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_nsg;

    const mul_mv_mma_tile tile = mul_mv_mma_tile_init<NT, RT>(tgpig, tiisg);

    device const ushort * x[NT];
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        x[t] = (device const ushort *) mul_mv_mma_src0_row(tile, args, src0, t) + Q::a_off(tile.fn);
    }

    device const float2 * y[RT][2];
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short e = 0; e < 2; ++e) {
            y[rt][e] = (device const float2 *) (mul_mv_mma_src1_row(tile, args, src1, rt, e) + Q::b_off(tile.fm));
        }
    }

    float acc[RT][NT][2] = {};

    const int nb = int(FC_ne00)/Q::qk;

    // a block is d, then the quants
    constexpr short us_blk = sizeof(typename Q::block)/2;

    typename Q::quants q[NT];
    float  d[NT];
    float2 b0[RT][2];
    float2 b1[RT][2];

    const int ib0 = min((int) sgitg, nb - 1);
    load_mma_blk_a<Q, NT>(x, ib0*us_blk, tile.fn, q, d);
    load_mma_blk_b<Q, RT>(y, ib0, b0, b1);

    for (int ib = sgitg; ib < nb; ib += NSG) {
        typename Q::quants qc[NT];
        float  dc[NT];
        float2 b0c[RT][2];
        float2 b1c[RT][2];

        FOR_UNROLL (short t = 0; t < NT; ++t) {
            qc[t] = Q::prep(q[t]);
            dc[t] = d[t];
        }
        FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
            FOR_UNROLL (short e = 0; e < 2; ++e) {
                b0c[rt][e] = b0[rt][e];
                b1c[rt][e] = Q::prep_b1(b1[rt][e]);
            }
        }

        const int ibn = min(ib + NSG, nb - 1);
        load_mma_blk_a<Q, NT>(x, ibn*us_blk, tile.fn, q, d);
        load_mma_blk_b<Q, RT>(y, ibn, b0, b1);

        simdgroup_float8x8 mp[RT][NT];
        FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
            FOR_UNROLL (short t = 0; t < NT; ++t) {
                mp[rt][t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
        }

        FOR_UNROLL (short s = 0; s < 4; ++s) {
            simdgroup_float8x8 mb[RT];
            FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                const float2 v0 = Q::b1_step(s) ? b1c[rt][0] : b0c[rt][0];
                const float2 v1 = Q::b1_step(s) ? b1c[rt][1] : b0c[rt][1];
                mb[rt].thread_elements()[0] = Q::y_step(s) ? v0.y : v0.x;
                mb[rt].thread_elements()[1] = Q::y_step(s) ? v1.y : v1.x;
            }

            FOR_UNROLL (short t = 0; t < NT; ++t) {
                const half2 h = Q::frag(qc[t], s);

                simdgroup_half8x8 ma;
                ma.thread_elements()[0] = h.x;
                ma.thread_elements()[1] = h.y;

                FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                    simdgroup_multiply_accumulate(mp[rt][t], ma, mb[rt], mp[rt][t]);
                }
            }
        }

        FOR_UNROLL (short t = 0; t < NT; ++t) {
            FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                acc[rt][t][0] = fma(dc[t], mp[rt][t].thread_elements()[0], acc[rt][t][0]);
                acc[rt][t][1] = fma(dc[t], mp[rt][t].thread_elements()[1], acc[rt][t][1]);
            }
        }
    }

    mul_mv_mma_store<NT, RT>(acc, args, dst, shmem, tile, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_mma_blk<4, 1, mul_mv_mma_q8_0>) mul_mv_mma_t;

template [[host_name("dinf_few_row_mma_q8_0_blk_nt1_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_blk<1, 1, mul_mv_mma_q8_0>;
template [[host_name("dinf_few_row_mma_q8_0_blk_nt2_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_blk<2, 1, mul_mv_mma_q8_0>;
template [[host_name("dinf_few_row_mma_q8_0_blk_nt4_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_blk<4, 1, mul_mv_mma_q8_0>;

// q4_K: derived from upstream's kernel_mul_mv_mma_q5_K_f32 (same block layout without the qh bits).
// the scale and min of sub-block j from the 12 packed bytes, held as 3 words
inline float2 mul_mv_mma_k4_scale_min(thread const uint * w, short j) {
    if (j < 4) {
        return float2((w[0] >> 8*j) & 63, (w[1] >> 8*j) & 63);
    }
    const short k = 8*(j - 4);
    return float2(((w[2] >> k) & 0xF) | (((w[0] >> (k + 6)) & 3) << 4), ((w[2] >> (k + 4)) & 0xF) | (((w[1] >> (k + 6)) & 3) << 4));
}

// the A fragments of one qs word: lo holds the 4-bit quants q of the low-nibble sub-block of a pair, hi holds 16*q of the high-nibble sub-block.
// step e0 takes bytes 0 and 2 of the word, step e1 takes bytes 1 and 3.
inline void mul_mv_mma_q4_K_frags(uint q, thread half2 * lo, thread half2 * hi) {
    const ushort2 qw = as_type<ushort2>(q);

    lo[0] = mul_mv_mma_1024_plus( qw       & 0x000F) - mma_f16_1024;
    lo[1] = mul_mv_mma_1024_plus((qw >> 8) & 0x000F) - mma_f16_1024;
    hi[0] = mul_mv_mma_1024_plus( qw       & 0x00F0) - mma_f16_1024;
    hi[1] = mul_mv_mma_1024_plus((qw >> 8) & 0x00F0) - mma_f16_1024;
}

struct mul_mv_mma_q4_K_a {
    uint2 q;
    uint  sc[3];
    uint  dm;
};

template<short NT>
inline void load_q4_K_mma_a(device const block_q4_K * const x[NT], int ip, short fn, thread mul_mv_mma_q4_K_a * a) {
    constexpr short pairs = QK_K/64;

    FOR_UNROLL (short t = 0; t < NT; ++t) {
        device const block_q4_K * xb = x[t] + ip/pairs;
        device const uint * sp = (device const uint *) xb->scales;

        a[t].q     = *((device const uint2 *) (xb->qs + 32*(ip%pairs)) + fn/2);
        a[t].sc[0] = sp[0];
        a[t].sc[1] = sp[1];
        a[t].sc[2] = sp[2];
        a[t].dm    = *((device const uint *) xb);
    }
}

// xs[rt][e][h]: src1 values of sub-block h of the pair, for the steps (b, b + 1) and (b + 4, b + 5)
template<short RT>
inline void load_k4_mma_b(device const float2 * const y[RT][2], int ip, thread float2 (*xs)[2][2][2]) {
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short e = 0; e < 2; ++e) {
            FOR_UNROLL (short h = 0; h < 2; ++h) {
                xs[rt][e][h][0] = y[rt][e][ip*32 + h*16 + 0];
                xs[rt][e][h][1] = y[rt][e][ip*32 + h*16 + 2];
            }
        }
    }
}

// few-row mat-mat for q4_K over pairs of 32-weight sub-blocks: A lane (m, j) holds qs bytes 4*j .. 4*j + 7 of a pair (j even).
// the 1/16 of the in-place high nibbles and the sub-block scales go into the accumulation; the mins are removed with the src1 sums.
template<short NT, short RT>
kernel void kernel_mul_mv_mma_q4_K_f32(
        constant dinf_few_row_mma_args & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_nsg;

    constexpr float hi_scale = 1.0f/16;
    constexpr short pairs    = QK_K/64;

    const mul_mv_mma_tile tile = mul_mv_mma_tile_init<NT, RT>(tgpig, tiisg);

    device const block_q4_K * x[NT];
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        x[t] = (device const block_q4_K *) mul_mv_mma_src0_row(tile, args, src0, t);
    }

    // B lane k = fm holds src1 values b, b + 1, b + 4, b + 5 of each sub-block, b = 8*(k/2) + 2*(k%2)
    device const float2 * y[RT][2];
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short e = 0; e < 2; ++e) {
            y[rt][e] = (device const float2 *) (mul_mv_mma_src1_row(tile, args, src1, rt, e) + 8*(tile.fm/2) + 2*(tile.fm%2));
        }
    }

    float acc[RT][NT][2] = {};

    const int np = int(FC_ne00)/64;

    mul_mv_mma_q4_K_a an[NT];
    float2 xn[RT][2][2][2];

    const int ip0 = min((int) sgitg, np - 1);
    load_q4_K_mma_a<NT>(x, ip0, tile.fn, an);
    load_k4_mma_b<RT>(y, ip0, xn);

    for (int ip = sgitg; ip < np; ip += NSG) {
        const short p = ip%pairs;

        mul_mv_mma_q4_K_a ac[NT];
        float2 xs[RT][2][2][2];
        FOR_UNROLL (short t = 0; t < NT; ++t) {
            ac[t] = an[t];
        }
        FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
            FOR_UNROLL (short e = 0; e < 2; ++e) {
                FOR_UNROLL (short h = 0; h < 2; ++h) {
                    xs[rt][e][h][0] = xn[rt][e][h][0];
                    xs[rt][e][h][1] = xn[rt][e][h][1];
                }
            }
        }

        const int ipn = min(ip + NSG, np - 1);
        load_q4_K_mma_a<NT>(x, ipn, tile.fn, an);
        load_k4_mma_b<RT>(y, ipn, xn);

        float c[RT][2][2];
        FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
            FOR_UNROLL (short e = 0; e < 2; ++e) {
                FOR_UNROLL (short h = 0; h < 2; ++h) {
                    float u = xs[rt][e][h][0].x + xs[rt][e][h][0].y + xs[rt][e][h][1].x + xs[rt][e][h][1].y;
                    u += simd_shuffle_xor(u, 2);
                    u += simd_shuffle_xor(u, 4);
                    u += simd_shuffle_xor(u, 16);
                    c[rt][e][h] = u;
                }
            }
        }

        FOR_UNROLL (short t = 0; t < NT; ++t) {
            const float2 dm  = float2(as_type<half2>(ac[t].dm));
            const float2 sm0 = mul_mv_mma_k4_scale_min(ac[t].sc, 2*p + 0);
            const float2 sm1 = mul_mv_mma_k4_scale_min(ac[t].sc, 2*p + 1);

            half2 a[2][2][2];
            mul_mv_mma_q4_K_frags(ac[t].q.x, a[0][0], a[1][0]);
            mul_mv_mma_q4_K_frags(ac[t].q.y, a[0][1], a[1][1]);

            FOR_UNROLL (short hh = 0; hh < 2; ++hh) {
                simdgroup_float8x8 mp[RT];
                FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                    mp[rt] = make_filled_simdgroup_matrix<float, 8>(0.0f);
                }

                FOR_UNROLL (short s = 0; s < 4; ++s) {
                    simdgroup_half8x8 ma;
                    ma.thread_elements()[0] = a[hh][s/2][s%2].x;
                    ma.thread_elements()[1] = a[hh][s/2][s%2].y;

                    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                        simdgroup_float8x8 mb;
                        mb.thread_elements()[0] = s % 2 == 0 ? xs[rt][0][hh][s/2].x : xs[rt][0][hh][s/2].y;
                        mb.thread_elements()[1] = s % 2 == 0 ? xs[rt][1][hh][s/2].x : xs[rt][1][hh][s/2].y;

                        simdgroup_multiply_accumulate(mp[rt], ma, mb, mp[rt]);
                    }
                }

                const float2 smh = hh == 0 ? sm0 : sm1;
                const float  dsc = dm.x*smh.x*(hh == 0 ? 1.0f : hi_scale);
                const float  dmn = dm.y*smh.y;

                FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                    acc[rt][t][0] = fma(dsc, mp[rt].thread_elements()[0], fma(-dmn, c[rt][0][hh], acc[rt][t][0]));
                    acc[rt][t][1] = fma(dsc, mp[rt].thread_elements()[1], fma(-dmn, c[rt][1][hh], acc[rt][t][1]));
                }
            }
        }
    }

    mul_mv_mma_store<NT, RT>(acc, args, dst, shmem, tile, tiisg, sgitg);
}

template [[host_name("dinf_few_row_mma_q4_K_blk_nt1_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_q4_K_f32<1, 1>;
template [[host_name("dinf_few_row_mma_q4_K_blk_nt2_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_q4_K_f32<2, 1>;
template [[host_name("dinf_few_row_mma_q4_K_blk_nt4_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_q4_K_f32<4, 1>;
template [[host_name("dinf_few_row_mma_q4_K_blk_nt1_rt2")]] kernel mul_mv_mma_t kernel_mul_mv_mma_q4_K_f32<1, 2>;
template [[host_name("dinf_few_row_mma_q4_K_blk_nt2_rt2")]] kernel mul_mv_mma_t kernel_mul_mv_mma_q4_K_f32<2, 2>;
template [[host_name("dinf_few_row_mma_q4_K_blk_nt4_rt2")]] kernel mul_mv_mma_t kernel_mul_mv_mma_q4_K_f32<4, 2>;

// few-row mat-mat for any type with a 16-weight dequantizer: a lane dequantizes 16 consecutive weights of a 64-weight chunk once for all input rows.
// MMA step s at MMA-k index j reads chunk weight 16*(j/2) + 8*(j%2) + s.
template<short NT, short RT, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
kernel void kernel_mul_mv_mma_gen(
        constant dinf_few_row_mma_args & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_nsg;

    const mul_mv_mma_tile tile = mul_mv_mma_tile_init<NT, RT>(tgpig, tiisg);

    device const block_q * x[NT];
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        x[t] = (device const block_q *) mul_mv_mma_src0_row(tile, args, src0, t);
    }

    // B lane j = fm reads chunk values 16*(fm/2) + 8*(fm%2) .. +7
    device const float4 * y[RT][2];
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short e = 0; e < 2; ++e) {
            y[rt][e] = (device const float4 *) mul_mv_mma_src1_row(tile, args, src1, rt, e) + 4*(tile.fm/2) + 2*(tile.fm%2);
        }
    }

    simdgroup_float8x8 mc[RT][NT];
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short t = 0; t < NT; ++t) {
            mc[rt][t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
    }

    const int nch = int(FC_ne00)/64;

    for (int g = sgitg; g < nch; g += NSG) {
        simdgroup_float8x8 mb[RT][8];
        FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
            const float4 a0 = y[rt][0][16*g + 0];
            const float4 a1 = y[rt][0][16*g + 1];
            const float4 b0 = y[rt][1][16*g + 0];
            const float4 b1 = y[rt][1][16*g + 1];
            FOR_UNROLL (short s = 0; s < 4; ++s) {
                mb[rt][s    ].thread_elements()[0] = a0[s];
                mb[rt][s    ].thread_elements()[1] = b0[s];
                mb[rt][s + 4].thread_elements()[0] = a1[s];
                mb[rt][s + 4].thread_elements()[1] = b1[s];
            }
        }

        const int ci = 4*g + tile.fn/2;

        FOR_UNROLL (short t = 0; t < NT; ++t) {
            float4x4 w;
            dequantize_func(x[t] + ci/nl, ci%nl, w);

            FOR_UNROLL (short s = 0; s < 8; ++s) {
                simdgroup_float8x8 ma;
                ma.thread_elements()[0] = w[s/4    ][s%4];
                ma.thread_elements()[1] = w[s/4 + 2][s%4];

                FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                    simdgroup_multiply_accumulate(mc[rt][t], ma, mb[rt][s], mc[rt][t]);
                }
            }
        }
    }

    float acc[RT][NT][2];
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short t = 0; t < NT; ++t) {
            acc[rt][t][0] = mc[rt][t].thread_elements()[0];
            acc[rt][t][1] = mc[rt][t].thread_elements()[1];
        }
    }

    mul_mv_mma_store<NT, RT>(acc, args, dst, shmem, tile, tiisg, sgitg);
}

#define MUL_MV_MMA_GEN(tname, bq, nl, deq) \
template [[host_name("dinf_few_row_mma_" tname "_nt1_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<1, 1, bq, nl, deq>; \
template [[host_name("dinf_few_row_mma_" tname "_nt2_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<2, 1, bq, nl, deq>; \
template [[host_name("dinf_few_row_mma_" tname "_nt4_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<4, 1, bq, nl, deq>; \
template [[host_name("dinf_few_row_mma_" tname "_nt1_rt2")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<1, 2, bq, nl, deq>; \
template [[host_name("dinf_few_row_mma_" tname "_nt2_rt2")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<2, 2, bq, nl, deq>; \
template [[host_name("dinf_few_row_mma_" tname "_nt4_rt2")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<4, 2, bq, nl, deq>;

MUL_MV_MMA_GEN("q4_K", block_q4_K, QK_NL, dequantize_q4_K<float4x4>)
MUL_MV_MMA_GEN("q6_K", block_q6_K, QK_NL, dequantize_q6_K<float4x4>)
MUL_MV_MMA_GEN("q8_0", block_q8_0, 2,     dequantize_q8_0<float4x4>)

#undef MUL_MV_MMA_GEN

// ---------------------------------------------------------------------------
// Prompt GEMM (17+ input rows).

// Whole 64x64 output tiles skip the bounds-checked store.
constant bool FC_mm_bc_out [[function_constant(11)]];

// Host layout: MmArgs in few_row_mma_metal.rs.
struct dinf_mm_args {
    int   ne00;  // K
    ulong nb01;  // bytes per weight row
    ulong nb11;  // bytes per input row
    int   ne0;   // weight rows (output width)
    int   ne1;   // input rows
};

// ---------------------------------------------------------------------------
// Prompt GEMM in MLX's steel structure with a dequantizing weight loader.
// A tiled quantized GEMM can dequantize on the fly into 8x8-blocked
// threadgroup tiles; the steel f16 GEMM vendored in Candle (mlx_gemm.metal,
// MIT, Apple) pads threadgroup rows by 16 bytes, interleaves each
// simdgroup's 8x8 fragments across the tile and reads fragments straight
// into registers. This kernel combines them, following BlockMMA for
// transpose_a = false, transpose_b = true:
// A is the input (tokens x K, f32 -> half), B^T the quantized weights
// (out rows x K), each 16-weight chunk dequantized once per tile.

template<typename TI, short BN, short BM, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void kernel_mul_mm_steel(
        constant dinf_mm_args & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {

    // BM tokens x BN weight rows. 128-token tiles (8 simdgroups) and half
    // MMA fragments measured the same: at this point the kernel
    // runs at the GPU's matrix rate.
    constexpr short BK = 32;
    constexpr short WM = BM/32;
    constexpr short WN = 2;

    constexpr short LDA = BK + 8; // half elements, 16 bytes of padding
    constexpr short LDB = BK + 8;

    constexpr short TM_stride = 8*WM;
    constexpr short TN_stride = 8*WN;
    constexpr short TM = BM/TM_stride; // 4
    constexpr short TN = BN/TN_stride; // 2 or 4

    threadgroup half * As = (threadgroup half *) shmem;          // [BM][LDA]
    threadgroup half * Bs = As + BM*LDA;                          // [BN][LDB]

    const int m0 = tgpig.y*BM;
    const int n0 = tgpig.x*BN;

    // A loader: two threads per token row, 16 K values each
    const short a_row = tiitg/2;
    const short a_col = 16*(tiitg%2);
    const int   a_src_row = min(m0 + a_row, args.ne1 - 1);
    device const TI * a_src = (device const TI *) (src1 + (uint64_t) a_src_row*args.nb11) + a_col;
    threadgroup half * a_dst = As + a_row*LDA + a_col;

    // B loader: two threads per weight row, 16 weights each (all 128
    // threads for BN = 64, the first 64 for BN = 32)
    const short b_row = (tiitg%(2*BN))/2;
    const short b_chunk = tiitg%2;
    const int   b_src_row = min(n0 + b_row, args.ne0 - 1);
    device const block_q * b_src = (device const block_q *) (src0 + (uint64_t) b_src_row*args.nb01);
    threadgroup half * b_dst = Bs + b_row*LDB + 16*b_chunk;

    // fragment position (BlockMMA)
    const short tm = 8*(sgitg/WN);
    const short tn = 8*(sgitg%WN);
    const short qid = tiisg/4;
    const short sm = (qid & 4) + (tiisg/2) % 4;
    const short sn = (qid & 2)*2 + (tiisg%2)*2;

    const short As_offset = sn + (tm + sm)*LDA;
    const short Bs_offset = (tn + sn)*LDB + sm;

    simdgroup_float8x8 Asimd[TM];
    simdgroup_float8x8 Bsimd[TN];
    simdgroup_float8x8 results[TM*TN];
    for (short i = 0; i < TM*TN; i++) {
        results[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int kk = 0; kk < args.ne00; kk += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short v = 0; v < 4; v++) {
            *((threadgroup half4 *) (a_dst + 4*v)) = half4(*((device const vec<TI, 4> *) (a_src + kk) + v));
        }

        if (tiitg < 2*BN) {
            const int k = kk + 16*b_chunk;
            half4x4 w;
            dequantize_func(b_src + k/(16*nl), (k/16) % nl, w);
            FOR_UNROLL (short v = 0; v < 4; v++) {
                *((threadgroup half4 *) (b_dst + 4*v)) = w[v];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * as = As + As_offset;
        threadgroup const half * bs = Bs + Bs_offset;

        FOR_UNROLL (short k8 = 0; k8 < BK; k8 += 8) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < TM; i++) {
                Asimd[i].thread_elements()[0] = float(as[i*TM_stride*LDA + 0]);
                Asimd[i].thread_elements()[1] = float(as[i*TM_stride*LDA + 1]);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short j = 0; j < TN; j++) {
                Bsimd[j].thread_elements()[0] = float(bs[j*TN_stride*LDB + 0]);
                Bsimd[j].thread_elements()[1] = float(bs[j*TN_stride*LDB + LDB]);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < TM; i++) {
                FOR_UNROLL (short j = 0; j < TN; j++) {
                    const short js = (i % 2) ? (TN - 1 - j) : j;
                    simdgroup_multiply_accumulate(results[i*TN + js], Asimd[i], Bsimd[js], results[i*TN + js]);
                }
            }

            as += 8;
            bs += 8;
        }
    }

    // D[m][n], m = token, n = weight row
    device float * D = (device float *) dst;
    const int dm = m0 + tm + sm;
    const int dn = n0 + tn + sn;
    const bool whole = !FC_mm_bc_out || (m0 + BM <= args.ne1 && n0 + BN <= args.ne0);
    FOR_UNROLL (short i = 0; i < TM; i++) {
        FOR_UNROLL (short j = 0; j < TN; j++) {
            thread const auto & acc = results[i*TN + j].thread_elements();
            const int m = dm + i*TM_stride;
            const int n = dn + j*TN_stride;
            device float * d = D + (uint64_t) m*args.ne0 + n;
            if (whole) {
                d[0] = acc[0];
                d[1] = acc[1];
            } else if (m < args.ne1) {
                if (n     < args.ne0) d[0] = acc[0];
                if (n + 1 < args.ne0) d[1] = acc[1];
            }
        }
    }
}

typedef decltype(kernel_mul_mm_steel<half, 64, 64, block_q4_K, QK_NL, dequantize_q4_K<half4x4>>) mul_mm_steel_t;

#define DINF_MMS(suffix, TI, BN, BM) \
template [[host_name("dinf_mms" suffix "_q4_K")]] kernel mul_mm_steel_t kernel_mul_mm_steel<TI, BN, BM, block_q4_K, QK_NL, dequantize_q4_K<half4x4>>; \
template [[host_name("dinf_mms" suffix "_q6_K")]] kernel mul_mm_steel_t kernel_mul_mm_steel<TI, BN, BM, block_q6_K, QK_NL, dequantize_q6_K<half4x4>>; \
template [[host_name("dinf_mms" suffix "_q8_0")]] kernel mul_mm_steel_t kernel_mul_mm_steel<TI, BN, BM, block_q8_0, 2,     dequantize_q8_0<half4x4>>;

// f16 input: each threadgroup reads its input rows across all of K, so f32
// input doubled the dominant cache traffic. The f32 version rounded to half
// on load anyway, so the products are the same. 64 weight rows per tile keep
// all threads dequantizing and halve input reuse again (faster than 32 on
// Qwen3.8's shapes).
DINF_MMS("", half, 64, 64)

#undef DINF_MMS

// f32 -> f16 copy of a prompt's input rows for the steel kernel.
kernel void dinf_cast_f32_f16(
        device const float4 * src,
        device       half4  * dst,
        constant     uint   & n4,
        uint i [[thread_position_in_grid]]) {
    if (i < n4) {
        dst[i] = half4(src[i]);
    }
}
