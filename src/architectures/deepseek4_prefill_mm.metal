// Adapted from ggml-org/llama.cpp de5122d879d00d2895c36be7e0cee94105aeeabd,
// ggml/src/ggml-metal/kernels/mul_mm.metal kernel_mul_mm/kernel_mul_mm_id.
// 64x32 output tile, K32, four SIMDgroups, half operands/F32 accumulators.
// Raw per-expert bindings replace upstream GPU-address tables. Canonical IDs
// remain GPU-generated; no four-occurrence packets or F32 operand fragments.
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
#include <metal_simdgroup_matrix>
#ifdef DINF_PREFILL_TENSOR
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
struct dinf_mm_args { uint k, m, n, row_bytes, mode, expert, kind, stride; };
struct dinf_mm_binding {uint expert, kind, count, pad;};
template<bool IQ1>
kernel void dinf_prefill_mm(device const uchar *w0 [[buffer(0)]],
    device const float *input [[buffer(1)]], device float *output [[buffer(2)]],
    device const uint *map [[buffer(3)]], constant dinf_mm_args &base [[buffer(4)]],
    constant dinf_mm_binding *bindings [[buffer(5)]],
    device const uchar *w1 [[buffer(6)]], device const uchar *w2 [[buffer(7)]],
    device const uchar *w3 [[buffer(8)]], device const uchar *w4 [[buffer(9)]],
    device const uchar *w5 [[buffer(10)]], device const uchar *w6 [[buffer(11)]],
    device const uchar *w7 [[buffer(12)]], device const uchar *w8 [[buffer(13)]],
    device const uchar *w9 [[buffer(14)]], device const uchar *w10 [[buffer(15)]],
    device const uchar *w11 [[buffer(16)]],
    uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    dinf_mm_args a=base;
    device const uchar *weights=w0;
    if (IQ1 && a.mode!=0) {
        const dinf_mm_binding b=bindings[group.z];
        a.expert=b.expert; a.kind=b.kind;
        if(a.expert>=384 || b.count>6*a.n || map[a.expert*a.stride]!=b.count) return;
        switch(group.z) {
            case 1:weights=w1;break;case 2:weights=w2;break;case 3:weights=w3;break;
            case 4:weights=w4;break;case 5:weights=w5;break;case 6:weights=w6;break;
            case 7:weights=w7;break;case 8:weights=w8;break;case 9:weights=w9;break;
            case 10:weights=w10;break;case 11:weights=w11;break;
        }
    }
    threadgroup half sa[2048];
    threadgroup half sb[1024];
    threadgroup float result[2048];
    const uint r0 = group.x*64, r1 = group.y*32;
    const uint count = a.mode == 0 ? a.n : map[a.expert*a.stride];
    // Uniform group exit before barriers and before unsigned tail subtraction.
    if(r0>=a.m || r1>=count) return;
    const uint nr0 = min(64u,a.m-r0), nr1 = min(32u,count-r1);
    const uint lr0 = min(uint(tid)/2,nr0-1), il0 = tid%2;
    const uint lr1 = min(uint(tid)/4,nr1-1), iy = 8*(tid%4);
    const uint canonical = a.mode == 0 ? r1+lr1 : map[a.expert*a.stride+1+r1+lr1];
    const uint source = a.mode == 1 ? canonical/6 : canonical;
#ifdef DINF_PREFILL_TENSOR
    auto tA=tensor<threadgroup half,dextents<int32_t,2>,tensor_inline>(sa,dextents<int32_t,2>(32,64));
    auto tB0=tensor<threadgroup half,dextents<int32_t,2>,tensor_inline>(sb,dextents<int32_t,2>(32,16));
    auto tB1=tensor<threadgroup half,dextents<int32_t,2>,tensor_inline>(sb+16*32,dextents<int32_t,2>(32,16));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(16,64,32,false,true,false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cT0=mm.get_destination_cooperative_tensor<decltype(tA),decltype(tB0),float>();
    auto cT1=mm.get_destination_cooperative_tensor<decltype(tA),decltype(tB1),float>();
#else
    simdgroup_half8x8 ma[4], mb[2];
    simdgroup_float8x8 mc[8];
    for (ushort i=0;i<8;++i) mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
#endif
    for (uint kk=0;kk<a.k;kk+=32) {
        half4x4 decoded;
        if (IQ1) {
            device const block_iq1_s *w=(device const block_iq1_s *)(weights+size_t(r0+lr0)*a.row_bytes);
            dequantize_iq1_s(w+(kk+16*il0)/256,short(((kk+16*il0)/16)%16),decoded);
        } else {
            device const block_q8_0 *w=(device const block_q8_0 *)(weights+size_t(r0+lr0)*a.row_bytes);
            dequantize_q8_0(w+kk/32,short(il0),decoded);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort i=0;i<16;++i) {
#ifdef DINF_PREFILL_TENSOR
            sa[(tid/2)*32+16*il0+i]=decoded[i/4][i%4];
#else
            const ushort sx=2*il0+i/8, sy=(tid/2)/8, lx=(tid/2)%8, ly=i%8;
            sa[64*(8*sx+sy)+8*ly+lx]=decoded[i/4][i%4];
#endif
        }
        for (ushort i=0;i<8;++i) {
            const ushort sx=tid%4, sy=(tid/4)/8, ly=(tid/4)%8;
            // IEEE conversion deliberately preserves NaN/Inf and overflows to
            // infinity, never clamps. Model-range qualification is mandatory.
#ifdef DINF_PREFILL_TENSOR
            sb[(tid/4)*32+iy+i]=half(input[size_t(source)*a.k+kk+iy+i]);
#else
            sb[64*(4*sx+sy)+8*ly+i]=half(input[size_t(source)*a.k+kk+iy+i]);
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#ifdef DINF_PREFILL_TENSOR
        auto sA=tA.slice(0,0);
        auto sB0=tB0.slice(0,0);
        mm.run(sB0,sA,cT0);
        if(nr1>16) {
            auto sB1=tB1.slice(0,0);
            mm.run(sB1,sA,cT1);
        }
#else
        threadgroup const half *la=sa+4*64*(sg%2), *lb=sb+2*64*(sg/2);
        for (ushort ik=0;ik<4;++ik) {
            for (ushort i=0;i<4;++i) simdgroup_load(ma[i],la+64*i,8,0,false);
            for (ushort i=0;i<2;++i) simdgroup_load(mb[i],lb+64*i,8,0,false);
            for (ushort i=0;i<8;++i) simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);
            la+=8*64; lb+=4*64;
        }
#endif
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#ifdef DINF_PREFILL_TENSOR
    auto tC0=tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(result,dextents<int32_t,2>(64,16));
    cT0.store(tC0);
    if(nr1>16) {
        auto tC1=tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(result+16*64,dextents<int32_t,2>(64,16));
        cT1.store(tC1);
    }
#else
    threadgroup float *dest=result+32*(sg&1)+16*(sg>>1)*64;
    for (ushort i=0;i<8;++i) simdgroup_store(mc[i],dest+8*(i%4)+8*64*(i/4),64,0,false);
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=tid;i<nr1*64;i+=128) {
        const uint j=i/64,col=i%64;
        if (col>=nr0) continue;
        const uint id=a.mode==0?r1+j:map[a.expert*a.stride+1+r1+j];
        const uint row=a.mode==1?12*(id/6)+2*(id%6)+a.kind:id;
        output[size_t(row)*a.m+r0+col]=result[j*64+col];
    }
}
typedef decltype(dinf_prefill_mm<true>) dinf_prefill_mm_fn;
template [[host_name("dinf_prefill_iq1_mm")]] kernel dinf_prefill_mm_fn dinf_prefill_mm<true>;
template [[host_name("dinf_prefill_q8_mm")]] kernel dinf_prefill_mm_fn dinf_prefill_mm<false>;

// Maya sqrt-softplus route oracle: strict bitonic comparisons preserve the
// existing network's ties/order; unbiased probabilities supply weights.
kernel void dinf_prefill_route(device const float *logits [[buffer(0)]],
    constant float *bias [[buffer(1)]], device uint *compact [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float p[512], score[512];
    threadgroup uint ids[512], bad[512];
    uint i=tid;
    float x=i<384?logits[row*384+i]:0, b=i<384?bias[i]:0;
    float ex=exp(x);
    float sp=x>20?x:(ex<0.03125f?ex*(1-ex*(0.5f-ex*(1.0f/3.0f-0.25f*ex))):log(1+ex));
    p[i]=sqrt(sp); score[i]=p[i]+b; ids[i]=i;
    bad[i]=i<384 && (!isfinite(x)||!isfinite(b)||!isfinite(score[i]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k=2;k<=512;k*=2) for (uint j=k/2;j>0;j/=2) {
        uint other=i^j;
        if (other>i) {
            uint a=ids[i],b=ids[other];
            bool swap=(i&k)==0?(a>=384||(b<384&&score[a]<score[b])):(b>=384||(a<384&&score[a]>score[b]));
            if (swap) {ids[i]=b;ids[other]=a;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Explicit same F32 SIMD tree as CPU oracle, not a reassociated reduction.
    if (tid==0) {
        float lanes[32]; for(uint j=0;j<32;++j) lanes[j]=j<6?p[ids[j]]:0;
        for(uint s=16;s>0;s/=2) for(uint j=0;j<s;++j) lanes[j]+=lanes[j+s];
        uint error=0; for(uint j=0;j<384;++j) error|=bad[j];
        float total=max(lanes[0],6.1035156e-5f);
        for(uint j=0;j<6;++j) {compact[row*13+j]=ids[j];float q=p[ids[j]]/total;compact[row*13+6+j]=as_type<uint>(q*1.5f);}
        compact[row*13+12]=error;
    }
}
kernel void dinf_prefill_map(device const uint *compact [[buffer(0)]], device uint *map [[buffer(1)]],
    constant uint &tokens [[buffer(2)]], uint expert [[thread_position_in_grid]]) {
    if(expert>=384) return;
    uint stride=1+6*tokens, count=0;
    for(uint t=0;t<tokens;++t) for(uint s=0;s<6;++s)
        if(compact[t*13+s]==expert) map[expert*stride+1+count++]=6*t+s;
    map[expert*stride]=count;
}

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
kernel void dinf_prefill_combine(device const float *down [[buffer(0)]],
    device const uint *compact [[buffer(1)]], device float *output [[buffer(2)]],
    uint2 id [[thread_position_in_grid]]) {
    if(id.x>=5120) return;
    volatile float sum=0.0f;
    for(uint s=0;s<6;++s) {
        volatile float weighted=fma(down[(6*id.y+s)*5120+id.x],as_type<float>(compact[id.y*13+6+s]),0.0f);
        sum=sum+weighted;
    }
    output[id.y*5120+id.x]=sum;
}

// Packed V4.1 Coder prompt projections. Same 64x32 tile, K32 steps, half
// operands and F32 accumulators as dinf_prefill_mm. Every output element sums
// K in one fixed order for any row count or expert grouping, so a prompt gives
// identical rows however it is chunked.
struct dinf_expert_ptr { device const uchar *w; };
struct dinf_expert_group { uint offset, count; };
template<typename block_q, short nl,
    void (*dequantize)(device const block_q *, short, thread half4x4 &), bool INDEXED>
inline void dinf_packed_tile(device const uchar *weights, device const float *input,
    device float *output, device const uint *rows, constant dinf_mm_args &a,
    uint offset, uint count, uint r0, uint r1, ushort tid, ushort sg,
    threadgroup half *sa, threadgroup half *sb, threadgroup float *result) {
    if(r0>=a.m || r1>=count) return;
    const uint nr0 = min(64u,a.m-r0), nr1 = min(32u,count-r1);
    const uint lr0 = min(uint(tid)/2,nr0-1), il0 = tid%2;
    const uint lr1 = min(uint(tid)/4,nr1-1), iy = 8*(tid%4);
    const uint source = INDEXED ? rows[offset+r1+lr1] : r1+lr1;
    device const block_q *w=(device const block_q *)(weights+size_t(r0+lr0)*a.row_bytes);
#ifdef DINF_PREFILL_TENSOR
    auto tA=tensor<threadgroup half,dextents<int32_t,2>,tensor_inline>(sa,dextents<int32_t,2>(32,64));
    auto tB0=tensor<threadgroup half,dextents<int32_t,2>,tensor_inline>(sb,dextents<int32_t,2>(32,16));
    auto tB1=tensor<threadgroup half,dextents<int32_t,2>,tensor_inline>(sb+16*32,dextents<int32_t,2>(32,16));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(16,64,32,false,true,false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cT0=mm.get_destination_cooperative_tensor<decltype(tA),decltype(tB0),float>();
    auto cT1=mm.get_destination_cooperative_tensor<decltype(tA),decltype(tB1),float>();
#else
    simdgroup_half8x8 ma[4], mb[2];
    simdgroup_float8x8 mc[8];
    for (ushort i=0;i<8;++i) mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
#endif
    for (uint kk=0;kk<a.k;kk+=32) {
        half4x4 decoded;
        const uint k16=(kk+16*il0)/16;
        dequantize(w+k16/uint(nl),short(k16%uint(nl)),decoded);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort i=0;i<16;++i) {
#ifdef DINF_PREFILL_TENSOR
            sa[(tid/2)*32+16*il0+i]=decoded[i/4][i%4];
#else
            const ushort sx=2*il0+i/8, sy=(tid/2)/8, lx=(tid/2)%8, ly=i%8;
            sa[64*(8*sx+sy)+8*ly+lx]=decoded[i/4][i%4];
#endif
        }
        for (ushort i=0;i<8;++i) {
            const ushort sx=tid%4, sy=(tid/4)/8, ly=(tid/4)%8;
#ifdef DINF_PREFILL_TENSOR
            sb[(tid/4)*32+iy+i]=half(input[size_t(source)*a.k+kk+iy+i]);
#else
            sb[64*(4*sx+sy)+8*ly+i]=half(input[size_t(source)*a.k+kk+iy+i]);
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#ifdef DINF_PREFILL_TENSOR
        auto sA=tA.slice(0,0);
        auto sB0=tB0.slice(0,0);
        mm.run(sB0,sA,cT0);
        if(nr1>16) {
            auto sB1=tB1.slice(0,0);
            mm.run(sB1,sA,cT1);
        }
#else
        threadgroup const half *la=sa+4*64*(sg%2), *lb=sb+2*64*(sg/2);
        for (ushort ik=0;ik<4;++ik) {
            for (ushort i=0;i<4;++i) simdgroup_load(ma[i],la+64*i,8,0,false);
            for (ushort i=0;i<2;++i) simdgroup_load(mb[i],lb+64*i,8,0,false);
            for (ushort i=0;i<8;++i) simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);
            la+=8*64; lb+=4*64;
        }
#endif
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#ifdef DINF_PREFILL_TENSOR
    auto tC0=tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(result,dextents<int32_t,2>(64,16));
    cT0.store(tC0);
    if(nr1>16) {
        auto tC1=tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(result+16*64,dextents<int32_t,2>(64,16));
        cT1.store(tC1);
    }
#else
    threadgroup float *dest=result+32*(sg&1)+16*(sg>>1)*64;
    for (ushort i=0;i<8;++i) simdgroup_store(mc[i],dest+8*(i%4)+8*64*(i/4),64,0,false);
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=tid;i<nr1*64;i+=128) {
        const uint j=i/64,col=i%64;
        if (col>=nr0) continue;
        output[size_t(offset+r1+j)*a.m+r0+col]=result[j*64+col];
    }
}
template<typename block_q, short nl,
    void (*dequantize)(device const block_q *, short, thread half4x4 &)>
kernel void dinf_prefill_dense_mm(device const uchar *weights [[buffer(0)]],
    device const float *input [[buffer(1)]], device float *output [[buffer(2)]],
    constant dinf_mm_args &a [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[2048];
    threadgroup half sb[1024];
    threadgroup float result[2048];
    dinf_packed_tile<block_q, nl, dequantize, false>(weights, input, output, nullptr, a,
        0, a.n, group.x*64, group.y*32, tid, sg, sa, sb, result);
}
// grid.z selects one expert: its weight address, input rows and output rows.
template<typename block_q, short nl,
    void (*dequantize)(device const block_q *, short, thread half4x4 &)>
kernel void dinf_prefill_experts_mm(device const dinf_expert_ptr *ptrs [[buffer(0)]],
    device const float *input [[buffer(1)]], device float *output [[buffer(2)]],
    device const uint *rows [[buffer(3)]], constant dinf_mm_args &a [[buffer(4)]],
    device const dinf_expert_group *groups [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[2048];
    threadgroup half sb[1024];
    threadgroup float result[2048];
    const dinf_expert_group g = groups[group.z];
    dinf_packed_tile<block_q, nl, dequantize, true>(ptrs[group.z].w, input, output, rows, a,
        g.offset, g.count, group.x*64, group.y*32, tid, sg, sa, sb, result);
}
typedef decltype(dinf_prefill_dense_mm<block_q5_K, 16, dequantize_q5_K>) dinf_prefill_dense_fn;
typedef decltype(dinf_prefill_experts_mm<block_q5_K, 16, dequantize_q5_K>) dinf_prefill_experts_fn;
template [[host_name("dinf_prefill_q5k_dense")]] kernel dinf_prefill_dense_fn
    dinf_prefill_dense_mm<block_q5_K, 16, dequantize_q5_K>;
template [[host_name("dinf_prefill_q5k_experts")]] kernel dinf_prefill_experts_fn
    dinf_prefill_experts_mm<block_q5_K, 16, dequantize_q5_K>;
template [[host_name("dinf_prefill_iq1m_dense")]] kernel dinf_prefill_dense_fn
    dinf_prefill_dense_mm<block_iq1_m, 16, dequantize_iq1_m>;
template [[host_name("dinf_prefill_iq1m_experts")]] kernel dinf_prefill_experts_fn
    dinf_prefill_experts_mm<block_iq1_m, 16, dequantize_iq1_m>;
template [[host_name("dinf_prefill_iq2xxs_dense")]] kernel dinf_prefill_dense_fn
    dinf_prefill_dense_mm<block_iq2_xxs, 16, dequantize_iq2_xxs>;
template [[host_name("dinf_prefill_iq2xxs_experts")]] kernel dinf_prefill_experts_fn
    dinf_prefill_experts_mm<block_iq2_xxs, 16, dequantize_iq2_xxs>;
template [[host_name("dinf_prefill_iq2xs_dense")]] kernel dinf_prefill_dense_fn
    dinf_prefill_dense_mm<block_iq2_xs, 16, dequantize_iq2_xs>;
template [[host_name("dinf_prefill_iq2xs_experts")]] kernel dinf_prefill_experts_fn
    dinf_prefill_experts_mm<block_iq2_xs, 16, dequantize_iq2_xs>;
