// Adapted from PyTorch d0f2af8b66b20da4baa50644c0cd3f887a2eae2e SoftMax.metal.
// BSD-3-Clause; license/source pins are recorded in the manifest.
#include <metal_stdlib>
using namespace metal;
kernel void softmax_row_f32(constant float* input [[buffer(0)]],device float* output [[buffer(1)]],constant uint2& p [[buffer(2)]],uint tgid [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint row=tgid*8+sg;if(row>=p.x)return;uint base=row*p.y;float maximum=-INFINITY;
 for(uint col=lane;col<p.y;col+=32)maximum=max(maximum,input[base+col]);maximum=simd_max(maximum);float sum=0.0f;
 for(uint col=lane;col<p.y;col+=32)sum+=precise::exp(input[base+col]-maximum);float inv=1.0f/simd_sum(sum);
 for(uint col=lane;col<p.y;col+=32)output[base+col]=precise::exp(input[base+col]-maximum)*inv;
}
