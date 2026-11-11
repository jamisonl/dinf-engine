// Adapted from PyTorch d0f2af8b66b20da4baa50644c0cd3f887a2eae2e ReduceOps.metal.
// Copyright PyTorch contributors. BSD-3-Clause; see THIRD_PARTY_LICENSES/PYTORCH-BSD-3-CLAUSE.txt.
#include <metal_stdlib>
using namespace metal;
kernel void mean_row(const device float* x [[buffer(0)]], device float* out [[buffer(1)]], constant uint4& p [[buffer(2)]], constant float& divisor [[buffer(3)]], uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
 uint row=tgid*8+sg;if(row>=p.x)return;float acc[8]={0,0,0,0,0,0,0,0};uint aligned=p.y/256*256;
 for(uint base=lane*8;base<aligned;base+=256)for(uint j=0;j<8;j++)acc[j]+=x[row*p.y+base+j];
 for(uint i=aligned+lane;i<p.y;i+=32)acc[0]+=x[row*p.y+i];
 float val=acc[0];for(uint j=1;j<8;j++)val+=acc[j];val=simd_sum(val);if(lane==0)out[row]=divisor>0?val/divisor:val;
}
kernel void mean_chunk(const device float* x [[buffer(0)]], device float* out [[buffer(1)]], constant uint4& p [[buffer(2)]], constant float& divisor [[buffer(3)]], uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
 uint rows=p.x,width=p.y,lanes=p.z,segs=p.w;uint oi=(tgid*8+sg)*(32/lanes)+lane/lanes;if(oi>=rows*segs)return;
 uint rowlane=lane%lanes,row=oi/segs,seg=oi%segs,len=(width+segs-1)/segs,begin=seg*len,end=min(begin+len,width),base=row*width;
 float acc[4]={0,0,0,0};uint k=begin+rowlane;
 if(lanes<32&&segs==1&&(width&3)==0){uint quads=(end-begin)/4;if(quads>=2){const device float4* v=(const device float4*)(x+base+begin);for(uint q=rowlane;q<quads;q+=lanes){float4 a=v[q];for(uint j=0;j<4;j++)acc[0]+=a[j];}k=begin+quads*4+rowlane;}}
 for(;k+3*lanes<end;k+=4*lanes)for(uint j=0;j<4;j++)acc[j]+=x[base+k+j*lanes];
 for(;k<end;k+=lanes)acc[0]+=x[base+k];float val=(acc[0]+acc[1])+(acc[2]+acc[3]);
 for(uint off=lanes>>1;off>0;off>>=1)val+=simd_shuffle_down(val,(ushort)off);
 if(rowlane==0)out[oi]=divisor>0?val/divisor:val;
}
kernel void square_input(const device bfloat* x [[buffer(0)]],device float* square [[buffer(1)]],constant uint& count [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i<count){float v=float(x[i]);square[i]=v*v;}
}
kernel void finish_norm(const device bfloat* x [[buffer(0)]],const device float* mean [[buffer(1)]],const device bfloat* weight [[buffer(2)]],device bfloat* out [[buffer(3)]],constant uint2& shape [[buffer(4)]],constant float& eps [[buffer(5)]],uint i [[thread_position_in_grid]]) {
 if(i>=shape.x*shape.y)return;float inv=1.0f/precise::sqrt(mean[i/shape.y]+eps);float v=float(x[i])*inv;bfloat rounded=bfloat(v);out[i]=bfloat(float(rounded)*float(weight[i%shape.y]));
}
