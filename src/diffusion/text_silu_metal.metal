#include <metal_stdlib>
using namespace metal;
kernel void silu_precise(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant uint&n [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i<n) y[i]=float(bfloat(x[i]/(1.0f+precise::exp(-x[i]))));
}
