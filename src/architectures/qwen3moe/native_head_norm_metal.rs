//! Native Metal D128 per-head weighted RMS normalization.
use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{CpuStorage, CustomOp2, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
// Arithmetic adapted from ggml-metal.metal kernel_rms_norm_fuse_impl<float4,2>,
// llama.cpp 15fa3c493bfcd040b5f4dcb29e1c998a0846de16.
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void qwen_native_head_norm(device const float4* src [[buffer(0)]],
 device const float4* weight [[buffer(1)]], device float4* dst [[buffer(2)]],
 constant float& eps [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
 ushort tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
 threadgroup float sums[32];
 sums[lane]=0.f;
 device const float4* x=src+row*32;
 float sum=0.f;
 sum += dot(x[tid],x[tid]);
 sum=simd_sum(sum);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(lane==0) sums[0]=sum;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 sum=sums[lane];
 sum=simd_sum(sum);
 const float mean=sum/128;
 const float scale=1.f/sqrt(mean+eps);
 dst[row*32+tid]=(x[tid]*scale)*weight[tid];
}
"#;
static PIPELINES: OnceLock<Mutex<HashMap<DeviceId, ComputePipeline>>> = OnceLock::new();
fn pipeline(device: &candle_core::MetalDevice) -> Result<ComputePipeline> {
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| candle_core::Error::Msg("head norm pipeline lock".into()))?;
    if let Some(p) = cache.get(&device.id()) {
        return Ok(p.clone());
    }
    let lib = device
        .metal_device()
        .new_library_with_source(SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let f = lib
        .get_function("qwen_native_head_norm", None)
        .map_err(candle_core::Error::wrap)?;
    let p = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&f)
        .map_err(candle_core::Error::wrap)?;
    cache.insert(device.id(), p.clone());
    Ok(p)
}
struct NativeHeadNorm {
    eps: f32,
}
impl CustomOp2 for NativeHeadNorm {
    fn name(&self) -> &'static str {
        "qwen-native-head-norm"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("head norm needs Metal")
    }
    fn metal_fwd(
        &self,
        x: &MetalStorage,
        xl: &Layout,
        w: &MetalStorage,
        wl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (b, h, q, d) = xl.shape().dims4()?;
        if x.dtype() != DType::F32
            || w.dtype() != DType::F32
            || x.device().id() != w.device().id()
            || !xl.is_contiguous()
            || !wl.is_contiguous()
            || d != 128
            || !matches!(h, 4 | 32)
            || wl.shape().dims() != [128]
            || b == 0
            || q == 0
        {
            candle_core::bail!("head norm geometry");
        }
        let n = xl.shape().elem_count();
        let xe = xl
            .start_offset()
            .checked_add(n)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| candle_core::Error::Msg("head norm input overflow".into()))?;
        let we = wl
            .start_offset()
            .checked_add(128)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| candle_core::Error::Msg("head norm weight overflow".into()))?;
        if xe > x.buffer().length()
            || we > w.buffer().length()
            || n > u32::MAX as usize
            || !xl.start_offset().is_multiple_of(4)
            || !wl.start_offset().is_multiple_of(4)
        {
            candle_core::bail!("head norm span");
        }
        let dev = x.device();
        let output = dev
            .new_buffer_builder()
            .with_size_for(n, DType::F32)
            .with_label("qwen-head-norm")
            .build()?;
        let p = pipeline(dev)?;
        let encoder = dev.command_encoder()?;
        encoder.set_compute_pipeline_state(&p);
        let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        enc.set_input_buffer(0, Some(x.buffer()), xl.start_offset() * 4);
        enc.set_input_buffer(1, Some(w.buffer()), wl.start_offset() * 4);
        enc.set_output_buffer(2, Some(&output), 0);
        enc.set_bytes(3, &self.eps);
        enc.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: n / 128,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, dev.clone(), n, DType::F32),
            xl.shape().clone(),
        ))
    }
}
pub fn forward(x: &Tensor, w: &Tensor, eps: f32) -> Result<Option<Tensor>> {
    if !x.device().is_metal()
        || !x.device().same_device(w.device())
        || x.dtype() != DType::F32
        || w.dtype() != DType::F32
        || x.rank() != 4
        || x.dim(3)? != 128
        || !matches!(x.dim(1)?, 4 | 32)
        || w.dims() != [128]
        || eps.to_bits() != 1e-6f32.to_bits()
    {
        return Ok(None);
    }
    if x.dim(0)? == 0 || x.dim(2)? == 0 {
        return Ok(None);
    }
    let x = x.contiguous()?;
    let w = w.contiguous()?;
    Ok(Some(x.apply_op2_no_bwd(&w, &NativeHeadNorm { eps })?))
}
