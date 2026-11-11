//! Native Metal unscaled NeoX rotary arithmetic for the selected Qwen geometry.
//! This is a compatibility policy, not a claim of greater accuracy.
use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{CpuStorage, CustomOp1, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

// Arithmetic adapted from ggml-metal.metal kernel_rope_neox and rope_yarn,
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
kernel void dinf_qwen_native_rope_f32(
 device const float *src [[buffer(0)]], device float *dst [[buffer(1)]],
 constant uint &heads [[buffer(2)]], constant uint &queries [[buffer(3)]],
 constant uint &dim [[buffer(4)]], constant uint &start [[buffer(5)]],
 constant float &base [[buffer(6)]], constant float &freq_scale [[buffer(7)]],
 constant float &attn_factor [[buffer(8)]],
 ushort tid [[thread_index_in_threadgroup]], ushort3 threads [[threads_per_threadgroup]],
 uint3 group [[threadgroup_position_in_grid]]) {
 const float theta_base = float(start + group.y);
 const float inv_ndims = -1.f / float(dim);
 const uint row = (group.z * heads + group.x) * queries + group.y;
 for (int i0 = 2*tid; i0 < int(dim); i0 += 2*threads.x) {
  const int ic = i0/2;
  const float theta_extrap = theta_base * pow(base, inv_ndims*i0);
  const float theta = freq_scale * theta_extrap;
  const float cos_theta = cos(theta) * attn_factor;
  const float sin_theta = sin(theta) * attn_factor;
  const float x0 = src[row*dim + ic];
  const float x1 = src[row*dim + ic + dim/2];
  dst[row*dim + ic] = x0*cos_theta - x1*sin_theta;
  // Pin the native second-half contraction; first half already matches.
  dst[row*dim + ic + dim/2] = fma(x0, sin_theta, x1*cos_theta);
 }
}
"#;
static PIPELINES: OnceLock<Mutex<HashMap<DeviceId, ComputePipeline>>> = OnceLock::new();
fn pipeline(device: &candle_core::MetalDevice) -> Result<ComputePipeline> {
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(p) = pipelines.lock().unwrap().get(&device.id()) {
        return Ok(p.clone());
    }
    // Match native's default Metal compile options; no global environment flag.
    let library = device
        .metal_device()
        .new_library_with_source(SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function("dinf_qwen_native_rope_f32", None)
        .map_err(candle_core::Error::wrap)?;
    let p = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    pipelines.lock().unwrap().insert(device.id(), p.clone());
    Ok(p)
}
struct NativeRotary {
    start: u32,
    base: f32,
}
impl CustomOp1 for NativeRotary {
    fn name(&self) -> &'static str {
        "qwen-native-metal-rotary"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("native Metal rotary requires Metal")
    }
    fn metal_fwd(&self, input: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let (batch, heads, queries, dim) = layout.shape().dims4()?;
        if input.dtype() != DType::F32
            || !layout.is_contiguous()
            || dim != 128
            || !matches!(heads, 4 | 32)
            || batch == 0
            || queries == 0
        {
            candle_core::bail!("native Metal rotary geometry")
        }
        let elements = layout.shape().elem_count();
        if elements > u32::MAX as usize || queries > u32::MAX as usize || batch > u32::MAX as usize
        {
            candle_core::bail!("native Metal rotary size overflow")
        }
        let device = input.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("qwen-native-rope")
            .build()?;
        let p = pipeline(device)?;
        let encoder = device.command_encoder()?;
        encoder.set_label("qwen-native-rope");
        encoder.set_compute_pipeline_state(&p);
        let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        enc.set_input_buffer(0, Some(input.buffer()), layout.start_offset() * 4);
        enc.set_output_buffer(1, Some(&output), 0);
        enc.set_bytes(2, &(heads as u32));
        enc.set_bytes(3, &(queries as u32));
        enc.set_bytes(4, &(dim as u32));
        enc.set_bytes(5, &self.start);
        enc.set_bytes(6, &self.base);
        enc.set_bytes(7, &1.0f32);
        enc.set_bytes(8, &1.0f32);
        enc.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: heads,
                height: queries,
                depth: batch,
            },
            objc2_metal::MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), elements, DType::F32),
            layout.shape().clone(),
        ))
    }
}
pub fn forward(
    input: &Tensor,
    start: usize,
    base: f64,
    context: usize,
    native_cache: bool,
) -> Result<Option<Tensor>> {
    if !native_cache
        || !input.device().is_metal()
        || input.dtype() != DType::F32
        || input.rank() != 4
        || input.dim(3)? != 128
        || !matches!(input.dim(1)?, 4 | 32)
        || base != 1_000_000.0
    {
        return Ok(None);
    }
    let (batch, _, queries, _) = input.dims4()?;
    let end = start
        .checked_add(queries)
        .ok_or_else(|| candle_core::Error::Msg("native Metal rotary position overflow".into()))?;
    if end > context || end > i32::MAX as usize || input.elem_count() > u32::MAX as usize {
        candle_core::bail!("native Metal rotary context exceeded")
    }
    if batch == 0 || queries == 0 {
        return Ok(None);
    }
    let input = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    Ok(Some(input.apply_op1_no_bwd(&NativeRotary {
        start: start as u32,
        base: base as f32,
    })?))
}

// Bind stored precision before QMatMul::from_arc dequantizes BF16 weights.
// Q8/K-quant projections keep the original rotary path.
fn supported_types(q: GgmlDType, k: GgmlDType) -> bool {
    q == GgmlDType::BF16 && k == GgmlDType::BF16
}
pub(super) fn supports_weights(q: &QTensor, k: &QTensor) -> bool {
    supported_types(q.dtype(), k.dtype())
        && q.device().is_metal()
        && q.device().same_device(&k.device())
}
#[cfg(test)]
mod bf16_guard_tests {
    use super::*;
    use candle_core::Device;
    #[test]
    fn stored_precision_requires_both_bf16_weights() {
        assert!(supported_types(GgmlDType::BF16, GgmlDType::BF16));
        for dtype in [
            GgmlDType::F32,
            GgmlDType::F16,
            GgmlDType::Q8_0,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
        ] {
            assert!(!supported_types(dtype, dtype));
            assert!(!supported_types(GgmlDType::BF16, dtype));
            assert!(!supported_types(dtype, GgmlDType::BF16));
        }
    }
    #[test]
    fn actual_stored_weights_admit_only_metal_bf16() -> Result<()> {
        let cpu = Device::Cpu;
        let metal = Device::new_metal(0)?;
        let input = Tensor::zeros((4, 256), DType::F32, &cpu)?;
        let bf16_cpu = QTensor::quantize(&input, GgmlDType::BF16)?;
        let bf16_metal = QTensor::quantize_onto(&input, GgmlDType::BF16, &metal)?;
        assert!(!supports_weights(&bf16_cpu, &bf16_cpu));
        assert!(!supports_weights(&bf16_cpu, &bf16_metal));
        assert!(!supports_weights(&bf16_metal, &bf16_cpu));
        assert!(supports_weights(&bf16_metal, &bf16_metal));
        for dtype in [GgmlDType::Q8_0, GgmlDType::Q4K, GgmlDType::F32] {
            let weight = QTensor::quantize_onto(&input, dtype, &metal)?;
            assert!(!supports_weights(&weight, &weight));
            assert!(!supports_weights(&bf16_metal, &weight));
        }
        let q = Tensor::zeros((1, 4, 1, 128), DType::F32, &metal)?;
        assert!(forward(&q, 0, 1_000_000.0, 4096, false)?.is_none());
        assert!(forward(&q.to_device(&cpu)?, 0, 1_000_000.0, 4096, true)?.is_none());
        Ok(())
    }
}
