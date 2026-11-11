use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{
    CpuStorage, CustomOp3, DType, Layout, MetalStorage, Result, Shape, Storage, Tensor,
};
use candle_metal_kernels::metal::ComputePipeline;

const MLA_LAYOUT_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void dinf_deepseek_mla_layout_rope_f32(
    device const float *q_input [[buffer(0)]],
    device const float *kv_input [[buffer(1)]],
    device const float *kv_mixed [[buffer(2)]],
    device const float *cos [[buffer(3)]],
    device const float *sin [[buffer(4)]],
    device float *output [[buffer(5)]],
    constant uint &batch [[buffer(6)]],
    constant uint &seq_len [[buffer(7)]],
    constant uint &num_heads [[buffer(8)]],
    constant uint &nope_dim [[buffer(9)]],
    constant uint &rope_dim [[buffer(10)]],
    constant uint &value_dim [[buffer(11)]],
    constant uint &kv_rank [[buffer(12)]],
    constant uint &start_pos [[buffer(13)]],
    constant uint &elements [[buffer(14)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= elements) {
    return;
  }

  const uint qk_dim = nope_dim + rope_dim;
  const uint qk_elements = batch * num_heads * seq_len * qk_dim;
  const uint value_elements = batch * num_heads * seq_len * value_dim;
  uint local = gid;
  uint output_kind = 0;
  uint head_dim = qk_dim;
  if (local >= qk_elements) {
    local -= qk_elements;
    output_kind = 1;
    if (local >= qk_elements) {
      local -= qk_elements;
      output_kind = 2;
      head_dim = value_dim;
    }
  }

  const uint column = local % head_dim;
  uint index = local / head_dim;
  const uint token = index % seq_len;
  index /= seq_len;
  const uint head = index % num_heads;
  const uint lane = index / num_heads;
  const uint token_row = lane * seq_len + token;

  if (output_kind == 2) {
    const uint kv_source_dim = num_heads * (nope_dim + value_dim);
    const uint source = token_row * kv_source_dim
        + head * (nope_dim + value_dim) + nope_dim + column;
    output[gid] = kv_input[source];
    return;
  }

  if (column < nope_dim) {
    if (output_kind == 0) {
      const uint source = token_row * num_heads * qk_dim + head * qk_dim + column;
      output[gid] = q_input[source];
    } else {
      const uint kv_head_dim = nope_dim + value_dim;
      const uint source = token_row * num_heads * kv_head_dim
          + head * kv_head_dim + column;
      output[gid] = kv_input[source];
    }
    return;
  }

  const uint rotary_column = column - nope_dim;
  const uint pair = rotary_column / 2;
  const uint first_column = pair * 2;
  const uint source_base = output_kind == 0
      ? token_row * num_heads * qk_dim + head * qk_dim + nope_dim
      : token_row * (kv_rank + rope_dim) + kv_rank;
  device const float *source_buffer = output_kind == 0 ? q_input : kv_mixed;
  const float first = source_buffer[source_base + first_column];
  const float second = source_buffer[source_base + first_column + 1];
  const uint rope_index = (start_pos + token) * (rope_dim / 2) + pair;
  const float c = cos[rope_index];
  const float s = sin[rope_index];
  if (rotary_column % 2 == 0) {
    output[gid] = first * c - second * s;
  } else {
    output[gid] = first * s + second * c;
  }
}
"#;

static PIPELINES: OnceLock<Mutex<HashMap<DeviceId, ComputePipeline>>> = OnceLock::new();

fn pipeline(device: &candle_core::MetalDevice) -> Result<ComputePipeline> {
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(pipeline) = pipelines.lock().unwrap().get(&device.id()) {
        return Ok(pipeline.clone());
    }
    let library = device
        .metal_device()
        .new_library_with_source(MLA_LAYOUT_MSL, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function("dinf_deepseek_mla_layout_rope_f32", None)
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    pipelines
        .lock()
        .unwrap()
        .insert(device.id(), pipeline.clone());
    Ok(pipeline)
}

struct DeepseekMlaLayout<'a> {
    cos: &'a Tensor,
    sin: &'a Tensor,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    value_dim: usize,
    kv_rank: usize,
    start_pos: usize,
}

impl CustomOp3 for DeepseekMlaLayout<'_> {
    fn name(&self) -> &'static str {
        "dinf-metal-deepseek-mla-layout"
    }

    fn cpu_fwd(
        &self,
        _q_storage: &CpuStorage,
        _q_layout: &Layout,
        _kv_storage: &CpuStorage,
        _kv_layout: &Layout,
        _mixed_storage: &CpuStorage,
        _mixed_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused DeepSeek MLA layout is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        q_storage: &MetalStorage,
        q_layout: &Layout,
        kv_storage: &MetalStorage,
        kv_layout: &Layout,
        mixed_storage: &MetalStorage,
        mixed_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let qk_dim = self.nope_dim + self.rope_dim;
        let expanded_kv_dim = self.num_heads * (self.nope_dim + self.value_dim);
        if q_storage.dtype() != DType::F32
            || kv_storage.dtype() != DType::F32
            || mixed_storage.dtype() != DType::F32
            || !q_layout.is_contiguous()
            || !kv_layout.is_contiguous()
            || !mixed_layout.is_contiguous()
            || q_layout.shape().dims() != [self.batch, self.seq_len, self.num_heads * qk_dim]
            || kv_layout.shape().dims() != [self.batch, self.seq_len, expanded_kv_dim]
            || mixed_layout.shape().dims()
                != [self.batch, self.seq_len, self.kv_rank + self.rope_dim]
        {
            candle_core::bail!("fused DeepSeek MLA layout requires contiguous f32 inputs")
        }
        if self.rope_dim == 0 || !self.rope_dim.is_multiple_of(2) {
            candle_core::bail!("fused DeepSeek MLA RoPE dimension is invalid")
        }

        let (cos_guard, cos_layout) = self.cos.storage_and_layout();
        let (sin_guard, sin_layout) = self.sin.storage_and_layout();
        let cos = match &*cos_guard {
            Storage::Metal(storage) => storage,
            _ => candle_core::bail!("DeepSeek MLA cosine table is not on Metal"),
        };
        let sin = match &*sin_guard {
            Storage::Metal(storage) => storage,
            _ => candle_core::bail!("DeepSeek MLA sine table is not on Metal"),
        };
        if cos.dtype() != DType::F32
            || sin.dtype() != DType::F32
            || !cos_layout.is_contiguous()
            || !sin_layout.is_contiguous()
            || cos_layout.shape() != sin_layout.shape()
        {
            candle_core::bail!("DeepSeek MLA RoPE tables must be matching contiguous f32 tensors")
        }
        let (context_len, half_dim) = cos_layout.shape().dims2()?;
        if half_dim * 2 != self.rope_dim
            || self.start_pos.saturating_add(self.seq_len) > context_len
        {
            candle_core::bail!("DeepSeek MLA RoPE table geometry differs")
        }
        if q_storage.device().id() != kv_storage.device().id()
            || q_storage.device().id() != mixed_storage.device().id()
            || q_storage.device().id() != cos.device().id()
            || q_storage.device().id() != sin.device().id()
        {
            candle_core::bail!("fused DeepSeek MLA tensors are on different Metal devices")
        }

        let qk_elements = self.batch * self.num_heads * self.seq_len * qk_dim;
        let value_elements = self.batch * self.num_heads * self.seq_len * self.value_dim;
        let elements = qk_elements * 2 + value_elements;
        if elements == 0
            || self.batch > u32::MAX as usize
            || self.seq_len > u32::MAX as usize
            || self.num_heads > u32::MAX as usize
            || self.nope_dim > u32::MAX as usize
            || self.rope_dim > u32::MAX as usize
            || self.value_dim > u32::MAX as usize
            || self.kv_rank > u32::MAX as usize
            || self.start_pos > u32::MAX as usize
            || elements > u32::MAX as usize
        {
            candle_core::bail!("fused DeepSeek MLA geometry is unsupported")
        }

        let device = q_storage.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-deepseek-mla-layout")
            .build()?;
        let pipeline = pipeline(device)?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-deepseek-mla-layout");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(
            0,
            Some(q_storage.buffer()),
            q_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            1,
            Some(kv_storage.buffer()),
            kv_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            2,
            Some(mixed_storage.buffer()),
            mixed_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            3,
            Some(cos.buffer()),
            cos_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_input_buffer(
            4,
            Some(sin.buffer()),
            sin_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder_ref.set_output_buffer(5, Some(&output), 0);
        encoder_ref.set_bytes(6, &(self.batch as u32));
        encoder_ref.set_bytes(7, &(self.seq_len as u32));
        encoder_ref.set_bytes(8, &(self.num_heads as u32));
        encoder_ref.set_bytes(9, &(self.nope_dim as u32));
        encoder_ref.set_bytes(10, &(self.rope_dim as u32));
        encoder_ref.set_bytes(11, &(self.value_dim as u32));
        encoder_ref.set_bytes(12, &(self.kv_rank as u32));
        encoder_ref.set_bytes(13, &(self.start_pos as u32));
        encoder_ref.set_bytes(14, &(elements as u32));
        encoder_ref.dispatch_threads(
            objc2_metal::MTLSize {
                width: elements,
                height: 1,
                depth: 1,
            },
            candle_metal_kernels::utils::get_block_dims(elements.min(256), 1, 1),
        );
        Ok((
            MetalStorage::new(output, device.clone(), elements, DType::F32),
            Shape::from(elements),
        ))
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_fused_mla_layout(
    q: &Tensor,
    kv: &Tensor,
    kv_mixed: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    value_dim: usize,
    kv_rank: usize,
    start_pos: usize,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    if !q.device().is_metal()
        || q.dtype() != DType::F32
        || kv.dtype() != DType::F32
        || kv_mixed.dtype() != DType::F32
        || !q.is_contiguous()
        || !kv.is_contiguous()
        || !kv_mixed.is_contiguous()
    {
        return Ok(None);
    }
    let packed = q.apply_op3_no_bwd(
        kv,
        kv_mixed,
        &DeepseekMlaLayout {
            cos,
            sin,
            batch,
            seq_len,
            num_heads,
            nope_dim,
            rope_dim,
            value_dim,
            kv_rank,
            start_pos,
        },
    )?;
    let qk_dim = nope_dim + rope_dim;
    let qk_elements = batch * num_heads * seq_len * qk_dim;
    let value_elements = batch * num_heads * seq_len * value_dim;
    let q = packed
        .narrow(0, 0, qk_elements)?
        .reshape((batch, num_heads, seq_len, qk_dim))?;
    let k = packed
        .narrow(0, qk_elements, qk_elements)?
        .reshape((batch, num_heads, seq_len, qk_dim))?;
    let v = packed
        .narrow(0, qk_elements * 2, value_elements)?
        .reshape((batch, num_heads, seq_len, value_dim))?;
    Ok(Some((q, k, v)))
}

#[cfg(test)]
mod tests {
    use candle_core::{D, Tensor};

    use super::try_fused_mla_layout;

    #[test]
    fn fused_mla_layout_matches_candle_assembly() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let batch = 2;
        let heads = 4;
        let nope_dim = 6;
        let rope_dim = 4;
        let value_dim = 5;
        let kv_rank = 8;
        let start_pos = 2;
        for seq_len in [2, 7, 64] {
            let qk_dim = nope_dim + rope_dim;
            let kv_head_dim = nope_dim + value_dim;
            let q = Tensor::arange(
                -200f32,
                (batch * seq_len * heads * qk_dim) as f32 - 200.0,
                &device,
            )?
            .affine(0.01, 0.0)?
            .reshape((batch, seq_len, heads * qk_dim))?;
            let kv = Tensor::arange(
                -100f32,
                (batch * seq_len * heads * kv_head_dim) as f32 - 100.0,
                &device,
            )?
            .affine(0.02, 0.0)?
            .reshape((batch, seq_len, heads * kv_head_dim))?;
            let mixed = Tensor::arange(
                -50f32,
                (batch * seq_len * (kv_rank + rope_dim)) as f32 - 50.0,
                &device,
            )?
            .affine(0.015, 0.0)?
            .reshape((batch, seq_len, kv_rank + rope_dim))?;
            let table_len = start_pos + seq_len + 1;
            let angles = Tensor::arange(0f32, (table_len * rope_dim / 2) as f32, &device)?
                .affine(0.03125, 0.0)?
                .reshape((table_len, rope_dim / 2))?;
            let cos = angles.cos()?;
            let sin = angles.sin()?;

            let q_heads = q
                .reshape((batch, seq_len, heads, qk_dim))?
                .transpose(1, 2)?;
            let q_nope = q_heads.narrow(D::Minus1, 0, nope_dim)?;
            let q_pe = q_heads.narrow(D::Minus1, nope_dim, rope_dim)?;
            let q_pe = candle_nn::rotary_emb::rope_i(
                &q_pe.contiguous()?,
                &cos.narrow(0, start_pos, seq_len)?,
                &sin.narrow(0, start_pos, seq_len)?,
            )?;
            let expected_q = Tensor::cat(&[q_nope, q_pe], D::Minus1)?;

            let kv_heads = kv
                .reshape((batch, seq_len, heads, kv_head_dim))?
                .transpose(1, 2)?;
            let k_nope = kv_heads.narrow(D::Minus1, 0, nope_dim)?;
            let expected_v = kv_heads.narrow(D::Minus1, nope_dim, value_dim)?;
            let k_pe = mixed
                .narrow(D::Minus1, kv_rank, rope_dim)?
                .reshape((batch, seq_len, 1, rope_dim))?
                .transpose(1, 2)?;
            let k_pe = candle_nn::rotary_emb::rope_i(
                &k_pe.contiguous()?,
                &cos.narrow(0, start_pos, seq_len)?,
                &sin.narrow(0, start_pos, seq_len)?,
            )?;
            let expected_k = Tensor::cat(&[k_nope, k_pe.repeat((1, heads, 1, 1))?], D::Minus1)?;

            let actual = try_fused_mla_layout(
                &q, &kv, &mixed, &cos, &sin, batch, seq_len, heads, nope_dim, rope_dim, value_dim,
                kv_rank, start_pos,
            )?
            .expect("Metal fused path");
            for (actual, expected) in [
                (actual.0, expected_q),
                (actual.1, expected_k),
                (actual.2, expected_v),
            ] {
                let actual = actual.flatten_all()?.to_vec1::<f32>()?;
                let expected = expected.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
                assert_eq!(actual.len(), expected.len());
                let max_error = actual
                    .iter()
                    .zip(&expected)
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0f32, f32::max);
                let max_value = expected
                    .iter()
                    .map(|value| value.abs())
                    .fold(1.0f32, f32::max);
                assert!(
                    max_error <= max_value * 1e-6,
                    "MLA layout max error {max_error} exceeds tolerance for scale {max_value}"
                );
            }
        }
        Ok(())
    }
}
