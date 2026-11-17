use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{DType, Device, Layout, MetalStorage, Result, Storage, Tensor};
use candle_metal_kernels::metal::ComputePipeline;

const KDA_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void dinf_glm5_kda_scan_f32(
    device const float *initial_state [[buffer(0)]],
    device const float *q             [[buffer(1)]],
    device const float *k             [[buffer(2)]],
    device const float *v             [[buffer(3)]],
    device const float *initial_conv  [[buffer(4)]],
    device const float *conv_weight   [[buffer(5)]],
    device const float *decay_exp     [[buffer(6)]],
    device const float *beta_raw      [[buffer(7)]],
    device float *state_output        [[buffer(8)]],
    device float *scan_output         [[buffer(9)]],
    device float *conv_output         [[buffer(10)]],
    constant uint &seq_len            [[buffer(11)]],
    constant uint &n_heads            [[buffer(12)]],
    constant uint &head_dim           [[buffer(13)]],
    constant uint &kernel_size        [[buffer(14)]],
    constant float &norm_eps          [[buffer(15)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  threadgroup float q_values[256];
  threadgroup float k_values[256];
  threadgroup float decay_values[256];
  threadgroup float q_sums[256];
  threadgroup float k_sums[256];

  if (head >= n_heads) {
    return;
  }
  const bool active = tid < head_dim;
  const uint inner = n_heads * head_dim;
  const uint conv_dim = 3 * inner;

  float q_history[8];
  float k_history[8];
  float v_history[8];
  if (active) {
    const uint channel = head * head_dim + tid;
    for (uint k = 0; k < kernel_size; ++k) {
      q_history[k] = initial_conv[channel * kernel_size + k];
      k_history[k] = initial_conv[(inner + channel) * kernel_size + k];
      v_history[k] = initial_conv[(2 * inner + channel) * kernel_size + k];
    }
  }

  for (uint token = 0; token < seq_len; ++token) {
    float q_value = 0.0f;
    float k_value = 0.0f;
    float v_value = 0.0f;
    if (active) {
      const uint channel = head * head_dim + tid;
      for (uint k = 1; k < kernel_size; ++k) {
        q_history[k - 1] = q_history[k];
        k_history[k - 1] = k_history[k];
        v_history[k - 1] = v_history[k];
      }
      const uint projection_index = token * inner + channel;
      q_history[kernel_size - 1] = q[projection_index];
      k_history[kernel_size - 1] = k[projection_index];
      v_history[kernel_size - 1] = v[projection_index];
      for (uint k = 0; k < kernel_size; ++k) {
        q_value += q_history[k] * conv_weight[channel * kernel_size + k];
        k_value += k_history[k] * conv_weight[(inner + channel) * kernel_size + k];
        v_value += v_history[k] * conv_weight[(2 * inner + channel) * kernel_size + k];
      }
      q_value = q_value / (1.0f + exp(-q_value));
      k_value = k_value / (1.0f + exp(-k_value));
      v_value = v_value / (1.0f + exp(-v_value));
    }

    q_values[tid] = q_value;
    k_values[tid] = k_value;
    decay_values[tid] = active
      ? exp(decay_exp[(token * n_heads + head) * head_dim + tid])
      : 0.0f;
    q_sums[tid] = q_value * q_value;
    k_sums[tid] = k_value * k_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads >> 1; stride > 0; stride >>= 1) {
      if (tid < stride) {
        q_sums[tid] += q_sums[tid + stride];
        k_sums[tid] += k_sums[tid + stride];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
      const float q_scale = rsqrt(q_sums[0] + norm_eps) * rsqrt(float(head_dim));
      const float k_scale = rsqrt(k_sums[0] + norm_eps);
      const float beta = 1.0f / (1.0f + exp(-beta_raw[token * n_heads + head]));
      const uint head_state = head * head_dim * head_dim;

      float retrieved = 0.0f;
      for (uint key = 0; key < head_dim; ++key) {
        const uint state_index = head_state + key * head_dim + tid;
        const float previous = token == 0 ? initial_state[state_index] : state_output[state_index];
        const float decay = decay_values[key];
        retrieved += decay * previous * (k_values[key] * k_scale);
      }

      const float delta = beta * (v_value - retrieved);
      float value = 0.0f;
      for (uint key = 0; key < head_dim; ++key) {
        const uint state_index = head_state + key * head_dim + tid;
        const float previous = token == 0 ? initial_state[state_index] : state_output[state_index];
        const float decay = decay_values[key];
        const float normalized_k = k_values[key] * k_scale;
        const float next = decay * previous + normalized_k * delta;
        state_output[state_index] = next;
        value += next * (q_values[key] * q_scale);
      }
      scan_output[token * inner + head * head_dim + tid] = value;
    }
    // Every lane writes one recurrent-state column to device memory and
    // rereads that column across all keys on the next token. A threadgroup-only
    // fence does not make those device-buffer writes visible to the next loop.
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  }

  if (active) {
    const uint channel = head * head_dim + tid;
    for (uint k = 0; k < kernel_size; ++k) {
      conv_output[channel * kernel_size + k] = q_history[k];
      conv_output[(inner + channel) * kernel_size + k] = k_history[k];
      conv_output[(2 * inner + channel) * kernel_size + k] = v_history[k];
    }
  }
}
"#;

static KDA_PIPELINE: OnceLock<Mutex<HashMap<DeviceId, ComputePipeline>>> = OnceLock::new();

fn pipeline(device: &candle_core::MetalDevice) -> Result<ComputePipeline> {
    pipeline_from_source(device, &KDA_PIPELINE)
}

fn pipeline_from_source(
    device: &candle_core::MetalDevice,
    slot: &OnceLock<Mutex<HashMap<DeviceId, ComputePipeline>>>,
) -> Result<ComputePipeline> {
    let pipelines = slot.get_or_init(|| Mutex::new(HashMap::new()));
    let device_id = device.id();
    if let Some(pipeline) = pipelines.lock().unwrap().get(&device_id).cloned() {
        return Ok(pipeline);
    }
    let library = device
        .metal_device()
        .new_library_with_source(KDA_MSL, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function("dinf_glm5_kda_scan_f32", None)
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    let mut pipelines = pipelines.lock().unwrap();
    Ok(pipelines.entry(device_id).or_insert(pipeline).clone())
}

pub(super) fn precompile(device: &Device) -> anyhow::Result<()> {
    if device.is_metal() {
        pipeline(device.as_metal_device()?)?;
    }
    Ok(())
}

struct KdaScan<'a> {
    k: &'a Tensor,
    v: &'a Tensor,
    conv_weight: &'a Tensor,
    decay_exp: &'a Tensor,
    beta_raw: &'a Tensor,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    kernel_size: usize,
    norm_eps: f32,
}

impl KdaScan<'_> {
    fn run(
        &self,
        state: &MetalStorage,
        state_layout: &Layout,
        q: &MetalStorage,
        q_layout: &Layout,
        conv_state: &MetalStorage,
        conv_state_layout: &Layout,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let inner = self.n_heads * self.head_dim;
        let conv_dim = 3 * inner;
        let state_elements = self.n_heads * self.head_dim * self.head_dim;
        for (name, storage, layout, expected) in [
            ("state", state, state_layout, state_elements),
            ("q", q, q_layout, self.seq_len * inner),
            (
                "conv_state",
                conv_state,
                conv_state_layout,
                conv_dim * self.kernel_size,
            ),
        ] {
            if storage.dtype() != DType::F32
                || !layout.is_contiguous()
                || layout.shape().elem_count() != expected
            {
                candle_core::bail!(
                    "{name} must be contiguous f32 with {expected} elements for GLM5 KDA"
                )
            }
            if storage.device().id() != state.device().id() {
                candle_core::bail!("GLM5 KDA inputs are on different Metal devices")
            }
        }
        if !(1..=8).contains(&self.kernel_size)
            || self.head_dim == 0
            || self.head_dim > 256
            || self.n_heads == 0
        {
            candle_core::bail!("GLM5 KDA geometry exceeds the Metal kernel limits")
        }

        let (k_guard, k_layout) = self.k.storage_and_layout();
        let (v_guard, v_layout) = self.v.storage_and_layout();
        let (weight_guard, weight_layout) = self.conv_weight.storage_and_layout();
        let (decay_guard, decay_layout) = self.decay_exp.storage_and_layout();
        let (beta_guard, beta_layout) = self.beta_raw.storage_and_layout();
        let k = metal_storage("k", &k_guard, k_layout)?;
        let v = metal_storage("v", &v_guard, v_layout)?;
        let weight = metal_storage("conv_weight", &weight_guard, weight_layout)?;
        let decay = metal_storage("decay", &decay_guard, decay_layout)?;
        let beta = metal_storage("beta", &beta_guard, beta_layout)?;
        if k_layout.shape().elem_count() != self.seq_len * inner
            || v_layout.shape().elem_count() != self.seq_len * inner
            || weight_layout.shape().elem_count() != conv_dim * self.kernel_size
            || decay_layout.shape().elem_count() != self.seq_len * inner
            || beta_layout.shape().elem_count() != self.seq_len * self.n_heads
        {
            candle_core::bail!("GLM5 KDA gate or convolution shape differs")
        }
        for storage in [k, v, weight, decay, beta] {
            if storage.device().id() != state.device().id() {
                candle_core::bail!("GLM5 KDA inputs are on different Metal devices")
            }
        }

        let scan_elements = self.seq_len * inner;
        let conv_elements = conv_dim * self.kernel_size;
        if [state_elements, scan_elements, conv_elements]
            .iter()
            .any(|&n| n > u32::MAX as usize)
        {
            candle_core::bail!("GLM5 KDA output geometry exceeds u32")
        }
        let device = state.device();
        // Separate allocations matter even for decode: adding scan/conv to a
        // power-of-two recurrent state doubles Candle's pooled buffer bucket.
        // One dispatch writes all three outputs; no post-scan copies needed.
        let allocate = |elements, label| {
            device
                .new_buffer_builder()
                .with_size_for(elements, DType::F32)
                .with_label(label)
                .build()
        };
        let next_state = allocate(state_elements, "dinf-glm5-kda-state")?;
        let output = allocate(scan_elements, "dinf-glm5-kda-output")?;
        let next_conv = allocate(conv_elements, "dinf-glm5-kda-conv")?;
        let pipeline = pipeline(device)?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-glm5-kda-scan");
        encoder.set_compute_pipeline_state(&pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        let offset = |layout: &Layout| layout.start_offset() * DType::F32.size_in_bytes();
        encoder_ref.set_input_buffer(0, Some(state.buffer()), offset(state_layout));
        encoder_ref.set_input_buffer(1, Some(q.buffer()), offset(q_layout));
        encoder_ref.set_input_buffer(2, Some(k.buffer()), offset(k_layout));
        encoder_ref.set_input_buffer(3, Some(v.buffer()), offset(v_layout));
        encoder_ref.set_input_buffer(4, Some(conv_state.buffer()), offset(conv_state_layout));
        encoder_ref.set_input_buffer(5, Some(weight.buffer()), offset(weight_layout));
        encoder_ref.set_input_buffer(6, Some(decay.buffer()), offset(decay_layout));
        encoder_ref.set_input_buffer(7, Some(beta.buffer()), offset(beta_layout));
        encoder_ref.set_output_buffer(8, Some(&next_state), 0);
        encoder_ref.set_output_buffer(9, Some(&output), 0);
        encoder_ref.set_output_buffer(10, Some(&next_conv), 0);
        encoder_ref.set_bytes(11, &(self.seq_len as u32));
        encoder_ref.set_bytes(12, &(self.n_heads as u32));
        encoder_ref.set_bytes(13, &(self.head_dim as u32));
        encoder_ref.set_bytes(14, &(self.kernel_size as u32));
        encoder_ref.set_bytes(15, &self.norm_eps);
        let max_threads = pipeline.max_total_threads_per_threadgroup().max(1);
        let max_power_of_two = 1usize << max_threads.ilog2();
        let threads = self
            .head_dim
            .next_power_of_two()
            .min(256)
            .min(max_power_of_two);
        if threads < self.head_dim {
            candle_core::bail!("GLM5 KDA pipeline cannot cover every head dimension")
        }
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: self.n_heads,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );

        Ok((
            Tensor::from((
                Storage::Metal(MetalStorage::new(
                    output,
                    device.clone(),
                    scan_elements,
                    DType::F32,
                )),
                (self.seq_len, self.n_heads, self.head_dim),
            )),
            Tensor::from((
                Storage::Metal(MetalStorage::new(
                    next_conv,
                    device.clone(),
                    conv_elements,
                    DType::F32,
                )),
                (conv_dim, self.kernel_size),
            )),
            Tensor::from((
                Storage::Metal(MetalStorage::new(
                    next_state,
                    device.clone(),
                    state_elements,
                    DType::F32,
                )),
                (self.n_heads, self.head_dim, self.head_dim),
            )),
        ))
    }
}

fn metal_storage<'a>(
    name: &str,
    storage: &'a candle_core::Storage,
    layout: &Layout,
) -> Result<&'a MetalStorage> {
    if storage.dtype() != DType::F32 || !layout.is_contiguous() {
        candle_core::bail!("GLM5 KDA {name} must be contiguous f32")
    }
    match storage {
        candle_core::Storage::Metal(storage) => Ok(storage),
        _ => candle_core::bail!("GLM5 KDA {name} must use Metal storage"),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn scan(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    conv_state: &Tensor,
    recurrent_state: &Tensor,
    conv_weight: &Tensor,
    decay_exp: &Tensor,
    beta_raw: &Tensor,
    n_heads: usize,
    head_dim: usize,
    kernel_size: usize,
    norm_eps: f32,
) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
    let (seq_len, projected_inner) = q.dims2()?;
    let inner = n_heads * head_dim;
    anyhow::ensure!(
        seq_len > 0
            && projected_inner == inner
            && k.dims2()? == (seq_len, inner)
            && v.dims2()? == (seq_len, inner),
        "invalid GLM5 KDA Q/K/V shape"
    );
    let (state_guard, state_layout) = recurrent_state.storage_and_layout();
    let (q_guard, q_layout) = q.storage_and_layout();
    let (conv_guard, conv_layout) = conv_state.storage_and_layout();
    KdaScan {
        k,
        v,
        conv_weight,
        decay_exp,
        beta_raw,
        seq_len,
        n_heads,
        head_dim,
        kernel_size,
        norm_eps,
    }
    .run(
        metal_storage("state", &state_guard, state_layout)?,
        state_layout,
        metal_storage("q", &q_guard, q_layout)?,
        q_layout,
        metal_storage("conv", &conv_guard, conv_layout)?,
        conv_layout,
    )
    .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kda_pipeline_cache_is_scoped_to_device_id() -> anyhow::Result<()> {
        let Some(first) = crate::test_metal_device() else {
            return Ok(());
        };
        let second = Device::new_metal(0)?;
        let first_id = first.as_metal_device()?.id();
        let second_id = second.as_metal_device()?.id();
        assert_ne!(first_id, second_id);

        let cache = OnceLock::new();
        pipeline_from_source(first.as_metal_device()?, &cache)?;
        pipeline_from_source(second.as_metal_device()?, &cache)?;
        pipeline_from_source(first.as_metal_device()?, &cache)?;

        let pipelines = cache.get().expect("cache initialized").lock().unwrap();
        assert_eq!(pipelines.len(), 2);
        assert!(pipelines.contains_key(&first_id));
        assert!(pipelines.contains_key(&second_id));
        Ok(())
    }

    fn silu(value: f32) -> f32 {
        value / (1.0 + (-value).exp())
    }

    fn split_qkv(qkv: &Tensor, inner: usize) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
        Ok((
            qkv.narrow(1, 0, inner)?.contiguous()?,
            qkv.narrow(1, inner, inner)?.contiguous()?,
            qkv.narrow(1, 2 * inner, inner)?.contiguous()?,
        ))
    }

    #[test]
    fn scan_fences_device_state_between_tokens() {
        let state_write = KDA_MSL
            .find("state_output[state_index] = next;")
            .expect("KDA recurrent-state write");
        let final_conv_write = KDA_MSL
            .find("conv_output[channel * kernel_size + k]")
            .expect("KDA final convolution-state write");
        let device_fence = KDA_MSL[state_write..]
            .find("threadgroup_barrier(mem_flags::mem_device")
            .map(|offset| state_write + offset)
            .expect("KDA device-memory fence after recurrent-state write");
        assert!(device_fence < final_conv_write);
    }

    #[test]
    fn scan_computes_each_key_decay_once() {
        assert_eq!(
            KDA_MSL.matches("exp(decay_exp[").count(),
            1,
            "the scan must not recompute key-invariant decay exponentials per output lane"
        );
        assert_eq!(
            KDA_MSL.matches("decay_values[key]").count(),
            2,
            "retrieval and update must reuse the staged per-key decay"
        );
    }

    #[test]
    fn prompt_and_decode_state_use_compact_independent_buffers() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (heads, dim, kernel) = (2, 128, 4);
        let inner = heads * dim;
        for seq in [1, 129] {
            let projected = Tensor::ones((seq, inner), DType::F32, &device)?;
            let conv = Tensor::zeros((3 * inner, kernel), DType::F32, &device)?;
            let recurrent = Tensor::zeros((heads, dim, dim), DType::F32, &device)?;
            let weights = Tensor::ones((3 * inner, kernel), DType::F32, &device)?;
            let decay = Tensor::full(-1f32, (seq, heads, dim), &device)?;
            let beta = Tensor::zeros((seq, heads), DType::F32, &device)?;
            let (output, conv, recurrent) = scan(
                &projected, &projected, &projected, &conv, &recurrent, &weights, &decay, &beta,
                heads, dim, kernel, 1e-6,
            )?;
            let output_buffer = {
                let (storage, _) = output.storage_and_layout();
                let candle_core::Storage::Metal(storage) = &*storage else {
                    unreachable!()
                };
                storage.buffer().clone()
            };
            for state in [&conv, &recurrent] {
                let (storage, layout) = state.storage_and_layout();
                let candle_core::Storage::Metal(storage) = &*storage else {
                    unreachable!()
                };
                assert_eq!(layout.start_offset(), 0);
                assert!(storage.buffer().length() <= (state.elem_count() * 4).next_power_of_two());
                assert_ne!(storage.buffer(), &output_buffer);
            }
            drop(output_buffer);
            drop(output);
            device.synchronize()?;
            for state in [&conv, &recurrent] {
                assert!(
                    state
                        .flatten_all()?
                        .to_vec1::<f32>()?
                        .iter()
                        .all(|value| value.is_finite())
                );
            }
        }
        Ok(())
    }

    #[test]
    fn fused_scan_matches_scalar_reference_at_glm_width() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        // Use GLM's real 128-lane head width and a multi-token prefill.  Every
        // lane writes one recurrent-state column that all lanes consume on the
        // following token, so this geometry exercises the device-memory fence
        // between scan iterations.
        let (seq_len, heads, head_dim, kernel) = (26usize, 2usize, 128usize, 4usize);
        let inner = heads * head_dim;
        let conv_dim = 3 * inner;
        let qkv_values = (0..seq_len * conv_dim)
            .map(|index| ((index * 13 % 41) as f32 - 20.0) / 17.0)
            .collect::<Vec<_>>();
        let weight_values = (0..conv_dim * kernel)
            .map(|index| ((index * 7 % 19) as f32 - 9.0) / 13.0)
            .collect::<Vec<_>>();
        let state_values = (0..heads * head_dim * head_dim)
            .map(|index| ((index * 5 % 23) as f32 - 11.0) / 29.0)
            .collect::<Vec<_>>();
        let decay_values = (0..seq_len * inner)
            .map(|index| -0.05 - (index % head_dim) as f32 * 0.03)
            .collect::<Vec<_>>();
        let beta_values = (0..seq_len * heads)
            .map(|index| (index as f32 - 1.0) * 0.4)
            .collect::<Vec<_>>();

        let qkv = Tensor::from_vec(qkv_values.clone(), (seq_len, conv_dim), &device)?;
        let conv = Tensor::zeros((conv_dim, kernel), DType::F32, &device)?;
        let state = Tensor::from_vec(state_values.clone(), (heads, head_dim, head_dim), &device)?;
        let weight = Tensor::from_vec(weight_values.clone(), (conv_dim, kernel), &device)?;
        let decay = Tensor::from_vec(decay_values.clone(), (seq_len, heads, head_dim), &device)?;
        let beta = Tensor::from_vec(beta_values.clone(), (seq_len, heads), &device)?;
        let (q, k, v) = split_qkv(&qkv, inner)?;
        let (actual_output, actual_conv, actual_state) = scan(
            &q, &k, &v, &conv, &state, &weight, &decay, &beta, heads, head_dim, kernel, 1e-5,
        )?;
        device.synchronize()?;

        let mut expected_conv = vec![0.0f32; conv_dim * kernel];
        let mut expected_state = state_values;
        let mut expected_output = vec![0.0f32; seq_len * inner];
        for token in 0..seq_len {
            let mut conv_output = vec![0.0f32; conv_dim];
            for channel in 0..conv_dim {
                let base = channel * kernel;
                for slot in 1..kernel {
                    expected_conv[base + slot - 1] = expected_conv[base + slot];
                }
                expected_conv[base + kernel - 1] = qkv_values[token * conv_dim + channel];
                let value = (0..kernel)
                    .map(|slot| expected_conv[base + slot] * weight_values[base + slot])
                    .sum::<f32>();
                conv_output[channel] = silu(value);
            }

            for head in 0..heads {
                let q = &conv_output[head * head_dim..(head + 1) * head_dim];
                let k = &conv_output[inner + head * head_dim..inner + (head + 1) * head_dim];
                let v =
                    &conv_output[2 * inner + head * head_dim..2 * inner + (head + 1) * head_dim];
                let q_scale = 1.0
                    / (q.iter().map(|value| value * value).sum::<f32>() + 1e-5).sqrt()
                    / (head_dim as f32).sqrt();
                let k_scale =
                    1.0 / (k.iter().map(|value| value * value).sum::<f32>() + 1e-5).sqrt();
                let beta = 1.0 / (1.0 + (-beta_values[token * heads + head]).exp());
                for value in 0..head_dim {
                    let mut retrieved = 0.0f32;
                    for key in 0..head_dim {
                        let index = (head * head_dim + key) * head_dim + value;
                        let decay = decay_values[(token * heads + head) * head_dim + key].exp();
                        retrieved += decay * expected_state[index] * k[key] * k_scale;
                    }
                    let delta = beta * (v[value] - retrieved);
                    let mut output = 0.0f32;
                    for key in 0..head_dim {
                        let index = (head * head_dim + key) * head_dim + value;
                        let decay = decay_values[(token * heads + head) * head_dim + key].exp();
                        expected_state[index] =
                            decay * expected_state[index] + k[key] * k_scale * delta;
                        output += expected_state[index] * q[key] * q_scale;
                    }
                    expected_output[token * inner + head * head_dim + value] = output;
                }
            }
        }

        let actual_output = actual_output
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let actual_conv = actual_conv
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let actual_state = actual_state
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (name, actual, expected) in [
            ("output", actual_output, expected_output),
            ("conv", actual_conv, expected_conv),
            ("state", actual_state, expected_state),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                anyhow::ensure!(
                    (actual - expected).abs() < 3e-4,
                    "{name}[{index}] differs: {actual} != {expected}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn fused_scan_split_decode_matches_one_contiguous_scan() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (seq_len, split, heads, head_dim, kernel) = (5usize, 4usize, 2usize, 4usize, 3usize);
        let inner = heads * head_dim;
        let conv_dim = 3 * inner;
        let qkv_values = (0..seq_len * conv_dim)
            .map(|index| ((index * 17 % 53) as f32 - 26.0) / 19.0)
            .collect::<Vec<_>>();
        let weight_values = (0..conv_dim * kernel)
            .map(|index| ((index * 11 % 29) as f32 - 14.0) / 23.0)
            .collect::<Vec<_>>();
        let state_values = (0..heads * head_dim * head_dim)
            .map(|index| ((index * 7 % 31) as f32 - 15.0) / 37.0)
            .collect::<Vec<_>>();
        let decay_values = (0..seq_len * inner)
            .map(|index| -0.04 - (index % head_dim) as f32 * 0.02)
            .collect::<Vec<_>>();
        let beta_values = (0..seq_len * heads)
            .map(|index| (index as f32 - 2.0) * 0.3)
            .collect::<Vec<_>>();

        let qkv = Tensor::from_vec(qkv_values, (seq_len, conv_dim), &device)?;
        let conv = Tensor::zeros((conv_dim, kernel), DType::F32, &device)?;
        let state = Tensor::from_vec(state_values, (heads, head_dim, head_dim), &device)?;
        let weight = Tensor::from_vec(weight_values, (conv_dim, kernel), &device)?;
        let decay = Tensor::from_vec(decay_values, (seq_len, heads, head_dim), &device)?;
        let beta = Tensor::from_vec(beta_values, (seq_len, heads), &device)?;
        let (q, k, v) = split_qkv(&qkv, inner)?;

        let (full_output, full_conv, full_state) = scan(
            &q, &k, &v, &conv, &state, &weight, &decay, &beta, heads, head_dim, kernel, 1e-5,
        )?;
        let (prefix_output, prefix_conv, prefix_state) = scan(
            &q.narrow(0, 0, split)?,
            &k.narrow(0, 0, split)?,
            &v.narrow(0, 0, split)?,
            &conv,
            &state,
            &weight,
            &decay.narrow(0, 0, split)?,
            &beta.narrow(0, 0, split)?,
            heads,
            head_dim,
            kernel,
            1e-5,
        )?;
        let (tail_output, tail_conv, tail_state) = scan(
            &q.narrow(0, split, 1)?,
            &k.narrow(0, split, 1)?,
            &v.narrow(0, split, 1)?,
            &prefix_conv,
            &prefix_state,
            &weight,
            &decay.narrow(0, split, 1)?,
            &beta.narrow(0, split, 1)?,
            heads,
            head_dim,
            kernel,
            1e-5,
        )?;
        device.synchronize()?;

        let full_output = full_output.to_device(&Device::Cpu)?.to_vec3::<f32>()?;
        let prefix_output = prefix_output.to_device(&Device::Cpu)?.to_vec3::<f32>()?;
        let tail_output = tail_output.to_device(&Device::Cpu)?.to_vec3::<f32>()?;
        for token in 0..split {
            for head in 0..heads {
                for value in 0..head_dim {
                    let expected = full_output[token][head][value];
                    let actual = prefix_output[token][head][value];
                    anyhow::ensure!(
                        (actual - expected).abs() < 3e-4,
                        "prefix output[{token}][{head}][{value}] differs: {actual} != {expected}"
                    );
                }
            }
        }
        for head in 0..heads {
            for value in 0..head_dim {
                let expected = full_output[split][head][value];
                let actual = tail_output[0][head][value];
                anyhow::ensure!(
                    (actual - expected).abs() < 3e-4,
                    "tail output[{head}][{value}] differs: {actual} != {expected}"
                );
            }
        }

        let full_conv = full_conv.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
        let tail_conv = tail_conv.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
        let full_state = full_state
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let tail_state = tail_state
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (name, actual, expected) in [
            (
                "conv",
                tail_conv.into_iter().flatten().collect::<Vec<_>>(),
                full_conv.into_iter().flatten().collect::<Vec<_>>(),
            ),
            ("state", tail_state, full_state),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                anyhow::ensure!(
                    (actual - expected).abs() < 3e-4,
                    "{name}[{index}] differs: {actual} != {expected}"
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "manual Metal microbenchmark"]
    fn benchmark_direct_projection_inputs() -> anyhow::Result<()> {
        use std::time::{Duration, Instant};

        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (seq_len, heads, head_dim, kernel) = (1usize, 64usize, 128usize, 4usize);
        let inner = heads * head_dim;
        let conv_dim = 3 * inner;
        let q = Tensor::zeros((seq_len, inner), DType::F32, &device)?;
        let k = Tensor::zeros((seq_len, inner), DType::F32, &device)?;
        let v = Tensor::zeros((seq_len, inner), DType::F32, &device)?;
        let conv = Tensor::zeros((conv_dim, kernel), DType::F32, &device)?;
        let state = Tensor::zeros((heads, head_dim, head_dim), DType::F32, &device)?;
        let weight = Tensor::zeros((conv_dim, kernel), DType::F32, &device)?;
        let decay = Tensor::full(-0.5f32, (seq_len, heads, head_dim), &device)?;
        let beta = Tensor::zeros((seq_len, heads), DType::F32, &device)?;
        let iterations = 1_000usize;

        let measure = |packed_inputs: bool| -> anyhow::Result<Duration> {
            device.synchronize()?;
            let started = Instant::now();
            let mut last = None;
            for _ in 0..iterations {
                let (q_input, k_input, v_input) = if packed_inputs {
                    // Reproduce the former production input path: materialize one
                    // packed QKV buffer before dispatching the scan.
                    let qkv = Tensor::cat(&[&q, &k, &v], 1)?;
                    (
                        qkv.narrow(1, 0, inner)?,
                        qkv.narrow(1, inner, inner)?,
                        qkv.narrow(1, 2 * inner, inner)?,
                    )
                } else {
                    (q.clone(), k.clone(), v.clone())
                };
                last = Some(scan(
                    &q_input, &k_input, &v_input, &conv, &state, &weight, &decay, &beta, heads,
                    head_dim, kernel, 1e-5,
                )?);
            }
            device.synchronize()?;
            std::hint::black_box(last);
            Ok(started.elapsed())
        };

        // Compile/warm both paths before sampling.
        measure(false)?;
        measure(true)?;
        let mut packed_runs = Vec::with_capacity(5);
        let mut direct_runs = Vec::with_capacity(5);
        for round in 0usize..5 {
            if round.is_multiple_of(2) {
                direct_runs.push(measure(false)?);
                packed_runs.push(measure(true)?);
            } else {
                packed_runs.push(measure(true)?);
                direct_runs.push(measure(false)?);
            }
        }
        packed_runs.sort_unstable();
        direct_runs.sort_unstable();
        let packed = packed_runs[packed_runs.len() / 2];
        let direct = direct_runs[direct_runs.len() / 2];
        eprintln!(
            "GLM_KDA_INPUT_BENCH iterations={iterations} packed_ms={:.3} direct_ms={:.3} speedup={:.3}x",
            packed.as_secs_f64() * 1_000.0,
            direct.as_secs_f64() * 1_000.0,
            packed.as_secs_f64() / direct.as_secs_f64()
        );
        Ok(())
    }
}
