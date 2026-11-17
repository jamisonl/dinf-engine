//! Single-query attention over an F16 KV cache with F32 query, accumulation
//! and output, for decode on Metal.
//!
//! The kernels are Candle's (MLX-derived) two-pass `sdpa_vector` with mixed
//! types: keys and values are read from the F16 cache directly instead of
//! widening the whole cache to F32 for every layer, and the partial outputs,
//! sums and maxima stay in F32. Half values are exact in F32, so this matches
//! running F32 SDPA on a widened cache, without the widened temporaries or a
//! device wait to release them.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{CpuStorage, CustomOp3, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;

// Adapted from Candle's scaled_dot_product_attention.metal (MLX), MIT.
const SDPA_F16KV_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr int BLOCKS = 32;

template <int D>
[[kernel]] void dinf_sdpa_f16kv_pass1(
    const device float* queries [[buffer(0)]],
    const device half* keys [[buffer(1)]],
    const device half* values [[buffer(2)]],
    device float* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    constant int& gqa_factor [[buffer(6)]],
    constant int& N [[buffer(7)]],
    constant uint& k_head_stride [[buffer(8)]],
    constant uint& v_head_stride [[buffer(9)]],
    constant uint& k_batch_stride [[buffer(10)]],
    constant uint& v_batch_stride [[buffer(11)]],
    constant int& q_heads [[buffer(12)]],
    constant float& scale [[buffer(13)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 8;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  constexpr int stride = BN * D;

  float q[elem_per_thread];
  float o[elem_per_thread];

  threadgroup float outputs[BN * BD];
  threadgroup float max_scores[BN];
  threadgroup float sum_exp_scores[BN];

  const int block_idx = tid.z;
  const int head_idx = tid.y;
  const int batch = tid.x;
  const int kv_head_idx = head_idx / gqa_factor;
  const int row = batch * q_heads + head_idx;
  queries += row * D + simd_lid * elem_per_thread;
  keys += batch * k_batch_stride + kv_head_idx * k_head_stride
      + (block_idx * BN + simd_gid) * D + simd_lid * elem_per_thread;
  values += batch * v_batch_stride + kv_head_idx * v_head_stride
      + (block_idx * BN + simd_gid) * D + simd_lid * elem_per_thread;
  out += row * BLOCKS * D + block_idx * D + simd_lid * elem_per_thread;
  sums += row * BLOCKS + block_idx;
  maxs += row * BLOCKS + block_idx;

  // Keep the query's half rounding, as the widened-cache path does.
  for (int i = 0; i < elem_per_thread; i++) {
    q[i] = scale * float(half(queries[i]));
    o[i] = 0;
  }

  float max_score = -1e9;
  float sum_exp_score = 0;
  for (int i = block_idx * BN + simd_gid; i < N; i += BLOCKS * BN) {
    float score = 0;
    for (int j = 0; j < elem_per_thread; j++) {
      score += q[j] * float(keys[j]);
    }
    score = simd_sum(score);
    const float new_max = max(max_score, score);
    const float factor = fast::exp(max_score - new_max);
    const float exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    for (int j = 0; j < elem_per_thread; j++) {
      o[j] = o[j] * factor + exp_score * float(values[j]);
    }
    keys += BLOCKS * stride;
    values += BLOCKS * stride;
  }

  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = (simd_lid < BN) ? max_scores[simd_lid] : -1e9;
  const float new_max = simd_max(max_score);
  const float factor = fast::exp(max_score - new_max);
  sum_exp_score = (simd_lid < BN) ? sum_exp_scores[simd_lid] : 0;
  sum_exp_score = simd_sum(sum_exp_score * factor);
  if (simd_gid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = new_max;
  }
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BN + simd_gid] = o[i] * fast::exp(max_scores[simd_gid] - new_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0) {
      float output = outputs[simd_lid * BN];
      for (int j = 1; j < BN; j++) {
        output += outputs[simd_lid * BN + j];
      }
      out[i] = output;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <int D>
[[kernel]] void dinf_sdpa_f16kv_pass2(
    const device float* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device float* out [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;

  float o[elem_per_thread];
  threadgroup float outputs[BN * BD];

  const int row = tid.y;
  partials += row * BLOCKS * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += row * BLOCKS;
  maxs += row * BLOCKS;
  out += row * D + simd_gid * elem_per_thread;

  const float max_score = maxs[simd_lid];
  const float new_max = simd_max(max_score);
  const float factor = fast::exp(max_score - new_max);
  const float sum_exp_score = simd_sum(sums[simd_lid] * factor);

  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = partials[i];
  }
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) / sum_exp_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = o[i];
    }
  }
}

template [[host_name("dinf_sdpa_f16kv_pass1_64")]] [[kernel]] decltype(dinf_sdpa_f16kv_pass1<64>) dinf_sdpa_f16kv_pass1<64>;
template [[host_name("dinf_sdpa_f16kv_pass1_128")]] [[kernel]] decltype(dinf_sdpa_f16kv_pass1<128>) dinf_sdpa_f16kv_pass1<128>;
template [[host_name("dinf_sdpa_f16kv_pass1_256")]] [[kernel]] decltype(dinf_sdpa_f16kv_pass1<256>) dinf_sdpa_f16kv_pass1<256>;
template [[host_name("dinf_sdpa_f16kv_pass2_64")]] [[kernel]] decltype(dinf_sdpa_f16kv_pass2<64>) dinf_sdpa_f16kv_pass2<64>;
template [[host_name("dinf_sdpa_f16kv_pass2_128")]] [[kernel]] decltype(dinf_sdpa_f16kv_pass2<128>) dinf_sdpa_f16kv_pass2<128>;
template [[host_name("dinf_sdpa_f16kv_pass2_256")]] [[kernel]] decltype(dinf_sdpa_f16kv_pass2<256>) dinf_sdpa_f16kv_pass2<256>;
"#;

// Causal prefill attention over an F16 cache (flash-attention tiling on Metal 4
// tensor ops). One threadgroup handles 32 query rows of one head. Q is staged
// as half (the query rounding the widened path also applies); S = Q K^T and
// P V run on the tensor units with float accumulation. P stays in float
// (rounding it to half, as llama.cpp's flash attention does, is ~1.5x faster
// here but measurably moves routing-sensitive MoE rows). The causal bound is
// computed per row: query i of an Lq-row chunk sees keys [0, Lk - Lq + i].
const SDPA_F16KV_PREFILL_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

template <int D>
kernel void dinf_sdpa_f16kv_prefill(
    device half* queries [[buffer(0)]],
    device half* keys [[buffer(1)]],
    device half* values [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int& gqa_factor [[buffer(4)]],
    constant int& q_len [[buffer(5)]],
    constant int& kv_len [[buffer(6)]],
    constant uint& k_head_stride [[buffer(7)]],
    constant uint& v_head_stride [[buffer(8)]],
    constant uint& k_batch_stride [[buffer(9)]],
    constant uint& v_batch_stride [[buffer(10)]],
    constant int& q_heads [[buffer(11)]],
    constant float& scale [[buffer(12)]],
    threadgroup uchar* shared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
  constexpr int BQ = 32;
  constexpr int BK = 32;
  constexpr int THREADS = 128;
  constexpr int ROW_THREADS = THREADS / BQ;   // 4 threads per query row
  constexpr int COLS = D / ROW_THREADS;        // output columns per thread
  constexpr int SCORES = BK / ROW_THREADS;     // scores per thread per tile

  threadgroup float* s_tile = (threadgroup float*)shared;              // [BQ][BK]
  threadgroup float* p_tile = s_tile + BQ * BK;                        // [BQ][BK]
  threadgroup float* pv_tile = p_tile + BQ * BK;                       // [BQ][D]

  const int q_start = int(group.x) * BQ;
  const int head = int(group.y);
  const int batch = int(group.z);
  const int kv_head = head / gqa_factor;
  const int offset = kv_len - q_len;

  // Device tensors; the tensor ops bound partial tiles by these extents.
  auto q_tensor = tensor(
      queries + ((batch * q_heads + head) * q_len) * D,
      dextents<int32_t, 2>(D, q_len), array<int, 2>({1, D}));
  auto k_tensor = tensor(
      keys + batch * k_batch_stride + kv_head * k_head_stride,
      dextents<int32_t, 2>(D, kv_len), array<int, 2>({1, D}));
  auto v_tensor = tensor(
      values + batch * v_batch_stride + kv_head * v_head_stride,
      dextents<int32_t, 2>(D, kv_len), array<int, 2>({1, D}));
  auto p_tensor = tensor(p_tile, dextents<int32_t, 2>(BK, BQ));
  auto s_out = tensor(s_tile, dextents<int32_t, 2>(BK, BQ));
  auto pv_out = tensor(pv_tile, dextents<int32_t, 2>(D, BQ));
  auto q_slice = q_tensor.slice(0, q_start);

  mpp::tensor_ops::matmul2d<
      mpp::tensor_ops::matmul2d_descriptor(BQ, BK, D, false, true, false,
          mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
      execution_simdgroups<4>> qk;
  mpp::tensor_ops::matmul2d<
      mpp::tensor_ops::matmul2d_descriptor(BQ, D, BK, false, false, false,
          mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
      execution_simdgroups<4>> pv;

  const int row = tid / ROW_THREADS;
  const int sub = tid % ROW_THREADS;
  const int q_row = q_start + row;
  const int visible_end = min(kv_len, offset + q_row + 1);  // exclusive
  const int tile_end = min(kv_len, offset + min(q_start + BQ, q_len));
  float o[COLS];
  for (int c = 0; c < COLS; ++c) o[c] = 0.0f;
  float m = -INFINITY;
  float l = 0.0f;

  for (int k_start = 0; k_start < tile_end; k_start += BK) {
    {
      auto k_slice = k_tensor.slice(0, k_start);
      auto s = qk.template get_destination_cooperative_tensor<
          decltype(q_slice), decltype(k_slice), float>();
      qk.run(q_slice, k_slice, s);
      s.store(s_out);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scores[SCORES];
    float tile_max = -INFINITY;
    for (int j = 0; j < SCORES; ++j) {
      const int col = sub * SCORES + j;
      const float value = k_start + col < visible_end
          ? s_tile[row * BK + col] * scale : -INFINITY;
      scores[j] = value;
      tile_max = max(tile_max, value);
    }
    for (ushort width = ROW_THREADS / 2; width > 0; width /= 2) {
      tile_max = max(tile_max, simd_shuffle_xor(tile_max, width));
    }
    const float new_m = max(m, tile_max);
    const float alpha = new_m == -INFINITY ? 1.0f : exp(m - new_m);
    float partial = 0.0f;
    for (int j = 0; j < SCORES; ++j) {
      const float p = scores[j] == -INFINITY ? 0.0f : exp(scores[j] - new_m);
      partial += p;
      p_tile[row * BK + sub * SCORES + j] = p;
    }
    for (ushort width = ROW_THREADS / 2; width > 0; width /= 2) {
      partial += simd_shuffle_xor(partial, width);
    }
    l = l * alpha + partial;
    m = new_m;
    for (int c = 0; c < COLS; ++c) o[c] *= alpha;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      auto v_slice = v_tensor.slice(0, k_start);
      auto r = pv.template get_destination_cooperative_tensor<
          decltype(p_tensor), decltype(v_slice), float>();
      pv.run(p_tensor, v_slice, r);
      r.store(pv_out);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int c = 0; c < COLS; ++c) o[c] += pv_tile[row * D + sub * COLS + c];
  }

  if (q_row < q_len) {
    // Token-major output [batch, q_len, heads, D]: the caller's head-merge
    // layout becomes a view instead of a copy.
    device float* o_row = out + ((batch * q_len + q_row) * q_heads + head) * D;
    const float inv = 1.0f / l;
    for (int c = 0; c < COLS; ++c) o_row[sub * COLS + c] = o[c] * inv;
  }
}

template [[host_name("dinf_sdpa_f16kv_prefill_64")]] kernel decltype(dinf_sdpa_f16kv_prefill<64>) dinf_sdpa_f16kv_prefill<64>;
template [[host_name("dinf_sdpa_f16kv_prefill_128")]] kernel decltype(dinf_sdpa_f16kv_prefill<128>) dinf_sdpa_f16kv_prefill<128>;
"#;

const BLOCKS: usize = 32;

static PIPELINES: OnceLock<Mutex<HashMap<(DeviceId, &'static str), ComputePipeline>>> =
    OnceLock::new();

fn pipeline(device: &candle_core::MetalDevice, name: &'static str) -> Result<ComputePipeline> {
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(pipeline) = cache.lock().unwrap().get(&(device.id(), name)) {
        return Ok(pipeline.clone());
    }
    let library = device
        .metal_device()
        .new_library_with_source(SDPA_F16KV_MSL, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function(name, None)
        .map_err(candle_core::Error::wrap)?;
    let compiled = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    let mut cache = cache.lock().unwrap();
    Ok(cache.entry((device.id(), name)).or_insert(compiled).clone())
}

fn kernel_names(head_dim: usize) -> Option<(&'static str, &'static str)> {
    match head_dim {
        64 => Some(("dinf_sdpa_f16kv_pass1_64", "dinf_sdpa_f16kv_pass2_64")),
        128 => Some(("dinf_sdpa_f16kv_pass1_128", "dinf_sdpa_f16kv_pass2_128")),
        256 => Some(("dinf_sdpa_f16kv_pass1_256", "dinf_sdpa_f16kv_pass2_256")),
        _ => None,
    }
}

/// Decode attention for one query row per sequence: `q` is `[batch, heads, 1,
/// D]` F32 or F16, `k`/`v` are `[batch, kv_heads, kv_seq, D]` F16 views of the
/// cache (rows contiguous; head and batch strides may include spare
/// capacity). Returns `[batch, heads, 1, D]` F32, or `None` when the inputs
/// do not fit this kernel.
pub(crate) fn try_decode(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Option<Tensor>> {
    if !q.device().is_metal()
        || !q.device().same_device(k.device())
        || !q.device().same_device(v.device())
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
        || !matches!(q.dtype(), DType::F32 | DType::F16)
        || [q, k, v].iter().any(|t| t.rank() != 4)
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Ok(None);
    }
    let (batch, q_heads, q_seq, head_dim) = q.dims4()?;
    let (k_batch, kv_heads, kv_seq, k_dim) = k.dims4()?;
    if q_seq != 1
        || kv_seq == 0
        || k.dims4()? != v.dims4()?
        || k_batch != batch
        || k_dim != head_dim
        || kv_heads == 0
        || !q_heads.is_multiple_of(kv_heads)
        || kernel_names(head_dim).is_none()
    {
        return Ok(None);
    }
    for t in [k, v] {
        let stride = t.stride();
        if stride[3] != 1 || stride[2] != head_dim {
            return Ok(None);
        }
    }
    let q = q.to_dtype(DType::F32)?.contiguous()?;
    let output = q.apply_op3_no_bwd(
        k,
        v,
        &DecodeSdpa {
            scale,
            k_strides: [k.stride()[0], k.stride()[1]],
            v_strides: [v.stride()[0], v.stride()[1]],
        },
    )?;
    Ok(Some(output))
}

struct DecodeSdpa {
    scale: f32,
    k_strides: [usize; 2],
    v_strides: [usize; 2],
}

impl CustomOp3 for DecodeSdpa {
    fn name(&self) -> &'static str {
        "dinf-sdpa-f16kv-decode"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("F16-cache decode attention is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        q: &MetalStorage,
        q_layout: &Layout,
        k: &MetalStorage,
        k_layout: &Layout,
        v: &MetalStorage,
        v_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (batch, q_heads, _, head_dim) = q_layout.shape().dims4()?;
        let (_, kv_heads, kv_seq, _) = k_layout.shape().dims4()?;
        if q.dtype() != DType::F32
            || k.dtype() != DType::F16
            || v.dtype() != DType::F16
            || !q_layout.is_contiguous()
            || k_layout.stride()[..2] != self.k_strides
            || v_layout.stride()[..2] != self.v_strides
            || k_layout.stride()[2..] != [head_dim, 1]
            || v_layout.stride()[2..] != [head_dim, 1]
            || q.device().id() != k.device().id()
            || q.device().id() != v.device().id()
        {
            candle_core::bail!("F16-cache decode attention tensor metadata changed")
        }
        let (pass1_name, pass2_name) =
            kernel_names(head_dim).ok_or_else(|| candle_core::Error::Msg("head dim".into()))?;
        let to_u32 = |value: usize| {
            u32::try_from(value).map_err(|_| {
                candle_core::Error::Msg("decode attention geometry exceeds u32".into())
            })
        };
        let rows = batch * q_heads;
        let device = q.device();
        let partials = device
            .new_buffer_builder()
            .with_size_for(rows * BLOCKS * head_dim, DType::F32)
            .with_label("dinf-sdpa-f16kv-partials")
            .build()?;
        let sums = device
            .new_buffer_builder()
            .with_size_for(rows * BLOCKS, DType::F32)
            .with_label("dinf-sdpa-f16kv-sums")
            .build()?;
        let maxs = device
            .new_buffer_builder()
            .with_size_for(rows * BLOCKS, DType::F32)
            .with_label("dinf-sdpa-f16kv-maxs")
            .build()?;
        let output = device
            .new_buffer_builder()
            .with_size_for(rows * head_dim, DType::F32)
            .with_label("dinf-sdpa-f16kv-output")
            .build()?;
        let f16 = DType::F16.size_in_bytes();
        {
            let pipeline = pipeline(device, pass1_name)?;
            let encoder = device.command_encoder()?;
            encoder.set_label("dinf-sdpa-f16kv-pass1");
            encoder.set_compute_pipeline_state(&pipeline);
            let e: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
            e.set_input_buffer(
                0,
                Some(q.buffer()),
                q_layout.start_offset() * DType::F32.size_in_bytes(),
            );
            e.set_input_buffer(1, Some(k.buffer()), k_layout.start_offset() * f16);
            e.set_input_buffer(2, Some(v.buffer()), v_layout.start_offset() * f16);
            e.set_output_buffer(3, Some(&partials), 0);
            e.set_output_buffer(4, Some(&sums), 0);
            e.set_output_buffer(5, Some(&maxs), 0);
            e.set_bytes(6, &(to_u32(q_heads / kv_heads)? as i32));
            e.set_bytes(7, &(to_u32(kv_seq)? as i32));
            e.set_bytes(8, &to_u32(self.k_strides[1])?);
            e.set_bytes(9, &to_u32(self.v_strides[1])?);
            e.set_bytes(10, &to_u32(self.k_strides[0])?);
            e.set_bytes(11, &to_u32(self.v_strides[0])?);
            e.set_bytes(12, &(to_u32(q_heads)? as i32));
            e.set_bytes(13, &self.scale);
            e.dispatch_thread_groups(
                objc2_metal::MTLSize {
                    width: batch,
                    height: q_heads,
                    depth: BLOCKS,
                },
                objc2_metal::MTLSize {
                    width: 8 * 32,
                    height: 1,
                    depth: 1,
                },
            );
        }
        {
            let pipeline = pipeline(device, pass2_name)?;
            let encoder = device.command_encoder()?;
            encoder.set_label("dinf-sdpa-f16kv-pass2");
            encoder.set_compute_pipeline_state(&pipeline);
            let e: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
            e.set_input_buffer(0, Some(&partials), 0);
            e.set_input_buffer(1, Some(&sums), 0);
            e.set_input_buffer(2, Some(&maxs), 0);
            e.set_output_buffer(3, Some(&output), 0);
            e.dispatch_thread_groups(
                objc2_metal::MTLSize {
                    width: 1,
                    height: rows,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: 32 * 32,
                    height: 1,
                    depth: 1,
                },
            );
        }
        Ok((
            MetalStorage::new(output, device.clone(), rows * head_dim, DType::F32),
            Shape::from((batch, q_heads, 1, head_dim)),
        ))
    }
}

fn prefill_pipeline(
    device: &candle_core::MetalDevice,
    name: &'static str,
) -> Result<ComputePipeline> {
    use objc2_metal::{MTLCompileOptions, MTLLanguageVersion};
    static PREFILL: OnceLock<Mutex<HashMap<(DeviceId, &'static str), ComputePipeline>>> =
        OnceLock::new();
    let cache = PREFILL.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(pipeline) = cache.lock().unwrap().get(&(device.id(), name)) {
        return Ok(pipeline.clone());
    }
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version4_0);
    let library = device
        .metal_device()
        .new_library_with_source(SDPA_F16KV_PREFILL_MSL, Some(&options))
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function(name, None)
        .map_err(candle_core::Error::wrap)?;
    let compiled = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    let mut cache = cache.lock().unwrap();
    Ok(cache.entry((device.id(), name)).or_insert(compiled).clone())
}

fn supports_tensor_prefill(device: &candle_core::MetalDevice) -> bool {
    use objc2_metal::{MTLDevice, MTLGPUFamily};
    device
        .metal_device()
        .as_ref()
        .supportsFamily(MTLGPUFamily::Metal4)
}

/// Causal attention for a prompt chunk of `q_len` rows over an F16 cache of
/// `kv_len >= q_len` rows (the chunk's own keys last). Shapes as in
/// [`try_decode`]; returns F32 `[batch, heads, q_len, D]` (a transposed view
/// of token-major storage, so merging heads needs no copy), or `None` when the
/// device or geometry is unsupported.
pub(crate) fn try_prefill(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
) -> Result<Option<Tensor>> {
    if !q.device().is_metal()
        || !q.device().same_device(k.device())
        || !q.device().same_device(v.device())
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
        || !matches!(q.dtype(), DType::F32 | DType::F16)
        || [q, k, v].iter().any(|t| t.rank() != 4)
        || !scale.is_finite()
        || scale <= 0.0
        || !supports_tensor_prefill(q.device().as_metal_device()?)
    {
        return Ok(None);
    }
    let (batch, q_heads, q_len, head_dim) = q.dims4()?;
    let (k_batch, kv_heads, kv_len, k_dim) = k.dims4()?;
    if q_len == 0
        || kv_len < q_len
        || k.dims4()? != v.dims4()?
        || k_batch != batch
        || k_dim != head_dim
        || kv_heads == 0
        || !q_heads.is_multiple_of(kv_heads)
        || !matches!(head_dim, 64 | 128)
    {
        return Ok(None);
    }
    for t in [k, v] {
        let stride = t.stride();
        if stride[3] != 1 || stride[2] != head_dim {
            return Ok(None);
        }
    }
    // Q is staged as half (the rounding the widened F32 path also applies).
    let q = q.to_dtype(DType::F16)?.contiguous()?;
    let output = q.apply_op3_no_bwd(
        k,
        v,
        &PrefillSdpa {
            scale,
            k_strides: [k.stride()[0], k.stride()[1]],
            v_strides: [v.stride()[0], v.stride()[1]],
        },
    )?;
    // Stored token-major; return the [batch, heads, q_len, D] view.
    Ok(Some(output.transpose(1, 2)?))
}

struct PrefillSdpa {
    scale: f32,
    k_strides: [usize; 2],
    v_strides: [usize; 2],
}

impl CustomOp3 for PrefillSdpa {
    fn name(&self) -> &'static str {
        "dinf-sdpa-f16kv-prefill"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("F16-cache prefill attention is only dispatched on Metal")
    }

    fn metal_fwd(
        &self,
        q: &MetalStorage,
        q_layout: &Layout,
        k: &MetalStorage,
        k_layout: &Layout,
        v: &MetalStorage,
        v_layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (batch, q_heads, q_len, head_dim) = q_layout.shape().dims4()?;
        let (_, kv_heads, kv_len, _) = k_layout.shape().dims4()?;
        if q.dtype() != DType::F16
            || k.dtype() != DType::F16
            || v.dtype() != DType::F16
            || !q_layout.is_contiguous()
            || k_layout.stride()[..2] != self.k_strides
            || v_layout.stride()[..2] != self.v_strides
            || k_layout.stride()[2..] != [head_dim, 1]
            || v_layout.stride()[2..] != [head_dim, 1]
            || q.device().id() != k.device().id()
            || q.device().id() != v.device().id()
        {
            candle_core::bail!("F16-cache prefill attention tensor metadata changed")
        }
        let name = match head_dim {
            64 => "dinf_sdpa_f16kv_prefill_64",
            128 => "dinf_sdpa_f16kv_prefill_128",
            _ => candle_core::bail!("F16-cache prefill attention head dimension"),
        };
        let to_u32 = |value: usize| {
            u32::try_from(value).map_err(|_| {
                candle_core::Error::Msg("prefill attention geometry exceeds u32".into())
            })
        };
        let elements = batch * q_heads * q_len * head_dim;
        let device = q.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-sdpa-f16kv-prefill")
            .build()?;
        let pipeline = prefill_pipeline(device, name)?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-sdpa-f16kv-prefill");
        encoder.set_compute_pipeline_state(&pipeline);
        let e: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        let f16 = DType::F16.size_in_bytes();
        e.set_input_buffer(0, Some(q.buffer()), q_layout.start_offset() * f16);
        e.set_input_buffer(1, Some(k.buffer()), k_layout.start_offset() * f16);
        e.set_input_buffer(2, Some(v.buffer()), v_layout.start_offset() * f16);
        e.set_output_buffer(3, Some(&output), 0);
        e.set_bytes(4, &(to_u32(q_heads / kv_heads)? as i32));
        e.set_bytes(5, &(to_u32(q_len)? as i32));
        e.set_bytes(6, &(to_u32(kv_len)? as i32));
        e.set_bytes(7, &to_u32(self.k_strides[1])?);
        e.set_bytes(8, &to_u32(self.v_strides[1])?);
        e.set_bytes(9, &to_u32(self.k_strides[0])?);
        e.set_bytes(10, &to_u32(self.v_strides[0])?);
        e.set_bytes(11, &(to_u32(q_heads)? as i32));
        e.set_bytes(12, &self.scale);
        // Scores in float, P in half, the P V tile in float (32 query rows).
        let shared = 32 * 32 * 4 + 32 * 32 * 4 + 32 * head_dim * 4;
        e.set_threadgroup_memory_length(0, shared);
        e.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: q_len.div_ceil(32),
                height: q_heads,
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
            Shape::from((batch, q_len, q_heads, head_dim)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn decode_matches_widened_f32_sdpa_with_cache_capacity() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for (batch, heads, kv_heads, keys, dim) in [
            (1, 32, 4, 1, 128),
            (1, 32, 4, 700, 128),
            (2, 8, 2, 1500, 64),
            (1, 16, 16, 3000, 256),
        ] {
            let capacity = keys + 9;
            let values = |len: usize, salt: f32| {
                (0..len)
                    .map(|i| (i as f32 * 0.37 + salt).sin() * 2.0)
                    .collect::<Vec<f32>>()
            };
            let q = Tensor::from_vec(
                values(batch * heads * dim, 0.1),
                (batch, heads, 1, dim),
                &device,
            )?;
            let kc = Tensor::from_vec(
                values(batch * kv_heads * capacity * dim, 0.5),
                (batch, kv_heads, capacity, dim),
                &Device::Cpu,
            )?
            .to_dtype(DType::F16)?
            .to_device(&device)?;
            let vc = Tensor::from_vec(
                values(batch * kv_heads * capacity * dim, 0.9),
                (batch, kv_heads, capacity, dim),
                &Device::Cpu,
            )?
            .to_dtype(DType::F16)?
            .to_device(&device)?;
            let k = kc.narrow(2, 0, keys)?;
            let v = vc.narrow(2, 0, keys)?;
            let scale = 1.0 / (dim as f32).sqrt();
            let actual = try_decode(&q, &k, &v, scale)?.expect("supported decode geometry");
            let expected = candle_nn::ops::sdpa(
                &q.to_dtype(DType::F16)?.to_dtype(DType::F32)?,
                &k.to_dtype(DType::F32)?,
                &v.to_dtype(DType::F32)?,
                None,
                false,
                scale,
                1.0,
            )?;
            let diff = (actual - expected)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(
                diff <= 2e-5,
                "batch={batch} heads={heads} keys={keys} dim={dim}: max abs {diff}"
            );
        }
        Ok(())
    }

    #[test]
    fn prefill_matches_masked_f32_sdpa_for_cached_chunks() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        // (batch, heads, kv_heads, q_len, kv_len, dim): partial tiles, whole
        // prompts and chunks that start partway through the cache.
        for (batch, heads, kv_heads, q_len, kv_len, dim) in [
            (1, 32, 4, 1, 1, 128),
            (1, 32, 4, 17, 17, 128),
            (1, 32, 4, 128, 700, 128),
            (2, 8, 2, 45, 99, 64),
            (1, 16, 16, 512, 1024, 128),
        ] {
            let capacity = kv_len + 5;
            let values = |len: usize, salt: f32| {
                (0..len)
                    .map(|i| (i as f32 * 0.37 + salt).sin() * 2.0)
                    .collect::<Vec<f32>>()
            };
            let q = Tensor::from_vec(
                values(batch * heads * q_len * dim, 0.1),
                (batch, heads, q_len, dim),
                &device,
            )?;
            let cache = |salt| -> anyhow::Result<Tensor> {
                Ok(Tensor::from_vec(
                    values(batch * kv_heads * capacity * dim, salt),
                    (batch, kv_heads, capacity, dim),
                    &Device::Cpu,
                )?
                .to_dtype(DType::F16)?
                .to_device(&device)?
                .narrow(2, 0, kv_len)?)
            };
            let (k, v) = (cache(0.5)?, cache(0.9)?);
            let scale = 1.0 / (dim as f32).sqrt();
            let actual = try_prefill(&q, &k, &v, scale)?.expect("supported prefill geometry");
            let offset = kv_len - q_len;
            let mask = (0..q_len * kv_len)
                .map(|i| {
                    if i % kv_len <= offset + i / kv_len {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
                .collect::<Vec<f32>>();
            let mask = Tensor::from_vec(mask, (1, 1, q_len, kv_len), &device)?
                .broadcast_as((batch, heads, q_len, kv_len))?;
            let n_rep = heads / kv_heads;
            let widen = |t: &Tensor| -> anyhow::Result<Tensor> {
                let t = t.to_dtype(DType::F32)?;
                let (b, h, l, d) = t.dims4()?;
                Ok(t.unsqueeze(2)?
                    .expand((b, h, n_rep, l, d))?
                    .reshape((b, h * n_rep, l, d))?)
            };
            let qr = q.to_dtype(DType::F16)?.to_dtype(DType::F32)?;
            let scores = (qr.matmul(&widen(&k)?.t()?)? * f64::from(scale))?.broadcast_add(&mask)?;
            let expected = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&widen(&v)?)?;
            let diff = (actual - &expected)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            let peak = expected.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            assert!(
                diff <= 2e-3 * peak.max(1.0),
                "batch={batch} heads={heads} q={q_len} kv={kv_len} dim={dim}: max abs {diff}"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "manual Metal timing"]
    fn bench_prefill_attention() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for kv_len in [512, 2048, 8192] {
            let (heads, kv_heads, q_len, dim) = (32, 4, 512, 128);
            let q = Tensor::randn(0f32, 1.0, (1, heads, q_len, dim), &device)?;
            let k = Tensor::randn(0f32, 1.0, (1, kv_heads, kv_len, dim), &device)?
                .to_dtype(DType::F16)?;
            let v = Tensor::randn(0f32, 1.0, (1, kv_heads, kv_len, dim), &device)?
                .to_dtype(DType::F16)?;
            for _ in 0..3 {
                try_prefill(&q, &k, &v, 0.088)?;
            }
            device.synchronize()?;
            let start = std::time::Instant::now();
            for _ in 0..20 {
                try_prefill(&q, &k, &v, 0.088)?;
            }
            device.synchronize()?;
            let ms = start.elapsed().as_secs_f64() * 1e3 / 20.0;
            let visible = (kv_len - q_len) as f64 + q_len as f64 / 2.0;
            let flops = 4.0 * q_len as f64 * visible * dim as f64 * heads as f64;
            println!(
                "prefill attention q={q_len} kv={kv_len}: {ms:.3} ms, {:.1} TFLOPS",
                flops / ms / 1e9
            );
        }
        Ok(())
    }
}
