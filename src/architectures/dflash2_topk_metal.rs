use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{CpuStorage, CustomOp1, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::MTLComputePipelineState;

const DFLASH_TOPK_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Strict order for every stage: higher score first, then lower token ID.
inline bool dinf_topk_before(float score, uint id, float other_score, uint other_id) {
  return score > other_score || (score == other_score && id < other_id);
}

inline void dinf_topk_insert(
    thread float *scores, thread uint *ids, uint top_k, float value, uint token) {
  uint insertion = top_k;
  for (uint route = 0; route < top_k; ++route) {
    if (dinf_topk_before(value, token, scores[route], ids[route])) {
      insertion = route;
      break;
    }
  }
  if (insertion < top_k) {
    for (uint route = top_k - 1; route > insertion; --route) {
      scores[route] = scores[route - 1];
      ids[route] = ids[route - 1];
    }
    scores[insertion] = value;
    ids[insertion] = token;
  }
}

// Merge each thread's sorted list across a 256-thread group; thread 0 returns
// the group's top-k in `selected_*`.
inline void dinf_topk_group_merge(
    thread float *local_scores,
    thread uint *local_ids,
    uint top_k,
    uint tid,
    uint lane,
    uint simdgroup,
    threadgroup float *group_scores,
    threadgroup uint *group_ids,
    thread float *selected_scores,
    thread uint *selected_ids) {
  // Advancing only the winning lane gives the next best item per SIMD group.
  uint local_route = 0;
  for (uint route = 0; route < top_k; ++route) {
    const float local_score = local_route < top_k ? local_scores[local_route] : -INFINITY;
    const uint local_id = local_route < top_k ? local_ids[local_route] : 0xffffffffu;
    const float simd_score = simd_max(local_score);
    const uint simd_id = simd_min(local_score == simd_score ? local_id : 0xffffffffu);
    if (local_score == simd_score && local_id == simd_id) {
      ++local_route;
    }
    if (lane == 0) {
      const uint index = simdgroup * top_k + route;
      group_scores[index] = simd_score;
      group_ids[index] = simd_id;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // The first SIMD group merges the eight sorted lists the same way: lane i
  // follows list i and only the winning lane advances.
  if (simdgroup == 0) {
    uint position = 0;
    for (uint route = 0; route < top_k; ++route) {
      const bool live = lane < 8 && position < top_k;
      const float score = live ? group_scores[lane * top_k + position] : -INFINITY;
      const uint id = live ? group_ids[lane * top_k + position] : 0xffffffffu;
      const float best_score = simd_max(score);
      const uint best_id = simd_min(score == best_score ? id : 0xffffffffu);
      if (live && score == best_score && id == best_id) {
        ++position;
      }
      if (lane == 0) {
        selected_scores[route] = best_score;
        selected_ids[route] = best_id;
      }
    }
  }
}

// Stage 1: many threadgroups per row, each over one vocabulary chunk, so the
// whole GPU reads the logits instead of one threadgroup per row.
kernel void dinf_dflash_vocab_topk_partial_f32(
    device const float *logits [[buffer(0)]],
    device float *partial_scores [[buffer(1)]],
    device uint *partial_ids [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &vocab [[buffer(4)]],
    constant uint &top_k [[buffer(5)]],
    constant uint &chunks [[buffer(6)]],
    device atomic_uint *invalid [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  const uint chunk = group.x;
  const uint row = group.y;
  if (row >= rows || chunk >= chunks) {
    return;
  }
  threadgroup float group_scores[8 * 16];
  threadgroup uint group_ids[8 * 16];
  const uint chunk_size = (vocab + chunks - 1) / chunks;
  const uint begin = chunk * chunk_size;
  const uint end = min(vocab, begin + chunk_size);
  const uint row_offset = row * vocab;

  float local_scores[16];
  uint local_ids[16];
  for (uint route = 0; route < top_k; ++route) {
    local_scores[route] = -INFINITY;
    local_ids[route] = 0xffffffffu;
  }
  for (uint token = begin + tid; token < end; token += 256) {
    const float value = logits[row_offset + token];
    if (!isfinite(value)) {
      atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
      continue;
    }
    if (dinf_topk_before(value, token, local_scores[top_k - 1], local_ids[top_k - 1])) {
      dinf_topk_insert(local_scores, local_ids, top_k, value, token);
    }
  }
  float selected_scores[16];
  uint selected_ids[16];
  dinf_topk_group_merge(local_scores, local_ids, top_k, tid, lane, simdgroup,
      group_scores, group_ids, selected_scores, selected_ids);
  if (tid == 0) {
    const uint base = (row * chunks + chunk) * top_k;
    for (uint route = 0; route < top_k; ++route) {
      partial_scores[base + route] = selected_scores[route];
      partial_ids[base + route] = selected_ids[route];
    }
  }
}

// Stage 2: one threadgroup per row merges the chunk lists into the packed
// [ids as f32 | scores] result.
kernel void dinf_dflash_vocab_topk_merge_f32(
    device const float *partial_scores [[buffer(0)]],
    device const uint *partial_ids [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &top_k [[buffer(4)]],
    constant uint &chunks [[buffer(5)]],
    device const uint *invalid [[buffer(6)]],
    constant uint &vocab [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  if (row >= rows) {
    return;
  }
  threadgroup float group_scores[8 * 16];
  threadgroup uint group_ids[8 * 16];
  float local_scores[16];
  uint local_ids[16];
  for (uint route = 0; route < top_k; ++route) {
    local_scores[route] = -INFINITY;
    local_ids[route] = 0xffffffffu;
  }
  const uint count = chunks * top_k;
  const uint base = row * count;
  for (uint index = tid; index < count; index += 256) {
    dinf_topk_insert(local_scores, local_ids, top_k,
        partial_scores[base + index], partial_ids[base + index]);
  }
  float selected_scores[16];
  uint selected_ids[16];
  dinf_topk_group_merge(local_scores, local_ids, top_k, tid, lane, simdgroup,
      group_scores, group_ids, selected_scores, selected_ids);
  if (tid == 0) {
    const uint row_routes = row * top_k;
    const uint plane = rows * top_k;
    // Non-finite logits mark every ID out of range so a caller that defers
    // the validity readback still fails closed when it reads the candidates.
    const bool poisoned = invalid[0] != 0;
    for (uint route = 0; route < top_k; ++route) {
      output[row_routes + route] = poisoned ? float(vocab) : float(selected_ids[route]);
      output[plane + row_routes + route] = selected_scores[route];
    }
  }
}

"#;

type PipelineKey = (DeviceId, &'static str);
static PIPELINES: OnceLock<Mutex<HashMap<PipelineKey, ComputePipeline>>> = OnceLock::new();
const PARTIAL_KERNEL: &str = "dinf_dflash_vocab_topk_partial_f32";
const MERGE_KERNEL: &str = "dinf_dflash_vocab_topk_merge_f32";
/// Vocabulary chunks per row for the first top-k stage.
const TOPK_CHUNKS: usize = 32;

fn pipeline(device: &candle_core::MetalDevice, name: &'static str) -> Result<ComputePipeline> {
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(pipeline) = pipelines.lock().unwrap().get(&(device.id(), name)) {
        return Ok(pipeline.clone());
    }
    let library = device
        .metal_device()
        .new_library_with_source(DFLASH_TOPK_MSL, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function(name, None)
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    pipelines
        .lock()
        .unwrap()
        .insert((device.id(), name), pipeline.clone());
    Ok(pipeline)
}

struct DFlashVocabTopK {
    top_k: usize,
    /// Skip the validity readback; invalid logits then yield IDs equal to
    /// the vocabulary size, which the caller must reject.
    defer_validity: bool,
}

fn checked_geometry(rows: usize, vocab: usize, top_k: usize) -> Option<(usize, usize, usize)> {
    // The packed result stores IDs as f32, so every vocabulary index must
    // survive the later f32-to-u32 conversion exactly.
    if rows == 0 || vocab == 0 || vocab - 1 > (1 << 24) || top_k == 0 || top_k > vocab || top_k > 16
    {
        return None;
    }
    let input_elements = rows.checked_mul(vocab)?;
    let route_elements = rows.checked_mul(top_k)?;
    let output_elements = route_elements.checked_mul(2)?;
    // MSL computes row offsets, both result planes, and their indices as uint.
    if input_elements > u32::MAX as usize || output_elements > u32::MAX as usize {
        return None;
    }
    Some((input_elements, route_elements, output_elements))
}

fn checked_input_offset(start: usize, elements: usize, buffer_bytes: usize) -> Result<usize> {
    let offset = start
        .checked_mul(DType::F32.size_in_bytes())
        .ok_or_else(|| {
            candle_core::Error::Msg("DFlash vocabulary top-k input offset overflows usize".into())
        })?;
    let end = elements
        .checked_mul(DType::F32.size_in_bytes())
        .and_then(|bytes| offset.checked_add(bytes))
        .ok_or_else(|| {
            candle_core::Error::Msg("DFlash vocabulary top-k input span overflows usize".into())
        })?;
    if end > buffer_bytes {
        candle_core::bail!("DFlash vocabulary top-k input span exceeds its Metal buffer")
    }
    Ok(offset)
}

fn launch_supported(simd_width: usize, max_threads: usize) -> bool {
    simd_width == 32 && max_threads >= 256
}

impl CustomOp1 for DFlashVocabTopK {
    fn name(&self) -> &'static str {
        "dinf-metal-dflash-vocab-topk"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("DFlash vocabulary top-k is only dispatched on Metal")
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        if storage.dtype() != DType::F32 || !layout.is_contiguous() {
            candle_core::bail!("DFlash vocabulary top-k requires contiguous f32 logits")
        }
        let (rows, vocab) = layout.shape().dims2()?;
        let Some((input_elements, _, output_elements)) = checked_geometry(rows, vocab, self.top_k)
        else {
            candle_core::bail!("DFlash vocabulary top-k geometry differs")
        };
        let input_offset = checked_input_offset(
            layout.start_offset(),
            input_elements,
            storage.buffer().length(),
        )?;
        let device = storage.device();
        let partial_pipeline = pipeline(device, PARTIAL_KERNEL)?;
        let merge_pipeline = pipeline(device, MERGE_KERNEL)?;
        for pipeline in [&partial_pipeline, &merge_pipeline] {
            if !launch_supported(
                pipeline.as_ref().threadExecutionWidth(),
                pipeline.max_total_threads_per_threadgroup(),
            ) {
                candle_core::bail!("DFlash vocabulary top-k requires eight 32-thread SIMD groups")
            }
        }
        let chunks = TOPK_CHUNKS.min(vocab.div_ceil(self.top_k));
        let partial_elements = rows * chunks * self.top_k;
        let partial_scores = device
            .new_buffer_builder()
            .with_size_for(partial_elements, DType::F32)
            .with_label("dinf-dflash-vocab-topk-partial-scores")
            .build()?;
        let partial_ids = device
            .new_buffer_builder()
            .with_size_for(partial_elements, DType::U32)
            .with_label("dinf-dflash-vocab-topk-partial-ids")
            .build()?;
        let output = device
            .new_buffer_builder()
            .with_size_for(output_elements, DType::F32)
            .with_label("dinf-dflash-vocab-topk")
            .build()?;
        let invalid_zero = [0u32];
        let invalid = device
            .new_buffer_builder()
            .with_data(&invalid_zero)
            .with_label("dinf-dflash-vocab-topk-invalid")
            .build()?;
        let threads = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        {
            let encoder = device.command_encoder()?;
            encoder.set_label("dinf-dflash-vocab-topk-partial");
            encoder.set_compute_pipeline_state(&partial_pipeline);
            let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
            encoder_ref.set_input_buffer(0, Some(storage.buffer()), input_offset);
            encoder_ref.set_output_buffer(1, Some(&partial_scores), 0);
            encoder_ref.set_output_buffer(2, Some(&partial_ids), 0);
            encoder_ref.set_bytes(3, &(rows as u32));
            encoder_ref.set_bytes(4, &(vocab as u32));
            encoder_ref.set_bytes(5, &(self.top_k as u32));
            encoder_ref.set_bytes(6, &(chunks as u32));
            encoder_ref.set_output_buffer(7, Some(&invalid), 0);
            encoder_ref.dispatch_thread_groups(
                objc2_metal::MTLSize {
                    width: chunks,
                    height: rows,
                    depth: 1,
                },
                threads,
            );
        }
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-dflash-vocab-topk-merge");
        encoder.set_compute_pipeline_state(&merge_pipeline);
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        encoder_ref.set_input_buffer(0, Some(&partial_scores), 0);
        encoder_ref.set_input_buffer(1, Some(&partial_ids), 0);
        encoder_ref.set_output_buffer(2, Some(&output), 0);
        encoder_ref.set_bytes(3, &(rows as u32));
        encoder_ref.set_bytes(4, &(self.top_k as u32));
        encoder_ref.set_bytes(5, &(chunks as u32));
        encoder_ref.set_input_buffer(6, Some(&invalid), 0);
        encoder_ref.set_bytes(7, &(vocab as u32));
        encoder_ref.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: rows,
                height: 1,
                depth: 1,
            },
            threads,
        );
        drop(encoder);
        if self.defer_validity {
            return Ok((
                MetalStorage::new(output, device.clone(), output_elements, DType::F32),
                Shape::from((2, rows, self.top_k)),
            ));
        }
        // The top-k candidates immediately feed Metal embedding lookups.  A
        // one-word validity readback lets us reject malformed logits before
        // those lookups without copying the vocabulary or candidate tensor to
        // the host on the normal path.
        let invalid = match MetalStorage::new(invalid, device.clone(), 1, DType::U32)
            .to_cpu_storage()?
        {
            CpuStorage::U32(values) => values.first().copied().ok_or_else(|| {
                candle_core::Error::Msg("DFlash vocabulary top-k validity buffer is empty".into())
            })?,
            _ => candle_core::bail!("DFlash vocabulary top-k validity buffer has wrong dtype"),
        };
        if invalid != 0 {
            candle_core::bail!("DFlash vocabulary logits must be finite")
        }
        Ok((
            MetalStorage::new(output, device.clone(), output_elements, DType::F32),
            Shape::from((2, rows, self.top_k)),
        ))
    }
}

fn top_k(logits: &Tensor, top_k: usize, defer_validity: bool) -> Result<(Tensor, Tensor)> {
    let packed = logits.apply_op1_no_bwd(&DFlashVocabTopK {
        top_k,
        defer_validity,
    })?;
    let ids = packed.narrow(0, 0, 1)?.squeeze(0)?.to_dtype(DType::U32)?;
    let values = packed.narrow(0, 1, 1)?.squeeze(0)?;
    Ok((ids, values))
}

/// Keep unsupported ID widths and MSL index ranges on the portable selector.
/// Fails closed on non-finite logits with its own validity readback; the
/// draft path uses [`try_top_k_deferred`] instead.
#[cfg(test)]
pub(super) fn try_top_k(logits: &Tensor, top_k: usize) -> Result<Option<(Tensor, Tensor)>> {
    try_top_k_with(logits, top_k, false)
}

/// Like [`try_top_k`] without the validity readback. Non-finite logits make
/// every returned ID equal the vocabulary size, which the caller must reject
/// when it reads the IDs.
pub(super) fn try_top_k_deferred(
    logits: &Tensor,
    top_k: usize,
) -> Result<Option<(Tensor, Tensor)>> {
    try_top_k_with(logits, top_k, true)
}

fn try_top_k_with(
    logits: &Tensor,
    top_k: usize,
    defer_validity: bool,
) -> Result<Option<(Tensor, Tensor)>> {
    let (rows, vocab) = logits.dims2()?;
    if checked_geometry(rows, vocab, top_k).is_none() {
        return Ok(None);
    }
    let device = logits.device().as_metal_device()?;
    for name in [PARTIAL_KERNEL, MERGE_KERNEL] {
        let pipeline = pipeline(device, name)?;
        if !launch_supported(
            pipeline.as_ref().threadExecutionWidth(),
            pipeline.max_total_threads_per_threadgroup(),
        ) {
            return Ok(None);
        }
    }
    self::top_k(logits, top_k, defer_validity).map(Some)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::{
        checked_geometry, checked_input_offset, launch_supported, try_top_k, try_top_k_deferred,
    };

    #[test]
    fn vocabulary_top_k_falls_back_before_rounding_large_token_ids() -> candle_core::Result<()> {
        let first_inexact_id = (1u32 << 24) + 1;
        assert_ne!(first_inexact_id as f32 as u32, first_inexact_id);
        let vocab = first_inexact_id as usize + 1;
        // A broadcast view proves the unsupported shape returns before a
        // device dispatch or an allocation proportional to the vocabulary.
        let logits = Tensor::zeros((1, 1), candle_core::DType::F32, &Device::Cpu)?
            .broadcast_as((1, vocab))?;
        assert!(try_top_k(&logits, 1)?.is_none());
        assert!(checked_geometry(1, (1 << 24) + 1, 1).is_some());
        Ok(())
    }

    #[test]
    fn vocabulary_top_k_rejects_wrapped_kernel_offsets() {
        assert!(checked_geometry(u32::MAX as usize / 2, 2, 1).is_some());
        assert!(checked_geometry(u32::MAX as usize / 2 + 1, 2, 1).is_none());
        assert!(checked_geometry(u32::MAX as usize / 32, 16, 16).is_some());
        assert!(checked_geometry(u32::MAX as usize / 32 + 1, 16, 16).is_none());
        assert!(checked_geometry(usize::MAX, 2, 1).is_none());
    }

    #[test]
    fn vocabulary_top_k_checks_offset_input_spans() {
        assert_eq!(checked_input_offset(2, 6, 32).unwrap(), 8);
        assert!(checked_input_offset(2, 6, 31).is_err());
        assert!(checked_input_offset(usize::MAX, 1, usize::MAX).is_err());
        assert!(checked_input_offset(0, usize::MAX, usize::MAX).is_err());
    }

    #[test]
    fn vocabulary_top_k_requires_eight_complete_simd_groups() {
        assert!(launch_supported(32, 256));
        assert!(launch_supported(32, 1024));
        for (width, threads) in [(16, 256), (64, 256), (32, 128)] {
            assert!(!launch_supported(width, threads));
        }
    }

    #[test]
    fn vocabulary_top_k_matches_cpu_reference() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let rows = 3;
        let vocab = 4099;
        let top = 16;
        let values = (0..rows * vocab)
            .map(|index| ((index * 104_729 % 1_000_003) as f32 - 500_000.0) / 8192.0)
            .collect::<Vec<_>>();
        let mut padded = vec![f32::NAN; vocab];
        padded.extend_from_slice(&values);
        let logits = Tensor::from_vec(padded, (rows + 1, vocab), &device)?.narrow(0, 1, rows)?;
        let (ids, scores) = try_top_k(&logits, top)?.expect("supported Metal top-k");
        let ids = ids.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
        let scores = scores.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
        for row in 0..rows {
            let mut expected = values[row * vocab..(row + 1) * vocab]
                .iter()
                .copied()
                .enumerate()
                .collect::<Vec<_>>();
            expected.sort_unstable_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            for route in 0..top {
                assert_eq!(ids[row][route], expected[route].0 as u32);
                assert_eq!(scores[row][route], expected[route].1);
            }
        }
        Ok(())
    }

    #[test]
    fn deferred_top_k_marks_nonfinite_rows_and_matches_on_finite_rows() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        // Wide enough for many chunks, with ties to exercise the ID order.
        let (rows, vocab, top) = (3, 40_000, 16);
        let values = (0..rows * vocab)
            .map(|index| ((index * 7919 % 10_007) as f32 / 97.0).floor())
            .collect::<Vec<_>>();
        let logits = Tensor::from_vec(values.clone(), (rows, vocab), &device)?;
        let (ids, scores) = try_top_k_deferred(&logits, top)?.expect("supported Metal top-k");
        let (checked_ids, checked_scores) =
            try_top_k(&logits, top)?.expect("supported Metal top-k");
        assert_eq!(ids.to_vec2::<u32>()?, checked_ids.to_vec2::<u32>()?);
        assert_eq!(scores.to_vec2::<f32>()?, checked_scores.to_vec2::<f32>()?);
        for (row, row_ids) in ids.to_vec2::<u32>()?.into_iter().enumerate() {
            let mut expected = (0..vocab).collect::<Vec<_>>();
            let row_values = &values[row * vocab..(row + 1) * vocab];
            expected.sort_by(|&a, &b| row_values[b].total_cmp(&row_values[a]).then(a.cmp(&b)));
            let expected = expected[..top]
                .iter()
                .map(|&id| id as u32)
                .collect::<Vec<_>>();
            assert_eq!(row_ids, expected, "row {row}");
        }

        let mut poisoned = values;
        poisoned[vocab + 5] = f32::NAN;
        let logits = Tensor::from_vec(poisoned, (rows, vocab), &device)?;
        let (ids, _) = try_top_k_deferred(&logits, top)?.expect("supported Metal top-k");
        assert!(
            ids.to_vec2::<u32>()?
                .iter()
                .flatten()
                .all(|&id| id as usize == vocab),
            "non-finite logits must mark every candidate out of range"
        );
        Ok(())
    }

    #[test]
    fn vocabulary_top_k_rejects_nonfinite_logits() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let rows = 2;
        let vocab = 8;
        let top = 4;
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut values = vec![0.0f32; rows * vocab];
            values[1] = invalid;
            values[vocab + 3] = invalid;
            let logits = Tensor::from_vec(values, (rows, vocab), &device)?;
            let error = try_top_k(&logits, top).expect_err("nonfinite logits must fail closed");
            assert!(error.to_string().contains("must be finite"));
        }

        let all_nan = Tensor::from_vec(vec![f32::NAN; vocab], (1, vocab), &device)?;
        let error = try_top_k(&all_nan, top).expect_err("all-NaN rows must fail closed");
        assert!(error.to_string().contains("must be finite"));

        // The invalid flag is per dispatch, so a failed request must not
        // poison a later finite request that reuses the cached pipeline.
        let finite = Tensor::from_vec(vec![1.0f32, 4.0, 3.0, 2.0], (1, 4), &device)?;
        let (ids, scores) = try_top_k(&finite, 2)?.expect("supported Metal top-k");
        assert_eq!(
            ids.to_device(&Device::Cpu)?.to_vec2::<u32>()?,
            vec![vec![1, 2]]
        );
        assert_eq!(
            scores.to_device(&Device::Cpu)?.to_vec2::<f32>()?,
            vec![vec![4.0, 3.0]]
        );
        Ok(())
    }

    #[test]
    fn vocabulary_top_k_rejects_rows_with_insufficient_finite_logits() -> candle_core::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let logits = Tensor::from_vec(
            vec![
                4.0f32,
                3.0,
                2.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            ],
            (1, 8),
            &device,
        )?;
        let error = try_top_k(&logits, 4).expect_err("insufficient finite logits must fail closed");
        assert!(error.to_string().contains("must be finite"));
        Ok(())
    }
}
