use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{CpuStorage, CustomOp1, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;

const GREEDY_ARGMAX_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Greedy argmax over finite logits: lowest index wins ties, and any
// non-finite logit marks the result invalid. Stage one reduces one contiguous
// chunk per threadgroup; stage two merges the per-chunk winners. A single
// threadgroup scanning the whole vocabulary leaves all but one GPU core idle.
inline bool dinf_argmax_better(float value, uint index, float best, uint best_index) {
  return value > best || (value == best && index < best_index);
}

inline void dinf_argmax_simd(thread float &value, thread uint &index, thread uint &invalid) {
  for (ushort offset = 16; offset > 0; offset >>= 1) {
    const float other = simd_shuffle_down(value, offset);
    const uint other_index = simd_shuffle_down(index, offset);
    if (dinf_argmax_better(other, other_index, value, index)) {
      value = other;
      index = other_index;
    }
  }
  invalid = simd_max(invalid);
}

inline void dinf_argmax_threadgroup(
    thread float &value, thread uint &index, thread uint &invalid,
    threadgroup float *maxima, threadgroup uint *indices, threadgroup uint *flags,
    uint simd_lane, uint simd_id, uint simd_groups) {
  dinf_argmax_simd(value, index, invalid);
  if (simd_lane == 0) {
    maxima[simd_id] = value;
    indices[simd_id] = index;
    flags[simd_id] = invalid;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_id == 0) {
    value = simd_lane < simd_groups ? maxima[simd_lane] : -INFINITY;
    index = simd_lane < simd_groups ? indices[simd_lane] : 0xffffffffu;
    invalid = simd_lane < simd_groups ? flags[simd_lane] : 0;
    dinf_argmax_simd(value, index, invalid);
  }
}

kernel void dinf_greedy_argmax_partial_f32(
    device const float *logits [[buffer(0)]],
    device uint *partial [[buffer(1)]],
    constant uint &vocab [[buffer(2)]],
    constant uint &chunk [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]]) {
  threadgroup float maxima[32];
  threadgroup uint indices[32];
  threadgroup uint flags[32];

  const uint begin = group * chunk;
  const uint end = min(vocab, begin + chunk);
  float value = -INFINITY;
  uint index = 0xffffffffu;
  uint invalid = 0;
  for (uint token = begin + tid; token < end; token += threads) {
    const float logit = logits[token];
    if (!isfinite(logit)) {
      invalid = 1;
      continue;
    }
    if (dinf_argmax_better(logit, token, value, index)) {
      value = logit;
      index = token;
    }
  }
  dinf_argmax_threadgroup(value, index, invalid, maxima, indices, flags,
                          simd_lane, simd_id, simd_groups);
  if (tid == 0) {
    partial[group * 3] = as_type<uint>(value);
    partial[group * 3 + 1] = index;
    partial[group * 3 + 2] = invalid;
  }
}

kernel void dinf_greedy_argmax_merge_f32(
    device const uint *partial [[buffer(0)]],
    device uint *output [[buffer(1)]],
    constant uint &groups [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]]) {
  threadgroup float maxima[32];
  threadgroup uint indices[32];
  threadgroup uint flags[32];

  float value = -INFINITY;
  uint index = 0xffffffffu;
  uint invalid = 0;
  for (uint group = tid; group < groups; group += threads) {
    const float candidate = as_type<float>(partial[group * 3]);
    const uint candidate_index = partial[group * 3 + 1];
    if (dinf_argmax_better(candidate, candidate_index, value, index)) {
      value = candidate;
      index = candidate_index;
    }
    invalid |= partial[group * 3 + 2];
  }
  dinf_argmax_threadgroup(value, index, invalid, maxima, indices, flags,
                          simd_lane, simd_id, simd_groups);
  if (tid == 0) {
    output[0] = index;
    output[1] = invalid;
  }
}
"#;

static PIPELINES: OnceLock<Mutex<HashMap<DeviceId, (ComputePipeline, ComputePipeline)>>> =
    OnceLock::new();

/// The `(partial, merge)` pipelines for the two-stage argmax.
fn pipelines(device: &candle_core::MetalDevice) -> Result<(ComputePipeline, ComputePipeline)> {
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(pair) = pipelines.lock().unwrap().get(&device.id()) {
        return Ok(pair.clone());
    }
    let library = device
        .metal_device()
        .new_library_with_source(GREEDY_ARGMAX_MSL, None)
        .map_err(candle_core::Error::wrap)?;
    let build = |name: &str| -> Result<ComputePipeline> {
        let function = library
            .get_function(name, None)
            .map_err(candle_core::Error::wrap)?;
        device
            .metal_device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(candle_core::Error::wrap)
    };
    let pair = (
        build("dinf_greedy_argmax_partial_f32")?,
        build("dinf_greedy_argmax_merge_f32")?,
    );
    pipelines.lock().unwrap().insert(device.id(), pair.clone());
    Ok(pair)
}

/// Threadgroups for the first argmax stage. Each covers a contiguous chunk of
/// at least `MIN_ARGMAX_CHUNK` logits, so small vocabularies use one group.
const MAX_ARGMAX_GROUPS: usize = 64;
const MIN_ARGMAX_CHUNK: usize = 2048;

fn argmax_groups(vocab: usize) -> usize {
    vocab.div_ceil(MIN_ARGMAX_CHUNK).clamp(1, MAX_ARGMAX_GROUPS)
}

struct GreedyArgmaxFinite;

pub(crate) fn precompile(device: &candle_core::Device) -> anyhow::Result<()> {
    if device.is_metal() && std::env::var_os("DINF_DISABLE_FUSED_METAL_GREEDY").is_none() {
        pipelines(device.as_metal_device()?)?;
    }
    Ok(())
}

impl CustomOp1 for GreedyArgmaxFinite {
    fn name(&self) -> &'static str {
        "dinf-metal-greedy-argmax-finite"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused greedy argmax is only dispatched on Metal")
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        if storage.dtype() != DType::F32 || !layout.is_contiguous() {
            candle_core::bail!("fused greedy argmax requires contiguous f32 logits")
        }
        let vocab = layout.shape().elem_count();
        // Keep the strided MSL loop's final `token += threads` below uint
        // overflow even at the largest 256-thread launch.
        if vocab == 0 || vocab > u32::MAX as usize - 255 {
            candle_core::bail!("fused greedy argmax vocabulary size is unsupported")
        }
        let device = storage.device();
        let input_offset = layout
            .start_offset()
            .checked_mul(DType::F32.size_in_bytes())
            .ok_or_else(|| {
                candle_core::Error::Msg("fused greedy argmax input offset overflows usize".into())
            })?;
        let input_bytes = vocab
            .checked_mul(DType::F32.size_in_bytes())
            .and_then(|bytes| input_offset.checked_add(bytes))
            .ok_or_else(|| {
                candle_core::Error::Msg("fused greedy argmax input span overflows usize".into())
            })?;
        if input_bytes > storage.buffer().length() {
            candle_core::bail!("fused greedy argmax input span exceeds its Metal buffer")
        }
        let output = device
            .new_buffer_builder()
            .with_size_for(2, DType::U32)
            .with_label("dinf-greedy-argmax-finite")
            .build()?;
        let groups = argmax_groups(vocab);
        let chunk = vocab.div_ceil(groups);
        let partial = device
            .new_buffer_builder()
            .with_size_for(groups * 3, DType::U32)
            .with_label("dinf-greedy-argmax-partial")
            .build()?;
        let (partial_pipeline, merge_pipeline) = pipelines(device)?;
        let encoder = device.command_encoder()?;
        encoder.set_label("dinf-greedy-argmax-finite");
        let encoder_ref: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
        let threads = |pipeline: &ComputePipeline| {
            (1usize << pipeline.max_total_threads_per_threadgroup().max(1).ilog2()).min(256)
        };
        let size = |width| objc2_metal::MTLSize {
            width,
            height: 1,
            depth: 1,
        };
        encoder_ref.set_compute_pipeline_state(&partial_pipeline);
        encoder_ref.set_input_buffer(0, Some(storage.buffer()), input_offset);
        encoder_ref.set_output_buffer(1, Some(&partial), 0);
        encoder_ref.set_bytes(2, &(vocab as u32));
        encoder_ref.set_bytes(3, &(chunk as u32));
        encoder_ref.dispatch_thread_groups(size(groups), size(threads(&partial_pipeline)));
        encoder_ref.set_compute_pipeline_state(&merge_pipeline);
        encoder_ref.set_input_buffer(0, Some(&partial), 0);
        encoder_ref.set_output_buffer(1, Some(&output), 0);
        encoder_ref.set_bytes(2, &(groups as u32));
        encoder_ref.dispatch_thread_groups(size(1), size(threads(&merge_pipeline).min(64)));
        Ok((
            MetalStorage::new(output, device.clone(), 2, DType::U32),
            Shape::from(2),
        ))
    }
}

fn greedy_argmax_deferred(logits: &Tensor) -> anyhow::Result<Tensor> {
    logits
        .apply_op1_no_bwd(&GreedyArgmaxFinite)
        .map_err(Into::into)
}

fn validate_greedy_result(result: &[u32], vocab: usize) -> anyhow::Result<u32> {
    anyhow::ensure!(result.len() == 2, "fused greedy argmax result is malformed");
    anyhow::ensure!(result[1] == 0, "logits must be finite");
    anyhow::ensure!(
        (result[0] as usize) < vocab,
        "fused greedy argmax returned an invalid token"
    );
    Ok(result[0])
}

pub(crate) fn greedy_argmax(logits: &Tensor) -> anyhow::Result<u32> {
    let result = greedy_argmax_deferred(logits)?.to_vec1::<u32>()?;
    validate_greedy_result(&result, logits.elem_count())
}

/// Enqueue one validated argmax reduction per logits view, then read back only
/// the tiny `[token, invalid]` results together. Unsupported inputs return
/// `None` before any command is encoded so callers can retain their scalar
/// sampling path without partially executing a batch.
pub(crate) fn try_greedy_argmax_batch(logits: &[Tensor]) -> anyhow::Result<Option<Vec<u32>>> {
    if logits.len() < 2
        || std::env::var_os("DINF_DISABLE_FUSED_METAL_GREEDY").is_some()
        || std::env::var_os("DINF_DISABLE_BATCHED_METAL_GREEDY").is_some()
    {
        return Ok(None);
    }
    let device = logits[0].device();
    if !device.is_metal()
        || logits.iter().any(|logits| {
            logits.dtype() != DType::F32
                || !logits.is_contiguous()
                || logits.elem_count() == 0
                || logits.elem_count() > u32::MAX as usize - 255
                || !device.same_device(logits.device())
        })
    {
        return Ok(None);
    }

    let vocabularies = logits.iter().map(Tensor::elem_count).collect::<Vec<_>>();
    let deferred = logits
        .iter()
        .map(greedy_argmax_deferred)
        .collect::<anyhow::Result<Vec<_>>>()?;
    let deferred_refs = deferred.iter().collect::<Vec<_>>();
    // Stack only the two-u32 reduction results, never the full vocabularies.
    // This produces one synchronization/readback for the complete batch.
    let results = Tensor::stack(&deferred_refs, 0)?.to_vec2::<u32>()?;
    anyhow::ensure!(
        results.len() == logits.len(),
        "fused greedy argmax batch result has the wrong lane count"
    );
    let tokens = results
        .iter()
        .zip(vocabularies)
        .map(|(result, vocab)| validate_greedy_result(result, vocab))
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(Some(tokens))
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Tensor};

    use super::{greedy_argmax, try_greedy_argmax_batch};

    #[test]
    fn fused_greedy_argmax_matches_large_vocabulary_reference() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let vocab = 248_320usize;
        let mut values = (0..vocab)
            .map(|index| ((index * 104_729 % 1_000_003) as f32 - 500_000.0) / 8192.0)
            .collect::<Vec<_>>();
        values[17] = 1000.0;
        values[200_001] = 1000.0;
        let logits = Tensor::from_vec(values, vocab, &device)?.to_dtype(DType::F32)?;
        assert_eq!(greedy_argmax(&logits)?, 17);
        Ok(())
    }

    #[test]
    fn fused_greedy_argmax_merges_chunks_in_index_order() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let groups = super::argmax_groups(248_320);
        assert!(groups > 1, "the vocabulary must span several chunks");
        let chunk = 248_320usize.div_ceil(groups);
        for (vocab, winners, expected) in [
            // The winner sits in the final, short chunk.
            (248_320usize, vec![248_319usize], 248_319u32),
            // Equal maxima in different chunks: the lowest index wins.
            (
                248_320,
                vec![chunk * 5 + 3, chunk * 2 + 7, chunk * 40],
                (chunk * 2 + 7) as u32,
            ),
            // Chunk boundaries on both sides.
            (248_320, vec![chunk, chunk - 1], (chunk - 1) as u32),
            // Vocabularies smaller than one chunk use a single group.
            (1_000, vec![999, 400], 400),
            (1, vec![0], 0),
        ] {
            let mut values = (0..vocab)
                .map(|index| -((index % 997) as f32) / 64.0)
                .collect::<Vec<_>>();
            for &winner in &winners {
                values[winner] = 50.0;
            }
            let logits = Tensor::from_vec(values, vocab, &device)?;
            assert_eq!(
                greedy_argmax(&logits)?,
                expected,
                "vocab {vocab} winners {winners:?}"
            );
        }

        let mut values = vec![0.0f32; 248_320];
        values[3] = 9.0;
        values[247_000] = f32::NAN;
        let logits = Tensor::from_vec(values, 248_320, &device)?;
        assert!(
            greedy_argmax(&logits).is_err(),
            "a late non-finite logit must be reported"
        );
        Ok(())
    }

    #[test]
    fn fused_greedy_argmax_rejects_nonfinite_logits() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let logits = Tensor::from_vec(vec![1.0f32, invalid, 2.0], 3, &device)?;
            assert!(greedy_argmax(&logits).is_err());
        }
        Ok(())
    }

    #[test]
    fn fused_greedy_argmax_handles_small_odd_negative_offset_view() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let storage = Tensor::from_vec(
            vec![99.0f32, -8.0, -3.0, -5.0, -3.0, -7.0, 101.0],
            7,
            &device,
        )?;
        let view = storage.narrow(0, 1, 5)?;
        anyhow::ensure!(view.is_contiguous() && view.layout().start_offset() > 0);
        assert_eq!(greedy_argmax(&view)?, 1);

        let singleton = Tensor::from_vec(vec![-42.0f32], 1, &device)?;
        assert_eq!(greedy_argmax(&singleton)?, 0);
        Ok(())
    }

    #[test]
    fn batched_greedy_matches_lane_order_ties_offsets_and_odd_vocabularies() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let (batch, vocab, prefix, suffix) = (4usize, 1_009usize, 7usize, 11usize);
        let mut values = vec![-10_000.0f32; prefix + batch * vocab + suffix];
        let expected = [17u32, 1_008, 3, 511];
        for lane in 0..batch {
            let start = prefix + lane * vocab;
            for token in 0..vocab {
                values[start + token] = -((token + lane * 13) as f32);
            }
            values[start + expected[lane] as usize] = 1_000.0;
        }
        // A later equal maximum must not displace the lowest token ID.
        values[prefix + 900] = 1_000.0;
        let backing = Tensor::from_vec(values, prefix + batch * vocab + suffix, &device)?;
        let lanes = (0..batch)
            .map(|lane| backing.narrow(0, prefix + lane * vocab, vocab))
            .collect::<candle_core::Result<Vec<_>>>()?;
        assert!(
            lanes
                .iter()
                .all(|lane| lane.is_contiguous() && lane.layout().start_offset() > 0)
        );
        assert_eq!(
            try_greedy_argmax_batch(&lanes)?.expect("Metal batch should be supported"),
            expected
        );
        Ok(())
    }

    #[test]
    fn batched_greedy_rejects_a_nonfinite_late_lane_atomically() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let first = Tensor::from_vec(vec![1.0f32, 3.0, 2.0], 3, &device)?;
            let second = Tensor::from_vec(vec![4.0f32, invalid, 5.0], 3, &device)?;
            let error = try_greedy_argmax_batch(&[first, second])
                .and_then(|result| result.ok_or_else(|| anyhow::anyhow!("unexpected fallback")))
                .unwrap_err()
                .to_string();
            assert!(error.contains("logits must be finite"), "{error}");
        }
        Ok(())
    }

    #[test]
    fn batched_greedy_falls_back_before_encoding_mixed_inputs() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let metal = Tensor::from_vec(vec![1.0f32, 2.0], 2, &device)?;
        let cpu = Tensor::from_vec(vec![3.0f32, 4.0], 2, &candle_core::Device::Cpu)?;
        assert!(try_greedy_argmax_batch(std::slice::from_ref(&metal))?.is_none());
        assert!(try_greedy_argmax_batch(&[metal.clone(), cpu])?.is_none());
        assert!(try_greedy_argmax_batch(&[metal.clone(), metal.to_dtype(DType::F16)?])?.is_none());
        Ok(())
    }

    #[test]
    #[ignore = "manual Metal ABBA microbenchmark"]
    fn benchmark_batched_greedy_batch2_and_batch4() -> anyhow::Result<()> {
        use std::hint::black_box;
        use std::time::{Duration, Instant};

        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for batch in [2usize, 4] {
            let vocab = 151_936usize;
            let values = (0..batch * vocab)
                .map(|index| ((index * 104_729 % 1_000_003) as f32 - 500_000.0) / 8192.0)
                .collect::<Vec<_>>();
            let backing = Tensor::from_vec(values, (batch, vocab), &device)?;
            let lanes = (0..batch)
                .map(|lane| backing.narrow(0, lane, 1)?.squeeze(0))
                .collect::<candle_core::Result<Vec<_>>>()?;
            let expected = lanes
                .iter()
                .map(greedy_argmax)
                .collect::<anyhow::Result<Vec<_>>>()?;
            assert_eq!(
                try_greedy_argmax_batch(&lanes)?.expect("Metal batch should be supported"),
                expected
            );

            let iterations = 256usize;
            let measure = |batched: bool| -> anyhow::Result<Duration> {
                device.synchronize()?;
                let started = Instant::now();
                for _ in 0..iterations {
                    let tokens = if batched {
                        try_greedy_argmax_batch(&lanes)?.unwrap()
                    } else {
                        lanes
                            .iter()
                            .map(greedy_argmax)
                            .collect::<anyhow::Result<Vec<_>>>()?
                    };
                    black_box(tokens);
                }
                device.synchronize()?;
                Ok(started.elapsed())
            };

            measure(false)?;
            measure(true)?;
            let mut scalar = Vec::with_capacity(7);
            let mut batched = Vec::with_capacity(7);
            for round in 0usize..7 {
                if round.is_multiple_of(2) {
                    batched.push(measure(true)?);
                    scalar.push(measure(false)?);
                } else {
                    scalar.push(measure(false)?);
                    batched.push(measure(true)?);
                }
            }
            scalar.sort_unstable();
            batched.sort_unstable();
            let scalar = scalar[scalar.len() / 2];
            let batched = batched[batched.len() / 2];
            eprintln!(
                "BATCHED_GREEDY_BENCH batch={batch} iterations={iterations} scalar_ms={:.3} batched_ms={:.3} speedup={:.3}x",
                scalar.as_secs_f64() * 1_000.0,
                batched.as_secs_f64() * 1_000.0,
                scalar.as_secs_f64() / batched.as_secs_f64(),
            );
        }
        Ok(())
    }
}
