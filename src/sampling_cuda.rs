//! Greedy argmax on CUDA with a small readback.
//!
//! The portable greedy path checks the logits for non-finite values, takes
//! the argmax and re-reads the selected logit: several launches and three
//! synchronous readbacks per token. Here one launch reduces the vocabulary
//! to a best candidate per block and the host merges those few candidates.

use std::sync::OnceLock;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use candle_core::{CpuStorage, CudaStorage, DType, Layout, Shape, Tensor};

const CUDA_MODULE_NAME: &str = "dinf_greedy_argmax_v1";
const CUDA_KERNEL_NAME: &str = "dinf_greedy_argmax_partial_f32";
const BLOCK_THREADS: usize = 256;
/// Logits reduced by one block.
const CHUNK: usize = 4096;
/// `[value bits, index, invalid]` per block.
const PARTIAL_WIDTH: usize = 3;
const NO_INDEX: u32 = u32::MAX;

// Each block reduces one contiguous chunk. The lowest index wins ties, and a
// NaN or infinite logit anywhere in the chunk sets its invalid flag.
const CUDA_SOURCE: &str = r#"
extern "C" __global__ void dinf_greedy_argmax_partial_f32(
    const float *logits,
    unsigned int *partial,
    long long vocab,
    long long chunk) {
    __shared__ float values[256];
    __shared__ unsigned int indices[256];
    __shared__ unsigned int flags[256];
    const unsigned int thread = threadIdx.x;
    const long long base = (long long) blockIdx.x * chunk;
    long long end = base + chunk;
    if (end > vocab) {
        end = vocab;
    }
    float best = 0.0f;
    unsigned int best_index = 0xffffffffu;
    unsigned int invalid = 0;
    for (long long index = base + thread; index < end; index += blockDim.x) {
        const float value = logits[index];
        if (!(value - value == 0.0f)) {
            invalid = 1;
        } else if (best_index == 0xffffffffu || value > best) {
            best = value;
            best_index = (unsigned int) index;
        }
    }
    values[thread] = best;
    indices[thread] = best_index;
    flags[thread] = invalid;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread < stride) {
            const unsigned int other = thread + stride;
            const unsigned int other_index = indices[other];
            const unsigned int index = indices[thread];
            if (other_index != 0xffffffffu &&
                (index == 0xffffffffu || values[other] > values[thread] ||
                 (values[other] == values[thread] && other_index < index))) {
                values[thread] = values[other];
                indices[thread] = other_index;
            }
            flags[thread] |= flags[other];
        }
        __syncthreads();
    }
    if (thread == 0) {
        unsigned int *out = partial + (long long) blockIdx.x * 3;
        out[0] = __float_as_uint(values[0]);
        out[1] = indices[0];
        out[2] = flags[0];
    }
}
"#;

static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();

fn ptx() -> candle_core::Result<&'static str> {
    match PTX.get_or_init(|| {
        compile_ptx_with_opts(
            CUDA_SOURCE,
            CompileOptions {
                name: Some("greedy_argmax.cu".to_string()),
                use_fast_math: Some(false),
                ..Default::default()
            },
        )
        .map(|ptx| ptx.to_src())
        .map_err(|error| error.to_string())
    }) {
        Ok(ptx) => Ok(ptx.as_str()),
        Err(error) => candle_core::bail!("failed to compile CUDA greedy argmax: {error}"),
    }
}

struct PartialArgmax {
    vocab: usize,
}

impl candle_core::CustomOp1 for PartialArgmax {
    fn name(&self) -> &'static str {
        "cuda-greedy-argmax-partial"
    }

    fn cpu_fwd(
        &self,
        _storage: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuda-greedy-argmax-partial is only dispatched on CUDA")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let slice = storage.as_cuda_slice::<f32>()?;
        let start = layout.start_offset();
        if !layout.is_contiguous()
            || layout.shape().elem_count() != self.vocab
            || start + self.vocab > slice.len()
        {
            candle_core::bail!("CUDA greedy argmax input geometry differs")
        }
        let input = slice.slice(start..start + self.vocab);
        let device = storage.device();
        let blocks = self.vocab.div_ceil(CHUNK);
        // SAFETY: Every block writes its three output words.
        let mut output = unsafe { device.alloc::<u32>(blocks * PARTIAL_WIDTH)? };
        let function =
            device.get_or_load_custom_func(CUDA_KERNEL_NAME, CUDA_MODULE_NAME, ptx()?)?;
        let (vocab, chunk) = (self.vocab as i64, CHUNK as i64);
        let mut builder = function.builder();
        builder.arg(&input).arg(&mut output).arg(&vocab).arg(&chunk);
        // SAFETY: The input view was bounds-checked above and the grid covers
        // exactly `blocks` output triples.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks as u32, 1, 1),
                block_dim: (BLOCK_THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from((blocks, PARTIAL_WIDTH)),
        ))
    }
}

/// The lowest index holding the largest logit of a contiguous F32 CUDA
/// tensor. Fails when any logit is NaN or infinite. Returns `None` for other
/// devices, dtypes and layouts so the caller keeps its portable path.
pub(crate) fn try_greedy_argmax(logits: &Tensor) -> anyhow::Result<Option<u32>> {
    let vocab = logits.elem_count();
    if !logits.device().is_cuda()
        || logits.dtype() != DType::F32
        || !logits.is_contiguous()
        || vocab == 0
        || vocab >= NO_INDEX as usize
    {
        return Ok(None);
    }
    let partial = logits
        .apply_op1_no_bwd(&PartialArgmax { vocab })?
        .to_vec2::<u32>()?;
    let mut best: Option<(f32, u32)> = None;
    for block in &partial {
        anyhow::ensure!(
            block.len() == PARTIAL_WIDTH,
            "fused greedy argmax result is malformed"
        );
        anyhow::ensure!(block[2] == 0, "logits must be finite");
        let (value, index) = (f32::from_bits(block[0]), block[1]);
        if index == NO_INDEX {
            continue;
        }
        anyhow::ensure!(
            (index as usize) < vocab,
            "fused greedy argmax returned an invalid token"
        );
        // Blocks are in index order, so a strict comparison keeps the lowest
        // index among equal logits.
        if best.is_none_or(|(best_value, _)| value > best_value) {
            best = Some((value, index));
        }
    }
    let (_, token) =
        best.ok_or_else(|| anyhow::anyhow!("fused greedy argmax found no candidate"))?;
    Ok(Some(token))
}

#[cfg(test)]
mod tests {
    use candle_core::Tensor;

    use super::{CHUNK, try_greedy_argmax};

    #[test]
    fn matches_the_host_argmax_and_rejects_non_finite_logits() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        for vocab in [
            1,
            7,
            255,
            257,
            CHUNK - 1,
            CHUNK,
            CHUNK + 1,
            3 * CHUNK + 911,
            262_144,
        ] {
            let mut values = (0..vocab)
                .map(|index| ((index * 7919 % 10_007) as f32 - 5_000.0) / 311.0)
                .collect::<Vec<_>>();
            let expected = crate::sampling::checked_first_argmax(&values)?;
            let logits = Tensor::from_vec(values.clone(), vocab, &device)?;
            assert_eq!(try_greedy_argmax(&logits)?, Some(expected), "vocab {vocab}");

            // Equal maxima: the lowest index wins, across threads and blocks.
            let top = 1.0e6;
            for index in [vocab - 1, vocab / 2, vocab / 3] {
                values[index] = top;
            }
            let expected = crate::sampling::checked_first_argmax(&values)?;
            assert_eq!(expected as usize, vocab / 3);
            let logits = Tensor::from_vec(values.clone(), vocab, &device)?;
            assert_eq!(try_greedy_argmax(&logits)?, Some(expected), "ties {vocab}");

            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut invalid = values.clone();
                invalid[vocab / 2] = bad;
                let logits = Tensor::from_vec(invalid, vocab, &device)?;
                let error = try_greedy_argmax(&logits).unwrap_err();
                assert!(error.to_string().contains("must be finite"), "{error}");
            }
        }
        // All logits negative, and other devices are declined.
        let logits = Tensor::from_vec(vec![-3.0f32, -1.0, -2.0], 3, &device)?;
        assert_eq!(try_greedy_argmax(&logits)?, Some(1));
        let cpu = Tensor::from_vec(vec![0.0f32, 1.0], 2, &candle_core::Device::Cpu)?;
        assert_eq!(try_greedy_argmax(&cpu)?, None);
        Ok(())
    }
}
