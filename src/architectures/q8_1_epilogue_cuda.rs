//! Q8_1 quantization as the last step of a CUDA kernel.
//!
//! The quantized matrix-vector kernels read their activation as Q8_1
//! blocks, and each product normally launches a kernel that quantizes it.
//! A kernel that produces an activation for one decode token can write the
//! blocks itself with [`SOURCE`] and publish them with [`publish`]; products
//! of that activation then skip their quantization launch (see
//! `q8_1_memo_cuda`).
//!
//! The device function performs the stock quantizer's operations in its
//! order, so the blocks are the bytes that kernel would write.

use candle_core::Tensor;

/// Device code to place before a kernel source that calls
/// `dinf_q8_1_store`.
pub(super) const SOURCE: &str = r#"
typedef struct {
    unsigned short ds[2];
    signed char qs[32];
} dinf_q8_1_block;

static __device__ __forceinline__ unsigned short dinf_q8_1_half(float value) {
    unsigned short result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(value));
    return result;
}

// Quantize value `index` of a row into `quantized`. All 32 lanes of a warp
// must call this together, lane `l` with index `base + l` for one `base`
// that is a multiple of 32. Values past the row's end are passed as zero.
static __device__ __forceinline__ void dinf_q8_1_store(
    void *quantized,
    int index,
    float value) {
    float amax = fabsf(value);
    float sum = value;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, mask);
    }
    const float scale = amax / 127.0f;
    const signed char level = amax == 0.0f ? 0 : (signed char) roundf(value / scale);
    dinf_q8_1_block *block = (dinf_q8_1_block *) quantized + index / 32;
    block->qs[index % 32] = level;
    if (index % 32 == 0) {
        block->ds[0] = dinf_q8_1_half(scale);
        block->ds[1] = dinf_q8_1_half(sum);
    }
}

// Values of a row padded to a multiple of 512, the quantizer's row length.
static __device__ __forceinline__ int dinf_q8_1_padded(int values) {
    return (values + 511) / 512 * 512;
}
"#;

/// F32 elements that hold the Q8_1 form of one row of `len` values.
pub(super) fn tail_elements(len: usize) -> usize {
    super::q8_1_memo_cuda::quantized_bytes(len) / std::mem::size_of::<f32>()
}

/// Publish the `len` values at element `activation_offset` of `packed`'s
/// storage with their Q8_1 form at element `quantized_offset`.
pub(super) fn publish(
    packed: &Tensor,
    activation_offset: usize,
    len: usize,
    quantized_offset: usize,
) -> anyhow::Result<()> {
    super::q8_1_memo_cuda::publish(packed, activation_offset, len, quantized_offset)
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{DType, Device, Tensor};

    use crate::architectures::dense_qmatmul_cuda::forward_one_to_f32;
    use crate::architectures::gemma4_norm_fusion_cuda::{
        MoePlanes, MoeTail, try_attention_planes, try_post_ffn,
    };
    use crate::architectures::q8_1_memo_cuda as q8_1_memo;
    use crate::qmatmul::QMatMul;

    fn values(elements: usize, multiplier: usize) -> Vec<f32> {
        (0..elements)
            .map(|index| ((index * multiplier % 1009) as f32 - 504.0) / 211.0)
            .collect()
    }

    fn row(device: &Device, hidden: usize, multiplier: usize) -> candle_core::Result<Tensor> {
        Tensor::from_vec(values(hidden, multiplier), (1, 1, hidden), device)
    }

    fn weight(device: &Device, hidden: usize, multiplier: usize) -> candle_core::Result<Tensor> {
        Tensor::from_vec(values(hidden, multiplier), hidden, device)
    }

    fn bits(tensor: &Tensor) -> anyhow::Result<Vec<u32>> {
        Ok(tensor
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }

    /// A product of `activation` uses its published Q8_1 form when
    /// `published`, and either way equals the product of a copy, which the
    /// stock kernel quantizes.
    fn assert_product_matches(activation: &Tensor, published: bool) -> anyhow::Result<()> {
        let device = activation.device();
        let input_dim = activation.elem_count();
        let source = Tensor::from_vec(values(48 * input_dim, 7), (48, input_dim), &Device::Cpu)?;
        for dtype in [GgmlDType::Q8_0, GgmlDType::Q4K] {
            let projection =
                QMatMul::from_arc(Arc::new(QTensor::quantize_onto(&source, dtype, device)?))?;
            let hits = q8_1_memo::hits();
            let actual = forward_one_to_f32(activation, &projection)?.expect("eligible product");
            assert_eq!(q8_1_memo::hits() - hits, u64::from(published));
            let copy = Tensor::from_vec(
                activation.flatten_all()?.to_vec1::<f32>()?,
                activation.shape(),
                device,
            )?;
            let hits = q8_1_memo::hits();
            let expected = forward_one_to_f32(&copy, &projection)?.expect("eligible product");
            assert_eq!(q8_1_memo::hits(), hits);
            assert_eq!(
                bits(&actual)?,
                bits(&expected)?,
                "{dtype:?}, {input_dim} values"
            );
        }
        Ok(())
    }

    /// The Gemma 4 norm kernels publish the next layer's input, the FFN
    /// input and the expert input of one token. Widths cover rows that end
    /// on, before and past a 1024-thread pass of the padded row.
    #[test]
    fn norm_kernels_publish_the_stock_quantization() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        for hidden in [1024usize, 1280, 2816, 3072, 8192] {
            let (planes, tail) = (row(&device, hidden, 3)?, row(&device, hidden, 5)?);
            let weights = (11..17)
                .map(|multiplier| weight(&device, hidden, multiplier))
                .collect::<candle_core::Result<Vec<_>>>()?;
            let (attn_out, ffn_in, experts) = try_attention_planes(
                &planes,
                &tail,
                (&weights[0], 1e-6),
                (&weights[1], 1e-6),
                Some(MoePlanes {
                    pre_weight: &weights[2],
                    eps: 1e-6,
                }),
            )?
            .expect("supported geometry");
            let (expert_in, router_in) = experts.expect("requested");
            assert_product_matches(&ffn_in, true)?;
            assert_product_matches(&expert_in, true)?;
            assert_product_matches(&attn_out, false)?;
            assert_product_matches(&router_in, false)?;

            let (block_out, next) = try_post_ffn(
                &ffn_in,
                &attn_out,
                (&weights[3], 1e-6),
                Some(MoeTail {
                    routed: &expert_in,
                    route_weights: None,
                    shared_weight: &weights[4],
                    routed_weight: &weights[5],
                    eps: 1e-6,
                }),
                None,
                Some((&weights[0], 1e-6)),
            )?
            .expect("supported geometry");
            assert_product_matches(&next.expect("requested"), true)?;
            assert_product_matches(&block_out, false)?;
        }
        // A prompt publishes nothing.
        let hidden = 1024;
        let rows = Tensor::from_vec(values(3 * hidden, 3), (1, 3, hidden), &device)?;
        let (weight_a, weight_b) = (weight(&device, hidden, 11)?, weight(&device, hidden, 13)?);
        let (_, ffn_in, _) =
            try_attention_planes(&rows, &rows, (&weight_a, 1e-6), (&weight_b, 1e-6), None)?
                .expect("supported geometry");
        assert_product_matches(&ffn_in.narrow(1, 0, 1)?, false)?;
        Ok(())
    }

    /// Decode attention publishes its whole output as the output
    /// projection's input, padded with zero blocks, when head rows are
    /// whole Q8_1 blocks.
    #[test]
    fn decode_attention_publishes_the_stock_quantization() -> anyhow::Result<()> {
        let Some(device) = crate::test_cuda_device() else {
            return Ok(());
        };
        for (heads, dim, published) in [(4usize, 64usize, true), (8, 96, true), (16, 48, false)] {
            let kv_len = 70;
            let q = Tensor::from_vec(values(heads * dim, 17), (1, heads, 1, dim), &device)?;
            let cache = |multiplier: usize| -> candle_core::Result<Tensor> {
                Tensor::from_vec(
                    values(heads * kv_len * dim, multiplier),
                    (1, heads, kv_len, dim),
                    &device,
                )?
                .to_dtype(DType::F16)
            };
            let (k, v) = (cache(23)?, cache(29)?);
            let output = crate::architectures::decode_sdpa_cuda::try_forward(&q, &k, &v, 0.125)?
                .expect("supported geometry");
            assert_product_matches(&output.reshape((1, 1, heads * dim))?, published)?;
            let gate = Tensor::from_vec(values(heads * dim, 31), (1, heads, 1, dim), &device)?;
            let gated = crate::architectures::decode_sdpa_cuda::try_forward_gated(
                &q,
                &k,
                &v,
                Some(&gate),
                0.125,
            )?
            .expect("supported geometry");
            assert_product_matches(&gated.reshape((1, 1, heads * dim))?, published)?;
        }
        Ok(())
    }
}
