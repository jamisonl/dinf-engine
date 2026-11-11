//! Q2_K Metal projection with a bounded eight-row scalar dispatch.
//!
//! Candle's Q2 shader writes four rows per SIMD group and uses two SIMD
//! groups. Its scalar host dispatch currently launches twice that many groups.
//! This module keeps the shader and valid output arithmetic unchanged.

use candle_core::backend::BackendStorage;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{CpuStorage, CustomOp2, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::source::Source;
use candle_metal_kernels::{Output, set_params};
use objc2_metal::{MTLComputePipelineState, MTLSize};

const BLOCK: usize = 256;
const BLOCK_BYTES: usize = 84;
const ROWS_PER_GROUP: usize = 8;
const MAX_PACKED_BYTES: usize = 2 * 1024 * 1024 * 1024;
const MAX_IO_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Debug)]
pub(crate) struct MetalQ2 {
    weights: Tensor,
    op: Projection,
}

#[derive(Clone, Debug)]
struct Projection {
    input_width: usize,
    output_width: usize,
    padded_output_width: usize,
    packed_bytes: usize,
}

fn checked_mul(a: usize, b: usize, what: &str) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| candle_core::Error::Msg(format!("Q2 Metal {what} overflow")))
}

fn geometry(n: usize, k: usize) -> Result<(usize, usize, usize)> {
    if n == 0 || k == 0 || !k.is_multiple_of(BLOCK) {
        candle_core::bail!("Q2 Metal weights require nonzero rows and 256-aligned columns")
    }
    let padded_n = n
        .checked_add(ROWS_PER_GROUP - 1)
        .map(|n| n / ROWS_PER_GROUP * ROWS_PER_GROUP)
        .ok_or_else(|| candle_core::Error::Msg("Q2 Metal padded row overflow".into()))?;
    let row_bytes = checked_mul(k / BLOCK, BLOCK_BYTES, "weight row")?;
    let original = checked_mul(n, row_bytes, "weight bytes")?;
    let padded = checked_mul(padded_n, row_bytes, "padded weight bytes")?;
    if padded > MAX_PACKED_BYTES || k > i32::MAX as usize || padded_n > i32::MAX as usize {
        candle_core::bail!("Q2 Metal packed weights exceed supported bounds")
    }
    Ok((padded_n, original, padded))
}

impl MetalQ2 {
    /// The caller may drop the Candle QTensor after construction. This object
    /// retains only the new packed bytes, including at most seven zero rows.
    pub(super) fn from_qtensor(qt: &QTensor) -> Result<Self> {
        if qt.dtype() != GgmlDType::Q2K || !qt.device().is_metal() {
            candle_core::bail!("Q2 Metal constructor requires a Metal Q2_K tensor")
        }
        let (n, k) = qt.shape().dims2()?;
        let (padded_n, original, padded) = geometry(n, k)?;
        if qt.storage_size_in_bytes() != original {
            candle_core::bail!("Q2 Metal packed storage extent differs from shape")
        }
        let bytes = qt.data()?;
        if bytes.len() != original {
            candle_core::bail!("Q2 Metal packed readback extent differs from shape")
        }
        let bytes = if padded == original {
            bytes.into_owned()
        } else {
            // Keep the constructor bound explicit: the original readback and
            // one exact padded Vec overlap, with no geometric Vec growth.
            let mut padded_bytes = vec![0; padded];
            padded_bytes[..original].copy_from_slice(&bytes);
            drop(bytes);
            padded_bytes
        };
        let weights = Tensor::from_vec(bytes, padded, &qt.device())?;
        Ok(Self {
            weights,
            op: Projection {
                input_width: k,
                output_width: n,
                padded_output_width: padded_n,
                packed_bytes: padded,
            },
        })
    }

    #[cfg(test)]
    pub(crate) fn packed_size_bytes(&self) -> usize {
        self.op.packed_bytes
    }

    #[cfg(test)]
    pub(crate) fn original_size_bytes(&self) -> usize {
        self.op.output_width * (self.op.input_width / BLOCK) * BLOCK_BYTES
    }

    #[cfg(test)]
    pub(crate) fn padding_size_bytes(&self) -> usize {
        self.packed_size_bytes() - self.original_size_bytes()
    }

    pub(super) fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.dtype() != DType::F32 || !input.device().same_device(self.weights.device()) {
            candle_core::bail!("Q2 Metal input must be F32 on the weight device")
        }
        self.op.input_shape(input.dims())?;
        // Validate the original view before making its bounded contiguous copy.
        let (storage, layout) = input.storage_and_layout();
        let candle_core::Storage::Metal(storage) = &*storage else {
            candle_core::bail!("Q2 Metal input storage is not Metal")
        };
        validate_span(layout, 4, storage.buffer().length())?;
        let input = input.contiguous()?;
        let output = input.apply_op2_no_bwd(&self.weights, &self.op)?;
        if output.dims().last() == Some(&self.op.padded_output_width)
            && self.op.output_width != self.op.padded_output_width
        {
            output
                .narrow(output.rank() - 1, 0, self.op.output_width)?
                .contiguous()
        } else {
            Ok(output)
        }
    }
}

fn validate_span(layout: &Layout, bytes: usize, storage_bytes: usize) -> Result<()> {
    let last = layout
        .dims()
        .iter()
        .zip(layout.stride())
        .try_fold(layout.start_offset(), |span, (&dim, &stride)| {
            if dim == 0 {
                return None;
            }
            (dim - 1).checked_mul(stride)?.checked_add(span)
        })
        .and_then(|span| span.checked_add(1))
        .and_then(|span| span.checked_mul(bytes));
    if last.is_none_or(|span| span > storage_bytes) {
        candle_core::bail!("Q2 Metal tensor view exceeds storage")
    }
    Ok(())
}

impl Projection {
    fn input_shape(&self, dims: &[usize]) -> Result<(usize, bool)> {
        if !(2..=3).contains(&dims.len()) || dims.last() != Some(&self.input_width) {
            candle_core::bail!("Q2 Metal input requires rank 2/3 and matching width")
        }
        let rows = dims[..dims.len() - 1]
            .iter()
            .try_fold(1usize, |rows, &dim| {
                if dim == 0 {
                    return Err(candle_core::Error::Msg("Q2 Metal empty input".into()));
                }
                checked_mul(rows, dim, "input rows")
            })?;
        let scalar = dims[dims.len() - 2] == 1;
        let out_width = if scalar {
            self.padded_output_width
        } else {
            self.output_width
        };
        let input_bytes = checked_mul(
            checked_mul(rows, self.input_width, "input elements")?,
            4,
            "input bytes",
        )?;
        let output_bytes = checked_mul(
            checked_mul(rows, out_width, "output elements")?,
            4,
            "output bytes",
        )?;
        if input_bytes > MAX_IO_BYTES || output_bytes > MAX_IO_BYTES {
            candle_core::bail!("Q2 Metal input/output byte cap exceeded")
        }
        Ok((rows, scalar))
    }
}

impl CustomOp2 for Projection {
    fn name(&self) -> &'static str {
        "dinf-bounded-q2-metal"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("Q2 Metal projection cannot run on CPU")
    }
    fn metal_fwd(
        &self,
        input: &MetalStorage,
        il: &Layout,
        weights: &MetalStorage,
        wl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if input.dtype() != DType::F32
            || weights.dtype() != DType::U8
            || !il.is_contiguous()
            || !wl.is_contiguous()
            || input.device().id() != weights.device().id()
            || wl.shape().elem_count() != self.packed_bytes
            || wl.start_offset() != 0
        {
            candle_core::bail!("Q2 Metal storage dtype/device/layout mismatch")
        }
        let (rows, scalar) = self.input_shape(il.dims())?;
        validate_span(il, 4, input.buffer().length())?;
        validate_span(wl, 1, weights.buffer().length())?;
        let d = input.device();
        let output_width = if scalar {
            self.padded_output_width
        } else {
            self.output_width
        };
        let elements = checked_mul(rows, output_width, "output elements")?;
        let output = d
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-q2-projection")
            .build()?;
        let encoder = d.command_encoder()?;
        if scalar {
            let p = d
                .kernels()
                .load_pipeline(d.device(), Source::Quantized, "kernel_mul_mv_q2_K_f32")
                .map_err(candle_core::Error::wrap)?;
            if p.as_ref().threadExecutionWidth() != 32 || p.max_total_threads_per_threadgroup() < 64
            {
                candle_core::bail!("Q2 Metal shader requires 32 lanes and 64 threads per group")
            }
            let e = encoder.as_ref();
            e.set_compute_pipeline_state(&p);
            for row in 0..rows {
                let k = self.input_width as i64;
                let n = self.padded_output_width as i64;
                let one = 1_i64;
                let zero = 0_i64;
                let one_u = 1_u32;
                set_params!(
                    e,
                    (
                        (weights.buffer(), wl.start_offset()),
                        (
                            input.buffer(),
                            (il.start_offset() + row * self.input_width) * 4
                        ),
                        Output::with_offset(&output, row * output_width * 4),
                        k,
                        n,
                        one,
                        zero,
                        zero,
                        zero,
                        k,
                        one,
                        one,
                        zero,
                        zero,
                        zero,
                        n,
                        one,
                        one_u,
                        one_u
                    )
                );
                e.dispatch_thread_groups(
                    MTLSize {
                        width: self.padded_output_width / ROWS_PER_GROUP,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 2,
                        height: 32,
                        depth: 1,
                    },
                );
            }
        } else {
            // The packed row pitch and original logical dimensions match Candle.
            let k = self.input_width;
            let n = self.output_width;
            let row_bytes = k / BLOCK * BLOCK_BYTES;
            let shape0 = [1, 1, n, k];
            let stride0 = [n * row_bytes, n * row_bytes, row_bytes, 0];
            let shape1 = if il.dims().len() == 2 {
                [1, 1, il.dims()[0], k]
            } else {
                [1, il.dims()[0], il.dims()[1], k]
            };
            let stride1 = [rows * k * 4, shape1[2] * k * 4, k * 4, 4];
            let mut shape = il.dims().to_vec();
            *shape.last_mut().unwrap() = n;
            candle_metal_kernels::call_quantized_matmul_mm_t(
                d.device(),
                &encoder,
                d.kernels(),
                candle_metal_kernels::GgmlDType::Q2K,
                &shape0,
                &stride0,
                weights.buffer(),
                &shape1,
                &stride1,
                input.buffer(),
                il.start_offset() * 4,
                &shape,
                0,
                &output,
            )
            .map_err(candle_core::Error::wrap)?;
        }
        let mut shape = il.dims().to_vec();
        *shape.last_mut().unwrap() = output_width;
        Ok((
            MetalStorage::new(output, d.clone(), elements, DType::F32),
            Shape::from(shape),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_rejects_empty_misaligned_and_overflowing_weights() {
        assert_eq!(geometry(1, 256).unwrap(), (8, 84, 672));
        assert_eq!(geometry(8, 256).unwrap(), (8, 672, 672));
        assert_eq!(geometry(9, 256).unwrap(), (16, 756, 1344));
        for (n, k) in [
            (0, 256),
            (8, 0),
            (8, 255),
            (usize::MAX, 256),
            (8, usize::MAX),
            (usize::MAX / 2, 256),
        ] {
            assert!(geometry(n, k).is_err());
        }
        assert!(geometry(8, MAX_PACKED_BYTES).is_err());
    }

    #[test]
    fn input_caps_and_spans_reject_invalid_views_before_dispatch() {
        let op = Projection {
            input_width: 256,
            output_width: 9,
            padded_output_width: 16,
            packed_bytes: 1344,
        };
        assert_eq!(op.input_shape(&[2, 1, 256]).unwrap(), (2, true));
        assert_eq!(op.input_shape(&[2, 3, 256]).unwrap(), (6, false));
        for dims in [
            vec![256],
            vec![1, 255],
            vec![0, 256],
            vec![1, 1, 1, 256],
            vec![usize::MAX, 256],
            vec![MAX_IO_BYTES / 4, 256],
        ] {
            assert!(op.input_shape(&dims).is_err());
        }
        let good = Layout::new(Shape::from((2, 3)), vec![8, 2], 1);
        assert!(validate_span(&good, 4, 14 * 4).is_ok());
        assert!(validate_span(&good, 4, 13 * 4).is_err());
        let bad = Layout::new(Shape::from((2, 3)), vec![usize::MAX, 2], 1);
        assert!(validate_span(&bad, 4, usize::MAX).is_err());
        assert!(
            validate_span(
                &Layout::contiguous_with_offset((1, 256), usize::MAX),
                4,
                usize::MAX
            )
            .is_err()
        );
    }
}
