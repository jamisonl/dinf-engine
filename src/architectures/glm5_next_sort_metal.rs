//! Wide pooled-index sorting without a thread for each column.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::BufferOffset;
use candle_metal_kernels::source::Source;

pub(super) fn descending_indices(scores: &Tensor) -> Result<Option<Tensor>> {
    if !scores.device().is_metal() {
        return Ok(None);
    }
    let columns = scores.dim(candle_core::D::Minus1)?;
    let device = scores.device().as_metal_device()?;
    let pipeline = device
        .kernels()
        .load_pipeline(device.metal_device(), Source::Sort, "asort_desc_f32")
        .map_err(candle_core::Error::wrap)?;
    if columns.next_power_of_two() <= pipeline.max_total_threads_per_threadgroup() {
        return Ok(None);
    }
    // MLX's bounded-block sort is ascending. Negating scores gives descending
    // indices, including the -infinity entries that mask future pools.
    Ok(Some(
        scores.neg()?.contiguous()?.apply_op1_no_bwd(&WideSort)?,
    ))
}

struct WideSort;

impl CustomOp1 for WideSort {
    fn name(&self) -> &'static str {
        "dinf-glm5-wide-pool-sort"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("wide pooled-index sort requires Metal")
    }

    fn metal_fwd(&self, input: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let (rows, columns) = layout.shape().dims2()?;
        if rows == 0
            || columns == 0
            || rows > i32::MAX as usize
            || columns > i32::MAX as usize
            || input.dtype() != DType::F32
            || !layout.is_contiguous()
        {
            candle_core::bail!("wide pool sort requires a nonempty contiguous F32 matrix")
        }
        let count = rows
            .checked_mul(columns)
            .ok_or_else(|| candle_core::Error::Msg("pool sort size overflow".into()))?;
        let device = input.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(count, DType::U32)
            .with_label("glm5-pool-indices")
            .build()?;
        candle_metal_kernels::call_mlx_arg_sort(
            device.metal_device(),
            &device.command_encoder()?,
            device.kernels(),
            candle_metal_kernels::DType::F32,
            rows,
            columns,
            BufferOffset {
                buffer: input.buffer(),
                offset_in_bytes: layout.start_offset() * 4,
            },
            &output,
        )
        .map_err(candle_core::Error::wrap)?;
        Ok((
            MetalStorage::new(output, device.clone(), count, DType::U32),
            layout.shape().clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn wide_pool_sort_matches_host_at_thread_and_merge_boundaries() -> anyhow::Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        for columns in [512, 1024, 1025, 2048, 4097, 16385, 262144] {
            let rows = if columns < 262144 { 3 } else { 1 };
            let values = (0..rows * columns)
                .map(|i| {
                    if i % columns >= columns - 7 {
                        f32::NEG_INFINITY
                    } else {
                        ((i * 37 + 11) % columns) as f32
                    }
                })
                .collect::<Vec<_>>();
            // Preserve a nonzero offset as well as contiguous row strides.
            let scores = Tensor::cat(
                &[
                    &Tensor::zeros((1, columns), DType::F32, &device)?,
                    &Tensor::from_vec(values.clone(), (rows, columns), &device)?,
                ],
                0,
            )?
            .narrow(0, 1, rows)?;
            let sorted = match descending_indices(&scores)? {
                Some(indices) => indices,
                None => scores.arg_sort_last_dim(false)?,
            }
            .to_vec2::<u32>()?;
            for (row, indices) in sorted.iter().enumerate() {
                assert_eq!(indices.len(), columns);
                assert!(indices.iter().all(|&i| (i as usize) < columns));
                assert_eq!(indices.iter().collect::<HashSet<_>>().len(), columns);
                let actual = indices
                    .iter()
                    .map(|&i| values[row * columns + i as usize])
                    .collect::<Vec<_>>();
                let mut expected = values[row * columns..(row + 1) * columns].to_vec();
                expected.sort_by(|a, b| b.total_cmp(a));
                assert_eq!(actual, expected, "pool width {columns}, row {row}");
            }
        }
        Ok(())
    }
}
