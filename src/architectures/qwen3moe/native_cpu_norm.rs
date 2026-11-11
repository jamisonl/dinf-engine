//! Native CPU BF16 norm: F32 products, F64 sum, then the native F32 scale and multiplies.
use anyhow::{Context, Result};
use candle_core::{DType, Tensor};
pub(super) fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    anyhow::ensure!(
        x.device().is_cpu()
            && weight.device().is_cpu()
            && x.dtype() == DType::F32
            && weight.dtype() == DType::F32,
        "Qwen CPU RMSNorm requires F32 CPU input and weight"
    );
    let width = *x
        .dims()
        .last()
        .context("Qwen CPU RMSNorm requires a row dimension")?;
    anyhow::ensure!(
        width > 0 && weight.dims() == [width],
        "Qwen CPU RMSNorm weight must match the nonempty last input dimension"
    );
    anyhow::ensure!(
        eps.is_finite() && eps >= 0.0 && (eps as f32).is_finite(),
        "Qwen CPU RMSNorm epsilon must be finite and nonnegative in F32"
    );
    if x.elem_count() == 0 {
        return Tensor::zeros(x.shape(), DType::F32, x.device()).map_err(Into::into);
    }
    // Contiguous tensors share storage; only strided views need a copy.
    let x = x.contiguous()?;
    let weight = weight.contiguous()?;
    let (xs, xl) = x.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let (candle_core::Storage::Cpu(xs), candle_core::Storage::Cpu(ws)) = (&*xs, &*ws) else {
        anyhow::bail!("Qwen CPU RMSNorm storage must be on CPU");
    };
    let (x_start, x_end) = xl
        .contiguous_offsets()
        .context("Qwen CPU RMSNorm input layout")?;
    let (w_start, w_end) = wl
        .contiguous_offsets()
        .context("Qwen CPU RMSNorm weight layout")?;
    let values = xs
        .as_slice::<f32>()?
        .get(x_start..x_end)
        .context("Qwen CPU RMSNorm input span")?;
    let weights = ws
        .as_slice::<f32>()?
        .get(w_start..w_end)
        .context("Qwen CPU RMSNorm weight span")?;
    let mut output = Vec::with_capacity(values.len());
    for row in values.chunks_exact(width) {
        let sum = row
            .iter()
            .map(|&value| f64::from(value * value))
            .sum::<f64>();
        let mean = (sum / width as f64) as f32;
        let scale = 1.0_f32 / (mean + eps as f32).sqrt();
        output.extend(
            row.iter()
                .zip(weights)
                .map(|(&value, &weight)| (value * scale) * weight),
        );
    }
    Tensor::from_vec(output, x.shape(), x.device()).map_err(Into::into)
}

#[cfg(test)]
#[path = "native_cpu_norm_tests.rs"]
mod tests;
