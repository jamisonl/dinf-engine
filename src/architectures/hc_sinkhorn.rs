//! Shared four-stream normalization for V4 and GLM5.

use anyhow::Result;
use candle_core::{D, Tensor};

pub(super) fn enabled(configured: Option<&std::ffi::OsStr>, apple_silicon_metal: bool) -> bool {
    configured.map_or(apple_silicon_metal, |value| value == "1")
}

pub(super) fn normalize(
    matrix: Tensor,
    iterations: usize,
    eps: f64,
    fused: bool,
) -> Result<Tensor> {
    // Normalize the last axis first, then axis 2. Both models require this order.
    let matrix = candle_nn::ops::softmax_last_dim(&matrix)?;
    if fused && (1..=128).contains(&iterations) {
        #[cfg(target_os = "macos")]
        if matrix.device().is_metal() {
            return super::hc_sinkhorn_metal::forward(&matrix, iterations, eps);
        }
    }
    finish(matrix, iterations, eps)
}

pub(super) fn finish(mut matrix: Tensor, iterations: usize, eps: f64) -> Result<Tensor> {
    matrix = (matrix + eps)?;
    for iteration in 0..iterations {
        if iteration > 0 {
            let row_sum = (matrix.sum(D::Minus1)?.unsqueeze(D::Minus1)? + eps)?;
            matrix = matrix.broadcast_div(&row_sum)?;
        }
        let column_sum = (matrix.sum(2)?.unsqueeze(2)? + eps)?;
        matrix = matrix.broadcast_div(&column_sum)?;
    }
    Ok(matrix)
}
