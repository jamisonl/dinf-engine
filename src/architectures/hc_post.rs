//! Shared four-stream residual update, with the original graph as fallback.

use anyhow::Result;
use candle_core::{D, Tensor};

pub(super) fn forward(
    update: &Tensor,
    residual: &Tensor,
    post: &Tensor,
    combination: &Tensor,
    fused: bool,
) -> Result<Tensor> {
    if fused {
        #[cfg(target_os = "macos")]
        if update.device().is_metal()
            && update.dtype() == candle_core::DType::F32
            && residual.dim(D::Minus1)? > 1
        {
            return super::hc_post_metal::forward(update, residual, post, combination);
        }
    }
    reference(update, residual, post, combination)
}

pub(super) fn reference(
    update: &Tensor,
    residual: &Tensor,
    post: &Tensor,
    combination: &Tensor,
) -> Result<Tensor> {
    let update = update
        .unsqueeze(2)?
        .broadcast_mul(&post.unsqueeze(D::Minus1)?)?;
    // GGUF stores combination axes as source, destination.
    let residual = combination
        .transpose(2, 3)?
        .contiguous()?
        .matmul(residual)?;
    Ok((update + residual)?)
}
