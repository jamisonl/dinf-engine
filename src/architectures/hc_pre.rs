//! Four-stream input mix, with the original graph as fallback.

use anyhow::Result;
use candle_core::{D, Tensor};

pub(super) fn forward(
    input: &Tensor,
    mixes: &Tensor,
    scale: &Tensor,
    base: &Tensor,
    eps: f64,
    fused: bool,
) -> Result<Tensor> {
    if fused {
        #[cfg(target_os = "macos")]
        if input.device().is_metal() && input.dtype() == candle_core::DType::F32 {
            return super::hc_pre_metal::forward(input, mixes, scale, base, eps);
        }
    }
    reference(input, mixes, scale, base, eps)
}

pub(super) fn reference(
    input: &Tensor,
    mixes: &Tensor,
    scale: &Tensor,
    base: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    mix(input, &weights(mixes, scale, base, eps)?)
}

pub(super) fn weights(mixes: &Tensor, scale: &Tensor, base: &Tensor, eps: f64) -> Result<Tensor> {
    let scale = scale.narrow(0, 0, 1)?.reshape((1, 1, 1))?;
    let base = base.narrow(0, 0, 4)?.reshape((1, 1, 4))?;
    let affine = mixes
        .narrow(D::Minus1, 0, 4)?
        .broadcast_mul(&scale)?
        .broadcast_add(&base)?;
    Ok(candle_nn::ops::sigmoid(&affine)?.affine(1., eps)?)
}

pub(super) fn mix(input: &Tensor, weights: &Tensor) -> Result<Tensor> {
    Ok(weights.unsqueeze(D::Minus1)?.broadcast_mul(input)?.sum(2)?)
}
