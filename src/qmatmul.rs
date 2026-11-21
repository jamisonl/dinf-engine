//! Engine projection dispatch. Packed Q2_K Metal tensors use a bounded kernel launch.

#[cfg(not(target_os = "macos"))]
pub(crate) use candle_core::quantized::QMatMul;

#[cfg(target_os = "macos")]
mod metal_q2;

#[cfg(target_os = "macos")]
use candle_core::quantized::{GgmlDType, QMatMul as CandleQMatMul, QTensor};
#[cfg(target_os = "macos")]
use candle_core::{Module, Result, Tensor};
#[cfg(target_os = "macos")]
use std::sync::Arc;

/// Matches Candle's existing forms and adds one bounded packed Metal Q2 form.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
pub(crate) enum QMatMul {
    QTensor(Arc<QTensor>),
    Tensor(Tensor),
    TensorF16(Tensor),
    MetalQ2(metal_q2::MetalQ2),
}

#[cfg(target_os = "macos")]
impl QMatMul {
    pub(crate) fn from_arc(weight: Arc<QTensor>) -> Result<Self> {
        // Preserve Candle's dense/dequantization selectors before replacing the
        // one packed form whose scalar host launch exceeds its output extent.
        Ok(match CandleQMatMul::from_arc(weight)? {
            CandleQMatMul::QTensor(weight)
                if weight.device().is_metal() && weight.dtype() == GgmlDType::Q2K =>
            {
                Self::MetalQ2(metal_q2::MetalQ2::from_qtensor(&weight)?)
            }
            CandleQMatMul::QTensor(weight) => Self::QTensor(weight),
            CandleQMatMul::Tensor(weight) => Self::Tensor(weight),
            CandleQMatMul::TensorF16(weight) => Self::TensorF16(weight),
        })
    }

    pub(crate) fn from_qtensor(weight: QTensor) -> Result<Self> {
        Self::from_arc(Arc::new(weight))
    }
}

#[cfg(target_os = "macos")]
impl Module for QMatMul {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::MetalQ2(projection) => projection.forward(input),
            Self::QTensor(weight) => {
                if weight.device().is_metal() && weight.dtype() == GgmlDType::Q2K {
                    candle_core::bail!("packed Metal Q2_K must use QMatMul::from_arc/from_qtensor")
                }
                // Candle's Metal matrix-matrix path (more than one row) reads the
                // input from the start of its buffer, ignoring a view's offset.
                let copied;
                let input = if input.device().is_metal()
                    && input.layout().start_offset() != 0
                    && input.rank() >= 2
                    && input.dim(candle_core::D::Minus2)? != 1
                {
                    copied = input.force_contiguous()?;
                    &copied
                } else {
                    input
                };
                CandleQMatMul::QTensor(Arc::clone(weight)).forward(input)
            }
            Self::Tensor(weight) => CandleQMatMul::Tensor(weight.clone()).forward(input),
            Self::TensorF16(weight) => CandleQMatMul::TensorF16(weight.clone()).forward(input),
        }
    }
}

#[cfg(test)]
#[path = "qmatmul/tests.rs"]
mod tests;
