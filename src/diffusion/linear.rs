//! Full-precision diffusion projections, with a native Metal 4 BF16 path.
use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

pub(super) struct Linear(candle_nn::Linear);

impl Linear {
    pub(super) fn weight(&self) -> &Tensor {
        self.0.weight()
    }
}

pub(super) fn linear_no_bias(input: usize, output: usize, vb: VarBuilder<'_>) -> Result<Linear> {
    Ok(Linear(candle_nn::linear_no_bias(input, output, vb)?))
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        #[cfg(target_os = "macos")]
        if let Some(output) = super::dense_metal::try_linear(input, self.0.weight())? {
            return Ok(output);
        }
        self.0.forward(input)
    }
}
