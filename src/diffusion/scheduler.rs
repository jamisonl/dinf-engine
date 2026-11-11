//! Qwen Image 2.1's resolution-shifted flow-matching Euler schedule.
//!
//! Equations follow Diffusers' Apache-2.0 FlowMatchEulerDiscreteScheduler.

use candle_core::{DType, Result, Tensor};
use serde::Deserialize;

/// Supported fields from the released scheduler configuration.
#[derive(Clone, Debug, Deserialize)]
pub struct SchedulerConfig {
    /// Number of training timesteps (1000 in the reference model).
    pub num_train_timesteps: usize,
    /// Static shift, when dynamic shifting is disabled.
    pub shift: f64,
    /// Whether image sequence length determines the shift.
    pub use_dynamic_shifting: bool,
    /// Reference sequence length for the lower shift.
    pub base_image_seq_len: usize,
    /// Reference sequence length for the upper shift.
    pub max_image_seq_len: usize,
    /// Shift at the lower reference sequence length.
    pub base_shift: f64,
    /// Shift at the upper reference sequence length.
    pub max_shift: f64,
    /// Terminal stretch before appending the final zero.
    pub shift_terminal: Option<f64>,
    /// Only the released exponential time shift is supported.
    pub time_shift_type: String,
    /// Inversion is not part of this model's supported schedule.
    pub invert_sigmas: bool,
    /// Stochastic sampling is not part of this model's supported schedule.
    pub stochastic_sampling: bool,
    /// Alternative sigma schedules are rejected, not silently ignored.
    pub use_karras_sigmas: bool,
    /// Alternative sigma schedules are rejected, not silently ignored.
    pub use_exponential_sigmas: bool,
    /// Alternative sigma schedules are rejected, not silently ignored.
    pub use_beta_sigmas: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            shift: 1.,
            use_dynamic_shifting: true,
            base_image_seq_len: 256,
            max_image_seq_len: 8192,
            base_shift: 0.5,
            max_shift: 0.9,
            shift_terminal: Some(0.02),
            time_shift_type: "exponential".into(),
            invert_sigmas: false,
            stochastic_sampling: false,
            use_karras_sigmas: false,
            use_exponential_sigmas: false,
            use_beta_sigmas: false,
        }
    }
}

/// Immutable schedule: the caller owns the step index, allowing cancellation
/// without accidentally advancing state.
#[derive(Debug)]
pub struct FlowMatchEuler {
    sigmas: Vec<f32>,
    training_steps: f32,
}

impl FlowMatchEuler {
    /// Build the reference schedule for a target latent sequence length.
    pub fn new(config: &SchedulerConfig, steps: usize, image_sequence_len: usize) -> Result<Self> {
        if !(2..=1000).contains(&steps)
            || image_sequence_len == 0
            || config.num_train_timesteps == 0
            || config.num_train_timesteps > 1_000_000
            || config.max_image_seq_len <= config.base_image_seq_len
            || !config.shift.is_finite()
            || config.shift <= 0.
            || !config.base_shift.is_finite()
            || !config.max_shift.is_finite()
            || config.time_shift_type != "exponential"
            || config.invert_sigmas
            || config.stochastic_sampling
            || config.use_karras_sigmas
            || config.use_exponential_sigmas
            || config.use_beta_sigmas
            || config
                .shift_terminal
                .is_some_and(|v| !v.is_finite() || v <= 0. || v >= 1.)
        {
            candle_core::bail!("unsupported or invalid Qwen Image flow scheduler configuration");
        }
        let slope = (config.max_shift - config.base_shift)
            / (config.max_image_seq_len - config.base_image_seq_len) as f64;
        let mu = slope * image_sequence_len as f64 + config.base_shift
            - slope * config.base_image_seq_len as f64;
        let exp_mu = mu.exp();
        if !exp_mu.is_finite() || exp_mu == 0. {
            candle_core::bail!("flow scheduler shift is outside the supported numerical range");
        }
        let mut sigmas = (0..steps)
            .map(|i| {
                // Diffusers converts the linspace to float32 before shifting.
                let t = (1. - i as f64 / steps as f64) as f32;
                if config.use_dynamic_shifting {
                    exp_mu as f32 / (exp_mu as f32 + (1. / t - 1.))
                } else {
                    config.shift as f32 * t / (1. + (config.shift as f32 - 1.) * t)
                }
            })
            .collect::<Vec<_>>();
        if let Some(terminal) = config.shift_terminal {
            let scale = (1. - sigmas[steps - 1]) / (1. - terminal as f32);
            for sigma in &mut sigmas {
                *sigma = 1. - (1. - *sigma) / scale;
            }
        }
        if sigmas.iter().any(|s| !s.is_finite() || *s <= 0. || *s > 1.)
            || sigmas.windows(2).any(|w| w[0] <= w[1])
        {
            candle_core::bail!("flow scheduler generated invalid sigmas");
        }
        sigmas.push(0.);
        Ok(Self {
            sigmas,
            training_steps: config.num_train_timesteps as f32,
        })
    }

    /// Noise levels, including the appended final zero.
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// Number of denoising iterations.
    pub fn steps(&self) -> usize {
        self.sigmas.len() - 1
    }

    /// Training timestep. The transformer consumes this divided by 1000.
    pub fn timestep(&self, step: usize) -> Result<f32> {
        if step >= self.steps() {
            candle_core::bail!("flow scheduler step out of bounds");
        }
        Ok(self.sigmas[step] * self.training_steps)
    }

    /// Euler update, accumulating in F32 and returning the prediction dtype.
    pub fn step(&self, step: usize, sample: &Tensor, prediction: &Tensor) -> Result<Tensor> {
        if step >= self.steps()
            || sample.shape() != prediction.shape()
            || sample.dtype() != prediction.dtype()
            || !matches!(sample.dtype(), DType::F32 | DType::BF16)
            || !sample.device().same_device(prediction.device())
        {
            candle_core::bail!("invalid flow scheduler step or sample");
        }
        let dt = self.sigmas[step + 1] - self.sigmas[step];
        // Reference multiplies the BF16 model output before adding it to the
        // explicitly promoted F32 sample; retain that rounding boundary.
        let delta = (prediction * dt as f64)?.to_dtype(DType::F32)?;
        (sample.to_dtype(DType::F32)? + delta)?.to_dtype(prediction.dtype())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn endpoints_and_resolution_shift() -> Result<()> {
        let small = FlowMatchEuler::new(&SchedulerConfig::default(), 40, 4096)?;
        let large = FlowMatchEuler::new(&SchedulerConfig::default(), 40, 16384)?;
        assert_eq!(small.sigmas()[0], 1.);
        assert!((small.sigmas()[39] - 0.02).abs() < 1e-6);
        assert_eq!(small.sigmas()[40], 0.);
        assert!(large.sigmas()[20] > small.sigmas()[20]);
        // Independent NumPy reference: linspace(float32), exponential shift,
        // stretch_shift_to_terminal. These are not generated by this module.
        for (index, expected) in [
            (1, 0.98696375_f32),
            (5, 0.93134063),
            (10, 0.8528682),
            (20, 0.6566663),
            (30, 0.38193327),
        ] {
            assert!((small.sigmas()[index] - expected).abs() < 2e-7);
        }
        assert!(small.timestep(40).is_err());
        Ok(())
    }

    #[test]
    fn euler_constant_velocity_and_invalid_inputs() -> Result<()> {
        let config = SchedulerConfig {
            use_dynamic_shifting: false,
            shift_terminal: None,
            ..SchedulerConfig::default()
        };
        let scheduler = FlowMatchEuler::new(&config, 4, 4)?;
        assert_eq!(scheduler.sigmas(), &[1., 0.75, 0.5, 0.25, 0.]);
        let mut x = Tensor::new(&[3_f32, 5.], &Device::Cpu)?;
        let velocity = Tensor::new(&[2_f32, 4.], &Device::Cpu)?;
        for i in 0..4 {
            x = scheduler.step(i, &x, &velocity)?;
        }
        assert_eq!(x.to_vec1::<f32>()?, vec![1., 1.]);
        assert!(FlowMatchEuler::new(&config, 1, 4).is_err());
        assert!(FlowMatchEuler::new(&config, 40, 0).is_err());
        assert!(
            FlowMatchEuler::new(
                &SchedulerConfig {
                    use_karras_sigmas: true,
                    ..config
                },
                40,
                4
            )
            .is_err()
        );
        Ok(())
    }
}
