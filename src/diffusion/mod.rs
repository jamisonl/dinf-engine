//! Native, precision-preserving image diffusion.
//!
//! This runtime is deliberately independent from token generation. Components
//! are loaded from a local, operator-verified Qwen Image 2.1 snapshot; no remote
//! model code or Python process is executed.

pub(crate) mod attention;
#[cfg(target_os = "macos")]
mod dense_metal;
#[cfg(target_os = "macos")]
mod fused_metal;
mod linear;
mod pipeline;
pub mod scheduler;
mod text_encoder;
pub mod transformer;
pub mod vae;
mod weights;

pub use pipeline::{DiffusionProgress, GeneratedImage, ImageGenerationOptions, QwenImagePipeline};

#[cfg(test)]
mod pipeline_tests;

#[cfg(all(test, target_os = "macos"))]
mod metal_tests;
