//! DeepSeek V4 Flash stage support.
//!
//! Candle 0.11 does not know the V4 GGUF types or its hyper-connection
//! layout. This stage keeps packed weights in the raw GGUF form and sends
//! rank-2 packed products through the fused V4 matmul. It uses the raw K/V
//! cache window plus learned CSA/HCA compressed keys. Pooling state belongs
//! to each request and survives chunk boundaries and prompt-cache forks.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::{D, DType, Device, IndexOp, Module, Tensor};

use super::activation::softplus;
use super::deepseek4_qmatmul::DeepSeek4QMatMul;
use super::{LayerCache, ModelArchitecture, StageInput, StageOutput};
use crate::deepseek4_gguf::{DeepSeek4DType, RawTensor};
use crate::gguf_loader::{GgufMetadata, StageTensors};
use crate::turboquant::TurboQuantConfig;

type RawMap = HashMap<String, Arc<RawTensor>>;

mod decode_batch;
#[cfg(all(test, target_os = "macos"))]
mod hc_tests;
mod profile;
mod routed_experts;
use profile::Region;
pub use profile::{report_layer_profile, reset_layer_profile};

const V4_HEAD_COUNT: usize = 4;
const V4_Q_NOPE_DIM: usize = 448;
const V4_Q_ROPE_DIM: usize = 64;

struct V4RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl V4RmsNorm {
    fn new(weight: Tensor, eps: f64) -> Result<Self> {
        anyhow::ensure!(weight.rank() == 1, "V4 RMS norm weight must be a vector");
        Ok(Self {
            weight: weight.to_dtype(DType::F32)?,
            eps,
        })
    }

    fn ones(size: usize, eps: f64, device: &Device) -> Result<Self> {
        Self::new(Tensor::ones(size, DType::F32, device)?, eps)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = if x.is_contiguous() {
            x.clone()
        } else {
            x.contiguous()?
        };
        candle_nn::ops::rms_norm(&x, &self.weight, self.eps as f32).map_err(Into::into)
    }
}

fn raw<'a>(weights: &'a RawMap, name: &str) -> Result<&'a Arc<RawTensor>> {
    weights
        .get(name)
        .with_context(|| format!("missing DeepSeek V4 tensor {name}"))
}

fn linear(weights: &RawMap, name: &str, device: &Device) -> Result<DeepSeek4QMatMul> {
    DeepSeek4QMatMul::new_on_device(Arc::clone(raw(weights, name)?), device)
}

fn vector(weights: &RawMap, name: &str, device: &Device) -> Result<Tensor> {
    raw(weights, name)?
        .to_dense(device)?
        .to_dtype(DType::F32)
        .map_err(Into::into)
}

fn optional_vector(weights: &RawMap, name: &str, device: &Device) -> Result<Option<Tensor>> {
    weights
        .get(name)
        .map(|value| {
            value
                .to_dense(device)?
                .to_dtype(DType::F32)
                .map_err(Into::into)
        })
        .transpose()
}

fn norm(weights: &RawMap, name: &str, md: &GgufMetadata, device: &Device) -> Result<V4RmsNorm> {
    V4RmsNorm::new(
        vector(weights, &format!("{name}.weight"), device)?,
        md.rms_norm_eps,
    )
}

enum V4Embedding {
    Dense(Tensor),
    Quantized {
        raw: Arc<RawTensor>,
        dtype: candle_core::quantized::GgmlDType,
        hidden_dim: usize,
    },
}

impl V4Embedding {
    fn forward(&self, tokens: &[u32], device: &Device) -> Result<Tensor> {
        match self {
            Self::Dense(embedding) => {
                let token_tensor = Tensor::new(tokens, device)?;
                embedding.index_select(&token_tensor, 0).map_err(Into::into)
            }
            Self::Quantized {
                raw,
                dtype,
                hidden_dim,
            } => {
                let mut rows = Vec::with_capacity(tokens.len());
                for &token in tokens {
                    let row = raw.slice_rows(token as usize, 1)?;
                    let bytes = row.to_bytes()?;
                    let quantized = candle_core::quantized::ggml_file::qtensor_from_ggml(
                        *dtype,
                        &bytes,
                        vec![1, *hidden_dim],
                        device,
                    )?;
                    rows.push(quantized.dequantize(device)?.to_dtype(DType::F32)?);
                }
                let row_refs = rows.iter().collect::<Vec<_>>();
                Tensor::cat(&row_refs, 0).map_err(Into::into)
            }
        }
    }
}

fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::softmax_last_dim(x).map_err(Into::into)
}

fn expert_activation(gate: &Tensor, up: &Tensor, clamp: f32) -> Result<Tensor> {
    if clamp > 0. {
        super::indexed_moe::silu_mul_clamped(gate, up, clamp)
    } else {
        super::indexed_moe::silu_mul(gate, up)
    }
}

fn precompute_rope(
    head_dim: usize,
    theta: f64,
    context_length: usize,
    device: &Device,
    yarn: Option<&GgufMetadata>,
) -> Result<(Tensor, Tensor, Tensor)> {
    anyhow::ensure!(
        head_dim > 0 && head_dim.is_multiple_of(2) && theta.is_finite() && theta > 1.,
        "invalid V4 rotary geometry"
    );
    anyhow::ensure!(
        context_length > 0 && context_length <= u32::MAX as usize,
        "invalid V4 rotary context"
    );
    let half_dim = head_dim / 2;
    let correction = if let Some(md) = yarn {
        anyhow::ensure!(
            md.rope_scaling_factor.is_finite()
                && md.rope_scaling_factor >= 1.
                && md.rope_yarn_beta_fast.is_finite()
                && md.rope_yarn_beta_fast > 0.
                && md.rope_yarn_beta_slow.is_finite()
                && md.rope_yarn_beta_slow > 0.
                && md.rope_yarn_beta_fast >= md.rope_yarn_beta_slow
                && md.rope_scaling_original_context_length > 0,
            "invalid V4 YaRN parameters"
        );
        let boundary = |rotations: f64| {
            head_dim as f64
                * (md.rope_scaling_original_context_length as f64
                    / (rotations * 2. * std::f64::consts::PI))
                    .ln()
                / (2. * theta.ln())
        };
        let low = boundary(md.rope_yarn_beta_fast).floor().max(0.);
        let high = boundary(md.rope_yarn_beta_slow)
            .ceil()
            .min((head_dim - 1) as f64);
        Some((low, high.max(low + 0.001), md.rope_scaling_factor))
    } else {
        None
    };
    let inv_freq = (0..half_dim)
        .map(|i| {
            let frequency = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
            let scale = correction.map_or(1., |(low, high, factor)| {
                let ramp = ((i as f64 - low) / (high - low)).clamp(0., 1.);
                1. - ramp + ramp / factor
            });
            (frequency * scale) as f32
        })
        .collect::<Vec<_>>();
    let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
    let positions = Tensor::arange(0u32, context_length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?;
    let angles = positions.matmul(&inv_freq.reshape((1, half_dim))?)?;
    let cos = angles.cos()?;
    let sin = angles.sin()?;
    let sin_neg = sin.neg()?;
    Ok((cos, sin, sin_neg))
}

fn apply_rope(x: &Tensor, start_pos: usize, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let seq_len = x.dim(2)?;
    let cos = cos.narrow(0, start_pos, seq_len)?;
    let sin = sin.narrow(0, start_pos, seq_len)?;
    Ok(candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)?)
}

fn affine(x: &Tensor, scale: &Tensor, base: &Tensor) -> Result<Tensor> {
    Ok(x.broadcast_mul(scale)?.broadcast_add(base)?)
}

struct HcMix {
    projection: DeepSeek4QMatMul,
    base: Tensor,
    scale: Tensor,
    norm: V4RmsNorm,
    stream_count: usize,
    hidden_dim: usize,
    sinkhorn_iterations: usize,
    eps: f64,
    fused_sinkhorn: bool,
    fused_post: bool,
    fused_pre: bool,
}

impl HcMix {
    fn new(weights: &RawMap, prefix: &str, md: &GgufMetadata, device: &Device) -> Result<Self> {
        let stream_count = md.hyperconnection_count;
        anyhow::ensure!(
            stream_count == V4_HEAD_COUNT,
            "DeepSeek V4 needs four hyper-connection streams, got {stream_count}"
        );
        let hidden_dim = md.hidden_dim;
        let mix_dim = (2 + stream_count) * stream_count;
        let base = vector(weights, &format!("{prefix}_base.weight"), device)?;
        let scale = vector(weights, &format!("{prefix}_scale.weight"), device)?;
        anyhow::ensure!(base.dims() == [mix_dim], "invalid {prefix}_base shape");
        anyhow::ensure!(scale.dims() == [3], "invalid {prefix}_scale shape");
        let apple_silicon_metal =
            cfg!(all(target_os = "macos", target_arch = "aarch64")) && device.is_metal();
        Ok(Self {
            projection: linear(weights, &format!("{prefix}_fn.weight"), device)?,
            base,
            scale,
            norm: V4RmsNorm::ones(stream_count * hidden_dim, md.rms_norm_eps, device)?,
            stream_count,
            hidden_dim,
            sinkhorn_iterations: md.hyperconnection_sinkhorn_iterations.max(1),
            eps: md.hyperconnection_epsilon.max(1e-6),
            fused_sinkhorn: apple_silicon_metal,
            fused_post: apple_silicon_metal,
            fused_pre: apple_silicon_metal,
        })
    }

    fn pre(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (_, seq_len, streams, hidden_dim) = x.dims4()?;
        anyhow::ensure!(
            streams == self.stream_count && hidden_dim == self.hidden_dim,
            "invalid V4 stream shape {:?}",
            x.dims()
        );
        let flat = x.reshape((1, seq_len, self.stream_count * self.hidden_dim))?;
        let mixes = self.projection.forward(&self.norm.forward(&flat)?)?;

        let scale_post = self.scale.narrow(0, 1, 1)?.reshape((1, 1, 1))?;
        let scale_comb = self.scale.narrow(0, 2, 1)?.reshape((1, 1, 1))?;
        let base_post = self
            .base
            .narrow(0, self.stream_count, self.stream_count)?
            .reshape((1, 1, self.stream_count))?;
        let base_comb = self
            .base
            .narrow(
                0,
                self.stream_count * 2,
                self.stream_count * self.stream_count,
            )?
            .reshape((1, 1, self.stream_count, self.stream_count))?;

        // Keep the original control/update order when fusion is disabled.
        let pre = if self.fused_pre {
            None
        } else {
            Some(super::hc_pre::weights(
                &mixes,
                &self.scale,
                &self.base,
                self.eps,
            )?)
        };
        let post = candle_nn::Activation::Sigmoid
            .forward(&affine(
                &mixes.narrow(D::Minus1, self.stream_count, self.stream_count)?,
                &scale_post,
                &base_post,
            )?)?
            .affine(2.0, 0.0)?;
        let comb = affine(
            &mixes.narrow(
                D::Minus1,
                self.stream_count * 2,
                self.stream_count * self.stream_count,
            )?,
            &scale_comb,
            &base_comb.reshape((1, 1, self.stream_count * self.stream_count))?,
        )?
        .reshape((1, seq_len, self.stream_count, self.stream_count))?;
        let comb = self.sinkhorn(comb)?;
        let update = if let Some(pre) = pre {
            super::hc_pre::mix(x, &pre)?
        } else {
            super::hc_pre::forward(x, &mixes, &self.scale, &self.base, self.eps, self.fused_pre)?
        };
        Ok((update, post, comb))
    }

    fn post(
        &self,
        update: &Tensor,
        residual: &Tensor,
        post: &Tensor,
        comb: &Tensor,
    ) -> Result<Tensor> {
        super::hc_post::forward(update, residual, post, comb, self.fused_post)
    }

    fn sinkhorn(&self, matrix: Tensor) -> Result<Tensor> {
        super::hc_sinkhorn::normalize(
            matrix,
            self.sinkhorn_iterations,
            self.eps,
            self.fused_sinkhorn,
        )
    }
}

struct HcHead {
    projection: DeepSeek4QMatMul,
    base: Tensor,
    scale: Tensor,
    norm: V4RmsNorm,
    stream_count: usize,
    hidden_dim: usize,
    eps: f64,
}

impl HcHead {
    fn new(weights: &RawMap, md: &GgufMetadata, device: &Device) -> Result<Self> {
        let stream_count = md.hyperconnection_count;
        let hidden_dim = md.hidden_dim;
        let base = vector(weights, "output_hc_base.weight", device)?;
        let scale = vector(weights, "output_hc_scale.weight", device)?;
        anyhow::ensure!(
            base.dims() == [stream_count],
            "invalid output_hc_base shape"
        );
        anyhow::ensure!(scale.dims() == [1], "invalid output_hc_scale shape");
        Ok(Self {
            projection: linear(weights, "output_hc_fn.weight", device)?,
            base,
            scale,
            norm: V4RmsNorm::ones(stream_count * hidden_dim, md.rms_norm_eps, device)?,
            stream_count,
            hidden_dim,
            eps: md.hyperconnection_epsilon.max(1e-6),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, seq_len, streams, hidden_dim) = x.dims4()?;
        anyhow::ensure!(
            streams == self.stream_count && hidden_dim == self.hidden_dim,
            "invalid V4 head stream shape {:?}",
            x.dims()
        );
        let flat = x.reshape((1, seq_len, self.stream_count * self.hidden_dim))?;
        let mixes = self.projection.forward(&self.norm.forward(&flat)?)?;
        let weights = candle_nn::Activation::Sigmoid
            .forward(&affine(
                &mixes,
                &self.scale.reshape((1, 1, 1))?,
                &self.base.reshape((1, 1, self.stream_count))?,
            )?)?
            .affine(1.0, self.eps)?;
        Ok(weights.unsqueeze(D::Minus1)?.broadcast_mul(x)?.sum(2)?)
    }
}

mod attention;
pub use attention::DeepSeek4Cache;
use attention::V4Attention;

struct V4Expert {
    gate: DeepSeek4QMatMul,
    up: DeepSeek4QMatMul,
    down: DeepSeek4QMatMul,
}

struct V4Moe {
    router: DeepSeek4QMatMul,
    hash_selected_experts: Option<Tensor>,
    experts: Vec<V4Expert>,
    shared_gate: DeepSeek4QMatMul,
    shared_up: DeepSeek4QMatMul,
    shared_down: DeepSeek4QMatMul,
    bias: Option<Tensor>,
    active_experts: usize,
    scale: f64,
    normalize_weights: bool,
    clamp: f32,
    shared_clamp: f32,
    group_experts: bool,
}

impl V4Moe {
    fn new(
        weights: &RawMap,
        prefix: &str,
        md: &GgufMetadata,
        layer: usize,
        device: &Device,
    ) -> Result<Self> {
        let clamp = md.swiglu_clamp_exp.get(layer).copied().unwrap_or(0.) as f32;
        let shared_clamp = md
            .swiglu_clamp_shexp
            .get(layer)
            .copied()
            .unwrap_or(clamp as f64) as f32;
        anyhow::ensure!(
            clamp.is_finite() && clamp >= 0. && shared_clamp.is_finite() && shared_clamp >= 0.,
            "invalid V4 expert clamp"
        );
        let gate_raw = raw(weights, &format!("{prefix}.ffn_gate_exps.weight"))?;
        let up_raw = raw(weights, &format!("{prefix}.ffn_up_exps.weight"))?;
        let down_raw = raw(weights, &format!("{prefix}.ffn_down_exps.weight"))?;
        anyhow::ensure!(
            gate_raw.shape().first().copied() == Some(md.expert_count)
                && up_raw.shape().first().copied() == Some(md.expert_count)
                && down_raw.shape().first().copied() == Some(md.expert_count),
            "DeepSeek V4 expert tensor count differs from metadata"
        );
        let mut experts = Vec::with_capacity(md.expert_count);
        for expert in 0..md.expert_count {
            experts.push(V4Expert {
                gate: DeepSeek4QMatMul::new_on_device(
                    Arc::new(gate_raw.slice_first_dim(expert)?),
                    device,
                )?,
                up: DeepSeek4QMatMul::new_on_device(
                    Arc::new(up_raw.slice_first_dim(expert)?),
                    device,
                )?,
                down: DeepSeek4QMatMul::new_on_device(
                    Arc::new(down_raw.slice_first_dim(expert)?),
                    device,
                )?,
            });
        }
        let hash_selected_experts = weights
            .get(&format!("{prefix}.ffn_gate_tid2eid.weight"))
            .map(|value| {
                let tensor = value
                    .to_dense(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .to_device(device)?;
                anyhow::ensure!(
                    tensor.dims() == [md.vocab_size, md.expert_used_count],
                    "invalid {prefix} hash expert map shape {:?}",
                    tensor.dims()
                );
                Ok(tensor)
            })
            .transpose()?;
        Ok(Self {
            router: linear(weights, &format!("{prefix}.ffn_gate_inp.weight"), device)?,
            hash_selected_experts,
            experts,
            shared_gate: linear(weights, &format!("{prefix}.ffn_gate_shexp.weight"), device)?,
            shared_up: linear(weights, &format!("{prefix}.ffn_up_shexp.weight"), device)?,
            shared_down: linear(weights, &format!("{prefix}.ffn_down_shexp.weight"), device)?,
            bias: optional_vector(weights, &format!("{prefix}.exp_probs_b.bias"), device)?,
            active_experts: md.expert_used_count,
            scale: md.expert_weights_scale,
            normalize_weights: md.expert_weights_norm,
            clamp,
            shared_clamp,
            group_experts: cfg!(all(target_os = "macos", target_arch = "aarch64"))
                && device.is_metal(),
        })
    }

    fn expert_forward(expert: &V4Expert, x: &Tensor, clamp: f32) -> Result<Tensor> {
        let gate = expert.gate.forward(x)?;
        let up = expert.up.forward(x)?;
        let activated = expert_activation(&gate, &up, clamp)?;
        expert.down.forward(&activated)
    }

    fn shared_forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.shared_gate.forward(x)?;
        let up = self.shared_up.forward(x)?;
        let activated = expert_activation(&gate, &up, self.shared_clamp)?;
        self.shared_down.forward(&activated)
    }

    fn forward(&self, x: &Tensor, token_ids: Option<&[u32]>) -> Result<Tensor> {
        self.forward_impl(x, token_ids, false)
    }

    fn forward_decode_rows(&self, x: &Tensor, token_ids: &[u32]) -> Result<Tensor> {
        self.forward_impl(x, Some(token_ids), true)
    }

    fn forward_impl(
        &self,
        x: &Tensor,
        token_ids: Option<&[u32]>,
        decode_rows: bool,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        anyhow::ensure!(batch == 1, "DeepSeek V4 stage expects batch size one");
        if self.hash_selected_experts.is_some() {
            anyhow::ensure!(
                token_ids.is_some_and(|tokens| tokens.len() == seq_len),
                "V4 hash-routed MoE needs token IDs for every input row"
            );
        }
        let stamp = profile::start(x.device(), seq_len, Region::Router)?;
        let router_logits = if decode_rows {
            self.router.forward_decode_rows(x)?
        } else {
            self.router.forward(x)?
        };
        let router_probs = softplus(&router_logits)?.sqrt()?;
        let selection = if let Some(bias) = &self.bias {
            router_probs.broadcast_add(&bias.reshape((1, 1, self.experts.len()))?)?
        } else {
            router_probs.clone()
        };
        let selection = selection.squeeze(0)?.to_vec2::<f32>()?;
        let probs = router_probs.squeeze(0)?.to_vec2::<f32>()?;
        let hash_choices = if let Some(hash_selected_experts) = &self.hash_selected_experts {
            let tokens = token_ids.context("V4 hash-routed MoE is missing token IDs")?;
            let token_tensor = Tensor::new(tokens, x.device())?;
            Some(
                hash_selected_experts
                    .index_select(&token_tensor, 0)?
                    .to_vec2::<f32>()?,
            )
        } else {
            None
        };
        let mut routes = Vec::with_capacity(seq_len);
        for token in 0..seq_len {
            let choices = if let Some(hash_choices) = &hash_choices {
                hash_choices[token]
                    .iter()
                    .map(|expert| (*expert as usize, 0.0))
                    .filter(|(expert, _)| *expert < self.experts.len())
                    .collect::<Vec<_>>()
            } else {
                let mut choices = selection[token]
                    .iter()
                    .copied()
                    .enumerate()
                    .collect::<Vec<_>>();
                choices.sort_by(|a, b| b.1.total_cmp(&a.1));
                choices.truncate(self.active_experts.min(choices.len()));
                choices
            };
            let denom = if self.normalize_weights {
                choices
                    .iter()
                    .map(|(expert, _)| probs[token][*expert])
                    .sum::<f32>()
                    .max(6.103_515_6e-5)
            } else {
                1.0
            };
            routes.push(
                choices
                    .into_iter()
                    .map(|(expert, _)| (expert, probs[token][expert] / denom * self.scale as f32))
                    .collect::<Vec<_>>(),
            );
        }
        profile::finish(x.device(), stamp)?;
        let stamp = profile::start(x.device(), seq_len, Region::RoutedExperts)?;
        let output = if routed_experts::should_group(self.group_experts, decode_rows, seq_len) {
            self.grouped_experts_forward(x, &routes)?
        } else {
            self.scalar_experts_forward(x, &routes)?
        };

        profile::finish(x.device(), stamp)?;
        let stamp = profile::start(x.device(), seq_len, Region::SharedExperts)?;
        let shared = if decode_rows {
            let gate = self.shared_gate.forward_decode_rows(x)?;
            let up = self.shared_up.forward_decode_rows(x)?;
            let activated = expert_activation(&gate, &up, self.shared_clamp)?;
            self.shared_down.forward_decode_rows(&activated)?
        } else {
            self.shared_forward(x)?
        };
        let output = (output + shared)?;
        profile::finish(x.device(), stamp)?;
        Ok(output)
    }
}

struct V4Block {
    attn_norm: V4RmsNorm,
    attention: V4Attention,
    attn_hc: HcMix,
    ffn_norm: V4RmsNorm,
    moe: V4Moe,
    ffn_hc: HcMix,
}

/// A loaded DeepSeek V4 Flash stage for pipeline execution.
pub struct DeepSeek4Stage {
    embedding: Option<V4Embedding>,
    blocks: Vec<V4Block>,
    output_norm: Option<V4RmsNorm>,
    lm_head: Option<DeepSeek4QMatMul>,
    output_hc: Option<HcHead>,
    is_head: bool,
    is_tail: bool,
    model_hidden_dim: usize,
    stage_hidden_dim: usize,
    stream_count: usize,
    num_layers: usize,
    vocab_size: usize,
    context_length: usize,
    decode_batch_enabled: bool,
    device: Device,
}

impl DeepSeek4Stage {
    pub(crate) fn validate_stage_config(
        md: &GgufMetadata,
        range: &crate::gguf_loader::LayerRange,
        quantized_kv: bool,
    ) -> Result<()> {
        range.validate()?;
        anyhow::ensure!(md.architecture == "deepseek4", "not a DeepSeek V4 stage");
        anyhow::ensure!(
            !quantized_kv,
            "V4 uses native window/compressed caches; TurboQuant is not supported"
        );
        anyhow::ensure!(
            range.total_layers == md.num_layers,
            "V4 layer range total differs from metadata"
        );
        anyhow::ensure!(
            md.hash_layer_count <= md.num_layers,
            "V4 hash prefix exceeds the model"
        );
        // Pipeline activation messages do not carry input token IDs. Hash
        // routing must therefore finish in the head, which owns those IDs.
        anyhow::ensure!(
            if range.is_head() {
                range.end >= md.hash_layer_count
            } else {
                range.start >= md.hash_layer_count
            },
            "V4 head must contain all {} hash-routed layers; cannot split inside that prefix",
            md.hash_layer_count
        );
        Ok(())
    }

    /// Construct a DeepSeek V4 stage from its assigned GGUF tensors.
    ///
    /// Returns an error when V4 hyper-connection metadata, compression
    /// schedules, or required tensors are missing or inconsistent.
    pub fn from_stage_tensors(
        stage: StageTensors,
        quant_config: Option<Arc<TurboQuantConfig>>,
    ) -> Result<Self> {
        let md = &stage.metadata;
        Self::validate_stage_config(md, &stage.range, quant_config.is_some())?;
        anyhow::ensure!(
            md.hyperconnection_count > 0,
            "V4 hyper-connection count is missing"
        );
        anyhow::ensure!(
            md.output_group_count > 0,
            "V4 output group count is missing"
        );
        anyhow::ensure!(md.output_lora_rank > 0, "V4 output LoRA rank is missing");
        anyhow::ensure!(
            md.compress_ratios.len() >= md.num_layers,
            "V4 compression schedule has {} entries for {} layers",
            md.compress_ratios.len(),
            md.num_layers
        );
        anyhow::ensure!(
            md.compress_ratios[..md.num_layers]
                .iter()
                .all(|ratio| matches!(ratio, 0 | 4 | 128)),
            "V4 compression schedule must use ratios 0, 4, or 128"
        );
        if md.compress_ratios[..md.num_layers].contains(&4) {
            attention::validate_indexer_geometry(md)?;
        }
        let weights = &stage.raw_tensors;
        anyhow::ensure!(!weights.is_empty(), "DeepSeek V4 stage has no raw tensors");
        let device = &stage.device;
        let rope = precompute_rope(
            md.qk_rope_head_dim,
            md.rope_theta,
            md.context_length,
            device,
            None,
        )?;
        let compressed_rope = if md.compress_ratios[..md.num_layers]
            .iter()
            .any(|&ratio| ratio > 0)
        {
            precompute_rope(
                md.qk_rope_head_dim,
                md.compress_rope_freq_base,
                md.context_length,
                device,
                Some(md),
            )?
        } else {
            rope.clone()
        };

        let embedding = if stage.range.is_head() {
            Some(load_embedding(
                Arc::clone(raw(weights, "token_embd.weight")?),
                device,
            )?)
        } else {
            None
        };

        let mut blocks = Vec::with_capacity(stage.range.layer_count());
        for layer in stage.range.start..stage.range.end {
            let prefix = format!("blk.{layer}");
            let layer_rope = if md.compress_ratios.get(layer).copied().unwrap_or(0) == 0 {
                &rope
            } else {
                &compressed_rope
            };
            blocks.push(V4Block {
                attn_norm: norm(weights, &format!("{prefix}.attn_norm"), md, device)?,
                attention: V4Attention::new(weights, &prefix, md, layer, layer_rope, device)?,
                attn_hc: HcMix::new(weights, &format!("{prefix}.hc_attn"), md, device)?,
                ffn_norm: norm(weights, &format!("{prefix}.ffn_norm"), md, device)?,
                moe: V4Moe::new(weights, &prefix, md, layer, device)?,
                ffn_hc: HcMix::new(weights, &format!("{prefix}.hc_ffn"), md, device)?,
            });
        }

        let output_hc = if stage.range.is_tail() {
            Some(HcHead::new(weights, md, device)?)
        } else {
            None
        };
        let output_norm = if stage.range.is_tail() {
            Some(norm(weights, "output_norm", md, device)?)
        } else {
            None
        };
        let lm_head = if stage.range.is_tail() {
            Some(linear(weights, "output.weight", device)?)
        } else {
            None
        };

        let decode_batch_enabled =
            cfg!(all(target_os = "macos", target_arch = "aarch64")) && device.is_metal();
        tracing::info!(
            decode_batch = decode_batch_enabled && stage.range.is_head() && stage.range.is_tail(),
            group_experts = blocks.iter().any(|block| block.moe.group_experts),
            "DeepSeek V4 stage: layers [{}, {}), window/compressed cache, {} experts top-{}, {} HC streams",
            stage.range.start,
            stage.range.end,
            md.expert_count,
            md.expert_used_count,
            md.hyperconnection_count,
        );

        Ok(Self {
            embedding,
            blocks,
            output_norm,
            lm_head,
            output_hc,
            is_head: stage.range.is_head(),
            is_tail: stage.range.is_tail(),
            model_hidden_dim: md.hidden_dim,
            stage_hidden_dim: md.hidden_dim * md.hyperconnection_count,
            stream_count: md.hyperconnection_count,
            num_layers: md.num_layers,
            vocab_size: md.vocab_size,
            context_length: md.context_length,
            decode_batch_enabled,
            device: device.clone(),
        })
    }
}

impl ModelArchitecture for DeepSeek4Stage {
    fn supports_decode_batch(&self) -> bool {
        self.decode_batch_enabled
            && self.is_head
            && self.is_tail
            && (self.device.is_cpu() || self.device.is_metal())
    }

    fn decode_batch_uses_packed_cache(&self) -> bool {
        false
    }

    fn forward_decode_batch(
        &self,
        tokens: &[u32],
        caches: &mut [&mut [Option<LayerCache>]],
        batch_cache: &mut [Option<LayerCache>],
        positions: &[usize],
    ) -> Result<Vec<Tensor>> {
        self.decode_batch(tokens, caches, batch_cache, positions)
    }

    fn prefers_f32_decode_activations(&self) -> bool {
        // Preserve all four residual streams across stage boundaries, as
        // during prefill. Upcasting an F16 packet cannot recover lost bits.
        true
    }

    fn prefers_f32_prefill_activations(&self) -> bool {
        true
    }

    fn forward(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        anyhow::ensure!(
            cache.len() >= self.blocks.len(),
            "V4 cache has too few layers"
        );
        let token_ids = match &input {
            StageInput::Tokens(tokens) => Some(tokens.clone()),
            StageInput::Activations { .. } => None,
        };
        let mut hidden = match input {
            StageInput::Tokens(tokens) => {
                anyhow::ensure!(self.is_head, "only the V4 head stage can receive tokens");
                anyhow::ensure!(
                    tokens.len() == seq_len,
                    "V4 token count differs from sequence length"
                );
                let embedding = self
                    .embedding
                    .as_ref()
                    .context("V4 embedding is not loaded")?;
                embedding
                    .forward(tokens.as_slice(), &self.device)?
                    .reshape((1, seq_len, self.model_hidden_dim))?
                    .unsqueeze(2)?
                    .repeat((1, 1, self.stream_count, 1))?
            }
            StageInput::Activations { tensor, .. } => {
                let tensor = tensor.to_dtype(DType::F32)?;
                match tensor.dims() {
                    [seq, width] => {
                        anyhow::ensure!(*seq == seq_len, "V4 activation sequence length differs");
                        anyhow::ensure!(
                            *width == self.stage_hidden_dim,
                            "V4 activation width differs"
                        );
                        tensor.reshape((1, seq_len, self.stream_count, self.model_hidden_dim))?
                    }
                    [1, seq, streams, width] => {
                        anyhow::ensure!(*seq == seq_len, "V4 activation sequence length differs");
                        anyhow::ensure!(*streams == self.stream_count, "V4 stream count differs");
                        anyhow::ensure!(
                            *width == self.model_hidden_dim,
                            "V4 activation width differs"
                        );
                        tensor
                    }
                    shape => anyhow::bail!("invalid V4 activation shape {shape:?}"),
                }
            }
        };

        for (layer, block) in self.blocks.iter().enumerate() {
            let stamp = profile::start(&self.device, seq_len, Region::AttentionPrepare)?;
            let residual = hidden.clone();
            let (attn_input, attn_post, attn_comb) = block
                .attn_hc
                .pre(&hidden)
                .with_context(|| format!("V4 layer {layer} attention HC pre"))?;
            let attn_input = block
                .attn_norm
                .forward(&attn_input)
                .with_context(|| format!("V4 layer {layer} attention norm"))?;
            profile::finish(&self.device, stamp)?;
            let attn_output = block
                .attention
                .forward(&attn_input, &mut cache[layer], seq_len, start_pos)
                .with_context(|| format!("V4 layer {layer} attention"))?;
            let stamp = profile::start(&self.device, seq_len, Region::AttentionPost)?;
            hidden = block
                .attn_hc
                .post(&attn_output, &residual, &attn_post, &attn_comb)
                .with_context(|| format!("V4 layer {layer} attention HC post"))?;

            profile::finish(&self.device, stamp)?;
            let stamp = profile::start(&self.device, seq_len, Region::FfnPrepare)?;
            let residual = hidden.clone();
            let (ffn_input, ffn_post, ffn_comb) = block
                .ffn_hc
                .pre(&hidden)
                .with_context(|| format!("V4 layer {layer} FFN HC pre"))?;
            let ffn_input = block
                .ffn_norm
                .forward(&ffn_input)
                .with_context(|| format!("V4 layer {layer} FFN norm"))?;
            profile::finish(&self.device, stamp)?;
            let ffn_output = block
                .moe
                .forward(&ffn_input, token_ids.as_deref())
                .with_context(|| format!("V4 layer {layer} MoE"))?;
            let stamp = profile::start(&self.device, seq_len, Region::FfnPost)?;
            hidden = block
                .ffn_hc
                .post(&ffn_output, &residual, &ffn_post, &ffn_comb)
                .with_context(|| format!("V4 layer {layer} FFN HC post"))?;
            profile::finish(&self.device, stamp)?;
        }

        if self.is_tail {
            let stamp = profile::start(&self.device, seq_len, Region::Output)?;
            let output_hc = self
                .output_hc
                .as_ref()
                .context("V4 output HC is not loaded")?;
            let hidden = output_hc.forward(&hidden)?;
            let hidden = self
                .output_norm
                .as_ref()
                .context("V4 output norm is not loaded")?
                .forward(&hidden)?;
            let last = hidden.i((.., seq_len - 1, ..))?;
            let logits = self
                .lm_head
                .as_ref()
                .context("V4 output head is not loaded")?
                .forward(&last)?;
            profile::finish(&self.device, stamp)?;
            Ok(StageOutput::Logits(logits.squeeze(0)?))
        } else {
            Ok(StageOutput::Activations(
                hidden
                    .contiguous()?
                    .reshape((seq_len, self.stage_hidden_dim))?,
            ))
        }
    }

    fn family(&self) -> &str {
        "deepseek4"
    }

    fn hidden_dim(&self) -> usize {
        self.stage_hidden_dim
    }

    fn layer_count(&self) -> usize {
        self.num_layers
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

fn load_embedding(raw: Arc<RawTensor>, device: &Device) -> Result<V4Embedding> {
    if matches!(
        raw.dtype(),
        DeepSeek4DType::F32 | DeepSeek4DType::F16 | DeepSeek4DType::BF16
    ) {
        let dense = raw.to_dense(device)?.to_dtype(DType::F32)?;
        return Ok(V4Embedding::Dense(dense));
    }

    let dtype = match raw.dtype() {
        DeepSeek4DType::Q6K => candle_core::quantized::GgmlDType::Q6K,
        DeepSeek4DType::Q8_0 => candle_core::quantized::GgmlDType::Q8_0,
        DeepSeek4DType::Q4K => candle_core::quantized::GgmlDType::Q4K,
        DeepSeek4DType::Q5K => candle_core::quantized::GgmlDType::Q5K,
        dtype => anyhow::bail!("DeepSeek V4 embedding does not support {dtype:?}"),
    };
    anyhow::ensure!(raw.shape().len() == 2, "V4 embedding must be rank 2");
    Ok(V4Embedding::Quantized {
        hidden_dim: raw.shape()[1],
        raw,
        dtype,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_expert_clamp(device: &Device) -> Result<()> {
        let gate = Tensor::new(&[-20f32, -2., 0., 2., 20.], device)?.reshape((1, 5))?;
        let up = Tensor::new(&[30f32, -30., 1., 30., -30.], device)?.reshape((1, 5))?;
        for limit in [0f32, 3., 10.] {
            let expected = if limit > 0. {
                (gate.clamp(f64::NEG_INFINITY, limit as f64)?.silu()?
                    * up.clamp(-(limit as f64), limit as f64)?)?
            } else {
                (gate.silu()? * &up)?
            };
            let actual = expert_activation(&gate, &up, limit)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for (a, e) in actual.iter().zip(expected.flatten_all()?.to_vec1::<f32>()?) {
                assert!((a - e).abs() <= 2e-5 * e.abs().max(1.), "{a} != {e}");
            }
        }
        Ok(())
    }

    #[test]
    fn expert_clamp_matches_reference_cpu() -> Result<()> {
        check_expert_clamp(&Device::Cpu)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn expert_clamp_matches_reference_metal() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        check_expert_clamp(&device)
    }
    use crate::gguf_loader::LayerRange;

    fn check_hc_source_destination_order(device: &Device) -> Result<()> {
        let mut weights = RawMap::new();
        add_dense(&mut weights, "hc_fn.weight", vec![24, 2048], 0., device)?;
        add_dense(&mut weights, "hc_base.weight", vec![24], 0., device)?;
        add_dense(&mut weights, "hc_scale.weight", vec![3], 1., device)?;
        let mut md = test_metadata();
        md.hyperconnection_sinkhorn_iterations = 1;
        let hc = HcMix::new(&weights, "hc", &md, device)?;
        // Stored rows are source streams, columns are destination streams.
        // Deliberately non-symmetric: equal rows cannot detect a transpose.
        let logits = [
            0.2f32, -0.3, 1.1, 0.7, -1.2, 0.4, 0.8, -0.9, 0.1, 0.5, -0.6, 1.3, -0.7, 0.9, 0.3, -0.2,
        ];
        let mut expected = logits.map(f32::exp);
        for row in expected.chunks_mut(4) {
            let sum: f32 = row.iter().sum();
            for v in row {
                *v = *v / sum + md.hyperconnection_epsilon as f32;
            }
        }
        for dst in 0..4 {
            let sum = (0..4).map(|src| expected[src * 4 + dst]).sum::<f32>()
                + md.hyperconnection_epsilon as f32;
            for src in 0..4 {
                expected[src * 4 + dst] /= sum;
            }
        }
        let comb = hc.sinkhorn(Tensor::from_slice(&logits, (1, 1, 4, 4), device)?)?;
        for (actual, expected) in comb.flatten_all()?.to_vec1::<f32>()?.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 2e-6,
                "Sinkhorn {actual} != {expected}"
            );
        }
        let residual = [1f32, 2., 3., 5., 7., 11., 13., 17.];
        let update = [0.25f32, -0.5];
        let post = [0.2f32, 0.4, 0.8, 1.6];
        let output = hc
            .post(
                &Tensor::from_slice(&update, (1, 1, 2), device)?,
                &Tensor::from_slice(&residual, (1, 1, 4, 2), device)?,
                &Tensor::from_slice(&post, (1, 1, 4), device)?,
                &comb,
            )?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for dst in 0..4 {
            for d in 0..2 {
                let value = post[dst] * update[d]
                    + (0..4)
                        .map(|src| expected[src * 4 + dst] * residual[src * 2 + d])
                        .sum::<f32>();
                assert!(
                    (output[dst * 2 + d] - value).abs() < 2e-5,
                    "HC destination {dst}: {} != {value}",
                    output[dst * 2 + d]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn hc_source_destination_order_cpu() -> Result<()> {
        check_hc_source_destination_order(&Device::Cpu)
    }

    #[test]
    fn v4_stage_rejects_unsupported_cache_and_hash_prefix_splits() -> Result<()> {
        let mut md = test_metadata();
        md.num_layers = 43;
        md.hash_layer_count = 3;
        for range in [
            LayerRange::solo(43),
            LayerRange::new(0, 3, 43),
            LayerRange::new(3, 43, 43),
        ] {
            DeepSeek4Stage::validate_stage_config(&md, &range, false)?;
            assert!(DeepSeek4Stage::validate_stage_config(&md, &range, true).is_err());
        }
        for range in [
            LayerRange::new(0, 2, 43),
            LayerRange::new(2, 43, 43),
            LayerRange::solo(42),
        ] {
            assert!(DeepSeek4Stage::validate_stage_config(&md, &range, false).is_err());
        }
        Ok(())
    }

    #[test]
    fn v4_yarn_uses_compressed_frequencies_only() -> Result<()> {
        let mut md = test_metadata();
        md.rope_scaling_factor = 16.;
        md.rope_scaling_original_context_length = 65_536;
        let (cos, sin, inverse) = precompute_rope(64, 160_000., 2, &Device::Cpu, Some(&md))?;
        let c = cos.to_vec2::<f32>()?;
        let s = sin.to_vec2::<f32>()?;
        let inverse = inverse.to_vec2::<f32>()?;
        // The source's correction range for these metadata values is 15..25.
        let low = (64. * (65_536f64 / (32. * 2. * std::f64::consts::PI)).ln()
            / (2. * 160_000f64.ln()))
        .floor();
        let high =
            (64. * (65_536f64 / (2. * std::f64::consts::PI)).ln() / (2. * 160_000f64.ln())).ceil();
        for i in 0..32 {
            let ramp = ((i as f64 - low) / (high - low)).clamp(0., 1.);
            let angle = 160_000f64.powf(-((2 * i) as f64) / 64.) * (1. - ramp + ramp / 16.);
            assert!((c[1][i] - angle.cos() as f32).abs() < 1e-6);
            assert!((s[1][i] - angle.sin() as f32).abs() < 1e-6);
            assert_eq!(inverse[1][i], -s[1][i]);
        }
        let (_, raw_sin, _) = precompute_rope(64, 160_000., 2, &Device::Cpu, None)?;
        assert!(raw_sin.to_vec2::<f32>()?[1][31] > s[1][31] * 15.);
        md.rope_yarn_beta_slow = 0.;
        assert!(precompute_rope(64, 160_000., 2, &Device::Cpu, Some(&md)).is_err());
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn hc_source_destination_order_metal() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        check_hc_source_destination_order(&device)
    }

    fn dense_raw(shape: Vec<usize>, value: f32, device: &Device) -> Result<Arc<RawTensor>> {
        let bytes = (0..shape.iter().product::<usize>())
            .flat_map(|_| value.to_le_bytes())
            .collect::<Vec<_>>();
        Ok(Arc::new(RawTensor::new(
            shape,
            DeepSeek4DType::F32,
            Tensor::from_vec(bytes.clone(), bytes.len(), device)?,
        )?))
    }

    fn add_dense(
        weights: &mut RawMap,
        name: impl Into<String>,
        shape: Vec<usize>,
        value: f32,
        device: &Device,
    ) -> Result<()> {
        weights.insert(name.into(), dense_raw(shape, value, device)?);
        Ok(())
    }

    #[test]
    fn quantized_embedding_decodes_only_requested_rows() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DeepSeek4DType::Q6K;
        let hidden_dim = 256;
        let byte_len = dtype.byte_len(hidden_dim * 2)?;
        let raw = Arc::new(RawTensor::new(
            vec![2, hidden_dim],
            dtype,
            Tensor::from_vec(vec![0u8; byte_len], byte_len, &device)?,
        )?);
        let embedding = load_embedding(raw, &device)?;
        let rows = embedding.forward(&[1, 0], &device)?;
        anyhow::ensure!(rows.dims() == [2, hidden_dim], "bad embedding shape");
        anyhow::ensure!(
            rows.to_vec2::<f32>()?
                .iter()
                .flatten()
                .all(|value| value.is_finite()),
            "embedding has non-finite values"
        );
        Ok(())
    }

    pub(super) fn test_metadata() -> GgufMetadata {
        GgufMetadata {
            architecture: "deepseek4".into(),
            num_layers: 1,
            hidden_dim: 512,
            num_attention_heads: 1,
            num_kv_heads: 1,
            head_dim: 512,
            intermediate_dim: 8,
            vocab_size: 4,
            rms_norm_eps: 1e-6,
            post_norm_eps: 0.0,
            moe_latent_size: 0,
            logit_scale: 1.0,
            rope_theta: 10_000.0,
            context_length: 8,
            expert_count: 2,
            expert_used_count: 1,
            expert_ffn_dim: 2,
            ssm_state_size: 0,
            ssm_conv_kernel: 0,
            ssm_group_count: 0,
            ssm_time_step_rank: 0,
            ssm_inner_size: 0,
            rope_dimension_count: 64,
            rope_dimension_sections: Vec::new(),
            rope_scaling_factor: 1.0,
            rope_scaling_original_context_length: 8,
            rope_yarn_log_multiplier: 0.0,
            rope_yarn_beta_fast: 32.,
            rope_yarn_beta_slow: 1.,
            rope_dimension_count_swa: 0,
            head_dim_swa: 0,
            rope_theta_swa: 0.0,
            sliding_window: 128,
            sliding_window_pattern: Vec::new(),
            final_logit_softcapping: 0.0,
            full_attention_interval: 0,
            head_count_kv_arr: Vec::new(),
            feed_forward_length_arr: Vec::new(),
            attention_scale: 0.0,
            expert_weights_scale: 1.0,
            expert_weights_norm: true,
            expert_gating_func: 4,
            shared_expert_count: 1,
            kv_lora_rank: 0,
            q_lora_rank: 1,
            qk_nope_head_dim: 448,
            qk_rope_head_dim: 64,
            v_head_dim: 512,
            first_k_dense_replace: 0,
            moe_layer_freq: 1,
            expert_group_count: 1,
            output_lora_rank: 1,
            output_group_count: 1,
            compress_ratios: vec![0],
            indexer_head_count: 0,
            indexer_head_dim: 0,
            indexer_top_k: 0,
            compress_rope_freq_base: 0.0,
            hyperconnection_count: 4,
            hyperconnection_sinkhorn_iterations: 2,
            hyperconnection_epsilon: 1e-6,
            hash_layer_count: 0,
            kda_head_dim: 0,
            kda_gate_lower_bound: f64::NEG_INFINITY,
            indexer_kpool: 0,
            indexer_kpool_select_tail: false,
            swiglu_clamp_exp: Vec::new(),
            swiglu_clamp_shexp: Vec::new(),
        }
    }

    #[test]
    fn dense_v4_stage_runs_through_mhc_attention_and_moe() -> Result<()> {
        let device = Device::Cpu;
        let md = test_metadata();
        let mut weights = RawMap::new();
        add_dense(
            &mut weights,
            "token_embd.weight",
            vec![4, 512],
            0.01,
            &device,
        )?;
        add_dense(&mut weights, "output.weight", vec![4, 512], 0.01, &device)?;
        add_dense(&mut weights, "output_norm.weight", vec![512], 1.0, &device)?;
        add_dense(
            &mut weights,
            "output_hc_fn.weight",
            vec![4, 2048],
            0.01,
            &device,
        )?;
        add_dense(&mut weights, "output_hc_base.weight", vec![4], 0.0, &device)?;
        add_dense(
            &mut weights,
            "output_hc_scale.weight",
            vec![1],
            1.0,
            &device,
        )?;

        let prefix = "blk.0";
        add_dense(
            &mut weights,
            format!("{prefix}.attn_norm.weight"),
            vec![512],
            1.0,
            &device,
        )?;
        add_dense(
            &mut weights,
            format!("{prefix}.attn_q_a.weight"),
            vec![1, 512],
            0.01,
            &device,
        )?;
        add_dense(
            &mut weights,
            format!("{prefix}.attn_q_a_norm.weight"),
            vec![1],
            1.0,
            &device,
        )?;
        add_dense(
            &mut weights,
            format!("{prefix}.attn_q_b.weight"),
            vec![512, 1],
            0.01,
            &device,
        )?;
        add_dense(
            &mut weights,
            format!("{prefix}.attn_kv.weight"),
            vec![512, 512],
            0.01,
            &device,
        )?;
        add_dense(
            &mut weights,
            format!("{prefix}.attn_kv_a_norm.weight"),
            vec![512],
            1.0,
            &device,
        )?;
        add_dense(
            &mut weights,
            format!("{prefix}.attn_output_a.weight"),
            vec![1, 512],
            0.01,
            &device,
        )?;
        add_dense(
            &mut weights,
            format!("{prefix}.attn_output_b.weight"),
            vec![512, 1],
            0.01,
            &device,
        )?;
        add_dense(
            &mut weights,
            format!("{prefix}.attn_sinks.weight"),
            vec![1],
            0.0,
            &device,
        )?;
        add_dense(
            &mut weights,
            format!("{prefix}.hc_attn_fn.weight"),
            vec![24, 2048],
            0.01,
            &device,
        )?;
        add_dense(
            &mut weights,
            format!("{prefix}.hc_attn_base.weight"),
            vec![24],
            0.0,
            &device,
        )?;
        add_dense(
            &mut weights,
            format!("{prefix}.hc_attn_scale.weight"),
            vec![3],
            1.0,
            &device,
        )?;
        add_dense(
            &mut weights,
            format!("{prefix}.ffn_norm.weight"),
            vec![512],
            1.0,
            &device,
        )?;
        add_dense(
            &mut weights,
            format!("{prefix}.hc_ffn_fn.weight"),
            vec![24, 2048],
            0.01,
            &device,
        )?;
        add_dense(
            &mut weights,
            format!("{prefix}.hc_ffn_base.weight"),
            vec![24],
            0.0,
            &device,
        )?;
        add_dense(
            &mut weights,
            format!("{prefix}.hc_ffn_scale.weight"),
            vec![3],
            1.0,
            &device,
        )?;
        add_dense(
            &mut weights,
            format!("{prefix}.ffn_gate_inp.weight"),
            vec![2, 512],
            0.01,
            &device,
        )?;
        add_dense(
            &mut weights,
            format!("{prefix}.ffn_gate_shexp.weight"),
            vec![2, 512],
            0.01,
            &device,
        )?;
        add_dense(
            &mut weights,
            format!("{prefix}.ffn_up_shexp.weight"),
            vec![2, 512],
            0.01,
            &device,
        )?;
        add_dense(
            &mut weights,
            format!("{prefix}.ffn_down_shexp.weight"),
            vec![512, 2],
            0.01,
            &device,
        )?;
        add_dense(
            &mut weights,
            format!("{prefix}.ffn_gate_exps.weight"),
            vec![2, 2, 512],
            0.01,
            &device,
        )?;
        add_dense(
            &mut weights,
            format!("{prefix}.ffn_up_exps.weight"),
            vec![2, 2, 512],
            0.01,
            &device,
        )?;
        add_dense(
            &mut weights,
            format!("{prefix}.ffn_down_exps.weight"),
            vec![2, 512, 2],
            0.01,
            &device,
        )?;

        let stage = StageTensors {
            range: LayerRange::solo(1),
            metadata: md,
            tensors: HashMap::new(),
            raw_tensors: weights,
            raw_source: None,
            device: device.clone(),
        };
        let model = DeepSeek4Stage::from_stage_tensors(stage, None)?;
        assert!(model.prefers_f32_prefill_activations());
        assert!(model.prefers_f32_decode_activations());
        let mut cache = vec![None];
        let output = model.forward(StageInput::Tokens(vec![0, 1]), &mut cache, 2, 0)?;
        let StageOutput::Logits(logits) = output else {
            anyhow::bail!("V4 solo stage did not return logits")
        };
        anyhow::ensure!(logits.dims() == [4], "unexpected V4 test logits shape");
        let values = logits.to_vec1::<f32>()?;
        anyhow::ensure!(
            values.iter().all(|value| value.is_finite()),
            "V4 test logits are not finite"
        );
        Ok(())
    }
}

#[cfg(test)]
mod softplus_reference_tests {
    use super::*;

    #[test]
    fn softplus_keeps_negative_gates() -> Result<()> {
        let values = [-30.0f32, -20., -15., -8.];
        let actual = softplus(&Tensor::new(&values, &Device::Cpu)?)?.to_vec1::<f32>()?;
        for (&x, actual) in values.iter().zip(actual) {
            let expected = f64::from(x).exp().ln_1p();
            let relative = (f64::from(actual) - expected).abs() / expected;
            assert!(
                relative < 2e-6,
                "softplus({x})={actual}, reference={expected}"
            );
        }
        Ok(())
    }
}
