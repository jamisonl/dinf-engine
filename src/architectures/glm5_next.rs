//! Native GLM-5.3 Flash (`glm5next`) text decoder.
//!
//! The model combines Kimi Delta Attention (KDA), no-RoPE latent attention
//! with a pooled sparse index, four mHC residual streams, and sigmoid-routed
//! experts. Vision stays disabled until its projector has a native path.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::qmatmul::QMatMul;
use anyhow::{Context, Result};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{D, DType, Device, IndexOp, Module, Tensor};

#[cfg(target_os = "macos")]
use super::fused_route_metal::fused_sigmoid_topk_route;
#[cfg(not(any(target_os = "macos", feature = "cuda")))]
use super::indexed_moe::silu_mul_clamped;
#[cfg(any(target_os = "macos", feature = "cuda"))]
use super::indexed_moe::{
    IndexedMoeMatMul, indexed_moe_available, normalized_route_weights, silu_mul_clamped,
    weighted_route_sum,
};
use super::{
    DeltaNetCache, KvCache, LayerCache, ModelArchitecture, StageInput, StageOutput,
    kv_cache_update_native, with_kv_cache,
};
use crate::gguf_loader::{GgufMetadata, LayerRange, StageTensorLoader, StageTensors};
use crate::turboquant::TurboQuantConfig;

#[cfg(all(test, target_os = "macos"))]
mod hc_tests;

mod decode_batch;

const GLM5_STREAMS: usize = 4;
const ROUTE_DENOMINATOR_FLOOR: f64 = 6.103_515_625e-5;
/// GLM5's KDA reference uses a literal epsilon for q/k L2 normalization.
/// This is intentionally independent of the model RMSNorm epsilon.
const GLM5_KDA_NORM_EPS: f64 = 1e-6;

fn finish_prefill_work(device: &Device, seq_len: usize) -> Result<()> {
    if seq_len > 1 && device.is_metal() {
        // A near-capacity model cannot retain a whole prompt's retired Metal
        // buffers. Wait after layer locals drop so Candle releases them.
        // Decode keeps its existing asynchronous path.
        device
            .synchronize()
            .context("releasing GLM5 prefill work buffers")?;
    }
    Ok(())
}

fn needs_glm_metal_single_token_gemm(dtype: GgmlDType) -> bool {
    matches!(dtype, GgmlDType::Q2K | GgmlDType::Q3K)
}

#[derive(Clone)]
enum GlmQMatMul {
    Candle(QMatMul),
    /// Candle's Metal Q2_K/Q3_K GEMV path is unstable for GLM decode at
    /// m=1. Duplicate the activation row so Candle dispatches GEMM, then
    /// retain only the first result row.
    CandlePaddedSingleToken(QMatMul),
    #[cfg(target_os = "macos")]
    MetalTensor(super::tensor_qmatmul_metal::MetalTensorQMatMul),
    #[cfg(target_os = "macos")]
    MetalGateRows(super::tensor_qmatmul_metal::MetalTensorQMatMul),
}

impl GlmQMatMul {
    fn forward_decode_rows(&self, x: &Tensor) -> Result<Tensor> {
        anyhow::ensure!(
            matches!(x.rank(), 2 | 3),
            "GLM5 decode projection requires rank two or three"
        );
        let width = x.dim(D::Minus1)?;
        anyhow::ensure!(
            width > 0 && x.elem_count() > 0,
            "GLM5 decode projection cannot be empty"
        );
        #[cfg(target_os = "macos")]
        if let Self::MetalTensor(matmul) | Self::MetalGateRows(matmul) = self {
            return matmul.forward_decode_rows(x).map_err(Into::into);
        }
        if let Self::CandlePaddedSingleToken(_) = self {
            // Scalar decode already uses GEMM with a duplicate row. Independent
            // rows use that same matrix kernel, without duplicate-row work.
            let x = x.contiguous()?;
            // Candle's quantized matrix path starts from storage byte zero.
            // A contiguous narrow can still have a nonzero storage offset.
            return self.forward(&if x.layout().start_offset() != 0 {
                x.force_contiguous()?
            } else {
                x
            });
        }
        // Dense routers keep their scalar sum order.
        let rows = x.elem_count() / width;
        let flat = x.contiguous()?.reshape((rows, width))?;
        let outputs = (0..rows)
            .map(|row| self.forward(&flat.narrow(0, row, 1)?))
            .collect::<Result<Vec<_>>>()?;
        let mut shape = x.dims().to_vec();
        *shape.last_mut().unwrap() = outputs[0].dim(D::Minus1)?;
        Ok(Tensor::cat(&outputs, 0)?.reshape(shape)?)
    }

    fn from_qtensor(qt: Arc<QTensor>, device: &Device) -> Result<Self> {
        let pad_single_token = device.is_metal() && needs_glm_metal_single_token_gemm(qt.dtype());
        #[cfg(target_os = "macos")]
        if let Some(matmul) = super::tensor_qmatmul_metal::MetalTensorQMatMul::try_from_qtensor(
            Arc::clone(&qt),
            device,
        )? {
            if qt.dtype() == GgmlDType::Q4_0 && qt.shape().dims().last() == Some(&128) {
                // GLM's 128-wide KDA gate projections feed its recurrent
                // state. Keep the same F32 row calculation for every prompt
                // length instead of switching to half-input tensor GEMM.
                return Ok(Self::MetalGateRows(matmul));
            }
            return Ok(Self::MetalTensor(matmul));
        }
        let matmul = QMatMul::from_arc(qt)?;
        if pad_single_token {
            Ok(Self::CandlePaddedSingleToken(matmul))
        } else {
            Ok(Self::Candle(matmul))
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Candle(matmul) => matmul.forward(x).map_err(Into::into),
            Self::CandlePaddedSingleToken(matmul) if x.dim(D::Minus2)? == 1 => {
                let padded = Tensor::cat(&[x, x], D::Minus2)?;
                Ok(matmul.forward(&padded)?.narrow(D::Minus2, 0, 1)?)
            }
            Self::CandlePaddedSingleToken(matmul) => matmul.forward(x).map_err(Into::into),
            #[cfg(target_os = "macos")]
            Self::MetalTensor(matmul) => matmul.forward(x).map_err(Into::into),
            #[cfg(target_os = "macos")]
            Self::MetalGateRows(matmul) => matmul.forward_decode_rows(x).map_err(Into::into),
        }
    }
}

fn qmatmul(
    tensors: &mut HashMap<String, Arc<QTensor>>,
    name: &str,
    device: &Device,
) -> Result<GlmQMatMul> {
    let tensor = tensors
        .remove(name)
        .with_context(|| format!("missing GLM5 tensor {name}"))?;
    GlmQMatMul::from_qtensor(tensor, device)
}

fn qmatmul_with_alias(
    tensors: &mut HashMap<String, Arc<QTensor>>,
    canonical_name: &str,
    legacy_name: &str,
    device: &Device,
) -> Result<GlmQMatMul> {
    let name = tensor_name_with_alias(tensors, canonical_name, legacy_name)?;
    qmatmul(tensors, name, device)
}

fn dense_tensor(
    tensors: &mut HashMap<String, Arc<QTensor>>,
    name: &str,
    device: &Device,
) -> Result<Tensor> {
    tensors
        .remove(name)
        .with_context(|| format!("missing GLM5 tensor {name}"))?
        .dequantize(device)
        .map_err(Into::into)
}

fn dense_tensor_with_alias(
    tensors: &mut HashMap<String, Arc<QTensor>>,
    canonical_name: &str,
    legacy_name: &str,
    device: &Device,
) -> Result<Tensor> {
    let name = tensor_name_with_alias(tensors, canonical_name, legacy_name)?;
    dense_tensor(tensors, name, device)
}

fn tensor_name_with_alias<'a>(
    tensors: &HashMap<String, Arc<QTensor>>,
    canonical_name: &'a str,
    legacy_name: &'a str,
) -> Result<&'a str> {
    choose_tensor_name(
        tensors.contains_key(canonical_name),
        tensors.contains_key(legacy_name),
        canonical_name,
        legacy_name,
    )
}

fn choose_tensor_name<'a>(
    canonical_present: bool,
    legacy_present: bool,
    canonical_name: &'a str,
    legacy_name: &'a str,
) -> Result<&'a str> {
    if canonical_present {
        anyhow::ensure!(
            !legacy_present,
            "GLM5 tensor has both canonical and legacy names: {canonical_name}"
        );
        Ok(canonical_name)
    } else {
        Ok(legacy_name)
    }
}

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(
        tensors: &mut HashMap<String, Arc<QTensor>>,
        name: &str,
        eps: f64,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            weight: dense_tensor(tensors, name, device)?,
            eps,
        })
    }

    fn ones(width: usize, eps: f64, device: &Device) -> Result<Self> {
        Ok(Self {
            weight: Tensor::ones(width, DType::F32, device)?,
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        #[cfg(target_os = "macos")]
        if let Some(output) =
            super::fused_norm_metal::try_fused_rms_norm(&x, &self.weight, self.eps as f32)?
        {
            return output.to_dtype(dtype).map_err(Into::into);
        }
        let variance = (&x * &x)?.mean_keepdim(D::Minus1)?;
        x.broadcast_div(&(variance + self.eps)?.sqrt()?)?
            .broadcast_mul(&self.weight)?
            .to_dtype(dtype)
            .map_err(Into::into)
    }
}

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.to_dtype(DType::F32)?;
        let mean = x.mean_keepdim(D::Minus1)?;
        let centered = x.broadcast_sub(&mean)?;
        let variance = (&centered * &centered)?.mean_keepdim(D::Minus1)?;
        centered
            .broadcast_div(&(variance + self.eps)?.sqrt()?)?
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)
            .map_err(Into::into)
    }
}

fn sigmoid(x: &Tensor) -> Result<Tensor> {
    candle_nn::Activation::Sigmoid
        .forward(x)
        .map_err(Into::into)
}

fn l2_normalize(x: &Tensor, eps: f64) -> Result<Tensor> {
    let norm = ((x * x)?.sum_keepdim(D::Minus1)? + eps)?.sqrt()?;
    x.broadcast_div(&norm).map_err(Into::into)
}

fn affine(x: &Tensor, scale: &Tensor, base: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(scale)?
        .broadcast_add(base)
        .map_err(Into::into)
}

struct HcMix {
    projection: GlmQMatMul,
    base: Tensor,
    scale: Tensor,
    norm: RmsNorm,
    streams: usize,
    hidden: usize,
    sinkhorn_iterations: usize,
    eps: f64,
    fused_sinkhorn: bool,
    fused_post: bool,
    fused_pre: bool,
    reuse_projection: bool,
}

impl HcMix {
    fn new(
        tensors: &mut HashMap<String, Arc<QTensor>>,
        prefix: &str,
        md: &GgufMetadata,
        device: &Device,
    ) -> Result<Self> {
        let streams = md.hyperconnection_count;
        anyhow::ensure!(streams == GLM5_STREAMS, "GLM5 requires four mHC streams");
        let mix_dim = (2 + streams) * streams;
        let base = dense_tensor(tensors, &format!("{prefix}_base.weight"), device)?;
        let scale = dense_tensor(tensors, &format!("{prefix}_scale.weight"), device)?;
        anyhow::ensure!(base.dims() == [mix_dim], "invalid {prefix}_base shape");
        anyhow::ensure!(scale.dims() == [3], "invalid {prefix}_scale shape");
        let apple_silicon_metal =
            cfg!(all(target_os = "macos", target_arch = "aarch64")) && device.is_metal();
        Ok(Self {
            projection: qmatmul(tensors, &format!("{prefix}_fn.weight"), device)?,
            base,
            scale,
            norm: RmsNorm::ones(streams * md.hidden_dim, md.rms_norm_eps, device)?,
            streams,
            hidden: md.hidden_dim,
            sinkhorn_iterations: md.hyperconnection_sinkhorn_iterations.max(1),
            eps: md.hyperconnection_epsilon.max(1e-6),
            fused_sinkhorn: apple_silicon_metal,
            fused_post: apple_silicon_metal,
            fused_pre: apple_silicon_metal,
            reuse_projection: apple_silicon_metal,
        })
    }

    fn projected_mixes(&self, x: &Tensor) -> Result<(Tensor, usize)> {
        let (_, seq_len, streams, hidden) = x.dims4()?;
        anyhow::ensure!(
            (streams, hidden) == (self.streams, self.hidden),
            "invalid GLM5 mHC input shape"
        );
        let flat = x.reshape((1, seq_len, streams * hidden))?;
        let mixes = self.projection.forward(&self.norm.forward(&flat)?)?;
        Ok((mixes, seq_len))
    }

    fn update_from_mixes(&self, x: &Tensor, mixes: &Tensor) -> Result<Tensor> {
        super::hc_pre::forward(x, mixes, &self.scale, &self.base, self.eps, self.fused_pre)
    }

    fn post_controls_from_mixes(&self, mixes: &Tensor, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let scale_post = self.scale.narrow(0, 1, 1)?.reshape((1, 1, 1))?;
        let scale_comb = self.scale.narrow(0, 2, 1)?.reshape((1, 1, 1))?;
        let base_post =
            self.base
                .narrow(0, self.streams, self.streams)?
                .reshape((1, 1, self.streams))?;
        let base_comb = self
            .base
            .narrow(0, 2 * self.streams, self.streams * self.streams)?
            .reshape((1, 1, self.streams * self.streams))?;
        let post = sigmoid(&affine(
            &mixes.narrow(D::Minus1, self.streams, self.streams)?,
            &scale_post,
            &base_post,
        )?)?
        .affine(2.0, 0.0)?;
        let combination = affine(
            &mixes.narrow(D::Minus1, 2 * self.streams, self.streams * self.streams)?,
            &scale_comb,
            &base_comb,
        )?
        .reshape((1, seq_len, self.streams, self.streams))?;
        let combination = self.sinkhorn(combination)?;
        Ok((post, combination))
    }

    fn pre(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (mixes, seq_len) = self.projected_mixes(x)?;
        let update = self.update_from_mixes(x, &mixes)?;
        let (post, combination) = self.post_controls_from_mixes(&mixes, seq_len)?;
        Ok((update, post, combination))
    }

    fn pre_update(&self, x: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let (mixes, _) = self.projected_mixes(x)?;
        let update = self.update_from_mixes(x, &mixes)?;
        Ok((update, self.reuse_projection.then_some(mixes)))
    }

    fn post_controls(&self, x: &Tensor, mixes: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        if let Some(mixes) = mixes {
            return self.post_controls_from_mixes(mixes, x.dim(1)?);
        }
        let (mixes, seq_len) = self.projected_mixes(x)?;
        self.post_controls_from_mixes(&mixes, seq_len)
    }

    fn post(
        &self,
        update: &Tensor,
        residual: &Tensor,
        post: &Tensor,
        combination: &Tensor,
    ) -> Result<Tensor> {
        super::hc_post::forward(update, residual, post, combination, self.fused_post)
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

struct DenseMlp {
    gate: GlmQMatMul,
    up: GlmQMatMul,
    down: GlmQMatMul,
    clamp: f32,
}

impl DenseMlp {
    fn forward_decode_rows(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self
            .gate
            .forward_decode_rows(x)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let up = self
            .up
            .forward_decode_rows(x)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        self.down
            .forward_decode_rows(&silu_mul_clamped(&gate, &up, self.clamp)?)
    }

    fn new(
        tensors: &mut HashMap<String, Arc<QTensor>>,
        prefix: &str,
        clamp: f32,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            gate: qmatmul(tensors, &format!("{prefix}.ffn_gate.weight"), device)?,
            up: qmatmul(tensors, &format!("{prefix}.ffn_up.weight"), device)?,
            down: qmatmul(tensors, &format!("{prefix}.ffn_down.weight"), device)?,
            clamp,
        })
    }

    fn shared(
        tensors: &mut HashMap<String, Arc<QTensor>>,
        prefix: &str,
        clamp: f32,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            gate: qmatmul(tensors, &format!("{prefix}.ffn_gate_shexp.weight"), device)?,
            up: qmatmul(tensors, &format!("{prefix}.ffn_up_shexp.weight"), device)?,
            down: qmatmul(tensors, &format!("{prefix}.ffn_down_shexp.weight"), device)?,
            clamp,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(x)?.to_dtype(DType::F32)?.contiguous()?;
        let up = self.up.forward(x)?.to_dtype(DType::F32)?.contiguous()?;
        self.down
            .forward(&silu_mul_clamped(&gate, &up, self.clamp)?)
    }
}

fn conv_weight(
    tensors: &mut HashMap<String, Arc<QTensor>>,
    name: &str,
    inner: usize,
    kernel: usize,
    device: &Device,
) -> Result<Tensor> {
    let tensor = dense_tensor(tensors, name, device)?;
    match tensor.dims() {
        [rows, 1, width] if *rows == inner && *width == kernel => {
            tensor.squeeze(1).map_err(Into::into)
        }
        [rows, width] if *rows == inner && *width == kernel => Ok(tensor),
        shape => anyhow::bail!("invalid GLM5 convolution tensor {name}: {shape:?}"),
    }
}

fn conv1d_step(state: &Tensor, input: &Tensor, weight: &Tensor) -> Result<(Tensor, Tensor)> {
    let (_, kernel) = state.dims2()?;
    let shifted = if kernel > 1 {
        Tensor::cat(&[&state.narrow(1, 1, kernel - 1)?, &input.unsqueeze(1)?], 1)?
    } else {
        input.unsqueeze(1)?
    };
    let output = (&shifted * weight)?.sum(1)?;
    Ok((candle_nn::Activation::Silu.forward(&output)?, shifted))
}

struct KdaAttention {
    q: GlmQMatMul,
    k: GlmQMatMul,
    v: GlmQMatMul,
    out: GlmQMatMul,
    f_a: GlmQMatMul,
    f_b: GlmQMatMul,
    beta: GlmQMatMul,
    a: Tensor,
    dt_bias: Tensor,
    g_a: GlmQMatMul,
    g_b: GlmQMatMul,
    output_norm: RmsNorm,
    conv_weight: Tensor,
    heads: usize,
    head_dim: usize,
    inner: usize,
    kernel: usize,
    gate_lower_bound: f32,
    norm_eps: f64,
}

impl KdaAttention {
    fn new(
        tensors: &mut HashMap<String, Arc<QTensor>>,
        prefix: &str,
        md: &GgufMetadata,
        device: &Device,
    ) -> Result<Self> {
        let heads = md.num_attention_heads;
        let head_dim = md.kda_head_dim;
        let inner = heads * head_dim;
        let kernel = md.ssm_conv_kernel;
        anyhow::ensure!(
            heads > 0 && head_dim > 0 && kernel > 0,
            "GLM5 KDA metadata is incomplete"
        );
        anyhow::ensure!(
            md.kda_gate_lower_bound.is_finite() && md.kda_gate_lower_bound < 0.0,
            "GLM5 KDA requires a finite negative gate lower bound"
        );
        let conv_q = conv_weight(
            tensors,
            &format!("{prefix}.ssm_conv1d_q.weight"),
            inner,
            kernel,
            device,
        )?;
        let conv_k = conv_weight(
            tensors,
            &format!("{prefix}.ssm_conv1d_k.weight"),
            inner,
            kernel,
            device,
        )?;
        let conv_v = conv_weight(
            tensors,
            &format!("{prefix}.ssm_conv1d_v.weight"),
            inner,
            kernel,
            device,
        )?;
        Ok(Self {
            q: qmatmul(tensors, &format!("{prefix}.attn_q.weight"), device)?,
            k: qmatmul(tensors, &format!("{prefix}.attn_k.weight"), device)?,
            v: qmatmul(tensors, &format!("{prefix}.attn_v.weight"), device)?,
            out: qmatmul(tensors, &format!("{prefix}.attn_output.weight"), device)?,
            f_a: qmatmul(tensors, &format!("{prefix}.ssm_f_a.weight"), device)?,
            f_b: qmatmul(tensors, &format!("{prefix}.ssm_f_b.weight"), device)?,
            beta: qmatmul(tensors, &format!("{prefix}.ssm_beta.weight"), device)?,
            a: dense_tensor(tensors, &format!("{prefix}.ssm_a"), device)?,
            dt_bias: dense_tensor(tensors, &format!("{prefix}.ssm_dt.bias"), device)?,
            g_a: qmatmul(tensors, &format!("{prefix}.ssm_g_a.weight"), device)?,
            g_b: qmatmul(tensors, &format!("{prefix}.ssm_g_b.weight"), device)?,
            output_norm: RmsNorm::new(
                tensors,
                &format!("{prefix}.ssm_norm.weight"),
                md.rms_norm_eps,
                device,
            )?,
            conv_weight: Tensor::cat(&[&conv_q, &conv_k, &conv_v], 0)?.contiguous()?,
            heads,
            head_dim,
            inner,
            kernel,
            gate_lower_bound: md.kda_gate_lower_bound as f32,
            norm_eps: GLM5_KDA_NORM_EPS,
        })
    }

    fn forward(&self, x: &Tensor, cache: &mut Option<DeltaNetCache>) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        anyhow::ensure!(batch == 1, "GLM5 KDA currently requires batch size one");
        let q = self.q.forward(x)?.to_dtype(DType::F32)?;
        let k = self.k.forward(x)?.to_dtype(DType::F32)?;
        let v = self.v.forward(x)?.to_dtype(DType::F32)?;

        let decay_input = self
            .f_b
            .forward(&self.f_a.forward(x)?)?
            .to_dtype(DType::F32)?
            .broadcast_add(&self.dt_bias)?
            .reshape((seq_len, self.heads, self.head_dim))?;
        let a = self.a.reshape((1, self.heads, 1))?;
        let decay_exp = sigmoid(&decay_input.broadcast_mul(&a)?.neg()?)?
            .affine(self.gate_lower_bound as f64, 0.0)?
            .contiguous()?;
        let beta_raw = self
            .beta
            .forward(x)?
            .to_dtype(DType::F32)?
            .reshape((seq_len, self.heads))?
            .contiguous()?;

        let (conv_state, recurrent_state) = match cache.take() {
            Some(cache) => (cache.conv_state, cache.recurrent_state),
            None => (
                Tensor::zeros((3 * self.inner, self.kernel), DType::F32, x.device())?,
                Tensor::zeros(
                    (self.heads, self.head_dim, self.head_dim),
                    DType::F32,
                    x.device(),
                )?,
            ),
        };

        #[cfg(target_os = "macos")]
        let accelerated = {
            let q = q.squeeze(0)?;
            let k = k.squeeze(0)?;
            let v = v.squeeze(0)?;
            if x.device().is_metal() && q.is_contiguous() && k.is_contiguous() && v.is_contiguous()
            {
                Some(super::glm5_next_kda_metal::scan(
                    &q,
                    &k,
                    &v,
                    &conv_state,
                    &recurrent_state,
                    &self.conv_weight,
                    &decay_exp,
                    &beta_raw,
                    self.heads,
                    self.head_dim,
                    self.kernel,
                    self.norm_eps as f32,
                )?)
            } else {
                None
            }
        };
        #[cfg(not(target_os = "macos"))]
        let accelerated: Option<(Tensor, Tensor, Tensor)> = None;

        let (recurrent_output, conv_state, recurrent_state) = if let Some(result) = accelerated {
            result
        } else {
            let qkv = Tensor::cat(&[&q, &k, &v], D::Minus1)?
                .squeeze(0)?
                .contiguous()?;
            self.portable_scan(&qkv, conv_state, recurrent_state, &decay_exp, &beta_raw)?
        };
        *cache = Some(DeltaNetCache {
            conv_state,
            recurrent_state,
            speculative_state: None,
        });

        let output = self
            .output_norm
            .forward(&recurrent_output)?
            .reshape((1, seq_len, self.inner))?;
        let gate = sigmoid(
            &self
                .g_b
                .forward(&self.g_a.forward(x)?)?
                .to_dtype(DType::F32)?,
        )?;
        self.out.forward(&(output * gate)?)
    }

    fn portable_scan(
        &self,
        qkv: &Tensor,
        mut conv_state: Tensor,
        mut recurrent_state: Tensor,
        decay_exp: &Tensor,
        beta_raw: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let seq_len = qkv.dim(0)?;
        let mut outputs = Vec::with_capacity(seq_len);
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        for token in 0..seq_len {
            let (conv, next_conv) = conv1d_step(
                &conv_state,
                &qkv.get(token)?.contiguous()?,
                &self.conv_weight,
            )?;
            conv_state = next_conv;
            let q = l2_normalize(
                &conv
                    .narrow(0, 0, self.inner)?
                    .reshape((self.heads, self.head_dim))?,
                self.norm_eps,
            )?
            .affine(scale, 0.0)?;
            let k = l2_normalize(
                &conv
                    .narrow(0, self.inner, self.inner)?
                    .reshape((self.heads, self.head_dim))?,
                self.norm_eps,
            )?;
            let v = conv
                .narrow(0, 2 * self.inner, self.inner)?
                .reshape((self.heads, self.head_dim))?;
            let decay = decay_exp.get(token)?.exp()?.unsqueeze(2)?;
            let decayed = recurrent_state.broadcast_mul(&decay)?;
            let retrieved = decayed.broadcast_mul(&k.unsqueeze(2)?)?.sum(1)?;
            let beta = sigmoid(&beta_raw.get(token)?)?.unsqueeze(1)?;
            let delta = (v - retrieved)?.broadcast_mul(&beta)?;
            recurrent_state = (decayed + k.unsqueeze(2)?.broadcast_mul(&delta.unsqueeze(1)?)?)?;
            let output = recurrent_state.broadcast_mul(&q.unsqueeze(2)?)?.sum(1)?;
            outputs.push(output.unsqueeze(0)?);
        }
        Ok((Tensor::cat(&outputs, 0)?, conv_state, recurrent_state))
    }
}

enum HeadwiseMatMul {
    #[cfg(any(target_os = "macos", feature = "cuda"))]
    Indexed(IndexedMoeMatMul),
    Dense(Tensor),
}

impl HeadwiseMatMul {
    fn new(
        packed: Arc<QTensor>,
        heads: usize,
        out_dim: usize,
        in_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        anyhow::ensure!(
            packed.shape().dims() == [heads, out_dim, in_dim],
            "GLM5 headwise tensor shape {:?} differs from [{heads}, {out_dim}, {in_dim}]",
            packed.shape()
        );
        #[cfg(any(target_os = "macos", feature = "cuda"))]
        if indexed_moe_available(device, packed.dtype()) {
            return Ok(Self::Indexed(IndexedMoeMatMul::from_qtensor(
                &packed, heads, out_dim, in_dim, device,
            )?));
        }
        Ok(Self::Dense(packed.dequantize(device)?))
    }

    fn forward(&self, x: &Tensor, _prefill: bool) -> Result<Tensor> {
        let (_tokens, heads, _) = x.dims3()?;
        match self {
            #[cfg(any(target_os = "macos", feature = "cuda"))]
            Self::Indexed(matmul) => {
                let ids = Tensor::arange(0u32, heads as u32, x.device())?
                    .reshape((1, heads))?
                    .repeat((_tokens, 1))?;
                let x = x.to_dtype(DType::F32)?.contiguous()?;
                if _prefill {
                    matmul.forward_prefill(&x, &ids)
                } else {
                    matmul.forward(&x, &ids)
                }
            }
            Self::Dense(weights) => {
                let mut outputs = Vec::with_capacity(heads);
                for head in 0..heads {
                    let input = x.narrow(1, head, 1)?.squeeze(1)?;
                    let weight = weights.get(head)?;
                    outputs.push(input.matmul(&weight.t()?)?.unsqueeze(1)?);
                }
                Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 1).map_err(Into::into)
            }
        }
    }
}

fn causal_mask(seq_len: usize, total: usize, device: &Device) -> Result<Tensor> {
    let first_query = total.saturating_sub(seq_len);
    let values = (0..seq_len)
        .flat_map(|query| {
            let position = first_query + query;
            (0..total).map(move |key| {
                if key <= position {
                    0.0f32
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (1, 1, seq_len, total), device).map_err(Into::into)
}

struct DsaAttention {
    q_a: GlmQMatMul,
    q_a_norm: RmsNorm,
    q_b: GlmQMatMul,
    kv_a: GlmQMatMul,
    kv_a_norm: RmsNorm,
    k_b: HeadwiseMatMul,
    v_b: HeadwiseMatMul,
    out: GlmQMatMul,
    index_q: GlmQMatMul,
    index_k: GlmQMatMul,
    index_k_norm: LayerNorm,
    index_weight: GlmQMatMul,
    pool_gate: GlmQMatMul,
    pool_ape: Tensor,
    heads: usize,
    q_head_dim: usize,
    value_head_dim: usize,
    q_rank: usize,
    kv_rank: usize,
    index_heads: usize,
    index_dim: usize,
    index_top_k: usize,
    kpool: usize,
}

impl DsaAttention {
    fn new(
        tensors: &mut HashMap<String, Arc<QTensor>>,
        prefix: &str,
        md: &GgufMetadata,
        device: &Device,
    ) -> Result<Self> {
        let heads = md.num_attention_heads;
        let q_head_dim = md.qk_nope_head_dim;
        let value_head_dim = md.v_head_dim;
        let q_rank = md.q_lora_rank;
        let kv_rank = md.kv_lora_rank;
        let index_heads = md.indexer_head_count;
        let index_dim = md.indexer_head_dim;
        let index_top_k = md.indexer_top_k;
        let kpool = md.indexer_kpool;
        anyhow::ensure!(
            heads > 0
                && q_head_dim > 0
                && value_head_dim > 0
                && q_rank > 0
                && kv_rank > 0
                && index_heads > 0
                && index_dim > 0
                && index_top_k > 0
                && kpool > 1
                && index_top_k.is_multiple_of(kpool),
            "GLM5 DSA metadata is incomplete"
        );
        anyhow::ensure!(
            md.qk_rope_head_dim == 0,
            "GLM5 DSA only supports no-RoPE latent attention"
        );
        anyhow::ensure!(
            md.indexer_kpool_select_tail,
            "GLM5 DSA requires tail-preserving pooled selection"
        );
        let k_b_name = format!("{prefix}.attn_k_b.weight");
        let v_b_name = format!("{prefix}.attn_v_b.weight");
        let k_b = tensors
            .remove(&k_b_name)
            .with_context(|| format!("missing GLM5 tensor {k_b_name}"))?;
        let v_b = tensors
            .remove(&v_b_name)
            .with_context(|| format!("missing GLM5 tensor {v_b_name}"))?;

        Ok(Self {
            q_a: qmatmul(tensors, &format!("{prefix}.attn_q_a.weight"), device)?,
            q_a_norm: RmsNorm::new(
                tensors,
                &format!("{prefix}.attn_q_a_norm.weight"),
                md.rms_norm_eps,
                device,
            )?,
            q_b: qmatmul(tensors, &format!("{prefix}.attn_q_b.weight"), device)?,
            kv_a: qmatmul(tensors, &format!("{prefix}.attn_kv_a_mqa.weight"), device)?,
            kv_a_norm: RmsNorm::new(
                tensors,
                &format!("{prefix}.attn_kv_a_norm.weight"),
                md.rms_norm_eps,
                device,
            )?,
            k_b: HeadwiseMatMul::new(k_b, heads, kv_rank, q_head_dim, device)?,
            v_b: HeadwiseMatMul::new(v_b, heads, value_head_dim, kv_rank, device)?,
            out: qmatmul(tensors, &format!("{prefix}.attn_output.weight"), device)?,
            index_q: qmatmul(
                tensors,
                &format!("{prefix}.indexer.attn_q_b.weight"),
                device,
            )?,
            index_k: qmatmul(tensors, &format!("{prefix}.indexer.attn_k.weight"), device)?,
            index_k_norm: LayerNorm {
                weight: dense_tensor(tensors, &format!("{prefix}.indexer.k_norm.weight"), device)?,
                bias: dense_tensor(tensors, &format!("{prefix}.indexer.k_norm.bias"), device)?,
                eps: 1e-6,
            },
            index_weight: qmatmul(tensors, &format!("{prefix}.indexer.proj.weight"), device)?,
            pool_gate: qmatmul_with_alias(
                tensors,
                &format!("{prefix}.indexer_compressor_gate.weight"),
                &format!("{prefix}.indexer_compressor_gate"),
                device,
            )?,
            pool_ape: dense_tensor_with_alias(
                tensors,
                &format!("{prefix}.indexer_compressor_ape.weight"),
                &format!("{prefix}.indexer_compressor_ape"),
                device,
            )?,
            heads,
            q_head_dim,
            value_head_dim,
            q_rank,
            kv_rank,
            index_heads,
            index_dim,
            index_top_k,
            kpool,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        cache: &mut Option<KvCache>,
        seq_len: usize,
        prefill: bool,
    ) -> Result<Tensor> {
        let (batch, actual_seq, _) = x.dims3()?;
        anyhow::ensure!(
            batch == 1 && actual_seq == seq_len,
            "GLM5 DSA currently requires one sequence"
        );
        let q_residual = self.q_a_norm.forward(&self.q_a.forward(x)?)?;
        anyhow::ensure!(
            q_residual.dim(D::Minus1)? == self.q_rank,
            "GLM5 DSA query rank differs"
        );
        let q = self
            .q_b
            .forward(&q_residual)?
            .to_dtype(DType::F32)?
            .reshape((seq_len, self.heads, self.q_head_dim))?;
        let q_absorbed = self
            .k_b
            .forward(&q, prefill)?
            .reshape((1, seq_len, self.heads, self.kv_rank))?
            .transpose(1, 2)?
            .contiguous()?;

        let latent = self
            .kv_a_norm
            .forward(&self.kv_a.forward(x)?)?
            .to_dtype(DType::F32)?;
        anyhow::ensure!(
            latent.dim(D::Minus1)? == self.kv_rank,
            "GLM5 DSA latent rank differs"
        );
        let index_key = self.index_k_norm.forward(&self.index_k.forward(x)?)?;
        let pool_gate = self.pool_gate.forward(x)?.to_dtype(DType::F32)?;
        let packed_index = Tensor::cat(&[&index_key, &pool_gate], D::Minus1)?;
        let (cached_latent, cached_index) = kv_cache_update_native(
            cache,
            latent.unsqueeze(1)?,
            packed_index.unsqueeze(1)?,
            None,
        )?;
        let total = cached_latent.dim(2)?;
        let scale = 1.0 / (self.q_head_dim as f64).sqrt();

        let latent_context = if total <= self.index_top_k {
            // Do not use the generic fused F16 SDPA here.  It casts the
            // absorbed MLA query to F16, while GLM5's reference keeps this
            // latent attention in F32 (`-fa off` in llama.cpp).  The latent
            // cache itself remains F16, but is upcast before score/value math.
            dense_latent_attention_f32(
                &q_absorbed,
                &cached_latent,
                seq_len,
                total,
                scale,
                x.device(),
                prefill,
            )?
        } else {
            let mask = self.selection_mask(x, &q_residual, &cached_index, seq_len, total)?;
            let latent = cached_latent.to_dtype(DType::F32)?;
            let scores = (shared_latent_matmul(&q_absorbed, &latent.transpose(2, 3)?)? * scale)?;
            let scores = scores.broadcast_add(&mask)?;
            let scores = candle_nn::ops::softmax_last_dim(&scores)?;
            latent_context(&scores, &latent, prefill)?
        };

        let values = latent_context.transpose(1, 2)?.contiguous()?.reshape((
            seq_len,
            self.heads,
            self.kv_rank,
        ))?;
        let values = self.v_b.forward(&values, prefill)?.reshape((
            1,
            seq_len,
            self.heads * self.value_head_dim,
        ))?;
        self.out.forward(&values)
    }

    fn selection_mask(
        &self,
        x: &Tensor,
        q_residual: &Tensor,
        cached_index: &Tensor,
        seq_len: usize,
        total: usize,
    ) -> Result<Tensor> {
        let complete_pools = total / self.kpool;
        anyhow::ensure!(
            complete_pools > 0,
            "GLM5 sparse index has no complete pools"
        );
        let packed = cached_index.to_dtype(DType::F32)?.squeeze(0)?.squeeze(0)?;
        let keys = packed.narrow(D::Minus1, 0, self.index_dim)?;
        let gates = packed.narrow(D::Minus1, self.index_dim, self.index_dim)?;
        let pooled_rows = complete_pools * self.kpool;
        let keys = keys.narrow(0, 0, pooled_rows)?.reshape((
            complete_pools,
            self.kpool,
            self.index_dim,
        ))?;
        let gates = gates.narrow(0, 0, pooled_rows)?.reshape((
            complete_pools,
            self.kpool,
            self.index_dim,
        ))?;
        anyhow::ensure!(
            self.pool_ape.dims() == [self.kpool, self.index_dim],
            "GLM5 pool position tensor shape differs"
        );
        let probabilities =
            candle_nn::ops::softmax(&gates.broadcast_add(&self.pool_ape.unsqueeze(0)?)?, 1)?;
        let pooled = probabilities.broadcast_mul(&keys)?.sum(1)?;

        let index_q = self
            .index_q
            .forward(q_residual)?
            .to_dtype(DType::F32)?
            .reshape((seq_len, self.index_heads, self.index_dim))?;
        let weights = self
            .index_weight
            .forward(x)?
            .to_dtype(DType::F32)?
            .squeeze(0)?
            .affine(
                1.0 / ((self.index_heads * self.index_dim) as f64).sqrt(),
                0.0,
            )?;
        let pool_scores = pooled_index_matmul(&index_q, &pooled)?
            .relu()?
            .broadcast_mul(&weights.unsqueeze(2)?)?
            .sum(1)?;

        let first_query = total - seq_len;
        let valid_pool_mask = (0..seq_len)
            .flat_map(|query| {
                let position = first_query + query;
                (0..complete_pools).map(move |pool| {
                    if pool * self.kpool + self.kpool - 1 <= position {
                        0.0f32
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect::<Vec<_>>();
        let valid_pool_mask =
            Tensor::from_vec(valid_pool_mask, (seq_len, complete_pools), x.device())?;
        let pool_scores = pool_scores.broadcast_add(&valid_pool_mask)?;
        let top_pools = complete_pools.min(self.index_top_k / self.kpool);
        let pool_scores = pool_scores.contiguous()?;
        #[cfg(target_os = "macos")]
        let wide_indices = super::glm5_next_sort_metal::descending_indices(&pool_scores)?;
        #[cfg(not(target_os = "macos"))]
        let wide_indices: Option<Tensor> = None;
        let selected_ids = match wide_indices {
            Some(indices) => indices,
            None => pool_scores.arg_sort_last_dim(false)?,
        }
        .narrow(D::Minus1, 0, top_pools)?
        .contiguous()?;
        let pool_selected = Tensor::zeros((seq_len, complete_pools), DType::F32, x.device())?;
        let ones = Tensor::ones((seq_len, top_pools), DType::F32, x.device())?;
        pool_selected.scatter_set(&selected_ids, &ones, D::Minus1)?;
        let mut token_selected = pool_selected
            .unsqueeze(2)?
            .repeat((1, 1, self.kpool))?
            .reshape((seq_len, pooled_rows))?;
        if pooled_rows < total {
            token_selected = Tensor::cat(
                &[
                    &token_selected,
                    &Tensor::zeros((seq_len, total - pooled_rows), DType::F32, x.device())?,
                ],
                D::Minus1,
            )?;
        }

        // Every query also keeps its own incomplete pool tail. This is needed
        // when a long prefill turns that pool into a complete pool for later
        // queries in the same call.
        let tail = (0..seq_len)
            .flat_map(|query| {
                let position = first_query + query;
                let count = (position + 1) % self.kpool;
                let start = position + 1 - count;
                (0..total).map(move |key| {
                    if count > 0 && key >= start && key <= position {
                        1.0f32
                    } else {
                        0.0f32
                    }
                })
            })
            .collect::<Vec<_>>();
        let tail = Tensor::from_vec(tail, (seq_len, total), x.device())?;
        let selected = (token_selected + tail)?;
        let zeros = Tensor::zeros((seq_len, total), DType::F32, x.device())?;
        let negative = Tensor::full(f32::NEG_INFINITY, (seq_len, total), x.device())?;
        let selected_mask = selected.gt(&zeros)?.where_cond(&zeros, &negative)?;
        let selected_mask = selected_mask.reshape((1, 1, seq_len, total))?;
        selected_mask
            .broadcast_add(&causal_mask(seq_len, total, x.device())?)
            .map_err(Into::into)
    }
}

/// Score each query head against the same pooled index keys, without copying
/// the key matrix for every query. Candle matmul does not broadcast 3D by 2D.
fn pooled_index_matmul(queries: &Tensor, pooled: &Tensor) -> Result<Tensor> {
    let (rows, heads, dim) = queries.dims3()?;
    let (pools, key_dim) = pooled.dims2()?;
    anyhow::ensure!(dim == key_dim, "GLM5 pooled index key width differs");
    let flat_rows = rows
        .checked_mul(heads)
        .context("GLM5 index size overflow")?;
    queries
        .reshape((flat_rows, dim))?
        .matmul(&pooled.t()?)?
        .reshape((rows, heads, pools))
        .map_err(Into::into)
}

/// Multiply a per-head query by one latent cache shared by all heads.
///
/// Candle's regular `matmul` requires identical batch dimensions, while DSA
/// naturally represents these tensors as `[batch, heads, rows, rank]` and
/// `[1, 1, rank, columns]`. `broadcast_matmul` fixes the shape error by
/// materializing one copy of the latent cache per head. Flattening the query's
/// batch and head axes instead keeps a single cache matrix and performs the
/// same independent matmuls in one operation.
fn shared_latent_matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let (batch, heads, rows, inner) = lhs.dims4()?;
    let (rhs_batch, rhs_heads, rhs_inner, columns) = rhs.dims4()?;
    anyhow::ensure!(
        rhs_batch == 1 && rhs_heads == 1,
        "GLM5 latent cache must have singleton batch/head dimensions, got {:?}",
        rhs.dims()
    );
    anyhow::ensure!(
        inner == rhs_inner,
        "GLM5 latent matmul inner dimensions differ: {inner} != {rhs_inner}"
    );
    let flattened_rows = batch
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(rows))
        .context("GLM5 latent matmul row count overflow")?;
    let flat_lhs = lhs.contiguous()?.reshape((flattened_rows, inner))?;
    let flat_rhs = rhs.contiguous()?.reshape((inner, columns))?;
    flat_lhs
        .matmul(&flat_rhs)?
        .reshape((batch, heads, rows, columns))
        .map_err(Into::into)
}

/// Compute GLM5's dense absorbed-MLA attention without narrowing the query.
///
/// DSA stores its latent cache in F16 for memory, but the reference disables
/// flash attention because the F32 query is materially sensitive to the
/// intermediate cast.  Keeping this helper separate makes that invariant
/// explicit and straightforward to test on CPU.
fn dense_latent_attention_f32(
    q_absorbed: &Tensor,
    cached_latent: &Tensor,
    seq_len: usize,
    total: usize,
    scale: f64,
    device: &Device,
    prefill: bool,
) -> Result<Tensor> {
    let latent = cached_latent.to_dtype(DType::F32)?;
    let scores = (shared_latent_matmul(q_absorbed, &latent.transpose(2, 3)?)? * scale)?;
    let scores = scores.broadcast_add(&causal_mask(seq_len, total, device)?)?;
    let scores = candle_nn::ops::softmax_last_dim(&scores)?;
    latent_context(&scores, &latent, prefill)
}

fn latent_context(probabilities: &Tensor, latent: &Tensor, _prefill: bool) -> Result<Tensor> {
    // Prompt partitions change the generic GEMM's reduction order. Decode
    // always has the same shape for a given position and keeps the faster
    // general path. Use the phase, not the row count: a prompt tail can be 1.
    #[cfg(target_os = "macos")]
    if _prefill && super::glm5_next_context_metal::supported(probabilities.device())? {
        return super::glm5_next_context_metal::weighted_context(probabilities, latent)
            .map_err(Into::into);
    }
    shared_latent_matmul(probabilities, latent)
}

struct MoeExpert {
    gate: GlmQMatMul,
    up: GlmQMatMul,
    down: GlmQMatMul,
    clamp: f32,
}

impl MoeExpert {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(x)?.to_dtype(DType::F32)?.contiguous()?;
        let up = self.up.forward(x)?.to_dtype(DType::F32)?.contiguous()?;
        self.down
            .forward(&silu_mul_clamped(&gate, &up, self.clamp)?)
    }
}

#[cfg(any(target_os = "macos", feature = "cuda"))]
struct IndexedExperts {
    gate: IndexedMoeMatMul,
    up: IndexedMoeMatMul,
    down: IndexedMoeMatMul,
    clamp: f32,
}

#[cfg(any(target_os = "macos", feature = "cuda"))]
impl IndexedExperts {
    const MAX_ROUTE_ROWS: usize = 4096;

    fn forward(
        &self,
        flat_x: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
        prefill: bool,
    ) -> Result<Tensor> {
        let (tokens, hidden) = flat_x.dims2()?;
        let (_, top_k) = ids.dims2()?;
        let chunk_tokens = (Self::MAX_ROUTE_ROWS / top_k).max(1);
        let mut outputs = Vec::with_capacity(tokens.div_ceil(chunk_tokens));
        for start in (0..tokens).step_by(chunk_tokens) {
            let len = (tokens - start).min(chunk_tokens);
            let input = flat_x
                .narrow(0, start, len)?
                .to_dtype(DType::F32)?
                .reshape((len, 1, hidden))?;
            let chunk_ids = ids.narrow(0, start, len)?;
            let chunk_weights = weights.narrow(0, start, len)?;
            let project = |matrix: &IndexedMoeMatMul, x: &Tensor| {
                if prefill {
                    matrix.forward_prefill(x, &chunk_ids)
                } else {
                    matrix.forward(x, &chunk_ids)
                }
            };
            let gate = project(&self.gate, &input)?;
            let up = project(&self.up, &input)?;
            let activated = silu_mul_clamped(&gate, &up, self.clamp)?;
            let routed = project(&self.down, &activated)?;
            outputs.push(weighted_route_sum(&routed, &chunk_weights)?);
        }
        Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 0).map_err(Into::into)
    }
}

enum MoeExperts {
    Split(Vec<MoeExpert>),
    #[cfg(any(target_os = "macos", feature = "cuda"))]
    Indexed(Box<IndexedExperts>),
}

impl MoeExperts {
    fn len(&self) -> usize {
        match self {
            Self::Split(experts) => experts.len(),
            #[cfg(any(target_os = "macos", feature = "cuda"))]
            Self::Indexed(experts) => experts.gate.num_experts(),
        }
    }
}

struct MoeLayer {
    gate: GlmQMatMul,
    bias: Tensor,
    experts: MoeExperts,
    active: usize,
    scale: f32,
    shared: DenseMlp,
}

impl MoeLayer {
    fn forward_decode_rows(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, rows, _) = x.dims3()?;
        anyhow::ensure!(
            batch == 1 && rows > 0,
            "GLM5 FFN decode rows require one nonempty batch"
        );
        // Indexed decode uses the same per-route vector kernel below 64 rows.
        // Bound chunks to keep larger batches off the prompt matrix path.
        #[cfg(any(target_os = "macos", feature = "cuda"))]
        let indexed = matches!(self.experts, MoeExperts::Indexed(_));
        #[cfg(not(any(target_os = "macos", feature = "cuda")))]
        let indexed = false;
        let chunk = if indexed { 32 } else { 1 };
        let routed = (0..rows)
            .step_by(chunk)
            .map(|row| {
                self.forward_routed_impl(&x.narrow(1, row, (rows - row).min(chunk))?, false, true)
            })
            .collect::<Result<Vec<_>>>()?;
        let shared = self.shared.forward_decode_rows(x)?.to_dtype(DType::F32)?;
        Ok((Tensor::cat(&routed, 1)?.to_dtype(DType::F32)? + shared)?.to_dtype(x.dtype())?)
    }

    fn forward(&self, x: &Tensor, prefill: bool) -> Result<Tensor> {
        let routed = self.forward_routed(x, prefill)?;
        let shared = self.shared.forward(x)?.to_dtype(DType::F32)?;
        (routed.to_dtype(DType::F32)? + shared)?
            .to_dtype(x.dtype())
            .map_err(Into::into)
    }

    fn selected_experts(
        &self,
        selection: &[f32],
        probabilities: &[f32],
    ) -> Result<Vec<(usize, f32)>> {
        let expert_count = self.experts.len();
        anyhow::ensure!(
            selection.len() == expert_count && probabilities.len() == expert_count,
            "GLM5 router row width differs"
        );
        let mut indexed = selection.iter().copied().enumerate().collect::<Vec<_>>();
        indexed.sort_by(|left, right| right.1.total_cmp(&left.1));
        let mut selected = indexed[..self.active]
            .iter()
            .map(|(expert, _)| (*expert, probabilities[*expert]))
            .collect::<Vec<_>>();
        let denominator = selected
            .iter()
            .map(|(_, weight)| *weight)
            .sum::<f32>()
            .max(ROUTE_DENOMINATOR_FLOOR as f32);
        for (_, weight) in &mut selected {
            *weight = *weight / denominator * self.scale;
        }
        Ok(selected)
    }

    fn forward_routed(&self, x: &Tensor, _prefill: bool) -> Result<Tensor> {
        self.forward_routed_impl(x, _prefill, false)
    }

    fn forward_routed_impl(&self, x: &Tensor, _prefill: bool, decode_rows: bool) -> Result<Tensor> {
        let (batch, seq_len, hidden) = x.dims3()?;
        let tokens = batch * seq_len;
        let flat = x.reshape((tokens, hidden))?;
        let logits = if decode_rows {
            self.gate.forward_decode_rows(&flat)?
        } else {
            self.gate.forward(&flat)?
        }
        .to_dtype(DType::F32)?
        .contiguous()?;

        #[cfg(target_os = "macos")]
        let fused = match &self.experts {
            MoeExperts::Indexed(_) if logits.device().is_metal() => {
                let (ids, weights) = fused_sigmoid_topk_route(&logits, &self.bias, self.active)?;
                Some((ids, (weights * self.scale as f64)?))
            }
            _ => None,
        };
        #[cfg(not(target_os = "macos"))]
        let fused: Option<(Tensor, Tensor)> = None;

        let probabilities = if fused.is_none() {
            Some(sigmoid(&logits)?)
        } else {
            None
        };
        let selection = probabilities
            .as_ref()
            .map(|probabilities| probabilities.broadcast_add(&self.bias))
            .transpose()?;

        #[cfg(any(target_os = "macos", feature = "cuda"))]
        let indexed_route = if let Some(route) = fused {
            Some(route)
        } else {
            match &self.experts {
                MoeExperts::Indexed(_) => {
                    let probabilities = probabilities
                        .as_ref()
                        .context("missing GLM5 router probabilities")?;
                    let ids = selection
                        .as_ref()
                        .context("missing GLM5 route scores")?
                        .contiguous()?
                        .arg_sort_last_dim(false)?
                        .narrow(D::Minus1, 0, self.active)?
                        .contiguous()?;
                    let weights =
                        (normalized_route_weights(probabilities, &ids)? * self.scale as f64)?;
                    Some((ids, weights))
                }
                MoeExperts::Split(_) => None,
            }
        };
        #[cfg(not(any(target_os = "macos", feature = "cuda")))]
        let indexed_route: Option<(Tensor, Tensor)> = None;

        let host_route = if indexed_route.is_none() {
            let selection = selection.as_ref().context("missing GLM5 route scores")?;
            let probabilities = probabilities
                .as_ref()
                .context("missing GLM5 router probabilities")?;
            Some((selection.to_vec2::<f32>()?, probabilities.to_vec2::<f32>()?))
        } else {
            None
        };

        let routed = match &self.experts {
            #[cfg(any(target_os = "macos", feature = "cuda"))]
            MoeExperts::Indexed(experts) => {
                let (ids, weights) = indexed_route
                    .as_ref()
                    .context("missing GLM5 indexed route")?;
                experts.forward(&flat, ids, weights, _prefill)?
            }
            MoeExperts::Split(experts) if tokens == 1 => {
                let (selection, probabilities) =
                    host_route.as_ref().context("missing GLM5 host route")?;
                let selected = self.selected_experts(&selection[0], &probabilities[0])?;
                let mut outputs = Vec::with_capacity(selected.len());
                for (expert, weight) in selected {
                    outputs.push(
                        (experts[expert].forward(&flat)?.to_dtype(DType::F32)? * weight as f64)?,
                    );
                }
                Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 0)?
                    .sum(0)?
                    .unsqueeze(0)?
            }
            MoeExperts::Split(experts) => {
                let (selection, probabilities) =
                    host_route.as_ref().context("missing GLM5 host route")?;
                let mut assignments = vec![Vec::<(usize, f32)>::new(); experts.len()];
                for token in 0..tokens {
                    for (expert, weight) in
                        self.selected_experts(&selection[token], &probabilities[token])?
                    {
                        assignments[expert].push((token, weight));
                    }
                }
                let mut outputs = Vec::new();
                let mut output_tokens = Vec::new();
                for (expert, rows) in assignments.iter().enumerate() {
                    if rows.is_empty() {
                        continue;
                    }
                    let token_ids = rows
                        .iter()
                        .map(|(token, _)| *token as u32)
                        .collect::<Vec<_>>();
                    let ids = Tensor::new(token_ids.as_slice(), x.device())?;
                    let input = flat.index_select(&ids, 0)?;
                    let expert_output = experts[expert].forward(&input)?.to_dtype(DType::F32)?;
                    let weights = rows.iter().map(|(_, weight)| *weight).collect::<Vec<_>>();
                    let weights =
                        Tensor::new(weights.as_slice(), x.device())?.reshape((weights.len(), 1))?;
                    outputs.push(expert_output.broadcast_mul(&weights)?);
                    output_tokens.extend(token_ids);
                }
                let outputs = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 0)?;
                let output_tokens = Tensor::new(output_tokens.as_slice(), x.device())?;
                Tensor::zeros((tokens, hidden), DType::F32, x.device())?.index_add(
                    &output_tokens,
                    &outputs,
                    0,
                )?
            }
        };
        routed.reshape((batch, seq_len, hidden)).map_err(Into::into)
    }
}

fn split_packed_experts(
    gate: &Arc<QTensor>,
    up: &Arc<QTensor>,
    down: &Arc<QTensor>,
    md: &GgufMetadata,
    clamp: f32,
    device: &Device,
) -> Result<Vec<MoeExpert>> {
    let experts = md.expert_count;
    let ffn = md.expert_ffn_dim;
    let hidden = md.hidden_dim;
    let split = |packed: &QTensor, out: usize, input: usize| -> Result<Vec<GlmQMatMul>> {
        let dtype = packed.dtype();
        let elements = out * input;
        anyhow::ensure!(
            elements.is_multiple_of(dtype.block_size()),
            "GLM5 expert tensor is not block aligned"
        );
        let bytes = elements / dtype.block_size() * dtype.type_size();
        let data = packed.data()?;
        let mut result = Vec::with_capacity(experts);
        for expert in 0..experts {
            let start = expert * bytes;
            let tensor = candle_core::quantized::ggml_file::qtensor_from_ggml(
                dtype,
                &data[start..start + bytes],
                vec![out, input],
                device,
            )?;
            result.push(GlmQMatMul::from_qtensor(Arc::new(tensor), device)?);
        }
        Ok(result)
    };
    let gates = split(gate, ffn, hidden)?;
    let ups = split(up, ffn, hidden)?;
    let downs = split(down, hidden, ffn)?;
    Ok((0..experts)
        .map(|expert| MoeExpert {
            gate: gates[expert].clone(),
            up: ups[expert].clone(),
            down: downs[expert].clone(),
            clamp,
        })
        .collect())
}

fn build_moe(
    tensors: &mut HashMap<String, Arc<QTensor>>,
    prefix: &str,
    md: &GgufMetadata,
    clamp: f32,
    shared_clamp: f32,
    device: &Device,
) -> Result<MoeLayer> {
    let gate = qmatmul(tensors, &format!("{prefix}.ffn_gate_inp.weight"), device)?;
    let bias = dense_tensor(tensors, &format!("{prefix}.exp_probs_b.bias"), device)?;
    let gate_name = format!("{prefix}.ffn_gate_exps.weight");
    let gate_exps = tensors
        .remove(&gate_name)
        .with_context(|| format!("missing GLM5 tensor {gate_name}"))?;
    let up_name = format!("{prefix}.ffn_up_exps.weight");
    let up_exps = tensors
        .remove(&up_name)
        .with_context(|| format!("missing GLM5 tensor {up_name}"))?;
    let down_name = format!("{prefix}.ffn_down_exps.weight");
    let down_exps = tensors
        .remove(&down_name)
        .with_context(|| format!("missing GLM5 tensor {down_name}"))?;

    #[cfg(any(target_os = "macos", feature = "cuda"))]
    let experts = if indexed_moe_available(device, gate_exps.dtype())
        && indexed_moe_available(device, up_exps.dtype())
        && indexed_moe_available(device, down_exps.dtype())
    {
        let experts = MoeExperts::Indexed(Box::new(IndexedExperts {
            gate: IndexedMoeMatMul::from_qtensor(
                &gate_exps,
                md.expert_count,
                md.expert_ffn_dim,
                md.hidden_dim,
                device,
            )?,
            up: IndexedMoeMatMul::from_qtensor(
                &up_exps,
                md.expert_count,
                md.expert_ffn_dim,
                md.hidden_dim,
                device,
            )?,
            down: IndexedMoeMatMul::from_qtensor(
                &down_exps,
                md.expert_count,
                md.hidden_dim,
                md.expert_ffn_dim,
                device,
            )?,
            clamp,
        }));
        device.synchronize()?;
        experts
    } else {
        MoeExperts::Split(split_packed_experts(
            &gate_exps, &up_exps, &down_exps, md, clamp, device,
        )?)
    };
    #[cfg(not(any(target_os = "macos", feature = "cuda")))]
    let experts = MoeExperts::Split(split_packed_experts(
        &gate_exps, &up_exps, &down_exps, md, clamp, device,
    )?);

    Ok(MoeLayer {
        gate,
        bias,
        experts,
        active: md.expert_used_count,
        scale: md.expert_weights_scale as f32,
        shared: DenseMlp::shared(tensors, prefix, shared_clamp, device)?,
    })
}

enum GlmAttention {
    Kda(Box<KdaAttention>),
    Dsa(Box<DsaAttention>),
}

enum GlmFfn {
    Dense(DenseMlp),
    Moe(MoeLayer),
}

impl GlmFfn {
    fn forward_decode_rows(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward_decode_rows(x),
            Self::Moe(moe) => moe.forward_decode_rows(x),
        }
    }

    fn forward(&self, x: &Tensor, prefill: bool) -> Result<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::Moe(moe) => moe.forward(x, prefill),
        }
    }
}

struct GlmBlock {
    attn_norm: RmsNorm,
    attention: GlmAttention,
    attn_hc: HcMix,
    ffn_norm: RmsNorm,
    ffn: GlmFfn,
    ffn_hc: HcMix,
}

/// A loaded GLM-5.3 Flash stage for pipeline execution.
pub struct Glm5NextStage {
    embedding: Option<Arc<QTensor>>,
    blocks: Vec<GlmBlock>,
    output_norm: Option<RmsNorm>,
    output: Option<GlmQMatMul>,
    is_head: bool,
    is_tail: bool,
    hidden: usize,
    stage_hidden: usize,
    vocab: usize,
    context_length: usize,
    decode_batch_enabled: bool,
    device: Device,
}

impl Glm5NextStage {
    /// Load a GLM5 stage without retaining the CPU QTensor sources for every
    /// assigned layer at once.
    ///
    /// GLM5's packed experts and DSA projections are read into CPU QTensor
    /// storage before their device representations are built.  The regular
    /// stage loader reads all of those sources into one map, which makes a
    /// large split GGUF stage transiently consume the sum of every layer's
    /// sources.  Load one layer map, construct its block, and drop that map
    /// before moving to the next layer instead.  The resulting GLM block owns
    /// only its final device tensors (or the deliberate CPU fallback), so the
    /// lifetime matches the streaming memory profile.
    pub fn from_streaming_gguf(
        gguf_path: &Path,
        range: LayerRange,
        device: &Device,
        quant_config: Option<Arc<TurboQuantConfig>>,
    ) -> Result<Self> {
        let mut loader = StageTensorLoader::open(gguf_path)?;
        let metadata = loader.metadata().clone();
        anyhow::ensure!(
            matches!(metadata.architecture.as_str(), "glm5next" | "glm5-next"),
            "not a GLM5-Next stage"
        );
        anyhow::ensure!(
            range.total_layers == metadata.num_layers,
            "Layer range total_layers ({}) doesn't match model num_layers ({})",
            range.total_layers,
            metadata.num_layers
        );

        let mut embedding = None;
        let mut blocks = Vec::with_capacity(range.layer_count());
        let mut output_norm = None;
        let mut output = None;

        for layer in range.start..range.end {
            let layer_range = LayerRange::new(layer, layer + 1, range.total_layers);
            let layer_tensors = loader
                .load_range(layer_range, device)
                .with_context(|| format!("failed to stream GLM5 layer {layer}"))?;
            let mut layer_stage = Self::from_stage_tensors(layer_tensors, quant_config.clone())?;

            // Only the first stage-layer can own the head embedding, and only
            // the final stage-layer can own the tail output.  Moving these
            // values out before dropping `layer_stage` releases its source
            // map immediately while preserving the original boundary
            // semantics for partial pipeline stages and solo loads.
            if embedding.is_none() {
                embedding = layer_stage.embedding.take();
            }
            blocks.append(&mut layer_stage.blocks);
            if output_norm.is_none() {
                output_norm = layer_stage.output_norm.take();
            }
            if output.is_none() {
                output = layer_stage.output.take();
            }

            // Construction copies quantized sources into their final device
            // representation asynchronously.  Finish this layer before
            // loading the next one so command-buffer retention cannot make
            // the transient source memory accumulate across layers.  This
            // also makes a Metal construction fault fail at its exact layer.
            drop(layer_stage);
            device
                .synchronize()
                .with_context(|| format!("failed to finish streaming GLM5 layer {layer}"))?;
        }

        Ok(Self {
            embedding,
            blocks,
            output_norm,
            output,
            is_head: range.is_head(),
            is_tail: range.is_tail(),
            hidden: metadata.hidden_dim,
            stage_hidden: metadata.hidden_dim * GLM5_STREAMS,
            vocab: metadata.vocab_size,
            context_length: metadata.context_length,
            decode_batch_enabled: cfg!(all(target_os = "macos", target_arch = "aarch64"))
                && device.is_metal(),
            device: device.clone(),
        })
    }

    /// Construct a GLM-5.3 stage from an already loaded tensor range.
    ///
    /// Returns an error when GLM5 metadata, hybrid-attention schedules, or
    /// required tensors are missing or incompatible with the supported model
    /// geometry.
    pub fn from_stage_tensors(
        stage: StageTensors,
        quant_config: Option<Arc<TurboQuantConfig>>,
    ) -> Result<Self> {
        let StageTensors {
            range,
            metadata: md,
            mut tensors,
            device,
            ..
        } = stage;
        anyhow::ensure!(
            matches!(md.architecture.as_str(), "glm5next" | "glm5-next"),
            "not a GLM5-Next stage"
        );
        anyhow::ensure!(
            md.hyperconnection_count == GLM5_STREAMS,
            "GLM5 needs four mHC streams"
        );
        anyhow::ensure!(
            md.head_count_kv_arr.len() >= md.num_layers,
            "GLM5 hybrid attention schedule is missing"
        );
        anyhow::ensure!(
            md.expert_count == 288 && md.expert_used_count == 8,
            "unsupported GLM5 expert geometry"
        );
        anyhow::ensure!(
            md.expert_gating_func == 2 && md.expert_weights_norm,
            "GLM5 requires normalized sigmoid expert routing"
        );
        anyhow::ensure!(
            md.swiglu_clamp_exp.len() >= md.num_layers
                && md.swiglu_clamp_shexp.len() >= md.num_layers,
            "GLM5 SwiGLU clamp schedule is missing"
        );
        if quant_config.is_some() {
            tracing::warn!("GLM5 latent caches stay f16; ignoring TurboQuant KV settings");
        }
        #[cfg(target_os = "macos")]
        super::glm5_next_kda_metal::precompile(&device)?;

        let embedding = if range.is_head() {
            Some(
                tensors
                    .remove("token_embd.weight")
                    .context("missing GLM5 token embedding")?,
            )
        } else {
            None
        };
        if !range.is_head() && range.is_tail() {
            tensors.remove("token_embd.weight");
        }

        let mut blocks = Vec::with_capacity(range.layer_count());
        for layer in range.start..range.end {
            let prefix = format!("blk.{layer}");
            let clamp = md.swiglu_clamp_exp[layer] as f32;
            let shared_clamp = md.swiglu_clamp_shexp[layer] as f32;
            anyhow::ensure!(
                clamp.is_finite() && clamp > 0.0 && shared_clamp.is_finite() && shared_clamp > 0.0,
                "invalid GLM5 SwiGLU clamp at layer {layer}"
            );
            let attention = if md.head_count_kv_arr[layer] == 0 {
                GlmAttention::Kda(Box::new(KdaAttention::new(
                    &mut tensors,
                    &prefix,
                    &md,
                    &device,
                )?))
            } else {
                GlmAttention::Dsa(Box::new(DsaAttention::new(
                    &mut tensors,
                    &prefix,
                    &md,
                    &device,
                )?))
            };
            let ffn = if layer < md.first_k_dense_replace {
                GlmFfn::Dense(DenseMlp::new(&mut tensors, &prefix, clamp, &device)?)
            } else {
                GlmFfn::Moe(build_moe(
                    &mut tensors,
                    &prefix,
                    &md,
                    clamp,
                    shared_clamp,
                    &device,
                )?)
            };
            blocks.push(GlmBlock {
                attn_norm: RmsNorm::new(
                    &mut tensors,
                    &format!("{prefix}.attn_norm.weight"),
                    md.rms_norm_eps,
                    &device,
                )?,
                attention,
                attn_hc: HcMix::new(&mut tensors, &format!("{prefix}.hc_attn"), &md, &device)?,
                ffn_norm: RmsNorm::new(
                    &mut tensors,
                    &format!("{prefix}.ffn_norm.weight"),
                    md.rms_norm_eps,
                    &device,
                )?,
                ffn,
                ffn_hc: HcMix::new(&mut tensors, &format!("{prefix}.hc_ffn"), &md, &device)?,
            });

            let layer_prefix = format!("{prefix}.");
            let names = tensors
                .keys()
                .filter(|name| name.starts_with(&layer_prefix))
                .cloned()
                .collect::<Vec<_>>();
            for name in names {
                tensors.remove(&name);
            }
        }

        let output_norm = if range.is_tail() {
            Some(RmsNorm::new(
                &mut tensors,
                "output_norm.weight",
                md.rms_norm_eps,
                &device,
            )?)
        } else {
            None
        };
        let output = if range.is_tail() {
            Some(qmatmul(&mut tensors, "output.weight", &device)?)
        } else {
            None
        };

        tracing::info!(
            layers = blocks.len(),
            start = range.start,
            end = range.end,
            kda = blocks
                .iter()
                .filter(|block| matches!(block.attention, GlmAttention::Kda(_)))
                .count(),
            dsa = blocks
                .iter()
                .filter(|block| matches!(block.attention, GlmAttention::Dsa(_)))
                .count(),
            "GLM5-Next text stage loaded"
        );

        Ok(Self {
            embedding,
            blocks,
            output_norm,
            output,
            is_head: range.is_head(),
            is_tail: range.is_tail(),
            hidden: md.hidden_dim,
            stage_hidden: md.hidden_dim * GLM5_STREAMS,
            vocab: md.vocab_size,
            context_length: md.context_length,
            decode_batch_enabled: cfg!(all(target_os = "macos", target_arch = "aarch64"))
                && device.is_metal(),
            device,
        })
    }
}

impl Glm5NextStage {
    fn forward_pass(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        _start_pos: usize,
        prefill: bool,
        wants_output: bool,
    ) -> Result<StageOutput> {
        anyhow::ensure!(
            cache.len() >= self.blocks.len(),
            "GLM5 cache has too few layers"
        );
        let mut hidden = match input {
            StageInput::Tokens(tokens) => {
                anyhow::ensure!(self.is_head, "only the GLM5 head stage accepts tokens");
                anyhow::ensure!(tokens.len() == seq_len, "GLM5 token count differs");
                let ids = Tensor::new(tokens.as_slice(), &self.device)?;
                self.embedding
                    .as_ref()
                    .context("GLM5 token embedding is not loaded")?
                    .embedding(&ids)?
                    .reshape((1, seq_len, self.hidden))?
                    .unsqueeze(2)?
                    .repeat((1, 1, GLM5_STREAMS, 1))?
            }
            StageInput::Activations {
                tensor,
                bidirectional_spans,
                positions,
            } => {
                anyhow::ensure!(
                    bidirectional_spans.is_empty() && positions.is_none(),
                    "GLM5 text stages do not accept vision position metadata"
                );
                match tensor.dims() {
                    [seq, width] => {
                        anyhow::ensure!(
                            *seq == seq_len && *width == self.stage_hidden,
                            "GLM5 activation shape differs"
                        );
                        tensor.reshape((1, seq_len, GLM5_STREAMS, self.hidden))?
                    }
                    [1, seq, streams, hidden] => {
                        anyhow::ensure!(
                            *seq == seq_len && *streams == GLM5_STREAMS && *hidden == self.hidden,
                            "GLM5 stream activation shape differs"
                        );
                        tensor
                    }
                    shape => anyhow::bail!("invalid GLM5 activation shape {shape:?}"),
                }
            }
        }
        .to_dtype(DType::F32)?;

        finish_prefill_work(&self.device, seq_len)?;
        for (layer, block) in self.blocks.iter().enumerate() {
            // Metal decode keeps a long asynchronous branch between mHC pre
            // and post. Derive the tiny post controls only after that branch.
            // Retain the immutable projected mixes, not the final controls,
            // to avoid repeating normalization and projection on decode.
            let defer_hc_post_controls = !prefill && seq_len == 1 && self.device.is_metal();
            {
                let residual = hidden.clone();
                let (attention_input, attention_controls, attention_mixes) =
                    if defer_hc_post_controls {
                        let (input, mixes) = block.attn_hc.pre_update(&hidden)?;
                        (input, None, mixes)
                    } else {
                        let (input, post, combination) = block.attn_hc.pre(&hidden)?;
                        (input, Some((post, combination)), None)
                    };
                let attention_input = block.attn_norm.forward(&attention_input)?;
                let attention_output = match &block.attention {
                    GlmAttention::Kda(attention) => {
                        let mut state = match cache[layer].take() {
                            Some(LayerCache::DeltaNet(state)) => Some(state),
                            Some(_) => anyhow::bail!("GLM5 KDA layer received an attention cache"),
                            None => None,
                        };
                        let output = attention
                            .forward(&attention_input, &mut state)
                            .with_context(|| format!("GLM5 stage layer {layer} KDA"))?;
                        cache[layer] = state.map(LayerCache::DeltaNet);
                        output
                    }
                    GlmAttention::Dsa(attention) => with_kv_cache(&mut cache[layer], |kv| {
                        attention.forward(&attention_input, kv, seq_len, prefill)
                    })
                    .with_context(|| format!("GLM5 stage layer {layer} DSA"))?,
                };
                let (post, combination) = match attention_controls {
                    Some(controls) => controls,
                    None => block
                        .attn_hc
                        .post_controls(&residual, attention_mixes.as_ref())?,
                };
                hidden = block
                    .attn_hc
                    .post(&attention_output, &residual, &post, &combination)?;
            }
            // Release retired attention buffers before allocating the routed
            // expert intermediates. The model can sit near the device limit
            // with two saved prompts; waiting only at layer end is too late.
            finish_prefill_work(&self.device, seq_len)
                .with_context(|| format!("GLM5 stage layer {layer} attention completion"))?;
            {
                let residual = hidden.clone();
                let (ffn_input, ffn_controls, ffn_mixes) = if defer_hc_post_controls {
                    let (input, mixes) = block.ffn_hc.pre_update(&hidden)?;
                    (input, None, mixes)
                } else {
                    let (input, post, combination) = block.ffn_hc.pre(&hidden)?;
                    (input, Some((post, combination)), None)
                };
                let ffn_input = block.ffn_norm.forward(&ffn_input)?;
                let ffn_output = block
                    .ffn
                    .forward(&ffn_input, prefill)
                    .with_context(|| format!("GLM5 stage layer {layer} FFN"))?;
                let (post, combination) = match ffn_controls {
                    Some(controls) => controls,
                    None => block.ffn_hc.post_controls(&residual, ffn_mixes.as_ref())?,
                };
                hidden = block
                    .ffn_hc
                    .post(&ffn_output, &residual, &post, &combination)?;
            }
            finish_prefill_work(&self.device, seq_len)
                .with_context(|| format!("GLM5 stage layer {layer} FFN completion"))?;
        }

        if self.is_tail && wants_output {
            let hidden = hidden.sum(2)?.affine(1.0 / GLM5_STREAMS as f64, 0.0)?;
            let hidden = self
                .output_norm
                .as_ref()
                .context("GLM5 output norm is not loaded")?
                .forward(&hidden)?;
            let last = hidden.i((.., seq_len - 1, ..))?;
            let logits = self
                .output
                .as_ref()
                .context("GLM5 output head is not loaded")?
                .forward(&last)?;
            Ok(StageOutput::Logits(logits.squeeze(0)?))
        } else {
            Ok(StageOutput::Activations(
                hidden.reshape((seq_len, self.stage_hidden))?,
            ))
        }
    }
}

impl ModelArchitecture for Glm5NextStage {
    fn fixed_cache_bytes(&self, _verification_rows: usize) -> Result<usize> {
        self.blocks.iter().try_fold(0usize, |total, block| {
            let GlmAttention::Kda(block) = &block.attention else {
                return Ok(total);
            };
            let recurrent = super::fixed_state_tensor_bytes(
                &[block.heads, block.head_dim, block.head_dim],
                &self.device,
            )?;
            let convolution =
                super::fixed_state_tensor_bytes(&[3, block.inner, block.kernel], &self.device)?;
            recurrent
                .checked_add(convolution)
                .and_then(|bytes| total.checked_add(bytes))
                .context("KDA fixed request state size overflow")
        })
    }

    fn supports_pipeline_decode_batch(&self) -> bool {
        self.decode_batch_enabled
            && !(self.is_head && self.is_tail)
            && (self.device.is_cpu() || self.device.is_metal())
    }

    fn forward_pipeline_decode_batch(
        &self,
        inputs: &[StageInput],
        caches: &mut [&mut [Option<LayerCache>]],
        batch_cache: &mut [Option<LayerCache>],
        positions: &[usize],
    ) -> Result<Vec<StageOutput>> {
        self.decode_pipeline_batch(inputs, caches, batch_cache, positions)
    }

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

    fn forward(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_pass(input, cache, seq_len, start_pos, false, true)
    }

    fn forward_prefill(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_pass(input, cache, seq_len, start_pos, true, true)
    }

    fn forward_prefill_cache_only(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_pass(input, cache, seq_len, start_pos, true, false)
    }

    fn embed_tokens(&self, tokens: &[u32]) -> Result<Tensor> {
        anyhow::ensure!(self.is_head, "only the GLM5 head stage can embed tokens");
        let ids = Tensor::new(tokens, &self.device)?;
        self.embedding
            .as_ref()
            .context("GLM5 token embedding is not loaded")?
            .embedding(&ids)
            .map_err(Into::into)
    }

    fn prefers_f32_decode_activations(&self) -> bool {
        // GLM5 carries four hyperconnection streams across pipeline stages.
        // Rounding that recurrent state to F16 at every decode step causes
        // rapid numerical drift, even though the receiving stage upcasts it
        // back to F32. Preserve the state at its compute precision between
        // stages.
        true
    }

    fn prefers_f32_prefill_activations(&self) -> bool {
        // The same four-stream hyperconnection state crosses the boundary
        // during prefill. Keeping only decode F32 leaves the first stage
        // boundary lossy, so a split pipeline can diverge before decode starts.
        true
    }

    fn family(&self) -> &str {
        "glm5next"
    }

    fn hidden_dim(&self) -> usize {
        self.stage_hidden
    }

    fn layer_count(&self) -> usize {
        self.blocks.len()
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glm5_compressor_tensor_alias_accepts_legacy_bare_name() {
        let name = choose_tensor_name(
            false,
            true,
            "blk.0.indexer_compressor_gate.weight",
            "blk.0.indexer_compressor_gate",
        )
        .unwrap();
        assert_eq!(name, "blk.0.indexer_compressor_gate");
    }

    #[test]
    fn glm5_compressor_tensor_alias_rejects_ambiguous_names() {
        assert!(
            choose_tensor_name(
                true,
                true,
                "blk.0.indexer_compressor_gate.weight",
                "blk.0.indexer_compressor_gate",
            )
            .is_err()
        );
    }

    #[test]
    fn glm5_activations_stay_f32_across_pipeline_stages() {
        let stage = Glm5NextStage {
            embedding: None,
            blocks: Vec::new(),
            output_norm: None,
            output: None,
            is_head: false,
            is_tail: false,
            hidden: 1,
            stage_hidden: GLM5_STREAMS,
            vocab: 1,
            context_length: 128,
            decode_batch_enabled: false,
            device: Device::Cpu,
        };
        assert!(stage.prefers_f32_decode_activations());
        assert!(stage.prefers_f32_prefill_activations());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn glm5_prefill_releases_retired_buffers_but_decode_does_not_wait() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let buffer = device
            .as_metal_device()?
            .new_buffer_with_data(&[0u8; 4096])?;
        let retired = Arc::downgrade(&buffer);
        drop(buffer);
        finish_prefill_work(&device, 1)?;
        assert!(retired.upgrade().is_some());
        finish_prefill_work(&device, 2)?;
        assert!(retired.upgrade().is_none());
        Ok(())
    }

    #[test]
    fn glm5_single_token_gemm_workaround_is_narrowly_scoped() {
        assert!(needs_glm_metal_single_token_gemm(GgmlDType::Q2K));
        assert!(needs_glm_metal_single_token_gemm(GgmlDType::Q3K));
        assert!(!needs_glm_metal_single_token_gemm(GgmlDType::Q4K));
        assert!(!needs_glm_metal_single_token_gemm(GgmlDType::Q6K));
        assert!(!needs_glm_metal_single_token_gemm(GgmlDType::F16));
    }

    #[test]
    fn glm5_single_token_gemm_padding_preserves_rank_and_value() -> Result<()> {
        let identity = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &Device::Cpu)?;
        let matmul = GlmQMatMul::CandlePaddedSingleToken(QMatMul::Tensor(identity));

        let rank_two = Tensor::new(&[[2.0f32, 3.0]], &Device::Cpu)?;
        assert_eq!(
            matmul.forward(&rank_two)?.to_vec2::<f32>()?,
            vec![vec![2.0, 3.0]]
        );

        let rank_three = rank_two.unsqueeze(0)?;
        assert_eq!(matmul.forward(&rank_three)?.dims(), &[1, 1, 2]);
        assert_eq!(
            matmul.forward(&rank_three)?.to_vec3::<f32>()?,
            vec![vec![vec![2.0, 3.0]]]
        );
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn glm5_metal_q2_q3_single_token_matches_cpu_reference() -> Result<()> {
        if candle_metal_kernels::metal::Device::all().is_empty() {
            return Ok(());
        }
        let Some(metal) = crate::test_metal_device() else {
            return Ok(());
        };
        let cpu = Device::Cpu;
        let source = Tensor::from_vec(
            (0..64 * 256)
                .map(|index| ((index % 97) as f32 - 48.0) / 32.0)
                .collect::<Vec<_>>(),
            (64, 256),
            &cpu,
        )?;
        let input = Tensor::from_vec(
            (0..256)
                .map(|index| ((index % 31) as f32 - 15.0) / 16.0)
                .collect::<Vec<_>>(),
            (1, 1, 256),
            &cpu,
        )?;

        for dtype in [GgmlDType::Q2K, GgmlDType::Q3K] {
            let cpu_weights = QTensor::quantize_onto(&source, dtype, &cpu)?;
            let cpu_matmul = QMatMul::from_arc(Arc::new(cpu_weights))?;
            let expected = cpu_matmul
                .forward(&input)?
                .flatten_all()?
                .to_vec1::<f32>()?;

            let metal_weights = QTensor::quantize_onto(&source, dtype, &metal)?;
            let metal_matmul = GlmQMatMul::from_qtensor(Arc::new(metal_weights), &metal)?;
            assert!(matches!(
                &metal_matmul,
                GlmQMatMul::CandlePaddedSingleToken(_)
            ));
            let actual = metal_matmul
                .forward(&input.to_device(&metal)?)?
                .to_device(&cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let max_abs = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(max_abs < 0.1, "{dtype:?} max_abs={max_abs}");
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn glm5_low_rank_gate_projection_keeps_f32_rows_across_prompt_splits() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        if !super::super::tensor_qmatmul_metal::supports_metal_tensor_qmatmul(&device)? {
            return Ok(());
        }
        let source = Tensor::from_vec(
            (0..512 * 128)
                .map(|i| (i as f32 * 0.017).sin() * 0.2)
                .collect::<Vec<_>>(),
            (512, 128),
            &Device::Cpu,
        )?;
        let weights = Arc::new(QTensor::quantize(&source, GgmlDType::Q4_0)?);
        let dense = weights.dequantize(&Device::Cpu)?;
        let input = Tensor::from_vec(
            (0..131 * 128)
                .map(|i| (i as f32 * 0.023).cos() * 0.3)
                .collect::<Vec<_>>(),
            (1, 131, 128),
            &device,
        )?;
        let projection = GlmQMatMul::from_qtensor(weights, &device)?;
        let whole = projection.forward(&input)?;
        for (start, len) in [(0, 16), (16, 16), (128, 2), (130, 1)] {
            let part = input.narrow(1, start, len)?;
            let actual = projection.forward(&part)?.flatten_all()?.to_vec1::<f32>()?;
            let expected = whole
                .narrow(1, start, len)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert!(actual == expected, "gate rows changed at {start}+{len}");
            let reference = part
                .to_device(&Device::Cpu)?
                .reshape((len, 128))?
                .matmul(&dense.t()?)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let max_error = actual
                .iter()
                .zip(reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_error < 2e-5, "F32 reference error {max_error}");
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn glm5_indexed_heads_and_experts_keep_prompt_matrix_math() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let source = Tensor::from_vec(
            (0..4 * 256 * 256)
                .map(|i| (i as f32 * 0.037).sin() * 0.2)
                .collect::<Vec<_>>(),
            (4, 256, 256),
            &Device::Cpu,
        )?;
        let input = Tensor::from_vec(
            (0..65 * 4 * 256)
                .map(|i| (i as f32 * 0.029).cos() * 0.3)
                .collect::<Vec<_>>(),
            (65, 4, 256),
            &device,
        )?;
        let flat = input.narrow(1, 0, 1)?.squeeze(1)?.contiguous()?;
        let ids = Tensor::new(&[0u32, 2], &device)?
            .reshape((1, 2))?
            .repeat((65, 1))?;
        let weights = Tensor::new(&[0.3f32, 0.7], &device)?
            .reshape((1, 2))?
            .repeat((65, 1))?;
        for dtype in [GgmlDType::Q2K, GgmlDType::Q3K] {
            let packed = Arc::new(QTensor::quantize(&source, dtype)?);
            let head = HeadwiseMatMul::new(Arc::clone(&packed), 4, 256, 256, &device)?;
            let matrix = || IndexedMoeMatMul::from_qtensor(&packed, 4, 256, 256, &device);
            let experts = IndexedExperts {
                gate: matrix()?,
                up: matrix()?,
                down: matrix()?,
                clamp: 10.0,
            };
            let whole_head = head.forward(&input, true)?;
            let whole_experts = experts.forward(&flat, &ids, &weights, true)?;
            for (start, len) in [(0, 16), (16, 16), (64, 1)] {
                let head_part = head.forward(&input.narrow(0, start, len)?, true)?;
                let expert_part = experts.forward(
                    &flat.narrow(0, start, len)?,
                    &ids.narrow(0, start, len)?,
                    &weights.narrow(0, start, len)?,
                    true,
                )?;
                for (name, expected, actual) in [
                    ("head", &whole_head, head_part),
                    ("experts", &whole_experts, expert_part),
                ] {
                    let expected = expected
                        .narrow(0, start, len)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
                    let max_error = expected
                        .iter()
                        .zip(actual)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    assert!(
                        max_error < 2e-5,
                        "{dtype:?} {name} changed at {start}+{len}: {max_error}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn glm5_kda_uses_fixed_norm_epsilon() -> Result<()> {
        assert_eq!(GLM5_KDA_NORM_EPS, 1e-6);
        let input = Tensor::new(&[[3.0f32, 4.0]], &Device::Cpu)?;
        let normalized = l2_normalize(&input, GLM5_KDA_NORM_EPS)?.to_vec2::<f32>()?;
        assert!((normalized[0][0] - 3.0 / (25.0 + 1e-6f32).sqrt()).abs() < 1e-7);
        assert!((normalized[0][1] - 4.0 / (25.0 + 1e-6f32).sqrt()).abs() < 1e-7);
        Ok(())
    }

    #[test]
    fn glm5_dense_latent_attention_keeps_f32_query() -> Result<()> {
        let q = Tensor::new(&[[[[1.234567f32, -0.7654321]]]], &Device::Cpu)?;
        let latent = Tensor::new(&[[[[0.3333f32, 1.2345], [1.7777, -0.2222]]]], &Device::Cpu)?
            .to_dtype(DType::F16)?;
        let actual = dense_latent_attention_f32(&q, &latent, 1, 2, 0.5, &Device::Cpu, true)?;
        assert_eq!(actual.dtype(), DType::F32);

        let q = [1.234567f32, -0.7654321];
        let keys = [[0.3333f32, 1.2345], [1.7777, -0.2222]];
        let keys = keys.map(|row| row.map(|value| half::f16::from_f32(value).to_f32()));
        let scores = keys.map(|key| 0.5 * (q[0] * key[0] + q[1] * key[1]));
        let normalizer = scores.map(f32::exp).into_iter().sum::<f32>();
        let expected = [
            (scores[0].exp() * keys[0][0] + scores[1].exp() * keys[1][0]) / normalizer,
            (scores[0].exp() * keys[0][1] + scores[1].exp() * keys[1][1]) / normalizer,
        ];
        let actual = actual.flatten_all()?.to_vec1::<f32>()?;
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
        Ok(())
    }

    #[test]
    fn glm5_dense_latent_attention_broadcasts_kv_across_heads() -> Result<()> {
        // A multi-token prefill has one shared latent cache per batch while
        // the absorbed query has one matrix per attention head. Plain matmul
        // rejects the singleton cache-head dimension; the flattened helper
        // computes the broadcast without materializing one cache per head.
        let q = Tensor::zeros((1, 64, 4, 8), DType::F32, &Device::Cpu)?;
        let latent = Tensor::zeros((1, 1, 4, 8), DType::F16, &Device::Cpu)?;
        let output = dense_latent_attention_f32(&q, &latent, 4, 4, 0.5, &Device::Cpu, true)?;
        assert_eq!(output.dims(), &[1, 64, 4, 8]);
        assert_eq!(output.dtype(), DType::F32);
        Ok(())
    }

    fn check_pooled_index_matmul(device: &Device) -> Result<()> {
        let queries = Tensor::from_vec(
            (0..4 * 3 * 4).map(|i| (i % 7) as f32 - 3.).collect(),
            (4, 3, 4),
            device,
        )?
        .narrow(0, 1, 3)?;
        let pooled = Tensor::from_vec(
            (0..4 * 5).map(|i| (i % 5) as f32 - 2.).collect(),
            (4, 5),
            device,
        )?
        .t()?;
        let q_values = queries.to_vec3::<f32>()?;
        let p_values = pooled.to_vec2::<f32>()?;
        for rows in [1, 3] {
            let scores =
                pooled_index_matmul(&queries.narrow(0, 0, rows)?, &pooled)?.to_vec3::<f32>()?;
            for row in 0..rows {
                for head in 0..3 {
                    for (pool, keys) in p_values.iter().enumerate() {
                        let expected: f32 = q_values[row][head]
                            .iter()
                            .zip(keys)
                            .map(|(query, key)| query * key)
                            .sum();
                        assert_eq!(scores[row][head][pool], expected);
                    }
                }
            }
        }
        // The first failing full-model chunk: 2,176 cached tokens form 544
        // pools; 128 new query rows each have 32 128-wide index heads.
        let queries = Tensor::ones((128, 32, 128), DType::F32, device)?;
        let pooled = Tensor::ones((544, 128), DType::F32, device)?;
        let scores = pooled_index_matmul(&queries, &pooled)?;
        assert_eq!(scores.dims(), &[128, 32, 544]);
        assert!(
            scores
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|&n| n == 128.)
        );
        assert!(
            pooled_index_matmul(&queries, &Tensor::zeros((544, 127), DType::F32, device)?).is_err()
        );
        Ok(())
    }

    #[test]
    fn glm5_pooled_index_broadcasts_shared_keys_cpu() -> Result<()> {
        check_pooled_index_matmul(&Device::Cpu)
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn glm5_pooled_index_broadcasts_shared_keys_metal() -> Result<()> {
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        check_pooled_index_matmul(&device)
    }

    #[test]
    fn kda_portable_decay_is_per_key_dimension() -> Result<()> {
        let state = Tensor::ones((1, 2, 2), DType::F32, &Device::Cpu)?;
        let decay = Tensor::new(&[[0.0f32, 0.5f32.ln()]], &Device::Cpu)?
            .exp()?
            .unsqueeze(2)?;
        let decayed = state.broadcast_mul(&decay)?.to_vec3::<f32>()?;
        assert!((decayed[0][0][0] - 1.0).abs() < 1e-6);
        assert!((decayed[0][1][0] - 0.5).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn causal_mask_handles_cached_prefix() -> Result<()> {
        let mask = causal_mask(2, 5, &Device::Cpu)?
            .reshape((2, 5))?
            .to_vec2::<f32>()?;
        assert!(mask[0][4].is_infinite());
        assert_eq!(mask[0][3], 0.0);
        assert_eq!(mask[1][4], 0.0);
        Ok(())
    }

    #[test]
    fn hc_sinkhorn_starts_on_the_destination_axis() -> Result<()> {
        let logits = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], (1, 1, 2, 2), &Device::Cpu)?;
        let values = super::super::hc_sinkhorn::normalize(logits, 1, 1e-6, false)?
            .reshape((2, 2))?
            .to_vec2::<f32>()?;
        for row in values {
            for value in row {
                assert!((value - 0.5).abs() < 1e-6, "unexpected value {value}");
            }
        }
        Ok(())
    }

    #[test]
    fn hc_post_maps_sources_into_each_destination() -> Result<()> {
        let update = Tensor::zeros((1, 1, 1), DType::F32, &Device::Cpu)?;
        let post = Tensor::zeros((1, 1, 2), DType::F32, &Device::Cpu)?;
        let residual = Tensor::from_vec(vec![10.0f32, 20.0], (1, 1, 2, 1), &Device::Cpu)?;
        // Stored as [source, destination], matching the flattened GGUF mixer.
        let combination =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 2, 2), &Device::Cpu)?;
        let values = super::super::hc_post::reference(&update, &residual, &post, &combination)?
            .reshape(2)?
            .to_vec1::<f32>()?;
        assert_eq!(values, vec![70.0, 100.0]);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_split_decode_defers_hc_post_controls() -> Result<()> {
        fn branch(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
            let (batch, seq_len, hidden) = input.dims3()?;
            let mut output = input.clone();
            for _ in 0..6 {
                output = output
                    .reshape((batch * seq_len, hidden))?
                    .matmul(&weight.t()?)?
                    .tanh()?
                    .reshape((batch, seq_len, hidden))?;
            }
            Ok(output)
        }

        fn ordinary(mix: &HcMix, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
            let (update, post, combination) = mix.pre(input)?;
            mix.post(&branch(&update, weight)?, input, &post, &combination)
        }

        if candle_metal_kernels::metal::Device::all().is_empty() {
            return Ok(());
        }
        let Some(device) = crate::test_metal_device() else {
            return Ok(());
        };
        let hidden = 64;
        let stage_hidden = GLM5_STREAMS * hidden;
        let mix_dim = (2 + GLM5_STREAMS) * GLM5_STREAMS;
        let projection = Tensor::from_vec(
            (0..mix_dim * stage_hidden)
                .map(|index| ((index % 31) as f32 - 15.0) / 2048.0)
                .collect::<Vec<_>>(),
            (mix_dim, stage_hidden),
            &device,
        )?;
        let mut mix = HcMix {
            projection: GlmQMatMul::Candle(QMatMul::Tensor(projection)),
            base: Tensor::from_vec(
                (0..mix_dim)
                    .map(|index| ((index % 7) as f32 - 3.0) / 16.0)
                    .collect::<Vec<_>>(),
                mix_dim,
                &device,
            )?,
            scale: Tensor::new(&[0.7f32, 0.8, 0.6], &device)?,
            norm: RmsNorm::ones(stage_hidden, 1e-6, &device)?,
            streams: GLM5_STREAMS,
            hidden,
            sinkhorn_iterations: 20,
            eps: 1e-6,
            fused_sinkhorn: true,
            fused_post: true,
            fused_pre: true,
            reuse_projection: true,
        };
        let branch_weight = Tensor::from_vec(
            (0..hidden * hidden)
                .map(|index| {
                    let row = index / hidden;
                    let column = index % hidden;
                    if row == column { 0.8f32 } else { 0.0005f32 }
                })
                .collect::<Vec<_>>(),
            (hidden, hidden),
            &device,
        )?;
        let input = Tensor::from_vec(
            (0..27 * stage_hidden)
                .map(|index| ((index % 101) as f32 - 50.0) / 127.0)
                .collect::<Vec<_>>(),
            (1, 27, GLM5_STREAMS, hidden),
            &device,
        )?;

        // Warm the ordinary prefill path, then exercise the production decode
        // ordering: compute only the branch update before the long branch and
        // derive the tiny post controls immediately before consuming them.
        let _ = ordinary(&mix, &input.narrow(1, 0, 26)?, &branch_weight)?;
        let decode_input = input.narrow(1, 26, 1)?;
        mix.reuse_projection = false;
        let (reference_update, no_mixes) = mix.pre_update(&decode_input)?;
        assert!(no_mixes.is_none());
        let reference_output = branch(&reference_update, &branch_weight)?;
        let (reference_post, reference_comb) = mix.post_controls(&decode_input, None)?;
        let expected = mix
            .post(
                &reference_output,
                &decode_input,
                &reference_post,
                &reference_comb,
            )?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        mix.reuse_projection = true;
        for _ in 0..32 {
            let (update, mixes) = mix.pre_update(&decode_input)?;
            assert!(mixes.is_some());
            let branch_output = branch(&update, &branch_weight)?;
            let (post, comb) = mix.post_controls(&decode_input, mixes.as_ref())?;
            let actual = mix.post(&branch_output, &decode_input, &post, &comb)?;
            drop((update, mixes, branch_output, post, comb));
            assert_eq!(
                actual
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                expected,
            );
        }
        let (update, mixes) = mix.pre_update(&decode_input)?;
        let branch_output = branch(&update, &branch_weight)?;
        let (post, combination) = mix.post_controls(&decode_input, mixes.as_ref())?;
        let split = mix.post(&branch_output, &decode_input, &post, &combination)?;
        let contiguous = ordinary(&mix, &input, &branch_weight)?.narrow(1, 26, 1)?;
        device.synchronize()?;

        for controls in [&post, &combination] {
            assert!(
                controls
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .all(f32::is_finite)
            );
        }
        let split = split
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let contiguous = contiguous
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let (max_abs, nonfinite) = split.iter().zip(&contiguous).fold(
            (0.0f32, 0usize),
            |(max_abs, nonfinite), (&split, &contiguous)| {
                (
                    max_abs.max((split - contiguous).abs()),
                    nonfinite + usize::from(!split.is_finite() || !contiguous.is_finite()),
                )
            },
        );
        assert_eq!(nonfinite, 0);
        assert!(max_abs < 1e-4, "split decode max error {max_abs}");
        Ok(())
    }
}
