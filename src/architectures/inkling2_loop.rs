//! Inkling 2 Mini (`inkling2_loop`): a looped mixture-of-experts transformer.
//!
//! This is the plain-tensor reference path: every step uses ordinary tensor
//! ops so it can be checked against the published references before any of
//! it is fused. The semantic reference is Transformers'
//! `models/inkling2_loop/modeling_inkling2_loop.py` (PR #51050 at 6173a2f);
//! the GGUF layout follows llama.cpp's `src/models/inkling2.cpp`.
//!
//! Per token:
//!
//! 1. Embed, then RMS-normalize with `token_embd_norm`.
//! 2. Visit layers `0..loop_start` once, the looped core
//!    `loop_start..loop_end` `loop_count` times, and the rest once. Each pass
//!    over the core restarts from the state entering it and reads the
//!    previous pass through four messages. Every visit owns its KV cache and
//!    convolution state.
//! 3. There is no running residual sum. Layers whose index is a multiple of
//!    `attn_res.block_size` close a block: their input joins the list of
//!    block states. Each layer builds its attention and MoE inputs by
//!    attending over the RMS-normalized block states and the current partial
//!    block with learned queries (`attn_res_query`, `ffn_res_query`).
//! 4. Attention has no RoPE. Queries are scaled per channel and clipped to
//!    an L2 norm of `qk_clip`; keys pass a short causal convolution and are
//!    clipped; values pass the convolution and an unweighted RMS norm. A
//!    per-token, per-head relative bias over backward distances
//!    `0..relative_extent` and a per-token sink logit enter the softmax.
//!    Full-attention layers also scale queries and relative states by
//!    `1 + alpha * ln(max((pos + 1) / floor, 1))`, use separate Q/K/V inputs
//!    and norms, and add input-dependent taps to the K/V convolutions.
//! 5. Experts are routed by sigmoid scores; `exp_probs_b` only biases the
//!    selection. The selected scores are normalized and scaled. Each expert
//!    computes `down(softcap(g, beta) * sigmoid(g) * softcap(u, linear_beta))`.
//! 6. A final attention-residual mix (`output_res_query`), `output_norm`, the
//!    output head, a logit softcap, and the unpadded vocabulary.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::quantized::QTensor;
use candle_core::{D, DType, Device, IndexOp, Module, Tensor};

#[cfg(feature = "cuda")]
use super::inkling2_cuda as cuda;
// The fused per-visit kernels; both modules share these contracts.
#[cfg(all(feature = "cuda", not(target_os = "macos")))]
use super::inkling2_cuda as kernels;
#[cfg(target_os = "macos")]
use super::inkling2_metal as kernels;
use super::raw_experts::RawExperts;
use super::{
    KvCache, LayerCache, ModelArchitecture, PreallocatedKvCache, StageInput, StageOutput,
    copy_persistent_tensor, kv_cache_update_native,
};
use crate::deepseek4_gguf::RawTensor;
use crate::gguf_loader::{GgufMetadata, Inkling2LoopMetadata, StageTensors};
use crate::qmatmul::QMatMul;
use crate::turboquant::TurboQuantConfig;

/// Tokens per internal pass. Longer requests run as consecutive chunks so
/// bias tables, masks and expert routes stay bounded.
const MAX_CHUNK_TOKENS: usize = 512;
/// Upper bound on `heads * queries * keys` for one attention score block.
const MAX_SCORE_ELEMENTS: usize = 1 << 24;
/// Expert route rows per dispatch, as in the other MoE stages.
const MAX_ROUTE_ROWS: usize = 4096;
/// Epsilon of the unweighted norms inside the attention residuals.
const RESIDUAL_NORM_EPS: f64 = 1e-6;
/// Epsilon added to head norms before clipping.
const CLIP_EPS: f64 = 1e-6;
/// Floor of the routed-weight normalizer.
const ROUTE_WEIGHT_EPS: f64 = 1e-20;
/// Chunks longer than this use the tiled tensor-op prompt attention where
/// the device supports it; shorter ones split keys across threadgroups.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
const PROMPT_ATTENTION_MIN_TOKENS: usize = 16;
/// Most tokens for which the MoE post convolution sums the expert routes
/// itself; it recomputes each sum for up to `l_cache` outputs.
#[cfg(any(target_os = "macos", feature = "cuda"))]
const POST_CONV_ROUTE_SUM_MAX_TOKENS: usize = 16;

/// One visit of the layer schedule: the physical layer and its pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Visit {
    pub(crate) layer: usize,
    pub(crate) pass: usize,
}

/// Visit order for one token: prelude once, the core `count` times, the rest once.
pub(crate) fn visit_schedule(
    num_layers: usize,
    start: usize,
    end: usize,
    count: usize,
) -> Vec<Visit> {
    let mut visits = Vec::with_capacity(num_layers + (count.saturating_sub(1)) * (end - start));
    visits.extend((0..start).map(|layer| Visit { layer, pass: 0 }));
    for pass in 0..count {
        visits.extend((start..end).map(|layer| Visit { layer, pass }));
    }
    visits.extend((end..num_layers).map(|layer| Visit { layer, pass: 0 }));
    visits
}

/// State of one layer visit.
pub(crate) struct VisitState {
    /// Keys and values. Sliding visits retain at most `window - 1` tokens
    /// between calls; full visits retain every token.
    kv: Option<KvCache>,
    /// The last `l_cache - 1` inputs of the K, V, post-attention and
    /// post-MoE convolutions, each `[l_cache - 1, channels]` F32.
    conv: [Option<Tensor>; 4],
}

impl VisitState {
    fn empty() -> Self {
        Self {
            kv: None,
            conv: [None, None, None, None],
        }
    }
}

/// Model-wide request state, kept in cache slot zero.
pub struct Inkling2Cache {
    visits: Vec<VisitState>,
    /// Tokens consumed so far; the next token's absolute position.
    position: usize,
}

impl Inkling2Cache {
    fn new(visits: usize) -> Self {
        Self {
            visits: (0..visits).map(|_| VisitState::empty()).collect(),
            position: 0,
        }
    }

    /// Share the current storage. KV appends only write unused rows and the
    /// convolution states are replaced, so this keeps the pre-pass state.
    pub(crate) fn checkpoint(&self) -> Result<Self> {
        let visits = self
            .visits
            .iter()
            .map(|visit| {
                Ok(VisitState {
                    kv: visit.kv.as_ref().map(share_kv).transpose()?,
                    conv: visit.conv.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            visits,
            position: self.position,
        })
    }

    /// Copy the live state into private storage.
    pub(crate) fn fork(&self) -> Result<Self> {
        let visits = self
            .visits
            .iter()
            .map(|visit| {
                let kv = match &visit.kv {
                    None => None,
                    Some(kv) => {
                        let (k, v) = live_kv(kv)?;
                        Some(KvCache::PreallocatedF16(PreallocatedKvCache::from_f16(
                            copy_persistent_tensor(&k)?,
                            copy_persistent_tensor(&v)?,
                        )?))
                    }
                };
                let mut conv = [None, None, None, None];
                for (slot, state) in conv.iter_mut().zip(&visit.conv) {
                    *slot = state.as_ref().map(copy_persistent_tensor).transpose()?;
                }
                Ok(VisitState { kv, conv })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            visits,
            position: self.position,
        })
    }

    /// Bytes held by live keys, values and convolution states.
    pub(crate) fn bytes(&self) -> usize {
        let tensor_bytes = |t: &Tensor| t.elem_count() * t.dtype().size_in_bytes();
        self.visits
            .iter()
            .map(|visit| {
                let kv = visit
                    .kv
                    .as_ref()
                    .and_then(|kv| live_kv(kv).ok())
                    .map_or(0, |(k, v)| tensor_bytes(&k) + tensor_bytes(&v));
                kv + visit.conv.iter().flatten().map(tensor_bytes).sum::<usize>()
            })
            .sum()
    }

    /// Largest KV allocation, in token slots, across visits.
    pub(crate) fn allocation_tokens(&self) -> usize {
        self.visits
            .iter()
            .filter_map(|visit| match &visit.kv {
                Some(KvCache::PreallocatedF16(cache)) => Some(cache.capacity),
                Some(KvCache::F16 { k, .. }) => k.dim(2).ok(),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }
}

fn share_kv(kv: &KvCache) -> Result<KvCache> {
    Ok(match kv {
        KvCache::F16 { k, v } => KvCache::F16 {
            k: k.clone(),
            v: v.clone(),
        },
        KvCache::PreallocatedF16(cache) => KvCache::PreallocatedF16(PreallocatedKvCache {
            k: cache.k.clone(),
            v: cache.v.clone(),
            start: cache.start,
            len: cache.len,
            capacity: cache.capacity,
            allocation_lanes: cache.allocation_lanes,
        }),
        KvCache::Quantized(_) => anyhow::bail!("Inkling 2 does not support quantized KV caches"),
    })
}

fn live_kv(kv: &KvCache) -> Result<(Tensor, Tensor)> {
    match kv {
        KvCache::F16 { k, v } => Ok((k.clone(), v.clone())),
        KvCache::PreallocatedF16(cache) => cache.active_f16(),
        KvCache::Quantized(_) => anyhow::bail!("Inkling 2 does not support quantized KV caches"),
    }
}

/// Model dimensions shared by every layer.
#[derive(Debug, Clone)]
struct Dims {
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rel_dim: usize,
    window: usize,
    rms_eps: f64,
    qk_clip: f64,
    rel_clip: f64,
    log_floor: f64,
    log_alpha: f64,
    situ_beta: f64,
    situ_linear_beta: f64,
    expert_scale: f64,
    active_experts: usize,
    expert_ffn: usize,
}

/// Whether `device` runs the fused Inkling 2 kernels. `DINF_INKLING2_PLAIN=1`
/// keeps the plain tensor path on Metal for comparisons.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn fused(device: &Device) -> bool {
    #[cfg(target_os = "macos")]
    {
        static PLAIN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        device.is_metal()
            && !*PLAIN.get_or_init(|| std::env::var_os("DINF_INKLING2_PLAIN").is_some())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = device;
        false
    }
}

/// Whether `device` runs the CUDA kernels in `inkling2_cuda.rs`: whole
/// visits in `layer_forward_fused`, and the plain helpers below where a
/// visit cannot. `DINF_INKLING2_PLAIN=1` keeps the plain tensor path here too.
#[cfg(feature = "cuda")]
fn cuda_fused(device: &Device) -> bool {
    static PLAIN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    device.is_cuda() && !*PLAIN.get_or_init(|| std::env::var_os("DINF_INKLING2_PLAIN").is_some())
}

/// Whether `device` runs each visit as the fused kernels of `kernels`.
#[cfg(any(target_os = "macos", feature = "cuda"))]
fn fused_layers(device: &Device) -> bool {
    #[cfg(target_os = "macos")]
    {
        fused(device)
    }
    #[cfg(not(target_os = "macos"))]
    {
        cuda_fused(device)
    }
}

/// Short causal convolution weights, one `[1, channels]` row per tap. Tap
/// `l_cache - 1` weights the current token (ggml's `ssm_conv` order).
struct ShortConv {
    taps: Vec<Tensor>,
    /// `[channels, width]`, for the fused kernels.
    #[cfg_attr(not(any(target_os = "macos", feature = "cuda")), allow(dead_code))]
    weight: Tensor,
}

impl ShortConv {
    fn from_weight(weight: &Tensor) -> Result<Self> {
        let weight = weight.contiguous()?;
        let (channels, width) = weight.dims2()?;
        let taps = (0..width)
            .map(|tap| {
                Ok(weight
                    .narrow(1, tap, 1)?
                    .reshape((1, channels))?
                    .contiguous()?)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { taps, weight })
    }

    fn width(&self) -> usize {
        self.taps.len()
    }
}

/// `(x + conv(x) [+ dynamic(x)]) / sqrt(taps + 1)` over `[tokens, channels]`.
///
/// `dynamic` holds `[tokens, heads * width]` per-token taps indexed by
/// backward distance (lag 0 is the current token), shared within each head.
/// `state` carries the last `width - 1` inputs and is replaced on return.
fn short_conv(
    x: &Tensor,
    conv: &ShortConv,
    dynamic: Option<(&Tensor, usize)>,
    state: &mut Option<Tensor>,
) -> Result<Tensor> {
    let (tokens, channels) = x.dims2()?;
    let width = conv.width();
    let past = match state.as_ref() {
        Some(past) => past.clone(),
        None => Tensor::zeros((width - 1, channels), DType::F32, x.device())?,
    };
    #[cfg(feature = "cuda")]
    if dynamic.is_none() && cuda_fused(x.device()) {
        let rows =
            cuda::short_conv_with_state(&x.contiguous()?, &past.contiguous()?, &conv.weight)?;
        let next_state = rows.narrow(0, tokens, width - 1)?;
        *state = Some(if tokens >= width {
            next_state.force_contiguous()?
        } else {
            next_state
        });
        return Ok(rows.narrow(0, 0, tokens)?);
    }
    let history = Tensor::cat(&[&past, x], 0)?;
    // A view of a long history would keep the whole chunk alive in the cache.
    let next_state = history.narrow(0, tokens, width - 1)?;
    let next_state = if tokens >= width {
        next_state.force_contiguous()?
    } else {
        next_state
    };
    #[cfg(feature = "cuda")]
    if cuda_fused(x.device()) {
        let taps = dynamic
            .map(|(taps, heads)| Ok::<_, anyhow::Error>((taps.contiguous()?, heads)))
            .transpose()?;
        let out = cuda::short_conv(
            &history.contiguous()?,
            &conv.weight,
            taps.as_ref().map(|(taps, heads)| (taps, *heads)),
            tokens,
        )?;
        *state = Some(next_state);
        return Ok(out);
    }
    let mut out = x.clone();
    for (tap, weight) in conv.taps.iter().enumerate() {
        out = (out + history.narrow(0, tap, tokens)?.broadcast_mul(weight)?)?;
    }
    let mut terms = width;
    if let Some((weights, heads)) = dynamic {
        let head_width = channels / heads;
        let weights = weights.reshape((tokens, heads, width))?;
        for lag in 0..width {
            let tap = weights.narrow(2, lag, 1)?;
            let window = history
                .narrow(0, width - 1 - lag, tokens)?
                .reshape((tokens, heads, head_width))?;
            out = (out + window.broadcast_mul(&tap)?.reshape((tokens, channels))?)?;
        }
        terms += width;
    }
    *state = Some(next_state);
    Ok((out / ((terms + 1) as f64).sqrt())?)
}

fn rms_unweighted(x: &Tensor, eps: f64) -> Result<Tensor> {
    #[cfg(target_os = "macos")]
    if fused(x.device())
        && let Some(out) =
            super::fused_norm_metal::try_fused_rms_norm_no_weight(&x.contiguous()?, eps as f32)?
    {
        return Ok(out);
    }
    #[cfg(feature = "cuda")]
    if cuda_fused(x.device()) {
        return Ok(cuda::rms_norm(&x.contiguous()?, None, eps as f32)?);
    }
    let mean_square = x.sqr()?.mean_keepdim(D::Minus1)?;
    Ok(x.broadcast_div(&(mean_square + eps)?.sqrt()?)?)
}

fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    #[cfg(target_os = "macos")]
    if fused(x.device())
        && let Some(out) =
            super::fused_norm_metal::try_fused_rms_norm(&x.contiguous()?, weight, eps as f32)?
    {
        return Ok(out);
    }
    #[cfg(feature = "cuda")]
    if cuda_fused(x.device()) {
        return Ok(cuda::rms_norm(&x.contiguous()?, Some(weight), eps as f32)?);
    }
    Ok(rms_unweighted(x, eps)?.broadcast_mul(weight)?)
}

/// `x` `[tokens, heads * dim]` times the per-head `scale`, clipped per head to
/// an L2 norm of `threshold`, times the per-token `token_scale` `[tokens, 1, 1]`
/// and `post_scale`.
fn scaled_clip(
    x: &Tensor,
    heads: usize,
    scale: Option<&Tensor>,
    token_scale: Option<&Tensor>,
    threshold: f64,
    post_scale: f64,
) -> Result<Tensor> {
    let (tokens, width) = x.dims2()?;
    let x = x.contiguous()?.reshape((tokens, heads, width / heads))?;
    #[cfg(feature = "cuda")]
    if cuda_fused(x.device()) {
        return Ok(cuda::scaled_l2_clip(
            &x,
            scale,
            token_scale,
            threshold as f32,
            post_scale as f32,
        )?);
    }
    let x = match scale {
        Some(scale) => x.broadcast_mul(scale)?,
        None => x.clone(),
    };
    let mut x = l2_clip(&x, threshold)?;
    if let Some(token_scale) = token_scale {
        x = x.broadcast_mul(token_scale)?;
    }
    if post_scale != 1.0 {
        x = (x * post_scale)?;
    }
    Ok(x)
}

/// Scale each vector along the last dimension down to an L2 norm of at most `threshold`.
fn l2_clip(x: &Tensor, threshold: f64) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let factor = ((norm + CLIP_EPS)?.recip()? * threshold)?.clamp(0f32, 1f32)?;
    Ok(x.broadcast_mul(&factor)?)
}

fn softcap(x: &Tensor, cap: f64) -> Result<Tensor> {
    Ok(((x / cap)?.tanh()? * cap)?)
}

/// Sum `[tokens, top_k, hidden]` route outputs with `[tokens, top_k]` weights.
fn route_sum(routed: &Tensor, weights: &Tensor) -> Result<Tensor> {
    #[cfg(any(target_os = "macos", feature = "cuda"))]
    if !routed.device().is_cpu() {
        return super::indexed_moe::weighted_route_sum(routed, weights);
    }
    Ok(routed.broadcast_mul(&weights.unsqueeze(2)?)?.sum(1)?)
}

/// Block states for the attention residuals, each `[tokens, hidden]`.
#[derive(Clone, Default)]
struct Blocks {
    items: Vec<Tensor>,
}

impl Blocks {
    fn push(&mut self, block: Tensor) {
        self.items.push(block);
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn tail(&self, from: usize) -> Self {
        Self {
            items: self.items[from..].to_vec(),
        }
    }

    /// `depth_mix` over these blocks and `current`, in one kernel on GPUs.
    fn mix(&self, current: &Tensor, queries: &Tensor) -> Result<Tensor> {
        #[cfg(any(target_os = "macos", feature = "cuda"))]
        if fused_layers(current.device())
            && (1..=kernels::DEPTH_MIX_MAX_BLOCKS).contains(&self.items.len())
            && queries.dim(0)? <= kernels::DEPTH_MIX_MAX_QUERIES
        {
            return Ok(kernels::depth_mix_one(
                &self.items,
                current,
                queries,
                RESIDUAL_NORM_EPS as f32,
                None,
                false,
            )?);
        }
        depth_mix(&self.items, current, queries)
    }

    /// The mix for one query (`[1, hidden]`), RMS-normalized and times
    /// `weight` `[hidden]`; `[tokens, hidden]`.
    fn mix_normed(
        &self,
        current: &Tensor,
        query: &Tensor,
        weight: &Tensor,
        eps: f64,
    ) -> Result<Tensor> {
        #[cfg(any(target_os = "macos", feature = "cuda"))]
        if fused_layers(current.device())
            && (1..=kernels::DEPTH_MIX_MAX_BLOCKS).contains(&self.items.len())
        {
            return Ok(kernels::depth_mix_one(
                &self.items,
                current,
                query,
                RESIDUAL_NORM_EPS as f32,
                Some((&weight.reshape((1, ()))?, eps as f32)),
                false,
            )?
            .squeeze(1)?);
        }
        let mixed = self.mix(current, query)?.i((.., 0))?;
        rms_norm(&mixed, weight, eps)
    }
}

/// Attention over depth: mix `blocks` and `current` (`[tokens, hidden]`)
/// with `queries` (`[queries, hidden]`), returning `[tokens, queries, hidden]`.
/// Keys and values are the RMS-normalized states.
fn depth_mix(blocks: &[Tensor], current: &Tensor, queries: &Tensor) -> Result<Tensor> {
    let (tokens, hidden) = current.dims2()?;
    let query_count = queries.dim(0)?;
    let mut states = blocks.iter().collect::<Vec<_>>();
    states.push(current);
    let sources = states.len();
    let stacked = rms_unweighted(&Tensor::stack(&states, 1)?, RESIDUAL_NORM_EPS)?;
    let scores = stacked
        .reshape((tokens * sources, hidden))?
        .matmul(&queries.t()?.contiguous()?)?
        .reshape((tokens, sources, query_count))?
        .transpose(1, 2)?
        .contiguous()?;
    let probabilities = candle_nn::ops::softmax_last_dim(&(scores / (hidden as f64).sqrt())?)?;
    Ok(probabilities.matmul(&stacked)?)
}

/// Loop message gate: `(1 - tanh(F h)) * x + tanh(G h) * m`, `h = D [rms(x); rms(m)]`.
struct MessageGate {
    /// `[rank, 2 * hidden]`; multiplied as its transpose.
    down: Tensor,
    /// `[rank, hidden]`.
    forget: Tensor,
    /// `[rank, hidden]`.
    gain: Tensor,
    /// `[rank, 2 * hidden]`: forget then gain, for the fused path.
    #[cfg_attr(not(any(target_os = "macos", feature = "cuda")), allow(dead_code))]
    forget_gain: Tensor,
    eps: f64,
}

impl MessageGate {
    fn new(down: Tensor, forget: Tensor, gain: Tensor, eps: f64) -> Result<Self> {
        let forget_gain = Tensor::cat(&[&forget, &gain], 1)?.contiguous()?;
        Ok(Self {
            down,
            forget,
            gain,
            forget_gain,
            eps,
        })
    }

    /// The gate in the fused kernels, then an RMS norm with `weight` `[1, hidden]`.
    #[cfg(all(feature = "cuda", not(target_os = "macos")))]
    fn forward_normed(
        &self,
        x: &Tensor,
        message: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        let gated = cuda::gates_normed(
            &x.unsqueeze(1)?,
            &message.unsqueeze(1)?,
            &self.down.unsqueeze(0)?,
            &self.forget_gain.unsqueeze(0)?,
            self.eps as f32,
            weight,
            eps,
        )?;
        Ok(gated.squeeze(0)?)
    }

    /// The gate in the fused kernels, then an RMS norm with `weight` `[1, hidden]`.
    #[cfg(target_os = "macos")]
    fn forward_normed(
        &self,
        x: &Tensor,
        message: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        let (x, message) = (x.contiguous()?, message.contiguous()?);
        let (tokens, hidden) = x.dims2()?;
        let joined = super::inkling2_metal::rms_pair(&x, &message, self.eps as f32)?;
        let forget_gain = joined.matmul(&self.down.t()?)?.matmul(&self.forget_gain)?;
        let gated = super::inkling2_metal::gate_combine_norm(
            &forget_gain,
            &x.reshape((tokens, 1, hidden))?,
            &message.reshape((tokens, 1, hidden))?,
            weight,
            1,
            eps,
            false,
            false,
        )?;
        Ok(gated.reshape((tokens, hidden))?)
    }

    fn forward(&self, x: &Tensor, message: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if cuda_fused(x.device()) {
            let (x, message) = (x.contiguous()?, message.contiguous()?);
            let joined = cuda::rms_pair(&x, &message, self.eps as f32)?;
            let forget_gain = joined
                .matmul(&self.down.t()?)?
                .matmul(&self.forget_gain)?
                .contiguous()?;
            return Ok(cuda::gate_combine(&forget_gain, &x, &message)?);
        }
        let joined = Tensor::cat(
            &[
                &rms_unweighted(x, self.eps)?,
                &rms_unweighted(message, self.eps)?,
            ],
            D::Minus1,
        )?;
        let reduced = joined.matmul(&self.down.t()?)?;
        let forget = reduced.matmul(&self.forget)?.tanh()?;
        let gain = reduced.matmul(&self.gain)?.tanh()?;
        Ok(((forget.affine(-1.0, 1.0)? * x)? + (gain * message)?)?)
    }
}

/// Looped-core layer weights: message queries (Q, K, V, MoE) and their gates.
struct LoopWeights {
    /// `[4, hidden]`.
    message_query: Tensor,
    gates: [MessageGate; 4],
    /// The Q, K and V gates' down projections, `[3, rank, 2 * hidden]`.
    #[cfg_attr(not(any(target_os = "macos", feature = "cuda")), allow(dead_code))]
    qkv_down: Tensor,
    /// The Q, K and V gates' forget and gain projections, `[3, rank, 2 * hidden]`.
    #[cfg_attr(not(any(target_os = "macos", feature = "cuda")), allow(dead_code))]
    qkv_forget_gain: Tensor,
}

enum AttentionInput {
    /// Layer 0 reads the normalized embedding directly.
    Direct,
    /// One mixed input, `[1, hidden]` query.
    Shared(Tensor),
    /// Separate Q, K and V inputs, `[3, hidden]` queries.
    Separate(Tensor),
}

enum AttentionNorms {
    Sliding(Tensor),
    Full { q: Tensor, k: Tensor, v: Tensor },
}

#[cfg(any(target_os = "macos", feature = "cuda"))]
struct IndexedExperts {
    gate_up: super::indexed_moe::IndexedMoeMatMul,
    down: super::indexed_moe::IndexedMoeMatMul,
}

/// Expert banks in formats Candle lacks (NVFP4, IQ4_NL), with any
/// per-expert output scales.
struct RawBanks {
    gate_up: RawExperts,
    down: RawExperts,
}

/// Routed experts: packed indexed kernels on accelerators, raw-format banks,
/// or one projection per expert on the CPU.
enum Experts {
    #[cfg(any(target_os = "macos", feature = "cuda"))]
    Indexed(Box<IndexedExperts>),
    Raw(Box<RawBanks>),
    PerExpert {
        gate_up: Vec<QMatMul>,
        down: Vec<QMatMul>,
    },
}

/// A dense projection: on Metal 4 the engine's tensor-op quantized matmul
/// (prompt chunks run on the GPU's matrix units), else Candle's.
enum Projection {
    Candle(QMatMul),
    #[cfg(target_os = "macos")]
    Tensor(super::tensor_qmatmul_metal::MetalTensorQMatMul),
    /// BF16 `[out, in]` on CUDA: a BF16 matrix-vector kernel for one token,
    /// cuBLAS BF16 products (the checkpoint's compute type) for more.
    #[cfg(feature = "cuda")]
    Bf16(Tensor),
    /// Stacked parts of more than one quantization type, one projection per
    /// run of a type, with the outputs joined in order. Unsloth's files keep
    /// the sink and conv-tap projections in BF16 beside Q8_0.
    Joined(Vec<Projection>),
}

impl Projection {
    fn new(weight: Arc<QTensor>, device: &Device) -> Result<Self> {
        // BF16 weights stay BF16 on CUDA; Candle would expand them to F32.
        #[cfg(feature = "cuda")]
        if device.is_cuda() && weight.dtype() == candle_core::quantized::GgmlDType::BF16 {
            let dims = weight.shape().dims().to_vec();
            let bytes = weight.data()?;
            return Ok(Self::Bf16(Tensor::from_raw_buffer(
                &bytes,
                DType::BF16,
                &dims,
                device,
            )?));
        }
        #[cfg(target_os = "macos")]
        if device.is_metal()
            && let Some(matmul) = super::tensor_qmatmul_metal::MetalTensorQMatMul::try_from_qtensor(
                Arc::clone(&weight),
                device,
            )?
        {
            return Ok(Self::Tensor(matmul));
        }
        let _ = device;
        Ok(Self::Candle(QMatMul::from_arc(weight)?))
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Candle(matmul) => matmul.forward(x),
            // The tensor-op path takes `[1, tokens, in]`.
            #[cfg(target_os = "macos")]
            Self::Tensor(matmul) if x.rank() == 2 => matmul.forward(&x.unsqueeze(0)?)?.squeeze(0),
            #[cfg(target_os = "macos")]
            Self::Tensor(matmul) => matmul.forward(x),
            #[cfg(feature = "cuda")]
            Self::Bf16(weight) => {
                if x.dim(0)? == 1 && x.rank() == 2 {
                    return cuda::bf16_matvec(&x.contiguous()?, weight);
                }
                x.to_dtype(DType::BF16)?
                    .matmul(&weight.t()?)?
                    .to_dtype(DType::F32)
            }
            Self::Joined(parts) => {
                let outputs = parts
                    .iter()
                    .map(|part| part.forward(x))
                    .collect::<candle_core::Result<Vec<_>>>()?;
                Tensor::cat(&outputs, D::Minus1)
            }
        }
    }

    /// [`Self::forward`] for each `(projection, input)` pair, all issued
    /// before any result is used. On CUDA, up to three BF16 projections of
    /// one token each and one input width run as a single launch.
    #[cfg_attr(not(any(target_os = "macos", feature = "cuda")), allow(dead_code))]
    fn forward_many(pairs: &[(&Self, &Tensor)]) -> candle_core::Result<Vec<Tensor>> {
        #[cfg(feature = "cuda")]
        {
            let weights = pairs
                .iter()
                .map(|(projection, x)| match projection {
                    Self::Bf16(weight) if x.rank() == 2 && x.dim(0).ok() == Some(1) => {
                        Some((*x, weight))
                    }
                    _ => None,
                })
                .collect::<Option<Vec<_>>>();
            if let Some(weights) = weights
                && (2..=3).contains(&weights.len())
                && weights
                    .iter()
                    .all(|(_, w)| w.dim(1).ok() == weights[0].1.dim(1).ok())
            {
                return cuda::bf16_matvec_many(&weights);
            }
        }
        pairs
            .iter()
            .map(|(projection, x)| projection.forward(x))
            .collect()
    }
}

/// Key and value projections, stacked by shared input.
enum KeyValueProjections {
    /// Sliding layers: `[k | v]` for one shared input, and separate `k` and
    /// `v` on looped layers, whose later passes gate the inputs apart.
    Sliding {
        kv: Projection,
        separate: Option<(Projection, Projection)>,
    },
    /// Full layers: `[k | k conv taps]` and `[v | v conv taps]`.
    Full { k: Projection, v: Projection },
}

struct Layer {
    sliding: bool,
    closes_block: bool,
    input: AttentionInput,
    norms: AttentionNorms,
    /// The attention input norms as rows: `[1, hidden]` on sliding layers,
    /// `[3, hidden]` (Q, K, V) on full layers.
    #[cfg_attr(not(any(target_os = "macos", feature = "cuda")), allow(dead_code))]
    norm_rows: Tensor,
    /// Rows `[q | output gate | relative states | sinks]`, which all read
    /// the query input: one matrix-vector product instead of four.
    query_side: Projection,
    key_value: KeyValueProjections,
    output: Projection,
    /// `[heads, head_dim]`.
    q_scale: Tensor,
    /// `[heads, rel_dim]`.
    relative_scale: Tensor,
    /// `[rel_dim, extent + 1]`; the extra zero column is every distance
    /// outside `0..extent`.
    relative_basis: Tensor,
    relative_extent: usize,
    k_conv: ShortConv,
    v_conv: ShortConv,
    attention_post_conv: ShortConv,
    moe_post_conv: ShortConv,
    /// `[1, hidden]`.
    moe_input_query: Tensor,
    moe_norm: Tensor,
    /// `[experts, hidden]`; multiplied as its transpose, which Metal runs
    /// as a fast row-major matrix-vector product.
    router: Tensor,
    /// `[experts]`.
    router_bias: Tensor,
    experts: Experts,
    looped: Option<LoopWeights>,
}

/// Positions of one chunk, shared by every visit of the same attention kind.
struct ChunkPositions {
    start: usize,
    tokens: usize,
    /// `[tokens, 1, 1]` log scales of the full-attention layers.
    log_scale: Tensor,
    /// `(sliding, keys, extent)` → `(bias index, additive mask)`, each `[tokens, keys]`.
    masks: HashMap<(bool, usize, usize), (Tensor, Tensor)>,
}

impl ChunkPositions {
    fn new(start: usize, tokens: usize, dims: &Dims, device: &Device) -> Result<Self> {
        let scales = (0..tokens)
            .map(|t| {
                let ratio = ((start + t + 1) as f64 / dims.log_floor).max(1.0);
                (1.0 + dims.log_alpha * ratio.ln()) as f32
            })
            .collect::<Vec<_>>();
        Ok(Self {
            start,
            tokens,
            log_scale: Tensor::from_vec(scales, (tokens, 1, 1), device)?,
            masks: HashMap::new(),
        })
    }

    /// Bias-table indices and the additive mask for `keys` cached keys whose
    /// newest entry is this chunk's last token.
    fn mask(
        &mut self,
        sliding: bool,
        keys: usize,
        extent: usize,
        window: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        if let Some(found) = self.masks.get(&(sliding, keys, extent)) {
            return Ok(found.clone());
        }
        let (index, mask) = relative_index_and_mask(
            self.start,
            self.tokens,
            keys,
            extent,
            sliding.then_some(window),
        );
        let built = (
            Tensor::from_vec(index, (self.tokens, keys), device)?,
            Tensor::from_vec(mask, (self.tokens, keys), device)?,
        );
        self.masks.insert((sliding, keys, extent), built.clone());
        Ok(built)
    }
}

/// For queries at `start..start + tokens` and `keys` keys ending at the last
/// query, return the relative-bias column for each pair (the distance, or
/// `extent` outside `0..extent`) and an additive causal/window mask.
pub(crate) fn relative_index_and_mask(
    start: usize,
    tokens: usize,
    keys: usize,
    extent: usize,
    window: Option<usize>,
) -> (Vec<u32>, Vec<f32>) {
    let first_key = (start + tokens) as i64 - keys as i64;
    let mut index = Vec::with_capacity(tokens * keys);
    let mut mask = Vec::with_capacity(tokens * keys);
    for t in 0..tokens {
        let query = (start + t) as i64;
        for s in 0..keys {
            let distance = query - (first_key + s as i64);
            let visible = distance >= 0 && window.is_none_or(|w| distance < w as i64);
            index.push(if (0..extent as i64).contains(&distance) {
                distance as u32
            } else {
                extent as u32
            });
            mask.push(if visible { 0.0 } else { f32::NEG_INFINITY });
        }
    }
    (index, mask)
}

/// Softmax attention with an additive relative bias and one zero-value sink
/// logit per query and head.
///
/// `q`: `[heads, tokens, head_dim]` (already scaled by `1/sqrt(head_dim)`),
/// `k`, `v`: `[kv_heads, keys, head_dim]`, `bias`: `[heads, tokens, keys]`,
/// `mask`: `[tokens, keys]`, `sinks`: `[heads, tokens]`. Query head `h` reads
/// key/value head `h / (heads / kv_heads)`. Returns `[heads, tokens, head_dim]`.
pub(crate) fn attend_with_sinks(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    bias: &Tensor,
    mask: &Tensor,
    sinks: &Tensor,
) -> Result<Tensor> {
    let (heads, tokens, head_dim) = q.dims3()?;
    let (kv_heads, keys, _) = k.dims3()?;
    let group = heads / kv_heads;
    let grouped = q.reshape((kv_heads, group * tokens, head_dim))?;
    let scores = grouped
        .matmul(&k.transpose(1, 2)?.contiguous()?)?
        .reshape((heads, tokens, keys))?;
    let scores = (scores + bias)?.broadcast_add(mask)?;
    let logits = Tensor::cat(&[&scores, &sinks.unsqueeze(2)?], 2)?.contiguous()?;
    let probabilities = candle_nn::ops::softmax_last_dim(&logits)?.narrow(2, 0, keys)?;
    let probabilities = probabilities
        .contiguous()?
        .reshape((kv_heads, group * tokens, keys))?;
    Ok(probabilities
        .matmul(v)?
        .reshape((heads, tokens, head_dim))?)
}

/// Inkling 2 Mini as one solo stage.
pub struct Inkling2LoopStage {
    dims: Dims,
    schedule: Vec<Visit>,
    loop_start: usize,
    embedding: Tensor,
    embedding_norm: Tensor,
    layers: Vec<Layer>,
    final_query: Tensor,
    output_norm: Tensor,
    output: Projection,
    logit_softcap: f64,
    vocab_size: usize,
    device: Device,
}

impl Inkling2LoopStage {
    /// Build the full model. Split stages are not supported yet.
    pub fn from_stage_tensors(
        stage: StageTensors,
        quant_config: Option<Arc<TurboQuantConfig>>,
    ) -> Result<Self> {
        let StageTensors {
            range,
            metadata: md,
            mut tensors,
            mut raw_tensors,
            device,
            ..
        } = stage;
        anyhow::ensure!(
            quant_config.is_none(),
            "Inkling 2 does not support quantized KV caches yet"
        );
        anyhow::ensure!(
            range.start == 0 && range.end == md.num_layers && range.is_head() && range.is_tail(),
            "Inkling 2 runs as one solo stage for now"
        );
        let config = md
            .inkling2_loop
            .clone()
            .context("Inkling 2 metadata is missing its loop parameters")?;
        let dims = dims_from_metadata(&md, &config)?;
        #[cfg(feature = "cuda")]
        cuda::precompile(&device)?;
        let mut weights = Weights {
            tensors: &mut tensors,
            raw: &mut raw_tensors,
            device: &device,
        };

        // A BF16 table stays BF16 on CUDA; rows are widened as they are read.
        let bf16_table = device.is_cuda()
            && weights
                .tensors
                .get("token_embd.weight")
                .is_some_and(|table| table.dtype() == candle_core::quantized::GgmlDType::BF16);
        let embedding = if bf16_table {
            let table = weights.take("token_embd.weight")?;
            Tensor::from_raw_buffer(&table.data()?, DType::BF16, table.shape().dims(), &device)?
        } else {
            weights.dense("token_embd.weight")?
        };
        let vocab_rows = embedding.dim(0)?;
        anyhow::ensure!(
            embedding.dims() == [vocab_rows, dims.hidden]
                && vocab_rows >= config.vocab_size_unpadded,
            "Inkling 2 token embedding shape {:?} differs from [{vocab_rows}, {}]",
            embedding.dims(),
            dims.hidden
        );
        let embedding_norm = weights.vector("token_embd_norm.weight", dims.hidden)?;
        let layers = (0..md.num_layers)
            .map(|layer| build_layer(&mut weights, layer, &md, &config, &dims))
            .collect::<Result<Vec<_>>>()?;
        let final_query = weights
            .vector("output_res_query.weight", dims.hidden)?
            .reshape((1, dims.hidden))?;
        let output_norm = weights.vector("output_norm.weight", dims.hidden)?;
        let output = weights.qmatmul("output.weight", vocab_rows, dims.hidden)?;
        if let Some(name) = raw_tensors.keys().next() {
            anyhow::bail!("Inkling 2 does not use tensor {name} in its GGUF type");
        }

        Ok(Self {
            schedule: visit_schedule(
                md.num_layers,
                config.loop_start,
                config.loop_end,
                config.loop_count,
            ),
            loop_start: config.loop_start,
            dims,
            embedding,
            embedding_norm,
            layers,
            final_query,
            output_norm,
            output,
            logit_softcap: md.final_logit_softcapping,
            vocab_size: config.vocab_size_unpadded,
            device,
        })
    }

    fn sliding_visits(&self) -> usize {
        self.schedule
            .iter()
            .filter(|visit| self.layers[visit.layer].sliding)
            .count()
    }

    /// Run every visit for one chunk. Returns the final block states followed
    /// by the last hidden state, each `[tokens, hidden]`, for the output mix.
    fn forward_chunk(&self, tokens: &[u32], state: &mut Inkling2Cache) -> Result<Vec<Tensor>> {
        let count = tokens.len();
        let ids = Tensor::from_slice(tokens, count, &self.device)?;
        let embedded = self.embedding.index_select(&ids, 0)?.to_dtype(DType::F32)?;
        let mut hidden = rms_norm(&embedded, &self.embedding_norm, self.dims.rms_eps)?;
        let mut positions = ChunkPositions::new(state.position, count, &self.dims, &self.device)?;

        let mut blocks = Blocks::default();
        let mut prelude: Option<(Tensor, Blocks)> = None;
        let mut previous: Option<(Tensor, Blocks)> = None;
        let mut current_pass = 0;
        for (index, visit) in self.schedule.iter().enumerate() {
            let layer = &self.layers[visit.layer];
            if visit.layer == self.loop_start {
                if visit.pass == 0 {
                    prelude = Some((hidden.clone(), blocks.clone()));
                } else {
                    let (prelude_hidden, prelude_blocks) = prelude
                        .as_ref()
                        .context("Inkling 2 loop has no prelude state")?;
                    // The first block of each pass is the prelude output
                    // itself, so the message reads the later blocks only.
                    let finished = blocks.tail(prelude_blocks.len() + 1);
                    previous = Some((hidden.clone(), finished));
                    hidden = prelude_hidden.clone();
                    blocks = prelude_blocks.clone();
                }
                current_pass = visit.pass;
            }
            let messages = match (&layer.looped, &mut previous) {
                (Some(looped), Some((last_hidden, last_blocks))) if current_pass > 0 => {
                    Some(last_blocks.mix(last_hidden, &looped.message_query)?)
                }
                _ => None,
            };
            #[cfg(any(target_os = "macos", feature = "cuda"))]
            if fused_layers(&self.device)
                && (matches!(layer.input, AttentionInput::Direct)
                    || (1..=kernels::DEPTH_MIX_MAX_BLOCKS).contains(&blocks.len()))
            {
                hidden = self.layer_forward_fused(
                    layer,
                    &hidden,
                    &mut blocks,
                    messages.as_ref(),
                    &mut state.visits[index],
                    &mut positions,
                )?;
                continue;
            }
            hidden = self.layer_forward(
                layer,
                &hidden,
                &mut blocks,
                messages.as_ref(),
                &mut state.visits[index],
                &mut positions,
            )?;
        }
        state.position += count;
        blocks.items.push(hidden);
        Ok(blocks.items)
    }

    fn layer_forward(
        &self,
        layer: &Layer,
        hidden: &Tensor,
        blocks: &mut Blocks,
        // `[tokens, 4, hidden]`: the Q, K, V and MoE messages.
        messages: Option<&Tensor>,
        visit: &mut VisitState,
        positions: &mut ChunkPositions,
    ) -> Result<Tensor> {
        let eps = self.dims.rms_eps;
        let mixed = match &layer.input {
            AttentionInput::Direct => None,
            AttentionInput::Shared(query) | AttentionInput::Separate(query) => {
                Some(blocks.mix(hidden, query)?)
            }
        };
        let gates = match (messages, &layer.looped) {
            (Some(_), Some(looped)) => Some(looped),
            (Some(_), None) => {
                anyhow::bail!("Inkling 2 message reached a layer without loop gates")
            }
            _ => None,
        };
        let message = |index: usize| -> Result<Tensor> {
            let messages = messages.context("Inkling 2 message is missing")?;
            Ok(messages.i((.., index))?.contiguous()?)
        };
        let split = |rows: &Tensor| -> Result<[Tensor; 3]> {
            Ok([
                rows.i((.., 0))?.contiguous()?,
                rows.i((.., 1))?.contiguous()?,
                rows.i((.., 2))?.contiguous()?,
            ])
        };
        let [q_in, k_in, v_in] = {
            let [mut q_in, mut k_in, mut v_in] = match (&layer.input, &mixed) {
                (AttentionInput::Separate(_), Some(mixed)) => split(mixed)?,
                (_, Some(mixed)) => {
                    let mixed = mixed.i((.., 0))?.contiguous()?;
                    [mixed.clone(), mixed.clone(), mixed]
                }
                (_, None) => [hidden.clone(), hidden.clone(), hidden.clone()],
            };
            if let Some(looped) = gates {
                // Messages enter before the input norms.
                q_in = looped.gates[0].forward(&q_in, &message(0)?)?;
                k_in = looped.gates[1].forward(&k_in, &message(1)?)?;
                v_in = looped.gates[2].forward(&v_in, &message(2)?)?;
            }
            match &layer.norms {
                AttentionNorms::Sliding(weight) if gates.is_some() => [
                    rms_norm(&q_in, weight, eps)?,
                    rms_norm(&k_in, weight, eps)?,
                    rms_norm(&v_in, weight, eps)?,
                ],
                AttentionNorms::Sliding(weight) => {
                    let normed = rms_norm(&q_in, weight, eps)?;
                    [normed.clone(), normed.clone(), normed]
                }
                AttentionNorms::Full { q, k, v } => [
                    rms_norm(&q_in, q, eps)?,
                    rms_norm(&k_in, k, eps)?,
                    rms_norm(&v_in, v, eps)?,
                ],
            }
        };
        let shared_kv = layer.sliding && gates.is_none();
        let attention = self.attention(layer, &q_in, &k_in, &v_in, shared_kv, visit, positions)?;
        let residual = if layer.closes_block {
            blocks.push(hidden.clone());
            attention
        } else {
            (hidden + attention)?
        };

        let mut moe_in = blocks
            .mix(&residual, &layer.moe_input_query)?
            .i((.., 0))?
            .contiguous()?;
        if let Some(looped) = gates {
            moe_in = looped.gates[3].forward(&moe_in, &message(3)?)?;
        }
        let moe_in = rms_norm(&moe_in, &layer.moe_norm, eps)?;
        let routed = self.moe(layer, &moe_in)?;
        let routed = short_conv(&routed, &layer.moe_post_conv, None, &mut visit.conv[3])?;
        let out = (residual + routed)?;
        Ok(out)
    }

    /// [`Self::layer_forward`] in the fused GPU kernels, each run of small
    /// steps one dispatch: the depth mixes with their input norms, the
    /// loop gates with theirs, the query clips with the bias table, the K/V
    /// convolutions with their head norms, and the post convolutions with
    /// the residual adds (and, for a few tokens, the expert route sum).
    #[cfg(any(target_os = "macos", feature = "cuda"))]
    fn layer_forward_fused(
        &self,
        layer: &Layer,
        hidden: &Tensor,
        blocks: &mut Blocks,
        // `[tokens, 4, hidden]`: the Q, K, V and MoE messages.
        messages: Option<&Tensor>,
        visit: &mut VisitState,
        positions: &mut ChunkPositions,
    ) -> Result<Tensor> {
        let eps = self.dims.rms_eps as f32;
        let mix_eps = RESIDUAL_NORM_EPS as f32;
        let gates = match (messages, &layer.looped) {
            (Some(messages), Some(looped)) => Some((looped, messages)),
            (Some(_), None) => {
                anyhow::bail!("Inkling 2 message reached a layer without loop gates")
            }
            _ => None,
        };
        // The attention inputs, each `[tokens, hidden]`.
        let [q_in, k_in, v_in] = match (&layer.input, gates) {
            (AttentionInput::Direct, None) => {
                let AttentionNorms::Sliding(weight) = &layer.norms else {
                    anyhow::bail!("Inkling 2 direct input needs a shared norm")
                };
                let normed = rms_norm(hidden, weight, eps as f64)?;
                [normed.clone(), normed.clone(), normed]
            }
            (AttentionInput::Direct, Some(_)) => {
                anyhow::bail!("Inkling 2 direct input cannot take loop messages")
            }
            (AttentionInput::Shared(query) | AttentionInput::Separate(query), None) => {
                // Group-major [queries, tokens, hidden], normalized.
                let normed = kernels::depth_mix_one(
                    &blocks.items,
                    hidden,
                    query,
                    mix_eps,
                    Some((&layer.norm_rows, eps)),
                    true,
                )?;
                if normed.dim(0)? == 1 {
                    let normed = normed.get(0)?;
                    [normed.clone(), normed.clone(), normed]
                } else {
                    [normed.get(0)?, normed.get(1)?, normed.get(2)?]
                }
            }
            (
                AttentionInput::Shared(query) | AttentionInput::Separate(query),
                Some((looped, messages)),
            ) => {
                // Messages enter before the input norms, all three gates at once.
                let mixed =
                    kernels::depth_mix_one(&blocks.items, hidden, query, mix_eps, None, false)?;
                let normed = kernels::message_gates(
                    &mixed,
                    &messages.narrow(1, 0, 3)?.contiguous()?,
                    &looped.qkv_down,
                    &looped.qkv_forget_gain,
                    eps,
                    &layer.norm_rows,
                )?;
                [normed.get(0)?, normed.get(1)?, normed.get(2)?]
            }
        };
        let shared_kv = layer.sliding && gates.is_none();
        let residual = if layer.closes_block {
            blocks.push(hidden.clone());
            None
        } else {
            Some(hidden)
        };
        let projected =
            self.attention_fused(layer, &q_in, &k_in, &v_in, shared_kv, visit, positions)?;
        let (residual, state) = kernels::post_conv(
            kernels::PostInput::Rows(&projected),
            visit.conv[2].as_ref(),
            &layer.attention_post_conv.weight,
            residual,
        )?;
        visit.conv[2] = Some(state);

        let moe_norm = layer.moe_norm.reshape((1, ()))?;
        let moe_in = match gates {
            None => kernels::depth_mix_one(
                &blocks.items,
                &residual,
                &layer.moe_input_query,
                mix_eps,
                Some((&moe_norm, eps)),
                false,
            )?
            .squeeze(1)?,
            Some((looped, messages)) => {
                let mixed = kernels::depth_mix_one(
                    &blocks.items,
                    &residual,
                    &layer.moe_input_query,
                    mix_eps,
                    None,
                    false,
                )?
                .squeeze(1)?;
                let message = messages.i((.., 3))?.contiguous()?;
                looped.gates[3].forward_normed(&mixed, &message, &moe_norm, eps)?
            }
        };
        self.moe_fused(layer, &moe_in, &residual, visit)
    }

    /// The attention branch in the fused kernels, up to its output projection.
    #[cfg(any(target_os = "macos", feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    fn attention_fused(
        &self,
        layer: &Layer,
        q_in: &Tensor,
        k_in: &Tensor,
        v_in: &Tensor,
        shared_kv: bool,
        visit: &mut VisitState,
        positions: &mut ChunkPositions,
    ) -> Result<Tensor> {
        let dims = &self.dims;
        let tokens = q_in.dim(0)?;
        let (heads, kv_heads, head_dim) = (dims.heads, dims.kv_heads, dims.head_dim);
        let (q_dim, kv_dim, rel_width) =
            (heads * head_dim, kv_heads * head_dim, heads * dims.rel_dim);
        // The projections all read the attention inputs, so issuing them
        // before any consumer lets the GPU run them side by side (on CUDA,
        // as one launch for a token).
        let taps_width = kv_heads * layer.k_conv.width();
        let pairs = match &layer.key_value {
            KeyValueProjections::Sliding { kv, .. } if shared_kv => {
                vec![(&layer.query_side, q_in), (kv, k_in)]
            }
            KeyValueProjections::Sliding {
                separate: Some((k, v)),
                ..
            }
            | KeyValueProjections::Full { k, v } => {
                vec![(&layer.query_side, q_in), (k, k_in), (v, v_in)]
            }
            KeyValueProjections::Sliding { separate: None, .. } => {
                anyhow::bail!("Inkling 2 sliding layer has no separate key and value projections")
            }
        };
        let mut projected = Projection::forward_many(&pairs)?.into_iter();
        let mut next = || projected.next().context("Inkling 2 projection is missing");
        let query_side = next()?;
        let (k_lin, v_lin, taps) = match &layer.key_value {
            KeyValueProjections::Sliding { .. } if shared_kv => {
                // The fused kernel reads column slices in place.
                let both = next()?;
                (
                    both.narrow(1, 0, kv_dim)?,
                    both.narrow(1, kv_dim, kv_dim)?,
                    None,
                )
            }
            KeyValueProjections::Sliding { .. } => (next()?, next()?, None),
            KeyValueProjections::Full { .. } => {
                let (k, v) = (next()?, next()?);
                (
                    k.narrow(1, 0, kv_dim)?,
                    v.narrow(1, 0, kv_dim)?,
                    Some((
                        k.narrow(1, kv_dim, taps_width)?,
                        v.narrow(1, kv_dim, taps_width)?,
                    )),
                )
            }
        };
        let gate_logits = query_side.narrow(1, q_dim, q_dim)?;
        let log_scale = (!layer.sliding).then_some(&positions.log_scale);
        // Queries carry the 1/sqrt(head_dim) score scale from here on.
        let (q, bias) = kernels::query_prep(
            &query_side,
            &kernels::QuerySide {
                q: 0,
                rel: 2 * q_dim,
                sink: 2 * q_dim + rel_width,
            },
            heads,
            head_dim,
            &layer.q_scale,
            &layer.relative_scale,
            log_scale,
            &layer.relative_basis,
            dims.qk_clip as f32,
            dims.rel_clip as f32,
            1.0 / (head_dim as f32).sqrt(),
        )?;

        let past = match (&visit.conv[0], &visit.conv[1]) {
            (Some(k), Some(v)) => Some((k.clone(), v.clone())),
            _ => None,
        };
        let finish = |output: kernels::KvOutput<'_>| {
            kernels::kv_finish(
                &k_lin,
                &v_lin,
                past.as_ref().map(|(k, v)| (k, v)),
                &layer.k_conv.weight,
                &layer.v_conv.weight,
                taps.as_ref().map(|(k, v)| (k, v)),
                kv_heads,
                dims.qk_clip as f32,
                dims.rms_eps as f32,
                output,
            )
        };
        // Once a visit has a cache, the kernel writes its f16 rows in place.
        let (finished, k_all, v_all) = match visit.kv.as_mut() {
            Some(KvCache::PreallocatedF16(cache)) => {
                let (k_cache, v_cache, row) = cache.reserve(tokens)?;
                let finished = finish(kernels::KvOutput::Cache {
                    k: &k_cache,
                    v: &v_cache,
                    row,
                })?;
                cache.commit(tokens)?;
                let (k_all, v_all) = cache.active_f16()?;
                (finished, k_all, v_all)
            }
            _ => {
                let mut finished = finish(kernels::KvOutput::Fresh)?;
                let (k, v) = (finished.k.take(), finished.v.take());
                let (k, v) = k.zip(v).context("Inkling 2 key/value output is missing")?;
                let (k_all, v_all) = kv_cache_update_native(&mut visit.kv, k, v, None)?;
                (finished, k_all, v_all)
            }
        };
        visit.conv[0] = Some(finished.k_state);
        visit.conv[1] = Some(finished.v_state);
        let keys = k_all.dim(2)?;

        let gated = match self.fused_attention(
            layer,
            &q,
            &bias,
            &k_all,
            &v_all,
            &gate_logits,
            tokens,
            keys,
            positions,
        )? {
            Some(gated) => gated,
            None => {
                let extent = layer.relative_extent;
                let table = bias.narrow(2, 0, extent + 1)?.contiguous()?;
                let sinks = bias.narrow(2, extent + 1, 1)?.squeeze(2)?.contiguous()?;
                let attended = self.plain_attention(
                    layer, &q, &table, &sinks, &k_all, &v_all, tokens, keys, positions,
                )?;
                (attended * candle_nn::ops::sigmoid(&gate_logits)?)?
            }
        };
        let projected = layer.output.forward(&gated)?;

        if layer.sliding {
            // Later queries can only see the last `window - 1` keys.
            let keep = dims.window - 1;
            if keys > keep
                && let Some(kv) = visit.kv.as_mut()
            {
                kv.evict_oldest(keys - keep)?;
            }
        }
        Ok(projected)
    }

    /// The MoE branch on Metal plus `residual`.
    #[cfg(target_os = "macos")]
    fn moe_fused(
        &self,
        layer: &Layer,
        x: &Tensor,
        residual: &Tensor,
        visit: &mut VisitState,
    ) -> Result<Tensor> {
        use super::inkling2_metal as metal;
        let dims = &self.dims;
        let (tokens, hidden) = x.dims2()?;
        let finish = |input: metal::PostInput<'_>, visit: &mut VisitState| -> Result<Tensor> {
            let (out, state) = metal::post_conv(
                input,
                visit.conv[3].as_ref(),
                &layer.moe_post_conv.weight,
                Some(residual),
            )?;
            visit.conv[3] = Some(state);
            Ok(out)
        };
        match &layer.experts {
            Experts::Indexed(experts) if tokens * dims.active_experts <= MAX_ROUTE_ROWS => {
                let logits = x.matmul(&layer.router.t()?)?;
                // Unique in-range IDs: the expert kernels skip their sanitize pass.
                let (ids, weights) = metal::sigmoid_route(
                    &logits,
                    &layer.router_bias,
                    dims.active_experts,
                    dims.expert_scale as f32,
                )?;
                let input = x.reshape((tokens, 1, hidden))?;
                let gate_up = experts
                    .gate_up
                    .forward_metal_sanitized(&input, &ids, false)?;
                let activated = metal::situ(
                    &gate_up.contiguous()?,
                    dims.situ_beta as f32,
                    dims.situ_linear_beta as f32,
                )?;
                let routed = experts
                    .down
                    .forward_metal_sanitized(&activated, &ids, false)?;
                if tokens <= POST_CONV_ROUTE_SUM_MAX_TOKENS {
                    finish(
                        metal::PostInput::Routes {
                            routed: &routed,
                            weights: &weights,
                        },
                        visit,
                    )
                } else {
                    let summed = super::indexed_moe::weighted_route_sum(&routed, &weights)?;
                    finish(metal::PostInput::Rows(&summed), visit)
                }
            }
            _ => {
                let routed = self.moe(layer, x)?;
                finish(metal::PostInput::Rows(&routed), visit)
            }
        }
    }

    /// The MoE branch in the CUDA kernels plus `residual`.
    #[cfg(all(feature = "cuda", not(target_os = "macos")))]
    fn moe_fused(
        &self,
        layer: &Layer,
        x: &Tensor,
        residual: &Tensor,
        visit: &mut VisitState,
    ) -> Result<Tensor> {
        let dims = &self.dims;
        let (tokens, hidden) = x.dims2()?;
        let finish = |input: cuda::PostInput<'_>, visit: &mut VisitState| -> Result<Tensor> {
            let (out, state) = cuda::post_conv(
                input,
                visit.conv[3].as_ref(),
                &layer.moe_post_conv.weight,
                Some(residual),
            )?;
            visit.conv[3] = Some(state);
            Ok(out)
        };
        let banked = matches!(layer.experts, Experts::Raw(_) | Experts::Indexed(_));
        if !banked || tokens * dims.active_experts > MAX_ROUTE_ROWS {
            let routed = self.moe(layer, x)?;
            return finish(cuda::PostInput::Rows(&routed), visit);
        }
        // The router emits unique in-range U32 IDs.
        let (ids, weights) = self.route(layer, x)?;
        let input = x.reshape((tokens, 1, hidden))?;
        let activate = |gate_up: &Tensor| -> Result<Tensor> {
            Ok(cuda::situ(
                &gate_up.contiguous()?,
                dims.situ_beta as f32,
                dims.situ_linear_beta as f32,
            )?)
        };
        let routed = match &layer.experts {
            Experts::Raw(banks) => {
                let gate_up = banks.gate_up.forward(&input, &ids)?;
                banks.down.forward(&activate(&gate_up)?, &ids)?
            }
            Experts::Indexed(experts) => {
                let gate_up = experts.gate_up.forward(&input, &ids)?;
                experts.down.forward(&activate(&gate_up)?, &ids)?
            }
            _ => unreachable!("banked experts were matched above"),
        };
        if tokens <= POST_CONV_ROUTE_SUM_MAX_TOKENS {
            finish(
                cuda::PostInput::Routes {
                    routed: &routed,
                    weights: &weights,
                },
                visit,
            )
        } else {
            let summed = route_sum(&routed, &weights)?;
            finish(cuda::PostInput::Rows(&summed), visit)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        layer: &Layer,
        q_in: &Tensor,
        k_in: &Tensor,
        v_in: &Tensor,
        shared_kv: bool,
        visit: &mut VisitState,
        positions: &mut ChunkPositions,
    ) -> Result<Tensor> {
        let dims = &self.dims;
        let tokens = q_in.dim(0)?;
        let (heads, kv_heads, head_dim) = (dims.heads, dims.kv_heads, dims.head_dim);
        let (q_dim, kv_dim, rel_width) =
            (heads * head_dim, kv_heads * head_dim, heads * dims.rel_dim);
        let query_side = layer.query_side.forward(q_in)?;
        // Single-token slices are already contiguous; longer ones are copied.
        let slice = |t: &Tensor, offset: usize, width: usize| -> Result<Tensor> {
            Ok(t.narrow(1, offset, width)?.contiguous()?)
        };
        let gate_logits = query_side.narrow(1, q_dim, q_dim)?;
        let sinks = slice(&query_side, 2 * q_dim + rel_width, heads)?;

        let log_scale = (!layer.sliding).then_some(&positions.log_scale);
        // Queries carry the 1/sqrt(head_dim) score scale from here on.
        let q = scaled_clip(
            &query_side.narrow(1, 0, q_dim)?,
            heads,
            Some(&layer.q_scale),
            log_scale,
            dims.qk_clip,
            1.0 / (head_dim as f64).sqrt(),
        )?;
        let relative = scaled_clip(
            &query_side.narrow(1, 2 * q_dim, rel_width)?,
            heads,
            Some(&layer.relative_scale),
            log_scale,
            dims.rel_clip,
            1.0,
        )?;

        let taps_width = kv_heads * layer.k_conv.width();
        let (k_lin, v_lin, k_taps, v_taps) = match &layer.key_value {
            KeyValueProjections::Sliding { kv, .. } if shared_kv => {
                // The fused convolution reads column slices in place.
                let both = kv.forward(k_in)?;
                (
                    both.narrow(1, 0, kv_dim)?,
                    both.narrow(1, kv_dim, kv_dim)?,
                    None,
                    None,
                )
            }
            KeyValueProjections::Sliding {
                separate: Some((k, v)),
                ..
            } => (k.forward(k_in)?, v.forward(v_in)?, None, None),
            KeyValueProjections::Sliding { separate: None, .. } => {
                anyhow::bail!("Inkling 2 sliding layer has no separate key and value projections")
            }
            KeyValueProjections::Full { k, v } => {
                let (k, v) = (k.forward(k_in)?, v.forward(v_in)?);
                (
                    slice(&k, 0, kv_dim)?,
                    slice(&v, 0, kv_dim)?,
                    Some(slice(&k, kv_dim, taps_width)?),
                    Some(slice(&v, kv_dim, taps_width)?),
                )
            }
        };
        let k = short_conv(
            &k_lin,
            &layer.k_conv,
            k_taps.as_ref().map(|taps| (taps, kv_heads)),
            &mut visit.conv[0],
        )?;
        let v = short_conv(
            &v_lin,
            &layer.v_conv,
            v_taps.as_ref().map(|taps| (taps, kv_heads)),
            &mut visit.conv[1],
        )?;
        let k = scaled_clip(&k, kv_heads, None, None, dims.qk_clip, 1.0)?;
        let v = rms_unweighted(&v.reshape((tokens, kv_heads, head_dim))?, dims.rms_eps)?;

        let (k_all, v_all) = kv_cache_update_native(
            &mut visit.kv,
            k.transpose(0, 1)?.unsqueeze(0)?.contiguous()?,
            v.transpose(0, 1)?.unsqueeze(0)?.contiguous()?,
            None,
        )?;
        let keys = k_all.dim(2)?;

        let table = relative
            .reshape((tokens * heads, dims.rel_dim))?
            .matmul(&layer.relative_basis)?
            .reshape((tokens, heads, layer.relative_extent + 1))?;
        #[cfg(feature = "cuda")]
        let fused_gated = self.cuda_attention(
            layer,
            &q,
            &table,
            &sinks,
            &k_all,
            &v_all,
            &gate_logits,
            tokens,
            positions.start,
        )?;
        #[cfg(not(feature = "cuda"))]
        let fused_gated = None;
        let gated = match fused_gated {
            Some(gated) => gated,
            None => {
                let attended = self.plain_attention(
                    layer, &q, &table, &sinks, &k_all, &v_all, tokens, keys, positions,
                )?;
                (attended * candle_nn::ops::sigmoid(&gate_logits)?)?
            }
        };
        let projected = layer.output.forward(&gated)?;
        let out = short_conv(
            &projected,
            &layer.attention_post_conv,
            None,
            &mut visit.conv[2],
        )?;

        if layer.sliding {
            // Later queries can only see the last `window - 1` keys.
            let keep = dims.window - 1;
            if keys > keep
                && let Some(kv) = visit.kv.as_mut()
            {
                kv.evict_oldest(keys - keep)?;
            }
        }
        Ok(out)
    }

    /// Attention in the GPU kernels reading the f16 cache: on Metal,
    /// tensor-op tiles for prompt chunks and keys split across threadgroups
    /// otherwise; on CUDA, split keys. Returns `[tokens, heads * head_dim]`
    /// with the output gate applied, or `None` where the kernels do not fit.
    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(any(target_os = "macos", feature = "cuda")), allow(dead_code))]
    fn fused_attention(
        &self,
        layer: &Layer,
        q: &Tensor,
        bias: &Tensor,
        k_all: &Tensor,
        v_all: &Tensor,
        gate_logits: &Tensor,
        tokens: usize,
        keys: usize,
        positions: &ChunkPositions,
    ) -> Result<Option<Tensor>> {
        #[cfg(target_os = "macos")]
        {
            let dims = &self.dims;
            let strides = k_all.stride();
            if !fused(&self.device)
                || tokens > super::inkling2_metal::FUSED_ATTENTION_MAX_TOKENS
                || !dims.head_dim.is_multiple_of(32)
                || dims.head_dim > 256
                || k_all.dtype() != DType::F16
                || v_all.dtype() != DType::F16
                || strides.len() != 4
                || strides[3] != 1
                || strides[2] != dims.head_dim
                || v_all.stride() != strides
            {
                return Ok(None);
            }
            let first_key = positions.start + tokens - keys;
            if tokens > PROMPT_ATTENTION_MIN_TOKENS
                && let Some(attended) = super::inkling2_flash_metal::try_attention(
                    &q.to_dtype(DType::F16)?,
                    k_all,
                    v_all,
                    bias,
                    layer.relative_extent,
                    layer.sliding.then_some(dims.window),
                    positions.start - first_key,
                )?
            {
                return Ok(Some(super::inkling2_metal::gate_transpose(
                    &attended,
                    gate_logits,
                )?));
            }
            let attended = super::inkling2_metal::attention(
                q,
                bias,
                k_all,
                v_all,
                dims.head_dim,
                layer.relative_extent,
                layer.sliding.then_some(dims.window),
                positions.start,
                first_key,
                Some(gate_logits),
            )?;
            Ok(Some(
                attended.reshape((tokens, dims.heads * dims.head_dim))?,
            ))
        }
        #[cfg(all(feature = "cuda", not(target_os = "macos")))]
        {
            if !self.cuda_attention_fits(k_all, v_all, tokens) {
                return Ok(None);
            }
            let attended = cuda::attention(
                q,
                bias,
                k_all,
                v_all,
                self.dims.head_dim,
                layer.relative_extent,
                layer.sliding.then_some(self.dims.window),
                positions.start,
                positions.start + tokens - keys,
                Some(gate_logits),
            )?;
            Ok(Some(
                attended.reshape((tokens, self.dims.heads * self.dims.head_dim))?,
            ))
        }
        #[cfg(not(any(target_os = "macos", feature = "cuda")))]
        {
            let _ = (
                layer,
                q,
                bias,
                k_all,
                v_all,
                gate_logits,
                tokens,
                keys,
                positions,
            );
            Ok(None)
        }
    }

    /// Whether the CUDA split-key attention takes this cache and chunk.
    #[cfg(feature = "cuda")]
    fn cuda_attention_fits(&self, k_all: &Tensor, v_all: &Tensor, tokens: usize) -> bool {
        let dims = &self.dims;
        let strides = k_all.stride();
        cuda_fused(&self.device)
            && tokens <= cuda::FUSED_ATTENTION_MAX_TOKENS
            && matches!(dims.head_dim, 64 | 128 | 256)
            && k_all.dtype() == DType::F16
            && v_all.dtype() == DType::F16
            && strides.len() == 4
            && strides[3] == 1
            && strides[2] == dims.head_dim
            && v_all.stride() == strides
    }

    /// Attention in the CUDA split-key kernel, straight from the f16 cache, for
    /// up to [`cuda::FUSED_ATTENTION_MAX_TOKENS`] queries. Returns `[tokens,
    /// heads * head_dim]` with the output gate applied, or `None` when the
    /// device or cache layout does not fit the kernel.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn cuda_attention(
        &self,
        layer: &Layer,
        q: &Tensor,
        table: &Tensor,
        sinks: &Tensor,
        k_all: &Tensor,
        v_all: &Tensor,
        gate_logits: &Tensor,
        tokens: usize,
        start: usize,
    ) -> Result<Option<Tensor>> {
        let dims = &self.dims;
        if !self.cuda_attention_fits(k_all, v_all, tokens) {
            return Ok(None);
        }
        let keys = k_all.dim(2)?;
        let bias = Tensor::cat(&[table, &sinks.reshape((tokens, dims.heads, 1))?], 2)?;
        let attended = cuda::attention(
            q,
            &bias,
            k_all,
            v_all,
            dims.head_dim,
            layer.relative_extent,
            layer.sliding.then_some(dims.window),
            start,
            start + tokens - keys,
            Some(gate_logits),
        )?;
        Ok(Some(
            attended.reshape((tokens, dims.heads * dims.head_dim))?,
        ))
    }

    /// Attention with matrix products over F32 keys and values, in query blocks.
    #[allow(clippy::too_many_arguments)]
    fn plain_attention(
        &self,
        layer: &Layer,
        q: &Tensor,
        table: &Tensor,
        sinks: &Tensor,
        k_all: &Tensor,
        v_all: &Tensor,
        tokens: usize,
        keys: usize,
        positions: &mut ChunkPositions,
    ) -> Result<Tensor> {
        let dims = &self.dims;
        let (heads, head_dim) = (dims.heads, dims.head_dim);
        let k_all = k_all.squeeze(0)?.to_dtype(DType::F32)?.contiguous()?;
        let v_all = v_all.squeeze(0)?.to_dtype(DType::F32)?.contiguous()?;
        let sinks = sinks.t()?.contiguous()?;
        let table = table.transpose(0, 1)?.contiguous()?;
        let q = q.transpose(0, 1)?.contiguous()?;
        let (index, mask) = positions.mask(
            layer.sliding,
            keys,
            layer.relative_extent,
            dims.window,
            &self.device,
        )?;
        let block = (MAX_SCORE_ELEMENTS / (heads * keys)).clamp(1, tokens);
        let mut outputs = Vec::with_capacity(tokens.div_ceil(block));
        for first in (0..tokens).step_by(block) {
            let len = block.min(tokens - first);
            let index = index
                .narrow(0, first, len)?
                .unsqueeze(0)?
                .broadcast_as((heads, len, keys))?
                .contiguous()?;
            let bias = table
                .narrow(1, first, len)?
                .contiguous()?
                .gather(&index, 2)?;
            outputs.push(attend_with_sinks(
                &q.narrow(1, first, len)?.contiguous()?,
                &k_all,
                &v_all,
                &bias,
                &mask.narrow(0, first, len)?,
                &sinks.narrow(1, first, len)?.contiguous()?,
            )?);
        }
        let attended = if outputs.len() == 1 {
            outputs.pop().expect("one attention block")
        } else {
            Tensor::cat(&outputs, 1)?
        };
        Ok(attended
            .transpose(0, 1)?
            .contiguous()?
            .reshape((tokens, heads * head_dim))?)
    }

    /// Selected experts (`[tokens, active]` u32) and their scaled weights.
    fn route(&self, layer: &Layer, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let dims = &self.dims;
        // Sigmoid scores, biased selection, normalized unbiased weights.
        #[cfg(feature = "cuda")]
        if cuda_fused(x.device()) {
            return Ok(cuda::route(
                x,
                &layer.router,
                &layer.router_bias,
                dims.active_experts,
                dims.expert_scale as f32,
            )?);
        }
        let logits = x.matmul(&layer.router.t()?)?;
        let scores = candle_nn::ops::sigmoid(&logits)?;
        let ids = scores
            .broadcast_add(&layer.router_bias)?
            .arg_sort_last_dim(false)?
            .narrow(1, 0, dims.active_experts)?
            .contiguous()?;
        let weights = scores.gather(&ids, 1)?;
        let weights = (weights.broadcast_div(&(weights.sum_keepdim(1)? + ROUTE_WEIGHT_EPS)?)?
            * dims.expert_scale)?;
        Ok((ids, weights))
    }

    fn moe(&self, layer: &Layer, x: &Tensor) -> Result<Tensor> {
        let dims = &self.dims;
        let (tokens, hidden) = x.dims2()?;
        let (ids, weights) = self.route(layer, x)?;
        let activate = |gate_up: &Tensor| -> Result<Tensor> {
            #[cfg(target_os = "macos")]
            if fused(gate_up.device()) {
                return Ok(super::inkling2_metal::situ(
                    &gate_up.contiguous()?,
                    dims.situ_beta as f32,
                    dims.situ_linear_beta as f32,
                )?);
            }
            #[cfg(feature = "cuda")]
            if cuda_fused(gate_up.device()) {
                return Ok(cuda::situ(
                    &gate_up.contiguous()?,
                    dims.situ_beta as f32,
                    dims.situ_linear_beta as f32,
                )?);
            }
            let half = dims.expert_ffn;
            let gate = gate_up.narrow(D::Minus1, 0, half)?;
            let up = gate_up.narrow(D::Minus1, half, half)?;
            Ok(
                ((softcap(&gate, dims.situ_beta)? * candle_nn::ops::sigmoid(&gate)?)?
                    * softcap(&up, dims.situ_linear_beta)?)?,
            )
        };
        match &layer.experts {
            #[cfg(any(target_os = "macos", feature = "cuda"))]
            Experts::Indexed(experts) => {
                use super::indexed_moe::weighted_route_sum;
                let chunk = (MAX_ROUTE_ROWS / dims.active_experts).max(1);
                let mut outputs = Vec::with_capacity(tokens.div_ceil(chunk));
                for first in (0..tokens).step_by(chunk) {
                    let len = chunk.min(tokens - first);
                    let input = x.narrow(0, first, len)?.reshape((len, 1, hidden))?;
                    let chunk_ids = ids.narrow(0, first, len)?.contiguous()?;
                    let project = |projection: &super::indexed_moe::IndexedMoeMatMul,
                                   input: &Tensor|
                     -> Result<Tensor> {
                        // The fused router emits unique in-range IDs.
                        #[cfg(target_os = "macos")]
                        if fused(input.device()) {
                            return projection.forward_metal_sanitized(input, &chunk_ids, false);
                        }
                        projection.forward(input, &chunk_ids)
                    };
                    let gate_up = project(&experts.gate_up, &input)?;
                    let activated = activate(&gate_up)?.contiguous()?;
                    let routed = project(&experts.down, &activated)?;
                    outputs.push(weighted_route_sum(
                        &routed,
                        &weights.narrow(0, first, len)?.contiguous()?,
                    )?);
                }
                if outputs.len() == 1 {
                    Ok(outputs.pop().expect("one route chunk"))
                } else {
                    Ok(Tensor::cat(&outputs, 0)?)
                }
            }
            Experts::Raw(banks) => {
                let ids = if ids.dtype() == DType::U32 {
                    ids
                } else {
                    ids.to_dtype(DType::U32)?
                };
                let chunk = (MAX_ROUTE_ROWS / dims.active_experts).max(1);
                let mut outputs = Vec::with_capacity(tokens.div_ceil(chunk));
                for first in (0..tokens).step_by(chunk) {
                    let len = chunk.min(tokens - first);
                    let input = x.narrow(0, first, len)?.reshape((len, 1, hidden))?;
                    let chunk_ids = ids.narrow(0, first, len)?.contiguous()?;
                    let gate_up = banks
                        .gate_up
                        .forward(&input, &chunk_ids)?
                        .to_device(x.device())?;
                    let activated = activate(&gate_up)?.contiguous()?;
                    let routed = banks
                        .down
                        .forward(&activated, &chunk_ids)?
                        .to_device(x.device())?;
                    outputs.push(route_sum(
                        &routed,
                        &weights.narrow(0, first, len)?.contiguous()?,
                    )?);
                }
                if outputs.len() == 1 {
                    Ok(outputs.pop().expect("one route chunk"))
                } else {
                    Ok(Tensor::cat(&outputs, 0)?)
                }
            }
            Experts::PerExpert { gate_up, down } => {
                let ids_host = ids.to_vec2::<u32>()?;
                let weights_host = weights.to_vec2::<f32>()?;
                let mut rows: Vec<Vec<(u32, f32)>> = vec![Vec::new(); gate_up.len()];
                for (token, (token_ids, token_weights)) in
                    ids_host.iter().zip(&weights_host).enumerate()
                {
                    for (&expert, &weight) in token_ids.iter().zip(token_weights) {
                        rows[expert as usize].push((token as u32, weight));
                    }
                }
                let mut out = Tensor::zeros((tokens, hidden), DType::F32, x.device())?;
                for (expert, routes) in rows.iter().enumerate() {
                    if routes.is_empty() {
                        continue;
                    }
                    let token_ids = routes.iter().map(|(t, _)| *t).collect::<Vec<_>>();
                    let route_weights = routes.iter().map(|(_, w)| *w).collect::<Vec<_>>();
                    let token_ids = Tensor::from_vec(token_ids, routes.len(), x.device())?;
                    let route_weights =
                        Tensor::from_vec(route_weights, (routes.len(), 1), x.device())?;
                    let selected = x.index_select(&token_ids, 0)?;
                    let activated = activate(&gate_up[expert].forward(&selected)?)?;
                    let projected = down[expert]
                        .forward(&activated)?
                        .broadcast_mul(&route_weights)?;
                    out = out.index_add(&token_ids, &projected, 0)?;
                }
                Ok(out)
            }
        }
    }

    fn forward_inner(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
        want_logits: bool,
    ) -> Result<StageOutput> {
        let tokens = match input {
            StageInput::Tokens(tokens) => tokens,
            StageInput::Activations { .. } => {
                anyhow::bail!("Inkling 2 takes token IDs; embedding inputs are not supported yet")
            }
        };
        anyhow::ensure!(
            !tokens.is_empty() && tokens.len() == seq_len,
            "Inkling 2 received {} tokens for a {seq_len}-token pass",
            tokens.len()
        );
        anyhow::ensure!(
            tokens
                .iter()
                .all(|&token| (token as usize) < self.vocab_size),
            "Inkling 2 token ID is outside the vocabulary"
        );
        let slot = cache
            .first_mut()
            .context("Inkling 2 needs one cache slot")?;
        // Work on a checkpoint so an error leaves the request state untouched.
        let mut state = match slot.as_ref() {
            None => Inkling2Cache::new(self.schedule.len()),
            Some(LayerCache::Inkling2(existing)) => existing.checkpoint()?,
            Some(_) => anyhow::bail!("Inkling 2 cache slot holds another model's state"),
        };
        anyhow::ensure!(
            state.position == start_pos,
            "Inkling 2 cache holds {} tokens but the pass starts at {start_pos}",
            state.position
        );

        let mut last = None;
        for chunk in tokens.chunks(MAX_CHUNK_TOKENS) {
            last = Some(self.forward_chunk(chunk, &mut state)?);
        }
        let states = last.context("Inkling 2 pass had no tokens")?;
        *slot = Some(LayerCache::Inkling2(Box::new(state)));

        let final_rows = states
            .iter()
            .map(|state| {
                let rows = state.dim(0)?;
                Ok(state.narrow(0, rows - 1, 1)?.contiguous()?)
            })
            .collect::<Result<Vec<_>>>()?;
        let (current, finished) = final_rows.split_last().expect("hidden state");
        if !want_logits {
            return Ok(StageOutput::Activations(current.clone()));
        }
        let finished = Blocks {
            items: finished.to_vec(),
        };
        let normed = finished.mix_normed(
            current,
            &self.final_query,
            &self.output_norm,
            self.dims.rms_eps,
        )?;
        let logits = self
            .output
            .forward(&normed)?
            .narrow(1, 0, self.vocab_size)?
            .squeeze(0)?;
        let logits = if self.logit_softcap > 0.0 {
            softcap(&logits, self.logit_softcap)?
        } else {
            logits
        };
        let logits = logits.contiguous()?;
        Ok(StageOutput::Logits(logits))
    }
}

impl ModelArchitecture for Inkling2LoopStage {
    fn fixed_cache_bytes(&self, _verification_rows: usize) -> Result<usize> {
        // Sliding visits retain one window of f16 keys and values; every
        // visit keeps three rows of four F32 convolution inputs.
        let dims = &self.dims;
        let kv_row = 2 * dims.kv_heads * dims.head_dim * 2;
        let sliding = self.sliding_visits() * (dims.window + MAX_CHUNK_TOKENS) * kv_row;
        let conv_channels = 2 * dims.kv_heads * dims.head_dim + 2 * dims.hidden;
        let conv_rows = self
            .layers
            .first()
            .map_or(0, |layer| layer.k_conv.width().saturating_sub(1));
        let conv = self.schedule.len() * conv_rows * conv_channels * 4;
        Ok(sliding + conv)
    }

    fn forward(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_inner(input, cache, seq_len, start_pos, true)
    }

    fn forward_prefill_cache_only(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.forward_inner(input, cache, seq_len, start_pos, false)
    }

    fn embed_tokens(&self, tokens: &[u32]) -> Result<Tensor> {
        let ids = Tensor::from_slice(tokens, tokens.len(), &self.device)?;
        Ok(self.embedding.index_select(&ids, 0)?.to_dtype(DType::F32)?)
    }

    fn family(&self) -> &str {
        "inkling2_loop"
    }

    fn hidden_dim(&self) -> usize {
        self.dims.hidden
    }

    fn layer_count(&self) -> usize {
        self.layers.len()
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

fn dims_from_metadata(md: &GgufMetadata, config: &Inkling2LoopMetadata) -> Result<Dims> {
    anyhow::ensure!(
        md.num_attention_heads > 0
            && md.num_kv_heads > 0
            && md.num_attention_heads.is_multiple_of(md.num_kv_heads)
            && md.head_dim > 0,
        "Inkling 2 attention heads are invalid"
    );
    anyhow::ensure!(
        md.head_count_kv_arr
            .iter()
            .all(|&heads| heads == md.num_kv_heads),
        "Inkling 2 needs the same KV head count on every layer"
    );
    anyhow::ensure!(
        md.expert_count > 0
            && md.expert_used_count > 0
            && md.expert_used_count <= md.expert_count
            && md.expert_ffn_dim > 0
            && md.expert_weights_scale > 0.0,
        "Inkling 2 expert metadata is invalid"
    );
    Ok(Dims {
        hidden: md.hidden_dim,
        heads: md.num_attention_heads,
        kv_heads: md.num_kv_heads,
        head_dim: md.head_dim,
        rel_dim: config.relative_length,
        window: md.sliding_window,
        rms_eps: md.rms_norm_eps,
        qk_clip: config.qk_clip,
        rel_clip: config.relative_clip,
        log_floor: config.log_scale_floor as f64,
        log_alpha: config.log_scale_alpha,
        situ_beta: config.situ_beta,
        situ_linear_beta: config.situ_linear_beta,
        expert_scale: md.expert_weights_scale,
        active_experts: md.expert_used_count,
        expert_ffn: md.expert_ffn_dim,
    })
}

struct Weights<'a> {
    tensors: &'a mut HashMap<String, Arc<QTensor>>,
    /// Tensors in GGUF types Candle lacks (NVFP4 and IQ4_NL expert banks).
    raw: &'a mut HashMap<String, Arc<RawTensor>>,
    device: &'a Device,
}

impl Weights<'_> {
    fn take(&mut self, name: &str) -> Result<Arc<QTensor>> {
        self.tensors
            .remove(name)
            .with_context(|| format!("Inkling 2 is missing tensor {name}"))
    }

    fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn dense(&mut self, name: &str) -> Result<Tensor> {
        Ok(self
            .take(name)?
            .dequantize(self.device)?
            .to_dtype(DType::F32)?
            .contiguous()?)
    }

    fn shaped(&mut self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let tensor = self.dense(name)?;
        anyhow::ensure!(
            tensor.dims() == shape,
            "Inkling 2 tensor {name} has shape {:?}, expected {shape:?}",
            tensor.dims()
        );
        Ok(tensor)
    }

    fn vector(&mut self, name: &str, len: usize) -> Result<Tensor> {
        self.shaped(name, &[len])
    }

    fn rows(&mut self, name: &str, rows: usize, cols: usize) -> Result<Arc<QTensor>> {
        let tensor = self.take(name)?;
        anyhow::ensure!(
            tensor.shape().dims() == [rows, cols],
            "Inkling 2 tensor {name} has shape {:?}, expected [{rows}, {cols}]",
            tensor.shape().dims()
        );
        Ok(tensor)
    }

    /// One projection whose output rows are `parts` in order. Each output row
    /// is computed independently, so the results equal separate products.
    /// Parts of one quantization type share one matrix; a stack of several
    /// types is [`Projection::Joined`].
    fn stacked(&mut self, parts: &[Arc<QTensor>]) -> Result<Projection> {
        let first = parts
            .first()
            .context("Inkling 2 stacks at least one projection")?;
        let cols = first.shape().dims()[1];
        anyhow::ensure!(
            parts.iter().all(|part| part.shape().dims()[1] == cols),
            "Inkling 2 stacked projections differ in width"
        );
        let mut runs = parts
            .chunk_by(|a, b| a.dtype() == b.dtype())
            .map(|run| self.stack_run(run, cols))
            .collect::<Result<Vec<_>>>()?;
        Ok(if runs.len() == 1 {
            runs.pop().expect("one run")
        } else {
            Projection::Joined(runs)
        })
    }

    /// One matrix of `run`'s rows; every part has one quantization type.
    fn stack_run(&self, run: &[Arc<QTensor>], cols: usize) -> Result<Projection> {
        if let [only] = run {
            return Projection::new(Arc::clone(only), self.device);
        }
        let mut bytes = Vec::new();
        let mut rows = 0;
        for part in run {
            bytes.extend_from_slice(&part.data()?);
            rows += part.shape().dims()[0];
        }
        Projection::new(
            Arc::new(candle_core::quantized::ggml_file::qtensor_from_ggml(
                run[0].dtype(),
                &bytes,
                vec![rows, cols],
                self.device,
            )?),
            self.device,
        )
    }

    fn qmatmul(&mut self, name: &str, rows: usize, cols: usize) -> Result<Projection> {
        let tensor = self.take(name)?;
        anyhow::ensure!(
            tensor.shape().dims() == [rows, cols],
            "Inkling 2 tensor {name} has shape {:?}, expected [{rows}, {cols}]",
            tensor.shape().dims()
        );
        Projection::new(tensor, self.device)
    }
}

fn build_layer(
    weights: &mut Weights<'_>,
    layer: usize,
    md: &GgufMetadata,
    config: &Inkling2LoopMetadata,
    dims: &Dims,
) -> Result<Layer> {
    let p = |name: &str| format!("blk.{layer}.{name}");
    let sliding = *md
        .sliding_window_pattern
        .get(layer)
        .context("Inkling 2 sliding-window pattern is too short")?;
    let (hidden, heads, kv_heads, head_dim) =
        (dims.hidden, dims.heads, dims.kv_heads, dims.head_dim);
    let q_dim = heads * head_dim;
    let kv_dim = kv_heads * head_dim;
    let width = config.shortconv_l_cache;

    let input = if layer == 0 {
        anyhow::ensure!(
            !weights.has(&p("attn_res_query.weight")),
            "Inkling 2 layer 0 must read the embedding directly"
        );
        AttentionInput::Direct
    } else if sliding {
        AttentionInput::Shared(
            weights
                .vector(&p("attn_res_query.weight"), hidden)?
                .reshape((1, hidden))?,
        )
    } else {
        AttentionInput::Separate(weights.shaped(&p("attn_res_query.weight"), &[3, hidden])?)
    };
    let norms = if sliding {
        AttentionNorms::Sliding(weights.vector(&p("attn_norm.weight"), hidden)?)
    } else {
        AttentionNorms::Full {
            q: weights.vector(&p("attn_q_inp_norm.weight"), hidden)?,
            k: weights.vector(&p("attn_k_inp_norm.weight"), hidden)?,
            v: weights.vector(&p("attn_v_inp_norm.weight"), hidden)?,
        }
    };

    let relative_bias = weights.dense(&p("attn_rel_bias.weight"))?;
    let (relative_extent, rel_dim) = relative_bias.dims2()?;
    anyhow::ensure!(
        rel_dim == dims.rel_dim
            && relative_extent
                == if sliding {
                    dims.window
                } else {
                    config.relative_extent
                },
        "Inkling 2 {} relative basis has shape {:?}",
        p("attn_rel_bias.weight"),
        relative_bias.dims()
    );
    let relative_basis = Tensor::cat(
        &[
            &relative_bias,
            &Tensor::zeros((1, rel_dim), DType::F32, weights.device)?,
        ],
        0,
    )?
    .t()?
    .contiguous()?;

    let conv = |weights: &mut Weights<'_>, name: &str, channels: usize| -> Result<ShortConv> {
        ShortConv::from_weight(&weights.shaped(&p(name), &[channels, width])?)
    };
    let query_side = {
        let parts = [
            weights.rows(&p("attn_q.weight"), q_dim, hidden)?,
            weights.rows(&p("attn_gate.weight"), q_dim, hidden)?,
            weights.rows(&p("attn_rel.weight"), heads * rel_dim, hidden)?,
            weights.rows(&p("attn_sink_proj.weight"), heads, hidden)?,
        ];
        weights.stacked(&parts)?
    };
    let looped_layer = (config.loop_start..config.loop_end).contains(&layer);
    let k = weights.rows(&p("attn_k.weight"), kv_dim, hidden)?;
    let v = weights.rows(&p("attn_v.weight"), kv_dim, hidden)?;
    let key_value = if sliding {
        KeyValueProjections::Sliding {
            kv: weights.stacked(&[Arc::clone(&k), Arc::clone(&v)])?,
            separate: if looped_layer {
                Some((
                    Projection::new(k, weights.device)?,
                    Projection::new(v, weights.device)?,
                ))
            } else {
                None
            },
        }
    } else {
        let k_taps = weights.rows(&p("attn_k_conv_proj.weight"), kv_heads * width, hidden)?;
        let v_taps = weights.rows(&p("attn_v_conv_proj.weight"), kv_heads * width, hidden)?;
        KeyValueProjections::Full {
            k: weights.stacked(&[k, k_taps])?,
            v: weights.stacked(&[v, v_taps])?,
        }
    };

    let experts = md.expert_count;
    let router = weights.shaped(&p("ffn_gate_inp.weight"), &[experts, hidden])?;
    let router_bias = if weights.has(&p("exp_probs_b.bias")) {
        weights.vector(&p("exp_probs_b.bias"), experts)?
    } else {
        Tensor::zeros(experts, DType::F32, weights.device)?
    };
    // Per-expert output scales come with raw (NVFP4) banks (llama.cpp #32657).
    let mut expert_scales = |name: &str| -> Result<Option<Vec<f32>>> {
        if !weights.has(name) {
            return Ok(None);
        }
        let scales = weights.vector(name, experts)?;
        Ok(Some(scales.to_device(&Device::Cpu)?.to_vec1::<f32>()?))
    };
    let gate_up_scales = expert_scales(&p("ffn_gate_up_exps.scale"))?;
    let down_scales = expert_scales(&p("ffn_down_exps.scale"))?;
    let (gate_up_name, down_name) = (p("ffn_gate_up_exps.weight"), p("ffn_down_exps.weight"));
    let experts = match (
        weights.raw.remove(&gate_up_name),
        weights.raw.remove(&down_name),
    ) {
        (Some(gate_up), Some(down)) => {
            let ffn = dims.expert_ffn;
            // MXFP8 banks carry their block scales in `.mx_scale`.
            let mut bank = |raw: Arc<RawTensor>,
                            name: &str,
                            scales: Option<Vec<f32>>,
                            out_dim: usize,
                            in_dim: usize|
             -> Result<RawExperts> {
                if raw.dtype() != crate::deepseek4_gguf::DeepSeek4DType::F8E4M3 {
                    return RawExperts::new(&raw, scales, experts, out_dim, in_dim);
                }
                anyhow::ensure!(scales.is_none(), "MXFP8 experts have no outer scale");
                let scale_name = format!("{}.mx_scale", name.trim_end_matches(".weight"));
                let block_scales = weights
                    .raw
                    .remove(&scale_name)
                    .with_context(|| format!("Inkling 2 is missing {scale_name}"))?;
                RawExperts::mxfp8(
                    &raw,
                    &block_scales,
                    experts,
                    out_dim,
                    in_dim,
                    weights.device,
                )
            };
            Experts::Raw(Box::new(RawBanks {
                gate_up: bank(gate_up, &gate_up_name, gate_up_scales, 2 * ffn, hidden)
                    .with_context(|| format!("Inkling 2 {gate_up_name}"))?,
                down: bank(down, &down_name, down_scales, hidden, ffn)
                    .with_context(|| format!("Inkling 2 {down_name}"))?,
            }))
        }
        (None, None) => {
            anyhow::ensure!(
                gate_up_scales.is_none() && down_scales.is_none(),
                "Inkling 2 layer {layer} has per-expert scales on Candle-type experts"
            );
            let gate_up = weights.take(&gate_up_name)?;
            let down = weights.take(&down_name)?;
            build_experts(&gate_up, &down, experts, dims, weights.device)?
        }
        _ => anyhow::bail!("Inkling 2 layer {layer} mixes raw and Candle expert types"),
    };

    let looped = if (config.loop_start..config.loop_end).contains(&layer) {
        let rank = config.loop_gate_rank;
        let mut gate = |name: &str| -> Result<MessageGate> {
            let down = weights.shaped(
                &p(&format!("loop_{name}_ctrl_down.weight")),
                &[rank, 2 * hidden],
            )?;
            let mut matrix = |part: &str, shape: &[usize]| -> Result<Tensor> {
                Ok(weights
                    .shaped(&p(&format!("loop_{name}_ctrl_{part}.weight")), shape)?
                    .t()?
                    .contiguous()?)
            };
            let forget = matrix("forget", &[hidden, rank])?;
            let gain = matrix("gain", &[hidden, rank])?;
            MessageGate::new(down, forget, gain, dims.rms_eps)
        };
        let gates = [gate("q")?, gate("k")?, gate("v")?, gate("ffn")?];
        let qkv_down = Tensor::stack(&[&gates[0].down, &gates[1].down, &gates[2].down], 0)?;
        let qkv_forget_gain = Tensor::stack(
            &[
                &gates[0].forget_gain,
                &gates[1].forget_gain,
                &gates[2].forget_gain,
            ],
            0,
        )?;
        Some(LoopWeights {
            message_query: weights.shaped(&p("loop_msg_query.weight"), &[4, hidden])?,
            gates,
            qkv_down,
            qkv_forget_gain,
        })
    } else {
        None
    };

    Ok(Layer {
        sliding,
        closes_block: layer.is_multiple_of(config.attn_res_block_size),
        input,
        norm_rows: match &norms {
            AttentionNorms::Sliding(weight) => weight.reshape((1, hidden))?,
            AttentionNorms::Full { q, k, v } => Tensor::stack(&[q, k, v], 0)?,
        },
        norms,
        query_side,
        key_value,
        output: weights.qmatmul(&p("attn_output.weight"), hidden, q_dim)?,
        q_scale: weights.shaped(&p("attn_q_clip.weight"), &[heads, head_dim])?,
        relative_scale: weights.shaped(&p("attn_rel_clip.weight"), &[heads, rel_dim])?,
        relative_basis,
        relative_extent,
        k_conv: conv(weights, "attn_k_conv.weight", kv_dim)?,
        v_conv: conv(weights, "attn_v_conv.weight", kv_dim)?,
        attention_post_conv: conv(weights, "attn_post_conv.weight", hidden)?,
        moe_post_conv: conv(weights, "ffn_post_conv.weight", hidden)?,
        moe_input_query: weights
            .vector(&p("ffn_res_query.weight"), hidden)?
            .reshape((1, hidden))?,
        moe_norm: weights.vector(&p("ffn_norm.weight"), hidden)?,
        router,
        router_bias,
        experts,
        looped,
    })
}

/// A host F16 copy of a BF16 expert bank, when the device's indexed expert
/// kernels read F16 but not BF16. Both are two bytes per weight; F16 keeps
/// three more mantissa bits and loses range far outside expert weight values.
#[cfg(any(target_os = "macos", feature = "cuda"))]
fn f16_bank_for_indexed(packed: &QTensor, device: &Device) -> Result<Option<QTensor>> {
    use super::indexed_moe::indexed_moe_available;
    use candle_core::quantized::GgmlDType;
    if packed.dtype() != GgmlDType::BF16
        || indexed_moe_available(device, GgmlDType::BF16)
        || !indexed_moe_available(device, GgmlDType::F16)
    {
        return Ok(None);
    }
    let values = packed
        .dequantize(device)?
        .to_dtype(DType::F16)?
        .flatten_all()?
        .to_vec1::<half::f16>()?;
    let bytes = values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect::<Vec<u8>>();
    Ok(Some(candle_core::quantized::ggml_file::qtensor_from_ggml(
        GgmlDType::F16,
        &bytes,
        packed.shape().dims().to_vec(),
        &Device::Cpu,
    )?))
}

fn build_experts(
    gate_up: &QTensor,
    down: &QTensor,
    experts: usize,
    dims: &Dims,
    device: &Device,
) -> Result<Experts> {
    let (hidden, ffn) = (dims.hidden, dims.expert_ffn);
    anyhow::ensure!(
        gate_up.shape().dims() == [experts, 2 * ffn, hidden]
            && down.shape().dims() == [experts, hidden, ffn],
        "Inkling 2 expert shapes {:?} / {:?} differ from [{experts}, {}, {hidden}] / [{experts}, {hidden}, {ffn}]",
        gate_up.shape().dims(),
        down.shape().dims(),
        2 * ffn
    );
    #[cfg(any(target_os = "macos", feature = "cuda"))]
    {
        use super::indexed_moe::{IndexedMoeMatMul, indexed_moe_available};
        let gate_up_f16 = f16_bank_for_indexed(gate_up, device)?;
        let down_f16 = f16_bank_for_indexed(down, device)?;
        let gate_up = gate_up_f16.as_ref().unwrap_or(gate_up);
        let down = down_f16.as_ref().unwrap_or(down);
        // The indexed kernels read rows in whole 32-wide blocks.
        if hidden.is_multiple_of(32)
            && ffn.is_multiple_of(32)
            && indexed_moe_available(device, gate_up.dtype())
            && indexed_moe_available(device, down.dtype())
        {
            #[allow(unused_mut)]
            let mut gate_up =
                IndexedMoeMatMul::from_qtensor(gate_up, experts, 2 * ffn, hidden, device)?;
            #[allow(unused_mut)]
            let mut down = IndexedMoeMatMul::from_qtensor(down, experts, hidden, ffn, device)?;
            // CUDA prompts use the tensor-core kernels where they cover the
            // format, and Q8_0 decode the exact F32 kernel.
            #[cfg(feature = "cuda")]
            if device.is_cuda() {
                gate_up.prefer_cuda_grouped_wmma()?;
                down.prefer_cuda_grouped_wmma()?;
            }
            return Ok(Experts::Indexed(Box::new(IndexedExperts { gate_up, down })));
        }
    }
    let split = |packed: &QTensor, rows: usize, cols: usize| -> Result<Vec<QMatMul>> {
        let dtype = packed.dtype();
        anyhow::ensure!(
            cols.is_multiple_of(dtype.block_size()),
            "Inkling 2 expert rows are not quantization aligned"
        );
        let bytes = rows * cols / dtype.block_size() * dtype.type_size();
        let raw = packed.data()?;
        anyhow::ensure!(
            raw.len() == experts * bytes,
            "Inkling 2 expert storage length differs from its shape"
        );
        (0..experts)
            .map(|expert| {
                let tensor = candle_core::quantized::ggml_file::qtensor_from_ggml(
                    dtype,
                    &raw[expert * bytes..(expert + 1) * bytes],
                    vec![rows, cols],
                    device,
                )?;
                Ok(QMatMul::from_qtensor(tensor)?)
            })
            .collect()
    };
    Ok(Experts::PerExpert {
        gate_up: split(gate_up, 2 * ffn, hidden)?,
        down: split(down, hidden, ffn)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_tensor(values: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_slice(values, shape, &Device::Cpu).unwrap()
    }

    /// A six-layer synthetic model: blocks of two, core layers 2..4 looped
    /// three times, a four-token window, and log scaling from position 4.
    fn tiny_stage() -> Result<Inkling2LoopStage> {
        tiny_stage_with(&Device::Cpu, 16, 4, 2, 8)
    }

    /// The synthetic model on `device`, with the given attention geometry.
    fn tiny_stage_with(
        device: &Device,
        hidden: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Inkling2LoopStage> {
        tiny_stage_full(device, hidden, heads, kv_heads, head_dim, false)
    }

    /// [`tiny_stage_with`], optionally with NVFP4 expert banks and per-expert
    /// scales, which need `hidden` and the experts' width to be multiples of 64.
    fn tiny_stage_full(
        device: &Device,
        hidden: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        nvfp4: bool,
    ) -> Result<Inkling2LoopStage> {
        use crate::gguf_loader::{LayerRange, parse_gguf_metadata};
        use candle_core::quantized::{GgmlDType, gguf_file::Value};

        let (layers, rel) = (6usize, 4usize);
        let (window, extent, width, experts, vocab, rank) = (4usize, 6, 4, 4, 32, 4);
        // 64-wide experts let accelerator stages run the indexed expert kernels.
        let ffn = if device.is_cpu() && hidden == 16 {
            8
        } else {
            64
        };
        let p = |key: &str| format!("inkling2_loop.{key}");
        let md = HashMap::from([
            (
                "general.architecture".to_string(),
                Value::String("inkling2_loop".into()),
            ),
            (p("block_count"), Value::U32(layers as u32)),
            (p("context_length"), Value::U32(4096)),
            (p("embedding_length"), Value::U32(hidden as u32)),
            (p("attention.head_count"), Value::U32(heads as u32)),
            (
                p("attention.head_count_kv"),
                Value::Array(vec![Value::U32(kv_heads as u32); layers]),
            ),
            (p("attention.key_length"), Value::U32(head_dim as u32)),
            (p("attention.layer_norm_rms_epsilon"), Value::F32(1e-6)),
            (p("vocab_size"), Value::U32(vocab as u32)),
            (p("vocab_size_unpadded"), Value::U32(30)),
            (p("expert_count"), Value::U32(experts as u32)),
            (p("expert_used_count"), Value::U32(2)),
            (p("expert_feed_forward_length"), Value::U32(ffn as u32)),
            (p("attention.sliding_window"), Value::U32(window as u32)),
            (
                p("attention.sliding_window_pattern"),
                Value::Array((0..layers).map(|i| Value::Bool((i + 1) % 2 != 0)).collect()),
            ),
            (p("shortconv.l_cache"), Value::U32(width as u32)),
            (p("expert_weights_scale"), Value::F32(2.0)),
            (p("activation.situ_beta"), Value::F32(4.0)),
            (p("activation.situ_linear_beta"), Value::F32(25.0)),
            (p("attn_res.block_size"), Value::U32(2)),
            (p("final_logit_softcapping"), Value::F32(30.0)),
            (p("attention.relative_length"), Value::U32(rel as u32)),
            (p("attention.relative_extent"), Value::U32(extent as u32)),
            (p("attention.qk_clip"), Value::F32(2.0)),
            (p("attention.relative_clip"), Value::F32(1.5)),
            (p("attention.log_scale_floor"), Value::U32(4)),
            (p("attention.log_scale_alpha"), Value::F32(0.5)),
            (p("loop.start"), Value::U32(2)),
            (p("loop.end"), Value::U32(4)),
            (p("loop.count"), Value::U32(3)),
            (p("loop.gate_rank"), Value::U32(rank as u32)),
        ]);
        let metadata = parse_gguf_metadata(&md)?;

        let mut seed = 0x9e37_79b9_u32;
        let mut tensors = HashMap::new();
        let mut raw_tensors = HashMap::new();
        let mut add = |name: String, shape: &[usize], scale: f32| {
            let values = (0..shape.iter().product::<usize>())
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
                })
                .collect::<Vec<_>>();
            let tensor = Tensor::from_vec(values, shape, device).unwrap();
            let qt = QTensor::quantize(&tensor, GgmlDType::F32).unwrap();
            tensors.insert(name, Arc::new(qt));
        };
        let fan = |n: usize| 1.0 / (n as f32).sqrt();
        add("token_embd.weight".into(), &[vocab, hidden], 1.0);
        add("token_embd_norm.weight".into(), &[hidden], 1.0);
        add("output.weight".into(), &[vocab, hidden], fan(hidden));
        add("output_norm.weight".into(), &[hidden], 1.0);
        add("output_res_query.weight".into(), &[hidden], 1.0);
        for layer in 0..layers {
            let b = |name: &str| format!("blk.{layer}.{name}");
            let sliding = (layer + 1) % 2 != 0;
            let (q, kv) = (heads * head_dim, kv_heads * head_dim);
            add(b("attn_q.weight"), &[q, hidden], fan(hidden));
            add(b("attn_k.weight"), &[kv, hidden], fan(hidden));
            add(b("attn_v.weight"), &[kv, hidden], fan(hidden));
            add(b("attn_output.weight"), &[hidden, q], fan(q));
            add(b("attn_gate.weight"), &[q, hidden], fan(hidden));
            add(b("attn_q_clip.weight"), &[heads, head_dim], 1.0);
            add(b("attn_sink_proj.weight"), &[heads, hidden], fan(hidden));
            add(b("attn_rel.weight"), &[heads * rel, hidden], fan(hidden));
            add(
                b("attn_rel_bias.weight"),
                &[if sliding { window } else { extent }, rel],
                1.0,
            );
            add(b("attn_rel_clip.weight"), &[heads, rel], 1.0);
            add(b("attn_k_conv.weight"), &[kv, width], 0.5);
            add(b("attn_v_conv.weight"), &[kv, width], 0.5);
            add(b("attn_post_conv.weight"), &[hidden, width], 0.5);
            add(b("ffn_post_conv.weight"), &[hidden, width], 0.5);
            if sliding {
                add(b("attn_norm.weight"), &[hidden], 1.0);
            } else {
                for n in ["q", "k", "v"] {
                    add(b(&format!("attn_{n}_inp_norm.weight")), &[hidden], 1.0);
                }
                add(
                    b("attn_k_conv_proj.weight"),
                    &[kv_heads * width, hidden],
                    fan(hidden),
                );
                add(
                    b("attn_v_conv_proj.weight"),
                    &[kv_heads * width, hidden],
                    fan(hidden),
                );
            }
            if layer > 0 {
                if sliding {
                    add(b("attn_res_query.weight"), &[hidden], 1.0);
                } else {
                    add(b("attn_res_query.weight"), &[3, hidden], 1.0);
                }
            }
            add(b("ffn_res_query.weight"), &[hidden], 1.0);
            add(b("ffn_norm.weight"), &[hidden], 1.0);
            add(b("ffn_gate_inp.weight"), &[experts, hidden], 1.0);
            add(b("exp_probs_b.bias"), &[experts], 0.1);
            if nvfp4 {
                let raw_device = if device.is_cuda() {
                    device
                } else {
                    &Device::Cpu
                };
                for (name, dims, bank_seed) in [
                    ("ffn_gate_up_exps", [experts, 2 * ffn, hidden], 100),
                    ("ffn_down_exps", [experts, hidden, ffn], 200),
                ] {
                    let bytes = super::super::raw_experts::tests::random_bank(
                        dims[0],
                        dims[1],
                        dims[2],
                        bank_seed + layer as u32,
                    );
                    let len = bytes.len();
                    let raw = RawTensor::new(
                        dims.to_vec(),
                        crate::deepseek4_gguf::DeepSeek4DType::NVFP4,
                        Tensor::from_vec(bytes, len, raw_device)?,
                    )?;
                    raw_tensors.insert(b(&format!("{name}.weight")), Arc::new(raw));
                    // NVFP4 values reach six times their scale; scale down.
                    add(b(&format!("{name}.scale")), &[experts], 0.05);
                }
            } else {
                add(
                    b("ffn_gate_up_exps.weight"),
                    &[experts, 2 * ffn, hidden],
                    fan(hidden),
                );
                add(b("ffn_down_exps.weight"), &[experts, hidden, ffn], fan(ffn));
            }
            if (2..4).contains(&layer) {
                add(b("loop_msg_query.weight"), &[4, hidden], 1.0);
                for n in ["q", "k", "v", "ffn"] {
                    add(
                        b(&format!("loop_{n}_ctrl_down.weight")),
                        &[rank, 2 * hidden],
                        fan(2 * hidden),
                    );
                    add(
                        b(&format!("loop_{n}_ctrl_forget.weight")),
                        &[hidden, rank],
                        fan(rank),
                    );
                    add(
                        b(&format!("loop_{n}_ctrl_gain.weight")),
                        &[hidden, rank],
                        fan(rank),
                    );
                }
            }
        }
        Inkling2LoopStage::from_stage_tensors(
            StageTensors {
                range: LayerRange::try_solo(layers)?,
                metadata,
                tensors,
                raw_tensors,
                raw_source: None,
                device: device.clone(),
            },
            None,
        )
    }

    fn run_chunks(stage: &Inkling2LoopStage, tokens: &[u32], chunks: &[usize]) -> Result<Vec<f32>> {
        let mut cache: Vec<Option<LayerCache>> = (0..stage.layer_count()).map(|_| None).collect();
        let mut start = 0;
        let mut logits = None;
        for &len in chunks {
            let part = tokens[start..start + len].to_vec();
            logits = Some(stage.forward(StageInput::Tokens(part), &mut cache, len, start)?);
            start += len;
        }
        assert_eq!(start, tokens.len());
        match logits.unwrap() {
            StageOutput::Logits(logits) => Ok(logits.to_vec1::<f32>()?),
            StageOutput::Activations(_) => anyhow::bail!("expected logits"),
        }
    }

    /// The whole fused Metal model against the plain CPU path on the same
    /// synthetic weights, through the few-query and tensor-op attention.
    #[cfg(target_os = "macos")]
    #[test]
    fn fused_stage_matches_plain_stage() -> Result<()> {
        let Some(metal) = metal_device() else {
            return Ok(());
        };
        let cpu = tiny_stage_with(&Device::Cpu, 64, 2, 1, 128)?;
        let gpu = tiny_stage_with(&metal, 64, 2, 1, 128)?;
        for (length, chunks) in [
            (13usize, vec![13usize]),
            (13, vec![5, 4, 4]),
            (13, vec![1; 13]),
            (40, vec![40]),
            (40, vec![24, 16]),
        ] {
            let tokens = (0..length as u32)
                .map(|i| (i * 7 + 3) % 30)
                .collect::<Vec<_>>();
            let expected = run_chunks(&cpu, &tokens, &[length])?;
            let actual = run_chunks(&gpu, &tokens, &chunks)?;
            let error = expected
                .iter()
                .zip(&actual)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                error < 5e-3,
                "{length} tokens in {chunks:?}: max logit difference {error}"
            );
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn fused_stage_is_deterministic() -> Result<()> {
        let Some(metal) = metal_device() else {
            return Ok(());
        };
        let gpu = tiny_stage_with(&metal, 64, 2, 1, 128)?;
        let tokens = (0..13u32).map(|i| (i * 7 + 3) % 30).collect::<Vec<_>>();
        let first = run_chunks(&gpu, &tokens, &[13])?;
        for _ in 0..4 {
            let again = run_chunks(&gpu, &tokens, &[13])?;
            assert!(
                first
                    .iter()
                    .zip(&again)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "fused logits changed between identical runs"
            );
        }
        Ok(())
    }

    #[test]
    fn chunked_prefill_matches_one_pass() -> Result<()> {
        let stage = tiny_stage()?;
        assert_eq!(stage.schedule.len(), 10);
        let tokens = (0..13u32).map(|i| (i * 7 + 3) % 30).collect::<Vec<_>>();
        let whole = run_chunks(&stage, &tokens, &[13])?;
        assert_eq!(whole.len(), 30, "padded rows are dropped");
        for chunks in [
            &[5usize, 4, 4][..],
            &[1; 13][..],
            &[12, 1][..],
            &[3, 7, 3][..],
        ] {
            let split = run_chunks(&stage, &tokens, chunks)?;
            let diff = whole
                .iter()
                .zip(&split)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                diff < 1e-4,
                "chunks {chunks:?}: max logit difference {diff}"
            );
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn metal_device() -> Option<Device> {
        match Device::new_metal(0) {
            Ok(device) => Some(device),
            Err(error) if std::env::var_os("DINF_REQUIRE_METAL_TESTS").is_none() => {
                eprintln!("Metal unavailable: {error}");
                None
            }
            Err(error) => panic!("Metal required: {error}"),
        }
    }

    #[cfg(any(target_os = "macos", feature = "cuda"))]
    fn random(shape: &[usize], seed: u32, scale: f32) -> Tensor {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        let values = (0..shape.iter().product::<usize>())
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, shape, &Device::Cpu).unwrap()
    }

    #[cfg(any(target_os = "macos", feature = "cuda"))]
    fn max_difference(a: &Tensor, b: &Tensor) -> f32 {
        let a = a
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = b
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn fused_small_ops_match_plain_composition() -> Result<()> {
        use super::super::inkling2_metal as metal;
        let Some(gpu) = metal_device() else {
            return Ok(());
        };
        let on = |t: &Tensor| t.to_device(&gpu).unwrap();

        // Depth mix with one to seven blocks and one to four queries, alone,
        // normalized per query or with one shared norm row, and group-major.
        for (count, queries, hidden) in [
            (1usize, 1usize, 48usize),
            (4, 3, 600),
            (2, 4, 2304),
            (7, 2, 512),
        ] {
            let items = (0..count)
                .map(|i| random(&[3, hidden], 10 + i as u32, 2.0))
                .collect::<Vec<_>>();
            let current = random(&[3, hidden], 20, 2.0);
            let query = random(&[queries, hidden], 21, 1.0);
            let norms = random(&[queries, hidden], 22, 1.0);
            let cpu = Blocks {
                items: items.clone(),
            }
            .mix(&current, &query)?;
            let blocks = items.iter().map(on).collect::<Vec<_>>();
            let mix = |norm: Option<(&Tensor, f32)>, major: bool| {
                metal::depth_mix_one(&blocks, &on(&current), &on(&query), 1e-6, norm, major)
            };
            assert!(
                max_difference(&cpu, &mix(None, false)?) < 1e-4,
                "mix {count} {queries}"
            );
            let major = mix(None, true)?.transpose(0, 1)?;
            assert!(max_difference(&cpu, &major) < 1e-4);
            let normed = mix(Some((&on(&norms), 1e-6)), true)?;
            let shared = mix(Some((&on(&norms.i(0..1)?), 1e-6)), false)?;
            for q in 0..queries {
                let row = cpu.i((.., q))?.contiguous()?;
                let expected = rms_norm(&row, &norms.i(q)?, 1e-6)?;
                assert!(
                    max_difference(&expected, &normed.i(q)?) < 1e-4,
                    "mix norm {q}"
                );
                let expected = rms_norm(&row, &norms.i(0)?, 1e-6)?;
                assert!(max_difference(&expected, &shared.i((.., q))?) < 1e-4);
            }
        }

        // K/V: the short convolutions (static, or with dynamic taps), K's L2
        // clip and V's RMS norm, from column slices of wider rows, with and
        // without a carried state.
        for (tokens, dynamic, carried) in [
            (1usize, false, true),
            (5, false, true),
            (2, false, false),
            (1, true, true),
            (6, true, false),
        ] {
            let (heads, dim, width) = (2usize, 64usize, 4usize);
            let channels = heads * dim;
            let k_rows = random(&[tokens, channels + 40], 80, 2.0);
            let v_rows = random(&[tokens, channels + 8], 81, 2.0);
            let taps = random(&[tokens, 2 * heads * width], 82, 0.5);
            let (k_weight, v_weight) = (
                random(&[channels, 4], 83, 0.5),
                random(&[channels, 4], 84, 0.5),
            );
            let (k_past, v_past) = (
                random(&[3, channels], 85, 1.0),
                random(&[3, channels], 86, 1.0),
            );
            let k_taps = taps.narrow(1, 0, heads * width)?;
            let v_taps = taps.narrow(1, heads * width, heads * width)?;
            let start = |past: &Tensor| carried.then(|| past.clone());
            let (mut k_state, mut v_state) = (start(&k_past), start(&v_past));
            let k = short_conv(
                &k_rows.narrow(1, 0, channels)?,
                &ShortConv::from_weight(&k_weight)?,
                dynamic.then_some((&k_taps, heads)),
                &mut k_state,
            )?;
            let v = short_conv(
                &v_rows.narrow(1, 8, channels)?,
                &ShortConv::from_weight(&v_weight)?,
                dynamic.then_some((&v_taps, heads)),
                &mut v_state,
            )?;
            let k = scaled_clip(&k, heads, None, None, 3.0, 1.0)?;
            let v = rms_unweighted(&v.reshape((tokens, heads, dim))?, 1e-6)?;
            let (k_rows_gpu, v_rows_gpu, taps_gpu) = (on(&k_rows), on(&v_rows), on(&taps));
            let past_gpu = (on(&k_past), on(&v_past));
            let taps_view = (
                taps_gpu.narrow(1, 0, heads * width)?,
                taps_gpu.narrow(1, heads * width, heads * width)?,
            );
            let (k_in, v_in) = (
                k_rows_gpu.narrow(1, 0, channels)?,
                v_rows_gpu.narrow(1, 8, channels)?,
            );
            let (k_weight_gpu, v_weight_gpu) = (on(&k_weight), on(&v_weight));
            let finish = |output: metal::KvOutput<'_>| {
                metal::kv_finish(
                    &k_in,
                    &v_in,
                    carried.then_some((&past_gpu.0, &past_gpu.1)),
                    &k_weight_gpu,
                    &v_weight_gpu,
                    dynamic.then_some((&taps_view.0, &taps_view.1)),
                    heads,
                    3.0,
                    1e-6,
                    output,
                )
            };
            let finished = finish(metal::KvOutput::Fresh)?;
            let head_major = |t: &Tensor| t.transpose(0, 1).unwrap().unsqueeze(0).unwrap();
            let what = format!("kv {tokens} dynamic {dynamic} carried {carried}");
            let (fresh_k, fresh_v) = (finished.k.unwrap(), finished.v.unwrap());
            assert!(max_difference(&head_major(&k), &fresh_k) < 1e-5, "{what}");
            assert!(max_difference(&head_major(&v), &fresh_v) < 1e-5, "{what}");
            let (k_state, v_state) = (k_state.unwrap(), v_state.unwrap());
            assert!(max_difference(&k_state, &finished.k_state) == 0.0, "{what}");
            assert!(max_difference(&v_state, &finished.v_state) == 0.0, "{what}");
            // In place: f16 rows from `row` on, and nothing else, in the caches.
            let (row, capacity) = (2usize, tokens + 5);
            let k_cache = Tensor::zeros((1, heads, capacity, dim), DType::F16, &gpu)?;
            let v_cache = Tensor::zeros((1, heads, capacity, dim), DType::F16, &gpu)?;
            let in_place = finish(metal::KvOutput::Cache {
                k: &k_cache,
                v: &v_cache,
                row,
            })?;
            assert!(in_place.k.is_none() && in_place.v.is_none());
            assert!(max_difference(&k_state, &in_place.k_state) == 0.0, "{what}");
            for (cache, expected) in [(&k_cache, &fresh_k), (&v_cache, &fresh_v)] {
                let cache = cache.to_dtype(DType::F32)?;
                let written = cache.narrow(2, row, tokens)?;
                let rounded = expected.to_dtype(DType::F16)?.to_dtype(DType::F32)?;
                assert!(max_difference(&rounded, &written) == 0.0, "{what} in place");
                let before = cache
                    .narrow(2, 0, row)?
                    .abs()?
                    .sum_all()?
                    .to_scalar::<f32>()?;
                let after = cache
                    .narrow(2, row + tokens, capacity - row - tokens)?
                    .abs()?
                    .sum_all()?
                    .to_scalar::<f32>()?;
                assert_eq!((before, after), (0.0, 0.0), "{what} neighbors");
            }
        }

        // Query side: clipped, scaled queries, and the bias by distance from
        // the clipped relative states, then the sink, for token blocks that
        // span several threadgroups and bias rows that span several slices.
        for (tokens, extent, log_scaled) in
            [(1usize, 5usize, false), (13, 300, true), (3, 1024, true)]
        {
            let (heads, dim, rel) = (3usize, 32usize, 16usize);
            let width = 2 * heads * dim + heads * rel + heads + 7;
            let rows = random(&[tokens, width], 90, 3.0);
            let q_scale = random(&[heads, dim], 91, 1.0);
            let rel_scale = random(&[heads, rel], 92, 1.0);
            let tau = (random(&[tokens, 1, 1], 93, 0.5).abs()? + 1.0)?;
            let basis = random(&[rel, extent + 1], 94, 1.0);
            let token_scale = log_scaled.then_some(&tau);
            let columns = metal::QuerySide {
                q: 0,
                rel: 2 * heads * dim,
                sink: 2 * heads * dim + heads * rel,
            };
            let q = scaled_clip(
                &rows.narrow(1, columns.q, heads * dim)?,
                heads,
                Some(&q_scale),
                token_scale,
                2.0,
                0.5,
            )?;
            let relative = scaled_clip(
                &rows.narrow(1, columns.rel, heads * rel)?,
                heads,
                Some(&rel_scale),
                token_scale,
                1.5,
                1.0,
            )?;
            let table = relative
                .reshape((tokens * heads, rel))?
                .matmul(&basis)?
                .reshape((tokens, heads, extent + 1))?;
            let sinks = rows.narrow(1, columns.sink, heads)?.unsqueeze(2)?;
            let bias = Tensor::cat(&[&table, &sinks], 2)?;
            let tau_gpu = on(&tau);
            let (q_gpu, bias_gpu) = metal::query_prep(
                &on(&rows),
                &columns,
                heads,
                dim,
                &on(&q_scale),
                &on(&rel_scale),
                log_scaled.then_some(&tau_gpu),
                &on(&basis),
                2.0,
                1.5,
                0.5,
            )?;
            assert!(max_difference(&q, &q_gpu) < 1e-5, "queries {tokens}");
            assert!(
                max_difference(&bias, &bias_gpu) < 1e-4,
                "bias {tokens} {extent}"
            );
        }

        // Post convolution plus the residual, from rows or from the weighted
        // sum of expert routes, with and without a carried state.
        for (tokens, routed, carried) in [
            (1usize, false, true),
            (5, false, false),
            (1, true, true),
            (3, true, true),
        ] {
            let (channels, routes) = (48usize, 4usize);
            let weight = random(&[channels, 4], 100, 0.5);
            let past = random(&[3, channels], 101, 1.0);
            let residual = random(&[tokens, channels], 102, 1.0);
            let wide = random(&[tokens, channels + 16], 103, 2.0);
            let experts = random(&[tokens, routes, channels], 104, 2.0);
            let route_weights = random(&[tokens, routes], 105, 1.0);
            let x = if routed {
                experts
                    .broadcast_mul(&route_weights.unsqueeze(2)?)?
                    .sum(1)?
            } else {
                wide.narrow(1, 16, channels)?
            };
            let mut state = carried.then(|| past.clone());
            let expected =
                (short_conv(&x, &ShortConv::from_weight(&weight)?, None, &mut state)? + &residual)?;
            let (wide_gpu, experts_gpu, route_weights_gpu) =
                (on(&wide), on(&experts), on(&route_weights));
            let rows_gpu = wide_gpu.narrow(1, 16, channels)?;
            let input = if routed {
                metal::PostInput::Routes {
                    routed: &experts_gpu,
                    weights: &route_weights_gpu,
                }
            } else {
                metal::PostInput::Rows(&rows_gpu)
            };
            let past_gpu = on(&past);
            let (out, next) = metal::post_conv(
                input,
                carried.then_some(&past_gpu),
                &on(&weight),
                Some(&on(&residual)),
            )?;
            let what = format!("post conv {tokens} routed {routed} carried {carried}");
            assert!(max_difference(&expected, &out) < 1e-5, "{what}");
            assert!(max_difference(&state.unwrap(), &next) < 1e-5, "{what}");
        }

        // The Q/K/V loop gates with their norms equal three gates and norms,
        // for one shared input or three, and three norm rows or one; and the
        // single MoE gate with its norm.
        let gates = (0..3)
            .map(|g| {
                MessageGate::new(
                    random(&[4, 96], 60 + g, 0.2),
                    random(&[4, 48], 63 + g, 0.5),
                    random(&[4, 48], 66 + g, 0.5),
                    1e-6,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let down = Tensor::stack(&[&gates[0].down, &gates[1].down, &gates[2].down], 0)?;
        let forget_gain = Tensor::stack(
            &[
                &gates[0].forget_gain,
                &gates[1].forget_gain,
                &gates[2].forget_gain,
            ],
            0,
        )?;
        let messages = random(&[3, 3, 48], 70, 2.0);
        let weights = random(&[3, 48], 71, 1.0);
        for (x_groups, norm_rows) in [(1usize, 3usize), (3, 3), (1, 1)] {
            let x = random(&[3, x_groups, 48], 72, 2.0);
            let norm = weights.i(0..norm_rows)?;
            let gated = metal::message_gates(
                &on(&x),
                &on(&messages),
                &on(&down),
                &on(&forget_gain),
                1e-6,
                &on(&norm),
            )?;
            for (g, gate) in gates.iter().enumerate() {
                let input = x.i((.., if x_groups == 1 { 0 } else { g }))?.contiguous()?;
                let plain = gate.forward(&input, &messages.i((.., g))?.contiguous()?)?;
                let expected = rms_norm(&plain, &norm.i(g % norm_rows)?, 1e-6)?;
                let error = max_difference(&expected, &gated.i(g)?);
                assert!(
                    error < 1e-4,
                    "gate {g}, {x_groups} inputs, {norm_rows} norms: {error}"
                );
            }
        }
        let gpu_gate = MessageGate::new(
            on(&gates[0].down),
            on(&gates[0].forget),
            on(&gates[0].gain),
            1e-6,
        )?;
        let (x, m) = (random(&[3, 48], 33, 2.0), random(&[3, 48], 34, 2.0));
        let expected = rms_norm(&gates[0].forward(&x, &m)?, &weights.i(1)?, 1e-6)?;
        let normed = gpu_gate.forward_normed(&on(&x), &on(&m), &on(&weights.i(1..2)?), 1e-6)?;
        assert!(max_difference(&expected, &normed) < 1e-4);

        // Sigmoid routing: the same experts, in order, and weights.
        let logits = random(&[5, 128], 110, 4.0);
        let bias = random(&[128], 111, 0.3);
        let scores = candle_nn::ops::sigmoid(&logits)?;
        let ids = scores
            .broadcast_add(&bias)?
            .arg_sort_last_dim(false)?
            .narrow(1, 0, 8)?
            .contiguous()?;
        let chosen = scores.gather(&ids, 1)?;
        let expected = (chosen.broadcast_div(&chosen.sum_keepdim(1)?)? * 8.0)?;
        let (gpu_ids, gpu_weights) = metal::sigmoid_route(&on(&logits), &on(&bias), 8, 8.0)?;
        assert_eq!(
            ids.to_vec2::<u32>()?,
            gpu_ids.to_device(&Device::Cpu)?.to_vec2::<u32>()?
        );
        assert!(max_difference(&expected, &gpu_weights) < 1e-5);

        // The prompt path's gate and transpose.
        let attended = random(&[4, 3, 32], 77, 1.0);
        let gate = random(&[3, 200], 78, 3.0);
        let fused = metal::gate_transpose(&on(&attended), &on(&gate).narrow(1, 10, 128)?)?;
        let expected = (attended.transpose(0, 1)?.contiguous()?.reshape((3, 128))?
            * candle_nn::ops::sigmoid(&gate.narrow(1, 10, 128)?)?)?;
        assert!(max_difference(&expected, &fused) < 1e-5);
        // SiTU against its definition.
        let gate_up = random(&[2, 8, 64], 40, 30.0);
        let fused = metal::situ(&on(&gate_up), 4.0, 25.0)?;
        let (g, u) = (gate_up.narrow(2, 0, 32)?, gate_up.narrow(2, 32, 32)?);
        let expected = ((softcap(&g, 4.0)? * candle_nn::ops::sigmoid(&g)?)? * softcap(&u, 25.0)?)?;
        assert!(max_difference(&expected, &fused) < 1e-4);
        Ok(())
    }

    /// Plain attention over f16-rounded keys and values for one query block.
    #[cfg(any(target_os = "macos", feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    fn reference_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        table: &Tensor,
        sinks: &Tensor,
        start: usize,
        extent: usize,
        window: Option<usize>,
    ) -> Result<Tensor> {
        let (tokens, heads, _) = q.dims3()?;
        let keys = k.dim(1)?;
        let (index, mask) = relative_index_and_mask(start, tokens, keys, extent, window);
        let index = Tensor::from_vec(index, (1, tokens, keys), &Device::Cpu)?
            .broadcast_as((heads, tokens, keys))?
            .contiguous()?;
        let bias = table.transpose(0, 1)?.contiguous()?.gather(&index, 2)?;
        let out = attend_with_sinks(
            &q.transpose(0, 1)?.contiguous()?,
            k,
            v,
            &bias,
            &Tensor::from_vec(mask, (tokens, keys), &Device::Cpu)?,
            &sinks.t()?.contiguous()?,
        )?;
        Ok(out.transpose(0, 1)?.contiguous()?)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn fused_attention_kernels_match_reference() -> Result<()> {
        let Some(metal) = metal_device() else {
            return Ok(());
        };
        let on = |t: &Tensor| t.to_device(&metal).unwrap();
        for (heads, kv_heads, dim, extent) in [
            (6usize, 2usize, 128usize, 24usize),
            (2, 1, 128, 4),
            (2, 1, 128, 6),
        ] {
            let round = |t: &Tensor| {
                t.to_dtype(DType::F16)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
            };
            // (queries, keys, window, start): decode, short chunks, a trimmed
            // sliding cache, and prompts long enough for several key splits.
            for (tokens, keys, window, start) in [
                (13usize, 13usize, Some(4usize), 0usize),
                (13, 13, None, 0),
                (1usize, 1usize, None, 0usize),
                (1, 300, None, 299),
                (3, 40, Some(16), 50),
                (40, 40, None, 0),
                (48, 200, Some(64), 400),
                (64, 1100, None, 1036),
            ] {
                let q = round(&(random(&[tokens, heads, dim], 50, 1.0) * 0.2)?);
                let k = round(&random(&[kv_heads, keys, dim], 51, 1.0));
                let v = round(&random(&[kv_heads, keys, dim], 52, 1.0));
                let table = random(&[tokens, heads, extent + 1], 53, 1.0);
                let table = table.slice_assign(
                    &[0..tokens, 0..heads, extent..extent + 1],
                    &Tensor::zeros((tokens, heads, 1), DType::F32, &Device::Cpu)?,
                )?;
                let sinks = random(&[tokens, heads], 54, 2.0);
                let expected =
                    reference_attention(&q, &k, &v, &table, &sinks, start, extent, window)?;
                let first_key = start + tokens - keys;
                let k16 = on(&k.unsqueeze(0)?.to_dtype(DType::F16)?);
                let v16 = on(&v.unsqueeze(0)?.to_dtype(DType::F16)?);
                let bias = Tensor::cat(&[&table, &sinks.unsqueeze(2)?], 2)?;
                let split = super::super::inkling2_metal::attention(
                    &on(&q),
                    &on(&bias),
                    &k16,
                    &v16,
                    dim,
                    extent,
                    window,
                    start,
                    first_key,
                    None,
                )?;
                // The fused gate equals sigmoid(gate) times the output.
                let gate = random(&[tokens, heads * dim], 55, 3.0);
                let gated = super::super::inkling2_metal::attention(
                    &on(&q),
                    &on(&bias),
                    &k16,
                    &v16,
                    dim,
                    extent,
                    window,
                    start,
                    first_key,
                    Some(&on(&gate)),
                )?;
                let expected_gated = (split
                    .to_device(&Device::Cpu)?
                    .reshape((tokens, heads * dim))?
                    * candle_nn::ops::sigmoid(&gate)?)?;
                assert!(
                    max_difference(&expected_gated, &gated.reshape((tokens, heads * dim))?) < 1e-5
                );
                let error = max_difference(&expected, &split);
                assert!(error < 2e-4, "split-key attention {tokens}x{keys}: {error}");
                if let Some(flash) = super::super::inkling2_flash_metal::try_attention(
                    &on(&q.to_dtype(DType::F16)?),
                    &k16,
                    &v16,
                    &on(&bias),
                    extent,
                    window,
                    start - first_key,
                )? {
                    let flash = flash.transpose(0, 1)?.contiguous()?;
                    let error = max_difference(&expected, &flash);
                    // Probabilities pass through f16 on their way to the value product.
                    assert!(error < 5e-3, "tensor-op attention {tokens}x{keys}: {error}");
                }
            }
        }
        Ok(())
    }

    /// The CUDA kernels behind the plain helpers against the CPU composition.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_helpers_match_plain_composition() -> Result<()> {
        let Some(gpu) = crate::test_cuda_device() else {
            return Ok(());
        };
        let on = |t: &Tensor| t.to_device(&gpu).unwrap();
        // Static and dynamic convolutions, carried over two calls.
        for (tokens, dynamic) in [(5usize, false), (1, false), (6, true), (1, true)] {
            let weight = random(&[16, 4], 1, 0.5);
            let x = random(&[tokens, 16], 2, 1.0);
            let past = random(&[3, 16], 3, 1.0);
            let taps = random(&[tokens, 2 * 4], 4, 0.5);
            let taps_gpu = on(&taps);
            let (conv_cpu, conv_gpu) = (
                ShortConv::from_weight(&weight)?,
                ShortConv::from_weight(&on(&weight))?,
            );
            let (mut state_cpu, mut state_gpu) = (Some(past.clone()), Some(on(&past)));
            let cpu = short_conv(&x, &conv_cpu, dynamic.then_some((&taps, 2)), &mut state_cpu)?;
            let gpu_out = short_conv(
                &on(&x),
                &conv_gpu,
                dynamic.then_some((&taps_gpu, 2)),
                &mut state_gpu,
            )?;
            assert!(
                max_difference(&cpu, &gpu_out) < 1e-5,
                "conv {tokens} {dynamic}"
            );
            assert!(max_difference(&state_cpu.unwrap(), &state_gpu.unwrap()) == 0.0);
        }
        // Clip of a column slice, with head, token and post scales.
        let wide = random(&[3, 200], 5, 3.0);
        let scale = random(&[4, 32], 6, 1.0);
        let tau = random(&[3, 1, 1], 7, 1.0).abs()?;
        let x_gpu = on(&wide).narrow(1, 40, 128)?;
        let cpu = scaled_clip(
            &wide.narrow(1, 40, 128)?,
            4,
            Some(&scale),
            Some(&tau),
            2.0,
            0.25,
        )?;
        let gpu_out = scaled_clip(&x_gpu, 4, Some(&on(&scale)), Some(&on(&tau)), 2.0, 0.25)?;
        assert!(max_difference(&cpu, &gpu_out) < 1e-5);
        let cpu = scaled_clip(&wide.narrow(1, 40, 128)?, 4, None, None, 2.0, 1.0)?;
        let gpu_out = scaled_clip(&x_gpu, 4, None, None, 2.0, 1.0)?;
        assert!(max_difference(&cpu, &gpu_out) < 1e-5);
        // Depth mix with one to seven blocks and one to four queries.
        for (count, queries, hidden) in [
            (1usize, 1usize, 48usize),
            (4, 3, 600),
            (2, 4, 2304),
            (7, 2, 512),
        ] {
            let items = (0..count)
                .map(|i| random(&[3, hidden], 10 + i as u32, 2.0))
                .collect::<Vec<_>>();
            let current = random(&[3, hidden], 20, 2.0);
            let query = random(&[queries, hidden], 21, 1.0);
            let cpu = Blocks {
                items: items.clone(),
            }
            .mix(&current, &query)?;
            let gpu_out = Blocks {
                items: items.iter().map(on).collect(),
            }
            .mix(&on(&current), &on(&query))?;
            assert!(
                max_difference(&cpu, &gpu_out) < 1e-4,
                "mix {count} {queries} {hidden}"
            );
        }
        // Message gate.
        let (down, forget, gain) = (
            random(&[4, 96], 30, 0.2),
            random(&[4, 48], 31, 0.5),
            random(&[4, 48], 32, 0.5),
        );
        let cpu_gate = MessageGate::new(down.clone(), forget.clone(), gain.clone(), 1e-6)?;
        let gpu_gate = MessageGate::new(on(&down), on(&forget), on(&gain), 1e-6)?;
        let (x, m) = (random(&[3, 48], 33, 2.0), random(&[3, 48], 34, 2.0));
        let cpu = cpu_gate.forward(&x, &m)?;
        let gpu_out = gpu_gate.forward(&on(&x), &on(&m))?;
        assert!(max_difference(&cpu, &gpu_out) < 1e-4);
        // SiTU against its definition.
        let gate_up = random(&[2, 8, 64], 40, 30.0);
        let gpu_out = cuda::situ(&on(&gate_up), 4.0, 25.0)?;
        let (g, u) = (gate_up.narrow(2, 0, 32)?, gate_up.narrow(2, 32, 32)?);
        let cpu = ((softcap(&g, 4.0)? * candle_nn::ops::sigmoid(&g)?)? * softcap(&u, 25.0)?)?;
        assert!(max_difference(&cpu, &gpu_out) < 1e-4);
        // RMS norms with and without a weight.
        let x = random(&[5, 2304], 41, 3.0);
        let weight = random(&[2304], 42, 1.0);
        let cpu = rms_unweighted(&x, 1e-6)?.broadcast_mul(&weight)?;
        assert!(max_difference(&cpu, &rms_norm(&on(&x), &on(&weight), 1e-6)?) < 1e-5);
        let cpu = rms_unweighted(&x, 1e-6)?;
        assert!(max_difference(&cpu, &rms_unweighted(&on(&x), 1e-6)?) < 1e-5);
        // Sigmoid routing: the same experts and weights as the plain route.
        let logits = random(&[9, 128], 43, 4.0);
        let bias = random(&[128], 44, 0.2);
        let (ids, weights) = cuda::sigmoid_top_k(&on(&logits), &on(&bias), 8, 8.0)?;
        let scores = candle_nn::ops::sigmoid(&logits)?;
        let plain_ids = scores
            .broadcast_add(&bias)?
            .arg_sort_last_dim(false)?
            .narrow(1, 0, 8)?
            .contiguous()?;
        let plain_weights = scores.gather(&plain_ids, 1)?;
        let plain_weights = (plain_weights
            .broadcast_div(&(plain_weights.sum_keepdim(1)? + ROUTE_WEIGHT_EPS)?)?
            * 8.0)?;
        assert_eq!(
            ids.to_device(&Device::Cpu)?.to_vec2::<u32>()?,
            plain_ids.to_vec2::<u32>()?
        );
        assert!(max_difference(&plain_weights, &weights) < 1e-5);
        // The loop gates' split-K down product against Candle's, with
        // partial token and rank tiles.
        for (groups, tokens, rank, inner) in
            [(3usize, 70usize, 64usize, 4608usize), (1, 13, 4, 128)]
        {
            let a = random(&[groups, tokens, inner], 47, 1.0);
            let w = random(&[groups, rank, inner], 48, 0.05);
            let expected = a.matmul(&w.transpose(1, 2)?)?;
            let got = cuda::gemm_nt(&on(&a), &on(&w))?;
            assert!(
                max_difference(&expected, &got) < 1e-4,
                "{groups}x{tokens}x{rank}"
            );
        }
        // The router with its logits: one launch for up to 8 tokens (run
        // twice, so its counter must come back zeroed), products beyond.
        let router = random(&[128, 2304], 45, 0.05);
        for tokens in [1usize, 3, 8, 8, 9] {
            let x = random(&[tokens, 2304], 46 + tokens as u32, 1.0);
            let logits = x.matmul(&router.t()?)?;
            let (expected_ids, expected_weights) =
                cuda::sigmoid_top_k(&on(&logits), &on(&bias), 8, 8.0)?;
            let (ids, weights) = cuda::route(&on(&x), &on(&router), &on(&bias), 8, 8.0)?;
            assert_eq!(
                ids.to_device(&Device::Cpu)?.to_vec2::<u32>()?,
                expected_ids.to_device(&Device::Cpu)?.to_vec2::<u32>()?,
                "{tokens} tokens"
            );
            assert!(max_difference(&expected_weights, &weights) < 1e-5);
        }
        Ok(())
    }

    /// The BF16 projection's one-token kernel and prompt product against F32.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_bf16_projection_matches_f32() -> Result<()> {
        use candle_core::quantized::GgmlDType;
        let Some(gpu) = crate::test_cuda_device() else {
            return Ok(());
        };
        let cols = 2304usize;
        let bf16_projection = |rows: usize, seed: u32| -> Result<(Projection, Tensor)> {
            let weight = random(&[rows, cols], seed, 0.05).to_dtype(DType::BF16)?;
            let bytes = weight
                .flatten_all()?
                .to_vec1::<half::bf16>()?
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect::<Vec<u8>>();
            let qt = candle_core::quantized::ggml_file::qtensor_from_ggml(
                GgmlDType::BF16,
                &bytes,
                vec![rows, cols],
                &gpu,
            )?;
            let projection = Projection::new(Arc::new(qt), &gpu)?;
            assert!(matches!(projection, Projection::Bf16(_)));
            Ok((projection, weight.to_dtype(DType::F32)?))
        };
        let (projection, reference) = bf16_projection(70, 90)?;
        // Up to three single-token products in one launch, each with its
        // own input; the second input starts at an odd offset.
        let others = [bf16_projection(33, 92)?, bf16_projection(129, 93)?];
        let inputs = random(&[3, cols + 1], 94, 1.0);
        let inputs = [
            inputs.narrow(0, 0, 1)?.narrow(1, 0, cols)?.contiguous()?,
            inputs
                .flatten_all()?
                .narrow(0, cols + 2, cols)?
                .reshape((1, cols))?,
            inputs.narrow(0, 2, 1)?.narrow(1, 1, cols)?.contiguous()?,
        ];
        let on_gpu = inputs
            .iter()
            .map(|x| x.to_device(&gpu))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let odd = gpu_slice_at_odd_offset(&on_gpu[1])?;
        let projections = [&projection, &others[0].0, &others[1].0];
        let references = [&reference, &others[0].1, &others[1].1];
        for count in [2usize, 3] {
            let pairs = (0..count)
                .map(|j| (projections[j], if j == 1 { &odd } else { &on_gpu[j] }))
                .collect::<Vec<_>>();
            let got = Projection::forward_many(&pairs)?;
            for j in 0..count {
                let expected = inputs[j].matmul(&references[j].t()?)?;
                let error = max_difference(&expected, &got[j]);
                assert!(error < 1e-4, "{count} products, product {j}: {error}");
            }
        }
        for tokens in [1usize, 5] {
            let x = random(&[tokens, cols], 91, 1.0);
            let expected = x.matmul(&reference.t()?)?;
            let got = projection.forward(&x.to_device(&gpu)?)?;
            // Prompts round their inputs and outputs to BF16, as the
            // checkpoint computes them.
            let tolerance = if tokens == 1 { 1e-4 } else { 3e-2 };
            assert!(
                max_difference(&expected, &got) < tolerance,
                "{tokens} tokens: {}",
                max_difference(&expected, &got)
            );
        }
        Ok(())
    }

    /// `x` as a view that starts one element into its storage.
    #[cfg(feature = "cuda")]
    fn gpu_slice_at_odd_offset(x: &Tensor) -> Result<Tensor> {
        let cols = x.dim(1)?;
        let padded = Tensor::cat(&[&x.narrow(1, 0, 1)?, x], 1)?;
        Ok(padded
            .flatten_all()?
            .narrow(0, 1, cols)?
            .reshape((1, cols))?)
    }

    /// The CUDA split-key attention against the plain reference.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_attention_matches_reference() -> Result<()> {
        let Some(gpu) = crate::test_cuda_device() else {
            return Ok(());
        };
        let on = |t: &Tensor| t.to_device(&gpu).unwrap();
        let round = |t: &Tensor| {
            t.to_dtype(DType::F16)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
        };
        let (heads, kv_heads, dim, extent) = (6usize, 2usize, 128usize, 24usize);
        // (queries, keys, window, start): decode, short chunks, a trimmed
        // sliding cache, and prompts long enough for several key splits;
        // from 64 queries the tensor-core kernel, with partial query and key
        // tiles, windows inside and across tiles, and a sliding cache that
        // starts mid-tile.
        for (tokens, keys, window, start) in [
            (1usize, 1usize, None, 0usize),
            (1, 300, None, 299),
            (3, 40, Some(16), 50),
            (13, 13, Some(4), 0),
            (40, 40, None, 0),
            (48, 200, Some(64), 400),
            (64, 1100, None, 1036),
            (100, 100, None, 0),
            (77, 77, Some(16), 0),
            (130, 400, Some(300), 270),
            (200, 1250, Some(1024), 1050),
            (100, 1123, Some(1024), 5000),
        ] {
            let q = round(&(random(&[tokens, heads, dim], 50, 1.0) * 0.2)?);
            let k = round(&random(&[kv_heads, keys, dim], 51, 1.0));
            let v = round(&random(&[kv_heads, keys, dim], 52, 1.0));
            let table = random(&[tokens, heads, extent + 1], 53, 1.0);
            let table = table.slice_assign(
                &[0..tokens, 0..heads, extent..extent + 1],
                &Tensor::zeros((tokens, heads, 1), DType::F32, &Device::Cpu)?,
            )?;
            let sinks = random(&[tokens, heads], 54, 2.0);
            let expected = reference_attention(&q, &k, &v, &table, &sinks, start, extent, window)?;
            let bias = Tensor::cat(&[&table, &sinks.unsqueeze(2)?], 2)?;
            // The gate's rows are slices of wider rows, as in the stacked
            // query-side projection.
            let wide = random(&[tokens, heads * dim + 8], 55, 1.0);
            let gate = wide.narrow(1, 8, heads * dim)?;
            let gated = (expected.reshape((tokens, heads * dim))?
                * candle_nn::ops::sigmoid(&gate)?)?
            .reshape((tokens, heads, dim))?;
            let wide = on(&wide);
            for (gate, expected) in [
                (None, &expected),
                (Some(wide.narrow(1, 8, heads * dim)?), &gated),
            ] {
                let attended = cuda::attention(
                    &on(&q),
                    &on(&bias),
                    &on(&k.unsqueeze(0)?.to_dtype(DType::F16)?),
                    &on(&v.unsqueeze(0)?.to_dtype(DType::F16)?),
                    dim,
                    extent,
                    window,
                    start,
                    start + tokens - keys,
                    gate.as_ref(),
                )?;
                let error = max_difference(expected, &attended);
                let gated = gate.is_some();
                // The tensor-core kernel rounds probabilities to f16.
                let tolerance = if tokens >= 64 { 2e-3 } else { 2e-4 };
                assert!(
                    error < tolerance,
                    "attention {tokens}x{keys} (gate {gated}): {error}"
                );
            }
        }
        Ok(())
    }

    /// The whole model on CUDA against the plain CPU path, with Candle-type
    /// and NVFP4 expert banks, in one pass and in chunks.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_stage_matches_plain_stage() -> Result<()> {
        let Some(gpu) = crate::test_cuda_device() else {
            return Ok(());
        };
        for (nvfp4, tolerance) in [(false, 5e-3f32), (true, 2e-2)] {
            let cpu = tiny_stage_full(&Device::Cpu, 64, 2, 1, 128, nvfp4)?;
            let accelerated = tiny_stage_full(&gpu, 64, 2, 1, 128, nvfp4)?;
            for (length, chunks) in [
                (13usize, vec![13usize]),
                (13, vec![5, 4, 4]),
                (13, vec![1; 13]),
                (40, vec![40]),
                (40, vec![24, 16]),
            ] {
                let tokens = (0..length as u32)
                    .map(|i| (i * 7 + 3) % 30)
                    .collect::<Vec<_>>();
                let expected = run_chunks(&cpu, &tokens, &[length])?;
                let actual = run_chunks(&accelerated, &tokens, &chunks)?;
                let error = expected
                    .iter()
                    .zip(&actual)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                // Keys and values are f16 in the cache; prompt expert tiles
                // round their inputs to f16.
                assert!(
                    error < tolerance,
                    "nvfp4 {nvfp4}, {length} tokens in {chunks:?}: max logit difference {error}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn schedule_visits_the_core_three_times() {
        let visits = visit_schedule(24, 6, 18, 3);
        assert_eq!(visits.len(), 48);
        assert_eq!(visits[5], Visit { layer: 5, pass: 0 });
        assert_eq!(visits[6], Visit { layer: 6, pass: 0 });
        assert_eq!(visits[17], Visit { layer: 17, pass: 0 });
        assert_eq!(visits[18], Visit { layer: 6, pass: 1 });
        assert_eq!(visits[30], Visit { layer: 6, pass: 2 });
        assert_eq!(visits[41], Visit { layer: 17, pass: 2 });
        assert_eq!(visits[42], Visit { layer: 18, pass: 0 });
        assert_eq!(visits[47], Visit { layer: 23, pass: 0 });
        let full = visits
            .iter()
            .filter(|visit| (visit.layer + 1) % 6 == 0)
            .count();
        assert_eq!(full, 8, "layers 5 and 23 once, 11 and 17 three times");
    }

    #[test]
    fn relative_index_masks_causal_and_window() {
        // Queries 5 and 6, keys 3..=6, window 3, extent 2.
        let (index, mask) = relative_index_and_mask(5, 2, 4, 2, Some(3));
        // Distances: query 5 → [2, 1, 0, -1]; query 6 → [3, 2, 1, 0].
        assert_eq!(index, vec![2, 1, 0, 2, 2, 2, 1, 0]);
        let visible = mask.iter().map(|m| m.is_finite()).collect::<Vec<_>>();
        assert_eq!(
            visible,
            vec![true, true, true, false, false, true, true, true]
        );
        let (_, full) = relative_index_and_mask(5, 2, 7, 2, None);
        assert!(full[..6].iter().all(|m| m.is_finite()) && full[6].is_infinite());
        assert!(full[7..].iter().all(|m| m.is_finite()));
    }

    #[test]
    fn short_conv_matches_direct_sum_and_carries_state() -> Result<()> {
        // Two channels, width 3; tap 2 weights the current token.
        let weight = cpu_tensor(&[0.5, -1.0, 2.0, 0.25, 1.5, -0.5], &[2, 3]);
        let conv = ShortConv::from_weight(&weight)?;
        let x = cpu_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]);
        let mut state = None;
        let whole = short_conv(&x, &conv, None, &mut state)?.to_vec2::<f32>()?;
        let rows = x.to_vec2::<f32>()?;
        let w = weight.to_vec2::<f32>()?;
        for t in 0..4 {
            for c in 0..2 {
                let mut sum = rows[t][c];
                for (tap, weight) in w[c].iter().enumerate() {
                    let source = t as i64 + tap as i64 - 2;
                    if source >= 0 {
                        sum += weight * rows[source as usize][c];
                    }
                }
                let expected = sum / 2.0;
                assert!((whole[t][c] - expected).abs() < 1e-5, "t={t} c={c}");
            }
        }
        // The same tokens split in two calls give the same outputs.
        let mut split_state = None;
        let first = short_conv(&x.narrow(0, 0, 3)?, &conv, None, &mut split_state)?;
        let second = short_conv(&x.narrow(0, 3, 1)?, &conv, None, &mut split_state)?;
        let joined = Tensor::cat(&[&first, &second], 0)?.to_vec2::<f32>()?;
        assert_eq!(joined, whole);
        assert_eq!(
            split_state.unwrap().to_vec2::<f32>()?,
            state.unwrap().to_vec2::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn dynamic_taps_follow_backward_distance() -> Result<()> {
        // One head of two channels, width 2, zero static taps: only lag terms.
        let conv = ShortConv::from_weight(&Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?)?;
        let x = cpu_tensor(&[1.0, 10.0, 2.0, 20.0], &[2, 2]);
        // Token 0: lag0 = 1, lag1 = 0; token 1: lag0 = 0, lag1 = 1.
        let taps = cpu_tensor(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let out = short_conv(&x, &conv, Some((&taps, 1)), &mut None)?.to_vec2::<f32>()?;
        let scale = 1.0 / 5f32.sqrt();
        // Token 0: x0 + x0; token 1: x1 + x0.
        let expected = [[2.0 * scale, 20.0 * scale], [3.0 * scale, 30.0 * scale]];
        for (row, want) in out.iter().zip(expected) {
            for (got, want) in row.iter().zip(want) {
                assert!((got - want).abs() < 1e-5);
            }
        }
        Ok(())
    }

    #[test]
    fn l2_clip_only_shrinks_long_vectors() -> Result<()> {
        let x = cpu_tensor(&[3.0, 4.0, 0.3, 0.4], &[2, 2]);
        let clipped = l2_clip(&x, 1.0)?.to_vec2::<f32>()?;
        assert!((clipped[0][0] - 0.6).abs() < 1e-5 && (clipped[0][1] - 0.8).abs() < 1e-5);
        assert_eq!(clipped[1], vec![0.3, 0.4]);
        Ok(())
    }

    #[test]
    fn depth_mix_attends_over_normalized_states() -> Result<()> {
        let block = cpu_tensor(&[2.0, 0.0], &[1, 2]);
        let current = cpu_tensor(&[0.0, 3.0], &[1, 2]);
        // A zero query averages the normalized states.
        let zero = Tensor::zeros((1, 2), DType::F32, &Device::Cpu)?;
        let mixed = depth_mix(std::slice::from_ref(&block), &current, &zero)?;
        let root2 = 2f32.sqrt();
        let got = mixed.flatten_all()?.to_vec1::<f32>()?;
        assert!((got[0] - root2 / 2.0).abs() < 1e-4 && (got[1] - root2 / 2.0).abs() < 1e-4);
        // A strong query along the block picks it.
        let query = cpu_tensor(&[100.0, 0.0], &[1, 2]);
        let got = depth_mix(&[block], &current, &query)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!((got[0] - root2).abs() < 1e-3 && got[1].abs() < 1e-3);
        Ok(())
    }

    #[test]
    fn message_gate_with_zero_projections_passes_input() -> Result<()> {
        let gate = MessageGate::new(
            Tensor::zeros((3, 4), DType::F32, &Device::Cpu)?,
            Tensor::zeros((3, 2), DType::F32, &Device::Cpu)?,
            Tensor::zeros((3, 2), DType::F32, &Device::Cpu)?,
            1e-6,
        )?;
        let x = cpu_tensor(&[1.0, -2.0], &[1, 2]);
        let m = cpu_tensor(&[5.0, 5.0], &[1, 2]);
        assert_eq!(
            gate.forward(&x, &m)?.to_vec2::<f32>()?,
            vec![vec![1.0, -2.0]]
        );
        Ok(())
    }

    #[test]
    fn attention_matches_scalar_reference() -> Result<()> {
        let (heads, kv_heads, tokens, keys, dim) = (4, 2, 2, 3, 2);
        let seq = |n: usize, scale: f32| {
            (0..n)
                .map(|i| ((i * 7 % 11) as f32 - 5.0) * scale)
                .collect::<Vec<_>>()
        };
        let q = seq(heads * tokens * dim, 0.3);
        let k = seq(kv_heads * keys * dim, 0.2);
        let v = seq(kv_heads * keys * dim, 0.5);
        let bias = seq(heads * tokens * keys, 0.1);
        let sinks = seq(heads * tokens, 0.4);
        let (_, mask) = relative_index_and_mask(1, tokens, keys, 4, None);
        let out = attend_with_sinks(
            &cpu_tensor(&q, &[heads, tokens, dim]),
            &cpu_tensor(&k, &[kv_heads, keys, dim]),
            &cpu_tensor(&v, &[kv_heads, keys, dim]),
            &cpu_tensor(&bias, &[heads, tokens, keys]),
            &cpu_tensor(&mask, &[tokens, keys]),
            &cpu_tensor(&sinks, &[heads, tokens]),
        )?
        .flatten_all()?
        .to_vec1::<f32>()?;
        for h in 0..heads {
            let g = h / (heads / kv_heads);
            for t in 0..tokens {
                let mut logits = (0..keys)
                    .map(|s| {
                        let dot = (0..dim)
                            .map(|d| q[(h * tokens + t) * dim + d] * k[(g * keys + s) * dim + d])
                            .sum::<f32>();
                        dot + bias[(h * tokens + t) * keys + s] + mask[t * keys + s]
                    })
                    .collect::<Vec<_>>();
                logits.push(sinks[h * tokens + t]);
                let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp = logits.iter().map(|l| (l - max).exp()).collect::<Vec<_>>();
                let total = exp.iter().sum::<f32>();
                for d in 0..dim {
                    let want = (0..keys)
                        .map(|s| exp[s] / total * v[(g * keys + s) * dim + d])
                        .sum::<f32>();
                    let got = out[(h * tokens + t) * dim + d];
                    assert!(
                        (got - want).abs() < 1e-5,
                        "h={h} t={t} d={d}: {got} vs {want}"
                    );
                }
            }
        }
        Ok(())
    }
}
