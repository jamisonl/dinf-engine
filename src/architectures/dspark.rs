//! DSpark block drafter for Inkling 2 Mini (`general.architecture = dflash`
//! with a recurrent Markov head).
//!
//! The drafter proposes `block - 1` tokens in one non-causal pass:
//!
//! 1. The block is the target's normalized embedding of the anchor (the last
//!    sampled token) followed by `block - 1` copies of `mask_embd`, at the
//!    positions after the committed context.
//! 2. Each of the three layers attends from the block to itself and to
//!    context keys and values projected from the target's features: the
//!    partial-block state entering three looped visits and the target's
//!    final normalized state. Each layer mixes those four rows with its own
//!    row of `softmax(dspark_fuse)`, normalizes with `dflare_norm`, and
//!    projects with `attn_{k,v}_ctx` (keys also `attn_k_ctx_norm` and RoPE).
//!    Sliding layers read the last `sliding_window` context positions, the
//!    others all of them. Block keys use `attn_q_norm` / `attn_k_norm` and
//!    NeoX RoPE; layers are pre-norm with a SwiGLU MLP.
//! 3. `output_norm`, then the target's output head gives base logits for
//!    the `block - 1` mask rows.
//! 4. A recurrent Markov head runs over those rows in order. From a zero
//!    state and the anchor, each step reads the previous token's
//!    `markov_w1` row, the step's hidden row and `f_rnn` (the last committed
//!    row's features mixed with the head's fusion row and `dspark_norm`):
//!    `joint_proj` gives gate, candidate and output (rank each); the state
//!    becomes `gate * state + (1 - gate) * tanh(candidate)`, and
//!    `head(dspark_hidden_out(tanh(output)))` is added to the base logits
//!    before the softcap. The argmax is the step's token and the next
//!    step's previous token.
//!
//! Numerics follow the public reference (vLLM's Inkling 2 DSpark model),
//! here in F32 with F16 cached keys and values.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::quantized::{QTensor, gguf_file};
use candle_core::{D, DType, Device, Tensor};

use super::dflash2::{
    DraftQMatMul, RmsNorm, metadata_f64, metadata_string, metadata_usize, metadata_usize_array,
    repeat_kv, take_norm, take_qtensor,
};
use super::{KvCache, ModelArchitecture, PreallocatedKvCache, kv_cache_update_native};
use crate::gguf_limits::{validate_gguf_header_budget, validate_layer_count};

const MAX_HIDDEN: usize = 16_384;
const MAX_HEAD_DIM: usize = 1_024;
const MAX_RANK: usize = 4_096;

/// Whether a GGUF holds a DSpark drafter (a DFlash file with a Markov head).
pub(crate) fn is_dspark(content: &gguf_file::Content) -> bool {
    content.tensor_infos.contains_key("markov_w1.weight")
}

/// Per-request drafter state: projected context keys and values per layer,
/// and the Markov head's context row.
pub(crate) struct DSparkCache {
    layers: Vec<Option<KvCache>>,
    /// `[1, hidden]`: the last committed row's features as the Markov head reads them.
    f_rnn: Option<Tensor>,
    next_position: usize,
}

fn share(kv: &KvCache) -> KvCache {
    match kv {
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
        KvCache::Quantized(_) => unreachable!("DSpark never quantizes its cache"),
    }
}

fn live(kv: &KvCache) -> Result<(Tensor, Tensor)> {
    match kv {
        KvCache::F16 { k, v } => Ok((k.clone(), v.clone())),
        KvCache::PreallocatedF16(cache) => cache.active_f16(),
        KvCache::Quantized(_) => anyhow::bail!("DSpark never quantizes its cache"),
    }
}

impl Clone for DSparkCache {
    /// Share storage: appends only write unused rows, so a clone keeps the
    /// state it was taken at, as a checkpoint must.
    fn clone(&self) -> Self {
        Self {
            layers: self
                .layers
                .iter()
                .map(|kv| kv.as_ref().map(share))
                .collect(),
            f_rnn: self.f_rnn.clone(),
            next_position: self.next_position,
        }
    }
}

impl DSparkCache {
    pub(crate) fn fork(&self) -> Result<Self> {
        Ok(Self {
            layers: self
                .layers
                .iter()
                .map(|kv| {
                    kv.as_ref()
                        .map(|kv| {
                            let (k, v) = live(kv)?;
                            Ok(KvCache::PreallocatedF16(PreallocatedKvCache::from_f16(
                                super::copy_persistent_tensor(&k)?,
                                super::copy_persistent_tensor(&v)?,
                            )?))
                        })
                        .transpose()
                })
                .collect::<Result<_>>()?,
            f_rnn: self
                .f_rnn
                .as_ref()
                .map(super::copy_persistent_tensor)
                .transpose()?,
            next_position: self.next_position,
        })
    }

    /// Exports live K/V and `f_rnn` tensors for disk persistence.
    pub(crate) fn export_tensors(
        &self,
    ) -> Result<(usize, Vec<Option<(Tensor, Tensor)>>, Option<Tensor>)> {
        let layers = self
            .layers
            .iter()
            .map(|kv| kv.as_ref().map(live).transpose())
            .collect::<Result<Vec<_>>>()?;
        Ok((self.next_position, layers, self.f_rnn.clone()))
    }

    /// Reconstructs a [`DSparkCache`] from exported K/V and `f_rnn` tensors.
    pub(crate) fn import_tensors(
        next_position: usize,
        layers: Vec<Option<(Tensor, Tensor)>>,
        f_rnn: Option<Tensor>,
    ) -> Result<Self> {
        let layers = layers
            .into_iter()
            .map(|kv| {
                kv.map(|(k, v)| {
                    Ok::<_, anyhow::Error>(KvCache::PreallocatedF16(PreallocatedKvCache::from_f16(
                        super::copy_persistent_tensor(&k)?,
                        super::copy_persistent_tensor(&v)?,
                    )?))
                })
                .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        let f_rnn = f_rnn
            .as_ref()
            .map(super::copy_persistent_tensor)
            .transpose()?;
        Ok(Self {
            layers,
            f_rnn,
            next_position,
        })
    }

    pub(crate) fn live_bytes(&self) -> usize {
        let bytes = |t: &Tensor| t.elem_count() * t.dtype().size_in_bytes();
        self.layers
            .iter()
            .flatten()
            .filter_map(|kv| live(kv).ok())
            .map(|(k, v)| bytes(&k) + bytes(&v))
            .sum::<usize>()
            + self.f_rnn.as_ref().map_or(0, bytes)
    }

    pub(crate) fn clear(&mut self) {
        self.layers.iter_mut().for_each(|kv| *kv = None);
        self.f_rnn = None;
        self.next_position = 0;
    }

    pub(crate) fn next_position(&self) -> usize {
        self.next_position
    }
}

#[cfg(target_os = "macos")]
struct MetalQ8Linear {
    bytes: Tensor,
    out_dim: usize,
    in_dim: usize,
}

#[cfg(target_os = "macos")]
impl MetalQ8Linear {
    fn from_qtensors(parts: &[&Arc<QTensor>], device: &Device) -> Result<Option<Self>> {
        use candle_core::quantized::GgmlDType;
        if !device.is_metal()
            || parts.is_empty()
            || parts.iter().any(|qt| qt.dtype() != GgmlDType::Q8_0)
        {
            return Ok(None);
        }
        let mut in_dim = None;
        let mut out_dim = 0usize;
        let mut raw = Vec::new();
        for qt in parts {
            let (o, i) = qt.shape().dims2()?;
            if !i.is_multiple_of(32) || in_dim.is_some_and(|w| w != i) {
                return Ok(None);
            }
            in_dim = Some(i);
            out_dim += o;
            let data = qt.data()?;
            raw.extend_from_slice(data.as_ref());
        }
        let in_dim = in_dim.unwrap();
        let len = raw.len();
        let bytes = Tensor::from_vec(raw, len, device)?;
        Ok(Some(Self {
            bytes,
            out_dim,
            in_dim,
        }))
    }

    fn forward(&self, x: &Tensor, residual: Option<&Tensor>, swiglu: bool) -> Result<Tensor> {
        super::dspark_metal::q8_0_matvec_rows(
            &self.bytes,
            self.out_dim,
            self.in_dim,
            x,
            residual,
            swiglu,
        )
        .map_err(Into::into)
    }
}

#[cfg(target_os = "macos")]
struct MetalDraftWeights {
    qkv: MetalQ8Linear,
    output: MetalQ8Linear,
    gate_up: MetalQ8Linear,
    down: MetalQ8Linear,
}

enum DraftLayerWeights {
    #[cfg(target_os = "macos")]
    MetalQ8(MetalDraftWeights),
    Generic {
        q: DraftQMatMul,
        k: DraftQMatMul,
        v: DraftQMatMul,
        output: DraftQMatMul,
        gate: DraftQMatMul,
        up: DraftQMatMul,
        down: DraftQMatMul,
    },
}

struct DraftLayer {
    attn_norm: RmsNorm,
    weights: DraftLayerWeights,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    k_ctx: DraftQMatMul,
    v_ctx: DraftQMatMul,
    #[cfg(target_os = "macos")]
    kv_ctx_q8: Option<MetalQ8Linear>,
    k_ctx_norm: RmsNorm,
    fuse_norm: RmsNorm,
    ffn_norm: RmsNorm,
    /// Context positions a sliding layer keeps; `None` keeps all.
    window: Option<usize>,
}

#[cfg(target_os = "macos")]
enum DraftHead {
    Q4_0 {
        bytes: Tensor,
        vocab: usize,
        hidden: usize,
    },
    Q8_0 {
        bytes: Tensor,
        vocab: usize,
        hidden: usize,
    },
}

/// Sampling parameters passed to DSpark when generating a block for exact
/// Leviathan speculative rejection sampling (`temperature > 0`).
#[derive(Clone, Copy)]
pub(crate) struct DraftSampling {
    pub(crate) temperature: f32,
    pub(crate) top_k: usize,
    pub(crate) top_p: f32,
    pub(crate) step_u: [f32; 8],
}

/// Proposed tokens from one draft block, with optional sparse per-step draft
/// distributions `(token_id, q_prob)` for exact rejection sampling.
pub(crate) struct DraftProposal {
    pub(crate) tokens: Vec<u32>,
    pub(crate) distributions: Option<Vec<Vec<(u32, f32)>>>,
}

/// Loaded DSpark drafter. The token embedding and output head are the
/// target's, reached through the target architecture at draft time.
pub(crate) struct DSparkDraft {
    target_layers: Vec<usize>,
    block_size: usize,
    context_length: usize,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rank: usize,
    eps: f64,
    rope_theta: f64,
    softcap: f64,
    layers: Vec<DraftLayer>,
    output_norm: RmsNorm,
    mask: Tensor,
    /// `[layers + 1, features]` softmaxed fusion weights; the last row is
    /// the Markov head's.
    fuse: Vec<Vec<f32>>,
    markov_norm: RmsNorm,
    /// `[vocab rows, rank]` BF16, gathered per step.
    markov_w1: Tensor,
    /// `joint_proj` columns for `[state, previous]`, transposed: `[2 rank, 3 rank]`.
    joint_recurrent: Tensor,
    /// `joint_proj` columns for `[state, previous]`, row-major `[3 rank, 2 rank]` for Metal.
    #[cfg(target_os = "macos")]
    joint_rec_t: Tensor,
    /// `joint_proj` columns for `[hidden, f_rnn]`, transposed: `[2 hidden, 3 rank]`.
    joint_context: Tensor,
    /// `joint_proj` columns for `[hidden, f_rnn]`, row-major `[3 rank, 2 hidden]` for Metal.
    #[cfg(target_os = "macos")]
    joint_context_t: Tensor,
    joint_bias: Tensor,
    /// `head(dspark_hidden_out)` folded once: `[rank, vocab]` (dropped on Metal
    /// once `fold_t` is built, saving ~102 MB of unified memory).
    fold: Option<Tensor>,
    /// `[vocab, rank]` transposed fold (Q4_0, Q8_0, or F16) for coalesced Metal Markov argmax.
    #[cfg(target_os = "macos")]
    fold_t: Option<(Tensor, super::dspark_metal::FoldFormat)>,
    fold_bias: Tensor,
    /// Compact draft-only quantized output head on Metal (target stays exact).
    #[cfg(target_os = "macos")]
    draft_head: Option<DraftHead>,
    /// Linear verification cost model `(base_ms, slope_ms)` for GPU-side
    /// confidence-based early stopping (`slope_ms == 0` disables trimming).
    #[cfg(target_os = "macos")]
    adaptive_cost: (f32, f32),
    vocab: usize,
    device: Device,
}

fn dense(
    tensors: &mut HashMap<String, Arc<QTensor>>,
    name: &str,
    device: &Device,
) -> Result<Tensor> {
    Ok(take_qtensor(tensors, name)?
        .dequantize(device)?
        .to_dtype(DType::F32)?)
}

fn metadata_bool_array(md: &HashMap<String, gguf_file::Value>, key: &str) -> Result<Vec<bool>> {
    let Some(gguf_file::Value::Array(values)) = md.get(key) else {
        anyhow::bail!("missing or invalid DSpark metadata key {key}");
    };
    values
        .iter()
        .map(|value| match value {
            gguf_file::Value::Bool(flag) => Ok(*flag),
            _ => anyhow::bail!("DSpark metadata {key} must hold booleans"),
        })
        .collect()
}

fn metadata_bool(md: &HashMap<String, gguf_file::Value>, key: &str) -> Option<bool> {
    match md.get(key) {
        Some(gguf_file::Value::Bool(flag)) => Some(*flag),
        _ => None,
    }
}

impl DSparkDraft {
    pub(crate) fn from_gguf(
        path: &Path,
        device: &Device,
        target: &dyn ModelArchitecture,
    ) -> Result<Self> {
        // The CUDA target's verification and rollback have not been run yet.
        anyhow::ensure!(
            !device.is_cuda(),
            "DSpark drafting is qualified on Metal and CPU only so far"
        );
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("failed to open DSpark GGUF: {}", path.display()))?;
        validate_gguf_header_budget(&mut file)
            .with_context(|| format!("failed to validate DSpark GGUF: {}", path.display()))?;
        let content = gguf_file::Content::read(&mut file)
            .with_context(|| format!("failed to parse DSpark GGUF: {}", path.display()))?;
        let md = &content.metadata;
        anyhow::ensure!(
            metadata_string(md, "general.architecture")? == "dflash" && is_dspark(&content),
            "speculative sidecar is not a DSpark GGUF"
        );
        anyhow::ensure!(
            metadata_bool(md, "dflash.sample_from_anchor") == Some(false)
                && metadata_bool(md, "dflash.attention.causal") != Some(true)
                && metadata_bool(md, "dflash.has_confidence_head") != Some(true),
            "DSpark drafter must draft after its anchor, without causal masking or a confidence head"
        );
        let block_count = metadata_usize(md, "dflash.block_count")?;
        let block_size = metadata_usize(md, "dflash.block_size")?;
        let hidden = metadata_usize(md, "dflash.embedding_length")?;
        let heads = metadata_usize(md, "dflash.attention.head_count")?;
        let kv_heads = metadata_usize(md, "dflash.attention.head_count_kv")?;
        let head_dim = metadata_usize(md, "dflash.attention.key_length")?;
        let context_length = metadata_usize(md, "dflash.context_length")?;
        let window = metadata_usize(md, "dflash.attention.sliding_window")?;
        let pattern = metadata_bool_array(md, "dflash.attention.sliding_window_pattern")?;
        let rank = metadata_usize(md, "dflash.dspark.markov_rank")?;
        let target_layers = metadata_usize_array(md, "dflash.target_layers")?;
        let eps = metadata_f64(md, "dflash.attention.layer_norm_rms_epsilon")?;
        let rope_theta = metadata_f64(md, "dflash.rope.freq_base")?;
        let softcap = metadata_f64(md, "dflash.final_logit_softcapping").unwrap_or(0.0);

        validate_layer_count(block_count, "DSpark block count")?;
        anyhow::ensure!(
            (1..=MAX_HIDDEN).contains(&hidden) && hidden == target.hidden_dim(),
            "DSpark hidden size {hidden} does not match the target's {}",
            target.hidden_dim()
        );
        anyhow::ensure!(
            block_size >= 2 && block_size <= context_length,
            "DSpark block size must be in 2..={context_length}"
        );
        anyhow::ensure!(
            (2..=MAX_HEAD_DIM).contains(&head_dim) && head_dim.is_multiple_of(2),
            "invalid DSpark head width"
        );
        anyhow::ensure!(
            heads > 0 && kv_heads > 0 && heads.is_multiple_of(kv_heads),
            "invalid DSpark GQA geometry"
        );
        anyhow::ensure!(
            pattern.len() == block_count && window > 0,
            "DSpark sliding-window pattern does not cover its layers"
        );
        anyhow::ensure!((1..=MAX_RANK).contains(&rank), "invalid DSpark Markov rank");
        anyhow::ensure!(
            target_layers.len() >= 2
                && target_layers
                    .iter()
                    .all(|&id| id < target.speculative_feature_limit()),
            "DSpark target features {target_layers:?} do not fit the target"
        );
        anyhow::ensure!(
            eps.is_finite() && eps > 0.0 && rope_theta.is_finite() && rope_theta > 0.0,
            "DSpark norm epsilon and RoPE base must be finite and positive"
        );
        anyhow::ensure!(
            softcap >= 0.0 && softcap.is_finite(),
            "invalid DSpark softcap"
        );

        let mut tensors = HashMap::with_capacity(content.tensor_infos.len());
        for name in content.tensor_infos.keys().cloned().collect::<Vec<_>>() {
            let tensor = content
                .tensor(&mut file, &name, device)
                .with_context(|| format!("failed to load DSpark tensor {name}"))?;
            tensors.insert(name, Arc::new(tensor));
        }
        let features = target_layers.len();
        let mut layers = Vec::with_capacity(block_count);
        for (layer, sliding) in pattern.iter().enumerate() {
            let name = |suffix: &str| format!("blk.{layer}.{suffix}.weight");
            let t = &mut tensors;
            let q_qt = take_qtensor(t, &name("attn_q"))?;
            let k_qt = take_qtensor(t, &name("attn_k"))?;
            let v_qt = take_qtensor(t, &name("attn_v"))?;
            let output_qt = take_qtensor(t, &name("attn_output"))?;
            let k_ctx_qt = take_qtensor(t, &name("attn_k_ctx"))?;
            let v_ctx_qt = take_qtensor(t, &name("attn_v_ctx"))?;
            let gate_qt = take_qtensor(t, &name("ffn_gate"))?;
            let up_qt = take_qtensor(t, &name("ffn_up"))?;
            let down_qt = take_qtensor(t, &name("ffn_down"))?;

            #[cfg(target_os = "macos")]
            let (weights, kv_ctx_q8) = {
                let kv_ctx_q8 = MetalQ8Linear::from_qtensors(&[&k_ctx_qt, &v_ctx_qt], device)?;
                let metal = if block_size <= 8
                    && head_dim <= 256
                    && let Some(qkv) = MetalQ8Linear::from_qtensors(&[&q_qt, &k_qt, &v_qt], device)?
                    && let Some(output) = MetalQ8Linear::from_qtensors(&[&output_qt], device)?
                    && let Some(gate_up) =
                        MetalQ8Linear::from_qtensors(&[&gate_qt, &up_qt], device)?
                    && let Some(down) = MetalQ8Linear::from_qtensors(&[&down_qt], device)?
                {
                    Some(MetalDraftWeights {
                        qkv,
                        output,
                        gate_up,
                        down,
                    })
                } else {
                    None
                };
                let weights = match metal {
                    Some(m) => DraftLayerWeights::MetalQ8(m),
                    None => DraftLayerWeights::Generic {
                        q: DraftQMatMul::from_qtensor(q_qt, device)?,
                        k: DraftQMatMul::from_qtensor(k_qt, device)?,
                        v: DraftQMatMul::from_qtensor(v_qt, device)?,
                        output: DraftQMatMul::from_qtensor(output_qt, device)?,
                        gate: DraftQMatMul::from_qtensor(gate_qt, device)?,
                        up: DraftQMatMul::from_qtensor(up_qt, device)?,
                        down: DraftQMatMul::from_qtensor(down_qt, device)?,
                    },
                };
                (weights, kv_ctx_q8)
            };
            #[cfg(not(target_os = "macos"))]
            let weights = DraftLayerWeights::Generic {
                q: DraftQMatMul::from_qtensor(q_qt, device)?,
                k: DraftQMatMul::from_qtensor(k_qt, device)?,
                v: DraftQMatMul::from_qtensor(v_qt, device)?,
                output: DraftQMatMul::from_qtensor(output_qt, device)?,
                gate: DraftQMatMul::from_qtensor(gate_qt, device)?,
                up: DraftQMatMul::from_qtensor(up_qt, device)?,
                down: DraftQMatMul::from_qtensor(down_qt, device)?,
            };

            layers.push(DraftLayer {
                attn_norm: take_norm(t, &name("attn_norm"), device, eps)?,
                weights,
                q_norm: take_norm(t, &name("attn_q_norm"), device, eps)?,
                k_norm: take_norm(t, &name("attn_k_norm"), device, eps)?,
                k_ctx: DraftQMatMul::from_qtensor(k_ctx_qt, device)?,
                v_ctx: DraftQMatMul::from_qtensor(v_ctx_qt, device)?,
                #[cfg(target_os = "macos")]
                kv_ctx_q8,
                k_ctx_norm: take_norm(t, &name("attn_k_ctx_norm"), device, eps)?,
                fuse_norm: take_norm(t, &name("dflare_norm"), device, eps)?,
                ffn_norm: take_norm(t, &name("ffn_norm"), device, eps)?,
                window: sliding.then_some(window),
            });
        }
        let fuse_logits = dense(&mut tensors, "dspark_fuse.weight", device)?;
        anyhow::ensure!(
            fuse_logits.dims() == [block_count + 1, features],
            "DSpark fusion weights are {:?}, expected [{}, {features}]",
            fuse_logits.dims(),
            block_count + 1
        );
        let fuse = candle_nn::ops::softmax_last_dim(&fuse_logits)?.to_vec2::<f32>()?;
        let mask = dense(&mut tensors, "mask_embd.weight", device)?.reshape((1, hidden))?;
        let output_norm = take_norm(&mut tensors, "output_norm.weight", device, eps)?;
        let markov_norm = take_norm(&mut tensors, "dspark_norm.weight", device, eps)?;
        // BF16 as stored; each step widens the one row it gathers.
        let markov_w1 = dense(&mut tensors, "markov_w1.weight", device)?.to_dtype(DType::BF16)?;
        let joint = dense(&mut tensors, "joint_proj.weight", device)?;
        anyhow::ensure!(
            joint.dims() == [3 * rank, 2 * rank + 2 * hidden],
            "DSpark joint projection is {:?}",
            joint.dims()
        );
        #[cfg(target_os = "macos")]
        let joint_rec_t = joint.narrow(1, 0, 2 * rank)?.contiguous()?;
        let joint_recurrent = joint.narrow(1, 0, 2 * rank)?.t()?.contiguous()?;
        #[cfg(target_os = "macos")]
        let joint_context_t = joint.narrow(1, 2 * rank, 2 * hidden)?.contiguous()?;
        let joint_context = joint.narrow(1, 2 * rank, 2 * hidden)?.t()?.contiguous()?;
        let joint_bias = dense(&mut tensors, "joint_proj.bias", device)?.reshape((1, 3 * rank))?;
        let hidden_out = dense(&mut tensors, "dspark_hidden_out.weight", device)?;
        let hidden_out_bias =
            dense(&mut tensors, "dspark_hidden_out.bias", device)?.reshape((1, hidden))?;
        anyhow::ensure!(
            hidden_out.dims() == [hidden, rank] && markov_w1.dim(1)? == rank,
            "DSpark Markov head shapes differ from rank {rank}"
        );
        if let Some(name) = tensors.keys().next() {
            anyhow::bail!("DSpark does not use tensor {name}");
        }
        // head(W h + b) = (head W) h + head b, folded once for every step.
        // F16 halves the table each Markov step reads; latents are in [-1, 1].
        let fold = target
            .speculative_output_head(&hidden_out.t()?.contiguous()?)?
            .to_dtype(DType::F16)?
            .contiguous()?;
        let fold_bias = target
            .speculative_output_head(&hidden_out_bias)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let vocab = fold.dim(1)?;
        anyhow::ensure!(
            vocab == target.vocab_size() && markov_w1.dim(0)? >= vocab,
            "DSpark Markov table does not cover the target vocabulary"
        );
        #[cfg(target_os = "macos")]
        let fold_t = if device.is_metal() && rank <= 256 {
            let f16_t = fold.t()?.contiguous()?;
            if rank == 256 {
                let mode = std::env::var("DINF_DSPARK_FOLD_QUANT").unwrap_or_else(|_| {
                    if std::env::var("DINF_DSPARK_FOLD_F16").as_deref() == Ok("1") {
                        "f16".into()
                    } else {
                        "q4_0".into()
                    }
                });
                match mode.as_str() {
                    "f16" => Some((f16_t, super::dspark_metal::FoldFormat::F16)),
                    "q8_0" => Some((
                        super::dspark_metal::quantize_f16_to_q8_0(&f16_t)?,
                        super::dspark_metal::FoldFormat::Q8_0,
                    )),
                    _ => Some((
                        super::dspark_metal::quantize_f16_to_q4_0(&f16_t)?,
                        super::dspark_metal::FoldFormat::Q4_0,
                    )),
                }
            } else {
                Some((f16_t, super::dspark_metal::FoldFormat::F16))
            }
        } else {
            None
        };
        #[cfg(target_os = "macos")]
        let fold = if fold_t.is_some() { None } else { Some(fold) };
        #[cfg(not(target_os = "macos"))]
        let fold = Some(fold);
        #[cfg(target_os = "macos")]
        let draft_head = if device.is_metal()
            && hidden.is_multiple_of(32)
            && let Some(bf16_head) = target.speculative_output_bf16_weight()
        {
            match std::env::var("DINF_DSPARK_HEAD_QUANT")
                .unwrap_or_else(|_| "q4_0".into())
                .as_str()
            {
                "bf16" | "none" => None,
                "q8_0" => Some(DraftHead::Q8_0 {
                    bytes: super::dspark_metal::quantize_bf16_to_q8_0(&bf16_head)?,
                    vocab,
                    hidden,
                }),
                _ => Some(DraftHead::Q4_0 {
                    bytes: super::dspark_metal::quantize_bf16_to_q4_0(&bf16_head)?,
                    vocab,
                    hidden,
                }),
            }
        } else {
            None
        };
        #[cfg(target_os = "macos")]
        let adaptive_cost = if std::env::var("DINF_DSPARK_ADAPTIVE").as_deref() == Ok("0") {
            (0.0f32, 0.0f32)
        } else {
            let base: f32 = std::env::var("DINF_DSPARK_COST_BASE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(17.5);
            let slope: f32 = std::env::var("DINF_DSPARK_COST_SLOPE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(4.0);
            (base, slope)
        };
        #[cfg(target_os = "macos")]
        if let Device::Metal(metal) = device {
            super::dspark_metal::precompile(metal)?;
        }
        tracing::info!(
            path = %path.display(),
            block_count,
            block_size,
            ?target_layers,
            "DSpark draft loaded"
        );
        Ok(Self {
            target_layers,
            block_size,
            context_length,
            hidden,
            heads,
            kv_heads,
            head_dim,
            rank,
            eps,
            rope_theta,
            softcap,
            layers,
            output_norm,
            mask,
            fuse,
            markov_norm,
            markov_w1,
            joint_recurrent,
            #[cfg(target_os = "macos")]
            joint_rec_t,
            joint_context,
            #[cfg(target_os = "macos")]
            joint_context_t,
            joint_bias,
            fold,
            #[cfg(target_os = "macos")]
            fold_t,
            fold_bias,
            #[cfg(target_os = "macos")]
            draft_head,
            #[cfg(target_os = "macos")]
            adaptive_cost,
            vocab,
            device: device.clone(),
        })
    }

    pub(crate) fn target_layers(&self) -> &[usize] {
        &self.target_layers
    }

    pub(crate) fn block_size(&self) -> usize {
        self.block_size
    }

    pub(crate) fn context_length(&self) -> usize {
        self.context_length
    }

    /// The sliding layers' context windows; full layers grow with the request.
    pub(crate) fn cache_budget_bytes(&self) -> Result<usize> {
        let row = 2 * self.kv_heads * self.head_dim * DType::F16.size_in_bytes();
        self.layers
            .iter()
            .filter_map(|layer| layer.window)
            .try_fold(0usize, |sum, window| {
                window
                    .checked_add(self.block_size)
                    .and_then(|rows| rows.checked_mul(row))
                    .and_then(|bytes| sum.checked_add(bytes))
            })
            .context("DSpark cache size overflow")
    }

    pub(crate) fn new_cache(&self) -> DSparkCache {
        DSparkCache {
            layers: (0..self.layers.len()).map(|_| None).collect(),
            f_rnn: None,
            next_position: 0,
        }
    }

    /// NeoX RoPE tables for `len` positions from `start`: `[len, head_dim / 2]`.
    fn rope(&self, start: usize, len: usize) -> Result<(Tensor, Tensor)> {
        let half = self.head_dim / 2;
        let mut cos = Vec::with_capacity(len * half);
        let mut sin = Vec::with_capacity(len * half);
        for position in start..start + len {
            for i in 0..half {
                let inv = self
                    .rope_theta
                    .powf(-((2 * i) as f64) / self.head_dim as f64);
                let angle = position as f64 * inv;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        Ok((
            Tensor::from_vec(cos, (len, half), &self.device)?,
            Tensor::from_vec(sin, (len, half), &self.device)?,
        ))
    }

    /// `[rows, heads * head_dim]` to RMS-normed, rotated `[1, heads, rows, head_dim]`.
    fn heads(
        &self,
        rows: &Tensor,
        heads: usize,
        norm: Option<&RmsNorm>,
        rope: Option<&(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let count = rows.dim(0)?;
        let mut x = rows.reshape((1, count, heads, self.head_dim))?;
        if let Some(norm) = norm {
            x = norm.forward(&x)?;
        }
        let x = x.transpose(1, 2)?.contiguous()?;
        Ok(match rope {
            Some((cos, sin)) => candle_nn::rotary_emb::rope(&x, cos, sin)?,
            None => x,
        })
    }

    /// Non-causal attention of the block's queries `[1, heads, rows, dim]`
    /// over F16 keys and values `[1, kv_heads, keys, dim]`. Every query sees
    /// the same keys, so the rows fold into heads (`head * rows + row` keeps
    /// each query's KV head) and run as one-query attention, whose Metal
    /// kernel splits long contexts across threadgroups.
    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
        let (_, heads, rows, dim) = q.dims4()?;
        let folded = q.reshape((1, heads * rows, 1, dim))?.to_dtype(DType::F16)?;
        if let Some(out) = super::try_fused_f16_sdpa_native(&folded, k, v, false, scale as f32)? {
            return Ok(out.reshape((1, heads, rows, dim))?.to_dtype(DType::F32)?);
        }
        let repeats = self.heads / self.kv_heads;
        let k = repeat_kv(k.to_dtype(DType::F32)?, repeats)?;
        let v = repeat_kv(v.to_dtype(DType::F32)?, repeats)?;
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        Ok(candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?)
    }

    /// The four features of each row mixed with fusion row `row`, then `norm`.
    fn mix(&self, features: &[Tensor], row: usize, norm: &RmsNorm) -> Result<Tensor> {
        #[cfg(target_os = "macos")]
        if self.device.is_metal() && (2..=4).contains(&features.len()) {
            return Ok(super::dspark_metal::mix_norm(
                features,
                &self.fuse[row],
                &norm.weight,
                norm.eps as f32,
            )?);
        }
        let mut mixed: Option<Tensor> = None;
        for (feature, &alpha) in features.iter().zip(&self.fuse[row]) {
            let term = (feature.to_dtype(DType::F32)? * alpha as f64)?;
            mixed = Some(match mixed {
                Some(sum) => (sum + term)?,
                None => term,
            });
        }
        norm.forward(&mixed.context("DSpark has no target features")?)
    }

    /// The largest logit's ID, `[1]` U32; non-finite rows give an
    /// out-of-vocabulary ID that the final read rejects.
    fn argmax(&self, logits: &Tensor) -> Result<Tensor> {
        #[cfg(target_os = "macos")]
        if self.device.is_metal()
            && let Some((ids, _)) = super::dflash2_topk_metal::try_top_k_deferred(logits, 1)?
        {
            return Ok(ids.reshape(1)?);
        }
        Ok(logits.argmax(D::Minus1)?.to_dtype(DType::U32)?)
    }

    /// Project committed target rows into each layer's context cache.
    pub(crate) fn inject(
        &self,
        cache: &mut DSparkCache,
        features: &[Tensor],
        start_pos: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            start_pos == cache.next_position,
            "non-contiguous DSpark injection: expected {}, got {start_pos}",
            cache.next_position
        );
        anyhow::ensure!(
            features.len() == self.target_layers.len(),
            "DSpark received {} target features, expected {}",
            features.len(),
            self.target_layers.len()
        );
        let rows = features[0].dim(0)?;
        anyhow::ensure!(rows > 0, "DSpark cannot inject an empty sequence");
        for feature in features {
            anyhow::ensure!(
                feature.dims2()? == (rows, self.hidden),
                "DSpark target feature shape changed"
            );
        }
        #[cfg(target_os = "macos")]
        let metal_fast = self.device.is_metal()
            && rows <= 8
            && self.head_dim <= 256
            && self.layers.iter().all(|layer| layer.kv_ctx_q8.is_some());
        #[cfg(not(target_os = "macos"))]
        let metal_fast = false;

        let rope = if metal_fast {
            None
        } else {
            Some(self.rope(start_pos, rows)?)
        };
        let mut next = cache.clone();
        for (index, (layer, slot)) in self.layers.iter().zip(next.layers.iter_mut()).enumerate() {
            let context = self.mix(features, index, &layer.fuse_norm)?;
            #[cfg(target_os = "macos")]
            let (k_f16, v_f16) = if metal_fast {
                let kv_ctx = layer
                    .kv_ctx_q8
                    .as_ref()
                    .expect("checked metal_fast kv_ctx_q8")
                    .forward(&context, None, false)?;
                let (_, k, v) = super::dspark_metal::qkv_prep(
                    &kv_ctx,
                    None,
                    &layer.k_ctx_norm.weight,
                    0,
                    self.kv_heads,
                    self.head_dim,
                    start_pos,
                    self.eps as f32,
                    self.rope_theta,
                )?;
                (k, v)
            } else {
                let rope = rope.as_ref().expect("rope table");
                let k = self.heads(
                    &layer.k_ctx.forward_rows(&context)?,
                    self.kv_heads,
                    Some(&layer.k_ctx_norm),
                    Some(rope),
                )?;
                let v = self.heads(
                    &layer.v_ctx.forward_rows(&context)?,
                    self.kv_heads,
                    None,
                    None,
                )?;
                (k.to_dtype(DType::F16)?, v.to_dtype(DType::F16)?)
            };
            #[cfg(not(target_os = "macos"))]
            let (k_f16, v_f16) = {
                let rope = rope.as_ref().expect("rope table");
                let k = self.heads(
                    &layer.k_ctx.forward_rows(&context)?,
                    self.kv_heads,
                    Some(&layer.k_ctx_norm),
                    Some(rope),
                )?;
                let v = self.heads(
                    &layer.v_ctx.forward_rows(&context)?,
                    self.kv_heads,
                    None,
                    None,
                )?;
                (k.to_dtype(DType::F16)?, v.to_dtype(DType::F16)?)
            };
            kv_cache_update_native(slot, k_f16, v_f16, None)?;
            if let (Some(window), Some(kv)) = (layer.window, slot.as_mut()) {
                let len = live(kv)?.0.dim(2)?;
                if len > window {
                    kv.evict_oldest(len - window)?;
                }
            }
            // Keep at least `block_size` free slots past the active span so
            // `draft()`'s shared scratch append never reallocates or compacts.
            if let Some(KvCache::PreallocatedF16(dense)) = slot.as_mut() {
                dense.reserve(self.block_size)?;
            }
        }
        let last = features
            .iter()
            .map(|feature| feature.narrow(0, rows - 1, 1))
            .collect::<candle_core::Result<Vec<_>>>()?;
        next.f_rnn = Some(self.mix(&last, self.layers.len(), &self.markov_norm)?);
        next.next_position = start_pos + rows;
        *cache = next;
        Ok(())
    }

    /// Draft `block_size - 1` tokens after `anchor`. The block's own keys
    /// and values are written past the cache's live rows and dropped again.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn draft(
        &self,
        cache: &DSparkCache,
        anchor: u32,
        start_pos: usize,
        block_size: usize,
        target: &dyn ModelArchitecture,
    ) -> Result<Vec<u32>> {
        Ok(self
            .draft_with_sampling(cache, anchor, start_pos, block_size, target, None)?
            .tokens)
    }

    /// Draft up to `block_size - 1` tokens after `anchor`, optionally sampling
    /// from the Markov head's sparse top-32 distribution for exact rejection
    /// sampling verification on Metal.
    pub(crate) fn draft_with_sampling(
        &self,
        cache: &DSparkCache,
        anchor: u32,
        start_pos: usize,
        block_size: usize,
        target: &dyn ModelArchitecture,
        sampling: Option<&DraftSampling>,
    ) -> Result<DraftProposal> {
        anyhow::ensure!(
            start_pos == cache.next_position,
            "DSpark target and draft caches are out of step"
        );
        anyhow::ensure!(
            (2..=self.block_size).contains(&block_size),
            "DSpark block size {block_size} is outside 2..={}",
            self.block_size
        );
        let f_rnn = cache
            .f_rnn
            .as_ref()
            .context("DSpark drafts only after the target's features are injected")?;
        let anchor_row = target
            .speculative_embed_normed(&[anchor])?
            .to_dtype(DType::F32)?;
        let mut rows = vec![anchor_row];
        rows.extend(std::iter::repeat_n(self.mask.clone(), block_size - 1));
        let mut x = Tensor::cat(&rows, 0)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let needs_rope = self
            .layers
            .iter()
            .any(|layer| matches!(layer.weights, DraftLayerWeights::Generic { .. }));
        let rope = if needs_rope {
            Some(self.rope(start_pos, block_size)?)
        } else {
            None
        };
        for (layer, kv) in self.layers.iter().zip(&cache.layers) {
            let h = layer.attn_norm.forward(&x)?;
            match &layer.weights {
                #[cfg(target_os = "macos")]
                DraftLayerWeights::MetalQ8(m) => {
                    let qkv = m.qkv.forward(&h, None, false)?;
                    let (q_folded, k, v) = super::dspark_metal::qkv_prep(
                        &qkv,
                        Some(&layer.q_norm.weight),
                        &layer.k_norm.weight,
                        self.heads,
                        self.kv_heads,
                        self.head_dim,
                        start_pos,
                        self.eps as f32,
                        self.rope_theta,
                    )?;
                    let q_folded = q_folded.expect("folded queries");
                    let mut scratch = kv.as_ref().map(share);
                    let (k_all, v_all) = kv_cache_update_native(&mut scratch, k, v, None)?;
                    let attention =
                        if self.head_dim == 128 && self.heads.is_multiple_of(self.kv_heads) {
                            super::dspark_metal::attend_128(
                                &q_folded,
                                &k_all,
                                &v_all,
                                block_size,
                                self.heads,
                                self.kv_heads,
                                scale as f32,
                            )?
                        } else if let Some(out) = super::try_fused_f16_sdpa_native(
                            &q_folded,
                            &k_all,
                            &v_all,
                            false,
                            scale as f32,
                        )? {
                            super::dspark_metal::attn_unfold(
                                &out,
                                block_size,
                                self.heads,
                                self.head_dim,
                            )?
                        } else {
                            let q = q_folded
                                .reshape((1, self.heads, block_size, self.head_dim))?
                                .to_dtype(DType::F32)?;
                            self.attend(&q, &k_all, &v_all, scale)?
                                .transpose(1, 2)?
                                .contiguous()?
                                .reshape((block_size, self.heads * self.head_dim))?
                        };
                    x = m.output.forward(&attention, Some(&x), false)?;
                    let h = layer.ffn_norm.forward(&x)?;
                    let activated = m.gate_up.forward(&h, None, true)?;
                    x = m.down.forward(&activated, Some(&x), false)?;
                }
                DraftLayerWeights::Generic {
                    q,
                    k,
                    v,
                    output,
                    gate,
                    up,
                    down,
                } => {
                    let rope = rope.as_ref().expect("rope table");
                    let q = self.heads(
                        &q.forward_rows(&h)?,
                        self.heads,
                        Some(&layer.q_norm),
                        Some(rope),
                    )?;
                    let k = self.heads(
                        &k.forward_rows(&h)?,
                        self.kv_heads,
                        Some(&layer.k_norm),
                        Some(rope),
                    )?;
                    let v = self.heads(&v.forward_rows(&h)?, self.kv_heads, None, None)?;
                    let (k, v) = (k.to_dtype(DType::F16)?, v.to_dtype(DType::F16)?);
                    // Append the block past the live rows of a shared view: no copy
                    // of the context, and the request's cache keeps its length.
                    let mut scratch = kv.as_ref().map(share);
                    let (k_all, v_all) = kv_cache_update_native(&mut scratch, k, v, None)?;
                    let attention = self.attend(&q, &k_all, &v_all, scale)?;
                    let attention = attention
                        .transpose(1, 2)?
                        .contiguous()?
                        .reshape((block_size, self.heads * self.head_dim))?;
                    x = (x + output.forward_rows(&attention)?)?;
                    let h = layer.ffn_norm.forward(&x)?;
                    let activated =
                        (candle_nn::ops::silu(&gate.forward_rows(&h)?)? * up.forward_rows(&h)?)?;
                    x = (x + down.forward_rows(&activated)?)?;
                }
            }
        }
        let hidden = self.output_norm.forward(&x)?.narrow(0, 1, block_size - 1)?;
        let steps = block_size - 1;
        #[cfg(target_os = "macos")]
        let base = match &self.draft_head {
            Some(DraftHead::Q4_0 {
                bytes,
                vocab,
                hidden: width,
            }) if steps <= 8 => {
                super::dspark_metal::q4_0_matvec_rows(bytes, *vocab, *width, &hidden)?
            }
            Some(DraftHead::Q8_0 {
                bytes,
                vocab,
                hidden: width,
            }) if steps <= 8 => {
                super::dspark_metal::q8_0_matvec_rows(bytes, *vocab, *width, &hidden, None, false)?
            }
            _ => target
                .speculative_output_head(&hidden)?
                .to_dtype(DType::F32)?,
        };
        #[cfg(not(target_os = "macos"))]
        let base = target
            .speculative_output_head(&hidden)?
            .to_dtype(DType::F32)?;
        let cat_ctx = Tensor::cat(&[&hidden, &f_rnn.broadcast_as((steps, self.hidden))?], 1)?;
        #[cfg(target_os = "macos")]
        let ctx_proj = if (1..=8).contains(&steps)
            && let Some(y) =
                super::inkling2_metal::try_f32_matvec_t_rows(&cat_ctx, &self.joint_context_t)?
        {
            y
        } else {
            cat_ctx.matmul(&self.joint_context)?
        };
        #[cfg(not(target_os = "macos"))]
        let ctx_proj = cat_ctx.matmul(&self.joint_context)?;
        let context = ctx_proj.broadcast_add(&self.joint_bias)?;
        #[cfg(target_os = "macos")]
        if let Some((fold_t, fold_format)) = &self.fold_t {
            let (cost_base, cost_slope) = if steps > 1 {
                self.adaptive_cost
            } else {
                (0.0, 0.0)
            };
            if let Some(s) = sampling
                && s.temperature > 0.0
                && steps <= 8
            {
                let (ids, dists) = super::dspark_metal::markov_sample_steps(
                    anchor,
                    steps,
                    self.rank,
                    self.vocab,
                    self.softcap as f32,
                    s.temperature,
                    s.top_k,
                    s.top_p,
                    &s.step_u,
                    cost_base,
                    cost_slope,
                    &self.markov_w1,
                    &self.joint_rec_t,
                    &context,
                    &base,
                    fold_t,
                    *fold_format,
                    &self.fold_bias,
                )?;
                anyhow::ensure!(
                    ids.iter().all(|&id| (id as usize) < self.vocab),
                    "DSpark drafted a token outside the vocabulary"
                );
                return Ok(DraftProposal {
                    tokens: ids,
                    distributions: Some(dists),
                });
            }
            let (ids, _probs) = super::dspark_metal::markov_decode_steps(
                anchor,
                steps,
                self.rank,
                self.vocab,
                self.softcap as f32,
                cost_base,
                cost_slope,
                &self.markov_w1,
                &self.joint_rec_t,
                &context,
                &base,
                fold_t,
                *fold_format,
                &self.fold_bias,
            )?;
            anyhow::ensure!(
                ids.iter().all(|&id| (id as usize) < self.vocab),
                "DSpark drafted a token outside the vocabulary"
            );
            return Ok(DraftProposal {
                tokens: ids,
                distributions: None,
            });
        }
        #[cfg(not(target_os = "macos"))]
        let _ = sampling;
        let fold = self
            .fold
            .as_ref()
            .context("DSpark fallback fold table missing")?;
        let rank = self.rank;
        let mut state = Tensor::zeros((1, rank), DType::F32, &self.device)?;
        let mut previous = Tensor::new(&[anchor], &self.device)?;
        let mut drafted = Vec::with_capacity(steps);
        for step in 0..steps {
            let w1 = self
                .markov_w1
                .index_select(&previous, 0)?
                .to_dtype(DType::F32)?;
            let raw = (Tensor::cat(&[&state, &w1], 1)?.matmul(&self.joint_recurrent)?
                + context.narrow(0, step, 1)?)?;
            let gate = candle_nn::ops::sigmoid(&raw.narrow(1, 0, rank)?)?;
            let candidate = raw.narrow(1, rank, rank)?.tanh()?;
            let latent = raw.narrow(1, 2 * rank, rank)?.tanh()?;
            state = (&candidate + gate.mul(&(&state - &candidate)?)?)?;
            let bias = latent
                .to_dtype(DType::F16)?
                .matmul(fold)?
                .to_dtype(DType::F32)?;
            let mut logits = (base.narrow(0, step, 1)? + bias)?.broadcast_add(&self.fold_bias)?;
            if self.softcap > 0.0 {
                logits = ((logits / self.softcap)?.tanh()? * self.softcap)?;
            }
            previous = self.argmax(&logits)?;
            drafted.push(previous.clone());
        }
        let ids = Tensor::cat(&drafted, 0)?.to_vec1::<u32>()?;
        anyhow::ensure!(
            ids.iter().all(|&id| (id as usize) < self.vocab),
            "DSpark drafted a token outside the vocabulary"
        );
        Ok(DraftProposal {
            tokens: ids,
            distributions: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;
    use gguf_file::Value;

    use crate::architectures::inkling2_loop::tests::tiny_stage;
    use crate::architectures::{LayerCache, StageInput};

    /// A random two-layer drafter for the tiny Inkling 2 stage (hidden 16,
    /// 30 tokens), reading three features: two visit inputs and the final state.
    fn tiny_draft_file(features: &[usize]) -> Result<tempfile::NamedTempFile> {
        let (hidden, heads, kv_heads, head_dim, ffn, rank, layers) = (16, 2, 1, 8, 32, 4, 2);
        let mut seed = 1u32;
        let mut random = |shape: &[usize]| -> Result<QTensor> {
            let count = shape.iter().product::<usize>();
            let values = (0..count)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
                })
                .collect::<Vec<_>>();
            let tensor = Tensor::from_vec(values, shape, &Device::Cpu)?;
            Ok(QTensor::quantize(&tensor, GgmlDType::F32)?)
        };
        let mut tensors = vec![
            ("output_norm.weight".to_string(), random(&[hidden])?),
            ("mask_embd.weight".to_string(), random(&[hidden])?),
            (
                "dspark_fuse.weight".to_string(),
                random(&[layers + 1, features.len()])?,
            ),
            ("dspark_norm.weight".to_string(), random(&[hidden])?),
            (
                "dspark_hidden_out.weight".to_string(),
                random(&[hidden, rank])?,
            ),
            ("dspark_hidden_out.bias".to_string(), random(&[hidden])?),
            ("markov_w1.weight".to_string(), random(&[32, rank])?),
            (
                "joint_proj.weight".to_string(),
                random(&[3 * rank, 2 * rank + 2 * hidden])?,
            ),
            ("joint_proj.bias".to_string(), random(&[3 * rank])?),
        ];
        for layer in 0..layers {
            for (name, shape) in [
                ("attn_norm", vec![hidden]),
                ("attn_q", vec![heads * head_dim, hidden]),
                ("attn_k", vec![kv_heads * head_dim, hidden]),
                ("attn_v", vec![kv_heads * head_dim, hidden]),
                ("attn_output", vec![hidden, heads * head_dim]),
                ("attn_q_norm", vec![head_dim]),
                ("attn_k_norm", vec![head_dim]),
                ("attn_k_ctx", vec![kv_heads * head_dim, hidden]),
                ("attn_v_ctx", vec![kv_heads * head_dim, hidden]),
                ("attn_k_ctx_norm", vec![head_dim]),
                ("dflare_norm", vec![hidden]),
                ("ffn_norm", vec![hidden]),
                ("ffn_gate", vec![ffn, hidden]),
                ("ffn_up", vec![ffn, hidden]),
                ("ffn_down", vec![hidden, ffn]),
            ] {
                tensors.push((format!("blk.{layer}.{name}.weight"), random(&shape)?));
            }
        }
        let u32_value = |v: usize| Value::U32(v as u32);
        let metadata = vec![
            ("general.architecture", Value::String("dflash".into())),
            ("dflash.block_count", u32_value(layers)),
            ("dflash.block_size", u32_value(4)),
            ("dflash.embedding_length", u32_value(hidden)),
            ("dflash.attention.head_count", u32_value(heads)),
            ("dflash.attention.head_count_kv", u32_value(kv_heads)),
            ("dflash.attention.key_length", u32_value(head_dim)),
            ("dflash.context_length", u32_value(256)),
            ("dflash.attention.sliding_window", u32_value(3)),
            (
                "dflash.attention.sliding_window_pattern",
                Value::Array(vec![Value::Bool(true), Value::Bool(false)]),
            ),
            ("dflash.dspark.markov_rank", u32_value(rank)),
            (
                "dflash.target_layers",
                Value::Array(features.iter().map(|&id| Value::I32(id as i32)).collect()),
            ),
            ("dflash.attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
            ("dflash.rope.freq_base", Value::F32(10_000.0)),
            ("dflash.final_logit_softcapping", Value::F32(30.0)),
            ("dflash.sample_from_anchor", Value::Bool(false)),
            ("dflash.attention.causal", Value::Bool(false)),
        ];
        let metadata = metadata.iter().map(|(k, v)| (*k, v)).collect::<Vec<_>>();
        let tensors = tensors
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect::<Vec<_>>();
        let mut file = tempfile::NamedTempFile::new()?;
        gguf_file::write(file.as_file_mut(), &metadata, &tensors)?;
        Ok(file)
    }

    #[test]
    fn dspark_drafts_from_injected_target_features() -> Result<()> {
        let stage = tiny_stage()?;
        let visits = stage.speculative_feature_limit() - 1;
        let features = [1, visits - 2, visits];
        let file = tiny_draft_file(&features)?;
        let draft = DSparkDraft::from_gguf(file.path(), &Device::Cpu, &stage)?;
        assert_eq!(draft.target_layers(), features);
        let mut cache = draft.new_cache();
        assert!(
            draft.draft(&cache, 3, 0, 4, &stage).is_err(),
            "no features yet"
        );

        let mut target: Vec<Option<LayerCache>> = (0..stage.layer_count()).map(|_| None).collect();
        let prompt = vec![3u32, 9, 14, 1, 22, 7];
        let (_, rows) = stage.forward_with_layer_inputs(
            StageInput::Tokens(prompt.clone()),
            &mut target,
            prompt.len(),
            0,
            &features,
            false,
        )?;
        // Two injections reach the same state as one.
        let mut split = draft.new_cache();
        let head = rows
            .iter()
            .map(|r| r.narrow(0, 0, 4))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let tail = rows
            .iter()
            .map(|r| r.narrow(0, 4, 2))
            .collect::<candle_core::Result<Vec<_>>>()?;
        draft.inject(&mut split, &head, 0)?;
        draft.inject(&mut split, &tail, 4)?;
        draft.inject(&mut cache, &rows, 0)?;
        assert!(
            draft.inject(&mut cache.clone(), &rows, 0).is_err(),
            "positions must follow"
        );
        assert_eq!(cache.next_position(), prompt.len());
        // The sliding layer keeps its window; the full layer every position.
        let lengths = |cache: &DSparkCache| {
            cache
                .layers
                .iter()
                .map(|kv| live(kv.as_ref().unwrap()).unwrap().0.dim(2).unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(lengths(&cache), [3, prompt.len()]);
        assert_eq!(lengths(&split), [3, prompt.len()]);

        let drafted = draft.draft(&cache, 5, prompt.len(), 4, &stage)?;
        assert_eq!(drafted.len(), 3);
        assert!(drafted.iter().all(|&id| id < 30));
        assert_eq!(draft.draft(&split, 5, prompt.len(), 4, &stage)?, drafted);
        assert_eq!(draft.draft(&cache, 5, prompt.len(), 2, &stage)?.len(), 1);
        assert!(draft.draft(&cache, 5, prompt.len(), 5, &stage).is_err());
        // Drafting leaves the cache as it was.
        assert_eq!(lengths(&cache), [3, prompt.len()]);
        let forked = cache.fork()?;
        assert_eq!(draft.draft(&forked, 5, prompt.len(), 4, &stage)?, drafted);
        Ok(())
    }

    #[test]
    fn dspark_rejects_features_the_target_lacks() -> Result<()> {
        let stage = tiny_stage()?;
        let visits = stage.speculative_feature_limit() - 1;
        let file = tiny_draft_file(&[1, visits + 1])?;
        assert!(DSparkDraft::from_gguf(file.path(), &Device::Cpu, &stage).is_err());
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_dspark_kernels_match_reference() -> Result<()> {
        use crate::architectures::dspark_metal;
        let Ok(metal) = Device::new_metal(0) else {
            return Ok(());
        };
        let mut seed = 42u32;
        let mut rand_vec = |count: usize, scale: f32| -> Vec<f32> {
            (0..count)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * scale
                })
                .collect()
        };

        // 1. Tiled GQA-12 non-causal attention (single-split and multi-split with start > 0).
        let (heads, kv_heads, rows, dim) = (24, 2, 8, 128);
        let scale = 1.0f32 / (dim as f32).sqrt();
        for &(total_keys, evict) in &[(17usize, 0usize), (313usize, 29usize)] {
            let q_cpu = Tensor::from_vec(
                rand_vec(heads * rows * dim, 1.0),
                (1, heads, rows, dim),
                &Device::Cpu,
            )?
            .to_dtype(DType::F16)?;
            let k_cpu = Tensor::from_vec(
                rand_vec(kv_heads * total_keys * dim, 1.0),
                (1, kv_heads, total_keys, dim),
                &Device::Cpu,
            )?
            .to_dtype(DType::F16)?;
            let v_cpu = Tensor::from_vec(
                rand_vec(kv_heads * total_keys * dim, 1.0),
                (1, kv_heads, total_keys, dim),
                &Device::Cpu,
            )?
            .to_dtype(DType::F16)?;
            let mut kv_cache =
                PreallocatedKvCache::from_f16(k_cpu.to_device(&metal)?, v_cpu.to_device(&metal)?)?;
            if evict > 0 {
                kv_cache.reserve(rows)?;
                let mut wrapped = KvCache::PreallocatedF16(kv_cache);
                wrapped.evict_oldest(evict)?;
                let KvCache::PreallocatedF16(updated) = wrapped else {
                    unreachable!()
                };
                kv_cache = updated;
            }
            let (k_live, v_live) = kv_cache.active_f16()?;
            let q_folded = q_cpu
                .reshape((1, heads * rows, 1, dim))?
                .to_device(&metal)?;
            let got = dspark_metal::attend_128(
                &q_folded, &k_live, &v_live, rows, heads, kv_heads, scale,
            )?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;

            let k_ref = repeat_kv(
                k_cpu
                    .narrow(2, evict, total_keys - evict)?
                    .to_dtype(DType::F32)?,
                heads / kv_heads,
            )?;
            let v_ref = repeat_kv(
                v_cpu
                    .narrow(2, evict, total_keys - evict)?
                    .to_dtype(DType::F32)?,
                heads / kv_heads,
            )?;
            let scores = (q_cpu
                .to_dtype(DType::F32)?
                .matmul(&k_ref.transpose(2, 3)?.contiguous()?)?
                * scale as f64)?;
            let want = candle_nn::ops::softmax_last_dim(&scores)?
                .matmul(&v_ref)?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((rows, heads * dim))?
                .to_vec2::<f32>()?;
            let max_diff = got
                .iter()
                .flatten()
                .zip(want.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-3,
                "attend_128 mismatch for keys={total_keys}, evict={evict}: max_diff={max_diff}"
            );
        }

        // 2. Multi-row Q8_0 matvec (R=2/NSG=2 for T=1..4 and R=1/NSG=4 for T=5..8, + SwiGLU).
        for &t in &[1usize, 4, 7, 8] {
            let (in_dim, out_dim) = (128usize, 64usize);
            let w_f32 = Tensor::from_vec(
                rand_vec(out_dim * in_dim, 1.0),
                (out_dim, in_dim),
                &Device::Cpu,
            )?;
            let qt = Arc::new(QTensor::quantize(&w_f32, GgmlDType::Q8_0)?);
            let w_deq = qt.dequantize(&Device::Cpu)?;
            let x_cpu = Tensor::from_vec(rand_vec(t * in_dim, 1.0), (t, in_dim), &Device::Cpu)?;
            let res_cpu = Tensor::from_vec(rand_vec(t * out_dim, 1.0), (t, out_dim), &Device::Cpu)?;
            let linear =
                MetalQ8Linear::from_qtensors(&[&qt], &metal)?.expect("Q8_0 linear on Metal");
            let got = linear
                .forward(
                    &x_cpu.to_device(&metal)?,
                    Some(&res_cpu.to_device(&metal)?),
                    false,
                )?
                .to_device(&Device::Cpu)?
                .to_vec2::<f32>()?;
            let want = (x_cpu.matmul(&w_deq.t()?)? + &res_cpu)?.to_vec2::<f32>()?;
            let max_diff = got
                .iter()
                .flatten()
                .zip(want.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-3,
                "q8_0 matvec mismatch for t={t}: {max_diff}"
            );

            let got_sw = linear
                .forward(&x_cpu.to_device(&metal)?, None, true)?
                .to_device(&Device::Cpu)?
                .to_vec2::<f32>()?;
            let proj = x_cpu.matmul(&w_deq.t()?)?;
            let gate = proj.narrow(1, 0, out_dim / 2)?;
            let up = proj.narrow(1, out_dim / 2, out_dim / 2)?;
            let want_sw = (candle_nn::ops::silu(&gate)? * up)?.to_vec2::<f32>()?;
            let max_diff_sw = got_sw
                .iter()
                .flatten()
                .zip(want_sw.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff_sw < 1e-3,
                "q8_0 swiglu mismatch for t={t}: {max_diff_sw}"
            );
        }

        // 3. Markov decode steps with F16 / Q8_0 / Q4_0 fold_t and GPU early stopping.
        let (vocab, rank, steps) = (128usize, 256usize, 7usize);
        let markov_w1 = Tensor::from_vec(rand_vec(vocab * rank, 0.5), (vocab, rank), &metal)?
            .to_dtype(DType::BF16)?;
        let joint_rec_t = Tensor::from_vec(
            rand_vec(3 * rank * 2 * rank, 0.2),
            (3 * rank, 2 * rank),
            &metal,
        )?;
        let context = Tensor::from_vec(rand_vec(steps * 3 * rank, 0.5), (steps, 3 * rank), &metal)?;
        // Make steps 0 and 1 sharply peaked, step 2 flat/uncertain so early stopping triggers.
        let mut base_vec = vec![0.0f32; steps * vocab];
        base_vec[0 * vocab + 11] = 12.0;
        base_vec[1 * vocab + 23] = 12.0;
        for step in 2..steps {
            base_vec[step * vocab + (step * 7) % vocab] = 0.2;
        }
        let base = Tensor::from_vec(base_vec, (steps, vocab), &metal)?;
        let fold_f16 = Tensor::from_vec(rand_vec(vocab * rank, 0.05), (vocab, rank), &metal)?
            .to_dtype(DType::F16)?;
        let fold_q8 = dspark_metal::quantize_f16_to_q8_0(&fold_f16)?;
        let fold_q4 = dspark_metal::quantize_f16_to_q4_0(&fold_f16)?;
        let fold_bias = Tensor::zeros((1, vocab), DType::F32, &metal)?;

        let (ids_f16, probs_f16) = dspark_metal::markov_decode_steps(
            5,
            steps,
            rank,
            vocab,
            30.0,
            0.0,
            0.0,
            &markov_w1,
            &joint_rec_t,
            &context,
            &base,
            &fold_f16,
            dspark_metal::FoldFormat::F16,
            &fold_bias,
        )?;
        assert_eq!(ids_f16.len(), steps);
        assert_eq!(ids_f16[0], 11);
        assert_eq!(ids_f16[1], 23);
        assert!(probs_f16[0] > 0.99 && probs_f16[1] > 0.99 && probs_f16[2] < 0.05);

        for (fold_buf, fmt) in [
            (&fold_q8, dspark_metal::FoldFormat::Q8_0),
            (&fold_q4, dspark_metal::FoldFormat::Q4_0),
        ] {
            let (ids_early, _) = dspark_metal::markov_decode_steps(
                5,
                steps,
                rank,
                vocab,
                30.0,
                17.5,
                2.7,
                &markov_w1,
                &joint_rec_t,
                &context,
                &base,
                fold_buf,
                fmt,
                &fold_bias,
            )?;
            assert_eq!(ids_early, vec![11, 23]);
        }

        // 4. Sampled Markov head for exact rejection sampling (Q4_0, Q8_0, F16).
        let step_u = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        for (fold_buf, fmt) in [
            (&fold_f16, dspark_metal::FoldFormat::F16),
            (&fold_q8, dspark_metal::FoldFormat::Q8_0),
            (&fold_q4, dspark_metal::FoldFormat::Q4_0),
        ] {
            let (ids_s, dists_s) = dspark_metal::markov_sample_steps(
                5,
                steps,
                rank,
                vocab,
                30.0,
                0.7,
                0,
                0.95,
                &step_u,
                17.5,
                2.7,
                &markov_w1,
                &joint_rec_t,
                &context,
                &base,
                fold_buf,
                fmt,
                &fold_bias,
            )?;
            assert!(!ids_s.is_empty() && ids_s.len() <= steps);
            assert_eq!(ids_s.len(), dists_s.len());
            assert_eq!(ids_s[0], 11);
            assert_eq!(ids_s[1], 23);
            for (tok, dist) in ids_s.iter().zip(&dists_s) {
                assert!(dist.iter().any(|&(id, _)| id == *tok));
                let sum: f32 = dist.iter().map(|&(_, p)| p).sum();
                assert!((sum - 1.0).abs() < 1e-4, "draft dist sum={sum}");
            }
        }
        Ok(())
    }
}
