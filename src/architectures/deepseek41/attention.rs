//! Maya raw-window + shared compressed/index attention. See NOTICE and LICENSE-MIT.
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::{D, Device, Tensor};

use super::super::deepseek4_qmatmul::DeepSeek4QMatMul;
use super::cache::{self, Compressed, DeepSeek41Cache, Prefix};
use super::math::{self, CacheFormat, RMS_EPS, Rope};
use super::ops;
use super::weights::{self, Weights};

type Linear = DeepSeek4QMatMul;

struct Compressor {
    kv: Linear,
    gate: Option<Linear>,
    norm: Tensor,
    index: Linear,
    index_norm: Tensor,
}

struct Indexer {
    q: Linear,
    weights: Linear,
}

pub(crate) struct Attention {
    qa: Linear,
    qa_norm: Tensor,
    qb: Linear,
    kv: Linear,
    kv_norm: Tensor,
    sinks: Tensor,
    output_a: Linear,
    output_b: Linear,
    compressor: Option<Compressor>,
    indexer: Option<Indexer>,
    rope: Rope,
}

/// Selections and candidate masks have current-token lifetime, never session
/// lifetime. Non-refresh layers reuse their owner's most recent selection.
pub(crate) struct Selection {
    pub(super) selected: [Option<Tensor>; 4],
    pub(super) candidate_mask: Option<Tensor>,
}

impl Default for Selection {
    fn default() -> Self {
        Self {
            selected: std::array::from_fn(|_| None),
            candidate_mask: None,
        }
    }
}

impl Selection {
    /// Tile-only retention boundary, before advancing to another row/layer.
    pub(super) fn retain_bounded(&mut self) -> Result<()> {
        for value in self
            .selected
            .iter_mut()
            .chain(std::iter::once(&mut self.candidate_mask))
            .flatten()
        {
            *value = cache::retain_bounded(value.clone())?;
        }
        Ok(())
    }

    pub(super) fn tensors(&self) -> impl Iterator<Item = &Tensor> {
        self.selected
            .iter()
            .filter_map(Option::as_ref)
            .chain(self.candidate_mask.iter())
    }
}

impl Attention {
    pub(crate) fn load(w: &Weights, l: usize, d: &Device) -> Result<Self> {
        let p = format!("blk.{l}.");
        let linear = |n: &str| weights::linear(w, &format!("{p}{n}.weight"), d);
        let dense = |n: &str| weights::dense(w, &format!("{p}{n}.weight"), d);
        let output_a = Linear::new_v41_on_device(
            Arc::clone(weights::raw(w, &format!("{p}attn_output_a.weight"))?),
            d,
        )?;
        Ok(Self {
            qa: linear("attn_q_a")?,
            qa_norm: dense("attn_q_a_norm")?,
            qb: linear("attn_q_b")?,
            kv: linear("attn_kv")?,
            kv_norm: dense("attn_kv_a_norm")?,
            sinks: dense("attn_sinks")?,
            output_a,
            output_b: linear("attn_output_b")?,
            compressor: if cache::OWNERS.contains(&l) {
                Some(Compressor {
                    kv: linear("attn_compressor_kv")?,
                    gate: if l == 20 {
                        None
                    } else {
                        Some(linear("attn_compressor_gate")?)
                    },
                    norm: dense("attn_compressor_norm")?,
                    index: linear("indexer.attn_k")?,
                    index_norm: dense("indexer.k_norm")?,
                })
            } else {
                None
            },
            indexer: if cache::refresh(l) {
                Some(Indexer {
                    q: linear("indexer.attn_q_b")?,
                    weights: linear("indexer.proj")?,
                })
            } else {
                None
            },
            rope: Rope::for_layer(l)?,
        })
    }

    pub(crate) fn forward(
        &self,
        x: &Tensor,
        layer: usize,
        pos: usize,
        cache: &mut DeepSeek41Cache,
        selection: &mut Selection,
    ) -> Result<Tensor> {
        self.forward_inner(x, layer, pos, cache, selection, false)
    }

    pub(crate) fn forward_tiled(
        &self,
        x: &Tensor,
        layer: usize,
        pos: usize,
        cache: &mut DeepSeek41Cache,
        selection: &mut Selection,
    ) -> Result<Tensor> {
        self.forward_inner(x, layer, pos, cache, selection, true)
    }

    fn forward_inner(
        &self,
        x: &Tensor,
        layer: usize,
        pos: usize,
        cache: &mut DeepSeek41Cache,
        selection: &mut Selection,
        tiled: bool,
    ) -> Result<Tensor> {
        let qa = math::rms_norm(&math::bf16(&self.qa.forward(x)?)?, &self.qa_norm, RMS_EPS)?;
        let kv = math::rms_norm(&math::bf16(&self.kv.forward(x)?)?, &self.kv_norm, RMS_EPS)?;
        let q = math::bf16(&self.qb.forward(&qa)?)?.reshape((64, 512))?;
        let low = self.attend_prepared(x, &qa, &kv, &q, layer, pos, cache, selection, tiled)?;
        Ok(math::bf16(&self.output_b.forward(&low)?)?.reshape(5120)?)
    }

    /// Independent projections only. The caller pins every returned carrier
    /// before enqueueing another phase; publication remains strictly row ordered.
    pub(super) fn prepare_qa(&self, x: &Tensor) -> Result<Tensor> {
        Ok(math::rms_norm(
            &math::bf16(&self.qa.forward_attention_prefill(x)?)?,
            &self.qa_norm,
            RMS_EPS,
        )?)
    }

    pub(super) fn prepare_kv(&self, x: &Tensor) -> Result<Tensor> {
        Ok(math::rms_norm(
            &math::bf16(&self.kv.forward_attention_prefill(x)?)?,
            &self.kv_norm,
            RMS_EPS,
        )?)
    }

    pub(super) fn prepare_q(&self, qa: &Tensor) -> Result<Tensor> {
        Ok(math::bf16(&self.qb.forward_attention_prefill(qa)?)?)
    }

    pub(super) fn project_output(&self, low: &Tensor) -> Result<Tensor> {
        Ok(math::bf16(&self.output_b.forward_attention_prefill(low)?)?)
    }

    /// Candidate tile preparation publishes ONLY compressed state, once. Raw
    /// windows still advance row-by-row. The caller's cache is already private.
    /// Page owners are dynamic KV (not tile scratch); all transient/copy owners
    /// are kept separately in `pins` through the existing router/error fence.
    #[allow(
        clippy::too_many_arguments,
        reason = "Keep tensor borrows and each GPU owner pool explicit across dispatch."
    )]
    pub(super) fn prepare_index_tile(
        &self,
        x: &Tensor,
        qa: &Tensor,
        layer: usize,
        base: usize,
        cache: &mut DeepSeek41Cache,
        pages: &mut Vec<Tensor>,
        cache_owners: &mut Vec<Tensor>,
        pins: &mut Vec<Tensor>,
    ) -> Result<Option<Tensor>> {
        let Some(owner) = cache::owner(layer) else {
            return Ok(None);
        };
        let count = x.dim(0)?;
        anyhow::ensure!(
            count > 0 && count <= super::profile::ATTENTION_V2_MAX_ROWS && qa.dim(0)? == count,
            "Maya candidate tile shape"
        );
        let compressed = &mut cache.compressed[owner];
        if let Some(compressor) = &self.compressor {
            anyhow::ensure!(
                compressed.kv.len() == base / if owner == 3 { 1 } else { 2 },
                "Maya candidate duplicate publication"
            );
            compressor.publish_many(x, base, &self.rope, compressed, pins)?;
        }
        cache_owners.extend(compressed.kv.chunks.iter().cloned());
        cache_owners.extend(compressed.index.chunks.iter().cloned());
        if let Some((kv, gate)) = &compressed.pending {
            cache_owners.push(kv.clone());
            cache_owners.push(gate.clone());
        }
        let Some(indexer) = &self.indexer else {
            return Ok(None);
        };
        let visible: Vec<usize> = (0..count)
            .map(|r| cache::visible_rows(owner, base + r))
            .collect();
        let last = *visible.last().unwrap();
        if last == 0 {
            return Ok(None);
        }
        // Publish owners before any scoring enqueue. Pages may include future
        // rows; the shader checks each query's bound BEFORE loading its key.
        *pages = compressed
            .index
            .prefix(last)?
            .chunks()
            .collect::<candle_core::Result<_>>()?;
        let q = pin(pins, indexer.q.forward_prefill_matrix(qa)?)?;
        let q = pin(pins, math::bf16(&q)?)?;
        let weights = pin(pins, indexer.weights.forward_prefill_matrix(x)?)?;
        let weights = pin(pins, math::bf16(&weights)?.reshape((count, 32))?)?;
        let mut queries = Vec::with_capacity(count);
        for r in 0..count {
            let row = q.narrow(0, r, 1)?.reshape((32, 128))?;
            let row = pin(
                pins,
                self.rope.apply(&row, (base + r) as u32, false, false)?,
            )?;
            queries.push(
                pin(
                    pins,
                    math::quantize_dequantize(&row, CacheFormat::IndexQkFp4E8M0)?,
                )?
                .reshape((1, 32, 128))?,
            );
        }
        let q = cache::cat_rows(&queries, pins)?;
        Ok(Some(ops::index_scores_many(
            &q, &weights, pages, &visible, pins,
        )?))
    }

    /// Explicit no-publication mode: never call Compressor::publish here.
    /// A score row is clipped BEFORE candidate blocks/topk/masks are formed.
    #[allow(
        clippy::too_many_arguments,
        reason = "Keep borrowed projections, cache, and selection explicit at the publication boundary."
    )]
    pub(super) fn attend_published(
        &self,
        x: &Tensor,
        qa: &Tensor,
        kv: &Tensor,
        q: &Tensor,
        layer: usize,
        pos: usize,
        cache: &mut DeepSeek41Cache,
        selection: &mut Selection,
        scores: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.attend_prepared_mode(
            x, qa, kv, q, layer, pos, cache, selection, true, true, scores,
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "Keep borrowed projections and their live cache separate across attention dispatch."
    )]
    pub(super) fn attend_prepared(
        &self,
        x: &Tensor,
        qa: &Tensor,
        kv: &Tensor,
        q: &Tensor,
        layer: usize,
        pos: usize,
        cache: &mut DeepSeek41Cache,
        selection: &mut Selection,
        tiled: bool,
    ) -> Result<Tensor> {
        self.attend_prepared_mode(
            x, qa, kv, q, layer, pos, cache, selection, tiled, false, None,
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "Keep the borrowed GPU inputs and independent publication controls explicit."
    )]
    fn attend_prepared_mode(
        &self,
        x: &Tensor,
        qa: &Tensor,
        kv: &Tensor,
        q: &Tensor,
        layer: usize,
        pos: usize,
        cache: &mut DeepSeek41Cache,
        selection: &mut Selection,
        tiled: bool,
        published: bool,
        scores: Option<&Tensor>,
    ) -> Result<Tensor> {
        let q = self
            .rope
            .apply(&q.reshape((64, 512))?, pos as u32, false, false)?;
        let raw = math::quantize_dequantize(
            &self.rope.apply(kv, pos as u32, false, false)?,
            CacheFormat::RawKvFp8E8M0,
        )?;
        cache.append_raw(layer, raw)?;
        let mut values = cache.raw[layer].as_ref().unwrap().clone();
        if let Some(owner) = cache::owner(layer) {
            let compressed = &mut cache.compressed[owner];
            if !published && let Some(compressor) = &self.compressor {
                compressor.publish(x, pos, &self.rope, compressed)?;
            }
            let visible = if tiled {
                cache::visible_rows(owner, pos)
            } else {
                compressed.kv.len()
            };
            let kv_prefix = compressed.kv.prefix(visible)?;
            let index_prefix = compressed.index.prefix(visible)?;
            if visible > 0 {
                if let Some(indexer) = &self.indexer {
                    let ids = if published {
                        let scores = scores.context("Maya prepared index scores missing")?;
                        select_scores(
                            &scores.flatten_all()?.narrow(0, 0, visible)?,
                            layer,
                            selection,
                        )?
                    } else {
                        indexer.select(x, qa, layer, pos, &self.rope, &index_prefix, selection)?
                    };
                    selection.selected[owner] = Some(ids);
                }
                let ids = selection.selected[owner]
                    .as_ref()
                    .context("Maya owner selection missing")?;
                let gathered = kv_prefix.gather(ids, 512)?;
                values = Tensor::cat(&[&values, &gathered], 0)?;
            }
        }
        let heads = heads(&q, &values, &self.sinks)?;
        let heads = self.rope.apply(&heads, pos as u32, true, true)?;
        Ok(math::bf16(
            &self
                .output_a
                .forward_grouped_packed(&heads.reshape((8, 4096))?, 8)?,
        )?)
    }
}

/// Two-row channelwise pooling. No projection rounding before the softmax;
/// pending rows stay F32 and survive an odd prefill boundary.
pub(crate) fn pool_pair(a: &Tensor, sa: &Tensor, b: &Tensor, sb: &Tensor) -> Result<Tensor> {
    let peak = sa.maximum(sb)?;
    let ea = sa.sub(&peak)?.exp()?;
    let eb = sb.sub(&peak)?.exp()?;
    Ok(math::bf16(
        &a.mul(&ea)?.add(&b.mul(&eb)?)?.div(&ea.add(&eb)?)?,
    )?)
}

/// Publish only complete pairs; RoPE uses the first token's position. Keeping
/// this transition separate from the projections lets small fixtures exercise
/// chunk/fork/error behavior without allocating production weight matrices.
pub(crate) fn pool_projected(
    cache: &mut Compressed,
    kv: Tensor,
    score: Option<Tensor>,
    pos: usize,
) -> Result<Option<(Tensor, usize)>> {
    let Some(score) = score else {
        return Ok(Some((math::bf16(&kv)?, pos)));
    };
    if pos.is_multiple_of(2) {
        cache.pending = Some((cache::retain_bounded(kv)?, cache::retain_bounded(score)?));
        return Ok(None);
    }
    let (previous, previous_score) = cache
        .pending
        .take()
        .context("missing Maya odd-pair carry")?;
    Ok(Some((
        pool_pair(&previous, &previous_score, &kv, &score)?,
        pos - 1,
    )))
}

#[track_caller]
fn pin(pins: &mut Vec<Tensor>, tensor: Tensor) -> Result<Tensor> {
    // Own even rejected pooled sources until the caller's existing error fence.
    pins.push(tensor.clone());
    cache::require_bounded_loan(&tensor)?;
    Ok(tensor)
}

/// Ordered pooling transition shared by candidate publication and small fixtures.
/// Missing carry/shape errors cannot consume the incoming pending pair.
pub(crate) fn pool_projected_tile(
    cache: &mut Compressed,
    kv: &Tensor,
    gate: Option<&Tensor>,
    base: usize,
    pins: &mut Vec<Tensor>,
) -> Result<(Vec<Tensor>, Vec<usize>)> {
    anyhow::ensure!(
        kv.rank() == 2
            && kv.dtype() == candle_core::DType::F32
            && kv.dim(0)? > 0
            && base.checked_add(kv.dim(0)?).is_some(),
        "invalid Maya pooling tile"
    );
    if let Some(gate) = gate {
        anyhow::ensure!(
            gate.dims() == kv.dims()
                && gate.dtype() == kv.dtype()
                && gate.device().same_device(kv.device()),
            "Maya pooling gate mismatch"
        );
    }
    anyhow::ensure!(
        cache.pending.is_some() == (gate.is_some() && base % 2 == 1),
        "Maya pooling incoming carry mismatch"
    );
    pins.push(kv.clone());
    if let Some(gate) = gate {
        pins.push(gate.clone());
    }
    let mut next = cache.clone();
    if let Some((kv, gate)) = &next.pending {
        pins.push(kv.clone());
        pins.push(gate.clone());
    }
    let mut latents = Vec::with_capacity(kv.dim(0)?);
    let mut positions = Vec::with_capacity(kv.dim(0)?);
    for r in 0..kv.dim(0)? {
        let score = gate.map(|g| g.narrow(0, r, 1)).transpose()?;
        if let Some((latent, pos)) =
            pool_projected(&mut next, kv.narrow(0, r, 1)?, score, base + r)?
        {
            latents.push(pin(pins, latent)?);
            positions.push(pos);
        }
        if let Some((kv, gate)) = &next.pending {
            pins.push(kv.clone());
            pins.push(gate.clone());
        }
    }
    cache.pending = next.pending;
    Ok((latents, positions))
}

impl Compressor {
    fn publish_many(
        &self,
        x: &Tensor,
        base: usize,
        rope: &Rope,
        cache: &mut Compressed,
        pins: &mut Vec<Tensor>,
    ) -> Result<()> {
        let mut next = cache.clone();
        let kv = pin(pins, self.kv.forward_prefill_matrix(x)?)?;
        let gate = self
            .gate
            .as_ref()
            .map(|g| -> Result<Tensor> { pin(pins, g.forward_prefill_matrix(x)?) })
            .transpose()?;
        let (latents, positions) = pool_projected_tile(&mut next, &kv, gate.as_ref(), base, pins)?;
        if !latents.is_empty() {
            let latent = cache::cat_rows(&latents, pins)?;
            let latent = pin(pins, math::rms_norm(&latent, &self.norm, RMS_EPS)?)?;
            let keys = pin(pins, self.index.forward_prefill_matrix(&latent)?)?;
            let keys = pin(pins, math::bf16(&keys)?)?;
            let keys = pin(pins, math::rms_norm(&keys, &self.index_norm, RMS_EPS)?)?;
            let mut index_rows = Vec::with_capacity(positions.len());
            let mut kv_rows = Vec::with_capacity(positions.len());
            for (r, position) in positions.into_iter().enumerate() {
                let key = pin(
                    pins,
                    rope.apply(&keys.narrow(0, r, 1)?, position as u32, false, false)?,
                )?;
                index_rows.push(pin(
                    pins,
                    math::quantize_dequantize(&key, CacheFormat::IndexQkFp4E8M0)?,
                )?);
                let value = pin(
                    pins,
                    rope.apply(&latent.narrow(0, r, 1)?, position as u32, false, false)?,
                )?;
                kv_rows.push(pin(
                    pins,
                    math::quantize_dequantize(&value, CacheFormat::CompressedKvFp4E4M3)?,
                )?);
            }
            let values = cache::cat_rows(&kv_rows, pins)?;
            let indices = cache::cat_rows(&index_rows, pins)?;
            next.kv.append_many(values, pins)?;
            next.index.append_many(indices, pins)?;
        }
        // No half-published KV/index pair on any projection/allocation error.
        *cache = next;
        Ok(())
    }

    fn publish(&self, x: &Tensor, pos: usize, rope: &Rope, cache: &mut Compressed) -> Result<()> {
        let kv = self.kv.forward(x)?;
        let score = self.gate.as_ref().map(|gate| gate.forward(x)).transpose()?;
        let Some((latent, position)) = pool_projected(cache, kv, score, pos)? else {
            return Ok(());
        };
        let latent = math::rms_norm(&latent, &self.norm, RMS_EPS)?;
        let key = math::bf16(&self.index.forward(&latent)?)?;
        let key = math::rms_norm(&key, &self.index_norm, RMS_EPS)?;
        let key = math::quantize_dequantize(
            &rope.apply(&key, position as u32, false, false)?,
            CacheFormat::IndexQkFp4E8M0,
        )?;
        let kv = math::quantize_dequantize(
            &rope.apply(&latent, position as u32, false, false)?,
            CacheFormat::CompressedKvFp4E4M3,
        )?;
        cache.kv.append(kv)?;
        cache.index.append(key)?;
        Ok(())
    }
}

impl Indexer {
    #[allow(
        clippy::too_many_arguments,
        reason = "Keep query inputs and the cache prefix borrowed across GPU dispatch."
    )]
    fn select(
        &self,
        x: &Tensor,
        qa: &Tensor,
        layer: usize,
        pos: usize,
        rope: &Rope,
        keys: &Prefix<'_>,
        selection: &mut Selection,
    ) -> Result<Tensor> {
        let q = math::bf16(&self.q.forward(qa)?)?.reshape((32, 128))?;
        let q = math::quantize_dequantize(
            &rope.apply(&q, pos as u32, false, false)?,
            CacheFormat::IndexQkFp4E8M0,
        )?;
        let weights = math::bf16(&self.weights.forward(x)?)?.reshape((32, 1))?;
        select_scores(&score_prefix(&q, &weights, keys)?, layer, selection)
    }
}

/// Shared CPU-fixture/production path: iteration is prefix-clipped before
/// scores/topk/mask exist. Preserve the original singleton GEMV boundary:
/// Candle's N=1 GEMV and N>1 GEMM need not produce identical F32 bits.
pub(crate) fn score_prefix(q: &Tensor, weights: &Tensor, keys: &Prefix<'_>) -> Result<Tensor> {
    let mut chunks = Vec::with_capacity(keys.len().div_ceil(cache::CHUNK_ROWS));
    let score = |chunk: &Tensor| -> Result<Tensor> {
        let dot = q.matmul(&chunk.t()?)?.relu()?;
        Ok(dot
            .broadcast_mul(weights)?
            .sum(0)?
            .affine(1.0 / 64.0, 0.0)?)
    };
    for chunk in keys.chunks() {
        let chunk = chunk?;
        let n = chunk.dim(0)?;
        if n > 1 && n % cache::CHUNK_ROWS == 1 {
            chunks.push(score(&chunk.narrow(0, 0, n - 1)?)?);
            chunks.push(score(&chunk.narrow(0, n - 1, 1)?)?);
        } else {
            chunks.push(score(&chunk)?);
        }
    }
    Ok(Tensor::cat(&chunks, 0)?)
}

pub(crate) fn select_scores(
    scores: &Tensor,
    layer: usize,
    selection: &mut Selection,
) -> Result<Tensor> {
    let n = scores.elem_count();
    let mut filtered = scores.clone();
    if n > 16384 {
        if layer == 20 {
            let blocks = ops::blocks(scores)?;
            let ids = ops::topk(&blocks, 2048)?;
            selection.candidate_mask = Some(ops::mask(&ids, blocks.elem_count())?);
            // Layer20 selects globally. Only later refreshes restrict candidates.
        } else if layer > 20 {
            filtered = ops::filter(
                scores,
                selection
                    .candidate_mask
                    .as_ref()
                    .context("missing layer20 candidate mask")?,
            )?;
        }
    }
    Ok(ops::topk(&filtered, n.min(512))?)
}

pub(crate) fn heads(q: &Tensor, values: &Tensor, sinks: &Tensor) -> Result<Tensor> {
    let scores = q
        .matmul(&values.t()?)?
        .affine(1.0 / (512.0f64).sqrt(), 0.0)?;
    let scores = Tensor::cat(&[&scores, &sinks.reshape((64, 1))?], 1)?;
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    let probs = probs.narrow(D::Minus1, 0, values.dim(0)?)?.contiguous()?;
    Ok(math::bf16(&probs.matmul(values)?)?)
}
