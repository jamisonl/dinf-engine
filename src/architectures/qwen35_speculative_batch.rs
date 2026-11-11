//! Shared matrix work for independent draft blocks. Scans and attention never
//! cross a request boundary; each lane retains its own speculative snapshots.

use super::super::{SpeculativeBatchOutput, checkpoint_layer_cache};
use super::*;

thread_local! {
    static ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(target_os = "macos")]
pub(super) fn active() -> bool {
    ACTIVE.get()
}

pub(super) fn with_batch<T>(f: impl FnOnce() -> T) -> T {
    struct Guard(bool);
    impl Drop for Guard {
        fn drop(&mut self) {
            ACTIVE.set(self.0);
        }
    }
    let _guard = Guard(ACTIVE.replace(true));
    f()
}

impl Qwen35Stage {
    pub(super) fn verify_batch(
        &self,
        inputs: &[Vec<u32>],
        caches: &mut [&mut [Option<LayerCache>]],
        positions: &[usize],
        layer_ids: &[usize],
    ) -> Result<Vec<SpeculativeBatchOutput>> {
        anyhow::ensure!(
            self.supports_speculative_batch(),
            "Qwen draft batching unavailable"
        );
        let lanes = inputs.len();
        let rows = inputs.first().map_or(0, Vec::len);
        anyhow::ensure!(
            (2..=4).contains(&lanes)
                && rows == 8
                && inputs.iter().all(|input| input.len() == rows)
                && caches.len() == lanes
                && positions.len() == lanes,
            "invalid Qwen draft batch geometry"
        );
        anyhow::ensure!(
            layer_ids.windows(2).all(|pair| pair[0] < pair[1])
                && layer_ids.iter().all(|id| self.layer_indices.contains(id)),
            "invalid Qwen draft capture layers"
        );
        let mut next = caches
            .iter()
            .map(|cache| {
                anyhow::ensure!(
                    cache.len() == self.blocks.len(),
                    "draft cache layer mismatch"
                );
                cache
                    .iter()
                    .zip(&self.blocks)
                    .map(|(slot, block)| {
                        anyhow::ensure!(
                            matches!(
                                (slot, block),
                                (None, _)
                                    | (Some(LayerCache::DeltaNet(_)), Qwen35Block::DeltaNet(_))
                                    | (
                                        Some(LayerCache::Attention(_)),
                                        Qwen35Block::GatedAttention(_)
                                    )
                            ),
                            "draft cache type mismatch"
                        );
                        slot.as_ref().map(checkpoint_layer_cache).transpose()
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let tokens = inputs.iter().flatten().copied().collect::<Vec<_>>();
        let mut hidden = self
            .embed_tokens(&tokens)?
            .reshape((1, lanes * rows, self.hidden_dim))?;
        let mut captures = (0..lanes)
            .map(|_| Vec::with_capacity(layer_ids.len()))
            .collect::<Vec<_>>();
        for (layer, block) in self.blocks.iter().enumerate() {
            if layer_ids.contains(&self.layer_indices[layer]) {
                for (lane, captures) in captures.iter_mut().enumerate() {
                    captures.push(super::super::copy_persistent_tensor(
                        &hidden
                            .narrow(1, lane * rows, rows)?
                            .squeeze(0)?
                            .to_dtype(DType::F32)?,
                    )?);
                }
            }
            hidden = match block {
                Qwen35Block::DeltaNet(block) => {
                    let normed = block.attn_norm.forward(&hidden)?;
                    let (qkv, gate, alpha, beta) = block.projections.forward(
                        &normed,
                        lanes * rows,
                        false,
                        block.conv_dim,
                        block.value_dim,
                        block.num_v_heads,
                    )?;
                    let mut outputs = Vec::with_capacity(lanes);
                    for (lane, cache) in next.iter_mut().enumerate() {
                        let mut state = match cache[layer].take() {
                            Some(LayerCache::DeltaNet(state)) => Some(state),
                            None => None,
                            _ => unreachable!("cache types checked above"),
                        };
                        outputs.push(block.scan_and_gate(
                            (
                                qkv.narrow(0, lane * rows, rows)?.contiguous()?,
                                gate.narrow(1, lane * rows, rows)?.contiguous()?,
                                alpha.narrow(0, lane * rows, rows)?.contiguous()?,
                                beta.narrow(0, lane * rows, rows)?.contiguous()?,
                            ),
                            &mut state,
                            true,
                        )?);
                        cache[layer] = state.map(LayerCache::DeltaNet);
                    }
                    block.output_and_ffn(&hidden, &Tensor::cat(&outputs, 1)?, false)?
                }
                Qwen35Block::GatedAttention(block) => {
                    let normed = block.attn_norm.forward(&hidden)?;
                    let (q, k, v) = block.input_projections.forward(
                        &normed,
                        lanes * rows,
                        false,
                        block.num_heads * block.head_dim * 2,
                        block.num_kv_heads * block.head_dim,
                    )?;
                    let mut outputs = Vec::with_capacity(lanes);
                    for (lane, cache) in next.iter_mut().enumerate() {
                        let mut state = match cache[layer].take() {
                            Some(LayerCache::Attention(state)) => Some(state),
                            None => None,
                            _ => unreachable!("cache types checked above"),
                        };
                        outputs.push(block.attention_and_gate(
                            (
                                q.narrow(1, lane * rows, rows)?.contiguous()?,
                                k.narrow(1, lane * rows, rows)?.contiguous()?,
                                v.narrow(1, lane * rows, rows)?.contiguous()?,
                            ),
                            &mut state,
                            positions[lane],
                            None,
                        )?);
                        cache[layer] = state.map(LayerCache::Attention);
                    }
                    block.output_and_ffn(&hidden, &Tensor::cat(&outputs, 1)?, false)?
                }
            };
        }
        let normed = self
            .output_norm
            .as_ref()
            .context("missing Qwen output norm")?
            .forward(&hidden)?;
        let logits = self
            .lm_head
            .as_ref()
            .context("missing Qwen output head")?
            .forward(&normed.squeeze(0)?)?;
        let outputs = captures
            .into_iter()
            .enumerate()
            .map(|(lane, captures)| {
                Ok(SpeculativeBatchOutput {
                    logits: logits.narrow(0, lane * rows, rows)?.contiguous()?,
                    captures,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // Only unused KV storage was written. Publish all layer lengths and
        // recurrent state together after every output and capture succeeded.
        for (cache, next) in caches.iter_mut().zip(next) {
            for (slot, value) in cache.iter_mut().zip(next) {
                *slot = value;
            }
        }
        Ok(outputs)
    }
}
