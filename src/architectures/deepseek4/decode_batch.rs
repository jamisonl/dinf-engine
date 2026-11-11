//! Shared FFN and vocabulary projections across independent decode requests.
//! Attention and routed experts still use their single-request kernels.

use super::*;

pub(super) fn enabled(configured: Option<&std::ffi::OsStr>, apple_silicon_metal: bool) -> bool {
    configured.map_or(apple_silicon_metal, |value| value == "1")
}

impl DeepSeek4Stage {
    pub(super) fn decode_batch(
        &self,
        tokens: &[u32],
        caches: &mut [&mut [Option<LayerCache>]],
        batch_cache: &mut [Option<LayerCache>],
        positions: &[usize],
    ) -> Result<Vec<Tensor>> {
        anyhow::ensure!(
            self.supports_decode_batch(),
            "V4 decode batching unavailable"
        );
        anyhow::ensure!(!tokens.is_empty(), "V4 decode batch cannot be empty");
        anyhow::ensure!(
            tokens.len() == caches.len(),
            "V4 decode batch lane mismatch"
        );
        anyhow::ensure!(
            tokens.len() == positions.len(),
            "V4 decode batch position mismatch"
        );
        anyhow::ensure!(
            batch_cache.len() == self.blocks.len()
                && caches.iter().all(|cache| cache.len() == self.blocks.len()),
            "V4 decode batch cache layer mismatch"
        );
        anyhow::ensure!(
            tokens
                .iter()
                .all(|&token| (token as usize) < self.vocab_size),
            "V4 decode batch token outside vocabulary"
        );
        anyhow::ensure!(
            positions
                .iter()
                .all(|&position| position < self.context_length),
            "V4 decode batch position outside context"
        );
        // Window and pool chunks are immutable. Update only checkpoints until
        // every layer and output succeeds; failures leave all requests intact.
        let mut next_caches = caches
            .iter()
            .zip(positions)
            .map(|(cache, &position)| {
                cache
                    .iter()
                    .map(|slot| {
                        anyhow::ensure!(
                            match slot {
                                None => position == 0,
                                Some(LayerCache::DeepSeek4(state)) => state.position() == position,
                                _ => false,
                            },
                            "V4 decode batch cache type or position mismatch"
                        );
                        slot.as_ref()
                            .map(super::super::checkpoint_layer_cache)
                            .transpose()
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;

        let embedded = self
            .embedding
            .as_ref()
            .context("V4 embedding is not loaded")?
            .forward(tokens, &self.device)?;
        let mut hidden = (0..tokens.len())
            .map(|lane| {
                Ok(embedded
                    .narrow(0, lane, 1)?
                    .reshape((1, 1, 1, self.model_hidden_dim))?
                    .repeat((1, 1, self.stream_count, 1))?)
            })
            .collect::<Result<Vec<_>>>()?;
        for (layer, block) in self.blocks.iter().enumerate() {
            let mut ffn_inputs = Vec::with_capacity(tokens.len());
            let mut controls = Vec::with_capacity(tokens.len());
            for (lane, hidden) in hidden.iter_mut().enumerate() {
                let (input, post, comb) = block.attn_hc.pre(hidden)?;
                let input = block.attn_norm.forward(&input)?;
                let output = block
                    .attention
                    .forward(&input, &mut next_caches[lane][layer], 1, positions[lane])
                    .with_context(|| format!("V4 batch layer {layer} lane {lane} attention"))?;
                *hidden = block.attn_hc.post(&output, hidden, &post, &comb)?;
                let (input, post, comb) = block.ffn_hc.pre(hidden)?;
                ffn_inputs.push(block.ffn_norm.forward(&input)?);
                controls.push((post, comb));
            }
            let input = Tensor::cat(&ffn_inputs, 1)?;
            let output = block
                .moe
                .forward_decode_rows(&input, tokens)
                .with_context(|| format!("V4 batch layer {layer} MoE"))?;
            for (lane, (hidden, (post, comb))) in hidden.iter_mut().zip(controls).enumerate() {
                *hidden = block
                    .ffn_hc
                    .post(&output.narrow(1, lane, 1)?, hidden, &post, &comb)?;
            }
        }
        let output_hc = self
            .output_hc
            .as_ref()
            .context("V4 output HC is not loaded")?;
        let output_norm = self
            .output_norm
            .as_ref()
            .context("V4 output norm is not loaded")?;
        let rows = hidden
            .iter()
            .map(|hidden| {
                output_norm
                    .forward(&output_hc.forward(hidden)?)?
                    .squeeze(0)
                    .map_err(Into::into)
            })
            .collect::<Result<Vec<_>>>()?;
        let logits = self
            .lm_head
            .as_ref()
            .context("V4 output head is not loaded")?
            .forward_decode_rows(&Tensor::cat(&rows, 0)?)?;
        let outputs = (0..tokens.len())
            .map(|lane| logits.get(lane).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;
        for (cache, next) in caches.iter_mut().zip(next_caches) {
            for (slot, state) in cache.iter_mut().zip(next) {
                *slot = state;
            }
        }
        // No packed cache retains another request's window or pooled keys.
        for slot in batch_cache {
            *slot = None;
        }
        Ok(outputs)
    }
}

#[cfg(test)]
#[path = "decode_batch_tests.rs"]
mod tests;
