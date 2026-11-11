//! Independent attention state with shared decode projections across requests.

use super::*;

impl Glm5NextStage {
    pub(super) fn decode_pipeline_batch(
        &self,
        inputs: &[StageInput],
        caches: &mut [&mut [Option<LayerCache>]],
        batch_cache: &mut [Option<LayerCache>],
        positions: &[usize],
    ) -> Result<Vec<StageOutput>> {
        anyhow::ensure!(
            self.supports_pipeline_decode_batch(),
            "GLM5 split batching unavailable"
        );
        anyhow::ensure!(
            !inputs.is_empty() && inputs.len() == caches.len() && inputs.len() == positions.len(),
            "GLM5 split batch lane mismatch"
        );
        anyhow::ensure!(
            batch_cache.len() == self.blocks.len()
                && caches.iter().all(|c| c.len() == self.blocks.len()),
            "GLM5 split batch cache layer mismatch"
        );
        for ((input, cache), &position) in inputs.iter().zip(caches.iter()).zip(positions) {
            anyhow::ensure!(
                position < self.context_length,
                "GLM5 split batch position outside context"
            );
            match input {
                StageInput::Tokens(tokens) => anyhow::ensure!(
                    self.is_head && tokens.len() == 1 && (tokens[0] as usize) < self.vocab,
                    "GLM5 split batch requires one valid head token"
                ),
                StageInput::Activations {
                    tensor,
                    bidirectional_spans,
                    positions,
                } => anyhow::ensure!(
                    !self.is_head
                        && tensor.dims() == [1, self.stage_hidden]
                        && tensor.dtype() == DType::F32
                        && tensor.device().same_device(&self.device)
                        && bidirectional_spans.is_empty()
                        && positions.is_none(),
                    "GLM5 split batch requires one flat F32 activation row without position metadata"
                ),
            }
            for (slot, block) in cache.iter().zip(&self.blocks) {
                self.check_decode_cache(slot, block, position)?;
            }
        }
        let result = (|| {
            let hidden = inputs
                .iter()
                .map(|input| match input {
                    StageInput::Tokens(tokens) => Ok(self
                        .embed_tokens(tokens)?
                        .to_dtype(DType::F32)?
                        .reshape((1, 1, 1, self.hidden))?
                        .repeat((1, 1, GLM5_STREAMS, 1))?),
                    StageInput::Activations { tensor, .. } => {
                        Ok(tensor.reshape((1, 1, GLM5_STREAMS, self.hidden))?)
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            self.decode_hidden_batch(hidden, caches)
        })();
        if result.is_err() {
            for cache in caches {
                for slot in cache.iter_mut() {
                    *slot = None;
                }
            }
        }
        for slot in batch_cache {
            *slot = None;
        }
        result
    }

    pub(super) fn decode_batch(
        &self,
        tokens: &[u32],
        caches: &mut [&mut [Option<LayerCache>]],
        batch_cache: &mut [Option<LayerCache>],
        positions: &[usize],
    ) -> Result<Vec<Tensor>> {
        anyhow::ensure!(
            self.supports_decode_batch(),
            "GLM5 decode batching unavailable"
        );
        anyhow::ensure!(!tokens.is_empty(), "GLM5 decode batch cannot be empty");
        anyhow::ensure!(
            tokens.len() == caches.len(),
            "GLM5 decode batch lane mismatch"
        );
        anyhow::ensure!(
            tokens.len() == positions.len(),
            "GLM5 decode batch position mismatch"
        );
        anyhow::ensure!(
            batch_cache.len() == self.blocks.len()
                && caches.iter().all(|cache| cache.len() == self.blocks.len()),
            "GLM5 decode batch cache layer mismatch"
        );
        anyhow::ensure!(
            tokens.iter().all(|&token| (token as usize) < self.vocab),
            "GLM5 decode batch token outside vocabulary"
        );
        anyhow::ensure!(
            positions.iter().all(|&pos| pos < self.context_length),
            "GLM5 decode batch position outside context"
        );

        // Complete preflight before any state writes. Execution errors require
        // a fresh prefill, matching ModelArchitecture's batch contract. Do not
        // retain a second full recurrent state just to roll back failed work.
        for (cache, &position) in caches.iter().zip(positions) {
            for (slot, block) in cache.iter().zip(&self.blocks) {
                self.check_decode_cache(slot, block, position)?;
            }
        }
        let result = self.decode_batch_inner(tokens, caches);
        if result.is_err() {
            for cache in caches {
                for slot in cache.iter_mut() {
                    *slot = None;
                }
            }
        }
        // No shared cache retains another request's state after this call.
        for slot in batch_cache {
            *slot = None;
        }
        result
    }

    fn decode_batch_inner(
        &self,
        tokens: &[u32],
        caches: &mut [&mut [Option<LayerCache>]],
    ) -> Result<Vec<Tensor>> {
        let embedded = self.embed_tokens(tokens)?;
        let hidden = (0..tokens.len())
            .map(|lane| {
                Ok(embedded
                    .narrow(0, lane, 1)?
                    .to_dtype(DType::F32)?
                    .reshape((1, 1, 1, self.hidden))?
                    .repeat((1, 1, GLM5_STREAMS, 1))?)
            })
            .collect::<Result<Vec<_>>>()?;

        self.decode_hidden_batch(hidden, caches)?
            .into_iter()
            .map(|output| match output {
                StageOutput::Logits(logits) => Ok(logits),
                StageOutput::Activations(_) => {
                    anyhow::bail!("GLM5 solo batch returned activations")
                }
            })
            .collect()
    }

    fn decode_hidden_batch(
        &self,
        mut hidden: Vec<Tensor>,
        caches: &mut [&mut [Option<LayerCache>]],
    ) -> Result<Vec<StageOutput>> {
        let lanes = hidden.len();

        for (layer, block) in self.blocks.iter().enumerate() {
            let mut inputs = Vec::with_capacity(lanes);
            let mut mixes = Vec::with_capacity(lanes);
            for (lane, hidden) in hidden.iter_mut().enumerate() {
                let (input, projected) = block.attn_hc.pre_update(hidden)?;
                let input = block.attn_norm.forward(&input)?;
                let output = match &block.attention {
                    GlmAttention::Kda(attention) => {
                        let mut state = match caches[lane][layer].take() {
                            Some(LayerCache::DeltaNet(state)) => Some(state),
                            None => None,
                            _ => anyhow::bail!("GLM5 batch KDA cache type changed"),
                        };
                        let output = attention
                            .forward(&input, &mut state)
                            .with_context(|| format!("GLM5 batch layer {layer} lane {lane} KDA"))?;
                        caches[lane][layer] = state.map(LayerCache::DeltaNet);
                        output
                    }
                    GlmAttention::Dsa(attention) => with_kv_cache(&mut caches[lane][layer], |kv| {
                        attention.forward(&input, kv, 1, false)
                    })
                    .with_context(|| format!("GLM5 batch layer {layer} lane {lane} DSA"))?,
                };
                let (post, combination) =
                    block.attn_hc.post_controls(hidden, projected.as_ref())?;
                *hidden = block.attn_hc.post(&output, hidden, &post, &combination)?;
                let (input, projected) = block.ffn_hc.pre_update(hidden)?;
                inputs.push(block.ffn_norm.forward(&input)?);
                mixes.push(projected);
            }
            let input = Tensor::cat(&inputs, 1)?;
            let output = block
                .ffn
                .forward_decode_rows(&input)
                .with_context(|| format!("GLM5 batch layer {layer} FFN"))?;
            for (lane, (hidden, mixes)) in hidden.iter_mut().zip(mixes).enumerate() {
                // Keep post controls short-lived, as in single-request decode.
                let (post, combination) = block.ffn_hc.post_controls(hidden, mixes.as_ref())?;
                *hidden =
                    block
                        .ffn_hc
                        .post(&output.narrow(1, lane, 1)?, hidden, &post, &combination)?;
            }
        }
        if !self.is_tail {
            return hidden
                .iter()
                .map(|x| Ok(StageOutput::Activations(x.reshape((1, self.stage_hidden))?)))
                .collect();
        }
        let norm = self
            .output_norm
            .as_ref()
            .context("GLM5 output norm is not loaded")?;
        let rows = hidden
            .iter()
            .map(|hidden| norm.forward(&hidden.sum(2)?.affine(1.0 / GLM5_STREAMS as f64, 0.0)?))
            .collect::<Result<Vec<_>>>()?;
        let logits = self
            .output
            .as_ref()
            .context("GLM5 output head is not loaded")?
            .forward_decode_rows(&Tensor::cat(&rows, 1)?)?
            .squeeze(0)?;
        let outputs = (0..lanes)
            .map(|lane| Ok(StageOutput::Logits(logits.get(lane)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(outputs)
    }

    fn check_decode_cache(
        &self,
        slot: &Option<LayerCache>,
        block: &GlmBlock,
        position: usize,
    ) -> Result<()> {
        let valid = match (slot, &block.attention) {
            (None, _) => position == 0,
            (Some(LayerCache::DeltaNet(state)), GlmAttention::Kda(attention)) => {
                position > 0
                    && state.speculative_state.is_none()
                    && state.conv_state.dims() == [3 * attention.inner, attention.kernel]
                    && state.recurrent_state.dims()
                        == [attention.heads, attention.head_dim, attention.head_dim]
                    && state.conv_state.dtype() == DType::F32
                    && state.recurrent_state.dtype() == DType::F32
                    && state.conv_state.device().same_device(&self.device)
                    && state.recurrent_state.device().same_device(&self.device)
            }
            (Some(LayerCache::Attention(kv)), GlmAttention::Dsa(attention)) => {
                let (k, v, length) = match kv {
                    KvCache::F16 { k, v } => (k, v, k.dim(2)?),
                    KvCache::PreallocatedF16(state) => (&state.k, &state.v, state.len),
                    KvCache::Quantized(_) => {
                        anyhow::bail!("GLM5 batch does not support quantized latent state")
                    }
                };
                let (kb, kh, kr, kw) = k.dims4()?;
                let (vb, vh, vr, vw) = v.dims4()?;
                length == position
                    && position > 0
                    && (kb, kh, vb, vh) == (1, 1, 1, 1)
                    && kr >= length
                    && vr >= length
                    && kw == attention.kv_rank
                    && vw == 2 * attention.index_dim
                    && k.dtype() == DType::F16
                    && v.dtype() == DType::F16
                    && k.device().same_device(&self.device)
                    && v.device().same_device(&self.device)
            }
            _ => false,
        };
        anyhow::ensure!(
            valid,
            "GLM5 decode batch cache type, shape, or position mismatch"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests;

#[cfg(all(test, target_os = "macos"))]
mod full_tests;
