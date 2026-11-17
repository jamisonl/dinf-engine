//! Group routed rows without changing routing or each token's sum order.

use super::*;

pub(super) fn should_group(enabled: bool, decode_rows: bool, rows: usize) -> bool {
    // Preserve the existing independent-request decode path: full requests
    // showed no clear decode gain. Group only multi-token prompt chunks.
    enabled && !decode_rows && rows > 1
}

impl V4Moe {
    pub(super) fn scalar_experts_forward(
        &self,
        x: &Tensor,
        routes: &[Vec<(usize, f32)>],
    ) -> Result<Tensor> {
        let (_, seq_len, hidden_dim) = x.dims3()?;
        let mut output = Tensor::zeros((1, seq_len, hidden_dim), DType::F32, x.device())?;
        for (token, choices) in routes.iter().enumerate() {
            let token_input = x.narrow(1, token, 1)?;
            let mut token_output = Tensor::zeros((1, 1, hidden_dim), DType::F32, x.device())?;
            for &(expert, weight) in choices {
                let expert_output =
                    Self::expert_forward(&self.experts[expert], &token_input, self.clamp)?;
                token_output = (token_output + expert_output * weight as f64)?;
            }
            output = output.slice_scatter(&token_output, 1, token)?;
        }
        Ok(output)
    }

    pub(super) fn grouped_experts_forward(
        &self,
        x: &Tensor,
        routes: &[Vec<(usize, f32)>],
    ) -> Result<Tensor> {
        let (_, seq_len, hidden_dim) = x.dims3()?;
        let mut indices = vec![Vec::new(); self.experts.len()];
        for (token, choices) in routes.iter().enumerate() {
            let token = u32::try_from(token).context("V4 expert row exceeds u32")?;
            for &(expert, _) in choices {
                indices[expert].push(token);
            }
        }
        if indices.iter().all(|rows| rows.len() <= 1) {
            return self.scalar_experts_forward(x, routes);
        }

        // Candle's row gather needs contiguous input (including on CPU).
        let flat = x.squeeze(0)?.contiguous()?;
        let mut outputs = Vec::with_capacity(self.experts.len());
        for (expert, rows) in self.experts.iter().zip(&indices) {
            let output = match rows.as_slice() {
                [] => None,
                &[token] => Some(Self::expert_forward(
                    expert,
                    &x.narrow(1, token as usize, 1)?,
                    self.clamp,
                )?),
                _ => {
                    let input = flat
                        .index_select(&Tensor::new(rows.as_slice(), x.device())?, 0)?
                        .unsqueeze(0)?;
                    // These products were single-token calls, even in prefill.
                    // Keep that reduction order, not the shared prefill kernel.
                    let gate = expert.gate.forward_decode_rows(&input)?;
                    let up = expert.up.forward_decode_rows(&input)?;
                    let activated = expert_activation(&gate, &up, self.clamp)?;
                    Some(expert.down.forward_decode_rows(&activated)?)
                }
            };
            outputs.push(output);
        }

        #[cfg(target_os = "macos")]
        if x.device().is_metal() {
            let top_k = routes.iter().map(Vec::len).max().unwrap_or(0);
            if top_k == 0 {
                return Ok(Tensor::zeros(
                    (1, seq_len, hidden_dim),
                    DType::F32,
                    x.device(),
                )?);
            }
            let zero = Tensor::zeros((1, 1, hidden_dim), DType::F32, x.device())?;
            let mut ordered = Vec::new();
            let mut weights = Vec::new();
            let mut next_row = vec![0; self.experts.len()];
            for choices in routes {
                for &(expert, weight) in choices {
                    ordered.push(
                        outputs[expert]
                            .as_ref()
                            .context("V4 grouped expert output is missing")?
                            .narrow(1, next_row[expert], 1)?,
                    );
                    next_row[expert] += 1;
                    weights.push(weight);
                }
                for _ in choices.len()..top_k {
                    ordered.push(zero.clone());
                    weights.push(0.);
                }
            }
            let ordered = Tensor::cat(&ordered, 1)?.reshape((seq_len, top_k, hidden_dim))?;
            let weights = Tensor::from_vec(weights, (seq_len, top_k), x.device())?;
            return Ok(
                super::super::indexed_moe_metal::fused_ordered_weighted_route_sum(
                    &ordered, &weights,
                )?
                .unsqueeze(0)?,
            );
        }

        let mut next_row = vec![0; self.experts.len()];
        let mut output = Tensor::zeros((1, seq_len, hidden_dim), DType::F32, x.device())?;
        for (token, choices) in routes.iter().enumerate() {
            let mut token_output = Tensor::zeros((1, 1, hidden_dim), DType::F32, x.device())?;
            // Do not sum by expert ID: floating-point addition is order-sensitive.
            // Duplicate choices also keep their separate, original contributions.
            for &(expert, weight) in choices {
                let expert_output = outputs[expert]
                    .as_ref()
                    .context("V4 grouped expert output is missing")?
                    .narrow(1, next_row[expert], 1)?;
                next_row[expert] += 1;
                token_output = (token_output + expert_output * weight as f64)?;
            }
            output = output.slice_scatter(&token_output, 1, token)?;
        }
        Ok(output)
    }
}

#[cfg(test)]
#[path = "routed_experts_tests.rs"]
mod tests;
