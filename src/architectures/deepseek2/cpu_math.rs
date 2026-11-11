use crate::qmatmul::QMatMul;
use anyhow::{Context, Result};
use candle_core::{DType, Module, Tensor};

/// Match native F16-cache query storage for CPU quantized MLA only.
/// Native flash attention casts the unscaled Q row to F16, then applies the
/// attention scale after QK. Keep FULL weights and other cache/device paths.
pub(super) fn native_quant_query(
    projection: &QMatMul,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<Tensor> {
    let supported_projection = matches!(projection, QMatMul::QTensor(weight)
        if !matches!(weight.dtype(), candle_core::quantized::GgmlDType::F32
            | candle_core::quantized::GgmlDType::F16
            | candle_core::quantized::GgmlDType::BF16));
    if supported_projection
        && q.device().is_cpu()
        && q.device().same_device(k.device())
        && q.device().same_device(v.device())
        && q.dtype() == DType::F32
        && k.dtype() == DType::F16
        && v.dtype() == DType::F16
        && q.rank() == 4
        && k.rank() == 4
        && v.rank() == 4
        && q.dim(3)? == 192
        && k.dim(3)? == 192
        && v.dim(3)? == 128
    {
        return Ok(q.to_dtype(DType::F16)?.to_dtype(DType::F32)?);
    }
    Ok(q.clone())
}

/// Keep DeepSeek MLA CPU reductions fixed to retained-history positions.
/// The selected native cache has 192-wide Q/K heads and 128-wide V heads.
/// Requires a causal, non-windowed history; callers keep windowed paths separate.
/// Full and chunked prefill must use the same floating-point operations: tiny
/// attention changes can alter Q8 activation codes and then routed experts.
/// Fixed query tiles keep matrix shapes stable while fixing the key extent.
/// Missing queries/keys in a partial tile are zero-filled, and the causal
/// selection removes every future score, including overflowed values.
pub(super) fn attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Option<Tensor>> {
    if !q.device().is_cpu()
        || !q.device().same_device(k.device())
        || !q.device().same_device(v.device())
        || q.dtype() != DType::F32
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
        || [q, k, v].iter().any(|x| x.rank() != 4)
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Ok(None);
    }
    let (batch, heads, queries, dim) = q.dims4()?;
    let (kbatch, kv_heads, keys, kdim) = k.dims4()?;
    let (vbatch, vheads, vkeys, value_dim) = v.dims4()?;
    if batch == 0
        || heads == 0
        || queries == 0
        || kv_heads == 0
        || keys < queries
        || batch != kbatch
        || !heads.is_multiple_of(kv_heads)
        || dim != 192
        || kdim != dim
        || value_dim != 128
        || vbatch != batch
        || vheads != kv_heads
        || vkeys != keys
    {
        return Ok(None);
    }
    let repeats = heads / kv_heads;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    const TILE: usize = 8;
    let tile = TILE;
    let past = keys - queries;
    let mut outputs = Vec::new();
    let mut row = 0;
    while row < queries {
        let position = past + row;
        let tile_start = position / tile * tile;
        let end = tile_start
            .checked_add(tile)
            .context("CPU attention tile overflow")?;
        let offset = position - tile_start;
        let take = (queries - row).min(tile - offset);
        let mut parts = Vec::new();
        if offset > 0 {
            parts.push(Tensor::zeros(
                (batch, heads, offset, dim),
                DType::F32,
                q.device(),
            )?);
        }
        parts.push(q.narrow(2, row, take)?);
        if offset + take < tile {
            parts.push(Tensor::zeros(
                (batch, heads, tile - offset - take, dim),
                DType::F32,
                q.device(),
            )?);
        }
        let grouped_rows = repeats
            .checked_mul(tile)
            .context("CPU attention row overflow")?;
        let query =
            Tensor::cat(&parts, 2)?
                .contiguous()?
                .reshape((batch, kv_heads, grouped_rows, dim))?;
        let prefix = |x: &Tensor| -> candle_core::Result<Tensor> {
            let width = x.dim(3)?;
            let known = x.narrow(2, 0, keys.min(end))?;
            if keys < end {
                Tensor::cat(
                    &[
                        known,
                        Tensor::zeros(
                            (batch, kv_heads, end - keys, width),
                            DType::F32,
                            q.device(),
                        )?,
                    ],
                    2,
                )?
                .contiguous()
            } else {
                known.contiguous()
            }
        };
        let keys = prefix(&k)?;
        let values = prefix(&v)?;
        let mask_size = grouped_rows
            .checked_mul(end)
            .context("CPU attention mask overflow")?;
        let mut mask = vec![0u8; mask_size];
        for (i, r) in mask.chunks_exact_mut(end).enumerate() {
            r[..tile_start + i % tile + 1].fill(1);
        }
        let mask = Tensor::from_vec(mask, (1, 1, grouped_rows, end), q.device())?;
        let scores = query.matmul(&keys.transpose(2, 3)?)? * scale;
        let scores = scores?;
        // Select replaces masked infinities/NaNs; adding -inf cannot do that.
        let negative = Tensor::full(f32::NEG_INFINITY, scores.shape(), q.device())?;
        let scores = mask
            .broadcast_as(scores.shape())?
            .contiguous()?
            .where_cond(&scores, &negative)?;
        let probabilities = candle_nn::ops::softmax_last_dim(&scores)?;
        let output = probabilities
            .matmul(&values)?
            .reshape((batch, heads, tile, value_dim))?;
        outputs.push(output.narrow(2, offset, take)?);
        row += take;
    }
    Ok(Some(Tensor::cat(&outputs, 2)?))
}

/// Use the existing fixed-eight-row reduction for CPU F32 weights only.
/// Quantized matrices keep their current per-row kernels; other devices keep
/// the original direct call. Tensor clones share weight storage.
pub(super) fn project(weights: &QMatMul, input: &Tensor) -> Result<Tensor> {
    if let Some(output) = super::native_cpu::project(weights, input)? {
        return Ok(output);
    }
    if input.device().is_cpu() {
        super::super::cpu_math::project(weights, input)
    } else {
        weights.forward(input).map_err(Into::into)
    }
}

pub(super) fn router(input: &Tensor, weights: &Tensor) -> Result<Tensor> {
    if input.device().is_cpu()
        && weights.device().is_cpu()
        && input.dtype() == DType::F32
        && weights.dtype() == DType::F32
    {
        project(&QMatMul::Tensor(weights.clone()), input)
    } else {
        input.matmul(&weights.t()?).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests;

pub(super) fn project_rows(
    weights: &QMatMul,
    input: &Tensor,
    policy: Option<super::native_cpu::RowPolicy>,
) -> Result<Tensor> {
    if let Some(output) = super::native_cpu::project_rows(weights, input, policy)? {
        return Ok(output);
    }
    project(weights, input)
}
