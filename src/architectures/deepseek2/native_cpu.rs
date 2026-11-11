//! DeepSeek CPU quantized policy, matched to the pinned ARM reference by saved-input tests.
//! Q8 uses the proved whole-block order; K formats keep request-wide M4/M1 order.
use crate::qmatmul::QMatMul;
use anyhow::{Context, Result};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Module, Tensor};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "native_cpu/native_q8.rs"]
mod native_q8;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "native_cpu/native_q8_dot.rs"]
mod native_q8_dot;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "native_cpu/native_swiglu_compat.rs"]
mod native_swiglu_compat;
pub(super) fn admitted_ffn(
    gate: &QMatMul,
    up: &QMatMul,
    down: &QMatMul,
    native_cache: bool,
) -> bool {
    if !native_cache || !supported_quant(gate) || !supported_quant(up) || !supported_quant(down) {
        return false;
    }
    let (QMatMul::QTensor(g), QMatMul::QTensor(u), QMatMul::QTensor(d)) = (gate, up, down) else {
        return false;
    };
    let gs = g.shape().dims();
    gs == u.shape().dims() && d.shape().dims() == [gs[1], gs[0]]
}
pub(super) fn swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (gate, up);
        anyhow::bail!("native DeepSeek SwiGLU is limited to macOS ARM CPU");
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        anyhow::ensure!(
            gate.device().is_cpu()
                && up.device().is_cpu()
                && gate.dtype() == DType::F32
                && up.dtype() == DType::F32
                && gate.dims() == up.dims(),
            "DeepSeek SwiGLU requires equal CPU F32 shapes"
        );
        let width = *gate.dims().last().context("DeepSeek SwiGLU rank")?;
        anyhow::ensure!(
            width > 0 && width.is_multiple_of(4),
            "DeepSeek SwiGLU row width"
        );
        let gate = gate.contiguous()?;
        let up = up.contiguous()?;
        let (gs, gl) = gate.storage_and_layout();
        let (us, ul) = up.storage_and_layout();
        let (candle_core::Storage::Cpu(gs), candle_core::Storage::Cpu(us)) = (&*gs, &*us) else {
            anyhow::bail!("DeepSeek SwiGLU storage");
        };
        let (g0, g1) = gl
            .contiguous_offsets()
            .context("DeepSeek SwiGLU gate span")?;
        let (u0, u1) = ul.contiguous_offsets().context("DeepSeek SwiGLU up span")?;
        let g = gs
            .as_slice::<f32>()?
            .get(g0..g1)
            .context("DeepSeek SwiGLU gate extent")?;
        let u = us
            .as_slice::<f32>()?
            .get(u0..u1)
            .context("DeepSeek SwiGLU up extent")?;
        anyhow::ensure!(
            g.iter().chain(u).all(|v| v.is_finite()),
            "DeepSeek SwiGLU finite inputs"
        );
        let out = native_swiglu_compat::native_swiglu(g, u);
        Tensor::from_vec(out, gate.shape(), gate.device()).map_err(Into::into)
    }
}

pub(super) fn supported(matmul: &QMatMul) -> bool {
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = matmul;
        false
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        matches!(matmul,QMatMul::QTensor(w) if w.device().is_cpu()
        && w.dtype()==candle_core::quantized::GgmlDType::Q8_0 && w.rank()==2
        && w.shape().dims()[0]>0 && w.shape().dims()[0].is_multiple_of(4)
        && w.shape().dims()[1]>0 && w.shape().dims()[1].is_multiple_of(32))
    }
}
pub(super) fn forward(matmul: &QMatMul, input: &Tensor) -> Result<Tensor> {
    match project(matmul, input)? {
        Some(y) => Ok(y),
        None => matmul.forward(input).map_err(Into::into),
    }
}
pub(super) fn project(matmul: &QMatMul, x: &Tensor) -> Result<Option<Tensor>> {
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (matmul, x);
        Ok(None)
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        if !supported(matmul) {
            return Ok(None);
        }
        let QMatMul::QTensor(weight) = matmul else {
            return Ok(None);
        };
        if !weight.device().is_cpu()
            || weight.dtype() != candle_core::quantized::GgmlDType::Q8_0
            || !x.device().is_cpu()
            || x.dtype() != DType::F32
            || !x.is_contiguous()
        {
            return Ok(None);
        }
        anyhow::ensure!(
            x.rank() >= 2 && weight.rank() == 2,
            "DeepSeek Q8 projection rank"
        );
        let width = *x.dims().last().context("DeepSeek Q8 input width")?;
        let dims = weight.shape().dims();
        let columns = dims[0];
        anyhow::ensure!(
            width > 0 && width.is_multiple_of(32) && dims[1] == width && columns > 0,
            "DeepSeek Q8 projection shape"
        );
        let rows = x.elem_count() / width;
        anyhow::ensure!(rows > 0, "DeepSeek Q8 input is empty");
        // Encode only the checked logical span. A contiguous view may have
        // a nonzero offset in its backing storage.
        let (storage, layout) = x.storage_and_layout();
        let candle_core::Storage::Cpu(storage) = &*storage else {
            anyhow::bail!("DeepSeek Q8 projection requires CPU storage");
        };
        let (start, end) = layout
            .contiguous_offsets()
            .context("DeepSeek Q8 input span")?;
        let values = storage
            .as_slice::<f32>()?
            .get(start..end)
            .context("DeepSeek Q8 storage span")?;
        let quantized = native_q8::quantize(values).map_err(anyhow::Error::msg)?;
        let weights = weight.data()?;
        let row_bytes = (width / 32)
            .checked_mul(34)
            .context("DeepSeek Q8 byte overflow")?;
        anyhow::ensure!(
            quantized.len()
                == rows
                    .checked_mul(row_bytes)
                    .context("DeepSeek Q8 activation overflow")?
                && weights.len()
                    == columns
                        .checked_mul(row_bytes)
                        .context("DeepSeek Q8 weight overflow")?,
            "DeepSeek Q8 encoded extent"
        );
        let total = rows
            .checked_mul(columns)
            .context("DeepSeek Q8 output overflow")?;
        let workers = candle_core::utils::get_num_threads().max(1).min(total);
        let chunk = total.div_ceil(workers);
        let slots: Vec<_> = (0..workers).map(|_| std::sync::Mutex::new(None)).collect();
        // Each worker owns complete output elements. No reduction crosses a
        // worker boundary, so changing worker count preserves the dot order.
        candle_core::cpu::kernels::par_for_each(workers, |worker| {
            let begin = worker * chunk;
            let end = (begin + chunk).min(total);
            let result = (|| -> Result<Vec<f32>> {
                if begin >= end {
                    return Ok(Vec::new());
                }
                let mut output = vec![0f32; end - begin];
                let mut at = begin;
                while at < end {
                    let row = at / columns;
                    let column = at % columns;
                    let count = (columns - column).min(end - at);
                    native_q8_dot::project_q8_0(
                        &weights[column * row_bytes..(column + count) * row_bytes],
                        &quantized[row * row_bytes..(row + 1) * row_bytes],
                        width,
                        &mut output[at - begin..at - begin + count],
                    )
                    .map_err(anyhow::Error::msg)?;
                    at += count;
                }
                Ok(output)
            })();
            *slots[worker].lock().expect("new DeepSeek Q8 worker lock") = Some(result);
        });
        let mut output = Vec::with_capacity(total);
        for slot in slots {
            let result = slot
                .into_inner()
                .map_err(|_| anyhow::anyhow!("DeepSeek Q8 worker lock poisoned"))?
                .context("DeepSeek Q8 worker did not finish")?;
            output.extend(result?);
        }
        anyhow::ensure!(output.len() == total, "DeepSeek Q8 output extent");
        let mut shape = x.dims().to_vec();
        *shape.last_mut().context("DeepSeek Q8 output rank")? = columns;
        Ok(Some(Tensor::from_vec(
            output,
            shape.as_slice(),
            x.device(),
        )?))
    }
}
pub(super) fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    anyhow::ensure!(
        x.device().is_cpu()
            && weight.device().is_cpu()
            && x.dtype() == DType::F32
            && weight.dtype() == DType::F32,
        "DeepSeek CPU RMSNorm requires F32 CPU input and weight"
    );
    let width = *x
        .dims()
        .last()
        .context("DeepSeek CPU RMSNorm requires a row dimension")?;
    anyhow::ensure!(
        width > 0 && weight.dims() == [width],
        "DeepSeek CPU RMSNorm weight must match the nonempty last input dimension"
    );
    anyhow::ensure!(
        eps.is_finite() && eps >= 0.0 && (eps as f32).is_finite(),
        "DeepSeek CPU RMSNorm epsilon must be finite and nonnegative in F32"
    );
    if x.elem_count() == 0 {
        return Tensor::zeros(x.shape(), DType::F32, x.device()).map_err(Into::into);
    }
    // Contiguous tensors share storage; only strided views need a copy.
    let x = x.contiguous()?;
    let weight = weight.contiguous()?;
    let (xs, xl) = x.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let (candle_core::Storage::Cpu(xs), candle_core::Storage::Cpu(ws)) = (&*xs, &*ws) else {
        anyhow::bail!("DeepSeek CPU RMSNorm storage must be on CPU");
    };
    let (x_start, x_end) = xl
        .contiguous_offsets()
        .context("DeepSeek CPU RMSNorm input layout")?;
    let (w_start, w_end) = wl
        .contiguous_offsets()
        .context("DeepSeek CPU RMSNorm weight layout")?;
    let values = xs
        .as_slice::<f32>()?
        .get(x_start..x_end)
        .context("DeepSeek CPU RMSNorm input span")?;
    let weights = ws
        .as_slice::<f32>()?
        .get(w_start..w_end)
        .context("DeepSeek CPU RMSNorm weight span")?;
    let mut output = Vec::with_capacity(values.len());
    for row in values.chunks_exact(width) {
        let sum = row
            .iter()
            .map(|&value| f64::from(value * value))
            .sum::<f64>();
        let mean = (sum / width as f64) as f32;
        let scale = 1.0_f32 / (mean + eps as f32).sqrt();
        output.extend(
            row.iter()
                .zip(weights)
                .map(|(&value, &weight)| (value * scale) * weight),
        );
    }
    Tensor::from_vec(output, x.shape(), x.device()).map_err(Into::into)
}
#[cfg(test)]
#[path = "native_cpu/tests.rs"]
mod tests;

pub(super) fn supported_quant(matmul: &QMatMul) -> bool {
    supported(matmul) || supported_k_shape(matmul)
}
fn supported_k_shape(matmul: &QMatMul) -> bool {
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = matmul;
        false
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        matches!(matmul, QMatMul::QTensor(w) if w.device().is_cpu() && w.rank()==2
        && matches!(w.dtype(), GgmlDType::Q4K|GgmlDType::Q5K|GgmlDType::Q6K)
        && w.shape().dims()[0]>0 && w.shape().dims()[0].is_multiple_of(8)
        && w.shape().dims()[1]>0 && w.shape().dims()[1].is_multiple_of(256))
    }
}
pub(super) fn norm_supported(matmul: &QMatMul, native_cache: bool) -> bool {
    supported(matmul) || (native_cache && supported_k_shape(matmul))
}
pub(super) fn supported_routed(matmul: &QMatMul) -> bool {
    if supported_quant(matmul) {
        return true;
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        false
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        matches!(matmul, QMatMul::QTensor(w) if w.device().is_cpu() && w.rank()==2
        && w.dtype()==GgmlDType::Q5_0 && w.shape().dims()[0]>0
        && w.shape().dims()[1]>0 && w.shape().dims()[1].is_multiple_of(64))
    }
}
pub(super) fn admitted_routed_ffn(
    gate: &QMatMul,
    up: &QMatMul,
    down: &QMatMul,
    native_cache: bool,
) -> bool {
    if !native_cache || !supported_quant(gate) || !supported_quant(up) || !supported_routed(down) {
        return false;
    }
    let (QMatMul::QTensor(g), QMatMul::QTensor(u), QMatMul::QTensor(d)) = (gate, up, down) else {
        return false;
    };
    let gs = g.shape().dims();
    gs == u.shape().dims() && d.shape().dims() == [gs[1], gs[0]]
}

/// Complete-request row order; never infer it from the submitted chunk.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    expect(dead_code, reason = "The native row-order path only runs on macOS ARM")
)]
pub(super) struct RowPolicy {
    pub start: usize,
    pub seq_len: usize,
    pub m4_prefix_end: usize,
}

pub(super) fn supported_k(matmul: &QMatMul) -> bool {
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = matmul;
        false
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        supported_quant(matmul)
            && matches!(matmul, QMatMul::QTensor(w) if matches!(w.dtype(), GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K))
    }
}

pub(super) fn project_rows(
    matmul: &QMatMul,
    input: &Tensor,
    policy: Option<RowPolicy>,
) -> Result<Option<Tensor>> {
    if policy.is_some()
        && let Some(y) = project_k(matmul, input, false, policy)?
    {
        return Ok(Some(y));
    }
    project(matmul, input)
}

pub(super) fn forward_rows(
    matmul: &QMatMul,
    input: &Tensor,
    policy: Option<RowPolicy>,
) -> Result<Tensor> {
    match project_rows(matmul, input, policy)? {
        Some(y) => Ok(y),
        None => matmul.forward(input).map_err(Into::into),
    }
}

pub(super) fn forward_routed(matmul: &QMatMul, input: &Tensor) -> Result<Tensor> {
    if let Some(y) = project_k(matmul, input, true, None)? {
        return Ok(y);
    }
    if let Some(y) = project_q5(matmul, input)? {
        return Ok(y);
    }
    forward(matmul, input)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "native_cpu/native_k_dot.rs"]
mod native_k_dot;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "native_cpu/native_q8k.rs"]
mod native_q8k;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "native_cpu/prepared_k_dot.rs"]
mod prepared_k_dot;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "native_cpu/native_k_m1.rs"]
mod native_k_m1;

fn project_k(
    matmul: &QMatMul,
    x: &Tensor,
    routed: bool,
    policy: Option<RowPolicy>,
) -> Result<Option<Tensor>> {
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (matmul, x, routed, policy);
        Ok(None)
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        if !supported_quant(matmul) {
            return Ok(None);
        }
        let QMatMul::QTensor(weight) = matmul else {
            return Ok(None);
        };
        if !weight.device().is_cpu()
            || !matches!(
                weight.dtype(),
                GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K
            )
            || !x.device().is_cpu()
            || x.dtype() != DType::F32
            || !x.is_contiguous()
        {
            return Ok(None);
        }
        anyhow::ensure!(
            x.rank() >= 2 && weight.rank() == 2,
            "DeepSeek K-quant projection rank"
        );
        let width = *x.dims().last().context("DeepSeek K-quant input width")?;
        let dims = weight.shape().dims();
        let columns = dims[0];
        anyhow::ensure!(
            width > 0 && width.is_multiple_of(256) && dims[1] == width && columns > 0,
            "DeepSeek K-quant projection shape"
        );
        let rows = x.elem_count() / width;
        anyhow::ensure!(rows > 0, "DeepSeek K-quant input is empty");
        if let Some(policy) = policy {
            anyhow::ensure!(
                policy.seq_len > 0 && rows.is_multiple_of(policy.seq_len),
                "DeepSeek request row extent"
            );
            anyhow::ensure!(
                policy.start.checked_add(policy.seq_len).is_some(),
                "DeepSeek request position overflow"
            );
            anyhow::ensure!(
                matches!(x.rank(), 2 | 3) && x.dims()[x.rank() - 2] == policy.seq_len,
                "DeepSeek request row layout"
            );
        }
        let uses_m1 = |row: usize| {
            routed || policy.is_some_and(|p| p.start + row % p.seq_len >= p.m4_prefix_end)
        };

        // Encode only the admitted logical F32 span, including a nonzero
        // tensor offset, with the native Q8_K row encoder.
        let (storage, layout) = x.storage_and_layout();
        let candle_core::Storage::Cpu(storage) = &*storage else {
            anyhow::bail!("DeepSeek K-quant projection requires CPU storage");
        };
        let (start, end) = layout
            .contiguous_offsets()
            .context("DeepSeek K-quant input span")?;
        let values = storage
            .as_slice::<f32>()?
            .get(start..end)
            .context("DeepSeek K-quant storage span")?;
        let quantized = native_q8k::quantize(values).map_err(anyhow::Error::msg)?;
        let kind = match weight.dtype() {
            GgmlDType::Q4K => native_k_dot::Kind::Q4,
            GgmlDType::Q5K => native_k_dot::Kind::Q5,
            GgmlDType::Q6K => native_k_dot::Kind::Q6,
            _ => unreachable!("checked K-quant dtype"),
        };
        let weights = weight.data()?;
        let row_bytes = (width / 256)
            .checked_mul(kind.block_bytes())
            .context("DeepSeek K-quant byte overflow")?;
        let activation_bytes = (width / 256)
            .checked_mul(292)
            .context("DeepSeek Q8K byte overflow")?;
        anyhow::ensure!(
            quantized.len()
                == rows
                    .checked_mul(activation_bytes)
                    .context("DeepSeek K-quant activation overflow")?
                && weights.len()
                    == columns
                        .checked_mul(row_bytes)
                        .context("DeepSeek K-quant weight overflow")?,
            "DeepSeek K-quant encoded extent"
        );
        // Preparation changes integer layout only. Small calls use the same
        // canonical M4 order directly; large calls decode each weight once.
        let prepared_bytes = (weights.len() / kind.block_bytes())
            .checked_mul(std::mem::size_of::<prepared_k_dot::Block>());
        let prepared = if !routed
            && (0..rows).any(|row| !uses_m1(row))
            && rows >= 8
            && prepared_bytes.is_some_and(|bytes| bytes <= 256 * 1024 * 1024)
        {
            let weights =
                prepared_k_dot::Prepared::new(kind, &weights, width).map_err(anyhow::Error::msg)?;
            anyhow::ensure!(
                weights.columns() == columns && weights.bytes() <= 256 * 1024 * 1024,
                "DeepSeek prepared K-quant extent"
            );
            let inputs = quantized
                .chunks_exact(activation_bytes)
                .map(|row| prepared_k_dot::Input::new(row, width).map_err(anyhow::Error::msg))
                .collect::<Result<Vec<_>>>()?;
            Some((weights, inputs))
        } else {
            None
        };
        let total = rows
            .checked_mul(columns)
            .context("DeepSeek K-quant output overflow")?;
        let workers = candle_core::utils::get_num_threads().max(1).min(total);
        let chunk = total.div_ceil(workers);
        let slots: Vec<_> = (0..workers).map(|_| std::sync::Mutex::new(None)).collect();
        // Each worker owns complete output elements. No reduction crosses a
        // worker boundary, so changing worker count preserves the dot order.
        candle_core::cpu::kernels::par_for_each(workers, |worker| {
            let begin = worker * chunk;
            let end = (begin + chunk).min(total);
            let result = (|| -> Result<Vec<f32>> {
                if begin >= end {
                    return Ok(Vec::new());
                }
                let mut output = vec![0f32; end - begin];
                let mut at = begin;
                while at < end {
                    let row = at / columns;
                    let column = at % columns;
                    let count = (columns - column).min(end - at);
                    let dst = &mut output[at - begin..at - begin + count];
                    if uses_m1(row) {
                        // Routed experts always use GEMV. Bounded ordinary
                        // projections also use GEMV for prompt tails and decode.
                        let routed_kind = match kind {
                            native_k_dot::Kind::Q4 => native_k_m1::Kind::Q4,
                            native_k_dot::Kind::Q5 => native_k_m1::Kind::Q5,
                            native_k_dot::Kind::Q6 => native_k_m1::Kind::Q6,
                        };
                        native_k_m1::project(
                            routed_kind,
                            &weights[column * row_bytes..(column + count) * row_bytes],
                            &quantized[row * activation_bytes..(row + 1) * activation_bytes],
                            width,
                            dst,
                        )
                        .map_err(anyhow::Error::msg)?;
                    } else if let Some((weights, inputs)) = &prepared {
                        weights
                            .project(&inputs[row], column, dst)
                            .map_err(anyhow::Error::msg)?;
                    } else {
                        native_k_dot::project(
                            kind,
                            &weights[column * row_bytes..(column + count) * row_bytes],
                            &quantized[row * activation_bytes..(row + 1) * activation_bytes],
                            width,
                            dst,
                        )
                        .map_err(anyhow::Error::msg)?;
                    }
                    at += count;
                }
                Ok(output)
            })();
            *slots[worker]
                .lock()
                .expect("new DeepSeek K-quant worker lock") = Some(result);
        });
        let mut output = Vec::with_capacity(total);
        for slot in slots {
            let result = slot
                .into_inner()
                .map_err(|_| anyhow::anyhow!("DeepSeek K-quant worker lock poisoned"))?
                .context("DeepSeek K-quant worker did not finish")?;
            output.extend(result?);
        }
        anyhow::ensure!(output.len() == total, "DeepSeek K-quant output extent");
        let mut shape = x.dims().to_vec();
        *shape.last_mut().context("DeepSeek K-quant output rank")? = columns;
        Ok(Some(Tensor::from_vec(
            output,
            shape.as_slice(),
            x.device(),
        )?))
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "native_cpu/native_q5_dot.rs"]
mod native_q5_dot;
fn project_q5(matmul: &QMatMul, x: &Tensor) -> Result<Option<Tensor>> {
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (matmul, x);
        Ok(None)
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let QMatMul::QTensor(weight) = matmul else {
            return Ok(None);
        };
        if !supported_routed(matmul)
            || weight.dtype() != GgmlDType::Q5_0
            || !x.device().is_cpu()
            || x.dtype() != DType::F32
            || !x.is_contiguous()
        {
            return Ok(None);
        }
        anyhow::ensure!(x.rank() >= 2, "DeepSeek Q5 input rank");
        let width = *x.dims().last().context("DeepSeek Q5 width")?;
        let columns = weight.shape().dims()[0];
        anyhow::ensure!(
            width == weight.shape().dims()[1] && width > 0 && width.is_multiple_of(64),
            "DeepSeek Q5 shape"
        );
        let rows = x.elem_count() / width;
        anyhow::ensure!(rows > 0, "DeepSeek Q5 empty input");
        let (storage, layout) = x.storage_and_layout();
        let candle_core::Storage::Cpu(storage) = &*storage else {
            anyhow::bail!("DeepSeek Q5 CPU storage");
        };
        let (start, end) = layout.contiguous_offsets().context("DeepSeek Q5 span")?;
        let values = storage
            .as_slice::<f32>()?
            .get(start..end)
            .context("DeepSeek Q5 input extent")?;
        let input = native_q8::quantize(values).map_err(anyhow::Error::msg)?;
        let weights = weight.data()?;
        let wb = (width / 32)
            .checked_mul(22)
            .context("DeepSeek Q5 weight bytes")?;
        let ab = (width / 32)
            .checked_mul(34)
            .context("DeepSeek Q5 activation bytes")?;
        anyhow::ensure!(
            weights.len() == columns.checked_mul(wb).context("DeepSeek Q5 weights")?
                && input.len() == rows.checked_mul(ab).context("DeepSeek Q5 inputs")?,
            "DeepSeek Q5 encoded extent"
        );
        let total = rows
            .checked_mul(columns)
            .context("DeepSeek Q5 output extent")?;
        let workers = candle_core::utils::get_num_threads().max(1).min(total);
        let chunk = total.div_ceil(workers);
        let slots: Vec<_> = (0..workers).map(|_| std::sync::Mutex::new(None)).collect();
        candle_core::cpu::kernels::par_for_each(workers, |worker| {
            let begin = worker * chunk;
            let end = (begin + chunk).min(total);
            let result = (|| -> Result<Vec<f32>> {
                if begin >= end {
                    return Ok(Vec::new());
                }
                let mut out = vec![0f32; end - begin];
                let mut at = begin;
                while at < end {
                    let row = at / columns;
                    let col = at % columns;
                    let count = (columns - col).min(end - at);
                    native_q5_dot::project_q5_0(
                        &weights[col * wb..(col + count) * wb],
                        &input[row * ab..(row + 1) * ab],
                        width,
                        &mut out[at - begin..at - begin + count],
                    )
                    .map_err(anyhow::Error::msg)?;
                    at += count;
                }
                Ok(out)
            })();
            *slots[worker].lock().expect("new DeepSeek Q5 worker lock") = Some(result);
        });
        let mut out = Vec::with_capacity(total);
        for slot in slots {
            out.extend(
                slot.into_inner()
                    .map_err(|_| anyhow::anyhow!("DeepSeek Q5 poisoned lock"))?
                    .context("DeepSeek Q5 missing worker")??,
            );
        }
        anyhow::ensure!(out.len() == total, "DeepSeek Q5 output extent");
        let mut shape = x.dims().to_vec();
        *shape.last_mut().context("DeepSeek Q5 rank")? = columns;
        Ok(Some(Tensor::from_vec(out, shape.as_slice(), x.device())?))
    }
}
