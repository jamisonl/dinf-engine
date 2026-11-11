//! Keep full-precision expert banks in BF16 with a cache for one layer call.
//!
//! Candle's ordinary QMatMul constructor expands every BF16 expert to F32.
//! Keeping all of those copies would double a large MoE's resident weights.
//! CPU keeps selected experts packed and rounds dot inputs to BF16. Metal
//! makes lossless F32 copies of selected experts. Eligible Metal
//! prefill batches independent one-row projections with shared weight storage.
//! Both retain scalar route order; Metal retains F32 inputs and accumulation.

use super::{BF16_GROUP_WORKSPACE_BYTES, MoeExpert, split_expert_tensor};
use crate::qmatmul::QMatMul;
use anyhow::{Context, Result, ensure};
use candle_core::quantized::{GgmlDType, QTensor, ggml_file};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use std::sync::Arc;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "native_bf16_dot.rs"]
mod native_bf16_dot;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "native_swiglu_compat.rs"]
mod native_swiglu_compat;

/// Native ARM BF16 projection: round each input to BF16, multiply in F32,
/// sum in F64, then round once. Each worker owns complete outputs.
pub(super) fn native_projection(
    weights: &QMatMul,
    input: &Tensor,
    native_cache: bool,
) -> Result<Option<Tensor>> {
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (weights, input, native_cache);
        Ok(None)
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let QMatMul::QTensor(weight) = weights else {
            return Ok(None);
        };
        if !native_cache
            || !weight.device().is_cpu()
            || weight.dtype() != GgmlDType::BF16
            || !input.device().is_cpu()
            || input.dtype() != DType::F32
        {
            return Ok(None);
        }
        ensure!(
            weight.rank() == 2 && input.rank() >= 2,
            "BF16 projection rank"
        );
        let dims = weight.shape().dims();
        let columns = dims[0];
        let width = dims[1];
        ensure!(
            width > 0 && columns > 0 && input.dims().last() == Some(&width),
            "BF16 projection shape"
        );
        let rows = input.elem_count() / width;
        let mut shape = input.dims().to_vec();
        *shape.last_mut().context("BF16 output rank")? = columns;
        if rows == 0 {
            return Ok(Some(Tensor::zeros(shape, DType::F32, input.device())?));
        }
        let logical = input.contiguous()?;
        let (storage, layout) = logical.storage_and_layout();
        let candle_core::Storage::Cpu(storage) = &*storage else {
            anyhow::bail!("BF16 CPU storage");
        };
        let (start, end) = layout.contiguous_offsets().context("BF16 logical span")?;
        let values = storage
            .as_slice::<f32>()?
            .get(start..end)
            .context("BF16 source span")?;
        let encoded = native_bf16_dot::encode(values);
        let bytes = weight.data()?;
        let row_bytes = width.checked_mul(2).context("BF16 width overflow")?;
        ensure!(
            bytes.len()
                == columns
                    .checked_mul(row_bytes)
                    .context("BF16 weight extent")?
                && encoded.len() == rows.checked_mul(row_bytes).context("BF16 input extent")?,
            "BF16 encoded extent"
        );
        let total = rows.checked_mul(columns).context("BF16 output extent")?;
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
                let mut output = vec![0.0; end - begin];
                let mut at = begin;
                while at < end {
                    let row = at / columns;
                    let column = at % columns;
                    let count = (columns - column).min(end - at);
                    native_bf16_dot::project(
                        &bytes[column * row_bytes..(column + count) * row_bytes],
                        &encoded[row * row_bytes..(row + 1) * row_bytes],
                        width,
                        &mut output[at - begin..at - begin + count],
                    )
                    .map_err(anyhow::Error::msg)?;
                    at += count;
                }
                Ok(output)
            })();
            *slots[worker].lock().expect("new BF16 worker lock") = Some(result);
        });
        let mut output = Vec::with_capacity(total);
        for slot in slots {
            output.extend(
                slot.into_inner()
                    .map_err(|_| anyhow::anyhow!("BF16 worker lock poisoned"))?
                    .context("BF16 worker incomplete")??,
            );
        }
        ensure!(output.len() == total, "BF16 output size");
        Ok(Some(Tensor::from_vec(output, shape, input.device())?))
    }
}

fn project_or_original(weights: &QMatMul, input: &Tensor, native_cache: bool) -> Result<Tensor> {
    match native_projection(weights, input, native_cache)? {
        Some(output) => Ok(output),
        None => weights.forward(input).map_err(Into::into),
    }
}

// Only stored CPU BF16 experts use the proved native split-SwiGLU rule.
// Other expert types and devices keep the original forward call.
fn native_bf16_swiglu(expert: &MoeExpert, gate: &Tensor, up: &Tensor) -> Result<Option<Tensor>> {
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (expert, gate, up);
        Ok(None)
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let supported = [&expert.gate_proj, &expert.up_proj, &expert.down_proj]
            .iter()
            .all(|projection| {
                matches!(projection, QMatMul::QTensor(w)
                if w.device().is_cpu() && w.dtype() == GgmlDType::BF16)
            });
        if !supported
            || !gate.device().is_cpu()
            || !up.device().is_cpu()
            || gate.dtype() != DType::F32
            || up.dtype() != DType::F32
            || gate.dims() != up.dims()
            || gate.rank() == 0
            || !gate.dims()[gate.rank() - 1].is_multiple_of(4)
        {
            return Ok(None);
        }
        // flatten_all/to_vec1 copy the logical view, including its offset.
        // Each row contains whole four-lane groups; there is no scalar tail.
        let g = gate.flatten_all()?.to_vec1::<f32>()?;
        let u = up.flatten_all()?.to_vec1::<f32>()?;
        let output = native_swiglu_compat::native_swiglu(&g, &u);
        Ok(Some(Tensor::from_vec(output, gate.shape(), gate.device())?))
    }
}

fn scalar_expert_forward(expert: &MoeExpert, input: &Tensor, native_cache: bool) -> Result<Tensor> {
    if native_cache
        && cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && input.device().is_cpu() && input.dtype() == DType::F32
        && [&expert.gate_proj, &expert.up_proj, &expert.down_proj].iter().all(|projection|
            matches!(projection, QMatMul::QTensor(w) if w.device().is_cpu() && w.dtype() == GgmlDType::BF16
                && w.rank() == 2 && w.shape().dims()[0] > 0))
        && matches!(&expert.gate_proj, QMatMul::QTensor(w) if w.shape().dims()[0].is_multiple_of(4))
    {
        let gate = project_or_original(&expert.gate_proj,input,native_cache)?;
        let up = project_or_original(&expert.up_proj,input,native_cache)?;
        if let Some(activated) = native_bf16_swiglu(expert, &gate, &up)? {
            return project_or_original(&expert.down_proj,&activated,native_cache);
        }
    }
    expert.forward(input)
}

enum ProjectionBank {
    Bf16 {
        packed: Arc<QTensor>,
        out_dim: usize,
        in_dim: usize,
        bytes_per_expert: usize,
    },
    Split(Vec<QMatMul>),
}

impl ProjectionBank {
    fn new(
        packed: Arc<QTensor>,
        experts: usize,
        out_dim: usize,
        in_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        ensure!(
            experts > 0 && out_dim > 0 && in_dim > 0,
            "empty BF16 expert geometry"
        );
        ensure!(
            packed.shape().dims() == [experts, out_dim, in_dim],
            "BF16 expert bank shape differs from metadata"
        );
        if packed.dtype() != GgmlDType::BF16 {
            return Ok(Self::Split(split_expert_tensor(
                &packed, experts, out_dim, in_dim, device,
            )?));
        }
        ensure!(
            packed.device().is_cpu(),
            "BF16 expert source must stay on CPU"
        );
        let bytes_per_expert = out_dim
            .checked_mul(in_dim)
            .and_then(|n| n.checked_mul(2))
            .context("BF16 expert byte count overflow")?;
        let expected = bytes_per_expert
            .checked_mul(experts)
            .context("BF16 bank byte count overflow")?;
        ensure!(
            packed.data()?.len() == expected,
            "wrong BF16 bank byte count"
        );
        Ok(Self::Bf16 {
            packed,
            out_dim,
            in_dim,
            bytes_per_expert,
        })
    }

    fn materialize(&self, expert: usize, device: &Device, packed_metal: bool) -> Result<QMatMul> {
        match self {
            Self::Split(experts) => experts
                .get(expert)
                .cloned()
                .context("expert ID out of range"),
            Self::Bf16 {
                packed,
                out_dim,
                in_dim,
                bytes_per_expert,
            } => {
                let start = expert
                    .checked_mul(*bytes_per_expert)
                    .context("BF16 expert offset overflow")?;
                let end = start
                    .checked_add(*bytes_per_expert)
                    .context("BF16 expert extent overflow")?;
                let data = packed.data()?;
                let bytes = data.get(start..end).context("expert ID out of range")?;
                // This kernel reads complete 32-element K tiles. Keep this
                // guard here too, even if the caller admits the whole layer.
                if packed_metal && device.is_metal() && in_dim.is_multiple_of(32) {
                    let weights = ggml_file::qtensor_from_ggml(
                        GgmlDType::BF16,
                        bytes,
                        vec![*out_dim, *in_dim],
                        device,
                    )?;
                    // Bypass QMatMul::from_arc, which expands BF16 to F32.
                    // The caller selects the packed mm kernel with two rows.
                    return Ok(QMatMul::QTensor(Arc::new(weights)));
                }
                if device.is_cpu() {
                    let weights = ggml_file::qtensor_from_ggml(
                        GgmlDType::BF16,
                        bytes,
                        vec![*out_dim, *in_dim],
                        device,
                    )?;
                    // Retain BF16 storage and its BF16 activation conversion.
                    return Ok(QMatMul::QTensor(Arc::new(weights)));
                }
                // BF16 -> F32 is exact. Do not pass through F16 or quantize
                // activations to fit a packed dot-product implementation.
                let source = ggml_file::qtensor_from_ggml(
                    GgmlDType::BF16,
                    bytes,
                    vec![*out_dim, *in_dim],
                    &Device::Cpu,
                )?;
                let weights = source.dequantize(&Device::Cpu)?.to_device(device)?;
                Ok(QMatMul::Tensor(weights))
            }
        }
    }
}

pub(super) struct Bf16Experts {
    gate: ProjectionBank,
    up: ProjectionBank,
    down: ProjectionBank,
    num_experts: usize,
    hidden_dim: usize,
    expert_dim: usize,
    grouped_metal: bool,
    native_cache: bool,
}

impl Bf16Experts {
    pub(super) fn new(
        gate: Arc<QTensor>,
        up: Arc<QTensor>,
        down: Arc<QTensor>,
        num_experts: usize,
        hidden_dim: usize,
        expert_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        // Keep grouped prefill within the existing pure BF16 Metal geometry.
        // Mixed or unaligned banks retain their scalar fallback.
        let grouped_metal = device.is_metal()
            && hidden_dim.is_multiple_of(32)
            && expert_dim.is_multiple_of(32)
            && [&gate, &up, &down]
                .iter()
                .all(|bank| bank.dtype() == GgmlDType::BF16);
        Ok(Self {
            gate: ProjectionBank::new(gate, num_experts, expert_dim, hidden_dim, device)?,
            up: ProjectionBank::new(up, num_experts, expert_dim, hidden_dim, device)?,
            down: ProjectionBank::new(down, num_experts, hidden_dim, expert_dim, device)?,
            num_experts,
            hidden_dim,
            expert_dim,
            grouped_metal,
            native_cache: true,
        })
    }

    pub(super) fn with_native_cache(mut self, native_cache: bool) -> Self {
        self.native_cache = native_cache;
        self
    }

    pub(super) fn len(&self) -> usize {
        self.num_experts
    }

    pub(super) fn forward(&self, input: &Tensor, routes: &[Vec<(usize, f32)>]) -> Result<Tensor> {
        self.forward_with_group_limit(input, routes, 128)
    }

    fn cached_expert<'a>(
        &self,
        cache: &'a mut [Option<MoeExpert>],
        expert: usize,
        device: &Device,
    ) -> Result<&'a MoeExpert> {
        let entry = cache.get_mut(expert).context("expert ID out of range")?;
        if entry.is_none() {
            *entry = Some(MoeExpert {
                native_quant: false,
                gate_proj: self.gate.materialize(expert, device, false)?,
                up_proj: self.up.materialize(expert, device, false)?,
                down_proj: self.down.materialize(expert, device, false)?,
            });
        }
        Ok(entry.as_ref().expect("initialized expert"))
    }

    // Include all queued group inputs, gate/up/SiLU/product tensors with spare
    // intermediate copies, group outputs, cat/reorder copies and each
    // route's separate weighted output and running sum. Reserve CPU/GPU index
    // and weight buffers too. Input, returned output and the layer's weight
    // cache have separate storage; a fence ends each chunk's temporary work.
    // Double the bound for Metal's next-power-of-two buffer allocation.
    fn grouped_workspace_bytes(&self, tokens: usize, top_k: usize) -> Option<usize> {
        let rows = tokens.checked_mul(top_k)?;
        u32::try_from(rows).ok()?;
        let per_route = self
            .hidden_dim
            .checked_mul(10)?
            .checked_add(self.expert_dim.checked_mul(8)?)?
            .checked_mul(4)?
            .checked_add(64)?;
        rows.checked_mul(per_route)?
            .checked_add(tokens.checked_mul(self.hidden_dim)?.checked_mul(16)?)?
            .checked_add(self.num_experts.checked_mul(32)?)?
            .checked_mul(2)
    }

    fn grouped_chunk_tokens(&self, top_k: usize, limit: usize) -> Option<usize> {
        (1..=limit.min(128)).rev().find(|&tokens| {
            self.grouped_workspace_bytes(tokens, top_k)
                .is_some_and(|bytes| bytes <= BF16_GROUP_WORKSPACE_BYTES)
        })
    }

    fn scalar_forward(
        &self,
        input: &Tensor,
        routes: &[Vec<(usize, f32)>],
        cache: &mut [Option<MoeExpert>],
    ) -> Result<Tensor> {
        let hidden = self.hidden_dim;
        let mut output = Vec::with_capacity(routes.len());
        for (token, selected) in routes.iter().enumerate() {
            let row = input.i(token)?.unsqueeze(0)?;
            let mut sum = Tensor::zeros((1, hidden), DType::F32, input.device())?;
            for &(expert, weight) in selected {
                let selected = self.cached_expert(cache, expert, input.device())?;
                let expert_output = scalar_expert_forward(selected, &row, self.native_cache)?;
                sum = (sum + (expert_output.to_dtype(DType::F32)? * f64::from(weight))?)?;
            }
            output.push(sum);
        }
        if output.is_empty() {
            Ok(Tensor::zeros((0, hidden), DType::F32, input.device())?)
        } else {
            Ok(Tensor::cat(&output, 0)?)
        }
    }

    fn grouped_chunk(
        &self,
        input: &Tensor,
        routes: &[Vec<(usize, f32)>],
        top_k: usize,
        cache: &mut [Option<MoeExpert>],
    ) -> Result<Tensor> {
        let tokens = routes.len();
        let rows = tokens
            .checked_mul(top_k)
            .context("BF16 route count overflow")?;
        ensure!(u32::try_from(rows).is_ok(), "BF16 route indices exceed u32");
        ensure!(
            self.grouped_workspace_bytes(tokens, top_k)
                .is_some_and(|bytes| bytes <= BF16_GROUP_WORKSPACE_BYTES),
            "BF16 grouped workspace exceeds limit"
        );
        // A duplicate expert in one token still has a distinct route slot.
        let mut groups = vec![Vec::new(); self.num_experts];
        for (token, selected) in routes.iter().enumerate() {
            let index = u32::try_from(token).context("BF16 token index exceeds u32")?;
            for (slot, &(expert, _)) in selected.iter().enumerate() {
                groups[expert].push((index, token * top_k + slot));
            }
        }
        let mut inverse = vec![0u32; rows];
        let mut grouped_outputs = Vec::new();
        let mut offset = 0usize;
        for (expert, selected) in groups.iter().enumerate() {
            if selected.is_empty() {
                continue;
            }
            let indices = selected.iter().map(|&(token, _)| token).collect::<Vec<_>>();
            let indices = Tensor::from_vec(indices, selected.len(), input.device())?;
            // Keep each token as a separate one-row matmul. QMatMul::Tensor
            // broadcasts the F32 weights as a zero-stride view, so Metal uses
            // batched MV without making one weight copy per token. Do not use
            // broadcast_matmul here: it makes the broadcast weights contiguous.
            let selected_input = input.index_select(&indices, 0)?.unsqueeze(1)?;
            let output = self
                .cached_expert(cache, expert, input.device())?
                .forward(&selected_input)?
                .squeeze(1)?;
            for (row, &(_, slot)) in selected.iter().enumerate() {
                inverse[slot] =
                    u32::try_from(offset + row).context("BF16 reordered index exceeds u32")?;
            }
            offset += selected.len();
            grouped_outputs.push(output);
        }
        ensure!(offset == rows, "BF16 grouped route count differs");
        let inverse = Tensor::from_vec(inverse, rows, input.device())?;
        let ordered = Tensor::cat(&grouped_outputs, 0)?
            .index_select(&inverse, 0)?
            .reshape((tokens, top_k, self.hidden_dim))?;
        let mut sum = Tensor::zeros((tokens, self.hidden_dim), DType::F32, input.device())?;
        // Do not use a reduction or scatter-add: preserve each token's original
        // route order, with a separate F32 multiply followed by a separate add.
        for slot in 0..top_k {
            let weights = routes
                .iter()
                .map(|selected| selected[slot].1)
                .collect::<Vec<_>>();
            let weights = Tensor::from_vec(weights, (tokens, 1), input.device())?;
            let output = ordered.narrow(1, slot, 1)?.squeeze(1)?;
            sum = (sum + output.broadcast_mul(&weights)?)?;
        }
        Ok(sum)
    }

    fn forward_with_group_limit(
        &self,
        input: &Tensor,
        routes: &[Vec<(usize, f32)>],
        group_limit: usize,
    ) -> Result<Tensor> {
        let (tokens, hidden) = input.dims2()?;
        ensure!(
            tokens == routes.len() && hidden == self.hidden_dim,
            "BF16 route shape mismatch"
        );
        tokens
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(4))
            .context("BF16 output byte count overflow")?;
        // Reject late invalid routes before dispatching any expert work.
        for selected in routes {
            for &(expert, weight) in selected {
                ensure!(expert < self.num_experts, "expert ID out of range");
                ensure!(weight.is_finite(), "non-finite BF16 route weight");
            }
        }
        let top_k = routes.first().map_or(0, Vec::len);
        let grouped = if self.grouped_metal
            && input.device().is_metal()
            && tokens > 1
            && top_k > 0
            && routes.iter().all(|selected| selected.len() == top_k)
        {
            self.grouped_chunk_tokens(top_k, group_limit.min(tokens))
        } else {
            None
        };
        // Nothing persists across layer calls. The cache remains live until
        // the success/error fence, including for groups split across chunks.
        let mut cache: Vec<Option<MoeExpert>> = (0..self.num_experts).map(|_| None).collect();
        let result = (|| -> Result<Tensor> {
            if let Some(chunk_tokens) = grouped {
                let mut output = Vec::with_capacity(tokens.div_ceil(chunk_tokens));
                for (chunk, selected) in routes.chunks(chunk_tokens).enumerate() {
                    let start = chunk
                        .checked_mul(chunk_tokens)
                        .context("BF16 chunk offset overflow")?;
                    let input = input.narrow(0, start, selected.len())?;
                    output.push(self.grouped_chunk(&input, selected, top_k, &mut cache)?);
                    // Release queued group temporaries before the next chunk.
                    input.device().synchronize()?;
                }
                // Join finished chunks once; do not repeatedly copy the prefix.
                Ok(Tensor::cat(&output, 0)?)
            } else {
                self.scalar_forward(input, routes, &mut cache)
            }
        })();
        let synchronized = input.device().synchronize();
        drop(cache);
        // The first fence runs while the cache still owns its uploaded weights.
        // Drain again after dropping it so Metal removes pool-only buffers
        // before the next layer can build another full F32 expert cache. Attempt
        // both fences even if the calculation or first fence returned an error.
        let drained = if input.device().is_metal() {
            input.device().synchronize()
        } else {
            Ok(())
        };
        let output = result?;
        synchronized?;
        drained?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Module;

    fn bank(shape: [usize; 3], salt: usize) -> Result<Arc<QTensor>> {
        let n = shape.iter().product();
        let values = (0..n)
            .map(|i| (((i * 17 + salt) % 31) as f32 - 15.0) / 128.0)
            .collect::<Vec<_>>();
        let tensor = Tensor::from_vec(values, &shape, &Device::Cpu)?;
        Ok(Arc::new(QTensor::quantize(&tensor, GgmlDType::BF16)?))
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn native_bf16_swiglu_keeps_views_and_chunks() -> Result<()> {
        let projection = |dtype| -> Result<QMatMul> {
            Ok(QMatMul::QTensor(Arc::new(QTensor::quantize(
                &Tensor::ones((4, 4), DType::F32, &Device::Cpu)?,
                dtype,
            )?)))
        };
        let expert = MoeExpert {
            native_quant: false,
            gate_proj: projection(GgmlDType::BF16)?,
            up_proj: projection(GgmlDType::BF16)?,
            down_proj: projection(GgmlDType::BF16)?,
        };
        let g = Tensor::from_vec(
            (0..24).map(|i| (i as f32 - 11.0) / 7.0).collect::<Vec<_>>(),
            (6, 4),
            &Device::Cpu,
        )?
        .narrow(0, 1, 4)?;
        let u = Tensor::from_vec(
            (0..24).map(|i| (i as f32 - 7.0) / 9.0).collect::<Vec<_>>(),
            (6, 4),
            &Device::Cpu,
        )?
        .narrow(0, 1, 4)?;
        assert_eq!(g.layout().start_offset(), 4);
        let full = native_bf16_swiglu(&expert, &g, &u)?.unwrap();
        let expected = native_swiglu_compat::native_swiglu(
            &g.flatten_all()?.to_vec1::<f32>()?,
            &u.flatten_all()?.to_vec1::<f32>()?,
        );
        assert_eq!(
            full.flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        // Independent real-number check; the native polynomial need not be
        // bit-identical to correctly rounded expf.
        let gv = g.flatten_all()?.to_vec1::<f32>()?;
        let uv = u.flatten_all()?.to_vec1::<f32>()?;
        for ((&gate, &up), &actual) in gv.iter().zip(&uv).zip(&expected) {
            let exact = f64::from(gate) / (1.0 + (-f64::from(gate)).exp()) * f64::from(up);
            assert!((f64::from(actual) - exact).abs() <= 2e-6);
        }
        for chunk in [1, 2, 3, 4] {
            let mut parts = Vec::new();
            for start in (0..4).step_by(chunk) {
                let n = chunk.min(4 - start);
                parts.push(
                    native_bf16_swiglu(&expert, &g.narrow(0, start, n)?, &u.narrow(0, start, n)?)?
                        .unwrap(),
                );
            }
            assert_eq!(
                Tensor::cat(&parts, 0)?.flatten_all()?.to_vec1::<f32>()?,
                expected
            );
        }
        for dtype in [GgmlDType::F32, GgmlDType::F16] {
            let other = MoeExpert {
                native_quant: false,
                gate_proj: projection(dtype)?,
                up_proj: expert.up_proj.clone(),
                down_proj: expert.down_proj.clone(),
            };
            assert!(native_bf16_swiglu(&other, &g, &u)?.is_none());
        }
        assert!(
            native_bf16_swiglu(&expert, &g.to_dtype(DType::F64)?, &u.to_dtype(DType::F64)?)?
                .is_none()
        );
        assert!(native_bf16_swiglu(&expert, &g.narrow(1, 0, 3)?, &u.narrow(1, 0, 3)?)?.is_none());
        Ok(())
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn native_bf16_projection_views_chunks_and_declines() -> Result<()> {
        let mut values = vec![0.0f32; 4 * 32];
        for row in values.chunks_exact_mut(32) {
            row[..3].copy_from_slice(&[16_777_216.0, 1.0, -16_777_216.0]);
        }
        let weight = Tensor::from_vec(values, (4, 32), &Device::Cpu)?;
        let mm = QMatMul::QTensor(Arc::new(QTensor::quantize(&weight, GgmlDType::BF16)?));
        let mut input = vec![0.0f32; 5 * 32];
        for (row, value) in [
            f32::from_bits(0x3f808000),
            f32::from_bits(0x3f818000),
            -f32::from_bits(0x3f808000),
        ]
        .into_iter()
        .enumerate()
        {
            input[(row + 1) * 32..(row + 2) * 32].fill(value);
        }
        let view = Tensor::from_vec(input, (5, 32), &Device::Cpu)?.narrow(0, 1, 3)?;
        assert_eq!(view.layout().start_offset(), 32);
        let full = native_projection(&mm, &view, true)?
            .unwrap()
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(
            full,
            [vec![1.0; 4], vec![1.015625; 4], vec![-1.0; 4]].concat()
        );
        for chunk in [1, 2, 3] {
            let mut parts = Vec::new();
            for at in (0..3).step_by(chunk) {
                parts.push(
                    native_projection(&mm, &view.narrow(0, at, chunk.min(3 - at))?, true)?.unwrap(),
                );
            }
            assert_eq!(
                Tensor::cat(&parts, 0)?.flatten_all()?.to_vec1::<f32>()?,
                full
            );
        }
        let strided = view.t()?.contiguous()?.t()?;
        assert!(!strided.is_contiguous());
        assert_eq!(
            native_projection(&mm, &strided, true)?
                .unwrap()
                .flatten_all()?
                .to_vec1::<f32>()?,
            full
        );
        assert!(native_projection(&mm, &view, false)?.is_none());
        assert!(native_projection(&mm, &view.to_dtype(DType::F16)?, true)?.is_none());
        assert!(native_projection(&QMatMul::Tensor(weight), &view, true)?.is_none());
        let empty = Tensor::zeros((0, 32), DType::F32, &Device::Cpu)?;
        assert_eq!(
            native_projection(&mm, &empty, true)?.unwrap().dims(),
            [0, 4]
        );
        assert!(native_projection(&mm, &view.narrow(1, 0, 31)?, true).is_err());
        Ok(())
    }

    #[test]
    fn grouped_workspace_limits_all_checked_shapes() -> Result<()> {
        let mut experts = Bf16Experts::new(
            bank([1, 32, 32], 1)?,
            bank([1, 32, 32], 2)?,
            bank([1, 32, 32], 3)?,
            1,
            32,
            32,
            &Device::Cpu,
        )?;
        assert_eq!(experts.grouped_chunk_tokens(8, 128), Some(128));
        experts.num_experts = 128;
        for (hidden, width) in [(2048, 768), (16384, 6144)] {
            experts.hidden_dim = hidden;
            experts.expert_dim = width;
            let count = experts.grouped_chunk_tokens(8, 128).unwrap();
            assert!(count > 1 && count <= 128);
            assert!(
                experts.grouped_workspace_bytes(count, 8).unwrap() <= BF16_GROUP_WORKSPACE_BYTES
            );
            if count < 128 {
                assert!(
                    experts.grouped_workspace_bytes(count + 1, 8).unwrap()
                        > BF16_GROUP_WORKSPACE_BYTES
                );
            }
        }
        assert_eq!(experts.grouped_chunk_tokens(8, 0), None);
        assert_eq!(experts.grouped_workspace_bytes(usize::MAX, 8), None);
        assert_eq!(experts.grouped_workspace_bytes(2, u32::MAX as usize), None);
        experts.hidden_dim = usize::MAX;
        assert_eq!(experts.grouped_workspace_bytes(1, 1), None);
        assert_eq!(experts.grouped_chunk_tokens(1, 128), None);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn grouped_metal_bf16_matches_scalar_routes_and_chunk_partitions() -> Result<()> {
        let device = match Device::new_metal(0) {
            Ok(device) => device,
            Err(error)
                if std::env::var_os("DINF_REQUIRE_METAL_TESTS").is_none()
                    && std::env::var_os("DINF_REQUIRE_METAL").is_none() =>
            {
                eprintln!("Metal unavailable: {error}");
                return Ok(());
            }
            Err(error) => return Err(error.into()),
        };
        let (count, hidden, width) = (4, 32, 64);
        let gate = bank([count, width, hidden], 3)?;
        let experts = Bf16Experts::new(
            gate.clone(),
            bank([count, width, hidden], 7)?,
            bank([count, hidden, width], 11)?,
            count,
            hidden,
            width,
            &device,
        )?;
        let compare = |input: &Tensor, routes: &[Vec<(usize, f32)>]| -> Result<()> {
            let expected = experts
                .forward_with_group_limit(input, routes, 0)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for limit in [32, 128] {
                let actual = experts
                    .forward_with_group_limit(input, routes, limit)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert!(actual.iter().all(|value| value.is_finite()));
                let max_error = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let different = actual
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                eprintln!(
                    "grouped BF16 tokens={} limit={limit} max_abs={max_error} bit_differences={different}",
                    routes.len()
                );
                assert_eq!(
                    different, 0,
                    "grouped output differs from separate F32 MV by {max_error}"
                );
            }
            Ok(())
        };
        // Matrix row sizes on both sides of the kernel's tile boundaries;
        // expert 3 stays unused and the final chunk can hold only one row.
        for tokens in [1, 2, 3, 8, 31, 32, 33, 129] {
            let values = (0..tokens * hidden)
                .map(|i| match i % 37 {
                    0 => -0.0,
                    1 => f32::from_bits(1),
                    _ => ((i * 13 % 97) as f32 - 48.0) / 64.0,
                })
                .collect::<Vec<_>>();
            let input = Tensor::from_vec(values, (tokens, hidden), &device)?;
            compare(&input, &vec![vec![(1, 0.7)]; tokens])?;
            // Duplicate expert IDs must keep both distinct route weights and
            // their original slot, even when the group order differs.
            let routes = (0..tokens)
                .map(|token| {
                    vec![
                        (token % 3, 0.7),
                        ((token + 1) % 3, -0.3),
                        (
                            token % 3,
                            if token % 2 == 0 {
                                -0.0
                            } else {
                                f32::MIN_POSITIVE
                            },
                        ),
                    ]
                })
                .collect::<Vec<_>>();
            compare(&input, &routes)?;
        }
        let input = Tensor::ones((5, hidden), DType::F32, &device)?;
        // Ragged and empty route rows retain the scalar behavior.
        let ragged = vec![
            vec![],
            vec![(2, 0.5)],
            vec![(1, 0.2), (1, 0.8)],
            vec![],
            vec![(0, 1.0)],
        ];
        compare(&input, &ragged)?;
        compare(&input, &vec![vec![]; 5])?;
        let empty = Tensor::zeros((0, hidden), DType::F32, &device)?;
        compare(&empty, &[])?;
        // Bad final routes must fail before an earlier group can dispatch;
        // neither error may retain cache state or affect a later valid call.
        let valid = vec![vec![(0, 0.6), (1, 0.4)]; 5];
        for weight_error in [false, true] {
            let mut invalid = valid.clone();
            if weight_error {
                invalid[4][1].1 = f32::NAN;
            } else {
                invalid[4][1].0 = count;
            }
            assert!(experts.forward(&input, &invalid).is_err());
            compare(&input, &valid)?;
        }
        assert_eq!(Arc::strong_count(&gate), 2);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn single_row_metal_bf16_matches_selected_f32_mv() -> Result<()> {
        let device = match Device::new_metal(0) {
            Ok(device) => device,
            Err(error)
                if std::env::var_os("DINF_REQUIRE_METAL_TESTS").is_none()
                    && std::env::var_os("DINF_REQUIRE_METAL").is_none() =>
            {
                eprintln!("Metal unavailable: {error}");
                return Ok(());
            }
            Err(error) => return Err(error.into()),
        };
        let (count, hidden, width) = (4, 32, 64);
        let gate = bank([count, width, hidden], 3)?;
        let up = bank([count, width, hidden], 7)?;
        let down = bank([count, hidden, width], 11)?;
        let experts = Bf16Experts::new(gate.clone(), up, down, count, hidden, width, &device)?;
        assert!(experts.grouped_metal);
        // The F32 copies must keep every BF16 weight bit, not round to F16.
        let mut cache = (0..count).map(|_| None).collect::<Vec<_>>();
        let selected = experts.cached_expert(&mut cache, 2, &device)?;
        for (actual, bank) in [
            (&selected.gate_proj, &experts.gate),
            (&selected.up_proj, &experts.up),
            (&selected.down_proj, &experts.down),
        ] {
            let QMatMul::Tensor(actual) = actual else {
                anyhow::bail!("single-row expert must use ordinary F32 matmul")
            };
            assert_eq!(actual.dtype(), DType::F32);
            let QMatMul::QTensor(expected) = bank.materialize(2, &Device::Cpu, false)? else {
                anyhow::bail!("expected packed CPU BF16 weights")
            };
            let expected = expected.dequantize(&Device::Cpu)?;
            assert_eq!(
                actual
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
            );
        }
        device.synchronize()?;
        drop(cache);
        let input = Tensor::from_vec(
            (0..hidden)
                .map(|i| ((i * 13 % 97) as f32 - 48.0) / 127.0)
                .collect::<Vec<_>>(),
            (1, hidden),
            &device,
        )?;
        let routes = vec![vec![(2, 0.7), (0, -0.3), (2, 0.125)]];
        let mut expected = Tensor::zeros((1, hidden), DType::F32, &device)?;
        // Keep the original route order, including duplicate expert IDs.
        for &(expert, weight) in &routes[0] {
            let selected = MoeExpert {
                native_quant: false,
                gate_proj: experts.gate.materialize(expert, &device, false)?,
                up_proj: experts.up.materialize(expert, &device, false)?,
                down_proj: experts.down.materialize(expert, &device, false)?,
            };
            expected = (expected + (selected.forward(&input)? * f64::from(weight))?)?;
        }
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        for _ in 0..2 {
            let actual = experts
                .forward(&input, &routes)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert!(actual.iter().all(|value| value.is_finite()));
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "one-row decode must use the F32 mv path for all three projections",
            );
        }
        for invalid in [vec![vec![(count, 1.0)]], vec![vec![(0, f32::NAN)]]] {
            assert!(experts.forward(&input, &invalid).is_err());
        }
        let recovered = experts
            .forward(&input, &routes)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(recovered, expected);
        assert_eq!(Arc::strong_count(&gate), 2);
        Ok(())
    }

    #[test]
    fn expansion_keeps_bf16_range_and_exact_bits() -> Result<()> {
        // Include values outside F16's finite/normal range and signed zero.
        let words = [0x2f80u16, 0x5f80, 0x8000, 0x3f81];
        let bytes = words
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let source = Arc::new(ggml_file::qtensor_from_ggml(
            GgmlDType::BF16,
            &bytes,
            vec![1, 1, 4],
            &Device::Cpu,
        )?);
        let bank = ProjectionBank::new(source.clone(), 1, 1, 4, &Device::Cpu)?;
        assert_eq!(Arc::strong_count(&source), 2);
        let QMatMul::QTensor(weights) = bank.materialize(0, &Device::Cpu, false)? else {
            anyhow::bail!("expected packed BF16 weights")
        };
        assert_eq!(weights.dtype(), GgmlDType::BF16);
        let weights = weights.dequantize(&Device::Cpu)?;
        let bits = weights
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(bits, words.map(|v| u32::from(v) << 16));
        assert!(bank.materialize(1, &Device::Cpu, false).is_err());
        assert!(bank.materialize(usize::MAX, &Device::Cpu, false).is_err());
        assert!(ProjectionBank::new(source, 2, 1, 4, &Device::Cpu).is_err());
        Ok(())
    }

    #[test]
    fn cpu_bf16_projection_preserves_storage_and_other_weight_types() -> Result<()> {
        let values = Tensor::ones((4, 32), DType::F32, &Device::Cpu)?;
        for dtype in [
            GgmlDType::BF16,
            GgmlDType::F32,
            GgmlDType::F16,
            GgmlDType::Q8_0,
        ] {
            let packed = Arc::new(QTensor::quantize(&values, dtype)?);
            let actual = super::super::qtensor_to_qmatmul(packed.clone())?;
            if dtype == GgmlDType::BF16 {
                let QMatMul::QTensor(actual) = actual else {
                    anyhow::bail!("CPU BF16 projection expanded its weights")
                };
                assert!(Arc::ptr_eq(&actual, &packed));
            } else {
                let expected = QMatMul::from_arc(packed)?;
                assert_eq!(
                    std::mem::discriminant(&actual),
                    std::mem::discriminant(&expected)
                );
            }
        }
        Ok(())
    }

    #[test]
    fn cpu_bf16_projection_rounds_ties_and_preserves_offset_chunks() -> Result<()> {
        // Independent BF16 halfway cases: even mantissas round toward/away
        // from zero as needed. F32-expanded matmul leaves these unrounded.
        let input = [
            f32::from_bits(0x3f808000),
            f32::from_bits(0x3f818000),
            -f32::from_bits(0x3f808000),
            -f32::from_bits(0x3f818000),
        ];
        let expected = [1.0f32, 1.015625, -1.0, -1.015625];
        let mut data = vec![7.0f32; 6 * 32];
        for (row, &value) in input.iter().enumerate() {
            data[(row + 1) * 32..(row + 2) * 32].fill(0.0);
            data[(row + 1) * 32] = value;
        }
        let tensor = Tensor::from_vec(data, (6, 32), &Device::Cpu)?;
        let view = tensor.narrow(0, 1, 4)?;
        assert_eq!(view.layout().start_offset(), 32);
        let weights = Tensor::ones((4, 32), DType::F32, &Device::Cpu)?;
        let weights = Arc::new(QTensor::quantize(&weights, GgmlDType::BF16)?);
        let projection = super::super::qtensor_to_qmatmul(weights)?;
        let full = projection.forward(&view)?.flatten_all()?.to_vec1::<f32>()?;
        for row in 0..4 {
            for column in 0..4 {
                assert_eq!(full[row * 4 + column].to_bits(), expected[row].to_bits());
            }
        }
        for chunk in [1, 2, 3, 4] {
            let outputs = (0..4)
                .step_by(chunk)
                .map(|start| projection.forward(&view.narrow(0, start, chunk.min(4 - start))?))
                .collect::<candle_core::Result<Vec<_>>>()?;
            assert_eq!(
                Tensor::cat(&outputs, 0)?.flatten_all()?.to_vec1::<f32>()?,
                full
            );
        }
        Ok(())
    }

    #[test]
    fn cpu_bf16_projection_keeps_nonfinite_results_visible() -> Result<()> {
        let weights = Tensor::ones((4, 32), DType::F32, &Device::Cpu)?;
        let projection = super::super::qtensor_to_qmatmul(Arc::new(QTensor::quantize(
            &weights,
            GgmlDType::BF16,
        )?))?;
        for value in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let mut data = vec![0.0f32; 32];
            data[0] = value;
            let input = Tensor::from_vec(data, (1, 32), &Device::Cpu)?;
            let output = projection
                .forward(&input)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert!(output.iter().all(|x| if value.is_nan() {
                x.is_nan()
            } else {
                *x == value
            }));
        }
        Ok(())
    }

    fn compare_selected_routes(device: &Device) -> Result<()> {
        let (count, hidden, width) = (4, 32, 64);
        let gate = bank([count, width, hidden], 1)?;
        let up = bank([count, width, hidden], 7)?;
        let down = bank([count, hidden, width], 11)?;
        let bounded = Bf16Experts::new(
            gate.clone(),
            up.clone(),
            down.clone(),
            count,
            hidden,
            width,
            device,
        )?;
        // Construction shares the CPU banks and has no persistent F32 cache.
        assert_eq!(Arc::strong_count(&gate), 2);
        assert_eq!(Arc::strong_count(&up), 2);
        assert_eq!(Arc::strong_count(&down), 2);
        let baseline_gate = split_expert_tensor(&gate, count, width, hidden, &Device::Cpu)?;
        let baseline_up = split_expert_tensor(&up, count, width, hidden, &Device::Cpu)?;
        let baseline_down = split_expert_tensor(&down, count, hidden, width, &Device::Cpu)?;
        // The eager CPU baseline also retains BF16. Metal still compares
        // against the original expanded F32 baseline with the same bound.
        let baseline = |projections: Vec<QMatMul>| -> Result<Vec<QMatMul>> {
            if !device.is_cpu() {
                return Ok(projections);
            }
            projections
                .into_iter()
                .map(|projection| {
                    let QMatMul::Tensor(weights) = projection else {
                        anyhow::bail!("expected eager BF16 expansion")
                    };
                    Ok(QMatMul::QTensor(Arc::new(QTensor::quantize(
                        &weights,
                        GgmlDType::BF16,
                    )?)))
                })
                .collect()
        };
        let baseline_gate = baseline(baseline_gate)?;
        let baseline_up = baseline(baseline_up)?;
        let baseline_down = baseline(baseline_down)?;
        let values = (0..3 * hidden)
            .map(|i| (i as f32 - 43.0) / 64.0)
            .collect::<Vec<_>>();
        let cpu_input = Tensor::from_vec(values, (3, hidden), &Device::Cpu)?;
        let routes = vec![
            vec![(3, 0.7), (1, 0.3)],
            vec![(0, 0.2), (3, 0.8)],
            vec![(1, 0.6), (2, 0.4)],
        ];
        let mut expected = Vec::new();
        for (token, selected) in routes.iter().enumerate() {
            let row = cpu_input.i(token)?.unsqueeze(0)?;
            let mut sum = Tensor::zeros((1, hidden), DType::F32, &Device::Cpu)?;
            for &(expert, weight) in selected {
                let g = if device.is_cpu() {
                    project_or_original(&baseline_gate[expert], &row, true)?
                } else {
                    baseline_gate[expert].forward(&row)?
                };
                let u = if device.is_cpu() {
                    project_or_original(&baseline_up[expert], &row, true)?
                } else {
                    baseline_up[expert].forward(&row)?
                };
                // Compare CPU storage and route order with the same native
                // activation policy; Metal keeps the original scalar exp.
                let selected = MoeExpert {
                    native_quant: false,
                    gate_proj: baseline_gate[expert].clone(),
                    up_proj: baseline_up[expert].clone(),
                    down_proj: baseline_down[expert].clone(),
                };
                let activated = if device.is_cpu() {
                    match native_bf16_swiglu(&selected, &g, &u)? {
                        Some(value) => value,
                        None => (candle_nn::Activation::Silu.forward(&g)? * u)?,
                    }
                } else {
                    (candle_nn::Activation::Silu.forward(&g)? * u)?
                };
                let y = if device.is_cpu() {
                    project_or_original(&baseline_down[expert], &activated, true)?
                } else {
                    baseline_down[expert].forward(&activated)?
                };
                sum = (sum + (y * f64::from(weight))?)?;
            }
            expected.extend(sum.flatten_all()?.to_vec1::<f32>()?);
        }
        let input = cpu_input.to_device(device)?;
        for _ in 0..2 {
            let actual = bounded
                .forward(&input, &routes)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let max_error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_error <= 2e-6, "selected expert error {max_error}");
            if device.is_cpu() {
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
        let mut invalid = routes.clone();
        invalid[1][0].0 = count;
        assert!(bounded.forward(&input, &invalid).is_err());
        invalid = routes.clone();
        invalid[1][0].1 = f32::NAN;
        assert!(bounded.forward(&input, &invalid).is_err());
        // An error releases temporary experts and does not poison a later call.
        bounded.forward(&input, &routes)?;
        assert_eq!(Arc::strong_count(&gate), 2);
        Ok(())
    }

    #[test]
    fn selected_bf16_cpu_experts_match_eager_packed_routes() -> Result<()> {
        compare_selected_routes(&Device::Cpu)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn selected_bf16_metal_experts_match_eager_f32_routes() -> Result<()> {
        let device = match Device::new_metal(0) {
            Ok(device) => device,
            Err(error)
                if std::env::var_os("DINF_REQUIRE_METAL_TESTS").is_none()
                    && std::env::var_os("DINF_REQUIRE_METAL").is_none() =>
            {
                eprintln!("Metal unavailable: {error}");
                return Ok(());
            }
            Err(error) => return Err(error.into()),
        };
        compare_selected_routes(&device)
    }

    #[cfg(target_os = "macos")]
    fn compare_metal_fallback(hidden: usize, width: usize, up_dtype: GgmlDType) -> Result<()> {
        let device = match Device::new_metal(0) {
            Ok(device) => device,
            Err(error)
                if std::env::var_os("DINF_REQUIRE_METAL_TESTS").is_none()
                    && std::env::var_os("DINF_REQUIRE_METAL").is_none() =>
            {
                eprintln!("Metal unavailable: {error}");
                return Ok(());
            }
            Err(error) => return Err(error.into()),
        };
        let gate = bank([2, width, hidden], 3)?;
        let up_values = bank([2, width, hidden], 7)?.dequantize(&Device::Cpu)?;
        let up = Arc::new(QTensor::quantize(&up_values, up_dtype)?);
        let down = bank([2, hidden, width], 13)?;
        let bounded = Bf16Experts::new(
            gate.clone(),
            up.clone(),
            down.clone(),
            2,
            hidden,
            width,
            &device,
        )?;
        assert!(!bounded.grouped_metal);
        assert!(matches!(
            bounded
                .gate
                .materialize(0, &device, bounded.grouped_metal)?,
            QMatMul::Tensor(_)
        ));
        if !hidden.is_multiple_of(32) {
            // Bypass the layer policy: a direct request still cannot send an
            // incomplete K tile to the unchecked packed kernel.
            assert!(matches!(
                bounded.gate.materialize(0, &device, true)?,
                QMatMul::Tensor(_)
            ));
        }
        let gates = split_expert_tensor(&gate, 2, width, hidden, &device)?;
        let ups = split_expert_tensor(&up, 2, width, hidden, &device)?;
        let downs = split_expert_tensor(&down, 2, hidden, width, &device)?;
        let experts = gates
            .into_iter()
            .zip(ups)
            .zip(downs)
            .map(|((gate_proj, up_proj), down_proj)| MoeExpert {
                native_quant: false,
                gate_proj,
                up_proj,
                down_proj,
            })
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(
            (0..2 * hidden)
                .map(|i| (i as f32 - 17.0) / 64.0)
                .collect::<Vec<_>>(),
            (2, hidden),
            &device,
        )?;
        let routes = vec![vec![(1, 0.7), (0, 0.3)], vec![(0, 0.6), (1, 0.4)]];
        let mut expected = Vec::new();
        for (token, selected) in routes.iter().enumerate() {
            let row = input.i(token)?.unsqueeze(0)?;
            let mut sum = Tensor::zeros((1, hidden), DType::F32, &device)?;
            for &(expert, weight) in selected {
                sum = (sum + (experts[expert].forward(&row)? * f64::from(weight))?)?;
            }
            expected.extend(sum.flatten_all()?.to_vec1::<f32>()?);
        }
        let actual = bounded
            .forward(&input, &routes)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(actual.iter().all(|value| value.is_finite()));
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "fallback must retain the old one-row projection and route order",
        );
        // No rows must return an empty output without constructing an expert
        // or dispatching a zero-sized packed matrix multiplication.
        let empty = Tensor::zeros((0, hidden), DType::F32, &device)?;
        assert_eq!(bounded.forward(&empty, &[])?.dims(), &[0, hidden]);
        device.synchronize()?;
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mixed_bf16_metal_banks_keep_scalar_fallback() -> Result<()> {
        compare_metal_fallback(32, 64, GgmlDType::Q4_0)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn unaligned_bf16_metal_banks_keep_exact_f32_fallback() -> Result<()> {
        compare_metal_fallback(3, 5, GgmlDType::BF16)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn packed_metal_bf16_keeps_weight_range_and_f32_activations() -> Result<()> {
        let device = match Device::new_metal(0) {
            Ok(device) => device,
            Err(error)
                if std::env::var_os("DINF_REQUIRE_METAL_TESTS").is_none()
                    && std::env::var_os("DINF_REQUIRE_METAL").is_none() =>
            {
                eprintln!("Metal unavailable: {error}");
                return Ok(());
            }
            Err(error) => return Err(error.into()),
        };
        // The first activation lies halfway between BF16 numbers. Rounding it
        // to BF16 would change each output by 32768. Both cases also reject an
        // intermediate F16 cast, once in the weights and once in the inputs.
        for (weight, activation) in [(131072.0f32, 1.0 + 1.0 / 256.0), (1.0, 131072.0)] {
            let weights = Tensor::full(weight, (1, 32, 64), &Device::Cpu)?;
            let source = Arc::new(QTensor::quantize(&weights, GgmlDType::BF16)?);
            let bank = ProjectionBank::new(source, 1, 32, 64, &device)?;
            let projection = bank.materialize(0, &device, true)?;
            let QMatMul::QTensor(packed) = &projection else {
                anyhow::bail!("Metal experts must retain packed BF16 weights")
            };
            assert_eq!(packed.dtype(), GgmlDType::BF16);
            let input = Tensor::full(activation, (2, 64), &device)?;
            let values = projection
                .forward(&input)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let expected = weight * activation * 64.0;
            assert_eq!(values.len(), 64);
            assert!(
                values
                    .iter()
                    .all(|value| value.to_bits() == expected.to_bits())
            );
        }
        device.synchronize()?;
        Ok(())
    }
}
