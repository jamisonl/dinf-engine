//! Native V4 sliding-window and learned compressed attention.
//!
//! Pooling follows DeepSeek's inference/model.py. Cache tensors are immutable:
//! appends replace a short chunk, so checkpoints and request forks may share
//! old tensors without sharing future writes.

use super::*;
use std::collections::HashSet;

#[cfg(test)]
#[path = "attention_tests.rs"]
pub(super) mod tests;

#[derive(Clone, Default)]
struct Rows {
    chunks: Vec<Tensor>,
    len: usize,
}

impl Rows {
    fn push(&mut self, row: Tensor) -> Result<()> {
        anyhow::ensure!(
            row.rank() == 2 && row.dim(0)? == 1,
            "V4 cache needs one row"
        );
        match self.chunks.last_mut() {
            Some(last) if last.dim(0)? < 64 => *last = Tensor::cat(&[&*last, &row], 0)?,
            _ => self.chunks.push(row.force_contiguous()?),
        }
        self.len += 1;
        Ok(())
    }

    fn tensor(&self) -> Result<Tensor> {
        Tensor::cat(&self.chunks.iter().collect::<Vec<_>>(), 0).map_err(Into::into)
    }

    #[cfg(test)]
    fn bytes(&self) -> usize {
        retained_tensor_bytes(self.chunks.iter())
    }
}

fn retained_tensor_bytes<'a>(tensors: impl Iterator<Item = &'a Tensor>) -> usize {
    let mut seen = HashSet::new();
    tensors.fold(0usize, |total, tensor| {
        let (storage, _) = tensor.storage_and_layout();
        let (key, bytes) = match &*storage {
            #[cfg(target_os = "macos")]
            candle_core::Storage::Metal(storage) => (
                (
                    1u8,
                    std::ptr::from_ref(storage.buffer().as_ref()) as *const () as usize,
                ),
                storage.buffer().length(),
            ),
            candle_core::Storage::Cpu(storage) => {
                // Views retain the whole Vec, including spare capacity. Cache
                // tensors use F16/F32; handle other types for complete storage
                // accounting if that representation changes later.
                use candle_core::CpuStorage;
                let bytes = match storage {
                    CpuStorage::U8(v)
                    | CpuStorage::F6E2M3(v)
                    | CpuStorage::F6E3M2(v)
                    | CpuStorage::F4(v)
                    | CpuStorage::F8E8M0(v) => v.capacity(),
                    CpuStorage::U32(v) => v.capacity().saturating_mul(4),
                    CpuStorage::I16(v) => v.capacity().saturating_mul(2),
                    CpuStorage::I32(v) => v.capacity().saturating_mul(4),
                    CpuStorage::I64(v) => v.capacity().saturating_mul(8),
                    CpuStorage::BF16(v) => v.capacity().saturating_mul(2),
                    CpuStorage::F16(v) => v.capacity().saturating_mul(2),
                    CpuStorage::F32(v) => v.capacity().saturating_mul(4),
                    CpuStorage::F64(v) => v.capacity().saturating_mul(8),
                    CpuStorage::F8E4M3(v) => v.capacity(),
                };
                ((0u8, std::ptr::from_ref(storage) as usize), bytes)
            }
            #[cfg(feature = "cuda")]
            candle_core::Storage::Cuda(storage) => {
                // A tensor view retains the full device slice. Count its
                // stored bytes, including byte-packed low precision types.
                use candle_core::cuda_backend::CudaStorageSlice;
                let bytes = match &storage.slice {
                    CudaStorageSlice::U8(v)
                    | CudaStorageSlice::F6E2M3(v)
                    | CudaStorageSlice::F6E3M2(v)
                    | CudaStorageSlice::F4(v)
                    | CudaStorageSlice::F8E8M0(v) => v.len(),
                    CudaStorageSlice::U32(v) => v.len().saturating_mul(4),
                    CudaStorageSlice::I16(v) => v.len().saturating_mul(2),
                    CudaStorageSlice::I32(v) => v.len().saturating_mul(4),
                    CudaStorageSlice::I64(v) => v.len().saturating_mul(8),
                    CudaStorageSlice::BF16(v) => v.len().saturating_mul(2),
                    CudaStorageSlice::F16(v) => v.len().saturating_mul(2),
                    CudaStorageSlice::F32(v) => v.len().saturating_mul(4),
                    CudaStorageSlice::F64(v) => v.len().saturating_mul(8),
                    CudaStorageSlice::F8E4M3(v) => v.len(),
                };
                ((2u8, std::ptr::from_ref(&*storage) as usize), bytes)
            }
            _ => (
                (3u8, std::ptr::from_ref(&*storage) as usize),
                tensor
                    .elem_count()
                    .saturating_mul(tensor.dtype().size_in_bytes()),
            ),
        };
        if seen.insert(key) {
            total.saturating_add(bytes)
        } else {
            total
        }
    })
}

#[derive(Clone, Default)]
struct PoolState {
    values: Option<Tensor>,
    scores: Option<Tensor>,
    previous: Option<(Tensor, Tensor)>,
    completed: Rows,
}

impl PoolState {
    fn tensors(&self) -> impl Iterator<Item = &Tensor> {
        self.values
            .iter()
            .chain(self.scores.iter())
            .chain(self.previous.iter().flat_map(|(a, b)| [a, b]))
            .chain(self.completed.chunks.iter())
    }
}

/// Request-owned V4 window, pooled keys, and partial pooling blocks.
#[derive(Clone, Default)]
pub struct DeepSeek4Cache {
    position: usize,
    window: Option<Tensor>,
    pool: PoolState,
    index: PoolState,
}

impl DeepSeek4Cache {
    pub(super) fn position(&self) -> usize {
        self.position
    }

    pub(crate) fn bytes(&self) -> usize {
        // Deduplicate by storage rather than tensor identity: window/pool
        // views can share one buffer, and prompt snapshots retain its padding.
        retained_tensor_bytes(
            self.window
                .iter()
                .chain(self.pool.tensors())
                .chain(self.index.tensors()),
        )
    }

    pub(crate) fn allocation_tokens(&self) -> usize {
        // Existing V4 admission reserves two F16 512-wide rows per position
        // per layer. Include partial F32 pooling states in that same unit.
        self.position.max(self.bytes().div_ceil(2 * 2 * 512))
    }
}

fn append_row(target: &mut Option<Tensor>, row: &Tensor) -> Result<()> {
    *target = Some(match target.as_ref() {
        Some(previous) => Tensor::cat(&[previous, row], 0)?,
        None => row.force_contiguous()?,
    });
    Ok(())
}

fn rotate_row(row: &Tensor, position: usize, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dim = row.dim(1)?;
    let plain = row.narrow(1, 0, dim - V4_Q_ROPE_DIM)?;
    let rotated = apply_rope(
        &row.narrow(1, dim - V4_Q_ROPE_DIM, V4_Q_ROPE_DIM)?
            .reshape((1, 1, 1, V4_Q_ROPE_DIM))?,
        position,
        cos,
        sin,
    )?
    .reshape((1, V4_Q_ROPE_DIM))?;
    Tensor::cat(&[plain, rotated], 1).map_err(Into::into)
}

struct Compressor {
    values: DeepSeek4QMatMul,
    gates: DeepSeek4QMatMul,
    ape: Tensor,
    norm: V4RmsNorm,
    ratio: usize,
    dim: usize,
}

impl Compressor {
    fn validate_shape(
        weights: &RawMap,
        prefix: &str,
        md: &GgufMetadata,
        ratio: usize,
        dim: usize,
    ) -> Result<usize> {
        let width = dim
            .checked_mul(if ratio == 4 { 2 } else { 1 })
            .context("V4 pooling width overflow")?;
        for (suffix, expected) in [
            ("_ape.weight", vec![ratio, width]),
            ("_kv.weight", vec![width, md.hidden_dim]),
            ("_gate.weight", vec![width, md.hidden_dim]),
            ("_norm.weight", vec![dim]),
        ] {
            let name = format!("{prefix}{suffix}");
            anyhow::ensure!(
                raw(weights, &name)?.shape() == expected,
                "invalid {prefix} pooling tensor shape for {name}"
            );
        }
        Ok(width)
    }

    fn new(
        weights: &RawMap,
        prefix: &str,
        md: &GgufMetadata,
        ratio: usize,
        dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let width = Self::validate_shape(weights, prefix, md, ratio, dim)?;
        let ape = vector(weights, &format!("{prefix}_ape.weight"), device)?;
        anyhow::ensure!(
            ape.dims() == [ratio, width],
            "invalid {prefix} pooling position shape {:?}",
            ape.dims()
        );
        Ok(Self {
            values: linear(weights, &format!("{prefix}_kv.weight"), device)?,
            gates: linear(weights, &format!("{prefix}_gate.weight"), device)?,
            ape,
            norm: norm(weights, &format!("{prefix}_norm"), md, device)?,
            ratio,
            dim,
        })
    }

    fn project(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((
            self.values.forward(x)?.squeeze(0)?,
            self.gates.forward(x)?.squeeze(0)?,
        ))
    }

    fn push(
        &self,
        values: &Tensor,
        scores: &Tensor,
        position: usize,
        state: &mut PoolState,
        rope: &(Tensor, Tensor, Tensor),
        dtype: DType,
    ) -> Result<()> {
        let scores = scores.broadcast_add(&self.ape.narrow(0, position % self.ratio, 1)?)?;
        append_row(&mut state.values, values)?;
        append_row(&mut state.scores, &scores)?;
        if !(position + 1).is_multiple_of(self.ratio) {
            return Ok(());
        }
        let values = state.values.take().context("missing V4 pooling values")?;
        let scores = state.scores.take().context("missing V4 pooling scores")?;
        anyhow::ensure!(
            values.dim(0)? == self.ratio,
            "V4 partial block has wrong length"
        );
        let (pool_values, pool_scores) = if self.ratio == 4 {
            let current_values = values.narrow(1, self.dim, self.dim)?;
            let current_scores = scores.narrow(1, self.dim, self.dim)?;
            let result = match &state.previous {
                Some((previous_values, previous_scores)) => (
                    Tensor::cat(
                        &[previous_values.narrow(1, 0, self.dim)?, current_values],
                        0,
                    )?,
                    Tensor::cat(
                        &[previous_scores.narrow(1, 0, self.dim)?, current_scores],
                        0,
                    )?,
                ),
                None => (current_values, current_scores),
            };
            state.previous = Some((values, scores));
            result
        } else {
            (values, scores)
        };
        let probabilities = softmax_last_dim(&pool_scores.t()?.contiguous()?)?.t()?;
        let pooled = (pool_values * probabilities)?.sum_keepdim(0)?;
        let pooled = self.norm.forward(&pooled)?;
        let pooled = rotate_row(&pooled, position + 1 - self.ratio, &rope.0, &rope.1)?;
        state.completed.push(pooled.to_dtype(dtype)?)
    }
}

struct Indexer {
    query: DeepSeek4QMatMul,
    weights: DeepSeek4QMatMul,
    compressor: Compressor,
    heads: usize,
    dim: usize,
    top_k: usize,
}

pub(super) fn validate_indexer_geometry(md: &GgufMetadata) -> Result<usize> {
    anyhow::ensure!(
        md.indexer_head_count > 0 && md.indexer_head_dim >= V4_Q_ROPE_DIM && md.indexer_top_k > 0,
        "invalid V4 indexer geometry"
    );
    md.indexer_head_dim
        .checked_mul(2)
        .context("V4 pooling width overflow")?;
    md.indexer_head_count
        .checked_mul(md.indexer_head_dim)
        .context("V4 index query width overflow")
}

fn select_index_keys(scores: &[f32], top_k: usize) -> Result<Vec<u32>> {
    anyhow::ensure!(
        top_k > 0 && scores.len() <= u32::MAX as usize,
        "invalid V4 index selection size"
    );
    anyhow::ensure!(
        scores.iter().all(|s| s.is_finite()),
        "non-finite V4 index score"
    );
    let mut indices = (0..scores.len()).collect::<Vec<_>>();
    if indices.len() > top_k {
        // Partition in linear time; don't sort the whole history every token.
        indices.select_nth_unstable_by(top_k, |&a, &b| {
            scores[b].total_cmp(&scores[a]).then(a.cmp(&b))
        });
        indices.truncate(top_k);
    }
    // Position order gives a fixed attention summation order, including ties.
    indices.sort_unstable();
    Ok(indices.into_iter().map(|i| i as u32).collect())
}

pub(super) struct V4Attention {
    q_a: DeepSeek4QMatMul,
    q_a_norm: V4RmsNorm,
    q_b: DeepSeek4QMatMul,
    q_norm: V4RmsNorm,
    kv: DeepSeek4QMatMul,
    kv_norm: V4RmsNorm,
    output_a: Vec<DeepSeek4QMatMul>,
    output_b: DeepSeek4QMatMul,
    sinks: Tensor,
    heads: usize,
    groups: usize,
    output_rank: usize,
    window: usize,
    compressor: Option<Compressor>,
    indexer: Option<Indexer>,
    rope: (Tensor, Tensor, Tensor),
}

impl V4Attention {
    pub(super) fn new(
        weights: &RawMap,
        prefix: &str,
        md: &GgufMetadata,
        layer: usize,
        rope: &(Tensor, Tensor, Tensor),
        device: &Device,
    ) -> Result<Self> {
        anyhow::ensure!(
            md.head_dim == 512 && md.qk_rope_head_dim == 64,
            "unsupported V4 head shape"
        );
        anyhow::ensure!(md.sliding_window > 0, "V4 sliding window is missing");
        anyhow::ensure!(
            md.output_group_count > 0
                && md.num_attention_heads.is_multiple_of(md.output_group_count),
            "invalid V4 output groups"
        );
        let ratio = *md
            .compress_ratios
            .get(layer)
            .context("V4 compression schedule is missing a layer")?;
        if ratio > 0 {
            Compressor::validate_shape(
                weights,
                &format!("{prefix}.attn_compressor"),
                md,
                ratio,
                512,
            )?;
        }
        if ratio == 4 {
            let query_width = validate_indexer_geometry(md)?;
            anyhow::ensure!(
                raw(weights, &format!("{prefix}.indexer.attn_q_b.weight"))?.shape()
                    == [query_width, md.q_lora_rank],
                "invalid V4 index query shape"
            );
            anyhow::ensure!(
                raw(weights, &format!("{prefix}.indexer.proj.weight"))?.shape()
                    == [md.indexer_head_count, md.hidden_dim],
                "invalid V4 index weight shape"
            );
            Compressor::validate_shape(
                weights,
                &format!("{prefix}.indexer_compressor"),
                md,
                ratio,
                md.indexer_head_dim,
            )?;
        }
        let output = raw(weights, &format!("{prefix}.attn_output_a.weight"))?;
        let output_a = (0..md.output_group_count)
            .map(|group| {
                DeepSeek4QMatMul::new_on_device(
                    Arc::new(output.slice_rows(group * md.output_lora_rank, md.output_lora_rank)?),
                    device,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let compressor = (ratio > 0)
            .then(|| {
                Compressor::new(
                    weights,
                    &format!("{prefix}.attn_compressor"),
                    md,
                    ratio,
                    512,
                    device,
                )
            })
            .transpose()?;
        let indexer = if ratio == 4 {
            Some(Indexer {
                query: linear(
                    weights,
                    &format!("{prefix}.indexer.attn_q_b.weight"),
                    device,
                )?,
                weights: linear(weights, &format!("{prefix}.indexer.proj.weight"), device)?,
                compressor: Compressor::new(
                    weights,
                    &format!("{prefix}.indexer_compressor"),
                    md,
                    ratio,
                    md.indexer_head_dim,
                    device,
                )?,
                heads: md.indexer_head_count,
                dim: md.indexer_head_dim,
                top_k: md.indexer_top_k,
            })
        } else {
            None
        };
        Ok(Self {
            q_a: linear(weights, &format!("{prefix}.attn_q_a.weight"), device)?,
            q_a_norm: norm(weights, &format!("{prefix}.attn_q_a_norm"), md, device)?,
            q_b: linear(weights, &format!("{prefix}.attn_q_b.weight"), device)?,
            q_norm: V4RmsNorm::ones(512, md.rms_norm_eps, device)?,
            kv: linear(weights, &format!("{prefix}.attn_kv.weight"), device)?,
            kv_norm: norm(weights, &format!("{prefix}.attn_kv_a_norm"), md, device)?,
            output_a,
            output_b: linear(weights, &format!("{prefix}.attn_output_b.weight"), device)?,
            sinks: vector(weights, &format!("{prefix}.attn_sinks.weight"), device)?,
            heads: md.num_attention_heads,
            groups: md.output_group_count,
            output_rank: md.output_lora_rank,
            window: md.sliding_window,
            compressor,
            indexer,
            rope: rope.clone(),
        })
    }

    pub(super) fn forward(
        &self,
        x: &Tensor,
        slot: &mut Option<LayerCache>,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<Tensor> {
        anyhow::ensure!(
            x.dims3()?.0 == 1 && x.dim(1)? == seq_len,
            "V4 attention expects one sequence"
        );
        if slot.is_none() {
            anyhow::ensure!(start_pos == 0, "V4 cannot resume missing compressed state");
            *slot = Some(LayerCache::DeepSeek4(Box::default()));
        }
        let Some(LayerCache::DeepSeek4(cache)) = slot else {
            anyhow::bail!("V4 cache has wrong type");
        };
        anyhow::ensure!(
            cache.position == start_pos,
            "V4 compressed cache position differs"
        );
        let stamp = profile::start(x.device(), seq_len, Region::AttentionProject)?;
        let qr = self.q_a_norm.forward(&self.q_a.forward(x)?)?;
        let q = self.q_norm.forward(
            &self
                .q_b
                .forward(&qr)?
                .reshape((1, seq_len, self.heads, 512))?,
        )?;
        let kv = self.kv_norm.forward(&self.kv.forward(x)?)?.squeeze(0)?;
        let projections = self.compressor.as_ref().map(|c| c.project(x)).transpose()?;
        let index_projections = self
            .indexer
            .as_ref()
            .map(|i| i.compressor.project(x))
            .transpose()?;
        let index_queries = self
            .indexer
            .as_ref()
            .filter(|i| (start_pos + seq_len) / 4 > i.top_k)
            .map(|i| Ok::<_, anyhow::Error>((i.query.forward(&qr)?, i.weights.forward(x)?)))
            .transpose()?;
        profile::finish(x.device(), stamp)?;
        let stamp = profile::start(x.device(), seq_len, Region::AttentionRows)?;
        let mut outputs = Vec::with_capacity(seq_len);
        for row in 0..seq_len {
            let position = start_pos + row;
            let key = rotate_row(&kv.narrow(0, row, 1)?, position, &self.rope.0, &self.rope.1)?
                .to_dtype(DType::F16)?;
            append_row(&mut cache.window, &key)?;
            let window = cache.window.as_ref().unwrap();
            if window.dim(0)? > self.window {
                cache.window = Some(
                    window
                        .narrow(0, window.dim(0)? - self.window, self.window)?
                        .force_contiguous()?,
                );
            }
            if let (Some(c), Some((values, scores))) = (&self.compressor, &projections) {
                c.push(
                    &values.narrow(0, row, 1)?,
                    &scores.narrow(0, row, 1)?,
                    position,
                    &mut cache.pool,
                    &self.rope,
                    DType::F16,
                )?;
            }
            if let (Some(i), Some((values, scores))) = (&self.indexer, &index_projections) {
                // Keep index keys in F32. An orthogonal Hadamard transform on
                // both operands leaves their dot product unchanged; no lossy
                // index-key quantization or rotation is needed on this path.
                i.compressor.push(
                    &values.narrow(0, row, 1)?,
                    &scores.narrow(0, row, 1)?,
                    position,
                    &mut cache.index,
                    &self.rope,
                    DType::F32,
                )?;
            }
            let mut keys = cache.window.as_ref().unwrap().to_dtype(DType::F32)?;
            if cache.pool.completed.len > 0 {
                let mut pooled = cache.pool.completed.tensor()?.to_dtype(DType::F32)?;
                if let Some(indexer) = &self.indexer {
                    anyhow::ensure!(
                        cache.pool.completed.len == cache.index.completed.len,
                        "V4 index/key count differs"
                    );
                    if cache.index.completed.len > indexer.top_k {
                        let (queries, weights) =
                            index_queries.as_ref().context("missing V4 index queries")?;
                        let query = queries.i((0, row))?.reshape((indexer.heads, indexer.dim))?;
                        let plain = query.narrow(1, 0, indexer.dim - 64)?;
                        let rotary = apply_rope(
                            &query.narrow(1, indexer.dim - 64, 64)?.reshape((
                                1,
                                indexer.heads,
                                1,
                                64,
                            ))?,
                            position,
                            &self.rope.0,
                            &self.rope.1,
                        )?
                        .reshape((indexer.heads, 64))?;
                        let query = Tensor::cat(&[plain, rotary], 1)?;
                        let scores = query
                            .matmul(&cache.index.completed.tensor()?.t()?.contiguous()?)?
                            .relu()?;
                        let weights = (weights.i((0, row))?.reshape((indexer.heads, 1))?
                            * ((indexer.heads * indexer.dim) as f64).sqrt().recip())?;
                        let scores = scores.broadcast_mul(&weights)?.sum(0)?.to_vec1::<f32>()?;
                        let selected = select_index_keys(&scores, indexer.top_k)?;
                        let indices = Tensor::new(selected.as_slice(), x.device())?;
                        pooled = pooled.index_select(&indices, 0)?;
                    }
                }
                keys = Tensor::cat(&[keys, pooled], 0)?;
            }
            let query = q.i((0, row))?;
            let plain = query.narrow(1, 0, V4_Q_NOPE_DIM)?;
            let rotary = apply_rope(
                &query
                    .narrow(1, V4_Q_NOPE_DIM, 64)?
                    .reshape((1, self.heads, 1, 64))?,
                position,
                &self.rope.0,
                &self.rope.1,
            )?
            .reshape((self.heads, 64))?;
            let query = Tensor::cat(&[plain, rotary], 1)?;
            let scores = (query.matmul(&keys.t()?.contiguous()?)? * 512f64.sqrt().recip())?;
            let scores = Tensor::cat(&[scores, self.sinks.reshape((self.heads, 1))?], 1)?;
            let probabilities = softmax_last_dim(&scores)?
                .narrow(1, 0, keys.dim(0)?)?
                .contiguous()?;
            let output = probabilities.matmul(&keys)?;
            let plain = output.narrow(1, 0, V4_Q_NOPE_DIM)?;
            let rotary = apply_rope(
                &output
                    .narrow(1, V4_Q_NOPE_DIM, 64)?
                    .reshape((1, self.heads, 1, 64))?,
                position,
                &self.rope.0,
                &self.rope.2,
            )?
            .reshape((self.heads, 64))?;
            outputs.push(Tensor::cat(&[plain, rotary], 1)?.unsqueeze(0)?);
            cache.position += 1;
        }
        profile::finish(x.device(), stamp)?;
        let stamp = profile::start(x.device(), seq_len, Region::AttentionOutput)?;
        let outputs = Tensor::cat(&outputs, 0)?.reshape((
            1,
            seq_len,
            self.groups,
            self.heads / self.groups * 512,
        ))?;
        let mut groups = Vec::with_capacity(self.groups);
        for (group, projection) in self.output_a.iter().enumerate() {
            groups.push(projection.forward(&outputs.i((.., .., group, ..))?.contiguous()?)?);
        }
        let output = self.output_b.forward(&Tensor::cat(&groups, 2)?.reshape((
            1,
            seq_len,
            self.groups * self.output_rank,
        ))?)?;
        profile::finish(x.device(), stamp)?;
        Ok(output)
    }
}
