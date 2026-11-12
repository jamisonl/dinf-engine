//! Immutable request-local Maya cache chunks. See adjacent NOTICE and LICENSE-MIT.
use std::collections::HashSet;

use anyhow::{Result, ensure};
use candle_core::{DType, Tensor};

use super::engram::History;

/// Model-wide dynamic F32 KV/index storage, averaged over completed pairs.
pub const DYNAMIC_BYTES_PER_TOKEN: usize = 6400;
/// Authoritative request reserve, funded by verified ordinary-Q8 upload charging.
/// Both resident and copy halves charge this; requested prefill still defaults32.
pub const FIXED_CACHE_BYTES: usize = 1536 * 1024 * 1024;
pub const CHUNK_ROWS: usize = 256;
/// Index scores use larger immutable pages; KV gather remains CHUNK_ROWS.
pub const INDEX_PAGE_ROWS: usize = 2048;
pub const OWNERS: [usize; 4] = [2, 8, 14, 20];

pub fn owner(layer: usize) -> Option<usize> {
    match layer {
        0..=1 => None,
        2..=7 => Some(0),
        8..=13 => Some(1),
        14..=19 => Some(2),
        20..=39 => Some(3),
        _ => None,
    }
}

pub fn refresh(layer: usize) -> bool {
    matches!(layer, 2 | 8 | 14 | 20 | 24 | 28 | 32 | 36)
}

/// Absolute causal frontier, independent of an owner's already-published tile.
pub(crate) fn visible_rows(owner: usize, pos: usize) -> usize {
    (pos + 1) / if owner == 3 { 1 } else { 2 }
}

/// Borrow chunks, not a cloned cache snapshot. The final chunk is clipped even
/// when the owner has already published later tokens from the same tile.
pub(crate) struct Prefix<'a> {
    rows: &'a Rows,
    len: usize,
}

impl Prefix<'_> {
    pub(crate) fn len(&self) -> usize {
        self.len
    }
    pub(crate) fn chunks(&self) -> impl Iterator<Item = candle_core::Result<Tensor>> + '_ {
        self.rows
            .chunks
            .iter()
            .take(self.len.div_ceil(self.rows.page_rows))
            .enumerate()
            .map(|(i, chunk)| {
                chunk.narrow(
                    0,
                    0,
                    (self.len - i * self.rows.page_rows).min(self.rows.page_rows),
                )
            })
    }
    pub(crate) fn gather(&self, ids: &Tensor, width: usize) -> Result<Tensor> {
        let chunks = self.chunks().collect::<candle_core::Result<Vec<_>>>()?;
        Ok(super::ops::gather(ids, &chunks, width, self.len)?)
    }
}

/// Keep pooled temporaries only when their backing fits the admission bound.
/// Views may retain arbitrarily oversized Metal buffers; the existing exact
/// private-copy path preserves their values/layout without another pool loan.
/// CPU oracle tensors and already-bounded Metal buffers need no copy.
pub(crate) fn retain_bounded(tensor: Tensor) -> Result<Tensor> {
    if RELAXED_TILE_RETENTION.with(|v| v.get()) {
        return Ok(tensor);
    }
    retain_cached(tensor)
}

/// Strict form of `retain_bounded` for tensors stored in the session cache,
/// which outlive the tile and are charged by the cache ledger.
pub(crate) fn retain_cached(tensor: Tensor) -> Result<Tensor> {
    #[cfg(target_os = "macos")]
    if tensor.device().is_metal()
        && super::super::state_storage_bytes(&tensor)?
            > super::super::fixed_state_tensor_bytes(tensor.dims(), tensor.device())?
    {
        return super::super::copy_persistent_tensor(&tensor);
    }
    Ok(tensor)
}

thread_local! {
    static RELAXED_TILE_RETENTION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Within this scope `retain_bounded` keeps tile transients in their pooled
/// buffers instead of copying oversized ones. The Coder's tiled prompt path
/// uses it: a tile retains at most a few hundred KiB per row, so pool
/// rounding costs tens of MiB against the 1.5 GiB request reserve, while the
/// copies cost a dispatch each. Session-cache tensors use `retain_cached`.
pub(crate) struct RelaxedTileRetention(bool, std::marker::PhantomData<std::rc::Rc<()>>);
impl RelaxedTileRetention {
    pub(crate) fn enter(enabled: bool) -> Self {
        Self(
            RELAXED_TILE_RETENTION.with(|v| v.replace(enabled)),
            std::marker::PhantomData,
        )
    }
}
impl Drop for RelaxedTileRetention {
    fn drop(&mut self) {
        RELAXED_TILE_RETENTION.with(|v| v.set(self.0));
    }
}

/// Candidate sources cannot hide an arbitrarily oversized pool loan behind an
/// exact replacement. Caller must retain the source before this fallible check.
#[track_caller]
pub(crate) fn require_bounded_loan(tensor: &Tensor) -> Result<()> {
    let actual = super::super::state_storage_bytes(tensor)?;
    let bound = super::super::fixed_state_tensor_bytes(tensor.dims(), tensor.device())?;
    ensure!(
        actual <= bound,
        "Maya candidate oversized pooled loan: shape={:?} actual={actual} bound={bound} at {}",
        tensor.dims(),
        std::panic::Location::caller()
    );
    Ok(())
}

/// Uninitialized private carrier: callers must write every element before use.
#[cfg(target_os = "macos")]
pub(crate) fn private_f32(
    shape: candle_core::Shape,
    device: &candle_core::Device,
) -> Result<Tensor> {
    #[cfg(target_os = "macos")]
    if let candle_core::Device::Metal(device) = device {
        let elements = shape.elem_count();
        let buffer = device.new_private_buffer(elements, DType::F32, "maya-candidate")?;
        return Ok(Tensor::from((
            candle_core::Storage::Metal(candle_core::MetalStorage::new(
                buffer,
                device.clone(),
                elements,
                DType::F32,
            )),
            shape,
        )));
    }
    Ok(Tensor::zeros(shape, DType::F32, device)?)
}

/// Candidate concatenation writes directly into its one admitted carrier.
#[cfg_attr(
    not(target_os = "macos"),
    allow(
        clippy::ptr_arg,
        reason = "the Metal path appends tensor owners to this shared Vec contract"
    )
)]
pub(crate) fn cat_rows<T: AsRef<Tensor>>(sources: &[T], _pins: &mut Vec<Tensor>) -> Result<Tensor> {
    let first = sources
        .first()
        .ok_or_else(|| anyhow::anyhow!("empty candidate join"))?
        .as_ref();
    ensure!(first.rank() > 0 && first.dtype() == DType::F32);
    let mut shape = first.dims().to_vec();
    shape[0] = 0;
    for source in sources {
        let source = source.as_ref();
        ensure!(
            source.dtype() == DType::F32
                && source.device().same_device(first.device())
                && source.rank() == first.rank()
                && source.dims()[1..] == first.dims()[1..]
        );
        shape[0] = shape[0]
            .checked_add(source.dim(0)?)
            .ok_or_else(|| anyhow::anyhow!("candidate join overflow"))?;
    }
    #[cfg(target_os = "macos")]
    if first.device().is_metal() {
        use candle_core::backend::BackendStorage;
        let output = private_f32(shape.into(), first.device())?;
        _pins.push(output.clone()); // before either fallible GPU copy
        let (destination_storage, _) = output.storage_and_layout();
        let candle_core::Storage::Metal(storage) = &*destination_storage else {
            unreachable!()
        };
        // A private, unpublished allocation; only these copies write it.
        let mut destination = candle_core::MetalStorage::new(
            storage.buffer().clone().into(),
            storage.device().clone(),
            output.elem_count(),
            DType::F32,
        );
        let mut offset = 0;
        for source in sources {
            let source = source.as_ref();
            let (storage, layout) = source.storage_and_layout();
            let candle_core::Storage::Metal(storage) = &*storage else {
                anyhow::bail!("Maya join device");
            };
            storage.copy_strided_src(&mut destination, offset, layout)?;
            offset += source.elem_count();
        }
        drop(destination_storage);
        return Ok(output);
    }
    Ok(Tensor::cat(sources, 0)?)
}

fn join_rows(left: &Tensor, right: &Tensor, pins: &mut Vec<Tensor>) -> Result<Tensor> {
    cat_rows(&[left, right], pins)
}

/// Sealed chunks are shared by forks; replacing the last chunk never writes
/// shared storage, including storage retained by a checkpoint or failed pass.
#[derive(Clone)]
pub struct Rows {
    pub(crate) chunks: Vec<Tensor>,
    pub(crate) len: usize,
    page_rows: usize,
}

impl Default for Rows {
    fn default() -> Self {
        Self::with_page_rows(CHUNK_ROWS)
    }
}

impl Rows {
    fn with_page_rows(page_rows: usize) -> Self {
        assert!(matches!(page_rows, 256 | 2048));
        Self {
            chunks: Vec::new(),
            len: 0,
            page_rows,
        }
    }

    #[cfg(test)]
    pub(crate) fn fixture_pages(page_rows: usize, chunks: Vec<Tensor>, len: usize) -> Self {
        Self {
            chunks,
            len,
            ..Self::with_page_rows(page_rows)
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn prefix(&self, len: usize) -> Result<Prefix<'_>> {
        ensure!(
            len <= self.len,
            "Maya visible prefix exceeds published rows"
        );
        Ok(Prefix { rows: self, len })
    }

    pub(crate) fn append(&mut self, row: Tensor) -> Result<()> {
        ensure!(
            row.rank() == 2 && row.dim(0)? == 1 && row.dtype() == DType::F32,
            "Maya cache append requires one F32 row"
        );
        if self.len.is_multiple_of(self.page_rows) {
            self.chunks.push(retain_cached(row)?);
        } else {
            let last = self.chunks.last_mut().expect("nonempty chunk");
            *last = retain_cached(Tensor::cat(&[&*last, &row], 0)?)?;
        }
        self.len += 1;
        Ok(())
    }

    /// Candidate-only bulk publication. Sealed pages retain identity; an old
    /// partial tail is replaced once, even when this batch crosses a page.
    /// Build privately so allocation/shape errors leave the receiver unchanged.
    /// `pins` belongs to the outer router/error fence, including copy sources.
    pub(crate) fn append_many(&mut self, rows: Tensor, pins: &mut Vec<Tensor>) -> Result<()> {
        ensure!(
            rows.rank() == 2
                && rows.dtype() == DType::F32
                && rows.is_contiguous()
                && rows.dim(0)? > 0,
            "invalid Maya bulk rows"
        );
        if let Some(first) = self.chunks.first() {
            ensure!(
                rows.dim(1)? == first.dim(1)? && rows.device().same_device(first.device()),
                "Maya bulk row mismatch"
            );
        }
        let count = rows.dim(0)?;
        let end = self
            .len
            .checked_add(count)
            .ok_or_else(|| anyhow::anyhow!("Maya bulk row overflow"))?;
        pins.push(rows.clone());
        let mut chunks = self.chunks.clone();
        let mut offset = 0;
        let tail = self.len % self.page_rows;
        if tail != 0 {
            let previous = chunks.last().expect("nonempty Maya tail").clone();
            pins.push(previous.clone());
            let take = count.min(self.page_rows - tail);
            let joined = join_rows(&previous, &rows.narrow(0, 0, take)?, pins)?;
            pins.push(joined.clone());
            require_bounded_loan(&joined)?;
            let replacement = retain_cached(joined)?;
            pins.push(replacement.clone());
            *chunks.last_mut().unwrap() = replacement;
            offset = take;
        }
        while offset < count {
            let take = (count - offset).min(self.page_rows);
            let page = retain_cached(rows.narrow(0, offset, take)?)?;
            pins.push(page.clone());
            chunks.push(page);
            offset += take;
        }
        self.chunks = chunks;
        self.len = end;
        Ok(())
    }
}

#[derive(Clone)]
pub(crate) struct Compressed {
    pub kv: Rows,
    pub index: Rows,
    pub pending: Option<(Tensor, Tensor)>,
}

impl Default for Compressed {
    fn default() -> Self {
        Self {
            kv: Rows::with_page_rows(CHUNK_ROWS),
            index: Rows::with_page_rows(INDEX_PAGE_ROWS),
            pending: None,
        }
    }
}

/// Slot zero owns the entire model cache. No state is carried in slots 1..39.
#[derive(Clone)]
pub struct DeepSeek41Cache {
    pub(crate) raw: [Option<Tensor>; 40],
    pub(crate) compressed: [Compressed; 4],
    pub(crate) history: History,
    pub(crate) position: usize,
}

impl Default for DeepSeek41Cache {
    fn default() -> Self {
        Self {
            raw: std::array::from_fn(|_| None),
            compressed: std::array::from_fn(|_| Compressed::default()),
            history: History::default(),
            position: 0,
        }
    }
}

impl DeepSeek41Cache {
    /// Absolute next-token position, unaffected by the bounded raw windows.
    pub fn position(&self) -> usize {
        self.position
    }

    pub(crate) fn validate_position(
        &self,
        start: usize,
        count: usize,
        context: usize,
    ) -> Result<()> {
        ensure!(
            count > 0 && start == self.position,
            "Maya cache position/sequence mismatch"
        );
        let end = start
            .checked_add(count)
            .ok_or_else(|| anyhow::anyhow!("Maya position overflow"))?;
        ensure!(
            end <= context && end <= 1_048_576,
            "Maya context limit exceeded"
        );
        self.validate_frontier()
    }

    pub(crate) fn validate_frontier(&self) -> Result<()> {
        let start = self.position;
        for (i, cache) in self.compressed.iter().enumerate() {
            let ratio = if i == 3 { 1 } else { 2 };
            ensure!(
                cache.kv.len == start / ratio && cache.index.len == start / ratio,
                "Maya compressed cache frontier mismatch"
            );
            ensure!(
                cache.pending.is_some() == (ratio == 2 && start % 2 == 1),
                "Maya pending pair frontier mismatch"
            );
        }
        for raw in &self.raw {
            ensure!(
                raw.as_ref().map_or(0, |x| x.dims()[0]) == start.min(128),
                "Maya raw cache frontier mismatch"
            );
        }
        Ok(())
    }

    /// Metadata only, after the existing whole-forward fence. No device reads.
    pub(crate) fn frontier_report(&self) -> Result<serde_json::Value> {
        self.validate_frontier()?;
        Ok(serde_json::json!({
            "validated": true, "position": self.position,
            "history_compressed_tail": self.history.compressed_tail(),
            "raw_rows": self.raw.iter().map(|x| x.as_ref().map_or(0, |x| x.dims()[0])).collect::<Vec<_>>(),
            "compressed_kv_rows": self.compressed.iter().map(|c| c.kv.len()).collect::<Vec<_>>(),
            "compressed_index_rows": self.compressed.iter().map(|c| c.index.len()).collect::<Vec<_>>(),
            "pending_pairs": self.compressed.iter().map(|c| c.pending.is_some()).collect::<Vec<_>>(),
            "retained_bytes": self.bytes(), "allocation_tokens": self.allocation_tokens(),
            "scope": "history and validated cache frontiers/backing sizes; not a cache tensor value hash",
        }))
    }

    pub(crate) fn append_raw(&mut self, layer: usize, row: Tensor) -> Result<&Tensor> {
        let next = match &self.raw[layer] {
            None => row,
            Some(previous) => {
                let n = previous.dim(0)?;
                // Trim BEFORE concatenation: 129 rows would round to a 512 KiB
                // Metal bucket instead of 256 KiB for each full raw ring.
                let retained = previous.narrow(0, n.saturating_sub(127), n.min(127))?;
                Tensor::cat(&[&retained, &row], 0)?
            }
        };
        self.raw[layer] = Some(retain_cached(next)?);
        Ok(self.raw[layer].as_ref().unwrap())
    }

    fn tensors(&self, dynamic_only: bool) -> impl Iterator<Item = &Tensor> {
        self.raw
            .iter()
            .filter_map(move |x| if dynamic_only { None } else { x.as_ref() })
            .chain(self.compressed.iter().flat_map(move |c| {
                c.kv.chunks.iter().chain(c.index.chunks.iter()).chain(
                    c.pending
                        .iter()
                        .flat_map(|(a, b)| [a, b])
                        .filter(move |_| !dynamic_only),
                )
            }))
    }

    /// Actual retained tensor allocations, deduplicating views of each buffer.
    pub fn bytes(&self) -> usize {
        retained_bytes(self.tensors(false))
    }

    /// Dynamic allocation in the same model-wide units used by GGUF budgeting.
    pub fn allocation_tokens(&self) -> usize {
        // Fixed rings/pairs are reserved separately. Dynamic storage is charged
        // using exactly the metadata's model-wide F32 byte rate, never *40.
        self.position
            .max(retained_bytes(self.tensors(true)).div_ceil(DYNAMIC_BYTES_PER_TOKEN))
    }

    pub(crate) fn predicted_allocation_tokens(
        position: usize,
        device: &candle_core::Device,
    ) -> Result<usize> {
        let bytes = Self::predicted_dynamic_bytes(position, device.is_metal())?;
        // Only dynamic carrier buffers belong in this unit count. The fixed
        // reserve and the engine's copy/snapshot headroom are charged elsewhere.
        Ok(position.max(bytes.div_ceil(DYNAMIC_BYTES_PER_TOKEN)))
    }

    pub(super) fn predicted_dynamic_bytes(position: usize, rounded: bool) -> Result<usize> {
        ensure!(position <= 1_048_576, "Maya context limit exceeded");
        let mut bytes = 0;
        for ratio in [2, 2, 2, 1] {
            let rows = position / ratio;
            for (width, page_rows) in [(512, CHUNK_ROWS), (128, INDEX_PAGE_ROWS)] {
                let full = rows / page_rows * page_rows * width * 4;
                let tail = rows % page_rows * width * 4;
                bytes += full
                    + if rounded && tail != 0 {
                        tail.next_power_of_two()
                    } else {
                        tail
                    };
            }
        }
        Ok(bytes)
    }
}

pub(super) fn retained_bytes<'a>(tensors: impl Iterator<Item = &'a Tensor>) -> usize {
    let mut seen = HashSet::new();
    tensors
        .map(|t| {
            let (storage, _) = t.storage_and_layout();
            let (key, bytes) = match &*storage {
                #[cfg(target_os = "macos")]
                candle_core::Storage::Metal(s) => (
                    std::ptr::from_ref(s.buffer().as_ref()) as *const () as usize,
                    s.buffer().length(),
                ),
                candle_core::Storage::Cpu(candle_core::CpuStorage::F32(v)) => {
                    (v.as_ptr() as usize, v.capacity() * 4)
                }
                _ => unreachable!("Maya cache storage must be F32 CPU oracle or Metal"),
            };
            if seen.insert(key) { bytes } else { 0 }
        })
        .sum()
}
