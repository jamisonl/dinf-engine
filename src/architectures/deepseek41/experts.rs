//! Bounded, packed Maya-XS expert triplets; no whole-bank or CPU weight cache.
//!
//! The graph MUST hold one whole-forward mutex across preparation AND execution
//! (`ModelArchitecture::forward` is `&self + Send + Sync`). A lookup-only mutex
//! is unsafe. Keep every selected Arc alive until GPU completion. Outstanding
//! Arcs remain charged and cannot be evicted; preparation refuses an impossible
//! union before disk I/O. Drop prior selections after synchronizing, then prepare
//! the next union. This module also synchronizes before eviction and, critically,
//! AFTER dropping victims and BEFORE uploads: Candle pools dropped buffers.
//!
//! Use the returned packed matmuls, not the generic indexed MoE dispatcher (which
//! cannot consume IQ1_S). The caller performs scalar-ordered expert accumulation
//! in F32, as in V4 `grouped_experts_forward`. This is not a model integration.
//!
//! Production uses exactly the raw loader's 72 GiB reservation. Every individual
//! GPU buffer is charged rounded up to 16 KiB. Temporary triplets are reserved
//! before the first read, never partially published, and remain charged until
//! dropped buffers have been synchronized/pruned, even if cleanup fails.

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use candle_core::Device;

use crate::architectures::deepseek4_qmatmul::DeepSeek4QMatMul;
use crate::deepseek4_gguf::{
    DEEPSEEK41_EXPERT_CACHE_BYTES, DeepSeek4DType, MAX_SOURCE_READ_BYTES, RawGgufSource, RawTensor,
};

const ALLOCATION_GRANULE: u64 = 16 * 1024;
const MAX_SELECTED: usize = 384;
const MAX_LAYERS: usize = 40;
const KEY_CAPACITY: usize = MAX_LAYERS * MAX_SELECTED;
const NO_SLOT: usize = usize::MAX;

// IQ1_S has 50 bytes per 256 values. Round each production matrix separately,
// exactly as geometry()/rounded_charge() do, not the combined packed triplet.
const PRODUCTION_MATRIX_BYTES: u64 = 2304 * 5120 / 256 * 50;
const PRODUCTION_TRIPLET_CHARGE: u64 =
    3 * PRODUCTION_MATRIX_BYTES.div_ceil(ALLOCATION_GRANULE) * ALLOCATION_GRANULE;
const MODEL_HOST_VECTOR_BYTES: u64 = super::engram::VOCAB_SIZE as u64
    * std::mem::size_of::<u32>() as u64
    + 40 * 384 * std::mem::size_of::<f32>() as u64;
const EXPERT_HOST_SLOT_BYTES: u64 = 4096;
const MODEL_HOST_FIXED_BYTES: u64 = 16 * 1024 * 1024;

/// Conservative retained MODEL-HOST allowance, separate from the source header.
///
/// Includes the independent Engram token_map (517,120 B) and forty router bias
/// Vecs (61,440 B), once. Before eviction, the stable-slot LRU fallibly reserves
/// at most min(40 * 384, budget / minimum triplet charge + 384) slots. Production
/// uses its exact triplet charge; bounded synthetic geometry uses 3 * 16 KiB.
/// Each 4 KiB slot allowance includes Slot (Entry, two links and an Option tag),
/// `Arc<ExpertTriplet>` (three QMatMuls), three `Arc<RawTensor>` values, their rank-two
/// shape Vecs and Tensor host allocations. Vacant slots reuse their next link
/// as the free list, with no separate allocation. Tests size these objects,
/// retaining >= 3 KiB/slot for allocator rounding and indirect Candle metadata.
///
/// The existing 16 MiB fixed allowance also covers the 40 * 384 usize dense
/// index (122,880 B on 64-bit hosts), three <= 384 preparation Vecs, a 384-bool
/// request-membership array, and the LRU headers. It still covers the forty-layer
/// Vec and output Linear objects, small Engram arrays and RoPE state.
/// weights::validate_source and Attention::load bound resident Tensor/RawTensor
/// host objects below 64 per layer plus three global objects. At 4 KiB/object
/// these use < 11 MiB; size assertions include all directly sized bookkeeping.
/// No disk bank/table payload or whole-weights CPU cache is included. Opaque
/// Metal driver state is not measured here; the existing 10% + 512 MiB runtime
/// reserve remains separate and unchanged.
///
/// Derived from the same reservation used by ExpertCache::new and the profile,
/// so a future cache-budget increase cannot silently outgrow this allowance.
pub const DEEPSEEK41_MODEL_HOST_BYTES: u64 = model_host_bytes(DEEPSEEK41_EXPERT_CACHE_BYTES);

const fn model_host_bytes(cache_bytes: u64) -> u64 {
    let slots = cache_bytes / PRODUCTION_TRIPLET_CHARGE + MAX_SELECTED as u64;
    MODEL_HOST_VECTOR_BYTES + MODEL_HOST_FIXED_BYTES + slots * EXPERT_HOST_SLOT_BYTES
}

/// Both coordinates are necessary: expert numbers are local to each layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertKey {
    /// Transformer layer number.
    pub layer: usize,
    /// Expert index within that layer.
    pub expert: usize,
}

/// An indivisible packed gate/up/down lease. Matmuls cannot escape independently.
pub struct ExpertTriplet {
    gate: DeepSeek4QMatMul,
    up: DeepSeek4QMatMul,
    down: DeepSeek4QMatMul,
}

impl ExpertTriplet {
    /// Packed gate projection; keep the owning Arc through GPU completion.
    pub fn gate(&self) -> &DeepSeek4QMatMul {
        &self.gate
    }

    /// Packed up projection; keep the owning Arc through GPU completion.
    pub fn up(&self) -> &DeepSeek4QMatMul {
        &self.up
    }

    /// Packed down projection; keep the owning Arc through GPU completion.
    pub fn down(&self) -> &DeepSeek4QMatMul {
        &self.down
    }
}

/// Deduplicated selection in first-requested order, pinning the complete union.
/// Cloning an Arc prolongs its charge; dropping this selection alone need not
/// make its entries evictable. No independent packed buffer handles are exposed.
pub struct SelectedExperts {
    entries: Vec<(ExpertKey, Arc<ExpertTriplet>)>,
}

impl SelectedExperts {
    /// Iterate the requested union; callers may retain individual Arc leases.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &(ExpertKey, Arc<ExpertTriplet>)> {
        self.entries.iter()
    }

    /// Find a selected layer-local expert without another cache lookup.
    pub fn get(&self, expert: usize) -> Option<&Arc<ExpertTriplet>> {
        self.entries
            .iter()
            .find(|(key, _)| key.expert == expert)
            .map(|(_, entry)| entry)
    }
}

/// Qualification counters, not per-token product logging. Hits/misses count
/// distinct requested keys after budget validation, including failed uploads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Resident + reserved in-flight + dropped-but-not-yet-pruned buffers.
    pub charged_bytes: u64,
    /// Fully published resident entries (including externally leased entries).
    pub entries: usize,
    /// Reserved charge for the currently staged union.
    pub in_flight_bytes: u64,
    /// Dropped buffer charge awaiting a successful device synchronization.
    pub retired_bytes: u64,
    /// Distinct resident key lookups.
    pub hits: u64,
    /// Distinct absent key lookups.
    pub misses: u64,
    /// Fully published triplets removed from the LRU.
    pub evictions: u64,
}

struct Entry {
    key: ExpertKey,
    weights: Arc<ExpertTriplet>,
    charge: u64,
}

/// Indices, rather than addresses, remain stable when the slot Vec reallocates.
/// A vacant slot's `next` is its free-list link; occupied slots form the exact
/// oldest-to-newest doubly linked list. NO_SLOT cannot be a bounded valid index.
struct Slot {
    entry: Option<Entry>,
    prev: usize,
    next: usize,
}

#[derive(Default)]
struct ExpertLru {
    slots: Vec<Slot>,
    index: Vec<usize>,
    oldest: usize,
    newest: usize,
    free: usize,
    len: usize,
}

impl ExpertLru {
    fn new() -> Self {
        Self {
            oldest: NO_SLOT,
            newest: NO_SLOT,
            free: NO_SLOT,
            ..Self::default()
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn lookup(&self, key: ExpertKey) -> usize {
        // prepare validates both coordinates before every production lookup.
        debug_assert!(key.layer < MAX_LAYERS && key.expert < MAX_SELECTED);
        self.index
            .get(key.layer * MAX_SELECTED + key.expert)
            .copied()
            .unwrap_or(NO_SLOT)
    }

    fn iter(&self) -> impl Iterator<Item = &Entry> {
        let mut next = self.oldest;
        std::iter::from_fn(move || {
            if next == NO_SLOT {
                return None;
            }
            let slot = &self.slots[next];
            next = slot.next;
            Some(slot.entry.as_ref().expect("occupied LRU link"))
        })
    }

    fn reserve(&mut self, misses: usize, budget: u64, production: bool) -> Result<()> {
        let minimum = if production {
            PRODUCTION_TRIPLET_CHARGE
        } else {
            3 * ALLOCATION_GRANULE
        };
        let limit = (budget / minimum)
            .saturating_add(MAX_SELECTED as u64)
            .min(KEY_CAPACITY as u64) as usize;
        let needed = self
            .len
            .checked_add(misses)
            .context("expert slot count overflow")?;
        ensure!(needed <= limit, "expert slot capacity exceeded");
        if self.index.is_empty() && needed != 0 {
            self.index.try_reserve_exact(KEY_CAPACITY)?;
            self.index.resize(KEY_CAPACITY, NO_SLOT);
        }
        if needed > self.slots.len() {
            self.slots.try_reserve_exact(needed - self.slots.len())?;
            while self.slots.len() < needed {
                let slot = self.slots.len();
                self.slots.push(Slot {
                    entry: None,
                    prev: NO_SLOT,
                    next: self.free,
                });
                self.free = slot;
            }
        }
        Ok(())
    }

    fn unlink(&mut self, index: usize) {
        let (prev, next) = (self.slots[index].prev, self.slots[index].next);
        if prev == NO_SLOT {
            self.oldest = next;
        } else {
            self.slots[prev].next = next;
        }
        if next == NO_SLOT {
            self.newest = prev;
        } else {
            self.slots[next].prev = prev;
        }
    }

    fn append(&mut self, index: usize) {
        self.slots[index].prev = self.newest;
        self.slots[index].next = NO_SLOT;
        if self.newest == NO_SLOT {
            self.oldest = index;
        } else {
            self.slots[self.newest].next = index;
        }
        self.newest = index;
    }

    fn insert(&mut self, entry: Entry) {
        let key = entry.key;
        debug_assert_eq!(self.lookup(key), NO_SLOT);
        let index = self.free;
        self.free = self.slots[index].next;
        self.slots[index].entry = Some(entry);
        self.index[key.layer * MAX_SELECTED + key.expert] = index;
        self.append(index);
        self.len += 1;
    }

    fn remove(&mut self, index: usize) -> Entry {
        self.unlink(index);
        let entry = self.slots[index].entry.take().expect("occupied victim");
        self.index[entry.key.layer * MAX_SELECTED + entry.key.expert] = NO_SLOT;
        self.slots[index].prev = NO_SLOT;
        self.slots[index].next = self.free;
        self.free = index;
        self.len -= 1;
        entry
    }

    fn touch(&mut self, index: usize) {
        if index != self.newest {
            self.unlink(index);
            self.append(index);
        }
    }

    #[cfg(test)]
    fn assert_integrity(&self, budget: u64, production: bool) {
        let minimum = if production {
            PRODUCTION_TRIPLET_CHARGE
        } else {
            3 * ALLOCATION_GRANULE
        };
        let limit = (budget / minimum)
            .saturating_add(MAX_SELECTED as u64)
            .min(KEY_CAPACITY as u64) as usize;
        assert!(self.slots.capacity() <= limit);
        assert!(self.index.capacity() <= KEY_CAPACITY);
        assert!(self.index.is_empty() || self.index.len() == KEY_CAPACITY);
        let mut seen = vec![false; self.slots.len()];
        let (mut prev, mut cursor, mut count) = (NO_SLOT, self.oldest, 0);
        while cursor != NO_SLOT {
            assert!(cursor < self.slots.len());
            assert!(!seen[cursor], "occupied list cycle");
            seen[cursor] = true;
            let slot = &self.slots[cursor];
            assert_eq!(slot.prev, prev);
            let entry = slot.entry.as_ref().expect("vacant occupied link");
            assert!(entry.key.layer < MAX_LAYERS && entry.key.expert < MAX_SELECTED);
            assert_eq!(self.lookup(entry.key), cursor);
            prev = cursor;
            cursor = slot.next;
            count += 1;
        }
        assert_eq!(prev, self.newest);
        assert_eq!(count, self.len);
        cursor = self.free;
        while cursor != NO_SLOT {
            assert!(cursor < self.slots.len());
            assert!(!seen[cursor], "free-list cycle or occupied free slot");
            seen[cursor] = true;
            let slot = &self.slots[cursor];
            assert!(slot.entry.is_none());
            assert_eq!(slot.prev, NO_SLOT);
            cursor = slot.next;
        }
        assert!(seen.into_iter().all(|v| v), "unreachable stable slot");
        for (key_index, &slot) in self.index.iter().enumerate() {
            if slot == NO_SLOT {
                continue;
            }
            let key = self.slots[slot]
                .entry
                .as_ref()
                .expect("dense index points to free slot")
                .key;
            assert_eq!(key.layer * MAX_SELECTED + key.expert, key_index);
        }
    }
}

struct Geometry {
    names: [String; 3],
    experts: usize,
    ffn: usize,
    hidden: usize,
    lengths: [usize; 3],
    dtypes: [DeepSeek4DType; 3],
    charge: u64,
}

/// Single-device cache. Only preparation needs `&mut self`; the graph must still
/// serialize entire forwards, not just calls to this cache. Both key coordinates
/// are bounded (40 layers, 384 experts); synthetic layers may use smaller banks
/// and different geometry/charges. No growing HashMap or wrapping clock.
pub struct ExpertCache {
    source: Arc<RawGgufSource>,
    device: Device,
    budget: u64,
    production_geometry: bool,
    entries: ExpertLru,
    resident_bytes: u64,
    in_flight_bytes: u64,
    retired_bytes: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl ExpertCache {
    /// Construct the production Metal cache with the exact loader reservation.
    /// This reserves a ledger limit, not an eager 72 GiB allocation. The graph
    /// owns one cache for this source/device; no backend or dispatch is selected.
    pub(crate) fn new(source: Arc<RawGgufSource>, device: Device) -> Result<Self> {
        ensure!(device.is_metal(), "native Maya expert cache requires Metal");
        Ok(Self::with_limit(
            source,
            device,
            DEEPSEEK41_EXPERT_CACHE_BYTES,
            true,
        ))
    }

    fn with_limit(
        source: Arc<RawGgufSource>,
        device: Device,
        budget: u64,
        production_geometry: bool,
    ) -> Self {
        Self {
            source,
            device,
            budget,
            production_geometry,
            entries: ExpertLru::new(),
            resident_bytes: 0,
            in_flight_bytes: 0,
            retired_bytes: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Snapshot the exact conservative ledger, including failed-cleanup charge.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            charged_bytes: self.resident_bytes + self.in_flight_bytes + self.retired_bytes,
            entries: self.entries.len(),
            in_flight_bytes: self.in_flight_bytes,
            retired_bytes: self.retired_bytes,
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
        }
    }

    fn geometry(&self, layer: usize) -> Result<Geometry> {
        ensure!(layer < MAX_LAYERS, "expert layer outside bounded cache");
        let names =
            ["gate", "up", "down"].map(|part| format!("blk.{layer}.ffn_{part}_exps.weight"));
        let [gate, up, down] = names.each_ref().map(|name| {
            self.source
                .content()
                .tensor_infos
                .get(name)
                .with_context(|| format!("missing expert bank {name}"))
        });
        let infos = [gate?, up?, down?];
        let coder = self.source.is_coder();
        ensure!(
            infos.iter().all(|info| info.shape.len() == 3
                && if coder {
                    matches!(
                        info.dtype,
                        DeepSeek4DType::IQ1M | DeepSeek4DType::IQ2XXS | DeepSeek4DType::IQ2XS
                    )
                } else {
                    info.dtype == DeepSeek4DType::IQ1S
                }),
            "unsupported rank-3 expert banks"
        );
        let (experts, ffn, hidden) = (infos[0].shape[0], infos[0].shape[1], infos[0].shape[2]);
        ensure!(
            experts > 0 && experts <= MAX_SELECTED && ffn > 0 && hidden > 0,
            "invalid expert bank geometry"
        );
        ensure!(
            infos[1].shape == [experts, ffn, hidden] && infos[2].shape == [experts, hidden, ffn],
            "gate/up/down expert bank shape/count mismatch"
        );
        if self.production_geometry {
            ensure!(
                (experts, ffn, hidden) == (if coder { 192 } else { 384 }, 2304, 5120),
                "unsupported Maya expert geometry"
            );
        }
        // Validate row block alignment, each complete bank count, and each
        // single-expert I/O bound BEFORE reading any of the three matrices.
        let mut lengths = [0; 3];
        let mut charge = 0u64;
        for (index, info) in infos.iter().enumerate() {
            let length = info
                .dtype
                .byte_len(info.shape[2])?
                .checked_mul(info.shape[1])
                .context("expert matrix length overflow")?;
            ensure!(
                length <= MAX_SOURCE_READ_BYTES,
                "expert exceeds bounded source I/O"
            );
            ensure!(
                length.checked_mul(experts) == Some(info.byte_len()?),
                "expert bank byte count mismatch"
            );
            lengths[index] = length;
            charge = charge
                .checked_add(rounded_charge(length)?)
                .context("expert charge overflow")?;
        }
        Ok(Geometry {
            names,
            experts,
            ffn,
            hidden,
            lengths,
            dtypes: infos.map(|info| info.dtype),
            charge,
        })
    }

    /// Atomically prepare up to 384 distinct experts in one layer. Repeated IDs
    /// are deduplicated. The requested union and every outstanding Arc lease are
    /// pinned across all misses. Insufficient capacity/invalid IDs fail before
    /// source I/O and do not mutate the cache. On a read/upload error no new
    /// entries are published; valid old entries may remain or have been evicted.
    pub fn prepare(&mut self, layer: usize, experts: &[usize]) -> Result<SelectedExperts> {
        self.prepare_with(layer, experts, RawGgufSource::read_expert_slice)
    }

    fn prepare_with(
        &mut self,
        layer: usize,
        experts: &[usize],
        mut read: impl FnMut(&RawGgufSource, &str, usize, &Device) -> Result<RawTensor>,
    ) -> Result<SelectedExperts> {
        use super::profile::{self, Detail};
        let bookkeeping = profile::detail(Detail::ExpertBookkeeping);
        let geometry = self.geometry(layer)?;
        let mut keys = Vec::new();
        keys.try_reserve_exact(experts.len().min(MAX_SELECTED))?;
        let mut requested = [false; MAX_SELECTED];
        for &expert in experts {
            ensure!(expert < geometry.experts, "expert ID outside bank");
            let key = ExpertKey { layer, expert };
            if !requested[expert] {
                requested[expert] = true;
                keys.push(key);
            }
        }
        let misses = keys
            .iter()
            .filter(|key| self.entries.lookup(**key) == NO_SLOT)
            .count();
        let missing_charge = geometry
            .charge
            .checked_mul(misses as u64)
            .context("expert union charge overflow")?;
        // Only skip the full pinned-union scan when ALL residents plus ALL
        // misses fit, regardless of how many outstanding leases callers hold.
        if self
            .resident_bytes
            .checked_add(missing_charge)
            .is_none_or(|n| n > self.budget)
        {
            let pinned_charge = self
                .entries
                .iter()
                .filter(|entry| {
                    (entry.key.layer == layer && requested[entry.key.expert])
                        || Arc::strong_count(&entry.weights) > 1
                })
                .try_fold(0u64, |sum, entry| {
                    sum.checked_add(entry.charge)
                        .context("pinned expert charge overflow")
                })?;
            ensure!(
                pinned_charge
                    .checked_add(missing_charge)
                    .is_some_and(|n| n <= self.budget),
                "expert cache budget insufficient for requested union and outstanding leases"
            );
        }

        let mut selected = Vec::new();
        selected.try_reserve_exact(keys.len())?;
        let mut staged = Vec::new();
        staged.try_reserve_exact(misses)?;
        // Every later slot/index/link operation is allocation-free. Reserve
        // before eviction/I/O, including slots needed by the complete staging.
        self.entries
            .reserve(misses, self.budget, self.production_geometry)?;
        drop(bookkeeping);
        // Any previous failed cleanup is retried before touching the LRU.
        self.synchronize_retired()?;
        let bookkeeping = profile::detail(Detail::ExpertBookkeeping);
        while self.resident_bytes + missing_charge > self.budget {
            let mut victim = self.entries.oldest;
            while victim != NO_SLOT {
                let slot = &self.entries.slots[victim];
                let entry = slot.entry.as_ref().expect("occupied victim candidate");
                if !(entry.key.layer == layer && requested[entry.key.expert])
                    && Arc::strong_count(&entry.weights) == 1
                {
                    break;
                }
                victim = slot.next;
            }
            ensure!(
                victim != NO_SLOT,
                "expert leases changed during preparation"
            );
            let entry = self.entries.remove(victim);
            self.resident_bytes -= entry.charge;
            self.retired_bytes += entry.charge;
            self.evictions = self.evictions.saturating_add(1);
            profile::cache_events(0, 0, 1);
            drop(entry);
        }
        drop(bookkeeping);
        if misses > 0 {
            // A pre-eviction wait is NOT sufficient: this wait happens after
            // victims were dropped. Candle's device wait prunes its buffer pool.
            self.synchronize_retired()?;
        }
        self.hits = self.hits.saturating_add((keys.len() - misses) as u64);
        self.misses = self.misses.saturating_add(misses as u64);
        profile::cache_events((keys.len() - misses) as u64, misses as u64, 0);
        self.in_flight_bytes = missing_charge;
        let loaded = (|| -> Result<()> {
            for &key in &keys {
                if self.entries.lookup(key) != NO_SLOT {
                    continue;
                }
                let mut matrix = |index: usize, shape: [usize; 2]| -> Result<DeepSeek4QMatMul> {
                    let raw = read(
                        &self.source,
                        &geometry.names[index],
                        key.expert,
                        &self.device,
                    )?;
                    ensure!(
                        raw.dtype() == geometry.dtypes[index]
                            && raw.shape() == shape
                            && raw.byte_len() == geometry.lengths[index],
                        "expert slice changed geometry"
                    );
                    DeepSeek4QMatMul::new(Arc::new(raw))
                };
                let weights = ExpertTriplet {
                    gate: matrix(0, [geometry.ffn, geometry.hidden])?,
                    up: matrix(1, [geometry.ffn, geometry.hidden])?,
                    down: matrix(2, [geometry.hidden, geometry.ffn])?,
                };
                staged.push(Entry {
                    key,
                    weights: Arc::new(weights),
                    charge: geometry.charge,
                });
            }
            Ok(())
        })();
        if let Err(error) = loaded {
            drop(staged); // Includes every complete temporary from earlier misses.
            self.retired_bytes += self.in_flight_bytes;
            self.in_flight_bytes = 0;
            if let Err(cleanup) = self.synchronize_retired() {
                // Keep the conservative reservation until a later successful
                // wait. Never upload over an unpruned failed temporary.
                return Err(error.context(format!("expert cleanup failed: {cleanup:#}")));
            }
            return Err(error);
        }
        let bookkeeping = profile::detail(Detail::ExpertBookkeeping);
        for entry in staged {
            self.entries.insert(entry);
        }
        self.resident_bytes += self.in_flight_bytes;
        self.in_flight_bytes = 0;
        for key in keys {
            let index = self.entries.lookup(key);
            let entry = self.entries.slots[index]
                .entry
                .as_ref()
                .expect("prepared expert");
            selected.push((key, Arc::clone(&entry.weights)));
            self.entries.touch(index);
        }
        debug_assert!(self.stats().charged_bytes <= self.budget);
        drop(bookkeeping);
        Ok(SelectedExperts { entries: selected })
    }

    fn synchronize_retired(&mut self) -> Result<()> {
        if self.retired_bytes == 0 {
            return Ok(());
        }
        let _wait = super::profile::detail(super::profile::Detail::ExpertRetirementWait);
        self.device
            .synchronize()
            .context("synchronizing/pruning expert buffers")?;
        self.retired_bytes = 0;
        Ok(())
    }
}

fn rounded_charge(bytes: usize) -> Result<u64> {
    let bytes = u64::try_from(bytes).context("expert buffer length exceeds u64")?;
    bytes
        .checked_add(ALLOCATION_GRANULE - 1)
        .map(|n| n / ALLOCATION_GRANULE * ALLOCATION_GRANULE)
        .context("expert buffer rounded charge overflow")
}

#[cfg(test)]
#[path = "expert_lru_cpu.rs"]
mod expert_lru_cpu;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deepseek4_gguf::fixtures;
    use std::collections::HashMap;
    use std::fs::File;
    use tempfile::NamedTempFile;

    const WIDTH: usize = 256;
    const MATRIX_BYTES: usize = WIDTH * WIDTH / 256 * 50;
    const TRIPLET_CHARGE: u64 = 3 * ALLOCATION_GRANULE;

    fn banks(layers: usize, experts: usize, ffn: usize, hidden: usize) -> NamedTempFile {
        let mut specs = Vec::new();
        let mut offset = 0;
        for layer in 0..layers {
            for part in ["gate", "up", "down"] {
                let shape = if part == "down" {
                    vec![experts, hidden, ffn]
                } else {
                    vec![experts, ffn, hidden]
                };
                let length = DeepSeek4DType::IQ1S
                    .byte_len(experts * ffn * hidden)
                    .unwrap() as u64;
                specs.push((
                    format!("blk.{layer}.ffn_{part}_exps.weight"),
                    shape,
                    19,
                    offset,
                ));
                offset += length;
            }
        }
        let specs = specs
            .iter()
            .map(|(name, shape, dtype, offset)| (name.as_str(), shape.as_slice(), *dtype, *offset))
            .collect::<Vec<_>>();
        fixtures::sparse(&HashMap::new(), &specs).0
    }

    fn source(file: &NamedTempFile) -> Result<Arc<RawGgufSource>> {
        Ok(Arc::new(RawGgufSource::new(File::open(file.path())?)?))
    }

    fn cache(file: &NamedTempFile, triplets: u64) -> Result<ExpertCache> {
        Ok(ExpertCache::with_limit(
            source(file)?,
            Device::Cpu,
            triplets * TRIPLET_CHARGE,
            false,
        ))
    }

    fn assert_ledger(cache: &ExpertCache) {
        let stats = cache.stats();
        let resident: u64 = cache.entries.iter().map(|entry| entry.charge).sum();
        assert_eq!(
            stats.charged_bytes,
            resident + stats.in_flight_bytes + stats.retired_bytes
        );
        assert!(stats.charged_bytes <= cache.budget, "{stats:?}");
        assert_eq!(stats.in_flight_bytes, 0);
        assert_eq!(stats.retired_bytes, 0);
    }

    #[test]
    fn host_diag_expert_selected_bytes_cache_and_identity_failure() -> Result<()> {
        if crate::test_in_subprocess(concat!(
            module_path!(),
            "::host_diag_expert_selected_bytes_cache_and_identity_failure"
        )) {
            return Ok(());
        }
        use super::super::profile::{self, Detail};
        let file = banks(1, 4, WIDTH, WIDTH);
        let mut cache = cache(&file, 2)?;
        let _guard = profile::HostGuard::enable();
        drop(cache.prepare(0, &[0, 0, 1])?);
        drop(cache.prepare(0, &[0])?);
        drop(cache.prepare(0, &[2])?);
        let snapshot = profile::host_snapshot();
        assert_eq!(snapshot.cache_events, [1, 3, 1]);
        let reads = snapshot.details[Detail::ExpertRead as usize];
        assert_eq!(reads.calls, 9);
        assert_eq!(reads.requested_bytes, 9 * MATRIX_BYTES as u64);
        assert_eq!(reads.completed_bytes, reads.requested_bytes);
        assert_eq!(
            snapshot.details[Detail::ExpertMetadataPre as usize].calls,
            9
        );
        assert_eq!(
            snapshot.details[Detail::ExpertMetadataPost as usize].calls,
            9
        );
        assert_eq!(
            snapshot.details[Detail::ExpertRetirementWait as usize].calls,
            1
        );
        assert_ledger(&cache);
        profile::reset();
        // Ordinary descriptor mutation must still reject BEFORE payload reads.
        file.as_file()
            .set_len(file.as_file().metadata()?.len() - 1)?;
        let error = cache
            .prepare(0, &[3])
            .err()
            .context("mutated source accepted")?;
        assert!(error.to_string().contains("identity"));
        let snapshot = profile::host_snapshot();
        assert_eq!(
            snapshot.details[Detail::ExpertMetadataPre as usize].calls,
            1
        );
        assert_eq!(
            snapshot.details[Detail::ExpertMetadataPost as usize].calls,
            0
        );
        assert_eq!(
            snapshot.details[Detail::ExpertRead as usize].requested_bytes,
            0
        );
        assert_eq!(snapshot.cache_events, [0, 1, 1]);
        assert_eq!(cache.stats().entries, 1);
        assert_ledger(&cache);
        Ok(())
    }

    #[test]
    fn deepseek41_host_budget_covers_production_cache_bookkeeping() -> Result<()> {
        use std::mem::size_of;

        // Descriptor-only production geometry: no bank reads or GPU allocation.
        let file = banks(1, 384, 2304, 5120);
        let cache = ExpertCache::with_limit(
            source(&file)?,
            Device::Cpu,
            DEEPSEEK41_EXPERT_CACHE_BYTES,
            true,
        );
        let geometry = cache.geometry(0)?;
        assert_eq!(geometry.charge, PRODUCTION_TRIPLET_CHARGE);
        assert_eq!(MODEL_HOST_VECTOR_BYTES, 578_560);
        let max_resident = cache.budget / geometry.charge;
        let slots = max_resident + MAX_SELECTED as u64;
        // Slot includes Entry, the Option tag and both occupied/free links.
        // Triplet includes all three QMatMuls; RawTensor its Tensor handle.
        // Add Arc control blocks and each independent rank-two shape allocation.
        let direct_slot_bytes = (size_of::<Slot>()
            + size_of::<ExpertTriplet>()
            + 2 * size_of::<usize>()
            + 3 * (size_of::<RawTensor>() + 4 * size_of::<usize>()))
            as u64;
        assert!(direct_slot_bytes * 4 <= EXPERT_HOST_SLOT_BYTES);
        let scratch_bytes = MAX_SELECTED
            * (size_of::<ExpertKey>()
                + size_of::<(ExpertKey, Arc<ExpertTriplet>)>()
                + size_of::<Entry>())
            + size_of::<[bool; MAX_SELECTED]>();
        let index_bytes = KEY_CAPACITY * size_of::<usize>();
        let lru_header_bytes = size_of::<ExpertLru>();
        let bounded_slots = slots.min(KEY_CAPACITY as u64);
        let mut lru = ExpertLru::new();
        lru.reserve(MAX_SELECTED, cache.budget, true)?;
        lru.assert_integrity(cache.budget, true);
        assert_eq!(lru.index.capacity(), KEY_CAPACITY);
        assert_eq!(lru.slots.capacity(), MAX_SELECTED);
        assert!(lru.slots.capacity() as u64 <= bounded_slots);
        assert_eq!(
            size_of::<Slot>(),
            size_of::<Option<Entry>>() + 2 * size_of::<usize>()
        );
        assert!(bounded_slots * direct_slot_bytes * 4 <= slots * EXPERT_HOST_SLOT_BYTES);
        let direct_model_bytes =
            size_of::<super::super::DeepSeek41Stage>() + 40 * size_of::<super::super::Layer>();
        // Conservative object count from weights::validate_source and
        // Attention::load, including dense companions.
        let resident_metadata_bytes = (40 * 64 + 3) * 4096;
        assert!(
            (scratch_bytes
                + index_bytes
                + lru_header_bytes
                + direct_model_bytes
                + resident_metadata_bytes) as u64
                + MODEL_HOST_VECTOR_BYTES
                < MODEL_HOST_FIXED_BYTES
        );
        let expected =
            MODEL_HOST_VECTOR_BYTES + MODEL_HOST_FIXED_BYTES + slots * EXPERT_HOST_SLOT_BYTES;
        assert_eq!(DEEPSEEK41_MODEL_HOST_BYTES, expected);
        assert!(expected >= MODEL_HOST_VECTOR_BYTES + slots * direct_slot_bytes * 4);
        // A larger reservation must fund every additional possible triplet,
        // even if it exceeds the original production cache/model size.
        assert_eq!(
            model_host_bytes(cache.budget + geometry.charge * 1024),
            expected + 1024 * EXPERT_HOST_SLOT_BYTES,
        );
        assert_eq!(
            model_host_bytes(cache.budget * 2),
            MODEL_HOST_VECTOR_BYTES
                + MODEL_HOST_FIXED_BYTES
                + (cache.budget * 2 / geometry.charge + MAX_SELECTED as u64)
                    * EXPERT_HOST_SLOT_BYTES,
        );
        eprintln!(
            "host allowance={expected} vectors={MODEL_HOST_VECTOR_BYTES} \
             triplet_charge={} max_resident={max_resident} slots={slots} \
             direct_slot_bytes={direct_slot_bytes} scratch_bytes={scratch_bytes} \
             index_bytes={index_bytes} lru_header_bytes={lru_header_bytes} \
             bounded_slots={bounded_slots} direct_model_bytes={direct_model_bytes} \
             resident_metadata_bytes={resident_metadata_bytes}",
            geometry.charge,
        );
        Ok(())
    }

    #[test]
    fn deepseek41_cache_hits_lru_duplicates_and_layer_keys() -> Result<()> {
        let file = banks(2, 4, WIDTH, WIDTH);
        let mut cache = cache(&file, 2)?;
        let first = cache.prepare(0, &[0, 0, 1, 0])?;
        assert_eq!(
            first.iter().map(|(key, _)| key.expert).collect::<Vec<_>>(),
            [0, 1]
        );
        let weak0 = Arc::downgrade(first.get(0).unwrap());
        let weak1 = Arc::downgrade(first.get(1).unwrap());
        drop(first);
        let hit = cache.prepare(0, &[0, 0])?;
        assert!(std::ptr::eq(
            weak0.as_ptr(),
            Arc::as_ptr(hit.get(0).unwrap())
        ));
        drop(hit);
        drop(cache.prepare(0, &[2])?); // Evicts expert 1, not the most recent hit.
        assert!(weak1.upgrade().is_none());
        assert!(weak0.upgrade().is_some());
        let other_layer = cache.prepare(1, &[0])?;
        assert_eq!(
            other_layer.iter().next().unwrap().0,
            ExpertKey {
                layer: 1,
                expert: 0
            }
        );
        assert!(!std::ptr::eq(
            weak0.as_ptr(),
            Arc::as_ptr(other_layer.get(0).unwrap())
        ));
        assert!(weak0.upgrade().is_none());
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 4);
        assert_eq!(cache.stats().evictions, 2);
        assert_eq!(cache.stats().charged_bytes, 2 * TRIPLET_CHARGE);
        assert_ledger(&cache);
        eprintln!("CACHE_CPU_LRU {:?}", cache.stats());
        Ok(())
    }

    #[test]
    fn deepseek41_cache_pins_requested_union_before_all_misses() -> Result<()> {
        let file = banks(1, 4, WIDTH, WIDTH);
        let mut cache = cache(&file, 2)?;
        let old = cache.prepare(0, &[0, 1])?;
        let weak0 = Arc::downgrade(old.get(0).unwrap());
        let weak1 = Arc::downgrade(old.get(1).unwrap());
        drop(old);
        // 0 is oldest, but requested after the miss. It must never be evicted.
        let selected = cache.prepare(0, &[2, 0, 2])?;
        assert!(std::ptr::eq(
            weak0.as_ptr(),
            Arc::as_ptr(selected.get(0).unwrap())
        ));
        assert!(weak1.upgrade().is_none());
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 3);
        assert_eq!(cache.stats().evictions, 1);
        assert_ledger(&cache);
        Ok(())
    }

    #[test]
    fn deepseek41_cache_insufficient_union_and_invalid_ids_before_io() -> Result<()> {
        let file = banks(1, 4, WIDTH, WIDTH);
        let mut cache = cache(&file, 2)?;
        drop(cache.prepare(0, &[0])?);
        let before = cache.stats();
        for ids in [&[0, 1, 2][..], &[0, 4], &[usize::MAX]] {
            assert!(
                cache
                    .prepare_with(0, ids, |_, _, _, _| panic!("unexpected disk I/O"))
                    .is_err()
            );
            assert_eq!(cache.stats(), before);
        }
        assert!(
            cache
                .prepare_with(99, &[0], |_, _, _, _| panic!("unexpected disk I/O"))
                .is_err()
        );
        assert_eq!(cache.stats(), before);
        let mut too_small =
            ExpertCache::with_limit(source(&file)?, Device::Cpu, TRIPLET_CHARGE - 1, false);
        assert!(
            too_small
                .prepare_with(0, &[0], |_, _, _, _| panic!("unexpected disk I/O"))
                .is_err()
        );
        assert_eq!(too_small.stats().charged_bytes, 0);
        assert_ledger(&cache);
        Ok(())
    }

    #[test]
    fn deepseek41_cache_arc_leases_refuse_then_recover() -> Result<()> {
        let file = banks(1, 4, WIDTH, WIDTH);
        let mut cache = cache(&file, 2)?;
        let selected0 = cache.prepare(0, &[0])?;
        let lease0 = Arc::clone(selected0.get(0).unwrap());
        let selected1 = cache.prepare(0, &[1])?;
        drop(selected0); // The separately cloned lease must still be counted.
        let before = cache.stats();
        assert!(
            cache
                .prepare_with(0, &[2], |_, _, _, _| {
                    panic!("leased entries must not be evicted")
                })
                .is_err()
        );
        assert_eq!(cache.stats(), before);
        drop(lease0);
        let selected2 = cache.prepare(0, &[2])?;
        assert!(cache.entries.iter().any(|entry| entry.key.expert == 1));
        assert!(cache.entries.iter().any(|entry| entry.key.expert == 2));
        assert!(!cache.entries.iter().any(|entry| entry.key.expert == 0));
        assert_eq!(cache.stats().evictions, 1);
        drop((selected1, selected2));
        assert_ledger(&cache);
        Ok(())
    }

    #[test]
    fn deepseek41_cache_read_failures_drop_partial_and_complete_staged_triplets() -> Result<()> {
        let file = banks(1, 4, WIDTH, WIDTH);
        for fail_at in 0..6 {
            let mut cache = cache(&file, 2)?;
            drop(cache.prepare(0, &[0])?);
            let mut calls = 0;
            let result = cache.prepare_with(0, &[1, 2], |source, name, expert, device| {
                let index = calls;
                calls += 1;
                if index == fail_at {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "injected expert read failure",
                    )
                    .into());
                }
                source.read_expert_slice(name, expert, device)
            });
            assert!(result.is_err());
            assert_eq!(calls, fail_at + 1);
            // Old 0 was evicted; even a fully staged expert 1 never publishes.
            assert_eq!(cache.stats().entries, 0);
            assert_eq!(cache.stats().charged_bytes, 0);
            assert_ledger(&cache);
            let recovered = cache.prepare(0, &[1, 2])?;
            assert_eq!(
                recovered
                    .iter()
                    .map(|(key, _)| key.expert)
                    .collect::<Vec<_>>(),
                [1, 2]
            );
            assert_eq!(cache.stats().charged_bytes, 2 * TRIPLET_CHARGE);
            assert_ledger(&cache);
        }
        // An error need not discard a valid, requested resident entry.
        let mut cache = cache(&file, 2)?;
        let old = cache.prepare(0, &[0])?;
        let weak = Arc::downgrade(old.get(0).unwrap());
        drop(old);
        let mut calls = 0;
        assert!(
            cache
                .prepare_with(0, &[0, 1], |source, name, expert, device| {
                    calls += 1;
                    if calls == 3 {
                        anyhow::bail!("injected down upload/read failure");
                    }
                    source.read_expert_slice(name, expert, device)
                })
                .is_err()
        );
        assert_eq!(cache.stats().entries, 1);
        assert_eq!(cache.stats().charged_bytes, TRIPLET_CHARGE);
        let recovered = cache.prepare(0, &[0])?;
        assert!(std::ptr::eq(
            weak.as_ptr(),
            Arc::as_ptr(recovered.get(0).unwrap())
        ));
        assert_ledger(&cache);
        Ok(())
    }

    #[test]
    fn deepseek41_cache_validates_three_banks_and_non_square_geometry() -> Result<()> {
        let names = [
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_up_exps.weight",
            "blk.0.ffn_down_exps.weight",
        ];
        // Every fixture is a valid bounded GGUF header/payload; these are cache
        // contract errors, not malformed headers rejected by raw preflight.
        for (up, down, dtype) in [
            (vec![3, 256, 256], vec![4, 256, 256], 19),
            (vec![4, 512, 256], vec![4, 256, 256], 19),
            (vec![4, 256, 256], vec![4, 256, 512], 19),
            (vec![4, 256, 256], vec![4, 256, 256], 0),
            (vec![4, 65536], vec![4, 256, 256], 19),
        ] {
            let gate = [4, 256, 256];
            let specs = [
                (names[0], gate.as_slice(), 19, 0),
                (names[1], up.as_slice(), dtype, 262144),
                (names[2], down.as_slice(), 19, 1310720),
            ];
            let file = fixtures::sparse(&HashMap::new(), &specs).0;
            let mut cache = cache(&file, 2)?;
            assert!(
                cache
                    .prepare_with(0, &[0], |_, _, _, _| {
                        panic!("geometry validation must precede I/O")
                    })
                    .is_err()
            );
            assert_eq!(cache.stats().charged_bytes, 0);
        }
        let gate = [4, 256, 256];
        let incomplete = [
            (names[0], gate.as_slice(), 19, 0),
            (names[1], gate.as_slice(), 19, 51200),
        ];
        let file = fixtures::sparse(&HashMap::new(), &incomplete).0;
        let mut incomplete_cache = cache(&file, 2)?;
        assert!(
            incomplete_cache
                .prepare_with(0, &[0], |_, _, _, _| {
                    panic!("missing down bank must precede I/O")
                })
                .is_err()
        );
        // A valid non-square triplet preserves gate/up versus down orientation.
        let file = banks(1, 4, 512, 256);
        let mut cache = cache(&file, 2)?;
        let selected = cache.prepare(0, &[0])?;
        let weights = selected.get(0).unwrap();
        assert_eq!(
            (weights.gate().out_dim(), weights.gate().in_dim()),
            (512, 256)
        );
        assert_eq!((weights.up().out_dim(), weights.up().in_dim()), (512, 256));
        assert_eq!(
            (weights.down().out_dim(), weights.down().in_dim()),
            (256, 512)
        );
        assert_eq!(cache.stats().charged_bytes, 2 * TRIPLET_CHARGE);
        assert_ledger(&cache);
        Ok(())
    }

    #[test]
    fn deepseek41_cache_accepts_384_distinct_and_many_duplicates() -> Result<()> {
        let file = banks(1, MAX_SELECTED, WIDTH, WIDTH);
        let ids = (0..MAX_SELECTED).collect::<Vec<_>>();
        let mut cache = cache(&file, MAX_SELECTED as u64 - 1)?;
        assert!(
            cache
                .prepare_with(0, &ids, |_, _, _, _| panic!(
                    "union budget must precede I/O"
                ))
                .is_err()
        );
        cache.budget += TRIPLET_CHARGE;
        let all = cache.prepare(0, &ids)?;
        assert_eq!(
            all.iter().map(|(key, _)| key.expert).collect::<Vec<_>>(),
            ids
        );
        assert_eq!(
            cache.stats().charged_bytes,
            MAX_SELECTED as u64 * TRIPLET_CHARGE
        );
        drop(all);
        let repeats = cache.prepare(0, &[7; 2048])?;
        assert_eq!(
            repeats
                .iter()
                .map(|(key, _)| key.expert)
                .collect::<Vec<_>>(),
            [7]
        );
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 384);
        assert_ledger(&cache);
        Ok(())
    }

    #[test]
    fn deepseek41_cache_production_geometry_and_per_buffer_rounding() -> Result<()> {
        let file = banks(1, 384, 2304, 5120); // Sparse ~2.65 GB, never loaded.
        let source = source(&file)?;
        let cache = ExpertCache::with_limit(
            Arc::clone(&source),
            Device::Cpu,
            DEEPSEEK41_EXPERT_CACHE_BYTES,
            true,
        );
        let geometry = cache.geometry(0)?;
        assert_eq!(geometry.lengths, [2_304_000; 3]);
        assert_eq!(geometry.lengths.iter().sum::<usize>(), 6_912_000);
        assert_eq!(geometry.lengths.iter().sum::<usize>() * 6, 41_472_000);
        assert_eq!(geometry.charge, 6_930_432);
        assert_eq!(geometry.charge * 6, 41_582_592);
        assert_eq!(cache.budget, 72 * 1024 * 1024 * 1024);
        assert_eq!(rounded_charge(1)?, ALLOCATION_GRANULE);
        assert_eq!(rounded_charge(16384)?, ALLOCATION_GRANULE);
        assert_eq!(rounded_charge(16385)?, 2 * ALLOCATION_GRANULE);
        assert!(ExpertCache::new(source, Device::Cpu).is_err());
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn scale(layer: usize, expert: usize, part: usize, row: usize) -> f32 {
        (1 + layer * 4 + expert * 2 + part + row % 4) as f32 / 256.0
    }

    #[cfg(target_os = "macos")]
    fn fill_nonzero(file: &NamedTempFile, source: &RawGgufSource) -> Result<()> {
        use std::os::unix::fs::FileExt;
        for layer in 0..2 {
            for (part, label) in ["gate", "up", "down"].iter().enumerate() {
                let name = format!("blk.{layer}.ffn_{label}_exps.weight");
                let info = &source.content().tensor_infos[&name];
                for expert in 0..4 {
                    let mut bytes = vec![0u8; MATRIX_BYTES];
                    for row in 0..WIDTH {
                        // GGML grid 0 is [-1; 8], high index 0 and delta +1/8.
                        // One block per row, so every scalar is -7/8 * d.
                        bytes[row * 50..row * 50 + 2].copy_from_slice(
                            &half::f16::from_f32(scale(layer, expert, part, row)).to_le_bytes(),
                        );
                    }
                    file.as_file().write_all_at(
                        &bytes,
                        source.content().tensor_data_offset
                            + info.offset
                            + (expert * MATRIX_BYTES) as u64,
                    )?;
                }
            }
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn reference(layer: usize, expert: usize, input: &[f32]) -> Vec<f32> {
        let project = |part, input: &[f32]| {
            (0..WIDTH)
                .map(|row| {
                    let weight = -0.875 * scale(layer, expert, part, row);
                    input
                        .iter()
                        .map(|&x| f64::from(x) * f64::from(weight))
                        .sum::<f64>() as f32
                })
                .collect::<Vec<_>>()
        };
        let gate = project(0, input);
        let up = project(1, input);
        let active = gate
            .iter()
            .zip(up)
            .map(|(&g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect::<Vec<_>>();
        project(2, &active)
    }

    #[cfg(target_os = "macos")]
    fn expert_output(
        weights: &ExpertTriplet,
        input: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor> {
        let gate = weights.gate().forward(input)?;
        let up = weights.up().forward_decode_rows(input)?;
        weights.down().forward(&(gate.silu()? * up)?)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn deepseek41_cache_real_metal_queued_matmuls_eviction_and_physical_plateau() -> Result<()> {
        if crate::test_in_subprocess(concat!(
            module_path!(),
            "::deepseek41_cache_real_metal_queued_matmuls_eviction_and_physical_plateau"
        )) {
            return Ok(());
        }
        use candle_core::Tensor;
        let device =
            crate::test_metal_device().context("expert cache qualification requires Metal")?;
        let file = banks(2, 4, WIDTH, WIDTH);
        fill_nonzero(&file, source(&file)?.as_ref())?;
        // Capture a fresh immutable source only after writing all fixture data.
        let source = source(&file)?;
        let production = ExpertCache::new(Arc::clone(&source), device.clone())?;
        assert_eq!(production.budget, DEEPSEEK41_EXPERT_CACHE_BYTES);
        drop(production);
        let mut cache = ExpertCache::with_limit(source, device.clone(), 2 * TRIPLET_CHARGE, false);
        let host = (0..WIDTH)
            .map(|i| (i % 11) as f32 / 16.0 - 0.1875)
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(host.clone(), (1, WIDTH), &device)?;
        let mut allocations = Vec::new();
        let mut comparisons = 0;
        for cycle in 0..24 {
            let layer = cycle % 2;
            let a = cycle % 4;
            let b = (a + 1) % 4;
            let selected = cache.prepare(layer, &[a, b, a])?;
            // F32 route sum, explicitly ordered. No indexed IQ1_S dispatcher.
            let first = expert_output(selected.get(a).unwrap(), &input)?.affine(0.25, 0.0)?;
            let second = expert_output(selected.get(b).unwrap(), &input)?.affine(0.75, 0.0)?;
            let queued = (&first + &second)?;
            drop((first, second));
            // Cache hits must preserve the queued work's pinned packed buffers,
            // even though zero-retirement preparation no longer waits/prunes.
            let hit = cache.prepare(layer, &[a, b])?;
            assert!(Arc::ptr_eq(selected.get(a).unwrap(), hit.get(a).unwrap()));
            let before = cache.stats();
            assert!(cache.prepare(layer, &[(a + 2) % 4]).is_err());
            assert_eq!(cache.stats(), before);
            if cycle % 2 == 0 {
                device.synchronize()?; // Preserve explicit-fence coverage.
            } else {
                // Like the next router readback: no explicit fence, and keep
                // both old leases alive until to_vec1 actually completes.
                let router = input.affine(2.0, 1.0)?.flatten_all()?.to_vec1::<f32>()?;
                assert_eq!(
                    router,
                    host.iter().map(|x| x * 2.0 + 1.0).collect::<Vec<_>>()
                );
            }
            drop((hit, selected));
            // Force both old triplets out BEFORE user readback of queued.
            let replacement = cache.prepare(layer, &[(a + 2) % 4, (a + 3) % 4])?;
            let actual = queued.flatten_all()?.to_vec1::<f32>()?;
            let ra = reference(layer, a, &host);
            let rb = reference(layer, b, &host);
            for ((actual, ra), rb) in actual.iter().zip(ra).zip(rb) {
                let expected = ra * 0.25 + rb * 0.75;
                assert!(
                    (actual - expected).abs() <= 2e-5 * expected.abs().max(1.0),
                    "cycle={cycle} actual={actual} expected={expected}"
                );
                comparisons += 1;
            }
            drop((replacement, queued));
            device.synchronize()?;
            assert_ledger(&cache);
            assert_eq!(cache.stats().charged_bytes, 2 * TRIPLET_CHARGE);
            allocations.push(device.as_metal_device()?.device().current_allocated_size());
        }
        let steady = &allocations[8..];
        assert_eq!(
            steady.iter().min(),
            steady.iter().max(),
            "physical Metal allocations must plateau, not just entry counts: {allocations:?}"
        );
        assert_eq!(
            cache.stats(),
            CacheStats {
                charged_bytes: 2 * TRIPLET_CHARGE,
                entries: 2,
                in_flight_bytes: 0,
                retired_bytes: 0,
                hits: 48,
                misses: 96,
                evictions: 94,
            }
        );
        eprintln!(
            "CACHE_REAL_METAL used=true cycles=24 comparisons={comparisons} allocated={allocations:?} ledger={:?}",
            cache.stats()
        );
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn deepseek41_cache_real_metal_partial_upload_cleanup() -> Result<()> {
        if crate::test_in_subprocess(concat!(
            module_path!(),
            "::deepseek41_cache_real_metal_partial_upload_cleanup"
        )) {
            return Ok(());
        }
        let device =
            crate::test_metal_device().context("expert cache qualification requires Metal")?;
        let file = banks(1, 4, WIDTH, WIDTH);
        let mut cache =
            ExpertCache::with_limit(source(&file)?, device.clone(), 2 * TRIPLET_CHARGE, false);
        device.synchronize()?;
        let before = device.as_metal_device()?.device().current_allocated_size();
        for fail_at in 0..6 {
            let mut calls = 0;
            assert!(
                cache
                    .prepare_with(0, &[0, 1], |source, name, expert, device| {
                        let index = calls;
                        calls += 1;
                        if index == fail_at {
                            anyhow::bail!("injected read failure after {fail_at} uploads");
                        }
                        source.read_expert_slice(name, expert, device)
                    })
                    .is_err()
            );
            assert_eq!(cache.stats().entries, 0);
            assert_eq!(cache.stats().charged_bytes, 0);
            assert_ledger(&cache);
            let after = device.as_metal_device()?.device().current_allocated_size();
            assert_eq!(
                after, before,
                "failed triplet leaked pooled Metal buffers at read {fail_at}"
            );
        }
        eprintln!(
            "CACHE_REAL_METAL_FAILURES used=true failure_points=6 retained_bytes=0 ledger={:?}",
            cache.stats()
        );
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn deepseek41_cache_real_metal_production_buffers_fit_charged_reservation() -> Result<()> {
        if crate::test_in_subprocess(concat!(
            module_path!(),
            "::deepseek41_cache_real_metal_production_buffers_fit_charged_reservation"
        )) {
            return Ok(());
        }
        let device =
            crate::test_metal_device().context("expert cache qualification requires Metal")?;
        let file = banks(1, 384, 2304, 5120);
        let mut cache = ExpertCache::with_limit(source(&file)?, device.clone(), 6_930_432, true);
        device.synchronize()?;
        let before = device.as_metal_device()?.device().current_allocated_size();
        let selected = cache.prepare(0, &[0])?;
        device.synchronize()?;
        let allocated = device.as_metal_device()?.device().current_allocated_size() - before;
        assert_eq!(cache.stats().charged_bytes, 6_930_432);
        assert!(
            allocated as u64 <= cache.stats().charged_bytes,
            "production buffers exceed charged reservation: allocated={allocated} ledger={:?}",
            cache.stats()
        );
        drop((selected, cache));
        device.synchronize()?;
        assert_eq!(
            device.as_metal_device()?.device().current_allocated_size(),
            before
        );
        Ok(())
    }
}
