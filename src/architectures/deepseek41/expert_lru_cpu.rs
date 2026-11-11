//! CPU-only differential checks against the former oldest-first Vec transaction.
use super::*;
use crate::deepseek4_gguf::fixtures;
use std::{collections::HashMap, fs::File};
use tempfile::NamedTempFile;

const UNIT: u64 = 3 * ALLOCATION_GRANULE;

fn fixture(geometry: &[(usize, usize, usize)]) -> NamedTempFile {
    let mut specs = Vec::new();
    let mut offset = 0;
    for (layer, &(experts, ffn, hidden)) in geometry.iter().enumerate() {
        for part in ["gate", "up", "down"] {
            let shape = if part == "down" {
                vec![experts, hidden, ffn]
            } else {
                vec![experts, ffn, hidden]
            };
            specs.push((
                format!("blk.{layer}.ffn_{part}_exps.weight"),
                shape,
                19,
                offset,
            ));
            offset += DeepSeek4DType::IQ1S
                .byte_len(experts * ffn * hidden)
                .unwrap() as u64;
        }
    }
    let specs = specs
        .iter()
        .map(|(n, s, d, o)| (n.as_str(), s.as_slice(), *d, *o))
        .collect::<Vec<_>>();
    fixtures::sparse(&HashMap::new(), &specs).0
}

#[derive(Default)]
struct Reference {
    entries: Vec<(ExpertKey, u64)>,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl Reference {
    fn prepare(
        &mut self,
        layer: usize,
        ids: &[usize],
        geometry: &[(usize, usize, usize)],
        budget: u64,
        leased: &[ExpertKey],
        fail: Option<usize>,
    ) -> (bool, Vec<ExpertKey>, Vec<(String, usize)>) {
        let mut reads = Vec::new();
        let Some(&(count, ffn, hidden)) = geometry.get(layer) else {
            return (false, vec![], reads);
        };
        let mut keys = Vec::new();
        for &expert in ids {
            if expert >= count {
                return (false, vec![], reads);
            }
            let key = ExpertKey { layer, expert };
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
        let charge = 3 * rounded_charge(ffn * hidden / 256 * 50).unwrap();
        let missing = keys
            .iter()
            .filter(|key| !self.entries.iter().any(|(k, _)| k == *key))
            .count();
        let pinned: u64 = self
            .entries
            .iter()
            .filter(|(k, _)| keys.contains(k) || leased.contains(k))
            .map(|(_, c)| c)
            .sum();
        if pinned + missing as u64 * charge > budget {
            return (false, vec![], reads);
        }
        while self.entries.iter().map(|(_, c)| c).sum::<u64>() + missing as u64 * charge > budget {
            let index = self
                .entries
                .iter()
                .position(|(k, _)| !keys.contains(k) && !leased.contains(k))
                .unwrap();
            self.entries.remove(index);
            self.evictions += 1;
        }
        self.hits += (keys.len() - missing) as u64;
        self.misses += missing as u64;
        let mut staged = Vec::new();
        for &key in &keys {
            if self.entries.iter().any(|(k, _)| *k == key) {
                continue;
            }
            for part in ["gate", "up", "down"] {
                reads.push((format!("blk.{layer}.ffn_{part}_exps.weight"), key.expert));
                if fail == Some(reads.len() - 1) {
                    return (false, vec![], reads);
                }
            }
            staged.push((key, charge));
        }
        self.entries.extend(staged);
        for key in &keys {
            let index = self.entries.iter().position(|(k, _)| k == key).unwrap();
            let entry = self.entries.remove(index);
            self.entries.push(entry);
        }
        (true, keys, reads)
    }
}

struct Harness {
    cache: ExpertCache,
    reference: Reference,
    geometry: Vec<(usize, usize, usize)>,
    leases: Vec<(ExpertKey, Arc<ExpertTriplet>)>,
    _file: NamedTempFile,
}

impl Harness {
    fn new(geometry: Vec<(usize, usize, usize)>, budget: u64) -> Result<Self> {
        let file = fixture(&geometry);
        let source = Arc::new(RawGgufSource::new(File::open(file.path())?)?);
        Ok(Self {
            cache: ExpertCache::with_limit(source, Device::Cpu, budget, false),
            reference: Reference::default(),
            geometry,
            leases: vec![],
            _file: file,
        })
    }

    fn step(
        &mut self,
        layer: usize,
        ids: &[usize],
        fail: Option<usize>,
        retain: Option<usize>,
    ) -> Result<()> {
        let leased = self.leases.iter().map(|(k, _)| *k).collect::<Vec<_>>();
        let before = self.cache.stats();
        let ref_before = (
            self.reference.hits,
            self.reference.misses,
            self.reference.evictions,
        );
        let old = self
            .cache
            .entries
            .iter()
            .map(|e| (e.key, Arc::downgrade(&e.weights)))
            .collect::<Vec<_>>();
        let (ok, selected, expected_reads) =
            self.reference
                .prepare(layer, ids, &self.geometry, self.cache.budget, &leased, fail);
        let mut reads = Vec::new();
        let result = self
            .cache
            .prepare_with(layer, ids, |source, name, expert, device| {
                // Victims' owning Arcs must already be gone BEFORE the first read,
                // not retained in a temporary collection through staging/uploads.
                for (key, weak) in &old {
                    if !self.reference.entries.iter().any(|(k, _)| k == key) {
                        assert!(weak.upgrade().is_none(), "victim retained through I/O");
                    }
                }
                reads.push((name.to_owned(), expert));
                if fail == Some(reads.len() - 1) {
                    anyhow::bail!("injected matrix read failure");
                }
                source.read_expert_slice(name, expert, device)
            });
        assert_eq!(result.is_ok(), ok, "layer={layer} ids={ids:?}");
        assert_eq!(reads, expected_reads);
        if let Ok(result) = result {
            assert_eq!(result.iter().map(|(k, _)| *k).collect::<Vec<_>>(), selected);
            for (key, weights) in result.iter() {
                assert!(Arc::ptr_eq(result.get(key.expert).unwrap(), weights));
                let (_, ffn, hidden) = self.geometry[key.layer];
                assert_eq!(
                    (weights.gate().out_dim(), weights.gate().in_dim()),
                    (ffn, hidden)
                );
                assert_eq!(
                    (weights.down().out_dim(), weights.down().in_dim()),
                    (hidden, ffn)
                );
                if retain == Some(key.expert) {
                    self.leases.push((*key, Arc::clone(weights)));
                }
            }
        }
        let actual = self
            .cache
            .entries
            .iter()
            .map(|e| (e.key, e.charge))
            .collect::<Vec<_>>();
        assert_eq!(actual, self.reference.entries, "oldest-to-newest mismatch");
        for (key, weak) in old {
            if let Some(entry) = self.cache.entries.iter().find(|e| e.key == key) {
                assert_eq!(
                    weak.as_ptr(),
                    Arc::as_ptr(&entry.weights),
                    "resident lease replaced"
                );
            } else {
                assert!(weak.upgrade().is_none(), "evicted owner retained");
            }
        }
        self.cache
            .entries
            .assert_integrity(self.cache.budget, false);
        let stats = self.cache.stats();
        assert_eq!(
            (
                stats.hits - before.hits,
                stats.misses - before.misses,
                stats.evictions - before.evictions
            ),
            (
                self.reference.hits - ref_before.0,
                self.reference.misses - ref_before.1,
                self.reference.evictions - ref_before.2
            )
        );
        assert_eq!(stats.entries, actual.len());
        assert_eq!(
            stats.charged_bytes,
            actual.iter().map(|(_, c)| c).sum::<u64>()
        );
        assert_eq!((stats.in_flight_bytes, stats.retired_bytes), (0, 0));
        Ok(())
    }
}

#[test]
fn expert_lru_cpu_reference_varied_traces() -> Result<()> {
    for seed in 1..=12u64 {
        let mut h = Harness::new(vec![(7, 256, 256), (5, 512, 256), (6, 256, 768)], 8 * UNIT)?;
        h.step(0, &[0, 1, 2], None, None)?;
        h.step(0, &[0, 0], None, None)?;
        h.step(0, &[3, 1, 3], None, Some(1))?;
        let mut state = seed;
        for turn in 0..64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let layer = (state >> 32) as usize % 3;
            let a = (state >> 16) as usize % h.geometry[layer].0;
            let b = (a + 1) % h.geometry[layer].0;
            h.step(
                layer,
                &[a, b, a],
                (turn % 11 == 0).then_some(turn % 6),
                None,
            )?;
            if turn == 23 {
                h.leases.clear();
            }
        }
    }
    Ok(())
}

#[test]
fn expert_lru_cpu_reference_pinning_refusal_failure_recovery() -> Result<()> {
    for fail in 0..9 {
        let mut h = Harness::new(vec![(6, 256, 256), (4, 256, 256)], 3 * UNIT)?;
        h.step(0, &[0, 1, 2], None, None)?;
        h.step(0, &[3, 0], None, Some(0))?;
        h.step(1, &[0], None, Some(0))?;
        h.step(1, &[1, 2], None, None)?; // Pinned external union refuses before I/O.
        h.leases.clear();
        h.step(0, &[1, 2, 4], Some(fail), None)?;
        h.step(0, &[1, 2, 4], None, None)?;
        h.step(0, &[5, 1, 5], None, None)?;
        h.step(0, &[usize::MAX], None, None)?;
        h.step(usize::MAX, &[0], None, None)?;
    }
    let mut small = Harness::new(vec![(4, 256, 256)], UNIT - 1)?;
    small.step(0, &[0], None, None)?;
    Ok(())
}

#[test]
fn expert_lru_cpu_reference_complete_union_and_duplicates() -> Result<()> {
    let mut h = Harness::new(vec![(384, 256, 256)], 383 * UNIT)?;
    let all = (0..384).collect::<Vec<_>>();
    h.step(0, &all, None, None)?;
    h.cache.budget += UNIT;
    h.step(0, &all, None, None)?;
    h.step(0, &[7; 4096], None, None)?;
    h.step(0, &[383, 0, 383, 7], None, None)?;
    Ok(())
}

#[test]
fn expert_lru_cpu_stable_slots_reuse_and_dense_boundaries() -> Result<()> {
    let mut geometry = vec![(1, 256, 256); 39];
    geometry.push((384, 256, 256));
    let mut h = Harness::new(geometry, 2 * UNIT)?;
    h.step(39, &[383], None, None)?;
    let key = ExpertKey {
        layer: 39,
        expert: 383,
    };
    let stable = h.cache.entries.lookup(key);
    h.step(0, &[0], None, None)?;
    h.step(39, &[383, 383], None, None)?;
    assert_eq!(
        h.cache.entries.lookup(key),
        stable,
        "touch moved a stable slot"
    );
    h.step(1, &[0], None, None)?;
    let capacity = h.cache.entries.slots.capacity();
    for layer in 2..39 {
        h.step(layer, &[0], None, None)?;
        assert_eq!(
            h.cache.entries.slots.capacity(),
            capacity,
            "free slot not reused"
        );
    }
    h.step(40, &[0], None, None)?;
    h.step(usize::MAX, &[0], None, None)?;
    h.step(39, &[384], None, None)?;
    h.step(39, &[usize::MAX], None, None)?;
    h.step(39, &[], None, None)?;
    Ok(())
}

#[test]
fn expert_lru_cpu_requested_hit_survives_every_staging_failure() -> Result<()> {
    for fail in 0..6 {
        let mut h = Harness::new(vec![(5, 256, 256)], 3 * UNIT)?;
        h.step(0, &[0, 1], None, None)?;
        // The oldest hit follows a miss; failure must not touch it or publish
        // an earlier complete staged triplet.
        h.step(0, &[2, 0, 3], Some(fail), None)?;
        h.step(0, &[2, 0, 3], None, None)?;
    }
    Ok(())
}

#[test]
fn expert_lru_cpu_changed_slice_rejects_entire_staging() -> Result<()> {
    for fail in 0..6 {
        let mut h = Harness::new(vec![(4, 256, 256), (4, 512, 256)], 2 * UNIT)?;
        h.step(0, &[0], None, None)?;
        let mut calls = 0;
        let result = h
            .cache
            .prepare_with(0, &[1, 2], |source, name, expert, device| {
                let index = calls;
                calls += 1;
                if index == fail {
                    // A successful read with unexpected geometry is not a fallback.
                    return source.read_expert_slice("blk.1.ffn_gate_exps.weight", expert, device);
                }
                source.read_expert_slice(name, expert, device)
            });
        assert!(result.is_err());
        assert_eq!(calls, fail + 1);
        assert_eq!(
            h.cache.stats(),
            CacheStats {
                charged_bytes: 0,
                entries: 0,
                in_flight_bytes: 0,
                retired_bytes: 0,
                hits: 0,
                misses: 3,
                evictions: 1,
            }
        );
        h.cache.entries.assert_integrity(h.cache.budget, false);
        let recovered = h.cache.prepare(0, &[1, 2])?;
        assert_eq!(
            recovered.iter().map(|(k, _)| k.expert).collect::<Vec<_>>(),
            [1, 2]
        );
    }
    Ok(())
}

#[test]
fn expert_lru_cpu_identity_failure_keeps_diagnostics_and_ledger() -> Result<()> {
    if crate::test_in_subprocess(concat!(
        module_path!(),
        "::expert_lru_cpu_identity_failure_keeps_diagnostics_and_ledger"
    )) {
        return Ok(());
    }
    use super::super::profile::{self, Detail};
    let mut h = Harness::new(vec![(3, 256, 256)], UNIT)?;
    let _guard = profile::HostGuard::enable();
    h.step(0, &[0], None, None)?;
    profile::reset();
    h._file
        .as_file()
        .set_len(h._file.as_file().metadata()?.len() - 1)?;
    let error = h
        .cache
        .prepare(0, &[1])
        .err()
        .context("source identity change accepted")?;
    assert!(error.to_string().contains("identity"));
    let snapshot = profile::host_snapshot();
    assert_eq!(snapshot.cache_events, [0, 1, 1]);
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
    assert_eq!(
        snapshot.details[Detail::ExpertRetirementWait as usize].calls,
        2
    );
    assert_eq!(
        snapshot.details[Detail::ExpertBookkeeping as usize].calls,
        2
    );
    assert_eq!(h.cache.stats().charged_bytes, 0);
    assert_eq!(
        (
            h.cache.stats().in_flight_bytes,
            h.cache.stats().retired_bytes
        ),
        (0, 0)
    );
    h.cache.entries.assert_integrity(h.cache.budget, false);
    Ok(())
}

#[test]
fn expert_lru_cpu_metadata_bounds_cover_synthetic_geometry() -> Result<()> {
    use std::mem::size_of;
    let direct_slot = size_of::<Slot>()
        + size_of::<ExpertTriplet>()
        + 2 * size_of::<usize>()
        + 3 * (size_of::<RawTensor>() + 4 * size_of::<usize>());
    assert!(direct_slot as u64 * 4 <= EXPERT_HOST_SLOT_BYTES);
    let scratch = MAX_SELECTED
        * (size_of::<ExpertKey>()
            + size_of::<(ExpertKey, Arc<ExpertTriplet>)>()
            + size_of::<Entry>())
        + size_of::<[bool; MAX_SELECTED]>();
    let fixed = KEY_CAPACITY * size_of::<usize>() + size_of::<ExpertLru>() + scratch;
    assert!((fixed as u64) < MODEL_HOST_FIXED_BYTES);
    // No weights or uploads: fill only the actual reserved vacant-slot metadata
    // to its finite synthetic ceiling, using the same pre-I/O reservation path.
    for budget in [0, UNIT - 1, UNIT, 4 * UNIT, u64::MAX] {
        let limit = (budget / UNIT)
            .saturating_add(MAX_SELECTED as u64)
            .min(KEY_CAPACITY as u64) as usize;
        let mut lru = ExpertLru::new();
        lru.reserve(limit, budget, false)?;
        lru.assert_integrity(budget, false);
        assert_eq!(lru.slots.capacity(), limit);
        assert_eq!(lru.index.capacity(), KEY_CAPACITY);
        assert!(lru.reserve(limit + 1, budget, false).is_err());
        lru.assert_integrity(budget, false);
        assert!(
            (lru.slots.capacity() * direct_slot) as u64 * 4
                <= limit as u64 * EXPERT_HOST_SLOT_BYTES
        );
    }
    Ok(())
}
