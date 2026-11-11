//! Opt-in synchronized region timings. These waits change GPU scheduling;
//! use them to find expensive regions, not to claim production throughput.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::Result;
use candle_core::Device;

#[derive(Clone, Copy)]
pub(super) enum Region {
    AttentionPrepare,
    AttentionProject,
    AttentionRows,
    AttentionOutput,
    AttentionPost,
    FfnPrepare,
    Router,
    RoutedExperts,
    SharedExperts,
    FfnPost,
    Output,
}

const NAMES: [&str; 11] = [
    "attention_prepare",
    "attention_projections",
    "attention_rows",
    "attention_output",
    "attention_post",
    "ffn_prepare",
    "router",
    "routed_experts",
    "shared_experts",
    "ffn_post",
    "output_head",
];

struct Counter {
    ns: AtomicU64,
    calls: AtomicU64,
    rows: AtomicU64,
}

impl Counter {
    const fn new() -> Self {
        Self {
            ns: AtomicU64::new(0),
            calls: AtomicU64::new(0),
            rows: AtomicU64::new(0),
        }
    }

    fn record(&self, ns: u64, rows: usize) {
        self.ns.fetch_add(ns, Ordering::Relaxed);
        self.rows.fetch_add(rows as u64, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
    }

    fn reset(&self) {
        self.ns.store(0, Ordering::Relaxed);
        self.calls.store(0, Ordering::Relaxed);
        self.rows.store(0, Ordering::Relaxed);
    }
}

static ENABLED: OnceLock<bool> = OnceLock::new();
static SINGLE: [Counter; NAMES.len()] = [const { Counter::new() }; NAMES.len()];
static MULTI: [Counter; NAMES.len()] = [const { Counter::new() }; NAMES.len()];

pub(super) struct Stamp {
    started: Instant,
    counter: &'static Counter,
    rows: usize,
}

pub(super) fn start(device: &Device, rows: usize, region: Region) -> Result<Option<Stamp>> {
    if !*ENABLED.get_or_init(|| std::env::var_os("DINF_PROFILE_LAYERS").is_some()) {
        return Ok(None);
    }
    device.synchronize()?;
    let counters = if rows == 1 { &SINGLE } else { &MULTI };
    Ok(Some(Stamp {
        started: Instant::now(),
        counter: &counters[region as usize],
        rows,
    }))
}

pub(super) fn finish(device: &Device, stamp: Option<Stamp>) -> Result<()> {
    if let Some(stamp) = stamp {
        device.synchronize()?;
        stamp.counter.record(
            stamp.started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            stamp.rows,
        );
    }
    Ok(())
}

/// Reset process-local V4 timing counters. Only intended for diagnostic runs.
pub fn reset_layer_profile() {
    for counter in SINGLE.iter().chain(&MULTI) {
        counter.reset();
    }
}

/// Print completed regions, separated by single-row and multi-row calls.
pub fn report_layer_profile() {
    for (kind, counters) in [("single_row", &SINGLE), ("multi_row", &MULTI)] {
        for (name, counter) in NAMES.iter().zip(counters) {
            let calls = counter.calls.load(Ordering::Relaxed);
            if calls == 0 {
                continue;
            }
            let ms = counter.ns.load(Ordering::Relaxed) as f64 / 1_000_000.;
            eprintln!(
                "DINF_V4_PROFILE kind={kind} region={name} calls={calls} rows={} total_ms={ms:.4} ms_per_call={:.4}",
                counter.rows.load(Ordering::Relaxed),
                ms / calls as f64
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_record_and_reset_without_global_test_state() {
        let counter = Counter::new();
        counter.record(100, 32);
        counter.record(150, 1);
        assert_eq!(counter.ns.load(Ordering::Relaxed), 250);
        assert_eq!(counter.calls.load(Ordering::Relaxed), 2);
        assert_eq!(counter.rows.load(Ordering::Relaxed), 33);
        counter.reset();
        assert_eq!(counter.ns.load(Ordering::Relaxed), 0);
        assert_eq!(counter.calls.load(Ordering::Relaxed), 0);
        assert_eq!(counter.rows.load(Ordering::Relaxed), 0);
    }
}
