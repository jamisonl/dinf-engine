//! Prefill tile planning and opt-in, process-local diagnostics for Maya.
//!
//! Tile planning (`prefill_tile_plan`) and the scalar opt-out are always active.
//! The only environment variable read here is `DINF_DEEPSEEK41_TILED_PREFILL`:
//! setting it to exactly `0` forces scalar (one-row) prefill. It is latched on
//! first use. All profiling and validation modes are default off and are only
//! enabled programmatically.
//!
//! Synchronized region timings add GPU fences and therefore change scheduling;
//! they are not production throughput. HOST mode reuses the same regions without
//! boundary synchronization, so its wall time includes implicit waits rather than
//! pure CPU or GPU kernel time. Use `HostGuard::enable` after load, take
//! `host_snapshot` at the first sample, reset, take another `host_snapshot` at
//! decode end, and call `host_report` only after measurement. The two modes are
//! mutually exclusive; no per-token or per-layer event list is kept. Optional
//! completed command-buffer GPU intervals are a separate, overlapping view.
//!
//! Region notes: attention projection/cache/attend/output is one region; FFN
//! post covers routed/shared combination plus residual/HC expansion; router
//! enqueue is host time, while readback/top-6 includes its own GPU wait.
//! Disabled regions cost only the opt-in branch (no clock, allocation or fence).
//! Callers must serialize enable/reset/report with model execution.

use std::ffi::OsStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::Device;
use serde_json::{Value, json};

use super::experts::CacheStats;

/// The loader/cache's packed expert charge ceiling.
pub const EXPERT_CACHE_BUDGET_BYTES: u64 = crate::deepseek4_gguf::DEEPSEEK41_EXPERT_CACHE_BYTES;

/// Default tile request. When native prefill is active the request is
/// `MAX_PREFILL_TILE`; both adapt down to the fixed reserve near the frontier.
pub const DEFAULT_PREFILL_TILE: usize = 32;
/// Structural cap shared by native activation and bounded tile planning.
pub const MAX_PREFILL_TILE: usize = 128;
/// Authoritative per-request admission charge, independent of diagnostic limits.
pub const FIXED_CACHE_BYTES: usize = super::cache::FIXED_CACHE_BYTES;
/// Authoritative dynamic admission rate, used for ledger checks.
pub const DYNAMIC_BYTES_PER_TOKEN: usize = super::cache::DYNAMIC_BYTES_PER_TOKEN;
static PREFILL_TILE: AtomicU64 = AtomicU64::new(DEFAULT_PREFILL_TILE as u64);
static PREFILL_PLANNER_LIMIT: AtomicU64 = AtomicU64::new(FIXED_CACHE_BYTES as u64);
static SCALAR_PREFILL_OPT_OUT: OnceLock<bool> = OnceLock::new();
static TILE_VALIDATION: AtomicBool = AtomicBool::new(false);

/// Request scalar through MAX_PREFILL_TILE; reject invalid sizes without mutation.
/// Serialize changes between requests. The scalar environment opt-out wins over
/// all operator requests; otherwise effective size may be smaller for memory.
pub fn set_prefill_tile(size: usize) -> Result<()> {
    anyhow::ensure!(
        (1..=MAX_PREFILL_TILE).contains(&size),
        "Maya prefill tile must be 1..={MAX_PREFILL_TILE}"
    );
    PREFILL_TILE.store(size as u64, Ordering::Relaxed);
    Ok(())
}

/// Raw operator request, before the environment policy. Capture this for restore.
pub fn requested_prefill_tile() -> usize {
    PREFILL_TILE.load(Ordering::Relaxed) as usize
}

/// Process-local planner ceiling only; never changes admission or owner bounds.
/// Serialize between requests. A lower ceiling is diagnostic, not legacy policy.
pub fn set_prefill_planner_limit(bytes: usize) -> Result<()> {
    anyhow::ensure!(
        (1..=FIXED_CACHE_BYTES).contains(&bytes),
        "Maya planner limit must be 1..={FIXED_CACHE_BYTES} bytes"
    );
    PREFILL_PLANNER_LIMIT.store(bytes as u64, Ordering::Relaxed);
    Ok(())
}
/// Current planner-only ceiling; actual fixed admission remains unchanged.
pub fn prefill_planner_limit() -> usize {
    PREFILL_PLANNER_LIMIT.load(Ordering::Relaxed) as usize
}

/// Restore request planning/validation controls on every success/error path.
/// Operator must serialize this scope with all native requests.
pub struct PrefillGuard {
    tile: usize,
    limit: usize,
    validation: bool,
}
impl PrefillGuard {
    /// Capture controls before any operator mutation, without resetting evidence.
    pub fn capture() -> Self {
        Self {
            tile: requested_prefill_tile(),
            limit: prefill_planner_limit(),
            validation: prefill_validation_enabled(),
        }
    }
}
impl Drop for PrefillGuard {
    fn drop(&mut self) {
        PREFILL_TILE.store(self.tile as u64, Ordering::Relaxed);
        PREFILL_PLANNER_LIMIT.store(self.limit as u64, Ordering::Relaxed);
        set_prefill_validation_enabled(self.validation);
    }
}

/// Only exact OS-string "0" disables tiling; missing/non-Unicode values do not.
/// Pure policy parser: no process environment mutation or device initialization.
pub fn scalar_prefill_opt_out(value: Option<&OsStr>) -> bool {
    value == Some(OsStr::new("0"))
}

/// Read-only production opt-out, latched on first use; configure before startup.
/// No repeated environment reads/allocations, logs, clocks or fences per forward.
pub fn scalar_prefill_opt_out_enabled() -> bool {
    *SCALAR_PREFILL_OPT_OUT.get_or_init(|| {
        scalar_prefill_opt_out(std::env::var_os("DINF_DEEPSEEK41_TILED_PREFILL").as_deref())
    })
}

/// Environment-effective request, NOT the adaptive effective size of a forward.
/// Exact DINF_DEEPSEEK41_TILED_PREFILL=0 forces scalar even after operator overrides.
pub fn prefill_tile() -> usize {
    if scalar_prefill_opt_out_enabled() {
        1
    } else {
        requested_prefill_tile()
    }
}

/// Static rounded allowances; retained tile tensors enforce these against actual
/// backing lengths before crossing row/layer/interval boundaries. Encoder-held
/// private replacements require an active allowance PER row until router wait;
/// this remains an analytical allowance, NOT a measured hardware peak.
/// Scalar fallback adds no tile scratch; the existing scalar path is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PrefillTilePlan {
    /// Whole-forward snapshot of the bounded process-local control.
    pub requested: usize,
    /// Maximum internal tile size; a smaller final tail is allowed.
    pub effective: usize,
    /// Exclusive absolute end of this whole forward, checked against 1M.
    pub end: usize,
    /// Total fixed-reserve bound for tiling, including raw/pending storage.
    /// Zero means no tiled scratch was admitted, NOT zero scalar memory usage.
    pub scratch_bound_bytes: usize,
    /// Stable explanation of requested, tail-limited, adaptive or scalar mode.
    pub reason: &'static str,
}

/// Existing worst-case raw128 rings and ratio2 pending-pair backing.
pub const TILE_RAW_PENDING_BYTES: usize = 10_498_048;
/// One executing row: gather 1 MiB + raw/gather cat 2 MiB + raw 256 KiB
/// plus q/projection intermediates < 768 KiB at concatenation. During heads,
/// gather is gone: values 2 MiB + scores/probs <= 3 * 256 KiB + q/heads
/// and projections < 1 MiB. FFN's six downs and temporaries peak below this.
/// Private replacement buffers can survive in encoder clones until router wait,
/// so this allowance is multiplied by rows, not charged once per tile. Index
/// pages2048 add2 MiB/row for larger score blocks and immutable tail replacements
/// over the previous4 MiB allowance; this is not persistent index duplication.
pub const TILE_ACTIVE_SCRATCH_BYTES: usize = 6 * 1024 * 1024;
/// Per row, including prior interval owners: two 128 KiB residual backings,
/// two 32 KiB inputs, 128 KiB grouped gate/up, HC/selection/router/host
/// metadata <= 64 KiB. Add128 KiB/row for the bounded joined shared-expert
/// input/output carriers (each <=2*rows*5120*4 bytes). Shared gate/up/mid
/// temporaries remain within the separate active-row allowance.
pub const TILE_ROW_BYTES: usize = 640 * 1024;

/// Independent attention phase carriers, in addition to the existing retained
/// 640KiB and active6MiB. Only the test-only old schedule omits this charge.
fn attention_phase_bytes() -> usize {
    if super::phased_attention_enabled() {
        512 * 1024
    } else {
        0
    }
}

/// Test-only routed-post path: the exposed clamp/clamp/silu/mul [6,2304]
/// carriers each retain a fitting 64 KiB pool bucket and down [6,5120] retains
/// 128 KiB. They survive into the next attention, separate from the row allowance.
#[cfg(test)]
pub(super) const TILE_ROUTED_POST_BYTES: usize = 384 * 1024;
fn routed_post_bytes() -> usize {
    // Six grouped Coder occurrences per row: gathered input, gate/up, SwiGLU
    // and down, plus per-buffer 16KiB rounding and bounded host route metadata.
    // Each rounded carrier is at most twice its payload; 1MiB/row covers both.
    if super::coder_routed_batch_enabled() {
        return 1024 * 1024;
    }
    #[cfg(test)]
    if super::ROUTED_POST_BATCH.with(|v| v.get()) {
        return TILE_ROUTED_POST_BYTES;
    }
    0
}

#[cfg(target_os = "macos")]
fn packet_retained_bytes(rows: usize) -> Option<usize> {
    if super::packet_policy(rows, grouped_iq1_effective()) {
        // An explicit additional host/inline metadata charge; no presumed spare
        // capacity in the existing HC/router metadata allowance.
        Some(
            super::iq1_packets::carrier_bytes(rows).expect("checked bounded packet rows")
                + rows * 16 * 1024,
        )
    } else {
        None
    }
}

fn active_scratch_bytes() -> usize {
    TILE_ACTIVE_SCRATCH_BYTES
}

/// Additional backing envelope for the test-only AttentionV2 path; batched
/// carriers never borrow the per-row allowance. `r` models the bounded Metal
/// bucket geometry. The full admitted tile is charged even for a short final
/// slice. The factor two covers source plus exact replacement and coexistence
/// with the prior complete carrier. This does not permit arbitrary pool loans:
/// AttentionV2 pins reject oversized sources, and scoring/tail-join
/// destinations allocate privately.
/// Dynamic sealed pages are excluded; displaced partial tails are NOT.
pub(super) fn attention_v2_bytes(rows: usize, end: usize) -> usize {
    if !super::attention_v2_enabled() {
        return 0;
    }
    let r = |bytes: usize| bytes.max(1).next_power_of_two();
    let projected = 6 * r(rows * 512 * 4) + 6 * r(rows * 128 * 4);
    let queries = 2 * r(rows * 32 * 128 * 4) + r(rows * 32 * 4);
    // Pool results + per-row latent/key RoPE/quantize + query RoPE/quantize.
    let row_carriers = rows * (2048 + 2 * (2048 + 512) + 2 * 16384);
    let scores = r(rows * end.max(1) * 4);
    // Old/new/cat tail sources for KV256 and index2048, plus pending pair.
    let tails = 3 * (256 * 512 * 4 + 2048 * 128 * 4) + 4 * 2048;
    // Host bounds, positions, page aliases and per-row tensor-owner metadata.
    let metadata = rows * 16 * 1024 + end.div_ceil(256) * 256;
    2 * (projected + queries + row_carriers + scores + tails + metadata)
}

/// Conservative pooled/private bound for 1..=MAX_PREFILL_TILE rows and end <= 1M.
/// Separate Engram admission: current/prior interval GPU input/output loans,
/// host decoded output plus flatten copy, sort plan, 16KiB encoded buffer and
/// an explicit 64KiB runtime/allocator/stack allowance. Never attention credit.
pub(super) fn engram_v2_bytes(rows: usize) -> usize {
    if !super::engram_v2_enabled() {
        return 0;
    }
    let r = |n: usize| n.max(1).next_power_of_two();
    2 * (r(rows * 6144 * 4) + r(rows * 25600 * 4))
        + rows * (2 * 6144 * 4 + 24 * (8 + 4))
        + super::engram::BATCHED_READ_BUFFER_BYTES
        + 64 * 1024
}

/// Production calls only through the checked plan. Dynamic persistent chunks and
/// engine's already-charged checkpoint/copy headroom are not fixed scratch.
pub fn prefill_tile_scratch(rows: usize, end: usize) -> usize {
    // Worst owner20 frontier. Chunk dots/relu/weighted sums, score cat,
    // filtered scores, all topk merge levels and candidate mask are bounded
    // by 32 bytes/rounded visible row PER token, even if queued intermediates
    // survive until the batched router barrier. Never retain q-by-whole-cache.
    let visible = end.max(1).next_power_of_two();
    let prefix_metadata = end.div_ceil(super::cache::CHUNK_ROWS) * 256;
    #[cfg(target_os = "macos")]
    if super::native_prefill_enabled() {
        return TILE_RAW_PENDING_BYTES
            + attention_v2_bytes(rows, end)
            + engram_v2_bytes(rows)
            + rows
                * (active_scratch_bytes()
                    + TILE_ROW_BYTES
                    + attention_phase_bytes()
                    + 32 * visible
                    + prefix_metadata)
            + super::prefill_mm::carrier_bytes(rows);
    }
    #[cfg(target_os = "macos")]
    if let Some(carriers) = packet_retained_bytes(rows) {
        return TILE_RAW_PENDING_BYTES
            + attention_v2_bytes(rows, end)
            + engram_v2_bytes(rows)
            + rows
                * (active_scratch_bytes() + TILE_ROW_BYTES - 128 * 1024
                    + attention_phase_bytes()
                    + 32 * visible
                    + prefix_metadata)
            + carriers;
    }
    TILE_RAW_PENDING_BYTES
        + attention_v2_bytes(rows, end)
        + engram_v2_bytes(rows)
        + rows
            * (active_scratch_bytes()
                + TILE_ROW_BYTES
                + attention_phase_bytes()
                + routed_post_bytes()
                + 32 * visible
                + prefix_metadata)
}

/// Recompute a reported Coder carrier plan after its forward scope has ended.
/// This restores both thread-local selectors before returning, including on unwind.
pub fn coder_prefill_tile_scratch(rows: usize, end: usize) -> usize {
    let _scope = super::CoderBatchScope::enter(true);
    prefill_tile_scratch(rows, end)
}

/// Plan against the whole-forward end (worst visible prefix), then split the
/// input into bounded internal tiles. Reject overflow/context before enqueue.
pub fn prefill_tile_plan(requested: usize, start: usize, count: usize) -> Result<PrefillTilePlan> {
    prefill_tile_plan_with_limit(requested, start, count, prefill_planner_limit())
}

/// Shared geometry limit for attention admission and native dispatch.
pub(super) const ATTENTION_V2_MAX_ROWS: usize = 32;
pub(super) fn effective_prefill_cap() -> usize {
    if super::attention_v2_enabled() {
        ATTENTION_V2_MAX_ROWS
    } else {
        MAX_PREFILL_TILE
    }
}

pub(super) fn prefill_tile_plan_with_limit(
    requested: usize,
    start: usize,
    count: usize,
    limit: usize,
) -> Result<PrefillTilePlan> {
    anyhow::ensure!(
        (1..=FIXED_CACHE_BYTES).contains(&limit),
        "invalid Maya planner limit"
    );
    anyhow::ensure!(
        (1..=MAX_PREFILL_TILE).contains(&requested) && count > 0,
        "invalid Maya prefill tile plan"
    );
    let end = start
        .checked_add(count)
        .context("Maya tile position overflow")?;
    anyhow::ensure!(end <= 1_048_576, "Maya tile context limit exceeded");
    let capped = requested.min(count).min(effective_prefill_cap());
    let mut effective = capped;
    while effective > 1 && prefill_tile_scratch(effective, end) > limit {
        effective -= 1;
    }
    let reason = if requested == 1 {
        "scalar_requested_or_environment_opt_out"
    } else if count == 1 {
        "single_token_decode_or_tail"
    } else if effective == 1 {
        "fixed_reserve_scalar_fallback"
    } else if effective < capped {
        "fixed_reserve_adaptive_tile"
    } else if capped < requested.min(count) {
        "attention_v2_geometry_cap"
    } else if effective < requested {
        "input_tail"
    } else {
        "requested_tile"
    };
    Ok(PrefillTilePlan {
        requested,
        effective,
        end,
        reason,
        scratch_bound_bytes: if effective == 1 {
            0
        } else {
            prefill_tile_scratch(effective, end)
        },
    })
}

/// Bounded request-level evidence. No clocks, device reads or owner retention.
#[derive(Clone, Debug, Default)]
struct TileEvidence {
    forwards: u64,
    tiled_forwards: u64,
    tiled_rows: u64,
    scalar_rows: u64,
    router_readbacks: u64,
    max_effective: usize,
    min_prefill_effective: usize,
    max_scratch_bound_bytes: usize,
    last: Option<PrefillTilePlan>,
    last_prefill: Option<PrefillTilePlan>,
    frontier: Option<Value>,
    owner_samples: u64,
    max_retained_tile_bytes: usize,
    retained_tile_comparison_bound_bytes: usize,
    retained_tile_plan: Option<PrefillTilePlan>,
    retained_tile_coder_batch: bool,
}
static TILE_EVIDENCE: Mutex<TileEvidence> = Mutex::new(TileEvidence {
    forwards: 0,
    tiled_forwards: 0,
    tiled_rows: 0,
    scalar_rows: 0,
    router_readbacks: 0,
    max_effective: 0,
    min_prefill_effective: 0,
    max_scratch_bound_bytes: 0,
    last: None,
    last_prefill: None,
    frontier: None,
    owner_samples: 0,
    max_retained_tile_bytes: 0,
    retained_tile_comparison_bound_bytes: 0,
    retained_tile_plan: None,
    retained_tile_coder_batch: false,
});

/// Opt in to bounded request-level control/frontier evidence, without clocks,
/// extra readbacks or fences. Default OFF; serialize with native requests.
pub fn set_prefill_validation_enabled(enabled: bool) {
    TILE_VALIDATION.store(enabled, Ordering::Relaxed);
}
/// Whether subsequent forwards publish bounded tile/frontier evidence.
pub fn prefill_validation_enabled() -> bool {
    TILE_VALIDATION.load(Ordering::Relaxed)
}
/// Clear request-level evidence between requests, preserving all controls.
pub fn reset_prefill_tile_evidence() {
    *TILE_EVIDENCE.lock().unwrap_or_else(|p| p.into_inner()) = TileEvidence::default();
}
// Test-only actual control snapshot, independent of validation scans and clocks.
#[cfg(test)]
thread_local! {
    pub(super) static TEST_RUNTIME_PLAN: std::cell::Cell<Option<PrefillTilePlan>> = const { std::cell::Cell::new(None) };
}
pub(super) fn record_prefill_plan(plan: PrefillTilePlan, count: usize) {
    #[cfg(test)]
    TEST_RUNTIME_PLAN.with(|v| v.set(Some(plan)));
    if !prefill_validation_enabled() {
        return;
    }
    let mut s = TILE_EVIDENCE.lock().unwrap_or_else(|p| p.into_inner());
    s.forwards += 1;
    s.max_effective = s.max_effective.max(plan.effective);
    if count > 1 {
        s.last_prefill = Some(plan);
        s.min_prefill_effective = if s.min_prefill_effective == 0 {
            plan.effective
        } else {
            s.min_prefill_effective.min(plan.effective)
        };
    }
    if plan.effective > 1 {
        s.tiled_forwards += 1;
        let scalar_tail = u64::from(count % plan.effective == 1);
        s.tiled_rows += count as u64 - scalar_tail;
        s.scalar_rows += scalar_tail;
    } else {
        s.scalar_rows += count as u64;
    }
    s.router_readbacks += (count.div_ceil(plan.effective) * 40) as u64;
    s.max_scratch_bound_bytes = s.max_scratch_bound_bytes.max(plan.scratch_bound_bytes);
    s.last = Some(plan);
}
/// Only tile-retained owners are measured. Raw/pending and per-row active
/// private replacement scratch are separately reserved and are not scanned.
pub(super) fn tile_owner_comparison_bound(plan: PrefillTilePlan) -> Result<usize> {
    anyhow::ensure!(
        plan.effective > 1,
        "Maya owner accounting requires a tiled plan"
    );
    plan.scratch_bound_bytes
        .checked_sub(TILE_RAW_PENDING_BYTES + plan.effective * active_scratch_bytes())
        .context("Maya tile owner bound underflow")
}

pub(super) fn check_tile_owner_bytes(bytes: usize, plan: PrefillTilePlan) -> Result<usize> {
    let bound = tile_owner_comparison_bound(plan)?;
    anyhow::ensure!(
        bytes <= bound,
        "Maya retained tile owners exceed static allowance: {bytes} > {bound}"
    );
    Ok(bound)
}

pub(super) fn record_tile_owners(bytes: usize, plan: PrefillTilePlan) -> Result<()> {
    let bound = check_tile_owner_bytes(bytes, plan)?;
    let mut s = TILE_EVIDENCE.lock().unwrap_or_else(|p| p.into_inner());
    s.owner_samples += 1;
    if bytes >= s.max_retained_tile_bytes {
        s.max_retained_tile_bytes = bytes;
        s.retained_tile_comparison_bound_bytes = bound;
        s.retained_tile_plan = Some(plan);
        s.retained_tile_coder_batch = super::coder_routed_batch_enabled();
    }
    Ok(())
}

pub(super) fn publish_frontier(frontier: Value) {
    if prefill_validation_enabled() {
        TILE_EVIDENCE
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .frontier = Some(frontier);
    }
}
/// Copy bounded planning counters and the latest successful history/frontiers.
/// Router counts are planned, not timing measurements. No device operations.
pub fn prefill_tile_evidence() -> Value {
    let s = TILE_EVIDENCE.lock().unwrap_or_else(|p| p.into_inner());
    json!({"requested": prefill_tile(), "raw_requested": requested_prefill_tile(),
        "default_tile": DEFAULT_PREFILL_TILE,
        "max_tile": MAX_PREFILL_TILE, "planner_limit_bytes": prefill_planner_limit(),
        "scalar_environment_opt_out": scalar_prefill_opt_out_enabled(), "forwards": s.forwards,
        "tiled_forwards": s.tiled_forwards, "tiled_rows": s.tiled_rows,
        "scalar_rows": s.scalar_rows, "planned_router_readbacks": s.router_readbacks,
        "max_effective": s.max_effective, "min_prefill_effective": s.min_prefill_effective,
        "max_scratch_bound_bytes": s.max_scratch_bound_bytes,
        "retained_owner_samples": s.owner_samples,
        "max_retained_tile_bytes": s.max_retained_tile_bytes,
        "retained_tile_comparison_bound_bytes": s.retained_tile_comparison_bound_bytes,
        "retained_tile_plan": s.retained_tile_plan.map(|p| json!({"coder_batch": s.retained_tile_coder_batch,
            "effective": p.effective, "end": p.end, "scratch_bound_bytes": p.scratch_bound_bytes })),
        "fixed_raw_pending_reserve_bytes": TILE_RAW_PENDING_BYTES,
        "max_retained_tile_plus_fixed_reserve_bytes": if s.owner_samples == 0 { 0 } else {
            s.max_retained_tile_bytes + TILE_RAW_PENDING_BYTES },
        "retained_tile_plus_fixed_comparison_bound_bytes": if s.owner_samples == 0 { 0 } else {
            s.retained_tile_comparison_bound_bytes + TILE_RAW_PENDING_BYTES },
        "active_scratch_allowance_bytes_per_row_unmeasured": active_scratch_bytes(),
        "retained_measurement_scope": "deduplicated actual tile row/HC/Selection and prior/current interval input/grouped backing; excludes per-row active/private replacement temporaries, packed model/expert weights and dynamic KV; not hardware high-water",
        "fixed_request_reserve_bytes": super::cache::FIXED_CACHE_BYTES,
        "last_forward": s.last.map(|p| json!({"requested": p.requested, "effective": p.effective,
            "reason": p.reason, "end": p.end, "scratch_bound_bytes": p.scratch_bound_bytes})),
        "last_prefill": s.last_prefill.map(|p| json!({"requested": p.requested, "effective": p.effective,
            "reason": p.reason, "end": p.end, "scratch_bound_bytes": p.scratch_bound_bytes})),
        "frontier": s.frontier})
}
#[cfg(target_os = "macos")]
use candle_metal_kernels::metal;

/// Fixed-size completed-drain observation; never reads a device or waits.
/// Unsupported platforms retain raw zero counters but report elapsed as null.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuTimingSnapshot {
    /// Platform has the Metal timestamp observer API, not proof of valid hardware times.
    pub supported: bool,
    /// Raw process-global observer opt-in state.
    pub enabled: bool,
    /// Successfully drained buffers observed in this window.
    pub drained_buffers: u64,
    /// Buffers with finite, positive, ordered, representable timestamp intervals.
    pub valid_timestamp_buffers: u64,
    /// Buffers with finite nonnegative endpoints, at least one zero/missing.
    pub zero_timestamp_buffers: u64,
    /// Buffers with other malformed or unrepresentable timestamp intervals.
    pub invalid_timestamp_buffers: u64,
    /// Saturating sum of valid command-buffer elapsed intervals, not GPU busy time.
    pub gpu_elapsed_ns: u64,
}

/// Copy raw atomics only; operator serializes this with requests and resets.
pub fn gpu_timing_snapshot() -> GpuTimingSnapshot {
    #[cfg(target_os = "macos")]
    {
        let s = metal::command_buffer_timing_snapshot();
        GpuTimingSnapshot {
            supported: true,
            enabled: s.enabled,
            drained_buffers: s.drained_buffers,
            valid_timestamp_buffers: s.valid_timestamp_buffers,
            zero_timestamp_buffers: s.zero_timestamp_buffers,
            invalid_timestamp_buffers: s.invalid_timestamp_buffers,
            gpu_elapsed_ns: s.gpu_elapsed_ns,
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        GpuTimingSnapshot::default()
    }
}

/// Reset observer counters without changing opt-in; no-op on unsupported platforms.
pub fn reset_gpu_timing() {
    #[cfg(target_os = "macos")]
    metal::reset_command_buffer_timing();
}

/// Diagnose ONLY, after the load-ending completion wait. No device operations.
/// Operator serializes requests; restore the prior opt-in even on errors.
pub struct GpuTimingGuard {
    prior: bool,
}
impl GpuTimingGuard {
    /// Enable/reset after load, retaining the prior opt-in for restoration.
    pub fn enable() -> Self {
        let prior = gpu_timing_snapshot().enabled;
        #[cfg(target_os = "macos")]
        metal::set_command_buffer_timing_enabled(true);
        reset_gpu_timing();
        Self { prior }
    }
}
impl Drop for GpuTimingGuard {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        metal::set_command_buffer_timing_enabled(self.prior);
        #[cfg(not(target_os = "macos"))]
        let _ = self.prior;
    }
}

/// Format only after timing. A valid zero-duration interval is measured zero;
/// disabled, unsupported, missing and invalid-only windows are NOT measured zero.
pub fn gpu_timing_report(s: &GpuTimingSnapshot) -> Value {
    let measured = s.supported && s.enabled && s.valid_timestamp_buffers > 0;
    let counts_saturated = [
        s.drained_buffers,
        s.valid_timestamp_buffers,
        s.zero_timestamp_buffers,
        s.invalid_timestamp_buffers,
    ]
    .contains(&u64::MAX);
    let saturated = counts_saturated || s.gpu_elapsed_ns == u64::MAX;
    let classified = u128::from(s.valid_timestamp_buffers)
        + u128::from(s.zero_timestamp_buffers)
        + u128::from(s.invalid_timestamp_buffers);
    let status = if !s.supported {
        "unsupported_platform"
    } else if !s.enabled {
        "disabled"
    } else if s.drained_buffers == 0 {
        "no_completed_drains_observed"
    } else if !measured {
        "no_valid_timestamps"
    } else if s.zero_timestamp_buffers > 0 || s.invalid_timestamp_buffers > 0 {
        "partial_valid_timestamps"
    } else {
        "valid_timestamps"
    };
    json!({
        "supported": s.supported, "enabled": s.enabled, "status": status,
        "drained_buffers": s.drained_buffers,
        "valid_timestamp_buffers": s.valid_timestamp_buffers,
        "zero_timestamp_buffers": s.zero_timestamp_buffers,
        "invalid_timestamp_buffers": s.invalid_timestamp_buffers,
        "raw_gpu_elapsed_ns": s.gpu_elapsed_ns,
        "elapsed_ns": measured.then_some(s.gpu_elapsed_ns),
        "elapsed_ms": measured.then_some(s.gpu_elapsed_ns as f64 / 1_000_000.),
        "counters_saturated": saturated,
        "drained_equals_valid_plus_zero_plus_invalid": (!counts_saturated)
            .then_some(u128::from(s.drained_buffers) == classified),
        "view": "OVERLAPPING completed COMMAND-BUFFER GPU elapsed intervals; NOT an additive host-wall region",
        "interpretation": "GPUStartTime/GPUEndTime exclude CPU enqueue and pre-start queue pending; may include GPU fence/resource waits, scheduling/preemption and overlapping intervals; NOT active-kernel/busy time or per-component attribution; never add to or blindly subtract from host wall",
        "window": "process-global successful drains observed between reset and raw snapshot, across all devices; pending buffers are not queried; failed drains contribute nothing; intervals are assigned by drain observation, not clipped to wall boundaries",
        "validity": "zero/missing endpoints stay in zero count; nonfinite/negative/reversed/unrepresentable times stay invalid; equal positive endpoints are valid zero duration; only valid intervals accumulate; each counter saturates at u64::MAX",
    })
}

#[derive(Clone, Copy)]
pub(super) enum Region {
    Embedding,
    Engram,
    AttentionPrepare,
    Attention,
    AttentionPost,
    FfnPrepare,
    RouterProjectionEnqueue,
    RouterReadbackTop6,
    ExpertCachePrepare,
    SharedExpert,
    RoutedExperts,
    FfnPost,
    OutputHead,
    FinalFence,
}

const NAMES: [&str; 14] = [
    "embedding",
    "engram",
    "attention_prepare",
    "attention_project_cache_attend_output",
    "attention_post",
    "ffn_prepare",
    "router_projection_enqueue",
    "router_readback_cpu_top6",
    "expert_cache_prepare",
    "shared_expert",
    "routed_experts",
    "ffn_post",
    "output_head",
    "final_fence",
];

struct Counter {
    ns: AtomicU64,
    calls: AtomicU64,
}
impl Counter {
    const fn new() -> Self {
        Self {
            ns: AtomicU64::new(0),
            calls: AtomicU64::new(0),
        }
    }
    fn record(&self, ns: u64) {
        self.ns.fetch_add(ns, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
    }
}

static ENABLED: AtomicBool = AtomicBool::new(false);
static CACHE_ENABLED: AtomicBool = AtomicBool::new(false);
// Bounded same-load native token/logit replay passed; not universal certification.
static GROUPED_IQ1: AtomicBool = AtomicBool::new(true);
static CACHE: Mutex<CacheStats> = Mutex::new(CacheStats {
    charged_bytes: 0,
    entries: 0,
    in_flight_bytes: 0,
    retired_bytes: 0,
    hits: 0,
    misses: 0,
    evictions: 0,
});
static COUNTERS: [Counter; NAMES.len()] = [const { Counter::new() }; NAMES.len()];

static HOST_ENABLED: AtomicBool = AtomicBool::new(false);

/// Process-local HOST elapsed wall diagnostics, not CPU or GPU kernel time.
/// Operator MUST serialize enable/reset/snapshot/restore between requests.
/// Never change modes during a request. No extra fences or readbacks.
pub fn set_host_enabled(enabled: bool) {
    if enabled {
        ENABLED.store(false, Ordering::Relaxed);
    }
    HOST_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether subsequent serialized forwards collect unfenced HOST wall counters.
#[inline]
pub fn host_enabled() -> bool {
    HOST_ENABLED.load(Ordering::Relaxed)
}

/// Serial operator scope: restore all diagnostic controls on success/error.
/// Reset AFTER loading so loader/SHA traffic cannot enter inference counters.
pub struct HostGuard {
    host: bool,
    sync: bool,
    cache: bool,
}
impl HostGuard {
    /// Capture controls without enabling clocks or changing counters.
    pub fn capture() -> Self {
        Self {
            host: host_enabled(),
            sync: ENABLED.load(Ordering::Relaxed),
            cache: CACHE_ENABLED.load(Ordering::Relaxed),
        }
    }
    /// Capture controls, reset counters and enable HOST mode (not synchronized
    /// mode). Call only after loading and before the single diagnostic request.
    pub fn enable() -> Self {
        let guard = Self::capture();
        set_enabled(false);
        set_cache_diagnostics_enabled(true);
        reset();
        set_host_enabled(true);
        guard
    }
}
impl Drop for HostGuard {
    fn drop(&mut self) {
        set_host_enabled(self.host);
        set_enabled(self.sync);
        set_cache_diagnostics_enabled(self.cache);
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Detail {
    EngramMetadataPre,
    EngramMetadataPost,
    EngramRead,
    EngramDecode,
    EngramProjection,
    RouterReadback,
    RouterTop6,
    ExpertBookkeeping,
    ExpertRetirementWait,
    ExpertMetadataPre,
    ExpertMetadataPost,
    ExpertRead,
    ExpertUpload,
}
const DETAIL_NAMES: [&str; 13] = [
    "engram_metadata_pre",
    "engram_metadata_post",
    "engram_row_read",
    "engram_scalar_decode",
    "engram_projection_enqueue",
    "router_readback",
    "router_cpu_top6",
    "expert_bookkeeping_lookup_lru",
    "expert_existing_retirement_wait",
    "expert_metadata_pre",
    "expert_metadata_post",
    "expert_selected_payload_read",
    "expert_upload_allocation_copy",
];
const DETAIL_PARENTS: [&str; 13] = [
    "engram",
    "engram",
    "engram",
    "engram",
    "engram",
    "router_readback_cpu_top6",
    "router_readback_cpu_top6",
    "expert_cache_prepare",
    "expert_cache_prepare",
    "expert_cache_prepare",
    "expert_cache_prepare",
    "expert_cache_prepare",
    "expert_cache_prepare",
];
static DETAILS: [Counter; 13] = [const { Counter::new() }; 13];
static REQUESTED_BYTES: [AtomicU64; 13] = [const { AtomicU64::new(0) }; 13];
static COMPLETED_BYTES: [AtomicU64; 13] = [const { AtomicU64::new(0) }; 13];
static CACHE_EVENTS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Drop records attempted spans, including errors. Completed bytes mean a whole
/// logical read returned success, even if a later identity/decode check fails.
pub(crate) struct DetailStamp {
    started: Instant,
    detail: Detail,
}
impl Drop for DetailStamp {
    fn drop(&mut self) {
        DETAILS[self.detail as usize].record(elapsed_ns(self.started));
    }
}
#[inline]
pub(crate) fn detail(detail: Detail) -> Option<DetailStamp> {
    host_enabled().then(|| DetailStamp {
        started: Instant::now(),
        detail,
    })
}
#[inline]
pub(crate) fn read_detail(detail: Detail, bytes: usize) -> Option<DetailStamp> {
    let stamp = self::detail(detail);
    if stamp.is_some() {
        REQUESTED_BYTES[detail as usize].fetch_add(bytes as u64, Ordering::Relaxed);
    }
    stamp
}
#[inline]
pub(crate) fn read_completed(stamp: &Option<DetailStamp>, bytes: usize) {
    if let Some(stamp) = stamp {
        COMPLETED_BYTES[stamp.detail as usize].fetch_add(bytes as u64, Ordering::Relaxed);
    }
}
#[inline]
pub(crate) fn cache_events(hits: u64, misses: u64, evictions: u64) {
    if host_enabled() {
        for (counter, value) in CACHE_EVENTS.iter().zip([hits, misses, evictions]) {
            counter.fetch_add(value, Ordering::Relaxed);
        }
    }
}
fn elapsed_ns(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

/// One fixed counter, using elapsed HOST nanoseconds and logical I/O bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Measurement {
    /// Accumulated HOST elapsed nanoseconds, including existing implicit waits.
    pub ns: u64,
    /// Completed/dropped span count, including attempted spans returning errors.
    pub calls: u64,
    /// Bytes requested at actual logical read calls; not physical SSD traffic.
    pub requested_bytes: u64,
    /// Bytes from whole logical reads that succeeded, even if later checks fail.
    pub completed_bytes: u64,
}
/// Bounded raw snapshot: no strings, JSON, allocation, device access or fence.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostSnapshot {
    /// Fourteen disjoint top-level regions in host_report's named order.
    pub regions: [Measurement; 14],
    /// Thirteen nested subsets in host_report's named order; never additive.
    pub details: [Measurement; 13],
    /// Window-local hits, misses, evictions; separate from cumulative CacheStats.
    pub cache_events: [u64; 3],
}
/// Copy bounded raw counters at a serial measurement boundary, without reset.
pub fn host_snapshot() -> HostSnapshot {
    let measurement = |c: &Counter| Measurement {
        ns: c.ns.load(Ordering::Relaxed),
        calls: c.calls.load(Ordering::Relaxed),
        ..Measurement::default()
    };
    HostSnapshot {
        regions: std::array::from_fn(|i| measurement(&COUNTERS[i])),
        details: std::array::from_fn(|i| Measurement {
            requested_bytes: REQUESTED_BYTES[i].load(Ordering::Relaxed),
            completed_bytes: COMPLETED_BYTES[i].load(Ordering::Relaxed),
            ..measurement(&DETAILS[i])
        }),
        cache_events: std::array::from_fn(|i| CACHE_EVENTS[i].load(Ordering::Relaxed)),
    }
}
impl HostSnapshot {
    /// Disjoint top-level regions ONLY. Nested details must never be added.
    pub fn top_level_ns(&self) -> u64 {
        self.regions.iter().map(|m| m.ns).sum()
    }
    /// Signed wall minus disjoint region sum; negative values expose inconsistent
    /// boundaries rather than hiding them with saturation.
    pub fn remainder_ns(&self, wall_ns: u64) -> i128 {
        i128::from(wall_ns) - i128::from(self.top_level_ns())
    }
}
/// Construct reports only after timing has finished.
pub fn host_report(snapshot: &HostSnapshot, wall_ns: u64) -> Value {
    let rows = |names: &[&str], counters: &[Measurement], parents: Option<&[&str]>| {
        names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let m = counters[i];
                json!({"region": name, "host_elapsed_ns": m.ns, "calls": m.calls,
                "logical_requested_bytes": m.requested_bytes,
                "successful_completed_bytes": m.completed_bytes,
                "subset_of": parents.map(|p| p[i])})
            })
            .collect::<Vec<_>>()
    };
    json!({
        "timing": "HOST elapsed wall; may include implicit GPU/filesystem waits, not pure CPU or GPU kernel time",
        "wall_ns": wall_ns, "top_level_sum_ns": snapshot.top_level_ns(),
        "unaccounted_ns": snapshot.remainder_ns(wall_ns),
        "regions": rows(&NAMES, &snapshot.regions, None),
        "nested_subsets_not_additive": rows(&DETAIL_NAMES, &snapshot.details, Some(&DETAIL_PARENTS)),
        "expert_cache_local": cache_local_report(snapshot),
        "cache_events": {"hits": snapshot.cache_events[0], "misses": snapshot.cache_events[1], "evictions": snapshot.cache_events[2]},
        "router_readback": "waits for previously enqueued GPU work plus copy; not solely router compute",
        "io": "logical calls/bytes, NOT physical SSD traffic; read wall includes CPU/filesystem waiting and may overlap GPU",
        "upload": "allocation/copy/enqueue HOST wall, NOT GPU kernel time",
        "gpu_hardware_timing": "separate optional completed-command-buffer OVERLAPPING view; never additive to host regions",
        "fences": "only existing native waits/final fence; no diagnostic boundary synchronization",
    })
}

fn cache_local_report(snapshot: &HostSnapshot) -> Value {
    let parent = snapshot.regions[Region::ExpertCachePrepare as usize];
    let bookkeeping = snapshot.details[Detail::ExpertBookkeeping as usize];
    let excludes = [
        Detail::ExpertMetadataPre,
        Detail::ExpertMetadataPost,
        Detail::ExpertRead,
        Detail::ExpertUpload,
        Detail::ExpertRetirementWait,
    ];
    let excluded_ns: i128 = excludes
        .iter()
        .map(|d| i128::from(snapshot.details[*d as usize].ns))
        .sum();
    let remainder = i128::from(parent.ns) - excluded_ns;
    let subparts: Vec<_> = excludes
        .iter()
        .map(|d| {
            let i = *d as usize;
            let m = snapshot.details[i];
            json!({"region": DETAIL_NAMES[i], "host_elapsed_ns": m.ns, "calls": m.calls,
            "logical_requested_bytes": m.requested_bytes,
            "successful_completed_bytes": m.completed_bytes,
            "subset_of": "expert_cache_prepare"})
        })
        .collect();
    json!({
        "parent": {"region": "expert_cache_prepare", "host_elapsed_ns": parent.ns, "calls": parent.calls},
        "measured_bookkeeping_subset": {"host_elapsed_ns": bookkeeping.ns, "calls": bookkeeping.calls,
            "scope": "ExpertBookkeeping spans only; NOT all lookup/LRU/validation/staging work; excludes some entries.iter().any in the load loop and staging validation"},
        "measured_disjoint_subparts": subparts,
        "disjoint_excluded_sum_ns": excluded_ns,
        "derived_parent_minus_io_upload_retirement_ns": remainder,
        "derived_remainder_consistent": remainder >= i128::from(bookkeeping.ns),
        "derivation": "ExpertCachePrepare parent minus disjoint metadata-pre + metadata-post + selected-payload-read + allocation/copy/upload + existing-retirement-wait spans; read span excludes both metadata spans, so each is subtracted exactly once",
        "remainder_scope": "DERIVED, NOT a measured LRU lookup timer; includes measured bookkeeping plus other lookup/LRU/validation/staging/counter overhead and span gaps; negative or smaller-than-bookkeeping reconciliation is flagged, never clamped",
        "accounting": "all subparts and bookkeeping are nested parent SUBSETS, never added to main-region sum; cache parent is NOT all disk time",
        "io_upload": "logical read bytes are NOT physical SSD traffic; Tensor::from_vec HOST time is allocation/copy/upload work, NOT isolated GPU execution",
    })
}

/// Enable synchronized timings only for a separate diagnostic request.
pub fn set_enabled(enabled: bool) {
    if enabled {
        set_host_enabled(false);
    }
    ENABLED.store(enabled, Ordering::Relaxed);
}

/// Enable bounded cache snapshots independently of synchronized timing.
/// Snapshots are published under the existing whole-forward cache lock.
pub fn set_cache_diagnostics_enabled(enabled: bool) {
    CACHE_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Select grouped routed gate/up for subsequent whole forwards (default ON).
/// Bounded same-load native token/logit replay passed; not universal certification.
/// Operator MUST serialize changes between requests; no weight/cache reset,
/// environment mutation or fence.
/// DINF_DEEPSEEK4_SIMD opt-out always overrides this requested mode.
pub fn set_grouped_iq1_enabled(enabled: bool) {
    GROUPED_IQ1.store(enabled, Ordering::Relaxed);
}

/// Requested process-local mode; operator serializes access between requests.
pub fn grouped_iq1_enabled() -> bool {
    GROUPED_IQ1.load(Ordering::Relaxed)
}

/// Existing native SIMD policy, including DINF_DEEPSEEK4_SIMD opt-out.
pub fn grouped_iq1_simd_enabled() -> bool {
    super::Linear::iq1_simd_enabled()
}

/// Effective mode, snapshotted once per whole forward.
pub fn grouped_iq1_effective() -> bool {
    grouped_iq1_enabled() && grouped_iq1_simd_enabled()
}

pub(super) fn publish_cache(stats: impl FnOnce() -> CacheStats) {
    if CACHE_ENABLED.load(Ordering::Relaxed) {
        *CACHE.lock().unwrap_or_else(|e| e.into_inner()) = stats();
    }
}

/// Latest completed native forward's cumulative model-global cache counters.
/// Valid after an opted-in forward; no weight or device access occurs here.
pub fn cache_snapshot() -> Value {
    let s = *CACHE.lock().unwrap_or_else(|e| e.into_inner());
    json!({
        "charged_bytes": s.charged_bytes, "entries": s.entries,
        "in_flight_bytes": s.in_flight_bytes, "retired_bytes": s.retired_bytes,
        "hits": s.hits, "misses": s.misses, "evictions": s.evictions,
    })
}

pub(super) struct Stamp {
    started: Instant,
    region: Region,
    synchronized: bool,
}

/// For router enqueue/readback and the final fence: no extra leading fence.
#[inline]
pub(super) fn start_unfenced(region: Region) -> Option<Stamp> {
    (ENABLED.load(Ordering::Relaxed) || host_enabled()).then(|| Stamp {
        started: Instant::now(),
        region,
        synchronized: false,
    })
}

#[inline]
pub(super) fn finish_unfenced(stamp: Option<Stamp>) {
    if let Some(stamp) = stamp {
        COUNTERS[stamp.region as usize]
            .record(stamp.started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
    }
}

#[inline]
pub(super) fn start(device: &Device, region: Region) -> Result<Option<Stamp>> {
    if ENABLED.load(Ordering::Relaxed) {
        device.synchronize()?;
        return Ok(Some(Stamp {
            started: Instant::now(),
            region,
            synchronized: true,
        }));
    }
    Ok(host_enabled().then(|| Stamp {
        started: Instant::now(),
        region,
        synchronized: false,
    }))
}

#[inline]
pub(super) fn finish(device: &Device, stamp: Option<Stamp>) -> Result<()> {
    if let Some(stamp) = stamp {
        if stamp.synchronized {
            device.synchronize()?;
        }
        finish_unfenced(Some(stamp));
    }
    Ok(())
}

/// Reset top-level/nested timings, logical I/O and diagnostic cache-event counts.
/// The model's CacheStats and published cumulative cache snapshot are unchanged.
pub fn reset() {
    for c in &COUNTERS {
        c.ns.store(0, Ordering::Relaxed);
        c.calls.store(0, Ordering::Relaxed);
    }
    for c in &DETAILS {
        c.ns.store(0, Ordering::Relaxed);
        c.calls.store(0, Ordering::Relaxed);
    }
    for c in REQUESTED_BYTES
        .iter()
        .chain(&COMPLETED_BYTES)
        .chain(&CACHE_EVENTS)
    {
        c.store(0, Ordering::Relaxed);
    }
}

/// Return JSON after measurement, without printing in the execution path.
pub fn report() -> Value {
    let regions: Vec<Value> = NAMES
        .iter()
        .zip(&COUNTERS)
        .filter_map(|(name, c)| {
            let calls = c.calls.load(Ordering::Relaxed);
            (calls != 0).then(|| {
                json!({
                    "region": name, "calls": calls,
                    "total_ms": c.ns.load(Ordering::Relaxed) as f64 / 1_000_000.,
                })
            })
        })
        .collect();
    json!({
        "enabled": ENABLED.load(Ordering::Relaxed), "regions": regions,
        "prefill_tile_control": {
            "default_tile": DEFAULT_PREFILL_TILE, "raw_requested": requested_prefill_tile(),
            "requested": prefill_tile(), "scalar_environment_opt_out": scalar_prefill_opt_out_enabled(),
        },
        "warning": "synchronized diagnostic timings alter scheduling, not production throughput",
        "scope": "whole request including story prefill; scalar native regions; attention project/cache/attend/output combined",
        "router_projection": "host enqueue only; GPU completion is included in unfenced router readback/CPU top6 timing",
        "timing": "region boundary synchronization overhead is not additive production latency",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    #[test]
    fn coder_batch_admission_uses_its_own_carriers() -> Result<()> {
        let _attention = super::super::AttentionV2Guard::set(false);
        let _coder = super::super::CoderBatchScope::enter(true);
        assert!(super::super::coder_routed_batch_enabled());
        assert!(!super::super::packet_policy(128, true));
        #[cfg(target_os = "macos")]
        assert!(packet_retained_bytes(128).is_none());
        assert_eq!(routed_post_bytes(), 1024 * 1024);
        for (start, count) in [(0, 108), (1024, 128), (1_048_448, 128)] {
            let plan = prefill_tile_plan_with_limit(128, start, count, FIXED_CACHE_BYTES)?;
            assert!(plan.effective > 1 && plan.scratch_bound_bytes <= FIXED_CACHE_BYTES);
            assert!(tile_owner_comparison_bound(plan)? >= plan.effective * 1024 * 1024);
        }
        Ok(())
    }

    #[test]
    fn attention_v2_requested128_shared_admission_cap() -> Result<()> {
        let _packet = super::super::Iq1PacketGuard::set(true);
        let _engram = super::super::EngramV2Guard::set(false);
        {
            let _attention = super::super::AttentionV2Guard::set(true);
            for start in [0, 64852, 1_048_448] {
                let plan = prefill_tile_plan_with_limit(128, start, 128, FIXED_CACHE_BYTES)?;
                assert!(plan.effective <= ATTENTION_V2_MAX_ROWS);
                assert_eq!(
                    plan.effective,
                    prefill_tile_plan_with_limit(32, start, 128, FIXED_CACHE_BYTES)?.effective
                );
            }
            assert_eq!(
                prefill_tile_plan_with_limit(128, 0, 128, FIXED_CACHE_BYTES)?.reason,
                "attention_v2_geometry_cap"
            );
        }
        let _attention = super::super::AttentionV2Guard::set(false);
        assert_eq!(
            prefill_tile_plan_with_limit(128, 0, 128, FIXED_CACHE_BYTES)?.effective,
            128
        );
        assert_eq!(
            prefill_tile_plan_with_limit(128, 64852, 128, FIXED_CACHE_BYTES)?.effective,
            128
        );
        Ok(())
    }

    #[test]
    fn host_diag_gpu_validity_and_unavailable_windows() {
        let mut s = GpuTimingSnapshot::default();
        let r = gpu_timing_report(&s);
        assert_eq!(r["status"], "unsupported_platform");
        assert!(r["elapsed_ns"].is_null());
        s.supported = true;
        assert_eq!(gpu_timing_report(&s)["status"], "disabled");
        s.enabled = true;
        assert_eq!(
            gpu_timing_report(&s)["status"],
            "no_completed_drains_observed"
        );
        s.drained_buffers = 3;
        s.zero_timestamp_buffers = 2;
        s.invalid_timestamp_buffers = 1;
        let r = gpu_timing_report(&s);
        assert_eq!(r["status"], "no_valid_timestamps");
        assert!(r["elapsed_ns"].is_null());
        assert!(r["elapsed_ms"].is_null());
        assert_eq!(r["drained_equals_valid_plus_zero_plus_invalid"], true);
        s.drained_buffers += 1;
        s.valid_timestamp_buffers = 1;
        // Equal positive endpoints produce a valid measured zero, not missing.
        assert_eq!(gpu_timing_report(&s)["elapsed_ns"], 0);
        s.gpu_elapsed_ns = 2_500_000;
        let r = gpu_timing_report(&s);
        assert_eq!(r["elapsed_ms"], 2.5);
        assert_eq!(r["status"], "partial_valid_timestamps");
        assert_eq!(r["zero_timestamp_buffers"], 2);
        assert_eq!(r["invalid_timestamp_buffers"], 1);
        s.gpu_elapsed_ns = u64::MAX;
        let r = gpu_timing_report(&s);
        assert_eq!(r["counters_saturated"], true);
        assert_eq!(r["drained_equals_valid_plus_zero_plus_invalid"], true);
        s.drained_buffers += 1;
        assert_eq!(
            gpu_timing_report(&s)["drained_equals_valid_plus_zero_plus_invalid"],
            false
        );
        s.drained_buffers = u64::MAX;
        assert!(gpu_timing_report(&s)["drained_equals_valid_plus_zero_plus_invalid"].is_null());
        s.enabled = false;
        assert!(gpu_timing_report(&s)["elapsed_ns"].is_null());
    }

    #[test]
    fn host_diag_gpu_snapshot_reset_disabled_and_raii() -> Result<()> {
        if crate::test_in_subprocess(concat!(
            module_path!(),
            "::host_diag_gpu_snapshot_reset_disabled_and_raii"
        )) {
            return Ok(());
        }
        #[cfg(target_os = "macos")]
        let original = gpu_timing_snapshot().enabled;
        for prior in [false, true] {
            #[cfg(target_os = "macos")]
            metal::set_command_buffer_timing_enabled(prior);
            for fail in [false, true] {
                let result = (|| -> Result<()> {
                    let _guard = GpuTimingGuard::enable();
                    let first = gpu_timing_snapshot();
                    assert_eq!(first.enabled, cfg!(target_os = "macos"));
                    assert_eq!(first.drained_buffers, 0);
                    assert_eq!(first.gpu_elapsed_ns, 0);
                    reset_gpu_timing();
                    let decode = gpu_timing_snapshot();
                    assert_eq!(decode, first);
                    if fail {
                        anyhow::bail!("CPU-only restore error");
                    }
                    Ok(())
                })();
                assert_eq!(result.is_err(), fail);
                assert_eq!(
                    gpu_timing_snapshot().enabled,
                    cfg!(target_os = "macos") && prior
                );
            }
        }
        #[cfg(target_os = "macos")]
        metal::set_command_buffer_timing_enabled(original);
        Ok(())
    }

    #[test]
    fn host_diag_cache_local_derived_remainder_excludes_disjoint_subparts_once() {
        let mut s = HostSnapshot::default();
        s.regions[Region::ExpertCachePrepare as usize].ns = 1_000;
        s.details[Detail::ExpertBookkeeping as usize].ns = 200;
        for (d, ns) in [
            (Detail::ExpertMetadataPre, 10),
            (Detail::ExpertMetadataPost, 20),
            (Detail::ExpertRead, 300),
            (Detail::ExpertUpload, 100),
            (Detail::ExpertRetirementWait, 50),
        ] {
            s.details[d as usize].ns = ns;
        }
        let r = host_report(&s, 1_200);
        assert_eq!(r["top_level_sum_ns"], 1_000);
        assert_eq!(r["unaccounted_ns"], 200);
        let c = &r["expert_cache_local"];
        assert_eq!(c["measured_bookkeeping_subset"]["host_elapsed_ns"], 200);
        assert_eq!(c["disjoint_excluded_sum_ns"], 480);
        // Includes measured bookkeeping: subtracting it again would give 320.
        assert_eq!(c["derived_parent_minus_io_upload_retirement_ns"], 520);
        assert_eq!(c["derived_remainder_consistent"], true);
        s.regions[Region::ExpertCachePrepare as usize].ns = 600;
        assert_eq!(
            cache_local_report(&s)["derived_remainder_consistent"],
            false
        );
        s.regions[Region::ExpertCachePrepare as usize].ns = 400;
        let c = cache_local_report(&s);
        assert_eq!(c["derived_parent_minus_io_upload_retirement_ns"], -80);
        assert_eq!(c["derived_remainder_consistent"], false);
    }

    #[test]
    fn host_diag_modes_windows_reconciliation_and_restore() -> Result<()> {
        if crate::test_in_subprocess(concat!(
            module_path!(),
            "::host_diag_modes_windows_reconciliation_and_restore"
        )) {
            return Ok(());
        }
        let _restore = HostGuard::capture();
        set_enabled(false);
        set_host_enabled(false);
        set_cache_diagnostics_enabled(false);
        reset();
        assert!(start(&Device::Cpu, Region::Embedding)?.is_none());
        assert!(detail(Detail::EngramRead).is_none());
        assert_eq!(host_snapshot().top_level_ns(), 0);
        set_enabled(true);
        {
            let _guard = HostGuard::enable();
            assert!(host_enabled());
            assert!(!ENABLED.load(Ordering::Relaxed));
            let stamp = start(&Device::Cpu, Region::Embedding)?.unwrap();
            assert!(!stamp.synchronized);
            finish(&Device::Cpu, Some(stamp))?;
            // Deterministic accounting: the detail is a subset, not additive.
            reset();
            COUNTERS[Region::Engram as usize].record(30);
            DETAILS[Detail::EngramRead as usize].record(20);
            let first = host_snapshot();
            assert_eq!(first.top_level_ns(), 30);
            assert_eq!(first.remainder_ns(100), 70);
            assert_eq!(first.remainder_ns(10), -20);
            reset();
            COUNTERS[Region::FfnPost as usize].record(7);
            COUNTERS[Region::FfnPost as usize].record(11);
            let decode = host_snapshot();
            assert_eq!(decode.top_level_ns(), 18);
            assert_eq!(decode.regions[Region::FfnPost as usize].calls, 2);
            assert_eq!(decode.details[Detail::EngramRead as usize].calls, 0);
            assert_eq!(first.top_level_ns(), 30); // Detached raw first window.
        }
        assert!(!host_enabled());
        assert!(ENABLED.load(Ordering::Relaxed));
        assert!(!CACHE_ENABLED.load(Ordering::Relaxed));
        let failed = (|| -> Result<()> {
            let _guard = HostGuard::enable();
            anyhow::bail!("deliberate CPU diagnostic error")
        })();
        assert!(failed.is_err());
        assert!(!host_enabled());
        assert!(ENABLED.load(Ordering::Relaxed));
        set_enabled(false);
        reset();
        assert!(read_detail(Detail::ExpertRead, 123).is_none());
        cache_events(1, 2, 3);
        assert_eq!(host_snapshot().cache_events, [0; 3]);
        assert_eq!(
            host_snapshot().details[Detail::ExpertRead as usize],
            Measurement::default()
        );
        Ok(())
    }

    #[test]
    fn diagnostic_cpu_tensor_smoke_and_disable() -> Result<()> {
        if crate::test_in_subprocess(concat!(
            module_path!(),
            "::diagnostic_cpu_tensor_smoke_and_disable"
        )) {
            return Ok(());
        }
        set_enabled(false);
        reset();
        assert!(start(&Device::Cpu, Region::Embedding)?.is_none());
        set_enabled(true);
        let stamp = start(&Device::Cpu, Region::Embedding)?;
        let x = Tensor::new(&[2f32, 3.], &Device::Cpu)?;
        assert_eq!(x.sqr()?.to_vec1::<f32>()?, vec![4., 9.]);
        finish(&Device::Cpu, stamp)?;
        set_enabled(false);
        let report = report();
        assert_eq!(report["regions"][0]["region"], "embedding");
        assert_eq!(report["regions"][0]["calls"], 1);
        assert_eq!(report["enabled"], false);
        reset();
        assert_eq!(self::report()["regions"], json!([]));
        Ok(())
    }
}
