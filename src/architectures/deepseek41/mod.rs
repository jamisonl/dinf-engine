//! Native Candle/Metal text graph for DeepSeek V4.1 Flash ("Maya").
//!
//! Decode runs one token at a time. Prefill is tiled: the default request is
//! `profile::DEFAULT_PREFILL_TILE` rows (`profile::MAX_PREFILL_TILE` when the
//! native prefill matmul is active), and `profile::prefill_tile_plan` shrinks the
//! tile near the context frontier so retained scratch stays within the fixed
//! per-request reserve. Packed weights are never expanded into vocabulary- or
//! expert-bank-sized F32 tensors.
//!
//! Parts of this module are derived from audreyt/ds4
//! (commit 0dff39f91721bcf3472813c6d2bebf9353df2a1a); see the adjacent `NOTICE`
//! and `LICENSE-MIT` for attribution and license terms.
//!
//! # Tile ownership
//!
//! Metal pool loans can be larger than the requested power-of-two bucket. Every
//! tensor that outlives a row or layer inside a tile (row residuals, HC splits,
//! attention selections, router inputs and grouped expert outputs) is passed
//! through `cache::retain_bounded`, which keeps buffers whose backing fits
//! `fixed_state_tensor_bytes` and replaces oversized ones with an exact private
//! copy. The copy preserves values, layout and BF16 rounding points. Owners are
//! released only at the next successful router readback or at the final/error
//! GPU fence, so no extra synchronization is introduced.
//!
//! When prefill validation is enabled (`profile::set_prefill_validation_enabled`),
//! the deduplicated bytes of live tile owners are checked against the plan's
//! scratch bound minus the fixed raw/pending reserve and the active-scratch
//! allowance. Dynamic KV pages and packed weights are charged by their own
//! ledgers and are not counted again. Encoder-internal temporaries are not
//! measured, so this check is not a hardware high-water mark.
//!
//! # Attention phasing
//!
//! Tiled prefill batches the QA/KV/QB and output-B projections across rows
//! while keeping explicit BF16 rounding, last-dimension RMS norm, row-causal
//! raw/index cache publication and the eight-group output-A projection. Retained
//! attention owners cost 640 KiB + 512 KiB per row, separate from the 6 MiB
//! active scratch allowance.
//!
//! # Routed experts
//!
//! On Metal, routed IQ1_S experts for a tile are dispatched as packets
//! (`deepseek4_qmatmul::iq1_packets`) that group route occurrences sharing an
//! expert. The planner charges T*(640+512-128) KiB plus the packet carriers and
//! 16 KiB of metadata per row. Scalar decode and the shared/grouped/SIMD
//! opt-outs keep the per-row path. The Coder variant groups inputs that share a
//! packed expert while keeping each projection's scalar summation order.
//!
//! # Test-only paths
//!
//! `AttentionV2Guard` batches compressor/index projections and multi-query page
//! scoring; `EngramV2Guard` batches the 24-row Engram table reads and one
//! `[T, 6144]` projection per Engram layer and tile. Both are disabled in
//! release builds and exist so their numerics and owner accounting can be
//! tested against the default path. Their additional memory envelopes are
//! `profile::attention_v2_bytes` and `profile::engram_v2_bytes`.

mod attention;
mod cache;
pub mod engram;
pub mod experts;
pub mod math;
mod ops;
pub mod profile;
mod route_combine;
mod tile_attention;
mod weights;

pub use cache::DeepSeek41Cache;
pub use cache::{DYNAMIC_BYTES_PER_TOKEN, FIXED_CACHE_BYTES};

use super::deepseek4_qmatmul::DeepSeek4QMatMul;
use super::{LayerCache, ModelArchitecture, StageInput, StageOutput};
use crate::deepseek4_gguf::RawTensor;
use crate::gguf_loader::{GgufMetadata, LayerRange, StageTensors};
use crate::turboquant::TurboQuantConfig;
use anyhow::{Context, Result, ensure};
use attention::{Attention, Selection};
use candle_core::{DType, Device, Tensor};
use experts::{ExpertCache, SelectedExperts};
use math::RMS_EPS;
use profile::Region;
use std::sync::{Arc, Mutex};
use weights::Weights;

type Linear = DeepSeek4QMatMul;

/// One router interval, never a growing per-layer/request owner list. Packed
/// leases remain charged in the unchanged 72 GiB ledger. Scalar x/grouped
/// gate-up owners retain <=160 KiB pooled backing; tiled owners are bounded by
/// the adaptive fixed-reserve plan. Narrow views share the retained backing.
/// Retire only at the EXISTING next successful router readback or final/error
/// fence. No cached model-bank copy and no extra lifetime synchronization.
#[derive(Default)]
struct MoeLeases {
    selected: Option<SelectedExperts>,
    input: Option<Tensor>,
    gate_up: Option<Tensor>,
    tile_inputs: Vec<Tensor>,
    tile_gate_up: Vec<Option<Tensor>>,
    shared_input: Option<Tensor>,
    shared_output: Option<Tensor>,
    attention: AttentionLeases,
    routed_post: RoutedPostLeases,
}

/// Packet carriers survive the same successful router readback or final/error
/// fence as selected packed weights, including across a short final tile.
#[derive(Default)]
struct RoutedPostLeases {
    tensors: Vec<Tensor>,
}

impl RoutedPostLeases {
    fn tensors(&self) -> impl Iterator<Item = &Tensor> {
        self.tensors.iter()
    }

    fn pin(&mut self, tensor: Tensor) -> Result<Tensor> {
        // Own the enqueued result BEFORE a possibly failing backing correction.
        self.tensors.push(tensor);
        let tensor = cache::retain_bounded(self.tensors.last().unwrap().clone())?;
        *self.tensors.last_mut().unwrap() = tensor.clone();
        Ok(tensor)
    }
}

#[cfg(test)]
thread_local! {
    static ROUTED_POST_BATCH: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Test-only opt-in, nested/panic-safe and local to this thread.
#[cfg(test)]
struct RoutedPostBatchGuard(bool);
#[cfg(test)]
impl RoutedPostBatchGuard {
    fn set(enabled: bool) -> Self {
        Self(ROUTED_POST_BATCH.with(|v| v.replace(enabled)))
    }
}
#[cfg(test)]
impl Drop for RoutedPostBatchGuard {
    fn drop(&mut self) {
        ROUTED_POST_BATCH.with(|v| v.set(self.0));
    }
}

// Test-only override that disables packet dispatch. Release always uses it.
#[cfg(test)]
thread_local! {
    static IQ1_PACKET_BATCH: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}
#[cfg(test)]
struct Iq1PacketGuard(bool);
#[cfg(test)]
impl Iq1PacketGuard {
    fn set(enabled: bool) -> Self {
        Self(IQ1_PACKET_BATCH.with(|v| v.replace(enabled)))
    }
}
#[cfg(test)]
impl Drop for Iq1PacketGuard {
    fn drop(&mut self) {
        IQ1_PACKET_BATCH.with(|v| v.set(self.0));
    }
}

/// One selector for packet dispatch and its exact retained-byte charge.
/// Native stages are Metal-only; scalar/decode and shared/grouped/SIMD opt-outs
/// retain the original path. Test overrides never enter release policy.
#[cfg(any(test, target_os = "macos"))]
fn packet_policy(rows: usize, grouped_iq1: bool) -> bool {
    #[cfg(target_os = "macos")]
    {
        #[cfg(test)]
        if !IQ1_PACKET_BATCH.with(|v| v.get()) {
            return false;
        }
        !CODER_MODEL_ACTIVE.with(|v| v.get())
            && rows > 1
            && rows <= profile::MAX_PREFILL_TILE
            && grouped_iq1
            && Linear::iq1_simd_enabled()
            && batched_shared_enabled()
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (rows, grouped_iq1);
        false
    }
}

/// Whole-forward eligibility, shared by dispatch and the scope-based planner.
/// No native-prefill work (including compilation) may precede these opt-outs.
#[cfg(target_os = "macos")]
pub(super) fn native_prefill_policy(
    rows: usize,
    grouped: bool,
    simd: bool,
    phased: bool,
    shared: bool,
    packets: bool,
) -> bool {
    rows > 1 && grouped && simd && phased && shared && packets
}

#[cfg(target_os = "macos")]
fn native_prefill_eligible(rows: usize, grouped: bool) -> bool {
    native_prefill_policy(
        rows,
        grouped,
        Linear::iq1_simd_enabled(),
        phased_attention_enabled(),
        batched_shared_enabled(),
        packet_policy(rows.min(profile::MAX_PREFILL_TILE), grouped),
    )
}

#[cfg(all(test, target_os = "macos"))]
pub(super) fn native_prefill_policy_checks(device: &Device) -> Result<serde_json::Value> {
    struct Restore(bool, bool, bool, bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            ATTENTION_PHASED.with(|v| v.set(self.0));
            SHARED_TILE_REFERENCE.with(|v| v.set(self.1));
            IQ1_PACKET_BATCH.with(|v| v.set(self.2));
            profile::set_grouped_iq1_enabled(self.3);
        }
    }
    let _restore = Restore(
        ATTENTION_PHASED.with(|v| v.replace(true)),
        SHARED_TILE_REFERENCE.with(|v| v.replace(false)),
        IQ1_PACKET_BATCH.with(|v| v.replace(true)),
        profile::grouped_iq1_enabled(),
    );
    profile::set_grouped_iq1_enabled(true);
    let before = prefill_mm::counts();
    let _request = prefill_mm::TestScope::new(Some(true));
    let mut cases = Vec::new();
    for name in [
        "scalar_step",
        "grouped",
        "simd",
        "phased",
        "shared",
        "packet",
    ] {
        profile::set_grouped_iq1_enabled(name != "grouped");
        ATTENTION_PHASED.with(|v| v.set(name != "phased"));
        SHARED_TILE_REFERENCE.with(|v| v.set(name == "shared"));
        IQ1_PACKET_BATCH.with(|v| v.set(name != "packet"));
        let eligible = if name == "simd" {
            native_prefill_policy(128, true, false, true, true, true)
        } else {
            native_prefill_eligible(
                if name == "scalar_step" { 1 } else { 128 },
                profile::grouped_iq1_effective(),
            )
        };
        let _scope = prefill_mm::Scope::enter(device, eligible);
        ensure!(!prefill_mm::active(), "opt-out {name} activated native MM");
        let plan = profile::prefill_tile_plan(128, 64852, 128)?;
        let expected = {
            let _old = prefill_mm::TestScope::new(Some(false));
            let _scope = prefill_mm::Scope::enter(device, true);
            profile::prefill_tile_plan(128, 64852, 128)?
        };
        ensure!(
            serde_json::to_value(plan)? == serde_json::to_value(expected)?,
            "opt-out {name} planner charged native prefill"
        );
        cases.push(serde_json::json!({"control":name,"native":false,"old_plan":plan}));
    }
    profile::set_grouped_iq1_enabled(true);
    ATTENTION_PHASED.with(|v| v.set(true));
    SHARED_TILE_REFERENCE.with(|v| v.set(false));
    IQ1_PACKET_BATCH.with(|v| v.set(true));
    ensure!(
        native_prefill_eligible(128, true),
        "default controls no longer eligible"
    );
    ensure!(prefill_mm::counts() == before, "policy checks enqueued MM");
    Ok(serde_json::json!({"pass":true,"cases":cases,
        "true_controls_eligible":true,"no_dispatch":true}))
}

#[cfg(target_os = "macos")]
use super::deepseek4_qmatmul::iq1_packets;
#[cfg(target_os = "macos")]
use super::deepseek4_qmatmul::prefill_mm;

fn native_prefill_enabled() -> bool {
    #[cfg(target_os = "macos")]
    {
        prefill_mm::active()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

#[cfg(target_os = "macos")]
fn packet_swiglu(backing: &Tensor, rows: usize, owners: &mut RoutedPostLeases) -> Result<Tensor> {
    ensure!(
        backing.dims() == [12 * rows, 2304]
            && backing.dtype() == DType::F32
            && backing.is_contiguous(),
        "packet canonical gate/up shape"
    );
    let interleaved = backing.reshape((6 * rows, 2, 2304))?;
    let gate = interleaved.narrow(1, 0, 1)?.squeeze(1)?;
    let up = interleaved.narrow(1, 1, 1)?.squeeze(1)?;
    let gate = owners.pin(gate.clamp(f64::NEG_INFINITY, 10.0)?)?;
    let up = owners.pin(up.clamp(-10.0, 10.0)?)?;
    let silu = owners.pin(candle_nn::ops::silu(&gate)?)?;
    owners.pin(silu.mul(&up)?)
}

/// Interleaved [gate0,up0,...,gate5,up5] views, NOT the first/last six rows.
/// Use the SAME Candle clamp/silu/mul operations as swiglu; no BF16 boundary.
/// Pin exposed intermediates before enqueueing their consumers. Candle still
/// implements silu itself; no replacement exp/reciprocal or fused math.
#[cfg(test)]
fn routed_post_swiglu(backing: &Tensor, owners: &mut RoutedPostLeases) -> Result<Tensor> {
    ensure!(
        backing.dims() == [12, 2304] && backing.dtype() == DType::F32 && backing.is_contiguous(),
        "routed post gate/up must be contiguous F32 [12,2304]"
    );
    let interleaved = backing.reshape((6, 2, 2304))?;
    let gate = interleaved.narrow(1, 0, 1)?.squeeze(1)?;
    let up = interleaved.narrow(1, 1, 1)?.squeeze(1)?;
    let gate = owners.pin(gate.clamp(f64::NEG_INFINITY, 10.0)?)?;
    let up = owners.pin(up.clamp(-10.0, 10.0)?)?;
    let silu = owners.pin(candle_nn::ops::silu(&gate)?)?;
    owners.pin(silu.mul(&up)?)
}
/// Phase owners share the outer router/final/error fence lifetime. Never reset
/// expert leases when beginning attention.
#[derive(Default)]
struct AttentionLeases {
    splits: Vec<math::HcSplit>,
    inputs: Vec<Tensor>,
    carriers: Vec<Tensor>,
    lows: Vec<Tensor>,
    /// Batched-path transients and tail owners; same router/error fence.
    batch: Vec<Tensor>,
    /// Aliases of immutable dynamic cache pages, already charged by KV ledger.
    index_pages: Vec<Tensor>,
    /// Exclusion owners: dynamic pages and fixed pending already charged once.
    batch_cache: Vec<Tensor>,
}

impl AttentionLeases {
    #[track_caller]
    fn pin(&mut self, tensor: Tensor) -> Result<usize> {
        let index = self.carriers.len();
        if attention_v2_enabled() || engram_v2_enabled() {
            self.batch.push(tensor.clone());
            cache::require_bounded_loan(&tensor)?;
            self.carriers.push(tensor);
        } else {
            self.carriers.push(cache::retain_bounded(tensor)?);
        }
        Ok(index)
    }
    fn tensors(&self) -> impl Iterator<Item = &Tensor> {
        self.splits
            .iter()
            .map(math::HcSplit::backing)
            .chain(self.inputs.iter())
            .chain(self.carriers.iter())
            .chain(self.lows.iter())
            .chain(self.batch.iter())
    }
}

#[cfg(test)]
thread_local! {
    static ATTENTION_PHASED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn phased_attention_enabled() -> bool {
    #[cfg(test)]
    {
        ATTENTION_PHASED.with(|v| v.get())
    }
    #[cfg(not(test))]
    {
        true
    }
}

// Test-only selectors for the batched AttentionV2/EngramV2 prefill paths.
// There is no release control or user-facing flag; release builds keep both off.
#[cfg(test)]
thread_local! {
    static ATTENTION_V2: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static ENGRAM_V2: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[cfg(test)]
struct AttentionV2Guard {
    previous: bool,
    phased: bool,
}
#[cfg(test)]
impl AttentionV2Guard {
    fn set(enabled: bool) -> Self {
        Self {
            previous: ATTENTION_V2.with(|v| v.replace(enabled)),
            phased: ATTENTION_PHASED.with(|v| v.replace(true)),
        }
    }
}
#[cfg(test)]
impl Drop for AttentionV2Guard {
    fn drop(&mut self) {
        ATTENTION_V2.with(|v| v.set(self.previous));
        ATTENTION_PHASED.with(|v| v.set(self.phased));
    }
}
fn attention_v2_enabled() -> bool {
    #[cfg(test)]
    {
        ATTENTION_V2.with(|v| v.get())
    }
    #[cfg(not(test))]
    {
        false
    }
}
#[cfg(test)]
struct EngramV2Guard(bool);
#[cfg(test)]
impl EngramV2Guard {
    fn set(enabled: bool) -> Self {
        Self(ENGRAM_V2.with(|v| v.replace(enabled)))
    }
}
#[cfg(test)]
impl Drop for EngramV2Guard {
    fn drop(&mut self) {
        ENGRAM_V2.with(|v| v.set(self.0));
    }
}
fn engram_v2_enabled() -> bool {
    #[cfg(test)]
    {
        ENGRAM_V2.with(|v| v.get())
    }
    #[cfg(not(test))]
    {
        false
    }
}

/// Only strict tiled carriers need exact private producer outputs. Ordinary row
/// work keeps its existing allocator; this scope changes no ownership or fences.
fn bounded_prefill_producer<T>(
    device: &Device,
    enabled: bool,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    #[cfg(target_os = "macos")]
    let _bounded = super::deepseek4_qmatmul::BoundedPrefillScope::enter(device, enabled)?;
    #[cfg(not(target_os = "macos"))]
    let _ = (device, enabled);
    f()
}

/// Bounded to MAX_PREFILL_TILE rows; survives errors until the existing outer fence.
/// HC-pre and Selection masks are per token, never shared by rows.
struct TileRow {
    residual: Tensor,
    previous: Tensor,
    selection: Selection,
    ffn: Option<math::HcSplit>,
    input: Option<Tensor>,
}

impl TileRow {
    fn new(residual: Tensor, previous: Tensor) -> Result<Self> {
        Ok(Self {
            residual: cache::retain_bounded(residual)?,
            previous: cache::retain_bounded(previous)?,
            selection: Selection::default(),
            ffn: None,
            input: None,
        })
    }

    fn set_residual(&mut self, residual: Tensor) -> Result<()> {
        self.residual = cache::retain_bounded(residual)?;
        Ok(())
    }

    fn set_ffn(&mut self, split: math::HcSplit) -> Result<()> {
        self.ffn = Some(retain_tile_hc(split)?);
        Ok(())
    }

    fn set_input(&mut self, input: Tensor) -> Result<()> {
        self.input = Some(cache::retain_bounded(input)?);
        Ok(())
    }

    fn finish_ffn(&mut self, out: &Tensor) -> Result<()> {
        let split = self.ffn.as_ref().context("missing Maya tile HC split")?;
        let residual = cache::retain_bounded(math::hc_expand(out, &self.residual, split)?)?;
        let previous = cache::retain_bounded(split.pre().clone())?;
        self.residual = residual;
        self.previous = previous;
        self.ffn = None;
        self.input = None; // The router interval lease now owns this input.
        Ok(())
    }

    fn tensors(&self) -> impl Iterator<Item = &Tensor> {
        [&self.residual, &self.previous]
            .into_iter()
            .chain(self.ffn.iter().map(math::HcSplit::backing))
            .chain(self.input.iter())
            .chain(self.selection.tensors())
    }
}

fn retain_tile_hc(split: math::HcSplit) -> Result<math::HcSplit> {
    Ok(math::HcSplit::from_backing(cache::retain_bounded(
        split.backing().clone(),
    )?)?)
}

impl MoeLeases {
    fn tensors(&self) -> impl Iterator<Item = &Tensor> {
        self.input
            .iter()
            .chain(self.gate_up.iter())
            .chain(self.tile_inputs.iter())
            .chain(self.tile_gate_up.iter().filter_map(Option::as_ref))
            .chain(self.shared_input.iter())
            .chain(self.shared_output.iter())
            .chain(self.attention.tensors())
            .chain(self.routed_post.tensors())
    }

    fn retain_tile_inputs(&mut self, rows: &[TileRow]) -> Result<()> {
        // Bounded before cloning into the interval; aliases are counted once.
        for row in rows {
            self.tile_inputs.push(cache::retain_bounded(
                row.input
                    .as_ref()
                    .context("missing Maya tile input")?
                    .clone(),
            )?);
        }
        self.tile_gate_up.resize_with(rows.len(), || None);
        Ok(())
    }
}

fn tile_owner_bytes(rows: &[TileRow], leases: &MoeLeases) -> usize {
    // Packed expert weights and dynamic KV are charged by their existing ledgers.
    let cached = &leases.attention.batch_cache;
    cache::retained_bytes(
        rows.iter()
            .flat_map(TileRow::tensors)
            .chain(leases.tensors())
            .chain(cached.iter()),
    ) - cache::retained_bytes(cached.iter())
}

fn observe_tile_owners(
    rows: &[TileRow],
    leases: &MoeLeases,
    plan: profile::PrefillTilePlan,
) -> Result<()> {
    if !profile::prefill_validation_enabled() && !attention_v2_enabled() && !engram_v2_enabled() {
        return Ok(());
    }
    profile::record_tile_owners(tile_owner_bytes(rows, leases), plan)
}

fn route_union(routes: &[math::RouterSelection]) -> Vec<usize> {
    let mut seen = [false; 384];
    let mut union = Vec::with_capacity(routes.len() * 6);
    for route in routes {
        for &id in &route.indices {
            if !seen[id] {
                seen[id] = true;
                union.push(id);
            }
        }
    }
    union
}

/// Largest tile for the Coder's tile-level layers (one lone-request prompt
/// chunk). Activations stay a few hundred MiB at this size.
const CODER_TILE_ROWS: usize = 512;

fn tile_routes(values: &[f32], bias: &[f32], rows: usize) -> Result<Vec<math::RouterSelection>> {
    ensure!(
        (1..=CODER_TILE_ROWS.max(profile::MAX_PREFILL_TILE)).contains(&rows)
            && matches!(bias.len(), 192 | 384)
            && values.len() == rows * bias.len(),
        "Maya tile router row mismatch"
    );
    Ok(values
        .chunks_exact(bias.len())
        .map(|v| math::router_top6(v, Some(bias)))
        .collect::<candle_core::Result<Vec<_>>>()?)
}

thread_local! {
    static CODER_MODEL_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static CODER_BATCH_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
struct CoderBatchScope(bool, bool, std::marker::PhantomData<std::rc::Rc<()>>);
impl CoderBatchScope {
    fn enter(coder: bool) -> Self {
        let previous_model = CODER_MODEL_ACTIVE.with(|v| v.replace(coder));
        let coder = coder && batched_shared_enabled();
        Self(
            previous_model,
            CODER_BATCH_ACTIVE.with(|v| v.replace(coder)),
            std::marker::PhantomData,
        )
    }
}
impl Drop for CoderBatchScope {
    fn drop(&mut self) {
        CODER_MODEL_ACTIVE.with(|v| v.set(self.0));
        CODER_BATCH_ACTIVE.with(|v| v.set(self.1));
    }
}
/// The Coder's whole-tile attention runs on Metal.
fn tile_attention_enabled(device: &Device) -> bool {
    device.is_metal()
}
fn coder_routed_batch_enabled() -> bool {
    CODER_BATCH_ACTIVE.with(|v| v.get())
}
/// The Coder's tiled expert kernels read each expert once per tile, so it
/// requests the largest tile unless an operator chose a size. The planner
/// still shrinks it near the fixed scratch reserve.
fn coder_prefill_tile() -> usize {
    match profile::prefill_tile() {
        profile::DEFAULT_PREFILL_TILE => profile::MAX_PREFILL_TILE,
        requested => requested,
    }
}

struct Hc {
    fc: Tensor,
    scale: Tensor,
    base: Tensor,
}
impl Hc {
    fn load(w: &Weights, p: &str, d: &Device) -> Result<Self> {
        Ok(Self {
            fc: weights::dense(w, &format!("{p}_fn.weight"), d)?,
            scale: weights::dense(w, &format!("{p}_scale.weight"), d)?,
            base: weights::dense(w, &format!("{p}_base.weight"), d)?,
        })
    }
    fn split(&self, residual: &Tensor) -> Result<math::HcSplit> {
        Ok(math::hc_split(residual, &self.fc, &self.scale, &self.base)?)
    }
}

struct Engram {
    table: engram::Table,
    kv: Linear,
    qnorm: Tensor,
    knorm: Tensor,
}
struct Layer {
    attn_hc: Hc,
    ffn_hc: Hc,
    attn_norm: Tensor,
    ffn_norm: Tensor,
    attention: Attention,
    router: Linear,
    bias: Vec<f32>,
    shared_gate: Linear,
    shared_up: Linear,
    shared_down: Linear,
    engram: Option<Engram>,
}

/// Only the serialized packed weight cache is model-global mutable state.
/// Histories, rings, compression carries and absolute positions belong to slots
/// in the caller's session. The mutex spans all GPU execution and error fences.
pub struct DeepSeek41Stage {
    layers: Vec<Layer>,
    embedding: Arc<RawTensor>,
    output_norm: Tensor,
    output: Linear,
    layout: engram::Layout,
    expert_cache: Mutex<ExpertCache>,
    device: Device,
    context: usize,
    coder: bool,
}

impl DeepSeek41Stage {
    /// Public preflight, also called before the loader uploads resident tensors.
    pub fn validate_stage_config(
        md: &GgufMetadata,
        range: &LayerRange,
        turboquant: bool,
        device: &Device,
    ) -> Result<()> {
        range.validate()?;
        ensure!(
            md.architecture == "deepseek41",
            "Maya requires native deepseek41 metadata"
        );
        ensure!(
            range.is_solo() && range.total_layers == 40 && md.num_layers == 40,
            "Maya requires all 40 local layers; partial/pipeline stages are unsupported"
        );
        ensure!(
            !turboquant,
            "Maya F32 QDQ cache does not support TurboQuant"
        );
        ensure!(device.is_metal(), "Maya native inference requires Metal");
        ensure!(
            md.hidden_dim == 5120
                && md.vocab_size == 129280
                && md.num_attention_heads == 64
                && md.num_kv_heads == 1
                && md.head_dim == 512
                && md.qk_rope_head_dim == 64
                && md.q_lora_rank == 1280
                && md.output_lora_rank == 1024
                && md.output_group_count == 8
                && matches!(md.expert_count, 192 | 384)
                && md.expert_used_count == 6
                && md.expert_ffn_dim == 2304
                && md.shared_expert_count == 1
                && md.indexer_head_count == 32
                && md.indexer_head_dim == 128
                && md.indexer_top_k == 512
                && md.hyperconnection_count == 4
                && md.hyperconnection_sinkhorn_iterations == 20
                && md.sliding_window == 128
                && md.context_length > 0
                && md.context_length <= 1048576,
            "unsupported Maya production geometry"
        );
        ensure!(
            md.compress_ratios == [vec![0; 2], vec![2; 18], vec![1; 20]].concat()
                && (md.rms_norm_eps - f64::from(RMS_EPS)).abs() < 1e-26
                && md.rope_theta == 10000.
                && md.compress_rope_freq_base == 160000.
                && md.rope_scaling_factor == 16.
                && md.rope_yarn_beta_fast == 32.
                && md.rope_yarn_beta_slow == 1.
                && md.expert_weights_norm
                && md.expert_weights_scale == 1.5,
            "unsupported Maya production math/schedule"
        );
        Ok(())
    }

    pub(crate) fn validate_source(source: &crate::deepseek4_gguf::RawGgufContent) -> Result<()> {
        weights::validate_source(source)
    }

    /// Construct the full local text graph from packed weights and its owned
    /// source. Validation precedes all dense-vector decoding and cache setup.
    pub fn from_stage_tensors(
        stage: StageTensors,
        quant: Option<Arc<TurboQuantConfig>>,
    ) -> Result<Self> {
        Self::validate_stage_config(
            &stage.metadata,
            &stage.range,
            quant.is_some(),
            &stage.device,
        )?;
        ensure!(
            stage.tensors.is_empty(),
            "Maya must retain packed raw tensors"
        );
        let source = stage
            .raw_source
            .as_ref()
            .context("Maya requires its owned GGUF source descriptor")?;
        source.require_verified_native()?;
        let coder = source.is_coder();
        let source_metadata = crate::gguf_loader::parse_gguf_metadata(&source.content().metadata)?;
        Self::validate_stage_config(
            &source_metadata,
            &stage.range,
            quant.is_some(),
            &stage.device,
        )?;
        ensure!(
            source_metadata.context_length == stage.metadata.context_length,
            "Maya resident/source context mismatch"
        );
        drop(source.clone_file()?);
        Self::validate_source(source.content())?;
        // Reject mismatched source/header or foreign allocations before decoding
        // even the small dense vectors. Disk-only banks cannot enter this map.
        for (name, info) in &source.content().tensor_infos {
            if name.ends_with("_exps.weight")
                || name.ends_with(".engram_embd.weight")
                || name == "per_layer_token_embd.weight"
            {
                continue;
            }
            let raw = weights::raw(&stage.raw_tensors, name)?;
            ensure!(
                raw.shape() == info.shape
                    && raw.dtype() == info.dtype
                    && raw.bytes().device().same_device(&stage.device),
                "Maya resident/source mismatch: {name}"
            );
        }
        ensure!(
            stage
                .raw_tensors
                .keys()
                .all(|n| source.content().tensor_infos.contains_key(n)
                    && !n.ends_with("_exps.weight")
                    && !n.ends_with(".engram_embd.weight")
                    && n != "per_layer_token_embd.weight"),
            "foreign Maya resident tensor"
        );
        let layout = weights::engram_layout(source.content())?;
        let d = &stage.device;
        let w = &stage.raw_tensors;
        let mut layers = Vec::with_capacity(40);
        for l in 0..40 {
            let p = format!("blk.{l}.");
            let dense = |n: &str| weights::dense(w, &format!("{p}{n}.weight"), d);
            let linear = |n: &str| weights::linear(w, &format!("{p}{n}.weight"), d);
            let engram = if engram::LAYER_IDS.contains(&(l as u32)) {
                let name = if coder {
                    "per_layer_token_embd.weight".to_owned()
                } else {
                    format!("{p}engram_embd.weight")
                };
                let info = &source.content().tensor_infos[&name];
                let (file, identity, mut offset) = source.tensor_file(&name)?;
                let bytes = if coder {
                    let table_index = usize::from(l == 14);
                    if table_index == 1 {
                        offset = offset
                            .checked_add(u64::from(engram::TABLE_ROWS[0]) * 144)
                            .context("Coder Engram offset overflow")?;
                    }
                    u64::from(engram::TABLE_ROWS[table_index]) * 144
                } else {
                    info.byte_len()? as u64
                };
                let span = layout.table_span(l as u32, offset, bytes)?;
                let (qnorm, knorm) = if coder {
                    (
                        dense("engram_gate")?,
                        Tensor::ones((4, 5120), DType::F32, d)?,
                    )
                } else {
                    (dense("engram_q_norm")?, dense("engram_k_norm")?)
                };
                Some(Engram {
                    table: engram::Table::from_file(file, identity, span)?,
                    kv: linear("engram_kv")?,
                    qnorm,
                    knorm,
                })
            } else {
                None
            };
            layers.push(Layer {
                attn_hc: Hc::load(w, &format!("{p}hc_attn"), d)?,
                ffn_hc: Hc::load(w, &format!("{p}hc_ffn"), d)?,
                attn_norm: dense("attn_norm")?,
                ffn_norm: dense("ffn_norm")?,
                attention: Attention::load(w, l, d)?,
                router: linear("ffn_gate_inp")?,
                bias: weights::dense(w, &format!("{p}exp_probs_b.bias"), &Device::Cpu)?
                    .to_vec1()?,
                shared_gate: linear("ffn_gate_shexp")?,
                shared_up: linear("ffn_up_shexp")?,
                shared_down: linear("ffn_down_shexp")?,
                engram,
            });
            #[cfg(all(test, target_os = "macos"))]
            crate::gguf_loader::MAYA_UPLOAD_PEAK.with(|peak| {
                peak.set(
                    peak.get().max(
                        d.as_metal_device()
                            .unwrap()
                            .device()
                            .current_allocated_size() as u64,
                    ),
                )
            });
        }
        Ok(Self {
            layers,
            embedding: Arc::clone(weights::raw(w, "token_embd.weight")?),
            output_norm: weights::dense(w, "output_norm.weight", d)?,
            output: weights::linear(w, "output.weight", d)?,
            layout,
            expert_cache: Mutex::new(ExpertCache::new(Arc::clone(source), d.clone())?),
            device: d.clone(),
            context: stage.metadata.context_length,
            coder,
        })
    }

    fn initial_residual(&self, token: usize) -> Result<Tensor> {
        if self.coder {
            let embedding = ops::embedding_q5(self.embedding.bytes(), token, 5120)?;
            Ok(embedding.broadcast_as((4, 5120))?.contiguous()?)
        } else {
            initial_residual(self.embedding.bytes(), token, 5120)
        }
    }

    fn run(
        &self,
        input: StageInput,
        slots: &mut [Option<LayerCache>],
        seq_len: usize,
        start: usize,
        logits: bool,
    ) -> Result<StageOutput> {
        let StageInput::Tokens(tokens) = input else {
            anyhow::bail!(
                "Maya accepts scalar text Tokens only; activations/multimodal are unsupported"
            );
        };
        ensure!(
            tokens.len() == seq_len && seq_len > 0 && tokens.iter().all(|&t| t < 129280),
            "invalid Maya token sequence"
        );
        ensure!(
            slots.len() == 40 && slots[1..].iter().all(Option::is_none),
            "Maya requires slot-zero model cache and 39 empty slots"
        );
        let mut candidate = match &slots[0] {
            None => DeepSeek41Cache::default(),
            Some(LayerCache::DeepSeek41(c)) => (**c).clone(),
            _ => anyhow::bail!("Maya cache type mismatch"),
        };
        candidate.validate_position(start, seq_len, self.context)?;
        let _coder_batch = CoderBatchScope::enter(self.coder);
        let _packed_reuse = super::deepseek4_qmatmul::PackedReuse4Scope::enter(self.coder);
        let grouped_iq1 = !self.coder && profile::grouped_iq1_effective();
        #[cfg(target_os = "macos")]
        let _native_prefill = prefill_mm::Scope::enter(
            &self.device,
            !self.coder && native_prefill_eligible(seq_len, grouped_iq1),
        );
        // One snapshot covers dispatch and retained-byte charges for every tile.
        // Coder keeps scalar arithmetic in each projection, batches the router
        // readback, and groups shared expert inputs within the admitted tile.
        let requested = if self.coder {
            coder_prefill_tile()
        } else if native_prefill_enabled() {
            profile::MAX_PREFILL_TILE
        } else {
            profile::prefill_tile()
        };
        let mut tile_plan = profile::prefill_tile_plan(requested, start, seq_len)?;
        // Tile-level Coder layers read every routed expert once per tile, so
        // larger tiles cut expert traffic per token. They keep their own cap,
        // separate from the row path's planner bound; a planner-reduced or
        // operator-chosen tile is left as is.
        if self.coder_tile_enabled(seq_len)
            && requested == profile::MAX_PREFILL_TILE
            && tile_plan.effective == requested.min(seq_len)
        {
            tile_plan.effective = CODER_TILE_ROWS.min(seq_len);
            tile_plan.reason = "coder_tile_layers";
        }
        let mut experts = self
            .expert_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("Maya whole-forward mutex poisoned"))?;
        let mut leases = MoeLeases::default();
        let mut tile_rows = if tile_plan.effective > 1 {
            Vec::with_capacity(tile_plan.effective)
        } else {
            Vec::new()
        };
        let result = (|| -> Result<Tensor> {
            profile::record_prefill_plan(tile_plan, seq_len);
            if tile_plan.effective > 1 {
                return self.run_tiled(
                    &tokens,
                    start,
                    logits,
                    tile_plan,
                    &mut candidate,
                    &mut experts,
                    &mut leases,
                    &mut tile_rows,
                    grouped_iq1,
                );
            }
            let mut last = None;
            for (offset, &token) in tokens.iter().enumerate() {
                let pos = start + offset;
                let stamp = profile::start(&self.device, Region::Embedding)?;
                let hash = self
                    .layout
                    .hash(&candidate.history, &[token as i32], None)?;
                let mut residual = self.initial_residual(token as usize)?;
                // HC carry is layer-local within this token, never history.
                let mut previous = initial_pre(&self.device)?;
                let mut selection = Selection::default();
                profile::finish(&self.device, stamp)?;
                for (l, layer) in self.layers.iter().enumerate() {
                    if let Some(engram) = &layer.engram {
                        let stamp = profile::start(&self.device, Region::Engram)?;
                        let table_index = usize::from(l == 14);
                        let rows = engram.table.read_rows(&hash.rows[0][table_index])?;
                        let projection = profile::detail(profile::Detail::EngramProjection);
                        let flat: Vec<f32> = rows.into_iter().flatten().collect();
                        let input = Tensor::from_vec(flat, (1, 6144), &self.device)?;
                        let kv = engram.kv.forward(&input)?.reshape((5, 5120))?;
                        residual = math::engram_add(
                            &residual,
                            &kv,
                            &engram.qnorm,
                            &engram.knorm,
                            RMS_EPS,
                        )?;
                        drop(projection);
                        profile::finish(&self.device, stamp)?;
                    }
                    let stamp = profile::start(&self.device, Region::AttentionPrepare)?;
                    let attn_split = layer.attn_hc.split(&residual)?;
                    let x = math::hc_collapse(&residual, &previous)?;
                    let x = math::rms_norm(&x, &layer.attn_norm, RMS_EPS)?.reshape((1, 5120))?;
                    profile::finish(&self.device, stamp)?;
                    let stamp = profile::start(&self.device, Region::Attention)?;
                    let out =
                        layer
                            .attention
                            .forward(&x, l, pos, &mut candidate, &mut selection)?;
                    profile::finish(&self.device, stamp)?;
                    let stamp = profile::start(&self.device, Region::AttentionPost)?;
                    residual = math::hc_expand(&out, &residual, &attn_split)?;
                    profile::finish(&self.device, stamp)?;
                    let stamp = profile::start(&self.device, Region::FfnPrepare)?;
                    let ffn_split = layer.ffn_hc.split(&residual)?;
                    let x = math::hc_collapse(&residual, attn_split.pre())?;
                    let x = math::rms_norm(&x, &layer.ffn_norm, RMS_EPS)?.reshape((1, 5120))?;
                    profile::finish(&self.device, stamp)?;
                    let out = layer.moe(
                        &x,
                        l,
                        &self.device,
                        &mut experts,
                        &mut leases,
                        grouped_iq1,
                        false,
                    )?;
                    let stamp = profile::start(&self.device, Region::FfnPost)?;
                    residual = math::hc_expand(&out, &residual, &ffn_split)?;
                    previous = ffn_split.pre().clone();
                    profile::finish(&self.device, stamp)?;
                }
                if logits && offset + 1 == seq_len {
                    let stamp = profile::start(&self.device, Region::OutputHead)?;
                    let x = math::hc_collapse(&residual, &previous)?;
                    let x = math::rms_norm(&x, &self.output_norm, RMS_EPS)?.reshape((1, 5120))?;
                    last = Some(self.output.forward(&x)?.reshape(129280)?);
                    profile::finish(&self.device, stamp)?;
                }
                candidate.history = hash.next_history;
                candidate.position = pos + 1;
            }
            Ok(match last {
                Some(x) => x,
                None => Tensor::zeros(1, DType::F32, &self.device)?,
            })
        })();
        // Always fence before lease release, including disk/math/cache failures.
        // Nothing in the caller's cache has been changed yet.
        let stamp = profile::start_unfenced(Region::FinalFence);
        let fence = self.device.synchronize();
        profile::finish_unfenced(stamp);
        profile::publish_cache(|| experts.stats());
        drop(leases);
        drop(tile_rows);
        fence.context("Maya whole-forward GPU fence")?;
        let output = result?;
        candidate.validate_frontier()?;
        if profile::prefill_validation_enabled() {
            profile::publish_frontier(candidate.frontier_report()?);
        }
        slots[0] = Some(LayerCache::DeepSeek41(Box::new(candidate)));
        Ok(StageOutput::Logits(output))
    }

    /// Tile-level Coder layers apply on qualified Metal devices for
    /// multi-row tiles, unless tile validation is on.
    fn coder_tile_enabled(&self, tokens: usize) -> bool {
        #[cfg(target_os = "macos")]
        {
            self.coder
                && tokens > 1
                && tile_attention_enabled(&self.device)
                && !profile::prefill_validation_enabled()
                && !attention_v2_enabled()
                && !engram_v2_enabled()
                && batched_shared_enabled()
                && prefill_mm::packed_available(&self.device)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = tokens;
            false
        }
    }

    /// One Coder prompt tile through every layer with tile-level tensors:
    /// residual streams `[T,4,5120]`, HC splits `[T,24]`, and one dispatch per
    /// step instead of one per row. Each row computes what the row loop
    /// computes, with the tiled kernels' summation order, and rows never mix,
    /// so results do not depend on how a prompt is split into tiles. Leaves
    /// each row's final residual and pre gates in `rows` for the output head.
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    fn coder_tile(
        &self,
        tokens: &[u32],
        base: usize,
        hash: &engram::HashBatch,
        mut engram_rows: Option<[Vec<Vec<[f32; engram::DIM]>>; 2]>,
        candidate: &mut DeepSeek41Cache,
        experts: &mut ExpertCache,
        leases: &mut MoeLeases,
        rows: &mut Vec<TileRow>,
    ) -> Result<()> {
        let t = tokens.len();
        let device = &self.device;
        let initial = tokens
            .iter()
            .map(|&token| self.initial_residual(token as usize))
            .collect::<Result<Vec<_>>>()?;
        let mut residual = Tensor::stack(&initial, 0)?;
        let mut previous = initial_pre(device)?.broadcast_as((t, 4))?.contiguous()?;
        let mut stride = 4;
        let mut selections = (0..t).map(|_| Selection::default()).collect::<Vec<_>>();
        let mut tile_selection: [Option<tile_attention::TileSelection>; 4] =
            std::array::from_fn(|_| None);
        for (l, layer) in self.layers.iter().enumerate() {
            if let Some(engram) = &layer.engram {
                let table = usize::from(l == 14);
                let mut flat = Vec::with_capacity(t * 6144);
                for r in 0..t {
                    let values = match &mut engram_rows {
                        Some(rows) => std::mem::take(&mut rows[table][r]),
                        None => engram.table.read_rows(&hash.rows[r][table])?,
                    };
                    flat.extend(values.into_iter().flatten());
                }
                let input = Tensor::from_vec(flat, (t, 6144), device)?;
                let kv = engram
                    .kv
                    .forward_prefill_matrix(&input)?
                    .reshape((t, 5, 5120))?;
                let weight = engram.qnorm.mul(&engram.knorm)?;
                residual = math::engram_add_rows(&residual, &kv, &weight, RMS_EPS)?;
            }
            let hc = &layer.attn_hc;
            let attn_split = math::hc_split_rows(&residual, &hc.fc, &hc.scale, &hc.base)?;
            let x = math::hc_collapse_rows(&residual, &previous, stride)?;
            let x = math::rms_norm(&x, &layer.attn_norm, RMS_EPS)?;
            let attention = &layer.attention;
            let qa = attention.prepare_qa(&x)?;
            let kv = attention.prepare_kv(&x)?;
            let q = attention.prepare_q(&qa)?;
            let mut pins = Vec::new();
            let mut refs = selections.iter_mut().collect::<Vec<_>>();
            let heads = bounded_prefill_producer(device, true, || {
                attention.attend_tile(
                    &x,
                    &qa,
                    &kv,
                    &q,
                    l,
                    base,
                    candidate,
                    &mut refs,
                    &mut tile_selection,
                    &mut pins,
                )
            })?;
            let low = match attention.project_heads(&heads, t)? {
                Some(low) => low,
                None => Tensor::cat(
                    &(0..t)
                        .map(|r| attention.project_heads_row(&heads.narrow(0, r * 8, 8)?))
                        .collect::<Result<Vec<_>>>()?,
                    0,
                )?,
            };
            let out = attention.project_output(&low)?;
            residual = math::hc_expand_rows(&out, &residual, &attn_split)?;
            let hc = &layer.ffn_hc;
            let ffn_split = math::hc_split_rows(&residual, &hc.fc, &hc.scale, &hc.base)?;
            // FFN collapses with this layer's attention pre, as the row loop does.
            let x = math::hc_collapse_rows(&residual, &attn_split, 24)?;
            let x = math::rms_norm(&x, &layer.ffn_norm, RMS_EPS)?;
            let out = layer.moe_coder_tile(&x, l, experts, leases)?;
            residual = math::hc_expand_rows(&out, &residual, &ffn_split)?;
            previous = ffn_split;
            stride = 24;
        }
        let previous = previous.narrow(1, 0, 4)?.contiguous()?;
        rows.clear();
        for r in 0..t {
            rows.push(TileRow::new(
                residual.get(r)?.contiguous()?,
                previous.get(r)?.contiguous()?,
            )?);
        }
        Ok(())
    }

    #[cfg(not(target_os = "macos"))]
    #[allow(clippy::too_many_arguments)]
    fn coder_tile(
        &self,
        _: &[u32],
        _: usize,
        _: &engram::HashBatch,
        _: Option<[Vec<Vec<[f32; engram::DIM]>>; 2]>,
        _: &mut DeepSeek41Cache,
        _: &mut ExpertCache,
        _: &mut MoeLeases,
        _: &mut Vec<TileRow>,
    ) -> Result<()> {
        anyhow::bail!("Coder tile layers require Metal")
    }

    /// Read every Engram row a tile needs, for both Engram layers, across
    /// several threads before the layer loop. The table bypasses the page
    /// cache, so each row is an SSD read; issuing them concurrently instead of
    /// one at a time per token hides most of the latency. Bytes and decoding
    /// are those of `Table::read_rows`. `None` keeps per-row reads (one-token
    /// tiles).
    #[allow(clippy::type_complexity)]
    fn prefetch_engram(
        &self,
        hash: &engram::HashBatch,
    ) -> Result<Option<[Vec<Vec<[f32; engram::DIM]>>; 2]>> {
        let tokens = hash.rows.len();
        if tokens < 2 {
            return Ok(None);
        }
        let mut tables = [None, None];
        for (l, layer) in self.layers.iter().enumerate() {
            if let Some(engram) = &layer.engram {
                tables[usize::from(l == 14)] = Some(&engram.table);
            }
        }
        let [Some(first), Some(second)] = tables else {
            return Ok(None);
        };
        const THREADS: usize = 16;
        let per = tokens.div_ceil(THREADS);
        let parts = std::thread::scope(|scope| {
            let handles = (0..tokens)
                .step_by(per)
                .map(|start| {
                    let end = (start + per).min(tokens);
                    scope.spawn(move || -> Result<Vec<[Vec<[f32; engram::DIM]>; 2]>> {
                        (start..end)
                            .map(|r| {
                                Ok([
                                    first.read_rows(&hash.rows[r][0])?,
                                    second.read_rows(&hash.rows[r][1])?,
                                ])
                            })
                            .collect()
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .map_err(|_| anyhow::anyhow!("Engram prefetch thread panicked"))?
                })
                .collect::<Result<Vec<_>>>()
        })?;
        let mut out = [Vec::with_capacity(tokens), Vec::with_capacity(tokens)];
        for [a, b] in parts.into_iter().flatten() {
            out[0].push(a);
            out[1].push(b);
        }
        Ok(Some(out))
    }

    // Keep each cache and owner borrow explicit through the outer GPU fence.
    #[allow(clippy::too_many_arguments)]
    fn run_tiled(
        &self,
        tokens: &[u32],
        start: usize,
        logits: bool,
        plan: profile::PrefillTilePlan,
        candidate: &mut DeepSeek41Cache,
        experts: &mut ExpertCache,
        leases: &mut MoeLeases,
        rows: &mut Vec<TileRow>,
        grouped_iq1: bool,
    ) -> Result<Tensor> {
        let tile = plan.effective;
        let _relaxed = cache::RelaxedTileRetention::enter(
            self.coder && !profile::prefill_validation_enabled(),
        );
        for (tile_index, tokens) in tokens.chunks(tile).enumerate() {
            let base = start + tile_index * tile;
            // Hash exactly in input order, with duplicates and original source
            // identity checks left in Table::read_rows. Caller state is untouched.
            let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            // The hasher takes at most MAX_BATCH_TOKENS; its history rolls
            // token by token, so pieces carrying it give the same rows.
            let mut hash = self.layout.hash(
                &candidate.history,
                &ids[..ids.len().min(engram::MAX_BATCH_TOKENS)],
                None,
            )?;
            for piece in
                ids[ids.len().min(engram::MAX_BATCH_TOKENS)..].chunks(engram::MAX_BATCH_TOKENS)
            {
                let next = self.layout.hash(&hash.next_history, piece, None)?;
                hash.rows.extend(next.rows);
                hash.next_history = next.next_history;
            }
            let mut engram_rows = self.prefetch_engram(&hash)?;
            if self.coder_tile_enabled(tokens.len()) {
                self.coder_tile(
                    tokens,
                    base,
                    &hash,
                    engram_rows,
                    candidate,
                    experts,
                    leases,
                    rows,
                )?;
                candidate.history = hash.next_history;
                candidate.position = base + tokens.len();
                continue;
            }
            rows.clear();
            for &token in tokens {
                let stamp = profile::start(&self.device, Region::Embedding)?;
                rows.push(TileRow::new(
                    self.initial_residual(token as usize)?,
                    initial_pre(&self.device)?,
                )?);
                observe_tile_owners(rows, leases, plan)?;
                profile::finish(&self.device, stamp)?;
            }
            // Each owner's per-row compressed selection, set by its refresh
            // layer and reused by its later layers within this tile.
            let mut tile_selection: [Option<tile_attention::TileSelection>; 4] =
                std::array::from_fn(|_| None);
            for (l, layer) in self.layers.iter().enumerate() {
                let phased = phased_attention_enabled() && tokens.len() > 1;
                // Each layer's raw ring and each owner's odd-pair carry advance
                // in token order. Consumers use explicit absolute prefix views.
                // Only the static lookup/projection is batched. Each original
                // row/layer still applies Engram before its HC transition.
                let batched_engram = if engram_v2_enabled() {
                    if let Some(engram) = &layer.engram {
                        let stamp = profile::start(&self.device, Region::Engram)?;
                        let table_index = usize::from(l == 14);
                        let ids: Vec<u32> = hash
                            .rows
                            .iter()
                            .flat_map(|row| row[table_index].iter().copied())
                            .collect();
                        let values = engram.table.read_rows_batched(&ids)?;
                        let projection = profile::detail(profile::Detail::EngramProjection);
                        let flat: Vec<f32> = values.into_iter().flatten().collect();
                        let input = Tensor::from_vec(flat, (tokens.len(), 6144), &self.device)?;
                        leases.attention.batch.push(input.clone());
                        cache::require_bounded_loan(&input)?;
                        let kv =
                            bounded_prefill_producer(&self.device, engram_v2_enabled(), || {
                                engram.kv.forward_prefill_matrix(&input)
                            })?;
                        leases.attention.batch.push(kv.clone());
                        cache::require_bounded_loan(&kv)?;
                        let kv = kv.reshape((tokens.len(), 5, 5120))?;
                        drop(projection);
                        profile::finish(&self.device, stamp)?;
                        Some(kv)
                    } else {
                        None
                    }
                } else {
                    None
                };
                for r in 0..rows.len() {
                    let row = &mut rows[r];
                    let pos = base + r;
                    if let Some(engram) = &layer.engram {
                        let stamp = profile::start(&self.device, Region::Engram)?;
                        let table_index = usize::from(l == 14);
                        let kv = if let Some(kv) = &batched_engram {
                            kv.narrow(0, r, 1)?.reshape((5, 5120))?
                        } else {
                            let values = match &mut engram_rows {
                                Some(rows) => std::mem::take(&mut rows[table_index][r]),
                                None => engram.table.read_rows(&hash.rows[r][table_index])?,
                            };
                            let projection = profile::detail(profile::Detail::EngramProjection);
                            let flat: Vec<f32> = values.into_iter().flatten().collect();
                            let input = Tensor::from_vec(flat, (1, 6144), &self.device)?;
                            let kv = engram.kv.forward(&input)?.reshape((5, 5120))?;
                            drop(projection);
                            kv
                        };
                        row.set_residual(math::engram_add(
                            &row.residual,
                            &kv,
                            &engram.qnorm,
                            &engram.knorm,
                            RMS_EPS,
                        )?)?;
                        profile::finish(&self.device, stamp)?;
                    }
                    let stamp = profile::start(&self.device, Region::AttentionPrepare)?;
                    let attn_split = retain_tile_hc(layer.attn_hc.split(&row.residual)?)?;
                    let x = math::hc_collapse(&row.residual, &row.previous)?;
                    let x = math::rms_norm(&x, &layer.attn_norm, RMS_EPS)?.reshape((1, 5120))?;
                    profile::finish(&self.device, stamp)?;
                    if phased {
                        // Pin each row before preparing the next. Previous expert
                        // interval remains owned until the normal router readback.
                        leases.attention.splits.push(attn_split);
                        leases.attention.inputs.push(cache::retain_bounded(x)?);
                        observe_tile_owners(rows, leases, plan)?;
                        continue;
                    }
                    let stamp = profile::start(&self.device, Region::Attention)?;
                    let out = if tokens.len() == 1 {
                        layer
                            .attention
                            .forward(&x, l, pos, candidate, &mut row.selection)?
                    } else {
                        layer
                            .attention
                            .forward_tiled(&x, l, pos, candidate, &mut row.selection)?
                    };
                    row.selection.retain_bounded()?;
                    profile::finish(&self.device, stamp)?;
                    let stamp = profile::start(&self.device, Region::AttentionPost)?;
                    row.set_residual(math::hc_expand(&out, &row.residual, &attn_split)?)?;
                    profile::finish(&self.device, stamp)?;
                    let stamp = profile::start(&self.device, Region::FfnPrepare)?;
                    row.set_ffn(layer.ffn_hc.split(&row.residual)?)?;
                    let x = math::hc_collapse(&row.residual, attn_split.pre())?;
                    row.set_input(
                        math::rms_norm(&x, &layer.ffn_norm, RMS_EPS)?.reshape((1, 5120))?,
                    )?;
                    profile::finish(&self.device, stamp)?;
                    drop(attn_split); // Active-row carrier, no longer tile-retained.
                    observe_tile_owners(rows, leases, plan)?;
                }
                if phased {
                    let a = &mut leases.attention;
                    let joined = if attention_v2_enabled() || engram_v2_enabled() {
                        cache::cat_rows(&a.inputs, &mut a.batch)?
                    } else {
                        Tensor::cat(&a.inputs, 0)?
                    };
                    let x = a.pin(joined)?;
                    let bounded = attention_v2_enabled() || engram_v2_enabled();
                    let qa = a.pin(bounded_prefill_producer(&self.device, bounded, || {
                        layer.attention.prepare_qa(&a.carriers[x])
                    })?)?;
                    let kv = a.pin(bounded_prefill_producer(&self.device, bounded, || {
                        layer.attention.prepare_kv(&a.carriers[x])
                    })?)?;
                    let q = a.pin(bounded_prefill_producer(&self.device, bounded, || {
                        layer.attention.prepare_q(&a.carriers[qa])
                    })?)?;
                    let v2 = attention_v2_enabled();
                    let scores = if v2 {
                        let a = &mut leases.attention;
                        bounded_prefill_producer(&self.device, v2, || {
                            layer.attention.prepare_index_tile(
                                &a.carriers[x],
                                &a.carriers[qa],
                                l,
                                base,
                                candidate,
                                &mut a.index_pages,
                                &mut a.batch_cache,
                                &mut a.batch,
                            )
                        })?
                    } else {
                        None
                    };
                    observe_tile_owners(rows, leases, plan)?;
                    // The Coder projects every row's heads in one output-A
                    // dispatch after the row-ordered attention loop.
                    let batch_heads = self.coder && !v2 && !engram_v2_enabled();
                    if batch_heads && tile_attention_enabled(&self.device) {
                        let a = &mut leases.attention;
                        let mut selections = rows
                            .iter_mut()
                            .map(|r| &mut r.selection)
                            .collect::<Vec<_>>();
                        let heads = bounded_prefill_producer(&self.device, true, || {
                            layer.attention.attend_tile(
                                &a.carriers[x],
                                &a.carriers[qa],
                                &a.carriers[kv],
                                &a.carriers[q],
                                l,
                                base,
                                candidate,
                                &mut selections,
                                &mut tile_selection,
                                &mut a.batch,
                            )
                        })?;
                        a.lows.push(cache::retain_bounded(heads)?);
                        observe_tile_owners(rows, leases, plan)?;
                    }
                    for r in 0..rows.len() {
                        let a = &mut leases.attention;
                        if batch_heads && tile_attention_enabled(&self.device) {
                            break;
                        }
                        if batch_heads {
                            let heads = layer.attention.attend_heads(
                                &a.carriers[x].narrow(0, r, 1)?,
                                &a.carriers[qa].narrow(0, r, 1)?,
                                &a.carriers[kv].narrow(0, r, 1)?,
                                &a.carriers[q].narrow(0, r, 1)?,
                                l,
                                base + r,
                                candidate,
                                &mut rows[r].selection,
                                None,
                            )?;
                            a.lows.push(cache::retain_bounded(heads)?);
                            rows[r].selection.retain_bounded()?;
                            observe_tile_owners(rows, leases, plan)?;
                            continue;
                        }
                        let score = scores.as_ref().map(|s| s.narrow(0, r, 1)).transpose()?;
                        let low = if v2 {
                            layer.attention.attend_published(
                                &a.carriers[x].narrow(0, r, 1)?,
                                &a.carriers[qa].narrow(0, r, 1)?,
                                &a.carriers[kv].narrow(0, r, 1)?,
                                &a.carriers[q].narrow(0, r, 1)?,
                                l,
                                base + r,
                                candidate,
                                &mut rows[r].selection,
                                score.as_ref(),
                            )?
                        } else {
                            layer.attention.attend_prepared(
                                &a.carriers[x].narrow(0, r, 1)?,
                                &a.carriers[qa].narrow(0, r, 1)?,
                                &a.carriers[kv].narrow(0, r, 1)?,
                                &a.carriers[q].narrow(0, r, 1)?,
                                l,
                                base + r,
                                candidate,
                                &mut rows[r].selection,
                                true,
                            )?
                        };
                        a.lows.push(cache::retain_bounded(low.reshape((1, 8192))?)?);
                        rows[r].selection.retain_bounded()?;
                        observe_tile_owners(rows, leases, plan)?;
                    }
                    let a = &mut leases.attention;
                    let low = if batch_heads {
                        let heads = a.pin(Tensor::cat(&a.lows, 0)?)?;
                        let low = match layer
                            .attention
                            .project_heads(&a.carriers[heads], rows.len())?
                        {
                            Some(low) => low,
                            None => {
                                let all = &a.carriers[heads];
                                let lows = (0..rows.len())
                                    .map(|t| {
                                        layer.attention.project_heads_row(&all.narrow(
                                            0,
                                            t * 8,
                                            8,
                                        )?)
                                    })
                                    .collect::<Result<Vec<_>>>()?;
                                Tensor::cat(&lows, 0)?
                            }
                        };
                        a.pin(low)?
                    } else {
                        let joined = if attention_v2_enabled() || engram_v2_enabled() {
                            cache::cat_rows(&a.lows, &mut a.batch)?
                        } else {
                            Tensor::cat(&a.lows, 0)?
                        };
                        a.pin(joined)?
                    };
                    let output = a.pin(bounded_prefill_producer(
                        &self.device,
                        attention_v2_enabled() || engram_v2_enabled(),
                        || layer.attention.project_output(&a.carriers[low]),
                    )?)?;
                    for (r, row) in rows.iter_mut().enumerate() {
                        let split = &leases.attention.splits[r];
                        let out = leases.attention.carriers[output]
                            .narrow(0, r, 1)?
                            .reshape(5120)?;
                        row.set_residual(math::hc_expand(&out, &row.residual, split)?)?;
                        row.set_ffn(layer.ffn_hc.split(&row.residual)?)?;
                        // FFN uses this row's attention pre, not previous.
                        let x = math::hc_collapse(&row.residual, split.pre())?;
                        row.set_input(
                            math::rms_norm(&x, &layer.ffn_norm, RMS_EPS)?.reshape((1, 5120))?,
                        )?;
                    }
                    observe_tile_owners(rows, leases, plan)?;
                }
                layer.moe_tile(rows, l, &self.device, experts, leases, grouped_iq1, plan)?;
            }
            let end = base + tokens.len();
            candidate.history = hash.next_history;
            candidate.position = end;
        }
        if logits {
            let row = rows.last().context("missing Maya tile output row")?;
            let stamp = profile::start(&self.device, Region::OutputHead)?;
            let x = math::hc_collapse(&row.residual, &row.previous)?;
            let x = math::rms_norm(&x, &self.output_norm, RMS_EPS)?.reshape((1, 5120))?;
            let out = self.output.forward(&x)?.reshape(129280)?;
            profile::finish(&self.device, stamp)?;
            Ok(out)
        } else {
            Ok(Tensor::zeros(1, DType::F32, &self.device)?)
        }
    }
}

fn initial_residual(bytes: &Tensor, token: usize, width: usize) -> Result<Tensor> {
    // ds4.c:42388-42395 repeats the F32 Q8 embedding without a BF16 round.
    let embedding = ops::embedding(bytes, token, width)?;
    Ok(embedding.broadcast_as((4, width))?.contiguous()?)
}

fn initial_pre(device: &Device) -> Result<Tensor> {
    Ok(Tensor::new(&[1.0f32, 0.0, 0.0, 0.0], device)?)
}

fn swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let gate = gate.clamp(f64::NEG_INFINITY, 10.0)?;
    let up = up.clamp(-10.0, 10.0)?;
    Ok(candle_nn::ops::silu(&gate)?.mul(&up)?)
}

impl Layer {
    // Cache and lease borrows have distinct roles at the router readback fence.
    #[allow(clippy::too_many_arguments)]
    fn moe(
        &self,
        x: &Tensor,
        layer: usize,
        device: &Device,
        cache: &mut ExpertCache,
        leases: &mut MoeLeases,
        grouped_iq1: bool,
        bounded_tile: bool,
    ) -> Result<Tensor> {
        let stamp = profile::start_unfenced(Region::RouterProjectionEnqueue);
        let logits = self.router.forward(x)?;
        profile::finish_unfenced(stamp);
        let stamp = profile::start_unfenced(Region::RouterReadbackTop6);
        let readback = profile::detail(profile::Detail::RouterReadback);
        let values = logits.reshape(self.bias.len())?.to_vec1::<f32>()?;
        drop(readback);
        let top6 = profile::detail(profile::Detail::RouterTop6);
        let route = math::router_top6(&values, Some(&self.bias))?;
        drop(values);
        drop(top6);
        profile::finish_unfenced(stamp);
        let stamp = profile::start(device, Region::ExpertCachePrepare)?;
        // Successful readback completed preceding GPU work; old owners retire.
        *leases = MoeLeases::default();
        leases.selected = Some(cache.prepare(layer, &route.indices)?);
        leases.input = Some(if bounded_tile {
            cache::retain_bounded(x.clone())?
        } else {
            x.clone()
        }); // Pin before enqueue.
        profile::finish(device, stamp)?;
        let stamp = profile::start(device, Region::SharedExpert)?;
        let gate = math::bf16(&self.shared_gate.forward(x)?)?;
        let up = math::bf16(&self.shared_up.forward(x)?)?;
        let mid = math::bf16(&swiglu(&gate, &up)?)?;
        let shared = math::bf16(&self.shared_down.forward(&mid)?)?;
        profile::finish(device, stamp)?;
        let stamp = profile::start(device, Region::RoutedExperts)?;
        let routed = routed_experts(leases, &route, grouped_iq1, bounded_tile)?;
        profile::finish(device, stamp)?;
        let stamp = profile::start(device, Region::FfnPost)?;
        let out = math::bf16(&routed.add(&shared)?)?.reshape(5120)?;
        profile::finish(device, stamp)?;
        Ok(out)
    }

    fn shared_batch(&self, input: &Tensor) -> Result<Tensor> {
        let gate = math::bf16(&self.shared_gate.forward_prefill_matrix(input)?)?;
        let up = math::bf16(&self.shared_up.forward_prefill_matrix(input)?)?;
        let mid = math::bf16(&swiglu(&gate, &up)?)?;
        let down = self.shared_down.forward_prefill_matrix(&mid)?;
        cache::retain_bounded(math::bf16(&down)?)
    }

    // Keep the caller's cache, leases, and tile plan explicit at this boundary.
    #[allow(clippy::too_many_arguments)]
    fn moe_tile(
        &self,
        rows: &mut [TileRow],
        layer: usize,
        device: &Device,
        cache: &mut ExpertCache,
        leases: &mut MoeLeases,
        grouped_iq1: bool,
        plan: profile::PrefillTilePlan,
    ) -> Result<()> {
        if rows.len() == 1 {
            let row = &mut rows[0];
            let out = self.moe(
                row.input.as_ref().context("missing Maya tile input")?,
                layer,
                device,
                cache,
                leases,
                grouped_iq1,
                true,
            )?;
            observe_tile_owners(rows, leases, plan)?;
            let stamp = profile::start(device, Region::FfnPost)?;
            rows[0].finish_ffn(&out)?;
            observe_tile_owners(rows, leases, plan)?;
            profile::finish(device, stamp)?;
            return Ok(());
        }
        observe_tile_owners(rows, leases, plan)?;
        let stamp = profile::start_unfenced(Region::RouterProjectionEnqueue);
        #[cfg(target_os = "macos")]
        let mut native_routes = None;
        // Retain projection inputs through router readback even without probes.
        let (_projected, joined) = if native_prefill_enabled() {
            let inputs = rows
                .iter()
                .map(|r| r.input.as_ref().context("missing native router input"))
                .collect::<Result<Vec<_>>>()?;
            let input = leases.routed_post.pin(Tensor::cat(&inputs, 0)?)?;
            let joined = leases
                .routed_post
                .pin(self.router.forward_prefill_matrix(&input)?)?;
            (Vec::new(), joined)
        } else {
            let projected = rows
                .iter()
                .map(|row| {
                    self.router
                        .forward(row.input.as_ref().context("missing Maya tile input")?)
                })
                .collect::<Result<Vec<_>>>()?;
            let joined = Tensor::cat(&projected, 0)?;
            (projected, joined)
        };
        profile::finish_unfenced(stamp);
        let stamp = profile::start_unfenced(Region::RouterReadbackTop6);
        let readback = profile::detail(profile::Detail::RouterReadback);
        let routes = {
            #[cfg(target_os = "macos")]
            if native_prefill_enabled() {
                let routed = prefill_mm::route(&joined, &self.bias)?;
                for tensor in routed.tensors() {
                    leases.routed_post.pin(tensor.clone())?;
                }
                let validated = routed.validate()?;
                let routes = validated.selections().to_vec();
                native_routes = Some(validated);
                routes
            } else {
                tile_routes(
                    &joined.flatten_all()?.to_vec1::<f32>()?,
                    &self.bias,
                    rows.len(),
                )?
            }
            #[cfg(not(target_os = "macos"))]
            {
                tile_routes(
                    &joined.flatten_all()?.to_vec1::<f32>()?,
                    &self.bias,
                    rows.len(),
                )?
            }
        };
        drop(readback);
        let top6 = profile::detail(profile::Detail::RouterTop6);
        drop(top6);
        profile::finish_unfenced(stamp);
        let stamp = profile::start(device, Region::ExpertCachePrepare)?;
        // Only this successful batched readback retires the preceding interval.
        // One lease covers the <=6*T union; no expert/cache budget changes.
        *leases = MoeLeases::default();
        leases.selected = Some(cache.prepare(layer, &route_union(&routes))?);
        leases.retain_tile_inputs(rows)?;
        observe_tile_owners(rows, leases, plan)?;
        profile::finish(device, stamp)?;
        if batched_shared_enabled() || native_prefill_enabled() {
            let stamp = profile::start(device, Region::SharedExpert)?;
            // Pin the pooled carrier BEFORE enqueue: projection failures still
            // leave this owner alive through the existing outer error fence.
            leases.shared_input =
                Some(cache::retain_bounded(Tensor::cat(&leases.tile_inputs, 0)?)?);
            leases.shared_output = Some(self.shared_batch(leases.shared_input.as_ref().unwrap())?);
            profile::finish(device, stamp)?;
            observe_tile_owners(rows, leases, plan)?;
        }
        #[cfg(target_os = "macos")]
        if let Some(routed) = native_routes {
            for tensor in routed.tensors() {
                leases.routed_post.pin(tensor.clone())?;
            }
            return self.moe_native(rows, &routed, leases, device, plan);
        }
        #[cfg(target_os = "macos")]
        {
            if packet_policy(rows.len(), grouped_iq1) {
                let packet_plan =
                    iq1_packets::Plan::new(&routes.iter().map(|r| r.indices).collect::<Vec<_>>())?;
                ensure!(
                    device.is_metal() && leases.shared_input.is_some(),
                    "packet policy requires native shared input"
                );
                #[cfg(test)]
                ensure!(
                    !ROUTED_POST_BATCH.with(|v| v.get()),
                    "packet and routed-post must not overlap"
                );
                self.moe_packet(rows, &routes, &packet_plan, leases, device, plan)?;
                return Ok(());
            }
        }
        if self.bias.len() == 192 && coder_routed_batch_enabled() {
            return self.moe_coder_batch(rows, &routes, leases, plan);
        }
        for r in 0..rows.len() {
            let route = &routes[r];
            let x = &leases.tile_inputs[r];
            let shared = if let Some(output) = &leases.shared_output {
                output.narrow(0, r, 1)?
            } else {
                let stamp = profile::start(device, Region::SharedExpert)?;
                let gate = math::bf16(&self.shared_gate.forward(x)?)?;
                let up = math::bf16(&self.shared_up.forward(x)?)?;
                let mid = math::bf16(&swiglu(&gate, &up)?)?;
                let shared = math::bf16(&self.shared_down.forward(&mid)?)?;
                profile::finish(device, stamp)?;
                shared
            };
            let stamp = profile::start(device, Region::RoutedExperts)?;
            let routed = routed_with_owners(
                x,
                leases.selected.as_ref().unwrap(),
                &mut leases.tile_gate_up[r],
                route,
                grouped_iq1,
                true,
                &mut leases.routed_post,
            )?;
            profile::finish(device, stamp)?;
            observe_tile_owners(rows, leases, plan)?;
            let stamp = profile::start(device, Region::FfnPost)?;
            let out = math::bf16(&routed.add(&shared)?)?.reshape(5120)?;
            rows[r].finish_ffn(&out)?;
            observe_tile_owners(rows, leases, plan)?;
            profile::finish(device, stamp)?;
        }
        Ok(())
    }

    /// Coder MoE for a whole tile `x` `[T,5120]`: one router readback, one
    /// expert-cache preparation, three expert dispatches, the shared expert,
    /// and one ordered combine. Returns BF16 `routed + shared` `[T,5120]`.
    /// The previous interval's leases retire at this readback, as in
    /// `moe_tile`; this interval's leases stay in `leases`.
    #[cfg(target_os = "macos")]
    fn moe_coder_tile(
        &self,
        x: &Tensor,
        layer: usize,
        cache: &mut ExpertCache,
        leases: &mut MoeLeases,
    ) -> Result<Tensor> {
        let t = x.dim(0)?;
        // Router scores choose experts; keep decode's per-row reduction.
        let logits = self.router.forward_decode_rows(x)?;
        let routes = tile_routes(&logits.flatten_all()?.to_vec1::<f32>()?, &self.bias, t)?;
        *leases = MoeLeases::default();
        leases.selected = Some(cache.prepare(layer, &route_union(&routes))?);
        leases.shared_input = Some(x.clone());
        let shared = self.shared_batch(x)?;
        leases.shared_output = Some(shared.clone());
        let mut groups: [Vec<(usize, usize)>; 192] = std::array::from_fn(|_| Vec::new());
        for (token, route) in routes.iter().enumerate() {
            let mut seen = [false; 192];
            for (slot, &id) in route.indices.iter().enumerate() {
                ensure!(id < 192 && !seen[id], "invalid Coder expert route");
                seen[id] = true;
                groups[id].push((token, slot));
            }
        }
        let down = Self::coder_experts_tiled(&groups, leases)?
            .context("Coder tiled experts unavailable")?;
        let mut map = vec![0u32; 6 * t];
        let mut offset = 0;
        for group in groups.iter().filter(|g| !g.is_empty()) {
            for (r, &(token, slot)) in group.iter().enumerate() {
                map[6 * token + slot] = (offset + r) as u32;
            }
            offset += group.len();
        }
        let weights = routes.iter().flat_map(|r| r.weights).collect::<Vec<_>>();
        let routed = route_combine::ordered_rows(&down, &map, &weights)?;
        Ok(math::bf16(&routed.add(&shared)?)?)
    }

    // Group only inputs that name the same packed Coder expert. Each projection
    // keeps its scalar sum order; each token keeps its original six route slots.
    // All carriers and packed weights survive the next readback or error fence.
    fn moe_coder_batch(
        &self,
        rows: &mut [TileRow],
        routes: &[math::RouterSelection],
        leases: &mut MoeLeases,
        plan: profile::PrefillTilePlan,
    ) -> Result<()> {
        ensure!(
            (2..=profile::MAX_PREFILL_TILE).contains(&rows.len()) && rows.len() == routes.len(),
            "invalid Coder expert batch"
        );
        let mut groups: [Vec<(usize, usize)>; 192] = std::array::from_fn(|_| Vec::new());
        for (t, route) in routes.iter().enumerate() {
            let mut seen = [false; 192];
            for (slot, &id) in route.indices.iter().enumerate() {
                ensure!(id < 192 && !seen[id], "invalid Coder expert route");
                seen[id] = true;
                groups[id].push((t, slot));
            }
        }
        let mut downs: Vec<[Option<Tensor>; 6]> = (0..rows.len())
            .map(|_| std::array::from_fn(|_| None))
            .collect();
        let tiled = Self::coder_experts_tiled(&groups, leases)?;
        if let Some(down) = &tiled {
            let mut offset = 0;
            for group in groups.iter().filter(|v| !v.is_empty()) {
                for (r, &(t, slot)) in group.iter().enumerate() {
                    downs[t][slot] = Some(down.narrow(0, offset + r, 1)?);
                }
                offset += group.len();
            }
        }
        let selected = leases.selected.as_ref().context("Coder selected experts")?;
        for (id, group) in groups
            .iter()
            .enumerate()
            .filter(|(_, v)| tiled.is_none() && !v.is_empty())
        {
            let expert = selected.get(id).context("missing Coder expert")?;
            let inputs = group
                .iter()
                .map(|&(t, _)| &leases.tile_inputs[t])
                .collect::<Vec<_>>();
            let input = leases.routed_post.pin(Tensor::cat(&inputs, 0)?)?;
            let gate = leases
                .routed_post
                .pin(expert.gate().forward_prefill_rows(&input)?)?;
            let up = leases
                .routed_post
                .pin(expert.up().forward_prefill_rows(&input)?)?;
            let mid = leases.routed_post.pin(swiglu(&gate, &up)?)?;
            let down = leases
                .routed_post
                .pin(expert.down().forward_prefill_rows(&mid)?)?;
            for (r, &(t, slot)) in group.iter().enumerate() {
                downs[t][slot] = Some(down.narrow(0, r, 1)?);
            }
        }
        for (t, (row, parts)) in rows.iter_mut().zip(downs).enumerate() {
            let parts = parts.map(|v| v.expect("all six Coder expert slots"));
            let routed = route_combine::ordered(&parts, &routes[t].weights)?;
            let shared = leases
                .shared_output
                .as_ref()
                .context("Coder shared output")?
                .narrow(0, t, 1)?;
            let out = math::bf16(&routed.add(&shared)?)?.reshape(5120)?;
            row.finish_ffn(&out)?;
        }
        observe_tile_owners(rows, leases, plan)
    }

    /// Every routed expert of a Coder tile in three dispatches (gate, up,
    /// down). Returns down rows grouped by expert in ID order, or `None` when
    /// the tiled kernels are unavailable; nothing is enqueued in that case.
    fn coder_experts_tiled(
        groups: &[Vec<(usize, usize)>; 192],
        leases: &mut MoeLeases,
    ) -> Result<Option<Tensor>> {
        let Some(input) = leases.shared_input.clone() else {
            return Ok(None);
        };
        let selected = leases.selected.as_ref().context("Coder selected experts")?;
        let mut experts = Vec::new();
        let mut token_rows = Vec::new();
        let mut mid_rows = Vec::new();
        let mut offset = 0u32;
        for (id, group) in groups.iter().enumerate().filter(|(_, v)| !v.is_empty()) {
            experts.push(selected.get(id).context("missing Coder expert")?.clone());
            token_rows.push(group.iter().map(|&(t, _)| t as u32).collect::<Vec<_>>());
            mid_rows.push((offset..offset + group.len() as u32).collect::<Vec<_>>());
            offset += group.len() as u32;
        }
        let gates = experts.iter().map(|e| e.gate()).collect::<Vec<_>>();
        let ups = experts.iter().map(|e| e.up()).collect::<Vec<_>>();
        let downs = experts.iter().map(|e| e.down()).collect::<Vec<_>>();
        // Check all three before enqueueing any of them.
        let uniform = |m: &[&Linear]| {
            m.iter().all(|x| {
                x.dtype() == m[0].dtype()
                    && x.in_dim() == m[0].in_dim()
                    && x.out_dim() == m[0].out_dim()
            })
        };
        if !(uniform(&gates) && uniform(&ups) && uniform(&downs)) {
            return Ok(None);
        }
        let Some(gate) = Linear::forward_experts_prefill(&gates, &token_rows, &input)? else {
            return Ok(None);
        };
        let gate = leases.routed_post.pin(gate)?;
        let up = Linear::forward_experts_prefill(&ups, &token_rows, &input)?
            .context("Coder tiled up projection unavailable after gate")?;
        let up = leases.routed_post.pin(up)?;
        let mid = leases.routed_post.pin(swiglu(&gate, &up)?)?;
        let down = Linear::forward_experts_prefill(&downs, &mid_rows, &mid)?
            .context("Coder tiled down projection unavailable after gate")?;
        Ok(Some(leases.routed_post.pin(down)?))
    }

    #[cfg(target_os = "macos")]
    fn moe_native(
        &self,
        rows: &mut [TileRow],
        routes: &prefill_mm::ValidatedRoutes,
        leases: &mut MoeLeases,
        device: &Device,
        plan: profile::PrefillTilePlan,
    ) -> Result<()> {
        let selected = leases.selected.as_ref().context("native selected leases")?;
        let select = |expert, kind| -> Result<&Linear> {
            let e = selected
                .get(expert)
                .context("native missing selected expert")?;
            Ok(match kind {
                0 => e.gate(),
                1 => e.up(),
                2 => e.down(),
                _ => unreachable!(),
            })
        };
        let gate = leases.routed_post.pin(prefill_mm::indexed(
            leases
                .shared_input
                .as_ref()
                .context("native shared input")?,
            routes,
            false,
            select,
        )?)?;
        let activation = packet_swiglu(&gate, rows.len(), &mut leases.routed_post)?;
        let down =
            leases
                .routed_post
                .pin(prefill_mm::indexed(&activation, routes, true, select)?)?;
        let combined = leases
            .routed_post
            .pin(prefill_mm::combine(&down, routes.tensors()[0])?)?;
        for (t, row) in rows.iter_mut().enumerate() {
            let shared = leases
                .shared_output
                .as_ref()
                .context("native shared output")?
                .narrow(0, t, 1)?;
            let out = math::bf16(&combined.narrow(0, t, 1)?.add(&shared)?)?.reshape(5120)?;
            row.finish_ffn(&out)?;
        }
        observe_tile_owners(rows, leases, plan)?;
        let _ = device;
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn moe_packet(
        &self,
        rows: &mut [TileRow],
        routes: &[math::RouterSelection],
        packets: &iq1_packets::Plan,
        leases: &mut MoeLeases,
        device: &Device,
        plan: profile::PrefillTilePlan,
    ) -> Result<()> {
        let selected = leases.selected.as_ref().context("packet selected leases")?;
        let select = |expert, kind| -> Result<&Linear> {
            let e = selected.get(expert).context("packet missing expert")?;
            Ok(match kind {
                0 => e.gate(),
                1 => e.up(),
                2 => e.down(),
                _ => unreachable!(),
            })
        };
        let stamp = profile::start(device, Region::RoutedExperts)?;
        let gate_up = iq1_packets::forward(
            leases
                .shared_input
                .as_ref()
                .context("packet shared input")?,
            packets,
            iq1_packets::Phase::GateUp,
            select,
        )?;
        let gate_up = leases.routed_post.pin(gate_up)?;
        let activation = packet_swiglu(&gate_up, rows.len(), &mut leases.routed_post)?;
        let down = iq1_packets::forward(&activation, packets, iq1_packets::Phase::Down, select)?;
        let down = leases.routed_post.pin(down)?;
        profile::finish(device, stamp)?;
        observe_tile_owners(rows, leases, plan)?;
        for (t, route) in routes.iter().enumerate() {
            let stamp = profile::start(device, Region::FfnPost)?;
            let downs = (0..6)
                .map(|s| down.narrow(0, 6 * t + s, 1))
                .collect::<candle_core::Result<Vec<_>>>()?;
            let downs: [Tensor; 6] = downs
                .try_into()
                .map_err(|_| anyhow::anyhow!("packet six route rows"))?;
            let routed = route_combine::ordered(&downs, &route.weights)?;
            let shared = leases
                .shared_output
                .as_ref()
                .context("packet shared output")?
                .narrow(0, t, 1)?;
            let out = math::bf16(&routed.add(&shared)?)?.reshape(5120)?;
            rows[t].finish_ffn(&out)?;
            observe_tile_owners(rows, leases, plan)?;
            profile::finish(device, stamp)?;
        }
        Ok(())
    }
}

fn routed_experts(
    leases: &mut MoeLeases,
    route: &math::RouterSelection,
    grouped_iq1: bool,
    bounded_tile: bool,
) -> Result<Tensor> {
    let x = leases
        .input
        .as_ref()
        .context("missing Maya routed input owner")?;
    let selected = leases
        .selected
        .as_ref()
        .context("missing Maya selected owners")?;
    routed_with_owners(
        x,
        selected,
        &mut leases.gate_up,
        route,
        grouped_iq1,
        bounded_tile,
        &mut leases.routed_post,
    )
}

fn routed_with_owners(
    x: &Tensor,
    selected: &SelectedExperts,
    gate_up: &mut Option<Tensor>,
    route: &math::RouterSelection,
    grouped_iq1: bool,
    bounded_tile: bool,
    _post: &mut RoutedPostLeases,
) -> Result<Tensor> {
    if grouped_iq1 {
        let experts = route.indices.map(|id| selected.get(id));
        ensure!(
            experts.iter().all(Option::is_some),
            "missing Maya selected expert lease"
        );
        let matrices: [&Linear; 12] = std::array::from_fn(|slot| {
            let expert = experts[slot / 2].unwrap();
            if slot % 2 == 0 {
                expert.gate()
            } else {
                expert.up()
            }
        });
        // No implicit error fallback: enabled failures reach the final fence.
        let backing = Linear::forward_grouped_iq1(x, &matrices)?;
        *gate_up = Some(if bounded_tile {
            cache::retain_bounded(backing)?
        } else {
            backing
        });
    }
    #[cfg(test)]
    if bounded_tile
        && grouped_iq1
        && ROUTED_POST_BATCH.with(|v| v.get())
        && x.device().is_metal()
        && x.dtype() == DType::F32
        && x.is_contiguous()
        && x.dims() == [1, 5120]
        && route.indices.iter().all(|&id| {
            selected.get(id).is_some_and(|e| {
                e.down().dtype() == crate::deepseek4_gguf::DeepSeek4DType::IQ1S
                    && e.down().in_dim() == 2304
                    && e.down().out_dim() == 5120
            })
        })
    {
        // Sources/input/selected weights and grouped gate/up are already pinned.
        // All six rows remain in route order, including repeated expert IDs.
        let mid = routed_post_swiglu(gate_up.as_ref().context("missing grouped gate/up")?, _post)?;
        let matrices: [&Linear; 6] =
            std::array::from_fn(|slot| selected.get(route.indices[slot]).unwrap().down());
        let backing = _post.pin(Linear::forward_grouped_iq1_down(&mid, &matrices)?)?;
        let downs = (0..6)
            .map(|slot| backing.narrow(0, slot, 1))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let downs: [Tensor; 6] = downs.try_into().expect("six route rows");
        return Ok(route_combine::ordered(&downs, &route.weights)?);
    }
    // Six bounded F32 outputs (120 KiB), retained in selected route order.
    let mut downs: [Option<Tensor>; 6] = std::array::from_fn(|_| None);
    for (slot, &id) in route.indices.iter().enumerate() {
        let expert = selected
            .get(id)
            .context("missing Maya selected expert lease")?;
        let (gate, up) = if let Some(backing) = gate_up.as_ref() {
            (
                backing.narrow(0, 2 * slot, 1)?,
                backing.narrow(0, 2 * slot + 1, 1)?,
            )
        } else {
            (expert.gate().forward(x)?, expert.up().forward(x)?)
        };
        let mid = swiglu(&gate, &up)?;
        downs[slot] = Some(expert.down().forward(&mid)?);
    }
    let downs = downs.map(|down| down.expect("all six fixed Maya routes produced an output"));
    Ok(route_combine::ordered(&downs, &route.weights)?)
}

impl ModelArchitecture for DeepSeek41Stage {
    fn fixed_cache_bytes(&self, rows: usize) -> Result<usize> {
        ensure!(rows == 0, "Maya speculative capture is unsupported");
        Ok(FIXED_CACHE_BYTES)
    }
    fn forward(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.run(input, cache, seq_len, start_pos, true)
    }
    fn forward_prefill_cache_only(
        &self,
        input: StageInput,
        cache: &mut [Option<LayerCache>],
        seq_len: usize,
        start_pos: usize,
    ) -> Result<StageOutput> {
        self.run(input, cache, seq_len, start_pos, false)
    }
    fn family(&self) -> &str {
        "deepseek41"
    }
    fn hidden_dim(&self) -> usize {
        5120
    }
    fn layer_count(&self) -> usize {
        40
    }
    fn vocab_size(&self) -> usize {
        129280
    }
}

#[cfg(test)]
mod model_tests;

#[cfg(test)]
mod attention_v2_tests;

#[cfg(test)]
thread_local! {
    static SHARED_TILE_REFERENCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[inline]
fn batched_shared_enabled() -> bool {
    #[cfg(test)]
    if SHARED_TILE_REFERENCE.with(|reference| reference.get()) {
        return false;
    }
    true
}
