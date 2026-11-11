//! IQ1_S routed-expert packets for bounded Metal prefill.
//!
//! A prefill tile routes up to `MAX_ROWS` tokens to six experts each. Packets
//! group route occurrences that share an expert so each packed weight matrix is
//! read once per group; every plan is validated before any encoder is acquired.
use super::*;
use anyhow::Context;
use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, DType, Layout, MetalStorage, Shape, Storage};
use candle_metal_kernels::metal::{Buffer, ComputePipeline};
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineState, MTLDevice, MTLGPUFamily, MTLLanguageVersion,
    MTLMathMode, MTLResource, MTLSize,
};
use std::collections::{BTreeMap, HashMap};
use std::sync::{LazyLock, Mutex};

const PACKED: usize = 2_304_000;
const MAX_ROWS: usize = 128;
const GROUP: usize = 12; // WEIGHT matrices, not experts.
const MSL: &str = include_str!("deepseek4_iq1_packets.metal");
const MMA_SCRATCH_BYTES: usize = 4 * 4 * 64 * 4;
const MMA_SOURCE_MARKER: &str = "// Adapted from ggml-org/llama.cpp";

// The SIMD-group matrix variant is test-only and never selected in release
// builds. Thread-local and !Send: parallel tests and nested model scopes cannot
// change another thread's dispatch policy.
#[cfg(test)]
thread_local! {
    static MATRIX_TEST_SELECTION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static MATRIX_TEST_COUNTS: std::cell::Cell<[usize;3]> = const { std::cell::Cell::new([0;3]) };
    static MATRIX_TEST_UNSUPPORTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[cfg(test)]
pub(crate) struct MatrixTestScope {
    previous: bool,
    counts: [usize; 3],
    unsupported: bool,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
#[cfg(test)]
impl MatrixTestScope {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            previous: MATRIX_TEST_SELECTION.with(|v| v.replace(enabled)),
            counts: MATRIX_TEST_COUNTS.with(|v| v.replace([0; 3])),
            unsupported: MATRIX_TEST_UNSUPPORTED.with(|v| v.replace(false)),
            _thread: std::marker::PhantomData,
        }
    }
    /// Exercise capability rejection independently of the physical test GPU.
    pub(crate) fn unsupported() -> Self {
        let scope = Self::new(true);
        MATRIX_TEST_UNSUPPORTED.with(|v| v.set(true));
        scope
    }
    /// Enqueued [matrix groups, scalar groups, requested-but-unsupported groups].
    /// Scalar N1 is not an unsupported group. GPU tile recompute remains internal.
    pub(crate) fn dispatch_counts(&self) -> [usize; 3] {
        MATRIX_TEST_COUNTS.with(|v| v.get())
    }
}
#[cfg(test)]
impl Drop for MatrixTestScope {
    fn drop(&mut self) {
        MATRIX_TEST_SELECTION.with(|v| v.set(self.previous));
        MATRIX_TEST_COUNTS.with(|v| v.set(self.counts));
        MATRIX_TEST_UNSUPPORTED.with(|v| v.set(self.unsupported));
    }
}
pub(crate) fn matrix_requested() -> bool {
    #[cfg(test)]
    {
        MATRIX_TEST_SELECTION.with(|v| v.get())
    }
    #[cfg(not(test))]
    {
        false
    }
}
fn matrix_supported(device: &candle_core::MetalDevice, k: usize, out: usize) -> bool {
    #[cfg(test)]
    if MATRIX_TEST_UNSUPPORTED.with(|v| v.get()) {
        return false;
    }
    k.is_multiple_of(256)
        && out > 0
        && device
            .metal_device()
            .as_ref()
            .supportsFamily(MTLGPUFamily::Apple7)
        && device.metal_device().as_ref().maxThreadgroupMemoryLength() >= MMA_SCRATCH_BYTES
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Occurrence {
    token: usize,
    slot: usize,
}
#[derive(Clone, Debug)]
struct Packet {
    expert: usize,
    occurrences: Vec<Occurrence>,
}
#[derive(Clone, Debug)]
pub(crate) struct Plan {
    routes: Vec<[usize; 6]>,
    packets: [Vec<Packet>; 4],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    GateUp,
    Down,
}
impl Phase {
    fn geometry(self, t: usize) -> (usize, usize, usize, usize) {
        match self {
            Self::GateUp => (t, 5120, 12 * t, 2304),
            Self::Down => (6 * t, 2304, 6 * t, 5120),
        }
    }
    fn kinds(self) -> &'static [usize] {
        match self {
            Self::GateUp => &[0, 1],
            Self::Down => &[2],
        }
    }
    fn rows(self, o: Occurrence, kind: usize) -> (usize, usize) {
        match self {
            Self::GateUp => (o.token, 12 * o.token + 2 * o.slot + kind),
            Self::Down => (6 * o.token + o.slot, 6 * o.token + o.slot),
        }
    }
}
impl Plan {
    pub(crate) fn new(routes: &[[usize; 6]]) -> Result<Self> {
        anyhow::ensure!(
            (1..=MAX_ROWS).contains(&routes.len()),
            "packet T must be 1..={MAX_ROWS}"
        );
        let mut experts: BTreeMap<usize, Vec<Occurrence>> = BTreeMap::new();
        for (token, route) in routes.iter().enumerate() {
            for (slot, &expert) in route.iter().enumerate() {
                anyhow::ensure!(expert < 384, "packet expert out of range");
                experts
                    .entry(expert)
                    .or_default()
                    .push(Occurrence { token, slot });
            }
        }
        let mut packets: [Vec<Packet>; 4] = std::array::from_fn(|_| Vec::new());
        for (expert, occurrences) in experts {
            for chunk in occurrences.chunks(4) {
                packets[chunk.len() - 1].push(Packet {
                    expert,
                    occurrences: chunk.to_vec(),
                });
            }
        }
        let plan = Self {
            routes: routes.to_vec(),
            packets,
        };
        plan.validate()?;
        Ok(plan)
    }
    fn validate(&self) -> Result<()> {
        let t = self.routes.len();
        anyhow::ensure!((1..=MAX_ROWS).contains(&t), "packet invalid T");
        let count = t.checked_mul(6).context("packet occurrence overflow")?;
        let mut seen = vec![false; count];
        for (i, bucket) in self.packets.iter().enumerate() {
            let mut last = None;
            for packet in bucket {
                anyhow::ensure!(
                    packet.expert < 384 && packet.occurrences.len() == i + 1,
                    "packet arity/expert mismatch"
                );
                let first = packet.occurrences[0];
                let key = (packet.expert, first.token, first.slot);
                anyhow::ensure!(
                    last.is_none_or(|old| old < key),
                    "packet nondeterministic order"
                );
                last = Some(key);
                let mut previous = None;
                for &o in &packet.occurrences {
                    anyhow::ensure!(o.token < t && o.slot < 6, "packet occurrence out of range");
                    anyhow::ensure!(
                        previous.is_none_or(|old| old < (o.token, o.slot)),
                        "packet occurrence order"
                    );
                    previous = Some((o.token, o.slot));
                    anyhow::ensure!(
                        self.routes[o.token][o.slot] == packet.expert,
                        "packet stale expert"
                    );
                    let index = o
                        .token
                        .checked_mul(6)
                        .and_then(|v| v.checked_add(o.slot))
                        .context("packet index overflow")?;
                    anyhow::ensure!(!seen[index], "packet duplicate destination");
                    seen[index] = true;
                }
            }
        }
        anyhow::ensure!(seen.iter().all(|&v| v), "packet incomplete output coverage");
        Ok(())
    }
}

/// Canonical rounded physical carriers, not a guessed per-row charge.
pub(crate) fn carrier_bytes(t: usize) -> Result<usize> {
    anyhow::ensure!(
        (1..=MAX_ROWS).contains(&t),
        "packet retained T out of range"
    );
    let rounded = |rows: usize, width: usize| -> Result<usize> {
        rows.checked_mul(t)
            .and_then(|v| v.checked_mul(width))
            .and_then(|v| v.checked_mul(4))
            .and_then(usize::checked_next_power_of_two)
            .context("packet carrier overflow")
    };
    rounded(12, 2304)?
        .checked_add(
            rounded(6, 2304)?
                .checked_mul(4)
                .context("packet activation overflow")?,
        )
        .and_then(|v| v.checked_add(rounded(6, 5120).ok()?))
        .context("packet retained overflow")
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Indices {
    source: [u32; 4],
    destination: [u32; 4],
}
struct MatrixPacket {
    weights: Arc<RawTensor>,
    indices: Indices,
}
struct PacketOp {
    plan: Plan,
    phase: Phase,
    buckets: [Vec<MatrixPacket>; 4],
}

pub(crate) fn forward<'a>(
    input: &Tensor,
    plan: &Plan,
    phase: Phase,
    mut select: impl FnMut(usize, usize) -> Result<&'a DeepSeek4QMatMul>,
) -> Result<Tensor> {
    plan.validate()?;
    let (input_rows, k, _, out) = phase.geometry(plan.routes.len());
    anyhow::ensure!(
        input.dims() == [input_rows, k] && input.dtype() == DType::F32 && input.is_contiguous(),
        "packet requires exact contiguous native F32 input"
    );
    anyhow::ensure!(input.device().is_metal(), "packet requires Metal backend");
    let mut buckets: [Vec<MatrixPacket>; 4] = std::array::from_fn(|_| Vec::new());
    for (n, bucket) in plan.packets.iter().enumerate() {
        for packet in bucket {
            for &kind in phase.kinds() {
                let matrix = select(packet.expert, kind)?;
                let weights = &matrix.weights;
                anyhow::ensure!(
                    weights.shape() == [out, k]
                        && weights.dtype() == DeepSeek4DType::IQ1S
                        && weights.byte_len() == PACKED,
                    "packet exact native IQ1_S matrix required"
                );
                let bytes = weights.bytes();
                anyhow::ensure!(
                    bytes.rank() == 1
                        && bytes.dims() == [PACKED]
                        && bytes.dtype() == DType::U8
                        && bytes.is_contiguous(),
                    "packet requires complete contiguous packed U8 vector"
                );
                anyhow::ensure!(
                    input.device().same_device(bytes.device()),
                    "packet Candle device mismatch"
                );
                let mut indices = Indices::default();
                for (i, &o) in packet.occurrences.iter().enumerate() {
                    let (src, dst) = phase.rows(o, kind);
                    indices.source[i] = src.try_into()?;
                    indices.destination[i] = dst.try_into()?;
                }
                buckets[n].push(MatrixPacket {
                    weights: weights.clone(),
                    indices,
                });
            }
        }
    }
    let op = PacketOp {
        plan: plan.clone(),
        phase,
        buckets,
    };
    Ok(input.apply_op1_no_bwd(&op)?)
}

fn pipeline(
    device: &candle_core::MetalDevice,
    n: usize,
    matrix: bool,
) -> candle_core::Result<ComputePipeline> {
    type PipelineCache =
        HashMap<(candle_core::metal_backend::DeviceId, usize, bool), ComputePipeline>;
    static CACHE: LazyLock<Mutex<PipelineCache>> = LazyLock::new(|| Mutex::new(HashMap::new()));
    let mut cache = CACHE
        .lock()
        .map_err(|_| candle_core::Error::Msg("packet pipeline poisoned".into()))?;
    if let Some(p) = cache.get(&(device.id(), n, matrix)) {
        return Ok(p.clone());
    }
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version4_0);
    // Candidate only; the established scalar library/options are unchanged.
    if matrix {
        options.setMathMode(MTLMathMode::Safe);
    }
    // Do not compile matrix code at all for the established scalar path.
    let scalar = MSL.split_once(MMA_SOURCE_MARKER).map_or(MSL, |(s, _)| s);
    let source = format!(
        "{}\n{}",
        super::metal::iq1_reuse_base_msl(),
        if matrix { MSL } else { scalar }
    );
    let library = device
        .metal_device()
        .new_library_with_source(&source, Some(&options))
        .map_err(candle_core::Error::wrap)?;
    let name = if matrix {
        format!("dinf_iq1_packet_mma_n{n}")
    } else {
        format!("dinf_iq1_packet_n{n}")
    };
    let function = library
        .get_function(&name, None)
        .map_err(candle_core::Error::wrap)?;
    let p = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    if p.as_ref().threadExecutionWidth() != 32 || p.max_total_threads_per_threadgroup() < 128 {
        candle_core::bail!("packet requires SIMD32 and128 threads");
    }
    if matrix
        && p.as_ref().staticThreadgroupMemoryLength() + MMA_SCRATCH_BYTES
            > device.metal_device().as_ref().maxThreadgroupMemoryLength()
    {
        candle_core::bail!("packet matrix threadgroup scratch unsupported");
    }
    cache.insert((device.id(), n, matrix), p.clone());
    Ok(p)
}
fn actual_device(buffer: &Buffer, device: &candle_core::MetalDevice) -> bool {
    std::ptr::eq(
        buffer.as_ref().device().as_ref(),
        device.metal_device().as_ref(),
    )
}
struct PreparedGroup {
    pipeline: ComputePipeline,
    matrix: bool,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Retains the preflight decision for test dispatch counters."
        )
    )]
    unsupported: bool,
    weights: Vec<(Buffer, usize)>,
    indices: [Indices; GROUP],
    count: usize,
}
impl CustomOp1 for PacketOp {
    fn name(&self) -> &'static str {
        "dinf-iq1-packet"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("packet requires Metal")
    }
    fn metal_fwd(
        &self,
        input: &MetalStorage,
        layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        self.plan.validate().map_err(candle_core::Error::wrap)?;
        let (input_rows, k, output_rows, out) = self.phase.geometry(self.plan.routes.len());
        if input.dtype() != DType::F32
            || layout.dims() != [input_rows, k]
            || !actual_device(input.buffer(), input.device())
        {
            candle_core::bail!("packet input storage identity/shape/dtype mismatch");
        }
        let input_offset =
            grouped_iq1_span(layout, DType::F32, input_rows * k, input.buffer().length())
                .map_err(candle_core::Error::wrap)?;
        let mut seen = vec![false; output_rows];
        let mut prepared = Vec::new();
        for (bucket_index, bucket) in self.buckets.iter().enumerate() {
            let arity = bucket_index + 1;
            let expected = self.plan.packets[bucket_index].len() * self.phase.kinds().len();
            if bucket.len() != expected {
                candle_core::bail!("packet stale phase/count");
            }
            if bucket.is_empty() {
                continue;
            }
            // Preselect capability, geometry and every pipeline before an encoder
            // exists. A compile/device rejection uses the original kernel only;
            // no fallback can occur after a partial enqueue.
            let requested = matrix_requested() && arity > 1;
            let candidate = requested && matrix_supported(input.device(), k, out);
            let selected = if candidate {
                pipeline(input.device(), arity, true).ok()
            } else {
                None
            };
            let matrix = selected.is_some();
            let p = match selected {
                Some(p) => p,
                None => pipeline(input.device(), arity, false)?,
            };
            for (group_index, chunk) in bucket.chunks(GROUP).enumerate() {
                let mut group = PreparedGroup {
                    pipeline: p.clone(),
                    matrix,
                    unsupported: requested && !matrix,
                    weights: Vec::with_capacity(chunk.len()),
                    indices: [Indices::default(); GROUP],
                    count: chunk.len(),
                };
                if chunk.is_empty() || chunk.len() > GROUP || 4 + GROUP > 31 || out % 4 != 0 {
                    candle_core::bail!("packet dispatch geometry/binding limit");
                }
                for (i, packet) in chunk.iter().enumerate() {
                    let ordinal = group_index * GROUP + i;
                    let expected_packet =
                        &self.plan.packets[bucket_index][ordinal / self.phase.kinds().len()];
                    let kind = self.phase.kinds()[ordinal % self.phase.kinds().len()];
                    for j in 0..arity {
                        let (src, dst) = self.phase.rows(expected_packet.occurrences[j], kind);
                        if packet.indices.source[j] as usize != src
                            || packet.indices.destination[j] as usize != dst
                            || src >= input_rows
                            || dst >= output_rows
                            || seen[dst]
                        {
                            candle_core::bail!(
                                "packet invalid indexed source/destination or coverage"
                            );
                        }
                        seen[dst] = true;
                    }
                    if packet.indices.source[arity..].iter().any(|&v| v != 0)
                        || packet.indices.destination[arity..].iter().any(|&v| v != 0)
                    {
                        candle_core::bail!("packet invalid inactive metadata");
                    }
                    let weights = &packet.weights;
                    if weights.shape() != [out, k]
                        || weights.dtype() != DeepSeek4DType::IQ1S
                        || weights.byte_len() != PACKED
                    {
                        candle_core::bail!("packet stale weight geometry");
                    }
                    let (storage, wl) = weights.bytes().storage_and_layout();
                    let Storage::Metal(storage) = &*storage else {
                        candle_core::bail!("packet non-Metal weights")
                    };
                    if storage.dtype() != DType::U8
                        || storage.device().id() != input.device().id()
                        || wl.dims() != [PACKED]
                        || !actual_device(storage.buffer(), input.device())
                    {
                        candle_core::bail!("packet weight storage identity/dtype/rank mismatch");
                    }
                    let offset = grouped_iq1_span(wl, DType::U8, PACKED, storage.buffer().length())
                        .map_err(candle_core::Error::wrap)?;
                    group.weights.push((storage.buffer().clone(), offset));
                    group.indices[i] = packet.indices;
                }
                prepared.push(group);
            }
        }
        if !seen.iter().all(|&v| v) {
            candle_core::bail!("packet incomplete canonical writes");
        }
        let elements = output_rows
            .checked_mul(out)
            .ok_or_else(|| candle_core::Error::Msg("packet output overflow".into()))?;
        let shape = Shape::from((output_rows, out));
        let geometry = [k as u32, out as u32, (k / 256 * 50) as u32, 0];
        let output = input
            .device()
            .new_buffer_builder()
            .with_size_for(elements, DType::F32)
            .with_label("dinf-iq1-packet")
            .build()?;
        if output.length() < elements * 4 || !actual_device(&output, input.device()) {
            candle_core::bail!("packet output storage mismatch");
        }
        #[cfg(test)]
        tests::ENCODER_PROBE.with(|probe| {
            if let Some(n) = probe.get() {
                probe.set(Some(n + 1));
            }
        });
        let encoder = input.device().command_encoder()?;
        let encoder = encoder.as_ref();
        // ALL Result-producing work is above this line. One encoder/output for
        // the whole phase. Only direct prevalidated bindings and void dispatches.
        // Metal buffer bindings persist across pipeline changes. Validation proves
        // disjoint canonical rows, so bind once without buffer-wide group WAWs.
        // Keep tracked output registration for the following activation's RAW.
        encoder.set_input_buffer(0, Some(input.buffer()), input_offset);
        encoder.set_output_buffer(1, Some(&output), 0);
        for group in &prepared {
            encoder.set_compute_pipeline_state(&group.pipeline);
            encoder.set_bytes(2, &group.indices);
            encoder.set_bytes(3, &geometry);
            for i in 0..GROUP {
                let (buffer, offset) = &group.weights[i.min(group.count - 1)];
                encoder.set_input_buffer(4 + i, Some(buffer), *offset);
            }
            if group.matrix {
                encoder.set_threadgroup_memory_length(0, MMA_SCRATCH_BYTES);
            }
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: out.div_ceil(if group.matrix { 32 } else { 4 }),
                    height: group.count,
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
            #[cfg(test)]
            MATRIX_TEST_COUNTS.with(|v| {
                let mut counts = v.get();
                counts[if group.matrix { 0 } else { 1 }] += 1;
                counts[2] += usize::from(group.unsupported);
                v.set(counts);
            });
        }
        Ok((
            MetalStorage::new(output, input.device().clone(), elements, DType::F32),
            shape,
        ))
    }
}

#[cfg(test)]
#[path = "deepseek4_iq1_packet_tests.rs"]
mod tests;
