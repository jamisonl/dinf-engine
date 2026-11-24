use crate::metal::{
    BlitCommandEncoder, Buffer, CommandBuffer, ComputeCommandEncoder, ComputePipeline, Device,
    Fence, PrevCeOutputs, ResidencySet,
};
use crate::MetalKernelError;
use block2::RcBlock;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

// Use Retained when appropriate. Gives us a more elegant way of handling memory (peaks) than autoreleasepool.
// https://docs.rs/objc2/latest/objc2/rc/struct.Retained.html
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

static TIMING_ENABLED: AtomicBool = AtomicBool::new(false);
static TIMING_DRAINED_BUFFERS: AtomicU64 = AtomicU64::new(0);
static TIMING_VALID_TIMESTAMP_BUFFERS: AtomicU64 = AtomicU64::new(0);
static TIMING_ZERO_TIMESTAMP_BUFFERS: AtomicU64 = AtomicU64::new(0);
static TIMING_INVALID_TIMESTAMP_BUFFERS: AtomicU64 = AtomicU64::new(0);
static TIMING_GPU_ELAPSED_NS: AtomicU64 = AtomicU64::new(0);

/// Process-global observation across all devices while enabled (disabled by default).
///
/// Callers must serialize enable/reset/snapshot with GPU requests. Snapshots are
/// not guaranteed consistent under concurrent mutation. Only fully successful
/// drains contribute; a failed drain contributes nothing, including earlier
/// successful buffers in that drain. All counters saturate at `u64::MAX`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CommandBufferTimingSnapshot {
    pub enabled: bool,
    pub drained_buffers: u64,
    pub valid_timestamp_buffers: u64,
    /// Finite, nonnegative timestamps with either endpoint zero (missing).
    pub zero_timestamp_buffers: u64,
    /// Other invalid timestamps: nonfinite, negative, reversed, or overflowing.
    pub invalid_timestamp_buffers: u64,
    /// Sum of command-buffer GPU elapsed durations, truncated to nanoseconds per
    /// buffer. May include GPU fence/resource waits: not pure active kernel time
    /// or a per-region split, nor an exact busy-time union if intervals overlap.
    /// Do not add this to overlapping host-wall totals.
    pub gpu_elapsed_ns: u64,
}

/// Enable/disable process-global timing without resetting counters.
/// Callers must serialize this with GPU requests and reset/snapshot calls.
pub fn set_command_buffer_timing_enabled(enabled: bool) {
    TIMING_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Reset all timing counters without changing enabled.
/// Callers must serialize this with GPU requests and enable/snapshot calls.
pub fn reset_command_buffer_timing() {
    TIMING_DRAINED_BUFFERS.store(0, Ordering::Relaxed);
    TIMING_VALID_TIMESTAMP_BUFFERS.store(0, Ordering::Relaxed);
    TIMING_ZERO_TIMESTAMP_BUFFERS.store(0, Ordering::Relaxed);
    TIMING_INVALID_TIMESTAMP_BUFFERS.store(0, Ordering::Relaxed);
    TIMING_GPU_ELAPSED_NS.store(0, Ordering::Relaxed);
}

/// Read the process-global observation across all devices.
/// Callers must serialize this with GPU requests and enable/reset calls;
/// concurrent mutation can produce an inconsistent snapshot.
pub fn command_buffer_timing_snapshot() -> CommandBufferTimingSnapshot {
    CommandBufferTimingSnapshot {
        enabled: TIMING_ENABLED.load(Ordering::Relaxed),
        drained_buffers: TIMING_DRAINED_BUFFERS.load(Ordering::Relaxed),
        valid_timestamp_buffers: TIMING_VALID_TIMESTAMP_BUFFERS.load(Ordering::Relaxed),
        zero_timestamp_buffers: TIMING_ZERO_TIMESTAMP_BUFFERS.load(Ordering::Relaxed),
        invalid_timestamp_buffers: TIMING_INVALID_TIMESTAMP_BUFFERS.load(Ordering::Relaxed),
        gpu_elapsed_ns: TIMING_GPU_ELAPSED_NS.load(Ordering::Relaxed),
    }
}

fn saturating_add_timing(counter: &AtomicU64, value: u64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_add(value))
    });
}

#[derive(Debug, PartialEq, Eq)]
enum GpuTimestampElapsed {
    Valid(u64),
    Zero,
    Invalid,
}

fn gpu_timestamp_elapsed_ns(start: f64, end: f64) -> GpuTimestampElapsed {
    if !start.is_finite() || !end.is_finite() || start < 0.0 || end < 0.0 {
        return GpuTimestampElapsed::Invalid;
    }
    if start == 0.0 || end == 0.0 {
        return GpuTimestampElapsed::Zero;
    }
    if end < start {
        return GpuTimestampElapsed::Invalid;
    }
    let elapsed_ns = (end - start) * 1_000_000_000.0;
    // u64::MAX rounds UP to 2^64 as f64: use an exclusive upper bound
    // before casting, rather than Rust's saturating float-to-integer conversion.
    if !elapsed_ns.is_finite() || elapsed_ns >= u64::MAX as f64 {
        return GpuTimestampElapsed::Invalid;
    }
    GpuTimestampElapsed::Valid(elapsed_ns as u64)
}

fn create_command_buffer(command_queue: &CommandQueue) -> Result<CommandBuffer, MetalKernelError> {
    command_queue.commandBuffer().map(CommandBuffer::new).ok_or(
        MetalKernelError::FailedToCreateResource("CommandBuffer".to_string()),
    )
}

/// RAII guard for compute command encoder operations.
pub struct CommandsGuard<'a> {
    guard: MutexGuard<'a, EntryState>,
}

impl AsRef<ComputeCommandEncoder> for CommandsGuard<'_> {
    fn as_ref(&self) -> &ComputeCommandEncoder {
        self.guard.current_encoder.as_ref().unwrap()
    }
}

impl CommandsGuard<'_> {
    pub fn set_label(&self, label: &str) {
        self.as_ref().set_label(label);
    }

    pub fn set_compute_pipeline_state(&self, pipeline: &ComputePipeline) {
        self.as_ref().set_compute_pipeline_state(pipeline);
    }

    #[cfg(feature = "debug-labels")]
    #[must_use = "the debug group is popped when the returned guard is dropped"]
    pub fn debug_group(&self, label: &str) -> crate::metal::DebugGroupGuard<'_> {
        self.as_ref().debug_group(label)
    }
}

/// RAII guard for blit command encoder operations.
pub struct BlitCommandsGuard<'a> {
    _guard: MutexGuard<'a, EntryState>,
    state: BlitCommandEncoder,
}

impl<'a> AsRef<BlitCommandEncoder> for BlitCommandsGuard<'a> {
    fn as_ref(&self) -> &BlitCommandEncoder {
        &self.state
    }
}

impl<'a> AsMut<BlitCommandEncoder> for BlitCommandsGuard<'a> {
    fn as_mut(&mut self) -> &mut BlitCommandEncoder {
        &mut self.state
    }
}

impl BlitCommandsGuard<'_> {
    pub fn set_label(&self, label: &str) {
        self.as_ref().set_label(label);
    }

    pub fn copy_from_buffer(
        &mut self,
        src_buffer: &Buffer,
        src_offset: usize,
        dst_buffer: &Buffer,
        dst_offset: usize,
        size: usize,
    ) {
        self.as_mut()
            .copy_from_buffer(src_buffer, src_offset, dst_buffer, dst_offset, size)
    }

    pub fn fill_buffer(&mut self, buffer: &Buffer, range: (usize, usize), value: u8) {
        self.as_mut().fill_buffer(buffer, range, value);
    }
}

impl Drop for BlitCommandsGuard<'_> {
    fn drop(&mut self) {
        self.as_ref().end_encoding();
    }
}

struct EntryState {
    current: CommandBuffer,
    in_flight: Vec<CommandBuffer>,
    current_encoder: Option<ComputeCommandEncoder>,
    buffers_since_wait: usize,
}

impl EntryState {
    pub fn new(cb: CommandBuffer) -> EntryState {
        EntryState {
            current: cb,
            in_flight: vec![],
            current_encoder: None,
            buffers_since_wait: 0,
        }
    }
}

pub struct Commands {
    state: Mutex<EntryState>,
    compute_count: AtomicUsize,
    command_queue: CommandQueue,
    /// Explicit override from `CANDLE_METAL_COMPUTE_PER_BUFFER`, if set.
    /// Otherwise uses progressive ramp-up (8 -> 24 -> 64 -> 128 -> 256 -> 512) after each wait so
    /// the GPU starts quickly while keeping steady-state command buffer count low.
    compute_per_buffer_override: Option<usize>,
    device: Device,
    residency_set: ResidencySet,
    /// Global cross-encoder output map. Maps buffer pointer to the fence of the last encoder
    /// that wrote it, enabling cross-command-buffer ordering for HazardTrackingModeUntracked.
    prev_ce_outputs: PrevCeOutputs,
}

unsafe impl Send for Commands {}
unsafe impl Sync for Commands {}

impl Commands {
    pub fn new(
        command_queue: CommandQueue,
        residency_set: &ResidencySet,
    ) -> Result<Self, MetalKernelError> {
        let compute_per_buffer_override = std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER")
            .ok()
            .and_then(|val| val.parse().ok());

        if let Some(raw) = residency_set.raw() {
            command_queue.addResidencySet(raw);
            residency_set.enable_deferred_commit();
        }

        let device = Device::new(command_queue.device());
        let cb = create_command_buffer(&command_queue)?;

        Ok(Self {
            state: Mutex::new(EntryState::new(cb)),
            compute_count: AtomicUsize::new(0),
            command_queue,
            compute_per_buffer_override,
            device,
            residency_set: residency_set.clone(),
            prev_ce_outputs: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    #[inline]
    fn buffer_limit(&self, buffers_since_wait: usize) -> usize {
        if let Some(limit) = self.compute_per_buffer_override {
            return limit;
        }
        match buffers_since_wait {
            0 => 8,
            1 => 24,
            2 => 64,
            3 => 128,
            4 => 256,
            _ => 512,
        }
    }

    pub fn command_encoder(&self) -> Result<CommandsGuard<'_>, MetalKernelError> {
        let mut state_guard = self.state.lock().unwrap();
        let count = self.compute_count.fetch_add(1, Ordering::Relaxed);
        let flush = count >= self.buffer_limit(state_guard.buffers_since_wait);

        if flush {
            self.commit_swap_locked(&mut state_guard, 1)?;
        }

        if state_guard.current_encoder.is_none() {
            let fence = Arc::new(Fence::new(&self.device));
            let enc = state_guard.current.compute_command_encoder(&fence);
            // Wait for all prior encoder fences before the first dispatch.
            // Using HazardTrackingModeUntracked implies that Metal does not automatically flush GPU caches
            // at encoder or command buffer boundaries.
            {
                use std::collections::HashSet;
                let map = self.prev_ce_outputs.lock().unwrap();
                let mut seen = HashSet::new();
                for f in map.values() {
                    let ptr = Arc::as_ptr(f) as usize;
                    if seen.insert(ptr) {
                        enc.wait_for_fence(f);
                    }
                }
            }
            state_guard.current_encoder = Some(enc);
        }

        Ok(CommandsGuard { guard: state_guard })
    }

    pub fn blit_command_encoder(&self) -> Result<BlitCommandsGuard<'_>, MetalKernelError> {
        let mut state_guard = self.state.lock().unwrap();
        let count = self.compute_count.fetch_add(1, Ordering::Relaxed);
        let flush = count >= self.buffer_limit(state_guard.buffers_since_wait);

        if flush {
            self.commit_swap_locked(&mut state_guard, 1)?;
        }

        // End compute encoder before starting blit.
        if let Some(enc) = state_guard.current_encoder.take() {
            self.end_encoding(enc);
        }

        let fence = Arc::new(Fence::new(&self.device));
        let encoder = state_guard
            .current
            .blit_command_encoder(&fence, &self.prev_ce_outputs);

        // Wait for all prior encoder fences before any blit commands execute.
        // Required for HazardTrackingModeUntracked: GPU caches are not auto-flushed.
        {
            use std::collections::HashSet;
            let map = self.prev_ce_outputs.lock().unwrap();
            let mut seen = HashSet::new();
            for f in map.values() {
                let ptr = Arc::as_ptr(f) as usize;
                if seen.insert(ptr) {
                    encoder.wait_for_fence(f);
                }
            }
        }

        Ok(BlitCommandsGuard {
            _guard: state_guard,
            state: encoder,
        })
    }

    pub fn wait_until_completed(&self) -> Result<(), MetalKernelError> {
        self.flush_and_wait()
    }

    pub fn flush_and_wait(&self) -> Result<(), MetalKernelError> {
        self.flush_and_wait_current()
    }

    /// Commit the current command buffer and wait on that specific buffer.
    /// Even with no new kernels, submit a marker: another waiter may have
    /// taken the in-flight list while earlier GPU work is still running.
    /// Completion handlers remove only their own fences. Never clear the
    /// shared fence map here: it may already track another caller's new work.
    pub fn flush_and_wait_current(&self) -> Result<(), MetalKernelError> {
        let to_wait = {
            let mut state = self.state.lock()?;
            self.commit_swap_locked(&mut state, 0)?;
            state.buffers_since_wait = 0;
            std::mem::take(&mut state.in_flight)
        };

        // The queue is FIFO, so waiting for the newest buffer also completes every
        // older buffer in this snapshot. Completion does not imply success: an
        // earlier command buffer can fail while a later readback buffer completes.
        // Check every status before exposing GPU bytes to the CPU.
        if let Some(last) = to_wait.last() {
            Self::ensure_completed(last)?;
        }
        for cb in &to_wait[..to_wait.len().saturating_sub(1)] {
            Self::ensure_completed(cb)?;
        }

        // Observe each buffer exactly once, only after the entire drain succeeds.
        // No timestamp queries while disabled or for pending/failed drains.
        if TIMING_ENABLED.load(Ordering::Relaxed) {
            for cb in &to_wait {
                let raw = cb.as_ref();
                let elapsed_ns = gpu_timestamp_elapsed_ns(raw.GPUStartTime(), raw.GPUEndTime());
                saturating_add_timing(&TIMING_DRAINED_BUFFERS, 1);
                match elapsed_ns {
                    GpuTimestampElapsed::Valid(ns) => {
                        saturating_add_timing(&TIMING_VALID_TIMESTAMP_BUFFERS, 1);
                        saturating_add_timing(&TIMING_GPU_ELAPSED_NS, ns);
                    }
                    GpuTimestampElapsed::Zero => {
                        saturating_add_timing(&TIMING_ZERO_TIMESTAMP_BUFFERS, 1);
                    }
                    GpuTimestampElapsed::Invalid => {
                        saturating_add_timing(&TIMING_INVALID_TIMESTAMP_BUFFERS, 1);
                    }
                }
            }
        }

        Ok(())
    }

    pub fn flush(&self) -> Result<(), MetalKernelError> {
        let mut state = self.state.lock()?;
        if self.compute_count.load(Ordering::Acquire) > 0 {
            self.commit_swap_locked(&mut state, 0)?;
        }
        Ok(())
    }

    fn commit_swap_locked(
        &self,
        state: &mut EntryState,
        reset_to: usize,
    ) -> Result<(), MetalKernelError> {
        if let Some(enc) = state.current_encoder.take() {
            self.end_encoding(enc);
        }

        self.residency_set.commit_if_dirty();

        match state.current.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                state.current.commit();
            }
            _ => {}
        }
        let new_cb = create_command_buffer(&self.command_queue)?;
        let old_cb = std::mem::replace(&mut state.current, new_cb);
        state.in_flight.push(old_cb);
        state.buffers_since_wait = state.buffers_since_wait.saturating_add(1);
        self.compute_count.store(reset_to, Ordering::Release);

        Ok(())
    }

    fn ensure_completed(cb: &CommandBuffer) -> Result<(), MetalKernelError> {
        match cb.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                cb.commit();
                cb.wait_until_completed();
            }
            MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled => {
                cb.wait_until_completed();
            }
            MTLCommandBufferStatus::Completed => {}
            MTLCommandBufferStatus::Error => return Err(Self::cb_error(cb)),
            _ => unreachable!(),
        }

        if cb.status() == MTLCommandBufferStatus::Error {
            return Err(Self::cb_error(cb));
        }

        Ok(())
    }

    fn cb_error(cb: &CommandBuffer) -> MetalKernelError {
        let msg = cb
            .error()
            .map(|e| e.to_string())
            .unwrap_or_else(|| "unknown error".to_string());
        MetalKernelError::CommandBufferError(msg)
    }

    fn end_encoding(&self, encoder: ComputeCommandEncoder) {
        use objc2_metal::MTLCommandEncoder as _;
        use objc2_metal::MTLComputeCommandEncoder as _;

        let all_outputs = {
            let s = encoder.state.lock().unwrap();
            s.all_outputs.clone()
        };

        {
            let mut prev_ce_outputs = self.prev_ce_outputs.lock().unwrap();
            // Register our outputs so subsequent encoders can wait for us.
            for output in all_outputs.iter() {
                let _ = prev_ce_outputs.insert(*output, encoder.fence.clone());
            }
        }

        // Signal this encoder's completion fence and end encoding.
        encoder.raw.updateFence(encoder.fence.raw());

        // Schedule cleanup of our output entries once the GPU completes.
        if !all_outputs.is_empty() {
            let fence_for_cleanup = Arc::clone(&encoder.fence);
            let map_for_cleanup = Arc::clone(&self.prev_ce_outputs);
            let block = RcBlock::new(move |_cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                let mut map = map_for_cleanup.lock().unwrap();
                for &buf in &all_outputs {
                    if let Some(f) = map.get(&buf) {
                        if Arc::ptr_eq(f, &fence_for_cleanup) {
                            map.remove(&buf);
                        }
                    }
                }
            });
            unsafe {
                encoder
                    .command_buffer
                    .addCompletedHandler(RcBlock::as_ptr(&block))
            };
        }

        encoder.raw.endEncoding();
        encoder.retain_buffers_until_completed();
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::{MTLDevice as _, MTLResourceOptions, MTLSharedEvent};
    use std::time::{Duration, Instant};

    #[test]
    fn gpu_timestamps_convert_positive_ordered_seconds() {
        use GpuTimestampElapsed::Valid;
        assert_eq!(gpu_timestamp_elapsed_ns(1.0, 1.25), Valid(250_000_000));
        assert_eq!(gpu_timestamp_elapsed_ns(1.0, 1.0), Valid(0));
        assert_eq!(gpu_timestamp_elapsed_ns(1.0, 1.0 + f64::EPSILON), Valid(0));
        // A large duration below the exclusive u64 bound still converts.
        let end = 18_446_744_073.0;
        assert_eq!(
            gpu_timestamp_elapsed_ns(1.0, end),
            Valid(18_446_744_072_000_000_000)
        );
    }

    #[test]
    fn gpu_timestamps_classify_zero_or_missing_times() {
        for (start, end) in [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (-0.0, 1.0)] {
            assert_eq!(
                gpu_timestamp_elapsed_ns(start, end),
                GpuTimestampElapsed::Zero,
                "{start}..{end}"
            );
        }
    }

    #[test]
    fn gpu_timestamps_reject_missing_invalid_and_unrepresentable_times() {
        for (start, end) in [
            (0.0, f64::NAN),
            (f64::NAN, 0.0),
            (0.0, -1.0),
            (-1.0, 0.0),
            (0.0, f64::INFINITY),
            (f64::INFINITY, 0.0),
            (0.0, f64::NEG_INFINITY),
            (f64::NEG_INFINITY, 0.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (2.0, 1.0),
            (f64::NAN, 1.0),
            (1.0, f64::NAN),
            (f64::INFINITY, f64::INFINITY),
            (1.0, f64::INFINITY),
            (f64::NEG_INFINITY, 1.0),
            (1.0, f64::MAX),
            (1.0, 1.0 + (u64::MAX as f64) / 1_000_000_000.0),
        ] {
            assert_eq!(
                gpu_timestamp_elapsed_ns(start, end),
                GpuTimestampElapsed::Invalid,
                "{start}..{end}"
            );
        }
    }

    #[test]
    fn timing_addition_saturates_without_wrapping() {
        let counter = AtomicU64::new(u64::MAX - 2);
        saturating_add_timing(&counter, 1);
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX - 1);
        saturating_add_timing(&counter, 2);
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
        saturating_add_timing(&counter, u64::MAX);
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
    }

    struct ReleaseEvent(Retained<ProtocolObject<dyn MTLSharedEvent>>);

    impl Drop for ReleaseEvent {
        fn drop(&mut self) {
            // Also release blocked GPU work when a test assertion fails.
            self.0.setSignaledValue(1);
        }
    }

    fn wait_for_snapshot_taken(commands: &Commands) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let state = commands.state.lock().unwrap();
            if commands.compute_count.load(Ordering::Acquire) == 0 && state.in_flight.is_empty() {
                return;
            }
            drop(state);
            assert!(
                Instant::now() < deadline,
                "waiter did not take its command snapshot"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn dropped_private_buffer_stays_resident_until_gpu_completion() -> Result<(), MetalKernelError>
    {
        let device = Device::system_default().expect("test requires Metal");
        let residency = ResidencySet::new(&device);
        let baseline = residency.allocation_count();
        let commands = Commands::new(device.new_command_queue()?, &residency)?;
        let event = device
            .as_ref()
            .newSharedEvent()
            .expect("Metal shared event");
        let release = ReleaseEvent(event.clone());
        commands
            .state
            .lock()
            .unwrap()
            .current
            .as_ref()
            .encodeWaitForEvent_value(ProtocolObject::from_ref(&*event), 1);
        let buffer = device.new_buffer(1024 * 1024, MTLResourceOptions::StorageModePrivate)?;
        residency.insert(&buffer);
        commands
            .blit_command_encoder()?
            .fill_buffer(&buffer, (0, buffer.length()), 7);
        commands.flush()?;
        drop(buffer);
        assert_eq!(
            residency.allocation_count(),
            baseline + 1,
            "queued work must retain residency"
        );
        drop(release);
        commands.flush_and_wait_current()?;
        let deadline = Instant::now() + Duration::from_secs(5);
        while residency.allocation_count() != baseline {
            assert!(
                Instant::now() < deadline,
                "completed work retained its buffer"
            );
            std::thread::yield_now();
        }
        Ok(())
    }

    #[test]
    fn concurrent_flush_preserves_fences_for_new_work() -> Result<(), MetalKernelError> {
        check_concurrent_flush(true, false)
    }

    #[test]
    fn concurrent_flush_preserves_new_writer_for_reused_buffer() -> Result<(), MetalKernelError> {
        check_concurrent_flush(true, true)
    }

    #[test]
    fn concurrent_flush_waits_for_work_taken_by_another_waiter() -> Result<(), MetalKernelError> {
        check_concurrent_flush(false, false)
    }

    fn check_concurrent_flush(new_work: bool, reuse_buffer: bool) -> Result<(), MetalKernelError> {
        let Some(device) = Device::system_default() else {
            assert!(
                std::env::var_os("DINF_REQUIRE_METAL_TESTS").is_none(),
                "Metal device required"
            );
            return Ok(());
        };
        let residency = ResidencySet::new(&device);
        let commands = Commands::new(device.new_command_queue()?, &residency)?;
        let event = device
            .as_ref()
            .newSharedEvent()
            .expect("Metal shared event");
        let options = MTLResourceOptions(
            MTLResourceOptions::StorageModeShared.0
                | MTLResourceOptions::HazardTrackingModeUntracked.0,
        );
        let old = device.new_buffer(16, options)?;
        let next = if reuse_buffer {
            old.clone()
        } else {
            device.new_buffer(16, options)?
        };
        residency.insert(&old);
        if !reuse_buffer {
            residency.insert(&next);
        }
        commands
            .state
            .lock()
            .unwrap()
            .current
            .as_ref()
            .encodeWaitForEvent_value(ProtocolObject::from_ref(&*event), 1);
        commands
            .blit_command_encoder()?
            .fill_buffer(&old, (0, 16), 1);

        std::thread::scope(|scope| -> Result<(), MetalKernelError> {
            // Declare inside the scope so panic releases the event before
            // scope teardown joins a worker blocked on that same event.
            let release = ReleaseEvent(event.clone());
            let waiter = scope.spawn(|| commands.flush_and_wait());
            wait_for_snapshot_taken(&commands);
            if !new_work {
                // No new kernels were encoded, but the first waiter's
                // snapshot still contains GPU work blocked on our event.
                let (entered_tx, entered_rx) = std::sync::mpsc::channel();
                let (done_tx, done_rx) = std::sync::mpsc::channel();
                let commands = &commands;
                let second = scope.spawn(move || {
                    entered_tx.send(()).unwrap();
                    done_tx.send(commands.flush_and_wait()).unwrap();
                });
                entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                let early = done_rx.recv_timeout(Duration::from_millis(100));
                drop(release);
                waiter.join().unwrap()?;
                second.join().unwrap();
                assert!(
                    matches!(early, Err(std::sync::mpsc::RecvTimeoutError::Timeout)),
                    "an empty snapshot returned before earlier GPU work completed"
                );
                done_rx.recv().unwrap()?;
                return Ok(());
            }
            commands
                .blit_command_encoder()?
                .fill_buffer(&next, (0, 16), 2);
            let key = next.raw_ptr() as usize;
            let fence = commands
                .prev_ce_outputs
                .lock()
                .unwrap()
                .get(&key)
                .unwrap()
                .clone();
            drop(release);
            waiter.join().unwrap()?;
            assert!(
                commands
                    .prev_ce_outputs
                    .lock()
                    .unwrap()
                    .get(&key)
                    .is_some_and(|current| Arc::ptr_eq(current, &fence)),
                "an older wait erased the fence for new, unsubmitted work"
            );
            commands.flush_and_wait_current()?;
            // The GPU wait completed; shared bytes are now safe to inspect.
            assert_eq!(
                unsafe { std::slice::from_raw_parts(next.contents(), 16) },
                &[2; 16]
            );
            Ok(())
        })
    }
}
