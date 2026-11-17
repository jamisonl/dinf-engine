//! Run one decode step's CUDA launches as a few graphs.
//!
//! A decode step issues a few hundred small kernel launches, and each one
//! costs several microseconds of driver time before the GPU does any work.
//! Here the step runs with its stream capturing: launches are recorded, not
//! run. A recorded graph then replaces the contents of an already
//! instantiated graph, which is cheap when the two have the same kernels in
//! the same order, and runs with one driver call. Pointers, scalars and grid
//! sizes may differ from the previous step; only the sequence of kernels has
//! to match, and when it does not the graph is instantiated again.
//!
//! The step is recorded in segments of a few layers. Each segment is
//! launched as soon as it is recorded, so the GPU runs earlier layers while
//! the host records later ones; recording the whole step first would leave
//! the GPU idle for that long on every token.
//!
//! The patched `cudarc` capture window (see `third_party/cudarc/UPSTREAM.md`)
//! makes allocation, frees and small host uploads safe while the stream
//! records. A device-to-host copy inside the step is an error, because
//! recorded work has not run yet; the caller then repeats the step with
//! direct launches.

use std::cell::RefCell;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::Device;
use candle_core::cuda_backend::cudarc::driver::{CudaStream, capture_window, result, sys};

/// Layers recorded before a segment is launched.
const LAYERS_PER_SEGMENT: usize = 4;

/// An instantiated graph. The driver does not synchronize graph objects;
/// this one is only reached through `&mut DecodeGraph`.
struct Exec(sys::CUgraphExec);

// SAFETY: The handle is used by one thread at a time, through `&mut`, on the
// context that created it.
unsafe impl Send for Exec {}
unsafe impl Sync for Exec {}

impl Drop for Exec {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: The handle came from `cuGraphInstantiate` and is not
            // used again.
            let _ = unsafe { result::graph::exec_destroy(self.0) };
        }
    }
}

/// A captured graph, destroyed when its segment is done with it.
struct Captured(sys::CUgraph);

impl Drop for Captured {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: The handle came from `cuStreamEndCapture`.
            let _ = unsafe { result::graph::destroy(self.0) };
        }
    }
}

/// Counters for graph-captured decode steps.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecodeGraphStats {
    /// Steps run as graphs.
    pub steps: u64,
    /// Segments launched.
    pub segments: u64,
    /// Segments whose graph had to be instantiated because the previous
    /// step's segment did not have the same kernel sequence.
    pub instantiations: u64,
    /// Host time spent recording steps, in nanoseconds: the step's own code
    /// with launches captured instead of run.
    pub record_ns: u64,
    /// Host time spent updating or instantiating graphs and launching them,
    /// in nanoseconds.
    pub submit_ns: u64,
    /// Kernel launches and other nodes in the launched segments.
    pub nodes: u64,
}

/// The step being recorded on this thread.
struct Active {
    stream: Arc<CudaStream>,
    /// One executable graph per segment of the step, reused across steps.
    segments: Vec<Option<Exec>>,
    /// Index of the segment being recorded.
    next: usize,
    capturing: bool,
    stats: DecodeGraphStats,
    /// A segment's launches were recorded but could not be run.
    failure: Option<anyhow::Error>,
}

thread_local! {
    static ACTIVE: RefCell<Option<Active>> = const { RefCell::new(None) };
}

fn begin_capture(stream: &CudaStream) -> Result<()> {
    // SAFETY: The stream is this device's stream and is not capturing.
    unsafe {
        result::stream::begin_capture(
            stream.cu_stream(),
            sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
        )
    }
    .context("could not begin CUDA graph capture")
}

/// End the capture in progress and run what it recorded.
fn launch_segment(active: &mut Active) -> Result<()> {
    let started = Instant::now();
    active.capturing = false;
    let cu_stream = active.stream.cu_stream();
    // SAFETY: A capture is in progress on this thread's stream.
    let graph = unsafe { result::stream::end_capture(cu_stream) }
        .context("CUDA graph capture was invalidated")?;
    let graph = Captured(graph);
    let index = active.next;
    active.next += 1;
    if graph.0.is_null() {
        return Ok(());
    }
    if active.segments.len() <= index {
        active.segments.resize_with(index + 1, || None);
    }
    let mut nodes = 0usize;
    // SAFETY: A null node array asks only for the count.
    let counted = unsafe { sys::cuGraphGetNodes(graph.0, std::ptr::null_mut(), &mut nodes) };
    if counted == sys::CUresult::CUDA_SUCCESS {
        active.stats.nodes += nodes as u64;
    }
    let updated = match &active.segments[index] {
        Some(exec) => {
            // SAFETY: Both handles are live; `info` is plain data.
            let mut info: sys::CUgraphExecUpdateResultInfo = unsafe { std::mem::zeroed() };
            let status = unsafe { sys::cuGraphExecUpdate_v2(exec.0, graph.0, &mut info) };
            status == sys::CUresult::CUDA_SUCCESS
        }
        None => false,
    };
    if !updated {
        // A different kernel sequence: build a fresh executable graph.
        active.segments[index] = None;
        let mut exec = std::ptr::null_mut();
        // SAFETY: `graph` is a live captured graph.
        let status = unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph.0, 0) };
        anyhow::ensure!(
            status == sys::CUresult::CUDA_SUCCESS,
            "could not instantiate a decode graph segment: {status:?}"
        );
        active.segments[index] = Some(Exec(exec));
        active.stats.instantiations += 1;
    }
    let exec = active.segments[index].as_ref().expect("instantiated above");
    // SAFETY: The executable graph matches `graph`, whose buffers the
    // capture window keeps alive until the window ends.
    unsafe { result::graph::launch(exec.0, cu_stream) }
        .context("could not launch a decode graph segment")?;
    active.stats.segments += 1;
    active.stats.submit_ns += started.elapsed().as_nanos() as u64;
    Ok(())
}

/// Called by an architecture before each layer of a decode step. When a step
/// is being recorded on this thread, every [`LAYERS_PER_SEGMENT`] layers the
/// recorded launches are run and a new capture begins.
pub(crate) fn segment_boundary(layer_index: usize) {
    // The GPU is idle until the first segment is submitted, so the first
    // segments are short: one layer, one layer, then the regular size.
    let boundary = layer_index == 1
        || layer_index == 2
        || (layer_index > 0 && layer_index.is_multiple_of(LAYERS_PER_SEGMENT));
    if !boundary {
        return;
    }
    ACTIVE.with(|active| {
        let mut active = active.borrow_mut();
        let Some(active) = active.as_mut() else {
            return;
        };
        if !active.capturing {
            return;
        }
        let outcome = launch_segment(active).and_then(|()| {
            begin_capture(&active.stream)?;
            active.capturing = true;
            Ok(())
        });
        if let Err(error) = outcome {
            // Later launches run directly; the step as a whole is reported
            // as failed so its caller rolls back and repeats it.
            active.failure.get_or_insert(error);
        }
    });
}

/// Per-stage state for graph-captured decode steps.
#[derive(Default)]
pub(crate) struct DecodeGraph {
    segments: Vec<Option<Exec>>,
    stats: DecodeGraphStats,
    /// Set after a capture failed while the same step succeeded with direct
    /// launches; later steps skip the capture.
    disabled: bool,
    /// The stage's buffers were all allocated without cudarc ordering
    /// events, which a captured graph cannot carry.
    event_free: bool,
}

impl DecodeGraph {
    /// State for a stage whose device had event tracking off before any of
    /// its buffers were allocated.
    pub(crate) fn event_free() -> Self {
        Self {
            event_free: true,
            ..Self::default()
        }
    }

    /// Whether steps on this stage may be captured.
    pub(crate) fn usable(&self) -> bool {
        self.event_free && !self.disabled
    }

    pub(crate) fn disable(&mut self) {
        self.disabled = true;
        self.segments.clear();
    }

    pub(crate) fn stats(&self) -> DecodeGraphStats {
        self.stats
    }

    /// Run `step` with its launches on `device` recorded and run as graphs.
    /// On an error some of the step's launches have not run, and any state
    /// `step` changed must be rolled back by the caller.
    pub(crate) fn run<T>(
        &mut self,
        device: &Device,
        step: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let cuda = device.as_cuda_device()?;
        let stream = cuda.cuda_stream();
        // With nothing pending, every pool block is safe for host uploads.
        // SAFETY: The stream handle is live.
        if unsafe { sys::cuStreamQuery(stream.cu_stream()) } == sys::CUresult::CUDA_SUCCESS {
            capture_window::mark_idle(&stream);
        }
        capture_window::begin(&stream).context("could not open a CUDA capture window")?;
        let started = Instant::now();
        let recorded = self.record(&stream, step);
        let elapsed = started.elapsed().as_nanos() as u64;
        let closed =
            capture_window::end(&stream).context("could not close the CUDA capture window");
        let (value, stats) = recorded?;
        closed?;
        self.stats.steps += 1;
        self.stats.segments += stats.segments;
        self.stats.instantiations += stats.instantiations;
        self.stats.submit_ns += stats.submit_ns;
        self.stats.nodes += stats.nodes;
        self.stats.record_ns += elapsed.saturating_sub(stats.submit_ns);
        Ok(value)
    }

    fn record<T>(
        &mut self,
        stream: &Arc<CudaStream>,
        step: impl FnOnce() -> Result<T>,
    ) -> Result<(T, DecodeGraphStats)> {
        begin_capture(stream)?;
        ACTIVE.with(|active| {
            *active.borrow_mut() = Some(Active {
                stream: Arc::clone(stream),
                segments: std::mem::take(&mut self.segments),
                next: 0,
                capturing: true,
                stats: DecodeGraphStats::default(),
                failure: None,
            });
        });
        let value = step();
        let mut active = ACTIVE
            .with(|active| active.borrow_mut().take())
            .expect("the active step was installed above");
        // Always end a capture in progress, so the stream is usable again.
        let last = if active.capturing {
            launch_segment(&mut active)
        } else {
            Ok(())
        };
        self.segments = std::mem::take(&mut active.segments);
        let value = value?;
        if let Some(error) = active.failure {
            return Err(error);
        }
        last?;
        Ok((value, active.stats))
    }
}
