//! Switches for the process-wide layer profiling counters.
//!
//! Each architecture that supports layer profiling keeps its timings in
//! process-wide counters, read by its `reset_layer_profile` and
//! `report_layer_profile` functions. These switches are process-wide for the
//! same reason. Both are off by default. On Metal and CPU, profiling
//! synchronizes the device around each timed region, so it slows inference
//! and should not be enabled for throughput measurements. On CUDA it records
//! a timing event on the stream at the end of each region instead and reads
//! the device-side intervals when the profile is reported, so regions are
//! timed as they run without stalling the pipeline. Time between two timed
//! regions is charged to the later one, and the first region after a reset
//! or report is not counted.

use std::sync::atomic::{AtomicBool, Ordering};

static LAYERS: AtomicBool = AtomicBool::new(false);
static MOE_ROUTES: AtomicBool = AtomicBool::new(false);
static REGIONS: AtomicBool = AtomicBool::new(false);

/// Collect per-layer timings in every architecture's profiling counters.
pub fn set_layer_profiling(enabled: bool) {
    LAYERS.store(enabled, Ordering::Relaxed);
}

/// Whether per-layer timings are being collected.
pub fn layer_profiling_enabled() -> bool {
    LAYERS.load(Ordering::Relaxed)
}

/// Collect Nemotron-H expert-route statistics in its layer profile.
///
/// This reads each route plan back from the device, which also slows
/// inference.
pub fn set_moe_route_profiling(enabled: bool) {
    MOE_ROUTES.store(enabled, Ordering::Relaxed);
}

/// Whether Nemotron-H expert-route statistics are being collected.
pub fn moe_route_profiling_enabled() -> bool {
    MOE_ROUTES.load(Ordering::Relaxed)
}

#[cfg(feature = "cuda")]
mod cuda_regions {
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicU64, Ordering};

    use candle_core::cuda_backend::cudarc::driver::{CudaEvent, sys::CUevent_flags};

    #[derive(Default)]
    struct Regions {
        /// End of the last region already read back.
        previous: Option<CudaEvent>,
        /// Region ends in stream order, each with the counter it feeds.
        pending: Vec<(CudaEvent, &'static AtomicU64)>,
    }

    thread_local! {
        static REGIONS: RefCell<Regions> = RefCell::new(Regions::default());
    }

    /// Read back once this many driver events are held.
    const MAX_PENDING: usize = 4096;

    /// Mark the end of a timed region on `device`'s stream. Returns `false`
    /// when `device` is not CUDA.
    pub(crate) fn finish(
        device: &candle_core::Device,
        counter: &'static AtomicU64,
    ) -> anyhow::Result<bool> {
        let candle_core::Device::Cuda(cuda) = device else {
            return Ok(false);
        };
        let event = cuda
            .cuda_stream()
            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        let full = REGIONS.with(|regions| {
            let mut regions = regions.borrow_mut();
            regions.pending.push((event, counter));
            regions.pending.len() >= MAX_PENDING
        });
        if full {
            read_back(true)?;
        }
        Ok(true)
    }

    /// Add each pending interval to its counter. With `keep_position` the
    /// last event stays as the start of the next interval.
    fn read_back(keep_position: bool) -> anyhow::Result<()> {
        REGIONS.with(|regions| {
            let mut regions = regions.borrow_mut();
            let mut start = regions.previous.take();
            for (end, counter) in std::mem::take(&mut regions.pending) {
                if let Some(start) = &start {
                    let milliseconds = start.elapsed_ms(&end)?;
                    counter.fetch_add((f64::from(milliseconds) * 1e6) as u64, Ordering::Relaxed);
                }
                start = Some(end);
            }
            regions.previous = if keep_position { start } else { None };
            Ok(())
        })
    }

    /// Count every pending interval and forget the stream position.
    pub(crate) fn drain() -> anyhow::Result<()> {
        read_back(false)
    }

    /// Drop pending intervals without counting them.
    pub(crate) fn clear() {
        REGIONS.with(|regions| *regions.borrow_mut() = Regions::default());
    }
}

/// Fold CUDA intervals recorded on this thread into the counters. Call
/// before reading them. A no-op without the `cuda` feature.
pub(crate) fn drain_device_regions() {
    #[cfg(feature = "cuda")]
    if let Err(error) = cuda_regions::drain() {
        tracing::warn!("could not read CUDA profile intervals: {error}");
    }
}

/// Drop CUDA intervals recorded on this thread. Call when resetting the
/// counters. A no-op without the `cuda` feature.
pub(crate) fn clear_device_regions() {
    #[cfg(feature = "cuda")]
    cuda_regions::clear();
}

/// End a timed region: on CUDA mark it on the stream, elsewhere synchronize
/// and add the host time since `started`.
pub(crate) fn finish_region(
    device: &candle_core::Device,
    started: std::time::Instant,
    counter: &'static std::sync::atomic::AtomicU64,
) -> anyhow::Result<()> {
    #[cfg(feature = "cuda")]
    if cuda_regions::finish(device, counter)? {
        return Ok(());
    }
    device.synchronize()?;
    counter.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(())
}

/// Time named regions of the production CUDA paths.
///
/// Unlike the layer profile, region marks do not change which code runs:
/// [`mark_region`] only records a timing event on the stream. Each mark
/// charges the device time since the previous mark to its name. Off CUDA the
/// marks do nothing.
pub fn set_region_profiling(enabled: bool) {
    REGIONS.store(enabled, Ordering::Relaxed);
}

/// Whether named regions are being timed.
pub fn region_profiling_enabled() -> bool {
    REGIONS.load(Ordering::Relaxed)
}

/// End the region `name` here. A no-op unless region profiling is on and
/// `device` is CUDA.
pub(crate) fn mark_region(device: &candle_core::Device, name: &'static str) {
    if !region_profiling_enabled() {
        return;
    }
    #[cfg(feature = "cuda")]
    if let Err(error) = cuda_named_regions::mark(device, name) {
        tracing::warn!("could not record a CUDA region mark: {error}");
    }
    #[cfg(not(feature = "cuda"))]
    let _ = (device, name);
}

/// Print each region's total device time, mark count and mean to stderr,
/// largest first, then clear them.
pub fn report_regions() {
    #[cfg(feature = "cuda")]
    cuda_named_regions::report();
}

/// Drop recorded region marks without reporting them.
pub fn reset_regions() {
    #[cfg(feature = "cuda")]
    cuda_named_regions::reset();
}

#[cfg(feature = "cuda")]
mod cuda_named_regions {
    use std::cell::RefCell;
    use std::collections::HashMap;

    use candle_core::cuda_backend::cudarc::driver::{CudaEvent, sys::CUevent_flags};

    #[derive(Default)]
    struct Marks {
        previous: Option<CudaEvent>,
        pending: Vec<(CudaEvent, &'static str)>,
        /// Name to (nanoseconds, marks).
        totals: HashMap<&'static str, (u64, u64)>,
    }

    thread_local! {
        static MARKS: RefCell<Marks> = RefCell::new(Marks::default());
    }

    const MAX_PENDING: usize = 4096;

    pub(super) fn mark(device: &candle_core::Device, name: &'static str) -> anyhow::Result<()> {
        let candle_core::Device::Cuda(cuda) = device else {
            return Ok(());
        };
        let event = cuda
            .cuda_stream()
            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        MARKS.with(|marks| {
            let mut marks = marks.borrow_mut();
            marks.pending.push((event, name));
            if marks.pending.len() >= MAX_PENDING {
                read_back(&mut marks)?;
            }
            Ok(())
        })
    }

    fn read_back(marks: &mut Marks) -> anyhow::Result<()> {
        let mut start = marks.previous.take();
        for (end, name) in std::mem::take(&mut marks.pending) {
            if let Some(start) = &start {
                let nanoseconds = (f64::from(start.elapsed_ms(&end)?) * 1e6) as u64;
                let total = marks.totals.entry(name).or_default();
                total.0 += nanoseconds;
                total.1 += 1;
            }
            start = Some(end);
        }
        marks.previous = start;
        Ok(())
    }

    pub(super) fn report() {
        MARKS.with(|marks| {
            let mut marks = marks.borrow_mut();
            if let Err(error) = read_back(&mut marks) {
                eprintln!("could not read CUDA region marks: {error}");
            }
            let mut rows = marks.totals.drain().collect::<Vec<_>>();
            rows.sort_by_key(|(_, (nanoseconds, _))| std::cmp::Reverse(*nanoseconds));
            let all = rows.iter().map(|(_, (ns, _))| *ns).sum::<u64>().max(1);
            for (name, (nanoseconds, count)) in rows {
                eprintln!(
                    "region {name:<24} {:>9.3} ms {:>5.1}%  {count:>7} marks  {:>8.2} us each",
                    nanoseconds as f64 / 1e6,
                    nanoseconds as f64 * 100.0 / all as f64,
                    nanoseconds as f64 / 1e3 / count.max(1) as f64
                );
            }
            marks.previous = None;
        });
    }

    pub(super) fn reset() {
        MARKS.with(|marks| *marks.borrow_mut() = Marks::default());
    }
}
