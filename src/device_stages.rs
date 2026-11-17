//! Live stages per CUDA device, and whose turn it is to record on one.
//!
//! A graph-captured decode step records everything issued on its device's
//! stream. That is safe only while no other thread issues work there. A
//! stage is driven through `&mut self`, so the only other users the engine
//! can know about are other stages. This registry counts the stages loaded
//! on each GPU: a stage may record a step only while it is the sole stage on
//! its GPU, and a stage being loaded on that GPU waits for a recording in
//! progress to end before it allocates anything.
//!
//! Stages are counted per GPU ordinal, not per stream. Two stages on separate
//! streams of one GPU would not record each other's work, but they share the
//! capture pool, so they are treated like stages on one stream.

use std::collections::HashMap;
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};

use candle_core::{Device, DeviceLocation};

#[derive(Default)]
struct DeviceState {
    stages: usize,
    recording: bool,
}

static DEVICES: Mutex<Option<HashMap<usize, DeviceState>>> = Mutex::new(None);
static RECORDING_ENDED: Condvar = Condvar::new();

fn devices() -> MutexGuard<'static, Option<HashMap<usize, DeviceState>>> {
    // The map holds counters only; a panic elsewhere cannot leave it torn.
    DEVICES.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One stage's place in the registry. Dropping it removes the stage.
///
/// Create it before the stage allocates on its device and drop it after the
/// stage's device buffers are freed.
#[derive(Debug)]
pub(crate) struct StageRegistration {
    gpu: Option<usize>,
}

impl StageRegistration {
    /// Count a stage about to be loaded on `device`. On a CUDA device this
    /// waits for a step being recorded there to finish. Other devices are
    /// not counted.
    pub(crate) fn register(device: &Device) -> Self {
        match device.location() {
            DeviceLocation::Cuda { gpu_id } => Self::register_gpu(gpu_id),
            _ => Self::uncounted(),
        }
    }

    fn register_gpu(gpu_id: usize) -> Self {
        let mut devices = devices();
        loop {
            let state = devices
                .get_or_insert_with(HashMap::new)
                .entry(gpu_id)
                .or_default();
            if !state.recording {
                state.stages += 1;
                return Self { gpu: Some(gpu_id) };
            }
            devices = RECORDING_ENDED
                .wait(devices)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// A registration that counts nothing and never records.
    pub(crate) fn uncounted() -> Self {
        Self { gpu: None }
    }

    /// Take the device for one recorded step. Returns `None` when another
    /// stage is loaded on the same GPU or the stage is not counted; the
    /// step then runs with direct launches. The turn ends when the returned
    /// value is dropped.
    #[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
    pub(crate) fn begin_recording(&self) -> Option<RecordingTurn> {
        let gpu = self.gpu?;
        let mut devices = devices();
        let state = devices.as_mut()?.get_mut(&gpu)?;
        if state.stages != 1 || state.recording {
            return None;
        }
        state.recording = true;
        Some(RecordingTurn { gpu })
    }
}

impl Drop for StageRegistration {
    fn drop(&mut self) {
        let Some(gpu) = self.gpu else {
            return;
        };
        let mut devices = devices();
        if let Some(map) = devices.as_mut()
            && let Some(state) = map.get_mut(&gpu)
        {
            state.stages = state.stages.saturating_sub(1);
            if state.stages == 0 && !state.recording {
                map.remove(&gpu);
            }
        }
    }
}

/// The right to record one step on a GPU.
#[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
pub(crate) struct RecordingTurn {
    gpu: usize,
}

impl Drop for RecordingTurn {
    fn drop(&mut self) {
        let mut devices = devices();
        if let Some(map) = devices.as_mut()
            && let Some(state) = map.get_mut(&self.gpu)
        {
            state.recording = false;
            if state.stages == 0 {
                map.remove(&self.gpu);
            }
        }
        drop(devices);
        RECORDING_ENDED.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    // Tests use GPU ordinals no machine has, one per test, so they neither
    // need a device nor disturb each other.
    fn stage(gpu: usize) -> StageRegistration {
        StageRegistration::register_gpu(gpu)
    }

    #[test]
    fn only_a_sole_stage_records() {
        let first = stage(9001);
        assert!(first.begin_recording().is_some());

        let second = stage(9001);
        assert!(first.begin_recording().is_none());
        assert!(second.begin_recording().is_none());

        drop(second);
        assert!(first.begin_recording().is_some());
    }

    #[test]
    fn stages_on_other_gpus_do_not_count() {
        let first = stage(9002);
        let _other = stage(9003);
        assert!(first.begin_recording().is_some());
    }

    #[test]
    fn one_recording_at_a_time() {
        let first = stage(9004);
        let turn = first.begin_recording().unwrap();
        assert!(first.begin_recording().is_none());
        drop(turn);
        assert!(first.begin_recording().is_some());
    }

    #[test]
    fn uncounted_stages_never_record() {
        assert!(StageRegistration::register(&Device::Cpu).gpu.is_none());
        assert!(StageRegistration::uncounted().begin_recording().is_none());
    }

    #[test]
    fn a_new_stage_waits_for_the_recording_to_end() {
        let first = stage(9005);
        let turn = first.begin_recording().unwrap();
        let (sender, receiver) = mpsc::channel();
        let loader = std::thread::spawn(move || {
            let second = stage(9005);
            sender.send(()).unwrap();
            second
        });
        assert!(receiver.recv_timeout(Duration::from_millis(100)).is_err());
        drop(turn);
        receiver.recv_timeout(Duration::from_secs(10)).unwrap();
        let second = loader.join().unwrap();
        assert!(first.begin_recording().is_none());
        drop(second);
        assert!(first.begin_recording().is_some());
    }
}
