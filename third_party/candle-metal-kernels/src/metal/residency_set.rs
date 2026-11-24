use super::{Buffer, Device};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLAllocation, MTLDevice as _, MTLResidencySet, MTLResidencySetDescriptor};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Keeps Metal buffers resident in GPU memory, removing the need for `useResource` calls.
///
/// Add the set to the command queue once via `MTLCommandQueue::addResidencySet`. Then
/// register every buffer at allocation time and unregister at free time.
#[derive(Clone)]
pub struct ResidencySet {
    raw: Option<Arc<ResidencyState>>,
}

#[derive(Debug)]
pub(crate) struct ResidencyState {
    raw: Retained<ProtocolObject<dyn MTLResidencySet>>,
    // Metal residency set methods do not support concurrent calls.
    lock: Mutex<()>,
    deferred: AtomicBool,
    dirty: AtomicBool,
}

unsafe impl Send for ResidencyState {}
unsafe impl Sync for ResidencyState {}

impl ResidencyState {
    pub(crate) fn remove(&self, allocation: &ProtocolObject<dyn MTLAllocation>) {
        let _guard = self.lock.lock().unwrap_or_else(|error| error.into_inner());
        self.raw.removeAllocation(allocation);
        self.dirty.store(false, Ordering::Release);
        self.raw.commit();
    }

    pub(crate) fn commit_if_dirty(&self) {
        if self.dirty.load(Ordering::Acquire) {
            let _guard = self.lock.lock().unwrap_or_else(|error| error.into_inner());
            if self.dirty.swap(false, Ordering::AcqRel) {
                self.raw.commit();
            }
        }
    }
}

unsafe impl Send for ResidencySet {}
unsafe impl Sync for ResidencySet {}

impl ResidencySet {
    pub fn new(device: &Device) -> Self {
        let descriptor = MTLResidencySetDescriptor::new();
        let raw = device
            .as_ref()
            .newResidencySetWithDescriptor_error(&descriptor)
            .ok()
            .inspect(|set| set.requestResidency());
        ResidencySet {
            raw: raw.map(|raw| {
                Arc::new(ResidencyState {
                    raw,
                    lock: Mutex::new(()),
                    deferred: AtomicBool::new(false),
                    dirty: AtomicBool::new(false),
                })
            }),
        }
    }

    /// Batch residency commits until `commit_if_dirty()` is called before
    /// command buffer submission (eliminates hundreds of per-buffer commits/token).
    pub(crate) fn enable_deferred_commit(&self) {
        if let Some(state) = &self.raw {
            state.deferred.store(true, Ordering::Release);
        }
    }

    pub(crate) fn commit_if_dirty(&self) {
        if let Some(state) = &self.raw {
            state.commit_if_dirty();
        }
    }

    pub fn raw(&self) -> Option<&ProtocolObject<dyn MTLResidencySet>> {
        self.raw.as_ref().map(|state| state.raw.as_ref())
    }

    #[cfg(test)]
    pub(crate) fn allocation_count(&self) -> usize {
        let state = self.raw.as_ref().expect("test requires a residency set");
        let _guard = state.lock.lock().unwrap_or_else(|error| error.into_inner());
        state.raw.allocationCount()
    }

    pub fn insert(&self, buf: &Buffer) {
        self.insert_buffers(&[buf]);
    }

    /// Adds multiple buffers in a single commit, retaining last-owner cleanup.
    pub fn insert_batch<'a>(&self, bufs: impl IntoIterator<Item = &'a Buffer>) {
        if self.raw.is_none() {
            return;
        }
        // An iterator may drop another registered buffer while it advances or
        // drops. Run that caller code before locking the set, since a buffer's
        // last-owner cleanup takes the same lock.
        let bufs: Vec<_> = bufs.into_iter().collect();
        self.insert_buffers(&bufs);
    }

    fn insert_buffers(&self, bufs: &[&Buffer]) {
        if let Some(set) = &self.raw {
            let mut added = 0u64;
            let _guard = set.lock.lock().unwrap_or_else(|error| error.into_inner());
            for buf in bufs {
                if buf.register_residency(set) {
                    set.raw.addAllocation(as_allocation(buf));
                    added += 1;
                }
            }
            if added > 0 {
                if set.deferred.load(Ordering::Relaxed) {
                    set.dirty.store(true, Ordering::Release);
                } else {
                    set.raw.commit();
                }
            }
        }
    }

    /// Removes multiple buffers in a single commit under the shared set lock.
    pub fn remove_batch<'a>(&self, bufs: impl IntoIterator<Item = &'a Buffer>) {
        if self.raw.is_none() {
            return;
        }
        let bufs: Vec<_> = bufs.into_iter().collect();
        self.remove_buffers(&bufs);
    }

    fn remove_buffers(&self, bufs: &[&Buffer]) {
        if let Some(set) = &self.raw {
            let _guard = set.lock.lock().unwrap_or_else(|error| error.into_inner());
            for buf in bufs {
                set.raw.removeAllocation(as_allocation(buf));
                buf.unregister_residency(set);
            }
            if !bufs.is_empty() {
                if set.deferred.load(Ordering::Relaxed) {
                    set.dirty.store(true, Ordering::Release);
                } else {
                    set.raw.commit();
                }
            }
        }
    }

    pub fn remove(&self, buf: &Buffer) {
        self.remove_buffers(&[buf]);
    }
}

/// Cast a `&Buffer` to `&ProtocolObject<dyn MTLAllocation>`.
///
/// Safe because `MTLBuffer: MTLResource: MTLAllocation`. All `ProtocolObject<P>` are
/// `repr(C)` thin ObjC pointers — the cast only changes the static protocol marker,
/// not the pointer value or runtime dispatch.
fn as_allocation(buf: &Buffer) -> &ProtocolObject<dyn MTLAllocation> {
    unsafe {
        &*(buf.as_ref() as *const ProtocolObject<dyn objc2_metal::MTLBuffer>
            as *const ProtocolObject<dyn MTLAllocation>)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLResourceOptions;

    #[test]
    fn unsupported_residency_does_not_advance_batch_iterators() {
        let set = ResidencySet { raw: None };
        set.insert_batch(std::iter::from_fn(|| -> Option<&Buffer> {
            panic!("unsupported residency must not advance an insertion iterator")
        }));
        set.remove_batch(std::iter::from_fn(|| -> Option<&Buffer> {
            panic!("unsupported residency must not advance a removal iterator")
        }));
    }

    #[test]
    fn batch_iterators_can_drop_other_registered_buffers_without_deadlock() {
        let device = Device::system_default().expect("test requires Metal");
        let set = ResidencySet::new(&device);
        let baseline = set.allocation_count();
        let new_buffer = || {
            device
                .new_buffer(4096, MTLResourceOptions::StorageModePrivate)
                .unwrap()
        };
        let buffer = new_buffer();
        let mut retired = Some(new_buffer());
        set.insert(retired.as_ref().unwrap());
        set.insert_batch(std::iter::once(&buffer).inspect(|_| {
            assert!(
                set.raw.as_ref().unwrap().lock.try_lock().is_ok(),
                "insertion iterator must run outside the residency lock"
            );
            drop(retired.take());
        }));
        assert_eq!(set.allocation_count(), baseline + 1);

        retired = Some(new_buffer());
        set.insert(retired.as_ref().unwrap());
        set.remove_batch(std::iter::once(&buffer).inspect(|_| {
            assert!(
                set.raw.as_ref().unwrap().lock.try_lock().is_ok(),
                "removal iterator must run outside the residency lock"
            );
            drop(retired.take());
        }));
        assert_eq!(set.allocation_count(), baseline);
        drop(buffer);
        assert_eq!(set.allocation_count(), baseline);
    }

    #[test]
    fn batch_registered_buffers_leave_residency_on_last_drop() {
        let device = Device::system_default().expect("test requires Metal");
        let set = ResidencySet::new(&device);
        let baseline = set.allocation_count();
        let buffers: Vec<_> = (0..3)
            .map(|_| {
                device
                    .new_buffer(4096, MTLResourceOptions::StorageModePrivate)
                    .unwrap()
            })
            .collect();
        set.insert_batch(&buffers);
        set.insert_batch(&buffers);
        assert_eq!(set.allocation_count(), baseline + 3);
        let last = buffers[0].clone();
        drop(buffers);
        assert_eq!(set.allocation_count(), baseline + 1);
        set.remove_batch([&last]);
        assert_eq!(set.allocation_count(), baseline);
        set.insert_batch([&last]);
        drop(last);
        assert_eq!(set.allocation_count(), baseline);
    }

    #[test]
    fn private_buffers_leave_residency_when_the_last_wrapper_drops() {
        let device = Device::system_default().expect("test requires Metal");
        let set = ResidencySet::new(&device);
        let raw = set.raw().expect("test requires a residency set");
        let baseline = raw.allocationCount();
        for _ in 0..256 {
            let buffer = device
                .new_buffer(1024 * 1024, MTLResourceOptions::StorageModePrivate)
                .unwrap();
            set.insert(&buffer);
            set.insert(&buffer);
            assert_eq!(raw.allocationCount(), baseline + 1);
            let clone = buffer.clone();
            drop(buffer);
            assert_eq!(raw.allocationCount(), baseline + 1);
            drop(clone);
            assert_eq!(raw.allocationCount(), baseline);
        }
        let buffer = device
            .new_buffer(4096, MTLResourceOptions::StorageModeShared)
            .unwrap();
        set.insert(&buffer);
        set.remove(&buffer);
        assert_eq!(raw.allocationCount(), baseline);
        drop(buffer);
        assert_eq!(raw.allocationCount(), baseline);
    }
}
