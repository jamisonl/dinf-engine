use super::residency_set::ResidencyState;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{MTLAllocation, MTLBuffer, MTLResource};
use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    sync::{Arc, Mutex},
};

pub type MetalResource = ProtocolObject<dyn MTLResource>;
pub type MTLResourceOptions = objc2_metal::MTLResourceOptions;

#[derive(Clone, Debug)]
pub struct Buffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    residency: Arc<BufferResidency>,
}

// Metal retains every allocation in a residency set. Non-pooled buffers need
// to unregister when their last Rust wrapper drops, or GPU memory never frees.
// Buffer clones share this guard; dropping one clone must not unregister a
// buffer that another clone still uses.
#[derive(Debug)]
struct BufferResidency {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    sets: Mutex<Vec<Arc<ResidencyState>>>,
}

impl Drop for BufferResidency {
    fn drop(&mut self) {
        let allocation = unsafe {
            &*(self.raw.as_ref() as *const ProtocolObject<dyn MTLBuffer>
                as *const ProtocolObject<dyn MTLAllocation>)
        };
        for set in self
            .sets
            .get_mut()
            .unwrap_or_else(|error| error.into_inner())
        {
            set.remove(allocation);
        }
    }
}

impl Hash for Buffer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
    }
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub fn new(raw: Retained<ProtocolObject<dyn MTLBuffer>>) -> Buffer {
        let residency = Arc::new(BufferResidency {
            raw: raw.clone(),
            sets: Mutex::new(Vec::new()),
        });
        Buffer { raw, residency }
    }

    pub(crate) fn register_residency(&self, set: &Arc<ResidencyState>) -> bool {
        let mut sets = self
            .residency
            .sets
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if !sets.iter().any(|old| Arc::ptr_eq(old, set)) {
            sets.push(set.clone());
            true
        } else {
            false
        }
    }

    pub(crate) fn unregister_residency(&self, set: &Arc<ResidencyState>) {
        self.residency
            .sets
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .retain(|old| !Arc::ptr_eq(old, set));
    }

    pub fn contents(&self) -> *mut u8 {
        self.data()
    }

    pub fn data(&self) -> *mut u8 {
        use objc2_metal::MTLBuffer as _;
        self.as_ref().contents().as_ptr() as *mut u8
    }

    /// Get the raw pointer to the underlying Metal buffer object.
    /// Used for dependency tracking in the compute encoder.
    pub(crate) fn raw_ptr(&self) -> *const ProtocolObject<dyn MTLBuffer> {
        Retained::as_ptr(&self.raw)
    }

    pub fn length(&self) -> usize {
        self.as_ref().length()
    }

    pub fn did_modify_range(&self, range: NSRange) {
        self.as_ref().didModifyRange(range);
    }

    pub fn set_label(&self, label: &str) {
        self.raw.setLabel(Some(&NSString::from_str(label)))
    }
}

impl AsRef<ProtocolObject<dyn MTLBuffer>> for Buffer {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }
}

impl<'a> From<&'a Buffer> for &'a MetalResource {
    fn from(val: &'a Buffer) -> Self {
        ProtocolObject::from_ref(val.as_ref())
    }
}

pub type BufferMap = HashMap<usize, Vec<Arc<Buffer>>>;
