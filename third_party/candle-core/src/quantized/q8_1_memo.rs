//! Activations that already have a Q8_1 form on the device.
//!
//! This module is a local addition; see `UPSTREAM.md` at the crate root.
//!
//! The quantized matrix-vector kernels read their activation as Q8_1 blocks,
//! and by default each product first launches a kernel that quantizes it.
//! A kernel that produces an activation can write the Q8_1 form in the same
//! launch. It then publishes the pair here, and a product whose input is
//! that activation uses the published blocks and skips its own quantization.
//!
//! The published form must be what the quantization kernels write for the
//! activation: rows padded to a multiple of 512 values, 36-byte blocks of
//! two half-precision values (scale, sum) and 32 signed bytes.
//!
//! An entry keeps the producing tensor's storage alive, so the address it is
//! keyed by cannot be given to another buffer while the entry exists.
//! Entries are per thread, and only the last few are kept. A tensor must not
//! be written in place after it is published.

use std::cell::RefCell;

use cudarc::driver::DevicePtr;

use crate::backend::BackendStorage;
use crate::{DType, Result, Storage, Tensor};

/// Entries kept per thread. A layer publishes at most this many activations
/// between the products that use them.
const MAX_ENTRIES: usize = 4;

struct Entry {
    /// Holds the storage both addresses point into.
    _storage: Tensor,
    /// Device address of the first activation value.
    activation: u64,
    /// Number of F32 activation values.
    len: usize,
    /// Device address of the first Q8_1 block.
    quantized: u64,
}

thread_local! {
    static ENTRIES: RefCell<Vec<Entry>> = const { RefCell::new(Vec::new()) };
    static HITS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Lookups on this thread that found a published form.
pub fn hits() -> u64 {
    HITS.with(std::cell::Cell::get)
}

/// Bytes of the Q8_1 form of one activation row of `len` values.
pub fn quantized_bytes(len: usize) -> usize {
    len.div_ceil(512) * 512 / 32 * 36
}

/// Record that `packed`, a contiguous F32 CUDA tensor, holds `len`
/// activation values starting at element `activation_offset` of its storage
/// and their Q8_1 form starting at element `quantized_offset`.
pub fn publish(
    packed: &Tensor,
    activation_offset: usize,
    len: usize,
    quantized_offset: usize,
) -> Result<()> {
    let (storage, _) = packed.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        crate::bail!("a published Q8_1 activation must be on CUDA")
    };
    if storage.dtype() != DType::F32 {
        crate::bail!("a published Q8_1 activation must be F32")
    }
    let slice = storage.as_cuda_slice::<f32>()?;
    let float = std::mem::size_of::<f32>();
    let activation_end = activation_offset.checked_add(len);
    let quantized_end = quantized_offset
        .checked_mul(float)
        .and_then(|start| start.checked_add(quantized_bytes(len)));
    let fits = len > 0
        && activation_end.is_some_and(|end| end <= slice.len())
        && quantized_end.is_some_and(|end| end <= slice.len() * float);
    if !fits {
        crate::bail!("a published Q8_1 activation exceeds its storage")
    }
    let stream = storage.device.cuda_stream();
    let (base, sync) = slice.device_ptr(&stream);
    drop(sync);
    let activation = base + (activation_offset * float) as u64;
    let quantized = base + (quantized_offset * float) as u64;
    ENTRIES.with(|entries| {
        let mut entries = entries.borrow_mut();
        entries.retain(|entry| entry.activation != activation);
        if entries.len() == MAX_ENTRIES {
            entries.remove(0);
        }
        entries.push(Entry {
            _storage: packed.clone(),
            activation,
            len,
            quantized,
        });
    });
    Ok(())
}

/// The device address of the published Q8_1 form of the `len` F32 values at
/// device address `activation`, if this thread published one.
pub fn lookup(activation: u64, len: usize) -> Option<u64> {
    let found = ENTRIES.with(|entries| {
        entries
            .borrow()
            .iter()
            .find(|entry| entry.activation == activation && entry.len == len)
            .map(|entry| entry.quantized)
    });
    if found.is_some() {
        HITS.with(|hits| hits.set(hits.get() + 1));
    }
    found
}

/// Forget this thread's entries and release the storage they hold.
pub fn clear() {
    ENTRIES.with(|entries| entries.borrow_mut().clear());
}
