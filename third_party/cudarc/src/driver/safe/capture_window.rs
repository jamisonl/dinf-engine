//! Allocation and host copies while a stream is capturing a CUDA graph.
//!
//! This module is a local addition; see `UPSTREAM.md` at the crate root.
//!
//! While a stream captures, its launches are recorded and run later, when
//! the graph is launched. Three ordinary operations are wrong in that
//! window:
//!
//! - `cuMemFreeAsync` of memory allocated before the capture invalidates the
//!   capture;
//! - `cuMemAllocAsync` creates graph-owned memory with its own lifetime
//!   rules;
//! - an asynchronous host-to-device copy records the host pointer and reads
//!   it at launch, after the caller's buffer may be gone.
//!
//! A capture window, opened on the capturing thread with [`begin`], changes
//! what the stream's safe wrappers do until [`end`]:
//!
//! - [`CudaStream::alloc`](super::CudaStream::alloc) takes memory from a
//!   process-wide pool of `cuMemAlloc` blocks instead of the stream;
//! - dropping a [`CudaSlice`](super::CudaSlice) holds the memory until
//!   [`end`], so nothing recorded in the graph can lose its buffer and no
//!   buffer is reused inside the window;
//! - a host upload (`clone_htod`, `memcpy_stod`) completes before the call
//!   returns, on a separate non-blocking stream (a synchronous copy would
//!   make the default stream wait on the capturing one, which the driver
//!   rejects), into a block that no pending work can touch; any other
//!   host-to-device copy, and every device-to-host copy, is an error.
//!
//! [`end`] must be called after the window's last graph has been launched
//! (or the capture abandoned). Pool blocks then return to the pool and other
//! memory is freed on the stream, ordered after the launch.
//!
//! A window may hold several captures in a row, each launched before the
//! next begins, so earlier work may still be running while later launches
//! are recorded. The pool therefore keeps released blocks apart from idle
//! ones: a block released since the stream was last known to be idle may
//! still be in use by work that has been launched but has not run. Recorded
//! launches may reuse such a block, because the stream runs them after that
//! work. A host upload runs at once, so it only takes a block that was
//! released before the stream was last idle, or a new one.
//! [`mark_idle`] tells the pool the stream has no work pending.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Mutex;

use crate::driver::{result, sys};

use super::{CudaStream, DriverError};

/// Smallest pool block.
const MIN_BLOCK_BYTES: usize = 256;
/// Largest request served from the pool outside a window. A decode step
/// allocates hundreds of buffers of a few megabytes at most, and the stream
/// allocator costs about a microsecond for each allocation and free.
const DIRECT_POOL_MAX_BYTES: usize = 4 << 20;
/// Blocks above this size are rounded to a multiple of it, not doubled.
const LARGE_BLOCK_BYTES: usize = 2 << 20;

#[derive(Default)]
struct Pool {
    /// Blocks no pending work can touch, by (stream, block size).
    free: HashMap<(usize, usize), Vec<sys::CUdeviceptr>>,
    /// Blocks released since the stream was last idle, by (stream, block
    /// size). Work already launched may still read or write them.
    released: HashMap<(usize, usize), Vec<sys::CUdeviceptr>>,
    /// Every block the pool owns, with its stream and size.
    owned: HashMap<sys::CUdeviceptr, (usize, usize)>,
}

static POOL: Mutex<Option<Pool>> = Mutex::new(None);

/// One non-blocking helper stream per capturing stream, for host copies.
static COPY_STREAMS: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);

fn copy_stream(stream: sys::CUstream) -> Result<sys::CUstream, DriverError> {
    let mut streams = COPY_STREAMS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let streams = streams.get_or_insert_with(HashMap::new);
    if let Some(copy) = streams.get(&(stream as usize)) {
        return Ok(*copy as sys::CUstream);
    }
    let copy = result::stream::create(result::stream::StreamKind::NonBlocking)?;
    streams.insert(stream as usize, copy as usize);
    Ok(copy)
}

/// Copy `src` to `dst` and wait for it, without involving `stream`.
///
/// # Safety
/// `dst` must be valid for `src.len()` values of `T`, and nothing may be
/// using it.
pub(crate) unsafe fn copy_from_host<T>(
    stream: sys::CUstream,
    dst: sys::CUdeviceptr,
    src: &[T],
) -> Result<(), DriverError> {
    let copy = copy_stream(stream)?;
    result::memcpy_htod_async(dst, src, copy)?;
    result::stream::synchronize(copy)
}

struct Window {
    stream: usize,
    /// Memory dropped during the window: pointer and whether the pool owns it.
    held: Vec<(sys::CUdeviceptr, bool)>,
    /// The most recent upload allocation, the only valid target of a host
    /// copy.
    last_upload: Option<sys::CUdeviceptr>,
}

thread_local! {
    static WINDOW: RefCell<Option<Window>> = const { RefCell::new(None) };
    /// The block this thread last allocated for a host upload outside a
    /// window.
    static UPLOAD_BLOCK: std::cell::Cell<sys::CUdeviceptr> = const { std::cell::Cell::new(0) };
}

/// Note a block allocated outside a window for the host upload that
/// follows. It came from the stream allocator, not the pool.
pub(crate) fn note_upload(pointer: sys::CUdeviceptr) {
    UPLOAD_BLOCK.with(|block| block.set(pointer));
}

/// Whether `pointer` is the block this thread just allocated for a host
/// upload. Any other destination of a host copy may be a pool block that
/// launched work still uses, and the caller must drain the stream first.
pub(crate) fn take_upload(pointer: sys::CUdeviceptr) -> bool {
    UPLOAD_BLOCK.with(|block| block.replace(0) == pointer && pointer != 0)
}

fn block_bytes(bytes: usize) -> usize {
    let bytes = bytes.max(MIN_BLOCK_BYTES);
    if bytes <= LARGE_BLOCK_BYTES {
        bytes.next_power_of_two()
    } else {
        bytes.div_ceil(LARGE_BLOCK_BYTES) * LARGE_BLOCK_BYTES
    }
}

fn with_pool<R>(f: impl FnOnce(&mut Pool) -> R) -> R {
    let mut pool = POOL.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    f(pool.get_or_insert_with(Pool::default))
}

/// Open a capture window for `stream` on this thread. Call before
/// `begin_capture`.
pub fn begin(stream: &CudaStream) -> Result<(), DriverError> {
    WINDOW.with(|window| {
        let mut window = window.borrow_mut();
        if window.is_some() {
            return Err(DriverError(sys::cudaError_enum::CUDA_ERROR_ILLEGAL_STATE));
        }
        *window = Some(Window {
            stream: stream.cu_stream as usize,
            held: Vec::new(),
            last_upload: None,
        });
        Ok(())
    })
}

/// Close this thread's capture window and release what it held. Call after
/// the captured graph has been launched or the capture has been abandoned.
pub fn end(stream: &CudaStream) -> Result<(), DriverError> {
    let Some(window) = WINDOW.with(|window| window.borrow_mut().take()) else {
        return Ok(());
    };
    let mut first_error = None;
    with_pool(|pool| {
        for (pointer, pooled) in window.held {
            if pooled {
                if let Some(key) = pool.owned.get(&pointer).copied() {
                    pool.released.entry(key).or_default().push(pointer);
                }
            } else if let Err(error) = unsafe { result::free_async(pointer, stream.cu_stream) } {
                first_error.get_or_insert(error);
            }
        }
    });
    first_error.map_or(Ok(()), Err)
}

/// Whether this thread has a window open for `stream`.
pub fn is_open(stream: sys::CUstream) -> bool {
    WINDOW.with(|window| {
        window
            .borrow()
            .as_ref()
            .is_some_and(|window| window.stream == stream as usize)
    })
}

/// Tell the pool that `stream` has no work pending: every block released so
/// far is idle and may take a host upload.
pub fn mark_idle(stream: &CudaStream) {
    let stream = stream.cu_stream as usize;
    with_pool(|pool| {
        let keys = pool
            .released
            .keys()
            .filter(|key| key.0 == stream)
            .copied()
            .collect::<Vec<_>>();
        for key in keys {
            if let Some(mut blocks) = pool.released.remove(&key) {
                pool.free.entry(key).or_default().append(&mut blocks);
            }
        }
    });
}

/// Return a stream's pool blocks to the driver. Called when the stream is
/// dropped, before it is destroyed: by then every slice allocated on it is
/// gone, so all of its blocks are in the pool. The stream is drained first,
/// because work it still has pending may write to them. Without this a new
/// stream that is given the same handle would inherit blocks that the old
/// stream's work can still touch.
pub(crate) fn stream_dropped(stream: sys::CUstream) -> Result<(), DriverError> {
    let key = stream as usize;
    let blocks = with_pool(|pool| {
        pool.free.retain(|(owner, _), _| *owner != key);
        pool.released.retain(|(owner, _), _| *owner != key);
        let blocks = pool
            .owned
            .iter()
            .filter(|(_, (owner, _))| *owner == key)
            .map(|(pointer, _)| *pointer)
            .collect::<Vec<_>>();
        for pointer in &blocks {
            pool.owned.remove(pointer);
        }
        blocks
    });
    if blocks.is_empty() {
        return Ok(());
    }
    let drained = unsafe { result::stream::synchronize(stream) };
    let mut first_error = drained.err();
    for pointer in blocks {
        if let Err(error) = unsafe { result::free_sync(pointer) } {
            first_error.get_or_insert(error);
        }
    }
    first_error.map_or(Ok(()), Err)
}

/// Bytes of device memory the pool owns.
pub fn pool_bytes() -> usize {
    with_pool(|pool| pool.owned.values().map(|(_, bytes)| *bytes).sum())
}

/// Allocation from the pool, or `None` when the caller should use the
/// stream allocator.
///
/// With a window open for the stream every request is served. An `upload`
/// block will be written at once by a host copy, so it must be idle; other
/// blocks are only touched by recorded launches.
///
/// Without a window, small requests that are not uploads are served: a
/// released block is reused at once, which is sound because launches on a
/// stream run in the order issued. Uploads keep the stream allocator. With
/// several streams active, uploads into pool blocks were corrupted in
/// about one parallel test run in ten, also into blocks no pending work
/// used; the cause was not found, and with uploads left alone sixty runs
/// were clean.
pub(crate) fn alloc(
    stream: sys::CUstream,
    bytes: usize,
    upload: bool,
) -> Option<Result<sys::CUdeviceptr, DriverError>> {
    if !is_open(stream) && (upload || bytes > DIRECT_POOL_MAX_BYTES) {
        return None;
    }
    let size = block_bytes(bytes);
    let key = (stream as usize, size);
    let reused = with_pool(|pool| {
        let released = if upload {
            None
        } else {
            pool.released.get_mut(&key).and_then(Vec::pop)
        };
        released.or_else(|| pool.free.get_mut(&key).and_then(Vec::pop))
    });
    let pointer = match reused {
        Some(pointer) => Ok(pointer),
        None => unsafe { result::malloc_sync(size) }.inspect(|pointer| {
            with_pool(|pool| pool.owned.insert(*pointer, key));
        }),
    };
    if let (Ok(pointer), true) = (pointer, upload) {
        WINDOW.with(|window| {
            if let Some(window) = window.borrow_mut().as_mut() {
                window.last_upload = Some(pointer);
            }
        });
    }
    Some(pointer)
}

/// Take over the release of `pointer`. Returns `false` when the caller
/// should free it the usual way.
pub(crate) fn release(pointer: sys::CUdeviceptr, stream: sys::CUstream) -> bool {
    let pooled = with_pool(|pool| pool.owned.get(&pointer).copied());
    let held = WINDOW.with(|window| match window.borrow_mut().as_mut() {
        Some(window) if window.stream == stream as usize => {
            window.held.push((pointer, pooled.is_some()));
            true
        }
        _ => false,
    });
    if held {
        return true;
    }
    match pooled {
        Some(key) => {
            with_pool(|pool| pool.released.entry(key).or_default().push(pointer));
            true
        }
        None => false,
    }
}

/// Whether a host copy into `pointer` may run at once in the window: it must
/// be the window's most recent upload allocation, which is idle and which no
/// recorded launch uses yet. Returns `None` when `stream` has no window.
pub(crate) fn host_copy_allowed(stream: sys::CUstream, pointer: sys::CUdeviceptr) -> Option<bool> {
    WINDOW.with(|window| match window.borrow().as_ref() {
        Some(window) if window.stream == stream as usize => {
            Some(window.last_upload == Some(pointer))
        }
        _ => None,
    })
}
