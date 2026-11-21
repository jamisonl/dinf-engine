//! Packed GGUF support for DeepSeek V4 and V4.1 Maya.
//!
//! Keep tensor-header dtypes authoritative, including IQ1_S routed banks and
//! I8 Engram tables. V4.1 loading retains only ordinary resident tensors;
//! `RawGgufSource` owns the validated file for bounded, positional expert/Engram
//! reads. It never maps the file, reads dead gaps, or allocates a whole disk table.

#[cfg(test)]
use crate::verified_file::SparseSetLen;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file::Value;
use candle_core::{DType, Device, Tensor};
use half::{bf16, f16};
use sha2::{Digest, Sha256};

use crate::architectures::deepseek41::engram::FileIdentity;
use crate::verified_file::{Pin, Verification, read_exact_at};

use crate::gguf_limits::{
    MAX_GGUF_ARRAY_ELEMENTS, MAX_GGUF_HEADER_BYTES, MAX_GGUF_METADATA_COUNT, MAX_GGUF_TENSOR_COUNT,
    validate_gguf_header_budget,
};

const GGUF_MAGIC: u32 = 0x4655_4747;
const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_STRING_LENGTH: u64 = MAX_GGUF_HEADER_BYTES;
const MAX_TENSOR_DIMS: u32 = 4;
const MAX_VALUE_DEPTH: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Version {
    V1,
    V2,
    V3,
}

impl Version {
    fn read(file: &mut File) -> Result<Self> {
        let magic = read_u32(file)?;
        anyhow::ensure!(
            magic == GGUF_MAGIC || magic == 0x4747_5546,
            "unknown GGUF magic 0x{magic:08x}"
        );
        match read_u32(file)? {
            1 => Ok(Self::V1),
            2 => Ok(Self::V2),
            3 => Ok(Self::V3),
            version => anyhow::bail!("unsupported GGUF version {version}"),
        }
    }

    fn read_len(self, file: &mut File) -> Result<u64> {
        match self {
            Self::V1 => Ok(read_u32(file)? as u64),
            Self::V2 | Self::V3 => read_u64(file),
        }
    }
}

/// GGML type IDs used by the V4 GGUF files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeepSeek4DType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    I8,
    I32,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    IQ1S,
    IQ1M,
    IQ4NL,
    IQ4XS,
    IQ2S,
    IQ3S,
    BF16,
    MXFP4,
    /// NVIDIA FP4: E2M1 values with one UE4M3 scale per 16.
    NVFP4,
    /// FP8 E4M3 values (no block scale).
    F8E4M3,
    Unknown(u32),
}

impl DeepSeek4DType {
    pub fn from_u32(value: u32) -> Self {
        match value {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::IQ2XXS,
            17 => Self::IQ2XS,
            18 => Self::IQ3XXS,
            19 => Self::IQ1S,
            20 => Self::IQ4NL,
            21 => Self::IQ3S,
            22 => Self::IQ2S,
            23 => Self::IQ4XS,
            24 => Self::I8,
            26 => Self::I32,
            29 => Self::IQ1M,
            30 => Self::BF16,
            39 => Self::MXFP4,
            40 => Self::NVFP4,
            43 => Self::F8E4M3,
            value => Self::Unknown(value),
        }
    }

    pub fn ggml_id(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            Self::IQ2XXS => 16,
            Self::IQ2XS => 17,
            Self::IQ3XXS => 18,
            Self::IQ1S => 19,
            Self::IQ4NL => 20,
            Self::IQ3S => 21,
            Self::IQ2S => 22,
            Self::IQ4XS => 23,
            Self::I8 => 24,
            Self::I32 => 26,
            Self::IQ1M => 29,
            Self::BF16 => 30,
            Self::MXFP4 => 39,
            Self::NVFP4 => 40,
            Self::F8E4M3 => 43,
            Self::Unknown(value) => value,
        }
    }

    /// Return `(elements_per_block, bytes_per_block)` for types used by the
    /// V4 model and the older types that can occur in its shared tensors.
    pub fn block_layout(self) -> Option<(usize, usize)> {
        Some(match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::I8 => (1, 1),
            Self::I32 => (1, 4),
            Self::BF16 => (1, 2),
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 36),
            Self::Q2K => (256, 84),
            Self::Q3K => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Q8K => (256, 292),
            Self::IQ2XXS => (256, 66),
            Self::IQ2XS => (256, 74),
            Self::IQ3XXS => (256, 98),
            Self::IQ1S => (256, 50),
            Self::IQ1M => (256, 56),
            Self::IQ4NL => (32, 18),
            Self::IQ4XS => (256, 136),
            Self::IQ2S => (256, 82),
            Self::IQ3S => (256, 110),
            Self::MXFP4 => (32, 17),
            Self::NVFP4 => (64, 36),
            Self::F8E4M3 => (1, 1),
            Self::Unknown(_) => return None,
        })
    }

    pub fn is_packed(self) -> bool {
        self.block_layout().is_some_and(|(block, _)| block > 1)
    }

    pub fn byte_len(self, elements: usize) -> Result<usize> {
        let (block, size) = self
            .block_layout()
            .with_context(|| format!("unsupported GGML tensor type {}", self.ggml_id()))?;
        anyhow::ensure!(
            elements.is_multiple_of(block),
            "tensor with type {:?} has {elements} elements, not divisible by block size {block}",
            self
        );
        elements
            .checked_div(block)
            .and_then(|blocks| blocks.checked_mul(size))
            .context("GGUF tensor byte length overflow")
    }

    /// The Candle quantization type with the same GGML ID, if Candle has one.
    pub(crate) fn candle_dtype(self) -> Option<candle_core::quantized::GgmlDType> {
        use candle_core::quantized::GgmlDType as G;
        Some(match self {
            Self::F32 => G::F32,
            Self::F16 => G::F16,
            Self::BF16 => G::BF16,
            Self::Q4_0 => G::Q4_0,
            Self::Q4_1 => G::Q4_1,
            Self::Q5_0 => G::Q5_0,
            Self::Q5_1 => G::Q5_1,
            Self::Q8_0 => G::Q8_0,
            Self::Q8_1 => G::Q8_1,
            Self::Q2K => G::Q2K,
            Self::Q3K => G::Q3K,
            Self::Q4K => G::Q4K,
            Self::Q5K => G::Q5K,
            Self::Q6K => G::Q6K,
            Self::Q8K => G::Q8K,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct RawTensorInfo {
    pub shape: Vec<usize>,
    pub dtype: DeepSeek4DType,
    pub offset: u64,
}

impl RawTensorInfo {
    pub fn element_count(&self) -> Result<usize> {
        self.shape.iter().try_fold(1usize, |count, dim| {
            count
                .checked_mul(*dim)
                .context("GGUF tensor element count overflow")
        })
    }

    pub fn byte_len(&self) -> Result<usize> {
        self.dtype.byte_len(self.element_count()?)
    }
}

/// Admission charge for Maya's exact `from_vec(U8)` ordinary Q8 uploads.
/// This is recovered accounting allowance, not newly freed device memory.
pub(crate) fn maya_q8_upload_charge(bytes: u64) -> Result<u64> {
    anyhow::ensure!(bytes > 0, "empty Maya Q8 upload");
    Ok(bytes
        .checked_add(16383)
        .context("Maya Q8 upload charge overflow")?
        & !16383)
}

/// Verify the actual exact-upload contract before publishing any Maya weights.
/// Numeric result: logical bytes, buffer length, allocatedSize, admission charge.
#[cfg(target_os = "macos")]
pub(crate) fn verify_maya_q8_upload(
    raw: &RawTensor,
    info: &RawTensorInfo,
    device: &Device,
) -> Result<[u64; 4]> {
    use candle_core::backend::BackendStorage;
    use objc2_metal::MTLResource;
    let expected = info.byte_len()?;
    let charge = maya_q8_upload_charge(expected as u64)?;
    let bytes = raw.bytes();
    anyhow::ensure!(
        info.dtype == DeepSeek4DType::Q8_0
            && raw.dtype() == info.dtype
            && raw.shape() == info.shape
            && raw.byte_len() == expected
            && bytes.dtype() == DType::U8
            && bytes.dims() == [expected]
            && bytes.is_contiguous()
            && bytes.device().same_device(device),
        "Maya Q8 upload metadata/layout/device mismatch"
    );
    let (storage, layout) = bytes.storage_and_layout();
    let candle_core::Storage::Metal(storage) = &*storage else {
        anyhow::bail!("Maya Q8 upload requires actual Metal storage");
    };
    let metal = device.as_metal_device()?;
    let buffer = storage.buffer();
    let length = buffer.length() as u64;
    let allocated = buffer.as_ref().allocatedSize() as u64;
    anyhow::ensure!(
        storage.dtype() == DType::U8
            && storage.device().id() == metal.id()
            && layout.start_offset() == 0
            && std::ptr::eq(
                buffer.as_ref().device().as_ref(),
                metal.metal_device().as_ref()
            )
            && length == expected as u64
            && allocated >= length
            && allocated <= charge,
        "Maya Q8 exact upload bound failed: logical={expected} length={length} allocated={allocated} charge={charge} offset={}",
        layout.start_offset()
    );
    Ok([expected as u64, length, allocated, charge])
}

// Native qualifier: numeric metadata only, never retains an extra tensor owner.
#[cfg(all(test, target_os = "macos"))]
#[derive(Default)]
pub(crate) struct MayaDenseProof {
    pub device: Option<Device>,
    pub copies: usize,
    pub logical: u64,
    pub length: u64,
    pub allocated: u64,
    pub charged: u64,
    pub peak: u64,
}
#[cfg(all(test, target_os = "macos"))]
thread_local! {
    pub(crate) static MAYA_DENSE_PROOF: std::cell::RefCell<MayaDenseProof> = Default::default();
}
#[cfg(all(test, target_os = "macos"))]
pub(crate) fn observe_maya_dense_copy(raw: &RawTensor, dense: &Tensor) -> Result<()> {
    use candle_core::backend::BackendStorage;
    use objc2_metal::MTLResource;
    MAYA_DENSE_PROOF.with(|p| {
        let mut p = p.borrow_mut();
        let Some(device) = &p.device else { return Ok(()); };
        if dense.device().is_cpu() { return Ok(()); } // CPU bias remains unchanged.
        anyhow::ensure!(dense.device().same_device(device) && dense.dtype() == DType::F32
            && dense.dims() == raw.shape() && dense.is_contiguous(), "Maya dense copy shape/dtype/device");
        let (storage,layout) = dense.storage_and_layout();
        let candle_core::Storage::Metal(storage) = &*storage else { anyhow::bail!("dense Metal storage"); };
        let metal = device.as_metal_device()?;
        let logical = dense.elem_count().checked_mul(4).context("dense span overflow")? as u64;
        let charge = logical.checked_next_power_of_two().context("dense bucket overflow")?;
        let length = storage.buffer().length() as u64;
        let allocated = storage.buffer().as_ref().allocatedSize() as u64;
        anyhow::ensure!(layout.start_offset() == 0 && storage.dtype() == DType::F32
            && storage.device().id() == metal.id()
            && std::ptr::eq(storage.buffer().as_ref().device().as_ref(),metal.metal_device().as_ref())
            && length >= logical && length <= charge && allocated >= length && allocated <= charge,
            "Maya dense copy exceeds unchanged bound: logical={logical} length={length} allocated={allocated} charge={charge}");
        let current = metal.device().current_allocated_size() as u64;
        p.copies += 1; p.logical += logical; p.length += length;
        p.allocated += allocated; p.charged += charge; p.peak = p.peak.max(current);
        Ok(())
    })
}

/// A packed V4 tensor kept as a byte tensor on the target device.
pub struct RawTensor {
    shape: Vec<usize>,
    dtype: DeepSeek4DType,
    bytes: Tensor,
}

impl RawTensor {
    pub(crate) fn new(shape: Vec<usize>, dtype: DeepSeek4DType, bytes: Tensor) -> Result<Self> {
        anyhow::ensure!(bytes.dtype() == DType::U8, "raw GGUF storage must use u8");
        Ok(Self {
            shape,
            dtype,
            bytes,
        })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> DeepSeek4DType {
        self.dtype
    }

    pub fn bytes(&self) -> &Tensor {
        &self.bytes
    }

    pub fn byte_len(&self) -> usize {
        self.bytes.elem_count()
    }

    /// View one row of a packed rank-3 expert tensor as a rank-2 matrix.
    ///
    /// GGUF stores the expert matrices back to back. The returned byte tensor
    /// keeps that layout on the device, so a fused matmul can read the row
    /// without copying a full expert.
    pub fn slice_first_dim(&self, index: usize) -> Result<Self> {
        anyhow::ensure!(self.shape.len() == 3, "expected a rank-3 expert tensor");
        anyhow::ensure!(
            index < self.shape[0],
            "expert index {index} is out of range"
        );
        let sliced = self.slice_rows(index, 1)?;
        Self::new(vec![self.shape[1], self.shape[2]], self.dtype, sliced.bytes)
    }

    /// View a contiguous range of rows from a packed matrix or expert tensor.
    pub fn slice_rows(&self, start_row: usize, row_count: usize) -> Result<Self> {
        anyhow::ensure!(self.shape.len() >= 2, "expected a rank-2 or rank-3 tensor");
        anyhow::ensure!(row_count > 0, "row slice cannot be empty");
        anyhow::ensure!(
            start_row
                .checked_add(row_count)
                .is_some_and(|end| end <= self.shape[0]),
            "row slice [{start_row}, {}) is out of range",
            start_row.saturating_add(row_count)
        );
        let row_elements = self.shape[1..]
            .iter()
            .try_fold(1usize, |count, dim| count.checked_mul(*dim))
            .context("tensor row element count overflow")?;
        let row_bytes = self.dtype.byte_len(row_elements)?;
        let start = start_row
            .checked_mul(row_bytes)
            .context("tensor row byte offset overflow")?;
        let byte_count = row_count
            .checked_mul(row_bytes)
            .context("tensor row byte count overflow")?;
        let bytes = self.bytes.narrow(0, start, byte_count)?;
        let mut shape = self.shape.clone();
        shape[0] = row_count;
        Self::new(shape, self.dtype, bytes)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.bytes.to_vec1::<u8>().map_err(Into::into)
    }

    /// Decode a dense shared tensor. Packed quant types must use a V4 kernel.
    pub fn to_dense(&self, device: &Device) -> Result<Tensor> {
        let raw = self.to_bytes()?;
        let elements = self
            .shape
            .iter()
            .try_fold(1usize, |count, dim| count.checked_mul(*dim))
            .context("GGUF tensor element count overflow")?;
        match self.dtype {
            DeepSeek4DType::F32 => {
                let expected_bytes = elements
                    .checked_mul(4)
                    .context("GGUF F32 tensor byte length overflow")?;
                anyhow::ensure!(raw.len() == expected_bytes, "bad F32 tensor byte length");
                let values = raw
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect::<Vec<_>>();
                Tensor::from_vec(values, self.shape.clone(), device).map_err(Into::into)
            }
            DeepSeek4DType::F16 => {
                let expected_bytes = elements
                    .checked_mul(2)
                    .context("GGUF F16 tensor byte length overflow")?;
                anyhow::ensure!(raw.len() == expected_bytes, "bad F16 tensor byte length");
                let values = raw
                    .chunks_exact(2)
                    .map(|chunk| f16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())))
                    .collect::<Vec<_>>();
                Tensor::from_vec(values, self.shape.clone(), device).map_err(Into::into)
            }
            DeepSeek4DType::BF16 => {
                let expected_bytes = elements
                    .checked_mul(2)
                    .context("GGUF BF16 tensor byte length overflow")?;
                anyhow::ensure!(raw.len() == expected_bytes, "bad BF16 tensor byte length");
                let values = raw
                    .chunks_exact(2)
                    .map(|chunk| bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())))
                    .collect::<Vec<_>>();
                Tensor::from_vec(values, self.shape.clone(), device).map_err(Into::into)
            }
            DeepSeek4DType::I32 => {
                let expected_bytes = elements
                    .checked_mul(4)
                    .context("GGUF I32 tensor byte length overflow")?;
                anyhow::ensure!(raw.len() == expected_bytes, "bad I32 tensor byte length");
                let values = raw
                    .chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect::<Vec<_>>();
                Tensor::from_vec(values, self.shape.clone(), device).map_err(Into::into)
            }
            dtype => anyhow::bail!("dense decode is not valid for packed type {dtype:?}"),
        }
    }
}

/// Parsed GGUF header. Tensor bytes remain on disk until `load_tensor` is
/// called for a tensor in the selected pipeline stage.
pub struct RawGgufContent {
    pub metadata: HashMap<String, Value>,
    pub tensor_infos: HashMap<String, RawTensorInfo>,
    pub tensor_data_offset: u64,
    file_size: u64,
    header_stats: crate::gguf_limits::GgufHeaderStats,
}

/// Maximum allocation/read for a lazy disk request (one Maya expert is ~2.3 MiB).
pub const MAX_SOURCE_READ_BYTES: usize = 16 * 1024 * 1024;
/// Lazy GPU expert-cache reservation, not an up-front allocation. The graph must
/// enforce this bound against actual device allocations (including Metal buckets).
pub const DEEPSEEK41_EXPERT_CACHE_BYTES: u64 = 72 * 1024 * 1024 * 1024;

// Production verification is deliberately locked to the one supported artifact.
pub(crate) const MAYA_ARTIFACT_BYTES: u64 = 365_713_686_528;
pub(crate) const MAYA_ARTIFACT_SHA256: [u8; 32] = [
    0xf4, 0x14, 0x99, 0x5b, 0x37, 0x97, 0xf7, 0x70, 0x09, 0x63, 0x4f, 0xac, 0x99, 0x5f, 0x50, 0x83,
    0x4e, 0x70, 0x39, 0x37, 0x71, 0xf9, 0x42, 0x3f, 0xca, 0x98, 0x4c, 0x89, 0x25, 0x03, 0xb4, 0x2a,
];
const SHA_SCRATCH_BYTES: usize = 1024 * 1024;

pub(crate) fn maya_pin() -> Pin {
    Pin {
        model_id: "deepseek-v4.1-flash-1017-maya-xs".into(),
        source_revision: "7832036cb557c6ccdacdff77ef4e6ea283e35f89".into(),
        native_version: 1,
        expected_bytes: MAYA_ARTIFACT_BYTES,
        expected_sha256: MAYA_ARTIFACT_SHA256,
    }
}

#[cfg(test)]
std::thread_local! {
    static ARTIFACT_SHA_PASSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn check_source_identity(file: &File, expected: FileIdentity) -> Result<()> {
    anyhow::ensure!(
        FileIdentity::from_metadata(&file.metadata()?)? == expected,
        "GGUF source identity/length/timestamps changed"
    );
    Ok(())
}

#[cfg(any(target_os = "macos", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HeaderIoPolicy {
    NoCache(bool),
    ReadAhead(bool),
}

#[cfg(target_os = "macos")]
fn set_header_io_policy(fd: std::os::fd::RawFd, policy: HeaderIoPolicy) -> Result<()> {
    let (command, value) = match policy {
        HeaderIoPolicy::NoCache(value) => (libc::F_NOCACHE, i32::from(value)),
        HeaderIoPolicy::ReadAhead(value) => (libc::F_RDAHEAD, i32::from(value)),
    };
    // These fcntl commands affect only the retained owned descriptor.
    if unsafe { libc::fcntl(fd, command, value) } == -1 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("setting GGUF descriptor policy {policy:?}"));
    }
    Ok(())
}

#[cfg(any(target_os = "macos", test))]
fn with_header_io_policy<T>(
    mut set: impl FnMut(HeaderIoPolicy) -> Result<()>,
    read: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let result = set(HeaderIoPolicy::NoCache(false))
        .and_then(|()| set(HeaderIoPolicy::ReadAhead(true)))
        .and_then(|()| read());
    // Attempt BOTH restores even after partial entry or a failed read/restore.
    // Never publish parsed content unless both descriptor policies are restored.
    let cache = set(HeaderIoPolicy::NoCache(true)).context("restoring GGUF NOCACHE");
    let readahead = set(HeaderIoPolicy::ReadAhead(false)).context("restoring GGUF RDAHEAD");
    cache?;
    readahead?;
    result
}

// Fixture digests exercise this helper without creating a production proof.
fn verify_artifact(
    file: &File,
    identity: FileIdentity,
    expected_bytes: u64,
    expected_sha256: &[u8; 32],
    mut read: impl FnMut(&File, &mut [u8], u64) -> std::io::Result<()>,
) -> Result<()> {
    check_source_identity(file, identity)?;
    anyhow::ensure!(
        identity.length == expected_bytes,
        "Maya artifact length mismatch: expected {expected_bytes}, got {}",
        identity.length
    );
    #[cfg(test)]
    ARTIFACT_SHA_PASSES.with(|passes| passes.set(passes.get() + 1));
    let mut scratch = vec![0; SHA_SCRATCH_BYTES];
    let mut hash = Sha256::new();
    let mut offset = 0;
    while offset < expected_bytes {
        let count = (expected_bytes - offset).min(scratch.len() as u64) as usize;
        read(file, &mut scratch[..count], offset)?;
        hash.update(&scratch[..count]);
        offset += count as u64;
    }
    check_source_identity(file, identity)?;
    anyhow::ensure!(
        hash.finalize().as_slice() == expected_sha256,
        "Maya artifact SHA256 mismatch"
    );
    Ok(())
}

/// Immutable, opened GGUF source. No path reopening, mmap, or whole-bank cache.
/// Metadata and spans are Arc-shared internally; callers may Arc-share the source.
/// Each dynamic read is capped at `MAX_SOURCE_READ_BYTES`. Positional I/O permits
/// concurrent readers without a shared seek cursor. Full identity is checked
/// before and after every read, including empty reads.
/// Keep the artifact immutable during inference: timestamp checks detect
/// ordinary mutation, not a hostile writer able to restore file metadata.
/// On macOS this descriptor disables read-ahead and page caching except while
/// parsing the header. The 256 MiB parser budget is logical; OS read-ahead may
/// fetch beyond it. Disk-only policies are restored before any source is exposed.
pub struct RawGgufSource {
    file: File,
    content: Arc<RawGgufContent>,
    identity: FileIdentity,
    verified_maya: bool,
    verification: Option<Verification>,
    parts: Vec<RawGgufSource>,
    tensor_parts: HashMap<String, usize>,
}

impl RawGgufSource {
    /// Unverified header inspection / synthetic reader constructor. This source
    /// cannot construct a production Maya graph.
    pub fn new(file: File) -> Result<Self> {
        Self::from_owned_file(file, false, None)
    }

    /// Explicit fullcheck entry for fixtures; never consults/publishes receipts.
    #[cfg(test)]
    pub(crate) fn verified_maya(file: File) -> Result<Self> {
        Self::from_owned_file(file, true, None)
    }

    /// Both native constructors use this path. Only one artifact open; cache
    /// paths supply a receipt location, never a second source of trusted bytes.
    pub(crate) fn open_maya(path: &Path) -> Result<Self> {
        Self::from_owned_file(File::open(path)?, true, Some(path))
    }
    /// Retain and verify all six immutable Coder parts before tensor allocation.
    pub(crate) fn open_coder(path: &Path) -> Result<Self> {
        use crate::gsq_coder::{MODEL_ID, PARTS, PREFIX, REVISION};
        let parent = path
            .parent()
            .context("split GGUF requires a parent directory")?;
        anyhow::ensure!(
            path.file_name().and_then(|s| s.to_str())
                == Some(format!("{PREFIX}-00001-of-00006.gguf").as_str()),
            "Coder entry must be canonical metadata shard one"
        );
        let mut parts = Vec::with_capacity(6);
        for (index, &(expected_bytes, digest)) in PARTS.iter().enumerate() {
            let part_path = parent.join(format!("{PREFIX}-{:05}-of-00006.gguf", index + 1));
            let mut source = Self::new(
                File::open(&part_path)
                    .with_context(|| format!("missing Coder part {}", index + 1))?,
            )?;
            let mut expected_sha256 = [0u8; 32];
            for (i, byte) in expected_sha256.iter_mut().enumerate() {
                *byte = u8::from_str_radix(&digest[i * 2..i * 2 + 2], 16)?;
            }
            let pin = Pin {
                model_id: format!("{MODEL_ID}/{}", index + 1),
                source_revision: REVISION.into(),
                native_version: 1,
                expected_bytes,
                expected_sha256,
            };
            source.verification = Some(crate::verified_file::verify_cached(
                &source.file,
                &part_path,
                source.identity,
                &pin,
                || {
                    verify_artifact(
                        &source.file,
                        source.identity,
                        expected_bytes,
                        &expected_sha256,
                        read_exact_at,
                    )
                },
            )?);
            let c = source.content();
            for (key, expected) in [
                ("split.no", index as u64),
                ("split.count", 6),
                ("split.tensors.count", 1043),
            ] {
                anyhow::ensure!(
                    c.metadata.get(key).and_then(value_as_u64) == Some(expected),
                    "Coder part {} has invalid {key}",
                    index + 1
                );
            }
            for (name, info) in &c.tensor_infos {
                let end = c
                    .tensor_data_offset
                    .checked_add(info.offset)
                    .and_then(|start| start.checked_add(info.byte_len().ok()? as u64))
                    .context("Coder tensor span overflow")?;
                anyhow::ensure!(
                    end <= source.identity.length,
                    "Coder tensor {name} extends outside its part"
                );
            }
            parts.push(source);
        }
        anyhow::ensure!(
            parts[0].content.tensor_infos.is_empty(),
            "Coder metadata shard unexpectedly contains tensors"
        );
        let mut root = parts.remove(0);
        let mut merged = HashMap::new();
        for (index, part) in parts.iter().enumerate() {
            for (name, info) in &part.content.tensor_infos {
                anyhow::ensure!(
                    merged.insert(name.clone(), info.clone()).is_none(),
                    "duplicate split tensor {name}"
                );
                root.tensor_parts.insert(name.clone(), index);
            }
        }
        anyhow::ensure!(merged.len() == 1043, "Coder split tensor count mismatch");
        Arc::get_mut(&mut root.content)
            .context("shared split header during construction")?
            .tensor_infos = merged;
        root.parts = parts;
        Ok(root)
    }

    pub(crate) fn require_verified_coder(&self) -> Result<()> {
        anyhow::ensure!(
            self.parts.len() == 5
                && self.verification.is_some()
                && self.parts.iter().all(|p| p.verification.is_some()),
            "Coder source lacks six pinned artifact proofs"
        );
        check_source_identity(&self.file, self.identity)?;
        for part in &self.parts {
            check_source_identity(&part.file, part.identity)?;
        }
        Ok(())
    }
    pub(crate) fn is_coder(&self) -> bool {
        !self.parts.is_empty()
    }

    pub(crate) fn require_verified_native(&self) -> Result<()> {
        if self.is_coder() {
            self.require_verified_coder()
        } else {
            self.require_verified_maya()
        }
    }

    pub(crate) fn tensor_file(&self, name: &str) -> Result<(File, FileIdentity, u64)> {
        if let Some(&part) = self.tensor_parts.get(name) {
            return self.parts[part].tensor_file(name);
        }
        let info = self
            .content
            .tensor_infos
            .get(name)
            .context("missing source tensor")?;
        let offset = self
            .content
            .tensor_data_offset
            .checked_add(info.offset)
            .context("tensor offset overflow")?;
        Ok((self.clone_file()?, self.identity, offset))
    }

    fn from_owned_file(
        mut file: File,
        verified_maya: bool,
        cache_path: Option<&Path>,
    ) -> Result<Self> {
        let identity = FileIdentity::from_metadata(&file.metadata()?)?;
        #[cfg(target_os = "macos")]
        {
            use std::os::fd::AsRawFd;
            set_header_io_policy(file.as_raw_fd(), HeaderIoPolicy::NoCache(true))?;
            set_header_io_policy(file.as_raw_fd(), HeaderIoPolicy::ReadAhead(false))?;
        }
        let verification = if verified_maya {
            let pin = maya_pin();
            let fullcheck = || {
                verify_artifact(
                    &file,
                    identity,
                    pin.expected_bytes,
                    &pin.expected_sha256,
                    read_exact_at,
                )
            };
            Some(if let Some(path) = cache_path {
                crate::verified_file::verify_cached(&file, path, identity, &pin, fullcheck)?
            } else {
                fullcheck()?;
                Verification {
                    full_passes: 1,
                    full_read_bytes: identity.length,
                    ..Default::default()
                }
            })
        } else {
            None
        };
        check_source_identity(&file, identity)?;
        #[cfg(target_os = "macos")]
        let content = {
            use std::os::fd::AsRawFd;
            let fd = file.as_raw_fd();
            with_header_io_policy(
                |policy| set_header_io_policy(fd, policy),
                || RawGgufContent::read(&mut file),
            )?
        };
        #[cfg(not(target_os = "macos"))]
        let content = RawGgufContent::read(&mut file)?;
        let content = Arc::new(content);
        check_source_identity(&file, identity)?;
        Ok(Self {
            file,
            content,
            identity,
            verified_maya,
            verification,
            parts: Vec::new(),
            tensor_parts: HashMap::new(),
        })
    }

    /// Require fresh pinned full SHA or a validated local receipt of prior full
    /// SHA, then recheck this retained descriptor's immutable source identity.
    pub(crate) fn require_verified_maya(&self) -> Result<()> {
        anyhow::ensure!(
            self.verified_maya,
            "Maya source lacks pinned artifact SHA256 proof (fresh fullcheck or validated prior-fullSHA local receipt)"
        );
        check_source_identity(&self.file, self.identity)
    }

    /// Complete typed header, shared without tokenizer/Engram metadata copies.
    pub fn content(&self) -> &RawGgufContent {
        &self.content
    }

    /// Duplicate the validated owned descriptor; never resolve its old path.
    pub(crate) fn clone_file(&self) -> Result<File> {
        check_source_identity(&self.file, self.identity)?;
        Ok(self.file.try_clone()?)
    }

    pub(crate) fn file_identity(&self) -> FileIdentity {
        self.identity
    }

    pub(crate) fn verification(&self) -> Option<Verification> {
        self.verification
    }

    /// Startup-only resident load; disk tables and routed banks are forbidden.
    pub(crate) fn load_resident_tensor(&self, name: &str, device: &Device) -> Result<RawTensor> {
        anyhow::ensure!(
            !name.ends_with("_exps.weight")
                && !name.contains("engram_embd")
                && name != "per_layer_token_embd.weight",
            "disk-only tensor cannot be loaded whole"
        );
        let info = self
            .content
            .tensor_infos
            .get(name)
            .context("missing resident tensor")?;
        if let Some(&part) = self.tensor_parts.get(name) {
            return self.parts[part].load_resident_tensor(name, device);
        }
        let length = info.byte_len()?;
        check_source_identity(&self.file, self.identity)?;
        let mut bytes = vec![0; length];
        // Each I/O request remains bounded; only genuinely resident tensors
        // receive a complete allocation. Never read the gap between tensors.
        for (index, chunk) in bytes.chunks_mut(MAX_SOURCE_READ_BYTES).enumerate() {
            let offset = index * MAX_SOURCE_READ_BYTES;
            self.read_tensor_into(name, offset as u64, chunk)?;
        }
        check_source_identity(&self.file, self.identity)?;
        RawTensor::new(
            info.shape.clone(),
            info.dtype,
            Tensor::from_vec(bytes, length, device)?,
        )
    }

    /// Read only within a validated named tensor, never a metadata-provided path.
    pub fn read_tensor_bytes(&self, name: &str, offset: u64, length: usize) -> Result<Vec<u8>> {
        anyhow::ensure!(
            length <= MAX_SOURCE_READ_BYTES,
            "GGUF source read exceeds bounded I/O limit"
        );
        let mut bytes = vec![0; length];
        self.read_tensor_into(name, offset, &mut bytes)?;
        Ok(bytes)
    }

    /// Read into caller-owned bounded scratch, suitable for Engram row batches.
    pub fn read_tensor_into(&self, name: &str, offset: u64, bytes: &mut [u8]) -> Result<()> {
        self.read_tensor_into_with(name, offset, bytes, read_exact_at)
    }

    fn read_tensor_into_with(
        &self,
        name: &str,
        offset: u64,
        bytes: &mut [u8],
        read: impl FnMut(&File, &mut [u8], u64) -> std::io::Result<()>,
    ) -> Result<()> {
        self.read_tensor_into_selected(name, offset, bytes, read, false)
    }

    // Only read_expert_slice opts in; resident loads and whole-file SHA do not.
    fn read_tensor_into_selected(
        &self,
        name: &str,
        offset: u64,
        bytes: &mut [u8],
        mut read: impl FnMut(&File, &mut [u8], u64) -> std::io::Result<()>,
        selected: bool,
    ) -> Result<()> {
        use crate::architectures::deepseek41::profile::{self, Detail};
        if let Some(&part) = self.tensor_parts.get(name) {
            return self.parts[part].read_tensor_into_selected(name, offset, bytes, read, selected);
        }
        let metadata = selected
            .then(|| profile::detail(Detail::ExpertMetadataPre))
            .flatten();
        let length = bytes.len();
        anyhow::ensure!(
            length <= MAX_SOURCE_READ_BYTES,
            "GGUF source read exceeds bounded I/O limit"
        );
        let info = self
            .content
            .tensor_infos
            .get(name)
            .context("missing GGUF source tensor")?;
        let end = offset
            .checked_add(length as u64)
            .context("GGUF read overflow")?;
        anyhow::ensure!(
            end <= info.byte_len()? as u64,
            "GGUF read outside tensor span"
        );
        check_source_identity(&self.file, self.identity)?;
        let start = self
            .content
            .tensor_data_offset
            .checked_add(info.offset)
            .and_then(|start| start.checked_add(offset))
            .context("GGUF read offset overflow")?;
        drop(metadata);
        let reading = selected
            .then(|| profile::read_detail(Detail::ExpertRead, length))
            .flatten();
        read(&self.file, bytes, start)?;
        profile::read_completed(&reading, length);
        drop(reading);
        let metadata = selected
            .then(|| profile::detail(Detail::ExpertMetadataPost))
            .flatten();
        check_source_identity(&self.file, self.identity)?;
        drop(metadata);
        Ok(())
    }

    /// Load one first-dimension routed expert slice, preserving packed IQ1_S.
    pub fn read_expert_slice(
        &self,
        name: &str,
        expert: usize,
        device: &Device,
    ) -> Result<RawTensor> {
        let info = self
            .content
            .tensor_infos
            .get(name)
            .context("missing expert bank")?;
        let valid_dtype = if self.is_coder() {
            matches!(
                info.dtype,
                DeepSeek4DType::IQ1M | DeepSeek4DType::IQ2XXS | DeepSeek4DType::IQ2XS
            )
        } else {
            info.dtype == DeepSeek4DType::IQ1S
        };
        anyhow::ensure!(
            name.ends_with("_exps.weight") && info.shape.len() == 3 && valid_dtype,
            "unsupported native expert bank layout"
        );
        anyhow::ensure!(expert < info.shape[0], "expert index outside bank");
        let elements = info.shape[1]
            .checked_mul(info.shape[2])
            .context("expert shape overflow")?;
        let length = info.dtype.byte_len(elements)?;
        let offset = expert
            .checked_mul(length)
            .context("expert offset overflow")?;
        use crate::architectures::deepseek41::profile::{self, Detail};
        anyhow::ensure!(
            length <= MAX_SOURCE_READ_BYTES,
            "GGUF source read exceeds bounded I/O limit"
        );
        let allocating = profile::detail(Detail::ExpertUpload);
        let mut bytes = vec![0; length];
        drop(allocating);
        self.read_tensor_into_selected(name, offset as u64, &mut bytes, read_exact_at, true)?;
        let _upload = profile::detail(Detail::ExpertUpload);
        RawTensor::new(
            info.shape[1..].to_vec(),
            info.dtype,
            Tensor::from_vec(bytes, length, device)?,
        )
    }
}

impl RawGgufContent {
    pub fn read(file: &mut File) -> Result<Self> {
        // Run the bounded aggregate scan first.  This also covers
        // callers that enter the DeepSeek/raw path directly rather than via
        // the standard GGUF loader preflight.
        let header_stats = validate_gguf_header_budget(file)?;
        let file_size = file.metadata()?.len();
        file.seek(SeekFrom::Start(0))?;
        let version = Version::read(file)?;
        let tensor_count = version.read_len(file)?;
        let metadata_count = version.read_len(file)?;
        anyhow::ensure!(
            tensor_count <= MAX_GGUF_TENSOR_COUNT,
            "GGUF tensor count ({tensor_count}) exceeds application limit ({MAX_GGUF_TENSOR_COUNT})"
        );
        anyhow::ensure!(
            metadata_count <= MAX_GGUF_METADATA_COUNT,
            "GGUF metadata count ({metadata_count}) exceeds application limit ({MAX_GGUF_METADATA_COUNT})"
        );

        // Every metadata entry contains at least its length-prefixed key, a
        // value type, and one value byte.  Tensor records contain a
        // length-prefixed name, dimensions count, dtype, and offset.  Check
        // the lower bound before reserving any collection capacity so forged
        // counts cannot turn a sparse/short file into a large allocation.
        let fixed_header_bytes = match version {
            Version::V1 => 16u64,
            Version::V2 | Version::V3 => 24u64,
        };
        let length_prefix_bytes = match version {
            Version::V1 => 4u64,
            Version::V2 | Version::V3 => 8u64,
        };
        let metadata_min_bytes = length_prefix_bytes
            .checked_add(5)
            .context("GGUF metadata minimum size overflow")?;
        let tensor_min_bytes = length_prefix_bytes
            .checked_add(16)
            .context("GGUF tensor minimum size overflow")?;
        let minimum_header_bytes = metadata_count
            .checked_mul(metadata_min_bytes)
            .and_then(|bytes| {
                tensor_count
                    .checked_mul(tensor_min_bytes)
                    .and_then(|tensor_bytes| bytes.checked_add(tensor_bytes))
            })
            .context("GGUF header minimum size overflow")?;
        let header_budget = MAX_GGUF_HEADER_BYTES
            .checked_sub(fixed_header_bytes)
            .context("GGUF fixed header exceeds application limit")?;
        anyhow::ensure!(
            minimum_header_bytes <= header_budget,
            "GGUF header minimum size ({minimum_header_bytes} bytes) exceeds application limit"
        );
        let header_end = file.stream_position()?;
        let remaining = file_size.saturating_sub(header_end);
        anyhow::ensure!(
            minimum_header_bytes <= remaining,
            "GGUF header declares more entries than the remaining file can contain"
        );

        let metadata_capacity =
            usize::try_from(metadata_count).context("GGUF metadata count does not fit usize")?;
        let tensor_capacity =
            usize::try_from(tensor_count).context("GGUF tensor count does not fit usize")?;
        let mut metadata = HashMap::new();
        metadata
            .try_reserve(metadata_capacity)
            .map_err(|error| anyhow::anyhow!("GGUF metadata allocation failed: {error}"))?;
        for _ in 0..metadata_count {
            let key = read_string(file, version, file_size)?;
            let value_type = read_u32(file)?;
            let value = read_value(file, version, file_size, value_type, 0)?;
            anyhow::ensure!(
                metadata.insert(key, value).is_none(),
                "duplicate GGUF metadata key"
            );
        }

        let mut tensor_infos = HashMap::new();
        tensor_infos
            .try_reserve(tensor_capacity)
            .map_err(|error| anyhow::anyhow!("GGUF tensor table allocation failed: {error}"))?;
        for _ in 0..tensor_count {
            let name = read_string(file, version, file_size)?;
            let dimensions = read_u32(file)?;
            anyhow::ensure!(
                dimensions <= MAX_TENSOR_DIMS,
                "GGUF tensor {name} has too many dimensions: {dimensions}"
            );
            let mut shape = Vec::new();
            shape
                .try_reserve_exact(dimensions as usize)
                .map_err(|error| anyhow::anyhow!("GGUF tensor shape allocation failed: {error}"))?;
            for _ in 0..dimensions {
                let dim = match version {
                    Version::V1 => read_u32(file)? as u64,
                    Version::V2 | Version::V3 => read_u64(file)?,
                };
                shape.push(usize::try_from(dim).context("GGUF tensor dimension is too large")?);
            }
            shape.reverse();
            let dtype = DeepSeek4DType::from_u32(read_u32(file)?);
            let offset = read_u64(file)?;
            anyhow::ensure!(
                tensor_infos
                    .insert(
                        name,
                        RawTensorInfo {
                            shape,
                            dtype,
                            offset
                        },
                    )
                    .is_none(),
                "duplicate GGUF tensor name"
            );
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(value_as_u64)
            .unwrap_or(DEFAULT_ALIGNMENT);
        anyhow::ensure!(alignment > 0, "GGUF alignment must be positive");
        let header_end = file.stream_position()?;
        let tensor_data_offset = header_end
            .checked_add(alignment - 1)
            .and_then(|value| value.checked_div(alignment))
            .and_then(|value| value.checked_mul(alignment))
            .context("GGUF tensor data offset overflow")?;
        anyhow::ensure!(
            tensor_data_offset <= file_size,
            "GGUF tensor data offset is outside the file"
        );

        Ok(Self {
            metadata,
            tensor_infos,
            tensor_data_offset,
            file_size,
            header_stats,
        })
    }

    /// Header bytes plus parsed metadata memory, for split-model budgets.
    pub(crate) fn parsed_header_bytes(&self) -> u64 {
        self.header_stats.parsed_bytes(self.tensor_data_offset)
    }

    pub fn load_tensor(&self, file: &mut File, name: &str, device: &Device) -> Result<RawTensor> {
        anyhow::ensure!(
            !name.contains("engram_embd") && name != "per_layer_token_embd.weight",
            "Engram tables are disk-only; whole-table allocation is forbidden"
        );
        anyhow::ensure!(
            !(matches!(self.metadata.get("general.architecture"), Some(Value::String(a)) if a == "deepseek41")
                && name.ends_with("_exps.weight")),
            "V4.1 expert banks are disk-only; load a single expert slice"
        );
        let (info, raw) = self.read_tensor_bytes(file, name)?;
        let byte_len = raw.len();
        let bytes = Tensor::from_vec(raw, byte_len, device)?;
        RawTensor::new(info.shape.clone(), info.dtype, bytes)
    }

    /// Read one tensor's bytes, in their GGUF layout, into host memory.
    pub(crate) fn read_tensor_bytes(
        &self,
        file: &mut File,
        name: &str,
    ) -> Result<(&RawTensorInfo, Vec<u8>)> {
        let info = self
            .tensor_infos
            .get(name)
            .with_context(|| format!("GGUF tensor is missing: {name}"))?;
        let byte_len = info.byte_len()?;
        let start = self
            .tensor_data_offset
            .checked_add(info.offset)
            .context("GGUF tensor offset overflow")?;
        let end = start
            .checked_add(byte_len as u64)
            .context("GGUF tensor end offset overflow")?;
        anyhow::ensure!(
            end <= self.file_size,
            "GGUF tensor {name} extends past the end of the file"
        );
        let mut raw = vec![0u8; byte_len];
        file.seek(SeekFrom::Start(start))?;
        file.read_exact(&mut raw)?;
        Ok((info, raw))
    }
}

pub fn read_architecture(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .with_context(|| format!("Failed to open GGUF file: {}", path.display()))?;
    let content = RawGgufContent::read(&mut file)?;
    match content.metadata.get("general.architecture") {
        Some(Value::String(value)) => Ok(value.clone()),
        _ => anyhow::bail!("missing or invalid GGUF metadata key: general.architecture"),
    }
}

pub fn is_deepseek4(path: &Path) -> Result<bool> {
    Ok(matches!(
        read_architecture(path)?.as_str(),
        "deepseek4" | "deepseek41"
    ))
}

fn read_value(
    file: &mut File,
    version: Version,
    file_size: u64,
    value_type: u32,
    depth: usize,
) -> Result<Value> {
    anyhow::ensure!(
        depth <= MAX_VALUE_DEPTH,
        "GGUF metadata nesting is too deep"
    );
    Ok(match value_type {
        0 => Value::U8(read_u8(file)?),
        1 => Value::I8(read_u8(file)? as i8),
        2 => Value::U16(read_u16(file)?),
        3 => Value::I16(read_u16(file)? as i16),
        4 => Value::U32(read_u32(file)?),
        5 => Value::I32(read_u32(file)? as i32),
        6 => Value::F32(f32::from_bits(read_u32(file)?)),
        7 => {
            let value = read_u8(file)?;
            anyhow::ensure!(value <= 1, "invalid GGUF boolean value {value}");
            Value::Bool(value != 0)
        }
        8 => Value::String(read_string(file, version, file_size)?),
        9 => {
            let element_type = read_u32(file)?;
            let count = version.read_len(file)?;
            anyhow::ensure!(
                count <= MAX_GGUF_ARRAY_ELEMENTS,
                "GGUF array element count ({count}) exceeds application limit ({MAX_GGUF_ARRAY_ELEMENTS})"
            );
            let min_element_bytes = min_value_disk_bytes(version, element_type)
                .with_context(|| format!("unknown GGUF array element type {element_type}"))?;
            let minimum_bytes = count
                .checked_mul(min_element_bytes)
                .context("GGUF array minimum size overflow")?;
            let current = file.stream_position()?;
            anyhow::ensure!(
                minimum_bytes <= file_size.saturating_sub(current),
                "GGUF array declares more elements than the remaining file can contain"
            );
            let count = usize::try_from(count).context("GGUF array is too large for usize")?;
            let mut values = Vec::new();
            values
                .try_reserve_exact(count)
                .map_err(|error| anyhow::anyhow!("GGUF array allocation failed: {error}"))?;
            for _ in 0..count {
                values.push(read_value(
                    file,
                    version,
                    file_size,
                    element_type,
                    depth + 1,
                )?);
            }
            Value::Array(values)
        }
        10 => Value::U64(read_u64(file)?),
        11 => Value::I64(read_u64(file)? as i64),
        12 => Value::F64(f64::from_bits(read_u64(file)?)),
        value_type => anyhow::bail!("unknown GGUF metadata value type {value_type}"),
    })
}

fn min_value_disk_bytes(version: Version, value_type: u32) -> Option<u64> {
    Some(match value_type {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        8 => match version {
            Version::V1 => 4,
            Version::V2 | Version::V3 => 8,
        },
        9 => match version {
            Version::V1 => 8,
            Version::V2 | Version::V3 => 12,
        },
        _ => return None,
    })
}

fn read_string(file: &mut File, version: Version, file_size: u64) -> Result<String> {
    let len = version.read_len(file)?;
    anyhow::ensure!(len <= MAX_STRING_LENGTH, "GGUF string is too long");
    let current = file.stream_position()?;
    anyhow::ensure!(
        len <= file_size.saturating_sub(current),
        "GGUF string extends past the end of the file"
    );
    let mut bytes = vec![0u8; len as usize];
    file.read_exact(&mut bytes)?;
    while bytes.last() == Some(&0) {
        bytes.pop();
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn value_as_u64(value: &Value) -> Option<u64> {
    match value {
        Value::U8(value) => Some(*value as u64),
        Value::U16(value) => Some(*value as u64),
        Value::U32(value) => Some(*value as u64),
        Value::U64(value) => Some(*value),
        Value::I8(value) if *value >= 0 => Some(*value as u64),
        Value::I16(value) if *value >= 0 => Some(*value as u64),
        Value::I32(value) if *value >= 0 => Some(*value as u64),
        Value::I64(value) if *value >= 0 => Some(*value as u64),
        _ => None,
    }
}

fn read_u8(file: &mut File) -> Result<u8> {
    let mut bytes = [0u8; 1];
    file.read_exact(&mut bytes)?;
    Ok(bytes[0])
}

fn read_u16(file: &mut File) -> Result<u16> {
    let mut bytes = [0u8; 2];
    file.read_exact(&mut bytes)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32(file: &mut File) -> Result<u32> {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(file: &mut File) -> Result<u64> {
    let mut bytes = [0u8; 8];
    file.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn v3_fixture(metadata_keys: &[&[u8]], tensor_names: &[&[u8]]) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&(tensor_names.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(metadata_keys.len() as u64).to_le_bytes());
        for key in metadata_keys {
            bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
            bytes.extend_from_slice(key);
            bytes.extend_from_slice(&0u32.to_le_bytes()); // U8
            bytes.push(1);
        }
        for name in tensor_names {
            bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
            bytes.extend_from_slice(name);
            bytes.extend_from_slice(&1u32.to_le_bytes()); // one dimension
            bytes.extend_from_slice(&1u64.to_le_bytes());
            bytes.extend_from_slice(&0u32.to_le_bytes()); // F32
            bytes.extend_from_slice(&0u64.to_le_bytes());
        }
        let aligned_header = (bytes.len() + 31) & !31;
        bytes.resize(aligned_header + 4, 0); // one F32 tensor at offset zero
        file.write_all(&bytes).unwrap();
        file
    }

    #[test]
    fn header_io_policy_restores_after_entry_read_and_restore_errors() -> Result<()> {
        use HeaderIoPolicy::{NoCache, ReadAhead};
        use std::cell::RefCell;

        // Each bit fails one policy call; 0b1100 fails BOTH restore calls.
        for failures in [0u8, 0b0001, 0b0010, 0b0100, 0b1000, 0b1100] {
            for read_error in [false, true] {
                let events = RefCell::new(Vec::new());
                let mut state = (true, false);
                let mut calls = 0;
                let keys: &[&[u8]] = if read_error {
                    &[b"key", b"key\0"]
                } else {
                    &[b"key"]
                };
                let mut fixture = v3_fixture(keys, &[b"tensor"]);
                let result = with_header_io_policy(
                    |policy| {
                        events.borrow_mut().push(Some(policy));
                        let fail = failures & (1 << calls) != 0;
                        calls += 1;
                        anyhow::ensure!(!fail, "injected policy failure");
                        match policy {
                            NoCache(value) => state.0 = value,
                            ReadAhead(value) => state.1 = value,
                        }
                        Ok(())
                    },
                    || {
                        events.borrow_mut().push(None); // entire preflight + parse
                        RawGgufContent::read(fixture.as_file_mut())
                    },
                );
                let mut expected = vec![Some(NoCache(false))];
                if failures & 1 == 0 {
                    expected.push(Some(ReadAhead(true)));
                    if failures & 2 == 0 {
                        expected.push(None);
                    }
                }
                expected.extend([Some(NoCache(true)), Some(ReadAhead(false))]);
                assert_eq!(*events.borrow(), expected);
                assert_eq!(result.is_ok(), failures == 0 && !read_error);
                if failures == 0 {
                    assert_eq!(state, (true, false));
                    if read_error {
                        assert!(
                            result
                                .err()
                                .unwrap()
                                .to_string()
                                .contains("duplicate GGUF metadata key")
                        );
                    } else {
                        let content = result?;
                        assert!(matches!(content.metadata.get("key"), Some(Value::U8(1))));
                        assert_eq!(content.tensor_infos["tensor"].shape, [1]);
                        assert_eq!(content.tensor_infos["tensor"].dtype, DeepSeek4DType::F32);
                    }
                } else if failures & 0b1100 != 0 {
                    assert!(result.err().unwrap().to_string().contains("restoring GGUF"));
                }
            }
        }
        // Exercise the actual owned-FD constructor on the same tiny header.
        let fixture = v3_fixture(&[b"key"], &[b"tensor"]);
        let source = RawGgufSource::new(File::open(fixture.path())?)?;
        assert!(matches!(
            source.content().metadata.get("key"),
            Some(Value::U8(1))
        ));
        assert_eq!(source.content().tensor_infos["tensor"].shape, [1]);
        assert_eq!(
            source.content().tensor_infos["tensor"].dtype,
            DeepSeek4DType::F32
        );
        Ok(())
    }

    #[test]
    fn v4_quant_layouts_match_ggml_block_sizes() {
        assert_eq!(DeepSeek4DType::IQ2XS.block_layout(), Some((256, 74)));
        assert_eq!(DeepSeek4DType::IQ3XXS.block_layout(), Some((256, 98)));
        assert_eq!(DeepSeek4DType::MXFP4.block_layout(), Some((32, 17)));
        assert_eq!(DeepSeek4DType::IQ3XXS.byte_len(256 * 6).unwrap(), 98 * 6);
    }

    #[test]
    fn unsupported_types_are_kept_in_the_header() {
        let dtype = DeepSeek4DType::from_u32(1234);
        assert_eq!(dtype, DeepSeek4DType::Unknown(1234));
        assert!(dtype.block_layout().is_none());
    }

    #[test]
    fn ggml_type_ids_match_v4_integer_and_iq3s_tensors() {
        assert_eq!(DeepSeek4DType::from_u32(21), DeepSeek4DType::IQ3S);
        assert_eq!(DeepSeek4DType::from_u32(26), DeepSeek4DType::I32);
        assert_eq!(DeepSeek4DType::from_u32(29), DeepSeek4DType::IQ1M);
        assert_eq!(DeepSeek4DType::I32.ggml_id(), 26);
    }

    #[test]
    fn rejects_huge_raw_header_counts_before_collection_allocation() {
        let path = std::env::temp_dir().join(format!(
            "dinf-deepseek-gguf-count-limit-{}-{}.gguf",
            std::process::id(),
            std::thread::current()
                .name()
                .unwrap_or("test")
                .replace(':', "_")
        ));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&(MAX_GGUF_TENSOR_COUNT + 1).to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();

        let mut file = File::open(&path).unwrap();
        let error = match RawGgufContent::read(&mut file) {
            Ok(_) => panic!("oversized header was accepted"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("tensor count"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_aggregate_raw_header_budget_before_collection_allocation() {
        let path = std::env::temp_dir().join(format!(
            "dinf-deepseek-gguf-header-budget-{}-{}.gguf",
            std::process::id(),
            std::thread::current()
                .name()
                .unwrap_or("test")
                .replace(':', "_")
        ));
        let mut file = File::create(&path).unwrap();
        file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();
        file.write_all(&1u64.to_le_bytes()).unwrap();
        file.write_all(&1u64.to_le_bytes()).unwrap();
        file.write_all(b"x").unwrap();
        file.write_all(&8u32.to_le_bytes()).unwrap(); // string metadata value
        file.write_all(&MAX_GGUF_HEADER_BYTES.to_le_bytes())
            .unwrap();
        file.set_len(MAX_GGUF_HEADER_BYTES + 4096).unwrap();
        drop(file);

        let mut file = File::open(&path).unwrap();
        let error = match RawGgufContent::read(&mut file) {
            Ok(_) => panic!("oversized header was accepted"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("application limit"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn raw_reader_rejects_duplicate_metadata_keys_before_hashmap_insert() {
        let mut file = v3_fixture(&[b"general.architecture", b"general.architecture\0"], &[]);
        let error = match RawGgufContent::read(file.as_file_mut()) {
            Ok(_) => panic!("duplicate metadata key was accepted"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("duplicate GGUF metadata key"), "{error}");
        let error = match RawGgufSource::new(File::open(file.path()).unwrap()) {
            Ok(_) => panic!("retained source accepted duplicate metadata"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("duplicate GGUF metadata key"), "{error}");
    }

    #[test]
    fn raw_reader_rejects_duplicate_tensor_names_before_hashmap_insert() {
        let mut file = v3_fixture(&[], &[b"blk.0.attn", b"blk.0.attn"]);
        let error = match RawGgufContent::read(file.as_file_mut()) {
            Ok(_) => panic!("duplicate tensor name was accepted"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("duplicate GGUF tensor name"), "{error}");
    }

    #[test]
    fn quantized_byte_length_rejects_checked_arithmetic_overflow() {
        assert!(DeepSeek4DType::F32.byte_len(usize::MAX).is_err());
        assert!(DeepSeek4DType::Q2K.byte_len(usize::MAX).is_err());
    }
    #[test]
    fn lazy_expert_reads_preserve_packed_slices_and_source_lifetime() -> Result<()> {
        let md = fixtures::maya_metadata(true);
        let name = "blk.0.ffn_gate_exps.weight";
        let (mut fixture, data) = fixtures::sparse(&md, &[(name, &[2, 256, 256], 19, 0)]);
        let size = 256 * 256 / 256 * 50;
        fixture.seek(SeekFrom::Start(data))?;
        fixture.write_all(&vec![17; size])?;
        fixture.write_all(&vec![93; size])?;
        let file = File::open(fixture.path())?;
        // Capture identity after unlink: unlink itself can change Unix ctime.
        drop(fixture);
        let source = Arc::new(RawGgufSource::new(file)?);
        let retained = Arc::clone(&source);
        drop(source);
        let first = retained.read_expert_slice(name, 0, &Device::Cpu)?;
        let second = retained.read_expert_slice(name, 1, &Device::Cpu)?;
        assert_eq!(first.shape(), &[256, 256]);
        assert_eq!(first.dtype(), DeepSeek4DType::IQ1S);
        assert_eq!(first.to_bytes()?, vec![17; size]);
        assert_eq!(second.to_bytes()?, vec![93; size]);
        assert!(retained.read_expert_slice(name, 2, &Device::Cpu).is_err());
        assert!(
            retained
                .read_expert_slice(name, usize::MAX, &Device::Cpu)
                .is_err()
        );
        assert!(retained.read_tensor_bytes(name, u64::MAX, 1).is_err());
        assert!(
            retained
                .read_tensor_bytes(name, 2 * size as u64, 1)
                .is_err()
        );
        assert!(
            retained
                .read_tensor_bytes(name, 0, MAX_SOURCE_READ_BYTES + 1)
                .is_err()
        );
        assert!(
            retained
                .read_tensor_bytes("../metadata-path", 0, 1)
                .is_err()
        );
        assert!(retained.load_resident_tensor(name, &Device::Cpu).is_err());
        Ok(())
    }

    #[test]
    fn disk_engram_reads_are_bounded_and_whole_allocation_is_refused() -> Result<()> {
        let name = "blk.1.engram_embd.weight";
        let md = fixtures::maya_metadata(true);
        let (mut fixture, _) = fixtures::sparse(&md, &[(name, &[384006168, 264], 24, 0)]);
        let source = RawGgufSource::new(File::open(fixture.path())?)?;
        assert_eq!(
            source.content().tensor_infos[name].dtype,
            DeepSeek4DType::I8
        );
        assert_eq!(
            source.read_tensor_bytes(name, (384006168 - 1) * 264, 264)?,
            vec![0; 264]
        );
        let error = source
            .content()
            .load_tensor(fixture.as_file_mut(), name, &Device::Cpu)
            .err()
            .unwrap();
        assert!(
            error
                .to_string()
                .contains("whole-table allocation is forbidden")
        );
        assert!(source.load_resident_tensor(name, &Device::Cpu).is_err());
        fixture.as_file().set_len(4096)?;
        assert!(source.read_tensor_bytes(name, 0, 264).is_err());
        Ok(())
    }

    #[test]
    fn raw_whole_expert_bank_load_is_refused_for_v41_only() -> Result<()> {
        let name = "blk.0.ffn_gate_exps.weight";
        let mut md = fixtures::maya_metadata(true);
        let (mut fixture, _) = fixtures::sparse(&md, &[(name, &[2, 256, 256], 19, 0)]);
        let content = RawGgufContent::read(fixture.as_file_mut())?;
        assert!(
            content
                .load_tensor(fixture.as_file_mut(), name, &Device::Cpu)
                .err()
                .unwrap()
                .to_string()
                .contains("single expert slice")
        );
        md.insert(
            "general.architecture".into(),
            Value::String("deepseek4".into()),
        );
        let (mut old, _) = fixtures::sparse(&md, &[(name, &[2, 256, 256], 19, 0)]);
        let content = RawGgufContent::read(old.as_file_mut())?;
        assert_eq!(
            content
                .load_tensor(old.as_file_mut(), name, &Device::Cpu)?
                .shape(),
            &[2, 256, 256]
        );
        Ok(())
    }

    const ABC_SHA256: [u8; 32] = [
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22,
        0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00,
        0x15, 0xad,
    ];

    fn overwrite_same_size(file: &mut File) -> std::io::Result<()> {
        file.seek(SeekFrom::End(-1))?;
        file.write_all(&[91])?;
        // Deterministic even on filesystems with coarse timestamp resolution.
        file.set_modified(std::time::UNIX_EPOCH)
    }

    #[test]
    fn artifact_sha_rejects_wrong_length_digest_and_mutation_before_or_during_hash() -> Result<()> {
        let mut fixture = NamedTempFile::new()?;
        fixture.write_all(b"abc")?;
        let file = File::open(fixture.path())?;
        let identity = FileIdentity::from_metadata(&file.metadata()?)?;
        verify_artifact(&file, identity, 3, &ABC_SHA256, read_exact_at)?;
        assert!(verify_artifact(&file, identity, 4, &ABC_SHA256, read_exact_at).is_err());
        assert!(verify_artifact(&file, identity, 3, &[0; 32], read_exact_at).is_err());
        // Mutation after reading the original bytes must fail even with a valid
        // digest of those bytes and an unchanged descriptor length.
        let during = verify_artifact(&file, identity, 3, &ABC_SHA256, |file, bytes, offset| {
            read_exact_at(file, bytes, offset)?;
            overwrite_same_size(fixture.as_file_mut())
        });
        assert!(during.is_err());
        assert_eq!(file.metadata()?.len(), identity.length);
        let mut read_called = false;
        let before = verify_artifact(&file, identity, 3, &ABC_SHA256, |_, _, _| {
            read_called = true;
            Ok(())
        });
        assert!(before.is_err());
        assert!(!read_called);
        Ok(())
    }

    #[test]
    fn artifact_sha_streams_all_bytes_with_bounded_scratch() -> Result<()> {
        let bytes = vec![37; SHA_SCRATCH_BYTES * 2 + 19];
        let expected: [u8; 32] = Sha256::digest(&bytes).into();
        let mut fixture = NamedTempFile::new()?;
        fixture.write_all(&bytes)?;
        let file = File::open(fixture.path())?;
        let identity = FileIdentity::from_metadata(&file.metadata()?)?;
        let mut offsets = Vec::new();
        verify_artifact(
            &file,
            identity,
            bytes.len() as u64,
            &expected,
            |file, bytes, offset| {
                assert!(bytes.len() <= SHA_SCRATCH_BYTES);
                offsets.push((offset, bytes.len()));
                read_exact_at(file, bytes, offset)
            },
        )?;
        assert_eq!(
            offsets,
            vec![
                (0, SHA_SCRATCH_BYTES),
                (SHA_SCRATCH_BYTES as u64, SHA_SCRATCH_BYTES),
                ((SHA_SCRATCH_BYTES * 2) as u64, 19),
            ]
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn artifact_sha_uses_open_descriptor_after_path_substitution_and_unlink() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("artifact");
        let moved = dir.path().join("original");
        std::fs::write(&path, b"abc")?;
        let file = File::open(&path)?;
        std::fs::rename(&path, &moved)?;
        std::fs::write(&path, b"xyz")?;
        std::fs::remove_file(&moved)?;
        let identity = FileIdentity::from_metadata(&file.metadata()?)?;
        verify_artifact(&file, identity, 3, &ABC_SHA256, read_exact_at)?;
        let replacement = File::open(&path)?;
        let replacement_identity = FileIdentity::from_metadata(&replacement.metadata()?)?;
        assert_ne!(identity.inode, replacement_identity.inode);
        assert!(
            verify_artifact(
                &replacement,
                replacement_identity,
                3,
                &ABC_SHA256,
                read_exact_at,
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn unverified_fixture_cannot_claim_maya_proof_or_satisfy_production_pin() -> Result<()> {
        let fixture = v3_fixture(&[], &[b"output_norm.weight"]);
        let source = RawGgufSource::new(File::open(fixture.path())?)?;
        assert!(source.require_verified_maya().is_err());
        assert!(RawGgufSource::verified_maya(File::open(fixture.path())?).is_err());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn source_and_hash_identity_reject_non_regular_files() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let file = File::open(dir.path())?;
        assert!(FileIdentity::from_metadata(&file.metadata()?).is_err());
        assert!(RawGgufSource::new(file.try_clone()?).is_err());
        assert!(RawGgufSource::verified_maya(file).is_err());
        Ok(())
    }

    #[test]
    fn host_diag_selected_source_error_and_post_identity_bytes() -> Result<()> {
        if crate::test_in_subprocess(concat!(
            module_path!(),
            "::host_diag_selected_source_error_and_post_identity_bytes"
        )) {
            return Ok(());
        }
        use crate::architectures::deepseek41::profile::{self, Detail};
        let name = "blk.0.ffn_gate_exps.weight";
        let (mut fixture, _) = fixtures::sparse(&HashMap::new(), &[(name, &[2, 256, 256], 19, 0)]);
        let source = RawGgufSource::new(File::open(fixture.path())?)?;
        let mut bytes = vec![0; 256 * 50];
        let _guard = profile::HostGuard::enable();
        let failed = source.read_tensor_into_selected(
            name,
            0,
            &mut bytes,
            |_, _, _| {
                Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "CPU payload failure",
                ))
            },
            true,
        );
        assert!(failed.is_err());
        let snapshot = profile::host_snapshot();
        let read = snapshot.details[Detail::ExpertRead as usize];
        assert_eq!(read.calls, 1);
        assert_eq!(read.requested_bytes, bytes.len() as u64);
        assert_eq!(read.completed_bytes, 0);
        assert_eq!(
            snapshot.details[Detail::ExpertMetadataPost as usize].calls,
            0
        );
        profile::reset();
        let changed = source.read_tensor_into_selected(
            name,
            0,
            &mut bytes,
            |file, bytes, offset| {
                read_exact_at(file, bytes, offset)?;
                overwrite_same_size(fixture.as_file_mut())
            },
            true,
        );
        assert!(changed.unwrap_err().to_string().contains("identity"));
        let snapshot = profile::host_snapshot();
        let read = snapshot.details[Detail::ExpertRead as usize];
        assert_eq!(read.completed_bytes, bytes.len() as u64);
        assert_eq!(
            snapshot.details[Detail::ExpertMetadataPost as usize].calls,
            1
        );
        profile::reset();
        assert!(source.read_expert_slice(name, 0, &Device::Cpu).is_err());
        assert_eq!(
            profile::host_snapshot().details[Detail::ExpertRead as usize].calls,
            0
        );
        Ok(())
    }

    #[test]
    fn lazy_source_rejects_same_size_mutation_before_and_after_even_empty_reads() -> Result<()> {
        for empty in [false, true] {
            let mut fixture = v3_fixture(&[], &[b"output_norm.weight"]);
            let source = RawGgufSource::new(File::open(fixture.path())?)?;
            let mut bytes = vec![0; if empty { 0 } else { 4 }];
            let during = source.read_tensor_into_with(
                "output_norm.weight",
                0,
                &mut bytes,
                |file, bytes, offset| {
                    read_exact_at(file, bytes, offset)?;
                    overwrite_same_size(fixture.as_file_mut())
                },
            );
            assert!(during.is_err(), "mutation during read escaped");
            assert_eq!(source.file.metadata()?.len(), source.identity.length);
            let mut read_called = false;
            let before =
                source.read_tensor_into_with("output_norm.weight", 0, &mut bytes, |_, _, _| {
                    read_called = true;
                    Ok(())
                });
            assert!(before.is_err());
            assert!(!read_called);
            assert!(source.clone_file().is_err());
            assert!(
                source
                    .load_resident_tensor("output_norm.weight", &Device::Cpu)
                    .is_err()
            );
        }
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn lazy_source_rejects_ctime_change_after_unlink_even_when_mtime_is_unchanged() -> Result<()> {
        let fixture = v3_fixture(&[], &[b"output_norm.weight"]);
        let source = RawGgufSource::new(File::open(fixture.path())?)?;
        drop(fixture);
        let after = FileIdentity::from_metadata(&source.file.metadata()?)?;
        assert_eq!(after.length, source.identity.length);
        assert_eq!(after.modified, source.identity.modified);
        assert!(
            source
                .read_tensor_into("output_norm.weight", 0, &mut [])
                .is_err()
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn maya_cached_verification_cpu() -> Result<()> {
        use crate::verified_file::{fingerprint_descriptor, verify_cached};
        use std::os::unix::fs::{FileExt, PermissionsExt, symlink};
        // Total fixture payload below 64MiB, no model files or devices.
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("fixture.gguf");
        let cache = dir.path().join("fixture.gguf.dinf-verified.json");
        let bytes = vec![37u8; 17 * 1024 * 1024 + 19];
        std::fs::write(&path, &bytes)?;
        let mut pin = maya_pin();
        pin.model_id = "cfg-test-only".into();
        pin.expected_bytes = bytes.len() as u64;
        pin.expected_sha256 = Sha256::digest(&bytes).into();
        let verify = |pin: &Pin| -> Result<(Verification, Vec<(u64, usize)>)> {
            let file = File::open(&path)?;
            let identity = FileIdentity::from_metadata(&file.metadata()?)?;
            let mut reads = Vec::new();
            let stats = verify_cached(&file, &path, identity, pin, || {
                verify_artifact(
                    &file,
                    identity,
                    pin.expected_bytes,
                    &pin.expected_sha256,
                    |file, bytes, offset| {
                        reads.push((offset, bytes.len()));
                        read_exact_at(file, bytes, offset)
                    },
                )
            })?;
            Ok((stats, reads))
        };
        let before = fixtures::sha_passes();
        let (cold, reads) = verify(&pin)?;
        assert_eq!(cold.full_passes, 1);
        assert!(!cold.sample_hit);
        assert_eq!(cold.full_read_bytes, pin.expected_bytes);
        assert_eq!(
            reads.iter().map(|(_, n)| *n as u64).sum::<u64>(),
            pin.expected_bytes
        );
        let mut end = 0;
        for (offset, n) in reads {
            assert_eq!(offset, end);
            assert!(n <= SHA_SCRATCH_BYTES);
            end += n as u64;
        }
        assert_eq!(
            std::fs::metadata(&cache)?.permissions().mode() & 0o777,
            0o600
        );
        let valid: serde_json::Value = serde_json::from_slice(&std::fs::read(&cache)?)?;
        assert_eq!(valid["version"], 1);
        for _ in 0..2 {
            let (warm, reads) = verify(&pin)?;
            assert!(warm.sample_hit);
            assert_eq!(
                (warm.full_passes, warm.full_read_bytes, warm.sample_bytes),
                (0, 0, 16 * 1024 * 1024)
            );
            assert!(reads.is_empty());
        }
        assert_eq!(fixtures::sha_passes(), before + 1);
        let identity = FileIdentity::from_metadata(&File::open(&path)?.metadata()?)?;
        let mut wrong_identity = identity;
        wrong_identity.inode += 1;
        assert!(
            crate::verified_file::with_open_guard(wrong_identity, true, || verify(&pin)).is_err()
        );
        assert!(
            crate::verified_file::with_open_guard(identity, false, || verify(&pin))?
                .0
                .sample_hit
        );
        std::fs::remove_file(&cache)?;
        let passes = fixtures::sha_passes();
        assert!(
            crate::verified_file::with_open_guard(identity, false, || verify(&pin))
                .unwrap_err()
                .to_string()
                .contains("cached-only")
        );
        assert_eq!(fixtures::sha_passes(), passes);
        verify(&pin)?;

        // ctime invalidates a middle mutation even with the original mtime and
        // unchanged head/tail digest. A fullcheck then rejects corrupted bytes.
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)?;
        let old_metadata = file.metadata()?;
        let old_identity = FileIdentity::from_metadata(&old_metadata)?;
        let sample = fingerprint_descriptor(&file)?;
        file.write_at(&[91], 8 * 1024 * 1024 + 3)?;
        file.set_modified(old_metadata.modified()?)?;
        let changed = FileIdentity::from_metadata(&file.metadata()?)?;
        assert_eq!(old_identity.modified, changed.modified);
        assert_ne!(old_identity.changed, changed.changed);
        assert_eq!(sample, fingerprint_descriptor(&file)?);
        let passes = fixtures::sha_passes();
        assert!(verify(&pin).unwrap_err().to_string().contains("SHA256"));
        assert_eq!(fixtures::sha_passes(), passes + 1);
        file.write_at(&[37], 8 * 1024 * 1024 + 3)?;
        verify(&pin)?;
        for offset in [0, pin.expected_bytes - 1] {
            file.write_at(&[91], offset)?;
            assert!(verify(&pin).is_err());
            file.write_at(&[37], offset)?;
            verify(&pin)?;
        }
        for length in [pin.expected_bytes - 1, pin.expected_bytes + 1] {
            file.set_len_sparse(length)?;
            let passes = fixtures::sha_passes();
            assert!(
                verify(&pin)
                    .unwrap_err()
                    .to_string()
                    .contains("length mismatch")
            );
            assert_eq!(fixtures::sha_passes(), passes);
            file.set_len_sparse(pin.expected_bytes)?;
            file.write_at(&[37], pin.expected_bytes - 1)?;
        }
        verify(&pin)?;
        // Same bytes at a replacement inode cannot inherit the receipt.
        let moved = dir.path().join("moved");
        std::fs::rename(&path, &moved)?;
        std::fs::write(&path, &bytes)?;
        assert!(!verify(&pin)?.0.sample_hit);
        assert_ne!(
            FileIdentity::from_metadata(&file.metadata()?)?.inode,
            FileIdentity::from_metadata(&File::open(&path)?.metadata()?)?.inode
        );
        assert!(crate::verified_file::check_identity(&file, old_identity).is_err());

        for field in [
            "model_id",
            "source_revision",
            "native_version",
            "expected_sha256",
        ] {
            let mut changed = pin.clone();
            match field {
                "model_id" => changed.model_id.push('x'),
                "source_revision" => changed.source_revision.push('x'),
                "native_version" => changed.native_version += 1,
                _ => changed.expected_sha256 = [0; 32],
            }
            let passes = fixtures::sha_passes();
            let result = verify(&changed);
            assert_eq!(fixtures::sha_passes(), passes + 1);
            if field == "expected_sha256" {
                assert!(result.is_err());
            } else {
                assert!(!result?.0.sample_hit);
            }
            verify(&pin)?;
        }
        // All cache-only failures are misses, never proof and never load errors.
        let valid = std::fs::read(&cache)?;
        let mut unknown_version: serde_json::Value = serde_json::from_slice(&valid)?;
        unknown_version["version"] = 2.into();
        let mut bad_sample: serde_json::Value = serde_json::from_slice(&valid)?;
        bad_sample["sample_sha256"][0] =
            (bad_sample["sample_sha256"][0].as_u64().unwrap() ^ 1).into();
        for bad in [
            b"{\"sha256\":\"legacy\"}".to_vec(),
            b"{broken".to_vec(),
            vec![b' '; 16 * 1024 + 1],
            serde_json::to_vec(&unknown_version)?,
            serde_json::to_vec(&bad_sample)?,
        ] {
            std::fs::write(&cache, bad)?;
            assert!(!verify(&pin)?.0.sample_hit);
        }
        std::fs::set_permissions(&cache, std::fs::Permissions::from_mode(0o0))?;
        assert!(!verify(&pin)?.0.sample_hit);
        std::fs::set_permissions(&cache, std::fs::Permissions::from_mode(0o666))?;
        assert!(!verify(&pin)?.0.sample_hit);
        let victim = dir.path().join("victim");
        std::fs::write(&victim, b"do not follow")?;
        std::fs::remove_file(&cache)?;
        symlink(&victim, &cache)?;
        assert!(!verify(&pin)?.0.sample_hit);
        assert_eq!(std::fs::read(&victim)?, b"do not follow");
        let linked = dir.path().join("linked");
        std::fs::hard_link(&cache, &linked)?;
        assert!(!verify(&pin)?.0.sample_hit);
        std::fs::remove_file(&cache)?;
        std::fs::create_dir(&cache)?;
        // rename onto a directory fails: successful fullcheck still returns.
        assert!(!verify(&pin)?.0.sample_hit);
        assert!(cache.is_dir());
        std::fs::remove_dir(&cache)?;
        // FIFO must not block cache lookup.
        use std::os::unix::ffi::OsStrExt;
        let fifo = std::ffi::CString::new(cache.as_os_str().as_bytes())?;
        assert_eq!(unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) }, 0);
        assert!(!verify(&pin)?.0.sample_hit);
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o777))?;
        assert!(!verify(&pin)?.0.sample_hit);
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700))?;
        let error_path = dir.path().join("read-error.gguf");
        std::fs::write(&error_path, b"abc")?;
        let error_file = File::open(&error_path)?;
        let identity = FileIdentity::from_metadata(&error_file.metadata()?)?;
        let mut error_pin = pin.clone();
        error_pin.expected_bytes = 3;
        error_pin.expected_sha256 = ABC_SHA256;
        let failed = verify_cached(&error_file, &error_path, identity, &error_pin, || {
            verify_artifact(&error_file, identity, 3, &ABC_SHA256, |_, _, _| {
                Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "fixture read failure",
                ))
            })
        });
        assert!(failed.is_err());
        assert!(
            !dir.path()
                .join("read-error.gguf.dinf-verified.json")
                .exists()
        );
        // Preserve existing fullcheck/read error/mutation and retained-FD gates.
        artifact_sha_rejects_wrong_length_digest_and_mutation_before_or_during_hash()?;
        artifact_sha_streams_all_bytes_with_bounded_scratch()?;
        artifact_sha_uses_open_descriptor_after_path_substitution_and_unlink()?;
        unverified_fixture_cannot_claim_maya_proof_or_satisfy_production_pin()?;
        source_and_hash_identity_reject_non_regular_files()?;
        lazy_source_rejects_same_size_mutation_before_and_after_even_empty_reads()?;
        lazy_source_rejects_ctime_change_after_unlink_even_when_mtime_is_unchanged()?;
        eprintln!(
            "cached CPU fixture: cold={} full bytes; warm=0 full bytes/{} sample bytes",
            cold.full_read_bytes,
            16 * 1024 * 1024
        );
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;
    use std::io::Write;

    pub(crate) fn sha_passes() -> usize {
        ARTIFACT_SHA_PASSES.with(std::cell::Cell::get)
    }

    /// Test-only injected pins. No production constructor accepts fixture pins
    /// or promotes an existing unverified source.
    pub(crate) fn verified_source(
        mut file: File,
        bytes: u64,
        sha: &[u8; 32],
    ) -> Result<RawGgufSource> {
        anyhow::ensure!(bytes <= SHA_SCRATCH_BYTES as u64, "fixture must stay small");
        let identity = FileIdentity::from_metadata(&file.metadata()?)?;
        verify_artifact(&file, identity, bytes, sha, read_exact_at)?;
        let content = Arc::new(RawGgufContent::read(&mut file)?);
        check_source_identity(&file, identity)?;
        Ok(RawGgufSource {
            file,
            content,
            identity,
            verified_maya: true,
            verification: None,
            parts: Vec::new(),
            tensor_parts: HashMap::new(),
        })
    }

    /// Nonproduction expected pin injection exists only in cfg(test).
    pub(crate) fn cached_source(path: &Path, bytes: u64, sha: [u8; 32]) -> Result<RawGgufSource> {
        anyhow::ensure!(bytes <= SHA_SCRATCH_BYTES as u64, "fixture must stay small");
        let mut file = File::open(path)?;
        let identity = FileIdentity::from_metadata(&file.metadata()?)?;
        let mut pin = maya_pin();
        pin.model_id = "cfg-test-only".into();
        pin.expected_bytes = bytes;
        pin.expected_sha256 = sha;
        let verification =
            crate::verified_file::verify_cached(&file, path, identity, &pin, || {
                verify_artifact(&file, identity, bytes, &sha, read_exact_at)
            })?;
        let content = Arc::new(RawGgufContent::read(&mut file)?);
        check_source_identity(&file, identity)?;
        Ok(RawGgufSource {
            file,
            content,
            identity,
            verified_maya: true,
            verification: Some(verification),
            parts: Vec::new(),
            tensor_parts: HashMap::new(),
        })
    }

    pub(crate) fn maya_metadata(tiny: bool) -> HashMap<String, Value> {
        let mut md = HashMap::from([(
            "general.architecture".into(),
            Value::String("deepseek41".into()),
        )]);
        for (key, value) in [
            ("vocab_size", 129280),
            ("hidden_size", 5120),
            ("moe_intermediate_size", 2304),
            ("num_hidden_layers", 40),
            ("num_attention_heads", 64),
            ("num_key_value_heads", 1),
            ("head_dim", 512),
            ("qk_rope_head_dim", 64),
            ("q_lora_rank", 1280),
            ("o_lora_rank", 1024),
            ("o_groups", 8),
            ("n_routed_experts", 384),
            ("n_shared_experts", 1),
            ("num_experts_per_tok", 6),
            ("max_position_embeddings", 1048576),
            ("sliding_window", 128),
            ("index_n_heads", 32),
            ("index_head_dim", 128),
            ("index_topk", 512),
            ("candidate_source_layer_id", 20),
            ("candidate_topk_blocks", 2048),
            ("candidate_block_size", 8),
            ("hc_mult", 4),
            ("hc_sinkhorn_iters", 20),
            ("rope_theta", 10000),
            ("compress_rope_theta", 160000),
        ] {
            md.insert(format!("deepseek41.{key}"), Value::U32(value));
        }
        for (key, value) in [
            ("rms_norm_eps", 1e-20),
            ("hc_eps", 1e-6),
            ("swiglu_limit", 10.),
            ("routed_scaling_factor", 1.5),
            ("rope_scaling.factor", 16.),
            ("rope_scaling.beta_fast", 32.),
            ("rope_scaling.beta_slow", 1.),
            ("rope_scaling.original_max_position_embeddings", 65536.),
        ] {
            md.insert(format!("deepseek41.{key}"), Value::F32(value));
        }
        for (key, value) in [
            ("scoring_func", "sqrtsoftplus"),
            ("hidden_act", "silu"),
            ("topk_method", "noaux_tc"),
            ("engram.encoding", "e4m3_e8m0_32_row264"),
        ] {
            md.insert(format!("deepseek41.{key}"), Value::String(value.into()));
        }
        md.insert("deepseek41.norm_topk_prob".into(), Value::Bool(true));
        for (key, values) in [
            (
                "compress_ratios",
                [vec![0; 2], vec![2; 18], vec![1; 20]].concat(),
            ),
            ("kv_source_layer_ids", vec![2, 8, 14, 20]),
            ("index_source_layer_ids", vec![2, 8, 14, 20, 24, 28, 32, 36]),
            ("engram.layer_ids", vec![1, 14]),
            ("engram.rows", vec![384006168, 384016682]),
        ] {
            md.insert(
                format!("deepseek41.{key}"),
                Value::Array(values.into_iter().map(Value::U32).collect()),
            );
        }
        if tiny {
            for (key, value) in [
                ("hidden_size", 256),
                ("moe_intermediate_size", 256),
                ("num_attention_heads", 1),
                ("head_dim", 64),
                ("q_lora_rank", 64),
                ("o_lora_rank", 64),
                ("o_groups", 1),
                ("n_routed_experts", 2),
                ("num_experts_per_tok", 1),
                ("vocab_size", 4),
            ] {
                md.insert(format!("deepseek41.{key}"), Value::U32(value));
            }
        }
        md
    }

    /// Sparse on disk: tensor shape describes extent, never an allocation.
    pub(crate) fn sparse(
        md: &HashMap<String, Value>,
        specs: &[(&str, &[usize], u32, u64)],
    ) -> (tempfile::NamedTempFile, u64) {
        fn string(out: &mut Vec<u8>, text: &str) {
            out.extend_from_slice(&(text.len() as u64).to_le_bytes());
            out.extend_from_slice(text.as_bytes());
        }
        fn kind(v: &Value) -> u32 {
            match v {
                Value::U16(_) => 2,
                Value::U32(_) => 4,
                Value::F32(_) => 6,
                Value::Bool(_) => 7,
                Value::String(_) => 8,
                Value::Array(_) => 9,
                Value::U64(_) => 10,
                _ => panic!("fixture type"),
            }
        }
        fn payload(out: &mut Vec<u8>, v: &Value) {
            match v {
                Value::U16(v) => out.extend_from_slice(&v.to_le_bytes()),
                Value::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
                Value::U64(v) => out.extend_from_slice(&v.to_le_bytes()),
                Value::F32(v) => out.extend_from_slice(&v.to_le_bytes()),
                Value::Bool(v) => out.push(u8::from(*v)),
                Value::String(v) => string(out, v),
                Value::Array(values) => {
                    out.extend_from_slice(&values.first().map(kind).unwrap_or(4).to_le_bytes());
                    out.extend_from_slice(&(values.len() as u64).to_le_bytes());
                    for v in values {
                        payload(out, v);
                    }
                }
                _ => panic!("fixture type"),
            }
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&(specs.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(md.len() as u64).to_le_bytes());
        for (key, value) in md {
            string(&mut bytes, key);
            bytes.extend_from_slice(&kind(value).to_le_bytes());
            payload(&mut bytes, value);
        }
        let mut extent = 0;
        for (name, shape, dtype, offset) in specs {
            string(&mut bytes, name);
            bytes.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for dim in shape.iter().rev() {
                bytes.extend_from_slice(&(*dim as u64).to_le_bytes());
            }
            bytes.extend_from_slice(&dtype.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
            let length = DeepSeek4DType::from_u32(*dtype)
                .byte_len(shape.iter().product())
                .unwrap();
            extent = extent.max(offset.saturating_add(length as u64));
        }
        let alignment = md
            .get("general.alignment")
            .and_then(value_as_u64)
            .unwrap_or(32);
        let data = (bytes.len() as u64).div_ceil(alignment) * alignment;
        bytes.resize(data as usize, 0);
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        file.as_file()
            .set_len_sparse(data.checked_add(extent).unwrap_or(data))
            .unwrap();
        (file, data)
    }
}
